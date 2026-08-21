"""
Job submission: build a run directory, compile into it, write job.sh, sbatch.

Consolidates the four near-identical copies that lived in notebook cells, and
adopts the two mechanisms Superconductivity added after the notebook version:

  already_exists()  refuse to resubmit a parameter point the archive already
                    holds, using the SAME matcher the analysis side uses
                    (AM3_submit_utils.py:298). One definition of "the same
                    run" for both finding data and deciding what to compute.

  preparing_run()   before resuming or restarting, cross-check candidate run
                    directories against squeue's %Z (working directory). A run
                    that has not written its completion marker is either dead
                    or simply still going; treating the second as the first is
                    what would move a live job's folder out from under it.
"""

import atexit
import itertools
import os
import random
import shutil
import subprocess
import sys
import textwrap
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))

from data_utils import clean_path, parse_master_log          # noqa: E402
from models import MODELS                                     # noqa: E402
from snippets import (CreateSnippet, DEFAULT_IGNORE_PATTERNS,  # noqa: E402
                      find_key_by_substring)
import hpc_config                                             # noqa: E402

COMPLETION_MARKER = "--- COMPLETING SIMULATION ---"
BACKUP_DIRNAME    = "incomplete_runs"
DIRECTORIES_TXT   = "directories.txt"
LOCKFILE          = ".submit_lock"
MASTER_LOG        = "logfile.txt"


@lru_cache(maxsize=8)
def _parse_cached(log_path_str, system):
    """parse_master_log is re-read once per submitted job otherwise; the log
    grows to thousands of lines. Cached on a string path so it stays hashable."""
    return parse_master_log(Path(log_path_str), system)


def count_running_jobs(user, partition):
    out = subprocess.run(f"squeue -u {user} -p {partition} --noheader | wc -l",
                         shell=True, capture_output=True, text=True)
    return int(out.stdout.strip() or 0)


def get_active_job_info(user):
    """{workdir: (job_id, state)} for this user's live jobs, or None if squeue
    could not be reached -- None and {} mean very different things here."""
    try:
        out = subprocess.run(["squeue", "-u", user, "--noheader", "-o", "%i|%T|%Z"],
                             capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    info = {}
    for line in out.stdout.splitlines():
        parts = line.strip().split("|")
        if len(parts) == 3:
            job_id, state, workdir = parts
            info[workdir] = (job_id, state)
    return info


class Submitter:
    def __init__(self, model, host=None, data_path=None, dry_run=False):
        if model not in MODELS:
            raise ValueError(f"unknown model {model!r}; have {sorted(MODELS)}")
        self.model = model
        self.schema = MODELS[model]
        self.host = hpc_config.get(host)
        self.cwd = Path.cwd()
        self.dry_run = dry_run
        self.iteration = 0
        self.job_name = "run"

        self.data_path = Path(clean_path(str(data_path))) if data_path else self.cwd

        for k in ("partition", "qos", "nodes", "npn", "cpt",
                  "max_concurrent_jobs", "slurm_user"):
            setattr(self, k, getattr(self.host, k))

        self._lockfile = None

    # ----------------------------------------------------------- partition --
    def use_partition(self, name, nodes=1):
        """Switch queue and core layout together, from one name.

        Setting self.partition alone is the mistake this exists to prevent:
        the seven resource attributes are copied off the host profile once,
        in __init__, and nothing recomputes them afterwards -- so a bare
        `s.partition = "large"` sends the job to large's 40-core nodes still
        carrying medium's 2 x 16 split, leaving 8 cores per node idle.

            s.use_partition("large")           # 1 node, 2 x 20, 2 concurrent
            s.use_partition("large", nodes=2)  # 2 nodes, 1 concurrent

        Anything assigned AFTER this call wins, so a hand-tuned layout is
        still one line away:

            s.use_partition("large")
            s.max_concurrent_jobs = 1          # be polite, keep the rest
        """
        if not self.host.partitions:
            raise ValueError(
                f"host {self.host.name!r} has no profiled partition table, so "
                f"there is nothing to select from -- set partition/npn/cpt "
                f"directly, or add a table for it in hpc_config.py. "
                f"(Only 'hpc' has one: its queues differ in cores per node "
                f"and in NUMA layout, which is what makes the lookup worth "
                f"having.)")

        plan = hpc_config.plan_partition(name, nodes=nodes,
                                         partitions=self.host.partitions)
        for k, v in plan.items():
            setattr(self, k, v)

        p = self.host.partitions[name]
        used = plan["nodes"] * plan["npn"] * plan["cpt"]
        print(f"partition={p.name}  {plan['nodes']} node(s) x {plan['npn']} "
              f"rank(s) x {plan['cpt']} thread(s) = {used} cores/job, "
              f"{plan['max_concurrent_jobs']} concurrent "
              f"({used * plan['max_concurrent_jobs']}/{p.cpu_cap} core cap)  "
              f"[{p.cpu}, walltime {p.walltime}]", flush=True)

        # The cap can force a layout that no longer fills a node -- ask for 3
        # medium nodes and 96 > 80, so cpt drops to 13 and 6 cores per node go
        # unused. That is a legal job and it will run, but it is not what the
        # "one rank per NUMA domain, fill the node" rule promises, so say so
        # rather than let the printed numbers look tuned when they are not.
        per_node = plan["npn"] * plan["cpt"]
        if per_node != p.cores_per_node:
            remedy = ("Fewer nodes would use each one fully."
                      if p.cores_per_node <= p.cpu_cap else
                      f"Inherent to this queue -- its {p.cpu_cap}-core QoS cap "
                      f"is smaller than one {p.cores_per_node}-core node, so "
                      f"no node count fills it. Use it for smoke tests only.")
            print(f"  ⚠️  {per_node} of {p.cores_per_node} cores per node in "
                  f"use: {plan['nodes']} x {p.cores_per_node} exceeds the "
                  f"{p.cpu_cap}-core cap, so threads per rank were reduced to "
                  f"fit. {remedy}", flush=True)
        return self

    # ---------------------------------------------------------------- lock --
    def acquire_lock(self):
        lock = self.cwd / LOCKFILE
        if lock.exists():
            raise SystemExit(
                f"{LOCKFILE} exists in {self.cwd} — another submitter may be "
                f"running. Remove it if you are sure it is stale.")
        lock.write_text(str(os.getpid()))
        self._lockfile = lock
        atexit.register(self.release_lock)

    def release_lock(self):
        try:
            if self._lockfile and self._lockfile.exists():
                self._lockfile.unlink()
        except OSError:
            pass

    # ------------------------------------------------------------- dedup ---
    def already_exists(self, p, data_path=None, master_log=MASTER_LOG,
                       ignore_patterns=None):
        """True if the archive already holds this parameter point.

        Deliberately the same CreateSnippet + find_key_by_substring the
        notebooks use to locate data, so 'already computed' and 'findable'
        can never disagree.
        """
        log_path = Path(data_path or self.data_path) / master_log
        if not log_path.exists():
            print(f"⚠️  no logfile at {log_path} — cannot skip duplicates", flush=True)
            return False
        entries = _parse_cached(str(log_path), self.schema.system)
        snpt = CreateSnippet(p, self.schema,
                             ignore_patterns or DEFAULT_IGNORE_PATTERNS)
        return find_key_by_substring(entries, snpt, verbose=False) is not None

    # -------------------------------------------------------------- resume --
    def is_job_complete(self, run_dir):
        """Both conditions, as AM3_submit_utils.is_job_complete requires: the
        marker can be present in a log from an earlier attempt in a reused
        directory, and data.csv can exist while the run was still writing."""
        run_dir = Path(run_dir)
        if not (run_dir / "data.csv").exists():
            return False
        for log in run_dir.glob("mpi_logs_*.out"):
            try:
                with open(log, errors="ignore") as f:
                    for line in f:
                        if COMPLETION_MARKER in line:
                            return True
            except OSError:
                continue
        return False

    def preparing_run(self):
        """Return how many leading directories in directories.txt are finished,
        so a resumed sweep picks up where it stopped.

        Refuses to act while any candidate directory belongs to a live job:
        'resume' would move the folder out from under it and 'begin' would
        delete it. See AM3_submit_utils.preparing_run.
        """
        dir_txt = self.cwd / DIRECTORIES_TXT
        if not dir_txt.exists():
            return 0

        lines = [l.strip() for l in dir_txt.read_text().splitlines() if l.strip()]
        start_index = 0
        for line in lines:
            parts = line.split(". ", 1)
            d_name = parts[1] if len(parts) == 2 else line.split()[-1]
            d = self.cwd / d_name
            if d.exists() and self.is_job_complete(d):
                start_index += 1
            else:
                break

        if start_index == len(lines):
            self.iteration = start_index
            return start_index

        active = get_active_job_info(self.slurm_user)
        active_here = {}
        if active is not None:
            for workdir, (job_id, state) in active.items():
                wp = Path(workdir)
                if wp.name.startswith("dir_") and wp.parent.resolve() == self.cwd.resolve():
                    active_here[wp.name] = (job_id, state)

        if active_here:
            details = "\n".join(f"  - {d}: job {j} is currently {s}"
                                for d, (j, s) in active_here.items())
            raise SystemExit(
                f"{len(active_here)} run director"
                f"{'y is' if len(active_here) == 1 else 'ies are'} still active:\n"
                f"{details}\nNeither 'resume' nor 'begin' is safe now — resume would "
                f"move a folder out from under a running job and begin would delete "
                f"it. Wait (squeue -u {self.slurm_user}) and rerun.")

        checked = ("checked against squeue: none of these are running"
                   if active is not None else
                   "could NOT reach squeue — verify none of these are still "
                   "running before choosing 'begin'")
        print(f"\n{len(lines) - start_index} incomplete run(s) ({checked}).", flush=True)
        choice = input("  [resume] quarantine them and continue, "
                       "[begin] delete and start over, [abort]: ").strip().lower()

        if choice == "resume":
            backup = self.cwd / BACKUP_DIRNAME
            backup.mkdir(exist_ok=True)
            for line in lines[start_index:]:
                parts = line.split(". ", 1)
                d_name = parts[1] if len(parts) == 2 else line.split()[-1]
                d = self.cwd / d_name
                if d.exists():
                    shutil.move(str(d), str(backup / d_name))
            dir_txt.write_text("\n".join(lines[:start_index]) + "\n")
        elif choice == "begin":
            for line in lines:
                parts = line.split(". ", 1)
                d_name = parts[1] if len(parts) == 2 else line.split()[-1]
                d = self.cwd / d_name
                if d.exists():
                    shutil.rmtree(d)
            dir_txt.unlink()
            start_index = 0
        else:
            raise SystemExit("aborted.")

        self.iteration = start_index
        return start_index

    # -------------------------------------------------------------- submit --
    def new_run_dir(self):
        existing = {d.name for d in self.cwd.iterdir() if d.is_dir()}
        for _ in range(1000):
            candidate = f"dir_{random.randint(10000, 99999)}"
            if candidate not in existing:
                d = self.cwd / candidate
                d.mkdir()
                return d
        raise RuntimeError("no free dir_NNNNN name after 1000 tries")

    def job_script(self, exe="./exe_run"):
        """Build job.sh.

        Assembled by concatenation rather than a dedent'd f-string: the
        interpolated module and env blocks are themselves multi-line, which
        defeats dedent's common-prefix detection and silently leaves stray
        indentation in front of #SBATCH directives, where slurm ignores them.
        """
        lines = ["#!/bin/bash",
                 f"#SBATCH --job-name={self.job_name}_{self.iteration}",
                 f"#SBATCH --partition={self.partition}",
                 f"#SBATCH --qos={self.qos}",
                 f"#SBATCH --nodes={self.nodes}",
                 f"#SBATCH --ntasks-per-node={self.npn}",
                 f"#SBATCH --cpus-per-task={self.cpt}",
                 "#SBATCH --hint=nomultithread",
                 "#SBATCH --output=mpi_logs_%j.out",
                 "#SBATCH --error=mpi_logs_%j.err",
                 *self.host.extra_sbatch,
                 ""]

        if self.host.modules:
            lines += [self.host.module_block(), ""]

        lines += ["cd $SLURM_SUBMIT_DIR", ""]

        if self.host.env_exports:
            lines += [*self.host.env_exports, ""]

        lines += ["export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK",
                  "export OMP_PROC_BIND=spread",
                  "export OMP_PLACES=cores",
                  "",
                  f"mpirun {self.host.mpirun_flags} \\",
                  "    --map-by ppr:${SLURM_NTASKS_PER_NODE}:node:PE=${SLURM_CPUS_PER_TASK} \\",
                  f"    --bind-to core {exe}",
                  "",
                  "date",
                  ""]
        return "\n".join(lines)

    def record_directory(self, run_dir):
        with open(self.cwd / DIRECTORIES_TXT, "a") as f:
            f.write(f"{self.iteration}. {run_dir.name}\n")

    def submit(self, p, model_source=None, repo=None):
        """Realise one parameter point: rewrite, compile, submit.

        Follows AM3_submit_utils.submit step for step:
          dir_NNNNN -> change_lines into it -> Makefile -> compile
                    -> job.sh -> sbatch -> directories.txt

        A compile failure quarantines the directory and returns None rather
        than aborting: one bad point in a 60-point sweep should not cost the
        other 59.
        """
        repo = Path(repo) if repo else Path(__file__).resolve().parent.parent
        model_source = Path(model_source) if model_source else (
            repo / "physics" / "models" / f"{self.model}.cpp")
        if not model_source.exists():
            raise FileNotFoundError(f"model source not found: {model_source}")

        # Only a real submission needs the engine object present. A dry run on
        # a laptop targeting `param` legitimately cannot have one.
        engine = repo / "physics" / "build" / self.host.name / "necessary_functions.o"
        if not self.dry_run and not engine.exists():
            raise FileNotFoundError(
                f"engine object missing: {engine}\n"
                f"run `make -C {repo/'physics'} HOST={self.host.name}` first")

        run_dir = self.new_run_dir()
        target = run_dir / f"{self.model}.cpp"

        try:
            change_lines(model_source, target, build_replacements(p, self.schema))
        except ValueError:
            shutil.rmtree(run_dir, ignore_errors=True)
            raise

        shutil.copy2(repo / "workflow" / "Makefile.job", run_dir / "Makefile")

        if self.dry_run:
            print(f"  [dry-run] prepared {run_dir.name} "
                  f"({self.model}.cpp rewritten, not compiled)", flush=True)
            return run_dir

        cmd = f"make MODEL={self.model} HOST={self.host.name} REPO={repo}"
        if self.host.modules:
            cmd = f"{self.host.module_block().replace(chr(10), ' && ')} && {cmd}"

        try:
            subprocess.run(cmd, shell=True, check=True, executable="/bin/bash",
                           cwd=run_dir, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            self._quarantine(run_dir, f"compile failed", e.stderr)
            return None

        self.iteration += 1
        (run_dir / "job.sh").write_text(self.job_script())

        try:
            subprocess.run(["sbatch", "job.sh"], cwd=run_dir, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            self._quarantine(run_dir, f"sbatch failed: {e}", "")
            self.iteration -= 1
            return None

        self.record_directory(run_dir)
        return run_dir

    def _quarantine(self, run_dir, why, detail=""):
        """Move a failed run dir aside so it can be inspected, without letting
        it look like a pending job to preparing_run()."""
        backup = self.cwd / BACKUP_DIRNAME
        backup.mkdir(exist_ok=True)
        target = backup / run_dir.name
        if target.exists():
            target = backup / f"{run_dir.name}_{int(time.time())}"
        try:
            shutil.move(str(run_dir), str(target))
        except OSError:
            shutil.rmtree(run_dir, ignore_errors=True)
            target = None
        print(f"  ⚠️  {why} for {run_dir.name}"
              + (f"; moved to {target}" if target else "; removed"), flush=True)
        if detail:
            tail = "\n".join(detail.strip().splitlines()[-8:])
            print(f"      {tail}", flush=True)

    def wait_for_slot(self, poll=300):
        while count_running_jobs(self.slurm_user, self.partition) >= self.max_concurrent_jobs:
            print("⏳ waiting for a job slot...", flush=True)
            time.sleep(poll)


# ------------------------------------------------------------ source rewrite --
# Interim mechanism, retired by the task.csv pass. Parameters are still
# compile-time constants in the model sources, so a parameter point is
# realised by rewriting the source and recompiling. Kept because it is what
# restores a working pipeline today; every call site goes through
# build_replacements() so the switch to task.csv touches one function.

def change_lines(src_path, dst_path, replacements):
    """Rewrite whole lines identified by their leading text.

    Raises on an unmatched prefix rather than silently leaving the old value
    in place -- a silent miss here means a job runs with the wrong parameter
    and lands in the archive labelled as something it is not.
    """
    lines = Path(src_path).read_text().splitlines(keepends=True)
    for prefix, new_line in replacements:
        for i, line in enumerate(lines):
            if line.lstrip().startswith(prefix):
                lines[i] = new_line.rstrip("\n") + "\n"
                break
        else:
            raise ValueError(f"no line starting with {prefix!r} in {src_path}")
    Path(dst_path).write_text("".join(lines))


_CPP_ENUM = {"sin^2": "Sin2", "Gaussian": "Gaussian", "sin": "Sin",
             "cos": "Cos", "Linear": "Linear", "NA": "NA", "Envelope": "Envelope"}


def build_replacements(p, schema):
    """Derive the (prefix, new_line) list from the schema, so adding a
    parameter means editing models.py rather than 30 call sites."""
    def num(x):
        x = float(x)
        return int(x) if x.is_integer() else x

    rep = []
    for k in schema.scalars:
        ctype = "int" if k in schema.ints else "double"
        val = int(p[k]) if k in schema.ints else num(p[k])
        rep.append((f"constexpr {ctype} {k} =", f"constexpr {ctype} {k} = {val};"))

    rep.append(("constexpr double ETP =", f"constexpr double ETP = {num(p['ETP'])};"))
    rep.append(("constexpr int TIME_DIVISION =",
                f"constexpr int TIME_DIVISION = {int(p['TIME_DIVISION'])};"))

    for axis in ("X", "Y"):
        for base, ctype in (("A0", "double"), ("nu", "double"), ("sigma", "double"),
                            ("ntt_sigma", "double"), ("delay", "double"),
                            ("duration", "double"), ("number_of_cycle", "int")):
            key = f"{base}{axis}"
            val = int(p[key]) if ctype == "int" else num(p[key])
            rep.append((f"constexpr {ctype} {key} =",
                        f"constexpr {ctype} {key} = {val};"))
        for base in ("pulse_type", "pulse_wrapper"):
            key = f"{base}{axis}"
            rep.append((f"const PulseKind {key} = PulseKind::",
                        f"const PulseKind {key} = PulseKind::{_CPP_ENUM[p[key]]};"))
    return rep


# ------------------------------------------------------------- sweep expansion --
# Ported from Superconductivity's AM3_old.ipynb, section "1.1 Submit jobs
# altermagnetic system AM3" (system_params/pulse_params + data_utils._expand_dict
# / data_utils._normalize_value). Stripped of what this project has no use for
# -- WILD placeholders auto-filled from a phase diagram, tagged variants,
# scallerDeltas, the A0/lpa <-> A0X/A0Y merge -- none of which apply here:
# new has no phase-diagram workflow. What is kept is the one rule that
# generalises, stated in the notebook as:
#
#     scalar OR [scalar]   -> fixed
#     list length > 1      -> swept
#
# GRID_FIXED_KEYS is new: the notebook's own generic rule would happily accept
# a multi-value list for any key, including its b1/b2/k_origin-equivalent
# (there, Q). This project's k-grid and Brillouin-zone geometry -- N1, N2, b1,
# b2, k_origin -- must stay fixed for a given sweep, so expand_sweep refuses a
# swept list on any of them rather than silently expanding one.

GRID_FIXED_KEYS = frozenset({"N1", "N2", "b1", "b2", "k_origin"})

# The two scalar members (N1, N2) and the three vector members (b1, b2,
# k_origin) need different guards: a vector's own *fixed* representation is
# already a length-2 list ([4.0, 0.0]), which the plain "list longer than
# one element -> swept" rule would otherwise shred into two bogus scalar
# options. VECTOR_KEYS marks the subset that needs the vector-shaped check
# instead of the scalar one.
VECTOR_KEYS = frozenset({"b1", "b2", "k_origin"})


def _grid_fixed_error(key, v):
    return ValueError(
        f"{key} defines the k-grid/Brillouin-zone geometry and must stay "
        f"fixed within a sweep -- got {v!r}. A grid-density or grid-geometry "
        f"scan is a convergence study; run it as its own explicit sweep, not "
        f"folded into this one.")


def _sweep_values(key, v):
    """One BASE entry -> the list of values it expands to."""
    if isinstance(v, np.ndarray):
        v = v.tolist()

    if key in VECTOR_KEYS:
        # Only a single flat 2-vector is accepted -- e.g. b1 = [4.0, 0.0].
        # A list of vectors ([[4,0],[5,0]]) would mean "sweep this vector",
        # which is exactly what GRID_FIXED_KEYS forbids for these three.
        is_one_vector = (isinstance(v, (list, tuple)) and len(v) == 2
                         and all(isinstance(c, (int, float)) for c in v))
        if not is_one_vector:
            raise _grid_fixed_error(key, v)
        return [list(v)]

    if isinstance(v, (list, tuple)):
        if not v:
            raise ValueError(f"{key}: empty list has nothing to sweep or fix")
        if len(v) > 1:
            if key in GRID_FIXED_KEYS:          # N1, N2
                raise _grid_fixed_error(key, v)
            return list(v)
        return [v[0]]                  # one-element list -> fixed, like a scalar
    return [v]                         # bare scalar -> fixed


def expand_sweep(base):
    """dict[str, scalar | list] -> list[dict[str, scalar]], one dict per
    Cartesian combination of every list-valued (length > 1) entry.

        expand_sweep({"A": 1, "B": [2, 3]})
        -> [{"A": 1, "B": 2}, {"A": 1, "B": 3}]

    A scalar, or a list of exactly one element, is always fixed -- so mixing
    a swept key with ten fixed ones costs nothing but the one line that
    actually varies. Keys that are coupled to each other (this project's
    ef <-> nu pairing) are not expressible as an independent Cartesian
    factor and are handled by the generator's own loop before this is
    called; expand_sweep only ever sees the keys that vary independently.
    """
    keys = list(base.keys())
    per_key = [_sweep_values(k, base[k]) for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*per_key)]
