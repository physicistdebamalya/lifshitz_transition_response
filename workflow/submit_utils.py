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
import os
import random
import shutil
import subprocess
import sys
import textwrap
import time
from functools import lru_cache
from pathlib import Path

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
        for log in Path(run_dir).glob("mpi_logs_*.out"):
            if COMPLETION_MARKER in log.read_text(errors="ignore"):
                return True
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

    def submit(self, p, binary):
        """Create a run dir, place the binary and job.sh in it, and sbatch."""
        run_dir = self.new_run_dir()
        self.iteration += 1

        shutil.copy2(binary, run_dir / "exe_run")
        (run_dir / "job.sh").write_text(self.job_script())
        self.record_directory(run_dir)

        if self.dry_run:
            print(f"  [dry-run] prepared {run_dir.name}, not submitting", flush=True)
            return run_dir

        try:
            subprocess.run(["sbatch", "job.sh"], cwd=run_dir, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"  ⚠️  sbatch failed for {run_dir.name}: {e}", flush=True)
        return run_dir

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
