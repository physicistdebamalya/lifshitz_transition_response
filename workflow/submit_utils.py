"""
Job submission: build a run directory, write its task.csv, copy in the
binary built once by ensure_binary(), and sbatch the whole sweep as one
SLURM array job (array_script() / submit_array()).

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
import re
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
from task_schema import write_task_csv                        # noqa: E402
import hpc_config                                             # noqa: E402

COMPLETION_MARKER = "--- COMPLETING SIMULATION ---"
BACKUP_DIRNAME    = "incomplete_runs"
DIRECTORIES_TXT   = "directories.txt"
LOCKFILE          = ".submit_lock"
MASTER_LOG        = "logfile.txt"

# ── the array job's three bookkeeping files ───────────────────────────────
# ARRAY_MAP    line N+1 is the run directory for SLURM_ARRAY_TASK_ID N. The
#              array script reads it; nothing else depends on its format.
# ARRAY_JOB_ID the id of the most recently submitted array. preparing_run()
#              needs it because an array element's WorkDir is the SUBMIT
#              directory, not its run directory, so the old per-directory
#              squeue matching cannot see array elements at all.
# ARRAY_LOG_DIR where SLURM writes each element's output BEFORE the script
#              cds into its run directory -- so a failure that happens
#              before the cd (a bad map file, a module load error) still
#              leaves a trace somewhere findable.
ARRAY_MAP         = "array_map.txt"
ARRAY_JOB_ID      = ".array_job_id"
ARRAY_LOG_DIR     = "array_logs"


@lru_cache(maxsize=8)
def _parse_cached(log_path_str, system):
    """parse_master_log is re-read once per submitted job otherwise; the log
    grows to thousands of lines. Cached on a string path so it stays hashable."""
    return parse_master_log(Path(log_path_str), system)


def count_running_jobs(user, partition):
    out = subprocess.run(f"squeue -u {user} -p {partition} --noheader | wc -l",
                         shell=True, capture_output=True, text=True)
    return int(out.stdout.strip() or 0)


def cpus_in_use(user):
    """Cores this user holds RUNNING, cluster-wide -- the unit and the scope
    the QoS ceiling actually enforces.

    `fatqos` caps MaxTRESPU at cpu=80 across small+medium+large together on
    hpc; `iitgn_qos` caps it at cpu=240 across small+medium+large+highmemory
    (+gpu) together on param. Both are the same shape of ceiling -- one pool,
    several partitions -- so a throttle that counts jobs in one partition
    measures the wrong thing twice over on either cluster: wrong unit (a
    64-core job and a 32-core job are not interchangeable) and wrong scope
    (work queued on medium is invisible while submitting to large). Adapted
    from Superconductivity/python/submit_utils_Phase.py.

    PENDING jobs hold no TRES yet -- that is exactly why SLURM lets them queue
    rather than rejecting them -- so they are deliberately not counted.
    Returns None if squeue cannot be reached, which the caller must treat as
    "unknown", never as "zero".
    """
    try:
        out = subprocess.run(["squeue", "-u", user, "-t", "running", "-h", "-o", "%C"],
                             capture_output=True, text=True, check=True, timeout=30)
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return sum(int(n) for n in out.stdout.split())


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

        # Cluster-wide core ceiling this user is subject to. Unknown on hosts
        # with no profiled QoS, where an effectively infinite budget makes
        # wait_for_slot fall back to the job-count limit alone.
        self.cpu_budget = getattr(self.host, "cpu_budget", 10 ** 9)

        self._lockfile = None
        self._warned_missing_logs = set()

    # ----------------------------------------------------------- partition --
    def use_partition(self, name, nodes=None):
        """Switch queue and core layout together, from one name.

        Setting self.partition alone is the mistake this exists to prevent:
        the seven resource attributes are copied off the host profile once,
        in __init__, and nothing recomputes them afterwards -- so a bare
        `s.partition = "large"` sends the job to large's 40-core nodes still
        carrying medium's 2 x 16 split, leaving 8 cores per node idle.

        `nodes` defaults to None, which asks plan_partition() to fill the
        chosen queue's QoS cap with as many WHOLE nodes as fit -- the
        2026-08-23 policy that a single parameter point should claim as much
        of the cluster as it can, since one point IS one job, with no
        sibling job whose slot it could be taking:

            s.use_partition("small")           # auto: 2 nodes, 2 x 20, 1 concurrent (80/80, hpc)
            s.use_partition("large", nodes=1)  # explicit: 1 node, 2 x 20, 2 concurrent -- honoured exactly

        Anything assigned AFTER this call wins, so a hand-tuned layout is
        still one line away:

            s.use_partition("small")
            s.max_concurrent_jobs = 2          # be polite, keep the rest
        """
        if not self.host.partitions:
            raise ValueError(
                f"host {self.host.name!r} has no profiled partition table, so "
                f"there is nothing to select from -- set partition/npn/cpt "
                f"directly, or add a table for it in hpc_config.py. "
                f"('hpc' and 'param' both have one; 'local' does not, since "
                f"it is not a queue -- there is nothing for a table to "
                f"disambiguate.)")

        plan = hpc_config.plan_partition(name, nodes=nodes,
                                         partitions=self.host.partitions)
        for k, v in plan.items():
            setattr(self, k, v)
        self.cpu_budget = self.host.partitions[name].cpu_cap
        # Deliberately silent. This used to print what it had chosen, which
        # read as authoritative and was not: anything assigned afterwards
        # (s.nodes = 1, say) changed the job without changing the line already
        # on screen. resource_summary() is printed once, later, from the
        # values that actually reach the array script (array_script()).
        return self

    def resource_summary(self):
        """The one report of what this sweep will ask SLURM for.

        Built from the live attributes at the moment it is called, so it
        cannot disagree with the #SBATCH block the way a message printed
        early in configuration can.
        """
        need = self.cores_per_job()
        lines = [f"host={self.host.name}  partition={self.partition}  qos={self.qos}",
                 f"  {self.nodes} node(s) x {self.npn} rank(s) x {self.cpt} "
                 f"thread(s) = {need} cores/job, up to "
                 f"{self.max_concurrent_jobs} at once"]

        p = (self.host.partitions or {}).get(self.partition)
        if p is None:
            lines.append(f"  no profiled table for {self.partition!r} on "
                         f"{self.host.name} — npn/cpt are unchecked")
            return "\n".join(lines)

        lines[-1] += (f"  ({min(self.cpu_budget, self.max_concurrent_jobs * need)}"
                      f"/{p.cpu_cap} core budget)")
        lines.append(f"  {p.cpu}, {p.cores_per_node} cores/node, "
                     f"{p.numa_per_node} NUMA domain(s), walltime {p.walltime}")

        per_node = self.npn * self.cpt
        if per_node != p.cores_per_node:
            lines.append(f"  ⚠️  {per_node} of {p.cores_per_node} cores per node "
                         f"in use — this layout does not fill a node")
        if need > p.cpu_cap:
            lines.append(f"  ⚠️  one job needs {need} cores but the cap is "
                         f"{p.cpu_cap} — it will never start")
        return "\n".join(lines)

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
            # Once, not once per point. A missing archive is a fact about the
            # machine, not about this parameter point -- on a cluster the
            # archive simply lives on the workstation -- so repeating it for
            # all 207 points buries the submission log in a message that
            # cannot change between iterations.
            if log_path not in self._warned_missing_logs:
                self._warned_missing_logs.add(log_path)
                print(f"⚠️  no logfile at {log_path} — cannot skip duplicates; "
                      f"every point will be submitted as new", flush=True)
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

    def preparing_run(self, interactive=None):
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

        # Is a previously submitted array still alive?
        #
        # This USED to walk squeue's %Z (WorkDir) and match dir_NNNNN entries
        # one by one. That silently stops working with an array: every element
        # reports the SUBMIT directory as its WorkDir, because the script cds
        # into its run directory itself. The check would find nothing, report
        # "none of these are running", and 'begin' would then delete
        # directories out from under live jobs -- the exact accident it exists
        # to prevent, made invisible.
        #
        # So ask about the array as a whole instead, which is both simpler and
        # strictly more reliable: one job id, recorded at submit time.
        live_elements = self._array_still_live()

        if live_elements:
            raise SystemExit(
                f"array job {self._recorded_array_id()} still has "
                f"{len(live_elements)} element(s) queued or running "
                f"({', '.join(live_elements[:6])}"
                f"{', ...' if len(live_elements) > 6 else ''}).\n"
                f"Neither 'resume' nor 'begin' is safe now — resume would move "
                f"a folder out from under a running element and begin would "
                f"delete it. Wait (squeue -u {self.slurm_user}) or scancel the "
                f"array, then rerun.")

        checked = ("checked against squeue: no element of the last array is "
                   "still live" if live_elements is not None else
                   "could NOT reach squeue — verify nothing is still running "
                   "before choosing 'begin'")
        print(f"\n{len(lines) - start_index} incomplete run(s) ({checked}).", flush=True)

        # Inside a batch job there is no stdin: input() would block until the
        # walltime killed the job, hours later, with nothing in the log to say
        # why. The resume/begin decision belongs on the login node, before
        # dispatch, which is where the generator makes it.
        if interactive is None:
            interactive = sys.stdin.isatty()
        if not interactive:
            raise SystemExit(
                f"{len(lines) - start_index} incomplete run(s) need a "
                f"resume/begin decision, but this process has no terminal to "
                f"ask on. Run the generator from a login-node shell (it makes "
                f"this call before dispatching), or clear "
                f"{DIRECTORIES_TXT} by hand.")

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

    def record_directory(self, run_dir):
        with open(self.cwd / DIRECTORIES_TXT, "a") as f:
            f.write(f"{self.iteration}. {run_dir.name}\n")

    def prepare(self, p, repo=None):
        """Realise one parameter point as a run directory, ready to run.

            dir_NNNNN -> task.csv -> exe_run -> directories.txt

        NO COMPILE. The two model binaries are built ONCE, on the login node,
        by ensure_binary(), before any of this. That is the whole point of the
        task.csv mechanism: `module load gcc/12.4` inside a SLURM job on param
        cannot find features.h (verified 2026-08-22 on cn001; the same test
        succeeds on hpc's cn17), so a per-point compile is not something that
        can be relocated -- it has to stop existing.

        Returns the run directory, or None if the point could not be prepared.
        A bad point is quarantined rather than aborting the sweep: one failure
        in 207 should not cost the other 206.
        """
        repo = Path(repo) if repo else Path(__file__).resolve().parent.parent
        validate_point(p)

        run_dir = self.new_run_dir()
        try:
            write_task_csv(run_dir / "task.csv", p, self.schema)
        except (KeyError, ValueError) as e:
            shutil.rmtree(run_dir, ignore_errors=True)
            self._quarantine_reason(run_dir.name, "task.csv could not be written", str(e))
            return None

        if self.dry_run:
            print(f"  [dry-run] prepared {run_dir.name} "
                  f"(task.csv written, binary not copied)", flush=True)
            self.iteration += 1
            self.record_directory(run_dir)
            return run_dir

        binary = self.binary_path(repo)
        if not binary.exists():
            shutil.rmtree(run_dir, ignore_errors=True)
            raise SystemExit(
                f"model binary missing: {binary}\n"
                f"ensure_binary() must run before the sweep loop -- the "
                f"generator does this on the login node.")

        # Copy, not symlink. ~90-105 KB per directory is nothing, and a
        # symlink into physics/build/ breaks the moment that tree is rebuilt
        # or moved mid-sweep -- while an archived run directory that carries
        # its own binary stays self-contained and honest about what ran.
        # copy2 preserves mtime, so `ls -l exe_run` is real provenance.
        shutil.copy2(binary, run_dir / "exe_run")

        self.iteration += 1
        self.record_directory(run_dir)
        return run_dir

    # ------------------------------------------------------- build & launch --
    def _with_modules(self, cmd):
        """Prefix a command with this host's module block.

        Every compile the generator drives goes through here, so `module purge
        && module load ...` is never something the user has to remember or
        type. Empty module tuple (local) leaves the command untouched.
        """
        if not self.host.modules:
            return cmd
        return f"{self.host.module_block().replace(chr(10), ' && ')} && {cmd}"

    def binary_path(self, repo=None):
        repo = Path(repo) if repo else Path(__file__).resolve().parent.parent
        return repo / "physics" / "build" / self.host.name / self.model

    def ensure_binary(self, repo=None):
        """Build physics/build/<host>/<model> on the LOGIN NODE.

        REFUSES to run inside a SLURM job. Not defensive style -- verified
        2026-08-22 by submitting real probe jobs: on hpc (cn17) compiling
        inside a job works, on param (cn001) it fails with

            fatal error: features.h: No such file or directory

        because gcc/12.4's C++ headers cannot locate glibc's headers on a
        compute node. This guard makes that unreachable rather than merely
        documented, following two_component_BH's compile_drivers().

        Runs `make` UNCONDITIONALLY rather than short-circuiting on
        binary.exists(). make is idempotent and returns in under a second when
        up to date, whereas an .exists() check is precisely how a binary built
        before the last source edit gets copied into 207 run directories.
        """
        if "SLURM_JOB_ID" in os.environ:
            raise SystemExit(
                f"refusing to compile inside a SLURM job "
                f"(SLURM_JOB_ID={os.environ['SLURM_JOB_ID']}). Model binaries "
                f"are built on the login node -- the generator does this "
                f"before it submits anything. On param a compile inside a job "
                f"fails outright (features.h is not reachable from a compute "
                f"node); on hpc it would work but still burns allocation on "
                f"work the login node does for free.")

        repo = Path(repo) if repo else Path(__file__).resolve().parent.parent
        binary = self.binary_path(repo)

        print(f"building {self.model} for host={self.host.name} ...", flush=True)
        cmd = self._with_modules(
            f"make -C {repo/'physics'} HOST={self.host.name} "
            f"build/{self.host.name}/{self.model}")
        r = subprocess.run(cmd, shell=True, executable="/bin/bash",
                           capture_output=True, text=True)
        if r.returncode != 0 or not binary.exists():
            tail = "\n".join((r.stderr or r.stdout).strip().splitlines()[-15:])
            raise SystemExit(f"build failed:\n{tail}")
        print(f"  -> {binary}", flush=True)
        return binary

    def max_array_size(self):
        """SLURM's MaxArraySize, read live. 1001 on both clusters as of
        2026-08-22, but a sweep is refused against the live value rather than
        a remembered one."""
        try:
            out = subprocess.run(["scontrol", "show", "config"],
                                 capture_output=True, text=True, check=True,
                                 timeout=20).stdout
            return int(re.search(r"MaxArraySize\s*=\s*(\d+)", out).group(1))
        except Exception:
            return None

    def array_script(self, count, walltime=None, exe="./exe_run",
                     task_file="task.csv"):
        """The one job script an entire sweep runs under.

        Every element is the SAME binary on the SAME resources, differing only
        in which directory it enters -- which is exactly what an array is for,
        and exactly what became possible once parameters stopped being
        compiled in.

        Two details that are not cosmetic:

        1. THE LOG REDIRECT. #SBATCH --output is resolved relative to the
           submit directory, and an array is submitted once from the parent,
           so %j would drop every element's log in the parent rather than in
           its run directory. Redirecting after the cd puts them back where
           is_job_complete() looks for them (mpi_logs_*.out) and where the
           archive expects them.
        2. `set -e` BEFORE the cd. Without it a failed cd runs the solver in
           the submit directory, overwriting the parent's files and reporting
           success -- the same shape as the venv bug in
           hpc_job_failure_modes.md, where a failed `source` did not stop the
           script and the run silently used the wrong Python.
        """
        p = (self.host.partitions or {}).get(self.partition)
        walltime = walltime or (p.walltime if p else "1-00:00:00")

        lines = ["#!/bin/bash",
                 f"#SBATCH --job-name={self.job_name}",
                 f"#SBATCH --partition={self.partition}",
                 f"#SBATCH --qos={self.qos}",
                 f"#SBATCH --nodes={self.nodes}",
                 f"#SBATCH --ntasks-per-node={self.npn}",
                 f"#SBATCH --cpus-per-task={self.cpt}",
                 "#SBATCH --hint=nomultithread",
                 f"#SBATCH --time={walltime}",
                 # %N throttling: SLURM itself runs at most this many elements
                 # at once. This replaces the old Python wait_for_slot() poll
                 # loop, and with it the only reason the submitting process had
                 # to stay alive at all.
                 f"#SBATCH --array=0-{count - 1}%{self.max_concurrent_jobs}",
                 f"#SBATCH --output={ARRAY_LOG_DIR}/%A_%a.out",
                 f"#SBATCH --error={ARRAY_LOG_DIR}/%A_%a.err",
                 *self.host.extra_sbatch,
                 "",
                 "set -e",
                 ""]

        if self.host.modules:
            lines += [self.host.module_block(), ""]

        lines += [f'RUN_DIR=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" '
                  f'"$SLURM_SUBMIT_DIR/{ARRAY_MAP}")',
                  'if [ -z "$RUN_DIR" ]; then',
                  '    echo "no run directory for array index $SLURM_ARRAY_TASK_ID" >&2',
                  "    exit 1",
                  "fi",
                  'cd "$SLURM_SUBMIT_DIR/$RUN_DIR"',
                  "",
                  "# From here on, everything belongs to this run directory.",
                  'exec > "mpi_logs_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" \\',
                  '     2> "mpi_logs_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"',
                  ""]

        if self.host.env_exports:
            lines += [*self.host.env_exports, ""]

        lines += ["export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK",
                  "export OMP_PROC_BIND=spread",
                  "export OMP_PLACES=cores",
                  "",
                  "# Provenance: which binary and which input actually ran.",
                  f"ls -l --time-style=full-iso {exe} {task_file}",
                  "",
                  f"mpirun {self.host.mpirun_flags} \\",
                  "    --map-by ppr:${SLURM_NTASKS_PER_NODE}:node:PE=${SLURM_CPUS_PER_TASK} \\",
                  f"    --bind-to core {exe} {task_file}",
                  "",
                  "date",
                  ""]
        return "\n".join(lines)

    def submit_array(self, run_dirs, walltime=None):
        """Submit a whole sweep as ONE array job, and return.

        Replaces both the login-node submission loop and the compute-node
        driver job. Neither was right:

          - a loop on the login node contradicts stated policy. IITGN HPC's
            /etc/motd: "Running any kind of simulation & jobs are NOT allowed
            in the login node; JOBS must run through a scheduler only." No
            enforcement daemon was found on either cluster in 2026-08, but the
            policy is written down and this code outlives that snapshot.
          - a driver job holds a core for the sweep's whole lifetime, and this
            solver is sized to fill its allocation exactly (npn x cpt = one
            node), so a stray core is not free.

        An array costs neither: nothing long-lived runs anywhere, SLURM does
        the throttling, and the whole sweep is one squeue line rather than N.
        """
        if not run_dirs:
            print("nothing to submit.", flush=True)
            return None

        limit = self.max_array_size()
        if limit is not None and len(run_dirs) > limit:
            raise SystemExit(
                f"{len(run_dirs)} points exceeds this cluster's MaxArraySize "
                f"of {limit}. Split the sweep, or raise it with the admins.")

        (self.cwd / ARRAY_LOG_DIR).mkdir(exist_ok=True)
        # The index -> directory map the array script reads. Line N+1 is the
        # directory for SLURM_ARRAY_TASK_ID N.
        (self.cwd / ARRAY_MAP).write_text(
            "\n".join(d.name for d in run_dirs) + "\n")

        script = self.cwd / f"sweep_{self.model}.sh"
        script.write_text(self.array_script(len(run_dirs), walltime))

        if self.dry_run:
            print(f"  [dry-run] would submit {len(run_dirs)} point(s) as an "
                  f"array, {self.max_concurrent_jobs} at a time", flush=True)
            return None

        r = subprocess.run(["sbatch", script.name], cwd=self.cwd,
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"could not submit the array job:\n{r.stderr.strip()}")

        print(r.stdout.strip(), flush=True)
        job_id = re.search(r"(\d+)", r.stdout)
        job_id = job_id.group(1) if job_id else ""
        # preparing_run() needs this on a later invocation: an array element
        # reports the SUBMIT directory as its WorkDir, not its run directory,
        # so directories can no longer be matched to live jobs one by one.
        (self.cwd / ARRAY_JOB_ID).write_text(job_id + "\n")

        print(f"{len(run_dirs)} point(s) queued as array {job_id}, "
              f"{self.max_concurrent_jobs} running at a time.\n"
              f"  squeue -u {self.slurm_user}\n"
              f"  logs: {self.cwd}/{ARRAY_LOG_DIR}/ (pre-cd) and each "
              f"dir_NNNNN/mpi_logs_*.out (the run itself)", flush=True)
        return job_id

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

    def _recorded_array_id(self):
        """The last array job id submitted from this directory, or None."""
        path = self.cwd / ARRAY_JOB_ID
        if not path.exists():
            return None
        return path.read_text().strip() or None

    def _array_still_live(self):
        """Element ids of the last submitted array that are still queued or
        running.

        Returns [] when the array is finished or there was never one, and
        None when squeue could not be reached -- the caller must treat None
        as "unknown", never as "nothing is running".
        """
        job_id = self._recorded_array_id()
        if job_id is None:
            return []
        try:
            out = subprocess.run(
                ["squeue", "-j", job_id, "-h", "-o", "%i %T"],
                capture_output=True, text=True, timeout=30)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None
        if out.returncode != 0:
            # squeue exits nonzero for a job id it no longer knows about,
            # which is the normal state once an array has drained.
            return []
        return [line.split()[0] for line in out.stdout.splitlines() if line.strip()]

    def _quarantine_reason(self, name, why, detail=""):
        """Report a point that never got as far as having a directory to move."""
        print(f"  ⚠️  {why} for {name}", flush=True)
        if detail:
            tail = "\n".join(detail.strip().splitlines()[-8:])
            print(f"      {tail}", flush=True)

    def cores_per_job(self):
        return self.nodes * self.npn * self.cpt

    # NOTE: wait_for_slot() is gone, and so is the process that had to stay
    # alive to run it. Throttling is now `#SBATCH --array=0-N%K` in
    # array_script(), enforced by SLURM.
    #
    # K is the same number the poll loop used as its ceiling:
    # max_concurrent_jobs, which plan_partition() derives as
    # cpu_budget // cores_per_job. By default (2026-08-23 resource-allocation
    # policy) a single point already claims the whole cap -- param medium
    # 240/240 = 1 at 5 nodes, hpc large 80/80 = 1 at 2 nodes -- so K is
    # usually 1 and the array runs its points strictly one after another.
    # The QoS core cap remains underneath as a backstop -- if other jobs of
    # yours are already running, array elements simply pend on
    # QOSMaxCpuPerUserLimit, which is SLURM enforcing the same budget the
    # loop used to poll for, without anything of ours having to be running.


# ---------------------------------------------------------------- validation --
# The two sides do not test "is this channel off" the same way, and this keeps
# the gap between them out of reach.
#
#   C++   physics/src/necessary_functions.cpp: `if (A0 == 0.0) return;`
#         physics/models/*.cpp:                `if (A0X != 0.0 && ...)`
#         -- exact, and deliberately so. It matches AM3.cpp, and A0 reaches
#         C++ through task_schema.py's repr()-round-trip guarantee (a task.csv
#         cell parses via std::stod to the SAME double Python held when it
#         wrote the cell), not a computed quantity, so there is no rounding
#         error for a tolerance to absorb.
#
#   Python analysis/snippets.py: format_number() flushes |x| < 1e-12 to zero,
#         and emit_pulse() then treats the channel as inactive.
#
# For 0 < |A0| < 1e-12 they disagree: C++ would write A0X plus every pulse
# field for that axis, while the query built from the same dict would carry
# `A0X = 0` and no pulse fields at all. The subset match then fails and the
# run becomes permanently unfindable -- already_exists() would resubmit it
# forever. No sweep has ever gone near that window (amplitudes here are 0 or
# ~0.01), so rather than change either emitter and risk the 99-run archive,
# refuse the input that would expose the difference.

_A0_MIN = 1e-12          # analysis/snippets.py:_THRESHOLD_ZERO


def validate_point(p):
    """Reject a parameter point that the C++ and Python emitters would
    describe differently. Raises ValueError; never returns a verdict to
    ignore."""
    for axis in ("X", "Y"):
        key = f"A0{axis}"
        if key not in p:
            continue
        a0 = abs(float(p[key]))
        if 0.0 < a0 < _A0_MIN:
            raise ValueError(
                f"{key} = {p[key]!r} sits in the one window where C++ and "
                f"analysis disagree about whether a pulse channel is active: "
                f"C++ tests A0 != 0.0 exactly, while analysis/snippets.py "
                f"flushes |A0| < {_A0_MIN:g} to zero. The run would be "
                f"written with every pulse field for axis {axis} and searched "
                f"for with none of them, so it could never be found again. "
                f"Use exactly 0.0 to switch the channel off, or an amplitude "
                f"at or above {_A0_MIN:g}.")


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
