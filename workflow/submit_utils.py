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
import signal
import subprocess
import sys
import textwrap
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))

from data_utils import parse_master_log                        # noqa: E402
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

# Every per-point line the sweep loop produces. The terminal gets the summary;
# this gets the detail, because 207 "skip"/"prepared" lines scrolled the two
# lines that mattered -- the resource plan and the sbatch result -- out of the
# window they were needed in. Append-only and block-delimited, the same shape
# as logfile.txt (philosophy §8), so successive submissions into one run root
# stay readable and greppable rather than overwriting each other.
SUBMIT_LOG        = "submission_log.txt"

# ══════════════════════════════════════════════════════════════════════════
# Where things are, derived rather than tabulated
# ══════════════════════════════════════════════════════════════════════════
#
# One rule for the workstation and both clusters: the repository knows where
# it is, so nothing here needs a per-host table of absolute paths to be kept
# in step by hand.
#
#   <repo>/workflow/submit_utils.py   this file  -> repo_root()
#   <repo>/<model>/                   run root   -> run_root_for(model)
#   <repo>/archive/<model>_model/     registry   -> default_archive(model)  (cluster)
#
# run_root_for() replaces Path.cwd(). The cwd rule was invisible in exactly
# the way that costs something: running the generator from workflow/ instead
# of from the model's own directory put 96 dir_NNNNN into the source tree,
# under a directories.txt that no later `down <model>` could separate. The
# generator is already cwd-independent everywhere else -- its imports and its
# build step both resolve from __file__ -- so this makes the run directories
# agree with the rest of it.
#
# <repo>/<model> is also exactly the directory hpc_transfers.py creates on a
# push (remote_model_root()), so a pushed tree is ready for the generator
# with no cd and no --archive to remember.


def repo_root():
    """This project's root, from this file's own location.

    Not Path.cwd() and not a per-host constant: the generator must land its
    run directories in the same place whether it is invoked from the model
    directory, from workflow/, or from $HOME on a cluster.
    """
    return Path(__file__).resolve().parent.parent


def run_root_for(model):
    """<repo>/<model> -- where this model's dir_NNNNN and directories.txt live.

    One directory per model, because `hpc_transfers.py down <model>` has to be
    able to tell the two models' runs apart, and a shared directory puts both
    under a single directories.txt that no later download can split.
    """
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; have {sorted(MODELS)}")
    return repo_root() / model


# The workstation archive. Named once, here, and read by the generators, by
# hpc_transfers.py and by the tests -- it used to be a literal in each of
# them. READ ONLY: ~/.claude/CLAUDE.md and the philosophy's §20 both make this
# tree off limits for writing, and nothing in workflow/ ever writes to it.
WORKSTATION_ARCHIVE = Path("/home/debamalya/ComputationData/PostDoc/"
                           "programming/new/data")

# Where a push puts the registries, relative to the repository root. The
# cluster has no ~/ComputationData, so already_exists() reads the copy that
# travelled with the code.
ARCHIVE_SUBDIR = "archive"


def archive_dirname(model):
    """The per-model archive directory name, in one place. `data/two_band_model`
    on the workstation, `archive/two_band_model` on a cluster -- same leaf."""
    return f"{model}_model"


def default_archive(model, host=None):
    """The registry already_exists() should read, for wherever this is running.

    On the workstation that is the real archive; on a cluster it is the copy
    hpc_transfers.py pushed alongside the code. Derived instead of passed so
    that forgetting `--archive` on a cluster cannot silently turn a sweep whose
    points are all archived into a sweep that recomputes all of them --
    which is what a missing registry does, loudly but too late to be free.
    """
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; have {sorted(MODELS)}")
    # detect_host(), not the Submitter's --host. Those answer different
    # questions: --host picks which cluster's RESOURCES to size the job for,
    # and can legitimately be forced to "param" during a dry run on the
    # workstation -- where the archive is still the workstation's. The
    # registry follows the filesystem this process can actually see.
    host = host or hpc_config.detect_host()
    if host == "local":
        return WORKSTATION_ARCHIVE / archive_dirname(model)
    return repo_root() / ARCHIVE_SUBDIR / archive_dirname(model)


@lru_cache(maxsize=8)
def _parse_cached(log_path_str, system):
    """parse_master_log is re-read once per submitted job otherwise; the log
    grows to thousands of lines. Cached on a string path so it stays hashable."""
    return parse_master_log(Path(log_path_str), system)


def live_jobs_by_workdir(user, timeout=30):
    """{workdir: [(job id, state), ...]} for this user's queued and running jobs.

    Returns None -- never {} -- when squeue could not be reached. The two mean
    opposite things to every caller here.

    %Z is the SUBMIT directory. For an array element that is the directory the
    array was sbatched from, NOT the run directory the element cds into, which
    is why this cannot be used to match dir_NNNNN to jobs one by one (see
    _array_still_live). It is exactly the right key for the one question it is
    asked: is a job of mine live somewhere in this repository?
    """
    try:
        out = subprocess.run(["squeue", "-u", user, "--noheader", "-o", "%i|%T|%Z"],
                             capture_output=True, text=True, check=True,
                             timeout=timeout)
    except (subprocess.CalledProcessError, FileNotFoundError,
            subprocess.TimeoutExpired):
        return None
    jobs = {}
    for line in out.stdout.splitlines():
        parts = line.strip().split("|")
        if len(parts) == 3:
            job_id, state, workdir = parts
            jobs.setdefault(workdir, []).append((job_id, state))
    return jobs


# NOTE (2026-08-27): count_running_jobs(), cpus_in_use() and
# get_active_job_info() lived here and are gone. All three existed to feed
# wait_for_slot(), the poll loop that `#SBATCH --array=0-N%K` replaced, and
# nothing has called them since. Each also had a failure mode worth not
# keeping: count_running_jobs() reported zero running jobs when squeue could
# not be reached at all, and get_active_job_info() matched live jobs by
# squeue's %Z, which an array element reports as the SUBMIT directory rather
# than its run directory -- the miss that _array_still_live() exists to
# avoid. Dead code that is also wrong is the kind that gets called again.


# ══════════════════════════════════════════════════════════════════════════
# The pre-submission gate
# ══════════════════════════════════════════════════════════════════════════
#
# One sbatch here dispatches a whole array, and the array is sized to occupy
# this user's entire core budget per point (see CLAUDE.md, "Resource
# allocation defaults"). There is no throttle to catch a mistake afterwards
# and no per-point compile to fail slowly: the next thing after this gate is
# N points running at full width. So the gate is typed, not a keystroke.
#
# It lives here rather than in each generator because both generators must ask
# the identical question -- a second copy is how the two drift, which is the
# same failure that put a stale -mtune in Makefile.job (CLAUDE.md, "hpc runs
# before 2026-08-21").

CONFIRM_TIMEOUT_S = 120

# What SLURM will accept after --job-name= without quoting, and what a shell
# reading array_logs/<name>_*.out will not misparse. Checked rather than
# quoted, because a name carrying a space or a $ is far more likely to be a
# typo than an intention.
JOB_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")


def _timeout_handler(signum, frame):
    raise TimeoutError


def _ask(prompt, timeout=CONFIRM_TIMEOUT_S):
    """input() that cannot leave a login-node shell blocked forever.

    A generator is run from an interactive login-node shell and then exits in
    seconds. If it is ever started detached, an unanswered input() would hold
    the terminal open indefinitely with no indication why, so the read is
    given an alarm and a timeout is a cancel -- never a default answer.
    """
    if hasattr(signal, "SIGALRM"):
        previous = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout)
        try:
            return input(prompt)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)
    return input(prompt)


def confirm_and_job_name(default_name, summary=(), timeout=CONFIRM_TIMEOUT_S,
                         interactive=None, name=None):
    """Show the plan, require a typed "yes", then read the SLURM job name.

    Returns the job name to put on the array. Any answer other than exactly
    "yes" aborts, as does a timeout, an EOF, or a name that is not a legal
    --job-name value after three attempts. Nothing is prepared, locked, built
    or submitted before this returns.

    Parameters
    ----------
    default_name : name used when the job-name answer is left empty. The
        model's own default, so pressing Enter is a choice rather than a gap.
    summary : lines describing what is about to be submitted, printed above
        the prompt. This is the only place the operator sees the sweep width
        and the skip/recompute mode together, which is what makes "yes"
        meaningful rather than reflexive.
    name : answers the SECOND question (--name on the command line) and is
        still validated. It does NOT answer the first: the generator used to
        skip this whole call when --name was given, so a flag whose documented
        job is to name the array quietly removed the confirmation as well.
        There is one way past this gate and it is typing yes.
    interactive : override the tty test. Default None = ask sys.stdin.

    Raises
    ------
    SystemExit : on decline, timeout, EOF, no terminal, or a rejected name.
        Always SystemExit and never a return value the caller might ignore --
        a gate that can be accidentally passed is not a gate
        (research_backup/.../research_code_development_philosophy.md, Part I).
    """
    if interactive is None:
        interactive = sys.stdin.isatty()

    for line in summary:
        print(line, flush=True)

    # Same reasoning as the resume/begin decision below: inside a batch job
    # there is no stdin, and input() would block until the walltime killed it
    # hours later with nothing in the log to say why. Refusing loudly beats
    # both that and the alternative of assuming "yes" on a sweep this size.
    if not interactive:
        raise SystemExit(
            "❌ this process has no terminal to confirm on, and a sweep is "
            "never submitted unconfirmed. Run the generator from an "
            "interactive login-node shell.")

    try:
        if _ask('\n⚠️  Type yes to submit these jobs '
                f'({timeout}s timeout): ', timeout) != "yes":
            raise SystemExit("🛑 Cancelled. Nothing was prepared or submitted.")

        if name is not None:
            if not JOB_NAME_RE.match(name):
                raise SystemExit(
                    f"❌ --name {name!r} is not usable after --job-name=. "
                    f"Letters, digits, dot, dash, underscore; must start with "
                    f"a letter or digit; 64 characters at most. Nothing was "
                    f"prepared or submitted.")
            print(f"📛 Job name for the array: {name}  (from --name)", flush=True)
            return name

        for attempt in (1, 2, 3):
            name = _ask(f'📛 Job name for the array [{default_name}] '
                        f'({timeout}s timeout): ', timeout).strip()
            if not name:
                return default_name
            if JOB_NAME_RE.match(name):
                return name
            print(f"   ✗ {name!r} is not usable after --job-name=. Letters, "
                  f"digits, dot, dash, underscore; must start with a letter "
                  f"or digit; 64 characters at most."
                  + (f"  ({3 - attempt} attempt(s) left)" if attempt < 3 else ""),
                  flush=True)
    except TimeoutError:
        raise SystemExit(f"⏳ No answer in {timeout}s. Cancelled. Nothing was "
                         f"prepared or submitted.")
    except EOFError:
        raise SystemExit("❌ stdin closed before the gate was answered. "
                         "Cancelled. Nothing was prepared or submitted.")

    raise SystemExit("❌ No usable job name after 3 attempts. Cancelled. "
                     "Nothing was prepared or submitted.")


class Submitter:
    def __init__(self, model, host=None, data_path=None, dry_run=False,
                 run_root=None):
        if model not in MODELS:
            raise ValueError(f"unknown model {model!r}; have {sorted(MODELS)}")
        self.model = model
        self.schema = MODELS[model]
        self.host = hpc_config.get(host)
        # <repo>/<model>, derived from this file (run_root_for). NOT Path.cwd():
        # see the comment above repo_root(). run_root= exists for tests that
        # need a scratch tree; production passes nothing and gets the one
        # layout, on the workstation and on both clusters alike.
        self.run_root = Path(run_root) if run_root else run_root_for(model)
        self.dry_run = dry_run
        self.iteration = 0
        self.job_name = "run"
        # expanduser(), NOT data_utils.clean_path(). clean_path maps between
        # this workstation's two roots (/home/debamalya <-> D:) and prefixes
        # anything it does not recognise with /home/debamalya -- so on hpc it
        # turned /iitgn/homedirs/<user>/projects/new/archive/two_band_model
        # into /home/debamalya//iitgn/homedirs/..., a path that cannot exist.
        # already_exists() would then find no registry and submit every
        # archived point as new. It is an analysis-side helper for this
        # machine's filesystem, and a cluster is not this machine.
        self.data_path = (Path(data_path).expanduser() if data_path
                          else default_archive(model))

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
                     f"{p.numa_per_node} NUMA domain(s)")
        # No walltime here. The job requests none (array_script), so the queue
        # default applies, and that number belongs to the administrators --
        # `scontrol show partition <queue>` is the live answer.

        per_node = self.npn * self.cpt
        if per_node != p.cores_per_node:
            lines.append(f"  ⚠️  {per_node} of {p.cores_per_node} cores per node "
                         f"in use — this layout does not fill a node")
        if need > p.cpu_cap:
            lines.append(f"  ⚠️  one job needs {need} cores but the cap is "
                         f"{p.cpu_cap} — it will never start")
        return "\n".join(lines)

    # ------------------------------------------------------------ run root --
    def ensure_run_root(self):
        """Create <repo>/<model>/ if it is not there, and say so when it does.

        Called once by the generator, right after the confirmation gate.
        Separate from __init__ because constructing a Submitter must not
        create directories -- the archive-match test builds one only to ask
        already_exists() questions -- and separate from new_run_dir() because
        a run root that appears silently is how 96 run directories ended up in
        workflow/ without anyone noticing until a download could not tell the
        two models apart.
        """
        if self.run_root.exists():
            if not self.run_root.is_dir():
                raise SystemExit(
                    f"🚨 ensure_run_root: {self.run_root} exists and is not a "
                    f"directory. That is where {self.model}'s run directories "
                    f"go; move or remove it before submitting.")
            return self.run_root
        self.run_root.mkdir(parents=True)
        print(f"created run root {self.run_root}", flush=True)
        return self.run_root

    def _require_run_root(self):
        """Every writer goes through here, so a missing run root is one
        sentence naming the fix rather than a FileNotFoundError raised from
        inside iterdir() three frames down."""
        if not self.run_root.is_dir():
            raise SystemExit(
                f"🚨 run root {self.run_root} does not exist. ensure_run_root() "
                f"must run before any point is prepared -- the generator calls "
                f"it after the confirmation gate.")
        return self.run_root

    # ---------------------------------------------------------------- lock --
    def acquire_lock(self):
        lock = self._require_run_root() / LOCKFILE
        if lock.exists():
            raise SystemExit(
                f"{LOCKFILE} exists in {self.run_root} — another submitter may be "
                f"running. Remove it if you are sure it is stale.")
        lock.write_text(str(os.getpid()))
        self._lockfile = lock
        atexit.register(self.release_lock)

    def release_lock(self):
        """Remove the lock, and say so loudly if it could not be removed.

        Registered with atexit, so raising here would replace whatever the
        process was already exiting for. But swallowing the error is worse in
        its own way: a lock left behind stops the NEXT submission with
        "another submitter may be running", and the operator then has no
        record of which run left it or why. Reported, not raised, and never
        silent.
        """
        if not (self._lockfile and self._lockfile.exists()):
            return
        try:
            self._lockfile.unlink()
        except OSError as e:
            print(f"⚠️  release_lock: could not remove {self._lockfile} ({e}). "
                  f"The next submission will refuse to start until it is gone "
                  f"-- delete it by hand once you have checked nothing else is "
                  f"submitting here.", flush=True)

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
            except OSError as e:
                # An unreadable log used to be skipped, which made it
                # indistinguishable from one that simply has no marker -- and
                # "no marker" is what preparing_run() acts on by quarantining
                # or DELETING the directory. A log this process cannot read is
                # not evidence about the run; it is a fact about the
                # filesystem, and it stops the sweep rather than deciding the
                # question the wrong way.
                raise SystemExit(
                    f"🚨 is_job_complete: cannot read {log} ({e}). Whether "
                    f"{run_dir.name} finished cannot be decided, and the next "
                    f"step would be to quarantine or delete it. Fix the "
                    f"permissions or move the directory aside by hand.")
        return False

    def preparing_run(self, interactive=None):
        """Return how many leading directories in directories.txt are finished,
        so a resumed sweep picks up where it stopped.

        Refuses to act while any candidate directory belongs to a live job:
        'resume' would move the folder out from under it and 'begin' would
        delete it. See AM3_submit_utils.preparing_run.
        """
        self.refuse_foreign_repo_jobs()

        dir_txt = self.run_root / DIRECTORIES_TXT
        if not dir_txt.exists():
            return 0

        lines = [l.strip() for l in dir_txt.read_text().splitlines() if l.strip()]
        start_index = 0
        for line in lines:
            parts = line.split(". ", 1)
            d_name = parts[1] if len(parts) == 2 else line.split()[-1]
            d = self.run_root / d_name
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
            backup = self.run_root / BACKUP_DIRNAME
            backup.mkdir(exist_ok=True)
            for line in lines[start_index:]:
                parts = line.split(". ", 1)
                d_name = parts[1] if len(parts) == 2 else line.split()[-1]
                d = self.run_root / d_name
                if d.exists():
                    shutil.move(str(d), str(backup / d_name))
            dir_txt.write_text("\n".join(lines[:start_index]) + "\n")
        elif choice == "begin":
            for line in lines:
                parts = line.split(". ", 1)
                d_name = parts[1] if len(parts) == 2 else line.split()[-1]
                d = self.run_root / d_name
                if d.exists():
                    shutil.rmtree(d)
            dir_txt.unlink()
            start_index = 0
        else:
            raise SystemExit("aborted.")

        self.iteration = start_index
        return start_index

    def refuse_foreign_repo_jobs(self):
        """Refuse to submit while a job of this user is live in THIS repository
        but outside every model's run root.

        The case this exists for is real and was live on hpc on 2026-08-27:
        array 214963 was still running 46 elements out of
        ~/projects/new/workflow/, because run directories used to be created
        in the cwd and the generator had been invoked from there. Moving the
        run root to <repo>/<model> makes that array invisible to
        preparing_run() -- it reads .array_job_id in the NEW root, which is
        empty and freshly created -- so a second array for the same 46 points
        would go out beside the first, both drawing on one 80-core cap and
        both computing the same physics.

        Scoped to this repository and to directories that are not run roots,
        so the legitimate case stays legitimate: a live four_band array
        submitted from <repo>/four_band is a run root and does not block a
        two_band submission, while this model's OWN live array is caught more
        precisely by _array_still_live() a few lines below.

        squeue unreachable is reported and does not block -- the check cannot
        be the reason a submission is impossible on a machine where nothing
        can be verified either way.
        """
        jobs = live_jobs_by_workdir(self.slurm_user)
        if jobs is None:
            print(f"⚠️  refuse_foreign_repo_jobs: could not reach squeue, so "
                  f"nothing was checked. Confirm by hand that no earlier "
                  f"sweep of this repository is still queued.", flush=True)
            return

        root = repo_root().resolve()
        known = {run_root_for(m).resolve() for m in MODELS}
        foreign = {}
        for workdir, entries in jobs.items():
            try:
                d = Path(workdir).resolve()
            except OSError:
                continue
            inside_repo = d == root or root in d.parents
            if not inside_repo or d in known:
                continue
            foreign[d] = entries

        if not foreign:
            return

        detail = "\n".join(
            f"     {d}: " + ", ".join(f"{j} ({s})" for j, s in entries[:4])
            + (", ..." if len(entries) > 4 else "")
            for d, entries in sorted(foreign.items()))
        raise SystemExit(
            f"🚨 a job of yours is still live inside {root}, submitted from a "
            f"directory that is not any model's run root:\n{detail}\n"
            f"   Run directories used to be created in the current directory; "
            f"they are now created in <repo>/<model>/, so that sweep is "
            f"invisible to this one and submitting now would put a second "
            f"array of the same points beside it, on the same core budget.\n"
            f"   Wait for it (squeue -u {self.slurm_user}), or scancel it, "
            f"then rerun. Nothing was prepared or submitted.")

    # -------------------------------------------------------------- submit --
    def new_run_dir(self):
        root = self._require_run_root()
        existing = {d.name for d in root.iterdir() if d.is_dir()}
        for _ in range(1000):
            candidate = f"dir_{random.randint(10000, 99999)}"
            if candidate not in existing:
                d = root / candidate
                # No exist_ok: the name was just checked against everything in
                # this directory, so a collision here means something else is
                # writing into the run root at the same time, and two sweeps
                # sharing one directories.txt is precisely what must not
                # happen quietly.
                d.mkdir()
                return d
        raise RuntimeError("no free dir_NNNNN name after 1000 tries")

    def record_directory(self, run_dir):
        with open(self._require_run_root() / DIRECTORIES_TXT, "a") as f:
            f.write(f"{self.iteration}. {run_dir.name}\n")

    # ------------------------------------------------------ submission log --
    def open_submission_log(self, header_lines=()):
        """Start this invocation's block in <run root>/submission_log.txt.

        The per-point lines go here and not to the terminal. On a 207-point
        sweep they pushed the two lines that decide whether to intervene --
        the resource plan and the sbatch result -- off the top of the screen,
        while adding nothing that is not also in directories.txt and the
        task.csv files. What the terminal keeps is the summary and anything
        that went wrong.
        """
        self._require_run_root()
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        head = (f"\n{'=' * 78}\n"
                f"submission {stamp}  model={self.model}  host={self.host.name}\n")
        body = "".join(f"{line}\n" for line in header_lines)
        with open(self.run_root / SUBMIT_LOG, "a") as f:
            f.write(head + body)
        return self.run_root / SUBMIT_LOG

    def note(self, line):
        """One line into the submission log. Never to the terminal."""
        with open(self._require_run_root() / SUBMIT_LOG, "a") as f:
            f.write(f"{line}\n")

    @property
    def submission_log(self):
        return self.run_root / SUBMIT_LOG

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
            self._discard(run_dir)
            self._quarantine_reason(run_dir.name, "task.csv could not be written", str(e))
            return None

        if self.dry_run:
            # No per-point note here: the caller already logs one line per
            # prepared point, and the log's header block records dry run =
            # True, which is the only thing that differs.
            self.iteration += 1
            self.record_directory(run_dir)
            return run_dir

        binary = self.binary_path(repo)
        if not binary.exists():
            self._discard(run_dir)
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
        a remembered one.

        None means the limit is UNKNOWN, never "no limit". Off-cluster that is
        silent and expected; on a cluster it is said out loud, because the
        caller's next move is to submit without having checked.
        """
        try:
            out = subprocess.run(["scontrol", "show", "config"],
                                 capture_output=True, text=True, check=True,
                                 timeout=20).stdout
        except FileNotFoundError:
            return None                  # no scontrol: not a cluster
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"⚠️  max_array_size: `scontrol show config` failed "
                  f"({type(e).__name__}); submitting without checking the "
                  f"array-size limit.", flush=True)
            return None
        found = re.search(r"MaxArraySize\s*=\s*(\d+)", out)
        if not found:
            print(f"⚠️  max_array_size: no MaxArraySize in `scontrol show "
                  f"config`; submitting without checking the array-size "
                  f"limit.", flush=True)
            return None
        return int(found.group(1))

    # A `#SBATCH` line after the first executable line is IGNORED by sbatch,
    # silently. So is one that does not start at column 0. Both are checked
    # rather than trusted -- see _validate_batch_script.
    _SBATCH = "#SBATCH"

    def array_script(self, count, exe="./exe_run", task_file="task.csv"):
        """The one job script an entire sweep runs under.

        Every element is the SAME binary on the SAME resources, differing only
        in which directory it enters -- which is exactly what an array is for,
        and exactly what became possible once parameters stopped being
        compiled in.

        Built as a list of lines, NOT as one dedented f-string. That is not a
        style preference. `textwrap.dedent(f"...")` interpolates first and
        dedents second, so the moment a substituted value carries its own
        newlines -- the module block does, on both clusters -- the common
        indent it can strip becomes the empty string and dedent returns the
        template untouched. Every line, `#!/bin/bash` included, then keeps the
        source file's indentation, and sbatch rejects the whole thing with
        "This does not look like a batch script. The first line must start
        with #!". It failed that way on param on 2026-08-27, after the sweep
        had already prepared 24 run directories.

        Three details that are not cosmetic:

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
        3. NO `--time`. The queue's limit is the cluster administrator's to
           set: a job that requests nothing gets the queue default, which is
           the number they chose. Requesting one here would be this project
           re-deciding it from a table that nobody re-measures, and a table
           that drifts LOW silently shortens every run.
        """
        if count < 1:
            raise ValueError(f"array_script: an array needs at least one "
                             f"element, got {count}")

        directives = [f"--job-name={self.job_name}",
                      f"--partition={self.partition}",
                      f"--qos={self.qos}",
                      f"--nodes={self.nodes}",
                      f"--ntasks-per-node={self.npn}",
                      f"--cpus-per-task={self.cpt}",
                      "--hint=nomultithread",
                      # %K throttling: SLURM itself runs at most this many
                      # elements at once. This replaces the old Python
                      # wait_for_slot() poll loop, and with it the only reason
                      # the submitting process had to stay alive at all.
                      f"--array=0-{count - 1}%{self.max_concurrent_jobs}",
                      f"--output={ARRAY_LOG_DIR}/%A_%a.out",
                      f"--error={ARRAY_LOG_DIR}/%A_%a.err"]

        lines = ["#!/bin/bash"]
        lines += [f"{self._SBATCH} {d}" for d in directives]
        # extra_sbatch entries are whole directives from the host profile.
        # Accepted with or without the prefix, because a profile that spells
        # one out and one that does not must not produce different scripts.
        for extra in self.host.extra_sbatch:
            extra = extra.strip()
            if not extra:
                continue
            lines.append(extra if extra.startswith(self._SBATCH)
                         else f"{self._SBATCH} {extra}")

        lines += ["", "set -e", ""]

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

        script = "\n".join(lines)
        self._validate_batch_script(script)
        return script

    def _validate_batch_script(self, script):
        """Refuse to hand sbatch something it would misread.

        Every rule here is a failure mode that produces no error at
        submission time, or produces one only after the sweep has already
        written its run directories:

        * a first line that is not `#!` -- sbatch rejects the file outright,
          which is the loud one, and it is what indentation did on param;
        * an indented `#SBATCH` -- ignored, so the job silently takes the
          cluster default for whatever that directive set;
        * a `#SBATCH` after the first executable line -- ignored for the same
          reason and much harder to see, because the line looks correct;
        * `--time` -- this project does not set walltimes (see array_script),
          so one appearing here means something re-introduced a hand-kept
          limit.

        Raises. A script that is wrong in any of these ways is wrong for every
        element of the array, and the array is the whole sweep.
        """
        lines = script.splitlines()
        if not lines or not lines[0].startswith("#!"):
            raise SystemExit(
                f"🚨 _validate_batch_script: the generated script does not "
                f"start with #!. sbatch refuses it with 'This does not look "
                f"like a batch script'. First line was "
                f"{(lines[0] if lines else '')!r}.")

        seen_command = False
        for n, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith(self._SBATCH):
                if not line.startswith(self._SBATCH):
                    raise SystemExit(
                        f"🚨 _validate_batch_script: line {n} is an indented "
                        f"#SBATCH directive, which sbatch ignores silently: "
                        f"{line!r}")
                if seen_command:
                    raise SystemExit(
                        f"🚨 _validate_batch_script: line {n} is a #SBATCH "
                        f"directive after the script's first command, where "
                        f"sbatch stops reading them: {line!r}")
                if "--time" in stripped:
                    raise SystemExit(
                        f"🚨 _validate_batch_script: line {n} requests a "
                        f"walltime: {line!r}. The queue's limit is the "
                        f"cluster administrator's to set and this project "
                        f"does not impose one -- see array_script().")
            elif stripped and not stripped.startswith("#"):
                seen_command = True
        return script

    def submit_array(self, run_dirs):
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
            # Not an error -- a sweep whose points are all archived prepares
            # nothing, which is the normal outcome of re-running one. But the
            # reason is worth saying, because "nothing to submit" and "the
            # sweep failed" look identical in a scrollback otherwise.
            print(f"nothing to submit: no point of this sweep needed "
                  f"preparing.", flush=True)
            return None

        self._require_run_root()
        limit = self.max_array_size()
        if limit is not None and len(run_dirs) > limit:
            raise SystemExit(
                f"{len(run_dirs)} points exceeds this cluster's MaxArraySize "
                f"of {limit}. Split the sweep, or raise it with the admins.")

        # SLURM writes here before the script cds anywhere, so a failure that
        # happens before the cd still leaves a trace. If it cannot be created
        # the array has nowhere to report such a failure, so this is a stop.
        log_dir = self.run_root / ARRAY_LOG_DIR
        log_dir.mkdir(exist_ok=True)
        if not log_dir.is_dir():
            raise SystemExit(
                f"🚨 {log_dir} is not a directory. SLURM writes every array "
                f"element's pre-cd output there; without it a failure before "
                f"the cd leaves nothing behind at all.")

        # The index -> directory map the array script reads. Line N+1 is the
        # directory for SLURM_ARRAY_TASK_ID N. Written and then read back:
        # every element resolves its own directory through this file, so a
        # short or reordered map is 1..N wrong runs in the right directories,
        # which nothing downstream would flag.
        names = [d.name for d in run_dirs]
        (self.run_root / ARRAY_MAP).write_text("\n".join(names) + "\n")
        written = (self.run_root / ARRAY_MAP).read_text().split()
        if written != names:
            raise SystemExit(
                f"🚨 {ARRAY_MAP} does not read back as it was written "
                f"({len(written)} of {len(names)} entries match). Every array "
                f"element resolves its run directory through that file. "
                f"Nothing was submitted.")

        script = self.run_root / f"sweep_{self.model}.sh"
        script.write_text(self.array_script(len(run_dirs)))

        if self.dry_run:
            print(f"  [dry-run] would submit {len(run_dirs)} point(s) as an "
                  f"array, {self.max_concurrent_jobs} at a time\n"
                  f"  [dry-run] script written: {script}", flush=True)
            return None

        try:
            r = subprocess.run(["sbatch", script.name], cwd=self.run_root,
                               capture_output=True, text=True)
        except FileNotFoundError:
            raise SystemExit(
                f"🚨 no sbatch on this machine, so {script} was written but "
                f"not submitted. Run the generator on a cluster login node, "
                f"or with --dry-run.")
        if r.returncode != 0:
            raise SystemExit(f"🚨 could not submit the array job:\n"
                             f"{r.stderr.strip()}\n   The script is at {script} "
                             f"and the run directories are prepared; fix the "
                             f"cause and sbatch it, or rerun the generator.")

        job_id = re.search(r"(\d+)", r.stdout)
        if not job_id:
            raise SystemExit(
                f"🚨 sbatch reported success but named no job id: "
                f"{r.stdout.strip()!r}. Without the id nothing can tell later "
                f"whether this array is still running, so check `squeue -u "
                f"{self.slurm_user}` by hand before submitting again.")
        job_id = job_id.group(1)
        # preparing_run() needs this on a later invocation: an array element
        # reports the SUBMIT directory as its WorkDir, not its run directory,
        # so directories can no longer be matched to live jobs one by one.
        (self.run_root / ARRAY_JOB_ID).write_text(job_id + "\n")

        self.note(f"array {job_id}: {len(run_dirs)} element(s), "
                  f"{self.max_concurrent_jobs} at a time, "
                  f"{self.cores_per_job()} cores each")
        print(f"array {job_id}: {len(run_dirs)} point(s), "
              f"{self.max_concurrent_jobs} at a time.  "
              f"squeue -u {self.slurm_user}", flush=True)
        return job_id

    def _discard(self, run_dir):
        """Remove a directory this sweep created moments ago and could not
        finish filling.

        Only ever called on a dir_NNNNN that new_run_dir() made in this
        process, before anything of value is in it -- that is what makes
        removal the right move rather than quarantine. It used to be
        rmtree(ignore_errors=True), which turned a failed removal into a
        leftover directory that the NEXT invocation reads as a prepared run:
        empty, in no directories.txt, and impossible to explain later.
        """
        try:
            shutil.rmtree(run_dir)
        except OSError as e:
            raise SystemExit(
                f"🚨 could not remove the half-prepared {run_dir} ({e}). It has "
                f"no task.csv and no binary, so leaving it in place would put "
                f"an empty run directory in the run root. Remove it by hand "
                f"and rerun.")

    def _quarantine(self, run_dir, why, detail=""):
        """Move a failed run dir aside so it can be inspected, without letting
        it look like a pending job to preparing_run().

        A failed MOVE used to fall through to deleting the directory. That is
        backwards: quarantine exists so a failure can be looked at, and the
        one case where the move fails is the one case where nobody gets to.
        Now the directory stays where it is and the sweep stops -- a run root
        that cannot be tidied is a fact about the filesystem, and the rest of
        the sweep would be writing into the same one.
        """
        backup = self.run_root / BACKUP_DIRNAME
        backup.mkdir(exist_ok=True)
        target = backup / run_dir.name
        if target.exists():
            target = backup / f"{run_dir.name}_{int(time.time())}"
        try:
            shutil.move(str(run_dir), str(target))
        except OSError as e:
            raise SystemExit(
                f"🚨 {why} for {run_dir.name}, and it could not be moved to "
                f"{target} ({e}). It is still in the run root, where "
                f"preparing_run() will read it as an unfinished run. Nothing "
                f"else was prepared or submitted."
                + (f"\n      {detail.strip()}" if detail else ""))
        print(f"  ⚠️  {why} for {run_dir.name}; moved to {target}", flush=True)
        if detail:
            tail = "\n".join(detail.strip().splitlines()[-8:])
            print(f"      {tail}", flush=True)

    def _recorded_array_id(self):
        """The last array job id submitted from this directory, or None."""
        path = self.run_root / ARRAY_JOB_ID
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
            # squeue exits nonzero BOTH for a job id it no longer knows about
            # -- the normal state once an array has drained -- and for a real
            # failure to answer. Only the first means "nothing is running",
            # and the difference matters: the caller's 'begin' branch DELETES
            # every directory in directories.txt on the strength of it. So the
            # message is read, and anything else is "unknown".
            if "invalid job id" in (out.stderr or "").lower():
                return []
            print(f"⚠️  _array_still_live: squeue could not be asked about "
                  f"array {job_id}: {(out.stderr or '').strip()}", flush=True)
            return None
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
