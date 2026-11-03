"""
SFTP transfer layer for pushing this project to PARAM/HPC and pulling run
results back. Adapted from two_component_BH/workflow/hpc_transfer_utils.py --
the mechanism (one multiplexed SSH session, keyboard-interactive auth for
2FA, recursive upload/download with exclude patterns, a percentage progress
callback) is generic and kept as-is. What changed is where a host's
connection details come from: hpc_config.HostConfig already holds everything
that differs between local/param/hpc, so hostname/port/user are read from
there instead of a second, parallel host registry.

check_remote_run_completion() / _warn_if_run_incomplete() are ported
separately, from Superconductivity/python/hpc_transfer_utils.py -- a safety
feature that project's transfer layer has and this one did not: before a
download, verify (one extra command on the already-open session) that every
SLURM job whose logs live under the remote source actually finished
cleanly, and warn loudly, without blocking, if not.

delete_directory_contents() comes from the same place, and lives here rather
than in hpc_transfers.py so that the only destructive routine in the transfer
path sits with its guards, in one copy, next to the code it protects.

Not imported by workflow/submit_utils.py or the generators -- this only runs
on the workstation, driving hpc_transfers.py. A cluster login node has no
outbound reason to open an SFTP session to itself.
"""

from __future__ import annotations

import fnmatch
import getpass
import os
import posixpath
import shutil
import signal
import stat as statmod
import sys
from pathlib import Path, PurePosixPath

import paramiko

import hpc_config

# ======================================================================================
# hosts
# ======================================================================================

# Only the two real clusters are transfer targets. LOCAL has no remote
# counterpart -- generator_*.py runs directly where you invoke it.
TRANSFER_HOSTS = ("param", "hpc")

_KNOWN_EXTENSIONLESS = {"Makefile", "Dockerfile", "LICENSE", "README"}

SSH_CLIENT = None
SFTP = None
_ACTIVE_HOST = None


def base_path(host: str) -> PurePosixPath:
    """Remote $HOME for host, as a POSIX path -- hpc_config.HostConfig.remote_home,
    which mirrors the LIBROOT convention physics/Makefile already uses."""
    _check_host(host)
    return PurePosixPath(hpc_config.get(host).remote_home)


def _check_host(host: str) -> None:
    if host not in TRANSFER_HOSTS:
        raise RuntimeError(f"unknown host '{host}'. Use one of {TRANSFER_HOSTS}.")


def timed_input(prompt: str, timeout: int = 30) -> str:
    """input() with a hard timeout."""
    if os.name == "nt":  # pragma: no cover - Windows path
        from inputimeout import inputimeout, TimeoutOccurred
        try:
            return inputimeout(prompt=prompt, timeout=timeout)
        except TimeoutOccurred:
            raise TimeoutError

    def _handler(signum, frame):
        raise TimeoutError

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout)
    try:
        return input(prompt)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def ask_host(timeout: int = 30) -> str:
    try:
        answer = timed_input(f"👉 Enter server name ({'/'.join(TRANSFER_HOSTS)}) [30s timeout]: ", timeout)
    except TimeoutError:
        raise SystemExit("🚨 timeout with no host given. Cancelled.")
    host = answer.strip().lower()
    _check_host(host)
    return host


def ask_mode(timeout: int = 30) -> str:
    try:
        answer = timed_input("👉 Enter operation (up/down) [30s timeout]: ", timeout)
    except TimeoutError:
        raise SystemExit("🚨 timeout with no operation given. Cancelled.")
    mode = answer.strip().lower()
    if mode not in ("up", "down"):
        raise SystemExit(f"🚨 invalid operation '{mode}'. Use 'up' or 'down'.")
    return mode


def ask_choice(label: str, options, *, note: str = "", timeout: int = 30) -> str:
    """A numbered 👉 menu over `options`, answerable by number or by name.

    Superconductivity's config menu, generalised: there the six AM*-phase
    names are written out as literal print() lines, which is fine for one
    fixed list and wrong for a list that comes from a schema. `options` is
    taken as given and the numbering is derived, so a model added to
    analysis/models.py appears here without this function being edited.

    Deliberately has no "skip" path. A menu whose blank answer means
    "something reasonable" puts the caller in the position of guessing, and
    every caller here needs a definite answer: the choice selects a remote
    directory, not a label. Nothing chosen is a SystemExit, same as a timeout.
    """
    options = list(options)
    if not options:
        raise SystemExit(f"🚨 ask_choice: no {label} to choose from -- nothing is configured.")

    print(f"\n{label}:" + (f"\n    {note}" if note else ""), flush=True)
    for i, name in enumerate(options, 1):
        print(f"  👉{i}. {name}", flush=True)

    try:
        answer = timed_input(f"👉 Enter number or name [{timeout}s timeout]: ", timeout).strip()
    except TimeoutError:
        raise SystemExit(f"🚨 timeout with no {label} chosen. Cancelled.")

    if answer.isdigit() and 1 <= int(answer) <= len(options):
        return options[int(answer) - 1]
    if answer in options:
        return answer
    raise SystemExit(f"🚨 invalid choice {answer!r}. Use a number 1-{len(options)} "
                     f"or one of {options}.")


# ======================================================================================
# connection
# ======================================================================================

def _progress(filename):
    last = {"value": -1}

    def callback(transferred, total):
        percentage = int((transferred / total) * 100) if total else 0
        if percentage != last["value"]:
            print(f"\r    {filename}: {percentage}%", end="", flush=True)
            last["value"] = percentage
        if transferred == total:
            print()

    return callback


def _interactive_handler(title, instructions, prompt_list):
    """Keyboard-interactive auth: password, captcha string, or 2FA code."""
    responses = []
    if instructions:
        print(f"\n{instructions}")
    for prompt, _echo in prompt_list:
        if any(token in prompt for token in ("string", "code", "Verification", "Password")):
            if "Password" in prompt:
                responses.append(getpass.getpass(prompt))
            else:
                responses.append(input(prompt))
        else:
            responses.append("")
    return responses


def _resolve_target(host: str):
    """hpc_config gives the built-in hostname/port/user; ~/.ssh/config (if it
    has an entry for `host`) overrides any of the three, same as the
    reference -- so an existing alias for 'param'/'hpc' still wins."""
    cfg = hpc_config.get(host)
    hostname, port, username = cfg.hostname, cfg.ssh_port, cfg.slurm_user
    key_filename = None

    config_path = Path.home() / ".ssh" / "config"
    if config_path.exists():
        try:
            config = paramiko.SSHConfig.from_path(str(config_path))
            resolved = config.lookup(host)
            if resolved.get("hostname") and resolved["hostname"] != host:
                hostname = resolved["hostname"]
            port = int(resolved.get("port", port))
            username = resolved.get("user", username)
            identity = resolved.get("identityfile")
            if identity:
                path = identity[0] if isinstance(identity, list) else identity
                key_filename = os.path.expanduser(path)
        except Exception as exc:
            # Not fatal -- the built-in hpc_config values are a complete answer
            # on their own. But it is never silent: an unparseable ~/.ssh/config
            # means an alias the user believes is in force is not, and the
            # symptom would otherwise be a connection to the wrong host or port
            # with nothing on screen to explain it.
            print(f"⚠️  _resolve_target: could not read {config_path} ({exc}); "
                  f"using hpc_config values for '{host}' instead.", flush=True)

    return hostname, port, username, key_filename


def connect(host: str) -> None:
    """Open one SSH session and one SFTP channel, reused for every transfer."""
    global SSH_CLIENT, SFTP, _ACTIVE_HOST

    hostname, port, username, key_filename = _resolve_target(host)
    print(f"connecting to '{hostname}' port {port} as {username}", flush=True)

    SSH_CLIENT = paramiko.SSHClient()
    SSH_CLIENT.load_system_host_keys()
    known_hosts = Path.home() / ".ssh" / "known_hosts"
    if known_hosts.exists():
        SSH_CLIENT.load_host_keys(str(known_hosts))
    SSH_CLIENT.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        SSH_CLIENT.connect(
            hostname=hostname, port=port, username=username, key_filename=key_filename,
            allow_agent=True, look_for_keys=True,
            timeout=60, banner_timeout=60, auth_timeout=60,
        )
    except (paramiko.AuthenticationException, paramiko.SSHException):
        # Falls back to keyboard-interactive, which is what carries the
        # cluster's captcha / 2FA prompts.
        SSH_CLIENT.close()
        SSH_CLIENT = paramiko.SSHClient()
        SSH_CLIENT.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        transport = paramiko.Transport((hostname, port))
        transport.banner_timeout = 60
        transport.start_client(timeout=60)
        transport.auth_interactive(username, _interactive_handler)
        SSH_CLIENT._transport = transport

    SFTP = SSH_CLIENT.open_sftp()
    _ACTIVE_HOST = host


def close() -> None:
    global SSH_CLIENT, SFTP, _ACTIVE_HOST
    if SFTP is not None:
        SFTP.close()
        SFTP = None
    if SSH_CLIENT is not None:
        SSH_CLIENT.close()
        SSH_CLIENT = None
    _ACTIVE_HOST = None


def run_remote(command: str) -> str:
    """Run a command on the login node; raise on nonzero exit."""
    if SSH_CLIENT is None:
        raise RuntimeError("not connected -- call connect(host) first")
    _stdin, stdout, stderr = SSH_CLIENT.exec_command(command)
    code = stdout.channel.recv_exit_status()
    out = stdout.read().decode("utf-8", "ignore")
    if code != 0:
        message = stderr.read().decode("utf-8", "ignore").strip()
        raise RuntimeError(message or f"remote command failed: {command}")
    return out


def run_remote_capture(command: str) -> tuple:
    """Like run_remote, but returns (rc, stdout_text) instead of raising on a
    nonzero exit -- for a diagnostic that has something useful to say either
    way (job-status checks: "no matching jobs" and "sacct not on PATH" are
    both a nonzero rc worth handling, not an exception worth aborting the
    transfer over). Adapted from Superconductivity/python/hpc_transfer_utils.py.
    """
    if SSH_CLIENT is None:
        raise RuntimeError("not connected -- call connect(host) first")
    _stdin, stdout, _stderr = SSH_CLIENT.exec_command(command)
    out = stdout.read().decode("utf-8", "ignore")
    code = stdout.channel.recv_exit_status()
    return code, out


# ======================================================================================
# transfer primitives
# ======================================================================================

def _excluded(name, patterns=None) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in (patterns or []))


def _remote_is_dir_like(dst) -> bool:
    dst = str(dst)
    if dst.endswith(("/", "\\")):
        return True
    if Path(dst).name in _KNOWN_EXTENSIONLESS:
        return False
    return Path(dst).suffix == ""


def _local_is_dir_like(dst) -> bool:
    path = Path(str(dst))
    if str(dst).endswith(("/", "\\")) or (path.exists() and path.is_dir()):
        return True
    if path.name in _KNOWN_EXTENSIONLESS:
        return False
    return path.suffix == ""


def mkdir_p(remote_dir: str) -> None:
    remote_dir = str(remote_dir).rstrip("/")
    if not remote_dir:
        return
    current = ""
    for part in remote_dir.split("/"):
        if not part:
            current = "/"
            continue
        current = f"/{part}" if current == "/" else (part if not current else f"{current}/{part}")
        try:
            SFTP.stat(current)
        except IOError:
            try:
                SFTP.mkdir(current)
            except PermissionError as exc:
                raise PermissionError(
                    f"cannot create remote directory '{current}' -- check the destination is writable"
                ) from exc


def _upload_tree(local_src: Path, remote_root: str, excludes) -> int:
    sent = 0
    for item in sorted(local_src.iterdir()):
        if _excluded(item.name, excludes):
            continue
        remote_item = posixpath.join(remote_root, item.name)
        if item.is_dir():
            mkdir_p(remote_item)
            sent += _upload_tree(item, remote_item, excludes)
        else:
            mkdir_p(posixpath.dirname(remote_item))
            SFTP.put(str(item), remote_item, callback=_progress(item.name))
            sent += 1
    return sent


def upload(src, dst, excludes=None) -> int:
    """Push one file or one tree. Returns how many files were sent.

    A missing source is a HARD failure, not a skipped line. Superconductivity's
    version prints `🚨 Missing: {src}` and returns, which is survivable there
    because its transfer lists are hand-maintained alongside the data. Here the
    list is built from constants (PHYSICS_DIR / "host.mk" and the rest), so a
    source that is not on disk means either the constant is wrong or the repo is
    incomplete -- and the one time it happened, host.mk went missing from the
    push and `make` failed on the cluster minutes later with no connection drawn
    back to the transfer that had reported success. Skipping is how that stays
    invisible until the compile.
    """
    src_path = Path(str(src))
    if not src_path.exists():
        raise SystemExit(f"🚨 upload: local source does not exist: {src}\n"
                         f"   Nothing was skipped -- the transfer is stopped, because a "
                         f"source named in the plan and absent on disk is a bug in the "
                         f"plan, not an optional file.")

    excludes = excludes or []

    if src_path.is_dir():
        # A trailing separator means "the CONTENTS of this directory".
        contents_only = str(src).endswith(("/", "\\"))
        remote_root = str(dst).rstrip("/")
        if not contents_only:
            remote_root = posixpath.join(remote_root, src_path.name)
        mkdir_p(remote_root)
        sent = _upload_tree(src_path, remote_root, excludes)
        if sent == 0:
            # Every file filtered out by excludes, or an empty directory. Worth
            # saying: an upload that moved nothing looks identical to one that
            # worked once the progress lines have scrolled past.
            print(f"⚠️  upload: {src} matched no files to send "
                  f"(empty, or everything excluded).", flush=True)
        return sent

    remote_dst = str(dst).rstrip("/\\")
    try:
        is_dir = statmod.S_ISDIR(SFTP.stat(remote_dst).st_mode)
    except IOError:
        is_dir = _remote_is_dir_like(dst)
    if is_dir:
        remote_dst = posixpath.join(remote_dst, src_path.name)

    mkdir_p(posixpath.dirname(remote_dst))
    SFTP.put(str(src_path), remote_dst, callback=_progress(src_path.name))
    return 1


def _download_tree(remote_src: str, local_root, excludes) -> int:
    local_root = Path(local_root)
    local_root.mkdir(parents=True, exist_ok=True)
    got = 0
    for entry in SFTP.listdir_attr(remote_src):
        if _excluded(entry.filename, excludes):
            continue
        remote_item = posixpath.join(remote_src, entry.filename)
        local_item = local_root / entry.filename
        if statmod.S_ISDIR(entry.st_mode):
            got += _download_tree(remote_item, local_item, excludes)
        else:
            local_item.parent.mkdir(parents=True, exist_ok=True)
            SFTP.get(remote_item, str(local_item), callback=_progress(entry.filename))
            got += 1
    return got


def download(src, dst, excludes=None) -> int:
    """Pull one file or one tree. Returns how many files were fetched.

    Both failure modes below are raises rather than the reference's prints,
    and for the same reason: by the time a download runs, staging has already
    been emptied. "Remote source not found, skipped" followed by `=== DONE ===`
    and a cheerful "next, archive it" hint is the worst possible outcome --
    the previous batch is gone, nothing replaced it, and the transcript reads
    like a success.
    """
    excludes = excludes or []
    src_clean = str(src).rstrip("/\\")
    dst_path = Path(str(dst))

    try:
        info = SFTP.stat(src_clean)
    except IOError:
        # Show what the parent DOES hold. The realistic cause is a layout the
        # remote side has not caught up with -- asking for .../new/two_band when
        # the sweep was run in .../new -- and the answer is then already on
        # screen instead of costing another login.
        parent = posixpath.dirname(src_clean) or "."
        try:
            siblings = sorted(SFTP.listdir(parent))
            shown = ", ".join(siblings[:20]) + (f", … ({len(siblings)} total)"
                                                if len(siblings) > 20 else "")
            nearby = f"\n   🔎 {parent} holds: {shown or '(empty)'}"
        except IOError as exc:
            nearby = f"\n   🔎 its parent {parent} is not readable either ({exc})."
        raise SystemExit(
            f"🚨 download: remote source does not exist: {src_clean}{nearby}\n"
            f"   Nothing was downloaded, and staging has already been emptied for this "
            f"transfer.\n   If the sweep ran somewhere else, that is where it has to be "
            f"pulled from.")

    if statmod.S_ISDIR(info.st_mode):
        target = dst_path if str(src).endswith(("/", "\\")) else dst_path / Path(src_clean).name
        got = _download_tree(src_clean, target, excludes)
    else:
        target = dst_path / Path(src_clean).name if _local_is_dir_like(dst) else dst_path
        target.parent.mkdir(parents=True, exist_ok=True)
        SFTP.get(src_clean, str(target), callback=_progress(Path(src_clean).name))
        got = 1

    if got == 0:
        raise SystemExit(
            f"🚨 download: {src_clean} exists but nothing came back from it.\n"
            f"   Every entry was either empty or matched an exclude pattern. Staging "
            f"is now empty;\n   this is a failed download, not an up-to-date one.")
    return got


# ======================================================================================
# pre-download safety check
#
# Adapted from Superconductivity/python/hpc_transfer_utils.py -- the one
# safety feature that project's transfer layer has and this one did not:
# verify, before pulling data back, that the SLURM jobs which produced it
# actually finished cleanly.
# ======================================================================================

# SLURM job-ID-bearing filenames this project's own job scripts produce.
# Since the array-job change (2026-08-22) each element redirects its own
# output into its run directory as
# mpi_logs_<arrayjobid>_<index>.out/.err (Submitter.array_script()), and
# anything failing before that redirect lands in array_logs/%A_%a.out.
# Before that change it was mpi_logs_<jobid>.out/.err (#SBATCH --output=
# mpi_logs_%j.out), one number, no index.
# Not a general SLURM convention -- specific to how this project's job script
# names them -- so this list is what ties the check to what is written.
#
# The prefix is a separate constant because BOTH the find globs and the
# job-ID extraction below are built from it. They have to agree about where
# in the filename the number lives, and until 2026-08-25 they did not: the
# extraction was anchored to the _<n>.out SUFFIX, which in the two-number
# array format is the array TASK INDEX, not the job ID. Nothing failed --
# `sacct -j 0,1,...,206` is a perfectly valid query, it just returns 200-odd
# unrelated jobs from the beginning of the account's history and reports
# their long-dead TIMEOUTs as if they were this sweep's. A check that lies
# confidently is worse than no check, so the two are now derived from one
# literal and cannot drift apart again.
_JOB_LOG_PREFIX = "mpi_logs_"
_JOB_LOG_GLOBS = (f"{_JOB_LOG_PREFIX}*.out", f"{_JOB_LOG_PREFIX}*.err")


def check_remote_run_completion(remote_src: str):
    """Every SLURM job ID whose log files live under `remote_src`
    (recursively -- one mpi_logs_<id>.out/.err per dir_NNNNN, plus the
    driver job's own log at the top level), and whether each finished
    cleanly.

    ONE extra command on the ALREADY-OPEN session (find | sacct, piped
    remotely, not two round trips) -- consistent with this project's
    one-session-many-commands rule (see research_backup/setup/
    supercomputing_cluster/README.md), and cheap enough to run before every
    download without it being a reason to skip it.

    Returns (unclean, checked, queried) where `unclean` is a list of (jobid,
    state, elapsed) for every job NOT cleanly COMPLETED (RUNNING, PENDING,
    FAILED, TIMEOUT, CANCELLED, OUT_OF_MEMORY, ... -- anything whose State
    does not start with "COMPLETED", plus a COMPLETED job whose ExitCode was
    not 0:0, which State alone would not catch), `checked` is how many
    distinct job IDs were actually found and looked up, and `queried` is the
    list of job IDs derived from the log filenames. checked==0 means nothing
    here could be verified either way -- silence in `unclean` then means
    "unknown", not "confirmed complete", and is reported as such by the
    caller.
    """
    name_pattern = r"\( " + " -o ".join(f"-name '{g}'" for g in _JOB_LOG_GLOBS) + r" \)"
    # Take the FIRST number after the mpi_logs_ prefix -- the job ID in the
    # one-number pre-array format, the ARRAY JOB ID in the two-number one.
    # Anchoring on the prefix (rather than on the .out suffix) is what makes
    # both formats give the job and never the task index, and it still cannot
    # pick up digits from dir_NNNNN or the model name, which was the original
    # reason not to grep the whole path. `sacct -j <arrayjobid>` expands to
    # every element of the array on its own, so the index is not needed.
    cmd = (
        f"ids=$(find '{remote_src}' {name_pattern} 2>/dev/null "
        f"| grep -oE '{_JOB_LOG_PREFIX}[0-9]+' | grep -oE '[0-9]+' | sort -un | paste -sd,); "
        f"echo \"QUERIED:$ids\"; "
        f"[ -n \"$ids\" ] && sacct -j \"$ids\" --noheader --parsable2 "
        f"--format=JobID,State,Elapsed,ExitCode 2>/dev/null"
    )
    _rc, out = run_remote_capture(cmd)

    checked_ids, unclean, queried = set(), [], []
    for line in out.splitlines():
        # The QUERIED: line is echoed by the command above so the caller can
        # SHOW which job IDs were looked up. The 2026-08-25 misparse produced
        # a wholly plausible-looking warning about jobs 1-202 that had nothing
        # to do with the sweep; printing the IDs makes that kind of mistake
        # visible in the transcript instead of only in the conclusion.
        if line.startswith("QUERIED:"):
            queried = [i for i in line[len("QUERIED:"):].strip().split(",") if i]
            continue
        parts = line.split("|")
        if len(parts) != 4:
            continue
        jobid, state, elapsed, exitcode = parts
        if "." in jobid:          # a step row (98578.batch, 98578.0), not a job itself
            continue
        checked_ids.add(jobid)
        state_clean = state.strip().upper()
        code = exitcode.split(":")[0].strip()
        if not state_clean.startswith("COMPLETED") or code not in ("0", ""):
            unclean.append((jobid, state.strip(), elapsed.strip()))
    return unclean, len(checked_ids), queried


def _warn_if_run_incomplete(host: str, remote_src: str) -> None:
    """Printed loudly before a download, never blocking it: an incomplete
    run's data.csv is still inspectable and sometimes legitimately worth
    downloading on purpose. What must not happen is finding that out only
    after archiving it, quietly, with no idea the source job never finished.
    """
    try:
        unclean, checked, queried = check_remote_run_completion(remote_src)
    except Exception as e:
        print(f"⚠️  could not verify remote job status before download ({e}). "
              f"Proceeding without that check.", flush=True)
        return

    # Always say which jobs the verdict is about. A sweep is one array job, so
    # this is normally one ID, and one line; seeing a long list of small
    # numbers here means the log filenames are not being read as expected and
    # the verdict below is about somebody else's history, not this run.
    shown = ",".join(queried[:8]) + (f",… ({len(queried)} total)" if len(queried) > 8 else "")
    print(f"🔎 job IDs from log filenames under {remote_src}: {shown or '(none)'}", flush=True)

    if checked == 0:
        print(f"ℹ️  no SLURM job logs found under {host}:{remote_src} to verify "
              f"completion against — proceeding without that check (not the "
              f"same as confirmed complete).", flush=True)
        return

    if unclean:
        # Cap the listing but never silently: a 200-element array with a bad
        # node can put every element in here, and 200 lines of scrollback
        # buries the count that actually matters.
        LIMIT = 20
        print(f"\n{'!' * 78}", flush=True)
        print(f"⚠️  WARNING: downloading from {host}:{remote_src}", flush=True)
        print(f"    {len(unclean)} of {checked} job(s) found there did NOT "
              f"complete cleanly — this run may be INCOMPLETE:", flush=True)
        for jobid, state, elapsed in unclean[:LIMIT]:
            print(f"      job {jobid}: {state}  (ran {elapsed})", flush=True)
        if len(unclean) > LIMIT:
            print(f"      ... and {len(unclean) - LIMIT} more not shown "
                  f"(sacct -j {','.join(queried[:4])} for the full list)", flush=True)
        print(f"    The data you are about to download may be partial. "
              f"Proceeding anyway — check job state above before trusting "
              f"the result.", flush=True)
        print(f"{'!' * 78}\n", flush=True)
    else:
        print(f"✅  all {checked} job(s) found under {host}:{remote_src} "
              f"completed cleanly.", flush=True)


# ======================================================================================
# emptying a local directory
#
# Moved here from hpc_transfers.py: the guards belong with the rest of the
# mechanism, and one copy is the only copy. Adapted from Superconductivity's
# delete_directory_contents() -- same shape (resolve, refuse system paths, list
# before asking, require a typed word, chmod-retry on PermissionError), with the
# Windows half of its protected list dropped because this project is Linux-only.
# ======================================================================================

# $HOME is in here, and it is the one that earns its place: the realistic typo
# is a STAGING_ROOT that lost its last path component, not one that says /etc.
PROTECTED_DIRS = frozenset({
    Path("/"), Path("/home"), Path("/usr"), Path("/bin"), Path("/sbin"),
    Path("/lib"), Path("/lib64"), Path("/etc"), Path("/var"), Path("/boot"),
    Path("/dev"), Path("/proc"), Path("/sys"), Path("/opt"), Path("/root"),
    Path("/tmp"), Path.home(),
})


def _force_remove(func, path, exc):
    """shutil.rmtree error hook: a read-only directory is a permission problem,
    not a reason to leave half a tree behind. Superconductivity does the same
    chmod-and-retry inline. Written to satisfy both hook signatures --
    onexc(func, path, exc) on Python >= 3.12, onerror(func, path, exc_info)
    before it -- so this does not become a version-specific silent no-op."""
    if isinstance(exc, tuple):          # onerror passes sys.exc_info()
        exc = exc[1]
    if isinstance(exc, PermissionError):
        try:
            os.chmod(path, 0o700)
            func(path)
            return
        except Exception:
            pass
    raise exc


_RMTREE_HOOK = ({"onexc": _force_remove} if sys.version_info >= (3, 12)
                else {"onerror": _force_remove})


def delete_directory_contents(directory, *, label: str = "directory",
                              why: str = "", timeout: int = 30) -> int:
    """Empty `directory` -- its contents, not the directory itself -- after
    naming every entry and requiring the word DELETE to be typed.

    Returns how many entries were deleted; 0 if there was nothing to delete.
    Raises SystemExit on a decline, on a timeout, on a partial failure, and on
    every guard. `why` is printed inside the warning block, indented, so the
    caller can say what THIS directory is without the mechanism knowing.

    Two deliberate differences from the Superconductivity original. It names
    every entry before asking, which is what matters when the directory is
    shared between projects. And declining RAISES rather than returning: SC's
    version returns and lets its caller carry on, but here the caller is a
    download, and downloading a new batch on top of an old one is the exact
    stale-directories.txt hazard the wipe exists to prevent. A decline has to
    stop the transfer, not just the deletion.
    """
    target = Path(directory).resolve()

    if target in {p.resolve() for p in PROTECTED_DIRS if p.exists()} or target.parent == target:
        raise SystemExit(f"🚨 refusing to empty {target} -- protected directory. "
                         f"Check the {label} path.")

    # Deleting the directory you are standing in, or an ancestor of it, leaves
    # the process in a path that no longer exists. It also means the path being
    # emptied is source, not staging.
    cwd = Path.cwd().resolve()
    if cwd == target or target in cwd.parents:
        raise SystemExit(f"🚨 refusing to empty {target} -- it contains the current "
                         f"working directory {cwd}. Check the {label} path.")

    if not target.exists():
        print(f"ℹ️  {label} : {target} does not exist yet -- nothing to clear.", flush=True)
        return 0
    if not target.is_dir():
        raise SystemExit(f"🚨 {target} is not a directory, so its contents cannot be "
                         f"cleared. Check the {label} path.")

    doomed = sorted(target.iterdir())
    if not doomed:
        print(f"ℹ️  {label} : {target} is already empty.", flush=True)
        return 0

    print(f"\n{'!' * 78}", flush=True)
    print(f"⚠️  DELETE: the following {len(doomed)} item(s) in", flush=True)
    print(f"        {target}", flush=True)
    print(f"    will be PERMANENTLY DELETED:", flush=True)
    for p in doomed:
        is_dir = p.is_dir() and not p.is_symlink()
        try:
            count = f"   ({len(list(p.iterdir()))} entries)" if is_dir else ""
        except OSError as exc:
            # Unreadable is not skippable-in-silence: it is also what will make
            # the deletion below fail, and knowing that now beats finding out
            # after DELETE has been typed.
            print(f"      [DIR?] {p.name}   ⚠️  unreadable: {exc}", flush=True)
            continue
        print(f"      [{'DIR ' if is_dir else 'FILE'}] {p.name}{count}", flush=True)
    for line in why.splitlines():
        print(f"    {line}", flush=True)
    print(f"{'!' * 78}", flush=True)

    try:
        answer = timed_input(f"👉 Type DELETE to empty it, anything else to abort "
                             f"[{timeout}s timeout]: ", timeout)
    except TimeoutError:
        raise SystemExit(f"🚨 timeout with no answer -- aborted, {label} left untouched "
                         f"and nothing else was done.")
    if answer.strip() != "DELETE":
        raise SystemExit(f"aborted -- {label} left untouched, nothing else was done.")

    deleted, failed = 0, []
    for p in doomed:
        try:
            if p.is_symlink() or not p.is_dir():
                p.unlink()
            else:
                shutil.rmtree(p, **_RMTREE_HOOK)
            deleted += 1
        except Exception as exc:
            failed.append((p.name, exc))

    print(f"🧹 deleted {deleted} of {len(doomed)} item(s) from {target}", flush=True)
    if failed:
        for name, exc in failed:
            print(f"   ❌ {name}: {exc}", flush=True)
        raise SystemExit(
            f"🚨 {len(failed)} item(s) could not be deleted -- {target} is now HALF-cleared.\n"
            f"   That is the state the wipe exists to prevent (leftover directories beside a\n"
            f"   fresh listing), so nothing further is done. Fix the permissions and retry.")
    return deleted


# ======================================================================================
# driver
# ======================================================================================

_MODES = ("mkdir", "upload", "download")


def run_transfers(host: str, transfers) -> dict:
    """Execute a list of {"mode", "src", "dst", "excludes"} entries.

    "mkdir" creates a remote directory and moves nothing -- it is how a
    transfer plan can lay out a directory the user is about to `cd` into.
    Returns {"mkdir": n, "upload": n_files, "download": n_files}.
    """
    # Validate the WHOLE list before a single byte moves. Superconductivity
    # checks the mode inside its loop and prints `❌ Invalid mode` on reaching a
    # bad one, which means a typo in entry 12 is discovered only after entries
    # 1-11 have been sent: a half-applied plan that still ends in DONE. There
    # is nothing to gain by starting a plan already known to be malformed.
    for i, entry in enumerate(transfers, 1):
        mode = entry.get("mode")
        if mode not in _MODES:
            raise SystemExit(
                f"🚨 run_transfers: entry {i} of {len(transfers)} has mode {mode!r}.\n"
                f"   Valid modes are {', '.join(_MODES)}. Nothing was transferred: an\n"
                f"   unrecognised mode is a bug in the transfer list, and skipping it would\n"
                f"   apply part of a plan and then report success.")
        for key in (("dst",) if mode == "mkdir" else ("src", "dst")):
            if not entry.get(key):
                raise SystemExit(f"🚨 run_transfers: entry {i} ({mode}) has no {key!r}. "
                                 f"Nothing was transferred.")

    print("=== TRANSFER START ===")
    tally = {"mkdir": 0, "upload": 0, "download": 0}
    finished = False
    connect(host)
    try:
        # Pre-create upload destinations so a first-time directory does not
        # fail one file at a time.
        wanted = set()
        for entry in transfers:
            if entry["mode"] == "upload":
                dst = str(entry["dst"])
                path = PurePosixPath(dst.rstrip("/\\"))
                wanted.add(str(path) if dst.endswith(("/", "\\")) else str(path.parent))
        for directory in sorted(wanted):
            mkdir_p(directory)

        for entry in transfers:
            mode, dst = entry["mode"], str(entry["dst"])
            if mode == "mkdir":
                print(f"\n>>> {'MKDIR':8s} : {dst}")
                mkdir_p(dst)
                tally["mkdir"] += 1
                print(f"    ✅ ready", flush=True)
                continue

            src = str(entry["src"])
            print(f"\n>>> {mode.upper():8s} : {src} -> {dst}")
            if mode == "upload":
                moved = upload(src, dst, entry.get("excludes"))
                tally["upload"] += moved
            else:
                _warn_if_run_incomplete(host, src)
                moved = download(src, dst, entry.get("excludes"))
                tally["download"] += moved
            print(f"    ✅ {moved} file(s)", flush=True)
        finished = True
    finally:
        close()
        # Counts, and never a bare DONE: an upload that sent 0 files and one
        # that sent 300 used to print the same closing line, and so did a run
        # that raised half way through.
        moved = ", ".join(f"{v} {k}" for k, v in tally.items() if v) or "nothing moved"
        if finished:
            print(f"\n=== DONE — {moved} ===")
        else:
            print(f"\n=== ABORTED — {moved} before the failure above ===")
    return tally
