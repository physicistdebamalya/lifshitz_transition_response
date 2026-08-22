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

Not imported by workflow/submit_utils.py or the generators -- this only runs
on the workstation, driving hpc_transfers.py. A cluster login node has no
outbound reason to open an SFTP session to itself.
"""

from __future__ import annotations

import fnmatch
import getpass
import os
import posixpath
import signal
import stat as statmod
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
        answer = timed_input(f"Enter server name ({'/'.join(TRANSFER_HOSTS)}) [30s timeout]: ", timeout)
    except TimeoutError:
        raise SystemExit("Timeout. Cancelled.")
    host = answer.strip().lower()
    _check_host(host)
    return host


def ask_mode(timeout: int = 30) -> str:
    try:
        answer = timed_input("Enter operation (up/down) [30s timeout]: ", timeout)
    except TimeoutError:
        raise SystemExit("Timeout. Cancelled.")
    mode = answer.strip().lower()
    if mode not in ("up", "down"):
        raise SystemExit("Invalid input. Use 'up' or 'down'.")
    return mode


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
        except Exception:
            pass

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


def _upload_tree(local_src: Path, remote_root: str, excludes) -> None:
    for item in sorted(local_src.iterdir()):
        if _excluded(item.name, excludes):
            continue
        remote_item = posixpath.join(remote_root, item.name)
        if item.is_dir():
            mkdir_p(remote_item)
            _upload_tree(item, remote_item, excludes)
        else:
            mkdir_p(posixpath.dirname(remote_item))
            SFTP.put(str(item), remote_item, callback=_progress(item.name))


def upload(src, dst, excludes=None) -> None:
    src_path = Path(str(src))
    if not src_path.exists():
        print(f"missing local source, skipped: {src}")
        return

    excludes = excludes or []

    if src_path.is_dir():
        # A trailing separator means "the CONTENTS of this directory".
        contents_only = str(src).endswith(("/", "\\"))
        remote_root = str(dst).rstrip("/")
        if not contents_only:
            remote_root = posixpath.join(remote_root, src_path.name)
        mkdir_p(remote_root)
        _upload_tree(src_path, remote_root, excludes)
        return

    remote_dst = str(dst).rstrip("/\\")
    try:
        is_dir = statmod.S_ISDIR(SFTP.stat(remote_dst).st_mode)
    except IOError:
        is_dir = _remote_is_dir_like(dst)
    if is_dir:
        remote_dst = posixpath.join(remote_dst, src_path.name)

    mkdir_p(posixpath.dirname(remote_dst))
    SFTP.put(str(src_path), remote_dst, callback=_progress(src_path.name))


def _download_tree(remote_src: str, local_root, excludes) -> None:
    local_root = Path(local_root)
    local_root.mkdir(parents=True, exist_ok=True)
    for entry in SFTP.listdir_attr(remote_src):
        if _excluded(entry.filename, excludes):
            continue
        remote_item = posixpath.join(remote_src, entry.filename)
        local_item = local_root / entry.filename
        if statmod.S_ISDIR(entry.st_mode):
            _download_tree(remote_item, local_item, excludes)
        else:
            local_item.parent.mkdir(parents=True, exist_ok=True)
            SFTP.get(remote_item, str(local_item), callback=_progress(entry.filename))


def download(src, dst, excludes=None) -> None:
    excludes = excludes or []
    src_clean = str(src).rstrip("/\\")
    dst_path = Path(str(dst))

    try:
        info = SFTP.stat(src_clean)
    except IOError:
        print(f"remote source not found, skipped: {src_clean}")
        return

    if statmod.S_ISDIR(info.st_mode):
        target = dst_path if str(src).endswith(("/", "\\")) else dst_path / Path(src_clean).name
        _download_tree(src_clean, target, excludes)
    else:
        target = dst_path / Path(src_clean).name if _local_is_dir_like(dst) else dst_path
        target.parent.mkdir(parents=True, exist_ok=True)
        SFTP.get(src_clean, str(target), callback=_progress(Path(src_clean).name))


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
# Not a general SLURM convention -- specific to how this project's job script
# names them -- so this list is what ties the check to what is written.
_JOB_LOG_GLOBS = ("mpi_logs_*.out", "mpi_logs_*.err")


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

    Returns (unclean, checked) where `unclean` is a list of (jobid, state,
    elapsed) for every job NOT cleanly COMPLETED (RUNNING, PENDING, FAILED,
    TIMEOUT, CANCELLED, OUT_OF_MEMORY, ... -- anything whose State does not
    start with "COMPLETED", plus a COMPLETED job whose ExitCode was not 0:0,
    which State alone would not catch) and `checked` is how many distinct
    job IDs were actually found and looked up. checked==0 means nothing here
    could be verified either way -- silence in `unclean` then means
    "unknown", not "confirmed complete", and is reported as such by the
    caller.
    """
    name_pattern = r"\( " + " -o ".join(f"-name '{g}'" for g in _JOB_LOG_GLOBS) + r" \)"
    # Extract the job ID from the FILENAME'S OWN _<id>.out/.err suffix only,
    # anchored to end of line -- never from the whole path. A bare
    # `[0-9]+` over the full path would also match digits inside dir_NNNNN
    # or the model name, neither of which is a job ID.
    cmd = (
        f"ids=$(find '{remote_src}' {name_pattern} 2>/dev/null "
        f"| grep -oE '_[0-9]+\\.(out|err)$' | grep -oE '[0-9]+' | sort -un | paste -sd,); "
        f"[ -n \"$ids\" ] && sacct -j \"$ids\" --noheader --parsable2 "
        f"--format=JobID,State,Elapsed,ExitCode 2>/dev/null"
    )
    _rc, out = run_remote_capture(cmd)

    checked_ids, unclean = set(), []
    for line in out.splitlines():
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
    return unclean, len(checked_ids)


def _warn_if_run_incomplete(host: str, remote_src: str) -> None:
    """Printed loudly before a download, never blocking it: an incomplete
    run's data.csv is still inspectable and sometimes legitimately worth
    downloading on purpose. What must not happen is finding that out only
    after archiving it, quietly, with no idea the source job never finished.
    """
    try:
        unclean, checked = check_remote_run_completion(remote_src)
    except Exception as e:
        print(f"⚠️  could not verify remote job status before download ({e}). "
              f"Proceeding without that check.", flush=True)
        return

    if checked == 0:
        print(f"ℹ️  no SLURM job logs found under {host}:{remote_src} to verify "
              f"completion against — proceeding without that check (not the "
              f"same as confirmed complete).", flush=True)
        return

    if unclean:
        print(f"\n{'!' * 78}", flush=True)
        print(f"⚠️  WARNING: downloading from {host}:{remote_src}", flush=True)
        print(f"    {len(unclean)} of {checked} job(s) found there did NOT "
              f"complete cleanly — this run may be INCOMPLETE:", flush=True)
        for jobid, state, elapsed in unclean:
            print(f"      job {jobid}: {state}  (ran {elapsed})", flush=True)
        print(f"    The data you are about to download may be partial. "
              f"Proceeding anyway — check job state above before trusting "
              f"the result.", flush=True)
        print(f"{'!' * 78}\n", flush=True)
    else:
        print(f"✅  all {checked} job(s) found under {host}:{remote_src} "
              f"completed cleanly.", flush=True)


# ======================================================================================
# driver
# ======================================================================================

def run_transfers(host: str, transfers) -> None:
    """Execute a list of {"mode", "src", "dst", "excludes"} entries."""
    print("=== TRANSFER START ===")
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
            mode, src, dst = entry["mode"], str(entry["src"]), str(entry["dst"])
            print(f"\n>>> {mode.upper()} : {src} -> {dst}")
            if mode == "upload":
                upload(src, dst, entry.get("excludes"))
            elif mode == "download":
                _warn_if_run_incomplete(host, src)
                download(src, dst, entry.get("excludes"))
            else:
                print(f"invalid mode '{mode}'")
    finally:
        close()
        print("\n=== DONE ===")
