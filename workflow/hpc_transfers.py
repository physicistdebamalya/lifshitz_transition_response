#!/usr/bin/env python3
"""Move this project between the workstation and PARAM/HPC. Never run on a cluster.

    python3 workflow/hpc_transfers.py                    # prompts for host, direction, model
    python3 workflow/hpc_transfers.py param up
    python3 workflow/hpc_transfers.py param down two_band
    python3 workflow/hpc_transfers.py hpc  down four_band

The transfer MECHANISM is all in hpc_transfer_utils.py -- one authenticated
SSH+SFTP session reused for every file, recursive copy with excludes, the
pre-download SLURM completion check, and the staging wipe with its guards. What
lives here is only WHAT moves and WHERE: two declarative lists, the prompts
around them, and the check that the batch arrived usable.

    UP    both models' sources + both archive registries, model-agnostic
          (transfers_up: why the registry is not optional, and why build/ never
          travels)
    DOWN  ONE model's runs, from that model's own remote directory
          (transfers_down)

    ~/projects/new/                     REMOTE_PROJECT_SUBDIR
        physics/ workflow/ analysis/    pushed by `up`
        archive/<model>_model/logfile.txt
        two_band/  four_band/           one model's runs

Run directories are created in <repo>/<model>/, which
submit_utils.run_root_for() derives from the generator's own location -- not
from the cwd, and not from a per-host table of absolute paths. One directory
per model is what makes `down <model>` a well-defined question: share one
directory and both models' dir_NNNNN sit under a single directories.txt, which
no download can separate afterwards. It WAS Path.cwd(), and that is how 96 run
directories landed in workflow/ on hpc, under exactly such a directories.txt.
`up` still mkdir's both directories so a fresh clone is laid out; the generator
now creates its own if they are missing, from wherever it is invoked.

    ~/temp_files_data                   ONE FLAT DIRECTORY, ONE batch

Superconductivity's LOCAL_DESTINATION shape, and the shape
data_utils.sync_simulation_data() expects: directories.txt and every dir_NNNNN
it names, side by side. DOWN clears it first (clear_staging) and verifies what
arrived (verify_staged_batch), because sync_simulation_data() walks
directories.txt and copies every dir_NNNNN it names -- so one stale listing
from a previous batch silently readmits that batch, or names directories whose
contents no longer match. Excluding directories.txt from the download does not
help: the exclude stops a stale file being overwritten, not being read
afterwards. Only a wipe does, and the gate on it is not removable
(research_code_development_philosophy.md Part I: friction in the data path is a
feature). ~/temp_files_data is shared with Superconductivity, which stages and
wipes there the same way -- hence the listing before the deletion.

The flatness is load-bearing and was learned the hard way; see
workflow/tests/test_staging_wipe.py, which pins it and tells that story.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))

from data_utils import clean_path                             # noqa: E402
from models import MODELS                                     # noqa: E402
from submit_utils import (ARCHIVE_SUBDIR, WORKSTATION_ARCHIVE,  # noqa: E402
                          archive_dirname)
import hpc_transfer_utils as T                                # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
PHYSICS_DIR = REPO_ROOT / "physics"
WORKFLOW_DIR = REPO_ROOT / "workflow"
ANALYSIS_DIR = REPO_ROOT / "analysis"

# Where this repository lives under $HOME on each cluster, mirroring
# physics/Makefile's own per-host layout. Confirmed in practice, not assumed:
# ~/projects/new/{physics,workflow} exist and build on param as of 2026-08-21
# (research_backup/setup/supercomputing_cluster/param/), and the same layout was
# exercised on hpc in the sessions before that.
REMOTE_PROJECT_SUBDIR = "projects/new"

# Where the uploaded registries land, relative to remote_root(). Imported
# rather than re-declared: submit_utils.default_archive() is what READS
# <repo>/archive/<model>_model on the cluster, and if these two ever named
# different subdirectories a push would put the registry somewhere
# already_exists() does not look -- which fails the expensive way, by
# resubmitting an entire archived sweep.
REMOTE_ARCHIVE_SUBDIR = ARCHIVE_SUBDIR

STAGING_ROOT = clean_path("/home/debamalya/temp_files_data")

# Read-only. This is the tree ~/.claude/CLAUDE.md and the philosophy's §20
# declare off limits for writing; every use of it below is a read (an upload
# source, or a file count), never a destination. Imported for the same reason
# as above -- the generators resolve this same root through
# submit_utils.default_archive() when they run on the workstation.
ARCHIVE_ROOT = WORKSTATION_ARCHIVE

# ── what never moves ──────────────────────────────────────────────────────
# *.cache.npz is not hypothetical: submit_utils_Phase.load_history() in the
# Superconductivity tree memoises CSV parsing into a hidden sidecar next to its
# input, so a helper that merely READS an archive path leaves a file beside it.
# Nothing in this project writes one today; the exclude is here so that if
# anything ever does, it cannot ride an upload onto a cluster.
UPLOAD_EXCLUDES = ["*.o", "*.pyc", "__pycache__", "build", "*.ipynb", ".ipynb_checkpoints",
                   "*.cache.npz"]

# Never pull back what was pushed up, or build products, or a previous
# download's bookkeeping.
DOWNLOAD_EXCLUDES = [
    "*.o", "build", "physics", "workflow", "analysis",
    "__pycache__", "*.pyc", "Makefile", ".submit_lock",
    # Pushed up by this same script (REMOTE_ARCHIVE_SUBDIR): copies of the
    # workstation's own logfile.txt. Pulling them back would drop a stray
    # logfile.txt into the staging root beside the batch, where the only
    # logfile.txt that means anything is the archive's.
    "archive",
    # A build product, ~90-105 KB, now in EVERY run directory because each gets
    # its own copy of the binary. task.csv is deliberately NOT excluded: it is
    # small, and it is the input-side provenance that the per-directory
    # rewritten <model>.cpp used to be.
    "exe_run",
]

ANALYSIS_FILES = ["models.py", "snippets.py", "data_utils.py",
                  "loaders.py", "observables.py", "plotstyle.py"]
WORKFLOW_FILES = ["hpc_config.py", "submit_utils.py", "task_schema.py",
                  "generator_two_band.py", "generator_four_band.py"]


# ======================================================================================
# paths
# ======================================================================================

def remote_root(host: str) -> str:
    """The remote project tree -- source, registries, and the per-model run
    directories underneath it."""
    return str(T.base_path(host) / REMOTE_PROJECT_SUBDIR)


def remote_model_root(host: str, model: str) -> str:
    """The remote directory ONE model's sweep runs in, and therefore the one a
    `down <model>` reads. The generator is invoked here, so this is where its
    dir_NNNNN and directories.txt are created."""
    return f"{remote_root(host)}/{require_model(model)}"


def archive_dir(model: str) -> Path:
    """The workstation archive directory for a model. READ ONLY -- see
    ARCHIVE_ROOT. Returned for its logfile.txt and for counting runs."""
    return ARCHIVE_ROOT / archive_dirname(require_model(model))


def require_model(model, what: str = "") -> str:
    """One place that decides whether a model name is real, so every rejection
    reads the same and none of them is a warning-and-continue. A model that is
    not in MODELS cannot name a remote directory, an archive, or a System
    string, so there is nothing further to try."""
    if model in MODELS:
        return model
    where = f" for {what}" if what else ""
    raise SystemExit(f"🚨 unknown model {model!r}{where}. Use one of {sorted(MODELS)}.")


def clear_staging(target=None) -> int:
    """Empty the staging root before a download.

    A thin wrapper on purpose: the mechanism and every guard are in
    hpc_transfer_utils.delete_directory_contents(). What stays here is the
    wording, which is about THIS root and nothing else.
    """
    return T.delete_directory_contents(
        STAGING_ROOT if target is None else target,
        label="staging",
        why=("This root is shared with other projects -- check the list above.\n"
             "Staging is disposable: anything already archived is safe, and anything\n"
             "not yet archived would have to be downloaded again."),
    )


# ======================================================================================
# transfer lists
# ======================================================================================

def transfers_up(host: str) -> list[dict]:
    """Everything the cluster needs to build and submit either model, plus the
    per-model run directories the generators are invoked from.

    Model-agnostic on purpose: physics/models/*.cpp carries both models and the
    per-point task.csv selects which binary runs, so there is nothing to select
    at push time. physics/build/ never travels -- a -march=native object built
    here must not be linked there, which is exactly what physics/Makefile's
    per-host build/$(HOST)/ split already prevents locally, and UPLOAD_EXCLUDES
    keeps that true across the wire.
    """
    destination = remote_root(host)

    entries = [
        # Laid out before anything is sent, so the `cd` in the closing hint
        # cannot fail on a fresh clone. mkdir is idempotent and moves nothing.
        *({"mode": "mkdir", "dst": remote_model_root(host, model)} for model in sorted(MODELS)),
        {"mode": "upload", 
         "src": PHYSICS_DIR / "include", 
         "dst": f"{destination}/physics",
         "excludes": UPLOAD_EXCLUDES},

        {"mode": "upload", 
         "src": PHYSICS_DIR / "src", 
         "dst": f"{destination}/physics",
         "excludes": UPLOAD_EXCLUDES},

        {"mode": "upload", 
         "src": PHYSICS_DIR / "models", 
         "dst": f"{destination}/physics",
         "excludes": UPLOAD_EXCLUDES},

        {"mode": "upload",
         "src": PHYSICS_DIR / "Makefile", 
         "dst": f"{destination}/physics/Makefile"},
        # host.mk is NOT optional: physics/Makefile includes it unconditionally
        # (line 16), so without it `make` fails immediately on a fresh cluster
        # tree. It was missing from this list from the moment it was extracted
        # -- the clusters only had it because it was copied there by hand.
        {"mode": "upload", 
         "src": PHYSICS_DIR / "host.mk", 
         "dst": f"{destination}/physics/host.mk"},
    ]

    for name in WORKFLOW_FILES:
        entries.append({"mode": "upload", "src": require_local(WORKFLOW_DIR / name),
                        "dst": f"{destination}/workflow/{name}"})

    for name in ANALYSIS_FILES:
        entries.append({"mode": "upload", "src": require_local(ANALYSIS_DIR / name),
                        "dst": f"{destination}/analysis/{name}"})

    # Every model's registry, so already_exists() can skip archived points on
    # the cluster instead of resubmitting them. submit_utils.already_exists()
    # matches a candidate point against that file, and looks for it at
    # generator_*.py's ARCHIVE -- a path under ~/ComputationData that exists on
    # the workstation and nowhere else. Run the generator on a cluster with no
    # registry there and it says so (submit_utils.py:264) and then submits EVERY
    # point as new: for four_band that is all 23 points of a sweep whose 23
    # points are already archived, days of quota spent recomputing data already
    # on disk. Two small text files (36 KB + 23 KB) close that hole, and because
    # they are what the check reads they go up on every push, not once.
    #
    # A MISSING registry is not fatal -- a model with nothing archived yet has
    # none, and the generator warns once and treats every point as new -- but it
    # is never silent, because pushing no registry looks exactly like pushing
    # one and is precisely how the 23/23 case wastes a sweep.
    for model in sorted(MODELS):
        registry = archive_dir(model) / "logfile.txt"
        if registry.exists():
            entries.append({"mode": "upload", "src": registry,
                            "dst": f"{destination}/{REMOTE_ARCHIVE_SUBDIR}/{model}_model/logfile.txt"})
        else:
            print(f"⚠️  transfers_up: no registry at {registry} -- "
                  f"a {model} sweep run on the cluster will treat every point as new.",
                  flush=True)

    return entries


def require_local(path: Path) -> Path:
    """A source named in the push plan and absent on disk is a bug in the plan.

    This used to be `if path.exists()`, which quietly shortened the upload. That
    is how host.mk went missing for weeks: the transfer reported success and
    `make` failed on the cluster minutes later, with nothing connecting the two.
    ANALYSIS_FILES and WORKFLOW_FILES are constants in this file, so a name in
    them that is not on disk means the constant is stale or the checkout is
    incomplete -- both worth stopping for, neither worth discovering remotely.
    """
    if not path.exists():
        raise SystemExit(
            f"🚨 transfers_up: {path} is named in this script's file list but does not "
            f"exist.\n   Nothing was uploaded. Either the file was renamed (fix "
            f"WORKFLOW_FILES / ANALYSIS_FILES\n   above) or this checkout is incomplete -- "
            f"an upload missing a source is how a\n   cluster-side `make` or import fails "
            f"minutes later for no visible reason.")
    return path


def transfers_down(host: str, model: str) -> list[dict]:
    """Pull ONE model's run output into the flat staging root, minus the source
    tree that was pushed up and build noise.

    The trailing "/" is load-bearing: it means "the CONTENTS of that remote
    directory", so dir_NNNNN and directories.txt land directly in the flat
    staging root rather than inside a <model>/ level that the wipe -- and
    sync_simulation_data() -- would then both look straight past.
    """
    return [
        {"mode": "download", "src": f"{remote_model_root(host, model)}/",
         "dst": STAGING_ROOT, "excludes": DOWNLOAD_EXCLUDES}
    ]


# ======================================================================================
# post-download verification
# ======================================================================================

def verify_staged_batch(model: str) -> int:
    """Assert the staged batch is usable, and fail here if it is not.

    A download that brings back a couple of stray files and no directories.txt
    is indistinguishable, in the transcript, from a good one: both end in DONE
    followed by "next, archive it". The notebook then fails much later, by which
    point the wipe has already discarded whatever staging held before. So the
    batch is inspected while the cause is still on screen.

    Checked, in the order the archive step will read them: directories.txt
    exists and parses; every dir_NNNNN it names exists; each carries the
    parameters.txt and data.csv that data_utils.main_copy() requires; and line 1
    of each parameters.txt is this model's System string. That last one is what
    catches the mistake the per-model download introduces -- asking for two_band
    and being handed four_band runs -- and it is the same rule main_copy()
    enforces at line 536, just applied before the wipe's cost is sunk rather
    than half way through an archive run.

    Returns the number of run directories verified.
    """
    print(f"\n🔎 verifying the staged batch in {STAGING_ROOT}", flush=True)

    listing = STAGING_ROOT / "directories.txt"
    if not listing.exists():
        raise SystemExit(
            f"🚨 no directories.txt in {STAGING_ROOT} after the download.\n"
            f"   That file is what data_utils.sync_simulation_data() walks, so this batch "
            f"cannot be\n   archived at all. Files did come back, so the remote directory "
            f"was reachable -- it\n   just does not look like a completed sweep. Check that "
            f"the generator was run in\n   .../{model} on the cluster.")

    names, malformed = [], []
    for lineno, line in enumerate(listing.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        # Exactly sync_simulation_data()'s own parse (data_utils.py:625), not a
        # second reading of the format: `serial. dir_NNNNN`. A line that fails
        # here raises a bare ValueError there, mid-archive.
        if ". " not in line:
            malformed.append((lineno, line))
            continue
        names.append(line.split(". ", 1)[1].strip())

    if malformed:
        for lineno, line in malformed[:10]:
            print(f"   ❌ directories.txt:{lineno} is not 'N. dir_NNNNN': {line!r}", flush=True)
        raise SystemExit(f"🚨 {len(malformed)} unparseable line(s) in {listing}. "
                         f"sync_simulation_data() splits on '. ' and would raise on these.")
    if not names:
        raise SystemExit(f"🚨 {listing} is empty -- the download produced a listing naming "
                         f"no runs.\n   This is a failed download, not an empty sweep.")

    system = MODELS[require_model(model)].system
    missing, incomplete, wrong_system = [], [], []
    for name in names:
        run = STAGING_ROOT / name
        if not run.is_dir():
            missing.append(name)
            continue
        absent = [f for f in ("parameters.txt", "data.csv") if not (run / f).exists()]
        if absent:
            incomplete.append((name, absent))
            continue
        first = (run / "parameters.txt").read_text().strip().split("\n")[0].strip()
        if first != system:
            wrong_system.append((name, first))

    for label, bad, detail in (
        ("named in directories.txt but not downloaded", missing, lambda x: ""),
        ("downloaded without the files the archive step reads", incomplete,
         lambda x: f" (no {', '.join(x[1])})"),
        (f"not {model} data — line 1 of parameters.txt is not {system!r}", wrong_system,
         lambda x: f" (found {x[1]!r})"),
    ):
        if not bad:
            continue
        print(f"   ❌ {len(bad)} run(s) {label}:", flush=True)
        for item in bad[:10]:
            name = item if isinstance(item, str) else item[0]
            print(f"        {name}{detail(item)}", flush=True)
        if len(bad) > 10:
            print(f"        ... and {len(bad) - 10} more not shown", flush=True)

    if missing or incomplete or wrong_system:
        raise SystemExit(
            f"🚨 the staged batch is not archivable: {len(names)} run(s) listed, "
            f"{len(missing)} absent,\n   {len(incomplete)} incomplete, "
            f"{len(wrong_system)} from a different model. Nothing was archived.\n"
            f"   Fix the source on the cluster and download again -- do NOT archive a "
            f"batch that\n   only partly arrived, because the runs that did arrive would "
            f"get IDs and look\n   complete afterwards.")

    quarantined = STAGING_ROOT / "incomplete_runs"
    if quarantined.is_dir() and any(quarantined.iterdir()):
        n = len(list(quarantined.iterdir()))
        print(f"   ⚠️  {n} quarantined point(s) in {quarantined} -- these failed to "
              f"PREPARE on the\n       cluster and are not in directories.txt, so the "
              f"archive step will not see them.", flush=True)

    # Staging was emptied before this download, so everything here arrived with
    # it. A run directory on disk that the listing does not name therefore means
    # the listing is not the whole batch, and the archive step walks the listing.
    unlisted = sorted(p.name for p in STAGING_ROOT.glob("dir_*")
                      if p.is_dir() and p.name not in set(names))
    if unlisted:
        shown = ", ".join(unlisted[:10]) + (f", … ({len(unlisted)} total)"
                                           if len(unlisted) > 10 else "")
        print(f"   ⚠️  {len(unlisted)} downloaded run(s) are NOT named in directories.txt "
              f"and will\n       not be archived: {shown}", flush=True)

    print(f"   ✅ {len(names)} run(s), all present, all carrying parameters.txt + "
          f"data.csv, all {model}.", flush=True)
    return len(names)


# ======================================================================================
# entry point
# ======================================================================================

def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    host = argv[0] if argv else T.ask_host()
    if host not in T.TRANSFER_HOSTS:
        raise SystemExit(f"🚨 unknown host '{host}'. Use one of {T.TRANSFER_HOSTS}.")

    mode = argv[1] if len(argv) > 1 else T.ask_mode()
    if mode not in ("up", "down"):
        raise SystemExit(f"🚨 invalid operation '{mode}'. Use 'up' or 'down'.")

    if mode == "down":
        # Required, and asked for if not given. The model IS the remote
        # directory being read, so there is no default that could be right and
        # nothing to skip -- an earlier version made it optional and used it
        # only to fill in a printed path, which is what made the prompt
        # unreadable ("only used to print the archive path afterwards").
        model = require_model(
            argv[2] if len(argv) > 2 else T.ask_choice(
                "Model to download", sorted(MODELS),
                note=("This picks the REMOTE DIRECTORY the runs are read from:\n"
                      f"        {remote_root(host)}/<model>\n"
                      "    and the System string the batch is checked against.")),
            "download")
    else:
        # Upload is model-agnostic (see transfers_up), so a third argument here
        # cannot do anything. Said out loud rather than accepted and dropped: a
        # `up two_band` that silently behaves as `up` teaches that the argument
        # works.
        if len(argv) > 2:
            print(f"ℹ️  '{argv[2]}' ignored: an upload carries both models' sources and "
                  f"both registries.\n    Nothing is model-specific at push time.", flush=True)
            require_model(argv[2], "upload")
        model = None

    # One banner at the end whatever happens, naming mode, model and host, so a
    # transfer's outcome is greppable in a scrollback. Deliberately try/finally
    # and not try/except: the banner is additional to a failure, never instead
    # of it -- an exception keeps its traceback and its exit status.
    label = model or "both models"
    try:
        return _run(host, mode, model)
    finally:
        print(f"\n=== *{mode.upper()}* end for *{label}* on *{host}* ===", flush=True)


def _run(host: str, mode: str, model) -> int:
    transfers = transfers_up(host) if mode == "up" else transfers_down(host, model)

    print(f"\n=== {mode.upper()} :: {host} ===")
    print(f"remote  : {remote_root(host)}")

    if mode == "up":
        # Count what the archive holds, so a push that is about to let the
        # cluster skip duplicates says so, and one that will not says why.
        for m in sorted(MODELS):
            d = archive_dir(m)
            held = len(list(d.glob("*.csv"))) if d.is_dir() else 0
            reg = (d / "logfile.txt").exists()
            print(f"archive : {m:10s} {held:3d} run(s) via {d}"
                  + ("" if reg else "   ⚠️  NO logfile.txt -- points will not be skipped"))
        for m in sorted(MODELS):
            print(f"runs in : {remote_model_root(host, m)}   (created if absent)")
    else:
        print(f"source  : {remote_model_root(host, model)}/   ({model} runs only)")
        print(f"staging : {STAGING_ROOT}   (flat, one batch, cleared before download)")

    print(f"entries : {len(transfers)}")
    if input("👉 Proceed? [y/N]: ").strip().lower() != "y":
        raise SystemExit("aborted.")

    # After the Proceed gate, never before: a download cancelled there is not a
    # download, and must not have cost the previous batch. clear_staging() then
    # asks again on its own account, so emptying staging takes two answers.
    if mode == "down":
        clear_staging()

    T.run_transfers(host, transfers)

    if mode == "down":
        verify_staged_batch(model)
        _hint_after_download(model)
    else:
        _hint_after_upload(host)
    return 0


def _hint_after_download(model: str) -> None:
    print(
        f"\n👉 Next: verify and archive with the notebook cell 'bulk copy paste local'\n"
        f"   (analysis/notebooks/test.ipynb), or with\n"
        f"       python3 -c \"import sys; sys.path.insert(0,'analysis'); "
        f"import data_utils as d; d.sync_simulation_data("
        f"'{STAGING_ROOT}', '{archive_dir(model)}', '{MODELS[model].system}')\"\n"
        f"   Staging is not cleared afterwards -- the NEXT download clears it, which is\n"
        f"   also the deadline for archiving this batch.",
        flush=True)


def _hint_after_upload(host: str) -> None:
    """One command per model, and nothing to remember.

    This used to spell out a `cd` into the model's directory and an
    `--archive` pointing at the registry pushed seconds earlier, then explain
    both at length. Neither is needed now that the generator derives its run
    root and its registry from its own location -- and a closing hint that
    lists arguments you do not need is a hint that gets pasted with them.
    """
    print(f"\n👉 Next, on the {host} login node:\n\n"
          + "".join(f"       python3 {remote_root(host)}/workflow/"
                    f"generator_{m}.py\n" for m in sorted(MODELS))
          + f"\n   Each one confirms before it submits, builds its binary on the "
            f"login node,\n   puts one run directory per new point in\n"
            f"       {remote_root(host)}/<model>/\n"
            f"   and submits the sweep as a single SLURM array.\n"
            f"   Watch: squeue -u {T.base_path(host).name}",
          flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
