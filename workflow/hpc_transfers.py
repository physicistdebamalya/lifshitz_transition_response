#!/usr/bin/env python3
# hpc_transfers.py -- run this on the workstation, never on a cluster.
#
#     python3 workflow/hpc_transfers.py                     # prompts for host, direction, model
#     python3 workflow/hpc_transfers.py param up two_band
#     python3 workflow/hpc_transfers.py param down two_band
#     python3 workflow/hpc_transfers.py hpc up four_band
#
# Adapted from two_component_BH/workflow/hpc_transfers.py: same declarative
# transfers_up()/transfers_down() shape, same exclude conventions, same
# one-authentication-per-session behaviour (hpc_transfer_utils.connect()
# opens a single SSH+SFTP session, reused for every file). What changed is
# WHAT moves and where it goes -- this project has one model tree, not
# per-stage archives, and its data locations are the three CLAUDE.md
# documents (see "Data locations" there), not two_component_BH's staging
# layout.
#
# UPLOAD carries physics/{include,src,models}, physics/Makefile + host.mk,
# workflow/*.py and analysis/*.py (the generators import from analysis/, so it
# has to exist wherever the generator runs) read-only onto the cluster.
# It never writes into a local directory and never uploads physics/build/ --
# a -march=native object built on the workstation must never be linked on a
# cluster, which is exactly what physics/Makefile's per-host build/$(HOST)/
# split already exists to prevent locally.
#
# DOWNLOAD pulls the remote project directory's run output -- every dir_NNNNN/
# (data.csv, parameters.txt, task.csv, mpi_logs_*.out/err), directories.txt,
# and incomplete_runs/ -- back into local staging at
# ~/temp_files_data/<model>_model/, which is exactly the directory shape
# analysis/data_utils.sync_simulation_data() expects: directories.txt and
# every dir_NNNNN it names, side by side, at the same level.
#
# REMOTE_PROJECT_SUBDIR below is where this repository lives under $HOME on
# each cluster, mirroring physics/Makefile's own per-host layout. Confirmed
# in practice on both hosts (not only assumed): ~/projects/new/physics and
# ~/projects/new/workflow exist and build correctly on param as of
# 2026-08-21 (see research_backup/setup/supercomputing_cluster/param/), and
# the same layout was exercised on hpc in the sessions before that.

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))

from data_utils import clean_path                             # noqa: E402
from models import MODELS                                     # noqa: E402
import hpc_transfer_utils as T                                 # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
PHYSICS_DIR = REPO_ROOT / "physics"
WORKFLOW_DIR = REPO_ROOT / "workflow"
ANALYSIS_DIR = REPO_ROOT / "analysis"

# See the note above.
REMOTE_PROJECT_SUBDIR = "projects/new"

STAGING_ROOT = clean_path("/home/debamalya/temp_files_data")

# ── what never moves ──────────────────────────────────────────────────────
UPLOAD_EXCLUDES = ["*.o", "*.pyc", "__pycache__", "build", "*.ipynb", ".ipynb_checkpoints"]

# Never pull back what was pushed up, or build products, or the archive
# staging area's own bookkeeping from a previous download.
DOWNLOAD_EXCLUDES = [
    "*.o", "build", "physics", "workflow", "analysis",
    "__pycache__", "*.pyc", "Makefile", ".submit_lock",
    # A build product, ~90-105 KB, now sitting in EVERY run directory because
    # each gets its own copy of the binary. task.csv is deliberately NOT
    # excluded: it is small, and it is the input-side provenance that the
    # per-directory rewritten <model>.cpp used to be.
    "exe_run",
]

ANALYSIS_FILES = ["models.py", "snippets.py", "data_utils.py",
                  "loaders.py", "observables.py", "plotstyle.py"]
WORKFLOW_FILES = ["hpc_config.py", "submit_utils.py", "task_schema.py",
                  "generator_two_band.py", "generator_four_band.py"]


def remote_root(host: str) -> str:
    return str(T.base_path(host) / REMOTE_PROJECT_SUBDIR)


def staging_dir(model: str) -> Path:
    if model not in MODELS:
        raise SystemExit(f"unknown model '{model}'. Use one of {sorted(MODELS)}.")
    return STAGING_ROOT / f"{model}_model"


# ======================================================================================
# transfer lists
# ======================================================================================

def transfers_up(host: str) -> list[dict]:
    """Everything the cluster needs to build and submit either model. Not
    model-specific -- physics/models/*.cpp carries both models, and the
    per-point task.csv selects which binary runs; nothing to select at push time.
    """
    destination = remote_root(host)

    entries = [
        {"mode": "upload", "src": PHYSICS_DIR / "include", "dst": f"{destination}/physics",
         "excludes": UPLOAD_EXCLUDES},
        {"mode": "upload", "src": PHYSICS_DIR / "src", "dst": f"{destination}/physics",
         "excludes": UPLOAD_EXCLUDES},
        {"mode": "upload", "src": PHYSICS_DIR / "models", "dst": f"{destination}/physics",
         "excludes": UPLOAD_EXCLUDES},
        {"mode": "upload", "src": PHYSICS_DIR / "Makefile", "dst": f"{destination}/physics/Makefile"},
        # host.mk is NOT optional: physics/Makefile includes it unconditionally
        # (line 16), so without it `make` fails immediately on a fresh cluster
        # tree. It was missing from this list from the moment it was extracted
        # -- the clusters only had it because it was copied there by hand.
        {"mode": "upload", "src": PHYSICS_DIR / "host.mk", "dst": f"{destination}/physics/host.mk"},
    ]

    for name in WORKFLOW_FILES:
        path = WORKFLOW_DIR / name
        if path.exists():
            entries.append({"mode": "upload", "src": path, "dst": f"{destination}/workflow/{name}"})

    for name in ANALYSIS_FILES:
        path = ANALYSIS_DIR / name
        if path.exists():
            entries.append({"mode": "upload", "src": path, "dst": f"{destination}/analysis/{name}"})

    return entries


def transfers_down(host: str, model: str) -> list[dict]:
    """Pull the whole remote project directory's run output into staging,
    minus the source tree that was pushed up and build noise."""
    return [
        {"mode": "download", "src": f"{remote_root(host)}/", "dst": staging_dir(model),
         "excludes": DOWNLOAD_EXCLUDES}
    ]


# ======================================================================================
# entry point
# ======================================================================================

def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    host = argv[0] if argv else T.ask_host()
    if host not in T.TRANSFER_HOSTS:
        raise SystemExit(f"unknown host '{host}'. Use one of {T.TRANSFER_HOSTS}.")

    mode = argv[1] if len(argv) > 1 else T.ask_mode()
    if mode not in ("up", "down"):
        raise SystemExit("mode must be 'up' or 'down'")

    if len(argv) > 2:
        model = argv[2]
    elif mode == "down":
        print(f"\nModels:\n  " + "\n  ".join(sorted(MODELS)))
        model = T.timed_input("Enter model name [30s timeout]: ", 30).strip()
    else:
        model = None  # upload is not model-specific

    if model is not None and model not in MODELS:
        raise SystemExit(f"unknown model '{model}'. Use one of {sorted(MODELS)}.")

    transfers = transfers_up(host) if mode == "up" else transfers_down(host, model)

    print(f"\n=== {mode.upper()} :: {host} ===")
    print(f"remote  : {remote_root(host)}")
    if mode == "down":
        print(f"staging : {staging_dir(model)}")
    print(f"entries : {len(transfers)}")
    confirm = input("Proceed? [y/N]: ").strip().lower()
    if confirm != "y":
        raise SystemExit("aborted.")

    T.run_transfers(host, transfers)

    if mode == "down":
        schema = MODELS[model]
        print(
            f"\nNext: verify and archive with\n"
            f"    python3 -c \"import sys; sys.path.insert(0,'analysis'); "
            f"import data_utils as d; d.sync_simulation_data("
            f"'{staging_dir(model)}', "
            f"'/home/debamalya/ComputationData/PostDoc/programming/new/data/{model}_model', "
            f"'{schema.system}')\""
        )
    else:
        print(
            f"\nNext, on the {host} login node:\n"
            f"    cd {remote_root(host)}\n"
            f"    python3 workflow/generator_two_band.py   # or generator_four_band.py\n"
            f"\n"
            f"That is the whole command. It builds the model binary once on the\n"
            f"login node, writes one task.csv per parameter point, submits the\n"
            f"sweep as a SINGLE SLURM array job, and exits in seconds -- nothing\n"
            f"is left running here, and SLURM does the throttling. Watch it with\n"
            f"    squeue -u {T.base_path(host).name}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
