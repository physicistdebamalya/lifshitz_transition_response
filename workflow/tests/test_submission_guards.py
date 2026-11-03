#!/usr/bin/env python3
"""Check where a sweep lands, and what stops it going out twice.

    python3 workflow/tests/test_submission_guards.py     # expect ALL PASS

Three things decided at submission time, none of which announces itself when
it goes wrong:

  1. THE RUN ROOT. Run directories used to be created in Path.cwd(). Invoking
     the generator from workflow/ instead of the model's directory therefore
     put dir_NNNNN into the source tree -- 96 of them on hpc, then 46 more --
     under a directories.txt that no later `down <model>` could split by
     model. run_root_for() derives <repo>/<model> from the generator's own
     location instead, so the cwd stops mattering.

  2. THE REGISTRY. already_exists() reads it, and a path that does not exist
     is not an error: it warns once and submits every point as new. That is
     the expensive silence -- for four_band it is 23 points of already
     archived physics -- so default_archive() derives the path per machine
     rather than leaving it to a remembered --archive.

  3. WHAT IS ALREADY RUNNING. Moving the run root makes a sweep submitted
     under the OLD rule invisible to the new one: preparing_run() reads
     .array_job_id in the new root, which is empty. On 2026-08-27 hpc had
     exactly that -- array 214963, 46 elements, live out of
     ~/projects/new/workflow/ -- so refuse_foreign_repo_jobs() looks for a
     live job anywhere in this repository that is not a run root.

No cluster is contacted: squeue is stubbed with squeue-shaped rows, including
the real shape of the hpc array that motivated the guard.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import submit_utils as S                                       # noqa: E402
from models import MODELS                                      # noqa: E402

failures = []


def check(label, got, want):
    if got == want:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}: got {got!r}, want {want!r}")
        failures.append(label)


REPO = S.repo_root()

print("=== the run root follows the repository, not the cwd ===")
check("repo_root() is this checkout", REPO, Path(__file__).resolve().parents[2])
for model in sorted(MODELS):
    check(f"{model}: run root is <repo>/{model}",
          S.run_root_for(model), REPO / model)
    # The same directory hpc_transfers.py creates on a push. If these two ever
    # disagree, `up` lays out one tree and the generator fills another.
    check(f"{model}: run root leaf matches the pushed layout",
          S.run_root_for(model).name, model)


def _catch(fn):
    """What did calling this throw? BaseException, not Exception: SystemExit
    is the whole point of half these checks and does not inherit from it."""
    try:
        fn()
    except BaseException as e:                                 # noqa: BLE001
        return e
    return None


check("an unknown model has no run root",
      type(_catch(lambda: S.run_root_for("three_band"))), ValueError)

print("\n=== the registry is derived per machine ===")
for model in sorted(MODELS):
    check(f"{model}: on the workstation, the real archive",
          S.default_archive(model, host="local"),
          S.WORKSTATION_ARCHIVE / f"{model}_model")
    for host in ("param", "hpc"):
        check(f"{model}: on {host}, the copy that travelled with the code",
              S.default_archive(model, host=host),
              REPO / S.ARCHIVE_SUBDIR / f"{model}_model")
check("an unknown model has no default archive",
      type(_catch(lambda: S.default_archive("three_band"))), ValueError)

# hpc_transfers pushes the registry to remote_root/<ARCHIVE_SUBDIR>/<model>_model
# and the generator reads <repo>/<ARCHIVE_SUBDIR>/<model>_model. One constant,
# imported by both -- checked here because two copies of "archive" would put
# the file somewhere already_exists() does not look.
sys.path.insert(0, str(REPO / "analysis"))
import hpc_transfers as H                                      # noqa: E402

check("hpc_transfers pushes to the subdirectory the generator reads",
      H.REMOTE_ARCHIVE_SUBDIR, S.ARCHIVE_SUBDIR)
check("hpc_transfers reads the archive root the generator reads",
      Path(H.ARCHIVE_ROOT), S.WORKSTATION_ARCHIVE)
for model in sorted(MODELS):
    check(f"{model}: remote run root matches run_root_for's leaf",
          H.remote_model_root("param", model).rsplit("/", 1)[-1],
          S.run_root_for(model).name)

print("\n=== a sweep still live in this repo blocks the next one ===")


def with_jobs(jobs):
    """Stub squeue. None means unreachable, {} means nothing queued."""
    S.live_jobs_by_workdir = lambda user, timeout=30: jobs


CASES = (
    # (label, squeue rows, must refuse?)
    ("the hpc case: array 214963 live out of <repo>/workflow/",
     {str(REPO / "workflow"): [("214963_24", "RUNNING"),
                               ("214963_[25-46]", "PENDING")]}, True),
    ("a job submitted from the repo root itself",
     {str(REPO): [("9000", "PENDING")]}, True),
    ("another model's run root is legitimate, not foreign",
     {str(REPO / "four_band"): [("9001", "RUNNING")]}, False),
    ("this model's own run root is _array_still_live()'s business",
     {str(REPO / "two_band"): [("9002", "RUNNING")]}, False),
    ("an unrelated project on the same account",
     {"/home/debamalya.dutta/Superconductivity/runs": [("123344", "RUNNING")]},
     False),
    ("nothing queued", {}, False),
    ("squeue unreachable warns and does not block", None, False),
)

original = S.live_jobs_by_workdir
try:
    for label, jobs, must_refuse in CASES:
        with_jobs(jobs)
        s = S.Submitter("two_band", host="param")
        refused = isinstance(_catch(s.refuse_foreign_repo_jobs), SystemExit)
        check(label, refused, must_refuse)
finally:
    S.live_jobs_by_workdir = original

print()
if failures:
    print(f"❌ {len(failures)} FAILED: {failures}")
    raise SystemExit(1)
print("✅ ALL PASS")
