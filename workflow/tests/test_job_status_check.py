#!/usr/bin/env python3
"""Check the pre-download SLURM completion check reads job IDs correctly.

    python3 workflow/tests/test_job_status_check.py     # expect ALL PASS

This is a regression test for a real misfire on 2026-08-25. A two_band
download from param printed "79 of 263 job(s) found there did NOT complete
cleanly" and listed TIMEOUTs of two and five days -- for a sweep whose 207
elements had all finished cleanly a few hours earlier under array job 116254.

Nothing had failed. check_remote_run_completion() built its sacct query by
extracting the number from the END of each log filename, and since the
array-job change (2026-08-22) the job script writes
mpi_logs_<arrayjobid>_<index>.out -- so the number at the end is the array
TASK INDEX. The query became `sacct -j 0,1,2,...,206`, which is valid, and
which returned 200-odd unrelated jobs from the beginning of the account's
history. The pre-array format (mpi_logs_%j.out) has only one number, so the
old regex had been right for it and wrong the moment the format gained an
index.

That failure mode is the dangerous kind -- no exception, no empty result, a
confident verdict about the wrong jobs -- so what is pinned here is the
extraction itself, run through a real shell over real filenames, in both log
formats. No cluster is contacted: run_remote_capture() is stubbed with
sacct-shaped text, and the find|grep half runs locally against a scratch tree
named exactly the way Submitter.array_script() names things.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hpc_transfer_utils as U  # noqa: E402

failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}" + (f"   {detail}" if detail else ""))
        failures.append(name)


def extract_ids(root):
    """The ID-extraction half of check_remote_run_completion()'s command,
    built from the same constants, run locally. sacct does not exist off the
    cluster, so the query is cut just before it."""
    globs = " -o ".join(f"-name '{g}'" for g in U._JOB_LOG_GLOBS)
    cmd = (f"find '{root}' \\( {globs} \\) 2>/dev/null "
           f"| grep -oE '{U._JOB_LOG_PREFIX}[0-9]+' | grep -oE '[0-9]+' "
           f"| sort -un | paste -sd,")
    return subprocess.run(["bash", "-c", cmd], capture_output=True, text=True).stdout.strip()


def stub(output):
    U.run_remote_capture = lambda cmd: (0, output)


scratch = Path(tempfile.mkdtemp(prefix="job_status_test_"))
try:
    print("\njob ID extraction from log filenames")

    # Post-2026-08-22: mpi_logs_<arrayjobid>_<index>. The whole point.
    arr = scratch / "array"
    for idx in (0, 7, 128, 206):
        d = arr / f"dir_{10000 + idx}"
        d.mkdir(parents=True)
        (d / f"mpi_logs_116254_{idx}.out").touch()
        (d / f"mpi_logs_116254_{idx}.err").touch()
    check("array format yields the array JOB id, not the task index",
          extract_ids(arr) == "116254", extract_ids(arr))

    # Pre-array: mpi_logs_%j, one number. Must keep working.
    old = scratch / "old"
    for jid in (98578, 98579):
        d = old / f"dir_{jid}"
        d.mkdir(parents=True)
        (d / f"mpi_logs_{jid}.out").touch()
    check("pre-array format still yields its job ids",
          extract_ids(old) == "98578,98579", extract_ids(old))

    # The reason the regex was suffix-anchored originally: digits elsewhere in
    # the path must not be read as job IDs.
    noisy = scratch / "two_band_model_4"
    d = noisy / "dir_55555"
    d.mkdir(parents=True)
    (d / "mpi_logs_116254_3.out").touch()
    (d / "data.csv").touch()
    (d / "parameters.txt").touch()
    check("digits in dir_NNNNN and the model name are not job ids",
          extract_ids(noisy) == "116254", extract_ids(noisy))

    empty = scratch / "empty"
    (empty / "dir_10000").mkdir(parents=True)
    check("a tree with no job logs yields no ids", extract_ids(empty) == "",
          repr(extract_ids(empty)))

    print("\nsacct parsing")

    stub("QUERIED:116254\n"
         "116254_[0-206]|COMPLETED|00:00:00|0:0\n"
         "116254_0|COMPLETED|10:11:12|0:0\n"
         "116254_0.batch|COMPLETED|10:11:12|0:0\n"
         "116254_0.0|COMPLETED|10:11:12|0:0\n"
         "116254_1|COMPLETED|09:58:01|0:0\n")
    unclean, checked, queried = U.check_remote_run_completion("/remote")
    check("a clean array reports nothing unclean", unclean == [], repr(unclean))
    check("step rows (.batch, .0) are not counted as jobs", checked == 3, checked)
    check("the queried ids are returned for display", queried == ["116254"], queried)

    # State alone would miss this one.
    stub("QUERIED:116254\n116254_5|COMPLETED|00:00:03|1:0\n")
    unclean, checked, queried = U.check_remote_run_completion("/remote")
    check("COMPLETED with a nonzero ExitCode is still unclean",
          len(unclean) == 1, repr(unclean))

    stub("QUERIED:116254\n"
         "116254_0|COMPLETED|10:00:00|0:0\n"
         "116254_1|TIMEOUT|2-00:00:02|0:1\n"
         "116254_2|FAILED|00:00:07|127:0\n")
    unclean, checked, queried = U.check_remote_run_completion("/remote")
    check("TIMEOUT and FAILED elements are caught", len(unclean) == 2, repr(unclean))
    check("the clean element among them is not", checked == 3, checked)

    # checked==0 must stay distinguishable from "confirmed complete".
    stub("QUERIED:\n")
    unclean, checked, queried = U.check_remote_run_completion("/remote")
    check("nothing found means checked==0, not a clean bill",
          checked == 0 and queried == [], f"{checked} {queried}")

    print("\nreporting never blocks the download")

    # A status check that cannot run must warn and return, not raise: an
    # incomplete run's data.csv is still worth downloading on purpose.
    def boom(cmd):
        raise OSError("socket closed")
    U.run_remote_capture = boom
    try:
        U._warn_if_run_incomplete("param", "/remote")
        check("a failed status check warns and continues", True)
    except Exception as e:
        check("a failed status check warns and continues", False, repr(e))

    # A whole array failing must not bury the count in 200 lines of scrollback.
    stub("\n".join(["QUERIED:116254"]
                   + [f"116254_{i}|FAILED|00:00:0{i % 10}|1:0" for i in range(40)]))
    unclean, checked, queried = U.check_remote_run_completion("/remote")
    check("40 failed elements all parsed", len(unclean) == 40, len(unclean))

finally:
    subprocess.run(["rm", "-rf", str(scratch)])

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("ALL PASS")
