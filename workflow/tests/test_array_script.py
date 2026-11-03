#!/usr/bin/env python3
"""Check the array job script sbatch is actually handed.

    python3 workflow/tests/test_array_script.py     # expect ALL PASS

Regression test for a real failure on 2026-08-27. A two_band sweep on param
prepared 24 run directories and then died at the last step with

    sbatch: error: This does not look like a batch script.  The first
    sbatch: error: line must start with #! followed by the path to an interpreter.

array_script() had been rewritten as textwrap.dedent(f\"\"\"...\"\"\"). That
composes in the wrong order: the f-string interpolates FIRST, so the moment a
substituted value carries its own newlines -- the module block does, on both
clusters -- the lines it contributes start at column 0, the common indent
dedent() can strip becomes the empty string, and dedent() returns the template
with every line still indented by the source file's twelve spaces. `#!/bin/bash`
at column 12 is not a shebang.

That failure was loud. The two next door are not, and are why this file checks
more than the first line:

  * an INDENTED #SBATCH is ignored by sbatch, silently -- the job then takes
    the cluster default for whatever that directive was setting, which for
    --array or --cpus-per-task is a different job than the one intended;
  * a #SBATCH AFTER the first command is ignored for the same reason, and is
    harder to see because the line itself looks perfectly correct.

Both are what _validate_batch_script() exists to refuse, and the tests below
break the generated script deliberately to confirm it does (philosophy §16:
mutation-test the harness -- a check that never fires is not a check).

No cluster is contacted: array_script() is pure text generation from a host
profile, so param's and hpc's real scripts can both be built here.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hpc_config                                              # noqa: E402
from submit_utils import Submitter                             # noqa: E402

failures = []


def check(label, got, want):
    if got == want:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}: got {got!r}, want {want!r}")
        failures.append(label)


def check_raises(label, fn, want=SystemExit):
    """Asserts that something DOES throw. A check that something must raise
    fails if nothing raises (philosophy §16)."""
    try:
        fn()
    except want as e:
        print(f"  PASS  {label} -> {str(e).splitlines()[0][:70]}")
        return
    except Exception as e:                       # noqa: BLE001
        print(f"  FAIL  {label}: raised {type(e).__name__}, want "
              f"{getattr(want, '__name__', want)}")
        failures.append(label)
        return
    print(f"  FAIL  {label}: nothing raised")
    failures.append(label)


HOSTS = ("param", "hpc")

for host in HOSTS:
    print(f"\n=== {host} ===")
    s = Submitter("two_band", host=host)
    s.job_name = "twoband"
    script = s.array_script(24)
    lines = script.splitlines()

    # The one that actually broke.
    check(f"{host}: first line is the shebang", lines[0], "#!/bin/bash")

    directives = [l for l in lines if l.lstrip().startswith("#SBATCH")]
    check(f"{host}: every #SBATCH starts at column 0",
          [l for l in directives if not l.startswith("#SBATCH")], [])

    # sbatch stops reading directives at the first command, so all of them
    # must precede it. `set -e` is that first command in this script.
    first_command = next(i for i, l in enumerate(lines)
                         if l.strip() and not l.strip().startswith("#"))
    last_directive = max(i for i, l in enumerate(lines)
                         if l.startswith("#SBATCH"))
    check(f"{host}: all {len(directives)} directives precede the first command",
          last_directive < first_command, True)

    # No walltime: the queue's limit is the administrators' to set, and the
    # job asks for nothing so it gets their default (2026-08-27).
    check(f"{host}: no --time is requested",
          [l for l in directives if "--time" in l], [])

    # The resources that reach the script are the ones plan_partition chose.
    plan = hpc_config.plan_partition(s.partition,
                                     partitions=s.host.partitions)
    for flag, key in (("--nodes", "nodes"), ("--ntasks-per-node", "npn"),
                      ("--cpus-per-task", "cpt")):
        check(f"{host}: {flag}={plan[key]}",
              f"#SBATCH {flag}={plan[key]}" in directives, True)
    check(f"{host}: --array=0-23%{plan['max_concurrent_jobs']}",
          f"#SBATCH --array=0-23%{plan['max_concurrent_jobs']}" in directives,
          True)

    # The module block has to survive interpolation intact and unindented --
    # it is the multi-line value that broke dedent, and a `    module load`
    # would run fine while telling you nothing about why.
    if s.host.modules:
        check(f"{host}: module block is present and unindented",
              all(l in lines for l in s.host.module_block().splitlines()), True)
    if s.host.env_exports:
        check(f"{host}: env exports are present and unindented",
              all(e in lines for e in s.host.env_exports), True)

    # set -e must precede the cd, or a failed cd runs the solver in the
    # submit directory and reports success.
    check(f"{host}: set -e precedes the cd",
          lines.index("set -e") < next(i for i, l in enumerate(lines)
                                       if l.startswith("cd ")), True)


print("\n=== the validator refuses what sbatch would misread ===")
s = Submitter("two_band", host="param")
s.job_name = "twoband"
good = s.array_script(4)

check_raises("indented shebang is refused",
             lambda: s._validate_batch_script(
                 "\n".join("    " + l for l in good.splitlines())))
check_raises("indented #SBATCH is refused",
             lambda: s._validate_batch_script(
                 good.replace("#SBATCH --nodes=", "    #SBATCH --nodes=")))
check_raises("#SBATCH after the first command is refused",
             lambda: s._validate_batch_script(
                 good.replace("set -e", "set -e\n#SBATCH --nodes=1")))
check_raises("a re-introduced --time is refused",
             lambda: s._validate_batch_script(
                 good.replace("#SBATCH --hint=nomultithread",
                              "#SBATCH --time=1-00:00:00")))
# A programmer error rather than an operational stop, so ValueError and not
# SystemExit -- but it must still throw: `--array=0--1%1` is a job SLURM would
# reject far from here.
check_raises("an empty array is refused", lambda: s.array_script(0), ValueError)

# The exact shape of the 2026-08-27 failure, reproduced: interpolate a
# multi-line value into an indented template and dedent afterwards.
import textwrap                                                # noqa: E402

block = s.host.module_block()
broken = textwrap.dedent(f"""\
    #!/bin/bash
    #SBATCH --job-name=x
    {block}
    set -e
    """)
check("the old dedent-after-interpolate shape really did indent the shebang",
      broken.splitlines()[0], "    #!/bin/bash")
check_raises("...and the validator would have caught it",
             lambda: s._validate_batch_script(broken))

print()
if failures:
    print(f"❌ {len(failures)} FAILED: {failures}")
    raise SystemExit(1)
print("✅ ALL PASS")
