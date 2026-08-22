#!/usr/bin/env python3
"""How many of each production sweep's points the archive already holds.

    python3 workflow/tests/test_sweep_archive_match.py     # expect ALL PASS

THE HIGHEST-VALUE END-TO-END CHECK IN THE REPO, and it exists because
`generator_*.py --dry-run` -- which is how this used to be done by hand --
CREATES 207 dir_NNNNN directories as a side effect, so nobody wants to run
it casually. This does the same matching with no side effects at all: it
only reads logfile.txt.

What it actually exercises, end to end, against the real 99-run archive:

    BASE -> expand_sweep -> CreateSnippet -> format_number(".6g")
         -> normalize_entry_to_line_set -> subset match

including b1/b2/k_origin, which no other test covers. Any change to the
identification layer, the six-significant-figure contract, or the schema
shows up here as a changed count.

The counts are pinned as literals. They are NOT arbitrary:

  two_band   207 points, 0 present -- BASE sweeps dephase and Temperature
             over values the 61 archived runs do not carry, so this sweep is
             genuinely new work. A nonzero count here means either the
             archive grew or BASE changed.
  four_band   23 points, 23 present -- BASE reproduces exactly the archived
             sweep, so all 23 match. This is the sharp one: any drift in the
             matcher turns matches into misses and the count drops.

If you deliberately change BASE or archive new runs, update the literals in
the same commit and say why.
"""

import importlib.util
import sys
from pathlib import Path

WORKFLOW = Path(__file__).resolve().parent.parent
REPO = WORKFLOW.parent
sys.path.insert(0, str(WORKFLOW))
sys.path.insert(0, str(REPO / "analysis"))

from submit_utils import Submitter                              # noqa: E402

# (generator module, model, expected total, expected already-present)
EXPECTED = (
    ("generator_two_band", "two_band", 207, 0),
    ("generator_four_band", "four_band", 23, 23),
)

failures = []


def load_generator(name):
    spec = importlib.util.spec_from_file_location(name, WORKFLOW / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


for gen_name, model, want_total, want_present in EXPECTED:
    generator = load_generator(gen_name)
    archive = Path(generator.ARCHIVE)

    if not (archive / "logfile.txt").exists():
        print(f"  SKIP  {model}: no archive at {archive} "
              f"(expected on a cluster; the archive lives on the workstation)")
        continue

    submitter = Submitter(model, host="local", data_path=archive, dry_run=True)
    points = list(generator.sweep())
    present = sum(1 for p in points if submitter.already_exists(p))

    for label, got, want in ((f"{model}: sweep size", len(points), want_total),
                             (f"{model}: already in archive", present, want_present)):
        if got == want:
            print(f"  PASS  {label} = {got}")
        else:
            print(f"  FAIL  {label}: got {got}, want {want}")
            failures.append(label)

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("ALL PASS")
