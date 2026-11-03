#!/usr/bin/env python3
"""How many of each production sweep's points the archive already holds.

    python3 workflow/tests/test_sweep_archive_match.py     # expect ALL PASS

THE HIGHEST-VALUE END-TO-END CHECK IN THE REPO, and it exists because
`generator_*.py --dry-run` -- which is how this used to be done by hand --
CREATES 207 dir_NNNNN directories as a side effect, so nobody wants to run
it casually. This does the same matching with no side effects at all: it
only reads logfile.txt.

What it actually exercises, end to end, against the real archive:

    BASE -> expand_sweep -> CreateSnippet -> format_number(".6g")
         -> normalize_entry_to_line_set -> subset match

including b1/b2/k_origin, which no other test covers. Any change to the
identification layer, the six-significant-figure contract, or the schema
shows up here as a changed count.

The counts are pinned as literals. They are NOT arbitrary:

  two_band    46 points, 46 present — BASE is the dephase = 0 Lifshitz scan,
             23 (ef, nu) pairs at T = 0.0 and 0.1, and the archive now holds
             every one of them. This is the sharp one: any drift in the
             matcher turns matches into misses and the count drops -- and with
             the sweep complete, dropping is the only thing it can do.
  four_band   23 points, 23 present — BASE reproduces exactly the archived
             sweep. SKIPped as of 2026-08-26: the earlier four_band runs were
             retired to `data_r/` and the rerun is not archived into `data/`
             yet, so there is no registry to match against. When it lands,
             this is the count it must show.

`present` is expected to CHANGE, and every change needs a look. It went
0 -> 207 on 2026-08-26 when `data/` replaced `data_r/` (see "Data locations"
in CLAUDE.md), and 207 -> 22 on 2026-08-27 when BASE moved to the dephase = 0
scan — the archive still holds all 207 of the old dephase 0.01/0.1/0.2 runs,
they are simply no longer points of this sweep. It went 22 -> 46 later the
same day, when the remaining 24 points of the scan were archived (229 -> 253
runs in `data/two_band_model`). Re-pin whenever a batch is archived, in the
same commit as the archiving, and say which batch. What must NEVER happen is
`present` falling with no archiving and no BASE edit behind it: that is the
matcher drifting, and it is the reason this file exists.

A skip is legitimate on a cluster, where the archive does not exist at all. On
the machine that holds it, a skip of EVERY model is not: it means the path
moved out from under the generators, and the check below turns that into a
failure rather than letting the run go green having verified nothing.

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
    ("generator_two_band", "two_band", 46, 46),
    ("generator_four_band", "four_band", 23, 23),
)

failures = []
checked = []
roots = set()


def load_generator(name):
    spec = importlib.util.spec_from_file_location(name, WORKFLOW / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


for gen_name, model, want_total, want_present in EXPECTED:
    generator = load_generator(gen_name)
    archive = Path(generator.ARCHIVE)
    roots.add(archive.parent)

    if not (archive / "logfile.txt").exists():
        print(f"  SKIP  {model}: no registry at {archive} -- "
              f"every point of this sweep would submit as new")
        continue

    checked.append(model)
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

# Skipping every model is only acceptable where the archive genuinely is not:
# on a cluster. If its root is right there and no model's registry was found
# beneath it, the layout moved and this test verified nothing at all.
if not checked:
    on_disk = sorted(str(r) for r in roots if r.is_dir())
    if on_disk:
        label = "some model's registry is reachable under an existing archive root"
        print(f"  FAIL  {label}: root exists at {on_disk[0]}, "
              f"no registry under it -- nothing was verified")
        failures.append(label)
    else:
        print("  SKIP  no archive on this machine (expected on a cluster; "
              "the archive lives on the workstation)")

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("ALL PASS")
