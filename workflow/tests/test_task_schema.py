#!/usr/bin/env python3
"""Check the task.csv layer: round-trip exactness, the guards, and the
identity-neutrality of moving b1/b2/k_origin into the task file.

    python3 workflow/tests/test_task_schema.py     # expect ALL PASS

WHY ROUND-TRIP EXACTNESS IS THE WHOLE POINT
-------------------------------------------
A value written to task.csv is parsed by std::stod, printed to
parameters.txt at setprecision(6), and compared to Python's
format_number(".6g") by EXACT STRING EQUALITY. One bit of drift in the
double and the two six-figure renderings can differ in the last place, at
which point the run can never be found in the archive again and
already_exists() resubmits it on every sweep. So this file walks EVERY point
of both real production sweeps, not a sample.

WHY b1/b2/k_origin GET THEIR OWN CHECK
--------------------------------------
Those three are in schema.identifying -- they reach the match query through
CreateSnippet -- but the OLD build_replacements() never wrote them into the
model source, so their values lived in two hand-maintained places at once:
the .cpp template and each generator's BASE, with nothing comparing them.
They happened to agree, but a divergence would have been silent and
permanent: a run written with one b1 and searched for with another can never
be matched again, so already_exists() would resubmit it on every sweep,
forever. task.csv makes the task file the single source of truth; the check
below is what stops a literal creeping back into the .cpp and recreating the
second copy.
"""

import importlib.util
import re
import sys
from pathlib import Path

WORKFLOW = Path(__file__).resolve().parent.parent
REPO = WORKFLOW.parent
sys.path.insert(0, str(WORKFLOW))
sys.path.insert(0, str(REPO / "analysis"))

import models                                                   # noqa: E402
import task_schema                                              # noqa: E402

failures = []


def check(label, got, want):
    if got == want:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}: got {got!r}, want {want!r}")
        failures.append(label)


def check_raises(label, exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc:
        print(f"  PASS  {label}")
    except Exception as e:                                      # noqa: BLE001
        print(f"  FAIL  {label}: raised {type(e).__name__}, want {exc.__name__}")
        failures.append(label)
    else:
        print(f"  FAIL  {label}: did not raise")
        failures.append(label)


def load_generator(name):
    spec = importlib.util.spec_from_file_location(name, WORKFLOW / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GENERATORS = {"two_band": "generator_two_band", "four_band": "generator_four_band"}


# ── 1. every column of every sweep point survives the round trip ──────────
print("round-trip exactness over EVERY point of both production sweeps")
for model_name, gen_name in GENERATORS.items():
    schema = models.MODELS[model_name]
    generator = load_generator(gen_name)
    points = list(generator.sweep())
    drifted = []
    for p in points:
        text = ",".join(schema.task_columns) + "\n" + ",".join(
            task_schema.format_cell(task_schema._lookup(p, c, schema),
                                    schema.column_kind(c))
            for c in schema.task_columns) + "\n"
        cells = dict(zip(text.splitlines()[0].split(","),
                         text.splitlines()[1].split(",")))
        for column, cell in cells.items():
            kind = schema.column_kind(column)
            original = task_schema._lookup(p, column, schema)
            if kind is str:
                ok = cell == str(original)
            else:
                ok = kind(cell) == kind(original)
            if not ok:
                drifted.append((column, original, cell))
    check(f"{model_name}: {len(points)} point(s) x {len(schema.task_columns)} columns exact",
          drifted, [])


# ── 2. the guards fire ────────────────────────────────────────────────────
print("\nformat_cell refuses what would corrupt a run silently")
check_raises("a comma in a string cell", ValueError,
             task_schema.format_cell, "sin^2,Envelope", str)
check_raises("a newline in a string cell", ValueError,
             task_schema.format_cell, "sin^2\nEnvelope", str)
check_raises("10.7 into an int column (used to truncate to 10)", ValueError,
             task_schema.format_cell, 10.7, int)
check("10.0 into an int column is fine", task_schema.format_cell(10.0, int), "10")
check("a float keeps full precision",
      task_schema.format_cell(1 / 3, float), repr(1 / 3))

print("\nwrite_task_csv names every missing column at once")
schema = models.TWO_BAND
try:
    task_schema.write_task_csv(Path("/dev/null"), {"System": schema.system}, schema)
except KeyError as e:
    message = str(e)
    named = [c for c in schema.task_columns if c != "System" and c in message]
    check(f"names all {len(schema.task_columns) - 1} missing columns",
          len(named), len(schema.task_columns) - 1)
else:
    check("raises KeyError on a mostly-empty dict", "no raise", "KeyError")


# ── 3. b1/b2/k_origin have exactly ONE source of truth ───────────────────
print("\nb1/b2/k_origin come from task.csv, not from a literal in the .cpp")
_VECTOR = re.compile(r"const\s+Eigen::Vector2d\s+(\w+)\s*\{([^}]*)\}")

for model_name in GENERATORS:
    schema = models.MODELS[model_name]
    source = (REPO / "physics" / "models" / f"{model_name}.cpp").read_text()
    declared = dict(_VECTOR.findall(source))
    for name in schema.vectors:
        body = declared.get(name, "")
        # Every initialiser must read out of the loaded TaskSpec. A numeric
        # literal here would mean the value is set in two places at once --
        # the .cpp and the generator's BASE -- with nothing comparing them,
        # which is what made this silently unfindable before: a run written
        # with one b1 and searched for with another can never be matched, so
        # already_exists() resubmits it on every sweep, forever.
        check(f"{model_name}: {name} initialised from the task struct, "
              f"not a literal  ({body})",
              body.count("T.") == 2, True)


print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("ALL PASS")
