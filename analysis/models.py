"""
Per-model parameter schemas: the single source of truth for what goes into
parameters.txt, in what order, and in which of the three field categories.

Both emitters are driven from here:
  physics/models/<name>.cpp  print_core_parameters()   (C++, hand-mirrored)
  analysis/snippets.py       CreateSnippet             (Python, reads this)

The C++ side cannot import this file, so `verify_cpp_mirror()` below reads the
.cpp source and asserts the two agree. Run it from the test suite -- silent
drift between these two emitters is the failure mode this whole layer exists
to prevent.
"""

from dataclasses import dataclass, field
from pathlib import Path
import re


# ── the pulse block ───────────────────────────────────────────────────────
# Model-independent: both models carry the same two drive channels, and both
# emit them through the same PulseType::printPulseParams (physics/src/
# necessary_functions.cpp). Declared once here so task_columns() and the C++
# KNOWN_COLUMNS list have a single origin rather than two hand-typed ones.
AXES = ("X", "Y")

# (base name, kind) -- the numeric pulse parameters, per axis.
PULSE_SCALARS = (("A0", float), ("nu", float), ("sigma", float),
                 ("ntt_sigma", float), ("delay", float),
                 ("duration", float), ("number_of_cycle", int))

# The two PulseKind enums, per axis. Their cell values are exactly the
# spellings operator<<(std::ostream&, PulseKind) prints -- "sin^2",
# "Envelope", "NA" -- so a task.csv cell and a parameters.txt line agree
# without a translation table on the C++ side.
PULSE_ENUMS = ("pulse_type", "pulse_wrapper")


@dataclass(frozen=True)
class ModelSchema:
    name: str
    system: str                       # the System line: run-integrity fingerprint
    scalars: tuple                    # category A, in parameters.txt order
    vectors: tuple = ("b1", "b2", "k_origin")
    # Keys the C++ declares as `int` rather than `double`. Consulted when
    # casting a value read back out of the archive, and when deciding a
    # task.csv cell's kind. number_of_cycleX/Y are here but not in `scalars`,
    # so callers that iterate `scalars` and test membership are unaffected.
    ints: frozenset = frozenset(("N1", "N2", "TIME_DIVISION",
                                 "number_of_cycleX", "number_of_cycleY"))

    @property
    def identifying(self):
        return tuple(self.scalars) + tuple(self.vectors)

    # ── task.csv ─────────────────────────────────────────────────────────
    @property
    def task_columns(self):
        """The task.csv header, in a stable order.

        The C++ looks columns up BY NAME, so this order is cosmetic there --
        it is fixed only so one generated task.csv diffs cleanly against the
        next. A vector is flattened to `<name>_0`, `<name>_1` because a CSV
        cell cannot hold a comma and the splitter is deliberately naive.

        System leads because it is the run-integrity fingerprint: the binary
        checks this cell against its own compile-time constant at startup.
        """
        cols = ["System", *self.scalars]
        for v in self.vectors:
            cols += [f"{v}_0", f"{v}_1"]
        cols += ["TIME_DIVISION", "ETP"]
        for axis in AXES:
            cols += [f"{base}{axis}" for base, _ in PULSE_SCALARS]
            cols += [f"{base}{axis}" for base in PULSE_ENUMS]
        return tuple(cols)

    def column_kind(self, name):
        """int | float | str for one task.csv column."""
        if name == "System" or name.startswith(PULSE_ENUMS):
            return str
        if name in self.ints:
            return int
        return float


TWO_BAND = ModelSchema(
    name="two_band",
    system="Response of Lifshitz transition in a two-band model",
    scalars=("alpha", "beta", "delta", "m",
             "Temperature", "ef", "dephase", "N1", "N2"),
)

FOUR_BAND = ModelSchema(
    name="four_band",
    system="Response of Lifshitz transition in a four-band model",
    scalars=("V", "v", "v3", "gamma1",
             "Temperature", "ef", "dephase", "N1", "N2"),
)

MODELS = {m.name: m for m in (TWO_BAND, FOUR_BAND)}


def by_system(system_string):
    for m in MODELS.values():
        if m.system == system_string:
            return m
    raise ValueError(f"no model schema for System {system_string!r}")


# ------------------------------------------------------------------ mirror ----
_CPP_DIR = Path(__file__).resolve().parent.parent / "physics" / "models"


def cpp_emitted_keys(model_name, cpp_dir=None):
    """Scalar keys that print_core_parameters() emits, in source order.

    Reads `os << "key = " << ...` out of the lambda. Only the unconditional
    prefix is compared -- the pulse block is conditional by design and is
    checked separately by the round-trip test over the archive.
    """
    src = (Path(cpp_dir or _CPP_DIR) / f"{model_name}.cpp").read_text()
    start = src.index("auto print_core_parameters")
    body = src[start:src.index("if (world_rank == 0)", start)]
    keys, seen = [], set()
    for m in re.finditer(r'os << "([A-Za-z_][A-Za-z_0-9]*) = "', body):
        k = m.group(1)
        if k not in seen:
            seen.add(k)
            keys.append(k)
    return keys


def verify_cpp_mirror(model_name, cpp_dir=None):
    """Assert the C++ emitter and this schema still agree. Returns the list of
    problems; empty means they are in step."""
    schema = MODELS[model_name]
    emitted = cpp_emitted_keys(model_name, cpp_dir)
    expected = list(schema.scalars) + ["TIME_DIVISION", "ETP"]

    problems = []
    prefix = emitted[:len(expected)]
    if prefix != expected:
        problems.append(f"{model_name}: parameters.txt scalar block differs\n"
                        f"    schema : {expected}\n"
                        f"    C++    : {prefix}")
    for stale in ("ETPX", "ETPY"):
        if stale in emitted:
            problems.append(f"{model_name}: C++ still emits {stale}; only ETP is supported")
    if "TIME_DIVISION" not in emitted:
        problems.append(f"{model_name}: C++ does not emit TIME_DIVISION")
    return problems


def cpp_known_columns(model_name, cpp_dir=None):
    """The KNOWN_COLUMNS initialiser-list the .cpp declares, in source order.

    The input-side twin of cpp_emitted_keys(): that one checks what the
    binary WRITES to parameters.txt, this one checks what it READS from
    task.csv. Both exist for the same reason -- C++ cannot import this
    schema, so the only thing standing between it and silent drift is a
    check that parses the source.
    """
    src = (Path(cpp_dir or _CPP_DIR) / f"{model_name}.cpp").read_text()
    start = src.index("KNOWN_COLUMNS")
    body = src[start:src.index("}", start)]
    return re.findall(r'"([^"]+)"', body)


def verify_cpp_task_columns(model_name, cpp_dir=None):
    """Assert the C++ task.csv reader and this schema declare the same
    columns. Returns the list of problems; empty means they are in step.

    Order is NOT compared -- the C++ looks columns up by name, so a
    reordering is harmless. Membership is compared in both directions,
    because the two failures are different and both are silent:
      missing in C++  -> the column is written but never read; the run uses
                         whatever the loader left in that field
      extra in C++    -> the binary requires a column write_task_csv() never
                         emits, so every run aborts at startup
    """
    schema = MODELS[model_name]
    try:
        declared = set(cpp_known_columns(model_name, cpp_dir))
    except ValueError:
        return [f"{model_name}: no KNOWN_COLUMNS list found in the .cpp source"]

    expected = set(schema.task_columns)
    problems = []
    missing = sorted(expected - declared)
    extra = sorted(declared - expected)
    if missing:
        problems.append(f"{model_name}: C++ does not read task.csv column(s) {missing}")
    if extra:
        problems.append(f"{model_name}: C++ reads task.csv column(s) {extra} "
                        f"that the schema never writes")
    return problems
