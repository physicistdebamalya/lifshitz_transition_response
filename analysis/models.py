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


@dataclass(frozen=True)
class ModelSchema:
    name: str
    system: str                       # the System line: run-integrity fingerprint
    scalars: tuple                    # category A, in parameters.txt order
    vectors: tuple = ("b1", "b2", "k_origin")
    ints: frozenset = frozenset(("N1", "N2", "TIME_DIVISION"))

    @property
    def identifying(self):
        return tuple(self.scalars) + tuple(self.vectors)


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
