"""
Identification layer: parameters.txt <-> logfile.txt <-> data file.

This module is the single hinge of the project. Two very different consumers
depend on it and must agree exactly:

  analysis  "where is the data for these parameters?"   -> plot_data(ID)
  workflow  "have I already run these parameters?"      -> already_exists()

Both build a snippet from a parameter dict and test it against the archive,
so a change here silently changes both what you can find and what you submit.

Ported from Superconductivity (AM3_utils.py, data_utils.py), which arrived at
this design after the substring-matching version proved too brittle.

--------------------------------------------------------------------------
THE 6-SIGNIFICANT-FIGURE CONTRACT
--------------------------------------------------------------------------
C++ writes parameters.txt under `std::setprecision(6)`; format_number here
uses ".6g". Both sides of a comparison are pushed through format_number, so
the final test is exact string equality -- no float tolerance anywhere.

Neither number may be changed alone. If C++ precision changes, this changes.

--------------------------------------------------------------------------
THE THREE FIELD CATEGORIES
--------------------------------------------------------------------------
A. identifying          in parameters.txt, in the match
B. present, ignored     in parameters.txt, in DEFAULT_IGNORE_PATTERNS.
                        Read by analysis (t_end needs ETP + final time), but
                        derived or output, so meaningless for identity.
C. omitted              absent from parameters.txt entirely, because the run
                        gives them no meaning -- when A0Y == 0 there is no
                        such thing as nuY, and writing a stale one would make
                        two physically identical runs stop matching.

Category C is why emit_pulse() below is conditional. That rule is mirrored in
C++ print_core_parameters(); the two must stay in step.
"""

import numpy as np
from textwrap import dedent

from models import MODELS, by_system

# ---------------------------------------------------------------- formatting --
_SCALAR_FMT     = ".6g"              # must equal C++ std::setprecision(6)
SCALAR_FMT_C    = f"%{_SCALAR_FMT}"  # C-style, for pandas float_format
_THRESHOLD_ZERO = 1e-12              # below this, a value is written as 0

WILD = "?"                           # placeholder: match any value for this key


def format_number(x):
    """Canonical scalar rendering. Both the query and the stored entry go
    through this, which is what lets the comparison be exact string equality."""
    if x == WILD:
        return x
    x = float(x)
    if abs(x) < _THRESHOLD_ZERO:
        x = 0.0
    return format(x, _SCALAR_FMT)


def float_number(x):
    x = float(x)
    return 0.0 if abs(x) < _THRESHOLD_ZERO else x


# ------------------------------------------------------------------ reading --
def get_param(parameters, key, cast=float):
    """Read one scalar out of a parameters.txt block. Splits on the first '='
    only, so 'non zero momentum pairing | Q[0] = ...' style labels survive."""
    for line in parameters.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            if k.strip() == key:
                return cast(v.strip())
    return None


def get_vector_param(parameters, base_key, size=2, cast=float):
    """Read 'b1[0] = 12, b1[1] = 0' back into array([12., 0.]).

    Uses `base_key + "[" in k` rather than startswith so a line carrying a
    descriptive prefix still resolves -- the convention Superconductivity
    needed for 'non zero momentum pairing | Q[0]'.
    """
    result = [None] * size
    for line in parameters.splitlines():
        if base_key in line:
            for part in line.split(","):
                if "=" in part:
                    k, v = part.split("=", 1)
                    k, v = k.strip(), v.strip()
                    if base_key + "[" in k:
                        idx = int(k[k.find("[") + 1:k.find("]")])
                        result[idx] = cast(v)
    return np.array(result)


def resolve(key, snippet):
    """Map a bare ignore-key onto the prefix that actually starts its line.

    'b1' must become 'b1[0]', otherwise the prefix 'b1 =' matches nothing and
    ignoring a vector silently does nothing at all.
    """
    for line in snippet.splitlines():
        stripped = line.strip()
        if key + "[" in stripped:
            return stripped.split("=")[0].rstrip()
    return key


def normalize_entry_to_line_set(entry_text, omitted_vars_tuple=()):
    """Canonicalize a stored log entry into a set of 'key = value' strings,
    through the same pipeline the query uses.

      scalar  -> float() -> format_number()   (6 sig figs, both sides)
      vector  -> verbatim   (C++ already wrote it at this precision)
      string  -> verbatim   (pulse_type etc; exact text, no drift possible)
      ignored -> dropped
    """
    lines = set()
    for raw in entry_text.splitlines():
        s = raw.strip()
        if not s or s.startswith(omitted_vars_tuple):
            continue
        if "=" not in s:
            lines.add(s)                       # the System line
            continue
        k, _, v = s.partition("=")
        key, v_str = k.strip(), v.strip()
        if "," in v_str:
            lines.add(s)                       # vector line, verbatim
            continue
        try:
            lines.add(f"{key} = {format_number(v_str)}")
        except ValueError:
            lines.add(s)                       # non-numeric value, verbatim
    return lines


# ------------------------------------------------------------ pulse emission --
def emit_pulse(parts, axis, p):
    """Append the pulse lines for one axis -- and only the ones this pulse
    actually gives meaning to (category C).

    A0 is always written: it is the channel on/off switch. If it is zero the
    channel contributes nothing and no further key for that axis is emitted,
    so a leftover nuY in the sweep dict cannot split two identical runs into
    two non-matching archive entries.

    Mirrors print_core_parameters() in physics/models/*.cpp.
    """
    A0 = p[f"A0{axis}"]
    parts.append(f"A0{axis} = {format_number(A0)}")
    if A0 != WILD and float_number(format_number(A0)) == 0.0:
        return

    parts.append(f"delay{axis} = {format_number(p[f'delay{axis}'])}")

    wrapper    = p[f"pulse_wrapper{axis}"]
    pulse_type = p[f"pulse_type{axis}"]

    if wrapper == "Envelope":
        parts.append(f"pulse_type{axis} = {pulse_type} with Envelope")
        if pulse_type == "Gaussian":
            parts += [f"nu{axis} = {format_number(p[f'nu{axis}'])}",
                      f"sigma{axis} = {format_number(p[f'sigma{axis}'])}",
                      f"ntt_sigma{axis} = {format_number(p[f'ntt_sigma{axis}'])}"]
        elif pulse_type == "sin^2":
            parts += [f"nu{axis} = {format_number(p[f'nu{axis}'])}",
                      f"number_of_cycle{axis} = {format_number(p[f'number_of_cycle{axis}'])}"]

    elif wrapper == "NA":
        parts.append(f"pulse_type{axis} = {pulse_type} without Envelope")
        if pulse_type == "Gaussian":
            parts += [f"sigma{axis} = {format_number(p[f'sigma{axis}'])}",
                      f"ntt_sigma{axis} = {format_number(p[f'ntt_sigma{axis}'])}"]
        elif pulse_type == "sin^2":
            parts += [f"number_of_cycle{axis} = {format_number(p[f'number_of_cycle{axis}'])}",
                      f"nu{axis} = {format_number(p[f'nu{axis}'])}"]
        elif pulse_type in ("sin", "cos"):
            parts.append(f"nu{axis} = {format_number(p[f'nu{axis}'])}")
        elif pulse_type == "Linear":
            parts.append(f"duration{axis} = {format_number(p[f'duration{axis}'])}")
    else:
        raise ValueError(f"unsupported pulse_wrapper{axis}: {wrapper!r}")


# Category B: written to parameters.txt, excluded from identity.
#   initial/final time  derived from the pulse, but needed for the post-pulse
#                       analysis window (see analysis/observables.py)
#   TIME_DIVISION, ETP  simulation knobs. Also, no archived run carries
#                       TIME_DIVISION and every archived run carries ETPX/ETPY
#                       rather than ETP, so ignoring them keeps the 99 existing
#                       runs reachable. Drop them from this list only once the
#                       archive no longer needs to be searched for pre-2026-08
#                       runs, or once you actually sweep them.
DEFAULT_IGNORE_PATTERNS = ["initial time", "final time", "TIME_DIVISION", "ETP"]


class CreateSnippet:
    """Build the query snippet for one parameter dict.

    Takes a dict, not 30 positional arguments: the positional signature is
    exactly why four_band.ipynb and prl_case.ipynb ended up with two
    incompatible copies of this class.

    Any value set to "?" is wildcarded -- its line is dropped from the query,
    so every archived value of that key matches.
    """

    def __init__(self, p, schema=None, ignore_patterns=None):
        if schema is None:
            schema = by_system(p["System"])
        elif isinstance(schema, str):
            schema = MODELS[schema]
        if p["System"] != schema.system:
            raise ValueError(f"System {p['System']!r} does not match schema "
                             f"{schema.name} ({schema.system!r})")
        self.schema = schema

        missing = [k for k in schema.identifying if k not in p]
        if missing:
            raise KeyError(f"{schema.name}: parameters missing from the dict: {missing}")

        parts = [schema.system,
                 *[f"{k} = {format_number(p[k])}" for k in schema.scalars],
                 *[f"{k}[0] = {format_number(p[k][0])}, {k}[1] = {format_number(p[k][1])}"
                   for k in schema.vectors],
                 f"TIME_DIVISION = {format_number(p['TIME_DIVISION'])}",
                 f"ETP = {format_number(p['ETP'])}"]

        emit_pulse(parts, "X", p)
        emit_pulse(parts, "Y", p)

        parts += [f"initial time = {format_number(p.get('initial_time', 0.0))}",
                  f"final time = {format_number(p.get('final_time', 0.0))}"]

        snippet = dedent("\n".join(parts)).rstrip("\n")

        # copy: binding the caller's list and appending to it makes a re-run of
        # a notebook cell accumulate wildcard names from the previous run.
        self.omitted_vars = list(DEFAULT_IGNORE_PATTERNS if ignore_patterns is None
                                 else ignore_patterns)

        for line in snippet.splitlines():
            if line.strip().endswith(WILD):
                name = line.split("=")[0].strip()
                if name:
                    self.omitted_vars.append(name)

        # " =" suffix keeps 'v =' from also matching 'v3 ='; resolve() turns
        # 'b1' into 'b1[0]' so vector keys can be ignored at all.
        self.omitted_vars_tuple = tuple(f"{resolve(v, snippet)} =" for v in self.omitted_vars)

        self.snippet = snippet
        self.new_snippet = "\n".join(
            l for l in snippet.splitlines()
            if not l.strip().startswith(self.omitted_vars_tuple))


def find_key_by_substring(master_dict, snpt, verbose=True, diagnostics=False, n=3):
    """Return every archive ID whose entry contains the query.

    Subset, not substring: the query lines must all be present, in any order,
    and the entry may carry extra lines. That is what lets a pre-2026-08 entry
    (with ETPX/ETPY, without TIME_DIVISION) still match a query built today.
    """
    query_lines = set(l.strip() for l in snpt.new_snippet.splitlines())
    keys = [k for k, v in master_dict.items()
            if query_lines.issubset(normalize_entry_to_line_set(v, snpt.omitted_vars_tuple))]

    if keys:
        return sorted(set(keys))
    if verbose:
        print("\n⚠️  No match found.\n")
        if diagnostics:
            print(f"  Searched for ({len(snpt.omitted_vars)} pattern(s) ignored):\n")
            for line in snpt.new_snippet.splitlines():
                print(f"    {line}")
            show_nearest_candidates(master_dict, snpt, query_lines, n=n)
    return None


def show_nearest_candidates(master_dict, snpt, query_lines, n=3):
    """Print the closest entries and exactly which lines they disagree on --
    the difference between 'no match' and a usable answer."""
    scored = []
    for key, value in master_dict.items():
        entry = normalize_entry_to_line_set(value, snpt.omitted_vars_tuple)
        missing = query_lines - entry
        scored.append((len(missing), key, missing, entry))
    scored.sort(key=lambda t: t[0])

    print(f"\n  Nearest {min(n, len(scored))} candidate(s):\n")
    for nmiss, key, missing, entry in scored[:n]:
        print(f"    id {key}: {nmiss} line(s) differ")
        for m in sorted(missing):
            lhs = m.split("=")[0].strip()
            found = next((e for e in entry if e.startswith(lhs + " =")), "(absent)")
            print(f"        wanted : {m}")
            print(f"        found  : {found}")
        print()
