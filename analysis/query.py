"""
Selecting archived runs from a parameter dict.

The layer between "here are the parameters I care about" and "here are the
runs, as a table". It owns no matching logic of its own: identity is
snippets.CreateSnippet + find_key_by_substring, exactly what
workflow/submit_utils.already_exists() calls, so "already computed" and
"findable from a notebook" cannot drift apart.

Ported from Superconductivity's AM3.ipynb / AM3_utils.py, where a single
`all_params` dict drives every lookup and WILD marks the axes a given figure
sweeps.

--------------------------------------------------------------------------
THE THREE KINDS OF ENTRY IN A PARAMETER DICT
--------------------------------------------------------------------------
fixed       a value. Constrains the query: only runs carrying it match.

WILD ("?")  "any archived value". The key's line is dropped from the query,
            so every value of it matches. This is how a figure sweeps an
            axis -- `ef=WILD` returns the whole E_F sweep in one call
            instead of a loop that has to know the sweep in advance.

not set     initial time, final time, TIME_DIVISION, ETP. Written by the
            solver, never chosen here. CreateSnippet defaults the two time
            fields and DEFAULT_IGNORE_PATTERNS drops all four from the
            match, so a value typed for them would be silently inert --
            which is why they are refused outright by check_params() rather
            than quietly accepted. They are still READ back off the matched
            entry: observables.pulse_window() needs ETP and `final time` to
            place the post-pulse window.

--------------------------------------------------------------------------
WHY A FIGURE OVERRIDES THE DICT
--------------------------------------------------------------------------
A figure plotting R against E_F cannot also pin E_F. So each figure declares
its own swept axes and forces them to WILD regardless of what the caller's
dict holds -- and every function here REPORTS the values it discarded that
way. A number typed into the dict and then ignored is otherwise exactly the
kind of thing that reads as a bug in the figure.

ef and nu must be wildcarded together. nu tracks ef across both sweeps
(omega = 2*pi*nu = 2*E_F, the interband resonance), so wildcarding ef alone
leaves the query pinned to one archived nu and matches a single run.
"""

import time

import numpy as np
import pandas as pd

import observables as obs
from loaders import plot_data
from models import MODELS, by_system
from snippets import (CreateSnippet, find_key_by_substring,
                      show_nearest_candidates, normalize_entry_to_line_set,
                      DEFAULT_IGNORE_PATTERNS, WILD, format_number)

# Category B: in parameters.txt, out of the match. Never set in a dict here.
NOT_SETTABLE = tuple(DEFAULT_IGNORE_PATTERNS)

# Aliases people actually type for the non-settable fields, so check_params()
# can name the offending key instead of letting it sit inert in the dict.
_NOT_SETTABLE_KEYS = frozenset(
    list(NOT_SETTABLE) + ["initial_time", "final_time", "ETPX", "ETPY"])

# CreateSnippet reads these four unconditionally, then drops their lines via
# DEFAULT_IGNORE_PATTERNS -- so it needs them present but never uses the value.
# Filled here so a query dict does not have to carry numbers that cannot
# affect it. Whatever the archive holds is what gets read back.
_INERT_FILL = {"TIME_DIVISION": 0, "ETP": 0.0,
               "initial_time": 0.0, "final_time": 0.0}

# Preferred left-to-right sort order for a run table. Only the columns that
# exist are used, so this serves both model schemas.
_SORT_PREFERENCE = ("Temperature", "dephase", "ef", "nu")


# ------------------------------------------------------------- the dict ------
def schema_of(params, schema=None):
    """Resolve the ModelSchema for a parameter dict."""
    if schema is None:
        return by_system(params["System"])
    if isinstance(schema, str):
        return MODELS[schema]
    return schema


def check_params(params, schema=None):
    """Raise unless `params` is a usable query dict.

    Catches the two failures that are otherwise silent: a missing identifying
    key (CreateSnippet would raise, but with less context), and a value set
    for a field that is excluded from identity -- which would look like a
    constraint while constraining nothing.
    """
    schema = schema_of(params, schema)

    inert = sorted(k for k in params if k in _NOT_SETTABLE_KEYS)
    if inert:
        raise KeyError(
            f"{inert} cannot be set: written by the solver and excluded from "
            f"identity ({list(NOT_SETTABLE)}). Setting one constrains nothing, "
            f"so it is refused rather than silently ignored. They are read back "
            f"off the matched run -- observables.pulse_window() uses ETP and "
            f"'final time' to place the post-pulse window.")

    missing = [k for k in schema.identifying if k not in params]
    if missing:
        raise KeyError(f"{schema.name}: missing identifying key(s) {missing}")
    return schema


def wildcards(params):
    """The keys currently set to WILD, sorted."""
    return sorted(k for k, v in params.items()
                  if isinstance(v, str) and v == WILD)


def resolve(params, wild=(), **overrides):
    """The dict a query actually runs on: `params`, then `overrides`, then
    `wild` forced to WILD.

    Order matters. `wild` is applied last because it is the figure's own
    swept axis, and a figure cannot plot against a pinned axis -- so it wins
    over both the caller's dict and an explicit override.

    Returns (resolved_dict, discarded), where `discarded` maps each
    wildcarded key to the value that was displaced, for reporting.
    """
    p = dict(params)

    unknown = [k for k in overrides if k not in p]
    if unknown:
        raise KeyError(f"override(s) {unknown} are not in the parameter dict; "
                       f"a typo here would otherwise add an unconstrained key")
    p.update(overrides)

    unknown = [k for k in wild if k not in p]
    if unknown:
        raise KeyError(f"cannot wildcard {unknown}: not in the parameter dict")

    discarded = {k: p[k] for k in wild
                 if not (isinstance(p[k], str) and p[k] == WILD)}
    for k in wild:
        p[k] = WILD
    return p, discarded


def describe(params, discarded=None, indent="  "):
    """One block saying what this query constrains, what it sweeps, and which
    entered values were discarded to sweep it."""
    lines = []
    wild = wildcards(params)
    if wild:
        lines.append(f"{indent}swept (WILD -- any archived value matches): "
                     f"{', '.join(wild)}")
    if discarded:
        shown = ", ".join(f"{k} = {format_number(v) if not isinstance(v, (list, tuple)) else v}"
                          for k, v in sorted(discarded.items()))
        lines.append(f"{indent}    entered value(s) ignored, this figure sweeps "
                     f"them: {shown}")
    lines.append(f"{indent}never set here (solver-written, excluded from "
                 f"identity): {', '.join(NOT_SETTABLE)}")
    return "\n".join(lines)


# ------------------------------------------------------------- selecting -----
def find_runs(params, master, schema=None, *, wild=(), required=True,
              verbose=True, **overrides):
    """Every archive ID matching `params`, with `wild` forced to WILD.

    Parameters
    ----------
    params    : the parameter dict.
    master    : parsed logfile, from data_utils.parse_master_log().
    schema    : ModelSchema, its name, or None to read it off params["System"].
    wild      : keys this call sweeps. Forced to WILD over anything in
                `params` or `overrides`; the displaced values are reported.
    overrides : per-call values, e.g. Temperature=0.0, dephase=0.0.
    required  : raise if nothing matches. A figure drawn from an empty
                selection is a blank axes, which looks like physics.

    Returns
    -------
    sorted list of run IDs (strings).
    """
    schema = check_params(params, schema)
    p, discarded = resolve(params, wild, **overrides)

    snpt = CreateSnippet({**_INERT_FILL, **p}, schema)
    ids = find_key_by_substring(master, snpt, verbose=False) or []

    if verbose:
        print(f"matched {len(ids)} run(s)")
        print(describe(p, discarded))

    if not ids and required:
        print(f"\nNo run matches. The query was "
              f"({len(snpt.omitted_vars)} pattern(s) ignored):\n")
        for line in snpt.new_snippet.splitlines():
            print(f"    {line}")
        show_nearest_candidates(
            master, snpt,
            set(l.strip() for l in snpt.new_snippet.splitlines()))
        raise LookupError(
            "no archived run matches this parameter dict -- see the nearest "
            "candidates above for which line(s) disagree")
    return ids


# ------------------------------------------------------------- tabulating ----
def ratio_table(params, data_path, master, schema=None, *, wild=(),
                verbose=True, **overrides):
    """One row per matching run: its sweep coordinates and its ratio.

    The table `plot_*` functions are drawn from. Every column is either read
    off the matched logfile entry or computed by `observables`; nothing is
    restated from the caller's dict, so a row always describes the run rather
    than the query that found it.

    Columns
    -------
    ``id``
        the 5-digit archive identifier, as a string.
    the schema's own scalars
        under their archive names -- ``Temperature``, ``ef``, ``dephase``,
        ``N1``, ``N2`` plus the model's band parameters. ``Temperature``, not
        ``T``: ``df.T`` is DataFrame transpose.
    ``nu``
        the drive frequency. Read from nuX, or from nuY when the x channel is
        off; the two are cross-checked and disagreement raises, since every
        figure here assumes one drive frequency per run.
    ``t_end``, ``t_final``
        pulse-off and end-of-trace, from ``observables.pulse_window()``.
    ``R_x``, ``R_y``
        the ratio per channel, ``observables.ratio_R``.
    ``tail_peak_x/y``, ``end_x/y``, ``off_x/y``
        the denominator actually used, the strict J_dot(t_end) reading, and
        how far past t_end the tail peak sits. Carried so the two conventions
        can be compared by dividing two columns rather than re-reading the
        archive.
    ``quant_x/y``
        ``observables.quantization_suspect``: True if J was written at too
        few significant figures for np.diff to be a derivative.

    Raises
    ------
    ValueError
        if two runs share every identifying parameter. That is a real
        archive state, not a corruption -- SKIP_EXISTING=False recomputes a
        point deliberately, and run IDs are random, so the duplicate lands
        beside the original. It is refused here rather than resolved,
        because silently taking one would hide exactly the comparison the
        recomputation was done to make.
    """
    schema = check_params(params, schema)
    ids = find_runs(params, master, schema, wild=wild, verbose=verbose,
                    **overrides)

    t0 = time.perf_counter()
    rows = []
    for n, run_id in enumerate(ids, start=1):
        run = plot_data(run_id, data_path, system=schema.system)
        pars = run.parameters
        _, t_end, t_final = obs.pulse_window(pars)

        row = {"id": run_id}
        for key in schema.scalars:
            row[key] = run.param(key, int if key in schema.ints else float)

        nux, nuy = run.param("nuX"), run.param("nuY")
        if nux is not None and nuy is not None and nux != nuy:
            raise ValueError(
                f"run {run_id} has nuX={nux} but nuY={nuy}; this table carries "
                f"one 'nu' per run and every figure here assumes a single "
                f"drive frequency")
        row["nu"] = nux if nux is not None else nuy

        row["t_end"], row["t_final"] = t_end, t_final

        tail = np.flatnonzero(obs.tail_mask(run.Time_dot, pars))
        if tail.size == 0:
            raise ValueError(f"run {run_id} has no samples after t_end={t_end}")
        for ch, j, j_dot in (("x", run.Jx, run.Jx_dot),
                             ("y", run.Jy, run.Jy_dot)):
            row[f"R_{ch}"] = obs.ratio_R(run.Time_dot, j_dot, pars)
            row[f"tail_peak_{ch}"] = float(np.max(j_dot[tail]))
            row[f"end_{ch}"] = float(j_dot[tail[0]])
            row[f"off_{ch}"] = obs.tail_peak_offset(run.Time_dot, j_dot, pars)
            row[f"quant_{ch}"] = obs.quantization_suspect(j)
        rows.append(row)

        if verbose and (n % 25 == 0 or n == len(ids)):
            print(f"  {n}/{len(ids)} runs   ({time.perf_counter() - t0:.0f} s)")

    df = pd.DataFrame(rows)
    check_unique(df, schema)

    by = [c for c in _SORT_PREFERENCE if c in df.columns]
    return df.sort_values(by or ["id"]).reset_index(drop=True)


def check_unique(df, schema):
    """Raise if two rows carry identical identifying parameters.

    The coordinate is every schema scalar plus nu: within one query
    everything else is pinned, so two rows agreeing on these agree on the
    whole parameter set and are the same physical point computed twice.
    """
    coords = [c for c in (*schema.scalars, "nu") if c in df.columns]
    if not coords or df.empty:
        return
    dup = df[df.duplicated(coords, keep=False)]
    if dup.empty:
        return
    report = []
    for values, g in dup.groupby(coords):
        where = ", ".join(f"{k} = {format_number(v)}"
                          for k, v in zip(coords, values)
                          if len(set(dup[k])) > 1 or k in ("ef", "Temperature",
                                                           "dephase"))
        report.append(f"    {where}   ->  runs {sorted(g.id)}")
    raise ValueError(
        "the archive holds more than one run for a single parameter point:\n"
        + "\n".join(report)
        + "\n  A recomputed point lands beside the original (run IDs are "
          "random, SKIP_EXISTING=False never overwrites), so this is a real "
          "archive state. Pick one explicitly -- e.g. "
          "df[df.id != '<id>'] -- rather than letting a figure choose.")


# ------------------------------------------------------------- selecting ----
def _apply(df, fixed):
    """Rows matching every (key, value) in `fixed`. No checking -- select()
    owns the diagnostics; this is the plain filter it reports against."""
    for key, value in fixed.items():
        df = df[df[key] == value]
    return df


def select(df, verbose=False, **fixed):
    """Rows of a run table matching `fixed`; every column not named is swept.

    The table-level twin of find_runs(): same idea -- pin some axes, sweep the
    rest -- one level down, where the archive has already been read and the
    question is which of the rows in hand to draw. A value of WILD is
    identical to not naming the key at all, so a figure can pass its swept
    axes through explicitly rather than by omission.

    Exists because `df[(df.Temperature == t) & (df.dephase == g)]` plus an
    is-it-empty check appeared in three figure functions with three different
    error messages, and an empty selection drawn without one is a blank axes
    that reads as a flat curve.
    """
    unknown = [k for k in fixed if k not in df.columns]
    if unknown:
        raise KeyError(f"no column(s) {unknown} in this table; have "
                       f"{[c for c in df.columns if c != 'id']}")

    pinned = {k: v for k, v in fixed.items()
              if not (isinstance(v, str) and v == WILD)}

    sel = df
    applied = {}
    for key, value in pinned.items():
        sel = sel[sel[key] == value]
        if sel.empty:
            # What is available is reported against the rows still standing when
            # this key was applied -- not against the whole table, which would
            # list values that the earlier pins had already excluded.
            available = ", ".join(format_number(v) for v in sorted(set(
                (df if not applied else _apply(df, applied))[key].tolist())))
            raise ValueError(
                f"no rows with {key} = {format_number(value)}"
                + (f", once {', '.join(f'{k} = {format_number(v)}' for k, v in applied.items())} "
                   f"had been applied" if applied else "")
                + f". Available {key} here: {available}")
        applied[key] = value

    if verbose:
        swept = [c for c in _SORT_PREFERENCE
                 if c in df.columns and c not in pinned
                 and len(set(sel[c].tolist())) > 1]
        print(f"  {len(sel)} row(s)"
              + (f"   pinned: " + ", ".join(f"{k} = {format_number(v)}"
                                            for k, v in pinned.items()) if pinned else "")
              + (f"   swept: {', '.join(swept)}" if swept else ""))
    return sel


# ------------------------------------------------------------- reporting -----
def sweep_summary(df, schema=None):
    """Compact description of what a run table actually spans: which columns
    vary, over what, and which are pinned. Replaces a hard-coded expectation
    about the grid's shape with a statement of what was found."""
    lines = [f"{len(df)} run(s)"]
    varying, fixed = [], []
    for col in df.columns:
        if col == "id" or df[col].dtype == bool:
            continue
        try:
            vals = sorted(set(df[col].tolist()))
        except TypeError:
            continue
        (varying if len(vals) > 1 else fixed).append((col, vals))

    for col, vals in varying:
        if col in ("t_end", "t_final") or col.startswith(
                ("R_", "tail_peak_", "end_", "off_")):
            lines.append(f"  {col:<12}: {min(vals):.6g} ... {max(vals):.6g}")
        else:
            shown = ", ".join(format_number(v) for v in vals[:12])
            more = "" if len(vals) <= 12 else f", ... ({len(vals)} values)"
            lines.append(f"  {col:<12}: {shown}{more}")
    if fixed:
        lines.append("  fixed        : "
                     + ", ".join(f"{c} = {format_number(v[0])}"
                                 for c, v in fixed
                                 if not c.startswith(("R_", "tail_peak_",
                                                      "end_", "off_", "t_"))))
    return "\n".join(lines)
