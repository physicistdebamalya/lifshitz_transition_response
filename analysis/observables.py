"""
Derived quantities. Everything here is computed from data.csv plus the
category-B fields of parameters.txt -- nothing is re-read from the C++ source.
"""

import numpy as np

from snippets import get_param


def pulse_window(parameters):
    """Return (t_start, t_end, t_final) for a run.

    t_end is when the pulse switches off; the interval (t_end, t_final] is the
    free-evolution tail. The headline ratio is measured on that tail, so this
    is the single most load-bearing thing parameters.txt has to support.

    C++ builds each pulse as  tf = ti + ttotal * (1 + ETP/100), so inverting:

        t_end = t_start + (t_final - t_start) / (1 + ETP/100)

    Two fixes over the notebook version:
      * the notebook used  t_end = t_final / (1 + ETP/100), which silently
        assumes t_start == 0 (true for every run so far, not by construction)
      * it read ETPX as though ETPX == ETPY. Runs written before 2026-08 carry
        separate ETPX/ETPY; newer ones carry a single ETP. Both are handled,
        and disagreeing ETPX/ETPY is reported rather than silently resolved.
    """
    etp = get_param(parameters, "ETP")
    if etp is None:
        etpx = get_param(parameters, "ETPX")
        etpy = get_param(parameters, "ETPY")
        candidates = [e for e in (etpx, etpy) if e is not None]
        if not candidates:
            raise ValueError("parameters carry neither ETP nor ETPX/ETPY")
        if len(candidates) == 2 and etpx != etpy:
            raise ValueError(
                f"legacy entry has ETPX={etpx} but ETPY={etpy}; the post-pulse "
                f"window is ambiguous. Pass etp= explicitly to resolve it.")
        etp = candidates[0]

    t_start = get_param(parameters, "initial time")
    t_final = get_param(parameters, "final time")
    if t_start is None or t_final is None:
        raise ValueError("parameters lack 'initial time' / 'final time'")

    t_end = t_start + (t_final - t_start) / (1.0 + etp / 100.0)
    return t_start, t_end, t_final


def tail_mask(time, parameters):
    """Boolean mask selecting the post-pulse tail (t_end, t_final]."""
    _, t_end, t_final = pulse_window(parameters)
    return (time > t_end) & (time <= t_final)


def ratio_R(time_dot, j_dot, parameters):
    """R = max(J_dot) / J_dot(t_end), measured on the post-pulse tail.

    The dip in R against E_F is the Lifshitz-transition signature this project
    exists to compute.

    NOTE on archived data: runs written before 2026-08 stored results at the
    C++ stream default of 6 significant figures. Where J is large and slowly
    varying, consecutive samples were identical after rounding, so np.diff of
    them is quantization noise rather than a derivative. R computed from those
    runs is unreliable for the x channel. Re-run, or treat with suspicion.
    """
    mask = tail_mask(time_dot, parameters)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        raise ValueError("no samples in the post-pulse window")
    tail = j_dot[idx]
    denom = tail[0]                      # J_dot at t_end
    if denom == 0.0:
        return np.nan
    return float(np.max(tail) / denom)


def quantization_suspect(j, ratio=0.01):
    """True if J was written at too few significant figures for np.diff(J) to
    mean anything.

    Detects it by how few DISTINCT values the difference takes. For data
    written at full precision every consecutive difference is essentially
    unique; for data rounded to 6 significant figures the differences collapse
    onto a small set of integer multiples of the rounding step.

    Real example, archived four_band run 10286: Jx steps along a ladder of
    exactly 1e-7, and Jx_dot takes 37 distinct values across 52359 samples
    (7e-4). Its R_x came out as exactly 1.25 -- the ratio 10 steps / 8 steps,
    containing no physics at all.
    """
    j = np.asarray(j, dtype=float)
    if j.size < 3:
        return False
    d = np.diff(j)
    if not np.any(d):
        return True
    return np.unique(d).size < ratio * d.size
