#!/usr/bin/env python3
"""
Acceptance test for the derived-quantity layer.

    python3 analysis/tests/test_observables.py      # expect ALL PASS

Synthetic traces with analytically known answers, then the real archive. The
synthetic half is the load-bearing one: observables.py is the only definition of
the headline ratio, and the value it returns has to be a property of the trace
rather than of whichever sample the window happened to start on.

Why this file exists. ratio_R's numerator was `np.max(tail)` while its docstring
said the global maximum. Restricted to the tail, numerator and denominator are
the same number, so it returned exactly 1.0 for the x channel of 206 of the 207
runs in the 2026-08 two_band archive -- a flat line where the Lifshitz signature
should be, and flat in a way that looks like a physical result rather than a bug.
It had no callers at the time, so nothing caught it. Test 1 below is the case
that would have.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import observables as obs                                     # noqa: E402
from data_utils import clean_path, parse_master_log           # noqa: E402
from loaders import plot_data, MASTER_LOG_FILENAME            # noqa: E402
from models import MODELS                                     # noqa: E402

ARCHIVE = Path(clean_path(
    "/home/debamalya/ComputationData/PostDoc/programming/new/data"))

failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}" + (f"   {detail}" if detail else ""))
        failures.append(name)


def close(name, got, want, tol=1e-12):
    check(name, abs(got - want) <= tol * max(1.0, abs(want)), f"got {got!r}, want {want!r}")


def raises(name, fn, expect):
    try:
        fn()
    except Exception as exc:                                   # noqa: BLE001
        check(name, expect in str(exc), f"{type(exc).__name__}: {exc}")
        return
    check(name, False, "nothing raised")


def params(t_start=0.0, t_final=100.0, etp=100.0, **extra):
    """A parameters block in the shape parse_master_log() produces: one
    'key = value' per line, as text."""
    lines = [f"initial time = {t_start}", f"final time = {t_final}"]
    lines += [f"{k} = {v}" for k, v in extra.items()]
    if etp is not None:
        lines.insert(0, f"ETP = {etp}")
    return "\n".join(lines) + "\n"


print("\npulse_window inverts the C++ construction")

# C++: tf = ti + ttotal * (1 + ETP/100). With ti = 0, ETP = 100, tf = 100:
# ttotal = 50, so the pulse ends halfway.
close("ETP=100, t_i=0 -> t_end is halfway", obs.pulse_window(params())[1], 50.0)

# The bug this function was written to fix: t_end = t_final/(1 + ETP/100) is only
# right when t_start == 0. Shift the whole run by 10 and the naive form gives 55.
close("t_i is respected, not assumed zero",
      obs.pulse_window(params(t_start=10.0, t_final=110.0))[1], 60.0)
close("ETP=0 -> no tail at all, t_end == t_final",
      obs.pulse_window(params(etp=0.0))[1], 100.0)

# Legacy entries carry ETPX/ETPY instead of a single ETP.
close("legacy ETPX/ETPY, in agreement, is accepted",
      obs.pulse_window(params(etp=None, ETPX=100.0, ETPY=100.0))[1], 50.0)
raises("legacy ETPX != ETPY is ambiguous and raises rather than picking one",
       lambda: obs.pulse_window(params(etp=None, ETPX=100.0, ETPY=50.0)), "ambiguous")
raises("neither ETP nor ETPX/ETPY raises",
       lambda: obs.pulse_window(params(etp=None)), "neither ETP")
raises("a block with no 'final time' raises",
       lambda: obs.pulse_window("ETP = 100\ninitial time = 0\n"), "final time")

print("\ntail_mask selects (t_end, t_final], the half-open post-pulse window")

t = np.linspace(0.0, 100.0, 101)
mask = obs.tail_mask(t, params())
check("t_end itself is excluded", not mask[50])
check("the sample after t_end is included", mask[51])
check("t_final is included", mask[100])
check("nothing before t_end is included", not mask[:51].any())
check("50 of 101 samples selected", int(mask.sum()) == 50, int(mask.sum()))

print("\nratio_R: the numerator is the GLOBAL max, not the tail max")

# The regression case. A trace whose driven peak is 10 and whose post-pulse tail
# peaks at 2: R must be 5. The pre-fix code returned 2/2 = 1.0 here.
j = np.zeros_like(t)
j[25] = 10.0        # during the pulse
j[75] = 2.0         # in the tail
close("driven peak 10 over tail peak 2 gives 5", obs.ratio_R(t, j, params()), 5.0)
check("and NOT 1.0 -- the pre-fix numerator", obs.ratio_R(t, j, params()) != 1.0)

# When the largest excursion of the whole trace is in the tail, R is 1 by
# construction. That is a real answer, not the bug: nothing decayed.
j2 = np.zeros_like(t)
j2[25], j2[75] = 2.0, 10.0
close("a response that grows into the tail gives R = 1", obs.ratio_R(t, j2, params()), 1.0)

# The denominator is the tail PEAK, not the first tail sample. Build a tail that
# rings: it starts small, swings up, and its first sample is not its peak.
j3 = np.zeros_like(t)
j3[25] = 100.0
j3[51], j3[70] = 1.0, 4.0
close("denominator is the tail peak (4), not the first tail sample (1)",
      obs.ratio_R(t, j3, params()), 25.0)

# The y-channel case that decided the convention: J_dot(t_end) negative. The
# strict reading returns a negative "ratio"; the peak reading stays positive.
j4 = np.zeros_like(t)
j4[25] = 100.0
j4[51], j4[70] = -8.0, 5.0
close("a negative first tail sample does not flip the sign of R",
      obs.ratio_R(t, j4, params()), 20.0)
check("the strict t_end reading WOULD flip it", 100.0 / j4[51] < 0)

# Signed maxima, not maxima of |J_dot|: the pulse drives J_dot positive first.
j5 = np.zeros_like(t)
j5[25], j5[30] = 10.0, -50.0
j5[75] = 2.0
close("abs() is not used -- a large negative recoil is not the numerator",
      obs.ratio_R(t, j5, params()), 5.0)

raises("an empty tail raises instead of returning nan quietly",
       lambda: obs.ratio_R(t, j, params(etp=0.0)), "no samples")
check("an identically zero tail is nan, not a division by zero",
      np.isnan(obs.ratio_R(t, np.zeros_like(t), params())))

print("\ntail_peak_offset says whether the two conventions coincide")

close("a monotone tail peaks at its first sample", obs.tail_peak_offset(t, j, params()),
      t[75] - 50.0)
close("a ringing tail peaks later", obs.tail_peak_offset(t, j3, params()), t[70] - 50.0)
close("the smallest possible offset is one sample step",
      obs.tail_peak_offset(t, np.where(np.arange(101) == 51, 1.0, 0.0), params()),
      t[51] - 50.0)
raises("an empty tail raises here too",
       lambda: obs.tail_peak_offset(t, j, params(etp=0.0)), "no samples")

print("\nquantization_suspect catches too-few-significant-figures data")

rng = np.random.default_rng(0)
full = np.cumsum(rng.normal(size=5000)) * 1e-9 + 1.0
check("full-precision data is clean", not obs.quantization_suspect(full))
check("the same data rounded to 6 significant figures is flagged",
      obs.quantization_suspect(np.round(full, 6)))
check("a constant trace is flagged", obs.quantization_suspect(np.ones(100)))
check("too few samples to judge is not a false positive",
      not obs.quantization_suspect(np.array([1.0, 2.0])))

print("\nagainst the real archive")

if not ARCHIVE.is_dir():
    print(f"  SKIP  {ARCHIVE} not present on this machine")
else:
    for name, schema in sorted(MODELS.items()):
        d = ARCHIVE / f"{name}_model"
        log = d / MASTER_LOG_FILENAME
        if not log.is_file():
            print(f"  SKIP  no {name} archive at {d}")
            continue
        master = parse_master_log(log, schema.system)
        ids = sorted(master, key=int)
        # A handful spread across the sweep -- this test must stay quick, and
        # test_archive_roundtrip.py is what walks every entry.
        sample = [ids[i] for i in np.linspace(0, len(ids) - 1, 8, dtype=int)]
        rx, ones = [], 0
        for run_id in sample:
            run = plot_data(run_id, d, system=schema.system)
            _, t_end, t_final = obs.pulse_window(run.parameters)
            check(f"{name}/{run_id}: t_start < t_end < t_final",
                  run.Time[0] <= t_end < t_final <= run.Time[-1] + run.dt)
            r = obs.ratio_R(run.Time_dot, run.Jx_dot, run.parameters)
            rx.append(r)
            ones += int(r == 1.0)
            check(f"{name}/{run_id}: R_x is finite and >= 1",
                  np.isfinite(r) and r >= 1.0, f"R_x={r}")
            check(f"{name}/{run_id}: offset >= 0 -- the peak cannot precede t_end",
                  obs.tail_peak_offset(run.Time_dot, run.Jx_dot, run.parameters) >= 0.0)
        # The regression, stated against real data: with the old numerator this
        # was 1.0 for essentially every two_band run.
        check(f"{name}: R_x is not identically 1 across the sample",
              ones < len(sample), f"{ones}/{len(sample)} equal 1.0")
        print(f"  ....  {name}: R_x over {len(sample)} runs spans "
              f"{min(rx):.3g} ... {max(rx):.3g}")

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("ALL PASS")
