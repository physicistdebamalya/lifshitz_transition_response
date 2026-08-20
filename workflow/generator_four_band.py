#!/usr/bin/env python3
"""
Define a sweep for the four-band (bilayer graphene) model and submit it.

    python3 workflow/generator_four_band.py --dry-run
    python3 workflow/generator_four_band.py

Sweep rules, following Superconductivity's generators:
    scalar        -> fixed
    list          -> swept
    "?"           -> wildcard, for the dedup query only
"""

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))

import numpy as np                                            # noqa: E402

from data_utils import clean_path                             # noqa: E402
from models import FOUR_BAND                                  # noqa: E402
from submit_utils import Submitter        # noqa: E402

ARCHIVE = clean_path("/home/debamalya/ComputationData/PostDoc/programming/new/data/four_band_model")

# ── the (E_F, nu) pairs the Lifshitz scan walks along ─────────────────────────
# nu is locked to ef rather than swept independently: the probe frequency has
# to track the band feature being probed. This is the ef_nu table that lived in
# four_band.ipynb cell 2, where every value was hand-rounded to 6 digits to
# work around the old format_number. That workaround is no longer needed --
# format_number now matches C++ setprecision(6) -- but the values are kept
# as-is so the sweep still lines up with the 38 runs already archived.
EF_NU = [
    (0.5, 0.159155), (0.55, 0.17507), (0.6, 0.190986), (0.62, 0.197352),
    (0.65, 0.206901), (0.66, 0.210085), (0.68, 0.216451), (0.7, 0.222817),
    (0.71, 0.226), (0.72, 0.229183), (0.73, 0.232366), (0.74, 0.235549),
    (0.75, 0.238732), (0.76, 0.241916), (0.77, 0.245099), (0.78, 0.248282),
    (0.79, 0.251465), (0.8, 0.254648), (0.82, 0.261014), (0.84, 0.26738),
    (0.86, 0.273747), (0.88, 0.280113), (0.9, 0.286479),
]

BASE = dict(
    System=FOUR_BAND.system,

    V=1.5, v=0.5, v3=1.0, gamma1=0.7,
    Temperature=0.001, dephase=0.0,
    N1=2001, N2=2001,
    b1=[12.0, 0.0], b2=[0.0, 12.0], k_origin=[-6.0, -6.0],

    TIME_DIVISION=1000,
    ETP=100.0,

    A0X=0.01, pulse_typeX="sin^2", pulse_wrapperX="Envelope",
    sigmaX=0.1, ntt_sigmaX=1.0, delayX=0.0,
    number_of_cycleX=10, durationX=5.0,

    A0Y=0.01, pulse_typeY="sin^2", pulse_wrapperY="Envelope",
    sigmaY=0.1, ntt_sigmaY=1.0, delayY=0.0,
    number_of_cycleY=10, durationY=5.0,
)


def sweep():
    for ef, nu in EF_NU:
        p = dict(BASE)
        p.update(ef=ef, nuX=nu, nuY=nu)
        yield p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="prepare run dirs and job.sh, stop before sbatch")
    ap.add_argument("--host", default=None, help="local | param | hpc")
    ap.add_argument("--name", default="fourband")
    ap.add_argument("--archive", default=ARCHIVE,
                    help="logfile.txt to check for already-computed points")
    args = ap.parse_args()

    s = Submitter("four_band", host=args.host, data_path=args.archive,
                  dry_run=args.dry_run)
    s.job_name = args.name

    points = list(sweep())
    print(f"host={s.host.name}  model=four_band  points={len(points)}", flush=True)

    if not args.dry_run:
        s.acquire_lock()
        s.preparing_run()

    submitted = skipped = failed = 0
    for p in points:
        if s.already_exists(p):
            print(f"  skip ef={p['ef']:<5} nu={p['nuX']:<9} — already in the archive", flush=True)
            skipped += 1
            continue

        if not args.dry_run:
            s.wait_for_slot()

        run_dir = s.submit(p)
        if run_dir is None:
            failed += 1
            continue
        submitted += 1
        print(f"  {'prepared' if args.dry_run else 'submitted'} ef={p['ef']:<5} nu={p['nuX']:<9} "
              f"-> {run_dir.name}", flush=True)

    verb = "prepared" if args.dry_run else "submitted"
    print(f"\n{submitted} {verb}, {skipped} already present"
          + (f", {failed} failed" if failed else "") + ".", flush=True)


if __name__ == "__main__":
    main()
