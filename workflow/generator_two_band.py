#!/usr/bin/env python3
"""
Define a sweep for the two-band model and submit it.

    python3 workflow/generator_two_band.py --dry-run

Same shape as generator_four_band.py; only BASE and the swept axis differ.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))

import numpy as np                                            # noqa: E402

from data_utils import clean_path                             # noqa: E402
from models import TWO_BAND                                   # noqa: E402
from submit_utils import Submitter        # noqa: E402

ARCHIVE = clean_path("/home/debamalya/ComputationData/PostDoc/programming/new/data/two_band_model")

BASE = dict(
    System=TWO_BAND.system,

    # Production values, read back out of the 61 archived runs -- NOT the
    # debug values left in physics/models/two_band.cpp (N1=31 etc), which are
    # there so the binary runs in seconds on a laptop.
    alpha=1.0, beta=1.0, delta=1.0, m=0.1,
    dephase=0.0,
    N1=1801, N2=1801,
    b1=[4.0, 0.0], b2=[0.0, 4.0], k_origin=[-2.0, -2.0],

    TIME_DIVISION=1000,
    ETP=100.0,

    A0X=0.01, pulse_typeX="sin^2", pulse_wrapperX="Envelope",
    sigmaX=0.1, ntt_sigmaX=1.0, delayX=0.0,
    number_of_cycleX=10, durationX=5.0,

    A0Y=0.01, pulse_typeY="sin^2", pulse_wrapperY="Envelope",
    sigmaY=0.1, ntt_sigmaY=1.0, delayY=0.0,
    number_of_cycleY=10, durationY=5.0,
)

# nu tracks ef rather than being swept independently -- the drive has to follow
# the band feature being probed. Pairs and temperatures reconstructed from the
# archive (23 ef points x 3 temperatures = the 61 completed runs).
EF_NU = [
    (0.7, 0.223), (0.8, 0.255), (0.82, 0.261), (0.84, 0.267),
    (0.86, 0.274), (0.88, 0.28), (0.9, 0.286), (0.92, 0.293),
    (0.94, 0.299), (0.96, 0.305), (0.98, 0.312), (1.0, 0.318),
    (1.02, 0.325), (1.04, 0.331), (1.06, 0.337), (1.08, 0.344),
    (1.1, 0.35), (1.12, 0.356), (1.14, 0.363), (1.16, 0.369),
    (1.2, 0.382), (1.3, 0.414), (1.4, 0.446),
]

TEMPERATURES = [0.0, 0.02, 0.1]


def sweep():
    for T in TEMPERATURES:
        for ef, nu in EF_NU:
            p = dict(BASE)
            p.update(Temperature=T, ef=ef, nuX=nu, nuY=nu)
            yield p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--host", default=None, help="local | param | hpc")
    ap.add_argument("--name", default="twoband")
    ap.add_argument("--archive", default=ARCHIVE)
    args = ap.parse_args()

    s = Submitter("two_band", host=args.host, data_path=args.archive,
                  dry_run=args.dry_run)
    s.job_name = args.name

    points = list(sweep())
    print(f"host={s.host.name}  model=two_band  points={len(points)}", flush=True)

    if not args.dry_run:
        s.acquire_lock()
        s.preparing_run()

    submitted = skipped = failed = 0
    for p in points:
        if s.already_exists(p):
            print(f"  skip T={p['Temperature']:<5} ef={p['ef']:<5} nu={p['nuX']:<6} — already in the archive", flush=True)
            skipped += 1
            continue

        if not args.dry_run:
            s.wait_for_slot()

        run_dir = s.submit(p)
        if run_dir is None:
            failed += 1
            continue
        submitted += 1
        print(f"  {'prepared' if args.dry_run else 'submitted'} T={p['Temperature']:<5} ef={p['ef']:<5} nu={p['nuX']:<6} "
              f"-> {run_dir.name}", flush=True)

    verb = "prepared" if args.dry_run else "submitted"
    print(f"\n{submitted} {verb}, {skipped} already present"
          + (f", {failed} failed" if failed else "") + ".", flush=True)


if __name__ == "__main__":
    main()
