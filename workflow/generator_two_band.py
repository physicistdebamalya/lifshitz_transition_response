#!/usr/bin/env python3
"""
Define a sweep for the two-band model and submit it.

    python3 workflow/generator_two_band.py --dry-run

Same shape as generator_four_band.py; only BASE and the swept axis differ.

Sweep rules
===========
Any entry in BASE can be a plain scalar or a list:

    scalar, or a one-element list   -> fixed
    a list of more than one value   -> swept

expand_sweep() (workflow/submit_utils.py) does the expansion -- Cartesian
product across every swept key. ef and nu are the one exception: they are
coupled (nu must track the band feature ef probes), so they are walked
together via the EF_NU table below, not independently swept like everything
else in BASE.

N1, N2, b1, b2, k_origin are refused as sweep targets -- they define the
k-grid and Brillouin-zone geometry for this run, not a physical sweep axis.
expand_sweep() raises rather than silently expanding one. A grid-density or
grid-geometry study is a convergence check, not this sweep; give it its own
script if you need one.

Ported from Superconductivity's AM3_old.ipynb, section "1.1 Submit jobs
altermagnetic system AM3" -- same scalar/list rule, without the WILD
placeholders, tagged variants or scallerDeltas machinery that section also
has, none of which this project needs (no phase-diagram workflow here).

HPC resource parameters
=======================
partition, qos, nodes, npn, cpt, max_concurrent_jobs, slurm_user

None of these are set in this file. They describe the machine, not the sweep,
so they live in exactly one place -- the HostConfig profiles in
workflow/hpc_config.py -- and reach a job like this:

    hpc_config.LOCAL/PARAM/HPC  ->  Submitter.__init__  ->  job_script()
                                    (copies all seven)      (writes #SBATCH)

Which profile is used is decided by hpc_config.detect_host(), which looks at
$HOME, or forced with  --host local|param|hpc .

What each one controls
----------------------
partition   #SBATCH --partition -- the SLURM queue. Used twice: it names the
            queue the job goes to, and it is the queue wait_for_slot() counts
            against. If it does not name the queue the jobs actually land in,
            the throttle is counting the wrong jobs.

qos         #SBATCH --qos -- the quality-of-service class.

nodes       #SBATCH --nodes -- compute nodes per job.

npn         #SBATCH --ntasks-per-node -- MPI ranks per node.
            Total ranks = nodes * npn. ChargeCurrent() splits the k-grid
            across them, so this is the coarse-grained parallelism.

cpt         #SBATCH --cpus-per-task -- cores per rank. job.sh exports it as
            OMP_NUM_THREADS, so it is the OpenMP thread count inside a rank.

max_concurrent_jobs
            Not a SLURM directive. wait_for_slot() blocks while
            `squeue -u <slurm_user> -p <partition>` already shows this many
            jobs, so a long sweep trickles in instead of flooding the queue.

slurm_user  The account passed to `squeue -u`, by both wait_for_slot() and
            preparing_run()'s live-job cross-check. A wrong value is silent
            and harmful: squeue returns nothing, so the throttle never
            engages and preparing_run() cannot tell a still-running job from
            a dead one.

The one arithmetic rule
-----------------------
npn * cpt must equal the cores per node you are entitled to, because job.sh
maps ranks onto cores with

    mpirun --map-by ppr:${SLURM_NTASKS_PER_NODE}:node:PE=${SLURM_CPUS_PER_TASK}

Go over and ranks share cores; go under and cores sit idle. The shipped
profiles, all of which satisfy it:

    host    nodes  npn  cpt   ranks  npn*cpt   note
    local     1     2    2      2       4      laptop; dry runs and smoke tests
    param     5     2   24     10      48      PARAM Ananta, 48 cores/node
    hpc       2     8    4     16      32      IITGN HPC2, under the 80-core cap

How to change them
------------------
1. Permanently, for every sweep on that machine -- the usual case.
   Edit the profile in workflow/hpc_config.py:

       PARAM = HostConfig(
           name="param",
           modules=("gcc", "openmpi"),
           nodes=5, npn=2, cpt=24,        # <-- edit here
           max_concurrent_jobs=25,
       )

   Anything a profile does not name falls back to the HostConfig defaults
   (partition="medium", qos="normal", slurm_user="debamalya.dutta"). So to
   change partition, qos or slurm_user for one machine, add it explicitly to
   that profile -- editing the dataclass default changes all three hosts.

2. For one sweep only, without touching the shared profile.
   Assign to the Submitter in main(), directly after constructing it and
   BEFORE acquire_lock()/preparing_run() -- preparing_run() already reads
   slurm_user, and every job.sh written later reads the rest:

       s = Submitter("two_band", host=args.host, ...)
       s.job_name = args.name         # the same pattern, already in main()
       s.nodes = 10                   # big grid: twice the usual nodes
       s.max_concurrent_jobs = 4      # be polite while the cluster is busy

   Only these seven are copied onto the Submitter. modules, env_exports,
   mpirun_flags and extra_sbatch are read from s.host directly and can only
   be changed in hpc_config.py.

Check before submitting
-----------------------
Print the directives a profile produces without queueing anything (run from
workflow/, or with workflow/ on sys.path):

    >>> from submit_utils import Submitter
    >>> print(Submitter("two_band", host="param").job_script())

--dry-run is the other half of the check: it builds dir_NNNNN and rewrites
the model source, then stops before compiling or submitting. It does not
write job.sh, so use job_script() above to inspect the #SBATCH block.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))

import numpy as np                                            # noqa: E402

from data_utils import clean_path                             # noqa: E402
from models import TWO_BAND                                   # noqa: E402
from submit_utils import Submitter, expand_sweep  # noqa: E402

ARCHIVE = clean_path("/home/debamalya/ComputationData/PostDoc/programming/new/data/two_band_model")

BASE = dict(
    System=TWO_BAND.system,

    # Production values, read back out of the 61 archived runs -- NOT the
    # debug values left in physics/models/two_band.cpp (N1=31 etc), which are
    # there so the binary runs in seconds on a laptop.
    alpha=1.0,
    beta=1.0,
    delta=1.0,
    m=0.1,
    dephase=[0.01, 0.1, 0.2],   # swept -- expand_sweep() below Cartesian-
                                # multiplies this against every (ef, nu) pair
    Temperature=[0.0, 0.02, 0.1],   # swept; was the standalone TEMPERATURES
                                     # list, folded in now that BASE itself
                                     # can carry a swept key

    # -------------------------------------------

    TIME_DIVISION=1000,
    ETP=100.0,

    A0X=0.01, 
    pulse_typeX="sin^2", 
    pulse_wrapperX="Envelope",
    sigmaX=0.1, 
    ntt_sigmaX=1.0, 
    delayX=0.0,
    number_of_cycleX=10, 
    durationX=5.0,

    A0Y=0.01, 
    pulse_typeY="sin^2", 
    pulse_wrapperY="Envelope",
    sigmaY=0.1, 
    ntt_sigmaY=1.0, 
    delayY=0.0,
    number_of_cycleY=10, 
    durationY=5.0,

    # -------------------------------------------

    N1=1801,
    N2=1801,
    b1=[4.0, 0.0], b2=[0.0, 4.0], k_origin=[-2.0, -2.0]
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


def sweep():
    """ef and nu are coupled (nu must track the band feature ef probes) and
    are walked together via EF_NU, not independently swept. Every other
    BASE entry goes through expand_sweep(), so a scalar there stays fixed
    and a list -- like Temperature or dephase above -- is swept and
    Cartesian-multiplied against each (ef, nu) pair.
    """
    for ef, nu in EF_NU:
        fixed_at_this_ef = dict(BASE)
        fixed_at_this_ef.update(ef=ef, nuX=nu, nuY=nu)
        yield from expand_sweep(fixed_at_this_ef)


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

    # Per-sweep overrides -- uncomment to override the host profile for this
    # sweep only (see the module docstring). Keep npn * cpt equal to the
    # node's core count: param 48 (e.g. 4 x 12), hpc 32 (e.g. 4 x 8).
    s.nodes = 1                 # param only -- hpc's 80-core cap allows 2
    # s.npn = 4                   # MPI ranks per node
    # s.cpt = 12                  # OMP threads per rank; param 4 * 12 = 48
    # s.max_concurrent_jobs = 10  # how many jobs may sit in the queue at once

    points = list(sweep())
    print(f"host={s.host.name}  model=two_band  points={len(points)}", flush=True)

    if not args.dry_run:
        s.acquire_lock()
        s.preparing_run()

    submitted = skipped = failed = 0
    for p in points:
        if s.already_exists(p):
            print(f"  skip T={p['Temperature']:<5} ef={p['ef']:<5} nu={p['nuX']:<6} dephase={p['dephase']:<5} "
                 f"— already in the archive", flush=True)
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
              f"dephase={p['dephase']:<5} -> {run_dir.name}", flush=True)

    verb = "prepared" if args.dry_run else "submitted"
    print(f"\n{submitted} {verb}, {skipped} already present"
          + (f", {failed} failed" if failed else "") + ".", flush=True)


if __name__ == "__main__":
    main()
