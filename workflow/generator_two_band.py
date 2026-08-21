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
            Not a SLURM directive. How much of the shared core budget this
            one sweep is willing to occupy: wait_for_slot() blocks until
            `cores already running + this job's cores` fits inside
            min(cpu_budget, max_concurrent_jobs * cores_per_job).
            Cores, not job count, because cores are what the QoS enforces --
            see "The 80-core cap" below.

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
    hpc       1     2   16      2      32      IITGN HPC2 medium, 32 cores/node

npn is one rank per NUMA domain, cpt the cores inside it. Measured 2026-08-21
(research_backup/setup/supercomputing_cluster/hpc_benchmark_flags.md): the
choice is safe and optimal, but not uniquely so. OpenMPI refuses outright to
bind a rank across two sockets, so 1x32 and 1x40 never run; among the layouts
that do, 2x16 / 4x8 / 8x4 (medium) and 2x20 / 4x10 / 8x5 (large) agree to
within 0.5%, and only pure MPI (cpt=1) is measurably worse -- 7% on large.
This is the coarsest split that avoids both extremes.

The hpc queues, and why the layout differs per queue
---------------------------------------------------
    queue    nodes  cores/node  NUMA  layout   CPU                 walltime
    small      5        40       4    4 x 10   Xeon 6230 2.10GHz     3 d
    medium    11        32       2    2 x 16   Xeon 6130 2.10GHz     5 d
    large     11        40       2    2 x 20   Xeon 6148 2.40GHz    10 d
    test       1        32       2    2 x  4   cn17 only             5 h

small runs with sub-NUMA clustering on -- 4 NUMA domains per node, not 2 --
so its layout is 4 x 10 rather than large's 2 x 20, despite both having 40
cores. That is precisely why partition and npn/cpt must move together, and
why use_partition() exists.

The 80-core cap
---------------
small, medium and large all run under SLURM QoS `fatqos`, whose MaxTRESPU is
cpu=80. That is ONE shared budget of 80 cores across all three, per user --
not 80 each. (`test` is a separate QoS, capped at 8, which is why a whole
node cannot be claimed there.) Going over is not an error: the excess jobs
simply sit pending with reason QOSMaxCpuPerUserLimit until earlier ones end.

use_partition() sizes max_concurrent_jobs to fill that cap and stop -- medium
at 1 node is 32 cores/job, so 2 jobs, 64 of 80.

wait_for_slot() measures the same thing SLURM does: cores this user has
RUNNING, cluster-wide (`squeue -u <user> -t running -o %C`), not jobs in one
partition. So work already running on medium correctly eats into the budget
of a sweep being submitted to large. Pending jobs hold no cores and are not
counted -- which is exactly why SLURM lets them queue.

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
       s.max_concurrent_jobs = 1      # be polite while the cluster is busy

   Only these seven are copied onto the Submitter. modules, env_exports,
   mpirun_flags and extra_sbatch are read from s.host directly and can only
   be changed in hpc_config.py.

   Switching queues is the override most likely to be done wrong, so on hpc
   it is one call rather than three assignments:

       s.use_partition("large")           # 1 node,  2 x 20, 2 concurrent
       s.use_partition("large", nodes=2)  # 2 nodes, 2 x 20, 1 concurrent

   It sets partition, nodes, npn, cpt and max_concurrent_jobs together, from
   the measured table in hpc_config.HPC_PARTITIONS. It prints nothing: the
   one report of what will be requested is resource_summary(), printed by
   main() after every override has been applied, so it cannot be contradicted
   by a later assignment the way an early message can.

   Assigning s.partition by itself does NOT bring npn/cpt along: the seven
   attributes are copied off the host profile exactly ONCE, in
   Submitter.__init__, and nothing recomputes them afterwards. s.partition is
   only the string written into #SBATCH --partition and the queue name
   wait_for_slot() polls with `squeue -p`, so a bare reassignment sends the
   job to large's 40-core nodes still carrying medium's 2 x 16 split.

   Hand-tuning still wins, as long as it comes after:

       s.use_partition("large")
       s.max_concurrent_jobs = 1      # keep large's 2 x 20, queue one job

   param has no partition table -- its queues have never been profiled the
   way research_backup/setup/hpc_compute_*.md profiles hpc's -- so
   use_partition() refuses there rather than applying another machine's
   numbers. Set partition/npn/cpt by hand for param.

Check before submitting
-----------------------
Print the directives a profile produces without queueing anything (run from
workflow/, or with workflow/ on sys.path):

    >>> from submit_utils import Submitter
    >>> print(Submitter("two_band", host="param").job_script())

--dry-run is the other half of the check: it builds dir_NNNNN and rewrites
the model source, then stops before compiling, submitting or dispatching. It
does not write job.sh, so use job_script() above to inspect the #SBATCH block.

Where the loop runs
===================
On param or hpc, `python3 workflow/generator_*.py` does the interactive part
here -- the resume/begin question, and one `make` for the shared engine
object -- then sbatches the sweep loop itself onto a compute node and
returns. That loop invokes g++ on an Eigen translation unit once per
parameter point, which is not work a shared login node should be doing; the
pattern is taken from Superconductivity's generator_AM5.py. Module purge and
load happen automatically around every compile, on both sides.

Follow it with `tail -f driver_<model>_<jobid>.out`. Use --here to keep the
loop in the foreground instead (debugging), and --dry-run never dispatches.
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
    ap = argparse.ArgumentParser(
        description="Define the two_band sweep and submit it. On a cluster the "
                    "submission loop is dispatched to a compute node; you only "
                    "ever run `python3 generator_two_band.py`.")
    ap.add_argument("--dry-run", action="store_true",
                    help="prepare run dirs and rewrite sources, stop before "
                         "compiling, sbatch or dispatch")
    ap.add_argument("--host", default=None, help="local | param | hpc "
                                                 "(default: autodetect from $HOME)")
    ap.add_argument("--name", default="twoband")
    ap.add_argument("--archive", default=ARCHIVE,
                    help="logfile.txt to check for already-computed points")
    ap.add_argument("--here", action="store_true",
                    help="run the loop in this process instead of dispatching "
                         "it to a compute node (debugging)")
    args = ap.parse_args()

    s = Submitter("two_band", host=args.host, data_path=args.archive,
                  dry_run=args.dry_run)
    s.job_name = args.name

    # ── Per-sweep overrides ───────────────────────────────────────────────
    # On hpc one call picks the queue AND the layout that fits it. It sets
    # partition, nodes, npn, cpt and max_concurrent_jobs together -- one MPI
    # rank per NUMA domain, sized to the QoS core cap. Never set s.partition
    # alone; it does not carry npn/cpt with it.
    #
    # s.use_partition("large")           # 1 node,  2 x 20, 2 concurrent
    # s.use_partition("large", nodes=2)  # 2 nodes, 2 x 20, 1 concurrent
    #
    # The hpc profile already defaults to medium at 1 node, so this is only
    # needed to change queue or node count. Anything assigned after it wins:
    # s.max_concurrent_jobs = 1     # occupy less of the shared core budget
    # s.npn, s.cpt = 4, 10          # only with a layout you have verified

    points = list(sweep())
    print(s.resource_summary(), flush=True)
    print(f"model=two_band  points={len(points)}", flush=True)

    # On a cluster the loop belongs on a compute node: it runs one g++ over an
    # Eigen translation unit per point. The resume/begin question and the
    # one-off engine build happen here first, while there is still a terminal
    # to ask on and before anything is queued.
    if s.should_dispatch() and not args.here:
        s.preparing_run()
        s.ensure_engine()
        s.dispatch_to_compute_node(__file__, sys.argv[1:])
        return

    if not args.dry_run:
        s.acquire_lock()
        s.preparing_run()

    submitted = skipped = failed = 0
    for p in points:
        if s.already_exists(p):
            print(f"  skip T={p['Temperature']:<5} ef={p['ef']:<5} nu={p['nuX']:<6} dephase={p['dephase']:<5}", flush=True)
            skipped += 1
            continue

        if not args.dry_run:
            s.wait_for_slot()

        run_dir = s.submit(p)
        if run_dir is None:
            failed += 1
            continue
        submitted += 1
        print(f"  {'prepared' if args.dry_run else 'submitted'} T={p['Temperature']:<5} ef={p['ef']:<5} nu={p['nuX']:<6} dephase={p['dephase']:<5}"
              f" -> {run_dir.name}", flush=True)

    verb = "prepared" if args.dry_run else "submitted"
    print(f"\n{submitted} {verb}, {skipped} already present"
          + (f", {failed} failed" if failed else "") + ".", flush=True)


if __name__ == "__main__":
    main()
