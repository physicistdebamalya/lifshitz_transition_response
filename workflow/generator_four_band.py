#!/usr/bin/env python3
"""
Define a sweep for the four-band (bilayer graphene) model and submit it.

    python3 workflow/generator_four_band.py --dry-run
    python3 workflow/generator_four_band.py

Sweep rules
===========
Any entry in BASE can be a plain scalar or a list:

    scalar, or a one-element list   -> fixed
    a list of more than one value   -> swept

expand_sweep() (workflow/submit_utils.py) does the expansion -- Cartesian
product across every swept key. ef and nu are the one exception: they are
coupled (nu must track the band feature ef probes), so they are walked
together via the EF_NU table below, not independently swept like everything
else in BASE. Ported from Superconductivity's AM3_old.ipynb, section
"1.1 Submit jobs altermagnetic system AM3" -- same scalar/list rule, without
the WILD placeholders, tagged variants or scallerDeltas machinery that
section also has, none of which this project needs (no phase-diagram
workflow here).

N1, N2, b1, b2, k_origin are refused as sweep targets -- they define the
k-grid and Brillouin-zone geometry for this run, not a physical sweep axis.
expand_sweep() raises rather than silently expanding one. A grid-density or
grid-geometry study is a convergence check, not this sweep; give it its own
script if you need one.

"?" is a separate feature, unrelated to sweeping: analysis/snippets.py's
CreateSnippet treats a bare "?" value as a wildcard for an already_exists()
LOOKUP (match any value for that field). It cannot be submitted -- a run
cannot be compiled with dephase = ?; -- so it belongs in a one-off query
dict you build by hand, never in BASE.

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
    hpc       1     2   16      2      32      IITGN HPC2 medium, 32 cores/node

npn is one rank per NUMA domain, cpt the cores inside it. That split is not
arbitrary: ChargeCurrent() is embarrassingly parallel over k, with a single
`omp critical` per thread and one MPI_Reduce at the very end, so neither the
rank count nor the thread count costs communication -- but a rank straddling
two sockets pays the interconnect for every access to its own J_thread
buffer.

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
at 1 node is 32 cores/job, so 2 jobs, 64 of 80. Two things it cannot know:

  - the cap counts YOUR jobs cluster-wide, so anything already running in
    another queue eats into the same 80.
  - wait_for_slot() throttles with `squeue -u <user> -p <partition>`, which
    counts a single queue. Run this sweep on large while other jobs sit on
    medium and the throttle will not see them.

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

       s = Submitter("four_band", host=args.host, ...)
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
   the measured table in hpc_config.HPC_PARTITIONS, and prints what it chose.

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
    >>> print(Submitter("four_band", host="param").job_script())

--dry-run is the other half of the check: it builds dir_NNNNN and rewrites
the model source, then stops before compiling or submitting. It does not
write job.sh, so use job_script() above to inspect the #SBATCH block.
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
from submit_utils import Submitter, expand_sweep  # noqa: E402

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
    """ef and nu are coupled and walked together via EF_NU, not independently
    swept. Every other BASE entry goes through expand_sweep(), so a scalar
    stays fixed (as every one of them is today) and a list would be swept --
    see the module docstring.
    """
    for ef, nu in EF_NU:
        fixed_at_this_ef = dict(BASE)
        fixed_at_this_ef.update(ef=ef, nuX=nu, nuY=nu)
        yield from expand_sweep(fixed_at_this_ef)


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

    # ── Per-sweep overrides (see the module docstring) ────────────────────
    # On hpc, ONE call picks the queue and the core layout that fits it:
    #
    #   s.use_partition("medium")          # 1 node, 2 x 16, 2 jobs  (default)
    #   s.use_partition("large")           # 1 node, 2 x 20, 2 jobs
    #   s.use_partition("large", nodes=2)  # 2 nodes, 2 x 20, 1 job
    #   s.use_partition("test")            # cn17 smoke test, 8-core QoS cap
    #
    # It sets partition, nodes, npn, cpt and max_concurrent_jobs together:
    # one MPI rank per NUMA domain, sized to fill the 80-core QoS cap without
    # going over it. Never set s.partition on its own -- it does not carry
    # npn/cpt with it, so the job lands on the new queue's nodes still split
    # for the old queue's core count.
    #
    # The hpc profile already defaults to medium at 1 node, so use_partition
    # is only needed to change queue or node count. Assignments placed AFTER
    # it win, for anything you want to hand-tune:
    # s.max_concurrent_jobs = 1   # how many jobs may sit in the queue at once
    # s.npn, s.cpt = 4, 10        # only with a node layout you have verified

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
