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

    hpc_config.LOCAL/PARAM/HPC  ->  Submitter.__init__  ->  array_script()
                                    (copies all seven)      (writes #SBATCH)

Which profile is used is decided by hpc_config.detect_host(), which looks at
$HOME, or forced with  --host local|param|hpc .

What each one controls
----------------------
partition   #SBATCH --partition -- the SLURM queue every array element runs
            in. It also selects the walltime (from the partition table) and,
            through cpu_budget, how many elements run at once.

qos         #SBATCH --qos -- the quality-of-service class.

nodes       #SBATCH --nodes -- compute nodes per job.

npn         #SBATCH --ntasks-per-node -- MPI ranks per node.
            Total ranks = nodes * npn. ChargeCurrent() splits the k-grid
            across them, so this is the coarse-grained parallelism.

cpt         #SBATCH --cpus-per-task -- cores per rank. array_script() exports it as
            OMP_NUM_THREADS, so it is the OpenMP thread count inside a rank.

max_concurrent_jobs
            Becomes the %K in `#SBATCH --array=0-N%K` -- how many elements
            SLURM lets run at once. plan_partition() derives it as
            cpu_budget // cores_per_job, so it fills the QoS core cap and
            stops. By default (2026-08-23 policy) a single point already
            claims the WHOLE cap -- param medium 240/240 = 1 (5 nodes), hpc
            large 80/80 = 1 (2 nodes) -- so the array runs its points
            strictly one after another, each at that queue's maximum
            possible throughput. SLURM enforces it; nothing of ours has to
            be running.

slurm_user  The account named in status messages, and in preparing_run()'s
            advice when it finds a live array. The live-job check itself now
            queries the recorded array job id rather than the user, which is
            both simpler and immune to a wrong value here.

The one arithmetic rule
-----------------------
npn * cpt must equal the cores per node you are entitled to, because the array script
maps ranks onto cores with

    mpirun --map-by ppr:${SLURM_NTASKS_PER_NODE}:node:PE=${SLURM_CPUS_PER_TASK}

Go over and ranks share cores; go under and cores sit idle. The shipped
profiles, all of which satisfy it:

    host    nodes  npn  cpt   ranks  npn*cpt  total cores  note
    local     1     2    2      2       4          4       laptop; dry runs and smoke tests
    param     5     2   24     10      48        240       PARAM Ananta medium, 5 x 48 cores/node = the whole 240-core cap
    hpc       2     2   20      4      40         80       IITGN HPC2 large, 2 x 40 cores/node = the whole 80-core cap

npn*cpt is cores per node -- the number the arithmetic rule above is about.
total cores is nodes * npn*cpt -- what one job actually holds, all at once,
for its whole run. On both clusters this now equals the full per-user QoS
cap, so max_concurrent_jobs is 1: a single parameter point IS the whole
allocation while it runs (2026-08-23 resource-allocation policy -- see "The
80-core cap" / "The 240-core cap" below).

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

The param queues, and why gpu is out
-------------------------------------
    queue        nodes  cores/node  NUMA  layout   CPU                  walltime
    debug          2        48       2    2 x 24   Xeon Platinum 8268     5 h
    small         18        48       2    2 x 24   Xeon Platinum 8268     3 d
    medium        40        48       2    2 x 24   Xeon Platinum 8268     5 d
    large         50        48       2    2 x 24   Xeon Platinum 8268    10 d
    highmemory    39        48       2    2 x 24   Xeon Platinum 8268    10 d

Every CPU queue here is the same layout -- 2 sockets, hyperthreading off, no
sub-NUMA clustering -- so unlike hpc, moving between them changes walltime
and queue-wait, never npn/cpt. The one queue genuinely left out is gpu (10
nodes, Xeon Gold 6248, 2 x 20 -- the login nodes' own CPU model): it needs a
--gres=gpu request that use_partition() has no field for, and this solver
has no GPU code path, so hpc_config.PARAM_PARTITIONS omits it deliberately
rather than by oversight -- use_partition("gpu") on param raises, same as an
unknown queue name would.

The 80-core cap
---------------
small, medium and large all run under SLURM QoS `fatqos`, whose MaxTRESPU is
cpu=80. That is ONE shared budget of 80 cores across all three, per user --
not 80 each. (`test` is a separate QoS, capped at 8, which is why a whole
node cannot be claimed there.) Going over is not an error: the excess jobs
simply sit pending with reason QOSMaxCpuPerUserLimit until earlier ones end.

The default queue is `large`, not `medium`: medium's 32 cores/node does not
divide 80 evenly (2 nodes = 64/80, 16 idle; 3 nodes = 96, over the cap), so
it cannot reach the cap without either wasting cores or overrunning it.
large's 40 cores/node does -- 2 nodes = 80/80 exactly, zero idle. use_partition()
(with nodes left unset) picks this automatically for whichever queue you
name: it fills the QoS cap with as many WHOLE nodes as fit, and stops.
Naming `medium` explicitly still works, it just tops out at 64 of 80 --
that queue's node size, not a mistake, is what leaves cores idle.

The array's %K is sized to fill that cap and stop. It is a ceiling on this
sweep alone; the QoS cap sits underneath it as the real limit, so if other
jobs of yours are already running, elements simply pend on
QOSMaxCpuPerUserLimit until room appears. That is SLURM enforcing the same
budget the old poll loop used to watch for, with nothing of ours alive. With
the default large layout at 80/80, %K comes out to 1 -- one point holds the
whole cap for the length of its own run, and the next point does not start
until it finishes.

The 240-core cap (param)
-------------------------
debug aside, every param queue above runs under SLURM QoS `iitgn_qos`, whose
MaxTRESPU is cpu=240,gres/gpu=2 -- again ONE shared budget across
small/medium/large/highmemory (and gpu), per user, not 240 each. debug
carries no QoS of its own (QoS=N/A on the partition, unlike the other four);
its cap in PARAM_PARTITIONS is a self-imposed one node (48 cores), not a
real SLURM limit -- debug is the cluster's shared default queue, used by
every account on PARAM Ananta, and claiming both its nodes at once would be
poor etiquette rather than a QoS violation.

use_partition() sizes max_concurrent_jobs the same way it does on hpc, and
here every queue's node size (48) divides the cap (240) exactly, so no queue
switch was needed the way hpc's was: the default queue, medium, at 5 nodes
is already 240 cores/job -- the whole cap, in one job, %K = 1.

How to change them
------------------
1. Permanently, for every sweep on that machine -- the usual case.
   Edit the profile in workflow/hpc_config.py:

       PARAM = HostConfig(
           name="param",
           modules=("gcc/12.4", "openmpi/4.1.8"),
           partition="medium",
           nodes=5, npn=2, cpt=24,        # <-- edit here
           max_concurrent_jobs=1,
       )

   Anything a profile does not name falls back to the HostConfig defaults
   (partition="medium", qos="normal", slurm_user="debamalya.dutta"). So to
   change partition, qos or slurm_user for one machine, add it explicitly to
   that profile -- editing the dataclass default changes all three hosts.

2. For one sweep only, without touching the shared profile.
   Assign to the Submitter in main(), directly after constructing it and
   BEFORE acquire_lock()/preparing_run() -- preparing_run() already reads
   slurm_user, and every array script written later reads the rest:

       s = Submitter("two_band", host=args.host, ...)
       s.job_name = args.name         # the same pattern, already in main()
       s.max_concurrent_jobs = 1      # be polite while the cluster is busy

   Only these seven are copied onto the Submitter. modules, env_exports,
   mpirun_flags and extra_sbatch are read from s.host directly and can only
   be changed in hpc_config.py.

   Switching queues is the override most likely to be done wrong, so on hpc
   it is one call rather than three assignments. Leave nodes unset and it
   auto-fills the QoS cap with whole nodes of whichever queue you name (the
   2026-08-23 default policy); pass nodes= explicitly to get anything else,
   honoured exactly even if it leaves cores idle:

       s.use_partition("small")           # auto: 2 nodes, 4 x 10, 1 concurrent (80/80)
       s.use_partition("medium")          # auto: 2 nodes, 2 x 16, 1 concurrent (64/80 -- 32 does not divide 80)
       s.use_partition("large", nodes=1)  # explicit: 1 node, 2 x 20, 2 concurrent

   Same call, same signature, on param -- it reads hpc_config.PARAM_PARTITIONS
   when host="param", exactly the way it reads HPC_PARTITIONS on hpc. Every
   param queue's node size (48) divides its cap (240) exactly, so auto
   always lands on a full 240/240 fit:

       s.use_partition("large")           # auto: 5 nodes, 2 x 24, 1 concurrent (240/240)
       s.use_partition("small", nodes=2)  # explicit: 2 nodes, 2 x 24, 2 concurrent (96/240)

   gpu is the one param queue this refuses -- see "the param queues" above --
   the same ValueError an unknown queue name on either host raises, not a
   silent fallback to another partition's numbers.

   It sets partition, nodes, npn, cpt and max_concurrent_jobs together, from
   the measured table for the current host (HPC_PARTITIONS or
   PARAM_PARTITIONS). It prints nothing: the one report of what will be
   requested is resource_summary(), printed by main() after every override
   has been applied, so it cannot be contradicted by a later assignment the
   way an early message can.

   Assigning s.partition by itself does NOT bring npn/cpt along: the seven
   attributes are copied off the host profile exactly ONCE, in
   Submitter.__init__, and nothing recomputes them afterwards. s.partition is
   only the string written into #SBATCH --partition and the queue name
   the array's #SBATCH block is built from, so a bare reassignment sends the
   job to large's 40-core nodes still carrying medium's 2 x 16 split.

   Hand-tuning still wins, as long as it comes after. The default policy
   optimises each POINT's throughput; the opposite trade -- more points
   finishing per unit wall-clock, at the cost of each one taking longer --
   is one explicit nodes= away:

       s.use_partition("small", nodes=1)  # 1 node, 4 x 10, 40 cores/job -> 2 concurrent (80/80)

Check before submitting
-----------------------
Print the directives a profile produces without queueing anything (run from
workflow/, or with workflow/ on sys.path):

    >>> from submit_utils import Submitter
    >>> print(Submitter("two_band", host="param").array_script(count=207))

--dry-run is the other half of the check: it builds dir_NNNNN and writes
task.csv into each, then stops before the binary is built, copied, or
anything is submitted. It does not write array_map.txt or sweep_<model>.sh,
so use array_script() above to inspect the #SBATCH block.

What running this actually does
==============================
    python3 workflow/generator_<model>.py

That is the whole command -- there is no `make` step to remember. In order:

  1. asks the resume/begin question if a previous sweep left incomplete
     directories (login node, where there is a terminal to ask on);
  2. builds the model binary ONCE, here, with module purge/load handled
     automatically. ensure_binary() REFUSES to do this inside a SLURM job;
  3. writes one dir_NNNNN per point, each holding a one-row task.csv and a
     copy of the binary, skipping points the archive already has;
  4. submits the whole sweep as ONE `sbatch --array=0-N%K` and exits.

Nothing is left running. SLURM runs at most K elements at a time (K fills
the QoS core cap -- see "The 240-core cap" above), so `squeue` shows one line
for the sweep rather than N. With the default layout, K = 1: each point
claims the entire per-user core budget and runs to completion before the
next one starts, so a sweep of many points finishes them at maximum
per-point speed but strictly serially. If a point cannot finish inside its
queue's walltime (see the queue tables above) it is killed mid-run and does
not resume automatically -- pick a longer-walltime queue with use_partition()
before submitting, or trade some per-point throughput for more concurrency
(see "How to change them" above) if a sweep's total wall-clock time matters
more than any single point's.

Why not a loop, on either machine: IITGN HPC's /etc/motd says "Running any
kind of simulation & jobs are NOT allowed in the login node; JOBS must run
through a scheduler only", and a throttling loop can sit there for days;
while running that loop as its own job holds a core for the sweep's whole
lifetime, out of an allocation this solver is sized to fill exactly. The
array costs neither.

Watch it with `squeue -u <user>`. Each element's own output lands in its run
directory as mpi_logs_<arrayid>_<index>.out; anything that fails before the
element reaches its directory lands in array_logs/.
"""

import argparse
import sys
from pathlib import Path

# Fail here, loudly, rather than as a SyntaxError from deep inside an import.
#
# param's system python3 is 3.6.8 while its venv is 3.12 -- and the venv is
# the one with numpy, which analysis/ needs. A generator started with the
# wrong interpreter is the same shape of bug as the one in
# research_backup/.../hpc_job_failure_modes.md section 5, where a failed
# `source .../activate` did not stop the job and the run silently completed
# on the system Python with different library versions, reporting success.
if sys.version_info < (3, 7):
    raise SystemExit(
        f"this needs Python 3.7+ (dataclasses); found {sys.version.split()[0]} "
        f"at {sys.executable}.\n"
        f"On a cluster, activate the project venv first:\n"
        f"    source ~/.py_env/bin/activate")

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
        description="Define the two_band sweep and submit it as one SLURM "
                    "array job. Builds the binary on the login node, writes a "
                    "task.csv per point, submits once, and exits.")
    ap.add_argument("--dry-run", action="store_true",
                    help="write run dirs and task.csv files, then stop: no "
                         "binary is built or copied and nothing is submitted")
    ap.add_argument("--host", default=None, help="local | param | hpc "
                                                 "(default: autodetect from $HOME)")
    ap.add_argument("--name", default="twoband")
    ap.add_argument("--archive", default=ARCHIVE,
                    help="logfile.txt to check for already-computed points")
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

    if not args.dry_run:
        s.acquire_lock()
        s.preparing_run()
        # Build the binary ONCE, here, on the login node. Every run directory
        # then gets a copy of it plus its own task.csv, and nothing compiles
        # again -- which is what makes one array job possible, and what makes
        # this work on param at all.
        s.ensure_binary()

    prepared, skipped, failed = [], 0, 0
    for p in points:
        if s.already_exists(p):
            print(f"  skip T={p['Temperature']:<5} ef={p['ef']:<5} nu={p['nuX']:<6} dephase={p['dephase']:<5}", flush=True)
            skipped += 1
            continue

        run_dir = s.prepare(p)
        if run_dir is None:
            failed += 1
            continue
        prepared.append(run_dir)
        print(f"  prepared T={p['Temperature']:<5} ef={p['ef']:<5} nu={p['nuX']:<6} dephase={p['dephase']:<5}"
              f" -> {run_dir.name}", flush=True)

    print(f"\n{len(prepared)} prepared, {skipped} already present"
          + (f", {failed} failed" if failed else "") + ".", flush=True)

    # One sbatch for the whole sweep, then this process exits. SLURM runs at
    # most max_concurrent_jobs elements at a time, so nothing has to stay
    # alive here to throttle -- see Submitter.submit_array().
    s.submit_array(prepared)


if __name__ == "__main__":
    main()
