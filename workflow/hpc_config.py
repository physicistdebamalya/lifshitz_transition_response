"""
Host profiles: everything that differs between the workstation and the two
clusters, in one place instead of being retyped into each notebook.

Extracted from docs/hpc_notes.md, which is now reference prose rather than the
operational source of truth.
"""

import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


def detect_host():
    """Which machine are we on? Decided from $HOME, same rule as
    physics/host.mk and as Superconductivity's get_hpc_cluster().

    The param test is `/home/debamalya.dutta` exists rather than
    get_hpc_cluster()'s `"/home/" in home`, because the workstation's home is
    /home/debamalya -- which that looser test would call `param`, silently
    handing a laptop the cluster's module list and library paths. Everything
    else about the rule is the same, and both agree on every real cluster.
    """
    home = os.path.expanduser("~")
    if "/iitgn/homedirs" in home:
        return "hpc"
    if Path("/home/debamalya.dutta").exists():
        return "param"
    return "local"


def cores_per_node_from_slurm(partition, timeout=10):
    """Cores/node read out of SLURM instead of a second hand-kept constant.

    `scontrol show partition` reports TotalCPUs and TotalNodes, so the ratio
    is right even if the cluster's node mix changes under us. Only valid
    where a partition's nodes are homogeneous, which is true of small,
    medium and large here (verified: 40, 32 and 40 cores/node uniformly).
    Returns None when SLURM is not reachable -- on a laptop, say -- and the
    caller then keeps the profiled value. Adapted from
    Superconductivity's CompileConfig.autodetect_npn().
    """
    try:
        out = subprocess.run(["scontrol", "show", "partition", partition],
                             capture_output=True, text=True, check=True,
                             timeout=timeout).stdout
        total_cpus = int(re.search(r"TotalCPUs=(\d+)", out).group(1))
        total_nodes = int(re.search(r"TotalNodes=(\d+)", out).group(1))
    except Exception:
        return None
    return total_cpus // total_nodes if total_nodes else None


@dataclass
class HostConfig:
    name: str
    modules: tuple                 # `module load` lines, in order
    partition: str = "medium"
    qos: str = "normal"
    nodes: int = 2
    npn: int = 4                   # tasks per node
    cpt: int = 8                   # cpus per task
    max_concurrent_jobs: int = 5
    slurm_user: str = "debamalya.dutta"
    env_exports: tuple = ()
    mpirun_flags: str = "--mca pml ucx --mca btl ^openib"
    extra_sbatch: tuple = ()
    # SSH endpoint for hpc_transfers.py. Empty on LOCAL, which is never a
    # transfer target -- generator_*.py runs there directly, nothing to push.
    # remote_home mirrors the LIBROOT pattern in physics/Makefile (param:
    # /home/<user>, hpc: /iitgn/homedirs/<user>) -- named here once instead
    # of re-deriving it from the host name wherever a remote path is built.
    hostname: str = ""
    ssh_port: int = 22
    remote_home: str = ""
    # {queue name: PartitionProfile} for hosts whose queues have been
    # profiled, so Submitter.use_partition() can size a job from one name.
    # Empty means "this host has no table" -- use_partition() then refuses
    # rather than applying another machine's numbers. See HPC_PARTITIONS.
    partitions: dict = field(default_factory=dict)
    # Cluster-wide cores this user may hold at once, when a QoS enforces one.
    # Left effectively unbounded where no ceiling has been established, so the
    # throttle falls back to counting jobs rather than inventing a limit.
    cpu_budget: int = 10 ** 9

    def module_block(self):
        return "\n".join(["module purge"] + [f"module load {m}" for m in self.modules])


# ─────────────────────────────────────────────────────────── partitions ──
# Choosing a queue and choosing a core layout are the same decision: the
# queues differ in cores per node AND in NUMA layout, so npn/cpt that are
# right for one are wrong for the next. This table is what lets
# Submitter.use_partition() set all of them from one name.
#
# Two tables: HPC_PARTITIONS for IITGN HPC2, PARAM_PARTITIONS for PARAM
# Ananta (added 2026-08-21, see research_backup/setup/supercomputing_cluster/
# param/). Both are measured the same way -- lscpu/numactl inside a real
# allocation, not read off a datasheet -- and each HostConfig below points
# at its own table, so plan_partition() never applies one cluster's numbers
# to the other's hardware.


@dataclass(frozen=True)
class PartitionProfile:
    """One SLURM queue's hardware, measured rather than assumed.

    Sources: research_backup/setup/hpc_compute_{small,medium,large}.md
    (lscpu + numactl --hardware from inside real allocations on cn03, cn15
    and cn21), plus sinfo / scontrol show partition / sacctmgr show qos on
    login.iitgn.ac.in. Re-verified live 2026-08-21.
    """
    name: str
    cores_per_node: int
    numa_per_node: int      # NUMA domains per node -- sets the rank count
    cpu: str
    walltime: str
    nodes_total: int
    cpu_cap: int            # QoS MaxTRESPU: cores this user may hold at once

    @property
    def cores_per_numa(self):
        return self.cores_per_node // self.numa_per_node


# The cap is a property of the QoS, not the queue: small, medium and large
# all run under QoS `fatqos`, whose MaxTRESPU is cpu=80. That is ONE shared
# 80-core budget across all three, per user -- not 80 each. `test` is a
# separate QoS with cpu=8. Exceeding it is not an error; jobs simply sit
# pending with reason QOSMaxCpuPerUserLimit until earlier ones finish.
FATQOS_CPU_CAP = 80

# PARAM Ananta's equivalent: `scontrol show partition` gives every one of
# small/medium/large/highmemory/gpu QoS=iitgn_qos (debug is the one
# exception -- QoS=N/A, so it draws on nothing here). sacctmgr shows
# iitgn_qos's MaxTRESPU as cpu=240,gres/gpu=2 -- again ONE shared budget
# across all five, not 240 each. Verified live 2026-08-21.
PARAM_QOS_CPU_CAP = 240

HPC_PARTITIONS = {
    # 4 NUMA domains per node: sub-NUMA clustering is on here, unlike the
    # others, so this one wants 4 ranks x 10 threads rather than 2 x 20.
    "small": PartitionProfile("small", 40, 4, "Xeon Gold 6230 @ 2.10GHz",
                              "3-00:00:00", 5, FATQOS_CPU_CAP),
    "medium": PartitionProfile("medium", 32, 2, "Xeon Gold 6130 @ 2.10GHz",
                               "5-00:00:00", 11, FATQOS_CPU_CAP),
    "large": PartitionProfile("large", 40, 2, "Xeon Gold 6148 @ 2.40GHz",
                              "10-00:00:00", 11, FATQOS_CPU_CAP),
    # cn17 alone, 2 x 16 like medium -- and IS medium's silicon: verified
    # 2026-08-21 via srun+lscpu, Xeon Gold 6130 @ 2.10GHz, no VNNI. Its 8-core
    # QoS cap makes it a smoke-test queue only: a whole node cannot be
    # claimed here.
    "test": PartitionProfile("test", 32, 2, "Xeon Gold 6130 @ 2.10GHz",
                             "5:00:00", 1, 8),
}


# PARAM Ananta. Unlike hpc, every CPU partition here is a plain 2-socket
# node with hyperthreading off -- no sub-NUMA clustering to worry about, so
# npn = 2 (one rank per socket) and cpt = cores/socket everywhere. Sources:
# research_backup/setup/supercomputing_cluster/param/param_login_1.md
# (sinfo / scontrol / sacctmgr on login02) plus srun+lscpu on cn003 (small),
# cn023 (medium), cn110 (large) and hm002 (highmemory) -- all four came back
# Xeon Platinum 8268 @ 2.90GHz, 2x24 cores, identical down to the AVX-512
# flag list. debug was not probed directly (it queues too fast to be worth
# occupying) but `sinfo -N` reports the same 2:24:1 socket:core:thread
# signature as the other four, so it is assumed to be the same hardware.
# gpu is the one partition that differs (Xeon Gold 6248, 2x20 -- the same
# model as the login nodes) and is intentionally left out of this table: it
# needs a --gres=gpu request that plan_partition() has no field for, and
# this solver has no GPU code path, so a CPU job there would just be
# occupying a GPU node's cores for no benefit over a CPU-only queue.
PARAM_PARTITIONS = {
    # debug is the cluster default queue and carries no per-user QoS cap of
    # its own (QoS=N/A on the partition) -- only its own 2-node, 96-core
    # ceiling. cpu_cap is set to one node here as etiquette, not a real
    # limit: it is meant for quick interactive checks shared by every user
    # on the cluster, not for occupying both of its nodes at once.
    "debug": PartitionProfile("debug", 48, 2, "Xeon Platinum 8268 @ 2.90GHz",
                              "5:00:00", 2, 48),
    "small": PartitionProfile("small", 48, 2, "Xeon Platinum 8268 @ 2.90GHz",
                              "3-00:00:00", 18, PARAM_QOS_CPU_CAP),
    "medium": PartitionProfile("medium", 48, 2, "Xeon Platinum 8268 @ 2.90GHz",
                               "5-00:00:00", 40, PARAM_QOS_CPU_CAP),
    "large": PartitionProfile("large", 48, 2, "Xeon Platinum 8268 @ 2.90GHz",
                              "10-00:00:00", 50, PARAM_QOS_CPU_CAP),
    "highmemory": PartitionProfile("highmemory", 48, 2,
                                   "Xeon Platinum 8268 @ 2.90GHz",
                                   "10-00:00:00", 39, PARAM_QOS_CPU_CAP),
}


def plan_partition(name, nodes=None, partitions=None):
    """queue name -> the #SBATCH numbers that use that queue best.

    Three rules, the first two consequences of how the solver is parallelised
    (physics/include/necessary_functions.h, ChargeCurrent), the third of the
    2026-08-23 resource-allocation policy: one parameter point should claim
    as much of the cluster as it can, not share it with sibling jobs.

    1. One MPI rank per NUMA domain: npn = NUMA domains per node, cpt =
       cores per domain.

       Measured 2026-08-21 on medium and large (see research_backup/setup/
       supercomputing_cluster/hpc_benchmark_flags.md). The result is more
       modest than the reason originally written here, which claimed a
       socket-straddling rank would pay the interconnect for its J_thread
       buffer. It never gets the chance: OpenMPI REFUSES to bind a rank
       across two sockets ("almost always results in a loss of performance"),
       so 1x32 and 1x40 do not run at all.

       Within the non-straddling layouts the split barely matters -- 2x16 /
       4x8 / 8x4 on medium, and 2x20 / 4x10 / 8x5 on large, agree to within
       0.5%. That is what you would expect from a k-loop with one `omp
       critical` per thread and a single MPI_Reduce at the end: there is
       almost nothing for the split to influence. Only the extreme costs
       anything -- pure MPI (cpt=1) is 7% slower on large, 2% on medium.

       So this is kept as the coarsest split that never spans a socket and
       never degenerates to pure MPI, not because finer splits are slower.

    2. Fill the QoS cap, do not exceed it. per_job * concurrent <= cpu_cap.

    3. `nodes=None` (the default) means "as many WHOLE nodes as fit inside
       the QoS cap" -- cpu_cap // cores_per_node, floored, never below one.
       A single parameter point already saturates a whole node on its own
       (necessary_functions.h block-decomposes one N1*N2 k-grid across every
       MPI rank), so claiming several nodes for it is the same move scaled
       up, not a change of kind -- and because one point is one job, there is
       no sibling job whose slot this could be taking. The result: per_job
       lands as close to the cap as whole-node granularity allows (exact,
       when cores_per_node divides cpu_cap; short by the remainder
       otherwise), and max_concurrent_jobs falls out at 1 whenever it does --
       the array then runs its elements strictly one after another, each at
       the maximum throughput this queue can give a single point. Pass an
       explicit nodes= to ask for anything else; it is honoured exactly, even
       when that leaves cores idle -- this function never overrides a value
       it was actually given.

    Returns a plain dict of Submitter attributes, so the caller decides what
    to do with it -- apply it, print it, or override one field afterwards.
    """
    table = partitions if partitions is not None else HPC_PARTITIONS
    if name not in table:
        raise ValueError(
            f"unknown partition {name!r}; this host has "
            f"{sorted(table)}. Add it to the matching *_PARTITIONS table in "
            f"hpc_config.py (HPC_PARTITIONS for hpc, PARAM_PARTITIONS for "
            f"param), with cores_per_node and numa_per_node taken from an "
            f"actual allocation (lscpu + numactl --hardware), not guessed.")
    if nodes is not None and nodes < 1:
        raise ValueError(f"nodes must be >= 1, got {nodes}")

    p = table[name]
    npn = p.numa_per_node
    cpt = p.cores_per_numa

    # The table is measured, not guessed -- but it was measured once, and the
    # cluster can change. SLURM is the live authority on cores per node, so
    # ask it and say something if the two disagree rather than quietly
    # sizing jobs for hardware that is no longer there. NUMA topology cannot
    # be read this way (scontrol reports sockets, and small runs sub-NUMA
    # clustering at 4 domains across 2 sockets), so that still comes from
    # the table.
    live = cores_per_node_from_slurm(name)
    if live is not None and live != p.cores_per_node:
        print(f"⚠️  {name}: SLURM reports {live} cores/node but hpc_config's "
              f"table says {p.cores_per_node}. Using SLURM's number and "
              f"keeping {p.numa_per_node} NUMA domain(s); re-measure the "
              f"node layout and update HPC_PARTITIONS.", flush=True)
        cpt = max(1, live // npn)

    if nodes is None:
        # cores_per_node here is npn*cpt post-live-check, i.e. what a whole
        # node of THIS queue actually costs right now.
        nodes = max(1, p.cpu_cap // (npn * cpt))

    # `test` is the only queue where a full node exceeds the cap (8 < 32) --
    # this also catches the nodes=None case there, since the formula above
    # gives nodes=1 (cpu_cap // cores_per_node floors to 0, clamped to 1) and
    # a single node still overshoots.
    per_job = nodes * npn * cpt
    if per_job > p.cpu_cap:
        cpt = max(1, p.cpu_cap // (nodes * npn))
        per_job = nodes * npn * cpt

    return dict(partition=p.name,
                nodes=nodes,
                npn=npn,
                cpt=cpt,
                max_concurrent_jobs=max(1, p.cpu_cap // per_job))


PARAM = HostConfig(
    name="param",
    # Pinned, not bare "gcc"/"openmpi": module avail lists gcc/{7.5,9.2,
    # 10.2,12.4} and openmpi/{4.1.4,4.1.8}, and the (D)efault marker on
    # either can move under us. openmpi/4.1.8 already carries a UCX pml
    # component (ompi_info --parsable), so no separate ucx module is needed
    # here the way hpc needs one.
    modules=("gcc/12.4", "openmpi/4.1.8"),
    # These four ARE plan_partition("medium") -- see the HPC profile below
    # for why that's asserted rather than just claimed, and
    # workflow/tests/test_partition_plan.py for where it's checked. One rank
    # per socket, 24 OpenMP threads per rank, on EVERY one of 5 nodes: a
    # single parameter point now claims the full 240-core iitgn_qos cap by
    # itself (2026-08-23 resource-allocation policy -- see plan_partition()'s
    # rule 3). max_concurrent_jobs falls out at 1: with nothing left of the
    # budget for a second element, the array runs its points strictly one
    # after another, each at the maximum throughput medium can give it.
    # 48 divides 240 exactly, so this is a full-budget fit with zero idle
    # cores -- unlike hpc, no queue switch was needed to get there.
    #
    # Measured, not just derived from topology: every factorisation of 48
    # (1x48 through 16x3) ties within 1% on both medium and large -- unlike
    # hpc, param's OpenMPI does not even refuse the single-rank case. 2x24
    # is kept anyway, for the same reason it is the default on hpc: "one
    # rank per NUMA domain" is one rule for both clusters, a tie is not a
    # reason to special-case one of them, and it is the choice that composes
    # correctly across nodes (OpenMP cannot span nodes, so some non-trivial
    # npn is unavoidable regardless). See param_benchmark_flags.md.
    partition="medium",
    nodes=5, npn=2, cpt=24,
    max_concurrent_jobs=1,
    partitions=PARAM_PARTITIONS,
    cpu_budget=PARAM_QOS_CPU_CAP,
    hostname="paramananta.iitgn.ac.in", ssh_port=4422,
    remote_home="/home/debamalya.dutta",
)

HPC = HostConfig(
    name="hpc",
    modules=("gcc/12.3.0-gcc-12.3.0-b3lfzmk", "ucx/1.15.0", "cuda/12.8", "openmpi/5.0.8"),
    # These four ARE plan_partition("large") -- kept literal so the default
    # costs no import-time computation, and asserted equal to it by
    # workflow/tests/test_partition_plan.py. Default queue is `large`, not
    # `medium`: medium's 32 cores/node does not divide the shared 80-core
    # fatqos cap evenly (2 nodes = 64/80, 16 idle; 3 nodes = 96, over the
    # cap), while large's 40 cores/node does -- 2 nodes = 80/80 exactly. One
    # rank per socket, 20 OpenMP threads per rank, on both nodes: a single
    # parameter point claims the full cap by itself (2026-08-23
    # resource-allocation policy -- see plan_partition()'s rule 3), so
    # max_concurrent_jobs is 1 and the array runs its points strictly one
    # after another. large also gives the longest walltime of the three
    # (10 days vs medium's 5), useful headroom now that a point runs to
    # completion alone rather than sharing a queue slot.
    # Change queue and layout together with s.use_partition(...) in the
    # generator rather than editing npn/cpt here.
    partition="large",
    nodes=2, npn=2, cpt=20,
    max_concurrent_jobs=1,
    partitions=HPC_PARTITIONS,
    cpu_budget=FATQOS_CPU_CAP,
    env_exports=(
        'LIBFABRIC_PATH="/iitgn/apps/spack/opt/spack/linux-rocky8-cascadelake/'
        'gcc-12.3.0/libfabric-1.22.0-kjorf5ozailt7w3dgotagzdhf5zgpcvd/lib"',
        'CUDA_COMPAT_PATH="/iitgn/apps/packages/nvhpc/Linux_x86_64/25.3/cuda/12.8/compat"',
        "export LD_LIBRARY_PATH=${LIBFABRIC_PATH}:${CUDA_COMPAT_PATH}:${LD_LIBRARY_PATH}",
    ),
    hostname="10.0.138.33", ssh_port=22,
    remote_home="/iitgn/homedirs/debamalya.dutta",
)

# Not a cluster: for dry runs and small local checks.
LOCAL = HostConfig(
    name="local",
    modules=(),
    nodes=1, npn=2, cpt=2,
    max_concurrent_jobs=1,
    mpirun_flags="--oversubscribe",
)

HOSTS = {h.name: h for h in (PARAM, HPC, LOCAL)}


def get(name=None):
    return HOSTS[name or detect_host()]
