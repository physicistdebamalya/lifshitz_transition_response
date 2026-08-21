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
# IITGN HPC2 only. PARAM Ananta has no equivalent table -- nobody has
# profiled its queues the way research_backup/setup/hpc_compute_*.md
# profiles these, and inventing numbers for it would be worse than the
# current situation, where param's single tuned layout lives in PARAM below.


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

HPC_PARTITIONS = {
    # 4 NUMA domains per node: sub-NUMA clustering is on here, unlike the
    # others, so this one wants 4 ranks x 10 threads rather than 2 x 20.
    "small": PartitionProfile("small", 40, 4, "Xeon Gold 6230 @ 2.10GHz",
                              "3-00:00:00", 5, FATQOS_CPU_CAP),
    "medium": PartitionProfile("medium", 32, 2, "Xeon Gold 6130 @ 2.10GHz",
                               "5-00:00:00", 11, FATQOS_CPU_CAP),
    "large": PartitionProfile("large", 40, 2, "Xeon Gold 6148 @ 2.40GHz",
                              "10-00:00:00", 11, FATQOS_CPU_CAP),
    # cn17 alone, 2 x 16 like medium. Its 8-core QoS cap makes it a smoke-test
    # queue only: a whole node cannot be claimed here.
    "test": PartitionProfile("test", 32, 2, "2 x 16 cores (model unprofiled)",
                             "5:00:00", 1, 8),
}


def plan_partition(name, nodes=1, partitions=None):
    """queue name -> the #SBATCH numbers that use that queue best.

    Two rules, both of them consequences of how the solver is parallelised
    (physics/include/necessary_functions.h, ChargeCurrent):

    1. One MPI rank per NUMA domain. The k-loop is embarrassingly parallel
       with a single `omp critical` reduction per thread and one MPI_Reduce
       at the very end, so neither rank count nor thread count costs
       communication. What does cost is a rank straddling two sockets and
       reaching across the interconnect for its own J_thread buffer. So
       npn = NUMA domains per node, cpt = cores per domain.

    2. Fill the QoS cap, do not exceed it. per_job * concurrent <= cpu_cap.

    Returns a plain dict of Submitter attributes, so the caller decides what
    to do with it -- apply it, print it, or override one field afterwards.
    """
    table = partitions if partitions is not None else HPC_PARTITIONS
    if name not in table:
        raise ValueError(
            f"unknown partition {name!r}; this host has "
            f"{sorted(table)}. Add it to HPC_PARTITIONS in hpc_config.py, "
            f"with cores_per_node and numa_per_node taken from an actual "
            f"allocation (lscpu + numactl --hardware), not guessed.")
    if nodes < 1:
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

    # `test` is the only queue where a full node exceeds the cap (8 < 32).
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
    modules=("gcc", "openmpi"),
    nodes=5, npn=2, cpt=24,        # PARAM Ananta: 5 nodes x 48 cores
    max_concurrent_jobs=25,
    hostname="paramananta.iitgn.ac.in", ssh_port=4422,
    remote_home="/home/debamalya.dutta",
)

HPC = HostConfig(
    name="hpc",
    modules=("gcc/12.3.0-gcc-12.3.0-b3lfzmk", "ucx/1.15.0", "cuda/12.8", "openmpi/5.0.8"),
    # These four ARE plan_partition("medium", nodes=1) -- kept literal so the
    # default costs no import-time computation, and asserted equal to it by
    # workflow/tests/test_partition_plan.py. One node, one rank per socket,
    # 16 OpenMP threads per rank; two such jobs = 64 of the 80-core cap.
    # Change queue and layout together with s.use_partition("large") in the
    # generator rather than editing npn/cpt here.
    partition="medium",
    nodes=1, npn=2, cpt=16,
    max_concurrent_jobs=2,
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
