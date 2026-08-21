"""
Host profiles: everything that differs between the workstation and the two
clusters, in one place instead of being retyped into each notebook.

Extracted from docs/hpc_notes.md, which is now reference prose rather than the
operational source of truth.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path


def detect_host():
    """Which machine are we on? Mirrors the logic in physics/Makefile."""
    home = os.path.expanduser("~")
    if "/iitgn/homedirs" in home:
        return "hpc"
    if Path("/home/debamalya.dutta").exists():
        return "param"
    return "local"


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

    def module_block(self):
        return "\n".join(["module purge"] + [f"module load {m}" for m in self.modules])


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
    nodes=2, npn=8, cpt=4,         # HPC2: 2 nodes x 32 cores, under the 80-core cap
    max_concurrent_jobs=5,
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
