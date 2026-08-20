See Nodes and Core Count
sinfo -N -o "%N %c %m %P"

Run this from your login terminal:

Bash
sinfo -p medium -o "%20N  %10c  %12m  %f %25h"
Or, to get a unique list of every CPU model present in that partition:

Bash
srun -p medium --ntasks-per-node=1 lscpu | grep "Model name" | sort | uniq -c
If the output is 1 line (e.g., 40  Model name: Intel...), the partition is perfectly homogeneous.

If you see multiple lines, you have a "Heterogeneous" partition. In that case, sitting on one node and using -march=native is risky, as the binary might not run on the other nodes.

These Linux commands show you what the CPU actually is and what features it supports:

On the login node
lscpu

and

grep flags /proc/cpuinfo | head -n1
On a compute node

Use srun to run the same commands on a compute node:

srun --partition=medium --ntasks=1 bash -lc "lscpu"

and:

srun --partition=medium --ntasks=1 bash -lc "grep flags /proc/cpuinfo | head -n1"

These will show the detailed architecture and supported instruction flags on the compute node.
The flags line from /proc/cpuinfo tells you exactly which SIMD/AVX/AVX512 features the CPU supports — you can tailor your compile flags accordingly.

# To compile the .cpp file do
**Paramananta**
```
/home/debamalya.dutta/gcc_local/bin/g++ -I/home/debamalya.dutta/cpp_libraries/boost_1_88_0 -I/home/debamalya.dutta/cpp_libraries/eigen-3.4.0 -I/home/debamalya.dutta/cpp_libraries/gsl/include -I. -Wall -Wextra test.cpp pulse_type.cpp -o test -L/home/debamalya.dutta/cpp_libraries/gsl/lib -lgsl -lgslcblas
```
*OpenMP parallelization*
```
/home/debamalya.dutta/gcc_local/bin/g++ -fopenmp -I/home/debamalya.dutta/cpp_libraries/boost_1_89_0 -I/home/debamalya.dutta/cpp_libraries/eigen-5.0.0 -I. -Wall -Wextra test.cpp necessary_functions.cpp -o exe_test
```
**MPI + OpenMP in Paramananta**
```
mpic++ -O3 -march=cascadelake -mtune=cascadelake -mprefer-vector-width=512 -mavx512f -mavx512dq -mavx512bw -mavx512vl -DNDEBUG -std=c++17 -fopenmp -I/home/debamalya.dutta/cpp_libraries/boost_1_89_0 -I/home/debamalya.dutta/cpp_libraries/eigen-5.0.0 -I. -Wall -Wextra -Wno-unknown-pragmas test.cpp necessary_functions.cpp -o exe_test

************************* cpp *******************************
#!/bin/bash
#SBATCH --job-name=hybrid_mpi_omp
#SBATCH --partition=medium
#SBATCH --qos=normal

# PARAM Ananta Hardware: 5 nodes * 48 cores = 240 cores
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=6
#SBATCH --hint=nomultithread

#SBATCH --output=mpi_logs_%j.out
#SBATCH --error=mpi_logs_%j.err

# 1. Load the EXACT same modules you used to compile
module purge
module load gcc
module load openmpi

cd $SLURM_SUBMIT_DIR

# OpenMP Thread Affinity
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# The optimal MPI launch command for InfiniBand & Hybrid PE binding
mpirun --map-by ppr:${SLURM_NTASKS_PER_NODE}:node:PE=${SLURM_CPUS_PER_TASK} ./exe_test

date
*****************************


************************* python *******************************
#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=medium
#SBATCH --qos=normal
#SBATCH --ntasks=200
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --output=mpi_logs_%j.out
#SBATCH --error=mpi_logs_%j.err

# ------------------------------------------------------------------
# 1. CLEAN ENVIRONMENT
# ------------------------------------------------------------------

# Load the exact toolchain used for build
module purge
module load python
module load gcc
module load openmpi

# ------------------------------------------------------------------
# 2. ACTIVATE VENV
# ------------------------------------------------------------------
source ~/.mpi_env/bin/activate 

# ------------------------------------------------------------------
# 4. THREADING OPTIMIZATIONS
# ------------------------------------------------------------------
# Prevent OpenBLAS/Numpy from spawning threads that fight MPI for CPU cores
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ------------------------------------------------------------------
# 5. EXECUTION
# ------------------------------------------------------------------
cd $SLURM_SUBMIT_DIR

mpirun --mca btl ^openib --bind-to core --map-by core python3 test.py

date
*****************************
```
**hpc**
*MPI
```
mpic++ -O3 -mtune=generic -DNDEBUG -std=c++17 -fopenmp -I/iitgn/homedirs/debamalya.dutta/cpp_libraries/boost_1_89_0 -I/iitgn/homedirs/debamalya.dutta/cpp_libraries/eigen-5.0.0 -I. -Wall -Wextra -Wno-unknown-pragmas test.cpp necessary_functions.cpp -o exe_test -L/iitgn/apps/packages/nvhpc/Linux_x86_64/25.3/cuda/12.8/targets/x86_64-linux/lib/stubs -L/iitgn/apps/spack/opt/spack/linux-rocky8-cascadelake/gcc-12.3.0/libfabric-1.22.0-kjorf5ozailt7w3dgotagzdhf5zgpcvd/lib -lcuda -lfabric

************************* cpp *******************************
#!/bin/bash
#SBATCH --job-name=hybrid_mpi_omp
#SBATCH --partition=medium
#SBATCH --qos=normal

# HPC2 Hardware: 2 nodes * 32 cores = 64 cores total (Safest symmetry under 80-core limit)
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread

#SBATCH --output=mpi_logs_%j.out
#SBATCH --error=mpi_logs_%j.err

# 1. Load the exact modules required for HPC2
module purge
module load gcc/12.3.0-gcc-12.3.0-b3lfzmk
module load ucx/1.15.0
module load cuda/12.8
module load openmpi/5.0.8

cd $SLURM_SUBMIT_DIR

LIBFABRIC_PATH="/iitgn/apps/spack/opt/spack/linux-rocky8-cascadelake/gcc-12.3.0/libfabric-1.22.0-kjorf5ozailt7w3dgotagzdhf5zgpcvd/lib"
CUDA_COMPAT_PATH="/iitgn/apps/packages/nvhpc/Linux_x86_64/25.3/cuda/12.8/compat"
export LD_LIBRARY_PATH=${LIBFABRIC_PATH}:${CUDA_COMPAT_PATH}:${LD_LIBRARY_PATH}

# 3. OpenMP Thread Affinity (Crucial for Intel Xeon Gold)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# 4. Execution
mpirun --mca pml ucx --mca btl ^openib --map-by ppr:${SLURM_NTASKS_PER_NODE}:node:PE=${SLURM_CPUS_PER_TASK} ./exe_test
date
*******************************************************************

************************* python *****************************
#!/bin/bash
#SBATCH --job-name=test_mpi
#SBATCH --partition=medium
#SBATCH --qos=normal
#SBATCH --ntasks=80
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --output=mpi_logs_%j.out
#SBATCH --error=mpi_logs_%j.err

# ------------------------------------------------------------------
# 1. CLEAN ENVIRONMENT
# ------------------------------------------------------------------
export LD_LIBRARY_PATH=""
module purge

# Load the exact toolchain used for build
module load python/3.12
module load gcc/12.3.0-gcc-12.3.0-b3lfzmk
module load ucx/1.15.0   # <--- RESTORED: Mandatory for OpenMPI to start
module load openmpi/5.0.8

# ------------------------------------------------------------------
# 2. ACTIVATE VENV
# ------------------------------------------------------------------
source ~/.mpi_env/bin/activate 

# ------------------------------------------------------------------
# 3. SET RUNTIME PATHS (Performance Critical)
# ------------------------------------------------------------------
# A. Spack Libfabric (High-Speed Interconnect)
LIBFABRIC_PATH="/iitgn/apps/spack/opt/spack/linux-rocky8-cascadelake/gcc-12.3.0/libfabric-1.22.0-kjorf5ozailt7w3dgotagzdhf5zgpcvd/lib"

# B. CUDA Compat (Linker Fix)
CUDA_COMPAT_PATH="/iitgn/apps/packages/nvhpc/Linux_x86_64/25.3/cuda/12.8/compat"

# C. UCX Library Path (Backup)
# Sometimes the module sets LD_LIBRARY_PATH, but sometimes it doesn't.
# We add the system path just to be safe based on your previous logs.
UCX_PATH="/opt/ohpc/pub/mpi/ucx-ohpc/1.15.0/lib"

# Prepend all critical paths
export LD_LIBRARY_PATH=${LIBFABRIC_PATH}:${CUDA_COMPAT_PATH}:${UCX_PATH}:${LD_LIBRARY_PATH}

# ------------------------------------------------------------------
# 4. THREADING OPTIMIZATIONS
# ------------------------------------------------------------------
# Prevent OpenBLAS/Numpy from spawning threads that fight MPI for CPU cores
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ------------------------------------------------------------------
# 5. EXECUTION
# ------------------------------------------------------------------
cd $SLURM_SUBMIT_DIR

mpirun --mca pml ^ucx --bind-to core --map-by core python3 test.py

date
**************************************************************
```
**local PC**
```
g++ -I/home/debamalya/cpp_libraries/boost_1_88_0 -I/home/debamalya/cpp_libraries/eigen-3.4.0 -I/home/debamalya/cpp_libraries/gsl/include -I. -Wall -Wextra test.cpp pulse_type.cpp -o test -L/home/debamalya/cpp_libraries/gsl/lib -lgsl -lgslcblas

g++ -I/home/debamalya/cpp_libraries/boost_1_88_0 -I/home/debamalya/cpp_libraries/eigen-3.4.0 -I/home/debamalya/cpp_libraries/gsl/include -I. -Wall -Wextra altermagnet_model.cpp pulse_type.cpp -o altermagnet_model
```
*OpenMP parallelization*
```
g++ -fopenmp -I/home/debamalya/cpp_libraries/boost_1_88_0 -I/home/debamalya/cpp_libraries/eigen-3.4.0 -I/home/debamalya/cpp_libraries/gsl/include -I. -Wall -Wextra test.cpp pulse_type.cpp -o test -L/home/debamalya/cpp_libraries/gsl/lib -lgsl -lgslcblas

g++ -fopenmp -I/home/debamalya/cpp_libraries/boost_1_88_0 -I/home/debamalya/cpp_libraries/eigen-3.4.0 -I/home/debamalya/cpp_libraries/gsl/include -I. -Wall -Wextra altermagnet_model_2.cpp pulse_type.cpp -o exe_altermagnet_model_2 -L/home/debamalya/cpp_libraries/gsl/lib -lgsl -lgslcblas
```
with
```
export LD_LIBRARY_PATH=/home/debamalya/cpp_libraries/gsl/lib:$LD_LIBRARY_PATH
```
*MPI
```
/home/debamalya/cpp_libraries/openmpi-5.0.8/build/bin/mpic++ -O3 -march=native -I/home/debamalya/cpp_libraries/boost_1_89_0 -I/home/debamalya/cpp_libraries/eigen-5.0.0 -I. -Wall -Wextra -Wno-unknown-pragmas test.cpp necessary_functions.cpp -o exe_test


/home/debamalya/cpp_libraries/openmpi-5.0.8/build/bin/mpirun -np 15 ./exe_test > run_stdout.log 2> run_stderr.log
```
# Job submission file
```
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=test
#SBATCH --partition=medium
#SBATCH --qos=normal
export LD_LIBRARY_PATH=/home/debamalya.dutta/cpp_libraries/gsl/lib:/home/debamalya.dutta/gcc_local/lib64:$LD_LIBRARY_PATH
cd $SLURM_SUBMIT_DIR
./exe_test >& test.out
date
```

*OpenMP parallelization*
```
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --job-name=test_parallel
#SBATCH --partition=medium
#SBATCH --qos=normal
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
export LD_LIBRARY_PATH=/home/debamalya.dutta/cpp_libraries/gsl/lib:/home/debamalya.dutta/gcc_local/lib64:$LD_LIBRARY_PATH
cd $SLURM_SUBMIT_DIR
./test >& test.out
date
```

see the link `https://drive.google.com/file/d/1BkjB7FUa6QbILp7IyErPy9CViDfs_fQZ/view?usp=drive_link`

# about *pulse_type*

The `pulse_type.h` and `pulse_type.cpp` helps us to impliment different types of pulses to for the external field. First of all we can use an envelop to the pulse, and that
is implimented using `const string pulse_wrapperX = "envelope"` or `const string pulse_wrapperX = "NA"`, and similarly for `Y` axis. We can use envelope of types `"gaussian"`
or `"sin^2"`, to put an envelope on a `sin` pulse. If no envelope is used, i.e. `"NA"`, then we can choose `"sin^2"` pulse 

**************************************************

**MPI + OpenMP in local PC**
```
/home/debamalya/cpp_libraries/openmpi-5.0.8/build/bin/mpic++ -O3 -march=native -DNDEBUG -std=c++17 -fopenmp -I/home/debamalya/cpp_libraries/boost_1_89_0 -I/home/debamalya/cpp_libraries/eigen-5.0.0 -I. -Wall -Wextra -Wno-unknown-pragmas test1.cpp necessary_functions.cpp -o exe_test


/home/debamalya/cpp_libraries/openmpi-5.0.8/build/bin/mpirun -np 15 ./exe_test > run_stdout.log 2> run_stderr.log
```

g++ -fopenmp -I/home/debamalya/cpp_libraries/boost_1_89_0 -I/home/debamalya/cpp_libraries/eigen-5.0.0 -I/home/debamalya/cpp_libraries/gsl/include/ test_corrected1.cpp pulse_type.cpp -o exe_test


# To compile the tight_binding.cpp file do
```
g++ -fopenmp -I/home/debamalya/cpp_libraries/eigen-5.0.0 -I. -Wall -Wextra tbAM2.cpp necessary_functions.cpp -o exe_test && ./exe_test
```
```
g++ -fopenmp -I/home/debamalya/cpp_libraries/eigen-5.0.0 -I/home/debamalya/cpp_libraries/nanoflann/include -I. -Wall -Wextra tbAM2.cpp necessary_functions.cpp -o exe_test && ./exe_test
```
