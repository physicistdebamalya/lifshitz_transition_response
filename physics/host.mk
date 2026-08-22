# physics/host.mk -- per-host toolchain settings, defined ONCE.
#
# Included by physics/Makefile, which is now the ONLY build path: since
# parameters moved into task.csv (2026-08-22) a run directory holds a copy of
# a prebuilt binary rather than a rewritten source to compile, so
# workflow/Makefile.job is gone.
#
# It is kept as a separate file anyway, because the drift it was extracted to
# stop was real: -march was once corrected in physics/Makefile while
# Makefile.job kept -mtune=generic, and every per-job compile silently stayed
# scalar for months. One definition, one includer, nothing to fall out of
# step -- and a second includer, if one is ever added, starts out correct.
#
#   HOST=local|param|hpc, or leave it to autodetect from $HOME.

HOST ?= auto
ifeq ($(HOST),auto)
  ifneq (,$(findstring /iitgn/homedirs,$(HOME)))
    HOST := hpc
  else ifneq (,$(wildcard /home/debamalya.dutta))
    HOST := param
  else
    HOST := local
  endif
endif

# mpic++ is on PATH on the clusters (after `module load`); locally it lives in
# the hand-built OpenMPI tree, which is not on PATH.
MPI_LOCAL := $(HOME)/cpp_libraries/openmpi-5.0.8/build/bin
CXX       := mpic++
STD       := -std=c++17

# -Wno-maybe-uninitialized is the ONLY warning suppressed, and it disables the
# speculative analysis, not the definite one. GCC splits this in two:
#
#   -Wuninitialized        "is used uninitialized"      -- a real defect
#   -Wmaybe-uninitialized  "may be used uninitialized"  -- a guess
#
# Only the second is turned off. At -O3 the inliner flattens Eigen's product
# kernels far enough that the analysis loses track of which branch initialised
# `aligned`, and reports four warnings from inside Eigen headers that no code
# here can act on. Suppressing it keeps the console readable without hiding
# anything about this project's own sources: a genuine uninitialised read
# still trips -Wuninitialized. Same choice, same flag, as
# Superconductivity/python/Makefile.job.hpc.
WARN   := -Wall -Wextra -Wno-unknown-pragmas -Wno-maybe-uninitialized
COMMON := -O3 -DNDEBUG -fopenmp $(STD) $(WARN)

ifeq ($(HOST),local)
  LIBROOT := $(HOME)/cpp_libraries
  ARCH    := -march=native -mtune=native
  LDEXTRA :=
  CXX     := $(MPI_LOCAL)/mpic++
  MPIRUN  := $(MPI_LOCAL)/mpirun
else ifeq ($(HOST),param)
  LIBROOT := /home/debamalya.dutta/cpp_libraries
  # Unlike hpc, login and compute here do NOT mismatch: verified 2026-08-21 by
  # srun'ing lscpu onto real nodes of every CPU partition. login02/login03 and
  # gpu001-010 are Xeon Gold 6248 (20 cores/socket); small/medium/large/
  # highmemory (cn003-110, hm001-039) are Xeon Platinum 8268 (24 cores/socket).
  # Different SKUs, but identical relevant flags on both:
  #   avx avx2 avx512f avx512dq avx512cd avx512bw avx512vl avx512_vnni
  # so one -march compiled on login runs on every queue without a SIGILL risk
  # -- cascadelake names that shared feature set (and -march=native on login
  # would emit the identical instruction set, since neither SKU has anything
  # beyond it). GCC's cascadelake preset already turns on avx512cd/vl/vnni
  # etc. itself, so listing the -mavx512* flags by hand after it was
  # redundant and has been dropped.
  #
  # -mprefer-vector-width=256: GCC's own default for this target, and
  # measured 2026-08-21 on param's own Platinum 8268 (medium and large, both
  # queues, 113x113 grid): 256 beats 512 by 5.1-5.6%, matching the direction
  # and rough size of hpc's Skylake-SP finding (~6%). No longer provisional
  # -- see research_backup/setup/supercomputing_cluster/param/
  # param_benchmark_flags.md. cascadelake itself (vs -mtune=generic, the
  # unset default) is worth ~1.5x on this workload.
  ARCH    := -march=cascadelake -mtune=cascadelake -mprefer-vector-width=256
  LDEXTRA :=
  MPIRUN  := mpirun
else ifeq ($(HOST),hpc)
  LIBROOT := /iitgn/homedirs/debamalya.dutta/cpp_libraries
  # You compile on the login node and run on cn01-28, and they are NOT the
  # same microarchitecture. login.iitgn.ac.in is a Xeon Gold 5215 -- Cascade
  # Lake, advertises avx512_vnni. medium (cn06-16, Gold 6130) and large
  # (cn18-28, Gold 6148) are Skylake-SP and have no VNNI. So -march=native
  # here emits instructions that SIGILL on the nodes the jobs actually land
  # on, which is why this used to read -mtune=generic -- correct, but it meant
  # baseline x86-64: no AVX, no AVX2, no FMA, on an Eigen RK4 kernel.
  #
  # skylake-avx512 is the common denominator: native on medium and large, and
  # a strict subset of small's Cascade Lake (cn01-05, Gold 6230). Verified by
  # running a binary carrying real zmm instructions on cn06, cn20 and cn01.
  ARCH      := -march=skylake-avx512 -mtune=skylake-avx512
  CUDA_STUB := /iitgn/apps/packages/nvhpc/Linux_x86_64/25.3/cuda/12.8/targets/x86_64-linux/lib/stubs
  LIBFABRIC := /iitgn/apps/spack/opt/spack/linux-rocky8-cascadelake/gcc-12.3.0/libfabric-1.22.0-kjorf5ozailt7w3dgotagzdhf5zgpcvd/lib
  LDEXTRA   := -L$(CUDA_STUB) -L$(LIBFABRIC) -lcuda -lfabric
  MPIRUN    := mpirun
else
  $(error unknown HOST '$(HOST)' -- use local, param or hpc)
endif

EIGEN := $(LIBROOT)/eigen-5.0.0
BOOST := $(LIBROOT)/boost_1_89_0
