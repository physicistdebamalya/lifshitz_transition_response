# physics/host.mk -- per-host toolchain settings, defined ONCE.
#
# Included by physics/Makefile (which builds the shared engine object) and by
# workflow/Makefile.job (which compiles one rewritten model per run dir).
# Those two objects are linked into the same binary, so their flags must
# agree. They used to be two hand-maintained copies, and they drifted: -march
# was corrected in physics/Makefile while Makefile.job kept -mtune=generic,
# so every per-job compile silently stayed scalar. One definition removes the
# failure mode rather than documenting it.
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
  ARCH    := -march=cascadelake -mtune=cascadelake -mprefer-vector-width=512 \
             -mavx512f -mavx512dq -mavx512bw -mavx512vl
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
