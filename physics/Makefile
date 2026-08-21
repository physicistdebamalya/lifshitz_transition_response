# physics/Makefile — build the Lindblad solvers.
#
#   make                      # auto-detect host, build every model
#   make HOST=param two_band  # force a host, build one model
#   make check                # verify the toolchain and headers exist
#   make clean
#
# Objects live in build/$(HOST)/ so a -march=native local build can never
# be linked into a cluster binary by accident.

# ---------------------------------------------------------------- host ----
# local : workstation                (~/cpp_libraries)
# param : PARAM Ananta, cascadelake  (AVX-512)
# hpc   : IITGN HPC2, rocky8         (skylake-avx512, needs cuda/libfabric)
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
CXX      := mpic++
STD      := -std=c++17
WARN     := -Wall -Wextra -Wno-unknown-pragmas
COMMON   := -O3 -DNDEBUG -fopenmp $(STD) $(WARN)

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
else ifeq ($(HOST),hpc)
  LIBROOT   := /iitgn/homedirs/debamalya.dutta/cpp_libraries
  # You compile on the login node and run on cn01-28, and they are NOT the
  # same microarchitecture. login.iitgn.ac.in is a Xeon Gold 5215 -- Cascade
  # Lake, advertises avx512_vnni. medium (cn06-16, Gold 6130) and large
  # (cn18-28, Gold 6148) are Skylake-SP and have no VNNI. So -march=native
  # here emits instructions that SIGILL on the nodes the jobs actually land
  # on, which is why this line used to read -mtune=generic -- correct, but it
  # meant baseline x86-64: no AVX, no AVX2, no FMA, on an Eigen RK4 kernel.
  #
  # skylake-avx512 is the common denominator: native on medium and large, and
  # a strict subset of small's Cascade Lake (cn01-05, Gold 6230). Verified
  # accepted by the cluster's gcc 12.3.0.
  #
  # Vector width is deliberately left at GCC's default (256) for this -march.
  # 512-bit ops trigger AVX-512 license-based downclocking on Skylake-SP, and
  # this solver's matrices are 2x2/4x4 -- far too small to earn the lost
  # clock back. Do not copy param's -mprefer-vector-width=512 here; param is
  # a different chip.
  ARCH      := -march=skylake-avx512 -mtune=skylake-avx512
  CUDA_STUB := /iitgn/apps/packages/nvhpc/Linux_x86_64/25.3/cuda/12.8/targets/x86_64-linux/lib/stubs
  LIBFABRIC := /iitgn/apps/spack/opt/spack/linux-rocky8-cascadelake/gcc-12.3.0/libfabric-1.22.0-kjorf5ozailt7w3dgotagzdhf5zgpcvd/lib
  LDEXTRA   := -L$(CUDA_STUB) -L$(LIBFABRIC) -lcuda -lfabric
else
  $(error unknown HOST '$(HOST)' — use local, param or hpc)
endif

EIGEN := $(LIBROOT)/eigen-5.0.0
BOOST := $(LIBROOT)/boost_1_89_0
INC   := -Iinclude -I$(EIGEN) -I$(BOOST)

CXXFLAGS := $(COMMON) $(ARCH) $(INC)
LDFLAGS  := $(LDEXTRA)

# ------------------------------------------------------------- targets ----
BUILD  := build/$(HOST)
MODELS := two_band four_band
BINS   := $(addprefix $(BUILD)/,$(MODELS))
ENGINE := $(BUILD)/necessary_functions.o

.PHONY: all clean check $(MODELS)
all: $(BINS)

# `make two_band` as a shorthand for the full path
$(MODELS): %: $(BUILD)/%

# The engine is a heavy translation unit: compile it once, then every model
# is a link, not a rebuild. This is what keeps a per-job build cheap.
$(ENGINE): src/necessary_functions.cpp include/necessary_functions.h | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/%.o: models/%.cpp include/necessary_functions.h | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/%: $(BUILD)/%.o $(ENGINE)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(BUILD):
	@mkdir -p $(BUILD)

# Fail loudly and specifically rather than emitting 400 lines of template
# errors when a header tree is simply not where the host config says it is.
check:
	@echo "HOST      = $(HOST)"
	@echo "CXX       = $(CXX)"
	@echo "ARCH      = $(ARCH)"
	@command -v $(CXX) >/dev/null || { echo "MISSING: $(CXX) not on PATH (module load gcc openmpi?)"; exit 1; }
	@test -f $(EIGEN)/Eigen/Dense || { echo "MISSING: Eigen at $(EIGEN)"; exit 1; }
	@test -f $(BOOST)/boost/numeric/odeint.hpp || { echo "MISSING: Boost.odeint at $(BOOST)"; exit 1; }
	@echo "OK: toolchain and headers present"

clean:
	@rm -rf build
	@echo "removed build/"

.SECONDARY:
