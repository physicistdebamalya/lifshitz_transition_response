# physics/Makefile — build the Lindblad solvers.
#
#   make                      # auto-detect host, build every model
#   make HOST=param two_band  # force a host, build one model
#   make check                # verify the toolchain and headers exist
#   make clean
#
# Objects live in build/$(HOST)/ so a -march=native local build can never
# be linked into a cluster binary by accident.
#
# Every toolchain setting -- host detection, CXX, warnings, -march, library
# roots -- comes from host.mk, the single definition of the toolchain.
# Do not redefine any of them here; that duplication is what let the two
# files drift apart before.

include $(dir $(lastword $(MAKEFILE_LIST)))host.mk

INC      := -Iinclude -I$(EIGEN) -I$(BOOST)
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
