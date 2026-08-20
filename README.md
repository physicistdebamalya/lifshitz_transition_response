# Lifshitz-transition response

Ultrafast charge-current response as a probe of a Lifshitz transition. Two
models share one solver engine:

- **two-band** — `H = (α k_x² − δ)σ_x + β k_y σ_y + m σ_z`
- **four-band** — bilayer graphene (McCann–Fal'ko), where trigonal warping
  from `v3` produces the Lifshitz feature

A two-component pulse enters through Peierls substitution; the per-k Lindblad
master equation is integrated with RK4 (Boost.odeint + Eigen), MPI over
k-blocks and OpenMP within each rank. The headline result is
`R = max(J̇)/J̇(t_end)` against `E_F`, measured on the post-pulse tail.

## Layout

```
physics/     C++ solver          include/ src/ models/ tests/ Makefile
workflow/    HPC + submission    hpc_config.py submit_utils.py generator_*.py
analysis/    data + plotting     models.py snippets.py loaders.py
                                 observables.py plotstyle.py notebooks/
mathematica/ scratch notebooks
legacy/      frozen, never built
docs/        cluster reconnaissance notes
```

## Build

```bash
make -C physics check           # verify toolchain and headers
make -C physics all             # both models -> physics/build/$HOST/
make -C physics HOST=param all  # force a host: local | param | hpc
```

The host is detected from `$HOME`. Objects are kept in per-host directories so
a `-march=native` local object can never be linked into a cluster binary.

## Run one point locally

```bash
mkdir -p /tmp/run && cd /tmp/run
OMP_NUM_THREADS=4 mpirun -np 2 ~/programming/new/physics/build/local/two_band
```

Writes `data.csv` and `parameters.txt` into the working directory.

## Submit a sweep

```bash
python3 workflow/generator_four_band.py --dry-run    # inspect, submit nothing
python3 workflow/generator_two_band.py  --dry-run
```

Points already present in the archive are skipped automatically, using the
same matcher the analysis side uses to find data.

## Find and plot archived data

```python
import sys; sys.path.insert(0, "analysis")
from data_utils import clean_path, parse_master_log
from models import FOUR_BAND
from snippets import CreateSnippet, find_key_by_substring
from loaders import plot_data
from observables import ratio_R, quantization_suspect

AR = clean_path("~/ComputationData/PostDoc/programming/new/data/four_band_model")
entries = parse_master_log(AR / "logfile.txt", FOUR_BAND.system)

params = dict(System=FOUR_BAND.system, V=1.5, v=0.5, v3=1.0, gamma1=0.7,
              Temperature=0.001, ef="?", dephase=0.0, N1=2001, N2=2001, ...)
ids = find_key_by_substring(entries, CreateSnippet(params, FOUR_BAND))

d = plot_data(ids[0], AR)
if not quantization_suspect(d.Jx):
    print(ratio_R(d.Time_dot, d.Jx_dot, d.parameters))
```

Any parameter set to `"?"` is wildcarded and matches every archived value.

## Tests

```bash
python3 analysis/tests/test_archive_roundtrip.py    # 99/99 archived runs
```

See `CLAUDE.md` for the conventions this project depends on, and for known
issues with archived data.
