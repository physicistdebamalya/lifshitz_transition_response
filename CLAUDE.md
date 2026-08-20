# CLAUDE.md

Ultrafast charge-current response as a probe of a **Lifshitz transition**.
A two-component (x, y) vector-potential pulse drives the system via Peierls
substitution; the per-k Lindblad master equation is integrated with RK4, and
the observable is the charge current J(t).

## Domains

| | |
|---|---|
| `physics/` | C++ solver. Nothing else goes here. |
| `workflow/` | HPC config, compilation, job submission. Imports from `analysis/`. |
| `analysis/` | Data management, identification, post-processing, plotting. Imports from nothing above it. |
| `legacy/` | Frozen. Never built, never imported. |

Dependency direction is one-way: `workflow → analysis`. This is not stylistic —
`already_exists()` must use the same matcher the notebooks use, or "already
computed" and "findable" can drift apart.

## Locked conventions

Each of these is load-bearing. Changing one alone breaks something.

**6 significant figures is a contract.** C++ writes `parameters.txt` under
`std::setprecision(6)`; `analysis/snippets.py:format_number` uses `".6g"`.
Both sides of every comparison go through `format_number`, so matching is
exact string equality with no tolerance. Neither number may change alone.

**Results are written at `setprecision(17)`, deliberately not `std::fixed`.**
`Jy` spans ~1e-18 to ~1e-7; fixed notation would flush the small end to zero.
Runs archived before 2026-08 used the stream default of 6 and their `Jx` is
quantized — see "Known data issues".

**Three field categories** decide what goes in `parameters.txt`:
- *identifying* — in the file, in the match
- *present but ignored* — in the file, in `DEFAULT_IGNORE_PATTERNS`. Read by
  analysis (`t_end` needs `ETP` + `final time`) but meaningless for identity.
- *omitted* — absent entirely, because the run gives them no meaning. When
  `A0Y == 0` there is no `nuY`; writing a stale one would make two physically
  identical runs stop matching each other.

**Matching is subset, never substring.** `query_lines.issubset(entry_lines)`.
The entry may carry extra lines, which is what keeps pre-2026-08 runs (with
`ETPX`/`ETPY`, without `TIME_DIVISION`) reachable from a query built today.

**One `ETP`, not `ETPX`/`ETPY`.** Extra-time-percentage is a simulation knob,
not a per-channel property; two of them made the post-pulse window ambiguous.

**Naming.** Pulse axes are the suffixes `X`/`Y`, never `_x`. Reciprocal
vectors are `b1`/`b2`/`k_origin`, never limits. Grid is `N1`/`N2`, never
`Nx`/`Ny`. Units are natural (ħ = e = a = 1) and never labelled.

**The `System` string is a run-integrity fingerprint.** It is line 1 of
`parameters.txt`, asserted in C++ `main`, and checked by `parse_master_log`.

**Run IDs are random 5-digit numbers**, no zero padding.

## Before changing the identification layer

Run this. It is the acceptance test, against the real archive:

```bash
python3 analysis/tests/test_archive_roundtrip.py     # expect 99/99
python3 -c "import sys; sys.path.insert(0,'analysis'); import models; \
  print([models.verify_cpp_mirror(m) for m in models.MODELS])"   # expect [[], []]
```

The second reads `print_core_parameters()` out of the `.cpp` and asserts it
still agrees with `analysis/models.py`. C++ cannot import the schema, so this
check is the only thing standing between you and silent emitter drift.

## Known data issues

`Jx` in **29 of 38** archived four_band runs is quantized: it was written at 6
significant figures while carrying a large DC offset, so consecutive samples
are identical after rounding and `np.diff(Jx)/dt` measures the rounding step.
Run 10286's `Jx_dot` takes 37 distinct values across 52,359 samples and its
`R_x` came out as exactly 1.25 — the integer ratio 10 steps / 8 steps.
`analysis/observables.py:quantization_suspect` flags these. `Jy` is unaffected
everywhere, and all 61 two_band runs are clean.

**The solver is not bitwise reproducible.** The `#pragma omp critical`
reduction in `ChargeCurrent()` sums thread contributions in arrival order, so
repeated identical runs differ at ~1e-14 relative. Compare numerically, never
by checksum.

## Data locations

Outside the repository, never in git:

```
~/ComputationData/PostDoc/programming/new/data/
    two_band_model/   <ID>.csv + logfile.txt      61 runs
    four_band_model/  <ID>.csv + logfile.txt      38 runs
    images/
~/temp_files_data/                                staging
~/research_backup/PostDoc/programming/new/data/   mirror
```

## Still to do

The task.csv pass: model parameters are still compile-time `constexpr`, so a
parameter point costs a full recompile and `workflow/submit_utils.py` reaches
them by rewriting the source. `build_replacements()` is the single seam where
that gets replaced by a runtime schema-validated `task.csv`.
