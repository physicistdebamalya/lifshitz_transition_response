# Design record

Why the architecture is the way it is. Dated; append, do not rewrite.

## 2026-08-20 — restructure

Adopted from `Superconductivity` (the verified reference) and the domain split
of `two_component_BH`. Deliberately **not** adopted: the phase-diagram and
sweep-merge machinery. This project's unit is one run = one parameter point =
one time series, which is what the `<ID>.csv` + `logfile.txt` archive already
assumes. There is no self-consistency loop and no free-energy minimisation.

### The identification layer is the hinge

`parameters.txt` was designed backwards from how data is consumed, not from
what parameters happen to exist. The chain is:

```
sweep dict → parameters.txt → logfile.txt entry → run ID → <ID>.csv → J̇, R
```

Two consumers depend on it, and Superconductivity's `already_exists()` showed
they must be the same code:

- analysis asks *where is the data for these parameters?*
- workflow asks *have I already run these parameters?*

If those ever use different notions of equality, the submitter recomputes data
that exists, or the notebook cannot find data that does.

### Why fields are omitted rather than written as `n/a`

The initial proposal was to always emit every field with `n/a` where
inapplicable, on the grounds that "absent" should never mean "forgotten".
That is wrong here. The matcher tests subset containment on lines. If `A0Y = 0`
and `nuY` were still emitted, two runs that are physically identical — both
have the y channel off — but were submitted with different leftover `nuY`
values would produce different entries and stop matching each other. A false
negative on your own data.

Omission makes the entry a canonical function of the run's *meaningful*
content. The problem was never the rule; it was that the rule was hand-written
twice, in C++ `print_core_parameters()` and in Python `CreateSnippet`, with
nothing checking they agreed. `analysis/models.py:verify_cpp_mirror()` now
reads the C++ source and asserts it.

### Why ETP and `final time` cannot be dropped

They look derived. They are the analysis window:

```python
t_end = t_start + (t_final - t_start) / (1 + ETP/100)   # pulse switch-off
R     = max(J̇) / J̇(t_end)      measured on (t_end, t_final]
```

Without both, the project's headline result is not computable from the
archive. They are *present but non-identifying*: read by analysis, excluded
from matching. That third category is what made the redundancy question
tractable — the answer was never "keep" or "drop" but "keep, and exclude".

### Backward compatibility, measured rather than assumed

Before changing anything, the archive was checked:

```
two_band  61 entries, 26 key-lines each, all "sin^2 with Envelope", A0=0: none
four_band 38 entries, 26 key-lines each, all "sin^2 with Envelope", A0=0: none
```

Perfectly homogeneous — the conditional branches had never been exercised in
production. So formalising the rule cost nothing for existing data. The two
new keys (`TIME_DIVISION`, and `ETP` replacing `ETPX`/`ETPY`) sit in
`DEFAULT_IGNORE_PATTERNS`, and because matching is subset-based, old entries
carrying `ETPX`/`ETPY` and lacking `TIME_DIVISION` still resolve. Verified:
all 99 runs round-trip.

### Defects found and fixed

- `format_number` returned the raw Python repr, so `1/(2π)` produced 17 digits
  and silently failed to match C++'s 6. The hand-rounded `ef_nu` table in
  `four_band.ipynb` was the workaround; it is no longer needed.
- Matching was a substring test, requiring query lines contiguous and in
  order. Inserting any field broke it.
- `resolve()` was absent, so ignoring a vector key like `b1` produced the
  prefix `b1 =`, which matches nothing — the ignore silently did nothing.
- `ignore_patterns` was aliased and appended to, so re-running a notebook cell
  accumulated wildcard names from the previous run.
- `data_utils.py` used `re` without importing it; `main_delete` crashed.
- `cpp/test.cpp` was deleted from disk while all four submitters compiled it —
  the pipeline could not run at all. Rescued to `legacy/test.cpp`.
- Results were written at the stream default of 6 significant figures. See
  below.

### The precision finding

`Jx` in 29 of 38 four_band runs carries a large DC offset with small
variation, so at 6 significant figures consecutive samples round to the same
value. `np.diff(Jx)/dt` then measures the rounding step, not the current. Run
10286: `Jx` steps along a ladder of exactly 1e-7, `Jx_dot` takes 37 distinct
values across 52,359 samples, and `R_x` evaluates to exactly 1.25 — the ratio
of 10 quantization steps to 8.

Results are now written at `setprecision(17)`. This deviates from AM3.cpp,
which uses `std::fixed` with `setprecision(12)`; `Jy` here spans ~1e-18 to
~1e-7 and fixed notation would flush the small end to zero.

`Jy` is unaffected everywhere — it spans decades, so `%g` preserved relative
precision. All 61 two_band runs are clean.

### Open

- The solver is not bitwise reproducible: the `#pragma omp critical` reduction
  in `ChargeCurrent()` sums in thread-arrival order, giving ~1e-14 relative
  spread between identical runs. Fixable with per-thread slots summed in a
  fixed order; not done, as the engine was out of scope for this pass.
### task.csv, after all — and why the original reasoning was wrong (2026-08-22)

Superconductivity's AM5 generation moved from rewriting source to a runtime
schema-validated `task.csv`, and adopting that here was the initial proposal.
**It was rejected, and that decision has now been reversed.** The original
argument is preserved here because the way it failed is the useful part:

> this project's unit is one run = one parameter point = one time series, so a
> task file would carry exactly one row, and the recompile it would save is a
> link against a prebuilt engine object rather than a full rebuild. The cost
> it removes is small; the indirection it adds is not.

Every sentence of that is true **on hpc**. It is false on param, and the
difference is not one of degree: `module load gcc/12.4` inside a SLURM job on
param cannot find `features.h`, so a per-point compile there does not cost a
little more — it does not work at all (verified 2026-08-22 by submitting probe
jobs to both clusters; hpc's cn17 compiled cleanly, param's cn001 did not).
The decision was made by weighing a cost, when the real question was whether
the operation was available. One cluster was never tested.

Two things followed that the cost argument could not have predicted:

- **A per-point compile is not relocatable, so it had to stop existing.** The
  binary is built once on the login node and copied into each run directory.
- **Every point then shares one binary, which is what an array job requires.**
  A whole sweep is now a single `sbatch --array=0-N%K`, which removed the
  submission loop entirely — and with it the question of where that loop
  should run, which had no good answer (a login-node loop contradicts IITGN
  HPC's stated policy; a driver job holds a core for days from an allocation
  this solver is sized to fill exactly).

`change_lines()` / `build_replacements()` are retired. The discipline they
carried is not: `write_task_csv()` raises naming every missing column,
`require_known_columns()` in C++ makes an unrecognised header a hard error
rather than an ignored one, and `TaskSpec` has no member defaults, so there
is nothing for a missed column to quietly fall back to.
