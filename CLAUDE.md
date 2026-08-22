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

Run these five. They are the acceptance test, against the real archive:

```bash
python3 analysis/tests/test_archive_roundtrip.py        # expect 99/99
python3 workflow/tests/test_task_schema.py              # expect ALL PASS
python3 workflow/tests/test_sweep_archive_match.py      # expect ALL PASS
python3 workflow/tests/test_partition_plan.py           # expect ALL PASS
python3 -c "import sys; sys.path.insert(0,'analysis'); import models; \
  print([models.verify_cpp_mirror(m) for m in models.MODELS], \
        [models.verify_cpp_task_columns(m) for m in models.MODELS])"  # [[], []] [[], []]
```

The last reads the `.cpp` twice — `print_core_parameters()` (what the binary
writes) and `KNOWN_COLUMNS` (what it reads) — and asserts both still agree with
`analysis/models.py`. C++ cannot import the schema, so these are the only
things standing between you and silent drift in either direction.

`test_sweep_archive_match.py` is the end-to-end one: it runs both production
sweeps' `BASE` through `expand_sweep` → `CreateSnippet` → the subset matcher
against the real 99-run archive and pins the skip counts (two_band 207/0,
four_band 23/23). It replaces `--dry-run`, which did the same matching but
created 207 directories as a side effect.

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

**hpc runs before 2026-08-21 were compiled scalar.** `workflow/Makefile.job`
carried its own stale `-mtune=generic`, so every per-job compile on hpc
missed the vectorized codegen `physics/Makefile` already had — the two files
had drifted. Fixed by extracting `physics/host.mk` as the one definition both
include. Runs from that date on build with `-march=skylake-avx512`. This is
a provenance fact, not a correctness one: outputs agree to ~1e-12 relative
across the flag change (see
`research_backup/setup/supercomputing_cluster/hpc/hpc_benchmark_flags.md`) —
comparing an hpc run's wall-clock time from before this date to one after is
comparing two different code paths, but comparing their physics is fine.

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

## How a parameter point becomes a job

Model parameters are **read at runtime from a one-row `task.csv`** in each run
directory. The binary is built once, on the login node, and copied into every
run directory. The whole sweep is submitted as **one SLURM job array**.

```
sweep dict -> already_exists()? -> dir_NNNNN -> task.csv + exe_run
           -> array_map.txt -> ONE sbatch --array=0-N%K -> directories.txt
```

The user command is exactly `python3 workflow/generator_two_band.py`. It
builds, prepares, submits, and exits in seconds. Nothing stays running.

**Why not compile-time `constexpr` any more** (until 2026-08-22 it was, and a
point was realised by rewriting the source and recompiling): a per-point
compile cannot happen inside a SLURM job on param — `module load gcc/12.4`
there cannot find `features.h` (verified on cn001; the same test succeeds on
hpc's cn17). Compiling on the login node and reading parameters at runtime
removes the per-point compile entirely, and because every point then shares
one binary it also makes the array possible.

**Why an array, and not a submission loop.** IITGN HPC's `/etc/motd`: *"Running
any kind of simulation & jobs are NOT allowed in the login node; JOBS must run
through a scheduler only."* A loop that throttles can sit there for days. The
alternative — dispatching the loop as its own job — holds a core for the whole
sweep, and this solver is sized to fill its allocation exactly. An array costs
neither, and `%K` throttling is enforced by SLURM instead of by a poll loop, so
`squeue` shows one line rather than N. `K = cpu_budget // cores_per_job`, the
same number `max_concurrent_jobs` always was.

**Resource allocation defaults to the whole per-user budget, per point
(2026-08-23).** `nodes` defaults to `None` in `plan_partition()`, meaning "as
many whole nodes of this queue as fit inside the QoS cap" —
`cpu_budget // cores_per_node`, never below one. Since one parameter point
already saturates a whole node by itself (the `N1*N2` k-grid is
block-decomposed across every MPI rank in the job), claiming several nodes
for it is the same move scaled up, and because one point is one job there is
no sibling job whose slot this could take. On param every queue's node size
(48) divides its 240-core cap exactly, so the default queue stays `medium`,
just at 5 nodes instead of 1: 240/240, zero idle. On hpc, `medium`'s 32
cores/node does *not* divide the 80-core cap evenly (2 nodes = 64/80, 3 =
96 > 80), so the default queue changed to `large` (40 cores/node, 2 nodes =
80/80 exactly) — the only way to reach the cap with no idle cores. On both
clusters `K` now comes out to 1: the array runs its points strictly one
after another, each at that queue's maximum possible throughput, rather than
several smaller jobs concurrently. `Submitter.use_partition(name, nodes=None)`
carries the same default when overriding queues; an explicit `nodes=` is
still honoured exactly, even when it leaves cores idle or trades per-point
throughput for sweep-level concurrency.

**Four checks keep the two languages in step**, because C++ cannot import the
schema in `analysis/models.py`:
- `verify_cpp_mirror()` — what the binary WRITES to `parameters.txt`
- `verify_cpp_task_columns()` — what it READS from `task.csv`
- `require_known_columns()` in C++ — an unknown header cell is a hard error,
  never an ignored column, because an ignored column means the run silently
  uses a value nobody chose
- `TaskSpec` has **no member defaults**: every column is required, since
  `task.csv` is always machine-written and a missing column is always a bug

`read_single_task_row()` aborts unless there is exactly **one** data row: one
parameter point expands into an `N1*N2` k-grid that all ranks and threads
cooperate on, so a row is a whole job, not a unit of work to pack.

A failure preparing a point quarantines the directory into `incomplete_runs/`
and the sweep continues.
