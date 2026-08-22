"""
Write and read the one-row task.csv that carries a parameter point to the
binary at runtime.

This is the MECHANISM. The DATA -- which columns exist, in what order, of
what kind -- lives in analysis/models.py's ModelSchema, next to the
parameters.txt schema it is derived from. Putting the column list here too
would recreate the failure this repository has already been burned by twice:
physics/Makefile and workflow/Makefile.job drifting on -march, and
print_core_parameters() drifting from ModelSchema (which is why
verify_cpp_mirror() exists at all). One table, two readers.

Nothing in analysis/ imports this file. analysis/ reads parameters.txt --
the run's OUTPUT record -- and never task.csv, its input. That keeps the
one-way workflow -> analysis dependency intact.

WHY task.csv AT ALL
-------------------
Parameters used to be compile-time constexpr, rewritten into the model
source per point by change_lines() and recompiled inside each run
directory. That cannot work on param: `module load gcc/12.4` inside a SLURM
job cannot find features.h (verified 2026-08-22 on cn001; the same test
succeeds on hpc's cn17). Compiling once on the login node and reading
parameters at runtime removes the per-point compile entirely -- and, because
every point then shares one binary, makes a SLURM job array possible, which
is what lets a whole sweep be submitted with a single sbatch and no
long-lived driver process anywhere.
"""

from pathlib import Path


def format_cell(value, kind):
    """One task.csv cell, at full round-trip precision.

    THE ROUND-TRIP RULE. A double written here is parsed back by std::stod
    and then printed to parameters.txt at setprecision(6), where it is
    compared by EXACT STRING EQUALITY against Python's format_number(".6g")
    (analysis/snippets.py). If the double the C++ holds is not bit-identical
    to the float Python holds, the two six-figure renderings can differ in
    the last place and the run becomes permanently unfindable --
    already_exists() would then resubmit it on every sweep, forever.

    repr() gives the shortest decimal string that round-trips exactly, and
    strtod is correctly rounded, so repr -> stod is the identity on IEEE-754
    doubles. Do NOT use "%.6g" here (lossy -- that precision belongs in
    parameters.txt, not in the input) and do NOT use "%.17g" (correct but
    unreadable: 0.01 becomes 0.01000000000000000021).

    This is the SAME guarantee the old mechanism relied on: change_lines()
    wrote str(float) into a C++ literal and g++ parsed it. The parser moved
    from g++ to std::stod; the doubles did not change.
    """
    if kind is str:
        s = str(value)
        if "," in s or "\n" in s or "\r" in s:
            raise ValueError(
                f"cell {s!r} contains a comma or newline. The C++ splitter is "
                f"naive by design (no quoting, no escapes), so a cell may not "
                f"contain either.")
        return s

    if kind is int:
        f = float(value)
        if f != int(f):
            # build_replacements() used to do a bare int(), so 10.7 silently
            # became 10 and the run was labelled with a value nobody chose.
            raise ValueError(
                f"{value!r} is not an integer, but this column is declared "
                f"int in ModelSchema.ints. Refusing to truncate silently.")
        return str(int(f))

    f = float(value)
    s = repr(f)
    # The contract above, executed rather than argued. Cheap, and it fires at
    # write time -- where the value is still traceable to a sweep dict --
    # rather than as an unfindable run months later.
    assert float(s) == f, f"repr() did not round-trip for {value!r}"
    return s


def _lookup(p, column, schema):
    """One task.csv column's value out of a parameter dict.

    Vector columns are flattened in the header (`b1_0`, `b1_1`) but live in
    the dict as a two-element sequence under the bare name (`p["b1"]`), which
    is the spelling every generator's BASE, CreateSnippet and the archive
    round-trip test all use. Unflatten here rather than asking callers to
    change shape.
    """
    if column.endswith(("_0", "_1")):
        base, _, index = column.rpartition("_")
        if base in schema.vectors:
            vector = p.get(base)
            if vector is None:
                return None
            return vector[int(index)]
    return p.get(column)


def write_task_csv(path, p, schema):
    """One parameter point -> the one-row task.csv its binary reads.

    Raises KeyError naming EVERY missing column at once, rather than the
    first. This is the replacement for change_lines()'s "raise on an
    unmatched prefix", and it exists for the identical reason: a silent miss
    means a job runs with a parameter nobody chose and lands in the archive
    labelled as something it is not.

    Exactly one row. In this project one parameter point expands into an
    N1*N2 k-grid that ALL MPI ranks and ALL OpenMP threads cooperate on, so a
    row is a whole job -- not a unit of work to pack alongside others. The
    C++ reader enforces the same invariant from its side.
    """
    columns = schema.task_columns
    cells, missing = [], []
    for column in columns:
        value = _lookup(p, column, schema)
        if value is None:
            missing.append(column)
            continue
        try:
            cells.append(format_cell(value, schema.column_kind(column)))
        except ValueError as e:
            raise ValueError(f"{schema.name}: column {column!r}: {e}") from None

    if missing:
        raise KeyError(
            f"{schema.name}: task.csv column(s) missing from the parameter "
            f"dict: {missing}")

    path = Path(path)
    path.write_text(",".join(columns) + "\n" + ",".join(cells) + "\n")
    return path


def read_task_csv(path, schema):
    """Parse a task.csv back into a parameter dict.

    For tests and A/B tooling only -- the production reader is the C++ one.
    Vector columns are re-nested, so the result compares directly against the
    dict that was written.
    """
    path = Path(path)
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    if len(lines) != 2:
        raise ValueError(
            f"{path}: expected a header and exactly one data row, found "
            f"{len(lines)} non-empty line(s).")

    header = [c.strip() for c in lines[0].split(",")]
    cells = [c.strip() for c in lines[1].split(",")]
    if len(header) != len(cells):
        raise ValueError(
            f"{path}: header has {len(header)} columns but the data row has "
            f"{len(cells)}. A cell most likely contains a comma.")

    p = {}
    for column, cell in zip(header, cells):
        kind = schema.column_kind(column)
        value = cell if kind is str else kind(cell)
        if column.endswith(("_0", "_1")):
            base, _, index = column.rpartition("_")
            if base in schema.vectors:
                p.setdefault(base, [None, None])[int(index)] = value
                continue
        p[column] = value
    return p
