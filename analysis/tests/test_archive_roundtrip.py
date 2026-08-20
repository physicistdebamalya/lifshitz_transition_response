"""
Acceptance test for the identification layer.

For every run in the archive: parse its logfile entry, rebuild a parameter
dict from it, push that back through CreateSnippet + find_key_by_substring,
and require the original ID to come back.

This is the test that has to pass before any change to snippets.py or to
print_core_parameters() can be trusted. It is run against the real archive,
not a fixture, because the point is that historical data stays reachable.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_utils import clean_path, parse_master_log          # noqa: E402
from models import MODELS                                     # noqa: E402
from snippets import (CreateSnippet, find_key_by_substring,   # noqa: E402
                      format_number, get_param, get_vector_param)

ARCHIVE = clean_path("/home/debamalya/ComputationData/PostDoc/programming/new/data")

# Pulse keys that may be absent from an entry (category C). Defaults only fill
# the dict so emit_pulse() can run; whichever ones the pulse actually uses are
# always read back from the entry itself.
PULSE_DEFAULTS = dict(sigma=0.1, ntt_sigma=1.0, number_of_cycle=10,
                      duration=5.0, nu=0.0, delay=0.0)


def params_from_entry(entry, schema):
    """Reconstruct the parameter dict a user would have typed for this run."""
    p = {"System": schema.system}
    for k in schema.scalars:
        p[k] = get_param(entry, k, int if k in schema.ints else float)
    for k in schema.vectors:
        p[k] = get_vector_param(entry, k)

    # Category B. Pre-2026-08 entries carry ETPX/ETPY and no TIME_DIVISION;
    # both keys are ignored for matching, so any value here is inert -- but
    # reading the real one when present keeps the reconstruction honest.
    p["TIME_DIVISION"] = get_param(entry, "TIME_DIVISION", int) or 1000
    p["ETP"] = (get_param(entry, "ETP") if "ETP =" in entry
                else get_param(entry, "ETPX"))
    p["initial_time"] = get_param(entry, "initial time")
    p["final_time"] = get_param(entry, "final time")

    for axis in ("X", "Y"):
        p[f"A0{axis}"] = get_param(entry, f"A0{axis}")
        for base, default in PULSE_DEFAULTS.items():
            p[f"{base}{axis}"] = get_param(entry, f"{base}{axis}")
            if p[f"{base}{axis}"] is None:
                p[f"{base}{axis}"] = default

        line = next((l for l in entry.splitlines()
                     if l.startswith(f"pulse_type{axis} =")), None)
        if line is None:                       # channel inactive: A0 == 0
            p[f"pulse_type{axis}"], p[f"pulse_wrapper{axis}"] = "sin^2", "Envelope"
        else:
            val = line.split("=", 1)[1].strip()
            if val.endswith(" with Envelope"):
                p[f"pulse_type{axis}"] = val[:-len(" with Envelope")]
                p[f"pulse_wrapper{axis}"] = "Envelope"
            elif val.endswith(" without Envelope"):
                p[f"pulse_type{axis}"] = val[:-len(" without Envelope")]
                p[f"pulse_wrapper{axis}"] = "NA"
            else:
                raise ValueError(f"unparseable pulse_type line: {line!r}")
    return p


def check_model(model_name, subdir):
    schema = MODELS[model_name]
    log = ARCHIVE / subdir / "logfile.txt"
    if not log.exists():
        print(f"  SKIP {model_name}: no logfile at {log}")
        return 0, 0, []

    entries = parse_master_log(log, schema.system)
    failures = []

    for run_id, entry in entries.items():
        p = params_from_entry(entry, schema)
        snpt = CreateSnippet(p, schema)
        found = find_key_by_substring(entries, snpt, verbose=False)

        if found is None:
            failures.append((run_id, "no match at all"))
        elif run_id not in found:
            failures.append((run_id, f"matched {found} instead"))

        # The precision contract: every scalar in the stored entry must survive
        # a format_number() round trip unchanged. This is what catches a drift
        # between C++ setprecision(6) and Python '.6g'.
        for line in entry.splitlines():
            if "=" not in line or "," in line.split("=", 1)[1]:
                continue
            key, _, val = line.partition("=")
            val = val.strip()
            try:
                float(val)
            except ValueError:
                continue
            if format_number(val) != val:
                failures.append((run_id, f"precision drift: {key.strip()} "
                                         f"stored {val!r}, reformatted {format_number(val)!r}"))

        # data file must actually be there
        if not (ARCHIVE / subdir / f"{run_id}.csv").exists():
            failures.append((run_id, "logfile entry has no .csv"))

    return len(entries), len(entries) - len({f[0] for f in failures}), failures


def main():
    total = passed = 0
    all_failures = []
    for model_name, subdir in (("two_band", "two_band_model"),
                               ("four_band", "four_band_model")):
        n, ok, failures = check_model(model_name, subdir)
        print(f"  {model_name:10s} {ok}/{n} runs round-trip")
        total += n
        passed += ok
        all_failures += [(model_name, *f) for f in failures]

    print(f"\n  TOTAL {passed}/{total}")
    if all_failures:
        print(f"\n  {len(all_failures)} failure(s):")
        for f in all_failures[:20]:
            print("   ", *f)
        return 1
    print("  all archived runs remain reachable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
