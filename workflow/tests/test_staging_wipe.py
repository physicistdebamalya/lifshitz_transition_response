#!/usr/bin/env python3
"""Check the download path: per-model source, flat staging, a gate, and no
silent passes anywhere along it.

    python3 workflow/tests/test_staging_wipe.py     # expect ALL PASS

Regression test for a real miss on 2026-08-25. A two_band download from param
printed

    staging : /home/debamalya/temp_files_data/two_band_model
    staging : ... (does not exist yet; nothing to clear)

and went ahead. The wipe had been written to target a per-model subdirectory,
which no download had ever created -- so it truthfully reported an empty
nothing while a previous batch sat untouched in the flat root one level up, and
then downloaded a new batch into that same flat root beside it. Two batches,
one directories.txt, no warning. The gate was there; it was aimed at the wrong
directory. A gate that cannot see the data it exists to protect is not a gate.

The fix was to drop the LOCAL subdirectory -- staging is one flat directory
holding one batch (Superconductivity's LOCAL_DESTINATION shape) and the wipe
targets that root -- and to put the per-model split on the REMOTE side instead,
where the runs are actually produced. What is pinned here:

  * the download destination and the wipe target are the same directory;
  * the model selects the remote directory read, so it is required, not a label;
  * the guards that keep a mistyped STAGING_ROOT from emptying $HOME;
  * the ordering that makes emptying staging cost two separate answers;
  * and the strict paths -- a plan naming an absent source, an unknown transfer
    mode, and a download that arrives unarchivable all raise instead of
    printing a line and continuing to DONE.

No cluster is contacted: run_transfers() is stubbed, and every strictness check
below is reached before connect(). Nothing outside a scratch directory is ever
a delete target -- STAGING_ROOT is monkeypatched for every test that can
delete, so the real ~/temp_files_data is never touched.
"""

import builtins
import contextlib
import io
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hpc_transfers as H  # noqa: E402

REAL_STAGING = H.STAGING_ROOT
REAL_ARCHIVE = H.ARCHIVE_ROOT
MODEL = "two_band"
SYSTEM = H.MODELS[MODEL].system
failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}" + (f"   {detail}" if detail else ""))
        failures.append(name)


def raises(name, fn, expect):
    """Assert fn() raises SystemExit whose message contains `expect`."""
    try:
        fn()
    except SystemExit as exc:
        check(name, expect in str(exc), str(exc))
        return
    check(name, False, "no SystemExit raised")


def answers(*seq):
    """Feed the prompts. Calling with no arguments asserts that nothing is
    asked at all, which is the point for every guard below: a refusal must
    happen before the user is given a chance to type DELETE."""
    it = iter(seq)

    def fake(prompt=""):
        try:
            return next(it)
        except StopIteration:
            raise AssertionError(f"unexpected prompt: {prompt!r}")
    builtins.input = fake
    H.T.timed_input = lambda prompt, timeout=30: fake(prompt)


def populate(root, n=3, system=SYSTEM, listed=None):
    """A staging root as it looks after a download -- plus two entries from
    another project, because ~/temp_files_data is shared and the wipe has to
    name them rather than silently take them."""
    root.mkdir(parents=True, exist_ok=True)
    names = [f"dir_{10000 + i}" for i in range(n)]
    for name in names:
        d = root / name
        d.mkdir(exist_ok=True)
        (d / "data.csv").write_text("t,Jx\n0,0\n")
        (d / "parameters.txt").write_text(f"{system}\nA0X 1.0\n")
    (root / "directories.txt").write_text(
        "\n".join(f"{i + 1}. {name}"
                  for i, name in enumerate(names if listed is None else listed)) + "\n")
    (root / "AM5_w0").mkdir(exist_ok=True)
    (root / "initial_conditions.csv").write_text("x\n")
    return names


print("\nthe download lands where the wipe looks, and reads one model's directory")

down = H.transfers_down("param", MODEL)
check("one download entry", len(down) == 1, len(down))
check("its destination IS the staging root", down[0]["dst"] == REAL_STAGING,
      str(down[0]["dst"]))
check("no per-model subdirectory in the LOCAL destination",
      "_model" not in str(down[0]["dst"]), str(down[0]["dst"]))
check("the REMOTE source is that model's own directory",
      down[0]["src"] == f"{H.remote_root('param')}/{MODEL}/", down[0]["src"])
check("the trailing / means 'contents of', so runs land flat in staging",
      down[0]["src"].endswith("/"), down[0]["src"])
check("four_band reads a different remote directory",
      H.transfers_down("param", "four_band")[0]["src"]
      != H.transfers_down("param", MODEL)[0]["src"])
check("transfers_down requires a model -- it names the directory read",
      H.transfers_down.__code__.co_argcount == 2)
raises("transfers_down rejects an unknown model",
       lambda: H.transfers_down("param", "three_band"), "unknown model")
check("staging_dir() is gone", not hasattr(H, "staging_dir"))
check("the pushed-up archive is not pulled back into staging",
      "archive" in H.DOWNLOAD_EXCLUDES)
check("the per-directory binary is still excluded", "exe_run" in H.DOWNLOAD_EXCLUDES)

print("\nupload still carries what the cluster needs")

# transfers_up() warns on stdout about a model with no registry yet. Capture it
# so the omission can be asserted rather than merely hoped for, then re-print it
# so it is still on screen where an operator reads this run.
buffered = io.StringIO()
with contextlib.redirect_stdout(buffered):
    up = H.transfers_up("param")
warned = buffered.getvalue()
if warned.strip():
    print("".join(f"  ....  {line}\n" for line in warned.splitlines() if line.strip()), end="")

registries = [e for e in up if str(e.get("src", "")).endswith("logfile.txt")]
archived = [m for m in sorted(H.MODELS) if (H.archive_dir(m) / "logfile.txt").exists()]
missing = [m for m in sorted(H.MODELS) if m not in archived]
# Not len(MODELS): a model being reran has nothing archived yet and so has no
# registry -- four_band, since its earlier runs were retired to data_r/ on
# 2026-08-26. Every registry that DOES exist must travel, because that is what
# already_exists() reads on the cluster, and one left behind resubmits a whole
# archived sweep.
check("every archive registry that exists on disk travels",
      len(registries) == len(archived), f"{len(registries)} of {len(archived)}")
check("a model with no registry is named out loud, never dropped silently",
      all(m in warned for m in missing), warned.strip() or "(nothing printed)")
check("at least one registry travels -- an empty archive root is a wrong path, "
      "not a state this project is ever in", bool(registries), str(REAL_ARCHIVE))
check("registries are read from the archive, never written to it",
      all(str(REAL_ARCHIVE) in str(e["src"]) for e in registries))
check("host.mk is pushed", any(str(e.get("src", "")).endswith("host.mk") for e in up))
check("no upload destination escapes the remote project root",
      all(e["dst"].startswith(H.remote_root("param")) for e in up))
check("build products never travel -- a -march=native object must not be "
      "linked on a cluster", "build" in H.UPLOAD_EXCLUDES)
check("no upload source is inside physics/build",
      not any("build" in Path(str(e.get("src", ""))).parts for e in up))
mkdirs = {e["dst"] for e in up if e["mode"] == "mkdir"}
check("the push creates one run directory per model, so the cd cannot fail",
      mkdirs == {H.remote_model_root("param", m) for m in H.MODELS}, sorted(mkdirs))
check("every workflow file named in the constant is actually pushed",
      all(any(str(e.get("src", "")).endswith(n) for e in up) for n in H.WORKFLOW_FILES))
check("every analysis file named in the constant is actually pushed",
      all(any(str(e.get("src", "")).endswith(n) for e in up) for n in H.ANALYSIS_FILES))
raises("a source named in the plan but absent on disk stops the push",
       lambda: H.require_local(H.WORKFLOW_DIR / "not_a_real_file.py"), "does not exist")

print("\nno silent passes in the transfer driver -- checked before connect()")

raises("an unknown transfer mode raises, and moves nothing first",
       lambda: H.T.run_transfers("param", [{"mode": "sync", "src": "a", "dst": "b"}]),
       "Nothing was transferred")
raises("the whole list is validated up front, not entry by entry",
       lambda: H.T.run_transfers("param", [{"mode": "upload", "src": "a", "dst": "b"},
                                           {"mode": "rsync", "src": "c", "dst": "d"}]),
       "entry 2 of 2")
raises("an upload entry with no src raises",
       lambda: H.T.run_transfers("param", [{"mode": "upload", "dst": "b"}]), "has no 'src'")
raises("a missing local upload source raises instead of printing 'Missing:'",
       lambda: H.T.upload("/nonexistent/definitely/not/here.txt", "/tmp/x"),
       "does not exist")
check("mkdir needs only a dst", "src" not in
      {k for e in up if e["mode"] == "mkdir" for k in e})

scratch = Path(tempfile.mkdtemp(prefix="staging_wipe_test_",
                                dir=str(Path(__file__).resolve().parent)))
try:
    H.STAGING_ROOT = scratch

    print("\nthe gate")

    absent = scratch / "not_yet"
    answers()
    H.clear_staging(absent)
    check("an absent root is a no-op, and is not created", not absent.exists())

    answers()
    H.clear_staging(scratch)
    check("an empty root is a no-op", scratch.is_dir())

    populate(scratch)
    before = sorted(p.name for p in scratch.iterdir())

    answers("no")
    raises("declining aborts the download, not just the deletion",
           lambda: H.clear_staging(scratch), "aborted")
    check("declining deletes nothing", sorted(p.name for p in scratch.iterdir()) == before)

    answers("delete")
    raises("lower-case 'delete' does not count",
           lambda: H.clear_staging(scratch), "aborted")
    check("still nothing deleted", sorted(p.name for p in scratch.iterdir()) == before)

    answers("DELETE")
    n = H.clear_staging(scratch)
    check("DELETE empties the root, including the other project's entries",
          scratch.is_dir() and not list(scratch.iterdir()))
    check("and reports how many items went", n == len(before), f"{n} vs {len(before)}")

    print("\nguards -- a mistyped STAGING_ROOT must refuse before it asks")

    for label, target in [("$HOME", Path.home()), ("/", Path("/")), ("/etc", Path("/etc")),
                          ("/var", Path("/var")), ("/tmp", Path("/tmp")),
                          ("/usr", Path("/usr"))]:
        answers()
        raises(f"refuses {label}", lambda t=target: H.clear_staging(t),
               "protected directory")

    # An ancestor of the cwd means STAGING_ROOT is pointing at source.
    answers()
    raises("refuses a root containing the cwd",
           lambda: H.clear_staging(Path.cwd().parent), "current working directory")

    afile = scratch / "afile"
    afile.write_text("x\n")
    answers()
    raises("refuses a non-directory", lambda: H.clear_staging(afile), "not a directory")
    afile.unlink()

    print("\nthe batch is verified after the download, not trusted")

    for p in scratch.iterdir():
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    raises("a download with no directories.txt is a failure, not a success",
           lambda: H.verify_staged_batch(MODEL), "no directories.txt")

    (scratch / "directories.txt").write_text("\n")
    raises("a listing naming no runs is a failure",
           lambda: H.verify_staged_batch(MODEL), "naming no runs")

    (scratch / "directories.txt").write_text("1. dir_10000\ngarbage line\n")
    raises("a line sync_simulation_data() could not split is a failure here first",
           lambda: H.verify_staged_batch(MODEL), "unparseable")

    populate(scratch, n=3, listed=["dir_10000", "dir_10001", "dir_10002", "dir_19999"])
    raises("a run named in the listing but not downloaded is a failure",
           lambda: H.verify_staged_batch(MODEL), "not archivable")

    populate(scratch, n=3)
    (scratch / "dir_10001" / "data.csv").unlink()
    raises("a run without the files the archive step reads is a failure",
           lambda: H.verify_staged_batch(MODEL), "not archivable")

    populate(scratch, n=3, system=H.MODELS["four_band"].system)
    raises("four_band data pulled from the two_band directory is a failure",
           lambda: H.verify_staged_batch(MODEL), "not archivable")

    populate(scratch, n=3)
    check("a complete batch verifies, and reports its run count",
          H.verify_staged_batch(MODEL) == 3)

    (scratch / "dir_19998").mkdir()
    (scratch / "dir_19998" / "data.csv").write_text("t\n")
    check("an unlisted run directory does not fail the batch (it is warned about)",
          H.verify_staged_batch(MODEL) == 3)
    shutil.rmtree(scratch / "dir_19998")

    print("\nordering -- the wipe happens after Proceed and before the transfer")

    real_run_transfers = H.T.run_transfers
    calls = []

    def stub(*a, **k):
        """Record what staging looked like when the transfer began -- the
        ordering assertion -- then leave the batch a real download would have
        left, so verify_staged_batch() runs on real input rather than being
        stubbed out along with the network."""
        calls.append(sorted(p.name for p in scratch.iterdir()))
        populate(scratch)

    H.T.run_transfers = stub

    populate(scratch)
    before = sorted(p.name for p in scratch.iterdir())
    answers("n")
    raises("declining Proceed aborts", lambda: H.main(["param", "down", MODEL]), "aborted")
    check("declining Proceed transfers nothing", not calls)
    check("declining Proceed keeps the previous batch -- a cancelled download "
          "must not cost it", sorted(p.name for p in scratch.iterdir()) == before)

    answers("y", "DELETE")
    rc = H.main(["param", "down", MODEL])
    check("an accepted download returns 0", rc == 0, repr(rc))
    check("staging was already empty when the transfer began", calls == [[]], repr(calls))

    # The model is no longer optional: it selects the remote directory, so a
    # download with none given must ASK, and an unanswered prompt must stop.
    populate(scratch)
    calls.clear()
    answers("2", "y", "DELETE")          # ask_choice -> sorted(MODELS)[1], Proceed, DELETE
    rc = H.main(["param", "down"])
    check("a download with no model on the command line asks for one",
          rc == 0 and sorted(H.MODELS)[1] == MODEL, repr(rc))
    check("and still wiped before transferring", calls == [[]], repr(calls))

    populate(scratch)
    before = sorted(p.name for p in scratch.iterdir())
    calls.clear()
    answers("")
    raises("an empty answer to the model prompt is not a skip",
           lambda: H.main(["param", "down"]), "invalid choice")
    check("an unanswered model prompt deletes nothing",
          sorted(p.name for p in scratch.iterdir()) == before)
    check("an unanswered model prompt transfers nothing", not calls)

    answers()
    raises("an unknown model is rejected",
           lambda: H.main(["param", "down", "three_band"]), "unknown model")
    check("an unknown model deletes nothing",
          sorted(p.name for p in scratch.iterdir()) == before)

    H.T.run_transfers = real_run_transfers

finally:
    H.STAGING_ROOT = REAL_STAGING
    shutil.rmtree(scratch, ignore_errors=True)

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("ALL PASS")
