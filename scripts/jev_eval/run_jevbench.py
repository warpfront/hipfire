# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Nick Woolmer
# hipfire — see LICENSE and NOTICE in the project root.

"""Run Running-Dolphins/jev-bench against local hipfire serve.

jevbench.py writes predictions/results under its module ROOT, which would
overwrite the committed Jev baselines — so ROOT is redirected to --out. The
dataset cache is copied from the clone's data/ directory (if it exists) into
out/data — cache misses then write only under --out, never into the clone.
"""
import argparse
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path

from raw_guard import check_raw

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=11435)
ap.add_argument("--model", required=True, help="hipfire model tag, echoed in results")
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=500, help="must be 500 to align with Jev's committed rows")
ap.add_argument("--allow-misaligned", action="store_true",
                help="allow --n != 500 (not recommended; Jev's predictions are n=500)")
ap.add_argument("--workers", type=int, default=1, help="serve serialises decides; >1 only queues")
ap.add_argument("what", nargs="+", help="task names, 'all', or x-<experiment>")
a = ap.parse_args()

if a.n != 500 and not a.allow_misaligned:
    sys.exit(f"--n {a.n} doesn't align with Jev's committed predictions (n=500);\n"
             "use --allow-misaligned to override, or set --n 500")

bench = Path(os.environ.get("JEVBENCH_DIR", Path.home() / "repos/jev-evals/jev-bench"))
out = Path(a.out).resolve()
out.mkdir(parents=True, exist_ok=True)

# Set up data directory: copy from bench/data (if it exists) into out/data
data_dir = out / "data"
if data_dir.is_symlink():
    # Remove stale symlink from older runs
    data_dir.unlink()
if not data_dir.exists():
    data_dir.mkdir(parents=True, exist_ok=True)
    bench_data = bench / "data"
    if bench_data.exists():
        for json_file in bench_data.glob("*.json"):
            shutil.copy2(json_file, data_dir)

os.environ.setdefault("TYPESAFE_API_KEY", "local-hipfire")
os.environ["JEV_WORKERS"] = str(a.workers)
sys.path.insert(0, str(bench))
import jevbench as jb  # noqa: E402

jb.ROOT = out
jb.API_URL = f"http://127.0.0.1:{a.port}/v1/systemone"
jb.MODEL = a.model
jb.WORKERS = a.workers

# jevbench.call() makes the HTTP request itself (urllib.request.urlopen), so
# there is no per-call hook to pass raw_guard's check into. Wrap urlopen
# instead: jevbench does `import urllib.request` and calls
# `urllib.request.urlopen(...)`, looking the name up on the module at call
# time, so patching the module attribute here is picked up by its calls too.
# Every reported row this writes must be a raw answer (spec §13.2, §13.6).
_real_urlopen = urllib.request.urlopen


def _guarded_urlopen(*args, **kwargs):
    resp = _real_urlopen(*args, **kwargs)
    try:
        check_raw(resp.headers.get)
    except Exception:
        resp.close()
        raise
    return resp


urllib.request.urlopen = _guarded_urlopen

# Fail loudly, now, if jevbench no longer resolves urllib.request.urlopen at
# call time (e.g. it switched to `from urllib.request import urlopen`) —
# rather than silently letting calibrated answers through the unpatched
# original. jb.urllib is the same module object this file imported, so a
# mismatch here (or a missing `urllib` attribute, raising AttributeError)
# means the patch above no longer covers jevbench's calls.
assert jb.urllib.request.urlopen is _guarded_urlopen, (
    "jevbench no longer calls urllib.request.urlopen the way this guard "
    "assumes; the raw-answer guard would not run. Adapt the patch.")

# Validate task/experiment names
for w in a.what:
    if w == "all":
        continue
    if w.startswith("x-"):
        name = w[2:]
        if name not in jb.EXPERIMENTS:
            sys.exit(f"unknown experiment {name!r}; see jevbench.py list of EXPERIMENTS\n"
                     f"valid: {', '.join(sorted(jb.EXPERIMENTS.keys()))}")
    else:
        if w not in jb.TASKS:
            sys.exit(f"unknown task {w!r}; see jevbench.py list of TASKS\n"
                     f"valid: {', '.join(sorted(jb.TASKS.keys()))}")

# Preflight: one trivial decide request built here, straight from this
# script rather than through jevbench, so a calibrated server is refused
# before the task loop runs even if jevbench's internals stop going through
# the urlopen patch above (spec §13.2, §13.6).
_preflight_body = json.dumps({
    "state": "preflight",
    "model": a.model,
    "questions": {"q": {"type": "noul", "instructions": "Answer yes or no."}},
}).encode()
_preflight_req = urllib.request.Request(
    jb.API_URL, data=_preflight_body, method="POST",
    headers={"Content-Type": "application/json"})
_preflight_resp = _real_urlopen(_preflight_req, timeout=60)
try:
    check_raw(_preflight_resp.headers.get)
finally:
    _preflight_resp.close()

for w in a.what:
    if w.startswith("x-"):
        name = w[2:]
        res, rows = jb.EXPERIMENTS[name][0](a.n, False)
        jb.save(f"x-{name}", rows, res, False)
    else:
        for name in (list(jb.TASKS) if w == "all" else [w]):
            jb.run_task(name, a.n, False)
print(f"done → {out}")
