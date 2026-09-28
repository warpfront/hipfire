"""Run Running-Dolphins/jev-bench against local hipfire serve.

jevbench.py writes predictions/results under its module ROOT, which would
overwrite the committed Jev baselines — so ROOT is redirected to --out and the
dataset cache is symlinked from the clone.
"""
import argparse
import os
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=11435)
ap.add_argument("--model", required=True, help="hipfire model tag, echoed in results")
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=500, help="must be 500 to align with Jev's committed rows")
ap.add_argument("--workers", type=int, default=1, help="serve serialises decides; >1 only queues")
ap.add_argument("what", nargs="+", help="task names, 'all', or x-<experiment>")
a = ap.parse_args()

bench = Path(os.environ.get("JEVBENCH_DIR", Path.home() / "repos/jev-evals/jev-bench"))
out = Path(a.out).resolve()
out.mkdir(parents=True, exist_ok=True)
if not (out / "data").exists():
    (out / "data").symlink_to(bench / "data")
os.environ.setdefault("TYPESAFE_API_KEY", "local-hipfire")
os.environ["JEV_WORKERS"] = str(a.workers)
sys.path.insert(0, str(bench))
import jevbench as jb  # noqa: E402

jb.ROOT = out
jb.API_URL = f"http://127.0.0.1:{a.port}/v1/systemone"
jb.MODEL = a.model
jb.WORKERS = a.workers

for w in a.what:
    if w.startswith("x-"):
        name = w[2:]
        res, rows = jb.EXPERIMENTS[name][0](a.n, False)
        jb.save(f"x-{name}", rows, res, False)
    else:
        for name in (list(jb.TASKS) if w == "all" else [w]):
            jb.run_task(name, a.n, False)
print(f"done → {out}")
