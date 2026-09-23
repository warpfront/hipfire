#!/usr/bin/env python3
"""Two fresh-process ABBA blocks: combined raster+norm versus official 2b48f9779."""
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / "wt-g12base"
OUT = ROOT / "scratch-g12rastergate" / "abba"
MODEL = "/home/kaden/qcal/qat/v25/qwen3.8-27b.mq4v2.xt.sym-a035.qat-r7s200.hfq"
DEVICE = "GPU-05f92432f2312a0e"
ORDER = "ABBAABBA"


def md5(path):
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(arm, name, warmup=False):
    root = BASE if arm == "A" else ROOT
    home = Path("/home/kaden/.hipfire-homes/g12rastergate-" + arm.lower())
    home.mkdir(parents=True, exist_ok=True)
    (home / "daemon.pid").unlink(missing_ok=True)
    cli, daemon = root / "target/release/hipfire", root / "target/release/daemon"
    env = {key: value for key, value in os.environ.items() if not key.startswith("HIPFIRE_")}
    env.update(HOME=str(home), ROCR_VISIBLE_DEVICES=DEVICE, HIP_VISIBLE_DEVICES=DEVICE,
               HIPFIRE_KERNEL_CACHE=str(home / ".hipfire_kernels"),
               HIPFIRE_MODELS_DIR="/home/kaden/.hipfire/models", HIPFIRE_GRAPH="1",
               HIPFIRE_DAEMON_BIN=str(daemon))
    command = [str(cli), "bench", MODEL, "--matrix", "--pp", "512,8192",
               "--ctx", "128", "--tg", "128", "--spec", "off",
               "--runs", "1" if warmup else "3", "--warmups", "1", "--kv-mode", "fp8", "--json"]
    with (OUT / f"{name}.daemon.log").open("w") as log:
        proc = subprocess.run(command, cwd=root, env=env, stdout=subprocess.PIPE,
                              stderr=log, text=True, check=False)
    (OUT / f"{name}.json").write_text(proc.stdout)
    text = (OUT / f"{name}.daemon.log").read_text(errors="replace")
    if proc.returncode or "KV cache: Fp8 vmm (" not in text or "gfx1201" not in text:
        raise RuntimeError(f"{name}: exit={proc.returncode}; gfx1201/fp8 VMM not confirmed; see daemon log")
    if warmup:
        print(name, "discarded", flush=True)
        return
    report = json.loads(proc.stdout)
    row = {"arm": arm, "process": name}
    for tokens in (512, 8192):
        matches = [r for r in report["prefill"] if r["tokens"] == tokens]
        if len(matches) != 1 or len(matches[0]["samples"]) != 3:
            raise RuntimeError(f"{name}: incomplete pp{tokens} samples")
        row[f"pp{tokens}"] = statistics.median(matches[0]["samples"])
        row[f"pp{tokens}_samples"] = matches[0]["samples"]
    row["tg128"] = statistics.median(report["decode"][0]["samples"])
    row["tg128_samples"] = report["decode"][0]["samples"]
    print(json.dumps(row), flush=True)
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    binaries = {arm: {name: md5(root / "target/release" / name)
                      for name in ("hipfire", "daemon")}
                for arm, root in (("A", BASE), ("B", ROOT))}
    fixture = {"order": ORDER, "device": DEVICE, "model": MODEL,
               "model_size": Path(MODEL).stat().st_size, "binaries_md5": binaries,
               "matrix": "--pp 512,8192 --ctx 128 --tg 128 --spec off --runs 3 --warmups 1 --kv-mode fp8 --json"}
    (OUT / "fixture.json").write_text(json.dumps(fixture, indent=2) + "\n")
    for arm in "AB":
        run(arm, "warmup-" + arm, warmup=True)
    rows = [run(arm, f"p{i + 1}-{arm}") for i, arm in enumerate(ORDER)]
    summary = {"fixture": fixture, "processes": rows, "aggregate": {}}
    for metric in ("pp512", "pp8192", "tg128"):
        a = [r[metric] for r in rows if r["arm"] == "A"]
        b = [r[metric] for r in rows if r["arm"] == "B"]
        summary["aggregate"][metric] = {"A": a, "B": b, "A_median": statistics.median(a),
                                           "B_median": statistics.median(b),
                                           "change_pct": 100 * (statistics.median(b) / statistics.median(a) - 1)}
    fast = {arm: [r["tg128"] for r in rows if r["arm"] == arm and r["tg128"] >= 35]
            for arm in "AB"}
    summary["fast_decode"] = {arm: statistics.median(vals) if vals else None
                               for arm, vals in fast.items()}
    summary["fast_decode_change_pct"] = (100 * (summary["fast_decode"]["B"] /
                                               summary["fast_decode"]["A"] - 1)
                                         if all(fast.values()) else None)
    summary["pass"] = (summary["aggregate"]["pp8192"]["change_pct"] > 0 and
                       summary["aggregate"]["pp512"]["change_pct"] >= -1 and
                       summary["fast_decode_change_pct"] is not None and
                       abs(summary["fast_decode_change_pct"]) <= 1)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"aggregate": summary["aggregate"], "fast_decode": summary["fast_decode"],
                      "fast_decode_change_pct": summary["fast_decode_change_pct"],
                      "pass": summary["pass"]}, indent=2), flush=True)
    if not summary["pass"]:
        raise SystemExit("ABBA gate failed")


if __name__ == "__main__":
    main()
