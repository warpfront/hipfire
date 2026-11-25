#!/usr/bin/env python3
"""CPU-only self-test of the harness python (stage helper, plan generator, scorer) on fabricated artifacts.
No GPU, no model (the model/ple paths are redirected to tiny temp files inside this process only).

  python3 gap_selftest.py          # exit 0 and prints SELFTEST OK
"""
import hashlib
import json
import os
import shutil
import sys
import tempfile
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.dont_write_bytecode = True
import gap_lib as L  # noqa: E402
import gap_plan  # noqa: E402
import gap_score  # noqa: E402
import gap_stage  # noqa: E402

PLACE = L.DEFAULT_PLACE


def write(path, text, mode=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    if mode:
        os.chmod(path, mode)


def run_main(mod, argv):
    old = sys.argv
    sys.argv = ["x"] + argv
    try:
        mod.main()
    finally:
        sys.argv = old


def build_run(tmp, mutate=None):
    """Fabricate a complete, valid run dir under tmp/run-N using a real stage + plan."""
    art = os.path.join(tmp, "art")
    write(os.path.join(art, "hipfire"), "cli-bytes", 0o755)
    write(os.path.join(art, "daemon"), "daemon-bytes", 0o755)
    for i in range(3):
        write(os.path.join(art, "k", f"m{i}.hsaco"), f"obj{i}")
        write(os.path.join(art, "k", f"m{i}.hash"), f"h{i}")
    write(os.path.join(art, "pack.manifest.json"), json.dumps({"arch": L.ARCH, "modules": 3, "commit": "c0ffee"}))
    stage = os.path.join(tmp, f"stage-{len(os.listdir(tmp))}")
    run_main(gap_stage, ["--out", stage, "--cli", os.path.join(art, "hipfire"), "--daemon", os.path.join(art, "daemon"),
                         "--kernels", os.path.join(art, "k"), "--pack-manifest", os.path.join(art, "pack.manifest.json")])
    plan_dir = os.path.join(tmp, f"plan-{len(os.listdir(tmp))}")
    run_main(gap_plan, ["make", "--stage-root", stage, "--out-dir", plan_dir])
    pins = L.load_pins(os.path.join(plan_dir, "pins.json"))
    pins_sha = L.sha256_file(os.path.join(plan_dir, "pins.json"))
    run = os.path.join(tmp, f"run-{len(os.listdir(tmp))}")
    cells = os.path.join(run, "cells")
    os.makedirs(cells)
    shutil.copy(os.path.join(plan_dir, "pins.json"), os.path.join(run, "pins.json"))
    write(os.path.join(run, "watch.exit"), "0\n")
    write(os.path.join(run, "watch.dmesg.log"), "")
    write(os.path.join(run, "watch.command.txt"), "fabric-run.sh --class timing --card xtx --mem-gb 22.4 "
          "--hostmem-gb 67 --hostmem-lock --max-secs 1800 --priority P1 --owner S1Gap -- bash gap-job.sh\n")
    kdir = os.path.join(stage, "bin", "kernels", "compiled", L.ARCH)
    rates = {"A": [1740.0, 1745.5, 1750.1], "B": [2050.0, 2060.5, 2055.2]}
    t = 1000.0
    for n, label in enumerate(L.ORDER):
        arm = label[0]
        c = os.path.join(cells, label)
        os.makedirs(c)
        cap = L.stage_capture(os.path.join(stage, "bin"))
        write(os.path.join(c, "stage-identity.json"), json.dumps({"captured": cap, "diffs": []}))
        write(os.path.join(c, "stage-recheck.txt"), "")
        write(os.path.join(c, "exit"), "0\n")
        write(os.path.join(c, "leftover-after.txt"), "")
        write(os.path.join(c, "cli.pgid"), f"{5000 + n}\n")
        write(os.path.join(c, "model-session.txt"), f"{5000 + n} {5000 + n} {5000 + n} timeout\n")
        write(os.path.join(c, "fds-model.txt"), "0\n1\n2\n")
        write(os.path.join(c, "command.txt"), " ".join(
            __import__("shlex").quote(x) for x in [os.path.join(stage, "bin", "hipfire")] + L.ARGV_TAIL) + "\n")
        env = L.expected_env(arm, PLACE)
        env.update(HOME=f"/r/cells/{label}/home", HIPFIRE_HOME=f"/r/cells/{label}/home/.hipfire",
                   HIPFIRE_LOCK_DIR=f"/r/cells/{label}/lockdir", HIPFIRE_KERNEL_CACHE=f"/r/cells/{label}/kcache")
        write(os.path.join(c, "env-applied.txt"), "".join(f"{k}={v}\n" for k, v in sorted(env.items())))
        write(os.path.join(c, "preflight.json"), json.dumps({
            "label": label, "arm": arm, "vendor": "0x1002", "device": "0x744c", "vram_used_before": 1000,
            "mem_available_kib_before": 120 * 1024 * 1024, "boot_id": "b", "pins_sha256": pins_sha,
            "start_unix": t}))
        write(os.path.join(c, "postflight.json"), json.dumps({"end_unix": t + 100}))
        t += 120
        mst = "".join(f"{p} ino={pins['model'][k]['ino']} size={pins['model'][k]['size']} "
                      f"mtime_ns={pins['model'][k]['mtime_ns']}\n" for k, p in (("model", L.MODEL), ("ple", L.PLE)))
        write(os.path.join(c, "model-stat.txt"), mst)
        write(os.path.join(c, "model-stat-after.txt"), mst)
        r = rates[arm]
        samples = [r[(n // 2 + i) % 3] for i in range(3)]
        write(os.path.join(c, "stdout.json"), json.dumps({
            "protocol": "synthetic-pp-tg-matrix-v1", "model": L.MODEL, "runs": 3, "max_seq": {"effective": 16416},
            "gpu": {"arch": L.ARCH}, "kv_mode": "q8", "warnings": [],
            "prefill": [{"tokens": 16384, "samples": samples,
                         "stats": {"median": L.median(samples)}}]}))
        lines = [f"  pre-compiled kernels: {kdir}",
                 "  qwen4 state: QSA q8 (Q8_0 blocks of 32) K/V (legacy context storage), GDN q8 recurrent",
                 "  qwen4 slow-link prefill (S1): 3434 MiB charged [x], 9 resident layers (no-stage incumbent baseline 5)",
                 "  qwen4 expert placement: HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS unset, auto",
                 "  qwen4 prefill chunk: 768 rows (requested 768)",
                 f"  tensor_ops: indexed package verified: {kdir}/tensor_ops.hsaco"]
        if arm == "A":
            lines.append("  qwen4 routed experts: layers 0..9 in VRAM, 80 expert tensors")
        else:
            # place_iu4: spread placement + IU4 panels, no QSA gather line/module (QSA=0 both arms).
            lines += ["  qwen4 routed experts: layers [" + ", ".join(PLACE.split(",")) + "] in VRAM, 80 expert tensors",
                      f"  {gap_score.IU4_MODULE}: indexed package verified: {kdir}/{gap_score.IU4_MODULE}.hsaco",
                      "  qwen4 symmetric IU4 MoE: 48/48 layers verified symmetric"]
        write(os.path.join(c, "stderr.log"), "\n".join(lines) + "\n")
    write(os.path.join(cells, "job.log"), "job start\njob end rc=0\n")
    write(os.path.join(cells, "post-job-survivors.txt"), "")
    mf = []
    for root, _, files in os.walk(cells):
        for fn in sorted(files):
            p = os.path.join(root, fn)
            rel = os.path.relpath(p, cells)
            if rel != "MANIFEST.sha256":
                mf.append(f"{L.sha256_file(p)}  {rel}\n")
    write(os.path.join(cells, "MANIFEST.sha256"), "".join(sorted(mf)))
    if mutate:
        mutate(run, cells)
    return run


def expect(name, run, verdict):
    res = gap_score.score(run)
    ok = res["verdict"] == verdict
    print(f"{'ok  ' if ok else 'FAIL'} {name}: verdict={res['verdict']} (want {verdict}) problems={res.get('problems', [])[:3]}")
    return ok, res


def refresh_manifest(cells):
    mf = []
    for root, _, files in os.walk(cells):
        for fn in sorted(files):
            p = os.path.join(root, fn)
            rel = os.path.relpath(p, cells)
            if rel != "MANIFEST.sha256":
                mf.append(f"{L.sha256_file(p)}  {rel}\n")
    write(os.path.join(cells, "MANIFEST.sha256"), "".join(sorted(mf)))


def edit_stdout(label, fn):
    def mut(run, cells):
        p = os.path.join(cells, label, "stdout.json")
        d = json.load(open(p))
        fn(d)
        write(p, json.dumps(d))
        refresh_manifest(cells)
    return mut


def main():
    tmp = tempfile.mkdtemp(prefix="gap-selftest-")
    try:
        # redirect the model identity to temp files inside this process only
        m = os.path.join(tmp, "model.mq4")
        write(m, "m")
        write(m + ".ple-q4", "p")
        L.MODEL, L.PLE = m, m + ".ple-q4"
        L.ARGV_TAIL[1] = m
        os.makedirs(os.path.join(tmp, "work"))
        wt = os.path.join(tmp, "work")
        results = []
        ok, res = expect("valid run passes", build_run(wt), "PASS")
        results.append(ok)
        if ok:
            a, b = res["arms"]["A"], res["arms"]["B"]
            results.append(a["flat_n"] == 9 and b["flat_n"] == 9 and len(b["per_process_medians"]) == 3
                           and b["flat_stats"]["median"] >= 2000 and res["notes"]["synthetic"]["recomputed_sha256"])
            print("ok   flat9/process medians/synthetic digest recorded" if results[-1] else "FAIL flat9 fields")
        results.append(expect("gate fail below 2000", build_run(wt, _low), "FAIL_GATE")[0])
        results.append(expect("2 samples rejected", build_run(
            wt, edit_stdout("A2", lambda d: d["prefill"][0].__setitem__("samples", d["prefill"][0]["samples"][:2]))),
            "INVALID")[0])
        results.append(expect("extra prefill row rejected", build_run(
            wt, edit_stdout("B3", lambda d: d["prefill"].append({"tokens": 4096, "samples": [1, 2, 3]}))),
            "INVALID")[0])
        results.append(expect("env drift rejected", build_run(wt, _env_drift), "INVALID")[0])
        results.append(expect("QSA gather drift rejected in B", build_run(wt, _gather_drift), "INVALID")[0])
        results.append(expect("QSA gather drift rejected in A", build_run(wt, _gather_drift_a), "INVALID")[0])
        results.append(expect("xtx drop is INCOMPLETE", build_run(wt, _drop), "INCOMPLETE")[0])
        # stage helper refuses a copy instead of a symlink, plan refuses a drifted stage
        stage = [d for d in os.listdir(wt) if d.startswith("stage-")][0]
        sroot = os.path.join(wt, stage)
        try:
            os.chmod(os.path.join(sroot, "bin"), 0o755)
            os.unlink(os.path.join(sroot, "bin", "daemon"))
            shutil.copy(os.path.join(wt, "art", "daemon"), os.path.join(sroot, "bin", "daemon"))
            L.stage_capture(os.path.join(sroot, "bin"))
            results.append(False)
            print("FAIL copied (non-symlink) stage entry accepted")
        except SystemExit:
            results.append(True)
            print("ok   non-symlink stage entry refused")
        print("SELFTEST OK" if all(results) else "SELFTEST FAILED")
        return 0 if all(results) else 1
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print("SELFTEST FAILED")
        return 1
    finally:
        for dp, dns, _ in os.walk(tmp):
            os.chmod(dp, 0o755)
        shutil.rmtree(tmp, ignore_errors=True)


def _low(run, cells):
    for label in ("B1", "B2", "B3"):
        edit_stdout(label, lambda d: d["prefill"][0].update(samples=[1900.0, 1910.0, 1920.0],
                                                            stats={"median": 1910.0}))(run, cells)


def _env_drift(run, cells):
    p = os.path.join(cells, "A1", "env-applied.txt")
    write(p, open(p).read() + "HIPFIRE_QWEN4_Q8_TRUNK_WMMA=1\n")
    refresh_manifest(cells)


def _gather_drift(run, cells):
    # combination place_iu4 pins QSA_WMMA_GATHER=0: a leaked gather line in a B cell is drift.
    p = os.path.join(cells, "B1", "stderr.log")
    write(p, open(p).read() + "  qwen4 QSA gather workspace: 32 MiB reserved for 16416 context tokens, 32 MiB committed\n")
    refresh_manifest(cells)


def _gather_drift_a(run, cells):
    p = os.path.join(cells, "A2", "stderr.log")
    write(p, open(p).read() + "  qwen4 QSA gather workspace: 32 MiB reserved for 16416 context tokens, 32 MiB committed\n")
    refresh_manifest(cells)

def _drop(run, cells):
    write(os.path.join(cells, "XTX_DROP"), "XTX_DROP: device missing\n")


def _leftover(run, cells):
    write(os.path.join(cells, "B2", "leftover.txt"), "123 1 1 1 S 5 daemon\n")
    refresh_manifest(cells)


def _fd(run, cells):
    write(os.path.join(cells, "A3", "fds-model.txt"), "0\n1\n2\n7\n")
    refresh_manifest(cells)


if __name__ == "__main__":
    sys.exit(main())
