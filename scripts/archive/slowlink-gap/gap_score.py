#!/usr/bin/env python3
"""CPU scorer for ONE completed ABBAAB run (output of gap-launch.sh). Reads receipts only; no GPU, no model.

  gap_score.py RUN_DIR [--out FRESH_SCORE.json]

RUN_DIR holds cells/{A1,B1,B2,A2,A3,B3}/, pins.json, plan.submitted.tsv, launch.json, watch.*  (the layout
gap-launch.sh creates). Asserts, per the qualification contract:
  * exactly six cells in order A1 B1 B2 A2 A3 B3; three fresh processes per arm (distinct session/PGID, sequential);
  * every cell: exit 0, H2 argv, stdout prefill has exactly ONE entry of tokens=16384 with exactly THREE
    daemon-measured samples (tok/s) and runs=3, stats.median == median(samples);
    HIPFIRE_KV_MODE=q8 D2 off, QSA_WMMA_GATHER=0; arm B differs from arm A ONLY in placement / IU4
    (combination `place_iu4`, QSA=0, Q8w=0); nothing else applied;
  * stage identity (symlinks, canonical targets, ino/size/mtime_ns, sha256, md5, kernel pack tree hash) equals the
    pins in every cell and was unchanged after; model stat unchanged (cached stat only; never hashed);
  * route identities from stderr (state, resident layers, IU4 verify line, kernel pack dir,
    module set) identical across an arm's cells and consistent with the arm's levers; QSA gather and the Q8 trunk
    lever are rejected in BOTH arms; optional carry route regex;
  * no XTX drop, no leftover processes, no fd>2 in the model group, no GPU reset/bus events in the kernel log,
    MemAvailable >= 80 GiB at every cell start, receipts MANIFEST.sha256 verifies.
Computes per arm: all nine daemon-measured rates (cell order), per-process medians, flat median/mean/min/max.
Gate: candidate (arm B) flat median over all nine >= 2000 tok/s.
Also records the synthetic token stream digest (the daemon's deterministic ids, recomputed here and pinned).
Exit: 0 PASS; 1 FAIL_GATE (valid run, below 2000); 2 INVALID (a contract assertion failed); 3 INCOMPLETE (XTX drop
or the run did not finish: stop, tell the parent, no retry).
"""
import argparse
import hashlib
import json
import math
import os
import re
import shlex
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gap_lib as L  # noqa: E402

DMESG_BAD = re.compile(r"GPU reset (begin|succeeded|failed)|ring \S+ timeout|VM_FAULT|page fault|fallen off the bus|"
                       r"device lost|amdgpu.*(hang|fatal)|pcieport.*(AER|Fatal)|BUG:|Oops", re.I)
ROUTE_PATTERNS = {
    "state": re.compile(r"^\s*qwen4 state: (.*)$"),
    "expert_placement": re.compile(r"^\s*qwen4 expert placement: (.*)$"),
    "routed_experts": re.compile(r"^\s*qwen4 routed experts: layers (.+?) in VRAM"),
    "s1_resident_layers": re.compile(r"qwen4 slow-link prefill \(S1\):.*?(\d+) resident layers"),
    "iu4": re.compile(r"^\s*qwen4 symmetric IU4 MoE: (.*)$"),
    "qsa_gather": re.compile(r"^\s*qwen4 QSA gather workspace: (.*)$"),
    "prefill_chunk": re.compile(r"^\s*qwen4 prefill chunk: (.*)$"),
    "no_closed_route_plan": re.compile(r"\[no closed route plan: (.*?)\]"),
    "kernels_dir": re.compile(r"^\s*pre-compiled kernels: (\S+)\s*$"),
}
MODULE2_RE = re.compile(r"^\s*(\S+): indexed package verified: (\S+)\s*$")
GATHER_MODULE = "indexed_attention_gathered_wmma"
IU4_MODULE = "qwen4_moe_sym_check_gfx1100"


def read(p, default=None):
    try:
        with open(p) as f:
            return f.read()
    except OSError:
        return default


def parse_env(text):
    env = {}
    for line in text.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            env[k] = v
    return env


def parse_route(stderr):
    route = {k: [] for k in ROUTE_PATTERNS}
    modules = {}
    for line in stderr.splitlines():
        for k, rx in ROUTE_PATTERNS.items():
            m = rx.search(line)
            if m:
                route[k].append(m.group(1))
        m = MODULE2_RE.match(line)
        if m:
            modules.setdefault(m.group(1), set()).add(os.path.dirname(m.group(2)))
    return route, modules


def stats(xs):
    return {"n": len(xs), "median": L.median(xs), "mean": statistics.fmean(xs), "min": min(xs), "max": max(xs),
            "stdev": statistics.pstdev(xs) if len(xs) > 1 else 0.0}


def score(run):
    problems = []
    notes = {}

    def bad(msg):
        problems.append(msg)

    pins_text = read(os.path.join(run, "pins.json"))
    if pins_text is None:
        return {"verdict": "INVALID", "problems": ["pins.json missing from the run dir"]}
    pins = json.loads(pins_text)
    if pins.get("schema") != L.SCHEMA_PINS:
        return {"verdict": "INVALID", "problems": ["pins.json schema mismatch"]}
    pins_sha = hashlib.sha256(pins_text.encode()).hexdigest()
    place = pins["place"]
    want_arms = {"A": L.arm_env("A", place), "B": L.arm_env("B", place)}
    if pins.get("combination") != L.COMBINATION or pins.get("levers_on") != ["place", "iu4"]:
        bad(f"pins combination {pins.get('combination')}/{pins.get('levers_on')} != place_iu4/[place, iu4]")
    if pins["arms"] != want_arms or pins["pinned_env"] != L.PINNED_ENV or pins["order"] != L.ORDER:
        bad("pins arms/pinned_env/order differ from the harness definition")
    if pins["argv_tail"] != L.ARGV_TAIL:
        bad("pins argv differs from the H2 argv")
    syn = L.synthetic_stream_digest()
    if pins["synthetic"].get("sha256") != syn or pins["synthetic"].get("tokens") != L.PP_TOKENS:
        bad("pinned synthetic stream digest differs from the recomputed daemon stream")
    stage_bin = os.path.join(pins["stage_root"], "bin")

    cells_dir = os.path.join(run, "cells")
    if os.path.exists(os.path.join(cells_dir, "XTX_DROP")) or any(
            os.path.exists(os.path.join(cells_dir, c, "XTX_DROP")) for c in L.ORDER):
        return {"verdict": "INCOMPLETE", "problems": ["XTX_DROP present: stop, tell the parent, no retry"],
                "xtx_drop": (read(os.path.join(cells_dir, "XTX_DROP")) or "").strip()}
    if not os.path.isdir(cells_dir):
        return {"verdict": "INCOMPLETE", "problems": ["cells/ missing: the job never ran"]}
    present = sorted(d for d in os.listdir(cells_dir) if os.path.isdir(os.path.join(cells_dir, d)))
    if present != sorted(L.ORDER):
        if set(present) < set(L.ORDER) and read(os.path.join(run, "watch.exit")) is not None:
            return {"verdict": "INCOMPLETE", "problems": [f"only cells {present} ran (job stopped early)"],
                    "cells_present": present}
        bad(f"cell dirs {present} != {sorted(L.ORDER)}")
    watch_exit = (read(os.path.join(run, "watch.exit")) or "").strip()
    if watch_exit != "0":
        bad(f"watch.exit={watch_exit!r} (job/hold did not finish cleanly)")
    wcmd = read(os.path.join(run, "watch.command.txt"), "")
    for tok in ("--class timing", "--card xtx", "--mem-gb 22.4", "--hostmem-gb 67", "--hostmem-lock",
                "--priority P1"):
        if tok not in wcmd:
            bad(f"fabric command lacks {tok!r}")
    m = re.search(r"--max-secs (\d+)", wcmd)
    if not m or int(m.group(1)) > 1800:
        bad("fabric --max-secs missing or > 1800")
    joblog = read(os.path.join(cells_dir, "job.log"), "")
    if not re.search(r"job end rc=0\s*$", joblog):
        bad("job.log does not end with 'job end rc=0'")
    if (read(os.path.join(cells_dir, "post-job-survivors.txt"), "x")).strip():
        bad("post-job survivors recorded (or file missing)")
    dmesg = read(os.path.join(run, "watch.dmesg.log"))
    if dmesg is None:
        bad("watch.dmesg.log missing (kernel watcher)")
    else:
        events = [l for l in dmesg.splitlines() if DMESG_BAD.search(l)]
        notes["dmesg_gpu_events"] = events[:20]
        if events:
            bad(f"{len(events)} GPU/bus event line(s) in the kernel log during the hold")
    manifest = read(os.path.join(cells_dir, "MANIFEST.sha256"))
    if manifest is None:
        bad("cells/MANIFEST.sha256 missing")
    else:
        n_ok = 0
        for line in manifest.splitlines():
            h, _, rel = line.partition("  ")
            if not os.path.isfile(os.path.join(cells_dir, rel)):
                bad(f"manifest file missing: {rel}")
                continue
            with open(os.path.join(cells_dir, rel), "rb") as f:
                got = hashlib.sha256(f.read()).hexdigest()
            if got != h:
                bad(f"manifest sha256 mismatch: {rel}")
            else:
                n_ok += 1
        notes["manifest_files_verified"] = n_ok
    notes["synthetic"] = dict(pins["synthetic"], recomputed_sha256=syn)

    cells = {}
    boot_ids, pgids, last_end = set(), [], 0.0
    for label in L.ORDER:
        c = os.path.join(cells_dir, label)
        arm = label[0]
        info = {"label": label, "arm": arm}
        cells[label] = info
        if not os.path.isdir(c):
            continue
        if (read(os.path.join(c, "exit"), "").strip()) != "0":
            bad(f"{label}: exit != 0")
        for fn in ("leftover.txt", "leftover-after.txt", "leftover-term.txt"):
            if (read(os.path.join(c, fn), "") or "").strip():
                bad(f"{label}: {fn} not empty (surviving owned process)")
        if read(os.path.join(c, "leftover-after.txt")) is None:
            bad(f"{label}: leftover-after.txt missing")
        if (read(os.path.join(c, "stage-recheck.txt"), "") or "").strip():
            bad(f"{label}: stage changed during the cell")
        # argv
        cmd = shlex.split(read(os.path.join(c, "command.txt"), ""))
        if cmd != [os.path.join(stage_bin, "hipfire")] + L.ARGV_TAIL:
            bad(f"{label}: command.txt != pinned H2 argv: {cmd}")
        info["command"] = cmd
        # preflight / postflight
        try:
            pre = json.loads(read(os.path.join(c, "preflight.json")))
            post = json.loads(read(os.path.join(c, "postflight.json")))
        except (TypeError, ValueError):
            bad(f"{label}: preflight/postflight missing")
            continue
        info["preflight"], info["postflight"] = pre, post
        if pre.get("vendor") != "0x1002" or pre.get("device") != "0x744c":
            bad(f"{label}: wrong device {pre.get('vendor')}:{pre.get('device')}")
        if pre.get("mem_available_kib_before", 0) < 80 * 1024 * 1024:
            bad(f"{label}: MemAvailable {pre.get('mem_available_kib_before')} KiB < 80 GiB")
        if pre.get("vram_used_before", 1 << 62) >= 2 * 1024 ** 3:
            bad(f"{label}: card not idle at start")
        if pre.get("arm") != arm:
            bad(f"{label}: preflight arm {pre.get('arm')}")
        if pre.get("pins_sha256") != pins_sha:
            bad(f"{label}: cell ran with pins {pre.get('pins_sha256')} != run pins {pins_sha}")
        boot_ids.add(pre.get("boot_id"))
        if pre.get("start_unix", 0) < last_end:
            bad(f"{label}: overlaps the previous cell")
        last_end = post.get("end_unix", 0)
        # fresh process: own session/PGID, no fd > 2
        pg = (read(os.path.join(c, "cli.pgid"), "").strip())
        pgids.append(pg)
        sess = (read(os.path.join(c, "model-session.txt"), "")).split()
        if len(sess) < 3 or not (sess[0] == sess[1] == sess[2] == pg):
            bad(f"{label}: model launcher not its own session/PGID leader ({sess}, cli.pgid {pg})")
        fds = [int(x) for x in (read(os.path.join(c, "fds-model.txt"), "") or "").split() if x.isdigit()]
        if not fds or max(fds) > 2:
            bad(f"{label}: model group fds {fds} (must be only 0-2)")
        info["model_fds"] = fds
        # env identities
        env = parse_env(read(os.path.join(c, "env-applied.txt"), ""))
        exp = L.expected_env(arm, place)
        paths = {}
        for k in L.CELL_PATH_KEYS:
            paths[k] = env.pop(k, None)
        info["fabric_admission_env"] = env.pop("HIPFIRE_FABRIC_ADMISSION", None)
        suffix = {"HOME": f"/cells/{label}/home", "HIPFIRE_HOME": f"/cells/{label}/home/.hipfire",
                  "HIPFIRE_LOCK_DIR": f"/cells/{label}/lockdir", "HIPFIRE_KERNEL_CACHE": f"/cells/{label}/kcache"}
        for k, sfx in suffix.items():
            if not (paths[k] or "").endswith(sfx):
                bad(f"{label}: {k}={paths[k]} not under the cell dir")
        if env != exp:
            changed = {k: env[k] for k in env if exp.get(k) != env[k]}
            bad(f"{label}: applied env differs from pinned definition: extra/changed={changed} "
                f"missing={sorted(set(exp) - set(env))}")
        info["env_applied"] = dict(sorted(env.items()))
        # stage identity
        sid = None
        try:
            sid = json.loads(read(os.path.join(c, "stage-identity.json")))
        except (TypeError, ValueError):
            bad(f"{label}: stage-identity.json missing")
        if sid is not None:
            diffs = L.diff_captures(pins["stage"], sid["captured"])
            if diffs or sid.get("diffs"):
                bad(f"{label}: stage identity differs from pins: {diffs[:3]}")
            info["stage_identity"] = {
                "hipfire_sha256": sid["captured"]["entries"]["hipfire"]["sha256"],
                "hipfire_md5": sid["captured"]["entries"]["hipfire"]["md5"],
                "daemon_sha256": sid["captured"]["entries"]["daemon"]["sha256"],
                "daemon_md5": sid["captured"]["entries"]["daemon"]["md5"],
                "kernels_tree_sha256": sid["captured"]["kernels"]["tree_sha256"],
            }
        mb, ma = read(os.path.join(c, "model-stat.txt")), read(os.path.join(c, "model-stat-after.txt"))
        if mb is None or mb != ma:
            bad(f"{label}: model cached-stat before/after differ or missing")
        for key, path in (("model", L.MODEL), ("ple", L.PLE)):
            p = pins["model"][key]
            want = f"{path} ino={p['ino']} size={p['size']} mtime_ns={p['mtime_ns']}"
            if want not in (mb or "").splitlines():
                bad(f"{label}: model stat line for {key} differs from pins")
        # stdout: exactly one prefill entry, exactly three daemon-measured samples
        try:
            out = json.loads(read(os.path.join(c, "stdout.json")))
        except (TypeError, ValueError):
            bad(f"{label}: stdout.json missing/unparseable")
            continue
        if out.get("protocol") != "synthetic-pp-tg-matrix-v1" or out.get("model") != L.MODEL:
            bad(f"{label}: protocol/model mismatch")
        if out.get("runs") != L.RUNS:
            bad(f"{label}: runs={out.get('runs')} != {L.RUNS}")
        if (out.get("max_seq") or {}).get("effective") != L.MAX_SEQ:
            bad(f"{label}: max_seq.effective != {L.MAX_SEQ}")
        if (out.get("gpu") or {}).get("arch") != L.ARCH:
            bad(f"{label}: gpu arch {(out.get('gpu') or {}).get('arch')}")
        pre_rows = out.get("prefill") or []
        if len(pre_rows) != 1 or pre_rows[0].get("tokens") != L.PP_TOKENS:
            bad(f"{label}: prefill rows {[r.get('tokens') for r in pre_rows]} != [{L.PP_TOKENS}]")
            continue
        samples = pre_rows[0].get("samples")
        if not isinstance(samples, list) or len(samples) != L.RUNS or not all(
                isinstance(x, (int, float)) and math.isfinite(x) and x > 0 for x in samples):
            bad(f"{label}: prefill samples {samples!r} are not exactly {L.RUNS} positive daemon rates")
            continue
        med = L.median(samples)
        if abs(pre_rows[0].get("stats", {}).get("median", -1) - med) > 0.051:
            bad(f"{label}: stats.median {pre_rows[0].get('stats', {}).get('median')} != median(samples) {med}")
        info["samples"] = [float(x) for x in samples]
        info["process_median"] = med
        info["stdout_warnings"] = out.get("warnings")
        info["kv_mode_reported"] = out.get("kv_mode")
        info["kv_backend"] = out.get("kv_backend")
        info["prefill_route"] = out.get("prefill_route")
        # route identity
        route, modules = parse_route(read(os.path.join(c, "stderr.log"), ""))
        info["route"] = route
        info["modules"] = sorted(modules)
        kd = set(route["kernels_dir"])
        allowed = {os.path.join(stage_bin, "kernels", "compiled", L.ARCH), pins["stage"]["kernels"]["canonical"]}
        if len(kd) != 1 or not (kd & allowed or any(p.endswith(f"/cells/{label}/kcache/{L.ARCH}") for p in kd)):
            bad(f"{label}: kernel pack dir logged {sorted(kd)} not the pinned stage pack {sorted(allowed)}")
        for name, dirs in modules.items():
            if len(kd) == 1 and dirs != kd:
                bad(f"{label}: module {name} verified from {sorted(dirs)} != {sorted(kd)}")
        if not route["s1_resident_layers"]:
            bad(f"{label}: no S1 slow-link runtime line (config not enabling S1?)")
        if not any("QSA q8" in s and "GDN q8" in s for s in route["state"]):
            bad(f"{label}: state route is not QSA q8 / GDN q8: {route['state']}")
        has_iu4 = any("48/48 layers verified symmetric" in s for s in route["iu4"])
        # QSA gather is pinned OFF in both arms on this runtime: its workspace line and gathered WMMA
        # module are drift, not a lever. Any Q8 trunk WMMA variable is likewise drift (no such runtime lever).
        if route["qsa_gather"] or GATHER_MODULE in modules:
            bad(f"{label}: QSA gather route observed (QSA=0 in combination place_iu4)")
        placed = [s for s in route["routed_experts"]]
        if arm == "B":
            want_layers = "[" + ", ".join(place.split(",")) + "]" if re.match(r"^[0-9,]+$", place) else None
            if want_layers and placed != [want_layers]:
                bad(f"{label}: resident experts {placed} != {want_layers}")
            if not has_iu4:
                bad(f"{label}: IU4 symmetric verification line missing in arm B")
        else:
            if has_iu4 or route["iu4"] or IU4_MODULE in modules:
                bad(f"{label}: IU4 route observed in arm A")
        rx = (pins.get("carry") or {}).get("route_regex")
        if rx and not re.search(rx, read(os.path.join(c, "stderr.log"), ""), re.M):
            bad(f"{label}: carry route regex {rx!r} not found in stderr")

    if len(set(pgids)) != len(L.ORDER) or "" in pgids:
        bad(f"cell session/PGIDs not six distinct fresh processes: {pgids}")
    if len(boot_ids) > 1:
        bad(f"boot_id changed during the run: {boot_ids}")

    # Route identity must be identical across the cells of an arm.
    for arm in "AB":
        labels = [l for l in L.ORDER if l[0] == arm and "route" in cells[l]]
        for key in ("state", "routed_experts", "s1_resident_layers", "iu4", "qsa_gather", "prefill_chunk",
                    "kernels_dir", "no_closed_route_plan"):
            vals = {json.dumps(cells[l]["route"][key]) for l in labels}
            if len(vals) > 1:
                bad(f"arm {arm}: route '{key}' differs between cells: {sorted(vals)}")
        mods = {json.dumps(cells[l]["modules"]) for l in labels}
        if len(mods) > 1:
            bad(f"arm {arm}: verified kernel module set differs between cells")
    a_route = [cells[l]["route"] for l in L.ORDER if l[0] == "A" and "route" in cells[l]]
    b_route = [cells[l]["route"] for l in L.ORDER if l[0] == "B" and "route" in cells[l]]
    if a_route and b_route and a_route[0]["state"] != b_route[0]["state"]:
        bad("state route differs between arms (arms must keep the same q8 state runtime)")
    a_env = [cells[l].get("env_applied") for l in L.ORDER if l[0] == "A"]
    b_env = [cells[l].get("env_applied") for l in L.ORDER if l[0] == "B"]
    if all(a_env) and all(b_env):
        diff_keys = sorted(k for k in set(a_env[0]) | set(b_env[0]) if a_env[0].get(k) != b_env[0].get(k))
        if diff_keys != sorted(L.ARM_KEYS):
            bad(f"arms differ in env keys {diff_keys}; only {sorted(L.ARM_KEYS)} may vary")
        notes["arm_env_diff_keys"] = diff_keys

    arms = {}
    for arm in "AB":
        ls = [l for l in L.ORDER if l[0] == arm]
        if not all("samples" in cells[l] for l in ls) or len(ls) != 3:
            bad(f"arm {arm}: missing samples")
            continue
        flat = [x for l in ls for x in cells[l]["samples"]]
        arms[arm] = {
            "cells_in_order": ls,
            "rates_by_cell": {l: cells[l]["samples"] for l in ls},
            "per_process_medians": {l: cells[l]["process_median"] for l in ls},
            "flat": flat,
            "flat_n": len(flat),
            "flat_stats": stats(flat),
        }
        if len(flat) != 9:
            bad(f"arm {arm}: flat sample count {len(flat)} != 9")
    result = {
        "schema": "slowlink-gap.score.v1",
        "run_dir": run,
        "pins_sha256": pins_sha,
        "stage_root": pins["stage_root"],
        "place": place,
        "combination": pins.get("combination", L.COMBINATION),
        "levers_on": pins.get("levers_on", ["place", "iu4"]),
        "order": L.ORDER,
        "gate_tok_s": L.GATE_TOK_S,
        "arms": arms,
        "cells": cells,
        "notes": notes,
        "problems": problems,
    }
    if "B" in arms:
        result["candidate_flat_median"] = arms["B"]["flat_stats"]["median"]
        result["gate_met"] = arms["B"]["flat_stats"]["median"] >= L.GATE_TOK_S
        if "A" in arms:
            result["candidate_over_base_flat_median"] = arms["B"]["flat_stats"]["median"] / arms["A"]["flat_stats"][
                "median"]
    if problems:
        result["verdict"] = "INVALID"
    else:
        result["verdict"] = "PASS" if result["gate_met"] else "FAIL_GATE"
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None, help="fresh path for the score json (refused if it exists)")
    a = ap.parse_args()
    if a.out and os.path.lexists(a.out):
        L.die(f"refusing existing {a.out}")
    res = score(os.path.abspath(a.run_dir))
    text = json.dumps(res, indent=1, sort_keys=True) + "\n"
    if a.out:
        with open(a.out, "x") as f:
            f.write(text)
    print(f"verdict={res['verdict']} candidate_flat_median={res.get('candidate_flat_median')} "
          f"gate={L.GATE_TOK_S} problems={len(res.get('problems', []))}")
    for p in res.get("problems", []):
        print("  PROBLEM:", p)
    for arm, d in sorted(res.get("arms", {}).items()):
        s = d["flat_stats"]
        print(f"  arm {arm}: flat9 median={s['median']:.1f} mean={s['mean']:.1f} min={s['min']} max={s['max']} "
              f"process medians={list(d['per_process_medians'].values())}")
    if not a.out:
        print(text)
    sys.exit({"PASS": 0, "FAIL_GATE": 1, "INVALID": 2, "INCOMPLETE": 3}[res["verdict"]])


if __name__ == "__main__":
    main()
