#!/usr/bin/env python3
"""ABBAAB plan generator + the pin checks gap-cell.sh / gap-job.sh / gap-launch.sh call.  CPU only.

Generate (run on the host that will launch, after gap_stage.py built the stage there):

  gap_plan.py make --stage-root STAGE_ROOT --out-dir FRESH_PLAN_DIR [--place 0,3,7,15,19,23,27,39,43]
                   [--config FILE] [--carry-route-regex REGEX]

Writes FRESH_PLAN_DIR/{plan.tsv,pins.json,candidate.toml} (read-only). plan.tsv is exactly
A1 B1 B2 A2 A3 B3 (six fresh model processes), every line
  LABEL <TAB> STAGE_ROOT/bin <TAB> FRESH_PLAN_DIR/candidate.toml <TAB> GAP_PINS=.../pins.json <TAB> GAP_ARM=A|B
Arm A = every S1 lever off. Arm B = placement spread + IU4 panels (qualified combination `place_iu4`, QSA=0,
Q8w=0). Both arms: same stage binaries, same config, same
--carry-route-regex (optional) is a regex the scorer requires in every cell's stderr (extra route identity pin,
e.g. a runtime log line a future stage must show).
"""
import argparse
import hashlib
import json
import os
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gap_lib as L  # noqa: E402

DEFAULT_CONFIG = os.path.join(L.HERE, "candidate.toml")


def _load_stage_manifest(stage_root):
    p = os.path.join(stage_root, "STAGE-MANIFEST.json")
    if not os.path.isfile(p):
        L.die(f"{p} missing: build the stage with gap_stage.py")
    with open(p) as f:
        m = json.load(f)
    if m.get("schema") != L.SCHEMA_STAGE:
        L.die(f"{p}: not a {L.SCHEMA_STAGE} manifest")
    return p, m


def cmd_make(a):
    stage_root = os.path.abspath(a.stage_root)
    bin_dir = os.path.join(stage_root, "bin")
    mpath, manifest = _load_stage_manifest(stage_root)
    cap = L.stage_capture(bin_dir, hashes=True)
    diffs = L.diff_captures(manifest["capture"], cap)
    if diffs:
        L.die("stage differs from STAGE-MANIFEST.json (rebuild the stage on this host):\n  " + "\n  ".join(diffs))
    pm = manifest["pack_manifest"]
    if L.sha256_file(pm["canonical"]) != pm["sha256"]:
        L.die("pack manifest sha256 changed since the stage was built")
    out_dir = os.path.abspath(a.out_dir)
    if os.path.lexists(out_dir):
        L.die(f"refusing existing plan dir {out_dir}")
    if not os.path.isdir(os.path.dirname(out_dir)):
        L.die(f"parent of {out_dir} does not exist")
    place = a.place
    L.arm_env("B", place)  # validates
    cfg_src = os.path.abspath(a.config)
    with open(cfg_src, "rb") as f:
        cfg_bytes = f.read()
    for p in (L.MODEL, L.PLE):
        if not os.path.isfile(p):
            L.die(f"missing {p}")
    os.mkdir(out_dir)
    cfg_path = os.path.join(out_dir, "candidate.toml")
    with open(cfg_path, "wb") as f:
        f.write(cfg_bytes)
    pins_path = os.path.join(out_dir, "pins.json")
    pins = {
        "schema": L.SCHEMA_PINS,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": socket.gethostname(),
        "stage_root": stage_root,
        "stage_manifest": {"path": mpath, "sha256": L.sha256_file(mpath)},
        "stage": cap,
        "pack_manifest": pm,
        "config": {"path": cfg_path, "sha256": hashlib.sha256(cfg_bytes).hexdigest()},
        "model": {"model": L.model_stat(L.MODEL), "ple": L.model_stat(L.PLE), "hashed": False},
        "argv_tail": L.ARGV_TAIL,
        "pinned_env": L.PINNED_ENV,
        "place": place,
        "combination": L.COMBINATION,
        "levers_on": ["place", "iu4"],
        "arms": {"A": L.arm_env("A", place), "B": L.arm_env("B", place)},
        "order": L.ORDER,
        "gate_tok_s": L.GATE_TOK_S,
        "carry": {"env_switch": None, "note": "no f32 carry: existing q8-cadence runtime; no env switch, no global f32",
                  "route_regex": a.carry_route_regex},
        "synthetic": dict(L.SYNTHETIC, sha256=L.synthetic_stream_digest()),
        "device": {"uuid": L.UUID, "bdf": L.BDF},
    }
    with open(pins_path, "w") as f:
        json.dump(pins, f, indent=1, sort_keys=True)
        f.write("\n")
    plan_path = os.path.join(out_dir, "plan.tsv")
    with open(plan_path, "w") as f:
        f.write(f"# S1 pp16384 >=2000 tok/s ABBAAB; stage {bin_dir}; place {place}; pins sha256 "
                f"{L.sha256_file(pins_path)}\n")
        for label in L.ORDER:
            f.write("\t".join([label, bin_dir, cfg_path, f"GAP_PINS={pins_path}", f"GAP_ARM={label[0]}"]) + "\n")
    for p in (cfg_path, pins_path, plan_path):
        os.chmod(p, 0o444)
    os.chmod(out_dir, 0o555)
    print(json.dumps({"plan": plan_path, "pins": pins_path, "stage_bin": bin_dir, "order": L.ORDER}, indent=1))


def _parse_plan(plan_path):
    rows = []
    with open(plan_path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            rows.append(line.split("\t"))
    return rows


def cmd_check_plan(a):
    plan = os.path.abspath(a.plan)
    rows = _parse_plan(plan)
    if [r[0] for r in rows] != L.ORDER:
        L.die(f"{plan}: labels {[r[0] for r in rows]} != {L.ORDER}")
    pins_path = None
    for r in rows:
        if len(r) != 5 or not r[3].startswith("GAP_PINS=") or r[4] != f"GAP_ARM={r[0][0]}":
            L.die(f"{plan}: malformed line for {r[0]}: {r!r}")
        if pins_path is None:
            pins_path = r[3].split("=", 1)[1]
        elif r[3] != f"GAP_PINS={pins_path}":
            L.die(f"{plan}: mixed pins files")
    pins = L.load_pins(pins_path)
    for r in rows:
        if r[1] != os.path.join(pins["stage_root"], "bin") or r[2] != pins["config"]["path"]:
            L.die(f"{plan}: {r[0]} stage/config differ from pins")
    if L.sha256_file(pins["config"]["path"]) != pins["config"]["sha256"]:
        L.die("config sha256 differs from pins")
    print(pins_path)


def cmd_verify_stage(a):
    pins = L.load_pins(a.pins)
    if os.path.abspath(a.bin_dir) != os.path.join(pins["stage_root"], "bin"):
        L.die(f"BIN_DIR {a.bin_dir} is not the pinned stage {pins['stage_root']}/bin")
    cap = L.stage_capture(a.bin_dir, hashes=True)
    diffs = L.diff_captures(pins["stage"], cap)
    with open(a.out_json, "w") as f:
        json.dump({"captured": cap, "diffs": diffs}, f, indent=1, sort_keys=True)
        f.write("\n")
    if a.binaries_out:
        with open(a.binaries_out, "w") as f:
            for name, e in sorted(cap["entries"].items()):
                f.write(f"{e['md5']} {e['sha256']} {e['canonical']} {name} readlink={e.get('readlink')} "
                        f"ino={e['ino']} size={e['size']} mtime_ns={e['mtime_ns']}\n")
            k = cap["kernels"]
            f.write(f"kernels tree_sha256={k['tree_sha256']} files={k['file_count']} hsaco={k['hsaco_count']} "
                    f"canonical={k['canonical']}\n")
    if diffs:
        L.die("stage identity differs from pins:\n  " + "\n  ".join(diffs))


def cmd_recheck_stage(a):
    pins = L.load_pins(a.pins)
    cap = L.stage_capture(a.bin_dir, hashes=False)
    diffs = L.diff_captures(pins["stage"], cap, stat_only=True)
    if diffs:
        L.die("stage changed during the cell:\n  " + "\n  ".join(diffs), code=6)


def cmd_cell_env(a):
    pins = L.load_pins(a.pins)
    want = {"A": L.arm_env("A", pins["place"]), "B": L.arm_env("B", pins["place"])}
    if pins["arms"] != want or pins["pinned_env"] != L.PINNED_ENV or pins["order"] != L.ORDER:
        L.die("pins arms/pinned_env/order differ from the harness definition")
    if pins["argv_tail"] != L.ARGV_TAIL:
        L.die("pins argv differs from the H2 argv")
    env = dict(L.PINNED_ENV)
    env.update(want[a.arm])
    for k, v in env.items():
        print(f"{k}={v}")


def cmd_check_model(a):
    pins = L.load_pins(a.pins)
    bad = []
    for key, path in (("model", L.MODEL), ("ple", L.PLE)):
        got = L.model_stat(path)
        if got != pins["model"][key]:
            bad.append(f"{key}: pinned={pins['model'][key]} actual={got}")
        print(f"{path} ino={got['ino']} size={got['size']} mtime_ns={got['mtime_ns']}")
    if bad:
        L.die("model stat differs from pins:\n  " + "\n  ".join(bad))


def cmd_check_config(a):
    pins = L.load_pins(a.pins)
    if os.path.abspath(a.config) != pins["config"]["path"] or L.sha256_file(a.config) != pins["config"]["sha256"]:
        L.die("config path/sha256 differs from pins")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("make")
    p.add_argument("--stage-root", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--place", default=L.DEFAULT_PLACE)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--carry-route-regex", default=None)
    p.set_defaults(fn=cmd_make)
    p = sub.add_parser("check-plan")
    p.add_argument("plan")
    p.set_defaults(fn=cmd_check_plan)
    p = sub.add_parser("verify-stage")
    p.add_argument("bin_dir")
    p.add_argument("pins")
    p.add_argument("out_json")
    p.add_argument("--binaries-out", default=None)
    p.set_defaults(fn=cmd_verify_stage)
    p = sub.add_parser("recheck-stage")
    p.add_argument("bin_dir")
    p.add_argument("pins")
    p.set_defaults(fn=cmd_recheck_stage)
    p = sub.add_parser("cell-env")
    p.add_argument("pins")
    p.add_argument("arm", choices=("A", "B"))
    p.set_defaults(fn=cmd_cell_env)
    p = sub.add_parser("check-model")
    p.add_argument("pins")
    p.set_defaults(fn=cmd_check_model)
    p = sub.add_parser("check-config")
    p.add_argument("pins")
    p.add_argument("config")
    p.set_defaults(fn=cmd_check_config)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
