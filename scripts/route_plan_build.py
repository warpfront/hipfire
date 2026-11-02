#!/usr/bin/env -S python3 -I
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Build a checked-in route plan from route walks (route_plan_walk.py).

Every (module, symbol) the walked dispatch bound is attributed through its module
handle to the HIPFIRE_MODULE_LOAD_AUDIT record that loaded it (HIP: source SHA-256
and recipe; embedded: image SHA-256). The `ar` rows are the union of the AR walks,
the `native_mtp` rows the union of every walk (the MTP route falls back to AR).
`kernel_registry::route_entries` reads the result; its tests check that every row
resolves to the registry recipe or embedded image with the recorded identity.

Policy axes come from the walk's own load log, never from the command line:
  * Flash-Next (model arch 16): the `qwen4 state:` / `QSA F16 arena` / `pooled BF16
    mirror` lines (axes qsa, kv_backend, gdn, qsa_f16_arena, qsa_pooled_bf16).
  * Qwen3.5 (model arch 5 dense, 6 MoE: A3B, Qwen3.6, Ornith): the `route plan axes:`
    line the loader prints under HIPFIRE_MODULE_LOAD_AUDIT=1 (axes kv_k, kv_v,
    state_quant, kv_backend, prefill_chunk_rows, num_experts, experts_per_tok,
    expert_placement; plus `#! heads` and the spec route, which must
    match the walk's route). Each Qwen3.5 walk's model header must hash to the plan
    layout; a native-MTP walk needs `--mtp-head` (the sidecar or bundled-trailer head
    header), whose trunk+head layout is recorded as `#! mtp_layout`.

A plan is written open. `#! sealed ROUTES` is written only for routes given a
`--sealed-replay ROUTE=DIR`: a `route_plan_walk.py --sealed` run of the same walk plan
against the checked-in plan, whose stderr must show the plan preloaded and sealed
(HIPFIRE_ROUTE_PLAN_SEAL=1), no unplanned or predicted kernel, the planned module count
and the walked axes, and whose requests all passed. Build once without replays, check
the plan in, replay it, then rebuild with the replays.

usage: route_plan_build.py --out PLAN.tsv --family NAME --model M
                           --walk ar=DIR [--walk ar=DIR ...] --walk native_mtp=DIR [...]
                           [--mtp-head FILE[@OFFSET]] [--sealed-replay ar=DIR ...]
"""
import argparse
import hashlib
import json
import pathlib
import re
import struct

# hipfire-runtime loader_api.rs `kernel_dtype_for_quant_type`, by DType Debug name.
DTYPES = {1: "F16", 2: "F32", 3: "Q8_0", 6: "HFQ4G256", 13: "MQ4G256", 15: "MQ6G256", 16: "BF16",
          44: "MQ4G256V2", 47: "MQ6G256V2", 48: "MQ5G256V2", 49: "MQ3G256V2", 50: "MQ2G256V2",
          53: "MQ4G128V2", 55: "FP8E4M3B128", 56: "FP8E4M3PT"}
# The `qwen4 state:` QSA description (hipfire-loader carriers.rs) to the axis value
# `resolve_qwen4_route_plan` derives from `QsaKvFormat`.
QSA_NAMES = {"fp8": "fp8", "fp8 (gfx942 FNUZ codes, F32 index keys)": "fp8-fnuz",
             "q8 (Q8_0 blocks of 32)": "q8", "bf16 (exact F32 state)": "f32"}
# `resolve_route_plan` (hipfire-loader lib.rs) names, in the loader's order.
QWEN35_AXES = ("kv_k", "kv_v", "state_quant", "kv_backend", "prefill_chunk_rows",
               "num_experts", "experts_per_tok", "expert_placement")
QWEN35_FACTS = ("arch", "model_arch", "spec", "heads", "max_seq")
AXES_LINE = re.compile(r"^  route plan axes: (.+)$")
SPEC_OF_ROUTE = {"ar": "Ar", "native_mtp": "NativeMtp"}
WALK_ROUTE_OF = {"ar": "ar", "native_mtp": "mtp"}
# Host-run noise in a walk's inherited HIPFIRE_* environment: per-run paths, the fabric admission token and
# the torch toolchain locations steer no kernel, so they stay out of the plan's receipts.
RECEIPT_ENV_SKIP = ("HIPFIRE_HOME", "HIPFIRE_LOCK_DIR", "HIPFIRE_KERNEL_CACHE", "HIPFIRE_DEVICES",
                    "HIPFIRE_MODULE_LOAD_AUDIT", "HIPFIRE_FABRIC_ADMISSION")
RECEIPT_ENV_SKIP_PREFIX = ("HIPFIRE_TORCH_",)


def header_tensors(model, offset=0):
    """(HFQM arch id, [(name, dtype, shape)]) of every header tensor; reads no payload.
    `offset` is where an HFQM header starts inside the file (a bundled MTP trailer)."""
    with open(model, "rb") as f:
        f.seek(offset)
        magic, version, arch, count, meta, data = struct.unpack("<4sIIIQQ", f.read(32))
        assert magic == b"HFQM" and version == 1 and 32 <= meta <= data <= 64 << 20, model
        f.seek(offset)
        header = f.read(data)
    depth, quoted, escape, end = 0, False, False, None
    for i, b in enumerate(header[meta:], meta):
        if quoted:
            if escape:
                escape = False
            elif b == 92:
                escape = True
            elif b == 34:
                quoted = False
        elif b == 34:
            quoted = True
        elif b == 123:
            depth += 1
        elif b == 125:
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    pos = end
    (n,) = struct.unpack_from("<I", header, pos)
    pos += 4
    assert n == count
    out = []
    for _ in range(n):
        (length,) = struct.unpack_from("<H", header, pos)
        pos += 2
        name = header[pos:pos + length].decode()
        pos += length
        quant, ndim = struct.unpack_from("<BB", header, pos)
        pos += 2
        shape = struct.unpack_from("<" + "I" * ndim, header, pos)
        pos += 4 * ndim + 12
        out.append((name, DTYPES.get(quant, "Raw"), shape))
    return arch, out


def layout_sha256(tensors):
    """`kernel_registry::route_layout_sha256`: sorted `name\\tdtype\\tshape` lines."""
    lines = sorted(f"{name}\t{dtype}\t{','.join(str(d) for d in shape)}\n" for name, dtype, shape in tensors)
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def parse_axes_line(text):
    """`key=value ...` of a `route plan axes:` line."""
    fields = dict(part.split("=", 1) for part in text.split(" ") if "=" in part)
    missing = [k for k in (*QWEN35_FACTS, *QWEN35_AXES) if k not in fields]
    if missing:
        raise SystemExit(f"route plan axes line lacks {missing}: {text}")
    return fields


def summarize(walk, accept_partial=False):
    """(arch, hip, embedded, direct, policy, facts, command) observed by one walk.
    `facts` is the Qwen3.5 axes line's fields, None for a Flash-Next walk."""
    result = json.loads((walk / "result.json").read_text())
    partial = accept_partial and result.get("partial") and result["errors"] == 0
    if not result["ok"] and not partial:
        raise SystemExit(f"{walk}: walk not ok (rc={result['rc']}, errors={result['errors']})")
    pending = None
    handles = {}
    hip, emb, direct = {}, {}, []
    # Flash-Next policy axes, named as the loader's `resolve_qwen4_route_plan` names them.
    qwen4 = {"qsa_f16_arena": "off", "qsa_pooled_bf16": "off"}
    qwen4_seen = False
    axes_lines = []
    arch = None
    for line in (walk / "stderr.log").read_text(errors="replace").splitlines():
        state = re.match(r"  qwen4 state: QSA (.+) K/V \((\S+) context storage\), GDN (\S+) recurrent$", line)
        if state:
            qwen4_seen = True
            qwen4["qsa"] = QSA_NAMES[state.group(1)]
            qwen4["kv_backend"], qwen4["gdn"] = state.group(2), state.group(3)
        if line.startswith("  qwen4 QSA F16 arena: on"):
            qwen4["qsa_f16_arena"] = "on"
        if line.startswith("  qwen4 QSA pooled BF16 mirror: on"):
            qwen4["qsa_pooled_bf16"] = "on"
        axes = AXES_LINE.match(line)
        if axes:
            axes_lines.append(parse_axes_line(axes.group(1)))
        if not line.startswith("HIPFIRE_MODULE_LOAD_AUDIT\t"):
            continue
        parts = line.split("\t")
        kind = parts[1]
        if kind in ("hip", "embedded"):
            if pending is not None:
                raise SystemExit(f"{walk}: audit record without a module load: {pending}")
            rec = json.loads(parts[2])
            arch = arch or rec["arch"]
            pending = (kind, rec)
        elif kind in ("load", "load_data"):
            handle = parts[2]
            if pending is None:
                direct.append(line)
                handles[handle] = None
                continue
            handles[handle] = pending
            pending = None
        elif kind == "get_function":
            handle, symbol = parts[2], parts[3]
            if handle not in handles:
                raise SystemExit(f"{walk}: get_function on an unknown handle: {line}")
            entry = handles[handle]
            if entry is None:
                continue
            k, rec = entry
            if k == "hip":
                ident = (rec["source_sha256"], tuple(rec["flags"]), rec["scheduler_profile"])
                prev = hip.setdefault(rec["module"], {"ident": ident, "symbols": set()})
                if prev["ident"] != ident:
                    raise SystemExit(f"{walk}: two recipes for {rec['module']}")
                prev["symbols"].add(symbol)
            else:
                prev = emb.setdefault(rec["module"], {"sha": rec["image_sha256"], "symbols": set()})
                if prev["sha"] != rec["image_sha256"]:
                    raise SystemExit(f"{walk}: two images for {rec['module']}")
                prev["symbols"].add(symbol)
    if pending is not None:
        raise SystemExit(f"{walk}: trailing audit record without a module load")
    if qwen4_seen and axes_lines:
        raise SystemExit(f"{walk}: both a qwen4 state line and a route plan axes line in one load log")
    facts = None
    if axes_lines:
        if any(a != axes_lines[0] for a in axes_lines):
            raise SystemExit(f"{walk}: the walk loaded with more than one resolved route: {axes_lines}")
        facts = axes_lines[0]
        policy = {name: facts[name] for name in QWEN35_AXES}
    elif qwen4_seen:
        policy = qwen4
    else:
        raise SystemExit(f"{walk}: no qwen4 state line or route plan axes line in the load log "
                         "(a Qwen3.5 walk needs HIPFIRE_MODULE_LOAD_AUDIT=1 and a loader that logs its axes)")
    command = json.loads((walk / "command.json").read_text())
    return arch, hip, emb, direct, policy, facts, command


def merge(into, walk_hip, walk_emb):
    hip, emb = into
    for module, v in walk_hip.items():
        prev = hip.setdefault(module, {"ident": v["ident"], "symbols": set()})
        if prev["ident"] != v["ident"]:
            raise SystemExit(f"walks disagree on the recipe of {module}")
        prev["symbols"] |= v["symbols"]
    for module, v in walk_emb.items():
        prev = emb.setdefault(module, {"sha": v["sha"], "symbols": set()})
        if prev["sha"] != v["sha"]:
            raise SystemExit(f"walks disagree on the image of {module}")
        prev["symbols"] |= v["symbols"]


def parse_head_arg(spec):
    match = re.fullmatch(r"(.*)@(\d+)", spec)
    return (match.group(1), int(match.group(2))) if match else (spec, 0)


def check_walk_facts(walk, route, facts, model_arch, arch, command):
    """A Qwen3.5 walk's resolved route must be the one the plan row claims."""
    if facts["model_arch"] != str(model_arch):
        raise SystemExit(f"{walk}: loaded model arch {facts['model_arch']}, plan model is {model_arch}")
    if facts["arch"] != arch:
        raise SystemExit(f"{walk}: route resolved for {facts['arch']}, the audit ran on {arch}")
    if int(facts["max_seq"]) != command["load"]["params"]["max_seq"]:
        raise SystemExit(f"{walk}: resolved max_seq {facts['max_seq']} differs from the load request")
    if facts["spec"] != SPEC_OF_ROUTE[route]:
        raise SystemExit(f"{walk}: walked as {route} but the loader resolved the {facts['spec']} route "
                         "(a native_mtp walk needs the MTP head file next to the model)")


def check_replay(walk, route, plan_modules, plan_axes, plan_heads, plan_max_seq):
    """Receipt line of one sealed replay, or SystemExit when it does not prove the route closed."""
    result = json.loads((walk / "result.json").read_text())
    if not result["ok"]:
        raise SystemExit(f"{walk}: sealed replay not ok (rc={result['rc']}, errors={result['errors']})")
    command = json.loads((walk / "command.json").read_text())
    env = command["env"]
    if command["route"] != WALK_ROUTE_OF[route]:
        raise SystemExit(f"{walk}: replay ran route {command['route']}, expected {WALK_ROUTE_OF[route]}")
    if env.get("HIPFIRE_ROUTE_PLAN_SEAL") != "1" or env.get("HIPFIRE_ROUTE_PLAN") == "0":
        raise SystemExit(f"{walk}: not a sealed replay (needs route_plan_walk.py --sealed)")
    if command["load"]["params"]["max_seq"] > plan_max_seq:
        raise SystemExit(f"{walk}: replay max_seq above the walked {plan_max_seq}")
    spec, modules, sealed, axes = SPEC_OF_ROUTE[route], None, False, []
    for line in (walk / "stderr.log").read_text(errors="replace").splitlines():
        m = re.match(rf"  kernels: {spec} route plan \((\d+) modules\)", line)
        if m:
            modules = int(m.group(1))
        if line.startswith(f"  kernels: {spec} route preloaded and sealed"):
            sealed = True
        if "unplanned kernel" in line or "predicted" in line:
            raise SystemExit(f"{walk}: replay used off-plan or predicted kernels: {line.strip()}")
        m = AXES_LINE.match(line)
        if m:
            axes.append(parse_axes_line(m.group(1)))
    if modules is None or not sealed:
        raise SystemExit(f"{walk}: no '{spec} route plan' + 'route preloaded and sealed' lines: the checked-in plan "
                         "did not apply to the replay")
    if modules != plan_modules:
        raise SystemExit(f"{walk}: replay planned {modules} modules, this plan has {plan_modules} for {route}")
    if plan_axes is not None:
        if not axes:
            raise SystemExit(f"{walk}: replay logged no route plan axes line")
        for fields in axes:
            if (tuple(int(x) for x in fields["heads"].split(",")) != plan_heads
                    or any(fields[name] not in plan_axes[name] for name in QWEN35_AXES)):
                raise SystemExit(f"{walk}: replay resolved axes outside the walked plan: {fields}")
    return (f"# replay {route} {walk.name} daemon_sha256={command['daemon_sha256'][:16]} "
            f"requests={command['requests']} sealed_modules={modules}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--model", required=True, help="the walked model (layout fingerprint)")
    ap.add_argument("--walk", action="append", required=True, help="ROUTE=DIR, ROUTE in ar|native_mtp")
    ap.add_argument("--mtp-head", help="Qwen3.5 native-MTP head header, FILE or FILE@OFFSET for a bundled trailer; "
                                       "its tensors join the trunk's in the native-MTP layout")
    ap.add_argument("--sealed-replay", action="append", default=[],
                    help="ROUTE=DIR: a route_plan_walk.py --sealed run of the checked-in plan; seals that route")
    ap.add_argument("--accept-partial", action="store_true",
                    help="take a walk the walker recorded as partial (`partial: true` in result.json: the daemon was "
                         "killed by a timeout after N error-free requests); its audit is a valid, smaller walk")
    ap.add_argument("--no-verify-walk-model", action="store_true",
                    help="skip hashing each Qwen3.5 walk's model header against the plan layout")
    a = ap.parse_args()

    model_arch, tensors = header_tensors(a.model)
    layout = layout_sha256(tensors)
    routes = {"ar": ({}, {}), "native_mtp": ({}, {})}
    arch, max_seq, axes, receipts, direct_total = None, set(), {}, [], 0
    heads, qwen35_walks, native_mtp_walks = set(), 0, 0
    for spec in a.walk:
        route, path = spec.split("=", 1)
        if route not in SPEC_OF_ROUTE:
            raise SystemExit(f"unknown route {route}")
        walk = pathlib.Path(path)
        w_arch, hip, emb, direct, policy, facts, command = summarize(walk, a.accept_partial)
        if arch not in (None, w_arch):
            raise SystemExit("walks span two architectures")
        arch = w_arch
        if facts is not None:
            qwen35_walks += 1
            check_walk_facts(walk, route, facts, model_arch, w_arch, command)
            heads.add(facts["heads"])
            if not a.no_verify_walk_model:
                walked_model = pathlib.Path(command["load"]["model"])
                if not walked_model.is_file():
                    raise SystemExit(f"{walk}: walked model {walked_model} is gone; pass --no-verify-walk-model "
                                     "to trust the walk's own load log")
                walked_arch, walked_tensors = header_tensors(walked_model)
                if walked_arch != model_arch or layout_sha256(walked_tensors) != layout:
                    raise SystemExit(f"{walk}: its model {walked_model} is not the plan layout {layout}")
        elif qwen35_walks:
            raise SystemExit(f"{walk}: a Flash-Next walk cannot join Qwen3.5 walks in one plan")
        merge(routes["native_mtp"], hip, emb)
        if route == "ar":
            merge(routes["ar"], hip, emb)
        else:
            native_mtp_walks += 1
        max_seq.add(command["load"]["params"]["max_seq"])
        for name, value in policy.items():
            axes.setdefault(name, set()).add(value)
        direct_total += len(direct)
        env = {k: v for k, v in command["env"].items()
               if k.startswith("HIPFIRE_") and k not in RECEIPT_ENV_SKIP and not k.startswith(RECEIPT_ENV_SKIP_PREFIX)}
        receipts.append(f"# walk {route} {walk.name} daemon_sha256={command['daemon_sha256'][:16]} "
                        f"model={command['load']['model']} requests={command['requests']} env={json.dumps(env, sort_keys=True)}"
                        f" direct_loads={len(direct)}")
    if len(max_seq) != 1:
        raise SystemExit(f"walks disagree on max_seq: {max_seq}")
    if qwen35_walks and len(heads) != 1:
        raise SystemExit(f"walks disagree on heads/kv heads/head dim: {heads}")
    if native_mtp_walks and qwen35_walks and not a.mtp_head:
        raise SystemExit("a Qwen3.5 native_mtp walk needs --mtp-head (the head header its load passed to the planner)")
    mtp_layout = None
    if a.mtp_head:
        if not qwen35_walks:
            raise SystemExit("--mtp-head is for Qwen3.5 plans (a Flash-Next header carries its own head)")
        head_path, head_offset = parse_head_arg(a.mtp_head)
        _, head_tensors = header_tensors(head_path, head_offset)
        mtp_layout = layout_sha256([*tensors, *head_tensors])

    counts, route_modules = {}, {}
    for route in ("ar", "native_mtp"):
        hip, emb = routes[route]
        route_modules[route] = len(hip) + len(emb)
    sealed = []
    for spec in a.sealed_replay:
        route, path = spec.split("=", 1)
        if route not in SPEC_OF_ROUTE:
            raise SystemExit(f"unknown replay route {route}")
        receipts.append(check_replay(pathlib.Path(path), route, route_modules[route],
                                     axes if qwen35_walks else None,
                                     tuple(int(x) for x in next(iter(heads)).split(",")) if qwen35_walks else None,
                                     next(iter(max_seq))))
        if route not in sealed:
            sealed.append(route)
    lines = [
        "# Generated by scripts/route_plan_build.py from HIPFIRE_MODULE_LOAD_AUDIT route walks",
        "# (scripts/route_plan_walk.py); do not edit. Columns: route kind module symbols sha256",
        "# (HIP: effective source; embedded: image). native_mtp rows include every ar row.",
        f"#! family {a.family}",
        f"#! model_arch {model_arch}",
        f"#! arch {arch}",
        f"#! layout {layout}",
        *([f"#! mtp_layout {mtp_layout}"] if mtp_layout else []),
        *([f"#! heads {next(iter(heads))}"] if heads else []),
        f"#! max_seq {max_seq.pop()}",
        *(f"#! axis {name} {' '.join(sorted(values))}" for name, values in sorted(axes.items())),
        *([f"#! sealed {' '.join(r for r in ('ar', 'native_mtp') if r in sealed)}"] if sealed else []),
        *receipts,
    ]
    for route in ("ar", "native_mtp"):
        hip, emb = routes[route]
        for module, v in sorted(hip.items()):
            lines.append(f"{route}\thip\t{module}\t{','.join(sorted(v['symbols']))}\t{v['ident'][0]}")
        for module, v in sorted(emb.items()):
            lines.append(f"{route}\tembedded\t{module}\t{','.join(sorted(v['symbols']))}\t{v['sha']}")
        counts[route] = {"hip": len(hip), "embedded": len(emb),
                         "symbols": sum(len(v["symbols"]) for v in [*hip.values(), *emb.values()])}
    pathlib.Path(a.out).write_text("\n".join(lines) + "\n")
    print(json.dumps({"arch": arch, "routes": counts, "direct_loads": direct_total, "sealed": sealed}))


if __name__ == "__main__":
    main()
