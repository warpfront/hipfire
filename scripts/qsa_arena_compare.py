#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed comparison of qwen4_qsa_arena_check stdout across arms.

usage: qsa_arena_compare.py ARM=STDOUT [ARM=STDOUT ...]   (first arm = reference, e.g. off)

Every arm must be a PASS run on the same device (arch + PCI bus id) with
exactly the expected hash lines; every QSA_ARENA_HASH label must be equal in
every arm. Exit 0 only when all hold. Nothing is inferred from missing lines.
"""
import json
import re
import sys

LABELS = ["tokens", "selection", "attention", "state", "logits", "ids", "canonical"]
HASH = re.compile(r"^QSA_ARENA_HASH (\w+) sha256=([0-9a-f]{64}) md5=([0-9a-f]{32})$")
LAYER_HASH = re.compile(r"^ARENA_LAYER_HASH slot=(\d+) sha256=([0-9a-f]{64}) md5=([0-9a-f]{32})$")


def parse(path):
    out = {"hashes": {}, "layers": {}, "device": None, "route": None, "result": [], "fails": []}
    with open(path) as f:
        for line in f.read().splitlines():
            if line.startswith("ARENA_DEVICE "):
                if out["device"] is not None:
                    raise SystemExit(f"{path}: duplicate ARENA_DEVICE")
                out["device"] = json.loads(line[len("ARENA_DEVICE "):])
            elif line.startswith("ARENA_ROUTE "):
                if out["route"] is not None:
                    raise SystemExit(f"{path}: duplicate ARENA_ROUTE")
                out["route"] = json.loads(line[len("ARENA_ROUTE "):])
            elif line.startswith("ARENA_FAIL "):
                out["fails"].append(line)
            elif line.startswith("ARENA_RESULT "):
                out["result"].append(line)
            else:
                m = HASH.match(line)
                if m:
                    if m.group(1) in out["hashes"]:
                        raise SystemExit(f"{path}: duplicate hash label {m.group(1)}")
                    out["hashes"][m.group(1)] = (m.group(2), m.group(3))
                m = LAYER_HASH.match(line)
                if m:
                    slot = int(m.group(1))
                    if slot in out["layers"]:
                        raise SystemExit(f"{path}: duplicate QSA layer {slot}")
                    out["layers"][slot] = (m.group(2), m.group(3))
    return out


def main():
    arms = []
    for arg in sys.argv[1:]:
        name, _, path = arg.partition("=")
        if not path:
            raise SystemExit(__doc__)
        arms.append((name, parse(path)))
    if len(arms) < 2:
        raise SystemExit(__doc__)
    errors = []
    for name, arm in arms:
        if arm["result"] != ["ARENA_RESULT PASS"] or arm["fails"]:
            errors.append(f"{name}: result {arm['result']} fails {arm['fails']}")
        if arm["device"] is None or arm["route"] is None:
            errors.append(f"{name}: missing ARENA_DEVICE/ARENA_ROUTE")
        if sorted(arm["hashes"]) != sorted(LABELS):
            errors.append(f"{name}: hash labels {sorted(arm['hashes'])}")
        if arm["route"] is not None:
            count = arm["route"].get("qsa_layers")
            if count is None or sorted(arm["layers"]) != list(range(count)):
                errors.append(f"{name}: incomplete per-layer attention hashes")
            if arm["route"].get("index_dtype") != "F32":
                errors.append(f"{name}: non-F32 QSA index state")
    if errors:
        raise SystemExit("\n".join(errors))
    if {arm["route"].get("f16_arena_present") for _, arm in arms} != {False, True}:
        raise SystemExit("comparison needs an arena-off and an arena-on process (set the kill switch explicitly)")
    ref_name, ref = arms[0]
    for name, arm in arms[1:]:
        for key in ("arch", "pci", "device_id"):
            if arm["device"][key] != ref["device"][key]:
                errors.append(f"{name}: device {key} {arm['device'][key]} != {ref['device'][key]}")
        if arm["device"]["prompt"] != ref["device"]["prompt"]:
            errors.append(f"{name}: prompt differs from {ref_name}")
        for key in ("model", "chunk", "decode"):
            if arm["device"][key] != ref["device"][key]:
                errors.append(f"{name}: {key} differs from {ref_name}")
        if arm["layers"] != ref["layers"]:
            errors.append(f"{name}: per-layer attention differs from {ref_name}")
        for label in LABELS:
            if arm["hashes"][label] != ref["hashes"][label]:
                errors.append(f"{name}: {label} differs from {ref_name}")
    for name, arm in arms:
        print(f"{name}: flags={arm['device']['flags']} index_dtype={arm['route']['index_dtype']} "
              f"arena={arm['route']['f16_arena_present']} canonical={arm['hashes']['canonical'][0][:16]}")
    if errors:
        raise SystemExit("\n".join(errors))
    print("ARENA_ARMS_EQUAL")


if __name__ == "__main__":
    main()
