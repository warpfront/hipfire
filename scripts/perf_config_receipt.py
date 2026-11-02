#!/usr/bin/env python3
"""Stamp a perf receipt with the startup configuration a daemon log proves.

Reads a Flash-Next (Qwen4) startup log, extracts raw evidence lines plus
normalized fields, and writes two keys into the receipt JSON, leaving every
other key untouched:

  startup_config        what the log shows (raw lines + normalized fields)
  product_default_check status match|mismatch|unknown + reasons[]

Usage:
  perf_config_receipt.py --log daemon.log --arch gfx1151 --receipt result.json
  perf_config_receipt.py --log daemon.log --arch gfx1201 --receipt result.json \\
      --model /models/qwen3.8-flash-next-base.mq4-xts

Extracted (all from the log; nothing is assumed when a line is absent):
  PLE rows            `qwen4 ple rows:` format (q4/q8), origin (sidecar path or
                      embedded model file), the q4-sidecar-not-found fallback
  expert hot set      applied (sha/counts/identity), present-not-applied,
                      explicit disable, or absent; plus expert-placement lines
                      and whole-layer placement evidence
  trunk overlay       applied (tensor count, bound base) / not applied / refused
  npu-trunk           mode, calibration, init line, off/WARN reason, per-request
                      route plans (gpu vs pipe steps)
  kernel route plans  `[no closed route plan: ...]`, `<Kind> route plan (N modules)`
  memlock / host mem  RLIMIT_MEMLOCK lines, HSA_USERPTR/GPU_PINNED hip-bridge line
  PLE hot store       pack active / off / failed (loader is silent at the
                      default budget 0)
  env pins            HIPFIRE_* NAME=value tokens and memory.*/fusions.* key
                      pins found in the log, with a normalized config-key map
  model               --model path and, only when --model is given, a CPU
                      sha256 of the whole file (slow on 100+ GB models); the
                      log's own hotset-identity model_sha256 / overlay base
                      binding are kept as evidence, never relabelled as computed
  layout              which model layout produced the log, recognised from the
                      identities the log or --model proves: the split base
                      (qwen3.8-flash-next-base.mq4-xts, sha256 eafaff94...,
                      70,888,270,360 B) or the grandfathered monolithic file
                      (sha256 8b15b6fe..., 125,288,544,792 B)

Missing log line = "unknown" (or "not logged"); an explicit pin / log line that
turns a feature off = "disabled". Those are never conflated.

Product defaults checked (fresh `mq4` pull: the split base
`qwen3.8-flash-next-base.mq4-xts` plus the shared PLE sidecar pair; the base has
no embedded PLE rows):
  PLE rows       gfx1151: q4 from a sidecar (rank order; `--ple auto` picks q4
                 there).  gfx1201: q8 from a sidecar (row order).  Embedded q8
                 rows are a grandfathered monolithic install, not a fresh-pull
                 layout: MISMATCH.  q8 on gfx1151 or q4 on gfx1201 is an
                 explicit override, not the default: MISMATCH.
  gfx1151        trunk overlay applied (108 tensors); npu-trunk left at mode auto
                 (not pinned off/on), either armed with an eligible calibration
                 and request plan, or a logged GPU-only fallback (not a mismatch;
                 the reason is recorded in notes); expert hot set not applied
                 (gfx1201 only).
  gfx1201        expert hot set applied; absent hot set with whole-layer
                 placement is a MISMATCH; overlay / NPU absent or "not applied
                 on gfx1201" is expected, not a mismatch.
  PLE hot store  default 0 (inactive).
Layout identity is evidence, not a check: a fresh pull's overlay is bound to the
split base (eafaff94...) and its hot set carries model_sha256 eafaff94...; the
old monolith-bound pins (8b15b6fe...) are still recognised and labelled
"monolith (grandfathered)" in `product_default_check.log_layout`.
Status is mismatch if any check mismatches, else unknown if any is unknown,
else match.  Exit status is 0 on success, 1 if --model hashing failed (receipt
still written), 2 for unusable inputs.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import tempfile

SCHEMA_VERSION = 1
RAW_CAP = 40  # raw lines kept per category (counts always cover every line)
EXPECTED_OVERLAY_TENSORS = 108

# Model layout identities (whole-file sha256 / byte size).
SPLIT_BASE_NAME = "qwen3.8-flash-next-base.mq4-xts"
SPLIT_BASE_SHA256 = "eafaff94fb9c6bb51282942e0ea6441cbdc266b634bb003b5913cfe2ccb90f45"
SPLIT_BASE_BYTES = 70888270360
MONOLITH_NAMES = ("qwen3.8-flash-next.mq4-xts", "qwen3.8-flash-next-gptq3.mq4")
MONOLITH_SHA256 = "8b15b6fede7d7c5bfed0db4720a8295bedda51bc93e545fa242bd50d0f200972"
MONOLITH_BYTES = 125288544792
LAYOUT_SPLIT = "split base (fresh pull)"
LAYOUT_MONOLITH = "monolith (grandfathered)"
LAYOUT_UNKNOWN = "unknown"


ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")

ENV_PIN_RE = re.compile(r"\b(HIPFIRE_[A-Z0-9_]+)\s*=\s*([^\s,;)\"'\]}]*)")
CFG_PIN_RE = re.compile(
    r"\b(memory\.ple_format|memory\.ple_hot_store_bytes|fusions\.npu_trunk_calibration"
    r"|fusions\.npu_trunk)\s*[=:]\s*\"?([^\s\",;)]+)")
# Only mappings confirmed in crates/hipfire-config; others stay env-only.
ENV_TO_CONFIG_KEY = {
    "HIPFIRE_PLE_FORMAT": "memory.ple_format",
    "HIPFIRE_PLE_HOT_STORE_BYTES": "memory.ple_hot_store_bytes",
    "HIPFIRE_QWEN4_NPU_TRUNK": "fusions.npu_trunk",
    "HIPFIRE_QWEN4_NPU_TRUNK_CAL": "fusions.npu_trunk_calibration",
}
OFF_VALUES = {"0", "off", "false", "no", "disabled", "none"}

PLE_ROWS_RE = re.compile(
    r"qwen4 ple rows: (?P<fmt>\w+) from (?P<src>.+?) "
    r"\((?P<enc>[^,()]+), (?P<order>\w+) order, (?P<gb>[\d.]+) GB\)\s*$")
PLE_NONE_RE = re.compile(r"qwen4 ple rows: none \(profile (?P<profile>[^)]*) declares no PLE\)")
PLE_DEV_RE = re.compile(r"qwen4 ple rows: embedded developer rows")
PLE_FALLBACK_RE = re.compile(r"qwen4 ple rows: (?P<note>q4 PLE sidecar not found.*)$")
PLE_TABLE_RE = re.compile(
    r"qwen4 ple hot store: table (?P<path>\S+): (?P<rows>\d+) rows in (?P<tiers>\d+) tiers")

HOTSET_APPLIED_RE = re.compile(
    r"qwen4 expert hot set: (?P<path>\S+) sha256 (?P<sha>[0-9a-f]+), budget (?P<budget>\d+) MiB, "
    r"(?P<vram>\d+) experts in VRAM \((?P<vram_mib>\d+) MiB\), "
    r"(?P<host>\d+) host-mapped \((?P<host_mib>\d+) MiB\), generation (?P<gen>\d+), "
    r"digest (?P<digest>[0-9a-f]+), bound to base (?P<base>\d+) B, index sha256 (?P<index>[0-9a-f]+)")
HOTSET_NOT_APPLIED_RE = re.compile(
    r"qwen4 expert hot set: (?P<path>.+?) present, not applied \((?P<reason>.*)\); whole-layer placement")
HOTSET_IDENTITY_RE = re.compile(r"qwen4 expert hot set identity: (?P<rest>.*)$")
PLACEMENT_RE = re.compile(r"qwen4 expert placement: (?P<what>.*)$")
AUTO_PLACEMENT_RE = re.compile(
    r"qwen4 auto expert placement: (?P<free>\d+) MiB free, (?P<non_expert>\d+) MiB non-expert weights, "
    r"(?P<reserved>\d+) MiB reserved.*?, (?P<per_layer>\d+) MiB per expert layer")

OVERLAY_APPLIED_RE = re.compile(
    r"qwen4 trunk overlay: applied (?P<n>\d+) (?P<kind>\w+) tensors from (?P<path>\S+) "
    r"\(default (?P<default>\S+); bound to base (?P<base>\d+) B, index sha256 (?P<index>[0-9a-f]+), "
    r"file sha256 (?P<file>[0-9a-f]+); (?P<before>\d+) -> (?P<after>\d+) bytes\)")
OVERLAY_NOT_APPLIED_RE = re.compile(
    r"qwen4 trunk overlay: (?P<path>\S+) present, not applied (?P<why>.*)$")
OVERLAY_REFUSED_RE = re.compile(
    r"qwen4 trunk overlay: REFUSED (?P<path>\S+): (?P<reason>.*?); base weights")

NPU_INIT_RE = re.compile(
    r"npu-trunk: mode=(?P<mode>\w+) families=(?P<families>\S+) contract=(?P<contract>\S+) "
    r"trunk_route=(?P<route>\S+)(?: zab_source=(?P<zab>\S+))?")
NPU_CAL_RE = re.compile(
    r"npu-trunk: mode=(?P<mode>\w+) contract=(?P<contract>\S+) families=(?P<families>\S+) "
    r"cal_key=(?P<key>\{.*\}) cal_sha256=(?P<sha>[0-9a-f]+)(?: n\*=(?P<nstar>\S+))?")
NPU_FORCED_NOCAL_RE = re.compile(
    r"npu-trunk: mode=(?P<mode>\w+) calibration=none \(forced\) reason=(?P<reason>.*)$")
NPU_OFF_RE = re.compile(r"npu-trunk: off \((?P<reason>.*?)\)(?:; npu_trunk=auto.*)?$")
NPU_BUILTIN_RE = re.compile(r"npu-trunk: built-in Halo calibration")
NPU_PLAN_RE = re.compile(
    r"npu-trunk: rows=(?P<rows>\d+) plan=\[(?P<steps>.*?)\] predicted=(?P<pred>[+-]?[\d.]+)%")
NPU_PAIR_RE = re.compile(r"npu-trunk: pair rows=")

ROUTE_NONE_RE = re.compile(r"\[no closed route plan: (?P<reason>.*)\]\s*$")
ROUTE_KIND_RE = re.compile(r"kernels: (?P<kind>\w+) route plan \((?P<n>\d+) modules\)")

MEMLOCK_SOFT_RE = re.compile(
    r"RLIMIT_MEMLOCK soft limit raised (?P<frm>\d+) -> (?P<to>\d+) B for (?P<need>\d+) B of userptr pins "
    r"\(hard limit (?P<hard>\d+) B\)")
MEMLOCK_HARD_RE = re.compile(
    r"RLIMIT_MEMLOCK hard limit (?P<hard>\d+) B is below the (?P<required>\d+) B the NPU needs to pin")
MEMLOCK_ERR_RE = re.compile(r"(?:getrlimit|setrlimit)\(RLIMIT_MEMLOCK.*errno=(?P<errno>\d+)")
HIP_BRIDGE_RE = re.compile(
    r"\[hip-bridge\] HSA_USERPTR_FOR_PAGED_MEM=(?P<userptr>\S+) GPU_PINNED_MIN_XFER_SIZE=(?P<pinned>\S+)"
    r"(?: \((?P<ctx>[^)]*)\))?")
ARCH_RE = re.compile(r"\b(gfx\d{3,4}[a-z]?)\b")
GPU_DEV_RE = re.compile(r"GPU dev \d+: (?P<arch>gfx\w+)")
DEVICES_RE = re.compile(r"\[devices\].*? index \d+ (?P<arch>gfx\w+)")

# Lines whose HIPFIRE_*=value text is advice, not a pin.
PIN_SKIP_MARKERS = ("qwen4 trunk overlay:", "sidecar not found", "is missing",
                    "HIPFIRE_MODULE_LOAD_AUDIT")



class Raw:
    """Raw evidence lines per category, capped, with total counts."""

    def __init__(self):
        self.lines = {}
        self.counts = {}

    def add(self, category, lineno, text):
        self.counts[category] = self.counts.get(category, 0) + 1
        bucket = self.lines.setdefault(category, [])
        if len(bucket) < RAW_CAP:
            bucket.append({"line": lineno, "text": text})


def sha256_file(path, chunk=8 * 1024 * 1024):
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
            size += len(block)
    return h.hexdigest(), size


def parse_log(path):
    raw = Raw()
    st = {
        "ple": None, "ple_fallback_note": None, "ple_table": None,
        "hot_store_events": [], "hotset_applied": None, "hotset_not_applied": None,
        "hotset_identity": None, "placement": None, "auto_placement": None,
        "overlay": None, "npu_init": None, "npu_cal": None, "npu_off": None,
        "npu_forced_nocal": None, "npu_builtin": False, "npu_plans": [], "npu_pairs": 0,
        "route_none": [], "route_kinds": [], "memlock": [], "host_memory": None,
        "arch": [], "pins": {}, "cfg_pins": {}, "lines": 0,
    }
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, 1):
            st["lines"] = lineno
            text = ANSI_RE.sub("", line.rstrip("\r\n"))
            if not text.strip():
                continue
            scan_line(st, raw, lineno, text)
    return st, raw


def scan_line(st, raw, lineno, text):
    # env / config pins
    if not any(m in text for m in PIN_SKIP_MARKERS):
        for m in ENV_PIN_RE.finditer(text):
            st["pins"].setdefault(m.group(1), []).append({"line": lineno, "value": m.group(2)})
            raw.add("env_pins", lineno, text)
        for m in CFG_PIN_RE.finditer(text):
            st["cfg_pins"].setdefault(m.group(1), []).append({"line": lineno, "value": m.group(2)})
            raw.add("env_pins", lineno, text)

    # architecture evidence
    m = GPU_DEV_RE.search(text) or DEVICES_RE.search(text)
    if m:
        if m.group("arch") not in st["arch"]:
            st["arch"].append(m.group("arch"))
        raw.add("arch", lineno, text)

    if "ple rows:" in text:
        raw.add("ple", lineno, text)
        m = PLE_ROWS_RE.search(text)
        if m:
            src = m.group("src")
            embedded = src == "embedded model file"
            st["ple"] = {
                "format": m.group("fmt"),
                "origin": "embedded" if embedded else "sidecar",
                "path": None if embedded else src,
                "encoding": m.group("enc"),
                "order": m.group("order"),
                "size_gb": float(m.group("gb")),
                "raw_line": text, "line": lineno,
            }
        elif PLE_NONE_RE.search(text):
            st["ple"] = {"format": "none", "origin": None, "path": None, "encoding": None,
                         "order": None, "size_gb": None,
                         "profile": PLE_NONE_RE.search(text).group("profile"),
                         "raw_line": text, "line": lineno}
        elif PLE_DEV_RE.search(text):
            st["ple"] = {"format": "developer-embedded", "origin": "embedded", "path": None,
                         "encoding": None, "order": None, "size_gb": None,
                         "raw_line": text, "line": lineno}
        else:
            m = PLE_FALLBACK_RE.search(text)
            if m:
                st["ple_fallback_note"] = {"note": m.group("note"), "raw_line": text, "line": lineno}
        return

    if "ple hot store" in text or "ple prewarm" in text or "ple load fill" in text:
        m = PLE_TABLE_RE.search(text)
        if m:
            raw.add("ple_table_store", lineno, text)
            st["ple_table"] = {"path": m.group("path"), "rows": int(m.group("rows")),
                               "tiers": int(m.group("tiers")), "serving": "; serving" in text,
                               "raw_line": text, "line": lineno}
            return
        if "ple prewarm" in text or "load fill" in text:
            raw.add("ple_prewarm_fill", lineno, text)
            return
        raw.add("ple_hot_store", lineno, text)
        if ": off (" in text or "not attached to the row store" in text:
            st["hot_store_events"].append({"kind": "off", "raw_line": text, "line": lineno})
        elif "the pack is not used" in text:
            st["hot_store_events"].append({"kind": "failed", "raw_line": text, "line": lineno})
        elif ": pack " in text and "kept" in text:
            st["hot_store_events"].append({"kind": "pack_loaded", "raw_line": text, "line": lineno})
        elif "; serving" in text:
            st["hot_store_events"].append({"kind": "pack_serving", "raw_line": text, "line": lineno})
        return

    if "expert hot set" in text or "expert placement" in text:
        raw.add("expert_hotset" if "hot set" in text else "expert_placement", lineno, text)
        m = HOTSET_APPLIED_RE.search(text)
        if m:
            st["hotset_applied"] = {
                "path": m.group("path"), "sha256": m.group("sha"),
                "budget_mib": int(m.group("budget")),
                "experts_in_vram": int(m.group("vram")), "vram_mib": int(m.group("vram_mib")),
                "experts_host_mapped": int(m.group("host")), "host_mib": int(m.group("host_mib")),
                "generation": int(m.group("gen")), "digest": m.group("digest"),
                "bound_base_bytes": int(m.group("base")), "bound_index_sha256": m.group("index"),
                "raw_line": text, "line": lineno,
            }
            return
        m = HOTSET_NOT_APPLIED_RE.search(text)
        if m:
            st["hotset_not_applied"] = {"path": m.group("path"), "reason": m.group("reason"),
                                        "raw_line": text, "line": lineno}
            return
        m = HOTSET_IDENTITY_RE.search(text)
        if m:
            rest = m.group("rest")
            ident = {"raw_line": text, "line": lineno}
            for key in ("model_sha256", "counts_sha256", "format_map_sha256"):
                mm = re.search(r"\b" + key + r" ([0-9a-f]+)", rest)
                if mm:
                    ident[key] = mm.group(1)
            mm = re.search(r"\bformat_map ([^\s,]+)", rest)
            if mm:
                ident["format_map"] = mm.group(1)
            st["hotset_identity"] = ident
            return
        m = AUTO_PLACEMENT_RE.search(text)
        if m:
            st["auto_placement"] = {
                "free_mib": int(m.group("free")), "non_expert_mib": int(m.group("non_expert")),
                "reserved_mib": int(m.group("reserved")), "per_expert_layer_mib": int(m.group("per_layer")),
                "raw_line": text, "line": lineno}
            return
        m = PLACEMENT_RE.search(text)
        if m:
            st["placement"] = classify_placement(m.group("what"), text, lineno)
        return

    if "trunk overlay" in text:
        raw.add("overlay", lineno, text)
        m = OVERLAY_APPLIED_RE.search(text)
        if m:
            st["overlay"] = {
                "state": "applied", "tensors": int(m.group("n")), "tensor_kind": m.group("kind"),
                "path": m.group("path"), "default": m.group("default"),
                "bound_base_bytes": int(m.group("base")), "bound_index_sha256": m.group("index"),
                "file_sha256": m.group("file"),
                "bytes_before": int(m.group("before")), "bytes_after": int(m.group("after")),
                "raw_line": text, "line": lineno}
            return
        m = OVERLAY_REFUSED_RE.search(text)
        if m:
            st["overlay"] = {"state": "refused", "path": m.group("path"), "reason": m.group("reason"),
                             "raw_line": text, "line": lineno}
            return
        m = OVERLAY_NOT_APPLIED_RE.search(text)
        if m:
            why = m.group("why")
            ov = {"state": "present_not_applied", "path": m.group("path"), "why": why,
                  "raw_line": text, "line": lineno}
            if re.search(r"\(HIPFIRE_QWEN4_TRUNK_OVERLAY=(0|off|false)\)", why):
                ov["state"] = "disabled"
                ov["disabled_by"] = "HIPFIRE_QWEN4_TRUNK_OVERLAY=0"
            else:
                mm = re.match(r"on (gfx\w+) \(overlay default (\S+) is (gfx\w+) only", why)
                if mm:
                    ov["not_applied_on_arch"] = mm.group(1)
                    ov["default"] = mm.group(2)
                    ov["default_arch"] = mm.group(3)
            st["overlay"] = ov
        return

    if "npu-trunk" in text:
        raw.add("npu", lineno, text)
        m = NPU_PLAN_RE.search(text)
        if m:
            steps = [s.strip() for s in m.group("steps").split("|") if s.strip()]
            st["npu_plans"].append({
                "rows": int(m.group("rows")), "steps": steps,
                "pipe_steps": sum(1 for s in steps if s.startswith("pipe")),
                "gpu_steps": sum(1 for s in steps if s.startswith("gpu")),
                "predicted_percent": float(m.group("pred")), "line": lineno})
            return
        if NPU_PAIR_RE.search(text):
            st["npu_pairs"] += 1
            return
        m = NPU_CAL_RE.search(text)
        if m:
            try:
                key = json.loads(m.group("key"))
            except ValueError:
                key = None
            st["npu_cal"] = {"mode": m.group("mode"), "contract": m.group("contract"),
                             "families": m.group("families"), "cal_key": key,
                             "cal_sha256": m.group("sha"), "n_star": m.group("nstar"),
                             "raw_line": text, "line": lineno}
            return
        m = NPU_INIT_RE.search(text)
        if m and "trunk_route=" in text:
            rt = re.search(r"\bruntime=(\S+)", text)
            st["npu_init"] = {"mode": m.group("mode"), "families": m.group("families"),
                              "contract": m.group("contract"), "trunk_route": m.group("route"),
                              "zab_source": m.group("zab"), "runtime": rt.group(1) if rt else None,
                              "raw_line": text, "line": lineno}
            return
        m = NPU_FORCED_NOCAL_RE.search(text)
        if m:
            st["npu_forced_nocal"] = {"mode": m.group("mode"), "reason": m.group("reason"),
                                      "raw_line": text, "line": lineno}
            return
        if NPU_BUILTIN_RE.search(text):
            st["npu_builtin"] = True
            return
        m = NPU_OFF_RE.search(text)
        if m:
            st["npu_off"] = {"reason": m.group("reason"), "warn": text.lstrip().startswith("WARN"),
                             "raw_line": text, "line": lineno}
            if "RLIMIT_MEMLOCK" in m.group("reason"):
                scan_memlock(st, raw, lineno, text)
        return

    if "route plan" in text:
        raw.add("route_plan", lineno, text)
        m = ROUTE_NONE_RE.search(text)
        if m:
            st["route_none"].append({"reason": m.group("reason"), "line": lineno})
        m = ROUTE_KIND_RE.search(text)
        if m:
            st["route_kinds"].append({"kind": m.group("kind"), "modules": int(m.group("n")),
                                      "line": lineno})
        return

    if re.search(r"memlock", text, re.I):
        scan_memlock(st, raw, lineno, text)
        return

    m = HIP_BRIDGE_RE.search(text)
    if m:
        raw.add("host_memory", lineno, text)
        st["host_memory"] = {"hsa_userptr_for_paged_mem": m.group("userptr"),
                             "gpu_pinned_min_xfer_size": m.group("pinned"),
                             "context": m.group("ctx"), "raw_line": text, "line": lineno}


def scan_memlock(st, raw, lineno, text):
    raw.add("memlock", lineno, text)
    m = MEMLOCK_SOFT_RE.search(text)
    if m:
        st["memlock"].append({"kind": "soft_limit_raised", "from_bytes": int(m.group("frm")),
                              "to_bytes": int(m.group("to")), "need_bytes": int(m.group("need")),
                              "hard_bytes": int(m.group("hard")), "line": lineno})
        return
    m = MEMLOCK_HARD_RE.search(text)
    if m:
        st["memlock"].append({"kind": "hard_limit_too_low", "hard_bytes": int(m.group("hard")),
                              "required_bytes": int(m.group("required")), "line": lineno})
        return
    m = MEMLOCK_ERR_RE.search(text)
    if m:
        st["memlock"].append({"kind": "rlimit_error", "errno": int(m.group("errno")), "line": lineno})
        return
    st["memlock"].append({"kind": "other", "line": lineno})


def classify_placement(what, raw_line, lineno):
    out = {"raw": what, "raw_line": raw_line, "line": lineno, "mode": "unknown",
           "whole_layer": None}
    m = re.search(r"agreed split (\d+) of (\d+) MoE layers resident", what)
    if m:
        out.update(mode="agreed_split", layers_resident=int(m.group(1)),
                   moe_layers=int(m.group(2)), whole_layer=True)
    elif re.search(r"unset, fully resident", what):
        out.update(mode="fully_resident", whole_layer=False)
    elif re.search(r"_EXPERT_VRAM_LAYERS=auto\b", what) or re.search(r"unset, auto\b", what):
        out.update(mode="auto", whole_layer=True)
    else:
        m = re.search(r"_EXPERT_VRAM_LAYERS=(\d+)", what)
        if m:
            out.update(mode="explicit_layers", layers_resident=int(m.group(1)), whole_layer=True)
    if "unset" in what:
        out["env_pin"] = False
    return out


def pin_last(st, name):
    vals = st["pins"].get(name)
    return vals[-1]["value"] if vals else None


def is_off_value(v):
    return v is not None and v.strip().strip('"').lower() in OFF_VALUES


def detect_layout(model_sha, ident_sha, bound_bytes, base_paths):
    """Label the model layout from the strongest identity the log or --model proves.

    Order: whole-file sha256 (computed, then the hot-set identity line), bound
    base byte counts (hot set, then overlay), then the model file name implied by
    a sidecar path. Returns {"name", "source"}; unknown when nothing is recognised.
    """
    for source, sha in (("computed --model sha256", model_sha),
                        ("hot-set identity model_sha256", ident_sha)):
        if sha == SPLIT_BASE_SHA256:
            return {"name": LAYOUT_SPLIT, "source": source}
        if sha == MONOLITH_SHA256:
            return {"name": LAYOUT_MONOLITH, "source": source}
    for source, size in bound_bytes:
        if size == SPLIT_BASE_BYTES:
            return {"name": LAYOUT_SPLIT, "source": source}
        if size == MONOLITH_BYTES:
            return {"name": LAYOUT_MONOLITH, "source": source}
    for source, path in base_paths:
        name = os.path.basename(path)
        if name == SPLIT_BASE_NAME:
            return {"name": LAYOUT_SPLIT, "source": source}
        if name in MONOLITH_NAMES:
            return {"name": LAYOUT_MONOLITH, "source": source}
    return {"name": LAYOUT_UNKNOWN, "source": None}


def build_startup_config(st, raw, arch, model_path, model_sha, model_sha_err, model_size):
    ple = st["ple"]
    note = st["ple_fallback_note"]
    if ple is None and note and "using the embedded q8 rows" in note["note"]:
        # The fallback note alone states which rows load; no `ple rows:` line was logged.
        ple = {"format": "q8", "origin": "embedded", "path": None, "encoding": None,
               "order": None, "size_gb": None, "inferred_from_fallback_note": True,
               "raw_line": note["raw_line"], "line": note["line"]}
    ple_format = ple["format"] if ple else "unknown"
    ple_path = ple["path"] if ple else None
    if ple and ple["origin"] == "embedded":
        ple_format_detail = ple["format"] + "-embedded"
    elif ple and ple["origin"] == "sidecar":
        ple_format_detail = ple["format"] + "-sidecar"
    else:
        ple_format_detail = ple_format

    # expert hot set
    hs_pin = pin_last(st, "HIPFIRE_QWEN4_EXPERT_HOTSET")
    whole_layer = None
    if st["hotset_applied"]:
        whole_layer = False
    elif st["hotset_not_applied"]:
        whole_layer = True
    elif st["placement"] and st["placement"]["whole_layer"] is not None:
        whole_layer = st["placement"]["whole_layer"]
    if st["hotset_applied"]:
        hs_state = "applied"
    elif st["hotset_not_applied"]:
        hs_state = "present_not_applied"
    elif is_off_value(hs_pin):
        hs_state = "disabled"
    elif whole_layer:
        hs_state = "absent"
    else:
        hs_state = "unknown"
    hotset = {"state": hs_state, "env_pin": hs_pin,
              "applied": st["hotset_applied"], "not_applied": st["hotset_not_applied"],
              "identity": st["hotset_identity"]}

    placement = {"whole_layer": whole_layer, "expert_placement": st["placement"],
                 "auto_placement": st["auto_placement"]}

    # overlay
    ov = st["overlay"]
    ov_pin = pin_last(st, "HIPFIRE_QWEN4_TRUNK_OVERLAY")
    if ov:
        overlay = dict(ov)
    elif is_off_value(ov_pin):
        overlay = {"state": "disabled", "disabled_by": "HIPFIRE_QWEN4_TRUNK_OVERLAY=" + ov_pin}
    else:
        overlay = {"state": "unknown"}
    overlay["env_pin"] = ov_pin

    # npu
    init, cal, off = st["npu_init"], st["npu_cal"], st["npu_off"]
    npu_pin = pin_last(st, "HIPFIRE_QWEN4_NPU_TRUNK")
    if npu_pin is None and st["cfg_pins"].get("fusions.npu_trunk"):
        npu_pin = st["cfg_pins"]["fusions.npu_trunk"][-1]["value"]
    if init:
        npu_state, npu_mode = "armed", init["mode"]
    elif off:
        npu_state, npu_mode = "off", None
    elif st["npu_forced_nocal"]:
        npu_state, npu_mode = "forced_no_calibration", st["npu_forced_nocal"]["mode"]
    elif is_off_value(npu_pin):
        npu_state, npu_mode = "disabled", None
    elif cal and st["npu_plans"]:
        npu_state, npu_mode = "armed", cal["mode"]
    elif cal:
        npu_state, npu_mode = "calibrated_not_initialized", cal["mode"]
    else:
        npu_state, npu_mode = "unknown", None
    if npu_mode is None and cal:
        npu_mode = cal["mode"]
    plans = st["npu_plans"]
    # calibration + request plans without an init line (partial log) still show an armed trunk
    npu = {
        "state": npu_state, "mode": npu_mode, "env_pin": npu_pin,
        "init": init, "calibration": cal, "builtin_calibration_line": st["npu_builtin"],
        "off": off, "forced_no_calibration": st["npu_forced_nocal"],
        "calibration_env_pin": pin_last(st, "HIPFIRE_QWEN4_NPU_TRUNK_CAL"),
        "off_reason_memlock": bool(off and "RLIMIT_MEMLOCK" in off["reason"]),
    }

    route_plans = {
        "kernel": {"no_closed_route_plan": st["route_none"], "route_kinds": st["route_kinds"]},
        "npu_requests": {
            "count": len(plans),
            "gpu_only_count": sum(1 for p in plans if p["pipe_steps"] == 0),
            "with_pipe_count": sum(1 for p in plans if p["pipe_steps"] > 0),
            "pair_lines": st["npu_pairs"],
            "rows_min": min((p["rows"] for p in plans), default=None),
            "rows_max": max((p["rows"] for p in plans), default=None),
            "plans": plans[:RAW_CAP],
            "plans_truncated": len(plans) > RAW_CAP,
            "note": "plan lines are the planner's choice, not evidence the NPU executed them",
        },
    }

    # PLE hot store (pack) state; the loader prints nothing at the default 0 budget.
    hs_events = st["hot_store_events"]
    hot_pin = pin_last(st, "HIPFIRE_PLE_HOT_STORE_BYTES")
    if hot_pin is None and st["cfg_pins"].get("memory.ple_hot_store_bytes"):
        hot_pin = st["cfg_pins"]["memory.ple_hot_store_bytes"][-1]["value"]
    kinds = {e["kind"] for e in hs_events}
    if kinds & {"pack_loaded", "pack_serving"}:
        hot_state = "active"
    elif "failed" in kinds:
        hot_state = "failed"
    elif "off" in kinds:
        hot_state = "off"
    elif hot_pin is not None:
        hot_state = "disabled" if re.fullmatch(r"0+", hot_pin.strip('"')) else "requested_nonzero"
    elif ple:
        hot_state = "inactive_not_logged"
    else:
        hot_state = "unknown"
    ple_hot_store = {"state": hot_state, "env_pin": hot_pin, "events": hs_events,
                     "table_store": st["ple_table"]}

    # env pins
    pins = {}
    for name, occ in sorted(st["pins"].items()):
        pins[name] = {"value": occ[-1]["value"], "occurrences": occ,
                      "config_key": ENV_TO_CONFIG_KEY.get(name)}
    normalized = {}
    for name, p in pins.items():
        if p["config_key"]:
            normalized[p["config_key"]] = {"value": p["value"], "via": name}
    for key, occ in sorted(st["cfg_pins"].items()):
        normalized[key] = {"value": occ[-1]["value"], "via": "log config pin"}
    env_pins = {"env": pins, "config_pins": st["cfg_pins"], "normalized": normalized,
                "ple_format_pin": normalized.get("memory.ple_format", {}).get("value")}

    # model path / sha: evidence only unless --model is hashed
    candidates = []
    for source, p, suffixes in (
        ("ple_sidecar", ple_path, (".ple-q4", ".ple-q8")),
        ("expert_hotset", (st["hotset_applied"] or st["hotset_not_applied"] or {}).get("path"),
         (".expert-hotset",)),
        ("trunk_overlay", (st["overlay"] or {}).get("path"), (".trunk-overlay",)),
    ):
        if not p:
            continue
        base = p
        for suf in suffixes:
            if base.endswith(suf):
                base = base[: -len(suf)]
                break
        else:
            continue
        candidates.append({"source": source, "path": base})
    ident = st["hotset_identity"] or {}
    log_evidence = {
        "hotset_identity_model_sha256": ident.get("model_sha256"),
        "hotset_bound_base_bytes": (st["hotset_applied"] or {}).get("bound_base_bytes"),
        "hotset_bound_index_sha256": (st["hotset_applied"] or {}).get("bound_index_sha256"),
        "overlay_bound_base_bytes": (st["overlay"] or {}).get("bound_base_bytes"),
        "overlay_bound_index_sha256": (st["overlay"] or {}).get("bound_index_sha256"),
        "path_candidates": candidates,
    }
    if model_path:
        mpath, psrc = model_path, "--model"
    elif candidates:
        mpath, psrc = candidates[0]["path"], "log:" + candidates[0]["source"]
    else:
        mpath, psrc = None, None
    model = {
        "path": mpath, "path_source": psrc,
        "path_matches_log": (any(c["path"] == model_path for c in candidates)
                             if model_path and candidates else None),
        "sha256": model_sha,
        "sha256_source": "computed:--model (CPU, whole file)" if model_sha else None,
        "sha256_error": model_sha_err, "size_bytes": model_size,
        "log_evidence": log_evidence,
    }
    model["layout"] = detect_layout(
        model_sha, ident.get("model_sha256"),
        (("hot-set bound base bytes", log_evidence["hotset_bound_base_bytes"]),
         ("overlay bound base bytes", log_evidence["overlay_bound_base_bytes"])),
        tuple(("%s path" % c["source"], c["path"]) for c in candidates))
    if model_sha and ident.get("model_sha256"):
        model["sha256_matches_log_identity"] = (model_sha == ident["model_sha256"])

    cfg = {
        "schema_version": SCHEMA_VERSION,
        "arch": arch, "log_arch": st["arch"], "log_lines": st["lines"],
        "ple_format": ple_format, "ple_path": ple_path,
        "ple": {"detail": ple_format_detail, "rows": ple,
                "auto_fallback_embedded_q8": st["ple_fallback_note"]},
        "expert_hotset": hotset,
        "expert_placement": placement,
        "overlay": overlay,
        "npu": npu,
        "route_plans": route_plans,
        "ple_hot_store": ple_hot_store,
        "memlock": {"events": st["memlock"], "host_memory": st["host_memory"]},
        "env_pins": env_pins,
        "model": model,
        "raw_lines": {"lines": raw.lines, "counts": raw.counts, "cap_per_category": RAW_CAP},
    }
    return cfg


def check(res, item, expected, observed, result, reason):
    res.append({"item": item, "expected": expected, "observed": observed,
                "result": result, "reason": reason})


def build_check(cfg, arch):
    checks = []
    notes = []
    halo = arch == "gfx1151"
    r9 = arch == "gfx1201"

    # arch sanity
    if cfg["log_arch"]:
        if arch in cfg["log_arch"]:
            check(checks, "arch", arch, cfg["log_arch"], "match", "log device arch agrees with --arch")
        else:
            check(checks, "arch", arch, cfg["log_arch"], "mismatch",
                  "arch: log shows device %s but --arch is %s" % (",".join(cfg["log_arch"]), arch))

    # PLE rows: the split base has no embedded rows, so a fresh pull always reads a sidecar
    ple = cfg["ple"]["rows"]
    exp_fmt, exp_order = ("q4", "rank") if halo else ("q8", "row")
    exp_ple = "%s from a sidecar (%s order); split base has no embedded PLE rows" % (exp_fmt, exp_order)
    if ple is None:
        check(checks, "ple_rows", exp_ple, "no `qwen4 ple rows:` line", "unknown",
              "ple: no `qwen4 ple rows:` line in log; q4 vs q8 unknown")
    elif ple["origin"] == "embedded" and ple["format"] in ("q4", "q8"):
        fb = " (q4 sidecar not found fallback logged)" if cfg["ple"]["auto_fallback_embedded_q8"] else ""
        check(checks, "ple_rows", exp_ple, "%s embedded%s" % (ple["format"], fb), "mismatch",
              "ple: %s rows embedded in the model file mean a grandfathered monolithic install, which is "
              "not a fresh-pull layout; a fresh pull is the split base (no embedded rows) with a %s "
              "sidecar on %s%s" % (ple["format"], exp_fmt, arch, fb))
    elif ple["format"] in ("q4", "q8"):
        where = "sidecar " + str(ple["path"])
        pin = cfg["env_pins"]["ple_format_pin"]
        how = "explicit memory.ple_format/HIPFIRE_PLE_FORMAT=%s" % pin if pin else "sidecar present"
        observed = "%s from %s" % (ple["format"], where)
        if ple["format"] != exp_fmt:
            check(checks, "ple_rows", exp_ple, observed, "mismatch",
                  "ple: %s rows from %s (%s) on %s; the fresh-pull default there is %s, so this is an "
                  "explicit override, not the default" % (ple["format"], where, how, arch, exp_fmt))
        elif ple.get("order") and ple["order"] != exp_order:
            check(checks, "ple_rows", exp_ple, "%s (%s order)" % (observed, ple["order"]), "mismatch",
                  "ple: %s rows from %s are in %s order, expected %s order" %
                  (ple["format"], where, ple["order"], exp_order))
        else:
            check(checks, "ple_rows", exp_ple, observed, "match",
                  "ple: %s rows from a sidecar (%s order) = fresh-pull product on %s" %
                  (exp_fmt, exp_order, arch))
            if halo:
                notes.append("gfx1151 default PLE rows are q4 (Lloyd-Max, rank order), which are not exact: "
                             "WT2 KLD 0.104865 vs 0.103252 with q8 rows (v0.4.2 measurement); "
                             "`hipfire pull qwen3.8:flash-next --ple q8` keeps exact rows")
    else:
        check(checks, "ple_rows", exp_ple, ple["format"], "unknown",
              "ple: rows format %r is not q4/q8; cannot compare with the product default" % ple["format"])

    layout = cfg["model"]["layout"]
    if layout["name"] == LAYOUT_MONOLITH:
        notes.append("log layout is the grandfathered monolithic file (sha256 8b15b6fe..., %d B; "
                     "equivalent to split base + shared q8 rows); a fresh pull is the split base "
                     "%s (sha256 eafaff94..., %d B); identity evidence: %s" %
                     (MONOLITH_BYTES, SPLIT_BASE_NAME, SPLIT_BASE_BYTES, layout["source"]))
    elif layout["name"] == LAYOUT_SPLIT:
        notes.append("log layout is the split base %s (sha256 eafaff94..., %d B); identity evidence: %s" %
                     (SPLIT_BASE_NAME, SPLIT_BASE_BYTES, layout["source"]))

    # expert hot set
    hs = cfg["expert_hotset"]
    wl = cfg["expert_placement"]["whole_layer"]
    if r9:
        exp = "applied (gfx1201 default; sibling .expert-hotset)"
        if hs["state"] == "applied":
            a = hs["applied"]
            check(checks, "expert_hotset", exp, "applied: %d VRAM / %d host experts" %
                  (a["experts_in_vram"], a["experts_host_mapped"]), "match",
                  "hot set applied (sha256 %s)" % a["sha256"][:12])
        elif hs["state"] == "present_not_applied":
            check(checks, "expert_hotset", exp, "present, not applied", "mismatch",
                  "expert hot set present but not applied (%s); whole-layer placement" %
                  hs["not_applied"]["reason"])
        elif hs["state"] == "disabled":
            check(checks, "expert_hotset", exp, "explicitly disabled (HIPFIRE_QWEN4_EXPERT_HOTSET=%s)" %
                  hs["env_pin"], "mismatch",
                  "expert hot set explicitly disabled (HIPFIRE_QWEN4_EXPERT_HOTSET=%s)%s" %
                  (hs["env_pin"], "; whole-layer placement" if wl else ""))
        elif hs["state"] == "absent":
            check(checks, "expert_hotset", exp, "no hot set line; whole-layer placement", "mismatch",
                  "no expert hot set applied and whole-layer placement in use (%s)" %
                  cfg["expert_placement"]["expert_placement"]["raw"][:80]
                  if cfg["expert_placement"]["expert_placement"] else
                  "no expert hot set applied and whole-layer placement in use")
        else:
            check(checks, "expert_hotset", exp, "no hot-set or placement line", "unknown",
                  "hot set: no expert hot set / expert placement line in log; applied state unknown")
    else:
        exp = "not applied (only gfx1201 applies the default hot set)"
        if hs["state"] == "applied":
            check(checks, "expert_hotset", exp, "applied", "mismatch",
                  "expert hot set applied on %s; product default applies it on gfx1201 only" % arch)
        else:
            check(checks, "expert_hotset", exp, hs["state"], "match",
                  "hot set not applied on %s (expected: gfx1201-only default)" % arch)

    # overlay
    ov = cfg["overlay"]
    if halo:
        exp = "applied (%d tensors; gfx1151 default)" % EXPECTED_OVERLAY_TENSORS
        if ov["state"] == "applied":
            if ov["tensors"] == EXPECTED_OVERLAY_TENSORS:
                check(checks, "overlay", exp, "applied %d tensors" % ov["tensors"], "match",
                      "overlay applied %d %s tensors" % (ov["tensors"], ov["tensor_kind"]))
            else:
                check(checks, "overlay", exp, "applied %d tensors" % ov["tensors"], "mismatch",
                      "overlay applied %d tensors, expected %d" % (ov["tensors"], EXPECTED_OVERLAY_TENSORS))
        elif ov["state"] in ("disabled", "present_not_applied", "refused"):
            check(checks, "overlay", exp, ov["state"], "mismatch",
                  "overlay not applied on gfx1151 (%s): %s" %
                  (ov["state"], ov.get("why") or ov.get("reason") or ov.get("disabled_by", "")))
        elif cfg["ple"]["rows"] is not None:
            check(checks, "overlay", exp, "no overlay line in a full startup log", "mismatch",
                  "overlay: no `trunk overlay: applied` line in startup log; gfx1151 default applies it "
                  "(overlay file absent or never loaded)")
        else:
            check(checks, "overlay", exp, "no overlay line", "unknown",
                  "overlay: no trunk-overlay line and log has no PLE startup lines; cannot tell")
    else:
        exp = "not applied (overlay default is gfx1151 only)"
        if ov["state"] == "applied":
            check(checks, "overlay", exp, "applied", "mismatch",
                  "overlay applied on %s; product default applies it on gfx1151 only" % arch)
        else:
            check(checks, "overlay", exp, ov["state"] if ov["state"] != "unknown" else "not logged",
                  "match", "overlay not applied on %s (expected)" % arch)

    # npu
    npu = cfg["npu"]
    plans = cfg["route_plans"]["npu_requests"]
    if halo:
        exp = "mode auto armed with eligible calibration and request plan"
        st = npu["state"]
        pin = (npu["env_pin"] or "auto").strip('"').lower()
        if pin != "auto":
            check(checks, "npu_trunk", exp, "pinned %s" % npu["env_pin"], "mismatch",
                  "npu-trunk pinned to %s (state %s); product default is mode auto" %
                  (npu["env_pin"], st))
        elif st == "armed":
            problems = []
            if npu["mode"] != "auto":
                problems.append("mode=%s, expected auto" % npu["mode"])
            if not npu["calibration"]:
                problems.append("no calibration line (cal_sha256) for the armed trunk")
            if problems:
                check(checks, "npu_trunk", exp, "armed mode=%s" % npu["mode"], "mismatch",
                      "npu: " + "; ".join(problems))
            elif plans["count"] == 0:
                check(checks, "npu_trunk", exp, "armed, calibrated, no request plan lines", "unknown",
                      "npu: armed with calibration but no `rows=N plan=[..]` request plan in log; "
                      "eligibility for a request unconfirmed")
            else:
                check(checks, "npu_trunk", exp,
                      "armed auto; %d plans (%d with pipe steps)" % (plans["count"], plans["with_pipe_count"]),
                      "match", "npu auto armed with calibration %s; %d request plans" %
                      (npu["calibration"]["cal_sha256"][:12], plans["count"]))
                if plans["with_pipe_count"] == 0:
                    notes.append("all %d npu request plans are GPU-only: not evidence the NPU executed" %
                                 plans["count"])
        elif st == "off":
            r = npu["off"]["reason"]
            check(checks, "npu_trunk", exp, "auto, trunk off (GPU-only fallback)", "match",
                  "npu auto fell back to GPU-only (product default allows it): %s%s" %
                  (r, " (memlock limit)" if npu["off_reason_memlock"] else ""))
            notes.append("npu-trunk auto did not arm: %s; prefill runs GPU-only" % r)
        elif st == "disabled":
            check(checks, "npu_trunk", exp, "explicitly disabled", "mismatch",
                  "npu-trunk explicitly disabled (pin %s)" % npu["env_pin"])
        elif st == "forced_no_calibration":
            check(checks, "npu_trunk", exp, "forced, no calibration", "mismatch",
                  "npu-trunk mode=%s with calibration=none (forced): %s" %
                  (npu["mode"], npu["forced_no_calibration"]["reason"]))
        elif st == "calibrated_not_initialized":
            check(checks, "npu_trunk", exp, "calibration line but no init line", "unknown",
                  "npu: calibration logged but no trunk init line; armed state unknown")
        else:
            check(checks, "npu_trunk", exp, "no npu-trunk lines", "unknown",
                  "npu: no npu-trunk lines in log; armed state unknown")
    else:
        exp = "not armed (NPU auto is gfx1151 only)"
        if npu["state"] == "armed":
            check(checks, "npu_trunk", exp, "armed", "mismatch",
                  "npu-trunk armed on %s; product default is gfx1151 only" % arch)
        else:
            check(checks, "npu_trunk", exp, npu["state"] if npu["state"] != "unknown" else "not logged",
                  "match", "npu-trunk not armed on %s (expected)" % arch)

    # PLE hot store (default 0)
    hot = cfg["ple_hot_store"]
    exp = "default 0 (inactive)"
    s = hot["state"]
    if s in ("active", "off", "failed", "requested_nonzero"):
        check(checks, "ple_hot_store", exp, s, "mismatch",
              "ple hot store not at default 0: %s" % (hot["env_pin"] or s))
    elif s == "disabled":
        check(checks, "ple_hot_store", exp, "pinned 0", "match", "ple hot store pinned to 0 (default)")
    elif s == "inactive_not_logged":
        check(checks, "ple_hot_store", exp, "no hot-store pack activity", "match",
              "ple hot store: no pack/off lines (loader is silent at the default 0 budget)")
    else:
        check(checks, "ple_hot_store", exp, "unknown", "unknown",
              "ple hot store: no PLE startup lines, state unknown")

    # model sha consistency (conflict only)
    model = cfg["model"]
    if model.get("sha256_matches_log_identity") is False:
        check(checks, "model_sha256", "computed == log hotset identity model_sha256",
              model["sha256"], "mismatch",
              "model sha256 computed from --model (%s) differs from the log's hotset identity "
              "model_sha256 (%s)" % (model["sha256"][:12],
                                     model["log_evidence"]["hotset_identity_model_sha256"][:12]))
    elif model.get("sha256_matches_log_identity") is True:
        check(checks, "model_sha256", "computed == log hotset identity model_sha256", model["sha256"],
              "match", "computed model sha256 equals the log's hotset identity model_sha256")

    results = [c["result"] for c in checks]
    if "mismatch" in results:
        status = "mismatch"
    elif "unknown" in results:
        status = "unknown"
    else:
        status = "match"
    if status == "mismatch":
        reasons = [c["reason"] for c in checks if c["result"] == "mismatch"]
    elif status == "unknown":
        reasons = [c["reason"] for c in checks if c["result"] == "unknown"]
    else:
        reasons = [c["reason"] for c in checks]
    return {
        "status": status,
        "reasons": reasons,
        "arch": arch,
        "product": "fresh mq4 pull (split base %s + shared PLE sidecar)" % SPLIT_BASE_NAME,
        "log_layout": layout["name"],
        "log_layout_source": layout["source"],
        "checks": checks,
        "notes": notes,
        "intended_product": (
            "split base %s; gfx1151: q4 PLE sidecar (rank order, auto) + overlay + npu auto; "
            "gfx1201: q8 PLE sidecar (row order) + hot set" % SPLIT_BASE_NAME),
    }


def load_receipt(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if not text.strip():
        return {}
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("receipt %s is not a JSON object" % path)
    return data


def write_receipt(path, data):
    d = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(prefix=".receipt.", suffix=".tmp", dir=d)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True, metavar="PATH",
                    help="daemon/startup log to parse (stderr of the run)")
    ap.add_argument("--arch", required=True, choices=("gfx1151", "gfx1201"),
                    help="GPU the log was produced on (Halo gfx1151 / R9700 gfx1201)")
    ap.add_argument("--receipt", required=True, metavar="PATH",
                    help="receipt JSON to update in place (created if missing); other keys preserved")
    ap.add_argument("--model", metavar="PATH",
                    help="model file; its path is recorded and its whole-file sha256 is computed "
                         "on CPU (slow for 100+ GB). Without it no sha256 is computed.")
    args = ap.parse_args(argv)

    if not os.path.isfile(args.log):
        print("perf_config_receipt: --log %s is not a file" % args.log, file=sys.stderr)
        return 2
    try:
        receipt = load_receipt(args.receipt)
    except (ValueError, OSError) as e:
        print("perf_config_receipt: cannot read receipt: %s" % e, file=sys.stderr)
        return 2

    rc = 0
    model_sha = model_err = model_size = None
    if args.model:
        try:
            model_sha, model_size = sha256_file(args.model)
        except OSError as e:
            model_err = "%s: %s" % (type(e).__name__, e)
            print("perf_config_receipt: sha256 of --model failed: %s" % model_err, file=sys.stderr)
            rc = 1

    st, raw = parse_log(args.log)
    cfg = build_startup_config(st, raw, args.arch, args.model, model_sha, model_err, model_size)
    cfg["log_path"] = args.log
    verdict = build_check(cfg, args.arch)

    receipt["startup_config"] = cfg
    receipt["product_default_check"] = verdict
    write_receipt(args.receipt, receipt)

    print("product_default_check: %s (%s)" % (verdict["status"], args.arch))
    for r in verdict["reasons"]:
        print("  - " + r)
    return rc


if __name__ == "__main__":
    sys.exit(main())
