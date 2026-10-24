#!/usr/bin/env python3
"""Read-only comparison of hipfire.tiered_kv_model.v2 lab oracle evidence.

This is NOT a quality, serving, or performance admission gate. Bounded sparse
attention is intentionally not required to reproduce later baseline tokens.
Baseline, identity, and bounded require >=256 generated tokens; wrong-rope may
stop earlier, but needs >=2 tokens and its first forwarded FA K negative control.
Bounded arms must also report byte-verified staged history, including every
selected K/V side and partial prompt tail. V1 evidence lacks that proof and is
rejected, never rewritten or upgraded by this tool.

Usage: compare_tiered_kv_oracles.py --baseline baseline.json \
  --identity identity.json [--bounded bounded.json --wrong-rope wrong.json] \
  [--guard baseline=baseline.vram.json ...] --out NEW_COMPARISON.json

Optional guards are independent sampled-memory evidence, never model or hard-cap
proof. Their command must explicitly select --mode and --out for the named arm.
Use absolute oracle output paths when running a guard in another working dir.
Inputs are never written; existing output files/symlinks are always refused.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys


SCHEMA = "hipfire.tiered_kv_model.v2"
ARMS = ("baseline", "identity", "bounded", "wrong-rope")
STATE_HASHES = ("dn_sha256", "kv_sha256", "hidden_sha256")
SCOPE = "Lab positional/state comparison only; NOT a quality, serving, or performance gate."


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, label, minimum=0):
    require(type(value) is int and value >= minimum,
            f"{label}: expected integer >= {minimum}")
    return value


def digest(value, label):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value),
            f"{label}: expected lowercase SHA-256")


def obj(value, label):
    require(type(value) is dict, f"{label}: expected object")
    return value


def string(value, label):
    require(isinstance(value, str) and bool(value), f"{label}: expected nonempty string")


def state(value, label):
    obj(value, label)
    for key in STATE_HASHES:
        digest(value.get(key), f"{label}.{key}")
    integer(value.get("visible_rows"), f"{label}.visible_rows", 1)


def validate_oracle(data, arm):
    obj(data, arm)
    require(data.get("schema") == SCHEMA,
            f"{arm}: wrong or missing schema; require {SCHEMA} (v1 evidence cannot be upgraded)")
    require(data.get("mode") == arm, f"{arm}: wrong or missing mode")
    require(data.get("pass") is True, f"{arm}: oracle did not report pass=true")
    for key in ("model", "prompt_file", "scope"):
        string(data.get(key), f"{arm}.{key}")
    require(isinstance(data.get("text"), str), f"{arm}.text: expected string")
    for key in ("model_sha256", "oracle_binary_sha256", "user_prompt_sha256", "rendered_prompt_sha256",
                "prompt_tokens_sha256", "prefill_logits_sha256", "first_new_fa_k_sha256"):
        digest(data.get(key), f"{arm}.{key}")
    prompt = integer(data.get("prompt_tokens"), f"{arm}.prompt_tokens", 1)
    requested = integer(data.get("requested_decode_tokens"), f"{arm}.requested_decode_tokens", 256)
    require(requested <= 32768, f"{arm}: requested decode exceeds oracle schema bound")
    require(integer(data.get("max_seq"), f"{arm}.max_seq") == prompt + requested,
            f"{arm}: inconsistent logical max_seq")
    generated = integer(data.get("generated_tokens"), f"{arm}.generated_tokens", 2)
    require(arm == "wrong-rope" or generated >= 256,
            f"{arm}: long validation insufficient: generated_tokens={generated} < 256")
    require(generated <= requested, f"{arm}: generated exceeds requested decode")
    trace = data.get("trace")
    require(type(trace) is list and len(trace) == generated,
            f"{arm}: trace length must equal generated_tokens (long validation incomplete)")
    for index, item in enumerate(trace):
        obj(item, f"{arm}.trace[{index}]")
        require(integer(item.get("step"), f"{arm}.trace[{index}].step") == index,
                f"{arm}: trace steps must start at zero and be contiguous")
        token = integer(item.get("token"), f"{arm}.trace[{index}].token")
        require(token <= 0xffffffff, f"{arm}: token outside u32 range")
        digest(item.get("logits_sha256"), f"{arm}.trace[{index}].logits_sha256")
    reason = data.get("finish_reason")
    require(reason in ("length", "stop"), f"{arm}: invalid finish_reason")
    require(reason != "length" or generated == requested,
            f"{arm}: length finish with incomplete decode")
    state(data.get("prefill_state"), f"{arm}.prefill_state")
    state(data.get("final_state"), f"{arm}.final_state")
    require(data["prefill_state"]["visible_rows"] == prompt,
            f"{arm}: prefill visible rows differ from prompt")
    require(data["prefill_logits_sha256"] == trace[0]["logits_sha256"],
            f"{arm}: first trace logits differ from prefill")
    route = obj(data.get("route"), f"{arm}.route")
    for key in ("arch", "bdf", "dn_quant"):
        string(route.get(key), f"{arm}.route.{key}")
    require(route.get("kv_backend") == "legacy" and route.get("kv_backend_legacy") is True
            and route.get("explicit_backend") == "HIPFIRE_KV_BACKEND=legacy"
            and route.get("kv_mode") == "q8/q8" and route.get("dn_ef") is True,
            f"{arm}: wrong KV/recurrent route")
    limit = integer(route.get("managed_limit_bytes"), f"{arm}.route.managed_limit_bytes", 1)
    require(limit <= 24 * 1024 ** 3 - 512 * 1024 ** 2, f"{arm}: managed limit too large")
    initial = integer(route.get("initial_physical_cap"), f"{arm}.route.initial_physical_cap", 1)
    require(initial >= data["max_seq"], f"{arm}: initial physical capacity too small")
    for key in ("managed_device_bytes_after_decode", "managed_host_bytes_after_decode"):
        integer(data.get(key), f"{arm}.{key}")
    if arm in ("baseline", "identity"):
        require("tiered" in data and data["tiered"] is None, f"{arm}: unexpected tiered window")
        visible = prompt + generated - 1
    else:
        tiered = obj(data.get("tiered"), f"{arm}.tiered")
        starts = tiered.get("selected_block_starts")
        require(type(starts) is list and len(starts) >= 2, f"{arm}: missing selected blocks")
        for index, start in enumerate(starts):
            integer(start, f"{arm}.selected_block_starts[{index}]")
        complete = prompt // 128
        require(len(starts) < complete, f"{arm}: no real history gap")
        expected = [0] + [i * 128 for i in range(complete - len(starts) + 1, complete)]
        require(starts == expected and any(b - a > 128 for a, b in zip(starts, starts[1:])),
                f"{arm}: selected blocks do not contain sink/recent real gap")
        selected = integer(tiered.get("selected_rows"), f"{arm}.selected_rows", 1)
        tail = integer(tiered.get("partial_prompt_rows"), f"{arm}.partial_prompt_rows")
        require(selected == len(starts) * 128 and tail == prompt % 128,
                f"{arm}: selected/partial prompt rows inconsistent")
        require(integer(tiered.get("committed_archived_rows"), f"{arm}.committed_archived_rows")
                == complete * 128, f"{arm}: incomplete immutable archive")
        for key in ("host_archive_bytes", "fixed_kv_bytes"):
            integer(tiered.get(key), f"{arm}.{key}", 1)
        cap = integer(tiered.get("physical_cap"), f"{arm}.physical_cap", 1)
        require(cap == selected + tail + requested and cap < initial,
                f"{arm}: physical capacity must shrink and match selected/decode extent")
        require(tiered.get("staged_history_verified") is True,
                f"{arm}: staged_history_verified must be true; first-new-FA-K alone is insufficient")
        row = integer(tiered.get("row_bytes_per_side"), f"{arm}.row_bytes_per_side", 1)
        require(row % 34 == 0, f"{arm}: row_bytes_per_side is not a Q8 row extent")
        layers = integer(tiered.get("full_attention_layers"), f"{arm}.full_attention_layers", 1)
        sides = integer(tiered.get("staged_history_verified_sides"),
                        f"{arm}.staged_history_verified_sides", 1)
        require(sides == layers * 2,
                f"{arm}: staged_history_verified_sides must cover K and V of every FA layer")
        verified = integer(tiered.get("staged_history_verified_bytes"),
                           f"{arm}.staged_history_verified_bytes", 1)
        require(verified == (selected + tail) * row * sides,
                f"{arm}: staged_history_verified_bytes must cover selected rows and partial tail on every side")
        require(tiered["fixed_kv_bytes"] == cap * row * sides,
                f"{arm}: fixed_kv_bytes inconsistent with physical capacity/row/FA sides")
        require(tiered.get("initial_prefill_bounded") is False
                and tiered.get("decode_restaging") is False,
                f"{arm}: unexpected prefill/restaging protocol")
        visible = selected + tail + generated - 1
        require(data["final_state"]["visible_rows"] <= cap,
                f"{arm}: final visible rows exceed physical cap")
    require(data["final_state"]["visible_rows"] == visible,
            f"{arm}: final visible rows inconsistent with actual forwards")


def compare_oracles(arms):
    for arm, data in arms.items():
        validate_oracle(data, arm)
    baseline = arms["baseline"]
    common = ("model_sha256", "oracle_binary_sha256", "user_prompt_sha256", "rendered_prompt_sha256",
              "prompt_tokens_sha256", "prompt_tokens", "requested_decode_tokens", "max_seq", "route")
    checks = []
    for arm, data in arms.items():
        if arm == "baseline":
            continue
        for key in common + ("prefill_logits_sha256", "prefill_state"):
            require(data[key] == baseline[key], f"{arm}: baseline mismatch in {key}")
        require(data["trace"][0] == baseline["trace"][0], f"{arm}: first token/logits mismatch")
        if arm == "identity":
            for key in ("trace", "final_state", "first_new_fa_k_sha256", "text", "finish_reason"):
                require(data[key] == baseline[key], f"identity: strict baseline mismatch in {key}")
            checks.append("identity: full trace, prefill/final DN/KV/hidden and first FA K bit-exact")
        else:
            equal = data["first_new_fa_k_sha256"] == baseline["first_new_fa_k_sha256"]
            require(equal if arm == "bounded" else not equal,
                    f"{arm}: first forwarded FA K {'must match baseline' if arm == 'bounded' else 'negative control must differ from baseline'}")
            if arm == "wrong-rope" and "bounded" in arms:
                for key in ("row_bytes_per_side", "full_attention_layers"):
                    require(data["tiered"][key] == arms["bounded"]["tiered"][key],
                            f"wrong-rope: bounded mismatch in {key}")
            checks.append(f"{arm}: byte-verified staged history, common prefill/first token, real gap, smaller capacity, "
                          + ("absolute-position FA K matches" if equal else "wrong-position FA K differs"))
    return checks


def finite_number(value, label):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            f"{label}: expected finite nonnegative number")
    return value


def validate_guard(data, arm, oracle_path, oracle):
    obj(data, f"{arm} guard")
    require(type(data.get("schema_version")) is int and data["schema_version"] == 1
            and data.get("guard_kind") == "sampled_not_hard_limit", f"{arm}: not a VRAM guard receipt")
    require(data.get("status") == "completed", f"{arm}: VRAM guard did not complete successfully")
    for key in ("exit_code", "child_exit_code"):
        require(type(data.get(key)) is int and data[key] == 0, f"{arm}: guard {key} is not zero")
    require(data.get("pci_bdf") == oracle["route"]["bdf"], f"{arm}: guard PCI device mismatch")
    command = data.get("command")
    require(type(command) is list and command and all(isinstance(x, str) for x in command),
            f"{arm}: malformed guard command")
    for flag, expected in (("--mode", arm), ("--out", oracle_path)):
        require(command.count(flag) == 1 and command.index(flag) + 1 < len(command),
                f"{arm}: guard command must explicitly bind {flag}")
        actual = command[command.index(flag) + 1]
        require(Path(actual).resolve() == Path(expected).resolve() if flag == "--out" else actual == expected,
                f"{arm}: guard command {flag} does not match oracle")
    limit = integer(data.get("limit_bytes"), f"{arm} guard.limit_bytes", 1)
    margin = integer(data.get("safety_margin_bytes"), f"{arm} guard.safety_margin_bytes")
    threshold = integer(data.get("guard_limit_bytes"), f"{arm} guard.guard_limit_bytes", 1)
    require(margin < limit and threshold == limit - margin, f"{arm}: inconsistent guard threshold")
    require(finite_number(data.get("sample_interval_seconds"), f"{arm} guard interval") == .05,
            f"{arm}: guard interval is not 50ms")
    samples = data.get("samples")
    require(type(samples) is list and len(samples) >= 2, f"{arm}: missing guard samples")
    values, times = [], []
    for index, sample in enumerate(samples):
        obj(sample, f"{arm} guard sample {index}")
        value = integer(sample.get("vram_used_bytes"), f"{arm} guard sample {index} bytes")
        raw = sample.get("raw")
        require(isinstance(raw, str) and re.fullmatch(r"[0-9]+\s*", raw, flags=re.ASCII)
                and int(raw) == value, f"{arm}: guard raw sample mismatch")
        values.append(value)
        times.append(finite_number(sample.get("elapsed_seconds"), f"{arm} guard sample time"))
    require(times == sorted(times), f"{arm}: guard sample times are not monotonic")
    for key in ("baseline_bytes", "peak_bytes", "peak_delta_bytes"):
        integer(data.get(key), f"{arm} guard.{key}")
    require(data["baseline_bytes"] == values[0] and data["peak_bytes"] == max(values)
            and data["peak_delta_bytes"] == max(values) - values[0]
            and max(values) <= threshold, f"{arm}: inconsistent/exceeded guard peak")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_evidence(path, report, label):
    raw = path.read_bytes()
    report["inputs"][label] = {"path": str(path.resolve()),
                               "sha256": hashlib.sha256(raw).hexdigest()}
    def invalid_constant(value):
        raise ValueError(f"nonfinite JSON constant: {value}")
    return json.loads(raw, object_pairs_hook=unique_object, parse_constant=invalid_constant)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    for arm in ARMS:
        parser.add_argument("--" + arm, required=arm in ("baseline", "identity"), type=Path)
    parser.add_argument("--guard", action="append", default=[], metavar="ARM=PATH")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    report = {"schema": "hipfire.tiered_kv_comparison.v1", "pass": False,
              "oracle_schema": SCHEMA,
              "scope": SCOPE, "inputs": {}, "checks": [],
              "guard_evidence": {}, "sparse_later_tokens_required_equal": False}
    try:
        output = args.out.open("x", encoding="utf-8")
    except OSError as exc:
        print(f"FAIL: cannot create new comparison JSON: {exc}", file=sys.stderr)
        return 2
    result = 1
    with output:
        try:
            paths = {arm: getattr(args, arm.replace("-", "_")) for arm in ARMS}
            arms = {arm: load_evidence(path, report, arm) for arm, path in paths.items() if path is not None}
            report["checks"] = compare_oracles(arms)
            report["validated_arms"] = {
                arm: {"requested_decode_tokens": data["requested_decode_tokens"],
                      "generated_tokens": data["generated_tokens"],
                      "finish_reason": data["finish_reason"],
                      "physical_cap": (data["tiered"]["physical_cap"] if data["tiered"]
                                       else data["route"]["initial_physical_cap"])}
                for arm, data in arms.items()}
            guards = {}
            for argument in args.guard:
                arm, separator, path = argument.partition("=")
                require(separator and arm in arms and path and arm not in guards,
                        "--guard requires a unique supplied ARM=PATH")
                guards[arm] = Path(path)
            for arm in arms:
                report["guard_evidence"][arm] = "not_supplied"
            for arm, path in guards.items():
                data = load_evidence(path, report, arm + "_guard")
                validate_guard(data, arm, paths[arm], arms[arm])
                report["guard_evidence"][arm] = "sampled_guard_passed_not_hard_cap_proof"
            report["pass"] = True
            result = 0
        except (OSError, ValueError, TypeError, OverflowError, RecursionError) as exc:
            report["error"] = str(exc)
        try:
            json.dump(report, output, indent=2, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        except OSError as exc:
            print(f"FAIL: comparison evidence write failed: {exc}", file=sys.stderr)
            return 2
    print(f"{'PASS' if report['pass'] else 'FAIL'}: {SCOPE} output={args.out}", file=sys.stderr)
    return result


if __name__ == "__main__":
    sys.exit(main())
