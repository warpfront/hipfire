#!/usr/bin/env python3
"""Read-only comparison of hipfire.tiered_prefill.v1 streaming lab receipts.

Not a serving, quality or performance gate. Inputs are never rewritten. Output
must be a NEW file. Natural stops below 256 tokens are identified, but cannot
pass long-decode validation. Optional 50ms VRAM guards are sampled evidence,
not a hard physical-memory bound.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

from compare_tiered_kv_oracles import (
    digest, integer, load_evidence, obj, require, state, string,
    validate_guard as validate_sampled_guard,
)


SCHEMA = "hipfire.tiered_prefill.v1"
ARMS = ("baseline", "identity", "bounded")
BLOCK = 128
STATE_HASHES = ("dn_sha256", "kv_sha256", "hidden_sha256")
SCOPE = "Lab streaming state comparison only; NOT a serving, quality or performance gate."


def selected(completed, blocks):
    """Independent absolute block policy; no planner offsets/receipt map used."""
    count = completed // BLOCK
    if count <= blocks:
        return list(range(0, completed, BLOCK))
    return [0] + list(range((count - blocks + 1) * BLOCK, completed, BLOCK))


def int_list(value, expected, label):
    require(type(value) is list, f"{label}: expected list")
    for i, item in enumerate(value):
        integer(item, f"{label}[{i}]")
    require(value == expected, f"{label}: incorrect rows or chronological map")


def equal_int(data, key, expected, label):
    require(integer(data.get(key), f"{label}.{key}") == expected,
            f"{label}.{key}: expected {expected}")


def checked_state(value, visible, label):
    state(value, label)
    equal_int(value, "visible_rows", visible, label)


def validate_ledger(data, label):
    ledger = obj(data.get("ledger"), f"{label}.ledger")
    common_limits = None
    for phase in ("after_bounded_load", "after_window_install", "after_prefill",
                  "after_decode", "after_cleanup"):
        snapshot = obj(ledger.get(phase), f"{label}.ledger.{phase}")
        limits = obj(snapshot.get("limits"), f"{label}.{phase}.limits")
        cap = integer(limits.get("device_bytes"), f"{label}.{phase}.device limit", 1)
        require(cap <= (24 << 30) - (512 << 20), f"{label}: managed limit exceeds 23.5GiB")
        require(type(limits.get("uma")) is bool, f"{label}: limits.uma must be boolean")
        for key in ("gtt_bytes", "shared_host_bytes"):
            require(key in limits, f"{label}: missing limit {key}")
            if limits[key] is not None:
                integer(limits[key], f"{label}.{phase}.{key}", 1)
        require(common_limits is None or common_limits == limits,
                f"{label}: managed limits changed between phases")
        common_limits = limits
        usage = {}
        for kind in ("committed", "pending"):
            usage[kind] = obj(snapshot.get(kind), f"{label}.{phase}.{kind}")
            for key in ("device_bytes", "host_bytes", "gtt_bytes", "shared_host_bytes"):
                integer(usage[kind].get(key), f"{label}.{phase}.{kind}.{key}")
        for key in ("device_bytes", "gtt_bytes", "shared_host_bytes"):
            require(limits[key] is None or sum(usage[k][key] for k in usage) <= limits[key],
                    f"{label}: managed {key} exceeds limit at {phase}")
    return common_limits


def validate_oracle(data, arm):
    obj(data, arm)
    require(data.get("schema") == SCHEMA, f"{arm}: require schema {SCHEMA}; old receipts cannot be upgraded")
    require(data.get("mode") == arm and data.get("pass") is True, f"{arm}: wrong mode or pass")
    for key in ("model", "prompt_file", "scope", "transient_kv_slabs"):
        string(data.get(key), f"{arm}.{key}")
    require(type(data.get("text")) is str, f"{arm}.text: expected string")
    for key in ("model_sha256", "oracle_binary_sha256", "user_prompt_sha256",
                "rendered_prompt_sha256", "prompt_tokens_sha256", "prefill_logits_sha256"):
        digest(data.get(key), f"{arm}.{key}")
    for key in ("performance_claim", "quality_claim", "decode_restaging"):
        require(data.get(key) is False, f"{arm}.{key}: require false")
    bounded = arm == "bounded"
    require(data.get("initial_prefill_bounded") is bounded,
            f"{arm}: incorrect initial_prefill_bounded")
    require(data.get("initial_full_history_allocation") is (not bounded),
            f"{arm}: incorrect initial_full_history_allocation")
    prompt = integer(data.get("prompt_tokens"), f"{arm}.prompt_tokens", 1)
    chunk = integer(data.get("chunk_tokens"), f"{arm}.chunk_tokens", 2)
    require(chunk <= 512, f"{arm}: chunk exceeds 512")
    requested = integer(data.get("requested_decode_tokens"), f"{arm}.requested_decode_tokens", 256)
    require(requested <= 32768, f"{arm}: requested decode too large")
    records = integer(data.get("prefix_records"), f"{arm}.prefix_records")
    require(records <= 16384, f"{arm}: too many prefix records")
    blocks = integer(data.get("window_blocks"), f"{arm}.window_blocks", 2)
    equal_int(data, "block_tokens", BLOCK, arm)
    reserve = max(chunk, requested) + BLOCK - 1
    cap = blocks * BLOCK + reserve
    equal_int(data, "generation_reserve_rows", reserve, arm)
    equal_int(data, "physical_cap", cap, arm)
    equal_int(data, "initial_physical_cap", cap, arm)
    equal_int(data, "max_seq", max(prompt + requested, cap + 1), arm)
    require(data["max_seq"] <= 2147483647, f"{arm}: logical extent exceeds i32")
    if bounded:
        require(blocks <= 2048, f"{arm}: window exceeds lab bound")
        require(cap < prompt, f"{arm}: physical cap must be smaller than the full prompt")
        require(prompt // BLOCK > blocks, f"{arm}: no real complete-block history gap")
    else:
        require(blocks == max(2, (prompt + BLOCK - 1) // BLOCK),
                f"{arm}: full-history reference window incorrect")
    row = integer(data.get("row_bytes_per_side"), f"{arm}.row_bytes_per_side", 1)
    require(row % 34 == 0 and row % 4 == 0, f"{arm}: invalid whole-u32 Q8 row")
    sides = 2 * integer(data.get("full_attention_layers"), f"{arm}.full_attention_layers", 1)
    equal_int(data, "fixed_kv_bytes", cap * row * sides, arm)
    require(cap * row * sides <= 4 << 30, f"{arm}: fixed KV exceeds lab byte limit")
    route = obj(data.get("route"), f"{arm}.route")
    for key in ("arch", "bdf"):
        string(route.get(key), f"{arm}.route.{key}")
    require(route.get("kv_backend") == "legacy" and route.get("kv_mode") == "q8/q8"
            and route.get("dn_quant") == "Q8" and route.get("dn_ef") is True
            and route.get("graph_forward") is False and route.get("replay_enabled") is False
            and route.get("pbs_lean") is False and route.get("pbs_gdn_tape") is False,
            f"{arm}: unsupported route")
    equal_int(route, "pbs_max_batch", chunk, f"{arm}.route")
    equal_int(route, "prefill_max_batch_override", 512, f"{arm}.route")
    require(route.get("prefill_path") == ("ordinary_with_pbs" if arm == "baseline" else "positioned_lab")
            and route.get("singleton_path") == ("ordinary_ar" if arm == "baseline" else "positioned_ar_lab"),
            f"{arm}: wrong prefill/singleton route")
    limits = validate_ledger(data, arm)
    trace = data.get("prefill_trace")
    require(type(trace) is list and len(trace) == (prompt + chunk - 1) // chunk,
            f"{arm}: incomplete prefill_trace")
    start, visible, saw_gap = 0, 0, False
    for index, item in enumerate(trace):
        label = f"{arm}.prefill_trace[{index}]"
        obj(item, label)
        n = min(chunk, prompt - start)
        end = start + n
        old_complete = start // BLOCK * BLOCK
        complete = end // BLOCK * BLOCK
        old = selected(old_complete, blocks)
        new = selected(complete, blocks)
        packed = len(old) * BLOCK + start % BLOCK
        require(packed == visible, f"{label}: noncontiguous packed frontier")
        saw_gap |= start > packed
        new_count = (complete - old_complete) // BLOCK
        source_rows = [len(old) * BLOCK + i * BLOCK for i in range(new_count)]
        next_visible = len(new) * BLOCK + end % BLOCK
        for key, value in (("chunk", index), ("absolute_start", start), ("absolute_end", end),
                           ("tokens", n), ("packed_start", packed), ("selected_rows", len(new) * BLOCK),
                           ("visible_rows", next_visible), ("archive_committed_rows", complete),
                           ("archive_host_bytes", complete * row * sides),
                           ("carry_source_row", len(old) * BLOCK + new_count * BLOCK),
                           ("carry_rows", end % BLOCK), ("carry_destination_row", len(new) * BLOCK),
                           ("carry_verified_bytes", (end % BLOCK) * row * sides),
                           ("staged_history_verified_bytes", len(new) * BLOCK * row * sides)):
            equal_int(item, key, value, label)
        require(packed + n <= cap and next_visible <= cap, f"{label}: physical capacity exceeded")
        int_list(item.get("selected_block_starts"), new, f"{label}.selected_block_starts")
        int_list(item.get("archive_source_rows"), source_rows, f"{label}.archive_source_rows")
        appended = item.get("archived_blocks")
        require(type(appended) is list and len(appended) == new_count, f"{label}: incomplete archived_blocks")
        for j, block in enumerate(appended):
            obj(block, f"{label}.archived_blocks[{j}]")
            equal_int(block, "absolute_start", old_complete + j * BLOCK, label)
            equal_int(block, "source_row", source_rows[j], label)
        changed = sum(i >= len(old) or old[i] != value for i, value in enumerate(new))
        equal_int(item, "stage_h2d_bytes", changed * BLOCK * row * sides, label)
        digest(item.get("logits_sha256"), f"{label}.logits_sha256")
        digest(item.get("carry_sha256"), f"{label}.carry_sha256")
        if end % BLOCK == 0:
            require(item["carry_sha256"] == hashlib.sha256(b"").hexdigest(),
                    f"{label}: empty carry hash is not SHA-256 of empty bytes")
        token = integer(item.get("argmax"), f"{label}.argmax")
        require(token <= 0xffffffff, f"{label}: token outside u32")
        require(item.get("frontier_published") is True, f"{label}: unpublished frontier")
        checked_state(item.get("forward_state"), packed + n, f"{label}.forward_state")
        checked_state(item.get("state"), next_visible, f"{label}.state")
        for key in ("dn_sha256", "hidden_sha256"):
            require(item["forward_state"][key] == item["state"][key],
                    f"{label}: storage changed {key}")
        if not bounded:
            require(item["forward_state"] == item["state"], f"{label}: full-history GPU source changed during staging")
        start, visible = end, next_visible
    require(not bounded or saw_gap, f"{arm}: no actual absolute/packed gap during prefill")
    equal_int(data, "archive_committed_rows", prompt // BLOCK * BLOCK, arm)
    equal_int(data, "archive_host_bytes", prompt // BLOCK * BLOCK * row * sides, arm)
    require(data["archive_host_bytes"] <= 12 << 30, f"{arm}: host archive exceeds lab limit")
    int_list(data.get("selected_block_starts"), selected(prompt // BLOCK * BLOCK, blocks),
             f"{arm}.selected_block_starts")
    checked_state(data.get("prefill_state"), visible, f"{arm}.prefill_state")
    require(data["prefill_state"] == trace[-1]["state"]
            and data["prefill_logits_sha256"] == trace[-1]["logits_sha256"],
            f"{arm}: final prefill state/logits mismatch")
    generated = integer(data.get("generated_tokens"), f"{arm}.generated_tokens", 1)
    require(generated <= requested, f"{arm}: generated exceeds request")
    reason = data.get("finish_reason")
    require(reason in ("length", "stop"), f"{arm}: invalid finish reason")
    require(reason != "length" or generated == requested, f"{arm}: incomplete length decode")
    decode = data.get("trace")
    require(type(decode) is list and len(decode) == generated, f"{arm}: incomplete decode trace")
    for index, item in enumerate(decode):
        obj(item, f"{arm}.trace[{index}]")
        equal_int(item, "step", index, f"{arm}.trace[{index}]")
        token = integer(item.get("token"), f"{arm}.trace[{index}].token")
        require(token <= 0xffffffff, f"{arm}: token outside u32")
        digest(item.get("logits_sha256"), f"{arm}.trace[{index}].logits_sha256")
    require(decode[0]["logits_sha256"] == data["prefill_logits_sha256"]
            and decode[0]["token"] == trace[-1]["argmax"], f"{arm}: first decode differs from prefill")
    if generated > 1:
        digest(data.get("first_new_fa_k_sha256"), f"{arm}.first_new_fa_k_sha256")
    else:
        require("first_new_fa_k_sha256" in data and data["first_new_fa_k_sha256"] is None,
                f"{arm}: no forwarded decode row can have a first K")
    checked_state(data.get("final_state"), visible + generated - 1, f"{arm}.final_state")
    if generated == 1:
        require(data["final_state"] == data["prefill_state"],
                f"{arm}: state changed without a forwarded decode row")
    require(visible + generated - 1 <= cap, f"{arm}: decode exceeds physical cap")
    return {"long_decode_pass": generated >= 256, "natural_short_stop": generated < 256 and reason == "stop",
            "generated_tokens": generated, "requested_decode_tokens": requested,
            "physical_cap": cap, "managed_limits": limits}


def compare_oracles(arms):
    require("baseline" in arms and "identity" in arms, "baseline and identity required")
    statuses = {arm: validate_oracle(data, arm) for arm, data in arms.items()}
    baseline = arms["baseline"]
    common = ("model_sha256", "oracle_binary_sha256", "user_prompt_sha256", "rendered_prompt_sha256",
              "prompt_tokens_sha256", "prompt_tokens", "prefix_records", "chunk_tokens",
              "requested_decode_tokens", "row_bytes_per_side", "full_attention_layers")
    for arm, data in arms.items():
        for key in common:
            require(data[key] == baseline[key], f"{arm}: baseline mismatch in {key}")
        for key in ("arch", "bdf"):
            require(data["route"][key] == baseline["route"][key], f"{arm}: baseline route mismatch in {key}")
        require(statuses[arm]["managed_limits"] == statuses["baseline"]["managed_limits"],
                f"{arm}: baseline managed limits mismatch")
    identity = arms["identity"]
    for key in ("prefill_trace", "prefill_logits_sha256", "prefill_state", "trace", "final_state",
                "first_new_fa_k_sha256", "text", "finish_reason", "generated_tokens", "physical_cap", "max_seq"):
        require(identity[key] == baseline[key], f"identity: strict baseline mismatch in {key}")
    return statuses


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in ARMS:
        parser.add_argument("--" + arm, type=Path, required=arm != "bounded")
    parser.add_argument("--guard", action="append", default=[], metavar="ARM=PATH")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = {"schema": "hipfire.tiered_prefill_comparison.v1", "oracle_schema": SCHEMA,
              "scope": SCOPE, "pass": False, "state_comparison_pass": False,
              "long_decode_pass": False, "inputs": {}, "guard_evidence": {},
              "sparse_later_tokens_required_equal": False}
    try:
        output = args.out.open("x", encoding="utf-8")
    except OSError as error:
        print(f"FAIL: cannot create NEW output: {error}", file=sys.stderr)
        return 2
    code = 1
    with output:
        try:
            paths = {arm: getattr(args, arm) for arm in ARMS if getattr(args, arm) is not None}
            arms = {arm: load_evidence(path, report, arm) for arm, path in paths.items()}
            statuses = compare_oracles(arms)
            report["validated_arms"] = statuses
            guards = {}
            for argument in args.guard:
                arm, separator, path = argument.partition("=")
                require(separator and arm in arms and path and arm not in guards,
                        "--guard requires a unique supplied ARM=PATH")
                guards[arm] = Path(path)
            for arm in arms:
                report["guard_evidence"][arm] = "not_supplied"
            for arm, path in guards.items():
                guard = load_evidence(path, report, arm + "_guard")
                # The guard always cleans its owned process group in finally,
                # even after an observed successful child exit. Cleanup signals
                # therefore do not imply a limit trip; status, both exit codes
                # and the validated sampled peak are the authoritative contract.
                validate_sampled_guard(guard, arm, paths[arm], arms[arm])
                report["guard_evidence"][arm] = "sampled_guard_passed_not_hard_cap_proof"
            report["state_comparison_pass"] = True
            report["long_decode_pass"] = all(s["long_decode_pass"] for s in statuses.values())
            report["pass"] = report["long_decode_pass"]
            if not report["pass"]:
                report["error"] = "natural short stop: state comparison valid, long decode insufficient"
            code = 0 if report["pass"] else 1
        except (OSError, ValueError, TypeError, KeyError, IndexError, OverflowError) as error:
            report["error"] = str(error)
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")
    print(f"{'PASS' if report['pass'] else 'FAIL'}: {SCOPE}")
    if "error" in report:
        print(report["error"], file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())
