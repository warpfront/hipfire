#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

"""Resident daemon phase harness for the default-off Redline graft.

Loads one model once, measures synthetic prefill and single-token decode
separately, and asks the daemon to delimit/fingerprint exactly one HIP launch
sequence for each phase. The optional DSpark / DFlash2 verify oracles
additionally lower one isolated fixed-B verify body and compare ordinary HIP,
captured HIP, and retained PM4 state without installing that route into serving.
"""

import argparse
import json
import os
import select
import statistics
import subprocess
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent


class Daemon:
    def __init__(self, binary: Path, log_path: Path, timeout_s: float, kv_mode: str):
        self.timeout_s = timeout_s
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = log_path.open("w")
        env = dict(os.environ)
        env.update(
            HIPFIRE_REPLAY_BACKEND="shadow",
            HIPFIRE_REPLAY_MANUAL_CAPTURE="1",
            HIPFIRE_KV_MODE=kv_mode,
            HIPFIRE_CASK_OFF="1",
            HIPFIRE_AR_GRAPH="0",
            HIPFIRE_GRAPH="0",
        )
        self.proc = subprocess.Popen(
            [str(binary)],
            cwd=REPO,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            bufsize=1,
            start_new_session=True,
        )

    def request(self, message):
        if self.proc.poll() is not None:
            raise RuntimeError(f"daemon exited early with code {self.proc.returncode}")
        assert self.proc.stdin is not None
        assert self.proc.stdout is not None
        self.proc.stdin.write(json.dumps(message, separators=(",", ":")) + "\n")
        self.proc.stdin.flush()
        ready, _, _ = select.select([self.proc.stdout], [], [], self.timeout_s)
        if not ready:
            raise TimeoutError(f"daemon response timed out after {self.timeout_s}s: {message['type']}")
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"daemon closed while handling {message['type']}")
        response = json.loads(line)
        if response.get("type") == "error":
            raise RuntimeError(response.get("message", "daemon error"))
        return response

    def close(self):
        if self.proc.poll() is None:
            try:
                self.request({"type": "unload"})
            except Exception:
                pass
            try:
                self.proc.terminate()
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
                self.proc.wait(timeout=5)
        self.log.close()


def summarize(values):
    return {
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def capture_key(row):
    capture = row["redline_capture"]
    return (
        capture["launches"],
        capture["unique_kernels"],
        capture["sequence_hash"],
    )


# Bit-exact fields. A retained-PM4 window that disagrees with the shipping HIP
# window on any of these is a correctness failure, not a tolerance question.
DFLASH_EXACT_FIELDS = (
    "tokens_equal",
    "argmax_equal",
    "ring_head_equal",
    "ring_written_equal",
    "kv_active_hash_equal",
    "kv_guard_equal",
    "pbs_guard_equal",
    "gdn_frame_equal",
    "dn_ef_after_forward_equal",
    "dn_ef_after_rollback_equal",
)

# Arms judged against the `direct_capture_safe` reference. `hip_auto` remains
# diagnostic rather than verdict-bearing: this harness certifies the retained
# recorded-HIP/PM4 routes, not the separate HipGraph route.
DFLASH_COMPARED_ARMS = ("recorded_hip", "pm4")

# Qwen4 (Qwen3.8 Flash-Next) state surfaces. The retained replay and the
# exact-kernarg HIP oracle must agree with ordinary HIP on every one of these at
# every compared position: the decoded token is not the gate, the state is.
QWEN4_EXACT_FIELDS = (
    "tokens_equal",
    "argmax_equal",
    "position_equal",
    "lengths_equal",
    "state_bit_exact",
    "qsa_full_equal",
    "gdn_equal",
    "ple_equal",
)


def qwen4_shadow_failures(shadow):
    """Every reason this Qwen4 multi-position parity run is not evidence."""
    failures = []
    route = shadow.get("route") or {}
    counters = route.get("counters") or {}
    if route.get("phase") != "ready":
        failures.append(f"route phase is {route.get('phase')!r}, not 'ready'")
    if int(counters.get("replays") or 0) == 0:
        failures.append("route.counters.replays == 0")
    for name in ("replay_failures", "poison_count", "contract_failures", "prepare_failures"):
        if int(counters.get(name) or 0) != 0:
            failures.append(f"counters.{name} == {counters.get(name)}")

    capture = shadow.get("capture") or {}
    identity = shadow.get("prepared_identity") or {}
    if not identity.get("dispatch_equals_launches"):
        failures.append(
            f"prepared dispatches {identity.get('dispatch_count')} != "
            f"captured launches {capture.get('launches')}"
        )
    if identity.get("queue_count") != 1 or identity.get("phase_count") != 1:
        failures.append(f"prepared route is not single-queue/single-phase: {identity!r}")

    if not shadow.get("bit_exact"):
        failures.append("retained transport != ordinary HIP over the compared state")
    if not shadow.get("blob_bit_exact"):
        failures.append("recorded-HIP oracle != ordinary HIP over the compared state")

    parity = shadow.get("parity") or {}
    windows = parity.get("windows") or []
    if not windows:
        failures.append("parity table has no windows")
    for window in windows:
        position = window.get("position")
        row = window.get("pm4") or {}
        if not row:
            failures.append(f"position {position}: retained arm missing from parity")
            continue
        failures.extend(qwen4_row_failures(row, f"position {position}: pm4"))

    # The recorded-HIP oracle is the captured blob replayed at its own capture
    # position: its substitution comes from the controller's synthesized-binding
    # calibration, which Qwen4 replaces with program-declared bindings. It is
    # therefore compared once, at the capture geometry, while the retained arm
    # carries the multi-position claim.
    blob = parity.get("blob") or {}
    row = blob.get("recorded_hip") or {}
    if not row:
        failures.append("parity.blob.recorded_hip missing (capture-position oracle)")
    elif blob.get("position") != windows[0].get("position"):
        failures.append(
            f"parity.blob.position {blob.get('position')} != first window "
            f"{windows[0].get('position')} ({row})"
        )
    else:
        failures.extend(qwen4_row_failures(row, "blob.recorded_hip"))

    failure = shadow.get("failure_behavior") or {}
    if not failure:
        failures.append("failure_behavior missing")
    else:
        if not failure.get("route_poisoned"):
            failures.append("induced replay failure did not poison the route")
        if "exceeds prepared max_position" not in (failure.get("error") or ""):
            failures.append(
                f"induced failure did not surface the plan's refusal: {failure.get('error')!r}"
            )
        if not failure.get("fallback_reason"):
            failures.append("poisoned route has no fallback reason")
        if not failure.get("recovery_uses_hip"):
            failures.append("recovery forward is not on HIP")
        if not failure.get("recovered_bit_exact_against_clean_hip"):
            failures.append("recovery forward is not bit-exact against clean HIP")
    return failures


def qwen4_row_failures(row, label):
    failures = []
    for field in QWEN4_EXACT_FIELDS:
        if field not in row:
            failures.append(f"{label}.{field} missing")
        elif not row[field]:
            failures.append(f"{label}.{field} is false")
    for field, value in row.items():
        if isinstance(value, dict) and "max_abs" in value:
            if (value.get("max_abs") or 0) != 0 or (value.get("max_rel") or 0) != 0:
                failures.append(
                    f"{label}.{field} diverged max_abs={value.get('max_abs')} "
                    f"max_rel={value.get('max_rel')}"
                )
    return failures


def dflash_shadow_failures(shadow):
    """Every reason this shadow run is not evidence. Empty list == pass.

    Route state alone is never evidence: Ready with zero replays means the
    prepared IB was never submitted, and a parity table with no windows means
    nothing was compared.
    """
    failures = []
    route = shadow.get("route") or {}
    counters = route.get("counters") or {}
    phase = route.get("phase")
    replays = int(counters.get("replays") or 0)
    if phase != "ready":
        failures.append(f"route phase is {phase!r}, not 'ready' ({route.get('reason')})")
    if replays == 0:
        failures.append("replays == 0 (Ready state alone is never evidence)")
    for name in ("replay_failures", "poison_count", "contract_failures", "prepare_failures"):
        if int(counters.get(name) or 0) != 0:
            failures.append(f"counters.{name} == {counters.get(name)}")

    capture = shadow.get("capture") or {}
    if not capture.get("aql_equals_unique_kernels"):
        failures.append(
            f"AQL contracts {capture.get('aql_contracts')} != unique captured kernels"
        )
    identity = shadow.get("prepared_identity") or {}
    if not identity.get("dispatch_equals_launches"):
        failures.append(
            f"prepared dispatches {identity.get('dispatch_count')} != "
            f"captured launches {capture.get('launches')}"
        )
    if identity.get("queue_count") != 1 or identity.get("phase_count") != 1:
        failures.append(
            f"prepared route is not single-queue/single-phase: {identity!r}"
        )

    parity = shadow.get("parity") or {}
    if "q8_byte_parity_invalid" not in parity:
        failures.append("parity.q8_byte_parity_invalid missing")
    elif parity.get("q8_byte_parity_invalid"):
        failures.append("parity.q8_byte_parity_invalid is true (production Q8 EF evidence required)")
    windows = parity.get("windows") or []
    if not windows:
        failures.append("parity table has no windows")
    for window in windows:
        position = window.get("position")
        for arm in DFLASH_COMPARED_ARMS:
            row = window.get(arm) or {}
            if not row:
                failures.append(f"position {position}: arm {arm} missing from parity")
                continue
            for field in DFLASH_EXACT_FIELDS:
                if field not in row:
                    failures.append(f"position {position}: {arm}.{field} missing")
                elif not row[field]:
                    failures.append(f"position {position}: {arm}.{field} is false")
            for field, value in row.items():
                if isinstance(value, dict) and "max_abs" in value:
                    if (value.get("max_abs") or 0) != 0 or (value.get("max_rel") or 0) != 0:
                        failures.append(
                            f"position {position}: {arm}.{field} diverged "
                            f"max_abs={value.get('max_abs')} max_rel={value.get('max_rel')}"
                        )
    return failures


# The trunk tier this harness is allowed to certify.  Four bits on a dense
# trunk projection (which writes straight into the residual stream) produced a
# model that opened a reasoning block it could not close, and BF16 is the
# unpacked path this packing exists to replace.  A recipe that drifts to either
# one is a recipe error, so the gate below refuses to run rather than certify it.
QWEN4_ADMITTED_TRUNK_TYPES = {47, 35}  # MQ6G256V2, and MFP4G32E8SOA for the E8 A/B
QWEN4_REQUIRED_SOURCE_EXACT = ("embed_tokens.weight", "lm_head.weight", "hyper_connection")


def read_qwen4_index(model):
    """Return [(name, quant_type)] read from the artifact's own index.

    Ground truth, not the artifact's self-description: a hardcoded recipe can
    claim a trunk tier the payload does not carry.  Only the metadata/index
    region is read; the payload is never touched.
    """
    with model.open("rb") as handle:
        header = handle.read(32)
        if len(header) < 32 or header[:4] != b"HFQM":
            return None
        count = int.from_bytes(header[12:16], "little")
        metadata_offset = int.from_bytes(header[16:24], "little")
        data_offset = int.from_bytes(header[24:32], "little")
        if not 32 <= metadata_offset <= data_offset:
            return None
        handle.seek(metadata_offset)
        region = handle.read(min(data_offset - metadata_offset, 64 * 1024 * 1024))
    depth = 0
    in_string = False
    escaped = False
    cursor = None
    for index, byte in enumerate(region):
        char = chr(byte)
        if escaped:
            escaped = False
            continue
        if in_string:
            if char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                cursor = index + 1
                break
    if cursor is None:
        return None
    import struct as _struct

    position = cursor
    (index_count,) = _struct.unpack_from("<I", region, position)
    position += 4
    if index_count != count:
        return None
    entries = []
    for _ in range(index_count):
        (name_len,) = _struct.unpack_from("<H", region, position)
        position += 2
        name = region[position : position + name_len].decode("utf-8")
        position += name_len
        quant_type = region[position]
        position += 1
        n_dims = region[position]
        position += 1
        dims = _struct.unpack_from("<%dI" % n_dims, region, position)
        position += 4 * n_dims
        position += 4  # group size
        position += 8  # data length
        entries.append((name, quant_type, dims))
    return entries


def qwen4_recipe_failures(model):
    """Check the artifact's actual trunk tier and source-exact classes."""
    entries = read_qwen4_index(model)
    if entries is None:
        return ["artifact index is unreadable; its trunk tier cannot be certified"]
    trunk = []
    sensitive = []
    for name, quant_type, dims in entries:
        short = name.replace("model.language_model.", "")
        if short.startswith("mtp."):
            continue  # the drafter's Q8 projections are not trunk projections
        if any(
            marker in short
            for marker in ("embed_tokens.weight", "lm_head.weight", "hyper_connection")
        ):
            sensitive.append((short, quant_type))
            continue
        if not any(marker in short for marker in (".linear_attn.", ".self_attn.")):
            continue
        if len(dims) != 2 or dims[-1] % 128 != 0:
            continue
        trunk.append((short, quant_type))
    failures = []
    if not trunk:
        failures.append("no packed trunk projection found; the trunk decodes unpacked")
    wrong = sorted({quant_type for _, quant_type in trunk} - QWEN4_ADMITTED_TRUNK_TYPES)
    if wrong:
        named = ", ".join(
            f"{quant_type} ({name})" for name, quant_type in trunk if quant_type in wrong
        )[:400]
        failures.append(
            f"trunk projections carry quant types {wrong}, admitted is "
            f"{sorted(QWEN4_ADMITTED_TRUNK_TYPES)}; the first offenders: {named}"
        )
    # The input/output distributions ship at BF16 (16) or the eight-bit class
    # tier Q8F16 (3); every MoE recipe in this tree uses one of those two for
    # them, and anything narrower is the regression this gate exists for.
    quantized_sensitive = sorted(
        {name for name, quant_type in sensitive if quant_type not in (16, 3)}
    )
    if quantized_sensitive:
        failures.append(
            "these classes carry the model's input/output distribution and must stay "
            f"BF16 (16) or Q8F16 (3): {quantized_sensitive[:6]}"
        )
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--daemon",
        default=str(REPO / "target/release/daemon"),
    )
    parser.add_argument("--out", default=str(REPO / ".redline-work/redline-daemon-phases.json"))
    parser.add_argument("--log", default=str(REPO / ".redline-work/redline-daemon-phases.log"))
    parser.add_argument("--prefill", type=int, nargs="+", default=[128, 512])
    parser.add_argument(
        "--skip-prefill",
        action="store_true",
        help="run only the decode capture, contract probe, and shadow parity gate",
    )
    parser.add_argument("--decode-context", type=int, default=128)
    parser.add_argument(
        "--kv-mode",
        choices=("q8", "fwht2", "fwht3", "fwht4"),
        default="q8",
        help="KV layout used by capture, shadow replay, and the HIP oracle",
    )
    parser.add_argument("--capture-repeats", type=int, default=2)
    parser.add_argument("--measure-repeats", type=int, default=5)
    parser.add_argument("--decode-iterations", type=int, default=100)
    parser.add_argument(
        "--shadow-iterations",
        type=int,
        default=1,
        help="consecutive token positions compared by the AQL/HIP/blob parity gate",
    )
    parser.add_argument(
        "--shadow-position-step",
        type=int,
        default=1,
        help="position increment per shadow replay; 0 isolates queue lifetime from context growth",
    )
    parser.add_argument(
        "--qwen4",
        action="store_true",
        help="gate the Qwen4 (Qwen3.8 Flash-Next) multi-position state-parity table",
    )
    parser.add_argument(
        "--shadow-replay-only",
        action="store_true",
        help="run only retained PM4 shadow replays, without HIP/blob parity arms",
    )
    parser.add_argument(
        "--state-quant",
        choices=("q8", "fp32", "q4"),
        help=(
            "DeltaNet state precision for the run. Omit to validate the production "
            "default: Q8 with default-on deterministic error feedback (q8_ef). "
            "Pass fp32 only when explicitly testing the FP32 state route. If "
            "HIPFIRE_DN_STATE_EF=0 selects legacy stochastic Q8, byte parity is "
            "meaningless; re-enable EF or use fp32 plus HIPFIRE_DETERMINISTIC=1."
        ),
    )
    parser.add_argument("--max-seq", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--prefix", type=int, help="compare only the first N captured launches")
    parser.add_argument(
        "--profile-prefix-step",
        type=int,
        help="profile retained-PM4 cumulative prefixes at this dispatch interval",
    )
    parser.add_argument(
        "--profile-prefix-repeats",
        type=int,
        default=3,
        help="GPU-timed repetitions per cumulative PM4 prefix (default: 3)",
    )
    parser.add_argument(
        "--profile-prefix-start",
        type=int,
        help="first cumulative dispatch prefix to profile (default: one step)",
    )
    parser.add_argument(
        "--profile-prefix-steady-state",
        action="store_true",
        help=(
            "prime the model once before retained-PM4 prefix timing instead of "
            "resetting and re-prefilling before every sample; diagnostic timing only"
        ),
    )
    parser.add_argument(
        "--pm4",
        action="store_true",
        help="lower --prefix to one retained PM4 indirect buffer",
    )
    parser.add_argument(
        "--dspark-verify-shadow",
        action="store_true",
        help=(
            "run the DSpark verify ordinary-HIP/capture-safe/blob/PM4 state oracle "
            "instead of the plain-AR phase harness"
        ),
    )
    parser.add_argument(
        "--dflash-verify-shadow",
        action="store_true",
        help=(
            "run the DFlash2 B=16 ordinary-HIP/capture-safe/blob/PM4 state oracle "
            "instead of the plain-AR phase harness"
        ),
    )
    parser.add_argument(
        "--dflash-timing-windows",
        type=int,
        default=200,
        help=(
            "interleaved HipGraph/PM4 steady-state windows for --dflash-verify-shadow "
            "(default: 200). Capture/prepare is amortized separately."
        ),
    )
    parser.add_argument(
        "--draft",
        help="draft sidecar path (required by --dspark-verify-shadow / --dflash-verify-shadow)",
    )
    parser.add_argument(
        "--verify-batch",
        type=int,
        default=3,
        help="fixed target verify batch for the DSpark shadow (default: 3)",
    )
    args = parser.parse_args()

    model = Path(args.model).expanduser().resolve()
    daemon_path = Path(args.daemon).expanduser().resolve()
    if not model.is_file():
        sys.exit(f"model not found: {model}")
    if not daemon_path.is_file():
        sys.exit(f"daemon not found: {daemon_path}")
    draft = Path(args.draft).expanduser().resolve() if args.draft else None
    if args.dspark_verify_shadow and (draft is None or not draft.is_file()):
        sys.exit("--dspark-verify-shadow requires an existing --draft sidecar")
    if args.dflash_verify_shadow and (draft is None or not draft.is_file()):
        sys.exit("--dflash-verify-shadow requires an existing --draft sidecar")
    if args.dspark_verify_shadow and args.dflash_verify_shadow:
        sys.exit("choose one of --dspark-verify-shadow or --dflash-verify-shadow")
    if args.dspark_verify_shadow:
        discovered_draft = model.with_name(f"{model.stem}-dspark{model.suffix}").resolve()
        if draft != discovered_draft:
            sys.exit(
                "DeepSeek4 discovers DSpark only as the sibling "
                f"{discovered_draft}; --draft resolved to {draft}"
            )

    if args.qwen4:
        recipe_failures = qwen4_recipe_failures(model)
        if recipe_failures:
            for line in recipe_failures:
                print(f"  FAIL {line}", flush=True)
            sys.exit(f"qwen4 artifact recipe preflight failed for {model}")

    report = {
        "model": str(model),
        "model_bytes": model.stat().st_size,
        "draft": str(draft) if draft is not None else None,
        "draft_bytes": draft.stat().st_size if draft is not None else None,
        "daemon": str(daemon_path),
        "kv_mode": args.kv_mode,
        "automatic_clocks_required": True,
        "prefill": {},
        "decode": {},
    }
    daemon = Daemon(daemon_path, Path(args.log), args.timeout, args.kv_mode)
    try:
        load_params = {
            "max_seq": args.max_seq,
            "kv_mode": args.kv_mode,
            "dflash_mode": "on" if args.dflash_verify_shadow else "off",
            "dspark_mode": "on" if args.dspark_verify_shadow else "off",
            "mtp_mode": "off",
            "ngram_draft": False,
        }
        # DeepSeek4 discovers `<stem>-dspark.<ext>` itself. Passing the same file
        # through params.draft would incorrectly enter the Qwen DFlash lm_head
        # eligibility gate before architecture dispatch. Qwen DFlash *does*
        # take params.draft.
        load_body = {
            **load_params,
            **({"state_quant": args.state_quant} if args.state_quant else {}),
            **({"draft": str(draft)} if args.dflash_verify_shadow else {}),
        }
        loaded = daemon.request(
            {
                "type": "load",
                "model": str(model),
                "params": load_body,
            }
        )
        if loaded.get("type") != "loaded":
            raise RuntimeError(f"unexpected load response: {loaded}")
        report["loaded"] = loaded
        print(
            f"loaded arch={loaded.get('arch')} dim={loaded.get('dim')} "
            f"layers={loaded.get('layers')} vocab={loaded.get('vocab')}",
            flush=True,
        )

        if args.dflash_verify_shadow:
            shadow = daemon.request(
                {
                    "type": "redline_dflash_verify_shadow_pm4",
                    "verify_batch": 16 if args.verify_batch == 3 else args.verify_batch,
                    "iterations": max(args.shadow_iterations, 4),
                    "steady_state_windows": args.dflash_timing_windows,
                }
            )
            report["dflash_verify_shadow"] = shadow
            failures = dflash_shadow_failures(shadow)
            route = shadow.get("route") or {}
            counters = route.get("counters") or {}
            timing = shadow.get("timing") or {}
            print(
                f"dflash-verify-shadow: B={shadow.get('verify_batch')} "
                f"positions={shadow.get('positions')} "
                f"phase={route.get('phase')} "
                f"replays={counters.get('replays')} "
                f"median_delta_ms={timing.get('median_delta_ms')} "
                f"percent_delta={timing.get('percent_delta')} "
                f"p95_delta_ms={timing.get('p95_delta_ms')}",
                flush=True,
            )
            report["pass"] = not failures
            report["dflash_verify_failures"] = failures
            output = Path(args.out)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2) + "\n")
            print(f"report={output} pass={report['pass']}", flush=True)
            for line in failures:
                print(f"  FAIL {line}", flush=True)
            if failures:
                raise SystemExit(
                    f"dflash-verify-shadow failed {len(failures)} check(s); see {output}"
                )
            return
        if args.dspark_verify_shadow:
            shadow = daemon.request(
                {
                    "type": "redline_dspark_shadow_pm4",
                    "context_tokens": args.decode_context,
                    "verify_batch": args.verify_batch,
                    "iterations": args.shadow_iterations,
                }
            )
            report["dspark_verify_shadow"] = shadow
            report["pass"] = bool(shadow.get("bit_exact"))
            print(
                f"dspark-verify-shadow: B={args.verify_batch} "
                f"positions={args.shadow_iterations} exact={report['pass']} "
                f"launches={shadow.get('capture', {}).get('launches')} "
                f"hash={shadow.get('capture', {}).get('sequence_hash')}",
                flush=True,
            )
            output = Path(args.out)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2) + "\n")
            print(f"report={output} pass={report['pass']}", flush=True)
            if not report["pass"]:
                raise SystemExit(1)
            return

        for tokens in ([] if args.skip_prefill else args.prefill):
            captures = [
                daemon.request(
                    {
                        "type": "bench_prefill",
                        "tokens": tokens,
                        "redline_capture": True,
                    }
                )
                for _ in range(args.capture_repeats)
            ]
            measures = [
                daemon.request({"type": "bench_prefill", "tokens": tokens})
                for _ in range(args.measure_repeats)
            ]
            stable = len({capture_key(row) for row in captures}) == 1
            report["prefill"][str(tokens)] = {
                "captures": captures,
                "sequence_stable": stable,
                "measurement": {
                    "tok_s": summarize([row["tok_s"] for row in measures]),
                    "ms": summarize([row["ms"] for row in measures]),
                    "runs": measures,
                },
            }
            cap = captures[0]["redline_capture"]
            print(
                f"prefill{tokens}: stable={stable} launches={cap['launches']} "
                f"kernels={cap['unique_kernels']} hash={cap['sequence_hash']} "
                f"median={report['prefill'][str(tokens)]['measurement']['tok_s']['median']:.1f} tok/s",
                flush=True,
            )

        captures = [
            daemon.request(
                {
                    "type": "bench_decode",
                    "context_tokens": args.decode_context,
                    "iterations": 1,
                    "redline_capture": True,
                    "redline_detail": True,
                }
            )
            for _ in range(args.capture_repeats)
        ]
        measures = [
            daemon.request(
                {
                    "type": "bench_decode",
                    "context_tokens": args.decode_context,
                    "iterations": args.decode_iterations,
                }
            )
            for _ in range(args.measure_repeats)
        ]
        stable = len({capture_key(row) for row in captures}) == 1
        report["decode"] = {
            "context_tokens": args.decode_context,
            "capture_iterations": 1,
            "captures": captures,
            "sequence_stable": stable,
            "measurement_iterations": args.decode_iterations,
            "measurement": {
                "tok_s": summarize([row["tok_s"] for row in measures]),
                "us_per_token": summarize([row["us_per_token"] for row in measures]),
                "runs": measures,
            },
        }
        cap = captures[0]["redline_capture"]
        print(
            f"decode: stable={stable} launches={cap['launches']} "
            f"kernels={cap['unique_kernels']} hash={cap['sequence_hash']} "
            f"median={report['decode']['measurement']['tok_s']['median']:.1f} tok/s",
            flush=True,
        )
        report["aql_contract_probe"] = daemon.request({"type": "redline_probe_aql"})
        print(
            f"aql-contracts: kernels={report['aql_contract_probe']['kernels']}",
            flush=True,
        )
        if args.profile_prefix_step is not None:
            report["pm4_prefix_profile"] = daemon.request(
                {
                    "type": "redline_pm4_prefix_profile",
                    "context_tokens": args.decode_context,
                    "step": args.profile_prefix_step,
                    "repeats": args.profile_prefix_repeats,
                    "steady_state": args.profile_prefix_steady_state,
                    **(
                        {"start": args.profile_prefix_start}
                        if args.profile_prefix_start is not None
                        else {}
                    ),
                }
            )
            print(
                f"pm4-prefix-profile: rows={len(report['pm4_prefix_profile']['rows'])} "
                f"repeats={args.profile_prefix_repeats} "
                f"steady_state={args.profile_prefix_steady_state}",
                flush=True,
            )
        if args.prefix is None:
            report["aql_shadow"] = daemon.request(
                {
                    "type": "redline_shadow_pm4" if args.pm4 else "redline_shadow_aql",
                    "context_tokens": args.decode_context,
                    "iterations": args.shadow_iterations,
                    "position_step": args.shadow_position_step,
                    "replay_only": args.shadow_replay_only,
                }
            )
            shadow = report["aql_shadow"]
            if args.shadow_replay_only:
                shadow_pass = bool(shadow.get("replay_only"))
                print(
                    f"shadow: backend=pm4_ib replay_only=true "
                    f"iterations={shadow.get('iterations')} "
                    f"position_step={shadow.get('position_step')} "
                    f"queue_id={shadow.get('queue_id')} "
                    f"host_us={shadow.get('aql_host_us'):.1f}",
                    flush=True,
                )
            elif args.qwen4:
                failures = qwen4_shadow_failures(shadow)
                report["qwen4_shadow_failures"] = failures
                shadow_pass = not failures
                positions = [w.get("position") for w in (shadow.get("parity") or {}).get("windows", [])]
                print(
                    f"qwen4-shadow: positions={positions} "
                    f"bit_exact={shadow.get('bit_exact')} "
                    f"blob_bit_exact={shadow.get('blob_bit_exact')} "
                    f"state_bytes={shadow.get('state_bytes_compared_per_position')} "
                    f"failures={failures}",
                    flush=True,
                )
            else:
                frame_values = [
                    shadow.get(arm, {}).get("gdn_frame")
                    for arm in ("hip", "aql", "blob")
                ]
                frame_present = any(frame is not None for frame in frame_values)
                frame_exact = not frame_present or (
                    all(frame is not None for frame in frame_values)
                    and frame_values[0] == frame_values[1] == frame_values[2]
                )
                shadow["gdn_frame_exact"] = frame_exact
                shadow_pass = (
                    bool(shadow["bit_exact"])
                    and bool(shadow["blob_bit_exact"])
                    and frame_exact
                )
                print(
                    f"shadow: backend={'pm4_ib' if args.pm4 else 'aql_packets'} "
                    f"exact={shadow_pass} gdn_frame_exact={frame_exact} "
                    f"aql={report['aql_shadow']['aql_host_us']:.1f}us "
                    f"hip={report['aql_shadow']['hip_host_us']:.1f}us",
                    flush=True,
                )
        else:
            report["prefix_shadow"] = daemon.request(
                {
                    "type": "redline_prefix_shadow",
                    "context_tokens": args.decode_context,
                    "prefix": args.prefix,
                    "pm4": args.pm4,
                }
            )
            shadow_pass = report["prefix_shadow"]["equal"]
            print(
                f"prefix-shadow: backend={'pm4_ib' if args.pm4 else 'aql_packets'} "
                f"prefix={args.prefix} exact={shadow_pass} "
                f"differing={report['prefix_shadow']['differing']}",
                flush=True,
            )

        report["pass"] = all(
            row["sequence_stable"] for row in report["prefill"].values()
        ) and report["decode"]["sequence_stable"] \
            and report["aql_contract_probe"]["kernels"] > 0 \
            and shadow_pass
    finally:
        daemon.close()

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"report={output} pass={report['pass']}", flush=True)
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
