#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""HIP-only gfx1100 packed LDS residual final product proof (developer runner).

Reuses the packed stage2 AB recipe shape as Python orchestration.
Parent owns GPU + shared hwgate locks and outer timeout; this script does NOT
acquire hardware locks or poll clocks/telemetry.

Two SEPARATE routes (never cross-compared):
  1. demo    — dflash_spec_demo; 1 throwaway/arm then ABBAABBA (4 measured/arm)
  2. product — hipfire bench --spec dflash; 1 throwaway/arm (--runs 1) then
               ABBAAB (3 measured/arm, --runs 5)

Baseline RESIDUAL_LDSSTAGE=0; candidate=1. Fail-fast on nonzero; no retry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths / pins (contract)
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent
BASE_BIN = REPO / ".scratch" / "residual-variant-sweep" / "baseline-bin"
CAND_BIN = REPO / ".scratch" / "residual-variant-sweep" / "candidate-bin"

PROMPT = REPO / "benchmarks" / "prompts" / "merge_sort_thinking_off.txt"
PROMPT_MD5 = "253c7ac50857fe6d0e10fb0d2c5e35c0"

MODEL_PATH = Path("/home/kaden/.hipfire/models/qwen3.8-27b.mq4-xt")
DRAFT_PATH = Path("/home/kaden/.hipfire/models/qwen38-27b-dflash-mq4.hfq")
MODELS_DIR = Path("/home/kaden/.hipfire/models")
PRODUCT_MODEL_TAG = "qwen3.8:27b-mq4-xt"

# SHA-256 pins (full model/draft files)
MODEL_SHA256 = "9f91556f7e0431a077d03756a7102d0154108757289e6e5fe9a2d204c0c9eeb7"
DRAFT_SHA256 = "d0a74a232a0e2166d889f823e91e0fbf778d21dd9668d7de055cdecb065401bc"
# MD5 pins recorded from residual-variant-sweep/full-model-identities.md5
MODEL_MD5 = "e45d15bfe0c9a87132697101d17cbed6"
DRAFT_MD5 = "013395583cd04206c8aa68f4d061983d"

PROC_TIMEOUT_S = 240
SCHEMA = "residual-final-ab-v1"

DEMO_ORDER_MEASURED = (
    "baseline",
    "candidate",
    "candidate",
    "baseline",
    "baseline",
    "candidate",
    "candidate",
    "baseline",
)  # ABBAABBA — 4/arm
PRODUCT_ORDER_MEASURED = (
    "baseline",
    "candidate",
    "candidate",
    "baseline",
    "baseline",
    "candidate",
)  # ABBAAB — 3/arm

BIN_NAMES = ("hipfire", "daemon", "dflash_spec_demo")


class Fatal(SystemExit):
    def __init__(self, msg: str, code: int = 1) -> None:
        super().__init__(code)
        self.msg = msg


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    print(f"[residual-final] {msg}", flush=True)


def die(msg: str) -> None:
    print(f"[residual-final] FATAL: {msg}", file=sys.stderr, flush=True)
    raise Fatal(msg)


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_exec(path: Path, label: str) -> None:
    if not path.is_file() or not os.access(path, os.X_OK):
        die(f"missing executable {label}: {path}")


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        die(f"missing {label}: {path}")


def clean_inherited_env() -> dict[str, str]:
    """Drop HIPFIRE_* and visible-device, keep host ROCm/path env intact."""
    out: dict[str, str] = {}
    for k, v in os.environ.items():
        if k.startswith("HIPFIRE_"):
            continue
        if k in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
            continue
        out[k] = v
    return out


def contract_env(arm: str, extra: dict[str, str] | None = None) -> dict[str, str]:
    """Explicit residual-final contract env for one arm."""
    env = clean_inherited_env()
    lds = "0" if arm == "baseline" else "1"
    base = {
        "HIP_VISIBLE_DEVICES": "0",
        "HIPFIRE_VERIFY_GRAPH": "0",
        "HIPFIRE_REPLAY_BACKEND": "hip",
        "HIPFIRE_RESIDUAL_KSPLIT_OFF": "0",
        "HIPFIRE_RESIDUAL_LDSSTAGE": lds,
        "HIPFIRE_DPM_WARMUP_SECS": "10",
        "HIPFIRE_NO_REGISTRY_FETCH": "1",
    }
    env.update(base)
    if extra:
        env.update(extra)
    return env


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def kill_process_group(proc: subprocess.Popen[Any]) -> None:
    if proc.pid is None:
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        proc.kill()
    except Exception:
        pass
    try:
        proc.wait(timeout=5)
    except Exception:
        pass


def run_captured(
    argv: list[str],
    env: dict[str, str],
    out_dir: Path,
    timeout_s: int = PROC_TIMEOUT_S,
) -> int:
    """Run argv in a new session; capture stdout/stderr/env/argv/rc. PG cleanup."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = out_dir / "stdout"
    stderr_path = out_dir / "stderr"
    write_json(
        out_dir / "env.json",
        {k: env[k] for k in sorted(env) if k.startswith("HIP") or k.startswith("HIPFIRE_")},
    )
    write_text(
        out_dir / "argv.txt",
        " ".join(json.dumps(a) for a in argv) + "\n",
    )
    write_json(out_dir / "argv.json", argv)
    write_text(out_dir / "start.iso", utc_now() + "\n")

    rc = 124  # timeout sentinel
    timed_out = False
    t0 = time.monotonic()
    with stdout_path.open("wb") as so, stderr_path.open("wb") as se:
        proc: subprocess.Popen[Any] | None = None
        try:
            proc = subprocess.Popen(
                argv,
                stdout=so,
                stderr=se,
                env=env,
                start_new_session=True,
                cwd=str(REPO),
            )
            try:
                rc = proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                kill_process_group(proc)
                rc = 124
        except Exception as exc:
            write_text(out_dir / "launch_error.txt", f"{type(exc).__name__}: {exc}\n")
            if proc is not None:
                kill_process_group(proc)
            rc = 127
        finally:
            if proc is not None and proc.poll() is None:
                kill_process_group(proc)
            # Always ensure no orphan group remains after abnormal end.
            if proc is not None and (timed_out or (rc is not None and rc != 0)):
                kill_process_group(proc)

    elapsed = time.monotonic() - t0
    write_text(out_dir / "exit_status", f"{rc}\n")
    write_text(out_dir / "end.iso", utc_now() + "\n")
    write_text(out_dir / "elapsed_s", f"{elapsed:.3f}\n")
    if timed_out:
        write_text(out_dir / "timeout.txt", f"timeout_s={timeout_s}\n")
    return int(rc)


def parse_demo_metrics(stderr_path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not stderr_path.is_file():
        out["parse_error"] = "missing_err"
        return out
    text = stderr_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    def last_kv(prefix: str) -> str | None:
        val = None
        for ln in lines:
            if ln.startswith(prefix):
                parts = ln.split(None, 1)
                if len(parts) >= 2:
                    val = parts[1].strip()
        return val

    for key, pref in (
        ("decode_tok_s", "decode_tok_s:"),
        ("decode_tau", "decode_tau:"),
        ("decode_tokens_emitted", "decode_tokens_emitted:"),
        ("decode_accept_rate", "decode_accept_rate:"),
    ):
        v = last_kv(pref)
        if v is not None:
            out[key] = v

    cycles = None
    for ln in lines:
        if ln.startswith("cycles:"):
            # "cycles: 11  committed: ..." or "cycles: 11"
            m = re.match(r"cycles:\s+(\S+)", ln)
            if m:
                cycles = m.group(1)
                break
    if cycles is not None:
        out["cycles"] = cycles

    token_lines = [ln for ln in lines if ln.startswith("DFlash tokens:")]
    if token_lines:
        blob = "\n".join(token_lines).encode("utf-8", errors="replace")
        digest = hashlib.sha256(blob).hexdigest()
        out["token_sha256"] = digest
        out["token_sha8"] = digest[:8]
    return out


def extract_last_json_object(text: str) -> Any | None:
    decoder = json.JSONDecoder()
    last = None
    i = 0
    n = len(text)
    while i < n:
        if text[i] == "{":
            try:
                obj, end = decoder.raw_decode(text, i)
                last = obj
                i = end
                continue
            except json.JSONDecodeError:
                pass
        i += 1
    return last


def parse_product_report(stdout_path: Path, report_path: Path) -> dict[str, str]:
    metrics: dict[str, str] = {}
    if not stdout_path.is_file():
        metrics["parse_error"] = "missing_stdout"
        write_json(report_path, {})
        return metrics
    text = stdout_path.read_text(encoding="utf-8", errors="replace")
    obj = extract_last_json_object(text)
    if obj is None:
        write_json(report_path, {})
        metrics["parse_error"] = "missing_or_unparsed"
        return metrics
    write_json(report_path, obj)

    def dig(d: Any, *ks: str) -> Any:
        cur = d
        for k in ks:
            if not isinstance(cur, dict) or k not in cur:
                return None
            cur = cur[k]
        return cur

    for key in ("decode_tok_s", "prefill_tok_s", "wall_tok_s", "ttft_ms"):
        stats = dig(obj, key)
        if isinstance(stats, dict):
            for sk in ("mean", "median", "min", "max", "std", "stdev"):
                if sk in stats:
                    metrics[f"{key}.{sk}"] = str(stats[sk])
        elif stats is not None:
            metrics[key] = str(stats)
    for key in ("prompt_tokens", "prompt_md5", "prompt_chars", "max_tokens", "runs"):
        if isinstance(obj, dict) and key in obj:
            metrics[key] = str(obj[key])
    samples = dig(obj, "samples")
    if samples is not None:
        # Keep whole product JSON samples (also in report.json).
        metrics["samples_json"] = json.dumps(samples, sort_keys=True)
    return metrics


def arm_stats(runs: list[dict[str, Any]], arm: str, key: str) -> dict[str, Any] | None:
    vals: list[float] = []
    for r in runs:
        if r.get("arm") != arm or r.get("throwaway"):
            continue
        m = r.get("metrics") or {}
        raw = m.get(key)
        if raw is None or raw == "":
            continue
        try:
            vals.append(float(raw))
        except (TypeError, ValueError):
            continue
    if not vals:
        return None
    vals_sorted = sorted(vals)
    n = len(vals_sorted)
    mean = sum(vals_sorted) / n
    med = (
        vals_sorted[n // 2]
        if n % 2
        else 0.5 * (vals_sorted[n // 2 - 1] + vals_sorted[n // 2])
    )
    return {
        "n": n,
        "mean": mean,
        "median": med,
        "min": vals_sorted[0],
        "max": vals_sorted[-1],
        "values": vals_sorted,
    }


class Runner:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.demo_root = out_dir / "demo"
        self.product_root = out_dir / "product"
        self.hashes_dir = out_dir / "hashes"
        self.meta_dir = out_dir / "meta"
        self.log_path = out_dir / "run.log"
        self.summary_path = out_dir / "summary.json"
        self.git_head: str | None = None
        self.binary_md5: dict[str, str] = {}
        self.binary_sha256: dict[str, str] = {}
        self.fixture_md5: dict[str, str] = {}
        self.fixture_sha256: dict[str, str] = {}
        self.bins_meta: dict[str, Any] = {}
        self.demo_runs: list[dict[str, Any]] = []
        self.product_runs: list[dict[str, Any]] = []
        self.base_hipfire = BASE_BIN / "hipfire"
        self.base_daemon = BASE_BIN / "daemon"
        self.base_demo = BASE_BIN / "dflash_spec_demo"
        self.cand_hipfire = CAND_BIN / "hipfire"
        self.cand_daemon = CAND_BIN / "daemon"
        self.cand_demo = CAND_BIN / "dflash_spec_demo"
        self._log_fp = None

    def open_log(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=False)
        self.demo_root.mkdir()
        self.product_root.mkdir()
        self.hashes_dir.mkdir()
        self.meta_dir.mkdir()
        self._log_fp = self.log_path.open("w", encoding="utf-8")

    def close_log(self) -> None:
        if self._log_fp is not None:
            self._log_fp.close()
            self._log_fp = None

    def progress(self, msg: str) -> None:
        line = f"[residual-final] {msg}"
        print(line, flush=True)
        if self._log_fp is not None:
            self._log_fp.write(line + "\n")
            self._log_fp.flush()

    def preflight(self) -> None:
        for p, lab in (
            (self.base_hipfire, "baseline-hipfire"),
            (self.base_daemon, "baseline-daemon"),
            (self.base_demo, "baseline-dflash_spec_demo"),
            (self.cand_hipfire, "candidate-hipfire"),
            (self.cand_daemon, "candidate-daemon"),
            (self.cand_demo, "candidate-dflash_spec_demo"),
        ):
            require_exec(p, lab)
        require_file(PROMPT, "prompt")
        require_file(MODEL_PATH, "model")
        require_file(DRAFT_PATH, "draft")

        prompt_md5 = file_md5(PROMPT)
        if prompt_md5 != PROMPT_MD5:
            die(f"prompt md5 drift: got {prompt_md5} want {PROMPT_MD5}")

        self.progress("hashing binaries + fixtures (pre-GPU reject gate)")
        labeled_md5: list[str] = []
        labeled_sha: list[str] = []
        all_lines: list[str] = [f"# residual-final provenance hashes {utc_now()}"]

        def record_bin(path: Path, label: str) -> None:
            md = file_md5(path)
            sh = file_sha256(path)
            self.binary_md5[label] = md
            self.binary_sha256[label] = sh
            labeled_md5.append(f"{label} {md}")
            labeled_sha.append(f"{label} {sh}")
            all_lines.append(f"md5 {md}  {path}")
            all_lines.append(f"sha256 {sh}  {path}")

        record_bin(self.base_hipfire, "baseline-hipfire")
        record_bin(self.base_daemon, "baseline-daemon")
        record_bin(self.base_demo, "baseline-dflash_spec_demo")
        record_bin(self.cand_hipfire, "candidate-hipfire")
        record_bin(self.cand_daemon, "candidate-daemon")
        record_bin(self.cand_demo, "candidate-dflash_spec_demo")

        # Fixtures: md5 + sha pins; reject mismatch before any GPU work.
        fixtures = [
            (PROMPT, "prompt-merge_sort_thinking_off", PROMPT_MD5, None),
            (MODEL_PATH, "model-qwen3.8-27b.mq4-xt", MODEL_MD5, MODEL_SHA256),
            (DRAFT_PATH, "draft-qwen38-27b-dflash-mq4.hfq", DRAFT_MD5, DRAFT_SHA256),
        ]
        for path, label, want_md5, want_sha in fixtures:
            md = file_md5(path)
            sh = file_sha256(path)
            self.fixture_md5[label] = md
            self.fixture_sha256[label] = sh
            labeled_md5.append(f"{label} {md}")
            labeled_sha.append(f"{label} {sh}")
            all_lines.append(f"md5 {md}  {path}")
            all_lines.append(f"sha256 {sh}  {path}")
            if want_md5 is not None and md != want_md5:
                die(f"{label} md5 mismatch: got {md} want {want_md5}")
            if want_sha is not None and sh != want_sha:
                die(f"{label} sha256 mismatch: got {sh} want {want_sha}")

        write_text(self.hashes_dir / "all.txt", "\n".join(all_lines) + "\n")
        write_text(self.hashes_dir / "labeled.md5", "\n".join(labeled_md5) + "\n")
        write_text(self.hashes_dir / "labeled.sha256", "\n".join(labeled_sha) + "\n")

        try:
            self.git_head = (
                subprocess.check_output(
                    ["git", "-C", str(REPO), "rev-parse", "HEAD"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
            )
        except Exception:
            self.git_head = None
        all_lines.insert(1, f"git_head {self.git_head}")
        write_text(self.hashes_dir / "all.txt", "\n".join(all_lines) + "\n")

        self.bins_meta = {
            "git_head": self.git_head,
            "run_dir": str(self.out_dir),
            "prompt": str(PROMPT),
            "prompt_md5": PROMPT_MD5,
            "model_path": str(MODEL_PATH),
            "model_md5": MODEL_MD5,
            "model_sha256": MODEL_SHA256,
            "draft_path": str(DRAFT_PATH),
            "draft_md5": DRAFT_MD5,
            "draft_sha256": DRAFT_SHA256,
            "demo": {
                "target": str(MODEL_PATH),
                "draft": str(DRAFT_PATH),
                "baseline_bin": str(self.base_demo),
                "candidate_bin": str(self.cand_demo),
                "flags": [
                    "--max",
                    "256",
                    "--temp",
                    "0.0",
                    "--no-chatml",
                    "--kv-mode",
                    "q8",
                    "--ctx",
                    "4096",
                    "--no-adaptive-b",
                ],
                "order": "ABBAABBA",
                "runs_per_arm": 4,
                "throwaway_per_arm": 1,
            },
            "product": {
                "model": PRODUCT_MODEL_TAG,
                "model_path": str(MODEL_PATH),
                "draft": str(DRAFT_PATH),
                "baseline_cli": str(self.base_hipfire),
                "baseline_daemon": str(self.base_daemon),
                "candidate_cli": str(self.cand_hipfire),
                "candidate_daemon": str(self.cand_daemon),
                "flags_measured": [
                    "bench",
                    PRODUCT_MODEL_TAG,
                    "--spec",
                    "dflash",
                    "--runs",
                    "5",
                    "--warmups",
                    "3",
                    "--max-tokens",
                    "256",
                    "--backend",
                    "noslots",
                    "--workload",
                    "stateless",
                    "--kv-mode",
                    "q8",
                    "--json",
                    "--prompt-file",
                    str(PROMPT),
                ],
                "flags_throwaway": [
                    "bench",
                    PRODUCT_MODEL_TAG,
                    "--spec",
                    "dflash",
                    "--runs",
                    "1",
                    "--warmups",
                    "3",
                    "--max-tokens",
                    "256",
                    "--backend",
                    "noslots",
                    "--workload",
                    "stateless",
                    "--kv-mode",
                    "q8",
                    "--json",
                    "--prompt-file",
                    str(PROMPT),
                ],
                "order": "ABBAAB",
                "runs_per_arm": 3,
                "throwaway_per_arm": 1,
                "warmups_note": (
                    "--warmups 3 is passed consistently; prior scout found "
                    "standard bench does one Hello warmup, not necessarily 3."
                ),
                "note": "baseline CLI → baseline daemon; candidate CLI → candidate daemon",
            },
            "contract_env": {
                "both": {
                    "HIP_VISIBLE_DEVICES": "0",
                    "HIPFIRE_VERIFY_GRAPH": "0",
                    "HIPFIRE_REPLAY_BACKEND": "hip",
                    "HIPFIRE_RESIDUAL_KSPLIT_OFF": "0",
                    "HIPFIRE_DPM_WARMUP_SECS": "10",
                    "HIPFIRE_NO_REGISTRY_FETCH": "1",
                },
                "baseline": {"HIPFIRE_RESIDUAL_LDSSTAGE": "0"},
                "candidate": {"HIPFIRE_RESIDUAL_LDSSTAGE": "1"},
            },
            "routes_isolated": True,
            "never_compare_demo_vs_product": True,
            "proc_timeout_s": PROC_TIMEOUT_S,
            "locks": "parent-owned (GPU + shared hwgate); runner does not acquire",
        }
        write_json(self.meta_dir / "bins.json", self.bins_meta)
        write_json(
            self.meta_dir / "binary_hashes.json",
            {"md5": self.binary_md5, "sha256": self.binary_sha256},
        )
        write_json(
            self.meta_dir / "fixture_hashes.json",
            {"md5": self.fixture_md5, "sha256": self.fixture_sha256},
        )

    def write_summary(self) -> None:
        labeled = dict(self.binary_md5)
        labeled.update(self.fixture_md5)

        def route_block(
            order: str,
            runs_per_arm: int,
            runs: list[dict[str, Any]],
            measured_key_primary: str,
            measured_key_fallback: str | None = None,
            tau_key: str | None = None,
        ) -> dict[str, Any]:
            measured = [r for r in runs if not r.get("throwaway")]
            throwaways = [r for r in runs if r.get("throwaway")]
            b = arm_stats(measured, "baseline", measured_key_primary)
            c = arm_stats(measured, "candidate", measured_key_primary)
            if b is None and measured_key_fallback:
                b = arm_stats(measured, "baseline", measured_key_fallback)
            if c is None and measured_key_fallback:
                c = arm_stats(measured, "candidate", measured_key_fallback)
            block: dict[str, Any] = {
                "order": order,
                "runs_per_arm": runs_per_arm,
                "runs": measured,
                "throwaways": throwaways,
                "baseline_decode_tok_s": b,
                "candidate_decode_tok_s": c,
            }
            if tau_key:
                block["baseline_decode_tau"] = arm_stats(measured, "baseline", tau_key)
                block["candidate_decode_tau"] = arm_stats(measured, "candidate", tau_key)
                block["token_sha8_by_run"] = [
                    {
                        "arm": r["arm"],
                        "slot": r.get("slot"),
                        "token_sha8": (r.get("metrics") or {}).get("token_sha8"),
                    }
                    for r in measured
                ]
            bm = (b or {}).get("mean")
            cm = (c or {}).get("mean")
            if isinstance(bm, (int, float)) and isinstance(cm, (int, float)) and bm > 0:
                block["within_route_gain_pct"] = (cm - bm) / bm * 100.0
            else:
                block["within_route_gain_pct"] = None
            return block

        summary = {
            "schema": SCHEMA,
            "run_dir": str(self.out_dir),
            "git_head": self.git_head,
            "prompt_md5": PROMPT_MD5,
            "model_md5": MODEL_MD5,
            "model_sha256": MODEL_SHA256,
            "draft_md5": DRAFT_MD5,
            "draft_sha256": DRAFT_SHA256,
            "binary_md5": labeled,
            "binary_sha256": dict(self.binary_sha256),
            "fixture_sha256": dict(self.fixture_sha256),
            "bins": self.bins_meta,
            "note": (
                "Routes are independent. NEVER compare demo tok/s against "
                "product bench tok/s. baseline=LDSSTAGE0 candidate=LDSSTAGE1."
            ),
            "demo": route_block(
                "ABBAABBA",
                4,
                self.demo_runs,
                "decode_tok_s",
                tau_key="decode_tau",
            ),
            "product": route_block(
                "ABBAAB",
                3,
                self.product_runs,
                "decode_tok_s.mean",
                measured_key_fallback="decode_tok_s",
            ),
            "updated_at": utc_now(),
        }
        write_json(self.summary_path, summary)

    def _record_common(
        self,
        *,
        route: str,
        arm: str,
        slot: int | None,
        order_index: int | None,
        throwaway: bool,
        out_dir: Path,
        argv: list[str],
        env: dict[str, str],
        rc: int,
        metrics: dict[str, str],
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "route": route,
            "arm": arm,
            "slot": slot,
            "order_index": order_index,
            "throwaway": throwaway,
            "dir": str(out_dir),
            "exit_status": rc,
            "stdout": str(out_dir / "stdout"),
            "stderr": str(out_dir / "stderr"),
            "command": str(out_dir / "command.txt"),
            "argv": argv,
            "env_hipfire": {
                k: env[k]
                for k in sorted(env)
                if k.startswith("HIPFIRE_") or k == "HIP_VISIBLE_DEVICES"
            },
            "metrics": metrics,
            "status": {"rc": str(rc), **metrics},
        }
        if extra:
            entry.update(extra)
        return entry

    def run_demo_once(
        self,
        arm: str,
        *,
        throwaway: bool,
        slot: int | None,
        order_index: int | None,
    ) -> None:
        demo_bin = self.base_demo if arm == "baseline" else self.cand_demo
        if throwaway:
            name = f"throwaway-{arm}"
        else:
            assert slot is not None
            name = f"{arm}-{slot}"
        out_dir = self.demo_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        home = out_dir / "hipfire-home"
        home.mkdir(parents=True, exist_ok=True)
        # Minimal config; draft still passed on CLI for demo.
        write_text(
            home / "config.toml",
            "schema_version = 1\n\n[speculation]\ndflash = \"on\"\nmtp = \"off\"\n",
        )

        argv = [
            str(demo_bin),
            "--target",
            str(MODEL_PATH),
            "--draft",
            str(DRAFT_PATH),
            "--prompt-file",
            str(PROMPT),
            "--max",
            "256",
            "--temp",
            "0.0",
            "--no-chatml",
            "--kv-mode",
            "q8",
            "--ctx",
            "4096",
            "--no-adaptive-b",
        ]
        env = contract_env(
            arm,
            {
                "HIPFIRE_HOME": str(home / ".hipfire"),
            },
        )
        # Demo is in-process; strip daemon/cli identity leaks.
        env.pop("HIPFIRE_DAEMON_BIN", None)
        env.pop("HIPFIRE_CLI_BIN", None)
        env.pop("HIPFIRE_KERNEL_CACHE", None)
        env.pop("HIPFIRE_DFLASH_DRAFT", None)

        cmd_txt = "\n".join(
            [
                f"arm={arm}",
                f"slot={slot}",
                f"order_index={order_index}",
                f"throwaway={int(throwaway)}",
                f"bin={demo_bin}",
                f"bin_md5={self.binary_md5.get('baseline-dflash_spec_demo' if arm == 'baseline' else 'candidate-dflash_spec_demo', '')}",
                f"start={utc_now()}",
                "argv=" + " ".join(json.dumps(a) for a in argv),
                f"HIPFIRE_RESIDUAL_LDSSTAGE={env['HIPFIRE_RESIDUAL_LDSSTAGE']}",
            ]
        )
        write_text(out_dir / "command.txt", cmd_txt + "\n")
        self.progress(
            f"DEMO START arm={arm} throwaway={throwaway} slot={slot} order={order_index} bin={demo_bin}"
        )
        rc = run_captured(argv, env, out_dir)
        metrics = parse_demo_metrics(out_dir / "stderr")
        write_text(
            out_dir / "metrics.txt",
            "".join(f"{k}={v}\n" for k, v in metrics.items()),
        )
        write_text(
            out_dir / "status.txt",
            f"rc={rc}\n" + "".join(f"{k}={v}\n" for k, v in metrics.items()),
        )
        entry = self._record_common(
            route="demo",
            arm=arm,
            slot=slot,
            order_index=order_index,
            throwaway=throwaway,
            out_dir=out_dir,
            argv=argv,
            env=env,
            rc=rc,
            metrics=metrics,
            extra={"bin": str(demo_bin), "bin_md5": file_md5(demo_bin)},
        )
        self.demo_runs.append(entry)
        self.write_summary()
        self.progress(
            f"DEMO DONE arm={arm} throwaway={throwaway} slot={slot} rc={rc} "
            f"tok_s={metrics.get('decode_tok_s')} tau={metrics.get('decode_tau')} "
            f"sha8={metrics.get('token_sha8')}"
        )
        if rc != 0:
            try:
                tail = (out_dir / "stderr").read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
                for ln in tail:
                    print(f"[residual-final] DEMO STDERR | {ln}", file=sys.stderr, flush=True)
            except Exception:
                pass
            die(f"demo arm={arm} throwaway={throwaway} slot={slot} exited rc={rc} (no autofallback)")

    def run_product_once(
        self,
        arm: str,
        *,
        throwaway: bool,
        slot: int | None,
        order_index: int | None,
    ) -> None:
        if arm == "baseline":
            cli, daemon = self.base_hipfire, self.base_daemon
            cli_label, daemon_label = "baseline-hipfire", "baseline-daemon"
        else:
            cli, daemon = self.cand_hipfire, self.cand_daemon
            cli_label, daemon_label = "candidate-hipfire", "candidate-daemon"
        if throwaway:
            name = f"throwaway-{arm}"
        else:
            assert slot is not None
            name = f"{arm}-{slot}"
        out_dir = self.product_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        home = out_dir / "home"
        (home / ".hipfire").mkdir(parents=True, exist_ok=True)

        runs_n = "1" if throwaway else "5"
        argv = [
            str(cli),
            "bench",
            PRODUCT_MODEL_TAG,
            "--spec",
            "dflash",
            "--runs",
            runs_n,
            "--warmups",
            "3",
            "--max-tokens",
            "256",
            "--backend",
            "noslots",
            "--workload",
            "stateless",
            "--kv-mode",
            "q8",
            "--json",
            "--prompt-file",
            str(PROMPT),
        ]
        env = contract_env(
            arm,
            {
                "HIPFIRE_HOME": str(home),
                "HIPFIRE_LOCAL": "1",
                "HIPFIRE_MODELS_DIR": str(MODELS_DIR),
                "HIPFIRE_DAEMON_BIN": str(daemon),
                "HIPFIRE_DFLASH_DRAFT": str(DRAFT_PATH),
            },
        )
        env.pop("HIPFIRE_CLI_BIN", None)
        env.pop("HIPFIRE_KERNEL_CACHE", None)

        cmd_txt = "\n".join(
            [
                f"arm={arm}",
                f"slot={slot}",
                f"order_index={order_index}",
                f"throwaway={int(throwaway)}",
                f"cli={cli}",
                f"daemon={daemon}",
                f"cli_md5={self.binary_md5.get(cli_label, '')}",
                f"daemon_md5={self.binary_md5.get(daemon_label, '')}",
                f"start={utc_now()}",
                "argv=" + " ".join(json.dumps(a) for a in argv),
                f"HIPFIRE_DAEMON_BIN={daemon}",
                f"HIPFIRE_DFLASH_DRAFT={DRAFT_PATH}",
                "HIPFIRE_LOCAL=1",
                f"HIPFIRE_MODELS_DIR={MODELS_DIR}",
                f"HIPFIRE_RESIDUAL_LDSSTAGE={env['HIPFIRE_RESIDUAL_LDSSTAGE']}",
                f"model={PRODUCT_MODEL_TAG}",
            ]
        )
        write_text(out_dir / "command.txt", cmd_txt + "\n")
        self.progress(
            f"PRODUCT START arm={arm} throwaway={throwaway} slot={slot} order={order_index} "
            f"cli={cli} daemon={daemon} runs={runs_n}"
        )
        rc = run_captured(argv, env, out_dir)
        metrics = parse_product_report(out_dir / "stdout", out_dir / "report.json")
        write_text(
            out_dir / "metrics.txt",
            "".join(f"{k}={v}\n" for k, v in metrics.items()),
        )
        report_status = (
            "report=present"
            if (out_dir / "report.json").is_file()
            and (out_dir / "report.json").stat().st_size > 3
            and "parse_error" not in metrics
            else "report=missing_or_unparsed"
        )
        write_text(out_dir / "report_status.txt", report_status + "\n")
        write_text(
            out_dir / "status.txt",
            f"rc={rc}\n"
            + "".join(f"{k}={v}\n" for k, v in metrics.items())
            + report_status
            + "\n",
        )
        report_obj = None
        try:
            report_obj = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
        except Exception:
            report_obj = None
        entry = self._record_common(
            route="product",
            arm=arm,
            slot=slot,
            order_index=order_index,
            throwaway=throwaway,
            out_dir=out_dir,
            argv=argv,
            env=env,
            rc=rc,
            metrics=metrics,
            extra={
                "cli": str(cli),
                "daemon": str(daemon),
                "cli_md5": self.binary_md5.get(cli_label),
                "daemon_md5": self.binary_md5.get(daemon_label),
                "report_json": str(out_dir / "report.json"),
                "report": report_obj,  # whole product JSON (incl. samples)
                "report_status": report_status,
            },
        )
        self.product_runs.append(entry)
        self.write_summary()
        self.progress(
            f"PRODUCT DONE arm={arm} throwaway={throwaway} slot={slot} rc={rc} "
            f"tok_s={metrics.get('decode_tok_s.mean') or metrics.get('decode_tok_s')}"
        )
        if rc != 0:
            try:
                tail = (out_dir / "stderr").read_text(encoding="utf-8", errors="replace").splitlines()[-60:]
                for ln in tail:
                    print(f"[residual-final] PRODUCT STDERR | {ln}", file=sys.stderr, flush=True)
            except Exception:
                pass
            die(
                f"product arm={arm} throwaway={throwaway} slot={slot} exited rc={rc} (no autofallback)"
            )

    def run_all(self) -> None:
        self.progress(f"READY run_dir={self.out_dir} head={self.git_head}")
        self.progress(
            "routes=demo(throwaway×2 + ABBAABBA dflash_spec_demo) "
            "product(throwaway×2 + ABBAAB hipfire-bench) — never cross-compared"
        )
        self.write_summary()

        # --- Route 1: demo ---
        self.progress("ROUTE demo THROW AWAY START (1/arm)")
        for arm in ("baseline", "candidate"):
            self.run_demo_once(arm, throwaway=True, slot=None, order_index=None)
        self.progress("ROUTE demo MEASURED START order=ABBAABBA")
        slots: dict[str, int] = defaultdict(int)
        for order_i, arm in enumerate(DEMO_ORDER_MEASURED):
            slot = slots[arm]
            self.run_demo_once(arm, throwaway=False, slot=slot, order_index=order_i)
            slots[arm] = slot + 1
        self.progress(
            f"ROUTE demo DONE baseline_slots={slots['baseline']} candidate_slots={slots['candidate']}"
        )

        # --- Route 2: product ---
        self.progress("ROUTE product THROW AWAY START (1/arm, --runs 1)")
        for arm in ("baseline", "candidate"):
            self.run_product_once(arm, throwaway=True, slot=None, order_index=None)
        self.progress("ROUTE product MEASURED START order=ABBAAB")
        slots = defaultdict(int)
        for order_i, arm in enumerate(PRODUCT_ORDER_MEASURED):
            slot = slots[arm]
            self.run_product_once(arm, throwaway=False, slot=slot, order_index=order_i)
            slots[arm] = slot + 1
        self.progress(
            f"ROUTE product DONE baseline_slots={slots['baseline']} candidate_slots={slots['candidate']}"
        )

        self.write_summary()
        self.progress(f"ALL-DONE run_dir={self.out_dir} summary={self.summary_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Residual packed-LDS final demo+product AB runner (parent holds locks)."
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory (must not already exist).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = args.out_dir.expanduser()
    if not out_dir.is_absolute():
        out_dir = (Path.cwd() / out_dir).resolve()
    else:
        out_dir = out_dir.resolve()
    if out_dir.exists():
        print(
            f"[residual-final] FATAL: --out-dir already exists: {out_dir}",
            file=sys.stderr,
            flush=True,
        )
        return 2

    runner = Runner(out_dir)
    try:
        runner.open_log()
    except FileExistsError:
        print(
            f"[residual-final] FATAL: --out-dir already exists: {out_dir}",
            file=sys.stderr,
            flush=True,
        )
        return 2

    try:
        runner.preflight()
        runner.run_all()
        return 0
    except Fatal as e:
        try:
            runner.write_summary()
        except Exception:
            pass
        runner.progress(f"STOPPED fatal={e.msg}")
        return int(e.code) if isinstance(e.code, int) else 1
    except Exception as e:
        try:
            runner.write_summary()
        except Exception:
            pass
        tb = traceback.format_exc()
        try:
            write_text(out_dir / "meta" / "exception.txt", tb)
        except Exception:
            pass
        print(f"[residual-final] FATAL unhandled: {e}", file=sys.stderr, flush=True)
        print(tb, file=sys.stderr, flush=True)
        return 1
    finally:
        runner.close_log()


if __name__ == "__main__":
    sys.exit(main())
