#!/usr/bin/env python3
"""Read-only validation of the frozen overall run; never starts a server/model.

Usage: python3 verify-overall-receipts.py [--partial] [--out REPORT.json]
Default root: this script's directory. Exit 1 = observed failure (including
the optional session failure); exit 2 = missing evidence/coverage; exit 0 =
all gates pass, or a clean explicitly partial report while work is pending.
An optional session failure is retained even when mandatory phases pass.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys

CELLS = ("A1", "B1", "B2", "A2", "A3", "B3")
PHASES = ("warmup", "battery", "chain", "session", "staged", "longdecode")
ROWS = {"warmup": 5, "battery": 5, "chain": 5, "session": 8, "staged": 2, "longdecode": 1}
PROOF = re.compile(r"HIPFIRE_REPLAY_ROUTE_PROOF transport=(\S+) position=(\d+) request_id=(\S+) replays=(\d+)(?:\s|$)")
STATE = re.compile(r"qwen4 state: QSA (\S+) K/V \((\S+) context storage\), GDN (\S+) recurrent")
MD5 = re.compile(r"[0-9a-f]{32}\Z")
REQUEST = re.compile(r"chatcmpl-(\d+)-(\d+)\Z")
TEXT_FIELDS = ("content", "reasoning_content", "assistant_content", "tool_calls", "finish", "gen", "ctx")


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class Audit:
    def __init__(self, root, partial):
        self.root, self.partial = root, partial
        self.errors, self.pending, self.coverage, self.notes = [], [], [], []
        self.inputs, self.cases, self.evidence = {}, {}, {}
        self.manifest = self.read_json(root / "runtime/MANIFEST.json", "frozen manifest")

    def issue(self, kind, where, detail):
        self.errors.append({"category": kind, "where": where, "detail": detail})

    def read_bytes(self, path):
        data = path.read_bytes()
        self.inputs[str(path.relative_to(self.root))] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        return data

    def read_json(self, path, where):
        try:
            return json.loads(self.read_bytes(path))
        except (OSError, ValueError) as error:
            self.pending.append({"where": where, "reason": f"unavailable/unsettled JSON: {error}"})
            return None

    def check(self, ok, kind, where, detail):
        if not ok:
            self.issue(kind, where, detail)
        return bool(ok)

    def case(self, directory):
        run = self.read_json(directory / "run.json", directory.name)
        if not isinstance(run, dict):
            return
        name = run.get("name", directory.name.removeprefix("serve-"))
        match = re.fullmatch(r"([AB][123])(?:[-_]?extra)?", name)
        if not match:
            self.notes.append(f"Ignored unrecognized case {name!r} in {directory.name}")
            return
        primary = match.group(1)
        extra = name != primary
        cache = str(run.get("cache"))
        expected_cache = "0" if primary.startswith("A") else "1"
        self.check(cache == expected_cache, "mandatory", name, "cache flag disagrees with A/B identity")
        self.check(run.get("frozen_manifest") == self.manifest, "mandatory", name, "frozen manifest mismatch")
        env = run.get("env") or {}
        for key, value in {"HIPFIRE_QWEN4_EXPERT_CACHE": expected_cache, "HIPFIRE_DN_STATE_EF": "1",
                           "HIPFIRE_QWEN4_DECODE_SHARE": "off", "HIPFIRE_OFFLOAD_EXEC": "pcie",
                           "HIPFIRE_MTP_MODE": "off", "HIPFIRE_PLE_FORMAT": "q4"}.items():
            self.check(env.get(key) == value, "mandatory", name, f"unexpected frozen setting {key}={env.get(key)!r}")
        try:
            text = self.read_bytes(directory / "serve.log").decode(errors="replace")
        except OSError as error:
            self.pending.append({"where": name, "reason": str(error)})
            return
        lines = text.splitlines()
        proofs, states, stats = {}, [], []
        for index, line in enumerate(lines, 1):
            proof = PROOF.search(line)
            if proof:
                transport, position, request_id, count = proof.groups()
                proofs.setdefault(request_id, []).append({"line": index, "transport": transport,
                    "position": int(position), "replays": int(count), "raw": line})
            state = STATE.search(line)
            if state:
                states.append({"line": index, "qsa": state[1], "backend": state[2], "gdn": state[3], "raw": line})
            marker = "qwen4 exact expert cache stats: "
            if marker in line:
                try:
                    stats.append({"line": index, **json.loads(line.split(marker, 1)[1])})
                except ValueError:
                    if run.get("status") in ("PASS", "FAIL"):
                        self.issue("mandatory", name, f"malformed cache stats at line {index}")
        completed = {x.get("phase") for x in run.get("phase_results", []) if x.get("status") == "PASS"}
        case = {"name": name, "primary": primary, "extra_process": extra, "directory": str(directory),
                "cache": cache, "driver_status": run.get("status", "IN_PROGRESS"),
                "driver_failure": run.get("failure"), "phases": {}, "cache_stats_rows": len(stats),
                "observed_backend_receipts": run.get("observed_backend_receipts", [])}
        self.cases[name] = case
        ids, cli_pids, proof_lines = set(), set(), []
        for phase in PHASES:
            path = directory / f"{phase}.json"
            if not path.exists():
                continue
            rows = self.read_json(path, f"{name}/{phase}")
            if isinstance(rows, dict):
                rows = rows.get("rows")
            if not isinstance(rows, list):
                self.issue("optional_session" if phase == "session" else "mandatory", f"{name}/{phase}", "rows are not a list")
                continue
            kind = "optional_session" if phase == "session" else "mandatory"
            where = f"{name}/{phase}"
            before_errors = len(self.errors)
            self.check(len(rows) == ROWS[phase], kind, where, f"expected {ROWS[phase]} rows, got {len(rows)}")
            summaries = []
            for index, row in enumerate(rows):
                location = f"{where}[{index}]"
                if not isinstance(row, dict):
                    self.issue(kind, location, "row is not an object")
                    continue
                request_id = row.get("request_id")
                self.check(isinstance(request_id, str) and bool(REQUEST.fullmatch(request_id)), kind, location, "missing/malformed request ID")
                if isinstance(request_id, str):
                    self.check(request_id not in ids, kind, location, "duplicate request ID in one process")
                    ids.add(request_id)
                if isinstance(request_id, str) and REQUEST.fullmatch(request_id):
                    cli_pids.add(REQUEST.fullmatch(request_id)[1])
                for field in ("prompt_md5", "request_md5"):
                    self.check(isinstance(row.get(field), str) and bool(MD5.fullmatch(row[field])), kind, location, f"missing/malformed {field}")
                for field in ("attractor", "empty", "atem_leak"):
                    self.check(row.get(field) is False, kind, location, f"{field} is not explicitly false")
                self.check(row.get("finish") in ("stop", "length", "eos", "tool_calls"), kind, location, f"invalid finish {row.get('finish')!r}")
                self.check(row.get("stream_error") is None, kind, location, f"stream error {row.get('stream_error')!r}")
                self.check(row.get("terminal_count") == 1, kind, location, "terminal_count != 1")
                self.check(row.get("saw_done") is True, kind, location, "missing SSE [DONE]")
                self.check(row.get("post_terminal_bytes") == 0, kind, location, "late payload bytes after terminal")
                self.check(not row.get("retrieval_missing"), kind, location, f"retrieval missing: {row.get('retrieval_missing')!r}")
                gen, ctx, cached = row.get("gen"), row.get("ctx"), row.get("cached")
                valid_counts = all(isinstance(x, int) and not isinstance(x, bool) for x in (gen, ctx, cached))
                self.check(valid_counts and gen > 0 and ctx > 0 and 0 <= cached <= ctx, kind, location, "invalid token/context/cache counts")
                joined = proofs.get(request_id, []) if isinstance(request_id, str) else []
                if len(joined) != 1:
                    if self.partial and run.get("status") not in ("PASS", "FAIL") and not joined:
                        self.pending.append({"where": location, "reason": "request proof not yet present in active log"})
                    else:
                        self.issue(kind, location, f"expected exactly one request-scoped PM4 proof, got {len(joined)}")
                else:
                    self.check(joined[0]["transport"] in ("pm4", "pm4_ib") and joined[0]["replays"] > 0,
                               kind, location, "request did not prove PM4 replay")
                    proof_lines.append(joined[0]["line"])
                summaries.append({"row": index, "request_id": request_id, "prompt_md5": row.get("prompt_md5"),
                    "request_md5": row.get("request_md5"), "text_projection_sha256": digest({k: row.get(k) for k in TEXT_FIELDS}),
                    "gen": gen, "finish": row.get("finish"), "ctx": ctx, "cached": cached,
                    "uncached_prompt_tokens": ctx - cached if valid_counts else None,
                    "request_pm4_proof": joined, "harness_kv_metadata": {k: row.get(k) if row.get(k) is not None else "UNOBSERVED"
                        for k in ("kv_mode", "kv_backend", "kv_backend_legacy", "kv_backend_reason")}})
            evidence = {"owner": name, "primary": primary, "phase": phase, "rows": rows,
                        "confirmed_by_driver": phase in completed, "row_checks_pass": len(self.errors) == before_errors,
                        "summary": summaries, "stats": stats, "proofs": proofs}
            self.evidence[(name, phase)] = evidence
            case["phases"][phase] = {"driver_confirmed": phase in completed, "row_checks_pass": evidence["row_checks_pass"], "rows": summaries}
        self.check(len(cli_pids) == 1, "mandatory", name, f"expected one CLI process, observed {sorted(cli_pids)}")
        case["cli_pids_from_request_ids"] = sorted(cli_pids)
        if proof_lines:
            first_line = min(proof_lines)
            before = [x for x in states if x["line"] < first_line]
            active_state = before[-1] if before else None
            case["raw_startup_state_receipt"] = active_state
            case["earlier_startup_state_markers"] = before[:-1]
            self.check(active_state is not None and active_state["qsa"] == "fp8" and active_state["backend"] == "vmm" and active_state["gdn"] == "q8",
                       "mandatory", name, "no active startup receipt for native QSA fp8 / VMM / GDN q8")
            self.check(not [x for x in states if x["line"] >= first_line], "mandatory", name, "model state reloaded during observed requests")
            self.check(proof_lines == sorted(proof_lines), "mandatory", name, "request proofs do not follow serial workload order")
        if expected_cache == "0":
            self.check(not stats, "mandatory", name, "baseline unexpectedly emitted exact-cache stats")
        else:
            if not stats and self.partial and run.get("status") not in ("PASS", "FAIL"):
                self.pending.append({"where": name, "reason": "candidate cache stats pending"})
            else:
                self.check(bool(stats), "mandatory", name, "candidate has no cache execution stats")
            for previous, current in zip(stats, stats[1:]):
                for key in ("tokens", "fills_started", "fills_completed", "prefill_restores", "restore_bytes"):
                    self.check(current.get(key, -1) >= previous.get(key, 0), "mandatory", name, f"cache {key} decreased (owner reloaded/reset?)")
            case["last_cache_stats"] = stats[-1] if stats else None
        if run.get("status") == "FAIL":
            failed_phase = next((x for x in run.get("phase_order", []) if x not in completed), None)
            self.issue("optional_session" if failed_phase == "session" else "mandatory", name,
                       f"canonical driver failed at {failed_phase}: {run.get('failure')}")

    def merged(self, primary, phase):
        direct = self.evidence.get((primary, phase))
        if direct:
            return direct
        alternatives = [e for e in self.evidence.values() if e["primary"] == primary and e["phase"] == phase]
        return sorted(alternatives, key=lambda x: x["owner"])[0] if alternatives else None

    def comparisons(self):
        results = {}
        for phase in PHASES:
            if phase == "session":
                continue
            evidence = [e for e in self.evidence.values() if e["phase"] == phase]
            evidence.sort(key=lambda e: (not e["owner"].startswith("A"), e["owner"]))
            if not evidence:
                continue
            reference, comparisons = evidence[0], []
            for candidate in evidence[1:]:
                differences = []
                if len(candidate["rows"]) != len(reference["rows"]):
                    differences.append("row count")
                for index, (left, right) in enumerate(zip(reference["rows"], candidate["rows"])):
                    for field in ("prompt_md5", "request_md5", *TEXT_FIELDS):
                        if left.get(field) != right.get(field):
                            differences.append(f"row {index}: {field}")
                self.check(not differences, "mandatory", f"compare/{phase}/{reference['owner']}:{candidate['owner']}", differences)
                comparisons.append({"reference": reference["owner"], "candidate": candidate["owner"],
                                    "byte_equal_prompt_request_and_text": not differences, "differences": differences})
            results[phase] = {"participants": [e["owner"] for e in evidence], "comparisons": comparisons,
                "scope": "six timing cells" if phase == "battery" else "observed matching workload processes; not a three-process timing claim"}
            if phase == "battery":
                results[phase]["all_six_cells_present"] = all(name in results[phase]["participants"] for name in CELLS)
                results[phase]["five_prompts_byte_equal_across_all_six_cells"] = (
                    results[phase]["all_six_cells_present"]
                    and all(len(e["rows"]) == 5 for e in evidence)
                    and all(x["byte_equal_prompt_request_and_text"] for x in comparisons)
                )
        return results

    def workloads(self):
        result = {}
        for primary in ("A1", "B1"):
            phases = {phase: self.merged(primary, phase) for phase in ("warmup", "battery", "chain", "staged", "longdecode")}
            owners = {phase: value["owner"] for phase, value in phases.items() if value}
            item = {"phase_owners": owners, "all_mandatory_phases_same_process": len(set(owners.values())) == 1,
                    "scope": "single observation per arm; separate from three-fresh-process battery timing"}
            if len(set(owners.values())) > 1:
                item["process_continuity_note"] = "A new extra process supplies later workloads; this is not uninterrupted full-sequence evidence from the original timing server."
            chain = phases["chain"]
            if chain:
                reused = any(r.get("cached", 0) > 0 for r in chain["rows"][1:])
                item["chain_prefix_reuse_observed"] = reused
                if not reused:
                    self.coverage.append({"where": f"{primary}/chain", "unmet": "no later-turn cached prompt tokens observed"})
            staged = phases["staged"]
            if staged and len(staged["rows"]) >= 2:
                dirty, long = staged["rows"][:2]
                uncached = (long["ctx"] - long["cached"]
                    if number(long.get("ctx")) and number(long.get("cached")) else -1)
                item["staged"] = {"owner": staged["owner"], "ctx": long.get("ctx"), "cached": long.get("cached"),
                    "uncached_prompt_tokens": uncached, "minimum_long_prefill_tokens": 2048, "long_prefill_observed": uncached >= 2048,
                    "preceding_dirtying_request_gen": dirty.get("gen"), "long_request_gen": long.get("gen")}
                if uncached < 2048:
                    self.coverage.append({"where": f"{primary}/staged", "unmet": "fewer than 2048 uncached prompt tokens"})
                if primary.startswith("B"):
                    dirty_proof = staged["proofs"].get(dirty.get("request_id"), [])
                    long_proof = staged["proofs"].get(long.get("request_id"), [])
                    before = [x for x in staged["stats"] if dirty_proof and x["line"] <= dirty_proof[0]["line"]]
                    during = [x for x in staged["stats"] if dirty_proof and long_proof and dirty_proof[0]["line"] < x["line"] <= long_proof[0]["line"]]
                    baseline = before[-1] if before else None
                    restores = [x for x in during if baseline and x.get("prefill_restores", 0) > baseline.get("prefill_restores", 0)
                                and x.get("restore_bytes", 0) > baseline.get("restore_bytes", 0)]
                    restore = restores[0] if restores else None
                    item["staged"].update(cache_stats_before_long_request=baseline, restore_stats=restore,
                        dirty_tail_restore_observed=restore is not None)
                    self.check(restore is not None, "mandatory", f"{primary}/staged", "no added prefill restore/bytes between dirtying and long request proofs")
            longdecode = phases["longdecode"]
            if longdecode and longdecode["rows"]:
                row = longdecode["rows"][0]
                reached = number(row.get("gen")) and row["gen"] >= 1024
                item["longdecode"] = {"owner": longdecode["owner"], "target_generated_tokens": 1024,
                    "actual_generated_tokens": row.get("gen"), "finish": row.get("finish"), "reached_1024": reached,
                    "automatic_parameter_change_or_retry": False}
                if not reached:
                    self.coverage.append({"where": f"{primary}/longdecode", "unmet": "actual generation did not reach 1024",
                        "actual_gen": row.get("gen"), "finish": row.get("finish"), "note": "Natural stop is recorded; no automatic retry or parameter change."})
            result[primary] = item
        return result

    def timing(self):
        cases = {}
        for name in CELLS:
            evidence = self.evidence.get((name, "battery"))
            if not evidence or not evidence["confirmed_by_driver"] or len(evidence["rows"]) != 5:
                continue
            rows = evidence["rows"]
            if not all(number(r.get("decode_tok_s")) and r["decode_tok_s"] > 0 and number(r.get("wall_s")) and r["wall_s"] > 0
                       and number(r.get("prefill_ms")) and isinstance(r.get("gen"), int) and r["gen"] > 0 for r in rows):
                self.issue("mandatory", f"timing/{name}", "missing/non-finite/invalid rates, wall time or prefill time")
                continue
            total = sum(r["gen"] for r in rows)
            derived = sum(r["gen"] / r["decode_tok_s"] for r in rows)
            wall = sum(r["wall_s"] for r in rows)
            cases[name] = {"total_generated_tokens": total, "sum_gen_over_decode_rate_seconds": derived,
                "derived_decode_tok_s": total / derived, "sum_request_wall_seconds": wall,
                "wall_tok_s": total / wall, "sum_prefill_ms": sum(r["prefill_ms"] for r in rows),
                "estimated_rate_rows": [i for i, r in enumerate(rows) if r.get("decode_estimated")],
                "source": f"serve-{name}/battery.json", "row_checks_pass": evidence["row_checks_pass"]}
        arms = {}
        for arm in ("A", "B"):
            selected = {name: value for name, value in cases.items() if name.startswith(arm)}
            arms[arm] = {"n_fresh_processes": len(selected), "complete_three_processes": len(selected) == 3, "cells": list(selected)}
            for metric in ("derived_decode_tok_s", "wall_tok_s", "sum_prefill_ms"):
                values = [row[metric] for row in selected.values()]
                arms[arm][metric] = {"median": statistics.median(values), "min": min(values), "max": max(values)} if values else None
        deltas = {}
        if all(arms[x]["complete_three_processes"] for x in ("A", "B")):
            for metric in ("derived_decode_tok_s", "wall_tok_s", "sum_prefill_ms"):
                a, b = arms["A"][metric]["median"], arms["B"][metric]["median"]
                deltas[metric] = {"baseline_A": a, "candidate_B": b, "B_over_A_change_percent": (b / a - 1) * 100 if a else None}
        return {"scope": "five-prompt 512-cap battery only; A=baseline, B=cache; three fresh processes per arm",
            "precision_note": "Server decode_tok_s is rounded to 0.1 tok/s by harness input; derived seconds/rates are reconstructed, not raw device timing. wall_s is rounded to 0.001 s.",
            "formula": {"derived_decode_tok_s": "sum(gen) / sum(gen / decode_tok_s)", "wall_tok_s": "sum(gen) / sum(wall_s)", "prefill": "sum(prefill_ms)"},
            "cases": cases, "arms": arms, "median_changes": deltas}

    def finish(self):
        for directory in sorted(self.root.iterdir()):
            if directory.is_dir() and directory.name.startswith("serve-"):
                self.case(directory)
        for primary in CELLS:
            for phase in ("warmup", "battery"):
                evidence = self.evidence.get((primary, phase))
                if not evidence or not evidence["confirmed_by_driver"]:
                    self.pending.append({"where": f"{primary}/{phase}", "reason": "mandatory timing-cell phase not confirmed"})
        for primary in ("A1", "B1"):
            for phase in ("chain", "staged", "longdecode"):
                evidence = self.merged(primary, phase)
                if not evidence or not evidence["confirmed_by_driver"]:
                    self.pending.append({"where": f"{primary}/{phase}", "reason": "mandatory lifecycle phase not confirmed"})
        pid_owners = {}
        for name in CELLS:
            case = self.cases.get(name)
            if case:
                for pid in case["cli_pids_from_request_ids"]:
                    if pid in pid_owners:
                        self.issue("mandatory", name, f"timing cells share CLI process {pid} with {pid_owners[pid]}")
                    pid_owners[pid] = name
        comparisons = self.comparisons()
        workloads = self.workloads()
        timing = self.timing()
        matrix = self.read_json(self.root / "overall-matrix.json", "overall driver matrix")
        mandatory_errors = [x for x in self.errors if x["category"] != "optional_session"]
        optional_errors = [x for x in self.errors if x["category"] == "optional_session"]
        complete = not self.pending
        all_pass = complete and not self.errors and not self.coverage
        status = "PASS" if all_pass else "HAS_FAILURES" if self.errors else "INCOMPLETE_COVERAGE" if complete else "PARTIAL"
        if self.partial and not complete and self.errors:
            status = "PARTIAL_WITH_FAILURES"
        return {"schema": 1, "status": status, "all_pass": all_pass,
            "mandatory_checks_pass": complete and not mandatory_errors and not self.coverage,
            "optional_session_quality_pass": False if optional_errors else None,
            "matrix_evidence_complete": complete, "partial_requested": self.partial,
            "no_model_load_no_gpu_no_parameter_changes": True, "frozen_manifest": self.manifest,
            "mandatory_failures": mandatory_errors, "optional_session_failures": optional_errors,
            "coverage_limitations": self.coverage, "pending_evidence": self.pending, "notes": self.notes,
            "cases": self.cases, "same_prompt_request_text_comparisons": comparisons,
            "lifecycle_workloads": workloads, "battery_timing": timing, "driver_matrix_raw": matrix,
            "input_receipts": self.inputs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--partial", action="store_true", help="Inspect completed phases while remaining cells are running; never call pending work passed.")
    args = parser.parse_args()
    root = args.root.resolve()
    report = Audit(root, args.partial).finish()
    output = args.out or root / ("overall-receipts-partial.json" if args.partial else "overall-receipts.json")
    temporary = output.with_name(output.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(output)
    print(json.dumps({"status": report["status"], "all_pass": report["all_pass"],
        "mandatory_checks_pass": report["mandatory_checks_pass"], "optional_session_quality_pass": report["optional_session_quality_pass"],
        "mandatory_failures": len(report["mandatory_failures"]), "optional_failures": len(report["optional_session_failures"]),
        "pending": len(report["pending_evidence"]), "coverage_limitations": len(report["coverage_limitations"]), "report": str(output)}, ensure_ascii=False))
    if report["mandatory_failures"] or report["optional_session_failures"]:
        return 1
    if not report["matrix_evidence_complete"] or report["coverage_limitations"]:
        return 0 if args.partial else 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
