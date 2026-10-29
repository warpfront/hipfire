#!/usr/bin/env python3
"""Posthoc receipts only: A=old LRU, B=mapped-short-prefill LRU; both cache=1.

Never starts a server, loads a model, retries a request or changes parameters.
--partial reports pending evidence without promoting it to PASS.
Exit 1: observed failures; 2: missing evidence/coverage (unless --partial); 0: PASS
or an explicitly partial report with no observed failure. No optional session.
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
EXTENDED = ("warmup", "battery", "chain", "staged", "longdecode")
SHORT = ("warmup", "battery")
ROWS = {"warmup": 5, "battery": 5, "chain": 5, "staged": 2, "longdecode": 1}
CAP = {"warmup": 16, "battery": 512, "chain": 512, "staged": 128, "longdecode": 1024}
TEXT = ("content", "reasoning_content", "assistant_content", "tool_calls", "finish", "gen", "ctx")
MATCH = ("prompt_md5", "request_md5", *TEXT)
META = ("kv_mode", "kv_backend", "kv_backend_legacy", "kv_backend_reason", "kv_backend_warning")
PROOF = re.compile(r"HIPFIRE_REPLAY_ROUTE_PROOF transport=(\S+) position=(\d+) request_id=(\S+) replays=(\d+)(?:\s|$)")
STATE = re.compile(r"qwen4 state: QSA (\S+) K/V \((\S+) context storage\), GDN (\S+) recurrent")
STAGE = re.compile(r"qwen4 long-prefill expert staging: forwards of >= (\d+) rows")
REQUEST = re.compile(r"chatcmpl-(\d+)-(\d+)\Z")
MD5 = re.compile(r"[0-9a-f]{32}\Z")
MONOTONIC = ("tokens", "fills_started", "fills_completed", "prefill_restores", "restore_bytes")
FROZEN = {"HIPFIRE_QWEN4_EXPERT_CACHE": "1", "HIPFIRE_DN_STATE_EF": "1",
          "HIPFIRE_QWEN4_EXPERT_CACHE_TAIL_PERCENT": "50", "HIPFIRE_QWEN4_EXPERT_CACHE_INFLIGHT": "64",
          "HIPFIRE_QWEN4_EXPERT_HOT_BUDGET_MIB": "20000", "HIPFIRE_PREFILL_CHUNK_ROWS": "4096",
          "HIPFIRE_QWEN4_DECODE_SHARE": "off", "HIPFIRE_OFFLOAD_EXEC": "pcie",
          "HIPFIRE_MTP_MODE": "off", "HIPFIRE_PLE_FORMAT": "q4"}


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def integer(x):
    return isinstance(x, int) and not isinstance(x, bool)


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def observed(row, fields):
    return {k: row[k] if row.get(k) is not None else "UNOBSERVED" for k in fields}


def argument(args, flag):
    if not isinstance(args, list) or args.count(flag) != 1:
        return None
    i = args.index(flag)
    return args[i + 1] if i + 1 < len(args) else None


class Audit:
    def __init__(self, root, partial=False):
        self.root, self.partial = root, partial
        self.errors, self.pending, self.coverage, self.notes = [], [], [], []
        self.inputs, self.cases, self.phases = {}, {}, {}
        self.baseline = self.read_json(root / "BASELINE.json", "baseline receipt")
        self.manifests = {v: self.read_json(root / ("runtime-" + v) / "MANIFEST.json", v + " manifest")
                          for v in ("old", "new")}

    def check(self, ok, where, detail):
        if not ok:
            self.errors.append({"where": where, "detail": detail})
        return bool(ok)

    def read_bytes(self, path):
        data = path.read_bytes()
        self.inputs[str(path.relative_to(self.root))] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        return data

    def read_json(self, path, where, settled=False):
        try:
            return json.loads(self.read_bytes(path))
        except (OSError, ValueError) as e:
            record = {"where": where, "reason": f"unavailable/unsettled JSON: {e}"}
            (self.errors if settled else self.pending).append(record)
            return None

    def case(self, name):
        d = self.root / ("serve-" + name)
        run = self.read_json(d / "run.json", name)
        if not isinstance(run, dict):
            return
        variant = "old" if name.startswith("A") else "new"
        expected = EXTENDED if name in ("A1", "B1") else SHORT
        settled = run.get("status") in ("PASS", "FAIL")
        self.check(run.get("name") == name and run.get("variant") == variant, name, "case/variant identity mismatch")
        self.check(str(run.get("cache")) == "1", name, "both arms must use cache=1")
        self.check(run.get("phase_order") == list(expected), name, "phase order differs from the authorized same-process sequence")
        manifest = self.manifests[variant]
        if manifest is not None:
            self.check(run.get("frozen_manifest") == manifest, name, "run manifest differs from runtime-" + variant + "/MANIFEST.json")
        env = run.get("env") or {}
        for k, v in FROZEN.items():
            self.check(env.get(k) == v, name, f"resolved experiment setting {k}={env.get(k)!r}, expected {v!r}")
        for key, filename in (("HIPFIRE_CLI_BIN", "hipfire"), ("HIPFIRE_DAEMON_BIN", "daemon_lru_" + variant)):
            path = self.root / ("runtime-" + variant) / filename
            self.check(env.get(key) == str(path), name, f"{key} does not identify this arm's frozen runtime")
            if isinstance(manifest, dict):
                entry = (manifest.get("binaries") or {}).get(filename)
                self.check(isinstance(entry, dict) and re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256", ""))) is not None,
                           name, f"no declared binary SHA-256 for {filename}")
        try:
            text = self.read_bytes(d / "serve.log").decode(errors="replace")
        except OSError as e:
            (self.errors if settled else self.pending).append({"where": name, "reason": str(e)})
            return
        proofs, states, stats, stage_markers = {}, [], [], []
        for line_no, line in enumerate(text.splitlines(), 1):
            m = PROOF.search(line)
            if m:
                transport, position, request_id, count = m.groups()
                proofs.setdefault(request_id, []).append({"line": line_no, "transport": transport,
                    "position": int(position), "replays": int(count), "raw": line})
            m = STATE.search(line)
            if m:
                states.append({"line": line_no, "qsa": m[1], "backend": m[2], "gdn": m[3], "raw": line})
            m = STAGE.search(line)
            if m:
                stage_markers.append({"line": line_no, "min_rows": int(m[1]), "raw": line})
            marker = "qwen4 exact expert cache stats: "
            if marker in line:
                try:
                    value = json.loads(line.split(marker, 1)[1])
                    if not isinstance(value, dict):
                        raise ValueError("stats are not an object")
                    stats.append(dict(value, log_line=line_no))
                except ValueError as e:
                    if settled:
                        self.check(False, name, f"malformed cache stats at line {line_no}: {e}")
                    else:
                        self.pending.append({"where": name, "reason": f"unsettled stats at line {line_no}"})
        completed = [r.get("phase") for r in run.get("phase_results", []) if r.get("status") == "PASS"]
        self.check(len(completed) == len(set(completed)), name, "duplicate phase completion receipts")
        case = {"variant": variant, "effective_cache": "1", "driver_status": run.get("status", "IN_PROGRESS"),
                "driver_failure": run.get("failure"), "phase_order": run.get("phase_order"), "phases": {},
                "observed_backend_receipts_raw": run.get("observed_backend_receipts", "UNOBSERVED"),
                "stats": stats, "startup_state_markers": states, "stage_policy_markers": stage_markers,
                "binary_hash_recomputed": False, "binary_identity_scope": "declared MANIFEST SHA plus exact run manifest/path equality"}
        self.cases[name] = case
        request_ids, pids, proof_lines = set(), set(), []
        for phase in expected:
            path = d / (phase + ".json")
            if not path.exists() and not settled:
                self.pending.append({"where": name + "/" + phase, "reason": "phase not yet available"})
                continue
            rows = self.read_json(path, name + "/" + phase, settled=settled)
            if not isinstance(rows, list):
                if rows is not None:
                    self.check(False, name + "/" + phase, "harness rows must be a list")
                continue
            where = name + "/" + phase
            before = len(self.errors)
            self.check(len(rows) == ROWS[phase], where, f"expected {ROWS[phase]} rows; got {len(rows)}")
            row_summaries = []
            for i, row in enumerate(rows):
                w = f"{where}[{i}]"
                if not isinstance(row, dict):
                    self.check(False, w, "row is not an object")
                    continue
                rid = row.get("request_id")
                match = REQUEST.fullmatch(rid) if isinstance(rid, str) else None
                self.check(match is not None, w, "missing/malformed request ID")
                if match:
                    pids.add(match[1])
                    self.check(rid not in request_ids, w, "duplicate request ID in one process")
                    request_ids.add(rid)
                for key in ("prompt_md5", "request_md5"):
                    self.check(isinstance(row.get(key), str) and MD5.fullmatch(row[key]) is not None, w, f"invalid {key}")
                for key in ("attractor", "empty", "atem_leak"):
                    self.check(row.get(key) is False, w, f"{key} is not explicitly false")
                self.check(row.get("finish") in ("stop", "length", "eos", "tool_calls"), w, "invalid finish")
                self.check(row.get("stream_error") is None, w, "stream error")
                self.check(row.get("terminal_count") == 1, w, "terminal_count != 1")
                self.check(row.get("saw_done") is True, w, "missing SSE [DONE]")
                self.check(row.get("post_terminal_bytes") == 0, w, "payload bytes after terminal")
                self.check(not row.get("retrieval_missing"), w, "retrieval strings missing")
                valid_counts = all(integer(row.get(k)) for k in ("gen", "ctx", "cached"))
                self.check(valid_counts and row["gen"] > 0 and row["ctx"] > 0 and 0 <= row["cached"] <= row["ctx"], w, "invalid token/context/cache counts")
                self.check(not valid_counts or row["gen"] <= CAP[phase], w, "generated tokens exceed requested cap")
                joined = proofs.get(rid, [])
                if len(joined) != 1 and not settled and not joined:
                    self.pending.append({"where": w, "reason": "request-scoped replay proof not yet observed"})
                else:
                    self.check(len(joined) == 1, w, f"expected exactly one PM4 proof; got {len(joined)}")
                if len(joined) == 1:
                    self.check(joined[0]["transport"] in ("pm4", "pm4_ib") and joined[0]["replays"] > 0, w, "no positive PM4 replay proof")
                    proof_lines.append(joined[0]["line"])
                row_summaries.append({"row": i, "request_id": rid, "prompt_md5": row.get("prompt_md5"),
                    "request_md5": row.get("request_md5"), "text_projection_sha256": digest({k: row.get(k) for k in TEXT}),
                    "gen": row.get("gen"), "ctx": row.get("ctx"), "cached": row.get("cached"),
                    "uncached_prompt_tokens": row["ctx"] - row["cached"] if valid_counts else None,
                    "finish": row.get("finish"), "runaway_harness_flag": row.get("runaway", "UNOBSERVED"),
                    "pm4_proof": joined, "harness_backend_fields": observed(row, META)})
            phase_receipt = next((r for r in run.get("phase_results", []) if r.get("phase") == phase), None)
            confirmed = phase in completed
            if not confirmed:
                self.pending.append({"where": where, "reason": "driver has not confirmed phase completion"})
            if phase_receipt:
                args = phase_receipt.get("arguments")
                for key, value in {"--max-tokens": str(CAP[phase]), "--max-seq": "16384", "--sampling": "greedy",
                                   "--speculation": "off", "--thinking": "off", "--seed": "123"}.items():
                    self.check(argument(args, key) == value, where, f"request setting {key} mismatch")
            entry = {"rows": rows, "summary": row_summaries, "confirmed": confirmed,
                     "checks_pass": len(self.errors) == before, "proofs": proofs}
            self.phases[(name, phase)] = entry
            case["phases"][phase] = {k: entry[k] for k in ("summary", "confirmed", "checks_pass")}
        self.check(len(pids) == 1, name, f"expected one serving PID in request IDs, observed {sorted(pids)}")
        case["request_id_pids"] = sorted(pids)
        if proof_lines:
            first = min(proof_lines)
            before = [s for s in states if s["line"] < first]
            active = before[-1] if before else None
            case["raw_active_startup_state_receipt"] = active
            self.check(active is not None and (active["qsa"], active["backend"], active["gdn"]) == ("fp8", "vmm", "q8"),
                       name, "no raw active startup proof of FP8/VMM/plain Q8")
            self.check(not any(s["line"] >= first for s in states), name, "model state reloaded during the workload sequence")
            self.check(proof_lines == sorted(proof_lines), name, "request proofs are out of serial phase order")
        if not stats:
            (self.errors if settled else self.pending).append({"where": name, "reason": "cache execution stats not observed"})
        count_keys = MONOTONIC + (("mapped_prefill_calls", "mapped_prefill_rows") if variant == "new" else ())
        for s in stats:
            for key in count_keys:
                self.check(integer(s.get(key)) and s[key] >= 0, name, f"missing/invalid cache counter {key}")
        for old, new in zip(stats, stats[1:]):
            for key in count_keys:
                if integer(new.get(key)) and integer(old.get(key)):
                    self.check(new[key] >= old[key], name, f"counter {key} decreased")
        if variant == "new" and stats:
            self.check(all(integer(s.get("mapped_prefill_rows")) and integer(s.get("mapped_prefill_calls"))
                           and s["mapped_prefill_rows"] >= s["mapped_prefill_calls"] for s in stats), name, "mapped rows < admitted calls")
            if settled:
                self.check(max((s["mapped_prefill_calls"] for s in stats if integer(s.get("mapped_prefill_calls"))), default=0) > 0
                           and max((s["mapped_prefill_rows"] for s in stats if integer(s.get("mapped_prefill_rows"))), default=0) > 0,
                           name, "no mapped-prefill admission counter increase")
        if settled and run.get("status") == "FAIL":
            self.check(False, name, "driver failed: " + str(run.get("failure")))
        if not settled:
            self.pending.append({"where": name, "reason": "case process/cleanup not yet complete"})

    def compare(self):
        result = {}
        for phase in EXTENDED:
            names = [n for n in CELLS if (n, phase) in self.phases]
            if not names:
                continue
            reference = "A1" if "A1" in names else names[0]
            left = self.phases[(reference, phase)]["rows"]
            comparisons = []
            for name in names:
                if name == reference:
                    continue
                right = self.phases[(name, phase)]["rows"]
                differences = ["row count"] if len(left) != len(right) else []
                for i, (a, b) in enumerate(zip(left, right)):
                    if not isinstance(a, dict) or not isinstance(b, dict):
                        differences.append(f"row {i}: not an object")
                        continue
                    differences += [f"row {i}: {k}" for k in MATCH if a.get(k) != b.get(k)]
                self.check(not differences, f"compare/{phase}/{reference}:{name}", differences)
                comparisons.append({"reference": reference, "candidate": name, "utf8_transcript_and_reported_hashes_equal": not differences,
                                    "differences": differences})
            result[phase] = {"participants": names, "comparisons": comparisons,
                             "scope": "harness UTF-8 text/structured transcript equality; not direct token-ID or logit parity"}
        return result

    def lifecycle(self):
        result = {}
        expected_bytes = self.baseline.get("measured_restore_bytes_per_request") if isinstance(self.baseline, dict) else None
        for name in ("A1", "B1"):
            case = self.cases.get(name)
            if not case:
                continue
            stats = case["stats"]
            item = {"all_extended_phases_same_case_process": all((name, p) in self.phases for p in EXTENDED)
                    and len(case.get("request_id_pids", [])) == 1}
            chain = self.phases.get((name, "chain"))
            if chain:
                reused = any(integer(r.get("cached")) and r["cached"] > 0 for r in chain["rows"][1:])
                item["chain_cached_prefix_observed"] = reused
                if chain["confirmed"] and not reused:
                    self.coverage.append({"where": name + "/chain", "unmet": "no later-turn prompt prefix reuse"})
            staged = self.phases.get((name, "staged"))
            if staged and len(staged["rows"]) == 2:
                dirty, long = staged["rows"]
                dp = staged["proofs"].get(dirty.get("request_id"), [])
                lp = staged["proofs"].get(long.get("request_id"), [])
                uncached = long["ctx"] - long["cached"] if integer(long.get("ctx")) and integer(long.get("cached")) else None
                thresholds = case["stage_policy_markers"]
                threshold = thresholds[-1]["min_rows"] if thresholds else None
                item["staged"] = {"dirty_request_gen": dirty.get("gen"), "long_ctx": long.get("ctx"), "cached": long.get("cached"),
                                  "uncached_tokens": uncached, "requested_uncached_target": 5103, "logged_stage_threshold": threshold,
                                  "stage_execution_counter": "UNOBSERVED"}
                if staged["confirmed"] and (uncached != 5103 or threshold is None or uncached < threshold):
                    self.coverage.append({"where": name + "/staged", "unmet": "5103 uncached rows and logged stage eligibility not jointly observed"})
                if staged["confirmed"] and dirty.get("gen") != 128:
                    self.coverage.append({"where": name + "/staged", "unmet": "dirtying request did not reach 128 tokens; no automatic retry"})
                if len(dp) == len(lp) == 1:
                    before = [s for s in stats if s["log_line"] <= dp[0]["line"]]
                    during = [s for s in stats if dp[0]["line"] < s["log_line"] <= lp[0]["line"]]
                    base, end = (before[-1] if before else None), (during[-1] if during else None)
                    delta = ({k: end[k] - base[k] for k in ("prefill_restores", "restore_bytes")}
                             if base and end and all(integer(z.get(k)) for z in (base, end)
                                                     for k in ("prefill_restores", "restore_bytes")) else None)
                    item["staged"].update(stats_before_long=base, stats_during_long=during, delta=delta,
                        expected_restore_bytes=expected_bytes)
                    if staged["confirmed"]:
                        self.check(integer(expected_bytes) and expected_bytes > 0, name + "/staged", "no exact baseline donor byte receipt")
                        self.check(delta is not None and delta["prefill_restores"] == 1 and delta["restore_bytes"] == expected_bytes,
                                   name + "/staged", "long prefill did not restore exactly once with the canonical donor byte count")
            ld = self.phases.get((name, "longdecode"))
            if ld and ld["rows"]:
                row = ld["rows"][0]
                item["longdecode"] = {"actual_gen": row.get("gen"), "finish": row.get("finish"), "target": 1024,
                                      "parameter_changes_or_retry_for_natural_stop": False}
                if ld["confirmed"] and row.get("gen") != 1024:
                    self.coverage.append({"where": name + "/longdecode", "unmet": "1024 tokens not reached; natural stop retained without retry"})
            result[name] = item
        for name, case in self.cases.items():
            if case["variant"] != "new":
                continue
            stats = case["stats"]
            if name != "B1":
                self.check(all(s.get("prefill_restores") == 0 and s.get("restore_bytes") == 0 for s in stats),
                           name, "bulk donor restore observed in a short-only case")
            else:
                short_proofs = []
                for phase in ("warmup", "battery", "chain"):
                    for row in self.phases.get((name, phase), {}).get("summary", []):
                        short_proofs += [p["line"] for p in row["pm4_proof"]]
                staged = self.phases.get((name, "staged"))
                if staged and staged["summary"]:
                    short_proofs += [p["line"] for p in staged["summary"][0]["pm4_proof"]]
                cutoff = max(short_proofs) if short_proofs else 0
                observed_short = [s for s in stats if s["log_line"] <= cutoff]
                self.check(all(s.get("prefill_restores") == 0 and s.get("restore_bytes") == 0 for s in observed_short),
                           name, "bulk donor restore observed before the long staged request")
                if case["driver_status"] == "PASS" and stats:
                    self.check(max((s["prefill_restores"] for s in stats if integer(s.get("prefill_restores"))), default=-1) == 1
                               and max((s["restore_bytes"] for s in stats if integer(s.get("restore_bytes"))), default=-1) == expected_bytes,
                               name, "new arm total restores differ from its one authorized long prefill")
        return result

    def timing(self):
        cases = {}
        for name in CELLS:
            e = self.phases.get((name, "battery"))
            if not e or not e["confirmed"] or len(e["rows"]) != 5:
                continue
            rows = e["rows"]
            good = all(isinstance(r, dict) and integer(r.get("gen")) and r["gen"] > 0 and finite(r.get("decode_tok_s")) and r["decode_tok_s"] > 0
                       and finite(r.get("wall_s")) and r["wall_s"] > 0 and finite(r.get("prefill_ms")) and r["prefill_ms"] > 0
                       and integer(r.get("ctx")) and integer(r.get("cached")) for r in rows)
            if not self.check(good, "timing/" + name, "invalid battery counters/timing"):
                continue
            generated = sum(r["gen"] for r in rows)
            uncached = sum(r["ctx"] - r["cached"] for r in rows)
            ds = sum(r["gen"] / r["decode_tok_s"] for r in rows)
            wall = sum(r["wall_s"] for r in rows)
            pp_ms = sum(r["prefill_ms"] for r in rows)
            cases[name] = {"generated_tokens": generated, "uncached_prompt_tokens": uncached,
                "derived_decode_tok_s": generated / ds, "prefill_tok_s": uncached / (pp_ms / 1000),
                "sum_prefill_ms": pp_ms, "sum_request_wall_s": wall, "wall_tok_s": generated / wall,
                "decode_estimated_rows": [i for i,r in enumerate(rows) if r.get("decode_estimated") is not False],
                "source": "serve-" + name + "/battery.json"}
            self.check(not cases[name]["decode_estimated_rows"], "timing/" + name, "decode rate not explicitly measured")
        arms = {}
        metrics = ("derived_decode_tok_s", "prefill_tok_s", "sum_prefill_ms", "sum_request_wall_s", "wall_tok_s")
        for arm in ("A", "B"):
            names = [n for n in cases if n.startswith(arm)]
            arms[arm] = {"variant": "old LRU" if arm == "A" else "mapped-short-prefill LRU", "n_fresh_processes": len(names), "cells": names}
            for m in metrics:
                v = [cases[n][m] for n in names]
                arms[arm][m] = {"median": statistics.median(v), "min": min(v), "max": max(v)} if v else None
        delta = {}
        if all(arms[a]["n_fresh_processes"] == 3 for a in ("A", "B")):
            for m in metrics:
                a, b = arms["A"][m]["median"], arms["B"][m]["median"]
                delta[m] = {"old_A": a, "new_B": b, "B_over_A_change_percent": (b / a - 1) * 100 if a else None}
        return {"scope": "five-prompt 512-cap battery after same-process warm16; both arms exact cache enabled",
                "precision_note": "decode rates rounded to 0.1 tok/s and wall to 0.001s by harness; derived decode time is not raw GPU timing",
                "formulas": {"decode_tok_s": "sum(gen)/sum(gen/reported_decode_tok_s)", "prefill_tok_s": "sum(ctx-cached)/sum(prefill_ms/1000)",
                             "wall_tok_s": "sum(gen)/sum(wall_s)"},
                "cases": cases, "arms": arms, "three_process_median_changes": delta}

    def finish(self):
        for name in CELLS:
            self.case(name)
        pids = {}
        for name, case in self.cases.items():
            for pid in case.get("request_id_pids", []):
                self.check(pid not in pids, name, f"serving PID {pid} reused by {pids.get(pid)}")
                pids[pid] = name
        comparisons = self.compare()
        lifecycle = self.lifecycle()
        timing = self.timing()
        matrix = self.read_json(self.root / "overall-matrix.json", "driver matrix")
        if isinstance(matrix, list):
            self.check([r.get("case") for r in matrix] == list(CELLS[:len(matrix)]),
                       "driver matrix", "matrix is not a unique prefix of fixed ABBAAB order")
            for row in matrix:
                self.check(row.get("case") in CELLS and row.get("exit_code") == 0, "driver matrix", row)
            for name in CELLS:
                if not any(r.get("case") == name for r in matrix):
                    self.pending.append({"where": name, "reason": "driver matrix cell not completed"})
        for name in CELLS:
            for phase in (EXTENDED if name in ("A1", "B1") else SHORT):
                if not self.phases.get((name, phase), {}).get("confirmed"):
                    self.pending.append({"where": name + "/" + phase, "reason": "required phase not confirmed"})
        first_pair = None
        if ("A1", "battery") in self.phases and ("B1", "battery") in self.phases:
            a, b = self.phases[("A1", "battery")]["rows"], self.phases[("B1", "battery")]["rows"]
            if len(a) == len(b) == 5 and all(isinstance(r, dict) and finite(r.get("prefill_ms")) and r["prefill_ms"] > 0
                                             and finite(r.get("wall_s")) and r["wall_s"] > 0 for r in a+b):
                first_pair = {"prefill_time_new_over_old": sum(r["prefill_ms"] for r in b)/sum(r["prefill_ms"] for r in a),
                              "request_wall_new_over_old": sum(r["wall_s"] for r in b)/sum(r["wall_s"] for r in a)}
                first_pair["driver_sanity_thresholds_pass"] = first_pair["prefill_time_new_over_old"] <= .90 and first_pair["request_wall_new_over_old"] <= 1.02
                self.check(first_pair["driver_sanity_thresholds_pass"], "first-pair driver sanity gate", first_pair)
        complete = not self.pending
        all_pass = complete and not self.errors and not self.coverage
        status = "PASS" if all_pass else "HAS_FAILURES" if self.errors else "INCOMPLETE_COVERAGE" if complete else "PARTIAL"
        if self.errors and not complete:
            status = "PARTIAL_WITH_FAILURES"
        return {"schema": "hipfire.prefill-comparison.receipts/v1", "status": status, "all_pass": all_pass,
            "partial_requested": self.partial, "matrix_evidence_complete": complete,
            "mandatory_checks_pass": complete and not self.errors and not self.coverage,
            "no_model_load_gpu_retry_or_parameter_changes_by_this_verifier": True,
            "arm_identity": {"A": "old LRU, cache=1", "B": "mapped-short-prefill LRU, cache=1"},
            "manifests": self.manifests, "baseline_receipt": self.baseline,
            "failures": self.errors, "pending_evidence": self.pending, "coverage_limitations": self.coverage,
            "notes": self.notes + ["No optional session required or merged from other processes.",
                "Matrix cache=0/1 values are worker variant selectors; effective feature flag is checked from each run.env.",
                "Report preserves raw FP8/VMM/Q8 startup proof separately from unobserved harness backend fields.",
                "runaway length-cap flags are preserved; valid length stops are not silently classified as attractors.",
                "CPU/negative lifecycle tests and Redline state parity are separate evidence, not established by text comparison here."],
            "cases": self.cases, "transcript_comparisons": comparisons, "lifecycle": lifecycle,
            "battery_timing": timing, "first_pair_sanity": first_pair, "driver_matrix_raw": matrix, "input_receipts": self.inputs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    report = Audit(root, args.partial).finish()
    out = args.out or root / ("prefill-comparison-partial.json" if args.partial else "prefill-comparison-receipts.json")
    out = out.resolve()
    if not out.is_relative_to(root):
        parser.error("--out must stay inside this QC comparison directory")
    temporary = out.with_name(out.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(out)
    print(json.dumps({"status": report["status"], "all_pass": report["all_pass"], "failures": len(report["failures"]),
        "pending": len(report["pending_evidence"]), "coverage": len(report["coverage_limitations"]), "report": str(out)}, ensure_ascii=False))
    if report["failures"]:
        return 1
    if not report["matrix_evidence_complete"] or report["coverage_limitations"]:
        return 0 if args.partial else 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
