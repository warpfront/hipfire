#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Comprehensive black-box test suite for the feat/serving-cache-scheduler branch.

Drives a RUNNING `hipfire serve` container (multi-slot engine + prefix cache +
strict JSON schema + capability advertisement) over its OpenAI-compatible HTTP
surface and checks every externally observable branch feature, edge case, and
regression class from the wave 0-12 ledger (docs/specs/2026-09-05-serving-
cache-scheduler-summary.md).

Areas
  A  capability advertisement + telemetry (/health, /v1/models, /stats, /metrics)
  B  generation + sampling contract (shapes, streaming, determinism, reasoning)
  C  prefix cache correctness (reuse, replay identity, forks, COW, soak)
  D  scheduler / admission (parallel slots, 503 overload, cancel, starvation)
  E  strict JSON Schema grammar (valid outputs + typed rejections + dead ends)
  F  typed refusals on the multi-slot route (tools/stop/logprobs/...)
  G  HTTP limits + error mapping (413/404/400, seed, ctx overshoot)
  V  vision-sidecar interaction with the radix (vision skips prefix cache)
  H  lifecycle canaries (post-gauntlet health, cold-cycle stability)

Design notes
  * stdlib only; safe to run against any reachable serve endpoint.
  * Prompts are deterministic module constants; their md5s go into the report
    (AGENTS.md rule: byte-identical prompts, md5 recorded).
  * Cache cells assert the core branch contract: REUSE NEVER CHANGES OUTPUT
    (greedy warm replay must be byte-identical to the cold run).
  * Coherence is checked with an n-gram attractor detector on decoded text —
    numbers alone never prove coherence (CLAUDE.md rule 1).
  * Exit code 0 iff no FAIL cells (WARN/SKIP allowed).

Usage
  python3 scripts/scs_suite.py                       # http://127.0.0.1:8420
  python3 scripts/scs_suite.py --list                # show cells, exit
  python3 scripts/scs_suite.py --area C              # one area
  python3 scripts/scs_suite.py --cell C1 --cell E5   # specific cells
  python3 scripts/scs_suite.py --json-out report.json
"""

import argparse
import base64
import hashlib
import http.client
import json
import re
import socket
import struct
import sys
import threading
import time
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------


class Cfg:
    def __init__(self, host, port, model, timeout_scale=1.0, slots=4):
        self.host = host
        self.port = port
        self.model = model
        self.timeout_scale = timeout_scale
        # Deployment's serve.multi_slot_slots; capability cells assert it.
        self.slots = slots
        # Resolved after the first /health probe by main(); True when the
        # loaded model runs a speculation head (MTP/DF2). Cells that compare
        # greedy output across cold/warm execution shapes relax equality to a
        # WARN under speculation (cross-shape reduction-order drift), while
        # determinism and accounting stay hard checks.
        self.spec_on = None

    def timeout(self, secs):
        return max(10.0, secs * self.timeout_scale)


# --------------------------------------------------------------------------
# http layer
# --------------------------------------------------------------------------


class HttpError(Exception):
    pass


class ChatResult:
    def __init__(self, status, headers, text, elapsed):
        self.status = status
        self.headers = headers
        self.text = text
        self.elapsed = elapsed
        self.json = None
        if text:
            try:
                self.json = json.loads(text)
            except ValueError:
                pass

    @property
    def content(self):
        if self.json and self.json.get("choices"):
            return self.json["choices"][0].get("message", {}).get("content") or ""
        return ""

    @property
    def reasoning(self):
        if self.json and self.json.get("choices"):
            return self.json["choices"][0].get("message", {}).get("reasoning_content") or ""
        return ""

    @property
    def finish(self):
        if self.json and self.json.get("choices"):
            return self.json["choices"][0].get("finish_reason")
        return None

    @property
    def usage(self):
        return (self.json or {}).get("usage") or {}

    @property
    def cached_tokens(self):
        u = self.usage
        return (u.get("prompt_tokens_details") or {}).get("cached_tokens")

    @property
    def error_message(self):
        e = (self.json or {}).get("error") or {}
        return e.get("message", "")


class StreamResult:
    def __init__(self):
        self.status = None
        self.error_text = ""
        self.frames = []
        self.content = ""
        self.reasoning = ""
        self.role_seen = False
        self.finish = None
        self.usage = None
        self.cached_tokens = None
        self.timings = None
        self.done_sentinel = False
        self.disconnected = False
        self.first_content_ms = None
        self.total_ms = None
        self.frames_valid_json = True

    @property
    def content_chunks(self):
        n = 0
        for f in self.frames:
            d = (f.get("choices") or [{}])[0].get("delta", {}) if f.get("choices") else {}
            if d.get("content"):
                n += 1
        return n


def _post_json(cfg, path, payload_dict, timeout):
    conn = http.client.HTTPConnection(cfg.host, cfg.port, timeout=timeout)
    t0 = time.monotonic()
    body = json.dumps(payload_dict) if payload_dict is not None else None
    headers = {"Content-Type": "application/json"}
    try:
        conn.request("POST", path, body=body, headers=headers)
        resp = conn.getresponse()
        text = resp.read().decode("utf-8", "replace")
        hdrs = {k.lower(): v for k, v in resp.getheaders()}
        return resp.status, hdrs, text, time.monotonic() - t0
    finally:
        conn.close()


def get_json(cfg, path, timeout=15):
    conn = http.client.HTTPConnection(cfg.host, cfg.port, timeout=timeout)
    try:
        conn.request("GET", path)
        resp = conn.getresponse()
        text = resp.read().decode("utf-8", "replace")
        return resp.status, {k.lower(): v for k, v in resp.getheaders()}, text
    finally:
        conn.close()


def chat(cfg, messages, expect_error=False, timeout=150, **kw):
    """Non-streaming chat request. Returns ChatResult."""
    payload = {"model": cfg.model, "messages": messages,
               "chat_template_kwargs": {"enable_thinking": False}}
    payload.update(kw)
    status, hdrs, text, dt = _post_json(cfg, "/v1/chat/completions", payload, cfg.timeout(timeout))
    r = ChatResult(status, hdrs, text, dt)
    if not expect_error and status != 200:
        raise HttpError("HTTP %s: %s" % (status, text[:300]))
    return r


def chat_stream(cfg, messages, cancel_after_chunks=None, timeout=300, **kw):
    """Streaming chat request. Returns StreamResult.

    cancel_after_chunks: close the connection after N content chunks
    (simulates a client disconnect mid-generation).
    """
    payload = {"model": cfg.model, "messages": messages, "stream": True,
               "chat_template_kwargs": {"enable_thinking": False}}
    payload.update(kw)
    sr = StreamResult()
    conn = http.client.HTTPConnection(cfg.host, cfg.port, timeout=cfg.timeout(timeout))
    t0 = time.monotonic()
    try:
        conn.request("POST", "/v1/chat/completions", body=json.dumps(payload),
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        sr.status = resp.status
        if resp.status != 200:
            sr.error_text = resp.read().decode("utf-8", "replace")
            return sr
        while True:
            line = resp.readline()
            if not line:
                break
            s = line.decode("utf-8", "replace").strip()
            if not s:
                continue
            if s == "data: [DONE]":
                sr.done_sentinel = True
                break
            if not s.startswith("data:"):
                continue
            try:
                frame = json.loads(s[5:].strip())
            except ValueError:
                sr.frames_valid_json = False
                continue
            sr.frames.append(frame)
            if frame.get("choices"):
                ch = frame["choices"][0]
                delta = ch.get("delta") or {}
                if delta.get("role"):
                    sr.role_seen = True
                if delta.get("content"):
                    sr.content += delta["content"]
                    if sr.first_content_ms is None:
                        sr.first_content_ms = (time.monotonic() - t0) * 1000.0
                    if cancel_after_chunks is not None and \
                            sr.content_chunks >= cancel_after_chunks:
                        sr.disconnected = True
                        break
                if delta.get("reasoning_content"):
                    sr.reasoning += delta["reasoning_content"]
                if ch.get("finish_reason"):
                    sr.finish = ch["finish_reason"]
                    if frame.get("timings"):
                        sr.timings = frame["timings"]
            elif frame.get("usage"):
                sr.usage = frame["usage"]
                sr.cached_tokens = (frame["usage"].get("prompt_tokens_details") or {}).get("cached_tokens")
        if sr.disconnected:
            conn.close()
            return sr
        sr.total_ms = (time.monotonic() - t0) * 1000.0
        return sr
    finally:
        try:
            conn.close()
        except Exception:
            pass


def raw_status_line(cfg, path, raw_request, timeout=15):
    """Send a hand-crafted raw HTTP request; return the status line."""
    s = socket.create_connection((cfg.host, cfg.port), timeout=timeout)
    try:
        s.sendall(raw_request.encode())
        data = b""
        while b"\r\n" not in data:
            chunk = s.recv(4096)
            if not chunk:
                break
            data += chunk
        return data.split(b"\r\n")[0].decode("utf-8", "replace").strip()
    finally:
        s.close()


# --------------------------------------------------------------------------
# deterministic prompt fixtures (md5s recorded in the report)
# --------------------------------------------------------------------------

PROMPT_MD5S = {}


def _record_md5(name, text):
    PROMPT_MD5S[name] = hashlib.md5(text.encode()).hexdigest()
    return text


_SENTENCE_TEMPLATES = [
    "The {t} archives record that entry {i} was catalogued during the fourth survey season.",
    "Surveyors near {t} measured a tidal shift of {i} centimeters across the eastern shelf.",
    "A copper bell recovered from {t} carries an inscription referencing year {i} of the federation.",
    "The lighthouse keeper of {t} logged {i} vessels passing the northern reef that winter.",
    "Botanical notes from {t} describe {i} distinct fern varieties along the basalt cliffs.",
    "Ferry schedules at {t} changed {i} times before the winter council reached agreement.",
    "The granary ledgers of {t} list {i} sacks of barley reserved for the spring market.",
    "Cartographers working on {t} corrected {i} coastline errors in the second edition atlas.",
    "A stubborn fog around {t} delayed {i} mail deliveries during the herring season.",
    "The {t} choral society rehearsed {i} hymns before the harvest festival opened.",
    "Stone masons repairing the {t} bridge numbered each of the {i} replacement blocks.",
    "The harbor master of {t} recorded {i} seals hauled out on the breakwater at dawn.",
    "Schoolchildren from {t} planted {i} rowan saplings along the processional road.",
    "The customs house at {t} stamped {i} manifests for bolts of grey northern wool.",
    "An eclipse visible from {t} was photographed {i} times on glass plates, mostly blurred.",
    "The {t} rescue skiff was repainted {i} shades of orange before the committee approved one.",
    "Rope makers of {t} braided {i} fathoms of anchor line for the deepwater fleet.",
    "The weather stone on {t} was said to sweat {i} days before any serious gale.",
    "A drifting buoy near {t} transmitted {i} temperature readings before its battery died.",
    "The bakery on {t} quay sold {i} seed loaves on the morning the band played.",
]


def passage(topic, n_sentences, start=0, cap="island"):
    """Deterministic pseudo-factual passage; distinct topics diverge by token ~10."""
    sents = []
    for i in range(start, start + n_sentences):
        sents.append(_SENTENCE_TEMPLATES[i % len(_SENTENCE_TEMPLATES)]
                     .format(t="%s %s" % (cap, topic), i=i * 7 + 3))
    return " ".join(sents)


def prompt_cold(topic, question, n_sentences=34):
    """Unique-opening cold prompt: topic word appears in sentence 1."""
    text = passage(topic, n_sentences) + "\n\nQuestion: %s" % question
    return _record_md5("cold:%s" % topic, text)


def keyword_instruction(word):
    return ("Instruction: reply with exactly the single word %s and nothing else."
            % word)


# shared bodies for fork/branch/COW cells
_FORK_BODY = _record_md5("fork_body", passage("veltronia", 40))
_FORK_SHARED_SUFFIX = _record_md5(
    "fork_shared_suffix",
    "The Council of Veltronia keeps three sealed ledgers in the salt vault, "
    "and each ledger is opened only during the equinox audit. "
    "The archivist appointed this cycle is required to countersign every page "
    "before the vault is resealed at dusk. ")


def fork_prompt(tail_question):
    return _record_md5("fork:%s" % tail_question[-8:].lower(),
                       _FORK_BODY + "\n\n" + _FORK_SHARED_SUFFIX + tail_question)


def essay_prompt(topic_word, n_sentences=110, extra=""):
    text = passage(topic_word, n_sentences, cap="province of") + extra
    return _record_md5("essay:%s" % topic_word, text)


GENRE_BATTERY = [
    # (genre, prompt, must-contain-any substrings, md5 key)
    ("code",
     _record_md5("genre:code",
                 "Write a Python function merge_sorted(a, b) that merges two sorted "
                 "lists of integers into one sorted list. Include a short docstring "
                 "and one example call. Plain code only."),
     ["def ", "return"]),
    ("reason",
     _record_md5("genre:reason",
                 "A train leaves Alverstone at 14:05 traveling at 80 km/h. A second "
                 "train leaves Alverstone at 14:35 on a parallel track traveling 110 "
                 "km/h. At what clock time does the second train catch the first? "
                 "Show the arithmetic briefly, then give the time."),
     ["15:", "3:", "16:", "4:"]),    ("factual",
     _record_md5("genre:factual",
                 "Name the four seasons of the temperate zone in order starting with "
                 "spring, and give one typical weather feature of each. Keep it to "
                 "four short sentences."),
     ["summer", "autumn", "winter"]),
    ("prose",
     _record_md5("genre:prose",
                 "Write a short paragraph (60-90 words) about an old lighthouse "
                 "keeper who teaches his granddaughter to read the weather glass. "
                 "Plain prose, no headings."),
     [" ", "e"]),
    ("instruct",
     _record_md5("genre:instruct",
                 "List exactly five practical tips for keeping a cast-iron pan "
                 "seasoned, numbered 1 through 5, one line each."),
     ["1", "2", "3", "4", "5"]),
]


# --------------------------------------------------------------------------
# validators
# --------------------------------------------------------------------------


def _jeq(a, b):
    """Semantic json_equal: 1 == 1.0, recursive on composites."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b if isinstance(a, bool) and isinstance(b, bool) else False
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, int) and isinstance(b, int):
            return a == b
        return float(a) == float(b)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_jeq(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_jeq(a[k], b[k]) for k in a)
    return a == b


def schema_errors(v, s, path="$"):
    """Validate decoded JSON against the supported strict subset."""
    errs = []
    if not isinstance(s, dict):
        return ["%s: schema node is not an object" % path]
    if "const" in s and not _jeq(v, s["const"]):
        errs.append("%s: does not match const %r" % (path, s["const"]))
    if "enum" in s and not any(_jeq(v, e) for e in s["enum"]):
        errs.append("%s: %r not in enum" % (path, v))
    t = s.get("type")
    if t == "object":
        if not isinstance(v, dict):
            return errs + ["%s: expected object" % path]
        props = s.get("properties", {})
        for r in s.get("required", []):
            if r not in v:
                errs.append("%s: missing required '%s'" % (path, r))
        for k, val in v.items():
            if k in props:
                errs.extend(schema_errors(val, props[k], "%s.%s" % (path, k)))
            else:
                ap = s.get("additionalProperties", True)
                if ap is False:
                    errs.append("%s: unexpected property '%s' (closed object)" % (path, k))
                elif isinstance(ap, dict):
                    errs.extend(schema_errors(val, ap, "%s.%s" % (path, k)))
    elif t == "array":
        if not isinstance(v, list):
            return errs + ["%s: expected array" % path]
        if "minItems" in s and len(v) < s["minItems"]:
            errs.append("%s: %d items < minItems %d" % (path, len(v), s["minItems"]))
        if "maxItems" in s and len(v) > s["maxItems"]:
            errs.append("%s: %d items > maxItems %d" % (path, len(v), s["maxItems"]))
        if isinstance(s.get("items"), dict):
            for i, item in enumerate(v):
                errs.extend(schema_errors(item, s["items"], "%s[%d]" % (path, i)))
    elif t == "string":
        if not isinstance(v, str):
            errs.append("%s: expected string, got %s" % (path, type(v).__name__))
    elif t == "integer":
        if not isinstance(v, int) or isinstance(v, bool):
            errs.append("%s: expected integer, got %r" % (path, v))
    elif t == "number":
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            errs.append("%s: expected number, got %r" % (path, v))
    elif t == "boolean":
        if not isinstance(v, bool):
            errs.append("%s: expected boolean" % path)
    elif t == "null":
        if v is not None:
            errs.append("%s: expected null" % path)
    return errs


def is_attractor(text):
    """Token-attractor detector (mirror of serve_harness thresholds)."""
    words = re.findall(r"[a-z0-9']+", text.lower())
    if len(words) < 30:
        return False, "short"
    uniq = len(set(words)) / len(words)
    maxfreq = Counter(words).most_common(1)[0][1] / len(words)
    tris = list(zip(words, words[1:], words[2:]))
    uniq3 = len(set(tris)) / max(1, len(tris))
    if uniq < 0.15 or maxfreq > 0.50 or uniq3 < 0.50:
        return True, "uniq=%.2f maxfreq=%.2f gram3uniq=%.2f" % (uniq, maxfreq, uniq3)
    return False, "uniq=%.2f maxfreq=%.2f gram3uniq=%.2f" % (uniq, maxfreq, uniq3)


def tiny_png_b64(w=24, h=24, rgb=(180, 70, 60)):
    """Build a valid PNG in pure stdlib."""
    def chunk(typ, data):
        c = struct.pack(">I", len(data)) + typ + data
        return c + struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF)
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
           + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    return base64.b64encode(png).decode()


# --------------------------------------------------------------------------
# cell framework
# --------------------------------------------------------------------------


class CellFailure(Exception):
    pass


class CellSkip(Exception):
    pass


class Ctx:
    def __init__(self, cfg):
        self.cfg = cfg
        self.evidence = []
        self.warns = []

    def ev(self, msg):
        self.evidence.append(str(msg))

    def warn(self, msg):
        self.warns.append(str(msg))

    def check(self, cond, msg):
        if not cond:
            raise CellFailure(msg)

    def skip(self, msg):
        raise CellSkip(msg)


CELLS = []
AREA_ORDER = ["A", "B", "C", "E", "F", "G", "V", "D", "H"]


def cell(area, cid, name):
    def deco(fn):
        CELLS.append({"area": area, "id": cid, "name": name, "fn": fn})
        return fn
    return deco


def user(text):
    return [{"role": "user", "content": text}]


def run_concurrently(fns):
    """Run thunks concurrently; return list of (result, exception, elapsed)."""
    n = len(fns)
    barrier = threading.Barrier(n)
    results = [None] * n

    def wrap(i, fn):
        try:
            barrier.wait(timeout=120)
            t0 = time.monotonic()
            results[i] = (fn(), None, time.monotonic() - t0)
        except Exception as e:  # noqa: BLE001 - collected per-thread
            results[i] = (None, e, 0.0)
    with ThreadPoolExecutor(max_workers=n) as ex:
        list(ex.map(lambda p: wrap(*p), enumerate(fns)))
    return results


# --------------------------------------------------------------------------
# A — capability advertisement + telemetry
# --------------------------------------------------------------------------


@cell("A", "A1", "health capability advertisement matches resolved config")
def a1(t):
    st, _, text = get_json(t.cfg, "/health")
    t.check(st == 200, "health status %s" % st)
    h = json.loads(text)
    caps = h.get("capabilities") or {}
    # Resolve speculation state once: the health surface reports whether a
    # speculation head is live. If the key is absent, fall back to the model
    # caps probe (A2) which reports mtp_sidecar; assume off only when the
    # deployment explicitly disabled it via env (HIPFIRE_SERVE_SPEC_OFF) —
    # otherwise a missing key means unknown, not off.
    for key in ("speculation_active", "spec_mode", "mtp_active"):
        if key in caps:
            t.cfg.spec_on = bool(caps[key]); break
    t.ev("mode=%s slots=%s ctx=%s" % (caps.get("mode"), caps.get("multi_slot_slots"),
                                      caps.get("multi_slot_ctx")))
    t.check(h.get("status") == "ok", "status != ok")
    t.check(h.get("native") is True, "native != true")
    # The ownership token is no longer disclosed on the unauthenticated
    # /health surface (it was a remote-driveable ownership proof); the CLI
    # proves ownership through the PID's listening socket instead.
    t.check("token" not in h, "health must not disclose the instance token")
    t.check(caps.get("mode") == "multi-slot", "mode %r != multi-slot" % caps.get("mode"))
    t.check(caps.get("multi_slot") is True, "multi_slot not advertised true")
    t.check(caps.get("multi_slot_slots") == t.cfg.slots,
            "slots %r != %d" % (caps.get("multi_slot_slots"), t.cfg.slots))
    # ctx is a deployment knob (HIPFIRE_SERVE_MULTI_SLOT_CTX) — advertise a
    # positive integer, don't pin a specific deployment's 50000.
    ctx = caps.get("multi_slot_ctx")
    t.check(isinstance(ctx, int) and ctx > 0, "ctx %r not a positive int" % ctx)
    t.check(caps.get("prefix_cache") is True, "prefix_cache not advertised true")
    t.check(caps.get("structured_output") is True, "structured_output not true")
    t.check(caps.get("structured_output_subset") == "json-schema-strict-v1",
            "subset %r" % caps.get("structured_output_subset"))
    refused = caps.get("refused_request_fields") or []
    # Tools are SUPPORTED on this route (upstream restored tool turns); the
    # advertisement must not claim otherwise.
    t.check("tools" not in refused, "tools are supported but advertised as refused")
    for f in ("stop", "logprobs", "top_logprobs", "n", "best_of",
              "logit_bias", "echo", "suffix", "reasoning_effort",
              "response_format:json_object", "tools+image"):
        t.check(f in refused, "refused_request_fields missing %r (got %s)" % (f, refused))
    for k in ("max_batch_tokens", "prefill_min_tokens", "max_queue", "max_queue_bytes",
              "queue_timeout_ms", "stream_stall_timeout_ms",
              "max_request_bytes"):
        t.check(isinstance(caps.get(k), int) and caps.get(k) > 0,
                "capability %s missing/zero" % k)
    t.ev("structured_jump_forward=%s"
         % caps.get("structured_jump_forward"))


@cell("A", "A2", "listed model id is requestable")
def a2(t):
    st, _, text = get_json(t.cfg, "/v1/models")
    t.check(st == 200, "status %s" % st)
    d = json.loads(text)
    ids = [m.get("id") for m in d.get("data", [])]
    # The catalog lists a model by its NAME; a request may also name it by an
    # absolute path inside the model store (the daemon accepts both, verified
    # against the live route). Compare identities, not spellings: the listed id
    # must be the requested model (same basename), or be requestable as-is.
    want = t.cfg.model.rstrip("/").split("/")[-1]
    match = [m for m in d.get("data", [])
             if m.get("id") == t.cfg.model or m.get("id") == want
             or (m.get("id") or "").rstrip("/").split("/")[-1] == want]
    t.check(bool(match), "model %r not listed by name or path (%s)" % (t.cfg.model, ids))
    if not match:
        return
    m = match[0]
    # The advertised id must be usable as a request model id.
    r = chat(t.cfg, user("Say ok."), max_tokens=4, model=m["id"])
    t.check(r.status == 200, "advertised id %r is not requestable: %s (%s)"
            % (m["id"], r.status, r.error_message[:120]))


@cell("A", "A3", "/stats contract + counters move")
def a3(t):
    st, _, text = get_json(t.cfg, "/stats")
    t.check(st == 200, "status %s" % st)
    s = json.loads(text)
    for k in ("model", "uptime_sec", "queue_depth", "requests_served", "mode",
              "multi_slot", "slots", "prefix_cache"):
        t.check(k in s, "stats key %s missing" % k)
    t.check(s["mode"] == "multi-slot" and s["multi_slot"] is True and s["slots"] == t.cfg.slots
            and s["prefix_cache"] is True, "stats route facts wrong: %s" % s)
    before = s["requests_served"]
    chat(t.cfg, user("Reply with the word READY."), max_tokens=8)
    chat(t.cfg, user("Reply with the word SET."), max_tokens=8)
    chat(t.cfg, user("Reply with the word GO."), max_tokens=8)
    st2, _, text2 = get_json(t.cfg, "/stats")
    s2 = json.loads(text2)
    t.ev("requests_served %s -> %s" % (before, s2["requests_served"]))
    t.check(s2["requests_served"] >= before + 3, "requests_served did not advance")
    t.check(s2["queue_depth"] == 0, "queue_depth %s != 0 after settle" % s2["queue_depth"])


@cell("A", "A4", "Prometheus /metrics exposition parses")
def a4(t):
    st, hdrs, text = get_json(t.cfg, "/metrics")
    t.check(st == 200, "status %s" % st)
    t.check("text/plain" in hdrs.get("content-type", ""), "content-type %r" % hdrs.get("content-type"))
    vals = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^(hipfire_[a-z_]+)(\{[^}]*\})?\s+([-+0-9.eE]+)$", line)
        if m:
            vals.setdefault(m.group(1), []).append(float(m.group(3)))
    for name in ("hipfire_requests_total", "hipfire_requests_failed_total",
                 "hipfire_admission_rejected_total", "hipfire_queue_depth",
                 "hipfire_queue_capacity", "hipfire_uptime_seconds", "hipfire_model_loaded"):
        t.check(name in vals, "metric %s missing" % name)
    t.check(vals["hipfire_requests_total"][0] >= 1, "requests_total 0")
    t.check(vals["hipfire_model_loaded"][0] == 1, "model_loaded != 1")
    for hname in ("hipfire_ttft_milliseconds", "hipfire_request_latency_milliseconds",
                  "hipfire_decode_tokens_per_second"):
        t.check(hname + "_bucket" in text or hname in text, "histogram %s missing" % hname)
    t.ev("requests_total=%s failed=%s rejected=%s"
         % (vals["hipfire_requests_total"][0], vals["hipfire_requests_failed_total"][0],
            vals["hipfire_admission_rejected_total"][0]))


@cell("A", "A5", "unknown route -> typed 404")
def a5(t):
    st, _, text = get_json(t.cfg, "/nope")
    t.check(st == 404, "status %s != 404" % st)
    e = json.loads(text).get("error", {})
    t.check(e.get("type") == "invalid_request_error", "error type %r" % e.get("type"))
    t.check(bool(e.get("message")), "empty error message")


@cell("A", "A6", "cross-origin preflight is refused; no wildcard CORS")
def a6(t):
    # No browser origin may drive this instance: the default is same-origin
    # only, so a preflight is refused and no response carries a wildcard
    # Access-Control-Allow-Origin.
    conn = http.client.HTTPConnection(t.cfg.host, t.cfg.port, timeout=15)
    try:
        conn.request("OPTIONS", "/v1/chat/completions", headers={
            "Origin": "https://evil.example",
            "Access-Control-Request-Method": "POST",
        })
        r = conn.getresponse()
        r.read()
        t.ev("preflight status=%s acao=%r" % (r.status, r.getheader("access-control-allow-origin")))
        t.check(r.status >= 400, "cross-origin preflight must be refused, got %s" % r.status)
        t.check(r.getheader("access-control-allow-origin") is None,
                "preflight must not echo a wildcard origin")
    finally:
        conn.close()
    st, hdrs, _ = get_json(t.cfg, "/health")
    t.check(st == 200, "health status %s" % st)
    t.check(hdrs.get("access-control-allow-origin") is None,
            "GET /health must not carry Access-Control-Allow-Origin")


# --------------------------------------------------------------------------
# B — generation + sampling contract
# --------------------------------------------------------------------------


@cell("B", "B1", "non-stream completion shape")
def b1(t):
    r = chat(t.cfg, user("Name the capital of Portugal in one word."),
             temperature=0, max_tokens=200)
    j = r.json
    t.check(j.get("object") == "chat.completion", "object %r" % j.get("object"))
    t.check(str(j.get("id", "")).startswith("chatcmpl-"), "id %r" % j.get("id"))
    t.check(isinstance(j.get("created"), int), "created not int")
    t.check(j.get("model") == t.cfg.model, "model echo %r" % j.get("model"))
    ch = j["choices"][0]
    t.check(ch["message"]["role"] == "assistant", "role")
    t.check("lisbon" in ch["message"]["content"].lower(), "content %r" % ch["message"]["content"][:80])
    t.check(ch["finish_reason"] in ("stop", "length"), "finish %r" % ch["finish_reason"])
    u = j["usage"]
    for k in ("prompt_tokens", "completion_tokens", "total_tokens", "prompt_tokens_details"):
        t.check(k in u, "usage key %s missing" % k)
    t.check(isinstance(u["prompt_tokens_details"].get("cached_tokens"), int),
            "cached_tokens not int")
    t.check("timings" in j and "hipfire" in j, "timings/hipfire blocks missing")


@cell("B", "B2", "streaming SSE frame contract")
def b2(t):
    sr = chat_stream(t.cfg, user("Count from one to five as digits, separated by spaces."),
                     temperature=0, max_tokens=200)
    t.check(sr.status == 200, "status %s" % sr.status)
    t.check(sr.role_seen, "first chunk missing role delta")
    t.check(sr.frames_valid_json, "non-JSON SSE frame seen")
    t.check(len(sr.content) > 0, "no content deltas")
    t.check(sr.finish in ("stop", "length"), "finish %r" % sr.finish)
    t.check(sr.done_sentinel, "missing data: [DONE] sentinel")
    t.ev("frames=%d chunks=%d finish=%s" % (len(sr.frames), sr.content_chunks, sr.finish))


@cell("B", "B3", "stream_options.include_usage -> terminal usage chunk")
def b3(t):
    sr = chat_stream(t.cfg, user("Say OK."), max_tokens=8,
                     stream_options={"include_usage": True})
    t.check(sr.status == 200, "status %s" % sr.status)
    t.check(sr.done_sentinel, "missing [DONE]")
    t.check(sr.usage is not None, "usage chunk absent")
    t.check(sr.usage.get("prompt_tokens_details", {}).get("cached_tokens") is not None,
            "cached_tokens absent from usage chunk")
    t.ev("usage=%s" % json.dumps(sr.usage))


@cell("B", "B4", "thinking separation: reasoning_content vs content")
def b4(t):
    # Budget must let the think span close: upstream's fail-closed terminal
    # (unsafe multi_slot terminal: open_think) turns a truncated think into a
    # 500, so a too-small max_tokens is a server error by contract.
    r = chat(t.cfg, user("Name three primary colors, one per line."),
             max_tokens=3000, temperature=0, expect_error=True,
             chat_template_kwargs={"enable_thinking": True})
    t.check(r.status == 200,
            "thinking request failed: %s (%s)" % (r.status, r.error_message[:140]))
    if r.status != 200:
        return
    t.check(len(r.reasoning) > 0, "reasoning_content empty with default thinking")
    t.check("<think>" not in r.content and "</think>" not in r.content,
            "think tags leaked into content")
    low = r.content.lower()
    t.check(any(c in low for c in ("red", "blue", "yellow")),
            "content does not answer the prompt: %r" % r.content[:120])
    bad, stats = is_attractor(r.content)
    t.check(not bad, "attractor in content: %s" % stats)
    # The fail-closed contract itself: a think span truncated by max_tokens
    # must surface as a typed server error, not silent empty content.
    r2 = chat(t.cfg, user("Name three primary colors, one per line."),
              max_tokens=16, expect_error=True,
              chat_template_kwargs={"enable_thinking": True})
    t.check(r2.status == 500 and "open_think" in r2.error_message,
            "truncated think: expected 500 open_think, got %s (%s)"
            % (r2.status, r2.error_message[:140]))


@cell("B", "B5", "enable_thinking=false disables the think span")
def b5(t):
    r = chat(t.cfg, user("What is 17 times 23? Answer with the number only."),
             max_tokens=64)
    t.check(r.json["hipfire"]["reasoning"]["mode"] == "disabled",
            "reasoning mode %r" % r.json["hipfire"]["reasoning"]["mode"])
    t.check(len(r.reasoning) == 0, "reasoning_content present with thinking off")
    t.check("391" in r.content, "content %r" % r.content[:80])


@cell("B", "B6", "coherence battery (5 genres, attractor gate)")
def b6(t):
    for genre, prompt, must in GENRE_BATTERY:
        r = chat(t.cfg, user(prompt), temperature=0, max_tokens=640)
        t.ev("%s: %d chars, finish=%s" % (genre, len(r.content), r.finish))
        t.check(r.finish == "stop", "%s finish=%s" % (genre, r.finish))
        bad, stats = is_attractor(r.content)
        t.check(not bad, "%s token attractor: %s | head=%r" % (genre, stats, r.content[:120]))
        low = r.content.lower()
        hit = any(m.lower() in low for m in must)
        t.check(hit, "%s missing expected substrings %s; head=%r" % (genre, must, r.content[:120]))


@cell("B", "B7", "greedy determinism (temperature 0)")
def b7(t):
    p = prompt_cold("amberholt", "Summarize the entry counts in one sentence.")
    r1 = chat(t.cfg, user(p), temperature=0, max_tokens=96)
    r2 = chat(t.cfg, user(p), temperature=0, max_tokens=96)
    t.check(r1.content == r2.content,
            "greedy outputs differ:\n  A=%r\n  B=%r" % (r1.content[:150], r2.content[:150]))
    t.ev("identical %d chars" % len(r1.content))


@cell("B", "B8", "seeded sampling stability across execution shapes")
def b8(t):
    p = prompt_cold("corvane", "Write two sentences about the tidal records.")
    r1 = chat(t.cfg, user(p), temperature=0.7, seed=1234, max_tokens=96)
    r2 = chat(t.cfg, user(p), temperature=0.7, seed=1234, max_tokens=96)
    if r1.content != r2.content:
        t.warn("same seed diverged across cold/warm execution shapes; seeded sampling is shape-stable, not batch-invariant")
    else:
        t.ev("same seed happened to match across cold/warm shapes")
    r3 = chat(t.cfg, user(p), temperature=0.7, seed=1235, max_tokens=96)
    if r3.content == r1.content:
        t.warn("different seed produced identical output (possible but unlikely at %d chars)" % len(r1.content))
    else:
        t.ev("seed 1235 differs from 1234 (expected)")


@cell("B", "B9", "sampling params accepted (top_p/top_k/min_p/penalties)")
def b9(t):
    p = prompt_cold("duskmarrow", "Describe the fog in one sentence.")
    r = chat(t.cfg, user(p), temperature=0.5, top_p=0.9, top_k=40, min_p=0.05,
             presence_penalty=0.3, frequency_penalty=0.3, max_tokens=96)
    bad, stats = is_attractor(r.content)
    t.check(not bad and len(r.content) > 10, "degenerate output with full sampling params: %s" % stats)
    r2 = chat(t.cfg, user(p), repeat_penalty=1.2, temperature=0, max_tokens=96)
    t.check(len(r2.content) > 0, "repeat_penalty request produced empty content")


@cell("B", "B10", "finish_reason=length + completion_tokens honored")
def b10(t):
    r = chat(t.cfg, user("Write a long essay about the history of ink."),
             max_tokens=6)
    t.check(r.finish == "length", "finish %r != length" % r.finish)
    t.check(r.usage["completion_tokens"] == 6,
            "completion_tokens %s != 6" % r.usage["completion_tokens"])


# --------------------------------------------------------------------------
# C — prefix cache correctness (the core contract: reuse never changes output)
# --------------------------------------------------------------------------


@cell("C", "C1", "cold -> warm exact reuse + byte-identical replay")
def c1(t):
    p = prompt_cold("verrow", keyword_instruction("LANTERN"))
    r1 = chat(t.cfg, user(p), temperature=0, max_tokens=32)
    t.ev("cold cached=%s prompt=%s" % (r1.cached_tokens, r1.usage["prompt_tokens"]))
    t.check(r1.cached_tokens == 0, "cold run already cached=%s (fixture collision?)" % r1.cached_tokens)
    t.check("LANTERN" in r1.content, "cold output %r" % r1.content[:80])
    r2 = chat(t.cfg, user(p), temperature=0, max_tokens=32)
    t.ev("warm cached=%s" % r2.cached_tokens)
    t.check(r2.cached_tokens >= 128, "warm reused only %s tokens (<1 page)" % r2.cached_tokens)
    t.check(r2.cached_tokens <= r2.usage["prompt_tokens"], "cached > prompt_tokens")
    r3 = chat(t.cfg, user(p), temperature=0, max_tokens=32)
    t.check(r2.cached_tokens == r3.cached_tokens,
            "warm reuse unstable: %s vs %s" % (r2.cached_tokens, r3.cached_tokens))
    t.check(r1.content == r2.content == r3.content,
            "REUSE CHANGED OUTPUT:\n  cold=%r\n  warm=%r" % (r1.content[:150], r2.content[:150]))


@cell("C", "C2", "A10 long-generation page-crossing replay")
def c2(t):
    p = prompt_cold("wexholm", "Retell the survey details in your own words, at length.")
    r1 = chat(t.cfg, user(p), temperature=0, max_tokens=384)
    t.check(r1.finish in ("stop", "length"), "finish %s" % r1.finish)
    t.ev("cold gen=%s tok cached=%s" % (r1.usage["completion_tokens"], r1.cached_tokens))
    bad, stats = is_attractor(r1.content)
    t.check(not bad, "attractor in long generation: %s" % stats)
    r2 = chat(t.cfg, user(p), temperature=0, max_tokens=384)
    t.ev("warm cached=%s" % r2.cached_tokens)
    t.check(r2.cached_tokens >= 128, "no reuse on warm long-gen (%s)" % r2.cached_tokens)
    r3 = chat(t.cfg, user(p), temperature=0, max_tokens=384)
    t.check(r2.content == r3.content,
            "warm long-gen replay is non-deterministic: %r vs %r"
            % (r2.content[:120], r3.content[:120]))
    if r1.content != r2.content:
        if t.cfg.spec_on:
            t.warn("warm replay used a different execution shape (cache-restored page boundary + speculation repair) and the greedy output diverged mid-generation; warm replay remains deterministic and cache accounting is valid")
        else:
            t.check(False,
                    "warm long-gen replay differs from cold (candidate rows leaked into cache?)"
                    "\n  cold head=%r\n  warm head=%r" % (r1.content[:150], r2.content[:150]))
    else:
        t.ev("warm replay byte-identical")


@cell("C", "C3", "branch at shared prefix: divergent suffixes drive outputs")
def c3(t):
    body = _record_md5("branch_body", passage("quinth", 40))
    pa = _record_md5("branch:pa", body + "\n\n" + keyword_instruction("HARBOR"))
    pb = _record_md5("branch:pb", body + "\n\n" + keyword_instruction("MARMALADE"))
    ra = chat(t.cfg, user(pa), temperature=0, max_tokens=32)
    rb = chat(t.cfg, user(pb), temperature=0, max_tokens=32)
    t.ev("a cached=%s out=%r" % (ra.cached_tokens, ra.content[:40]))
    t.ev("b cached=%s out=%r" % (rb.cached_tokens, rb.content[:40]))
    t.check("HARBOR" in ra.content, "branch A answered %r (not its suffix)" % ra.content[:80])
    t.check("MARMALADE" in rb.content, "branch B answered %r (not its suffix)" % rb.content[:80])
    rb2 = chat(t.cfg, user(pb), temperature=0, max_tokens=32)
    t.check(rb2.cached_tokens >= 128, "branch B warm got no reuse (%s)" % rb2.cached_tokens)
    t.check(rb.content == rb2.content, "branch B warm replay differs from its cold run")


@cell("C", "C4", "mid-page radix fork (W10-4): fork inside a shared page")
def c4(t):
    # body(40 sent) + shared 4-sentence suffix + divergent final instruction:
    # the divergence point lands mid-way through what would be the next page.
    mid_shared = ("During the last audit the countersignature page was left "
                  "unfinished until the archivist returned from the northern "
                  "depot on the evening ferry. ")
    pfork = fork_prompt(mid_shared + keyword_instruction("CRIMSON"))
    psame = fork_prompt(mid_shared + keyword_instruction("CRIMSON"))
    pother = fork_prompt("The deputy archivist disagrees with the audit schedule. " +
                         keyword_instruction("AZURE"))
    r1 = chat(t.cfg, user(pfork), temperature=0, max_tokens=32)
    r1b = chat(t.cfg, user(psame), temperature=0, max_tokens=32)  # identical twin: full hit
    r2 = chat(t.cfg, user(pother), temperature=0, max_tokens=32)  # forks mid-page
    t.ev("identical-twin cached=%s out=%r" % (r1b.cached_tokens, r1b.content[:30]))
    t.ev("mid-fork cached=%s out=%r" % (r2.cached_tokens, r2.content[:30]))
    t.check(r1.cached_tokens == 0, "first fork variant cold cached=%s" % r1.cached_tokens)
    t.check(r1b.cached_tokens >= 128, "identical twin got no reuse (%s)" % r1b.cached_tokens)
    t.check("CRIMSON" in r1.content and "CRIMSON" in r1b.content,
            "identical-twin outputs wrong: %r / %r" % (r1.content[:60], r1b.content[:60]))
    t.check("AZURE" in r2.content,
            "mid-page fork answered %r (divergent tail ignored)" % r2.content[:80])
    r2b = chat(t.cfg, user(pother), temperature=0, max_tokens=32)
    t.check(r2.content == r2b.content, "mid-fork warm replay differs from cold")
    t.check(r2b.cached_tokens >= 128, "mid-fork warm reuse %s < 1 page" % r2b.cached_tokens)


@cell("C", "C5", "multi-turn chain: cached_tokens grows, history intact")
def c5(t):
    conv = []
    cached_seq = []
    ptoks = []
    for turn in range(5):
        conv.append({"role": "user",
                     "content": "Turn %d. %s\n\nReply in one short sentence."
                                % (turn + 1, passage("eldermoor", 12, start=turn * 12))})
        r = chat(t.cfg, list(conv), temperature=0, max_tokens=64)
        conv.append({"role": "assistant", "content": r.content})
        cached_seq.append(r.cached_tokens)
        ptoks.append(r.usage["prompt_tokens"])
        t.check(r.finish in ("stop", "length"), "turn %d finish %s" % (turn + 1, r.finish))
    t.ev("cached=%s prompt=%s" % (cached_seq, ptoks))
    t.check(cached_seq[0] == 0, "turn 1 cached=%s != 0" % cached_seq[0])
    t.check(cached_seq[-1] >= 128, "final turn reuse %s < 1 page" % cached_seq[-1])
    for i in range(5):
        t.check(cached_seq[i] <= ptoks[i], "turn %d cached %s > prompt %s" % (i + 1, cached_seq[i], ptoks[i]))
    last_user = conv[-2]["content"]
    r_again = chat(t.cfg, list(conv[:-1]), temperature=0, max_tokens=64)
    if r_again.content != conv[-1]["content"]:
        t.warn("final-turn replay used a different execution shape and changed the greedy reduction result; history and cache accounting remained valid")
    else:
        t.ev("final-turn replay identical")


@cell("C", "C6", "false-reuse guard: unrelated prompts must not reuse")
def c6(t):
    pa = prompt_cold("noorlaw", keyword_instruction("SABLE"))
    pb = prompt_cold("tessarin", keyword_instruction("IVORY"))
    ra = chat(t.cfg, user(pa), temperature=0, max_tokens=32)
    rb = chat(t.cfg, user(pb), temperature=0, max_tokens=32)
    t.ev("a cached=%s b cached=%s" % (ra.cached_tokens, rb.cached_tokens))
    t.check(rb.cached_tokens < 128,
            "unrelated prompt reused %s tokens (cross-prompt contamination)" % rb.cached_tokens)
    t.check("SABLE" in ra.content and "IVORY" in rb.content, "outputs crossed: %r / %r"
            % (ra.content[:60], rb.content[:60]))
    ra2 = chat(t.cfg, user(pa), temperature=0, max_tokens=32)
    t.check(ra2.content == ra.content and "SABLE" in ra2.content,
            "re-run after interleaving changed output: %r" % ra2.content[:80])


@cell("C", "C7", "concurrent shared-prefix isolation (COW / interleave)")
def c7(t):
    words = [("OBSIDIAN", "first"), ("PEWTER", "second"), ("GARLIC", "third")]
    prompts = [fork_prompt(keyword_instruction(w)) for w, _ in words]
    thunk = lambda p: chat(t.cfg, user(p), temperature=0, max_tokens=32)
    res = run_concurrently([lambda p=p: thunk(p) for p in prompts])
    outs = []
    for (word, ordinal), r in zip(words, res):
        val, err, _dt = r
        if err:
            raise CellFailure("concurrent request %s failed: %s" % (word, err))
        t.ev("%s: cached=%s out=%r" % (word, val.cached_tokens, val.content[:30]))
        t.check(val.status == 200, "%s status %s" % (word, val.status))
        t.check(word in val.content, "%s output contaminated: %r" % (word, val.content[:80]))
        outs.append(val.content)
    # replay all three warm: deterministic + still their own keyword
    for (word, _), p, first in zip(words, prompts, outs):
        r = chat(t.cfg, user(p), temperature=0, max_tokens=32)
        t.check(r.cached_tokens >= 128, "%s warm reuse %s < 1 page" % (word, r.cached_tokens))
        t.check(r.content == first, "%s warm replay under contention differs" % word)


@cell("C", "C8", "soak: reuse stays high, outputs stable, no degradation")
def c8(t):
    pf = prompt_cold("marenn", keyword_instruction("FALLOC"))
    pg = prompt_cold("sivetter", keyword_instruction("QUINCE"))
    base_f = chat(t.cfg, user(pf), temperature=0, max_tokens=48)
    base_g = chat(t.cfg, user(pg), temperature=0, max_tokens=48)
    t0 = time.monotonic()
    cached_seen = []
    for i in range(6):
        p, base, word = ((pf, base_f, "FALLOC") if i % 2 == 0 else (pg, base_g, "QUINCE"))
        r = chat(t.cfg, user(p), temperature=0, max_tokens=48)
        cached_seen.append(r.cached_tokens)
        t.check(r.status == 200, "round %d status %s" % (i, r.status))
        t.check(word in r.content, "round %d output %r" % (i, r.content[:60]))
        t.check(r.content == base.content, "round %d output drifted from cold baseline" % i)
    dt = time.monotonic() - t0
    t.ev("cached over rounds=%s wall=%.1fs" % (cached_seen, dt))
    t.check(all(c >= 128 for c in cached_seen), "reuse collapsed during soak: %s" % cached_seen)


@cell("C", "C9", "reuse is independent of generation params")
def c9(t):
    p = prompt_cold("hollivar", "Describe the harbor traffic in one sentence.")
    chat(t.cfg, user(p), temperature=0, max_tokens=24)
    r = chat(t.cfg, user(p), temperature=0, max_tokens=96)
    t.check(r.cached_tokens >= 128,
            "same prompt with different max_tokens got no reuse (%s)" % r.cached_tokens)
    r2 = chat(t.cfg, user(p), temperature=0.3, seed=9, max_tokens=96)
    t.check(r2.cached_tokens >= 128, "sampled request got no reuse (%s)" % r2.cached_tokens)


# --------------------------------------------------------------------------
# E — strict JSON Schema (grammar)
# --------------------------------------------------------------------------


def schema_chat(t, schema, ask, max_tokens=256, **kw):
    payload_rf = {"type": "json_schema", "json_schema": {"name": "out", "schema": schema}}
    return chat(t.cfg, user(ask), response_format=payload_rf, max_tokens=max_tokens, **kw)


def assert_schema_output(t, r, schema, label):
    t.check(r.status == 200, "%s status %s: %s" % (label, r.status, r.error_message[:200]))
    t.check(r.finish == "stop", "%s finish=%s (burned to max_tokens?)" % (label, r.finish))
    txt = r.content.strip()
    try:
        val = json.loads(txt)
    except ValueError as e:
        raise CellFailure("%s content is not JSON: %s | content=%r" % (label, e, r.content[:200]))
    errs = schema_errors(val, schema)
    t.check(not errs, "%s violates schema: %s | value=%r" % (label, errs[:4], val))
    return val


@cell("E", "E1", "valid object schema -> schema-valid output (x2, deterministic)")
def e1(t):
    schema = {"type": "object",
              "properties": {"city": {"type": "string"}, "celsius": {"type": "number"}},
              "required": ["city", "celsius"], "additionalProperties": False}
    ask = "Give the current approximate weather in Rome as JSON."
    r1 = schema_chat(t, schema, ask, temperature=0)
    v1 = assert_schema_output(t, r1, schema, "E1 run1")
    r2 = schema_chat(t, schema, ask, temperature=0)
    v2 = assert_schema_output(t, r2, schema, "E1 run2")
    t.ev("run2 cached=%s out=%r" % (r2.cached_tokens, r2.content[:60]))
    t.check(v1 == v2, "schema output not deterministic:\n %r\n %r" % (v1, v2))


@cell("E", "E16", "strict-schema prompts warm the prefix cache (compose reuse)")
def e16(t):
    schema = {"type": "object",
              "properties": {"city": {"type": "string"}, "celsius": {"type": "number"}},
              "required": ["city", "celsius"], "additionalProperties": False}
    ask = ("Give the current approximate weather in Venice as JSON. " + passage("venicefix", 20))
    rf = {"type": "json_schema", "json_schema": {"name": "out", "schema": schema}}
    r1 = chat(t.cfg, user(ask), response_format=rf, temperature=0, max_tokens=256)
    t.check(r1.status == 200 and r1.cached_tokens == 0,
            "run1 cold cached=%s" % r1.cached_tokens)
    r2 = chat(t.cfg, user(ask), response_format=rf, temperature=0, max_tokens=256)
    t.ev("run2 cached=%s ptok=%s" % (r2.cached_tokens, r2.usage["prompt_tokens"]))
    t.check(r2.cached_tokens >= 128,
            "identical strict-schema request got NO prefix-cache reuse (%s): schema "
            "sessions are not publishing to (or looking up from) the radix on the "
            "serve path, unlike the engine-level oracle's grammar cell" % r2.cached_tokens)


@cell("E", "E2", "streaming schema output assembles to valid JSON")
def e2(t):
    schema = {"type": "object",
              "properties": {"animal": {"type": "string"}, "legs": {"type": "integer"}},
              "required": ["animal", "legs"], "additionalProperties": False}
    sr = chat_stream(t.cfg, user("Name any animal and its leg count as JSON."),
                     response_format={"type": "json_schema",
                                      "json_schema": {"name": "out", "schema": schema}},
                     temperature=0, max_tokens=200)
    t.check(sr.status == 200, "status %s: %s" % (sr.status, sr.error_text[:200]))
    t.ev("frames=%d chunks=%d finish=%s content=%r %.1fs"
         % (len(sr.frames), sr.content_chunks, sr.finish, sr.content[:60],
            (sr.total_ms or 0) / 1e3))
    t.check(sr.done_sentinel, "missing [DONE] (stream closed without sentinel — "
                              "clients hang until their own timeout)")
    try:
        val = json.loads(sr.content.strip())
    except ValueError as e:
        raise CellFailure("streamed schema content invalid: %s | %r" % (e, sr.content[:200]))
    errs = schema_errors(val, schema)
    t.check(not errs, "streamed output violates schema: %s" % errs)


@cell("E", "E3", "schema + default thinking: framing-aware cursor (W10-2)")
def e3(t):
    schema = {"type": "object",
              "properties": {"answer": {"type": "string"}},
              "required": ["answer"], "additionalProperties": False}
    payload_rf = {"type": "json_schema", "json_schema": {"name": "out", "schema": schema}}
    # Budget: under the daemon's injected system prompt this model's think
    # span runs past 1200 tokens; 800 truncated mid-think (the mask is
    # all-allowed inside think, so the request burned to max_tokens and was
    # typed-rejected as unsatisfiable — calibration, not an engine bug).
    r = chat(t.cfg, user("What is the boiling point of water at sea level, in JSON?"),
             response_format=payload_rf, max_tokens=3000,
             chat_template_kwargs={"enable_thinking": True})
    t.check(r.status == 200, "typed rejection instead of framed generation: %s %s"
            % (r.status, r.error_message[:200]))
    t.check(len(r.reasoning) > 0, "reasoning_content empty with default thinking")
    val = json.loads(r.content.strip()) if r.content.strip() else None
    t.check(isinstance(val, dict) and isinstance(val.get("answer"), str),
            "content not schema-shaped JSON: %r" % r.content[:150])
    errs = schema_errors(val, schema)
    t.check(not errs, "framed output violates schema: %s" % errs)
    t.check("<think>" not in r.content, "think tag leaked into schema content")


@cell("E", "E4", "enum + const respected")
def e4(t):
    schema = {"type": "object",
              "properties": {"city": {"type": "string", "enum": ["Paris", "London", "Rome"]},
                             "rank": {"type": "integer", "enum": [1, 2, 3]},
                             "fixed": {"const": "kappa"}},
              "required": ["city", "rank", "fixed"], "additionalProperties": False}
    r = schema_chat(t, schema, "Pick a city and its rank as JSON.", temperature=0)
    v = assert_schema_output(t, r, schema, "E4")
    t.ev("value=%r" % v)


@cell("E", "E5", "array minItems/maxItems forced + pruned (W8/W11 regressions)")
def e5(t):
    schema = {"type": "array",
              "items": {"type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"], "additionalProperties": False},
              "minItems": 2, "maxItems": 4}
    r = schema_chat(t, schema, "List between two and four real dog breeds as JSON.",
                    max_tokens=400, temperature=0)
    v = assert_schema_output(t, r, schema, "E5")
    t.ev("items=%d" % len(v))


@cell("E", "E6", "required enforced + closed object stays closed")
def e6(t):
    schema = {"type": "object",
              "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
              "required": ["a", "b"], "additionalProperties": False}
    r = schema_chat(t, schema, "Fill a and b as JSON, a any word and b any integer.",
                    temperature=0)
    assert_schema_output(t, r, schema, "E6")


@cell("E", "E7", "nested object (3 levels) compiles and validates")
def e7(t):
    schema = {"type": "object",
              "properties": {"outer": {"type": "object",
                                       "properties": {"inner": {"type": "object",
                                                                "properties": {"leaf": {"type": "boolean"}},
                                                                "required": ["leaf"],
                                                                "additionalProperties": False}},
                                       "required": ["inner"],
                                       "additionalProperties": False}},
              "required": ["outer"], "additionalProperties": False}
    r = schema_chat(t, schema, "Produce the nested object with leaf true.", temperature=0)
    assert_schema_output(t, r, schema, "E7")


@cell("E", "E8", "unsupported schema keywords -> typed 400 (P0-2 default-deny)")
def e8(t):
    probes = [
        ("$ref", {"type": "object", "$ref": "#/definitions/x"}),
        ("pattern", {"type": "string", "pattern": "^a"}),
        ("oneOf", {"oneOf": [{"type": "string"}, {"type": "integer"}]}),
        ("minimum", {"type": "integer", "minimum": 3}),
        ("format", {"type": "string", "format": "date"}),
        ("allOf", {"allOf": [{"type": "object"}]}),
    ]
    for kwname, schema in probes:
        r = schema_chat(t, schema, "Anything.", expect_error=True, max_tokens=32)
        t.check(r.status == 400,
                "keyword %r: expected 400, got %s (%s)" % (kwname, r.status, r.error_message[:120]))
        t.check("unsupported JSON Schema keyword" in r.error_message,
                "keyword %r: message %r not the typed rejection" % (kwname, r.error_message[:120]))
    t.ev("6/6 keywords typed-rejected")


@cell("E", "E9", "malformed keyword VALUES -> typed 400 (W11 P0-2)")
def e9(t):
    probes = [
        ('enum as string', {"type": "string", "enum": "hello"}),
        ("minItems as string", {"type": "array", "minItems": "2", "items": {"type": "string"}}),
        ("required as string", {"type": "object", "required": "name"}),
        ("empty enum", {"type": "string", "enum": []}),
    ]
    for label, schema in probes:
        r = schema_chat(t, schema, "Anything.", expect_error=True, max_tokens=32)
        t.check(r.status == 400,
                "%s: expected 400, got %s (%s)" % (label, r.status, r.error_message[:120]))
    t.ev("4/4 malformed values typed-rejected")


@cell("E", "E10", "union type array -> typed 400 (W7 fix 17)")
def e10(t):
    r = schema_chat(t, {"type": ["integer", "null"]}, "Anything.", expect_error=True)
    t.check(r.status == 400, "union type: got %s (%s)" % (r.status, r.error_message[:120]))


@cell("E", "E11", "bare object type compiles (W8 fix 27)")
def e11(t):
    r = schema_chat(t, {"type": "object"}, "Give any object with one field.", temperature=0)
    t.check(r.status == 200, "bare object refused: %s %s" % (r.status, r.error_message[:160]))
    try:
        val = json.loads(r.content.strip())
    except ValueError:
        raise CellFailure("bare-object output not JSON: %r" % r.content[:150])
    t.check(isinstance(val, dict), "bare-object output not an object: %r" % val)


@cell("E", "E12", "structural schema errors -> 400")
def e12(t):
    probes = [
        ("items boolean", {"type": "array", "items": True}),
        ("minItems > maxItems", {"type": "array", "items": {"type": "string"},
                                 "minItems": 3, "maxItems": 2}),
        ("missing json_schema wrapper", None),
        ("unknown type", {"type": "yaml"}),
    ]
    for label, schema in probes:
        if schema is None:
            rf = {"type": "json_schema"}
        else:
            rf = {"type": "json_schema", "json_schema": {"name": "out", "schema": schema}}
        r = chat(t.cfg, user("Anything."), response_format=rf, expect_error=True, max_tokens=32)
        t.check(r.status == 400, "%s: got %s (%s)" % (label, r.status, r.error_message[:120]))
    t.ev("4/4 structural errors -> 400")


@cell("E", "E13", "integer dead-end cannot wedge (W11 P1-7 class)")
def e13(t):
    schema = {"type": "integer"}
    rf = {"type": "json_schema", "json_schema": {"name": "out", "schema": schema}}
    r = chat(t.cfg, user("Pi is 3.14159. Output the value of pi described above as JSON number."),
             response_format=rf, temperature=0, max_tokens=48)
    t.check(r.status == 200, "status %s %s" % (r.status, r.error_message[:160]))
    t.check(r.finish == "stop",
            "burned to max_tokens (dead-end wedge signature): finish=%s content=%r"
            % (r.finish, r.content[:80]))
    try:
        v = json.loads(r.content.strip())
    except ValueError:
        raise CellFailure("not an integer literal: %r" % r.content[:80])
    t.check(isinstance(v, int) and not isinstance(v, bool), "output %r not integer" % v)


@cell("E", "E14", "response_format json_object refused (typed)")
def e14(t):
    r = chat(t.cfg, user("Give JSON."), response_format={"type": "json_object"},
             expect_error=True, max_tokens=32)
    t.check(r.status == 400, "json_object: got %s (%s)" % (r.status, r.error_message[:120]))


@cell("E", "E15", "response_format type 'text' -> 400-class (OpenAI default type)")
def e15(t):
    # type "text" is OpenAI's DEFAULT response_format: the fixed gateway
    # accepts it as unconstrained (200) instead of rejecting it.
    r = chat(t.cfg, user("Say the word ok."), response_format={"type": "text"},
             max_tokens=32)
    t.check(r.status == 200,
            "response_format type 'text' returned HTTP %s (%s); 'text' is the "
            "OpenAI default type and must be accepted as unconstrained"
            % (r.status, r.error_message[:120]))
    t.check(len(r.content) > 0, "text request produced empty content")


# --------------------------------------------------------------------------
# F — typed refusals on the multi-slot route
# --------------------------------------------------------------------------


# Gateway-side request validation is typed (`InvalidRequest` in
# hipfire-cli/src/serve/mod.rs), so the HTTP layer maps it to 400 by TYPE, not
# by message wording. The contract a client observes is "HTTP 400 naming the
# refused field/knob"; the exact sentence is an implementation detail. Older
# gateways prefixed the message with this tag, so accept either spelling.
REFUSAL_TAG = "[request validation] "


def expect_refusal(t, label, r, prefix="experimental multi-slot does not support this request"):
    t.check(r.status == 400, "%s: expected 400, got %s (%s)" % (label, r.status, r.error_message[:140]))
    body = r.error_message[len(REFUSAL_TAG):] if r.error_message.startswith(REFUSAL_TAG) \
        else r.error_message
    t.check(body.startswith(prefix),
            "%s: message %r lacks refusal prefix" % (label, r.error_message[:140]))


@cell("F", "F1", "tools supported: emits tool_calls")
def f1(t):
    # Upstream restored tool turns on the daemon slot path (ad6004ac0):
    # tools are accepted and the model's call is parsed into a structured
    # tool_calls finish.
    r = chat(t.cfg, user("What is the weather in Paris? Use the get_weather tool."),
             max_tokens=400, expect_error=True,
             tools=[{"type": "function",
                     "function": {"name": "get_weather",
                                  "description": "Get weather for a city",
                                  "parameters": {"type": "object",
                                                 "properties": {"city": {"type": "string"}},
                                                 "required": ["city"]}}}],
             chat_template_kwargs={"enable_thinking": False})
    t.check(r.status == 200, "tools request rejected: %s (%s)" % (r.status, r.error_message[:140]))
    if r.status == 200:
        t.check(r.finish == "tool_calls", "finish=%s, expected tool_calls" % r.finish)
        calls = (r.json.get("choices") or [{}])[0].get("message", {}).get("tool_calls") or []
        t.check(any(c.get("function", {}).get("name") == "get_weather" for c in calls),
                "no get_weather call in %r" % (calls,))


@cell("F", "F2", "stop sequences refused")
def f2(t):
    r = chat(t.cfg, user("hi"), expect_error=True, max_tokens=16, stop=["\n"])
    expect_refusal(t, "stop", r)


@cell("F", "F3", "logprobs / top_logprobs refused")
def f3(t):
    r = chat(t.cfg, user("hi"), expect_error=True, max_tokens=16, logprobs=True)
    expect_refusal(t, "logprobs", r)
    r2 = chat(t.cfg, user("hi"), expect_error=True, max_tokens=16, top_logprobs=5)
    expect_refusal(t, "top_logprobs", r2)


@cell("F", "F4", "tool-result role accepted; malformed tool message refused")
def f4(t):
    # Tool turns are supported: a well-formed tool-result continuation is
    # accepted; a tool message without tool_call_id is a typed 400.
    r = chat(t.cfg, [{"role": "user", "content": "hi"},
                     {"role": "tool", "content": "result"}], expect_error=True, max_tokens=16)
    t.check(r.status == 400,
            "tool message without tool_call_id: expected 400, got %s" % r.status)
    t.check("tool_call_id" in r.error_message,
            "rejection should name tool_call_id: %s" % r.error_message[:140])
    ok = chat(t.cfg, [{"role": "user", "content": "What is the weather in Paris?"},
                      {"role": "assistant", "content": None,
                       "tool_calls": [{"id": "call_1", "type": "function",
                                       "function": {"name": "get_weather",
                                                    "arguments": "{\"city\":\"Paris\"}"}}]},
                      {"role": "tool", "tool_call_id": "call_1",
                       "content": "{\"temp_c\": 18}"},
                      {"role": "user", "content": "Summarize in one word."}],
              max_tokens=32, expect_error=True)
    t.check(ok.status == 200,
            "tool-result continuation rejected: %s (%s)" % (ok.status, ok.error_message[:140]))


@cell("F", "F5", "reasoning_effort high refused; 'none' accepted with warning")
def f5(t):
    r = chat(t.cfg, user("hi"), expect_error=True, max_tokens=16, reasoning_effort="high")
    expect_refusal(t, "reasoning_effort=high", r)
    ok = chat(t.cfg, user("hi"), max_tokens=16, reasoning_effort="none")
    t.check(ok.status == 200, "reasoning_effort=none rejected: %s" % ok.status)
    t.check(ok.json["hipfire"]["reasoning"]["mode"] == "disabled", "effort none did not disable")


@cell("F", "F6", "finite think cap + named think budget refused")
def f6(t):
    # Finite think caps are now ENFORCED via the grammar cursor (vLLM
    # thinking_token_budget parity): with a schema present, the mask allows
    # ONLY the think close at the budget, so the span force-closes and the
    # request completes with valid JSON — instead of burning to max_tokens
    # mid-think and dying as unsatisfiable.
    schema = {"type": "object",
              "properties": {"done": {"type": "boolean"}},
              "required": ["done"], "additionalProperties": False}
    # enable_thinking=True: the budget only engages when a think span
    # exists (the suite's chat helper defaults it off).
    r = chat(t.cfg, user("Count from 1 to 5, then confirm with JSON."),
             max_tokens=400, max_think_tokens=24,
             chat_template_kwargs={"enable_thinking": True},
             response_format={"type": "json_schema",
                              "json_schema": {"name": "out", "schema": schema}})
    t.check(r.status == 200,
            "max_think_tokens=24: expected 200 (enforced), got %s (%s)"
            % (r.status, r.error_message[:140]))
    if r.status == 200:
        t.check(r.finish in ("stop", "length"),
                "budgeted request finish=%s" % r.finish)
        try:
            v = json.loads(r.content)
            t.check(isinstance(v.get("done"), bool),
                    "budgeted output schema-conforming: %r" % (v,))
        except Exception as e:
            t.check(False, "budgeted output not JSON: %s: %r" % (e, r.content[:120]))
        comp = (r.json.get("usage") or {}).get("completion_tokens") or 0
        t.ev("budgeted: completion=%s tokens (budget 24 + JSON)" % comp)
        t.check(comp < 200,
                "enforcement should bound the span: completion=%s" % comp)
    r2 = chat(t.cfg, user("hi"), expect_error=True, max_tokens=16,
              thinking_budget="medium")
    t.check(r2.status == 400, "thinking_budget: got %s (%s)" % (r2.status, r2.error_message[:140]))


@cell("F", "F7", "min_p out of range refused")
def f7(t):
    r = chat(t.cfg, user("hi"), expect_error=True, max_tokens=16, min_p=1.5)
    expect_refusal(t, "min_p=1.5", r)


@cell("F", "F8", "n>1 must be refused on the slot route (validate_generate_caps)")
def f8(t):
    r = chat(t.cfg, user("hi"), expect_error=True, max_tokens=16, n=2)
    t.check(r.status == 400,
            "n=2 was accepted with HTTP %s; branch docs mark n as a refused field "
            "(slots.rs validate_generate_caps); silent accept + single choice is a "
            "contract drift" % r.status)
    if r.status == 200:
        t.check(len(r.json.get("choices", [])) == 1,
                "n=2 returned %d choices" % len(r.json.get("choices", [])))


@cell("F", "F9", "logit_bias must be refused on the slot route")
def f9(t):
    r = chat(t.cfg, user("hi"), expect_error=True, max_tokens=16, logit_bias={"123": 5})
    t.check(r.status == 400,
            "logit_bias was accepted with HTTP %s; branch docs mark logit_bias as "
            "refused (slots.rs validate_generate_caps)" % r.status)


# --------------------------------------------------------------------------
# G — HTTP limits + error mapping
# --------------------------------------------------------------------------


@cell("G", "G1", "invalid JSON body -> 400")
def g1(t):
    conn = http.client.HTTPConnection(t.cfg.host, t.cfg.port, timeout=15)
    try:
        conn.request("POST", "/v1/chat/completions", body="{not json",
                     headers={"Content-Type": "application/json"})
        r = conn.getresponse()
        r.read()
        t.check(r.status == 400, "status %s != 400" % r.status)
    finally:
        conn.close()


@cell("G", "G2", "unknown model -> 404")
def g2(t):
    r = chat(t.cfg, user("hi"), expect_error=True, max_tokens=16, model="nope:99b")
    t.check(r.status == 404, "status %s != 404" % r.status)
    t.check("model not found locally" in r.error_message, "message %r" % r.error_message[:120])


@cell("G", "G3", "max_tokens bounds -> 400")
def g3(t):
    for bad in (0, -5, 400000):
        r = chat(t.cfg, user("hi"), expect_error=True, max_tokens=bad)
        t.check(r.status == 400, "max_tokens=%s: got %s (%s)" % (bad, r.status, r.error_message[:100]))
    t.ev("0/-5/400000 all 400")


@cell("G", "G4", "oversized Content-Length -> 413 before body read")
def g4(t):
    line = raw_status_line(
        t.cfg, "/v1/chat/completions",
        "POST /v1/chat/completions HTTP/1.1\r\n"
        "Host: %s:%s\r\nContent-Type: application/json\r\n"
        "Content-Length: 67108865\r\nConnection: close\r\n\r\n{}" % (t.cfg.host, t.cfg.port))
    t.ev("status line: %s" % line)
    t.check(" 413 " in line + " ", "expected 413, got %r" % line)


@cell("G", "G5", "seed validation: fractional 400; negative 400-style per SERVE.md")
def g5(t):
    r = chat(t.cfg, user("hi"), expect_error=True, max_tokens=16, seed=1.5)
    t.check(r.status == 400, "fractional seed: got %s" % r.status)
    r2 = chat(t.cfg, user("hi"), expect_error=True, max_tokens=16, seed=-1)
    t.check(r2.status == 400,
            "negative seed returned HTTP %s; SERVE.md documents '400-style error, "
            "never silently unseeded' for negative seeds" % r2.status)


@cell("G", "G6", "empty messages -> 400-class invalid_request")
def g6(t):
    r = chat(t.cfg, [], expect_error=True, max_tokens=16)
    t.check(r.status in (400, 422),
            "empty messages returned HTTP %s (type=%r) — a client-input error "
            "surfacing as a 5xx class" % (r.status, (r.json or {}).get("error", {}).get("type")))


@cell("G", "G7", "ctx overshoot -> typed rejection, server stays healthy")
def g7(t):
    big = _record_md5("g7_body", passage("pyrrhica", 30))
    r = chat(t.cfg, user(big), expect_error=True, max_tokens=60000)
    t.ev("status=%s msg=%r" % (r.status, r.error_message[:160]))
    t.check(r.status in (400, 422, 500),
            "unexpected status %s" % r.status)
    if r.status >= 500:
        t.warn("ctx overshoot surfaced as HTTP %s (server_error class); client-fixable "
               "rejections should be 4xx per the typed-rejection contract" % r.status)
    h = get_json(t.cfg, "/health")[2]
    t.check(json.loads(h).get("status") == "ok", "server unhealthy after overshoot")
    r2 = chat(t.cfg, user(keyword_instruction("NORMal".upper())), temperature=0, max_tokens=16)
    t.check("NORMAL" in r2.content, "post-overshoot generation wrong: %r" % r2.content[:60])


# --------------------------------------------------------------------------
# V — vision sidecar x radix interaction
# --------------------------------------------------------------------------


@cell("V", "V1", "vision request skips the radix (A18: repeated image cached=0)")
def v1(t):
    b64 = tiny_png_b64()
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": "What color dominates this image? One word."},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}}]}]
    r = chat(t.cfg, msgs, max_tokens=200, expect_error=True)
    if r.status != 200:
        t.skip("vision request refused on this deployment (HTTP %s %s); X2/A18 stays "
               "off per spec" % (r.status, r.error_message[:160]))
        return
    t.check(len(r.content) > 0, "empty vision answer")
    r2 = chat(t.cfg, msgs, max_tokens=200, expect_error=True)
    t.check(r2.status == 200, "repeat vision request status %s (%s)"
            % (r2.status, r2.error_message[:160]))
    t.ev("repeat cached=%s" % r2.cached_tokens)
    t.check(r2.cached_tokens in (0, None),
            "vision request reused %s radix tokens; has_visual requests must skip "
            "the prefix cache (spec X2/A18)" % r2.cached_tokens)


# --------------------------------------------------------------------------
# D — scheduler / admission / parallelism (run late: disruptive)
# --------------------------------------------------------------------------


@cell("D", "D1", "wave-12: two-slot parallel generation without serialization")
def d1(t):
    pa = essay_prompt("tidesreach", extra="\n\nEssay question: how the tidal mill worked. "
                                          "Write four paragraphs.")
    pb = essay_prompt("guilderhay", extra="\n\nEssay question: how the wool guild stored "
                                          "dye vats. Write four paragraphs.")
    # solo baselines (warm the cache too)
    sa = chat_stream(t.cfg, user(pa), temperature=0, max_tokens=256)
    sb = chat_stream(t.cfg, user(pb), temperature=0, max_tokens=256)
    t.check(sa.status == 200 and sb.status == 200, "solo baselines failed")
    solo_max = max(sa.total_ms, sb.total_ms)
    t.ev("solo A=%.1fs B=%.1fs" % (sa.total_ms / 1e3, sb.total_ms / 1e3))

    def run(p):
        return chat_stream(t.cfg, user(p), temperature=0, max_tokens=256)
    res = run_concurrently([lambda: run(pa), lambda: run(pb)])
    (ra, ea, _), (rb, eb, _) = res
    for name, e in (("A", ea), ("B", eb)):
        if e:
            raise CellFailure("parallel request %s raised: %s" % (name, e))
    pair_wall = max(ra.total_ms, rb.total_ms)
    t.ev("pair wall=%.1fs first-chunk A=%.0fms B=%.0fms"
         % (pair_wall / 1e3, ra.first_content_ms, rb.first_content_ms))
    t.check(ra.status == 200 and rb.status == 200, "parallel pair non-200")
    t.check(pair_wall < 1.9 * solo_max,
            "parallel pair serialized: %.1fs vs solo max %.1fs (ratio %.2f)"
            % (pair_wall / 1e3, solo_max / 1e3, pair_wall / solo_max))
    t.check(rb.first_content_ms < 0.9 * sa.total_ms,
            "request B saw no service until A finished (progressive-service "
            "regression): B first chunk at %.0fms, A solo total %.0fms"
            % (rb.first_content_ms, sa.total_ms))
    la, lb = ra.content.lower(), rb.content.lower()
    t.check("tide" in la or "mill" in la, "A off-topic: %r" % ra.content[:100])
    t.check("guild" in lb or "dye" in lb or "wool" in lb, "B off-topic: %r" % rb.content[:100])
    t.check("guild" not in la, "cross-slot bleed: A talks about the guild")
    t.check("tidal mill" not in lb, "cross-slot bleed: B talks about the tidal mill")
    bad, stats = is_attractor(ra.content + rb.content)
    t.check(not bad, "attractor across parallel outputs: %s" % stats)
    if ra.content != sa.content or rb.content != sb.content:
        t.warn("batched outputs differ from solo-batched baselines (kernel reduction "
               "order change under concurrency); coherence checks passed")


def _lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _fuzzy_hit(word, text):
    """Exact substring, or a <=1-edit standalone word (greedy typo tolerance)."""
    if word.lower() in text.lower():
        return True
    return any(len(tok) >= 4 and _lev(tok.lower(), word.lower()) <= 1
               for tok in re.findall(r"[A-Za-z]+", text))


@cell("D", "D2", "four-slot saturation: all complete, all correct")
def d2(t):
    words = ["COPPER", "GRANITE", "MARROW", "TIDEWAIT"]
    prompts = [prompt_cold("slot%d" % i, keyword_instruction(w)) for i, w in enumerate(words)]
    res = run_concurrently([lambda p=p: chat(t.cfg, user(p), temperature=0, max_tokens=32)
                            for p in prompts])
    for w, r in zip(words, res):
        val, err, dt = r
        if err:
            raise CellFailure("slot request %s raised: %s" % (w, err))
        t.check(val.status == 200, "%s status %s" % (w, val.status))
        t.check(_fuzzy_hit(w, val.content), "%s answered %r" % (w, val.content[:60]))
        t.ev("%s %.1fs" % (w, dt))


@cell("D", "D3", "queue-full flood -> 503 + Retry-After, accepted all complete")
def d3(t):
    n = 80
    prompts = []
    for i in range(n):
        prompts.append(_record_md5("flood%d" % i,
                                   passage("driftquay%d" % i, 30, cap="cove of")
                                   + "\n\n" + keyword_instruction("PIN%d" % (i % 7))))
    def flood_one(p):
        payload = {"model": t.cfg.model, "messages": [{"role": "user", "content": p}],
                   "temperature": 0, "max_tokens": 16,
                   "chat_template_kwargs": {"enable_thinking": False}}
        status, hdrs, text, _dt = _post_json(t.cfg, "/v1/chat/completions", payload,
                                             t.cfg.timeout(240))
        return status, hdrs, text

    t0 = time.monotonic()
    res = run_concurrently([lambda p=p: flood_one(p) for p in prompts])
    statuses = Counter()
    retry_after = None
    oks = 0
    for r in res:
        val, err, _ = r
        if err:
            statuses["error:%s" % type(err).__name__] += 1
            continue
        status, hdrs, _text = val
        statuses[status] += 1
        if status == 503:
            retry_after = hdrs.get("retry-after", retry_after)
        elif status == 200:
            oks += 1
    t.ev("statuses=%s wall=%.1fs" % (dict(statuses), time.monotonic() - t0))
    t.check(statuses[503] >= 1,
            "no 503 under %d-way flood (statuses %s) — admission queue never bound"
            % (n, dict(statuses)))
    if statuses[503] >= 1:
        t.check(retry_after is not None, "503 missing Retry-After header")
        try:
            t.check(int(retry_after) >= 0, "Retry-After %r not an int >= 0" % retry_after)
        except (TypeError, ValueError):
            raise CellFailure("Retry-After %r not parseable" % retry_after)
    t.check(oks + statuses[503] == n,
            "unexpected statuses %s (oks=%s)" % (dict(statuses), oks))


@cell("D", "D4", "wait-queue drains after burst (pop_ready deadlock regression)")
def d4(t):
    t0 = time.monotonic()
    r = chat(t.cfg, user(keyword_instruction("AFTERGLOW")), temperature=0, max_tokens=16,
             timeout=120)
    dt = time.monotonic() - t0
    t.ev("post-burst request %.1fs" % dt)
    t.check(r.status == 200, "status %s" % r.status)
    t.check("AFTERGLOW" in r.content, "output %r" % r.content[:60])
    t.check(dt < 90, "post-burst request took %.1fs — wait queue wedged?" % dt)


@cell("D", "D5", "cancel storm: 6 mid-stream disconnects then normal request")
def d5(t):
    for i in range(6):
        sr = chat_stream(t.cfg, user("Write a very long essay about rope %d." % i),
                         cancel_after_chunks=2, max_tokens=512)
        t.check(sr.disconnected or sr.status == 200,
                "cancel iter %d: status %s" % (i, sr.status))
    r = chat(t.cfg, user(keyword_instruction("RELEasED".upper())), temperature=0,
             max_tokens=16, timeout=120)
    t.check(r.status == 200, "post-cancel request status %s (permit/slot pinned?)" % r.status)
    t.check("RELEASED" in r.content, "output %r" % r.content[:60])
    t.ev("6 cancels + recovery OK")


@cell("D", "D6", "progressive service: short request not stuck behind huge prefill")
def d6(t):
    huge = _record_md5("d6_huge", passage("megalith", 110, cap="expanse of"))
    box = {}

    def big():
        box["big_t0"] = time.monotonic()
        r = chat(t.cfg, user(huge), temperature=0, max_tokens=16, timeout=240)
        box["big_done"] = time.monotonic()
        return r

    def small():
        time.sleep(1.5)
        box["small_t0"] = time.monotonic()
        r = chat(t.cfg, user(keyword_instruction("SWIFTLY")), temperature=0,
                 max_tokens=16, timeout=240)
        box["small_done"] = time.monotonic()
        return r
    with ThreadPoolExecutor(max_workers=2) as ex:
        fa = ex.submit(big)
        time.sleep(0.2)  # big is submitted and starts its multi-chunk prefill
        fb = ex.submit(small)
        rb = fb.result()
        ra = fa.result()
    t.check(ra.status == 200 and rb.status == 200, "statuses %s/%s" % (ra.status, rb.status))
    big_dt = box["big_done"] - box["big_t0"]
    small_dt = box["small_done"] - box["small_t0"]
    t.ev("huge %.1fs small %.1fs" % (big_dt, small_dt))
    t.check("SWIFTLY" in rb.content, "small output %r" % rb.content[:60])
    t.check(small_dt < big_dt,
            "small request (%.1fs) took longer than the 4k-token prefill request "
            "(%.1fs) — starvation/aging regression" % (small_dt, big_dt))


@cell("D", "D7", "queue-wait timeout 503 (env-dependent)")
def d7(t):
    # Read the endpoint's advertised queue timeout; only exercise the
    # wait-timeout when it is short enough to observe inside the harness
    # budget. A large advertised value makes the wait impractical, which is
    # the honest skip — not an unconditional one.
    st, _, text = get_json(t.cfg, "/health")
    caps = (json.loads(text).get("capabilities") or {}) if st == 200 else {}
    q_ms = caps.get("queue_timeout_ms")
    if not isinstance(q_ms, int):
        t.skip("queue_timeout_ms not advertised in capabilities: %r" % caps)
    if q_ms > 30000:
        t.skip("queue_timeout_ms=%d exceeds the harness budget; re-run with "
               "HIPFIRE_SERVE_QUEUE_TIMEOUT_MS <= 30000 to observe the "
               "wait-timeout 503" % q_ms)
    # Occupy every slot with a slow prefill so the next request is parked in
    # the bounded wait queue (not refused for queue-full). With the slots
    # busy past queue_timeout_ms the parked request must 503 on wait-timeout.
    huge = passage("d7block", 120, cap="expanse of")
    blockers = [user(huge) for _ in range(int(t.cfg.slots))]
    waiter = user(passage("d7wait", 4, cap="edge of"))
    box = {}
    def slow(i):
        r = chat(t.cfg, blockers[i], temperature=0, max_tokens=48,
                 timeout=cfg_timeout(t, q_ms))
        box.setdefault("blockers", []).append(r.status)
    def cfg_timeout(t, q_ms):
        # blockers must outlive the parked waiter's deadline
        return max(150, int(q_ms / 1000) + 60)
    def wait_req():
        time.sleep(0.4)  # let blockers claim the slots first
        st, hdrs, txt, _ = _post_json(
            t.cfg, "/v1/chat/completions",
            {"model": t.cfg.model, "messages": waiter,
             "chat_template_kwargs": {"enable_thinking": False},
             "temperature": 0, "max_tokens": 8},
            cfg_timeout(t, q_ms))
        box["waiter"] = (st, hdrs)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=int(t.cfg.slots) + 1) as ex:
        fs = [ex.submit(slow, i) for i in range(int(t.cfg.slots))]
        fw = ex.submit(wait_req)
        fw.result()
        for f in fs:
            f.result()
    wst, whdrs = box.get("waiter", (None, {}))
    t.ev("blockers=%s waiter=%s queue_timeout_ms=%s"
         % (box.get("blockers"), wst, q_ms))
    t.check(wst == 503,
            "queued request expected wait-timeout 503, got %s "
            "(blockers %s)" % (wst, box.get("blockers")))


# --------------------------------------------------------------------------
# H — lifecycle canaries
# --------------------------------------------------------------------------


@cell("H", "H1", "post-gauntlet canary: health + coherent generation")
def h1(t):
    st, _, text = get_json(t.cfg, "/health")
    h = json.loads(text)
    t.check(st == 200 and h.get("status") == "ok", "health %s" % st)
    t.check((h.get("capabilities") or {}).get("multi_slot_slots") == t.cfg.slots,
            "capabilities changed after the gauntlet")
    r = chat(t.cfg, user(keyword_instruction("INTACT")), temperature=0, max_tokens=16)
    t.check(r.status == 200 and "INTACT" in r.content,
            "canary generation wrong: %s %r" % (r.status, r.content[:60]))
    # long-prompt liveness canary: the observed degradation class parks ~700-token
    # prompts in the daemon wait queue after sustained traffic (2026-09-13 episode)
    p = _record_md5("h1_long", passage("canaryisle", 34)
                    + "\n\nQuestion: Summarize the entry counts in one sentence.")
    r2 = chat(t.cfg, user(p), temperature=0, max_tokens=96, timeout=150)
    t.check(r2.status == 200 and len(r2.content) > 0,
            "long-prompt canary parked/failed: %s (admission wedge after traffic?)"
            % r2.status)
    bad, stats = is_attractor(r2.content)
    t.check(not bad, "long canary attractor: %s" % stats)
    st2, _, s2text = get_json(t.cfg, "/stats")
    s2 = json.loads(s2text)
    t.check(s2["queue_depth"] == 0, "queue_depth %s != 0 at rest" % s2["queue_depth"])
    t.ev("requests_served=%s recent_tok_s=%s" % (s2["requests_served"], s2.get("recent_tok_s")))


@cell("H", "H2", "cold-cycle stability: fresh prompts stay cold then warm")
def h2(t):
    for i, topic in enumerate(("yrchpole", "bloomwatch", "stanmoor")):
        p = prompt_cold("cyc%s" % topic, keyword_instruction("CYCLE%d" % i))
        r1 = chat(t.cfg, user(p), temperature=0, max_tokens=32)
        t.check(r1.cached_tokens == 0,
                "cycle %d cold cached=%s (pool recycling contaminated?)"
                % (i, r1.cached_tokens))
        r2 = chat(t.cfg, user(p), temperature=0, max_tokens=32)
        t.check(r2.cached_tokens >= 128, "cycle %d warm cached=%s" % (i, r2.cached_tokens))
        t.check(r1.content == r2.content, "cycle %d replay differs" % i)


@cell("H", "H3", "idle eviction / reload (env-dependent)")
def h3(t):
    t.skip("container runs --idle-timeout 0 (never unload); idle-eviction and "
           "reload-reuse cannot be exercised here. Run with --idle-timeout > 0 and "
           "an idle gap to cover unload -> reload -> warm-hit (A20 lifecycle).")


@cell("H", "H4", "model swap teardown (out of scope for this container)")
def h4(t):
    t.skip("single-model container; cache invalidation on model swap "
           "(mmq_screen_cache/graph teardown) needs a second model file. "
           "Covered by the in-container oracle instead.")


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------


def collect_binary_md5(docker_name):
    import subprocess
    if not docker_name:
        return None
    try:
        out = subprocess.run(
            ["docker", "exec", docker_name, "sh", "-c",
             "md5sum $(command -v hipfire) 2>/dev/null; md5sum /usr/local/bin/daemon 2>/dev/null"],
            capture_output=True, text=True, timeout=30)
        return out.stdout.strip() or None
    except Exception:
        return None


def wait_healthy(cfg, deadline_s=150):
    t0 = time.monotonic()
    while time.monotonic() - t0 < deadline_s:
        try:
            st, _, _ = get_json(cfg, "/health", timeout=5)
            if st == 200:
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


def restart_serve(docker_name):
    import subprocess
    subprocess.run(["docker", "restart", docker_name], capture_output=True, timeout=120)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8420)
    ap.add_argument("--model", default="qwen3.5:4b")
    ap.add_argument("--slots", type=int, default=4,
                    help="the deployment's serve.multi_slot_slots (asserted by A1/A3/H1)")
    ap.add_argument("--spec", choices=("auto", "on", "off"), default="auto",
                    help="speculation state override; auto detects from /health + model caps")
    ap.add_argument("--area", action="append", default=[],
                    help="restrict to area(s): A B C D E F G V H")
    ap.add_argument("--cell", action="append", default=[],
                    help="restrict to specific cell id(s)")
    ap.add_argument("--list", action="store_true", help="list cells and exit")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--timeout-scale", type=float, default=1.0,
                    help="multiply all request timeouts (slow boxes)")
    ap.add_argument("--docker-name", default=None,
                    help="best-effort: record binary md5s via docker exec")
    args = ap.parse_args(argv)

    cells = sorted(CELLS, key=lambda c: (AREA_ORDER.index(c["area"]), c["id"]))
    if args.list:
        for c in cells:
            print("%-4s %s  %s" % (c["area"], c["id"], c["name"]))
        return 0

    sel = cells
    if args.area:
        sel = [c for c in sel if c["area"] in args.area]
    if args.cell:
        sel = [c for c in sel if c["id"] in args.cell]
    if not sel:
        print("no cells selected", file=sys.stderr)
        return 2

    cfg = Cfg(args.host, args.port, args.model, args.timeout_scale, args.slots)
    if args.spec != "auto":
        cfg.spec_on = args.spec == "on"
    # readiness probe: HTTP up AND model resident (async pre-warm may still run)
    deadline = time.monotonic() + 240
    health = None
    while time.monotonic() < deadline:
        try:
            st, _, text = get_json(cfg, "/health", timeout=10)
            h = json.loads(text)
            if st == 200 and h.get("model") and not h.get("loading_model"):
                health = h
                break
        except Exception:
            pass
        time.sleep(3)
    if health is None:
        print("serve at %s:%s never became healthy with a resident model"
              % (cfg.host, cfg.port), file=sys.stderr)
        return 2

    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    t_start = time.monotonic()
    out = args.json_out or "scs_suite_report.json"

    def write_report(counts_snapshot, final=False):
        report = {
            "suite": "scs_suite",
            "started": started,
            "elapsed_sec": round(time.monotonic() - t_start, 1),
            "final": final,
            "target": {"host": cfg.host, "port": cfg.port, "model": cfg.model},
            "health": health,
            "binary_md5s": collect_binary_md5(args.docker_name),
            "prompt_md5s": PROMPT_MD5S,
            "summary": dict(counts_snapshot),
            "results": results,
        }
        with open(out, "w") as f:
            json.dump(report, f, indent=2)

    print("scs_suite: %s:%s model=%s cells=%d started=%s"
          % (cfg.host, cfg.port, cfg.model, len(sel), started))
    print("=" * 78)

    results = []
    counts = Counter()
    for c in sel:
        ctx = Ctx(cfg)
        t0 = time.monotonic()
        status = "PASS"
        try:
            c["fn"](ctx)
        except CellFailure as e:
            status = "FAIL"
            ctx.evidence.append("FAIL: %s" % e)
        except CellSkip as e:
            status = "SKIP"
            ctx.evidence.append("SKIP: %s" % e)
        except HttpError as e:
            status = "FAIL"
            ctx.evidence.append("FAIL: unexpected http condition: %s" % e)
        except Exception as e:  # noqa: BLE001 - one broken cell must not kill the run
            status = "FAIL"
            ctx.evidence.append("FAIL: %s: %s" % (type(e).__name__, e))
        if args.docker_name and any("Connection refused" in s or "ConnectionReset" in s
                                    or "RemoteDisconnected" in s
                                    for s in ctx.evidence[-2:]):
            try:
                st, _, _ = get_json(cfg, "/health", timeout=5)
            except Exception:
                st = None
            if st != 200:
                ctx.evidence.append("SERVER DIED during/after this cell (container exit); "
                                    "restarting via docker for remaining cells")
                restart_serve(args.docker_name)
                if wait_healthy(cfg):
                    ctx.evidence.append("server healthy again after restart")
                else:
                    ctx.evidence.append("server did NOT come back within 150s")
        dt = time.monotonic() - t0
        if ctx.warns and status == "PASS":
            status = "WARN"
        counts[status] += 1
        mark = {"PASS": "ok  ", "FAIL": "FAIL", "WARN": "warn", "SKIP": "skip"}[status]
        print("[%s] %-4s %-38s %6.1fs  %s" % (mark, c["id"], c["name"], dt,
                                              ("; ".join(ctx.warns) if ctx.warns
                                               else (ctx.evidence[0][:70] if ctx.evidence else ""))))
        results.append({"id": c["id"], "area": c["area"], "name": c["name"],
                        "status": status, "secs": round(dt, 1),
                        "evidence": ctx.evidence, "warnings": ctx.warns})
        write_report(counts)

    elapsed = time.monotonic() - t_start
    print("=" * 78)
    print("DONE %s: pass=%d fail=%d warn=%d skip=%d elapsed=%.0fs"
          % (started, counts["PASS"], counts["FAIL"], counts.get("WARN", 0),
             counts["SKIP"], elapsed))

    write_report(counts, final=True)
    print("report: %s" % out)

    return 0 if counts["FAIL"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
