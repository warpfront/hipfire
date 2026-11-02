#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Freeze a reproducible token corpus (``hipfire.qwen4.kl_corpus.v1``) for KL runs.

CPU-only, no network, no generation.  Real reference texts / assistant answers
are tokenized with the pinned local Qwen fast tokenizer and chat template and
written as one JSON file::

    {"schema": "hipfire.qwen4.kl_corpus.v1", "name": ..., "seed": ...,
     "sequences": [{"id", "stratum": "text"|"assistant", "tokens",
                    "score_positions", "reset": true, "role_mask"?, "meta"}],
     "meta": {"provenance": {...}}}

Scoring contract: position ``p`` is an *input* position and is scored against
target ``tokens[p + 1]``; the last token is therefore never a scored input.
``text`` sequences score every ``p``.  ``assistant`` sequences carry a
per-token ``role_mask`` (1 only for assistant *content* tokens) and score only
the ``p`` whose target ``p + 1`` has ``role_mask == 1`` -- role delimiters
(``<|im_start|>assistant``, ``<|im_end|>``), template thinking scaffolds and
user/system tokens are never targets.  Every sequence is an independent reset.

Input JSONL (``--input``), one object per line::

    {"id": "x1", "stratum": "text", "text": "..."}
    {"id": "x2", "stratum": "assistant",
     "messages": [{"role": "system"|"user"|"assistant", "content": "..."}, ...]}

``--human-eval-input`` adds the public HumanEval JSONL as two proxy pools:
``prompt`` as a ``text`` sequence and ``prompt -> canonical_solution`` as an
``assistant`` sequence labelled ``meta.proxy ==
"humaneval_canonical_solution_code_only"`` (a code-only reference completion,
NOT a natural chat answer).

The assistant mask comes from the *actual rendered chat template*: each
assistant turn's trimmed content is located as the suffix of the rendered
assistant segment (the template trims whitespace, so the original string need
not be present), the scaffold before it must be only whitespace / an empty
``<think></think>`` block, and tokens are classified by the fast tokenizer's
offset mapping.  Any ambiguity (token straddling the content boundary,
unexpected scaffold text, special tokens inside content, missing content) makes
the *record* rejected -- it is never scored with a guessed role.

Example (pilot: 4 text x128 + 4 assistant x256)::

    python quality_corpus.py --tokenizer-dir /path/to/Qwen3.8-Flash-Next \
        --input corpus.jsonl --human-eval-input ~/.hipfire/datasets/HumanEval.jsonl \
        --max-input-tokens 128 --assistant-max-input-tokens 256 \
        --units-per-stratum 4 --seed 0 --out corpus.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

SCHEMA = "hipfire.qwen4.kl_corpus.v1"
PINNED_TOKENIZER_SOURCE = "Qwen/Qwen3.8-Flash-Next"
PINNED_TOKENIZER_REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
STRATA = ("text", "assistant")

ASSISTANT_MARKER = "<|im_start|>assistant"
TURN_END = "<|im_end|>"
# Everything the template may place between the assistant marker and the
# content: a newline and (for this Qwen template) an *empty* think block.
_SCAFFOLD_RE = re.compile(r"\s*(?:<think>\s*</think>\s*)?")
_ROLES = ("system", "user", "assistant")

HUMANEVAL_PROXY = "humaneval_canonical_solution_code_only"
HUMANEVAL_TEXT_PROXY = "humaneval_prompt_code_text"
HUMANEVAL_USER_PREFIX = "Complete the following Python function.\n\n"

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "chat_template.json",
    "generation_config.json",
)

MAX_REPORTED_REJECTIONS = 200


class CorpusError(Exception):
    """Fatal problem: bad input shape, bad tokenizer, invalid corpus."""


class Rejected(CorpusError):
    """One candidate record cannot be scored unambiguously; skip and report."""


# --------------------------------------------------------------------------- #
# Hashing / IO
# --------------------------------------------------------------------------- #
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fsync_dir(directory: Path) -> None:
    fd = os.open(str(directory), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_durable(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically and fsync file + directory."""
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with open(tmp, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise
    _fsync_dir(path.parent)


# --------------------------------------------------------------------------- #
# Tokenizer adapter
# --------------------------------------------------------------------------- #
class TokenizerAdapter:
    """Thin wrapper over a HF *fast* tokenizer (or a test double of that surface).

    Uses only ``apply_chat_template(..., tokenize=False)``, ``__call__`` with
    ``return_offsets_mapping``, ``all_special_tokens`` and ``len()``.
    """

    def __init__(self, tokenizer: Any, chat_template_kwargs: dict[str, Any] | None = None) -> None:
        self._tok = tokenizer
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.vocab_size = int(len(tokenizer))
        self.special_tokens = sorted({t for t in tokenizer.all_special_tokens if t})
        for marker in ("<|im_start|>", TURN_END):
            if marker not in self.special_tokens:
                raise CorpusError(f"tokenizer does not declare {marker!r} as a special token")

    def render_chat(self, messages: Sequence[dict[str, str]]) -> str:
        rendered = self._tok.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=False, **self.chat_template_kwargs
        )
        if not isinstance(rendered, str):
            raise CorpusError("apply_chat_template(tokenize=False) did not return a string")
        return rendered

    def encode(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        enc = self._tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = [int(i) for i in enc["input_ids"]]
        offsets = [(int(s), int(e)) for s, e in enc["offset_mapping"]]
        if len(ids) != len(offsets):
            raise CorpusError("tokenizer returned mismatched ids/offsets")
        prev = 0
        for (start, end), tid in zip(offsets, ids):
            if not (0 <= start <= end <= len(text)):
                raise CorpusError(f"token offset {(start, end)} outside text of length {len(text)}")
            if not (0 <= tid < self.vocab_size):
                raise CorpusError(f"token id {tid} outside vocabulary of size {self.vocab_size}")
            if start < prev and start != end:
                # Offsets of consecutive tokens must be non-decreasing in start.
                raise CorpusError("tokenizer offsets are not monotone")
            if start != end:
                prev = start
        return ids, offsets


def _tokenizer_provenance(
    directory: Path, tokenizer: Any, *, source: str, revision: str, strict_revision: bool,
    expect_tokenizer_json_sha256: str | None,
) -> dict[str, Any]:
    files: dict[str, str] = {}
    for name in TOKENIZER_FILES:
        path = directory / name
        if path.is_file():
            files[name] = sha256_file(path)
    if "tokenizer.json" not in files:
        raise CorpusError(f"{directory}: tokenizer.json is required (fast tokenizer)")
    if expect_tokenizer_json_sha256 and files["tokenizer.json"] != expect_tokenizer_json_sha256.lower():
        raise CorpusError(
            f"tokenizer.json sha256 {files['tokenizer.json']} != expected {expect_tokenizer_json_sha256}"
        )
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str) or not template.strip():
        raise CorpusError("tokenizer has no chat_template string")
    template_source = "chat_template.jinja" if "chat_template.jinja" in files else (
        "chat_template.json" if "chat_template.json" in files else "tokenizer_config.json")

    evidence: list[str] = []
    resolved = directory.resolve()
    if revision in resolved.parts:
        evidence.append("path_component")
    meta_path = resolved / ".cache" / "huggingface" / "download" / "tokenizer.json.metadata"
    if meta_path.is_file():
        first = meta_path.read_text(encoding="utf-8", errors="replace").splitlines()[:1]
        commit = first[0].strip() if first else ""
        if commit and commit != revision:
            raise CorpusError(f"tokenizer download metadata commit {commit} != pinned revision {revision}")
        if commit == revision:
            evidence.append("hf_download_metadata")
    if strict_revision and not evidence:
        raise CorpusError(
            "no on-disk evidence that the tokenizer directory is revision "
            f"{revision} (path component / hf download metadata); pass --expect-tokenizer-json-sha256 "
            "or drop --strict-revision"
        )
    return {
        "source": source,
        "revision": revision,
        "revision_check": evidence if evidence else "declared_only",
        "dir_name": resolved.name,
        "files_sha256": files,
        "chat_template_sha256": sha256_text(template),
        "chat_template_source": template_source,
        "is_fast": True,
    }


def load_tokenizer(
    directory: str | Path,
    *,
    source: str = PINNED_TOKENIZER_SOURCE,
    revision: str = PINNED_TOKENIZER_REVISION,
    strict_revision: bool = False,
    expect_tokenizer_json_sha256: str | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> tuple[TokenizerAdapter, dict[str, Any]]:
    """Load the pinned tokenizer from a local directory only (never the network)."""
    path = Path(directory)
    if not path.is_dir():
        raise CorpusError(f"--tokenizer-dir is not a directory: {path}")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        from transformers import AutoTokenizer  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on rental env
        raise CorpusError(f"transformers is required to load the tokenizer: {exc}") from exc
    tok = AutoTokenizer.from_pretrained(
        str(path), local_files_only=True, use_fast=True, trust_remote_code=False
    )
    if not getattr(tok, "is_fast", False):
        raise CorpusError("loaded tokenizer is not a fast tokenizer (offset mapping unavailable)")
    provenance = _tokenizer_provenance(
        path, tok, source=source, revision=revision, strict_revision=strict_revision,
        expect_tokenizer_json_sha256=expect_tokenizer_json_sha256,
    )
    adapter = TokenizerAdapter(tok, chat_template_kwargs)
    provenance["vocab_size"] = adapter.vocab_size
    provenance["chat_template_kwargs"] = adapter.chat_template_kwargs
    return adapter, provenance


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Record:
    stratum: str
    source: str
    id: str
    text: str | None = None
    messages: tuple[tuple[str, str], ...] | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def _iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CorpusError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            if not isinstance(obj, dict):
                raise CorpusError(f"{path}:{lineno}: record must be a JSON object")
            yield lineno, obj


def load_general_records(path: Path, source: str) -> list[Record]:
    records: list[Record] = []
    for lineno, obj in _iter_jsonl(path):
        where = f"{path}:{lineno}"
        rid = obj.get("id")
        if not isinstance(rid, str) or not rid:
            raise CorpusError(f"{where}: 'id' must be a non-empty string")
        stratum = obj.get("stratum")
        if stratum == "text":
            if not isinstance(obj.get("text"), str) or "messages" in obj:
                raise CorpusError(f"{where}: text record needs string 'text' and no 'messages'")
            records.append(Record("text", source, rid, text=obj["text"]))
        elif stratum == "assistant":
            messages = obj.get("messages")
            if not isinstance(messages, list) or "text" in obj:
                raise CorpusError(f"{where}: assistant record needs 'messages' list and no 'text'")
            pairs: list[tuple[str, str]] = []
            for message in messages:
                if (not isinstance(message, dict) or set(message) != {"role", "content"}
                        or not isinstance(message["role"], str) or not isinstance(message["content"], str)):
                    raise CorpusError(f"{where}: each message must be exactly {{role, content}} strings")
                pairs.append((message["role"], message["content"]))
            records.append(Record("assistant", source, rid, messages=tuple(pairs)))
        else:
            raise CorpusError(f"{where}: 'stratum' must be 'text' or 'assistant', got {stratum!r}")
    return records


def load_humaneval_records(path: Path, source: str = "humaneval") -> list[Record]:
    records: list[Record] = []
    for lineno, obj in _iter_jsonl(path):
        where = f"{path}:{lineno}"
        for key in ("task_id", "prompt", "canonical_solution"):
            if not isinstance(obj.get(key), str) or not obj[key]:
                raise CorpusError(f"{where}: HumanEval record needs non-empty string {key!r}")
        task = obj["task_id"]
        records.append(Record("text", source, f"{task}#prompt-text", text=obj["prompt"],
                              meta={"proxy": HUMANEVAL_TEXT_PROXY, "task_id": task}))
        records.append(Record(
            "assistant", source, f"{task}#assistant-proxy",
            messages=(("user", HUMANEVAL_USER_PREFIX + obj["prompt"]),
                      ("assistant", obj["canonical_solution"])),
            meta={"proxy": HUMANEVAL_PROXY, "task_id": task,
                  "proxy_note": "code-only canonical_solution rendered as an assistant turn; not a natural chat answer"},
        ))
    return records


def check_unique_ids(records: Sequence[Record]) -> None:
    seen: set[str] = set()
    for record in records:
        if record.id in seen:
            raise CorpusError(f"duplicate sequence id {record.id!r}")
        seen.add(record.id)


# --------------------------------------------------------------------------- #
# Unit construction
# --------------------------------------------------------------------------- #
def _check_no_special(adapter: TokenizerAdapter, text: str, what: str) -> None:
    for token in adapter.special_tokens:
        if token in text:
            raise Rejected(f"{what} contains special token {token!r}")


def normalize_messages(adapter: TokenizerAdapter, messages: Sequence[tuple[str, str]]) -> list[dict[str, str]]:
    if not messages:
        raise Rejected("no messages")
    roles = [role for role, _ in messages]
    for index, role in enumerate(roles):
        if role not in _ROLES:
            raise Rejected(f"unsupported role {role!r}")
        if role == "system" and index != 0:
            raise Rejected("system message must be first")
    if "user" not in roles or "assistant" not in roles:
        raise Rejected("conversation needs at least one user and one assistant message")
    out: list[dict[str, str]] = []
    for role, content in messages:
        _check_no_special(adapter, content, f"{role} content")
        if role == "assistant":
            if not content.strip():
                raise Rejected("assistant content is empty after trimming")
            if "<think>" in content or "</think>" in content:
                raise Rejected("assistant content contains a think tag")
        out.append({"role": role, "content": content})
    return out


def locate_assistant_spans(rendered: str, messages: Sequence[dict[str, str]]) -> list[tuple[int, int]]:
    """Character spans of each assistant turn's (template-trimmed) content."""
    assistants = [m["content"] for m in messages if m["role"] == "assistant"]
    if rendered.count(ASSISTANT_MARKER) != len(assistants):
        raise Rejected(
            f"rendered template has {rendered.count(ASSISTANT_MARKER)} assistant turns, expected {len(assistants)}")
    spans: list[tuple[int, int]] = []
    cursor = 0
    for content in assistants:
        marker = rendered.find(ASSISTANT_MARKER, cursor)
        body_start = marker + len(ASSISTANT_MARKER)
        end = rendered.find(TURN_END, body_start)
        if end < 0:
            raise Rejected("assistant turn has no terminating <|im_end|>")
        segment = rendered[body_start:end]
        expected = content.strip()
        if not segment.endswith(expected):
            raise Rejected("rendered assistant segment does not end with the trimmed content")
        scaffold = segment[: len(segment) - len(expected)]
        if not _SCAFFOLD_RE.fullmatch(scaffold):
            raise Rejected(f"unexpected scaffold before assistant content: {scaffold!r}")
        spans.append((end - len(expected), end))
        cursor = end + len(TURN_END)
    return spans


def role_mask_from_offsets(
    rendered: str, offsets: Sequence[tuple[int, int]], spans: Sequence[tuple[int, int]]
) -> list[int]:
    """Per-token mask: 1 iff the token lies entirely inside an assistant content span."""
    mask = [0] * len(offsets)
    members: list[list[int]] = [[] for _ in spans]
    for index, (start, end) in enumerate(offsets):
        for k, (s0, s1) in enumerate(spans):
            if start == end:
                if s0 < start < s1:
                    raise Rejected("zero-width token inside assistant content")
                continue
            if s0 <= start and end <= s1:
                mask[index] = 1
                members[k].append(index)
            elif max(start, s0) < min(end, s1):
                raise Rejected(f"token {index} {(start, end)} straddles an assistant content boundary")
    for (s0, s1), idx in zip(spans, members):
        if not idx:
            raise Rejected("assistant content produced no tokens")
        first, last = offsets[idx[0]], offsets[idx[-1]]
        # Allow only whitespace gaps (byte-level offset trimming); never dropped text.
        if rendered[s0:first[0]].strip() or rendered[last[1]:s1].strip():
            raise Rejected("assistant content is not fully covered by its tokens")
        reach = first[1]
        for i in idx[1:]:
            start, end = offsets[i]
            if rendered[reach:start].strip():
                raise Rejected("gap between assistant content tokens contains text")
            reach = max(reach, end)
    return mask


def _mask_spans(mask: Sequence[int]) -> list[list[int]]:
    spans: list[list[int]] = []
    start = None
    for i, bit in enumerate(mask):
        if bit and start is None:
            start = i
        elif not bit and start is not None:
            spans.append([start, i])
            start = None
    if start is not None:
        spans.append([start, len(mask)])
    return spans


def build_text_sequence(adapter: TokenizerAdapter, record: Record, max_tokens: int, min_scored: int) -> dict[str, Any]:
    assert record.text is not None
    if not record.text.strip():
        raise Rejected("text is empty")
    _check_no_special(adapter, record.text, "text")
    ids, _ = adapter.encode(record.text)
    full = len(ids)
    tokens = ids[:max_tokens]
    positions = list(range(len(tokens) - 1))
    if len(positions) < min_scored:
        raise Rejected(f"only {len(positions)} scoreable positions (< {min_scored})")
    meta = dict(record.meta)
    meta.update({
        "source": record.source,
        "n_tokens_untruncated": full,
        "truncated": full > len(tokens),
        "max_input_tokens": max_tokens,
        "text_sha256": sha256_text(record.text),
    })
    return {"id": record.id, "stratum": "text", "tokens": tokens, "score_positions": positions,
            "reset": True, "meta": meta}


def build_assistant_sequence(
    adapter: TokenizerAdapter, record: Record, max_tokens: int, min_scored: int
) -> dict[str, Any]:
    assert record.messages is not None
    messages = normalize_messages(adapter, record.messages)
    try:
        rendered = adapter.render_chat(messages)
    except CorpusError:
        raise
    except Exception as exc:  # jinja / template rejection of this conversation
        raise Rejected(f"chat template rejected conversation: {type(exc).__name__}: {exc}") from exc
    spans = locate_assistant_spans(rendered, messages)
    ids, offsets = adapter.encode(rendered)
    mask = role_mask_from_offsets(rendered, offsets, spans)
    full = len(ids)
    tokens = ids[:max_tokens]
    mask = mask[:max_tokens]
    positions = [p for p in range(len(tokens) - 1) if mask[p + 1] == 1]
    if len(positions) < min_scored:
        raise Rejected(f"only {len(positions)} scoreable assistant targets (< {min_scored})")
    meta = dict(record.meta)
    meta.update({
        "source": record.source,
        "n_tokens_untruncated": full,
        "truncated": full > len(tokens),
        "max_input_tokens": max_tokens,
        "rendered_sha256": sha256_text(rendered),
        "messages_sha256": sha256_text(canonical_json(messages)),
        "roles": [m["role"] for m in messages],
        "assistant_token_spans": _mask_spans(mask),
        "n_assistant_tokens": sum(mask),
        "assistant_content_trimmed_by_template": any(
            m["role"] == "assistant" and m["content"] != m["content"].strip() for m in messages),
    })
    return {"id": record.id, "stratum": "assistant", "tokens": tokens, "score_positions": positions,
            "reset": True, "role_mask": mask, "meta": meta}


# --------------------------------------------------------------------------- #
# Deterministic selection
# --------------------------------------------------------------------------- #
def _order_key(seed: int, record: Record) -> str:
    return sha256_text(f"{seed}\x00{record.stratum}\x00{record.source}\x00{record.id}")


def select_stratum(
    adapter: TokenizerAdapter, records: Sequence[Record], *, stratum: str, quota: int,
    max_tokens: int, seed: int, min_scored: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pick ``quota`` valid units: per-source hash order, round-robin over sources."""
    builder = build_text_sequence if stratum == "text" else build_assistant_sequence
    by_source: dict[str, list[Record]] = {}
    for record in records:
        if record.stratum == stratum:
            by_source.setdefault(record.source, []).append(record)
    queues = {s: sorted(rs, key=lambda r: _order_key(seed, r)) for s, rs in by_source.items()}
    sources = sorted(queues)
    cursors = {s: 0 for s in sources}
    units: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    considered = 0
    while len(units) < quota:
        progressed = False
        for source in sources:
            if len(units) >= quota:
                break
            queue = queues[source]
            if cursors[source] >= len(queue):
                continue
            record = queue[cursors[source]]
            cursors[source] += 1
            progressed = True
            considered += 1
            try:
                units.append(builder(adapter, record, max_tokens, min_scored))
            except Rejected as exc:
                rejected.append({"id": record.id, "reason": str(exc)})
        if not progressed:
            break
    report = {
        "requested": quota,
        "selected": len(units),
        "available": sum(len(q) for q in queues.values()),
        "considered": considered,
        "rejected_count": len(rejected),
        "rejected": rejected[:MAX_REPORTED_REJECTIONS],
        "n_tokens": sum(len(u["tokens"]) for u in units),
        "n_scored": sum(len(u["score_positions"]) for u in units),
        "max_input_tokens": max_tokens,
        "selected_ids": [u["id"] for u in units],
    }
    return units, report


# --------------------------------------------------------------------------- #
# Validation (tokenizer-independent structural checks)
# --------------------------------------------------------------------------- #
def validate_corpus(corpus: dict[str, Any], *, vocab_size: int | None = None) -> None:
    if corpus.get("schema") != SCHEMA:
        raise CorpusError(f"schema must be {SCHEMA!r}")
    if not isinstance(corpus.get("name"), str) or not isinstance(corpus.get("seed"), int):
        raise CorpusError("corpus needs string 'name' and integer 'seed'")
    extra = set(corpus) - {"schema", "name", "seed", "sequences", "meta"}
    if extra:
        raise CorpusError(f"unknown top-level keys {sorted(extra)} (extras belong under meta)")
    if "meta" in corpus and not isinstance(corpus["meta"], dict):
        raise CorpusError("corpus meta must be an object")
    sequences = corpus.get("sequences")
    if not isinstance(sequences, list) or not sequences:
        raise CorpusError("corpus has no sequences")
    seen: set[str] = set()
    for seq in sequences:
        sid = seq.get("id")
        if not isinstance(sid, str) or not sid or sid in seen:
            raise CorpusError(f"missing or duplicate sequence id {sid!r}")
        seen.add(sid)
        stratum, tokens, positions = seq.get("stratum"), seq.get("tokens"), seq.get("score_positions")
        if stratum not in STRATA:
            raise CorpusError(f"{sid}: bad stratum {stratum!r}")
        if seq.get("reset") is not True:
            raise CorpusError(f"{sid}: reset must be true")
        extra = set(seq) - {"id", "stratum", "tokens", "score_positions", "reset", "role_mask", "meta"}
        if extra:
            raise CorpusError(f"{sid}: unknown sequence keys {sorted(extra)} (extras belong under meta)")
        if not (isinstance(tokens, list) and len(tokens) >= 2
                and all(isinstance(t, int) and not isinstance(t, bool) and t >= 0 for t in tokens)):
            raise CorpusError(f"{sid}: tokens must be a list of >=2 non-negative ints")
        if vocab_size is not None and max(tokens) >= vocab_size:
            raise CorpusError(f"{sid}: token id >= vocab size {vocab_size}")
        if not isinstance(positions, list) or not positions:
            raise CorpusError(f"{sid}: score_positions must be non-empty")
        if any(not isinstance(p, int) or isinstance(p, bool) for p in positions) \
                or any(b <= a for a, b in zip(positions, positions[1:])):
            raise CorpusError(f"{sid}: score_positions must be strictly increasing ints")
        if positions[0] < 0 or positions[-1] > len(tokens) - 2:
            raise CorpusError(f"{sid}: score position needs a valid next target (0 <= p <= len-2)")
        if not isinstance(seq.get("meta"), dict):
            raise CorpusError(f"{sid}: meta must be an object")
        if stratum == "text":
            if "role_mask" in seq:
                raise CorpusError(f"{sid}: text sequences carry no role_mask")
            if positions != list(range(len(tokens) - 1)):
                raise CorpusError(f"{sid}: text sequences must score every position except the last")
        else:
            mask = seq.get("role_mask")
            if not (isinstance(mask, list) and len(mask) == len(tokens) and all(m in (0, 1) for m in mask)):
                raise CorpusError(f"{sid}: assistant role_mask must be 0/1 per token")
            expected = [p for p in range(len(tokens) - 1) if mask[p + 1] == 1]
            if positions != expected:
                raise CorpusError(f"{sid}: score_positions are not exactly the assistant-target positions")
            if any(mask[p + 1] != 1 for p in positions):
                raise CorpusError(f"{sid}: a scored target is not assistant content")


# --------------------------------------------------------------------------- #
# Build + CLI
# --------------------------------------------------------------------------- #
def _versions() -> dict[str, str]:
    from importlib import metadata

    out = {"python": sys.version.split()[0]}
    for pkg in ("transformers", "tokenizers"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = "unavailable"
    return out


def build_corpus(
    adapter: TokenizerAdapter,
    tokenizer_provenance: dict[str, Any],
    records: Sequence[Record],
    inputs: Sequence[dict[str, Any]],
    *,
    name: str,
    seed: int,
    units_per_stratum: int,
    max_input_tokens: int,
    assistant_max_input_tokens: int,
    min_scored: int,
    allow_short: bool,
) -> dict[str, Any]:
    check_unique_ids(records)
    caps = {"text": max_input_tokens, "assistant": assistant_max_input_tokens}
    sequences: list[dict[str, Any]] = []
    selection: dict[str, Any] = {}
    for stratum in STRATA:
        units, report = select_stratum(
            adapter, records, stratum=stratum, quota=units_per_stratum, max_tokens=caps[stratum],
            seed=seed, min_scored=min_scored)
        selection[stratum] = report
        if len(units) < units_per_stratum and not allow_short:
            raise CorpusError(
                f"stratum {stratum!r}: only {len(units)}/{units_per_stratum} valid units "
                f"({report['available']} candidates, {report['rejected_count']} rejected); "
                "add inputs or pass --allow-short")
        sequences.extend(units)
    if not sequences:
        raise CorpusError("no valid sequences selected")
    proxies = sorted({s["meta"]["proxy"] for s in sequences if "proxy" in s["meta"]})
    provenance = {
        "tool": {"script": "quality_corpus.py", "script_sha256": sha256_file(Path(__file__)), **_versions()},
        "tokenizer": tokenizer_provenance,
        "inputs": list(inputs),
        "selection": {
            "algorithm": "sha256(seed|stratum|source|id) order per source; round-robin over sorted sources; "
                         "first units that build without role ambiguity",
            "seed": seed,
            "units_per_stratum": units_per_stratum,
            "min_scored_positions": min_scored,
            "allow_short": allow_short,
            "per_stratum": selection,
        },
        "target_shift": 1,
        "proxies": proxies,
        "proxy_notice": (
            "sequences with meta.proxy are proxies, not natural reference chat data: "
            f"{HUMANEVAL_PROXY} = HumanEval canonical_solution (code only) as the assistant turn; "
            f"{HUMANEVAL_TEXT_PROXY} = HumanEval prompt as text") if proxies else None,
        "totals": {
            "sequences": len(sequences),
            "tokens": sum(len(s["tokens"]) for s in sequences),
            "scored_positions": sum(len(s["score_positions"]) for s in sequences),
            "per_stratum": {st: sum(1 for s in sequences if s["stratum"] == st) for st in STRATA},
        },
        "sequences_sha256": sha256_text(canonical_json(sequences)),
    }
    corpus = {"schema": SCHEMA, "name": name, "seed": seed, "sequences": sequences,
              "meta": {"provenance": provenance}}
    validate_corpus(corpus, vocab_size=adapter.vocab_size)
    return corpus


def serialize(corpus: dict[str, Any]) -> bytes:
    return (canonical_json(corpus) + "\n").encode("utf-8")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokenizer-dir", required=True, type=Path, help="local pinned tokenizer directory (no network)")
    p.add_argument("--input", action="append", default=[], type=Path, help="general JSONL (repeatable)")
    p.add_argument("--human-eval-input", type=Path, help="existing public HumanEval.jsonl (proxy pools)")
    p.add_argument("--expect-human-eval-sha256", help="fail unless --human-eval-input has this sha256")
    p.add_argument("--out", required=True, type=Path, help="corpus.json to write (refuses to overwrite)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--max-input-tokens", required=True, type=int, help="cap for text sequences (and assistant default)")
    p.add_argument("--assistant-max-input-tokens", type=int, help="cap for assistant sequences (default: --max-input-tokens)")
    p.add_argument("--units-per-stratum", required=True, type=int)
    p.add_argument("--min-scored-positions", type=int, default=16, help="reject units with fewer scored positions")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default="qwen4-kl-corpus")
    p.add_argument("--allow-short", action="store_true", help="report actual counts instead of failing when a stratum is short")
    p.add_argument("--tokenizer-source", default=PINNED_TOKENIZER_SOURCE)
    p.add_argument("--tokenizer-revision", default=PINNED_TOKENIZER_REVISION)
    p.add_argument("--strict-revision", action="store_true", help="fail without on-disk revision evidence")
    p.add_argument("--expect-tokenizer-json-sha256")
    p.add_argument("--chat-template-kwargs", default="{}", help="JSON kwargs for apply_chat_template, e.g. '{\"enable_thinking\": false}'")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None, *, tokenizer_loader: Callable[..., tuple[Any, dict[str, Any]]] = load_tokenizer) -> int:
    args = _parse_args(argv)
    try:
        if args.units_per_stratum < 1:
            raise CorpusError("--units-per-stratum must be >= 1")
        if args.min_scored_positions < 1:
            raise CorpusError("--min-scored-positions must be >= 1")
        assistant_cap = args.assistant_max_input_tokens or args.max_input_tokens
        for cap in (args.max_input_tokens, assistant_cap):
            if cap < args.min_scored_positions + 1:
                raise CorpusError("max input tokens must exceed --min-scored-positions")
        if not args.input and not args.human_eval_input:
            raise CorpusError("give --input and/or --human-eval-input")
        if args.out.exists() and not args.overwrite:
            raise CorpusError(f"{args.out} exists; pass --overwrite")
        try:
            kwargs = json.loads(args.chat_template_kwargs)
        except json.JSONDecodeError as exc:
            raise CorpusError(f"--chat-template-kwargs is not JSON: {exc}") from exc
        if not isinstance(kwargs, dict):
            raise CorpusError("--chat-template-kwargs must be a JSON object")

        records: list[Record] = []
        inputs: list[dict[str, Any]] = []
        labels: set[str] = set()
        for path in args.input:
            label = f"input:{path.name}"
            if label in labels:
                raise CorpusError(f"two --input files share the basename {path.name!r}")
            labels.add(label)
            loaded = load_general_records(path, label)
            records.extend(loaded)
            inputs.append({"label": label, "kind": "general_jsonl", "name": path.name,
                           "bytes": path.stat().st_size, "sha256": sha256_file(path), "records": len(loaded)})
        if args.human_eval_input:
            actual = sha256_file(args.human_eval_input)
            if args.expect_human_eval_sha256 and actual != args.expect_human_eval_sha256.lower():
                raise CorpusError(f"HumanEval sha256 {actual} != expected {args.expect_human_eval_sha256}")
            loaded = load_humaneval_records(args.human_eval_input)
            records.extend(loaded)
            inputs.append({"label": "humaneval", "kind": "humaneval_jsonl", "name": args.human_eval_input.name,
                           "bytes": args.human_eval_input.stat().st_size,
                           "sha256": sha256_file(args.human_eval_input), "records": len(loaded) // 2,
                           "user_prefix": HUMANEVAL_USER_PREFIX})

        adapter, tok_prov = tokenizer_loader(
            args.tokenizer_dir, source=args.tokenizer_source, revision=args.tokenizer_revision,
            strict_revision=args.strict_revision,
            expect_tokenizer_json_sha256=args.expect_tokenizer_json_sha256, chat_template_kwargs=kwargs)
        corpus = build_corpus(
            adapter, tok_prov, records, inputs, name=args.name, seed=args.seed,
            units_per_stratum=args.units_per_stratum, max_input_tokens=args.max_input_tokens,
            assistant_max_input_tokens=assistant_cap, min_scored=args.min_scored_positions,
            allow_short=args.allow_short)
        data = serialize(corpus)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_durable(args.out, data)
        # Re-read what is on disk and re-validate it.
        validate_corpus(json.loads(args.out.read_bytes().decode("utf-8")), vocab_size=adapter.vocab_size)
    except CorpusError as exc:
        print(f"quality_corpus: error: {exc}", file=sys.stderr)
        return 2
    totals = corpus["meta"]["provenance"]["totals"]
    prov = corpus["meta"]["provenance"]
    print(json.dumps({
        "out": str(args.out), "file_sha256": sha256_bytes(data),
        "sequences_sha256": prov["sequences_sha256"],
        "totals": totals, "proxies": prov["proxies"],
        "per_stratum": {st: {k: v for k, v in rep.items() if k not in ("rejected", "selected_ids")}
                        for st, rep in prov["selection"]["per_stratum"].items()},
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
