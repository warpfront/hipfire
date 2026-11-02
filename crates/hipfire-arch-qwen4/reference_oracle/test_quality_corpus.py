# SPDX-License-Identifier: Apache-2.0
"""Standard-library tests for ``quality_corpus`` using a fake fast tokenizer.

No torch / transformers / network.  Run from this directory::

    python -m unittest test_quality_corpus -v
"""

from __future__ import annotations

import io
import json
import re
import tempfile
import unittest
import zlib
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import quality_corpus as qc

SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
_FIXED = {"<|endoftext|>": 0, "<|im_start|>": 1, "<|im_end|>": 2, "<think>": 3, "</think>": 4}
_TOKEN_RE = re.compile(r"<\|[a-z_]+\|>|</?think>|\s*[\r\n]+| ?\w+| ?[^\s\w]+|\s+|.", re.S)
_MERGING_RE = re.compile(r"<\|[a-z_]+\|>|</?think>|[\r\n]+\w+|\s*[\r\n]+| ?\w+| ?[^\s\w]+|\s+|.", re.S)
_SPECIAL_SPLIT_RE = re.compile(r"<\|[a-z_]+\|>|</?think>")


class FakeFastTokenizer:
    """HF-fast-tokenizer-shaped double with a Qwen-like chat template."""

    is_fast = True
    all_special_tokens = SPECIALS
    chat_template = "fake-qwen-template-v1"

    def __init__(self, *, trim: bool = True, reasoning: str = "", merge_newline: bool = False) -> None:
        self.trim = trim
        self.reasoning = reasoning
        self.regex = _MERGING_RE if merge_newline else _TOKEN_RE
        self.pieces: dict[int, str] = {}

    def __len__(self) -> int:
        return 60000

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        assert tokenize is False and add_generation_prompt is False
        if not any(m["role"] == "user" for m in messages):
            raise ValueError("No user query found in messages.")
        out = []
        for m in messages:
            content = m["content"].strip() if self.trim else m["content"]
            if m["role"] == "assistant":
                out.append(f"<|im_start|>assistant\n<think>\n{self.reasoning}\n</think>\n\n{content}<|im_end|>\n")
            else:
                out.append(f"<|im_start|>{m['role']}\n{content}<|im_end|>\n")
        return "".join(out)

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        assert add_special_tokens is False and return_offsets_mapping is True
        ids, offsets = [], []

        def emit(piece, span):
            tid = _FIXED.get(piece, 10 + zlib.crc32(piece.encode()) % 50000)
            self.pieces[tid] = piece
            ids.append(tid)
            offsets.append(span)

        # Like HF fast tokenizers, added/special tokens are split out *before* the
        # BPE pre-tokenizer runs, so punctuation can never swallow "<|im_end|>".
        cursor = 0
        for special in list(_SPECIAL_SPLIT_RE.finditer(text)) + [None]:
            stop = special.start() if special else len(text)
            for match in self.regex.finditer(text, cursor, stop):
                emit(match.group(0), match.span())
            if special:
                emit(special.group(0), special.span())
                cursor = special.end()
        return {"input_ids": ids, "offset_mapping": offsets}


def make_adapter(**kwargs) -> tuple[qc.TokenizerAdapter, FakeFastTokenizer]:
    tok = FakeFastTokenizer(**kwargs)
    return qc.TokenizerAdapter(tok), tok


def fake_loader(**fake_kwargs):
    def loader(directory, **kwargs):
        adapter, _ = make_adapter(**fake_kwargs)
        return adapter, {"source": kwargs["source"], "revision": kwargs["revision"], "fake": True}
    return loader


def text_record(rid, text):
    return qc.Record("text", "t", rid, text=text)


def chat_record(rid, *messages, source="t"):
    return qc.Record("assistant", source, rid, messages=tuple(messages))


class RoleMaskTests(unittest.TestCase):
    def test_trimmed_content_exact_tokens_and_targets(self) -> None:
        adapter, tok = make_adapter()
        content = "  \n hello world \n\n"
        seq = qc.build_assistant_sequence(
            adapter, chat_record("a", ("user", "hi"), ("assistant", content)), 100, 1)
        pieces = [tok.pieces[t] for t in seq["tokens"]]
        self.assertEqual(pieces, [
            "<|im_start|>", "user", "\n", "hi", "<|im_end|>", "\n",
            "<|im_start|>", "assistant", "\n", "<think>", "\n\n", "</think>", "\n\n",
            "hello", " world", "<|im_end|>", "\n"])
        masked = [p for p, bit in zip(pieces, seq["role_mask"]) if bit]
        self.assertEqual(masked, ["hello", " world"])  # trimmed content only
        # Targets: first content token (from the scaffold position) and last content token;
        # the <|im_end|> target (p=14) and the trailing newline are excluded.
        self.assertEqual(seq["score_positions"], [12, 13])
        self.assertEqual(seq["meta"]["assistant_token_spans"], [[13, 15]])
        self.assertTrue(seq["meta"]["assistant_content_trimmed_by_template"])
        self.assertEqual(seq["role_mask"][len(pieces) - 1], 0)

    def test_multi_turn_system_user_assistant_roles(self) -> None:
        adapter, tok = make_adapter()
        seq = qc.build_assistant_sequence(adapter, chat_record(
            "m", ("system", "be brief"), ("user", "q one"), ("assistant", "answer one"),
            ("user", "q two"), ("assistant", " answer two. ")), 200, 1)
        pieces = [tok.pieces[t] for t in seq["tokens"]]
        self.assertEqual(seq["role_mask"].count(1), 5)
        self.assertEqual([p for p, b in zip(pieces, seq["role_mask"]) if b],
                         ["answer", " one", "answer", " two", "."])
        for p in seq["score_positions"]:
            self.assertEqual(seq["role_mask"][p + 1], 1)
        # no user/system/delimiter token is ever a scored target
        targets = {pieces[p + 1] for p in seq["score_positions"]}
        self.assertFalse(targets & {"<|im_start|>", "<|im_end|>", "user", "system", "q", "be", " brief", "<think>"})
        self.assertEqual(len(seq["meta"]["assistant_token_spans"]), 2)
        self.assertEqual(seq["meta"]["roles"], ["system", "user", "assistant", "user", "assistant"])

    def test_truncation_keeps_mask_aligned_and_last_token_unscored(self) -> None:
        adapter, tok = make_adapter()
        content = " ".join(f"w{i}" for i in range(40))
        seq = qc.build_assistant_sequence(adapter, chat_record("t", ("user", "hi"), ("assistant", content)), 20, 1)
        self.assertEqual(len(seq["tokens"]), 20)
        self.assertEqual(len(seq["role_mask"]), 20)
        self.assertTrue(seq["meta"]["truncated"])
        self.assertEqual(seq["score_positions"][-1], 18)  # last input p = len-2
        self.assertEqual(seq["score_positions"], list(range(12, 19)))

    def test_text_scores_every_position_but_the_last(self) -> None:
        adapter, _ = make_adapter()
        seq = qc.build_text_sequence(adapter, text_record("x", "one two three four five"), 4, 1)
        self.assertEqual(len(seq["tokens"]), 4)
        self.assertEqual(seq["score_positions"], [0, 1, 2])
        self.assertNotIn("role_mask", seq)

    def test_ambiguity_is_rejected_not_mis_scored(self) -> None:
        adapter, _ = make_adapter()
        cases = {
            "special in content": chat_record("s", ("user", "hi"), ("assistant", "x <|im_end|> y")),
            "think in content": chat_record("k", ("user", "hi"), ("assistant", "a</think>b")),
            "empty after trim": chat_record("e", ("user", "hi"), ("assistant", " \n ")),
            "no assistant": chat_record("n", ("user", "hi")),
            "no user (template raises)": chat_record("u", ("assistant", "hi")),
            "tool role": chat_record("r", ("user", "hi"), ("tool", "x"), ("assistant", "y")),
        }
        for label, record in cases.items():
            with self.subTest(label), self.assertRaises(qc.Rejected):
                qc.build_assistant_sequence(adapter, record, 100, 1)
        with self.assertRaises(qc.Rejected):
            qc.build_text_sequence(adapter, text_record("z", "has <|im_start|> inside"), 100, 1)

    def test_nonempty_reasoning_scaffold_is_rejected(self) -> None:
        adapter, _ = make_adapter(reasoning="secret thoughts")
        with self.assertRaises(qc.Rejected):
            qc.build_assistant_sequence(adapter, chat_record("r", ("user", "hi"), ("assistant", "yo")), 100, 1)

    def test_untrimmed_template_with_trailing_whitespace_is_rejected(self) -> None:
        adapter, _ = make_adapter(trim=False)
        with self.assertRaises(qc.Rejected):
            qc.build_assistant_sequence(adapter, chat_record("w", ("user", "hi"), ("assistant", "yo \n")), 100, 1)

    def test_token_straddling_content_boundary_is_rejected(self) -> None:
        adapter, _ = make_adapter(merge_newline=True)  # "\n\nhello" becomes one token
        with self.assertRaises(qc.Rejected):
            qc.build_assistant_sequence(adapter, chat_record("b", ("user", "hi"), ("assistant", "hello")), 100, 1)

    def test_too_few_scored_targets_rejected(self) -> None:
        adapter, _ = make_adapter()
        with self.assertRaises(qc.Rejected):
            qc.build_assistant_sequence(adapter, chat_record("f", ("user", "hi"), ("assistant", "yo")), 100, 5)
        # truncation that cuts the answer away entirely
        with self.assertRaises(qc.Rejected):
            qc.build_assistant_sequence(adapter, chat_record("g", ("user", "hi"), ("assistant", "yo")), 10, 1)


class HumanEvalTests(unittest.TestCase):
    def test_proxy_labels_and_trimmed_canonical_solution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "he.jsonl"
            path.write_text(json.dumps({
                "task_id": "HumanEval/0", "prompt": "def f(x):\n    \"\"\"d\"\"\"\n",
                "canonical_solution": "    return x + 1\n", "entry_point": "f", "test": "t"}) + "\n")
            records = qc.load_humaneval_records(path)
        self.assertEqual([r.stratum for r in records], ["text", "assistant"])
        self.assertEqual(records[0].id, "HumanEval/0#prompt-text")
        adapter, tok = make_adapter()
        seq = qc.build_assistant_sequence(adapter, records[1], 200, 1)
        self.assertEqual(seq["meta"]["proxy"], qc.HUMANEVAL_PROXY)
        self.assertTrue(seq["meta"]["assistant_content_trimmed_by_template"])
        masked = "".join(tok.pieces[t] for t, b in zip(seq["tokens"], seq["role_mask"]) if b)
        self.assertEqual(masked, "return x + 1")  # leading indent and trailing newline trimmed by template
        text_seq = qc.build_text_sequence(adapter, records[0], 200, 1)
        self.assertEqual(text_seq["meta"]["proxy"], qc.HUMANEVAL_TEXT_PROXY)


def _write_general(path: Path, order: list[int]) -> None:
    records = []
    for i in range(12):
        records.append({"id": f"t{i}", "stratum": "text",
                        "text": " ".join(f"tok{i}x{j}" for j in range(30 + i))})
        records.append({"id": f"a{i}", "stratum": "assistant", "messages": [
            {"role": "user", "content": f"question {i}?"},
            {"role": "assistant", "content": "\n  " + " ".join(f"ans{i}x{j}" for j in range(25)) + "  \n"}]})
    path.write_text("".join(json.dumps(records[k]) + "\n" for k in order))


def run_cli(args: list[str], **fake) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = qc.main(args, tokenizer_loader=fake_loader(**fake))
    return code, out.getvalue(), err.getvalue()


class CliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.tok_dir = self.dir / "tok"
        self.tok_dir.mkdir()

    def args(self, inp: Path, out: Path, *extra: str) -> list[str]:
        return ["--tokenizer-dir", str(self.tok_dir), "--input", str(inp), "--out", str(out),
                "--max-input-tokens", "24", "--assistant-max-input-tokens", "40",
                "--units-per-stratum", "4", "--min-scored-positions", "4", "--seed", "7", *extra]

    def test_deterministic_and_order_independent_and_valid(self) -> None:
        (self.dir / "d1").mkdir()
        (self.dir / "d2").mkdir()
        a, b = self.dir / "d1" / "g.jsonl", self.dir / "d2" / "g.jsonl"
        _write_general(a, list(range(24)))
        _write_general(b, list(reversed(range(24))))
        outs = []
        for name, inp in (("1.json", a), ("2.json", a), ("3.json", b)):
            code, stdout, stderr = run_cli(self.args(inp, self.dir / name))
            self.assertEqual(code, 0, stderr)
            outs.append((self.dir / name).read_bytes())
        self.assertEqual(outs[0], outs[1])
        # input row order must not change the frozen tokens/selection
        c1, c3 = json.loads(outs[0]), json.loads(outs[2])
        self.assertEqual(c1["sequences"], c3["sequences"])
        p1, p3 = c1["meta"]["provenance"], c3["meta"]["provenance"]
        self.assertEqual(p1["sequences_sha256"], p3["sequences_sha256"])
        self.assertNotEqual(p1["inputs"][0]["sha256"], p3["inputs"][0]["sha256"])
        self.assertEqual(c1["schema"], "hipfire.qwen4.kl_corpus.v1")
        self.assertEqual(c1["seed"], 7)
        self.assertEqual(p1["totals"]["per_stratum"], {"text": 4, "assistant": 4})
        self.assertEqual(set(c1), {"schema", "name", "seed", "sequences", "meta"})
        self.assertEqual(set(c1["meta"]), {"provenance"})
        qc.validate_corpus(c1, vocab_size=60000)
        for seq in c1["sequences"]:
            self.assertLessEqual(len(seq["tokens"]), 24 if seq["stratum"] == "text" else 40)
        # a different seed picks a different set
        code, _, _ = run_cli(self.args(a, self.dir / "s.json", "--seed", "8"))
        self.assertEqual(code, 0)
        self.assertNotEqual(json.loads((self.dir / "s.json").read_bytes())["sequences"], c1["sequences"])

    def test_fsync_file_and_directory_and_no_temp_left(self) -> None:
        inp = self.dir / "g.jsonl"
        _write_general(inp, list(range(24)))
        with mock.patch.object(qc.os, "fsync", wraps=qc.os.fsync) as fsync:
            code, _, stderr = run_cli(self.args(inp, self.dir / "o.json"))
        self.assertEqual(code, 0, stderr)
        self.assertGreaterEqual(fsync.call_count, 2)
        self.assertEqual(sorted(p.name for p in self.dir.glob("o.json*")), ["o.json"])

    def test_refuses_overwrite_and_short_strata(self) -> None:
        inp = self.dir / "g.jsonl"
        _write_general(inp, list(range(24)))
        out = self.dir / "o.json"
        self.assertEqual(run_cli(self.args(inp, out))[0], 0)
        code, _, stderr = run_cli(self.args(inp, out))
        self.assertEqual(code, 2)
        self.assertIn("exists", stderr)
        short = self.args(inp, self.dir / "short.json") + ["--units-per-stratum", "50"]
        code, _, stderr = run_cli(short)
        self.assertEqual(code, 2)
        self.assertIn("only 12/50", stderr)
        code, stdout, _ = run_cli(short + ["--allow-short"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stdout)["totals"]["per_stratum"], {"text": 12, "assistant": 12})

    def test_humaneval_pool_is_labelled_proxy_and_hashes_recorded(self) -> None:
        he = self.dir / "he.jsonl"
        he.write_text("".join(json.dumps({
            "task_id": f"HumanEval/{i}", "prompt": f"def f{i}(a, b):\n    \"\"\"doc string {i} for the function\"\"\"\n",
            "canonical_solution": f"    result = a + b + {i}\n    return result\n"}) + "\n" for i in range(10)))
        out = self.dir / "he.json"
        code, stdout, stderr = run_cli([
            "--tokenizer-dir", str(self.tok_dir), "--human-eval-input", str(he), "--out", str(out),
            "--max-input-tokens", "24", "--assistant-max-input-tokens", "96",
            "--units-per-stratum", "3", "--min-scored-positions", "4"])
        self.assertEqual(code, 0, stderr)
        corpus = json.loads(out.read_bytes())
        prov = corpus["meta"]["provenance"]
        self.assertEqual(sorted(prov["proxies"]), sorted([qc.HUMANEVAL_PROXY, qc.HUMANEVAL_TEXT_PROXY]))
        self.assertIn("proxy", prov["proxy_notice"])
        self.assertEqual(prov["inputs"][0]["sha256"], qc.sha256_file(he))
        self.assertEqual(prov["tokenizer"]["revision"], qc.PINNED_TOKENIZER_REVISION)
        self.assertEqual(len(prov["tool"]["script_sha256"]), 64)
        assistants = [s for s in corpus["sequences"] if s["stratum"] == "assistant"]
        self.assertEqual(len(assistants), 3)
        self.assertTrue(all(s["meta"]["proxy"] == qc.HUMANEVAL_PROXY for s in assistants))

    def test_rejected_records_are_skipped_and_reported(self) -> None:
        inp = self.dir / "g.jsonl"
        rows = [{"id": f"bad{i}", "stratum": "assistant", "messages": [
            {"role": "user", "content": "hi"}, {"role": "assistant", "content": "x <|im_end|> y"}]}
            for i in range(3)]
        rows += [{"id": f"t{i}", "stratum": "text", "text": " ".join(f"w{i}x{j}" for j in range(30))} for i in range(4)]
        rows += [{"id": f"a{i}", "stratum": "assistant", "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": " ".join(f"v{i}x{j}" for j in range(20))}]} for i in range(4)]
        inp.write_text("".join(json.dumps(r) + "\n" for r in rows))
        out = self.dir / "r.json"
        code, _, stderr = run_cli(self.args(inp, out))
        self.assertEqual(code, 0, stderr)
        report = json.loads(out.read_bytes())["meta"]["provenance"]["selection"]["per_stratum"]["assistant"]
        self.assertEqual(report["selected"], 4)
        order = sorted(
            (r for r in rows if r["stratum"] == "assistant"),
            key=lambda r: qc._order_key(7, qc.Record("assistant", "input:g.jsonl", r["id"])))
        expected_rejected, good = 0, 0
        for r in order:
            if good == 4:
                break
            if r["id"].startswith("bad"):
                expected_rejected += 1
            else:
                good += 1
        self.assertEqual(report["rejected_count"], expected_rejected)
        self.assertTrue(all("special token" in r["reason"] for r in report["rejected"]))

    def test_bad_input_shape_is_fatal(self) -> None:
        inp = self.dir / "bad.jsonl"
        inp.write_text(json.dumps({"id": "x", "stratum": "text"}) + "\n")
        code, _, stderr = run_cli(self.args(inp, self.dir / "o.json"))
        self.assertEqual(code, 2)
        self.assertIn("text record", stderr)


class ValidatorTests(unittest.TestCase):
    def _corpus(self) -> dict:
        adapter, _ = make_adapter()
        seqs = [
            qc.build_text_sequence(adapter, text_record("t", "a b c d e f g h"), 100, 1),
            qc.build_assistant_sequence(adapter, chat_record("a", ("user", "q"), ("assistant", "x y z")), 100, 1),
        ]
        return {"schema": qc.SCHEMA, "name": "n", "seed": 0, "sequences": seqs}

    def test_valid_passes_and_corruptions_fail(self) -> None:
        qc.validate_corpus(self._corpus(), vocab_size=60000)
        mutations = {
            "last position scored": lambda c: c["sequences"][0]["score_positions"].append(len(c["sequences"][0]["tokens"]) - 1),
            "text drops position": lambda c: c["sequences"][0]["score_positions"].pop(0),
            "target not assistant": lambda c: c["sequences"][1]["score_positions"].insert(0, 0),
            "mask length": lambda c: c["sequences"][1]["role_mask"].pop(),
            "missing mask": lambda c: c["sequences"][1].pop("role_mask"),
            "reset false": lambda c: c["sequences"][0].__setitem__("reset", False),
            "duplicate id": lambda c: c["sequences"][1].__setitem__("id", "t"),
            "top-level provenance": lambda c: c.__setitem__("provenance", {}),
            "unknown sequence key": lambda c: c["sequences"][0].__setitem__("provenance", {}),
        }
        for label, mutate in mutations.items():
            corpus = self._corpus()
            mutate(corpus)
            with self.subTest(label), self.assertRaises(qc.CorpusError):
                qc.validate_corpus(corpus, vocab_size=60000)


if __name__ == "__main__":
    unittest.main()
