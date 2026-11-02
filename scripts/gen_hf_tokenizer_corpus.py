#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Regenerate the HF-tokenizer exact-equality corpus fixtures.

Output: crates/hipfire-runtime/tests/fixtures/hf_tokenizer_corpus/
    tokenizer.json  verbatim copy of the HF reference tokenizer.json
    cases.jsonl     one case per line: rendered input text + HF reference IDs
    manifest.json   provenance (md5s, versions), per-group counts summary

The Rust integration test `crates/hipfire-runtime/tests/hf_tokenizer_corpus.rs`
re-encodes every case's text with `hipfire_runtime::tokenizer::Tokenizer` and
requires the IDs to equal the HF IDs exactly.  The test is fully offline
(no network, no Python, no absolute paths): everything it needs is committed.

Reference = `tokenizers.Tokenizer.from_file(tokenizer.json).encode(text,
add_special_tokens=False)`; chat-request inputs are first rendered with the
reference `chat_template.jinja` through `transformers`' `apply_chat_template`
(tokenize=False, add_generation_prompt=True).

Requirements: python3, tokenizers==0.22.x, transformers>=5.8 (generation only).

Example (the invocation that produced the committed fixtures):

    scripts/gen_hf_tokenizer_corpus.py \\
      --tokenizer-dir  .../fn-quant-research/floor/fn-tokenizer \\
      --hermes-dir     .../hermes-ciru-mtp-p1/remote-http \\
      --tc-dir         .../tools-tc70-84-r9700-iu4-sf-mtpoff-s123/http \\
      --tc14-dir       .../tc-s123-mtp-cacheon/raw/http --tc14-range 13-26 \\
      --prompts-dir    benchmarks/prompts

Every source group is optional; omitted groups are simply not regenerated
(the committed manifest then records only the groups that were produced).
"""
import argparse
import glob
import hashlib
import json
import os
import random
import sys
import unicodedata

DEFAULT_OUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "crates", "hipfire-runtime", "tests", "fixtures", "hf_tokenizer_corpus",
)
EXPECT_TOKENIZER_MD5 = "085e10165f24499bb9d8fcee7e17f9ee"
EXPECT_TEMPLATE_MD5 = "519239a4908bb1f805bbce5fa8c8a242"


def md5_bytes(b):
    return hashlib.md5(b).hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def file_bytes(p):
    with open(p, "rb") as f:
        return f.read()


# --------------------------------------------------------------------------
# Chat-request rendering
# --------------------------------------------------------------------------
def normalize_messages(messages):
    out = []
    for m in messages:
        m = dict(m)
        if m.get("content") is None:
            m["content"] = ""
        tcs = m.get("tool_calls")
        if tcs:
            new = []
            for tc in tcs:
                tc = json.loads(json.dumps(tc))
                fn = tc.get("function", {})
                a = fn.get("arguments")
                if isinstance(a, str):
                    fn["arguments"] = json.loads(a) if a.strip() else {}
                elif a is None:
                    fn["arguments"] = {}
                new.append(tc)
            m["tool_calls"] = new
        out.append(m)
    return out


def template_kwargs(body):
    kw = {
        "enable_thinking": True,
        "preserve_thinking": True,
        "reasoning_effort": body.get("reasoning_effort") or "xhigh",
    }
    kw.update(body.get("chat_template_kwargs") or {})
    return kw


def render_request(hf_tok, body):
    kw = template_kwargs(body)
    args = dict(
        tokenize=False,
        add_generation_prompt=True,
        **kw,
    )
    tools = body.get("tools")
    if tools:
        args["tools"] = tools
    text = hf_tok.apply_chat_template(normalize_messages(body["messages"]), **args)
    return text, kw


def served_prompt_tokens(resp_path):
    """Best-effort `usage.prompt_tokens` the original server reported."""
    if not os.path.exists(resp_path):
        return None
    raw = file_bytes(resp_path).decode("utf-8", "replace")
    found = None
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            line = line[5:].strip()
        if not line.startswith("{") or '"prompt_tokens"' not in line:
            continue
        try:
            u = json.loads(line).get("usage")
        except ValueError:
            continue
        if u and "prompt_tokens" in u:
            found = u["prompt_tokens"]
    return found


def request_cases(group, src_dir, label, hf_tok, lo=None, hi=None):
    cases = []
    for d in sorted(glob.glob(os.path.join(src_dir, "*"))):
        name = os.path.basename(d)
        if not name.isdigit():
            continue
        n = int(name)
        if lo is not None and not (lo <= n <= hi):
            continue
        req = os.path.join(d, "request.json")
        if not os.path.exists(req):
            continue  # GET /v1/models etc. carry no request body
        raw = file_bytes(req)
        body = json.loads(raw)
        text, kw = render_request(hf_tok, body)
        cases.append(dict(
            id=f"{group}/{name}",
            group=group,
            source=f"{label}/{name}/request.json",
            source_md5=md5_bytes(raw),
            kind="chat_request",
            template_kwargs=kw,
            served_prompt_tokens=served_prompt_tokens(os.path.join(d, "response.bin")),
            text=text,
        ))
    return cases


def prompt_cases(prompts_dir):
    cases = []
    for p in sorted(glob.glob(os.path.join(prompts_dir, "*.txt"))):
        raw = file_bytes(p)
        cases.append(dict(
            id=f"prompts/{os.path.basename(p)}",
            group="prompts",
            source=f"benchmarks/prompts/{os.path.basename(p)}",
            source_md5=md5_bytes(raw),
            kind="raw_text",
            text=raw.decode("utf-8"),
        ))
    return cases


# --------------------------------------------------------------------------
# Synthetic cases (fully deterministic)
# --------------------------------------------------------------------------
WS_CHARS = [
    ("sp", " "), ("tab", "\t"), ("lf", "\n"), ("cr", "\r"), ("crlf", "\r\n"),
    ("vt", "\x0b"), ("ff", "\x0c"), ("nel", "\u0085"), ("nbsp", "\u00a0"),
    ("ogham", "\u1680"), ("ensp", "\u2002"), ("emsp", "\u2003"),
    ("thinsp", "\u2009"), ("hairsp", "\u200a"), ("ls", "\u2028"),
    ("ps", "\u2029"), ("nnbsp", "\u202f"), ("mmsp", "\u205f"),
    ("ideo", "\u3000"), ("zwsp", "\u200b"), ("bom", "\ufeff"),
]


def synthetic_cases():
    named = [
        ("empty", ""),
        ("single_space", " "),
        ("single_lf", "\n"),
        ("tabs_basic", "\tindented\n\t\tdeeper\n\t\t\tdeepest\n"),
        ("tabs_mid_line", "a\tb\t\tc\t\t\td"),
        ("mixed_indent", "def f():\n\tif x:\n        y = 1\n\t    z = 2\n  \t  w = 3\n"),
        ("mixed_indent_tabs_spaces_blank", "{\n\t\n  \n \t\n\t \n}\n"),
        ("trailing_newline", "line one\nline two\n"),
        ("trailing_newline_space", "line one\nline two\n "),
        ("trailing_newlines_spaces", "x\n\n  \n   "),
        ("trailing_space_then_nl", "x   \ny \t \nz"),
        ("only_newlines", "\n\n\n\n"),
        ("only_ws_mix", " \n \t\n\t \n  "),
        ("space_run_before_word", "a      b     c  d"),
        ("space_run_2", "foo  bar"),
        ("space_run_3_end", "foo   "),
        ("nl_then_space_word", "foo\n bar\n  baz\n\n qux"),
        ("crlf_lines", "a\r\nb\r\n\r\nc\r\n"),
        ("lone_cr", "a\rb\r\rc"),
        ("unicode_ws_inline", "a\u00a0b\u2003c\u3000d\u202fe\u2009f\u200ag"),
        ("unicode_ws_runs", "a\u00a0\u00a0\u00a0b\u2003\u2003c  \u3000 d"),
        ("unicode_ws_nl", "a\u2028b\u2029c\u0085d\n\u00a0e"),
        ("unicode_ws_trailing", "word \u00a0"),
        ("zero_width", "a\u200bb\u200c\u200dc\ufeffd"),
        ("combining_decomposed", "e\u0301 a\u0300 o\u0308 n\u0303 cafe\u0301"),
        ("combining_precomposed", "\u00e9 \u00e0 \u00f6 \u00f1 caf\u00e9"),
        ("combining_stacked", "a\u0301\u0302\u0303\u0304 z\u0338\u0336"),
        ("combining_leading", "\u0301abc \u0308\u0308x"),
        ("combining_after_digit", "1\u0301 23\u0300 \u0661\u0301"),
        ("combining_after_punct", "!\u0301 ?\u0308?"),
        ("combining_hangul_jamo", "\u1112\u1161\u11ab \ud55c\uae00"),
        ("devanagari", "\u0939\u093f\u0928\u094d\u0926\u0940 \u0915\u094d\u0937"),
        ("thai_marks", "\u0e01\u0e34\u0e48\u0e07 \u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e35"),
        ("arabic_marks", "\u0645\u064f\u062d\u064e\u0645\u0651\u064e\u062f \u0627\u0644\u0639\u0631\u0628\u064a\u0629"),
        ("digits_ascii", "0 1 12 123 1234 12345 123456 1234567 12345678901234567890"),
        ("digits_adjacent_letters", "abc123def 4x5 x6y a1b2c3"),
        ("digits_signed_decimal", "-12 +3.14 1,234,567.89 1e-9 0x1F 1_000"),
        ("digits_arabic_indic", "\u0661\u0662\u0663 \u06f4\u06f5\u06f6 \u0967\u0968\u0969"),
        ("digits_fullwidth", "\uff11\uff12\uff13 \uff21\uff22"),
        ("digits_other_n", "\u00b2\u00b3 \u2160\u2161\u216b \u2460\u2461 \u00bd\u00be \u3007"),
        ("digits_ws_between", "1 2\t3\n4\u00a05\u30006"),
        ("contractions", "don't I'm you're we've they'll he'd it's"),
        ("contractions_upper", "DON'T I'M YOU'RE WE'VE THEY'LL HE'D IT'S"),
        ("contractions_curly", "don\u2019t it\u2019s \u2018quoted\u2019"),
        ("contractions_edge", "'s 't 're ve'm ll'd 'S 'T x's's 'sx"),
        ("punct_runs", "!!! ??? ... --- ::: ;;; ((( ))) [[ ]] {{ }}"),
        ("punct_then_newline", "foo;\nbar.\n\nbaz!?\r\nqux"),
        ("punct_symbol_mix", "a+b=c; x->y <= z !== w ?? !!"),
        ("code_snippet",
         "fn main() {\n\tlet x = vec![1, 2, 3];\n\tfor i in &x {\n\t\tprintln!(\"{}\", i);\n\t}\n}\n"),
        ("json_snippet", "{\"a\": [1, 2.5, null], \"b\": {\"c\": \"d\\n\"}, \"e\": true}"),
        ("url_path", "https://example.com/a/b?c=d&e=f#g /usr/local/bin ./x/../y"),
        ("emoji", "hello \U0001f600 \U0001f468\u200d\U0001f469\u200d\U0001f467 \U0001f1fa\U0001f1f8 \u2764\ufe0f"),
        ("cjk", "\u4f60\u597d\uff0c\u4e16\u754c\uff01 \u3053\u3093\u306b\u3061\u306f\u3001\u4e16\u754c\u3002"),
        ("mixed_scripts", "hello\u043c\u0438\u0440\u4e16\u754c123\u0645\u0631\u062d\u0628\u0627"),
        ("special_tokens_inline",
         "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"),
        ("special_tokens_partial", "<|im_st <|im_start| <im_start|> <|endoftext|"),
        ("special_tokens_adjacent", "<|im_end|><|im_end|>\n\n<|im_start|> <|im_start|>"),
        ("tool_call_markup",
         "<tool_call>\n<function=read>\n<parameter=path>\n/etc/hostname\n</parameter>\n</function>\n</tool_call>"),
        ("tool_response_markup", "<tool_response>\n{\"ok\": true}\n</tool_response>"),
        ("long_repeat_a", "a" * 300),
        ("long_repeat_space", " " * 200 + "x"),
        ("long_repeat_nl", "\n" * 120 + "x"),
        ("long_repeat_mixed", ("ab \t\n" * 80)),
    ]
    cases = [(f"synthetic/{n}", t) for n, t in named]

    # Systematic whitespace-character matrix.
    templates = [
        ("a{c}b", "mid"), ("{c}a", "lead"), ("a{c}", "trail"),
        ("a{c}{c}b", "dbl"), ("a {c}b", "sp_pre"), ("a{c} b", "sp_post"),
        ("a{c}\nb", "nl_post"), ("a\n{c}b", "nl_pre"), ("x\n{c}\ny", "nl_both"),
        ("1{c}2", "digit_mid"), ("({c})", "paren"),
    ]
    for wname, w in WS_CHARS:
        for tpl, tname in templates:
            cases.append((f"synthetic/wsmatrix_{wname}_{tname}", tpl.format(c=w)))

    # Deterministic fuzz over a hostile alphabet.
    rng = random.Random(0x68697066)
    alphabet = (
        ["a", "b", "Z", "x", "1", "2", "0", "9", " ", " ", " ", "\n", "\n", "\t", "\r",
         "'", "s", "t", "'s", "'re", ".", ",", "!", "-", "_", "/", "\\", "(", ")", "{", "}",
         "\u00a0", "\u2003", "\u3000", "\u2028", "\u0301", "\u0308", "\u00e9", "e\u0301",
         "\u0661", "\u00b2", "\u4f60", "\u0436", "\U0001f600", "<|im_end|>", "<think>"]
    )
    for i in range(400):
        n = rng.randint(1, 48)
        cases.append((f"synthetic/fuzz_{i:03d}", "".join(rng.choice(alphabet) for _ in range(n))))

    out = []
    for cid, text in cases:
        raw = text.encode("utf-8")
        out.append(dict(
            id=cid, group="synthetic", source="generator:synthetic_cases()",
            source_md5=md5_bytes(raw), kind="raw_text", text=text,
        ))
    return out


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer-dir", required=True,
                    help="HF dir with tokenizer.json, tokenizer_config.json, chat_template.jinja")
    ap.add_argument("--hermes-dir", help="Hermes remote-http dir (NNNNN/request.json + meta.json)")
    ap.add_argument("--tc-dir", help="complete TC corpus http dir (NNNNN/request.json)")
    ap.add_argument("--tc14-dir", help="swarm TC capture http dir (NNNNN/request.json)")
    ap.add_argument("--tc14-range", default="13-26", help="inclusive index range within --tc14-dir")
    ap.add_argument("--prompts-dir", help="benchmarks/prompts (top-level *.txt)")
    ap.add_argument("--no-synthetic", action="store_true")
    ap.add_argument("--out", default=os.path.normpath(DEFAULT_OUT))
    ap.add_argument("--expect-tokenizer-md5", default=EXPECT_TOKENIZER_MD5)
    ap.add_argument("--expect-template-md5", default=EXPECT_TEMPLATE_MD5)
    args = ap.parse_args()

    import tokenizers
    import transformers
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    tok_path = os.path.join(args.tokenizer_dir, "tokenizer.json")
    tpl_path = os.path.join(args.tokenizer_dir, "chat_template.jinja")
    tok_bytes = file_bytes(tok_path)
    tpl_bytes = file_bytes(tpl_path)
    if md5_bytes(tok_bytes) != args.expect_tokenizer_md5:
        sys.exit(f"tokenizer.json md5 {md5_bytes(tok_bytes)} != expected {args.expect_tokenizer_md5}")
    if md5_bytes(tpl_bytes) != args.expect_template_md5:
        sys.exit(f"chat_template.jinja md5 {md5_bytes(tpl_bytes)} != expected {args.expect_template_md5}")

    ref = Tokenizer.from_file(tok_path)
    hf_tok = AutoTokenizer.from_pretrained(args.tokenizer_dir)

    cases = []
    if args.hermes_dir:
        cases += request_cases("hermes", args.hermes_dir, "hermes-ciru-mtp-p1/remote-http", hf_tok)
    if args.tc_dir:
        cases += request_cases("tc55", args.tc_dir, "tools-tc70-84-r9700-iu4-sf-mtpoff-s123/http", hf_tok)
    if args.tc14_dir:
        lo, hi = (int(x) for x in args.tc14_range.split("-"))
        cases += request_cases("tc14", args.tc14_dir, "tc-s123-mtp-cacheon/raw/http", hf_tok, lo, hi)
    if args.prompts_dir:
        cases += prompt_cases(args.prompts_dir)
    if not args.no_synthetic:
        cases += synthetic_cases()
    ids_seen = set()
    for c in cases:
        if c["id"] in ids_seen:
            sys.exit(f"duplicate case id {c['id']}")
        ids_seen.add(c["id"])

    summary = {}
    for c in cases:
        text = c["text"]
        enc = ref.encode(text, add_special_tokens=False).ids
        # Informational only: transformers' Qwen2Tokenizer swaps in its own
        # pre-tokenizer regex (no \p{M} in the letter class), so it is NOT the
        # reference for IDs; tokenizer.json via `tokenizers` is.
        alt = hf_tok(text, add_special_tokens=False)["input_ids"]
        raw = text.encode("utf-8")
        c["text_md5"] = md5_bytes(raw)
        c["text_sha256"] = sha256_bytes(raw)
        c["text_bytes"] = len(raw)
        c["n_tokens"] = len(enc)
        c["ids"] = enc
        g = summary.setdefault(c["group"], dict(
            n_cases=0, n_tokens=0, n_text_bytes=0, n_nfc_changed=0,
            n_transformers_qwen2_tokenizer_differs=0,
            served_prompt_tokens_cases=0, served_equal_hf=0, served_differ_hf=0, served_differ_ids=[],
        ))
        g["n_cases"] += 1
        g["n_tokens"] += len(enc)
        g["n_text_bytes"] += len(raw)
        if alt != enc:
            g["n_transformers_qwen2_tokenizer_differs"] += 1
        if unicodedata.normalize("NFC", text) != text:
            g["n_nfc_changed"] += 1
        sp = c.get("served_prompt_tokens")
        if sp is not None:
            g["served_prompt_tokens_cases"] += 1
            if sp == len(enc):
                g["served_equal_hf"] += 1
            else:
                g["served_differ_hf"] += 1
                g["served_differ_ids"].append(c["id"])
    for g in summary.values():
        if not g["served_prompt_tokens_cases"]:
            for k in ("served_prompt_tokens_cases", "served_equal_hf", "served_differ_hf", "served_differ_ids"):
                del g[k]

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "tokenizer.json"), "wb") as f:
        f.write(tok_bytes)
    with open(os.path.join(args.out, "cases.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False, separators=(",", ":")) + "\n")
    manifest = dict(
        format=1,
        generator="scripts/gen_hf_tokenizer_corpus.py",
        reference=dict(
            tokenizer_json_md5=md5_bytes(tok_bytes),
            tokenizer_json_sha256=sha256_bytes(tok_bytes),
            chat_template_jinja_md5=md5_bytes(tpl_bytes),
            tokenizers_version=tokenizers.__version__,
            transformers_version=transformers.__version__,
            encode="Tokenizer.from_file(tokenizer.json).encode(text, add_special_tokens=False).ids",
            render="transformers apply_chat_template(tokenize=False, add_generation_prompt=True, **template_kwargs); "
                   "assistant tool_calls function.arguments JSON strings -> dict, content null -> ''",
        ),
        cases_file="cases.jsonl",
        n_cases=len(cases),
        n_tokens=sum(c["n_tokens"] for c in cases),
        groups=summary,
    )
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "served_differ_ids"}
                      for k, v in summary.items()}, indent=2))
    print("total cases", len(cases), "tokens", manifest["n_tokens"])


if __name__ == "__main__":
    main()
