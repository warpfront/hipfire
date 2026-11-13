#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Flags migration gate GB, static half: call-site defaults.

Extracts every production call of `developer_bool(NAME, default)`,
`developer_typed_bool(NAME)`, `developer_var(NAME)` and `developer_var_os(NAME)`
from `crates/*/src/**/*.rs` (comments and `#[cfg(test)] mod` blocks removed) and
writes sorted, de-duplicated rows

    developer_callsite<TAB>*<TAB>NAME#fn=default

to `tests/golden/flags-defaults/developer-callsites.tsv`. The default is
file-independent: the normalised second argument of `developer_bool`, `-` for
`developer_typed_bool`, and for `developer_var[_os]` the method chain applied to
the result up to its first terminator (`ok.unwrap_or("x")`, `ok.as_deref.==Some("1")`,
`is_ok`, ...). A name that is a `const NAME: &str = "HIPFIRE_..."` is resolved
file-first, then crate-wide, then tree-wide; one that is not a literal or known const is reported as `?<expr>`.

Usage (from anywhere):
    scripts/flags-golden-extract.py            # compare with the committed golden
    scripts/flags-golden-extract.py --write    # regenerate it (review the diff)
"""
import difflib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "tests/golden/flags-defaults/developer-callsites.tsv"
REGEN = "scripts/flags-golden.sh --regen"

CALL = re.compile(r"\b(developer_bool|developer_typed_bool|developer_var_os|developer_var)\s*\(")
CONST = re.compile(r'\b(?:pub(?:\([a-z]+\))?\s+)?(?:const|static)\s+([A-Z][A-Z0-9_]*)\s*:\s*&(?:\'static\s+)?str\s*=\s*"([^"\\]*)"\s*;')
ALIAS = re.compile(r'\b(?:pub(?:\([a-z]+\))?\s+)?const\s+([A-Z][A-Z0-9_]*)\s*:\s*&(?:\'static\s+)?str\s*=\s*(?:[A-Za-z_]\w*::)*([A-Z][A-Z0-9_]*)\s*;')
TERMINATORS = {
    "unwrap_or", "unwrap_or_else", "unwrap_or_default", "is_ok", "is_err", "is_some", "is_none",
    "map_or", "map_or_else", "unwrap", "expect", "eq_ignore_ascii_case",
}


def strip_comments(src):
    """Remove // and nested /* */ comments; string and char literals are kept intact."""
    out = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if src.startswith("//", i):
            while i < n and src[i] != "\n":
                i += 1
        elif src.startswith("/*", i):
            depth = 1
            i += 2
            while i < n and depth:
                if src.startswith("/*", i):
                    depth += 1
                    i += 2
                elif src.startswith("*/", i):
                    depth -= 1
                    i += 2
                else:
                    i += 1
            out.append(" ")
        elif c == '"' or (c == "r" and re.match(r'r#*"', src[i:i + 12]) and not (i and (src[i - 1].isalnum() or src[i - 1] == "_"))):
            j = skip_string(src, i)
            out.append(src[i:j])
            i = j
        elif c == "'":
            j = skip_char(src, i)
            out.append(src[i:j])
            i = j
        else:
            out.append(c)
            i += 1
    return "".join(out)


def skip_string(s, i):
    if s[i] == "r":
        m = re.match(r'r(#*)"', s[i:])
        end = '"' + m.group(1)
        j = s.find(end, i + len(m.group(0)))
        return len(s) if j < 0 else j + len(end)
    j = i + 1
    while j < len(s):
        if s[j] == "\\":
            j += 2
        elif s[j] == '"':
            return j + 1
        else:
            j += 1
    return j


def skip_char(s, i):
    """Return the end of a char literal at s[i] == "'" (a lifetime consumes only the quote)."""
    if s[i + 1:i + 2] == "\\":
        j = s.find("'", i + 2)
        return i + 1 if j < 0 else j + 1
    if s[i + 2:i + 3] == "'":
        return i + 3
    return i + 1


def match_close(s, i):
    """s[i] is an opening bracket; return the index after its matching closer."""
    pairs = {"(": ")", "[": "]", "{": "}"}
    stack = [pairs[s[i]]]
    j = i + 1
    while j < len(s) and stack:
        c = s[j]
        if c == '"' or (c == "r" and re.match(r'r#*"', s[j:j + 12]) and not (s[j - 1].isalnum() or s[j - 1] == "_")):
            j = skip_string(s, j)
            continue
        if c == "'":
            j = skip_char(s, j)
            continue
        if c in pairs:
            stack.append(pairs[c])
        elif c == stack[-1]:
            stack.pop()
        j += 1
    return j


def drop_test_modules(src):
    out, pos = [], 0
    for m in re.finditer(r"#\[cfg\(test\)\]\s*(?:#\[[^\]]*\]\s*)*(?:pub(?:\([a-z]+\))?\s+)?mod\s+\w+\s*\{", src):
        if m.start() < pos:
            continue
        brace = m.end() - 1
        out.append(src[pos:m.start()])
        pos = match_close(src, brace)
    out.append(src[pos:])
    return "".join(out)


def split_args(inner):
    args, depth, cur, j = [], 0, [], 0
    while j < len(inner):
        c = inner[j]
        if c == '"' or (c == "r" and re.match(r'r#*"', inner[j:j + 12])):
            k = skip_string(inner, j)
            cur.append(inner[j:k])
            j = k
            continue
        if c == "'":
            k = skip_char(inner, j)
            cur.append(inner[j:k])
            j = k
            continue
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        if c == "," and depth == 0:
            args.append("".join(cur))
            cur = []
        else:
            cur.append(c)
        j += 1
    tail = "".join(cur)
    if tail.strip():
        args.append(tail)
    return args


def norm(text, limit=80):
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def tail_chain(src, end):
    """Method chain applied to the call result, up to and including its first terminator."""
    parts, j = [], end
    while True:
        m = re.compile(r"\s*\.\s*([A-Za-z_]\w*)").match(src, j)
        if not m:
            break
        name, j = m.group(1), m.end()
        arg = ""
        if src[j:j + 1] == "(":
            k = match_close(src, j)
            arg = src[j + 1:k - 1]
            j = k
        if name in ("unwrap_or", "map_or") and arg:
            parts.append(f"{name}({norm(arg, 40)})")
        else:
            parts.append(name)
        if name in TERMINATORS:
            break
    m = re.compile(r"\s*(==|!=)\s*((?:Some|Ok|Err)\(\s*\"[^\"]{0,20}\"\s*\)|\"[^\"]{0,20}\")").match(src, j)
    if m:
        parts.append(f"{m.group(1)}{norm(m.group(2), 30)}")
    return ".".join(parts) if parts else "-"


def source_files():
    for path in sorted(ROOT.glob("crates/*/src/**/*.rs")):
        yield path


def extract():
    texts = {}
    for path in source_files():
        texts[path] = drop_test_modules(strip_comments(path.read_text(errors="replace")))
    by_file, by_crate, by_all = {}, {}, {}
    for path, text in texts.items():
        crate = path.relative_to(ROOT).parts[1]
        for m in CONST.finditer(text):
            by_file.setdefault((path, m.group(1)), set()).add(m.group(2))
            by_crate.setdefault((crate, m.group(1)), set()).add(m.group(2))
            by_all.setdefault(m.group(1), set()).add(m.group(2))
    for _ in range(4):  # resolve `const A: &str = path::B;` alias chains
        for path, text in texts.items():
            crate = path.relative_to(ROOT).parts[1]
            for m in ALIAS.finditer(text):
                target = by_all.get(m.group(2))
                if target and len(target) == 1:
                    by_file.setdefault((path, m.group(1)), set()).update(target)
                    by_crate.setdefault((crate, m.group(1)), set()).update(target)
                    by_all.setdefault(m.group(1), set()).update(target)
    rows = set()
    for path, text in texts.items():
        for m in CALL.finditer(text):
            if re.search(r"\bfn\s+$", text[: m.start()]):
                continue
            func = m.group(1)
            paren = m.end() - 1
            close = match_close(text, paren)
            args = split_args(text[paren + 1:close - 1])
            if not args:
                continue
            name_arg = args[0].strip()
            lit = re.fullmatch(r'"([^"\\]*)"', name_arg)
            if lit:
                name = lit.group(1)
            else:
                ident = re.fullmatch(r"(?:[A-Za-z_][\w]*::)*([A-Z][A-Z0-9_]*)", name_arg)
                values = None
                if ident:
                    key = ident.group(1)
                    crate = path.relative_to(ROOT).parts[1]
                    for table, tkey in ((by_file, (path, key)), (by_crate, (crate, key)), (by_all, key)):
                        if tkey in table:
                            values = table[tkey]
                            if len(values) == 1:
                                break
                if values and len(values) == 1:
                    name = next(iter(values))
                elif ident and values:
                    name = f"?const:{ident.group(1)}"
                else:
                    name = f"?{norm(name_arg, 40)}"
            if func == "developer_bool":
                default = norm(args[1]) if len(args) > 1 else "?missing"
            elif func == "developer_typed_bool":
                default = "-"
            else:
                default = tail_chain(text, close)
            rows.add(f"developer_callsite\t*\t{name}#{func}={default}")
    return sorted(rows)


def render(rows):
    head = [
        "# Flags migration gate GB, static half: call-site defaults. Do not edit by hand.",
        f"# regenerate: {REGEN}",
        "# extractor: scripts/flags-golden-extract.py (production crates/*/src, comments and cfg(test) mods removed)",
        "# columns: section<TAB>arch<TAB>NAME#fn=default (sorted, deduplicated)",
    ]
    return "\n".join(head + rows) + "\n"


def main(argv):
    text = render(extract())
    if "--write" in argv:
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(text)
        print(f"wrote {GOLDEN.relative_to(ROOT)} ({text.count(chr(10))} lines)")
        return 0
    if not GOLDEN.exists():
        print(f"missing {GOLDEN.relative_to(ROOT)}; run {REGEN}", file=sys.stderr)
        return 1
    want = GOLDEN.read_text()
    if want == text:
        print(f"developer call-site defaults match {GOLDEN.relative_to(ROOT)}")
        return 0
    diff = list(difflib.unified_diff(want.splitlines(), text.splitlines(), "golden", "actual", lineterm="", n=0))
    print("developer call-site defaults diverged from the golden; if intended run " + REGEN, file=sys.stderr)
    for line in diff[:60]:
        print(line, file=sys.stderr)
    if len(diff) > 60:
        print(f"... {len(diff) - 60} more diff lines", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
