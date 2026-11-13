#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Flags migration gate GC: map every ledger name to its legacy grammar ids.

Generates `tests/golden/flags-legacy-spelling/names.tsv` (`name<TAB>grammar[,grammar..]`,
sorted by name) from the migration ledger's `type_parse` column. The ledger column
is truncated at 21 characters on eight rows; the census column of the same name is
used when the ledger value is a prefix of it. Grammar ids are the ones defined by
`crates/hipfire-runtime/src/config/legacy_spelling_golden.rs`.

`type_parse` atom -> grammar id (the census infers atoms by regex, so a name that
mixes read styles carries several ids):

    bool(developer_bool strict)  developer_bool
    Option<bool>                 developer_typed_bool
    bool                         schema_bool      (schema ValueRule::Bool key)
    bool("1")                    eq_one           (value == "1")
    bool(!="0"/=="0")            ne_zero          (value != "0")
    presence                     presence         (is_some / is_ok)
    usize u8 u32 u64             uint_parse
    int integer i32              int_parse
    f32 f64 float                float_parse
    string path shell null       raw_string
    (empty)                      unclassified

On top of the ledger: every `parse_bool("NAME")` call in
`crates/rdna-compute/src/feature_flags.rs` adds `featureflags_parse_bool`, and the
`RuntimeConfig` lookups with their own spellings add `runtime_*` / `tri_state_alias`.

Usage:
    scripts/flags-golden-names.py --ledger L.csv [--census C.csv]   # compare
    scripts/flags-golden-names.py --ledger L.csv [--census C.csv] --write
"""
import argparse
import csv
import difflib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "tests/golden/flags-legacy-spelling/names.tsv"
GRAMMAR_GOLDEN = ROOT / "tests/golden/flags-legacy-spelling/grammars.tsv"

ATOMS = {
    'bool(developer_bool strict)': "developer_bool",
    "Option<bool>": "developer_typed_bool",
    "bool": "schema_bool",
    'bool("1")': "eq_one",
    'bool(!="0"/=="0")': "ne_zero",
    "presence": "presence",
    "usize": "uint_parse", "u8": "uint_parse", "u32": "uint_parse", "u64": "uint_parse",
    "int": "int_parse", "integer": "int_parse", "i32": "int_parse",
    "f32": "float_parse", "f64": "float_parse", "float": "float_parse",
    "string": "raw_string", "path": "raw_string", "shell": "raw_string", "null": "raw_string",
}

# RuntimeConfig::from_lookup names whose reader has its own spelling table.
RUNTIME_NAMES = {
    "HIPFIRE_NORMALIZE_PROMPT": "runtime_normalize_prompt",
    "HIPFIRE_DFLASH_NGRAM_BLOCK": "runtime_tri_on_true",
    "HIPFIRE_TP_USE_RCCL": "runtime_tp_rccl",
    "HIPFIRE_NGRAM_DRAFT": "tri_state_alias",
}


def atom_grammar(atom):
    if atom in ATOMS:
        return ATOMS[atom]
    # truncated ledger atom: unique prefix of a known atom
    hits = {g for a, g in ATOMS.items() if a.startswith(atom) and atom}
    return hits.pop() if len(hits) == 1 else None


def grammars_for(ledger_value, census_value):
    value = ledger_value
    if census_value is not None and census_value != ledger_value and census_value.startswith(ledger_value):
        value = census_value
    out = set()
    for atom in value.split("; "):
        if atom.strip():
            grammar = atom_grammar(atom)
            if grammar is None:
                raise SystemExit(f"unmapped type_parse atom {atom!r}")
            out.add(grammar)
    return out


def known_ids():
    for line in GRAMMAR_GOLDEN.read_text().splitlines():
        if line.startswith("# grammar-ids: "):
            return set(line.split(": ", 1)[1].split(","))
    raise SystemExit("grammars.tsv has no grammar-ids header; generate it first")


def render(ledger, census):
    ids = known_ids()
    census_parse = {}
    if census:
        with open(census, newline="") as fh:
            census_parse = {r["name"]: r["type_parse"] for r in csv.DictReader(fh)}
    ff_names = set(re.findall(r'parse_bool\("([A-Za-z0-9_]+)"\)',
                              (ROOT / "crates/rdna-compute/src/feature_flags.rs").read_text()))
    rows = {}
    with open(ledger, newline="") as fh:
        for r in csv.DictReader(fh):
            rows[r["name"]] = grammars_for(r["type_parse"], census_parse.get(r["name"]))
    for name in ff_names & rows.keys():
        rows[name].add("featureflags_parse_bool")
    for name, grammar in RUNTIME_NAMES.items():
        if name in rows:
            rows[name].add(grammar)
    lines = [
        "# Flags migration gate GC: ledger name -> legacy grammar ids. Do not edit by hand.",
        "# regenerate: scripts/flags-golden.sh --regen-names LEDGER.csv [CENSUS.csv]",
        "# grammar-ids: " + ",".join(sorted(ids)),
        "# columns: name<TAB>grammar[,grammar..] (sorted by name; `unclassified` = the ledger records no read style)",
    ]
    for name in sorted(rows):
        gs = sorted(rows[name]) or ["unclassified"]
        unknown = set(gs) - ids - {"unclassified"}
        if unknown:
            raise SystemExit(f"{name}: grammar ids {unknown} are not in grammars.tsv")
        lines.append(f"{name}\t{','.join(gs)}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--census")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    text = render(args.ledger, args.census)
    if args.write:
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(text)
        print(f"wrote {GOLDEN.relative_to(ROOT)} ({text.count(chr(10))} lines)")
        return 0
    want = GOLDEN.read_text()
    if want == text:
        print("names.tsv matches the ledger")
        return 0
    for line in list(difflib.unified_diff(want.splitlines(), text.splitlines(), "golden", "actual", lineterm="", n=0))[:60]:
        print(line, file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
