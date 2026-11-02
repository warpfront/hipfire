// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Drift ratchet: every real ISA instruction name spelled as a string literal
//! in the emulator's `alu`/`memory`/`mma`/`trans`/`wide` modules must have a
//! row in `isa/semantics.tbl`, so `Emulator::new` never fails with
//! "no semantics for {name}" on a name the handlers already spell.
//!
//! Candidate filter: a literal counts only if it is a real instruction name,
//! i.e. column 1 of one of the decode tables (`isa/gfx1100.tbl`,
//! `isa/gfx1151.tbl`, `isa/gfx12.tbl`). That filter excludes handler-internal
//! spellings "naturally", with no hand allowlist: `stem()`-stripped forms
//! (`v_add_f32` for decoded `v_add_f32_e32`), `strip_prefix` fragments
//! (`s_cmp_`, `v_cmp_`, `v_cmpx_`, `s_load_`), `starts_with` family gates
//! (`v_cmp`, `v_dot`, `v_wmma`, `v_swmmac`, `ds_`, `buffer_`, `global_`,
//! `flat_`), width-token probes (`2addr`), and test-only adversarial inputs
//! (`v_madeup_f32_e32`). None of those ever reaches `semantic_class` as a
//! decoded name, so none needs a table row.
//!
//! The reverse is deliberately NOT checked: many `semantics.tbl` rows have no
//! handler literal because one handler covers a whole family via prefix
//! dispatch (`starts_with("v_cmp")` serves every `v_cmp_*` row) or a shared
//! stem arm (`match stem(name)` serves each `_e32`/`_e64`/`_dpp` row through
//! one bare-stem arm). A row-per-literal requirement in that direction would
//! forbid family handling.

use std::collections::HashSet;

const HANDLERS: [&str; 5] = [
    include_str!("../src/emu/alu.rs"),
    include_str!("../src/emu/memory.rs"),
    include_str!("../src/emu/mma.rs"),
    include_str!("../src/emu/trans.rs"),
    include_str!("../src/emu/wide.rs"),
];

/// Column 1 of the decode tables: the real instruction names.
fn decode_names() -> HashSet<&'static str> {
    const TABLES: [&str; 3] = [
        include_str!("../isa/gfx1100.tbl"),
        include_str!("../isa/gfx1151.tbl"),
        include_str!("../isa/gfx12.tbl"),
    ];
    let mut names = HashSet::new();
    for table in TABLES {
        for line in table.lines() {
            if line.starts_with('#') || line.trim().is_empty() {
                continue;
            }
            if let Some((name, _)) = line.split_once('\t') {
                names.insert(name);
            }
        }
    }
    names
}

/// Column 1 of `semantics.tbl`: every name `Emulator::new` admits.
fn semantic_rows() -> HashSet<&'static str> {
    let mut rows = HashSet::new();
    for line in include_str!("../isa/semantics.tbl").lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        if let Some((name, _)) = line.split_once('\t') {
            rows.insert(name);
        }
    }
    rows
}

/// `^"...` string-literal scan: a literal is the bytes between a `"` that
/// opens one (previous char is not part of an escape or identifier) and the
/// next unescaped `"`. Handler sources use plain `"..."` literals only.
fn literals(source: &str) -> Vec<&str> {
    let bytes = source.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'"' {
            let mut j = i + 1;
            let mut escaped = false;
            while j < bytes.len() && (escaped || bytes[j] != b'"') {
                escaped = !escaped && bytes[j] == b'\\';
                j += 1;
            }
            if j < bytes.len() {
                out.push(&source[i + 1..j]);
                i = j + 1;
                continue;
            }
            break;
        }
        i += 1;
    }
    out
}

fn is_mnemonic_shape(literal: &str) -> bool {
    let stem = literal
        .strip_suffix("_e32")
        .or_else(|| literal.strip_suffix("_e64"))
        .or_else(|| literal.strip_suffix("_dpp"))
        .unwrap_or(literal);
    let (head, tail) = match stem.find('_') {
        Some(at) => (&stem[..at], &stem[at + 1..]),
        None => return false,
    };
    matches!(
        head,
        "v" | "s" | "ds" | "buffer" | "global" | "scratch" | "flat" | "image"
    ) && !tail.is_empty()
        && tail
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
}

/// Every real ISA name spelled in the handlers must have a semantics row.
/// Stems, prefixes, fragments and test-only inputs never reach
/// `semantic_class` as decoded names, so the decode-table filter excludes
/// them without an allowlist.
#[test]
fn handler_literals_are_covered_by_semantics_tbl() {
    let decoded = decode_names();
    let rows = semantic_rows();
    let mut missing: Vec<String> = Vec::new();
    for source in HANDLERS {
        let mut seen = HashSet::new();
        for literal in literals(source) {
            if !is_mnemonic_shape(literal) || !seen.insert(literal) {
                continue;
            }
            if decoded.contains(literal) && !rows.contains(literal) {
                missing.push(literal.to_string());
            }
        }
    }
    missing.sort();
    missing.dedup();
    assert!(
        missing.is_empty(),
        "handler spells real ISA names with no semantics.tbl row: {missing:?}"
    );
}

/// The decode tables themselves parse: spot-check the filter's ground truth
/// so a malformed table cannot silently empty the candidate set.
#[test]
fn decode_tables_yield_candidates() {
    let decoded = decode_names();
    assert!(decoded.len() > 400, "unexpectedly few decode rows");
    for known in [
        "v_add_f32_e32",
        "s_sub_u32",
        "ds_load_b64",
        "v_bfi_b32",
        "s_cmp_eq_u64",
    ] {
        assert!(decoded.contains(known), "decode tables lack {known}");
    }
    let rows = semantic_rows();
    assert!(rows.len() > 250, "unexpectedly few semantics rows");
}
