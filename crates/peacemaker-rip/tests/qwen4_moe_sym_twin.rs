// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The committed `.rip` twin of the gfx1201 MQ4 entries of `qwen4_moe_sym`
//! (`kernels/qwen4_moe_sym.rip`, composed from the `moent::*` / `moe1::*` natives of
//! `qwen4_moe_sym::{blocks, nt1_blocks}`) re-emits the Rust generators' bytes: `qwen4_moe_sym::emit`
//! for the flat-slot entries and `emit_down_gp` for the grouped-position twins, `s_text` and proof,
//! for `gate_up` and `down` at NT1, NT4 and NT8.

use peacemaker_isa::kernels::qwen4_moe_sym::{self as moe, Kind, Spec, WeightCodec};
use peacemaker_isa::Arch;

const TWIN: &str = include_str!("../../peacemaker-isa/src/kernels/qwen4_moe_sym.rip");
const NAME: &str = "qwen4_moe_sym.rip";

fn spec(kind: Kind, nt: u8) -> Spec {
    Spec {
        arch: Arch::Gfx1201,
        kind,
        nt,
        rr: 1,
        codec: WeightCodec::Mq4Sym,
    }
}

/// `(variant, spec, gp)` for every entry the twin declares.
fn entries() -> Vec<(String, Spec, bool)> {
    let mut out = Vec::new();
    for (kind, tag) in [(Kind::GateUp, "gate_up"), (Kind::Down, "down")] {
        out.push((tag.to_owned(), spec(kind, 1), false));
        for nt in [4u8, 8] {
            out.push((format!("{tag}_nt{nt}"), spec(kind, nt), false));
        }
    }
    for nt in [4u8, 8] {
        out.push((format!("down_gp_nt{nt}"), spec(Kind::Down, nt), true));
    }
    out
}

fn first_difference(a: &str, b: &str) -> String {
    for (i, (x, y)) in a.lines().zip(b.lines()).enumerate() {
        if x != y {
            return format!("line {}: rust `{x}` vs rip `{y}`", i + 1);
        }
    }
    format!(
        "lengths {} vs {} lines",
        a.lines().count(),
        b.lines().count()
    )
}

fn rust(spec: Spec, gp: bool) -> peacemaker_isa::Emitted {
    if gp {
        moe::emit_down_gp(spec)
    } else {
        moe::emit(spec)
    }
    .unwrap()
}

/// `s_text` and the builder proof are byte-identical for every entry.
#[test]
fn qwen4_moe_sym_twin_matches_the_rust_emit() {
    for (name, spec, gp) in entries() {
        let rust = if gp {
            moe::emit_down_gp(spec)
        } else {
            moe::emit(spec)
        }
        .unwrap_or_else(|e| panic!("{name}: {e}"));
        let rip = peacemaker_rip::compile_named(TWIN, NAME, spec.arch, &name)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(
            rust.s_text == rip.s_text,
            "{name}: {}",
            first_difference(&rust.s_text, &rip.s_text)
        );
        assert_eq!(
            serde_json::to_value(&rust.proof).unwrap(),
            serde_json::to_value(&rip.proof).unwrap(),
            "{name}: proof"
        );
    }
}

#[test]
fn the_twin_declares_exactly_its_entries() {
    let mut declared = peacemaker_rip::variants(TWIN, NAME).unwrap();
    let mut want: Vec<String> = entries().into_iter().map(|e| e.0).collect();
    declared.sort();
    want.sort();
    assert_eq!(declared, want);
}

#[test]
fn a_malformed_variant_is_refused_with_its_grammar() {
    for v in [
        "gate_up_gp_nt4",
        "down_nt1",
        "down_gp_nt16",
        "gate_up_nt2",
        "up",
    ] {
        let e = peacemaker_rip::compile_named(TWIN, NAME, Arch::Gfx1201, v)
            .err()
            .unwrap_or_else(|| panic!("{v} built"));
        assert!(
            e.contains("expected gate_up|down[_ntN] or down_gp_ntN"),
            "{v}: {e}"
        );
    }
}

#[test]
fn the_twin_builds_on_gfx1201_only() {
    let e = peacemaker_rip::compile_named(TWIN, NAME, Arch::Gfx1151, "down_nt4")
        .expect_err("gfx1151 must be refused");
    assert!(e.contains("gfx1201"), "{e}");
}

/// Edit the twin, require the edit to exist, and require that the program changes (or the checked
/// driver / builder refuses it): it never emits the original bytes.
fn changes_program(from: &str, to: &str, variant: &str) {
    let edited = TWIN.replacen(from, to, 1);
    assert_ne!(edited, TWIN, "`{from}` must exist in the twin");
    let (spec, gp) = entries()
        .into_iter()
        .find(|e| e.0 == variant)
        .map(|e| (e.1, e.2))
        .unwrap();
    let base = rust(spec, gp);
    if let Ok(moved) = peacemaker_rip::compile_named(&edited, NAME, Arch::Gfx1201, variant) {
        assert_ne!(base.s_text, moved.s_text, "{variant}: `{from}` -> `{to}`");
    }
}

/// The K-loop is the script's: swapping the descriptor step and the second weight load changes the
/// program (or the builder refuses it).
#[test]
fn reordering_the_nt_kloop_changes_the_program() {
    for v in ["down_gp_nt4", "gate_up_nt8"] {
        changes_program(
            "            moent::advance(wg)?;\n            if g.gate_up() && g.nt() == 8 {\n                wg.barrier(())?;\n            }\n            moent::load_w(wg, g, 0)?;\n",
            "            if g.gate_up() && g.nt() == 8 {\n                wg.barrier(())?;\n            }\n            moent::load_w(wg, g, 0)?;\n            moent::advance(wg)?;\n",
            v,
        );
    }
}

/// The gfx1201 NT8 `gate_up` epoch lock is the script's: one workgroup barrier per K128 epoch (at the
/// top of the trip and between the descriptor step and the second epoch's weight loads). Dropping
/// either barrier is a different program.
#[test]
fn the_nt8_gate_up_epoch_lock_is_script_level() {
    let gate = "            if g.gate_up() && g.nt() == 8 {\n                wg.barrier(())?;\n            }\n";
    assert_eq!(TWIN.matches(gate).count(), 2, "two epoch barriers");
    changes_program(
        &format!("{gate}            moent::load_w(wg, g, 1)?;\n"),
        "            moent::load_w(wg, g, 1)?;\n",
        "gate_up_nt8",
    );
    changes_program(
        &format!("{gate}            moent::load_w(wg, g, 0)?;\n"),
        "            moent::load_w(wg, g, 0)?;\n",
        "gate_up_nt8",
    );
}

#[test]
fn reordering_the_nt1_kloop_changes_the_program() {
    for v in ["gate_up", "down"] {
        changes_program(
            "            moe1::advance(wg, g)?;\n            moe1::load_set(wg, g, 0)?;\n",
            "            moe1::load_set(wg, g, 0)?;\n            moe1::advance(wg, g)?;\n",
            v,
        );
    }
}

/// Where each wait sits is the script's: the per-tile wait of the fold is a statement, so moving
/// it after the fold's liveness branch (or deleting it) is a different program, or refused.
#[test]
fn moving_the_tile_wait_changes_the_program() {
    for v in ["down_nt4", "gate_up_nt4"] {
        changes_program("            moent::wait_tile(wg, g, j)?;\n", "", v);
        changes_program(
            "            moent::wait_tile(wg, g, j)?;\n            let live = wg.scmp(moent::tile_live_insn(g, j)?)?;\n",
            "            let live = wg.scmp(moent::tile_live_insn(g, j)?)?;\n            moent::wait_tile(wg, g, j)?;\n",
            v,
        );
    }
}

/// The hand-off's stores and loads are statements: dropping the second `ds_store` of a tile's
/// publication, or the gate read-back, is refused (the read of an unpublished or partly written
/// region) or a different program; swapping the roles of the two waves changes it.
#[test]
fn the_handoff_is_script_level() {
    changes_program(
        "                        for insn in moent::publish_insns(g, j)? {\n                            w.ds_store(r, insn)?;\n                        }\n",
        "",
        "gate_up_nt4",
    );
    changes_program(
        "            for insn in moent::gate_load_insns(g, j)? {\n                w.ds_load(r, insn)?;\n            }\n",
        "",
        "gate_up_nt8",
    );
    changes_program(
        "let up = wg.scmp(moent::up_insn())?;",
        "let up = wg.scmp(moent::tile_live_insn(g, 1)?)?;",
        "gate_up_nt4",
    );
}

/// The prologue's exits are statements: removing the sentinel exit changes the program.
#[test]
fn dropping_an_exit_changes_the_program() {
    changes_program(
        "    wg.exit_if(sentinel, end)?;\n    moent::pro_run(wg, g)?;",
        "    moent::pro_run(wg, g)?;",
        "down_nt4",
    );
    changes_program(
        "    wg.exit_if(sentinel, end)?;\n    moe1::pro_setup(wg, g)?;",
        "    moe1::pro_setup(wg, g)?;",
        "down",
    );
}

/// A native refuses an out-of-range tile or parity before it emits anything.
#[test]
fn natives_refuse_out_of_range_indices() {
    for (from, to, want) in [
        (
            "moent::load_w(wg, g, 0)?;\n            compute(wg, g, 1,",
            "moent::load_w(wg, g, 2)?;\n            compute(wg, g, 1,",
            "epoch parity 2",
        ),
        (
            "moent::pro_slot_load(wg, g, j)?;",
            "moent::pro_slot_load(wg, g, j + 8)?;",
            "tile 8",
        ),
        (
            "moent::fold_tile(w, g, p, j)",
            "moent::fold_tile(w, g, p, j + 4)",
            "tile 5",
        ),
    ] {
        let edited = TWIN.replacen(from, to, 1);
        assert_ne!(edited, TWIN, "`{from}` must exist in the twin");
        let e = peacemaker_rip::compile_named(&edited, NAME, Arch::Gfx1201, "down_nt4")
            .err()
            .unwrap_or_else(|| panic!("`{to}` built"));
        assert!(e.contains(want), "{to}: {e}");
    }
}
