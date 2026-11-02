// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The committed `.rip` twin of `mq6_i8_gemm::emit` (`kernels/mq6_i8_gemm.rip`, composed from the
//! `mq6::*` natives and the checked driver's rings) re-emits the production kernel's bytes for every
//! valid `Spec` on gfx1201, and the coarse natives return the Rust functions' values.

use peacemaker_isa::kernels::mq6_i8_gemm::{self as mq6, Epi, Raster, RingDepth, Spec, Tile, Tune};
use peacemaker_isa::Arch;

const TWIN: &str = include_str!("../../peacemaker-isa/src/kernels/mq6_i8_gemm.rip");

fn every_spec() -> Vec<Spec> {
    let mut out = Vec::new();
    for tile in Tile::ALL {
        for epi in [Epi::Set, Epi::SetBf16] {
            for raster in [Raster::Banded, Raster::Linear] {
                out.push(Spec {
                    epi,
                    raster,
                    tile,
                    arch: Arch::Gfx1201,
                    tune: Default::default(),
                });
            }
        }
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

/// `s_text` is byte-identical for every valid spec. The proof is compared too: it records the
/// builder's facts about the program (variant, symbol, resource, wait and hazard digests), which a
/// `.rip` composition of the same calls reproduces; a difference would mean another program.
#[test]
fn mq6_i8_gemm_twin_matches_the_rust_emit() {
    let mut valid = 0;
    for spec in every_spec() {
        let name = spec.variant_name();
        if spec.validate().is_err() {
            continue;
        }
        valid += 1;
        let rust = mq6::emit(spec).unwrap_or_else(|e| panic!("{name}: {e}"));
        let rip = peacemaker_rip::compile_named(TWIN, "mq6_i8_gemm.rip", spec.arch, &name)
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
    assert_eq!(
        valid, 12,
        "the number of valid specs changed: update VARIANTS"
    );
}

/// The twin with `const RING = 3`: the depth-3 composition (`block3`, `group3`, `head3`, `arm3`,
/// `kloop3` over `Ring::new(a, b, c)`, `starting_at`, `advance` and `wg.wg_if_else`).
fn twin_ring3() -> String {
    let text = TWIN.replace("const RING: u32 = 2;", "const RING: u32 = 3;");
    assert_ne!(text, TWIN, "the twin declares `const RING: u32 = 2;`");
    text
}

/// With `const RING = 3` the script emits, byte for byte and proof for proof, the generator's
/// depth-3 kernel (`Tune { ring: Three }`) for every variant that fits, and refuses the others with
/// `Spec::validate`'s numbers (the 256-wide tiles do not fit: LDS and VGPRs).
#[test]
fn the_ring3_twin_matches_the_generators_depth_3() {
    let src = twin_ring3();
    let mut fits = 0;
    for spec in every_spec() {
        let name = spec.variant_name();
        let spec3 = spec.with_tune(Tune {
            ring: RingDepth::Three,
        });
        let rip = peacemaker_rip::compile_named(&src, "mq6_i8_gemm.rip", spec.arch, &name);
        match spec3.validate() {
            Ok(()) => {
                fits += 1;
                let rust = mq6::emit(spec3).unwrap_or_else(|e| panic!("{name}: {e}"));
                let rip = rip.unwrap_or_else(|e| panic!("{name}: {e}"));
                assert!(
                    rust.s_text == rip.s_text,
                    "{name} ring 3: {}",
                    first_difference(&rust.s_text, &rip.s_text)
                );
                assert_eq!(
                    serde_json::to_value(&rust.proof).unwrap(),
                    serde_json::to_value(&rip.proof).unwrap(),
                    "{name} ring 3: proof"
                );
                // It is the depth-3 program, not the depth-2 one.
                let d2 = mq6::emit(spec).unwrap();
                assert_ne!(rip.s_text, d2.s_text, "{name}");
            }
            Err(e) => {
                let got = rip.err().unwrap_or_else(|| panic!("{name} ring 3 built"));
                assert!(got.contains(&e), "{name}: {got} vs {e}");
            }
        }
    }
    assert_eq!(
        fits, 4,
        "128x128x8 x {{set, set_bf16}} x {{banded, linear}}"
    );
}

#[test]
fn the_twin_declares_exactly_the_valid_variants() {
    let mut declared = peacemaker_rip::variants(TWIN, "mq6_i8_gemm.rip").unwrap();
    let mut valid: Vec<String> = every_spec()
        .into_iter()
        .filter(|s| s.validate().is_ok())
        .map(|s| s.variant_name())
        .collect();
    valid.sort();
    declared.sort();
    assert_eq!(declared, valid);
}

#[test]
fn a_malformed_variant_is_refused_with_its_grammar() {
    for (v, want) in [
        ("128x128x8", "expected TILE-EPI[-linear]"),
        ("64x64x4-set", "unknown tile `64x64x4`"),
        ("512x128x32-set", "unknown tile `512x128x32`"),
        ("128x128x8-mul", "unknown epilogue `mul`"),
        ("128x128x8-set-banded", "unknown raster `banded`"),
    ] {
        let e = peacemaker_rip::compile_named(TWIN, "mq6_i8_gemm.rip", Arch::Gfx1201, v)
            .err()
            .unwrap_or_else(|| panic!("{v} built"));
        assert!(e.contains(want), "{v}: {e}");
    }
}

#[test]
fn the_twin_builds_on_gfx1201_only() {
    let e = peacemaker_rip::compile_named(TWIN, "mq6_i8_gemm.rip", Arch::Gfx1151, "128x128x8-set")
        .expect_err("gfx1151 must be refused");
    assert!(e.contains("mq6_i8_gemm targets gfx1201 only"), "{e}");
}

// ---- coarse natives ---------------------------------------------------------

const COARSE: &str = r#"
fn emit(spec: Spec) -> Result<Emitted, String> {
    Mq6Gemm::emit(Mq6Spec { epi: Mq6Epi::SetBf16, raster: Raster::Linear, tile: Mq6Tile::T256x128, arch: spec.arch })
}
"#;

#[test]
fn mq6_emit_native_returns_the_rust_emitted() {
    let want = mq6::emit(Spec {
        epi: Epi::SetBf16,
        raster: Raster::Linear,
        tile: Tile::T256x128,
        arch: Arch::Gfx1201,
        tune: Default::default(),
    })
    .unwrap();
    let got = peacemaker_rip::compile(COARSE, Arch::Gfx1201, "default").unwrap();
    assert_eq!(want.s_text, got.s_text);
    // Spec::validate stays the gate.
    let e = peacemaker_rip::compile(COARSE, Arch::Gfx1151, "default")
        .err()
        .unwrap();
    assert!(e.contains("mq6_i8_gemm targets gfx1201 only"), "{e}");
}

/// `Mq6Spec` takes one optional field, `ring` (the A/W ring depth, 2 | 3): absent means 2.
#[test]
fn mq6_spec_ring_field_selects_the_depth() {
    let src = |fields: &str| {
        format!(
            r#"
fn emit(spec: Spec) -> Result<Emitted, String> {{
    Mq6Gemm::emit(Mq6Spec {{ epi: Mq6Epi::Set, raster: Raster::Linear, tile: Mq6Tile::T128x128, arch: spec.arch{fields} }})
}}
"#
        )
    };
    let base = Spec::emulation(Epi::Set);
    let d3 = base.with_tune(Tune {
        ring: RingDepth::Three,
    });
    let got = |fields: &str| peacemaker_rip::compile(&src(fields), Arch::Gfx1201, "default");
    assert_eq!(got("").unwrap().s_text, mq6::emit(base).unwrap().s_text);
    assert_eq!(
        got(", ring: 2").unwrap().s_text,
        mq6::emit(base).unwrap().s_text
    );
    assert_eq!(
        got(", ring: 3").unwrap().s_text,
        mq6::emit(d3).unwrap().s_text
    );
    let e = got(", ring: 4").err().unwrap();
    assert!(e.contains("unknown ring depth 4"), "{e}");
    // A tile that does not fit is refused by the generator's validate, with the numbers.
    let wide = src(", ring: 3").replace("T128x128", "T128x256");
    let e = peacemaker_rip::compile(&wide, Arch::Gfx1201, "default")
        .err()
        .unwrap();
    assert!(
        e.contains("ring depth 3 does not fit the 128x256x16 tile"),
        "{e}"
    );
    assert!(e.contains("79872 B"), "{e}");
}

const COARSE_FIELDS: &str = r#"
fn emit(spec: Spec) -> Result<Emitted, String> {
    Mq6Gemm::emit(Mq6Spec { epi: Mq6Epi::Set, raster: Raster::Banded, arch: spec.arch })
}
"#;

#[test]
fn mq6_spec_requires_every_field() {
    let e = peacemaker_rip::compile(COARSE_FIELDS, Arch::Gfx1201, "default")
        .err()
        .unwrap();
    assert!(e.contains("missing the field `tile`"), "{e}");
}

const MODULE: &str = r#"
fn emit(spec: Spec) -> Result<Emitted, String> {
    let (all, text, proof) = Mq6Gemm::emit_module(spec.arch)?;
    assert!(text.contains(".amdgpu_metadata"));
    assert!(proof.contains("s_text_sha256"));
    Ok(all[1])
}
fn count() -> usize { let (a, _t, _p) = Mq6Gemm::emit_module(Arch::Gfx1201).unwrap(); a.len() }
fn text() -> String { let (_a, t, _p) = Mq6Gemm::emit_module(Arch::Gfx1201).unwrap(); t }
fn proof() -> String { let (_a, _t, p) = Mq6Gemm::emit_module(Arch::Gfx1201).unwrap(); p }
"#;

#[test]
fn mq6_emit_module_returns_the_rust_module() {
    let (emitted, text, proof) = mq6::emit_module(Arch::Gfx1201).unwrap();
    let got = peacemaker_rip::compile(MODULE, Arch::Gfx1201, "default").unwrap();
    assert_eq!(emitted[1].s_text, got.s_text);
    assert_eq!(
        peacemaker_rip::eval(MODULE, "count").unwrap(),
        emitted.len().to_string()
    );
    assert_eq!(
        peacemaker_rip::eval(MODULE, "text").unwrap(),
        format!("{text:?}")
    );
    let json = serde_json::to_string(&proof).unwrap();
    assert_eq!(
        peacemaker_rip::eval(MODULE, "proof").unwrap(),
        format!("{json:?}")
    );
}

// ---- compositions the script can change -------------------------------------------

const SET: &str = "128x128x8-set";

fn build(src: &str, variant: &str) -> Result<peacemaker_isa::Emitted, String> {
    peacemaker_rip::compile_named(src, "variation.rip", Arch::Gfx1201, variant)
}

fn edit(from: &str, to: &str) -> String {
    assert!(TWIN.contains(from), "the twin no longer contains `{from}`");
    TWIN.replacen(from, to, 1)
}

/// A timing experiment: run the K-loop at the default wave priority.
#[test]
fn a_variation_without_the_k_loop_priority_builds_and_differs() {
    let src = edit(
        "        op(wg.isa(), \"s_setprio 1\", vec![], vec![])?;\n",
        "",
    );
    let base = build(TWIN, SET).unwrap();
    let varied = build(&src, SET).unwrap();
    assert_eq!(base.s_text.matches("s_setprio 1").count(), 1);
    assert_eq!(varied.s_text.matches("s_setprio 1").count(), 0);
}

/// Decode placement: the decode of the next slab starts after the 12th WMMA of phase A (instead of
/// the 8th), three units per WMMA: the same instructions, another order, so the program differs but
/// every counted instruction stays.
#[test]
fn moving_the_decode_between_other_wmmas_reorders_without_losing_work() {
    let src = edit(
        "phase(wg, g, &mut st, 0, true, 8, 3,",
        "phase(wg, g, &mut st, 0, true, 12, 3,",
    );
    let base = build(TWIN, SET).unwrap();
    let varied = build(&src, SET).unwrap();
    assert_ne!(base.s_text, varied.s_text);
    let count = |t: &str, m: &str| t.lines().filter(|l| l.trim_start().starts_with(m)).count();
    for m in ["v_wmma_", "v_perm_b32", "ds_store_b64", "v_dual_"] {
        assert_eq!(count(&base.s_text, m), count(&varied.s_text, m), "{m}");
    }
}

/// Fold order: row group 1's four fragments in the opposite order. The fragments are independent
/// (each reads its own `cacc` octet and accumulators; the `sc`/`z'` octets and the product octet are
/// reloaded per row group), so the program stays correct; the 20 packets of each fragment stay, in
/// another sequence. (Reordering the stages INSIDE a fragment is not free: stage 2 overwrites the
/// product octet stage 1 reads.)
#[test]
fn reordering_the_row_group_1_fragments_changes_the_packet_sequence() {
    let src = edit(
        "    for nb in 0..4usize {\n        frag(wg, g, nb, 1, cacc, scratch)?;\n    }\n",
        "    for nb in [3usize, 2, 1, 0] {\n        frag(wg, g, nb, 1, cacc, scratch)?;\n    }\n",
    );
    let base = build(TWIN, SET).unwrap();
    let varied = build(&src, SET).unwrap();
    assert_ne!(base.s_text, varied.s_text);
    let count = |t: &str| t.lines().filter(|l| l.contains("v_dual_")).count();
    assert_eq!(count(&base.s_text), count(&varied.s_text));
}

/// The next block's token word / row header fetch is a queued work unit (issued behind the half-1
/// refetch, after WMMA 7). Split-barrier position (B1): it moves after the signal instead, so it
/// overlaps the rendezvous together with the slab-0 fetch.
const META_QUEUED: &str = "    if fetch_next {\n        // Early producer: the next block's token word / row header issue right behind the half-1\n        // refetch (after WMMA 7), so they are long landed when phase B's Meta consumes them.\n        st.work.push(Work::FetchMeta(h, if h == 0 { 4 } else { group }));\n    }\n";
const META_AFTER_SIGNAL: &str = "    if fetch_next {\n        mq6::fetch_meta(wg, g, h, if h == 0 { 4 } else { group })?;\n    }\n";

#[test]
fn signalling_b1_before_the_meta_fetch_builds_and_differs() {
    let signal = "    let b1 = wg.signal((rotate(st.a, da), rotate(st.w, dw)))?;\n";
    assert!(TWIN.contains(META_QUEUED) && TWIN.contains(signal));
    let moved = TWIN.replacen(META_QUEUED, "", 1).replacen(
        signal,
        &format!("{signal}{META_AFTER_SIGNAL}"),
        1,
    );
    let base = build(TWIN, SET).unwrap();
    let varied = build(&moved, SET).unwrap();
    assert_ne!(base.s_text, varied.s_text);
    // Same instructions; only the position of the s_barrier_signal relative to the meta loads moved.
    let instructions = |t: &str| t.lines().filter(|l| l.starts_with('\t')).count();
    assert_eq!(instructions(&base.s_text), instructions(&varied.s_text));
}

/// The wait before a raw staging refill is the script's (`st.drain`), and without it the builder
/// still guards the refill: the staging registers are still sources of the previous half's LDS
/// stores, so `fetch_a` settles that hazard itself (the minimal `s_wait_dscnt`, outside the clause)
/// instead of the script's ledger-wide wait. A different program that builds.
#[test]
fn dropping_the_wait_before_a_staging_refill_leaves_the_hazard_to_the_builder() {
    let src = edit(
        "                // The raw staging registers are still sources of the previous half's stores.\n                if st.drain {\n                    mq6::wait_all(wg)?;\n                }\n",
        "",
    );
    let base = build(TWIN, SET).unwrap();
    let varied = build(&src, SET).unwrap();
    assert_ne!(base.s_text, varied.s_text);
}

/// A second wait right after the refill (the wait moved later as well as kept) is a different
/// program that still builds.
#[test]
fn a_wait_after_the_refill_changes_the_program() {
    let src = edit(
        "                mq6::fetch_a(wg, g, slab, sb, srd)?;\n",
        "                mq6::fetch_a(wg, g, slab, sb, srd)?;\n                mq6::wait_all(wg)?;\n",
    );
    let base = build(TWIN, SET).unwrap();
    let varied = build(&src, SET).unwrap();
    assert_ne!(base.s_text, varied.s_text);
}

// ---- the checked driver refuses what the typed core refuses ----------------------

#[test]
fn a_split_barrier_that_is_never_waited_is_refused() {
    let src = edit("    wg.wait_arrived(b1)?;\n", "");
    let e = build(&src, SET).expect_err("a signalled barrier must be waited");
    assert!(
        e.contains("held by a split barrier in flight until wait_arrived"),
        "{e}"
    );
}

#[test]
fn a_signal_without_the_store_drain_is_refused() {
    // The tokens must come from the wait that drained these very stores.
    let src = edit(
        "    let (da, dw) = wg.wait_all((st.a, st.w))?;\n    let b1 = wg.signal((rotate(st.a, da), rotate(st.w, dw)))?;\n",
        "    let b1 = wg.signal((rotate(st.a), rotate(st.w)))?;\n",
    );
    let e = build(&src, SET).expect_err("an undrained ring cannot be rotated");
    assert!(
        e.contains("with an undrained LDS store: wait for it first"),
        "{e}"
    );
}

#[test]
fn a_block_must_run_on_the_plane_the_rings_are_on() {
    let src = edit("\"m0\", 0, true", "\"m0\", 1, true");
    let e = build(&src, SET).expect_err("an out-of-phase block must be refused");
    assert!(e.contains("rings out of phase with plane 1"), "{e}");
}

#[test]
fn trips_cmp_takes_only_scalar_compares() {
    let src = edit("\"s_cmp_eq_u32\"", "\"s_mov_b32\"");
    let e = build(&src, SET).expect_err("a non-compare must be refused");
    assert!(e.contains("is not a 32-bit scalar compare"), "{e}");
}

#[test]
fn a_block_native_refuses_out_of_range_indices() {
    for (from, to, want) in [
        (
            "mq6::mma(wg, g, step, set, first && step == 0, i)?;",
            "mq6::mma(wg, g, step, set, first && step == 0, 8)?;",
            "mma: step",
        ),
        (
            "queue_decode(&mut st, mq6::decode_insns(g, 0)?);\n    st.work.push(Work::StoreW(0));",
            "queue_decode(&mut st, mq6::decode_insns(g, 2)?);\n    st.work.push(Work::StoreW(0));",
            "decode_round: round 2 out of range",
        ),
        (
            "mq6::fold_addr(wg, g, h)?;",
            "mq6::fold_addr(wg, g, 2)?;",
            "fold_addr: plane 2",
        ),
    ] {
        let e = build(&edit(from, to), SET).expect_err(want);
        assert!(e.contains(want), "{want}: {e}");
    }
}

// ---- a variation keeps the kernel's output (the CPU interpreter) -----------------------------
//
// The builder proves the ring protocol, the waits and the register lifetimes; it does not prove the
// arithmetic. A moved decode, a moved split barrier or another fold order must leave the output
// bit-identical, which the interpreter runs on the linear-raster twin (the banded raster needs SALU
// float ops the interpreter lacks).

mod interpreter {
    use peacemaker_ir::emu::{Emulator, Launch, Memory};
    use peacemaker_ir::inst::Frontend;
    use peacemaker_isa::kernels::mq6_i8_gemm::Tile;
    use peacemaker_isa::{Arch, Emitted};

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    /// Normal f16 with exponent field `6..18` (keeps every fold product a normal f32).
    fn half(rng: &mut Rng) -> u16 {
        (((rng.next() & 1) as u16) << 15)
            | (((6 + rng.below(12)) as u16) << 10)
            | (rng.next() as u16 & 0x3ff)
    }

    /// `A`: `[M][K/256][200]` MQ6G256V2 groups; `Xq`: `[K/128][N][136]` `block_i8_128` blocks.
    fn data(m: usize, k: usize, n: usize, seed: u64) -> (Vec<u8>, Vec<u8>) {
        let rng = &mut Rng(seed);
        let g = k / 256;
        let mut a = vec![0u8; m * g * 200];
        for group in a.chunks_mut(200) {
            for h in 0..2 {
                let (s, z) = (half(rng), half(rng));
                group[4 * h..4 * h + 2].copy_from_slice(&s.to_le_bytes());
                group[4 * h + 2..4 * h + 4].copy_from_slice(&z.to_le_bytes());
            }
            for byte in &mut group[8..] {
                *byte = rng.next() as u8;
            }
        }
        let mut x = vec![0u8; (k / 128) * n * 136];
        for block in x.chunks_mut(136) {
            let mut sum = 0i32;
            for j in 0..128 {
                let q = rng.below(255) as i32 - 127;
                sum += q;
                block[8 + j] = q as i8 as u8;
            }
            let d = f32::from_bits(
                (((110 + rng.below(18)) as u32) << 23) | (rng.next() as u32 & 0x7f_ffff),
            );
            block[0..4].copy_from_slice(&d.to_le_bytes());
            block[4..8].copy_from_slice(&sum.to_le_bytes());
        }
        (a, x)
    }

    /// Run the emitted kernel on `M = 128, K = 512, N = 128` (one tile, one full trip plus the peeled
    /// group) and return `Y` (`[N][M]` f32 bytes).
    pub fn run(e: &Emitted, symbol: &str) -> Vec<u8> {
        run_tile(e, symbol, Tile::T128x128)
    }

    /// [`run`] on `tile`: one tile of `tile.rows() x tile.tokens()` outputs.
    pub fn run_tile(e: &Emitted, symbol: &str, tile: Tile) -> Vec<u8> {
        const KERNARG: u64 = 0x1_0000;
        const A_BASE: u64 = 0x100_0000;
        const X_BASE: u64 = 0x1000_0000;
        const Y_BASE: u64 = 0x2000_0000;
        let (m, k, n) = (tile.rows() as usize, 512usize, tile.tokens() as usize);
        let (a, x) = data(m, k, n, 7);
        let elf = peacemaker_isa::native::assemble(&e.s_text, Arch::Gfx1201).unwrap();
        let program = peacemaker_lift::lift_object(
            &elf,
            peacemaker_lift::Options {
                frontend: Frontend::Builder,
            },
        )
        .unwrap()
        .program;
        let emu = Emulator::new(&program, symbol).unwrap();
        let mut karg = vec![0u8; 296];
        karg[0..8].copy_from_slice(&A_BASE.to_le_bytes());
        karg[8..16].copy_from_slice(&X_BASE.to_le_bytes());
        karg[16..24].copy_from_slice(&Y_BASE.to_le_bytes());
        karg[24..28].copy_from_slice(&(m as u32).to_le_bytes());
        karg[28..32].copy_from_slice(&(k as u32).to_le_bytes());
        karg[32..36].copy_from_slice(&(n as u32).to_le_bytes());
        karg[40..44].copy_from_slice(&1u32.to_le_bytes());
        karg[44..48].copy_from_slice(&1u32.to_le_bytes());
        karg[48..52].copy_from_slice(&1u32.to_le_bytes());
        let mut mem = Memory::default();
        mem.map("karg".into(), KERNARG, karg).unwrap();
        mem.map("a".into(), A_BASE, a).unwrap();
        mem.map("x".into(), X_BASE, x).unwrap();
        mem.map("y".into(), Y_BASE, vec![0xa5; m * n * 4]).unwrap();
        emu.run_workgroup(
            &Launch {
                kernarg_ptr: KERNARG,
                workgroup: [0, 0, 0],
                block: [tile.threads(), 1, 1],
                dynamic_lds: tile.layout().launch,
            },
            &mut mem,
        )
        .unwrap();
        mem.regions
            .iter()
            .find(|r| r.name == "y")
            .unwrap()
            .bytes
            .clone()
    }
}

#[test]
fn moved_decode_split_barrier_and_fold_order_keep_the_output_bit_identical() {
    let linear = "128x128x8-set-linear";
    let build_linear = |src: &str| build(src, linear).unwrap();
    let want = {
        let e = build_linear(TWIN);
        interpreter::run(&e, &peacemaker_rip::symbol_of(&e).unwrap())
    };
    assert!(
        want.iter().any(|&b| b != 0xa5),
        "the baseline wrote no output"
    );
    let signal = "    let b1 = wg.signal((rotate(st.a, da), rotate(st.w, dw)))?;\n";
    let variations = [
        (
            "decode starts after the 12th WMMA",
            edit("phase(wg, g, &mut st, 0, true, 8, 3,", "phase(wg, g, &mut st, 0, true, 12, 3,"),
        ),
        (
            "two work units per WMMA",
            edit("phase(wg, g, &mut st, 1, false, 6, 3,", "phase(wg, g, &mut st, 1, false, 6, 2,"),
        ),
        (
            "B1 signalled before the meta fetch",
            TWIN.replacen(META_QUEUED, "", 1)
                .replacen(signal, &format!("{signal}{META_AFTER_SIGNAL}"), 1),
        ),
        (
            "row group 1 folded in reverse order",
            edit(
                "    for nb in 0..4usize {\n        frag(wg, g, nb, 1, cacc, scratch)?;\n    }\n",
                "    for nb in [3usize, 2, 1, 0] {\n        frag(wg, g, nb, 1, cacc, scratch)?;\n    }\n",
            ),
        ),
    ];
    for (what, src) in variations {
        let e = build_linear(&src);
        assert_ne!(e.s_text, build_linear(TWIN).s_text, "{what}: no change");
        let got = interpreter::run(&e, &peacemaker_rip::symbol_of(&e).unwrap());
        assert!(got == want, "{what}: the output changed");
    }
}

/// The interpreter is what catches a wrong move: the fold stages of one fragment reordered (stage 2
/// overwrites the product octet that stage 1 still has to read) build, but compute another result.
#[test]
fn a_wrong_fold_stage_order_builds_but_changes_the_output() {
    let linear = "128x128x8-set-linear";
    let run = |src: &str| {
        let e = build(src, linear).unwrap();
        interpreter::run(&e, &peacemaker_rip::symbol_of(&e).unwrap())
    };
    let wrong = edit(
        "    for stage in 0..4usize {\n        mq6::fold_stage(wg, g, nb * 2 + rg, stage, sc, zp)?;\n    }\n",
        "    for stage in [0usize, 2, 1, 3] {\n        mq6::fold_stage(wg, g, nb * 2 + rg, stage, sc, zp)?;\n    }\n",
    );
    assert!(run(&wrong) != run(TWIN), "the wrong order was not detected");
}

/// The decode scratch is also the WMMA seed octet: decoding before the last seeded WMMA has read it
/// builds (it is arithmetic, not a hazard the builder models) and computes another result, which
/// the interpreter sees.
#[test]
fn decoding_before_the_seed_is_consumed_builds_but_changes_the_output() {
    let linear = "128x128x8-set-linear";
    let run = |src: &str| {
        let e = build(src, linear).unwrap();
        interpreter::run(&e, &peacemaker_rip::symbol_of(&e).unwrap())
    };
    let early = edit(
        "phase(wg, g, &mut st, 0, true, 8, 3,",
        "phase(wg, g, &mut st, 0, true, 4, 3,",
    );
    assert!(
        run(&early) != run(TWIN),
        "the early decode was not detected"
    );
}

/// Split-barrier position (B2): signalled one fold fragment later, so the first packets of
/// fragment (2, 0) run before the rendezvous starts instead of overlapping it.
#[test]
fn signalling_b2_one_fragment_later_keeps_the_output() {
    let linear = "128x128x8-set-linear";
    let frag = "    frag(wg, g, 2, 0, fold_sc, fold_zp)?;\n";
    let start = TWIN.find("    let arrived = if b2 {").unwrap();
    let end = start + TWIN[start..].find("    };\n").unwrap() + "    };\n".len();
    let block = &TWIN[start..end];
    let tail = &TWIN[end..];
    assert!(
        tail.starts_with(frag),
        "the fold no longer follows the B2 block"
    );
    let moved = format!("{}{frag}{block}{}", &TWIN[..start], &tail[frag.len()..]);
    let run = |src: &str| {
        let e = build(src, linear).unwrap();
        (
            e.s_text.clone(),
            interpreter::run(&e, &peacemaker_rip::symbol_of(&e).unwrap()),
        )
    };
    let (base_text, base_out) = run(TWIN);
    let (text, out) = run(&moved);
    assert_ne!(base_text, text);
    assert!(out == base_out, "the output changed");
}

/// A row-plane load issued while the SZ ring is held by the split barrier in flight is refused.
#[test]
fn a_ring_read_between_signal_and_wait_is_refused() {
    let signal = "        let b2 = wg.signal((rotate(st.a, da), rotate(st.w, dw), rotate(st.d, dd), rotate(st.s, ds)))?;\n";
    assert!(TWIN.contains(signal));
    let src = TWIN.replacen(
        signal,
        &format!("{signal}        sz_load(wg, g, &st, h, scratch, 0)?;\n"),
        1,
    );
    let e = build(&src, SET).expect_err("a read of a held ring must be refused");
    assert!(e.contains("split barrier in flight"), "{e}");
}

// ---- the definedness check and wave32 lane masks ------------------------------------------------

fn mask_kernel(define_mask: bool) -> String {
    let define = if define_mask {
        "sop(wg.isa(), \"s_mov_b32 s12, 0xaaaaaaaa\", &[12], &[])?;\n    "
    } else {
        ""
    };
    format!(
        r#"fn emit(spec: Spec) -> Result<Emitted, String> {{
    let arch = spec.arch;
    let kspec = KernelSpec {{ kernel_id: "mask".into(), variant: "default".into(), arch, symbol: "mask_kernel".into(),
        kernargs: KernargLayout::new(16).pointer("a", 0).pointer("b", 8), user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: false, workgroup_size: 32, group_segment_fixed_size: 0, wave32: true, cu_mode: false }};
    let mut p = RegPlan::new(256, 104)?;
    p.s::<2>("kernarg_ptr", 0, Live::Whole)?;
    p.s::<4>("t", 12, Live::Whole)?;
    p.v::<8>("v", 0, Live::Whole)?;
    let mut builder = Builder::new(kspec, p);
    let mut wg = Workgroup::new(&mut builder, arch)?;
    author(&mut wg)?;
    builder.finish()
}}
region Doc;
fn author<T: Target>(wg: &mut Wg<T>) -> Result<(), String> {{
    let end = wg.exit(".Ldoc_end")?;
    {define}op(wg.isa(), "v_mov_b32_e32 v1, 0", &[v(1)], &[])?;
    op(wg.isa(), "v_mov_b32_e32 v2, 1", &[v(2)], &[])?;
    op(wg.isa(), "v_cndmask_b32_e64 v3, v1, v2, s12", &[v(3)], &[v(1), v(2), sr(12, 2)])?;
    wg.end(end)
}}
"#
    )
}

/// A wave32 mask is one SGPR: `s_mov_b32` defines the whole select. The table models the slot as a
/// pair, which the definedness check must not read as a demand for the high register; an SGPR that
/// nothing writes is still refused.
#[test]
fn a_wave32_lane_mask_is_defined_by_its_low_register() {
    peacemaker_rip::compile(&mask_kernel(true), Arch::Gfx1201, "default")
        .expect("s_mov_b32 defines a wave32 mask");
    let e = peacemaker_rip::compile(&mask_kernel(false), Arch::Gfx1201, "default")
        .expect_err("an undefined mask must be refused");
    assert!(e.contains("register read before it is defined: s12"), "{e}");
}

/// The other tiles: the same moves keep each tile's output bit-identical (a tile's script differs
/// from the shipped one only by `g.w_rounds()` in the decode queue and the prologue).
#[test]
fn moves_keep_the_output_on_the_wide_tiles() {
    for (tile, variant) in [
        (Tile::T256x128, "256x128x16-set-linear"),
        (Tile::T128x256, "128x256x16-set-linear"),
    ] {
        let run = |src: &str| {
            let e = build(src, variant).unwrap();
            interpreter::run_tile(&e, &peacemaker_rip::symbol_of(&e).unwrap(), tile)
        };
        let want = run(TWIN);
        assert!(want.iter().any(|&b| b != 0xa5), "{variant}: no output");
        for src in [
            edit("phase(wg, g, &mut st, 0, true, 8, 3,", "phase(wg, g, &mut st, 0, true, 12, 3,"),
            edit(
                "    for nb in 0..4usize {\n        frag(wg, g, nb, 1, cacc, scratch)?;\n    }\n",
                "    for nb in [3usize, 2, 1, 0] {\n        frag(wg, g, nb, 1, cacc, scratch)?;\n    }\n",
            ),
        ] {
            assert!(run(&src) == want, "{variant}: the output changed");
        }
    }
}
