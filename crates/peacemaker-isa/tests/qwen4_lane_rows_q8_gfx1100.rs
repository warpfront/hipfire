// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Portable gfx1100 Q8 row-tile ABI, addressing, and arithmetic tests.
//!
//! PM C1 below means **this same emitted entry launched at R=1**: a scheduling
//! oracle, NOT execution of the runtime singleton object. The independent raw
//! DAGs transcribe the acquired XTX objects: q8-scalar.co SHA256
//! 5462cc8b601c3d3f227046fefeeb2a4ac5c638df8516f9b7c341157e2637f762,
//! q8-hc.co c9343949ec0225b8206d82e0b69a80253f5de5e75c14074d72a16c70a9a8f7d5,
//! and gemv_q8_0_wide.gfx1100.co
//! 4de9f1a0a0c6f7ab9e7030ae8cfc920da48c2b4e70b9344c8267b855291bfe0e.
//! No local reference-object paths or fake singleton launches are used.
//! Activated HC is compared only to PM C1, never a host exp/div approximation.
//! These tests do not certify actual-singleton execution, GPU HC exp/div bytes,
//! exceptional-input semantics, or hardware correctness. Emulator refusals fail.

use peacemaker_ir::emu::{Emulator, Launch, Memory};
use peacemaker_ir::inst::Frontend;
use peacemaker_isa::kernels::qwen4_lane_rows_q8_gfx1100::build_rowtiles_gfx1100;
use peacemaker_isa::{Arch, Emitted};
use peacemaker_lift::{lift_object, Options};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Entry {
    Hc,
    K320,
    Plain,
    X2,
    Wide,
}
const ENTRIES: [Entry; 5] = [Entry::Hc, Entry::K320, Entry::Plain, Entry::X2, Entry::Wide];
const SYMBOLS: [&str; 5] = [
    "gemv_q8_0_k8_rows_t8_pm_gfx1100",
    "gemv_q8_0_k320_rows_t8_pm_gfx1100",
    "gemv_q8_0_k2560_rows_t8_pm_gfx1100",
    "gemv_q8_0_k2560_rows_x2_t8_pm_gfx1100",
    "gemv_q8_0_wide_rows_t8_pm_gfx1100",
];
impl Entry {
    fn index(self) -> usize {
        ENTRIES.iter().position(|&e| e == self).unwrap()
    }
    fn symbol(self) -> &'static str {
        SYMBOLS[self.index()]
    }
    fn four(self) -> bool {
        matches!(self, Self::K320 | Self::Wide)
    }
    fn matrices(self) -> usize {
        if self == Self::X2 {
            2
        } else {
            1
        }
    }
    fn block(self) -> u32 {
        if self == Self::Hc {
            512
        } else {
            256
        }
    }
    fn columns(self) -> usize {
        if self == Self::Hc {
            2
        } else {
            8
        }
    }
    fn arg_bytes(self) -> usize {
        match self {
            Self::Hc => 44,
            Self::X2 => 56,
            _ => 36,
        }
    }
    fn good_k(self) -> usize {
        match self {
            Self::Hc => 768,
            Self::K320 => 320,
            Self::Wide => 640,
            _ => 2560,
        }
    }
    fn grid_x(self, ms: [usize; 2]) -> usize {
        if self == Self::X2 {
            ms[0].div_ceil(8) + ms[1].div_ceil(8)
        } else {
            ms[0].div_ceil(self.columns())
        }
    }
}

fn metadata_numbers(text: &str, key: &str) -> Vec<u32> {
    text.lines()
        .filter_map(|line| line.trim().strip_prefix(key))
        .map(|value| value.trim().parse().unwrap())
        .collect()
}

#[test]
fn five_exports_freeze_arch_abi_and_resources() {
    let exports = build_rowtiles_gfx1100().expect("XTX Q8 rowtiles");
    assert_eq!(exports.len(), 5);
    for (e, entry) in exports.iter().zip(ENTRIES) {
        let t = &e.s_text;
        assert_eq!(e.proof.arch, Arch::Gfx1100);
        assert!(t.contains(&format!(".amdhsa_kernel {}\n", entry.symbol())));
        // Core reserves `.L{symbol}_end` for the mandatory end-of-text footer; the
        // private body prefix lives under `.L{symbol}_body` so no emitted
        // instruction label collides with that reserved footer label.
        let body = format!(".L{}_body", entry.symbol());
        let footer = format!(".L{}_end", entry.symbol());
        assert_ne!(body, footer, "{}", entry.symbol());
        assert_eq!(
            t.matches(&format!("{footer}:\n")).count(),
            1,
            "{}: {footer}",
            entry.symbol()
        );
        assert_eq!(
            t.matches(&format!("{body}_end:\n")).count(),
            1,
            "{}: {body}_end",
            entry.symbol()
        );
        assert!(!t.contains("gfx1151"), "no image retagging");
        let args: Vec<(u32, u32)> = if entry == Entry::X2 {
            vec![
                (0, 8),
                (8, 8),
                (16, 8),
                (24, 8),
                (32, 8),
                (40, 4),
                (44, 4),
                (48, 4),
                (52, 4),
            ]
        } else {
            let mut a = vec![(0, 8), (8, 8), (16, 8), (24, 4), (28, 4), (32, 4)];
            if entry == Entry::Hc {
                a.extend([(36, 4), (40, 4)]);
            }
            a
        };
        assert_eq!(
            metadata_numbers(t, ".offset:"),
            args.iter().map(|a| a.0).collect::<Vec<_>>()
        );
        assert_eq!(
            metadata_numbers(t, ".size:"),
            args.iter().map(|a| a.1).collect::<Vec<_>>()
        );
        for line in [
            format!(".kernarg_segment_size: {}\n", entry.arg_bytes()),
            format!(".amdhsa_kernarg_size {}\n", entry.arg_bytes()),
            format!(".max_flat_workgroup_size: {}\n", entry.block()),
            format!(
                ".group_segment_fixed_size: {}\n",
                if entry == Entry::Hc { 16384 } else { 8192 }
            ),
            format!(
                ".amdhsa_group_segment_fixed_size {}\n",
                if entry == Entry::Hc { 16384 } else { 8192 }
            ),
        ] {
            assert!(t.contains(&line), "{}: {line}", entry.symbol());
        }
        for line in [
            ".amdhsa_system_sgpr_workgroup_id_x 1\n",
            ".amdhsa_system_sgpr_workgroup_id_y 1\n",
            ".wavefront_size: 32\n",
            ".amdhsa_wavefront_size32 1\n",
            ".private_segment_fixed_size: 0\n",
            ".vgpr_spill_count: 0\n",
            ".sgpr_spill_count: 0\n",
        ] {
            assert!(t.contains(line), "{}: {line}", entry.symbol());
        }
        assert_eq!(
            t.matches(".actual_access: read_only\n").count(),
            entry.matrices() + 1
        );
        assert_eq!(
            t.matches(".actual_access: write_only\n").count(),
            entry.matrices()
        );
        assert_eq!(e.shape.next_free_vgpr, if entry.four() { 72 } else { 48 });
        assert!(e.shape.next_free_sgpr <= 56);
        assert_eq!(e.proof.next_free_vgpr, e.shape.next_free_vgpr);
        assert_eq!(e.proof.next_free_sgpr, e.shape.next_free_sgpr);
        assert!(!t.contains("scratch_"));
        assert!(e.shape.barriers > 0);
        assert!(!e.proof.loop_fixpoints.is_empty());
        assert_eq!(
            e.proof.lds_slots.len(),
            if entry == Entry::Hc { 2 } else { 1 }
        );
        for slot in &e.proof.lds_slots {
            assert!(slot.base + slot.len <= if entry == Entry::Hc { 16384 } else { 8192 });
        }
        if entry == Entry::Hc {
            assert_eq!(
                e.proof.lds_slots[1].len, 512,
                "bounded per-tile partial relayout"
            );
            for op in ["v_exp_f32", "v_div_fmas_f32", "v_div_fixup_f32"] {
                assert!(t.contains(op));
            }
        }
        for op in [
            "v_fma_mix_f32",
            "neg(0) op_sel_hi:[1,0,0]",
            "ds_swizzle_b32",
            "ds_bpermute_b32",
        ] {
            assert!(t.contains(op), "{}: {op}", entry.symbol());
        }
    }
}

/// gfx1100 `FeatureUserSGPRInit16Bug` (LLVM): >= 16 user + system SGPRs, so the two kernarg SGPRs are padded
/// to 14 user SGPRs and the workgroup ids x, y are s14, s15 (gfx1151: s2, s3). The wave index and the HC flag
/// that used to sit in s15/s14 live above the old plan (s49/s50).
#[test]
fn workgroup_ids_follow_the_padded_user_sgprs() {
    for (e, entry) in build_rowtiles_gfx1100().unwrap().iter().zip(ENTRIES) {
        let t = &e.s_text;
        let user = metadata_numbers(t, ".amdhsa_user_sgpr_count");
        assert_eq!(user, [14], "{}", entry.symbol());
        assert!(
            u32::from(Arch::Gfx1100.min_user_system_sgprs()) <= user[0] + 2,
            "{}",
            entry.symbol()
        );
        for line in ["s_lshl_b32 s15, s15, 3\n", "v_readfirstlane_b32 s49, v13\n"] {
            assert!(t.contains(line), "{}: {line}", entry.symbol());
        }
        assert!(!t.contains("s_lshl_b32 s3, s3, 3"), "{}", entry.symbol());
        // `native::assemble` refuses a gfx1100 descriptor with user + system < 16.
        peacemaker_isa::native::assemble(t, Arch::Gfx1100).expect("assemble");
    }
}

const KA: u64 = 0x10000;
// Every operand crosses a low-word boundary during live address arithmetic.
const A: [u64; 2] = [0xffff_ffc0, 0x2_ffff_ffc0];
const X: u64 = 0x4_ffff_ff00;
const Y: [u64; 2] = [0x6_ffff_fff0, 0x8_ffff_fff0];
const GUARD: usize = 16;
const POISON: u8 = 0xa5;

fn args(entry: Entry, k: u32, ms: [u32; 2], rows: u32, scale: f32, activated: bool) -> Vec<u8> {
    let mut out = Vec::new();
    let ptrs = if entry == Entry::X2 {
        vec![A[0], A[1], X, Y[0], Y[1]]
    } else {
        vec![A[0], X, Y[0]]
    };
    for p in ptrs {
        out.extend(p.to_le_bytes());
    }
    let dims = if entry == Entry::X2 {
        vec![ms[0], ms[1], k, rows]
    } else {
        vec![ms[0], k, rows]
    };
    for d in dims {
        out.extend(d.to_le_bytes());
    }
    if entry == Entry::Hc {
        out.extend(scale.to_le_bytes());
        out.extend(u32::from(activated).to_le_bytes());
    }
    assert_eq!(out.len(), entry.arg_bytes());
    out
}

fn weights(m: usize, k: usize, seed: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(m * (k / 32) * 34);
    for b in 0..m * (k / 32) {
        // Finite normal f16, nontrivial mantissa, varying signs.
        let d = (0x2800 | ((b * 73 + seed * 19) & 1023) | ((b & 1) << 15)) as u16;
        out.extend(d.to_le_bytes());
        for lane in 0..32 {
            out.push((b * 37 + lane * 29 + seed * 11) as u8);
        }
    }
    out
}

fn inputs(k: usize) -> Vec<f32> {
    (0..64 * k)
        .map(|i| {
            let row = i / k;
            // All 64 rows have distinct data; signed zeros, cancellation, broad
            // finite exponents and full mantissas exercise FMA vs mul+add rounding.
            if i % 97 == 0 {
                return f32::from_bits(((row & 1) as u32) << 31);
            }
            let bits = (((i ^ row) & 1) as u32) << 31
                | ((112 + (i * 13 + row * 7) % 18) as u32) << 23
                | ((i as u32)
                    .wrapping_mul(0x174321)
                    .wrapping_add(row as u32 * 319)
                    & 0x7fffff);
            f32::from_bits(bits)
        })
        .collect()
}

/// Run the real lifted entry, optionally one isolated workgroup. X has exactly
/// its live extent unless poisoned_tail requests mapped signalling-NaN rows.
/// All output bytes are checked against the precise owner set, including guards.
#[allow(clippy::too_many_arguments)]
fn run(
    emu: &Emulator<'_>,
    entry: Entry,
    k: usize,
    ms: [usize; 2],
    a: &[Vec<u8>; 2],
    x: &[f32],
    scale: f32,
    activated: bool,
    poisoned_tail: bool,
    only: Option<[usize; 2]>,
) -> [Vec<u32>; 2] {
    let rows = x.len() / k;
    assert_eq!(rows * k, x.len());
    assert!((1..=64).contains(&rows));
    let ka = args(
        entry,
        k as u32,
        ms.map(|m| m as u32),
        rows as u32,
        scale,
        activated,
    );
    let mut mem = Memory::default();
    mem.map("kernarg".into(), KA, ka.clone()).unwrap();
    let mut xb: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    if poisoned_tail {
        for _ in x.len()..rows.div_ceil(8) * 8 * k {
            xb.extend(0x7f80_0001u32.to_le_bytes());
        }
    }
    mem.map("x".into(), X, xb.clone()).unwrap();
    for i in 0..entry.matrices() {
        if !a[i].is_empty() {
            mem.map(format!("a{i}"), A[i], a[i].clone()).unwrap();
        }
    }
    for i in 0..2 {
        mem.map(
            format!("y{i}"),
            Y[i] - GUARD as u64,
            vec![POISON; 2 * GUARD + rows.div_ceil(8) * 8 * ms[i] * 4],
        )
        .unwrap();
    }
    let gx_count = entry.grid_x(ms);
    let groups: Vec<[usize; 2]> = if let Some(group) = only {
        vec![group]
    } else {
        (0..=rows.div_ceil(8))
            .flat_map(|gy| (0..=gx_count).map(move |gx| [gx, gy]))
            .collect()
    };
    for [gx, gy] in groups {
        emu.run_workgroup(
            &Launch {
                kernarg_ptr: KA,
                workgroup: [gx as u32, gy as u32, 0],
                block: [entry.block(), 1, 1],
                dynamic_lds: 0,
            },
            &mut mem,
        )
        .unwrap_or_else(|e| {
            panic!("{entry:?} K{k} M{ms:?} R{rows} act{activated} wg({gx},{gy}): {e}")
        });
    }
    for region in &mem.regions {
        let before = match region.name.as_str() {
            "kernarg" => Some(&ka),
            "x" => Some(&xb),
            "a0" => Some(&a[0]),
            "a1" => Some(&a[1]),
            _ => None,
        };
        if let Some(before) = before {
            assert_eq!(&region.bytes, before, "{} immutable", region.name);
            assert_eq!(
                region.written.count(),
                0,
                "{} must never be written, even with identical bytes",
                region.name
            );
        }
    }
    std::array::from_fn(|i| {
        let region = mem
            .regions
            .iter()
            .find(|r| r.name == format!("y{i}"))
            .unwrap();
        let owns = |row: usize, m: usize| {
            if i >= entry.matrices() || row >= rows {
                return false;
            }
            match only {
                None => true,
                Some([gx, gy]) => {
                    let base = if entry == Entry::X2 && i == 1 {
                        ms[0].div_ceil(8)
                    } else {
                        0
                    };
                    row / 8 == gy && gx == base + m / entry.columns()
                }
            }
        };
        for byte in 0..region.bytes.len() {
            let live = byte >= GUARD && byte < GUARD + rows * ms[i] * 4;
            let element = byte.saturating_sub(GUARD) / 4;
            let owned = live && ms[i] != 0 && owns(element / ms[i], element % ms[i]);
            assert_eq!(
                region.written.contains(byte),
                owned,
                "{entry:?} K{k} R{rows} M{ms:?} matrix{i} byte{byte} group{only:?}"
            );
            if !owned {
                assert_eq!(region.bytes[byte], POISON, "guard/inactive byte changed");
            }
        }
        region.bytes[GUARD..GUARD + rows * ms[i] * 4]
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    })
}

/// The activated HC twin evaluates `v_rndne_f32` and `v_exp_f32`, whose gfx1100 numerics the interpreter
/// refuses (`peacemaker-ir` `emu/alu.rs`: "unqualified numerical architecture" for the transcendental, and
/// the negative-to-zero `v_rndne_f32` case measured only on gfx1151/gfx1201): there is no CPU byte claim for
/// the activated path, so one of those exact refusals is asserted instead (same contract as the BF16 gfx1100 twin).
fn assert_activation_refused(
    emu: &Emulator<'_>,
    entry: Entry,
    k: usize,
    ms: [usize; 2],
    a: &[Vec<u8>; 2],
    x: &[f32],
    scale: f32,
) {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run(emu, entry, k, ms, a, x, scale, true, false, None)
    }));
    let payload = result
        .expect_err("activated gfx1100 HC became executable: require an actual-object byte oracle");
    let message = payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_owned()))
        .unwrap_or_default();
    assert!(
        message.contains("unqualified numerical architecture Gfx1100")
            || message.contains("only gfx1151 and gfx1201 are measured"),
        "activated HC refused for a different reason: {message}"
    );
}

/// Actual XTX generic-small DAG: ascending blocks; quad FMA(dq,x,acc),
/// remainder RN(x*dq) then addition to b&3; fold (a3+a2)+(a1+a0).
/// Plain/HC use one FMA(x,dq,acc) chain, including legal block tails.
fn raw_dot(a: &[u8], x: &[f32], four: bool) -> f32 {
    let nb = x.len() / 32;
    assert_eq!(nb * 32, x.len());
    let mut lanes = [0.0f32; 32];
    for lane in 0..32 {
        let mut acc = [0.0f32; 4];
        for block in 0..nb {
            let d = u16::from_le_bytes(a[34 * block..34 * block + 2].try_into().unwrap());
            // Generated scales are finite normal binary16; convert exactly.
            let scale = f32::from_bits(
                (u32::from(d & 0x8000) << 16)
                    | (((u32::from(d & 0x7c00) >> 10) + 112) << 23)
                    | (u32::from(d & 1023) << 13),
            );
            let q = f32::from(a[34 * block + 2 + lane] as i8);
            let dq = scale.mul_add(q, -0.0);
            let value = x[32 * block + lane];
            let chain = if four { block & 3 } else { 0 };
            acc[chain] = if four && block >= nb / 4 * 4 {
                let product = value * dq;
                acc[chain] + product
            } else if four {
                dq.mul_add(value, acc[chain])
            } else {
                value.mul_add(dq, acc[chain])
            };
        }
        lanes[lane] = if four {
            (acc[3] + acc[2]) + (acc[1] + acc[0])
        } else {
            acc[0]
        };
    }
    // Native swizzle first swaps half-waves. Subsequent bpermutes use
    // lane+delta if in bounds, otherwise own lane; add own + other.
    let prev = lanes;
    for lane in 0..32 {
        lanes[lane] = prev[lane] + prev[lane ^ 16];
    }
    for delta in [8, 4, 2, 1] {
        let prev = lanes;
        for lane in 0..32 {
            let other = if lane + delta < 32 {
                lane + delta
            } else {
                lane
            };
            lanes[lane] = prev[lane] + prev[other];
        }
    }
    lanes[0]
}

fn raw_reference(entry: Entry, a: &[u8], x: &[f32]) -> u32 {
    if entry != Entry::Hc {
        return raw_dot(a, x, entry.four()).to_bits();
    }
    let k = x.len();
    let p: [f32; 8] = std::array::from_fn(|part| {
        raw_dot(
            &a[part * (k / 256) * 34..],
            &x[part * (k / 8)..(part + 1) * (k / 8)],
            false,
        )
    });
    (((p[0] + p[1]) + (p[2] + p[3])) + ((p[4] + p[5]) + (p[6] + p[7]))).to_bits()
}

fn lifted(e: &Emitted) -> peacemaker_ir::state::Lifted<peacemaker_ir::Program> {
    let elf = peacemaker_isa::native::assemble(&e.s_text, Arch::Gfx1100)
        .expect("native gfx1100 assemble");
    lift_object(
        &elf,
        Options {
            frontend: Frontend::Builder,
        },
    )
    .expect("lift native gfx1100 object")
}

#[test]
fn every_r_1_through_64_matches_pm_c1_scheduling_and_independent_raw_dags() {
    let exports = build_rowtiles_gfx1100().unwrap();
    for (e, entry) in exports.iter().zip(ENTRIES) {
        let lifted = lifted(e);
        let emu =
            Emulator::new(&lifted.program, entry.symbol()).expect("real emulator, no softpass");
        let ks: Vec<usize> = match entry {
            Entry::Hc => vec![256, 768, 2048, 2304],
            Entry::K320 => vec![320],
            Entry::Plain => vec![1568, 2048, 2560, 2880, 3072],
            Entry::X2 => vec![2560],
            Entry::Wide => (1..=16).map(|b| b * 32).chain([640, 1504, 1536]).collect(),
        };
        let shapes: Vec<[usize; 2]> = match entry {
            Entry::Hc => vec![[1, 0], [3, 0]],
            Entry::X2 => vec![[0, 0], [0, 9], [9, 0], [1, 9], [9, 1], [3, 7]],
            Entry::K320 => vec![[1, 0], [2, 0], [3, 0], [7, 0], [8, 0], [9, 0]],
            _ => vec![[3, 0], [9, 0]],
        };
        // Ks are independent: run them concurrently; any panic fails the scope.
        std::thread::scope(|s| {
            for k in ks {
                let (emu, shapes) = (&emu, &shapes);
                s.spawn(move || {
            let x = inputs(k);
            for ms in shapes.iter().copied() {
                let a = std::array::from_fn(|i| weights(ms[i], k, i + 1));
                let modes: &[(bool, f32)] = if entry == Entry::Hc {
                    &[(false, 0.375), (true, 0.0), (true, 0.0078125), (true, -2.0)]
                } else {
                    &[(false, 1.0)]
                };
                for &(activated, scale) in modes {
                    if activated {
                        assert_activation_refused(&emu, entry, k, ms, &a, &x[..k], scale);
                        continue;
                    }
                    let mut c1: [Vec<u32>; 2] = [Vec::new(), Vec::new()];
                    for row in 0..64 {
                        let xr = &x[row * k..(row + 1) * k];
                        let one = run(&emu, entry, k, ms, &a, xr, scale, activated, false, None);
                        for i in 0..entry.matrices() {
                            if !activated {
                                for m in 0..ms[i] {
                                    let want = raw_reference(entry, &a[i][m * (k / 32) * 34..], xr);
                                    assert_eq!(
                                        one[i][m], want,
                                        "raw DAG {entry:?} K{k} matrix{i} row{row} m{m}"
                                    );
                                }
                            }
                            c1[i].extend_from_slice(&one[i]);
                        }
                    }
                    for rows in 1..=64 {
                        let got = run(
                            &emu,
                            entry,
                            k,
                            ms,
                            &a,
                            &x[..rows * k],
                            scale,
                            activated,
                            false,
                            None,
                        );
                        for i in 0..2 {
                            assert_eq!(got[i], c1[i][..got[i].len()],
                                "PM C1 scheduling only: {entry:?} K{k} M{ms:?} R{rows} matrix{i} act{activated} scale{scale}");
                        }
                    }
                }
            }
            });
            }
        });
    }
}

#[test]
fn poisoned_inactive_rows_and_isolated_grid_owners_preserve_exact_live_mapping() {
    for (e, entry) in build_rowtiles_gfx1100().unwrap().iter().zip(ENTRIES) {
        let lifted = lifted(e);
        let emu = Emulator::new(&lifted.program, entry.symbol()).unwrap();
        let k = entry.good_k();
        let ms = if entry == Entry::X2 { [1, 9] } else { [9, 0] };
        let a = std::array::from_fn(|i| weights(ms[i], k, 3 + i));
        let x = inputs(k);
        for activated in [false, true]
            .into_iter()
            .take(if entry == Entry::Hc { 2 } else { 1 })
        {
            if activated {
                assert_activation_refused(&emu, entry, k, ms, &a, &x[..k], 0.125);
                continue;
            }
            for rows in [1usize, 7, 8, 9, 15, 17, 33, 63, 64] {
                let xr = &x[..rows * k];
                let exact = run(&emu, entry, k, ms, &a, xr, 0.125, activated, false, None);
                let poisoned = run(&emu, entry, k, ms, &a, xr, 0.125, activated, true, None);
                assert_eq!(
                    poisoned, exact,
                    "poison affected live outputs {entry:?} R{rows}"
                );
                if rows != 17 {
                    continue;
                }
                for gy in 0..=rows.div_ceil(8) {
                    for gx in 0..=entry.grid_x(ms) {
                        let isolated = run(
                            &emu,
                            entry,
                            k,
                            ms,
                            &a,
                            xr,
                            0.125,
                            activated,
                            false,
                            Some([gx, gy]),
                        );
                        for i in 0..entry.matrices() {
                            let base = if i == 1 { ms[0].div_ceil(8) } else { 0 };
                            for row in 0..rows {
                                for m in 0..ms[i] {
                                    if row / 8 == gy && gx == base + m / entry.columns() {
                                        assert_eq!(
                                            isolated[i][row * ms[i] + m],
                                            exact[i][row * ms[i] + m]
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn invalid_domains_and_dead_groups_exit_uniformly_before_unmapped_operands() {
    for (e, entry) in build_rowtiles_gfx1100().unwrap().iter().zip(ENTRIES) {
        let lifted = lifted(e);
        let emu = Emulator::new(&lifted.program, entry.symbol()).unwrap();
        let good_k = entry.good_k() as u32;
        let bad_ks: &[u32] = match entry {
            Entry::Hc => &[0, 1, 32, 255, 257, 640],
            Entry::K320 => &[0, 1, 256, 640],
            Entry::Plain => &[0, 1, 320, 1536, 1569],
            Entry::X2 => &[0, 1, 2048, 2561, 2880],
            Entry::Wide => &[0, 1, 31, 33, 1568, 2560],
        };
        let mut cases = vec![
            (0, good_k, [3, 5], [0, 0]),
            (65, good_k, [3, 5], [0, 0]),
            (u32::MAX, good_k, [3, 5], [0, 0]),
            (1, good_k, [3, 5], [0, 1]),
            (9, good_k, [3, 5], [0, 2]),
            (1, good_k, [0, 0], [0, 0]),
        ];
        for &k in bad_ks {
            cases.push((1, k, [3, 5], [0, 0]));
        }
        // Whole dead x workgroup as well as dead y, with nonzero live shapes.
        cases.push((1, good_k, [3, 5], [entry.grid_x([3, 5]) as u32, 0]));
        for (rows, k, ms, wg) in cases {
            let ka = args(entry, k, ms, rows, 0.375, true);
            let mut mem = Memory::default();
            // A/X/Y pointers deliberately unmapped: any operand read, write,
            // or divergent barrier path is a hard failure, never a skip.
            mem.map("kernarg".into(), KA, ka.clone()).unwrap();
            emu.run_workgroup(
                &Launch {
                    kernarg_ptr: KA,
                    workgroup: [wg[0], wg[1], 0],
                    block: [entry.block(), 1, 1],
                    dynamic_lds: 0,
                },
                &mut mem,
            )
            .unwrap_or_else(|e| {
                panic!("{entry:?} R{rows} K{k} M{ms:?} wg{wg:?} must return before operands: {e}")
            });
            assert_eq!(mem.regions.len(), 1);
            assert_eq!(mem.regions[0].bytes, ka);
            assert_eq!(mem.regions[0].written.count(), 0);
        }
    }
}
