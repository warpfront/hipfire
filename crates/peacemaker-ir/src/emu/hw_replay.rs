// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Raw-hardware replay of `isa/emu/numerics-probe` receipts through the emulator's own `execute` paths (test data only).
//!
//! The probe (`numerics_probe.hip`) loads each dense WMMA wave from three logical matrices (`WCase`: `A[r*16+k]`,
//! `Bt[c*16+k]`, `C[r*16+c]`) into fragment registers with its own index formulas and stores the raw D registers
//! (`[vgpr][lane]`). This module rebuilds exactly those registers with the probe's formulas (NOT the emulator's
//! `f16_loc`/`acc_loc`, so a fragment-mapping error cannot hide), executes `v_wmma_*`/`v_dot*` through
//! `wmma::execute`/`dot::execute`, and compares every raw D word bit-for-bit.
//!
//! * `probe_receipt_replay` (opt-in, `PM_NUMERICS_RECEIPT_DIR=<receipt> cargo test --release -p peacemaker-ir --lib
//!   hw_replay::probe_receipt_replay -- --ignored --nocapture`): the whole receipt (identity.json names the arch).
//! * `halo_recorded_waves_match_hardware`: the committed subset `tests/fixtures/halo_numerics_waves.bin`, recorded
//!   from the Halo (gfx1151, BDF 0000:c0:00.0) receipt `pmrevive/PmBf16Halo/rcpt` by
//!   `numerics-probe/make_halo_fixtures.py`. Record = 16-byte header (u32 tag 1 = WMMA / 2 = dot, u32 op, u8 neg_lo,
//!   neg_hi, op_sel, op_sel_hi, u8 clamp + 3 pad) then, for WMMA, a `WCase` (2048 B) and the raw D image (8 x 32 u32);
//!   for a dot, `a, b, c, hardware d` (4 x u32).
use super::{dot, wmma, State};
use crate::{
    operand::{Modifiers, Operand},
    reg::{Kind, RegRef},
    Arch, FormFields, Inst,
};

const D: u16 = 0;
const A: u16 = 16;
const B: u16 = 32;
const C: u16 = 48;
const WCASE: usize = 2048;

/// WMMA ops of the probe: kind index (0 bf16->f32, 1 f16->f32, 2 iu8->i32).
const WMMA_OPS: [&str; 3] = [
    "v_wmma_f32_16x16x16_bf16",
    "v_wmma_f32_16x16x16_f16",
    "v_wmma_i32_16x16x16_iu8",
];
const DOT_OPS: [&str; 4] = [
    "v_dot2_f32_f16",
    "v_dot2_f32_bf16",
    "v_dot4_i32_iu8",
    "v_dot8_i32_iu4",
];

fn state(lanes: usize) -> State {
    State {
        s: [0; 106],
        t: [0; 16],
        v: Box::new([[0; 64]; 256]),
        lanes,
        nan_loose: false,
        exec: if lanes >= 32 {
            0xffff_ffff
        } else {
            (1u64 << lanes) - 1
        },
        vcc: 0,
        scc: false,
        m0: 0,
        pc: 0,
        ended: false,
        barrier: None,
        signals: 0,
        scratch: Vec::new(),
    }
}

fn reg(base: u16, len: u8) -> Operand {
    Operand::Reg(RegRef {
        kind: Kind::V,
        base,
        len,
    })
}

fn inst(arch: Arch, name: &str, ops: Vec<Operand>, mods: Modifiers) -> Inst {
    let table = if arch == Arch::Gfx1201 {
        crate::isa::gfx12()
    } else {
        crate::isa::gfx1151()
    };
    let row = table.iter().find(|r| r.name == name).unwrap();
    Inst::from_parts(
        arch,
        row.op,
        row.form,
        FormFields::default(),
        ops.into_iter().collect(),
        mods,
        None,
        Default::default(),
    )
    .unwrap()
}

/// Modifier tuple recorded with every case: (neg_lo, neg_hi, op_sel, op_sel_hi, clamp).
type Mods = (u8, u8, u8, u8, bool);

fn modifiers((neg_lo, neg_hi, op_sel, op_sel_hi, clamp): Mods) -> Modifiers {
    Modifiers {
        neg_lo,
        neg_hi,
        op_sel,
        op_sel_hi,
        clamp,
        ..Default::default()
    }
}

/// Parse the probe's modifier text (`neg_lo:[1,0,0] clamp`, `op_sel_hi:[0,1,1]`, ...). The assembler default of
/// `op_sel_hi` is all ones.
fn parse_mods(text: &str) -> Mods {
    let mut m: Mods = (0, 0, 0, 7, false);
    for tok in text.split_whitespace() {
        if tok == "clamp" {
            m.4 = true;
            continue;
        }
        let (key, rest) = tok
            .split_once(':')
            .unwrap_or_else(|| panic!("modifier {tok}"));
        let bits = rest
            .trim_matches(|c| c == '[' || c == ']')
            .split(',')
            .enumerate()
            .fold(0u8, |acc, (i, v)| acc | (u8::from(v == "1") << i));
        match key {
            "neg_lo" => m.0 = bits,
            "neg_hi" => m.1 = bits,
            "op_sel" => m.2 = bits,
            "op_sel_hi" => m.3 = bits,
            _ => panic!("modifier {tok}"),
        }
    }
    m
}

/// One dense WMMA wave: the probe's own register load, `execute`, the raw D image (`[vgpr][lane]`).
fn wmma_wave(arch: Arch, kind: usize, mods: Mods, case: &[u8]) -> [[u32; 32]; 8] {
    let a16 = |i: usize| u32::from(u16::from_le_bytes([case[2 * i], case[2 * i + 1]]));
    let bt16 = |i: usize| a16(256 + i);
    let c32 = |i: usize| u32::from_le_bytes(case[1024 + 4 * i..1028 + 4 * i].try_into().unwrap());
    let gfx12 = arch == Arch::Gfx1201;
    let mut st = state(32);
    for l in 0..32usize {
        let (r, h) = (l & 15, l >> 4);
        if kind == 2 {
            let nw = if gfx12 { 2 } else { 4 };
            for i in 0..nw {
                let k0 = if gfx12 { 8 * h + 4 * i } else { 4 * i };
                let pack = |f: &dyn Fn(usize) -> u32| {
                    (0..4).fold(0u32, |w, n| w | ((f(r * 16 + k0 + n) & 255) << (8 * n)))
                };
                st.v[usize::from(A) + i][l] = pack(&a16);
                st.v[usize::from(B) + i][l] = pack(&bt16);
            }
        } else {
            let nw = if gfx12 { 4 } else { 8 };
            for i in 0..nw {
                let k0 = if gfx12 { 8 * h + 2 * i } else { 2 * i };
                st.v[usize::from(A) + i][l] = a16(r * 16 + k0) | (a16(r * 16 + k0 + 1) << 16);
                st.v[usize::from(B) + i][l] = bt16(r * 16 + k0) | (bt16(r * 16 + k0 + 1) << 16);
            }
        }
        for j in 0..8 {
            let row = if gfx12 { 8 * h + j } else { 2 * j + h };
            st.v[usize::from(C) + j][l] = c32(row * 16 + r);
        }
    }
    let regs = match (gfx12, kind) {
        (true, 2) => 2,
        (true, _) => 4,
        (false, 2) => 4,
        (false, _) => 8,
    };
    let i = inst(
        arch,
        WMMA_OPS[kind],
        vec![reg(D, 8), reg(A, regs), reg(B, regs), reg(C, 8)],
        modifiers(mods),
    );
    wmma::execute(arch, WMMA_OPS[kind], &i, &mut st)
        .unwrap_or_else(|e| panic!("{}: {e}", WMMA_OPS[kind]));
    let mut out = [[0u32; 32]; 8];
    for (j, row) in out.iter_mut().enumerate() {
        row.copy_from_slice(&st.v[usize::from(D) + j][..32]);
    }
    out
}

/// Up to 32 dot cases at once (one per lane). `cases` = (a, b, c); returns the destination of each.
fn dot_lanes(arch: Arch, op: usize, mods: Mods, cases: &[(u32, u32, u32)]) -> Vec<u32> {
    let mut st = state(32);
    for (l, &(a, b, c)) in cases.iter().enumerate() {
        st.v[1][l] = a;
        st.v[2][l] = b;
        st.v[3][l] = c;
    }
    st.exec = if cases.len() >= 32 {
        0xffff_ffff
    } else {
        (1u64 << cases.len()) - 1
    };
    let i = inst(
        arch,
        DOT_OPS[op],
        vec![reg(0, 1), reg(1, 1), reg(2, 1), reg(3, 1)],
        modifiers(mods),
    );
    dot::execute(arch, DOT_OPS[op], &i, &mut st).unwrap_or_else(|e| panic!("{}: {e}", DOT_OPS[op]));
    (0..cases.len()).map(|l| st.v[0][l]).collect()
}

fn word(buf: &[u8], i: usize) -> u32 {
    u32::from_le_bytes(buf[4 * i..4 * i + 4].try_into().unwrap())
}

#[test]
fn halo_recorded_waves_match_hardware() {
    const FIXTURE: &[u8] = include_bytes!("../../tests/fixtures/halo_numerics_waves.bin");
    let (mut at, mut wmma_n, mut dot_n) = (0usize, 0usize, 0usize);
    while at < FIXTURE.len() {
        let (tag, op) = (word(&FIXTURE[at..], 0), word(&FIXTURE[at..], 1) as usize);
        let m: Mods = (
            FIXTURE[at + 8],
            FIXTURE[at + 9],
            FIXTURE[at + 10],
            FIXTURE[at + 11],
            FIXTURE[at + 12] != 0,
        );
        at += 16;
        match tag {
            1 => {
                let case = &FIXTURE[at..at + WCASE];
                let hw = &FIXTURE[at + WCASE..at + WCASE + 1024];
                at += WCASE + 1024;
                let got = wmma_wave(Arch::Gfx1151, op, m, case);
                for (j, row) in got.iter().enumerate() {
                    for (l, &g) in row.iter().enumerate() {
                        assert_eq!(
                            g,
                            word(hw, j * 32 + l),
                            "{} {m:?} record {wmma_n} vgpr {j} lane {l}",
                            WMMA_OPS[op]
                        );
                    }
                }
                wmma_n += 1;
            }
            2 => {
                let (a, b, c, d) = (
                    word(&FIXTURE[at..], 0),
                    word(&FIXTURE[at..], 1),
                    word(&FIXTURE[at..], 2),
                    word(&FIXTURE[at..], 3),
                );
                at += 16;
                assert_eq!(
                    dot_lanes(Arch::Gfx1151, op, m, &[(a, b, c)]),
                    [d],
                    "{} {m:?} a={a:#x} b={b:#x} c={c:#x}",
                    DOT_OPS[op]
                );
                dot_n += 1;
            }
            t => panic!("fixture tag {t}"),
        }
    }
    assert!(
        wmma_n >= 40 && dot_n >= 400,
        "fixture shrank: {wmma_n} waves, {dot_n} dots"
    );
}

/// Whole-receipt replay (opt-in). Every `wmma.<op>.v<V>.m<M>.out.bin` of the three executing kinds and every
/// `dot.<op>.<set>.v<V>.m<M>.out.bin` is run through `execute` and compared bit for bit.
#[test]
#[ignore]
fn probe_receipt_replay() {
    let dir = std::env::var("PM_NUMERICS_RECEIPT_DIR").expect("PM_NUMERICS_RECEIPT_DIR");
    let json = |name: &str| -> serde_json::Value {
        serde_json::from_slice(&std::fs::read(format!("{dir}/{name}.json")).unwrap()).unwrap()
    };
    let arch = match json("identity")["compiled_arch"].as_str().unwrap() {
        "gfx1151" => Arch::Gfx1151,
        "gfx1201" => Arch::Gfx1201,
        a => panic!("arch {a}"),
    };
    let threads = std::env::var("PM_REPLAY_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8usize);
    let mut total_words = 0u64;
    let mut total_bad = 0u64;
    let index = std::fs::read_to_string(format!("{dir}/INDEX.txt")).unwrap();
    for name in index.split_whitespace() {
        let meta = json(name);
        let kind = meta["kind"].as_str().unwrap_or("");
        if kind == "wmma_result" {
            let Some(op) = WMMA_OPS.iter().position(|o| meta["op"].as_str() == Some(o)) else {
                continue; // 16-bit-output and other unqualified kinds are not executed
            };
            let mods = parse_mods(meta["mods"].as_str().unwrap());
            let input =
                std::fs::read(format!("{dir}/{}.bin", meta["inputs"].as_str().unwrap())).unwrap();
            let hw = std::fs::read(format!("{dir}/{name}.bin")).unwrap();
            let n = hw.len() / 1024;
            assert_eq!(n as u64, meta["cases"].as_u64().unwrap());
            let bad = std::sync::atomic::AtomicU64::new(0);
            std::thread::scope(|sc| {
                for t in 0..threads {
                    let (input, hw, bad) = (&input, &hw, &bad);
                    sc.spawn(move || {
                        for w in (t..n).step_by(threads) {
                            let got = wmma_wave(arch, op, mods, &input[w * WCASE..(w + 1) * WCASE]);
                            let mut b = 0u64;
                            for (j, row) in got.iter().enumerate() {
                                for (l, &g) in row.iter().enumerate() {
                                    if g != word(&hw[w * 1024..], j * 32 + l) {
                                        b += 1;
                                    }
                                }
                            }
                            if b != 0 {
                                if bad.load(std::sync::atomic::Ordering::Relaxed) < 4 {
                                    eprintln!("{name}: wave {w} has {b} word mismatches");
                                }
                                bad.fetch_add(b, std::sync::atomic::Ordering::Relaxed);
                            }
                        }
                    });
                }
            });
            let bad = bad.into_inner();
            println!("{name}: words {} mismatches {bad}", n * 256);
            total_words += (n * 256) as u64;
            total_bad += bad;
        } else if kind == "dot_result" {
            let Some(op) = DOT_OPS.iter().position(|o| meta["op"].as_str() == Some(o)) else {
                continue;
            };
            let mods = parse_mods(meta["mods"].as_str().unwrap());
            let input =
                std::fs::read(format!("{dir}/{}.bin", meta["inputs"].as_str().unwrap())).unwrap();
            let hw = std::fs::read(format!("{dir}/{name}.bin")).unwrap();
            let n = hw.len() / 4;
            assert_eq!(n as u64, meta["cases"].as_u64().unwrap());
            let groups = n.div_ceil(32);
            let bad = std::sync::atomic::AtomicU64::new(0);
            std::thread::scope(|sc| {
                for t in 0..threads {
                    let (input, hw, bad) = (&input, &hw, &bad);
                    sc.spawn(move || {
                        for g in (t..groups).step_by(threads) {
                            let lo = g * 32;
                            let hi = (lo + 32).min(n);
                            let cases: Vec<(u32, u32, u32)> = (lo..hi)
                                .map(|i| {
                                    (
                                        word(input, 3 * i),
                                        word(input, 3 * i + 1),
                                        word(input, 3 * i + 2),
                                    )
                                })
                                .collect();
                            let got = dot_lanes(arch, op, mods, &cases);
                            for (k, &v) in got.iter().enumerate() {
                                if v != word(hw, lo + k) {
                                    if bad.load(std::sync::atomic::Ordering::Relaxed) < 4 {
                                        eprintln!(
                                            "{name}: case {} {:x?} model {v:#x} hardware {:#x}",
                                            lo + k,
                                            cases[k],
                                            word(hw, lo + k)
                                        );
                                    }
                                    bad.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                }
                            }
                        }
                    });
                }
            });
            let bad = bad.into_inner();
            println!("{name}: words {n} mismatches {bad}");
            total_words += n as u64;
            total_bad += bad;
        }
    }
    println!("TOTAL words {total_words} mismatches {total_bad}");
    assert_eq!(total_bad, 0, "hardware replay mismatches");
}

/// One streamed chunk of the `sfu_full` probe suite (opt-in; run by the probe as an external command). Environment:
/// `PM_SFU_FILE` (the GPU's raw output words), `PM_SFU_OP` (`exp|rcp|rcp_iflag|log|rsq|sqrt`), `PM_SFU_ARCH`
/// (`gfx1151|gfx1201`), `PM_SFU_BASE`, `PM_SFU_STRIDE` (input of word `i` is `base + i * stride`, modulo 2^32),
/// `PM_SFU_MODE` (the probe's MODE index and readback, for the log line), optional `PM_SFU_CSV` (first mismatches
/// written as `input,hardware,model`, capped at 100,000 rows) and `PM_REPLAY_THREADS`. Prints one `SFU ...` line and
/// fails on any mismatch. The model is `emu::trans`, the same integer function `v_exp_f32` etc. execute.
#[test]
#[ignore]
fn probe_sfu_chunk() {
    use super::trans;
    let var = |k: &str| std::env::var(k).unwrap_or_else(|_| panic!("{k}"));
    let model: fn(Arch, u32) -> u32 = match var("PM_SFU_OP").as_str() {
        "exp" => trans::exp_f32,
        "rcp" => trans::rcp_f32,
        "rcp_iflag" => trans::rcp_iflag_f32,
        "log" => trans::log_f32,
        "rsq" => trans::rsq_f32,
        "sqrt" => trans::sqrt_f32,
        op => panic!("op {op}"),
    };
    let arch = match var("PM_SFU_ARCH").as_str() {
        "gfx1151" => Arch::Gfx1151,
        "gfx1201" => Arch::Gfx1201,
        a => panic!("arch {a}"),
    };
    let (base, stride): (u32, u32) = (
        var("PM_SFU_BASE").parse().unwrap(),
        var("PM_SFU_STRIDE").parse().unwrap(),
    );
    let data = std::fs::read(var("PM_SFU_FILE")).unwrap();
    let n = data.len() / 4;
    let threads = std::env::var("PM_REPLAY_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16usize);
    let per = n.div_ceil(threads);
    let rows: Vec<(Vec<(u32, u32, u32)>, u64)> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..threads)
            .map(|t| {
                let data = &data;
                sc.spawn(move || {
                    let mut bad = Vec::new();
                    let mut count = 0u64;
                    for i in t * per..((t + 1) * per).min(n) {
                        let x = base.wrapping_add((i as u32).wrapping_mul(stride));
                        let hw = word(data, i);
                        let m = model(arch, x);
                        if hw != m {
                            count += 1;
                            if bad.len() < 100_000 / 16 {
                                bad.push((x, hw, m));
                            }
                        }
                    }
                    (bad, count)
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut total = 0u64;
    let mut listed: Vec<(u32, u32, u32)> = Vec::new();
    for (r, count) in rows {
        total += count;
        listed.extend(r);
    }
    if let Ok(csv) = std::env::var("PM_SFU_CSV") {
        if !listed.is_empty() {
            let mut s = String::from("input,hardware,model\n");
            for (x, h, m) in &listed {
                s.push_str(&format!("{x:08x},{h:08x},{m:08x}\n"));
            }
            std::fs::write(csv, s).unwrap();
        }
    }
    println!(
        "SFU arch={} op={} mode={} base={base:#010x} stride={stride} checked={n} mismatches={total} first={}",
        var("PM_SFU_ARCH"),
        var("PM_SFU_OP"),
        std::env::var("PM_SFU_MODE").unwrap_or_default(),
        listed
            .iter()
            .min_by_key(|r| r.0)
            .map(|(x, h, m)| format!("{x:08x}:hw={h:08x}:model={m:08x}"))
            .unwrap_or_else(|| "none".into())
    );
    assert_eq!(total, 0, "SFU model mismatches");
}
