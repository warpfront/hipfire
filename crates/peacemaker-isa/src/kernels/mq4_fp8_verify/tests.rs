// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use super::*;

#[test]
fn entries_are_deterministic_and_target_only_gfx1201() {
    for spec in Spec::all(Arch::Gfx1201) {
        let (a, b) = (emit(spec).unwrap(), emit(spec).unwrap());
        assert_eq!(a.proof.s_text_sha256, b.proof.s_text_sha256, "{spec:?}");
        if spec.direct() {
            // Two chains of eight WMMAs per group, no barriers, no LDS.
            assert_eq!(a.shape.wmma, 16, "{spec:?}");
            assert_eq!(a.shape.barriers, 0, "{spec:?}");
            assert!(a.proof.lds_slots.is_empty(), "{spec:?}");
        } else {
            // Two 16-K fragments x BT token tiles per 32-K slab, eight slabs per group.
            assert_eq!(a.shape.wmma, 16 * usize::from(spec.bt), "{spec:?}");
            // One prologue barrier, one per slab.
            assert_eq!(a.shape.barriers, 9, "{spec:?}");
        }
        assert!(
            a.shape.next_free_vgpr <= 256 && a.shape.next_free_sgpr <= 80,
            "{spec:?}: {:?}",
            a.shape
        );
    }
    for arch in [Arch::Gfx1100, Arch::Gfx1151] {
        assert!(emit(Spec {
            arch,
            bt: 8,
            waves: 4,
            op: Fp8Op::Residual
        })
        .is_err());
    }
    for (bt, waves) in [(1, 4), (2, 2), (3, 4), (6, 4), (8, 2), (8, 16), (16, 4)] {
        assert!(emit(Spec {
            arch: Arch::Gfx1201,
            bt,
            waves,
            op: Fp8Op::Qkv
        })
        .is_err());
    }
}

#[test]
fn module_exports_the_thirtyfive_frozen_symbols_in_order() {
    let (emitted, text, _) = emit_module(Arch::Gfx1201).unwrap();
    assert_eq!(emitted.len(), 35);
    let got: Vec<String> = emitted.iter().map(|e| e.proof.variant.clone()).collect();
    let want: Vec<String> = Spec::all(Arch::Gfx1201).iter().map(Spec::tag).collect();
    assert_eq!(got, want);
    assert_eq!(
        &want[..8],
        &[
            "qkvza_bt1w1",
            "qkvza_bt2w4",
            "qkvza_bt2w8",
            "qkvza_bt4w4",
            "qkvza_bt4w8",
            "qkvza_bt8w4",
            "qkvza_bt8w8",
            "qkv_bt1w1"
        ]
    );
    assert_eq!(want.last().map(String::as_str), Some("set_bt8w8"));
    for spec in Spec::all(Arch::Gfx1201) {
        assert!(text.contains(&format!(".amdhsa_kernel {}\n", spec.symbol())));
    }
}

#[test]
fn kernarg_abi_is_the_spec_layout() {
    assert_eq!(Fp8Op::ALL.map(Fp8Op::kernarg_bytes), [104, 84, 64, 44, 44]);
    for op in Fp8Op::ALL {
        let spec = Spec {
            arch: Arch::Gfx1201,
            bt: 8,
            waves: 4,
            op,
        };
        let k = kernargs(&spec);
        k.validate().unwrap();
        assert_eq!(
            k.args.last().map(|a| a.offset + a.size),
            Some(op.kernarg_bytes())
        );
        let covered: u32 = kernarg_loads(&spec)
            .iter()
            .map(|(_, d)| 4 * u32::from(*d))
            .sum();
        assert_eq!(covered, op.kernarg_bytes());
        let n = u32::from(op.matrices());
        let find = |name: &str| k.args.iter().find(|a| a.name == name).unwrap().offset;
        assert_eq!(find("X"), 8 * n);
        assert_eq!(find("S"), 8 * n + 8);
        assert_eq!(find("K"), 20 * n + 16);
        assert_eq!(find("N"), 20 * n + 20);
    }
}

/// The documented register high waters and LDS (the contracts and M7 receipts
/// carry the same numbers).
#[test]
fn register_budgets_are_the_documented_ones() {
    let vgpr = |bt: u8, waves: u8| match (bt, waves) {
        (1, 1) => 147,
        (2, 4) => 103,
        (2, 8) => 101,
        (4, 4) => 147,
        (4, 8) => 143,
        (8, 4) => 235,
        (8, 8) => 227,
        _ => unreachable!(),
    };
    let lds = |bt: u8| match bt {
        1 => 0,
        2 => 3072,
        4 => 6144,
        _ => 12288,
    };
    for spec in Spec::all(Arch::Gfx1201) {
        let e = emit(spec).unwrap();
        assert_eq!(
            e.shape.next_free_vgpr,
            vgpr(spec.bt, spec.waves),
            "{spec:?}"
        );
        assert_eq!(
            e.shape.next_free_sgpr,
            u16::from(KA) + u16::from(5 * spec.op.matrices()) + 6,
            "{spec:?}"
        );
        assert_eq!(spec.lds_bytes(), lds(spec.bt), "{spec:?}");
    }
}

/// The runtime embeds `kernels/mq4_fp8_verify_pm_gfx1201.hxaco`: it must be
/// exactly the native writer's bundle of today's emission, and every
/// committed shape contract exactly today's.
#[test]
fn committed_bundle_and_contracts_equal_fresh_emission() {
    let (emitted, text, _) = emit_module(Arch::Gfx1201).unwrap();
    let elf = crate::native::assemble(&text, Arch::Gfx1201).unwrap();
    let fresh = crate::native::bundle(&elf, Arch::Gfx1201, "host-x86_64-unknown-linux-gnu-");
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../kernels/mq4_fp8_verify_pm_gfx1201.hxaco"
    );
    assert!(
        std::fs::read(path).unwrap() == fresh,
        "{path}: fresh emission differs from the committed bundle"
    );
    for (spec, e) in Spec::all(Arch::Gfx1201).into_iter().zip(&emitted) {
        let path = format!(
            "{}/kernels/mq4_fp8_verify.gfx1201.{}.contract.json",
            env!("CARGO_MANIFEST_DIR"),
            spec.tag()
        );
        let committed: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(committed, contract(spec, e), "{path}");
    }
}

/// M7 on the linked object: every export lifts byte-exact with no
/// obligations and no ambiguous delays, and the receipts equal the committed ones.
#[cfg(feature = "toolchain")]
#[test]
fn m7_certifies_every_export_and_matches_the_committed_receipts() {
    let (_, text, _) = emit_module(Arch::Gfx1201).unwrap();
    let elf = crate::native::assemble(&text, Arch::Gfx1201).unwrap();
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!(
        "../../kernels/mq4_fp8_verify.test-{}.co",
        std::process::id()
    ));
    std::fs::write(&path, elf).unwrap();
    let mut receipts = serde_json::Map::new();
    let mut result = Ok(());
    for spec in Spec::all(Arch::Gfx1201) {
        match crate::pm_check::m7(&path, "gfx1201", &spec.symbol()) {
            Ok(r) => {
                if r["lift"] != "byte-exact"
                    || r["obligations"] != serde_json::json!({})
                    || r["ambiguous_delays"] != 0
                {
                    result = Err(format!("{}: {r}", spec.symbol()))
                }
                receipts.insert(spec.symbol(), r);
            }
            Err(e) => result = Err(e),
        }
    }
    std::fs::remove_file(&path).unwrap();
    result.unwrap();
    let committed: serde_json::Value = serde_json::from_slice(
        &std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/kernels/mq4_fp8_verify.gfx1201.m7.json"
        ))
        .unwrap(),
    )
    .unwrap();
    assert_eq!(committed, serde_json::Value::Object(receipts));
}

// ------------------------------------------------- interpreted reference check
// Every entry's emitted text runs whole grids in a small in-order wave32
// interpreter of exactly the instructions this builder emits (waits and
// hints are no-ops: memory completes at issue; barriers rendezvous the
// block's waves), on small problems — projection rows straddling 8/16,
// rows past total_m, tokens past N, a second grid row and column, one, two
// and three 256-K groups — against an independent reference of the
// KERNEL-SPEC arithmetic contract: per output, per 128-K half the exact sum of
// the E4M3 x (q - 8) products (the WMMA chain, exact because the test
// data keeps every partial sum exactly representable in f32: E4M3
// activations with exponents in [-3, 4] and integer weights -8..7), folded
// as `T = fma(sc, P, T)` from +0 in ascending (g, h), then `Y = a * T`
// (residual: `fma(a, T, Y_old)`). Every byte of every output buffer
// (including rows past total_m, tokens past N and a canary tail) is
// compared. This checks data movement, indexing and the dequant byte
// tables; the GPU byte oracle is the authority for the hardware arithmetic.
use peacemaker_ir::emu::convert::{f16_to_f32, f32_to_f16, fp8_to_f32};
use std::collections::HashMap;

#[derive(Clone, Debug)]
enum Opd {
    V(u16, Option<bool>),
    S(u16, u16),
    Exec,
    Null,
    Ttmp(u16),
    Imm(u32),
    Label(String),
}
struct Ins {
    m: String,
    o: Vec<Opd>,
    off: u32,
}
fn opd(t: &str) -> Opd {
    let t = t.trim();
    let range = |r: &str| {
        let (a, b) = r.trim_end_matches(']').split_once(':').unwrap();
        let (a, b): (u16, u16) = (a.parse().unwrap(), b.parse().unwrap());
        (a, b - a + 1)
    };
    if t == "exec_lo" {
        return Opd::Exec;
    }
    if t == "null" || t == "off" {
        return Opd::Null;
    }
    if t.starts_with(".L") {
        return Opd::Label(t.into());
    }
    if let Some(n) = t.strip_prefix("ttmp") {
        return Opd::Ttmp(n.parse().unwrap());
    }
    if let Some(r) = t.strip_prefix("v[") {
        return Opd::V(range(r).0, None);
    }
    if let Some(r) = t.strip_prefix("s[") {
        let (a, n) = range(r);
        return Opd::S(a, n);
    }
    if let Some(r) = t.strip_prefix('v') {
        let (n, half) = if let Some(n) = r.strip_suffix(".l") {
            (n, Some(false))
        } else if let Some(n) = r.strip_suffix(".h") {
            (n, Some(true))
        } else {
            (r, None)
        };
        if let Ok(n) = n.parse() {
            return Opd::V(n, half);
        }
    }
    if let Some(r) = t.strip_prefix('s') {
        if let Ok(n) = r.parse() {
            return Opd::S(n, 1);
        }
    }
    if let Some(h) = t.strip_prefix("0x") {
        return Opd::Imm(u32::from_str_radix(h, 16).unwrap());
    }
    Opd::Imm(t.parse::<i64>().unwrap_or_else(|_| panic!("operand {t}")) as u32)
}
/// The kernel's instructions and labels.
fn parse(text: &str) -> (Vec<Ins>, HashMap<String, usize>) {
    let (mut prog, mut labels) = (Vec::new(), HashMap::new());
    for line in text
        .lines()
        .skip_while(|l| !l.ends_with(':') || l.starts_with('.'))
        .skip(1)
    {
        if line.starts_with('.') && !line.ends_with(':') {
            break;
        }
        if let Some(l) = line.trim().strip_suffix(':') {
            labels.insert(l.to_string(), prog.len());
            continue;
        }
        let line = line.trim();
        let (m, rest) = line.split_once(' ').map_or((line, ""), |(m, r)| (m, r));
        if m.starts_with("s_wait") || m == "s_delay_alu" || m == "v_nop" {
            continue;
        }
        let mut rest = rest.to_string();
        if let Some(p) = rest.find(" op_sel:") {
            rest.truncate(p)
        }
        let mut off = 0;
        if let Some(p) = rest.find(" offset:") {
            off = rest[p + 8..]
                .split_whitespace()
                .next()
                .unwrap()
                .parse()
                .unwrap();
            rest.truncate(p)
        }
        if let Some(p) = rest.find(" offen") {
            rest.truncate(p)
        }
        let o = if rest.is_empty() {
            vec![]
        } else {
            rest.split(", ").map(opd).collect()
        };
        prog.push(Ins {
            m: m.to_string(),
            o,
            off,
        });
    }
    (prog, labels)
}
struct Mem {
    regions: Vec<(u64, Vec<u8>)>,
}
impl Mem {
    fn at(&mut self, addr: u64, n: usize) -> &mut [u8] {
        let r = self
            .regions
            .iter_mut()
            .find(|(b, v)| addr >= *b && addr + n as u64 <= *b + v.len() as u64)
            .unwrap_or_else(|| panic!("unmapped access {addr:#x}+{n}"));
        let o = (addr - r.0) as usize;
        &mut r.1[o..o + n]
    }
}
struct Wave {
    v: Vec<[u32; 32]>,
    s: [u32; 128],
    exec: u32,
    scc: bool,
    pc: usize,
    done: bool,
}
impl Wave {
    fn sc(&self, o: &Opd, t: [u32; 2]) -> u32 {
        match o {
            Opd::S(n, _) => self.s[usize::from(*n)],
            Opd::Imm(x) => *x,
            Opd::Ttmp(9) => t[0],
            Opd::Ttmp(7) => t[1],
            Opd::Exec => self.exec,
            Opd::Null => 0,
            _ => panic!("scalar {o:?}"),
        }
    }
    fn rd(&self, o: &Opd, l: usize, t: [u32; 2]) -> u32 {
        match o {
            Opd::V(n, h) => {
                let x = self.v[usize::from(*n)][l];
                match h {
                    Some(false) => x & 0xffff,
                    Some(true) => x >> 16,
                    None => x,
                }
            }
            _ => self.sc(o, t),
        }
    }
    fn wr(&mut self, o: &Opd, l: usize, x: u32) {
        let Opd::V(n, h) = o else {
            panic!("vector destination {o:?}")
        };
        let r = &mut self.v[usize::from(*n)][l];
        *r = match h {
            Some(false) => (*r & 0xffff_0000) | (x & 0xffff),
            Some(true) => (*r & 0xffff) | (x << 16),
            None => x,
        };
    }
    fn base(o: &Opd) -> usize {
        match o {
            Opd::V(n, _) | Opd::S(n, _) => usize::from(*n),
            _ => panic!("register {o:?}"),
        }
    }
    fn sdst(&mut self, o: &Opd, x: u32) {
        match o {
            Opd::Exec => self.exec = x,
            _ => self.s[Self::base(o)] = x,
        }
    }
}
fn f16f(h: u32) -> f32 {
    f32::from_bits(f16_to_f32(h as u16))
}
fn f16(x: f32) -> u16 {
    f32_to_f16(x.to_bits())
}
fn fp8(b: u8) -> f64 {
    f64::from(f32::from_bits(fp8_to_f32(b)))
}
/// `v_perm_b32` byte select over {S0, S1} (S1 = bytes 0..3).
fn perm(s0: u32, s1: u32, sel: u32) -> u32 {
    let data = u64::from(s0) << 32 | u64::from(s1);
    (0..4).fold(0u32, |out, k| {
        let sl = (sel >> (8 * k)) & 0xff;
        let b = match sl {
            0..=7 => (data >> (8 * sl)) as u32 & 0xff,
            8..=11 => {
                if data >> (8 * (2 * (sl - 8) + 1) + 7) & 1 == 1 {
                    0xff
                } else {
                    0
                }
            }
            12 => 0,
            _ => 0xff,
        };
        out | b << (8 * k)
    })
}

/// Run one workgroup of the parsed kernel to completion.
fn run_wg(
    prog: &(Vec<Ins>, HashMap<String, usize>),
    mem: &mut Mem,
    lds: &mut [u8],
    waves: usize,
    kernarg: u64,
    t: [u32; 2],
) {
    let (code, labels) = prog;
    let mut w: Vec<Wave> = (0..waves)
        .map(|i| {
            let mut s = [0u32; 128];
            s[0] = kernarg as u32;
            s[1] = (kernarg >> 32) as u32;
            let mut v = vec![[0u32; 32]; 256];
            for l in 0..32 {
                v[0][l] = (32 * i + l) as u32;
            }
            Wave {
                v,
                s,
                exec: u32::MAX,
                scc: false,
                pc: 0,
                done: false,
            }
        })
        .collect();
    loop {
        let mut waiting = 0;
        for wv in w.iter_mut().filter(|x| !x.done) {
            loop {
                let i = &code[wv.pc];
                wv.pc += 1;
                if step(wv, i, labels, mem, lds, t) {
                    waiting += 1;
                    break;
                }
                if wv.done {
                    break;
                }
            }
        }
        if w.iter().all(|x| x.done) {
            return;
        }
        assert_eq!(
            waiting,
            w.iter().filter(|x| !x.done).count(),
            "a wave ended while others wait at a barrier"
        );
    }
}
/// Execute one instruction; true at a barrier wait.
fn step(
    w: &mut Wave,
    i: &Ins,
    labels: &HashMap<String, usize>,
    mem: &mut Mem,
    lds: &mut [u8],
    t: [u32; 2],
) -> bool {
    let o = &i.o;
    let lanes = |exec: u32| (0..32usize).filter(move |l| exec >> l & 1 != 0);
    let valu = |w: &mut Wave, f: &dyn Fn(&Wave, usize) -> u32| {
        for l in lanes(w.exec) {
            let x = f(w, l);
            w.wr(&o[0], l, x);
        }
    };
    let vcmp = |w: &mut Wave, f: &dyn Fn(i32, i32) -> bool| {
        let mut m = 0u32;
        for l in lanes(w.exec) {
            if f(w.rd(&o[1], l, t) as i32, w.rd(&o[2], l, t) as i32) {
                m |= 1 << l
            }
        }
        w.sdst(&o[0], m);
    };
    let s = |w: &Wave, k: usize| w.sc(&o[k], t);
    let label = |o: &Opd| match o {
        Opd::Label(l) => labels[l],
        _ => panic!("branch target {o:?}"),
    };
    let width = |m: &str| match m.rsplit_once("_b").unwrap().1 {
        "32" => 1usize,
        "64" => 2,
        "128" => 4,
        x => panic!("{x}"),
    };
    match i.m.as_str() {
        "s_barrier_signal" => {}
        "s_barrier_wait" => return true,
        "s_endpgm" => w.done = true,
        "s_branch" => w.pc = label(&o[0]),
        "s_cbranch_scc1" => {
            if w.scc {
                w.pc = label(&o[0])
            }
        }
        "s_load_b32" | "s_load_b64" | "s_load_b128" | "s_load_b256" => {
            let p = Wave::base(&o[1]);
            let addr = u64::from(w.s[p]) | u64::from(w.s[p + 1]) << 32;
            let (Opd::S(d, n), Opd::Imm(off)) = (&o[0], &o[2]) else {
                panic!("{}", i.m)
            };
            for k in 0..usize::from(*n) {
                w.s[usize::from(*d) + k] = u32::from_le_bytes(
                    mem.at(addr + u64::from(*off) + 4 * k as u64, 4)
                        .try_into()
                        .unwrap(),
                );
            }
        }
        "s_mov_b32" => {
            let x = s(w, 1);
            w.sdst(&o[0], x)
        }
        "s_and_b32" => {
            let x = s(w, 1) & s(w, 2);
            w.sdst(&o[0], x);
            w.scc = x != 0
        }
        "s_or_b32" => {
            let x = s(w, 1) | s(w, 2);
            w.sdst(&o[0], x);
            w.scc = x != 0
        }
        "s_mul_i32" => {
            let x = s(w, 1).wrapping_mul(s(w, 2));
            w.sdst(&o[0], x)
        }
        "s_lshl_b32" => {
            let x = s(w, 1) << (s(w, 2) & 31);
            w.sdst(&o[0], x)
        }
        "s_ashr_i32" => {
            let x = ((s(w, 1) as i32) >> (s(w, 2) & 31)) as u32;
            w.sdst(&o[0], x)
        }
        "s_add_co_i32" => {
            let x = s(w, 1).wrapping_add(s(w, 2));
            w.sdst(&o[0], x)
        }
        "s_sub_co_i32" => {
            let x = s(w, 1).wrapping_sub(s(w, 2));
            w.sdst(&o[0], x)
        }
        "s_min_u32" => {
            let x = s(w, 1).min(s(w, 2));
            w.sdst(&o[0], x)
        }
        "s_cmp_ge_i32" => w.scc = s(w, 0) as i32 >= s(w, 1) as i32,
        "s_cmp_lt_i32" => w.scc = (s(w, 0) as i32) < s(w, 1) as i32,
        "s_cmp_lt_u32" => w.scc = s(w, 0) < s(w, 1),
        "s_cmp_lg_u32" => w.scc = s(w, 0) != s(w, 1),
        "v_mov_b32_e32" => valu(w, &|w, l| w.rd(&o[1], l, t)),
        "v_and_b32_e32" => valu(w, &|w, l| w.rd(&o[1], l, t) & w.rd(&o[2], l, t)),
        "v_or_b32_e32" => valu(w, &|w, l| w.rd(&o[1], l, t) | w.rd(&o[2], l, t)),
        "v_xor_b32_e32" => valu(w, &|w, l| w.rd(&o[1], l, t) ^ w.rd(&o[2], l, t)),
        "v_add_nc_u32_e32" => valu(w, &|w, l| w.rd(&o[1], l, t).wrapping_add(w.rd(&o[2], l, t))),
        "v_sub_nc_u32_e32" => valu(w, &|w, l| w.rd(&o[1], l, t).wrapping_sub(w.rd(&o[2], l, t))),
        "v_min_i32_e32" => valu(w, &|w, l| {
            (w.rd(&o[1], l, t) as i32).min(w.rd(&o[2], l, t) as i32) as u32
        }),
        "v_mul_lo_u32" => valu(w, &|w, l| w.rd(&o[1], l, t).wrapping_mul(w.rd(&o[2], l, t))),
        "v_mul_u32_u24_e32" => valu(w, &|w, l| {
            (w.rd(&o[1], l, t) & 0xff_ffff).wrapping_mul(w.rd(&o[2], l, t) & 0xff_ffff)
        }),
        "v_lshlrev_b32_e32" => valu(w, &|w, l| w.rd(&o[2], l, t) << (w.rd(&o[1], l, t) & 31)),
        "v_lshrrev_b32_e32" => valu(w, &|w, l| w.rd(&o[2], l, t) >> (w.rd(&o[1], l, t) & 31)),
        "v_lshl_add_u32" => valu(w, &|w, l| {
            (w.rd(&o[1], l, t) << (w.rd(&o[2], l, t) & 31)).wrapping_add(w.rd(&o[3], l, t))
        }),
        "v_bfe_u32" => valu(w, &|w, l| {
            let (x, a, b) = (
                w.rd(&o[1], l, t),
                w.rd(&o[2], l, t) & 31,
                w.rd(&o[3], l, t) & 31,
            );
            (x >> a) & ((1u64 << b) - 1) as u32
        }),
        "v_mad_u32_u24" => valu(w, &|w, l| {
            (w.rd(&o[1], l, t) & 0xff_ffff)
                .wrapping_mul(w.rd(&o[2], l, t) & 0xff_ffff)
                .wrapping_add(w.rd(&o[3], l, t))
        }),
        "v_cndmask_b32_e64" => valu(w, &|w, l| {
            if w.sc(&o[3], t) >> l & 1 != 0 {
                w.rd(&o[2], l, t)
            } else {
                w.rd(&o[1], l, t)
            }
        }),
        "v_perm_b32" => valu(w, &|w, l| {
            perm(w.rd(&o[1], l, t), w.rd(&o[2], l, t), w.rd(&o[3], l, t))
        }),
        "v_bfi_b32" => valu(w, &|w, l| {
            let m = w.rd(&o[1], l, t);
            (m & w.rd(&o[2], l, t)) | (!m & w.rd(&o[3], l, t))
        }),
        "v_fma_f32" => valu(w, &|w, l| {
            f32::from_bits(w.rd(&o[1], l, t))
                .mul_add(
                    f32::from_bits(w.rd(&o[2], l, t)),
                    f32::from_bits(w.rd(&o[3], l, t)),
                )
                .to_bits()
        }),
        "v_mul_f32_e32" => valu(w, &|w, l| {
            (f32::from_bits(w.rd(&o[1], l, t)) * f32::from_bits(w.rd(&o[2], l, t))).to_bits()
        }),
        "v_cvt_f32_f16_e64" => valu(w, &|w, l| f16f(w.rd(&o[1], l, t)).to_bits()),
        "v_cmp_gt_i32_e64" => vcmp(w, &|a, b| a > b),
        "v_cmp_lt_i32_e64" => vcmp(w, &|a, b| a < b),
        "v_cmp_le_i32_e64" => vcmp(w, &|a, b| a <= b),
        "v_mad_co_u64_u32" => {
            let (d, c) = (Wave::base(&o[0]), Wave::base(&o[4]));
            for l in lanes(w.exec) {
                let x = u64::from(w.rd(&o[2], l, t)) * u64::from(w.rd(&o[3], l, t))
                    + (u64::from(w.v[c][l]) | u64::from(w.v[c + 1][l]) << 32);
                w.v[d][l] = x as u32;
                w.v[d + 1][l] = (x >> 32) as u32;
            }
        }
        "v_wmma_f32_16x16x16_fp8_fp8" => {
            // D[token][column] = C + sum_k A[token][k] * B[k][column]: lane = row|col + 16*(k/8),
            // VGPR (k%8)/4, byte k%4. Exact in f64; the test data keeps it exact in f32.
            let (d, a, b) = (Wave::base(&o[0]), Wave::base(&o[1]), Wave::base(&o[2]));
            let byte =
                |r: usize, l: usize, k: usize| (w.v[r + (k % 8) / 4][l] >> (8 * (k % 4))) as u8;
            let mut out = [[0u32; 32]; 8];
            for row in 0..16 {
                for col in 0..16 {
                    let (lane, reg) = (col + 16 * (row / 8), row % 8);
                    let c = match &o[3] {
                        Opd::Imm(0) => 0.0f32,
                        Opd::V(..) => f32::from_bits(w.v[Wave::base(&o[3]) + reg][lane]),
                        x => panic!("WMMA C {x:?}"),
                    };
                    let sum: f64 = f64::from(c)
                        + (0..16)
                            .map(|k| {
                                fp8(byte(a, row + 16 * (k / 8), k))
                                    * fp8(byte(b, col + 16 * (k / 8), k))
                            })
                            .sum::<f64>();
                    assert_eq!(
                        f64::from(sum as f32),
                        sum,
                        "WMMA sum {sum} is not exact in f32 (test data)"
                    );
                    out[reg][lane] = (sum as f32).to_bits();
                }
            }
            for (r, x) in out.into_iter().enumerate() {
                w.v[d + r] = x;
            }
        }
        "ds_load_b128" | "ds_store_b32" | "ds_store_b64" | "ds_store_b128" => {
            let load = i.m == "ds_load_b128";
            let n = if load { 4 } else { width(&i.m) };
            let (addr, data) = if load {
                (&o[1], Wave::base(&o[0]))
            } else {
                (&o[0], Wave::base(&o[1]))
            };
            for l in lanes(w.exec) {
                let at = (w.rd(addr, l, t) + i.off) as usize;
                assert!(
                    at + 4 * n <= lds.len(),
                    "LDS access {at} past {}",
                    lds.len()
                );
                for k in 0..n {
                    if load {
                        w.v[data + k][l] =
                            u32::from_le_bytes(lds[at + 4 * k..at + 4 * k + 4].try_into().unwrap())
                    } else {
                        lds[at + 4 * k..at + 4 * k + 4]
                            .copy_from_slice(&w.v[data + k][l].to_le_bytes())
                    }
                }
            }
        }
        "buffer_load_b32" | "buffer_load_b64" | "buffer_load_b128" => {
            let srd = Wave::base(&o[2]);
            let base = u64::from(w.s[srd]) | u64::from(w.s[srd + 1] & 0xffff) << 32;
            let d = Wave::base(&o[0]);
            for l in lanes(w.exec) {
                let at = u64::from(w.rd(&o[1], l, t)) + u64::from(s(w, 3)) + u64::from(i.off);
                for k in 0..width(&i.m) as u64 {
                    // Raw buffer, per-dword range check against num_records (bytes): out of range reads 0.
                    let x = if at + 4 * k + 4 > u64::from(w.s[srd + 2]) {
                        0
                    } else {
                        u32::from_le_bytes(mem.at(base + at + 4 * k, 4).try_into().unwrap())
                    };
                    w.v[d + k as usize][l] = x;
                }
            }
        }
        m if m.starts_with("global_load_b") || m.starts_with("global_store_b") => {
            let load = m.starts_with("global_load");
            let dw = width(m);
            let (p, d) = if load {
                (Wave::base(&o[1]), Wave::base(&o[0]))
            } else {
                (Wave::base(&o[0]), Wave::base(&o[1]))
            };
            for l in lanes(w.exec) {
                let addr =
                    (u64::from(w.v[p][l]) | u64::from(w.v[p + 1][l]) << 32) + u64::from(i.off);
                for k in 0..dw {
                    let at = mem.at(addr + 4 * k as u64, 4);
                    if load {
                        w.v[d + k][l] = u32::from_le_bytes((&*at).try_into().unwrap())
                    } else {
                        at.copy_from_slice(&w.v[d + k][l].to_le_bytes())
                    }
                }
            }
        }
        m => panic!("interpreter: no semantics for {m}"),
    }
    false
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 16) as u32
    }
}
const CANARY: usize = 64;
const SENTINEL: u32 = 0x7FBA_DBAD;

/// One problem: weights `a`, X bytes, scales, initial Y; returns each Y buffer's final bytes.
fn run_case(text: &str, spec: Spec, ms: &[u32], k: u32, n: u32, seed: u64) -> Vec<Vec<u8>> {
    let mut rng = Rng(seed);
    let nm = ms.len();
    let gpr = (k / 256) as usize;
    let row_bytes = gpr * GROUP as usize;
    // Weights: f16 scale headers (mixed signs; the zero-point halves are never read), random nibbles.
    let a: Vec<Vec<u8>> = ms
        .iter()
        .map(|&m| {
            let mut w = vec![0u8; m as usize * row_bytes];
            for row in w.chunks_mut(GROUP as usize) {
                for h in 0..2 {
                    let sc = f16(((rng.next() % 2001) as f32 - 1000.0) * 1.3e-4);
                    let zp = f16(((rng.next() % 2001) as f32 - 1000.0) * 9e-4);
                    row[4 * h..4 * h + 2].copy_from_slice(&sc.to_le_bytes());
                    row[4 * h + 2..4 * h + 4].copy_from_slice(&zp.to_le_bytes());
                }
                for byte in &mut row[8..] {
                    *byte = rng.next() as u8;
                }
            }
            w
        })
        .collect();
    // Activations: E4M3 normals with exponents in [-3, 4] (field 4..=11), both zeros.
    let x: Vec<u8> = (0..n as usize * k as usize)
        .map(|_| match rng.next() % 41 {
            0 => 0x80,
            1 => 0,
            _ => {
                ((rng.next() % 2) as u8) << 7
                    | (4 + (rng.next() % 8) as u8) << 3
                    | (rng.next() % 8) as u8
            }
        })
        .collect();
    let sv: Vec<f32> = (0..n)
        .map(|_| ((rng.next() % 4001) as f32 - 2000.0) * 3.7e-4)
        .collect();
    let y0: Vec<Vec<u32>> = ms
        .iter()
        .enumerate()
        .map(|(i, &m)| {
            (0..n as usize * m as usize + CANARY)
                .map(|e| {
                    if spec.op.accumulates() && e < n as usize * m as usize {
                        (1.0f32 + (e + 7 * i) as f32 * 0.001).to_bits()
                            ^ (u32::from(e % 5 == 0) << 31)
                    } else {
                        SENTINEL
                    }
                })
                .collect()
        })
        .collect();
    // Reference.
    let mut want = y0.clone();
    for (i, &m) in ms.iter().enumerate() {
        for r in 0..m as usize {
            let g = |gg: usize| &a[i][r * row_bytes + gg * GROUP as usize..][..GROUP as usize];
            let wt = |kk: usize| -> f64 {
                let byte = g(kk / 256)[8 + (kk % 256) / 2];
                f64::from(if kk % 2 == 0 { byte & 15 } else { byte >> 4 }) - 8.0
            };
            for t in 0..n as usize {
                let mut tt = 0.0f32;
                for gg in 0..gpr {
                    for h in 0..2 {
                        let hd = g(gg);
                        let sc = f16f(u32::from(u16::from_le_bytes([hd[4 * h], hd[4 * h + 1]])));
                        let p: f64 = (0..128)
                            .map(|kk| {
                                fp8(x[t * k as usize + 256 * gg + 128 * h + kk])
                                    * wt(256 * gg + 128 * h + kk)
                            })
                            .sum();
                        assert_eq!(
                            f64::from(p as f32),
                            p,
                            "reference partial {p} is not exact in f32 (test data)"
                        );
                        tt = sc.mul_add(p as f32, tt);
                    }
                }
                let o = &mut want[i][t * m as usize + r];
                *o = if spec.op.accumulates() {
                    sv[t].mul_add(tt, f32::from_bits(*o)).to_bits()
                } else {
                    (sv[t] * tt).to_bits()
                };
            }
        }
    }
    // Device image.
    let le = |v: &[u32]| v.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>();
    let a_base = |i: usize| 0x1000_0000u64 + 0x0100_0000 * i as u64;
    let y_base = |i: usize| 0x3000_0000u64 + 0x0100_0000 * i as u64;
    let (x_base, s_base, kernarg_ptr) = (0x2000_0000u64, 0x2800_0000u64, 0x4000_0000u64);
    let mut karg = Vec::new();
    for i in 0..nm {
        karg.extend(a_base(i).to_le_bytes());
    }
    karg.extend(x_base.to_le_bytes());
    karg.extend(s_base.to_le_bytes());
    for i in 0..nm {
        karg.extend(y_base(i).to_le_bytes());
    }
    for &m in ms {
        karg.extend(m.to_le_bytes());
    }
    karg.extend(k.to_le_bytes());
    karg.extend(n.to_le_bytes());
    assert_eq!(karg.len() as u32, spec.op.kernarg_bytes());
    let mut regions: Vec<(u64, Vec<u8>)> = (0..nm).map(|i| (a_base(i), a[i].clone())).collect();
    regions.push((x_base, x));
    regions.push((s_base, sv.iter().flat_map(|f| f.to_le_bytes()).collect()));
    regions.extend((0..nm).map(|i| (y_base(i), le(&y0[i]))));
    regions.push((kernarg_ptr, karg));
    let mut mem = Mem { regions };
    let prog = parse(text);
    let total: u32 = ms.iter().sum();
    for gy in 0..n.div_ceil(spec.tokens()) {
        for gx in 0..total.div_ceil(spec.rows()) {
            let mut lds = vec![0xA5u8; spec.lds_bytes() as usize];
            run_wg(
                &prog,
                &mut mem,
                &mut lds,
                usize::from(spec.waves),
                kernarg_ptr,
                [gx, gy],
            );
        }
    }
    ms.iter().enumerate().map(|(i, &m)| {
        let got = mem.at(y_base(i), want[i].len() * 4).to_vec();
        for (e, w) in want[i].iter().enumerate() {
            let g = u32::from_le_bytes(got[4 * e..4 * e + 4].try_into().unwrap());
            assert_eq!(g, *w, "{} rows {ms:?} K {k} N {n}: Y{i} element {e} (token {}, row {}): got {g:#010x} want {w:#010x}",
                spec.symbol(), e / m as usize, e % m as usize);
        }
        got
    }).collect()
}

fn problems(op: Fp8Op) -> ([u32; 4], [u32; 4], usize) {
    match op {
        Fp8Op::Qkvza => ([40, 24, 8, 32], [37, 21, 13, 27], 4),
        Fp8Op::Qkv => ([40, 24, 32, 0], [37, 21, 22, 0], 3),
        Fp8Op::GateUp => ([48, 40, 0, 0], [45, 30, 0, 0], 2),
        Fp8Op::Residual | Fp8Op::Set => ([100, 0, 0, 0], [61, 0, 0, 0], 1),
    }
}

#[test]
fn interpreted_entries_are_byte_exact_against_the_contract() {
    for spec in Spec::all(Arch::Gfx1201) {
        let text = emit(spec).unwrap().s_text;
        let (aligned, ragged, n) = problems(spec.op);
        let (aligned, ragged) = (&aligned[..n], &ragged[..n]);
        let tok = spec.tokens();
        // Second grid row and column (tokens past N in the second column), K = 768, rows past total_m.
        run_case(
            &text,
            spec,
            aligned,
            768,
            tok + 5,
            0x9e37_79b9_7f4a_7c15 ^ u64::from(spec.bt),
        );
        // Ragged projections, tokens past N inside the tile, K = 512.
        run_case(
            &text,
            spec,
            ragged,
            512,
            tok - 3,
            0x2545_f491_4f6c_dd1d ^ u64::from(spec.waves),
        );
        // One token, one group.
        run_case(&text, spec, aligned, 256, 1, 0x1234_5678_9abc_def1);
        // A full tile exactly.
        run_case(&text, spec, ragged, 256, tok, 0x0fed_cba9_8765_4321);
    }
}

/// Row-count independence by construction, checked directly: every tile (the
/// direct template and all six staged ones) produces identical bytes for the
/// same problem.
#[test]
fn every_template_produces_identical_bytes_on_the_same_problem() {
    for op in Fp8Op::ALL {
        let (_, ragged, n) = problems(op);
        let ragged = &ragged[..n];
        let mut first: Option<(String, Vec<Vec<u8>>)> = None;
        for spec in Spec::all(Arch::Gfx1201).into_iter().filter(|s| s.op == op) {
            let out = run_case(
                &emit(spec).unwrap().s_text,
                spec,
                ragged,
                512,
                21,
                0xdead_beef_cafe_f00d,
            );
            match &first {
                None => first = Some((spec.tag(), out)),
                Some((tag, want)) => assert!(&out == want, "{} differs from {tag}", spec.tag()),
            }
        }
    }
}
