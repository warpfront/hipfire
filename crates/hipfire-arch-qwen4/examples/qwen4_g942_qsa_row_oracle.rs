// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Exact-image byte oracle of the exact-gfx942 single-row parallel QSA attention module
//! (`kernels/src/qwen4_g942_qsa_row_attention.gfx942.hip`, module
//! `qwen4_g942_qsa_row_attention`, entries `qwen4_g942_qsa_row_f32` and
//! `qwen4_g942_qsa_row_fp8_fnuz`).
//!
//! Three arms run on the same inputs; the whole output of every launch (with 0x7f guard words on
//! both sides that must survive) must be byte-equal:
//!
//! - REFERENCE: the `tensor_ops` entries `indexed_attention_attention_f32` /
//!   `indexed_attention_attention_fp8_fnuz`, launched directly;
//! - MODULE: the module entries launched directly. The module is loaded the way the product
//!   loads it (`lab_ensure_qsa_row_entry`): from the hipcc source, or, with
//!   `HIPFIRE_QWEN4_G942_QSA_ROW_IMAGE=/absolute/device.elf`, from that code object (the twin
//!   proof: same module, same entry names, same kernarg ABI);
//! - PRODUCT: `tensor_ops::indexed_attention_attention`, the public launcher the MTP head and the
//!   AR lanes call (it routes the parallel row kernel to the module on exact gfx942).
//!
//! A second REFERENCE launch must equal the first (the reference is repeatable). Shapes: the
//! Flash-Next QSA head geometry (24 query heads, 2 KV heads, head_dim 256; the MTP head's) and
//! other (heads, KV heads, head_dim) geometries for F32 (head_dim 64 / 98 / 128 / 384: the
//! float4 and scalar key loops and the channel-grid guard) and E4M3 FNUZ (the formats gfx942
//! admits: head_dim 256, even KV heads); `selected_len` 1 .. 8188 (the dynamic-LDS limit
//! `selected_len * 8 + 32 <= 64 KiB`) around the 24-row unroll and the 256-thread block
//! boundaries; capacities `selected_len`, `2 * selected_len + 7`, 4096 and 16384; selections of
//! random (duplicate-carrying) in-range tokens, selections with planted out-of-range ids
//! (-1, capacity, capacity + 1, i32::MAX, i32::MIN, -1000) and all-invalid selections (the
//! zero-denominator output).
//!
//! Planted controls the oracle MUST detect (module output differs from the clean reference):
//! one flipped byte in V, in K and in Q; the module run with the K and V bindings swapped; with
//! `selected_len - 1`; and the OTHER format's module entry on the same buffers. Exit 0 only when
//! every case matches and every control is detected. Prints PASS / FAIL counts.
//!
//! `qwen4_g942_qsa_row_oracle [--out FILE.json] [--seed U64] [--quick]`

use hip_bridge::KernargBlob;
use rdna_compute::tensor_ops::{
    indexed_attention_attention, lab_ensure_qsa_row_entry, IndexedAttentionAttention, QsaKvFormat,
};
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::json;

type Res<T> = Result<T, String>;

fn err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn f32(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Fmt {
    F32,
    Fnuz,
}

impl Fmt {
    fn name(self) -> &'static str {
        match self {
            Fmt::F32 => "f32",
            Fmt::Fnuz => "fp8_fnuz",
        }
    }
    fn reference(self) -> &'static str {
        match self {
            Fmt::F32 => "indexed_attention_attention_f32",
            Fmt::Fnuz => "indexed_attention_attention_fp8_fnuz",
        }
    }
    fn qsa(self) -> QsaKvFormat {
        match self {
            Fmt::F32 => QsaKvFormat::F32,
            Fmt::Fnuz => QsaKvFormat::Fp8Fnuz,
        }
    }
    fn other(self) -> Fmt {
        match self {
            Fmt::F32 => Fmt::Fnuz,
            Fmt::Fnuz => Fmt::F32,
        }
    }
    /// Bytes of one K (or V) token row.
    fn row_bytes(self, kv_heads: usize, head_dim: usize) -> usize {
        match self {
            Fmt::F32 => kv_heads * head_dim * 4,
            Fmt::Fnuz => kv_heads * (head_dim + 2),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Geo {
    nh: usize,
    nkv: usize,
    hd: usize,
}

/// Where the planted out-of-range ids go.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Sel {
    Valid,
    Mixed,
    AllInvalid,
}

const GUARD: usize = 64;
const LDS_MAX_SELECTED: usize = (64 * 1024 - 32) / 8;

struct Inputs {
    q: Vec<f32>,
    k: Vec<u8>,
    v: Vec<u8>,
}

fn kv_bytes(rng: &mut Rng, fmt: Fmt, geo: Geo, cap: usize, scale: f32) -> Vec<u8> {
    match fmt {
        Fmt::F32 => {
            let mut out = Vec::with_capacity(cap * geo.nkv * geo.hd * 4);
            for _ in 0..cap * geo.nkv * geo.hd {
                out.extend_from_slice(&(rng.f32() * scale).to_le_bytes());
            }
            out
        }
        Fmt::Fnuz => {
            let codes = geo.nkv * geo.hd;
            let mut out = Vec::with_capacity(cap * (codes + 2 * geo.nkv));
            for _ in 0..cap {
                for _ in 0..codes {
                    let r = rng.next();
                    let mag = ((r >> 8) % 0x50) as u8;
                    let sign = if mag != 0 && r & 1 == 1 { 0x80 } else { 0 };
                    out.push(mag | sign);
                }
                for _ in 0..geo.nkv {
                    // Positive finite halves, 0.03 .. 1.
                    let bits = 0x2800 + ((rng.next() >> 33) % 0x1400) as u16;
                    out.extend_from_slice(&bits.to_le_bytes());
                }
            }
            out
        }
    }
}

fn inputs(rng: &mut Rng, fmt: Fmt, geo: Geo, cap: usize, wide: bool) -> Inputs {
    let scale = if wide { 3.0 } else { 1.0 };
    let q = (0..geo.nh * geo.hd * 2).map(|_| rng.f32() * 2.0 * scale).collect();
    Inputs {
        q,
        k: kv_bytes(rng, fmt, geo, cap, scale),
        v: kv_bytes(rng, fmt, geo, cap, 1.0),
    }
}

/// A planted out-of-range token id.
fn planted(rng: &mut Rng, cap: usize) -> i32 {
    [-1, cap as i32, cap as i32 + 1, i32::MAX, i32::MIN, -1000][(rng.next() % 6) as usize]
}

fn selection(rng: &mut Rng, mode: Sel, len: usize, cap: usize) -> Vec<i32> {
    (0..len)
        .map(|i| {
            let plant = match mode {
                Sel::Valid => false,
                Sel::Mixed => rng.next() % 7 == 0 || (i == 0 && len > 1),
                Sel::AllInvalid => true,
            };
            if plant {
                planted(rng, cap)
            } else {
                (rng.next() % cap as u64) as i32
            }
        })
        .collect()
}

struct Dev {
    q: GpuTensor,
    k: GpuTensor,
    v: GpuTensor,
}

fn upload_kv(gpu: &mut Gpu, fmt: Fmt, bytes: &[u8]) -> Res<GpuTensor> {
    match fmt {
        Fmt::F32 => {
            let floats: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            gpu.upload_f32(&floats, &[floats.len()]).map_err(err)
        }
        Fmt::Fnuz => gpu.upload_raw(bytes, &[bytes.len()]).map_err(err),
    }
}

fn upload_dev(gpu: &mut Gpu, fmt: Fmt, host: &Inputs) -> Res<Dev> {
    let q = gpu.upload_f32(&host.q, &[host.q.len()]).map_err(err)?;
    let k = upload_kv(gpu, fmt, &host.k)?;
    let v = upload_kv(gpu, fmt, &host.v)?;
    Ok(Dev { q, k, v })
}

fn upload_sel(gpu: &Gpu, sel: &[i32]) -> Res<GpuTensor> {
    let bytes: Vec<u8> = sel.iter().flat_map(|t| t.to_le_bytes()).collect();
    gpu.upload_raw(&bytes, &[bytes.len()]).map_err(err)
}

/// A fresh output tensor with 0x7f guards; the body view is `[nh * hd]` F32.
fn out_buffer(gpu: &mut Gpu, geo: Geo) -> Res<(GpuTensor, GpuTensor)> {
    let n = geo.nh * geo.hd;
    let full = gpu.alloc_tensor(&[n + 2 * GUARD], DType::F32).map_err(err)?;
    gpu.hip
        .memset(&full.buf, 0x7f, full.byte_size())
        .map_err(err)?;
    let body = full.sub_offset(GUARD, n);
    Ok((full, body))
}

/// Download a guarded output: the body bytes, or an error if a guard word changed.
fn read_out(gpu: &Gpu, full: &GpuTensor, geo: Geo) -> Res<(Vec<u8>, bool)> {
    gpu.hip.device_synchronize().map_err(err)?;
    let bytes = gpu.download_raw_bytes(full).map_err(err)?;
    let n = geo.nh * geo.hd * 4;
    let guard_ok = bytes[..GUARD * 4].iter().all(|&b| b == 0x7f)
        && bytes[GUARD * 4 + n..].iter().all(|&b| b == 0x7f);
    Ok((bytes[GUARD * 4..GUARD * 4 + n].to_vec(), guard_ok))
}

struct Launch<'a> {
    symbol: &'a str,
    dev: &'a Dev,
    sel: &'a GpuTensor,
    geo: Geo,
    sel_len: usize,
    cap: usize,
    swap_kv: bool,
}

fn launch_direct(gpu: &mut Gpu, l: &Launch<'_>, out: &GpuTensor) -> Res<()> {
    let (k, v) = if l.swap_kv {
        (&l.dev.v, &l.dev.k)
    } else {
        (&l.dev.k, &l.dev.v)
    };
    let mut args = KernargBlob::new();
    for t in [&l.dev.q, k, v, l.sel, out] {
        args.push_ptr(t.buf.as_ptr());
    }
    for value in [l.geo.nh, l.geo.nkv, l.geo.hd, l.sel_len, l.cap] {
        args.push_i32(value as i32);
    }
    args.pad_to(16);
    gpu.launch_kernel_blob(
        l.symbol,
        [l.geo.nh as u32, l.geo.hd.div_ceil(256) as u32, 1],
        [256, 1, 1],
        (l.sel_len * 8) as u32,
        args.as_mut_slice(),
    )
    .map_err(err)
}

fn diff(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count() + a.len().abs_diff(b.len())
}

#[derive(Default)]
struct Tally {
    pass: usize,
    fail: usize,
    detected: usize,
    missed: usize,
    rows: Vec<serde_json::Value>,
}

/// The three arms on one prepared case; `ref_bytes` is the reference output.
#[allow(clippy::too_many_arguments)]
fn run_case(
    gpu: &mut Gpu,
    tally: &mut Tally,
    label: &str,
    fmt: Fmt,
    geo: Geo,
    dev: &Dev,
    sel: &[i32],
    cap: usize,
) -> Res<()> {
    let sel_dev = upload_sel(gpu, sel)?;
    let reference = fmt.reference();
    let module = lab_ensure_qsa_row_entry(gpu, reference).map_err(err)?;
    if module == reference {
        return Err(format!("{reference} was not routed to the gfx942 row module"));
    }
    ensure_reference(gpu, reference)?;
    let mk = |symbol| Launch {
        symbol,
        dev,
        sel: &sel_dev,
        geo,
        sel_len: sel.len(),
        cap,
        swap_kv: false,
    };
    let (r_full, r_body) = out_buffer(gpu, geo)?;
    launch_direct(gpu, &mk(reference), &r_body)?;
    let (r, r_guard) = read_out(gpu, &r_full, geo)?;
    let (r2_full, r2_body) = out_buffer(gpu, geo)?;
    launch_direct(gpu, &mk(reference), &r2_body)?;
    let (r2, r2_guard) = read_out(gpu, &r2_full, geo)?;
    let (m_full, m_body) = out_buffer(gpu, geo)?;
    launch_direct(gpu, &mk(module), &m_body)?;
    let (m, m_guard) = read_out(gpu, &m_full, geo)?;
    let (p_full, p_body) = out_buffer(gpu, geo)?;
    indexed_attention_attention(
        gpu,
        &IndexedAttentionAttention {
            q_with_gate: &dev.q,
            full_keys: &dev.k,
            full_values: &dev.v,
            selected: &sel_dev,
            output: &p_body,
            n_heads: geo.nh,
            n_kv_heads: geo.nkv,
            head_dim: geo.hd,
            selected_len: sel.len(),
            full_capacity: cap,
            end_position: cap,
            format: fmt.qsa(),
        },
    )
    .map_err(err)?;
    let (p, p_guard) = read_out(gpu, &p_full, geo)?;
    let (d_m, d_p, d_r2) = (diff(&m, &r), diff(&p, &r), diff(&r2, &r));
    let guards = r_guard && r2_guard && m_guard && p_guard;
    let ok = d_m == 0 && d_p == 0 && d_r2 == 0 && guards;
    if ok {
        tally.pass += 1;
    } else {
        tally.fail += 1;
        eprintln!(
            "FAIL {label}: module_vs_ref={d_m} product_vs_ref={d_p} ref_repeat={d_r2} guards_ok={guards}"
        );
    }
    if !ok || tally.rows.len() < 8 {
        tally.rows.push(json!({
            "case": label, "ok": ok, "module_vs_reference": d_m,
            "product_vs_reference": d_p, "reference_repeat": d_r2, "guards_ok": guards,
        }));
    }
    Ok(())
}

fn ensure_reference(gpu: &mut Gpu, kernel: &str) -> Res<()> {
    rdna_compute::tensor_ops::lab_ensure_tensor_ops_kernel(gpu, kernel).map_err(err)
}

const SEL_LENS: [usize; 26] = [
    1, 2, 3, 7, 23, 24, 25, 47, 48, 49, 50, 63, 64, 65, 127, 128, 255, 256, 257, 511, 512, 1000,
    2051, 4096, 8187, LDS_MAX_SELECTED,
];
const SEL_LENS_SHORT: [usize; 8] = [1, 23, 24, 25, 48, 257, 1000, 2051];

fn sweep(gpu: &mut Gpu, rng: &mut Rng, tally: &mut Tally, fmt: Fmt, geo: Geo, quick: bool) -> Res<()> {
    let main = geo.nh == 24 && geo.nkv == 2 && geo.hd == 256;
    let lens: Vec<usize> = if quick {
        vec![1, 24, 257, 2051]
    } else if main {
        SEL_LENS.to_vec()
    } else {
        SEL_LENS_SHORT.to_vec()
    };
    let modes: &[Sel] = &[Sel::Valid, Sel::Mixed, Sel::AllInvalid];
    // Fixed capacities: one K/V upload serves every selection length.
    let fixed: &[usize] = if main && !quick { &[4096, 16384] } else { &[4096] };
    for &cap in fixed {
        let host = inputs(rng, fmt, geo, cap, cap == 4096 && geo.hd != 98);
        let dev = upload_dev(gpu, fmt, &host)?;
        for &len in &lens {
            for &mode in modes {
                let sel = selection(rng, mode, len, cap);
                let label = format!("{} {geo:?} cap={cap} len={len} {mode:?}", fmt.name());
                run_case(gpu, tally, &label, fmt, geo, &dev, &sel, cap)?;
            }
        }
    }
    // Capacities tied to the selection length (the smallest cache the selection can address).
    for &len in &lens {
        if len > 4096 && !main {
            continue;
        }
        for cap in [len, 2 * len + 7] {
            let host = inputs(rng, fmt, geo, cap, false);
            let dev = upload_dev(gpu, fmt, &host)?;
            for &mode in modes {
                let sel = selection(rng, mode, len, cap);
                let label = format!("{} {geo:?} cap={cap} len={len} {mode:?}", fmt.name());
                run_case(gpu, tally, &label, fmt, geo, &dev, &sel, cap)?;
            }
        }
    }
    Ok(())
}

fn detect(tally: &mut Tally, name: &str, differing: usize) {
    if differing > 0 {
        tally.detected += 1;
    } else {
        tally.missed += 1;
        eprintln!("CONTROL MISSED {name}: no differing byte");
    }
    tally.rows.push(json!({"control": name, "differing_bytes": differing, "detected": differing > 0}));
}

/// The planted controls of one format on a small clean case (8 valid rows, capacity 64).
fn controls(gpu: &mut Gpu, rng: &mut Rng, tally: &mut Tally, fmt: Fmt) -> Res<()> {
    let geo = Geo { nh: 24, nkv: 2, hd: 256 };
    let (cap, len) = (64usize, 8usize);
    let host = inputs(rng, fmt, geo, cap, false);
    let sel: Vec<i32> = (0..len as i32).map(|i| i * 5 + 1).collect();
    let sel_dev = upload_sel(gpu, &sel)?;
    let clean = upload_dev(gpu, fmt, &host)?;
    let reference = fmt.reference();
    let module = lab_ensure_qsa_row_entry(gpu, reference).map_err(err)?;
    ensure_reference(gpu, reference)?;
    let run = |gpu: &mut Gpu, symbol: &str, dev: &Dev, len: usize, swap: bool| -> Res<Vec<u8>> {
        let (full, body) = out_buffer(gpu, geo)?;
        launch_direct(
            gpu,
            &Launch { symbol, dev, sel: &sel_dev, geo, sel_len: len, cap, swap_kv: swap },
            &body,
        )?;
        Ok(read_out(gpu, &full, geo)?.0)
    };
    let r = run(gpu, reference, &clean, len, false)?;
    // Sanity: the clean module equals the reference (also a counted case above).
    let m = run(gpu, module, &clean, len, false)?;
    if diff(&m, &r) != 0 {
        return Err(format!("{}: clean control case: module differs from reference", fmt.name()));
    }
    let row = fmt.row_bytes(geo.nkv, geo.hd);
    // One flipped byte: V of the first selected token (kv head 0, channel 0), K of it, Q of head 0.
    let first = sel[0] as usize;
    let flip = |bytes: &mut Vec<u8>, at: usize| bytes[at] ^= 0x40;
    for (name, which) in [("flip_v_byte", 'v'), ("flip_k_byte", 'k'), ("flip_q_byte", 'q')] {
        let mut h = Inputs { q: host.q.clone(), k: host.k.clone(), v: host.v.clone() };
        match (which, fmt) {
            ('v', Fmt::F32) => flip(&mut h.v, first * row + 3),
            ('k', Fmt::F32) => flip(&mut h.k, first * row + 3),
            ('v', Fmt::Fnuz) => flip(&mut h.v, first * row),
            ('k', Fmt::Fnuz) => flip(&mut h.k, first * row),
            _ => h.q[0] = f32::from_bits(h.q[0].to_bits() ^ 0x4000_0000),
        }
        let dev = upload_dev(gpu, fmt, &h)?;
        let got = run(gpu, module, &dev, len, false)?;
        detect(tally, &format!("{} {name}", fmt.name()), diff(&got, &r));
    }
    detect(tally, &format!("{} swap_kv_binding", fmt.name()), diff(&run(gpu, module, &clean, len, true)?, &r));
    detect(tally, &format!("{} selected_len_minus_1", fmt.name()), diff(&run(gpu, module, &clean, len - 1, false)?, &r));
    // The other format's module entry on the same buffers. F32 buffers hold more bytes than the
    // FNUZ rows read; FNUZ buffers are zero-padded to the F32 row size first.
    let wrong_symbol = lab_ensure_qsa_row_entry(gpu, fmt.other().reference()).map_err(err)?;
    let wrong_dev = match fmt {
        Fmt::F32 => upload_dev(gpu, fmt, &host)?,
        Fmt::Fnuz => {
            let pad = |bytes: &[u8]| -> Vec<u8> {
                let mut v = bytes.to_vec();
                v.resize(cap * Fmt::F32.row_bytes(geo.nkv, geo.hd), 0);
                v
            };
            Dev {
                q: gpu.upload_f32(&host.q, &[host.q.len()]).map_err(err)?,
                k: upload_kv(gpu, Fmt::Fnuz, &pad(&host.k))?,
                v: upload_kv(gpu, Fmt::Fnuz, &pad(&host.v))?,
            }
        }
    };
    let got = run(gpu, wrong_symbol, &wrong_dev, len, false)?;
    detect(tally, &format!("{} wrong_entry({wrong_symbol})", fmt.name()), diff(&got, &r));
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let flag = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let quick = args.iter().any(|a| a == "--quick");
    let seed = flag("--seed")
        .map(|s| s.parse::<u64>().expect("--seed U64"))
        .unwrap_or(0x9E3779B97F4A7C15);
    let mut gpu = Gpu::init().expect("gpu init");
    if gpu.try_gfx942().is_none() {
        eprintln!("skip: needs an exact gfx942 device");
        std::process::exit(0);
    }
    let image = std::env::var("HIPFIRE_QWEN4_G942_QSA_ROW_IMAGE").ok().filter(|p| !p.is_empty());
    println!(
        "module arm: {}",
        image
            .as_deref()
            .map_or("hipcc source (module qwen4_g942_qsa_row_attention)".to_owned(), |p| {
                format!("code object {p}")
            })
    );
    let mut rng = Rng(seed);
    let mut tally = Tally::default();
    let geoms_f32 = [
        Geo { nh: 24, nkv: 2, hd: 256 },
        Geo { nh: 24, nkv: 2, hd: 128 },
        Geo { nh: 8, nkv: 1, hd: 256 },
        Geo { nh: 6, nkv: 3, hd: 98 },
        Geo { nh: 16, nkv: 4, hd: 64 },
        Geo { nh: 4, nkv: 2, hd: 384 },
    ];
    let geoms_fnuz = [
        Geo { nh: 24, nkv: 2, hd: 256 },
        Geo { nh: 8, nkv: 2, hd: 256 },
        Geo { nh: 12, nkv: 4, hd: 256 },
        Geo { nh: 16, nkv: 2, hd: 256 },
    ];
    for (fmt, geoms) in [(Fmt::F32, &geoms_f32[..]), (Fmt::Fnuz, &geoms_fnuz[..])] {
        for &geo in geoms {
            if quick && !(geo.nh == 24 && geo.nkv == 2 && geo.hd == 256) {
                continue;
            }
            if let Err(e) = sweep(&mut gpu, &mut rng, &mut tally, fmt, geo, quick) {
                eprintln!("{} {geo:?}: {e}", fmt.name());
                std::process::exit(1);
            }
            println!("{} {geo:?}: cumulative PASS {} FAIL {}", fmt.name(), tally.pass, tally.fail);
        }
    }
    for fmt in [Fmt::F32, Fmt::Fnuz] {
        if let Err(e) = controls(&mut gpu, &mut rng, &mut tally, fmt) {
            eprintln!("controls {}: {e}", fmt.name());
            std::process::exit(1);
        }
    }
    println!(
        "RESULT cases PASS {} FAIL {} | controls DETECTED {} MISSED {}",
        tally.pass, tally.fail, tally.detected, tally.missed
    );
    if let Some(path) = flag("--out") {
        let report = json!({
            "oracle": "qwen4_g942_qsa_row_oracle", "seed": seed, "quick": quick,
            "image": image, "cases_pass": tally.pass, "cases_fail": tally.fail,
            "controls_detected": tally.detected, "controls_missed": tally.missed,
            "rows": tally.rows,
        });
        std::fs::write(&path, serde_json::to_string_pretty(&report).unwrap()).expect("write --out");
    }
    let ok = tally.fail == 0 && tally.missed == 0 && tally.pass > 0 && tally.detected > 0;
    println!("{}", if ok { "PASS" } else { "FAIL" });
    std::process::exit(if ok { 0 } else { 1 });
}
