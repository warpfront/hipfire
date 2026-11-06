// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Hardware byte oracle, not a performance benchmark. Run only inside a one-card
//! gfx1201 fabric correctness hold: `hc8_byte_oracle RECEIPT.json`.
//! Compares every live word against an independently emitted row-major decoded
//! BF16 WMMA reference and the compressed n1 singleton for that input row. Every
//! rows=1..64 and every compatible NT is exercised, including unused capacity.
//! Reference construction is shared with the CPU oracle, not the kernel generator.

#[path = "../tests/support/hc8_reference.rs"]
mod reference;

use hip_bridge::{DeviceBuffer, Function, HipRuntime};
use peacemaker_isa::kernels::hc8_wmma_gfx1201::{self as hc8, Fmt, Geo};
use peacemaker_isa::{native, Arch};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::process::ExitCode;

const GUARD: usize = 128;
const SENT: u8 = 0xa5;

struct Card(HipRuntime);
impl Card {
    fn upload(&self, bytes: &[u8]) -> Result<DeviceBuffer, String> {
        let d = self.0.malloc(bytes.len()).map_err(|e| e.to_string())?;
        self.0.memcpy_htod(&d, bytes).map_err(|e| e.to_string())?;
        Ok(d)
    }
    fn download(&self, d: &DeviceBuffer, n: usize) -> Result<Vec<u8>, String> {
        let mut bytes = vec![0; n];
        self.0.memcpy_dtoh(&mut bytes, d).map_err(|e| e.to_string())?;
        Ok(bytes)
    }
    fn launch(&self, f: &Function, geo: Geo, reference: bool, blob: &mut [u8]) -> Result<(), String> {
        let m = u32::from_le_bytes(blob[32..36].try_into().unwrap());
        let (grid, block) = if reference {
            (reference::reference_grid(m), reference::REFERENCE_BLOCK)
        } else {
            (geo.grid(m), geo.block())
        };
        // SAFETY: the complete 48-byte ABI holds live, guarded allocations. The
        // shape matrix satisfies M%(16*MW)==0 and K%(32*KW)==0 for each entry.
        unsafe {
            self.0.launch_kernel_blob(f, [grid, 1, 1], [block, 1, 1], 0, None, blob)
        }.map_err(|e| e.to_string())?;
        self.0.device_synchronize().map_err(|e| e.to_string())
    }
}

fn guarded(data: &[u8]) -> Vec<u8> {
    let mut bytes = vec![SENT; GUARD];
    bytes.extend_from_slice(data);
    bytes.resize(bytes.len() + GUARD, SENT);
    bytes
}
fn ptr(d: &DeviceBuffer) -> u64 { d.as_ptr() as u64 + GUARD as u64 }
fn args(q: u64, s: u64, x: u64, y: u64, m: usize, k: usize, rows: usize) -> Vec<u8> {
    let mut blob = Vec::with_capacity(48);
    for p in [q, s, x, y] { blob.extend(p.to_le_bytes()); }
    for v in [m, k, rows] { blob.extend((v as u32).to_le_bytes()); }
    blob.extend(0.75f32.to_le_bytes());
    blob
}
fn compare(got: &[u8], want: &[u8], context: &str) -> Result<usize, String> {
    if got.len() != want.len() { return Err(format!("{context}: unequal extents")); }
    if let Some((i, (g, w))) = got.chunks_exact(4).zip(want.chunks_exact(4)).enumerate().find(|(_, (g,w))| g != w) {
        return Err(format!("{context}: word {i}: got {:08x}, want {:08x}",
            u32::from_le_bytes(g.try_into().unwrap()), u32::from_le_bytes(w.try_into().unwrap())));
    }
    Ok(got.len())
}
fn check_extent(bytes: &[u8], live: usize, context: &str) -> Result<(), String> {
    if bytes[..GUARD].iter().chain(bytes[GUARD + live..].iter()).any(|&b| b != SENT) {
        return Err(format!("{context}: guard/dead-column write"));
    }
    Ok(())
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
fn data(fmt: Fmt, m: usize, k: usize, kind: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut rng = Rng(0x8b16_5eed_ba5e_0001 ^ (m as u64) << 32 ^ k as u64 ^ kind as u64);
    let weights = (0..m*k).map(|i| {
        let mut q = if kind == 0 { rng.next() as u8 } else { i as u8 };
        if matches!(fmt, Fmt::E4M3) && q & 0x7f == 0x7f && kind != 2 { q &= 0xfe; }
        q
    }).collect();
    let edge = [0u32, 0x8000_0000, 0x0001_0000, 0x8001_0000, 0x0080_0000,
        0x3f80_0000, 0xbf80_0000, 0x3f00_0000, 0x4000_0000, 0x7f7f_0000,
        0xff7f_0000, 0x3f81_0000, 0x3380_0000];
    let x = (0..64*k).flat_map(|i| {
        let bits = if kind == 0 {
            (rng.next() & 0x807f_0000) | ((120 + rng.next() % 14) << 23)
        } else if kind == 2 && i % k == (i / k) % k {
            [0x7f80_0000, 0xff80_0000, 0x7fc1_0000, 0xffa1_0000][(i / k) % 4]
        } else { edge[(i * 7 + i / k) % edge.len()] };
        bits.to_le_bytes()
    }).collect();
    let scales = (0..m).flat_map(|i| {
        [1.0f32, -0.5, 0.03125, 2.0, 0.0, -0.0, 0.75][i % 7].to_le_bytes()
    }).collect();
    (weights, scales, x)
}

struct Input {
    name: String,
    fmt: Fmt,
    m: usize,
    k: usize,
    w: Vec<u8>,
    scales: Vec<u8>,
    x: Vec<u8>,
}
fn fmt_name(fmt: Fmt) -> &'static str {
    if matches!(fmt, Fmt::E4M3) { "e4m3" } else { "i8" }
}
fn load_fixtures(paths: &[String]) -> Result<Vec<Input>, String> {
    paths.iter().map(|path| {
        let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
        let v: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| format!("{path}: {e}"))?;
        let number = |field: &str| v[field].as_u64().map(|n| n as usize).ok_or_else(|| format!("{path}: missing {field}"));
        let payload = |field: &str| -> Result<Vec<u8>, String> {
            let p = v[field].as_str().ok_or_else(|| format!("{path}: missing {field}"))?;
            std::fs::read(p).map_err(|e| format!("{p}: {e}"))
        };
        let fmt = match v["fmt"].as_str() {
            Some("e4m3") => Fmt::E4M3,
            Some("i8") => Fmt::I8,
            _ => return Err(format!("{path}: fmt must be e4m3 or i8")),
        };
        let (m,k) = (number("m")?, number("k")?);
        let (w,scales,x) = (payload("q")?,payload("scales")?,payload("x")?);
        if m == 0 || k == 0 || m%16 != 0 || k%32 != 0 || w.len()!=m*k || scales.len()!=m*4 || x.len()!=64*k*4 {
            return Err(format!("{path}: fixture extents must be q[M,K], scales[M], x[64,K]"));
        }
        if x.chunks_exact(4).any(|b| b[0]!=0 || b[1]!=0) {
            return Err(format!("{path}: x is not BF16-exact"));
        }
        Ok(Input {name:format!("real:{path}"),fmt,m,k,w,scales,x})
    }).collect()
}

fn run(out: &str, fixture_paths: &[String]) -> Result<(), String> {
    if std::env::var_os("FABRIC_JOB").is_none() {
        return Err("run through fabric-run.sh --class correct --card <one gfx1201 card>".into());
    }
    let rt = HipRuntime::load().map_err(|e| e.to_string())?;
    rt.set_device(0).map_err(|e| e.to_string())?;
    let arch = rt.get_arch(0).map_err(|e| e.to_string())?;
    if arch.split(':').next() != Some("gfx1201") { return Err(format!("requires exact gfx1201, got {arch}")); }
    if rt.device_count().map_err(|e| e.to_string())? != 1 { return Err("requires exactly one visible card".into()); }
    let card = Card(rt);
    let fixtures = load_fixtures(fixture_paths)?;
    let (_, text, _) = hc8::emit_module()?;
    let elf = native::assemble(&text, Arch::Gfx1201)?;
    let module = card.0.module_load_data(&elf).map_err(|e| e.to_string())?;
    let mut cases = Vec::new();
    let mut compared_bytes = 0usize;
    let mut launches = 0usize;
    for fmt in [Fmt::E4M3, Fmt::I8] {
        for (mw, kw) in [(1,8), (1,16), (4,1), (8,1), (2,4)] {
            let small_m = 16 * mw as usize;
            let small_k = 32 * kw as usize * 2;
            for act in [false, true] {
                let base = Geo { fmt, mw, kw, nt: 1, act };
                // The public symbols list is authoritative if the permitted
                // reduced i8 module was selected by the generator.
                if !hc8::symbols().contains(&base.symbol()) { continue; }
                let solo = card.0.module_get_function(&module, &base.symbol()).map_err(|e| e.to_string())?;
                let mut functions = Vec::new();
                for nt in 1..=4 {
                    let geo = Geo { nt, ..base };
                    let pm = card.0.module_get_function(&module, &geo.symbol()).map_err(|e| e.to_string())?;
                    let reference = reference::emit_reference(geo)?;
                    let re = native::assemble(&reference.s_text, Arch::Gfx1201)?;
                    let rm = card.0.module_load_data(&re).map_err(|e| e.to_string())?;
                    let rf = card.0.module_get_function(&rm, &reference::reference_symbol(geo)).map_err(|e| e.to_string())?;
                    functions.push((geo, pm, rm, rf));
                }
                let synthetic: Vec<Input> = (0..3).map(|kind| {
                    let (w,scales,x) = data(fmt,small_m,small_k,kind);
                    Input {name:["random_finite","finite_edges","nan_inf_edges"][kind].into(),
                        fmt,m:small_m,k:small_k,w,scales,x}
                }).collect();
                for case in synthetic.iter().chain(fixtures.iter().filter(|c|
                    fmt_name(c.fmt)==fmt_name(fmt) && c.m%(16*mw as usize)==0 && c.k%(32*kw as usize)==0)) {
                    let (m,k) = (case.m,case.k);
                    let (w,scales,x) = (&case.w,&case.scales,&case.x);
                    let kind = &case.name;
                    let q = guarded(&hc8::swizzle_hc8(&w, m, k));
                    let decoded: Vec<u8> = reference::decode_weights(fmt, &w).iter().flat_map(|v| v.to_le_bytes()).collect();
                    let decoded = guarded(&decoded);
                    let scales = guarded(&scales);
                    let x = guarded(&x);
                    let dq = card.upload(&q)?;
                    let dw = card.upload(&decoded)?;
                    let ds = card.upload(&scales)?;
                    let dx = card.upload(&x)?;
                    let poison = vec![SENT; 2 * GUARD + 64*m*4];
                    let dy = card.upload(&poison)?;
                    let dr = card.upload(&poison)?;
                    let mut singleton = Vec::with_capacity(64*m*4);
                    for row in 0..64 {
                        card.0.memcpy_htod(&dy, &poison).map_err(|e| e.to_string())?;
                        let mut blob = args(ptr(&dq), ptr(&ds), ptr(&dx) + (row*k*4) as u64, ptr(&dy), m, k, 1);
                        card.launch(&solo, base, false, &mut blob)?;
                        launches += 1;
                        let bytes = card.download(&dy, poison.len())?;
                        check_extent(&bytes, m*4, "singleton")?;
                        singleton.extend_from_slice(&bytes[GUARD..GUARD+m*4]);
                    }
                    let mut comparisons = 0;
                    for (geo, pm, _rm, rf) in &functions {
                        for rows in 1..=geo.max_rows() as usize {
                            for d in [&dy, &dr] { card.0.memcpy_htod(d, &poison).map_err(|e| e.to_string())?; }
                            let mut blob = args(ptr(&dq), ptr(&ds), ptr(&dx), ptr(&dy), m, k, rows);
                            card.launch(pm, *geo, false, &mut blob)?;
                            let mut reference_blob = args(ptr(&dw), ptr(&ds), ptr(&dx), ptr(&dr), m, k, rows);
                            card.launch(rf, *geo, true, &mut reference_blob)?;
                            launches += 2;
                            let got = card.download(&dy, poison.len())?;
                            let want = card.download(&dr, poison.len())?;
                            let context = format!("{} kind={kind} rows={rows}", geo.symbol());
                            check_extent(&got, rows*m*4, &context)?;
                            check_extent(&want, rows*m*4, &format!("reference {context}"))?;
                            compared_bytes += compare(&got, &want, &format!("dequant-BF16 WMMA {context}"))?;
                            compared_bytes += compare(&got[GUARD..GUARD+rows*m*4], &singleton[..rows*m*4], &format!("singleton {context}"))?;
                            comparisons += 1;
                        }
                    }
                    for (d, bytes) in [(&dq, &q), (&dw, &decoded), (&ds, &scales), (&dx, &x)] {
                        if card.download(d, bytes.len())? != *bytes { return Err(format!("{} kind={kind}: readonly input changed", base.symbol())); }
                    }
                    cases.push(json!({"fmt":fmt_name(fmt),"mw":mw,"kw":kw,"act":act,"m":m,"k":k,"kind":kind,
                        "q_rowmajor_sha256":format!("{:x}",Sha256::digest(w)),
                        "x_sha256":format!("{:x}",Sha256::digest(&case.x)),
                        "scales_sha256":format!("{:x}",Sha256::digest(&case.scales)),
                        "comparisons":comparisons,"rows":"every 1..64, every compatible NT","pass":true}));
                    println!("{} kind={kind}: {comparisons} comparisons PASS", base.symbol());
                    for d in [dq,dw,ds,dx,dy,dr] { card.0.free(d).map_err(|e| e.to_string())?; }
                }
            }
        }
    }
    let receipt = json!({"schema":"hc8-byte-oracle-v1","pass":true,"arch":arch,
        "fabric_job":std::env::var("FABRIC_JOB").ok(),"elf_sha256":format!("{:x}",Sha256::digest(&elf)),
        "reference":"independent row-major dequantized BF16 WMMA; identical fixed KW fold; no compressed decoder",
        "compared_bytes":compared_bytes,"launches":launches,"cases":cases});
    std::fs::write(out, serde_json::to_string_pretty(&receipt).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    println!("PASS: {compared_bytes} bytes, {launches} launches; receipt {out}");
    Ok(())
}
fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(out) = args.first() else { eprintln!("usage: hc8_byte_oracle RECEIPT.json [FIXTURE.json ...]"); return ExitCode::from(2); };
    match run(out, &args[1..]) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => { eprintln!("FAIL: {e}"); ExitCode::FAILURE }
    }
}
