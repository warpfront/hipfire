// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Hardware byte oracle, not a performance benchmark. Run only inside a one-card
//! gfx1201 fabric correctness hold: `hc8_byte_oracle RECEIPT.json`.
//! Compares every live word against an independently emitted row-major decoded
//! BF16 WMMA reference and the compressed n1 singleton for that input row. Every
//! rows=1..64 and every compatible NT is exercised, including unused capacity.
//! Reference construction is shared with the CPU oracle, not the kernel generator.
//!
//! `--read-up --projected-object OBJECT.co RECEIPT.json [FIXTURE.json ...]`
//! compares each fused `hc8_read_up_e4m3_n{1..4}` entry bitwise against the
//! original `hc8_gemm_e4m3_mw4_kw1_nN_set` output fed to the product
//! `hyper_read_projected_f32` loaded from a pre-compiled code object (the cached
//! hipcc anchor; nothing is compiled here), plus 64 singleton rows for row
//! invariance and sentinel dead-store checks. A mismatch is counted into the
//! receipt, not fatal; the process still exits non-zero.

#[path = "../tests/support/hc8_reference.rs"]
mod reference;
#[path = "../tests/support/s_bfm_probe.rs"]
mod s_bfm_probe;

use hip_bridge::{DeviceBuffer, Function, HipRuntime};
use peacemaker_isa::kernels::hc8_wmma_gfx1201::{self as hc8, Fmt, Geo};
use peacemaker_isa::{native, Arch};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::process::ExitCode;

const GUARD: usize = 128;
const SENT: u8 = 0xa5;
const BUNDLE: &[u8] = include_bytes!("../../../kernels/hc8_wmma_pm_gfx1201.hxaco");
const BUNDLE_SHA: &str = include_str!("../../../kernels/hc8_wmma_pm_gfx1201.hxaco.sha256");

struct Card(HipRuntime);
impl Card {
    fn upload(&self, bytes: &[u8]) -> Result<DeviceBuffer, String> {
        let d = self.0.malloc(bytes.len()).map_err(|e| e.to_string())?;
        self.0.memcpy_htod(&d, bytes).map_err(|e| e.to_string())?;
        Ok(d)
    }
    fn download(&self, d: &DeviceBuffer, n: usize) -> Result<Vec<u8>, String> {
        let mut bytes = vec![0; n];
        self.0
            .memcpy_dtoh(&mut bytes, d)
            .map_err(|e| e.to_string())?;
        Ok(bytes)
    }
    fn launch(
        &self,
        f: &Function,
        geo: Geo,
        reference: bool,
        blob: &mut [u8],
    ) -> Result<(), String> {
        let m = u32::from_le_bytes(blob[32..36].try_into().unwrap());
        let (grid, block) = if reference {
            let rows = u32::from_le_bytes(blob[40..44].try_into().unwrap());
            let (gx, gy) = reference::reference_grid(m, rows);
            ([gx, gy, 1], reference::REFERENCE_BLOCK)
        } else {
            ([geo.grid(m), 1, 1], geo.block())
        };
        // SAFETY: the complete 48-byte ABI holds live, guarded allocations. The
        // shape matrix satisfies M%(16*MW)==0 and K%(32*KW)==0 for each entry.
        unsafe {
            self.0
                .launch_kernel_blob(f, grid, [block, 1, 1], 0, None, blob)
        }
        .map_err(|e| e.to_string())?;
        self.0.device_synchronize().map_err(|e| e.to_string())
    }
}

fn guarded(data: &[u8]) -> Vec<u8> {
    let mut bytes = vec![SENT; GUARD];
    bytes.extend_from_slice(data);
    bytes.resize(bytes.len() + GUARD, SENT);
    bytes
}
fn ptr(d: &DeviceBuffer) -> u64 {
    d.as_ptr() as u64 + GUARD as u64
}
fn args(q: u64, s: u64, x: u64, y: u64, m: usize, k: usize, rows: usize) -> Vec<u8> {
    let mut blob = Vec::with_capacity(48);
    for p in [q, s, x, y] {
        blob.extend(p.to_le_bytes());
    }
    for v in [m, k, rows] {
        blob.extend((v as u32).to_le_bytes());
    }
    blob.extend(0.75f32.to_le_bytes());
    blob
}
fn compare(got: &[u8], want: &[u8], context: &str) -> Result<usize, String> {
    if got.len() != want.len() {
        return Err(format!("{context}: unequal extents"));
    }
    if let Some((i, (g, w))) = got
        .chunks_exact(4)
        .zip(want.chunks_exact(4))
        .enumerate()
        .find(|(_, (g, w))| g != w)
    {
        return Err(format!(
            "{context}: word {i}: got {:08x}, want {:08x}",
            u32::from_le_bytes(g.try_into().unwrap()),
            u32::from_le_bytes(w.try_into().unwrap())
        ));
    }
    Ok(got.len())
}
fn check_extent(bytes: &[u8], live: usize, context: &str) -> Result<(), String> {
    if bytes[..GUARD]
        .iter()
        .chain(bytes[GUARD + live..].iter())
        .any(|&b| b != SENT)
    {
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
    let weights = (0..m * k)
        .map(|i| {
            let mut q = if kind == 0 { rng.next() as u8 } else { i as u8 };
            if matches!(fmt, Fmt::E4M3) && q & 0x7f == 0x7f && kind != 2 {
                q &= 0xfe;
            }
            q
        })
        .collect();
    let edge = [
        0u32,
        0x8000_0000,
        0x0001_0000,
        0x8001_0000,
        0x0080_0000,
        0x3f80_0000,
        0xbf80_0000,
        0x3f00_0000,
        0x4000_0000,
        0x7f7f_0000,
        0xff7f_0000,
        0x3f81_0000,
        0x3380_0000,
    ];
    let x = (0..64 * k)
        .flat_map(|i| {
            let bits = if kind == 0 {
                (rng.next() & 0x807f_0000) | ((120 + rng.next() % 14) << 23)
            } else if kind == 2 && i % k == (i / k) % k {
                [0x7f80_0000, 0xff80_0000, 0x7fc1_0000, 0xffa1_0000][(i / k) % 4]
            } else {
                edge[(i * 7 + i / k) % edge.len()]
            };
            bits.to_le_bytes()
        })
        .collect();
    let scales = (0..m)
        .flat_map(|i| [1.0f32, -0.5, 0.03125, 2.0, 0.0, -0.0, 0.75][i % 7].to_le_bytes())
        .collect();
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
    if matches!(fmt, Fmt::E4M3) {
        "e4m3"
    } else {
        "i8"
    }
}
fn load_fixtures(paths: &[String]) -> Result<Vec<Input>, String> {
    paths
        .iter()
        .map(|path| {
            let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
            let v: serde_json::Value =
                serde_json::from_slice(&bytes).map_err(|e| format!("{path}: {e}"))?;
            let number = |field: &str| {
                v[field]
                    .as_u64()
                    .map(|n| n as usize)
                    .ok_or_else(|| format!("{path}: missing {field}"))
            };
            let payload = |field: &str| -> Result<Vec<u8>, String> {
                let p = v[field]
                    .as_str()
                    .ok_or_else(|| format!("{path}: missing {field}"))?;
                std::fs::read(p).map_err(|e| format!("{p}: {e}"))
            };
            let fmt = match v["fmt"].as_str() {
                Some("e4m3") => Fmt::E4M3,
                Some("i8") => Fmt::I8,
                _ => return Err(format!("{path}: fmt must be e4m3 or i8")),
            };
            let (m, k) = (number("m")?, number("k")?);
            let (w, scales, x) = (payload("q")?, payload("scales")?, payload("x")?);
            if m == 0
                || k == 0
                || m % 16 != 0
                || k % 32 != 0
                || w.len() != m * k
                || scales.len() != m * 4
                || x.len() != 64 * k * 4
            {
                return Err(format!(
                    "{path}: fixture extents must be q[M,K], scales[M], x[64,K]"
                ));
            }
            if x.chunks_exact(4).any(|b| b[0] != 0 || b[1] != 0) {
                return Err(format!("{path}: x is not BF16-exact"));
            }
            Ok(Input {
                name: format!("real:{path}"),
                fmt,
                m,
                k,
                w,
                scales,
                x,
            })
        })
        .collect()
}

fn scalar_mask_oracle(card: &Card) -> Result<(serde_json::Value, usize), String> {
    let emitted = s_bfm_probe::emit_probe()?;
    let elf = native::assemble(&emitted.s_text, Arch::Gfx1201)?;
    let module = card.0.module_load_data(&elf).map_err(|e| e.to_string())?;
    let function = card
        .0
        .module_get_function(&module, s_bfm_probe::SYMBOL)
        .map_err(|e| e.to_string())?;
    let records = s_bfm_probe::inputs();
    let inputs = guarded(
        &records
            .iter()
            .flatten()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    let mut expected = vec![SENT; GUARD];
    expected.extend(
        records
            .iter()
            .flat_map(|&i| s_bfm_probe::expected(i))
            .flat_map(|v| v.to_le_bytes()),
    );
    expected.resize(expected.len() + GUARD, SENT);
    let din = card.upload(&inputs)?;
    let dout = card.upload(&vec![SENT; expected.len()])?;
    let mut blob = Vec::with_capacity(s_bfm_probe::KERNARG_BYTES as usize);
    for p in [ptr(&din), ptr(&dout)] {
        blob.extend(p.to_le_bytes());
    }
    // SAFETY: the 16-byte probe ABI contains live guarded input/output records;
    // one workgroup handles each input and only its lane 0 stores two words.
    unsafe {
        card.0.launch_kernel_blob(
            &function,
            [records.len() as u32, 1, 1],
            [s_bfm_probe::BLOCK, 1, 1],
            0,
            None,
            &mut blob,
        )
    }
    .map_err(|e| e.to_string())?;
    card.0.device_synchronize().map_err(|e| e.to_string())?;
    let got = card.download(&dout, expected.len())?;
    let compared = compare(&got, &expected, "s_bfm_b32 result/SCC probe")?;
    if card.download(&din, inputs.len())? != inputs {
        return Err("s_bfm_b32 probe changed its inputs".into());
    }
    for d in [din, dout] {
        card.0.free(d).map_err(|e| e.to_string())?;
    }
    Ok((
        json!({"symbol":s_bfm_probe::SYMBOL,"records":records.len(),"words":records.len()*2,
        "elf_sha256":format!("{:x}",Sha256::digest(&elf)),"pass":true}),
        compared,
    ))
}

fn dequant_oracle(
    card: &Card,
    functions: [&Function; 2],
    fixtures: &[Input],
) -> Result<(Vec<serde_json::Value>, usize), String> {
    let synthetic: Vec<Input> = [(16, 32), (32, 64), (64, 320)]
        .into_iter()
        .map(|(m, k)| {
            let w = (0..m * k)
                .map(|i| {
                    let q = i as u8;
                    if q & 0x7f == 0x7f {
                        q & 0xfe
                    } else {
                        q
                    }
                })
                .collect();
            let scales = (0..m)
                .flat_map(|r| {
                    let exponent = [-24i32, -16, -8, -1, 0, 8, 16][r % 7];
                    f32::from_bits(((127 + exponent) as u32) << 23).to_le_bytes()
                })
                .collect();
            Input {
                name: "all-finite-codes-and-half-edges".into(),
                fmt: Fmt::E4M3,
                m,
                k,
                w,
                scales,
                x: Vec::new(),
            }
        })
        .collect();
    let mut cases = Vec::new();
    let mut compared = 0;
    for c in synthetic
        .iter()
        .chain(fixtures.iter().filter(|c| c.fmt == Fmt::E4M3))
    {
        let q = guarded(&hc8::swizzle_hc8(&c.w, c.m, c.k));
        let scales = guarded(&c.scales);
        let dq = card.upload(&q)?;
        let ds = card.upload(&scales)?;
        let poison = vec![SENT; 2 * GUARD + c.m * c.k * 2];
        let dy = card.upload(&poison)?;
        let decoded = reference::decode_weights(Fmt::E4M3, &c.w);
        for (f, half) in functions.into_iter().zip([false, true]) {
            let mut expected = poison.clone();
            for (i, &bits) in decoded.iter().enumerate() {
                let at = (i / c.k) * 4;
                let s = f32::from_le_bytes(c.scales[at..at + 4].try_into().unwrap());
                let v = f32::from_bits(u32::from(bits) << 16) * s;
                let word = if half {
                    peacemaker_ir::emu::convert::f32_to_f16(v.to_bits())
                } else {
                    peacemaker_ir::emu::convert::f32_to_bf16(v.to_bits())
                };
                expected[GUARD + 2 * i..GUARD + 2 * i + 2].copy_from_slice(&word.to_le_bytes());
            }
            card.0
                .memcpy_htod(&dy, &poison)
                .map_err(|e| e.to_string())?;
            let mut blob = Vec::with_capacity(32);
            for p in [ptr(&dq), ptr(&ds), ptr(&dy)] {
                blob.extend(p.to_le_bytes());
            }
            blob.extend((c.m as u32).to_le_bytes());
            blob.extend((c.k as u32).to_le_bytes());
            // SAFETY: exact 32-byte dequant ABI; guarded allocations above cover
            // every input and output element, grid uses the fixed block of 256.
            unsafe {
                card.0.launch_kernel_blob(
                    f,
                    [(c.m * c.k).div_ceil(256) as u32, 1, 1],
                    [256, 1, 1],
                    0,
                    None,
                    &mut blob,
                )
            }
            .map_err(|e| e.to_string())?;
            card.0.device_synchronize().map_err(|e| e.to_string())?;
            let got = card.download(&dy, poison.len())?;
            let symbol = if half {
                "hc8_dequant_f16"
            } else {
                "hc8_dequant_bf16"
            };
            compared += compare(&got, &expected, &format!("{symbol} {}", c.name))?;
            for (d, bytes) in [(&dq, &q), (&ds, &scales)] {
                if card.download(d, bytes.len())? != *bytes {
                    return Err(format!("{symbol}: readonly input changed"));
                }
            }
            cases.push(json!({"symbol":symbol,"m":c.m,"k":c.k,"kind":c.name,"bytes":got.len(),"pass":true}));
        }
        for d in [dq, ds, dy] {
            card.0.free(d).map_err(|e| e.to_string())?;
        }
    }
    Ok((cases, compared))
}

fn run(out: &str, fixture_paths: &[String]) -> Result<(), String> {
    if std::env::var_os("FABRIC_JOB").is_none() {
        return Err("run through fabric-run.sh --class correct --card <one gfx1201 card>".into());
    }
    let rt = HipRuntime::load().map_err(|e| e.to_string())?;
    rt.set_device(0).map_err(|e| e.to_string())?;
    let arch = rt.get_arch(0).map_err(|e| e.to_string())?;
    if arch.split(':').next() != Some("gfx1201") {
        return Err(format!("requires exact gfx1201, got {arch}"));
    }
    if rt.device_count().map_err(|e| e.to_string())? != 1 {
        return Err("requires exactly one visible card".into());
    }
    let card = Card(rt);
    let fixtures = load_fixtures(fixture_paths)?;
    let (_, text, _) = hc8::emit_module()?;
    let elf = native::assemble(&text, Arch::Gfx1201)?;
    let bundle_sha = format!("{:x}", Sha256::digest(BUNDLE));
    if bundle_sha != BUNDLE_SHA.trim()
        || native::bundle(&elf, Arch::Gfx1201, native::DEFAULT_HOST_TARGET) != BUNDLE
    {
        return Err("committed bundle/sha differs from fresh generator output".into());
    }
    if compare(&0u32.to_le_bytes(), &1u32.to_le_bytes(), "negative control").is_ok() {
        return Err("byte comparator missed a one-bit negative control".into());
    }
    let module = card.0.module_load_data(&elf).map_err(|e| e.to_string())?;
    let mut cases = Vec::new();
    let (scalar_mask, mut compared_bytes) = scalar_mask_oracle(&card)?;
    let mut launches = 1usize;
    for fmt in [Fmt::E4M3, Fmt::I8] {
        for (mw, kw) in [(1, 8), (1, 16), (4, 1), (8, 1), (2, 4)] {
            let small_m = 16 * mw as usize;
            let small_k = 32 * kw as usize * 2;
            for act in [false, true] {
                let base = Geo {
                    fmt,
                    mw,
                    kw,
                    nt: 1,
                    act,
                };
                // The public symbols list is authoritative if the permitted
                // reduced i8 module was selected by the generator.
                if !hc8::symbols().contains(&base.symbol()) {
                    continue;
                }
                let solo = card
                    .0
                    .module_get_function(&module, &base.symbol())
                    .map_err(|e| e.to_string())?;
                let mut functions = Vec::new();
                for nt in 1..=4 {
                    let geo = Geo { nt, ..base };
                    let pm = card
                        .0
                        .module_get_function(&module, &geo.symbol())
                        .map_err(|e| e.to_string())?;
                    functions.push((geo, pm));
                }
                let synthetic: Vec<Input> = (0..3)
                    .map(|kind| {
                        let (w, scales, x) = data(fmt, small_m, small_k, kind);
                        Input {
                            name: ["random_finite", "finite_edges", "nan_inf_edges"][kind].into(),
                            fmt,
                            m: small_m,
                            k: small_k,
                            w,
                            scales,
                            x,
                        }
                    })
                    .collect();
                let mut references = std::collections::BTreeMap::new();
                for case in synthetic.iter().chain(fixtures.iter().filter(|c| {
                    fmt_name(c.fmt) == fmt_name(fmt)
                        && c.m % (16 * mw as usize) == 0
                        && c.k % (32 * kw as usize) == 0
                })) {
                    let (m, k) = (case.m, case.k);
                    let (w, scales, x) = (&case.w, &case.scales, &case.x);
                    let kind = &case.name;
                    let (_reference_module, rf) = match references.entry(k) {
                        std::collections::btree_map::Entry::Occupied(entry) => entry.into_mut(),
                        std::collections::btree_map::Entry::Vacant(entry) => {
                            let reference = reference::emit_reference(base, k as u32)?;
                            let re = native::assemble(&reference.s_text, Arch::Gfx1201)?;
                            let rm = card.0.module_load_data(&re).map_err(|e| e.to_string())?;
                            let rf = card
                                .0
                                .module_get_function(
                                    &rm,
                                    &reference::reference_symbol(base, k as u32),
                                )
                                .map_err(|e| e.to_string())?;
                            entry.insert((rm, rf))
                        }
                    };
                    let q = guarded(&hc8::swizzle_hc8(&w, m, k));
                    let decoded: Vec<u8> = reference::decode_weights(fmt, &w)
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect();
                    let decoded = guarded(&decoded);
                    let scales = guarded(&scales);
                    let x = guarded(&x);
                    let dq = card.upload(&q)?;
                    let dw = card.upload(&decoded)?;
                    let ds = card.upload(&scales)?;
                    let dx = card.upload(&x)?;
                    let poison = vec![SENT; 2 * GUARD + 64 * m * 4];
                    let dy = card.upload(&poison)?;
                    let dr = card.upload(&poison)?;
                    let mut singleton = Vec::with_capacity(64 * m * 4);
                    for row in 0..64 {
                        card.0
                            .memcpy_htod(&dy, &poison)
                            .map_err(|e| e.to_string())?;
                        let mut blob = args(
                            ptr(&dq),
                            ptr(&ds),
                            ptr(&dx) + (row * k * 4) as u64,
                            ptr(&dy),
                            m,
                            k,
                            1,
                        );
                        card.launch(&solo, base, false, &mut blob)?;
                        launches += 1;
                        let bytes = card.download(&dy, poison.len())?;
                        check_extent(&bytes, m * 4, "singleton")?;
                        singleton.extend_from_slice(&bytes[GUARD..GUARD + m * 4]);
                    }
                    let mut comparisons = 0;
                    for (geo, pm) in &functions {
                        for rows in 1..=geo.max_rows() as usize {
                            for d in [&dy, &dr] {
                                card.0.memcpy_htod(d, &poison).map_err(|e| e.to_string())?;
                            }
                            let mut blob = args(ptr(&dq), ptr(&ds), ptr(&dx), ptr(&dy), m, k, rows);
                            card.launch(pm, *geo, false, &mut blob)?;
                            let mut reference_blob =
                                args(ptr(&dw), ptr(&ds), ptr(&dx), ptr(&dr), m, k, rows);
                            card.launch(&rf, *geo, true, &mut reference_blob)?;
                            launches += 2;
                            let got = card.download(&dy, poison.len())?;
                            let want = card.download(&dr, poison.len())?;
                            let context = format!("{} kind={kind} rows={rows}", geo.symbol());
                            check_extent(&got, rows * m * 4, &context)?;
                            check_extent(&want, rows * m * 4, &format!("reference {context}"))?;
                            compared_bytes +=
                                compare(&got, &want, &format!("dequant-BF16 WMMA {context}"))?;
                            compared_bytes += compare(
                                &got[GUARD..GUARD + rows * m * 4],
                                &singleton[..rows * m * 4],
                                &format!("singleton {context}"),
                            )?;
                            comparisons += 1;
                        }
                    }
                    for (d, bytes) in [(&dq, &q), (&dw, &decoded), (&ds, &scales), (&dx, &x)] {
                        if card.download(d, bytes.len())? != *bytes {
                            return Err(format!(
                                "{} kind={kind}: readonly input changed",
                                base.symbol()
                            ));
                        }
                    }
                    cases.push(json!({"fmt":fmt_name(fmt),"mw":mw,"kw":kw,"act":act,"m":m,"k":k,"kind":kind,
                        "q_rowmajor_sha256":format!("{:x}",Sha256::digest(w)),
                        "x_sha256":format!("{:x}",Sha256::digest(&case.x)),
                        "scales_sha256":format!("{:x}",Sha256::digest(&case.scales)),
                        "comparisons":comparisons,"rows":"every 1..64, every compatible NT","pass":true}));
                    println!(
                        "{} kind={kind}: {comparisons} comparisons PASS",
                        base.symbol()
                    );
                    for d in [dq, dw, ds, dx, dy, dr] {
                        card.0.free(d).map_err(|e| e.to_string())?;
                    }
                }
            }
        }
    }
    let bf16 = card
        .0
        .module_get_function(&module, "hc8_dequant_bf16")
        .map_err(|e| e.to_string())?;
    let f16 = card
        .0
        .module_get_function(&module, "hc8_dequant_f16")
        .map_err(|e| e.to_string())?;
    let (dequant_cases, dequant_bytes) = dequant_oracle(&card, [&bf16, &f16], &fixtures)?;
    compared_bytes += dequant_bytes;
    launches += dequant_cases.len();
    let receipt = json!({"schema":"hc8-byte-oracle-v1","pass":true,"arch":arch,
        "fabric_job":std::env::var("FABRIC_JOB").ok(),"elf_sha256":format!("{:x}",Sha256::digest(&elf)),
        "bundle_sha256":bundle_sha,"fixtures":fixture_paths,
        "reference":"independent row-major dequantized BF16 WMMA; identical fixed KW fold; no compressed decoder",
        "compared_bytes":compared_bytes,"launches":launches,"cases":cases,"dequant":dequant_cases,"s_bfm_b32":scalar_mask});
    std::fs::write(
        out,
        serde_json::to_string_pretty(&receipt).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    println!("PASS: {compared_bytes} bytes, {launches} launches; receipt {out}");
    Ok(())
}
/// Words are compared bitwise. Both-NaN differences are tallied separately so a
/// payload/sign-only divergence is visible but still counts as a mismatch.
#[derive(Default)]
struct Tally {
    comparisons: usize,
    words: usize,
    mismatches: usize,
    nan_both: usize,
    samples: Vec<serde_json::Value>,
}
impl Tally {
    fn cmp(&mut self, context: &str, got: &[u8], want: &[u8]) {
        assert_eq!(got.len(), want.len(), "{context}: unequal extents");
        self.comparisons += 1;
        for (i, (g, w)) in got
            .chunks_exact(4)
            .zip(want.chunks_exact(4))
            .enumerate()
        {
            self.words += 1;
            if g == w {
                continue;
            }
            let (g, w) = (
                u32::from_le_bytes(g.try_into().unwrap()),
                u32::from_le_bytes(w.try_into().unwrap()),
            );
            self.mismatches += 1;
            self.nan_both += (f32::from_bits(g).is_nan() && f32::from_bits(w).is_nan()) as usize;
            if self.samples.len() < 8 {
                self.samples.push(json!({"context":context,"word":i,
                    "got":format!("{g:08x}"),"want":format!("{w:08x}")}));
            }
        }
    }
    fn json(&self) -> serde_json::Value {
        json!({"comparisons":self.comparisons,"words":self.words,"mismatches":self.mismatches,
            "mismatches_both_nan":self.nan_both,"samples":self.samples})
    }
}

#[derive(Default)]
struct Timings(std::collections::BTreeMap<String, (u64, u128)>);
impl Timings {
    fn json(&self) -> serde_json::Value {
        serde_json::Value::Object(
            self.0
                .iter()
                .map(|(k, (n, ns))| {
                    (
                        k.clone(),
                        json!({"launches":n,"host_wall_ns_incl_sync":ns.to_string()}),
                    )
                })
                .collect(),
        )
    }
}

/// One launch plus device sync, timed on the host wall clock. This is a
/// correctness-run diagnostic, not a performance measurement.
fn launch_timed(
    card: &Card,
    timings: &mut Timings,
    name: &str,
    f: &Function,
    grid: [u32; 3],
    block: u32,
    blob: &mut [u8],
) -> Result<(), String> {
    let start = std::time::Instant::now();
    // SAFETY: every blob pointer is a live guarded device allocation sized for
    // the shape the caller built; the ABI byte layout matches each entry.
    unsafe {
        card.0
            .launch_kernel_blob(f, grid, [block, 1, 1], 0, None, blob)
    }
    .map_err(|e| format!("{name}: {e}"))?;
    card.0
        .device_synchronize()
        .map_err(|e| format!("{name}: {e}"))?;
    let e = timings.0.entry(name.to_string()).or_default();
    e.0 += 1;
    e.1 += start.elapsed().as_nanos();
    Ok(())
}

/// Non-sentinel bytes outside the live extent (guards and dead rows).
fn dead_bytes(bytes: &[u8], live: usize) -> usize {
    bytes[..GUARD]
        .iter()
        .chain(bytes[GUARD + live..].iter())
        .filter(|&&b| b != SENT)
        .count()
}
/// Live words still equal to the sentinel. A BF16-rounded output has zero low
/// 16 bits, so a legitimate result can never equal 0xa5a5a5a5.
fn unstored_words(bytes: &[u8], live: usize) -> usize {
    bytes[GUARD..GUARD + live]
        .chunks_exact(4)
        .filter(|w| w.iter().all(|&b| b == SENT))
        .count()
}

struct ReadUpCase {
    name: String,
    hidden: usize,
    k: usize,
    w: Vec<u8>,
    scales: Vec<u8>,
    low: Vec<u8>,
    normalized: Vec<u8>,
    normalized_source: &'static str,
}

fn bf16_word(v: f32) -> u32 {
    peacemaker_ir::emu::convert::f32_to_bf16(v.to_bits()) as u32
}
/// Seeded BF16-exact normalized [64, 4*hidden]: realistic uniform |x|<=4 with
/// `kind` selecting sprinkled edges (1) or edges plus NaN/Inf (2).
fn normalized_data(hidden: usize, kind: usize) -> Vec<u8> {
    let mut rng = Rng(0x7e5d_0a11_c0de_0003 ^ (hidden as u64) << 20 ^ kind as u64);
    let edge = [
        0x0000_0000u32,
        0x8000_0000,
        0x0001_0000,
        0x8001_0000,
        0x0080_0000,
        0x7f7f_0000,
        0xff7f_0000,
        0x3f80_0000,
        0xbf80_0000,
        0x4080_0000,
        0xc080_0000,
    ];
    let special = [0x7f80_0000u32, 0xff80_0000, 0x7fc1_0000, 0xffa1_0000];
    (0..64 * 4 * hidden)
        .flat_map(|i| {
            let bits = if kind >= 1 && i % 16 == 5 {
                edge[(i / 16) % edge.len()]
            } else if kind == 2 && i % 37 == 11 {
                special[(i / 37) % special.len()]
            } else {
                let u = rng.next() as f64 / (1u64 << 16) as f64 / 65536.0;
                bf16_word((u * 8.0 - 4.0) as f32) << 16
            };
            bits.to_le_bytes()
        })
        .collect()
}

fn read_up_cases(fixtures: &[Input]) -> Vec<ReadUpCase> {
    let mut cases = Vec::new();
    for (hidden, k) in [(16usize, 32usize), (32, 96), (80, 320)] {
        for kind in 0..4 {
            let (w, mut scales, low) = data(Fmt::E4M3, 4 * hidden, k, kind % 3);
            if kind == 3 {
                // Moderate scales keep the sigmoid interior live (kind 0 saturates).
                scales = (0..4 * hidden)
                    .flat_map(|i| {
                        let e = -14 + (i % 9) as i32;
                        f32::from_bits(((127 + e) as u32) << 23).to_le_bytes()
                    })
                    .collect();
            }
            cases.push(ReadUpCase {
                name: [
                    "random_finite",
                    "finite_edges",
                    "nan_inf_edges",
                    "moderate_scales",
                ][kind]
                    .into(),
                hidden,
                k,
                w,
                scales,
                low,
                normalized: normalized_data(hidden, [0, 1, 2, 0][kind]),
                normalized_source: "synthetic_seeded_bf16",
            });
        }
    }
    for f in fixtures.iter().filter(|f| f.fmt == Fmt::E4M3 && f.m % 64 == 0) {
        let hidden = f.m / 4;
        for (suffix, kind) in [("", 0), ("+nan_inf_normalized", 2)] {
            cases.push(ReadUpCase {
                name: format!("{}{suffix}", f.name),
                hidden,
                k: f.k,
                w: f.w.clone(),
                scales: f.scales.clone(),
                low: f.x.clone(),
                normalized: normalized_data(hidden, kind),
                normalized_source: "synthetic_seeded_bf16",
            });
        }
    }
    cases
}

/// Kernarg of the fused read-up entry: q,s,low,normalized,mixed u64; hidden,
/// low_rank, rows u32; pad.
fn read_up_args(
    p: [u64; 5],
    hidden: usize,
    k: usize,
    rows: usize,
) -> Result<Vec<u8>, String> {
    let mut blob = Vec::with_capacity(hc8::READ_UP_KERNARG_BYTES as usize);
    for v in p {
        blob.extend(v.to_le_bytes());
    }
    for v in [hidden, k, rows] {
        blob.extend((v as u32).to_le_bytes());
    }
    if blob.len() > hc8::READ_UP_KERNARG_BYTES as usize {
        return Err("read-up kernarg fields overflow READ_UP_KERNARG_BYTES".into());
    }
    blob.resize(hc8::READ_UP_KERNARG_BYTES as usize, 0);
    Ok(blob)
}
/// Product ABI: five pointers, branches/hidden/rows i32, padded to 16 as
/// `hyper_read_projected` does in rdna-compute.
fn projected_args(p: [u64; 5], hidden: usize, rows: usize) -> Vec<u8> {
    let mut blob = Vec::with_capacity(64);
    for v in p {
        blob.extend(v.to_le_bytes());
    }
    for v in [4usize, hidden, rows] {
        blob.extend((v as u32).to_le_bytes());
    }
    blob.resize(blob.len().next_multiple_of(16), 0);
    blob
}

fn run_read_up(out: &str, projected_object: &str, fixture_paths: &[String]) -> Result<(), String> {
    if std::env::var_os("FABRIC_JOB").is_none() {
        return Err("run through fabric-run.sh --class correct --card <one gfx1201 card>".into());
    }
    let object = std::fs::read(projected_object).map_err(|e| format!("{projected_object}: {e}"))?;
    let rt = HipRuntime::load().map_err(|e| e.to_string())?;
    rt.set_device(0).map_err(|e| e.to_string())?;
    let arch = rt.get_arch(0).map_err(|e| e.to_string())?;
    if arch.split(':').next() != Some("gfx1201") {
        return Err(format!("requires exact gfx1201, got {arch}"));
    }
    if rt.device_count().map_err(|e| e.to_string())? != 1 {
        return Err("requires exactly one visible card".into());
    }
    let card = Card(rt);
    let fixtures = load_fixtures(fixture_paths)?;
    let (_, text, _) = hc8::emit_module()?;
    let elf = native::assemble(&text, Arch::Gfx1201)?;
    let bundle_sha = format!("{:x}", Sha256::digest(BUNDLE));
    if bundle_sha != BUNDLE_SHA.trim()
        || native::bundle(&elf, Arch::Gfx1201, native::DEFAULT_HOST_TARGET) != BUNDLE
    {
        return Err("committed bundle/sha differs from fresh generator output".into());
    }
    let symbols = hc8::symbols();
    let module = card.0.module_load_data(&elf).map_err(|e| e.to_string())?;
    let projected_module = card
        .0
        .module_load_data(&object)
        .map_err(|e| format!("{projected_object}: {e}"))?;
    let projected = card
        .0
        .module_get_function(&projected_module, "hyper_read_projected_f32")
        .map_err(|e| e.to_string())?;
    let mut fused = Vec::new();
    let mut sets = Vec::new();
    for nt in 1..=4u32 {
        let geo = Geo {
            fmt: Fmt::E4M3,
            mw: 4,
            kw: 1,
            nt,
            act: false,
        };
        for name in [hc8::read_up_symbol(nt), geo.symbol()] {
            if !symbols.contains(&name) {
                return Err(format!("module symbols() lacks {name}"));
            }
        }
        fused.push(
            card.0
                .module_get_function(&module, &hc8::read_up_symbol(nt))
                .map_err(|e| e.to_string())?,
        );
        sets.push((
            geo,
            card.0
                .module_get_function(&module, &geo.symbol())
                .map_err(|e| e.to_string())?,
        ));
    }

    let dummy = card.upload(&guarded(&[0u8; 16]))?;
    let mut timings = Timings::default();
    let mut cases = Vec::new();
    let (mut launches, mut compared_words) = (0usize, 0usize);
    let mut total_mismatches = 0usize;
    let mut failures = Vec::new();
    for case in read_up_cases(&fixtures) {
        let (hidden, k) = (case.hidden, case.k);
        let m = 4 * hidden;
        if hidden % 16 != 0 || k % 32 != 0 {
            return Err(format!("{}: hidden%16/K%32 violated", case.name));
        }
        let q = guarded(&hc8::swizzle_hc8(&case.w, m, k));
        let scales = guarded(&case.scales);
        let low = guarded(&case.low);
        let norm = guarded(&case.normalized);
        let (dq, ds, dl, dn) = (
            card.upload(&q)?,
            card.upload(&scales)?,
            card.upload(&low)?,
            card.upload(&norm)?,
        );
        let out_poison = vec![SENT; 2 * GUARD + 64 * hidden * 4];
        let set_poison = vec![SENT; 2 * GUARD + 64 * m * 4];
        let (dfused, dprod) = (card.upload(&out_poison)?, card.upload(&out_poison)?);
        let dset = card.upload(&set_poison)?;
        let (mut vs_product, mut vs_single) = (Tally::default(), Tally::default());
        let (mut dead, mut unstored, mut set_dead) = (0usize, 0usize, 0usize);
        let mut case_launches = 0usize;
        let grid = (hidden / 16) as u32;
        let block = hc8::READ_UP_BLOCK as u32;

        // Row-invariance anchor: each input row alone through the n1 entry.
        let mut singleton = Vec::with_capacity(64 * hidden * 4);
        for row in 0..64usize {
            card.0
                .memcpy_htod(&dfused, &out_poison)
                .map_err(|e| e.to_string())?;
            let mut blob = read_up_args(
                [
                    ptr(&dq),
                    ptr(&ds),
                    ptr(&dl) + (row * k * 4) as u64,
                    ptr(&dn) + (row * m * 4) as u64,
                    ptr(&dfused),
                ],
                hidden,
                k,
                1,
            )?;
            launch_timed(
                &card,
                &mut timings,
                &hc8::read_up_symbol(1),
                &fused[0],
                [grid, 1, 1],
                block,
                &mut blob,
            )?;
            case_launches += 1;
            let bytes = card.download(&dfused, out_poison.len())?;
            dead += dead_bytes(&bytes, hidden * 4);
            unstored += unstored_words(&bytes, hidden * 4);
            singleton.extend_from_slice(&bytes[GUARD..GUARD + hidden * 4]);
        }

        for nt in 1..=4usize {
            let (geo, set_fn) = &sets[nt - 1];
            for rows in 1..=16 * nt {
                for (d, p) in [(&dfused, &out_poison), (&dprod, &out_poison), (&dset, &set_poison)]
                {
                    card.0.memcpy_htod(d, p).map_err(|e| e.to_string())?;
                }
                let mut set_blob = args(ptr(&dq), ptr(&ds), ptr(&dl), ptr(&dset), m, k, rows);
                launch_timed(
                    &card,
                    &mut timings,
                    &geo.symbol(),
                    set_fn,
                    [geo.grid(m as u32), 1, 1],
                    geo.block(),
                    &mut set_blob,
                )?;
                let mut prod_blob = projected_args(
                    [ptr(&dummy), ptr(&dummy), ptr(&dset), ptr(&dn), ptr(&dprod)],
                    hidden,
                    rows,
                );
                launch_timed(
                    &card,
                    &mut timings,
                    "hyper_read_projected_f32",
                    &projected,
                    [(m as u32).div_ceil(256), rows as u32, 1],
                    256,
                    &mut prod_blob,
                )?;
                let mut fused_blob = read_up_args(
                    [ptr(&dq), ptr(&ds), ptr(&dl), ptr(&dn), ptr(&dfused)],
                    hidden,
                    k,
                    rows,
                )?;
                launch_timed(
                    &card,
                    &mut timings,
                    &hc8::read_up_symbol(nt as u32),
                    &fused[nt - 1],
                    [grid, 1, 1],
                    block,
                    &mut fused_blob,
                )?;
                case_launches += 3;
                let got = card.download(&dfused, out_poison.len())?;
                let want = card.download(&dprod, out_poison.len())?;
                let mid = card.download(&dset, set_poison.len())?;
                let live = rows * hidden * 4;
                dead += dead_bytes(&got, live) + dead_bytes(&want, live);
                unstored += unstored_words(&got, live) + unstored_words(&want, live);
                set_dead += dead_bytes(&mid, rows * m * 4);
                let context = format!("{} rows={rows}", hc8::read_up_symbol(nt as u32));
                vs_product.cmp(
                    &format!("fused vs product {context}"),
                    &got[GUARD..GUARD + live],
                    &want[GUARD..GUARD + live],
                );
                vs_single.cmp(
                    &format!("fused vs singleton {context}"),
                    &got[GUARD..GUARD + live],
                    &singleton[..live],
                );
            }
        }
        let mut readonly_changed = Vec::new();
        for (name, d, bytes) in [
            ("q", &dq, &q),
            ("s", &ds, &scales),
            ("low", &dl, &low),
            ("normalized", &dn, &norm),
        ] {
            if card.download(d, bytes.len())? != *bytes {
                readonly_changed.push(name);
            }
        }
        let mismatches = vs_product.mismatches
            + vs_single.mismatches
            + dead
            + unstored
            + set_dead
            + readonly_changed.len();
        let pass = mismatches == 0;
        if !pass {
            failures.push(format!(
                "{} hidden={hidden} k={k}: product {} singleton {} dead {dead} unstored {unstored} set_dead {set_dead} readonly {readonly_changed:?}",
                case.name, vs_product.mismatches, vs_single.mismatches
            ));
        }
        total_mismatches += mismatches;
        launches += case_launches;
        compared_words += vs_product.words + vs_single.words;
        println!(
            "{} hidden={hidden} k={k}: {} words vs product, {} vs singleton, {mismatches} mismatches {}",
            case.name,
            vs_product.words,
            vs_single.words,
            if pass { "PASS" } else { "FAIL" }
        );
        cases.push(json!({"kind":case.name,"hidden":hidden,"m":m,"k":k,
            "normalized_source":case.normalized_source,
            "q_rowmajor_sha256":format!("{:x}",Sha256::digest(&case.w)),
            "scales_sha256":format!("{:x}",Sha256::digest(&case.scales)),
            "low_sha256":format!("{:x}",Sha256::digest(&case.low)),
            "normalized_sha256":format!("{:x}",Sha256::digest(&case.normalized)),
            "rows":"every 1..=16*NT for each NT 1..4; plus 64 singleton rows",
            "launches":case_launches,
            "fused_vs_product":vs_product.json(),"fused_vs_singleton":vs_single.json(),
            "dead_store_bytes":dead,"unstored_live_words":unstored,
            "set_intermediate_dead_store_bytes":set_dead,
            "readonly_changed":readonly_changed,"pass":pass}));
        for d in [dq, ds, dl, dn, dfused, dprod, dset] {
            card.0.free(d).map_err(|e| e.to_string())?;
        }
    }
    card.0.free(dummy).map_err(|e| e.to_string())?;
    let pass = total_mismatches == 0;
    let receipt = json!({"schema":"hc8-read-up-byte-oracle-v1","pass":pass,"arch":arch,
        "fabric_job":std::env::var("FABRIC_JOB").ok(),
        "elf_sha256":format!("{:x}",Sha256::digest(&elf)),"bundle_sha256":bundle_sha,
        "projected_object":projected_object,
        "projected_object_sha256":format!("{:x}",Sha256::digest(&object)),
        "projected_symbol":"hyper_read_projected_f32","fixtures":fixture_paths,
        "baseline":"hc8_gemm_e4m3_mw4_kw1_nN_set then product hyper_read_projected_f32 (compiled object, no hipcc, no CPU libm)",
        "normalized":"seeded BF16-exact synthetic, |x|<=4 plus +-0/tiny/large/NaN/Inf edges; no real normalized capture",
        "compared_words":compared_words,"launches":launches,
        "total_mismatches":total_mismatches,"failures":failures,
        "timings_note":"host wall clock around launch+sync; correctness diagnostic, not performance",
        "timings":timings.json(),"cases":cases});
    std::fs::write(
        out,
        serde_json::to_string_pretty(&receipt).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    if !pass {
        return Err(format!(
            "{total_mismatches} mismatches; receipt {out}: {}",
            failures.join("; ")
        ));
    }
    println!("PASS: {compared_words} words, {launches} launches; receipt {out}");
    Ok(())
}

fn main() -> ExitCode {
    let mut read_up = false;
    let mut projected = None;
    let mut positional = Vec::new();
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--read-up" => read_up = true,
            "--projected-object" => projected = it.next(),
            _ => positional.push(a),
        }
    }
    let usage = "usage: hc8_byte_oracle RECEIPT.json [FIXTURE.json ...]\n       hc8_byte_oracle --read-up --projected-object OBJECT.co RECEIPT.json [FIXTURE.json ...]";
    let Some(out) = positional.first() else {
        eprintln!("{usage}");
        return ExitCode::from(2);
    };
    let result = match (read_up, &projected) {
        (true, Some(object)) => run_read_up(out, object, &positional[1..]),
        (false, None) => run(out, &positional[1..]),
        _ => {
            eprintln!("{usage}\n--projected-object is mandatory with --read-up and only valid with it");
            return ExitCode::from(2);
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("FAIL: {e}");
            ExitCode::FAILURE
        }
    }
}
