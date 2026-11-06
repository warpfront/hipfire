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
fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(out) = args.first() else {
        eprintln!("usage: hc8_byte_oracle RECEIPT.json [FIXTURE.json ...]");
        return ExitCode::from(2);
    };
    match run(out, &args[1..]) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("FAIL: {e}");
            ExitCode::FAILURE
        }
    }
}
