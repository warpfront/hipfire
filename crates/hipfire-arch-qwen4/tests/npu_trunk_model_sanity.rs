// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Ignored CPU-only sanity check of `pack_matrix` on real MQ6G256V2 trunk tensors of the Flash-Next artifact.
//!
//! Run: `cargo test -p hipfire-arch-qwen4 --release --test npu_trunk_model_sanity -- --ignored --nocapture`
//! Model: `$HIPFIRE_NPU_TRUNK_TEST_MODEL`, default `/home/kaden/qcal/models/qwen3.8-flash-next-gptq3.mq4`.
//!
//! Model pin: `qwen3.8-flash-next-gptq3.mq4`, md5 `be007fc3219e9f6cdb1d4dfa8380625f` (recorded in
//! `docs/npu/coop-flashnext.md:135`, observed on `/home/kaden/pm-wave/fngptq/models/...`; AGENTS.md:703 calls
//! the old filename a byte-identical alias). The 125 GB file is NOT hashed here (a full digest per test run is
//! not feasible); the test checks the exact file size of the default path and the HFQM index extent, prints the
//! path, and asserts the raw quant type of each tensor is MQ6G256V2 (qt 47).
//!
//! HFQM reader: the index walk is copied from `crates/npu-tools/src/bin/coop27/hfqm.rs` (`index`, `read_tensor`;
//! port of `tools/npu/a4-ec/src/hfqm.rs::index`), reading the stored bytes raw without decoding. The only change is
//! that the leading JSON metadata object is skipped by a small brace matcher instead of `serde_json`, so this test
//! needs no extra dev-dependency. Tensor names are those of the artifact (`weights.rs` inventory test:
//! `model.language_model.layers.{0.linear_attn.in_proj_qkv,3.self_attn.q_proj}.weight`).
//!
//! The decode + quantizer oracle is the lab CPU twin, copied with the same citation as `tests/packing.rs`
//! (`npu_a8emu_check.rs` in `/home/kaden/ClaudeCode/warpfront/wt-npu-a8emu`: `f16_bits_to_f32` l.58-75,
//! `unpack_group` l.138-147, `decode_mq6` l.175-196, weights half of `cpu_w8a8` l.202-217).

use hipfire_arch_qwen4::npu_trunk::{pack_matrix, TRUNK_K};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::time::Instant;

const DEFAULT_MODEL: &str = "/home/kaden/qcal/models/qwen3.8-flash-next-gptq3.mq4";
const DEFAULT_MODEL_BYTES: u64 = 125_288_544_792;
const PINNED_MD5: &str = "be007fc3219e9f6cdb1d4dfa8380625f";
const QT_MQ6G256V2: u8 = 47; // crates/hipfire-arch-qwen4/src/artifact.rs:39
const GROUP_BYTES: usize = 200;

// ---------------------------------------------------------------------------------------------
// HFQM index (copied from npu-tools coop27/hfqm.rs, see header).
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Tensor {
    name: String,
    qt: u8,
    dims: Vec<usize>,
    group: usize,
    off: u64,
    size: u64,
}

fn take<'a>(bytes: &'a [u8], pos: &mut usize, n: usize) -> &'a [u8] {
    let v = &bytes[*pos..*pos + n];
    *pos += n;
    v
}
fn word(bytes: &[u8], pos: &mut usize) -> usize {
    u32::from_le_bytes(take(bytes, pos, 4).try_into().unwrap()) as usize
}

/// Byte length of the leading JSON object (replaces `serde_json::Deserializer::byte_offset`).
fn json_object_len(bytes: &[u8]) -> Result<usize, String> {
    if bytes.first() != Some(&b'{') {
        return Err("HFQM metadata does not start with a JSON object".into());
    }
    let (mut depth, mut in_str, mut esc) = (0usize, false, false);
    for (i, &c) in bytes.iter().enumerate() {
        if in_str {
            if esc {
                esc = false;
            } else if c == b'\\' {
                esc = true;
            } else if c == b'"' {
                in_str = false;
            }
        } else {
            match c {
                b'"' => in_str = true,
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        return Ok(i + 1);
                    }
                }
                _ => {}
            }
        }
    }
    Err("unterminated HFQM JSON metadata".into())
}

fn index(file: &mut File) -> Result<Vec<Tensor>, String> {
    let mut header = [0; 32];
    file.read_exact(&mut header)
        .map_err(|e| format!("hfqm header: {e}"))?;
    if &header[..4] != b"HFQM" || u32::from_le_bytes(header[4..8].try_into().unwrap()) != 1 {
        return Err("not an HFQM v1 file".into());
    }
    let metadata = u64::from_le_bytes(header[16..24].try_into().unwrap());
    let data = u64::from_le_bytes(header[24..32].try_into().unwrap());
    if data < metadata || data - metadata >= 32_000_000 {
        return Err("bad HFQM metadata range".into());
    }
    file.seek(SeekFrom::Start(metadata))
        .map_err(|e| e.to_string())?;
    let mut bytes = vec![0; (data - metadata) as usize];
    file.read_exact(&mut bytes).map_err(|e| e.to_string())?;
    let mut pos = json_object_len(&bytes)?;
    let count = word(&bytes, &mut pos);
    let mut tensors = Vec::with_capacity(count);
    let mut offset = data;
    for _ in 0..count {
        let n = u16::from_le_bytes(take(&bytes, &mut pos, 2).try_into().unwrap()) as usize;
        let name =
            String::from_utf8(take(&bytes, &mut pos, n).to_vec()).map_err(|e| e.to_string())?;
        let qt = take(&bytes, &mut pos, 1)[0];
        let ndim = take(&bytes, &mut pos, 1)[0];
        let dims = (0..ndim).map(|_| word(&bytes, &mut pos)).collect();
        let group = word(&bytes, &mut pos);
        let size = u64::from_le_bytes(take(&bytes, &mut pos, 8).try_into().unwrap());
        tensors.push(Tensor {
            name,
            qt,
            dims,
            group,
            off: offset,
            size,
        });
        offset += size;
    }
    if offset != file.metadata().map_err(|e| e.to_string())?.len() {
        return Err("HFQM tensor sizes do not cover the file".into());
    }
    Ok(tensors)
}

fn read_raw(file: &mut File, t: &Tensor) -> Result<Vec<u8>, String> {
    file.seek(SeekFrom::Start(t.off))
        .map_err(|e| e.to_string())?;
    let mut v = vec![0u8; t.size as usize];
    file.read_exact(&mut v).map_err(|e| e.to_string())?;
    Ok(v)
}

// ---------------------------------------------------------------------------------------------
// Lab CPU oracle (copied, see header).
// ---------------------------------------------------------------------------------------------

fn f16_bits_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) as u32) << 31;
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as u32;
    let bits = if e == 0 {
        if m == 0 {
            s
        } else {
            let v = m as f32 * (1.0 / 16_777_216.0);
            return if s != 0 { -v } else { v };
        }
    } else if e == 31 {
        s | 0x7f80_0000 | (m << 13)
    } else {
        s | (((e - 15 + 127) as u32) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}

fn unpack_group(payload: &[u8], q: &mut [u8; 256]) {
    for c in 0..64 {
        let (b0, b1, b2) = (payload[c * 3], payload[c * 3 + 1], payload[c * 3 + 2]);
        q[c * 4] = b0 & 63;
        q[c * 4 + 1] = (b0 >> 6) | ((b1 & 0x0f) << 2);
        q[c * 4 + 2] = (b1 >> 4) | ((b2 & 0x03) << 4);
        q[c * 4 + 3] = b2 >> 2;
    }
}

/// Independent bit-stream decoder (field `j` = bits `6j..6j+6` of the LE payload), cross-check only.
fn unpack_group_bitstream(payload: &[u8], q: &mut [u8; 256]) {
    for (j, out) in q.iter_mut().enumerate() {
        let bit = j * 6;
        let byte = bit / 8;
        let lo = payload[byte] as u16;
        let hi = payload.get(byte + 1).copied().unwrap_or(0) as u16;
        *out = ((((hi << 8) | lo) >> (bit % 8)) & 63) as u8;
    }
}

struct DecodedWeights {
    codes: Vec<u8>,
    s: Vec<f32>,
    z: Vec<f32>,
}

fn decode_mq6(blob: &[u8], m: usize, k: usize) -> DecodedWeights {
    let epochs = k / 128;
    let mut out = DecodedWeights {
        codes: vec![0; m * k],
        s: vec![0.0; m * epochs],
        z: vec![0.0; m * epochs],
    };
    let mut q = [0u8; 256];
    let mut q2 = [0u8; 256];
    for (gi, g) in blob.chunks_exact(GROUP_BYTES).enumerate() {
        let h = |o: usize| f16_bits_to_f32(u16::from_le_bytes([g[o], g[o + 1]]));
        unpack_group(&g[8..], &mut q);
        unpack_group_bitstream(&g[8..], &mut q2);
        assert_eq!(q[..], q2[..], "oracle decoders disagree in group {gi}");
        out.codes[gi * 256..(gi + 1) * 256].copy_from_slice(&q);
        out.s[gi * 2] = h(0);
        out.z[gi * 2] = h(2);
        out.s[gi * 2 + 1] = h(4);
        out.z[gi * 2 + 1] = h(6);
    }
    out
}

fn clamp_code(q: f32, qmax: i32) -> i32 {
    (q.round_ties_even() as i32).clamp(-qmax, qmax)
}

fn wprep_w8(w: &DecodedWeights, m: usize, k: usize) -> (Vec<i8>, Vec<f32>) {
    let epochs = k / 128;
    let mut w8 = vec![0i8; m * k];
    let mut ws = vec![0f32; m];
    for n in 0..m {
        let wf = |i: usize| {
            (w.codes[n * k + i] as f32) * w.s[n * epochs + i / 128] + w.z[n * epochs + i / 128]
        };
        let absmax = (0..k).map(|i| wf(i).abs()).fold(0.0f32, f32::max);
        let wsn = (absmax as f64 / 127.0) as f32;
        ws[n] = wsn;
        if wsn != 0.0 {
            for i in 0..k {
                w8[n * k + i] = clamp_code(wf(i) / wsn, 127) as i8;
            }
        }
    }
    (w8, ws)
}

// ---------------------------------------------------------------------------------------------
// Test.
// ---------------------------------------------------------------------------------------------

fn find_tensor<'a>(tensors: &'a [Tensor], exact: &str, suffix: &str) -> &'a Tensor {
    if let Some(t) = tensors.iter().find(|t| t.name == exact) {
        return t;
    }
    let hits: Vec<&Tensor> = tensors
        .iter()
        .filter(|t| t.name.ends_with(suffix) && !t.name.starts_with("mtp."))
        .collect();
    match hits.as_slice() {
        [t] => t,
        _ => panic!(
            "tensor {exact} not found; {} suffix matches for {suffix}: {:?}",
            hits.len(),
            hits.iter().map(|t| &t.name).collect::<Vec<_>>()
        ),
    }
}

#[test]
#[ignore = "reads the 125 GB Flash-Next artifact; run explicitly with --ignored"]
fn real_mq6_trunk_tensors_match_oracle() {
    let path =
        std::env::var("HIPFIRE_NPU_TRUNK_TEST_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
    let mut file = File::open(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let len = file.metadata().unwrap().len();
    println!("model: {path} ({len} bytes); pinned md5 {PINNED_MD5} (docs/npu/coop-flashnext.md:135, full digest not computed)");
    if path == DEFAULT_MODEL {
        assert_eq!(
            len, DEFAULT_MODEL_BYTES,
            "default model size differs from the pinned artifact"
        );
    }
    let tensors = index(&mut file).unwrap_or_else(|e| panic!("{path}: {e}"));
    println!("hfqm: {} tensors", tensors.len());

    let wanted = [
        (
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            "layers.0.linear_attn.in_proj_qkv.weight",
        ),
        (
            "model.language_model.layers.3.self_attn.q_proj.weight",
            "layers.3.self_attn.q_proj.weight",
        ),
    ];
    for (exact, suffix) in wanted {
        let t = find_tensor(&tensors, exact, suffix);
        println!(
            "tensor {}: raw qt={} dims={:?} group={} off={} size={}",
            t.name, t.qt, t.dims, t.group, t.off, t.size
        );
        assert_eq!(
            t.qt, QT_MQ6G256V2,
            "{}: raw quant type {} is not MQ6G256V2 (47)",
            t.name, t.qt
        );
        assert_eq!(t.dims.len(), 2, "{}: dims", t.name);
        let (m, k) = (t.dims[0], t.dims[1]);
        assert_eq!(k, TRUNK_K, "{}: K", t.name);
        assert_eq!(t.group, 256, "{}: group size", t.name);
        assert_eq!(
            t.size as usize,
            m * (k / 256) * GROUP_BYTES,
            "{}: stored bytes",
            t.name
        );

        let blob = read_raw(&mut file, t).unwrap_or_else(|e| panic!("{}: {e}", t.name));

        let t0 = Instant::now();
        let (w8, ws) = pack_matrix(&blob, m, k);
        let pack_elapsed = t0.elapsed();

        assert_eq!(ws.len(), m);
        assert_eq!(w8.len(), m * k);
        let (mut min, mut max, mut sum, mut zeros) = (f32::INFINITY, 0.0f32, 0.0f64, 0usize);
        for &w in &ws {
            assert!(w.is_finite() && w >= 0.0, "{}: bad ws {w}", t.name);
            min = min.min(w);
            max = max.max(w);
            sum += w as f64;
            zeros += (w == 0.0) as usize;
        }
        let saturated = w8.iter().filter(|&&v| v == 127 || v == -127).count();
        assert!(w8.iter().all(|&v| v != i8::MIN), "{}: -128 code", t.name);
        println!(
            "  pack_matrix m={m} k={k}: {:.3} s | ws min={min:e} max={max:e} mean={:e} zeros={zeros}/{m} | |w8|=127 count={saturated}",
            pack_elapsed.as_secs_f64(),
            sum / m as f64
        );

        let t1 = Instant::now();
        let (w8_ref, ws_ref) = wprep_w8(&decode_mq6(&blob, m, k), m, k);
        println!("  oracle: {:.3} s", t1.elapsed().as_secs_f64());
        for n in 0..m {
            assert_eq!(
                ws[n].to_bits(),
                ws_ref[n].to_bits(),
                "{}: ws[{n}] {:e} vs oracle {:e}",
                t.name,
                ws[n],
                ws_ref[n]
            );
        }
        if let Some(i) = (0..m * k).find(|&i| w8[i] != w8_ref[i]) {
            panic!(
                "{}: w8[{}][{}] = {} vs oracle {}",
                t.name,
                i / k,
                i % k,
                w8[i],
                w8_ref[i]
            );
        }
        println!("  exact w8/ws match with the lab oracle over all {m} rows");
    }
}
