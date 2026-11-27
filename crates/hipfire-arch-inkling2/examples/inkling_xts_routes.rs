// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Developer-only operator-level per-route quality harness for the Inkling-2-Mini
//! (arch 24) `mq4-xts` artifact (typed layout `qt44-padded-awq-v1`) and the shipped
//! incumbent A. Not product code, no product dispatch is changed.
//!
//! Question answered: for one artifact, how far is each activation route's OUTPUT from
//! the exact F32 source teacher, per matrix / per expert block, on held-out check rows
//! of the real teacher forward (`inkling_kld calib --role check`)?
//!
//! Two modes (both developer tools):
//!
//! `teacher` (CPU, F64, no GPU): reads the check capture index + rows and the srcdump
//! F32 source weights, normalizes the routed-expert rows to slot-major arrays
//! (`slot = token * 8 + rank`) and writes the F64 teacher outputs of every operator
//! to `--out/L{LL}/`: per-matrix `y = W x` for wq / wk / wv / wr / gate_attn / wo and
//! the router, per slot `gate_up = Wgu x`, per slot `down(captured act)`, and the
//! routed block `y = sum_r w_r * Down(SiTU(Wgu x))` in rank order. `unembed` does the
//! same for a strided row subset of the BF16 source unembed.
//! With `--weights artifact:PATH` the same code runs on the artifact's own dequantized
//! (effective, AWQ-divided) weights: the exact-arithmetic quantization floor.
//!
//! `run` (GPU, gfx1201, real kernels): uploads the artifact's layers, runs every
//! route the artifact admits on the same rows and scores each against the teacher:
//!   experts: scalar decode chain (XTS: rotate_x_mq_awq_topk -> qt44 expanded GEMV
//!            gate_up -> SiTU+FfnAwq::PerExpert -> rotate 1280 -> expanded down ->
//!            Rank combine; A: the incumbent chain), grouped weight-only, grouped W4A8,
//!            grouped W4A4 (and W4FP8) with the per-slot AWQ producer;
//!   dense:   native weight_gemv, A4 IU4 prefill (`HIPFIRE_IU4_PREFILL`), A8 prefill
//!            (`HIPFIRE_A8_PREFILL`), and the `HIPFIRE_IU4_PREFILL=0` FP8 / F16 WMMA route.
//! One JSONL record per (artifact, layer, site, route).
//!
//! Usage:
//!   inkling_xts_routes teacher --layers 5,6,20 --check-dir D --src-dir D --out D [--model A.hfq]
//!                              [--weights artifact:PATH --a-down-rt DIR]
//!   inkling_xts_routes unembed --src-st F.safetensors --check-dir D --out D [--rows 8192]
//!                              [--weights artifact:PATH]
//!   inkling_xts_routes run --label B --artifact F.hfq --layers 5,6,20 --teacher-dir D
//!                          --check-dir D --out FILE.jsonl [--a-down-rt DIR] [--unembed-dir D]

use half::f16;
use hipfire_arch_inkling2::config::Inkling2Config;
use hipfire_arch_inkling2::weights::awq_sidecar_name;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::{weight_gemv, WeightTensor};
use hipfire_runtime::weight_backend::dequantize_weight_to_f32;
use rayon::prelude::*;
use rdna_compute::ffn_activation::{
    CombineOrder, FfnActivation, FfnActivationGrouped, FfnActivationGroupedI8, FfnActivationRows,
    FfnAwq, MoeCombine,
};
use rdna_compute::moe_grouped_qt44_iu4 as g;
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Value};
use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io::Write;
use std::path::{Path, PathBuf};

const E: usize = 128;
const H: usize = 2304;
const INTER: usize = 1152;
const DK: usize = 1280;
const TOPK: usize = 8;
const EATTN: usize = 3072;
const SCHEMA: &str = "hipfire.inkling2.xts_route_ops.v1";

// ───────────────────────── small utilities ─────────────────────────

fn arg(a: &[String], k: &str, d: &str) -> String {
    a.iter()
        .position(|x| x == k)
        .map(|i| a[i + 1].clone())
        .unwrap_or_else(|| d.to_string())
}

fn has(a: &[String], k: &str) -> bool {
    a.iter().any(|x| x == k)
}

fn die<T>(m: impl AsRef<str>) -> T {
    eprintln!("inkling_xts_routes: {}", m.as_ref());
    std::process::exit(2)
}

fn ok<T, Er: std::fmt::Debug>(r: Result<T, Er>, what: &str) -> T {
    match r {
        Ok(v) => v,
        Err(e) => die(format!("{what}: {e:?}")),
    }
}

fn read_f32_file(p: &Path) -> Vec<f32> {
    use std::io::Read;
    let mut f = ok(std::fs::File::open(p), &p.display().to_string());
    let len = ok(f.metadata(), "stat").len() as usize;
    assert!(len % 4 == 0, "{}: not f32", p.display());
    let mut v = vec![0f32; len / 4];
    let bytes = unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, len) };
    ok(f.read_exact(bytes), &p.display().to_string());
    v
}

fn write_f32_file(p: &Path, v: &[f32]) {
    let bytes = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
    ok(std::fs::write(p, bytes), &p.display().to_string());
}

fn write_i32_file(p: &Path, v: &[i32]) {
    let bytes = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
    ok(std::fs::write(p, bytes), &p.display().to_string());
}

fn read_i32_file(p: &Path) -> Vec<i32> {
    read_f32_file(p)
        .iter()
        .map(|v| v.to_bits() as i32)
        .collect()
}

fn read_json(p: &Path) -> Value {
    let b = ok(std::fs::read(p), &p.display().to_string());
    ok(serde_json::from_slice(&b), &p.display().to_string())
}

fn cmd_out(cmd: &str, args: &[&str]) -> String {
    let o = std::process::Command::new(cmd).args(args).output();
    match o {
        Ok(o) => String::from_utf8_lossy(&o.stdout)
            .split_whitespace()
            .next()
            .unwrap_or("")
            .to_string(),
        Err(_) => String::new(),
    }
}

fn exe_md5() -> String {
    let p = std::env::current_exe().unwrap();
    cmd_out("md5sum", &[p.to_str().unwrap()])
}

fn layer_dir(base: &Path, layer: usize) -> PathBuf {
    base.join(format!("L{layer:02}"))
}

fn f32_bits_of_i32(v: &[i32]) -> Vec<f32> {
    v.iter().map(|&x| f32::from_bits(x as u32)).collect()
}

fn parse_layers(s: &str) -> Vec<usize> {
    s.split(',').map(|x| x.parse().unwrap()).collect()
}

// ───────────────────────── check capture normalization ─────────────────────────

/// Routed-expert capture of one layer in slot-major form.
struct ExpertCapture {
    /// tokens = distinct (pass, span id, position) of the check rows
    t: usize,
    /// `[T, H]` normalized FFN row of each token (the gate_up input, shared by its 8 experts)
    x_tok: Vec<f32>,
    /// `[T * 8]` expert id per slot (`slot = token * 8 + rank`)
    ids: Vec<i32>,
    /// `[T * 8]` route weight per slot
    w: Vec<f32>,
    /// `[T * 8, INTER]` teacher post-SiTU row per slot (the captured `down` site)
    act_slot: Vec<f32>,
    /// gate_up rows of one token that differed bitwise from the first copy (must be 0)
    x_mismatch_rows: usize,
}

struct RowMeta {
    expert: usize,
    pass: u64,
    span: Vec<String>,
    pos: Vec<u64>,
    rank: Vec<u64>,
    weight: Vec<f64>,
    row0: usize,
    rows: usize,
}

fn site_segments(site: &Value) -> Vec<RowMeta> {
    site["segments"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| RowMeta {
            expert: s["expert"].as_u64().unwrap_or(u64::MAX) as usize,
            pass: s["pass"].as_u64().unwrap_or(0),
            span: s["span"]
                .as_array()
                .map(|a| a.iter().map(|v| v.as_str().unwrap().to_string()).collect())
                .unwrap_or_default(),
            pos: s["pos"]
                .as_array()
                .map(|a| a.iter().map(|v| v.as_u64().unwrap()).collect())
                .unwrap_or_default(),
            rank: s["rank"]
                .as_array()
                .map(|a| a.iter().map(|v| v.as_u64().unwrap()).collect())
                .unwrap_or_default(),
            weight: s["weight"]
                .as_array()
                .map(|a| a.iter().map(|v| v.as_f64().unwrap()).collect())
                .unwrap_or_default(),
            row0: s["row0"].as_u64().unwrap() as usize,
            rows: s["rows"].as_u64().unwrap() as usize,
        })
        .collect()
}

fn normalize_capture(check_dir: &Path, layer: usize, idx: &Value) -> ExpertCapture {
    let gu_site = &idx["sites"]["gate_up"];
    let dn_site = &idx["sites"]["down"];
    assert_eq!(gu_site["K"].as_u64().unwrap() as usize, H);
    assert_eq!(dn_site["K"].as_u64().unwrap() as usize, INTER);
    let gu_rows = read_f32_file(&check_dir.join(gu_site["file"].as_str().unwrap()));
    let dn_rows = read_f32_file(&check_dir.join(dn_site["file"].as_str().unwrap()));
    let gu_segs = site_segments(gu_site);
    let dn_segs = site_segments(dn_site);
    let mut keys: BTreeSet<(u64, String, u64)> = BTreeSet::new();
    for s in &gu_segs {
        assert_eq!(s.span.len(), s.rows);
        for i in 0..s.rows {
            keys.insert((s.pass, s.span[i].clone(), s.pos[i]));
        }
    }
    let tok_of: BTreeMap<(u64, String, u64), usize> = keys
        .iter()
        .cloned()
        .enumerate()
        .map(|(i, k)| (k, i))
        .collect();
    let t = tok_of.len();
    let mut cap = ExpertCapture {
        t,
        x_tok: vec![0.0; t * H],
        ids: vec![-1; t * TOPK],
        w: vec![0.0; t * TOPK],
        act_slot: vec![0.0; t * TOPK * INTER],
        x_mismatch_rows: 0,
    };
    let mut x_set = vec![false; t];
    for s in &gu_segs {
        for i in 0..s.rows {
            let tok = tok_of[&(s.pass, s.span[i].clone(), s.pos[i])];
            let r = s.rank[i] as usize;
            assert!(r < TOPK, "rank {r}");
            let slot = tok * TOPK + r;
            assert!(
                cap.ids[slot] < 0,
                "L{layer}: duplicate (token, rank) {tok},{r}"
            );
            cap.ids[slot] = s.expert as i32;
            cap.w[slot] = s.weight[i] as f32;
            let row = &gu_rows[(s.row0 + i) * H..(s.row0 + i + 1) * H];
            if !x_set[tok] {
                cap.x_tok[tok * H..(tok + 1) * H].copy_from_slice(row);
                x_set[tok] = true;
            } else if cap.x_tok[tok * H..(tok + 1) * H]
                .iter()
                .zip(row)
                .any(|(a, b)| a.to_bits() != b.to_bits())
            {
                cap.x_mismatch_rows += 1;
            }
        }
    }
    assert!(
        cap.ids.iter().all(|&e| e >= 0),
        "L{layer}: some token has fewer than 8 captured experts"
    );
    for s in &dn_segs {
        for i in 0..s.rows {
            let tok = *tok_of
                .get(&(s.pass, s.span[i].clone(), s.pos[i]))
                .expect("down row of an unknown token");
            let slot = tok * TOPK + s.rank[i] as usize;
            assert_eq!(
                cap.ids[slot] as usize, s.expert,
                "down/gate_up expert mismatch"
            );
            cap.act_slot[slot * INTER..(slot + 1) * INTER]
                .copy_from_slice(&dn_rows[(s.row0 + i) * INTER..(s.row0 + i + 1) * INTER]);
        }
    }
    cap
}

// ───────────────────────── CPU teacher (F64) ─────────────────────────

/// Weight provider of the teacher / floor passes.
trait WeightSrc {
    /// Named weight as `[m, k]` row-major F32 (the logical, un-rotated, AWQ-divided matrix).
    fn get(&self, name: &str) -> (Cow<'_, [f32]>, usize, usize);
}

/// srcdump F32 file + index (`L{LL}.f32` / `L{LL}.json`).
struct SrcDump {
    map: memmap2::Mmap,
    index: HashMap<String, (usize, usize, usize)>,
    meta: Value,
}

impl SrcDump {
    fn open(dir: &Path, layer: usize) -> Self {
        let meta = read_json(&dir.join(format!("L{layer:02}.json")));
        let f = ok(
            std::fs::File::open(dir.join(format!("L{layer:02}.f32"))),
            "src f32",
        );
        let map = unsafe { ok(memmap2::Mmap::map(&f), "mmap src") };
        let mut index = HashMap::new();
        for t in meta["tensors"].as_array().unwrap() {
            let sh = t["shape"].as_array().unwrap();
            let (m, k) = if sh.len() == 2 {
                (
                    sh[0].as_u64().unwrap() as usize,
                    sh[1].as_u64().unwrap() as usize,
                )
            } else {
                (1, sh[0].as_u64().unwrap() as usize)
            };
            index.insert(
                t["name"].as_str().unwrap().to_string(),
                (t["offset_bytes"].as_u64().unwrap() as usize, m, k),
            );
        }
        Self { map, index, meta }
    }
}

impl WeightSrc for SrcDump {
    fn get(&self, name: &str) -> (Cow<'_, [f32]>, usize, usize) {
        let &(off, m, k) = self
            .index
            .get(name)
            .unwrap_or_else(|| die(format!("src tensor {name}")));
        assert!(off % 4 == 0);
        let p = unsafe { (self.map.as_ptr() as *const u8).add(off) as *const f32 };
        (
            Cow::Borrowed(unsafe { std::slice::from_raw_parts(p, m * k) }),
            m,
            k,
        )
    }
}

/// Artifact-effective weights: decode + (AWQ divide per column) + crop the down pad. The
/// quantization floor of an exact-arithmetic route (no activation quantization).
struct ArtifactWeights {
    hfq: HfqFile,
    xts: bool,
    a_down_rt: Option<PathBuf>,
    layer: usize,
}

impl ArtifactWeights {
    fn decode(&self, name: &str, m: usize, k: usize) -> Vec<f32> {
        let (info, data) = self
            .hfq
            .tensor_data(name)
            .unwrap_or_else(|| die(format!("artifact tensor {name}")));
        match info.quant_type {
            44 | 13 => dequantize_weight_to_f32(info.quant_type, data, m * k),
            1 => data
                .chunks_exact(2)
                .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect(),
            q => die(format!("{name}: unsupported qt {q}")),
        }
    }

    fn sidecar(&self, name: &str, k: usize) -> Option<Vec<f32>> {
        let sn = awq_sidecar_name(name);
        let (_, d) = self.hfq.tensor_data(&sn)?;
        assert_eq!(d.len(), 2 * k, "{sn}");
        Some(
            d.chunks_exact(2)
                .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect(),
        )
    }
}

impl WeightSrc for ArtifactWeights {
    fn get(&self, name: &str) -> (Cow<'_, [f32]>, usize, usize) {
        // Logical shapes by name.
        let (m, k_log) =
            if name.ends_with("attn.wq_du.weight") || name.ends_with("attn.gate_attn.weight") {
                (EATTN, H)
            } else if name.ends_with("attn.wk_dv.weight") || name.ends_with("attn.wv_dv.weight") {
                (1024, H)
            } else if name.ends_with("attn.wr_du.weight") {
                (384, H)
            } else if name.ends_with("attn.wo_ud.weight") {
                (H, EATTN)
            } else if name.ends_with("mlp.gate.weight") {
                (E, H)
            } else if name.ends_with("gate_up_proj.weight") {
                (2 * INTER, H)
            } else if name.ends_with("down_proj.weight") {
                (H, INTER)
            } else {
                die(format!("artifact floor: unknown tensor {name}"))
            };
        let is_down = name.ends_with("down_proj.weight");
        if is_down && !self.xts {
            // A: the GPU multiplies the runtime-repacked (pad 1280, per-128 asymmetric) down.
            let dir = self
                .a_down_rt
                .as_ref()
                .unwrap_or_else(|| die("A down floor needs --a-down-rt"));
            let e: usize = name.split('.').nth(6).unwrap().parse().unwrap();
            let per = H * DK / 256 * 136;
            let blob = ok(
                std::fs::read(dir.join(format!("L{}.a-down-rt.bin", self.layer))),
                "a-down-rt",
            );
            let full = dequantize_weight_to_f32(44, &blob[e * per..(e + 1) * per], H * DK);
            let mut out = Vec::with_capacity(H * INTER);
            for r in 0..H {
                out.extend_from_slice(&full[r * DK..r * DK + INTER]);
            }
            return (Cow::Owned(out), m, k_log);
        }
        let k_phys = if is_down { DK } else { k_log };
        let full = self.decode(name, m, k_phys);
        let s = if self.xts {
            self.sidecar(name, k_phys)
        } else {
            None
        };
        let mut out = Vec::with_capacity(m * k_log);
        for r in 0..m {
            let row = &full[r * k_phys..r * k_phys + k_log];
            match &s {
                Some(s) => out.extend(row.iter().zip(&s[..k_log]).map(|(w, s)| w / s)),
                None => out.extend_from_slice(row),
            }
        }
        (Cow::Owned(out), m, k_log)
    }
}

/// `out[n, m] = x[n, k] . w[m, k]^T` in F64 (weights widened on the fly).
fn matmul_nt(x: &[f64], n: usize, k: usize, w: &[f32], m: usize) -> Vec<f64> {
    assert_eq!(x.len(), n * k);
    assert_eq!(w.len(), m * k);
    let mut out_t = vec![0f64; m * n];
    out_t
        .par_chunks_mut(n * 4)
        .enumerate()
        .for_each(|(blk, chunk)| {
            let j0 = blk * 4;
            let nr = chunk.len() / n;
            let mut wr = vec![0f64; 4 * k];
            for r in 0..nr {
                for (d, s) in wr[r * k..(r + 1) * k]
                    .iter_mut()
                    .zip(&w[(j0 + r) * k..(j0 + r + 1) * k])
                {
                    *d = *s as f64;
                }
            }
            for i in 0..n {
                let xr = &x[i * k..(i + 1) * k];
                let mut acc = [[0f64; 2]; 4];
                let mut t = 0;
                while t + 1 < k {
                    let (x0, x1) = (xr[t], xr[t + 1]);
                    for r in 0..4 {
                        acc[r][0] += x0 * wr[r * k + t];
                        acc[r][1] += x1 * wr[r * k + t + 1];
                    }
                    t += 2;
                }
                if t < k {
                    for r in 0..4 {
                        acc[r][0] += xr[t] * wr[r * k + t];
                    }
                }
                for r in 0..nr {
                    chunk[r * n + i] = acc[r][0] + acc[r][1];
                }
            }
        });
    let mut out = vec![0f64; n * m];
    for j in 0..m {
        for i in 0..n {
            out[i * m + j] = out_t[j * n + i];
        }
    }
    out
}

fn to_f64(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| x as f64).collect()
}

fn to_f32(v: &[f64]) -> Vec<f32> {
    v.iter().map(|&x| x as f32).collect()
}

fn situ64(g: f64, u: f64, gb: f64, vb: f64) -> f64 {
    let sig = 1.0 / (1.0 + (-g).exp());
    (gb * (g / gb).tanh() * sig) * (if vb > 0.0 { vb * (u / vb).tanh() } else { u })
}

fn teacher_layer(a: &[String], layer: usize, weights: &dyn WeightSrc, src_meta: &Value) {
    let t0 = std::time::Instant::now();
    let check_dir = PathBuf::from(arg(a, "--check-dir", ""));
    let out_base = PathBuf::from(arg(a, "--out", ""));
    let out = layer_dir(&out_base, layer);
    ok(std::fs::create_dir_all(&out), "mkdir out");
    let model = arg(
        a,
        "--model",
        "/home/kaden/.hipfire/models/inkling2/inkling2-mini-mq4.hfq",
    );
    let hfq = ok(HfqFile::open(Path::new(&model)), "open model for config");
    let cfg = ok(
        Inkling2Config::from_metadata_json(&hfq.metadata_json),
        "config",
    );
    let (gb, vb) = (cfg.situ_beta as f64, cfg.situ_linear_beta as f64);
    let idx = read_json(&check_dir.join(format!("L{layer:02}.index.json")));
    let p = format!("model.llm.layers.{layer}.");

    // Dense linears: (tag, tensor, input site).
    let mut dense_meta = serde_json::Map::new();
    for (tag, wname, site) in [
        ("q", "attn.wq_du.weight", "q"),
        ("k", "attn.wk_dv.weight", "k"),
        ("v", "attn.wv_dv.weight", "v"),
        ("wr", "attn.wr_du.weight", "q"),
        ("gate", "attn.gate_attn.weight", "q"),
        ("o", "attn.wo_ud.weight", "attn_out"),
    ] {
        let s = &idx["sites"][site];
        let (rows, kk) = (
            s["rows"].as_u64().unwrap() as usize,
            s["K"].as_u64().unwrap() as usize,
        );
        let x = to_f64(&read_f32_file(&check_dir.join(s["file"].as_str().unwrap())));
        let (w, m, k) = weights.get(&format!("{p}{wname}"));
        assert_eq!(k, kk, "{wname} K");
        let y = matmul_nt(&x, rows, k, &w, m);
        write_f32_file(&out.join(format!("y_{tag}.f32")), &to_f32(&y));
        dense_meta.insert(
            tag.into(),
            json!({"rows": rows, "m": m, "k": k, "site": site}),
        );
        eprintln!(
            "L{layer} dense {tag}: rows {rows} m {m} k {k} ({:.1}s)",
            t0.elapsed().as_secs_f64()
        );
    }

    // Routed experts.
    let cap = normalize_capture(&check_dir, layer, &idx);
    let slots = cap.t * TOPK;
    write_f32_file(&out.join("x_tok.f32"), &cap.x_tok);
    write_i32_file(&out.join("ids.i32"), &cap.ids);
    write_f32_file(&out.join("w.f32"), &cap.w);
    write_f32_file(&out.join("act_slot.f32"), &cap.act_slot);

    // Router logits (linear part only): x_tok through mlp.gate.weight.
    {
        let (w, m, k) = weights.get(&format!("{p}mlp.gate.weight"));
        let y = matmul_nt(&to_f64(&cap.x_tok), cap.t, k, &w, m);
        write_f32_file(&out.join("y_router.f32"), &to_f32(&y));
    }

    let mut by_expert: Vec<Vec<usize>> = vec![Vec::new(); E];
    for (s, &e) in cap.ids.iter().enumerate() {
        by_expert[e as usize].push(s);
    }
    let mut y_gu = vec![0f32; slots * 2 * INTER];
    let mut y_dn = vec![0f32; slots * H];
    let mut d_chain = vec![0f64; slots * H];
    let (mut div_num, mut div_den) = (0f64, 0f64);
    for e in 0..E {
        let sl = &by_expert[e];
        if sl.is_empty() {
            continue;
        }
        let n = sl.len();
        let (wgu, m1, k1) = weights.get(&format!("{p}mlp.experts.{e}.gate_up_proj.weight"));
        assert_eq!((m1, k1), (2 * INTER, H));
        let (wdn, m2, k2) = weights.get(&format!("{p}mlp.experts.{e}.down_proj.weight"));
        assert_eq!((m2, k2), (H, INTER));
        let mut xe = Vec::with_capacity(n * H);
        let mut af = Vec::with_capacity(n * INTER);
        for &s in sl {
            xe.extend(
                cap.x_tok[(s / TOPK) * H..(s / TOPK + 1) * H]
                    .iter()
                    .map(|&v| v as f64),
            );
            af.extend(
                cap.act_slot[s * INTER..(s + 1) * INTER]
                    .iter()
                    .map(|&v| v as f64),
            );
        }
        let gu = matmul_nt(&xe, n, H, &wgu, 2 * INTER);
        let mut act = vec![0f64; n * INTER];
        for i in 0..n {
            for c in 0..INTER {
                act[i * INTER + c] =
                    situ64(gu[i * 2 * INTER + c], gu[i * 2 * INTER + INTER + c], gb, vb);
            }
        }
        for (i, &s) in sl.iter().enumerate() {
            for c in 0..2 * INTER {
                y_gu[s * 2 * INTER + c] = gu[i * 2 * INTER + c] as f32;
            }
        }
        let dc = matmul_nt(&act, n, INTER, &wdn, H);
        let df = matmul_nt(&af, n, INTER, &wdn, H);
        for (i, &s) in sl.iter().enumerate() {
            d_chain[s * H..(s + 1) * H].copy_from_slice(&dc[i * H..(i + 1) * H]);
            for c in 0..H {
                y_dn[s * H + c] = df[i * H + c] as f32;
            }
        }
        for i in 0..n * INTER {
            div_num += (act[i] - af[i]).powi(2);
            div_den += act[i] * act[i];
        }
        if e % 16 == 0 {
            eprintln!(
                "L{layer} expert {e}/{E}: slots {n} ({:.1}s)",
                t0.elapsed().as_secs_f64()
            );
        }
    }
    let mut blk = vec![0f32; cap.t * H];
    for t in 0..cap.t {
        let mut acc = vec![0f64; H];
        for r in 0..TOPK {
            let s = t * TOPK + r;
            let w = cap.w[s] as f64;
            for c in 0..H {
                acc[c] += w * d_chain[s * H + c];
            }
        }
        for c in 0..H {
            blk[t * H + c] = acc[c] as f32;
        }
    }
    write_f32_file(&out.join("y_gu.f32"), &y_gu);
    write_f32_file(&out.join("y_dn.f32"), &y_dn);
    write_f32_file(&out.join("y_blk.f32"), &blk);
    let meta = json!({
        "schema": "hipfire.inkling2.xts_route_teacher.v1",
        "layer": layer,
        "weights": arg(a, "--weights", "src"),
        "formats": src_meta["formats"],
        "src_sha256": src_meta["sha256"],
        "check_index_sha256": cmd_out("sha256sum", &[check_dir.join(format!("L{layer:02}.index.json")).to_str().unwrap()]),
        "tokens": cap.t, "slots": slots,
        "x_mismatch_rows": cap.x_mismatch_rows,
        "chain_act_vs_captured_act_rel_rms": (div_num / div_den).sqrt(),
        "situ": {"gate_beta": gb, "value_beta": vb},
        "dense": dense_meta,
        "seconds": t0.elapsed().as_secs_f64(),
        "binary_md5": exe_md5(),
    });
    ok(
        std::fs::write(
            out.join("meta.json"),
            serde_json::to_vec_pretty(&meta).unwrap(),
        ),
        "meta",
    );
    eprintln!(
        "L{layer} teacher done in {:.1}s; chain/captured act rel {:.3e}",
        t0.elapsed().as_secs_f64(),
        (div_num / div_den).sqrt()
    );
}

/// Strided row subset of the unembed (rows `i * V / n`).
fn unembed_rows(v: usize, n: usize) -> Vec<usize> {
    (0..n).map(|i| i * v / n).collect()
}

fn teacher_unembed(a: &[String]) {
    let st = arg(
        a,
        "--src-st",
        "/home/kaden/qcal/release-0.5.0/inkling/teacher/source/model-00001-of-00001.safetensors",
    );
    let check_dir = PathBuf::from(arg(a, "--check-dir", ""));
    let out = PathBuf::from(arg(a, "--out", "")).join("unembed");
    ok(std::fs::create_dir_all(&out), "mkdir");
    let nrows: usize = arg(a, "--rows", "8192").parse().unwrap();
    let nin: usize = arg(a, "--inputs", "1024").parse().unwrap();
    let rows = unembed_rows(201024, nrows);
    let idx = read_json(&check_dir.join("L20.index.json"));
    let q = &idx["sites"]["q"];
    let x_all = read_f32_file(&check_dir.join(q["file"].as_str().unwrap()));
    let x = to_f64(&x_all[..nin * H]);
    // Weights: the BF16 source (exact), or the artifact's decoded rows.
    let wsel = arg(a, "--weights", "src");
    let mut w = vec![0f32; nrows * H];
    if let Some(path) = wsel.strip_prefix("artifact:") {
        let hfq = ok(HfqFile::open(Path::new(path)), "open artifact");
        let (info, data) = hfq.tensor_data("model.llm.unembed.weight").unwrap();
        assert_eq!(info.quant_type, 44);
        let stride = H / 256 * 136;
        for (i, &r) in rows.iter().enumerate() {
            let dq = dequantize_weight_to_f32(44, &data[r * stride..(r + 1) * stride], H);
            w[i * H..(i + 1) * H].copy_from_slice(&dq);
        }
    } else {
        let f = ok(std::fs::File::open(&st), "open safetensors");
        let map = unsafe { ok(memmap2::Mmap::map(&f), "mmap st") };
        let n = u64::from_le_bytes(map[..8].try_into().unwrap()) as usize;
        let hdr: Value = ok(serde_json::from_slice(&map[8..8 + n]), "st header");
        let t = &hdr["model.llm.unembed.weight"];
        assert_eq!(t["dtype"].as_str().unwrap(), "BF16");
        let off = t["data_offsets"][0].as_u64().unwrap() as usize + 8 + n;
        for (i, &r) in rows.iter().enumerate() {
            for c in 0..H {
                let p = off + (r * H + c) * 2;
                let b = u16::from_le_bytes([map[p], map[p + 1]]);
                w[i * H + c] = f32::from_bits((b as u32) << 16);
            }
        }
    }
    let y = matmul_nt(&x, nin, H, &w, nrows);
    write_f32_file(&out.join("y.f32"), &to_f32(&y));
    let meta = json!({
        "schema": "hipfire.inkling2.xts_route_teacher_unembed.v1",
        "weights": wsel, "rows": nrows, "inputs": nin,
        "row_indices_rule": "i * 201024 / rows",
        "input_proxy": "L20 check q-site rows (post-norm hidden-class rows; the real final-norm hidden is not captured)",
        "binary_md5": exe_md5(),
    });
    ok(
        std::fs::write(
            out.join("meta.json"),
            serde_json::to_vec_pretty(&meta).unwrap(),
        ),
        "meta",
    );
}

fn teacher_main(a: &[String]) {
    let layers = parse_layers(&arg(a, "--layers", "5,6,20"));
    let src_dir = PathBuf::from(arg(a, "--src-dir", ""));
    let wsel = arg(a, "--weights", "src");
    for layer in layers {
        let src = SrcDump::open(&src_dir, layer);
        if let Some(path) = wsel.strip_prefix("artifact:") {
            let hfq = ok(HfqFile::open(Path::new(path)), "open artifact");
            let xts = ok(
                Inkling2Config::from_metadata_json(&hfq.metadata_json),
                "cfg",
            )
            .mq4_layout
            .is_xts();
            let aw = ArtifactWeights {
                hfq,
                xts,
                a_down_rt: has(a, "--a-down-rt").then(|| PathBuf::from(arg(a, "--a-down-rt", ""))),
                layer,
            };
            teacher_layer(a, layer, &aw, &src.meta);
        } else {
            teacher_layer(a, layer, &src, &src.meta);
        }
    }
}

// ───────────────────────── metrics ─────────────────────────

#[derive(Clone)]
struct Metric {
    rows: usize,
    cols: usize,
    sum_err2: f64,
    sum_ref2: f64,
    rel_rms: f64,
    row_rel_p50: f64,
    row_rel_p99: f64,
    row_rel_max: f64,
    cos_min: f64,
    nonfinite: usize,
}

fn pct(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let pos = q * (sorted.len() - 1) as f64;
    let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
    sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)
}

fn metric(y: &[f32], t: &[f32], rows: usize, cols: usize) -> Metric {
    assert_eq!(y.len(), rows * cols, "route output size");
    assert_eq!(t.len(), rows * cols, "teacher size");
    struct R {
        e2: f64,
        r2: f64,
        yy: f64,
        yt: f64,
        bad: usize,
    }
    let per: Vec<R> = (0..rows)
        .into_par_iter()
        .map(|i| {
            let (mut e2, mut r2, mut yy, mut yt, mut bad) = (0f64, 0f64, 0f64, 0f64, 0usize);
            for c in 0..cols {
                let (a, b) = (y[i * cols + c] as f64, t[i * cols + c] as f64);
                if !a.is_finite() {
                    bad += 1;
                    continue;
                }
                e2 += (a - b) * (a - b);
                r2 += b * b;
                yy += a * a;
                yt += a * b;
            }
            R {
                e2,
                r2,
                yy,
                yt,
                bad,
            }
        })
        .collect();
    let (mut se, mut sr, mut bad) = (0f64, 0f64, 0usize);
    let mut rel = Vec::with_capacity(rows);
    let mut cos_min = f64::INFINITY;
    for r in &per {
        se += r.e2;
        sr += r.r2;
        bad += r.bad;
        if r.r2 > 0.0 {
            rel.push((r.e2 / r.r2).sqrt());
            if r.yy > 0.0 {
                cos_min = cos_min.min(r.yt / (r.yy.sqrt() * r.r2.sqrt()));
            } else {
                cos_min = cos_min.min(0.0);
            }
        }
    }
    rel.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Metric {
        rows,
        cols,
        sum_err2: se,
        sum_ref2: sr,
        rel_rms: if bad > 0 { f64::NAN } else { (se / sr).sqrt() },
        row_rel_p50: pct(&rel, 0.5),
        row_rel_p99: pct(&rel, 0.99),
        row_rel_max: rel.last().copied().unwrap_or(f64::NAN),
        cos_min,
        nonfinite: bad,
    }
}

/// rel RMS per expert over slot rows.
fn per_expert_rel(y: &[f32], t: &[f32], cols: usize, ids: &[i32]) -> Vec<Value> {
    let mut e2 = vec![0f64; E];
    let mut r2 = vec![0f64; E];
    for (s, &e) in ids.iter().enumerate() {
        for c in 0..cols {
            let (a, b) = (y[s * cols + c] as f64, t[s * cols + c] as f64);
            e2[e as usize] += (a - b) * (a - b);
            r2[e as usize] += b * b;
        }
    }
    (0..E)
        .map(|e| {
            if r2[e] > 0.0 {
                json!((e2[e] / r2[e]).sqrt())
            } else {
                Value::Null
            }
        })
        .collect()
}

fn metric_json(m: &Metric) -> Value {
    json!({
        "rows": m.rows, "cols": m.cols,
        "rel_rms": m.rel_rms, "row_rel_p50": m.row_rel_p50, "row_rel_p99": m.row_rel_p99,
        "row_rel_max": m.row_rel_max, "cos_min": m.cos_min,
        "sum_err2": m.sum_err2, "sum_ref2": m.sum_ref2, "nonfinite": m.nonfinite,
    })
}

// ───────────────────────── GPU harness ─────────────────────────

struct Ctx {
    gpu: Gpu,
    label: String,
    artifact_path: String,
    artifact_sha256: String,
    binary_md5: String,
    out: std::fs::File,
    xts: bool,
    sym_verdict: Option<bool>,
    cfg: Inkling2Config,
    default_iu4: Option<bool>,
    default_a8: bool,
    default_fp8_resid: bool,
    job: String,
    /// Negative control: replace every AWQ sidecar by ones (the producers still run).
    ablate_awq: bool,
    /// Row-dependence probe: use only the first N tokens / rows (usize::MAX = all).
    tok_limit: usize,
}

impl Ctx {
    fn rec(&mut self, layer: Option<usize>, fmt: &str, site: &str, route: &str, mut body: Value) {
        let o = body.as_object_mut().unwrap();
        o.insert("schema".into(), json!(SCHEMA));
        o.insert("artifact".into(), json!(self.label));
        o.insert("artifact_path".into(), json!(self.artifact_path));
        o.insert("artifact_sha256".into(), json!(self.artifact_sha256));
        o.insert("layer".into(), json!(layer));
        o.insert("source_format".into(), json!(fmt));
        o.insert("site".into(), json!(site));
        o.insert("route".into(), json!(route));
        o.insert("binary_md5".into(), json!(self.binary_md5));
        o.insert("fabric_job".into(), json!(self.job));
        o.insert("arch".into(), json!(self.gpu.arch));
        o.insert(
            "mq4v2_symmetric_flag".into(),
            json!(self.gpu.mq4v2_symmetric),
        );
        writeln!(self.out, "{}", body).unwrap();
        self.out.flush().unwrap();
        let summ = body["rel_rms"]
            .as_f64()
            .map(|v| format!("rel_rms {v:.5}"))
            .unwrap_or_else(|| body["status"].to_string());
        eprintln!("  L{:?} {site:<18} {route:<16} {summ}", layer);
    }

    fn scored(
        &mut self,
        layer: usize,
        fmt: &str,
        site: &str,
        route: &str,
        y: &[f32],
        t: &[f32],
        rows: usize,
        cols: usize,
        extra: Value,
        ids: Option<&[i32]>,
        ms: f64,
    ) {
        let m = metric(y, t, rows, cols);
        let mut body = metric_json(&m);
        let o = body.as_object_mut().unwrap();
        o.insert("status".into(), json!("measured"));
        o.insert("ms_wall".into(), json!(ms));
        if let Some(ids) = ids {
            o.insert(
                "per_expert_rel_rms".into(),
                json!(per_expert_rel(y, t, cols, ids)),
            );
        }
        o.insert("detail".into(), extra);
        self.rec(Some(layer), fmt, site, route, body);
    }

    fn unscored(
        &mut self,
        layer: Option<usize>,
        fmt: &str,
        site: &str,
        route: &str,
        status: &str,
        why: &str,
    ) {
        self.rec(
            layer,
            fmt,
            site,
            route,
            json!({"status": status, "why": why}),
        );
    }

    fn set_flags(&mut self, iu4: Option<bool>, a8: bool, fp8_resid: bool) {
        let f = std::sync::Arc::make_mut(&mut self.gpu.flags);
        f.iu4_prefill = iu4;
        f.a8_prefill = a8;
        f.gfx12_mq4v2_fp8_resid = fp8_resid;
    }

    fn restore_flags(&mut self) {
        let (a, b, c) = (self.default_iu4, self.default_a8, self.default_fp8_resid);
        self.set_flags(a, b, c);
    }

    fn sync(&mut self) {
        ok(self.gpu.hip.device_synchronize(), "sync");
    }

    fn up_f32(&mut self, v: &[f32], shape: &[usize]) -> GpuTensor {
        ok(self.gpu.upload_f32(v, shape), "upload_f32")
    }

    fn up_i32(&mut self, v: &[i32], shape: &[usize]) -> GpuTensor {
        let b = f32_bits_of_i32(v);
        ok(self.gpu.upload_f32(&b, shape), "upload_i32")
    }

    fn zeros(&mut self, shape: &[usize]) -> GpuTensor {
        ok(self.gpu.zeros(shape, DType::F32), "zeros")
    }

    fn dl(&mut self, t: &GpuTensor) -> Vec<f32> {
        self.sync();
        ok(self.gpu.download_f32(t), "download")
    }

    fn free(&mut self, t: GpuTensor) {
        ok(self.gpu.free_tensor(t), "free");
    }
}

/// Whether every group header of a qt44 tensor is symmetric (`d >= +0`, `z == f16(-8 d)`).
fn qt44_symmetric(data: &[u8]) -> bool {
    data.chunks_exact(136).all(|g| {
        (0..2).all(|h| {
            let d = u16::from_le_bytes([g[4 * h], g[4 * h + 1]]);
            let z = u16::from_le_bytes([g[4 * h + 2], g[4 * h + 3]]);
            let df = f16::from_bits(d).to_f32();
            let zero_ok = if df == 0.0 {
                z == 0x8000 || z == 0x0000
            } else {
                z == f16::from_f32(-8.0 * df).to_bits()
            };
            d & 0x8000 == 0 && zero_ok
        })
    })
}

fn tensor_bytes<'a>(
    hfq: &'a HfqFile,
    name: &str,
) -> (&'a hipfire_runtime::hfq::HfqTensorInfo, &'a [u8]) {
    hfq.tensor_data(name)
        .unwrap_or_else(|| die(format!("artifact has no tensor {name}")))
}

/// Dense linear of the artifact on the device.
struct DenseW {
    w: WeightTensor,
    sym: bool,
    qt: u8,
    awq: bool,
}

fn upload_dense(
    ctx: &mut Ctx,
    hfq: &HfqFile,
    name: &str,
    m: usize,
    k: usize,
    sym_all: &mut bool,
) -> DenseW {
    let (info, data) = tensor_bytes(hfq, name);
    let qt = info.quant_type;
    let shape: Vec<usize> = info.shape.iter().map(|&d| d as usize).collect();
    assert_eq!(shape, [m, k], "{name} shape");
    let (dtype, stride) = match qt {
        44 => (DType::MQ4G256V2, k / 256 * 136),
        1 => (DType::F16, 2 * k),
        q => die(format!("{name}: dense qt {q} not handled")),
    };
    assert_eq!(data.len(), stride * m, "{name} bytes");
    let sym = qt == 44 && qt44_symmetric(data);
    if qt == 44 {
        *sym_all &= sym;
    }
    let buf = ok(ctx.gpu.upload_raw(data, &[data.len()]), "upload weight");
    let mut awq = None;
    if ctx.xts && qt == 44 {
        let sn = awq_sidecar_name(name);
        let (si, sd) = tensor_bytes(hfq, &sn);
        assert_eq!(si.quant_type, 1);
        assert_eq!(sd.len(), 2 * k, "{sn}");
        let mut s: Vec<f32> = sd
            .chunks_exact(2)
            .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect();
        if ctx.ablate_awq {
            s.iter_mut().for_each(|v| *v = 1.0);
        }
        awq = Some(ctx.up_f32(&s, &[k]));
    }
    let has_awq = awq.is_some();
    DenseW {
        w: WeightTensor {
            buf,
            gpu_dtype: dtype,
            m,
            k,
            row_stride: stride,
            awq_scale: awq,
            lloyd_lut_e4m3: None,
            lloyd_lut_f16: None,
            lloyd_lut_c16: None,
        },
        sym,
        qt,
        awq: has_awq,
    }
}

fn free_dense(ctx: &mut Ctx, d: DenseW) {
    ctx.free(d.w.buf);
    if let Some(t) = d.w.awq_scale {
        ctx.free(t);
    }
}

/// Run every dense route of one linear on rows `x` `[n, k]` and score against `teacher`.
fn dense_routes(
    ctx: &mut Ctx,
    layer: usize,
    fmt: &str,
    site: &str,
    d: &DenseW,
    x: &[f32],
    n: usize,
    teacher: &[f32],
) {
    let (m, k) = (d.w.m, d.w.k);
    let xd = ctx.up_f32(x, &[n, k]);
    let xrot = ctx.zeros(&[n, k]);
    let mut routes: Vec<(&str, Option<bool>, bool, bool)> = Vec::new();
    if d.qt == 44 {
        routes.push(("iu4_prefill", ctx.default_iu4, false, ctx.default_fp8_resid));
        if ctx.gpu.mq4v2_symmetric {
            routes.push(("a8_prefill", ctx.default_iu4, true, ctx.default_fp8_resid));
        }
        routes.push(("iu4_off_fp8", Some(false), false, true));
        routes.push(("iu4_off_f16", Some(false), false, false));
    }
    // native: decode GEMV per row (production `weight_gemv`, AWQ via the attached sidecar).
    {
        let y = ctx.zeros(&[n, m]);
        ctx.restore_flags();
        let t0 = std::time::Instant::now();
        for r in 0..n {
            ok(
                weight_gemv(
                    &mut ctx.gpu,
                    &d.w,
                    &xd.sub_offset(r * k, k),
                    &y.sub_offset(r * m, m),
                ),
                "weight_gemv",
            );
        }
        ctx.sync();
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        let yh = ctx.dl(&y);
        ctx.scored(
            layer,
            fmt,
            site,
            "native_gemv",
            &yh,
            teacher,
            n,
            m,
            json!({"launcher": "hipfire_runtime::llama::weight_gemv per row", "awq": d.awq, "weight_headers_symmetric": d.sym}),
            None,
            ms,
        );
        ctx.free(y);
    }
    if d.qt == 44 && !ctx.gpu.mq4v2_symmetric {
        ctx.unscored(Some(layer), fmt, site, "a8_prefill", "not_applicable", "a8_prefill_active needs gpu.mq4v2_symmetric; the legacy asymmetric qt44 artifact does not set it");
    }
    for (route, iu4, a8, fp8r) in routes {
        ctx.set_flags(iu4, a8, fp8r);
        let dispatch_shared_iu4 = ctx.gpu.mq4g256v2_shared_iu4_supported(k, n);
        let a8_active = ctx.gpu.a8_prefill_active(n, k);
        let want_iu4 = route == "iu4_prefill";
        let want_a8 = route == "a8_prefill";
        if want_iu4 && !dispatch_shared_iu4 {
            ctx.restore_flags();
            ctx.unscored(
                Some(layer),
                fmt,
                site,
                route,
                "not_admitted",
                "mq4g256v2_shared_iu4_supported false for this call",
            );
            continue;
        }
        if want_a8 && !a8_active {
            ctx.restore_flags();
            ctx.unscored(
                Some(layer),
                fmt,
                site,
                route,
                "not_admitted",
                "a8_prefill_active false (needs gpu.mq4v2_symmetric, HIPFIRE_A8_PREFILL)",
            );
            continue;
        }
        if !want_iu4 && !want_a8 && (dispatch_shared_iu4 || a8_active) {
            ctx.restore_flags();
            ctx.unscored(
                Some(layer),
                fmt,
                site,
                route,
                "route_not_isolated",
                "IU4/A8 still admitted with iu4_prefill=0",
            );
            continue;
        }
        // Producer: AWQ divide + FWHT (rotate_x_mq_awq_batched) or plain rotate for A.
        match &d.w.awq_scale {
            Some(s) => ok(
                ctx.gpu.rotate_x_mq_awq_batched(&xd, s, &xrot, k, n),
                "rotate_x_mq_awq_batched",
            ),
            None => ok(
                ctx.gpu.rotate_x_mq_batched(&xd, &xrot, k, n),
                "rotate_x_mq_batched",
            ),
        }
        let y = ctx.zeros(&[n, m]);
        let t0 = std::time::Instant::now();
        ok(
            ctx.gpu.gemm_mq4g256v2(&d.w.buf, &xrot, &y, m, k, n),
            "gemm_mq4g256v2",
        );
        ctx.sync();
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        let yh = ctx.dl(&y);
        ctx.scored(
            layer,
            fmt,
            site,
            route,
            &yh,
            teacher,
            n,
            m,
            json!({
                "launcher": "Gpu::gemm_mq4g256v2 -> gemm_hfq4g256_residual_wmma_gfx12_mq4v2",
                "flags": {"iu4_prefill": iu4, "a8_prefill": a8, "gfx12_mq4v2_fp8_resid": fp8r},
                "shared_iu4_supported": dispatch_shared_iu4, "a8_prefill_active": a8_active,
            }),
            None,
            ms,
        );
        ctx.free(y);
        ctx.restore_flags();
    }
    ctx.free(xd);
    ctx.free(xrot);
}

// ── experts ──

struct ExpertW {
    gu_ptrs: GpuTensor,
    dn_ptrs: GpuTensor,
    gu_awq_ptrs: Option<GpuTensor>,
    dn_awq_ptrs: Option<GpuTensor>,
    bufs: Vec<GpuTensor>,
    sym: bool,
}

fn upload_ptr_table(ctx: &mut Ctx, bufs: &[&GpuTensor]) -> GpuTensor {
    let mut p = Vec::new();
    for b in bufs {
        p.extend_from_slice(&(b.buf.as_ptr() as u64).to_ne_bytes());
    }
    ok(ctx.gpu.upload_raw(&p, &[p.len()]), "ptr table")
}

fn upload_experts(ctx: &mut Ctx, hfq: &HfqFile, layer: usize, a_down_rt: &Path) -> ExpertW {
    let p = format!("model.llm.layers.{layer}.");
    let mut bufs: Vec<GpuTensor> = Vec::new();
    let mut gu_idx = Vec::new();
    let mut dn_idx = Vec::new();
    let mut sym = true;
    let mut gu_awq: Vec<f32> = Vec::new();
    let mut dn_awq: Vec<f32> = Vec::new();
    let a_blob = if ctx.xts {
        Vec::new()
    } else {
        ok(
            std::fs::read(a_down_rt.join(format!("L{layer}.a-down-rt.bin"))),
            "a-down-rt blob",
        )
    };
    let per_dn = H * DK / 256 * 136;
    if !ctx.xts {
        assert_eq!(a_blob.len(), per_dn * E, "a-down-rt size");
    }
    for e in 0..E {
        let gname = format!("{p}mlp.experts.{e}.gate_up_proj.weight");
        let dname = format!("{p}mlp.experts.{e}.down_proj.weight");
        let (gi, gd) = tensor_bytes(hfq, &gname);
        let want_qt = if ctx.xts { 44 } else { 13 };
        assert_eq!(gi.quant_type, want_qt, "{gname} qt");
        assert_eq!(gd.len(), H * H / 256 * 136, "{gname} bytes");
        if ctx.xts {
            sym &= qt44_symmetric(gd);
        }
        bufs.push(ok(ctx.gpu.upload_raw(gd, &[gd.len()]), "gu"));
        gu_idx.push(bufs.len() - 1);
        if ctx.xts {
            let (di, dd) = tensor_bytes(hfq, &dname);
            assert_eq!(di.quant_type, 44);
            assert_eq!(dd.len(), per_dn, "{dname} bytes");
            sym &= qt44_symmetric(dd);
            bufs.push(ok(ctx.gpu.upload_raw(dd, &[dd.len()]), "dn"));
            for (nm, k, dst) in [(&gname, H, &mut gu_awq), (&dname, DK, &mut dn_awq)] {
                let (si, sd) = tensor_bytes(hfq, &awq_sidecar_name(nm));
                assert_eq!(si.quant_type, 1);
                assert_eq!(sd.len(), 2 * k);
                dst.extend(
                    sd.chunks_exact(2)
                        .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()),
                );
            }
        } else {
            let blob = &a_blob[e * per_dn..(e + 1) * per_dn];
            bufs.push(ok(ctx.gpu.upload_raw(blob, &[blob.len()]), "dn"));
        }
        dn_idx.push(bufs.len() - 1);
    }
    if ctx.ablate_awq {
        gu_awq.iter_mut().for_each(|v| *v = 1.0);
        dn_awq.iter_mut().for_each(|v| *v = 1.0);
    }
    let gu_refs: Vec<&GpuTensor> = gu_idx.iter().map(|&i| &bufs[i]).collect();
    let dn_refs: Vec<&GpuTensor> = dn_idx.iter().map(|&i| &bufs[i]).collect();
    let gu_ptrs = upload_ptr_table(ctx, &gu_refs);
    let dn_ptrs = upload_ptr_table(ctx, &dn_refs);
    let (gu_awq_ptrs, dn_awq_ptrs) = if ctx.xts {
        let gt = ctx.up_f32(&gu_awq, &[E, H]);
        let dt = ctx.up_f32(&dn_awq, &[E, DK]);
        let gp: Vec<u8> = (0..E)
            .flat_map(|e| (gt.buf.as_ptr() as u64 + (e * H * 4) as u64).to_ne_bytes())
            .collect();
        let dp: Vec<u8> = (0..E)
            .flat_map(|e| (dt.buf.as_ptr() as u64 + (e * DK * 4) as u64).to_ne_bytes())
            .collect();
        let gpt = ok(ctx.gpu.upload_raw(&gp, &[gp.len()]), "awq ptrs");
        let dpt = ok(ctx.gpu.upload_raw(&dp, &[dp.len()]), "awq ptrs");
        bufs.push(gt);
        bufs.push(dt);
        (Some(gpt), Some(dpt))
    } else {
        (None, None)
    };
    ExpertW {
        gu_ptrs,
        dn_ptrs,
        gu_awq_ptrs,
        dn_awq_ptrs,
        bufs,
        sym,
    }
}

fn free_experts(ctx: &mut Ctx, ew: ExpertW) {
    for b in ew.bufs {
        ctx.free(b);
    }
    ctx.free(ew.gu_ptrs);
    ctx.free(ew.dn_ptrs);
    if let Some(t) = ew.gu_awq_ptrs {
        ctx.free(t);
    }
    if let Some(t) = ew.dn_awq_ptrs {
        ctx.free(t);
    }
}

/// Device copies of one layer's routing inputs and the scatter products.
struct Dev {
    t: usize,
    slots: usize,
    bound: usize,
    x: GpuTensor,
    ids: GpuTensor,
    w: GpuTensor,
    offsets: GpuTensor,
    sorted: GpuTensor,
    tiles: GpuTensor,
    inv: GpuTensor,
    inv_h: Vec<i32>,
}

fn make_dev(ctx: &mut Ctx, x_tok: &[f32], ids: &[i32], w: &[f32]) -> Dev {
    let t = x_tok.len() / H;
    let slots = t * TOPK;
    let live = slots.min(E);
    let bound = (slots + live * 15).div_ceil(16) * 16;
    let x = ctx.up_f32(x_tok, &[t, H]);
    let idt = ctx.up_i32(ids, &[slots]);
    let wt = ctx.up_f32(w, &[slots]);
    let counts = ctx.zeros(&[E]);
    let offsets = ctx.zeros(&[E + 1]);
    let sorted = ctx.zeros(&[bound]);
    let tiles = ctx.zeros(&[bound / 16]);
    let inv = ctx.zeros(&[slots]);
    ok(
        ctx.gpu.moe_scatter_fused_k8(
            &idt, &counts, &offsets, &sorted, &tiles, &inv, slots, E, bound, 16,
        ),
        "scatter",
    );
    ctx.sync();
    let inv_h: Vec<i32> = ctx.dl(&inv).iter().map(|v| v.to_bits() as i32).collect();
    ctx.free(counts);
    Dev {
        t,
        slots,
        bound,
        x,
        ids: idt,
        w: wt,
        offsets,
        sorted,
        tiles,
        inv,
        inv_h,
    }
}

fn free_dev(ctx: &mut Ctx, d: Dev) {
    for t in [d.x, d.ids, d.w, d.offsets, d.sorted, d.tiles, d.inv] {
        ctx.free(t);
    }
}

fn unscatter(y_grouped: &[f32], inv: &[i32], cols: usize) -> Vec<f32> {
    let mut out = vec![0f32; inv.len() * cols];
    for (s, &g) in inv.iter().enumerate() {
        assert!(g >= 0, "slot {s} unrouted");
        out[s * cols..(s + 1) * cols]
            .copy_from_slice(&y_grouped[g as usize * cols..(g as usize + 1) * cols]);
    }
    out
}

fn situ_cfg(cfg: &Inkling2Config) -> FfnActivation {
    FfnActivation::Situ {
        gate_beta: cfg.situ_beta,
        value_beta: cfg.situ_linear_beta,
    }
}

fn combine(
    ctx: &mut Ctx,
    down_rows: &GpuTensor,
    row_of_slot: Option<&GpuTensor>,
    dev: &Dev,
) -> GpuTensor {
    let out = ctx.zeros(&[dev.t, H]);
    ok(
        ctx.gpu.ffn_moe_combine_f32(&MoeCombine {
            order: CombineOrder::Rank,
            round_expert_outputs: false,
            round_routed_sum: false,
            round_final: false,
            down_rows,
            row_of_slot,
            topk_indices: &dev.ids,
            topk_weights: &dev.w,
            shared: None,
            routed_out: None,
            out: &out,
            rows: dev.t,
            hidden: H,
            top_k: TOPK,
            n_experts: E,
        }),
        "combine",
    );
    out
}

/// Per-slot gate_up input rows `FWHT(x_token / S_gu[e])`, `[slots, H]`.
fn gu_rows_xts(ctx: &mut Ctx, ew: &ExpertW, x: &GpuTensor, ids: &GpuTensor, t: usize) -> GpuTensor {
    let rows = ctx.zeros(&[t * TOPK, H]);
    let tbl = ew.gu_awq_ptrs.as_ref().unwrap();
    for i in 0..t {
        ok(
            ctx.gpu.rotate_x_mq_awq_topk(
                &x.sub_offset(i * H, H),
                tbl,
                &ids.sub_offset(i * TOPK, TOPK),
                &rows.sub_offset(i * TOPK * H, TOPK * H),
                H,
                TOPK,
                E,
            ),
            "rotate_x_mq_awq_topk",
        );
    }
    rows
}

/// Per-slot down input `FWHT(act_row / S_dn[e])` for host-provided activations `[slots, DK]`.
fn dn_rows_xts(
    ctx: &mut Ctx,
    ew: &ExpertW,
    act: &GpuTensor,
    ids: &GpuTensor,
    slots: usize,
) -> GpuTensor {
    let rows = ctx.zeros(&[slots, DK]);
    let tbl = ew.dn_awq_ptrs.as_ref().unwrap();
    for s in 0..slots {
        ok(
            ctx.gpu.rotate_x_mq_awq_topk(
                &act.sub_offset(s * DK, DK),
                tbl,
                &ids.sub_offset(s, 1),
                &rows.sub_offset(s * DK, DK),
                DK,
                1,
                E,
            ),
            "rotate_x_mq_awq_topk(down)",
        );
    }
    rows
}

/// XTS scalar decode chain over `t` tokens (per-row kernels; the token axis is only batched).
fn chain_scalar_xts(
    ctx: &mut Ctx,
    ew: &ExpertW,
    x: &GpuTensor,
    ids: &GpuTensor,
    w: &GpuTensor,
    t: usize,
) -> (GpuTensor, GpuTensor) {
    let slots = t * TOPK;
    let rows = gu_rows_xts(ctx, ew, x, ids, t);
    let gu = ctx.zeros(&[slots, 2 * INTER]);
    ok(
        ctx.gpu.gemv_mq4g256v2_moe_down_k8_indexed_batched_expanded(
            &ew.gu_ptrs,
            ids,
            &rows,
            &gu,
            2 * INTER,
            H,
            TOPK,
            t,
        ),
        "xts gate_up gemv",
    );
    let act = ctx.zeros(&[slots, DK]);
    ok(
        ctx.gpu.ffn_activation_rows_f32(&FfnActivationRows {
            activation: situ_cfg(&ctx.cfg),
            gate: &gu,
            gate_offset: 0,
            gate_ld: 2 * INTER,
            up: &gu,
            up_offset: INTER,
            up_ld: 2 * INTER,
            out: &act,
            rows: slots,
            logical_n: INTER,
            phys_n: DK,
            round_inputs: false,
            round_output: false,
            awq: FfnAwq::PerExpert {
                table: ew.dn_awq_ptrs.as_ref().unwrap(),
                row_expert: ids,
                n_experts: E,
            },
        }),
        "situ rows",
    );
    let rot = ctx.zeros(&[slots, DK]);
    ok(
        ctx.gpu.rotate_x_mq_batched(&act, &rot, DK, slots),
        "rotate 1280",
    );
    let dexp = ctx.zeros(&[slots, H]);
    ok(
        ctx.gpu.gemv_mq4g256v2_moe_down_k8_indexed_batched_expanded(
            &ew.dn_ptrs,
            ids,
            &rot,
            &dexp,
            H,
            DK,
            TOPK,
            t,
        ),
        "xts down gemv",
    );
    let out = ctx.zeros(&[t, H]);
    ok(
        ctx.gpu.ffn_moe_combine_f32(&MoeCombine {
            order: CombineOrder::Rank,
            round_expert_outputs: false,
            round_routed_sum: false,
            round_final: false,
            down_rows: &dexp,
            row_of_slot: None,
            topk_indices: ids,
            topk_weights: w,
            shared: None,
            routed_out: None,
            out: &out,
            rows: t,
            hidden: H,
            top_k: TOPK,
            n_experts: E,
        }),
        "combine",
    );
    for tt in [rows, act, rot, dexp] {
        ctx.free(tt);
    }
    (gu, out)
}

#[derive(Clone, Copy, PartialEq)]
enum Grp {
    Wonly,
    A8,
    A4,
    Fp8,
}

impl Grp {
    fn name(self) -> &'static str {
        match self {
            Grp::Wonly => "grouped_wonly",
            Grp::A8 => "grouped_a8",
            Grp::A4 => "grouped_a4",
            Grp::Fp8 => "grouped_fp8",
        }
    }
}

/// One grouped expert GEMM `[bound, m]` over per-slot F32 rows `x_rows [slots, k]` (x_row_div = 1).
fn grouped_gemm(
    ctx: &mut Ctx,
    arm: Grp,
    ptrs: &GpuTensor,
    dev: &Dev,
    x_rows: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
) -> Value {
    let (nt, rf);
    match arm {
        Grp::Wonly => {
            ok(
                ctx.gpu.gemm_mq4g256v2_moe_grouped_wmma_k2(
                    ptrs,
                    &dev.tiles,
                    &dev.sorted,
                    x_rows,
                    y,
                    m,
                    k,
                    1,
                    dev.bound,
                    dev.slots,
                ),
                "grouped wmma_k2",
            );
            return json!({"launcher": "gemm_mq4g256v2_moe_grouped_wmma_k2", "x_row_div": 1});
        }
        Grp::A8 => {
            let xq = ok(
                ctx.gpu.ensure_int8_mmq_x(x_rows, dev.slots, k),
                "ensure_int8_mmq_x",
            );
            (nt, rf) = g::select_entry_a8(dev.bound, m);
            ok(
                ctx.gpu.gemm_mq4g256v2_moe_grouped_a8_sym(
                    ptrs,
                    &dev.tiles,
                    &dev.offsets,
                    &dev.sorted,
                    xq,
                    y,
                    m,
                    k,
                    1,
                    dev.bound,
                    dev.slots,
                    nt,
                    rf,
                ),
                "grouped a8",
            );
        }
        Grp::A4 => {
            let xq = ok(
                ctx.gpu.ensure_int4_mmq_x(x_rows, dev.slots, k),
                "ensure_int4_mmq_x",
            );
            (nt, rf) = g::select_entry(dev.bound, m);
            ok(
                ctx.gpu.gemm_mq4g256v2_moe_grouped_iu4_sym(
                    ptrs,
                    &dev.tiles,
                    &dev.offsets,
                    &dev.sorted,
                    xq,
                    y,
                    m,
                    k,
                    1,
                    dev.bound,
                    dev.slots,
                    nt,
                    rf,
                ),
                "grouped a4",
            );
        }
        Grp::Fp8 => {
            let xq = ok(
                ctx.gpu.ensure_fp8_mmq_x(x_rows, dev.slots, k),
                "ensure_fp8_mmq_x",
            );
            (nt, rf) = g::select_entry_fp8(dev.bound, m);
            ok(
                ctx.gpu.gemm_mq4g256v2_moe_grouped_fp8_sym(
                    ptrs,
                    &dev.tiles,
                    &dev.offsets,
                    &dev.sorted,
                    xq,
                    y,
                    m,
                    k,
                    1,
                    dev.bound,
                    dev.slots,
                    nt,
                    rf,
                ),
                "grouped fp8",
            );
        }
    }
    json!({"nt": nt, "rf": rf, "x_row_div": 1, "m_total": dev.bound})
}

/// Whole grouped chain; returns (gate_up per slot, block output).
fn chain_grouped(
    ctx: &mut Ctx,
    arm: Grp,
    ew: &ExpertW,
    dev: &Dev,
    rows_gu: &GpuTensor,
) -> (Vec<f32>, Vec<f32>, Value) {
    let y_gu = ctx.zeros(&[dev.bound, H]);
    let d_gu = grouped_gemm(ctx, arm, &ew.gu_ptrs, dev, rows_gu, &y_gu, H, H);
    let act = ctx.zeros(&[dev.slots, DK]);
    ok(
        ctx.gpu.ffn_activation_grouped_f32(&FfnActivationGrouped {
            activation: situ_cfg(&ctx.cfg),
            y_grouped: &y_gu,
            sorted_slot_index: &dev.sorted,
            expert_tile_ids: &dev.tiles,
            out: &act,
            m_total: dev.bound,
            block_m: 16,
            logical_n: INTER,
            phys_n: DK,
            round_inputs: false,
            round_output: false,
            awq_table: ew.dn_awq_ptrs.as_ref(),
            n_experts: E,
        }),
        "situ grouped",
    );
    let rot = ctx.zeros(&[dev.slots, DK]);
    ok(
        ctx.gpu.rotate_x_mq_batched(&act, &rot, DK, dev.slots),
        "rotate 1280",
    );
    let y_dn = ctx.zeros(&[dev.bound, H]);
    let d_dn = grouped_gemm(ctx, arm, &ew.dn_ptrs, dev, &rot, &y_dn, H, DK);
    let out = combine(ctx, &y_dn, Some(&dev.inv), dev);
    let gu_h = ctx.dl(&y_gu);
    let blk = ctx.dl(&out);
    for t in [y_gu, act, rot, y_dn, out] {
        ctx.free(t);
    }
    (
        unscatter(&gu_h, &dev.inv_h, H),
        blk,
        json!({"gate_up": d_gu, "down": d_dn}),
    )
}

/// W4A8 grouped chain whose down input comes from the fused gfx1201 producer
/// (`ffn_activation_grouped_rotate_i8_f32`: SiTU + per-expert AWQ + FWHT + `block_i8_128` in one launch).
/// Returns the block output; it must equal [`chain_grouped`]'s `Grp::A8` block bit for bit.
fn chain_grouped_a8_fused(ctx: &mut Ctx, ew: &ExpertW, dev: &Dev, rows_gu: &GpuTensor) -> Vec<f32> {
    let y_gu = ctx.zeros(&[dev.bound, H]);
    grouped_gemm(ctx, Grp::A8, &ew.gu_ptrs, dev, rows_gu, &y_gu, H, H);
    let res = ok(ctx.gpu.reserve_int8_mmq(DK, dev.slots), "reserve_int8_mmq");
    let act = situ_cfg(&ctx.cfg);
    let prep = ok(
        ctx.gpu.ffn_activation_grouped_rotate_i8_f32(
            &FfnActivationGroupedI8 {
                activation: act,
                y_grouped: &y_gu,
                sorted_slot_index: &dev.sorted,
                expert_tile_ids: &dev.tiles,
                x_out: None,
                m_total: dev.bound,
                block_m: 16,
                logical_n: INTER,
                phys_n: DK,
                total_slots: dev.slots,
                round_inputs: false,
                round_output: false,
                awq_table: ew.dn_awq_ptrs.as_ref(),
                n_experts: E,
            },
            res,
        ),
        "ffn_activation_grouped_rotate_i8_f32",
    );
    let xq = ok(
        ctx.gpu.int8_mmq_prepared_ptr(&prep, DK, dev.slots),
        "int8_mmq_prepared_ptr",
    );
    let (nt, rf) = g::select_entry_a8(dev.bound, H);
    let y_dn = ctx.zeros(&[dev.bound, H]);
    ok(
        ctx.gpu.gemm_mq4g256v2_moe_grouped_a8_sym(
            &ew.dn_ptrs,
            &dev.tiles,
            &dev.offsets,
            &dev.sorted,
            xq,
            &y_dn,
            H,
            DK,
            1,
            dev.bound,
            dev.slots,
            nt,
            rf,
        ),
        "fused a8 down",
    );
    let out = combine(ctx, &y_dn, Some(&dev.inv), dev);
    let blk = ctx.dl(&out);
    for t in [y_gu, y_dn, out] {
        ctx.free(t);
    }
    blk
}

/// Teacher-fed down: captured act rows (pad 1280) through the route's down consumer.
fn fed_down_grouped(
    ctx: &mut Ctx,
    arm: Grp,
    ew: &ExpertW,
    dev: &Dev,
    rot_fed: &GpuTensor,
) -> Vec<f32> {
    let y = ctx.zeros(&[dev.bound, H]);
    grouped_gemm(ctx, arm, &ew.dn_ptrs, dev, rot_fed, &y, H, DK);
    let h = ctx.dl(&y);
    ctx.free(y);
    unscatter(&h, &dev.inv_h, H)
}

fn pad_act(act_slot: &[f32]) -> Vec<f32> {
    let slots = act_slot.len() / INTER;
    let mut out = vec![0f32; slots * DK];
    for s in 0..slots {
        out[s * DK..s * DK + INTER].copy_from_slice(&act_slot[s * INTER..(s + 1) * INTER]);
    }
    out
}

fn expert_routes_xts(
    ctx: &mut Ctx,
    layer: usize,
    fmt: &str,
    ew: &ExpertW,
    dev: &Dev,
    td: &Path,
    act_slot: &[f32],
    ids_h: &[i32],
) {
    let t = dev.t;
    let slots = dev.slots;
    let mut y_gu_t = read_f32_file(&td.join("y_gu.f32"));
    let mut y_dn_t = read_f32_file(&td.join("y_dn.f32"));
    let mut y_blk_t = read_f32_file(&td.join("y_blk.f32"));
    y_gu_t.truncate(slots * 2 * INTER);
    y_dn_t.truncate(slots * H);
    y_blk_t.truncate(t * H);

    // (a) scalar decode chain over all tokens.
    let t0 = std::time::Instant::now();
    let (gu, blk) = chain_scalar_xts(ctx, ew, &dev.x, &dev.ids, &dev.w, t);
    ctx.sync();
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    let gu_h = ctx.dl(&gu);
    let blk_h = ctx.dl(&blk);
    // Bitwise batch-1 identity of the token-batched chain (decode arithmetic is per row).
    let mut ident = true;
    for &tok in &[0usize, 1, t / 2, t - 1] {
        let (g1, b1) = chain_scalar_xts(
            ctx,
            ew,
            &dev.x.sub_offset(tok * H, H),
            &dev.ids.sub_offset(tok * TOPK, TOPK),
            &dev.w.sub_offset(tok * TOPK, TOPK),
            1,
        );
        let bh = ctx.dl(&b1);
        let gh = ctx.dl(&g1);
        ident &= bh
            .iter()
            .zip(&blk_h[tok * H..(tok + 1) * H])
            .all(|(a, b)| a.to_bits() == b.to_bits());
        ident &= gh
            .iter()
            .zip(&gu_h[tok * TOPK * 2 * INTER..(tok + 1) * TOPK * 2 * INTER])
            .all(|(a, b)| a.to_bits() == b.to_bits());
        ctx.free(g1);
        ctx.free(b1);
    }
    let det = json!({"chain": "rotate_x_mq_awq_topk -> gemv_mq4g256v2_moe_down_k8_indexed_batched_expanded(gate_up) -> ffn_activation_rows_f32(FfnAwq::PerExpert) -> rotate_x_mq_batched(1280) -> expanded down -> ffn_moe_combine_f32(Rank)", "batch1_bitwise_identical_tokens_0_1_mid_last": ident});
    ctx.scored(
        layer,
        fmt,
        "expert.gate_up",
        "scalar_decode",
        &gu_h,
        &y_gu_t,
        slots,
        2 * INTER,
        det.clone(),
        Some(ids_h),
        ms,
    );
    ctx.scored(
        layer,
        fmt,
        "expert.block",
        "scalar_decode",
        &blk_h,
        &y_blk_t,
        t,
        H,
        det,
        None,
        ms,
    );
    ctx.free(gu);
    ctx.free(blk);

    // Teacher-fed down: pad 1152 -> 1280 (exact zeros), per-slot AWQ + FWHT, expanded down GEMV.
    let act_pad = ctx.up_f32(&pad_act(act_slot), &[slots, DK]);
    let rot_fed = dn_rows_xts(ctx, ew, &act_pad, &dev.ids, slots);
    {
        let dexp = ctx.zeros(&[slots, H]);
        let t0 = std::time::Instant::now();
        ok(
            ctx.gpu.gemv_mq4g256v2_moe_down_k8_indexed_batched_expanded(
                &ew.dn_ptrs,
                &dev.ids,
                &rot_fed,
                &dexp,
                H,
                DK,
                TOPK,
                t,
            ),
            "fed down gemv",
        );
        ctx.sync();
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        let h = ctx.dl(&dexp);
        ctx.scored(layer, fmt, "expert.down_fed", "scalar_decode", &h, &y_dn_t, slots, H, json!({"input": "captured teacher post-SiTU rows, padded, per-slot AWQ + FWHT (rotate_x_mq_awq_topk top_k=1)"}), Some(ids_h), ms);
        ctx.free(dexp);
    }

    // Grouped arms with the per-slot AWQ producer.
    let rows_gu = gu_rows_xts(ctx, ew, &dev.x, &dev.ids, t);
    for arm in [Grp::Wonly, Grp::A8, Grp::A4, Grp::Fp8] {
        let t0 = std::time::Instant::now();
        let (gu_s, blk_s, det) = chain_grouped(ctx, arm, ew, dev, &rows_gu);
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        ctx.scored(
            layer,
            fmt,
            "expert.gate_up",
            arm.name(),
            &gu_s,
            &y_gu_t,
            slots,
            2 * INTER,
            det.clone(),
            Some(ids_h),
            ms,
        );
        ctx.scored(
            layer,
            fmt,
            "expert.block",
            arm.name(),
            &blk_s,
            &y_blk_t,
            t,
            H,
            det,
            None,
            ms,
        );
        let t0 = std::time::Instant::now();
        let dn = fed_down_grouped(ctx, arm, ew, dev, &rot_fed);
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        ctx.scored(
            layer,
            fmt,
            "expert.down_fed",
            arm.name(),
            &dn,
            &y_dn_t,
            slots,
            H,
            json!({"input": "captured teacher post-SiTU rows, per-slot AWQ + FWHT, x_row_div=1"}),
            Some(ids_h),
            ms,
        );
        if arm == Grp::A8 {
            let t0 = std::time::Instant::now();
            let fused = chain_grouped_a8_fused(ctx, ew, dev, &rows_gu);
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            let same = fused
                .iter()
                .zip(&blk_s)
                .all(|(a, b)| a.to_bits() == b.to_bits());
            ctx.scored(layer, fmt, "expert.block", "grouped_a8_fused", &fused, &y_blk_t, t, H, json!({"producer": "ffn_activation_grouped_rotate_i8_f32 (SiTU + awq_table + FWHT + block_i8_128)", "bitwise_equal_to_grouped_a8_block": same}), None, ms);
        }
    }
    ctx.free(rows_gu);
    ctx.free(rot_fed);
    ctx.free(act_pad);
}

fn expert_routes_a(
    ctx: &mut Ctx,
    layer: usize,
    fmt: &str,
    ew: &ExpertW,
    dev: &Dev,
    td: &Path,
    act_slot: &[f32],
    ids_h: &[i32],
) {
    let t = dev.t;
    let slots = dev.slots;
    let mut y_gu_t = read_f32_file(&td.join("y_gu.f32"));
    let mut y_dn_t = read_f32_file(&td.join("y_dn.f32"));
    let mut y_blk_t = read_f32_file(&td.join("y_blk.f32"));
    y_gu_t.truncate(slots * 2 * INTER);
    y_dn_t.truncate(slots * H);
    y_blk_t.truncate(t * H);
    let xrot = ctx.zeros(&[t, H]);
    ok(ctx.gpu.rotate_x_mq_batched(&dev.x, &xrot, H, t), "rotate");
    for arm in ["inc_batch", "inc_scalar"] {
        let gate = ctx.zeros(&[slots, INTER]);
        let up = ctx.zeros(&[slots, INTER]);
        let act = ctx.zeros(&[slots, DK]);
        let rot = ctx.zeros(&[slots, DK]);
        let dexp = ctx.zeros(&[slots, H]);
        let out = ctx.zeros(&[t, H]);
        let t0 = std::time::Instant::now();
        let act_cfg = situ_cfg(&ctx.cfg);
        if arm == "inc_batch" {
            ok(
                ctx.gpu.gemv_hfq4g256_moe_gate_up_k8_indexed_batched(
                    &ew.gu_ptrs,
                    &dev.ids,
                    &xrot,
                    &gate,
                    &up,
                    2 * INTER,
                    H,
                    TOPK,
                    t,
                ),
                "inc gate_up",
            );
            ok(
                ctx.gpu.ffn_activation_rows_f32(&FfnActivationRows {
                    activation: act_cfg,
                    gate: &gate,
                    gate_offset: 0,
                    gate_ld: INTER,
                    up: &up,
                    up_offset: 0,
                    up_ld: INTER,
                    out: &act,
                    rows: slots,
                    logical_n: INTER,
                    phys_n: DK,
                    round_inputs: false,
                    round_output: false,
                    awq: FfnAwq::None,
                }),
                "inc situ",
            );
            ok(
                ctx.gpu.rotate_x_mq_batched(&act, &rot, DK, slots),
                "rotate 1280",
            );
            ok(
                ctx.gpu.gemv_mq4g256v2_moe_down_k8_indexed_batched_expanded(
                    &ew.dn_ptrs,
                    &dev.ids,
                    &rot,
                    &dexp,
                    H,
                    DK,
                    TOPK,
                    t,
                ),
                "inc down",
            );
            ok(
                ctx.gpu.ffn_moe_combine_f32(&MoeCombine {
                    order: CombineOrder::Rank,
                    round_expert_outputs: false,
                    round_routed_sum: false,
                    round_final: false,
                    down_rows: &dexp,
                    row_of_slot: None,
                    topk_indices: &dev.ids,
                    topk_weights: &dev.w,
                    shared: None,
                    routed_out: None,
                    out: &out,
                    rows: t,
                    hidden: H,
                    top_k: TOPK,
                    n_experts: E,
                }),
                "combine",
            );
        } else {
            for i in 0..t {
                let xr = xrot.sub_offset(i * H, H);
                let iv = dev.ids.sub_offset(i * TOPK, TOPK);
                let gv = gate.sub_offset(i * TOPK * INTER, TOPK * INTER);
                let uv = up.sub_offset(i * TOPK * INTER, TOPK * INTER);
                let av = act.sub_offset(i * TOPK * DK, TOPK * DK);
                let rv = rot.sub_offset(i * TOPK * DK, TOPK * DK);
                let dv = dexp.sub_offset(i * TOPK * H, TOPK * H);
                ok(
                    ctx.gpu.gemv_hfq4g256_moe_gate_up_k8_indexed_batched(
                        &ew.gu_ptrs,
                        &iv,
                        &xr,
                        &gv,
                        &uv,
                        2 * INTER,
                        H,
                        TOPK,
                        1,
                    ),
                    "inc gate_up",
                );
                ok(
                    ctx.gpu.ffn_activation_rows_f32(&FfnActivationRows {
                        activation: act_cfg,
                        gate: &gv,
                        gate_offset: 0,
                        gate_ld: INTER,
                        up: &uv,
                        up_offset: 0,
                        up_ld: INTER,
                        out: &av,
                        rows: TOPK,
                        logical_n: INTER,
                        phys_n: DK,
                        round_inputs: false,
                        round_output: false,
                        awq: FfnAwq::None,
                    }),
                    "inc situ",
                );
                ok(
                    ctx.gpu.rotate_x_mq_batched(&av, &rv, DK, TOPK),
                    "rotate 1280",
                );
                ok(
                    ctx.gpu.gemv_mq4g256v2_moe_down_k8_indexed_batched_expanded(
                        &ew.dn_ptrs,
                        &iv,
                        &rv,
                        &dv,
                        H,
                        DK,
                        TOPK,
                        1,
                    ),
                    "inc down",
                );
                let wv = dev.w.sub_offset(i * TOPK, TOPK);
                let ov = out.sub_offset(i * H, H);
                ok(
                    ctx.gpu.ffn_moe_combine_f32(&MoeCombine {
                        order: CombineOrder::Rank,
                        round_expert_outputs: false,
                        round_routed_sum: false,
                        round_final: false,
                        down_rows: &dv,
                        row_of_slot: None,
                        topk_indices: &iv,
                        topk_weights: &wv,
                        shared: None,
                        routed_out: None,
                        out: &ov,
                        rows: 1,
                        hidden: H,
                        top_k: TOPK,
                        n_experts: E,
                    }),
                    "combine",
                );
            }
        }
        ctx.sync();
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        let gh = ctx.dl(&gate);
        let uh = ctx.dl(&up);
        let mut gu = vec![0f32; slots * 2 * INTER];
        for s in 0..slots {
            gu[s * 2 * INTER..s * 2 * INTER + INTER]
                .copy_from_slice(&gh[s * INTER..(s + 1) * INTER]);
            gu[s * 2 * INTER + INTER..(s + 1) * 2 * INTER]
                .copy_from_slice(&uh[s * INTER..(s + 1) * INTER]);
        }
        let blk = ctx.dl(&out);
        let det = json!({"chain": "incumbent: rotate_x_mq_batched -> gemv_hfq4g256_moe_gate_up_k8_indexed_batched (qt13) -> ffn_activation_rows_f32 -> rotate_x_mq_batched(1280) -> gemv_mq4g256v2_moe_down_k8_indexed_batched_expanded (runtime-repacked qt44 down, pilot a-down-rt) -> Rank combine", "batch": if arm == "inc_batch" { "token-batched" } else { "one token per launch (scalar decode)" }});
        ctx.scored(
            layer,
            fmt,
            "expert.gate_up",
            arm,
            &gu,
            &y_gu_t,
            slots,
            2 * INTER,
            det.clone(),
            Some(ids_h),
            ms,
        );
        ctx.scored(
            layer,
            fmt,
            "expert.block",
            arm,
            &blk,
            &y_blk_t,
            t,
            H,
            det,
            None,
            ms,
        );
        for tt in [gate, up, act, rot, dexp, out] {
            ctx.free(tt);
        }
    }
    // Teacher-fed down through the incumbent repacked down (no AWQ).
    let act_pad = ctx.up_f32(&pad_act(act_slot), &[slots, DK]);
    let rot_fed = ctx.zeros(&[slots, DK]);
    ok(
        ctx.gpu.rotate_x_mq_batched(&act_pad, &rot_fed, DK, slots),
        "rotate fed",
    );
    let dexp = ctx.zeros(&[slots, H]);
    ok(
        ctx.gpu.gemv_mq4g256v2_moe_down_k8_indexed_batched_expanded(
            &ew.dn_ptrs,
            &dev.ids,
            &rot_fed,
            &dexp,
            H,
            DK,
            TOPK,
            t,
        ),
        "fed down",
    );
    let h = ctx.dl(&dexp);
    ctx.scored(
        layer,
        fmt,
        "expert.down_fed",
        "inc_batch",
        &h,
        &y_dn_t,
        slots,
        H,
        json!({"input": "captured teacher post-SiTU rows, padded, plain FWHT"}),
        Some(ids_h),
        0.0,
    );
    for tt in [act_pad, rot_fed, dexp, xrot] {
        ctx.free(tt);
    }
    for arm in ["grouped_wonly", "grouped_a8", "grouped_a4", "grouped_fp8"] {
        ctx.unscored(
            Some(layer), fmt, "expert.*", arm, "not_applicable",
            "incumbent A experts are asymmetric qt13 gate_up (no AWQ sidecars, mq4v2.symmetric absent): grouped sym A8/A4/FP8 launchers require symmetric qt44 and wmma_k2 needs the qt44 v2 layout",
        );
    }
}

fn layer_run(
    ctx: &mut Ctx,
    hfq: &HfqFile,
    layer: usize,
    teacher_dir: &Path,
    check_dir: &Path,
    a_down_rt: &Path,
    only: &[String],
) {
    let td = layer_dir(teacher_dir, layer);
    let tmeta = read_json(&td.join("meta.json"));
    let fmt = tmeta["formats"]["w13"].as_str().unwrap_or("?").to_string();
    eprintln!("== layer {layer} source {fmt} ==");
    let want = |s: &str| only.is_empty() || only.iter().any(|o| o == s);
    let idx = read_json(&check_dir.join(format!("L{layer:02}.index.json")));
    let p = format!("model.llm.layers.{layer}.");
    let mut sym_all = true;

    // Dense.
    if want("dense") {
        let lim = ctx.tok_limit;
        for (tag, wname, m, k, site) in [
            ("q", "attn.wq_du.weight", EATTN, H, "q"),
            ("k", "attn.wk_dv.weight", 1024, H, "k"),
            ("v", "attn.wv_dv.weight", 1024, H, "v"),
            ("wr", "attn.wr_du.weight", 384, H, "q"),
            ("gate", "attn.gate_attn.weight", EATTN, H, "q"),
            ("o", "attn.wo_ud.weight", H, EATTN, "attn_out"),
        ] {
            let d = upload_dense(ctx, hfq, &format!("{p}{wname}"), m, k, &mut sym_all);
            let s = &idx["sites"][site];
            let n = (s["rows"].as_u64().unwrap() as usize).min(lim);
            let mut x = read_f32_file(&check_dir.join(s["file"].as_str().unwrap()));
            x.truncate(n * k);
            let mut teacher = read_f32_file(&td.join(format!("y_{tag}.f32")));
            teacher.truncate(n * m);
            let name = match tag {
                "q" => "dense.wq",
                "k" => "dense.wk",
                "v" => "dense.wv",
                "wr" => "dense.wr(rel)",
                "gate" => "dense.gate_attn",
                _ => "dense.wo",
            };
            dense_routes(ctx, layer, &fmt, name, &d, &x, n, &teacher);
            free_dense(ctx, d);
        }
        // Router: A is qt44, XTS keeps F16 from source.
        let rn = format!("{p}mlp.gate.weight");
        let (ri, _) = tensor_bytes(hfq, &rn);
        let d = upload_dense(ctx, hfq, &rn, E, H, &mut sym_all);
        let mut xt = read_f32_file(&td.join("x_tok.f32"));
        xt.truncate(ctx.tok_limit.min(xt.len() / H) * H);
        let n = xt.len() / H;
        let mut teacher = read_f32_file(&td.join("y_router.f32"));
        teacher.truncate(n * E);
        let site = if ri.quant_type == 1 {
            "router.f16"
        } else {
            "router.qt44"
        };
        dense_routes(ctx, layer, &fmt, site, &d, &xt, n, &teacher);
        free_dense(ctx, d);
    }

    // Experts.
    if want("experts") {
        let mut x_tok = read_f32_file(&td.join("x_tok.f32"));
        let mut ids = read_i32_file(&td.join("ids.i32"));
        let mut w = read_f32_file(&td.join("w.f32"));
        let mut act_slot = read_f32_file(&td.join("act_slot.f32"));
        let nt = ctx.tok_limit.min(x_tok.len() / H);
        x_tok.truncate(nt * H);
        ids.truncate(nt * TOPK);
        w.truncate(nt * TOPK);
        act_slot.truncate(nt * TOPK * INTER);
        let ew = upload_experts(ctx, hfq, layer, a_down_rt);
        if ctx.xts {
            ctx.sym_verdict = Some(ctx.sym_verdict.unwrap_or(true) && ew.sym);
            assert!(ew.sym, "L{layer}: XTS expert headers not all symmetric");
        }
        let dev = make_dev(ctx, &x_tok, &ids, &w);
        eprintln!(
            "  experts uploaded: tokens {} slots {} bound {}",
            dev.t, dev.slots, dev.bound
        );
        if ctx.xts {
            expert_routes_xts(ctx, layer, &fmt, &ew, &dev, &td, &act_slot, &ids);
        } else {
            expert_routes_a(ctx, layer, &fmt, &ew, &dev, &td, &act_slot, &ids);
        }
        free_dev(ctx, dev);
        free_experts(ctx, ew);
    }
}

fn unembed_run(ctx: &mut Ctx, hfq: &HfqFile, ue_dir: &Path, check_dir: &Path) {
    let meta = read_json(&ue_dir.join("meta.json"));
    let nrows = meta["rows"].as_u64().unwrap() as usize;
    let nin_all = meta["inputs"].as_u64().unwrap() as usize;
    let nin = nin_all.min(ctx.tok_limit);
    let rows = unembed_rows(201024, nrows);
    let (info, data) = tensor_bytes(hfq, "model.llm.unembed.weight");
    assert_eq!(info.quant_type, 44);
    let stride = H / 256 * 136;
    let mut sub = Vec::with_capacity(nrows * stride);
    for &r in &rows {
        sub.extend_from_slice(&data[r * stride..(r + 1) * stride]);
    }
    let sym = qt44_symmetric(&sub);
    let buf = ok(ctx.gpu.upload_raw(&sub, &[sub.len()]), "unembed subset");
    let d = DenseW {
        w: WeightTensor {
            buf,
            gpu_dtype: DType::MQ4G256V2,
            m: nrows,
            k: H,
            row_stride: stride,
            awq_scale: None,
            lloyd_lut_e4m3: None,
            lloyd_lut_f16: None,
            lloyd_lut_c16: None,
        },
        sym,
        qt: 44,
        awq: false,
    };
    let idx = read_json(&check_dir.join("L20.index.json"));
    let q = &idx["sites"]["q"];
    let x_all = read_f32_file(&check_dir.join(q["file"].as_str().unwrap()));
    let mut teacher = read_f32_file(&ue_dir.join("y.f32"));
    teacher.truncate(nin * nrows);
    eprintln!("== unembed subset ({nrows} rows, symmetric headers: {sym}) ==");
    dense_routes(
        ctx,
        20,
        "bf16",
        "dense.unembed_subset",
        &d,
        &x_all[..nin * H],
        nin,
        &teacher,
    );
    free_dense(ctx, d);
}

fn run_main(a: &[String]) {
    let label = arg(a, "--label", "");
    let path = arg(a, "--artifact", "");
    let teacher_dir = PathBuf::from(arg(a, "--teacher-dir", ""));
    let check_dir = PathBuf::from(arg(a, "--check-dir", ""));
    let out_path = arg(a, "--out", "");
    let a_down_rt = PathBuf::from(arg(
        a,
        "--a-down-rt",
        "/home/kaden/qcal/release-0.5.0/inkling/speed/sym-expert/weights",
    ));
    let layers = parse_layers(&arg(a, "--layers", "5,6,20"));
    let only: Vec<String> = arg(a, "--only", "")
        .split(',')
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect();
    let hfq = ok(HfqFile::open(Path::new(&path)), "open artifact");
    let cfg = ok(
        Inkling2Config::from_metadata_json(&hfq.metadata_json),
        "config",
    );
    let xts = cfg.mq4_layout.is_xts();
    assert_eq!(
        (cfg.n_embd, cfg.inter, cfg.n_experts, cfg.n_expert_used),
        (H, INTER, E, TOPK)
    );
    let gpu = ok(Gpu::init_with_device(0), "Gpu::init_with_device");
    if gpu.arch != "gfx1201" {
        die::<()>(format!("needs gfx1201, got {}", gpu.arch));
    }
    let sha_file = format!("{path}.sha256");
    let artifact_sha256 = std::fs::read_to_string(&sha_file)
        .ok()
        .and_then(|s| s.split_whitespace().next().map(String::from))
        .unwrap_or_else(|| "unrecorded".into());
    let default_iu4 = gpu.flags.iu4_prefill;
    let default_a8 = gpu.flags.a8_prefill;
    let default_fp8 = gpu.flags.gfx12_mq4v2_fp8_resid;
    // The XTS loader sets the device flag for a certified typed layout; mirror it. Legacy A leaves it as loaded.
    let sym_flag_default = gpu.mq4v2_symmetric;
    let out = ok(
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&out_path),
        "open out",
    );
    let mut ctx = Ctx {
        gpu,
        label,
        artifact_path: path.clone(),
        artifact_sha256,
        binary_md5: exe_md5(),
        out,
        xts,
        sym_verdict: None,
        cfg,
        default_iu4,
        default_a8,
        default_fp8_resid: default_fp8,
        job: std::env::var("FABRIC_JOB_ID").unwrap_or_else(|_| arg(a, "--job", "unknown")),
        ablate_awq: has(a, "--ablate-awq"),
        tok_limit: arg(a, "--tokens", "0")
            .parse::<usize>()
            .ok()
            .filter(|&n| n > 0)
            .unwrap_or(usize::MAX),
    };
    ctx.gpu.mq4v2_symmetric = xts;
    eprintln!(
        "artifact {} ({}): xts={xts}, device flag mq4v2_symmetric {} -> {}; defaults iu4_prefill={:?} a8_prefill={} fp8_resid={}",
        ctx.label, path, sym_flag_default, ctx.gpu.mq4v2_symmetric, default_iu4, default_a8, default_fp8
    );
    ctx.rec(None, "-", "run", "header", json!({"status": "info", "xts": xts, "hfq_mq4v2_symmetric": hfq.mq4v2_symmetric(),
        "gpu_mq4v2_symmetric_set_to": xts, "default_flags": {"iu4_prefill": default_iu4, "a8_prefill": default_a8, "gfx12_mq4v2_fp8_resid": default_fp8},
        "situ": {"gate_beta": ctx.cfg.situ_beta, "value_beta": ctx.cfg.situ_linear_beta}, "layers": layers, "capability_eval": {
            "iu4_prefill_enabled": ctx.gpu.flags.iu4_prefill_enabled(),
            "a8_prefill_enabled_default": default_a8,
            "moe_sym_iu4_requested": ctx.gpu.moe_sym_iu4_requested(),
            "moe_sym_iu4_disabled": ctx.gpu.moe_sym_iu4_disabled(),
            "qwen4_moe_sym_iu4_requested": ctx.gpu.qwen4_moe_sym_iu4_requested(),
            "qwen4_moe_down_a8_requested": ctx.gpu.qwen4_moe_down_a8_requested(),
            "hipcc_extra_flags": ctx.gpu.flags.hipcc_extra_flags.clone(),
        },
        "ablate_awq": ctx.ablate_awq, "tokens_limit": if ctx.tok_limit == usize::MAX { Value::Null } else { json!(ctx.tok_limit) }, "argv": a}));
    for &layer in &layers {
        layer_run(
            &mut ctx,
            &hfq,
            layer,
            &teacher_dir,
            &check_dir,
            &a_down_rt,
            &only,
        );
    }
    let ue = arg(a, "--unembed-dir", "");
    if !ue.is_empty() && (only.is_empty() || only.iter().any(|o| o == "unembed")) {
        unembed_run(&mut ctx, &hfq, Path::new(&ue), &check_dir);
    }
    eprintln!("done");
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    match a.get(1).map(String::as_str) {
        Some("teacher") => teacher_main(&a),
        Some("unembed") => teacher_unembed(&a),
        Some("run") => run_main(&a),
        _ => die::<()>("usage: inkling_xts_routes teacher|unembed|run ... (see the file header)"),
    }
}
