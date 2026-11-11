// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Test-local launch path for the committed `lane_leviathan_verify` objects
//! (`kernels/lane_leviathan_{hipcc,pm}_gfx120{0,1}.{co,hxaco}`).
//!
//! The product launchers (`Gpu::lane_leviathan_*_enqueue`) JIT the HIP source and the
//! staged arena keeps its device region private, so a committed object cannot be run
//! through them. This module packs the same `[out | lanes | rows | inputs]` region the
//! arena packs (`hipfire_runtime::lane_draw`, the layout constants of
//! `rdna_compute::lane_leviathan`), loads an object with `module_load_data` and launches
//! `lane_leviathan_verify` with the product's seven kernargs, grid and block. No hipcc,
//! no JIT, no dispatch: the bytes under test are the committed `include_bytes!` objects.

use hip_bridge::{Function, Module};
use hipfire_runtime::lane_draw::{LaneDrawJob, LANE_DRAW_REPEAT_CAP};
use hipfire_runtime::sampler::SamplerConfig;
use rdna_compute::lane_leviathan::{
    LaneLeviathanLayout, LLV_LANE_WORDS, LLV_NO_EOS, LLV_OUT_WORDS, LLV_Q_MAX, LLV_Q_WORDS,
    LLV_ROW_WORDS, LL_BASE, LL_BLOCKED, LL_DRAFTS, LL_EOS, LL_FREQ, LL_MIN_P, LL_NBLOCKED,
    LL_NROWS, LL_OUT, LL_PRES, LL_Q, LL_REP, LL_RNG, LL_ROW0, LL_TEMP, LL_TOP_K, LL_TOP_P,
    LL_VOCAB, LO_ACCEPTED, LO_DONE, LO_NTOK, LO_RNG, LO_TICKET, LO_TOKENS, LR_J, LR_LANE,
    LR_PEN_LEN, LR_PEN_OFF, LR_PTR_HI, LR_PTR_LO,
};
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::ffi::c_void;
use std::path::PathBuf;

pub const SYMBOL: &str = "lane_leviathan_verify";
pub const BLOCK: u32 = 512;

pub const MODE_FULL: i32 = 0;
pub const MODE_SELECT: i32 = 1;
pub const MODE_MERGED: i32 = 2;

/// The committed objects of one exact architecture.
pub struct Objects {
    pub arch: &'static str,
    pub hipcc: &'static [u8],
    pub pm: &'static [u8],
}

static GFX1200: Objects = Objects {
    arch: "gfx1200",
    hipcc: include_bytes!("../../../../kernels/lane_leviathan_hipcc_gfx1200.co"),
    pm: include_bytes!("../../../../kernels/lane_leviathan_pm_gfx1200.hxaco"),
};
static GFX1201: Objects = Objects {
    arch: "gfx1201",
    hipcc: include_bytes!("../../../../kernels/lane_leviathan_hipcc_gfx1201.co"),
    pm: include_bytes!("../../../../kernels/lane_leviathan_pm_gfx1201.hxaco"),
};

/// The committed objects for the actual device. Any other architecture fails the test:
/// a twin test that ran nothing must not pass.
pub fn objects_for(gpu: &Gpu) -> &'static Objects {
    match gpu.arch.as_str() {
        "gfx1200" => &GFX1200,
        "gfx1201" => &GFX1201,
        other => panic!(
            "lane_leviathan twin: no committed object for arch {other:?} (gfx1200 and gfx1201 only)"
        ),
    }
}

/// One loaded object's `lane_leviathan_verify`. The module owns the function.
pub struct Impl {
    pub name: &'static str,
    _module: Module,
    func: Function,
}

impl Impl {
    pub fn load(gpu: &mut Gpu, name: &'static str, image: &[u8]) -> Impl {
        assert!(!image.is_empty(), "{name}: committed object is empty");
        // A raw committed-module launch needs only the thread bound to the device:
        // route preloading could JIT unrelated product kernels.
        gpu.bind_thread().expect("bind thread");
        let module = gpu
            .hip
            .module_load_data(image)
            .unwrap_or_else(|e| panic!("{name}: module_load_data on {}: {e}", gpu.arch));
        let func = gpu
            .hip
            .module_get_function(&module, SYMBOL)
            .unwrap_or_else(|e| panic!("{name}: symbol {SYMBOL}: {e}"));
        Impl {
            name,
            _module: module,
            func,
        }
    }

    /// Launch with the product's seven kernargs: region, lanes_off, rows_off, ptab, mode,
    /// cand_local, cand_peer. One workgroup of 512 per verify row, no dynamic LDS.
    #[allow(clippy::too_many_arguments)]
    pub fn launch(
        &self,
        gpu: &mut Gpu,
        region: &GpuTensor,
        layout: LaneLeviathanLayout,
        ptab: Option<&GpuTensor>,
        mode: i32,
        cand_local: u64,
        cand_peer: u64,
    ) {
        gpu.bind_thread().expect("bind thread");
        let mut region_ptr = region.buf.as_ptr();
        let mut lanes_off = layout.lanes_off() as i32;
        let mut rows_off = layout.rows_off() as i32;
        let mut ptab_ptr = ptab.map_or(std::ptr::null_mut(), |p| p.buf.as_ptr());
        let mut mode_arg = mode;
        let mut cand_local_arg = cand_local;
        let mut cand_peer_arg = cand_peer;
        let mut params: Vec<*mut c_void> = vec![
            &mut region_ptr as *mut _ as *mut c_void,
            &mut lanes_off as *mut _ as *mut c_void,
            &mut rows_off as *mut _ as *mut c_void,
            &mut ptab_ptr as *mut _ as *mut c_void,
            &mut mode_arg as *mut _ as *mut c_void,
            &mut cand_local_arg as *mut _ as *mut c_void,
            &mut cand_peer_arg as *mut _ as *mut c_void,
        ];
        // SAFETY: `params` holds the seven kernarg values of the frozen ABI, live for the
        // call (the launch copies them); the region, ptab and candidate tensors are live
        // allocations of at least the sizes the kernel indexes, held by the caller until
        // the caller's device sync.
        unsafe {
            gpu.hip.launch_kernel(
                &self.func,
                [layout.rows as u32, 1, 1],
                [BLOCK, 1, 1],
                0,
                gpu.active_stream.as_ref(),
                &mut params,
            )
        }
        .unwrap_or_else(|e| panic!("{}: launch mode {mode}: {e}", self.name));
    }
}

fn penalty_active(cfg: &SamplerConfig) -> bool {
    (cfg.repeat_penalty > 1.0 || cfg.presence_penalty > 0.0 || cfg.frequency_penalty > 0.0)
        && cfg.repeat_window > 0
}

/// The packed `[out | lanes | rows | inputs]` words of one launch: the arena's packing
/// (out records zero).
pub struct Packed {
    pub layout: LaneLeviathanLayout,
    pub words: Vec<u32>,
}

pub fn pack(jobs: &[LaneDrawJob<'_>]) -> Packed {
    let rows: usize = jobs.iter().map(|j| j.rows).sum();
    let layout = LaneLeviathanLayout {
        lanes: jobs.len(),
        rows,
    };
    let mut w: Vec<u32> = vec![0; layout.inputs_off()];
    let mut grow = 0usize;
    let mut tail: Vec<u32> = Vec::new();
    for (li, job) in jobs.iter().enumerate() {
        let cfg = job.cfg;
        let drafts_off = w.len();
        w.extend_from_slice(job.drafts);
        let q_off = w.len();
        for (j, &d) in job.drafts.iter().enumerate() {
            let mut rec = [0u32; LLV_Q_WORDS];
            match job.q.get(j).filter(|q| !q.is_empty()) {
                Some(q) => {
                    rec[0] = q.len() as u32;
                    for (k, &(t, p)) in q.iter().enumerate() {
                        rec[1 + k] = t;
                        rec[1 + LLV_Q_MAX + k] = p.to_bits();
                    }
                }
                None => {
                    rec[0] = 1;
                    rec[1] = d;
                    rec[1 + LLV_Q_MAX] = 1.0f32.to_bits();
                }
            }
            w.extend_from_slice(&rec);
        }
        let blocked_off = w.len();
        w.extend_from_slice(&cfg.blocked_tokens);
        let window = if penalty_active(cfg) {
            cfg.repeat_window.min(LANE_DRAW_REPEAT_CAP)
        } else {
            0
        };
        let tail_start = job.history.len().saturating_sub(window);
        tail.clear();
        if window > 0 {
            tail.extend_from_slice(&job.history[tail_start..]);
            tail.extend_from_slice(job.drafts);
        }
        let tail_off = w.len();
        w.extend_from_slice(&tail);
        let lr = layout.lanes_off() + li * LLV_LANE_WORDS;
        w[lr + LL_ROW0] = grow as u32;
        w[lr + LL_NROWS] = job.rows as u32;
        w[lr + LL_RNG] = job.rng_state;
        w[lr + LL_EOS] = job.eos.unwrap_or(LLV_NO_EOS);
        w[lr + LL_TEMP] = cfg.temperature.to_bits();
        w[lr + LL_TOP_P] = cfg.top_p.to_bits();
        w[lr + LL_TOP_K] = cfg.top_k.map_or(20, |k| k.min(i32::MAX as u32) as i32) as u32;
        w[lr + LL_MIN_P] = cfg.min_p.unwrap_or(0.0).to_bits();
        w[lr + LL_REP] = cfg.repeat_penalty.to_bits();
        w[lr + LL_PRES] = cfg.presence_penalty.to_bits();
        w[lr + LL_FREQ] = cfg.frequency_penalty.to_bits();
        w[lr + LL_VOCAB] = job.vocab as u32;
        w[lr + LL_BASE] = job.base as u32;
        w[lr + LL_DRAFTS] = drafts_off as u32;
        w[lr + LL_Q] = q_off as u32;
        w[lr + LL_BLOCKED] = blocked_off as u32;
        w[lr + LL_NBLOCKED] = cfg.blocked_tokens.len() as u32;
        w[lr + LL_OUT] = layout.out_off(li) as u32;
        let base = job.logits.buf.as_ptr() as u64;
        for j in 0..job.rows {
            let (pen_off, pen_len) = if window > 0 {
                let end = job.history.len() + j;
                let start = end.saturating_sub(window);
                (tail_off + (start - tail_start), end - start)
            } else {
                (tail_off, 0)
            };
            let rr = layout.rows_off() + grow * LLV_ROW_WORDS;
            let ptr = base + (j * job.vocab * 4) as u64;
            w[rr + LR_PTR_LO] = ptr as u32;
            w[rr + LR_PTR_HI] = (ptr >> 32) as u32;
            w[rr + LR_LANE] = li as u32;
            w[rr + LR_J] = j as u32;
            w[rr + LR_PEN_OFF] = pen_off as u32;
            w[rr + LR_PEN_LEN] = pen_len as u32;
            grow += 1;
        }
    }
    Packed { layout, words: w }
}

/// A zeroed device buffer of `words` 32-bit words.
pub fn zeros_words(gpu: &mut Gpu, words: usize) -> GpuTensor {
    gpu.zeros(&[words.max(1)], DType::F32).expect("zeros")
}

/// A fresh device region holding `packed` (out records zero), synchronised before return.
pub fn upload_region(gpu: &mut Gpu, packed: &Packed) -> GpuTensor {
    let t = zeros_words(gpu, packed.words.len());
    let bytes: Vec<u8> = packed.words.iter().flat_map(|w| w.to_le_bytes()).collect();
    gpu.hip.memcpy_htod(&t.buf, &bytes).expect("region upload");
    gpu.hip.device_synchronize().expect("sync after upload");
    t
}

/// The first `words` 32-bit words of `t`, after a device sync (raw bits, no float
/// conversion).
pub fn download_words(gpu: &mut Gpu, t: &GpuTensor, words: usize) -> Vec<u32> {
    gpu.hip.device_synchronize().expect("sync before download");
    gpu.take_device_wait_error().expect("device wait");
    let mut bytes = vec![0u8; words * 4];
    gpu.hip
        .memcpy_dtoh(&mut bytes, &t.buf)
        .expect("download words");
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// One lane's out record decoded (`tokens`, accepted draft count, advanced RNG).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Draw {
    pub tokens: Vec<u32>,
    pub accepted: u32,
    pub rng: u32,
}

/// Validate every lane's out record the way the arena's readback does (done, ticket equals
/// the lane's rows, token count in range, nonzero RNG) and decode the draws. A region that
/// was packed wrong or a kernel that did not finish fails here instead of comparing equal
/// garbage.
pub fn decode_out(out: &[u32], rows_per_lane: &[usize], ctx: &str) -> Vec<Draw> {
    assert_eq!(
        out.len(),
        rows_per_lane.len() * LLV_OUT_WORDS,
        "{ctx}: out length"
    );
    rows_per_lane
        .iter()
        .enumerate()
        .map(|(i, &rows)| {
            let o = &out[i * LLV_OUT_WORDS..(i + 1) * LLV_OUT_WORDS];
            let ntok = o[LO_NTOK] as usize;
            let accepted = o[LO_ACCEPTED];
            assert!(
                o[LO_DONE] == 1
                    && o[LO_TICKET] as usize == rows
                    && (1..=rows).contains(&ntok)
                    && accepted as usize <= ntok
                    && o[LO_RNG] != 0,
                "{ctx}: lane {i}: corrupt out record {o:?}"
            );
            Draw {
                tokens: o[LO_TOKENS..LO_TOKENS + ntok].to_vec(),
                accepted,
                rng: o[LO_RNG],
            }
        })
        .collect()
}

/// Byte-exact comparison of two word buffers, panicking at the first differing word.
pub fn assert_words_eq(a: &[u32], b: &[u32], row_words: usize, what: &str, ctx: &str) {
    assert_eq!(a.len(), b.len(), "{ctx}: {what}: length");
    if let Some(at) = a.iter().zip(b).position(|(x, y)| x != y) {
        panic!(
            "{ctx}: {what} differ at row {} word {} (flat {at}): {:#010x} vs {:#010x}",
            at / row_words,
            at % row_words,
            a[at],
            b[at]
        );
    }
}

// ───────────────────────── GPU capture for the emulator replay ─────────────────────────
//
// The capture format is documented in the doc comment of `x1_lane_draw_shard_gpu.rs`
// (`leviathan_emu_capture`); `crates/peacemaker-isa/tests/leviathan_emu_replay.rs` reads it.

/// Environment variable naming the directory `leviathan_emu_capture` writes.
pub const CAPTURE_ENV: &str = "HIPFIRE_LEV_EMU_CAPTURE";
/// The `format` field of `index.json` and of every launch header.
pub const CAPTURE_FORMAT: &str = "leviathan_emu_capture/1";

/// One kernel launch as the capture needs it: the seven kernarg values (derived from the
/// tensors exactly as [`Impl::launch`] derives them) and every device buffer the kernel can
/// reach: the region, the optional p table and candidate tables, and each lane's logits
/// (the region's row records point into them).
pub struct Call<'a> {
    /// Which of the oracle's launches this is: `mode0_full`, `mode1_shard_a`,
    /// `mode1_shard_b` or `mode2_merged`.
    pub label: &'static str,
    pub mode: i32,
    pub region: &'a GpuTensor,
    pub layout: LaneLeviathanLayout,
    pub ptab: Option<&'a GpuTensor>,
    pub cand_local: Option<&'a GpuTensor>,
    pub cand_peer: Option<&'a GpuTensor>,
    /// Which logits tensors `logits` are: `full` (rows of V), `shard_a` or `shard_b`
    /// (packed halves). Names the shared on-disk blob of a launch set's logits.
    pub logits_tag: &'static str,
    /// Every lane's logits tensor, in lane order.
    pub logits: &'a [GpuTensor],
}

/// Device pointer of an optional tensor, 0 when absent: the kernarg value the launch passes.
fn device_ptr(t: Option<&GpuTensor>) -> u64 {
    t.map_or(0, |t| t.buf.as_ptr() as u64)
}

/// One device buffer of a launch. A `logits` buffer is an input the kernel never writes.
struct Buf<'a> {
    role: &'static str,
    lane: Option<usize>,
    tensor: &'a GpuTensor,
    read_only: bool,
}

/// Every buffer the launch can read or write, in capture order.
fn buffers<'a>(call: &Call<'a>) -> Vec<Buf<'a>> {
    let mut v = vec![Buf {
        role: "region",
        lane: None,
        tensor: call.region,
        read_only: false,
    }];
    for (role, tensor) in [
        ("ptab", call.ptab),
        ("cand_local", call.cand_local),
        ("cand_peer", call.cand_peer),
    ] {
        if let Some(tensor) = tensor {
            v.push(Buf {
                role,
                lane: None,
                tensor,
                read_only: false,
            });
        }
    }
    for (lane, tensor) in call.logits.iter().enumerate() {
        v.push(Buf {
            role: "logits",
            lane: Some(lane),
            tensor,
            read_only: true,
        });
    }
    v
}

/// Every whole tensor of the launch as raw words, after a device sync.
fn snapshot(gpu: &mut Gpu, call: &Call<'_>) -> Vec<Vec<u32>> {
    buffers(call)
        .iter()
        .map(|b| download_words(gpu, b.tensor, b.tensor.numel()))
        .collect()
}

/// A blob already on disk.
struct Blob {
    bytes: usize,
    sha256: String,
}

/// The launch set the next launches belong to.
struct SetCtx {
    vocab: usize,
    launch: usize,
    ctx: String,
}

/// Writes a capture directory: per launch a JSON header plus one raw `.bin` blob per
/// buffer and phase, and `index.json`. It only observes: the launches it records are the
/// oracle's own ([`Impl::launch_call`]).
pub struct Capture {
    dir: PathBuf,
    arch: String,
    object: &'static str,
    object_sha256: String,
    blobs: HashMap<String, Blob>,
    /// One `index.json` entry per launch, in launch order.
    launches: Vec<Value>,
    blob_bytes: u64,
    total_bytes: u64,
    set: Option<SetCtx>,
}

impl Capture {
    /// The capture directory from [`CAPTURE_ENV`]; unset or empty is an error, not a skip.
    pub fn dir_from_env() -> PathBuf {
        match std::env::var(CAPTURE_ENV) {
            Ok(dir) if !dir.is_empty() => PathBuf::from(dir),
            _ => panic!("{CAPTURE_ENV} must name the directory the capture is written to"),
        }
    }

    /// An empty capture of `image` (the committed object `object` names) run on `gpu`.
    /// Refuses a directory that already holds a finished capture.
    pub fn new(dir: PathBuf, gpu: &Gpu, object: &'static str, image: &[u8]) -> Capture {
        assert!(
            !dir.join("index.json").exists(),
            "{}: already holds a capture (index.json); refusing to overwrite it",
            dir.display()
        );
        for sub in ["blobs", "launches"] {
            std::fs::create_dir_all(dir.join(sub))
                .unwrap_or_else(|e| panic!("create {}/{sub}: {e}", dir.display()));
        }
        Capture {
            dir,
            arch: gpu.arch.clone(),
            object,
            object_sha256: format!("{:x}", Sha256::digest(image)),
            blobs: HashMap::new(),
            launches: Vec::new(),
            blob_bytes: 0,
            total_bytes: 0,
            set: None,
        }
    }

    /// The launches until the next call are launch set `launch` of vocab `vocab`.
    pub fn begin_set(&mut self, vocab: usize, launch: usize, ctx: &str) {
        self.set = Some(SetCtx {
            vocab,
            launch,
            ctx: ctx.to_string(),
        });
    }

    fn write_file(&mut self, rel: &str, bytes: &[u8]) {
        let path = self.dir.join(rel);
        std::fs::write(&path, bytes).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
        self.total_bytes += bytes.len() as u64;
    }

    /// Write `words` (little endian) as blob `name` once and return its `{file, bytes,
    /// sha256}` entry. A second request for the same name must carry the same bytes (the
    /// logits of a launch set are shared by the launches that read them).
    fn blob(&mut self, name: &str, words: &[u32]) -> Value {
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let sha256 = format!("{:x}", Sha256::digest(&bytes));
        let file = format!("blobs/{name}.bin");
        match self.blobs.get(name) {
            Some(have) => assert!(
                have.bytes == bytes.len() && have.sha256 == sha256,
                "capture blob {name} changed between the launches that share it"
            ),
            None => {
                self.write_file(&file, &bytes);
                self.blob_bytes += bytes.len() as u64;
                self.blobs.insert(
                    name.to_string(),
                    Blob {
                        bytes: bytes.len(),
                        sha256: sha256.clone(),
                    },
                );
            }
        }
        json!({"file": file, "bytes": bytes.len(), "sha256": sha256})
    }

    /// Record the launch `call` just made: `before` is [`snapshot`] taken before it. Logits
    /// are asserted unchanged and stored once; every other buffer stores both phases.
    fn record(&mut self, gpu: &mut Gpu, call: &Call<'_>, before: Vec<Vec<u32>>, imp: &str) {
        let (vocab, launch, set_ctx) = {
            let s = self
                .set
                .as_ref()
                .expect("Capture::begin_set before a launch");
            (s.vocab, s.launch, s.ctx.clone())
        };
        let ctx = format!("{set_ctx} [{imp}] {}", call.label);
        let bufs = buffers(call);
        assert_eq!(bufs.len(), before.len(), "{ctx}: snapshot buffer count");
        let prefix = format!("v{vocab}_l{launch:03}");
        let mut entries = Vec::with_capacity(bufs.len());
        for (b, before) in bufs.iter().zip(before) {
            let after = download_words(gpu, b.tensor, b.tensor.numel());
            let addr = b.tensor.buf.as_ptr() as u64;
            let unchanged = before == after;
            let (before_blob, after_blob) = if b.read_only {
                assert_words_eq(
                    &after,
                    &before,
                    1,
                    &format!(
                        "lane {} logits after vs before the launch",
                        b.lane.unwrap_or(0)
                    ),
                    &ctx,
                );
                let name = format!(
                    "{prefix}_logits_{}_l{}",
                    call.logits_tag,
                    b.lane.expect("logits buffers carry a lane")
                );
                (self.blob(&name, &before), None)
            } else {
                let stem = format!("{prefix}_{}_{}", call.label, b.role);
                let before_blob = self.blob(&format!("{stem}_before"), &before);
                let after_blob = self.blob(&format!("{stem}_after"), &after);
                (before_blob, Some(after_blob))
            };
            entries.push(json!({
                "role": b.role,
                "lane": b.lane,
                "addr": addr,
                "bytes": before.len() * 4,
                "before": before_blob,
                "after": after_blob,
                "after_equals_before": unchanged,
            }));
        }
        let header = json!({
            "format": CAPTURE_FORMAT,
            "arch": self.arch,
            "object": self.object,
            "symbol": SYMBOL,
            "vocab": vocab,
            "launch": launch,
            "ctx": set_ctx,
            "label": call.label,
            "mode": call.mode,
            "grid": [call.layout.rows, 1, 1],
            "block": [BLOCK, 1, 1],
            "lanes": call.layout.lanes,
            "rows": call.layout.rows,
            "kernargs": {
                "region_ptr": call.region.buf.as_ptr() as u64,
                "lanes_off": call.layout.lanes_off(),
                "rows_off": call.layout.rows_off(),
                "ptab_ptr": device_ptr(call.ptab),
                "mode": call.mode,
                "cand_local": device_ptr(call.cand_local),
                "cand_peer": device_ptr(call.cand_peer),
            },
            "buffers": entries,
        });
        let rel = format!("launches/L{:05}.json", self.launches.len());
        let bytes = serde_json::to_vec_pretty(&header).expect("launch header json");
        self.write_file(&rel, &bytes);
        self.launches.push(json!({
            "header": rel,
            "vocab": vocab,
            "launch": launch,
            "label": call.label,
            "mode": call.mode,
        }));
    }

    /// Write `index.json` and report the bytes written.
    pub fn finish(mut self) {
        let index = json!({
            "format": CAPTURE_FORMAT,
            "arch": self.arch,
            "object": self.object,
            "object_sha256": self.object_sha256,
            "symbol": SYMBOL,
            "block": BLOCK,
            "kernarg_order": [
                "region_ptr", "lanes_off", "rows_off", "ptab_ptr", "mode", "cand_local", "cand_peer"
            ],
            "blob_bytes": self.blob_bytes,
            "launches": self.launches,
        });
        let bytes = serde_json::to_vec_pretty(&index).expect("index json");
        self.write_file("index.json", &bytes);
        eprintln!(
            "lane draw emu capture: {} launches, {} blob bytes, {} bytes ({:.1} MiB) in total written to {}",
            self.launches.len(),
            self.blob_bytes,
            self.total_bytes,
            self.total_bytes as f64 / (1024.0 * 1024.0),
            self.dir.display()
        );
    }
}

impl Impl {
    /// [`Impl::launch`] with the buffers and kernarg values of `call`; when `capture` is
    /// set, every buffer is snapshotted before and after (device synchronised each time).
    /// The launch itself is the same call either way.
    pub fn launch_call(&self, gpu: &mut Gpu, call: &Call<'_>, capture: Option<&mut Capture>) {
        let mut capture = capture;
        let before = capture.as_mut().map(|_| snapshot(gpu, call));
        self.launch(
            gpu,
            call.region,
            call.layout,
            call.ptab,
            call.mode,
            device_ptr(call.cand_local),
            device_ptr(call.cand_peer),
        );
        if let (Some(capture), Some(before)) = (capture, before) {
            capture.record(gpu, call, before, self.name);
        }
    }
}
