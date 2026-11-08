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
use std::ffi::c_void;

pub const SYMBOL: &str = "lane_leviathan_verify";
const BLOCK: u32 = 512;

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
