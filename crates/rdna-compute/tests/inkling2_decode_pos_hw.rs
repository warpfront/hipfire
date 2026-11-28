// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Hardware gate (gfx1201) for the device-position singleton decode launchers
//! (REDLINE-PLAN §2 / §7 S1):
//!
//! * `Gpu::inkling2_fz_attn_post_decode` (plane BASE pointers + `pos % ring`
//!   formed on the device) against the legacy scalar-slot
//!   `Gpu::inkling2_fz_attn_post` (host-offset plane pointers): bytes of the
//!   clipped Q / Rel, the whole `[3, 6656]` conv state, and every byte of all
//!   sixteen K/V planes (written slot, untouched rows), including signed zero;
//! * `Gpu::inkling2_attn_decode_fa2_pos` against the incumbent
//!   `Gpu::inkling2_attn_decode_fa2` with `Fa2Bias::Rel` over host slot / dist
//!   maps: bytes of `out`, and of the WHOLE `lse` / `oacc` scratch (so the
//!   active-but-empty splits' neutral records and the untouched poison of the
//!   overlaunched splits are compared too), full and SWA rings, the §5
//!   positions, wraps and a dense split-rule sweep, with dirty (NaN) rows
//!   past the live window and poisoned scratch;
//! * kernarg / geometry identity of two recordings at different positions of
//!   one bucket on the same allocations, with a positive control (the legacy
//!   launchers differ), and that every converted raw blob site (`fuse`,
//!   `fuse_post`, `fuse_row`) now lands in the replay recorder.
//!
//! `#[ignore]`d: needs gfx1201. Run through the GPU fabric, one thread:
//!
//!   cargo test -p rdna-compute --release --test inkling2_decode_pos_hw -- --ignored --nocapture --test-threads=1

use std::ffi::c_void;

use rdna_compute::inkling2_attention::{
    inkling2_attn_fa2_splits, Fa2Bias, Inkling2Fa2DecodePos, INKLING2_ATTN_FA2_MERGE_POS,
    INKLING2_ATTN_FA2_PARTIAL_POS,
};
use rdna_compute::inkling2_fuse::{FzRow, Inkling2FzMixNormRot, INKLING2_FZ_MIX_NORM_ROT};
use rdna_compute::inkling2_fuse_post::{
    FzKv16, Inkling2DecodeAddr, Inkling2FzAttnPost, INKLING2_FZ_ATTN_POST,
    INKLING2_FZ_ATTN_POST_DECODE, INKLING2_FZ_CONV_RES, INKLING2_FZ_CONV_STATE,
    INKLING2_FZ_GATE_ROT,
};
use rdna_compute::inkling2_fuse_row::{
    FzGemvSeg, Inkling2FzRouteRow, INKLING2_FZ_GEMV_SEGS, INKLING2_FZ_ROUTE_ROW,
};
use rdna_compute::replay::{ReplayBackendRequest, ReplayController};
use rdna_compute::{DType, Gpu, GpuTensor};

const H: usize = 24;
const KVH: usize = 8;
const D: usize = 128;
const NREL: usize = 16;
const EXT: usize = 1024;
const STRIDE: usize = 6656;
const K_OFF: usize = 0;
const V_OFF: usize = 1024;
const EKV: usize = KVH * D;
const DYN_LD: usize = KVH * 4;
const QK_CLIP: f32 = 4.0;
const REL_CLIP: f32 = 1.5;
const CONV_SCALE: f32 = 0.5;
const V_EPS: f32 = 1e-6;
const MAX_SPLITS: usize = 32;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn unit(&mut self) -> f32 {
        (self.next() as f32 / 4294967296.0) * 2.0 - 1.0
    }
    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.unit() * scale).collect()
    }
}

fn gfx1201() -> Option<Gpu> {
    let gpu = Gpu::init().expect("GPU required");
    if gpu.arch != "gfx1201" {
        eprintln!("skip: {} is not gfx1201", gpu.arch);
        return None;
    }
    Some(gpu)
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn f16_plane(rng: &mut Lcg, rows: usize, scale: f32) -> Vec<u8> {
    (0..rows * D)
        .flat_map(|_| half::f16::from_f32(rng.unit() * scale).to_bits().to_le_bytes())
        .collect()
}

fn put(gpu: &Gpu, t: &GpuTensor, bytes: &[u8]) {
    gpu.memcpy_htod_auto(&t.buf, bytes).expect("htod");
}

fn get(gpu: &Gpu, t: &GpuTensor) -> Vec<u8> {
    gpu.hip.device_synchronize().expect("sync");
    gpu.download_raw_bytes(t).expect("dtoh")
}

fn raw(gpu: &Gpu, bytes: &[u8]) -> GpuTensor {
    gpu.upload_raw(bytes, &[bytes.len()]).expect("upload raw")
}

fn f32t(gpu: &mut Gpu, v: &[f32]) -> GpuTensor {
    gpu.upload_f32(v, &[v.len()]).expect("upload f32")
}

fn assert_bytes_eq(what: &str, ctx: &str, a: &[u8], b: &[u8]) {
    assert_eq!(a.len(), b.len(), "{what} [{ctx}]: length");
    if let Some(i) = (0..a.len()).find(|&i| a[i] != b[i]) {
        let w = i / 4 * 4;
        let word = |s: &[u8]| {
            let mut x = [0u8; 4];
            let end = (w + 4).min(s.len());
            x[..end - w].copy_from_slice(&s[w..end]);
            u32::from_le_bytes(x)
        };
        panic!(
            "{what} [{ctx}]: first differing byte {i} (word {} : {:#010x} vs {:#010x})",
            w / 4,
            word(a),
            word(b)
        );
    }
}

/// Positions of REDLINE-PLAN §5 (plus the first rows and tile edges).
fn plan_positions() -> Vec<usize> {
    let mut p = vec![0usize, 1, 2, 63, 64, 65, 127];
    for r in [
        126..=130usize,
        190..=194,
        1022..=1026,
        2046..=2050,
        4094..=4098,
        8190..=8194,
    ] {
        p.extend(r);
    }
    p.sort_unstable();
    p.dedup();
    p
}

fn pos_bytes(pos: usize) -> [u8; 16] {
    // One u32 then 12 bytes of non-zero padding the kernels must never read.
    let mut b = [0xABu8; 16];
    b[..4].copy_from_slice(&(pos as u32).to_le_bytes());
    b
}

fn plane_ptrs(planes: &[GpuTensor], byte_off: usize) -> [*const c_void; 8] {
    std::array::from_fn(|h| {
        (planes[h].buf.as_ptr() as *const u8).wrapping_add(byte_off) as *const c_void
    })
}

// ---------------------------------------------------------------------------
// post epilogue: legacy scalar slot vs device base
// ---------------------------------------------------------------------------

struct PostArm {
    q: GpuTensor,
    rel: GpuTensor,
    state: GpuTensor,
    kp: Vec<GpuTensor>,
    vp: Vec<GpuTensor>,
}

struct PostFx {
    ring: usize,
    dynw: bool,
    q_scale: GpuTensor,
    rel_scale: GpuTensor,
    k_raw: GpuTensor,
    v_raw: GpuTensor,
    k_sw: GpuTensor,
    v_sw: GpuTensor,
    dyn_k: GpuTensor,
    dyn_v: GpuTensor,
    pos: GpuTensor,
    /// arm 0 = legacy scalar slot, arm 1 = device base.
    arms: [PostArm; 2],
}

impl PostFx {
    fn new(gpu: &mut Gpu, ring: usize, dynw: bool) -> Self {
        let z = |gpu: &mut Gpu, n: usize| f32t(gpu, &vec![0.0f32; n]);
        let planes = |gpu: &mut Gpu| -> Vec<GpuTensor> {
            (0..KVH).map(|_| raw(gpu, &vec![0u8; ring * D * 2])).collect()
        };
        let mut arm = |gpu: &mut Gpu| PostArm {
            q: z(gpu, H * D),
            rel: z(gpu, H * NREL),
            state: z(gpu, 3 * STRIDE),
            kp: planes(gpu),
            vp: planes(gpu),
        };
        let a0 = arm(gpu);
        let a1 = arm(gpu);
        Self {
            ring,
            dynw,
            q_scale: z(gpu, H * D),
            rel_scale: z(gpu, H * NREL),
            k_raw: z(gpu, EKV),
            v_raw: z(gpu, EKV),
            k_sw: z(gpu, 4 * EKV),
            v_sw: z(gpu, 4 * EKV),
            dyn_k: z(gpu, DYN_LD),
            dyn_v: z(gpu, DYN_LD),
            pos: raw(gpu, &pos_bytes(0)),
            arms: [a0, a1],
        }
    }

    fn free(self, gpu: &mut Gpu) {
        let PostFx {
            q_scale,
            rel_scale,
            k_raw,
            v_raw,
            k_sw,
            v_sw,
            dyn_k,
            dyn_v,
            pos,
            arms,
            ..
        } = self;
        for t in [
            q_scale, rel_scale, k_raw, v_raw, k_sw, v_sw, dyn_k, dyn_v, pos,
        ] {
            gpu.free_tensor(t).unwrap();
        }
        for a in arms {
            for t in [a.q, a.rel, a.state].into_iter().chain(a.kp).chain(a.vp) {
                gpu.free_tensor(t).unwrap();
            }
        }
    }

    fn args<'a>(&'a self, arm: &'a PostArm, kv16: FzKv16) -> Inkling2FzAttnPost<'a> {
        Inkling2FzAttnPost {
            q: &arm.q,
            q_scale: &self.q_scale,
            qk_clip: QK_CLIP,
            rel: &arm.rel,
            rel_scale: &self.rel_scale,
            rel_clip: REL_CLIP,
            k_raw: &self.k_raw,
            v_raw: &self.v_raw,
            k_sw: &self.k_sw,
            v_sw: &self.v_sw,
            dyn_kv: self.dynw.then_some((&self.dyn_k, &self.dyn_v, DYN_LD)),
            conv_scale: CONV_SCALE,
            state: &arm.state,
            state_stride: STRIDE,
            k_off: K_OFF,
            v_off: V_OFF,
            state_update: true,
            v_eps: V_EPS,
            k_f32: None,
            v_f32: None,
            kv16: Some(kv16),
            rows: 1,
            nh: H,
            nkv: KVH,
        }
    }

    fn legacy(&self, gpu: &mut Gpu, pos: usize) {
        let off = (pos % self.ring) * D * 2;
        let kv = FzKv16 {
            k: plane_ptrs(&self.arms[0].kp, off),
            v: plane_ptrs(&self.arms[0].vp, off),
            row_ld: 0,
        };
        gpu.inkling2_fz_attn_post(&self.args(&self.arms[0], kv))
            .expect("legacy post");
    }

    fn decode(&self, gpu: &mut Gpu) {
        let kv = FzKv16 {
            k: plane_ptrs(&self.arms[1].kp, 0),
            v: plane_ptrs(&self.arms[1].vp, 0),
            row_ld: 0,
        };
        gpu.inkling2_fz_attn_post_decode(
            &self.args(&self.arms[1], kv),
            Inkling2DecodeAddr {
                pos: &self.pos,
                ring_rows: self.ring as u32,
            },
        )
        .expect("decode post");
    }
}

fn post_case(gpu: &mut Gpu, fx: &PostFx, pos: usize, rng: &mut Lcg) {
    let ring = fx.ring;
    let slot = pos % ring;
    let ctx = format!("ring {ring} dyn {} pos {pos} slot {slot}", fx.dynw);
    let mut k_raw = rng.vec(EKV, 2.0);
    let mut v_raw = rng.vec(EKV, 2.0);
    let mut k_sw = rng.vec(4 * EKV, 0.5);
    let mut v_sw = rng.vec(4 * EKV, 0.5);
    let mut dyn_k = rng.vec(DYN_LD, 0.5);
    let mut dyn_v = rng.vec(DYN_LD, 0.5);
    let q = rng.vec(H * D, 3.0);
    let rel = rng.vec(H * NREL, 3.0);
    let q_scale = rng.vec(H * D, 1.0);
    let rel_scale = rng.vec(H * NREL, 1.0);
    let mut state = rng.vec(3 * STRIDE, 2.0);
    // Channel 0 of head 0 of K and V is exactly -0.0 end to end: history and
    // current input -0.0, every tap +0.0, so the stored F16 and the shifted
    // history carry the sign of zero.
    for (raw_row, sw, dynw, off) in [
        (&mut k_raw, &mut k_sw, &mut dyn_k, K_OFF),
        (&mut v_raw, &mut v_sw, &mut dyn_v, V_OFF),
    ] {
        raw_row[0] = -0.0;
        for t in 0..4 {
            sw[t * EKV] = 0.0;
        }
        dynw[..4].fill(0.0);
        for r in 0..3 {
            state[r * STRIDE + off] = -0.0;
        }
    }
    // Dirty planes: every byte poisoned with random F16 bits (NaNs included).
    let mut init_k: Vec<Vec<u8>> = Vec::new();
    let mut init_v: Vec<Vec<u8>> = Vec::new();
    for _ in 0..KVH {
        for out in [&mut init_k, &mut init_v] {
            let mut b = vec![0u8; ring * D * 2];
            for w in b.chunks_exact_mut(2) {
                w.copy_from_slice(&(rng.next() as u16).to_le_bytes());
            }
            out.push(b);
        }
    }
    for (t, v) in [
        (&fx.q_scale, &q_scale),
        (&fx.rel_scale, &rel_scale),
        (&fx.k_raw, &k_raw),
        (&fx.v_raw, &v_raw),
        (&fx.k_sw, &k_sw),
        (&fx.v_sw, &v_sw),
        (&fx.dyn_k, &dyn_k),
        (&fx.dyn_v, &dyn_v),
    ] {
        put(gpu, t, &f32_bytes(v));
    }
    for a in &fx.arms {
        put(gpu, &a.q, &f32_bytes(&q));
        put(gpu, &a.rel, &f32_bytes(&rel));
        put(gpu, &a.state, &f32_bytes(&state));
        for h in 0..KVH {
            put(gpu, &a.kp[h], &init_k[h]);
            put(gpu, &a.vp[h], &init_v[h]);
        }
    }
    put(gpu, &fx.pos, &pos_bytes(pos));
    fx.legacy(gpu, pos);
    fx.decode(gpu);

    let (a0, a1) = (&fx.arms[0], &fx.arms[1]);
    assert_bytes_eq("q", &ctx, &get(gpu, &a0.q), &get(gpu, &a1.q));
    assert_bytes_eq("rel", &ctx, &get(gpu, &a0.rel), &get(gpu, &a1.rel));
    let (s0, s1) = (get(gpu, &a0.state), get(gpu, &a1.state));
    assert_bytes_eq("conv state", &ctx, &s0, &s1);
    assert_ne!(s0, f32_bytes(&state), "conv state unchanged [{ctx}]");
    // History shifted exactly as the legacy launch did, sign of zero included.
    let s0f: Vec<f32> = s0
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    for off in [K_OFF, V_OFF] {
        assert_eq!(
            s0f[off].to_bits(),
            (-0.0f32).to_bits(),
            "history ch0 -0.0 [{ctx}]"
        );
    }
    for h in 0..KVH {
        for (what, p0, p1, init) in [
            ("k plane", &a0.kp[h], &a1.kp[h], &init_k[h]),
            ("v plane", &a0.vp[h], &a1.vp[h], &init_v[h]),
        ] {
            let (b0, b1) = (get(gpu, p0), get(gpu, p1));
            assert_bytes_eq(&format!("{what} {h}"), &ctx, &b0, &b1);
            let row = slot * D * 2..(slot + 1) * D * 2;
            assert_ne!(&b1[row.clone()], &init[row.clone()], "{what} {h} not written");
            assert_bytes_eq(
                &format!("{what} {h} head"),
                &ctx,
                &b1[..row.start],
                &init[..row.start],
            );
            assert_bytes_eq(
                &format!("{what} {h} tail"),
                &ctx,
                &b1[row.end..],
                &init[row.end..],
            );
        }
    }
    let k0 = get(gpu, &a1.kp[0]);
    let v0 = get(gpu, &a1.vp[0]);
    for (n, p) in [("K", &k0), ("V", &v0)] {
        let at = slot * D * 2;
        assert_eq!(
            u16::from_le_bytes([p[at], p[at + 1]]),
            0x8000,
            "{n} ch0 stored as F16 -0.0 [{ctx}]"
        );
    }
}

#[test]
#[ignore = "requires gfx1201"]
fn post_decode_matches_legacy_scalar_slot_gfx1201() {
    let Some(mut gpu) = gfx1201() else { return };
    let positions = plan_positions();
    let mut rng = Lcg(0x5105_7a7e_c0de_0001);
    let mut cases = 0usize;
    // (ring rows, dynamic conv taps): SWA ring, a smaller SWA ring, the
    // longest full ring, a clipped non-power-of-two full ring.
    for (ring, dynw) in [(1024usize, false), (640, false), (4352, true), (2053, true)] {
        let fx = PostFx::new(&mut gpu, ring, dynw);
        for &p in &positions {
            // Full visits reject pos >= ring (no overwrite).
            if dynw && p >= ring {
                continue;
            }
            post_case(&mut gpu, &fx, p, &mut rng);
            cases += 1;
        }
        // Last legal row of a full ring.
        if dynw {
            post_case(&mut gpu, &fx, ring - 1, &mut rng);
            cases += 1;
        }
        fx.free(&mut gpu);
    }
    println!("post decode vs legacy scalar slot: {cases} cases bit-identical");
}

// ---------------------------------------------------------------------------
// FA2: map launcher vs device-position launcher
// ---------------------------------------------------------------------------

struct Scratch {
    lse: GpuTensor,
    oacc: GpuTensor,
    out: GpuTensor,
}

struct Fa2Fx {
    ring: usize,
    base_k: Vec<Vec<u8>>,
    base_v: Vec<Vec<u8>>,
    k: Vec<GpuTensor>,
    v: Vec<GpuTensor>,
    q: GpuTensor,
    rel: GpuTensor,
    table: GpuTensor,
    sink: GpuTensor,
    pos: GpuTensor,
    old: Scratch,
    new: Scratch,
}

impl Fa2Fx {
    fn new(gpu: &mut Gpu, ring: usize, rng: &mut Lcg) -> Self {
        let base_k: Vec<Vec<u8>> = (0..KVH).map(|_| f16_plane(rng, ring, 1.5)).collect();
        let base_v: Vec<Vec<u8>> = (0..KVH).map(|_| f16_plane(rng, ring, 1.0)).collect();
        let k = base_k.iter().map(|b| raw(gpu, b)).collect();
        let v = base_v.iter().map(|b| raw(gpu, b)).collect();
        let q = f32t(gpu, &rng.vec(H * D, 1.0));
        let rel = f32t(gpu, &rng.vec(H * NREL, 1.0));
        let table = f32t(gpu, &rng.vec(NREL * EXT, 0.5));
        let sink = f32t(gpu, &rng.vec(H, 1.0));
        let pos = raw(gpu, &pos_bytes(0));
        let scratch = |gpu: &mut Gpu| Scratch {
            lse: f32t(gpu, &vec![0.0; MAX_SPLITS * H]),
            oacc: f32t(gpu, &vec![0.0; MAX_SPLITS * H * D]),
            out: f32t(gpu, &vec![0.0; H * D]),
        };
        let old = scratch(gpu);
        let new = scratch(gpu);
        Self {
            ring,
            base_k,
            base_v,
            k,
            v,
            q,
            rel,
            table,
            sink,
            pos,
            old,
            new,
        }
    }

    fn free(self, gpu: &mut Gpu) {
        let Fa2Fx {
            k,
            v,
            q,
            rel,
            table,
            sink,
            pos,
            old,
            new,
            ..
        } = self;
        for t in k.into_iter().chain(v).chain([
            q, rel, table, sink, pos, old.lse, old.oacc, old.out, new.lse, new.oacc, new.out,
        ]) {
            gpu.free_tensor(t).unwrap();
        }
    }

    fn pos_launch(&self, gpu: &mut Gpu, cap: usize) {
        gpu.inkling2_attn_decode_fa2_pos(&Inkling2Fa2DecodePos {
            q: &self.q,
            k_cache: &self.k,
            v_cache: &self.v,
            rel: &self.rel,
            table: &self.table,
            sink: &self.sink,
            lse: &self.new.lse,
            oacc: &self.new.oacc,
            out: &self.new.out,
            addr: Inkling2DecodeAddr {
                pos: &self.pos,
                ring_rows: self.ring as u32,
            },
            n_rel: NREL,
            n_extent: EXT,
            split_cap: cap,
        })
        .expect("fa2 pos");
    }
}

/// `max(splits(min(n, ring)))` over the inclusive key-count bucket of
/// `n_full` (cuts 128 / 192 / 2048 / 4096, final bound 8192), REDLINE-PLAN §3.
fn bucket_cap(n_full: usize, ring: usize) -> usize {
    let cuts = [128usize, 192, 2048, 4096, 8192];
    let hi = *cuts.iter().find(|&&c| n_full <= c).unwrap_or(&n_full);
    let lo = cuts.iter().rev().find(|&&c| c < n_full).map_or(1, |&c| c + 1);
    (lo..=hi)
        .map(|n| inkling2_attn_fa2_splits(n.min(ring)))
        .max()
        .unwrap()
}

fn poison_f32(len: usize, tag: u32) -> Vec<u8> {
    (0..len as u32)
        .flat_map(|i| {
            (0x7fc0_0000u32 | ((i.wrapping_mul(2654435761) ^ tag) & 0x003f_ffff)).to_le_bytes()
        })
        .collect()
}

fn fa2_case(gpu: &mut Gpu, fx: &Fa2Fx, pos: usize, v_neg_zero: bool, tag: &str, checked: &mut usize) {
    let ring = fx.ring;
    let n = (pos + 1).min(ring);
    let start = pos + 1 - n;
    let ctx = format!("ring {ring} pos {pos} n {n} vneg0 {v_neg_zero} {tag}");
    // Planes: dirty NaN rows past the live window (never-written rows).
    for h in 0..KVH {
        let mut kb = fx.base_k[h].clone();
        let mut vb = fx.base_v[h].clone();
        if v_neg_zero {
            for w in vb.chunks_exact_mut(2) {
                w.copy_from_slice(&0x8000u16.to_le_bytes());
            }
        }
        if pos + 1 < ring {
            for r in pos + 1..ring {
                let at = r * D * 2;
                for w in kb[at..at + D * 2].chunks_exact_mut(2) {
                    w.copy_from_slice(&0x7e01u16.to_le_bytes());
                }
                for w in vb[at..at + D * 2].chunks_exact_mut(2) {
                    w.copy_from_slice(&0xfe02u16.to_le_bytes());
                }
            }
        }
        put(gpu, &fx.k[h], &kb);
        put(gpu, &fx.v[h], &vb);
    }
    let slots: Vec<u8> = (0..n)
        .flat_map(|j| (((start + j) % ring) as i32).to_le_bytes())
        .collect();
    let dists: Vec<u8> = (0..n)
        .flat_map(|j| ((pos - (start + j)) as i32).to_le_bytes())
        .collect();
    let d_slots = raw(gpu, &slots);
    let d_dists = raw(gpu, &dists);
    let tagv = pos as u32;
    let (pl, po, pq) = (
        poison_f32(MAX_SPLITS * H, tagv),
        poison_f32(MAX_SPLITS * H * D, tagv),
        poison_f32(H * D, tagv),
    );
    for s in [&fx.old, &fx.new] {
        put(gpu, &s.lse, &pl);
        put(gpu, &s.oacc, &po);
        put(gpu, &s.out, &pq);
    }
    gpu.inkling2_attn_decode_fa2(
        &fx.q,
        &fx.k,
        &fx.v,
        Fa2Bias::Rel {
            rel: &fx.rel,
            table: &fx.table,
            sink: &fx.sink,
            dists: &d_dists,
            n_rel: NREL,
            n_extent: EXT,
        },
        &d_slots,
        &fx.old.lse,
        &fx.old.oacc,
        &fx.old.out,
        n,
        inkling2_attn_fa2_splits(n),
    )
    .expect("fa2 map");
    let (lse_o, oacc_o, out_o) = (
        get(gpu, &fx.old.lse),
        get(gpu, &fx.old.oacc),
        get(gpu, &fx.old.out),
    );
    // The dirty rows are never read: the incumbent output is finite.
    for c in out_o.chunks_exact(4) {
        assert!(
            f32::from_le_bytes([c[0], c[1], c[2], c[3]]).is_finite(),
            "incumbent out not finite [{ctx}]"
        );
    }
    put(gpu, &fx.pos, &pos_bytes(pos));
    let mut caps = vec![
        inkling2_attn_fa2_splits(n),
        bucket_cap(pos + 1, ring),
        MAX_SPLITS,
    ];
    caps.sort_unstable();
    caps.dedup();
    for cap in caps {
        for s in [&fx.new] {
            put(gpu, &s.lse, &pl);
            put(gpu, &s.oacc, &po);
            put(gpu, &s.out, &pq);
        }
        fx.pos_launch(gpu, cap);
        let c = format!("{ctx} cap {cap}");
        assert_bytes_eq("out", &c, &out_o, &get(gpu, &fx.new.out));
        assert_bytes_eq("lse (whole scratch)", &c, &lse_o, &get(gpu, &fx.new.lse));
        assert_bytes_eq("oacc (whole scratch)", &c, &oacc_o, &get(gpu, &fx.new.oacc));
        *checked += 1;
    }
    gpu.free_tensor(d_slots).unwrap();
    gpu.free_tensor(d_dists).unwrap();
}

#[test]
#[ignore = "requires gfx1201"]
fn fa2_pos_matches_map_launcher_plan_positions_gfx1201() {
    let Some(mut gpu) = gfx1201() else { return };
    let mut rng = Lcg(0xfa2_0905_0000_0001);
    let positions = plan_positions();
    let mut checked = 0usize;
    // SWA ring, smaller SWA ring, the longest full ring, a clipped
    // non-power-of-two full ring, a full ring cut exactly at the 1024 break.
    for ring in [1024usize, 640, 4352, 2053, 1025] {
        let full = ring != 1024 && ring != 640;
        let fx = Fa2Fx::new(&mut gpu, ring, &mut rng);
        for &p in &positions {
            if full && p >= ring {
                continue;
            }
            fa2_case(&mut gpu, &fx, p, false, "random v", &mut checked);
            // Signed-zero values: every V row -0.0.
            if matches!(p, 0 | 126 | 190 | 1023 | 1024 | 2047 | 4095) {
                fa2_case(&mut gpu, &fx, p, true, "v = -0.0", &mut checked);
            }
        }
        if full {
            fa2_case(&mut gpu, &fx, ring - 1, false, "last row", &mut checked);
        }
        fx.free(&mut gpu);
    }
    println!("fa2 pos vs map launcher: {checked} (case, split_cap) arms bit-identical");
}

#[test]
#[ignore = "requires gfx1201"]
fn fa2_pos_matches_map_launcher_split_rule_sweep_gfx1201() {
    let Some(mut gpu) = gfx1201() else { return };
    let mut rng = Lcg(0xfa2_5ee9_0000_0002);
    let mut checked = 0usize;
    let ranges: [(usize, Vec<usize>); 3] = [
        (
            4352,
            (0..=300).chain(2040..=2056).chain(4090..=4110).collect(),
        ),
        (
            1024,
            (0..=300)
                .chain(1016..=1032)
                .chain(2040..=2056)
                .chain(4090..=4110)
                .collect(),
        ),
        (2053, (0..=200).chain(1020..=1030).chain(2040..=2052).collect()),
    ];
    for (ring, positions) in ranges {
        let fx = Fa2Fx::new(&mut gpu, ring, &mut rng);
        for p in positions {
            fa2_case(&mut gpu, &fx, p, false, "sweep", &mut checked);
        }
        fx.free(&mut gpu);
    }
    println!("fa2 split-rule sweep: {checked} (case, split_cap) arms bit-identical");
}

// ---------------------------------------------------------------------------
// recordings: kernarg / geometry identity
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
struct Rec {
    kernel: String,
    grid: [u32; 3],
    block: [u32; 3],
    shared: u32,
    kernarg: Vec<u8>,
}

fn record(gpu: &mut Gpu, body: impl FnOnce(&mut Gpu)) -> Vec<Rec> {
    gpu.replay = ReplayController::new_armed(ReplayBackendRequest::Auto);
    gpu.replay.begin_capture().expect("open recording window");
    body(gpu);
    gpu.replay.finish_capture().expect("close recording window");
    gpu.replay
        .recorded_launches()
        .iter()
        .map(|l| Rec {
            kernel: l.kernel.clone(),
            grid: l.grid,
            block: l.block,
            shared: l.shared_mem,
            kernarg: l.kernarg.clone(),
        })
        .collect()
}

#[test]
#[ignore = "requires gfx1201"]
fn decode_launches_record_identical_kernargs_across_positions_gfx1201() {
    let Some(mut gpu) = gfx1201() else { return };
    let mut rng = Lcg(0x4ec0_4d00_0000_0003);
    // (ring, dyn taps, two positions of one bucket): the SWA ring wraps
    // between them, the full ring stays in 193..=2048.
    for (ring, dynw, p1, p2) in [
        (1024usize, false, 300usize, 1500usize),
        (1024, false, 1023, 1024),
        (4352, true, 200, 2000),
        (4352, true, 2049, 4095),
        (640, false, 300, 700),
    ] {
        let pfx = PostFx::new(&mut gpu, ring, dynw);
        let ffx = Fa2Fx::new(&mut gpu, ring, &mut rng);
        let cap = bucket_cap(p1 + 1, ring).max(bucket_cap(p2 + 1, ring));
        assert_eq!(bucket_cap(p1 + 1, ring), bucket_cap(p2 + 1, ring));
        let mut tapes = Vec::new();
        for p in [p1, p2] {
            put(&gpu, &pfx.pos, &pos_bytes(p));
            put(&gpu, &ffx.pos, &pos_bytes(p));
            tapes.push(record(&mut gpu, |g| {
                pfx.decode(g);
                ffx.pos_launch(g, cap);
            }));
        }
        let ctx = format!("ring {ring} p1 {p1} p2 {p2} cap {cap}");
        let names: Vec<&str> = tapes[0].iter().map(|r| r.kernel.as_str()).collect();
        assert_eq!(
            names,
            [
                INKLING2_FZ_ATTN_POST_DECODE,
                INKLING2_ATTN_FA2_PARTIAL_POS,
                INKLING2_ATTN_FA2_MERGE_POS
            ],
            "recorded launches [{ctx}]"
        );
        assert_eq!(tapes[0][1].grid, [cap as u32, 8, 1]);
        assert_eq!(tapes[0][1].block, [128, 1, 1]);
        assert_eq!(tapes[0][2].grid, [24, 1, 1]);
        assert_eq!(tapes[0][2].block, [128, 1, 1]);
        assert_eq!(tapes[0], tapes[1], "decode tapes differ across pos [{ctx}]");
        for (a, b) in tapes[0].iter().zip(&tapes[1]) {
            assert_bytes_eq(&a.kernel, &ctx, &a.kernarg, &b.kernarg);
            assert_eq!(a.kernarg.len() % 16, 0, "{} kernarg 16B padded", a.kernel);
        }

        // Positive control: the legacy launchers bake the position (slot
        // pointer offset / live length), so the same comparison must differ.
        if p1 % ring != p2 % ring {
            let mut leg = Vec::new();
            for p in [p1, p2] {
                leg.push(record(&mut gpu, |g| pfx.legacy(g, p)));
            }
            assert_eq!(leg[0][0].kernel, INKLING2_FZ_ATTN_POST);
            assert_ne!(
                leg[0][0].kernarg, leg[1][0].kernarg,
                "legacy post kernargs equal across slots [{ctx}] (control)"
            );
        }
        pfx.free(&mut gpu);
        ffx.free(&mut gpu);
    }
    println!("decode launches: kernargs/geometry identical across positions of a bucket");
}

#[test]
#[ignore = "requires gfx1201"]
fn converted_blob_sites_reach_the_recorder_gfx1201() {
    let Some(mut gpu) = gfx1201() else { return };
    let ring = 1024;
    let pfx = PostFx::new(&mut gpu, ring, true);
    let ffx = Fa2Fx::new(&mut gpu, ring, &mut Lcg(7));
    let k = 256usize;
    let n = 2304usize;
    let attn = f32t(&mut gpu, &vec![0.5f32; k]);
    let gate = f32t(&mut gpu, &vec![0.25f32; k]);
    let awq = f32t(&mut gpu, &vec![1.0f32; k]);
    let x_rot = f32t(&mut gpu, &vec![0.0f32; k]);
    let cur = f32t(&mut gpu, &vec![0.1f32; EKV]);
    let sw = f32t(&mut gpu, &vec![0.2f32; 4 * EKV]);
    let state = f32t(&mut gpu, &vec![0.3f32; 3 * STRIDE]);
    let out = f32t(&mut gpu, &vec![0.0f32; EKV]);
    let m = 16usize;
    let w = raw(&gpu, &vec![0u8; m * (k / 256) * 136]);
    let y = f32t(&mut gpu, &vec![0.0f32; m]);
    let st = f32t(&mut gpu, &vec![0.7f32; n]);
    let mix_out = f32t(&mut gpu, &vec![0.0f32; n]);
    // route row
    let (ne, tk, bm) = (8usize, 2usize, 16usize);
    let mt = tk * bm;
    let logits = f32t(&mut gpu, &(0..ne).map(|i| i as f32 * 0.1).collect::<Vec<_>>());
    let topk_idx = f32t(&mut gpu, &vec![0.0f32; tk]);
    let topk_w = f32t(&mut gpu, &vec![0.0f32; tk]);
    let counts = f32t(&mut gpu, &vec![0.0f32; ne]);
    let offsets = f32t(&mut gpu, &vec![0.0f32; ne + 1]);
    let sorted = f32t(&mut gpu, &vec![0.0f32; mt]);
    let tiles = f32t(&mut gpu, &vec![0.0f32; mt / bm]);
    let inverse = f32t(&mut gpu, &vec![0.0f32; tk]);
    let xrow = f32t(&mut gpu, &vec![0.3f32; k]);
    let awq_ptrs_host: Vec<u8> = (0..ne)
        .flat_map(|_| (awq.buf.as_ptr() as u64).to_le_bytes())
        .collect();
    let awq_ptrs = raw(&gpu, &awq_ptrs_host);
    put(&gpu, &pfx.pos, &pos_bytes(5));
    put(&gpu, &ffx.pos, &pos_bytes(5));

    let tape = record(&mut gpu, |g| {
        g.inkling2_fz_gate_rot(&attn, &gate, &awq, &x_rot, k, 1)
            .expect("gate_rot");
        g.inkling2_fz_conv_res(&cur, &sw, &state, STRIDE, 0, EKV, 0.5, None, &out, 1, true)
            .expect("conv_res");
        g.inkling2_fz_conv_state(&[(&cur, 0, EKV)], &state, STRIDE, 1)
            .expect("conv_state");
        g.inkling2_fz_gemv_segs(
            &[FzGemvSeg {
                w: &w,
                x_rot: &x_rot,
                y: &y,
                m,
            }],
            k,
        )
        .expect("gemv_segs");
        g.inkling2_fz_mix_norm_rot(&Inkling2FzMixNormRot {
            states: &[&st],
            query: None,
            out: &[FzRow {
                raw: Some((&mix_out, n)),
                targets: Vec::new(),
            }],
            rows: 1,
            n,
            mix_eps: 1e-6,
        })
        .expect("mix_norm_rot");
        let _prepared = g
            .inkling2_fz_route_row(&Inkling2FzRouteRow {
                logits: &logits,
                correction_bias: None,
                eps: 1e-6,
                scale: 1.0,
                topk_idx: &topk_idx,
                topk_w: &topk_w,
                counts: &counts,
                offsets: &offsets,
                sorted: &sorted,
                tiles: &tiles,
                inverse: &inverse,
                x: &xrow,
                awq_ptrs: &awq_ptrs,
                n_experts: ne,
                top_k: tk,
                k,
                m_total_max: mt,
                block_m: bm,
            })
            .expect("route_row");
        pfx.legacy(g, 5);
        pfx.decode(g);
        ffx.pos_launch(g, 8);
    });
    let names: Vec<&str> = tape.iter().map(|r| r.kernel.as_str()).collect();
    println!("recorded: {names:?}");
    for want in [
        INKLING2_FZ_GATE_ROT,
        INKLING2_FZ_CONV_RES,
        INKLING2_FZ_CONV_STATE,
        INKLING2_FZ_GEMV_SEGS,
        INKLING2_FZ_MIX_NORM_ROT,
        INKLING2_FZ_ROUTE_ROW,
        INKLING2_FZ_ATTN_POST,
        INKLING2_FZ_ATTN_POST_DECODE,
        INKLING2_ATTN_FA2_PARTIAL_POS,
        INKLING2_ATTN_FA2_MERGE_POS,
    ] {
        assert_eq!(
            names.iter().filter(|&&n| n == want).count(),
            1,
            "{want} recorded exactly once in {names:?}"
        );
    }
    assert_eq!(tape.len(), 10, "no unrecorded or extra launches: {names:?}");
    for r in &tape {
        assert_eq!(r.kernarg.len() % 16, 0, "{} kernarg 16B padded", r.kernel);
    }
}
