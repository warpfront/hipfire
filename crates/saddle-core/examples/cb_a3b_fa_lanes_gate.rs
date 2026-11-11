// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//
// Gate for the lane-batched A3B (H16/KV2, head_dim 256, tile 128, fp8 VMM KV)
// FullAttention mixer of the exact CbVmm route. For lane counts N in
// {1, 2, 3, 8} (different data per lane, different positions incl. the
// tile-boundary lengths 1/127/128/129/255/256/257, shuffled row -> slot
// maps, a row-group-split launch) every lane's results are BYTE-identical to
// the singleton launches the product's default route runs:
//
//   qwen35_fa_prep_gfx1100 (16Q/2K)           vs qwen35_a3b_fa_prep_rows
//   kv_cache_write_fp8_e4m3 K and V           vs kv_cache_write_batched_vmm K, V
//   attention_flash_fp8_e4m3_tile (+ gate)    vs attention_flash_decode_vmm_gated_mq_rotate
//     (group-8 tile + attention_flash_q8_0_reduce_gated_mq_rotate_gfx1201)
//
// compared on q / gate / k (prep), every KV byte of rows [0, p] of both caches
// and the rotated `wo` input, plus the plain (ungated) group-8 route
// (`attention_flash_decode_vmm`) against the singleton's dsplit pair. Negative
// controls: one perturbed V history byte / one perturbed q byte must change
// exactly that lane's output; the group-8 gated entry refuses the group-6
// plan; the host row validator refuses a wrong-epoch row.
//
// Build + run (card A, via the fabric runner):
//   cargo build --release -p saddle-core --features lab,rdna-compute/deltanet \
//       --example cb_a3b_fa_lanes_gate
//   <target>/release/examples/cb_a3b_fa_lanes_gate

use hip_bridge::HipResult;
use rdna_compute::attention::rows_batched::FaPrepRow;
use rdna_compute::attention::{VmmFlashDecodeRoute, VmmKvFormat, VmmKvSide};
use rdna_compute::kv_slots::{validate_vmm_rows, VmmKvSlotDesc};
use rdna_compute::{DType, Gpu, GpuTensor};
use saddle_core::kv::{KvBackend, KvCache, KvDims, KvLayers, KvMode};

const NH: usize = 16;
const NKV: usize = 2;
const HD: usize = 256;
const MAX_SEQ: usize = 4096;
const Q_ROW: usize = NH * HD;
const KV_ROW: usize = NKV * HD;
const EPS: f32 = 1e-6;
const THETA: f32 = 10_000_000.0;
const LANES: usize = 8;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn vec(&mut self, n: usize, scale: f32, bias: f32) -> Vec<f32> {
        (0..n).map(|_| self.next() * scale + bias).collect()
    }
}

fn err(msg: impl Into<String>) -> hip_bridge::HipError {
    hip_bridge::HipError::new(0, &msg.into())
}

struct Checks {
    pass: usize,
    fails: Vec<String>,
}
impl Checks {
    fn check(&mut self, ok: bool, what: &str) {
        println!("[{}] {what}", if ok { "PASS" } else { "FAIL" });
        if ok {
            self.pass += 1;
        } else {
            self.fails.push(what.into());
        }
    }
}

fn i32_bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// 4-byte-element i32 tensor (row-group views address elements).
fn up_i32(gpu: &mut Gpu, v: &[i32]) -> HipResult<GpuTensor> {
    let t = gpu.zeros(&[v.len().max(1)], DType::F32)?;
    gpu.hip.memcpy_htod(&t.buf, &i32_bytes(v))?;
    Ok(t)
}

fn up_f(gpu: &mut Gpu, v: &[f32]) -> HipResult<GpuTensor> {
    gpu.upload_f32(v, &[v.len()])
}

fn bytes(gpu: &Gpu, t: &GpuTensor) -> HipResult<Vec<u8>> {
    gpu.download_raw_bytes(t)
}

fn p(t: &GpuTensor) -> u64 {
    t.buf.as_ptr() as u64
}

fn new_owner(gpu: &mut Gpu) -> HipResult<KvCache> {
    let dims = KvDims {
        layers: KvLayers::Mask(vec![true]),
        n_kv_heads: NKV,
        head_dim: HD,
        max_seq: MAX_SEQ,
        physical_cap: Some(MAX_SEQ),
    };
    KvCache::from_mode_with_backend(KvMode::Fp8, KvBackend::Vmm, gpu, &dims)
}

/// One request lane: its VMM owner, its isolated legacy (singleton) cache and
/// the per-lane random data.
struct Lane {
    owner: KvCache,
    leg_k: GpuTensor,
    leg_v: GpuTensor,
    hist_k: Vec<f32>,
    hist_v: Vec<f32>,
    desc: VmmKvSlotDesc,
}

fn row_bytes() -> usize {
    VmmKvFormat::Fp8E4M3.bytes_per_token(NKV, HD)
}

fn vmm_prefix(gpu: &Gpu, t: &GpuTensor, len: usize) -> HipResult<Vec<u8>> {
    let mut out = vec![0u8; len * row_bytes()];
    gpu.hip.memcpy_dtoh_at(&mut out, &t.buf, 0)?;
    Ok(out)
}

fn bits_eq(a: &[u8], b: &[u8]) -> bool {
    a == b
}

/// Per-step inputs of one lane.
struct StepIn {
    q_inter: Vec<f32>,
    k_raw: Vec<f32>,
    v_raw: Vec<f32>,
}

fn step_in(r: &mut Lcg) -> StepIn {
    StepIn {
        q_inter: r.vec(NH * HD * 2, 2.0, 0.0),
        k_raw: r.vec(KV_ROW, 2.0, 0.0),
        v_raw: r.vec(KV_ROW, 1.0, 0.0),
    }
}

/// Singleton reference of one lane: the product's default route sequence.
struct RefOut {
    q: Vec<u8>,
    gate: Vec<u8>,
    k: Vec<u8>,
    out_rot: Vec<u8>,
    out_plain: Vec<u8>,
}

#[allow(clippy::too_many_arguments)]
fn singleton(
    gpu: &mut Gpu,
    lane: &Lane,
    pos: usize,
    si: &StepIn,
    qw: &GpuTensor,
    kw: &GpuTensor,
) -> HipResult<RefOut> {
    let q_inter = up_f(gpu, &si.q_inter)?;
    let q = gpu.zeros(&[Q_ROW], DType::F32)?;
    let gate = gpu.zeros(&[Q_ROW], DType::F32)?;
    let k = up_f(gpu, &si.k_raw)?;
    let v = up_f(gpu, &si.v_raw)?;
    let out_rot = gpu.zeros(&[Q_ROW], DType::F32)?;
    let out_plain = gpu.zeros(&[Q_ROW], DType::F32)?;
    let pos_t = up_i32(gpu, &[pos as i32])?;
    let part = gpu.zeros(&[NH * (MAX_SEQ / 128) * (2 + HD)], DType::F32)?;
    gpu.qwen35_fa_prep_gfx1100(
        &q_inter, &q, &gate, &k, qw, kw, &pos_t.buf, EPS, THETA, NH, NKV,
    )?;
    gpu.kv_cache_write_fp8_e4m3(&lane.leg_k, &k, &pos_t.buf, NKV, HD)?;
    gpu.kv_cache_write_fp8_e4m3(&lane.leg_v, &v, &pos_t.buf, NKV, HD)?;
    // Fused epilogue (the A3B default): gated MQ-rotate reducer.
    gpu.attention_flash_fp8_e4m3_tile(
        &q,
        &lane.leg_k,
        &lane.leg_v,
        &out_rot,
        &pos_t.buf,
        pos + 1,
        NH,
        NKV,
        HD,
        MAX_SEQ,
        &part,
        Some(&gate),
        None,
    )?;
    // Unfused epilogue: the dsplit reduce pair.
    gpu.attention_flash_fp8_e4m3_tile(
        &q,
        &lane.leg_k,
        &lane.leg_v,
        &out_plain,
        &pos_t.buf,
        pos + 1,
        NH,
        NKV,
        HD,
        MAX_SEQ,
        &part,
        None,
        None,
    )?;
    gpu.hip.device_synchronize()?;
    let r = RefOut {
        q: bytes(gpu, &q)?,
        gate: bytes(gpu, &gate)?,
        k: bytes(gpu, &k)?,
        out_rot: bytes(gpu, &out_rot)?,
        out_plain: bytes(gpu, &out_plain)?,
    };
    for t in [q_inter, q, gate, k, v, out_rot, out_plain, pos_t, part] {
        gpu.free_tensor(t)?;
    }
    Ok(r)
}

struct BatchOut {
    q: Vec<u8>,
    gate: Vec<u8>,
    k: Vec<u8>,
    out_rot: Vec<u8>,
    out_plain: Vec<u8>,
}

/// One batched step over `rows` = (slot, position, input). `group` caps the
/// rows per attention launch (0 = all), exercising the partials-group loop.
#[allow(clippy::too_many_arguments)]
fn batched(
    gpu: &mut Gpu,
    descs_dev: &GpuTensor,
    rows: &[(usize, usize, &StepIn)],
    qw: &GpuTensor,
    kw: &GpuTensor,
    group: usize,
) -> HipResult<BatchOut> {
    let n = rows.len();
    let q_inter: Vec<f32> = rows.iter().flat_map(|r| r.2.q_inter.clone()).collect();
    let k_raw: Vec<f32> = rows.iter().flat_map(|r| r.2.k_raw.clone()).collect();
    let v_raw: Vec<f32> = rows.iter().flat_map(|r| r.2.v_raw.clone()).collect();
    let b_qi = up_f(gpu, &q_inter)?;
    let b_q = gpu.zeros(&[n * Q_ROW], DType::F32)?;
    let b_gate = gpu.zeros(&[n * Q_ROW], DType::F32)?;
    let b_k = up_f(gpu, &k_raw)?;
    let b_v = up_f(gpu, &v_raw)?;
    let b_rot = gpu.zeros(&[n * Q_ROW], DType::F32)?;
    let b_plain = gpu.zeros(&[n * Q_ROW], DType::F32)?;
    let pos: Vec<i32> = rows.iter().map(|r| r.1 as i32).collect();
    let slot: Vec<i32> = rows.iter().map(|r| r.0 as i32).collect();
    let pos_t = up_i32(gpu, &pos)?;
    let slot_t = up_i32(gpu, &slot)?;
    // Rows twin: every row points at its own batch rows and position word.
    let fa_rows: Vec<FaPrepRow> = (0..n)
        .map(|r| FaPrepRow {
            q_interleaved: p(&b_qi) + (r * NH * HD * 2 * 4) as u64,
            q: p(&b_q) + (r * Q_ROW * 4) as u64,
            gate: p(&b_gate) + (r * Q_ROW * 4) as u64,
            k: p(&b_k) + (r * KV_ROW * 4) as u64,
            pos: p(&pos_t) + (r * 4) as u64,
        })
        .collect();
    let table = gpu.zeros(&[n * 40 / 4], DType::F32)?;
    let tags: Vec<u64> = (0..n as u64).map(|i| 1000 + i).collect();
    gpu.stage_qwen35_a3b_fa_prep_rows(&table, &fa_rows, &tags, &tags)?;
    gpu.qwen35_a3b_fa_prep_rows(&table, n, qw, kw, EPS, THETA)?;
    // Batched VMM KV write (K and V), then the flash decode.
    let rows_desc: Vec<VmmKvSlotDesc> = (0..LANES).map(|_| VmmKvSlotDesc::MASKED).collect();
    let _ = rows_desc;
    for (side, src) in [(VmmKvSide::K, &b_k), (VmmKvSide::V, &b_v)] {
        gpu.kv_cache_write_batched_vmm(
            VmmKvFormat::Fp8E4M3,
            side,
            src,
            &pos_t,
            NKV,
            HD,
            n,
            descs_dev,
            &slot_t,
        )?;
    }
    let max_ctx = rows.iter().map(|r| r.1 + 1).max().unwrap();
    let plan = gpu.vmm_flash_decode_plan(VmmKvFormat::Fp8E4M3, NH, NKV, HD, MAX_SEQ, max_ctx)?;
    if plan.route != VmmFlashDecodeRoute::Fp8Gqa8Gfx1201 {
        return Err(err(format!(
            "plan route {:?} != Fp8Gqa8Gfx1201",
            plan.route
        )));
    }
    let per_row = plan.partial_floats_per_row;
    let g = if group == 0 { n } else { group.min(n) };
    let partials = gpu.zeros(&[g * per_row], DType::F32)?;
    for gated in [true, false] {
        let mut r0 = 0;
        while r0 < n {
            let len = g.min(n - r0);
            let q = b_q.sub_offset(r0 * Q_ROW, len * Q_ROW);
            let positions = pos_t.sub_offset(r0, len);
            let row_slot = slot_t.sub_offset(r0, len);
            if gated {
                gpu.attention_flash_decode_vmm_gated_mq_rotate(
                    &plan,
                    &q,
                    &b_gate.sub_offset(r0 * Q_ROW, len * Q_ROW),
                    &b_rot.sub_offset(r0 * Q_ROW, len * Q_ROW),
                    &positions,
                    NH,
                    NKV,
                    HD,
                    len,
                    &partials,
                    descs_dev,
                    &row_slot,
                )?;
            } else {
                gpu.attention_flash_decode_vmm(
                    &plan,
                    &q,
                    &b_plain.sub_offset(r0 * Q_ROW, len * Q_ROW),
                    &positions,
                    NH,
                    NKV,
                    HD,
                    len,
                    &partials,
                    descs_dev,
                    &row_slot,
                )?;
            }
            r0 += len;
        }
    }
    gpu.hip.device_synchronize()?;
    let out = BatchOut {
        q: bytes(gpu, &b_q)?,
        gate: bytes(gpu, &b_gate)?,
        k: bytes(gpu, &b_k)?,
        out_rot: bytes(gpu, &b_rot)?,
        out_plain: bytes(gpu, &b_plain)?,
    };
    for t in [
        b_qi, b_q, b_gate, b_k, b_v, b_rot, b_plain, pos_t, slot_t, table, partials,
    ] {
        gpu.free_tensor(t)?;
    }
    Ok(out)
}

fn row<'a>(b: &'a [u8], r: usize, floats: usize) -> &'a [u8] {
    &b[r * floats * 4..(r + 1) * floats * 4]
}

/// Compare every lane of one batched step against its singleton reference.
/// Returns per-row pass flags (prep q/gate/k, rotated out, plain out).
fn compare(refs: &[RefOut], got: &BatchOut) -> Vec<bool> {
    refs.iter()
        .enumerate()
        .map(|(r, rf)| {
            bits_eq(&rf.q, row(&got.q, r, Q_ROW))
                && bits_eq(&rf.gate, row(&got.gate, r, Q_ROW))
                && bits_eq(&rf.k, row(&got.k, r, KV_ROW))
                && bits_eq(&rf.out_rot, row(&got.out_rot, r, Q_ROW))
                && bits_eq(&rf.out_plain, row(&got.out_plain, r, Q_ROW))
        })
        .collect()
}

fn kv_equal(gpu: &Gpu, lanes: &[Lane], upto: &[usize]) -> HipResult<bool> {
    // `upto[i]`: inclusive last row of lane i that must match.
    let mut ok = true;
    for (l, &p_last) in lanes.iter().zip(upto) {
        let len = p_last + 1;
        ok &= vmm_prefix(gpu, &l.owner.k_gpu[0], len)? == vmm_prefix(gpu, &l.leg_k, len)?
            && vmm_prefix(gpu, &l.owner.v_gpu[0], len)? == vmm_prefix(gpu, &l.leg_v, len)?;
    }
    Ok(ok)
}

fn write_vmm_rows(
    gpu: &mut Gpu,
    descs_dev: &GpuTensor,
    side: VmmKvSide,
    slot: usize,
    pos: usize,
    data: &[f32],
) -> HipResult<()> {
    let src = up_f(gpu, data)?;
    let pos_t = up_i32(gpu, &[pos as i32])?;
    let slot_t = up_i32(gpu, &[slot as i32])?;
    gpu.kv_cache_write_batched_vmm(
        VmmKvFormat::Fp8E4M3,
        side,
        &src,
        &pos_t,
        NKV,
        HD,
        1,
        descs_dev,
        &slot_t,
    )?;
    for t in [src, pos_t, slot_t] {
        gpu.free_tensor(t)?;
    }
    Ok(())
}

fn run(gpu: &mut Gpu, c: &mut Checks) -> HipResult<()> {
    println!("=== A3B FA lanes gate arch={} ===", gpu.arch);
    let mut r = Lcg(0xA3B_FA);
    let qw = up_f(gpu, &r.vec(HD, 0.3, 1.0))?;
    let kw = up_f(gpu, &r.vec(HD, 0.3, 1.0))?;
    let rb = row_bytes();

    // ---- lanes: VMM owner + isolated legacy cache, random history ---------
    let budget = usize::MAX;
    let mut lanes: Vec<Lane> = vec![];
    for i in 0..LANES {
        let mut owner = new_owner(gpu)?;
        owner.provision_vmm_positions(gpu, MAX_SEQ, budget)?;
        let desc = owner.vmm_slot_descs(gpu)?[0];
        let hist_k = r.vec(MAX_SEQ * KV_ROW, 1.0 + 0.1 * i as f32, 0.0);
        let hist_v = r.vec(MAX_SEQ * KV_ROW, 1.0, 0.0);
        let leg_k = gpu.zeros(&[MAX_SEQ * rb / 4], DType::F32)?;
        let leg_v = gpu.zeros(&[MAX_SEQ * rb / 4], DType::F32)?;
        lanes.push(Lane {
            owner,
            leg_k,
            leg_v,
            hist_k,
            hist_v,
            desc,
        });
    }
    let descs: Vec<VmmKvSlotDesc> = lanes.iter().map(|l| l.desc).collect();
    let descs_dev = gpu.upload_raw(VmmKvSlotDesc::as_bytes(&descs), &[descs.len() * 32])?;
    let gens: Vec<u64> = descs.iter().map(|d| d.owner_generation).collect();
    // History rows [0, MAX_SEQ): VMM via the batched `_vmm` writer, legacy via
    // the legacy batched writer; equal by the G1 gate, re-checked below.
    let all_pos: Vec<i32> = (0..MAX_SEQ as i32).collect();
    let pos_t = up_i32(gpu, &all_pos)?;
    for (i, l) in lanes.iter().enumerate() {
        let slot_t = up_i32(gpu, &vec![i as i32; MAX_SEQ])?;
        validate_vmm_rows(
            &descs,
            &vec![i as i32; MAX_SEQ],
            &all_pos,
            &vec![gens[i]; MAX_SEQ],
        )
        .map_err(err)?;
        for (side, data, leg) in [
            (VmmKvSide::K, &l.hist_k, &l.leg_k),
            (VmmKvSide::V, &l.hist_v, &l.leg_v),
        ] {
            let src = up_f(gpu, data)?;
            gpu.kv_cache_write_batched_vmm(
                VmmKvFormat::Fp8E4M3,
                side,
                &src,
                &pos_t,
                NKV,
                HD,
                MAX_SEQ,
                &descs_dev,
                &slot_t,
            )?;
            gpu.kv_cache_write_fp8_e4m3_batched(leg, &src, &pos_t, NKV, HD, MAX_SEQ)?;
            gpu.free_tensor(src)?;
        }
        gpu.free_tensor(slot_t)?;
    }
    gpu.free_tensor(pos_t)?;
    gpu.hip.device_synchronize()?;
    c.check(
        kv_equal(gpu, &lanes, &[MAX_SEQ - 1; LANES])?,
        "history KV (all lanes, 4096 rows) VMM bytes == legacy bytes",
    );

    // ---- plan admission ------------------------------------------------
    let plan = gpu.vmm_flash_decode_plan(VmmKvFormat::Fp8E4M3, NH, NKV, HD, MAX_SEQ, 300)?;
    c.check(
        plan.route == VmmFlashDecodeRoute::Fp8Gqa8Gfx1201 && plan.tile_size == 128,
        "plan: H16/KV2/hd256 fp8 -> Fp8Gqa8Gfx1201, tile 128",
    );
    c.check(
        gpu.vmm_flash_decode_plan(VmmKvFormat::Fp8E4M3, 24, 4, HD, MAX_SEQ, 300)?
            .route
            == VmmFlashDecodeRoute::Fp8GqaGfx1201,
        "plan: H24/KV4 stays the group-6 route",
    );
    c.check(
        gpu.vmm_flash_decode_plan(VmmKvFormat::Fp8E4M3, 16, 4, HD, MAX_SEQ, 300)
            .is_err(),
        "plan: H16/KV4 (no pair) refused",
    );
    {
        let g6 = gpu.vmm_flash_decode_plan(VmmKvFormat::Fp8E4M3, 24, 4, HD, MAX_SEQ, 300)?;
        let dummy = gpu.zeros(&[Q_ROW * 8], DType::F32)?;
        let pos1 = up_i32(gpu, &[0])?;
        let part = gpu.zeros(&[g6.partial_floats_per_row], DType::F32)?;
        c.check(
            gpu.attention_flash_decode_vmm_gated_mq_rotate(
                &g6, &dummy, &dummy, &dummy, &pos1, 24, 4, HD, 1, &part, &descs_dev, &pos1,
            )
            .is_err(),
            "gated entry refuses the group-6 plan",
        );
        for t in [dummy, pos1, part] {
            gpu.free_tensor(t)?;
        }
    }

    // ---- configurations ---------------------------------------------------
    // (positions per row, slot per row, row group size)
    let cases: Vec<(Vec<usize>, Vec<usize>, usize)> = vec![
        (vec![0], vec![0], 0),
        (vec![126], vec![1], 0),
        (vec![127], vec![2], 0),
        (vec![128], vec![3], 0),
        (vec![254], vec![4], 0),
        (vec![255], vec![5], 0),
        (vec![256], vec![6], 0),
        (vec![1000], vec![7], 0),
        (vec![MAX_SEQ - 1], vec![0], 0),
        (vec![0, 127], vec![0, 1], 0),
        (vec![128, 129], vec![3, 2], 0),
        (vec![255, 256], vec![1, 0], 0),
        (vec![256, 3000], vec![5, 4], 0),
        (vec![0, 128, 256], vec![2, 0, 1], 0),
        (
            vec![0, 126, 127, 128, 254, 255, 256, 1000],
            (0..8).collect(),
            0,
        ),
        (
            vec![MAX_SEQ - 1, 1, 2047, 129, 128, 127, 3, 700],
            vec![7, 6, 5, 4, 3, 2, 1, 0],
            0,
        ),
        (
            vec![0, 126, 127, 128, 254, 255, 256, 1000],
            vec![3, 1, 7, 0, 5, 2, 6, 4],
            3,
        ),
        (vec![255; 8], (0..8).collect(), 0),
    ];
    let mut seed = Lcg(0x57E9);
    let mut perturb_case: Option<(Vec<usize>, Vec<usize>, Vec<StepIn>)> = None;
    for (ci, (positions, slots, group)) in cases.iter().enumerate() {
        let n = positions.len();
        let ins: Vec<StepIn> = (0..n).map(|_| step_in(&mut seed)).collect();
        let mut refs = vec![];
        for rr in 0..n {
            refs.push(singleton(
                gpu,
                &lanes[slots[rr]],
                positions[rr],
                &ins[rr],
                &qw,
                &kw,
            )?);
        }
        let rows: Vec<(usize, usize, &StepIn)> = (0..n)
            .map(|rr| (slots[rr], positions[rr], &ins[rr]))
            .collect();
        validate_vmm_rows(
            &descs,
            &slots.iter().map(|&s| s as i32).collect::<Vec<_>>(),
            &positions.iter().map(|&q| q as i32).collect::<Vec<_>>(),
            &slots.iter().map(|&s| gens[s]).collect::<Vec<_>>(),
        )
        .map_err(err)?;
        let got = batched(gpu, &descs_dev, &rows, &qw, &kw, *group)?;
        let flags = compare(&refs, &got);
        // KV rows written: every touched lane's rows [0, p] equal.
        let mut upto = vec![0usize; LANES];
        for rr in 0..n {
            upto[slots[rr]] = positions[rr];
        }
        let touched: Vec<usize> = slots.clone();
        let kv_ok = {
            let mut ok = true;
            for &s in &touched {
                let len = upto[s] + 1;
                ok &= vmm_prefix(gpu, &lanes[s].owner.k_gpu[0], len)?
                    == vmm_prefix(gpu, &lanes[s].leg_k, len)?
                    && vmm_prefix(gpu, &lanes[s].owner.v_gpu[0], len)?
                        == vmm_prefix(gpu, &lanes[s].leg_v, len)?;
            }
            ok
        };
        let live = refs[0].out_rot.iter().any(|&b| b != 0)
            && refs[0].out_plain.iter().any(|&b| b != 0)
            && refs[0].gate.iter().any(|&b| b != 0);
        c.check(
            flags.iter().all(|&f| f) && kv_ok && live,
            &format!(
                "case {ci}: N={n} pos={positions:?} slots={slots:?} group={group}: \
                 q/gate/k + rotated wo input + plain out + KV rows bitwise == singleton \
                 (rows ok {flags:?}, kv {kv_ok}, nonzero {live})"
            ),
        );
        if n == 8 && perturb_case.is_none() && *group == 0 && slots[0] == 0 && positions[2] == 127 {
            perturb_case = Some((positions.clone(), slots.clone(), ins));
        }
    }

    // ---- negative controls -------------------------------------------------
    let (positions, slots, ins) = perturb_case.expect("perturbation case");
    let n = positions.len();
    let mut refs = vec![];
    for rr in 0..n {
        refs.push(singleton(
            gpu,
            &lanes[slots[rr]],
            positions[rr],
            &ins[rr],
            &qw,
            &kw,
        )?);
    }
    let rows: Vec<(usize, usize, &StepIn)> = (0..n)
        .map(|rr| (slots[rr], positions[rr], &ins[rr]))
        .collect();
    let base = compare(&refs, &batched(gpu, &descs_dev, &rows, &qw, &kw, 0)?);
    c.check(
        base.iter().all(|&f| f),
        "negative-control baseline: unperturbed batch == singleton",
    );
    // (1) one V history value of row 3's slot (position 2) perturbed on the
    // VMM side only: that row's rotated output must differ; others equal.
    let victim = 3usize;
    let vslot = slots[victim];
    let mut pv = lanes[vslot].hist_v[2 * KV_ROW..3 * KV_ROW].to_vec();
    pv[17] += 3.0;
    write_vmm_rows(gpu, &descs_dev, VmmKvSide::V, vslot, 2, &pv)?;
    let got = batched(gpu, &descs_dev, &rows, &qw, &kw, 0)?;
    let flags = compare(&refs, &got);
    c.check(
        !flags[victim] && flags.iter().enumerate().all(|(i, &f)| f || i == victim),
        &format!(
            "negative control: perturbed V history value changes exactly row {victim} ({flags:?})"
        ),
    );
    let orig = lanes[vslot].hist_v[2 * KV_ROW..3 * KV_ROW].to_vec();
    write_vmm_rows(gpu, &descs_dev, VmmKvSide::V, vslot, 2, &orig)?;
    // (2) one q_interleaved value of row 5 perturbed: that row's q and output
    // must differ.
    let mut ins2: Vec<StepIn> = ins
        .iter()
        .map(|s| StepIn {
            q_inter: s.q_inter.clone(),
            k_raw: s.k_raw.clone(),
            v_raw: s.v_raw.clone(),
        })
        .collect();
    ins2[5].q_inter[5] += 1.0;
    let rows2: Vec<(usize, usize, &StepIn)> = (0..n)
        .map(|rr| (slots[rr], positions[rr], &ins2[rr]))
        .collect();
    let got = batched(gpu, &descs_dev, &rows2, &qw, &kw, 0)?;
    let flags = compare(&refs, &got);
    c.check(
        !flags[5] && flags.iter().enumerate().all(|(i, &f)| f || i == 5),
        &format!("negative control: perturbed q value changes exactly row 5 ({flags:?})"),
    );
    // restored data -> equal again (the perturbations were the cause).
    let got = batched(gpu, &descs_dev, &rows, &qw, &kw, 0)?;
    c.check(
        compare(&refs, &got).iter().all(|&f| f),
        "negative control: restoring the inputs restores byte identity",
    );

    // ---- host-side row validator -------------------------------------------
    c.check(
        validate_vmm_rows(&descs, &[0], &[5], &[gens[0] + 1000]).is_err(),
        "wrong-epoch row rejected by the host validator",
    );

    for l in lanes {
        gpu.free_tensor(l.leg_k)?;
        gpu.free_tensor(l.leg_v)?;
        let mut o = l.owner;
        o.release_vmm_after(gpu, None)?;
    }
    gpu.free_tensor(descs_dev)?;
    gpu.free_tensor(qw)?;
    gpu.free_tensor(kw)?;
    Ok(())
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    println!("arch={}", gpu.arch);
    if !gpu.arch_caps.is_gfx1201() {
        eprintln!("requires gfx1201");
        std::process::exit(2);
    }
    let mut c = Checks {
        pass: 0,
        fails: vec![],
    };
    if let Err(e) = run(&mut gpu, &mut c) {
        c.check(false, &format!("aborted: {e}"));
    }
    println!("A3B FA LANES: {} PASS, {} FAIL", c.pass, c.fails.len());
    if c.fails.is_empty() {
        println!("A3B FA LANES RESULT: PASS");
    } else {
        println!("A3B FA LANES RESULT: FAIL");
        for f in &c.fails {
            println!("  FAILED: {f}");
        }
        std::process::exit(1);
    }
}
