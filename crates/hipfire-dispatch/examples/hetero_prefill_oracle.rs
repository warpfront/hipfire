//! Oracle for the expert device's cold prefill (`hetero_prefill::run_cold_prefill`).
//!
//! Synthetic qt44/qt53 experts in pinned host memory (distinct bytes per
//! expert, so an id mix-up shows). Rows of normed activations and a random
//! top-10 route over 512 experts, a third of them cold. The sealed HeteroCold
//! prefill must produce, per row, the weighted sum of the cold slots' rows as
//! Plan 3's `IgpuColdArm` computes them (decode GEMVs over the rotated rows);
//! hot slots must contribute nothing. Summation order and the grouped kernels'
//! activation formats differ, so the gate is a relative L2 bound, reported.
//! Also checks: zero cold slots (all-zero partial), rows past a partial chunk
//! untouched, and bitwise determinism across two runs; then times 2,048 rows.
//!
//! ```text
//! HIP_VISIBLE_DEVICES=1 HIPFIRE_DEVICES=gfx1151 cargo run --release -p hipfire-dispatch --example hetero_prefill_oracle
//! ```

use std::time::Instant;

use hipfire_cpu::quant::CpuQuant;
use hipfire_dispatch::cpu_exec::CpuMoeExpert;
use hipfire_dispatch::families::moe::{
    MoeRecipe, MoeRouteCapability, MoeRouteFormats, MoeRouteGeometry, MoeRoutePolicy,
};
use hipfire_dispatch::hetero_mailbox::{int4_row_bytes, ROWS_F16, ROWS_INT4};
use hipfire_dispatch::hetero_prefill::{
    run_cold_prefill, ColdPrefillConfig, ColdPrefillIo, ColdPrefillLayer, ColdPrefillScratch,
};
use hipfire_dispatch::hetero_responder::ColdLayer;
use hipfire_dispatch::igpu_split::{ColdArmDims, IgpuColdArm};
use rdna_compute::{DType, Gpu};

const N_EXP: usize = 512;
const TOP_K: usize = 10;
const DIM: usize = 2560;
const MI: usize = 640;
const MAX_ROWS: usize = 2048;
const SENTINEL: f32 = -7.0;
/// Relative L2 bound of the cold partial against the decode-GEMV reference on
/// the F16 route (`Qt44Qt53Grouped`). The symmetric route quantizes both GEMMs'
/// activations to int4 (hipfire's own oracles check it against an exact
/// integer reference, not F32), so there the F32 distance is only reported,
/// under a sanity ceiling, and the partition check is the gate.
const REL_L2_MAX: f64 = 2e-2;
const REL_L2_MAX_SYMMETRIC: f64 = 0.25;

/// Finite synthetic weights (`testfix::weight_bytes` with an expert salt).
/// `symmetric` writes `zp = -8 sc` headers (the symmetric IU4 route).
fn weight_bytes(q: CpuQuant, m: usize, k: usize, expert: usize, symmetric: bool) -> Vec<u8> {
    let groups = k / q.group_elems();
    let hdr = q.row_header_bytes();
    // f16: sc = 1/16 (0x2c00), zp = -1/4 (0xb400) or -1/2 (0xb800, symmetric).
    let zp: u16 = if symmetric { 0xb800 } else { 0xb400 };
    let quad: Vec<u8> = [0x2c00u16, zp, 0x2c00, zp]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let mut out = Vec::new();
    for row in 0..m {
        for g in 0..groups {
            let salt = row * groups + g + expert * 7919;
            let mut b: Vec<u8> = (0..hdr + q.group_bytes())
                .map(|i| (((i + salt) * 37 + 11) & 0xFF) as u8)
                .collect();
            match q {
                CpuQuant::Mq4G256V2 => b[..8].copy_from_slice(&quad),
                CpuQuant::Mq4G128V2 => b[..4].copy_from_slice(&quad[..4]),
                _ => unreachable!(),
            }
            if g == 0 {
                out.extend_from_slice(&b[..hdr]);
            }
            out.extend_from_slice(&b[hdr..]);
        }
    }
    out
}

fn pinned(gpu: &Gpu, bytes: usize, coherent: bool) -> *mut u8 {
    let mut flags = hip_bridge::HIP_HOST_MALLOC_MAPPED;
    if coherent {
        flags |= hip_bridge::HIP_HOST_MALLOC_COHERENT;
    }
    let p = gpu
        .hip
        .host_malloc(bytes, flags)
        .expect("host_malloc")
        .cast::<u8>();
    // SAFETY: fresh allocation of `bytes`.
    unsafe { std::ptr::write_bytes(p, 0, bytes) };
    p
}

fn fail(msg: String) -> ! {
    println!("FAIL: {msg}");
    std::process::exit(1);
}

/// f32 -> F16 bits, round-to-nearest-even, for normal-range values (the rows
/// here are |v| <= 0.1, far from F16 overflow; subnormals flush to the nearest
/// normal or zero, which the device cast matches for these inputs).
fn f16_bits(v: f32) -> u16 {
    let u = v.to_bits();
    let sign = ((u >> 16) & 0x8000) as u16;
    let e = ((u >> 23) & 0xff) as i32 - 127 + 15;
    let man = u & 0x7f_ffff;
    if e <= 0 {
        return sign;
    }
    let mut r = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && r & 1 == 1) {
        r += 1;
    }
    sign | r as u16
}

fn f16_value(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let e = ((h >> 10) & 0x1f) as i32;
    if e == 0 {
        return sign * 0.0;
    }
    sign * (1.0 + (h & 0x3ff) as f32 / 1024.0) * 2f32.powi(e - 15)
}

/// xorshift64*, deterministic.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

fn policy(capability: MoeRouteCapability) -> MoeRoutePolicy {
    MoeRoutePolicy {
        capability,
        geometry: MoeRouteGeometry {
            experts: N_EXP,
            top_k: TOP_K,
            hidden: DIM,
            routed_intermediate: MI,
            shared_intermediate: MI,
        },
        formats: MoeRouteFormats {
            router: DType::BF16,
            shared_selector: DType::BF16,
            shared_gate: DType::BF16,
            shared_up: DType::BF16,
            shared_down: DType::MQ4G128V2,
            routed_gate_up: DType::MQ4G256V2,
            routed_down: DType::MQ4G128V2,
        },
        slab_cold_codec: None,
        expert_keep_mask: 0,
    }
}

struct Case {
    rows: usize,
    ids: Vec<i32>,
    weights: Vec<f32>,
    x: Vec<f32>,
}

fn case(rng: &mut Rng, rows: usize) -> Case {
    let mut ids = Vec::with_capacity(rows * TOP_K);
    let mut weights = Vec::with_capacity(rows * TOP_K);
    for _ in 0..rows {
        let mut picked: Vec<i32> = Vec::new();
        while picked.len() < TOP_K {
            let e = (rng.next() % N_EXP as u64) as i32;
            if !picked.contains(&e) {
                picked.push(e);
            }
        }
        ids.extend_from_slice(&picked);
        let w: Vec<f32> = (0..TOP_K).map(|_| 0.02 + rng.unit()).collect();
        let sum: f32 = w.iter().sum();
        weights.extend(w.iter().map(|v| v / sum));
    }
    // F16-exact, so the F16 leg carries the reference's own values.
    let x = (0..rows * DIM)
        .map(|_| f16_value(f16_bits((rng.unit() - 0.5) * 0.2)))
        .collect();
    Case {
        rows,
        ids,
        weights,
        x,
    }
}

fn main() {
    let mut gpu = Gpu::init_with_device(0).expect("gfx1151");
    assert_eq!(
        gpu.arch, "gfx1151",
        "run with HIP_VISIBLE_DEVICES=<strix> HIPFIRE_DEVICES=gfx1151"
    );
    let symmetric = std::env::var("ORACLE_SYMMETRIC").map_or(true, |v| v != "0");
    let capability = if symmetric {
        MoeRouteCapability::Qt44Qt53GroupedSymmetric
    } else {
        MoeRouteCapability::Qt44Qt53Grouped
    };
    println!("route {capability:?}");

    // Every expert's bytes, distinct per expert, pinned (Mapped only, as
    // hipfire's host tier). The cold set A is every third id; ALL is every
    // expert (the masking-equivalence check).
    let gu_len = weight_bytes(CpuQuant::Mq4G256V2, 2 * MI, DIM, 0, symmetric).len();
    let dn_len = weight_bytes(CpuQuant::Mq4G128V2, DIM, MI, 0, symmetric).len();
    let per = gu_len + dn_len;
    let mem = pinned(&gpu, N_EXP * per, false);
    for e in 0..N_EXP {
        let gu = weight_bytes(CpuQuant::Mq4G256V2, 2 * MI, DIM, e, symmetric);
        let dn = weight_bytes(CpuQuant::Mq4G128V2, DIM, MI, e, symmetric);
        // SAFETY: inside `mem`.
        unsafe {
            let g = mem.add(e * per);
            std::ptr::copy_nonoverlapping(gu.as_ptr(), g, gu_len);
            std::ptr::copy_nonoverlapping(dn.as_ptr(), g.add(gu_len), dn_len);
        }
    }
    let cold_set = |pick: &dyn Fn(usize) -> bool| {
        let mut c = ColdLayer {
            gate_up: vec![0; N_EXP],
            down: vec![0; N_EXP],
            cold_bitmap: vec![0; N_EXP],
        };
        for e in (0..N_EXP).filter(|&e| pick(e)) {
            // SAFETY: inside `mem`.
            let g = unsafe { mem.add(e * per) };
            c.gate_up[e] = g as u64;
            c.down[e] = g as u64 + gu_len as u64;
            c.cold_bitmap[e] = 1;
        }
        c
    };
    let cold = cold_set(&|e| e % 3 == 1);
    let cold_all = cold_set(&|_| true);
    let cfg = ColdPrefillConfig {
        dim: DIM,
        mi: MI,
        top_k: TOP_K,
        n_experts: N_EXP,
        max_rows: MAX_ROWS,
        gate_up_dtype: DType::MQ4G256V2,
        down_dtype: DType::MQ4G128V2,
        gate_up_bytes: gu_len,
        down_bytes: dn_len,
        route_policy: Some(policy(capability)),
        experts_all_gate_up_mq4: true,
        recipe: MoeRecipe::SoftmaxGatedShared {
            bf16_round_trip: true,
            shared_after_combine: true,
        },
    };
    let layer = ColdPrefillLayer::new(&mut gpu, cfg, &cold)
        .unwrap_or_else(|e| fail(format!("layer: {e:?}")));
    let layer_all = ColdPrefillLayer::new(&mut gpu, cfg, &cold_all)
        .unwrap_or_else(|e| fail(format!("layer ALL: {e:?}")));
    let scratch =
        ColdPrefillScratch::new(&mut gpu, cfg).unwrap_or_else(|e| fail(format!("scratch: {e:?}")));

    // The prefill mailbox fields (coherent, as the real mailbox).
    // Rows travel as F16 and the partial as BF16 (hetero_mailbox::PrefillGeometry).
    let x_mb = pinned(&gpu, MAX_ROWS * DIM * 2, true).cast::<u16>();
    let ids_mb = pinned(&gpu, MAX_ROWS * TOP_K * 4, true).cast::<i32>();
    let w_mb = pinned(&gpu, MAX_ROWS * TOP_K * 4, true).cast::<f32>();
    let part_mb = pinned(&gpu, MAX_ROWS * DIM * 2, true).cast::<u16>();
    let io = ColdPrefillIo {
        x_norm: x_mb as u64,
        format: ROWS_F16,
        ids: ids_mb as u64,
        weights: w_mb as u64,
        partial: part_mb as u64,
    };
    // SAFETY: the pinned BF16 partial of MAX_ROWS rows.
    let partial = || -> Vec<f32> {
        unsafe { std::slice::from_raw_parts(part_mb, MAX_ROWS * DIM) }
            .iter()
            .map(|&b| f32::from_bits((b as u32) << 16))
            .collect()
    };
    let run_on = |gpu: &mut Gpu, layer: &ColdPrefillLayer, c: &Case| {
        // SAFETY: mailbox buffers of MAX_ROWS rows; rows <= MAX_ROWS.
        unsafe {
            let x16: Vec<u16> = c.x.iter().map(|&v| f16_bits(v)).collect();
            std::ptr::copy_nonoverlapping(x16.as_ptr(), x_mb, x16.len());
            std::ptr::copy_nonoverlapping(c.ids.as_ptr(), ids_mb, c.ids.len());
            std::ptr::copy_nonoverlapping(c.weights.as_ptr(), w_mb, c.weights.len());
            std::slice::from_raw_parts_mut(part_mb, MAX_ROWS * DIM)
                .fill((SENTINEL.to_bits() >> 16) as u16);
        }
        run_cold_prefill(gpu, layer, &scratch, c.rows, io)
            .unwrap_or_else(|e| fail(format!("run {} rows: {e:?}", c.rows)));
        gpu.hip.device_synchronize().unwrap();
    };

    // Reference rows: rotate on this GPU, then Plan 3's arm per cold slot.
    let arm = IgpuColdArm::new(
        0,
        ColdArmDims {
            dim: DIM,
            mi: MI,
            top_k: TOP_K,
            n_experts: N_EXP,
            max_rows: MAX_ROWS,
        },
    )
    .unwrap_or_else(|e| fail(format!("arm: {e:?}")));
    let reference = |gpu: &mut Gpu, c: &Case| -> Vec<f64> {
        let xs = gpu.upload_f32(&c.x, &[c.rows, DIM]).unwrap();
        let xr = gpu.alloc_tensor(&[c.rows, DIM], DType::F32).unwrap();
        gpu.rotate_x_mq_batched(&xs, &xr, DIM, c.rows).unwrap();
        let x_rot = gpu.download_f32(&xr).unwrap();
        let host_slots: Vec<u32> = (0..c.rows * TOP_K)
            .filter(|&s| cold.cold_bitmap[c.ids[s] as usize] != 0)
            .map(|s| s as u32)
            .collect();
        let experts: Vec<CpuMoeExpert<'_>> = host_slots
            .iter()
            .map(|&s| {
                let e = c.ids[s as usize] as usize;
                // SAFETY: spans inside `mem`.
                unsafe {
                    CpuMoeExpert {
                        gate_up_quant: CpuQuant::Mq4G256V2,
                        gate_up: std::slice::from_raw_parts(cold.gate_up[e] as *const u8, gu_len),
                        down_quant: CpuQuant::Mq4G128V2,
                        down: std::slice::from_raw_parts(cold.down[e] as *const u8, dn_len),
                    }
                }
            })
            .collect();
        let x_pin = pinned(gpu, c.rows * DIM * 4, false).cast::<f32>();
        let out_pin = pinned(gpu, c.rows * TOP_K * DIM * 4, false).cast::<f32>();
        // SAFETY: pinned buffers of the sizes used.
        let (x_host, rows_out) = unsafe {
            std::ptr::copy_nonoverlapping(x_rot.as_ptr(), x_pin, c.rows * DIM);
            (
                std::slice::from_raw_parts(x_pin, c.rows * DIM),
                std::slice::from_raw_parts_mut(out_pin, c.rows * TOP_K * DIM),
            )
        };
        arm.host_slot_outputs(
            c.rows,
            &c.ids,
            &host_slots,
            &experts,
            x_host,
            rows_out,
            std::time::Duration::from_secs(60),
        )
        .unwrap_or_else(|e| fail(format!("arm call: {e:?}")));
        let mut sum = vec![0f64; c.rows * DIM];
        for &s in &host_slots {
            let (r, s) = (s as usize / TOP_K, s as usize);
            for i in 0..DIM {
                sum[r * DIM + i] += c.weights[s] as f64 * rows_out[s * DIM + i] as f64;
            }
        }
        sum
    };

    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut worst = 0f64;
    for rows in [1usize, 64, 2000, 2048] {
        let c = case(&mut rng, rows);
        run_on(&mut gpu, &layer, &c);
        let got = partial();
        let want = reference(&mut gpu, &c);
        let (mut err2, mut ref2, mut max_abs) = (0f64, 0f64, 0f64);
        for i in 0..rows * DIM {
            let d = got[i] as f64 - want[i];
            if !d.is_finite() {
                fail(format!(
                    "rows {rows}: non-finite at {i}: got {} want {}",
                    got[i], want[i]
                ));
            }
            err2 += d * d;
            ref2 += want[i] * want[i];
            max_abs = max_abs.max(d.abs());
        }
        let rel = (err2 / ref2.max(1e-30)).sqrt();
        worst = worst.max(rel);
        if let Some(i) = (rows * DIM..MAX_ROWS * DIM).find(|&i| got[i] != SENTINEL) {
            fail(format!("rows {rows}: wrote past the chunk at element {i}"));
        }
        println!(
            "rows {rows:>4}: rel L2 {rel:.3e}, max |d| {max_abs:.3e}, ref rms {:.3e}",
            (ref2 / (rows * DIM) as f64).sqrt()
        );
        let bound = if symmetric {
            REL_L2_MAX_SYMMETRIC
        } else {
            REL_L2_MAX
        };
        if rel > bound {
            fail(format!("rows {rows}: rel L2 {rel:.3e} > {bound:.0e}"));
        }
        // Determinism: a second run of the same chunk is bitwise equal.
        run_on(&mut gpu, &layer, &c);
        if partial()[..rows * DIM]
            .iter()
            .zip(&got)
            .any(|(a, b)| a.to_bits() != b.to_bits())
        {
            fail(format!("rows {rows}: second run differs"));
        }
    }

    // Masking equivalence (bitwise): layer A with the device mask must equal
    // the all-cold layer fed the same route with A's hot slots already -1:
    // the same slots, in the same order, over the same expert bytes.
    for rows in [64usize, 2048] {
        let c = case(&mut rng, rows);
        run_on(&mut gpu, &layer, &c);
        let masked = partial();
        let mut pre = Case {
            rows,
            ids: c.ids.clone(),
            weights: c.weights.clone(),
            x: c.x.clone(),
        };
        for id in pre.ids.iter_mut() {
            if cold.cold_bitmap[*id as usize] == 0 {
                *id = -1;
            }
        }
        run_on(&mut gpu, &layer_all, &pre);
        let premasked = partial();
        let n = rows * DIM;
        if let Some(i) = (0..n).find(|&i| masked[i].to_bits() != premasked[i].to_bits()) {
            fail(format!(
                "masking rows {rows}: element {i} differs: {} vs {}",
                masked[i], premasked[i]
            ));
        }
        println!("masking rows {rows:>4}: device mask == pre-masked route, bitwise");
    }

    // No cold slot in the chunk: the partial is exactly zero.
    let mut c = case(&mut rng, 64);
    for s in 0..c.ids.len() {
        while cold.cold_bitmap[c.ids[s] as usize] != 0 {
            c.ids[s] = (c.ids[s] + 1) % N_EXP as i32;
        }
    }
    run_on(&mut gpu, &layer, &c);
    if let Some(i) = partial()[..64 * DIM].iter().position(|v| *v != 0.0) {
        fail(format!("zero-cold chunk: element {i} is nonzero"));
    }
    println!("zero-cold chunk: partial all zero");

    // The shipped int4 activation (symmetric route): the rows are F16-exact, so
    // this device's own producer over them is the sidecar the primary would
    // ship, and the int4 call must equal the F16 call bitwise. A 256-row block
    // (below the IU4 arm) and the F16 route must refuse it.
    let x4_mb = pinned(&gpu, MAX_ROWS * int4_row_bytes(DIM), true);
    let io4 = ColdPrefillIo {
        x_norm: x4_mb as u64,
        format: ROWS_INT4,
        ..io
    };
    for rows in [512usize, 2048, 256] {
        let c = case(&mut rng, rows);
        run_on(&mut gpu, &layer, &c);
        let want = partial();
        let xs = gpu.upload_f32(&c.x, &[rows, DIM]).unwrap();
        let prepared = gpu.qwen4_moe_rotate256_i4(&xs, DIM, rows).unwrap();
        let ptr = gpu.int4_mmq_prepared_ptr(&prepared, DIM, rows).unwrap();
        let bytes = rows * int4_row_bytes(DIM);
        // SAFETY: the live sidecar and the pinned int4 area.
        let (src, dst) = unsafe {
            (
                hip_bridge::DeviceBuffer::from_raw(ptr, bytes),
                hip_bridge::DeviceBuffer::from_raw(x4_mb.cast(), bytes),
            )
        };
        gpu.hip.memcpy_dtod(&dst, &src, bytes).unwrap();
        // SAFETY: poison the partial; ids/weights are still this case's.
        unsafe {
            std::slice::from_raw_parts_mut(part_mb, MAX_ROWS * DIM)
                .fill((SENTINEL.to_bits() >> 16) as u16)
        };
        let got = run_cold_prefill(&mut gpu, &layer, &scratch, rows, io4);
        gpu.hip.device_synchronize().unwrap();
        let expect_ok =
            symmetric && rows >= hipfire_dispatch::hetero_prefill::COLD_PREFILL_SYM_IU4_FLOOR;
        match (got, expect_ok) {
            (Ok(()), true) => {
                let g = partial();
                if let Some(i) = (0..rows * DIM).find(|&i| g[i].to_bits() != want[i].to_bits()) {
                    fail(format!(
                        "int4 rows {rows}: element {i}: {} != F16 path {}",
                        g[i], want[i]
                    ));
                }
                println!("int4 rows {rows}: shipped sidecar == own F16 basis, bitwise");
            }
            (Err(e), false) => println!(
                "int4 rows {rows}: refused as expected ({})",
                format!("{e:?}").chars().take(90).collect::<String>()
            ),
            (Ok(()), false) => fail(format!(
                "int4 rows {rows}: accepted where the IU4 arm does not run"
            )),
            (Err(e), true) => fail(format!("int4 rows {rows}: {e:?}")),
        }
    }

    // Timing: 2,048 rows, synchronous per call (the responder's shape).
    let c = case(&mut rng, MAX_ROWS);
    run_on(&mut gpu, &layer, &c);
    let n = 20;
    let t = Instant::now();
    for _ in 0..n {
        run_cold_prefill(&mut gpu, &layer, &scratch, MAX_ROWS, io).unwrap();
        gpu.hip.device_synchronize().unwrap();
    }
    let per = t.elapsed().as_secs_f64() / n as f64;
    let cold_slots = c
        .ids
        .iter()
        .filter(|&&e| cold.cold_bitmap[e as usize] != 0)
        .count();
    println!(
        "timing: {MAX_ROWS} rows ({cold_slots} cold slots) {:.2} ms/layer, {:.0} rows/s per layer",
        per * 1e3,
        MAX_ROWS as f64 / per
    );
    println!("PASS (worst rel L2 {worst:.3e})");
}
