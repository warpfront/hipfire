//! Oracle for the Strix responder of the device-side hetero split.
//!
//! The R9700 runs `hetero_token_advance`, `hetero_publish_cold`,
//! `hetero_await_cold` and `hetero_install_cold`. The gfx1151 responder answers
//! from its recorded layer graph. The installed cold rows must be bitwise equal
//! to Plan 3's `IgpuColdArm` (the same kernels) on the same synthetic, finite
//! qt44/qt53 experts in pinned host memory. Hot rows must stay untouched. Then
//! 2,000 round trips are timed.
//!
//! ```text
//! HIPFIRE_DEVICES=gfx1201,gfx1151 cargo run --release -p hipfire-dispatch --example hetero_responder_oracle
//! ```

use std::time::Instant;

use hipfire_cpu::quant::CpuQuant;
use hipfire_dispatch::cpu_exec::CpuMoeExpert;
use hipfire_dispatch::hetero_mailbox::MailboxGeometry;
use hipfire_dispatch::hetero_responder::{ColdLayer, HeteroResponder, ResponderConfig};
use hipfire_dispatch::igpu_split::{ColdArmDims, IgpuColdArm};
use rdna_compute::{DType, Gpu, GpuTensor, HeteroSlotPtrs};

const N_EXP: usize = 512;
const TOP_K: usize = 10;
const DIM: usize = 2560;
const MI: usize = 640;
const LAYERS: usize = 2;
const SENTINEL: f32 = -7.0;
const F16_QUAD: [u16; 4] = [0x2c00, 0xb400, 0x3800, 0x3a00];

/// Finite synthetic weights, built as `hipfire-cpu`'s `testfix::weight_bytes`.
fn weight_bytes(q: CpuQuant, m: usize, k: usize) -> Vec<u8> {
    let groups = k / q.group_elems();
    let hdr = q.row_header_bytes();
    let mut out = Vec::new();
    for row in 0..m {
        for g in 0..groups {
            let salt = row * groups + g;
            let mut b: Vec<u8> = (0..hdr + q.group_bytes()).map(|i| (((i + salt) * 37 + 11) & 0xFF) as u8).collect();
            let quad: Vec<u8> = F16_QUAD.iter().flat_map(|v| v.to_le_bytes()).collect();
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
    let p = gpu.hip.host_malloc(bytes, flags).expect("host_malloc").cast::<u8>();
    // SAFETY: fresh allocation of `bytes`.
    unsafe { std::ptr::write_bytes(p, 0, bytes) };
    p
}

fn upload_bytes(gpu: &mut Gpu, b: &[u8]) -> GpuTensor {
    let t = gpu.alloc_tensor(&[b.len().div_ceil(4)], DType::F32).unwrap();
    gpu.memcpy_htod_auto(&t.buf, b).unwrap();
    t
}

fn fail(msg: String) -> ! {
    println!("FAIL: {msg}");
    std::process::exit(1);
}

fn main() {
    let mut r = Gpu::init_with_device(0).expect("gfx1201");
    assert_eq!(r.arch, "gfx1201", "run with HIPFIRE_DEVICES=gfx1201,gfx1151");

    // Experts: 10 routed ids; slots 1,2,4,5,7,8 cold. Bytes pinned (Mapped only,
    // like hipfire's host tier).
    let ids: Vec<i32> = (0..TOP_K as i32).map(|s| 3 + 37 * s).collect();
    let cold: Vec<bool> = (0..TOP_K).map(|s| s % 3 != 0).collect();
    let gu = weight_bytes(CpuQuant::Mq4G256V2, 2 * MI, DIM);
    let dn = weight_bytes(CpuQuant::Mq4G128V2, DIM, MI);
    let per = gu.len() + dn.len();
    let experts_mem = pinned(&r, TOP_K * per, false);
    let mut gate_up = vec![0u64; N_EXP];
    let mut down = vec![0u64; N_EXP];
    let mut bitmap = vec![0u8; N_EXP];
    for s in 0..TOP_K {
        // SAFETY: inside `experts_mem`.
        let (g, d) = unsafe {
            let g = experts_mem.add(s * per);
            std::ptr::copy_nonoverlapping(gu.as_ptr(), g, gu.len());
            let d = g.add(gu.len());
            std::ptr::copy_nonoverlapping(dn.as_ptr(), d, dn.len());
            (g, d)
        };
        if cold[s] {
            let e = ids[s] as usize;
            gate_up[e] = g as u64;
            down[e] = d as u64;
            bitmap[e] = 1;
        }
    }
    let x: Vec<f32> = (0..DIM).map(|i| ((i % 97) as f32 - 48.0) * 0.01).collect();
    let w: Vec<f32> = (0..TOP_K).map(|s| 0.1 * (s as f32 + 1.0)).collect();

    // Reference: Plan 3's arm on the same pinned inputs.
    let x_pin = pinned(&r, DIM * 4, false).cast::<f32>();
    let ref_pin = pinned(&r, TOP_K * DIM * 4, false).cast::<f32>();
    // SAFETY: pinned buffers of the sizes used.
    let (x_host, ref_rows) = unsafe {
        std::ptr::copy_nonoverlapping(x.as_ptr(), x_pin, DIM);
        (std::slice::from_raw_parts(x_pin, DIM), std::slice::from_raw_parts_mut(ref_pin, TOP_K * DIM))
    };
    let host_slots: Vec<u32> = (0..TOP_K as u32).filter(|&s| cold[s as usize]).collect();
    let experts: Vec<CpuMoeExpert<'_>> = host_slots
        .iter()
        .map(|&s| {
            let e = ids[s as usize] as usize;
            // SAFETY: spans inside `experts_mem`.
            unsafe {
                CpuMoeExpert {
                    gate_up_quant: CpuQuant::Mq4G256V2,
                    gate_up: std::slice::from_raw_parts(gate_up[e] as *const u8, gu.len()),
                    down_quant: CpuQuant::Mq4G128V2,
                    down: std::slice::from_raw_parts(down[e] as *const u8, dn.len()),
                }
            }
        })
        .collect();
    let arm = IgpuColdArm::new(1, ColdArmDims { dim: DIM, mi: MI, top_k: TOP_K, n_experts: N_EXP, max_rows: 1 })
        .unwrap_or_else(|e| fail(format!("arm: {e:?}")));
    arm.host_slot_outputs(1, &ids, &host_slots, &experts, x_host, ref_rows, std::time::Duration::from_secs(5))
        .unwrap_or_else(|e| fail(format!("arm call: {e:?}")));
    drop(arm);

    // Responder over a 2-layer mailbox.
    let geom = MailboxGeometry { layers: LAYERS, max_rows: 1, top_k: TOP_K, dim: DIM };
    let mb = pinned(&r, geom.total_bytes(), true);
    let layer = ColdLayer { gate_up: gate_up.clone(), down: down.clone(), cold_bitmap: bitmap.clone() };
    let cfg = ResponderConfig { ordinal: 1, geom, mailbox_base: mb as u64, n_experts: N_EXP, mi: MI };
    let responder = HeteroResponder::start(cfg, vec![layer; LAYERS]).unwrap_or_else(|e| fail(format!("start: {e:?}")));

    let slot = |l: usize| {
        let o = geom.slot(l);
        let b = mb as u64;
        HeteroSlotPtrs {
            req_seq: b + o.req_seq as u64,
            resp_seq: b + o.resp_seq as u64,
            error: b + o.error as u64,
            x_rot: b + o.x_rot as u64,
            ids: b + o.ids as u64,
            weights: b + o.weights as u64,
            cold_out: b + o.cold_out as u64,
        }
    };
    let counter = upload_bytes(&mut r, &0u32.to_ne_bytes());
    let x_t = r.upload_f32(&x, &[DIM]).unwrap();
    let ids_t = upload_bytes(&mut r, &ids.iter().flat_map(|v| v.to_ne_bytes()).collect::<Vec<u8>>());
    let w_t = r.upload_f32(&w, &[TOP_K]).unwrap();
    let bm_t = upload_bytes(&mut r, &bitmap);
    let masked_t = r.alloc_tensor(&[TOP_K], DType::F32).unwrap();
    let expanded = r.upload_f32(&vec![SENTINEL; TOP_K * DIM], &[TOP_K * DIM]).unwrap();
    let s0 = slot(0);
    let mut round = |r: &mut Gpu, install: bool| {
        r.hetero_token_advance(&counter).unwrap();
        r.hetero_publish_cold(&counter, 0, LAYERS as u32, &x_t, &ids_t, &w_t, &bm_t, N_EXP as i32, 1, TOP_K as i32, DIM as i32, &s0, &masked_t)
            .unwrap();
        r.hetero_await_cold(&counter, 0, LAYERS as u32, &s0, 50_000_000).unwrap();
        if install {
            r.hetero_install_cold(&s0, &ids_t, &bm_t, N_EXP as i32, TOP_K as u32, DIM as i32, &expanded).unwrap();
        }
    };
    round(&mut r, true);
    r.hip.device_synchronize().unwrap();
    // SAFETY: the slot's error word.
    if unsafe { *(s0.error as *const u32) } != 0 {
        fail(format!("await timed out (responder: {:?})", responder.failure()));
    }
    let got = r.download_f32(&expanded).unwrap();
    for s in 0..TOP_K {
        for i in 0..DIM {
            let want = if cold[s] { ref_rows[s * DIM + i] } else { SENTINEL };
            if got[s * DIM + i].to_bits() != want.to_bits() {
                fail(format!("slot {s} elem {i}: {} != {want}", got[s * DIM + i]));
            }
        }
    }
    let n = 2000;
    let t = Instant::now();
    for _ in 0..n {
        round(&mut r, false);
    }
    r.hip.device_synchronize().unwrap();
    let us = t.elapsed().as_secs_f64() * 1e6 / n as f64;
    // SAFETY: the slot's error word.
    if unsafe { *(s0.error as *const u32) } != 0 {
        fail(format!("timed loop hit a timeout (responder: {:?})", responder.failure()));
    }
    println!(
        "PASS: cold rows bitwise == IgpuColdArm ({} cold slots); round trip incl. cold compute {us:.1} us over {n} rounds; answered {}",
        host_slots.len(),
        responder.answered()
    );
}
