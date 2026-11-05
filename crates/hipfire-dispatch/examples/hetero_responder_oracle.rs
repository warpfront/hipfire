//! Oracle for the Strix responder of the device-side hetero split.
//!
//! The R9700 runs `hetero_token_advance`, `hetero_publish_cold`,
//! `hetero_await_cold` and `hetero_install_cold`. The gfx1151 responder answers
//! from its recorded layer graph. The installed cold rows must be bitwise equal
//! to Plan 3's `IgpuColdArm` (the same kernels) on the same synthetic, finite
//! qt44/qt53 experts in pinned host memory. Hot rows must stay untouched. Then
//! 2,000 round trips are timed. Prefill (Plan 5): the R9700 publishes a chunk
//! (`hetero_publish_rows` + `hetero_release_rows`), awaits and adds the partial
//! (`hetero_add_partial`) into a zeroed `x_batch`, which must then equal,
//! bitwise, `run_cold_prefill` run directly on a second gfx1151 context; 2,048-
//! and 37-row chunks, then 2,048-row layers are timed.
//!
//! ```text
//! HIPFIRE_DEVICES=gfx1201,gfx1151 cargo run --release -p hipfire-dispatch --features deltanet --example hetero_responder_oracle
//! ```

use std::time::Instant;

use hipfire_cpu::quant::CpuQuant;
use hipfire_dispatch::cpu_exec::CpuMoeExpert;
use hipfire_dispatch::families::moe::{
    MoeRecipe, MoeRouteCapability, MoeRouteFormats, MoeRouteGeometry, MoeRoutePolicy,
};
use hipfire_dispatch::hetero_mailbox::{
    int4_row_bytes, MailboxGeometry, PrefillGeometry, MAX_SUB_CHUNKS, ROWS_F16, ROWS_INT4,
};
use hipfire_dispatch::hetero_prefill::{
    run_cold_prefill, ColdPrefillConfig, ColdPrefillIo, ColdPrefillLayer, ColdPrefillScratch,
};
use hipfire_dispatch::hetero_responder::{ColdLayer, HeteroResponder, PrefillServe, ResponderConfig};
use hipfire_dispatch::igpu_split::{ColdArmDims, IgpuColdArm};
use rdna_compute::{DType, Gpu, GpuTensor, HeteroSlotPtrs};

const N_EXP: usize = 512;
const TOP_K: usize = 10;
const DIM: usize = 2560;
const MI: usize = 640;
const LAYERS: usize = 2;
const SENTINEL: f32 = -7.0;
const PREFILL_ROWS: usize = 2048;
/// Symmetric headers (`zp = -8 sc`: sc 1/16, zp -1/2), as Flash-Next's
/// verified experts, so the prefill takes the symmetric IU4 route.
const F16_QUAD: [u16; 4] = [0x2c00, 0xb800, 0x2c00, 0xb800];

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

/// f32 -> IEEE half bits, round-to-nearest-even, normal range (the device cast).
fn f32_to_f16_bits(v: f32) -> u16 {
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

fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let e = ((h >> 10) & 0x1f) as i32;
    if e == 0 {
        return sign * (h & 0x3ff) as f32 * 2f32.powi(-24);
    }
    sign * (1.0 + (h & 0x3ff) as f32 / 1024.0) * 2f32.powi(e - 15)
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
    // The slot carries MAX_SUB_CHUNKS sub-chunks; a chunk may use fewer.
    let pgeom = PrefillGeometry { max_rows: PREFILL_ROWS, top_k: TOP_K, dim: DIM, sub_chunks: MAX_SUB_CHUNKS };
    let po = pgeom.offsets(geom.total_bytes());
    let mb = pinned(&r, po.error + pgeom.bytes(), true);
    let layer = ColdLayer { gate_up: gate_up.clone(), down: down.clone(), cold_bitmap: bitmap.clone() };
    let cold_cfg = ColdPrefillConfig {
        dim: DIM,
        mi: MI,
        top_k: TOP_K,
        n_experts: N_EXP,
        max_rows: PREFILL_ROWS,
        gate_up_dtype: DType::MQ4G256V2,
        down_dtype: DType::MQ4G128V2,
        gate_up_bytes: gu.len(),
        down_bytes: dn.len(),
        route_policy: Some(MoeRoutePolicy {
            capability: MoeRouteCapability::Qt44Qt53GroupedSymmetric,
            geometry: MoeRouteGeometry { experts: N_EXP, top_k: TOP_K, hidden: DIM, routed_intermediate: MI, shared_intermediate: MI },
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
        }),
        experts_all_gate_up_mq4: true,
        recipe: MoeRecipe::SoftmaxGatedShared { bf16_round_trip: true, shared_after_combine: true },
    };
    let cfg = ResponderConfig {
        ordinal: 1,
        geom,
        mailbox_base: mb as u64,
        n_experts: N_EXP,
        mi: MI,
        prefill: vec![PrefillServe { offsets: po, sub_chunks: MAX_SUB_CHUNKS, cold: cold_cfg }],
    };
    let responder = HeteroResponder::start(cfg, vec![layer.clone(); LAYERS]).unwrap_or_else(|e| fail(format!("start: {e:?}")));

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
    let round = |r: &mut Gpu, install: bool| {
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
    let decode_answered = responder.answered();

    // Prefill round trips. The reference runs the same call directly on a
    // second gfx1151 context over a private copy of the inputs.
    let mut strix = Gpu::init_with_device(1).expect("gfx1151");
    let ref_layer = ColdPrefillLayer::new(&mut strix, cold_cfg, &layer).unwrap_or_else(|e| fail(format!("ref layer: {e:?}")));
    let ref_scratch = ColdPrefillScratch::new(&mut strix, cold_cfg).unwrap_or_else(|e| fail(format!("ref scratch: {e:?}")));
    let ref_x = pinned(&r, PREFILL_ROWS * DIM * 2, true).cast::<u16>();
    let ref_ids = pinned(&r, PREFILL_ROWS * TOP_K * 4, true).cast::<i32>();
    let ref_w = pinned(&r, PREFILL_ROWS * TOP_K * 4, true).cast::<f32>();
    let ref_part = pinned(&r, PREFILL_ROWS * DIM * 2, true).cast::<u16>();
    let b = mb as u64;
    let chunk = |rows: usize| {
        let ids: Vec<i32> = (0..rows * TOP_K).map(|s| ids[(s + s / TOP_K) % TOP_K]).collect();
        let w: Vec<f32> = (0..rows * TOP_K).map(|s| 0.05 + 0.01 * ((s * 7) % 11) as f32).collect();
        // F16-exact, so the int4 sidecar (quantized from these F32 rows) is
        // the one the reference's F16 basis yields.
        let x: Vec<f32> = (0..rows * DIM).map(|i| f16_bits_to_f32(f32_to_f16_bits((((i * 31) % 193) as f32 - 96.0) * 0.002))).collect();
        (ids, w, x)
    };
    let p_bm = upload_bytes(&mut r, &bitmap);
    let p_masked = r.alloc_tensor(&[PREFILL_ROWS * TOP_K], DType::F32).unwrap();
    let x_f16 = r.alloc_tensor(&[PREFILL_ROWS * DIM / 2], DType::F32).unwrap();
    let part_local = r.alloc_tensor(&[PREFILL_ROWS * DIM / 2], DType::F32).unwrap();
    // SAFETY: views of the pinned prefill slot.
    let (mb_x, mb_part) = unsafe {
        (
            hip_bridge::DeviceBuffer::from_raw((b + po.x_norm as u64) as *mut std::ffi::c_void, PREFILL_ROWS * DIM * 2),
            hip_bridge::DeviceBuffer::from_raw((b + po.partial as u64) as *mut std::ffi::c_void, PREFILL_ROWS * DIM * 2),
        )
    };
    // Three R9700 streams: main (publish, add), out (copy engine + release
    // per sub-chunk), back (bounded await + copy engine per sub-chunk).
    let mut streams = [
        Some(r.hip.stream_create_non_blocking().unwrap()),
        Some(r.hip.stream_create_non_blocking().unwrap()),
        Some(r.hip.stream_create_non_blocking().unwrap()),
    ];
    let (published, returned) = (r.hip.event_create().unwrap(), r.hip.event_create().unwrap());
    let main_stream = r.active_stream.take();
    // Launch `f` with stream `k` active.
    fn on<T>(r: &mut Gpu, streams: &mut [Option<hip_bridge::Stream>; 3], k: usize, f: impl FnOnce(&mut Gpu) -> T) -> T {
        r.active_stream = streams[k].take();
        let out = f(r);
        streams[k] = r.active_stream.take();
        out
    }
    // The expert device's int4 floor (0 = F16 only), published at its start.
    // SAFETY: the slot's word, written once before `start` returned.
    let int4_min_rows = unsafe { *((b + po.int4_min_rows as u64) as *const u32) } as usize;
    println!("expert device int4 floor: {int4_min_rows} rows");
    let i4_row = int4_row_bytes(DIM);
    let marks: std::cell::RefCell<Option<Vec<(String, hip_bridge::Event)>>> = std::cell::RefCell::new(None);
    let mark = |r: &mut Gpu, label: String| {
        if let Some(v) = marks.borrow_mut().as_mut() {
            let e = r.hip.event_create().unwrap();
            r.hip.event_record(&e, r.active_stream.as_ref()).unwrap();
            v.push((label, e));
        }
    };
    let prefill_round = |r: &mut Gpu, streams: &mut [Option<hip_bridge::Stream>; 3], l: u32, rows: usize, subs: usize, int4: bool, x: &GpuTensor, ids: &GpuTensor, w: &GpuTensor, xb: &GpuTensor| {
        let split = PrefillGeometry { sub_chunks: subs, ..pgeom };
        let fmt = |n: usize| if int4 && int4_min_rows > 0 && n >= int4_min_rows { ROWS_INT4 } else { ROWS_F16 };
        let sidecar = on(r, streams, 0, |r| {
            mark(r, "start".into());
            r.hetero_token_advance(&counter).unwrap();
            r.hetero_publish_rows(x, ids, w, &p_bm, N_EXP as i32, rows as i32, TOP_K as i32, DIM as i32, &x_f16, b + po.ids as u64, b + po.weights as u64, &p_masked)
                .unwrap();
            // The chunk's int4 sidecar, once (the primary's own gate/up input
            // in the real program); gathered per block below.
            let sidecar = (0..subs).any(|i| fmt(split.sub_range(rows, i).1) == ROWS_INT4).then(|| {
                let prepared = r.qwen4_moe_rotate256_i4(x, DIM, rows).unwrap();
                r.int4_mmq_prepared_ptr(&prepared, DIM, rows).unwrap()
            });
            r.hip.event_record(&published, r.active_stream.as_ref()).unwrap();
            mark(r, "published".into());
            sidecar
        });
        on(r, streams, 1, |r| {
            r.hip.stream_wait_event_on(r.active_stream.as_ref(), &published).unwrap();
            for i in 0..subs {
                let (off, n) = split.sub_range(rows, i);
                if n == 0 {
                    continue;
                }
                let f = fmt(n);
                if f == ROWS_INT4 {
                    // Block-major records: one run per K128 block, gathered
                    // into the block's own [K/128][n] sidecar.
                    let ptr = sidecar.expect("sidecar produced");
                    // SAFETY: the live chunk sidecar.
                    let src = unsafe { hip_bridge::DeviceBuffer::from_raw(ptr, rows * i4_row) };
                    r.hip
                        .memcpy_dtod_2d_async_at(&mb_x, off * i4_row, n * 72, &src, off * 72, rows * 72, n * 72, DIM / 128, r.active_stream.as_ref())
                        .unwrap();
                } else {
                    let st = r.active_stream.as_ref().unwrap();
                    r.hip.memcpy_dtod_async_at(&mb_x, off * DIM * 2, &x_f16.buf, off * DIM * 2, n * DIM * 2, st).unwrap();
                }
                let wd = po.sub[i];
                r.hetero_release_rows(&counter, l, LAYERS as u32, off as i32, n as i32, f, b + wd.layer as u64, b + wd.req_seq as u64).unwrap();
                mark(r, format!("out {i} released"));
            }
        });
        on(r, streams, 2, |r| {
            // The counter is read by the awaits: order them after the advance.
            r.hip.stream_wait_event_on(r.active_stream.as_ref(), &published).unwrap();
            for i in 0..subs {
                let (off, n) = split.sub_range(rows, i);
                if n == 0 {
                    continue;
                }
                let wd = po.sub[i];
                let slot = HeteroSlotPtrs { req_seq: b + wd.req_seq as u64, resp_seq: b + wd.resp_seq as u64, error: b + po.error as u64, x_rot: 0, ids: 0, weights: 0, cold_out: 0 };
                r.hetero_await_cold(&counter, l, LAYERS as u32, &slot, 50_000_000).unwrap();
                mark(r, format!("back {i} answered"));
                let st = r.active_stream.as_ref().unwrap();
                r.hip.memcpy_dtod_async_at(&part_local.buf, off * DIM * 2, &mb_part, off * DIM * 2, n * DIM * 2, st).unwrap();
                mark(r, format!("back {i} copied"));
            }
            r.hip.event_record(&returned, r.active_stream.as_ref()).unwrap();
        });
        on(r, streams, 0, |r| {
            r.hip.stream_wait_event_on(r.active_stream.as_ref(), &returned).unwrap();
            r.hetero_add_partial(b + po.error as u64, &part_local, (rows * DIM) as i32, xb).unwrap();
        });
    };
    let sync_all = |r: &mut Gpu, streams: &[Option<hip_bridge::Stream>; 3]| {
        for st in streams.iter().flatten() {
            r.hip.stream_synchronize(st).unwrap();
        }
    };
    let error_word = || unsafe { *((b + po.error as u64) as *const u32) };
    for (l, rows, subs, int4) in [
        (0u32, PREFILL_ROWS, 4usize, true),
        (1, PREFILL_ROWS, 4, false),
        (0, 1525, 2, true),
        (1, 37, 4, true),
        (0, 2000, 1, true),
        (1, 5, 8, false),
    ] {
        let (ids_v, w_v, x_v) = chunk(rows);
        let x_t = r.upload_f32(&x_v, &[rows, DIM]).unwrap();
        let ids_t = upload_bytes(&mut r, &ids_v.iter().flat_map(|v| v.to_ne_bytes()).collect::<Vec<u8>>());
        let w_t = r.upload_f32(&w_v, &[rows * TOP_K]).unwrap();
        let xb = r.upload_f32(&vec![0.0; rows * DIM], &[rows * DIM]).unwrap();
        r.hip.device_synchronize().unwrap();
        prefill_round(&mut r, &mut streams, l, rows, subs, int4, &x_t, &ids_t, &w_t, &xb);
        sync_all(&mut r, &streams);
        if error_word() != 0 {
            fail(format!("prefill await timed out (responder: {:?})", responder.failure()));
        }
        let got = r.download_f32(&xb).unwrap();
        // The reference: the same F16 rows, one direct call over all rows.
        let x16: Vec<u16> = x_v.iter().map(|&v| f32_to_f16_bits(v)).collect();
        // SAFETY: private pinned reference buffers sized for PREFILL_ROWS.
        unsafe {
            std::ptr::copy_nonoverlapping(x16.as_ptr(), ref_x, x16.len());
            std::ptr::copy_nonoverlapping(ids_v.as_ptr(), ref_ids, ids_v.len());
            std::ptr::copy_nonoverlapping(w_v.as_ptr(), ref_w, w_v.len());
        }
        let io = ColdPrefillIo { x_norm: ref_x as u64, format: ROWS_F16, ids: ref_ids as u64, weights: ref_w as u64, partial: ref_part as u64 };
        run_cold_prefill(&mut strix, &ref_layer, &ref_scratch, rows, io).unwrap_or_else(|e| fail(format!("ref run: {e:?}")));
        strix.hip.device_synchronize().unwrap();
        // SAFETY: as above.
        let want: Vec<f32> = unsafe { std::slice::from_raw_parts(ref_part, rows * DIM) }
            .iter()
            .map(|&h| f32::from_bits((h as u32) << 16))
            .collect();
        if want.iter().all(|v| *v == 0.0) {
            fail(format!("prefill layer {l} rows {rows}: vacuous (all-zero partial)"));
        }
        // Sub-chunked blocks run the grouped program per block, so the
        // reference over all rows matches only where per-row results do not
        // depend on the block (they should not: rows are independent).
        if let Some(i) = (0..rows * DIM).find(|&i| got[i].to_bits() != want[i].to_bits()) {
            fail(format!("prefill layer {l} rows {rows} subs {subs}: element {i}: {} != {}", got[i], want[i]));
        }
        println!("prefill layer {l}, {rows} rows, {subs} sub-chunks, int4 {int4}: x_batch += partial bitwise == direct run_cold_prefill");
    }
    // Timing: 2,048-row layers, one per round, by sub-chunk count.
    let (ids_v, w_v, x_v) = chunk(PREFILL_ROWS);
    let x_t = r.upload_f32(&x_v, &[PREFILL_ROWS, DIM]).unwrap();
    let ids_t = upload_bytes(&mut r, &ids_v.iter().flat_map(|v| v.to_ne_bytes()).collect::<Vec<u8>>());
    let w_t = r.upload_f32(&w_v, &[PREFILL_ROWS * TOP_K]).unwrap();
    let xb = r.upload_f32(&vec![0.0; PREFILL_ROWS * DIM], &[PREFILL_ROWS * DIM]).unwrap();
    let rounds = 48;
    for (subs, int4) in [(1usize, false), (4, false), (1, true), (2, true), (4, true)] {
        r.hip.device_synchronize().unwrap();
        let t = Instant::now();
        for i in 0..rounds {
            prefill_round(&mut r, &mut streams, (i % LAYERS) as u32, PREFILL_ROWS, subs, int4, &x_t, &ids_t, &w_t, &xb);
        }
        sync_all(&mut r, &streams);
        let ms = t.elapsed().as_secs_f64() * 1e3 / rounds as f64;
        if error_word() != 0 {
            fail(format!("timed prefill hit a timeout (responder: {:?})", responder.failure()));
        }
        println!(
            "prefill: {PREFILL_ROWS}-row layer, {subs} sub-chunks, int4 {int4}: {ms:.2} ms -> {:.0} rows/s over 48 layers",
            PREFILL_ROWS as f64 / (ms * 48.0 / 1e3)
        );
    }
    // One instrumented round (int4, 4 blocks): event times from the start.
    for _ in 0..3 {
        prefill_round(&mut r, &mut streams, 0, PREFILL_ROWS, 4, true, &x_t, &ids_t, &w_t, &xb);
    }
    sync_all(&mut r, &streams);
    *marks.borrow_mut() = Some(Vec::new());
    prefill_round(&mut r, &mut streams, 1, PREFILL_ROWS, 4, true, &x_t, &ids_t, &w_t, &xb);
    sync_all(&mut r, &streams);
    let taken = marks.borrow_mut().take().unwrap();
    let start = &taken[0].1;
    for (label, e) in &taken[1..] {
        println!("  timeline {label:>18}: {:6.2} ms", r.hip.event_elapsed_ms(start, e).unwrap());
    }
    // The link legs alone, by copy engine (10.5 MB each way).
    let time_copy = |r: &mut Gpu, dst: &hip_bridge::DeviceBuffer, src: &hip_bridge::DeviceBuffer| {
        r.hip.device_synchronize().unwrap();
        let t = Instant::now();
        for _ in 0..rounds {
            r.hip.memcpy_dtod_async_default_at(dst, 0, src, 0, PREFILL_ROWS * DIM * 2).unwrap();
        }
        r.hip.device_synchronize().unwrap();
        t.elapsed().as_secs_f64() * 1e3 / rounds as f64
    };
    let out_ms = time_copy(&mut r, &mb_x, &x_f16.buf);
    let back_ms = time_copy(&mut r, &part_local.buf, &mb_part);
    let mbytes = (PREFILL_ROWS * DIM * 2) as f64 / 1e6;
    println!(
        "prefill link legs ({mbytes:.1} MB each, copy engine): out {out_ms:.2} ms ({:.2} GB/s), back {back_ms:.2} ms ({:.2} GB/s)",
        mbytes / out_ms,
        mbytes / back_ms
    );
    // Full duplex? Both legs at once on two streams.
    r.hip.device_synchronize().unwrap();
    let t = Instant::now();
    for _ in 0..rounds {
        let (s1, s2) = (streams[1].as_ref().unwrap(), streams[2].as_ref().unwrap());
        r.hip.memcpy_dtod_async_at(&mb_x, 0, &x_f16.buf, 0, PREFILL_ROWS * DIM * 2, s1).unwrap();
        r.hip.memcpy_dtod_async_at(&part_local.buf, 0, &mb_part, 0, PREFILL_ROWS * DIM * 2, s2).unwrap();
    }
    sync_all(&mut r, &streams);
    let both_ms = t.elapsed().as_secs_f64() * 1e3 / rounds as f64;
    println!("prefill link legs concurrently: {both_ms:.2} ms (serial sum {:.2} ms)", out_ms + back_ms);
    // Strix compute alone per block size (direct calls on the second context).
    for rows in [256usize, 512, 2048] {
        let io = ColdPrefillIo { x_norm: ref_x as u64, format: ROWS_F16, ids: ref_ids as u64, weights: ref_w as u64, partial: ref_part as u64 };
        // SAFETY: reference buffers sized for PREFILL_ROWS.
        unsafe {
            let x16: Vec<u16> = x_v.iter().map(|&v| f32_to_f16_bits(v)).collect();
            std::ptr::copy_nonoverlapping(x16.as_ptr(), ref_x, x16.len());
            std::ptr::copy_nonoverlapping(ids_v.as_ptr(), ref_ids, ids_v.len());
            std::ptr::copy_nonoverlapping(w_v.as_ptr(), ref_w, w_v.len());
        }
        run_cold_prefill(&mut strix, &ref_layer, &ref_scratch, rows, io).unwrap();
        strix.hip.device_synchronize().unwrap();
        let t = Instant::now();
        for _ in 0..10 {
            run_cold_prefill(&mut strix, &ref_layer, &ref_scratch, rows, io).unwrap();
            strix.hip.device_synchronize().unwrap();
        }
        println!("strix compute, {rows} rows: {:.2} ms", t.elapsed().as_secs_f64() * 1e3 / 10.0);
    }
    r.active_stream = main_stream;
    let _ = decode_answered;
    println!(
        "PASS: cold rows bitwise == IgpuColdArm ({} cold slots); round trip incl. cold compute {us:.1} us over {n} rounds; answered {}",
        host_slots.len(),
        responder.answered()
    );
}
