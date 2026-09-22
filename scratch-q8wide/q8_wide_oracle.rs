// Temporary slice-0 oracle: gate 1 (wide-vs-segmented equivalence) and
// gate 2 (captured attention-step replay) for the opt-in wide Q8/f16 FA2
// route. NOT a deliverable: the source is archived under scratch-q8wide/
// and this file is removed before landing.
//
// Usage (card B only):
//   HOME=/home/kaden/.hipfire-homes/ab1 ROCR_VISIBLE_DEVICES=GPU-e475645fe0200397 \
//   HIP_VISIBLE_DEVICES=GPU-e475645fe0200397 \
//   HIPFIRE_KERNEL_CACHE=/home/kaden/.hipfire-homes/ab1/.hipfire_kernels \
//   ./target/debug/examples/q8_wide_oracle equiv
//   ./target/debug/examples/q8_wide_oracle capture

use rdna_compute::{DType, Gpu};
use std::sync::Arc;

const NH: usize = 24;
const NKV: usize = 4;
const HD: usize = 256;
const QDIM: usize = NH * HD;
const KVDIM: usize = NKV * HD;
const ROW_BYTES: usize = NKV * (HD / 32) * 34; // 1088

fn lcg(seed: u32, n: usize, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            (((s >> 16) & 0x7fff) as f32 / 32_768.0 - 0.5) * 2.0 * scale
        })
        .collect()
}

fn fnv1a(b: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &x in b {
        h ^= x as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn i32_le_bytes(v: &[i32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(v.len() * 4);
    for &x in v {
        b.extend_from_slice(&x.to_ne_bytes());
    }
    b
}

fn f32_bits_hash(v: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &x in v {
        h ^= x.to_bits() as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn dtoh(gpu: &Gpu, t: &rdna_compute::GpuTensor) -> Vec<u8> {
    let mut b = vec![0u8; t.byte_size()];
    gpu.hip.memcpy_dtoh(&mut b, &t.buf).unwrap();
    b
}

fn upload_positions(gpu: &mut Gpu, pos: &[i32]) -> rdna_compute::GpuTensor {
    let t = gpu.alloc_tensor(&[pos.len()], DType::F32).unwrap();
    gpu.hip.memcpy_htod(&t.buf, &i32_le_bytes(pos)).unwrap();
    t
}

fn sync(gpu: &Gpu) {
    gpu.hip.device_synchronize().unwrap();
}

struct Case {
    q: rdna_compute::GpuTensor,
    k_cache: rdna_compute::GpuTensor,
    v_cache: rdna_compute::GpuTensor,
    positions: rdna_compute::GpuTensor,
    out_wide: rdna_compute::GpuTensor,
    out_seg: rdna_compute::GpuTensor,
    b: usize,
    c: usize,
    start: usize,
}

fn build_case(gpu: &mut Gpu, b: usize, start: usize, seed: u32) -> Case {
    let c = start + b;
    let k_src = lcg(seed, c * KVDIM, 4.0);
    let v_src = lcg(seed ^ 0x9e37, c * KVDIM, 4.0);
    let q_src = lcg(seed ^ 0x51f3, b * QDIM, 4.0);
    let q = gpu.upload_f32(&q_src, &[b * QDIM]).unwrap();
    let kf = gpu.upload_f32(&k_src, &[c * KVDIM]).unwrap();
    let vf = gpu.upload_f32(&v_src, &[c * KVDIM]).unwrap();
    let cache_elems = (c * ROW_BYTES + 3) / 4;
    let k_cache = gpu.alloc_tensor(&[cache_elems], DType::F32).unwrap();
    let v_cache = gpu.alloc_tensor(&[cache_elems], DType::F32).unwrap();
    let pos_full: Vec<i32> = (0..c as i32).collect();
    let pos_full_t = upload_positions(gpu, &pos_full);
    gpu.kv_cache_write_q8_0_batched(&k_cache, &kf, &pos_full_t, NKV, HD, c)
        .unwrap();
    gpu.kv_cache_write_q8_0_batched(&v_cache, &vf, &pos_full_t, NKV, HD, c)
        .unwrap();
    let _ = gpu.free_tensor(kf);
    let _ = gpu.free_tensor(vf);
    let _ = gpu.free_tensor(pos_full_t);
    let pos: Vec<i32> = (start as i32..(start + b) as i32).collect();
    let positions = upload_positions(gpu, &pos);
    let out_wide = gpu.zeros(&[b * QDIM], DType::F32).unwrap();
    let out_seg = gpu.zeros(&[b * QDIM], DType::F32).unwrap();
    sync(gpu);
    Case { q, k_cache, v_cache, positions, out_wide, out_seg, b, c, start }
}

fn launch_direct(
    gpu: &mut Gpu,
    q: &rdna_compute::GpuTensor,
    k: &rdna_compute::GpuTensor,
    v: &rdna_compute::GpuTensor,
    out: &rdna_compute::GpuTensor,
    pos: &rdna_compute::GpuTensor,
    n: usize,
    max_ctx: usize,
) {
    gpu.attention_q8_0_fa2_gqa_gfx1201(q, k, v, out, pos, NH, NKV, HD, max_ctx, n)
        .unwrap();
}

// The captured call lives in its own frame: on return the Rust stack that
// held the kernarg locals is gone before replay runs.
#[inline(never)]
fn launch_direct_frame(
    gpu: &mut Gpu,
    q: &rdna_compute::GpuTensor,
    k: &rdna_compute::GpuTensor,
    v: &rdna_compute::GpuTensor,
    out: &rdna_compute::GpuTensor,
    pos: &rdna_compute::GpuTensor,
    n: usize,
    max_ctx: usize,
) {
    let mut pad = [0u8; 4096];
    for (i, x) in pad.iter_mut().enumerate() {
        *x = (i & 0xff) as u8;
    }
    launch_direct(gpu, q, k, v, out, pos, n, max_ctx);
    std::hint::black_box(pad[1024]);
}
fn zero_out(gpu: &mut Gpu, t: &rdna_compute::GpuTensor) {
    gpu.hip.memset(&t.buf, 0, t.byte_size()).unwrap();
}
fn churn_stack() {
    let mut buf = vec![0u8; 1 << 20];
    for (i, x) in buf.iter_mut().enumerate() {
        *x = ((i * 2654435761usize) & 0xff) as u8;
    }
    std::hint::black_box(&buf[..]);
    fn rec(d: u32, acc: &mut usize) {
        let local = [0xA5u8; 2048];
        *acc += local[(d as usize) & 1023] as usize;
        if d > 0 {
            rec(d - 1, acc);
        }
    }
    let mut acc = 0usize;
    rec(64, &mut acc);
    std::hint::black_box(acc);
}

fn snapshot(gpu: &Gpu, case: &Case) -> (u64, u64, u64) {
    (fnv1a(&dtoh(gpu, &case.q)), fnv1a(&dtoh(gpu, &case.k_cache)), fnv1a(&dtoh(gpu, &case.v_cache)))
}

fn run_equiv_case(gpu: &mut Gpu, b: usize, start: usize, seed: u32) -> bool {
    let case = build_case(gpu, b, start, seed);
    let before = snapshot(gpu, &case);
    // Wide: one launch.
    launch_direct(gpu, &case.q, &case.k_cache, &case.v_cache, &case.out_wide, &case.positions, b, case.c);
    sync(gpu);
    let wide = gpu.download_f32(&case.out_wide).unwrap();
    // Segmented: concatenated <=512 launches, split at 8 below 512 (exercises
    // the q_base=8 workgroup boundary), 512-chunks above.
    let bounds: Vec<(usize, usize)> = if b <= 8 {
        vec![(0, b)]
    } else if b <= 512 {
        vec![(0, 8), (8, b - 8)]
    } else {
        (0..b).step_by(512).map(|off| (off, (b - off).min(512))).collect()
    };
    for (off, seg) in &bounds {
        let qv = case.q.sub_offset(off * QDIM, seg * QDIM);
        let pv = case.positions.sub_offset(*off, *seg);
        let ov = case.out_seg.sub_offset(off * QDIM, seg * QDIM);
        let seg_ctx = case.start + off + seg;
        launch_direct(gpu, &qv, &case.k_cache, &case.v_cache, &ov, &pv, *seg, seg_ctx);
    }
    sync(gpu);
    let seg = gpu.download_f32(&case.out_seg).unwrap();
    let after = snapshot(gpu, &case);
    let mut bad = 0usize;
    let mut nonfinite = 0usize;
    for i in 0..b * QDIM {
        if wide[i].to_bits() != seg[i].to_bits() {
            bad += 1;
        }
        if !wide[i].is_finite() {
            nonfinite += 1;
        }
    }
    let unchanged = before == after;
    let ok = bad == 0 && nonfinite == 0 && unchanged;
    println!(
        "equiv,B={},start={},segs={},wide_hash={:016x},seg_hash={:016x},mismatch={},nonfinite={},cache_unchanged={},{}",
        b,
        start,
        bounds.len(),
        f32_bits_hash(&wide),
        f32_bits_hash(&seg),
        bad,
        nonfinite,
        unchanged,
        if ok { "PASS" } else { "FAIL" }
    );
    ok
}

fn main() {
    let mode = std::env::args().nth(1).unwrap_or_else(|| "equiv".to_string());
    let mut gpu = Gpu::init().expect("Gpu::init");
    eprintln!("arch={} flags.wide={}", gpu.arch, gpu.flags.gfx12_q8_fa2_wide);
    assert_eq!(gpu.arch, "gfx1201", "oracle requires exact gfx1201");
    if !gpu.flags.gfx12_q8_fa2_wide {
        eprintln!("arming wide knob in-process for the oracle (serving gates use typed config)");
        gpu.flags = Arc::new(rdna_compute::FeatureFlags {
            gfx12_q8_fa2_wide: true,
            ..(*gpu.flags).clone()
        });
    }
    // Warm the JIT once (compile tax outside every measurement).
    {
        let warm = build_case(&mut gpu, 64, 0, 0x1234);
        launch_direct(&mut gpu, &warm.q, &warm.k_cache, &warm.v_cache, &warm.out_wide, &warm.positions, 64, 64);
        sync(&gpu);
        eprintln!("warm done");
    }

    let mut all_ok = true;
    if mode == "equiv" {
        for &b in &[64usize, 65, 127, 511, 512, 1024, 8192, 32768] {
            all_ok &= run_equiv_case(&mut gpu, b, 0, 0x1000 + b as u32);
        }
        for &b in &[1usize, 7, 8, 9, 15, 16, 17] {
            all_ok &= run_equiv_case(&mut gpu, b, 0, 0x2000 + b as u32);
        }
        // Offset positions: queries at S..S+B over a longer written prefix,
        // exercising causal bounds away from the regular positions.
        all_ok &= run_equiv_case(&mut gpu, 1024, 1000, 0x3000);
        all_ok &= run_equiv_case(&mut gpu, 65, 7, 0x3001);
        all_ok &= run_equiv_case(&mut gpu, 8192, 256, 0x3002);
    } else if mode == "capture" {
        all_ok &= run_capture(&mut gpu);
    } else {
        eprintln!("unknown mode {mode}");
        std::process::exit(2);
    }
    if !all_ok {
        eprintln!("ORACLE FAILED");
        std::process::exit(1);
    }
    eprintln!("ORACLE PASS ({mode})");
}

fn run_capture(gpu: &mut Gpu) -> bool {
    let mut ok = true;
    // Specified order: B512 eager+capture+replay on fresh scratch first,
    // then B8192 eager (scratch growth WITH the 512-graph alive must
    // invalidate it) + graph rebuild + replay. Every capture is preceded by
    // an eager call at >= that batch: compile + map KV + grow scratch BEFORE
    // capture, never under a captured pointer lifetime.
    for &b in &[512usize, 8192] {
        let case = build_case(gpu, b, 0, 0xB00 + b as u32);
        launch_direct(gpu, &case.q, &case.k_cache, &case.v_cache, &case.out_wide, &case.positions, b, case.c);
        sync(gpu);
        let eager = gpu.download_f32(&case.out_wide).unwrap();
        let before = snapshot(gpu, &case);
        if b == 8192 {
            // The 512-graph must be gone: growth with a live graph destroys
            // it. If it survived, the rebuild below could replay a stale
            // shape.
            let alive = gpu.graphs.graph_exec.is_some();
            let blobs = gpu.graphs.ar_forward_blobs.len();
            let pass = !alive && blobs == 0;
            println!("growth_invalidation,graph_alive={alive},blobs={blobs},{}", if pass { "PASS" } else { "FAIL" });
            ok &= pass;
        }
        // Canaries: patterned device buffers flanking the computation.
        let can_a = gpu.upload_f32(&lcg(0xCA4, 4096, 1.0), &[4096]).unwrap();
        let can_b = gpu.upload_f32(&lcg(0x9CA, 4096, 1.0), &[4096]).unwrap();
        let can_before = (fnv1a(&dtoh(gpu, &can_a)), fnv1a(&dtoh(gpu, &can_b)));

        let stream = gpu.hip.stream_create().unwrap();
        gpu.active_stream = Some(stream);
        gpu.graphs.capture_mode = true;
        gpu.graphs
            .begin_graph_capture(&gpu.hip, gpu.device_id, gpu.active_stream.as_ref().unwrap())
            .expect("begin_graph_capture");
        // Captured call in a returned frame: stack locals are gone at replay.
        launch_direct_frame(gpu, &case.q, &case.k_cache, &case.v_cache, &case.out_seg, &case.positions, b, case.c);
        gpu.graphs
            .end_graph_capture(&gpu.hip, gpu.device_id, gpu.active_stream.as_ref().unwrap())
            .expect("end_graph_capture");
        gpu.graphs.capture_mode = false;
        eprintln!("B={b}: capture ok, blobs={}", gpu.graphs.ar_forward_blobs.len());
        churn_stack();
        for rep in 0..3 {
            zero_out(gpu, &case.out_seg);
            sync(gpu);
            gpu.graphs
                .graph_launch(&gpu.hip, gpu.device_id, gpu.active_stream.as_ref().unwrap())
                .expect("graph_launch");
            gpu.hip.stream_synchronize(gpu.active_stream.as_ref().unwrap()).unwrap();
            let got = gpu.download_f32(&case.out_seg).unwrap();
            let mut bad = 0usize;
            for i in 0..b * QDIM {
                if got[i].to_bits() != eager[i].to_bits() {
                    bad += 1;
                }
            }
            let can_after = (fnv1a(&dtoh(gpu, &can_a)), fnv1a(&dtoh(gpu, &can_b)));
            let inputs_same = snapshot(gpu, &case) == before;
            let pass = bad == 0 && can_after == can_before && inputs_same;
            println!(
                "capture,B={},rep={},mismatch={},canary_same={},inputs_same={},{}",
                b,
                rep,
                bad,
                can_after == can_before,
                inputs_same,
                if pass { "PASS" } else { "FAIL" }
            );
            ok &= pass;
        }
        if b == 512 {
            // Liveness proof for the growth test below: a graph is alive here,
            // so the 8192 eager's scratch growth must destroy it.
            let alive = gpu.graphs.graph_exec.is_some();
            println!("graph_alive_after_512_capture,alive={alive},{}", if alive { "PASS" } else { "FAIL" });
            ok &= alive;
        }
        gpu.active_stream = None;
        let _ = gpu.free_tensor(can_a);
        let _ = gpu.free_tensor(can_b);
    }

    // Ingress under capture_mode (no hip capture): the wide predicate must
    // fire — old code would have fallen through to the incumbent. Route proof
    // is the recorded kernel name plus output equality with the direct call.
    {
        let case = build_case(gpu, 1024, 0, 0x1E55);
        launch_direct(gpu, &case.q, &case.k_cache, &case.v_cache, &case.out_wide, &case.positions, 1024, 1024);
        sync(gpu);
        let direct = gpu.download_f32(&case.out_wide).unwrap();
        gpu.graphs.capture_mode = true;
        gpu.attention_q8_0_flash_prefill_wmma(
            &case.q, &case.k_cache, &case.v_cache, &case.out_seg, &case.positions, NH, NKV, HD, 1024, 1024,
        )
        .unwrap();
        gpu.graphs.capture_mode = false;
        sync(gpu);
        let via_ingress = gpu.download_f32(&case.out_seg).unwrap();
        let mut bad = 0usize;
        for i in 0..1024 * QDIM {
            if via_ingress[i].to_bits() != direct[i].to_bits() {
                bad += 1;
            }
        }
        let route = gpu.last_launched_kernel().unwrap_or_default().to_string();
        let pass = bad == 0 && route == "attention_q8_0_fa2_gqa_gfx1201";
        println!("ingress_capture_mode,route={route},mismatch={bad},{}", if pass { "PASS" } else { "FAIL" });
        ok &= pass;
    }

    // Retained-recording backend, kept separate from the hipGraph evidence
    // above: record the wide step, then re-execute the retained prefix.
    {
        use rdna_compute::replay::{ReplayBackendRequest, ReplayController};
        let case = build_case(gpu, 512, 0, 0xE7A1);
        launch_direct(gpu, &case.q, &case.k_cache, &case.v_cache, &case.out_wide, &case.positions, 512, 512);
        sync(gpu);
        let eager = gpu.download_f32(&case.out_wide).unwrap();
        gpu.replay = ReplayController::new_armed(ReplayBackendRequest::Auto);
        match gpu.replay.begin_capture() {
            Ok(()) => {
                eprintln!("retained: begin_capture ok, recording={}", gpu.replay.is_recording());
                launch_direct_frame(gpu, &case.q, &case.k_cache, &case.v_cache, &case.out_seg, &case.positions, 512, 512);
                sync(gpu);
                match gpu.replay.finish_capture() {
                    Ok(sum) => {
                        eprintln!(
                            "retained: finish ok launches={} unique={}",
                            sum.launch_count, sum.unique_kernel_count
                        );
                        churn_stack();
                        zero_out(gpu, &case.out_seg);
                        sync(gpu);
                        match gpu.replay_recorded_hip_prefix(sum.launch_count) {
                            Ok(()) => {
                                sync(gpu);
                                let got = gpu.download_f32(&case.out_seg).unwrap();
                                let mut bad = 0usize;
                                for i in 0..512 * QDIM {
                                    if got[i].to_bits() != eager[i].to_bits() {
                                        bad += 1;
                                    }
                                }
                                let pass = bad == 0;
                                println!("retained,B=512,mismatch={bad},{}", if pass { "PASS" } else { "FAIL" });
                                ok &= pass;
                            }
                            Err(e) => {
                                println!("retained,replay_prefix_unsupported: {e:?}");
                            }
                        }
                    }
                    Err(e) => println!("retained,finish_unsupported: {e}"),
                }
            }
            Err(e) => println!("retained,begin_unsupported: {e}"),
        }
    }
    ok
}
