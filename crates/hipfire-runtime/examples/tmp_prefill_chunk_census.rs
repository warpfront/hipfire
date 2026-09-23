// SPDX-License-Identifier: Apache-2.0
// Temporary prefill-graph launch census (PrefillGraph slice).
// O-owned throwaway: public forward/wrapper APIs + counters only.
//
// Drives ONE 8192-token prefill (2x4096 widened chunks, production layout:
// shared owned PBS, carried DN state) under tape recording, splits the
// recording at the chunk boundary by kernel-sequence repetition, and diffs
// the two halves' kernargs slot by slot.
//
// Usage (ordinal 0 box):
//   HOME=/home/kaden/.hipfire-homes/ab0 ROCR_VISIBLE_DEVICES=0 \
//   HIPFIRE_MODELS_DIR=/home/kaden/.hipfire/models \
//   HIPFIRE_KERNEL_CACHE=/home/kaden/.hipfire-homes/ab0/.hipfire_kernels \
//   HIPFIRE_LLOYD_GFX12=1 HIPFIRE_REPLAY_BACKEND=auto HIPFIRE_REPLAY_MANUAL_CAPTURE=1 \
//   cargo run --release -p hipfire-runtime --features deltanet \
//     --example tmp_prefill_chunk_census -- \
//     --model /home/kaden/.hipfire/models/qwen3.8-27b.mq4-xt \
//     --out-dir scratch-2026-09-17/PrefillGraph
// Exit: 0 census written; 1 mismatch/launch failure; 2 usage/environment error.

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("tmp_prefill_chunk_census: build with --features deltanet");
    std::process::exit(2);
}

#[cfg(feature = "deltanet")]
mod ora {
    use hipfire_arch_qwen35::qwen35::{
        self, DeltaNetState, LayerWeights, PrefillBatchScratch, Qwen35Scratch,
    };
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::llama::KvCache;
    use rdna_compute::{DType, Gpu};
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    fn det_tokens(l: usize, vocab: usize) -> Vec<u32> {
        (0..l).map(|i| (((i * 131) % (vocab - 1)) + 1) as u32).collect()
    }

    fn dtype_of(layer: &LayerWeights) -> Vec<DType> {
        match layer {
            LayerWeights::DeltaNet(l) => vec![
                l.wqkv.gpu_dtype,
                l.wz.gpu_dtype,
                l.w_beta.gpu_dtype,
                l.w_alpha.gpu_dtype,
                l.wo.gpu_dtype,
                l.w_gate.gpu_dtype,
                l.w_up.gpu_dtype,
                l.w_down.gpu_dtype,
            ],
            LayerWeights::FullAttn(l) => vec![
                l.wq.gpu_dtype,
                l.wk.gpu_dtype,
                l.wv.gpu_dtype,
                l.wo.gpu_dtype,
                l.w_gate.gpu_dtype,
                l.w_up.gpu_dtype,
                l.w_down.gpu_dtype,
            ],
            LayerWeights::DeltaNetMoe(_) => vec![],
            LayerWeights::FullAttnMoe(_) => vec![],
        }
    }

    pub fn real_main() {
        let a: Vec<String> = std::env::args().collect();
        let mut model: Option<PathBuf> = None;
        let mut out_dir = PathBuf::from("scratch-2026-09-17/PrefillGraph");
        let mut single = false;
        let mut i = 1;
        while i < a.len() {
            match a[i].as_str() {
                "--model" => {
                    model = Some(PathBuf::from(&a[i + 1]));
                    i += 2;
                }
                "--out-dir" => {
                    out_dir = PathBuf::from(&a[i + 1]);
                    i += 2;
                }
                "--single" => {
                    single = true;
                    i += 1;
                }
                "--driver-test" => {
                    i += 2;
                }
                h => {
                    eprintln!("unknown flag {h}; want --model --out-dir [--single]");
                    std::process::exit(2);
                }
            }
        }
        let model = model.unwrap_or_else(|| {
            eprintln!("--model required");
            std::process::exit(2);
        });
        std::fs::create_dir_all(&out_dir).unwrap_or_else(|e| {
            eprintln!("out-dir: {e:?}");
            std::process::exit(2);
        });
        let mut out = String::new();
        let mut log = |s: &str| {
            eprintln!("{s}");
            out.push_str(s);
            out.push('\n');
        };

        let mut gpu = Gpu::init().unwrap_or_else(|e| {
            eprintln!("gpu init failed: {e:?}");
            std::process::exit(2);
        });
        if gpu.arch != "gfx1201" {
            eprintln!("census requires exact gfx1201, got {}", gpu.arch);
            std::process::exit(2);
        }
        log(&format!("arch={} replay_backend_state_init", gpu.arch));

        // ── load ──
        let mut hfq = HfqFile::open(&model).expect("open model");
        let config = qwen35::config_from_hfq(&hfq).expect("read config");
        log(&format!(
            "model: dim={} hidden={} layers={} heads={}/{}/{} kv_heads={} head_dim={} vocab={} experts={}",
            config.dim,
            config.hidden_dim,
            config.n_layers,
            config.n_heads,
            config.n_kv_heads,
            config.head_dim,
            config.n_kv_heads,
            config.head_dim,
            config.vocab_size,
            config.num_experts,
        ));
        let mut src = qwen35::HfqSource::new(&mut hfq, &config);
        let layout = qwen35::Layout::single(config.n_layers);
        let weights =
            qwen35::load_weights(&mut src, std::slice::from_mut(&mut gpu), &layout)
                .expect("load weights");
        for (li, lw) in weights.layers.iter().enumerate().take(2) {
            log(&format!("L{li} dtypes={:?}", dtype_of(lw)));
        }
        log(&format!(
            "flags: fp8_gateup={} fp8_resid={} fp8_qkvza={} fp8_qkv={} fp8_v2={} gdn_pre_fused={} silu_quant_fused={} producer_quant_fused={} fp8_stream={} fa2_prefill={}",
            gpu.flags.gfx12_mq4v2_fp8_gateup,
            gpu.flags.gfx12_mq4v2_fp8_resid,
            gpu.flags.gfx12_mq4v2_fp8_qkvza,
            gpu.flags.gfx12_mq4v2_fp8_qkv,
            gpu.flags.gfx12_mq4v2_fp8_v2,
            gpu.flags.gfx12_gdn_pre_fused,
            gpu.flags.gfx12_silu_quant_fused,
            gpu.flags.gfx12_producer_quant_fused,
            gpu.flags.gfx12_fp8_stream,
            gpu.flags.gfx12_fa2_prefill,
        ));

        let l = 8192usize;
        let kv_max = l + 16;
        let is_kv_layer: Vec<bool> = config
            .layer_types
            .iter()
            .map(|t| *t == qwen35::LayerType::FullAttention)
            .collect();
        let n_fa = is_kv_layer.iter().filter(|x| **x).count();
        let n_la = config.n_layers - n_fa;
        log(&format!("layers: LA={n_la} FA={n_fa}"));
        let mut kv =
            KvCache::new_gpu_fp8_filtered(&mut gpu, &is_kv_layer, config.n_kv_heads, config.head_dim, kv_max)
                .unwrap_or_else(|e| {
                    eprintln!("kv fp8: {e:?}");
                    std::process::exit(1);
                });
        let mut dn = DeltaNetState::new(&mut gpu, &config).expect("dn fresh");
        log(&format!(
            "dn: quant={:?} s={} sc={} ef={} conv={}",
            dn.quant,
            dn.s_matrices.len(),
            dn.s_scales.len(),
            dn.s_ef_residual.len(),
            dn.conv_states.len()
        ));
        match qwen35::ordinary_prefill_chunk_limit(&gpu, &weights, &config, &dn, &kv, None) {
            Ok(a) => log(&format!("ordinary_prefill_chunk_limit admitted={a}")),
            Err(e) => {
                log(&format!("admission-query failed: {e:?}"));
                std::fs::write(out_dir.join("census.txt"), &out).ok();
                std::process::exit(1);
            }
        }
        let scratch = Qwen35Scratch::new_with_kv_max(&mut gpu, &config, 128, kv_max)
            .unwrap_or_else(|e| panic!("scratch: {e:?}"));
        let toks = det_tokens(l, config.vocab_size);
        // Census needs >8192 launch capacity per chunk: manual PM4 secondary
        // controller with a raised cap. Recording is transport-independent;
        // only the launch funnel matters here.
        gpu.replay = rdna_compute::replay::ReplayController::new_manual_pm4()
            .with_max_recorded_launches(16_384);
        log("replay controller: manual_pm4 cap=16384");
        log("warmup: forward_prefill_batch 8192 (JIT)");
        qwen35::forward_prefill_batch(
            &mut gpu, &weights, &config, &toks, 0, &mut kv, &mut dn, &scratch, None, None, None,
            None,
        )
        .unwrap_or_else(|e| panic!("warmup: {e:?}"));
        gpu.hip.device_synchronize().expect("sync");
        // Driver exactness protocol (two processes; the flag is a
        // process-start snapshot, invisible to mid-process set_var):
        //   pass 1: outer HIPFIRE_GFX12_PREFILL_GRAPH=1 --driver-test snap.bin
        //           (no file yet -> run + write snap with frame checkpoint)
        //   pass 2: outer flag unset          --driver-test snap.bin
        //           (file exists -> restore checkpoint, run, compare exact)
        // Eager-vs-eager (both flag unset) must also PASS (harness check).
        if let Some(pos) = a.iter().position(|x| x == "--driver-test") {
            let snap_path = PathBuf::from(&a[pos + 1]);
            let mut kv = KvCache::new_gpu_fp8_filtered(
                &mut gpu, &is_kv_layer, config.n_kv_heads, config.head_dim, kv_max,
            )
            .expect("kv fresh");
            let mut dn = DeltaNetState::new(&mut gpu, &config).expect("dn fresh");
            let hbuf = gpu
                .alloc_tensor(&[l, config.dim], DType::F32)
                .expect("hbuf");
            // (live-env read mirrors the process-start snapshot taken from the same env)
            let flag_on = std::env::var("HIPFIRE_GFX12_PREFILL_GRAPH")
                .map(|v| {
                    let v = v.trim().to_owned();
                    !v.is_empty() && v != "0"
                })
                .unwrap_or(false);
            if !std::path::Path::new(&snap_path).exists() {
                let ck = rdna_compute::norm::gdn_requant_frame_checkpoint();
                qwen35::forward_prefill_batch(
                    &mut gpu, &weights, &config, &toks, 0, &mut kv, &mut dn, &scratch,
                    None, Some(&hbuf), None, None,
                )
                .expect("snap forward");
                gpu.hip.device_synchronize().expect("sync");
                let mut bytes = Vec::new();
                bytes.extend_from_slice(&ck.to_le_bytes());
                for t in [
                    gpu.download_f32(&scratch.logits).expect("logits"),
                    gpu.download_f32(&hbuf).expect("hidden"),
                    gpu.download_f32(&dn.s_scales[0]).expect("sc0"),
                    gpu.download_f32(&dn.conv_states[0]).expect("cv0"),
                ] {
                    bytes.extend_from_slice(&(t.len() as u64).to_le_bytes());
                    for v in &t {
                        bytes.extend_from_slice(&v.to_le_bytes());
                    }
                }
                std::fs::write(&snap_path, &bytes).expect("write snap");
                log(&format!("snap written: {} B ck={ck}", bytes.len()));
                std::process::exit(0);
            }
            let raw = std::fs::read(&snap_path).expect("read snap");
            let ck = u32::from_le_bytes(raw[0..4].try_into().unwrap());
            let mut off = 4usize;
            let mut expect = Vec::new();
            while off < raw.len() {
                let n = u64::from_le_bytes(raw[off..off + 8].try_into().unwrap()) as usize;
                off += 8;
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    v.push(f32::from_le_bytes(raw[off..off + 4].try_into().unwrap()));
                    off += 4;
                }
                expect.push(v);
            }
            rdna_compute::norm::restore_gdn_requant_frame_checkpoint(ck);
            log(&format!("restored checkpoint ck={ck}, running"));
            qwen35::forward_prefill_batch(
                &mut gpu, &weights, &config, &toks, 0, &mut kv, &mut dn, &scratch, None,
                Some(&hbuf), None, None,
            )
            .expect("cmp forward");
            gpu.hip.device_synchronize().expect("sync");
            let got = [
                gpu.download_f32(&scratch.logits).expect("logits"),
                gpu.download_f32(&hbuf).expect("hidden"),
                gpu.download_f32(&dn.s_scales[0]).expect("sc0"),
                gpu.download_f32(&dn.conv_states[0]).expect("cv0"),
            ];
            let mut fails = 0;
            for (name, (x, y)) in ["logits", "hidden", "scales", "conv"]
                .iter()
                .zip(expect.iter().zip(got.iter()))
            {
                if x == y {
                    log(&format!("  ok   {name} ({} f32)", x.len()));
                } else {
                    let nd = x.iter().zip(y.iter()).filter(|(p, q)| p != q).count();
                    fails += 1;
                    log(&format!("  DIFF {name}: {nd}/{} f32 differ", x.len()));
                }
            }
            log(&format!("driver-test: {}", if fails == 0 { "PASS" } else { "FAIL" }));
            std::fs::write(out_dir.join("census-driver.txt"), &out).ok();
            std::process::exit(if fails == 0 { 0 } else { 1 });
        }

        // Fresh state, same sizes (steady-state census, no growth inside).
        let mut kv =
            KvCache::new_gpu_fp8_filtered(&mut gpu, &is_kv_layer, config.n_kv_heads, config.head_dim, kv_max)
                .expect("kv fresh");
        let mut dn = DeltaNetState::new(&mut gpu, &config).expect("dn fresh");

        // Two-call mode: two consecutive 4096 calls (resume; per-call PBS, so
        // pbs pointers slide between halves). Single mode: ONE 8192 call
        // (production layout: shared PBS, continued GDN frames); the recording
        // is split at the chunk boundary by kernel-sequence repetition.
        let drives: Vec<(&[u32], usize)> = if single {
            vec![(&toks[..], 0)]
        } else {
            vec![(&toks[..4096], 0), (&toks[4096..], 4096)]
        };
        let mut recs = Vec::new();
        for (ci, (toks_c, start)) in drives.iter().enumerate()
        {
            hip_bridge::launch_counters::reset();
            gpu.replay.set_forward_eligible(true);
            if let Err(e) = gpu.replay.begin_capture() {
                log(&format!("chunk{ci} begin_capture failed: {e}"));
                std::fs::write(out_dir.join("census.txt"), &out).ok();
                std::process::exit(2);
            }
            qwen35::forward_prefill_batch(
                &mut gpu, &weights, &config, toks_c, *start, &mut kv, &mut dn, &scratch, None,
                None, None, None,
            )
            .unwrap_or_else(|e| panic!("chunk{ci} forward: {e:?}"));
            gpu.hip.device_synchronize().expect("sync");
            log(&format!(
                "chunk{ci} post: state={:?} fallback={:?} recorded={}",
                gpu.replay.state(),
                gpu.replay.fallback_reason(),
                gpu.replay.recorded_launches().len(),
            ));
            let summary = gpu
                .replay
                .finish_capture()
                .unwrap_or_else(|e| panic!("chunk{ci} finish_capture: {e}"));
            let total = hip_bridge::launch_counters::launch_kernel::count();
            log(&format!(
                "chunk{ci}: tape={} total={} rawinv={} htod={} dtod={} dtoh={} memset={} ssync={} esync={} dsync={} enslookup={} unique={} seq_hash={:016x}",
                summary.launch_count,
                total,
                total.saturating_sub(summary.launch_count as u64),
                hip_bridge::launch_counters::memcpy_htod::count(),
                hip_bridge::launch_counters::memcpy_dtod::count(),
                hip_bridge::launch_counters::memcpy_dtoh::count(),
                hip_bridge::launch_counters::memset::count(),
                hip_bridge::launch_counters::stream_sync::count(),
                hip_bridge::launch_counters::event_sync::count(),
                hip_bridge::launch_counters::device_sync::count(),
                hip_bridge::launch_counters::ensure_kernel_lookup::count(),
                summary.unique_kernel_count,
                summary.sequence_hash,
            ));
            recs.push(gpu.replay.recorded_launches().to_vec());
        }
        // Single mode: split the one recording at the chunk boundary (kernel
        // sequence must repeat exactly; production-true frames + shared PBS).
        let halves;
        if single {
            let full = &recs[0];
            let n = full.len();
            let mut m = 0usize;
            if n % 2 == 0 && (0..n / 2).all(|k| full[k].kernel == full[k + n / 2].kernel) {
                m = n / 2;
            }
            if m == 0 {
                log(&format!("single: NO clean 2x kernel repetition over {n} launches"));
                std::fs::write(out_dir.join("census.txt"), &out).ok();
                std::process::exit(1);
            }
            log(&format!("single: split M={m} (shared PBS, continued frames)"));
            halves = vec![full[..m].to_vec(), full[m..].to_vec()];
        } else {
            halves = std::mem::take(&mut recs);
        }
        let rec: &Vec<rdna_compute::replay::RecordedHipLaunch> = &halves[0];
        let rec1: &Vec<rdna_compute::replay::RecordedHipLaunch> = &halves[1];
        // Histogram.
        let mut hist: BTreeMap<&str, usize> = BTreeMap::new();
        let mut typed = 0usize;
        let mut grid_bound = 0usize;
        for r in rec.iter() {
            *hist.entry(r.kernel.as_str()).or_default() += 1;
            if r.binding_layout.is_some() {
                typed += 1;
            }
            if r.grid_binding.is_some() {
                grid_bound += 1;
            }
        }
        log(&format!(
            "typed(binding_layout)={typed} untyped={} grid_bound={}",
            rec.len() - typed,
            grid_bound
        ));
        log("per-kernel histogram (all launches):");
        for (k, c) in &hist {
            log(&format!("  {c:5}  {k}"));
        }
        // Untyped kernel list (needs-ABI set for patching).
        {
            let mut untyped: BTreeMap<&str, usize> = BTreeMap::new();
            for r in rec.iter() {
                if r.binding_layout.is_none() {
                    *untyped.entry(r.kernel.as_str()).or_default() += 1;
                }
            }
            if untyped.is_empty() {
                log("untyped kernels: NONE");
            } else {
                log("untyped kernels (raw-snapshot path):");
                for (k, c) in &untyped {
                    log(&format!("  {c:5}  {k}"));
                }
            }
        }

        // ── diff chunk0 vs chunk1 ──
        // Group differing runs by (kernel, offset, len); collapse 64 layers.
        #[derive(PartialEq, Eq, PartialOrd, Ord)]
        struct Key {
            kernel: String,
            off: usize,
            len: usize,
        }
        struct Agg {
            count: usize,
            a0: Vec<u8>,
            b0: Vec<u8>,
            idx0: usize,
        }
        let mut aggs: BTreeMap<Key, Agg> = BTreeMap::new();
        let mut grid_mismatch = 0usize;
        let mut len_mismatch = 0usize;
        let mut kernel_mismatch = 0usize;
        if rec.len() != rec1.len() {
            log(&format!("LAUNCH COUNT differs: chunk0={} chunk1={}", rec.len(), rec1.len()));
        }
        let m = rec.len().min(rec1.len());
        for k in 0..m {
            let (x, y) = (&rec[k], &rec1[k]);
            if x.kernel != y.kernel {
                kernel_mismatch += 1;
                if kernel_mismatch <= 10 {
                    log(&format!("KERNEL mismatch launch {k}: {} vs {}", x.kernel, y.kernel));
                }
                continue;
            }
            // (x, y) already bound to (chunk0[k], chunk1[k]) above.
            if x.grid != y.grid || x.block != y.block || x.shared_mem != y.shared_mem {
                grid_mismatch += 1;
                log(&format!(
                    "GRID/BLOCK/LDS mismatch launch {k} {}: {:?}/{:?}/{:?} vs {:?}/{:?}/{:?}",
                    x.kernel, x.grid, x.block, x.shared_mem, y.grid, y.block, y.shared_mem
                ));
            }
            if x.kernarg.len() != y.kernarg.len() {
                len_mismatch += 1;
                continue;
            }
            let (a, b) = (&x.kernarg, &y.kernarg);
            let mut o = 0;
            while o < a.len() {
                if a[o] != b[o] {
                    let mut e = o + 1;
                    while e < a.len() && a[e] != b[e] {
                        e += 1;
                    }
                    let key = Key {
                        kernel: x.kernel.clone(),
                        off: o,
                        len: e - o,
                    };
                    aggs.entry(key).or_insert_with(|| Agg {
                        count: 0,
                        a0: a[o..e].to_vec(),
                        b0: b[o..e].to_vec(),
                        idx0: k,
                    }).count += 1;
                    o = e;
                } else {
                    o += 1;
                }
            }
        }
        log(&format!("grid/block/lds mismatches={grid_mismatch} kernarg-len mismatches={len_mismatch}"));
        fn u64le(v: &[u8]) -> u128 {
            let mut x: u128 = 0;
            for (j, by) in v.iter().enumerate().take(16) {
                x |= (*by as u128) << (8 * j);
            }
            x
        }
        // Split aggs into ptr-like (any side >= 0x10_0000 and len==8) vs scalar.
        let mut ptr_deltas: BTreeMap<i128, usize> = BTreeMap::new();
        let mut n_ptr = 0usize;
        log("scalar (non-pointer) variant slots, grouped by (kernel, off, len):");
        for (key, ag) in &aggs {
            let va = u64le(&ag.a0);
            let vb = u64le(&ag.b0);
            let ptr_like = key.len == 8 && (va >= 0x10_0000 || vb >= 0x10_0000);
            if ptr_like {
                n_ptr += 1;
                *ptr_deltas.entry(vb as i128 - va as i128).or_default() += 1;
                continue;
            }
            let fa = if ag.a0.len() == 4 {
                let f = f32::from_le_bytes(ag.a0[0..4].try_into().unwrap());
                format!(" f32={f:e}")
            } else {
                String::new()
            };
            log(&format!(
                "  n={:4}  {:64} off={:3} len={:2} A={:#x} B={:#x} d={:+} idx0={} {fa}",
                ag.count,
                key.kernel,
                key.off,
                key.len,
                va,
                vb,
                vb as i128 - va as i128,
                ag.idx0,
            ));
        }
        log(&format!("pointer-like variant slots: {n_ptr} in {} distinct deltas", ptr_deltas.len()));
        for (d, c) in ptr_deltas.iter().take(40) {
            log(&format!("  ptr-delta {d:+#x} x{c}"));
        }
        // Suppress unused-import warning for PrefillBatchScratch if untouched.
        let _ = std::mem::size_of::<PrefillBatchScratch>();
        std::fs::write(out_dir.join("census.txt"), &out).unwrap_or_else(|e| panic!("write: {e:?}"));
    }
}

#[cfg(feature = "deltanet")]
fn main() {
    ora::real_main();
}
