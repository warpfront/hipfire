    #[test]
    #[ignore = "requires ordinal-2 gfx1201 + qwen3.8-27b.mq4-xt"]
    fn producer_dead_store_real_prefill_probe() {
        use crate::qwen35::batch::PrefillBatchScratch;
        use crate::qwen35::forward::Qwen35Scratch;
        use crate::qwen35::load::{HfqSource, Layout};
        use crate::qwen35::{
            config_from_hfq, load_weights, DeltaNetState, LayerWeights, Qwen35Config,
        };
        use hipfire_runtime::hfq::HfqFile;
        use hipfire_runtime::llama::KvCache;
        use rdna_compute::{DType, Gpu, GpuTensor};
        use std::time::Duration;

        fn clone_device(gpu: &mut Gpu, src: &GpuTensor, n_f32: usize) -> GpuTensor {
            let dst = gpu.alloc_tensor(&[n_f32], DType::F32).expect("alloc clone");
            gpu.hip.memcpy_dtod(&dst.buf, &src.buf, n_f32 * 4).expect("D2D clone");
            dst
        }
        fn copy_blocks(gpu: &mut Gpu, prep: &rdna_compute::Int4MmqPrepared, k: usize, n: usize) -> GpuTensor {
            let nbytes = (k / 128) * n * 72;
            let dst = gpu.alloc_tensor(&[nbytes.div_ceil(4)], DType::F32).expect("alloc block copy");
            let ptr = gpu.int4_mmq_prepared_ptr(prep, k, n).expect("prepared ptr");
            let src = unsafe { hip_bridge::DeviceBuffer::from_raw(ptr, nbytes) };
            gpu.hip.memcpy_dtod(&dst.buf, &src, nbytes).expect("D2D blocks");
            std::mem::forget(src);
            dst
        }
        fn raw_bytes(gpu: &Gpu, t: &GpuTensor, nbytes: usize) -> Vec<u8> {
            let mut out = vec![0u8; nbytes];
            gpu.hip.memcpy_dtoh(&mut out, &t.buf).expect("D2H compared bytes");
            out
        }
        fn assert_tensor_bytes(gpu: &Gpu, a: &GpuTensor, b: &GpuTensor, nbytes: usize, label: &str) {
            let av = raw_bytes(gpu, a, nbytes);
            let bv = raw_bytes(gpu, b, nbytes);
            assert_eq!(av, bv, "{label} byte mismatch");
        }
        fn event_us(gpu: &Gpu, launch: impl FnOnce()) -> f64 {
            let start = gpu.hip.event_create().expect("start event");
            let stop = gpu.hip.event_create().expect("stop event");
            gpu.hip.event_record(&start, gpu.active_stream.as_ref()).expect("record start");
            launch();
            gpu.hip.event_record(&stop, gpu.active_stream.as_ref()).expect("record stop");
            gpu.hip.event_synchronize(&stop).expect("sync stop");
            let us = gpu.hip.event_elapsed_ms(&start, &stop).expect("elapsed") as f64 * 1000.0;
            gpu.hip.event_destroy(start).expect("destroy start");
            gpu.hip.event_destroy(stop).expect("destroy stop");
            us
        }
        fn stats(xs: &[f64]) -> (f64, f64, f64) {
            let mut sorted = xs.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
            let median = (sorted[9] + sorted[10]) * 0.5;
            (mean, median, sorted[19] - sorted[0])
        }

        let mut gpu = Gpu::init().expect("GPU init");
        assert_eq!(gpu.arch, "gfx1201");
        let models_dir = std::env::var("HIPFIRE_MODELS_DIR").expect("HIPFIRE_MODELS_DIR");
        let model = format!("{models_dir}/qwen3.8-27b.mq4-xt");
        let mut hfq = HfqFile::open(std::path::Path::new(&model)).expect("open model");
        let config: Qwen35Config = config_from_hfq(&hfq).expect("config");
        let weights = {
            let mut src = HfqSource::new(&mut hfq, &config);
            let layout = Layout::single(config.n_layers);
            load_weights(&mut src, std::slice::from_mut(&mut gpu), &layout)
        }.expect("load weights");
        let fa_idx = weights.layers.iter().position(|l| matches!(l, LayerWeights::FullAttn(_))).expect("FA layer");
        let dn_idx = (0..fa_idx).rev().find(|&i| matches!(weights.layers[i], LayerWeights::DeltaNet(_))).expect("preceding GDN layer");
        let dn = match &weights.layers[dn_idx] { LayerWeights::DeltaNet(l) => l, _ => unreachable!() };
        let fa = match &weights.layers[fa_idx] { LayerWeights::FullAttn(l) => l, _ => unreachable!() };
        let n: usize = std::env::var("PROBE_N").ok().and_then(|v| v.parse().ok()).unwrap_or(4096);
        assert_eq!(n % 128, 0);
        let tokens: Vec<u32> = (0..n as u32).collect();
        let mut kv_cache = KvCache::new_gpu_q8(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, n + 32).expect("kv cache");
        let mut dn_state = DeltaNetState::new(&mut gpu, &config).expect("dn state");
        let scratch = Qwen35Scratch::new(&mut gpu, &config, n).expect("scratch");
        let pbs = PrefillBatchScratch::new_opt(&mut gpu, &config, n, false).expect("pbs");
        forward_prefill_batch_with_pbs(
            &mut gpu, &weights, &config, &tokens, 0, &mut kv_cache, &mut dn_state,
            &scratch, None, None, None, None, Some(&pbs), None, Some(fa_idx + 1),
        ).expect("real prefix prefill");
        gpu.sync_with_deadline(Duration::from_secs(60)).expect("prefix sync");
        eprintln!("PROBE real-prefix N={n} dn_layer={dn_idx} fa_layer={fa_idx}");

        // Fork the actual live producer inputs device-to-device. No host-upload fixture participates.
        let residual_src = clone_device(&mut gpu, &pbs.x_batch, n * config.dim);
        let gdn_x = clone_device(&mut gpu, &pbs.dn_attn_out_batch, n * dn.wo.k);
        let gdn_z = clone_device(&mut gpu, &pbs.dn_z_batch, n * dn.wo.k);
        let fa_x = clone_device(&mut gpu, &pbs.fa_attn_out_batch, n * fa.wo.k);

        // LA RMS producer + qkvza consumer. Exact gfx1201 temporary probe route sends all
        // four projections (including beta/alpha tails) directly to IU4.
        let la_k = dn.wqkv.k;
        let la_rot = gpu.alloc_tensor(&[n * la_k], DType::F32).expect("la rot");
        let la_nbytes = (la_k / 128) * n * 72;
        let la_old_res = gpu.reserve_int4_mmq(la_k, n).expect("la old reserve");
        let la_old = gpu.fused_rmsnorm_rotate_mq_i4_gfx12_batched(
            &residual_src, &dn.attn_norm, dn.wqkv.awq_scale.as_ref(), Some(&la_rot),
            la_old_res, la_k, config.norm_eps, n,
        ).expect("la old producer");
        let la_old_blocks = copy_blocks(&mut gpu, &la_old, la_k, n);
        let yq_old = gpu.alloc_tensor(&[n * dn.wqkv.m], DType::F32).expect("yq old");
        let yz_old = gpu.alloc_tensor(&[n * dn.wz.m], DType::F32).expect("yz old");
        let yb_old = gpu.alloc_tensor(&[n * dn.w_beta.m], DType::F32).expect("yb old");
        let ya_old = gpu.alloc_tensor(&[n * dn.w_alpha.m], DType::F32).expect("ya old");
        gpu.gemm_qkvza_mq4g256v2_wmma_iu4_prepared(
            &dn.wqkv.buf, &dn.wz.buf, &dn.w_beta.buf, &dn.w_alpha.buf, &la_rot, &la_old,
            &yq_old, &yz_old, &yb_old, &ya_old, dn.wqkv.m, dn.wz.m, dn.w_beta.m,
            dn.w_alpha.m, la_k, n,
        ).expect("la old projection");
        gpu.hip.memset(&la_rot.buf, 0xA5, n * la_k * 4).expect("poison la f32");
        let yq_canary = gpu.alloc_tensor(&[n * dn.wqkv.m], DType::F32).expect("yq canary");
        let yz_canary = gpu.alloc_tensor(&[n * dn.wz.m], DType::F32).expect("yz canary");
        let yb_canary = gpu.alloc_tensor(&[n * dn.w_beta.m], DType::F32).expect("yb canary");
        let ya_canary = gpu.alloc_tensor(&[n * dn.w_alpha.m], DType::F32).expect("ya canary");
        gpu.gemm_qkvza_mq4g256v2_wmma_iu4_prepared(
            &dn.wqkv.buf, &dn.wz.buf, &dn.w_beta.buf, &dn.w_alpha.buf, &la_rot, &la_old,
            &yq_canary, &yz_canary, &yb_canary, &ya_canary, dn.wqkv.m, dn.wz.m,
            dn.w_beta.m, dn.w_alpha.m, la_k, n,
        ).expect("la poisoned projection");
        assert_tensor_bytes(&gpu, &yq_old, &yq_canary, n * dn.wqkv.m * 4, "LA qkv canary");
        assert_tensor_bytes(&gpu, &yz_old, &yz_canary, n * dn.wz.m * 4, "LA z canary");
        assert_tensor_bytes(&gpu, &yb_old, &yb_canary, n * dn.w_beta.m * 4, "LA beta canary");
        assert_tensor_bytes(&gpu, &ya_old, &ya_canary, n * dn.w_alpha.m * 4, "LA alpha canary");
        let la_new_res = gpu.reserve_int4_mmq(la_k, n).expect("la new reserve");
        let la_new = gpu.fused_rmsnorm_rotate_mq_i4_gfx12_batched(
            &residual_src, &dn.attn_norm, dn.wqkv.awq_scale.as_ref(), None,
            la_new_res, la_k, config.norm_eps, n,
        ).expect("la no-store producer");
        let la_new_blocks = copy_blocks(&mut gpu, &la_new, la_k, n);
        assert_tensor_bytes(&gpu, &la_old_blocks, &la_new_blocks, la_nbytes, "LA block_i4_128");
        eprintln!("PROBE LA canary=PASS blocks={la_nbytes} bytes");

        // GDN producer + prepared residual wo. Poisoned rotated F32 is not in the consumer ABI.
        let gdn_k = dn.wo.k;
        let gdn_rot = gpu.alloc_tensor(&[n * gdn_k], DType::F32).expect("gdn rot");
        let gdn_nbytes = (gdn_k / 128) * n * 72;
        let gdn_old_res = gpu.reserve_int4_mmq(gdn_k, n).expect("gdn old reserve");
        let gdn_old = gpu.gated_norm_rotate_mq_i4_gfx12_batched(
            &gdn_x, &gdn_z, &dn.norm_weight, dn.wo.awq_scale.as_ref(), Some(&gdn_rot),
            gdn_old_res, gdn_k / config.linear_value_head_dim, config.linear_value_head_dim,
            config.norm_eps, gdn_k, n,
        ).expect("gdn old producer");
        let gdn_old_blocks = copy_blocks(&mut gpu, &gdn_old, gdn_k, n);
        let gdn_y_old = clone_device(&mut gpu, &residual_src, n * dn.wo.m);
        gpu.gemm_mq4g256v2_residual_wmma_iu4_prepared(&dn.wo.buf, &gdn_old, &gdn_y_old, dn.wo.m, gdn_k, n).expect("gdn old wo");
        gpu.hip.memset(&gdn_rot.buf, 0xA5, n * gdn_k * 4).expect("poison gdn f32");
        let gdn_y_canary = clone_device(&mut gpu, &residual_src, n * dn.wo.m);
        gpu.gemm_mq4g256v2_residual_wmma_iu4_prepared(&dn.wo.buf, &gdn_old, &gdn_y_canary, dn.wo.m, gdn_k, n).expect("gdn poisoned wo");
        assert_tensor_bytes(&gpu, &gdn_y_old, &gdn_y_canary, n * dn.wo.m * 4, "GDN wo canary");
        let gdn_new_res = gpu.reserve_int4_mmq(gdn_k, n).expect("gdn new reserve");
        let gdn_new = gpu.gated_norm_rotate_mq_i4_gfx12_batched(
            &gdn_x, &gdn_z, &dn.norm_weight, dn.wo.awq_scale.as_ref(), None,
            gdn_new_res, gdn_k / config.linear_value_head_dim, config.linear_value_head_dim,
            config.norm_eps, gdn_k, n,
        ).expect("gdn no-store producer");
        let gdn_new_blocks = copy_blocks(&mut gpu, &gdn_new, gdn_k, n);
        assert_tensor_bytes(&gpu, &gdn_old_blocks, &gdn_new_blocks, gdn_nbytes, "GDN block_i4_128");
        eprintln!("PROBE GDN canary=PASS blocks={gdn_nbytes} bytes");

        // Full-attention sigmoid output rotate producer + prepared residual wo.
        let fa_k = fa.wo.k;
        let fa_rot = gpu.alloc_tensor(&[n * fa_k], DType::F32).expect("fa rot");
        let fa_nbytes = (fa_k / 128) * n * 72;
        let fa_old_res = gpu.reserve_int4_mmq(fa_k, n).expect("fa old reserve");
        let fa_old = gpu.rotate_x_mq_i4_gfx12_batched(
            &fa_x, fa.wo.awq_scale.as_ref(), Some(&fa_rot), fa_old_res, fa_k, n,
        ).expect("fa old producer");
        let fa_old_blocks = copy_blocks(&mut gpu, &fa_old, fa_k, n);
        let fa_y_old = clone_device(&mut gpu, &residual_src, n * fa.wo.m);
        gpu.gemm_mq4g256v2_residual_wmma_iu4_prepared(&fa.wo.buf, &fa_old, &fa_y_old, fa.wo.m, fa_k, n).expect("fa old wo");
        gpu.hip.memset(&fa_rot.buf, 0xA5, n * fa_k * 4).expect("poison fa f32");
        let fa_y_canary = clone_device(&mut gpu, &residual_src, n * fa.wo.m);
        gpu.gemm_mq4g256v2_residual_wmma_iu4_prepared(&fa.wo.buf, &fa_old, &fa_y_canary, fa.wo.m, fa_k, n).expect("fa poisoned wo");
        assert_tensor_bytes(&gpu, &fa_y_old, &fa_y_canary, n * fa.wo.m * 4, "FA wo canary");
        let fa_new_res = gpu.reserve_int4_mmq(fa_k, n).expect("fa new reserve");
        let fa_new = gpu.rotate_x_mq_i4_gfx12_batched(
            &fa_x, fa.wo.awq_scale.as_ref(), None, fa_new_res, fa_k, n,
        ).expect("fa no-store producer");
        let fa_new_blocks = copy_blocks(&mut gpu, &fa_new, fa_k, n);
        assert_tensor_bytes(&gpu, &fa_old_blocks, &fa_new_blocks, fa_nbytes, "FA block_i4_128");
        eprintln!("PROBE FA canary=PASS blocks={fa_nbytes} bytes");

        // Warm both arms once at each site before 20 event samples/arm.
        for emit in [true, false] {
            let r = gpu.reserve_int4_mmq(la_k, n).expect("warm la reserve");
            let p = gpu.fused_rmsnorm_rotate_mq_i4_gfx12_batched(
                &residual_src, &dn.attn_norm, dn.wqkv.awq_scale.as_ref(), emit.then_some(&la_rot),
                r, la_k, config.norm_eps, n,
            ).expect("warm la producer");
            gpu.gemm_qkvza_mq4g256v2_wmma_iu4_prepared(
                &dn.wqkv.buf, &dn.wz.buf, &dn.w_beta.buf, &dn.w_alpha.buf, &la_rot, &p,
                &yq_canary, &yz_canary, &yb_canary, &ya_canary, dn.wqkv.m, dn.wz.m,
                dn.w_beta.m, dn.w_alpha.m, la_k, n,
            ).expect("warm la projection");
            let r = gpu.reserve_int4_mmq(gdn_k, n).expect("warm gdn reserve");
            let p = gpu.gated_norm_rotate_mq_i4_gfx12_batched(
                &gdn_x, &gdn_z, &dn.norm_weight, dn.wo.awq_scale.as_ref(), emit.then_some(&gdn_rot),
                r, gdn_k / config.linear_value_head_dim, config.linear_value_head_dim,
                config.norm_eps, gdn_k, n,
            ).expect("warm gdn producer");
            gpu.hip.memcpy_dtod(&gdn_y_canary.buf, &residual_src.buf, n * dn.wo.m * 4).expect("warm gdn residual");
            gpu.gemm_mq4g256v2_residual_wmma_iu4_prepared(&dn.wo.buf, &p, &gdn_y_canary, dn.wo.m, gdn_k, n).expect("warm gdn wo");
            let r = gpu.reserve_int4_mmq(fa_k, n).expect("warm fa reserve");
            let p = gpu.rotate_x_mq_i4_gfx12_batched(&fa_x, fa.wo.awq_scale.as_ref(), emit.then_some(&fa_rot), r, fa_k, n).expect("warm fa producer");
            gpu.hip.memcpy_dtod(&fa_y_canary.buf, &residual_src.buf, n * fa.wo.m * 4).expect("warm fa residual");
            gpu.gemm_mq4g256v2_residual_wmma_iu4_prepared(&fa.wo.buf, &p, &fa_y_canary, fa.wo.m, fa_k, n).expect("warm fa wo");
        }
        gpu.sync_with_deadline(Duration::from_secs(60)).expect("warm sync");

        let mut la_old_us = Vec::with_capacity(20); let mut la_new_us = Vec::with_capacity(20);
        let mut gdn_old_us = Vec::with_capacity(20); let mut gdn_new_us = Vec::with_capacity(20);
        let mut fa_old_us = Vec::with_capacity(20); let mut fa_new_us = Vec::with_capacity(20);
        for pair in 0..10 {
            for emit in if pair & 1 == 0 { [true, false, false, true] } else { [false, true, true, false] } {
                let r = gpu.reserve_int4_mmq(la_k, n).expect("timed la reserve");
                let start = gpu.hip.event_create().unwrap(); let stop = gpu.hip.event_create().unwrap();
                gpu.hip.event_record(&start, gpu.active_stream.as_ref()).unwrap();
                let p = gpu.fused_rmsnorm_rotate_mq_i4_gfx12_batched(
                    &residual_src, &dn.attn_norm, dn.wqkv.awq_scale.as_ref(), emit.then_some(&la_rot),
                    r, la_k, config.norm_eps, n,
                ).expect("timed la producer");
                gpu.gemm_qkvza_mq4g256v2_wmma_iu4_prepared(
                    &dn.wqkv.buf, &dn.wz.buf, &dn.w_beta.buf, &dn.w_alpha.buf, &la_rot, &p,
                    &yq_canary, &yz_canary, &yb_canary, &ya_canary, dn.wqkv.m, dn.wz.m,
                    dn.w_beta.m, dn.w_alpha.m, la_k, n,
                ).expect("timed la projection");
                gpu.hip.event_record(&stop, gpu.active_stream.as_ref()).unwrap(); gpu.hip.event_synchronize(&stop).unwrap();
                let us = gpu.hip.event_elapsed_ms(&start, &stop).unwrap() as f64 * 1000.0;
                gpu.hip.event_destroy(start).unwrap(); gpu.hip.event_destroy(stop).unwrap();
                (if emit { &mut la_old_us } else { &mut la_new_us }).push(us);

                gpu.hip.memcpy_dtod(&gdn_y_canary.buf, &residual_src.buf, n * dn.wo.m * 4).unwrap();
                let r = gpu.reserve_int4_mmq(gdn_k, n).expect("timed gdn reserve");
                let start = gpu.hip.event_create().unwrap(); let stop = gpu.hip.event_create().unwrap();
                gpu.hip.event_record(&start, gpu.active_stream.as_ref()).unwrap();
                let p = gpu.gated_norm_rotate_mq_i4_gfx12_batched(
                    &gdn_x, &gdn_z, &dn.norm_weight, dn.wo.awq_scale.as_ref(), emit.then_some(&gdn_rot),
                    r, gdn_k / config.linear_value_head_dim, config.linear_value_head_dim,
                    config.norm_eps, gdn_k, n,
                ).expect("timed gdn producer");
                gpu.gemm_mq4g256v2_residual_wmma_iu4_prepared(&dn.wo.buf, &p, &gdn_y_canary, dn.wo.m, gdn_k, n).expect("timed gdn wo");
                gpu.hip.event_record(&stop, gpu.active_stream.as_ref()).unwrap(); gpu.hip.event_synchronize(&stop).unwrap();
                let us = gpu.hip.event_elapsed_ms(&start, &stop).unwrap() as f64 * 1000.0;
                gpu.hip.event_destroy(start).unwrap(); gpu.hip.event_destroy(stop).unwrap();
                (if emit { &mut gdn_old_us } else { &mut gdn_new_us }).push(us);

                gpu.hip.memcpy_dtod(&fa_y_canary.buf, &residual_src.buf, n * fa.wo.m * 4).unwrap();
                let r = gpu.reserve_int4_mmq(fa_k, n).expect("timed fa reserve");
                let start = gpu.hip.event_create().unwrap(); let stop = gpu.hip.event_create().unwrap();
                gpu.hip.event_record(&start, gpu.active_stream.as_ref()).unwrap();
                let p = gpu.rotate_x_mq_i4_gfx12_batched(&fa_x, fa.wo.awq_scale.as_ref(), emit.then_some(&fa_rot), r, fa_k, n).expect("timed fa producer");
                gpu.gemm_mq4g256v2_residual_wmma_iu4_prepared(&fa.wo.buf, &p, &fa_y_canary, fa.wo.m, fa_k, n).expect("timed fa wo");
                gpu.hip.event_record(&stop, gpu.active_stream.as_ref()).unwrap(); gpu.hip.event_synchronize(&stop).unwrap();
                let us = gpu.hip.event_elapsed_ms(&start, &stop).unwrap() as f64 * 1000.0;
                gpu.hip.event_destroy(start).unwrap(); gpu.hip.event_destroy(stop).unwrap();
                (if emit { &mut fa_old_us } else { &mut fa_new_us }).push(us);
            }
        }
        assert_eq!((la_old_us.len(), la_new_us.len(), gdn_old_us.len(), gdn_new_us.len(), fa_old_us.len(), fa_new_us.len()), (20,20,20,20,20,20));
        let (laom, laomed, laosp) = stats(&la_old_us); let (lanm, lanmed, lansp) = stats(&la_new_us);
        let (gom, gomed, gosp) = stats(&gdn_old_us); let (gnm, gnmed, gnsp) = stats(&gdn_new_us);
        let (fom, fomed, fosp) = stats(&fa_old_us); let (fnm, fnmed, fnsp) = stats(&fa_new_us);
        let net_us_token = (laom - lanm) * 48.0 / n as f64 + (gom - gnm) * 48.0 / n as f64 + (fom - fnm) * 16.0 / n as f64;
        eprintln!("PROBE TIMING LA old_mean_us={laom:.3} new_mean_us={lanm:.3} old_median={laomed:.3} new_median={lanmed:.3} old_span={laosp:.3} new_span={lansp:.3}");
        eprintln!("PROBE TIMING GDN old_mean_us={gom:.3} new_mean_us={gnm:.3} old_median={gomed:.3} new_median={gnmed:.3} old_span={gosp:.3} new_span={gnsp:.3}");
        eprintln!("PROBE TIMING FA old_mean_us={fom:.3} new_mean_us={fnm:.3} old_median={fomed:.3} new_median={fnmed:.3} old_span={fosp:.3} new_span={fnsp:.3}");
        eprintln!("PROBE NET_US_TOKEN={net_us_token:.6} threshold=6.399");
    }
