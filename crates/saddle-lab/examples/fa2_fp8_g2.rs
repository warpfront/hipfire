// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Keeper real-data oracle for the gfx1201 FA2 fp8 stage-a planes.
//!
//! Plan of record: `docs/plans/2026-09-17-gfx1201-fa2-fp8-legs.md` §4 / stage a.
//! `--kernel-out PATH` dumps pre-gate O + gate from the flag-resolved FA2
//! body for an ON vs OFF comparison. CPU paths remain for the screen table.
//!
//! # Data provenance ("which and how")
//!
//! Real Q, K, V from `qwen3.8-27b.mq4-xt` at a 4224-token prompt, produced by
//! the REAL engine on gfx1201 — not by the FA2 oracle fixture path (that file
//! named in the plan, `fa2_prefill_gfx1201_oracle.rs`, does not exist in this
//! tree) and not by `dump_qwen35_hidden_states.rs` (that harness needs an
//! `.hfq` model + a `.kldref` token source; the canonical trunk is `.mq4-xt`
//! and there is no kldref handy).
//!
//! Instead this example drives the production prefill entry point directly:
//! `forward_prefill_batch_with_pbs_opts` with `max_layer = Some(L+1)` on a
//! caller-owned `PrefillBatchScratch`, `KvCache::new_gpu_q8`, and the prompt
//! `tokens = 0..N` (deterministic synthetic IDs, the same convention as
//! `profile_prefill_qwen35.rs` — valid token IDs through the real embedding
//! and the real weights, so every activation is a genuine model activation).
//! After the truncated forward:
//!
//! - Q: `pbs.fa_q_batch` rows of the final chunk — post-QK-norm, post-RoPE
//!   f32, i.e. bit-for-bit the Q the FA2 kernel consumes (pre-f16-convert).
//! - K/V: the engine's q8_0 KV cache rows (`k_gpu[L]` / `v_gpu[L]`) parsed per
//!   the kernel's addressing (`row_stride = n_kv_heads·(head_dim/32)·34`,
//!   34 B blocks: LE f16 scale + 32 int8 codes) — the codes + scales AS STORED.
//! - O_engine: `pbs.fa_attn_out_batch` for cross-checking the CPU reference.
//!   NOTE: this buffer is POST-gate — the engine applies `sigmoid(gate)·attn`
//!   in place (prefill.rs:6667). The screen downloads `fa_gate_batch` and
//!   applies the same gate to every CPU O path, so all metrics are on the
//!   true O and check-B compares like with like.
//!
//! Two self-checks validate the taps before the screen runs:
//! (A) dequant(stored K/V blocks) vs downloaded `fa_k/fa_v` rows (must agree
//! to f16-rounding level — proves block parsing); (B) CPU reference O vs
//! `fa_attn_out_batch` (must agree to wmma-f16 level — proves Q/K/V/scale/
//! mask handling).
//!
//! # The four CPU paths (f64 accumulation; rounding points match
//! `attention_q8_0_fa2_gqa.gfx1201.hip`)
//!
//! 1. reference: f16 Q/K/V/P — K/V plane values `rne_f16(sf·code)` (:252,:300),
//!    Q `f32→f16` pre-convert (F4b), P `f16(exp)` (:498), scale `1/sqrt(d)`.
//! 2. stage-a: K/V as e4m3 codes with f32 block scales applied post-dot
//!    (per §2). NOT exactly lossless: e4m3 holds exact integers only to
//!    ±14, so codes above that round to the grid (half-step 1/2/4 by
//!    binade) — the screen models the real rounding. Bit-identity vs (1)
//!    is CHECKED and reported; it is expected to FAIL, and the report
//!    quantifies by how much.
//! 4. stage-c: + P e4m3 stored as `×64` with 1/64 folded at the end (×256 also
//!    run and reported; the kill criterion is evaluated at ×64 per §4).
//!
//! e4m3 = the plan's format: bias-7, max 448 (0x7E), min normal 2⁻⁶ (0x08),
//! submin 2⁻⁹, NaN = {0x7F, 0xFF}. RNE everywhere. Note: -0.0 (0x80) can only
//! arise from RNE of a tiny negative; the report counts Q-row occurrences.
//!
//! Metrics per stage vs (1) on O: max-abs err and tail-1% = mean |err| over
//! entries with |O_ref| ≥ 99th-pct(|O_ref|) (the repo's tail discipline per
//! `tools/quant-design/README.md`: rank on the tail, never on RMS). Plus the
//! fraction of valid (unmasked) P entries encoding to zero at ×1 / ×64 / ×256.
//! Kill (§4): stage-c tail > 2× stage-b tail, or P underflow > 0.1% at ×64.
//!
//! Usage (needs GPU + the trunk; ordinal-N box convention):
//!   HOME=/home/kaden/.hipfire-homes/ab2 ROCR_VISIBLE_DEVICES=2 \
//!   HIPFIRE_KERNEL_CACHE=$HOME/.hipfire_kernels/gfx1201 \
//!   cargo run -p saddle-lab --example fa2_fp8_screen \
//!     -- --model /home/kaden/.hipfire/models/qwen3.8-27b.mq4-xt [--tokens 4224] \
//!        [--layer L] [--qstride 4] [--kernel-out PATH]
//!
//! Env init mirrors `dump_qwen35_hidden_states.rs` (graphs off, no prompt
//! normalization) so kernel paths match the floor measurements.

#[cfg(not(all(feature = "deltanet", feature = "arch-qwen35")))]
fn main() {
    eprintln!("build with --features deltanet,arch-qwen35");
}

#[cfg(all(feature = "deltanet", feature = "arch-qwen35"))]
fn main() {
    use hipfire_arch_qwen35::qwen35::{
        self, DflashFusionCtx, DeltaNetState, LayerType, LayerWeights, PrefillBatchScratch,
        Qwen35Scratch,
    };
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::llama::KvCache;
    use std::path::PathBuf;

    // ---------------- args ----------------
    let argv: Vec<String> = std::env::args().collect();
    let mut model = PathBuf::from("/srv/hw-gate/models/qwen3.8-27b.mq4-xt");
    let mut n_tokens: usize = 4224; // 11 full 384-row chunks on gfx1201
    let mut layer_arg: Option<usize> = None;
    let mut kernel_out: Option<PathBuf> = None;
    let mut qstride: usize = 4; // sample every 4th final-chunk query for the CPU math
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model" => {
                model = PathBuf::from(&argv[i + 1]);
                i += 2;
            }
            "--tokens" => {
                n_tokens = argv[i + 1].parse().expect("--tokens");
                i += 2;
            }
            "--layer" => {
                layer_arg = Some(argv[i + 1].parse().expect("--layer"));
                i += 2;
            }
            "--qstride" => {
                qstride = argv[i + 1].parse().expect("--qstride");
                i += 2;
            }
            "--kernel-out" => {
                kernel_out = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "-h" | "--help" => {
                eprintln!(
                    "Usage: fa2_fp8_screen [--model path] [--tokens N] [--layer L] [--qstride S] [--kernel-out PATH]"
                );
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(1);
            }
        }
    }

    // SAFETY: single-threaded init phase; no other threads observing env.
    unsafe {
        std::env::set_var("HIPFIRE_NORMALIZE_PROMPT", "0");
        std::env::set_var("HIPFIRE_GRAPH", "0");
        std::env::set_var("HIPFIRE_KV_MODE", "q8");
    }
    // Codec self-test: catches exp-bias/grid bugs before the GPU run.
    // Grid truth (bias-7 e4m3): exact integers to ±14, then half-step
    // 1 / 2 / 4 codes by binade. The plan's "(e4m3 holds all integers
    // to 256)" is wrong past 14 — noted, and stage-a below models the
    // real rounding via kce/vce.
    for i in -127..=127i32 {
        let v = i as f64;
        let err = (e4m3_decode(e4m3_encode(v)) - v).abs();
        // Half-step bound of the containing binade: exact to ±14, then
        // 1 / 2 / 4 codes. (The plan's "all integers to 256" is wrong
        // past 14; stage-a models the real rounding via kce/vce.)
        let bound = if v.abs() <= 14.0 {
            0.0
        } else if v.abs() < 32.0 {
            1.0
        } else if v.abs() < 64.0 {
            2.0
        } else {
            4.0
        };
        assert!(err <= bound, "int8 {i} e4m3 err {err}");
    }
    assert_eq!(e4m3_encode(1.0), 0x38);
    assert_eq!(e4m3_decode(0x38), 1.0);
    assert_eq!(e4m3_encode(448.0), 0x7E);
    assert_eq!(e4m3_decode(0x7E), 448.0);
    assert_eq!(e4m3_encode(0.015625), 0x08);
    assert_eq!(e4m3_encode(0.0), 0x00);
    assert_eq!(e4m3_encode(-0.0), 0x80);
    assert_eq!(e4m3_encode(127.0), 0x70); // nearest grid to 127 is 128
    assert_eq!(e4m3_encode(120.0), 0x6F);
    assert_eq!(f32_to_f16_bits(1.0), 0x3C00);
    assert_eq!(f32_to_f16_bits(-0.0), 0x8000);
    eprintln!("codec self-test OK (int8 exact to ±120; 1.0=0x38, 448=0x7E)");

    // ---------------- load model ----------------
    let mut hfq = HfqFile::open(&model).expect("open model");
    let config = qwen35::config_from_hfq(&hfq).expect("config");
    eprintln!(
        "model={} dim={} layers={} heads={} kv_heads={} head_dim={} norm_eps={}",
        model.display(),
        config.dim,
        config.n_layers,
        config.n_heads,
        config.n_kv_heads,
        config.head_dim,
        config.norm_eps
    );
    let fa_layers: Vec<usize> = config
        .layer_types
        .iter()
        .enumerate()
        .filter(|(_, t)| **t == LayerType::FullAttention)
        .map(|(l, _)| l)
        .collect();
    eprintln!("full-attention layers: {fa_layers:?}");
    assert!(
        !fa_layers.is_empty(),
        "no FullAttention layers in config.layer_types"
    );
    // Middle-most dense-FullAttn layer (weights must be the dense variant the
    // FA2 kernel serves; FullAttnMoe layers take a different prefill body).
    let want = match layer_arg {
        Some(l) => l,
        None => fa_layers[fa_layers.len() / 2],
    };
    assert!(
        fa_layers.contains(&want),
        "layer {want} is not FullAttention per config"
    );

    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    eprintln!("GPU: {}", gpu.arch);
    let weights = {
        let mut src = qwen35::HfqSource::new(&mut hfq, &config);
        let layout = qwen35::Layout::single(config.n_layers);
        qwen35::load_weights(&mut src, std::slice::from_mut(&mut gpu), &layout)
    }
    .expect("weights");
    match &weights.layers[want] {
        LayerWeights::FullAttn(_) => {}
        other => {
            let tag = match other {
                LayerWeights::DeltaNet(_) => "DeltaNet",
                LayerWeights::DeltaNetMoe(_) => "DeltaNetMoe",
                LayerWeights::FullAttnMoe(_) => "FullAttnMoe",
                LayerWeights::FullAttn(_) => unreachable!(),
            };
            eprintln!("layer {want} weights are {tag}, need dense FullAttn; abort");
            std::process::exit(2);
        }
    }
    eprintln!("screen layer L={want} (dense FullAttn)");

    let (nh, nkv, hd) = (config.n_heads, config.n_kv_heads, config.head_dim);
    assert_eq!((nh, nkv, hd), (24, 4, 256), "FA2 gfx1201 needs H24/KV4/D256");
    let q_dim = nh * hd;
    let kv_dim = nkv * hd;
    let row_stride = nkv * (hd / 32) * 34; // kernel's row_stride (:112)
    let blks_per_head = hd / 32;
    let scale_attn = 1.0f64 / (hd as f64).sqrt(); // launcher: 1/sqrt(head_dim)

    // ---------------- prefill (truncated at L) ----------------
    // Deterministic synthetic prompt, same convention as
    // profile_prefill_qwen35.rs: token IDs 0..N through the real embedding.
    let tokens: Vec<u32> = (0..n_tokens as u32).collect();
    let kv_max = n_tokens + 16;
    let mut kv_cache =
        KvCache::new_gpu_q8(&mut gpu, config.n_layers, nkv, hd, kv_max).expect("kv cache q8");
    let mut dn_state = DeltaNetState::new(&mut gpu, &config).expect("dn state");
    dn_state.reset(&mut gpu).expect("dn reset");
    let scratch = Qwen35Scratch::new(&mut gpu, &config, 128).expect("scratch");
    // Caller-owned PBS so the final chunk's fa_* buffers survive the call.
    // max_batch = 384 = the gfx1201 arch chunk: N must be a multiple of it so
    // every chunk (including the last) is full.
    const CHUNK: usize = 384;
    assert_eq!(
        n_tokens % CHUNK,
        0,
        "N={n_tokens} must be a multiple of {CHUNK}"
    );
    let pbs = PrefillBatchScratch::new(&mut gpu, &config, CHUNK).expect("pbs");
    let t0 = std::time::Instant::now();
    qwen35::forward_prefill_batch_with_pbs_opts(
        &mut gpu,
        &weights,
        &config,
        &tokens,
        0,
        &mut kv_cache,
        &mut dn_state,
        &scratch,
        None, // hidden_rb
        None, // per_token_hidden_out
        None, // gdn_tape
        None, // tree_verify
        Some(&pbs),
        None, // mask_override
        Some(want + 1), // max_layer: exclusive count → runs layers 0..=L
        false, // needs_last_token_logits
        DflashFusionCtx::Off,
    )
    .expect("truncated prefill");
    eprintln!(
        "truncated prefill (layers 0..={want}) done in {:.1}s",
        t0.elapsed().as_secs_f64()
    );

    let pos_bytes = dtoh_bytes(&gpu, &pbs.positions.buf, CHUNK * 4, "positions");
    let pos: Vec<i32> = pos_bytes
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert_eq!(pos.len(), CHUNK);
    for r in 0..CHUNK {
        assert_eq!(
            pos[r] as usize,
            n_tokens - CHUNK + r,
            "positions[{r}]={}: chunking shifted (arch max_batch != {CHUNK}?); abort",
            pos[r]
        );
    }
    // ---------------- downloads ----------------
    let q_f32 = gpu.download_f32(&pbs.fa_q_batch).expect("fa_q");
    let k_f32 = gpu.download_f32(&pbs.fa_k_batch).expect("fa_k");
    let v_f32 = gpu.download_f32(&pbs.fa_v_batch).expect("fa_v");
    let o_eng = gpu.download_f32(&pbs.fa_attn_out_batch).expect("fa_attn_out");
    // Post-gate attention output (engine applies sigmoid(gate)·attn in place
    // at prefill.rs:6667); gate logits needed to replicate it on CPU.
    let gate_f32 = gpu.download_f32(&pbs.fa_gate_batch).expect("fa_gate");
    assert_eq!(gate_f32.len(), CHUNK * q_dim, "fa_gate shape");
    assert_eq!(o_eng.len(), CHUNK * q_dim, "fa_attn_out shape");
    assert_eq!(q_f32.len(), CHUNK * q_dim, "fa_q shape");
    assert_eq!(k_f32.len(), CHUNK * kv_dim, "fa_k shape");
    assert_eq!(v_f32.len(), CHUNK * kv_dim, "fa_v shape");
    for (name, v) in [
        ("Q", &q_f32),
        ("K", &k_f32),
        ("V", &v_f32),
        ("Oeng", &o_eng),
        ("gate", &gate_f32),
    ] {
        assert!(
            v.iter().all(|x| x.is_finite()),
            "non-finite in downloaded {name}"
        );
    }
    let k_raw = dtoh_bytes(&gpu, &kv_cache.k_gpu[want].buf, n_tokens * row_stride, "k_cache");
    let v_raw = dtoh_bytes(&gpu, &kv_cache.v_gpu[want].buf, n_tokens * row_stride, "v_cache");
    eprintln!(
        "downloaded: Q/K/V/Oeng final-chunk rows + {n_tokens}-pos q8 K/V cache ({row_stride} B/pos)"
    );
    // ---------------- kernel differential (real-data oracle) ----------------
    // Runs the flag-resolved FA2 body kernel on the tapped final chunk
    // (Q/K/V/positions already in FA2 layout on-device) and writes pre-gate
    // O + gate for offline comparison across two flag-resolved processes.
    if let Some(kpath) = &kernel_out {
        eprintln!(
            "kernel differential: fp8 flag on: {}",
            gpu.flags.gfx12_fa2_fp8_enabled()
        );
        let d_out = gpu
            .zeros(&[CHUNK * q_dim], rdna_compute::DType::F32)
            .expect("fa2 diff out");
        gpu.attention_q8_0_fa2_gqa_gfx1201(
            &pbs.fa_q_batch,
            &kv_cache.k_gpu[want],
            &kv_cache.v_gpu[want],
            &d_out,
            &pbs.positions,
            nh,
            nkv,
            hd,
            n_tokens,
            CHUNK,
        )
        .unwrap_or_else(|e| panic!("fa2 diff launch: {e:?}"));
        let o_kern = gpu.download_f32(&d_out).expect("fa2 diff O");
        assert_eq!(o_kern.len(), CHUNK * q_dim, "fa2 diff O shape");
        assert!(o_kern.iter().all(|x| x.is_finite()), "non-finite in fa2 diff O");
        let ob: &[u8] = unsafe {
            std::slice::from_raw_parts(o_kern.as_ptr() as *const u8, o_kern.len() * 4)
        };
        let gb: &[u8] = unsafe {
            std::slice::from_raw_parts(gate_f32.as_ptr() as *const u8, gate_f32.len() * 4)
        };
        let o_path = format!("{}.O.f32", kpath.display());
        let g_path = format!("{}.gate.f32", kpath.display());
        let m_path = format!("{}.meta.txt", kpath.display());
        std::fs::write(&o_path, ob).expect("write diff O");
        std::fs::write(&g_path, gb).expect("write diff gate");
        std::fs::write(
            &m_path,
            format!(
                "layer={want} tokens={n_tokens} chunk={CHUNK} nh={nh} nkv={nkv} hd={hd} fp8={}\n",
                gpu.flags.gfx12_fa2_fp8_enabled()
            ),
        )
        .expect("write diff meta");
        eprintln!("kernel differential wrote {o_path} + gate");
        // G2 input provenance (disposable): dump tapped kernel inputs
        // for cross-arm hash comparison.
        std::fs::write(
            format!("{}.q.f32", kpath.display()),
            unsafe {
                std::slice::from_raw_parts(
                    q_f32.as_ptr() as *const u8,
                    q_f32.len() * 4,
                )
            },
        )
        .expect("write diff Q");
        std::fs::write(format!("{}.kraw.bin", kpath.display()), &k_raw)
            .expect("write diff Kraw");
        std::fs::write(format!("{}.vraw.bin", kpath.display()), &v_raw)
            .expect("write diff Vraw");
        std::fs::write(format!("{}.pos.i32", kpath.display()), &pos_bytes)
            .expect("write diff pos");
        eprintln!("kernel differential wrote input dumps for hashing");
    }
    fn dtoh_bytes(
        gpu: &rdna_compute::Gpu,
        buf: &hip_bridge::DeviceBuffer,
        n: usize,
        tag: &str,
    ) -> Vec<u8> {
        let mut out = vec![0u8; n];
        gpu.hip
            .memcpy_dtoh(&mut out, buf)
            .unwrap_or_else(|e| panic!("dtoh {tag}: {e:?}"));
        out
    }
    // f32 → f16 bits, round-to-nearest-even (matches the kernel's
    // (_Float16) cast which is RN; subnormals handled, overflow → inf).
    fn f32_to_f16_bits(x: f32) -> u16 {
        assert!(x.is_finite(), "non-finite f32→f16");
        if x == 0.0 {
            return if x.is_sign_negative() { 0x8000 } else { 0 };
        }
        let b = x.to_bits();
        let sign = ((b >> 16) & 0x8000) as u16;
        let exp = ((b >> 23) & 0xFF) as i32;
        let mant = b & 0x007F_FFFF;
        if exp == 0 {
            return sign; // f32 subnormal → f16 zero
        }
        let e16 = exp - 112;
        if e16 >= 31 {
            return sign | 0x7C00; // overflow → inf (data never hits this)
        }
        if e16 <= 0 {
            // f16 subnormal: m' = rne(mant|hidden × 2^(e16-14)), 24-bit m.
            let m = mant | 0x0080_0000;
            let shift = 14 - e16; // >= 14
            if shift >= 25 {
                return sign;
            }
            let half = 1u32 << (shift - 1);
            let mask = (half << 1).wrapping_sub(1);
            let rem = m & mask;
            let mut q = m >> shift;
            if rem > half || (rem == half && (q & 1) != 0) {
                q += 1;
            }
            if q >= 1024 {
                return sign | 0x0400; // carry to normal min
            }
            return sign | (q as u16);
        }
        let keep = mant & !0x1FFF;
        let rem = mant & 0x1FFF;
        let mut om = keep;
        let mut oe = e16;
        if rem > 0x1000 || (rem == 0x1000 && (keep & 0x2000) != 0) {
            om = keep + 0x2000;
            if om > 0x007F_FFFF {
                om = 0;
                oe += 1;
            }
        }
        if oe >= 31 {
            return sign | 0x7C00;
        }
        sign | ((oe as u16) << 10) | ((om >> 13) as u16)
    }
    fn f16_to_f32(b: u16) -> f32 {
        let sign = ((b as u32) & 0x8000) << 16;
        let exp = ((b >> 10) & 0x1F) as u32;
        let mant = (b & 0x3FF) as u32;
        let bits = if exp == 0 {
            if mant == 0 {
                sign
            } else {
                // subnormal → normalize
                let mut e = 127 - 14;
                let mut m = mant;
                while (m & 0x400) == 0 {
                    m <<= 1;
                    e -= 1;
                }
                m &= 0x3FF;
                sign | (e << 23) | (m << 13)
            }
        } else if exp == 31 {
            sign | (0xFF << 23) | (mant << 13)
        } else {
            sign | ((exp + 112) << 23) | (mant << 13)
        };
        f32::from_bits(bits)
    }
    fn f16_to_f64(b: u16) -> f64 {
        f16_to_f32(b) as f64
    }
    // rne-to-f16 of an f64 value, returned as f64. K/V products here are
    // f32-exact so the f64→f32 narrowing is bit-exact for them; softmax
    // probs narrow with one extra rounding (noted; absorbed by check-B tol).
    fn rne_f16_f64(x: f64) -> f64 {
        f16_to_f64(f32_to_f16_bits(x as f32))
    }
    // Round-half-even on f64 (exact-half detection is exact in binary).
    fn rne(v: f64) -> f64 {
        let lo = v.floor();
        let f = v - lo;
        if f < 0.5 {
            lo
        } else if f > 0.5 {
            lo + 1.0
        } else if (lo as i64) % 2 == 0 {
            lo
        } else {
            lo + 1.0
        }
    }
    // Plan's e4m3: bias 7, max 448 (0x7E), min-normal 2^-6, submin 2^-9,
    // NaN = e==15 && m==7. RNE, finite saturation (data never saturates:
    // Q scaled by absmax/448, P·256 ≤ 256).
    // Codec self-test lives at the top of main (runs before the GPU work).
    fn e4m3_encode(x: f64) -> u8 {
        if x.is_nan() {
            return 0x7F;
        }
        let sign = if x < 0.0 || (x == 0.0 && x.is_sign_negative()) {
            0x80
        } else {
            0
        };
        let ax = x.abs();
        if ax == 0.0 {
            return sign;
        }
        if ax < 0.0009765625 {
            // below 2^-10 → RNE gives 0 (exact tie at 2^-10 → even → 0).
            return sign;
        }
        if ax < 0.015625 {
            // subnormal: k = rne(ax·512) ∈ 1..=8
            let k = rne(ax * 512.0) as u32;
            if k >= 8 {
                return sign | 0x08;
            }
            return sign | (k as u8);
        }
        // normal: e = floor(log2(ax)) ∈ [-6, 8]; saturate beyond top.
        let mut e = ax.log2().floor() as i32;
        if e > 8 {
            return sign | 0x7E;
        }
        if e < -6 {
            e = -6;
        }
        let step = 2f64.powi(e - 3);
        let k = rne(ax / step) as i32;
        let mut ee = e;
        let mut kk = k;
        if kk >= 16 {
            kk = 8;
            ee += 1;
        }
        if ee > 8 {
            return sign | 0x7E;
        }
        if kk > 14 && ee == 8 {
            return sign | 0x7E; // top-binade saturation (m=7@e=15 is NaN)
        }
        sign | (((ee + 7) as u8) << 3) | ((kk - 8) as u8)
    }
    fn e4m3_decode(b: u8) -> f64 {
        let s = if b & 0x80 != 0 { -1.0 } else { 1.0 };
        let e = ((b >> 3) & 0xF) as i32;
        let m = (b & 7) as i32;
        if e == 0 {
            s * (m as f64) * 2f64.powi(-9)
        } else {
            // e==15,m==7 would be NaN — never produced by the saturating
            // encode above; the formula is applied uniformly.
            s * ((8 + m) as f64) * 2f64.powi(e - 7 - 3)
        }
    }
}
