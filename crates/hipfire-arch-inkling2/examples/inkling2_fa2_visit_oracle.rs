//! Developer oracle (not a gate, not a route): runs the gfx1201 split-KV FA2
//! decode attention (`Gpu::inkling2_attn_decode_fa2`) on the REAL operands of
//! one attention visit and compares it to the scalar kernel's output for the
//! same visit, captured through the public `forward_token_gpu_capture` seam.
//! No forward code changes: the scalar path still produces every token; FA2
//! only re-reads that visit's live KV cache, captured Q and biased scores.
//!
//! ```text
//! HIPFIRE_GRAPH=0 INKLING2_MQ4=/path/inkling2-mini-mq4.hfq \
//!   cargo run --release -p hipfire-arch-inkling2 --example inkling2_fa2_visit_oracle -- \
//!   <n_tokens> <out.jsonl>
//! ```
//! Token stream is synthetic (`(i*7919+13) % 199_000 + 100`), so this is an
//! operator oracle on model-shaped operands, not a quality (KLD) result.
use hipfire_arch_inkling2::arch_model::{Inkling2Bundle, Inkling2State};
use hipfire_arch_inkling2::config::Inkling2Config;
use hipfire_arch_inkling2::gpu::{swa_gather_map, upload};
use hipfire_arch_inkling2::weights::Inkling2Weights;
use hipfire_arch_inkling2::{forward_token_gpu, forward_token_gpu_capture};
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::inkling2_attention::inkling2_attn_fa2_splits;
use rdna_compute::{DType, Gpu};
use std::io::Write;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n_tokens: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1200);
    let out_path = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "fa2-visit-oracle.jsonl".into());
    assert_eq!(
        std::env::var("HIPFIRE_GRAPH").as_deref(),
        Ok("0"),
        "set HIPFIRE_GRAPH=0"
    );
    let art = std::env::var("INKLING2_MQ4").expect("set INKLING2_MQ4");
    let hfq = HfqFile::open(std::path::Path::new(&art)).expect("open artifact");
    let cfg = Inkling2Config::from_metadata_json(&hfq.metadata_json).expect("config");
    let weights = Inkling2Weights::load(&hfq, &cfg).expect("weights");
    let mut gpu = Gpu::init().expect("gpu init");
    let max_seq = n_tokens + 64;
    let (gw, mut gs) = upload(&weights, &cfg, &mut gpu, max_seq).expect("upload");
    // The reference is the scalar kernel: pin the forward to it (FA2 is the
    // default on gfx1201) so the captured `attn_heads` are scalar output.
    gs.attn_fa2 = false;
    let mut bundle = Inkling2Bundle {
        config: cfg,
        weights,
        state: Inkling2State { seq_pos: 0 },
        eos_tok: 200006,
        gpu_weights: Some(gw),
        gpu_state: Some(gs),
    };
    let nh = 24usize;
    let hd = 128usize;
    // Positions: short window, tile/partition boundaries, the SWA ring wrap.
    let mut caps: Vec<usize> = [
        0usize, 1, 63, 64, 127, 191, 1022, 1023, 1024, 1025, 1500, 2047, 2111,
    ]
    .into_iter()
    .filter(|&p| p < n_tokens)
    .collect();
    caps.push(n_tokens - 1);
    caps.dedup();
    let mut f = std::fs::File::create(&out_path).expect("out");
    let mut visit_rr = 0usize;
    for pos in 0..n_tokens {
        let tok = ((pos * 7919 + 13) % 199_000 + 100) as u32;
        if !caps.contains(&pos) {
            forward_token_gpu(&mut bundle, &mut gpu, tok, pos).expect("forward");
            continue;
        }
        // One visit per captured position, rotating by 7 so local and global
        // visits both appear across the sweep.
        let visit = visit_rr % 48;
        visit_rr += 7;
        let cap =
            forward_token_gpu_capture(&mut bundle, &mut gpu, tok, pos, visit).expect("capture");
        let gs = bundle.gpu_state.as_ref().unwrap();
        let rows = gs.visit_rows[visit];
        let (slots, _dists) =
            swa_gather_map(&gs.visit_pos_tags[visit], pos, rows).expect("gather map");
        let n_kv = slots.len();
        let q = &cap["qc"];
        let bias = &cap["rel_bias"];
        let want = &cap["attn_heads"];
        assert_eq!(q.len(), nh * hd);
        assert_eq!(bias.len(), nh * n_kv);
        let splits = inkling2_attn_fa2_splits(n_kv);
        let tq = gpu.upload_f32(q, &[nh, hd]).expect("q");
        let tb = gpu.upload_f32(bias, &[nh, n_kv]).expect("bias");
        let sb: Vec<u8> = slots.iter().flat_map(|s| s.to_le_bytes()).collect();
        let ts = gpu.upload_raw(&sb, &[sb.len()]).expect("slots");
        let tl = gpu.alloc_tensor(&[splits, nh], DType::F32).expect("lse");
        let to = gpu
            .alloc_tensor(&[splits, nh, hd], DType::F32)
            .expect("oacc");
        let tout = gpu.alloc_tensor(&[nh, hd], DType::F32).expect("out");
        gpu.inkling2_attn_decode_fa2(
            &tq,
            &gs.k_cache[visit],
            &gs.v_cache[visit],
            &tb,
            &ts,
            &tl,
            &to,
            &tout,
            n_kv,
            splits,
        )
        .expect("fa2");
        let got = gpu.download_f32(&tout).expect("download");
        let (mut maxd, mut maxw) = (0f64, 0f64);
        for (a, b) in got.iter().zip(want) {
            maxd = maxd.max((*a as f64 - *b as f64).abs());
            maxw = maxw.max((*b as f64).abs());
        }
        let bitwise = got
            .iter()
            .zip(want)
            .all(|(a, b)| a.to_bits() == b.to_bits());
        writeln!(
            f,
            "{{\"pos\":{pos},\"visit\":{visit},\"rows\":{rows},\"n_kv\":{n_kv},\"splits\":{splits},\"max_abs_diff\":{maxd:e},\"max_abs_scalar\":{maxw:e},\"bitwise\":{bitwise},\"finite\":{}}}",
            got.iter().all(|x| x.is_finite())
        )
        .unwrap();
        eprintln!("pos {pos} visit {visit} n_kv {n_kv} S{splits}: max|d| {maxd:.3e} (|ref|max {maxw:.3e})");
        for t in [tq, tb, ts, tl, to, tout] {
            gpu.free_tensor(t).expect("free");
        }
    }
    if let Some(gw) = bundle.gpu_weights.take() {
        gw.free_gpu(&mut gpu);
    }
    if let Some(gs) = bundle.gpu_state.take() {
        gs.free_gpu(&mut gpu);
    }
}
