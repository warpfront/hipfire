// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Frozen one-row decode step (`decode_step`: stage, body, finish) against the
//! stock eager decode (`forward_decode_gpu`) on the typed-XTS artifact, gfx1201.
//!
//! Every compared step starts both arms from the **same** state: the pre-step
//! conv histories, the K/V rows at the write slot of every visit and the host
//! cursors are saved, the stock arm runs, its post-state is recorded, the saved
//! state is restored, and the frozen arm runs. Compared bit for bit after each
//! step:
//!
//! * the full unpadded-vocab logits (`to_bits`);
//! * all eight K and eight V F16 planes of all 48 visits at the write slot;
//! * the F32 `[3, 6656]` conv history of every visit;
//! * every `visit_cap`, every `visit_pos_tags`, `seq_pos`;
//! * (traced steps) all 48 `visit_out` rows and the frozen pass banks
//!   (prelude / previous / current / stream rows) against the stock trace;
//! * (witness steps) a hash of every live K/V row except the write slot, before
//!   and after, in both arms (unchanged-witness / no stray write).
//!
//! Scratch the two arms must not depend on is poisoned with a distinct,
//! changing pattern before every arm (the FA2 partials, the whole decode
//! arena slab).
//!
//! ```sh
//! INKLING2_MQ4=.../inkling2-mini-xts-c.hfq HIPFIRE_GRAPH=0 \
//!   cargo test --release -p hipfire-arch-inkling2 --test decode_step_hw -- \
//!   --ignored --nocapture --test-threads=1
//! ```

use hipfire_arch_inkling2::arch_model::{Inkling2Bundle, Inkling2State};
use hipfire_arch_inkling2::config::Inkling2Config;
use hipfire_arch_inkling2::decode_step::Slot;
use hipfire_arch_inkling2::gpu::upload;
use hipfire_arch_inkling2::weights::Inkling2Weights;
use hipfire_arch_inkling2::{
    enqueue_decode_body_trace, ensure_decode_arena, finish_decode_step, forward_decode_frozen,
    forward_decode_gpu, forward_prefill_gpu, forward_prefill_gpu_with, plan_decode_step,
    prefill_chunk_rows, stage_decode_input, PrefillLogits, PrefillMode, PrefillRoutes,
};
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::{DType, Gpu, GpuTensor};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::Write;

fn tokens(n: usize) -> Vec<u32> {
    // Deterministic ids inside the vocabulary (LCG); content is irrelevant to parity.
    let mut s = 0x1234_5678_9abc_def0u64;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) % 199_000) as u32 + 100
        })
        .collect()
}

fn load(gpu: &mut Gpu, max_seq: usize) -> Inkling2Bundle {
    assert_eq!(
        std::env::var("HIPFIRE_GRAPH").as_deref(),
        Ok("0"),
        "the oracle requires HIPFIRE_GRAPH=0"
    );
    let art = std::env::var("INKLING2_MQ4").expect("set INKLING2_MQ4 to the typed-XTS .hfq");
    let hfq = HfqFile::open(std::path::Path::new(&art)).expect("open artifact");
    let cfg = Inkling2Config::from_metadata_json(&hfq.metadata_json).expect("config");
    let weights = Inkling2Weights::load(&hfq, &cfg).expect("weights");
    let (gw, mut gs) = upload(&weights, &cfg, gpu, max_seq).expect("upload");
    if let Ok(r) = std::env::var("INKLING2_ROUTES") {
        gs.prefill_routes = PrefillRoutes::parse(&r).expect("INKLING2_ROUTES");
    }
    Inkling2Bundle {
        config: cfg,
        weights,
        state: Inkling2State { seq_pos: 0 },
        eos_tok: 200006,
        gpu_weights: Some(gw),
        gpu_state: Some(gs),
    }
}

fn reset(b: &mut Inkling2Bundle, gpu: &mut Gpu) {
    b.state.seq_pos = 0;
    b.gpu_state
        .as_mut()
        .unwrap()
        .reset_session_state(gpu)
        .expect("reset");
}

fn prefill(b: &mut Inkling2Bundle, gpu: &mut Gpu, toks: &[u32]) {
    if !toks.is_empty() {
        forward_prefill_gpu(b, gpu, toks, 0, PrefillLogits::None).expect("prefill");
    }
}

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn f32_bytes(v: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding; the slice is plain bytes.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

// ---------------------------------------------------------------------------
// State save / restore / snapshot
// ---------------------------------------------------------------------------

fn slot_view(plane: &GpuTensor, slot: usize, hd: usize) -> GpuTensor {
    GpuTensor {
        buf: plane.buf.byte_view(slot * hd * 2, hd * 2),
        shape: vec![hd * 2],
        dtype: DType::Raw,
    }
}

/// Everything one step can change on device or host, as of before it.
struct Pre {
    pos: usize,
    conv: Vec<Vec<u8>>,
    /// `[visit][plane 0..16]` slot bytes (K planes then V planes).
    slot_rows: Vec<Vec<Vec<u8>>>,
    tag_slot: Vec<u32>,
    caps: Vec<usize>,
    seq_pos: usize,
}

fn planes(b: &Inkling2Bundle, v: usize) -> Vec<&GpuTensor> {
    let gs = b.gpu_state.as_ref().unwrap();
    gs.k_cache[v].iter().chain(gs.v_cache[v].iter()).collect()
}

fn save_pre(b: &Inkling2Bundle, gpu: &Gpu, pos: usize) -> Pre {
    let gs = b.gpu_state.as_ref().unwrap();
    let hd = b.config.head_dim;
    let n = gs.k_cache.len();
    let mut p = Pre {
        pos,
        conv: Vec::with_capacity(n),
        slot_rows: Vec::with_capacity(n),
        tag_slot: Vec::with_capacity(n),
        caps: gs.visit_cap.clone(),
        seq_pos: gs.seq_pos,
    };
    for v in 0..n {
        p.conv
            .push(gpu.download_raw_bytes(&gs.conv[v]).expect("conv"));
        let slot = pos % gs.visit_rows[v];
        p.slot_rows.push(
            planes(b, v)
                .into_iter()
                .map(|pl| {
                    gpu.download_raw_bytes(&slot_view(pl, slot, hd))
                        .expect("slot row")
                })
                .collect(),
        );
        p.tag_slot.push(gs.visit_pos_tags[v][slot]);
    }
    p
}

fn restore(b: &mut Inkling2Bundle, gpu: &Gpu, p: &Pre) {
    let hd = b.config.head_dim;
    let n = p.conv.len();
    for v in 0..n {
        let gs = b.gpu_state.as_ref().unwrap();
        gpu.memcpy_htod_auto(&gs.conv[v].buf, &p.conv[v])
            .expect("conv restore");
        let slot = p.pos % gs.visit_rows[v];
        for (pl, bytes) in planes(b, v).into_iter().zip(&p.slot_rows[v]) {
            gpu.memcpy_htod_auto(&slot_view(pl, slot, hd).buf, bytes)
                .expect("slot restore");
        }
    }
    let gs = b.gpu_state.as_mut().unwrap();
    for v in 0..n {
        let slot = p.pos % gs.visit_rows[v];
        gs.visit_pos_tags[v][slot] = p.tag_slot[v];
    }
    gs.visit_cap = p.caps.clone();
    gs.seq_pos = p.seq_pos;
}

/// Post-step state (everything a step may have written).
struct Post {
    logits: Vec<f32>,
    conv: Vec<Vec<u8>>,
    slot_rows: Vec<Vec<Vec<u8>>>,
    tags: Vec<Vec<u32>>,
    caps: Vec<usize>,
    seq_pos: usize,
}

fn post_state(b: &Inkling2Bundle, gpu: &Gpu, pos: usize, logits: Vec<f32>) -> Post {
    let p = save_pre(b, gpu, pos);
    let gs = b.gpu_state.as_ref().unwrap();
    Post {
        logits,
        conv: p.conv,
        slot_rows: p.slot_rows,
        tags: gs.visit_pos_tags.clone(),
        caps: gs.visit_cap.clone(),
        seq_pos: gs.seq_pos,
    }
}

fn first_diff_u8(a: &[u8], b: &[u8]) -> Option<usize> {
    if a.len() != b.len() {
        return Some(a.len().min(b.len()));
    }
    a.iter().zip(b).position(|(x, y)| x != y)
}

/// First differing site of `want` (stock) vs `got` (frozen), or `None`.
fn compare_post(want: &Post, got: &Post) -> Option<String> {
    if want.logits.len() != got.logits.len() {
        return Some(format!(
            "logits length {} != {}",
            want.logits.len(),
            got.logits.len()
        ));
    }
    if let Some(i) = want
        .logits
        .iter()
        .zip(&got.logits)
        .position(|(a, b)| a.to_bits() != b.to_bits())
    {
        return Some(format!(
            "logits[{i}]: stock {:#010x} ({}) vs frozen {:#010x} ({})",
            want.logits[i].to_bits(),
            want.logits[i],
            got.logits[i].to_bits(),
            got.logits[i]
        ));
    }
    for v in 0..want.slot_rows.len() {
        for (pi, (a, b)) in want.slot_rows[v].iter().zip(&got.slot_rows[v]).enumerate() {
            if let Some(o) = first_diff_u8(a, b) {
                return Some(format!(
                    "visit {v} {} plane {} slot byte {o}: stock {:#04x} vs frozen {:#04x}",
                    if pi < 8 { "K" } else { "V" },
                    pi % 8,
                    a[o],
                    b[o]
                ));
            }
        }
        if let Some(o) = first_diff_u8(&want.conv[v], &got.conv[v]) {
            return Some(format!(
                "visit {v} conv byte {o} (elem {}): stock {:#04x} vs frozen {:#04x}",
                o / 4,
                want.conv[v][o],
                got.conv[v][o]
            ));
        }
    }
    for v in 0..want.tags.len() {
        if want.tags[v] != got.tags[v] {
            let i = want.tags[v]
                .iter()
                .zip(&got.tags[v])
                .position(|(a, b)| a != b)
                .unwrap_or(0);
            return Some(format!(
                "visit {v} tag[{i}]: stock {} vs frozen {}",
                want.tags[v][i], got.tags[v][i]
            ));
        }
    }
    if want.caps != got.caps {
        return Some(format!("visit_cap {:?} vs {:?}", want.caps, got.caps));
    }
    if want.seq_pos != got.seq_pos {
        return Some(format!("seq_pos {} vs {}", want.seq_pos, got.seq_pos));
    }
    None
}

/// Hash of every live K/V row of every visit EXCEPT the write slot of `pos`
/// (rows `[0, min(pos, ring))`): unchanged by a step.
fn witness(b: &Inkling2Bundle, gpu: &Gpu, pos: usize) -> String {
    let gs = b.gpu_state.as_ref().unwrap();
    let hd = b.config.head_dim;
    let mut h = Sha256::new();
    for v in 0..gs.k_cache.len() {
        let ring = gs.visit_rows[v];
        let live = pos.min(ring);
        let slot = pos % ring;
        for pl in planes(b, v) {
            let all = gpu.download_raw_bytes(pl).expect("plane");
            for r in 0..live {
                if r != slot {
                    h.update(&all[r * hd * 2..(r + 1) * hd * 2]);
                }
            }
        }
    }
    format!("{:x}", h.finalize())
}

// ---------------------------------------------------------------------------
// Poison
// ---------------------------------------------------------------------------

fn poison(b: &mut Inkling2Bundle, gpu: &mut Gpu, salt: u32) {
    let gs = b.gpu_state.as_ref().unwrap();
    // Distinct, changing NaN-adjacent payloads: a merge that reads an inactive
    // partial, or a body that reads unwritten arena rows, cannot match.
    gpu.fill_f32(&gs.fa2_lse, f32::from_bits(0x7fc0_0000 | (salt & 0xffff)))
        .expect("poison lse");
    gpu.fill_f32(
        &gs.fa2_oacc,
        f32::from_bits(0xffc0_0000 | ((salt * 7) & 0xffff)),
    )
    .expect("poison oacc");
    if let Some(a) = gs.decode_arena.as_ref() {
        let s = a.slab();
        gpu.memset_stream_ordered(&s.buf, ((salt * 37 + 11) & 0xff) as i32, s.buf.size())
            .expect("poison arena");
    }
}

// ---------------------------------------------------------------------------
// Arms
// ---------------------------------------------------------------------------

fn stock_arm(b: &mut Inkling2Bundle, gpu: &mut Gpu, tok: u32, pos: usize) -> Vec<f32> {
    forward_decode_gpu(b, gpu, tok, pos).expect("stock decode")
}

fn frozen_arm(b: &mut Inkling2Bundle, gpu: &mut Gpu, tok: u32, pos: usize) -> Vec<f32> {
    let mut l = Vec::new();
    forward_decode_frozen(b, gpu, tok, pos, &mut l).expect("frozen decode");
    l
}

/// The stock body with the per-visit `visit_out` trace (evidence entry; same
/// mode/routes/blocking as `forward_decode_gpu`).
fn stock_traced(
    b: &mut Inkling2Bundle,
    gpu: &mut Gpu,
    tok: u32,
    pos: usize,
) -> (Vec<f32>, Vec<Vec<f32>>) {
    let routes = b.gpu_state.as_ref().unwrap().prefill_routes;
    let block_rows = prefill_chunk_rows(b, gpu);
    let mut trace = Vec::new();
    let mut logits = Vec::new();
    forward_prefill_gpu_with(
        b,
        gpu,
        &[tok],
        pos,
        PrefillLogits::Last(&mut logits),
        PrefillMode::BLOCKED,
        routes,
        block_rows,
        Some(&mut trace),
    )
    .expect("stock traced");
    (logits, trace)
}

/// Plant: the host plan and commit say `pos`, the staged device position is
/// `pos - 1` (a stale position in the input record).
fn frozen_stale_pos(b: &mut Inkling2Bundle, gpu: &mut Gpu, tok: u32, pos: usize) -> Vec<f32> {
    ensure_decode_arena(gpu, b).expect("arena");
    let plan = plan_decode_step(b, gpu, pos).expect("plan");
    let cfg = &b.config;
    let gw = b.gpu_weights.as_ref().unwrap();
    let gs = b.gpu_state.as_mut().unwrap();
    let mut arena = gs.decode_arena.take().unwrap();
    let mut logits = Vec::new();
    stage_decode_input(gpu, gw, &mut arena, tok, (pos - 1) as u32).expect("stage");
    hipfire_arch_inkling2::enqueue_decode_body(gpu, cfg, gw, gs, &arena, &plan).expect("enqueue");
    finish_decode_step(gpu, gs, &arena, pos, &mut logits).expect("finish");
    gs.decode_arena = Some(arena);
    logits
}

fn frozen_traced(
    b: &mut Inkling2Bundle,
    gpu: &mut Gpu,
    tok: u32,
    pos: usize,
) -> (Vec<f32>, Vec<Vec<f32>>) {
    ensure_decode_arena(gpu, b).expect("arena");
    let plan = plan_decode_step(b, gpu, pos).expect("plan");
    let cfg = &b.config;
    let gw = b.gpu_weights.as_ref().unwrap();
    let gs = b.gpu_state.as_mut().unwrap();
    let mut arena = gs.decode_arena.take().unwrap();
    let mut logits = Vec::new();
    let mut trace = Vec::new();
    stage_decode_input(gpu, gw, &mut arena, tok, pos as u32).expect("stage");
    enqueue_decode_body_trace(gpu, cfg, gw, gs, &arena, &plan, &mut trace).expect("enqueue");
    finish_decode_step(gpu, gs, &arena, pos, &mut logits).expect("finish");
    gs.decode_arena = Some(arena);
    (logits, trace)
}

/// Expected content of every bank slot after a step, from the stock trace.
fn check_banks(
    b: &Inkling2Bundle,
    gpu: &mut Gpu,
    tok: u32,
    stock_trace: &[Vec<f32>],
) -> Option<String> {
    let gs = b.gpu_state.as_ref().unwrap();
    let arena = gs.decode_arena.as_ref().unwrap();
    let cfg = &b.config;
    let e = cfg.n_embd;
    let gw = b.gpu_weights.as_ref().unwrap();
    // The embedding-norm output through independent test-level launches.
    let mut row = vec![0u8; gw.embed_packed.row_bytes()];
    gw.embed_packed.copy_row(tok, &mut row).unwrap();
    let packed = gpu.upload_raw(&row, &[row.len()]).unwrap();
    let emb = gpu.alloc_tensor(&[1, e], DType::F32).unwrap();
    let x = gpu.alloc_tensor(&[e], DType::F32).unwrap();
    gpu.inkling2_embed_qt44_rows(&packed, &emb, 1, e).unwrap();
    gpu.rmsnorm_batched(&emb, &gw.embed_norm, &x, 1, e, cfg.rms_norm_eps)
        .unwrap();
    let emb0 = gpu.download_f32(&x).unwrap();
    for t in [packed, emb, x] {
        let _ = gpu.free_tensor(t);
    }
    #[derive(Clone, Copy)]
    enum Src {
        Emb,
        Visit(usize),
    }
    let mut val: HashMap<String, Src> = HashMap::new();
    let key = |s: Slot| format!("{s:?}");
    val.insert(key(Slot::X(0)), Src::Emb);
    for d in &arena.schedule().visits {
        for c in &d.before {
            let s = val.get(&key(c.src)).copied().expect("simulated source");
            val.insert(key(c.dst), s);
        }
        if let Some(k) = d.push {
            let s = val.get(&key(Slot::X(d.x_in))).copied().unwrap();
            val.insert(key(Slot::Cur(k)), s);
        }
        val.insert(key(Slot::X(d.x_out)), Src::Visit(d.visit as usize));
    }
    let names: HashMap<String, &GpuTensor> = arena.named_tensors().into_iter().collect();
    let lookup = |s: Slot| -> String {
        match s {
            Slot::X(0) => "x0".into(),
            Slot::X(_) => "x1".into(),
            Slot::PreludeX => "prelude_x".into(),
            Slot::PrevOut => "prev_out".into(),
            Slot::Pre(i) => format!("pre{i}"),
            Slot::Prev(i) => format!("prev{i}"),
            Slot::Cur(i) => format!("cur{i}"),
        }
    };
    let mut keys: Vec<_> = val.keys().cloned().collect();
    keys.sort();
    for k in keys {
        let slot = match k.as_str() {
            "PreludeX" => Slot::PreludeX,
            "PrevOut" => Slot::PrevOut,
            s if s.starts_with("X(") => Slot::X(s[2..3].parse().unwrap()),
            s if s.starts_with("Pre(") => Slot::Pre(s[4..s.len() - 1].parse().unwrap()),
            s if s.starts_with("Prev(") => Slot::Prev(s[5..s.len() - 1].parse().unwrap()),
            s if s.starts_with("Cur(") => Slot::Cur(s[4..s.len() - 1].parse().unwrap()),
            other => panic!("bank key {other}"),
        };
        let t = names[&lookup(slot)];
        let dev = gpu.download_f32(t).unwrap();
        let want: &[f32] = match val[&k] {
            Src::Emb => &emb0,
            Src::Visit(v) => &stock_trace[v],
        };
        if let Some(i) = dev
            .iter()
            .zip(want)
            .position(|(a, b)| a.to_bits() != b.to_bits())
        {
            return Some(format!(
                "pass bank {k} elem {i}: frozen {:#010x} vs stock-derived {:#010x}",
                dev[i].to_bits(),
                want[i].to_bits()
            ));
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Lockstep driver
// ---------------------------------------------------------------------------

struct Opts {
    traced: bool,
    witness: bool,
    /// Positive control: 0 none, 1 flip a conv bit in the frozen arm only,
    /// 2 stage `pos - 1` on the device while the host plan says `pos`.
    plant: u8,
}

struct Receipt(Option<std::fs::File>);

impl Receipt {
    fn new(name: &str) -> Self {
        Receipt(std::env::var("INKLING2_RECEIPT_DIR").ok().map(|d| {
            std::fs::create_dir_all(&d).ok();
            std::fs::File::create(std::path::Path::new(&d).join(name)).expect("receipt file")
        }))
    }
    fn line(&mut self, s: String) {
        if let Some(f) = self.0.as_mut() {
            writeln!(f, "{s}").unwrap();
        }
    }
}

/// One compared step at `pos`: stock arm, restore, frozen arm (and, when
/// traced, the traced arms and the pass banks). Leaves the state post-step.
fn lockstep(
    b: &mut Inkling2Bundle,
    gpu: &mut Gpu,
    rec: &mut Receipt,
    tok: u32,
    pos: usize,
    o: &Opts,
    salt: &mut u32,
) {
    let mut next_salt = || {
        *salt += 1;
        *salt
    };
    ensure_decode_arena(gpu, b).expect("ensure arena");
    let pre = save_pre(b, gpu, pos);
    let w_pre = o.witness.then(|| witness(b, gpu, pos));
    poison(b, gpu, next_salt());
    let ls = stock_arm(b, gpu, tok, pos);
    let ps = post_state(b, gpu, pos, ls);
    let w_s = o.witness.then(|| witness(b, gpu, pos));
    restore(b, gpu, &pre);
    poison(b, gpu, next_salt());
    if o.plant == 1 {
        let gs = b.gpu_state.as_ref().unwrap();
        let mut bytes = gpu.download_raw_bytes(&gs.conv[20]).unwrap();
        // A high exponent bit of a surviving history tap (row 1, channel 5).
        bytes[6656 * 4 + 5 * 4 + 3] ^= 0x20;
        gpu.memcpy_htod_auto(&gs.conv[20].buf, &bytes).unwrap();
    }
    let lf = if o.plant == 2 {
        frozen_stale_pos(b, gpu, tok, pos)
    } else {
        frozen_arm(b, gpu, tok, pos)
    };
    let pf = post_state(b, gpu, pos, lf);
    let w_f = o.witness.then(|| witness(b, gpu, pos));
    if let Some(d) = compare_post(&ps, &pf) {
        panic!("MISMATCH pos {pos}: frozen vs stock first differing site: {d}");
    }
    if o.witness {
        let (wp, ws, wf) = (w_pre.unwrap(), w_s.unwrap(), w_f.unwrap());
        assert_eq!(
            ws, wp,
            "pos {pos}: stock arm changed a non-write-slot KV row"
        );
        assert_eq!(
            wf, wp,
            "pos {pos}: frozen arm changed a non-write-slot KV row"
        );
    }
    let mut note = String::new();
    if o.traced {
        restore(b, gpu, &pre);
        poison(b, gpu, next_salt());
        let (lt, tr_s) = stock_traced(b, gpu, tok, pos);
        let pt = post_state(b, gpu, pos, lt);
        if let Some(d) = compare_post(&ps, &pt) {
            panic!("pos {pos}: stock traced arm differs from stock: {d}");
        }
        restore(b, gpu, &pre);
        poison(b, gpu, next_salt());
        let (lf2, tr_f) = frozen_traced(b, gpu, tok, pos);
        let pf2 = post_state(b, gpu, pos, lf2);
        if let Some(d) = compare_post(&ps, &pf2) {
            panic!("MISMATCH pos {pos}: frozen traced arm differs from stock: {d}");
        }
        assert_eq!(tr_s.len(), tr_f.len());
        for v in 0..tr_s.len() {
            if let Some(i) = tr_s[v]
                .iter()
                .zip(&tr_f[v])
                .position(|(a, b)| a.to_bits() != b.to_bits())
            {
                panic!(
                    "MISMATCH pos {pos}: visit {v} visit_out[{i}]: stock {:#010x} vs frozen {:#010x}",
                    tr_s[v][i].to_bits(),
                    tr_f[v][i].to_bits()
                );
            }
        }
        if let Some(d) = check_banks(b, gpu, tok, &tr_s) {
            panic!("MISMATCH pos {pos}: {d}");
        }
        note = format!(" traced(48 visit_out + banks)");
    }
    let plan_bucket = {
        // The plan of the NEXT position is not valid yet; report the bucket of
        // this step from the key arithmetic: n_full = pos + 1.
        let a = b.gpu_state.as_ref().unwrap().decode_arena.as_ref().unwrap();
        a.buckets()
            .iter()
            .find(|k| k.lo as usize <= pos + 1 && pos < k.hi as usize)
            .map(|k| (k.lo, k.hi))
            .unwrap_or((0, 0))
    };
    rec.line(format!(
        "{{\"pos\":{pos},\"bucket\":[{},{}],\"traced\":{},\"witness\":{},\"logits_sha256\":\"{}\",\"ok\":true}}",
        plan_bucket.0,
        plan_bucket.1,
        o.traced,
        o.witness,
        sha(f32_bytes(&ps.logits))
    ));
    eprintln!(
        "  pos {pos:5} bucket [{},{}] bitwise-equal{note}{}",
        plan_bucket.0,
        plan_bucket.1,
        if o.witness { " witness" } else { "" }
    );
}

fn segment(
    b: &mut Inkling2Bundle,
    gpu: &mut Gpu,
    rec: &mut Receipt,
    toks: &[u32],
    start: usize,
    end_inclusive: usize,
    traced: &dyn Fn(usize) -> bool,
    witness_at: &dyn Fn(usize) -> bool,
    salt: &mut u32,
) {
    reset(b, gpu);
    prefill(b, gpu, &toks[..start]);
    eprintln!("segment {start}..={end_inclusive}");
    for pos in start..=end_inclusive {
        let o = Opts {
            traced: traced(pos),
            witness: witness_at(pos) || pos == start,
            plant: 0,
        };
        lockstep(b, gpu, rec, toks[pos], pos, &o, salt);
    }
}

/// Last legal position + 1 is refused by the plan, and every refusal leaves
/// the state (host and device-visible) untouched.
fn refusals(b: &mut Inkling2Bundle, gpu: &mut Gpu, last_legal: usize) {
    let gs = b.gpu_state.as_ref().unwrap();
    assert_eq!(gs.seq_pos, last_legal + 1);
    let before = (gs.seq_pos, gs.visit_cap.clone(), gs.visit_pos_tags.clone());
    let e = plan_decode_step(b, gpu, last_legal + 1).expect_err("next position past the cap");
    eprintln!("  refused pos {} (cap): {e}", last_legal + 1);
    let e = plan_decode_step(b, gpu, last_legal).expect_err("out-of-order position");
    eprintln!("  refused pos {last_legal} (out-of-order): {e}");
    let gs = b.gpu_state.as_ref().unwrap();
    assert_eq!(
        (gs.seq_pos, &gs.visit_cap, &gs.visit_pos_tags),
        (before.0, &before.1, &before.2)
    );
}

/// Bad cursors and tags are refused before any submission.
fn bad_state_refusals(b: &mut Inkling2Bundle, gpu: &mut Gpu, toks: &[u32]) {
    reset(b, gpu);
    prefill(b, gpu, &toks[..40]);
    plan_decode_step(b, gpu, 40).expect("valid plan");
    let gs = b.gpu_state.as_mut().unwrap();
    // visit_cap mismatch.
    gs.visit_cap[7] += 1;
    let e = plan_decode_step(b, gpu, 40).expect_err("bad visit_cap");
    eprintln!("  refused (visit_cap): {e}");
    let gs = b.gpu_state.as_mut().unwrap();
    gs.visit_cap[7] -= 1;
    // A stale position tag on a key the step reads.
    let t = gs.visit_pos_tags[3][10];
    gs.visit_pos_tags[3][10] = 999;
    let e = plan_decode_step(b, gpu, 40).expect_err("bad tag");
    eprintln!("  refused (tag): {e}");
    let gs = b.gpu_state.as_mut().unwrap();
    gs.visit_pos_tags[3][10] = t;
    plan_decode_step(b, gpu, 40).expect("restored state plans again");
    // Out-of-order, both directions.
    plan_decode_step(b, gpu, 41).expect_err("ahead");
    plan_decode_step(b, gpu, 39).expect_err("behind");
}

fn run_matrix(max_seq: usize, segments: &[(usize, usize)], name: &str) {
    let mut gpu = Gpu::init().expect("gpu");
    let mut b = load(&mut gpu, max_seq);
    let toks = tokens(max_seq + 16);
    let mut rec = Receipt::new(name);
    let mut salt = 0u32;
    let wit = |p: usize| matches!(p, 1023 | 1024 | 1025 | 2047 | 2048 | 4095 | 4096);
    let tr = |p: usize| {
        p % 16 == 0
            || matches!(
                p,
                1 | 2 | 127 | 128 | 129 | 191 | 192 | 193 | 1023 | 1024 | 1025
            )
            || p + 12 > max_seq.min(8192)
    };
    for &(s, e) in segments {
        segment(
            &mut b, &mut gpu, &mut rec, &toks, s, e, &tr, &wit, &mut salt,
        );
        if e + 1 == max_seq.min(8192) {
            refusals(&mut b, &mut gpu, e);
        }
    }
    bad_state_refusals(&mut b, &mut gpu, &toks);
    eprintln!("decode_step_hw {name}: ALL BITWISE-EQUAL");
}

/// Clipped non-power-of-two capacity 2053: first/consecutive rows, 126..130
/// and 190..194 bucket cuts, 1022..1026 (SWA wrap), the 2046..2050 wrap +
/// bucket crossing, the last legal row 2052 and the refused 2053; reset then a
/// same-bucket and a different-bucket prompt.
#[test]
#[ignore = "needs the typed-XTS artifact on a gfx1201 card"]
fn frozen_vs_stock_clipped_2053() {
    run_matrix(
        2053,
        &[
            (0, 200),
            (1018, 1030),
            (2044, 2052),
            (300, 312),
            (96, 134),
            (186, 198),
        ],
        "clipped2053.jsonl",
    );
}

/// 4094..4098 across the 4095 -> 4096 16 -> 8 split-band change, the last
/// legal row of a 4100-row load and the refused next one.
#[test]
#[ignore = "needs the typed-XTS artifact on a gfx1201 card"]
fn frozen_vs_stock_4100() {
    run_matrix(4100, &[(4090, 4099)], "cap4100.jsonl");
}

/// Capacity exactly at the 1024 ring break (SWA ring == full rows).
#[test]
#[ignore = "needs the typed-XTS artifact on a gfx1201 card"]
fn frozen_vs_stock_1024() {
    run_matrix(1024, &[(1010, 1023), (0, 30)], "cap1024.jsonl");
}

/// The longest admitted context on this fixture: the unit-tau position limit
/// (8192) of a 8200-row load. The last legal row 8191 is stepped, the next one
/// (non-unit tau) is refused.
#[test]
#[ignore = "needs the typed-XTS artifact on a gfx1201 card"]
fn frozen_vs_stock_longest_8192() {
    run_matrix(8200, &[(8184, 8191)], "cap8200.jsonl");
}

// ---------------------------------------------------------------------------
// Census
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
struct Census {
    pool_new: usize,
    pool_reused: usize,
    parked: usize,
    epoch: u64,
    kernels: usize,
    eff: hip_bridge::memory_effects::MemoryEffects,
}

fn census(gpu: &Gpu) -> Census {
    let (n, r, _) = gpu.pool_stats();
    Census {
        pool_new: n,
        pool_reused: r,
        parked: gpu.pool_parked_bytes(),
        epoch: rdna_compute::scratch::scratch_growth_epoch(),
        kernels: gpu.loaded_kernel_names().len(),
        eff: hip_bridge::memory_effects::snapshot(),
    }
}

/// Pointers a retained tape would bake (arena, KV, conv, FA2 scratch, shared
/// scratch the body touches).
fn pointers(b: &Inkling2Bundle, gpu: &Gpu) -> Vec<(String, usize)> {
    let gs = b.gpu_state.as_ref().unwrap();
    let a = gs.decode_arena.as_ref().unwrap();
    let mut v = vec![("arena".to_string(), a.base_ptr())];
    for (n, t) in a.named_tensors() {
        v.push((n, t.buf.as_ptr() as usize));
    }
    v.push(("fa2_lse".into(), gs.fa2_lse.buf.as_ptr() as usize));
    v.push(("fa2_oacc".into(), gs.fa2_oacc.buf.as_ptr() as usize));
    for (i, c) in gs.conv.iter().enumerate() {
        v.push((format!("conv{i}"), c.buf.as_ptr() as usize));
    }
    for i in 0..gs.k_cache.len() {
        for (h, t) in gs.k_cache[i].iter().chain(gs.v_cache[i].iter()).enumerate() {
            v.push((format!("kv{i}.{h}"), t.buf.as_ptr() as usize));
        }
    }
    let sc = &gpu.scratch;
    let p = |o: &Option<hip_bridge::DeviceBuffer>| o.as_ref().map_or(0, |b| b.as_ptr() as usize);
    let t = |o: &Option<GpuTensor>| o.as_ref().map_or(0, |b| b.buf.as_ptr() as usize);
    v.push(("int8_mmq_x".into(), p(&sc.int8_mmq_x_scratch)));
    v.push(("int8_mmq_down_x".into(), p(&sc.int8_mmq_down_x_scratch)));
    v.push(("mq_signs1".into(), t(&sc.mq_signs1)));
    v.push(("mq_signs2".into(), t(&sc.mq_signs2)));
    v.push(("mq_x_rot".into(), t(&sc.mq_x_rot)));
    v
}

/// 0 hot allocations / releases / JIT inside the body, one H2D at stage, one
/// D2H at finish, no memcpy/memset/readback in the body, stable pointers, no
/// host commit before finish; exact added VRAM.
#[test]
#[ignore = "needs the typed-XTS artifact on a gfx1201 card"]
fn decode_census() {
    let mut gpu = Gpu::init().expect("gpu");
    let mut b = load(&mut gpu, 2053);
    let toks = tokens(2100);
    reset(&mut b, &mut gpu);
    prefill(&mut b, &mut gpu, &toks[..186]);
    let cfg_bytes = hipfire_arch_inkling2::DecodeArena::bytes_for(&b.config).unwrap();
    let (free0, total) = gpu.hip.get_vram_info().unwrap();
    let c_pre = census(&gpu);
    ensure_decode_arena(&mut gpu, &mut b).expect("arena");
    let (free1, _) = gpu.hip.get_vram_info().unwrap();
    let c_arena = census(&gpu);
    let arena_bytes = b
        .gpu_state
        .as_ref()
        .unwrap()
        .decode_arena
        .as_ref()
        .unwrap()
        .bytes();
    assert_eq!(arena_bytes, cfg_bytes, "bytes_for == allocated");
    println!(
        "VRAM: decode arena {arena_bytes} B ({:.2} MiB); device free before/after ensure+prime {free0} -> {free1} B (delta {} B) of {total}; scratch epoch {} -> {}",
        arena_bytes as f64 / 1048576.0,
        free0 as i64 - free1 as i64,
        c_pre.epoch,
        c_arena.epoch
    );
    {
        let a = b.gpu_state.as_ref().unwrap().decode_arena.as_ref().unwrap();
        for r in a.regions() {
            println!(
                "  region {:10} off {:10} bytes {:10}",
                r.name, r.offset, r.bytes
            );
        }
    }
    // Stock one-step launch census for comparison (restored afterwards).
    let pre0 = save_pre(&b, &gpu, 186);
    let s0 = census(&gpu);
    let _ = forward_decode_gpu(&mut b, &mut gpu, toks[186], 186).unwrap();
    let s1 = census(&gpu);
    println!(
        "stock step: launches {}, htod {}, dtoh {}, dtod {}, memset {}, pool new {} reused {}",
        s1.eff.launches - s0.eff.launches,
        s1.eff.htod - s0.eff.htod,
        s1.eff.dtoh - s0.eff.dtoh,
        s1.eff.dtod - s0.eff.dtod,
        s1.eff.memset - s0.eff.memset,
        s1.pool_new - s0.pool_new,
        s1.pool_reused - s0.pool_reused
    );
    restore(&mut b, &gpu, &pre0);
    // First frozen token after construction: report what it still resolved.
    let w0 = census(&gpu);
    let _ = frozen_arm(&mut b, &mut gpu, toks[186], 186);
    let w1 = census(&gpu);
    println!(
        "first frozen token after construction: scratch epoch {} -> {}, pool new {}, kernels loaded +{}",
        w0.epoch,
        w1.epoch,
        w1.pool_new - w0.pool_new,
        w1.kernels - w0.kernels
    );
    assert_eq!(
        w0.epoch, w1.epoch,
        "pre-primed scratch must not grow on the first token"
    );
    let ptrs0 = pointers(&b, &gpu);
    let mut launches = None;
    for pos in 187..=200 {
        let plan = plan_decode_step(&b, &gpu, pos).expect("plan");
        let cfg = &b.config;
        let gw = b.gpu_weights.as_ref().unwrap();
        let gs = b.gpu_state.as_mut().unwrap();
        let mut arena = gs.decode_arena.take().unwrap();
        let mut logits = Vec::new();
        let host_before = (gs.seq_pos, gs.visit_cap.clone(), gs.visit_pos_tags.clone());
        let c0 = census(&gpu);
        stage_decode_input(&mut gpu, gw, &mut arena, toks[pos], pos as u32).unwrap();
        let c1 = census(&gpu);
        enqueue_decode_body_untraced(&mut gpu, cfg, gw, gs, &arena, &plan);
        let c2 = census(&gpu);
        // Nothing committed by the enqueue.
        assert_eq!(
            (gs.seq_pos, &gs.visit_cap, &gs.visit_pos_tags),
            (host_before.0, &host_before.1, &host_before.2),
            "pos {pos}: body mutated host cursors/tags"
        );
        finish_decode_step(&gpu, gs, &arena, pos, &mut logits).unwrap();
        let c3 = census(&gpu);
        gs.decode_arena = Some(arena);
        let (st, bd, fin) = (
            c1.eff.since(c0.eff),
            c2.eff.since(c1.eff),
            c3.eff.since(c2.eff),
        );
        assert_eq!(
            (st.htod, st.dtoh, st.dtod, st.memset, st.launches),
            (1, 0, 0, 0, 0),
            "stage: one H2D only: {st:?}"
        );
        assert_eq!(
            (bd.htod, bd.dtoh, bd.dtod, bd.memset),
            (0, 0, 0, 0),
            "body effects: {bd:?}"
        );
        assert_eq!(
            (fin.htod, fin.dtoh, fin.dtod, fin.memset, fin.launches),
            (0, 1, 0, 0, 0),
            "finish: one D2H only: {fin:?}"
        );
        assert_eq!(
            (c2.pool_new, c2.pool_reused, c2.parked, c2.epoch, c2.kernels),
            (c1.pool_new, c1.pool_reused, c1.parked, c1.epoch, c1.kernels),
            "pos {pos}: body allocated/released/grew scratch/loaded a module"
        );
        match launches {
            None => launches = Some(bd.launches),
            Some(l) => assert_eq!(l, bd.launches, "pos {pos}: launch count not stable"),
        }
        assert_eq!(ptrs0, pointers(&b, &gpu), "pos {pos}: a pointee moved");
    }
    let l = launches.unwrap();
    println!(
        "frozen body: {l} launches/step, 0 alloc/release/scratch growth/module load, 0 memcpy/memset/readback; stage 1 H2D, finish 1 D2H; {} stable pointers; bucket crossing 192 -> 193 inside pos 187..=200",
        ptrs0.len()
    );
    let (hw, cap) = b
        .gpu_state
        .as_ref()
        .unwrap()
        .decode_arena
        .as_ref()
        .unwrap()
        .vis_high_water();
    println!("visit-temporary slab high-water {hw} of {cap} B");
}

fn enqueue_decode_body_untraced(
    gpu: &mut Gpu,
    cfg: &Inkling2Config,
    gw: &hipfire_arch_inkling2::Inkling2GpuWeights,
    gs: &hipfire_arch_inkling2::Inkling2GpuState,
    arena: &hipfire_arch_inkling2::DecodeArena,
    plan: &hipfire_arch_inkling2::DecodeStepPlan,
) {
    hipfire_arch_inkling2::enqueue_decode_body(gpu, cfg, gw, gs, arena, plan).expect("enqueue");
}

fn planted(plant: u8) {
    let mut gpu = Gpu::init().expect("gpu");
    let mut b = load(&mut gpu, 300);
    let toks = tokens(320);
    reset(&mut b, &mut gpu);
    prefill(&mut b, &mut gpu, &toks[..40]);
    let mut rec = Receipt::new("planted.jsonl");
    let mut salt = 0u32;
    for pos in 40..44 {
        let o = Opts {
            traced: false,
            witness: false,
            plant,
        };
        lockstep(&mut b, &mut gpu, &mut rec, toks[pos], pos, &o, &mut salt);
    }
}

/// Positive control: a flipped conv bit in the frozen arm must be reported at
/// its first differing site.
#[test]
#[ignore = "needs the typed-XTS artifact on a gfx1201 card"]
#[should_panic(expected = "MISMATCH")]
fn planted_conv_fault_is_caught() {
    planted(1);
}

/// Positive control: a stale device position (host plan says `pos`, the staged
/// record says `pos - 1`) must change the output and be caught.
#[test]
#[ignore = "needs the typed-XTS artifact on a gfx1201 card"]
#[should_panic(expected = "MISMATCH")]
fn planted_stale_position_is_caught() {
    planted(2);
}

/// The selection key: equal for equal state, a new bucket across the 192/193
/// cut with everything else equal, a new layout epoch when shared scratch
/// grows, a new load generation for a re-created arena.
#[test]
#[ignore = "needs the typed-XTS artifact on a gfx1201 card"]
fn key_follows_bucket_scratch_and_reload() {
    let mut gpu = Gpu::init().expect("gpu");
    let mut b = load(&mut gpu, 2053);
    let toks = tokens(400);
    reset(&mut b, &mut gpu);
    prefill(&mut b, &mut gpu, &toks[..190]);
    ensure_decode_arena(&mut gpu, &mut b).unwrap();
    let k190 = plan_decode_step(&b, &gpu, 190).unwrap().key;
    assert_eq!(k190, plan_decode_step(&b, &gpu, 190).unwrap().key);
    for pos in 190..192 {
        frozen_arm(&mut b, &mut gpu, toks[pos], pos);
    }
    let ka = plan_decode_step(&b, &gpu, 192).unwrap().key; // pos 192: 193 live keys
    assert_eq!(ka.bucket.lo, 193, "pos 192 has 193 live keys");
    let k191 = {
        // pos 191 (n_full 192) was the last step of the [129,192] bucket.
        assert_eq!(k190.bucket.hi, 192);
        k190.clone()
    };
    assert_ne!(k191.bucket, ka.bucket);
    assert_eq!(k191.load_generation, ka.load_generation);
    assert_eq!(k191.layout_epoch, ka.layout_epoch);
    assert_eq!(k191.recipe, ka.recipe);
    assert_eq!(k191.schedule, ka.schedule);
    assert_eq!(k191.capacity, ka.capacity);
    println!(
        "bucket [129,192] split_cap[0..4] {:?}; [193,2048] {:?}; schedule digest {:#018x}; recipe {:?}",
        &k191.bucket.split_cap[..4],
        &ka.bucket.split_cap[..4],
        ka.schedule.digest(),
        ka.recipe
    );
    let kb = ka.clone();
    frozen_arm(&mut b, &mut gpu, toks[192], 192);
    // Shared scratch growth changes the layout epoch (and so the key); the
    // next step still runs.
    gpu.reserve_int8_mmq(2304, 4096).unwrap();
    let kc = plan_decode_step(&b, &gpu, 193).unwrap().key;
    assert_ne!(
        kb.layout_epoch, kc.layout_epoch,
        "scratch growth must change the epoch"
    );
    assert_ne!(kb, kc);
    frozen_arm(&mut b, &mut gpu, toks[193], 193);
    // A re-created arena (new owner) is a new load generation.
    let gs = b.gpu_state.as_mut().unwrap();
    gs.decode_arena.take().unwrap().free(&mut gpu);
    ensure_decode_arena(&mut gpu, &mut b).unwrap();
    let kd = plan_decode_step(&b, &gpu, 194).unwrap().key;
    assert_ne!(kc.load_generation, kd.load_generation);
    frozen_arm(&mut b, &mut gpu, toks[194], 194);
}

/// Exact device bytes the frozen decode adds to a fresh load, from the
/// process's own device-byte ledger (`HipRuntime::own_device_bytes`): the
/// arena allocation plus whatever priming grows in the shared scratch, then
/// what the first real token still grows (nothing, if priming is exact).
#[test]
#[ignore = "needs the typed-XTS artifact on a gfx1201 card"]
fn added_vram_on_fresh_load() {
    let mut gpu = Gpu::init().expect("gpu");
    let mut b = load(&mut gpu, 2053);
    let toks = tokens(8);
    reset(&mut b, &mut gpu);
    let l0 = gpu.hip.own_device_bytes();
    ensure_decode_arena(&mut gpu, &mut b).unwrap();
    let l1 = gpu.hip.own_device_bytes();
    let arena = b
        .gpu_state
        .as_ref()
        .unwrap()
        .decode_arena
        .as_ref()
        .unwrap()
        .bytes();
    let _ = frozen_arm(&mut b, &mut gpu, toks[0], 0);
    let l2 = gpu.hip.own_device_bytes();
    println!(
        "ADDED VRAM (fresh load, before any prefill): arena {arena} B; ledger delta at ensure+prime {} B (priming scratch growth {} B); first real token grows {} B more; total {} B ({:.3} MiB)",
        l1 - l0,
        (l1 - l0) as i64 - arena as i64,
        l2 as i64 - l1 as i64,
        l2 - l0,
        (l2 - l0) as f64 / 1048576.0
    );
}
