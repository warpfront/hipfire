// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Real-GPU correctness and timing probe for the gfx1010 tiled Q8 attention candidate.
//!
//! Compares `attention_q8_0_flash_tiled_swa_strided_gfx1010` (candidate) against
//! `attention_q8_0_kv_batched_swa_strided_gfx1010` (baseline) on identical strided
//! QKV + plain Q8 K/V + positions bytes. KV is built through the existing
//! `kv_cache_write_q8_0_batched` writer — no hand-packed echo oracle.
//!
//! Contract (local://spark-gfx1010-attention-tiled-spec.json + probe assignment):
//!   - HD fixed at 256; candidate rejects other head dims before mutation.
//!   - Cases: batch c ∈ {1,7,9,63,64,65}; windows 0/512; seq edges 511/512/513/1160;
//!     MANDATORY empty-row-tile fixture bs=8 pos0=539 window=512; one permuted
//!     (non-monotonic) positions case.
//!   - Numeric bar: finite outputs, max-abs ≤ 1e-4 vs baseline (online association,
//!     not bit-exact). Written Y extent only; pad outside extent unchanged.
//!   - Inputs (qkv/K/V/positions) unchanged across candidate launch.
//!   - Malformed hd≠256 / stride / extents → Err and no Y mutation.
//!   - Host launch+completion timings (not kernel-only) for bs=64 × ctx 302/1160 ×
//!     windows 0/512; warmups=10 samples=5; both entry orders (or `--reverse` first).
//!
//! Machine-readable JSON on stdout; exit 0 on full pass, 1 on any failure.
//! Does NOT call or claim full-model parity.
//!
//! Run (after worker lands Cargo example + Rust method):
//!   cargo run --release -p rdna-compute --features lab \
//!     --example test_spark_attn_tiled_gfx1010_parity [-- --reverse]

use rdna_compute::{DType, Gpu, GpuTensor};
use std::time::Instant;

const NH: usize = 16;
const NKV: usize = 4;
const HD: usize = 256;
const TOL: f32 = 1e-4;
const WARMUPS: usize = 10;
const SAMPLES: usize = 5;
/// Extra f32 elements past the written `[bs * nh * hd]` extent; poison-checked.
const OUT_PAD: usize = 64;
const POISON: f32 = 1337.042;

#[derive(Clone, Debug)]
struct Case {
    name: String,
    bs: usize,
    positions: Vec<i32>,
    window: usize,
    max_seq: usize,
}

struct Verdict {
    pass: bool,
    failures: usize,
    cases: Vec<String>,
    guards: Vec<String>,
    timings: Vec<String>,
}

impl Verdict {
    fn new() -> Self {
        Self {
            pass: true,
            failures: 0,
            cases: Vec::new(),
            guards: Vec::new(),
            timings: Vec::new(),
        }
    }
    fn fail_case(&mut self, rec: String) {
        self.failures += 1;
        self.pass = false;
        self.cases.push(rec);
    }
    fn ok_case(&mut self, rec: String) {
        self.cases.push(rec);
    }
    fn fail_guard(&mut self, rec: String) {
        self.failures += 1;
        self.pass = false;
        self.guards.push(rec);
    }
    fn ok_guard(&mut self, rec: String) {
        self.guards.push(rec);
    }
}

fn main() {
    let reverse = std::env::args().any(|a| a == "--reverse");
    let mut v = Verdict::new();

    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            println!(
                r#"{{"pass":false,"error":"gpu_init","detail":"{e:?}","cases":[],"guards":[],"timings":[]}}"#
            );
            std::process::exit(1);
        }
    };
    let arch = gpu.arch.clone();
    if !gpu.arch_caps.is_gfx1010() {
        println!(
            r#"{{"pass":false,"arch":"{arch}","error":"exact_gfx1010_required","cases":[],"guards":[],"timings":[]}}"#
        );
        std::process::exit(1);
    }

    for c in build_cases() {
        run_parity_case(&mut gpu, &c, &mut v);
    }
    run_malformed_guards(&mut gpu, &mut v);
    run_timings(&mut gpu, reverse, &mut v);

    println!("{{");
    println!(r#"  "pass": {},"#, if v.pass { "true" } else { "false" });
    println!(r#"  "arch": "{arch}","#);
    println!(r#"  "nh": {NH}, "nkv": {NKV}, "hd": {HD}, "tol": {TOL},"#);
    println!(r#"  "reverse": {},"#, reverse);
    println!(r#"  "failures": {},"#, v.failures);
    emit_json_array("cases", &v.cases);
    print!(",\n");
    emit_json_array("guards", &v.guards);
    print!(",\n");
    emit_json_array("timings", &v.timings);
    print!(",\n");
    println!(
        r#"  "note": "host_launch_completion_per_call samples+median; not kernel-only; full-model parity not claimed""#
    );
    println!("}}");

    if !v.pass {
        std::process::exit(1);
    }
}

fn emit_json_array(key: &str, items: &[String]) {
    println!(r#"  "{key}": ["#);
    for (i, rec) in items.iter().enumerate() {
        let comma = if i + 1 == items.len() { "" } else { "," };
        println!("    {rec}{comma}");
    }
    print!(r#"  ]"#);
}

fn build_cases() -> Vec<Case> {
    let mut out = Vec::new();

    // MANDATORY empty-row-tile: bs=8, pos0=539, window=512 → t_lo ∈ 28..35;
    // tile0 [0,32) fully masked for rows with t_lo >= 32 while siblings active.
    out.push(Case {
        name: "empty_row_tile_bs8_pos0_539_w512".into(),
        bs: 8,
        positions: (0..8).map(|i| 539 + i as i32).collect(),
        window: 512,
        max_seq: 600,
    });

    // Non-monotonic positions — no host monotonic assumption.
    out.push(Case {
        name: "permuted_positions_bs8_w512".into(),
        bs: 8,
        positions: vec![400, 37, 511, 200, 3, 450, 100, 333],
        window: 512,
        max_seq: 512,
    });

    let batches = [1usize, 7, 9, 63, 64, 65];
    let windows = [0usize, 512];
    let ends = [511i32, 512, 513, 1160];
    for &c in &batches {
        for &w in &windows {
            for &end in &ends {
                if (c as i32) > end + 1 {
                    continue;
                }
                let pos0 = end - c as i32 + 1;
                if pos0 < 0 {
                    continue;
                }
                let max_seq = (end as usize + 1).max(c).max(16);
                out.push(Case {
                    name: format!("tail_c{c}_end{end}_w{w}"),
                    bs: c,
                    positions: (0..c).map(|i| pos0 + i as i32).collect(),
                    window: w,
                    max_seq,
                });
            }
        }
    }
    out
}

fn run_parity_case(gpu: &mut Gpu, c: &Case, v: &mut Verdict) {
    let q_dim = NH * HD;
    let kv_dim = NKV * HD;
    let q_stride = q_dim + 2 * kv_dim;
    let out_extent = c.bs * q_dim;
    let out_alloc = out_extent + OUT_PAD;
    let max_pos = c.positions.iter().copied().max().unwrap_or(0) as usize;
    if max_pos >= c.max_seq {
        v.fail_case(format!(
            r#"{{"name":"{}","ok":false,"error":"pos_exceeds_max_seq","max_pos":{max_pos},"max_seq":{}}}"#,
            c.name, c.max_seq
        ));
        return;
    }
    let max_ctx = max_pos + 1;

    let qkv_host = synth_f32(c.bs * q_stride, 0xA11C_E001 ^ ((c.bs as u64) << 17));
    let k_host = synth_f32(c.max_seq * kv_dim, 0x4B51_E001 ^ (c.max_seq as u64));
    let v_host = synth_f32(c.max_seq * kv_dim, 0x05A1_E002 ^ ((c.window as u64) << 9));

    let qkv = match gpu.upload_f32(&qkv_host, &[c.bs * q_stride]) {
        Ok(t) => t,
        Err(e) => {
            v.fail_case(format!(
                r#"{{"name":"{}","ok":false,"error":"qkv_upload","detail":"{e:?}"}}"#,
                c.name
            ));
            return;
        }
    };
    let d_k = match gpu.upload_f32(&k_host, &[c.max_seq * kv_dim]) {
        Ok(t) => t,
        Err(e) => {
            let _ = gpu.free_tensor(qkv);
            v.fail_case(format!(
                r#"{{"name":"{}","ok":false,"error":"k_f32_upload","detail":"{e:?}"}}"#,
                c.name
            ));
            return;
        }
    };
    let d_v = match gpu.upload_f32(&v_host, &[c.max_seq * kv_dim]) {
        Ok(t) => t,
        Err(e) => {
            let _ = gpu.free_tensor(qkv);
            let _ = gpu.free_tensor(d_k);
            v.fail_case(format!(
                r#"{{"name":"{}","ok":false,"error":"v_f32_upload","detail":"{e:?}"}}"#,
                c.name
            ));
            return;
        }
    };

    let pos_all: Vec<i32> = (0..c.max_seq as i32).collect();
    let pos_all_t = match upload_i32(gpu, &pos_all) {
        Ok(t) => t,
        Err(e) => {
            let _ = gpu.free_tensor(qkv);
            let _ = gpu.free_tensor(d_k);
            let _ = gpu.free_tensor(d_v);
            v.fail_case(format!(
                r#"{{"name":"{}","ok":false,"error":"pos_all_upload","detail":"{e}"}}"#,
                c.name
            ));
            return;
        }
    };
    let cache_bytes = c.max_seq * NKV * (HD / 32) * 34;
    let k_cache = match gpu.zeros(&[cache_bytes], DType::Q8_0) {
        Ok(t) => t,
        Err(e) => {
            let _ = gpu.free_tensor(qkv);
            let _ = gpu.free_tensor(d_k);
            let _ = gpu.free_tensor(d_v);
            let _ = gpu.free_tensor(pos_all_t);
            v.fail_case(format!(
                r#"{{"name":"{}","ok":false,"error":"k_cache_alloc","detail":"{e:?}"}}"#,
                c.name
            ));
            return;
        }
    };
    let v_cache = match gpu.zeros(&[cache_bytes], DType::Q8_0) {
        Ok(t) => t,
        Err(e) => {
            let _ = gpu.free_tensor(qkv);
            let _ = gpu.free_tensor(d_k);
            let _ = gpu.free_tensor(d_v);
            let _ = gpu.free_tensor(pos_all_t);
            let _ = gpu.free_tensor(k_cache);
            v.fail_case(format!(
                r#"{{"name":"{}","ok":false,"error":"v_cache_alloc","detail":"{e:?}"}}"#,
                c.name
            ));
            return;
        }
    };
    if let Err(e) = gpu.kv_cache_write_q8_0_batched(&k_cache, &d_k, &pos_all_t, NKV, HD, c.max_seq)
    {
        let _ = gpu.free_tensor(qkv);
        let _ = gpu.free_tensor(d_k);
        let _ = gpu.free_tensor(d_v);
        let _ = gpu.free_tensor(pos_all_t);
        let _ = gpu.free_tensor(k_cache);
        let _ = gpu.free_tensor(v_cache);
        v.fail_case(format!(
            r#"{{"name":"{}","ok":false,"error":"kv_write_k","detail":"{e:?}"}}"#,
            c.name
        ));
        return;
    }
    if let Err(e) = gpu.kv_cache_write_q8_0_batched(&v_cache, &d_v, &pos_all_t, NKV, HD, c.max_seq)
    {
        let _ = gpu.free_tensor(qkv);
        let _ = gpu.free_tensor(d_k);
        let _ = gpu.free_tensor(d_v);
        let _ = gpu.free_tensor(pos_all_t);
        let _ = gpu.free_tensor(k_cache);
        let _ = gpu.free_tensor(v_cache);
        v.fail_case(format!(
            r#"{{"name":"{}","ok":false,"error":"kv_write_v","detail":"{e:?}"}}"#,
            c.name
        ));
        return;
    }
    let _ = gpu.free_tensor(d_k);
    let _ = gpu.free_tensor(d_v);
    let _ = gpu.free_tensor(pos_all_t);

    let positions = match upload_i32(gpu, &c.positions) {
        Ok(t) => t,
        Err(e) => {
            let _ = gpu.free_tensor(qkv);
            let _ = gpu.free_tensor(k_cache);
            let _ = gpu.free_tensor(v_cache);
            v.fail_case(format!(
                r#"{{"name":"{}","ok":false,"error":"positions_upload","detail":"{e}"}}"#,
                c.name
            ));
            return;
        }
    };

    let mut out_seed = vec![POISON; out_alloc];
    for x in out_seed.iter_mut().take(out_extent) {
        *x = 0.0;
    }
    let out_ref = match gpu.upload_f32(&out_seed, &[out_alloc]) {
        Ok(t) => t,
        Err(e) => {
            let _ = gpu.free_tensor(qkv);
            let _ = gpu.free_tensor(k_cache);
            let _ = gpu.free_tensor(v_cache);
            let _ = gpu.free_tensor(positions);
            v.fail_case(format!(
                r#"{{"name":"{}","ok":false,"error":"out_ref_upload","detail":"{e:?}"}}"#,
                c.name
            ));
            return;
        }
    };
    let out_new = match gpu.upload_f32(&out_seed, &[out_alloc]) {
        Ok(t) => t,
        Err(e) => {
            let _ = gpu.free_tensor(qkv);
            let _ = gpu.free_tensor(k_cache);
            let _ = gpu.free_tensor(v_cache);
            let _ = gpu.free_tensor(positions);
            let _ = gpu.free_tensor(out_ref);
            v.fail_case(format!(
                r#"{{"name":"{}","ok":false,"error":"out_new_upload","detail":"{e:?}"}}"#,
                c.name
            ));
            return;
        }
    };

    let snap_qkv = download_bytes(gpu, &qkv);
    let snap_k = download_bytes(gpu, &k_cache);
    let snap_v = download_bytes(gpu, &v_cache);
    let snap_pos = download_bytes(gpu, &positions);

    if let Err(e) = gpu.attention_q8_0_kv_batched_swa_strided_gfx1010(
        &qkv,
        &k_cache,
        &v_cache,
        &out_ref,
        &positions,
        NH,
        NKV,
        HD,
        c.max_seq,
        max_ctx,
        c.bs,
        c.window,
        q_stride,
    ) {
        let _ = gpu.free_tensor(qkv);
        let _ = gpu.free_tensor(k_cache);
        let _ = gpu.free_tensor(v_cache);
        let _ = gpu.free_tensor(positions);
        let _ = gpu.free_tensor(out_ref);
        let _ = gpu.free_tensor(out_new);
        v.fail_case(format!(
            r#"{{"name":"{}","ok":false,"error":"baseline_launch","detail":"{e:?}"}}"#,
            c.name
        ));
        return;
    }
    if let Err(e) = gpu.hip.device_synchronize() {
        let _ = gpu.free_tensor(qkv);
        let _ = gpu.free_tensor(k_cache);
        let _ = gpu.free_tensor(v_cache);
        let _ = gpu.free_tensor(positions);
        let _ = gpu.free_tensor(out_ref);
        let _ = gpu.free_tensor(out_new);
        v.fail_case(format!(
            r#"{{"name":"{}","ok":false,"error":"baseline_sync","detail":"{e:?}"}}"#,
            c.name
        ));
        return;
    }

    // Candidate: no max_ctx arg.
    if let Err(e) = gpu.attention_q8_0_flash_tiled_swa_strided_gfx1010(
        &qkv,
        &k_cache,
        &v_cache,
        &out_new,
        &positions,
        NH,
        NKV,
        HD,
        c.max_seq,
        c.bs,
        c.window,
        q_stride,
    ) {
        let _ = gpu.free_tensor(qkv);
        let _ = gpu.free_tensor(k_cache);
        let _ = gpu.free_tensor(v_cache);
        let _ = gpu.free_tensor(positions);
        let _ = gpu.free_tensor(out_ref);
        let _ = gpu.free_tensor(out_new);
        v.fail_case(format!(
            r#"{{"name":"{}","ok":false,"error":"candidate_launch","detail":"{e:?}"}}"#,
            c.name
        ));
        return;
    }
    if let Err(e) = gpu.hip.device_synchronize() {
        let _ = gpu.free_tensor(qkv);
        let _ = gpu.free_tensor(k_cache);
        let _ = gpu.free_tensor(v_cache);
        let _ = gpu.free_tensor(positions);
        let _ = gpu.free_tensor(out_ref);
        let _ = gpu.free_tensor(out_new);
        v.fail_case(format!(
            r#"{{"name":"{}","ok":false,"error":"candidate_sync","detail":"{e:?}"}}"#,
            c.name
        ));
        return;
    }

    let inputs_ok = snap_qkv == download_bytes(gpu, &qkv)
        && snap_k == download_bytes(gpu, &k_cache)
        && snap_v == download_bytes(gpu, &v_cache)
        && snap_pos == download_bytes(gpu, &positions);

    let a = match gpu.download_f32(&out_ref) {
        Ok(x) => x,
        Err(e) => {
            let _ = gpu.free_tensor(qkv);
            let _ = gpu.free_tensor(k_cache);
            let _ = gpu.free_tensor(v_cache);
            let _ = gpu.free_tensor(positions);
            let _ = gpu.free_tensor(out_ref);
            let _ = gpu.free_tensor(out_new);
            v.fail_case(format!(
                r#"{{"name":"{}","ok":false,"error":"dl_ref","detail":"{e:?}"}}"#,
                c.name
            ));
            return;
        }
    };
    let b = match gpu.download_f32(&out_new) {
        Ok(x) => x,
        Err(e) => {
            let _ = gpu.free_tensor(qkv);
            let _ = gpu.free_tensor(k_cache);
            let _ = gpu.free_tensor(v_cache);
            let _ = gpu.free_tensor(positions);
            let _ = gpu.free_tensor(out_ref);
            let _ = gpu.free_tensor(out_new);
            v.fail_case(format!(
                r#"{{"name":"{}","ok":false,"error":"dl_new","detail":"{e:?}"}}"#,
                c.name
            ));
            return;
        }
    };

    let _ = gpu.free_tensor(qkv);
    let _ = gpu.free_tensor(k_cache);
    let _ = gpu.free_tensor(v_cache);
    let _ = gpu.free_tensor(positions);
    let _ = gpu.free_tensor(out_ref);
    let _ = gpu.free_tensor(out_new);

    let mut max_abs = 0.0f32;
    let mut finite = true;
    let mut worst = 0usize;
    for i in 0..out_extent {
        if !a[i].is_finite() || !b[i].is_finite() {
            finite = false;
        }
        let d = (a[i] - b[i]).abs();
        if d > max_abs {
            max_abs = d;
            worst = i;
        }
    }
    let mut pad_ok = true;
    for i in out_extent..out_alloc {
        if a[i] != POISON || b[i] != POISON {
            pad_ok = false;
            break;
        }
    }
    let ok = finite && max_abs <= TOL && pad_ok && inputs_ok;
    let rec = format!(
        r#"{{"name":"{}","ok":{},"bs":{},"window":{},"max_seq":{},"max_ctx":{},"max_abs":{:.6e},"finite":{},"pad_ok":{},"inputs_unchanged":{},"worst_i":{},"pos0":{},"pos_last":{}}}"#,
        c.name,
        ok,
        c.bs,
        c.window,
        c.max_seq,
        max_ctx,
        max_abs,
        finite,
        pad_ok,
        inputs_ok,
        worst,
        c.positions.first().copied().unwrap_or(-1),
        c.positions.last().copied().unwrap_or(-1),
    );
    if ok {
        v.ok_case(rec);
    } else {
        v.fail_case(rec);
    }
}

fn run_malformed_guards(gpu: &mut Gpu, v: &mut Verdict) {
    let q_dim = NH * HD;
    let kv_dim = NKV * HD;
    let q_stride = q_dim + 2 * kv_dim;
    let bs = 8usize;
    let max_seq = 64usize;
    let window = 0usize;

    let qkv_host = synth_f32(bs * q_stride, 0x6A1D_0001);
    let qkv = gpu
        .upload_f32(&qkv_host, &[bs * q_stride])
        .expect("guard qkv");
    let cache_bytes = max_seq * NKV * (HD / 32) * 34;
    let k_cache = gpu.zeros(&[cache_bytes], DType::Q8_0).expect("guard k");
    let v_cache = gpu.zeros(&[cache_bytes], DType::Q8_0).expect("guard v");
    let pos_host: Vec<i32> = (0..bs as i32).collect();
    let pos = upload_i32(gpu, &pos_host).expect("guard pos");
    let out_extent = bs * q_dim;
    let out_alloc = out_extent + OUT_PAD;
    let mut y_host = vec![POISON; out_alloc];
    for x in y_host.iter_mut().take(out_extent) {
        *x = 42.0;
    }
    let y = gpu.upload_f32(&y_host, &[out_alloc]).expect("guard y");
    let y_before = gpu.download_f32(&y).expect("guard y snap");

    // hd != 256
    {
        let r = gpu.attention_q8_0_flash_tiled_swa_strided_gfx1010(
            &qkv, &k_cache, &v_cache, &y, &pos, NH, NKV, 128, max_seq, bs, window, q_stride,
        );
        let unchanged = y_before == gpu.download_f32(&y).expect("dl");
        let ok = r.is_err() && unchanged;
        let rec = format!(
            r#"{{"name":"reject_hd_128","ok":{},"err":{},"y_unchanged":{}}}"#,
            ok,
            r.is_err(),
            unchanged
        );
        if ok {
            v.ok_guard(rec);
        } else {
            v.fail_guard(rec);
        }
    }

    // q_stride == 0
    {
        let r = gpu.attention_q8_0_flash_tiled_swa_strided_gfx1010(
            &qkv, &k_cache, &v_cache, &y, &pos, NH, NKV, HD, max_seq, bs, window, 0,
        );
        let unchanged = y_before == gpu.download_f32(&y).expect("dl");
        let ok = r.is_err() && unchanged;
        let rec = format!(
            r#"{{"name":"reject_qstride_0","ok":{},"err":{},"y_unchanged":{}}}"#,
            ok,
            r.is_err(),
            unchanged
        );
        if ok {
            v.ok_guard(rec);
        } else {
            v.fail_guard(rec);
        }
    }

    // q_stride < q_dim
    {
        let r = gpu.attention_q8_0_flash_tiled_swa_strided_gfx1010(
            &qkv,
            &k_cache,
            &v_cache,
            &y,
            &pos,
            NH,
            NKV,
            HD,
            max_seq,
            bs,
            window,
            q_dim - 1,
        );
        let unchanged = y_before == gpu.download_f32(&y).expect("dl");
        let ok = r.is_err() && unchanged;
        let rec = format!(
            r#"{{"name":"reject_qstride_lt_qdim","ok":{},"err":{},"y_unchanged":{}}}"#,
            ok,
            r.is_err(),
            unchanged
        );
        if ok {
            v.ok_guard(rec);
        } else {
            v.fail_guard(rec);
        }
    }

    // undersized out
    {
        let short_n = (out_extent / 2).max(1);
        let y_short = gpu
            .upload_f32(&y_host[..short_n].to_vec(), &[short_n])
            .expect("short y");
        let short_before = gpu.download_f32(&y_short).expect("dl");
        let r = gpu.attention_q8_0_flash_tiled_swa_strided_gfx1010(
            &qkv,
            &k_cache,
            &v_cache,
            &y_short,
            &pos,
            NH,
            NKV,
            HD,
            max_seq,
            bs,
            window,
            q_stride,
        );
        let unchanged = short_before == gpu.download_f32(&y_short).expect("dl");
        let ok = r.is_err() && unchanged;
        let rec = format!(
            r#"{{"name":"reject_out_extent","ok":{},"err":{},"y_unchanged":{}}}"#,
            ok,
            r.is_err(),
            unchanged
        );
        if ok {
            v.ok_guard(rec);
        } else {
            v.fail_guard(rec);
        }
        let _ = gpu.free_tensor(y_short);
    }

    // undersized K cache
    {
        let k_short = gpu.zeros(&[8], DType::Q8_0).expect("k short");
        let r = gpu.attention_q8_0_flash_tiled_swa_strided_gfx1010(
            &qkv, &k_short, &v_cache, &y, &pos, NH, NKV, HD, max_seq, bs, window, q_stride,
        );
        let unchanged = y_before == gpu.download_f32(&y).expect("dl");
        let ok = r.is_err() && unchanged;
        let rec = format!(
            r#"{{"name":"reject_k_extent","ok":{},"err":{},"y_unchanged":{}}}"#,
            ok,
            r.is_err(),
            unchanged
        );
        if ok {
            v.ok_guard(rec);
        } else {
            v.fail_guard(rec);
        }
        let _ = gpu.free_tensor(k_short);
    }

    let _ = gpu.free_tensor(qkv);
    let _ = gpu.free_tensor(k_cache);
    let _ = gpu.free_tensor(v_cache);
    let _ = gpu.free_tensor(pos);
    let _ = gpu.free_tensor(y);
}

fn run_timings(gpu: &mut Gpu, reverse: bool, v: &mut Verdict) {
    let q_dim = NH * HD;
    let kv_dim = NKV * HD;
    let q_stride = q_dim + 2 * kv_dim;
    let bs = 64usize;
    let contexts = [302usize, 1160usize];
    let windows = [0usize, 512usize];

    for &ctx in &contexts {
        let max_seq = ctx;
        let max_ctx = ctx;
        let pos0 = ctx as i32 - bs as i32;
        if pos0 < 0 {
            continue;
        }
        let positions: Vec<i32> = (0..bs).map(|i| pos0 + i as i32).collect();

        let qkv_host = synth_f32(bs * q_stride, 0x7100_0000 ^ ctx as u64);
        let k_host = synth_f32(max_seq * kv_dim, 0x8100_0001 ^ ctx as u64);
        let v_host = synth_f32(max_seq * kv_dim, 0x9100_0002 ^ ctx as u64);

        let qkv = match gpu.upload_f32(&qkv_host, &[bs * q_stride]) {
            Ok(t) => t,
            Err(e) => {
                v.timings.push(format!(
                    r#"{{"ctx":{ctx},"ok":false,"error":"qkv_upload","detail":"{e:?}"}}"#
                ));
                v.pass = false;
                v.failures += 1;
                continue;
            }
        };
        let d_k = gpu
            .upload_f32(&k_host, &[max_seq * kv_dim])
            .expect("time k");
        let d_v = gpu
            .upload_f32(&v_host, &[max_seq * kv_dim])
            .expect("time v");
        let pos_all: Vec<i32> = (0..max_seq as i32).collect();
        let pos_all_t = upload_i32(gpu, &pos_all).expect("time pos_all");
        let cache_bytes = max_seq * NKV * (HD / 32) * 34;
        let k_cache = gpu.zeros(&[cache_bytes], DType::Q8_0).expect("time kq8");
        let v_cache = gpu.zeros(&[cache_bytes], DType::Q8_0).expect("time vq8");
        gpu.kv_cache_write_q8_0_batched(&k_cache, &d_k, &pos_all_t, NKV, HD, max_seq)
            .expect("time write k");
        gpu.kv_cache_write_q8_0_batched(&v_cache, &d_v, &pos_all_t, NKV, HD, max_seq)
            .expect("time write v");
        let _ = gpu.free_tensor(d_k);
        let _ = gpu.free_tensor(d_v);
        let _ = gpu.free_tensor(pos_all_t);
        let positions_t = upload_i32(gpu, &positions).expect("time pos");
        let out_b = gpu.zeros(&[bs * q_dim], DType::F32).expect("time out b");
        let out_c = gpu.zeros(&[bs * q_dim], DType::F32).expect("time out c");

        for &window in &windows {
            let measure_baseline = |gpu: &mut Gpu| -> [f64; SAMPLES] {
                for _ in 0..WARMUPS {
                    gpu.attention_q8_0_kv_batched_swa_strided_gfx1010(
                        &qkv,
                        &k_cache,
                        &v_cache,
                        &out_b,
                        &positions_t,
                        NH,
                        NKV,
                        HD,
                        max_seq,
                        max_ctx,
                        bs,
                        window,
                        q_stride,
                    )
                    .expect("warmup baseline");
                }
                gpu.hip.device_synchronize().expect("sync");
                let mut samples = [0.0f64; SAMPLES];
                for s in &mut samples {
                    let t0 = Instant::now();
                    gpu.attention_q8_0_kv_batched_swa_strided_gfx1010(
                        &qkv,
                        &k_cache,
                        &v_cache,
                        &out_b,
                        &positions_t,
                        NH,
                        NKV,
                        HD,
                        max_seq,
                        max_ctx,
                        bs,
                        window,
                        q_stride,
                    )
                    .expect("sample baseline");
                    gpu.hip.device_synchronize().expect("sync");
                    *s = t0.elapsed().as_secs_f64() * 1e6;
                }
                samples
            };
            let measure_candidate = |gpu: &mut Gpu| -> [f64; SAMPLES] {
                for _ in 0..WARMUPS {
                    gpu.attention_q8_0_flash_tiled_swa_strided_gfx1010(
                        &qkv,
                        &k_cache,
                        &v_cache,
                        &out_c,
                        &positions_t,
                        NH,
                        NKV,
                        HD,
                        max_seq,
                        bs,
                        window,
                        q_stride,
                    )
                    .expect("warmup candidate");
                }
                gpu.hip.device_synchronize().expect("sync");
                let mut samples = [0.0f64; SAMPLES];
                for s in &mut samples {
                    let t0 = Instant::now();
                    gpu.attention_q8_0_flash_tiled_swa_strided_gfx1010(
                        &qkv,
                        &k_cache,
                        &v_cache,
                        &out_c,
                        &positions_t,
                        NH,
                        NKV,
                        HD,
                        max_seq,
                        bs,
                        window,
                        q_stride,
                    )
                    .expect("sample candidate");
                    gpu.hip.device_synchronize().expect("sync");
                    *s = t0.elapsed().as_secs_f64() * 1e6;
                }
                samples
            };

            // Both entry orders within the probe. `--reverse` only flips which
            // order runs first (still records both).
            let orders: [(&str, bool); 2] = if reverse {
                [("candidate_then_baseline", true), ("baseline_then_candidate", false)]
            } else {
                [("baseline_then_candidate", false), ("candidate_then_baseline", true)]
            };
            for &(order, cand_first) in &orders {
                let (base_samples, cand_samples) = if cand_first {
                    let c = measure_candidate(gpu);
                    let b = measure_baseline(gpu);
                    (b, c)
                } else {
                    let b = measure_baseline(gpu);
                    let c = measure_candidate(gpu);
                    (b, c)
                };
                let base_med = median_us(base_samples);
                let cand_med = median_us(cand_samples);
                v.timings.push(format!(
                    r#"{{"bs":{bs},"ctx":{ctx},"window":{window},"order":"{order}","baseline_median_us":{base_med:.3},"candidate_median_us":{cand_med:.3},"baseline_samples_us":{},"candidate_samples_us":{},"warmups":{WARMUPS},"sample_count":{SAMPLES},"metric":"host_launch_completion_per_call"}}"#,
                    format_samples_us(&base_samples),
                    format_samples_us(&cand_samples),
                ));
            }
        }

        let _ = gpu.free_tensor(qkv);
        let _ = gpu.free_tensor(k_cache);
        let _ = gpu.free_tensor(v_cache);
        let _ = gpu.free_tensor(positions_t);
        let _ = gpu.free_tensor(out_b);
        let _ = gpu.free_tensor(out_c);
    }
}

fn synth_f32(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u = ((s >> 32) as u32) as f32 / (u32::MAX as f32);
            (u - 0.5) * 0.5
        })
        .collect()
}

fn upload_i32(gpu: &mut Gpu, data: &[i32]) -> Result<GpuTensor, String> {
    let bytes =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data)) };
    gpu.upload_raw(bytes, &[data.len()])
        .map_err(|e| format!("{e:?}"))
}

fn download_bytes(gpu: &Gpu, t: &GpuTensor) -> Vec<u8> {
    let n = t.buf.size();
    let mut v = vec![0u8; n];
    gpu.hip.memcpy_dtoh(&mut v, &t.buf).expect("memcpy_dtoh");
    v
}

/// Median of a chronological per-call sample array. Sorts a stack copy so the
/// raw order stays available for JSON emission.
fn median_us(samples: [f64; SAMPLES]) -> f64 {
    let mut sorted = samples;
    sorted.sort_by(|a, b| a.total_cmp(b));
    sorted[SAMPLES / 2]
}

fn format_samples_us(samples: &[f64; SAMPLES]) -> String {
    let mut s = String::from("[");
    for (i, v) in samples.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&format!("{v:.3}"));
    }
    s.push(']');
    s
}
