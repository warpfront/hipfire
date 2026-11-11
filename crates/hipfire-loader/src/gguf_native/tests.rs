// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

use super::test_writer::*;
use super::*;
use hipfire_runtime::gguf::GgmlType;
use hipfire_runtime::weight_manifest::ShardPolicy;
use rdna_compute::DType;

fn write_fixture(
    dir: &tempfile::TempDir,
    kvs: &[(String, Kv)],
    tensors: &[Spec],
    with_data: bool,
) -> std::path::PathBuf {
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let path = dir.path().join(format!("fixture-{n}.gguf"));
    write_gguf(&path, kvs, tensors, with_data);
    path
}

fn open(tensors: &[Spec], nextn: u32) -> Result<GgufNativeModel, String> {
    let dir = tempfile::tempdir().unwrap();
    let p = write_fixture(&dir, &base_kvs("qwen35", nextn), tensors, true);
    GgufNativeModel::open_gated(&p, true)
}

fn expect_refusal(tensors: &[Spec], needles: &[&str]) -> String {
    let err = match open(tensors, 0) {
        Ok(_) => panic!("expected a refusal containing {needles:?}"),
        Err(e) => e,
    };
    assert!(err.starts_with("load refused:"), "wording: {err}");
    for n in needles {
        assert!(err.contains(n), "missing {n:?} in {err}");
    }
    err
}

#[test]
fn parses_maps_and_builds_the_manifest() {
    let tensors = base_tensors();
    let m = open(&tensors, 0).expect("canonical fixture admits");
    assert_eq!(m.arch_id, 5);
    let c = &m.hparams.config;
    assert_eq!(
        (c.dim, c.n_layers, c.vocab_size, c.hidden_dim),
        (256, 2, 512, 512)
    );

    // One manifest entry per file tensor, none tied (output.weight present).
    assert_eq!(m.manifest.entries.len(), tensors.len());
    assert!(m.manifest.entries.iter().all(|e| e.tied_to.is_none()));
    assert!(!m.manifest.mtp_present);

    let by_gguf = |n: &str| {
        m.manifest
            .entries
            .iter()
            .find(|e| e.gguf_name == n)
            .unwrap_or_else(|| panic!("no entry {n}"))
    };
    // dtype per tensor: distinct Ggml dtypes for quantised types, plain ones for floats.
    assert_eq!(by_gguf("token_embd.weight").weight.dtype, DType::GgmlQ8_0);
    assert_eq!(by_gguf("output.weight").weight.dtype, DType::GgmlQ6K);
    assert_eq!(
        by_gguf("blk.0.attn_qkv.weight").weight.dtype,
        DType::GgmlQ4K
    );
    assert_eq!(
        by_gguf("blk.0.attn_gate.weight").weight.dtype,
        DType::GgmlIQ4XS
    );
    assert_eq!(
        by_gguf("blk.1.attn_output.weight").weight.dtype,
        DType::GgmlQ5K
    );
    assert_eq!(by_gguf("blk.0.ssm_a").weight.dtype, DType::F32);
    assert_eq!(by_gguf("blk.0.ssm_beta.weight").weight.dtype, DType::F16);
    // names and layers follow the HFQ convention; logical shape is [rows, cols].
    let qkv = by_gguf("blk.0.attn_qkv.weight");
    assert_eq!(
        qkv.weight.name,
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
    );
    assert_eq!(qkv.weight.layer, Some(0));
    assert_eq!(qkv.weight.logical_shape, vec![256, 256]);
    assert_eq!(qkv.weight.policy, ShardPolicy::Replicate);
    assert_eq!(
        by_gguf("blk.1.attn_q.weight").weight.logical_shape,
        vec![512, 256]
    );
    assert_eq!(by_gguf("token_embd.weight").weight.layer, None);
    // exact byte extents.
    for spec in &tensors {
        let e = by_gguf(&spec.name);
        assert_eq!(e.byte_extent as usize, spec.bytes(), "{}", spec.name);
        assert_eq!(e.params as usize, spec.params(), "{}", spec.name);
    }
    assert_eq!(
        by_gguf("blk.0.attn_qkv.weight").byte_extent,
        (256 / 256) * 144 * 256
    );
    assert_eq!(by_gguf("blk.0.attn_gate.weight").byte_extent, 136 * 128);
    // the tensor bytes are reachable and exactly the extent.
    let bytes = m.tensor_bytes(by_gguf("blk.1.attn_v.weight")).unwrap();
    assert_eq!(bytes.len(), 210 * 128);

    // bpw census over every tensor.
    let bytes: usize = tensors.iter().map(Spec::bytes).sum();
    let params: usize = tensors.iter().map(Spec::params).sum();
    assert_eq!(m.manifest.census.total_bytes as usize, bytes);
    assert_eq!(m.manifest.census.total_params as usize, params);
    let want = bytes as f64 * 8.0 / params as f64;
    assert!(
        (m.manifest.bpw - want).abs() < 1e-12,
        "{} vs {want}",
        m.manifest.bpw
    );
    assert_eq!(m.manifest.census.by_type["Q4_K"].tensors, 7);
    assert_eq!(m.manifest.census.by_type["F32"].tensors, 11);
    // weights() is the shared declaration list.
    assert_eq!(m.manifest.weights().len(), tensors.len());
}

#[test]
fn an_absent_output_tensor_is_a_tied_head() {
    let tensors: Vec<Spec> = base_tensors()
        .into_iter()
        .filter(|t| t.name != "output.weight")
        .collect();
    let m = open(&tensors, 0).unwrap();
    let head = m
        .manifest
        .entries
        .iter()
        .find(|e| e.weight.name == "lm_head.weight")
        .expect("tied head declared");
    assert_eq!(
        head.tied_to.as_deref(),
        Some("model.language_model.embed_tokens.weight")
    );
    assert_eq!(head.byte_extent, 0);
    assert!(matches!(head.weight.policy, ShardPolicy::Tied { .. }));
    assert_eq!(head.weight.dtype, DType::GgmlQ8_0);
    // the tied head adds no bytes or params to the census.
    let bytes: usize = tensors.iter().map(Spec::bytes).sum();
    assert_eq!(m.manifest.census.total_bytes as usize, bytes);
}

#[test]
fn the_mtp_block_is_all_or_nothing() {
    let mut tensors = base_tensors();
    tensors.extend(mtp_tensors());
    let dir = tempfile::tempdir().unwrap();
    let p = write_fixture(&dir, &base_kvs("qwen35", 1), &tensors, true);
    let m = GgufNativeModel::open_gated(&p, true).expect("trunk + MTP admits");
    assert!(m.manifest.mtp_present);
    assert_eq!(m.hparams.nextn_layers, 1);
    let eh = m
        .manifest
        .entries
        .iter()
        .find(|e| e.gguf_name == "blk.2.nextn.eh_proj.weight")
        .unwrap();
    assert_eq!(eh.weight.name, "mtp.eh_proj");
    assert_eq!(eh.weight.layer, None);

    // trunk-only file that declares an MTP block in metadata: admitted, no MTP.
    let trunk_only = write_fixture(&dir, &base_kvs("qwen35", 1), &base_tensors(), true);
    let m = GgufNativeModel::open_gated(&trunk_only, true).expect("trunk-only admits");
    assert!(!m.manifest.mtp_present);

    // half a block: refuses, naming the missing tensor.
    let mut half = base_tensors();
    half.extend(
        mtp_tensors()
            .into_iter()
            .filter(|t| t.name != "blk.2.nextn.enorm.weight"),
    );
    let p = write_fixture(&dir, &base_kvs("qwen35", 1), &half, true);
    let err = GgufNativeModel::open_gated(&p, true)
        .err()
        .expect("half MTP refuses");
    assert!(err.contains("blk.2.nextn.enorm.weight"), "{err}");
}

#[test]
fn env_unset_refuses_with_the_stable_prefix() {
    let dir = tempfile::tempdir().unwrap();
    let p = write_fixture(&dir, &base_kvs("qwen35", 0), &base_tensors(), true);
    let err = GgufNativeModel::open_gated(&p, false)
        .err()
        .expect("gate refuses");
    assert!(err.starts_with("load refused:"), "{err}");
    assert!(err.contains(GGUF_NATIVE_ENV), "{err}");
    assert!(err.contains("fixture-"), "{err}");
    // the gate is the only difference.
    assert!(GgufNativeModel::open_gated(&p, true).is_ok());
    assert!(require_gguf_native(true, "x").is_ok());
}

#[test]
fn unknown_type_id_fails_closed_at_parse() {
    let mut tensors = base_tensors();
    // 250 is no ggml type at all.
    tensors[5].ty = 250;
    let dir = tempfile::tempdir().unwrap();
    let p = write_fixture(&dir, &base_kvs("qwen35", 0), &tensors, true);
    let err = GgufNativeModel::open_gated(&p, true)
        .err()
        .expect("unknown id refuses");
    assert!(err.starts_with("load refused:"), "{err}");
    assert!(err.contains("unknown GGML type: 250"), "{err}");
}

#[test]
fn a_known_but_unserved_type_refuses_naming_type_and_tensor() {
    let mut tensors = base_tensors();
    let i = tensors
        .iter()
        .position(|t| t.name == "blk.1.ffn_gate.weight")
        .unwrap();
    tensors[i] = Spec::new("blk.1.ffn_gate.weight", &[DIM, FF], GgmlType::IQ2XS);
    expect_refusal(
        &tensors,
        &[
            "blk.1.ffn_gate.weight",
            "IQ2_XS",
            "ggml id 17",
            "Q8_0, IQ4_XS",
        ],
    );
    // Q4_0 and MXFP4 parse but are not tier 1.
    tensors[i] = Spec::new("blk.1.ffn_gate.weight", &[DIM, FF], GgmlType::Q4_0);
    expect_refusal(&tensors, &["blk.1.ffn_gate.weight", "Q4_0"]);
    tensors[i] = Spec::new("blk.1.ffn_gate.weight", &[DIM, FF], GgmlType::MXFP4);
    expect_refusal(&tensors, &["blk.1.ffn_gate.weight", "MXFP4"]);
}

#[test]
fn a_missing_tensor_refuses_naming_it() {
    let tensors: Vec<Spec> = base_tensors()
        .into_iter()
        .filter(|t| t.name != "blk.1.attn_k_norm.weight")
        .collect();
    let err = expect_refusal(&tensors, &["missing tensor 'blk.1.attn_k_norm.weight'"]);
    assert!(
        err.contains("self_attn.k_norm.weight"),
        "names the hipfire side too: {err}"
    );
    // a required global likewise.
    let tensors: Vec<Spec> = base_tensors()
        .into_iter()
        .filter(|t| t.name != "output_norm.weight")
        .collect();
    expect_refusal(&tensors, &["missing tensor 'output_norm.weight'"]);
}

#[test]
fn an_extra_tensor_refuses_naming_it() {
    let mut tensors = base_tensors();
    tensors.push(Spec::new("rope_freqs.weight", &[32], GgmlType::F32));
    expect_refusal(&tensors, &["rope_freqs.weight", "extra tensor"]);
    // a fused full-attention QKV is not the canonical export either.
    let mut tensors = base_tensors();
    tensors.push(Spec::new(
        "blk.1.attn_qkv.weight",
        &[DIM, 768],
        GgmlType::Q4K,
    ));
    expect_refusal(&tensors, &["blk.1.attn_qkv.weight", "extra tensor"]);
    // and a layer the config does not have.
    let mut tensors = base_tensors();
    tensors.push(Spec::new("blk.7.attn_norm.weight", &[DIM], GgmlType::F32));
    expect_refusal(&tensors, &["blk.7.attn_norm.weight", "extra tensor"]);
}

#[test]
fn a_row_that_is_not_whole_blocks_refuses() {
    let mut tensors = base_tensors();
    let i = tensors
        .iter()
        .position(|t| t.name == "blk.0.ssm_out.weight")
        .unwrap();
    // ne0 = 128 is whole Q8_0 blocks but not whole 256-element K-quant blocks.
    tensors[i] = Spec::new("blk.0.ssm_out.weight", &[128, DIM], GgmlType::Q4K);
    expect_refusal(
        &tensors,
        &["blk.0.ssm_out.weight", "Q4_K", "128", "256-element"],
    );
    // IQ4_XS has the same 256 requirement.
    tensors[i] = Spec::new("blk.0.ssm_out.weight", &[128, DIM], GgmlType::IQ4XS);
    expect_refusal(&tensors, &["blk.0.ssm_out.weight", "IQ4_XS", "256-element"]);
}

#[test]
fn a_wrong_shape_refuses_naming_the_tensor() {
    let mut tensors = base_tensors();
    let i = tensors
        .iter()
        .position(|t| t.name == "blk.1.attn_q.weight")
        .unwrap();
    // a plain Qwen3 q_proj (no output gate) is half the rows.
    tensors[i] = Spec::new("blk.1.attn_q.weight", &[DIM, 256], GgmlType::Q4K);
    expect_refusal(
        &tensors,
        &["blk.1.attn_q.weight", "[256, 256]", "[256, 512]"],
    );
}

#[test]
fn a_quantised_norm_is_refused() {
    let mut tensors = base_tensors();
    let i = tensors
        .iter()
        .position(|t| t.name == "blk.0.ssm_norm.weight")
        .unwrap();
    // 32 elements of Q8_0 is one valid block, but a norm must stay float.
    tensors[i] = Spec::new("blk.0.ssm_norm.weight", &[32], GgmlType::Q8_0);
    expect_refusal(
        &tensors,
        &["blk.0.ssm_norm.weight", "Q8_0", "F32, F16 or BF16"],
    );
}

#[test]
fn a_truncated_file_refuses_but_a_header_fixture_analyses() {
    let dir = tempfile::tempdir().unwrap();
    let p = write_fixture(&dir, &base_kvs("qwen35", 0), &base_tensors(), false);
    let err = GgufNativeModel::open_gated(&p, true)
        .err()
        .expect("header-only is truncated");
    assert!(err.contains("re-download"), "{err}");
    let g = hipfire_runtime::gguf::GgufFile::open_native(&p).unwrap();
    let a = analyze(&g, false);
    assert!(a.admitted(), "{:?}", a.refusals);
    assert_eq!(a.census.tensor_count, base_tensors().len());
    // inspect says so once instead of refusing every tensor.
    let r = inspect_report(&p, false).unwrap();
    assert!(r.contains("INCOMPLETE"), "{r}");
    assert!(r.contains("ADMITTED"), "{r}");
    assert!(!r.contains("download may be incomplete"), "{r}");
    let full = write_fixture(&dir, &base_kvs("qwen35", 0), &base_tensors(), true);
    assert!(inspect_report(&full, false)
        .unwrap()
        .contains("data:          complete"));
}

#[test]
fn architecture_is_resolved_through_the_shared_table() {
    let dir = tempfile::tempdir().unwrap();
    let check = |arch: &str, needles: &[&str]| {
        let p = write_fixture(&dir, &base_kvs(arch, 0), &base_tensors(), true);
        let err = GgufNativeModel::open_gated(&p, true)
            .err()
            .expect("refuses");
        assert!(err.starts_with("load refused:"), "{err}");
        for n in needles {
            assert!(err.contains(n), "{n:?} in {err}");
        }
    };
    // unknown strings fail closed (MoE export included: not in the table).
    check(
        "qwen35moe",
        &["'qwen35moe'", "not a model family hipfire knows"],
    );
    check("mamba", &["'mamba'", "not a model family hipfire knows"]);
    // known, but another family.
    check("llama", &["'llama'", "arch 0", "dense Qwen3.5"]);
    // the right arch id under a name the dense export never uses.
    check("qwen3_5", &["'qwen3_5'", "'qwen35'"]);
}

#[test]
fn inspect_prints_arch_histogram_bpw_and_verdict() {
    let dir = tempfile::tempdir().unwrap();
    let p = write_fixture(&dir, &base_kvs("qwen35", 0), &base_tensors(), true);
    let r = inspect_report(&p, false).unwrap();
    assert!(r.contains("qwen35 (hipfire arch 5)"), "{r}");
    assert!(
        r.contains(&format!("tensors:       {}", base_tensors().len())),
        "{r}"
    );
    assert!(
        r.contains("Q4_K") && r.contains("IQ4_XS") && r.contains("F32"),
        "{r}"
    );
    assert!(r.contains("model bpw:"), "{r}");
    assert!(r.contains("ADMITTED"), "{r}");
    assert!(r.contains("switch OFF"), "{r}");
    assert!(inspect_report(&p, true).unwrap().contains("switch ON"));

    // a refused file shows every problem, not just the first.
    let mut tensors = base_tensors();
    tensors[5] = Spec::new("blk.0.attn_qkv.weight", &[DIM, 256], GgmlType::IQ3S);
    tensors.push(Spec::new("stray.weight", &[32], GgmlType::F32));
    let p = write_fixture(&dir, &base_kvs("qwen35", 0), &tensors, true);
    let r = inspect_report(&p, true).unwrap();
    assert!(
        r.contains("REFUSED (2 problems: Extra=1, Type=1; mapping problems = 1)"),
        "{r}"
    );
    assert!(r.contains("IQ3_S") && r.contains("NOT admitted"), "{r}");
    assert!(r.contains("stray.weight"), "{r}");
}

/// Real-header check for the S2 kill criterion: set
/// `GGUF_TEST_HEADER=<path to a 27B .gguf or header-only dump>` and run
/// with `--ignored --nocapture`. The test fails when any tensor is unmapped
/// (missing, extra, mis-shaped or duplicated), naming it. Tensors of types
/// outside the admitted set are *reported* (a quantisation choice, not a
/// mapping failure); set `GGUF_TEST_EXPECT_ADMITTED=1` to require none.
#[test]
#[ignore]
fn real_27b_header_maps_every_tensor() {
    let path = std::env::var("GGUF_TEST_HEADER")
        .expect("set GGUF_TEST_HEADER to a 27B GGUF (or its header dump)");
    let g = hipfire_runtime::gguf::GgufFile::open_native(std::path::Path::new(&path))
        .expect("parse real header");
    let a = analyze(&g, false);
    println!("architecture: {:?} arch_id {:?}", a.architecture, a.arch_id);
    println!(
        "tensors {} bpw {:.4}  by type: {:#?}",
        a.census.tensor_count,
        a.census.bpw(),
        a.census.by_type
    );
    if let Some(hp) = &a.hparams {
        println!(
            "hparams: dim {} layers {} (+{} nextn) vocab {} ff {} heads {}/{} hd {} linear k/v heads {}/{} kd/vd {}/{}",
            hp.config.dim,
            hp.config.n_layers,
            hp.nextn_layers,
            hp.config.vocab_size,
            hp.config.hidden_dim,
            hp.config.n_heads,
            hp.config.n_kv_heads,
            hp.config.head_dim,
            hp.config.linear_num_key_heads,
            hp.config.linear_num_value_heads,
            hp.config.linear_key_head_dim,
            hp.config.linear_value_head_dim
        );
    }
    println!("refusals by kind: {:?}", a.refusal_counts());
    let mut by_type: std::collections::BTreeMap<String, Vec<&str>> = Default::default();
    for r in &a.refusals {
        if r.kind == RefusalKind::Type {
            let t = r.tensor.as_deref().unwrap_or("?");
            let ty = g.find_tensor(t).map_or("?", |i| i.dtype.name());
            by_type.entry(ty.to_string()).or_default().push(t);
        } else {
            println!("REFUSAL[{:?}]: {}", r.kind, r.message);
        }
    }
    for (ty, names) in &by_type {
        println!(
            "type refusal {ty}: {} tensors, e.g. {:?}",
            names.len(),
            &names[..names.len().min(3)]
        );
    }
    let mapping = a.mapping_refusals();
    assert!(
        mapping.is_empty(),
        "{} unmapped/mis-shaped tensors; first: {:?}",
        mapping.len(),
        mapping.first().map(|r| &r.message)
    );
    println!("every tensor maps ({} file tensors)", a.census.tensor_count);
    if std::env::var_os("GGUF_TEST_EXPECT_ADMITTED").is_some() {
        assert!(a.admitted(), "{} refusals", a.refusals.len());
        let m = a.manifest.as_ref().unwrap();
        println!(
            "manifest entries {} mtp_present {}",
            m.entries.len(),
            m.mtp_present
        );
    }
}
