// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Mechanical `KernelAbiV1` drift check (PM/railgun bedrock R1). CPU only: no GPU, no
//! toolchain process, no kernel launch.
//!
//! `hipfire-kernel-abi` states, per export, the explicit/hidden kernarg layout, the kernarg
//! segment, the block/LDS/grid and the object identity. This test compares those statements
//! with every independent source of the same facts that exists offline:
//!
//! * **Contract self-consistency** (`every_contract_self_checks`): fields/hidden words are
//!   sorted, disjoint, aligned and inside the segment, every pointer field has its operand,
//!   every expression resolves, LDS/block limits hold. Proves the contract is internally
//!   coherent; says nothing about any kernel.
//! * **Real object metadata** (`embedded_objects_match_contracts`): every committed `.hxaco`
//!   named by an `ObjectRefV1::Embedded` export is hashed (the SHA-256 must equal the pinned
//!   one), lifted with `peacemaker_lift::lift_object` (byte-exact round-trip proof) and the
//!   entry's AMDGPU metadata (args `name/offset/size/value_kind`, kernarg segment size/align,
//!   `group_segment_fixed_size`, `max_flat_workgroup_size`, wavefront size) is compared with
//!   the contract through `check_export`. A symbol missing from the object is a failure.
//!   Proves the contract describes the object that is loaded. Object metadata carries no
//!   dynamic LDS, grid or `actual_access` for these entries, so those are not checked here.
//! * **Generator specs** (`generators_match_contracts`): the PeaceMaker generator side
//!   (`peacemaker_isa` `mq6_i8_gemm`, `qwen4_moe_sym`, `qwen4_moe_sym_gfx11`; for the entries
//!   that expose no public spec, `qwen4_moe_down_gp` / `emit_gate_up_stream8`, the metadata
//!   block of the emitted assembly) states symbol, kernarg layout, threads, static LDS and, for
//!   MQ6, the dynamic LDS the launch must pass (`Tile::layout().launch`). Every embedded
//!   export must be matched by at least one generator entry and every matching entry must
//!   agree with the contract; an export with no generator path must be listed in
//!   [`NO_GENERATOR_PATH`] with a reason, and that list is asserted exact. Proves the contract
//!   agrees with what the generator would emit (and, for MQ6, with the dynamic LDS the object
//!   metadata cannot show).
//! * **Hipcc exports** (`hipcc_exports_are_listed`): `HipccSource` exports have no committed
//!   object. Their `(arch, symbol)` set must equal an explicit list, so a new export cannot
//!   appear silently.
//! * **Hipcc compiled metadata** (`hipcc_exports_match_compiled_metadata`): each hipcc source is
//!   compiled for its arch with the runtime `KernelCompiler` (CPU only, no device), the AMDGPU
//!   metadata of the object is read without decoding and compared with the contract through
//!   `check_export`. Skipped with a printed reason when no hipcc resolves.
//! * **Deliberately broken fixtures** (`broken_*`): each corrupts one fact of real evidence or
//!   of a leaked copy of a real contract and must fail with the named `Drift` / `AbiError`,
//!   proving the check is live and not vacuously green. The recipe-interchange fixture runs
//!   `validate_invocation` on the REAL contracts, i.e. the rejection happens before enqueue.
//!
//! What none of this proves: numeric or semantic equivalence of any kernel (the
//! `NumericContractId`s are identities, not proofs); that a hipcc twin's code (as opposed to its
//! kernarg metadata) matches its contract; the content of weight slabs or activation planes; or that the launcher
//! calls the contract (the launcher legs are tested in their own crates).

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use hipfire_kernel_abi::families::{
    self, ggml_convert as convert_abi, ggml_mmvq as ggml_abi, ggml_mmvq_fused as ggml_fused,
    mq6_i8, qwen4_moe_sym as moe_abi,
};
use hipfire_kernel_abi::{
    check_export, validate_invocation, AbiError, Arch, ArgEvidence, Drift, EvidenceSource,
    ExportEvidence, ExportV1, Expr, FormatId, InvocationV1, KernelAbiV1, ObjectRefV1,
    OperandBinding, Rule,
};
use peacemaker_ir::envelope::Source as IrSource;
use peacemaker_ir::inst::{Abi, Arch as IrArch, Frontend};
use peacemaker_ir::metadata::KernelMeta;
use peacemaker_ir::Program;
use peacemaker_isa::kernels::mq6_i8_gemm::{
    self as mq6_gen,
    spec::{Epi, Spec as Mq6Spec, Tile},
};
use peacemaker_isa::kernels::qwen4_moe_sym::{self as moe_gen, Kind, Spec as MoeSpec, WeightCodec};
use peacemaker_isa::kernels::qwen4_moe_sym_gfx11::{
    self as gfx11_gen, Codec as Gfx11Codec, Negative as Gfx11Negative, Spec as Gfx11Spec,
};
use peacemaker_isa::{Arch as IsaArch, KernargLayout};
use peacemaker_lift::bundle::SourceCodec;
use sha2::{Digest, Sha256};

/// Embedded exports with no PeaceMaker generator path, as `(arch, symbol, reason)`. Exact: an
/// export without a generator that is not listed here fails, and a listed export that gains a
/// generator (or disappears) fails. Every embedded export currently has one.
const NO_GENERATOR_PATH: &[(&str, &str, &str)] = &[
    ("gfx1201", "ggml_mmvq_iq4_xs_gfx1201", GGML_SHIM_REASON),
    ("gfx1201", "ggml_mmvq_q4_k_gfx1201", GGML_SHIM_REASON),
    ("gfx1201", "ggml_mmvq_q6_k_gfx1201", GGML_SHIM_REASON),
    ("gfx1201", "ggml_mmvq_q8_0_gfx1201", GGML_SHIM_REASON),
    ("gfx1201", "ggml_q8_1_quantize_gfx1201", GGML_SHIM_REASON),
    ("gfx1201", "ggml_mmvq_q5_k_gfx1201", GGML_FUSED_SHIM_REASON),
    (
        "gfx1201",
        "ggml_mmvq_iq4_xs_res_gfx1201",
        GGML_FUSED_SHIM_REASON,
    ),
    (
        "gfx1201",
        "ggml_mmvq_q4_k_res_gfx1201",
        GGML_FUSED_SHIM_REASON,
    ),
    (
        "gfx1201",
        "ggml_mmvq_q5_k_res_gfx1201",
        GGML_FUSED_SHIM_REASON,
    ),
    (
        "gfx1201",
        "ggml_mmvq_q6_k_res_gfx1201",
        GGML_FUSED_SHIM_REASON,
    ),
    (
        "gfx1201",
        "ggml_mmvq_q8_0_res_gfx1201",
        GGML_FUSED_SHIM_REASON,
    ),
    (
        "gfx1201",
        "ggml_q8_1_quantize_gather_gfx1201",
        GGML_FUSED_SHIM_REASON,
    ),
    (
        "gfx1201",
        "ggml_q81_roundtrip_gfx1201",
        GGML_CONVERT_SHIM_REASON,
    ),
    (
        "gfx1201",
        "ggml_dequant_iq4_xs_f16_gfx1201",
        GGML_CONVERT_SHIM_REASON,
    ),
    (
        "gfx1201",
        "ggml_dequant_iq4_xs_f32_gfx1201",
        GGML_CONVERT_SHIM_REASON,
    ),
    (
        "gfx1201",
        "ggml_get_row_iq4_xs_f32_gfx1201",
        GGML_CONVERT_SHIM_REASON,
    ),
    (
        "gfx1201",
        "ggml_get_row1_iq4_xs_f32_gfx1201",
        GGML_CONVERT_SHIM_REASON,
    ),
    (
        "gfx1201",
        "ggml_dequant_q4_k_f16_gfx1201",
        GGML_CONVERT_SHIM_REASON,
    ),
    (
        "gfx1201",
        "ggml_dequant_q4_k_f32_gfx1201",
        GGML_CONVERT_SHIM_REASON,
    ),
    (
        "gfx1201",
        "ggml_get_row_q4_k_f32_gfx1201",
        GGML_CONVERT_SHIM_REASON,
    ),
    (
        "gfx1201",
        "ggml_get_row1_q4_k_f32_gfx1201",
        GGML_CONVERT_SHIM_REASON,
    ),
    (
        "gfx1201",
        "ggml_dequant_q5_k_f16_gfx1201",
        GGML_CONVERT_SHIM_REASON,
    ),
    (
        "gfx1201",
        "ggml_dequant_q5_k_f32_gfx1201",
        GGML_CONVERT_SHIM_REASON,
    ),
    (
        "gfx1201",
        "ggml_get_row_q5_k_f32_gfx1201",
        GGML_CONVERT_SHIM_REASON,
    ),
    (
        "gfx1201",
        "ggml_get_row1_q5_k_f32_gfx1201",
        GGML_CONVERT_SHIM_REASON,
    ),
    (
        "gfx1201",
        "ggml_dequant_q6_k_f16_gfx1201",
        GGML_CONVERT_SHIM_REASON,
    ),
    (
        "gfx1201",
        "ggml_dequant_q6_k_f32_gfx1201",
        GGML_CONVERT_SHIM_REASON,
    ),
    (
        "gfx1201",
        "ggml_get_row_q6_k_f32_gfx1201",
        GGML_CONVERT_SHIM_REASON,
    ),
    (
        "gfx1201",
        "ggml_get_row1_q6_k_f32_gfx1201",
        GGML_CONVERT_SHIM_REASON,
    ),
    (
        "gfx1201",
        "ggml_dequant_q8_0_f16_gfx1201",
        GGML_CONVERT_SHIM_REASON,
    ),
    (
        "gfx1201",
        "ggml_dequant_q8_0_f32_gfx1201",
        GGML_CONVERT_SHIM_REASON,
    ),
    (
        "gfx1201",
        "ggml_get_row_q8_0_f32_gfx1201",
        GGML_CONVERT_SHIM_REASON,
    ),
    (
        "gfx1201",
        "ggml_get_row1_q8_0_f32_gfx1201",
        GGML_CONVERT_SHIM_REASON,
    ),
];

/// The ggml shim is hipcc-compiled from `kernels/src/ggml_mmvq_shim.hip`; PeaceMaker emits
/// nothing for it, so its only offline evidence is the committed object's metadata.
const GGML_SHIM_REASON: &str =
    "hipcc-built from third_party/ggml upstream device code; no PeaceMaker generator";

/// The S3 additions come from `kernels/src/ggml_mmvq_fused_shim.hip`, the same way.
const GGML_FUSED_SHIM_REASON: &str =
    "hipcc-built from third_party/ggml upstream device code (fused shim); no PeaceMaker generator";

/// The S3 convert exports come from `kernels/src/ggml_convert_shim.hip`, the same way.
const GGML_CONVERT_SHIM_REASON: &str =
    "hipcc-built from third_party/ggml upstream device code (convert shim); no PeaceMaker generator";

/// Every `HipccSource` export `(arch, symbol)`: no offline object, so no metadata evidence.
const EXPECTED_HIPCC_EXPORTS: &[(&str, &str)] = &[
    ("gfx1100", "bf16_round_trip_f32_strided"),
    ("gfx1100", "convert_f32_to_f16"),
    ("gfx1100", "embedding_q8_batched"),
    ("gfx1100", "ffn_moe_combine_f32"),
    ("gfx1100", "ffn_moe_router_topk_f32"),
    ("gfx1100", "ffn_round_bf16_f32"),
    ("gfx1100", "ffn_single_expert_tables"),
    ("gfx1100", "ffn_situ_grouped_f32"),
    ("gfx1100", "ffn_situ_rows_f32"),
    ("gfx1100", "gemm_bf16_xf32_batched"),
    ("gfx1100", "gemm_f32_batched"),
    ("gfx1100", "gemm_mq4g256v2_moe_grouped_wmma_k2"),
    ("gfx1100", "gemm_q8_0_wmma"),
    ("gfx1100", "gemv_q8_0"),
    ("gfx1100", "moe_scatter_fused_k8"),
    ("gfx1100", "mq_rotate_x"),
    ("gfx1100", "rotate_x_mq_awq"),
    ("gfx1151", "gemm_mq6g256v2_mmq_i8_set_bf16_gfx11"),
    ("gfx1151", "gemm_mq6g256v2_mmq_i8_set_bf16_gfx11_t256"),
    ("gfx1151", "gemm_mq6g256v2_mmq_i8_set_gfx11"),
    ("gfx1151", "gemm_mq6g256v2_mmq_i8_set_gfx11_t256"),
    ("gfx1151", "qwen4_moe_down_iu4_sym_gfx1151_nt4"),
    ("gfx1151", "qwen4_moe_gate_up_silu_iu4_sym_gfx1151_nt4"),
    ("gfx1201", "bf16_round_trip_f32_strided"),
    ("gfx1201", "causal_conv_depthwise_silu_rows_f32"),
    ("gfx1201", "convert_f32_to_f16"),
    ("gfx1201", "embedding_q8_batched"),
    ("gfx1201", "ffn_moe_combine_f32"),
    ("gfx1201", "ffn_moe_router_topk_f32"),
    ("gfx1201", "ffn_round_bf16_f32"),
    ("gfx1201", "ffn_single_expert_tables"),
    ("gfx1201", "ffn_situ_grouped_f32"),
    ("gfx1201", "ffn_situ_rows_f32"),
    ("gfx1201", "gated_delta_net_vector_serial_f32"),
    ("gfx1201", "gated_rmsnorm_sigmoid_rows_f32"),
    ("gfx1201", "gemm_bf16_xf32_batched"),
    ("gfx1201", "gemm_bf16_xf32_tiled_rows"),
    ("gfx1201", "gemm_bf16_xf32_tiled_rows_mp4"),
    ("gfx1201", "gemm_f32_batched"),
    ("gfx1201", "gemm_mq4g256v2_moe_grouped_wmma_gfx12"),
    ("gfx1201", "gemm_mq6g256v2_mmq_i8_set_bf16_gfx12"),
    ("gfx1201", "gemm_mq6g256v2_mmq_i8_set_gfx12"),
    ("gfx1201", "gemm_q8_0_wmma_gfx12"),
    ("gfx1201", "gemv_q8_0"),
    ("gfx1201", "latent_attention_absorb_pack_bf16"),
    ("gfx1201", "latent_attention_flash_prefill"),
    ("gfx1201", "latent_attention_gate_bf16"),
    ("gfx1201", "latent_attention_gather_latent_f32"),
    ("gfx1201", "latent_attention_rms_rows_bf16"),
    ("gfx1201", "latent_attention_scatter_expanded_bf16"),
    ("gfx1201", "latent_attention_split_kv_bf16"),
    ("gfx1201", "moe_grouped_qt44_a8_sym_gfx1201_nt1r1"),
    ("gfx1201", "moe_grouped_qt44_a8_sym_gfx1201_nt1r2"),
    ("gfx1201", "moe_grouped_qt44_a8_sym_gfx1201_nt2r1"),
    ("gfx1201", "moe_grouped_qt44_a8_sym_gfx1201_nt2r2"),
    ("gfx1201", "moe_grouped_qt44_iu4_sym_gfx1201_nt1r1"),
    ("gfx1201", "moe_grouped_qt44_iu4_sym_gfx1201_nt1r2"),
    ("gfx1201", "moe_grouped_qt44_iu4_sym_gfx1201_nt2r1"),
    ("gfx1201", "moe_grouped_qt44_iu4_sym_gfx1201_nt2r2"),
    ("gfx1201", "moe_grouped_qt44_iu4_sym_gfx1201_nt4r1"),
    ("gfx1201", "moe_scatter_fused_k8"),
    ("gfx1201", "mq_rotate_x"),
    ("gfx1201", "raon_rda_prepare_f32"),
    ("gfx1201", "rotate_x_mq_awq"),
    ("gfx1201", "stream_mix_aggregate"),
    ("gfx1201", "stream_mix_broadcast"),
    ("gfx1201", "stream_mix_mean"),
    ("gfx1201", "stream_mix_postmix"),
    ("gfx1201", "stream_mix_route_helmert"),
];

// ---- shared helpers ------------------------------------------------------------------

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn real_export(abi: &'static KernelAbiV1, arch: Arch, symbol: &str) -> &'static ExportV1 {
    abi.export(arch, symbol)
        .unwrap_or_else(|| panic!("{}: no export {symbol} for {}", abi.id, arch.name()))
}

/// A `'static` copy of `base` with `exports` replaced (test-only leak).
fn leak_with_exports(base: &KernelAbiV1, exports: Vec<ExportV1>) -> &'static KernelAbiV1 {
    let mut abi = *base;
    abi.exports = Box::leak(exports.into_boxed_slice());
    Box::leak(Box::new(abi))
}

fn drift_lines(abi: &KernelAbiV1, drifts: &[Drift]) -> String {
    drifts
        .iter()
        .map(|d| format!("[{}] {d}", abi.id))
        .collect::<Vec<_>>()
        .join("\n")
}

fn rejected(abi: &KernelAbiV1, inv: &InvocationV1<'_>) -> AbiError {
    match validate_invocation(abi, inv) {
        Ok(_) => panic!(
            "{}: {} was admitted, expected a rejection",
            abi.id, inv.symbol
        ),
        Err(e) => e,
    }
}

fn admitted(abi: &KernelAbiV1, inv: &InvocationV1<'_>) {
    if let Err(e) = validate_invocation(abi, inv) {
        panic!("{}: {} was rejected: {e}", abi.id, inv.symbol);
    }
}

// ---- object evidence (real metadata of the committed objects) ------------------------

struct LiftedObject {
    sha256: String,
    program: Program,
}

/// Every artifact named by an embedded export, read, hashed and lifted once.
static OBJECTS: LazyLock<BTreeMap<&'static str, LiftedObject>> = LazyLock::new(|| {
    let mut artifacts = BTreeSet::new();
    for abi in families::all() {
        for export in abi.exports {
            if let ObjectRefV1::Embedded { artifact, .. } = export.object {
                artifacts.insert(artifact);
            }
        }
    }
    artifacts
        .into_iter()
        .map(|artifact| {
            let path = repo_root().join(artifact);
            let bytes = std::fs::read(&path).unwrap_or_else(|e| {
                panic!("{}: cannot read the committed object: {e}", path.display())
            });
            let lifted = peacemaker_lift::lift_object(
                &bytes,
                peacemaker_lift::Options {
                    frontend: Frontend::Builder,
                },
            )
            .unwrap_or_else(|e| panic!("{artifact}: lift_object failed: {e}"));
            (
                artifact,
                LiftedObject {
                    sha256: hex(&Sha256::digest(&bytes)),
                    program: lifted.program,
                },
            )
        })
        .collect()
});

/// Name of the contract's explicit field at `offset` (empty if there is none, so a metadata
/// arg the contract lacks still drifts).
fn unnamed_arg_name(export: &ExportV1, offset: u32) -> String {
    families::find_export(export.arch, export.symbol)
        .and_then(|(abi, _)| abi.fields.iter().find(|f| f.offset == offset))
        .map(|f| f.name.to_string())
        .unwrap_or_default()
}

/// `ExportEvidence` of `export` from the real metadata of its committed object.
fn object_evidence(export: &ExportV1) -> Result<ExportEvidence, String> {
    let ObjectRefV1::Embedded { artifact, .. } = export.object else {
        return Err(format!("{}: not an embedded export", export.symbol));
    };
    let object = OBJECTS
        .get(artifact)
        .ok_or_else(|| format!("{artifact}: object was not lifted"))?;
    program_evidence(
        export,
        &program_meta(&object.program),
        artifact,
        Some(object.sha256.clone()),
    )
}

/// The per-kernel AMDGPU metadata of one code object (`None`: the entry has no HSA metadata).
struct ObjectMeta<'a> {
    arch: IrArch,
    kernels: Vec<(String, Option<&'a KernelMeta>)>,
}

fn program_meta(program: &Program) -> ObjectMeta<'_> {
    ObjectMeta {
        arch: program.target.arch,
        kernels: program
            .kernels
            .iter()
            .map(|k| {
                let name = k
                    .symbol
                    .0
                    .strip_suffix(".kd")
                    .unwrap_or(k.symbol.0.as_str());
                let meta = match &k.abi {
                    Abi::Hsa { metadata, .. } => Some(&metadata.parsed),
                    _ => None,
                };
                (name.to_string(), meta)
            })
            .collect(),
    }
}

/// `ExportEvidence` of `export` from the AMDGPU metadata of `object` (named `artifact` in
/// messages); `sha256` is the object identity to attach, if any.
fn program_evidence(
    export: &ExportV1,
    object: &ObjectMeta<'_>,
    artifact: &str,
    sha256: Option<String>,
) -> Result<ExportEvidence, String> {
    let (_, meta) = object
        .kernels
        .iter()
        .find(|(name, _)| name == export.symbol)
        .ok_or_else(|| {
            format!(
                "{}: symbol is missing from {artifact} (kernels: {:?})",
                export.symbol,
                object
                    .kernels
                    .iter()
                    .map(|(n, _)| n.as_str())
                    .collect::<Vec<_>>()
            )
        })?;
    let Some(meta) = meta else {
        return Err(format!(
            "{}: {artifact} entry carries no HSA metadata",
            export.symbol
        ));
    };
    let arch = match object.arch {
        IrArch::Gfx1100 => Arch::Gfx1100,
        IrArch::Gfx1151 => Arch::Gfx1151,
        IrArch::Gfx1201 => Arch::Gfx1201,
        other => return Err(format!("{artifact}: unexpected target {other:?}")),
    };
    Ok(ExportEvidence {
        source: EvidenceSource::ObjectMetadata,
        arch: Some(arch),
        symbol: export.symbol.to_string(),
        args: meta
            .args
            .iter()
            .map(|a| ArgEvidence {
                // hipcc-built objects (the ggml shim) carry no `.name`: the contract's field
                // name at the same offset stands in; offset, size and kind are still compared.
                name: if a.name.is_empty() {
                    unnamed_arg_name(export, a.offset)
                } else {
                    a.name.clone()
                },
                offset: a.offset,
                size: a.size,
                value_kind: a.value_kind.clone(),
                access: None,
            })
            .collect(),
        segment_bytes: Some(meta.kernarg_segment_size),
        segment_align: Some(meta.kernarg_segment_align),
        max_flat_workgroup_size: Some(meta.max_flat_workgroup_size),
        threads: None,
        dynamic_lds: None,
        group_segment_fixed: Some(meta.group_segment_fixed_size),
        wave: Some(meta.wavefront_size),
        object_sha256: sha256,
    })
}

fn object_evidence_ok(export: &ExportV1) -> ExportEvidence {
    object_evidence(export).unwrap_or_else(|e| panic!("{e}"))
}

// ---- generator evidence (PeaceMaker generator side) ----------------------------------

struct GenEntry {
    arch: Arch,
    symbol: String,
    args: Vec<ArgEvidence>,
    segment_bytes: u32,
    threads: u32,
    /// Dynamic LDS the launch passes, where the generator states one (MQ6 only: the MoE
    /// generators have static LDS only).
    dynamic_lds: Option<u32>,
    group_segment_fixed: u32,
    wave: Option<u32>,
    max_flat: Option<u32>,
    origin: String,
}

impl GenEntry {
    fn evidence(&self) -> ExportEvidence {
        ExportEvidence {
            source: EvidenceSource::Generator,
            arch: Some(self.arch),
            symbol: self.symbol.clone(),
            args: self.args.clone(),
            segment_bytes: Some(self.segment_bytes),
            segment_align: None,
            max_flat_workgroup_size: self.max_flat,
            threads: Some(self.threads),
            dynamic_lds: self.dynamic_lds,
            group_segment_fixed: Some(self.group_segment_fixed),
            wave: self.wave,
            object_sha256: None,
        }
    }
}

fn abi_arch(arch: IsaArch) -> Arch {
    match arch {
        IsaArch::Gfx1100 => Arch::Gfx1100,
        IsaArch::Gfx1151 => Arch::Gfx1151,
        IsaArch::Gfx1201 => Arch::Gfx1201,
        other => panic!("generator arch {} has no KernelAbiV1 arch", other.name()),
    }
}

fn layout_args(layout: &KernargLayout) -> Vec<ArgEvidence> {
    layout
        .args
        .iter()
        .map(|a| ArgEvidence {
            name: a.name.clone(),
            offset: a.offset,
            size: a.size,
            value_kind: a.value_kind.clone(),
            access: a.actual_access.clone(),
        })
        .collect()
}

#[derive(Default)]
struct EmittedMeta {
    symbol: String,
    args: Vec<ArgEvidence>,
    segment_bytes: u32,
    group_segment_fixed: u32,
    max_flat: u32,
    wave: u32,
}

fn blank_arg() -> ArgEvidence {
    ArgEvidence {
        name: String::new(),
        offset: 0,
        size: 0,
        value_kind: String::new(),
        access: None,
    }
}

fn number(key: &str, value: &str) -> u32 {
    value
        .parse::<u32>()
        .unwrap_or_else(|e| panic!("emitted metadata {key}: {value:?}: {e}"))
}

/// The `.amdgpu_metadata` block the generator writes into its assembly (`emit::assembly`):
/// args at indent 6/8 (`- ` opens an item at indent 6), kernel keys at indent 4.
fn parse_emitted_metadata(s_text: &str) -> EmittedMeta {
    let mut meta = EmittedMeta::default();
    let mut args = Vec::new();
    let mut cur: Option<ArgEvidence> = None;
    let mut in_metadata = false;
    for line in s_text.lines() {
        let rest = line.trim_start();
        if rest == ".amdgpu_metadata" {
            in_metadata = true;
            continue;
        }
        if !in_metadata {
            continue;
        }
        if rest == ".end_amdgpu_metadata" {
            break;
        }
        if rest.is_empty() || rest == "---" || rest == "..." {
            continue;
        }
        let indent = line.len() - rest.len();
        let (is_item, body) = match rest.strip_prefix('-') {
            Some(b) => (true, b.trim()),
            None => (false, rest.trim_end()),
        };
        if indent == 6 && is_item {
            if let Some(a) = cur.take() {
                args.push(a);
            }
            cur = Some(blank_arg());
        }
        let Some((key, value)) = body.split_once(':') else {
            continue;
        };
        let value = value.trim();
        match indent {
            6 | 8 => {
                if let Some(a) = cur.as_mut() {
                    match key {
                        ".actual_access" => a.access = Some(value.to_string()),
                        ".name" => a.name = value.to_string(),
                        ".offset" => a.offset = number(key, value),
                        ".size" => a.size = number(key, value),
                        ".value_kind" => a.value_kind = value.to_string(),
                        _ => {}
                    }
                }
            }
            4 => {
                if let Some(a) = cur.take() {
                    args.push(a);
                }
                match key {
                    ".group_segment_fixed_size" => meta.group_segment_fixed = number(key, value),
                    ".kernarg_segment_size" => meta.segment_bytes = number(key, value),
                    ".max_flat_workgroup_size" => meta.max_flat = number(key, value),
                    ".wavefront_size" => meta.wave = number(key, value),
                    ".name" => meta.symbol = value.to_string(),
                    _ => {}
                }
            }
            _ => {}
        }
    }
    if let Some(a) = cur.take() {
        args.push(a);
    }
    assert!(
        !meta.symbol.is_empty(),
        "emitted assembly has no kernel metadata:\n{s_text}"
    );
    assert!(
        !args.is_empty(),
        "{}: emitted metadata has no args",
        meta.symbol
    );
    meta.args = args;
    meta
}

fn moe_entry(spec: MoeSpec) -> GenEntry {
    let layout = spec.kernargs();
    GenEntry {
        arch: abi_arch(spec.arch),
        symbol: spec.symbol(),
        args: layout_args(&layout),
        segment_bytes: layout.size,
        threads: spec.threads(),
        dynamic_lds: None,
        group_segment_fixed: spec.lds_bytes(),
        wave: None,
        max_flat: None,
        origin: format!("qwen4_moe_sym::Spec {}", spec.variant()),
    }
}

fn gfx11_entry(spec: Gfx11Spec) -> GenEntry {
    let layout = spec.kernargs();
    GenEntry {
        arch: abi_arch(spec.arch),
        symbol: spec.symbol(),
        args: layout_args(&layout),
        segment_bytes: layout.size,
        threads: spec.threads(),
        dynamic_lds: None,
        group_segment_fixed: spec.lds_bytes(),
        wave: None,
        max_flat: None,
        origin: format!("qwen4_moe_sym_gfx11::Spec {}", spec.variant()),
    }
}

/// An entry with no public spec-level header (the grouped-position down twins, the stream8
/// gate/up): symbol, args, segment, LDS and workgroup size come from the emitted assembly's
/// metadata block; `threads` from the base spec.
fn emitted_entry(emitted: &peacemaker_isa::Emitted, spec: MoeSpec, what: &str) -> GenEntry {
    let meta = parse_emitted_metadata(&emitted.s_text);
    GenEntry {
        arch: abi_arch(spec.arch),
        symbol: meta.symbol,
        args: meta.args,
        segment_bytes: meta.segment_bytes,
        threads: spec.threads(),
        dynamic_lds: None,
        group_segment_fixed: meta.group_segment_fixed,
        wave: Some(meta.wave),
        max_flat: Some(meta.max_flat),
        origin: format!("emitted assembly of {what}"),
    }
}

static GENERATORS: LazyLock<Vec<GenEntry>> = LazyLock::new(|| {
    let mut entries = Vec::new();

    // MQ6: the product module (`mq6_i8_gemm::emit_module`): both epilogues on the shipped and
    // the 128x256 tile, then the QSA norm+RoPE entry on 256x128. The kernel header the emitter
    // itself uses (`Gen::kernel_spec`) is the evidence; the dynamic LDS is the tile's launch
    // figure.
    for (tile, epi) in [
        (Tile::T128x128, Epi::Set),
        (Tile::T128x128, Epi::SetBf16),
        (Tile::T128x256, Epi::Set),
        (Tile::T128x256, Epi::SetBf16),
        (Tile::T256x128, Epi::SetNr),
    ] {
        let spec = Mq6Spec::product(epi).with_tile(tile);
        let generator = mq6_gen::Gen::new(spec)
            .unwrap_or_else(|e| panic!("mq6_i8_gemm {}: {e}", spec.variant_name()));
        let ks = generator.kernel_spec();
        assert_eq!(ks.symbol, spec.symbol(), "{}", spec.variant_name());
        entries.push(GenEntry {
            arch: abi_arch(ks.arch),
            symbol: ks.symbol.clone(),
            args: layout_args(&ks.kernargs),
            segment_bytes: ks.kernargs.size,
            threads: u32::from(ks.workgroup_size),
            dynamic_lds: Some(tile.layout().launch),
            group_segment_fixed: ks.group_segment_fixed_size,
            wave: Some(if ks.wave32 { 32 } else { 64 }),
            max_flat: None,
            origin: format!("mq6_i8_gemm::Gen {}", spec.variant_name()),
        });
    }

    // MoE: the gfx1151/gfx1201 modules of every codec (MQ4, MQ3-I4 W3, E8P-I4).
    for arch in [IsaArch::Gfx1151, IsaArch::Gfx1201] {
        for codec in [WeightCodec::Mq4Sym, WeightCodec::Mq3I4, WeightCodec::E8pI4] {
            for spec in moe_gen::module_specs_codec(arch, codec) {
                entries.push(moe_entry(spec));
            }
        }
    }
    // MoE: the gfx11 NT1 module (gfx1100 MQ4 + E8P, gfx1151 E8P).
    for arch in gfx11_gen::ARCHES {
        for codec in [Gfx11Codec::Mq4, Gfx11Codec::E8p] {
            for kind in Kind::ALL {
                let spec = Gfx11Spec {
                    arch,
                    codec,
                    kind,
                    acc: false,
                    negative: Gfx11Negative::None,
                };
                if spec.validate().is_ok() {
                    entries.push(gfx11_entry(spec));
                }
            }
        }
    }
    // MoE: grouped-position down twins (`qwen4_moe_down_gp`) and the stream8 gate/up.
    for nt in [4u8, 8u8] {
        let spec = MoeSpec {
            arch: IsaArch::Gfx1201,
            kind: Kind::Down,
            nt,
            rr: 1,
            codec: WeightCodec::Mq4Sym,
        };
        let emitted = moe_gen::emit_down_gp(spec)
            .unwrap_or_else(|e| panic!("qwen4_moe_sym::emit_down_gp nt{nt}: {e}"));
        entries.push(emitted_entry(
            &emitted,
            spec,
            &format!("emit_down_gp nt{nt}"),
        ));
    }
    let stream8 = MoeSpec {
        arch: IsaArch::Gfx1151,
        kind: Kind::GateUp,
        nt: 8,
        rr: 1,
        codec: WeightCodec::Mq4Sym,
    };
    let emitted = moe_gen::emit_gate_up_stream8(IsaArch::Gfx1151)
        .unwrap_or_else(|e| panic!("qwen4_moe_sym::emit_gate_up_stream8: {e}"));
    entries.push(emitted_entry(&emitted, stream8, "emit_gate_up_stream8"));
    entries
});

fn generator_for(arch: Arch, symbol: &str) -> &'static GenEntry {
    GENERATORS
        .iter()
        .find(|g| g.arch == arch && g.symbol == symbol)
        .unwrap_or_else(|| panic!("no generator entry for {} {symbol}", arch.name()))
}

// ---- legs ----------------------------------------------------------------------------

#[test]
fn every_contract_self_checks() {
    let mut failures = Vec::new();
    let mut contracts = 0;
    for abi in families::all() {
        contracts += 1;
        if let Err(errors) = abi.self_check() {
            failures.push(format!("{}: {errors:#?}", abi.id));
        }
    }
    assert!(contracts > 0, "families::all() is empty");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn embedded_objects_match_contracts() {
    let mut failures = Vec::new();
    let mut checked = 0usize;
    for abi in families::all() {
        for export in abi.exports {
            let ObjectRefV1::Embedded {
                artifact, sha256, ..
            } = export.object
            else {
                continue;
            };
            let evidence = match object_evidence(export) {
                Ok(e) => e,
                Err(e) => {
                    failures.push(format!("[{}] {e}", abi.id));
                    continue;
                }
            };
            checked += 1;
            let file = &OBJECTS[artifact];
            if !file.sha256.eq_ignore_ascii_case(sha256) {
                failures.push(format!(
                    "[{}] {}: {artifact} sha256 is {}, the contract pins {sha256}",
                    abi.id, export.symbol, file.sha256
                ));
            }
            let drifts = check_export(abi, export, &evidence);
            if !drifts.is_empty() {
                failures.push(drift_lines(abi, &drifts));
            }
        }
    }
    assert!(checked > 0, "no embedded export was checked");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn generators_match_contracts() {
    let entries = &*GENERATORS;
    let mut failures = Vec::new();
    let mut unmatched: BTreeSet<(String, String)> = BTreeSet::new();
    let mut matched = 0usize;
    for abi in families::all() {
        for export in abi.exports {
            if !matches!(export.object, ObjectRefV1::Embedded { .. }) {
                continue;
            }
            let hits: Vec<&GenEntry> = entries
                .iter()
                .filter(|g| g.arch == export.arch && g.symbol == export.symbol)
                .collect();
            if hits.is_empty() {
                unmatched.insert((export.arch.name().to_string(), export.symbol.to_string()));
                continue;
            }
            matched += 1;
            for g in hits {
                let drifts = check_export(abi, export, &g.evidence());
                for d in drifts {
                    failures.push(format!("[{}] ({}) {d}", abi.id, g.origin));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert!(matched > 0, "no embedded export was matched by a generator");

    for (_, symbol, reason) in NO_GENERATOR_PATH {
        assert!(
            !reason.trim().is_empty(),
            "{symbol}: allowlist entry has no reason"
        );
    }
    let allowed: BTreeSet<(String, String)> = NO_GENERATOR_PATH
        .iter()
        .map(|(arch, symbol, _)| (arch.to_string(), symbol.to_string()))
        .collect();
    assert_eq!(
        unmatched, allowed,
        "embedded exports without a generator path must be exactly NO_GENERATOR_PATH"
    );
}

#[test]
fn hipcc_exports_are_listed() {
    let mut got: BTreeSet<(String, String)> = BTreeSet::new();
    for abi in families::all() {
        for export in abi.exports {
            if let ObjectRefV1::HipccSource { source, .. } = export.object {
                assert!(
                    repo_root().join(source).is_file(),
                    "{}: hipcc source {source} does not exist",
                    export.symbol
                );
                let (found_abi, found) = families::find_export(export.arch, export.symbol)
                    .unwrap_or_else(|| panic!("{}: find_export misses it", export.symbol));
                assert_eq!(found_abi.id, abi.id);
                assert_eq!(found.symbol, export.symbol);
                got.insert((export.arch.name().to_string(), export.symbol.to_string()));
            }
        }
    }
    let want: BTreeSet<(String, String)> = EXPECTED_HIPCC_EXPORTS
        .iter()
        .map(|(a, s)| (a.to_string(), s.to_string()))
        .collect();
    assert_eq!(
        got, want,
        "hipcc exports have no offline object: a new one must be added to EXPECTED_HIPCC_EXPORTS"
    );
}

// ---- deliberately broken fixtures ----------------------------------------------------

const MQ6_SET: &str = "gemm_mq6g256v2_mmq_i8_pm_set_gfx1201";
const MQ6_SET_T128X256: &str = "gemm_mq6g256v2_mmq_i8_pm_set_gfx1201_t128x256";
const MQ6_BF16: &str = "gemm_mq6g256v2_mmq_i8_pm_set_bf16_gfx1201";
const MOE_GATE_UP_NT4: &str = "qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1201_nt4";
const MOE_DOWN_NT4: &str = "qwen4_moe_down_iu4_sym_pm_gfx1201_nt4";
const MOE_DOWN_GP_NT4: &str = "qwen4_moe_down_iu4_sym_gp_pm_gfx1201_nt4";

/// Fixture a: an ABI from before `n_experts@60` was added (60-byte segment) fails against
/// the real gfx1201 MoE object-shaped evidence.
#[test]
fn broken_moe_pre_widening_abi() {
    let export = real_export(
        &moe_abi::QWEN4_MOE_SYM_GATE_UP,
        Arch::Gfx1201,
        MOE_GATE_UP_NT4,
    );
    let real = object_evidence_ok(export);
    assert_eq!(
        check_export(&moe_abi::QWEN4_MOE_SYM_GATE_UP, export, &real),
        vec![],
        "the unbroken object evidence must agree first"
    );
    assert!(real
        .args
        .iter()
        .any(|a| a.offset == 60 && a.name == "n_experts"));

    let mut old = real.clone();
    old.args.retain(|a| a.offset != 60);
    old.segment_bytes = Some(60);
    assert_eq!(
        check_export(&moe_abi::QWEN4_MOE_SYM_GATE_UP, export, &old),
        vec![
            Drift::MissingArg {
                symbol: MOE_GATE_UP_NT4.into(),
                name: "n_experts".into(),
                offset: 60,
                value_kind: "by_value".into(),
            },
            Drift::Segment {
                symbol: MOE_GATE_UP_NT4.into(),
                contract: 64,
                evidence: 60,
            },
        ]
    );
}

/// Fixture b: swapped M/K names in the evidence.
#[test]
fn broken_swapped_field_names() {
    let abi = &mq6_i8::MQ6_I8_PM_SET_F32;
    let export = real_export(abi, Arch::Gfx1201, MQ6_SET_T128X256);
    let real = object_evidence_ok(export);
    assert_eq!(
        check_export(abi, export, &real),
        vec![],
        "unbroken evidence must agree"
    );

    let mut swapped = real.clone();
    for a in swapped.args.iter_mut() {
        match a.offset {
            24 => a.name = "K".into(),
            28 => a.name = "M".into(),
            _ => {}
        }
    }
    assert_eq!(
        check_export(abi, export, &swapped),
        vec![
            Drift::ArgName {
                symbol: MQ6_SET_T128X256.into(),
                offset: 24,
                contract: "M".into(),
                evidence: "K".into(),
            },
            Drift::ArgName {
                symbol: MQ6_SET_T128X256.into(),
                offset: 28,
                contract: "K".into(),
                evidence: "M".into(),
            },
        ]
    );

    // The same corruption in the generator evidence is caught too.
    let mut swapped = generator_for(Arch::Gfx1201, MQ6_SET_T128X256).evidence();
    for a in swapped.args.iter_mut() {
        match a.offset {
            24 => a.name = "K".into(),
            28 => a.name = "M".into(),
            _ => {}
        }
    }
    let names: Vec<_> = check_export(abi, export, &swapped)
        .into_iter()
        .filter(|d| matches!(d, Drift::ArgName { .. }))
        .collect();
    assert_eq!(names.len(), 2, "{names:?}");
}

/// Gate of `hipcc_exports_match_compiled_metadata`: `Ok(toolchain id)` when a device compiler
/// resolves, `Err(reason)` (the test then prints it and skips) when none does.
fn hipcc_gate(toolchain: Option<String>) -> Result<String, String> {
    toolchain.ok_or_else(|| {
        "no hipcc resolves (KernelCompiler::local_toolchain_id() is None); the compiled-metadata \
         leg needs a ROCm toolchain (CPU only, no GPU)"
            .to_string()
    })
}

#[test]
fn hipcc_gate_skips_without_a_compiler() {
    let reason = hipcc_gate(None).expect_err("no toolchain must skip");
    assert!(reason.contains("no hipcc resolves"), "{reason}");
    assert_eq!(
        hipcc_gate(Some("AMD clang version 99".into())).as_deref(),
        Ok("AMD clang version 99")
    );
}

/// Every `HipccSource` export: its `source` is compiled with the runtime `KernelCompiler` for the
/// export's arch (`compile_to_paths`: the JIT's own argv, module flags and per-source flags; no
/// device), the object is lifted with `peacemaker_lift` and its AMDGPU metadata is compared with
/// the contract through `check_export`. Each `(module, source, arch)` is compiled once. Skips
/// (printing the reason) when no hipcc resolves.
#[test]
fn hipcc_exports_match_compiled_metadata() {
    let toolchain = match hipcc_gate(rdna_compute::KernelCompiler::local_toolchain_id()) {
        Ok(id) => id,
        Err(reason) => {
            eprintln!("SKIP hipcc_exports_match_compiled_metadata: {reason}");
            return;
        }
    };

    // (abi, export) pairs and the distinct compiles they need.
    let mut pairs: Vec<(&'static KernelAbiV1, &'static ExportV1)> = Vec::new();
    let mut jobs: BTreeSet<(&'static str, &'static str, &'static str)> = BTreeSet::new();
    for abi in families::all() {
        for export in abi.exports {
            if let ObjectRefV1::HipccSource { module, source } = export.object {
                pairs.push((abi, export));
                jobs.insert((export.arch.name(), module, source));
            }
        }
    }
    assert!(!pairs.is_empty(), "no hipcc export was found");

    // Keep the compiler's cache side effects out of $HOME (single-threaded setup: the only test
    // of this file that touches a KernelCompiler).
    let work = std::env::temp_dir().join(format!("kernel_abi_drift_hipcc_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&work);
    std::fs::create_dir_all(work.join("cache")).unwrap();
    std::fs::create_dir_all(work.join("home")).unwrap();
    std::env::set_var("HIPFIRE_KERNEL_CACHE", work.join("cache"));
    std::env::set_var("HIPFIRE_HOME", work.join("home"));

    let jobs: Vec<_> = jobs.into_iter().collect();
    let next = std::sync::atomic::AtomicUsize::new(0);
    type Compiled = (IrArch, Vec<(String, KernelMeta)>);
    let results: std::sync::Mutex<BTreeMap<(&str, &str, &str), Result<Compiled, String>>> =
        std::sync::Mutex::new(BTreeMap::new());
    std::thread::scope(|scope| {
        for _ in 0..8usize.min(jobs.len()) {
            scope.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let Some(&(arch, module, source)) = jobs.get(i) else {
                    break;
                };
                let built = (|| -> Result<Compiled, String> {
                    let text = runtime_source(module, source)?;
                    let dir = work.join(format!("{i}_{arch}_{module}"));
                    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
                    let (src, obj) = (dir.join(format!("{module}.hip")), dir.join("o.hsaco"));
                    let kc = rdna_compute::KernelCompiler::new(arch, String::new())
                        .map_err(|e| e.to_string())?;
                    kc.compile_to_paths(module, &text, &src, &obj)
                        .map_err(|e| format!("hipcc failed: {e}"))?;
                    let bytes = std::fs::read(&obj).map_err(|e| e.to_string())?;
                    // Metadata only (envelope + AMDGPU note): the object is not decoded, so a
                    // kernel using an instruction the lifter has no table for is still checked.
                    let ir_arch = match arch {
                        "gfx1100" => IrArch::Gfx1100,
                        "gfx1151" => IrArch::Gfx1151,
                        "gfx1201" => IrArch::Gfx1201,
                        other => return Err(format!("{other}: no peacemaker arch")),
                    };
                    let (_, images) = IrSource::read(&bytes, ir_arch)
                        .map_err(|e| format!("envelope read failed: {e}"))?;
                    Ok((
                        ir_arch,
                        images
                            .into_iter()
                            .map(|k| (k.name, k.metadata.parsed))
                            .collect(),
                    ))
                })();
                results
                    .lock()
                    .unwrap()
                    .insert((arch, module, source), built);
            });
        }
    });
    let results = results.into_inner().unwrap();

    let mut failures = Vec::new();
    let mut checked = 0usize;
    for (abi, export) in pairs {
        let ObjectRefV1::HipccSource { module, source } = export.object else {
            unreachable!()
        };
        let key = (export.arch.name(), module, source);
        let (ir_arch, kernels) = match &results[&key] {
            Ok(p) => p,
            Err(e) => {
                failures.push(format!(
                    "[{}] {} ({}, {module}, {source}): {e}",
                    abi.id,
                    export.symbol,
                    export.arch.name()
                ));
                continue;
            }
        };
        let object = ObjectMeta {
            arch: *ir_arch,
            kernels: kernels
                .iter()
                .map(|(n, m)| (n.strip_suffix(".kd").unwrap_or(n).to_string(), Some(m)))
                .collect(),
        };
        let evidence = match program_evidence(export, &object, source, None) {
            Ok(e) => e,
            Err(e) => {
                failures.push(format!("[{}] {e}", abi.id));
                continue;
            }
        };
        checked += 1;
        let drifts = check_export(abi, export, &evidence);
        if !drifts.is_empty() {
            failures.push(drift_lines(abi, &drifts));
        }
    }

    let _ = std::fs::remove_dir_all(&work);
    eprintln!(
        "hipcc_exports_match_compiled_metadata: {checked} exports checked against compiled metadata \
         ({} distinct compiles, toolchain {toolchain})",
        jobs.len()
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert!(checked > 0, "no hipcc export was checked");
}

/// The text the runtime hands hipcc for `module`: the `.hip` file, behind the `block_i8_128`
/// prelude the MQ6 INT8 GEMM modules get in `rdna_compute::kernels` (`GEMM_MQ6G256V2_MMQ_I8_*`).
fn runtime_source(module: &str, source: &str) -> Result<String, String> {
    let read =
        |p: &str| std::fs::read_to_string(repo_root().join(p)).map_err(|e| format!("{p}: {e}"));
    let body = read(source)?;
    Ok(match module {
        "gemm_mq6g256v2_mmq_i8_gfx12" | "gemm_mq6g256v2_mmq_i8_gfx11" => format!(
            "#define HIPFIRE_BLOCK_I8_128_QUANT_NO_STANDALONE 1\n{}{body}",
            read("kernels/src/block_i8_128_quant.hip")?
        ),
        "gemm_mq6g256v2_mmq_i8_gfx11_t256" => format!(
            "#define HIPFIRE_BLOCK_I8_128_QUANT_NO_STANDALONE 1\n#define I8_TILE 256\n{}{body}",
            read("kernels/src/block_i8_128_quant.hip")?
        ),
        _ => body,
    })
}

/// Fixture c: a contract whose t128x256 export says 36864 B of dynamic LDS drifts from the
/// generator (55296), and the REAL contract rejects a 36864 B launch of that symbol.
#[test]
fn broken_dynamic_lds() {
    let real_abi = &mq6_i8::MQ6_I8_PM_SET_F32;
    let real = real_export(real_abi, Arch::Gfx1201, MQ6_SET_T128X256);
    assert_eq!(real.dynamic_lds, 55296);
    let generator = generator_for(Arch::Gfx1201, MQ6_SET_T128X256);
    assert_eq!(generator.dynamic_lds, Some(55296));
    assert_eq!(
        check_export(real_abi, real, &generator.evidence()),
        vec![],
        "the real contract must agree with the generator first"
    );

    let mut exports = real_abi.exports.to_vec();
    for e in exports.iter_mut().filter(|e| e.symbol == MQ6_SET_T128X256) {
        e.dynamic_lds = 36864;
    }
    let leaked = leak_with_exports(real_abi, exports);
    let leaked_export = real_export(leaked, Arch::Gfx1201, MQ6_SET_T128X256);
    assert_eq!(
        check_export(leaked, leaked_export, &generator.evidence()),
        vec![Drift::DynamicLds {
            symbol: MQ6_SET_T128X256.into(),
            contract: 36864,
            evidence: 55296,
        }]
    );
    // Object metadata cannot see dynamic LDS: only the generator leg catches this.
    assert_eq!(
        check_export(leaked, leaked_export, &object_evidence_ok(leaked_export)),
        vec![]
    );

    // The real contract refuses the 36864 B launch of the t128x256 entry before enqueue.
    let operands = mq6_operands(mq6_i8::FORMAT_F32_N_M, 0x3000_0000, 131_072);
    let mut inv = mq6_inv(MQ6_SET_T128X256, &MQ6_GOOD_PARAMS, &operands);
    inv.block = [512, 1, 1];
    inv.dynamic_lds = 55296;
    admitted(real_abi, &inv);
    inv.dynamic_lds = 36864;
    assert_eq!(
        rejected(real_abi, &inv),
        AbiError::Launch {
            abi: "mq6_i8_gemm.pm.set_f32",
            what: "dynamic_lds",
            expected: [55296, 0, 0],
            got: [36864, 0, 0],
        }
    );
}

/// Fixture d: a contract with the `M`/`K` field offsets swapped drifts from the real object.
#[test]
fn broken_swapped_field_offsets() {
    let real_abi = &mq6_i8::MQ6_I8_PM_SET_F32;
    let export = real_export(real_abi, Arch::Gfx1201, MQ6_SET);
    let evidence = object_evidence_ok(export);
    assert_eq!(check_export(real_abi, export, &evidence), vec![]);

    let mut fields = real_abi.fields.to_vec();
    for f in fields.iter_mut() {
        match f.name {
            "M" => f.offset = 28,
            "K" => f.offset = 24,
            _ => {}
        }
    }
    fields.sort_by_key(|f| f.offset);
    let mut abi = *real_abi;
    abi.fields = Box::leak(fields.into_boxed_slice());
    let leaked: &'static KernelAbiV1 = Box::leak(Box::new(abi));
    let leaked_export = real_export(leaked, Arch::Gfx1201, MQ6_SET);
    assert_eq!(
        check_export(leaked, leaked_export, &evidence),
        vec![
            Drift::ArgName {
                symbol: MQ6_SET.into(),
                offset: 24,
                contract: "K".into(),
                evidence: "M".into(),
            },
            Drift::ArgName {
                symbol: MQ6_SET.into(),
                offset: 28,
                contract: "M".into(),
                evidence: "K".into(),
            },
        ]
    );
}

/// Fixture e: a contract pinning another object hash drifts from the real object.
#[test]
fn broken_object_sha() {
    const WRONG: &str = "0000000000000000000000000000000000000000000000000000000000000000";
    let real_abi = &mq6_i8::MQ6_I8_PM_SET_F32;
    let real = real_export(real_abi, Arch::Gfx1201, MQ6_SET);
    let ObjectRefV1::Embedded {
        module,
        artifact,
        sha256,
    } = real.object
    else {
        panic!("{MQ6_SET} is not an embedded export");
    };
    let evidence = object_evidence_ok(real);
    assert_eq!(check_export(real_abi, real, &evidence), vec![]);

    let mut exports = real_abi.exports.to_vec();
    for e in exports.iter_mut().filter(|e| e.symbol == MQ6_SET) {
        e.object = ObjectRefV1::Embedded {
            module,
            artifact,
            sha256: WRONG,
        };
    }
    let leaked = leak_with_exports(real_abi, exports);
    let leaked_export = real_export(leaked, Arch::Gfx1201, MQ6_SET);
    assert_eq!(
        check_export(leaked, leaked_export, &evidence),
        vec![Drift::ObjectSha {
            symbol: MQ6_SET.into(),
            contract: WRONG.into(),
            evidence: sha256.to_string(),
        }]
    );
}

/// Good launch of the MQ6 PM entries: `M = 256, K = 512, N = 128` (grid `[2, 1, 1]` on both
/// tiles).
const MQ6_GOOD_PARAMS: [u64; 3] = [256, 512, 128];

fn mq6_operands(y: FormatId, y_address: u64, y_bytes: u64) -> [OperandBinding; 3] {
    [
        OperandBinding {
            address: 0x1000_0000,
            bytes: Some(102_400), // M * (K / 256) * 200
            format: mq6_i8::FORMAT_MQ6G256V2_WEIGHTS,
        },
        OperandBinding {
            address: 0x2000_0000,
            bytes: Some(69_632), // N * (K / 128) * 136
            format: mq6_i8::FORMAT_BLOCK_I8_128_KB_N,
        },
        OperandBinding {
            address: y_address,
            bytes: Some(y_bytes),
            format: y,
        },
    ]
}

fn mq6_inv<'a>(
    symbol: &'a str,
    params: &'a [u64],
    operands: &'a [OperandBinding],
) -> InvocationV1<'a> {
    InvocationV1 {
        arch: Arch::Gfx1201,
        symbol,
        params,
        operands,
        grid: [2, 1, 1],
        block: [256, 1, 1],
        dynamic_lds: 36864,
    }
}

/// Good launch of the MoE entries: `M = 128, K = 256, x_row_div = 1, m_total = 32,
/// x_src_rows = 64, n_experts = 8` (grid `[2, 2, 1]`, block 128).
const MOE_GOOD_PARAMS: [u64; 6] = [128, 256, 1, 32, 64, 8];

fn moe_down_operands(xq: FormatId, xq_bytes: u64, y: FormatId) -> [OperandBinding; 5] {
    [
        OperandBinding {
            address: 0x1000_0000,
            bytes: Some(64), // n_experts * 8
            format: moe_abi::FORMAT_EXPERT_PTRS_DOWN_QT53,
        },
        OperandBinding {
            address: 0x2000_0000,
            bytes: Some(8), // ceil(m_total / 16) * 4
            format: moe_abi::FORMAT_EXPERT_TILE_IDS,
        },
        OperandBinding {
            address: 0x3000_0000,
            bytes: Some(128), // m_total * 4
            format: moe_abi::FORMAT_SORTED_SLOT_INDEX,
        },
        OperandBinding {
            address: 0x4000_0000,
            bytes: Some(xq_bytes),
            format: xq,
        },
        OperandBinding {
            address: 0x5000_0000,
            bytes: Some(8192), // m_total * M * 2
            format: y,
        },
    ]
}

fn moe_inv<'a>(symbol: &'a str, operands: &'a [OperandBinding]) -> InvocationV1<'a> {
    InvocationV1 {
        arch: Arch::Gfx1201,
        symbol,
        params: &MOE_GOOD_PARAMS,
        operands,
        grid: [2, 2, 1],
        block: [128, 1, 1],
        dynamic_lds: 0,
    }
}

/// Fixture f: recipe interchange, alignment, domain, extent and grid, all rejected by
/// `validate_invocation` of the REAL contracts before anything is enqueued.
#[test]
fn broken_recipe_interchange_is_rejected_before_enqueue() {
    // --- MQ6 ---
    let f32_abi = &mq6_i8::MQ6_I8_PM_SET_F32;
    let bf16_abi = &mq6_i8::MQ6_I8_PM_SET_BF16;

    let f32_ops = mq6_operands(mq6_i8::FORMAT_F32_N_M, 0x3000_0000, 131_072);
    admitted(f32_abi, &mq6_inv(MQ6_SET, &MQ6_GOOD_PARAMS, &f32_ops));
    let bf16_ops = mq6_operands(mq6_i8::FORMAT_BF16_N_M, 0x3000_0000, 65_536);
    admitted(bf16_abi, &mq6_inv(MQ6_BF16, &MQ6_GOOD_PARAMS, &bf16_ops));

    // BF16 Y bound to the F32 contract (and the reverse).
    let ops = mq6_operands(mq6_i8::FORMAT_BF16_N_M, 0x3000_0000, 131_072);
    assert_eq!(
        rejected(f32_abi, &mq6_inv(MQ6_SET, &MQ6_GOOD_PARAMS, &ops)),
        AbiError::Format {
            abi: "mq6_i8_gemm.pm.set_f32",
            operand: "Y",
            expected: mq6_i8::FORMAT_F32_N_M,
            got: mq6_i8::FORMAT_BF16_N_M,
        }
    );
    let ops = mq6_operands(mq6_i8::FORMAT_F32_N_M, 0x3000_0000, 65_536);
    assert_eq!(
        rejected(bf16_abi, &mq6_inv(MQ6_BF16, &MQ6_GOOD_PARAMS, &ops)),
        AbiError::Format {
            abi: "mq6_i8_gemm.pm.set_bf16",
            operand: "Y",
            expected: mq6_i8::FORMAT_BF16_N_M,
            got: mq6_i8::FORMAT_F32_N_M,
        }
    );

    // BF16 Y address = 4 mod 8.
    let ops = mq6_operands(mq6_i8::FORMAT_BF16_N_M, 0x3000_0004, 65_536);
    assert_eq!(
        rejected(bf16_abi, &mq6_inv(MQ6_BF16, &MQ6_GOOD_PARAMS, &ops)),
        AbiError::Misaligned {
            abi: "mq6_i8_gemm.pm.set_bf16",
            operand: "Y",
            address: 0x3000_0004,
            align: 8,
        }
    );

    // K % 256 != 0.
    let bad_k = [256u64, 384, 128];
    let err = rejected(f32_abi, &mq6_inv(MQ6_SET, &bad_k, &f32_ops));
    assert!(
        matches!(
            err,
            AbiError::Domain {
                rule: Rule::MultipleOf(Expr::Param("K"), 256),
                ..
            }
        ),
        "{err}"
    );

    // Undersized weight bytes.
    let mut ops = mq6_operands(mq6_i8::FORMAT_F32_N_M, 0x3000_0000, 131_072);
    ops[0].bytes = Some(102_399);
    assert_eq!(
        rejected(f32_abi, &mq6_inv(MQ6_SET, &MQ6_GOOD_PARAMS, &ops)),
        AbiError::Extent {
            abi: "mq6_i8_gemm.pm.set_f32",
            operand: "A",
            need: 102_400,
            have: 102_399,
        }
    );

    // Wrong grid.
    let mut inv = mq6_inv(MQ6_SET, &MQ6_GOOD_PARAMS, &f32_ops);
    inv.grid = [3, 1, 1];
    assert_eq!(
        rejected(f32_abi, &inv),
        AbiError::Launch {
            abi: "mq6_i8_gemm.pm.set_f32",
            what: "grid",
            expected: [2, 1, 1],
            got: [3, 1, 1],
        }
    );

    // --- grouped MoE ---
    let down = &moe_abi::QWEN4_MOE_SYM_DOWN;
    let down_gp = &moe_abi::QWEN4_MOE_SYM_DOWN_GP;
    let flat = moe_down_operands(
        moe_abi::FORMAT_XQ_DOWN_ROT128_FLAT_SLOTS,
        9216, // K/128 * x_src_rows * 72
        moe_abi::FORMAT_Y_DOWN_BF16,
    );
    admitted(down, &moe_inv(MOE_DOWN_NT4, &flat));
    let grouped = moe_down_operands(
        moe_abi::FORMAT_XQ_DOWN_ROT128_GROUPED_POSITION,
        4608, // K/128 * m_total * 72
        moe_abi::FORMAT_Y_DOWN_BF16,
    );
    admitted(down_gp, &moe_inv(MOE_DOWN_GP_NT4, &grouped));

    // Grouped-position Xq bound to the flat-slot down contract (and the reverse).
    let ops = moe_down_operands(
        moe_abi::FORMAT_XQ_DOWN_ROT128_GROUPED_POSITION,
        9216,
        moe_abi::FORMAT_Y_DOWN_BF16,
    );
    assert_eq!(
        rejected(down, &moe_inv(MOE_DOWN_NT4, &ops)),
        AbiError::Format {
            abi: "qwen4_moe_sym.down.pm",
            operand: "Xq",
            expected: moe_abi::FORMAT_XQ_DOWN_ROT128_FLAT_SLOTS,
            got: moe_abi::FORMAT_XQ_DOWN_ROT128_GROUPED_POSITION,
        }
    );
    let ops = moe_down_operands(
        moe_abi::FORMAT_XQ_DOWN_ROT128_FLAT_SLOTS,
        4608,
        moe_abi::FORMAT_Y_DOWN_BF16,
    );
    assert_eq!(
        rejected(down_gp, &moe_inv(MOE_DOWN_GP_NT4, &ops)),
        AbiError::Format {
            abi: "qwen4_moe_sym.down_gp.pm",
            operand: "Xq",
            expected: moe_abi::FORMAT_XQ_DOWN_ROT128_GROUPED_POSITION,
            got: moe_abi::FORMAT_XQ_DOWN_ROT128_FLAT_SLOTS,
        }
    );

    // A gate_up Y format on a down contract.
    let ops = moe_down_operands(
        moe_abi::FORMAT_XQ_DOWN_ROT128_FLAT_SLOTS,
        9216,
        moe_abi::FORMAT_Y_GATE_UP_SWIGLU_BF16,
    );
    assert_eq!(
        rejected(down, &moe_inv(MOE_DOWN_NT4, &ops)),
        AbiError::Format {
            abi: "qwen4_moe_sym.down.pm",
            operand: "Y_grouped",
            expected: moe_abi::FORMAT_Y_DOWN_BF16,
            got: moe_abi::FORMAT_Y_GATE_UP_SWIGLU_BF16,
        }
    );
}

// ---- ggml shim: pin, object and invocation evidence -----------------------------------

const GGML_IQ4_XS: &str = "ggml_mmvq_iq4_xs_gfx1201";

/// Every `sha256  local-path  upstream-path` line of `third_party/ggml/PIN` names a file of the
/// snapshot whose bytes hash to the pinned digest: an upstream re-pin without updating the PIN,
/// or an edit inside the verbatim snapshot, fails here. Lines starting with `#` and
/// `key=value` metadata lines (an `=` before the first space) are skipped.
#[test]
fn ggml_snapshot_matches_pin() {
    let dir = repo_root().join("third_party/ggml");
    let pin = std::fs::read_to_string(dir.join("PIN")).expect("third_party/ggml/PIN");
    let mut checked = 0usize;
    let mut failures = Vec::new();
    for line in pin.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let first = line.split(' ').next().unwrap_or("");
        if first.contains('=') {
            continue;
        }
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() != 3 || fields[0].len() != 64 {
            failures.push(format!("malformed PIN line: {line:?}"));
            continue;
        }
        let path = dir.join(fields[1]);
        match std::fs::read(&path) {
            Ok(bytes) => {
                let got = hex(&Sha256::digest(&bytes));
                if !got.eq_ignore_ascii_case(fields[0]) {
                    failures.push(format!(
                        "{}: sha256 is {got}, PIN pins {}",
                        fields[1], fields[0]
                    ));
                }
            }
            Err(e) => failures.push(format!("{}: unreadable: {e}", path.display())),
        }
        checked += 1;
    }
    assert!(checked > 0, "PIN lists no files");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The ggml exports live in an offload bundle (`.co`), lifted by the same path as the `.hxaco`
/// images: all five must be present in `embedded_objects_match_contracts`' evidence.
#[test]
fn ggml_exports_have_object_evidence() {
    let mut n = 0;
    for abi in ggml_abi::ABIS {
        for export in abi.exports {
            let evidence = object_evidence_ok(export);
            assert_eq!(check_export(abi, export, &evidence), vec![], "{}", abi.id);
            n += 1;
        }
    }
    assert_eq!(n, 5);
}

/// Negative control: a ggml contract pinning another object hash drifts from the real object.
#[test]
fn broken_ggml_object_sha() {
    const WRONG: &str = "0000000000000000000000000000000000000000000000000000000000000000";
    let real_abi = &ggml_abi::GGML_MMVQ_IQ4_XS;
    let real = real_export(real_abi, Arch::Gfx1201, GGML_IQ4_XS);
    let ObjectRefV1::Embedded {
        module,
        artifact,
        sha256,
    } = real.object
    else {
        panic!("{GGML_IQ4_XS} is not an embedded export");
    };
    let evidence = object_evidence_ok(real);
    assert_eq!(check_export(real_abi, real, &evidence), vec![]);

    let mut exports = real_abi.exports.to_vec();
    for e in exports.iter_mut() {
        e.object = ObjectRefV1::Embedded {
            module,
            artifact,
            sha256: WRONG,
        };
    }
    let leaked = leak_with_exports(real_abi, exports);
    let leaked_export = real_export(leaked, Arch::Gfx1201, GGML_IQ4_XS);
    assert_eq!(
        check_export(leaked, leaked_export, &evidence),
        vec![Drift::ObjectSha {
            symbol: GGML_IQ4_XS.into(),
            contract: WRONG.into(),
            evidence: sha256.to_string(),
        }]
    );
}

/// `iq4_xs` mmvq at `M = 5120, K = 5120`: admitted with exact extents, rejected one weight
/// byte short.
#[test]
fn ggml_iq4_xs_invocation_admission() {
    let abi = &ggml_abi::GGML_MMVQ_IQ4_XS;
    let params = [5120u64, 5120];
    let operands = |weight_bytes: u64| {
        [
            OperandBinding {
                address: 0x1000_0000,
                bytes: Some(weight_bytes), // M * (K / 256) * 136
                format: ggml_abi::FORMAT_IQ4_XS_136,
            },
            OperandBinding {
                address: 0x4000_0000,
                bytes: Some(5760), // K / 32 * 36
                format: ggml_abi::FORMAT_Q8_1_MMVQ,
            },
            OperandBinding {
                address: 0x5000_0000,
                bytes: Some(20_480), // M * 4
                format: ggml_abi::FORMAT_F32_M,
            },
        ]
    };
    let inv = |ops: &'static [OperandBinding]| InvocationV1 {
        arch: Arch::Gfx1201,
        symbol: GGML_IQ4_XS,
        params: &params,
        operands: ops,
        grid: [5120, 1, 1],
        block: [32, 8, 1],
        dynamic_lds: 0,
    };
    let good: &'static [OperandBinding] = Box::leak(Box::new(operands(13_926_400)));
    admitted(abi, &inv(good));
    let short: &'static [OperandBinding] = Box::leak(Box::new(operands(13_926_399)));
    assert_eq!(
        rejected(abi, &inv(short)),
        AbiError::Extent {
            abi: "ggml_mmvq.iq4_xs.gfx1201",
            operand: "vx",
            need: 13_926_400,
            have: 13_926_399,
        }
    );
}

/// The S3 additions (`q5_k`, the `_res` epilogue variants, the gather producer) are
/// hipcc-compiled from `kernels/src/ggml_mmvq_fused_shim.hip`: all seven exports must
/// be present in the committed object with the contracts' metadata.
#[test]
fn ggml_fused_exports_have_object_evidence() {
    let mut n = 0;
    for abi in ggml_fused::ABIS {
        for export in abi.exports {
            let evidence = object_evidence_ok(export);
            assert_eq!(check_export(abi, export, &evidence), vec![], "{}", abi.id);
            n += 1;
        }
    }
    assert_eq!(n, 7);
}

/// Negative control: a fused contract pinning another object hash drifts from the real object.
#[test]
fn broken_ggml_fused_object_sha() {
    const WRONG: &str = "0000000000000000000000000000000000000000000000000000000000000000";
    const SYMBOL: &str = "ggml_mmvq_q5_k_res_gfx1201";
    let real_abi = &ggml_fused::GGML_MMVQ_Q5_K_RES;
    let real = real_export(real_abi, Arch::Gfx1201, SYMBOL);
    let ObjectRefV1::Embedded {
        module,
        artifact,
        sha256,
    } = real.object
    else {
        panic!("{SYMBOL} is not an embedded export");
    };
    let evidence = object_evidence_ok(real);
    assert_eq!(check_export(real_abi, real, &evidence), vec![]);

    let mut exports = real_abi.exports.to_vec();
    for e in exports.iter_mut() {
        e.object = ObjectRefV1::Embedded {
            module,
            artifact,
            sha256: WRONG,
        };
    }
    let leaked = leak_with_exports(real_abi, exports);
    let leaked_export = real_export(leaked, Arch::Gfx1201, SYMBOL);
    assert_eq!(
        check_export(leaked, leaked_export, &evidence),
        vec![Drift::ObjectSha {
            symbol: SYMBOL.into(),
            contract: WRONG.into(),
            evidence: sha256.to_string(),
        }]
    );
}

/// `q5_k` residual mmvq at `M = 2560, K = 5120`: admitted with exact extents (the `dst`
/// operand is read-write), rejected one destination byte short.
#[test]
fn ggml_fused_q5_k_res_invocation_admission() {
    let abi = &ggml_fused::GGML_MMVQ_Q5_K_RES;
    let params = [2560u64, 5120];
    let operands = |dst_bytes: u64| {
        [
            OperandBinding {
                address: 0x1000_0000,
                bytes: Some(9_011_200), // M * (K / 256) * 176
                format: ggml_fused::FORMAT_Q5_K_176,
            },
            OperandBinding {
                address: 0x4000_0000,
                bytes: Some(5760), // K / 32 * 36
                format: ggml_abi::FORMAT_Q8_1_MMVQ,
            },
            OperandBinding {
                address: 0x5000_0000,
                bytes: Some(dst_bytes), // M * 4
                format: ggml_abi::FORMAT_F32_M,
            },
        ]
    };
    fn inv<'a>(params: &'a [u64], ops: &'a [OperandBinding]) -> InvocationV1<'a> {
        InvocationV1 {
            arch: Arch::Gfx1201,
            symbol: "ggml_mmvq_q5_k_res_gfx1201",
            params,
            operands: ops,
            grid: [2560, 1, 1],
            block: [32, 8, 1],
            dynamic_lds: 0,
        }
    }
    let good = operands(10_240);
    admitted(abi, &inv(&params, &good));
    let short = operands(10_239);
    assert_eq!(
        rejected(abi, &inv(&params, &short)),
        AbiError::Extent {
            abi: "ggml_mmvq_fused.q5_k_res.gfx1201",
            operand: "dst",
            need: 10_240,
            have: 10_239,
        }
    );
}

/// The gather producer at `K = 6144`: admitted with exact extents, rejected for a
/// non-multiple-of-32 `K` and one map byte short.
#[test]
fn ggml_fused_gather_invocation_admission() {
    let abi = &ggml_fused::GGML_Q8_1_GATHER;
    let operands = |map_bytes: u64| {
        [
            OperandBinding {
                address: 0x1000_0000,
                bytes: Some(24_576), // K * 4
                format: ggml_abi::FORMAT_F32_ROWS_S01,
            },
            OperandBinding {
                address: 0x4000_0000,
                bytes: Some(6912), // K / 32 * 36
                format: ggml_abi::FORMAT_Q8_1_MMVQ,
            },
            OperandBinding {
                address: 0x5000_0000,
                bytes: Some(map_bytes), // K / 32 * 4
                format: ggml_fused::FORMAT_I32_BLK_MAP,
            },
        ]
    };
    fn inv<'a>(params: &'a [u64], grid: u32, ops: &'a [OperandBinding]) -> InvocationV1<'a> {
        InvocationV1 {
            arch: Arch::Gfx1201,
            symbol: "ggml_q8_1_quantize_gather_gfx1201",
            params,
            operands: ops,
            grid: [grid, 1, 1],
            block: [256, 1, 1],
            dynamic_lds: 0,
        }
    }
    let good = operands(768);
    admitted(abi, &inv(&[6144u64], 24, &good));
    let short = operands(767);
    assert_eq!(
        rejected(abi, &inv(&[6144u64], 24, &short)),
        AbiError::Extent {
            abi: "ggml_mmvq_fused.q8_1_gather.gfx1201",
            operand: "blk_map",
            need: 768,
            have: 767,
        }
    );
    assert!(matches!(
        rejected(abi, &inv(&[6145u64], 25, &good)),
        AbiError::Domain { .. }
    ));
}

// ---- ggml convert shim: object and invocation evidence --------------------------------

const CONVERT_Q4_K_F16: &str = "ggml_dequant_q4_k_f16_gfx1201";

/// The S3 convert shim (dequant f16 / f32, gathers, A8 round trip) is hipcc-compiled from
/// `kernels/src/ggml_convert_shim.hip`: all 21 exports must be present in the committed object with
/// the contracts' metadata (explicit offsets / sizes, segment 36 / 40 / 28 / 12, no hidden args, no LDS).
#[test]
fn ggml_convert_exports_have_object_evidence() {
    let mut n = 0;
    for abi in convert_abi::ABIS {
        for export in abi.exports {
            let evidence = object_evidence_ok(export);
            assert_eq!(check_export(abi, export, &evidence), vec![], "{}", abi.id);
            n += 1;
        }
    }
    assert_eq!(n, 21);
}

/// Both S3 families name the same `q5_k` format id (the weight is one byte layout).
#[test]
fn ggml_convert_q5_k_format_is_the_fused_family_s() {
    assert_eq!(convert_abi::FORMAT_Q5_K_176, ggml_fused::FORMAT_Q5_K_176);
}

/// Negative control: a convert contract pinning another object hash drifts from the real object.
#[test]
fn broken_ggml_convert_object_sha() {
    const WRONG: &str = "0000000000000000000000000000000000000000000000000000000000000000";
    let real_abi = &convert_abi::q4_k::DEQUANT_F16;
    let real = real_export(real_abi, Arch::Gfx1201, CONVERT_Q4_K_F16);
    let ObjectRefV1::Embedded {
        module,
        artifact,
        sha256,
    } = real.object
    else {
        panic!("{CONVERT_Q4_K_F16} is not an embedded export");
    };
    let evidence = object_evidence_ok(real);
    assert_eq!(check_export(real_abi, real, &evidence), vec![]);

    let mut exports = real_abi.exports.to_vec();
    for e in exports.iter_mut() {
        e.object = ObjectRefV1::Embedded {
            module,
            artifact,
            sha256: WRONG,
        };
    }
    let leaked = leak_with_exports(real_abi, exports);
    let leaked_export = real_export(leaked, Arch::Gfx1201, CONVERT_Q4_K_F16);
    assert_eq!(
        check_export(leaked, leaked_export, &evidence),
        vec![Drift::ObjectSha {
            symbol: CONVERT_Q4_K_F16.into(),
            contract: WRONG.into(),
            evidence: sha256.to_string(),
        }]
    );
}

/// `q4_k` -> f16 dequant of `M = 5120, K = 6144` with a column permutation: admitted with exact
/// extents (weights, output, colmap), rejected one weight byte / one colmap byte short and for a
/// `K` that is not a chunk multiple when a colmap is passed; a null colmap needs no chunk multiple.
#[test]
fn ggml_convert_q4_k_dequant_invocation_admission() {
    let abi = &convert_abi::q4_k::DEQUANT_F16;
    let (m, k) = (5120u64, 6144u64);
    let weight = m * (k / 256) * 144;
    let operands = |weight_bytes: u64, colmap_bytes: u64| {
        [
            OperandBinding {
                address: 0x1000_0000,
                bytes: Some(weight_bytes), // M * (K / 256) * 144
                format: ggml_abi::FORMAT_Q4_K_144,
            },
            OperandBinding {
                address: 0x6000_0000,
                bytes: Some(m * k * 2), // M * K * 2
                format: convert_abi::FORMAT_F16_FLAT,
            },
            OperandBinding {
                address: 0x5000_0000,
                bytes: Some(colmap_bytes), // K / 128 * 4
                format: convert_abi::FORMAT_I32_CHUNK_PERM,
            },
        ]
    };
    fn inv<'a>(params: &'a [u64], grid: u32, ops: &'a [OperandBinding]) -> InvocationV1<'a> {
        InvocationV1 {
            arch: Arch::Gfx1201,
            symbol: CONVERT_Q4_K_F16,
            params,
            operands: ops,
            grid: [grid, 1, 1],
            block: [32, 1, 1],
            dynamic_lds: 0,
        }
    }
    let nb = (m * k / 256) as u32;
    let params = [m, k, 1];
    let good = operands(weight, 192);
    admitted(abi, &inv(&params, nb, &good));
    let short_w = operands(weight - 1, 192);
    assert_eq!(
        rejected(abi, &inv(&params, nb, &short_w)),
        AbiError::Extent {
            abi: "ggml_convert.dequant_q4_k_f16.gfx1201",
            operand: "vx",
            need: weight,
            have: weight - 1,
        }
    );
    let short_c = operands(weight, 191);
    assert_eq!(
        rejected(abi, &inv(&params, nb, &short_c)),
        AbiError::Extent {
            abi: "ggml_convert.dequant_q4_k_f16.gfx1201",
            operand: "colmap",
            need: 192,
            have: 191,
        }
    );
    // The chunk rule (`K * COLMAP` a multiple of 128) only bites on the 32-element q8_0 contract: the 256-element
    // formats already force K % 128 == 0. K = 160 (5 blocks) is admitted with a null colmap and rejected with one.
    let q8 = &convert_abi::q8_0::DEQUANT_F16;
    let q8_ops = |colmap_present: bool| {
        [
            OperandBinding {
                address: 0x1000_0000,
                bytes: Some(4 * 5 * 34), // M * (K / 32) * 34 at M = 4, K = 160
                format: ggml_abi::FORMAT_Q8_0_34,
            },
            OperandBinding {
                address: 0x6000_0000,
                bytes: Some(4 * 160 * 2),
                format: convert_abi::FORMAT_F16_FLAT,
            },
            OperandBinding {
                address: if colmap_present { 0x5000_0000 } else { 0 },
                bytes: Some(if colmap_present { 8 } else { 0 }),
                format: convert_abi::FORMAT_I32_CHUNK_PERM,
            },
        ]
    };
    let q8_inv = |params: &'static [u64], ops: &'static [OperandBinding]| InvocationV1 {
        arch: Arch::Gfx1201,
        symbol: "ggml_dequant_q8_0_f16_gfx1201",
        params,
        operands: ops,
        grid: [2, 1, 1], // ceil(4 * 160 / 512)
        block: [256, 1, 1],
        dynamic_lds: 0,
    };
    let null_ops: &'static [OperandBinding] = Box::leak(Box::new(q8_ops(false)));
    admitted(q8, &q8_inv(&[4, 160, 0], null_ops));
    let colmap_ops: &'static [OperandBinding] = Box::leak(Box::new(q8_ops(true)));
    assert!(matches!(
        rejected(q8, &q8_inv(&[4, 160, 1], colmap_ops)),
        AbiError::Domain { .. }
    ));
}

/// The by-value gather: a token past the table is rejected by the contract (expression error of the
/// `TOK < ROWS` rule), the last row is admitted.
#[test]
fn ggml_convert_get_row1_invocation_admission() {
    let abi = &convert_abi::q6_k::GET_ROW1;
    let (rows, k) = (248_320u64, 5120u64);
    let table = rows * (k / 256) * 210;
    let ops: &'static [OperandBinding] = Box::leak(Box::new([
        OperandBinding {
            address: 0x1_0000_0000,
            bytes: Some(table),
            format: ggml_abi::FORMAT_Q6_K_210,
        },
        OperandBinding {
            address: 0x6000_0000,
            bytes: Some(k * 4),
            format: convert_abi::FORMAT_F32_ROWS_NK,
        },
    ]));
    let inv = |params: &'static [u64]| InvocationV1 {
        arch: Arch::Gfx1201,
        symbol: "ggml_get_row1_q6_k_f32_gfx1201",
        params,
        operands: ops,
        grid: [20, 1, 1],
        block: [64, 1, 1],
        dynamic_lds: 0,
    };
    admitted(abi, &inv(Box::leak(Box::new([rows, k, rows - 1]))));
    // `ROWS - (TOK + 1)` is negative: an expression error, which the validator rejects like a failed rule.
    assert!(matches!(
        rejected(abi, &inv(Box::leak(Box::new([rows, k, rows])))),
        AbiError::Expr { .. }
    ));
}
