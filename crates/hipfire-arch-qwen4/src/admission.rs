// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Pure Qwen4 capability and artifact admission.
//!
//! Admission is intentionally a value-only operation.  It validates source
//! shape, request modality, effective mesh, container format, and quantization
//! geometry before a loader is allowed to allocate weights or start a reader.

use crate::config::{Qwen4Config, ARCH_ID};
use crate::weights::Qwen4TierRecipe;
use std::fmt;

/// Request modality understood by the Qwen4 carrier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InputModality {
    Text,
    Image,
    Video,
}

/// Short alias for callers that use request rather than input terminology.
pub type RequestKind = InputModality;

/// Capability bits exposed by this first Qwen4 implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Qwen4Capability {
    Text,
    NativeMtp,
    /// qt=55 `FP8E4M3B128` routed experts (the stock-FP8 artifact).
    Fp8Experts,
}

/// The only GPU architecture with FP8 expert kernels (MI300X, wave64 MFMA).
pub const FP8_EXPERTS_ARCH: &str = "gfx942";

/// Qwen4 is text-only at the modality boundary. Image and video requests are
/// represented as unsupported rather than silently routed to text. Native
/// greedy-MTP is a capability of the executable artifact when its validated
/// MTP manifest is requested by the caller.  `fp8_experts` is a device
/// capability: true only on [`FP8_EXPERTS_ARCH`], so every other card refuses
/// a stock-FP8 artifact (fail closed) until it has kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Qwen4Capabilities {
    pub text: bool,
    pub native_mtp: bool,
    pub image: bool,
    pub video: bool,
    pub fp8_experts: bool,
}

impl Qwen4Capabilities {
    pub const fn text_ar() -> Self {
        Self {
            text: true,
            native_mtp: false,
            image: false,
            video: false,
            fp8_experts: false,
        }
    }

    /// Native MTP is enabled only for an explicitly requested executable
    /// artifact. The artifact parser remains responsible for validating that
    /// the one-layer MTP manifest/config is present.
    pub const fn text_mtp() -> Self {
        Self {
            text: true,
            native_mtp: true,
            image: false,
            video: false,
            fp8_experts: false,
        }
    }

    /// These capabilities on the device `gpu_arch`: `fp8_experts` only on
    /// [`FP8_EXPERTS_ARCH`].
    pub fn on_device(self, gpu_arch: &str) -> Self {
        Self {
            fp8_experts: gpu_arch == FP8_EXPERTS_ARCH,
            ..self
        }
    }

    pub const fn supports_modality(self, modality: InputModality) -> bool {
        match modality {
            InputModality::Text => self.text,
            InputModality::Image => self.image,
            InputModality::Video => self.video,
        }
    }

    pub const fn supports(self, capability: Qwen4Capability) -> bool {
        match capability {
            Qwen4Capability::Text => self.text,
            Qwen4Capability::NativeMtp => self.native_mtp,
            Qwen4Capability::Fp8Experts => self.fp8_experts,
        }
    }
}

impl Default for Qwen4Capabilities {
    fn default() -> Self {
        Self::text_ar()
    }
}

/// Effective (post-policy) mesh sizes.  A size of zero is retained so a
/// malformed plan is refused rather than normalized into a single-device
/// request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EffectiveMesh {
    pub pp: usize,
    pub tp: usize,
    pub ep: usize,
}

/// Alias used by mesh planners and tests.
pub type MeshShape = EffectiveMesh;

impl EffectiveMesh {
    pub const fn single() -> Self {
        Self {
            pp: 1,
            tp: 1,
            ep: 1,
        }
    }

    pub const fn new(pp: usize, tp: usize, ep: usize) -> Self {
        Self { pp, tp, ep }
    }

    /// The degrees of an already-normalized `DeviceMesh`: the loader's single
    /// mesh normalization is the only source of an admitted `EffectiveMesh`,
    /// so a request's raw pp/tp/ep are never reinterpreted a second time.
    /// An absent axis is degree 1.
    pub fn from_device_mesh(mesh: &hipfire_runtime::device_mesh::DeviceMesh) -> Self {
        use hipfire_runtime::device_mesh::DimKind;
        Self::new(
            mesh.size_of(DimKind::Pp),
            mesh.size_of(DimKind::Tp),
            mesh.size_of(DimKind::Ep),
        )
    }

    pub const fn is_single(self) -> bool {
        self.pp == 1 && self.tp == 1 && self.ep == 1
    }

    /// Largest expert-parallel degree Qwen4 admits. Experts shard `e % N`
    /// across `N` ranks and the replicated program runs on each card, so
    /// uneven splits (512 over 6 ranks = 86/86/85/85/85/85) are admitted.
    pub const EP_MAX_DEGREE: usize = 8;

    /// Smallest and largest pipeline-parallel degree Qwen4 admits
    /// (`topology.pp`): contiguous layer bands over homogeneous cards with
    /// tp = ep = 1.
    pub const PP_MIN_DEGREE: usize = 2;
    pub const PP_MAX_DEGREE: usize = 8;

    /// A pure pipeline-parallel mesh in the admitted degree range.
    pub const fn is_pipeline(self) -> bool {
        self.pp >= Self::PP_MIN_DEGREE
            && self.pp <= Self::PP_MAX_DEGREE
            && self.tp == 1
            && self.ep == 1
    }

    /// The meshes Qwen4 admits: Single, expert parallelism of 1..=8 ranks
    /// with no pipeline or tensor parallelism, or a pure pipeline-parallel
    /// mesh of [`Self::PP_MIN_DEGREE`]..=[`Self::PP_MAX_DEGREE`] stages
    /// (tp = ep = 1; PP never composes with EP or TP).
    pub const fn is_qwen4_admitted(self) -> bool {
        (self.pp == 1 && self.tp == 1 && self.ep >= 1 && self.ep <= Self::EP_MAX_DEGREE)
            || self.is_pipeline()
    }
}

impl Default for EffectiveMesh {
    fn default() -> Self {
        Self::single()
    }
}

/// Wire/storage format admitted by the runtime Qwen4 artifact boundary.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Qwen4ArtifactFormat {
    /// Versioned HFQM package containing all resident and external-row records.
    Hfqm,
    /// Safetensors is a source/conversion input, not a serving artifact.
    Safetensors,
    /// Explicitly retain unknown formats so they fail closed with a useful
    /// diagnostic instead of being mistaken for HFQM.
    Other(String),
}

/// Short format alias for callers sharing an artifact admission interface.
pub type ArtifactFormat = Qwen4ArtifactFormat;

impl Qwen4ArtifactFormat {
    pub const fn is_hfqm(&self) -> bool {
        matches!(self, Self::Hfqm)
    }
}

/// Quantized tensor format relevant to the Qwen4 manifest recipe.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum QuantFormat {
    /// MQ4G256V2, HFQM quant_type 44, 136 B per 256 weights.
    Mq4G256V2,
    /// MQ4G128V2, HFQM quant_type 53, 68 B per 128 weights.
    Mq4G128V2,
    /// Native bfloat16 record, HFQM quant_type 16.
    BF16,
    /// Raw signed-I64 metadata record, HFQM quant_type 54.
    I64,
    /// OCP e4m3 codes + BF16 128x128 `s_inv`, expert-blocked, HFQM quant_type
    /// 55 (stock-FP8 routed experts).
    Fp8E4M3B128,
    /// OCP e4m3 codes behind a 16-byte per-tensor BF16 scale header, HFQM
    /// quant_type 56 (stock-FP8 PLE rows).
    Fp8E4M3PerTensor,
    Other(String),
}

/// Shape/format identity for one logical matrix family.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct QuantTensorGeometry {
    pub format: QuantFormat,
    /// HFQM quant_type byte (44 for MQ4G256V2, 53 for MQ4G128V2, 16 for BF16,
    /// 55 for FP8E4M3B128).
    pub quant_type: u8,
    /// Logical input width K of one projection row.
    pub k: usize,
    /// Quantization block width (1 for BF16).
    pub block_size: usize,
}

impl QuantTensorGeometry {
    pub const fn mq4g256v2(k: usize) -> Self {
        Self {
            format: QuantFormat::Mq4G256V2,
            quant_type: 44,
            k,
            block_size: 256,
        }
    }

    pub const fn mq4g128v2(k: usize) -> Self {
        Self {
            format: QuantFormat::Mq4G128V2,
            quant_type: 53,
            k,
            block_size: 128,
        }
    }

    /// Source BF16 matrix rows (qt=16), loaded raw.
    pub const fn bf16(k: usize) -> Self {
        Self {
            format: QuantFormat::BF16,
            quant_type: 16,
            k,
            block_size: 1,
        }
    }

    /// qt=55 FP8E4M3B128 expert blocks with reduction width `k`.
    pub const fn fp8e4m3b128(k: usize) -> Self {
        Self {
            format: QuantFormat::Fp8E4M3B128,
            quant_type: 55,
            k,
            block_size: 128,
        }
    }

    /// Bytes emitted for `rows` logical rows, including every row-local group.
    ///
    /// `None` means the geometry is not a valid exact extent (zero dimensions,
    /// overflow, or an aligned MQ4G256V2 tensor whose K is not divisible by
    /// 256).  The row product is intentionally supplied by the caller so
    /// stacked expert and ordinary matrix shapes share this check.  For the
    /// expert-blocked FP8E4M3B128 format `rows` is one block's rows (a single
    /// expert, `E = 1`); stacked extents come from
    /// [`Self::expert_payload_bytes`].
    pub fn payload_bytes(&self, rows: usize) -> Option<usize> {
        self.expert_payload_bytes(1, rows)
    }

    /// Bytes of `experts` stacked `[rows, k]` matrices.  Row-strided formats
    /// are `experts * rows` rows; FP8E4M3B128 is `experts` blocks of
    /// `S(rows, k) = round_up(rows*k + 2*ceil(rows/128)*ceil(k/128), 256)`.
    pub fn expert_payload_bytes(&self, experts: usize, rows: usize) -> Option<usize> {
        if experts == 0 || rows == 0 || self.k == 0 || self.block_size == 0 {
            return None;
        }
        let group_bytes = match self.format {
            QuantFormat::Mq4G256V2 if self.quant_type == 44 && self.block_size == 256 => 136,
            QuantFormat::Mq4G128V2 if self.quant_type == 53 && self.block_size == 128 => 68,
            QuantFormat::BF16 if self.quant_type == 16 && self.block_size == 1 => 2,
            QuantFormat::Fp8E4M3B128 if self.quant_type == 55 && self.block_size == 128 => {
                return rdna_compute::fp8_e4m3::b128_tensor_bytes(experts, rows, self.k);
            }
            _ => return None,
        };
        if matches!(self.format, QuantFormat::Mq4G256V2) && self.k % 256 != 0 {
            return None;
        }
        let groups = self
            .k
            .checked_add(self.block_size - 1)?
            .checked_div(self.block_size)?;
        experts
            .checked_mul(rows)?
            .checked_mul(groups)?
            .checked_mul(group_bytes)
    }
}

/// Exact Qwen4 manifest storage recipe.  The packed recipe uses the aligned
/// MQ4G256V2 form for every logical matrix whose K is divisible by 256 and the
/// row-local MQ4G128V2 form otherwise; the stock-FP8 recipe
/// ([`Self::stock_fp8`]) keeps every matrix source BF16 and ships the routed
/// experts as FP8E4M3B128 blocks and the PLE rows as FP8E4M3PT.  Non-matrix
/// payload classes retain their native wire representation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Qwen4QuantGeometry {
    pub matrix_aligned: QuantTensorGeometry,
    pub matrix_unaligned: QuantTensorGeometry,
    pub routed_gate_up: QuantTensorGeometry,
    pub routed_down: QuantTensorGeometry,
    pub ple_row_dtype: QuantFormat,
    pub scalar_dtype: QuantFormat,
    pub vector_dtype: QuantFormat,
    pub conv_dtype: QuantFormat,
    pub metadata_dtype: QuantFormat,
}

/// Compatibility spelling for callers that call expert projections simply
/// gate/up and down.
pub type QuantGeometry = Qwen4QuantGeometry;

const ROUTED_EXPERTS: usize = 512;
const ROUTED_GATE_UP_EXPERT_ROWS: usize = 1_280;
const ROUTED_DOWN_EXPERT_ROWS: usize = 2_560;
const ROUTED_GATE_UP_ROWS: usize = ROUTED_EXPERTS * ROUTED_GATE_UP_EXPERT_ROWS;
const ROUTED_GATE_UP_BYTES: usize = 891_289_600;
const ROUTED_DOWN_ROWS: usize = ROUTED_EXPERTS * ROUTED_DOWN_EXPERT_ROWS;
const ROUTED_DOWN_BYTES: usize = 445_644_800;
/// Stock-FP8 gate/up: 512 blocks of `S(1280, 2560) = 3,277,312` bytes.
const FP8_ROUTED_GATE_UP_BYTES: usize = 512 * 3_277_312;
/// Stock-FP8 down: 512 blocks of `S(2560, 640) = 1,638,656` bytes.
const FP8_ROUTED_DOWN_BYTES: usize = 512 * 1_638_656;

impl Qwen4QuantGeometry {
    /// Manifest-derived recipe for the pinned Qwen4 text checkpoint.
    pub const fn manifest() -> Self {
        Self {
            matrix_aligned: QuantTensorGeometry::mq4g256v2(2_560),
            matrix_unaligned: QuantTensorGeometry::mq4g128v2(640),
            routed_gate_up: QuantTensorGeometry::mq4g256v2(2_560),
            routed_down: QuantTensorGeometry::mq4g128v2(640),
            ple_row_dtype: QuantFormat::BF16,
            scalar_dtype: QuantFormat::BF16,
            vector_dtype: QuantFormat::BF16,
            conv_dtype: QuantFormat::BF16,
            metadata_dtype: QuantFormat::I64,
        }
    }

    /// The stock-FP8 recipe (`Qwen4TierRecipe::StockFp8`): BF16 trunk
    /// matrices at both widths, FP8E4M3B128 routed experts, FP8E4M3PT PLE rows.
    pub const fn stock_fp8() -> Self {
        Self {
            matrix_aligned: QuantTensorGeometry::bf16(2_560),
            matrix_unaligned: QuantTensorGeometry::bf16(640),
            routed_gate_up: QuantTensorGeometry::fp8e4m3b128(2_560),
            routed_down: QuantTensorGeometry::fp8e4m3b128(640),
            ple_row_dtype: QuantFormat::Fp8E4M3PerTensor,
            scalar_dtype: QuantFormat::BF16,
            vector_dtype: QuantFormat::BF16,
            conv_dtype: QuantFormat::BF16,
            metadata_dtype: QuantFormat::I64,
        }
    }

    /// The geometry of `recipe`'s artifact (the packed and synthetic recipes
    /// are judged against the pinned manifest geometry).
    pub const fn for_recipe(recipe: &Qwen4TierRecipe) -> Self {
        if recipe.is_stock_fp8() {
            Self::stock_fp8()
        } else {
            Self::manifest()
        }
    }

    pub const fn expected() -> Self {
        Self::manifest()
    }

    /// Whether any component names a stock-FP8 format; such a geometry is
    /// judged against [`Self::stock_fp8`] as a whole, so a mix is refused.
    pub fn is_stock_fp8(&self) -> bool {
        self.routed_gate_up.format == QuantFormat::Fp8E4M3B128
            || self.routed_down.format == QuantFormat::Fp8E4M3B128
            || self.ple_row_dtype == QuantFormat::Fp8E4M3PerTensor
    }

    /// Validate the exact matrix policy, native payload classes, and routed
    /// extents selected by the Qwen4 manifest (or the stock-FP8 recipe).
    pub fn validate(&self) -> Result<(), QuantAdmissionError> {
        let stock_fp8 = self.is_stock_fp8();
        let expected = if stock_fp8 {
            Self::stock_fp8()
        } else {
            Self::expected()
        };
        for (component, got, expected_geometry) in [
            (
                "aligned matrix",
                &self.matrix_aligned,
                &expected.matrix_aligned,
            ),
            (
                "unaligned matrix",
                &self.matrix_unaligned,
                &expected.matrix_unaligned,
            ),
            (
                "routed gate/up",
                &self.routed_gate_up,
                &expected.routed_gate_up,
            ),
            ("routed down", &self.routed_down, &expected.routed_down),
        ] {
            if got != expected_geometry {
                return Err(QuantAdmissionError::WrongGeometry {
                    component,
                    expected: expected_geometry.clone(),
                    got: got.clone(),
                });
            }
        }
        for (component, got, expected_dtype) in [
            ("PLE rows", &self.ple_row_dtype, &expected.ple_row_dtype),
            ("scalars", &self.scalar_dtype, &expected.scalar_dtype),
            ("vectors", &self.vector_dtype, &expected.vector_dtype),
            ("convolution", &self.conv_dtype, &expected.conv_dtype),
            ("metadata", &self.metadata_dtype, &expected.metadata_dtype),
        ] {
            if got != expected_dtype {
                return Err(QuantAdmissionError::WrongDtype {
                    component,
                    expected: expected_dtype.clone(),
                    got: got.clone(),
                });
            }
        }
        let (gate_up_bytes, down_bytes) = if stock_fp8 {
            (
                self.routed_gate_up
                    .expert_payload_bytes(ROUTED_EXPERTS, ROUTED_GATE_UP_EXPERT_ROWS)
                    == Some(FP8_ROUTED_GATE_UP_BYTES),
                self.routed_down
                    .expert_payload_bytes(ROUTED_EXPERTS, ROUTED_DOWN_EXPERT_ROWS)
                    == Some(FP8_ROUTED_DOWN_BYTES),
            )
        } else {
            (
                self.routed_gate_up.payload_bytes(ROUTED_GATE_UP_ROWS)
                    == Some(ROUTED_GATE_UP_BYTES),
                self.routed_down.payload_bytes(ROUTED_DOWN_ROWS) == Some(ROUTED_DOWN_BYTES),
            )
        };
        if !gate_up_bytes {
            return Err(QuantAdmissionError::WrongGeometry {
                component: "routed gate/up extent",
                expected: expected.routed_gate_up,
                got: self.routed_gate_up.clone(),
            });
        }
        if !down_bytes {
            return Err(QuantAdmissionError::WrongGeometry {
                component: "routed down extent",
                expected: expected.routed_down,
                got: self.routed_down.clone(),
            });
        }
        Ok(())
    }
}

/// One artifact identity presented to admission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen4Artifact {
    pub format: Qwen4ArtifactFormat,
    pub quant_geometry: Qwen4QuantGeometry,
}

impl Qwen4Artifact {
    pub const fn new(format: Qwen4ArtifactFormat, quant_geometry: Qwen4QuantGeometry) -> Self {
        Self {
            format,
            quant_geometry,
        }
    }

    pub const fn manifest_hfqm() -> Self {
        Self::new(Qwen4ArtifactFormat::Hfqm, Qwen4QuantGeometry::manifest())
    }

    /// The stock-FP8 HFQM artifact (`--recipe stock-fp8`).
    pub const fn stock_fp8_hfqm() -> Self {
        Self::new(Qwen4ArtifactFormat::Hfqm, Qwen4QuantGeometry::stock_fp8())
    }
}

/// Pure request to the Qwen4 admission boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen4AdmissionRequest {
    pub modality: InputModality,
    pub native_mtp: bool,
    pub effective_mesh: EffectiveMesh,
    pub artifact: Qwen4Artifact,
    /// GPU architecture the load targets (`gfx942`, `gfx1201`, ...).  `None`
    /// is an unknown device, which has no FP8 expert kernels: a stock-FP8
    /// artifact is refused there (fail closed).
    pub gpu_arch: Option<String>,
}

impl Qwen4AdmissionRequest {
    pub const fn text_ar(effective_mesh: EffectiveMesh, artifact: Qwen4Artifact) -> Self {
        Self {
            modality: InputModality::Text,
            native_mtp: false,
            effective_mesh,
            artifact,
            gpu_arch: None,
        }
    }

    pub const fn text_mtp(effective_mesh: EffectiveMesh, artifact: Qwen4Artifact) -> Self {
        Self {
            modality: InputModality::Text,
            native_mtp: true,
            effective_mesh,
            artifact,
            gpu_arch: None,
        }
    }

    /// This request on the device `gpu_arch`.
    pub fn on_device(self, gpu_arch: impl Into<String>) -> Self {
        Self {
            gpu_arch: Some(gpu_arch.into()),
            ..self
        }
    }
}

/// Successful pure admission result.  This value carries only validated
/// architecture/capability/topology facts; GPU allocations and source readers
/// remain owned by the later carrier lifecycle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen4Admission {
    pub architecture_id: u32,
    pub capabilities: Qwen4Capabilities,
    pub effective_mesh: EffectiveMesh,
    pub native_mtp: bool,
}

/// Why a request was refused.  Every variant is detected before allocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdmissionError {
    InvalidConfig(String),
    UnsupportedModality(InputModality),
    UnsupportedCapability(Qwen4Capability),
    NonSingleMesh(EffectiveMesh),
    UnsupportedFormat(Qwen4ArtifactFormat),
    WrongGeometry {
        component: &'static str,
        expected: QuantTensorGeometry,
        got: QuantTensorGeometry,
    },
    WrongDtype {
        component: &'static str,
        expected: QuantFormat,
        got: QuantFormat,
    },
    /// A stock-FP8 artifact (qt=55 experts) on a device without FP8 expert
    /// kernels (anything but [`FP8_EXPERTS_ARCH`]).
    Fp8ExpertsUnsupported {
        gpu_arch: Option<String>,
    },
    /// Native MTP requested for the stock-FP8 artifact admitted without its head sidecar.
    ArOnlyArtifact,
    /// The stock-FP8 artifact is admitted fully resident on one device only
    /// (P1): no expert parallelism.
    Fp8SingleDeviceOnly(EffectiveMesh),
    /// The stock-FP8 artifact with routed experts placed in host RAM.
    Fp8HostMappedExperts,
    /// The stock-FP8 artifact under a config that is not the AR-only
    /// Flash-Next profile.
    Fp8ProfileMismatch(String),
}

/// Alias retaining the distinction between format/quant admission errors and
/// config parser errors for callers that expose one error type.
pub type Qwen4AdmissionError = AdmissionError;

impl fmt::Display for AdmissionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(error) => write!(f, "qwen4 admission: invalid config: {error}"),
            Self::UnsupportedModality(modality) => {
                write!(f, "qwen4 admission: unsupported modality {modality:?}")
            }
            Self::UnsupportedCapability(capability) => {
                write!(f, "qwen4 admission: unsupported capability {capability:?}")
            }
            Self::NonSingleMesh(mesh) => write!(
                f,
                "qwen4 admission: only pp=1, tp=1 with expert-parallel ep in 1..=8, or tp=1, ep=1 with pipeline-parallel pp in 2..=8 is admitted (pp={}, tp={}, ep={})",
                mesh.pp, mesh.tp, mesh.ep
            ),
            Self::UnsupportedFormat(format) => {
                write!(f, "qwen4 admission: unsupported artifact format {format:?}")
            }
            Self::WrongGeometry {
                component,
                expected,
                got,
            } => write!(
                f,
                "qwen4 admission: {component} geometry {:?} does not match {:?}",
                got, expected
            ),
            Self::WrongDtype {
                component,
                expected,
                got,
            } => write!(
                f,
                "qwen4 admission: {component} dtype {got:?} does not match {expected:?}"
            ),
            Self::Fp8ExpertsUnsupported { gpu_arch } => write!(
                f,
                "qwen4 admission: the stock-FP8 artifact (qt=55 FP8E4M3B128 routed experts) \
                 runs only on {FP8_EXPERTS_ARCH} (MI300X); this device is {}; refusing it \
                 (no FP8 expert kernels for this arch)",
                gpu_arch.as_deref().unwrap_or("unknown")
            ),
            Self::ArOnlyArtifact => write!(
                f,
                "qwen4 admission: the stock-FP8 artifact was admitted without its MTP head \
                 sidecar (<model>.mtp); native MTP (--spec mtp) cannot be admitted"
            ),
            Self::Fp8SingleDeviceOnly(mesh) => write!(
                f,
                "qwen4 admission: the stock-FP8 artifact loads fully resident on one device \
                 (pp=1, tp=1, ep=1); got pp={}, tp={}, ep={}",
                mesh.pp, mesh.tp, mesh.ep
            ),
            Self::Fp8HostMappedExperts => write!(
                f,
                "qwen4 admission: the stock-FP8 artifact keeps every routed expert resident in \
                 device memory; host-mapped experts ({}) are refused",
                crate::expert_residency::EXPERT_VRAM_LAYERS_ENV
            ),
            Self::Fp8ProfileMismatch(profile) => write!(
                f,
                "qwen4 admission: the stock-FP8 artifact is admitted under the AR-only \
                 Flash-Next profile, not {profile}"
            ),
        }
    }
}

impl std::error::Error for AdmissionError {}

/// Successful, immutable result passed to a later loader/carrier.  It carries
/// no GPU or reader handle and therefore cannot imply production admission.
impl Qwen4Admission {
    /// Validate all pure boundaries in order: config, capability, mesh,
    /// container, then exact quant recipe.
    pub fn admit(
        config: &Qwen4Config,
        request: &Qwen4AdmissionRequest,
    ) -> Result<Self, AdmissionError> {
        config.validate().map_err(AdmissionError::InvalidConfig)?;
        let stock_fp8 = request.artifact.quant_geometry.is_stock_fp8();
        if stock_fp8 && !config.profile.is_stock_fp8() {
            return Err(AdmissionError::Fp8ProfileMismatch(
                config.profile.id().to_string(),
            ));
        }
        let capabilities = if request.native_mtp {
            Qwen4Capabilities::text_mtp()
        } else {
            Qwen4Capabilities::text_ar()
        };
        let capabilities = match request.gpu_arch.as_deref() {
            Some(gpu_arch) => capabilities.on_device(gpu_arch),
            None => capabilities,
        };
        if !capabilities.supports_modality(request.modality) {
            return Err(AdmissionError::UnsupportedModality(request.modality));
        }
        if request.native_mtp && !capabilities.supports(Qwen4Capability::NativeMtp) {
            return Err(AdmissionError::UnsupportedCapability(
                Qwen4Capability::NativeMtp,
            ));
        }
        if !request.effective_mesh.is_qwen4_admitted() {
            return Err(AdmissionError::NonSingleMesh(request.effective_mesh));
        }
        if !request.artifact.format.is_hfqm() {
            return Err(AdmissionError::UnsupportedFormat(
                request.artifact.format.clone(),
            ));
        }
        if stock_fp8 {
            check_fp8_device(
                capabilities,
                request.gpu_arch.as_deref(),
                request.effective_mesh,
                request.native_mtp,
                false,
                config.profile == crate::config::Qwen4Profile::FlashNextStockMtp,
            )?;
        }
        request
            .artifact
            .quant_geometry
            .validate()
            .map_err(|error| match error {
                QuantAdmissionError::WrongGeometry {
                    component,
                    expected,
                    got,
                } => AdmissionError::WrongGeometry {
                    component,
                    expected,
                    got,
                },
                QuantAdmissionError::WrongDtype {
                    component,
                    expected,
                    got,
                } => AdmissionError::WrongDtype {
                    component,
                    expected,
                    got,
                },
            })?;
        Ok(Self {
            architecture_id: ARCH_ID,
            capabilities,
            effective_mesh: request.effective_mesh,
            native_mtp: request.native_mtp,
        })
    }

    /// Free-function style entry point for callers that do not retain an
    /// admission object as a type.
    pub fn admit_text_mtp(
        config: &Qwen4Config,
        request: &Qwen4AdmissionRequest,
    ) -> Result<Self, AdmissionError> {
        Self::admit(config, request)
    }
}

/// The stock-FP8 artifact's device admission: FP8 expert kernels (gfx942
/// only), single device, every routed expert resident; native MTP only with
/// the head sidecar attached.
fn check_fp8_device(
    capabilities: Qwen4Capabilities,
    gpu_arch: Option<&str>,
    mesh: EffectiveMesh,
    native_mtp: bool,
    host_mapped_experts: bool,
    head_attached: bool,
) -> Result<(), AdmissionError> {
    if !capabilities.supports(Qwen4Capability::Fp8Experts) {
        return Err(AdmissionError::Fp8ExpertsUnsupported {
            gpu_arch: gpu_arch.map(str::to_string),
        });
    }
    if native_mtp && !head_attached {
        return Err(AdmissionError::ArOnlyArtifact);
    }
    if !mesh.is_single() {
        return Err(AdmissionError::Fp8SingleDeviceOnly(mesh));
    }
    if host_mapped_experts {
        return Err(AdmissionError::Fp8HostMappedExperts);
    }
    Ok(())
}

/// Loader-side device admission of an admitted artifact's `recipe`: the
/// capabilities `gpu_arch` offers it.  A recipe without FP8 experts is
/// device-neutral here; the stock-FP8 recipe is admitted only on
/// [`FP8_EXPERTS_ARCH`], on a single device, with `native_mtp` admitted only
/// when the head sidecar is attached (`head_attached`; else `ArOnlyArtifact`),
/// and with no host-mapped expert placement requested
/// (`host_mapped_experts`: an explicit `HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS`).
pub fn admit_recipe_on_device(
    recipe: &Qwen4TierRecipe,
    gpu_arch: &str,
    mesh: EffectiveMesh,
    native_mtp: bool,
    host_mapped_experts: bool,
    head_attached: bool,
) -> Result<Qwen4Capabilities, AdmissionError> {
    let capabilities = if native_mtp {
        Qwen4Capabilities::text_mtp()
    } else {
        Qwen4Capabilities::text_ar()
    }
    .on_device(gpu_arch);
    if recipe.fp8_experts() {
        check_fp8_device(
            capabilities,
            Some(gpu_arch),
            mesh,
            native_mtp,
            host_mapped_experts,
            head_attached,
        )?;
    }
    Ok(capabilities)
}

/// Errors local to quant geometry validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuantAdmissionError {
    WrongGeometry {
        component: &'static str,
        expected: QuantTensorGeometry,
        got: QuantTensorGeometry,
    },
    WrongDtype {
        component: &'static str,
        expected: QuantFormat,
        got: QuantFormat,
    },
}

/// Free device bytes a pipeline-parallel rank keeps beyond its band census:
/// the first long request's other lazy allocations, kernel-cache warmup and
/// VMM page rounding (the loader's single-device headroom, 2 GiB).
pub const QWEN4_PP_RESERVE_BYTES: u64 = 2 << 30;

/// Inputs of [`qwen4_pp_band_budget`]. Everything is a value the loader has
/// before its first allocation: the manifest's entries, payload sizes from the
/// open artifact, the resolved state format, context and scratch mode, and
/// each rank's measured free VRAM.
pub struct Qwen4PpBudgetInputs<'a> {
    pub config: &'a Qwen4Config,
    /// The admitted placement (`BandPlacement::with_first_band_min(2)`).
    pub placement: &'a hipfire_runtime::mesh_placement::BandPlacement,
    /// `manifest.weights`, the whole manifest. The output band's `mtp.*` head
    /// entries (experts, projections, the tied embedding's materialization)
    /// are counted: the band loads them whether or not the head attaches.
    pub weights: &'a [hipfire_runtime::weight_manifest::WeightEntry],
    pub bytes_of: &'a dyn Fn(&hipfire_runtime::weight_manifest::WeightEntry) -> Option<u64>,
    pub format: crate::Qwen4StateFormat,
    /// The actual backend with the whole admitted context mapped.
    pub context: &'a crate::Qwen4ContextCommit,
    pub chunk_rows: usize,
    pub mode: crate::gpu_forward::Qwen4ForwardScratchMode,
    /// Free VRAM by LOGICAL rank (`AdmittedMesh::ranks` order).
    pub free_per_rank: &'a [u64],
    pub reserve_bytes: u64,
    /// The native MTP request, `None` for an AR load (no MTP reserve and no
    /// row capture on any band): the output band charges the head and its
    /// request buffers, every band its own row capture.
    pub native_mtp: Option<crate::expert_residency::NativeMtpCensusInputs>,
}

/// One rank's admitted budget line: the band census terms and the rank's free
/// VRAM. The line is kept whether or not the rank fits, so the auto-fit search
/// ([`qwen4_pp_auto_fit`]) can rank candidates by their headroom.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen4PpRankFit {
    pub stage: usize,
    pub band: std::ops::Range<usize>,
    /// Weights + F16 shadows + QSA context + state + forward scratch + native
    /// MTP + reserve.
    pub required: u64,
    /// The native MTP term of `required` (0 for an AR load).
    pub native_mtp: u64,
    pub free: u64,
    /// The other terms of `required` (the refusal text names each).
    pub weight_bytes: u64,
    pub f16_shadow_bytes: u64,
    pub qsa_context: u64,
    pub state: u64,
    pub forward: u64,
    pub reserve: u64,
}

impl Qwen4PpRankFit {
    /// The rank holds its census.
    pub fn fits(&self) -> bool {
        self.required <= self.free
    }

    /// Bytes the census exceeds the rank's free VRAM by (0 when it fits).
    pub fn shortfall(&self) -> u64 {
        self.required.saturating_sub(self.free)
    }
}

/// Per-rank census lines for `i.placement` at `i.chunk_rows`, one for every
/// rank whether or not it fits. Pure; only a malformed input (a reading count
/// that differs from the rank count, a band the census refuses, an overflow)
/// is an error.
pub fn qwen4_pp_rank_lines(i: &Qwen4PpBudgetInputs<'_>) -> Result<Vec<Qwen4PpRankFit>, String> {
    use crate::weights::Qwen4BandScope;
    let stages = i.placement.stages();
    let prefix = format!("qwen4: pipeline parallelism (topology.pp = {stages})");
    if i.free_per_rank.len() != stages {
        return Err(format!(
            "{prefix}: {} free-VRAM readings for {stages} ranks",
            i.free_per_rank.len()
        ));
    }
    let overflow = || format!("{prefix}: band budget overflows");
    let mut lines = Vec::with_capacity(stages);
    for stage in 0..stages {
        let scope = Qwen4BandScope::from_placement(i.config, i.placement, stage)
            .map_err(|error| format!("{prefix}: {error}"))?;
        let census = crate::expert_residency::band_census_native_mtp(
            i.config,
            i.weights,
            |entry| (i.bytes_of)(entry),
            &scope,
            i.format,
            i.context,
            i.chunk_rows,
            i.mode,
            i.native_mtp.as_ref(),
        )
        .map_err(|error| format!("{prefix}: rank {stage}: {error}"))?;
        let weight_bytes = census
            .weight_bytes()
            .map_err(|e| format!("{prefix}: {e}"))?;
        let native_mtp = census
            .mtp
            .reserve()
            .map_err(|e| format!("{prefix}: rank {stage}: {e}"))?;
        let required = [
            census.f16_shadow_bytes,
            census.state.qsa_context,
            census.state.state,
            census.state.forward,
            native_mtp,
            i.reserve_bytes,
        ]
        .into_iter()
        .try_fold(weight_bytes, |total, bytes| total.checked_add(bytes))
        .ok_or_else(overflow)?;
        lines.push(Qwen4PpRankFit {
            stage,
            band: scope.band,
            required,
            native_mtp,
            free: i.free_per_rank[stage],
            weight_bytes,
            f16_shadow_bytes: census.f16_shadow_bytes,
            qsa_context: census.state.qsa_context,
            state: census.state.state,
            forward: census.state.forward,
            reserve: i.reserve_bytes,
        });
    }
    Ok(lines)
}

/// The refusal for `lines` (from [`qwen4_pp_rank_lines`]), naming `topology.pp`,
/// every failing rank and its shortfall; `None` when every rank fits. `note`
/// (the auto-fit search's nearest candidate) is inserted before the advice.
pub fn qwen4_pp_budget_refusal(lines: &[Qwen4PpRankFit], note: Option<&str>) -> Option<String> {
    let failures: Vec<String> = lines
        .iter()
        .filter(|line| !line.fits())
        .map(|line| {
            format!(
                "rank {} (layers {}..{}) needs {} bytes ({} weights + {} F16 shadows + {} QSA context + {} state + {} forward scratch + {} native MTP + {} reserve) but {} bytes are free (short by {})",
                line.stage,
                line.band.start,
                line.band.end,
                line.required,
                line.weight_bytes,
                line.f16_shadow_bytes,
                line.qsa_context,
                line.state,
                line.forward,
                line.native_mtp,
                line.reserve,
                line.free,
                line.shortfall()
            )
        })
        .collect();
    if failures.is_empty() {
        return None;
    }
    Some(format!(
        "qwen4: pipeline parallelism (topology.pp = {}): band does not fit device memory: {}{}; rebalance HIPFIRE_PP_LAYERS or lower memory.max_seq (host-mapped band layers are not admitted)",
        lines.len(),
        failures.join("; "),
        note.map(|note| format!("; {note}")).unwrap_or_default()
    ))
}

/// Per-rank band budget from the S5 band census: every rank's required bytes
/// must fit its free VRAM (no host-mapped band layers). Pure; refuses before
/// any allocation, naming `topology.pp`, every failing rank and its shortfall.
pub fn qwen4_pp_band_budget(i: &Qwen4PpBudgetInputs<'_>) -> Result<Vec<Qwen4PpRankFit>, String> {
    let lines = qwen4_pp_rank_lines(i)?;
    match qwen4_pp_budget_refusal(&lines, None) {
        Some(refusal) => Err(refusal),
        None => Ok(lines),
    }
}

/// The PLE layer is layer 1, so the first band must hold at least this many
/// layers (`BandPlacement::with_first_band_min`).
pub const QWEN4_PP_FIRST_BAND_MIN_LAYERS: usize = 2;

/// Prefill chunk rungs the PP auto-fit steps down through when the requested
/// chunk fits no split, largest first; only the rungs below the requested
/// chunk are tried (every rung is a whole 256-row GEMM tile multiple).
pub const QWEN4_PP_FIT_CHUNK_RUNGS: [usize; 4] = [2048, 1024, 512, 256];

/// The chunk rungs of one auto-fit: `requested` first, then each
/// [`QWEN4_PP_FIT_CHUNK_RUNGS`] entry below it.
pub fn qwen4_pp_fit_chunk_rungs(requested: usize) -> Vec<usize> {
    std::iter::once(requested)
        .chain(
            QWEN4_PP_FIT_CHUNK_RUNGS
                .into_iter()
                .filter(|&rows| rows < requested),
        )
        .collect()
}

/// The default (the admitted split at the requested chunk) that an auto-fit
/// moved off, with its tightest rank.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen4PpDefaultShortfall {
    pub counts: Vec<usize>,
    pub chunk_rows: usize,
    /// The rank with the largest shortfall and that shortfall in bytes.
    pub rank: usize,
    pub bytes: u64,
}

/// The split and chunk one PP load runs with, and every rank's budget line at
/// them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen4PpAutoFit {
    /// Layers per band, logical band order.
    pub counts: Vec<usize>,
    /// The canonical prefill chunk of every band and the mesh.
    pub chunk_rows: usize,
    pub fits: Vec<Qwen4PpRankFit>,
    /// `Some` when the search moved off the default; `None` when the admitted
    /// split fit at the requested chunk (today's load, unchanged).
    pub moved_from: Option<Qwen4PpDefaultShortfall>,
}

fn join_counts(counts: &[usize]) -> String {
    counts
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

impl Qwen4PpAutoFit {
    /// The one log line of a search that moved off the default.
    pub fn log_line(&self) -> Option<String> {
        let from = self.moved_from.as_ref()?;
        Some(format!(
            "qwen4 PP auto-fit: layers {} chunk {} (uniform {} at chunk {} short by {} MiB on rank {})",
            join_counts(&self.counts),
            self.chunk_rows,
            join_counts(&from.counts),
            from.chunk_rows,
            from.bytes.div_ceil(1 << 20),
            from.rank
        ))
    }
}

/// A candidate (split, chunk) and its budget lines.
#[derive(Clone)]
struct FitCandidate {
    counts: Vec<usize>,
    chunk_rows: usize,
    fits: Vec<Qwen4PpRankFit>,
}

impl FitCandidate {
    fn all_fit(&self) -> bool {
        self.fits.iter().all(Qwen4PpRankFit::fits)
    }

    /// `(rank, bytes)` of the largest shortfall (the earliest rank on a tie).
    fn worst_shortfall(&self) -> (usize, u64) {
        self.fits.iter().fold((0, 0), |worst, line| {
            if line.shortfall() > worst.1 {
                (line.stage, line.shortfall())
            } else {
                worst
            }
        })
    }

    /// The line with the largest `required / free` (exact; the earliest rank
    /// on a tie).
    fn tightest(&self) -> &Qwen4PpRankFit {
        let mut tightest = &self.fits[0];
        for line in &self.fits[1..] {
            if u128::from(line.required) * u128::from(tightest.free)
                > u128::from(tightest.required) * u128::from(line.free)
            {
                tightest = line;
            }
        }
        tightest
    }
}

/// Sum of absolute per-band count differences from the admitted split.
fn split_distance(counts: &[usize], admitted: &[usize]) -> usize {
    counts
        .iter()
        .zip(admitted)
        .map(|(count, admitted)| count.abs_diff(*admitted))
        .sum()
}

/// Exact order of two ranks' loads, `required / free`.
fn load_order(a: &Qwen4PpRankFit, b: &Qwen4PpRankFit) -> std::cmp::Ordering {
    (u128::from(a.required) * u128::from(b.free))
        .cmp(&(u128::from(b.required) * u128::from(a.free)))
}

/// Balance order of two candidates at one chunk (`Less` = `a` more balanced):
/// the lower maximum `required / free` over the ranks. For more than two
/// ranks the whole load vector, sorted tightest first, is compared
/// lexicographically, so the steepest-descent walk of [`search_rung`] also
/// progresses across plateaus where two ranks share the maximum.
fn balance_order(a: &FitCandidate, b: &FitCandidate) -> std::cmp::Ordering {
    if a.fits.len() <= 2 {
        return load_order(a.tightest(), b.tightest());
    }
    fn tightest_first(candidate: &FitCandidate) -> Vec<&Qwen4PpRankFit> {
        let mut lines: Vec<&Qwen4PpRankFit> = candidate.fits.iter().collect();
        lines.sort_by(|x, y| load_order(y, x));
        lines
    }
    let (a_lines, b_lines) = (tightest_first(a), tightest_first(b));
    a_lines
        .iter()
        .zip(&b_lines)
        .map(|(x, y)| load_order(x, y))
        .find(|order| order.is_ne())
        .unwrap_or(std::cmp::Ordering::Equal)
}

/// Whether `a` is a better candidate than `b` at one chunk: more balanced
/// ([`balance_order`]), then the closer to the admitted split, then the larger
/// earlier bands. A strict total order on distinct splits.
fn balanced_before(a: &FitCandidate, b: &FitCandidate, admitted: &[usize]) -> bool {
    balance_order(a, b)
        .then_with(|| split_distance(&a.counts, admitted).cmp(&split_distance(&b.counts, admitted)))
        .then_with(|| b.counts.cmp(&a.counts))
        .is_lt()
}

/// Whether `a` is nearer to fitting than `b` (both non-fitting): the smaller
/// worst shortfall, then the larger chunk, then the admitted-split tie breaks
/// of [`balanced_before`].
fn nearer_before(a: &FitCandidate, b: &FitCandidate, admitted: &[usize]) -> bool {
    a.worst_shortfall()
        .1
        .cmp(&b.worst_shortfall().1)
        .then_with(|| b.chunk_rows.cmp(&a.chunk_rows))
        .then_with(|| split_distance(&a.counts, admitted).cmp(&split_distance(&b.counts, admitted)))
        .then_with(|| b.counts.cmp(&a.counts))
        .is_lt()
}

/// Search the splits of one chunk rung: every split for two stages, a
/// deterministic steepest-descent walk from `admitted` otherwise (each step
/// moves one layer between any ordered pair of bands and takes the neighbour
/// that is [`balanced_before`] every other; it stops at a local optimum, which
/// for the monotone per-band census is the minimum of the maximum
/// `required / free`). Returns the best candidate if it fits. `best_miss`
/// tracks the nearest non-fitting candidate over every rung. `seed` is the
/// already evaluated admitted split at this chunk, if any.
fn search_rung<E>(
    admitted: &[usize],
    first_band_min: usize,
    chunk_rows: usize,
    evaluate: &mut E,
    seed: Option<Vec<Qwen4PpRankFit>>,
    best_miss: &mut Option<FitCandidate>,
) -> Result<Option<FitCandidate>, String>
where
    E: FnMut(&[usize], usize) -> Result<Vec<Qwen4PpRankFit>, String>,
{
    use std::collections::BTreeMap;
    let stages = admitted.len();
    let n_layers: usize = admitted.iter().sum();
    let min_count = |stage: usize| if stage == 0 { first_band_min.max(1) } else { 1 };
    let mut seen: BTreeMap<Vec<usize>, FitCandidate> = BTreeMap::new();
    let mut candidate = |counts: Vec<usize>,
                         seen: &mut BTreeMap<Vec<usize>, FitCandidate>,
                         best_miss: &mut Option<FitCandidate>|
     -> Result<FitCandidate, String> {
        if let Some(known) = seen.get(&counts) {
            return Ok(known.clone());
        }
        let fits = match &seed {
            Some(seed) if counts.as_slice() == admitted => seed.clone(),
            _ => evaluate(&counts, chunk_rows)?,
        };
        let found = FitCandidate {
            counts: counts.clone(),
            chunk_rows,
            fits,
        };
        if !found.all_fit()
            && best_miss
                .as_ref()
                .is_none_or(|best| nearer_before(&found, best, admitted))
        {
            *best_miss = Some(found.clone());
        }
        seen.insert(counts, found.clone());
        Ok(found)
    };
    if stages == 2 {
        let mut best: Option<FitCandidate> = None;
        for first in min_count(0)..n_layers {
            let found = candidate(vec![first, n_layers - first], &mut seen, &mut *best_miss)?;
            if found.all_fit()
                && best
                    .as_ref()
                    .is_none_or(|best| balanced_before(&found, best, admitted))
            {
                best = Some(found);
            }
        }
        return Ok(best);
    }
    let mut current = candidate(admitted.to_vec(), &mut seen, &mut *best_miss)?;
    loop {
        let mut best: Option<FitCandidate> = None;
        for src in 0..stages {
            if current.counts[src] <= min_count(src) {
                continue;
            }
            for dst in (0..stages).filter(|&dst| dst != src) {
                let mut counts = current.counts.clone();
                counts[src] -= 1;
                counts[dst] += 1;
                let found = candidate(counts, &mut seen, &mut *best_miss)?;
                if best
                    .as_ref()
                    .is_none_or(|best| balanced_before(&found, best, admitted))
                {
                    best = Some(found);
                }
            }
        }
        match best {
            Some(best) if balanced_before(&best, &current, admitted) => current = best,
            _ => break,
        }
    }
    Ok(current.all_fit().then_some(current))
}

/// The one PP split-and-chunk decision of a load, pure over `evaluate(counts,
/// chunk_rows)` (the per-rank census lines, [`qwen4_pp_rank_lines`], of a
/// candidate; only a malformed input is an `Err`).
///
/// `admitted` is the admitted split (the uniform default, or the operator's
/// `HIPFIRE_PP_LAYERS` when `explicit_layers`); `requested` the default
/// prefill chunk (`HIPFIRE_PREFILL_CHUNK_ROWS` or the arch default, never
/// above `max_seq`); `chunk_pinned` that the chunk is the operator's.
///
/// - `explicit_layers`: the operator's split at the requested chunk is the
///   whole answer; nothing is searched and a miss is the budget refusal, as
///   it always was.
/// - The admitted split at `requested` fitting is the answer unchanged
///   (`moved_from` is `None`).
/// - Otherwise the chunk steps down [`qwen4_pp_fit_chunk_rungs`] (only
///   `requested` when `chunk_pinned`) and the largest chunk for which some
///   split fits wins; at that chunk the most balanced fitting split (lowest
///   maximum `required / free`; ties: closest to the admitted split, then
///   larger earlier bands) is chosen. Splits honor `first_band_min`.
/// - When nothing fits the refusal reports the nearest candidate seen.
pub fn qwen4_pp_auto_fit<E>(
    admitted: &[usize],
    explicit_layers: bool,
    first_band_min: usize,
    requested: usize,
    chunk_pinned: bool,
    mut evaluate: E,
) -> Result<Qwen4PpAutoFit, String>
where
    E: FnMut(&[usize], usize) -> Result<Vec<Qwen4PpRankFit>, String>,
{
    let default = FitCandidate {
        counts: admitted.to_vec(),
        chunk_rows: requested,
        fits: evaluate(admitted, requested)?,
    };
    if default.all_fit() || explicit_layers {
        return match qwen4_pp_budget_refusal(&default.fits, None) {
            Some(refusal) => Err(refusal),
            None => Ok(Qwen4PpAutoFit {
                counts: default.counts,
                chunk_rows: requested,
                fits: default.fits,
                moved_from: None,
            }),
        };
    }
    let rungs = if chunk_pinned {
        vec![requested]
    } else {
        qwen4_pp_fit_chunk_rungs(requested)
    };
    let (default_rank, default_bytes) = default.worst_shortfall();
    let moved_from = Qwen4PpDefaultShortfall {
        counts: admitted.to_vec(),
        chunk_rows: requested,
        rank: default_rank,
        bytes: default_bytes,
    };
    let mut best_miss = Some(default.clone());
    for (index, &chunk_rows) in rungs.iter().enumerate() {
        let seed = (index == 0).then(|| default.fits.clone());
        let found = search_rung(
            admitted,
            first_band_min,
            chunk_rows,
            &mut evaluate,
            seed,
            &mut best_miss,
        )?;
        if let Some(found) = found {
            return Ok(Qwen4PpAutoFit {
                counts: found.counts,
                chunk_rows,
                fits: found.fits,
                moved_from: Some(moved_from),
            });
        }
    }
    let best = best_miss.unwrap_or(default);
    let (rank, bytes) = best.worst_shortfall();
    let note = format!(
        "auto-fit searched chunk rows {} and every split with at least {first_band_min} layers in band 0 and none fits; the nearest, layers {} at chunk {}, is short by {} MiB on rank {rank} (the default, uniform {} at chunk {requested}, was short by {} MiB on rank {default_rank})",
        join_counts(&rungs),
        join_counts(&best.counts),
        best.chunk_rows,
        bytes.div_ceil(1 << 20),
        join_counts(admitted),
        default_bytes.div_ceil(1 << 20),
    );
    Err(qwen4_pp_budget_refusal(&best.fits, Some(&note))
        .unwrap_or_else(|| format!("qwen4: pipeline parallelism: {note}")))
}

/// Every prefill chunk a PP mesh may settle on, largest first (each a whole
/// 256-row tile multiple): the union of the ordinary admission rungs
/// (8192, 4096, 2048, 1536) and the auto-fit rungs (2048, 1024, 512, 256).
/// [`qwen4_pp_chunk_ladder`] keeps the rungs at or below a ceiling.
pub const QWEN4_PP_CHUNK_LADDER: [usize; 7] = [8192, 4096, 2048, 1536, 1024, 512, 256];

/// The rungs the mesh-wide chunk negotiation ([`qwen4_pp_negotiate_chunk`])
/// tries, largest first: `ceiling` itself (the chunk the auto-fit settled on,
/// the requested chunk when it fit), then every [`QWEN4_PP_CHUNK_LADDER`]
/// rung below it. A `pinned` chunk (`HIPFIRE_PREFILL_CHUNK_ROWS`) is the only
/// rung: it is never lowered.
pub fn qwen4_pp_chunk_ladder(ceiling: usize, pinned: bool) -> Vec<usize> {
    if pinned {
        return vec![ceiling];
    }
    std::iter::once(ceiling)
        .chain(
            QWEN4_PP_CHUNK_LADDER
                .into_iter()
                .filter(|&rows| rows < ceiling),
        )
        .collect()
}

/// One band's line of the chunk negotiation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen4PpBandChunk {
    pub stage: usize,
    pub band: std::ops::Range<usize>,
    /// The capacity this band was judged against (bytes).
    pub capacity: u64,
    /// The largest rung of the ladder whose census fits `capacity`; `None`
    /// when not even the smallest rung does.
    pub max_rows: Option<usize>,
    /// The census bytes at the ladder's first (largest) rung and at the
    /// smallest one, for the refusal text.
    pub required_largest: u64,
    pub required_smallest: u64,
}

/// The mesh-wide canonical prefill chunk and the per-band admissible rungs it
/// was the minimum of.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen4PpChunkNegotiation {
    /// The chunk every band and the mesh load at: the minimum over the bands
    /// of each band's largest admissible rung.
    pub chunk_rows: usize,
    /// The chunk the load asked for (the ladder's first rung).
    pub requested: usize,
    pub bands: Vec<Qwen4PpBandChunk>,
}

impl Qwen4PpChunkNegotiation {
    /// The one log line naming every band's admissible rung and the chosen
    /// canonical chunk.
    pub fn log_line(&self) -> String {
        let bands = self
            .bands
            .iter()
            .map(|band| {
                format!(
                    "band {} (layers {}..{}) {}",
                    band.stage,
                    band.band.start,
                    band.band.end,
                    band.max_rows
                        .map_or_else(|| "none".to_string(), |rows| rows.to_string())
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        let limit = self
            .bands
            .iter()
            .filter(|band| band.max_rows == Some(self.chunk_rows))
            .map(|band| band.stage.to_string())
            .collect::<Vec<_>>()
            .join(",");
        if self.chunk_rows == self.requested {
            format!(
                "qwen4 PP canonical chunk: {} rows (every band admits it; per-band largest admissible rung: {bands})",
                self.chunk_rows
            )
        } else {
            format!(
                "qwen4 PP canonical chunk: {} rows, lowered from {} for the whole mesh by band {limit} (per-band largest admissible rung: {bands})",
                self.chunk_rows, self.requested
            )
        }
    }
}

/// The mesh-wide canonical prefill chunk, decided once before any band loads.
///
/// Every band must prefill the same chunk (the bands' chunk is the mesh's
/// canonical chunk: an edge carries one chunk's streams and a different
/// chunking changes the numerics), but each band's VRAM budget admits a
/// different largest rung. `tables` holds, for each ladder rung largest first
/// ([`qwen4_pp_chunk_ladder`]), every rank's census line at that rung
/// ([`qwen4_pp_rank_lines`]); `capacity[stage]` is the VRAM that rank can hold
/// (the load passes the smaller of its free VRAM and its freshly opened
/// device's planned capacity, the figure the band's own admission reads). A
/// band admits a rung when its census at that rung fits `capacity`; the
/// canonical chunk is the smallest of the bands' largest admitted rungs, so a
/// band that is tighter than the rest lowers the chunk for everyone instead
/// of failing its own load.
///
/// With `pinned` (a single-rung ladder: the operator's chunk) nothing is
/// lowered: a band that cannot admit it is a refusal naming that band and the
/// chunk it could admit.
pub fn qwen4_pp_negotiate_chunk(
    tables: &[(usize, Vec<Qwen4PpRankFit>)],
    capacity: &[u64],
    pinned: bool,
) -> Result<Qwen4PpChunkNegotiation, String> {
    let (requested, first) = tables
        .first()
        .ok_or_else(|| "qwen4: topology.pp chunk negotiation without a rung".to_string())?;
    let stages = first.len();
    if capacity.len() != stages || tables.iter().any(|(_, lines)| lines.len() != stages) {
        return Err(format!(
            "qwen4: topology.pp chunk negotiation: {} capacity readings for {stages} ranks",
            capacity.len()
        ));
    }
    let smallest = &tables[tables.len() - 1].1;
    let bands: Vec<Qwen4PpBandChunk> = (0..stages)
        .map(|stage| Qwen4PpBandChunk {
            stage,
            band: first[stage].band.clone(),
            capacity: capacity[stage],
            max_rows: tables
                .iter()
                .find(|(_, lines)| lines[stage].required <= capacity[stage])
                .map(|(rows, _)| *rows),
            required_largest: first[stage].required,
            required_smallest: smallest[stage].required,
        })
        .collect();
    let unfit: Vec<String> = bands
        .iter()
        .filter(|band| band.max_rows.is_none())
        .map(|band| {
            format!(
                "band {} (layers {}..{}) needs {} MiB even at the smallest chunk ({} rows) but its device holds {} MiB",
                band.stage,
                band.band.start,
                band.band.end,
                band.required_smallest.div_ceil(1 << 20),
                tables[tables.len() - 1].0,
                band.capacity >> 20
            )
        })
        .collect();
    if !pinned && !unfit.is_empty() {
        return Err(format!(
            "qwen4: topology.pp: no prefill chunk fits every band: {}; rebalance HIPFIRE_PP_LAYERS or lower memory.max_seq",
            unfit.join("; ")
        ));
    }
    let chunk_rows = bands
        .iter()
        .filter_map(|band| band.max_rows)
        .min()
        .unwrap_or(*requested);
    if pinned && bands.iter().any(|band| band.max_rows != Some(*requested)) {
        let short: Vec<String> = bands
            .iter()
            .filter(|band| band.max_rows != Some(*requested))
            .map(|band| {
                format!(
                    "band {} (layers {}..{}) needs {} MiB at {requested} rows but its device holds {} MiB",
                    band.stage,
                    band.band.start,
                    band.band.end,
                    band.required_largest.div_ceil(1 << 20),
                    band.capacity >> 20
                )
            })
            .collect();
        return Err(format!(
            "qwen4: topology.pp: the explicit prefill chunk HIPFIRE_PREFILL_CHUNK_ROWS={requested} is not admitted by every band and a band never shrinks the canonical chunk: {}",
            short.join("; ")
        ));
    }
    Ok(Qwen4PpChunkNegotiation {
        chunk_rows,
        requested: *requested,
        bands,
    })
}

/// Largest streams payload of one band edge at `chunk_rows` canonical rows:
/// `rows * hc_count * hidden * 4` (the F32 worst case; BF16 halves it), the
/// `max_bytes` for `Gpus::provision_mesh_edges`.
pub fn qwen4_pp_edge_max_bytes(config: &Qwen4Config, chunk_rows: usize) -> Result<usize, String> {
    if chunk_rows == 0 {
        return Err("qwen4: pipeline-parallel edge payload for zero rows".to_string());
    }
    chunk_rows
        .checked_mul(config.hc_count)
        .and_then(|bytes| bytes.checked_mul(config.hidden_size))
        .and_then(|bytes| bytes.checked_mul(std::mem::size_of::<f32>()))
        .filter(|bytes| bytes % 16 == 0)
        .ok_or_else(|| {
            format!("qwen4: pipeline-parallel edge payload for {chunk_rows} rows is not a 16-byte multiple or overflows")
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Qwen4Config;
    use serde_json::json;

    fn config() -> Qwen4Config {
        let layers: Vec<_> = (0..48)
            .map(|idx| {
                if idx % 4 == 3 {
                    "full_attention"
                } else {
                    "linear_attention"
                }
            })
            .collect();
        Qwen4Config::from_value(&json!({
            "model_type": "qwen4_exp",
            "text_config": {
                "model_type": "qwen4_exp_text", "dtype": "bfloat16",
                "hidden_size": 2560, "vocab_size": 248320, "num_hidden_layers": 48,
                "max_position_embeddings": 262144, "num_attention_heads": 24,
                "num_key_value_heads": 2, "head_dim": 256, "partial_rotary_factor": 0.25,
                "rope_parameters": {"rope_theta": 10000000.0}, "attention_bias": false,
                "layer_types": layers, "full_attention_interval": 4,
                "linear_num_key_heads": 16, "linear_num_value_heads": 48,
                "linear_key_head_dim": 128, "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4, "mamba_ssm_dtype": "float32",
                "indexer_n_heads": 4, "indexer_kv_heads": 1, "indexer_head_dim": 128,
                "indexer_budget": 2048, "indexer_compress_ratio": 4,
                "num_experts": 512, "num_experts_per_tok": 10,
                "moe_intermediate_size": 640, "shared_expert_intermediate_size": 640,
                "hc_count": 4, "hc_lowrank": 320, "ple_layer_ids": [2],
                "ple_conv_kernel_size": 4, "ple_embed_dim": 2560,
                "split_ngram_parts": 128, "heads_per_ngram": 8, "ngram_size": 3,
                "ngram_vocab_size_base": 20000000, "make_ngram_vocab_size_divisible_by": 128,
                "eos_token_id": 248044, "tie_word_embeddings": false,
                "mtp_num_hidden_layers": 1, "mtp_use_dedicated_embeddings": false,
                "output_gate_type": "sigmoid",
                "mtp": {"hybrid": true, "layer_types": ["full_attention"],
                        "num_hidden_layers": 1, "rope_theta": 10000000.0}
            }
        }))
        .unwrap()
    }

    #[test]
    fn native_mtp_is_admitted_when_requested() {
        let request = Qwen4AdmissionRequest::text_mtp(
            EffectiveMesh::single(),
            Qwen4Artifact::manifest_hfqm(),
        );
        let mtp = Qwen4Admission::admit(&config(), &request).expect("native MTP is admitted");
        assert!(mtp.capabilities.text);
        assert!(mtp.capabilities.native_mtp);
        assert!(mtp.native_mtp);

        let ar = Qwen4Admission::admit(
            &config(),
            &Qwen4AdmissionRequest::text_ar(
                EffectiveMesh::single(),
                Qwen4Artifact::manifest_hfqm(),
            ),
        )
        .expect("AR remains admitted");
        assert!(ar.capabilities.text);
        assert!(!ar.capabilities.native_mtp);
        assert!(!ar.native_mtp);
    }

    #[test]
    fn images_videos_meshes_and_formats_fail_closed() {
        let mut request =
            Qwen4AdmissionRequest::text_ar(EffectiveMesh::single(), Qwen4Artifact::manifest_hfqm());
        request.modality = InputModality::Image;
        assert!(matches!(
            Qwen4Admission::admit(&config(), &request),
            Err(AdmissionError::UnsupportedModality(InputModality::Image))
        ));
        request.modality = InputModality::Video;
        assert!(matches!(
            Qwen4Admission::admit(&config(), &request),
            Err(AdmissionError::UnsupportedModality(InputModality::Video))
        ));
        request.modality = InputModality::Text;
        request.effective_mesh = EffectiveMesh::new(1, 2, 1);
        assert!(matches!(
            Qwen4Admission::admit(&config(), &request),
            Err(AdmissionError::NonSingleMesh(_))
        ));
        request.effective_mesh = EffectiveMesh::single();
        request.artifact.format = Qwen4ArtifactFormat::Safetensors;
        assert!(matches!(
            Qwen4Admission::admit(&config(), &request),
            Err(AdmissionError::UnsupportedFormat(
                Qwen4ArtifactFormat::Safetensors
            ))
        ));
    }

    #[test]
    fn wrong_quant_geometry_is_refused() {
        let mut artifact = Qwen4Artifact::manifest_hfqm();
        artifact.quant_geometry.routed_down.k = 2560;
        let request = Qwen4AdmissionRequest::text_ar(EffectiveMesh::single(), artifact);
        assert!(matches!(
            Qwen4Admission::admit(&config(), &request),
            Err(AdmissionError::WrongGeometry {
                component: "routed down",
                ..
            })
        ));
    }
    #[test]
    fn manifest_recipe_has_exact_extents_and_native_payload_classes() {
        let geometry = Qwen4QuantGeometry::manifest();
        assert_eq!(
            geometry
                .matrix_aligned
                .payload_bytes(2)
                .expect("aligned matrix extent"),
            2 * 10 * 136
        );
        assert_eq!(
            geometry
                .matrix_unaligned
                .payload_bytes(2)
                .expect("row-local matrix extent"),
            2 * 5 * 68
        );
        assert_eq!(
            geometry.routed_gate_up.payload_bytes(ROUTED_GATE_UP_ROWS),
            Some(ROUTED_GATE_UP_BYTES)
        );
        assert_eq!(
            geometry.routed_down.payload_bytes(ROUTED_DOWN_ROWS),
            Some(ROUTED_DOWN_BYTES)
        );
        assert_eq!(geometry.ple_row_dtype, QuantFormat::BF16);
        assert_eq!(geometry.scalar_dtype, QuantFormat::BF16);
        assert_eq!(geometry.vector_dtype, QuantFormat::BF16);
        assert_eq!(geometry.conv_dtype, QuantFormat::BF16);
        assert_eq!(geometry.metadata_dtype, QuantFormat::I64);
        geometry.validate().expect("manifest recipe");
    }

    #[test]
    fn effective_mesh_is_derived_from_the_normalized_device_mesh() {
        use hipfire_runtime::device_mesh::{DeviceMesh, DimKind};
        let single = DeviceMesh::single().expect("single mesh");
        assert_eq!(
            EffectiveMesh::from_device_mesh(&single),
            EffectiveMesh::single()
        );
        let ep = DeviceMesh::rect(&[(DimKind::Ep, 6)]).expect("ep mesh");
        let derived = EffectiveMesh::from_device_mesh(&ep);
        assert_eq!(derived, EffectiveMesh::new(1, 1, 6));
        assert!(derived.is_qwen4_admitted());
        let pp = DeviceMesh::rect(&[(DimKind::Pp, 2), (DimKind::Ep, 2)]).expect("pp mesh");
        let derived = EffectiveMesh::from_device_mesh(&pp);
        assert_eq!(derived, EffectiveMesh::new(2, 1, 2));
        assert!(!derived.is_qwen4_admitted());
    }

    #[test]
    fn ep_mesh_matrix_admits_one_through_eight() {
        let admit = |mesh: EffectiveMesh, mtp: bool| {
            let artifact = Qwen4Artifact::manifest_hfqm();
            let request = if mtp {
                Qwen4AdmissionRequest::text_mtp(mesh, artifact)
            } else {
                Qwen4AdmissionRequest::text_ar(mesh, artifact)
            };
            Qwen4Admission::admit(&config(), &request)
        };
        for ep in 1..=8 {
            let mesh = EffectiveMesh::new(1, 1, ep);
            assert!(mesh.is_qwen4_admitted());
            assert_eq!(
                admit(mesh, false)
                    .expect("ep AR admitted")
                    .effective_mesh
                    .ep,
                ep
            );
        }
        assert!(EffectiveMesh::single().is_single());
        assert!(!EffectiveMesh::new(1, 1, 2).is_single());
        // 0 is malformed; above the 8-rank maximum is refused.
        for ep in [0, 9, 16] {
            assert!(matches!(
                admit(EffectiveMesh::new(1, 1, ep), false),
                Err(AdmissionError::NonSingleMesh(_))
            ));
        }
        // tp stays refused at every pp/ep, and pp never composes with ep.
        for (pp, tp, ep) in [
            (1, 2, 1),
            (2, 1, 2),
            (1, 2, 4),
            (4, 1, 4),
            (1, 2, 6),
            (2, 2, 1),
            (9, 1, 1),
            (0, 1, 1),
        ] {
            assert!(matches!(
                admit(EffectiveMesh::new(pp, tp, ep), false),
                Err(AdmissionError::NonSingleMesh(_))
            ));
        }
        // Pure pipeline parallelism is admitted at pp 2..=8 only, AR and
        // native MTP alike (the output band attaches the one head).
        for pp in 2..=8 {
            let mesh = EffectiveMesh::new(pp, 1, 1);
            assert!(mesh.is_qwen4_admitted() && mesh.is_pipeline() && !mesh.is_single());
            let admitted = admit(mesh, false).expect("PP AR admitted");
            assert_eq!(admitted.effective_mesh.pp, pp);
            assert!(!admitted.native_mtp);
            let admitted = admit(mesh, true).expect("PP native MTP admitted");
            assert_eq!(admitted.effective_mesh.pp, pp);
            assert!(admitted.native_mtp && admitted.capabilities.native_mtp);
        }
        assert!(!EffectiveMesh::single().is_pipeline());
        assert!(!EffectiveMesh::new(1, 1, 4).is_pipeline());
        // Native MTP is admitted on every admitted mesh: under EP the drafter
        // is replicated on every rank.
        assert!(admit(EffectiveMesh::single(), true).is_ok());
        for ep in [2, 3, 4, 6, 8] {
            let admitted = admit(EffectiveMesh::new(1, 1, ep), true).expect("EP native MTP");
            assert!(admitted.native_mtp);
            assert_eq!(admitted.effective_mesh.ep, ep);
        }
    }

    /// The stock-FP8 geometry: BF16 trunk, qt=55 experts at the contract
    /// stride, qt=56 PLE rows; a mix with the packed geometry is refused.
    #[test]
    fn stock_fp8_geometry_has_contract_extents() {
        let geometry = Qwen4QuantGeometry::stock_fp8();
        geometry.validate().expect("stock-FP8 recipe");
        assert!(geometry.is_stock_fp8() && !Qwen4QuantGeometry::manifest().is_stock_fp8());
        assert_eq!(
            Qwen4QuantGeometry::for_recipe(&Qwen4TierRecipe::StockFp8),
            geometry
        );
        // One block: S(R,K) = round_up(R*K + 2*ceil(R/128)*ceil(K/128), 256).
        assert_eq!(
            geometry.routed_gate_up.payload_bytes(1_280),
            Some(3_277_312)
        );
        assert_eq!(geometry.routed_down.payload_bytes(2_560), Some(1_638_656));
        assert_eq!(
            geometry.routed_gate_up.expert_payload_bytes(512, 1_280),
            Some(1_677_983_744)
        );
        assert_eq!(
            geometry.routed_down.expert_payload_bytes(512, 2_560),
            Some(838_991_872)
        );
        // Non-multiple-of-128 shapes pad their scale grid and the block.
        assert_eq!(
            QuantTensorGeometry::fp8e4m3b128(129).payload_bytes(3),
            Some((3usize * 129 + 2 * 2).div_ceil(256) * 256)
        );
        assert_eq!(
            geometry.matrix_aligned.payload_bytes(2),
            Some(2 * 2_560 * 2)
        );
        assert_eq!(
            geometry.matrix_unaligned.payload_bytes(2),
            Some(2 * 640 * 2)
        );
        assert_eq!(geometry.ple_row_dtype, QuantFormat::Fp8E4M3PerTensor);

        let mut mixed = Qwen4QuantGeometry::manifest();
        mixed.routed_down = QuantTensorGeometry::fp8e4m3b128(640);
        assert!(matches!(
            mixed.validate(),
            Err(QuantAdmissionError::WrongGeometry {
                component: "aligned matrix",
                ..
            })
        ));
        let mut short = Qwen4QuantGeometry::stock_fp8();
        short.routed_gate_up.block_size = 256;
        assert!(short.validate().is_err());
    }

    /// A stock-FP8 artifact is admitted on gfx942 only (fail closed on every
    /// other device and on an unknown one), AR-only, single device, and only
    /// under the AR-only profile.
    #[test]
    fn stock_fp8_artifact_admits_on_gfx942_only() {
        let ar_only = config().into_ar_only().expect("AR-only Flash-Next");
        let request = |arch: Option<&str>| {
            let request = Qwen4AdmissionRequest::text_ar(
                EffectiveMesh::single(),
                Qwen4Artifact::stock_fp8_hfqm(),
            );
            match arch {
                Some(arch) => request.on_device(arch),
                None => request,
            }
        };
        let admitted =
            Qwen4Admission::admit(&ar_only, &request(Some("gfx942"))).expect("gfx942 admits");
        assert!(admitted.capabilities.fp8_experts);
        assert!(!admitted.native_mtp && !admitted.capabilities.native_mtp);
        for arch in [
            Some("gfx1201"),
            Some("gfx1151"),
            Some("gfx1100"),
            Some("gfx940"),
            None,
        ] {
            let error = Qwen4Admission::admit(&ar_only, &request(arch)).unwrap_err();
            assert_eq!(
                error,
                AdmissionError::Fp8ExpertsUnsupported {
                    gpu_arch: arch.map(str::to_string)
                }
            );
            assert!(error.to_string().contains("gfx942"), "{error}");
        }
        // Native MTP: the artifact carries no head.
        let mut mtp = request(Some("gfx942"));
        mtp.native_mtp = true;
        assert_eq!(
            Qwen4Admission::admit(&ar_only, &mtp),
            Err(AdmissionError::ArOnlyArtifact)
        );
        // Expert parallelism is refused for P1.
        let mut ep = request(Some("gfx942"));
        ep.effective_mesh = EffectiveMesh::new(1, 1, 2);
        assert!(matches!(
            Qwen4Admission::admit(&ar_only, &ep),
            Err(AdmissionError::Fp8SingleDeviceOnly(_))
        ));
        // The checkpoint profile (which names an MTP head) is not the
        // stock-FP8 artifact's profile.
        assert!(matches!(
            Qwen4Admission::admit(&config(), &request(Some("gfx942"))),
            Err(AdmissionError::Fp8ProfileMismatch(_))
        ));
        // The packed artifact stays device-neutral at this boundary.
        let packed =
            Qwen4AdmissionRequest::text_ar(EffectiveMesh::single(), Qwen4Artifact::manifest_hfqm())
                .on_device("gfx1201");
        let packed = Qwen4Admission::admit(&config(), &packed).expect("packed on gfx1201");
        assert!(!packed.capabilities.fp8_experts);
    }

    /// The loader-side recipe check mirrors the pure admission.
    #[test]
    fn recipe_device_admission_gates_stock_fp8() {
        let single = EffectiveMesh::single();
        let stock = Qwen4TierRecipe::StockFp8;
        let caps = admit_recipe_on_device(&stock, "gfx942", single, false, false, false).unwrap();
        assert!(caps.fp8_experts && caps.supports(Qwen4Capability::Fp8Experts));
        for arch in ["gfx1201", "gfx1151"] {
            for head in [false, true] {
                assert!(matches!(
                    admit_recipe_on_device(&stock, arch, single, false, false, head),
                    Err(AdmissionError::Fp8ExpertsUnsupported { .. })
                ));
                // Other archs refuse on the device before the MTP question (U6).
                assert!(matches!(
                    admit_recipe_on_device(&stock, arch, single, true, false, head),
                    Err(AdmissionError::Fp8ExpertsUnsupported { .. })
                ));
            }
        }
        // gfx942 without the head sidecar: R0.
        let r0 = admit_recipe_on_device(&stock, "gfx942", single, true, false, false);
        assert_eq!(r0, Err(AdmissionError::ArOnlyArtifact));
        assert_eq!(
            r0.unwrap_err().to_string(),
            "qwen4 admission: the stock-FP8 artifact was admitted without its MTP head sidecar \
             (<model>.mtp); native MTP (--spec mtp) cannot be admitted"
        );
        // gfx942 with the head attached: admitted with native MTP + FP8 experts.
        let caps = admit_recipe_on_device(&stock, "gfx942", single, true, false, true).unwrap();
        assert!(caps.native_mtp && caps.fp8_experts);
        // The single-device and resident-expert checks still run for native MTP.
        let mesh2 = EffectiveMesh::new(1, 1, 2);
        assert_eq!(
            admit_recipe_on_device(&stock, "gfx942", mesh2, true, false, true),
            Err(AdmissionError::Fp8SingleDeviceOnly(mesh2))
        );
        assert_eq!(
            admit_recipe_on_device(&stock, "gfx942", single, true, true, true),
            Err(AdmissionError::Fp8HostMappedExperts)
        );
        // Head attached without native MTP requested is device-admitted.
        assert!(admit_recipe_on_device(&stock, "gfx942", single, false, false, true).is_ok());
        assert_eq!(
            admit_recipe_on_device(&stock, "gfx942", single, false, true, false),
            Err(AdmissionError::Fp8HostMappedExperts)
        );
        assert!(matches!(
            admit_recipe_on_device(
                &stock,
                "gfx942",
                EffectiveMesh::new(1, 1, 4),
                false,
                false,
                false
            ),
            Err(AdmissionError::Fp8SingleDeviceOnly(_))
        ));
        let packed = Qwen4TierRecipe::FlashNextPacked {
            trunk_tier: rdna_compute::DType::Q8_0,
            mtp_source: false,
        };
        let caps = admit_recipe_on_device(&packed, "gfx1201", single, true, true, false).unwrap();
        assert!(!caps.fp8_experts && caps.native_mtp);
    }

    mod pp_budget {
        use super::*;
        use crate::expert_residency::NativeMtpCensusInputs;
        use crate::gpu_forward::Qwen4ForwardScratchMode::Full;
        use hipfire_runtime::device_mesh::{DeviceMesh, DimKind};
        use hipfire_runtime::mesh_placement::BandPlacement;
        use hipfire_runtime::weight_manifest::WeightEntry;
        use rdna_compute::tensor_ops::QsaKvFormat;

        fn bytes_of(entry: &WeightEntry) -> Option<u64> {
            Some(entry.logical_shape.iter().product::<usize>() as u64 * 2)
        }

        fn placement(counts: &[usize]) -> BandPlacement {
            let mesh = DeviceMesh::rect(&[(DimKind::Pp, counts.len())]).unwrap();
            BandPlacement::from_counts(&mesh, 48, counts).unwrap()
        }

        fn budget(
            config: &Qwen4Config,
            weights: &[WeightEntry],
            placement: &BandPlacement,
            free: &[u64],
        ) -> Result<Vec<Qwen4PpRankFit>, String> {
            budget_mtp(config, weights, placement, free, None)
        }

        fn budget_mtp(
            config: &Qwen4Config,
            weights: &[WeightEntry],
            placement: &BandPlacement,
            free: &[u64],
            native_mtp: Option<NativeMtpCensusInputs>,
        ) -> Result<Vec<Qwen4PpRankFit>, String> {
            let context = crate::Qwen4ContextCommit::legacy(1024);
            qwen4_pp_band_budget(&Qwen4PpBudgetInputs {
                config,
                placement,
                weights,
                bytes_of: &bytes_of,
                format: crate::Qwen4StateFormat {
                    qsa: QsaKvFormat::Fp8,
                    gdn: crate::GdnStateFormat::Q8,
                    qsa_f16_arena: false,
                    qsa_pooled_bf16: false,
                },
                context: &context,
                chunk_rows: 256,
                mode: Full,
                free_per_rank: free,
                reserve_bytes: QWEN4_PP_RESERVE_BYTES,
                native_mtp,
            })
        }

        #[test]
        fn every_rank_must_fit_its_own_census() {
            let config = crate::config::compact_test_config();
            let manifest = crate::weights::Qwen4Manifest::build(&config).unwrap();
            let placement = placement(&[9, 8, 8, 8, 8, 7]);
            let fits = budget(&config, &manifest.weights, &placement, &[u64::MAX; 6]).unwrap();
            assert_eq!(fits.len(), 6);
            assert_eq!(fits[0].band, 0..9);
            assert_eq!(fits[5].band, 41..48);
            assert!(fits.iter().all(|fit| fit.required > QWEN4_PP_RESERVE_BYTES));
            // Exactly enough fits; one byte short on one rank refuses, naming
            // topology.pp and that rank only.
            let exact: Vec<u64> = fits.iter().map(|fit| fit.required).collect();
            assert!(budget(&config, &manifest.weights, &placement, &exact).is_ok());
            let mut short = exact.clone();
            short[3] -= 1;
            let error = budget(&config, &manifest.weights, &placement, &short).unwrap_err();
            assert!(
                error.contains("topology.pp = 6")
                    && error.contains("rank 3 (layers 25..33)")
                    && error.contains("short by 1")
                    && !error.contains("rank 2")
                    && !error.contains("rank 4"),
                "{error}"
            );
            // The reading count must match the rank count.
            let error = budget(&config, &manifest.weights, &placement, &exact[..5]).unwrap_err();
            assert!(error.contains("topology.pp = 6"), "{error}");
        }

        #[test]
        fn the_output_rank_counts_the_mtp_heads_weights_even_for_ar() {
            // The output band loads its `mtp.*` entries whether or not the
            // head attaches, so an AR budget counts them on that rank only.
            let config = crate::config::compact_test_config();
            let manifest = crate::weights::Qwen4Manifest::build(&config).unwrap();
            assert!(manifest.weights.iter().any(|e| e.name.starts_with("mtp.")));
            let without: Vec<WeightEntry> = manifest
                .weights
                .iter()
                .filter(|e| !e.name.starts_with("mtp."))
                .cloned()
                .collect();
            let placement = placement(&[24, 24]);
            let with = budget(&config, &manifest.weights, &placement, &[u64::MAX; 2]).unwrap();
            let without = budget(&config, &without, &placement, &[u64::MAX; 2]).unwrap();
            assert_eq!(with[0], without[0]);
            assert!(with[1].required > without[1].required);
            assert!(with.iter().all(|fit| fit.native_mtp == 0));
        }

        #[test]
        fn native_mtp_charges_the_head_on_the_output_rank_and_capture_on_every_rank() {
            let config = crate::config::compact_test_config();
            let manifest = crate::weights::Qwen4Manifest::build(&config).unwrap();
            let placement = placement(&[9, 8, 8, 8, 8, 7]);
            let gdn = crate::GdnStateFormat::Q8;
            let mtp = NativeMtpCensusInputs {
                max_k: 3,
                row_capture: Some((gdn, 4)),
                batched_fill: false,
            };
            let free = [u64::MAX; 6];
            let ar = budget(&config, &manifest.weights, &placement, &free).unwrap();
            let on = budget_mtp(&config, &manifest.weights, &placement, &free, Some(mtp)).unwrap();
            let capture = |band: &std::ops::Range<usize>| {
                let gdn_layers = band
                    .clone()
                    .filter(|&layer| {
                        config.layer_types[layer] == crate::config::LayerType::LinearAttention
                    })
                    .count();
                crate::state::Qwen4State::row_capture_bytes_scoped(&config, gdn, 4, gdn_layers)
                    .unwrap() as u64
            };
            for (stage, (ar, on)) in ar.iter().zip(&on).enumerate() {
                // The whole MTP term, and nothing else, is the difference.
                assert_eq!(on.required - ar.required, on.native_mtp, "rank {stage}");
                assert_eq!(ar.native_mtp, 0);
                if stage + 1 < 6 {
                    // A rank without the head pays its own capture alone.
                    assert_eq!(on.native_mtp, capture(&on.band), "rank {stage}");
                } else {
                    assert!(on.native_mtp > capture(&on.band));
                }
            }
            // A byte short on the output rank alone refuses, naming the term.
            let mut exact: Vec<u64> = on.iter().map(|fit| fit.required).collect();
            assert!(budget_mtp(&config, &manifest.weights, &placement, &exact, Some(mtp)).is_ok());
            exact[5] -= 1;
            let error =
                budget_mtp(&config, &manifest.weights, &placement, &exact, Some(mtp)).unwrap_err();
            assert!(
                error.contains("rank 5 (layers 41..48)")
                    && error.contains("native MTP")
                    && error.contains("short by 1")
                    && !error.contains("rank 4"),
                "{error}"
            );
            // The same free memory does not fit once MTP is on where the AR
            // budget fit it exactly.
            let ar_exact: Vec<u64> = ar.iter().map(|fit| fit.required).collect();
            assert!(budget(&config, &manifest.weights, &placement, &ar_exact).is_ok());
            assert!(
                budget_mtp(&config, &manifest.weights, &placement, &ar_exact, Some(mtp)).is_err()
            );
        }

        #[test]
        fn a_band_without_the_ple_layer_is_refused_before_any_census() {
            let config = crate::config::compact_test_config();
            let manifest = crate::weights::Qwen4Manifest::build(&config).unwrap();
            // `with_first_band_min(2)` already refuses this split; the budget
            // refuses it independently.
            assert!(placement(&[1, 47]).with_first_band_min(2).is_err());
            let error = budget(
                &config,
                &manifest.weights,
                &placement(&[1, 47]),
                &[u64::MAX; 2],
            )
            .unwrap_err();
            assert!(
                error.contains("topology.pp = 2") && error.contains("PLE"),
                "{error}"
            );
        }

        #[test]
        fn edge_payload_matches_the_frozen_chunk_sizes() {
            let config = crate::config::compact_test_config();
            assert_eq!(qwen4_pp_edge_max_bytes(&config, 1), Ok(40_960));
            assert_eq!(qwen4_pp_edge_max_bytes(&config, 1536), Ok(62_914_560));
            assert_eq!(qwen4_pp_edge_max_bytes(&config, 4096), Ok(167_772_160));
            assert!(qwen4_pp_edge_max_bytes(&config, 0).is_err());
            assert!(qwen4_pp_edge_max_bytes(&config, usize::MAX).is_err());
        }
    }

    /// The mesh-wide canonical prefill chunk ([`qwen4_pp_negotiate_chunk`]) on
    /// the field-reported box: Flash-Next PP5 on 24 GB 7900 XTX cards, pure CPU.
    ///
    /// The reported load (`dg-pp6` receipt, build f648d3a9a8): six cards of
    /// 24,560 MiB, PP5 on the first five, the host link measured narrow
    /// (7.1 GB/s) so the requested chunk is 8192, bands 0..10, 10..19,
    /// 19..29, 29..39, 39..48 (the band loads logged 19..29 and 29..39, the
    /// refusal named 39..48). Bands 0 to 3 loaded at the 8192-row chunk; the
    /// output band (39..48) picked 4096 in its own rung search and the load
    /// died on `band 39..48 allocated chunk rows Some(4096) but the canonical
    /// chunk is 8192`.
    ///
    /// Root cause pinned here: the PP census charges every band its own
    /// forward scratch (`forward_census_scope`: the PLE input buffers only on
    /// the input band, the head only on the output band) while the band's own
    /// rung search charged the WHOLE-model forward
    /// (`qwen4_forward_device_bytes`), 2 GiB more on a middle band at 8192
    /// rows, so the tightest band (the output band: head and MTP weights)
    /// failed the search the census had admitted. The receipt records the
    /// cards' total (24,560 MiB) and per-card peaks, not each rank's free
    /// VRAM, so the per-rank capacity is emulated from the census itself (see
    /// `Fixture::dg_capacity`).
    mod pp_chunk_negotiation {
        use super::*;
        use crate::gpu_forward::Qwen4ForwardScratchMode::Full;
        use crate::gpu_forward::QWEN4_FORWARD_HEADROOM_BYTES;
        use hipfire_runtime::device_mesh::{DeviceMesh, DimKind};
        use hipfire_runtime::mesh_placement::BandPlacement;
        use hipfire_runtime::weight_manifest::WeightEntry;
        use rdna_compute::tensor_ops::QsaKvFormat;

        const MIB: u64 = 1 << 20;
        /// The reported PP5 split (band 4 is the output band).
        const COUNTS: [usize; 5] = [10, 9, 10, 10, 9];
        const REQUESTED: usize = 8192;

        fn bytes_of(entry: &WeightEntry) -> Option<u64> {
            Some(entry.logical_shape.iter().product::<usize>() as u64 * 2)
        }

        fn lines(
            config: &Qwen4Config,
            weights: &[WeightEntry],
            rows: usize,
        ) -> Vec<Qwen4PpRankFit> {
            let mesh = DeviceMesh::rect(&[(DimKind::Pp, COUNTS.len())]).unwrap();
            let placement = BandPlacement::from_counts(&mesh, 48, &COUNTS).unwrap();
            let context = crate::Qwen4ContextCommit::legacy(8192);
            qwen4_pp_rank_lines(&Qwen4PpBudgetInputs {
                config,
                placement: &placement,
                weights,
                bytes_of: &bytes_of,
                format: crate::Qwen4StateFormat {
                    qsa: QsaKvFormat::Fp8,
                    gdn: crate::GdnStateFormat::Q8,
                    qsa_f16_arena: false,
                    qsa_pooled_bf16: false,
                },
                context: &context,
                chunk_rows: rows,
                mode: Full,
                free_per_rank: &[u64::MAX; 5],
                reserve_bytes: QWEN4_PP_RESERVE_BYTES,
                native_mtp: None,
            })
            .unwrap()
        }

        struct Fixture {
            config: Qwen4Config,
            /// The census of every ladder rung, largest first.
            tables: Vec<(usize, Vec<Qwen4PpRankFit>)>,
            scopes: Vec<crate::weights::Qwen4BandScope>,
        }

        impl Fixture {
            fn at(&self, rows: usize) -> &[Qwen4PpRankFit] {
                &self.tables.iter().find(|(r, _)| *r == rows).unwrap().1
            }

            /// Bytes a band holds besides its chunk-sized forward and the
            /// reserve: weights, F16 shadows, QSA context, request state.
            fn resident(&self, stage: usize) -> u64 {
                let line = &self.at(REQUESTED)[stage];
                line.required - line.forward - line.reserve
            }

            /// Planned capacity left after the band's resident bytes.
            fn free_after(&self, capacity: &[u64], stage: usize) -> u64 {
                capacity[stage] - self.resident(stage)
            }

            /// The field-reported box, emulated: bands 0 to 3 have the room the
            /// OLD per-band rung search (whole-model forward plus 1 GiB
            /// headroom) needed to take 8192 rows, and at least their census;
            /// the output band has
            /// exactly its census at 8192 rows (forward scratch of its own
            /// plus the 2 GiB reserve), which the census admits and the old
            /// search did not.
            fn dg_capacity(&self) -> Vec<u64> {
                let whole =
                    crate::gpu_forward::qwen4_forward_device_bytes(&self.config, 8192, Full)
                        .unwrap()
                        + QWEN4_FORWARD_HEADROOM_BYTES;
                (0..COUNTS.len())
                    .map(|stage| {
                        if stage < 4 {
                            (self.resident(stage) + whole).max(self.at(8192)[stage].required)
                        } else {
                            self.at(8192)[stage].required
                        }
                    })
                    .collect()
            }

            /// A box whose output card genuinely cannot hold 8192 rows: it
            /// has exactly its 4096-row census.
            fn tight_output_capacity(&self) -> Vec<u64> {
                let mut capacity = self.dg_capacity();
                capacity[4] = self.at(4096)[4].required;
                assert!(self.at(8192)[4].required > capacity[4] + 512 * MIB);
                capacity
            }

            /// Every band's own attach at `canonical` rows (the pure rung
            /// choice `Qwen4Bundle::attach_forward_canonical` makes), against
            /// what the band has left after its resident bytes.
            fn attach_all(&self, capacity: &[u64], canonical: usize) -> Vec<Result<usize, String>> {
                (0..COUNTS.len())
                    .map(|stage| {
                        crate::bundle::attach_chunk_rows(
                            &self.config,
                            canonical,
                            Full,
                            self.free_after(capacity, stage),
                            Some(&self.scopes[stage]),
                        )
                    })
                    .collect()
            }
        }

        fn fixture() -> Fixture {
            let config = crate::config::compact_test_config();
            let manifest = crate::weights::Qwen4Manifest::build(&config).unwrap();
            let tables: Vec<_> = qwen4_pp_chunk_ladder(REQUESTED, false)
                .into_iter()
                .map(|rows| (rows, lines(&config, &manifest.weights, rows)))
                .collect();
            let mesh = DeviceMesh::rect(&[(DimKind::Pp, COUNTS.len())]).unwrap();
            let placement = BandPlacement::from_counts(&mesh, 48, &COUNTS).unwrap();
            let scopes = (0..COUNTS.len())
                .map(|stage| {
                    crate::weights::Qwen4BandScope::from_placement(&config, &placement, stage)
                        .unwrap()
                })
                .collect();
            Fixture {
                config,
                tables,
                scopes,
            }
        }

        #[test]
        fn the_ladder_is_the_ceiling_then_every_smaller_rung_and_a_pinned_chunk_is_alone() {
            assert_eq!(
                qwen4_pp_chunk_ladder(8192, false),
                [8192, 4096, 2048, 1536, 1024, 512, 256]
            );
            assert_eq!(qwen4_pp_chunk_ladder(4096, false)[0], 4096);
            assert_eq!(qwen4_pp_chunk_ladder(1024, false), [1024, 512, 256]);
            assert_eq!(qwen4_pp_chunk_ladder(3072, false)[..3], [3072, 2048, 1536]);
            assert_eq!(qwen4_pp_chunk_ladder(8192, true), [8192]);
        }

        /// The reported load. Each band's own (pre-fix) rung search, run on
        /// the emulated DG capacities, picks 8192 on bands 0 to 3 and 4096 on
        /// the output band: the disagreement the old loader turned into
        /// `allocated chunk rows Some(4096) but the canonical chunk is 8192`.
        /// The census charged the output band's forward 2 GiB less than that
        /// search did.
        #[test]
        fn dg_pp5_old_per_band_rung_search_disagreed_on_the_chunk() {
            let fx = fixture();
            let capacity = fx.dg_capacity();
            let whole =
                crate::gpu_forward::qwen4_forward_device_bytes(&fx.config, 8192, Full).unwrap();
            // The middle bands' census forward is 2 GiB under the whole model's.
            assert!(whole - fx.at(8192)[2].forward >= 2000 * MIB);
            // The census admits 8192 on every band.
            assert!(fx
                .at(8192)
                .iter()
                .zip(&capacity)
                .all(|(line, capacity)| line.required <= *capacity));
            let independent: Vec<usize> = (0..COUNTS.len())
                .map(|stage| {
                    crate::bundle::attach_chunk_rows(
                        &fx.config,
                        8192,
                        Full,
                        fx.free_after(&capacity, stage),
                        None,
                    )
                    .unwrap()
                })
                .collect();
            assert_eq!(independent, [8192, 8192, 8192, 8192, 4096]);
        }

        /// The fix on that box: one mesh chunk, decided before any band
        /// loads, and every band attaches exactly it. The band's forward is
        /// charged as the census charges it, so the output band keeps 8192.
        #[test]
        fn dg_pp5_every_band_loads_the_identical_canonical_chunk() {
            let fx = fixture();
            let capacity = fx.dg_capacity();
            let negotiation = qwen4_pp_negotiate_chunk(&fx.tables, &capacity, false).unwrap();
            assert_eq!(negotiation.requested, 8192);
            assert_eq!(negotiation.chunk_rows, 8192);
            assert!(negotiation
                .log_line()
                .starts_with("qwen4 PP canonical chunk: 8192 rows (every band admits it;"));
            let chunks: Vec<usize> = fx
                .attach_all(&capacity, negotiation.chunk_rows)
                .into_iter()
                .enumerate()
                .map(|(stage, chunk)| chunk.unwrap_or_else(|error| panic!("band {stage}: {error}")))
                .collect();
            assert_eq!(chunks, [8192; 5]);
        }

        /// A box whose output card truly cannot hold 8192 rows: the whole mesh
        /// settles on 4096 before any band loads, and the output band's own
        /// attach agrees (no band disagrees with the mesh).
        #[test]
        fn a_tight_output_band_lowers_the_mesh_chunk_for_every_band() {
            let fx = fixture();
            let capacity = fx.tight_output_capacity();
            let negotiation = qwen4_pp_negotiate_chunk(&fx.tables, &capacity, false).unwrap();
            assert_eq!(negotiation.chunk_rows, 4096);
            let per_band: Vec<Option<usize>> =
                negotiation.bands.iter().map(|band| band.max_rows).collect();
            assert_eq!(
                per_band,
                [Some(8192), Some(8192), Some(8192), Some(8192), Some(4096)]
            );
            assert_eq!(negotiation.bands[4].band, 39..48);
            assert_eq!(
                negotiation.log_line(),
                "qwen4 PP canonical chunk: 4096 rows, lowered from 8192 for the whole mesh by band 4 \
                 (per-band largest admissible rung: band 0 (layers 0..10) 8192, band 1 (layers 10..19) 8192, \
                 band 2 (layers 19..29) 8192, band 3 (layers 29..39) 8192, band 4 (layers 39..48) 4096)"
            );
            let chunks: Vec<usize> = fx
                .attach_all(&capacity, negotiation.chunk_rows)
                .into_iter()
                .enumerate()
                .map(|(stage, chunk)| chunk.unwrap_or_else(|error| panic!("band {stage}: {error}")))
                .collect();
            assert_eq!(chunks, [4096; 5]);
        }

        /// A band attach never shrinks the canonical chunk: the output band of
        /// the tight box asked for 8192 refuses with the bytes.
        #[test]
        fn a_band_attach_never_shrinks_the_canonical_chunk() {
            let fx = fixture();
            let capacity = fx.tight_output_capacity();
            let attached = fx.attach_all(&capacity, 8192);
            assert!(attached[..4].iter().all(|chunk| chunk == &Ok(8192)));
            let error = attached[4].clone().unwrap_err();
            assert!(
                error.contains("canonical prefill chunk of 8192 rows")
                    && error.contains("never shrinks the canonical chunk"),
                "{error}"
            );
        }

        #[test]
        fn an_explicit_chunk_a_band_cannot_admit_is_refused_naming_the_band() {
            let fx = fixture();
            let capacity = fx.tight_output_capacity();
            let pinned = vec![fx.tables[0].clone()];
            let error = qwen4_pp_negotiate_chunk(&pinned, &capacity, true).unwrap_err();
            assert!(
                error.contains("HIPFIRE_PREFILL_CHUNK_ROWS=8192")
                    && error.contains("band 4 (layers 39..48)")
                    && error.contains("never shrinks the canonical chunk")
                    && !error.contains("band 0 ")
                    && !error.contains("band 3 "),
                "{error}"
            );
            // The same pin at a chunk every band admits is taken as given.
            let at_4096 = vec![fx.tables[1].clone()];
            let kept = qwen4_pp_negotiate_chunk(&at_4096, &capacity, true).unwrap();
            assert_eq!((kept.chunk_rows, kept.requested), (4096, 4096));
        }

        #[test]
        fn the_chunk_steps_down_the_ladder_until_every_band_fits_or_refuses_naming_the_bands() {
            let fx = fixture();
            // Band 2 only holds 1536: it sets the mesh chunk, whatever the rest hold.
            let mut capacity = fx.tight_output_capacity();
            capacity[2] = fx.at(1536)[2].required;
            let negotiation = qwen4_pp_negotiate_chunk(&fx.tables, &capacity, false).unwrap();
            assert_eq!(negotiation.chunk_rows, 1536);
            assert_eq!(negotiation.bands[2].max_rows, Some(1536));
            assert_eq!(negotiation.bands[4].max_rows, Some(4096));
            // Nothing fits band 1 even at 256 rows: refused, naming that band only.
            capacity[1] = fx.at(256)[1].required - 1;
            let error = qwen4_pp_negotiate_chunk(&fx.tables, &capacity, false).unwrap_err();
            assert!(
                error.contains("no prefill chunk fits every band")
                    && error.contains("band 1 (layers 10..19)")
                    && !error.contains("band 2 (layers"),
                "{error}"
            );
            // Mismatched readings are malformed input, not a fit.
            assert!(qwen4_pp_negotiate_chunk(&fx.tables, &capacity[..4], false).is_err());
            assert!(qwen4_pp_negotiate_chunk(&[], &capacity, false).is_err());
        }
    }

    /// The PP split-and-chunk auto-fit ([`qwen4_pp_auto_fit`]), pure CPU.
    mod pp_auto_fit {
        use super::*;
        use crate::gpu_forward::Qwen4ForwardScratchMode::Full;
        use hipfire_runtime::device_mesh::{DeviceMesh, DimKind};
        use hipfire_runtime::mesh_placement::BandPlacement;
        use hipfire_runtime::weight_manifest::WeightEntry;
        use rdna_compute::tensor_ops::QsaKvFormat;
        use std::cell::RefCell;

        const MIB: u64 = 1 << 20;

        // The measured 2x R9700 Flash-Next mq3-xts (59.25 GB) load, 48 layers,
        // free per card 32,550 MiB, max_seq 16,384:
        // - uniform 24,24 at the gfx1201 PP chunk 4096: rank 1 needs
        //   35.97 GB (30.87 weights incl. the output extras + 0.62 F16 + 0.16
        //   QSA + 0.03 state + 1.18 forward + 0.93 native MTP + 2.18
        //   reserve), 1.84 GB short;
        // - hand-tuned 25,23 at chunk 1024: rank 0 32,476 MiB, rank 1
        //   32,377 MiB (incl. 755 MiB MTP).
        // The forward scratch scales with the rows as `qwen4_forward_device_bytes`
        // does (1.18 GB at 4096 rows), the MTP term with 755 MiB at 1024 rows
        // against 0.93 GB at 4096, and one band layer costs what the two
        // measured rank-1 points imply.
        const FREE: u64 = 32_550 * MIB;
        const R1_UNIFORM_4096: u64 = 35_970_000_000;
        const R0_HAND_1024: u64 = 32_476 * MIB;
        const R1_HAND_1024: u64 = 32_377 * MIB;
        const FWD_PER_ROW: u64 = 1_180_000_000 / 4096;
        const MTP_PER_ROW: u64 = (930_000_000 - 755 * MIB) / 3072;
        /// One band layer: rank 1 goes from 24 layers at 4096 rows to 23 at 1024.
        const LAYER: u64 = R1_UNIFORM_4096 - 3072 * (FWD_PER_ROW + MTP_PER_ROW) - R1_HAND_1024;
        /// Rank 0 with 24 layers at 4096 rows, from the hand-tuned 25 at 1024.
        const R0_UNIFORM_4096: u64 = R0_HAND_1024 - LAYER + 3072 * FWD_PER_ROW;
        /// The extra per-rank context bytes (QSA context and gather scratch)
        /// of max_seq 66,048 over 16,384. INFERENCE: the measurements say a
        /// 64K context needed chunk 256 (25,23 fits at 256 but not at 512), which
        /// holds exactly for an extra in (224.8 MB, 298.6 MB]; this is its middle.
        const CONTEXT_66K: u64 = 262_000_000;

        type Calls = RefCell<Vec<(Vec<usize>, usize)>>;

        fn line(
            stage: usize,
            band: std::ops::Range<usize>,
            required: u64,
            free: u64,
        ) -> Qwen4PpRankFit {
            Qwen4PpRankFit {
                stage,
                band,
                required,
                native_mtp: 0,
                free,
                weight_bytes: required,
                f16_shadow_bytes: 0,
                qsa_context: 0,
                state: 0,
                forward: 0,
                reserve: 0,
            }
        }

        /// The two ranks' lines of the measured fixture.
        fn measured_free(
            counts: &[usize],
            chunk: usize,
            context: u64,
            free: u64,
        ) -> Vec<Qwen4PpRankFit> {
            let cut = (4096 - chunk) as u64;
            let rank0 = R0_UNIFORM_4096 + counts[0] as u64 * LAYER + context
                - 24 * LAYER
                - cut * FWD_PER_ROW;
            let rank1 = R1_UNIFORM_4096 + counts[1] as u64 * LAYER + context
                - 24 * LAYER
                - cut * (FWD_PER_ROW + MTP_PER_ROW);
            vec![
                line(0, 0..counts[0], rank0, free),
                line(1, counts[0]..counts[0] + counts[1], rank1, free),
            ]
        }

        fn measured(counts: &[usize], chunk: usize, context: u64) -> Vec<Qwen4PpRankFit> {
            measured_free(counts, chunk, context, FREE)
        }

        #[allow(clippy::too_many_arguments)]
        fn search(
            admitted: &[usize],
            explicit: bool,
            first_min: usize,
            requested: usize,
            pinned: bool,
            context: u64,
            free: u64,
            calls: &Calls,
        ) -> Result<Qwen4PpAutoFit, String> {
            qwen4_pp_auto_fit(
                admitted,
                explicit,
                first_min,
                requested,
                pinned,
                |counts, chunk| {
                    calls.borrow_mut().push((counts.to_vec(), chunk));
                    Ok(measured_free(counts, chunk, context, free))
                },
            )
        }

        fn fits_at(counts: &[usize], chunk: usize, context: u64) -> bool {
            measured(counts, chunk, context)
                .iter()
                .all(Qwen4PpRankFit::fits)
        }

        #[test]
        fn the_fixture_reproduces_the_measured_loads() {
            // Uniform 24,24 at 4096: rank 1 needs 35.97 GB, 1.84 GB short; rank 0 fits.
            let uniform = measured(&[24, 24], 4096, 0);
            assert_eq!(uniform[1].required, 35_970_000_000);
            assert_eq!(uniform[1].shortfall(), 35_970_000_000 - FREE);
            // "short by 1.84 GB" in the measured refusal is rounded: 1,838,851,200 B.
            assert!((1_835_000_000..1_845_000_000).contains(&uniform[1].shortfall()));
            assert!(uniform[0].fits());
            // The hand-tuned 25,23 at 1024 is the measured MiB, and fits.
            let hand = measured(&[25, 23], 1024, 0);
            assert_eq!(
                [hand[0].required, hand[1].required],
                [R0_HAND_1024, R1_HAND_1024]
            );
            assert!(hand.iter().all(Qwen4PpRankFit::fits));
            assert!(!fits_at(&[24, 24], 1024, 0));
            // At 16K no split fits at 4096 or 2048 and only 25,23 fits at 1024.
            for chunk in [4096, 2048] {
                assert!((2..48).all(|first| !fits_at(&[first, 48 - first], chunk, 0)));
            }
            let at_1024: Vec<usize> = (2..48)
                .filter(|&first| fits_at(&[first, 48 - first], 1024, 0))
                .collect();
            assert_eq!(at_1024, [25]);
        }

        #[test]
        fn uniform_is_kept_when_it_fits() {
            let calls = Calls::default();
            let fit = search(&[24, 24], false, 2, 4096, false, 0, 36_000_000_000, &calls).unwrap();
            assert_eq!(fit.counts, [24, 24]);
            assert_eq!(fit.chunk_rows, 4096);
            assert!(fit.moved_from.is_none() && fit.log_line().is_none());
            assert!(fit.fits.iter().all(Qwen4PpRankFit::fits));
            // One census of the default, no search at all.
            assert_eq!(*calls.borrow(), [(vec![24, 24], 4096)]);
        }

        #[test]
        fn a_16k_default_load_is_rejected_at_24_24_4096_and_fit_at_25_23_1024() {
            let calls = Calls::default();
            let fit = search(&[24, 24], false, 2, 4096, false, 0, FREE, &calls).unwrap();
            assert_eq!(fit.counts, [25, 23]);
            assert_eq!(fit.chunk_rows, 1024);
            assert_eq!(
                [fit.fits[0].required, fit.fits[1].required],
                [R0_HAND_1024, R1_HAND_1024]
            );
            assert_eq!(
                fit.moved_from,
                Some(Qwen4PpDefaultShortfall {
                    counts: vec![24, 24],
                    chunk_rows: 4096,
                    rank: 1,
                    bytes: 35_970_000_000 - FREE,
                })
            );
            assert_eq!(
                fit.log_line().as_deref(),
                Some("qwen4 PP auto-fit: layers 25,23 chunk 1024 (uniform 24,24 at chunk 4096 short by 1754 MiB on rank 1)")
            );
            // Every probed split honors the PLE minimum and the layer count,
            // and no rung below the chosen chunk was tried.
            assert!(calls.borrow().iter().all(|(counts, chunk)| {
                counts[0] >= 2 && counts.iter().sum::<usize>() == 48 && *chunk >= 1024
            }));
            // Deterministic.
            let again = search(&[24, 24], false, 2, 4096, false, 0, FREE, &Calls::default());
            assert_eq!(again, Ok(fit));
        }

        #[test]
        fn a_66k_default_load_steps_down_to_25_23_at_chunk_256() {
            // INFERENCE: the context extra is the fixture's CONTEXT_66K.
            assert!((2..48).all(|first| !fits_at(&[first, 48 - first], 512, CONTEXT_66K)));
            let calls = Calls::default();
            let fit = search(&[24, 24], false, 2, 4096, false, CONTEXT_66K, FREE, &calls).unwrap();
            assert_eq!(fit.counts, [25, 23]);
            assert_eq!(fit.chunk_rows, 256);
            assert!(fit.fits.iter().all(Qwen4PpRankFit::fits));
            assert_eq!(
                fit.log_line().as_deref(),
                Some("qwen4 PP auto-fit: layers 25,23 chunk 256 (uniform 24,24 at chunk 4096 short by 2004 MiB on rank 1)")
            );
        }

        #[test]
        fn the_first_band_minimum_bounds_every_probe() {
            let calls = Calls::default();
            let fit = search(&[24, 24], false, 4, 4096, false, 0, FREE, &calls).unwrap();
            assert_eq!(fit.counts, [25, 23]);
            assert!(calls.borrow().iter().all(|(counts, _)| counts[0] >= 4));
        }

        #[test]
        fn a_pinned_chunk_searches_only_the_split() {
            let calls = Calls::default();
            let fit = search(&[24, 24], false, 2, 1024, true, 0, FREE, &calls).unwrap();
            assert_eq!(
                (fit.counts.as_slice(), fit.chunk_rows),
                (&[25, 23][..], 1024)
            );
            assert!(calls.borrow().iter().all(|(_, chunk)| *chunk == 1024));
            // A pinned chunk no split fits is the refusal, naming only that chunk.
            let calls = Calls::default();
            let error = search(&[24, 24], false, 2, 4096, true, 0, FREE, &calls).unwrap_err();
            assert!(calls.borrow().iter().all(|(_, chunk)| *chunk == 4096));
            assert!(
                error.contains("auto-fit searched chunk rows 4096 and"),
                "{error}"
            );
        }

        #[test]
        fn explicit_layers_are_never_searched() {
            // The operator's 24,24 at the requested chunk: today's refusal, one census.
            let calls = Calls::default();
            let error = search(&[24, 24], true, 2, 4096, false, 0, FREE, &calls).unwrap_err();
            assert_eq!(
                error,
                qwen4_pp_budget_refusal(&measured(&[24, 24], 4096, 0), None).unwrap()
            );
            assert!(!error.contains("auto-fit"), "{error}");
            assert_eq!(*calls.borrow(), [(vec![24, 24], 4096)]);
            // Not even the chunk steps down for an explicit split that would fit smaller.
            let calls = Calls::default();
            assert!(search(&[25, 23], true, 2, 4096, false, 0, FREE, &calls).is_err());
            assert_eq!(calls.borrow().len(), 1);
            // An explicit split that fits is taken as given.
            let calls = Calls::default();
            let fit = search(&[25, 23], true, 2, 1024, true, 0, FREE, &calls).unwrap();
            assert_eq!(
                (fit.counts.as_slice(), fit.chunk_rows),
                (&[25, 23][..], 1024)
            );
            assert!(fit.moved_from.is_none());
            assert_eq!(calls.borrow().len(), 1);
        }

        #[test]
        fn nothing_fitting_reports_the_nearest_candidate() {
            let calls = Calls::default();
            let error =
                search(&[24, 24], false, 2, 4096, false, 0, 20_000_000_000, &calls).unwrap_err();
            for part in [
                "topology.pp = 2",
                "band does not fit device memory",
                "auto-fit searched chunk rows 4096,2048,1024,512,256 and every split",
                "the nearest, layers ",
                " at chunk 256, is short by ",
                "(the default, uniform 24,24 at chunk 4096, was short by ",
                "rebalance HIPFIRE_PP_LAYERS",
            ] {
                assert!(error.contains(part), "missing {part:?} in {error}");
            }
        }

        #[test]
        fn rungs_are_the_request_then_the_smaller_tiles() {
            assert_eq!(qwen4_pp_fit_chunk_rungs(4096), [4096, 2048, 1024, 512, 256]);
            assert_eq!(qwen4_pp_fit_chunk_rungs(2048), [2048, 1024, 512, 256]);
            assert_eq!(qwen4_pp_fit_chunk_rungs(1536), [1536, 1024, 512, 256]);
            assert_eq!(qwen4_pp_fit_chunk_rungs(256), [256]);
            assert_eq!(qwen4_pp_fit_chunk_rungs(100), [100]);
        }

        #[test]
        fn four_stages_walk_to_the_most_balanced_fitting_split() {
            // Band cost = a fixed part + 1 GB per layer; the first and last
            // bands carry the embedding side and the head.
            let base = [
                6_000_000_000u64,
                2_000_000_000,
                2_000_000_000,
                8_000_000_000,
            ];
            let lines = |counts: &[usize], free: u64| -> Vec<Qwen4PpRankFit> {
                counts
                    .iter()
                    .enumerate()
                    .map(|(stage, &count)| {
                        let start: usize = counts[..stage].iter().sum();
                        line(
                            stage,
                            start..start + count,
                            base[stage] + count as u64 * 1_000_000_000,
                            free,
                        )
                    })
                    .collect()
            };
            let run = |free: u64| {
                qwen4_pp_auto_fit(&[12, 12, 12, 12], false, 2, 4096, true, |counts, _| {
                    Ok(lines(counts, free))
                })
            };
            // Uniform fits: kept.
            let kept = run(20_000_000_000).unwrap();
            assert_eq!(kept.counts, [12, 12, 12, 12]);
            assert!(kept.moved_from.is_none());
            // Uniform does not fit: the walk finds the minimum of the maximum load.
            let fit = run(17_000_000_000).unwrap();
            assert!(fit.fits.iter().all(Qwen4PpRankFit::fits));
            assert_eq!(fit.counts.iter().sum::<usize>(), 48);
            assert!(fit.counts[0] >= 2 && fit.counts.iter().all(|&count| count >= 1));
            let mut best = u64::MAX;
            for c0 in 2..=45usize {
                for c1 in 1..=(46 - c0) {
                    for c2 in 1..=(47 - c0 - c1) {
                        let c3 = 48 - c0 - c1 - c2;
                        let max = [c0, c1, c2, c3]
                            .iter()
                            .enumerate()
                            .map(|(stage, &count)| base[stage] + count as u64 * 1_000_000_000)
                            .max()
                            .unwrap();
                        best = best.min(max);
                    }
                }
            }
            assert_eq!(fit.fits.iter().map(|fit| fit.required).max().unwrap(), best);
            assert_eq!(run(17_000_000_000), Ok(fit));
            // Too little memory anywhere: the nearest candidate is reported.
            assert!(run(10_000_000_000)
                .unwrap_err()
                .contains("the nearest, layers "));
        }

        fn real_bytes_of(entry: &WeightEntry) -> Option<u64> {
            Some(entry.logical_shape.iter().product::<usize>() as u64 * 2)
        }

        /// The REAL census lines of `counts` at `chunk` on the compact test config.
        fn real_lines(
            config: &Qwen4Config,
            weights: &[WeightEntry],
            counts: &[usize],
            chunk: usize,
            free: &[u64],
        ) -> Result<Vec<Qwen4PpRankFit>, String> {
            let mesh = DeviceMesh::rect(&[(DimKind::Pp, counts.len())]).unwrap();
            let placement = BandPlacement::from_counts(&mesh, 48, counts)?;
            let context = crate::Qwen4ContextCommit::legacy(1024);
            qwen4_pp_rank_lines(&Qwen4PpBudgetInputs {
                config,
                placement: &placement,
                weights,
                bytes_of: &real_bytes_of,
                format: crate::Qwen4StateFormat {
                    qsa: QsaKvFormat::Fp8,
                    gdn: crate::GdnStateFormat::Q8,
                    qsa_f16_arena: false,
                    qsa_pooled_bf16: false,
                },
                context: &context,
                chunk_rows: chunk,
                mode: Full,
                free_per_rank: free,
                reserve_bytes: QWEN4_PP_RESERVE_BYTES,
                native_mtp: None,
            })
        }

        #[test]
        fn the_real_census_search_matches_a_brute_force_oracle() {
            let config = crate::config::compact_test_config();
            let manifest = crate::weights::Qwen4Manifest::build(&config).unwrap();
            let weights = &manifest.weights;
            let at = |counts: &[usize], chunk: usize, free: &[u64]| {
                real_lines(&config, weights, counts, chunk, free)
            };
            let uniform = at(&[24, 24], 4096, &[u64::MAX; 2]).unwrap();
            // Free exactly the uniform census at 4096: kept, unmoved.
            let exact = [uniform[0].required, uniform[1].required];
            let kept = qwen4_pp_auto_fit(&[24, 24], false, 2, 4096, false, |counts, chunk| {
                at(counts, chunk, &exact)
            })
            .unwrap();
            assert_eq!(
                (kept.counts.as_slice(), kept.chunk_rows),
                (&[24, 24][..], 4096)
            );
            assert!(kept.moved_from.is_none());
            // One byte short on the heavier rank: the search moves; an oracle
            // over every split at every rung, largest chunk first, agrees.
            let heaviest = uniform.iter().map(|line| line.required).max().unwrap();
            let free = [heaviest - 1; 2];
            let search = || {
                qwen4_pp_auto_fit(&[24, 24], false, 2, 4096, false, |counts, chunk| {
                    at(counts, chunk, &free)
                })
            };
            let moved = search().unwrap();
            assert!(moved.fits.iter().all(Qwen4PpRankFit::fits));
            assert!(moved.moved_from.is_some());
            assert!(moved.chunk_rows < 4096 || moved.counts != [24, 24]);
            let (chunk, fitting) = qwen4_pp_fit_chunk_rungs(4096)
                .into_iter()
                .find_map(|chunk| {
                    let fitting: Vec<(Vec<usize>, u64)> = (2..48)
                        .filter_map(|first| {
                            let counts = vec![first, 48 - first];
                            let lines = at(&counts, chunk, &free).unwrap();
                            lines
                                .iter()
                                .all(Qwen4PpRankFit::fits)
                                .then(|| (counts, lines.iter().map(|l| l.required).max().unwrap()))
                        })
                        .collect();
                    (!fitting.is_empty()).then_some((chunk, fitting))
                })
                .expect("some rung fits one byte below the uniform census");
            assert_eq!(moved.chunk_rows, chunk);
            // Both ranks have the same free bytes, so the most balanced split
            // is the one with the smallest maximum census.
            let best = fitting.iter().map(|(_, max)| *max).min().unwrap();
            let balanced: Vec<&Vec<usize>> = fitting
                .iter()
                .filter(|(_, max)| *max == best)
                .map(|(counts, _)| counts)
                .collect();
            assert!(
                balanced.contains(&&moved.counts),
                "{:?} vs {balanced:?}",
                moved.counts
            );
            assert_eq!(search(), Ok(moved));
            // An explicit split is never searched: today's refusal, verbatim.
            let error = qwen4_pp_auto_fit(&[30, 18], true, 2, 4096, false, |counts, chunk| {
                at(counts, chunk, &free)
            })
            .unwrap_err();
            assert_eq!(
                error,
                qwen4_pp_budget_refusal(&at(&[30, 18], 4096, &free).unwrap(), None).unwrap()
            );
        }
    }
}
