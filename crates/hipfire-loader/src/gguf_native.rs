// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! GGUF as a native model source (CPU side only): parse, arch lookup, name
//! map, per-tensor admission, byte extents and the bpw census.
//!
//! This is the GGUF arm of the source-metadata seam in `carriers.rs`
//! (`SourceMeta`). The weights are never re-quantised: a GGUF tensor is
//! served as its ggml block bytes, so the manifest carries the ggml type
//! (as a distinct `DType::Ggml*`) and the exact byte extent of every tensor.
//! No kernel consumes these dtypes yet; execution sites refuse them
//! (`DType::ggml_refusal`).
//!
//! Loading is default-off: [`GGUF_NATIVE_ENV`] must be `1`. Every refusal is
//! worded `load refused: ...` (docs/load-refusal-wording.md) and names the
//! tensor, type or key at fault.

use hipfire_arch_qwen35::gguf_map::{
    hparams_from_gguf, ExpectedTensor, GgufTransform, Qwen35GgufHparams, Qwen35GgufMap,
    TensorClass, TensorGroup, GGUF_ARCHITECTURE,
};
use hipfire_runtime::gguf::{GgmlType, GgufFile, TensorInfo};
use hipfire_runtime::weight_manifest::{DTypeConstraint, ShardPolicy, WeightEntry};
use rdna_compute::DType;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

/// Opt-in switch for GGUF-native loading. Unset or anything but `1` refuses.
pub const GGUF_NATIVE_ENV: &str = "HIPFIRE_GGUF_NATIVE";

/// The hipfire arch id this slice admits (dense Qwen3.5 / 3.6 / 3.8).
pub const GGUF_NATIVE_ARCH_ID: u32 = 5;

/// ggml types served natively, with the dtype that carries their bytes.
/// Tier 1 of the admission plan: the K-quants and Q8_0 of the "XL" lifts,
/// IQ4_XS, and the float passthroughs. Anything else refuses per tensor.
pub const ADMITTED_TYPES: &[(GgmlType, DType)] = &[
    (GgmlType::F32, DType::F32),
    (GgmlType::F16, DType::F16),
    (GgmlType::BF16, DType::BF16),
    (GgmlType::Q4K, DType::GgmlQ4K),
    (GgmlType::Q5K, DType::GgmlQ5K),
    (GgmlType::Q6K, DType::GgmlQ6K),
    (GgmlType::Q8_0, DType::GgmlQ8_0),
    (GgmlType::IQ4XS, DType::GgmlIQ4XS),
];

/// The dtype carrying `ty`'s bytes, or `None` when the type is not admitted.
pub fn admitted_dtype(ty: GgmlType) -> Option<DType> {
    ADMITTED_TYPES
        .iter()
        .find_map(|&(t, d)| (t == ty).then_some(d))
}

fn admitted_list() -> String {
    ADMITTED_TYPES
        .iter()
        .map(|(t, _)| t.name())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Whether the opt-in switch is on, per the process config snapshot.
pub fn gguf_native_enabled() -> bool {
    hipfire_config::developer_var(GGUF_NATIVE_ENV).is_ok_and(|v| v == "1")
}

/// The default-off gate: `enabled` is [`gguf_native_enabled`] (injectable so
/// the refusal is testable without touching the process environment).
pub fn require_gguf_native(enabled: bool, path: &str) -> Result<(), String> {
    if enabled {
        return Ok(());
    }
    Err(format!(
        "load refused: {path} is a GGUF file and GGUF-native loading is off. Set \
         {GGUF_NATIVE_ENV}=1 to load GGUF models natively (experimental), or use an MQ/HFQ model"
    ))
}

/// Per-type totals over every tensor in the file.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TypeStat {
    pub tensors: usize,
    pub params: u64,
    pub bytes: u64,
}

/// What the file holds, before any admission decision.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GgufCensus {
    pub tensor_count: usize,
    pub total_params: u64,
    pub total_bytes: u64,
    /// ggml type name -> totals.
    pub by_type: BTreeMap<String, TypeStat>,
}

impl GgufCensus {
    /// Model bits per weight: `Σ bytes × 8 / Σ params` over every tensor.
    pub fn bpw(&self) -> f64 {
        if self.total_params == 0 {
            0.0
        } else {
            self.total_bytes as f64 * 8.0 / self.total_params as f64
        }
    }
}

/// Census of every tensor. A tensor with a ragged row (refused later by
/// admission) is still counted, at its rounded-up block extent.
pub fn census(tensors: &[TensorInfo]) -> GgufCensus {
    let mut c = GgufCensus {
        tensor_count: tensors.len(),
        ..Default::default()
    };
    for t in tensors {
        let params = t.numel() as u64;
        let bytes = t.extent_bytes().unwrap_or_else(|_| t.byte_size()) as u64;
        c.total_params += params;
        c.total_bytes += bytes;
        let s = c.by_type.entry(t.dtype.name().to_string()).or_default();
        s.tensors += 1;
        s.params += params;
        s.bytes += bytes;
    }
    c
}

/// One manifest row: the shared [`WeightEntry`] plus what GGUF adds (the
/// ggml type, exact byte extent and the exporter's data edits).
#[derive(Debug, Clone)]
pub struct GgufManifestEntry {
    /// The same declaration the HFQ path produces; `dtype` is the
    /// `DType::Ggml*` (or float) carrying the bytes, `logical_shape` is
    /// `[rows, cols]` (HF order, the reverse of the GGUF `ne` order).
    pub weight: WeightEntry,
    pub gguf_name: String,
    pub ggml_type: GgmlType,
    pub class: TensorClass,
    pub group: TensorGroup,
    pub transform: GgufTransform,
    /// Absolute byte offset of the tensor in the file (0 for a tied entry).
    pub file_offset: u64,
    /// Exact stored bytes (0 for a tied entry).
    pub byte_extent: u64,
    pub params: u64,
    /// Set for an absent `output.weight`: the LM head is the embedding.
    pub tied_to: Option<String>,
}

/// The loader's weight manifest for one GGUF: what to read, in what type,
/// from where, plus the bpw census (the §4 matching key).
#[derive(Debug, Clone)]
pub struct GgufManifest {
    pub arch_id: u32,
    pub entries: Vec<GgufManifestEntry>,
    pub census: GgufCensus,
    /// `census.bpw()`, stored so the badge / G3 reads one number.
    pub bpw: f64,
    /// Whether the file carries the NextN/MTP block.
    pub mtp_present: bool,
    /// Named reason the MTP block cannot be served natively (a tensor of a type
    /// this build does not serve, e.g. Q4_0 in bartowski files). The trunk
    /// still loads and serves AR; an MTP request refuses with this reason.
    pub mtp_refusal: Option<String>,
}

impl GgufManifest {
    /// The logical declarations, ready for the shared planner.
    pub fn weights(&self) -> Vec<WeightEntry> {
        self.entries.iter().map(|e| e.weight.clone()).collect()
    }

    /// Bytes the mapped tensors occupy (the quantised payload).
    pub fn payload_bytes(&self) -> u64 {
        self.entries.iter().map(|e| e.byte_extent).sum()
    }
}

/// Why a GGUF was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum RefusalKind {
    /// Unknown or wrong architecture.
    Arch,
    /// A hyper-parameter key is missing or inconsistent.
    Metadata,
    /// The same tensor name listed twice.
    Duplicate,
    /// A file tensor the dense qwen35 export does not contain.
    Extra,
    /// A tensor the export must contain is absent.
    Missing,
    /// A mapped tensor has the wrong shape for the config.
    Shape,
    /// A mapped tensor has a type this build does not serve.
    Type,
    /// Ragged rows, extent overflow, or bytes past the end of the file.
    Extent,
}

impl RefusalKind {
    /// Refusals that mean a tensor of the file is not covered by the name map
    /// (the S2 kill criterion): a different export layout, not a different
    /// quantisation.
    pub fn is_mapping(self) -> bool {
        matches!(
            self,
            RefusalKind::Extra | RefusalKind::Missing | RefusalKind::Shape | RefusalKind::Duplicate
        )
    }
}

/// One refusal: the kind, the tensor at fault (if any) and the
/// `load refused: ...` text.
#[derive(Debug, Clone)]
pub struct Refusal {
    pub kind: RefusalKind,
    pub tensor: Option<String>,
    pub message: String,
}

impl Refusal {
    fn tensor(kind: RefusalKind, tensor: &str, message: String) -> Self {
        Self {
            kind,
            tensor: Some(tensor.to_string()),
            message,
        }
    }
}

/// The result of reading one GGUF header: either a plan with no refusals or
/// the full refusal list (so `inspect` can show every problem at once).
#[derive(Debug)]
pub struct GgufAnalysis {
    pub architecture: Option<String>,
    pub arch_id: Option<u32>,
    pub hparams: Option<Qwen35GgufHparams>,
    pub census: GgufCensus,
    pub manifest: Option<GgufManifest>,
    /// Every refusal. Empty = admitted.
    pub refusals: Vec<Refusal>,
}

impl GgufAnalysis {
    pub fn admitted(&self) -> bool {
        self.refusals.is_empty() && self.manifest.is_some()
    }

    fn refuse(&mut self, kind: RefusalKind, tensor: Option<&str>, message: String) {
        self.refusals.push(Refusal {
            kind,
            tensor: tensor.map(str::to_string),
            message,
        });
    }

    /// Refusals that mean the name map does not cover the file (see
    /// [`RefusalKind::is_mapping`]).
    pub fn mapping_refusals(&self) -> Vec<&Refusal> {
        self.refusals
            .iter()
            .filter(|r| r.kind.is_mapping())
            .collect()
    }

    /// Refusal count per kind.
    pub fn refusal_counts(&self) -> BTreeMap<RefusalKind, usize> {
        let mut m = BTreeMap::new();
        for r in &self.refusals {
            *m.entry(r.kind).or_insert(0) += 1;
        }
        m
    }
}

fn arch_refusal(arch: &str, id: Option<u32>) -> Option<String> {
    match id {
        None => Some(format!(
            "load refused: GGUF general.architecture '{arch}' is not a model family hipfire knows"
        )),
        Some(GGUF_NATIVE_ARCH_ID) if arch == GGUF_ARCHITECTURE => None,
        Some(GGUF_NATIVE_ARCH_ID) => Some(format!(
            "load refused: GGUF general.architecture '{arch}' maps to arch {GGUF_NATIVE_ARCH_ID}, \
             but a dense Qwen3.5 GGUF names itself '{GGUF_ARCHITECTURE}'"
        )),
        Some(other) => Some(format!(
            "load refused: GGUF general.architecture '{arch}' is hipfire arch {other}; this build \
             loads only dense Qwen3.5 (arch {GGUF_NATIVE_ARCH_ID}) from GGUF"
        )),
    }
}

/// Read one GGUF header end to end: arch, hyper-parameters, name map,
/// admission, extents and the census. `check_extents` additionally requires
/// every tensor's bytes to lie inside the file (a truncated download refuses);
/// a header-only fixture analyses with it off.
pub fn analyze(gguf: &GgufFile, check_extents: bool) -> GgufAnalysis {
    let census = census(&gguf.tensors);
    let mut a = GgufAnalysis {
        architecture: None,
        arch_id: None,
        hparams: None,
        census,
        manifest: None,
        refusals: Vec::new(),
    };

    let Some(arch) = gguf.meta_str("general.architecture") else {
        a.refuse(
            RefusalKind::Arch,
            None,
            "load refused: GGUF has no general.architecture metadata, so the model family is \
             unknown"
                .to_string(),
        );
        return a;
    };
    a.architecture = Some(arch.to_string());
    a.arch_id = hipfire_runtime::arch_mapping::lookup_model_type(arch);
    if let Some(r) = arch_refusal(arch, a.arch_id) {
        a.refuse(RefusalKind::Arch, None, r);
        return a;
    }

    let hp = match hparams_from_gguf(&gguf.metadata, &gguf.tensors) {
        Ok(hp) => hp,
        Err(e) => {
            a.refuse(RefusalKind::Metadata, None, format!("load refused: {e}"));
            return a;
        }
    };
    let map = Qwen35GgufMap::new(&hp);

    // Duplicate names make "which tensor" ambiguous; refuse before mapping.
    let mut seen: HashSet<&str> = HashSet::new();
    for t in &gguf.tensors {
        if !seen.insert(t.name.as_str()) {
            a.refuse(
                RefusalKind::Duplicate,
                Some(&t.name),
                format!(
                    "load refused: GGUF lists tensor '{}' more than once",
                    t.name
                ),
            );
        }
    }

    // Extra tensors: present in the file, absent from the canonical export.
    let mut present: HashMap<&str, &TensorInfo> = HashMap::new();
    for t in &gguf.tensors {
        if map.lookup(&t.name).is_none() {
            a.refuse(
                RefusalKind::Extra,
                Some(&t.name),
                format!(
                    "load refused: GGUF tensor '{}' ({}) is not part of the dense qwen35 export \
                     (extra tensor); this file is not a canonical export",
                    t.name,
                    t.dtype.name()
                ),
            );
        }
        present.entry(t.name.as_str()).or_insert(t);
    }

    // Missing tensors. The MTP block is all-or-nothing: a trunk-only file has
    // none of it, a half-present block is a damaged export.
    let mtp_present = map
        .tensors
        .iter()
        .any(|e| e.group == TensorGroup::Mtp && present.contains_key(e.gguf_name.as_str()));
    for e in &map.tensors {
        let needed = !e.optional && (e.group == TensorGroup::Trunk || mtp_present);
        if needed && !present.contains_key(e.gguf_name.as_str()) {
            a.refuse(
                RefusalKind::Missing,
                Some(&e.gguf_name),
                format!(
                    "load refused: GGUF is missing tensor '{}' (hipfire '{}'); this file is not a \
                     complete dense qwen35 export",
                    e.gguf_name, e.hipfire_name
                ),
            );
        }
    }

    // Per-tensor shape, type, extent.
    let mut entries: Vec<GgufManifestEntry> = Vec::new();
    let mut embed_entry: Option<(String, DType)> = None;
    let mut have_output = false;
    let mut mtp_refusal: Option<String> = None;
    for e in &map.tensors {
        let Some(info) = present.get(e.gguf_name.as_str()).copied() else {
            continue;
        };
        match entry_for(gguf, e, info, check_extents) {
            Ok(entry) => {
                if e.gguf_name == "token_embd.weight" {
                    embed_entry = Some((e.hipfire_name.clone(), entry.weight.dtype));
                }
                if e.gguf_name == "output.weight" {
                    have_output = true;
                }
                entries.push(entry);
            }
            // A NextN/MTP tensor of a type this build does not serve (Q4_0 in
            // plain imatrix files) makes the MTP block unservable, not the
            // trunk unloadable: AR serves, MTP refuses with this reason.
            Err(r) if e.group == TensorGroup::Mtp && r.kind == RefusalKind::Type => {
                mtp_refusal.get_or_insert(format!(
                    "MTP is not served from this GGUF: {}",
                    r.message.trim_start_matches("load refused: ")
                ));
            }
            Err(r) => a.refusals.push(r),
        }
    }
    if !have_output {
        if let Some((name, dtype)) = embed_entry {
            // Tied head: a declaration with no bytes of its own.
            let weight = WeightEntry::model_with_dtype_constraint(
                "lm_head.weight",
                vec![hp.config.vocab_size, hp.config.dim],
                dtype,
                DTypeConstraint::source_exact(dtype),
                ShardPolicy::Tied {
                    source: name.clone(),
                },
            );
            let ggml_type = ADMITTED_TYPES
                .iter()
                .find_map(|&(t, d)| (d == dtype).then_some(t))
                .unwrap_or(GgmlType::F16);
            entries.push(GgufManifestEntry {
                weight,
                gguf_name: "output.weight".to_string(),
                ggml_type,
                class: TensorClass::Matrix,
                group: TensorGroup::Trunk,
                transform: GgufTransform::AS_STORED,
                file_offset: 0,
                byte_extent: 0,
                params: 0,
                tied_to: Some(name),
            });
        }
    }

    a.hparams = Some(hp);
    if a.refusals.is_empty() {
        let bpw = a.census.bpw();
        a.manifest = Some(GgufManifest {
            arch_id: GGUF_NATIVE_ARCH_ID,
            entries,
            census: a.census.clone(),
            bpw,
            mtp_present,
            mtp_refusal,
        });
    }
    a
}

fn entry_for(
    gguf: &GgufFile,
    e: &ExpectedTensor,
    info: &TensorInfo,
    check_extents: bool,
) -> Result<GgufManifestEntry, Refusal> {
    let refuse = |kind, message: String| Refusal::tensor(kind, &info.name, message);
    if info.shape != e.ggml_shape {
        return Err(refuse(
            RefusalKind::Shape,
            format!(
                "load refused: GGUF tensor '{}' has shape {:?}, but this model's config expects \
                 {:?} (GGUF axis order, fastest first)",
                info.name, info.shape, e.ggml_shape
            ),
        ));
    }
    let dtype = admitted_dtype(info.dtype).ok_or_else(|| {
        refuse(
            RefusalKind::Type,
            format!(
                "load refused: GGUF tensor '{}' uses type {} (ggml id {}), which this build does \
                 not serve natively; admitted types: {}. Use a quantization built from those \
                 types",
                info.name,
                info.dtype.name(),
                info.dtype.ggml_id(),
                admitted_list()
            ),
        )
    })?;
    if e.class != TensorClass::Matrix && !info.dtype.is_float() {
        return Err(refuse(
            RefusalKind::Type,
            format!(
                "load refused: GGUF tensor '{}' is a {:?} tensor stored as {}; only F32, F16 or \
                 BF16 are admitted for it",
                info.name,
                e.class,
                info.dtype.name()
            ),
        ));
    }
    let extent = info.extent_bytes().map_err(|why| {
        refuse(
            RefusalKind::Extent,
            format!(
                "load refused: {why}; ggml stores whole blocks per row ({} elements for {}), so \
                 this file does not match its declared type",
                info.dtype.block_size(),
                info.dtype.name()
            ),
        )
    })?;
    let file_offset = gguf
        .tensor_data_offset
        .checked_add(info.offset)
        .ok_or_else(|| {
            refuse(
                RefusalKind::Extent,
                format!("load refused: GGUF tensor '{}' offset overflows", info.name),
            )
        })?;
    if check_extents {
        let end = file_offset.checked_add(extent);
        if end.map_or(true, |end| end > gguf.file_len()) {
            return Err(refuse(
                RefusalKind::Extent,
                format!(
                    "load refused: GGUF tensor '{}' needs {extent} bytes at offset {file_offset} \
                     but the file is only {} bytes; the download may be incomplete, so \
                     re-download the model",
                    info.name,
                    gguf.file_len()
                ),
            ));
        }
    }
    let logical_shape: Vec<usize> = e.ggml_shape.iter().rev().copied().collect();
    let constraint = DTypeConstraint::source_exact(dtype);
    let policy = ShardPolicy::Replicate;
    let weight = match e.layer {
        Some(layer) => WeightEntry::layer_with_dtype_constraint(
            e.hipfire_name.clone(),
            layer,
            logical_shape,
            dtype,
            constraint,
            policy,
        ),
        None => WeightEntry::model_with_dtype_constraint(
            e.hipfire_name.clone(),
            logical_shape,
            dtype,
            constraint,
            policy,
        ),
    };
    Ok(GgufManifestEntry {
        weight,
        gguf_name: info.name.clone(),
        ggml_type: info.dtype,
        class: e.class,
        group: e.group,
        transform: e.transform,
        file_offset: file_offset as u64,
        byte_extent: extent as u64,
        params: info.numel() as u64,
        tied_to: None,
    })
}

/// A parsed, admitted GGUF model (CPU side): the mapped file, the
/// hyper-parameters in the shared qwen35 config struct and the manifest.
pub struct GgufNativeModel {
    pub path: PathBuf,
    pub arch_id: u32,
    pub hparams: Qwen35GgufHparams,
    pub manifest: GgufManifest,
    gguf: GgufFile,
}

impl GgufNativeModel {
    /// Open a GGUF for native serving. Refuses (default) unless
    /// `HIPFIRE_GGUF_NATIVE=1`; then parses, maps, admits per tensor, checks
    /// every extent against the file, and logs the bpw census.
    pub fn open(path: &Path) -> Result<Self, String> {
        Self::open_gated(path, gguf_native_enabled())
    }

    /// [`GgufNativeModel::open`] with the gate injected.
    pub fn open_gated(path: &Path, enabled: bool) -> Result<Self, String> {
        let shown = path.display().to_string();
        require_gguf_native(enabled, &shown)?;
        let gguf = GgufFile::open_native(path)
            .map_err(|e| format!("load refused: could not read GGUF {shown}: {e}"))?;
        let mut analysis = analyze(&gguf, true);
        if !analysis.admitted() {
            return Err(analysis
                .refusals
                .first()
                .map(|r| r.message.clone())
                .unwrap_or_else(|| format!("load refused: {shown} was not admitted")));
        }
        let manifest = analysis.manifest.take().expect("admitted has a manifest");
        let hparams = analysis.hparams.take().expect("admitted has hparams");
        eprintln!(
            "  [gguf] {shown}: arch {} ({}), {} tensors, {:.1} MiB payload, {:.4} bpw (Σbytes*8/Σparams over {} params){}",
            manifest.arch_id,
            analysis.architecture.as_deref().unwrap_or("?"),
            manifest.census.tensor_count,
            manifest.payload_bytes() as f64 / (1024.0 * 1024.0),
            manifest.bpw,
            manifest.census.total_params,
            if manifest.mtp_present { ", MTP block present" } else { "" },
        );
        eprintln!("  {}", GgufBadge::of(&manifest).line());
        if let Some(reason) = &manifest.mtp_refusal {
            eprintln!("  [gguf] {reason}; the trunk serves AR only");
        }
        Ok(Self {
            path: path.to_path_buf(),
            arch_id: manifest.arch_id,
            hparams,
            manifest,
            gguf,
        })
    }

    pub fn gguf(&self) -> &GgufFile {
        &self.gguf
    }

    /// The tensor's bytes, by manifest entry.
    pub fn tensor_bytes(&self, entry: &GgufManifestEntry) -> Option<&[u8]> {
        self.gguf.find_tensor(&entry.gguf_name).and_then(|t| {
            if entry.tied_to.is_some() {
                return None;
            }
            self.gguf.try_tensor_data(t)
        })
    }

    /// The source metadata the carrier seam needs (tokenizer, chat template,
    /// arch id): the GGUF arm of `resolve_source_meta`.
    pub fn source_meta(&self) -> Result<GgufSourceMeta, String> {
        let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_gguf(&self.gguf)
            .map_err(|e| format!("tokenizer not found: {e}"))?;
        let model_path = self.path.to_string_lossy();
        let (chat_template, chat_template_source) =
            match crate::resolve_chat_template_overrides(&model_path) {
                Some((s, source)) => (Some(s), Some(source)),
                None => {
                    let embedded = self
                        .gguf
                        .meta_str("tokenizer.chat_template")
                        .filter(|t| !t.trim().is_empty())
                        .map(str::to_owned);
                    let from_file = embedded.is_some();
                    let (t, source) = crate::qwen35_template_from_embedded(embedded, &model_path);
                    let source = if from_file {
                        "embedded GGUF tokenizer.chat_template".to_string()
                    } else {
                        source
                    };
                    (Some(t), Some(source))
                }
            };
        Ok(GgufSourceMeta {
            tokenizer,
            chat_template,
            chat_template_source,
            arch_id: self.arch_id,
        })
    }

    /// Load the dense trunk onto `gpu` (single device): every tensor stays in
    /// its ggml block bytes (`DType::Ggml*`), the exporter's data edits are
    /// undone by [`hipfire_arch_qwen35::gguf_source::GgufSource`]. The NextN/MTP
    /// block is never loaded.
    pub fn load_qwen35_weights(
        &self,
        gpu: &mut rdna_compute::Gpu,
    ) -> Result<hipfire_arch_qwen35::qwen35::Qwen35Weights, String> {
        use hipfire_arch_qwen35::qwen35::{load_weights, Layout};
        let mut src = hipfire_arch_qwen35::gguf_source::GgufSource::new(&self.gguf, &self.hparams);
        let layout = Layout::single(self.hparams.config.n_layers);
        load_weights(&mut src, std::slice::from_mut(gpu), &layout)
            .map_err(|e| format!("load_weights: {e:?}"))
    }

    /// The named reason an MTP request refuses on this file, if any: either the
    /// file has no MTP block, or its block holds a tensor type this build does
    /// not serve. GGUF-native serving is AR-only; MTP is never loaded.
    pub fn mtp_refusal(&self) -> String {
        match (&self.manifest.mtp_present, &self.manifest.mtp_refusal) {
            (_, Some(reason)) => reason.clone(),
            (true, None) => "MTP from GGUF files is not served by the GGUF-native route yet: the \
                NextN block is a sidecar head with its own tensor contract; the trunk serves AR"
                .to_string(),
            (false, None) => "this GGUF has no NextN/MTP block; the trunk serves AR".to_string(),
        }
    }
}

/// The G3 badge fields of a GGUF-native model: measured bits per weight, the
/// numeric route its linears run on and the quality class (`UNMEASURED` until
/// a KLD campaign registers one).
#[derive(Debug, Clone, PartialEq)]
pub struct GgufBadge {
    pub bpw: f64,
    pub route: &'static str,
    pub kld_class: &'static str,
}

/// Numeric route of the decode linears (`ggml.a8q81.<type>`: q8_1 activations,
/// int8 block dots, f32 scale fold).
pub const GGUF_ROUTE: &str = "ggml.a8q81";

impl GgufBadge {
    pub fn of(manifest: &GgufManifest) -> Self {
        GgufBadge {
            bpw: manifest.bpw,
            route: GGUF_ROUTE,
            kld_class: "UNMEASURED",
        }
    }

    /// The one-line form used by the load log and `hipfire inspect`.
    pub fn line(&self) -> String {
        format!(
            "[gguf] badge: bpw={:.4} route={} kld_class={}",
            self.bpw, self.route, self.kld_class
        )
    }
}

/// Source-only metadata of a GGUF model; converts into the carrier seam's
/// `SourceMeta` (`carriers.rs`).
pub struct GgufSourceMeta {
    pub tokenizer: hipfire_runtime::tokenizer::Tokenizer,
    pub chat_template: Option<String>,
    pub chat_template_source: Option<String>,
    pub arch_id: u32,
}

/// `hipfire inspect <file.gguf>`: the report, as text. Reads the header only
/// (no gate: inspecting is read-only) and prints the load verdict, including
/// whether the opt-in switch is on.
pub fn inspect_report(path: &Path, native_enabled: bool) -> Result<String, String> {
    use std::fmt::Write as _;
    let shown = path.display();
    let gguf =
        GgufFile::open_native(path).map_err(|e| format!("could not read GGUF {shown}: {e}"))?;
    // Bytes the tensor table says the file must hold; a header-only dump or a
    // partial download falls short. Extents are then reported once here
    // instead of as one refusal per tensor.
    let needed = gguf
        .tensors
        .iter()
        .filter_map(|t| {
            let end = gguf.tensor_data_offset.checked_add(t.offset)?;
            end.checked_add(t.extent_bytes().ok()?)
        })
        .max()
        .unwrap_or(gguf.tensor_data_offset);
    let truncated = needed > gguf.file_len();
    let analysis = analyze(&gguf, !truncated);
    let mut out = String::new();
    let _ = writeln!(out, "file:          {shown}");
    let _ = writeln!(out, "gguf version:  {}", gguf.version);
    if truncated {
        let _ = writeln!(
            out,
            "data:          INCOMPLETE: the file is {} bytes but its tensor table needs {needed} \
             (header-only dump or partial download); tensor extents were not verified, and \
             loading this file would refuse",
            gguf.file_len()
        );
    } else {
        let _ = writeln!(out, "data:          complete ({} bytes)", gguf.file_len());
    }
    let _ = writeln!(
        out,
        "architecture:  {} (hipfire arch {})",
        analysis.architecture.as_deref().unwrap_or("<missing>"),
        analysis
            .arch_id
            .map_or_else(|| "unknown".to_string(), |id| id.to_string())
    );
    let _ = writeln!(out, "tensors:       {}", analysis.census.tensor_count);
    let _ = writeln!(out, "type histogram:");
    let mut rows: Vec<(&String, &TypeStat)> = analysis.census.by_type.iter().collect();
    rows.sort_by(|a, b| b.1.bytes.cmp(&a.1.bytes).then(a.0.cmp(b.0)));
    for (name, s) in rows {
        let admitted = ggml_by_name(name).and_then(admitted_dtype).is_some();
        let _ = writeln!(
            out,
            "  {name:<8} {:>5} tensors  {:>14} params  {:>10.2} MiB  {}",
            s.tensors,
            s.params,
            s.bytes as f64 / (1024.0 * 1024.0),
            if admitted { "admitted" } else { "NOT admitted" }
        );
    }
    let _ = writeln!(
        out,
        "model bpw:     {:.4}  (Σbytes*8 / Σparams = {} B * 8 / {} params)",
        analysis.census.bpw(),
        analysis.census.total_bytes,
        analysis.census.total_params
    );
    if let Some(hp) = &analysis.hparams {
        let c = &hp.config;
        let _ = writeln!(
            out,
            "shape:         dim {}, {} layers (+{} MTP), vocab {}, heads {}/{} kv x {} head_dim, ffn {}",
            c.dim, c.n_layers, hp.nextn_layers, c.vocab_size, c.n_heads, c.n_kv_heads, c.head_dim, c.hidden_dim
        );
    }
    if analysis.admitted() {
        let _ = writeln!(
            out,
            "admission:     ADMITTED (every tensor mapped; {} manifest entries{})",
            analysis.manifest.as_ref().map_or(0, |m| m.entries.len()),
            if analysis.manifest.as_ref().is_some_and(|m| m.mtp_present) {
                ", MTP block present"
            } else {
                ""
            }
        );
        if let Some(m) = &analysis.manifest {
            let _ = writeln!(out, "{}", GgufBadge::of(m).line());
            if let Some(reason) = &m.mtp_refusal {
                let _ = writeln!(out, "mtp:           {reason}");
            }
        }
    } else {
        let counts = analysis
            .refusal_counts()
            .iter()
            .map(|(k, n)| format!("{k:?}={n}"))
            .collect::<Vec<_>>()
            .join(", ");
        let _ = writeln!(
            out,
            "admission:     REFUSED ({} problems: {counts}; mapping problems = {})",
            analysis.refusals.len(),
            analysis.mapping_refusals().len()
        );
        // Mapping problems first: they mean the name map does not cover the file.
        let mut ordered: Vec<&Refusal> = analysis.refusals.iter().collect();
        ordered.sort_by_key(|r| (!r.kind.is_mapping(), r.kind));
        for r in ordered.iter().take(20) {
            let _ = writeln!(out, "  - {}", r.message);
        }
        if ordered.len() > 20 {
            let _ = writeln!(out, "  ... and {} more", ordered.len() - 20);
        }
    }
    let _ =
        writeln!(
        out,
        "native load:   {} ({GGUF_NATIVE_ENV}=1 {}; admitted files serve AR on gfx1201: decode \
         GEMV on q8_1 activations, prefill by dequantisation to F16)",
        if native_enabled { "switch ON" } else { "switch OFF" },
        if native_enabled { "set" } else { "required" },
    );
    Ok(out)
}

/// Inverse of [`GgmlType::name`] over the recognised set (for the histogram).
fn ggml_by_name(name: &str) -> Option<GgmlType> {
    (0u32..64)
        .filter_map(GgmlType::from_u32)
        .find(|t| t.name() == name)
}

#[cfg(test)]
pub(crate) mod test_writer;

#[cfg(test)]
mod tests;
