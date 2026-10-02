// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Family-neutral serving cache/scheduler contract types (spec §3).
//!
//! These are the **proposed operation** data contracts that freeze the
//! cross-slice boundary before allocator, radix index, scheduler, admission
//! and grammar implementation proceed independently against them (spec §3,
//! plan §3 "Cross-slice contract freeze"). They are plain owned structs/enums:
//! no trait hierarchies, no architecture downcasts. `rdna-compute` owns the
//! matching *physical* types (`PageHandle`, `PageState`, `LeaseClass`,
//! `CowPlan`); this module owns the *logical* request/step surface consumed by
//! arch crates without upward dependencies (spec §3 ownership table,
//! `hipfire-runtime` row).
//!
//! Nothing here performs GPU mutation or scheduling; the types only describe
//! what the seven proposed operations exchange (spec §3 operation table):
//! prefix lookup, resume planning, step reservation, submit step, commit step,
//! publish cache, and release.

use std::fmt;

/// Cryptographic content digest for model/sidecar/tokenizer/template
/// identity. Computed once at load, never per request or per token (spec
/// §4.1). Stored as raw bytes; the canonical serialization is
/// length-delimited so equal digests serialize identically and a
/// non-cryptographic hash collision is never treated as equality (spec §4.1).
pub type Digest = Vec<u8>;

/// SHA-256 over length-prefixed byte parts. Equal parts produce equal
/// digests; a single-field change produces a different digest. Used by
/// tokenizer/template/HFQ identity constructors (spec §4.1).
pub fn sha256_len_prefixed(parts: &[&[u8]]) -> Digest {
    use sha2::{Digest as Sha256Digest, Sha256};
    let mut h = Sha256::new();
    for p in parts {
        h.update((*p).len().to_le_bytes());
        h.update(*p);
    }
    h.finalize().to_vec()
}

/// SHA-256 of a file's contents. Used for sidecar/adapter identity when
/// the sidecar is not an HFQ (spec §4.1). Load-time only.
pub fn sha256_file(path: &std::path::Path) -> std::io::Result<Digest> {
    use sha2::{Digest as Sha256Digest, Sha256};
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(h.finalize().to_vec())
}

// =========================================================================
// C1 — Cache domain identity (spec §4.1)
// =========================================================================

/// Tokenizer vocabulary/configuration identity (spec §4.1: "tokenizer
/// vocabulary/configuration and prompt-template/normalization identity").
///
/// Re-encoding an old assistant transcript is not proof of a hit, so the
/// vocab and config digests are independent components of the domain.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TokenizerIdentity {
    /// Digest of the decoded vocabulary table actually used for encoding.
    pub vocab_digest: Digest,
    /// Digest of tokenizer configuration (special tokens, merge rules, BOS).
    pub config_digest: Digest,
}

/// Prompt-template and normalization identity (spec §4.1). Verbatim
/// template-splice behavior must be preserved; a different template or
/// normalization stream is a different domain even for equal tokens.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TemplateIdentity {
    /// Digest of the rendered chat template program (e.g. the HF chat_template).
    pub template_digest: Digest,
    /// Tag identifying the normalization/splice policy in use.
    pub normalization_tag: String,
}

/// Architecture, numerical state ABI, and position/attention policy tag
/// (spec §4.1: "architecture, numerical state ABI and position/attention
/// policy"). A full-attention, sliding-window and recurrent model each carry
/// a different resume contract; this tag prevents cross-family reuse.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ArchPolicy {
    /// Architecture family tag (e.g. "qwen35-deltanet", "llama-full-attn").
    pub arch_tag: String,
    /// Numerical state ABI tag (recurrent matrix layout, error-feedback mode).
    pub state_abi_tag: String,
    /// Position/attention policy tag (M-RoPE, sliding window, full causal).
    pub position_attention_tag: String,
}

/// KV K/V encoding, strides and layout tag (spec §4.1: "KV K/V encoding,
/// strides, group/layer layout, recurrent quantization and error-feedback
/// mode"). Copies must preserve encoded bytes and distinct K/V strides (spec
/// §4.3); mismatched encoding or strides is a different domain.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct KvLayout {
    /// K stride bytes per layer (includes quant scales/headers, spec §5.1).
    pub k_stride_bytes: Vec<u64>,
    /// V stride bytes per layer.
    pub v_stride_bytes: Vec<u64>,
    /// Encoding tag (q8/asym/fwht) plus group/layer layout descriptor.
    pub layout_tag: String,
}

/// Device/topology identity and allocation epoch for device-resident
/// artifacts (spec §4.1: "device/topology and allocation epoch"). Model
/// unload/reload advances the epoch and invalidates lookups (spec §4.6).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DeviceTopology {
    /// Stable device identity (not a HIP ordinal, which is renumbered).
    pub device_id: String,
    /// Topology/placement descriptor for multi-GPU placements.
    pub topology_id: String,
    /// Allocation epoch; advanced on unload/reload to invalidate stale state.
    pub allocation_epoch: u64,
}

/// Authenticated sharing namespace (spec §4.1: "authenticated sharing
/// namespace"; spec §4.1 isolation). A client-supplied namespace/salt cannot
/// grant membership in another domain; sharing is private to a trust domain
/// by default.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SharingNamespace {
    /// Authenticated trust domain identifier (operator-authorized).
    pub domain_id: String,
}

/// C1 cache domain identity (spec §4.1).
///
/// Binds every forward-affecting property that must be identical for two
/// processed token boundaries to share state: model content + load epoch,
/// sidecar/adapter digests, tokenizer identity, template/normalization
/// identity, arch + state ABI + position/attention policy, KV encoding +
/// strides + layout, device/topology + allocation epoch, and the
/// authenticated sharing namespace. Keys use the actual model input token
/// sequence (held by the radix index, not here), never rendered-character
/// prefixes, filenames or slot IDs (spec §4.1).
///
/// Derives `Eq`/`Hash` so domains can be keyed directly, and provides a
/// canonical, versioned, length-delimited serialization for radix edge
/// comparison and digest acceleration (spec §4.1: "canonical, versioned,
/// length-delimited serialization").
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CacheDomain {
    /// Digest of the model content (weights) and the load epoch that stamped
    /// it. Never a filename (spec §4.1, §4.6).
    pub model_content_digest: Digest,
    /// Load epoch for the model content.
    pub model_load_epoch: u64,
    /// Digests of relevant sidecars/adapters, in canonical (sorted) order.
    pub sidecar_digests: Vec<Digest>,
    /// Tokenizer vocabulary/configuration identity.
    pub tokenizer: TokenizerIdentity,
    /// Prompt-template/normalization identity.
    pub template: TemplateIdentity,
    /// Architecture + state ABI + position/attention policy.
    pub arch_policy: ArchPolicy,
    /// KV K/V encoding, strides and layout.
    pub kv_layout: KvLayout,
    /// Device/topology id + allocation epoch.
    pub device: DeviceTopology,
    /// Authenticated sharing namespace.
    pub namespace: SharingNamespace,
}

/// Canonical serialization version tag. Bumped only on a breaking change to
/// the field order or encoding; consumers compare exact serialized bytes
/// within a version (spec §4.1).
const CACHE_DOMAIN_CANONICAL_VERSION: u8 = 1;

impl CacheDomain {
    /// Canonical, versioned, length-delimited serialization for keying and
    /// radix edge comparison (spec §4.1). Every field is emitted in a fixed
    /// order; byte vectors and strings are length-prefixed, integers are
    /// fixed-width little-endian. Equal domains produce equal bytes, and two
    /// domains differing in any single field serialize differently. No raw
    /// pointers, no filenames.
    pub fn to_canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        put_u8(&mut out, CACHE_DOMAIN_CANONICAL_VERSION);
        put_bytes(&mut out, &self.model_content_digest);
        put_u64(&mut out, self.model_load_epoch);
        put_u64(&mut out, self.sidecar_digests.len() as u64);
        for d in &self.sidecar_digests {
            put_bytes(&mut out, d);
        }
        put_bytes(&mut out, &self.tokenizer.vocab_digest);
        put_bytes(&mut out, &self.tokenizer.config_digest);
        put_bytes(&mut out, &self.template.template_digest);
        put_str(&mut out, &self.template.normalization_tag);
        put_str(&mut out, &self.arch_policy.arch_tag);
        put_str(&mut out, &self.arch_policy.state_abi_tag);
        put_str(&mut out, &self.arch_policy.position_attention_tag);
        put_u64(&mut out, self.kv_layout.k_stride_bytes.len() as u64);
        for s in &self.kv_layout.k_stride_bytes {
            put_u64(&mut out, *s);
        }
        put_u64(&mut out, self.kv_layout.v_stride_bytes.len() as u64);
        for s in &self.kv_layout.v_stride_bytes {
            put_u64(&mut out, *s);
        }
        put_str(&mut out, &self.kv_layout.layout_tag);
        put_str(&mut out, &self.device.device_id);
        put_str(&mut out, &self.device.topology_id);
        put_u64(&mut out, self.device.allocation_epoch);
        put_str(&mut out, &self.namespace.domain_id);
        out
    }

    /// Inverse of [`CacheDomain::to_canonical_bytes`]. Returns the domain
    /// only if the byte stream is a complete, well-formed canonical encoding
    /// at the current version; otherwise an error. Used for radix persistence
    /// and roundtrip verification.
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, CanonicalError> {
        let mut cur = bytes;
        let version = take_u8(&mut cur)?;
        if version != CACHE_DOMAIN_CANONICAL_VERSION {
            return Err(CanonicalError::Version(version));
        }
        let model_content_digest = take_bytes(&mut cur)?;
        let model_load_epoch = take_u64(&mut cur)?;
        let sidecar_count = take_count(&mut cur, 8)?;
        let mut sidecar_digests = Vec::with_capacity(sidecar_count);
        for _ in 0..sidecar_count {
            sidecar_digests.push(take_bytes(&mut cur)?);
        }
        let vocab_digest = take_bytes(&mut cur)?;
        let config_digest = take_bytes(&mut cur)?;
        let template_digest = take_bytes(&mut cur)?;
        let normalization_tag = take_str(&mut cur)?;
        let arch_tag = take_str(&mut cur)?;
        let state_abi_tag = take_str(&mut cur)?;
        let position_attention_tag = take_str(&mut cur)?;
        let k_len = take_count(&mut cur, 8)?;
        let mut k_stride_bytes = Vec::with_capacity(k_len);
        for _ in 0..k_len {
            k_stride_bytes.push(take_u64(&mut cur)?);
        }
        let v_len = take_count(&mut cur, 8)?;
        let mut v_stride_bytes = Vec::with_capacity(v_len);
        for _ in 0..v_len {
            v_stride_bytes.push(take_u64(&mut cur)?);
        }
        let layout_tag = take_str(&mut cur)?;
        let device_id = take_str(&mut cur)?;
        let topology_id = take_str(&mut cur)?;
        let allocation_epoch = take_u64(&mut cur)?;
        let domain_id = take_str(&mut cur)?;
        if !cur.is_empty() {
            return Err(CanonicalError::Trailing(cur.len()));
        }
        Ok(Self {
            model_content_digest,
            model_load_epoch,
            sidecar_digests,
            tokenizer: TokenizerIdentity {
                vocab_digest,
                config_digest,
            },
            template: TemplateIdentity {
                template_digest,
                normalization_tag,
            },
            arch_policy: ArchPolicy {
                arch_tag,
                state_abi_tag,
                position_attention_tag,
            },
            kv_layout: KvLayout {
                k_stride_bytes,
                v_stride_bytes,
                layout_tag,
            },
            device: DeviceTopology {
                device_id,
                topology_id,
                allocation_epoch,
            },
            namespace: SharingNamespace { domain_id },
        })
    }
}

/// Error raised by [`CacheDomain::from_canonical_bytes`] when the byte stream
/// is malformed, truncated, carries an unknown version, or has trailing
/// bytes (spec §4.1 canonical serialization).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CanonicalError {
    /// The byte stream ended before a complete field was read.
    Truncated,
    /// A length prefix exceeded the remaining bytes.
    Oversized,
    /// An unexpected canonical version tag was encountered.
    Version(u8),
    /// A string field was not valid UTF-8.
    Utf8,
    /// Unconsumed trailing bytes after the domain decoded.
    Trailing(usize),
}

impl fmt::Display for CanonicalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated => f.write_str("canonical cache domain stream truncated"),
            Self::Oversized => f.write_str("canonical cache domain length prefix oversized"),
            Self::Version(v) => write!(f, "canonical cache domain version {v} unsupported"),
            Self::Utf8 => f.write_str("canonical cache domain string not valid UTF-8"),
            Self::Trailing(n) => {
                write!(f, "canonical cache domain stream has {n} trailing bytes")
            }
        }
    }
}

impl std::error::Error for CanonicalError {}

// ---- canonical serialization helpers ----

fn put_u8(out: &mut Vec<u8>, v: u8) {
    out.push(v);
}

fn put_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn put_bytes(out: &mut Vec<u8>, b: &[u8]) {
    put_u64(out, b.len() as u64);
    out.extend_from_slice(b);
}

fn put_str(out: &mut Vec<u8>, s: &str) {
    put_bytes(out, s.as_bytes());
}

fn take_u8(cur: &mut &[u8]) -> Result<u8, CanonicalError> {
    let (b, rest) = cur.split_first().ok_or(CanonicalError::Truncated)?;
    *cur = rest;
    Ok(*b)
}

fn take_u64(cur: &mut &[u8]) -> Result<u64, CanonicalError> {
    if cur.len() < 8 {
        return Err(CanonicalError::Truncated);
    }
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&cur[..8]);
    *cur = &cur[8..];
    Ok(u64::from_le_bytes(buf))
}

/// Read a collection count without allocating from an attacker-controlled
/// u64. Every encoded member consumes at least `min_member_bytes` bytes, so a
/// count larger than the remaining stream can never be valid.
fn take_count(cur: &mut &[u8], min_member_bytes: usize) -> Result<usize, CanonicalError> {
    let raw = take_u64(cur)?;
    let count = usize::try_from(raw).map_err(|_| CanonicalError::Oversized)?;
    let max_count = cur.len() / min_member_bytes.max(1);
    if count > max_count {
        return Err(CanonicalError::Oversized);
    }
    Ok(count)
}

fn take_bytes(cur: &mut &[u8]) -> Result<Vec<u8>, CanonicalError> {
    let len = take_u64(cur)? as usize;
    if cur.len() < len {
        return Err(CanonicalError::Oversized);
    }
    let v = cur[..len].to_vec();
    *cur = &cur[len..];
    Ok(v)
}

fn take_str(cur: &mut &[u8]) -> Result<String, CanonicalError> {
    String::from_utf8(take_bytes(cur)?).map_err(|_| CanonicalError::Utf8)
}

// =========================================================================
// Prefix lookup — C2 (spec §4.2)
// =========================================================================

/// Why a prefix lookup did not yield a resumable boundary (spec §4.2, §4.5).
///
/// A prefix match that lacks a recurrent checkpoint is a miss in usage
/// accounting, not a hit (spec §4.2). Unknown adapters return a miss rather
/// than relaxing `CachePolicy` safety (spec §4.5).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum MissReason {
    /// No equal token prefix exists in the index.
    NoMatch,
    /// A token prefix matched but no resumable-state checkpoint exists at any
    /// boundary (spec §4.2: matched != resumable).
    NoCheckpoint,
    /// The matching entry's [`CacheDomain`] differs from the request's
    /// identity (spec §4.1).
    IdentityMismatch,
    /// The matching pages/checkpoint were evicted and no longer resident
    /// (spec §4.4).
    Evicted,
    /// The matching entry's position/attention or KV policy is incompatible
    /// with the request (spec §4.5).
    IncompatiblePolicy,
    /// A required adapter is unknown; reuse is refused rather than relaxing
    /// safety (spec §4.5).
    UnknownAdapter,
}

impl fmt::Display for MissReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::NoMatch => "no matching token prefix",
            Self::NoCheckpoint => "matched prefix lacks a resumable checkpoint",
            Self::IdentityMismatch => "cache domain identity mismatch",
            Self::Evicted => "matching pages evicted",
            Self::IncompatiblePolicy => "incompatible position/attention or KV policy",
            Self::UnknownAdapter => "unknown adapter",
        };
        f.write_str(s)
    }
}

/// The token-count breakdown of a successful prefix lookup (spec §4.2).
///
/// The four quantities the spec distinguishes are NOT interchangeable:
/// - [`Self::matched_tokens`]: equal token prefix in the index.
/// - [`Self::resident_kv_tokens`]: matching attention rows that still exist.
/// - [`Self::resumable_tokens`]: boundary for which every required state
///   component exists.
/// Only **reused** tokens (forward rows actually skipped, tracked separately
/// by the scheduler) contribute to `cached_tokens`; a prefix match that lacks
/// a recurrent checkpoint is not a hit in usage accounting (spec §4.2).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PrefixLookup {
    /// Equal token prefix length found in the radix index.
    pub matched_tokens: u64,
    /// Matching attention rows that are still physically resident.
    pub resident_kv_tokens: u64,
    /// Largest boundary for which every required state component exists.
    pub resumable_tokens: u64,
}

/// Result of the prefix-lookup operation (spec §3 operation table: "identity
/// + canonical input → longest resumable boundary and pinned handles, or
/// miss; no GPU mutation").
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PrefixLookupResult {
    /// A resumable boundary was found; the caller receives the token counts.
    /// Pinned physical handles are issued by the pool owner, not this type.
    Hit(PrefixLookup),
    /// No resumable boundary; the reason explains why (spec §4.2, §4.5).
    Miss(MissReason),
}

// =========================================================================
// Resume planning — C5 (spec §4.5)
// =========================================================================

/// How the drafter (DFlash/MTP) is brought to the resume boundary (spec
/// §4.5). The drafter's KV and hidden buffers are separate from the target
/// cache; reusing the target prefix is not evidence that the drafter is
/// ready. The decision is made **before execution** (spec §4.5).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum DrafterDecision {
    /// Restore a separately identity-qualified drafter checkpoint.
    Checkpoint,
    /// Rebuild the drafter via the existing reseed path.
    Reseed,
    /// Take the supported autoregressive route instead of speculation.
    Ar,
}

/// Flags naming each required component of a hybrid-state resume bundle
/// (spec §4.5). The Qwen bundle includes all DeltaNet matrices, scales,
/// convolution rings and indices, and error-feedback residuals. A
/// layer-local hit is insufficient; lookup chooses the largest boundary
/// satisfying **all** layer requirements (spec §4.5).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ResumeBundle {
    /// Attention KV pages for the prefix exist and are pinned.
    pub attention_pages: bool,
    /// DeltaNet matrices and scales are available at the boundary.
    pub dn_matrices_scales: bool,
    /// Convolution rings and indices are available at the boundary.
    pub conv_rings: bool,
    /// Error-feedback residual is available at the boundary.
    pub ef_residual: bool,
    /// Drafter readiness decision (checkpoint / reseed / AR).
    pub drafter: DrafterDecision,
}

/// How the last token of an exactly-matching prompt is handled (spec §4.5:
/// "Never run the last token again while leaving recurrent state at
/// S_prompt_len; restore the corresponding earlier state first").
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum LastTokenHandling {
    /// Resume at `p < prompt_len` and process the remaining suffix to obtain
    /// first-token logits (spec §4.5 default).
    SuffixRecompute,
    /// For a prompt that exactly matches cached tokens, select an earlier
    /// valid checkpoint/page boundary and resume there (spec §4.5).
    EarlierBoundary,
}

/// Error raised when a [`ResumePlan`] cannot be constructed because a
/// required state-bundle component is missing (spec §4.5: "cannot return an
/// incomplete state bundle").
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResumePlanError {
    /// A required bundle component was absent. The field names the component.
    MissingComponent(&'static str),
}

impl fmt::Display for ResumePlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingComponent(c) => {
                write!(f, "resume plan missing required state component: {c}")
            }
        }
    }
}

impl std::error::Error for ResumePlanError {}

/// A resume plan for boundary `p` = exactly tokens `[0, p)` processed by the
/// target (spec §4.5). The snapshot at `p` is state `S_p`; it cannot be
/// combined with KV from another boundary.
///
/// Cannot be constructed missing a required component: the constructor
/// validates the [`ResumeBundle`] and returns [`ResumePlanError`] if any
/// required flag is false (spec §4.5: "cannot return an incomplete state
/// bundle").
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ResumePlan {
    /// Boundary `p`: tokens `[0, p)` have been processed by the target.
    pub boundary: u64,
    /// Required state bundle flags (all must be present to construct).
    pub bundle: ResumeBundle,
    /// Byte cost of restoring the bundle into private mutable buffers.
    pub byte_cost: u64,
    /// Last-token handling decision.
    pub last_token: LastTokenHandling,
}

impl ResumePlan {
    /// Construct a resume plan, refusing an incomplete state bundle (spec
    /// §4.5). Every required component flag (`attention_pages`,
    /// `dn_matrices_scales`, `conv_rings`, `ef_residual`) must be true; the
    /// drafter decision is always set (it is an enum, not optional). Returns
    /// [`ResumePlanError::MissingComponent`] naming the first absent
    /// component otherwise.
    pub fn new(
        boundary: u64,
        bundle: ResumeBundle,
        byte_cost: u64,
        last_token: LastTokenHandling,
    ) -> Result<Self, ResumePlanError> {
        if !bundle.attention_pages {
            return Err(ResumePlanError::MissingComponent("attention_pages"));
        }
        if !bundle.dn_matrices_scales {
            return Err(ResumePlanError::MissingComponent("dn_matrices_scales"));
        }
        if !bundle.conv_rings {
            return Err(ResumePlanError::MissingComponent("conv_rings"));
        }
        if !bundle.ef_residual {
            return Err(ResumePlanError::MissingComponent("ef_residual"));
        }
        Ok(Self {
            boundary,
            bundle,
            byte_cost,
            last_token,
        })
    }
}

// =========================================================================
// Step reservation — S1/S2 (spec §5.1, §5.2)
// =========================================================================

/// Scratch, snapshot and output byte needs a step reserves up front (spec
/// §5.1: "reserved temporary workspace"; spec §5.2: "Size ragged row arrays
/// and scratch to the global capacity").
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct StepNeeds {
    /// Worst-case scratch bytes for this step's row count.
    pub scratch_bytes: u64,
    /// Snapshot capture bytes (zero when no checkpoint is captured this step).
    pub snapshot_bytes: u64,
    /// Bounded pending output bytes for this step's emitted tokens.
    pub output_bytes: u64,
}

/// Error raised by [`StepReservation`] arithmetic (spec §5.2: checked
/// arithmetic; spec §5.1: "Use checked arithmetic").
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReservationError {
    /// Summing candidate row counts overflowed `u64`.
    RowOverflow,
}

impl fmt::Display for ReservationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RowOverflow => f.write_str("step reservation row count overflow"),
        }
    }
}

impl std::error::Error for ReservationError {}

/// A candidate step's resource reservation (spec §3 operation table: step
/// reservation; spec §5.1/S1, §5.2/S2).
///
/// Candidate row counts are validated against the global trunk-row budget
/// via [`StepReservation::fits`]. Page growth credits, COW destination
/// count and scratch/snapshot/output needs are reserved before mutation so a
/// failed reservation leaves live request state unchanged (spec §5.4/S4).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct StepReservation {
    /// Prefill rows requested this step.
    pub prefill_rows: u64,
    /// Ordinary decode rows requested this step.
    pub decode_rows: u64,
    /// MTP verify rows (seed/bonus-position work, `k+1`) requested this step.
    pub verify_rows: u64,
    /// Forced-token (jump-forward) rows requested this step.
    pub forced_rows: u64,
    /// Page growth credits reserved for this request's future private suffix.
    pub page_growth_credits: u64,
    /// Number of COW destinations reserved before any KV write (spec §4.3).
    pub cow_destinations: u64,
    /// Scratch/snapshot/output byte needs.
    pub needs: StepNeeds,
}

impl StepReservation {
    /// Sum of all candidate row counts, using checked arithmetic (spec §5.1:
    /// "Use checked arithmetic"). Returns [`ReservationError::RowOverflow`]
    /// rather than wrapping on overflow.
    pub fn total_rows(&self) -> Result<u64, ReservationError> {
        let a = self
            .prefill_rows
            .checked_add(self.decode_rows)
            .ok_or(ReservationError::RowOverflow)?;
        let b = a
            .checked_add(self.verify_rows)
            .ok_or(ReservationError::RowOverflow)?;
        b.checked_add(self.forced_rows)
            .ok_or(ReservationError::RowOverflow)
    }

    /// Whether this reservation fits the global trunk-row budget (spec §5.2:
    /// `sum(prefill_rows + ordinary_decode_rows + verify_rows + forced_rows)
    /// <= max_batch_tokens`). Uses checked arithmetic; overflow is reported
    /// as an error rather than a silent fit/no-fit.
    pub fn fits(&self, max_batch_tokens: u64) -> Result<bool, ReservationError> {
        Ok(self.total_rows()? <= max_batch_tokens)
    }
}

// =========================================================================
// Submit / commit / publish / release (spec §3 operation table, §4.6)
// =========================================================================

/// In-flight completion ticket for a submitted step (spec §3: "reservation +
/// stable descriptor generation → one in-flight completion ticket; all
/// referenced storage outlives completion"). The handle id identifies the
/// completion handle; the generation ties it to the referenced storage
/// generation so stale handles fail before dereference (spec §4.3).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct StepTicket {
    /// In-flight completion handle id.
    pub handle_id: u64,
    /// Generation of the referenced storage; stale tickets mismatch.
    pub generation: u64,
}

/// A committed state boundary after a successful step (spec §3: "successful
/// completion + accepted-token count → committed state boundary; no rejected
/// draft data becomes reusable"; spec §6.1/X1 commit frontier).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CommitBoundary {
    /// Number of committed tokens (the materialized committed prefix).
    pub committed_tokens: u64,
    /// Materialized target rows (distinct from accepted token history, spec
    /// §4.5, §6.1).
    pub materialized_rows: u64,
    /// Optional recurrent checkpoint boundary stamped at this commit, if one
    /// was captured (spec §4.5).
    pub checkpoint: Option<u64>,
}

/// A cache-owned publication lease (spec §3: "immutable committed state +
/// identity + policy → cache-owned leases; failure to cache cannot undo a
/// successful request"; spec §4.6 publication transitions). The lease covers
/// the committed boundary; the issuing owner tracks the full domain identity.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PublishLease {
    /// Lease id assigned by the cache owner.
    pub lease_id: u64,
    /// The committed boundary this lease publishes.
    pub boundary: CommitBoundary,
}

/// Disposition of released leases (spec §3: "leases → cache-resident or
/// reclaim-pending; never immediate reuse while a device or transport reader
/// remains"; spec §4.4 completion and eviction).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ReleaseDisposition {
    /// The released pages/state remain cache-resident (still pinned by the
    /// cache owner).
    CacheResident,
    /// Reclaim is pending device/transport completion; immediate reuse is
    /// forbidden (spec §4.4).
    ReclaimPending,
}

// =========================================================================
// Checkpoint identity — C5 (spec §4.5)
// =========================================================================

/// Opaque identifier for a recurrent-state checkpoint captured at a
/// page-aligned boundary (spec §4.5).  Minted by the family adapter (e.g.
/// the Qwen DeltaNet snapshot pool) as a monotonic `u64` starting at 1.
/// The value `0` is reserved for "no checkpoint" (spec §4.2: a prefix match
/// that lacks a recurrent checkpoint is a miss in usage accounting).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CheckpointId(pub u64);

impl CheckpointId {
    /// Sentinel meaning "no checkpoint exists at this boundary".
    pub const NONE: CheckpointId = CheckpointId(0);

    /// Returns `true` if this is the "no checkpoint" sentinel.
    #[inline]
    pub fn is_none(&self) -> bool {
        self.0 == 0
    }

    /// Returns `true` if a real checkpoint id is present.
    #[inline]
    pub fn is_some(&self) -> bool {
        self.0 != 0
    }
}

impl fmt::Display for CheckpointId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ckpt#{}", self.0)
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_domain(load_epoch: u64) -> CacheDomain {
        CacheDomain {
            model_content_digest: vec![0xab; 16],
            model_load_epoch: load_epoch,
            sidecar_digests: vec![vec![0x01; 8], vec![0x02; 8]],
            tokenizer: TokenizerIdentity {
                vocab_digest: vec![0xcd; 16],
                config_digest: vec![0xce; 16],
            },
            template: TemplateIdentity {
                template_digest: vec![0xee; 16],
                normalization_tag: "chatml".to_owned(),
            },
            arch_policy: ArchPolicy {
                arch_tag: "qwen35-deltanet".to_owned(),
                state_abi_tag: "dn-v1".to_owned(),
                position_attention_tag: "mrope".to_owned(),
            },
            kv_layout: KvLayout {
                k_stride_bytes: vec![128, 128],
                v_stride_bytes: vec![128, 128],
                layout_tag: "q8-g128".to_owned(),
            },
            device: DeviceTopology {
                device_id: "pci-03:00.0".to_owned(),
                topology_id: "single".to_owned(),
                allocation_epoch: 7,
            },
            namespace: SharingNamespace {
                domain_id: "owner-alpha".to_owned(),
            },
        }
    }

    #[test]
    fn cache_domain_canonical_roundtrip_is_identity() {
        let d = sample_domain(42);
        let bytes = d.to_canonical_bytes();
        let back = CacheDomain::from_canonical_bytes(&bytes).expect("roundtrip decodes");
        assert_eq!(d, back);
    }

    #[test]
    fn cache_domain_differing_in_one_field_serializes_differently() {
        let a = sample_domain(1);
        let b = sample_domain(2); // differs only in model_load_epoch
        assert_ne!(a, b);
        assert_ne!(
            a.to_canonical_bytes(),
            b.to_canonical_bytes(),
            "domains differing in one field must serialize differently"
        );
    }

    #[test]
    fn cache_domain_canonical_rejects_truncated_and_trailing() {
        let d = sample_domain(1);
        let bytes = d.to_canonical_bytes();
        // Truncated: drop the last byte. The final string's length prefix
        // then exceeds the remaining content, so either Truncated or Oversized
        // is a correct rejection of a truncated stream.
        let truncated = &bytes[..bytes.len() - 1];
        let err = CacheDomain::from_canonical_bytes(truncated).unwrap_err();
        assert!(
            matches!(err, CanonicalError::Truncated | CanonicalError::Oversized),
            "expected truncation rejection, got {err:?}"
        );
        // Trailing: append an extra byte.
        let mut trailing = bytes.clone();
        trailing.push(0xff);
        match CacheDomain::from_canonical_bytes(&trailing).unwrap_err() {
            CanonicalError::Trailing(_) => {}
            other => panic!("expected Trailing, got {other:?}"),
        }
    }

    #[test]
    fn cache_domain_canonical_rejects_unbounded_collection_count() {
        let mut bytes = sample_domain(1).to_canonical_bytes();
        // version + model digest length + digest + model_load_epoch lands at
        // the sidecar count. Replace it with u64::MAX; decoding must reject
        // before Vec::with_capacity attempts an attacker-sized allocation.
        let digest_len = u64::from_le_bytes(bytes[1..9].try_into().unwrap()) as usize;
        let sidecar_count = 1 + 8 + digest_len + 8;
        bytes[sidecar_count..sidecar_count + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        assert_eq!(
            CacheDomain::from_canonical_bytes(&bytes).unwrap_err(),
            CanonicalError::Oversized
        );
    }

    #[test]
    fn cache_domain_canonical_rejects_unknown_version() {
        let mut bytes = sample_domain(1).to_canonical_bytes();
        bytes[0] = CACHE_DOMAIN_CANONICAL_VERSION.wrapping_add(1);
        assert!(matches!(
            CacheDomain::from_canonical_bytes(&bytes).unwrap_err(),
            CanonicalError::Version(_)
        ));
    }

    fn complete_bundle() -> ResumeBundle {
        ResumeBundle {
            attention_pages: true,
            dn_matrices_scales: true,
            conv_rings: true,
            ef_residual: true,
            drafter: DrafterDecision::Checkpoint,
        }
    }

    #[test]
    fn resume_plan_accepts_complete_bundle() {
        let plan = ResumePlan::new(
            128,
            complete_bundle(),
            4096,
            LastTokenHandling::SuffixRecompute,
        )
        .expect("complete bundle constructs");
        assert_eq!(plan.boundary, 128);
        assert_eq!(plan.bundle.drafter, DrafterDecision::Checkpoint);
    }

    #[test]
    fn resume_plan_refuses_incomplete_bundle() {
        // Missing attention pages.
        let mut b = complete_bundle();
        b.attention_pages = false;
        let err = ResumePlan::new(128, b, 4096, LastTokenHandling::SuffixRecompute).unwrap_err();
        assert!(matches!(
            err,
            ResumePlanError::MissingComponent("attention_pages")
        ));

        // Missing DeltaNet matrices/scales.
        let mut b = complete_bundle();
        b.dn_matrices_scales = false;
        let err = ResumePlan::new(128, b, 4096, LastTokenHandling::SuffixRecompute).unwrap_err();
        assert!(matches!(
            err,
            ResumePlanError::MissingComponent("dn_matrices_scales")
        ));

        // Missing conv rings.
        let mut b = complete_bundle();
        b.conv_rings = false;
        let err = ResumePlan::new(128, b, 4096, LastTokenHandling::EarlierBoundary).unwrap_err();
        assert!(matches!(
            err,
            ResumePlanError::MissingComponent("conv_rings")
        ));

        // Missing EF residual.
        let mut b = complete_bundle();
        b.ef_residual = false;
        let err = ResumePlan::new(128, b, 4096, LastTokenHandling::SuffixRecompute).unwrap_err();
        assert!(matches!(
            err,
            ResumePlanError::MissingComponent("ef_residual")
        ));
    }

    #[test]
    fn step_reservation_fits_within_budget() {
        let r = StepReservation {
            prefill_rows: 100,
            decode_rows: 200,
            verify_rows: 4,
            forced_rows: 0,
            page_growth_credits: 32,
            cow_destinations: 1,
            needs: StepNeeds {
                scratch_bytes: 1 << 20,
                snapshot_bytes: 0,
                output_bytes: 4096,
            },
        };
        assert_eq!(r.total_rows().unwrap(), 304);
        assert!(r.fits(304).unwrap());
        assert!(r.fits(400).unwrap());
        assert!(!r.fits(303).unwrap());
    }

    #[test]
    fn step_reservation_overflow_refuses() {
        let r = StepReservation {
            prefill_rows: u64::MAX,
            decode_rows: 1,
            verify_rows: 0,
            forced_rows: 0,
            page_growth_credits: 0,
            cow_destinations: 0,
            needs: StepNeeds {
                scratch_bytes: 0,
                snapshot_bytes: 0,
                output_bytes: 0,
            },
        };
        assert_eq!(r.total_rows().unwrap_err(), ReservationError::RowOverflow);
        assert_eq!(r.fits(4096).unwrap_err(), ReservationError::RowOverflow);
    }

    #[test]
    fn miss_reason_display_is_nonempty() {
        for reason in [
            MissReason::NoMatch,
            MissReason::NoCheckpoint,
            MissReason::IdentityMismatch,
            MissReason::Evicted,
            MissReason::IncompatiblePolicy,
            MissReason::UnknownAdapter,
        ] {
            assert!(!reason.to_string().is_empty());
        }
    }

    #[test]
    fn prefix_lookup_distinguishes_token_counts() {
        // spec §4.2: matched, resident_kv and resumable are distinct.
        let lk = PrefixLookup {
            matched_tokens: 512,
            resident_kv_tokens: 384,
            resumable_tokens: 256,
        };
        assert!(lk.matched_tokens > lk.resident_kv_tokens);
        assert!(lk.resident_kv_tokens > lk.resumable_tokens);
    }

    #[test]
    fn release_disposition_variants_are_distinct() {
        assert_ne!(
            ReleaseDisposition::CacheResident,
            ReleaseDisposition::ReclaimPending
        );
    }
}
