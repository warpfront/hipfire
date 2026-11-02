// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen4 QSA context storage backend: stable-VA VMM owners with demand-mapped
//! pages, or legacy full-capacity allocations. Load admission resolves the
//! backend ([`qwen4_vmm_refusal`]); state owners only receive it.

use crate::config::Qwen4Config;
use hipfire_runtime::kv_backend::{
    KvBackend, KvChunkPlan, DEFAULT_KV_CHUNK_TOKENS, DEFAULT_VMM_PHYSICAL_CHUNK_BYTES,
};
use rdna_compute::tensor_ops::{qsa_f16_arena_rows, QsaStateLayout};
use rdna_compute::Gpu;

/// Storage of Qwen4 QSA context arenas.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Qwen4KvBackend {
    /// Full admitted-capacity physical allocations, zeroed at construction.
    Legacy,
    /// Registered VMM owners reserving the admitted capacity; pages are
    /// mapped on demand.
    Vmm,
}

impl Qwen4KvBackend {
    /// The automatic selection: VMM wherever [`qwen4_vmm_supported`].
    pub fn automatic(gpu: &Gpu) -> Self {
        if qwen4_vmm_supported(gpu) {
            Self::Vmm
        } else {
            Self::Legacy
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Legacy => "legacy",
            Self::Vmm => "vmm",
        }
    }
}

impl From<KvBackend> for Qwen4KvBackend {
    fn from(backend: KvBackend) -> Self {
        match backend {
            KvBackend::Legacy => Self::Legacy,
            KvBackend::Vmm => Self::Vmm,
        }
    }
}

/// The architectures whose Qwen4 QSA VMM state is certified unconditionally:
/// gfx1151 (F32 state) and gfx1201 (fp8 state). gfx1100 is admitted too
/// unless the `HIPFIRE_QWEN4_VMM_GFX1100=0` kill switch is set.
const VMM_ARCHS: [&str; 2] = ["gfx1151", "gfx1201"];

/// Why Qwen4 QSA context arenas on a `gpu_arch` device cannot be VMM owners,
/// or `None` when they can: exactly gfx1151, gfx1201 or (unless
/// `HIPFIRE_QWEN4_VMM_GFX1100=0`) gfx1100, not Windows (whose driver needs the
/// full-map workaround), the platform VMM KV refusal
/// ([`hipfire_config::devices::vmm_kv_platform_refusal`]) clear, and a HIP
/// VMM runtime that reports an allocation granularity
/// (`vmm_runtime_available`). Load admission resolves the backend from this.
pub fn qwen4_vmm_refusal(gpu_arch: &str, vmm_runtime_available: bool) -> Option<String> {
    qwen4_vmm_refusal_with(
        gpu_arch,
        vmm_runtime_available,
        hipfire_config::devices::qwen4_vmm_gfx1100_enabled(),
    )
}

/// [`qwen4_vmm_refusal`] with the gfx1100 kill-switch state explicit
/// (`gfx1100_enabled` is false when `HIPFIRE_QWEN4_VMM_GFX1100=0`).
fn qwen4_vmm_refusal_with(
    gpu_arch: &str,
    vmm_runtime_available: bool,
    gfx1100_enabled: bool,
) -> Option<String> {
    if cfg!(windows) {
        Some("Windows maps whole VMM reservations; Qwen4 QSA keeps legacy storage".to_string())
    } else if gpu_arch == "gfx1100" && !gfx1100_enabled {
        Some(format!(
            "Qwen4 VMM QSA state on gfx1100 is disabled by {}=0",
            hipfire_config::devices::QWEN4_VMM_GFX1100
        ))
    } else if !VMM_ARCHS.contains(&gpu_arch) && gpu_arch != "gfx1100" {
        Some(format!(
            "Qwen4 VMM QSA state is certified only on {}, gfx1100, not {gpu_arch}",
            VMM_ARCHS.join(", ")
        ))
    } else if let Some(reason) = hipfire_config::devices::vmm_kv_platform_refusal() {
        Some(reason)
    } else if !vmm_runtime_available {
        Some("HIP VMM symbols/granularity unavailable".to_string())
    } else {
        None
    }
}

/// Capability check only: whether Qwen4 QSA context arenas on `gpu` may be
/// VMM owners ([`qwen4_vmm_refusal`] on this device).
pub fn qwen4_vmm_supported(gpu: &Gpu) -> bool {
    qwen4_vmm_refusal(&gpu.arch, gpu.vmm_recommended_granularity().is_ok()).is_none()
}

/// What a load's QSA context arenas commit, for the VRAM reserve and the
/// load census. Every arena reserves `max_seq` tokens of address space
/// (virtual); the committed part is all of it for legacy storage (allocated
/// at construction) and, for VMM owners, the pages the first forward chunk of
/// `tokens` rows maps (the shared KV chunk plan at `granularity`). Later
/// growth maps more at forward time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Qwen4ContextCommit {
    pub backend: Qwen4KvBackend,
    /// Admitted (logical) context: every arena's virtual extent.
    pub max_seq: usize,
    /// Tokens charged as committed.
    pub tokens: usize,
    /// VMM driver page granularity (unused by legacy storage).
    pub granularity: usize,
}

impl Qwen4ContextCommit {
    /// Legacy storage of `max_seq` tokens.
    pub fn legacy(max_seq: usize) -> Self {
        Self {
            backend: Qwen4KvBackend::Legacy,
            max_seq,
            tokens: max_seq,
            granularity: 1,
        }
    }

    /// `backend` storage of `max_seq` tokens whose first forward runs a
    /// `chunk_rows`-row prefill chunk.
    pub fn new(
        backend: Qwen4KvBackend,
        max_seq: usize,
        chunk_rows: usize,
        granularity: usize,
    ) -> Self {
        match backend {
            Qwen4KvBackend::Legacy => Self::legacy(max_seq),
            Qwen4KvBackend::Vmm => Self {
                backend,
                max_seq,
                tokens: chunk_rows.min(max_seq),
                granularity,
            },
        }
    }

    /// What the `auto` expert-placement reserve charges for a load's context
    /// arenas. Unified memory charges the arenas as committed ([`Self::new`]:
    /// VMM pages commit on demand out of the same pool the weights live in).
    /// A discrete card charges legacy storage of `max_seq` tokens whatever
    /// the backend: VMM growth during forwards maps VRAM that the placement
    /// must already have left free, so the reserve, and with it the expert
    /// layers placed, match legacy storage at every `max_seq`.
    pub fn for_expert_reserve(
        backend: Qwen4KvBackend,
        unified_memory: bool,
        max_seq: usize,
        chunk_rows: usize,
        granularity: usize,
    ) -> Self {
        if unified_memory {
            Self::new(backend, max_seq, chunk_rows, granularity)
        } else {
            Self::legacy(max_seq)
        }
    }

    /// Virtual bytes of one layer's context arenas in `format`.
    pub fn virtual_layer_bytes(
        &self,
        config: &Qwen4Config,
        format: impl Into<QsaStateLayout>,
    ) -> Option<usize> {
        config.qsa_context_arena_bytes(self.max_seq, format)
    }

    /// Committed bytes of one layer's context arenas in `format`: the whole
    /// allocation (legacy) or each arena's mapped growth pages covering
    /// [`Self::tokens`] (VMM; pooled keys by `ceil(tokens / compress)` rows),
    /// as `ensure_mapped_capacity` maps them.
    pub fn committed_layer_bytes(
        &self,
        config: &Qwen4Config,
        layout: impl Into<QsaStateLayout>,
    ) -> Option<usize> {
        let layout = layout.into();
        let format = layout.kv;
        if self.backend == Qwen4KvBackend::Legacy {
            return self.virtual_layer_bytes(config, layout);
        }
        let kv_row = format.kv_row_bytes(config.num_key_value_heads, config.head_dim);
        let raw = config
            .indexer_kv_heads
            .checked_mul(config.indexer_head_dim)?;
        let index_row = raw.checked_mul(format.index_dtype().size())?;
        let compress = config.indexer_compress_ratio;
        let pooled_capacity = self.max_seq.div_ceil(compress);
        let pooled_tokens = self.tokens.div_ceil(compress);
        let mapped = |row_bytes: usize, rows: usize, tokens: usize| {
            KvChunkPlan::new(
                row_bytes,
                rows,
                DEFAULT_KV_CHUNK_TOKENS,
                self.granularity,
                DEFAULT_VMM_PHYSICAL_CHUNK_BYTES,
            )
            .and_then(|plan| plan.mapped_bytes_for_tokens(tokens))
            .ok()
        };
        mapped(kv_row, self.max_seq, self.tokens)?
            .checked_mul(2)?
            .checked_add(mapped(index_row, self.max_seq, self.tokens)?)?
            .checked_add(mapped(index_row, pooled_capacity, pooled_tokens)?)?
            // The pooled BF16 mirror grows with the pooled keys.
            .checked_add(if layout.pooled_bf16 {
                mapped(
                    layout.pooled_bf16_bytes_per_block(raw),
                    pooled_capacity,
                    pooled_tokens,
                )?
            } else {
                0
            })?
            .checked_add(if layout.f16_arena {
                let row = config
                    .num_key_value_heads
                    .checked_mul(config.head_dim)?
                    .checked_mul(2)?;
                mapped(
                    row,
                    qsa_f16_arena_rows(self.max_seq),
                    qsa_f16_arena_rows(self.tokens),
                )?
                .checked_mul(2)?
            } else {
                0
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rdna_compute::tensor_ops::QsaKvFormat;

    #[test]
    fn vmm_is_refused_off_certified_arches_and_without_the_runtime() {
        // gfx1100: admitted by default, refused (naming the switch) when
        // HIPFIRE_QWEN4_VMM_GFX1100=0.
        let off = qwen4_vmm_refusal_with("gfx1100", true, false);
        assert!(off.is_some_and(|reason| {
            reason.contains("HIPFIRE_QWEN4_VMM_GFX1100=0") && reason.contains("gfx1100")
        }));
        // Other arches are unaffected by the gfx1100 switch.
        for gfx1100_enabled in [false, true] {
            assert!(
                qwen4_vmm_refusal_with("gfx1200", true, gfx1100_enabled).is_some_and(
                    |reason| reason.contains("gfx1151")
                        && reason.contains("gfx1201")
                        && reason.contains("gfx1100")
                        && reason.contains("gfx1200")
                )
            );
        }
        for arch in ["gfx1100", "gfx1151", "gfx1201"] {
            if cfg!(windows) || hipfire_config::devices::vmm_kv_platform_refusal().is_some() {
                assert!(qwen4_vmm_refusal_with(arch, true, true).is_some());
            } else {
                assert_eq!(qwen4_vmm_refusal_with(arch, true, true), None);
                assert!(qwen4_vmm_refusal_with(arch, false, true)
                    .is_some_and(|reason| reason.contains("granularity")));
            }
        }
        // The process-reading wrapper agrees with the explicit form (the
        // test process leaves the switch unset, i.e. enabled).
        assert_eq!(
            qwen4_vmm_refusal("gfx1100", true),
            qwen4_vmm_refusal_with(
                "gfx1100",
                true,
                hipfire_config::devices::qwen4_vmm_gfx1100_enabled()
            )
        );
    }

    #[test]
    fn vmm_commit_charges_the_first_chunk_not_the_logical_context() {
        let config = crate::config::compact_test_config();
        let format = QsaKvFormat::F32;
        let max_seq = 262_144;
        let legacy = Qwen4ContextCommit::new(Qwen4KvBackend::Legacy, max_seq, 8192, 1 << 21);
        let full = config.qsa_context_arena_bytes(max_seq, format).unwrap();
        assert_eq!(legacy.tokens, max_seq);
        assert_eq!(legacy.committed_layer_bytes(&config, format), Some(full));

        let vmm = Qwen4ContextCommit::new(Qwen4KvBackend::Vmm, max_seq, 8192, 1 << 21);
        assert_eq!(vmm.virtual_layer_bytes(&config, format), Some(full));
        let committed = vmm.committed_layer_bytes(&config, format).unwrap();
        let touched = config.qsa_context_arena_bytes(8192, format).unwrap();
        // Whole growth pages covering the chunk: at least the touched rows,
        // at most one extra 2 MiB growth step per arena.
        assert!(committed >= touched && committed <= touched + 4 * (2 << 20));
        assert!(committed < full / 16);
        // The chunk never charges past the admitted context.
        let short = Qwen4ContextCommit::new(Qwen4KvBackend::Vmm, 100, 8192, 1 << 21);
        assert_eq!(short.tokens, 100);
    }

    #[test]
    fn pooled_bf16_mirror_grows_by_ceil_blocks_at_nonmultiple_depth() {
        let config = crate::config::compact_test_config();
        let layout = QsaStateLayout {
            pooled_bf16: true,
            ..QsaStateLayout::F32
        };
        // 32_769 tokens: 8193 pooled blocks (ceil), not 8192 (floor). At 256
        // bytes per block that is 2 MiB + 256 B, one growth page past the
        // exactly-2-MiB floor mapping.
        let tokens = 32_769usize;
        let vmm = Qwen4ContextCommit {
            backend: Qwen4KvBackend::Vmm,
            max_seq: 262_144,
            tokens,
            granularity: 1 << 21,
        };
        let delta = vmm.committed_layer_bytes(&config, layout).unwrap()
            - vmm
                .committed_layer_bytes(&config, QsaKvFormat::F32)
                .unwrap();
        assert_eq!(delta, 2 * (2 << 20));
        // Legacy and the logical size count the full ceil(max_seq / compress).
        let logical = config.qsa_context_arena_bytes(tokens, layout).unwrap()
            - config
                .qsa_context_arena_bytes(tokens, QsaKvFormat::F32)
                .unwrap();
        assert_eq!(logical, 8193 * 256);
    }
}
