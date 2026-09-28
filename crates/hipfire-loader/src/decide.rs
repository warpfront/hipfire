// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Model-state snapshot for the decide runner: the KV cursor plus, for
//! hybrid Qwen3.5/3.6, the GatedDeltaNet recurrent state. KV rewind is
//! positional (valid only without eviction/compaction — the runner checks).

use hipfire_arch_qwen35::speculative::DeltaNetSnapshot;

pub struct DecideSnapshot {
    pub seq_pos: usize,
    pub recurrent: Option<DeltaNetSnapshot>,
}

impl DecideSnapshot {
    /// `DeltaNetSnapshot` has no `Drop`; this is the only release path.
    pub fn free(self, gpu: &mut rdna_compute::Gpu) {
        if let Some(r) = self.recurrent {
            r.free_gpu(gpu);
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn only_qwen35_and_llama_carriers_claim_decide() {
        // Default hooks return None without touching the model, so we can
        // probe support through a trait-level flag instead of a GPU.
        for c in crate::registry_carriers() {
            let expected = matches!(c.name(), "qwen35" | "llama");
            assert_eq!(c.decide_supported(), expected, "carrier {}", c.name());
        }
    }
}
