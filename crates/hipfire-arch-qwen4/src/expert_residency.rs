// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Routed-expert residency for cards whose VRAM cannot hold every expert.
//!
//! Flash-Next's routed experts are ~65 GB, the rest of the model ~5 GB. On a
//! discrete card the experts of trunk layers at or past the VRAM layer count
//! ([`EXPERT_VRAM_LAYERS_ENV`]) and the MTP layer's are fulfilled into pinned,
//! device-mapped host RAM instead of VRAM. The sealed MoE kernels already read
//! every expert through per-layer pointer tables, so a host-mapped layer is
//! read over PCIe (zero-copy) with no kernel or dispatch change: a decode
//! token reads only its routed experts' bytes from host RAM.

use crate::Qwen4Config;
use hipfire_runtime::weight_manifest::{ShardPolicy, WeightEntry, WeightResidency};

/// `N` keeps the routed experts of trunk layers `0..N` in VRAM; `auto` picks
/// the largest `N` that fits the card's free VRAM. Unset keeps every expert
/// resident where they fit ([`resolve_expert_vram_layers`]) and is `auto`
/// otherwise. Set, it also keeps host memory out of reclaim from before the
/// HIP runtime loads (hip-bridge owns the name), as a process about to load
/// a Qwen4 model on a discrete GPU does unset
/// ([`keeps_host_memory_out_of_reclaim`]).
pub const EXPERT_VRAM_LAYERS_ENV: &str = hip_bridge::QWEN4_EXPERT_VRAM_LAYERS_ENV;

/// VRAM left free by `auto` beyond the resident non-expert weights: forward
/// scratch, KV and state for [`AUTO_VRAM_RESERVE_MAX_SEQ`] tokens plus
/// headroom, without native MTP. Measured on a gfx1201 R9700 at N=16 with a
/// [`AUTO_VRAM_RESERVE_CHUNK`]-row prefill chunk: every context-sized buffer
/// is allocated at load, and a 32,700-token prefill plus decode at that
/// context peaked at 6,199 MiB beyond the non-expert and expert weights,
/// 457 MiB under this reserve. [`auto_vram_reserve`] adds what a longer
/// context, a different chunk and native MTP take; a shorter context keeps it.
pub const AUTO_VRAM_RESERVE_BYTES: u64 = 6656 << 20;

/// Context [`AUTO_VRAM_RESERVE_BYTES`] was measured at (`hipfire run`'s
/// legacy-KV `max_seq`).
pub const AUTO_VRAM_RESERVE_MAX_SEQ: usize = 32768;

/// Prefill chunk [`AUTO_VRAM_RESERVE_BYTES`] was measured at; its forward
/// scratch then carried speculative logits and argmax for every chunk row.
pub const AUTO_VRAM_RESERVE_CHUNK: usize = 1536;

/// Host RAM that must remain available after the pinned experts are placed.
pub const HOST_RAM_HEADROOM_BYTES: u64 = 4 << 30;

/// Slack `Gpu::upload_raw_host_mapped` adds to every host-mapped tensor.
pub const HOST_MAPPED_PAD_BYTES: u64 = 1 << 20;

const GIB: f64 = (1u64 << 30) as f64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpertVramLayers {
    Layers(usize),
    Auto,
}

/// Parse [`EXPERT_VRAM_LAYERS_ENV`]. `Ok(None)` = unset
/// ([`resolve_expert_vram_layers`] picks the placement).
pub fn expert_vram_layers_from_env() -> Result<Option<ExpertVramLayers>, String> {
    match hipfire_config::developer_var(EXPERT_VRAM_LAYERS_ENV) {
        Ok(value) => parse_expert_vram_layers(&value).map(Some),
        Err(_) => Ok(None),
    }
}

/// The placement for a discrete-GPU load: an explicit `policy` always wins.
/// Unset (`None`) keeps every routed expert resident (`Ok(None)`) when
/// `fits_resident` reports that the whole expert footprint, the non-expert
/// weights and the `auto` reserve fit in VRAM, and is `auto` otherwise.
/// `fits_resident` runs only when unset.
pub fn resolve_expert_vram_layers(
    policy: Option<ExpertVramLayers>,
    fits_resident: impl FnOnce() -> Result<bool, String>,
) -> Result<Option<ExpertVramLayers>, String> {
    match policy {
        Some(policy) => Ok(Some(policy)),
        None => Ok((!fits_resident()?).then_some(ExpertVramLayers::Auto)),
    }
}

/// Whether every routed expert (`expert_bytes`), the non-expert weights and
/// `reserve` ([`auto_vram_reserve`]) fit in `free_vram`.
pub fn fits_fully_resident(free_vram: u64, non_expert_bytes: u64, expert_bytes: u64, reserve: u64) -> bool {
    non_expert_bytes
        .checked_add(expert_bytes)
        .and_then(|bytes| bytes.checked_add(reserve))
        .is_some_and(|need| need <= free_vram)
}

/// Whether a process about to load the model with HFQ `arch_id` on the GPUs
/// `device_archs` (every card it may use) keeps host memory out of reclaim
/// before the HIP runtime loads (`hip_bridge::keep_host_memory_out_of_reclaim`),
/// as an explicit [`EXPERT_VRAM_LAYERS_ENV`] does: a Qwen4 model on discrete
/// GPUs. It rests on those static facts, not on the placement the load
/// resolves: unset, that depends on the free VRAM and the reserve at load
/// time, after the runtime has read the switches, and the upload of tens of
/// GB out of the mapped file needs them under host-memory pressure whatever
/// the placement. Unified memory never host-maps experts, and an
/// unrecognized arch keeps ROCm's defaults.
pub fn keeps_host_memory_out_of_reclaim(arch_id: u32, device_archs: &[&str]) -> bool {
    arch_id == crate::ARCH_ID
        && !device_archs.is_empty()
        && device_archs.iter().all(|arch| hipfire_config::is_discrete_memory_arch(arch))
}

/// Bytes of every routed expert payload (trunk and MTP layers).
pub fn routed_expert_bytes(
    weights: &[WeightEntry],
    bytes_of: impl Fn(&WeightEntry) -> Option<u64>,
) -> Result<u64, String> {
    weights
        .iter()
        .filter(|entry| {
            is_routed_expert(&entry.name)
                && !entry.residency.is_external()
                && !matches!(entry.policy, ShardPolicy::Tied { .. })
        })
        .try_fold(0u64, |total, entry| {
            bytes_of(entry)
                .map(|bytes| total + bytes)
                .ok_or_else(|| format!("no payload size for '{}'", entry.name))
        })
}

fn parse_expert_vram_layers(value: &str) -> Result<ExpertVramLayers, String> {
    let value = value.trim();
    if value == "auto" {
        return Ok(ExpertVramLayers::Auto);
    }
    value
        .parse::<usize>()
        .map(ExpertVramLayers::Layers)
        .map_err(|_| format!("{EXPERT_VRAM_LAYERS_ENV}={value:?} is neither `auto` nor a layer count"))
}

fn is_routed_expert(name: &str) -> bool {
    name.ends_with(".mlp.experts.gate_up_proj") || name.ends_with(".mlp.experts.down_proj")
}

/// Resident bytes outside the routed experts, and the routed-expert bytes of
/// one trunk layer (layer 0), from each entry's payload size. Tied aliases
/// have no payload of their own and external rows stay in the file.
pub fn resident_split(
    weights: &[WeightEntry],
    bytes_of: impl Fn(&WeightEntry) -> Option<u64>,
) -> Result<(u64, u64), String> {
    let mut non_expert = 0u64;
    let mut layer_experts = 0u64;
    for entry in weights {
        if entry.residency.is_external() || matches!(entry.policy, ShardPolicy::Tied { .. }) {
            continue;
        }
        let bytes = bytes_of(entry).ok_or_else(|| format!("no payload size for '{}'", entry.name))?;
        if !is_routed_expert(&entry.name) {
            non_expert += bytes;
        } else if !entry.name.starts_with("mtp.") && entry.layer == Some(0) {
            layer_experts += bytes;
        }
    }
    Ok((non_expert, layer_experts))
}

/// Stored dtype of the language head, which the native MTP draft head ranks
/// with.
pub fn language_head_dtype(weights: &[WeightEntry]) -> Option<rdna_compute::DType> {
    weights
        .iter()
        .find(|entry| entry.name == crate::weights::QWEN4_LM_HEAD)
        .map(|entry| entry.dtype)
}

/// VRAM `auto` leaves free beyond the non-expert weights for a load at
/// `max_seq` with a `chunk_rows` prefill chunk: [`AUTO_VRAM_RESERVE_BYTES`],
/// the trunk QSA arenas' growth in `qsa_format` past
/// [`AUTO_VRAM_RESERVE_MAX_SEQ`], the chunk-sized forward resources' change
/// from the measured [`AUTO_VRAM_RESERVE_CHUNK`] layout, `mtp_bytes` when a
/// native MTP speculator attaches (`mtp_spec::native_mtp_device_bytes`), and
/// `gather_bytes` when the gathered QSA prefill attention reserves its
/// context-sized scratch at load
/// (`rdna_compute::tensor_ops::qsa_gathered_wmma_scratch_bytes`). G2's
/// opt-in full-layer staging adds two expert layers' worth of device bytes.
pub fn auto_vram_reserve(
    config: &Qwen4Config,
    max_seq: usize,
    chunk_rows: usize,
    qsa_format: rdna_compute::tensor_ops::QsaKvFormat,
    mtp_bytes: Option<u64>,
    gather_bytes: Option<u64>,
) -> Result<u64, String> {
    let arena = |seq| {
        config
            .qsa_context_arena_bytes(seq, qsa_format)
            .and_then(|bytes| bytes.checked_mul(config.n_full_layers()))
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| format!("QSA context state for max_seq {seq} overflows"))
    };
    let context = arena(max_seq)?.saturating_sub(arena(AUTO_VRAM_RESERVE_MAX_SEQ)?);
    let overflow = || format!("forward resources for a {chunk_rows}-row chunk overflow");
    let measured =
        crate::gpu_forward::Qwen4GpuForwardScratch::device_bytes(config, AUTO_VRAM_RESERVE_CHUNK)
            .ok()
            .and_then(|scratch| {
                let spec =
                    AUTO_VRAM_RESERVE_CHUNK.checked_mul(config.vocab_size.checked_mul(4)? + 4)?;
                scratch.checked_add(spec as u64)
            })
            .ok_or_else(overflow)?;
    let forward =
        crate::gpu_forward::qwen4_forward_device_bytes(config, chunk_rows).ok_or_else(overflow)?;
    let stage_bytes = if rdna_compute::gemm::qwen4_expert_stage_requested() {
        rdna_compute::gemm::QWEN4_EXPERT_STAGE_BYTES
    } else {
        0
    };
    AUTO_VRAM_RESERVE_BYTES
        .checked_add(context)
        .and_then(|bytes| bytes.checked_add(forward))
        .and_then(|bytes| bytes.checked_sub(measured))
        .and_then(|bytes| bytes.checked_add(mtp_bytes.unwrap_or(0)))
        .and_then(|bytes| bytes.checked_add(gather_bytes.unwrap_or(0)))
        .and_then(|bytes| bytes.checked_add(stage_bytes))
        .ok_or_else(|| "auto expert VRAM reserve overflows".to_string())
}

/// The largest trunk-layer count whose routed experts fit in `free_vram`
/// after the non-expert weights and `reserve` ([`auto_vram_reserve`]).
pub fn auto_vram_layers(
    free_vram: u64,
    non_expert_bytes: u64,
    layer_expert_bytes: u64,
    num_layers: usize,
    reserve: u64,
) -> usize {
    let budget = free_vram
        .saturating_sub(non_expert_bytes)
        .saturating_sub(reserve);
    match budget.checked_div(layer_expert_bytes) {
        Some(layers) => (layers as usize).min(num_layers),
        None => num_layers,
    }
}

/// Mark the routed-expert entries of trunk layers `>= vram_layers`, and of
/// the MTP layer, [`WeightResidency::HostMapped`]. Returns the number of
/// entries moved to host RAM.
pub fn place_routed_experts(weights: &mut [WeightEntry], vram_layers: usize) -> usize {
    let mut moved = 0;
    for entry in weights.iter_mut() {
        if entry.residency != WeightResidency::Resident || !is_routed_expert(&entry.name) {
            continue;
        }
        let trunk_resident = !entry.name.starts_with("mtp.")
            && entry.layer.is_some_and(|layer| layer < vram_layers);
        if !trunk_resident {
            entry.residency = WeightResidency::HostMapped;
            moved += 1;
        }
    }
    moved
}

/// Pinned host bytes the [`WeightResidency::HostMapped`] entries will take.
pub fn host_mapped_bytes(
    weights: &[WeightEntry],
    bytes_of: impl Fn(&WeightEntry) -> Option<u64>,
) -> Result<u64, String> {
    let mut total = 0u64;
    for entry in weights {
        if entry.residency == WeightResidency::HostMapped {
            let bytes =
                bytes_of(entry).ok_or_else(|| format!("no payload size for '{}'", entry.name))?;
            total += bytes + HOST_MAPPED_PAD_BYTES;
        }
    }
    Ok(total)
}

/// Refuse before any allocation when the pinned experts would not leave
/// [`HOST_RAM_HEADROOM_BYTES`] of the host RAM they can take: `MemAvailable`
/// plus `ttm_pool_bytes`, the estimate of freed GTT pages parked in TTM's
/// page pool ([`ttm_pool_estimate`]). Pinned pages cannot be reclaimed, so
/// over-committing them starves the rest of the host.
pub fn check_host_ram(
    host_bytes: u64,
    mem_available: Option<u64>,
    ttm_pool_bytes: u64,
) -> Result<(), String> {
    if host_bytes == 0 {
        return Ok(());
    }
    let Some(available) = mem_available else {
        return Err(format!(
            "routed experts need {:.1} GiB of pinned host RAM, but MemAvailable is unreadable",
            host_bytes as f64 / GIB
        ));
    };
    let needed = host_bytes + HOST_RAM_HEADROOM_BYTES;
    if available.saturating_add(ttm_pool_bytes) < needed {
        return Err(format!(
            "routed experts need {:.1} GiB of pinned host RAM plus {:.0} GiB headroom, but \
             MemAvailable is {:.1} GiB (plus {:.1} GiB estimated in TTM's page pool); free host \
             memory or keep more expert layers in VRAM ({EXPERT_VRAM_LAYERS_ENV})",
            host_bytes as f64 / GIB,
            HOST_RAM_HEADROOM_BYTES as f64 / GIB,
            available as f64 / GIB,
            ttm_pool_bytes as f64 / GIB
        ));
    }
    Ok(())
}

/// TTM's page limit, in pages, which caps every GTT allocation on the host.
const TTM_PAGES_LIMIT: &str = "/sys/module/ttm/parameters/pages_limit";

/// TTM's page pool cap, in pages. Freed GTT pages past it go back to the
/// kernel at once.
const TTM_PAGE_POOL_SIZE: &str = "/sys/module/ttm/parameters/page_pool_size";

/// TTM's page size on x86_64, the only host ROCm supports for discrete GPUs.
const TTM_PAGE_BYTES: u64 = 4096;

/// Host memory outside every `/proc/meminfo` counter that is not TTM's pool:
/// other drivers' pages, DMA buffers, firmware. [`ttm_pool_estimate`] never
/// counts this much as pool. Measured on the 5-card gfx1201 host with the
/// pool empty and 46.2 GiB of live GTT: 2.76 GiB.
pub const UNTRACKED_KERNEL_BYTES: u64 = 4 << 30;

/// `/proc/meminfo` fields, in bytes, that together account for every
/// allocated page except driver pages (TTM's pool and live GTT among them).
/// Subset fields (`Shmem`, `Mlocked`, `AnonHugePages`, ...) are left out.
const MEMINFO_TRACKED: &[&str] = &[
    "MemFree",
    "Buffers",
    "Cached",
    "SwapCached",
    "AnonPages",
    "Slab",
    "KernelStack",
    "ShadowCallStack",
    "PageTables",
    "SecPageTables",
    "VmallocUsed",
    "Percpu",
    "Hugetlb",
    "Zswap",
    "Unaccepted",
    "Balloon",
];

fn host_mapped_is_gtt() -> bool {
    std::env::var("HSA_USERPTR_FOR_PAGED_MEM").is_ok_and(|value| value.trim() == "0")
}

fn read_u64(path: &std::path::Path) -> Option<u64> {
    std::fs::read_to_string(path).ok()?.trim().parse().ok()
}

/// `mem_info_gtt_used` summed over every amdgpu device. Host allocations may
/// count against another device than the one the process runs on (on a
/// 5-card gfx1201 host, a card-2 process's host-mapped experts show in card
/// 0's `mem_info_gtt_used`).
fn amdgpu_gtt_used_bytes() -> Option<u64> {
    let mut used_bytes = 0u64;
    for card in std::fs::read_dir("/sys/class/drm").ok()?.flatten() {
        let name = card.file_name();
        let is_card = name
            .to_str()
            .and_then(|name| name.strip_prefix("card"))
            .is_some_and(|index| !index.is_empty() && index.bytes().all(|b| b.is_ascii_digit()));
        if is_card {
            used_bytes += read_u64(&card.path().join("device/mem_info_gtt_used")).unwrap_or(0);
        }
    }
    Some(used_bytes)
}

/// Estimate of the freed GTT pages TTM keeps in its page pool, from
/// `/proc/meminfo` text, the GTT amdgpu devices hold, and TTM's
/// `page_pool_size` in pages.
///
/// The pool is invisible to an unprivileged process: its pages are in no
/// `/proc/meminfo` counter, so `MemAvailable` excludes them, and
/// `mem_info_gtt_used` drops them when their buffer is freed (only root's
/// `/sys/kernel/debug/ttm/page_pool` reads it). It is `MemTotal` minus every
/// tracked counter, minus the live GTT and [`UNTRACKED_KERNEL_BYTES`], capped
/// at `page_pool_size`. `None` when a field is missing.
pub fn ttm_pool_estimate_from(meminfo: &str, gtt_used: u64, pool_size_pages: u64) -> Option<u64> {
    let field = |key: &str| -> Option<u64> {
        meminfo.lines().find_map(|line| {
            let rest = line.strip_prefix(key)?.strip_prefix(':')?;
            Some(rest.split_whitespace().next()?.parse::<u64>().ok()? * 1024)
        })
    };
    let total = field("MemTotal")?;
    field("MemFree")?;
    let tracked: u64 = MEMINFO_TRACKED.iter().filter_map(|key| field(key)).sum::<u64>()
        + field("KReclaimable")?.saturating_sub(field("SReclaimable")?);
    let untracked = total
        .saturating_sub(tracked)
        .saturating_sub(gtt_used)
        .saturating_sub(UNTRACKED_KERNEL_BYTES);
    Some(untracked.min(pool_size_pages.saturating_mul(TTM_PAGE_BYTES)))
}

/// Host RAM held in TTM's page pool that the host-mapped experts can take.
/// amdgpu parks the write-combined and uncached pages of freed GTT buffers
/// there, up to `page_pool_size` (half of RAM by default), so a Flash-Next
/// process that exits leaves its host-mapped experts' pages in it. A new
/// GTT allocation takes pages from the pool first, and TTM's shrinker frees
/// it under memory pressure (`echo 2 | sudo tee /proc/sys/vm/drop_caches`
/// empties it at once). 0 unless host-mapped memory is GTT-backed
/// (`HSA_USERPTR_FOR_PAGED_MEM=0`), or when sysfs or `/proc/meminfo` is
/// unreadable.
pub fn ttm_pool_estimate() -> u64 {
    if !host_mapped_is_gtt() {
        return 0;
    }
    let estimate = || -> Option<u64> {
        let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
        let pool_size_pages = read_u64(TTM_PAGE_POOL_SIZE.as_ref())?;
        ttm_pool_estimate_from(&meminfo, amdgpu_gtt_used_bytes()?, pool_size_pages)
    };
    estimate().unwrap_or(0)
}

/// GTT room on the host: TTM's cap and what amdgpu devices already hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GttBudget {
    pub limit_bytes: u64,
    pub used_bytes: u64,
}

/// The GTT budget when host-mapped memory is GTT-backed, i.e. when
/// `HSA_USERPTR_FOR_PAGED_MEM` is `0` (hip-bridge's default). It is `None`
/// under userptr, or when sysfs does not expose TTM's limit. `used_bytes`
/// sums `mem_info_gtt_used` over every amdgpu device.
pub fn gtt_budget() -> Option<GttBudget> {
    if !host_mapped_is_gtt() {
        return None;
    }
    let limit_bytes = read_u64(TTM_PAGES_LIMIT.as_ref())?.checked_mul(TTM_PAGE_BYTES)?;
    Some(GttBudget { limit_bytes, used_bytes: amdgpu_gtt_used_bytes()? })
}

/// Refuse before any allocation when GTT-backed host-mapped experts would not
/// fit under TTM's `pages_limit` beside what amdgpu devices already hold.
/// Past it `hipHostMalloc` fails, after the load has uploaded the VRAM
/// weights. `None` (userptr, or no TTM limit in sysfs) skips the check.
pub fn check_gtt_cap(host_bytes: u64, budget: Option<GttBudget>) -> Result<(), String> {
    let Some(GttBudget { limit_bytes, used_bytes }) = budget else {
        return Ok(());
    };
    let free = limit_bytes.saturating_sub(used_bytes);
    if host_bytes > free {
        return Err(format!(
            "routed experts need {:.1} GiB of GTT-backed host RAM, but TTM's GTT cap \
             ({TTM_PAGES_LIMIT} = {} pages, {:.1} GiB) has {:.1} GiB left beside the {:.1} GiB \
             amdgpu devices already hold; keep more expert layers in VRAM \
             ({EXPERT_VRAM_LAYERS_ENV}), raise ttm.pages_limit, or set \
             HSA_USERPTR_FOR_PAGED_MEM=1 for pageable userptr host memory, which host-memory \
             pressure can stall (ROCm/rocm-systems#12528)",
            host_bytes as f64 / GIB,
            limit_bytes / TTM_PAGE_BYTES,
            limit_bytes as f64 / GIB,
            free as f64 / GIB,
            used_bytes as f64 / GIB
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_runtime::weight_manifest::ShardPolicy;
    use rdna_compute::tensor_ops::QsaKvFormat::{self, F32};
    use rdna_compute::DType;

    fn entry(name: &str, layer: usize) -> WeightEntry {
        let mut entry =
            WeightEntry::model(name, vec![4, 4, 256], DType::MQ4G256V2, ShardPolicy::Replicate);
        entry.layer = Some(layer);
        entry
    }

    #[test]
    fn layers_past_the_budget_and_mtp_go_to_host() {
        let mut weights = vec![
            entry("model.language_model.layers.0.mlp.experts.gate_up_proj", 0),
            entry("model.language_model.layers.1.mlp.experts.down_proj", 1),
            entry("model.language_model.layers.1.mlp.shared_expert.down_proj.weight", 1),
            entry("model.language_model.layers.2.mlp.experts.gate_up_proj", 2),
            entry("mtp.layers.0.mlp.experts.down_proj", 0),
        ];
        assert_eq!(place_routed_experts(&mut weights, 2), 2);
        let host: Vec<_> = weights
            .iter()
            .filter(|entry| entry.residency == WeightResidency::HostMapped)
            .map(|entry| entry.name.as_str())
            .collect();
        assert_eq!(
            host,
            [
                "model.language_model.layers.2.mlp.experts.gate_up_proj",
                "mtp.layers.0.mlp.experts.down_proj"
            ]
        );
    }

    #[test]
    fn auto_fits_the_measured_gfx1201_card() {
        // R9700 before load: 32548 MiB free; Flash-Next non-expert weights
        // 5.364 GB; one trunk layer's routed experts 1.3369 GB.
        let free = 32548u64 << 20;
        let config = crate::config::compact_test_config();
        let reserve = |chunk| {
            auto_vram_reserve(&config, AUTO_VRAM_RESERVE_MAX_SEQ, chunk, F32, None, None).unwrap()
        };
        // G2's opt-in staging reserves two expert layers' worth more.
        let staged = if rdna_compute::gemm::qwen4_expert_stage_requested() { 2 } else { 0 };
        // Verify-sized spec logits free 1472 logit rows of the measured
        // 1536-row layout: one layer past the measured N = 16 fits.
        let measured_chunk = reserve(AUTO_VRAM_RESERVE_CHUNK);
        assert_eq!(
            auto_vram_layers(free, 5_364_000_000, 1_336_900_000, 48, measured_chunk),
            17 - staged
        );
        // gfx1201's 4096-row chunk spends that and one layer more on
        // forward scratch.
        let reserve = reserve(crate::gpu_forward::qwen4_prefill_chunk_default("gfx1201"));
        assert!(reserve > measured_chunk);
        assert_eq!(
            auto_vram_layers(free, 5_364_000_000, 1_336_900_000, 48, reserve),
            15 - staged
        );
        // A card that holds everything keeps every layer resident.
        assert_eq!(
            auto_vram_layers(u64::MAX / 2, 5_364_000_000, 1_336_900_000, 48, reserve),
            48
        );
        // No room past the reserve places every expert in host RAM.
        assert_eq!(
            auto_vram_layers(8 << 30, 5_364_000_000, 1_336_900_000, 48, reserve),
            0
        );
    }

    #[test]
    fn native_mtp_and_longer_context_grow_the_reserve() {
        let config = crate::config::compact_test_config();
        let base = auto_vram_reserve(
            &config,
            AUTO_VRAM_RESERVE_MAX_SEQ,
            AUTO_VRAM_RESERVE_CHUNK,
            F32,
            None,
            None,
        )
        .unwrap();
        let mtp = crate::mtp_spec::native_mtp_device_bytes(
            &config,
            AUTO_VRAM_RESERVE_MAX_SEQ,
            AUTO_VRAM_RESERVE_CHUNK,
            3,
            DType::MQ6G256V2,
            true,
        )
        .unwrap();
        let with_mtp = auto_vram_reserve(
            &config,
            AUTO_VRAM_RESERVE_MAX_SEQ,
            AUTO_VRAM_RESERVE_CHUNK,
            F32,
            Some(mtp),
            None,
        )
        .unwrap();
        assert_eq!(with_mtp - base, mtp);
        // The gathered QSA attention's context-sized scratch is charged on
        // top when that route reserves it at load.
        let gather = rdna_compute::tensor_ops::qsa_gathered_wmma_scratch_bytes(
            config.num_key_value_heads,
            AUTO_VRAM_RESERVE_MAX_SEQ,
        )
        .unwrap() as u64;
        let with_gather = auto_vram_reserve(
            &config,
            AUTO_VRAM_RESERVE_MAX_SEQ,
            AUTO_VRAM_RESERVE_CHUNK,
            F32,
            None,
            Some(gather),
        )
        .unwrap();
        assert_eq!(with_gather - base, gather);
        // The draft head's F32 requant scratch (vocab x hidden x 4) goes back
        // to the device at attach, before the first request allocates its
        // verify rows and GDN capture: the reserve holds the larger phase on
        // top of what the head keeps, not both.
        let (resident, scratch) = crate::mtp_gpu::Qwen4MtpGpu::device_bytes(
            &config,
            AUTO_VRAM_RESERVE_MAX_SEQ,
            DType::MQ6G256V2,
        )
        .unwrap();
        let (resident, scratch) = (resident as u64, scratch as u64);
        assert_eq!(scratch, (config.vocab_size * config.hidden_size * 4) as u64);
        let native = |max_k, row_capture| {
            crate::mtp_spec::native_mtp_device_bytes(
                &config,
                AUTO_VRAM_RESERVE_MAX_SEQ,
                AUTO_VRAM_RESERVE_CHUNK,
                max_k,
                DType::MQ6G256V2,
                row_capture,
            )
            .unwrap()
        };
        // At K = 3 the scratch outweighs the request, row capture included.
        assert_eq!(mtp, resident + scratch);
        assert_eq!(native(3, false), mtp);
        // At K = 10 the 11-row GDN capture outweighs the scratch.
        assert!(native(10, true) > resident + scratch);
        assert_eq!(native(10, false), resident + scratch);
        // On the measured R9700 the MTP bytes fit beside the chosen layers,
        // and one layer more would not have left them.
        let free = 32548u64 << 20;
        let (weights, layer) = (5_364_000_000u64, 1_336_900_000u64);
        let with = auto_vram_layers(free, weights, layer, 48, with_mtp) as u64;
        assert!(with < auto_vram_layers(free, weights, layer, 48, base) as u64);
        assert!(free - weights - with * layer >= with_mtp);
        assert!(free - weights - (with + 1) * layer < with_mtp);
        // A context past the measured one adds the trunk QSA arenas' growth
        // in the load's QSA format: fp8 K/V grows less than the F32 state.
        let growth = |format| {
            let long = auto_vram_reserve(
                &config,
                4 * AUTO_VRAM_RESERVE_MAX_SEQ,
                AUTO_VRAM_RESERVE_CHUNK,
                format,
                None,
                None,
            )
            .unwrap();
            let arena = (config
                .qsa_context_arena_bytes(4 * AUTO_VRAM_RESERVE_MAX_SEQ, format)
                .unwrap()
                - config
                    .qsa_context_arena_bytes(AUTO_VRAM_RESERVE_MAX_SEQ, format)
                    .unwrap())
                * config.n_full_layers();
            assert_eq!(long - base, arena as u64);
            long - base
        };
        assert!(growth(QsaKvFormat::Fp8) < growth(F32));
        // A shorter context keeps the measured reserve.
        let short = auto_vram_reserve(
            &config,
            2048,
            AUTO_VRAM_RESERVE_CHUNK,
            QsaKvFormat::Fp8,
            None,
            None,
        );
        assert_eq!(short.unwrap(), base);
    }

    #[test]
    fn host_ram_check_refuses_below_headroom() {
        let host = 60u64 << 30;
        assert!(check_host_ram(host, Some(host + HOST_RAM_HEADROOM_BYTES), 0).is_ok());
        let error = check_host_ram(host, Some(host + HOST_RAM_HEADROOM_BYTES - 1), 0).unwrap_err();
        assert!(error.contains("MemAvailable"), "{error}");
        assert!(check_host_ram(host, None, 0).is_err());
        assert!(check_host_ram(host, None, u64::MAX).is_err());
        assert!(check_host_ram(0, None, 0).is_ok());
        // The pool counts toward the room, byte for byte.
        assert!(check_host_ram(host, Some(host), HOST_RAM_HEADROOM_BYTES).is_ok());
        assert!(check_host_ram(host, Some(host), HOST_RAM_HEADROOM_BYTES - 1).is_err());
    }

    /// `/proc/meminfo` text from `(field, MiB)` pairs.
    fn meminfo(fields: &[(&str, u64)]) -> String {
        fields.iter().map(|(key, mib)| format!("{key}:{:>16} kB\n", mib << 10)).collect()
    }

    /// The measured 5-card gfx1201 host (MemTotal 128865156 kB, pool cap
    /// 16108144 pages), 5 x 16 MiB of idle GTT, and `pool_mib` of pages outside
    /// every counter on top of `baseline_mib`.
    fn gfx1201_host(pool_mib: u64, baseline_mib: u64) -> String {
        let total = 128_865_156 >> 10;
        let (anon, slab, sreclaim) = (14 << 10, 3 << 10, 2 << 10);
        let cached = total - pool_mib - baseline_mib - 80 - anon - slab - 1024;
        meminfo(&[
            ("MemTotal", total),
            ("MemFree", 1024),
            ("MemAvailable", 45_800),
            ("Cached", cached),
            ("SwapCached", 0),
            ("AnonPages", anon),
            ("Shmem", 2048),
            ("KReclaimable", sreclaim),
            ("Slab", slab),
            ("SReclaimable", sreclaim),
        ])
    }

    #[test]
    fn ttm_pool_estimate_credits_only_untracked_pages_past_the_baseline() {
        const POOL_PAGES: u64 = 16_108_144;
        let gtt = 5 * (16 << 20);
        let mib = |bytes: u64| bytes >> 20;
        // The measured baseline (2.76 GiB, pool empty) is never credited.
        assert_eq!(ttm_pool_estimate_from(&gfx1201_host(0, 2826), gtt, POOL_PAGES), Some(0));
        // The reported leftover: 15,292,712 pool pages (59,737 MiB).
        let pool = ttm_pool_estimate_from(&gfx1201_host(59_737, 2826), gtt, POOL_PAGES).unwrap();
        assert_eq!(mib(pool), 59_737 + 2826 - (UNTRACKED_KERNEL_BYTES >> 20));
        // Capped at page_pool_size.
        let capped = ttm_pool_estimate_from(&gfx1201_host(59_737, 2826), gtt, 1 << 20).unwrap();
        assert_eq!(capped, (1 << 20) * TTM_PAGE_BYTES);
        // Live GTT (another host-mapped Flash-Next) is not pool.
        let live = ttm_pool_estimate_from(&gfx1201_host(0, 2826), 46 << 30, POOL_PAGES);
        let held = ttm_pool_estimate_from(&gfx1201_host(46 << 10, 2826), 46 << 30, POOL_PAGES);
        assert_eq!((live, held), (Some(0), Some(0)));
    }

    #[test]
    fn leftover_pool_admits_the_reload_and_a_real_shortage_still_refuses() {
        // N=12 host-maps 46.1 GiB; the reported refusal read MemAvailable 45.8 GiB.
        let host = 47_206u64 << 20;
        let available = Some(46_899u64 << 20);
        let gtt = 5 * (16 << 20);
        assert!(check_host_ram(host, available, 0).is_err());
        let leftover = ttm_pool_estimate_from(&gfx1201_host(59_737, 2826), gtt, 16_108_144);
        assert!(check_host_ram(host, available, leftover.unwrap()).is_ok());
        // The same MemAvailable with nothing parked in the pool.
        let empty = ttm_pool_estimate_from(&gfx1201_host(0, 2826), gtt, 16_108_144);
        let error = check_host_ram(host, available, empty.unwrap()).unwrap_err();
        assert!(error.contains("MemAvailable is 45.8 GiB (plus 0.0 GiB"), "{error}");
        // A pool smaller than the 4.3 GiB gap still refuses.
        let small = ttm_pool_estimate_from(&gfx1201_host(3 << 10, 2826), gtt, 16_108_144);
        assert!(check_host_ram(host, available, small.unwrap()).is_err());
    }

    #[test]
    fn gtt_cap_counts_what_devices_already_hold() {
        // This host: pages_limit 16108144 (61.4 GiB); N=12 host-maps 46.1 GiB.
        let limit_bytes = 16_108_144 * TTM_PAGE_BYTES;
        let host = 46u64 << 30;
        let fresh = GttBudget { limit_bytes, used_bytes: 16 << 20 };
        assert!(check_gtt_cap(host, Some(fresh)).is_ok());
        assert!(check_gtt_cap(limit_bytes - fresh.used_bytes, Some(fresh)).is_ok());
        let error = check_gtt_cap(limit_bytes - fresh.used_bytes + 1, Some(fresh)).unwrap_err();
        assert!(error.contains("pages_limit") && error.contains("16108144 pages"), "{error}");
        // A second load beside one already holding 46 GiB of GTT is refused.
        let beside = GttBudget { limit_bytes, used_bytes: 46 << 30 };
        assert!(check_gtt_cap(host, Some(beside)).is_err());
        // Userptr host memory, or no TTM limit in sysfs, is not GTT-capped.
        assert!(check_gtt_cap(u64::MAX, None).is_ok());
    }

    #[test]
    fn parse_accepts_auto_and_counts_only() {
        assert_eq!(parse_expert_vram_layers(" auto "), Ok(ExpertVramLayers::Auto));
        assert_eq!(parse_expert_vram_layers("16"), Ok(ExpertVramLayers::Layers(16)));
        assert!(parse_expert_vram_layers("16GB").is_err());
    }

    #[test]
    fn unset_is_resident_only_where_everything_fits_and_explicit_wins() {
        const GIB: u64 = 1 << 30;
        // Flash-Next: ~5 GiB non-expert, ~64 GiB experts, 6.5 GiB reserve.
        let r9700 = fits_fully_resident(32 * GIB, 5 * GIB, 64 * GIB, 6 * GIB + GIB / 2);
        let halo = fits_fully_resident(120 * GIB, 5 * GIB, 64 * GIB, 6 * GIB + GIB / 2);
        assert!(!r9700 && halo);
        assert!(fits_fully_resident(10 * GIB, 2 * GIB, 7 * GIB, GIB));
        assert!(!fits_fully_resident(10 * GIB, 2 * GIB, 7 * GIB, GIB + 1));
        assert!(!fits_fully_resident(u64::MAX, 1, u64::MAX, 0));
        assert_eq!(resolve_expert_vram_layers(None, || Ok(true)), Ok(None));
        assert_eq!(resolve_expert_vram_layers(None, || Ok(false)), Ok(Some(ExpertVramLayers::Auto)));
        assert!(resolve_expert_vram_layers(None, || Err("vram".into())).is_err());
        // Explicit values never consult the fit.
        for policy in [ExpertVramLayers::Layers(12), ExpertVramLayers::Layers(99), ExpertVramLayers::Auto] {
            let resolved = resolve_expert_vram_layers(Some(policy), || panic!("explicit consulted the fit"));
            assert_eq!(resolved, Ok(Some(policy)));
        }
    }

    #[test]
    fn only_qwen4_on_discrete_gpus_keeps_host_memory_out_of_reclaim() {
        assert!(keeps_host_memory_out_of_reclaim(crate::ARCH_ID, &["gfx1201"]));
        assert!(keeps_host_memory_out_of_reclaim(crate::ARCH_ID, &["gfx1100", "gfx942"]));
        // Another model on the same card (arch 5: the Qwen3.5-family trunk).
        assert!(!keeps_host_memory_out_of_reclaim(5, &["gfx1201"]));
        // Unified memory, alone or beside a discrete card ROCr also exposes.
        assert!(!keeps_host_memory_out_of_reclaim(crate::ARCH_ID, &["gfx1151"]));
        assert!(!keeps_host_memory_out_of_reclaim(crate::ARCH_ID, &["gfx1100", "gfx1151"]));
        // No card known, or one whose memory kind is not known.
        assert!(!keeps_host_memory_out_of_reclaim(crate::ARCH_ID, &[]));
        assert!(!keeps_host_memory_out_of_reclaim(crate::ARCH_ID, &["gfx90a"]));
    }
}
