// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Opt-in developer dump of the prefill CHUNK boundaries of selected layers, for the
//! single-vs-EP2 bitwise comparison of the owner-routed chunk MoE
//! (`fn-ep/oracle/compare.py chunk`).
//!
//! Off by default: with `HIPFIRE_CHUNK_DUMP_DIR` unset nothing is read, split or
//! synchronized. When set, a chunk forward (more than one routing row) is executed in
//! segments that end after each selected layer, and each boundary synchronizes the device
//! and writes `$DIR/rank-<device>/chunk-<start position>-<rows>/layer-<L>/digest.json`:
//!
//! * `route` — SHA-256 of the chunk's routed expert ids.
//! * `fold_output` — SHA-256 of the layer's FFN output rows (the MoE fold output the
//!   mlp hyper write consumes).
//! * `hc_streams` — SHA-256 of the hyper-connection streams after the layer.
//! * `grouped` — `y_down_grouped` of the layer: the tile ids (SHA-256) and one SHA-256
//!   per LIVE tile of its 16 BF16 grouped rows, with the tile's expert. On the
//!   owner-routed route a tile's rows are either computed here or pulled from the peer, so
//!   every live tile must equal the single card's tile bit for bit. Absent (with a
//!   `reason`) off the BF16 grouped route.
//!
//! `HIPFIRE_CHUNK_DUMP_LAYERS` (default `0,2,3,31,47`) selects layers;
//! `HIPFIRE_CHUNK_DUMP_RAW=1` additionally writes the raw bytes (`fold-output.bin`,
//! `hc-streams.bin`, `y-down-grouped.live.bin`, the live tiles concatenated) — failure
//! diagnosis only, they are large. The segmenting keeps every launch of the forward; it
//! only forgoes fusions that span a selected layer's last step.

use crate::config::Qwen4Config;
use crate::gpu_forward::Qwen4GpuForwardScratch;
use rdna_compute::{Gpu, GpuTensor};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::LazyLock;

pub const CHUNK_DUMP_DIR_ENV: &str = "HIPFIRE_CHUNK_DUMP_DIR";
pub const CHUNK_DUMP_LAYERS_ENV: &str = "HIPFIRE_CHUNK_DUMP_LAYERS";
pub const CHUNK_DUMP_RAW_ENV: &str = "HIPFIRE_CHUNK_DUMP_RAW";
const DEFAULT_LAYERS: [usize; 5] = [0, 2, 3, 31, 47];

pub(crate) struct ChunkDump {
    dir: PathBuf,
    layers: BTreeSet<usize>,
    raw: bool,
}

static CONFIG: LazyLock<Option<ChunkDump>> = LazyLock::new(|| {
    let dir = hipfire_config::developer_var(CHUNK_DUMP_DIR_ENV)
        .ok()
        .filter(|dir| !dir.is_empty())?;
    let layers = match hipfire_config::developer_var(CHUNK_DUMP_LAYERS_ENV) {
        Ok(list) if !list.trim().is_empty() => list
            .split(',')
            .filter_map(|x| x.trim().parse().ok())
            .collect(),
        _ => DEFAULT_LAYERS.into_iter().collect(),
    };
    let raw = hipfire_config::developer_var(CHUNK_DUMP_RAW_ENV).as_deref() == Ok("1");
    Some(ChunkDump {
        dir: PathBuf::from(dir),
        layers,
        raw,
    })
});

/// The dump configuration, `None` unless `HIPFIRE_CHUNK_DUMP_DIR` is set.
pub(crate) fn config() -> Option<&'static ChunkDump> {
    CONFIG.as_ref()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn sha(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

/// Host copy of the first `bytes` bytes of `tensor` (device-synchronized by the caller).
fn read(gpu: &Gpu, tensor: &GpuTensor, bytes: usize) -> Result<Vec<u8>, String> {
    let mut out = vec![0u8; bytes.min(tensor.buf.size())];
    gpu.hip
        .memcpy_dtoh(&mut out, &tensor.buf)
        .map_err(|error| error.to_string())?;
    Ok(out)
}

/// The first `rows` rows of a tensor that holds `capacity` rows of equal width.
fn rows_bytes(tensor: &GpuTensor, rows: usize, capacity: usize) -> usize {
    let total = tensor.buf.size();
    if capacity == 0 || total % capacity != 0 {
        total
    } else {
        total / capacity * rows
    }
}

impl ChunkDump {
    pub(crate) fn wants(&self, layer: usize) -> bool {
        self.layers.contains(&layer)
    }

    /// Write one layer boundary of one chunk. `bf16_grouped`: the layer ran the BF16
    /// grouped route (symmetric IU4), so `y_down_grouped` holds BF16 rows.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dump_layer(
        &self,
        gpu: &mut Gpu,
        scratch: &Qwen4GpuForwardScratch,
        config: &Qwen4Config,
        layer: usize,
        start_position: usize,
        rows: usize,
        bf16_grouped: bool,
    ) -> Result<(), String> {
        gpu.bind_thread().map_err(|error| error.to_string())?;
        gpu.hip
            .device_synchronize()
            .map_err(|error| error.to_string())?;
        let dir = self
            .dir
            .join(format!("rank-{}", gpu.device_id))
            .join(format!("chunk-{start_position}-{rows}"))
            .join(format!("layer-{layer}"));
        std::fs::create_dir_all(&dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
        let write = |name: &str, bytes: &[u8]| -> Result<(), String> {
            let path = dir.join(name);
            std::fs::write(&path, bytes).map_err(|e| format!("write {}: {e}", path.display()))
        };
        let capacity = scratch.max_chunk;
        let k = config.num_experts_per_tok;
        let route = read(
            gpu,
            &scratch.moe_topk_indices,
            rows_bytes(&scratch.moe_topk_indices, rows, capacity).min(rows * k * 4),
        )?;
        let fold = read(
            gpu,
            &scratch.ffn_output,
            rows_bytes(&scratch.ffn_output, rows, capacity),
        )?;
        let streams = read(
            gpu,
            &scratch.streams,
            rows_bytes(&scratch.streams, rows, capacity),
        )?;
        let mut grouped = serde_json::json!({"present": false});
        let mut live_rows: Vec<u8> = Vec::new();
        if bf16_grouped {
            let slots = rows * k;
            let bound = crate::program::grouped_m_total_bound(slots, config.num_experts)
                .map_err(|error| error.to_string())?;
            let tiles = bound / 16;
            let tile_bytes = 16 * config.hidden_size * 2;
            let ids = read(gpu, &scratch.moe_expert_tile_ids, tiles * 4)?;
            let down = read(gpu, &scratch.moe_y_down_grouped, tiles * tile_bytes)?;
            let mut live = Vec::new();
            let mut experts = Vec::new();
            let mut digests = Vec::new();
            for tile in 0..tiles {
                let id = i32::from_le_bytes(ids[tile * 4..tile * 4 + 4].try_into().unwrap());
                if id < 0 {
                    continue;
                }
                let rows_of_tile = &down[tile * tile_bytes..(tile + 1) * tile_bytes];
                live.push(tile);
                experts.push(id);
                digests.push(sha(rows_of_tile));
                if self.raw {
                    live_rows.extend_from_slice(rows_of_tile);
                }
            }
            grouped = serde_json::json!({
                "present": true,
                "tiles": tiles,
                "tile_bytes": tile_bytes,
                "tile_ids": sha(&ids),
                "live_tiles": live,
                "tile_experts": experts,
                "tile_digests": digests,
            });
        } else {
            grouped["reason"] =
                "the layer did not run the BF16 grouped (symmetric IU4) route".into();
        }
        let digest = serde_json::json!({
            "layer": layer,
            "start_position": start_position,
            "rows": rows,
            "device": gpu.device_id,
            "route": sha(&route),
            "fold_output": sha(&fold),
            "hc_streams": sha(&streams),
            "grouped": grouped,
        });
        write(
            "digest.json",
            serde_json::to_string_pretty(&digest)
                .map_err(|e| e.to_string())?
                .as_bytes(),
        )?;
        if self.raw {
            write("fold-output.bin", &fold)?;
            write("hc-streams.bin", &streams)?;
            if !live_rows.is_empty() {
                write("y-down-grouped.live.bin", &live_rows)?;
            }
        }
        Ok(())
    }
}
