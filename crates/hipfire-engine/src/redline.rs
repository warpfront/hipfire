// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Generic Redline bench helpers — architecture-neutral GPU snapshot primitives.
//!
//! Arch-specific snapshot builders (`redline_qwen_snapshot`, etc.) remain in
//! `daemon.rs` because they branch on `Qwen35Bundle` / `Deepseek4Bundle` etc.
//! Only the generic helpers that operate on raw buffers are moved here.
//!
//! Relocated verbatim from `crates/hipfire-daemon/src/main.rs` (wave 3).

pub fn redline_capture_json(
    gpu: &rdna_compute::Gpu,
    summary: rdna_compute::replay::ReplayCaptureSummary,
    detail: bool,
) -> serde_json::Value {
    let mut value = serde_json::json!({
        "launches": summary.launch_count,
        "unique_kernels": summary.unique_kernel_count,
        "sequence_hash": format!("{:016x}", summary.sequence_hash),
    });
    if detail {
        value["sequence"] = serde_json::Value::Array(
            gpu.replay
                .recorded_launches()
                .iter()
                .map(|launch| {
                    serde_json::json!({
                        "kernel": launch.kernel.as_str(),
                        "artifact": launch.artifact.as_ref().map(|artifact| artifact.to_string()),
                        "code_object": launch.artifact.as_ref().map(|artifact| artifact.provenance_json()),
                        "grid": launch.grid,
                        "block": launch.block,
                        "shared_mem": launch.shared_mem,
                        "kernarg_bytes": launch.kernarg.len(),
                        "kernarg_hex": launch.kernarg.iter().map(|byte| format!("{byte:02x}")).collect::<String>(),
                        "kernarg_hash": format!("{:016x}", {
                            let mut hash = 0xcbf29ce484222325_u64;
                            for byte in &launch.kernarg {
                                hash ^= u64::from(*byte);
                                hash = hash.wrapping_mul(0x100000001b3);
                            }
                            hash
                        }),
                    })
                })
                .collect(),
        );
    }
    value
}

pub fn redline_hash(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Thread-local HIP API counters sampled around one G0 arm (railgun design
/// §5 G0). A launch that bypasses the recorder funnels is counted here but
/// missing from the arm's sequence, so the harness can refuse the comparison.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct G0HipCounters {
    pub launch_kernel: u64,
    pub memcpy_dtod: u64,
    pub memcpy_htod: u64,
    pub memset: u64,
}

impl G0HipCounters {
    pub fn now() -> Self {
        use hip_bridge::launch_counters as lc;
        Self {
            launch_kernel: lc::launch_kernel::count(),
            memcpy_dtod: lc::memcpy_dtod::count(),
            memcpy_htod: lc::memcpy_htod::count(),
            memset: lc::memset::count(),
        }
    }

    fn since(self, before: Self) -> Self {
        Self {
            launch_kernel: self.launch_kernel.saturating_sub(before.launch_kernel),
            memcpy_dtod: self.memcpy_dtod.saturating_sub(before.memcpy_dtod),
            memcpy_htod: self.memcpy_htod.saturating_sub(before.memcpy_htod),
            memset: self.memset.saturating_sub(before.memset),
        }
    }
}

/// One side of the G0 recording-invariance comparison.
/// `Observe`: the eager forward with `is_recording()` false, every funnel
/// launch observed. `Record`: the same forward under a replay recording (the
/// railgun authoring state: `is_recording()` true, `capture_mode` false).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum G0Arm {
    Observe,
    Record,
}

impl G0Arm {
    /// The optional `"g0": "observe" | "record"` request field.
    pub fn from_request(msg: &serde_json::Value) -> Result<Option<Self>, String> {
        match msg.get("g0") {
            None | Some(serde_json::Value::Null) => Ok(None),
            Some(value) => match value.as_str() {
                Some("observe") => Ok(Some(Self::Observe)),
                Some("record") => Ok(Some(Self::Record)),
                _ => Err(format!("g0 must be \"observe\" or \"record\", got {value}")),
            },
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Observe => "observe",
            Self::Record => "record",
        }
    }

    /// Start the arm; returns the counters to pass to [`Self::finish`].
    pub fn begin(self, gpu: &mut rdna_compute::Gpu) -> Result<G0HipCounters, String> {
        match self {
            Self::Observe => gpu.replay.begin_g0_observation(),
            Self::Record => gpu.replay.begin_capture(),
        }
        .map_err(|reason| format!("g0 {} refused: {reason}", self.name()))?;
        Ok(G0HipCounters::now())
    }

    /// Close the arm and serialise its launch sequence (the same row schema
    /// as [`redline_capture_json`]'s detail rows) plus the HIP counter deltas.
    pub fn finish(
        self,
        gpu: &mut rdna_compute::Gpu,
        before: G0HipCounters,
    ) -> Result<serde_json::Value, String> {
        let hip = G0HipCounters::now().since(before);
        let rows: Vec<serde_json::Value> = match self {
            Self::Observe => gpu
                .replay
                .finish_g0_observation()
                .map_err(|reason| format!("g0 observe: {reason}"))?
                .iter()
                .map(|l| g0_launch_json(&l.kernel, l.grid, l.block, l.shared_mem, &l.kernarg))
                .collect(),
            Self::Record => {
                gpu.replay
                    .finish_capture()
                    .map_err(|reason| format!("g0 record: {reason}"))?;
                gpu.replay
                    .recorded_launches()
                    .iter()
                    .map(|l| g0_launch_json(&l.kernel, l.grid, l.block, l.shared_mem, &l.kernarg))
                    .collect()
            }
        };
        let unique = rows
            .iter()
            .filter_map(|r| r["kernel"].as_str())
            .collect::<std::collections::BTreeSet<_>>()
            .len();
        Ok(serde_json::json!({
            "arm": self.name(),
            "launches": rows.len(),
            "unique_kernels": unique,
            "hip": {
                "launch_kernel": hip.launch_kernel,
                "memcpy_dtod": hip.memcpy_dtod,
                "memcpy_htod": hip.memcpy_htod,
                "memset": hip.memset,
            },
            "sequence": rows,
        }))
    }
}

fn g0_launch_json(
    kernel: &str,
    grid: [u32; 3],
    block: [u32; 3],
    shared_mem: u32,
    kernarg: &[u8],
) -> serde_json::Value {
    serde_json::json!({
        "kernel": kernel,
        "grid": grid,
        "block": block,
        "shared_mem": shared_mem,
        "kernarg_bytes": kernarg.len(),
        "kernarg_hex": kernarg.iter().map(|byte| format!("{byte:02x}")).collect::<String>(),
    })
}

pub fn redline_append_buffer(
    gpu: &rdna_compute::Gpu,
    output: &mut Vec<u8>,
    buffer: &hip_bridge::DeviceBuffer,
) -> Result<(), String> {
    let start = output.len();
    output.resize(start + buffer.size(), 0);
    gpu.hip
        .memcpy_dtoh(&mut output[start..], buffer)
        .map_err(|error| error.to_string())
}

pub fn redline_append_tensor(
    gpu: &rdna_compute::Gpu,
    output: &mut Vec<u8>,
    tensor: &Option<rdna_compute::GpuTensor>,
) -> Result<(), String> {
    if let Some(tensor) = tensor {
        redline_append_buffer(gpu, output, &tensor.buf)?;
    }
    Ok(())
}

pub fn redline_append_tensor_region(
    gpu: &rdna_compute::Gpu,
    output: &mut Vec<u8>,
    regions: &mut Vec<RedlineRegionHash>,
    name: String,
    tensor: &Option<rdna_compute::GpuTensor>,
) -> Result<(), String> {
    let Some(tensor) = tensor else {
        return Ok(());
    };
    let start = output.len();
    redline_append_buffer(gpu, output, &tensor.buf)?;
    let bytes = output.len() - start;
    regions.push(RedlineRegionHash {
        name,
        bytes,
        hash: redline_hash(&output[start..]),
    });
    Ok(())
}

#[derive(PartialEq, Debug)]
pub struct RedlineRegionHash {
    pub name: String,
    pub bytes: usize,
    pub hash: u64,
}

pub fn redline_append_tensor_slice(
    gpu: &rdna_compute::Gpu,
    output: &mut Vec<u8>,
    tensor: &rdna_compute::GpuTensor,
    offset: usize,
    len: usize,
) -> Result<(), String> {
    if offset.saturating_add(len) > tensor.numel() {
        return Err(format!(
            "redline tensor slice {}+{} exceeds {}",
            offset,
            len,
            tensor.numel()
        ));
    }
    let view = tensor.sub_offset(offset, len);
    redline_append_buffer(gpu, output, &view.buf)
}
