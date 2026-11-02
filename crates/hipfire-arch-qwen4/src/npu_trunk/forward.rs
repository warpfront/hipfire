// SPDX-License-Identifier: Apache-2.0
//! Calibrated Halo scheduler. Effective off performs no new I/O or allocation.
use super::*;
use crate::npu_trunk::{self as trunk, Family, FamilyMask, NpuTrunkMode, Shape, FLASH_NEXT};
use hipfire_dispatch::pipeline::{
    execute_gated_delta_net_core_rows, execute_gated_delta_net_uncovered,
    execute_grouped_depthwise_rows, execute_indexed_attention_core_rows,
    execute_indexed_attention_uncovered, execute_validated_steps_window, gdn_qkv_projection_bf16,
    CoreWindow, GdnTrunkArenas, HcFusionCursor, QsaTrunkArenas,
};
use hipfire_runtime::coproc::{self, EpilogueContract, PairOp};
use railgun::npu::trunk::{
    padded_rows_of, LayerPanels, NpuTrunkRuntime, OpenOptions, PanelBuild, PanelProgram, Seq,
    TxnBuilder, MAX_ADAPTIVE_CONTEXTS,
};
use rdna_compute::npu_trunk::{NpuBf16Layout, NpuTrunkOutput};
use rdna_compute::tensor_ops::Bf16Arena;
use std::{ffi::c_void, io::Write};

pub(super) struct NpuForward {
    mode: NpuTrunkMode,
    families: FamilyMask,
    request_families: FamilyMask,
    family_override: Option<FamilyMask>,
    rates: Option<coproc::RateEstimator>,
    sampling: Option<PairSampling>,
    request_epoch: u64,
    /// Snapshotted at request begin; changing the diagnostic control cannot alter an in-flight request.
    request_gpu: bool,
    guard_tripped: bool,
    /// `HIPFIRE_QWEN4_NPU_TRUNK_SAMPLE_LOG=1`: one `pair-sample` line per resolved pair (auto), no added sync.
    sample_log: bool,
    /// The calibration (and estimator generation) the current auto request was planned with, kept for the sample log.
    request_cal: Option<(u64, coproc::Calibration)>,
    layers: Vec<NpuLayer>,
    emulation: Option<EmulationScratch>,
    // The pack producer writes compact activation sidecars in both modes.
    scales: [GpuTensor; 2],
    bank_b: Option<Qwen4GpuForwardScratch>,
    runtime: Option<NpuTrunkRuntime>,
    arenas: Option<[[Option<GpuTensor>; 2]; 2]>,
    trace: Option<std::io::BufWriter<std::fs::File>>,
    host_tokens: Vec<u8>,
    host_ple: Vec<u8>,
    /// The pair or diagnostic-route line was already emitted for this request.
    pair_logged: bool,
    calibration: Option<coproc::Calibration>,
    n_star_cache: Option<(u64, FamilyMask, Option<usize>)>,
    request_plan: Option<coproc::Plan>,
    request_origin: usize,
    /// Wave-padded bank rows below 8192 whose row-adaptive NPU programs this load admitted ([`admitted_plan`]).
    adaptive: Vec<usize>,
    /// The fault latch: set by the first pair fault under `auto`, it keeps the route off for the process.
    latch: trunk::containment::FaultLatch,
    /// PCI bus id the runtime opened on (host facts in the fault line); `None` when no runtime opened.
    gpu_pci: Option<String>,
}

// SAFETY: this private owner exclusively owns the entire railgun HIP Rc graph;
// none of its HIP wrappers/Rc clones escape it. ArchModel moves the owner, never
// shares it, and every execution uses the destination thread's selected Gpu.
// The relay only sees immutable entry points and raw events, not those Rcs.
unsafe impl Send for NpuForward {}
struct NpuLayer {
    shape: Shape,
    map: trunk::ColumnMap,
    twin: Option<GpuTwin>,
}
struct GpuTwin {
    w: GpuTensor,
    scales: GpuTensor,
}
struct EmulationScratch {
    a: GpuTensor,
    sums: GpuTensor,
}
fn fail(e: impl std::fmt::Display) -> Qwen4GpuForwardError {
    invalid(format!("npu-trunk: {e}"))
}
/// Compare the live `hipEventRecord` implementations before passing serving
/// streams to the relay (#926). Both runtime owners stay alive for this check.
/// Paths are only diagnostic: renamed/replaced libraries and failed `dladdr`
/// lookups must not let distinct implementations pass as one runtime.
pub(super) fn two_hip_runtime_reason(
    relay_address: usize,
    serving_address: usize,
    relay: Option<&str>,
    serving: Option<&str>,
) -> Option<String> {
    if relay_address != 0 && relay_address == serving_address {
        return None;
    }
    let relay = relay.unwrap_or("<unknown library>");
    let serving = serving.unwrap_or("<unknown library>");
    Some(format!(
        "two HIP runtimes: the NPU relay loaded {relay} but the serving GPU runtime \
         runs {serving}; hipEventRecord identity differs \
         (relay={relay_address:#x}, serving={serving_address:#x}); \
         point HIPFIRE_ROCM_PATH (and LD_LIBRARY_PATH) at one install \
         so a single runtime serves both"
    ))
}
fn stream(gpu: &Gpu) -> *mut c_void {
    gpu.active_stream
        .as_ref()
        .map_or(std::ptr::null_mut(), |s| s.as_raw())
}
fn layout(map: &trunk::ColumnMap, rows: usize) -> NpuBf16Layout {
    NpuBf16Layout {
        padded_rows: rows.div_ceil(256) * 256,
        k: 2560,
        panels: map.panel_count(),
        n512: false,
    }
}
fn family_view<'a>(
    map: &trunk::ColumnMap,
    tensor: &'a GpuTensor,
    family: Family,
) -> Option<Bf16Arena<'a>> {
    map.parts
        .iter()
        .find(|&&(f, _, _)| f == family)
        .map(|&(_, col, _)| Bf16Arena {
            tensor,
            row_pitch_elems: map.padded_cols(),
            col0_elems: col as usize,
        })
}
fn transport_shape(shape: Shape) -> railgun::npu::trunk::Shape {
    use railgun::npu::trunk::Shape as T;
    match shape {
        Shape::GdnIn => T::GdnIn,
        Shape::QsaIn => T::QsaIn,
        Shape::GdnOut => T::GdnOut,
        Shape::QsaO => T::QsaO,
    }
}

fn parse_family_override(value: &str) -> Result<FamilyMask, String> {
    let mut mask = FamilyMask::NONE;
    for name in value.split(',').map(str::trim) {
        mask = mask | match name {
            "all" => FamilyMask::ALL,
            "qkv" => FamilyMask::from_bits(0x01),
            "zab" => FamilyMask::from_bits(0x0e),
            "q" => FamilyMask::from_bits(0x10),
            "kvidx" => FamilyMask::from_bits(0xe0),
            _ => return Err(format!("invalid HIPFIRE_QWEN4_NPU_TRUNK_FAMILIES group {name:?}; expected qkv,zab,q,kvidx or all")),
        };
    }
    Ok(mask)
}

fn intersect_families(a: FamilyMask, b: FamilyMask) -> FamilyMask {
    FamilyMask::from_bits(a.bits() & b.bits())
}

fn family_names(mask: FamilyMask) -> String {
    if mask.is_empty() {
        return "none".to_owned();
    }
    [("qkv", 0x01), ("zab", 0x0e), ("q", 0x10), ("kvidx", 0xe0)]
        .into_iter()
        .filter_map(|(name, bits)| (mask.bits() & bits != 0).then_some(name))
        .collect::<Vec<_>>()
        .join(",")
}

fn registered_c_view(
    rt: &NpuTrunkRuntime,
    bank: usize,
    shape: Shape,
    families: FamilyMask,
) -> Result<Option<GpuTensor>, Qwen4GpuForwardError> {
    let map = shape.column_map(families);
    if map.panel_count() == 0 {
        return Ok(None);
    }
    let arena = rt.c_arena(bank, transport_shape(shape)).map_err(fail)?;
    // Borrowed metadata only: runtime owns the registration/pages until all
    // bank retirement events complete. Never free this as a GPU allocation.
    Ok(Some(GpuTensor {
        buf: unsafe { hip_bridge::DeviceBuffer::from_raw(arena.device as *mut c_void, arena.len) },
        shape: vec![8192, map.padded_cols()],
        dtype: DType::BF16,
    }))
}

/// Rebind logical extents only; the dedicated bank's allocation capacity and
/// ownership slots remain unchanged, so a later pair can grow back to 8192.
fn bind_bank_extents(
    tensors: &mut [&mut GpuTensor],
    slots: &[ScratchSlot],
) -> Result<(), Qwen4GpuForwardError> {
    if tensors.len() != ScratchField::COUNT || slots.len() != tensors.len() {
        return Err(fail("bank scratch field count mismatch"));
    }
    for (tensor, slot) in tensors.iter().zip(slots) {
        if tensor.shape.is_empty()
            || tensor.dtype != slot.dtype
            || slot.bytes()? > tensor.buf.size()
        {
            return Err(fail(
                "bank live view exceeds its dedicated scratch allocation",
            ));
        }
    }
    for (tensor, slot) in tensors.iter_mut().zip(slots) {
        tensor.shape.truncate(1);
        tensor.shape[0] = slot.elements;
    }
    Ok(())
}

fn preserve_native_projection_capacity(live: &mut [ScratchSlot], capacity: &[ScratchSlot]) {
    // The native ragged producer pads only these four dedicated buffers.
    // Eligibility must see the same max_chunk capacity as sequential prefill;
    // mixer/kernel row counts stay the live rows, never the padded capacity.
    for field in [
        ScratchField::HcMixed,
        ScratchField::GdnZ,
        ScratchField::GdnA,
        ScratchField::GdnB,
    ] {
        live[field as usize].elements = capacity[field as usize].elements;
    }
}

fn bind_bank_rows(
    s: &mut Qwen4GpuForwardScratch,
    config: &Qwen4Config,
    rows: usize,
) -> Result<(), Qwen4GpuForwardError> {
    if s.mode != Qwen4ForwardScratchMode::Full || rows == 0 || rows > s.max_chunk {
        return Err(fail(
            "bank live views require full scratch within its row capacity",
        ));
    }
    let (mut slots, _) = Qwen4GpuForwardScratch::plan(config, rows, Qwen4ForwardScratchMode::Full)?;
    preserve_native_projection_capacity(&mut slots, &s.slots);
    // Other row-checked bindings (including PLE and grouped MoE) use live rows.
    // ScratchField order is identical to the existing allocation/free tables.
    bind_bank_extents(
        &mut [
            &mut s.token_ids,
            &mut s.embedding_rot,
            &mut s.embeddings,
            &mut s.streams,
            &mut s.hc_normalized,
            &mut s.hc_low,
            &mut s.hc_up,
            &mut s.hc_mixed,
            &mut s.hc_gates,
            &mut s.rotation,
            &mut s.projection,
            &mut s.projection2,
            &mut s.gdn_a,
            &mut s.gdn_b,
            &mut s.gdn_gate,
            &mut s.gdn_beta,
            &mut s.gdn_recurrent_output,
            &mut s.gdn_bf16,
            &mut s.gdn_z,
            &mut s.gdn_output,
            &mut s.qsa_index,
            &mut s.qsa_qgate,
            &mut s.qsa_k,
            &mut s.qsa_v,
            &mut s.qsa_output,
            &mut s.qsa_selected,
            &mut s.attention_output,
            &mut s.ple_staged,
            &mut s.ple_rows,
            &mut s.ple_key,
            &mut s.ple_value,
            &mut s.ple_query,
            &mut s.ple_gated,
            &mut s.ple_normed,
            &mut s.ple_normed_wide,
            &mut s.ple_output,
            &mut s.router_logits,
            &mut s.moe_x_rot,
            &mut s.moe_gate_up,
            &mut s.moe_gate,
            &mut s.moe_up,
            &mut s.moe_hidden,
            &mut s.ffn_output,
            &mut s.moe_shared_output,
            &mut s.moe_gate_batch,
            &mut s.moe_up_batch,
            &mut s.moe_rot_batch,
            &mut s.moe_topk_indices,
            &mut s.moe_topk_weights,
            &mut s.moe_down_expanded,
            &mut s.moe_scalar,
            &mut s.moe_expert_token_counts,
            &mut s.moe_expert_offsets,
            &mut s.moe_sorted_slot_index,
            &mut s.moe_expert_tile_ids,
            &mut s.moe_inverse_perm,
            &mut s.moe_y_gate_up_grouped,
            &mut s.moe_y_down_grouped,
            &mut s.logits,
        ],
        &slots,
    )
}

/// Auto panels are packed before the overlay swap; forced modes retain stock
/// bytes until the successfully applied route determines their calibration.
enum PreparedLayers {
    Stock(Vec<(Shape, Vec<(usize, Vec<u8>)>)>),
    Packed(Vec<(Shape, trunk::PackedB, Vec<u8>)>),
}
pub(crate) struct PreparedTrunk {
    mode: NpuTrunkMode,
    layers: PreparedLayers,
    download_ms: f64,
    pack_ms: f64,
    pre_overlay: bool,
    overlay_route: Option<rdna_compute::TrunkIu4Default>,
    matched: Option<(coproc::Calibration, coproc::CalKey, &'static str)>,
    families: FamilyMask,
}

impl PreparedTrunk {
    pub(crate) fn set_overlay_route(&mut self, route: Option<rdna_compute::TrunkIu4Default>) {
        self.overlay_route = route;
    }
}

fn match_calibration(
    gpu: &Gpu,
    bundle: &Qwen4Bundle,
    route: &str,
    mode: NpuTrunkMode,
    firmware: Option<&str>,
) -> Result<
    (
        coproc::Calibration,
        coproc::CalKey,
        &'static str,
        FamilyMask,
    ),
    String,
> {
    let (mut cal, source) =
        trunk::load_calibration(hipfire_config::npu_trunk_calibration_path(), route)
            .map_err(|e| format!("calibration unavailable: {e}"))?;
    let index = bundle
        .npu_model_index
        .as_ref()
        .ok_or("model index identity unavailable")?
        .as_ref()
        .map_err(Clone::clone)?;
    let pci = gpu
        .hip
        .device_pci_bus_id(gpu.device_id)
        .map_err(|e| e.to_string())?;
    let mut key = match firmware {
        Some(firmware) => {
            trunk::runtime_key_with_firmware(&gpu.arch, &pci, firmware, index, route)?
        }
        None => trunk::runtime_calibration_key(&gpu.arch, &pci, index, route)?,
    };
    // A user calibration file stays fully strict; the built-in one binds on the hard key fields
    // (arch, build, model, epilogue, route) and takes this host's GPU PCI address, NPU firmware and
    // kernel release.
    if source == "builtin" {
        let (kernel, firmware, bdf) = (
            cal.key.kernel.uname_release.clone(),
            cal.key.device.npu_fw_version.clone(),
            cal.key.device.pci_bdf.clone(),
        );
        if !cal.bind_builtin(&key)?.is_empty() {
            eprintln!("npu-trunk: built-in Halo calibration (measured on kernel {kernel}, NPU firmware {firmware}, GPU {bdf}) applies by arch/model/route; this host kernel {}, NPU firmware {}, GPU {}",
                key.kernel.uname_release, key.device.npu_fw_version, key.device.pci_bdf);
        }
    }
    let families = if mode == NpuTrunkMode::Auto {
        coproc::auto_admission(&FLASH_NEXT, &cal, &key)?
    } else {
        cal.key_matches(&key)?;
        coproc::select_families(&FLASH_NEXT, &cal).0
    };
    Ok((cal, key, source, families.into()))
}

fn pack_stock(
    mode: NpuTrunkMode,
    raw: Vec<(Shape, Vec<(usize, Vec<u8>)>)>,
    families: FamilyMask,
) -> Result<Vec<(Shape, trunk::PackedB, Vec<u8>)>, Qwen4GpuForwardError> {
    let refs: Vec<_> = raw
        .iter()
        .map(|(shape, mats)| trunk::LayerTrunkRefs {
            shape: *shape,
            matrices: shape
                .column_map(FamilyMask::ALL)
                .parts
                .iter()
                .zip(mats)
                .map(|(&(family, _, _), (m, bytes))| trunk::MatrixRef {
                    family,
                    mq6: bytes,
                    m: *m,
                    k: 2560,
                })
                .collect(),
        })
        .collect();
    let packed_trunk =
        trunk::pack_trunk_with_contract(EpilogueContract::Bf16Direct, &refs, families)
            .map_err(fail)?;
    drop(refs);
    let mut packed = Vec::with_capacity(raw.len());
    for (entry, (shape, mats)) in packed_trunk.per_layer.into_iter().zip(raw) {
        let b = entry.map(|(_, b)| b).unwrap_or_else(|| trunk::PackedB {
            panels: Vec::new(),
            ws: Vec::new(),
            scales: Vec::new(),
            map: shape.column_map(families),
        });
        let mut codes = if mode == NpuTrunkMode::Emulate {
            vec![0u8; b.map.padded_cols() * 2560]
        } else {
            Vec::new()
        };
        if mode == NpuTrunkMode::Emulate {
            let full = shape.column_map(FamilyMask::ALL);
            for &(family, col, width) in &b.map.parts {
                let i = full
                    .parts
                    .iter()
                    .position(|&(f, _, _)| f == family)
                    .expect("shape family");
                let (m, bytes) = &mats[i];
                debug_assert_eq!(*m, width as usize);
                trunk::pack_matrix_direct(
                    bytes,
                    *m,
                    2560,
                    &mut codes[col as usize * 2560..(col as usize + m) * 2560],
                );
            }
        }
        packed.push((shape, b, codes));
    }
    Ok(packed)
}

impl NpuForward {
    pub(crate) fn prepare_stock(
        gpu: &mut Gpu,
        bundle: &Qwen4Bundle,
        pre_overlay: bool,
        rows: usize,
        scratch_mode: Qwen4ForwardScratchMode,
    ) -> Result<Option<PreparedTrunk>, Qwen4GpuForwardError> {
        let mode = NpuTrunkMode::from_env().map_err(fail)?.resolve(&gpu.arch);
        if mode.is_off() {
            return Ok(None);
        }
        let result = trunk::admission::prepare_after_checks(
            || {
                if *bundle.weights.band_scope()
                    != crate::weights::Qwen4BandScope::single(&bundle.config)
                {
                    return Err(fail("requires the whole-model scope; the NPU trunk refuses pipeline-parallel bands"));
                }
                if gpu.arch != "gfx1151"
                    || !bundle.config.profile.is_flash_next()
                    || bundle.config.hidden_size != 2560
                {
                    return Err(fail("requires Flash-Next family B, K=2560, gfx1151"));
                }
                if mode != NpuTrunkMode::Emulate
                    && (rows != 8192
                        || scratch_mode != Qwen4ForwardScratchMode::Full
                        || bundle.weights.ep_placement().is_some())
                {
                    return Err(fail(
                        "requires full 8192-row scratch and single-device/no-EP placement",
                    ));
                }
                // Validate every source before any download, packing or NPU allocation.
                for l in &bundle.weights.layer_refs {
                    let (refs, widths) = if let Some(g) = &l.gdn {
                        (
                            [&g.qkv, &g.z, &g.in_proj_a, &g.in_proj_b],
                            [10240, 6144, 48, 48],
                        )
                    } else if let Some(q) = &l.attention {
                        ([&q.q, &q.k, &q.v, &q.indexer_qk], [12288, 512, 512, 640])
                    } else {
                        return Err(fail("layer has no trunk projections"));
                    };
                    for (reference, width) in refs.into_iter().zip(widths) {
                        let w = bundle.weights.resident(reference)?;
                        if w.dtype != DType::MQ6G256V2
                            || reference.shape.as_slice() != [width, 2560]
                        {
                            return Err(fail(format!(
                                "{} requires stock MQ6G256V2 [{width},2560]",
                                reference.name
                            )));
                        }
                    }
                }
                // The planner prices the map's layer counts and K: they must be this model's.
                let kinds: Vec<usize> = bundle
                    .weights
                    .layer_refs
                    .iter()
                    .map(|l| usize::from(l.gdn.is_none()))
                    .collect();
                FLASH_NEXT
                    .check_model(&kinds, bundle.config.hidden_size)
                    .map_err(fail)?;
                if mode == NpuTrunkMode::Auto {
                    // Reject the memory rung downgrade before downloading stock rows.
                    let (free, _) = gpu.device_mem_info()?;
                    let required = qwen4_forward_device_bytes(&bundle.config, rows, scratch_mode)
                        .and_then(|bytes| bytes.checked_add(QWEN4_FORWARD_HEADROOM_BYTES));
                    if required.is_none_or(|bytes| bytes > free as u64) {
                        return Err(fail(
                            "8192-row full scratch does not fit the single-device memory rung",
                        ));
                    }
                }
                Ok(())
            },
            || {
                if mode == NpuTrunkMode::Auto {
                    trunk::npu_firmware().map(Some).map_err(fail)
                } else {
                    Ok(None)
                }
            },
            |firmware| {
                let route = bundle
                    .trunk_overlay
                    .as_ref()
                    .map(|overlay| overlay.default)
                    .or(bundle.config.trunk_iu4_default);
                if mode != NpuTrunkMode::Auto {
                    return Ok((None, FamilyMask::ALL, route));
                }
                let trunk_route = route.map_or_else(
                    || "stock".to_owned(),
                    |default| format!("overlay:{}", default.marker()),
                );
                let (cal, key, source, _) =
                    match_calibration(gpu, bundle, &trunk_route, mode, firmware.as_deref())
                        .map_err(fail)?;
                Ok((Some((cal, key, source)), FamilyMask::ALL, route))
            },
            |(matched, families, overlay_route)| {
                let started = Instant::now();
                let mut raw = Vec::with_capacity(bundle.weights.layer_refs.len());
                for l in &bundle.weights.layer_refs {
                    let (shape, refs) = if let Some(g) = &l.gdn {
                        (Shape::GdnIn, [&g.qkv, &g.z, &g.in_proj_a, &g.in_proj_b])
                    } else if let Some(q) = &l.attention {
                        (Shape::QsaIn, [&q.q, &q.k, &q.v, &q.indexer_qk])
                    } else {
                        unreachable!("eligibility checked");
                    };
                    let mut mats = Vec::with_capacity(4);
                    for (reference, &family) in refs.into_iter().zip(shape.families()) {
                        let m = reference.shape[0];
                        let bytes = if mode != NpuTrunkMode::Auto || families.contains(family) {
                            let w = bundle.weights.resident(reference)?;
                            let mut bytes = vec![0; m * 2560 / 256 * 200];
                            gpu.hip.memcpy_dtoh(&mut bytes, &w.buf)?;
                            bytes
                        } else {
                            Vec::new()
                        };
                        mats.push((m, bytes));
                    }
                    raw.push((shape, mats));
                }
                let download_ms = started.elapsed().as_secs_f64() * 1000.0;
                let packing_started = Instant::now();
                let layers = if mode == NpuTrunkMode::Auto {
                    PreparedLayers::Packed(pack_stock(mode, raw, families)?)
                } else {
                    PreparedLayers::Stock(raw)
                };
                let pack_ms = packing_started.elapsed().as_secs_f64() * 1000.0;
                Ok(Some(PreparedTrunk {
                    mode,
                    layers,
                    download_ms,
                    pack_ms,
                    pre_overlay,
                    overlay_route,
                    matched,
                    families,
                }))
            },
        );
        trunk::admission::finish(mode, result)
    }

    pub(super) fn load(
        gpu: &mut Gpu,
        bundle: &Qwen4Bundle,
        scratch: &Qwen4GpuForwardScratch,
        rows: usize,
        scratch_mode: Qwen4ForwardScratchMode,
        prepared: Option<Option<PreparedTrunk>>,
    ) -> Result<Option<Self>, Qwen4GpuForwardError> {
        let mode = NpuTrunkMode::from_env().map_err(fail)?.resolve(&gpu.arch);
        if mode.is_off() {
            return Ok(None);
        }
        // Eligibility was settled in `prepare_stock`; anything failing here failed to arm a qualifying model.
        trunk::admission::finish(
            mode,
            Self::load_admitted(gpu, bundle, scratch, rows, scratch_mode, prepared)
                .map_err(trunk::admission::Refusal::Failed),
        )
    }

    fn load_admitted(
        gpu: &mut Gpu,
        bundle: &Qwen4Bundle,
        scratch: &Qwen4GpuForwardScratch,
        rows: usize,
        scratch_mode: Qwen4ForwardScratchMode,
        prepared: Option<Option<PreparedTrunk>>,
    ) -> Result<Option<Self>, Qwen4GpuForwardError> {
        let Some(prepared) = (match prepared {
            Some(prepared) => prepared,
            None => Self::prepare_stock(gpu, bundle, false, rows, scratch_mode)?,
        }) else {
            return Ok(None);
        };
        let PreparedTrunk {
            mode,
            layers,
            download_ms,
            mut pack_ms,
            pre_overlay,
            overlay_route,
            matched,
            mut families,
        } = prepared;
        let sample_log =
            hipfire_config::developer_bool("HIPFIRE_QWEN4_NPU_TRUNK_SAMPLE_LOG", false);
        let family_override = hipfire_config::developer_var("HIPFIRE_QWEN4_NPU_TRUNK_FAMILIES")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .map(|value| parse_family_override(&value))
            .transpose()
            .or_else(|reason| {
                if mode == NpuTrunkMode::Auto {
                    eprintln!("npu-trunk: {reason}; ignoring override");
                    Ok(None)
                } else {
                    Err(fail(reason))
                }
            })?;
        let trunk_route = overlay_route.map_or_else(
            || "stock".to_owned(),
            |default| format!("overlay:{}", default.marker()),
        );
        let matched = if mode == NpuTrunkMode::Auto {
            matched
        } else {
            match match_calibration(gpu, bundle, &trunk_route, mode, None) {
                Ok((cal, key, source, selected)) => {
                    families = selected;
                    Some((cal, key, source))
                }
                Err(reason) => {
                    eprintln!("npu-trunk: mode={mode} calibration=none (forced) reason={reason}");
                    None
                }
            }
        };
        let (calibration, n_star, cal_key, cal_source) = match matched {
            Some((cal, mut key, source)) => {
                key.trunk_route.clone_from(&trunk_route);
                cal.key_matches(&key).map_err(fail)?;
                let n_star = coproc::crossover(&FLASH_NEXT, &cal, families.into());
                (Some(cal), n_star, Some(key), source)
            }
            None => (None, None, None, "none"),
        };
        if families.is_empty() {
            return Ok(None);
        }
        let physical = mode != NpuTrunkMode::Emulate;
        if physical
            && (rows != 8192
                || scratch_mode != Qwen4ForwardScratchMode::Full
                || bundle.weights.ep_placement().is_some())
        {
            return Err(fail(
                "requires two full 8192-row banks, full scratch and single-device placement",
            ));
        }
        let packing_started = Instant::now();
        let packed = match layers {
            PreparedLayers::Packed(packed) => packed,
            PreparedLayers::Stock(raw) => pack_stock(mode, raw, families)?,
        };
        pack_ms += packing_started.elapsed().as_secs_f64() * 1000.0;
        let started = Instant::now();
        let mut allocations = Vec::new();
        let result = (|| {
            let mut layers = Vec::new();
            let mut resident = Vec::new();
            let mut npu_panels_bytes = 0usize;
            let mut gpu_twin_bytes = 0usize;
            for (index, (shape, b, codes)) in packed.into_iter().enumerate() {
                let twin = if !physical && !b.map.parts.is_empty() {
                    let sidecars: Vec<_> = b
                        .scales
                        .iter()
                        .flat_map(|(m, e)| m.to_le_bytes().into_iter().chain(e.to_le_bytes()))
                        .collect();
                    let w = gpu.upload_raw(&codes, &[codes.len()])?;
                    allocations.push(w.sub_offset(0, w.numel()));
                    let scales = gpu.upload_raw(&sidecars, &[sidecars.len()])?;
                    allocations.push(scales.sub_offset(0, scales.numel()));
                    gpu_twin_bytes += codes.len() + sidecars.len();
                    Some(GpuTwin { w, scales })
                } else {
                    None
                };
                if physical && !b.panels.is_empty() {
                    npu_panels_bytes += b.panels.iter().map(Vec::len).sum::<usize>();
                    resident.push(LayerPanels {
                        layer: index,
                        shape: transport_shape(shape),
                        panels: b.panels,
                    });
                }
                layers.push(NpuLayer {
                    shape,
                    map: b.map,
                    twin,
                });
            }
            let emulation = if physical {
                None
            } else {
                // A geometry depends only on rows and K, not selected columns.
                let a_bytes = NpuBf16Layout {
                    padded_rows: rows.div_ceil(256) * 256,
                    k: 2560,
                    panels: 1,
                    n512: false,
                }
                .a_bytes();
                let a = gpu.zeros(&[a_bytes], DType::Raw)?;
                allocations.push(a.sub_offset(0, a.numel()));
                let cols = layers
                    .iter()
                    .map(|layer| layer.map.padded_cols())
                    .max()
                    .unwrap_or(0)
                    .max(1);
                let sums = gpu.zeros(&[rows * cols * 4], DType::Raw)?;
                allocations.push(sums.sub_offset(0, sums.numel()));
                Some(EmulationScratch { a, sums })
            };
            let d0 = gpu.zeros(&[rows * 4], DType::Raw)?;
            allocations.push(d0.sub_offset(0, d0.numel()));
            let d1 = gpu.zeros(&[rows * 4], DType::Raw)?;
            allocations.push(d1.sub_offset(0, d1.numel()));
            let resident_bytes = npu_panels_bytes + gpu_twin_bytes;
            let trace = hipfire_config::developer_var("HIPFIRE_QWEN4_NPU_TRUNK_TRACE")
                .ok()
                .filter(|s| !s.is_empty())
                .map(|path| {
                    std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(path)
                        .map(std::io::BufWriter::new)
                        .map_err(fail)
                })
                .transpose()?;
            pack_ms += started.elapsed().as_secs_f64() * 1000.0;
            let mut gpu_pci: Option<String> = None;
            let runtime = if physical {
                let programs =
                    trunk::build_rowmajor_programs_for_families(8192, families).map_err(fail)?;
                let programs = programs
                    .per_shape
                    .into_iter()
                    .filter(|p| p.panel_count != 0)
                    .map(|p| PanelProgram {
                        shape: transport_shape(p.shape),
                        pdi: p.design.pdi,
                        txn: p.design.insts,
                        padded_rows: 8192,
                        pitch: p.padded_cols,
                    })
                    .collect();
                let pci = gpu.hip.device_pci_bus_id(gpu.device_id)?;
                // Row-adaptive chains: a bank of fewer rows runs the program of its own wave count (the cores' wave loop
                // count is in the PDI), built on the relay thread the first time that row count is kicked.
                let cols = [Shape::GdnIn, Shape::QsaIn]
                    .map(|shape| shape.column_map(families).padded_cols());
                let builder: TxnBuilder = std::sync::Arc::new(move |shape, rows| {
                    use railgun::npu::trunk::Shape as T;
                    let (shape, cols) = match shape {
                        T::GdnIn => (Shape::GdnIn, cols[0]),
                        T::QsaIn => (Shape::QsaIn, cols[1]),
                        T::GdnOut => (Shape::GdnOut, 0),
                        T::QsaO => (Shape::QsaO, 0),
                        T::DenseGdnIn | T::DenseFaIn => {
                            return Err(format!("{shape:?} is not a Flash-Next trunk shape"))
                        }
                    };
                    trunk::build_program_for_rows(shape, cols, rows)
                        .map(|(pdi, txn)| PanelBuild { pdi, txn })
                });
                gpu_pci = Some(pci.clone());
                // Developer fault injection: force the chain watchdog of the Nth relay chain (the real NPU-timeout path).
                let fault_chain = hipfire_config::developer_var("HIPFIRE_QWEN4_NPU_TRUNK_FAULT_CHAIN").ok().filter(|v| !v.trim().is_empty()).and_then(|v| match v.trim().parse::<u64>() {
                    Ok(n) if n > 0 => Some(n),
                    _ => { eprintln!("npu-trunk: HIPFIRE_QWEN4_NPU_TRUNK_FAULT_CHAIN={v:?} ignored (expects a chain number >= 1)"); None }
                });
                let opened = NpuTrunkRuntime::open_with_builder(
                    programs,
                    resident,
                    trunk::TRUNK_BANKS,
                    trunk::TRUNK_ROWS_MAX,
                    &pci,
                    OpenOptions {
                        fault_chain,
                        ..OpenOptions::default()
                    },
                    Some(builder),
                )
                .map_err(fail)?;
                // Fail closed on the residual #926 shape: `open_hip` reuses an
                // already-mapped runtime by SONAME, so a second runtime only
                // enters through a pre-existing mapping of another install. Two
                // coexisting runtimes do not order their events against each
                // other, so paired prefill would read stale panels; refuse
                // instead. Under `auto` the admission barrier renders this as one
                // WARN line and a GPU-only load; under `on` it fails the load.
                if let Some(reason) = two_hip_runtime_reason(
                    opened.hip_event_record_address(),
                    gpu.hip.event_record_address(),
                    opened.hip_mapped_library().as_deref(),
                    gpu.hip.mapped_hip_library().as_deref(),
                ) {
                    return Err(fail(reason));
                }
                Some(opened)
            } else {
                None
            };
            let arenas = runtime
                .as_ref()
                .map(|rt| -> Result<_, Qwen4GpuForwardError> {
                    Ok([
                        [
                            registered_c_view(rt, 0, Shape::GdnIn, families)?,
                            registered_c_view(rt, 0, Shape::QsaIn, families)?,
                        ],
                        [
                            registered_c_view(rt, 1, Shape::GdnIn, families)?,
                            registered_c_view(rt, 1, Shape::QsaIn, families)?,
                        ],
                    ])
                })
                .transpose()?;
            // `emulate` runs the same pair executor as `on` (twin projection
            // instead of pack + kick), so it needs bank B too; its in-projection
            // outputs land in scratch, so bank B keeps private copies of them.
            let pairable = physical
                || (rows == 8192
                    && scratch_mode == Qwen4ForwardScratchMode::Full
                    && bundle.weights.ep_placement().is_none());
            let (bank_b, host_tokens, host_ple) = if pairable {
                let (scratch, ids, ple) =
                    Qwen4GpuForwardScratch::new_npu_bank(gpu, &bundle.config, scratch, !physical)?;
                (Some(scratch), ids, ple)
            } else {
                (None, Vec::new(), Vec::new())
            };
            let zab_source = if pre_overlay {
                "stock-MQ6-DtoH-before-overlay"
            } else {
                "stock-MQ6-DtoH"
            };
            let bank_b_bytes = match bank_b.as_ref() {
                Some(bank) => Qwen4GpuForwardScratch::npu_bank_plan_bytes(&bank.slots, !physical)?,
                None => 0,
            };
            if let (Some(cal), Some(key)) = (&calibration, &cal_key) {
                eprintln!(
                    "{} cal_source={cal_source} host=[{}]",
                    coproc::load_log_line(
                        &FLASH_NEXT,
                        cal,
                        &coproc::LoadLogKey {
                            current: key,
                            mode: mode.as_str(),
                            families: families.into(),
                        },
                        n_star,
                        &coproc::plan(&FLASH_NEXT, cal, families.into(), 16384),
                        &coproc::plan(&FLASH_NEXT, cal, families.into(), 65536)
                    ),
                    trunk::host_facts(gpu_pci.as_deref())
                );
            }
            eprintln!("npu-trunk: mode={mode} families={:#x} contract=bf16_direct trunk_route={trunk_route} zab_source={zab_source} pack_ms={pack_ms:.3} download_ms={download_ms:.3} resident_bytes={resident_bytes} npu_panels_bytes={npu_panels_bytes} gpu_twin_bytes={gpu_twin_bytes} bank_b_bytes={bank_b_bytes} runtime={} raw=DtoH(no-retained-host-copy)", families.bits(), if runtime.is_some() { "open" } else { "not-opened" });
            hipfire_config::load_summary::note(
                hipfire_config::load_summary::Facet::Prefill,
                format!("npu trunk mode={mode} route={trunk_route}"),
            );
            let rates = (mode == NpuTrunkMode::Auto && runtime.is_some()).then(|| {
                coproc::RateEstimator::new(calibration.as_ref().expect("auto calibration").clone())
            });
            let sampling = if rates.is_some() {
                Some(PairSampling::new(gpu, layers.len())?)
            } else {
                None
            };
            let request_families =
                intersect_families(family_override.unwrap_or(families), families);
            Ok(Some(Self {
                mode,
                families,
                request_families,
                family_override,
                rates,
                sampling,
                request_epoch: 0,
                request_gpu: false,
                guard_tripped: false,
                sample_log,
                request_cal: None,
                layers,
                emulation,
                scales: [d0, d1],
                bank_b,
                runtime,
                arenas,
                trace,
                host_tokens,
                host_ple,
                pair_logged: false,
                calibration,
                n_star_cache: None,
                request_plan: None,
                request_origin: 0,
                adaptive: Vec::new(),
                latch: trunk::containment::FaultLatch::default(),
                gpu_pci,
            }))
        })();
        if result.is_err() {
            for tensor in allocations {
                let _ = gpu.free_tensor(tensor);
            }
        }
        result
    }
    /// Pairs run layer-major through the pair executor: `on`/`auto` on the NPU,
    /// `emulate` with the GPU twin projection in its place.
    pub(super) fn pipelined(&self) -> bool {
        self.bank_b.is_some()
    }
    /// GPU-only scheduling: either the process fault latch is set or this request is diagnostic GPU-only.
    pub(super) fn route_off(&self) -> bool {
        self.latch.tripped() || self.request_gpu
    }
    /// Whether the request in flight faulted; clears the per-request flag.
    pub(super) fn take_request_fault(&mut self) -> bool {
        self.latch.take_request_fault()
    }
    /// A pair executor error: release every outstanding GPU wait (all hardware modes) and, under `auto`, latch the route
    /// off for the process. An NPU runtime fault logs one WARN and marks the request for its GPU-only rerun; a GPU HipError
    /// (sticky context: a rerun could only fail) logs nothing here and the request fails with its original error. `on`
    /// keeps failing closed (the runtime's sticky failure); `emulate` has no runtime.
    pub(super) fn contain_fault(&mut self, error: &Qwen4GpuForwardError) {
        use trunk::containment::FaultClass;
        let class = FaultClass::of(error);
        let reason = format!("{}: {error}", class.as_str());
        let Some(rt) = &self.runtime else {
            return;
        };
        rt.abort(&reason);
        if self.mode == NpuTrunkMode::Auto
            && self.latch.trip(class, &reason)
            && class == FaultClass::NpuRuntime
        {
            eprintln!("WARN npu-trunk: NPU pair fault, the route is off for the rest of this process and prefill runs GPU-only ({reason}); host {}", trunk::host_facts(self.gpu_pci.as_deref()));
        }
    }
    pub(super) fn begin_request(&mut self, gpu: &Gpu) -> Result<(), Qwen4GpuForwardError> {
        self.request_gpu = trunk::diagnostic_gpu_route();
        self.pair_logged = false;
        self.request_plan = None;
        self.latch.begin_request();
        if self.latch.tripped() {
            return Ok(());
        }
        self.poll(gpu)?;
        self.request_epoch = self.request_epoch.wrapping_add(1);
        self.guard_tripped = false;
        Ok(())
    }
    /// Schedule the request begun last GPU-only (the `off` schedule), for a
    /// request the pair executor cannot serve; `why` is logged once. Like the
    /// diagnostic route it never touches the fault latch.
    pub(super) fn force_gpu_request(&mut self, why: &str) {
        if !self.request_gpu && !self.pair_logged {
            eprintln!("npu-trunk: route=gpu ({why})");
            self.pair_logged = true;
        }
        self.request_gpu = true;
    }
    /// With a matched calibration every mode (`auto`, `on`, `emulate`) executes the planner's pairs, including the
    /// rebalanced last pair; `auto` logs the plan. Without one the forced modes keep the natural 8192 + B pairing.
    pub(super) fn plan_request(
        &mut self,
        gpu: &Gpu,
        rows: usize,
        position: usize,
    ) -> Result<(), Qwen4GpuForwardError> {
        if self.request_gpu {
            if !self.pair_logged {
                eprintln!("npu-trunk: rows={rows} route=gpu (diagnostic)");
                self.pair_logged = true;
            }
            return Ok(());
        }
        if self.latch.tripped() {
            return Ok(());
        }
        self.poll(gpu)?;
        if self.request_plan.is_some() {
            return Ok(());
        }
        let snapshot = self.rates.as_ref().map(|rates| rates.snapshot());
        let cal = snapshot
            .map(|snapshot| &snapshot.calibration)
            .or(self.calibration.as_ref());
        let selected = self.family_override.unwrap_or_else(|| match cal {
            Some(cal) if self.mode == NpuTrunkMode::Auto => FamilyMask::from(
                coproc::choose_families(&FLASH_NEXT, cal, self.families.into(), rows),
            ),
            Some(cal) => coproc::select_families(&FLASH_NEXT, cal).0.into(),
            None => self.families,
        });
        self.request_families = intersect_families(selected, self.families);
        let Some(cal) = cal else {
            return Ok(());
        };
        let families = self.request_families;
        let generation = snapshot.map_or(0, |snapshot| snapshot.generation);
        let n_star = if self.mode == NpuTrunkMode::Auto {
            match self.n_star_cache {
                Some((g, mask, star)) if g == generation && mask == families => star,
                _ => {
                    let star = coproc::crossover(&FLASH_NEXT, cal, families.into());
                    self.n_star_cache = Some((generation, families, star));
                    star
                }
            }
        } else {
            Some(0)
        };
        let executable = admitted_plan(
            request_plan(cal, self.mode, families, n_star, rows),
            &mut self.adaptive,
            |planned| {
                coproc::executable_plan_for(
                    &FLASH_NEXT,
                    cal,
                    families.into(),
                    planned,
                    rows,
                    armed_from(self.mode, n_star),
                )
            },
        );
        if self.mode == NpuTrunkMode::Auto {
            let line = coproc::request_log_line(&FLASH_NEXT, rows, &executable);
            let rest = line
                .strip_prefix(&format!("npu-trunk: rows={rows}"))
                .unwrap_or(&line);
            let (gpu_scale, npu_scale, frozen) =
                snapshot.map_or((1.0, 1.0, false), |s| (s.gpu_scale, s.npu_scale, s.frozen));
            eprintln!("npu-trunk: rows={rows} families={}{} rates=gen{generation} gpu_x{gpu_scale:.3} npu_x{npu_scale:.3}{}",
                family_names(families), rest, if frozen { " frozen" } else { "" });
        }
        self.request_cal =
            (self.sample_log && self.mode == NpuTrunkMode::Auto).then(|| (generation, cal.clone()));
        self.request_origin = position;
        self.request_plan = Some(executable);
        Ok(())
    }
    /// Rows of banks A and B of the pair starting at `position`: the plan's
    /// split, or (forced modes without a matched calibration) the natural
    /// 8192 + B. `None` without a bank-B scratch (an `emulate` load that cannot
    /// hold one runs every chunk native).
    pub(super) fn selected_pair(
        &self,
        position: usize,
        remaining: usize,
        chunk: usize,
    ) -> Option<[usize; 2]> {
        if self.route_off() {
            return None;
        }
        self.bank_b.as_ref()?;
        if self.guard_tripped || self.request_families.is_empty() {
            return None;
        }
        if self.calibration.is_none() {
            return pair_rows(remaining, chunk).map(|rows| [chunk, rows - chunk]);
        }
        planned_banks(
            self.request_plan.as_ref()?,
            self.request_origin,
            position,
            remaining,
            chunk,
        )
    }
    pub(super) fn note_pair(&mut self, rows: [usize; 2], position: usize) {
        if !self.pair_logged && self.mode != NpuTrunkMode::Auto {
            eprintln!(
                "npu-trunk: pair rows={}+{} position={position} families={}",
                rows[0],
                rows[1],
                family_names(self.request_families)
            );
            self.pair_logged = true;
        }
    }
    pub(super) fn free(self, gpu: &mut Gpu) -> Result<(), hip_bridge::HipError> {
        let mut first = None;
        if let Some(rt) = self.runtime {
            if let Err(e) = rt.close() {
                first = Some(hip_bridge::HipError::new(1, &e));
            }
        }
        if let Some(sampling) = self.sampling {
            for set in sampling.sets {
                for event in set.events {
                    if let Err(e) = gpu.hip.event_destroy(event) {
                        first.get_or_insert(e);
                    }
                }
            }
        }
        if let Some(b) = self.bank_b {
            if let Some(e) = b.free_gpu(gpu) {
                first.get_or_insert(e);
            }
        }
        let twins = self
            .layers
            .into_iter()
            .filter_map(|layer| layer.twin)
            .flat_map(|twin| [twin.w, twin.scales]);
        let emulation = self
            .emulation
            .into_iter()
            .flat_map(|scratch| [scratch.a, scratch.sums]);
        for t in twins.chain(emulation).chain(self.scales) {
            if let Err(e) = gpu.free_tensor(t) {
                first.get_or_insert(e);
            }
        }
        first.map_or(Ok(()), Err)
    }
    fn qkv_view(&self, gpu: &Gpu, step: &Step<'_>) -> Result<GpuTensor, Qwen4GpuForwardError> {
        match step {
            Step::GatedDeltaNet(op) => {
                let mut qkv = op.projection.sub_offset(0, op.rows * 10240);
                // Match the core's native projection dtype, including chunk-off.
                if gdn_qkv_projection_bf16(gpu, op) {
                    qkv.dtype = DType::BF16;
                }
                Ok(qkv)
            }
            Step::IndexedAttention(op) => Ok(op.qgate_scratch.sub_offset(0, op.rows * 12288)),
            _ => Err(fail("expected mixer step")),
        }
    }
    fn shape_map(&self, shape: Shape) -> &trunk::ColumnMap {
        &self
            .layers
            .iter()
            .find(|l| l.shape == shape)
            .expect("Flash-Next shape")
            .map
    }
    fn destinations<'a>(
        &self,
        step: &'a Step<'_>,
        qkv: &'a GpuTensor,
    ) -> Result<SmallVec<[NpuTrunkOutput<'a>; 4]>, Qwen4GpuForwardError> {
        let (shape, targets): (Shape, [(Family, &GpuTensor); 4]) = match step {
            Step::GatedDeltaNet(op) => (
                Shape::GdnIn,
                [
                    (Family::GdnQkv, qkv),
                    (Family::GdnZ, op.z_output),
                    (Family::GdnA, op.a),
                    (Family::GdnB, op.b),
                ],
            ),
            Step::IndexedAttention(op) => (
                Shape::QsaIn,
                [
                    (Family::QsaQ, op.qgate_scratch),
                    (Family::QsaK, op.k_scratch),
                    (Family::QsaV, op.v_scratch),
                    (Family::QsaIdx, op.index_scratch),
                ],
            ),
            _ => return Err(fail("expected mixer")),
        };
        let map = self.shape_map(shape);
        targets
            .into_iter()
            .filter(|(f, _)| self.request_families.contains(*f))
            .map(|(f, t)| map.gpu_output(f, t).map_err(Into::into))
            .collect()
    }
    fn pack(
        &self,
        gpu: &mut Gpu,
        step: &Step<'_>,
        bank: usize,
        pipe: bool,
        prep_end: Option<&hip_bridge::Event>,
    ) -> Result<u64, Qwen4GpuForwardError> {
        let (input, rows, shape) = match step {
            Step::GatedDeltaNet(op) => (op.input, op.rows, Shape::GdnIn),
            Step::IndexedAttention(op) => (op.input, op.rows, Shape::QsaIn),
            _ => return Err(fail("pack expected mixer")),
        };
        let map = self.shape_map(shape);
        let x = gpu.rotate_x_mq_batched_f16(input, 2560, rows)?;
        if let Some(event) = prep_end {
            gpu.hip.event_record(event, gpu.active_stream.as_ref())?;
        }
        let pack_issue_ns = if pipe && self.trace.is_some() {
            monotonic_ns()
        } else {
            0
        };
        if pipe {
            let rt = self
                .runtime
                .as_ref()
                .ok_or_else(|| fail("runtime missing"))?;
            // The A stream is laid out for the bank's own wave count: the chain the relay runs for this kick has exactly that
            // many waves (`chain_rows` is the one decision both sides use).
            let padded = rt.chain_rows(transport_shape(shape), rows);
            unsafe {
                gpu.npu_trunk_pack_a_bf16_ptr(
                    &x,
                    rows,
                    layout(map, padded),
                    rt.a_arena(bank).map_err(fail)?.device as *mut c_void,
                    &self.scales[bank],
                )?;
            }
        } else {
            gpu.npu_trunk_pack_a_bf16(
                &x,
                rows,
                layout(map, rows),
                &self.emulation.as_ref().expect("emulate scratch").a,
                &self.scales[bank],
            )?;
        }
        Ok(pack_issue_ns)
    }
    fn emulate(
        &self,
        gpu: &mut Gpu,
        layer: usize,
        step: &Step<'_>,
    ) -> Result<(), Qwen4GpuForwardError> {
        self.pack(gpu, step, 0, false, None)?;
        let qkv = self.qkv_view(gpu, step)?;
        let rows = match step {
            Step::GatedDeltaNet(op) => op.rows,
            Step::IndexedAttention(op) => op.rows,
            _ => unreachable!(),
        };
        let outs = self.destinations(step, &qkv)?;
        let l = &self.layers[layer];
        let scratch = self.emulation.as_ref().expect("emulate scratch");
        let twin = l.twin.as_ref().expect("emulate twin");
        gpu.npu_trunk_emulate_bf16(
            &scratch.a,
            &twin.w,
            &twin.scales,
            &self.scales[0],
            &scratch.sums,
            layout(&l.map, rows),
            rows,
            l.map.padded_cols(),
            &outs,
        )?;
        Ok(())
    }
    fn uncovered(&self, gpu: &mut Gpu, step: &Step<'_>) -> Result<(), Qwen4GpuForwardError> {
        let covered = |families: [Family; 4]| families.map(|f| self.request_families.contains(f));
        match step {
            Step::GatedDeltaNet(op) => execute_gated_delta_net_uncovered(
                gpu,
                op,
                covered([Family::GdnQkv, Family::GdnA, Family::GdnB, Family::GdnZ]),
            )
            .map_err(fail),
            Step::IndexedAttention(op) => execute_indexed_attention_uncovered(
                gpu,
                op,
                covered([Family::QsaIdx, Family::QsaQ, Family::QsaK, Family::QsaV]),
            )
            .map_err(fail),
            _ => unreachable!("typed mixer"),
        }
    }
}

fn skips_input_projections(phase: MixerPhase) -> bool {
    matches!(phase, MixerPhase::Core | MixerPhase::OutProj)
}
/// `program[window]` of one bank, with the bank's HC pairings carried in `hc`
/// across its prepare / Core / OutProj windows.
fn execute(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    program: &[Step<'_>],
    window: std::ops::Range<usize>,
    hc: &mut HcFusionCursor,
) -> Result<(), Qwen4GpuForwardError> {
    // The dispatch guards project_trunk (MQ6 AND A4/IU4) with Whole|InProj.
    // Every mixer reaching this covered executor must therefore omit both.
    debug_assert!(
        program[window.clone()].iter().all(|step| match step {
            Step::GatedDeltaNet(op) => skips_input_projections(op.phase),
            Step::IndexedAttention(op) => skips_input_projections(op.phase),
            _ => true,
        }),
        "covered NPU/twin mixer must not launch native input projections"
    );
    execute_validated_steps_window(gpu, ctx, program, window, hc).map_err(fail)
}
fn phase(step: &mut Step<'_>, value: MixerPhase) {
    match step {
        Step::GatedDeltaNet(op) => op.phase = value,
        Step::IndexedAttention(op) => op.phase = value,
        _ => unreachable!(),
    }
}

#[path = "layers.rs"]
mod layers;
use railgun::npu::trunk::monotonic_ns;

struct TraceRow {
    layer: usize,
    bank: usize,
    position: usize,
    rows: usize,
    seq: u32,
    pack_issue_ns: u64,
    kick_ns: u64,
    submit_ns: u64,
    done_ns: u64,
    signal_write_ns: u64,
    wait_enqueue_ns: u64,
    consume_issue_ns: u64,
    start: hip_bridge::Event,
    end: hip_bridge::Event,
    prep_begin: hip_bridge::Event,
    prep_end: hip_bridge::Event,
    pack_end: hip_bridge::Event,
    wait_begin: hip_bridge::Event,
    wait_end: hip_bridge::Event,
}
/// Scalar JSONL schema, independent of HIP handles so CPU tests exercise the
/// same serializer as the after-pair event resolver. Existing keys stay ordered.
#[derive(Default, serde::Serialize)]
struct TraceRecord {
    layer: usize,
    bank: &'static str,
    position: usize,
    rows: usize,
    seq: u32,
    pack_issue_ns: u64,
    kick_ns: u64,
    npu_submit_ns: u64,
    npu_done_ns: u64,
    gpu_wait_enqueue_ns: u64,
    consume_issue_ns: u64,
    mixer_start_ns: u64,
    mixer_end_ns: u64,
    anchor_uncertainty_ns: u64,
    prep_begin_ns: u64,
    prep_end_ns: u64,
    pack_end_ns: u64,
    wait_begin_ns: u64,
    wait_end_ns: u64,
    signal_write_ns: u64,
}
impl TraceRecord {
    fn write(&self, out: &mut impl Write) -> std::io::Result<()> {
        serde_json::to_writer(&mut *out, self)?;
        writeln!(out)
    }
}
fn mixer_index(program: &layers::BankProgram<'_>, layer: usize) -> usize {
    program.ranges[layer]
        .clone()
        .find(|&i| {
            matches!(
                program.steps[i],
                Step::GatedDeltaNet(_) | Step::IndexedAttention(_)
            )
        })
        .expect("typed layer has a mixer")
}
fn commit_qsa(bundle: &mut Qwen4Bundle, commits: &[(usize, usize, usize, usize, usize)]) {
    for (slot, &(full, raw, pooled, selected, position)) in commits.iter().enumerate() {
        let state = bundle.state.qsa_mut(slot).expect("validated QSA slot");
        state.full_len = full;
        state.raw_len = raw;
        state.pooled_len = pooled;
        state.selected_len = selected;
        state.position = position;
        state.clamp_f16_valid();
    }
}

/// What one bank launches in a pair group: its place in the group, where its
/// Core and PLE step split at the session snapshot row (`cut`, bank-local), or
/// whether its whole-bank state is staged after it (`stage_after`: the cut is
/// the A/B split). The single source of the launch geometry for every NPU mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BankLaunch {
    offset: usize,
    rows: usize,
    cut: Option<usize>,
    stage_after: bool,
}

fn launch_geometry(shape: PairShape) -> [BankLaunch; 2] {
    let site = shape.site();
    std::array::from_fn(|bank| BankLaunch {
        offset: if bank == 0 { 0 } else { shape.rows[0] },
        rows: shape.rows[bank],
        cut: match site {
            CutSite::Bank {
                bank: cut_bank,
                row,
            } if cut_bank == bank => Some(row),
            _ => None,
        },
        stage_after: bank == 0 && site == CutSite::Split,
    })
}

/// The session snapshot staged inside a pair, and the launch geometry it follows.
#[derive(Clone, Copy)]
struct Snapshot<'a> {
    stage: &'a crate::bundle::PairSessionStage,
    geometry: [BankLaunch; 2],
}

/// A bank whose Core splits at its local `row`: the snapshot's mixer state is
/// copied between the two launches (this layer's slot among GDN / QSA states).
struct CoreCut<'a> {
    stage: &'a crate::bundle::PairSessionStage,
    row: usize,
    gdn_total: usize,
    gdn_slot: usize,
    qsa_slot: usize,
}

/// Stage a layer's mixer state at the snapshot row.
fn copy_mixer_state(
    gpu: &mut Gpu,
    stage: &crate::bundle::PairSessionStage,
    mixer: &Step<'_>,
    gdn_total: usize,
    gdn_slot: usize,
    qsa_slot: usize,
) -> Result<(), Qwen4GpuForwardError> {
    match mixer {
        Step::GatedDeltaNet(op) => stage
            .copy_parts(gpu, 2 * gdn_slot, [op.recurrent, op.conv_state])
            .map_err(fail),
        Step::IndexedAttention(op) => stage
            .copy(gpu, 2 * gdn_total + qsa_slot, op.state.selected_indices)
            .map_err(fail),
        _ => unreachable!("typed mixer"),
    }
}

/// Core launches of a `rows`-row bank: one, or two split at the snapshot row
/// `cut` (the state is staged between them, never after the second).
fn core_windows(rows: usize, cut: Option<usize>) -> SmallVec<[CoreWindow; 2]> {
    match cut {
        Some(row) => {
            debug_assert!(
                0 < row && row < rows,
                "a sub-launch cut is strictly inside its bank"
            );
            SmallVec::from_buf([
                CoreWindow { lo: 0, rows: row },
                CoreWindow {
                    lo: row,
                    rows: rows - row,
                },
            ])
        }
        None => SmallVec::from_iter([CoreWindow { lo: 0, rows }]),
    }
}

/// Two reusable timing sets. Events are explicitly destroyed by `free`; Event has no Drop.
struct PairSampling {
    sets: [SampleEvents; 2],
    next: usize,
    active: Option<usize>,
}
struct SampleEvents {
    // Three events per (layer, bank), followed by pair_start and pair_end.
    events: Vec<hip_bridge::Event>,
    pending: Option<PendingSample>,
}
struct PendingSample {
    rows: [usize; 2],
    prefixes: [usize; 2],
    mask: FamilyMask,
    covered: SmallVec<[bool; 128]>,
    /// Stage-map kind of each layer (0 = GDN, 1 = QSA).
    kinds: SmallVec<[usize; 64]>,
    epoch: u64,
    position: usize,
    npu_busy_ms: f64,
    planned: Option<(u64, coproc::Calibration)>,
}
impl PairSampling {
    fn new(gpu: &Gpu, layers: usize) -> Result<Self, Qwen4GpuForwardError> {
        let count = 2 * (6 * layers + 2);
        let mut events = Vec::with_capacity(count);
        for _ in 0..count {
            match gpu.hip.event_create() {
                Ok(event) => events.push(event),
                Err(error) => {
                    for event in events {
                        let _ = gpu.hip.event_destroy(event);
                    }
                    return Err(error.into());
                }
            }
        }
        let second = events.split_off(6 * layers + 2);
        Ok(Self {
            sets: [
                SampleEvents {
                    events,
                    pending: None,
                },
                SampleEvents {
                    events: second,
                    pending: None,
                },
            ],
            next: 0,
            active: None,
        })
    }
}
impl SampleEvents {
    fn pair_start(&self) -> &hip_bridge::Event {
        &self.events[self.events.len() - 2]
    }
    fn pair_end(&self) -> &hip_bridge::Event {
        &self.events[self.events.len() - 1]
    }
}

impl NpuForward {
    fn record_sample_event(
        &self,
        gpu: &Gpu,
        layer: usize,
        bank: usize,
        phase: usize,
    ) -> Result<(), Qwen4GpuForwardError> {
        if let Some(sampling) = &self.sampling {
            if let Some(active) = sampling.active {
                gpu.hip.event_record(
                    &sampling.sets[active].events[(layer * 2 + bank) * 3 + phase],
                    gpu.active_stream.as_ref(),
                )?;
            }
        }
        Ok(())
    }

    /// Query completed sets only: no synchronization, and no elapsed-time read until pair_end is complete.
    pub(super) fn poll(&mut self, gpu: &Gpu) -> Result<(), Qwen4GpuForwardError> {
        if self.latch.tripped() {
            return Ok(());
        }
        let (Some(sampling), Some(rates)) = (&mut self.sampling, &mut self.rates) else {
            return Ok(());
        };
        for set in &mut sampling.sets {
            if set.pending.is_none() || !gpu.hip.event_query(set.pair_end())? {
                continue;
            }
            let pending = set.pending.as_ref().expect("queried pending sample");
            // Chain index = layer * 2 + bank is also the run_bank visit order. The window of chain c (its pack_end ->
            // its wait_begin) contains run_bank(c - 1), whose own NPU wait (stall) is not GPU work: subtract it, so a slow
            // NPU never reads as a slow GPU.
            let mut windows_ms = 0.0;
            let mut stalls_ms = 0.0;
            let mut previous_stall = 0.0;
            for (chain, &covered) in pending.covered.iter().enumerate() {
                if !covered {
                    previous_stall = 0.0;
                    continue;
                }
                let events = &set.events[chain * 3..chain * 3 + 3];
                if chain != 0 {
                    windows_ms +=
                        gpu.hip.event_elapsed_ms(&events[0], &events[1])? as f64 - previous_stall;
                }
                let stall = gpu.hip.event_elapsed_ms(&events[1], &events[2])? as f64;
                stalls_ms += stall;
                previous_stall = stall;
            }
            let span_ms = gpu.hip.event_elapsed_ms(set.pair_start(), set.pair_end())? as f64;
            let pending = set.pending.take().expect("completed sample");
            let model = coproc::pair_model(
                &FLASH_NEXT,
                rates.prior(),
                coproc::Moved::of(&FLASH_NEXT, pending.mask.into()),
                pending.rows,
                pending.prefixes,
                &pending.kinds,
            );
            rates.observe(&coproc::RateSample {
                npu_measured_ms: pending.npu_busy_ms,
                npu_model_ms: model.npu_ms,
                window_measured_ms: windows_ms,
                window_model_ms: model.window_ms,
            });
            let snapshot = rates.snapshot();
            // The pair loses to GPU-only once the GPU's measured waits on the NPU exceed what pairing saves it. Both
            // sides avoid the model's prefix-dependent terms, so the model's error at depth cannot trip the guard.
            let savings = coproc::pair_model(
                &FLASH_NEXT,
                &snapshot.calibration,
                coproc::Moved::of(&FLASH_NEXT, pending.mask.into()),
                pending.rows,
                pending.prefixes,
                &pending.kinds,
            )
            .savings_ms;
            if self.sample_log || self.trace.is_some() {
                let moved = coproc::Moved::of(&FLASH_NEXT, pending.mask.into());
                let planned = pending.planned.as_ref().map_or_else(|| "planned=n/a".to_owned(), |(generation, cal)| {
                    let m = coproc::pair_model(&FLASH_NEXT, cal, moved, pending.rows, pending.prefixes, &pending.kinds);
                    let t = coproc::pair_time_moved(&FLASH_NEXT, cal, moved, pending.rows[0], pending.prefixes[0], pending.rows[1], pending.prefixes[1]);
                    format!("planned(gen{generation}) npu={:.1} window={:.1} wait={:.1} pair={:.1} gpu_only={:.1} saving={:.1}",
                        m.npu_ms, m.window_ms, t.stall + t.fill, t.t, m.gpu_only_ms, m.savings_ms)
                });
                eprintln!("npu-trunk: pair-sample row={} rows={}+{} families={} measured npu={:.1} window={windows_ms:.1} wait={stalls_ms:.1} pair={span_ms:.1} | {planned} | prior npu={:.1} window={:.1} | now gen{} gpu_x{:.3} npu_x{:.3}{}",
                    pending.position, pending.rows[0], pending.rows[1], family_names(pending.mask), pending.npu_busy_ms,
                    model.npu_ms, model.window_ms, snapshot.generation, snapshot.gpu_scale, snapshot.npu_scale,
                    if snapshot.frozen { " frozen" } else { "" });
            }
            if pending.epoch == self.request_epoch
                && !self.request_gpu
                && !self.guard_tripped
                && stalls_ms > savings
            {
                self.guard_tripped = true;
                eprintln!("npu-trunk: guard: pair at row {}: the GPU waited {stalls_ms:.0} ms on the NPU, more than the {savings:.0} ms pairing saves it (pair {span_ms:.0} ms; gpu_x{:.3} npu_x{:.3}); the rest of this request runs GPU-only",
                    pending.position, snapshot.gpu_scale, snapshot.npu_scale);
            }
        }
        Ok(())
    }
}

impl NpuForward {
    /// One Core launch over `window` of the bank's mixer; arenas only when the
    /// layer's projections came from the NPU.
    fn core_launch(
        &self,
        gpu: &mut Gpu,
        layer: usize,
        bank: usize,
        step: &Step<'_>,
        covered: bool,
        window: CoreWindow,
    ) -> Result<(), Qwen4GpuForwardError> {
        let shape = self.layers[layer].shape;
        // Fused arenas exist only under the NPU runtime; `emulate` Core reads scratch.
        let arenas = covered && self.arenas.is_some();
        let tensor = arenas.then(|| {
            self.arenas.as_ref().expect("on arena views")[bank][shape.index()]
                .as_ref()
                .expect("selected shape arena")
        });
        let map = &self.layers[layer].map;
        let family = |family| {
            tensor
                .filter(|_| self.request_families.contains(family))
                .and_then(|tensor| family_view(map, tensor, family))
        };
        match step {
            Step::GatedDeltaNet(op) => execute_gated_delta_net_core_rows(
                gpu,
                op,
                arenas.then(|| GdnTrunkArenas {
                    qkv: family(Family::GdnQkv),
                    z: family(Family::GdnZ),
                    a: family(Family::GdnA),
                    b: family(Family::GdnB),
                }),
                window,
            )
            .map_err(fail),
            Step::IndexedAttention(op) => execute_indexed_attention_core_rows(
                gpu,
                op,
                arenas.then(|| QsaTrunkArenas {
                    q: family(Family::QsaQ),
                    k: family(Family::QsaK),
                    v: family(Family::QsaV),
                    idx: family(Family::QsaIdx),
                }),
                window,
            )
            .map_err(fail),
            _ => unreachable!("typed layer mixer"),
        }
    }
    fn prepare(
        &self,
        gpu: &mut Gpu,
        ctx: &DispatchCtx,
        program: &layers::BankProgram<'_>,
        hc: &mut HcFusionCursor,
        layer: usize,
        bank: usize,
        records: &mut [TraceRow],
        snapshot: Option<Snapshot<'_>>,
    ) -> Result<Option<Seq>, Qwen4GpuForwardError> {
        let index = mixer_index(program, layer);
        let rt = self.runtime.as_ref();
        let covered = !self.guard_tripped
            && self.layers[layer]
                .map
                .parts
                .iter()
                .any(|&(family, _, _)| self.request_families.contains(family));
        if let Some(r) = records.get(layer * 2 + bank) {
            gpu.hip
                .event_record(&r.prep_begin, gpu.active_stream.as_ref())?;
        }
        if let (true, Some(rt)) = (covered, rt) {
            unsafe {
                rt.prepare_bank(bank, stream(gpu)).map_err(fail)?;
            }
            if let Some(r) = records
                .iter_mut()
                .take(layer * 2)
                .rev()
                .find(|r| r.bank == bank && r.seq != 0 && r.done_ns == 0)
            {
                let completion = rt.wait_host(Seq { bank, value: r.seq }).map_err(fail)?;
                r.submit_ns = completion.submit_ns;
                r.done_ns = completion.done_ns;
                r.signal_write_ns = completion.signal_write_ns;
            }
        }
        // Includes PLE (when present) and the attention HC read. Neither reads
        // recurrent/QSA state; bank B's input can be prepared ahead of bank A.
        let start = program.ranges[layer].start;
        let pre = &program.steps[start..index];
        let ple_cut = snapshot.and_then(|snapshot| {
            let launch = snapshot.geometry[bank];
            match (launch.cut, launch.stage_after) {
                (Some(row), _) => Some((snapshot.stage, Some(row))),
                (None, true) => Some((snapshot.stage, None)),
                _ => None,
            }
        });
        match ple_cut {
            None => execute(gpu, ctx, &program.steps, start..index, hc)?,
            Some((stage, row)) => {
                // The PLE depthwise convolution carries state: the snapshot holds
                // it at the snapshot row, so a cut inside this bank splits it too.
                let mut from = 0;
                for (at, step) in pre.iter().enumerate() {
                    let Step::GroupedDepthwise(op) = step else {
                        continue;
                    };
                    if from < at {
                        execute(gpu, ctx, &program.steps, start + from..start + at, hc)?;
                    }
                    match row {
                        Some(row) => {
                            execute_grouped_depthwise_rows(
                                gpu,
                                op,
                                CoreWindow { lo: 0, rows: row },
                            )
                            .map_err(fail)?;
                            stage
                                .copy(gpu, stage.fixed.len() - 2, op.state)
                                .map_err(fail)?;
                            execute_grouped_depthwise_rows(
                                gpu,
                                op,
                                CoreWindow {
                                    lo: row,
                                    rows: op.rows - row,
                                },
                            )
                            .map_err(fail)?;
                        }
                        None => {
                            execute(gpu, ctx, &program.steps, start + at..start + at + 1, hc)?;
                            stage
                                .copy(gpu, stage.fixed.len() - 2, op.state)
                                .map_err(fail)?;
                        }
                    }
                    from = at + 1;
                }
                if from < pre.len() {
                    execute(gpu, ctx, &program.steps, start + from..index, hc)?;
                }
            }
        }
        let prep_end = records.get(layer * 2 + bank).map(|r| &r.prep_end);
        if !covered {
            if let Some(r) = records.get(layer * 2 + bank) {
                for event in [&r.prep_end, &r.pack_end] {
                    gpu.hip.event_record(event, gpu.active_stream.as_ref())?;
                }
            }
            return Ok(None);
        }
        let Some(rt) = rt else {
            // `emulate`: the GPU twin projection fills the bank's scratch where
            // `on` packs and kicks; its Core then reads that scratch.
            self.emulate(gpu, layer, &program.steps[index])?;
            return Ok(None);
        };
        let pack_issue_ns = self.pack(gpu, &program.steps[index], bank, true, prep_end)?;
        if let Some(r) = records.get(layer * 2 + bank) {
            gpu.hip
                .event_record(&r.pack_end, gpu.active_stream.as_ref())?;
        }
        self.record_sample_event(gpu, layer, bank, 0)?;
        let kick_ns = if records.is_empty() {
            0
        } else {
            monotonic_ns()
        };
        let panels = self.layers[layer]
            .map
            .panel_range(self.request_families)
            .expect("selected family groups are contiguous within a shape");
        let seq = unsafe {
            rt.kick_panels(
                layer,
                transport_shape(self.layers[layer].shape),
                bank,
                program.rows,
                panels,
                stream(gpu),
            )
            .map_err(fail)?
        };
        if !records.is_empty() {
            let r = &mut records[layer * 2 + bank];
            r.pack_issue_ns = pack_issue_ns;
            r.kick_ns = kick_ns;
            r.seq = seq.value;
        }
        Ok(Some(seq))
    }
    fn run_bank(
        &self,
        gpu: &mut Gpu,
        ctx: &DispatchCtx,
        program: &mut layers::BankProgram<'_>,
        hc: &mut HcFusionCursor,
        layer: usize,
        bank: usize,
        seq: Option<Seq>,
        cut: Option<CoreCut<'_>>,
        records: &mut [TraceRow],
    ) -> Result<(), Qwen4GpuForwardError> {
        let index = mixer_index(program, layer);
        let rt = self.runtime.as_ref();
        let covered = !self.guard_tripped
            && self.layers[layer]
                .map
                .parts
                .iter()
                .any(|&(family, _, _)| self.request_families.contains(family));
        if let Some(r) = records.get(layer * 2 + bank) {
            gpu.hip
                .event_record(&r.wait_begin, gpu.active_stream.as_ref())?;
        }
        if covered {
            self.record_sample_event(gpu, layer, bank, 1)?;
        }
        if let (Some(rt), Some(seq)) = (rt, seq) {
            unsafe {
                rt.wait_gpu(seq, stream(gpu)).map_err(fail)?;
            }
        }
        if covered {
            self.record_sample_event(gpu, layer, bank, 2)?;
        }
        let wait_enqueue_ns = monotonic_ns();
        if !records.is_empty() {
            let r = &mut records[layer * 2 + bank];
            r.wait_enqueue_ns = wait_enqueue_ns;
            r.consume_issue_ns = 0;
            gpu.hip
                .event_record(&r.wait_end, gpu.active_stream.as_ref())?;
            gpu.hip.event_record(&r.start, gpu.active_stream.as_ref())?;
        }
        let rows = program.rows;
        let cut_row = cut.as_ref().map(|cut| cut.row);
        if !covered {
            let end = program.ranges[layer].end;
            match &cut {
                None => execute_validated_steps_window(gpu, ctx, &program.steps, index..end, hc)
                    .map_err(fail)?,
                Some(cut) => {
                    // Native route with a snapshot row inside this bank: bank-wide
                    // input projections, the Core in two launches, then the rest.
                    phase(&mut program.steps[index], MixerPhase::InProj);
                    execute_validated_steps_window(gpu, ctx, &program.steps, index..index + 1, hc)
                        .map_err(fail)?;
                    phase(&mut program.steps[index], MixerPhase::Core);
                    for (at, window) in core_windows(rows, Some(cut.row)).into_iter().enumerate() {
                        self.core_launch(gpu, layer, bank, &program.steps[index], false, window)?;
                        if at == 0 {
                            copy_mixer_state(
                                gpu,
                                cut.stage,
                                &program.steps[index],
                                cut.gdn_total,
                                cut.gdn_slot,
                                cut.qsa_slot,
                            )?;
                        }
                    }
                    phase(&mut program.steps[index], MixerPhase::OutProj);
                    execute_validated_steps_window(gpu, ctx, &program.steps, index..end, hc)
                        .map_err(fail)?;
                    phase(&mut program.steps[index], MixerPhase::Whole);
                }
            }
            if !records.is_empty() {
                gpu.hip
                    .event_record(&records[layer * 2 + bank].end, gpu.active_stream.as_ref())?;
            }
            return Ok(());
        }
        self.uncovered(gpu, &program.steps[index])?;
        phase(&mut program.steps[index], MixerPhase::Core);
        // One launch, or two split at the snapshot row with the stage copies
        // taken after the first and never after the second.
        for (at, window) in core_windows(rows, cut_row).into_iter().enumerate() {
            self.core_launch(gpu, layer, bank, &program.steps[index], true, window)?;
            if let (0, Some(cut)) = (at, &cut) {
                copy_mixer_state(
                    gpu,
                    cut.stage,
                    &program.steps[index],
                    cut.gdn_total,
                    cut.gdn_slot,
                    cut.qsa_slot,
                )?;
            }
        }
        // Core queues every arena reader (including the GDN Z gate and QSA
        // cache/index appends). OutProj and the remaining layer read only scratch.
        // prepare_bank waits this event before packing A; kick records ready
        // after that wait/pack, and the relay waits ready before its next C write.
        // `emulate` has no relay to retire.
        if let (Some(rt), Some(seq)) = (rt, seq) {
            unsafe {
                rt.retire_bank(seq, stream(gpu)).map_err(fail)?;
            }
        }
        phase(&mut program.steps[index], MixerPhase::OutProj);
        let end = program.ranges[layer].end;
        execute(gpu, ctx, &program.steps, index..end, hc)?;
        if !records.is_empty() {
            gpu.hip
                .event_record(&records[layer * 2 + bank].end, gpu.active_stream.as_ref())?;
        }
        Ok(())
    }
    pub(super) fn pair(
        &mut self,
        owner: &mut Qwen4GpuForward,
        bundle: &mut Qwen4Bundle,
        gpu: &mut Gpu,
        tokens: &[u32],
        banks: [usize; 2],
        logits: Option<&GpuTensor>,
        top1: Option<&GpuTensor>,
        capture: Option<&GpuTensor>,
        output: Qwen4OutputPolicy,
    ) -> Result<(), Qwen4GpuForwardError> {
        let chunk = owner.scratch.max_chunk;
        if banks[0] + banks[1] != tokens.len()
            || !(1..=chunk).contains(&banks[0])
            || !(1..=chunk).contains(&banks[1])
        {
            return Err(fail(
                "pair requires two banks of 1..=8192 live rows covering the tokens",
            ));
        }
        // Every bank must be large enough that the Core routes of a (sub-)launch
        // are those the bank's input projections were stored for.
        if banks
            .iter()
            .any(|&bank| bank < rdna_compute::gemm::QWEN4_F16_WMMA_MIN_TOKENS)
        {
            return Err(fail("pair banks need at least 512 live rows"));
        }
        let old_scope = std::mem::replace(&mut gpu.qwen4_scope, true);
        let invariant =
            owner.prefill_scope && rdna_compute::prefill_invariant::prefill_pm_invariant_enabled();
        let old_invariant = std::mem::replace(&mut gpu.prefill_invariant_scope, invariant);
        let ragged_scope = native_trunk_prefill_scope(
            owner.prefill_scope,
            gpu.qwen4_trunk_iu4,
            &gpu.arch,
            owner.scratch.mode,
        );
        let old_ragged = std::mem::replace(&mut gpu.qwen4_trunk_iu4_prefill_scope, ragged_scope);
        let result = self.pair_scoped(
            owner, bundle, gpu, tokens, banks, logits, top1, capture, output,
        );
        // Bank A's live extents were bound to its rows; the sequential route
        // expects the full chunk.
        let restored = if banks[0] != chunk {
            bind_bank_rows(&mut owner.scratch, &bundle.config, chunk)
        } else {
            Ok(())
        };
        gpu.qwen4_scope = old_scope;
        gpu.prefill_invariant_scope = old_invariant;
        gpu.qwen4_trunk_iu4_prefill_scope = old_ragged;
        if result.is_err() {
            bundle.state.invalidate_f16_arenas();
        }
        result.and(restored)
    }
    fn pair_scoped(
        &mut self,
        owner: &mut Qwen4GpuForward,
        bundle: &mut Qwen4Bundle,
        gpu: &mut Gpu,
        tokens: &[u32],
        rows: [usize; 2],
        logits: Option<&GpuTensor>,
        top1: Option<&GpuTensor>,
        capture: Option<&GpuTensor>,
        output: Qwen4OutputPolicy,
    ) -> Result<(), Qwen4GpuForwardError> {
        self.poll(gpu)?;
        // A sample may complete between the driver's decision and this poll.
        // Such an already-admitted pair uses native projections throughout.
        if let Some(sampling) = &mut self.sampling {
            sampling.active = None;
            if !self.guard_tripped {
                let active = sampling.next;
                sampling.next = (active + 1) % sampling.sets.len();
                // poll just queried this set. Drop an unready sample rather than wait for reuse.
                sampling.sets[active].pending = None;
                sampling.active = Some(active);
            }
        }
        owner.restore_expert_stage(gpu)?;
        let position = bundle.state.position;
        let end = position
            .checked_add(tokens.len())
            .ok_or_else(|| fail("position overflow"))?;
        // The session snapshot row inside the group: at the A/B split, or inside
        // a bank, whose Core then runs as two launches split at it.
        let shape = PairShape::new(rows, position, bundle.session_next_boundary());
        if let Some(cut) = shape.cut {
            if bundle
                .session_boundary_after(position + cut)
                .is_some_and(|next| next < end)
            {
                return Err(fail(
                    "a pair holds at most one interior session snapshot row",
                ));
            }
        }
        let site = shape.site();
        let geometry = launch_geometry(shape);
        let offsets = [geometry[0].offset, geometry[1].offset];
        bundle.release_session_for_growth(gpu, end);
        bundle
            .state
            .ensure_mapped_capacity(gpu, end)
            .map_err(fail)?;
        if let Some(qsa) = bundle.state.qsa.first().filter(|q| q.f16_arena.is_none()) {
            rdna_compute::tensor_ops::ensure_qsa_gathered_wmma_workspace(
                gpu,
                qsa.format,
                bundle.config.num_key_value_heads,
                end,
            )?;
        }
        let mut stage = match shape.cut {
            Some(cut) => bundle
                .prepare_pair_session_stage(gpu, position + cut)
                .map_err(fail)?,
            None => None,
        };
        self.note_pair(rows, position);
        // No tape can contain relay/host operations. Decode retains its old route.
        gpu.replay.set_forward_eligible(false);
        owner.uploaded_id_rows = 0;
        // Bank A adopts the first-tile PLE ticket started at admission; bank
        // B's rows are enqueued once A's fetch owns the epoch and adopted by
        // B's initialization (rows are a pure function of their ids).
        let ple_ahead = owner.ple_ahead.take();
        // Bank B binds its live rows; a partial bank A does too (`pair` restores its full extents).
        if rows[0] != owner.scratch.max_chunk {
            bind_bank_rows(&mut owner.scratch, &bundle.config, rows[0])?;
        }
        bind_bank_rows(
            self.bank_b.as_mut().expect("on bank B"),
            &bundle.config,
            rows[1],
        )?;
        let b = self.bank_b.as_ref().expect("on bank B");
        let ra = matrix_view(
            &owner.scratch.router_logits,
            rows[0],
            router_columns(program_dims(&bundle.config)),
        )?;
        let rb = matrix_view(
            &b.router_logits,
            rows[1],
            router_columns(program_dims(&bundle.config)),
        )?;
        let ctx = DispatchCtx::new(gpu);
        let sa = layer_scratch(&owner.scratch, &ra);
        let sb = layer_scratch(b, &rb);
        let mut programs = [
            layers::build_program(
                owner,
                &owner.scratch,
                bundle,
                gpu,
                &ctx,
                &sa,
                offsets[0],
                rows[0],
            )?,
            layers::build_program(owner, b, bundle, gpu, &ctx, &sb, offsets[1], rows[1])?,
        ];
        let mut history = bundle.state.ple_history;
        // Bank A only: bank B initializes after kick(0, A) (L1 fill reorder),
        // so NPU(0, A) does not wait behind bank B's embedding/PLE staging.
        if let Some(sampling) = &self.sampling {
            if let Some(active) = sampling.active {
                gpu.hip.event_record(
                    sampling.sets[active].pair_start(),
                    gpu.active_stream.as_ref(),
                )?;
            }
        }
        let mut ple_next = initialize(
            gpu,
            bundle,
            &owner.scratch,
            &tokens[..rows[0]],
            &history,
            programs[0].ple_rows_f16,
            &mut self.host_tokens,
            &mut self.host_ple,
            ple_ahead,
            Some(&tokens[rows[0]..]),
        )?;
        for &token in &tokens[..rows[0]] {
            history.push(token);
        }
        if let (Some(stage), Some(cut)) = (&mut stage, shape.cut) {
            let (cut_bank, cut_row) = match site {
                CutSite::Split => (0, rows[0]),
                CutSite::Bank { bank, row } => (bank, row),
                CutSite::None => unreachable!("a cut has a site"),
            };
            let mut boundary_history = bundle.state.ple_history;
            for &token in &tokens[..cut] {
                boundary_history.push(token);
            }
            let commits: SmallVec<[_; QWEN4_QSA_INLINE_CAPACITY]> = programs[cut_bank]
                .steps
                .iter()
                .filter_map(|step| match step {
                    Step::IndexedAttention(op) => {
                        Some(op.next_lengths_after(cut_row).map_err(fail))
                    }
                    _ => None,
                })
                .collect::<Result<_, _>>()?;
            stage
                .configure(&bundle.state, boundary_history, &commits)
                .map_err(fail)?;
        }
        let commits: SmallVec<[_; QWEN4_QSA_INLINE_CAPACITY]> = programs[1]
            .steps
            .iter()
            .filter_map(|step| match step {
                Step::IndexedAttention(op) => Some(op.next_lengths().map_err(fail)),
                _ => None,
            })
            .collect::<Result<_, _>>()?;
        let anchor = if self.trace.is_some() && self.runtime.is_some() {
            Some(gpu.hip.event_create()?)
        } else {
            None
        };
        let mut anchor_ns = 0;
        let mut anchor_uncertainty_ns = 0;
        let mut records = Vec::new();
        if let Some(anchor) = &anchor {
            // Drain initialization before calibrating the HIP clock. Both waits
            // are before the first pipeline pack, never between its layers.
            gpu.hip.event_record(anchor, gpu.active_stream.as_ref())?;
            gpu.hip.event_synchronize(anchor)?;
            let before = monotonic_ns();
            gpu.hip.event_record(anchor, gpu.active_stream.as_ref())?;
            gpu.hip.event_synchronize(anchor)?;
            let after = monotonic_ns();
            anchor_ns = before + (after - before) / 2;
            anchor_uncertainty_ns = after - before;
            for layer in 0..bundle.config.num_hidden_layers {
                for bank in 0..2 {
                    records.push(TraceRow {
                        layer,
                        bank,
                        position: position + offsets[bank],
                        rows: rows[bank],
                        seq: 0,
                        pack_issue_ns: 0,
                        kick_ns: 0,
                        submit_ns: 0,
                        done_ns: 0,
                        signal_write_ns: 0,
                        wait_enqueue_ns: 0,
                        consume_issue_ns: 0,
                        prep_begin: gpu.hip.event_create()?,
                        prep_end: gpu.hip.event_create()?,
                        pack_end: gpu.hip.event_create()?,
                        wait_begin: gpu.hip.event_create()?,
                        wait_end: gpu.hip.event_create()?,
                        start: gpu.hip.event_create()?,
                        end: gpu.hip.event_create()?,
                    });
                }
            }
        }
        // The closure borrows `self` for prepare/run_bank; bank B's
        // initialization borrows the host upload buffers, so they move out.
        let mut host_tokens = std::mem::take(&mut self.host_tokens);
        let mut host_ple = std::mem::take(&mut self.host_ple);
        let attempt: Result<(), Qwen4GpuForwardError> = (|| {
            let snapshot = stage.as_ref().map(|stage| Snapshot { stage, geometry });
            // Each bank's HC pairings cross its windows (prepare, Core, OutProj);
            // the gates they hand over live in the bank's private `hc_gates`.
            let mut hc = [HcFusionCursor::default(), HcFusionCursor::default()];
            let mut seq = [None; 2];
            let mut last_seq = seq;
            // (GDN, QSA) layers already run per bank: the snapshot slots.
            let mut slots = [(0usize, 0usize); 2];
            let gdn_total = bundle.state.gdn.len();
            for op in coproc::pair_schedule(bundle.config.num_hidden_layers) {
                match op {
                    PairOp::Prepare { layer, bank } => {
                        seq[bank] = self.prepare(
                            gpu,
                            &ctx,
                            &programs[bank],
                            &mut hc[bank],
                            layer,
                            bank,
                            &mut records,
                            snapshot,
                        )?;
                        if seq[bank].is_some() {
                            last_seq[bank] = seq[bank];
                        }
                    }
                    // Bank A initialized before the pair; bank B's init follows kick(0, A) only as the
                    // pipeline fill (NPU(0, A) does not wait behind bank B's embedding/PLE staging).
                    PairOp::Init { bank: 1 } => {
                        let bank_b_tokens = &tokens[rows[0]..];
                        initialize(
                            gpu,
                            bundle,
                            b,
                            bank_b_tokens,
                            &history,
                            programs[1].ple_rows_f16,
                            &mut host_tokens,
                            &mut host_ple,
                            ple_next.take(),
                            None,
                        )?;
                        for &token in bank_b_tokens {
                            history.push(token);
                        }
                    }
                    PairOp::Run { layer, bank } => {
                        let cut = match (geometry[bank].cut, &stage) {
                            (Some(row), Some(stage)) => Some(CoreCut {
                                stage,
                                row,
                                gdn_total,
                                gdn_slot: slots[bank].0,
                                qsa_slot: slots[bank].1,
                            }),
                            _ => None,
                        };
                        self.run_bank(
                            gpu,
                            &ctx,
                            &mut programs[bank],
                            &mut hc[bank],
                            layer,
                            bank,
                            seq[bank],
                            cut,
                            &mut records,
                        )?;
                        let mixer = &programs[bank].steps[mixer_index(&programs[bank], layer)];
                        if geometry[bank].stage_after {
                            if let Some(stage) = &stage {
                                copy_mixer_state(
                                    gpu, stage, mixer, gdn_total, slots[0].0, slots[0].1,
                                )?;
                            }
                        }
                        match mixer {
                            Step::GatedDeltaNet(_) => slots[bank].0 += 1,
                            Step::IndexedAttention(_) => slots[bank].1 += 1,
                            _ => unreachable!("typed mixer"),
                        }
                    }
                    PairOp::Init { bank } => {
                        unreachable!("the pair schedule initializes bank B only, not {bank}")
                    }
                }
            }
            // Only already-enqueued chains are checked here. Failed terminal
            // signals may have admitted speculative fragment work, but no pair
            // output/head/capture or request state is published on that path.
            if let Some(rt) = self.runtime.as_ref() {
                for bank in 0..2 {
                    if let Some(seq) = last_seq[bank] {
                        let completion = rt.wait_host(seq).map_err(fail)?;
                        if let Some(r) = records
                            .iter_mut()
                            .rev()
                            .find(|r| r.bank == bank && r.seq == seq.value)
                        {
                            r.submit_ns = completion.submit_ns;
                            r.done_ns = completion.done_ns;
                            r.signal_write_ns = completion.signal_write_ns;
                        }
                    }
                }
                rt.check_error().map_err(fail)?;
            }
            // Both trunks finish before prompt-end head/capture work.
            finish_bank(
                gpu,
                bundle,
                &owner.scratch,
                logits,
                top1,
                capture,
                0,
                0,
                rows[0],
                output,
            )?;
            finish_bank(
                gpu, bundle, b, logits, top1, capture, 1, rows[0], rows[1], output,
            )?;
            if let Some(sampling) = &self.sampling {
                if let Some(active) = sampling.active {
                    gpu.hip.event_record(
                        sampling.sets[active].pair_end(),
                        gpu.active_stream.as_ref(),
                    )?;
                }
            }
            Ok(())
        })();
        if let (Some(sampling), Some(rt)) = (&mut self.sampling, &self.runtime) {
            let stats = rt.take_chain_stats();
            if let Some(active) = sampling.active.take() {
                if attempt.is_ok() {
                    use railgun::npu::trunk::Shape as T;
                    let npu_busy_ms = (stats.shape(T::GdnIn).npu_busy_ns as f64
                        + stats.shape(T::QsaIn).npu_busy_ns as f64)
                        / 1e6;
                    let kinds = (0..bundle.config.num_hidden_layers)
                        .map(
                            |layer| match &programs[0].steps[mixer_index(&programs[0], layer)] {
                                Step::GatedDeltaNet(_) => 0,
                                Step::IndexedAttention(_) => 1,
                                _ => unreachable!("typed layer mixer"),
                            },
                        )
                        .collect();
                    let covered =
                        self.layers
                            .iter()
                            .flat_map(|layer| {
                                let covered =
                                    layer.map.parts.iter().any(|&(family, _, _)| {
                                        self.request_families.contains(family)
                                    });
                                [covered; 2]
                            })
                            .collect();
                    sampling.sets[active].pending = Some(PendingSample {
                        rows,
                        prefixes: [position, position + rows[0]],
                        mask: self.request_families,
                        covered,
                        kinds,
                        epoch: self.request_epoch,
                        position,
                        npu_busy_ms,
                        planned: self.request_cal.clone(),
                    });
                }
            }
        }
        self.host_tokens = host_tokens;
        self.host_ple = host_ple;
        drop(programs);
        let attempt = attempt.and_then(|_| {
            if let (Some(out), Some(anchor)) = (&mut self.trace, &anchor) {
                if let Some(last) = records.last() {
                    gpu.hip.event_synchronize(&last.end)?;
                }
                for r in &records {
                    let start_ns = anchor_ns
                        + (gpu.hip.event_elapsed_ms(anchor, &r.start)? as f64 * 1e6) as u64;
                    let end_ns =
                        anchor_ns + (gpu.hip.event_elapsed_ms(anchor, &r.end)? as f64 * 1e6) as u64;
                    let stamp = |event| -> Result<u64, hip_bridge::HipError> {
                        Ok(anchor_ns
                            + (gpu.hip.event_elapsed_ms(anchor, event)? as f64 * 1e6) as u64)
                    };
                    TraceRecord {
                        layer: r.layer,
                        bank: if r.bank == 0 { "A" } else { "B" },
                        position: r.position,
                        rows: r.rows,
                        seq: r.seq,
                        pack_issue_ns: r.pack_issue_ns,
                        kick_ns: r.kick_ns,
                        npu_submit_ns: r.submit_ns,
                        npu_done_ns: r.done_ns,
                        gpu_wait_enqueue_ns: r.wait_enqueue_ns,
                        consume_issue_ns: r.consume_issue_ns,
                        mixer_start_ns: start_ns,
                        mixer_end_ns: end_ns,
                        anchor_uncertainty_ns,
                        prep_begin_ns: stamp(&r.prep_begin)?,
                        prep_end_ns: stamp(&r.prep_end)?,
                        pack_end_ns: stamp(&r.pack_end)?,
                        wait_begin_ns: stamp(&r.wait_begin)?,
                        wait_end_ns: stamp(&r.wait_end)?,
                        signal_write_ns: r.signal_write_ns,
                    }
                    .write(out)
                    .map_err(fail)?;
                }
                out.flush().map_err(fail)?;
            }
            Ok(())
        });
        for r in records {
            for event in [
                r.prep_begin,
                r.prep_end,
                r.pack_end,
                r.wait_begin,
                r.wait_end,
                r.start,
                r.end,
            ] {
                let _ = gpu.hip.event_destroy(event);
            }
        }
        if let Some(anchor) = anchor {
            let _ = gpu.hip.event_destroy(anchor);
        }
        bundle.pair_session_stage = stage;
        attempt?;
        commit_qsa(bundle, &commits);
        bundle.state.ple_history = history;
        bundle.state.commit_row_capture(tokens.len());
        bundle.state.position = end;
        if shape.cut.is_some() && bundle.pair_boundary_is_ar() {
            bundle
                .publish_pair_session_boundary(
                    gpu,
                    position + shape.cut.expect("an AR boundary publish needs a cut"),
                )
                .map_err(fail)?;
        }
        Ok(())
    }
}

/// Stages one bank's embedding and PLE rows. `ahead` is adopted when it names
/// exactly this bank's rows. With `next`, the following bank's rows are
/// enqueued (best effort) once this fetch owns the epoch; the returned ticket
/// is that bank's `ahead`.
#[allow(clippy::too_many_arguments)]
fn initialize(
    gpu: &mut Gpu,
    bundle: &Qwen4Bundle,
    scratch: &Qwen4GpuForwardScratch,
    tokens: &[u32],
    history: &crate::ple::PleHistory,
    ple_f16: bool,
    ids: &mut [u8],
    ple_bytes: &mut [u8],
    ahead: Option<hipfire_runtime::external_rows::RowTicket>,
    next: Option<&[u32]>,
) -> Result<Option<hipfire_runtime::external_rows::RowTicket>, Qwen4GpuForwardError> {
    let mut fetch = match &bundle.ple {
        Some(ple) => Some(
            RowFetch::begin_or_adopt(&ple.rows, ahead, history.row_ids(&ple.metadata, tokens))
                .map_err(fail)?,
        ),
        None => {
            drop(ahead);
            None
        }
    };
    let next_ticket = match (&bundle.ple, next) {
        (Some(ple), Some(next)) if !next.is_empty() => {
            let mut next_history = *history;
            for &token in tokens {
                next_history.push(token);
            }
            ple.rows
                .prefetch(
                    ple.rows.current_epoch(),
                    next_history.row_ids(&ple.metadata, next),
                )
                .ok()
        }
        _ => None,
    };
    let n = tokens.len();
    let config = &bundle.config;
    for (dst, &token) in ids[..n * 4].chunks_exact_mut(4).zip(tokens) {
        dst.copy_from_slice(&(token as i32).to_ne_bytes());
    }
    gpu.memcpy_htod_auto(&scratch.token_ids.buf, &ids[..n * 4])?;
    let rotated = view(&scratch.embedding_rot, 0, n * config.hidden_size);
    let embeddings = view(&scratch.embeddings, 0, n * config.hidden_size);
    execute_embedding(
        gpu,
        &EmbeddingOp {
            table: bundle.weights.resident(&bundle.weights.root.embedding)?,
            rotated: &rotated,
            token_ids: &scratch.token_ids,
            output: &embeddings,
            rows: n,
            dim: config.hidden_size,
        },
    )
    .map_err(fail)?;
    if gpu.qwen4_bf16_streams(n) {
        gpu.hc_streams_init_from_embed_batched_bf16(
            &embeddings,
            &scratch.streams,
            config.hidden_size as i32,
            config.hc_count as i32,
            n as i32,
        )?;
    } else {
        gpu.hc_streams_init_from_embed_batched(
            &embeddings,
            &scratch.streams,
            config.hidden_size as i32,
            config.hc_count as i32,
            n as i32,
        )?;
    }
    if let Some(fetch) = fetch.as_mut() {
        let lease = fetch.wait().map_err(fail)?;
        let bytes = &mut ple_bytes[..lease.as_bytes().map_err(fail)?.len()];
        lease.stage_into(bytes).map_err(fail)?;
        gpu.memcpy_htod_auto(&scratch.ple_staged.buf, bytes)?;
        lease.validate_after_upload().map_err(fail)?;
        let geometry = config.ple.as_ref().expect("PLE owner has config");
        if ple_f16 {
            gpu.grouped_gather_convert_bf16_f16(
                &scratch.ple_staged,
                &scratch.ple_rows,
                n,
                geometry.head_count,
                geometry.row_width,
            )?;
        } else {
            gpu.grouped_gather_convert_bf16(
                &scratch.ple_staged,
                &scratch.ple_rows,
                n,
                geometry.head_count,
                geometry.row_width,
            )?;
        }
        fetch.complete();
    }
    Ok(next_ticket)
}

fn finish_bank(
    gpu: &mut Gpu,
    bundle: &Qwen4Bundle,
    scratch: &Qwen4GpuForwardScratch,
    logits: Option<&GpuTensor>,
    top1: Option<&GpuTensor>,
    capture: Option<&GpuTensor>,
    bank: usize,
    row0: usize,
    n: usize,
    output: Qwen4OutputPolicy,
) -> Result<(), Qwen4GpuForwardError> {
    let config = &bundle.config;
    let dims = program_dims(config);
    let bf16 = gpu.qwen4_bf16_streams(n);
    if let Some(capture) = capture {
        let dst = f32_view(capture, row0 * dims.wide(), n * dims.wide());
        let src = view(&scratch.streams, 0, n * dims.wide());
        if bf16 {
            rdna_compute::tensor_ops::hc_state_bf16_to_f32(gpu, &src, &dst, n * dims.wide())?;
        } else {
            gpu.copy_d2d(&src, &dst, src.byte_size())?;
        }
    }
    let rows = match output {
        Qwen4OutputPolicy::Rows(Qwen4OutputRows::All) => n,
        Qwen4OutputPolicy::Rows(Qwen4OutputRows::Final) if bank == 1 => 1,
        _ => return Ok(()),
    };
    let scratch_desc = layer_scratch(scratch, &scratch.router_logits);
    let hyper = hyper_read_desc(&bundle.weights, &bundle.weights.root.final_hyper)?;
    let lm = dense_ref(&bundle.weights, &bundle.weights.root.lm_head)?;
    execute_final_hyper(gpu, dims, &hyper, &scratch.streams, &scratch_desc, n, bf16)
        .map_err(fail)?;
    let hidden = view(&scratch.hc_mixed, 0, n * config.hidden_size);
    let offset = if rows == n {
        row0 * config.vocab_size
    } else {
        0
    };
    let logits = logits
        .ok_or_else(|| fail("head requires logits"))?
        .sub_offset(offset, rows * config.vocab_size);
    execute_lm_head(gpu, &lm, &hidden, &logits, n, rows, Some(&scratch.rotation)).map_err(fail)?;
    if let Some(top1) = top1 {
        let indices = top1.sub_offset(if rows == n { row0 * 4 } else { 0 }, rows * 4);
        argmax_f32(
            gpu,
            &ArgmaxF32 {
                logits: &logits,
                indices: &indices,
                rows,
                vocab: config.vocab_size,
            },
        )?;
    }
    Ok(())
}

/// The request's executable plan. `auto` arms it from the crossover `n_star`; the forced modes (`on`, `emulate`)
/// take the planner's pairs from the first row.
fn request_plan(
    cal: &coproc::Calibration,
    mode: NpuTrunkMode,
    families: FamilyMask,
    n_star: Option<usize>,
    rows: usize,
) -> coproc::Plan {
    let planned = coproc::plan_executable(&FLASH_NEXT, cal, families.into(), rows);
    let armed = if families.is_empty() {
        None
    } else {
        armed_from(mode, n_star)
    };
    coproc::executable_plan_for(&FLASH_NEXT, cal, families.into(), &planned, rows, armed)
}
/// The row the executable plan arms pairs from: the crossover `n_star` in `auto`, the first row in the forced modes.
fn armed_from(mode: NpuTrunkMode, n_star: Option<usize>) -> Option<usize> {
    if mode == NpuTrunkMode::Auto {
        n_star
    } else {
        Some(0)
    }
}
/// A bank below 8192 rows runs the row-adaptive NPU program of its own wave count, one relay hardware context per distinct
/// count and at most [`MAX_ADAPTIVE_CONTEXTS`] per load (the max-window budget, [`coproc::admit_wave_budget`]). Every
/// mode applies it, so `on` and `emulate` demote the same pairs.
fn admitted_plan(
    executable: coproc::Plan,
    admitted: &mut Vec<usize>,
    replan: impl Fn(&coproc::Plan) -> coproc::Plan,
) -> coproc::Plan {
    coproc::admit_wave_budget(
        executable,
        admitted,
        MAX_ADAPTIVE_CONTEXTS,
        |rows| padded_rows_of(transport_shape(Shape::GdnIn), rows),
        replan,
    )
}
/// Rows of the planned pair starting at `position` (the request began at `origin`) when it fits the remaining rows.
fn planned_banks(
    plan: &coproc::Plan,
    origin: usize,
    position: usize,
    remaining: usize,
    chunk: usize,
) -> Option<[usize; 2]> {
    if chunk != coproc::CHUNK {
        return None;
    }
    let (a, b) = coproc::planned_pair(plan, position.checked_sub(origin)?)?;
    (a + b <= remaining).then_some([a, b])
}

/// The natural pair: 8192 + B (B in 512..=8192); bank B may be partial.
pub(super) fn pair_rows(remaining: usize, chunk: usize) -> Option<usize> {
    if remaining < chunk + 512 {
        return None;
    }
    let rows = qwen4_prefill_segment_rows(0, remaining, chunk, SessionBoundaries::default(), true);
    (rows > chunk).then_some(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::npu_trunk::load_calibration;
    use hipfire_runtime::coproc::Calibration;
    #[test]
    fn two_hip_runtime_guard_uses_live_symbol_identity() {
        let path = "/opt/rocm/lib/libamdhip64.so";
        // One live implementation passes even if a path lookup failed or two
        // diagnostic paths name different aliases. No filesystem lookup is needed.
        for (relay, serving) in [
            (None, None),
            (Some(path), None),
            (None, Some(path)),
            (Some(path), Some("/opt/rocm/lib/libamdhip64.so.7")),
        ] {
            assert_eq!(two_hip_runtime_reason(0x1000, 0x1000, relay, serving), None);
        }
        // Distinct mappings refuse even when their recorded paths are identical
        // (a library was replaced) or when either diagnostic lookup failed.
        for (relay, serving) in [
            (Some(path), Some(path)),
            (Some(path), Some("/tmp/other/libamdhip64.so")),
            (None, Some(path)),
            (Some(path), None),
            (None, None),
        ] {
            let reason = two_hip_runtime_reason(0x1000, 0x2000, relay, serving)
                .expect("distinct implementations must refuse");
            assert!(reason.starts_with("two HIP runtimes:"), "{reason}");
            assert!(
                reason.contains(relay.unwrap_or("<unknown library>")),
                "{reason}"
            );
            assert!(
                reason.contains(serving.unwrap_or("<unknown library>")),
                "{reason}"
            );
            assert!(reason.contains("HIPFIRE_ROCM_PATH"), "{reason}");
            assert!(
                reason.contains("hipEventRecord identity differs"),
                "{reason}"
            );
        }
        assert!(two_hip_runtime_reason(0, 0, None, None).is_some());
        assert!(two_hip_runtime_reason(0, 0x1000, None, None).is_some());
    }
    #[test]
    fn family_override_parser_selects_groups_and_rejects_unknowns() {
        assert_eq!(parse_family_override("all").unwrap(), FamilyMask::ALL);
        for (name, bits) in [("qkv", 0x01), ("zab", 0x0e), ("q", 0x10), ("kvidx", 0xe0)] {
            assert_eq!(
                parse_family_override(name).unwrap(),
                FamilyMask::from_bits(bits)
            );
        }
        assert_eq!(
            parse_family_override(" qkv, zab,q,kvidx ").unwrap(),
            FamilyMask::ALL
        );
        assert_eq!(parse_family_override("qkv,q").unwrap().bits(), 0x11);
        assert_eq!(parse_family_override("q,q").unwrap().bits(), 0x10);
        for bad in ["", "z", "qsa_q", "qkv,", "ALL", "qkv,nope"] {
            assert!(parse_family_override(bad).is_err(), "{bad}");
        }
        assert_eq!(
            intersect_families(parse_family_override("all").unwrap(), FamilyMask::QSA),
            FamilyMask::QSA
        );
    }
    #[test]
    fn trace_jsonl_preserves_schema_and_adds_fine_timestamps() {
        let record = TraceRecord {
            layer: 3,
            bank: "B",
            rows: 512,
            seq: 7,
            prep_begin_ns: 11,
            prep_end_ns: 12,
            pack_end_ns: 13,
            wait_begin_ns: 14,
            wait_end_ns: 17,
            signal_write_ns: 16,
            ..Default::default()
        };
        let mut bytes = Vec::new();
        record.write(&mut bytes).unwrap();
        assert_eq!(bytes.iter().filter(|&&b| b == b'\n').count(), 1);
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let fields = [
            "layer",
            "bank",
            "position",
            "rows",
            "seq",
            "pack_issue_ns",
            "kick_ns",
            "npu_submit_ns",
            "npu_done_ns",
            "gpu_wait_enqueue_ns",
            "consume_issue_ns",
            "mixer_start_ns",
            "mixer_end_ns",
            "anchor_uncertainty_ns",
            "prep_begin_ns",
            "prep_end_ns",
            "pack_end_ns",
            "wait_begin_ns",
            "wait_end_ns",
            "signal_write_ns",
        ];
        assert_eq!(json.as_object().unwrap().len(), fields.len());
        for key in fields {
            assert!(json.get(key).is_some(), "{key}");
        }
        assert_eq!(json["bank"], "B");
        assert_eq!(json["prep_begin_ns"], 11);
        assert_eq!(json["wait_end_ns"], 17);
        assert_eq!(json["signal_write_ns"], 16);
    }
    #[test]
    fn every_mask_selects_only_covered_family_views() {
        for bits in 0..=u8::MAX {
            let mask = FamilyMask::from_bits(bits);
            for shape in Shape::ALL {
                let map = shape.column_map(mask);
                let tensor = GpuTensor {
                    buf: unsafe {
                        hip_bridge::DeviceBuffer::from_raw(
                            std::ptr::dangling_mut(),
                            256 * map.padded_cols() * 2,
                        )
                    },
                    shape: vec![256, map.padded_cols()],
                    dtype: DType::BF16,
                };
                let mut col = 0;
                for family in Family::ALL.into_iter().filter(|f| f.shape() == shape) {
                    let view = family_view(&map, &tensor, family);
                    assert_eq!(view.is_some(), mask.contains(family));
                    if let Some(view) = view {
                        assert_eq!(
                            (view.col0_elems, view.row_pitch_elems),
                            (col, map.padded_cols())
                        );
                        col += family.cols();
                    }
                }
                assert_eq!(map.panel_count(), col.div_ceil(1280));
                assert_eq!(map.padded_cols(), col.div_ceil(1280) * 1280);
            }
        }
    }
    #[test]
    fn partial_bank_views_use_live_rows_and_keep_allocation_capacity() {
        let config = crate::config::compact_test_config();
        let dims = program_dims(&config);
        let (capacity, _) =
            Qwen4GpuForwardScratch::plan(&config, 8192, Qwen4ForwardScratchMode::Full).unwrap();
        let mut bank: Vec<_> = capacity
            .iter()
            .map(|slot| GpuTensor {
                buf: unsafe {
                    hip_bridge::DeviceBuffer::from_raw(
                        std::ptr::dangling_mut(),
                        slot.bytes().unwrap(),
                    )
                },
                shape: vec![slot.elements],
                dtype: slot.dtype,
            })
            .collect();
        for rows in [8151, 512, 8192] {
            let (mut live, _) =
                Qwen4GpuForwardScratch::plan(&config, rows, Qwen4ForwardScratchMode::Full).unwrap();
            preserve_native_projection_capacity(&mut live, &capacity);
            bind_bank_extents(&mut bank.iter_mut().collect::<Vec<_>>(), &live).unwrap();
            for (field, (tensor, slot)) in bank.iter().zip(&live).enumerate() {
                assert_eq!(tensor.numel(), slot.elements, "field {field} rows {rows}");
                assert_eq!(tensor.buf.size(), capacity[field].bytes().unwrap());
            }
            let at = |field: ScratchField| &bank[field as usize];
            assert_eq!(at(ScratchField::PleStaged).numel(), rows * 2560);
            assert_eq!(at(ScratchField::PleRows).numel(), rows * 2560);
            assert_eq!(
                matrix_view(at(ScratchField::RouterLogits), rows, router_columns(dims))
                    .unwrap()
                    .numel(),
                rows * router_columns(dims)
            );
            assert_eq!(at(ScratchField::MoeXRot).numel(), rows * dims.hidden);
            assert_eq!(
                at(ScratchField::MoeTopkIndices).numel(),
                rows * config.num_experts_per_tok * 4
            );
            assert_eq!(
                at(ScratchField::MoeYGateUpGrouped).numel(),
                live[ScratchField::MoeYGateUpGrouped as usize].elements
            );
            assert_eq!(
                f32_view(at(ScratchField::Streams), 0, rows * dims.wide()).numel(),
                rows * dims.wide()
            );
            for field in [
                ScratchField::HcMixed,
                ScratchField::GdnZ,
                ScratchField::GdnA,
                ScratchField::GdnB,
            ] {
                assert_eq!(at(field).numel(), capacity[field as usize].elements);
            }
        }
    }
    #[test]
    fn native_trunk_bank_decisions_use_live_rows_and_sequential_capacity() {
        let config = crate::config::compact_test_config();
        let (capacity, _) =
            Qwen4GpuForwardScratch::plan(&config, 8192, Qwen4ForwardScratchMode::Full).unwrap();
        assert!(native_trunk_prefill_scope(
            true,
            true,
            "gfx1151",
            Qwen4ForwardScratchMode::Full
        ));
        assert!(!native_trunk_prefill_scope(
            false,
            true,
            "gfx1151",
            Qwen4ForwardScratchMode::Full
        ));
        assert!(!native_trunk_prefill_scope(
            true,
            true,
            "gfx1151",
            Qwen4ForwardScratchMode::FnSymGfx1201
        ));
        for rows in [8192, 8151, 512] {
            let (mut live, _) =
                Qwen4GpuForwardScratch::plan(&config, rows, Qwen4ForwardScratchMode::Full).unwrap();
            preserve_native_projection_capacity(&mut live, &capacity);
            let padded = rdna_compute::gemm::qwen4_trunk_iu4_ragged_rows(rows);
            assert_eq!(padded, (rows == 8151).then_some(8192));
            for (field, width) in [
                (ScratchField::HcMixed, config.hidden_size),
                (
                    ScratchField::GdnZ,
                    config.linear_num_value_heads * config.linear_value_head_dim,
                ),
                (ScratchField::GdnA, config.linear_num_value_heads),
                (ScratchField::GdnB, config.linear_num_value_heads),
            ] {
                let padded = padded.unwrap_or(rows);
                assert!(live[field as usize].elements >= padded * width);
                assert_eq!(
                    live[field as usize].elements,
                    capacity[field as usize].elements
                );
            }
        }
    }
    #[test]
    fn covered_phases_skip_all_native_projection_routes() {
        // The phase gate precedes project_trunk's dtype/IU4 mask selection:
        // these decisions are identical for stock MQ6 and overlay A4.
        assert!(!skips_input_projections(MixerPhase::Whole));
        assert!(!skips_input_projections(MixerPhase::InProj));
        assert!(skips_input_projections(MixerPhase::Core));
        assert!(skips_input_projections(MixerPhase::OutProj));
    }
    #[test]
    fn arena_family_windows_never_include_padding() {
        for (shape, live, pitch) in [(Shape::GdnIn, 16480, 16640), (Shape::QsaIn, 13952, 14080)] {
            let map = shape.column_map(FamilyMask::ALL);
            assert_eq!(map.padded_cols(), pitch);
            assert_eq!(
                map.parts.last().map(|&(_, col, width)| col + width),
                Some(live)
            );
            for &(_, col, width) in &map.parts {
                for rows in [512, 4647, 8192] {
                    let last_read = (rows - 1) * pitch + col as usize + width as usize - 1;
                    assert!(last_read < (rows - 1) * pitch + live as usize);
                }
            }
        }
    }
    #[test]
    fn pairs_cover_exact_ranges_and_keep_tail() {
        for rows in [
            0, 511, 8192, 8703, 8704, 16200, 16383, 16384, 16385, 20000, 32768, 40000, 40960,
        ] {
            let mut at = 0;
            let mut ranges = Vec::new();
            let mut banks = Vec::new();
            while at < rows {
                let paired = pair_rows(rows - at, 8192);
                let n = paired.unwrap_or((rows - at).min(8192));
                ranges.push(at..at + n);
                if paired.is_some() {
                    banks.extend([at..at + 8192, at + 8192..at + n]);
                    assert!((512..=8192).contains(&(n - 8192)));
                } else {
                    banks.push(at..at + n);
                }
                at += n;
            }
            assert_eq!(at, rows);
            for pair in ranges.windows(2) {
                assert_eq!(pair[0].end, pair[1].start);
            }
            // Emulate splits at the same 8192 boundary, including ragged B.
            let sequential: Vec<_> = (0..rows)
                .step_by(8192)
                .map(|start| start..(start + 8192).min(rows))
                .collect();
            assert_eq!(banks, sequential);
            match rows {
                16200 => assert_eq!(ranges, vec![0..16200]),
                16384 => assert_eq!(ranges, vec![0..16384]),
                20000 => assert_eq!(ranges, vec![0..16384, 16384..20000]),
                40000 => assert_eq!(ranges, vec![0..16384, 16384..32768, 32768..40000]),
                _ => {}
            }
        }
        assert_eq!(pair_rows(8703, 8192), None);
        assert_eq!(pair_rows(8704, 8192), Some(8704));
        assert_eq!(pair_rows(32768, 4096), None);
    }
    /// The prefill loop over a request of `end` rows from 0: `(start, rows, bank A rows, cut)`.
    /// `planned` makes every group with a partner chunk a natural pair; the plan
    /// form takes the planner's pairs.
    fn schedule_with(
        end: usize,
        boundaries: &[usize],
        pair_at: &dyn Fn(usize) -> Option<[usize; 2]>,
    ) -> Vec<(usize, usize, Option<usize>, Option<usize>)> {
        let mut pos = 0;
        let mut groups = Vec::new();
        while pos < end {
            let next = boundaries.iter().copied().find(|&b| b > pos);
            let after = next.and_then(|next| boundaries.iter().copied().find(|&b| b > next));
            let pair = pair_at(pos);
            let rows = qwen4_prefill_segment_rows(
                pos,
                end,
                8192,
                SessionBoundaries { next, after },
                pair.is_some(),
            );
            assert!(rows > 0);
            let shape = pair
                .filter(|banks| banks[0] + banks[1] == rows)
                .map(|banks| PairShape::new(banks, pos, next));
            groups.push((
                pos,
                rows,
                shape.map(|s| s.rows[0]),
                shape.and_then(|s| s.cut),
            ));
            pos += rows;
        }
        groups
    }
    fn natural_pair(end: usize) -> impl Fn(usize) -> Option<[usize; 2]> {
        move |pos| pair_rows(end - pos, 8192).map(|rows| [8192, rows - 8192])
    }
    #[test]
    fn segments_preserve_snapshots_and_native_tails() {
        let ranges = |end: usize, boundaries: &[usize]| -> Vec<std::ops::Range<usize>> {
            schedule_with(end, boundaries, &natural_pair(end))
                .into_iter()
                .map(|(start, rows, ..)| start..start + rows)
                .collect()
        };
        assert_eq!(ranges(16343, &[]), vec![0..16343]);
        assert_eq!(ranges(8703, &[]), vec![0..8192, 8192..8703]);
        // One interior snapshot row rides the bank's Core sub-launch; a second
        // interior row ends the group at the first.
        assert_eq!(ranges(16343, &[8192, 16343]), vec![0..16343]);
        assert_eq!(ranges(16384, &[8192, 16384]), vec![0..16384]);
        assert_eq!(
            ranges(16384, &[8192, 10000, 16384]),
            vec![0..8192, 8192..10000, 10000..16384]
        );
        assert_eq!(ranges(20000, &[12000, 20000]), vec![0..16384, 16384..20000]);
        assert_eq!(
            ranges(20000, &[4000, 13000, 20000]),
            vec![0..4000, 4000..20000]
        );
        let cut = |end: usize, boundaries: &[usize]| {
            schedule_with(end, boundaries, &natural_pair(end))
                .into_iter()
                .map(|g| g.3)
                .collect::<Vec<_>>()
        };
        assert_eq!(cut(20000, &[12000, 20000]), vec![Some(12000), None]);
        assert_eq!(cut(20000, &[4000, 13000, 20000]), vec![None, Some(9000)]);
        assert_eq!(
            qwen4_prefill_segment_rows(
                8192,
                40000,
                8192,
                SessionBoundaries {
                    next: Some(8192),
                    after: None
                },
                true
            ),
            8192
        );
        assert!(schedule_with(40000, &[], &|_| None)
            .iter()
            .all(|g| g.2.is_none()));
        // Every group's pieces tile it: banks of at most a chunk, cut once.
        for end in [16343, 20000, 40000] {
            for boundaries in [&[][..], &[8192, 16384, 24576, 32768, 40000][..]] {
                let groups = schedule_with(end, boundaries, &natural_pair(end));
                let mut at = 0;
                for (start, rows, split, cut) in groups {
                    assert_eq!(start, at);
                    if let Some(a) = split {
                        let pieces = PairShape::new(
                            [a, rows - a],
                            start,
                            boundaries.iter().copied().find(|&b| b > start),
                        )
                        .pieces();
                        assert_eq!(pieces.first().map(|p| p.start), Some(0));
                        assert_eq!(pieces.last().map(|p| p.end), Some(rows));
                        assert!(pieces.windows(2).all(|w| w[0].end == w[1].start));
                        assert!(pieces.iter().all(|p| p.len() <= 8192));
                        assert_eq!(pieces.len(), 2 + usize::from(cut.is_some_and(|c| c != a)));
                    }
                    at += rows;
                }
                assert_eq!(at, end);
            }
        }
    }
    /// A lineage split p inside a pair group that already holds a snapshot row
    /// (32K prime: 24576 and p = 32001 in 16384..32010) is a second interior row,
    /// which drops the group to GPU tiles. The bundle plans and schedules through
    /// p instead, so the group ends at p, keeps its one cut, and only the template
    /// tail past p runs as a tile.
    #[test]
    fn lineage_split_ends_the_pair_group() {
        let (p, end) = (32001, 32010);
        let boundaries = SessionBoundaries {
            next: Some(24576),
            after: Some(p),
        };
        assert_eq!(
            qwen4_prefill_segment_rows(16384, end, 8192, boundaries, true),
            8192
        );
        assert_eq!(
            qwen4_prefill_segment_rows(16384, p, 8192, boundaries, true),
            p - 16384
        );
        let shape = PairShape::new([7872, p - 16384 - 7872], 16384, Some(24576));
        assert_eq!(
            shape.site(),
            CutSite::Bank {
                bank: 1,
                row: 24576 - 16384 - 7872
            }
        );
        assert_eq!(
            qwen4_prefill_segment_rows(p, end, 8192, SessionBoundaries::default(), false),
            end - p
        );
    }
    #[test]
    fn pair_shape_splits_banks_and_cut_sites() {
        // 9,319 rows rebalanced; the default cache's snapshot at 8192 falls in bank B.
        let shape = PairShape::new([4672, 4647], 0, Some(8192));
        assert_eq!((shape.total(), shape.cut), (9319, Some(8192)));
        assert_eq!(
            shape.site(),
            CutSite::Bank {
                bank: 1,
                row: 8192 - 4672
            }
        );
        assert_eq!(shape.pieces(), vec![0..4672, 4672..8192, 8192..9319]);
        // At the A/B split: bank A runs whole, no sub-launch.
        let split = PairShape::new([8192, 8163], 0, Some(8192));
        assert_eq!(
            (split.site(), split.pieces()),
            (CutSite::Split, vec![0..8192, 8192..16355])
        );
        // A boundary at the group end (the driver publishes it) or outside is no cut.
        for boundary in [None, Some(0), Some(9319), Some(20000)] {
            let none = PairShape::new([4672, 4647], 0, boundary);
            assert_eq!(
                (none.cut, none.site(), none.pieces()),
                (None, CutSite::None, vec![0..4672, 4672..9319])
            );
        }
        // Absolute boundaries become group-local rows.
        let tail = PairShape::new([4224, 4180], 49152, Some(57344));
        assert_eq!(
            (tail.cut, tail.site()),
            (
                Some(8192),
                CutSite::Bank {
                    bank: 1,
                    row: 8192 - 4224
                }
            )
        );
        assert_eq!(
            PairShape::new([4224, 4180], 49152, Some(50000)).site(),
            CutSite::Bank { bank: 0, row: 848 }
        );
    }
    #[test]
    fn sub_launch_windows_cover_each_bank_at_boundary_rows() {
        for rows in [4647usize, 4672, 8163, 8192] {
            assert_eq!(
                core_windows(rows, None).as_slice(),
                &[CoreWindow { lo: 0, rows }]
            );
            for r in [1, 64, 4096, rows - 1] {
                let windows = core_windows(rows, Some(r));
                assert_eq!(
                    windows.as_slice(),
                    &[
                        CoreWindow { lo: 0, rows: r },
                        CoreWindow {
                            lo: r,
                            rows: rows - r
                        }
                    ]
                );
                // State stages after the first launch only: model the layer's ops.
                let ops: Vec<_> = windows
                    .iter()
                    .enumerate()
                    .flat_map(|(at, w)| {
                        let mut ops = vec![("core", w.lo, w.rows)];
                        if at == 0 {
                            ops.push(("stage", w.lo + w.rows, 0));
                        }
                        ops
                    })
                    .collect();
                assert_eq!(ops.iter().filter(|op| op.0 == "stage").count(), 1);
                assert_eq!(ops[1], ("stage", r, 0));
                assert_eq!(ops[2], ("core", r, rows - r));
            }
        }
    }
    #[test]
    fn launch_geometry_matches_pair_pieces_at_boundary_rows() {
        // The executor's launches (bank by bank, each bank's Core split at its
        // cut row) in group-local rows are the pair's pieces, at every cut class.
        for banks in [
            [8192, 8163],
            [4672, 4647],
            [4224, 4180],
            [6144, 6144],
            [8192, 1127],
        ] {
            let total = banks[0] + banks[1];
            let mut cuts = vec![
                None,
                Some(1),
                Some(64),
                Some(4096),
                Some(banks[0]),
                Some(banks[0] + 1),
                Some(total - 1),
            ];
            cuts.extend([Some(banks[0] - 1), Some(banks[0] + 4096.min(banks[1] - 1))]);
            for cut in cuts {
                let shape = PairShape { rows: banks, cut };
                let launches: Vec<_> = launch_geometry(shape)
                    .iter()
                    .flat_map(|bank| {
                        core_windows(bank.rows, bank.cut)
                            .into_iter()
                            .map(move |w| bank.offset + w.lo..bank.offset + w.lo + w.rows)
                    })
                    .collect();
                assert_eq!(launches, shape.pieces(), "{banks:?} cut {cut:?}");
            }
        }
    }
    fn fixture_calibration(file: &str) -> Calibration {
        Calibration::load(
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/npu_trunk")
                .join(file),
            &FLASH_NEXT,
        )
        .unwrap()
    }
    #[test]
    fn forced_modes_with_a_matched_calibration_run_the_planned_pairs() {
        // The built-in v4 overlay calibration: on and emulate take the planner's pairs
        // (rebalanced last pair included), exactly as auto does once armed.
        let (cal, _) = load_calibration(
            "/nonexistent/npu-trunk-calibration.json",
            "overlay:gfx1151-zab-v1",
        )
        .unwrap();
        let families = FamilyMask::from(coproc::select_families(&FLASH_NEXT, &cal).0);
        let star = coproc::crossover(&FLASH_NEXT, &cal, families.into());
        let pairs = |mode: NpuTrunkMode, rows: usize| {
            let plan = request_plan(&cal, mode, families, star, rows);
            let mut at = 0;
            let mut out = Vec::new();
            while at < rows {
                match planned_banks(&plan, 0, at, rows - at, coproc::CHUNK) {
                    Some(banks) => {
                        out.push((at, banks));
                        at += banks[0] + banks[1];
                    }
                    None => at = (at + coproc::CHUNK).min(rows),
                }
            }
            out
        };
        for mode in [NpuTrunkMode::On, NpuTrunkMode::Emulate, NpuTrunkMode::Auto] {
            assert_eq!(pairs(mode, 9319), vec![(0, [4672, 4647])], "{mode:?}");
            assert_eq!(
                pairs(mode, 57556).last(),
                Some(&(49152, [4224, 4180])),
                "{mode:?}"
            );
        }
        // The forced modes pair from the first row, auto only from the crossover.
        assert!(pairs(NpuTrunkMode::Auto, 4096).is_empty());
        assert_eq!(
            planned_banks(
                &request_plan(&cal, NpuTrunkMode::On, families, star, 9319),
                0,
                0,
                9319,
                4096
            ),
            None
        );
    }
    #[test]
    fn forced_request_subsets_use_only_the_static_prior() {
        let (cal, _) = load_calibration(
            "/nonexistent/npu-trunk-calibration.json",
            "overlay:gfx1151-zab-v1",
        )
        .unwrap();
        for names in ["qkv", "zab", "q", "kvidx", "qkv,q", "all"] {
            let mask = parse_family_override(names).unwrap();
            let on = request_plan(&cal, NpuTrunkMode::On, mask, None, 16384);
            let emulate = request_plan(&cal, NpuTrunkMode::Emulate, mask, Some(usize::MAX), 16384);
            assert_eq!(on, emulate, "{names}");
            assert_eq!(
                on,
                coproc::executable_plan_for(
                    &FLASH_NEXT,
                    &cal,
                    mask.into(),
                    &coproc::plan(&FLASH_NEXT, &cal, mask.into(), 16384),
                    16384,
                    Some(0)
                ),
                "{names}"
            );
        }
        for mode in [NpuTrunkMode::Auto, NpuTrunkMode::On, NpuTrunkMode::Emulate] {
            let plan = request_plan(&cal, mode, FamilyMask::NONE, Some(0), 16384);
            assert!(plan
                .steps
                .iter()
                .all(|step| matches!(step, coproc::Step::Gpu { .. })));
        }
    }
    #[test]
    fn row_adaptive_budget_demotes_pairs_beyond_it_to_gpu_chunks() {
        // A long-lived load meets many bank sizes: the first MAX_ADAPTIVE_CONTEXTS wave counts are admitted and keep their
        // pairs; a pair needing another wave count runs GPU-native, never through a full 8192-row NPU chain.
        let (cal, _) = load_calibration(
            "/nonexistent/npu-trunk-calibration.json",
            "overlay:gfx1151-zab-v1",
        )
        .unwrap();
        let families = FamilyMask::from(coproc::select_families(&FLASH_NEXT, &cal).0);
        let star = coproc::crossover(&FLASH_NEXT, &cal, families.into());
        let classes = |plan: &coproc::Plan| {
            plan.steps
                .iter()
                .flat_map(|step| match step {
                    coproc::Step::Pipe { a, b } => vec![a.len(), b.len()],
                    coproc::Step::Gpu { .. } => Vec::new(),
                })
                .map(|rows| padded_rows_of(transport_shape(Shape::GdnIn), rows))
                .filter(|&rows| rows < coproc::CHUNK)
                .collect::<Vec<_>>()
        };
        let (mut admitted, mut demoted) = (Vec::new(), 0);
        for rows in (8704..16384).step_by(256) {
            let full = request_plan(&cal, NpuTrunkMode::Auto, families, star, rows);
            let got = admitted_plan(full.clone(), &mut admitted, |kept| {
                coproc::executable_plan(
                    &FLASH_NEXT,
                    &cal,
                    kept,
                    rows,
                    armed_from(NpuTrunkMode::Auto, star),
                )
            });
            assert!(admitted.len() <= MAX_ADAPTIVE_CONTEXTS, "{rows}");
            assert!(classes(&got).iter().all(|r| admitted.contains(r)), "{rows}");
            if got != full {
                assert!(classes(&got).is_empty(), "{rows}");
                demoted += 1;
            }
            let end = got.steps.last().map(|step| match step {
                coproc::Step::Gpu { chunk } => chunk.end,
                coproc::Step::Pipe { b, .. } => b.end,
            });
            assert_eq!(end, Some(rows), "{rows}");
        }
        assert_eq!(admitted.len(), MAX_ADAPTIVE_CONTEXTS);
        assert!(demoted > 0);
        // An admitted wave count keeps its rebalanced pair.
        let nine = request_plan(&cal, NpuTrunkMode::Auto, families, star, 9319);
        let mut first = Vec::new();
        assert_eq!(
            admitted_plan(nine.clone(), &mut first, |_| unreachable!()),
            nine
        );
        assert_eq!(first, vec![4864]);
    }
    #[test]
    fn rebalanced_pairs_schedule_with_and_without_the_session_cache() {
        let cal = fixture_calibration("g80bf16-overlay-v2.json");
        let star = coproc::crossover(&FLASH_NEXT, &cal, FamilyMask::ALL.into());
        let schedule = |rows: usize, cache: bool| {
            let exec = coproc::executable_plan(
                &FLASH_NEXT,
                &cal,
                &coproc::plan(&FLASH_NEXT, &cal, FamilyMask::ALL.into(), rows),
                rows,
                star,
            );
            let boundaries: Vec<usize> = if cache {
                (1..=rows / 8192).map(|k| k * 8192).collect()
            } else {
                Vec::new()
            };
            schedule_with(rows, &boundaries, &|pos| {
                coproc::planned_pair(&exec, pos).map(|(a, b)| [a, b])
            })
        };
        // 9,319: 4672 + 4647; the 8192 snapshot is bank B's local row 3520.
        assert_eq!(schedule(9319, false), vec![(0, 9319, Some(4672), None)]);
        assert_eq!(
            schedule(9319, true),
            vec![(0, 9319, Some(4672), Some(8192))]
        );
        assert_eq!(
            PairShape::new([4672, 4647], 0, Some(8192)).site(),
            CutSite::Bank { bank: 1, row: 3520 }
        );
        // 16,355: the natural 8192 + 8163, the snapshot at the split.
        assert_eq!(schedule(16355, false), vec![(0, 16355, Some(8192), None)]);
        assert_eq!(
            schedule(16355, true),
            vec![(0, 16355, Some(8192), Some(8192))]
        );
        // 57,556: three natural pairs, then the 8,404-row tail as 4224 + 4180
        // whose 57,344 snapshot is bank B's local row 3968.
        let tail = |cache: bool| schedule(57556, cache).pop().unwrap();
        assert_eq!(tail(false), (49152, 8404, Some(4224), None));
        assert_eq!(tail(true), (49152, 8404, Some(4224), Some(8192)));
        assert_eq!(schedule(57556, true).len(), 4);
        assert_eq!(
            PairShape::new([4224, 4180], 49152, Some(57344)).site(),
            CutSite::Bank { bank: 1, row: 3968 }
        );
    }
    #[test]
    fn emulate_and_on_share_the_pair_launch_geometry() {
        // Both modes run `pair_scoped`, whose launches follow `launch_geometry`
        // alone: banks at the plan's rows, each bank's Core split only at its
        // own snapshot row, state staged after the first launch (or after bank A
        // when the row is the A/B split). Pin it for the C1 sizes, cache on/off.
        let cal = fixture_calibration("g80bf16-overlay-v2.json");
        let star = coproc::crossover(&FLASH_NEXT, &cal, FamilyMask::ALL.into());
        for (rows, cache, expected) in [
            (
                9319usize,
                false,
                vec![[(0, 4672, None, false), (4672, 4647, None, false)]],
            ),
            (
                9319,
                true,
                vec![[(0, 4672, None, false), (4672, 4647, Some(3520), false)]],
            ),
            (
                16355,
                false,
                vec![[(0, 8192, None, false), (8192, 8163, None, false)]],
            ),
            (
                16355,
                true,
                vec![[(0, 8192, None, true), (8192, 8163, None, false)]],
            ),
            (
                57556,
                false,
                vec![
                    [(0, 8192, None, false), (8192, 8192, None, false)],
                    [(0, 8192, None, false), (8192, 8192, None, false)],
                    [(0, 8192, None, false), (8192, 8192, None, false)],
                    [(0, 4224, None, false), (4224, 4180, None, false)],
                ],
            ),
            (
                57556,
                true,
                vec![
                    [(0, 8192, None, true), (8192, 8192, None, false)],
                    [(0, 8192, None, true), (8192, 8192, None, false)],
                    [(0, 8192, None, true), (8192, 8192, None, false)],
                    [(0, 4224, None, false), (4224, 4180, Some(3968), false)],
                ],
            ),
        ] {
            let exec = coproc::executable_plan(
                &FLASH_NEXT,
                &cal,
                &coproc::plan(&FLASH_NEXT, &cal, FamilyMask::ALL.into(), rows),
                rows,
                star,
            );
            let boundaries: Vec<usize> = if cache {
                (1..=rows / 8192).map(|k| k * 8192).collect()
            } else {
                Vec::new()
            };
            let groups = schedule_with(rows, &boundaries, &|pos| {
                coproc::planned_pair(&exec, pos).map(|(a, b)| [a, b])
            });
            let geometries: Vec<_> = groups
                .iter()
                .filter_map(|&(start, total, split, _)| {
                    let a = split?;
                    let boundary = boundaries.iter().copied().find(|&b| b > start);
                    let shape = PairShape::new([a, total - a], start, boundary);
                    let geometry = launch_geometry(shape);
                    // The launches of that geometry, group-local, are the shape's pieces.
                    let launches: Vec<_> = geometry
                        .iter()
                        .flat_map(|bank| {
                            core_windows(bank.rows, bank.cut)
                                .into_iter()
                                .map(move |w| bank.offset + w.lo..bank.offset + w.lo + w.rows)
                        })
                        .collect();
                    assert_eq!(launches, shape.pieces(), "{rows} cache {cache} at {start}");
                    assert_eq!(
                        geometry
                            .iter()
                            .filter(|bank| bank.cut.is_some() || bank.stage_after)
                            .count(),
                        usize::from(shape.cut.is_some())
                    );
                    Some(geometry.map(|bank| (bank.offset, bank.rows, bank.cut, bank.stage_after)))
                })
                .collect();
            assert_eq!(geometries, expected, "{rows} cache {cache}");
        }
    }
    #[test]
    fn executable_plan_and_schedule_match_python_goldens() {
        let vectors: serde_json::Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/npu_trunk/golden-exec.json"
        )))
        .unwrap();
        let num = |v: &serde_json::Value| v.as_u64().unwrap() as usize;
        let opt = |v: &serde_json::Value| (!v.is_null()).then(|| num(v));
        let mut cases = 0;
        for v in vectors.as_array().unwrap() {
            let file = v["file"].as_str().unwrap();
            let cal = fixture_calibration(file);
            let star = coproc::crossover(&FLASH_NEXT, &cal, FamilyMask::ALL.into());
            assert_eq!(star, v["n_star"].as_u64().map(|n| n as usize), "{file}");
            for case in v["cases"].as_array().unwrap() {
                let rows = num(&case["rows"]);
                let exec = coproc::executable_plan(
                    &FLASH_NEXT,
                    &cal,
                    &coproc::plan(&FLASH_NEXT, &cal, FamilyMask::ALL.into(), rows),
                    rows,
                    star,
                );
                let steps: Vec<_> = case["steps"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|s| {
                        let range = |r: &serde_json::Value| num(&r[0])..num(&r[1]);
                        if let Some(r) = s.get("gpu") {
                            coproc::Step::Gpu { chunk: range(r) }
                        } else {
                            coproc::Step::Pipe {
                                a: range(&s["pipe"][0]),
                                b: range(&s["pipe"][1]),
                            }
                        }
                    })
                    .collect();
                assert_eq!(exec.steps, steps, "{file} rows={rows}");
                for (value, key) in [(exec.t_pipe_ms, "t_pipe_ms"), (exec.t_gpu_ms, "t_gpu_ms")] {
                    let expected = case[key].as_f64().unwrap();
                    assert!(
                        (value - expected).abs() <= expected.abs() * 1e-6,
                        "{file} {rows} {key}: {value} != {expected}"
                    );
                }
                let cache: Vec<usize> = (1..=rows / 8192).map(|k| k * 8192).collect();
                for (key, boundaries) in [("groups_off", &[][..]), ("groups_on", &cache[..])] {
                    let actual = schedule_with(rows, boundaries, &|pos| {
                        coproc::planned_pair(&exec, pos).map(|(a, b)| [a, b])
                    });
                    let expected: Vec<_> = case[key]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|g| (num(&g[0]), num(&g[1]), opt(&g[2]), opt(&g[3])))
                        .collect();
                    assert_eq!(actual, expected, "{file} rows={rows} {key}");
                }
                cases += 1;
            }
        }
        assert_eq!(cases, 24);
    }
    #[test]
    fn capture_capacity_only_doubles_for_actual_pipeline() {
        use crate::bundle::spec_hidden_capacity_rows;
        assert_eq!(spec_hidden_capacity_rows(8192, 8192, true), Some(16384));
        assert_eq!(spec_hidden_capacity_rows(8192, 8192, false), Some(8192));
        assert_eq!(spec_hidden_capacity_rows(20000, 8192, true), Some(20000));
        assert_eq!(spec_hidden_capacity_rows(8192, usize::MAX, true), None);
    }
}

pub(super) fn run_pair_request(
    owner: &mut Qwen4GpuForward,
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    tokens: &[u32],
    offset: usize,
    banks: [usize; 2],
    logits: Option<&GpuTensor>,
    top1: Option<&GpuTensor>,
    capture: Option<&GpuTensor>,
    output: Qwen4OutputPolicy,
) -> Result<(), Qwen4GpuForwardError> {
    let all = matches!(output, Qwen4OutputPolicy::Rows(Qwen4OutputRows::All));
    let logits = logits.map(|t| {
        if all {
            t.sub_offset(
                offset * bundle.config.vocab_size,
                tokens.len() * bundle.config.vocab_size,
            )
        } else {
            t.sub_offset(0, t.numel())
        }
    });
    let top1 = top1.map(|t| {
        if all {
            t.sub_offset(offset * 4, tokens.len() * 4)
        } else {
            t.sub_offset(0, t.numel())
        }
    });
    let capture = capture.map(|t| {
        t.sub_offset(
            offset * program_dims(&bundle.config).wide(),
            tokens.len() * program_dims(&bundle.config).wide(),
        )
    });
    let mut npu = owner.npu_trunk.take().expect("on owner");
    let result = npu.pair(
        owner,
        bundle,
        gpu,
        tokens,
        banks,
        logits.as_ref(),
        top1.as_ref(),
        capture.as_ref(),
        output,
    );
    if let Err(e) = &result {
        npu.contain_fault(e);
    }
    owner.npu_trunk = Some(npu);
    result
}
