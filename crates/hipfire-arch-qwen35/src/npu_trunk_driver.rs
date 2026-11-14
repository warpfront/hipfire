//! Device half of the qwen3.8-27b bank-pair prefill. `on` packs a bank's rotated F16 activation
//! into the relay's A arena, kicks the layer's resident W8 panels and later copies the BF16 C
//! arena into the bank's in-projection scratch; `emulate` runs the same W8 x A8 GEMM and
//! `bf16_direct` epilogue on the GPU twin, so `on == emulate` byte for byte. Every 27B family is
//! owned (no family selection without a calibration).

use super::{pack_group, Mode, FAMILIES, FAULT_CHAIN, K, STAGE_MAP, SWITCH};
use crate::qwen35::{
    DeltaNetState, LayerType, LayerWeights, PrefillBatchScratch, Qwen35Config, Qwen35Weights,
};
use crate::speculative::DeltaNetSnapshot;
use hip_bridge::{HipError, HipResult};
use hipfire_runtime::coproc::containment::{FaultClass, FaultLatch};
use hipfire_runtime::coproc::Calibration;
use peacemaker_npu::kernels::gemm_g80_n512::build_panel_program_n512;
use railgun::npu::trunk::{
    LayerPanels, NpuTrunkRuntime, OpenOptions, PanelBuild, PanelProgram, Seq, Shape, TxnBuilder,
};
use rdna_compute::gdn_stream::GdnStreamMeta;
use rdna_compute::npu_trunk::{NpuBf16Layout, NpuTrunkOutput};
use rdna_compute::tensor_ops::{npu_trunk_copy_projection_arena, Bf16Arena};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::cell::{Cell, RefCell};
use std::ffi::c_void;

/// Rows of one bank: the relay arenas, the A-stream sidecars and bank B's scratch.
pub const BANK_ROWS: usize = 8192;
/// Smallest bank B worth a pair; shorter tails run GPU-only in every mode.
pub const MIN_BANK_B_ROWS: usize = 512;
/// Transport shape per layer kind (0 = GDN, 1 = FA).
const SHAPES: [Shape; 2] = [Shape::DenseGdnIn, Shape::DenseFaIn];

/// Live fused columns of kind `kind`: its families' widths in declaration order.
fn live_cols(kind: usize) -> usize {
    FAMILIES
        .iter()
        .filter(|f| f.kind == kind)
        .map(|f| f.cols)
        .sum()
}

/// `(col0, cols, destination)` of every family of `kind` in bank scratch `pbs`.
fn outputs(kind: usize, pbs: &PrefillBatchScratch) -> Vec<NpuTrunkOutput<'_>> {
    let dst = [
        &pbs.dn_qkv_batch,
        &pbs.dn_z_batch,
        &pbs.dn_alpha_batch,
        &pbs.dn_beta_batch,
        &pbs.fa_q_full_batch,
        &pbs.fa_k_batch,
        &pbs.fa_v_batch,
    ];
    let mut col0 = 0;
    FAMILIES
        .iter()
        .zip(dst)
        .filter(|(f, _)| f.kind == kind)
        .map(|(f, tensor)| {
            let out = NpuTrunkOutput {
                col0,
                cols: f.cols,
                tensor,
            };
            col0 += f.cols;
            out
        })
        .collect()
}

fn fail(e: impl std::fmt::Display) -> HipError {
    HipError::new(0, &format!("{}: {e}", SWITCH.tag))
}

fn stream(gpu: &Gpu) -> *mut c_void {
    gpu.active_stream
        .as_ref()
        .map_or(std::ptr::null_mut(), |s| s.as_raw())
}

/// The GPU twin of one layer: `[live][K]` W8 codes and the compact `(Sm, Se)` column sidecars.
struct Twin {
    w: GpuTensor,
    scales: GpuTensor,
}

pub struct NpuTrunk {
    mode: Mode,
    /// Layer kind per layer (0 = GDN, 1 = FA).
    kinds: Vec<usize>,
    /// `emulate` only, per layer.
    twins: Vec<Twin>,
    /// `emulate` only: the N1280 A stream and the `[rows][cols]` I32 sums.
    emulation: Option<(GpuTensor, GpuTensor)>,
    /// Per-bank compact activation sidecars the pack writes.
    d: [GpuTensor; 2],
    /// `[BANK_ROWS][K]` F16 pack input; the prepare casts the bank's F32 rotated rows into it.
    x16: GpuTensor,
    runtime: Option<NpuTrunkRuntime>,
    /// `on` only: borrowed views of the relay's C arenas, `[bank][kind]`.
    c: Vec<[GpuTensor; 2]>,
    /// Bank B's private scratch; bank A uses the caller's widened scratch.
    pub bank_b: PrefillBatchScratch,
    /// The recurrent state at the last pair's bank boundary, staged layer by layer (allocated
    /// at the first pair), and that boundary's position until the caller adopts it.
    stage: RefCell<Option<DeltaNetSnapshot>>,
    cut: Cell<Option<usize>>,
    /// `auto` only: the bound calibration that prices each candidate pair against GPU-only.
    cal: Option<Calibration>,
    /// Process-wide fault latch: the first NPU fault turns the route off for good.
    latch: RefCell<FaultLatch>,
    /// Set by a failed relay call of the current pair (versus a GPU HipError).
    npu_fault: Cell<bool>,
    /// The recurrent state before the current pair, restored when an NPU fault reruns its rows
    /// GPU-only (allocated at the first pair).
    pre: RefCell<Option<DeltaNetSnapshot>>,
}

// SAFETY: the driver exclusively owns the relay's HIP Rc graph; none of it escapes, and the
// model owner moves (never shares) it between threads, as with the Flash-Next trunk.
unsafe impl Send for NpuTrunk {}

impl NpuTrunk {
    /// Arms `mode` (`on`, `emulate`, or an admitted `auto` with its bound calibration) for a
    /// loaded 27B, or refuses with the reason. `auto` opens the NPU exactly as `on` does.
    pub fn load(
        gpu: &mut Gpu,
        weights: &Qwen35Weights,
        config: &Qwen35Config,
        mode: Mode,
        cal: Option<Calibration>,
    ) -> Result<Self, String> {
        let started = std::time::Instant::now();
        if gpu.arch != "gfx1151" {
            return Err(fail("requires gfx1151").to_string());
        }
        let kinds: Vec<usize> = config
            .layer_types
            .iter()
            .map(|t| usize::from(matches!(t, LayerType::FullAttention)))
            .collect();
        STAGE_MAP
            .check_model(&kinds, config.dim)
            .map_err(|e| fail(e).to_string())?;
        // Every source is checked before any download, pack or allocation.
        for (i, layer) in weights.layers.iter().enumerate() {
            let refs = match layer {
                LayerWeights::DeltaNet(l) => vec![&l.wqkv, &l.wz, &l.w_alpha, &l.w_beta],
                LayerWeights::FullAttn(l) => vec![&l.wq, &l.wk, &l.wv],
                _ => return Err(fail(format!("layer {i} is not dense GDN/FA")).to_string()),
            };
            if refs
                .iter()
                .any(|w| w.gpu_dtype != DType::MQ4G256V2 || w.k != K)
            {
                return Err(fail(format!(
                    "layer {i} in-projections must be MQ4G256V2 with K={K}"
                ))
                .to_string());
            }
        }
        let twin = mode == Mode::Emulate;
        let row_bytes = K / 256 * rdna_compute::MQ4V2_GROUP_BYTES;
        let download = |gpu: &mut Gpu, buf: &GpuTensor, rows: usize| -> HipResult<Vec<u8>> {
            let mut bytes = vec![0u8; rows * row_bytes];
            gpu.hip.memcpy_dtoh(&mut bytes, &buf.buf)?;
            Ok(bytes)
        };
        let mut allocations: Vec<GpuTensor> = Vec::new();
        let result = (|| -> Result<Self, String> {
            let mut resident = Vec::new();
            let mut twins = Vec::new();
            let (mut panel_bytes, mut twin_bytes) = (0usize, 0usize);
            for (index, layer) in weights.layers.iter().enumerate() {
                // Families in STAGE_MAP declaration order; `wz` may carry the folded beta/alpha
                // rows after z, so only its first 6144 rows are read.
                let groups: Vec<(usize, Vec<Vec<u8>>)> = match layer {
                    LayerWeights::DeltaNet(l) => vec![
                        (
                            0,
                            vec![download(gpu, &l.wqkv.buf, l.wqkv.m).map_err(fail_s)?],
                        ),
                        (
                            1,
                            vec![
                                download(gpu, &l.wz.buf, FAMILIES[1].cols).map_err(fail_s)?,
                                download(gpu, &l.w_alpha.buf, l.w_alpha.m).map_err(fail_s)?,
                                download(gpu, &l.w_beta.buf, l.w_beta.m).map_err(fail_s)?,
                            ],
                        ),
                    ],
                    LayerWeights::FullAttn(l) => vec![
                        (2, vec![download(gpu, &l.wq.buf, l.wq.m).map_err(fail_s)?]),
                        (
                            3,
                            vec![
                                download(gpu, &l.wk.buf, l.wk.m).map_err(fail_s)?,
                                download(gpu, &l.wv.buf, l.wv.m).map_err(fail_s)?,
                            ],
                        ),
                    ],
                    _ => unreachable!("checked above"),
                };
                let mut panels = Vec::new();
                let (mut codes, mut scales) = (Vec::new(), Vec::new());
                for (group, parts) in &groups {
                    let parts: Vec<&[u8]> = parts.iter().map(Vec::as_slice).collect();
                    let packed =
                        pack_group(*group, &parts, twin).map_err(|e| fail(e).to_string())?;
                    let live = packed.codes.len() / K;
                    panels.extend(packed.panels);
                    codes.extend(packed.codes);
                    scales.extend(packed.scales.into_iter().take(if twin { live } else { 0 }));
                }
                if twin {
                    let sidecars: Vec<u8> = scales
                        .iter()
                        .flat_map(|(m, e)| m.to_le_bytes().into_iter().chain(e.to_le_bytes()))
                        .collect();
                    let codes: Vec<u8> = codes.into_iter().map(|c| c as u8).collect();
                    let w = gpu.upload_raw(&codes, &[codes.len()]).map_err(fail_s)?;
                    allocations.push(w.sub_offset(0, w.numel()));
                    let s = gpu
                        .upload_raw(&sidecars, &[sidecars.len()])
                        .map_err(fail_s)?;
                    allocations.push(s.sub_offset(0, s.numel()));
                    twin_bytes += codes.len() + sidecars.len();
                    twins.push(Twin { w, scales: s });
                } else {
                    panel_bytes += panels.iter().map(Vec::len).sum::<usize>();
                    resident.push(LayerPanels {
                        layer: index,
                        shape: SHAPES[kinds[index]],
                        panels,
                    });
                }
            }
            let zeros = |gpu: &mut Gpu, bytes: usize, allocations: &mut Vec<GpuTensor>| {
                let t = gpu.zeros(&[bytes], DType::Raw).map_err(fail_s)?;
                allocations.push(t.sub_offset(0, t.numel()));
                Ok::<_, String>(t)
            };
            let d = [
                zeros(gpu, BANK_ROWS * 4, &mut allocations)?,
                zeros(gpu, BANK_ROWS * 4, &mut allocations)?,
            ];
            let x16 = gpu.zeros(&[BANK_ROWS * K], DType::F16).map_err(fail_s)?;
            allocations.push(x16.sub_offset(0, x16.numel()));
            let emulation = if twin {
                let a = NpuBf16Layout {
                    padded_rows: BANK_ROWS,
                    k: K,
                    panels: 1,
                    n512: false,
                }
                .a_bytes();
                let cols = live_cols(0).max(live_cols(1));
                Some((
                    zeros(gpu, a, &mut allocations)?,
                    zeros(gpu, BANK_ROWS * cols * 4, &mut allocations)?,
                ))
            } else {
                None
            };
            let (runtime, c) = if twin {
                (None, Vec::new())
            } else {
                let program = |shape: Shape, rows: usize| {
                    build_panel_program_n512(rows, shape.pitch(), 0, K).map(|d| PanelBuild {
                        pdi: d.pdi,
                        txn: d.insts,
                    })
                };
                let programs = SHAPES
                    .iter()
                    .map(|&shape| {
                        program(shape, BANK_ROWS).map(|b| PanelProgram {
                            shape,
                            pdi: b.pdi,
                            txn: b.txn,
                            padded_rows: BANK_ROWS,
                            pitch: shape.pitch(),
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| fail(e).to_string())?;
                // A bank of fewer rows runs the program of its own wave count, built on first use.
                let builder: TxnBuilder = std::sync::Arc::new(move |shape, rows| match shape {
                    Shape::DenseGdnIn | Shape::DenseFaIn => program(shape, rows),
                    other => Err(format!("{other:?} is not a qwen3.8-27b trunk shape")),
                });
                let pci = gpu.hip.device_pci_bus_id(gpu.device_id).map_err(fail_s)?;
                // Developer fault injection: force the watchdog of the Nth relay chain (the real
                // NPU-timeout path).
                let fault_chain = hipfire_config::developer_var(FAULT_CHAIN)
                    .ok()
                    .and_then(|v| v.trim().parse::<u64>().ok().filter(|&n| n > 0));
                let rt = NpuTrunkRuntime::open_with_builder(
                    programs,
                    resident,
                    2,
                    BANK_ROWS,
                    &pci,
                    OpenOptions {
                        fault_chain,
                        ..OpenOptions::default()
                    },
                    Some(builder),
                )
                .map_err(|e| fail(e).to_string())?;
                let mut c = Vec::new();
                for bank in 0..2 {
                    let view = |shape: Shape| -> Result<GpuTensor, String> {
                        let arena = rt.c_arena(bank, shape).map_err(|e| fail(e).to_string())?;
                        // Borrowed: the runtime owns the registration until every bank retires.
                        Ok(GpuTensor {
                            buf: unsafe {
                                hip_bridge::DeviceBuffer::from_raw(
                                    arena.device as *mut c_void,
                                    arena.len,
                                )
                            },
                            shape: vec![BANK_ROWS, shape.pitch()],
                            dtype: DType::BF16,
                        })
                    };
                    c.push([view(SHAPES[0])?, view(SHAPES[1])?]);
                }
                (Some(rt), c)
            };
            let bank_b =
                PrefillBatchScratch::new_opt(gpu, config, BANK_ROWS, false).map_err(fail_s)?;
            eprintln!(
                "{}: mode={mode} families=all contract=bf16_direct npu_panels_bytes={panel_bytes} gpu_twin_bytes={twin_bytes} runtime={} load_ms={:.0}",
                SWITCH.tag,
                if runtime.is_some() { "open" } else { "not-opened" },
                started.elapsed().as_secs_f64() * 1e3
            );
            hipfire_config::load_summary::note(
                hipfire_config::load_summary::Facet::Prefill,
                format!("qwen35 npu trunk mode={mode}"),
            );
            Ok(Self {
                mode,
                kinds: kinds.clone(),
                twins,
                emulation,
                d,
                x16,
                runtime,
                c,
                bank_b,
                stage: RefCell::new(None),
                cut: Cell::new(None),
                cal,
                latch: RefCell::new(FaultLatch::default()),
                npu_fault: Cell::new(false),
                pre: RefCell::new(None),
            })
        })();
        if result.is_err() {
            for tensor in allocations {
                let _ = gpu.free_tensor(tensor);
            }
        }
        result
    }

    /// Rows of the pair a chunk loop should run at `remaining` rows when its next chunk is a
    /// full `BANK_ROWS` bank: `Some(b)` for bank B's rows, `None` for the ordinary GPU route.
    pub fn bank_b_rows(remaining: usize) -> Option<usize> {
        let b = remaining.checked_sub(BANK_ROWS)?.min(BANK_ROWS);
        (b >= MIN_BANK_B_ROWS).then_some(b)
    }

    /// Whether the pair at prompt position `pos` (bank A = `BANK_ROWS` rows, bank B = `b` rows)
    /// runs: never after an NPU fault; always in `on` / `emulate`; in `auto` per
    /// [`super::pair_beats_gpu`].
    pub fn admits_pair(&self, pos: usize, b: usize) -> bool {
        if self.latch.borrow().tripped() {
            return false;
        }
        let Some(cal) = &self.cal else {
            return true;
        };
        super::pair_beats_gpu(cal, pos, b)
    }

    /// Before a pair: keep the whole recurrent state (GDN stream carry and cursors included), so
    /// an NPU fault can rerun the pair's rows GPU-only. KV needs no copy: the rerun rewrites the
    /// same positions.
    pub fn save_pre(&self, gpu: &mut Gpu, dn: &DeltaNetState) -> HipResult<()> {
        self.npu_fault.set(false);
        let mut pre = self.pre.borrow_mut();
        if pre.is_none() {
            *pre = Some(DeltaNetSnapshot::new_for(gpu, dn)?);
        }
        pre.as_mut().expect("allocated above").save_from(dn, gpu)
    }

    /// After a failed pair (the relay already aborted, and the pair's GDN forward failed closed
    /// with a state reset): an NPU fault latches the route off, restores the pre-pair state and
    /// returns `Ok(true)`, so the caller reruns the same rows on the GPU route (the request then
    /// equals `off` from this pair on). A GPU HipError returns `Ok(false)`: its context is
    /// sticky, the request fails with it.
    pub fn contain(&self, gpu: &mut Gpu, dn: &mut DeltaNetState, e: &HipError) -> HipResult<bool> {
        if !self.npu_fault.take() {
            return Ok(false);
        }
        if self
            .latch
            .borrow_mut()
            .trip(FaultClass::NpuRuntime, &e.to_string())
        {
            eprintln!(
                "{}: fault ({}): {e}; route off for this process, pair rows rerun GPU-only",
                SWITCH.tag,
                FaultClass::NpuRuntime.as_str()
            );
        }
        gpu.hip.device_synchronize()?;
        self.pre
            .borrow()
            .as_ref()
            .expect("save_pre ran before the pair")
            .restore_to(dn, gpu)?;
        Ok(true)
    }

    /// A failed relay call: marks the pair's fault as the NPU's.
    fn npu_fail(&self, e: impl std::fmt::Display) -> HipError {
        self.npu_fault.set(true);
        fail(e)
    }

    /// Projects layer `layer` of bank `bank` from `pbs.x_rot_batch` (RMSNorm + rotation + AWQ
    /// already applied, F32): `on` packs and kicks, `emulate` writes the in-projection scratch.
    pub fn prepare(
        &self,
        gpu: &mut Gpu,
        layer: usize,
        bank: usize,
        pbs: &PrefillBatchScratch,
        rows: usize,
    ) -> HipResult<Option<Seq>> {
        let kind = self.kinds[layer];
        gpu.cast_f32_to_f16_pitched(&pbs.x_rot_batch, &self.x16, rows, K, K)?;
        let x = &self.x16;
        let Some(rt) = &self.runtime else {
            let (a, sums) = self.emulation.as_ref().expect("emulate scratch");
            let live = live_cols(kind);
            let layout = NpuBf16Layout {
                padded_rows: rows.div_ceil(256) * 256,
                k: K,
                panels: live.div_ceil(1280),
                n512: false,
            };
            gpu.npu_trunk_pack_a_bf16(x, rows, layout, a, &self.d[bank])?;
            let twin = &self.twins[layer];
            gpu.npu_trunk_emulate_bf16(
                a,
                &twin.w,
                &twin.scales,
                &self.d[bank],
                sums,
                layout,
                rows,
                live,
                &outputs(kind, pbs),
            )?;
            return Ok(None);
        };
        let shape = SHAPES[kind];
        unsafe {
            rt.prepare_bank(bank, stream(gpu))
                .map_err(|e| self.npu_fail(e))?;
            let layout = NpuBf16Layout {
                padded_rows: rt.chain_rows(shape, rows),
                k: K,
                panels: shape.panels(),
                n512: true,
            };
            gpu.npu_trunk_pack_a_bf16_ptr(
                x,
                rows,
                layout,
                rt.a_out_arena(bank).map_err(|e| self.npu_fail(e))?.device as *mut c_void,
                &self.d[bank],
            )?;
            rt.kick(layer, shape, bank, rows, stream(gpu))
                .map(Some)
                .map_err(|e| self.npu_fail(e))
        }
    }

    /// `on`: waits for `seq` on the stream, copies the C arena into the bank's in-projection
    /// scratch and retires the bank. `emulate` already wrote the scratch in [`Self::prepare`].
    pub fn consume(
        &self,
        gpu: &mut Gpu,
        layer: usize,
        bank: usize,
        seq: Option<Seq>,
        pbs: &PrefillBatchScratch,
        rows: usize,
    ) -> HipResult<()> {
        let (Some(rt), Some(seq)) = (&self.runtime, seq) else {
            return Ok(());
        };
        let kind = self.kinds[layer];
        unsafe {
            rt.wait_gpu(seq, stream(gpu))
                .map_err(|e| self.npu_fail(e))?
        };
        let arena = &self.c[bank][kind];
        for out in outputs(kind, pbs) {
            npu_trunk_copy_projection_arena(
                gpu,
                Bf16Arena {
                    tensor: arena,
                    row_pitch_elems: SHAPES[kind].pitch(),
                    col0_elems: out.col0,
                },
                out.tensor,
                rows,
                out.cols,
                out.cols,
                0,
                0,
            )?;
        }
        unsafe {
            rt.retire_bank(seq, stream(gpu))
                .map_err(|e| self.npu_fail(e))
        }
    }

    /// After the pair's last consume: surfaces a relay failure before the pair's outputs are used.
    pub fn finish(&self, last: Option<Seq>) -> HipResult<()> {
        let (Some(rt), Some(seq)) = (&self.runtime, last) else {
            return Ok(());
        };
        rt.wait_host(seq).map_err(|e| self.npu_fail(e))?;
        rt.check_error().map_err(|e| self.npu_fail(e))
    }

    /// A pair failed: release every outstanding GPU wait so the stream drains. `on` stays failed
    /// closed (the relay's sticky error); the caller's rollback restores the pre-pair state.
    pub fn abort(&self, reason: &str) {
        if let Some(rt) = &self.runtime {
            rt.abort(reason);
        }
    }

    pub fn mode(&self) -> Mode {
        self.mode
    }

    /// Stages recurrent layer `slot` of `dn` (its stream cursor `meta`, the pair's unpublished
    /// candidate) as the pair's bank-boundary state; call between the two banks' runs of that
    /// layer.
    pub fn stage_layer(
        &self,
        gpu: &mut Gpu,
        dn: &DeltaNetState,
        slot: usize,
        meta: GdnStreamMeta,
    ) -> HipResult<()> {
        let mut stage = self.stage.borrow_mut();
        if stage.is_none() {
            *stage = Some(DeltaNetSnapshot::new_for(gpu, dn)?);
        }
        stage
            .as_mut()
            .expect("allocated above")
            .save_layer_from(dn, slot, meta, gpu)
    }

    /// A completed pair staged the state at position `pos`.
    pub fn set_cut(&self, pos: usize) {
        self.cut.set(Some(pos));
    }

    /// Forgets a cut no caller took.
    pub fn discard_cut(&self) {
        self.cut.set(None);
    }

    /// Stages all of `dn` (its published cursors) as the bank-boundary state at `pos`: the
    /// GPU-only rerun of a contained pair offers the same cut the pair would have.
    pub fn cut_here(&self, gpu: &mut Gpu, dn: &DeltaNetState, pos: usize) -> HipResult<()> {
        {
            let mut stage = self.stage.borrow_mut();
            if stage.is_none() {
                *stage = Some(DeltaNetSnapshot::new_for(gpu, dn)?);
            }
            stage
                .as_mut()
                .expect("allocated above")
                .save_from(dn, gpu)?;
        }
        self.set_cut(pos);
        Ok(())
    }

    /// The staged bank-boundary state and its position, once per pair.
    pub fn take_cut(&self) -> Option<(usize, std::cell::Ref<'_, DeltaNetSnapshot>)> {
        let pos = self.cut.take()?;
        let stage = std::cell::Ref::filter_map(self.stage.borrow(), Option::as_ref).ok()?;
        Some((pos, stage))
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        let Self {
            mode: _,
            kinds: _,
            twins,
            emulation,
            d,
            x16,
            runtime,
            c,
            bank_b,
            stage,
            cut: _,
            cal: _,
            latch: _,
            npu_fault: _,
            pre,
        } = self;
        drop(c);
        if let Some(rt) = runtime {
            if let Err(e) = rt.close() {
                eprintln!("{}: close: {e}", SWITCH.tag);
            }
        }
        let _ = bank_b.free_gpu(gpu);
        for snap in [stage.into_inner(), pre.into_inner()].into_iter().flatten() {
            snap.free_gpu(gpu);
        }
        for t in twins
            .into_iter()
            .flat_map(|t| [t.w, t.scales])
            .chain(d)
            .chain([x16])
            .chain(emulation.into_iter().flat_map(|(a, s)| [a, s]))
        {
            let _ = gpu.free_tensor(t);
        }
    }
}

fn fail_s(e: HipError) -> String {
    fail(e).to_string()
}
