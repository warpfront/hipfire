// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! DSpine drafter sidecar metadata (arch 24).
//!
//! `dspine_convert` writes the fail-closed contract at conversion time; the
//! loader re-checks it here before any bridge tensor reaches the GPU (the
//! plan doc's double-check rule, mirroring the upstream runtime's own config
//! validation). A sidecar that claims arch 24 but violates the contract is a
//! pointing refusal, never a silent degradation.

use crate::dflash::DflashConfig;
use crate::hfq::HfqFile;
use rdna_compute::Gpu;
use rdna_compute::GpuTensor;

/// Parsed + re-validated DSpine sidecar metadata. `trunk` carries the DFlash
/// decoder knobs (the trunk rides the existing DFlash draft-load path); the
/// remaining fields are the code-bridge contract the arch-24 loader consumes.
#[derive(Debug)]
pub struct DspineConfig {
    pub trunk: DflashConfig,
    pub code_size: usize,
    pub bridge_rank: usize,
    pub second_pass_mode: String,
    pub first_exact: usize,
    pub head_topk: usize,
    pub feature_layers: Vec<usize>,
    /// Whether the sidecar carries the prebuilt TG/TSV sender tables
    /// (converter `--target-hfq` pass). `false` = legal older sidecar;
    /// the loader then stays on the plain greedy-argmax refinement.
    pub serial_tables: bool,
}

impl DspineConfig {
    /// Open an HFQ sidecar and parse its metadata. `Ok(None)` when the file
    /// is not a DSpine sidecar (no `dspine` metadata block) — callers pick
    /// the right refusal for that case; `Err` carries the contract violation.
    pub fn from_hfq(hfq: &HfqFile) -> Result<Option<Self>, String> {
        let meta: serde_json::Value = serde_json::from_str(&hfq.metadata_json)
            .map_err(|e| format!("metadata JSON malformed: {e}"))?;
        Self::from_metadata(&meta)
    }

    /// Metadata-only parse + contract double-check.
    pub fn from_metadata(meta: &serde_json::Value) -> Result<Option<Self>, String> {
        let dsp = match meta.get("dspine") {
            Some(v) => v,
            None => return Ok(None),
        };
        let trunk = DflashConfig::from_metadata(meta)
            .ok_or("sidecar is missing the `dflash` trunk metadata block")?;

        let code_size = dsp
            .get("code_size")
            .and_then(|v| v.as_u64())
            .ok_or("dspine block missing code_size")? as usize;
        let bridge_rank = dsp
            .get("bridge_rank")
            .and_then(|v| v.as_u64())
            .ok_or("dspine block missing bridge_rank")? as usize;
        let mode = dsp
            .get("second_pass")
            .and_then(|s| s.get("mode"))
            .and_then(|v| v.as_str())
            .ok_or("dspine block missing second_pass.mode")?;
        if mode != "serial_table" {
            return Err(format!("second_pass.mode {mode:?} != \"serial_table\""));
        }
        let first_exact = dsp
            .get("second_pass")
            .and_then(|s| s.get("first_exact"))
            .and_then(|v| v.as_u64())
            .ok_or("dspine block missing second_pass.first_exact")? as usize;
        if first_exact != trunk.n_layers.saturating_sub(1) {
            return Err(format!(
                "second_pass.first_exact {first_exact} != num_hidden_layers {} - 1",
                trunk.n_layers
            ));
        }
        let head_topk = dsp
            .get("second_pass")
            .and_then(|s| s.get("head_topk"))
            .and_then(|v| v.as_u64())
            .ok_or("dspine block missing second_pass.head_topk")? as usize;
        if !(2..=64).contains(&head_topk) {
            return Err(format!("head_topk {head_topk} outside [2, 64]"));
        }
        if !head_topk.is_power_of_two() {
            return Err(format!("head_topk {head_topk} is not a power of two"));
        }
        let feature_layers = dsp
            .get("feature_layers")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_u64().map(|x| x as usize))
                    .collect::<Vec<usize>>()
            })
            .ok_or("dspine block missing feature_layers")?;
        if feature_layers.is_empty() {
            return Err("feature_layers is empty".into());
        }
        for &l in &feature_layers {
            if l == 0 || l > trunk.n_layers {
                return Err(format!(
                    "feature_layers entry {l} outside 1..={}",
                    trunk.n_layers
                ));
            }
        }

        // The four version markers the converter validated; the loader
        // re-checks them against the same contract (plan doc §6).
        let versions = dsp
            .get("versions")
            .and_then(|v| v.as_object())
            .ok_or("dspine block missing versions")?;
        const VERSION_KEYS: [&str; 4] = [
            "draft_block_causal_version",
            "last_write_refine_version",
            "selected_prefix_bridge.independent_write_version",
            "selected_prefix_bridge.terminal_disabled_version",
        ];
        for key in VERSION_KEYS {
            match versions.get(key).and_then(|v| v.as_i64()) {
                Some(1) => {}
                Some(other) => return Err(format!("version scalar {key} = {other}, expected 1")),
                None => return Err(format!("versions missing {key}")),
            }
        }

        // Prebuilt serial sender tables (TG / TSV) from the converter's
        // `--target-hfq` pass. Their absence is legal: the loader then
        // stays on the plain greedy-argmax refinement (fail-closed).
        let serial_tables = dsp.get("serial_tables").is_some();

        Ok(Some(Self {
            trunk,
            code_size,
            bridge_rank,
            second_pass_mode: mode.to_string(),
            first_exact,
            head_topk,
            feature_layers,
            serial_tables,
        }))
    }
}

/// GPU-resident sender tables for the serial last-write refinement
/// (upstream `DSpineDraftSampler.build_serial_tables`, prebuilt by the
/// converter): `tg[cand]`, `tsv_gate[sid]`, `tsv_val[sid]`, each [V, R] F32.
pub struct SerialTables {
    pub tg: GpuTensor,
    pub tsv_gate: GpuTensor,
    pub tsv_val: GpuTensor,
}

/// GPU-resident code-bridge runner for the arch-24 DSpine drafter: loads the
/// `selected_prefix_bridge` section out of the converted sidecar, owns the
/// bridge scratch, and implements [`crate::dflash::DraftLayerBridge`] so the
/// shared trunk loop splices a bridge write after every draft layer.
///
/// Storage policy (semantics-map §3/§6): the codebook stays raw BF16 (a
/// gather table, never a GEMM operand); every GEMM weight is lifted
/// BF16→F32 at load (exact) and rides `gemm_f32_batched` in its natural
/// [out, in] row-major layout — no host transposes. Activations are F32 end
/// to end in the bridge; the kernels mirror `cpu_oracle` operation order.
pub struct DspineBridge {
    pub cfg: DspineConfig,
    h: usize,
    c: usize,
    r: usize,
    t: usize,
    codebook: GpuTensor,    // [V*C] raw bf16
    read_w: GpuTensor,      // [C, H] f32 (last layer projection)
    early_w: Vec<GpuTensor>, // 4 × [C, H] f32 (layers 0..3)
    write_w: Vec<GpuTensor>, // 5 × [H, R] f32
    packed_w: GpuTensor,    // [3R, C] f32 = gate ⊕ predecessor_gate ⊕ prev
    // Standalone predecessor/value projections for the refinement lattice
    // (the packed concat is consumed by the trunk bridge; the lattice reads
    // them row-wise per predecessor candidate).
    pred_gate: GpuTensor,   // [R, C] f32
    prev_w: GpuTensor,      // [R, C] f32
    gate_bias: GpuTensor,   // [R] f32
    // Prebuilt serial sender tables ([V, R] f32 each), `Some` only
    // when the sidecar carried them (converter `--target-hfq`).
    serial: Option<SerialTables>,
    // scratch (sized to the block)
    s_normed: GpuTensor,     // [b, H] f32
    s_projected: GpuTensor,  // [b, C] f32
    s_codes: GpuTensor,      // [b, C] f32
    s_packed: GpuTensor,     // [b, 3R] f32
    s_message: GpuTensor,    // [b, R] f32
    s_delta: GpuTensor,      // [b, H] f32
    s_rms: GpuTensor,        // [b] f32
    // Pre-write hidden snapshot ([block, H] f32): the `base` term of the
    // serial refinement reads the hidden the way the upstream sampler
    // does, BEFORE this layer's bridge write lands (finish is in-place).
    s_x_pre: GpuTensor,
    s_anchor: GpuTensor,     // [1] u32 — the block's first real token id
    /// Host-side copy of the anchor id (the table chain seeds its walk with it).
    anchor_id: u32,
    // Reused across draft windows: ~10 MB of device scratch and the FWHT sign
    // tables are constant per row count, so the hot path allocates them once.
    lattice_scratch: std::cell::RefCell<Option<LatticeScratch>>,
}

/// Big per-window lattice buffers plus the FWHT sign tables.
struct LatticeScratch {
    rows: usize,
    lm: GpuTensor,
    refined: GpuTensor,
    scores: GpuTensor,
    dbg: GpuTensor,
    /// Message per (row, predecessor): the lattice GEMM's right operand.
    msg: GpuTensor,
    s1t: GpuTensor,
    s2t: GpuTensor,
}

fn lift_bf16_tensor(
    gpu: &mut Gpu,
    hfq: &HfqFile,
    name: &str,
    shape: Vec<usize>,
) -> Result<GpuTensor, String> {
    let (meta, raw) = hfq.tensor_data(name).ok_or_else(|| format!("dspine tensor missing: {name}"))?;
    if meta.quant_type != 16 {
        return Err(format!("dspine tensor {name} quant_type {} != 16 (BF16)", meta.quant_type));
    }
    let n: usize = shape.iter().product();
    assert_eq!(raw.len(), n * 2, "dspine tensor {name} byte size");
    let f32_data: Vec<f32> = raw
        .chunks_exact(2)
        .map(|c| crate::llama::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    gpu.upload_f32(&f32_data, &shape)
        .map_err(|e| format!("dspine upload {name}: {e}"))
}

impl DspineBridge {
    /// Load the bridge section out of a converted arch-24 sidecar and
    /// allocate the bridge scratch. `block` is the trunk block size (16).
    pub fn new(gpu: &mut Gpu, hfq: &HfqFile, cfg: DspineConfig) -> Result<Self, String> {
        let h = cfg.trunk.hidden;
        let c = cfg.code_size;
        let r = cfg.bridge_rank;
        let t = cfg.trunk.block_size;
        let block = t;

        let codebook = {
            let (meta, raw) = hfq
                .tensor_data("selected_prefix_bridge.codebook")
                .ok_or("dspine tensor missing: selected_prefix_bridge.codebook")?;
            assert_eq!(meta.quant_type, 16, "codebook must be BF16");
            let v = meta.shape[0] as usize;
            assert_eq!(meta.shape[1] as usize, c, "codebook width must equal code_size");
            gpu.upload_raw(raw, &[v * c])
                .map_err(|e| format!("dspine upload codebook: {e}"))?
        };

        let read_w = lift_bf16_tensor(
            gpu,
            hfq,
            "selected_prefix_bridge.read_weight",
            vec![c, h],
        )?;
        let mut early_w = Vec::with_capacity(4);
        {
            // One [4, C, H] tensor; slice it on the host into four [C, H] lifts.
            let (meta, raw) = hfq
                .tensor_data("selected_prefix_bridge.early_read_weights")
                .ok_or("dspine tensor missing: selected_prefix_bridge.early_read_weights")?;
            assert_eq!(meta.quant_type, 16);
            assert_eq!(meta.shape, vec![4u32, c as u32, h as u32]);
            for i in 0..4usize {
                let slice = &raw[i * c * h * 2..(i + 1) * c * h * 2];
                let f32_data: Vec<f32> = slice
                    .chunks_exact(2)
                    .map(|ch| crate::llama::bf16_to_f32(u16::from_le_bytes([ch[0], ch[1]])))
                    .collect();
                early_w.push(
                    gpu.upload_f32(&f32_data, &[c, h])
                        .map_err(|e| format!("dspine upload early[{i}]: {e}"))?,
                );
            }
        }
        let mut write_w = Vec::with_capacity(5);
        {
            let (meta, raw) = hfq
                .tensor_data("selected_prefix_bridge.feature_write_weights")
                .ok_or("dspine tensor missing: selected_prefix_bridge.feature_write_weights")?;
            assert_eq!(meta.quant_type, 16);
            assert_eq!(meta.shape, vec![5u32, h as u32, r as u32]);
            for i in 0..5usize {
                let slice = &raw[i * h * r * 2..(i + 1) * h * r * 2];
                let f32_data: Vec<f32> = slice
                    .chunks_exact(2)
                    .map(|ch| crate::llama::bf16_to_f32(u16::from_le_bytes([ch[0], ch[1]])))
                    .collect();
                write_w.push(
                    gpu.upload_f32(&f32_data, &[h, r])
                        .map_err(|e| format!("dspine upload write[{i}]: {e}"))?,
                );
            }
        }
        // Host-side vertical concat [3R, C]: packed = codes @ packed_w^T.
        let packed_w = {
            let names = [
                "selected_prefix_bridge.gate.weight",
                "selected_prefix_bridge.predecessor_gate.weight",
                "selected_prefix_bridge.prev.weight",
            ];
            let mut f32_data = Vec::with_capacity(3 * r * c);
            for name in names {
                let (meta, raw) = hfq
                    .tensor_data(name)
                    .ok_or_else(|| format!("dspine tensor missing: {name}"))?;
                assert_eq!(meta.quant_type, 16);
                assert_eq!(meta.shape, vec![r as u32, c as u32]);
                for ch in raw.chunks_exact(2) {
                    f32_data.push(crate::llama::bf16_to_f32(u16::from_le_bytes([ch[0], ch[1]])));
                }
            }
            gpu.upload_f32(&f32_data, &[3 * r, c])
                .map_err(|e| format!("dspine upload packed_w: {e}"))?
        };
        let gate_bias = lift_bf16_tensor(
            gpu,
            hfq,
            "selected_prefix_bridge.gate.bias",
            vec![r],
        )?;        let pred_gate = lift_bf16_tensor(
            gpu,
            hfq,
            "selected_prefix_bridge.predecessor_gate.weight",
            vec![r, c],
        )?;
        let prev_w = lift_bf16_tensor(
            gpu,
            hfq,
            "selected_prefix_bridge.prev.weight",
            vec![r, c],
        )?;


        // Prebuilt serial sender tables: fail-closed lift when the
        // metadata promises them (a missing tensor is a corrupt sidecar,
        // not a silent downgrade to the argmax walk).
        let serial = if cfg.serial_tables {
            let v = {
                let (meta, _) = hfq
                    .tensor_data("selected_prefix_bridge.codebook")
                    .ok_or("dspine tensor missing: selected_prefix_bridge.codebook")?;
                meta.shape[0] as usize
            };
            // The converter stores the tables fp32 (exact, matching the
            // upstream GPU residency); older BF16 sidecars still lift.
            let mut lift_table = |name: &str| -> Result<GpuTensor, String> {
                let (meta, raw) = hfq
                    .tensor_data(name)
                    .ok_or_else(|| format!("dspine tensor missing: {name}"))?;
                match meta.quant_type {
                    2 => {
                        assert_eq!(raw.len(), v * r * 4, "dspine {name} f32 size");
                        let mut data = vec![0f32; v * r];
                        for (i, chunk) in raw.chunks_exact(4).enumerate() {
                            data[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                        }
                        gpu.upload_f32(&data, &[v, r])
                            .map_err(|e| format!("dspine upload {name}: {e}"))
                    }
                    16 => lift_bf16_tensor(gpu, hfq, name, vec![v, r]),
                    q => Err(format!("dspine {name}: quant_type {q} is neither F32 nor BF16")),
                }
            };
            Some(SerialTables {
                tg: lift_table("dspine.tg")?,
                tsv_gate: lift_table("dspine.tsv_gate")?,
                tsv_val: lift_table("dspine.tsv_val")?,
            })
        } else {
            None
        };

        Ok(Self {
            s_normed: gpu
                .alloc_tensor(&[block, h], rdna_compute::DType::F32)
                .map_err(|e| format!("dspine scratch normed: {e}"))?,
            s_projected: gpu
                .alloc_tensor(&[block, c], rdna_compute::DType::F32)
                .map_err(|e| format!("dspine scratch projected: {e}"))?,
            s_codes: gpu
                .alloc_tensor(&[block, c], rdna_compute::DType::F32)
                .map_err(|e| format!("dspine scratch codes: {e}"))?,
            s_packed: gpu
                .alloc_tensor(&[block, 3 * r], rdna_compute::DType::F32)
                .map_err(|e| format!("dspine scratch packed: {e}"))?,
            s_message: gpu
                .alloc_tensor(&[block, r], rdna_compute::DType::F32)
                .map_err(|e| format!("dspine scratch message: {e}"))?,
            s_delta: gpu
                .alloc_tensor(&[block, h], rdna_compute::DType::F32)
                .map_err(|e| format!("dspine scratch delta: {e}"))?,
            s_rms: gpu
                .alloc_tensor(&[block], rdna_compute::DType::F32)
                .map_err(|e| format!("dspine scratch rms: {e}"))?,
            s_x_pre: gpu
                .alloc_tensor(&[block, h], rdna_compute::DType::F32)
                .map_err(|e| format!("dspine scratch x_pre: {e}"))?,
            s_anchor: gpu
                .alloc_tensor(&[1], rdna_compute::DType::Raw)
                .map_err(|e| format!("dspine scratch anchor: {e}"))?,
            cfg,
            h,
            c,
            r,
            t,
            codebook,
            read_w,
            early_w,
            write_w,
            packed_w,
            pred_gate,
            prev_w,
            gate_bias,
            serial,
            anchor_id: 0,
            lattice_scratch: std::cell::RefCell::new(None),
        })
    }

    /// The pre-write hidden snapshot ([block, H] f32).
    pub fn x_pre(&self) -> &GpuTensor {
        &self.s_x_pre
    }

    /// The per-row RMS of the pre-write snapshot (prepare_read's stats).
    pub fn rms_pre(&self) -> &GpuTensor {
        &self.s_rms
    }

    /// Refinement-term scale for weight sweeps (upstream weight=1.0).
    pub fn rms_scale(&self) -> f32 {
        let v = hipfire_config::developer_var("HIPFIRE_DSPINE_SERIAL_WEIGHT")
            .ok()
            .and_then(|s| s.parse::<f32>().ok());
        v.unwrap_or(1.0)
    }

    /// Serial last-write refinement over the block's head logits:
    /// top-16 candidates per row (`topk_values_batched_f32`, regrid), then the fused
    /// sender-table score chain (`dspine_serial_chain`) walked from
    /// the anchor. `logits` is the [rows, vocab] device block produced
    /// by the batched lm_head (rows = block_size - 1; the seed row is
    /// not refined). The per-row queries live in `s_packed` (stride
    /// 3R) and the rms in `s_rms`, both left by the bridged forward.
    /// Returns the picked id per row in chain order; the caller
    /// prepends the seed token.
    pub fn serial_refine(
        &self,
        gpu: &mut Gpu,
        logits_post: &GpuTensor,
        rows: usize,
        lm_head_raw: &crate::llama::WeightTensor,
        out_norm: &GpuTensor,
    ) -> hip_bridge::HipResult<Vec<u32>> {
        let tables = self.serial.as_ref();
        let vocab = self.cfg.trunk.vocab_size;
        // u32 payloads ride F32 tensors (4 B/elem): Raw is 1 B/elem and
        // download_f32 reads numel*4, so a Raw alloc here is a bounded
        // overread that only survives on pool granularity.
        let cand = gpu.alloc_tensor(&[rows, 16], rdna_compute::DType::F32)?;
        let base = gpu.alloc_tensor(&[rows, 16], rdna_compute::DType::F32)?;
        let out = gpu.alloc_tensor(&[rows], rdna_compute::DType::F32)?;
        let diag =
            hipfire_config::developer_var("HIPFIRE_DSPINE_SERIAL_DIAG").as_deref() == Ok("1");
        if diag {
            eprintln!("[dspine-serial] rows={} base.buf={}B cand.buf={}B out.buf={}B",
                rows, base.byte_size(), cand.byte_size(), out.byte_size());
        }
        // Experiment switch: HIPFIRE_DSPINE_SERIAL_MODE=argmax returns the
        // pre-write argmax (topk16 lane 0) and skips the sender chain —
        // isolates how much of the acceptance gap is the refinement term
        // vs the base ordering itself.
        let serial_mode = hipfire_config::developer_var("HIPFIRE_DSPINE_SERIAL_MODE")
            .unwrap_or_default();
        let mode_argmax = serial_mode == "argmax";
        // Greedy refinement defaults to upstream's served `serial_table`
        // scoring (shipped TG/TSV tables + the pre-write base term): the exact
        // lattice's rmsnorm scale depends only on (row, predecessor), so it
        // cancels in the per-row argmax and both paths pick the same tokens
        // (verified pick-for-pick on every captured window) — the chain just
        // skips the K*K full-H recomputation. `exact` re-runs the audited
        // lattice, `argmax` skips refinement entirely.
        let mode_table = serial_mode != "exact" && !mode_argmax;
        let mode_table = mode_table && tables.is_some();
        if serial_mode == "table" && tables.is_none() {
            return Err(hip_bridge::HipError::new(
                0,
                "serial mode `table` needs the sidecar's TG/TSV tables",
            ));
        }
        // Capture root is a run-scoped directory chosen by the caller
        // (default keeps the legacy path for ad-hoc runs).
        let cap_root = hipfire_config::developer_var("HIPFIRE_DSPINE_CAPTURE_DIR")
            .ok()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "/tmp/dspine_capture_l".to_string());
        let timing =
            hipfire_config::developer_var("HIPFIRE_DSPINE_LATTICE_TIMING").as_deref() == Ok("1");
        let pick = (|| -> hip_bridge::HipResult<Vec<u32>> {
            let mut cap_dir: Option<String> = None;
            let t0 = std::time::Instant::now();
            // Upstream lattice (build_dspine_lattice): candidates from the
            // post-write logits, then EXACT per-(candidate, predecessor)
            // bridge-write recomputation — no sender-table approximation.
            gpu.topk_values_batched_f32(logits_post, &cand, &base, vocab, 16, rows)?;
            let t_topk = t0.elapsed();
            if mode_argmax {
                // stream-ordered: the null-stream readback drains the whole
                // queue and would dominate this control arm's window time.
                let raw = gpu.download_f32_stream_ordered(&cand)?;
                return Ok((0..rows).map(|i| raw[i * 16].to_bits()).collect());
            }
            // stream-ordered: the plain readback blocks on the legacy null
            // stream, syncing the whole decode queue (8x measured earlier).
            // Lattice inputs: raw/rms/query all take rows 1..n (the seed row
            // has no refinement); the anchor code is the bridge's row-0 code
            // (already sqrt(C)-scaled by normalize_codes).
            let raw_rows = self.s_x_pre.sub_offset(self.h, rows * self.h);
            let rms_rows = self.s_rms.sub_offset(1, rows);
            let query_rows = self
                .s_packed
                .sub_offset(3 * self.r, rows * 3 * self.r);
            let anchor_code = self.s_codes.sub_offset(0, self.c);
            let mut held = self.lattice_scratch.replace(None);
            if held.as_ref().map_or(true, |s| s.rows != rows) {
                if let Some(old) = held.take() {
                    let _ = gpu.free_tensor(old.lm);
                    let _ = gpu.free_tensor(old.refined);
                    let _ = gpu.free_tensor(old.scores);
                    let _ = gpu.free_tensor(old.dbg);
                    let _ = gpu.free_tensor(old.s1t);
                    let _ = gpu.free_tensor(old.s2t);
                }
                // MQ4V2 head rows encode FWHT(W): the score dot product
                // consumes the FWHT-rotated refined vector (engine sign
                // convention, seeds 42/1042 — same tables as ensure_mq_signs).
                let mut signs1 = vec![0f32; 256];
                let mut signs2 = vec![0f32; 256];
                {
                    let mut st: u32 = 42;
                    for i in 0..256 {
                        st = st.wrapping_mul(1103515245).wrapping_add(12345) & 0x7fffffff;
                        signs1[i] = if (st >> 16) & 1 == 1 { 1.0 } else { -1.0 };
                    }
                    st = 1042;
                    for i in 0..256 {
                        st = st.wrapping_mul(1103515245).wrapping_add(12345) & 0x7fffffff;
                        signs2[i] = if (st >> 16) & 1 == 1 { 1.0 } else { -1.0 };
                    }
                }
                held = Some(LatticeScratch {
                    rows,
                    lm: gpu.alloc_tensor(&[rows, 16, self.h], rdna_compute::DType::F32)?,
                    refined: gpu.alloc_tensor(&[rows, 16, self.h], rdna_compute::DType::F32)?,
                    scores: gpu.alloc_tensor(&[rows, 16, 16], rdna_compute::DType::F32)?,
                    dbg: gpu.alloc_tensor(&[12800], rdna_compute::DType::F32)?,
                    msg: gpu.alloc_tensor(&[rows * 16, self.r], rdna_compute::DType::F32)?,
                    s1t: gpu.upload_f32(&signs1, &[256])?,
                    s2t: gpu.upload_f32(&signs2, &[256])?,
                });
            }
            let LatticeScratch {
                rows: _sr, lm: lm_scratch, refined: refined_out, scores, dbg: dbg_seg, s1t, s2t,
                msg,
            } = held.take().unwrap();
            let row_bytes = ((self.h + 255) / 256) * 136;
            let last = self.write_w.len() - 1;
            if mode_table {
                let tables = tables.expect("table mode requires TG/TSV tables");
                // Sender-table chain: candidates from the post-write top-16,
                // base from the pre-write stash, one dot per (row, candidate)
                // against TG — no per-predecessor full-H recomputation.
                gpu.dspine_lattice_head_rows(
                    &cand,
                    &lm_head_raw.buf,
                    &lm_scratch,
                    rows,
                    self.h,
                    self.r,
                    row_bytes,
                )?;
                gpu.dspine_lattice_base_pre(
                    &raw_rows,
                    out_norm,
                    &lm_scratch,
                    &s1t,
                    &s2t,
                    &base,
                    rows,
                    self.h,
                )?;
                gpu.dspine_serial_chain(
                    &base,
                    &cand,
                    &rms_rows,
                    &query_rows,
                    &tables.tsv_gate,
                    &tables.tsv_val,
                    &tables.tg,
                    &self.gate_bias,
                    self.anchor_id,
                    &out,
                    rows,
                    16,
                    self.r,
                    self.t,
                    3 * self.r,
                    self.rms_scale(),
                )?;
                let ids = gpu.download_f32_stream_ordered(&out)?;
                let picks_v: Vec<u32> = (0..rows).map(|i| ids[i].to_bits()).collect();
                if hipfire_config::developer_var("HIPFIRE_DSPINE_SERIAL_CAPTURE")
                    .as_deref()
                    == Ok("1")
                {
                    static CAP_SEQ: std::sync::atomic::AtomicU64 =
                        std::sync::atomic::AtomicU64::new(0);
                    let seq = CAP_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let dir = format!("{}/win_{:05}", cap_root, seq);
                    let _ = std::fs::create_dir_all(&dir);
                    if let Ok(mut f) = std::fs::File::create(format!("{}/picks.bin", dir)) {
                        use std::io::Write as _;
                        let mut bytes = Vec::with_capacity(picks_v.len() * 4);
                        for id in &picks_v {
                            bytes.extend_from_slice(&id.to_le_bytes());
                        }
                        let _ = f.write_all(&bytes);
                    }
                }
                self.lattice_scratch.replace(Some(LatticeScratch {
                    rows, lm: lm_scratch, refined: refined_out, scores, dbg: dbg_seg,
                    s1t, s2t, msg,
                }));
                if timing {
                    let t_walk = t0.elapsed();
                    eprintln!("[dspine-table-timing] rows={} total={:.3}ms",
                        rows, t_walk.as_secs_f64() * 1e3);
                }
                return Ok(picks_v);
            }
            gpu.dspine_lattice_head_rows(
                &cand,
                &lm_head_raw.buf,
                &lm_scratch,
                rows,
                self.h,
                self.r,
                row_bytes,
            )?;
            gpu.dspine_lattice(
                &raw_rows,
                &rms_rows,
                &query_rows,
                &self.codebook,
                &self.pred_gate,
                &self.prev_w,
                &self.gate_bias,
                &self.write_w[last],
                out_norm,
                &s1t,
                &s2t,
                &cand,
                &anchor_code,
                &lm_scratch,
                &refined_out,
                &scores,
                &dbg_seg,
                &msg,
                rows,
                self.h,
                self.r,
                self.c,
            )?;
            let t_launch = t0.elapsed();
            if timing {
                // Null-stream readback forces the whole queue to drain, so
                // this brackets the lattice kernel's true device time.
                let _ = gpu.download_f32(&scores)?;
            }
            let t_sync = t0.elapsed();
            // Host walk: row 0's predecessor is the anchor (single code, all
            // lanes equivalent); deeper rows condition on the previous row's
            // chosen lane.
            if diag {
                // Debug-only readback: ~4.9 MB per window, never needed for
                // the pick itself (the capture path dumps refined separately).
                let refined_h = gpu.download_f32_stream_ordered(&refined_out)?;
                if let Some(_w0) = refined_h.first() {
                    let mx = refined_h.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
                    eprintln!("[dspine-lattice] refined0 maxabs={:.4} first4={:?}", mx, &refined_h[..4]);
                }
            }
            let scores_h = gpu.download_f32_stream_ordered(&scores)?;
            let cand_h = gpu.download_f32_stream_ordered(&cand)?;
            let t_dl = t0.elapsed();
            if hipfire_config::developer_var("HIPFIRE_DSPINE_SERIAL_CAPTURE")
                .as_deref()
                == Ok("1")
            {
                static CAP_SEQ: std::sync::atomic::AtomicU64 =
                    std::sync::atomic::AtomicU64::new(0);
                let seq = CAP_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let dir = format!("{}/win_{:05}", cap_root, seq);
                let _ = std::fs::create_dir_all(&dir);
                cap_dir = Some(dir.clone());
                for (name, t) in [
                    ("raw", &raw_rows),
                    ("rms", &rms_rows),
                    ("query", &query_rows),
                    ("cand", &cand),
                    ("refined", &refined_out),
                    ("anchor", &self.s_anchor),
                    ("codes", &self.s_codes),
                    ("scores", &scores),
                    ("dbg", &dbg_seg),
                ] {
                    let v = gpu.download_f32(t);
                    if let Ok(v) = v {
                        if let Ok(mut f) = std::fs::File::create(format!("{}/{}.bin", dir, name)) {
                            use std::io::Write as _;
                            let _ = f.write_all(unsafe {
                                std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4)
                            });
                        }
                    }
                }
                if let Ok(mut f) = std::fs::File::create(format!("{}/meta.json", dir)) {
                    use std::io::Write as _;
                    let _ = f.write_all(
                        serde_json::json!({
                            "rows": rows, "h": self.h, "r": self.r, "c": self.c,
                            "vocab": self.cfg.trunk.vocab_size,
                            "anchor_lane": 0,
                        })
                        .to_string()
                        .as_bytes(),
                    );
                }
            }
            if diag {
                if let Some(_w0) = scores_h.first() {
                    let mx = scores_h.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
                    eprintln!(
                        "[dspine-lattice] rows={} scores maxabs={:.3} first8={:?}",
                        rows,
                        mx,
                        &scores_h[..8.min(scores_h.len())]
                    );
                }
            }
            let mut picks_v: Vec<u32> = Vec::with_capacity(rows);
            let mut lane_prev = 0usize;
            for r in 0..rows {
                let mut best_c = 0usize;
                let mut best_s = f32::NEG_INFINITY;
                for c in 0..16 {
                    let sc = scores_h[(r * 16 + c) * 16 + lane_prev];
                    if sc > best_s {
                        best_s = sc;
                        best_c = c;
                    }
                }
                picks_v.push(cand_h[r * 16 + best_c].to_bits());
                lane_prev = best_c;
            }
            self.lattice_scratch.replace(Some(LatticeScratch {
                rows, lm: lm_scratch, refined: refined_out, scores, dbg: dbg_seg, s1t, s2t,
                msg,
            }));
            if let Some(dir) = &cap_dir {
                if let Ok(mut f) = std::fs::File::create(format!("{}/picks.bin", dir)) {
                    use std::io::Write as _;
                    let mut bytes = Vec::with_capacity(picks_v.len() * 4);
                    for id in &picks_v {
                        bytes.extend_from_slice(&id.to_le_bytes());
                    }
                    let _ = f.write_all(&bytes);
                }
            }
            if timing {
                let t_walk = t0.elapsed();
                eprintln!(
                    "[dspine-lattice-timing] rows={} topk={:.3}ms launch={:.3}ms sync={:.3}ms dl={:.3}ms walk={:.3}ms",
                    rows,
                    t_topk.as_secs_f64() * 1e3,
                    (t_launch - t_topk).as_secs_f64() * 1e3,
                    (t_sync - t_launch).as_secs_f64() * 1e3,
                    (t_dl - t_sync).as_secs_f64() * 1e3,
                    (t_walk - t_dl).as_secs_f64() * 1e3,
                );
            }
            return Ok(picks_v);
        })();
        let _ = gpu.free_tensor(cand);
        let _ = gpu.free_tensor(base);
        return pick;
    }

    /// The prebuilt sender tables, when the sidecar carried them.
    pub fn serial(&self) -> Option<&SerialTables> {
        self.serial.as_ref()
    }

    /// Explicit GPU release (mirrors `DflashWeights::free_gpu`: tensor drops
    /// do not return pool pages on their own).
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let DspineBridge {
            codebook,
            read_w,
            early_w,
            write_w,
            packed_w,
            pred_gate,
            prev_w,
            gate_bias,
            serial,
            s_normed,
            s_projected,
            s_codes,
            s_packed,
            s_message,
            s_delta,
            s_rms,
            s_x_pre,
            s_anchor,
            lattice_scratch,
            ..
        } = self;
        let _ = gpu.free_tensor(codebook);
        let _ = gpu.free_tensor(read_w);
        for t in early_w {
            let _ = gpu.free_tensor(t);
        }
        for t in write_w {
            let _ = gpu.free_tensor(t);
        }
        let _ = gpu.free_tensor(packed_w);
        let _ = gpu.free_tensor(pred_gate);
        let _ = gpu.free_tensor(prev_w);
        let _ = gpu.free_tensor(gate_bias);
        if let Some(tables) = serial {
            let _ = gpu.free_tensor(tables.tg);
            let _ = gpu.free_tensor(tables.tsv_gate);
            let _ = gpu.free_tensor(tables.tsv_val);
        }
        if let Some(sc) = lattice_scratch.into_inner() {
            let _ = gpu.free_tensor(sc.lm);
            let _ = gpu.free_tensor(sc.refined);
            let _ = gpu.free_tensor(sc.scores);
            let _ = gpu.free_tensor(sc.dbg);
            let _ = gpu.free_tensor(sc.msg);
            let _ = gpu.free_tensor(sc.s1t);
            let _ = gpu.free_tensor(sc.s2t);
        }
        for t in [s_normed, s_projected, s_codes, s_packed, s_message, s_delta, s_rms, s_anchor, s_x_pre] {
            let _ = gpu.free_tensor(t);
        }
    }

    /// Feed the block's first real token id (the code anchor) before each
    /// bridged forward. The speculator driver owns the token.
    pub fn set_anchor(&mut self, gpu: &mut Gpu, anchor: u32) -> Result<(), String> {
        self.anchor_id = anchor;
        let bytes = anchor.to_le_bytes();
        gpu.upload_raw(&bytes, &[1])
            .map(|t| {
                self.s_anchor = t;
            })
            .map_err(|e| format!("dspine anchor upload: {e}"))
    }
}

impl crate::dflash::DraftLayerBridge for DspineBridge {
    fn after_layer(
        &mut self,
        gpu: &mut Gpu,
        scratch: &mut crate::dflash::DflashScratch,
        li: usize,
        b: usize,
        _ctx_len: usize,
    ) -> hip_bridge::HipResult<()> {
        // Scratch is sized to one full block; a shorter tail block (the
        // last window when the emit budget runs out) is fine — every
        // kernel is row-count driven and the anchor keeps its meaning.
        assert!(b <= self.t, "bridge rows exceed the block-sized scratch");
        let h = self.h;
        let c = self.c;
        let r = self.r;
        let t = self.t;
        let eps = self.cfg.trunk.norm_eps;

        // Temporary bridge attribution (T4): HIPFIRE_DSPINE_BRIDGE_DIAG=1
        // drains the queue at every boundary and prints one line per layer.
        static BRIDGE_DIAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let diag = *BRIDGE_DIAG
            .get_or_init(|| std::env::var_os("HIPFIRE_DSPINE_BRIDGE_DIAG").map_or(false, |v| v != "0"));
        let mut us = [0u128; 8];

        // 2. projection GEMM: normed [b,H] × W_i[C,H]^T → projected [b,C].
        //    Layers 0..3 use early_read_weights[i]; the last layer read_weight.
        let w = if li < self.early_w.len() {
            &self.early_w[li]
        } else {
            &self.read_w
        };
        static FUSED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let fused = *FUSED.get_or_init(|| {
            std::env::var_os("HIPFIRE_DSPINE_FUSED_BRIDGE").map_or(true, |v| v != "0")
        });
        if fused {
            // Fused path: the pre-write snapshot rides prepare_read (us[1]),
            // the read-side GEMMs use the 8-outputs-per-block bridge kernel,
            // and finish is folded into the delta GEMM (us[6]).
            if diag {
                gpu.hip.device_synchronize()?;
            }
            let tq = std::time::Instant::now();
            gpu.dspine_prepare_read_snap(
                &scratch.x,
                &self.s_rms,
                &self.s_normed,
                &self.s_x_pre,
                b,
                h,
                eps,
            )?;
            if diag {
                gpu.hip.device_synchronize()?;
                us[1] = tq.elapsed().as_micros();
            }
            if diag {
                gpu.hip.device_synchronize()?;
            }
            let tq = std::time::Instant::now();
            gpu.dspine_gemm_f32_batch(w, &self.s_normed, &self.s_projected, c, h, b)?;
            if diag {
                gpu.hip.device_synchronize()?;
                us[2] = tq.elapsed().as_micros();
            }
            if diag {
                gpu.hip.device_synchronize()?;
            }
            let tq = std::time::Instant::now();
            gpu.dspine_normalize_codes(
                &self.s_projected,
                &self.codebook,
                &self.s_anchor,
                &self.s_codes,
                b,
                c,
                t,
            )?;
            if diag {
                gpu.hip.device_synchronize()?;
                us[3] = tq.elapsed().as_micros();
            }
            if diag {
                gpu.hip.device_synchronize()?;
            }
            let tq = std::time::Instant::now();
            gpu.dspine_gemm_f32_batch(&self.packed_w, &self.s_codes, &self.s_packed, 3 * r, c, b)?;
            if diag {
                gpu.hip.device_synchronize()?;
                us[4] = tq.elapsed().as_micros();
            }
            if diag {
                gpu.hip.device_synchronize()?;
            }
            let tq = std::time::Instant::now();
            gpu.dspine_feature_message(&self.s_packed, &self.gate_bias, &self.s_message, b, r, t)?;
            if diag {
                gpu.hip.device_synchronize()?;
                us[5] = tq.elapsed().as_micros();
            }
            if diag {
                gpu.hip.device_synchronize()?;
            }
            let tq = std::time::Instant::now();
            gpu.dspine_delta_finish(
                &self.write_w[li],
                &self.s_message,
                &scratch.x,
                &self.s_rms,
                h,
                r,
                b,
                t,
            )?;
            if diag {
                gpu.hip.device_synchronize()?;
                us[6] = tq.elapsed().as_micros();
            }
        } else {
        // Snapshot the pre-write hidden rows first: finish rewrites x
        // in place, and the serial refinement's base term reads the
        // pre-write values exactly like the upstream stash.
        if diag {
            gpu.hip.device_synchronize()?;
        }
        let tq = std::time::Instant::now();
        gpu
            .memcpy_dtod_at_ordered_async(&self.s_x_pre.buf, 0, &scratch.x.buf, 0, b * h * 4)
            .map_err(|e| hip_bridge::HipError::new(e.code, &format!("dspine x_pre snapshot: {e}")))?;
        if diag {
            gpu.hip.device_synchronize()?;
            us[0] = tq.elapsed().as_micros();
        }

        // 1. rms stats + normalized rows (kernel 1).
        if diag {
            gpu.hip.device_synchronize()?;
        }
        let tq = std::time::Instant::now();
        gpu.dspine_prepare_read(&scratch.x, &self.s_rms, &self.s_normed, b, h, eps)?;
        if diag {
            gpu.hip.device_synchronize()?;
            us[1] = tq.elapsed().as_micros();
        }
        // (weight selection moved above the fused/legacy branch)
        // gemm computes y = b_arg·a_arg^T ([n,m] row-major): pass the weight
        // first so y lands as projected [b, C] row-major.
        if diag {
            gpu.hip.device_synchronize()?;
        }
        let tq = std::time::Instant::now();
        gpu.gemm_f32_batched(w, &self.s_normed, &self.s_projected, c, h, b)?;
        if diag {
            gpu.hip.device_synchronize()?;
            us[2] = tq.elapsed().as_micros();
        }
        // 3. normalize_codes (kernel 2): anchor identity / L2 to sqrt(C).
        if diag {
            gpu.hip.device_synchronize()?;
        }
        let tq = std::time::Instant::now();
        gpu.dspine_normalize_codes(&self.s_projected, &self.codebook, &self.s_anchor, &self.s_codes, b, c, t)?;
        if diag {
            gpu.hip.device_synchronize()?;
            us[3] = tq.elapsed().as_micros();
        }
        // 4. packed GEMM: codes [b,C] × packed_w[3R,C]^T → packed [b,3R].
        if diag {
            gpu.hip.device_synchronize()?;
        }
        let tq = std::time::Instant::now();
        gpu.gemm_f32_batched(&self.packed_w, &self.s_codes, &self.s_packed, 3 * r, c, b)?;
        if diag {
            gpu.hip.device_synchronize()?;
            us[4] = tq.elapsed().as_micros();
        }
        // 5. feature_message (kernel 3): gate · predecessor value.
        if diag {
            gpu.hip.device_synchronize()?;
        }
        let tq = std::time::Instant::now();
        gpu.dspine_feature_message(&self.s_packed, &self.gate_bias, &self.s_message, b, r, t)?;
        if diag {
            gpu.hip.device_synchronize()?;
            us[5] = tq.elapsed().as_micros();
        }
        // 6. delta GEMM: message [b,R] × W_i[H,R]^T → delta [b,H].
        if diag {
            gpu.hip.device_synchronize()?;
        }
        let tq = std::time::Instant::now();
        gpu.gemm_f32_batched(&self.write_w[li], &self.s_message, &self.s_delta, h, r, b)?;
        if diag {
            gpu.hip.device_synchronize()?;
            us[6] = tq.elapsed().as_micros();
        }
        // 7. finish (kernel 4): x += rms * delta on non-first rows, in place.
        if diag {
            gpu.hip.device_synchronize()?;
        }
        let tq = std::time::Instant::now();
        gpu.dspine_finish(&scratch.x, &self.s_rms, &self.s_delta, b, h, t)?;
        if diag {
            gpu.hip.device_synchronize()?;
            us[7] = tq.elapsed().as_micros();
        }
        }
        if diag {
            eprintln!(
                "[dspine-bridge] li={} b={} snap={} prep={} read={} codes={} pack={} msg={} delta={} fin={} us",
                li, b, us[0], us[1], us[2], us[3], us[4], us[5], us[6], us[7]
            );
        }
        Ok(())
    }
}

/// Scalar reference for the bridge stages, shared by the Tier C kernel
/// assertions (`test_kernels` oracles) and these unit tests. All accumulation
/// is F32 exactly as the kernels will do it; BF16 rounding is applied by the
/// caller where the kernel stores BF16.
pub mod cpu_oracle {
    /// Serial refinement (): per block of  rows, walk
    /// from the anchor sender id; each row scores its K candidates as
    ///  (F32 accumulate over R), picks the argmax, and advances
    ///  to the winning candidate token id. / are the
    /// head-logits top-K of that row ([n, K] each); the three tables are
    /// flat [V, R] f32 views of the load-time sender tables. Returns the
    /// selected token id per row.
    pub fn serial_chain(
        base: &[f32],
        cand: &[u32],
        rms: &[f32],
        query: &[f32],
        tsv_gate: &[f32],
        tsv_val: &[f32],
        tg: &[f32],
        gate_b: &[f32],
        anchor_sid: u32,
        n: usize,
        k: usize,
        r: usize,
        t: usize,
        v: usize,
    ) -> Vec<u32> {
        assert_eq!(base.len(), n * k);
        assert_eq!(cand.len(), n * k);
        assert_eq!(rms.len(), n);
        assert_eq!(query.len(), n * r);
        assert_eq!(gate_b.len(), r);
        let mut sid = anchor_sid as usize;
        let mut out = Vec::with_capacity(n);
        for row in 0..n {
            if row % t == 0 {
                // Each draft block restarts its chain at the anchor: the
                // block-first row sender is the anchor token (upstream
                // semantic, mirrored by the kernel which initializes s_sid
                // once per block workgroup).
                sid = anchor_sid as usize;
            }
            assert!(sid < v, "sender id {sid} out of vocab");
            let q = &query[row * r..(row + 1) * r];
            let s_gate = &tsv_gate[sid * r..(sid + 1) * r];
            let s_val = &tsv_val[sid * r..(sid + 1) * r];
            let mut sender = vec![0.0f32; r];
            for j in 0..r {
                let gate = 1.0 / (1.0 + (-(q[j] + s_gate[j] + gate_b[j])).exp());
                sender[j] = gate * s_val[j];
            }
            let mut best_c = 0usize;
            let mut best_score = f32::NEG_INFINITY;
            for c in 0..k {
                let cand_id = cand[row * k + c] as usize;
                assert!(cand_id < v);
                let tg_row = &tg[cand_id * r..(cand_id + 1) * r];
                let mut s = 0.0f32;
                for j in 0..r {
                    s += sender[j] * tg_row[j];
                }
                let score = base[row * k + c] + rms[row] * s;
                if score > best_score {
                    best_score = score;
                    best_c = c;
                }
            }
            sid = cand[row * k + best_c] as usize;
            out.push(sid as u32);
        }
        out
    }
    /// Kernel 1 (`dspine_prepare_read`) stats half: RMS denominator of each
    /// row, F32 accumulate — the kernel stores this value so the finish stage
    /// can scale `delta` with it. `y = x / rms` defines `rms = sqrt(mean(x²)+eps)`.
    pub fn rms_rows(x: &[f32], h: usize, eps: f32) -> Vec<f32> {
        assert!(h > 0 && x.len() % h == 0);
        x.chunks_exact(h)
            .map(|row| {
                let sum: f32 = row.iter().map(|&v| v * v).sum();
                (sum / h as f32 + eps).sqrt()
            })
            .collect()
    }

    /// Kernel 2 (`dspine_normalize_codes`), one row. `None` = anchor row: the
    /// code is the codebook row scaled by sqrt(C) (identity path, no
    /// projection). `Some(p)` = non-anchor: L2-normalize `p` to sqrt(C) scale,
    /// F32 accumulate for the norm.
    pub fn normalize_code_row(
        projected: Option<&[f32]>,
        codebook_row: Option<&[f32]>,
        c: usize,
    ) -> Vec<f32> {
        assert!(c > 0);
        let scale = (c as f32).sqrt();
        match (projected, codebook_row) {
            (None, Some(code)) => {
                assert_eq!(code.len(), c);
                code.iter().map(|&v| v * scale).collect()
            }
            (Some(p), None) => {
                assert_eq!(p.len(), c);
                let norm: f32 = p.iter().map(|&v| v * v).sum::<f32>().sqrt();
                if norm > 0.0 {
                    let k = scale / norm;
                    p.iter().map(|&v| v * k).collect()
                } else {
                    vec![0.0; c]
                }
            }
            _ => panic!("normalize_code_row needs exactly one of projected/codebook_row"),
        }
    }

    /// Kernel 3 (`dspine_feature_message`), one row. `gate = sigmoid(query +
    /// sender_prev + bias)`, `message = gate * value_prev`. Block-first rows
    /// pass `None` predecessors and get an all-zero message.
    pub fn feature_message_row(
        query: &[f32],
        sender_prev: Option<&[f32]>,
        value_prev: Option<&[f32]>,
        bias: &[f32],
        r: usize,
    ) -> Vec<f32> {
        assert_eq!(query.len(), r);
        assert_eq!(bias.len(), r);
        match (sender_prev, value_prev) {
            (None, None) => vec![0.0; r],
            (Some(s), Some(v)) => {
                assert_eq!(s.len(), r);
                assert_eq!(v.len(), r);
                query
                    .iter()
                    .zip(s.iter())
                    .zip(v.iter())
                    .zip(bias.iter())
                    .map(|(((&q, &s), &v), &b)| {
                        let gate = 1.0 / (1.0 + (-(q + s + b)).exp());
                        gate * v
                    })
                    .collect()
            }
            _ => panic!("feature_message_row predecessors must be both None or both Some"),
        }
    }

    /// Kernel 4 (`dspine_feature_finish`), one row. Returns `(output, normed)`
    /// where `output = raw + rms*delta` on non-block-first rows (`raw` as-is
    /// on first rows) and `normed` is the next layer's input RMSNorm of
    /// `output`. The GPU side fuses `normed` into the next layer's existing
    /// `draft_layer_pre` attention norm — same math, one norm total.
    pub fn finish_row(
        raw: &[f32],
        rms: f32,
        delta: &[f32],
        first_row: bool,
        h: usize,
        eps: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        assert_eq!(raw.len(), h);
        assert_eq!(delta.len(), h);
        let output: Vec<f32> = raw
            .iter()
            .zip(delta.iter())
            .map(|(&r, &d)| if first_row { r } else { r + rms * d })
            .collect();
        let inv = 1.0 / rms_rows(&output, h, eps)[0];
        let normed = output.iter().map(|&v| v * inv).collect();
        (output, normed)
    }
}

#[cfg(test)]
mod tests {
    use super::DspineConfig;
    use serde_json::json;

    fn valid_metadata() -> serde_json::Value {
        json!({
            "architecture": "dspine",
            "dflash": {
                "num_hidden_layers": 5,
                "hidden_size": 5120,
                "intermediate_size": 17408,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "vocab_size": 248320,
                "block_size": 16,
                "mask_token_id": 248070,
                "target_layer_ids": [1, 16, 31, 46, 61],
                "num_target_layers": 64,
                "draft_dtype": "mq4"
            },
            "dspine": {
                "code_size": 1024,
                "bridge_rank": 512,
                "feature_layers": [1, 2, 3, 4, 5],
                "second_pass": {"mode": "serial_table", "first_exact": 4, "head_topk": 16},
                "versions": {
                    "draft_block_causal_version": 1,
                    "last_write_refine_version": 1,
                    "selected_prefix_bridge.independent_write_version": 1,
                    "selected_prefix_bridge.terminal_disabled_version": 1
                }
            }
        })
    }

    fn with_dspine_key(key: &str, value: serde_json::Value) -> serde_json::Value {
        let mut m = valid_metadata();
        m.get_mut("dspine")
            .unwrap()
            .as_object_mut()
            .unwrap()
            .insert(key.into(), value);
        m
    }

    #[test]
    fn accepts_the_converted_contract_shape() {
        let c = DspineConfig::from_metadata(&valid_metadata())
            .unwrap()
            .expect("valid metadata must parse");
        assert_eq!(c.code_size, 1024);
        assert_eq!(c.bridge_rank, 512);
        assert_eq!(c.trunk.hidden, 5120);
        assert_eq!(c.trunk.block_size, 16);
        assert_eq!(c.trunk.target_layer_ids, vec![1, 16, 31, 46, 61]);
        assert_eq!(c.head_topk, 16);
        assert_eq!(c.feature_layers, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn a_non_dspine_hfq_is_ok_none() {
        let mut m = valid_metadata();
        m.as_object_mut().unwrap().remove("dspine");
        assert!(DspineConfig::from_metadata(&m).unwrap().is_none());
    }

    #[test]
    fn missing_trunk_block_is_a_refusal() {
        let mut m = valid_metadata();
        m.as_object_mut().unwrap().remove("dflash");
        let err = DspineConfig::from_metadata(&m).unwrap_err();
        assert!(err.contains("trunk metadata"), "{err}");
    }

    #[test]
    fn non_serial_table_mode_is_refused() {
        let mut m = valid_metadata();
        m["dspine"]["second_pass"] = json!({"mode": "chain", "first_exact": 4, "head_topk": 16});
        let err = DspineConfig::from_metadata(&m).unwrap_err();
        assert!(err.contains("serial_table"), "{err}");
    }

    #[test]
    fn first_exact_must_match_layers_minus_one() {
        let mut m = valid_metadata();
        m["dspine"]["second_pass"]["first_exact"] = json!(3);
        let err = DspineConfig::from_metadata(&m).unwrap_err();
        assert!(err.contains("first_exact"), "{err}");
    }

    #[test]
    fn head_topk_bounds_and_pow2() {
        for bad in [1u64, 3, 48, 128] {
            let mut m = valid_metadata();
            m["dspine"]["second_pass"]["head_topk"] = json!(bad);
            let err = DspineConfig::from_metadata(&m).unwrap_err();
            assert!(err.contains("head_topk"), "{err}");
        }
    }

    #[test]
    fn feature_layers_must_land_on_draft_layers() {
        let m = with_dspine_key("feature_layers", json!([0, 2, 3, 4, 5]));
        let err = DspineConfig::from_metadata(&m).unwrap_err();
        assert!(err.contains("feature_layers"), "{err}");
        let m = with_dspine_key("feature_layers", json!([1, 2, 3, 4, 6]));
        let err = DspineConfig::from_metadata(&m).unwrap_err();
        assert!(err.contains("feature_layers"), "{err}");
        let m = with_dspine_key("feature_layers", json!([]));
        let err = DspineConfig::from_metadata(&m).unwrap_err();
        assert!(err.contains("empty"), "{err}");
    }

    #[test]
    fn every_version_scalar_is_rechecked() {
        for key in [
            "draft_block_causal_version",
            "last_write_refine_version",
            "selected_prefix_bridge.independent_write_version",
            "selected_prefix_bridge.terminal_disabled_version",
        ] {
            let mut m = valid_metadata();
            m["dspine"]["versions"][key] = json!(2);
            let err = DspineConfig::from_metadata(&m).unwrap_err();
            assert!(err.contains(key) && err.contains("expected 1"), "{err}");
        }
        let m = with_dspine_key(
            "versions",
            json!({"draft_block_causal_version": 1, "last_write_refine_version": 1}),
        );
        let err = DspineConfig::from_metadata(&m).unwrap_err();
        assert!(err.contains("missing"), "{err}");
    }

    #[test]
    fn oracle_rms_matches_the_norm_denominator_definition() {
        use super::cpu_oracle::rms_rows;
        let row = [3.0f32, 4.0];
        // mean(x²) = 12.5; sqrt(12.5 + eps) is the norm denominator.
        let rms = rms_rows(&row, 2, 1e-6)[0];
        let expected = (12.5f32 + 1e-6).sqrt();
        assert!((rms - expected).abs() < 1e-6, "{rms} vs {expected}");
        // A zero row must divide cleanly: rms = sqrt(eps), not 0.
        assert!((rms_rows(&[0.0, 0.0], 2, 1e-6)[0] - 1e-3f32).abs() < 1e-9);
    }

    #[test]
    fn oracle_anchor_rows_take_the_codebook_identity_path() {
        use super::cpu_oracle::normalize_code_row;
        let code = [0.5f32, -0.5, 1.0];
        let c = 3usize;
        let out = normalize_code_row(None, Some(&code), c);
        let scale = (c as f32).sqrt();
        for (got, &want) in out.iter().zip(code.iter()) {
            assert!((got - want * scale).abs() < 1e-6);
        }
    }

    #[test]
    fn oracle_non_anchor_rows_l2_normalize_to_sqrt_c_scale() {
        use super::cpu_oracle::normalize_code_row;
        let p = [3.0f32, 4.0];
        let out = normalize_code_row(Some(&p), None, 2);
        let norm = (3.0f32 * 3.0 + 4.0 * 4.0).sqrt();
        let want: Vec<f32> = p.iter().map(|&v| v / norm * 2.0f32.sqrt()).collect();
        for (got, w) in out.iter().zip(want.iter()) {
            assert!((got - w).abs() < 1e-6);
        }
        // Degenerate zero projection normalizes to zeros, not NaN.
        assert!(normalize_code_row(Some(&[0.0, 0.0]), None, 2).iter().all(|&v| v == 0.0));
    }

    #[test]
    fn oracle_block_first_rows_have_zero_message() {
        use super::cpu_oracle::feature_message_row;
        let r = 2usize;
        assert!(feature_message_row(&[1.0, 1.0], None, None, &[0.0, 0.0], r)
            .iter()
            .all(|&v| v == 0.0));
    }

    #[test]
    fn oracle_message_applies_the_gate_to_the_predecessor_value() {
        use super::cpu_oracle::feature_message_row;
        // query + sender_prev + bias = 0 → gate = 0.5; message = 0.5 * value_prev.
        let m = feature_message_row(
            &[0.0, 0.0],
            Some(&[0.0, 0.0]),
            Some(&[2.0, -6.0]),
            &[0.0, 0.0],
            2,
        );
        assert!((m[0] - 1.0).abs() < 1e-6, "{:?}", m);
        assert!((m[1] + 3.0).abs() < 1e-6, "{:?}", m);
        // Large negative argument drives the gate to 0.
        let m = feature_message_row(
            &[-100.0, -100.0],
            Some(&[-100.0, -100.0]),
            Some(&[2.0, 2.0]),
            &[0.0, 0.0],
            2,
        );
        assert!(m.iter().all(|&v| v.abs() < 1e-30));
    }

    #[test]
    fn oracle_finish_scales_delta_by_rms_and_skips_first_rows() {
        use super::cpu_oracle::finish_row;
        let h = 2usize;
        let eps = 1e-6f32;
        let (output, normed) = finish_row(&[1.0, 1.0], 2.0, &[0.5, -0.5], false, h, eps);
        assert!((output[0] - 2.0).abs() < 1e-6, "{:?}", output);
        assert!((output[1] - 0.0).abs() < 1e-6, "{:?}", output);
        // normed = output / rms(output): mean(out²) = 2 → rms = sqrt(2+eps).
        let inv = 1.0 / (2.0f32 + eps).sqrt();
        assert!((normed[0] - 2.0 * inv).abs() < 1e-5, "{:?}", normed);
        // Block-first rows pass raw through and still norm it.
        let (output, _) = finish_row(&[1.0, 1.0], 2.0, &[9.0, 9.0], true, h, eps);
        assert_eq!(output, vec![1.0, 1.0]);
    }

    #[test]
    fn serial_tables_flag_follows_metadata() {
        let base = valid_metadata();
        let cfg = DspineConfig::from_metadata(&base).unwrap().unwrap();
        assert!(!cfg.serial_tables, "plain sidecar must not claim tables");

        let with_tables = {
            let mut m = base.clone();
            m.as_object_mut()
                .unwrap()
                .get_mut("dspine")
                .unwrap()
                .as_object_mut()
                .unwrap()
                .insert(
                    "serial_tables".to_string(),
                    json!({"kind": "exact_sender", "head_topk": 16, "version": 1}),
                );
            m
        };
        let cfg = DspineConfig::from_metadata(&with_tables).unwrap().unwrap();
        assert!(cfg.serial_tables, "converter --target-hfq pass must flag tables");
    }
}
