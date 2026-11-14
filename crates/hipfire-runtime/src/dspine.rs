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
            gate_bias,
            serial,
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
    /// top-16 candidates per row (`dspine_topk16`), then the fused
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
        logits_pre: &GpuTensor,
        logits_post: &GpuTensor,
        rows: usize,
    ) -> hip_bridge::HipResult<Vec<u32>> {
        let tables = match self.serial.as_ref() {
            Some(t) => t,
            None => {
                return Err(hip_bridge::HipError::new(
                    0,
                    "serial_refine on a sidecar without TG/TSV tables",
                ));
            }
        };
        let vocab = self.cfg.trunk.vocab_size;
        // u32 payloads ride F32 tensors (4 B/elem): Raw is 1 B/elem and
        // download_f32 reads numel*4, so a Raw alloc here is a bounded
        // overread that only survives on pool granularity.
        let cand = gpu.alloc_tensor(&[rows, 16], rdna_compute::DType::F32)?;
        let base = gpu.alloc_tensor(&[rows, 16], rdna_compute::DType::F32)?;
        let out = gpu.alloc_tensor(&[rows], rdna_compute::DType::F32)?;
        eprintln!("[dspine-serial] rows={} base.buf={}B cand.buf={}B out.buf={}B",
            rows, base.byte_size(), cand.byte_size(), out.byte_size());
        // Experiment switch: HIPFIRE_DSPINE_SERIAL_MODE=argmax returns the
        // pre-write argmax (topk16 lane 0) and skips the sender chain —
        // isolates how much of the acceptance gap is the refinement term
        // vs the base ordering itself.
        let mode_argmax =
            hipfire_config::developer_var("HIPFIRE_DSPINE_SERIAL_MODE").as_deref() == Ok("argmax");
        let pick = (|| -> hip_bridge::HipResult<Vec<u32>> {
            // Upstream: candidates come from the forward's OUTPUT logits
            // (post-write hidden @ lm_head), while the score's base term is
            // evaluated at those ids on the PRE-write logits (serial_table
            // rescoring). One topk on the post logits + one gather.
            gpu.dspine_topk16(logits_post, &cand, &base, rows, vocab)?;
            gpu.dspine_gather16(logits_pre, &cand, &base, rows, vocab)?;
            if mode_argmax {
                let raw = gpu.download_f32(&cand)?;
                return Ok((0..rows).map(|i| raw[i * 16].to_bits()).collect());
            }
            // stream-ordered: the plain readback blocks on the legacy null
            // stream, syncing the whole decode queue (8x measured earlier).
            let anchor_raw = gpu.download_f32_stream_ordered(&self.s_anchor)?;
            let anchor_sid = anchor_raw[0].to_bits();
            // logits row r is draft row r+1 (the seed row has no head
            // logits), so the sender's rms/query must come from scratch
            // rows 1..=rows, not 0..rows: one row off = every gate reads
            // its neighbor's packed code and acceptance collapses.
            let rms_rows = self.s_rms.sub_offset(1, rows);
            let packed_rows = self
                .s_packed
                .sub_offset(3 * self.r, rows * 3 * self.r);
            gpu.dspine_serial_chain(
                &base,
                &cand,
                &rms_rows,
                &packed_rows,
                &tables.tsv_gate,
                &tables.tsv_val,
                &tables.tg,
                &self.gate_bias,
                anchor_sid,
                &out,
                rows,
                16,
                self.r,
                self.t,
                3 * self.r,
                self.rms_scale(),
            )?;
            let raw = gpu.download_f32_stream_ordered(&out)?;
            if hipfire_config::developer_var("HIPFIRE_DSPINE_SERIAL_CAPTURE")
                .as_deref()
                == Ok("1")
            {
                // Full-input snapshot for offline oracle replay: every
                // serial_chain input downloaded with checked results, one
                // directory per window. The offline python oracle replays
                // the chain and compares picks bit-for-bit.
                static CAP_SEQ: std::sync::atomic::AtomicU64 =
                    std::sync::atomic::AtomicU64::new(0);
                let seq = CAP_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let dir = format!("/tmp/dspine_capture/win_{:05}", seq);
                let _ = std::fs::create_dir_all(&dir);
                let dl = |t: &GpuTensor, name: &str| -> Option<Vec<f32>> {
                    match gpu.download_f32(t) {
                        Ok(v) => Some(v),
                        Err(e) => {
                            eprintln!("[dspine-capture] {} download failed: {}", name, e);
                            None
                        }
                    }
                };
                let cand_d = dl(&cand, "cand");
                let base_d = dl(&base, "base");
                let out_d = dl(&out, "out");
                let rms_d = dl(&rms_rows, "rms_rows");
                let packed_d = dl(&packed_rows, "packed_rows");
                let gb_d = dl(&self.gate_bias, "gate_bias");
                let anchor_d = dl(&self.s_anchor, "anchor");
                let mut snap = |name: &str, v: &Option<Vec<f32>>| {
                    if let Some(v) = v {
                        if let Ok(mut f) = std::fs::File::create(format!("{}/{}.bin", dir, name)) {
                            use std::io::Write as _;
                            let _ = f.write_all(unsafe {
                                std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4)
                            });
                        }
                    }
                };
                snap("cand", &cand_d);
                snap("base", &base_d);
                snap("out", &out_d);
                snap("rms", &rms_d);
                snap("packed", &packed_d);
                snap("gate_bias", &gb_d);
                snap("anchor", &anchor_d);
                if let (Some(c), Some(o), Some(a)) = (&cand_d, &out_d, &anchor_d) {
                    let meta = serde_json::json!({
                        "rows": rows,
                        "vocab": self.cfg.trunk.vocab_size,
                        "r": self.r,
                        "t": self.t,
                        "anchor_sid": a[0].to_bits(),
                        "picks": o.iter().map(|f| f.to_bits()).collect::<Vec<u32>>(),
                        "argmax": c.chunks_exact(16).map(|c| c[0].to_bits()).collect::<Vec<u32>>(),
                    });
                    if let Ok(mut f) = std::fs::File::create(format!("{}/meta.json", dir)) {
                        use std::io::Write as _;
                        let _ = f.write_all(meta.to_string().as_bytes());
                    }
                }
            }
            let picks: Vec<u32> = raw.iter().map(|f| f.to_bits()).collect();
            if hipfire_config::developer_var("HIPFIRE_DSPINE_SERIAL_DIAG").as_deref() == Ok("1") {
                let base_raw = gpu.download_f32(&base)?;
                let cand_raw = gpu.download_f32(&cand)?;
                let top1: Vec<u32> = cand_raw.iter().map(|f| f.to_bits()).collect();
                let agree = picks.iter().enumerate().filter(|(i, p)| **p == top1[i * 16]).count();
                // Coverage: is the post-write argmax id even inside this
                // row's pre-write top-16 shortlist? Below 100% caps the
                // refinement's acceptance no matter how good the score is.
                let covered = (0..rows)
                    .filter(|&i| (0..16).any(|k| top1[i * 16 + k] == picks[i]))
                    .count();
                let top1_in_short = (0..rows)
                    .filter(|&i| {
                        let want = top1[i * 16];
                        (0..16).any(|k| {
                            cand_raw[i * 16 + k].to_bits() == want
                        })
                    })
                    .count();
                // Per-row hit bitmap against the post-write argmax: shows
                // where the chain drifts (row 0 has no accumulated sid
                // error; deep rows inherit every earlier miss).
                let hits: String = (0..rows)
                    .map(|i| {
                        if picks[i] == top1[i * 16] { '1' } else { '0' }
                    })
                    .collect();
                eprintln!("[dspine-serial] rows={} agree={} covered={} top1_in_short={}/{} hits={} picks0..8={:?} top1-0..8={:?} base00={:.3}",
                    rows, agree, covered, top1_in_short, rows, hits,
                    &picks[..rows.min(8)],
                    &top1[..rows.min(8)],
                    base_raw.first().copied().unwrap_or(0.0));
            }
            Ok(picks)
        })();
        let _ = gpu.free_tensor(cand);
        let _ = gpu.free_tensor(base);
        let _ = gpu.free_tensor(out);
        pick
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
        let _ = gpu.free_tensor(gate_bias);
        if let Some(tables) = serial {
            let _ = gpu.free_tensor(tables.tg);
            let _ = gpu.free_tensor(tables.tsv_gate);
            let _ = gpu.free_tensor(tables.tsv_val);
        }
        for t in [s_normed, s_projected, s_codes, s_packed, s_message, s_delta, s_rms, s_anchor, s_x_pre] {
            let _ = gpu.free_tensor(t);
        }
    }

    /// Feed the block's first real token id (the code anchor) before each
    /// bridged forward. The speculator driver owns the token.
    pub fn set_anchor(&mut self, gpu: &mut Gpu, anchor: u32) -> Result<(), String> {
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

        // Snapshot the pre-write hidden rows first: finish rewrites x
        // in place, and the serial refinement's base term reads the
        // pre-write values exactly like the upstream stash.
        gpu
            .memcpy_dtod_at_ordered_async(&self.s_x_pre.buf, 0, &scratch.x.buf, 0, b * h * 4)
            .map_err(|e| hip_bridge::HipError::new(e.code, &format!("dspine x_pre snapshot: {e}")))?;

        // 1. rms stats + normalized rows (kernel 1).
        gpu.dspine_prepare_read(&scratch.x, &self.s_rms, &self.s_normed, b, h, eps)?;
        // 2. projection GEMM: normed [b,H] × W_i[C,H]^T → projected [b,C].
        //    Layers 0..3 use early_read_weights[i]; the last layer read_weight.
        let w = if li < self.early_w.len() {
            &self.early_w[li]
        } else {
            &self.read_w
        };
        // gemm computes y = b_arg·a_arg^T ([n,m] row-major): pass the weight
        // first so y lands as projected [b, C] row-major.
        gpu.gemm_f32_batched(w, &self.s_normed, &self.s_projected, c, h, b)?;
        // 3. normalize_codes (kernel 2): anchor identity / L2 to sqrt(C).
        gpu.dspine_normalize_codes(&self.s_projected, &self.codebook, &self.s_anchor, &self.s_codes, b, c, t)?;
        // 4. packed GEMM: codes [b,C] × packed_w[3R,C]^T → packed [b,3R].
        gpu.gemm_f32_batched(&self.packed_w, &self.s_codes, &self.s_packed, 3 * r, c, b)?;
        // 5. feature_message (kernel 3): gate · predecessor value.
        gpu.dspine_feature_message(&self.s_packed, &self.gate_bias, &self.s_message, b, r, t)?;
        // 6. delta GEMM: message [b,R] × W_i[H,R]^T → delta [b,H].
        gpu.gemm_f32_batched(&self.write_w[li], &self.s_message, &self.s_delta, h, r, b)?;
        // 7. finish (kernel 4): x += rms * delta on non-first rows, in place.
        gpu.dspine_finish(&scratch.x, &self.s_rms, &self.s_delta, b, h, t)?;
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
