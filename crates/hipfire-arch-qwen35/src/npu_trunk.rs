//! The qwen3.8-27b dense co-processor stage (NPU trunk): its switch, its op-level stage map, the
//! W8 weight packer and the device half of the bank-pair prefill (`on`: pack, kick, wait, consume
//! on the NPU; `emulate`: the same W8 x A8 GEMM on the GPU twin). The pair loop itself is
//! `qwen35::prefill::forward_prefill_npu_pair`.

use hipfire_runtime::coproc::calibration::{model_trunk_index_sha256, runtime_calibration_key};
use hipfire_runtime::coproc::{
    admission::off_line, auto_admission, pair_time, Calibration, EpilogueContract, FamilyDecl,
    GroupDecl, LayerKindDecl, Mode, StageMap, Switch, CHUNK,
};
use peacemaker_npu::kernels::gemm_core::{Control, Epilogue};
use peacemaker_npu::kernels::gemm_g80_n512::{design_g80_n512, pack_b_bf16, WAVE_N};
use peacemaker_npu::kernels::iu4::{f16_to_f32, GROUP_BYTES};
use peacemaker_npu::kernels::w8a8_epi::w8_direct_row;
use rayon::prelude::*;

/// Model width: the reduction dimension of every input projection of the 27B trunk.
const K: usize = 5120;

/// Developer switch of the 27B stage; `fusions.qwen35_npu_trunk` is its config key.
pub const SWITCH: Switch = Switch {
    env: "HIPFIRE_QWEN35_NPU_TRUNK",
    config_key: "fusions.qwen35_npu_trunk",
    tag: "qwen35-npu",
    name: "qwen35_npu_trunk",
};

/// Developer fault injection: the 1-based relay chain whose watchdog is forced (the real NPU
/// timeout path).
pub const FAULT_CHAIN: &str = "HIPFIRE_QWEN35_NPU_TRUNK_FAULT_CHAIN";

/// The compiled-in measured Halo record (`qwen3.8-27b.mq4`, gfx1151 + XDNA2), the only record
/// `auto` binds.
const BUILTIN: &str = include_str!("npu_trunk/halo-xdna2-27b-v1.json");

/// The resolved mode (default `off`), or a load error naming the switch.
pub fn resolve_mode() -> Result<Mode, String> {
    Mode::from_config(SWITCH, Mode::Off)
}

/// The index digest `auto` keys its calibration on, taken while the loader still owns the HFQ;
/// `None` in every other mode.
pub fn model_index(
    mode: Mode,
    hfq: &hipfire_runtime::hfq::HfqFile,
) -> Option<Result<String, String>> {
    (mode == Mode::Auto).then(|| model_trunk_index_sha256(hfq))
}

/// `auto` admission: the built-in record bound to this host (GPU `pci`, NPU firmware, kernel
/// release), for the model `index`. Admits only when the planner selects every family (the
/// driver owns all seven); otherwise logs one `off` line with the reason and returns `None`.
pub fn auto_calibration(
    arch: &str,
    pci: &str,
    index: &Result<String, String>,
) -> Option<Calibration> {
    let admit = || -> Result<Calibration, String> {
        let mut cal: Calibration = serde_json::from_str(BUILTIN).map_err(|e| e.to_string())?;
        cal.validate(&STAGE_MAP)?;
        let key =
            runtime_calibration_key(arch, pci, index.as_ref().map_err(Clone::clone)?, "stock")?;
        cal.bind_builtin(&key)?;
        if auto_admission(&STAGE_MAP, &cal, &key)? != STAGE_MAP.all() {
            return Err("calibration selects a family subset; the driver owns every family".into());
        }
        Ok(cal)
    };
    admit()
        .map_err(|e| eprintln!("{}", off_line(SWITCH, false, &e)))
        .ok()
}

/// The `auto` pair test at prompt position `pos`: a full bank A (`CHUNK` rows) plus a bank B of
/// `b` rows pairs only when the calibrated pair time strictly beats GPU-only by the record's
/// margin, the planner's natural-pair test at this prefix.
pub fn pair_beats_gpu(cal: &Calibration, pos: usize, b: usize) -> bool {
    b >= cal.gpu.min_bank_rows && {
        let t = pair_time(&STAGE_MAP, cal, STAGE_MAP.all(), CHUNK, pos, b, pos + CHUNK);
        t.t < (1.0 - cal.margin) * t.gpu_only
    }
}

/// The input projections the NPU may own, grouped by kind: GDN qkv, z, a, b, then FA q, k, v.
/// Columns are output rows: GDN 10240 + 6144 + 48 + 48, FA gated q 12288 + k 1024 + v 1024.
const FAMILIES: [FamilyDecl; 7] = [
    FamilyDecl {
        name: "qkv",
        kind: 0,
        cols: 10240,
    },
    FamilyDecl {
        name: "z",
        kind: 0,
        cols: 6144,
    },
    FamilyDecl {
        name: "a",
        kind: 0,
        cols: 48,
    },
    FamilyDecl {
        name: "b",
        kind: 0,
        cols: 48,
    },
    FamilyDecl {
        name: "q",
        kind: 1,
        cols: 12288,
    },
    FamilyDecl {
        name: "k",
        kind: 1,
        cols: 1024,
    },
    FamilyDecl {
        name: "v",
        kind: 1,
        cols: 1024,
    },
];

// Layer counts are the checkpoint's `layer_types` (qwen3.8-27b.mq4: 48 linear_attention,
// 16 full_attention); `hidden_size` is 5120.
// ponytail: out-family names mirror qwen4 and are checked only for distinctness; the driver PR
// calibrates the real families and runs `check_model` on the loaded layer kinds before any plan.
pub static STAGE_MAP: StageMap = StageMap {
    tag: "qwen35-npu",
    kinds: [
        LayerKindDecl {
            name: "gdn",
            layers: 48,
            in_k: K,
            out_family: "gdn.out",
        },
        LayerKindDecl {
            name: "fa",
            layers: 16,
            in_k: K,
            out_family: "fa.o",
        },
    ],
    families: &FAMILIES,
    groups: [
        GroupDecl {
            name: "qkv",
            kind: 0,
            families: 0x01,
        },
        GroupDecl {
            name: "zab",
            kind: 0,
            families: 0x0e,
        },
        GroupDecl {
            name: "q",
            kind: 1,
            families: 0x10,
        },
        GroupDecl {
            name: "kv",
            kind: 1,
            families: 0x60,
        },
    ],
    contract: EpilogueContract::Bf16Direct,
};

/// Bytes of one K-wide MQ4G256V2 row.
const ROW_BYTES: usize = K / 256 * GROUP_BYTES;

/// One stage group packed for the N512 K5120 program (contract `bf16_direct`): its fused columns
/// (the group's families in declaration order) in 512-column B panels, the last one zero padded.
/// A twin pack (the `emulate` GPU GEMM) keeps the row-major W8 codes instead of the panels.
#[derive(Clone, Debug, PartialEq)]
pub struct PackedGroup {
    /// `pack_b_bf16` of the N512 design, one per 512 fused columns; empty for a twin pack.
    pub panels: Vec<Vec<u8>>,
    /// The grid15-up row scale per padded fused column (zero in the padding).
    pub ws: Vec<f32>,
    /// `(Sm, Se)` of `ws`, what the panels carry as sidecars.
    pub scales: Vec<(u16, i16)>,
    /// Twin pack only: `[live cols][K]` W8 codes; empty otherwise.
    pub codes: Vec<i8>,
}

/// `w = q * sc_h + zp_h` of one MQ4G256V2 row: header `[sc0 zp0 sc1 zp1]` f16, element `e` in
/// byte `8 + e / 2` (low nibble even). The rows stay in the rotated domain they are stored in.
fn decode_row(row: &[u8], out: &mut [f32]) {
    for (group, dst) in row.chunks_exact(GROUP_BYTES).zip(out.chunks_exact_mut(256)) {
        let hdr = |i: usize| f16_to_f32(u16::from_le_bytes([group[2 * i], group[2 * i + 1]]));
        let halves = [(hdr(0), hdr(1)), (hdr(2), hdr(3))];
        for (e, w) in dst.iter_mut().enumerate() {
            let q = (group[8 + e / 2] >> (4 * (e % 2))) & 15;
            let (sc, zp) = halves[e / 128];
            // Separate multiply and add, never fused.
            *w = f32::from(q) * sc + zp;
        }
    }
}

/// Pack stage group `group` of [`STAGE_MAP`]: `parts` are its families' MQ4G256V2 matrices in
/// declaration order, each exactly `cols * K / 256 * 136` bytes (slice a folded buffer such as the
/// loader's `wz`, whose first 6144 rows are z, down to the family's rows first). `twin` keeps the
/// codes for the GPU twin instead of building NPU panels.
pub fn pack_group(group: usize, parts: &[&[u8]], twin: bool) -> Result<PackedGroup, String> {
    let decl = STAGE_MAP
        .groups
        .get(group)
        .ok_or_else(|| format!("no stage group {group}"))?;
    let families: Vec<&FamilyDecl> = (0..FAMILIES.len())
        .filter(|f| decl.families & (1 << f) != 0)
        .map(|f| &FAMILIES[f])
        .collect();
    if parts.len() != families.len() {
        return Err(format!(
            "group {}: {} matrices for {} families",
            decl.name,
            parts.len(),
            families.len()
        ));
    }
    let mut rows: Vec<&[u8]> = Vec::new();
    for (fam, bytes) in families.iter().zip(parts) {
        if bytes.len() != fam.cols * ROW_BYTES {
            return Err(format!(
                "{}: expected {} MQ4G256V2 bytes for {}x{K}, got {}",
                fam.name,
                fam.cols * ROW_BYTES,
                fam.cols,
                bytes.len()
            ));
        }
        rows.extend(bytes.chunks_exact(ROW_BYTES));
    }
    let design = design_g80_n512(1, WAVE_N, K, Epilogue::Bf16, Control::Fast);
    let done: Vec<(Vec<u8>, Vec<i8>, Vec<f32>, Vec<(u16, i16)>)> = rows
        .par_chunks(WAVE_N)
        .map(|panel| {
            let mut b = if twin {
                Vec::new()
            } else {
                vec![0i8; K * WAVE_N]
            };
            let mut row_codes = if twin {
                vec![0i8; K * panel.len()]
            } else {
                Vec::new()
            };
            let (mut w, mut codes) = (vec![0.0f32; K], vec![0i8; K]);
            let mut ws = vec![0.0f32; WAVE_N];
            let mut scales = vec![(0, 0); WAVE_N];
            for (c, row) in panel.iter().enumerate() {
                decode_row(row, &mut w);
                (ws[c], scales[c]) = w8_direct_row(&w, &mut codes);
                if twin {
                    row_codes[c * K..(c + 1) * K].copy_from_slice(&codes);
                } else {
                    for (k, &v) in codes.iter().enumerate() {
                        b[k * WAVE_N + c] = v;
                    }
                }
            }
            let panel = if twin {
                Vec::new()
            } else {
                pack_b_bf16(&design, &b, &scales)
            };
            (panel, row_codes, ws, scales)
        })
        .collect();
    let mut out = PackedGroup {
        panels: Vec::with_capacity(done.len()),
        ws: Vec::with_capacity(done.len() * WAVE_N),
        scales: Vec::with_capacity(done.len() * WAVE_N),
        codes: Vec::with_capacity(if twin { rows.len() * K } else { 0 }),
    };
    for (panel, codes, ws, scales) in done {
        if !twin {
            out.panels.push(panel);
        }
        out.codes.extend(codes);
        out.ws.extend(ws);
        out.scales.extend(scales);
    }
    Ok(out)
}

#[cfg(target_os = "linux")]
#[path = "npu_trunk_driver.rs"]
mod driver;
#[cfg(target_os = "linux")]
pub use driver::{NpuTrunk, BANK_ROWS};

/// Outer prefill chunk for a caller that splits a prompt before `forward_prefill_batch`, at
/// prompt position `pos` with `remaining` rows left: a full-bank `chunk_max` doubles exactly when
/// the armed trunk admits the pair there (both banks must be in one call). A refused pair keeps
/// the caller's chunking, so its end-of-chunk checkpoints equal `off`. Drops any cut an earlier
/// forward left unconsumed, so [`take_pair_checkpoint`] only sees this chunk's.
pub fn outer_chunk_rows(
    scratch: &crate::qwen35::Qwen35Scratch,
    chunk_max: usize,
    pos: usize,
    remaining: usize,
) -> usize {
    #[cfg(target_os = "linux")]
    if let Some(npu) = scratch.npu_trunk.borrow().as_ref() {
        npu.discard_cut();
        if chunk_max >= BANK_ROWS
            && NpuTrunk::bank_b_rows(remaining).is_some_and(|b| npu.admits_pair(pos, b))
        {
            return 2 * BANK_ROWS;
        }
    }
    let _ = (scratch, pos, remaining);
    chunk_max
}

/// After a `forward_prefill_batch` whose outer chunk came from [`outer_chunk_rows`]: offers the
/// recurrent state a pair staged at its bank boundary to the caller's checkpoint ring, ahead of
/// the caller's own end-of-chunk checkpoint, so a paired prompt keeps the snapshots an
/// unpaired one takes between its chunks. `end` is where the forward ended and `prefix` the
/// committed tokens `[0..end]`. In `off`'s order: the stride snapshot that ends the first chunk,
/// then, when bank B holds the user-turn boundary, the pinned before-chunk snapshot of the second.
#[allow(clippy::too_many_arguments)]
pub fn take_pair_checkpoint(
    scratch: &crate::qwen35::Qwen35Scratch,
    cks: &mut Vec<(usize, crate::speculative::DeltaNetSnapshot)>,
    dn: &crate::qwen35::DeltaNetState,
    kv: &hipfire_runtime::llama::KvCache,
    gpu: &mut rdna_compute::Gpu,
    interval: usize,
    cap: usize,
    turn: &mut crate::speculative::TurnCheckpoint,
    end: usize,
    prefix: (&[u32], &[u32]),
) {
    #[cfg(target_os = "linux")]
    if let Some(npu) = scratch.npu_trunk.borrow().as_ref() {
        if let Some((pos, staged)) = npu.take_cut() {
            let (head, tail) = prefix;
            let prefix = if pos <= head.len() {
                (&head[..pos], &[][..])
            } else {
                (head, &tail[..(pos - head.len()).min(tail.len())])
            };
            crate::speculative::take_dn_checkpoint_staged(
                cks, &staged, dn, kv, gpu, pos, interval, cap, turn.pin, prefix,
            );
            if hipfire_runtime::prompt_frame::chunk_holds_turn_boundary(
                pos,
                end.saturating_sub(pos),
                turn.boundary,
            ) {
                crate::speculative::take_dn_checkpoint_staged(
                    cks,
                    &staged,
                    dn,
                    kv,
                    gpu,
                    pos,
                    0,
                    cap,
                    Some(pos),
                    prefix,
                );
                if cks.iter().any(|(p, _)| *p == pos) {
                    turn.pin = Some(pos);
                }
            }
        }
    }
    let _ = (scratch, cks, dn, kv, gpu, interval, cap, turn, end, prefix);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_is_valid_and_matches_the_27b_dims() {
        STAGE_MAP.validate().unwrap();
        let kinds: Vec<usize> = (0..64).map(|l| usize::from(l % 4 == 3)).collect();
        STAGE_MAP.check_model(&kinds, 5120).unwrap();
    }

    fn builtin() -> Calibration {
        serde_json::from_str(BUILTIN).unwrap()
    }

    /// The shipped record binds its own key (qwen3.8-27b.mq4 index digest) and `auto` admits every
    /// family, the only ownership the driver runs.
    #[test]
    fn builtin_record_admits_every_family_for_its_key() {
        let cal = builtin();
        cal.validate(&STAGE_MAP).unwrap();
        assert_eq!(
            cal.key.model_trunk_index_sha256,
            "feb45abd725a69db31863f457c3fb90cab9f84526117e799af6383a4f382e03a"
        );
        assert_eq!(
            auto_admission(&STAGE_MAP, &cal, &cal.key).unwrap(),
            STAGE_MAP.all()
        );
    }

    /// Halo measured `on` against `off`: an 8192 + 8192 pair saves 11.5% (p16k) and an 8192 + 1305
    /// pair loses 6.7% (p9); `auto` must take the first and refuse the second, at any prefix.
    #[test]
    fn auto_pairs_full_banks_and_refuses_a_short_bank_b() {
        let cal = builtin();
        assert!(pair_beats_gpu(&cal, 0, CHUNK));
        assert!(pair_beats_gpu(&cal, 6 * CHUNK, CHUNK));
        assert!(!pair_beats_gpu(&cal, 0, 1305));
        assert!(!pair_beats_gpu(&cal, 6 * CHUNK, 1305));
        assert!(!pair_beats_gpu(&cal, 0, cal.gpu.min_bank_rows - 64));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn bank_pairs_need_a_full_bank_a_and_a_real_bank_b() {
        use driver::{NpuTrunk, BANK_ROWS, MIN_BANK_B_ROWS};
        assert_eq!(NpuTrunk::bank_b_rows(BANK_ROWS), None);
        assert_eq!(NpuTrunk::bank_b_rows(BANK_ROWS + MIN_BANK_B_ROWS - 1), None);
        assert_eq!(
            NpuTrunk::bank_b_rows(BANK_ROWS + MIN_BANK_B_ROWS),
            Some(MIN_BANK_B_ROWS)
        );
        assert_eq!(NpuTrunk::bank_b_rows(3 * BANK_ROWS), Some(BANK_ROWS));
    }

    /// Header f16 `(bits, value)` pairs the reference widens without a converter.
    const SC: [(u16, f32); 3] = [(0x3800, 0.5), (0x3C00, 1.0), (0x4000, 2.0)];
    const ZP: [(u16, f32); 3] = [(0xC000, -2.0), (0x0000, 0.0), (0xBC00, -1.0)];

    /// `rows` synthetic MQ4G256V2 rows (distinct halves and nibbles per row), plus the reference
    /// decode `[rows][K]` taken element by element from the header table.
    fn synth(rows: usize, salt: usize) -> (Vec<u8>, Vec<Vec<f32>>) {
        let mut bytes = Vec::with_capacity(rows * ROW_BYTES);
        let mut want = Vec::with_capacity(rows);
        for r in 0..rows {
            let mut w = Vec::with_capacity(K);
            for g in 0..K / 256 {
                let h = [(r + g + salt) % 3, (r + 2 * g + salt + 1) % 3];
                for half in h {
                    bytes.extend(SC[half].0.to_le_bytes());
                    bytes.extend(ZP[half].0.to_le_bytes());
                }
                let q = |e: usize| ((r * 7 + g * 3 + e * 5 + salt) % 16) as u8;
                bytes.extend((0..128).map(|j| q(2 * j) | (q(2 * j + 1) << 4)));
                w.extend((0..256).map(|e| f32::from(q(e)) * SC[h[e / 128]].1 + ZP[h[e / 128]].1));
            }
            want.push(w);
        }
        (bytes, want)
    }

    #[test]
    fn kv_group_packs_k_then_v_rows_into_n512_panels_and_twin_codes() {
        let (k, kw) = synth(1024, 0);
        let (v, vw) = synth(1024, 1);
        let got = pack_group(3, &[&k, &v], false).unwrap();
        let twin = pack_group(3, &[&k, &v], true).unwrap();
        assert!(twin.panels.is_empty() && got.codes.is_empty());
        assert_eq!((&twin.ws, &twin.scales), (&got.ws, &got.scales));
        assert_eq!(twin.codes.len(), 2048 * K);
        let rows: Vec<&Vec<f32>> = kw.iter().chain(&vw).collect();
        assert_eq!(got.panels.len(), 4);
        let design = design_g80_n512(1, WAVE_N, K, Epilogue::Bf16, Control::Fast);
        for (p, panel) in got.panels.iter().enumerate() {
            let mut b = vec![0i8; K * WAVE_N];
            let mut scales = Vec::new();
            let mut codes = vec![0i8; K];
            for c in 0..WAVE_N {
                let row = p * WAVE_N + c;
                let (ws, grid) = w8_direct_row(rows[row], &mut codes);
                assert_eq!(got.ws[row], ws);
                assert_eq!(&twin.codes[row * K..(row + 1) * K], &codes[..], "row {row}");
                scales.push(grid);
                for (kk, &x) in codes.iter().enumerate() {
                    b[kk * WAVE_N + c] = x;
                }
            }
            assert_eq!(panel, &pack_b_bf16(&design, &b, &scales), "panel {p}");
        }
    }

    #[test]
    fn zab_group_pads_its_last_panel_and_checks_lengths() {
        let (z, _) = synth(6144, 0);
        let (a, _) = synth(48, 1);
        let (b, _) = synth(48, 2);
        let got = pack_group(1, &[&z, &a, &b], false).unwrap();
        assert_eq!(got.panels.len(), 13);
        assert!(got.ws[..6240].iter().all(|&w| w > 0.0));
        assert!(got.ws[6240..].iter().all(|&w| w == 0.0));
        let err = pack_group(1, &[&z[..ROW_BYTES * 6143], &a, &b], false).unwrap_err();
        assert!(err.starts_with("z: expected"), "{err}");
        assert!(pack_group(1, &[&z, &a], false).is_err());
    }
}
