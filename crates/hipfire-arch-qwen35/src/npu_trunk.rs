//! The qwen3.8-27b dense co-processor stage (NPU trunk): its switch, its op-level stage map, the
//! load-time gate and the W8 weight packer. The prefill driver is not in this crate yet, so every
//! non-off mode is refused at load. The map is the validated shape the driver will consume.

use hipfire_runtime::coproc::{
    EpilogueContract, FamilyDecl, GroupDecl, LayerKindDecl, Mode, StageMap, Switch,
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

/// Refuses every non-off mode at load: the 27B prefill driver is not in this build.
pub fn check_mode() -> Result<(), String> {
    refuse_unless_off(Mode::from_config(SWITCH, Mode::Off)?)
}

fn refuse_unless_off(mode: Mode) -> Result<(), String> {
    if mode.is_off() {
        return Ok(());
    }
    Err(format!(
        "{}={mode}: the qwen3.8-27b NPU prefill driver is not in this build; keep the switch off",
        SWITCH.env
    ))
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
#[derive(Clone, Debug, PartialEq)]
pub struct PackedGroup {
    /// `pack_b_bf16` of the N512 design, one per 512 fused columns.
    pub panels: Vec<Vec<u8>>,
    /// The grid15-up row scale per padded fused column (zero in the padding).
    pub ws: Vec<f32>,
    /// `(Sm, Se)` of `ws`, what the panels carry as sidecars.
    pub scales: Vec<(u16, i16)>,
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
/// loader's `wz`, whose first 6144 rows are z, down to the family's rows first).
pub fn pack_group(group: usize, parts: &[&[u8]]) -> Result<PackedGroup, String> {
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
    let done: Vec<(Vec<u8>, Vec<f32>, Vec<(u16, i16)>)> = rows
        .par_chunks(WAVE_N)
        .map(|panel| {
            let mut b = vec![0i8; K * WAVE_N];
            let (mut w, mut codes) = (vec![0.0f32; K], vec![0i8; K]);
            let mut ws = vec![0.0f32; WAVE_N];
            let mut scales = vec![(0, 0); WAVE_N];
            for (c, row) in panel.iter().enumerate() {
                decode_row(row, &mut w);
                (ws[c], scales[c]) = w8_direct_row(&w, &mut codes);
                for (k, &v) in codes.iter().enumerate() {
                    b[k * WAVE_N + c] = v;
                }
            }
            (pack_b_bf16(&design, &b, &scales), ws, scales)
        })
        .collect();
    let mut out = PackedGroup {
        panels: Vec::with_capacity(done.len()),
        ws: Vec::with_capacity(done.len() * WAVE_N),
        scales: Vec::with_capacity(done.len() * WAVE_N),
    };
    for (panel, ws, scales) in done {
        out.panels.push(panel);
        out.ws.extend(ws);
        out.scales.extend(scales);
    }
    Ok(out)
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

    #[test]
    fn only_off_passes_the_load_gate() {
        assert!(refuse_unless_off(Mode::Off).is_ok());
        for mode in [Mode::Auto, Mode::On, Mode::Emulate] {
            let err = refuse_unless_off(mode).unwrap_err();
            assert!(err.starts_with("HIPFIRE_QWEN35_NPU_TRUNK="), "{err}");
        }
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
    fn kv_group_packs_k_then_v_rows_into_n512_panels() {
        let (k, kw) = synth(1024, 0);
        let (v, vw) = synth(1024, 1);
        let got = pack_group(3, &[&k, &v]).unwrap();
        let rows: Vec<&Vec<f32>> = kw.iter().chain(&vw).collect();
        assert_eq!(got.panels.len(), 4);
        let design = design_g80_n512(1, WAVE_N, K, Epilogue::Bf16, Control::Fast);
        for (p, panel) in got.panels.iter().enumerate() {
            let mut b = vec![0i8; K * WAVE_N];
            let mut scales = Vec::new();
            let mut codes = vec![0i8; K];
            for c in 0..WAVE_N {
                let (ws, grid) = w8_direct_row(rows[p * WAVE_N + c], &mut codes);
                assert_eq!(got.ws[p * WAVE_N + c], ws);
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
        let got = pack_group(1, &[&z, &a, &b]).unwrap();
        assert_eq!(got.panels.len(), 13);
        assert!(got.ws[..6240].iter().all(|&w| w > 0.0));
        assert!(got.ws[6240..].iter().all(|&w| w == 0.0));
        let err = pack_group(1, &[&z[..ROW_BYTES * 6143], &a, &b]).unwrap_err();
        assert!(err.starts_with("z: expected"), "{err}");
        assert!(pack_group(1, &[&z, &a]).is_err());
    }
}
