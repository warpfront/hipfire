// SPDX-License-Identifier: Apache-2.0
//! Thin projection adapters; route choice and device transport are not owned here.
use super::{ColumnMap, Family};
use hip_bridge::{HipError, HipResult};
use rdna_compute::{
    npu_trunk::{NpuBf16Layout, NpuTrunkOutput},
    Gpu, GpuTensor,
};
impl ColumnMap {
    /// Bind a selected family to its native [rows,family.cols()] destination.
    /// Construct once per bank; no fused output allocation or launch-time map upload.
    pub fn gpu_output<'a>(
        &self,
        family: Family,
        tensor: &'a GpuTensor,
    ) -> HipResult<NpuTrunkOutput<'a>> {
        let &(_, col0, cols) = self
            .parts
            .iter()
            .find(|&&(f, _, _)| f == family)
            .ok_or_else(|| HipError::new(1, "npu trunk: destination family not selected"))?;
        Ok(NpuTrunkOutput {
            col0: col0 as usize,
            cols: cols as usize,
            tensor,
        })
    }
    /// Geometry matching S1's panel programs, including a reusable rows_max arena.
    pub fn gpu_layout(&self, padded_rows: usize, k: usize) -> HipResult<NpuBf16Layout> {
        let layout = NpuBf16Layout {
            padded_rows,
            k,
            panels: self.panel_count(),
            n512: false,
        };
        layout.validate()?;
        Ok(layout)
    }
    /// All selected destinations must occur once, in fused order.
    pub fn validate_gpu_outputs(&self, outs: &[NpuTrunkOutput<'_>]) -> HipResult<()> {
        if outs.len() != self.parts.len()
            || outs
                .iter()
                .zip(&self.parts)
                .any(|(o, &(_, c, m))| o.col0 != c as usize || o.cols != m as usize)
        {
            return Err(HipError::new(1, "npu trunk: outputs differ from ColumnMap"));
        }
        Ok(())
    }
    pub fn consume_bf16(
        &self,
        gpu: &mut Gpu,
        c: &GpuTensor,
        layout: NpuBf16Layout,
        rows: usize,
        outs: &[NpuTrunkOutput<'_>],
    ) -> HipResult<()> {
        self.validate_gpu_outputs(outs)?;
        if layout.panels != self.panel_count() {
            return Err(HipError::new(1, "npu trunk: panel map mismatch"));
        }
        gpu.npu_trunk_consume_bf16(c, layout, outs, rows)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::npu_trunk::{FamilyMask, Shape};
    use peacemaker_npu::kernels::{
        gemm_core::{Control, Epilogue},
        gemm_g80::{bf16_output_offset, design_g80},
    };
    #[test]
    fn bf16_tile_offset_matches_native_unpack_and_is_bijective() {
        for p in [256usize, 512, 768] {
            let design = design_g80(p, 1280, 64, Epilogue::Bf16, Control::Fast);
            let mut packed = vec![0u8; p * 1280 * 2];
            let mut seen = vec![false; p * 1280];
            for t in 0..p {
                for n in 0..1280 {
                    let off = bf16_output_offset(p, t, n);
                    assert!(!seen[off / 2]);
                    seen[off / 2] = true;
                    packed[off..off + 2]
                        .copy_from_slice(&((t * 131 + n * 37) as u16).to_le_bytes());
                }
            }
            assert!(seen.into_iter().all(|b| b));
            let plain = design.unpack_out(&packed);
            for t in 0..p {
                for n in 0..1280 {
                    assert_eq!(plain[t * 1280 + n], ((t * 131 + n * 37) as u16) as i32);
                }
            }
        }
    }
    #[test]
    fn all_masks_scatter_to_native_family_columns() {
        for shape in Shape::ALL {
            for bits in 1..=255 {
                let map = shape.column_map(FamilyMask::from_bits(bits));
                if map.parts.is_empty() {
                    continue;
                }
                let layout = map.gpu_layout(4864, 2560).unwrap();
                assert_eq!(layout.padded_cols(), map.padded_cols());
                for &(family, col0, cols) in &map.parts {
                    for local in 0..cols {
                        assert_eq!(
                            map.locate((col0 + local) as usize),
                            Some((family, local as usize))
                        );
                    }
                    // Family ranges may cross panels (q, qkv, z); they have no per-family padding.
                    let first = col0 as usize;
                    let last = first + cols as usize - 1;
                    for t in [0, 127, 128, 255, 256, 4646] {
                        for col in [first, last] {
                            let offset = col / 1280 * 4864 * 1280 * 2
                                + bf16_output_offset(4864, t, col % 1280);
                            assert!(offset + 2 <= layout.c_bytes());
                        }
                    }
                }
                for n in map.useful_cols()..map.padded_cols() {
                    assert_eq!(map.locate(n), None);
                }
            }
        }
    }
    #[test]
    fn full_maps_have_frozen_panel_counts() {
        let g = Shape::GdnIn.column_map(FamilyMask::ALL);
        let q = Shape::QsaIn.column_map(FamilyMask::ALL);
        assert_eq!(
            g.parts.iter().map(|&(_, c, m)| (c, m)).collect::<Vec<_>>(),
            vec![(0, 10240), (10240, 6144), (16384, 48), (16432, 48)]
        );
        assert_eq!(
            q.parts.iter().map(|&(_, c, m)| (c, m)).collect::<Vec<_>>(),
            vec![(0, 12288), (12288, 512), (12800, 512), (13312, 640)]
        );
        assert_eq!((g.padded_cols(), q.padded_cols()), (16640, 14080));
    }
}
