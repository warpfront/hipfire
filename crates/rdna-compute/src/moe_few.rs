// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Few-row Qwen4 MoE: compact inverse gate/up and exact K640 down.
//! Both retain the original F32 per-output arithmetic.

use crate::dispatch::{DType, Gpu, GpuTensor};
use hip_bridge::{HipError, HipResult, KernargBlob};
use std::ffi::c_void;

const MODULE: &str = "gemm_mq4g256v2_moe_grouped_top10_o4_r8_x4";
const ENTRY: &str = "gemm_mq4g256v2_moe_grouped_top10_o4_r2_x4_inverse_k2560";

impl Gpu {
    /// Exact gfx1151 M=1280/K=2560/top-10 F32 gate/up for 1..=8 source
    /// rows. `inverse_perm[flat_slot]` is the current scatter's grouped row;
    /// a nonnegative even row leads its pair. Pad16 expert boundaries ensure
    /// every live pair has exactly one such slot. Negative and odd rows exit.
    ///
    /// The original R2 kernel remains the arithmetic anchor. Both leave empty
    /// pairs untouched, and both write a last live pair's padded second row.
    /// Constant K folds group/stride control without changing the F32 math.
    /// No additional allocation, grouping pass or synchronization is needed.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_mq4g256v2_moe_grouped_top10_inverse(
        &mut self,
        expert_weight_ptrs: &GpuTensor,
        expert_tile_ids: &GpuTensor,
        sorted_slot_index: &GpuTensor,
        inverse_perm: &GpuTensor,
        x_src: &GpuTensor,
        y_grouped: &GpuTensor,
        m: usize,
        k: usize,
        x_row_div: usize,
        grouped_rows: usize,
        x_src_rows: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if !self.arch_caps.is_gfx1151()
            || m != 1280
            || k != 2560
            || x_row_div != 10
            || !(1..=8).contains(&x_src_rows)
            || grouped_rows % 16 != 0
            || grouped_rows > i32::MAX as usize / (m * 4)
            || x_src.dtype != DType::F32
            || y_grouped.dtype != DType::F32
        {
            return Err(HipError::new(
                0,
                "grouped top10 inverse: unsupported geometry",
            ));
        }
        let slots = x_src_rows * x_row_div;
        if grouped_rows < slots || expert_weight_ptrs.buf.size() < 8 {
            return Err(HipError::new(
                0,
                "grouped top10 inverse: empty expert table or short grouping",
            ));
        }
        for (tensor, extent, label) in [
            (inverse_perm, slots * 4, "inverse permutation"),
            (expert_tile_ids, grouped_rows / 16 * 4, "tile ids"),
            (sorted_slot_index, grouped_rows * 4, "sorted slots"),
            (x_src, x_src_rows * k * 4, "input"),
            (y_grouped, grouped_rows * m * 4, "output"),
        ] {
            if tensor.buf.size() < extent {
                return Err(HipError::new(
                    0,
                    &format!("grouped top10 inverse: {label} buffer too small"),
                ));
            }
        }
        self.ensure_kernel(
            MODULE,
            crate::kernels::GEMM_MQ4G256V2_MOE_GROUPED_TOP10_O4_R8_X4_GFX1151_SRC,
            ENTRY,
        )?;
        let pointers = [
            expert_weight_ptrs.buf.as_ptr(),
            expert_tile_ids.buf.as_ptr(),
            sorted_slot_index.buf.as_ptr(),
            inverse_perm.buf.as_ptr(),
            x_src.buf.as_ptr(),
            y_grouped.buf.as_ptr(),
        ];
        let dims = [
            m as i32,
            k as i32,
            x_row_div as i32,
            grouped_rows as i32,
            crate::gemm::moe_expert_table_len(expert_weight_ptrs),
            slots as i32,
        ];
        let mut params: Vec<*mut c_void> = pointers
            .iter()
            .map(|p| p as *const _ as *mut c_void)
            .collect();
        params.extend(dims.iter().map(|n| n as *const _ as *mut c_void));
        let bytes =
            grouped_rows.saturating_mul(m.saturating_mul(4).saturating_add(k.saturating_mul(4)));
        let timer = crate::profile::begin_timer(&self.hip, "gemm", ENTRY, bytes);
        let result = self.launch_maybe_blob(
            ENTRY,
            [m.div_ceil(16) as u32, slots as u32, 1],
            [128, 1, 1],
            0,
            &mut params,
            || {
                let mut args = KernargBlob::new();
                for p in pointers {
                    args.push_ptr(p);
                }
                for n in dims {
                    args.push_i32(n);
                }
                args
            },
        );
        if let Some(timer) = timer {
            timer.finish(&self.hip);
        }
        result
    }
    /// Exact gfx1151 2560x640 QT53 down for 1..=8 source rows. Constant K
    /// removes dynamic loop/tail control; original arithmetic and ABI remain.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_mq4g128v2_moe_down_top10_k640(
        &mut self,
        expert_ptrs: &GpuTensor,
        topk_indices: &GpuTensor,
        hidden_batch: &GpuTensor,
        expert_outputs: &GpuTensor,
        m: usize,
        k: usize,
        rows: usize,
        n_exp: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if !self.arch_caps.is_gfx1151()
            || m != 2560
            || k != 640
            || !(1..=8).contains(&rows)
            || n_exp == 0
            || n_exp > i32::MAX as usize
            || hidden_batch.dtype != DType::F32
            || expert_outputs.dtype != DType::F32
            || expert_ptrs.buf.size() / 8 < n_exp
            || topk_indices.buf.size() / 4 < rows * 10
            || hidden_batch.buf.size() / 4 < rows * 10 * k
            || expert_outputs.buf.size() / 4 < rows * 10 * m
        {
            return Err(HipError::new(
                0,
                "QT53 K640 down: unsupported geometry or short buffer",
            ));
        }
        const FUNC: &str = "gemv_mq4g128v2_moe_down_top10_k640";
        self.ensure_kernel(
            FUNC,
            crate::kernels::GEMV_MQ4G128V2_MOE_DOWN_TOP10_K640_SRC,
            FUNC,
        )?;
        let pointers = [
            expert_ptrs.buf.as_ptr(),
            topk_indices.buf.as_ptr(),
            hidden_batch.buf.as_ptr(),
            expert_outputs.buf.as_ptr(),
        ];
        let dims = [m as i32, k as i32, rows as i32, n_exp as i32];
        let mut params: Vec<*mut c_void> = pointers
            .iter()
            .map(|p| p as *const _ as *mut c_void)
            .collect();
        params.extend(dims.iter().map(|n| n as *const _ as *mut c_void));
        let bytes = rows * 10 * (m * 5 * 68 + k * 4 + m * 4);
        let timer = crate::profile::begin_timer(&self.hip, "gemv", FUNC, bytes);
        let result = self.launch_maybe_blob(
            FUNC,
            [m as u32, 10, rows as u32],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut args = KernargBlob::new();
                for p in pointers {
                    args.push_ptr(p);
                }
                for n in dims {
                    args.push_i32(n);
                }
                args
            },
        );
        if let Some(timer) = timer {
            timer.finish(&self.hip);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ints(gpu: &mut Gpu, data: &[i32]) -> GpuTensor {
        let raw: Vec<_> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
        gpu.upload_raw(&raw, &[raw.len()]).unwrap()
    }

    /// Independent original launch vs the production inverse launcher, with
    /// poisoned padding/guards, real nonzero weights, invalid routes, a missing
    /// leader negative control, and capture/replay on the actual MTP widths.
    #[test]
    fn inverse_gate_is_original_bytes_and_captures() {
        let Ok(mut gpu) = Gpu::init() else {
            eprintln!("skip: no GPU");
            return;
        };
        if !gpu.arch_caps.is_gfx1151() {
            eprintln!("skip: needs gfx1151");
            return;
        }
        let (m, k, experts, distinct, guard) = (1280, 2560, 512, 16, 1024);
        let stride = m * (k / 256) * 136;
        let mut bytes = vec![0u8; distinct * stride];
        let mut seed = 0x12345678u32;
        for b in &mut bytes {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            *b = seed as u8;
        }
        // Finite asymmetric half headers: the F32 QT44 arithmetic has no
        // symmetric-IU4 assumption. Payloads stay independently randomized.
        for group in bytes.chunks_exact_mut(136) {
            group[..8].copy_from_slice(&[0x00, 0x24, 0x00, 0xb0, 0x00, 0xa0, 0x00, 0x2c]);
        }
        let weights = gpu.upload_raw(&bytes, &[bytes.len()]).unwrap();
        let raw: Vec<_> = (0..experts)
            .flat_map(|e| {
                (weights.buf.as_ptr() as u64 + ((e % distinct) * stride) as u64).to_le_bytes()
            })
            .collect();
        let ptrs = gpu.upload_raw(&raw, &[raw.len()]).unwrap();
        let poison = f32::from_bits(0xa5a5a5a5);
        gpu.ensure_capture_stream().unwrap();
        for rows in [1, 2, 3, 4, 8] {
            let slots = rows * 10;
            let grouped = slots * 16;
            for pattern in 0..4 {
                let mut ids = vec![0; slots];
                let mut counts = vec![0; experts];
                for (i, e) in ids.iter_mut().enumerate() {
                    *e = match pattern {
                        0 => i as i32,
                        1 => (i % 10) as i32,
                        _ if i % 2 == 0 => (10 + i) as i32,
                        _ => (i % 10) as i32,
                    };
                    if pattern == 3 && i % 7 == 0 {
                        *e = -1;
                    }
                    if *e >= 0 {
                        counts[*e as usize] += 1;
                    }
                }
                let mut offsets = vec![0usize; experts + 1];
                for e in 0..experts {
                    offsets[e + 1] = offsets[e] + (counts[e] + 15) / 16 * 16;
                }
                let mut next = offsets.clone();
                let mut sorted = vec![-1; grouped];
                let mut inverse = vec![-1; slots];
                let mut tiles = vec![-1; grouped / 16];
                for (i, &e) in ids.iter().enumerate() {
                    if e >= 0 {
                        let pos = next[e as usize];
                        next[e as usize] += 1;
                        sorted[pos] = i as i32;
                        inverse[i] = pos as i32;
                    }
                }
                for e in 0..experts {
                    tiles[offsets[e] / 16..offsets[e + 1] / 16].fill(e as i32);
                }
                let sorted = ints(&mut gpu, &sorted);
                let inv = ints(&mut gpu, &inverse);
                let tiles = ints(&mut gpu, &tiles);
                let values: Vec<_> = (0..rows * k)
                    .map(|i| ((i * 37 % 1021) as f32 - 510.0) / 1024.0)
                    .collect();
                let x = gpu.upload_f32(&values, &[rows * k]).unwrap();
                let len = grouped * m + 2 * guard;
                let ref_all = gpu.upload_f32(&vec![poison; len], &[len]).unwrap();
                let out_all = gpu.upload_f32(&vec![poison; len], &[len]).unwrap();
                let reference = ref_all.sub_offset(guard, grouped * m);
                let out = out_all.sub_offset(guard, grouped * m);
                gpu.gemm_mq4g256v2_moe_grouped_top10(
                    &ptrs, &tiles, &sorted, &x, &reference, m, k, 10, grouped, rows,
                )
                .unwrap();
                gpu.gemm_mq4g256v2_moe_grouped_top10_inverse(
                    &ptrs, &tiles, &sorted, &inv, &x, &out, m, k, 10, grouped, rows,
                )
                .unwrap();
                assert_eq!(gpu.last_launched_kernel(), Some(super::ENTRY));
                eprintln!(
                    "inverse entry={} rows={rows} pattern={pattern}",
                    super::ENTRY
                );
                let want = gpu.download_raw_bytes(&ref_all).unwrap();
                let got = gpu.download_raw_bytes(&out_all).unwrap();
                assert_eq!(want, got, "eager rows={rows} pattern={pattern}");
                assert!(want[guard * 4..(guard + grouped * m) * 4]
                    .chunks_exact(4)
                    .any(|v| v != poison.to_le_bytes() && v != [0, 0, 0, 0]));
                assert!(got[..guard * 4]
                    .chunks_exact(4)
                    .all(|v| v == poison.to_le_bytes()));
                assert!(got[(guard + grouped * m) * 4..]
                    .chunks_exact(4)
                    .all(|v| v == poison.to_le_bytes()));
                if (2..=4).contains(&rows) && pattern == 1 {
                    let cap_all = gpu.upload_f32(&vec![poison; len], &[len]).unwrap();
                    let cap = cap_all.sub_offset(guard, grouped * m);
                    gpu.begin_stream_capture().unwrap();
                    gpu.gemm_mq4g256v2_moe_grouped_top10_inverse(
                        &ptrs, &tiles, &sorted, &inv, &x, &cap, m, k, 10, grouped, rows,
                    )
                    .unwrap();
                    let graph = gpu.end_stream_capture().unwrap();
                    let exec = gpu.hip.graph_instantiate(&graph).unwrap();
                    gpu.launch_graph(&exec).unwrap();
                    assert_eq!(
                        want,
                        gpu.download_raw_bytes(&cap_all).unwrap(),
                        "capture rows={rows}"
                    );
                    gpu.hip.graph_exec_destroy(exec).unwrap();
                    gpu.hip.graph_destroy(graph).unwrap();
                }
                if rows == 2 && pattern == 1 {
                    let mut bad = inverse.clone();
                    bad[0] = -1;
                    let bad = ints(&mut gpu, &bad);
                    let bad_all = gpu.upload_f32(&vec![poison; len], &[len]).unwrap();
                    let bad_y = bad_all.sub_offset(guard, grouped * m);
                    gpu.gemm_mq4g256v2_moe_grouped_top10_inverse(
                        &ptrs, &tiles, &sorted, &bad, &x, &bad_y, m, k, 10, grouped, rows,
                    )
                    .unwrap();
                    assert_ne!(
                        want,
                        gpu.download_raw_bytes(&bad_all).unwrap(),
                        "missing leader must be detected"
                    );
                }
            }
        }
    }

    /// Dynamic-K entry vs exact production entry, asymmetric headers, poisoned
    /// full output and guards, invalid experts, capture/replay, negative control.
    #[test]
    fn k640_down_is_original_bytes_and_captures() {
        let Ok(mut gpu) = Gpu::init() else {
            eprintln!("skip: no GPU");
            return;
        };
        if !gpu.arch_caps.is_gfx1151() {
            eprintln!("skip: needs gfx1151");
            return;
        }
        let (m, k, experts, distinct, guard) = (2560, 640, 512, 16, 1024);
        let stride = m * 5 * 68;
        let mut bytes = vec![0u8; distinct * stride];
        let mut seed = 0x56781234u32;
        for b in &mut bytes {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            *b = seed as u8;
        }
        for (i, group) in bytes.chunks_exact_mut(68).enumerate() {
            group[..4].copy_from_slice(if i % 2 == 0 {
                &[0x00, 0x24, 0x00, 0xb0]
            } else {
                &[0x00, 0xa0, 0x00, 0x2c]
            });
        }
        let weights = gpu.upload_raw(&bytes, &[bytes.len()]).unwrap();
        let raw: Vec<_> = (0..experts)
            .flat_map(|e| {
                (weights.buf.as_ptr() as u64 + ((e % distinct) * stride) as u64).to_le_bytes()
            })
            .collect();
        let ptrs = gpu.upload_raw(&raw, &[raw.len()]).unwrap();
        let poison = f32::from_bits(0xa5a5a5a5);
        gpu.ensure_capture_stream().unwrap();
        for rows in [1, 2, 3, 4, 8] {
            let slots = rows * 10;
            for pattern in 0..4 {
                let ids: Vec<_> = (0..slots)
                    .map(|i| {
                        if pattern == 3 && i % 7 == 0 {
                            if i % 2 == 0 {
                                -1
                            } else {
                                experts as i32 + 3
                            }
                        } else {
                            match pattern {
                                0 => i as i32,
                                1 => (i % 10) as i32,
                                _ => ((i * 13 + 7) % experts) as i32,
                            }
                        }
                    })
                    .collect();
                let indices = ints(&mut gpu, &ids);
                let values: Vec<_> = (0..slots * k)
                    .map(|i| ((i * 37 % 1021) as f32 - 510.0) / 1024.0)
                    .collect();
                let x = gpu.upload_f32(&values, &[slots * k]).unwrap();
                let len = slots * m + 2 * guard;
                let ref_all = gpu.upload_f32(&vec![poison; len], &[len]).unwrap();
                let out_all = gpu.upload_f32(&vec![poison; len], &[len]).unwrap();
                let reference = ref_all.sub_offset(guard, slots * m);
                let out = out_all.sub_offset(guard, slots * m);
                gpu.gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded(
                    &ptrs, &indices, &x, &reference, m, k, rows, experts, 10,
                )
                .unwrap();
                gpu.gemv_mq4g128v2_moe_down_top10_k640(
                    &ptrs, &indices, &x, &out, m, k, rows, experts,
                )
                .unwrap();
                let want = gpu.download_raw_bytes(&ref_all).unwrap();
                let got = gpu.download_raw_bytes(&out_all).unwrap();
                assert_eq!(want, got, "eager rows={rows} pattern={pattern}");
                assert!(got[guard * 4..(guard + slots * m) * 4]
                    .chunks_exact(4)
                    .all(|v| f32::from_le_bytes(v.try_into().unwrap()).is_finite()));
                assert!(got[guard * 4..(guard + slots * m) * 4]
                    .chunks_exact(4)
                    .any(|v| v != [0, 0, 0, 0]));
                assert!(got[..guard * 4]
                    .chunks_exact(4)
                    .all(|v| v == poison.to_le_bytes()));
                assert!(got[(guard + slots * m) * 4..]
                    .chunks_exact(4)
                    .all(|v| v == poison.to_le_bytes()));
                if pattern == 1 {
                    let cap_all = gpu.upload_f32(&vec![poison; len], &[len]).unwrap();
                    let cap = cap_all.sub_offset(guard, slots * m);
                    gpu.begin_stream_capture().unwrap();
                    gpu.gemv_mq4g128v2_moe_down_top10_k640(
                        &ptrs, &indices, &x, &cap, m, k, rows, experts,
                    )
                    .unwrap();
                    let graph = gpu.end_stream_capture().unwrap();
                    let exec = gpu.hip.graph_instantiate(&graph).unwrap();
                    gpu.launch_graph(&exec).unwrap();
                    assert_eq!(
                        want,
                        gpu.download_raw_bytes(&cap_all).unwrap(),
                        "capture rows={rows}"
                    );
                    gpu.hip.graph_exec_destroy(exec).unwrap();
                    gpu.hip.graph_destroy(graph).unwrap();
                    let bad = ints(&mut gpu, &vec![-1; slots]);
                    gpu.gemv_mq4g128v2_moe_down_top10_k640(
                        &ptrs, &bad, &x, &out, m, k, rows, experts,
                    )
                    .unwrap();
                    assert_ne!(
                        want,
                        gpu.download_raw_bytes(&out_all).unwrap(),
                        "negative control rows={rows}"
                    );
                }
                assert!(gpu
                    .gemv_mq4g128v2_moe_down_top10_k640(
                        &ptrs, &indices, &x, &out, m, 512, rows, experts
                    )
                    .is_err());
            }
        }
    }
}
