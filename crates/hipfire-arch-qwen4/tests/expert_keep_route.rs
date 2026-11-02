// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
use rdna_compute::{DType, Gpu};

fn bf16_round_trip(v: f32) -> f32 {
    if !v.is_finite() {
        return v;
    }
    let u = v.to_bits();
    f32::from_bits(u.wrapping_add(0x7fff + ((u >> 16) & 1)) & 0xffff0000)
}

/// Full index-order float32 softmax, then top-10 selected renormalization.
/// Input rows retain original ids in increasing order, even when sliced.
fn route(mut logits: Vec<(usize, f32)>, normalize: bool, round: bool) -> (Vec<usize>, Vec<f32>) {
    if round {
        for (_, value) in &mut logits {
            *value = bf16_round_trip(*value);
        }
    }
    let mut selected = logits.clone();
    selected.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
    selected.truncate(10);
    let max = selected[0].1;
    let denominator = logits
        .iter()
        .fold(0.0f32, |sum, (_, v)| sum + (*v - max).exp());
    let inverse = 1.0f32 / denominator;
    let mut probabilities: Vec<f32> = selected
        .iter()
        .map(|(_, v)| (*v - max).exp() * inverse)
        .collect();
    if normalize {
        let selected_sum = probabilities.iter().fold(0.0f32, |sum, &p| sum + p);
        let inverse_selected = 1.0f32 / selected_sum;
        for p in &mut probabilities {
            *p *= inverse_selected;
        }
    }
    if round {
        for p in &mut probabilities {
            *p = bf16_round_trip(*p);
        }
    }
    (selected.iter().map(|(id, _)| *id).collect(), probabilities)
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn masked_router_equals_sliced_router_original_id_ties() {
    for count in [16, 31, 128, 511, 512] {
        let keep: Vec<usize> = (0..512).filter(|e| (e * 37) % 512 < count).collect();
        for ties in [false, true] {
            let logits: Vec<f32> = (0..512)
                .map(|e| {
                    if ties {
                        0.25
                    } else {
                        ((e * 13) % 47) as f32 * 0.1273
                    }
                })
                .collect();
            for normalize in [false, true] {
                for round in [false, true] {
                    let masked = route(
                        logits
                            .iter()
                            .enumerate()
                            .map(|(e, &v)| {
                                (
                                    e,
                                    if keep.contains(&e) {
                                        v
                                    } else {
                                        f32::NEG_INFINITY
                                    },
                                )
                            })
                            .collect(),
                        normalize,
                        round,
                    );
                    let sliced = route(
                        keep.iter().map(|&e| (e, logits[e])).collect(),
                        normalize,
                        round,
                    );
                    assert_eq!(masked.0, sliced.0);
                    assert_eq!(bits(&masked.1), bits(&sliced.1));
                    if ties {
                        assert_eq!(masked.0, keep[..10]);
                    }
                }
            }
        }
    }
}

/// CPU model of the masked kernel's fault ABI: only kept logits (after the
/// optional rounding) are checked, absent ones are never read, and a failed
/// denominator or selected sum fails the row closed (`Err`).
fn checked_masked_route(
    logits: &[f32],
    keep: &[bool],
    normalize: bool,
    round: bool,
) -> Result<(Vec<usize>, Vec<f32>), ()> {
    let mut entries = Vec::with_capacity(512);
    for e in 0..512 {
        if !keep[e] {
            entries.push((e, f32::NEG_INFINITY));
            continue;
        }
        let v = if round {
            bf16_round_trip(logits[e])
        } else {
            logits[e]
        };
        if !v.is_finite() {
            return Err(());
        }
        entries.push((e, v));
    }
    let (ids, weights) = route(entries, normalize, false);
    let weights: Vec<f32> = if round {
        weights.iter().map(|&w| bf16_round_trip(w)).collect()
    } else {
        weights
    };
    if weights.iter().any(|w| !w.is_finite()) {
        return Err(());
    }
    Ok((ids, weights))
}

#[test]
fn masked_router_cpu_absent_nonfinite_ignored_kept_nonfinite_faults() {
    let keep: Vec<bool> = (0..512).map(|e| (e * 37) % 512 < 31).collect();
    let clean: Vec<f32> = (0..512).map(|e| ((e * 13) % 47) as f32 * 0.1273).collect();
    for normalize in [false, true] {
        for round in [false, true] {
            let expected = checked_masked_route(&clean, &keep, normalize, round).unwrap();
            for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let mut absent_poisoned = clean.clone();
                for e in 0..512 {
                    if !keep[e] {
                        absent_poisoned[e] = poison;
                    }
                }
                let got = checked_masked_route(&absent_poisoned, &keep, normalize, round).unwrap();
                assert_eq!(got.0, expected.0);
                assert_eq!(bits(&got.1), bits(&expected.1));
                for kept in (0..512).filter(|&e| keep[e]) {
                    let mut kept_poisoned = clean.clone();
                    kept_poisoned[kept] = poison;
                    assert!(
                        checked_masked_route(&kept_poisoned, &keep, normalize, round).is_err(),
                        "kept {kept} {poison}"
                    );
                }
            }
        }
    }
    // No kept expert: the softmax denominator is NaN and must fault, not return.
    assert!(checked_masked_route(&clean, &vec![false; 512], false, false).is_err());
}

/// The kernel must keep the dense router's fault ABI (`..., keep, fault,
/// layer, row_base`) and fail every closed path through the shared fail-row
/// helper, like the dense kernel does.
#[test]
fn masked_router_kernel_source_follows_dense_fault_abi() {
    let masked = include_str!("../../../kernels/src/moe_router_softmax_top10_masked_f32.hip");
    let dense = include_str!("../../../kernels/src/moe_router_softmax_top10_f32.hip");
    for needle in [
        "const unsigned char* __restrict__ keep,\n    unsigned long long* __restrict__ fault,",
        "    int layer, ",
        "    int row_base ",
        "__syncthreads_or((first_kept && !isfinite(first_logit)) || (second_kept && !isfinite(second_logit)))",
    ] {
        assert!(masked.contains(needle), "masked kernel missing {needle:?}");
    }
    let call = "router_fail_row(topk_idx, topk_w, token, fault, layer, row_base + token)";
    // Vote, denominator and selected-sum paths.
    assert_eq!(masked.matches(call).count(), 3);
    assert_eq!(dense.matches(call).count(), 3);
    assert!(
        !masked.contains("isfinite(denom)) return;")
            && !masked.contains("isfinite(selected_sum)) return;")
    );
}

fn poison_value(i: usize) -> f32 {
    match i % 3 {
        0 => f32::NAN,
        1 => f32::INFINITY,
        _ => f32::NEG_INFINITY,
    }
}

/// Absent logits replaced by a finite floor that underflows `expf` to exactly 0.
/// The dense router faults on +/-Inf, so this is its masked-equivalent input.
const DENSE_ABSENT_FLOOR: f32 = -1.0e30;

/// Wait, then read the fault word before the `download_*` readers fold it into
/// an error, and clear it.
fn take_fault(gpu: &Gpu) -> Option<rdna_compute::router_fault::RouterFault> {
    gpu.hip.device_synchronize().expect("sync");
    let fault = gpu.peek_router_fault();
    gpu.clear_router_fault();
    fault
}

#[test]
#[ignore = "requires gfx1201 HIP GPU; decode and grouped-prefill masked route"]
fn masked_router_decode_and_prefill_gfx1201() {
    let mut gpu = Gpu::init().expect("HIP GPU");
    assert_eq!(gpu.arch, "gfx1201", "run on the R9700");
    const LAYER: usize = 5;
    const ROW_BASE: usize = 0;
    for rows in [1, 2, 4, 17, 128] {
        let keep: Vec<u8> = (0..512).map(|e| u8::from((e * 37) % 512 < 31)).collect();
        let logits: Vec<f32> = (0..rows * 512)
            .map(|i| {
                if i / 512 % 2 == 0 {
                    ((i * 13) % 47) as f32 * 0.1273
                } else {
                    0.25
                }
            })
            .collect();
        // Masked input: absent experts hold NaN / +Inf / -Inf and must be ignored.
        let poisoned_logits: Vec<f32> = logits
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                if keep[i % 512] != 0 {
                    v
                } else {
                    poison_value(i)
                }
            })
            .collect();
        let reference_logits: Vec<f32> = logits
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                if keep[i % 512] != 0 {
                    v
                } else {
                    DENSE_ABSENT_FLOOR
                }
            })
            .collect();
        let mask = gpu.upload_raw(&keep, &[512]).unwrap();
        let input = gpu.upload_f32(&poisoned_logits, &[rows, 512]).unwrap();
        let reference_input = gpu.upload_f32(&reference_logits, &[rows, 512]).unwrap();
        let ids = gpu.alloc_tensor(&[rows, 10], DType::F32).unwrap();
        let weights = gpu.alloc_tensor(&[rows, 10], DType::F32).unwrap();
        let reference_ids = gpu.alloc_tensor(&[rows, 10], DType::F32).unwrap();
        let reference_weights = gpu.alloc_tensor(&[rows, 10], DType::F32).unwrap();
        for normalize in [false, true] {
            for round in [false, true] {
                gpu.clear_router_fault();
                gpu.moe_router_softmax_top10_masked_f32(
                    &input,
                    &ids,
                    &weights,
                    rows,
                    normalize,
                    mask.buf.as_ptr() as usize,
                    round,
                    LAYER,
                    ROW_BASE,
                )
                .unwrap();
                gpu.moe_router_softmax_top10_f32(
                    &reference_input,
                    &reference_ids,
                    &reference_weights,
                    rows,
                    normalize,
                    round,
                    LAYER,
                    ROW_BASE,
                )
                .unwrap();
                assert_eq!(
                    take_fault(&gpu),
                    None,
                    "absent NaN/Inf must not fault rows={rows} normalize={normalize} round={round}"
                );
                let got_ids = gpu.download_f32(&ids).unwrap();
                let got_weights = gpu.download_f32(&weights).unwrap();
                assert_eq!(
                    bits(&got_ids),
                    bits(&gpu.download_f32(&reference_ids).unwrap()),
                    "ids rows={rows} normalize={normalize} round={round}"
                );
                assert_eq!(
                    bits(&got_weights),
                    bits(&gpu.download_f32(&reference_weights).unwrap()),
                    "weights rows={rows} normalize={normalize} round={round}"
                );
                for row in 0..rows {
                    let expected = route(
                        (0..512)
                            .filter(|&e| keep[e] != 0)
                            .map(|e| (e, logits[row * 512 + e]))
                            .collect(),
                        normalize,
                        round,
                    );
                    assert_eq!(
                        got_ids[row * 10..(row + 1) * 10]
                            .iter()
                            .map(|v| v.to_bits() as usize)
                            .collect::<Vec<_>>(),
                        expected.0
                    );
                }
            }
        }
        for tensor in [
            mask,
            input,
            reference_input,
            ids,
            weights,
            reference_ids,
            reference_weights,
        ] {
            gpu.free_tensor(tensor).unwrap();
        }
    }
}

#[test]
#[ignore = "requires gfx1201 HIP GPU; masked route fault ABI"]
fn masked_router_faults_kept_nonfinite_like_dense_gfx1201() {
    let mut gpu = Gpu::init().expect("HIP GPU");
    assert_eq!(gpu.arch, "gfx1201", "run on the R9700");
    const LAYER: usize = 9;
    const ROW_BASE: usize = 100;
    const ROWS: usize = 6;
    const BAD_ROW: usize = 3;
    let keep: Vec<u8> = (0..512).map(|e| u8::from((e * 37) % 512 < 31)).collect();
    let kept_experts: Vec<usize> = (0..512).filter(|&e| keep[e] != 0).collect();
    assert!(kept_experts.iter().any(|&e| e < 256) && kept_experts.iter().any(|&e| e >= 256));
    let mask = gpu.upload_raw(&keep, &[512]).unwrap();
    let ids = gpu.alloc_tensor(&[ROWS, 10], DType::F32).unwrap();
    let weights = gpu.alloc_tensor(&[ROWS, 10], DType::F32).unwrap();
    let reference_ids = gpu.alloc_tensor(&[ROWS, 10], DType::F32).unwrap();
    let reference_weights = gpu.alloc_tensor(&[ROWS, 10], DType::F32).unwrap();
    // A kept expert carrying NaN / +Inf / -Inf: first, middle and last kept
    // expert, covering both 256-lane halves of the workgroup.
    let cases: Vec<(usize, f32)> = [
        kept_experts[0],
        kept_experts[kept_experts.len() / 2],
        kept_experts[kept_experts.len() - 1],
    ]
    .into_iter()
    .flat_map(|e| [f32::NAN, f32::INFINITY, f32::NEG_INFINITY].map(|v| (e, v)))
    .collect();
    for &(bad_expert, bad_value) in &cases {
        for normalize in [false, true] {
            for round in [false, true] {
                let base: Vec<f32> = (0..ROWS * 512)
                    .map(|i| ((i * 13) % 47) as f32 * 0.1273)
                    .collect();
                // Masked: absent experts poisoned everywhere, one kept logit bad in BAD_ROW.
                let mut masked_logits: Vec<f32> = base
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| {
                        if keep[i % 512] != 0 {
                            v
                        } else {
                            poison_value(i)
                        }
                    })
                    .collect();
                masked_logits[BAD_ROW * 512 + bad_expert] = bad_value;
                // Dense reference: same kept values, absent at the finite floor.
                let mut dense_logits: Vec<f32> = base
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| {
                        if keep[i % 512] != 0 {
                            v
                        } else {
                            DENSE_ABSENT_FLOOR
                        }
                    })
                    .collect();
                dense_logits[BAD_ROW * 512 + bad_expert] = bad_value;
                let input = gpu.upload_f32(&masked_logits, &[ROWS, 512]).unwrap();
                let reference_input = gpu.upload_f32(&dense_logits, &[ROWS, 512]).unwrap();
                let context = format!(
                    "expert={bad_expert} value={bad_value} normalize={normalize} round={round}"
                );

                gpu.clear_router_fault();
                gpu.moe_router_softmax_top10_f32(
                    &reference_input,
                    &reference_ids,
                    &reference_weights,
                    ROWS,
                    normalize,
                    round,
                    LAYER,
                    ROW_BASE,
                )
                .unwrap();
                let dense_fault = take_fault(&gpu);
                gpu.moe_router_softmax_top10_masked_f32(
                    &input,
                    &ids,
                    &weights,
                    ROWS,
                    normalize,
                    mask.buf.as_ptr() as usize,
                    round,
                    LAYER,
                    ROW_BASE,
                )
                .unwrap();
                let masked_fault = take_fault(&gpu);

                let expected_fault = rdna_compute::router_fault::RouterFault {
                    device_id: gpu.device_id,
                    layer: LAYER as u32,
                    row: (ROW_BASE + BAD_ROW) as u64,
                };
                assert_eq!(dense_fault, Some(expected_fault), "dense {context}");
                assert_eq!(
                    masked_fault, dense_fault,
                    "masked fault differs from dense {context}"
                );
                let got_ids = gpu.download_f32(&ids).unwrap();
                let got_weights = gpu.download_f32(&weights).unwrap();
                assert_eq!(
                    bits(&got_ids),
                    bits(&gpu.download_f32(&reference_ids).unwrap()),
                    "ids {context}"
                );
                assert_eq!(
                    bits(&got_weights),
                    bits(&gpu.download_f32(&reference_weights).unwrap()),
                    "weights {context}"
                );
                // Failed row: ids -1, weights +0; the other rows route normally.
                assert!(
                    got_ids[BAD_ROW * 10..(BAD_ROW + 1) * 10]
                        .iter()
                        .all(|v| v.to_bits() as i32 == -1),
                    "{context}"
                );
                assert!(
                    got_weights[BAD_ROW * 10..(BAD_ROW + 1) * 10]
                        .iter()
                        .all(|v| v.to_bits() == 0),
                    "{context}"
                );
                for row in (0..ROWS).filter(|&row| row != BAD_ROW) {
                    let expected = route(
                        kept_experts
                            .iter()
                            .map(|&e| (e, base[row * 512 + e]))
                            .collect(),
                        normalize,
                        round,
                    );
                    assert_eq!(
                        got_ids[row * 10..(row + 1) * 10]
                            .iter()
                            .map(|v| v.to_bits() as usize)
                            .collect::<Vec<_>>(),
                        expected.0,
                        "row {row} {context}"
                    );
                }

                gpu.free_tensor(input).unwrap();
                gpu.free_tensor(reference_input).unwrap();
            }
        }
    }

    // A row with no kept expert has a NaN denominator: it must fault and fail
    // closed, not return with the output untouched.
    let empty = gpu.upload_raw(&vec![0u8; 512], &[512]).unwrap();
    let input = gpu.upload_f32(&vec![0.5f32; 512], &[1, 512]).unwrap();
    let sentinel_ids = gpu
        .upload_f32(&vec![f32::from_bits(7); 10], &[1, 10])
        .unwrap();
    let sentinel_weights = gpu.upload_f32(&vec![1.0f32; 10], &[1, 10]).unwrap();
    gpu.clear_router_fault();
    gpu.moe_router_softmax_top10_masked_f32(
        &input,
        &sentinel_ids,
        &sentinel_weights,
        1,
        false,
        empty.buf.as_ptr() as usize,
        false,
        LAYER,
        ROW_BASE,
    )
    .unwrap();
    let fault = take_fault(&gpu);
    assert_eq!(
        fault.map(|f| (f.layer, f.row)),
        Some((LAYER as u32, ROW_BASE as u64)),
        "empty keep mask must fault"
    );
    let got_ids = gpu.download_f32(&sentinel_ids).unwrap();
    let got_weights = gpu.download_f32(&sentinel_weights).unwrap();
    assert!(
        got_ids.iter().all(|v| v.to_bits() as i32 == -1)
            && got_weights.iter().all(|v| v.to_bits() == 0),
        "empty keep mask left stale outputs"
    );
    for tensor in [
        empty,
        input,
        sentinel_ids,
        sentinel_weights,
        mask,
        ids,
        weights,
        reference_ids,
        reference_weights,
    ] {
        gpu.free_tensor(tensor).unwrap();
    }
}
