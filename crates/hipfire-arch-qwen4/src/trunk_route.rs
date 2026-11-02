// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Per-rank handle of the Flash-Next trunk-shard route (`HIPFIRE_QWEN4_EP_TRUNK_SHARD`):
//! the fused TP2 all-reduce + HC write ([`ArHcExchange`]) a rank's forward hands to its
//! [`HyperWriteOp`](hipfire_dispatch::pipeline::HyperWriteOp)s.
//!
//! Every EP rank runs its forward on its own thread (rank 0 on the caller's, a peer on its
//! pool worker), so the route is a thread-local installed when that rank attaches its EP
//! tables ([`install`]) and read where the forward builds its HC writes. It is leaked once
//! per load (a few bytes): the forward borrows it as `&'static dyn HyperWriteAr`.

use hipfire_dispatch::pipeline::HyperWriteAr;
use hipfire_runtime::ar_hc_exchange::ArHcExchange;
use rdna_compute::{Gpu, GpuTensor};
use std::cell::Cell;
use std::sync::Arc;

/// One rank's route.
pub(crate) struct TrunkShardRoute {
    exchange: Arc<ArHcExchange>,
    rank: usize,
    /// The mixer (GDN/QSA out-projection) writes carry the all-reduce.
    attn: bool,
    /// The FFN writes carry it (the sealed MoE then yields this rank's partial: no row
    /// gather).
    ffn: bool,
}

impl TrunkShardRoute {
    pub(crate) fn new(exchange: Arc<ArHcExchange>, rank: usize, attn: bool, ffn: bool) -> Self {
        Self {
            exchange,
            rank,
            attn,
            ffn,
        }
    }
}

impl HyperWriteAr for TrunkShardRoute {
    fn ar_hyper_write(
        &self,
        gpu: &mut Gpu,
        partial: &GpuTensor,
        streams: &GpuTensor,
        gates: &GpuTensor,
        rows: usize,
    ) -> Result<(), String> {
        // SAFETY: the tensors are live device buffers of this rank's forward with the
        // `[rows, 2560]` f32 partial, `[rows, 4, 2560]` bf16-bit streams and `[rows, 4]` f32
        // gate extents (`execute_hyper_write_inner` validated the op).
        unsafe {
            self.exchange.write(
                self.rank,
                gpu,
                partial.buf.as_ptr(),
                streams.buf.as_ptr(),
                gates.buf.as_ptr(),
                rows,
                std::ptr::null_mut(),
            )
        }
        .map_err(|error| format!("rank {}: {error}", self.rank))
    }
}

thread_local! {
    static ROUTE: Cell<Option<&'static TrunkShardRoute>> = const { Cell::new(None) };
}

/// Install (or with `None` clear) this thread's route.
pub(crate) fn install(route: Option<TrunkShardRoute>) {
    let leaked = route.map(|route| &*Box::leak(Box::new(route)));
    ROUTE.with(|cell| cell.set(leaked));
}

/// The writer of the mixer HC writes, when this thread's route carries them.
pub(crate) fn attn_writer() -> Option<&'static dyn HyperWriteAr> {
    ROUTE
        .with(Cell::get)
        .filter(|r| r.attn)
        .map(|r| r as &dyn HyperWriteAr)
}

/// The writer of the FFN HC writes, when this thread's route carries them.
pub(crate) fn ffn_writer() -> Option<&'static dyn HyperWriteAr> {
    ROUTE
        .with(Cell::get)
        .filter(|r| r.ffn)
        .map(|r| r as &dyn HyperWriteAr)
}

/// Whether this thread's FFN is the partial route (no expert-row gather).
pub(crate) fn ffn_partial_active() -> bool {
    ROUTE.with(Cell::get).is_some_and(|r| r.ffn)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn route_is_per_thread_and_clearable() {
        assert!(attn_writer().is_none() && ffn_writer().is_none() && !ffn_partial_active());
        let x = Arc::new(ArHcExchange::new(8).unwrap());
        install(Some(TrunkShardRoute::new(x.clone(), 0, false, true)));
        assert!(attn_writer().is_none());
        assert!(ffn_writer().is_some() && ffn_partial_active());
        std::thread::spawn(|| assert!(ffn_writer().is_none()))
            .join()
            .unwrap();
        install(None);
        assert!(ffn_writer().is_none() && !ffn_partial_active());
    }

    /// Host model of the FFN partial contract (the sealed decode MoE of a trunk-shard rank):
    /// `moe_down_combine_top10_batched` accumulates the bf16 chain
    /// `acc = bf16(acc + bf16(bf16(out) * bf16(w)))` over the slots in ascending expert order
    /// from a CLEARED accumulator, non-owned slots hold exactly-zero rows (their expert is the
    /// zero dummy: the down kernel's `(scale * q + zero) * x` over all-zero bytes is `+0`), and
    /// `bf16_scaled_add` folds the rank's K-half of the shared expert as
    /// `bf16(bf16(res) + bf16(bf16(s) * bf16(dot)))`; the fused write then takes
    /// `m = bf16(p0 + p1)` of the two rank partials. This pins (a) zero rows are the identity
    /// of the chain, so a rank's partial is the chain over ITS owned slots only, (b) no slot and
    /// no shared K-half is counted twice or dropped, (c) the partial-sum grouping differs from
    /// the one-card chain only at bf16 rounding level.
    fn bf16r(x: f32) -> f32 {
        let u = x.to_bits();
        f32::from_bits(((u.wrapping_add(0x7FFF + ((u >> 16) & 1))) >> 16) << 16)
    }

    /// `slots`: (expert, weight, down row element) per slot; ascending expert order, ties by
    /// slot (the kernel's strict `<` scan).
    fn combine_chain(start: f32, slots: &[(i32, f32, f32)]) -> f32 {
        let mut order: Vec<usize> = (0..slots.len()).collect();
        order.sort_by_key(|&r| (slots[r].0, r));
        let mut acc = bf16r(start);
        for r in order {
            let (_, weight, output) = slots[r];
            let product = bf16r(bf16r(output) * bf16r(weight));
            acc = bf16r(acc + product);
        }
        acc
    }

    fn shared_add(residual: f32, dot: f32, scalar: f32) -> f32 {
        bf16r(bf16r(residual) + bf16r(bf16r(scalar) * bf16r(dot)))
    }

    #[test]
    fn ffn_partial_rows_chain_and_fold_contract() {
        // The dummy expert's down row: scale = zero = 0 (f16 zero bytes), any nibbles, any finite x.
        for nibble in 0..16u32 {
            for x in [0.0f32, 1.5, -3.25e4, 1e-30] {
                assert_eq!(((0.0f32 * nibble as f32) + 0.0) * x, 0.0);
            }
        }
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut draw = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            ((seed >> 11) as f64 / (1u64 << 53) as f64) as f32
        };
        let mut worst = 0.0f64;
        for trial in 0..4000 {
            // 10 distinct routed experts; owners by `e % 2` (static) or dealt alternately over
            // the ascending routed set (balanced), both exactly one owner per expert.
            let mut experts = [0i32; 10];
            let mut used = std::collections::BTreeSet::new();
            for slot in experts.iter_mut() {
                loop {
                    let e = (draw() * 512.0) as i32;
                    if used.insert(e) {
                        *slot = e;
                        break;
                    }
                }
            }
            let mut sorted = experts.to_vec();
            sorted.sort_unstable();
            let owner = |e: i32| -> usize {
                if trial % 2 == 0 {
                    (e % 2) as usize
                } else {
                    sorted.iter().position(|&x| x == e).unwrap() % 2
                }
            };
            let weights: [f32; 10] = std::array::from_fn(|_| bf16r(draw() * 0.3));
            let outputs: [f32; 10] = std::array::from_fn(|_| (draw() - 0.5) * 8.0);
            let selector = bf16r(1.0 / (1.0 + (-(draw() - 0.5) * 6.0).exp()));
            let (d0, d1) = ((draw() - 0.5) * 6.0, (draw() - 0.5) * 6.0);
            // One card (EP2 / reference): chain over all slots, then the full shared down
            // (the K-halves accumulated in f32, one rounding on the way out of the GEMV).
            let all_slots: Vec<(i32, f32, f32)> = (0..10)
                .map(|r| (experts[r], weights[r], outputs[r]))
                .collect();
            let one_card = shared_add(combine_chain(0.0, &all_slots), d0 + d1, selector);
            // TP2 ranks: the kernel keeps every slot, a non-owned one with an exactly-zero row.
            let mut partial = [0.0f32; 2];
            for rank in 0..2 {
                let with_zero_rows: Vec<(i32, f32, f32)> = (0..10)
                    .map(|r| {
                        let row = if owner(experts[r]) == rank {
                            outputs[r]
                        } else {
                            0.0
                        };
                        (experts[r], weights[r], row)
                    })
                    .collect();
                let owned_only: Vec<(i32, f32, f32)> = (0..10)
                    .filter(|&r| owner(experts[r]) == rank)
                    .map(|r| (experts[r], weights[r], outputs[r]))
                    .collect();
                let zero_chain = combine_chain(0.0, &with_zero_rows);
                let skip_chain = combine_chain(0.0, &owned_only);
                assert_eq!(
                    zero_chain.to_bits(),
                    skip_chain.to_bits(),
                    "zero rows are the chain identity"
                );
                partial[rank] = shared_add(zero_chain, if rank == 0 { d0 } else { d1 }, selector);
            }
            let m = bf16r(partial[0] + partial[1]);
            // Bound: the two groupings differ by a handful of bf16 roundings of terms no
            // larger than the sum of the magnitudes involved.
            let magnitude: f64 = (0..10)
                .map(|r| (bf16r(outputs[r]) as f64 * weights[r] as f64).abs())
                .sum::<f64>()
                + (selector as f64 * (d0.abs() + d1.abs()) as f64);
            let diff = (m as f64 - one_card as f64).abs();
            worst = worst.max(diff / magnitude.max(1e-30));
            assert!(
                diff <= 32.0 * magnitude * 2f64.powi(-8),
                "trial {trial}: m {m} one-card {one_card} magnitude {magnitude}"
            );
        }
        assert!(
            worst > 0.0,
            "the two groupings must differ at rounding level somewhere"
        );
    }
}
