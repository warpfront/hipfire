// SPDX-License-Identifier: Apache-2.0
//! Execution order of one chunk pair (two micro-batches, banks A = 0 and B = 1) through a co-processor stage.
//!
//! Each (layer, bank) visit is split around one co-processor chain:
//! - [`PairOp::Prepare`]: GPU work up to the layer's owned projections (pre-mixer, norm), the activation pack
//!   into the bank's A arena (GPU -> NPU transfer), then the chain kick;
//! - [`PairOp::Run`]: the GPU waits for the chain, runs the mixer core on its C arena (NPU -> GPU transfer),
//!   retires the A arena, then runs out-projection, residual and the rest of the layer.
//!
//! The order is layer-major: while the GPU runs one bank's layer, the co-processor computes the other bank's
//! projections. Bank B's own initialization (embedding) is queued after bank A's first kick, so the first chain
//! (the pipeline fill) does not wait behind it. [`pair_edges`] lists every dependency the order must respect.

/// One step of the pair schedule.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PairOp {
    /// Initialize a bank's rows (bank B only; bank A is initialized before the pair starts).
    Init {
        bank: usize,
    },
    Prepare {
        layer: usize,
        bank: usize,
    },
    Run {
        layer: usize,
        bank: usize,
    },
}

/// Why one [`PairOp`] must follow another.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Edge {
    /// The chain kicked by `Prepare(l, b)` produces the projections `Run(l, b)` consumes (A arena, C arena).
    Activation,
    /// A bank's residual rows leave `Run(l, b)` and enter `Prepare(l + 1, b)`.
    Residual,
    /// `Prepare(l + 1, b)` repacks bank `b`'s A arena only after `Run(l, b)` retired the chain that read it.
    ArenaReuse,
    /// Layer state (recurrent state, KV cache) updated by bank A's rows precedes bank B's rows of the same layer.
    State,
    /// Bank B's initialized rows enter its first prepare.
    Init,
}

/// The pair schedule of a `layers`-layer stack.
pub fn pair_schedule(layers: usize) -> impl Iterator<Item = PairOp> {
    let fill = [
        PairOp::Prepare { layer: 0, bank: 0 },
        PairOp::Init { bank: 1 },
        PairOp::Prepare { layer: 0, bank: 1 },
    ];
    let steady = (0..layers).flat_map(move |layer| {
        [0, 1].into_iter().flat_map(move |bank| {
            let next = (layer + 1 < layers).then_some(PairOp::Prepare {
                layer: layer + 1,
                bank,
            });
            std::iter::once(PairOp::Run { layer, bank }).chain(next)
        })
    });
    fill.into_iter().filter(move |_| layers != 0).chain(steady)
}

/// Every dependency `(before, after, why)` of a `layers`-layer pair.
pub fn pair_edges(layers: usize) -> Vec<(PairOp, PairOp, Edge)> {
    let mut edges = Vec::new();
    if layers != 0 {
        edges.push((
            PairOp::Init { bank: 1 },
            PairOp::Prepare { layer: 0, bank: 1 },
            Edge::Init,
        ));
    }
    for layer in 0..layers {
        for bank in 0..2 {
            let run = PairOp::Run { layer, bank };
            edges.push((PairOp::Prepare { layer, bank }, run, Edge::Activation));
            if layer + 1 < layers {
                let next = PairOp::Prepare {
                    layer: layer + 1,
                    bank,
                };
                edges.push((run, next, Edge::Residual));
                edges.push((run, next, Edge::ArenaReuse));
            }
        }
        edges.push((
            PairOp::Run { layer, bank: 0 },
            PairOp::Run { layer, bank: 1 },
            Edge::State,
        ));
    }
    edges
}

#[cfg(test)]
mod tests {
    use super::*;

    fn position(order: &[PairOp], op: PairOp) -> usize {
        order.iter().position(|o| *o == op).expect("op scheduled")
    }

    #[test]
    fn schedule_is_a_topological_order_of_every_edge() {
        for layers in [0, 1, 2, 3, 64] {
            let order: Vec<_> = pair_schedule(layers).collect();
            let expected = if layers == 0 { 0 } else { 1 + 4 * layers };
            assert_eq!(order.len(), expected, "every op once at {layers} layers");
            for (i, op) in order.iter().enumerate() {
                assert!(!order[..i].contains(op), "{op:?} scheduled twice");
            }
            for (before, after, why) in pair_edges(layers) {
                assert!(
                    position(&order, before) < position(&order, after),
                    "{why:?}: {before:?} must precede {after:?}"
                );
            }
        }
    }

    #[test]
    fn the_coprocessor_works_on_one_bank_while_the_gpu_runs_the_other() {
        let order: Vec<_> = pair_schedule(4).collect();
        // Fill: bank A's first chain is kicked before bank B is even initialized.
        assert_eq!(order[0], PairOp::Prepare { layer: 0, bank: 0 });
        assert_eq!(order[1], PairOp::Init { bank: 1 });
        for layer in 0..3 {
            // A's next chain is in flight while the GPU runs B's layer, and B's while it runs A's next layer.
            assert!(
                position(
                    &order,
                    PairOp::Prepare {
                        layer: layer + 1,
                        bank: 0
                    }
                ) < position(&order, PairOp::Run { layer, bank: 1 })
            );
            assert!(
                position(
                    &order,
                    PairOp::Prepare {
                        layer: layer + 1,
                        bank: 1
                    }
                ) < position(
                    &order,
                    PairOp::Run {
                        layer: layer + 1,
                        bank: 0
                    }
                )
            );
        }
    }
}
