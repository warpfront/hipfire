//! Hot-expert placement for the Qwen4 (Flash-Next) heterogeneous expert server.
//!
//! The primary device (gfx1201) runs the trunk and the hot routed experts; the
//! expert server (gfx1151) holds and computes the cold ones. Which experts are
//! hot is chosen once at load from a per-layer routing frequency table, built
//! here from a `HIPFIRE_QWEN4_ROUTE_TRACE` file.
//!
//! Trace format (`hipfire-arch-qwen4` `ROUTE_TRACE_ENV`): one line per
//! single-token forward, holding the position and then
//! `n_layers * top_k` routed expert ids in layer order, whitespace-separated.

/// Routing geometry of a trace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RouteShape {
    pub n_layers: usize,
    pub top_k: usize,
    pub n_experts: usize,
}

/// Routed-slot counts per layer and expert (`counts[layer][expert]`).
#[derive(Debug, Clone, PartialEq)]
pub struct ExpertFrequency {
    pub shape: RouteShape,
    pub counts: Vec<Vec<u64>>,
    pub tokens: u64,
    /// A final line cut short (the tracing process was killed mid-write).
    pub dropped_partial_lines: usize,
}

/// Parse a route trace into per-layer expert counts.
///
/// Only the final line may be short, and only when it is not newline-terminated
/// (a writer killed mid-line); it is dropped and counted. Any other malformed
/// line is an error naming its 1-based line number.
pub fn parse_route_trace(text: &str, shape: RouteShape) -> Result<ExpertFrequency, String> {
    let fields_per_line = 1 + shape.n_layers * shape.top_k;
    let mut freq = ExpertFrequency {
        shape,
        counts: vec![vec![0; shape.n_experts]; shape.n_layers],
        tokens: 0,
        dropped_partial_lines: 0,
    };
    let terminated = text.ends_with('\n');
    let lines: Vec<&str> = text.lines().collect();
    for (i, line) in lines.iter().enumerate() {
        let lineno = i + 1;
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.is_empty() {
            continue;
        }
        if fields.len() != fields_per_line {
            if lineno == lines.len() && !terminated && fields.len() < fields_per_line {
                freq.dropped_partial_lines += 1;
                continue;
            }
            return Err(format!(
                "route trace line {lineno}: {} fields, expected {fields_per_line}",
                fields.len()
            ));
        }
        for (slot, field) in fields[1..].iter().enumerate() {
            let expert: usize = field
                .parse()
                .map_err(|e| format!("route trace line {lineno}: bad expert id {field:?}: {e}"))?;
            if expert >= shape.n_experts {
                return Err(format!(
                    "route trace line {lineno}: expert {expert} >= n_experts {}",
                    shape.n_experts
                ));
            }
            freq.counts[slot / shape.top_k][expert] += 1;
        }
        freq.tokens += 1;
    }
    Ok(freq)
}

/// Per layer, the share of routed slots that land on that layer's top
/// `ceil(fraction * n_experts)` experts by count (0.0 for a layer with none).
pub fn coverage_of_top(freq: &ExpertFrequency, fraction: f64) -> Vec<f64> {
    let take = (fraction * freq.shape.n_experts as f64).ceil() as usize;
    freq.counts
        .iter()
        .map(|layer| {
            let total: u64 = layer.iter().sum();
            if total == 0 {
                return 0.0;
            }
            let mut sorted = layer.clone();
            sorted.sort_unstable_by(|a, b| b.cmp(a));
            sorted.iter().take(take).sum::<u64>() as f64 / total as f64
        })
        .collect()
}

/// Static hot/cold split of the routed experts (`hot[layer][expert]`).
#[derive(Debug, Clone, PartialEq)]
pub struct HeteroPlacement {
    pub hot: Vec<Vec<bool>>,
    pub hot_bytes: u64,
    pub cold_bytes: u64,
    /// Share of the trace's routed slots that land on hot experts.
    pub hot_traffic_share: f64,
}

/// Choose the hot experts that fit `vram_budget`.
///
/// Greedy over every `(layer, expert)` by routed count per byte, descending;
/// ties go to the lower layer, then the lower expert id. An expert that does
/// not fit is skipped and the scan continues, so smaller experts can still fill
/// the budget.
pub fn plan_hot_set(
    freq: &ExpertFrequency,
    expert_bytes: &[Vec<u64>],
    vram_budget: u64,
) -> HeteroPlacement {
    assert_eq!(expert_bytes.len(), freq.counts.len(), "expert_bytes layer count");
    let mut order = Vec::new();
    for (layer, (counts, bytes)) in freq.counts.iter().zip(expert_bytes).enumerate() {
        assert_eq!(bytes.len(), counts.len(), "expert_bytes[{layer}] expert count");
        for expert in 0..counts.len() {
            order.push((layer, expert));
        }
    }
    // count_a / bytes_a > count_b / bytes_b, compared exactly by cross-multiplication.
    order.sort_by(|&(la, ea), &(lb, eb)| {
        let lhs = freq.counts[la][ea] as u128 * expert_bytes[lb][eb] as u128;
        let rhs = freq.counts[lb][eb] as u128 * expert_bytes[la][ea] as u128;
        rhs.cmp(&lhs).then((la, ea).cmp(&(lb, eb)))
    });

    let mut hot: Vec<Vec<bool>> = freq.counts.iter().map(|c| vec![false; c.len()]).collect();
    let mut hot_bytes = 0u64;
    for (layer, expert) in order {
        let bytes = expert_bytes[layer][expert];
        if hot_bytes + bytes <= vram_budget {
            hot[layer][expert] = true;
            hot_bytes += bytes;
        }
    }

    let total_bytes: u64 = expert_bytes.iter().flatten().sum();
    let mut hot_count = 0u64;
    let mut all_count = 0u64;
    for (layer, counts) in freq.counts.iter().enumerate() {
        for (expert, &c) in counts.iter().enumerate() {
            all_count += c;
            if hot[layer][expert] {
                hot_count += c;
            }
        }
    }
    HeteroPlacement {
        hot,
        hot_bytes,
        cold_bytes: total_bytes - hot_bytes,
        hot_traffic_share: if all_count == 0 { 0.0 } else { hot_count as f64 / all_count as f64 },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const S: RouteShape = RouteShape { n_layers: 2, top_k: 2, n_experts: 4 };

    #[test]
    fn parses_counts_per_layer() {
        let f = parse_route_trace("0 1 2 3 3\n1 1 0 3 2\n", S).unwrap();
        assert_eq!(f.tokens, 2);
        assert_eq!(f.counts[0], vec![1, 2, 1, 0]);
        assert_eq!(f.counts[1], vec![0, 0, 1, 3]);
    }

    #[test]
    fn drops_only_a_truncated_last_line() {
        let f = parse_route_trace("0 1 2 3 3\n1 1 0", S).unwrap();
        assert_eq!((f.tokens, f.dropped_partial_lines), (1, 1));
    }

    #[test]
    fn malformed_middle_line_is_an_error_with_line_number() {
        let e = parse_route_trace("0 1 2 3 3\n1 1 0\n2 1 2 3 3\n", S).unwrap_err();
        assert!(e.contains("line 2"), "{e}");
    }

    #[test]
    fn out_of_range_expert_is_an_error() {
        assert!(parse_route_trace("0 1 2 3 9\n", S).unwrap_err().contains("expert 9"));
    }

    fn freq(counts: Vec<Vec<u64>>) -> ExpertFrequency {
        let n_layers = counts.len();
        let n_experts = counts[0].len();
        ExpertFrequency {
            shape: RouteShape { n_layers, top_k: 1, n_experts },
            tokens: 1,
            dropped_partial_lines: 0,
            counts,
        }
    }

    #[test]
    fn takes_highest_traffic_per_byte_within_budget() {
        let p = plan_hot_set(&freq(vec![vec![10, 1, 5]]), &[vec![2, 2, 2]], 4);
        assert_eq!(p.hot, vec![vec![true, false, true]]);
        assert_eq!((p.hot_bytes, p.cold_bytes), (4, 2));
        assert!((p.hot_traffic_share - 15.0 / 16.0).abs() < 1e-12);
    }

    #[test]
    fn zero_or_tiny_budget_is_all_cold() {
        for b in [0, 1] {
            assert!(plan_hot_set(&freq(vec![vec![3, 4]]), &[vec![2, 2]], b).hot[0].iter().all(|h| !h));
        }
    }

    #[test]
    fn budget_above_total_is_all_hot() {
        assert!(plan_hot_set(&freq(vec![vec![3, 0]]), &[vec![2, 2]], 100).hot[0].iter().all(|h| *h));
    }

    #[test]
    fn keeps_scanning_after_a_miss() {
        // ratios: e0 = 9/5, e1 = 3/2, e2 = 1/1. Budget 6: e0 (5) fits, e1 (2) misses, e2 (1) still fits.
        let p = plan_hot_set(&freq(vec![vec![9, 3, 1]]), &[vec![5, 2, 1]], 6);
        assert_eq!(p.hot, vec![vec![true, false, true]]);
    }

    #[test]
    fn ties_prefer_lower_layer_then_lower_expert() {
        let p = plan_hot_set(&freq(vec![vec![1, 1], vec![1, 1]]), &[vec![1, 1], vec![1, 1]], 1);
        assert_eq!(p.hot, vec![vec![true, false], vec![false, false]]);
    }

    #[test]
    fn coverage_of_top_half() {
        let f = parse_route_trace("0 1 2 3 3\n1 1 0 3 2\n", S).unwrap();
        // layer 0 counts [1,2,1,0]: top 2 = 2+1 = 3 of 4; layer 1 [0,0,1,3]: top 2 = 4 of 4
        assert_eq!(coverage_of_top(&f, 0.5), vec![0.75, 1.0]);
    }
}
