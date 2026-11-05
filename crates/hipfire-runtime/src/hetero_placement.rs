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

    #[test]
    fn coverage_of_top_half() {
        let f = parse_route_trace("0 1 2 3 3\n1 1 0 3 2\n", S).unwrap();
        // layer 0 counts [1,2,1,0]: top 2 = 2+1 = 3 of 4; layer 1 [0,0,1,3]: top 2 = 4 of 4
        assert_eq!(coverage_of_top(&f, 0.5), vec![0.75, 1.0]);
    }
}
