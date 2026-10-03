//! Host work list for persistent grouped GEMM. Zero-row experts and dead
//! panels have no entries, so consumers need not issue loads for them.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WorkItem {
    pub expert: u32,
    pub first_row: u32,
    pub real_rows: u32,
    pub executed_rows: u32,
}
/// Exactly the hardware's 16-row floor, never a whole-panel padding rule.
pub fn pad16(rows: usize) -> Result<usize, String> {
    rows.checked_add(15).map(|r| r / 16 * 16).ok_or_else(|| "row count overflow".into())
}
pub fn tail_rows(rows: usize, first_row: usize, panel_rows: usize) -> usize {
    rows.saturating_sub(first_row).min(panel_rows)
}
/// Experts run largest first (expert index breaks ties), panels in row order.
/// ABI fields are row counts, not byte offsets; `real_rows` bounds all loads.
pub fn plan(rows: &[usize], panel_rows: usize) -> Result<Vec<WorkItem>, String> {
    if panel_rows == 0 || panel_rows % 16 != 0 { return Err("panel rows must be a positive multiple of 16".into()) }
    if rows.len() > u32::MAX as usize || rows.iter().any(|&r| r > u32::MAX as usize - 15) { return Err("work list exceeds u32 ABI".into()) }
    let mut experts: Vec<_> = rows.iter().copied().enumerate().filter(|(_, r)| *r > 0).collect();
    experts.sort_unstable_by(|(a, ra), (b, rb)| rb.cmp(ra).then(a.cmp(b)));
    let mut work = Vec::new();
    for (expert, rows) in experts {
        for first_row in (0..rows).step_by(panel_rows) {
            let real_rows = tail_rows(rows, first_row, panel_rows);
            work.push(WorkItem { expert: expert as u32, first_row: first_row as u32,
                real_rows: real_rows as u32, executed_rows: pad16(real_rows)? as u32 });
        }
    }
    Ok(work)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn grouped_tails_cover_rows_once_without_dead_loads() {
        let rows = [0, 17, 65, 17, 1];
        let work = plan(&rows, 64).unwrap();
        assert_eq!(work.iter().map(|w| w.expert).collect::<Vec<_>>(), [2, 2, 1, 3, 4]);
        assert_eq!(work[1], WorkItem { expert: 2, first_row: 64, real_rows: 1, executed_rows: 16 });
        for (expert, &count) in rows.iter().enumerate() {
            let items: Vec<_> = work.iter().filter(|w| w.expert as usize == expert).collect();
            assert_eq!(items.iter().map(|w| w.real_rows as usize).sum::<usize>(), count);
            assert_eq!(items.iter().map(|w| w.executed_rows as usize).sum::<usize>(), pad16(count).unwrap());
            let mut end = 0;
            for w in items { assert_eq!(w.first_row, end); assert!(w.real_rows > 0); end += w.real_rows; }
        }
        assert!(plan(&rows, 0).is_err());
        assert!(plan(&rows, 17).is_err());
        assert_eq!(tail_rows(65, 128, 64), 0);
        assert!(pad16(usize::MAX).is_err());
    }
    #[test]
    fn captured_l4_l24_l44_routes_reach_pad16_floor() {
        // Histograms of tokens*10 little-endian i32 IDs from the prefix of
        // qcal/release-0.4.1/iu4-roofline/raw/routes/L{4,24,44}.topk.i32.
        // source_md5 identifies the full 8192-token capture; real_rows/q16
        // are independently pinned by fn-wmo/gemm-compare/real-routing.json.
        let fixtures: serde_json::Value = serde_json::from_str(include_str!("routes.json")).unwrap();
        for route in fixtures.as_array().unwrap() {
            let rows: Vec<usize> = route["rows"].as_array().unwrap().iter().map(|r| r.as_u64().unwrap() as usize).collect();
            for width in [16, 32, 64, 128, 256] {
                let work = plan(&rows, width).unwrap();
                assert_eq!(work.iter().map(|w| w.real_rows as u64).sum::<u64>(), route["real_rows"].as_u64().unwrap());
                assert_eq!(work.iter().map(|w| w.executed_rows as u64).sum::<u64>(), route["q16"].as_u64().unwrap());
                for (expert, &count) in rows.iter().enumerate() {
                    let mut end = 0;
                    for w in work.iter().filter(|w| w.expert as usize == expert) {
                        assert_eq!(w.first_row as usize, end);
                        assert_eq!(w.real_rows as usize, tail_rows(count, end, width));
                        end += w.real_rows as usize;
                    }
                    assert_eq!(end, count);
                }
                for pair in work.windows(2) {
                    assert!(rows[pair[0].expert as usize] >= rows[pair[1].expert as usize]);
                }
            }
        }
    }
}
