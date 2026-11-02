/// Replay of the `QWEN4_FLOOR_CALL` lines of recorded serve logs
/// (`HIPFIRE_MTP_FLOOR_TRACE=2`): every `ret`, `retired_after` and
/// `needs_ar_after` must come back bit for bit.  Usage:
/// `HIPFIRE_FLOOR_CALL_LOGS=a/serve.log,b/serve.log cargo test -p
/// hipfire-arch-qwen4 floor_call_log_replays_bit_identically -- --ignored`.
#[test]
#[ignore = "replays the serve logs named by HIPFIRE_FLOOR_CALL_LOGS"]
fn floor_call_log_replays_bit_identically() {
    let paths = std::env::var("HIPFIRE_FLOOR_CALL_LOGS")
        .expect("HIPFIRE_FLOOR_CALL_LOGS: comma-separated serve.log paths");
    let mut calls = 0;
    let mut mismatches = Vec::new();
    for path in paths
        .split(',')
        .map(str::trim)
        .filter(|path| !path.is_empty())
    {
        let bytes = std::fs::read(path).unwrap_or_else(|error| panic!("{path}: {error}"));
        let log = String::from_utf8_lossy(&bytes);
        let report = replay_qwen4_floor_log(log.lines());
        eprintln!(
            "{path}: {} floors, {} calls replayed, {} mismatches, {} orphan calls",
            report.floors.len(),
            report.calls(),
            report.mismatches(),
            report.orphan_calls
        );
        assert_eq!(report.malformed, 0, "{path}: {:?}", report.first_malformed);
        calls += report.calls();
        for floor in &report.floors {
            if let Some(first) = &floor.first_mismatch {
                mismatches.push(format!(
                    "{path}: floor {} ({}): {} mismatches, first {first}",
                    floor.floor, floor.owner, floor.mismatches
                ));
            }
        }
    }
    assert!(mismatches.is_empty(), "{}", mismatches.join("\n"));
    assert!(calls > 0, "no replayable QWEN4_FLOOR_CALL calls in {paths}");
}
