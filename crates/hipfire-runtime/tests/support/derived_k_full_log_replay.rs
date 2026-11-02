/// Full-log oracle: `HIPFIRE_DERIVED_K_ORACLE_FIXTURE=<spec-k>/phase3-oracle/full-fn-v1.json`.
#[test]
#[ignore = "needs HIPFIRE_DERIVED_K_ORACLE_FIXTURE (the full FN oracle export)"]
fn derived_k_replays_replay2_full_logs() {
    let path = std::env::var("HIPFIRE_DERIVED_K_ORACLE_FIXTURE")
        .expect("HIPFIRE_DERIVED_K_ORACLE_FIXTURE must name the full oracle fixture");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("cannot read {path}: {error}"));
    let stats = replay_fixture(&text);
    assert!(
        stats.decisions > 0,
        "full oracle made no decisions: {stats:?}"
    );
    println!("derived_k full oracle {path}: {stats:?}");
}
