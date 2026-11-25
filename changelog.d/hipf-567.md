### Fixes

- **Fixed: DeepSeek4 DSpark fault-matrix fixture setup (G4.4 `g4-dspark-fault-matrix`).** Since `d32fb6cd1`, `config_from_hfq` refused the 24-tensor MQ2R DSpark sidecar because the file-identity route check needs the full P3 map, so the 4 `dspark_after_*` tests panicked before fault injection. The fixture now verifies the sidecar with `validate_mq2r_dspark_sidecar`, as the product load does, and takes the config from the new `DeepseekV4Config::from_mq2r_dspark_sidecar`. The product path does not change.
