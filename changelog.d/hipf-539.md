### Fixes: device-mesh catalogue

- **Device-mesh catalogue: `g11-glimmer-dflash-max-seq` also runs `kv_cap_gpu::rows_past_max_seq_are_an_error`.** The Glimmer test that a verify block or prefill chunk ending past `max_seq` returns an error (#666 G11) never had a row. It shares the Glimmer fixture, so it joins that row as a second cargo command; `min_passed` is now 3 and its fixture-missing skip line is forbidden (HIPF-539, found in HIPF-535 N3).
