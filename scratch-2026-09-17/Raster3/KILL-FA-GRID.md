# KILL — Unit A: packet-FA grid order (Raster3, ordinal 4, 2026-09-19)

Baseline: query-tile-major (`x` = query tile, `y` = kv head, `z` = run).
Screened via in-kernel `blockIdx` remap (reverted after the screen; no
shipped code): kv-head-major (order 1), grouped query tiles G=2/4/8
(orders 2/3/4, tile-fastest within each G×4 chunk, baseline fallback when
G ∤ nx).

## Exactness — 921-case FA oracle, all PASS

`tmp_fa_packet_oracle` (device-resident, packet-direct vs route-N fp8 +
split-1/8 records/merges), ordinal 4:

| order | result |
|---|---|
| 0 baseline | 921 PASS, failed=0 |
| 1 kv-major | 921 PASS, failed=0 |
| 2 grouped G=2 | 921 PASS, failed=0 |
| 3 grouped G=4 | 921 PASS, failed=0 |
| 4 grouped G=8 | 921 PASS, failed=0 |

Screen bug found and fixed along the way: the grouped decode is only a
bijection when G | nx; the first version stranded (tile, kv) combos on
indivisible shapes, and the ternary fallback computed sdiv/srem-by-zero
(ng=0) whose poison leaked through the select. Restructured to a fully
separated branch. Oracle logs: `fa-oracle-g{0..4}/oracle.log`.

## Timing — no order wins, KILL

Device-resident prefill-8192-scale probe (`tmp_fa_grid_order_probe`,
untracked; one FA launch = 8192 queries × H24/KV4/D256, fp8 KV), ordinal 4.

rocprofv3 `--kernel-trace` one-pass sums, 25 calls each
(`rocprofFA-g{0..4}/trace_kernel_stats.csv`, `attention_fp8_e4m3_fa2_gqa_packet_gfx1201`):

| order | total (25 calls) | mean/call | Δ vs warm baseline |
|---|---|---|---|
| 0 baseline (warm re-run) | 280.32 ms | 11.21 ms | — |
| 1 kv-major | 282.57 ms | 11.30 ms | +0.8% |
| 2 grouped G=2 | 282.11 ms | 11.28 ms | +0.6% |
| 3 grouped G=4 | 281.31 ms | 11.25 ms | +0.4% |
| 4 grouped G=8 | 283.42 ms | 11.34 ms | +1.1% |

Interleaved event medians (15 runs, `fa-probe-screen.log`) agree:
g0≈g1≈g3 (11.26 ms) < g2 (+0.9%) < g4 (+1.4%).

The baseline already maximizes KV-stream focus (24 consecutive
workgroups share one kv head); all alternatives lie within ±1% noise.
No variant clears any bar. **Killed — baseline kept, remap reverted.**
`profile_prefill_qwen35` could not serve as the vehicle (no full qwen35
`.hfq` on disk; all `*dflash*.hfq` are 5-layer drafts); bench-daemon
rocprof cannot finalize CSVs (daemon is SIGKilled), so the synthetic
device-resident probe + direct-process rocprofv3 stands in. Bench
end-to-end pp8192 rows per order (`bench-g{0..4}.json`) corroborate
(no order effect outside drift).
