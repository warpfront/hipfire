# Fp8Geom gate 3 — TIME N=512 --quick, same-session interleaved (ordinal 2)

Env: `HIPFIRE_GFX12_MQ4V2_FP8_V2=1 HIPFIRE_GFX12_MQ4V2_FP8_SLABS=2 TIME=1`,
binary `target/release/examples/tmp_gemm_v2_oracle` (md5 359306ec…),
rows `base128 new64 base128b new64b` (script `gate3-time.sh`).
Raw: `time-512-{base128,new64,base128b,new64b}.stdout`.

## Medians (us, n=6 per kernel per row)

| row (GEOM)        | gate_up | residual | qkv   | qkvza | pack |
|-------------------|--------:|---------:|------:|------:|-----:|
| base128 (128x128) | 1440.7  | 801.7    | 601.5 | 698.8 | 28.6 |
| new64 (128x64w8)  | 1579.7  | 815.1    | 645.6 | 716.6 | 28.0 |
| base128b          | 1446.3  | 796.2    | 604.7 | 695.9 | 28.2 |
| new64b            | 1586.6  | 815.9    | 648.6 | 720.0 | 28.6 |

Means: control gate_up 1443.5 / residual 799.0 / qkv 603.1 / qkvza 697.4;
new 1583.2 (+9.7%) / 815.5 (+2.1%) / 647.1 (+7.3%) / 718.3 (+3.0%).
Interleave stability ±0.2–0.4% — no state/thermal confound.

## Verdict: NOT ADMITTED (abandon)
Bar: gate_up(new) ≤ 1300 us AND no shape worse. Got 1583 (+9.7%) with all
four shapes worse. Both interleaves agree.

## Read
M-heavy shapes suffer most (gate_up +9.7%, qkv +7.3%), K-heavy least
(residual +2.1%, qkvza +3.0%). Consistent with per-WG fixed costs scaling
with WG count (2× WGs): prologue S/SZ/address staging per output doubles,
and A-panel re-reads double (W streaming evicts A from L2 across 2× row
tiles). The 2→4 waves/SIMD occupancy gain does not cover it; the dropped W
prefetch (X2) exposes W-load latency on top. 128x128 is the better operating
point. 64x128w8 not timed: same WG-doubling with worse traffic (doubles
HBM-streaming W instead of L2-resident A) — analyzed-rejected in
METADATA.md. 128x128x32 not built: keeps 128 acc VGPRs, fails gate 2 by
construction.
