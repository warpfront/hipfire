# AttnBundle — KILL

Base: `mq4-lloyd` / `2af9a7960`
Branch: `gfx1201-attn-bundle`
Ordinal: 2
Result: no lever meets all ship gates; no source commit.

## Baseline

- Packet direct/partial: 239/239 VGPR, 6/6 waves/SIMD, 0/0 spill.
- Packet census: 2798 common, 2820 header-copy, 2803.5 fair packets/wave/KT64.
- Evidence: `baseline/resources.log`, `baseline/census.json`.

## A2a — rcp + one Newton iteration

- Compile screen: PASS — direct/partial 238/238 VGPR, 6/6 waves/SIMD, 0/0 spill.
- Packet census: 2295 common, 2315 header-copy, 2300.0 fair packets/wave/KT64; 128 WMMA and 16 fp8 converts retained.
- KLD c24: **0.067222**, above the 0.064390 ceiling (pin 0.063890 + 0.000500). KILL.
- Evidence: `a2a/resources.log`, `a2a/census.json`, `a2a/kld.json`, `a2a/c24.bin`.

## A2b — fixed shift

- A compile/census screen completed while A2a quality evaluation was in flight: direct/partial 237/237 VGPR, 6/6 waves/SIMD, 0/0 spill; 2203 common, 2223 header-copy, 2208.0 fair packets/wave/KT64.
- Per the ordering gate, A2b was halted without KLD or timing after A2a failed c24 quality. KILL.
- Evidence: `a2b/resources.log`, `a2b/census.json`.

## A3 — Q raw prefetch + DS/WMMA schedule groups

- Applied to the direct template only; the partial template remains exact baseline code.
- Compile screen: PASS — direct 227 VGPR / 6 waves / 0 spill; partial 239 VGPR / 6 waves / 0 spill.
- Packet census: 2865 common, 2887 header-copy, 2870.5 fair packets/wave/KT64 (+67.0 versus baseline).
- Device-resident exactness oracle: PASS, 921/921 cases, direct + split1/split8 record/merge, zero input diffs.
- Pinned-daemon hashes: baseline `4dc5ac92a22dc43d72e4683b48a13a3c`; candidate `7618852681c3275b18ca306595293169`.
- Paired median matrix (runs=3, warmups=1; decode 36.4951 vs 36.5018 tok/s, +0.018%, both >=36.4):
  - pp512: 2344.0 -> 2348.4 tok/s (+0.188%)
  - pp2048: 2420.7 -> 2419.3 tok/s (-0.058%)
  - pp8192: 2325.7 -> 2303.0 tok/s (**-0.976%**)
  - pp32768: 1939.8 -> 1865.2 tok/s (**-3.846%**)
- Fails both pp8192 +1.5% ship gate and the no-row-worse-than-1% gate. KILL.
- Evidence: `a3/resources.log`, `a3/census.json`, `a3/oracle/oracle.log`, `a3/daemon-md5.txt`, `a3/baseline-r3.jsonl`, `a3/candidate-r3.jsonl`, plus discarded warmups.

## A5 — K-only next-tile prefetch

- Compile screen: FAIL — direct 253 VGPR, 5 waves/SIMD, 0 spill; partial baseline remains 239 VGPR / 6 waves / 0 spill.
- Exceeds the <=240 VGPR / 6-wave gate, so census remapping, oracle, and timing were halted. KILL.
- Evidence: `a5/resources.log`, `a5/packet.s`.
