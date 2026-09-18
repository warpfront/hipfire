# iu4sched — next-block fetch-epoch reschedule (§7): ABANDON (gate-set −1.4%)

Branch `gfx12-iu4-k32` @ `c3a543394`. Sole production file touched (REVERTED
after verdict — this note + artifacts are the record):
`kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx12.hip` (+92/−94).
Candidate kernel SHA256: `3585d28af313d60359655a12886152bb9cf5353eb2fd86a5a8456b6efd30b4cd`.
Candidate TU SHA256 `5384cb735ed1fc9f78932752506192f00d9dcfa2b391995ac28e37d7b2400846`
(= JIT `radiowave.json` source_sha256). Baseline binary md5 `1b2d835b…`,
candidate binary md5 `2f7e2439…`.
Baseline kernel
`90ccba37923fe6e73e303bd77165b55cdae7ad0c39dc478de8e1688ce2d9f10b`
(= plan's observed hash; pristine baseline worktree `wt-iu4-base` @ HEAD).
Env: ordinal 1 (announced/released on hub; Fp8FragFix stayed CPU-only).
Model `/home/kaden/.hipfire/models/qwen3.8-27b.mq4-xt`. All GPU runs same
session 2026-09-18 ~09:05–09:10 UTC. Ordinal-1 edge temp 47 °C post-run.

## Change (source-scheduling form 1 of 2; no second form needed — ISA hit)

- Deleted the next-A refill above B1; new fetch epoch immediately after B1:
  all six next-block raw loads (4× b64 A/W nibble + DS word + header word)
  into reused `A_pf`/`W_pf` + new `ds_nx`/`sz_nx` (`unsigned ... = 0u` init).
  No masks/conversion/stores/first-use there.
- Fold moved above B2a (identical `IU4_FOLD_RN` body, same `ds_lds`/`sz_lds`).
- Publish after B2a: original A zero-select, clamped W, masked DS store,
  header bit-extract + `__half2float` + masked SZ store, original producer
  map; `w_lds` re-point W1→(publish)→W0 preserved. Four barriers unchanged;
  fold DAG, stores, LDS layout, ABI/geometry untouched. `last` guards all
  kb+1 accesses. One stale comment fixed (`below its B2b` → `above its B2a`).

## Gate 1 — oracle: PASS 9/9 (`oracle-sched.log`, exit=0)

`cpu_bitwise=OK (mism=0)` + `repeat_identical=OK (mism=0)` on all nine
(gate/set, gate/add, down/set, down/add, m48tail, n80cols, k256/set,
k256/add, m100n100). JIT module `tmp_iu4_gfx12_oracle.08d8dc9d7ca45748`;
cache `.hip` identity verified byte-equal to prelude + candidate kernel.

## Gate 2 — metadata: PASS on assignment thresholds (plan 200-cap noted)

JIT `radiowave.json` (all three entries): VGPR **202** / SGPR 40/40/43 /
spill 0/0 / scratch 0. OCC=1: **3 WG/CU** both entries (block 256,
LDS 19456). Assignment (0 spill, VGPR ≤ 256, 3 WG/CU) passes. NOTE: plan
§7.2's inference cap (≤200 raw VGPR, target 183–192) is breached by 2;
per the plan's own "allocation-rounded VGPR and live peak decide" and
empirically intact occupancy, treated as recorded deviation, not a stop.
hipcc TU cross-check: baseline 183 (= JIT baseline), candidate 202.

## Gate 3 — ISA: PASS, intended schedule achieved (JIT hsaco, `jit-sched.dis.txt`)

JIT disasm is byte-identical to the hipcc TU disasm (modulo filename).
`full_set` steady trip: B0 0xEC6C/0xEC7C, B1 0xECA0/0xECA4, fetch loads
0xEDA4/0xEDB0/0xEDC0/0xEDCC (4× b64) + 0xEDD8/0xEDE4 (DS/header b32),
compute1 16 WMMAs 0xEE28–0xEECC, fold DS reads 0xEEE8–0xEF80, FIRST
covering wait 0xEF88 (B2a drain) → publish cndmasks/stores 0xEFA8–0xF024.
Zero `s_wait_loadcnt` between 0xEDE4 and 0xEF88. Baseline contrast
(`tu-base.dis.txt`): next-A loads 0xEFFC/0xF008 waited at 0xF014/0xF028 +
B1 drain 0xF03C before first compute1 WMMA 0xF078. Trip census: 32 WMMA,
8 b64 + 2 b32 globals, 12 frag reads (=24 b64), 20 b32 metadata, 8 logical
b64 + 2 b32 publishes (6 store lines: 4 dual + 2 b32), 4 barrier pairs.

## Gate 3 — TIME same session: FAIL → ABANDON (3 interleaved brackets)

Fresh process per binary per bracket (`time-brackets.sh`), order base→cand.
Rows: gate/set, gate/add, down/add, qkvza/set, qkv/set (N=512).

| row | base a/b/c (med) | cand a/b/c (med) | Δ |
|---|---|---|---|
| gate set | 504.8/501.5/504.5 (504.5) | 495.7/499.1/497.4 (497.4) | −1.4% |
| gate add | 520.8/522.5/523.7 (522.5) | 513.9/519.3/518.9 (518.9) | −0.7% |
| down add | 548.0/548.5/552.2 (548.5) | 507.8/507.4/510.6 (507.8) | −7.4% |
| qkvza set | 492.9/500.8/498.9 (498.9) | 488.4/489.2/489.0 (489.0) | −2.0% |
| qkv set | 412.7/413.8/413.3 (413.3) | 408.5/405.0/406.6 (406.6) | −1.6% |

Gate/set −1.4% < required ≥3% → reject per §8. No row regresses >2%
(all improve). down/add −7.4% is consistent across brackets but does not
override the gate/set gate; no claim made. Gate 4 (pins/bench/decode/serve)
NOT run. Binaries md5 (build log): candidate `2f7e2439…`; baseline binary
in `wt-iu4-base` (worktree removed after verdict).

## Files

- `tu-sched.hip` (candidate TU), `tu-base.hip` (baseline TU)
- `tu-sched.dis.txt` / `jit-sched.dis.txt` (identical), `tu-base.dis.txt`,
  `trip.txt` (candidate trip census window)
- `oracle-sched.log`, `time-base-{a,b,c}.log`, `time-cand-{a,b,c}.log`
- `build-cand-oracle.log`, `build-base-oracle.log`, `tu-sched.build.log`
- `time-brackets.sh` (bracket recipe)
