# StageBDecode receipt (S8+S9+S6-decode) — commit b34bbeead

Binaries: hipfire f88fefa446a35a655d475f9d0ec8b2a4,
daemon beb213097b326c958c6aa915ba935f92,
eval_hipfire 00fe05ad36903501b012d93697ff45d1.
Env ordinal 2: HOME=ab2 ROCR_VISIBLE_DEVICES=2 + HIPFIRE_MODELS_DIR,
HIPFIRE_KERNEL_CACHE, HIPFIRE_GRAPH=1, HIPFIRE_LLOYD_GFX12=1.

## 1. Untouched legs .text-identical (cuid stripped): cmp_asm.py
- attention_q8_0_kv 2949c4c93df46f6cfe598b73b0da61d6 IDENTICAL
- attention_bf16_kv f7e9f8191a1b7e8c718da6a59a4024ac IDENTICAL
- attention_flash_q8_0_tile 65d46415fb2ca3cd14c080b4dc2b6cbd IDENTICAL

## 2. Builtin probe: probe_decode_rt PROBE_PASS (256/256 codes, both sels)

## 3. Oracle old-vs-new (seed 42, H8/Hkv2/D256, seq 1/3/5/64/65/127/128/129/300)
- ks1: BIT-EXACT both kernels all seqs (mechanisms 1+2 exact)
- ksnan: BIT-EXACT incl NaN payloads
- ksrand (writer-faithful): scalar maxabs<=2.4e-7, tile<=7.6e-6
- ksfull (extreme codes): scalar<=5e-5, tile<=7.3e-4
Full table: oracle_results.txt. (Oracle history: first version was vacuous —
lost scale memcpys gave ks=0 — and saturated with uniform codes; rebuilt
writer-faithful: host RNE encoder, amax/448 scales, q~N(0,0.15).)

## 4. Decode bench (qwen3.8:27b-mq4-xt, spec off, noslots, stateless)
- q8 control: 36.60 tight (36.50-36.70, sd 0.06)
- fp8 JSON (warm cache): [36.6,35.8,36.7,36.6,36.4] median 36.6 mean 36.42
  min 35.8 sd 0.32 — GATE >=36.4 PASS
- fp8 deep-warmup: [35.7,36.6,36.6,36.6,36.6] median 36.6
- fp8 run1: median 36.60
- Noise note: cold kernel cache produces single-run ~28.7 outliers on EITHER
  arm (JIT inside measurement; blobs timestamped mid-run); an mq4e8 CPU
  harness (~200% CPU) also dragged later runs on both arms. Steady state
  both arms 36.6-36.7. Evidence: bench_fp8_json.log, bench_q8_final.json
  (60573b1b), bench_fp8_run1.log.

## 5. Profile (rocprofv3 --kernel-trace, oracle seq300, median-of-5): profile_table.txt
- attention_flash_fp8_e4m3_tile: 245.8 -> 37.1 us (-208.7, -85%)
- attention_fp8_e4m3_kv: 225.8 -> 101.9 us (-123.9, -55%)
- VGPR 56/40. Daemon-child tracing yields no records, so no bench-level
  table; mechanism (4) is structural (refusal deleted, shared reducer reused;
  this model never sets output_gate on gfx1201, same as q8).

## 6. KLD (eval_hipfire 00fe05ad, prefill scoring, same protocol as Q0 pins)
- WT2 1-chunk: 0.031193 (NLL 2.202592) vs FA2 pin 0.031193 EXACT
- WT2 2-chunk: 0.037398 (NLL 2.253107) vs FA2 pin 0.037398 EXACT
- WT2 24-chunk: 0.048413 (NLL 1.855469) vs Q0 pin 0.048413 EXACT, <=0.049159
- ag-24 x3 fresh: 0.148392/0.148392/0.148392 (NLL 2.108473, no NaN),
  spread 0.000000, identical md5 5ed582fc; matches on-24b 0.148392.
  KLD table: wt2_*.kldseq (a8208c16/5eb73219/76c1087d), ag kldseqs + logs.
