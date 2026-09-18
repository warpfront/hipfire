# iu4sched gate 4 (override by Main 2026-09-18): PASS with noted anomalies

Candidate kernel `3585d28a…` re-applied on branch (this gate's commit).
Env for ALL gate-4 GPU runs: HOME=/home/kaden (normal), ROCR_VISIBLE_DEVICES=1,
HIPFIRE_KERNEL_CACHE=ab1 path, HIPFIRE_MODELS_DIR=/home/kaden/.hipfire/models,
plus per-gate flags below. Ordinal 1 announced/taken/released on hub
(Fp8FragFix waited; Fa2Fill yielded after a 10-min pair).
Model fixture verified: `scripts/check_fixture.sh --sha` OK
(14987185152 B, sha256 pinned).

Binaries (rebuilt from accepted source):
- ON `hipfire` ea4ee884cfbcc8f290017edc4b045c55, ON `daemon`
  181d6aa46338ee53c6a4e1f4563a66c3, eval_hipfire release (exit 0)
- OFF `hipfire` 1dd5f084a9a83498adba39bad375d586 (wt-lloyd @ af16ec4bd)

## KLD pins (HIPFIRE_GRAPH=0 NORMALIZE=0 LLOYD_GFX12=1 IU4_PREFILL=1): PASS

`eval_hipfire --model …mq4-xt --ref /home/kaden/kldrefs/qwen3.8-27b.ref_wt2.bin
--kv-mode q8 --kv-v q8 --scoring-mode prefill --max-chunks {1,2,24}`:
- c1.bin md5 `032ebad84f2c1dd5e2980fa805215ac0` ✓ (KLD 0.041329)
- c2.bin md5 `dc7e53181662271780374f0a85fd7732` ✓ (KLD 0.048091)
- c24 slice-mean KLD `0.063410` (NLL 1.864044, PPL 6.4498) ✓ — all bit-for-bit.

## Interleave bench OFF/ON/OFF/ON + pair 3 (GRAPH=1 LLOYD=1): receipt

`hipfire bench qwen3.8:27b-mq4-xt --matrix --pp 512,2048,8192,32768 --ctx 128
--tg 64 --spec off --runs 3 --warmups 1 --kv-mode q8 --json`.
OFF = wt-lloyd binary, IU4_PREFILL unset; ON = candidate binary, IU4_PREFILL=1.
Cache identity: bench JIT entry
`gemm_mq4g256v2_residual_mmq_iu4_gfx12.08d8dc9d7ca45748.hip` carries the
fetch-epoch marker → ON ran the candidate. All six arms decoded in
36.5-state (36.50–36.56) EXCEPT off3 (28.85, 29-state); pp512 is
state-independent per Main's bracket-3 finding.

| arm | pp512 | pp2048 | pp8192 | pp32768 | decode |
|---|---|---|---|---|---|
| off1 | 1489.3 | 1421.0 | 1309.9 | 1038.9 | 36.51 |
| on1 | 2159.5 | 2088.5 | 1877.2 | 1328.5 | 36.51 |
| off2 | 1481.7 | 1453.4 | 1348.8 | 1039.0 | 36.50 |
| on2 | 1967.0 | 2085.8 | 1873.9 | 1326.7 | 36.55 |
| off3 | 1489.3 | 1420.6 | 1308.8 | 1013.4 | 28.85 |
| on3 | 2151.1 | 2080.9 | 1871.0 | 1325.7 | 36.56 |

ON 3-run medians: 2151.1 / 2085.8 / 1873.9 / 1326.7 vs prior ON ranges
2060-2100 / 1910-2020 / 1713-1819 / 1244-1294 (+2.4% / +3.3% / +3.0% / +2.5%).
pp2048+ deltas are consistent with 36.5-state inflation (~+3% per Main);
pp512 (+2.4%, state-independent) is consistent with the small kernel win.
NOT "stale" (outside 2%), but explainable; on2-pp512 (1967) is an outlier
against on1/on3 (~2155, 1-warmup noise on a ~0.25 s row). OFF pp512 is
rock-stable (1489.3/1481.7/1489.3).

## Decode-only (noslots stateless, 5 runs/3 warmups, 128 tok): PASS

OFF median 36.7, ON median 36.7 — both ≥ 36.4 (and above today's OFF floor
36.2). Decode unaffected, as expected for a prefill-only change.

## Serve battery + chain (ON daemon 181d6aa4, GRAPH=0 IU4_PREFILL=1): PASS

`serve_harness.py --mode battery/chain --model …mq4-xt --kv q8 --max-tokens 256`
(both exit 0). Battery 5 turns: 4× stop coherent + 1× LENGTH truncation of
coherent prose (harness runaway flag = length heuristic; tail text verified
clean). Chain 5 entries: 4× stop coherent (decode 35.8–36.0), 1× empty turn
recorded after daemon validation `open think span at end of generation`
(rolled back). The SAME validator message appears in the prior OFF baseline
`iu4stage/serve-off.log` → pre-existing model/harness failure mode, not a
candidate regression (prefill is oracle-proven bit-exact; both runs used
seed=None sampling). `connection closed` teardown lines appear in baseline
logs too (2 each). No attractor/repetition/stream-error flags anywhere.

Excerpt 1 (battery code): correct `merge_sorted` with docstring, two-pointer
loop, `extend` tails. Excerpt 2 (chain ctx-1051): clean 5-item numbered list
of naming/function guidelines. Full texts in serve-battery.json/serve-chain.json.

## Verdict

Gate 4 complete: pins exact, bench receipt recorded (small real movement,
no staleness claim), decode ≥ floor, serve coherent with one
pre-existing-type validation anomaly reported, not hidden.
