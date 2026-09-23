# Stage B3 — StageB3 stacked evidence (KLD / bench / serve + gate reruns)

Base: `ac347a2ab` (StageB1's B3 commit; kernel bytes there already contain
my P-pack hoist — `packed ONCE per subtile` at K:1235 — because StageB1
committed the worktree with my edit in it; verified: worktree clean at
`ac347a2ab`, `git diff 421857b4a ac347a2ab` == the hoist, 15+/13-).
Design receipt: `B3_receipt.md` (StageB1). This file adds: independent
reruns of gates 1–4 on the landed bytes + the full gate 5 (KLD) and
gate 6 (bench/decode/serve) with numbers and md5s.
GPU: ordinal 1 (ab1). Env per run noted below.

## Gate 1 — K0/K3/K8-Q0 objects bit-identical to e7a43c418: RERAN PASS

Same-filename `--genco` (`hipcc --genco --offload-arch=gfx1201 -O3
--no-offload-compress`) object `cmp` ×3, base source from
`git show e7a43c418:...hip`, new source = landed bytes.
Evidence: `g1/eq_base/*.o`, `g1/eq_new/*.o`.

- K0: `5190514ccc08257670a8e45ead30aade` == both → IDENTICAL
- K3: `16ff00a8c8b58160aa4f4627796be29e` == both → IDENTICAL
- K8-Q0: `df8a62fdeb6c0d4cf8427567b94e91b9` == both → IDENTICAL

(Different-filename builds differ at early bytes = embedded source
pathname only; same-basename builds are byte-identical.)

## Gate 2 — resources (K8+FP8 TU): RERAN PASS, matches B3_receipt.md

`-Rpass-analysis=kernel-resource-usage`, evidence `g1/new_k8fp8_res.log`:

| entry | VGPR | waves/SIMD | spill V/S | scratch |
|---|---|---|---|---|
| direct `attention_fp8_e4m3_fa2_gqa_gfx1201` | 190 | 8 | 0/0 | 0 |
| partial | 191 | 8 | 0/0 | 0 |
| fp8 pre-convert | 56 | 16 | 0/0 | 0 |
| f16 pre-convert | 13 | 16 | 0/0 | 0 |
| merge | 17 | 16 | 0/0 | 0 |

≤240 VGPR ✓, 0 spill/scratch ✓, 8 waves (exceeds the 6-wave class).

## Gate 3 — ISA (direct entry): RERAN, corroborated with one correction

`--save-temps` `.s` + `llvm-objdump` path notes in `g1/isatmp/`
(`direct.s` = direct-entry function slice):

- fp8 WMMA static: **20** = 4 QK + 16 PV → dynamic per compute wave per
  full KT64: QK 4×(4 dg×4 sub) = **64** ✓, PV 16×4 sub = **64** ✓ (128 total).
- f16 WMMA anywhere in the new module: **0** ✓.
- `v_cvt_pk_f32_fp8` (fp8→f32 decode): **0** — V-expand gone ✓.
- P-pack: **4 `v_cvt_pk_fp8_f32`** static, once per subtile (hoisted) ✓.
- Scale widens: 16 `v_cvt_f32_f16` (= f16→f32; 8 sk + 8 sv, allowed) ✓.
- **Correction to B3_receipt.md §gate-3:** the landed (hoisted) bytes
  build the 8 `w/b_new` quotients as **true IEEE divisions** (each =
  `v_div_scale`×2 + `v_rcp` + `v_div_fmas` + `v_div_fixup`; plus
  `wmax/448` and `rho` divs, ~10 total), exactly matching the host
  division model. StageB1's "reciprocal+mul" note described the pre-hoist
  CSE build. Ulp-level either way; division is the better host-model match.
- QK calibration reproduces B2: dg-iter 26 static × 16 trips = **416** ✓.
- Per-wave packet total per KT64: loop-annotation census
  (K-fill ×8, V-fill ×16, dg ×16, sub ×4, tile ×1):
  raw static×trips = **5035** (both diamond sides);
  fast-path-adjusted ≈ **3700** (≈ StageB1's ~3800 ±10%).
  Vs plan cap **2216 → OVER ~70%** (same conclusion as B3_receipt.md:
  V-fill overhead + rolled-loop latch/index + new support work).
  Bench adjudicates (gate 6: +7.1% @32k).

## Gate 4 — layer-35 O compare vs arm1 (device, real taps): RERAN PASS

Vehicle `tmp_b2_qk_compare.rs` rebuilt on landed bytes (11:02; the
`include_str!` kernel embed requires rebuild — the 20:17 binary predates
the hoist), `HIPFIRE_GFX12_FA2_FP8=1`, evidence `compare/b3*.log`
(the `B2-vs-arm1` label is the vehicle's static string; the binary is B3):

| prompt | Q0-vs-arm1 (control) | B3-vs-arm1 | B3-vs-Q0 | nonfinite |
|---|---|---|---|---|
| default | 1.0123/2.01e-2 | **1.9155/3.11e-2** | 1.9152/1.90e-2 | 0 |
| p4096 | 1.0949/2.29e-2 | **2.5590/4.19e-2** | 2.4140/3.08e-2 | 0 |

- Screen arm5 (route N = what B3 implements): default 1.915/3.21e-2,
  p4096 2.559/4.23e-2 → max-abs matches to **4 digits** both prompts.
- Ticket arm-6 envelope (2.04/3.3e-2; 2.46/4.2e-2): default inside;
  p4096 max 4% over (2.559 vs 2.46), tail inside — same disposition as
  B3_receipt.md (source-correct arm5 match <1% is the evidence).
- Underflow: host census 0.0198%/0.0183% ≤ 0.1% (S7, both prompts);
  device O reproducing arm5 implies matching device underflow.

## Gate 5 — KLD on the production route: PASS

`eval_hipfire` rebuilt on landed bytes (21:04).
`HIPFIRE_GRAPH=0 HIPFIRE_NORMALIZE_PROMPT=0 HIPFIRE_LLOYD_GFX12=1
HIPFIRE_GFX12_FA2_PREFILL=1 HIPFIRE_GFX12_FA2_FP8=1
--model .../qwen3.8-27b.mq4-xt --kv-mode fp8 --kv-v q8 --scoring-mode prefill`.
Flag-off twins (no `FP8`) are the matched Q0 pins. Evidence `eval/`.

| run | ON (B3) | md5 | OFF (Q0, same tree) | q8 retained |
|---|---|---|---|---|
| WT2 1-ch | **0.030887** | 59d6e83a…322d48 | 0.031193 (a8208c16…1dea5) | 0.029580 |
| WT2 2-ch | **0.036857** | 7bd0ef53…0bc988 | — | 0.036694 |
| WT2 24-ch | **0.048124** | 70367c88…3cb9b | 0.048413 (76c1087d…f94303) | 0.048659 |
| ag 24-ch | **0.145290** | 9efa8568…275ac | 0.148392 (5ed582fc…c0a1ab) | 0.141396 |

- Engagement proof: ON md5s differ from OFF; OFF reproduces retained
  pins to 6dp (WT2-24 0.048413, ag-24 0.148392 with NLL 2.108473).
- Admission: WT2-24 ON 0.048124 ≤ 0.049159 ✓ (margin 0.001035);
  kernel-isolated ON ≤ OFF+0.0005: 0.048124 ≤ 0.048913 ✓
  (beats OFF by 0.000289, beats q8 by 0.000535).
- ag: ON beats OFF by 0.003102, NLL finite (2.102311), no NaN;
  ON inside the prior ON spread, under the 0.148392 pin.
- 1/2-ch ON reproduce the route numbers exactly (0.030887/0.036857):
  hoist is KLD-neutral at short context, as expected (ulp-level div change).

## Gate 6 — bench / decode / serve: PASS

Rebuilt `target/release/hipfire` md5 `bbd4d4ff1bd7de75f5e6c5709d307bec`,
`target/release/daemon` md5 `403d619db3d5589a7d4075faf77f7b2f`
(battery log independently confirms the daemon md5).
Matrix: `hipfire bench qwen3.8:27b-mq4-xt --matrix
--pp 512,2048,8192,32768 --ctx 128 --tg 64 --spec off --runs 3
--warmups 1 --kv-mode {q8|fp8} --json`, fp8 arms add
`HIPFIRE_GFX12_FA2_PREFILL=1 HIPFIRE_GFX12_FA2_FP8=1`.
Interleaved q8 / fp8+flag / q8 / fp8+flag (graphless), evidence `bench/`:

| row | q8-a | fp8-b | q8-c | fp8-d | Δ (means) |
|---|---|---|---|---|---|
| pp512 | 1481.7 | 1483.0 | 1479.4 | 1482.0 | **+0.13%** |
| pp2048 | 1447.6 | 1456.7 | 1446.9 | 1455.0 | **+0.59%** |
| pp8192 | 1340.3 | 1367.5 | 1340.5 | 1366.0 | **+1.97%** |
| pp32768 | 1030.3 | 1104.5 | 1030.9 | 1102.4 | **+7.07%** |
| tg64@128 | 36.01 | 35.98 | 35.99 | 35.95 | −0.1% |

q8 arms repeat within 0.2%; fp8 arms within 0.2%. Prefill win grows
with pp (+7.1% @32k); decode column ≈36.0 both arms (no discrimination).

- Decode-only (graphs ON, fp8+flag): tg64@128 = **36.45 ≥ 36.4** ✓
  (`bench/fp8-graphon.json`; prefill rows corroborate:
  1482.4/1457.3/1370.2/1106.0).
- Serve: `serve_harness.py --mode battery --model qwen3.8:27b-mq4-xt
  --kv fp8 --thinking off` with both flags in the daemon env:
  **5/5 turns finish=stop, runaway=0, empty=0**, all texts read
  (code/reasoning/factual/prose/instruct), no missing-substring
  failures, avg_decode 36.4 tok/s (`serve/battery.{json,log}`).
  (`--mode` is the correct selector; bare `battery` is rejected.
  `--thinking off` needed: default thinking cap eats the whole budget.)

## Disposition

All six gates green on the landed bytes. No kernel change beyond
`ac347a2ab` was required or made (worktree clean). Stacked commit is
content-empty by design; gates 5–6 evidence anchored here.

## Gate 6b — IU4 bracket (Main follow-up): PASS with notes

Second bracket with `HIPFIRE_IU4_PREFILL=1` on both arms
(iu4+q8 flag-unset vs iu4+fp8+flag), same matrix command, 2 runs each
interleaved (graphless), evidence `bench/iu4*.json`:

| row | iu4q8-a | iu4fp8-b | iu4q8-c | iu4fp8-d | Δ (means) |
|---|---|---|---|---|---|
| pp512 | 2085.4 | 2088.7 | 2095.5 | 2096.6 | **+0.11%** |
| pp2048 | 2022.9 | 2036.0 | 2029.6 | 2044.1 | **+0.68%** |
| pp8192 | 1821.7 | 1868.4 | 1825.6 | 1873.1 | **+2.58%** |
| pp32768 | 1295.1 | 1412.9 | 1295.8 | 1413.0 | **+9.07%** |
| tg64@128 | 36.02 | 35.98 | 36.02 | 35.98 | −0.1% |

- Same shape as the non-IU4 bracket, amplified at 32k (+9.1% vs +7.1%).
- Note: my iu4+q8 means sit ~3–5% under the prior iu4+q8 reference
  (2189/2117/1898/1338): 2090/2026/1824/1295. Same-tree, same-settings
  paired deltas are unaffected; the absolute gap (graphless here,
  possibly graphs-on reference, or tree drift) is recorded, not chased.
- IU4 1-chunk eval (combined route runs?):
  `IU4_PREFILL=1` + fp8 + flag → KLD **0.043126**
  (md5 `5c49765c97632d8fdfdee16a7115da53`, NLL finite 2.207421);
  same-tree iu4+q8 flag-unset → **0.041329**
  (`032ebad84f2c1dd5e2980fa805215ac0`), exactly the retained iu4 pin.
  Combined-route delta +0.001797 at 1 chunk (vs +0.001307 non-IU4
  fp8-vs-q8 at 1 chunk) — consistent with additive IU4-GEMM + fp8-attention
  error. Reported per Main; no admission bar was set for this arm.
