# Lloyd full twin (qkv/qkvza + fused producers): on-model A/B (gfx1100, 27B mq4l)

**Lifecycle:** `historical`. Fixture-bound measurement under the exact method
below. Not a current default, baseline, or admission decision.

## Fixture

- GPU: gfx1100 (RX 7900 XTX), HIP 7.16, ROCm 10.1 therock toolchain
  (`--rocm-device-lib-path=<therock>/lib/llvm/amdgcn/bitcode`); pinned clocks
  per `GPUCruise lloyd-bench.json` (branch convention).
- Models: `qwen3.8-27b.mq4l` md5 `d020f248ca52258b1aff623c084719b6`
  (qt=52 Lloyd); `qwen3.8-27b.mq4` md5 `d1292b4d5bd6046693604201a6ca8074`
  (qt=44 uniform, decode-gap arm only).
- Binaries (release): `hipfire 3706ca6b…`, `daemon 8b66f6c5…` (branch
  `feat/gfx1100-mq4v2-lloyd-prefill` + uncommitted dispatch work: decode D1,
  prefill P1-Step 1 qkv/qkvza/partial/single-weight SET/ADD, P2 producer
  admission with A4-instead-of-symmetric). Reference binaries
  `hipfire 226ae11d…` / `daemon 4ceda09f…` (= `2026-10-31` staged-twin
  checkpoint pair) used only to prove the old-CLI+new-daemon confound (discarded).
- Route under test: `HIPFIRE_LLOYD_MMQ_A4=1` (default OFF); OFF leg is X128.
  Exact gfx1100, eager-only, AR (`--spec off`), KV q8 legacy,
  `HIPFIRE_GFX1100_FA2_SPLIT_VERIFY=0`, `HIPFIRE_DAEMON_BIN` pinned per leg
  (unpinned CLI resolves `target/release/daemon.exe`, which voids old/new
  comparisons — see lesson).
- Prompts: `ctx32k_code.txt` md5 `3b80bc02…` (26681 tok),
  `ctx100k_code.txt` md5 `39cdbb50…` (106731 tok),
  `humaneval_3_below_zero.txt` md5 `37c5aad9…`,
  `humaneval_2_truncate.txt` (identity only).

## Method

- pp2048: `hipfire bench <model> --matrix --pp 2048 --tg 32 --runs 3
  --warmups 2 --backend noslots --workload stateless --json`, fresh daemon
  per leg, order new-OFF / new-ON / new-ON / oldCLI-newDaemon-ON(discarded) /
  oldCLI-newDaemon-OFF (= third X128 replicate).
- Long ctx: same flags via `--prompt-file <ctx> --ttft --max-tokens 32
  --runs 3 --warmups 1`, order ON then OFF per depth (same session).
- Decode-with-fill: `--matrix --pp 512 --ctx 26700,107000 --tg 32`,
  mq4l then mq4 (synthetic KV fill; isolates ctx dependence from the ~47 s
  prefill thermal delta of end-to-end runs).
- Identity: `hipfire run -j --no-stream --temp 0 --max-tokens 512`,
  HE/2 + HE/3, old vs new and A4-off vs A4-on, byte-compared `.content`.
- Serve: `serve_harness.py --mode battery` (5 turns) with and without
  thinking, A4 on, new pair.

## Results

Prefill, mq4l (median of 3, stdev in parens):

| ctx | X128 OFF | full twin ON | Δ |
|---|---|---|---|
| pp2048 synthetic | 1051.3 (0.82) / 1050.9 (0.51) | 1336.2 (1.13) / 1338.3 (0.68) | **+27.2%** |
| 26.7K (TTFT 29.2s → 23.8s) | 913.5 (0.82) | 1119.5 (0.19) | **+22.5%** |
| 107K (TTFT 160.7s → 139.9s) | 664.2 (0.51) | 762.9 (0.33) | **+14.9%** |

X128 legs reproduce the 2026-10-31 draft baselines (1051/913.5≈916/664.2≈666)
within 0.3%. Shipped-T3 reference from the 2026-10-31 checkpoint on this
rig/instrument: 1218.2 (same pp2048) → this work's additions ≈ +9.8%
(cross-session anchor, md5-pinned; deltas are 10-100× run stdev).

De-confounded decode TG32, synthetic fill (median, stdev ≤0.08):

| ctx fill | mq4 uniform | mq4l Lloyd | gap |
|---|---|---|---|
| 26.7K | 46.09 | 38.30 | **−16.9%** |
| 107K | 38.03 | 32.57 | **−14.3%** |

The gap is ctx-flat (~15-17%): the draft's widening (−22% → −33%) was
thermal soak + shared-KV L2 effects, not local LUT behavior. Short-ctx
decode tied across arms (~41 tok/s TG32; serve battery 41.4-41.6 steady).

Dispatch parity (decode D1): old-vs-new bind sets identical on qwen3.8 and
tok/s unmoved — the FA side already fuses via the GQA decode path and the
DeltaNet gate is shape-shut here. D1 kept (bit-exact) but contributes 0 on
this model; decode gains are owed by D2-D4 (not built).

## Exactness / quality

- HE/3: old==new==A4off==A4on, 512/512 tokens byte-identical (D1 proof).
- HE/3 twin: A4off==A4on byte-identical (Step 1 proof, pre-producer).
- HE/3 producers-live: A4off==A4on byte-identical (Step 2 proof; binds show
  all four fused i8 producers + twin SET/ADD, X128 binds gone).
- HE/2: old==new-off==new-on byte-identical (489/489, stopped).
- Serve batteries (thinking on + off): 10 varied turns, 0 attractors,
  0 empty (thinking-off), coherent on-topic text; daemon md5 `8b66f6c5…`
  pinned per turn.

## Disposition

Perf evidence for the final plans' prefill Steps 1+2
(`2026-10-31-lloyd-prefill-speedup-next-final.md`, local-only) and decode
Phase 0 + D1/D5 findings (`2026-10-31-lloyd-decode-speedup-final.md`,
local-only). Decoding Steps 2-4 unbuilt. No promotion claimed; landing
still owes PR review + pre-commit-ci on the touched dispatch files.
