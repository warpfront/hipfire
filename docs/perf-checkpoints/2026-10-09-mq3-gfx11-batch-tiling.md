# MQ3 gfx11 batch-tiling — measured policy for the Lloyd mb4 gates; MQ3V2 raw-bit screen

- **Date:** 2026-10-09
- **Lifecycle:** historical
- **GPU:** gfx1100 (Radeon RX 7900 XTX, 24 GiB), exclusive, ROCm 7.15 container
  (`buun-llama:git-9be465c6c-rocm10.0.0`).
- **Fixtures:** `qwen3.5-4b.mq3` (loads the MQ3-G256-**Lloyd** family — verified
  by prefill profile) and `qwen3.5-9b.mq3`; product path
  `hipfire bench --matrix --kv-mode q8 --spec off`, fresh process per arm,
  interleaved.
- **Parent context:**
  [2026-10-09-gfx1100-campaign-reverify.md](2026-10-09-gfx1100-campaign-reverify.md)
  (quant-class sweep; the v1-only BT2 removal).

## The gap, stated precisely

Three batch-tiling (weight-reuse) mechanisms exist upstream, none of which
helped MQ3 prefill on gfx1100 before this change:

| family | mechanism | gfx1100 state before |
|---|---|---|
| MQ4G256V2 | `gemm_*_mq4g256v2_wmma_gfx11_bt` (BT4/6/12), production via `mqv2_prefill_batch_tile` | engaged at batch ≥ 96 |
| MQ{2,3,5,6}G256V2 | generic `gemm_mqv2_wmma_gfx11_bt.hip` (`template<int BITS>`) + dynamic launchers that already accept bits=3 | **quarantined**: `mqv2_gfx11_bt_admitted("gfx1100", 3) == false`; the parity harness hard-skipped bits3 on gfx1100 ("covered on gfx1151 only") |
| MQ3-G256-Lloyd (+ HFQ3-G256 v1 siblings) | `gemm_*_mb4` (4× batch-tile fanout), shipped production kernels | gated `batch ≥ 128 && rows ≥ 4096`, so the shipped 4B artifact never saw it at pp64, and its residual kernel (rows 2560) never saw it at all |

## Measurement 1 — MQ3V2 raw-bit screen on gfx1100 (quarantine test)

`test_mqv2_bt_gfx11` gains `HIPFIRE_MQV2_BT_SCREEN_BITS3=1` to opt gfx1100 into
the bits=3 arms. Result: **every bits=3 arm PASSES raw f32::to_bits equality**
against the per-format base kernels — qkvza BT4/BT12, qkv BT4/BT12, gate_up
BT6/BT12, residual BT4/6/8, at N=128 and N=256, finite/nondegenerate, residual
`Y+=` preserved (61 bits=3 checks in the run log). The gfx1100 quarantine was
therefore precautionary, not measured. **The production quarantine itself is
KEPT**: no shipped artifact loads MQ3G256V2 (the profiled `qwen3.5-4b.mq3`
resolves to the Lloyd family), so there is no e2e case to promote yet; the
screen override is the groundwork for when one exists.

## Measurement 2 — Lloyd mb4 floor matrix (forced vs default)

`HIPFIRE_MQ3_MB4=4` bypasses both floors (batch ≥ 128, rows ≥ 4096); the delta
vs default isolates what each floor was hiding, on `qwen3.5-4b.mq3`:

| batch | forced mb4 | default | Δ | reading |
|---|---|---|---|---|
| 32 | 1989.9 | 2268.9 | **−12.3%** | 4× fanout wastes half the tile below batch 64 |
| 64 | 3512.9 | 3001.1 | **+17.1%** | the rows ≥ 4096 projections want mb4 here |
| 128 | 4145.1 | 3859.6 | **+7.4%** | only the residual kernel (rows 2560) differs — its m-floor was hiding a win |
| 256 | 3894.2 | 3930.6 | −0.9% | residual crossover |
| 512 | 3579.0 | 3922.6 | **−8.8%** | residual mb4 loses at large batch |
| 1024 | 3558.1 | 3882.9 | **−8.4%** | "
| 2048 | 3460.2 | 3783.1 | **−8.5%** | " |

## Policy change

The four Lloyd gates (`gemm_{qkvza,qkv,gate_up}_mq3g256_lloyd_wmma`,
`gemm_mq3g256_lloyd_residual_wmma`) now call a pure helper
`mq3_lloyd_mb4_default(arch, arch_supports, batch, rows)` instead of repeating
the inline `batch >= 128 && rows >= 4096` test:

- **gfx1100**: `rows ≥ 4096` → mb4 from **batch 64** (was 128); rows < 4096 →
  mb4 in **batch 128..=191** only (was never).
- every other arch: unchanged (`batch ≥ 128 && rows ≥ 4096`).
- `HIPFIRE_MQ3_MB4` still dominates everything.
- The HFQ3-G256 (non-Lloyd) sibling gates are **untouched** — same structure,
  but unmeasured on gfx1100.

## Validation (post-change binaries, master `0f999cb4dc` vs pruned branch + policy)

`qwen3.5-4b.mq3` (5-run medians, interleaved):

| metric | master | new policy | Δ |
|---|---|---|---|
| pp32 | 2287.5 | 2270.9 | −0.7% (flat, as designed) |
| pp64 | 2989.1 | **3667.2** | **+22.7%** |
| pp128 | 3860.9 | **4168.2** | **+8.0%** |
| pp2048 | — | — | −0.2% (flat) |
| decode @64 | 209.05 | 216.67 | +3.4% (the branch's decode fusions, unchanged) |

The pp64 gain exceeds the forced-env bundle (+17.1%) because the policy keeps
the residual kernel plain at batch 64 where forced mb4 hurt it.

`qwen3.5-9b.mq3` (edge case: its residual has rows = 4096 exactly, so the
lowered floor newly admits it at batch 64):

| metric | master | new policy | Δ |
|---|---|---|---|
| pp64 | 1742.4 | **2277.8** | **+30.7%** (non-overlapping samples) |
| pp128 | 2324.2 | 2302.8 | −0.9% (flat within noise) |

Engagement proof (prefill profile, 4B @pp64): `gemm_{gate_up,qkvza,qkv}_mq3g256_lloyd_wmma_mb4`
fired; `gemm_mq3g256_lloyd_residual_wmma` stayed on the plain kernel — exactly
the intended per-projection split.

Greedy parity (serve, temp 0, thinking off, 64 tokens, `qwen3.5-4b.mq3`):
**master ≡ branch byte-identical** (and master self-deterministic across
restarts). The mb4 kernels are shipped production code above the old floor; the
newly-admitted ranges produce token-identical output.

## Band validation (post-change, master vs policy branch, 4B mq3)

The policy is banded, so "does the gain carry?" decomposes into per-band
predictions, each now measured (5-run medians, fresh interleaved processes):

| batch | master | branch | Δ | why |
|---|---|---|---|---|
| 48 | 2650.4 | 2650.4 | **+0.0%** | below the 64 floor on both: identical path |
| 64 | 2989.1 | 3667.2 | **+22.7%** | band 1 (rows ≥ 4096 → mb4) |
| 96 | 3125.2 | **3575.1** | **+14.4%** | band 1, interpolated point confirmed |
| 128 | 3860.9 | 4168.2 | **+8.0%** | band 1 + residual band |
| 160 | 3810.6 | **4130.3** | **+8.4%** | residual band, interpolated point confirmed |
| 192 | 3887.3 | 3903.0 | +0.4% | **identity by construction** — bands end at 191 |
| 224 | 3804.0 | 3814.6 | +0.3% | identity by construction |
| 4096 | 3561.1 | 3556.8 | −0.1% | large prefill chunk: identical path |

Non-uniformity is the intended shape, not a defect: the 4× fanout is a
tile-geometry lever. It wins where the batch fills the 64-wide tile grid with
enough rows to amortize (64–191), loses where the tile is half-empty (32) and
where the fat residual kernel already ran better plain (≥ 256 — which is why
upstream's own MQ4V2 BT policy is likewise banded per projection). Outside the
bands the binary is behaviorally identical to master, and the identity rows
(48/192/224) plus 4096 verify that directly. Decode at ctx2048 (+3.1%) is the
branch's decode fusions; mb4 never engages at batch 1.

## Depth validation — 20k / 40k context (master `0f999cb4dc` vs branch)

Does any of it hold at depth? Yes, with the expected shape (product bench,
fresh interleaved processes, 5-run medians, `--pp/--ctx 64,20480,40960`):

4B `qwen3.5-4b.mq3`:

| point | master | branch | Δ |
|---|---|---|---|
| decode @ctx64 | 198.93 | 206.13 | +3.6% |
| decode @ctx20480 | 166.77 | 171.81 | **+3.0%** |
| decode @ctx40960 | 140.67 | 144.37 | **+2.6%** |
| pp64 prefill | 2942.8 | 3602.1 | +22.4% (the mb4 policy) |
| pp20480 prefill | 2424.9 | 2415.6 | −0.4% (policy identical at 512-chunks) |
| pp40960 prefill | 1729.3 | 1726.8 | −0.1% (") |

4B `qwen3.5-4b.mq4` (the decode-fusion artifact):

| point | master | branch | Δ |
|---|---|---|---|
| decode @ctx64 | 192.73 | 203.76 | +5.7% |
| decode @ctx20480 | 162.96 | 171.13 | **+5.0%** |
| decode @ctx40960 | 138.16 | 144.52 | **+4.6%** |
| pp20480 / pp40960 prefill | — | — | +0.2% / +0.1% (flat) |

Reading: the decode gains survive deep KV with mild attenuation
(+5.7 → +4.6% and +3.6 → +2.6% from ctx 64 to 41k) — the per-token attention
cost grows in both arms equally (mq4 decode drops 193 → 138 tok/s absolute),
and the branch's fusions shave the same fixed per-token work. Long prefill at
depth is flat for both artifacts: prefill GEMMs run in 512-token chunks where
the two binaries dispatch identically by construction, and attention — which
grows with depth — is untouched shared code. Nothing regresses at depth.

## What this does not change

- MQ3G256V2 gfx1100 production quarantine (kept; see Measurement 1).
- HFQ3-G256 v1 mb4 floors (kept pending their own A/B).
- Non-gfx1100 archs (kept pending their own A/B; gfx1151 already promotes
  MQ{2,3,5,6}V2 BT via `mqv2_gfx11_bt_admitted`).
- Decode paths (mb4 is prefill-only; decode deltas above are the branch's
  existing decode fusions).
