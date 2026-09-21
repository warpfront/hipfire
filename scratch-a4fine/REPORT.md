# gfx1201 IU4 activation quantizer screen

## Verdict

**KILL all tested changes; ship nothing.**

- Restoring two fused-producer scale candidates (`N=2`) is the only quality result that clears the serving-reference threshold: agentic c24 is **0.196354**, 3.85% below the exact shipped control **0.204209**, and WT2 c24 is **0.076071**, below the 0.10 budget.
- That win is too expensive. Candidate search raises the production producer subtotal from **39.654 to 45.350 us/token (+14.37%)**. Card-B ABBA throughput is **-1.26% at pp512** and **-2.40% at pp8192**, and TTFT is **+2.08%**. Both pp8192 and TTFT fail their gates. The GEMM family is unchanged within noise (+0.206 us/token), which localizes the loss to the producers.
- K64 and K32 improve the rebuilt evaluator's self-control, but neither improves the exact shipped agentic control by the required 2%: K64 is **0.207819** (+1.77%, worse) and K32 is **0.201110** (-1.52%). They therefore do not enter the daemon gate.
- The finer-scale premise is not producer-only. The existing format and gfx1201 GEMM fold once per K128. Uniform K64/K32 requires 2/4 `(d,s)` headers per K128 and 2x/4x consumer fold cadence. The compiled consumer spills heavily: K64 full-set/full-add use 256 VGPR with 32 spill slots; K32 rises to 255/256 spill slots and roughly 1 KiB private storage per lane. Warm full-reference evaluator time is about 81 s for K64 and 275 s for K32 versus 66 s for K128.

The default kernel sources remain byte-identical to the shipped sources. Experimental finer formats live in separate source files and are selected only by `developer.a4_group_k`; candidate count is a per-IU4-producer compiler flag selected by `developer.a4_candidates`. Default behavior is unchanged.

## Reproduction identity and methodology

- Source base: `34ccae74b`; experimental commits: `5acfb4707`, `8b317bbef`.
- Model: `/home/kaden/qcal/qat/v25/qwen3.8-27b.mq4v2.xt.sym-a035.qat-r7s200.hfq`.
- Model SHA-256: `de8ee8256033c3690b0f1a2aff14e77cc88fff490e118648b04a833a3f2969b5`.
- KLD card: `GPU-085289909a86cc63`, HOME `ab2`.
- Perf/profile card: `GPU-e475645fe0200397`, HOME `ab1`.
- Every daemon measurement used a persisted/configured process value or an explicit replay configure object; ambient kernel-policy variables were not used as daemon proof.
- The pp8192 profiles contain two complete 8192-token dispatch sets. Durations are divided by 16,384 tokens.

### Baseline caveat

The pristine shipped evaluator reproduces the required agentic c24 control exactly: **0.204209**. Its WT2 c24 reference is **0.082527**.

A rebuilt evaluator from the experimental worktree, even after restoring the default quantizer and consumer source files byte-for-byte, gives K128/N1 **0.083278 WT2 / 0.215471 agentic**. This rebuilt-binary drift is not accepted as the shipped baseline. Both baselines are shown below: the experimental self-control checks direction within one binary, while pass/fail is computed against the exact shipped control as required.

## What the candidate knob actually controls

`IU4_A4_CANDIDATES` affects only `quantize_block_i4_128_wave<true>`, used by the fused gfx1201 producers:

- N=1: RTN, `d = amax / 7`.
- N=2: the existing `{5,7}` candidate subset.
- N=4: the existing `{0,2,5,7}` subset.

The standalone `quantize_int4_mmq_ds128` calls `quantize_block_i4_128_wave<false>` and retains its original eight-candidate search for every N setting. Thus N=2/N=4 are fused-producer-only changes; the standalone path does not silently change candidate count.

## Finer-window ownership and format

All current producers can compute finer amax windows without changing launch geometry because each quantizing wave already owns a K128 block:

| Producer family | Existing ownership | K64/K32 reduction |
|---|---|---|
| RMSNorm/FWHT fused producer | 8 floats/lane remapped to two K128 blocks | 16-lane / 8-lane subwave |
| SwiGLU/FWHT fused producer | one wave owns a K256 slice | 16-lane / 8-lane subwave |
| rotate/AWQ and sigmoid twin | one wave owns a K256 slice | 16-lane / 8-lane subwave |
| gated norm/FWHT | wave 0 emits two K128 blocks | 16-lane / 8-lane subwave |
| standalone quantizer | one wave owns one K128 | 16-lane / 8-lane subwave |

The outer index remains K128 and the nibble payload remains 64 bytes. Block size changes from 72 bytes at K128 to 80 bytes at K64 and 96 bytes at K32. The consumer must load and fold 2/4 metadata pairs. This consumer change, not launch ownership, is the limiting mechanism.

## Quality ladder

All numbers use the requested QAT artifact. Lower is better.

| Activation window | Fused candidates | WT2 c2 | WT2 c24 | Agentic c2 | Agentic c24 | Agentic delta vs exact shipped 0.204209 | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| K128 shipped binary | 1 | — | **0.082527** | — | **0.204209** | — | exact control |
| K128 rebuilt self-control | 1 | 0.065213 | 0.083278 | 0.212926 | 0.215471 | +5.52% | diagnostic only |
| K64 | 1 | 0.057371 | 0.078027 | 0.239362 | 0.207819 | **+1.77%** | kill: worse than shipped |
| K32 | 1 | 0.053966 | 0.070896 | 0.140563 | 0.201110 | **-1.52%** | kill: below 2% bar |
| K128 | 2 | 0.061678 | 0.076071 | 0.156847 | **0.196354** | **-3.85%** | quality survivor |
| K128 | 4 | 0.058558 | 0.077228 | 0.175945 | 0.208730 | **+2.21%** | kill: worse than shipped |

Against the rebuilt self-control, K64/K32/N2/N4 agentic c24 deltas are -3.55%, -6.66%, -8.87%, and -3.13%, respectively. The exact shipped baseline remains the decision baseline because it is the only control that reproduces the required artifact result.

## Consumer resource signature for finer windows

Radiowave manifests from the KLD card provide an unambiguous compiled-route signature:

| Window | Consumer cache key | full-set VGPR | full-set spill slots | full-set private bytes/lane | full-add spill slots | Other evidence |
|---|---|---:|---:|---:|---:|---|
| K128 | `e10958f04968623c` | 182 | 0 | 0 | 0 | shipped fold cadence |
| K64 | `43731899a4150a8d` | 256 | 32 | 100 | 32 | core entry: 68 spills, 108 private bytes/lane |
| K32 | `81ce23dfc5929a6b` | 256 | 255 | 972 | 256 | core entry: 514 spills, 1172 private bytes/lane |

This is why the apparently small metadata plane is not near-free in the current consumer dataflow. K64 doubles an already dominant fold; K32 quadruples it and produces extreme scratch traffic.

## Candidate-N2 daemon gates

### Decode-first screen

| Arm | pp512 tok/s | tg128 tok/s | Candidate delta |
|---|---:|---:|---:|
| N1 control | 3402.10 | 36.5967 | — |
| N2 candidate | 3280.90 | 36.5605 | pp512 -3.56%; tg128 -0.10% |

### Card-B ABBA matrix

| Pair | Arm | pp512 | pp8192 | tg128 |
|---|---|---:|---:|---:|
| 1 | N1 control | 3326.7 | 3670.7 | 36.5143 |
| 1 | N2 candidate | 3293.6 | 3554.2 | 36.4684 |
| 2 | N2 candidate | 3293.2 | 3566.6 | 36.4805 |
| 2 | N1 control | 3344.0 | 3625.1 | 36.4448 |
| mean | N1 control | **3335.35** | **3647.90** | **36.4796** |
| mean | N2 candidate | **3293.40** | **3560.40** | **36.4745** |
| delta | candidate/control | **-1.26%** | **-2.40%** | **-0.014%** |

The candidate fails the pp8192 no-worse-than-1% gate.

### TTFT

5,909-token prompt, two warmups and eight measured runs:

| Arm | Median TTFT | Prompt tok/s |
|---|---:|---:|
| N1 control | 1644.782 ms | 3592.574 |
| N2 candidate | 1679.016 ms | 3519.325 |
| delta | **+2.081%** | **-2.039%** |

The candidate narrowly but clearly fails the +2% TTFT gate.

### Kernel profile and route proof

Both traces contain 736 IU4 `full_set_symfold` and 256 IU4 `full_add_symfold` calls, and no FP8 projection GEMM or FP8 activation packer. Each contains the four fused IU4 producer symbols at the expected counts: RMSNorm 256, SiLU 128, gated norm 96, sigmoid/rotate 32. The replay configure objects are archived as `replay-control.jsonl` and `replay-candidate.jsonl`; the latter contains `developer.a4_candidates="2"`.

The candidate compiler log explicitly records `-DIU4_A4_CANDIDATES=2` for all four fused producers. The route also has an arm-specific resource signature in the trace: `gated_norm_mq_rotate_awq_i4_gfx12` is 40 trace VGPR in control and 48 in candidate. A representative Radiowave manifest changes RMSNorm from 90 VGPR/34 SGPR to 91 VGPR/33 SGPR with no spills.

| Profile subtotal | N1 control us/token | N2 candidate us/token | Delta |
|---|---:|---:|---:|
| Producer campaign subtotal | **39.654** | **45.350** | **+5.696 (+14.37%)** |
| IU4 GEMM family | 199.488 | 199.694 | +0.206 (+0.10%) |
| All device kernels | 272.695 | 278.586 | +5.890 (+2.16%) |

The producer subtotal includes fused SiLU, RMSNorm, gated-norm and sigmoid producers plus GDN prep, deinterleave, standalone RMSNorm, and rotate, matching the campaign definition. The measurement directly confirms the mechanism: restored candidate search costs producer time; the consumer is unchanged.

The full serve battery was not run after the candidate had already failed both hard performance gates. No behavioral shipping claim is made.

## Artifacts

- KLD outputs/logs: `isolated_*` and `clean_*` files in this directory.
- Matrix and TTFT JSON: `decode-*.json`, `matrix-*-pair*.json`, `ttft-*.json`.
- Route replays: `replay-control.jsonl`, `replay-candidate.jsonl`.
- Kernel traces/stats: `prof-control/`, `prof-candidate/`.

## Final gate accounting

| Gate | Result |
|---|---|
| Exact control reproduction | pass only with pristine shipped evaluator: 0.204209 agentic c24 |
| WT2 c24 <= 0.10 | all cells pass |
| Agentic improvement >= 2% | only N2 passes (-3.85% vs shipped) |
| tg128 no worse than 1.5% | N2 passes (-0.014%) |
| pp8192 no worse than 1% | **N2 fails (-2.40%)** |
| TTFT no worse than +2% | **N2 fails (+2.081%)** |
| Producer cost | **N2 fails mechanism screen (+14.37%)** |
| Battery | not run after hard perf kill |

**Final recommendation: preserve the screen as evidence, but do not merge or ship any activation-window or candidate-count change.**
