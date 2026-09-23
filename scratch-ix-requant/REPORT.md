# Phase IX requantization — Qwen3.8-27B MQ4V2 XT

## Decision

Keep **Arm C**, the corrected linear-attention head-order artifact, as this branch's candidate. It improves every measured full-WT2 route against the shipped QAT artifact, but **does not meet the iu4 c24 target <0.05**: c2 is 0.075777. Do not promote Arm A+C: its synthetic imatrix-RMS activation surrogate degrades both early screens. No full c24 was run for that failed arm. These artifacts are AWQ-only requants of the BF16 parent; neither contains the shipped QAT r7s200 weight update.

- Arm C: `/home/kaden/qcal/qat/v25x/qwen3.8-27b.mq4v2.xt.sym-a035.la-headfix.hfq`
- Arm C SHA-256: `58c43c6373e287176c52ab53177a71d952c3f0f5df3fe66ab4999aeec197c456`
- Arm A+C (rejected): `/home/kaden/qcal/qat/v25x/qwen3.8-27b.mq4v2.xt.sym-a035.la-headfix-a4c2.hfq`
- Arm A+C SHA-256: `4481efaf9d6b874b2ffca7a559b831994e44d1fc1b58515e95055eaa10bb5518`

## WT2 prefill KLD

All evaluations used `--ref /home/kaden/kldrefs/qwen3.8-27b.ref_wt2.bin --kv-mode fp8 --kv-v q8 --scoring-mode prefill` on gfx1201, with `--max-chunks 1`, `4`, or `24`. The c2 runs used `HIPFIRE_HIPCC_EXTRA_FLAGS=-DIU4_A4_CANDIDATES=2`; RTN used the default gfx1201 route with that flag absent; native fp8 used `HIPFIRE_IU4_PREFILL=0`. The shipped QAT baseline is `/home/kaden/qcal/qat/v25/qwen3.8-27b.mq4v2.xt.sym-a035.qat-r7s200.hfq`.

| Arm | c2 c1 | c2 c4 | c2 c24 | RTN c24 | native fp8 c24 |
|---|---:|---:|---:|---:|---:|
| Shipped sym-a035 QAT r7s200 | 0.049130 | 0.055624 | 0.076071 | 0.083278 | 0.049580 |
| Plain sym-a035 AWQ-only | 0.052389 | 0.054780 | — | — | — |
| **C: fix LA out_proj head order** | **0.049649** | **0.051662** | **0.075777** | **0.081473** | **0.046730** |
| A+C: additionally search c2/symmetric alpha with imatrix RMS | 0.053125 | 0.054813 | not run: gate failed | not run | not run |

Arm C improves c2 c4 by 0.003962 versus shipped QAT, c2 c24 by 0.000294, RTN c24 by 0.001805, and native fp8 c24 by 0.002850. There is no native-fp8 quality regression. The route/weight layout is unchanged; this work did not measure an ABBA runtime latency comparison for Arm C. Evaluation outputs are under `scratch-ix-requant/` with prefixes `headfix-c2`, `headfix-rtn`, `headfix-fp8`, and `headfix-a4c2-c2`.

## Recipe and correctness

The plain sym-a035 recipe was byte-reproduced before changing the imatrix order (SHA-256 original and local reproduction both `de0cda5b406d9ce14b5bb398eff201ea7be56e4c51be53cd56be20098fc01254`). Arm C uses the same BF16 parent, `--format mq4v2 --tier xt --imatrix /home/kaden/qcal/imatrix/Qwen3.8-27B-imatrix.gguf --awq-alpha 0.35 --mq4v2-symmetric --threads 16`, plus `--awq-fix-la-head-order`. Arm A+C additionally uses `--awq-a4-route-c2`. The signed-capture option is implemented but was not used in either quantified arm; no 64-block capture was generated.

`--awq-fix-la-head-order` is a **real calibration bug fix**, not an alpha or post-hoc tensor tweak. The imatrix stored all 48 `blk.N.ssm_out.weight` vectors in LA V-head component-major order, whereas the HF `linear_attn.out_proj.weight` consumes head-major order. For each of the 48 length-6144 vectors, the corrected entry is `fixed[h*128+c] = raw[((h%3)*16+h/3)*128+c]`. Only the 48 LA out_proj AWQ sidecars change; weights are pre-scaled using those corrected sidecars before FWHT/4-bit quantization. **Every future Qwen3.8 MQ4 requant should default to this correction**, even though the current CLI keeps it opt-in to preserve historical recipe reproducibility.

RotScreen's independent `check_headfix.py` passed: captured `corr(log AWQ scale, log producer RMS)` is 0.772 (block 0) and 0.827 (block 60), versus 0.005 and 0.018 for the shipped raw-order artifact; FA block 31 control is 0.880. All 48 corrected LA sidecars have minimum correlation 1.00000 with the HF-reindexed imatrix and all non-out_proj sidecars are byte-identical to the plain reproduced artifact. Reconstructed unrotated W matches parent W pre-scaled by those sidecars with relative MSE 0.0106/0.0107 on blocks 0/60. The checker exits 0 with `RESULT: PASS`.

The optional route-aware AWQ objective uses the gfx1201 c2 A4 scale selection and the symmetric qt44 W writer's four fp16-rounded grids; scoped CPU/GPU fixtures matched c2 activation nibbles/scale, and scoped Rust quantization tests passed. Arm A+C's fallback activation is an unsigned imatrix RMS representative, not real signed producer rows; its c1/c4 failure is an empirical rejection of that fallback for this model, **not** evidence that the corrected-head Arm C failed. `cargo build -p hipfire-quantize --release` passed; no project-wide suite or formatter ran.
