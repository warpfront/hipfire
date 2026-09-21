# gfx1201 rig-to-daemon inversion diagnosis

## Verdict

**Final diagnosis:** the original daemon gates did not prove that their experimental flags reached the daemon. A native client sends a `configure` message first; the daemon installs that `ProcessConfig`, and all subsequent `developer_var`/feature-flag reads consult the installed snapshot, not the daemon's ambient environment. The shift-fold replay saved in `wt-iu4free/scratch-iu4free/replay-fullpow2.jsonl` contains none of the shift-fold or row-global keys. Both original rocprof databases therefore contain only `..._symfold` symbols. They are control/control, not a candidate/control comparison.

A second routing trap matters for this experiment. Once the producer-fusion opt-outs are put in their correct typed keys, the default `kernel.gfx12_fp8_stream=true` routes the workload to FP8v2. An honest row-global IU4 replay therefore also needs `kernel.gfx12_fp8_stream=false`. With all required keys embedded, the candidate trace contains `..._shiftfold` plus `quantize_int4_mmq_row`, while the control contains `..._symfold` plus the same row quantizer.

The corrected result resolves the apparent contradiction without a hidden daemon wall:

* the standalone rig measured the wrong shapes/epilogue: one `full_set` M=34816 gate/up launch and `full_set` down;
* the daemon runs gate and up as separate M=17408 `full_set` launches and down as `full_add`;
* at the daemon shapes, shift-fold is not faster: the corrected daemon GEMM family is **193.094 -> 193.708 us/token, +0.615 us/token (+0.318%, slower)**;
* pp8192 is neutral/noisy (Card-C paired wall -0.422% throughput; independent Card-E paired wall +0.126%);
* 5909-token TTFT is a real fail because the candidate's 256-row batch tile pads 5909 to 6144 rows, while control's 128-row tile pads only to 6016. The stronger independent Card-E client pair is **1849.956 -> 1883.053 ms, +33.097 ms (+1.789%)**.

Therefore the original shift-fold kill was reached from invalid route evidence, but the corrected current implementation is still **do not ship**. The route/config diagnosis is final. Campaign verdicts without route proof are provisional until re-gated.

## Reproduction identity

* Card-C only for the primary measurements: `GPU-085289909a86cc63`; HOME `ab2`.
* Candidate binary: `wt-iu4free/target/release/daemon`, source HEAD `463731a6f3a0268b9ee5cebc33fc9b9ad92e907c`, binary md5 `e0659c8e702f6f905888f83dcf10eb19`.
* Artifact: `/home/kaden/qcal/qat/v25/qwen3.8-27b.mq4v2.xt.sym-pow2h-a035.hfq`, sha256 `febf70570354ccf8867ae2c7330da0dd80b4263e3ce2603f724abd62eccaf7c8`.
* Correct replays: `replay-control.jsonl`, `replay-candidate.jsonl`.
* Correct traces: `prof-control-correct/inv_kernel_{stats,trace}.csv`, `prof-candidate-correct/inv_kernel_{stats,trace}.csv`.
* Every profile contains two complete pp8192 dispatch sets; device totals below divide nanoseconds by 16,384 tokens.

The exact ProcessConfig differences are strings for developer keys and booleans for registered keys:

```json
{
  "hardware.devices": "GPU-085289909a86cc63",
  "experimental.graph.forward": true,
  "kernel.iu4_prefill": true,
  "kernel.gfx12_silu_quant_fused": false,
  "kernel.gfx12_producer_quant_fused": false,
  "kernel.gfx12_fp8_stream": false,
  "developer.iu4_global_a_screen": "1",
  "developer.iu4_shift_foldfree": "0 or 1"
}
```

Using `developer.iu4_prefill`, `developer.gfx12_silu_quant_fused`, or `developer.gfx12_producer_quant_fused` is also wrong: these names are registered typed fields, so `legacy_value` resolves their typed key instead of a same-suffix developer key.

## Why daemon ambient flags disappear

The source path is deterministic:

1. `crates/hipfire-daemon/src/main.rs:610-678` reads the first stdin message. If it is `configure`, lines 647-655 parse its supplied `ProcessConfig`. Only the non-configure fallback at lines 657-658 calls `load_local_process_config()`.
2. `main.rs:757-768` installs that snapshot before GPU construction.
3. `crates/hipfire-config/src/lib.rs:3442-3445` makes `process_value` read the installed/local `ProcessConfig`; it does not call `getenv`.
4. `lib.rs:3587-3595` implements `developer_var` through `process_value`.
5. `lib.rs:4356-4394` snapshots ambient long-tail variables into `developer.*`, but only when `load_env_layer` is actually used. A configured daemon does not merge its child environment afterward.
6. `crates/rdna-compute/src/gemm.rs:19712-19729` chooses symfold versus shiftfold through `developer_var("HIPFIRE_IU4_SHIFT_FOLDFREE")`; hence the configure snapshot controls dispatch.

Ambient variables still used before/around ProcessConfig are bootstrap concerns, not kernel policy: `HIPFIRE_LOG`/`RUST_LOG`, `HIPFIRE_LOG_FORMAT`, `ROCR_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES`, path variables such as `HOME`, and the parent-side `HIPFIRE_DAEMON_BIN`. Setting a kernel experiment only in the daemon wrapper environment is ineffective after a configure message.

A standalone utility can appear to honor ambient variables because it calls `load_local_process_config`, which snapshots the environment before installing a local ProcessConfig. Direct HIP rigs may instead select code at compile time. Neither behavior implies that a configured daemon reads ambient policy live.

## Campaign flag classification

All policy flags in this table must be present in the client-generated ProcessConfig for a configured native daemon. “Typed” means use the schema key and JSON boolean/enum. “Developer” means use `developer.<lowercase suffix>` and usually a string value.

| legacy spelling | ProcessConfig key | class | configured daemon |
|---|---|---|---|
| `HIPFIRE_GRAPH` | `experimental.graph.forward` | typed | configure only |
| `HIPFIRE_NORMALIZE_PROMPT` | `prompt.normalize` | typed | configure only |
| `HIPFIRE_IU4_PREFILL` | `kernel.iu4_prefill` | typed | configure only |
| `HIPFIRE_GFX12_SILU_QUANT_FUSED` | `kernel.gfx12_silu_quant_fused` | typed | configure only |
| `HIPFIRE_GFX12_PRODUCER_QUANT_FUSED` | `kernel.gfx12_producer_quant_fused` | typed | configure only |
| `HIPFIRE_GFX12_FP8_STREAM` | `kernel.gfx12_fp8_stream` | typed | configure only |
| `HIPFIRE_ATTN_QRESIDENT` | `kernel.attn_qresident` | typed | configure only |
| `HIPFIRE_ATTN_QK8` | `kernel.attn_qk8` | typed | configure only |
| `HIPFIRE_ATTN_PV8` | `kernel.attn_pv8` | typed | configure only |
| `HIPFIRE_IU4_SYMFOLD` | `developer.iu4_symfold` | developer | configure only |
| `HIPFIRE_IU4_SHIFT_FOLDFREE` | `developer.iu4_shift_foldfree` | developer | configure only |
| `HIPFIRE_IU4_WPRESHUFFLE` | `developer.iu4_wpreshuffle` | developer | configure only |
| `HIPFIRE_IU4_GLOBAL_A_SCREEN` | `developer.iu4_global_a_screen` | developer | configure only |
| `HIPFIRE_FP8_SYMFOLD` | `developer.fp8_symfold` | developer | configure only |
| `HIPFIRE_FP8_FOLDFREE` | `developer.fp8_foldfree` | developer | configure only |
| `HIPFIRE_FP8_WPRESHUFFLE` | `developer.fp8_wpreshuffle` | developer | configure only |
| `HIPFIRE_FP8_FRAGMENT_ORDER` | `developer.fp8_fragment_order` | developer | configure only |
| `HIPFIRE_A4_ROWGLOBAL` | `developer.a4_rowglobal` | developer | configure only |
| `HIPFIRE_A4_GROUP_K` / experimental `HIPFIRE_A4_GROUP` | matching `developer.*` key used by that branch | developer | configure only |
| `HIPFIRE_VERIFY_GRAPH` | `developer.verify_graph` | developer | configure only |

There is no campaign policy flag in this list that is intentionally “standalone-only.” The apparent category came from different ProcessConfig bootstrap paths or compile-time rig selection.

## Original route proof failure

The original replay's configure object has no relevant experimental keys. Direct inspection of both original profile databases shows only:

* `gemm_mq4g256v2_residual_mmq_iu4_full_set_symfold`
* `gemm_mq4g256v2_residual_mmq_iu4_full_add_symfold`

There are zero shiftfold dispatches. The old 16 x 512 projected chunks were another symptom of missing configured policy, not evidence that production pp8192 uses a 512-row GEMM batch. In the corrected trace every GEMM has Grid Y 64 for symfold or 32 for shiftfold, proving B=8192 with 128- or 256-row batch tiles.

## Corrected complete per-kernel profile

Calls cover two pp8192 passes. Delta is candidate minus control; negative is faster.

| logical kernel | calls | control us/tok | candidate us/tok | delta us/tok |
|---|---:|---:|---:|---:|
| IU4 `full_set` (symfold -> shiftfold) | 736 | 131.362 | 132.288 | +0.926 |
| IU4 `full_add` (symfold -> shiftfold) | 256 | 61.732 | 61.421 | -0.311 |
| attention packet | 32 | 24.025 | 22.806 | -1.219 |
| fused silu/rotate F32 producer | 128 | 22.069 | 22.207 | +0.138 |
| `quantize_int4_mmq_row` | 512 | 21.008 | 21.052 | +0.045 |
| fused rmsnorm/rotate F32 producer | 256 | 8.690 | 8.801 | +0.111 |
| GDN chunk scan | 1536 | 9.101 | 8.482 | -0.619 |
| gated norm F32 | 96 | 5.874 | 5.871 | -0.003 |
| GDN chunk prep | 96 | 5.278 | 5.356 | +0.078 |
| rotate X | 128 | 4.904 | 4.909 | +0.005 |
| deinterleave/rmsnorm | 32 | 2.732 | 2.795 | +0.063 |
| GDN KKT solve | 1536 | 2.095 | 1.954 | -0.140 |
| sigmoid multiply | 32 | 1.905 | 1.905 | -0.000 |
| RoPE | 32 | 0.714 | 0.744 | +0.029 |
| rmsnorm | 34 | 0.441 | 0.418 | -0.023 |
| KV write | 64 | 0.304 | 0.404 | +0.100 |
| fillBuffer | 2016 | 0.229 | 0.252 | +0.023 |
| GEMV | 2 | 0.133 | 0.134 | +0.001 |
| embedding | 2 | 0.037 | 0.030 | -0.007 |
| copyBuffer | 180 | 0.030 | 0.031 | +0.001 |
| rotate helper | 2 | 0.000 | 0.000 | -0.000 |
| **all device kernels** | | **302.663** | **301.859** | **-0.803** |
| **converted GEMMs only** | | **193.094** | **193.708** | **+0.615** |

The experimental symbols demonstrably ran, but the converted family regressed. Candidate total device time improves only because unrelated attention/GDN timing differs by more than the GEMM loss. This is run variance, not credit to shift-fold.

## Symbol, launch, and shape mapping

For these kernels Grid X is global threads: 256 threads/workgroup and one workgroup per 128 output rows, so output M = Grid X / 2. The common B=8192 gives Grid Y=64 for symfold (128 rows/block) and 32 for shiftfold (256 rows/block).

| daemon role | entry | M | K | calls/pass | control -> candidate us/tok |
|---|---|---:|---:|---:|---:|
| LA beta + alpha | full_set | 128 | 5120 | 96 | 0.703 -> 1.333 |
| FA K + V | full_set | 1024 | 5120 | 32 | 1.374 -> 1.390 |
| LA z | full_set | 6144 | 5120 | 48 | 12.089 -> 11.873 |
| LA qkv | full_set | 10240 | 5120 | 48 | 20.042 -> 19.704 |
| FA q/gate packed projection | full_set | 12288 | 5120 | 16 | 8.044 -> 7.889 |
| FFN gate and up, separate launches | full_set | 17408 | 5120 | 128 | 89.110 -> 90.099 |
| LA/FA output projections | full_add | 5120 | 6144 | 64 | 16.291 -> 16.959 |
| FFN down | full_add | 5120 | 17408 | 64 | 45.441 -> 44.462 |

The candidate flag changes the generic IU4 entries, so qkv/qkvza launches also carry the shiftfold symbol. The earlier “only gate/up and down converted” assumption was not true for this build.

The standalone rig instead used:

* gate/up: one `full_set` M=34816, K=5120 launch;
* down: `full_set` M=5120, K=17408.

A shape-matched standalone check on Card-C at B=8192 gives:

| exact daemon shape/entry | symfold | shiftfold | result |
|---|---:|---:|---:|
| gate or up, `full_set`, M17408 K5120 | 5239.812 us | 5462.217 us | candidate +4.24% time |
| down, `full_add`, M5120 K17408 | 5240.696 us | 5328.355 us | candidate +1.67% time |

Measured occupancy from the HIP API is 3 resident blocks/CU for the 20 KiB control and 2 for the 30 KiB candidate. This can contribute to the shape crossover, but no daemon-only occupancy interaction is needed to explain it: the exact kernels already lose in isolation.

## Arithmetic ceiling and baseline sanity

The original standalone aggregate claimed 141.087 -> 131.680 us/token, a 9.407 us/token saving. If that saving applied to the corrected 302.663 us/token kernel pass, the ceiling would be about **3.11% latency**. It did not apply because it came from different M and epilogue shapes.

The corrected converted GEMMs are 63.8% of device kernel time and regress by 0.615 us/token. Their arithmetic prediction is only **+0.203% pass latency**, consistent with a neutral/slightly negative pp8192 gate.

The earlier “crippled baseline made GEMM irrelevant” is not the primary explanation. In the corrected row-global route, GEMM remains 193.094 / 302.663 = **63.8%** of device time. Even the reported 61.3 us/token producer cost would still leave GEMM large enough that a real 9.407 us/token saving should be visible. The saving simply is not present at production shapes.

For comparison, the shipped profile summary (about 197 GEMM, 39.5 producers, 24.1 attention, 16.6 GDN us/token) puts GEMM near 71%. Row-global reduces its share but does not create a wall capable of erasing a true multi-percent win.

## Kernel sum versus wall

Using the second, warm pp8192 dispatch from each rocprof process:

| arm | kernel sum us/tok | warm wall us/tok | wall - kernel us/tok |
|---|---:|---:|---:|
| control | 302.663 | 305.459 | 2.796 |
| candidate | 301.859 | 304.513 | 2.654 |

The total-kernel change (-0.803 us/token) and warm-wall change (-0.946 us/token) agree within 0.143 us/token. There is no missing-time bucket and no evidence that saved GEMM time reappears in host/launch gaps. The first candidate profile pass included fresh candidate JIT and is deliberately excluded from the warm comparison.

A non-profiled Card-C C-X-C-X sequence, six measured pp8192 calls per process after warmup, gave control medians 2512.025/2510.730 ms and candidate medians 2524.410/2519.650 ms. Pair means are **2511.378 -> 2522.030 ms**, candidate +0.424% latency (-0.422% throughput). An independent Card-E ABBA was +0.126% throughput. The correct pp8192 conclusion is neutral/noise, not a multi-percent GEMM win.

## TTFT and the 256-row tail

At 5909 tokens:

* control: ceil(5909/128) = 47 batch blocks, covering 6016 rows;
* candidate: ceil(5909/256) = 24 batch blocks, covering 6144 rows.

The candidate computes 128 more padded rows for every converted GEMM. Card-C direct 5909-prefill C-X-C-X pair means were **1845.568 -> 1874.373 ms**, +28.805 ms (+1.560%). Card-C client TTFT had high first-pair variance; pooled eight-sample medians were **1840.104 -> 1859.549 ms**, +19.445 ms (+1.057%). The stronger independent Card-E C-X-X-C client run (eight samples after two warmups per arm) measured **1849.956 -> 1883.053 ms**, +33.097 ms (+1.789%). This is a real tail-geometry failure, not a host wall.

## Campaign verdict audit

| campaign result | route evidence in its report/artifacts | verdict now |
|---|---|---|
| IU4 shift-fold original pp/TTFT/profile | original control and candidate profiles both symfold; configure omitted flags | **void**; replaced by corrected no-ship result above |
| IU4 W-preshuffle pp8192 -0.32%, TTFT +5.24 ms, decode kill | report has standalone ISA/resources but no paired daemon route trace; baseline was an environment override | **at risk; re-gate**. Standalone +1.27% remains valid |
| FP8 fold-free daemon +1.21% | candidate trace exists, but no paired control route proof; control relied on `...=0` | **at risk; re-gate**. Standalone numbers remain valid |
| folded FP8 W-preshuffle daemon +2.72%, decode -15% | no final paired route proof in the report; environment-controlled arms | **at risk; re-gate before using the kill** |
| fold-free + W-preshuffle 186.1 TF/s | standalone only; daemon gate was not claimed | standalone result valid, no daemon verdict |
| A4 row-global performance kill | report proves the row quantizer in at least one profile, but saved replay/trace does not prove both timing arms' configure payload | mechanism/quality proof valid; **paired performance verdict should be re-gated or raw configure logs recovered** |
| A4 K512 rerun | post-discovery replay embeds `developer.a4_group="512"` and trace shows the K512-specific producer | corrected rerun is route-proven; pre-fix env-only samples are not |
| attention Q-resident +1.344% | report uses typed `kernel.attn_qresident=true` and candidate-specific symbol trace | **stands** |
| attention QK8/PV8 | report states configured replay and distinguishes all four route symbols | **stands**; already-hardwired conclusion unaffected |

The audit rule is strict: a benchmark report proves that a candidate ran only when an arm-specific symbol/grid/resource signature appears, or when the exact first configure message is retained and the dispatch is otherwise unambiguous. Environment assignment in a shell command is not route proof.

## Copy-paste verification protocol

1. Capture the daemon protocol or edit a replay's first `configure` object. Put registered switches at typed keys and long-tail switches at developer keys exactly as in the JSON above. Do not rely on the daemon wrapper environment.
2. Pin the UUID both in outer visibility and `hardware.devices`.
3. Warm a fresh build once. Then profile two complete prefills:

```bash
env HOME=/home/kaden/.hipfire-homes/ab2 \
  ROCR_VISIBLE_DEVICES=GPU-085289909a86cc63 \
  HIP_VISIBLE_DEVICES=GPU-085289909a86cc63 \
  HIPFIRE_KERNEL_CACHE=/home/kaden/.hipfire-homes/ab2/.hipfire_kernels \
  HIPFIRE_MODELS_DIR=/home/kaden/.hipfire/models \
  rocprofv3 --kernel-trace --stats -f csv -d scratch-inv/prof-candidate-correct -o inv -- \
  bash -c 'exec /path/to/daemon < scratch-inv/replay-candidate.jsonl'
```

4. Before accepting performance, assert all of the following from the CSV:
   * control has nonzero `full_set_symfold` and `full_add_symfold`, and zero shiftfold;
   * candidate has nonzero `full_set_shiftfold` and `full_add_shiftfold`, and zero symfold;
   * both have 512 `quantize_int4_mmq_row` calls over two prefills;
   * neither arm dispatches the FP8v2 WMMA GEMM family;
   * pp8192 Grid Y is 64 control versus 32 candidate;
   * normalize by the number of complete dispatch sets, not the number of protocol commands sent.
5. Archive the first configure line beside every gate result. Treat a missing arm-specific dispatch signature as a failed experiment, not neutral performance.

## Final status

* **Why the contradiction existed:** final — invalid daemon flag propagation plus rig/daemon shape mismatch.
* **Is there a hidden scheduling/dependency wall?:** no evidence; corrected kernel sum tracks warm wall.
* **Current shift-fold implementation:** final no-ship — pp8192 neutral, production GEMM family slightly slower, 5909 TTFT clearly worse from padding.
* **Other environment-only daemon verdicts:** provisional until their configure payload and candidate dispatch are proven.
