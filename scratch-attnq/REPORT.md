# gfx1201 register-resident Q prefill attention — final verdict

**SHIP.** The Q-resident route clears the production gate on both eligible end-to-end metrics: two independent typed-config TTFT pairs improve by **+1.865%** and **+3.153%** (pooled **+2.507%**), and the card-A ABBA pp8192 pairs improve by **+1.727%** and **+1.860%** (pooled **+1.793%**). No pp512, pp8192, or tg128 row regresses by 1%; tg128 is effectively flat; both KLD routes pass; and the mandatory five-prompt battery is coherent. Commit `1492b6cb2` therefore makes the narrowly admitted route the gfx1201 default, with both typed config and environment opt-outs retained. The rejected Phase 2 raw-V experiment is preserved as `f24fb8a39` and reverted by `f08c881b9`.

## Identity and method

- Date: 2026-09-21.
- Branch/worktree: `gfx1201-attn-qresident`, `/home/kaden/ClaudeCode/warpfront/wt-attnq`.
- Device for all reported gates: **card-A**, UUID `GPU-9eb7aeda51c88ffd`. No cross-card comparison is used.
- Model symlink target: `qwen3.8-27b.mq4-xt` → `/home/kaden/qcal/qat/v25/qwen3.8-27b.mq4v2.xt.sym-a035.qat-r5s100.hfq`.
- Artifact SHA-256: `56e4a67ce6e3f7578ebdd6061a56f8eba9e5c962c62efed7c71e64442da2a867`.
- Canonical TTFT prompt: `benchmarks/prompts/ttft_5900.txt`, 5,909 tokens, MD5 `ed720348b81a19fab64d4783c75c1ae3`.
- Phase 1 binary MD5s: CLI `7e78886070309ba216de32b051f0b3f4`, daemon `fcf37df829d1175c86f36b65c86811f5`.
- Final shipped binary MD5s after the default flip/rejected experiment revert: CLI `6f6c1fb7e49e59169dbe6974820886c2`, daemon `d097c9a25d53f18366889d6d83bb76c6`.
- All timed arms used `HOME=/home/kaden/.hipfire-homes/ab0`, the card-A UUID in both visibility variables, that HOME's kernel cache, the shipped models directory, `HIPFIRE_GRAPH=1`, and this worktree's release daemon.

The daemon-config trap was handled explicitly. For the control and candidate Phase 1 arms, `./target/release/hipfire config set kernel.attn_qresident false|true` wrote the typed `ProcessConfig`; no performance result relies on an ambient long-tail environment flag. The same-build profiler replays selected `attention_fp8_e4m3_fa2_gqa_packet_gfx1201` for the false/default-old arm and `attention_fp8_e4m3_fa2_gqa_qresident_gfx1201` for the true arm. After the default flip, a fresh HOME containing no config file and no attention environment variable selected only the Q-resident symbol.

## Design and implementation

The model has 24 query heads, 4 KV heads, head dimension 256, and GQA 6. The Q-resident launch uses 768 threads / 24 wave32s and owns 384 query-head rows: 64 query positions times the six query heads associated with one KV head. The grid is `[ceil(min(batch,512)/64), 4, ceil(batch/512)]`. K64/V64 tiles remain shared in LDS while each compute wave keeps its Q fragments live in registers across the whole context.

The f32 Q ABI needs an amax pass and a conversion pass. Keeping converted Q resident makes Q traffic `384 × 256 × 4 × 2 = 786,432 B` per workgroup, independent of context. At equal 384-row output coverage, this is **65× less Q traffic** than three incumbent 128-row workgroups. The final kernel retains deferred online-softmax scaling: reference/denominator state advances cheaply and the 128-value output accumulator is rescaled only when the persistent FP8 weighted-V range requires it.

Admission remains narrow: exact gfx1201, H24/KV4/D256, native FP8 KV, batch 64..32768 with the existing chunk/alignment constraints, context 64..32768, and no tree-bias path. All other shapes and architectures retain their previous route. On an admitted gfx1201 process the route now defaults on; `kernel.attn_qresident=false` or `HIPFIRE_ATTN_QRESIDENT=0` restores the packet kernel.

Final code-object/runtime receipts:

- 216 VGPR, 28 SGPR, 49,408 B dynamic LDS, zero VGPR/SGPR spills, zero private bytes.
- 768 threads, 24 wave32s, **one resident block/CU**.
- 32 FP8 WMMA, zero F16 WMMA, 68 packed FP8 conversions, 48 global loads, 32 global stores, zero scratch references, and four barrier signal/wait pairs.
- Real-slab comparison against an fp32 reference: relRMS `0.03232714207`; against the packet control: relRMS `0.003622857630`; zero non-finite outputs.
- Standalone full-launch screen: packet `1.364466 us/token`, Q-resident `1.112124 us/token`, **1.226901×** faster.

## Phase 1 production gates

### Canonical TTFT (`--runs 8 --warmups 2`)

Every row below used an explicit typed config value. The selection column ties each arm to its same-build trace symbol.

| Pair | Arm | Typed selection / trace proof | median TTFT (ms) | prompt tok/s | Candidate delta |
|---|---|---|---:|---:|---:|
| 1 | control | `kernel.attn_qresident=false` → `…_packet_gfx1201` | 1,680.812 | 3,515.564 | baseline |
| 1 | candidate | `kernel.attn_qresident=true` → `…_qresident_gfx1201` | 1,650.033 | 3,581.144 | **+1.865%** |
| 2 | control | `kernel.attn_qresident=false` → `…_packet_gfx1201` | 1,688.306 | 3,499.959 | baseline |
| 2 | candidate | `kernel.attn_qresident=true` → `…_qresident_gfx1201` | 1,636.695 | 3,610.326 | **+3.153%** |
| pooled | control | packet symbol | 1,684.559 | — | baseline |
| pooled | candidate | Q-resident symbol | 1,643.364 | — | **+2.507%** |

Raw TTFT samples (ms):

- Pair 1 control: `[1672.129000, 1673.701994, 1676.664984, 1679.916020, 1681.707925, 1685.129991, 1688.278692, 1687.717221]`.
- Pair 1 candidate: `[1640.822301, 1643.755406, 1646.636221, 1648.256507, 1651.809871, 1653.124273, 1654.763286, 1657.208801]`.
- Pair 2 control: `[1680.415777, 1682.829215, 1684.060620, 1687.209392, 1689.403447, 1691.044102, 1692.387298, 1693.972851]`.
- Pair 2 candidate: `[1632.475352, 1632.754035, 1634.226337, 1635.870086, 1637.519356, 1639.987376, 1641.566579, 1643.070518]`.

The +1% to +2% first-pair result triggered the required second pair. Both pairs independently pass the +1.5% gate.

### Exact pp512 / pp8192 / tg128 ABBA matrix

Each row used `--runs 3 --warmups 1`, context 128, tg128. A rows explicitly set the typed key false and therefore select the traced packet symbol; B rows explicitly set it true and therefore select the traced Q-resident symbol.

| Order | Arm / selection proof | pp512 tok/s | pp8192 tok/s | tg128 tok/s |
|---|---|---:|---:|---:|
| A1 | control / typed false → packet | 3,347.9 | 3,567.2 | 36.45076 |
| B1 | candidate / typed true → Q-resident | 3,356.4 | 3,628.8 | 36.41670 |
| B2 | candidate / typed true → Q-resident | 3,329.2 | 3,603.1 | 36.37594 |
| A2 | control / typed false → packet | 3,318.1 | 3,537.3 | 36.40716 |

| Pair | pp512 delta | pp8192 delta | tg128 delta |
|---|---:|---:|---:|
| A1→B1 | +0.254% | **+1.727%** | −0.093% |
| A2→B2 | +0.335% | **+1.860%** | −0.086% |
| pooled ABBA | **+0.294%** | **+1.793%** | **−0.090%** |

No row regresses by 1%. A separate exact candidate `--pp 512 --ctx 128 --tg 128` run measured pp512 `3,314.1 tok/s` and tg128 `36.47335 tok/s`; tg128 is within 0.02% of the 36.48 reference.

### Quality

| Route | c2 KLD | c24 KLD | c24 limit | Result |
|---|---:|---:|---:|---|
| IU4 prefill | 0.061176 | 0.078594 | 0.0815 | pass |
| fp8v2 prefill | 0.037087 | 0.045574 | 0.0465 | pass |

Both full-sequence routes pass. The candidate also improves both c24 values relative to their controls.

### Mandatory no-think serve battery

Command shape: `benchmarks/serve_battery.py --model qwen3.8:27b-mq4-xt --profile battery --repeat 1 --no-think --seed 1 --no-lock`. The isolated harness config explicitly contained `kernel.attn_qresident=true`; the temporary harness edit was reverted after the run. The daemon MD5 matches the traced Phase 1 candidate binary. Verbatim summary block:

```text
### RUN qwen3.8:27b-mq4-xt|off|battery  kv=auto sampling={'temperature': 1.0, 'top_p': 0.95, 'top_k': 20, 'min_p': 0.0, 'presence_penalty': 0.0} seed=None ###
  [code]t1  finish=stop   ctx=84     cached=0      gen=205  (think 62/ans 46w) prefill=67.5ms/1245.3tok/s decode=36.5tok/s tau=None | '```python\ndef merge_sorted(a, b):\n    """Merge two sorted lists into one sorted list."""\n '
  [reason]t2  finish=stop   ctx=95     cached=0      gen=179  (think 30/ans 53w) prefill=61.1ms/1555.2tok/s decode=36.5tok/s tau=None | 'Step 1: Distance at 60 mph for 2.5 hours  \n\\[\n60 \\times 2.5 = 150 \\text{ miles}\n\\]\n\nStep 2'
  [factual]t3  finish=stop   ctx=65     cached=0      gen=186  (think 90/ans 47w) prefill=57.7ms/1126.6tok/s decode=36.5tok/s tau=None | 'The seasons are caused by Earth’s 23.5-degree axial tilt relative to its orbital plane. As'
  [prose]t4  finish=stop   ctx=73     cached=0      gen=320  (think 164/ans 87w) prefill=58.6ms/1245.9tok/s decode=36.4tok/s tau=None | 'During the first fog of winter, the lighthouse keeper found a sealed brass tube lying amon'
  [instruct]t5  finish=stop   ctx=71     cached=0      gen=149  (think 58/ans 48w) prefill=58.4ms/1215.0tok/s decode=36.4tok/s tau=None | '1. Use clear, descriptive names for variables, functions, and modules.\n2. Keep functions s'
[qwen3.8:27b-mq4-xt|off|battery DONE] turns=5 runaway=0 empty=0 attractor=0 retrieval_miss=0 avg_prefill=1277.6tok/s avg_decode=36.5tok/s
  prompt_md5=43ca0d15712d3dfb777b51ae76d8fd5f request_md5=b45624e909a1eb3f40f63272c1f41355 step=
  prompt_md5=640e0fd4f55996cb175a422f0a12cef5 request_md5=98908f629ef79dce205ca9aea163085b step=
  prompt_md5=8f66b4c97988825bd8e7840aaf44357e request_md5=9502b5d5a6579d96c0a00e350be5375f step=
  prompt_md5=8fe0ad36f61bcf4992cc9df81cdf3817 request_md5=cfbd00094fb95f078fad91980cc566bf step=
  prompt_md5=8bed8e2d056dc1d47dccae9d32dbecf4 request_md5=ae84c5a3746b4b9a8247c6601ff2687c step=
  daemon_binary_md5=fcf37df829d1175c86f36b65c86811f5 path=/home/kaden/ClaudeCode/warpfront/wt-attnq/target/release/daemon
```

The full decoded responses in `battery-candidate.json` were inspected, not just previews: the merge function is a correct two-pointer implementation; the arithmetic reaches 210 miles; the factual answer contains exactly three coherent sentences; the lighthouse story contains exactly four coherent sentences; and the maintainability response contains exactly five numbered one-line tips. All five terminate with `stop`; none is empty, runaway, repetitive, or non-finite.

## In-daemon kernel timing and flag-free shipped proof

The same-build typed traces over two pp8192 passes contain 32 attention calls = 16 full-attention layers × 2 passes. Normalization is therefore 16,384 tokens.

| Arm | Selection proof | total attention ns | attention us/token | Delta |
|---|---|---:|---:|---:|
| control | typed false → packet symbol, 32 calls | 396,793,841 | 24.218 | baseline |
| candidate | typed true → Q-resident symbol, 32 calls | 332,164,926 | 20.274 | **−16.29%** |
| final shipped binary | flag-free fresh HOME → Q-resident symbol only, 32 calls | 331,985,631 | **20.263** | **−16.33%** vs control |

The final trace records 216 VGPR and a `[768,1,1]` workgroup for every resident dispatch; no packet dispatch appears. The 13 us/token stretch target is not met, but it is not the production gate once the end-to-end TTFT or pp8192 threshold passes.

After commit `1492b6cb2`, a fresh HOME with no config file and no `HIPFIRE_ATTN_QRESIDENT` variable produced a flag-free canonical TTFT median of **1,635.666 ms / 3,612.60 prompt tok/s** over 8 runs after 2 warmups. The fresh-home trace selected only the Q-resident symbol. This is the shipped headline.

## Phase 2 residual audit and experiment

### Launch count source answer

There is no 48→16 attention-launch reduction available on this model. The 64-layer model has **16 FullAttention layers** (layers 3, 7, …, 63) and **48 LinearAttention/GDN layers**. `batch_chunk_full_attn_attn` is called only for `LayerType::FullAttention`, and its widened pp8192 path issues one Q-resident launch per full-attention layer. Both the packet and Q-resident traces show 32 calls for two passes, hence **16 launches/pass**, exactly the stated radiance count. The reported 48 count belongs to the GDN/linear layers, not full attention. Source and trace therefore answer the fusion question: attention launches are already 16→16, so no attention-layer launch fusion was implemented.

### Raw-V / on-demand transpose experiment

The cheapest concrete residual experiment replaced the 16 KiB transposed-V plane plus eight private transpose scratch regions with a 16 KiB raw-V plane. Each of the 24 compute waves then reconstructed the V WMMA fragment on demand with lane shuffles/permutations. This simultaneously tested V-transpose removal and whether the lower LDS allocation could unlock two blocks/CU.

| Variant | Commit | Dynamic LDS | VGPR / SGPR | spills / private | blocks/CU | attention us/token | pp8192 screen | Outcome |
|---|---|---:|---:|---:|---:|---:|---:|---|
| shipped transpose-once | `1492b6cb2` / final `f08c881b9` | 49,408 B | 216 / 28 | 0 / 0 B | 1 | **20.263** | 3,603.1–3,628.8 | keep |
| raw V, transpose per compute wave | `f24fb8a39` | 33,024 B | 240 / 27 | 6 VGPR / 28 B | 1 | **29.710** | 3,516.6 | kill |

The experiment cut LDS by 16,384 B (33.2%) but did **not** reach two blocks/CU: two allocations already require 66,048 B, over the 64 KiB LDS ceiling, before its 48-wave and 240-VGPR two-block demands are considered. Code-object inspection also found six VGPR spills and a 28-byte private segment. Its traced attention time regressed **46.62%**, from 20.263 to 29.710 us/token, because the formerly one-time eight-wave transpose work became repeated on all 24 compute waves and raised register pressure. The one-run pp8192 screen likewise fell to 3,516.6 tok/s. It was rejected and reverted; no Phase 1 re-gate was warranted for a clear loser.

| Residual attacked | Before | After | Finding |
|---|---|---|---|
| V transpose / LDS occupancy | transpose once, 49,408 B, 216 VGPR, 1 block/CU | on-demand transpose, 33,024 B, 240 VGPR + spills, 1 block/CU | LDS savings do not buy occupancy; repeated transpose dominates |
| launch count | 16 attention launches/pass | 16 attention launches/pass | premise corrected by source + trace; already matches radiance |

## Final decision

**SHIP the Q-resident route as the exact-gfx1201 default for its admitted H24/KV4/D256 native-FP8 shapes.** It passes the production rule twice over: TTFT pooled **+2.507%** and pp8192 pooled **+1.793%**, with no row slower by 1%, tg128 unchanged, both quality routes under limit, and all five decoded battery outputs coherent. The flag-free shipped headline is **1,635.666 ms TTFT / 3,612.60 prompt tok/s**, final traced attention is **20.263 us/token**, and residency remains **one block/CU**. Keep both opt-outs. Do not retain the raw-V Phase 2 variant.