# gfx1201 fp8 attention legs

## Result

The requested QK8 and PV8 controls are implemented as independent process flags that default on for gfx1201:

- `kernel.attn_qk8` / `HIPFIRE_ATTN_QK8`
- `kernel.attn_pv8` / `HIPFIRE_ATTN_PV8`

The exact gfx1201 H24/KV4/D256 fp8-KV dispatch has four clean routes. Off/off uses the f16 baseline, QK8-only uses fp8 QK plus f16 PV, PV8-only uses f16 QK plus fp8 PV, and both-on uses the existing 256-thread packet kernel. Widened 8192-row packet dispatch is admitted only when both bits are enabled.

Important baseline finding: commit `6be650494` already contained the combined fp8 packet arithmetic. Before this change its packet body quantized Q once, kept K/V as raw LDS bytes, and issued fp8 WMMA for both products. The new controls preserve that shipped both-on behavior on gfx1201 and permit either 8-bit leg to be disabled independently; they do not create a faster combined kernel than the packet body already present at the base commit.

## Source and ISA

- f16 baseline: `attention_fp8_e4m3_fa2_gqa_f16_gfx1201`, f16 QK + f16 PV, 65,536 B LDS.
- QK8-only: `attention_fp8_e4m3_fa2_gqa_qk8_gfx1201`, one f32-to-E4M3 Q conversion, raw K/V bytes, fp8 QK, V decoded at PV consumption, 32,768 B LDS.
- PV8-only: `attention_fp8_e4m3_fa2_gqa_pv8_gfx1201`, f16 QK, raw V bytes, pre-quantization denominator, four packed-P conversions per subtile, fp8 PV, 49,152 B LDS.
- both-on: existing `attention_fp8_e4m3_fa2_gqa_packet_gfx1201`, 49,408 B LDS.

Cached-object disassembly census across the route-N direct/partial object:

| route | fp8 WMMA mnemonics | f16 WMMA mnemonics | packed fp8 conversions | scratch refs |
| --- | ---: | ---: | ---: | ---: |
| QK8-only | 8 | 32 | 4 | 0 |
| PV8-only | 32 | 128 | 8 | 0 |

Resources from generated metadata and the HIP occupancy query:

| route | VGPR | SGPR | spills/private | resident blocks/CU |
| --- | ---: | ---: | ---: | ---: |
| f16 baseline | 235 | 34 | 0 / 0 | 1 (65,536 B LDS) |
| QK8-only | 209 | 33 | 0 / 0 | 2 (32,768 B LDS) |
| PV8-only | 239 | 30 | 0 / 0 | 1 (49,152 B LDS) |
| both packet, base receipt | 237 | 29 | 0 / 0 | 1 (49,408 B LDS) |

Evidence: `device-isa-counts.json`, `qk8-cache.isa`, `pv8-cache.isa`, `leg-occupancy.json`, and the generated kernel-cache metadata.

## Real pp8192 numerical screen

Layer 35 real Q/K/V taps were captured from a production 8192-token prefill in sixteen 512-row commits, then evaluated against the f64 oracle (64 sampled queries, all 24 heads, full attended prefixes). This is not RNG data.

- K raw tap census: 8,388,608 values, amax 19.6983, clipped-to-zero 0.0010%.
- V raw tap census: 8,388,608 values, amax 37.3849, clipped-to-zero 0.0007%.
- f16 production gap vs exact: tail-1% 0.005308.
- native-fp8 format-only baseline: tail-1% 0.02067.
- QK8 with exact weighted PV: tail-1% 0.03703.
- both fp8 legs, native route: tail-1% 0.03727.
- weighted-P underflow: 3,784 / 12,185,088 = 0.0311%, lost mass 0.0000%; PASS against the 0.1% stop gate.

Full operands and output: `real-pp8192/layer35/`. Full transcript: `real-pp8192.log`.

## End-task KLD

All runs used native fp8 KV and the same WT2 reference. Delta is relative to off/off.

| flags | 2-chunk KLD | delta | 24-chunk KLD | delta |
| --- | ---: | ---: | ---: | ---: |
| off/off | 0.062735 | — | 0.080750 | — |
| QK8 only | 0.061778 | -0.000957 | 0.080977 | +0.000227 |
| PV8 only | 0.061700 | -0.001035 | 0.080194 | -0.000556 |
| both | 0.064864 | +0.002129 | 0.081199 | +0.000449 |

The 24-chunk combined delta is +0.000449. Every arm stayed finite. Logs and `.kldseq` files are `kld-{off,qk8,pv8,both}-{2,24}.*`.

## In-daemon performance

Configured replay, second 8192-token pass:

| flags | ms | tok/s |
| --- | ---: | ---: |
| off/off, f16 | 3,587.47 | 2,283.5 |
| QK8 only | 2,600.76 | 3,149.9 |
| PV8 only | 3,292.91 | 2,487.8 |
| both, packet | 2,256.49 | 3,630.4 |

The both-on rocprof trace selected the packet symbol: 32 calls, 398,287,556 ns total, or **24.309 us/token** over 16,384 timed tokens. The base-commit receipt was 24.200 us/token and 3,633.3 tok/s. This is a kill as a speed candidate: both fp8 legs were already hardwired in the base kernel, and the new flags only permit disabling them. The combined arm is quality-safe at the measured KLD gate and preserves the pre-existing packet high-water, but it does **not** approach radiance's 8.76 us/token or cross 4,000 tok/s. The remaining gap is packet geometry/scheduling, not absent 8-bit WMMA legs.

Evidence: `both-configured-warm.log`, `qk8-configured-warm.log`, `pv8-configured-warm.log`, and `prof-both-configured/`.

## ATTN_DIFFERENTIAL

### Headline 1: Q-tile reuse

**Radiance loads each Q tile once and keeps it in registers while every KV tile streams past it. Hipfire converts Q once, but then re-reads the resulting Q8 operands from global scratch every 16 KV positions.**

The radiance trace symbol is
`attn_prefill_kernel<24,48,256,6,16,0,3955307>`. The template declaration identifies those fields as
`NWARPS=24`, `TILE=48`, `HEAD_DIM=256`, `GQA=6`, `BS=16`, `KVP=0` (fp8 KV), and `OPT=3955307`
(`attn_prefill_h256_gqa6.hip:109-127`). It derives `ROWS=NWARPS*16=384`,
`BLOCK_Q=ROWS/GQA=64`, `NKS=16`, and `MT=3` at lines 131-144. Lines 185-203 load and quantize
the 384 bf16 query-head rows into `qf8[16]` before the KV loop; the `for (ti...)` loop begins at
line 357 and lines 378-390 consume those same register fragments for every K48 tile.

- Radiance Q bytes loaded per workgroup per pass:
  `384 query-head rows * 256 dims * 2 B = 196,608 B` (192 KiB), independent of context length.
- Q remains resident for the whole attended prefix: all `ctx` KV positions, rounded into K48
  iterations. At an 8192-position pass, it survives all 171 K48 iterations.

Hipfire's packet mapping is 128 query-head rows over eight waves
(`attention_q8_0_fa2_gqa.gfx1201.hip:1918-1927,1969-1974`). Its fused prologue scans the original
f32 Q row once for amax and again for conversion, then writes 256 Q8 bytes per row to global
scratch (`:1984-2028`). Inside every K16 recurrence, lines 2185-2215 reload all 256 Q8 bytes per
query-head row. The outer K64 loop at lines 2061-2064 therefore reloads Q four times per tile.

- Hipfire Q bytes loaded per workgroup per pass are
  `128*256*4*2 + 128*256*ceil(ctx/16)`.
- At the profiled `ctx=8192`, that is `262,144 + 16,777,216 = 17,039,360 B` (16.25 MiB),
  plus a 32,768 B scratch write.
- At the radiance trace's 5,892-token normalization, the same formula is 12,353,536 B
  (11.78125 MiB).
- The Q8 operand is re-fetched after 16 KV positions. Only the original f32-to-Q8 conversion
  survives the full prefix.

Per workgroup, the 8192-position Q-read ratio is 86.67x. A radiance workgroup owns three times as
many query-head rows, so at equal output work the ratio is **260x** (188.5x at 5,892 positions).
Those are instruction/cache bytes, not claimed DRAM bytes: the Q8 scratch should hit cache after
the first access. The directly comparable inner-operand traffic is a conservative 1.5x:
radiance issues 16 K `b64` plus 16 V `b64` fragment loads per K16, while hipfire issues eight K
`b128`, sixteen Q `b64`, and eight V `b128` loads. The source sites are radiance
`:378-390,450-464` and hipfire `:2185-2215,2278-2292`.

### Headline 2: GQA sharing

**Both kernels share one staged K/V tile across the six query heads of a KV head; neither loads
K/V once per query head. Radiance's advantage is that the same tile also serves three times as
many query positions.**

Radiance's 384 query-head rows are exactly `64 query positions * GQA 6`. Its grid is
`[ceil(q_len/64), kv_heads, num_seqs]`, block 768 (`attn_paged_h256_gqa6.hip:49-59,74-87`).
One K48 tile contains `48*256 = 12,288 B` of K and the same of V, or **24,576 B per workgroup per
KV tile**. It serves all 384 query-head rows, which is `24,576/(384*48) = 1.333 B` of global K/V
per query-head row per KV position.

Hipfire's grid is `[ceil(run_rows*6/128),4,ceil(batch/512)]`, block 256
(`attention.rs:4876-4885,4970-4973,5007-5014`). One K64 tile loads 32,768 code bytes plus 256 B
of per-token K/V scale headers, or **33,024 B per workgroup per KV tile**
(`attention_q8_0_fa2_gqa.gfx1201.hip:2068-2164`). It serves 128 query-head rows, or about
21 1/3 query positions times six heads, giving `33,024/(128*64) = 4.031 B` per query-head row per
KV position.

Thus the GQA-head reuse multiplier is 6x in both designs (1.0x differential). Hipfire's dense
128-row boundaries can split one six-head group between adjacent workgroups, but it is not a
per-head K/V load. The important multiplier is query-position reuse:
`4.031/1.333 = 3.023x` less global K/V traffic per output row for radiance.

### Work shape and resource table

| Property | radiance trace kernel | hipfire packet kernel |
| --- | ---: | ---: |
| Threads / waves per workgroup | 768 / 24 | 256 / 8 |
| Query-head rows per workgroup | 384 | 128 |
| Query positions represented | 64, exactly all 6 heads | 21 1/3, dense rows across 6 heads |
| KV positions per iteration | 48 | 64 |
| QK + PV WMMA per wave/iteration | 48 + 48 = 96 | 64 + 64 = 128 |
| Raw K+V bytes per KV iteration | 24,576 | 32,768 (+256 B headers) |
| LDS | 39,296 B | 49,408 B |
| VGPR / SGPR | 213 / 32 | 237 / 29 |
| Scratch / spills | 0 / 0 | 0 / 0 |

Radiance's two named raw-fp8 planes are 27,008 B (`attn_prefill_h256_gqa6.hip:153-163`).
The exact object metadata reports 39,296 B: the additional 12,288 B is compiler-generated group
storage (16 B/thread across 768 threads; the source has no third explicit shared array).
**[INFERENCE]** Its source candidate is the per-thread tile bookkeeping at `:348-373`.
Hipfire's 49,408 B is 32,768 B of raw planes, eight wave-private 2,048 B V-transpose scratch
regions, and 256 B of scale headers
(`attention_q8_0_fa2_gqa.gfx1201.hip:1963-1968`). The resource counts are from the exact cached
code objects for the traced symbols, not source estimates.

### Softmax and KV dataflow

Both designs are fused, single-pass online softmax; there is no two-pass advantage. Radiance uses
a lazy reference maximum: it rescales the 128-VGPR output accumulator only when the row maximum
has grown beyond the E4M3 eight-octave range (`attn_prefill_h256_gqa6.hip:425-448`). Hipfire
updates `m_old` and executes all sixteen `float8` accumulator rescales after every K16 recurrence
(`attention_q8_0_fa2_gqa.gfx1201.hip:2232-2266`). That is up to 128 scalar accumulator multiplies
per wave every 16 keys versus a conditional rare path.

Radiance also fetches K with one wide 16-byte transaction per thread, writes K/V directly into
their fragment layouts, scalar-prefetches the next block-table entries, and issues the next V
tile's global loads before 96 WMMAs (`attn_prefill_h256_gqa6.hip:147-163,311-373`). Hipfire
uses a 16 KiB wave-private V transpose area, a full-workgroup barrier before transpose consumption,
another before compute, and a third at tile turnover
(`attention_q8_0_fa2_gqa.gfx1201.hip:2093-2164,2300`). No K prefetch is proposed.

### Launch count and ratio arithmetic

Both paths are one fused attention launch per layer per pass. Radiance's launcher emits one kernel
for each equal-query-length run; the trace has **48 launches/pass**. Hipfire fuses Q conversion into
the packet prologue and has no preconvert or merge launch (`attention.rs:4949-4953`); the profile
has 32 calls across two pp8192 commands, hence **16 launches/pass**. The models have different
numbers of attention layers. Launch count therefore cannot explain hipfire being slower: hipfire
has one third as many attention launches.

The measured aggregate ratio is:

`24.309543 / 8.757567 = 2.7758x`.

Normalizing by attention-layer calls exposes the body difference:

`(24.309543/16) / (8.757567/48) = 8.3275x` per layer per token.

The source-derived query-position/KV amortization factor is 3.0234x. The conservative Q-residency
operand-traffic factor is 1.5x. Together the two reuse differences account for a
`3.0234*1.5 = 4.5352x` per-layer factor. The remaining body factor is:

`8.3275 / 4.5352 = 1.8362x`.

Putting the three-times-higher radiance layer count back in reproduces the aggregate observation:

`(4.5352 * 1.8362) / 3 = 2.7758x`.

This is a traffic/shape decomposition, not an isolated timing ablation. The Q scratch reads are
cacheable, so their 188-260x pass-byte ratio must not be presented as a time multiplier.

Requested ranked time budget, using the logarithmic shares of the source-derived multiplicative
factors to distribute the measured 15.55 us/token gap (estimate, not a measured ablation):

| Rank | Structural difference | estimated gap |
| ---: | --- | ---: |
| 1 | 384-row/64-position Q tile, hence 3.023x K/V amortization | ~8.1 us/token |
| 2 | Q held in registers across the full context instead of global Q8 reload every K16 | ~3.0 us/token |
| 3 | direct/prefetched KV staging instead of the 16 KiB V transpose and three barriers | ~2.9 us/token |
| 4 | lazy online-softmax rescale instead of an eager 128-element rescale every K16 | ~1.5 us/token |
| 5 | GQA-6 head sharing | ~0: both do it |
| 6 | attention launch count | ~0: both are one/layer, and radiance has 3x more layers here |

**Single biggest structural difference:** radiance keeps a 64-position by six-head Q tile in
registers for the whole context and streams K48/V48 past it; hipfire owns only 128 query-head rows
and reloads their Q8 fragments from global scratch every K16.
