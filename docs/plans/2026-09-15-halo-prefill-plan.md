# Strix Halo batched-prefill plan of record — Qwen3.8-27B MQ4V2-xt

Date: 2026-09-15  
Planning base: `/home/kaden/ClaudeCode/warpfront/wt-lloyd` at `c6853370b`  
Target: exact `gfx1151`, Qwen3.8-27B `qwen3.8-27b.mq4-xt`, ordinary eager Q8-EF prefill  
Authoring seat: Sol. Final review/admission belongs to Fable, the foreman, or the main seat; `@reviewer` is not an admissible review destination for this plan.

No GPU program was run while producing this plan. No source was edited. All performance deltas below are explicitly planning estimates or gates, never results. The source audit starts from the brainstorm, which itself declares that its magnitudes are estimates rather than measurements (`docs/plans/2026-09-15-halo-prefill-levers.md:3-7`).

## 1. Decision summary

Execute and admit in this order:

1. **A2 — IU4 X-tile stride 44 → 42.** One-hour, bit-exact bank-layout experiment.
2. **A1 — IU4 global→VGPR→LDS software pipeline.** One-day make-or-break screen. The new caller-supplied IU4 twin result—doubling WMMA work made the kernel 2.7× slower with zero spills—supersedes the brainstorm's latency-bound inference. Bank no gain until the gate passes.
3. **B1 — FA2 even swizzle plus aligned LDS b64 fragment loads.** Rebuild a seven-shape incumbent/old-FA2/new-FA2 oracle for both Q8-K and fwht3-K; no such harness is currently in-tree.
4. **B2 — remove the two compiler-only memory clobbers.** Incremental over accepted B1; same oracle and resource gate.
5. **C2 — emit the existing IU4 `block_i4_128` representation in the RMSNorm/FWHT and SwiGLU/FWHT producers.** Keep the exact quantizer arithmetic and feed an explicit prepared handle to the existing IU4 consumers. Do not claim that the quantization arithmetic disappeared.
6. **C1 — Q8-EF GDN over the existing F32 chunked algebra, carrying transient F32 state across internal CS≤32 chunks and requantizing only at the existing outer prefill-chunk boundary.** This is the multi-day long pole and starts with a no-new-kernel speed screen.
7. **F — N=1024 prefill-chunk experiment.** It is forbidden before direct FA2 N=1024 compatibility and the 512-token GDN state-commit cadence are both proven.

Every stage is a kill gate. A rejected stage is reverted before the next measurement; no failed variant remains as an environment arm, alias, dormant symbol, or fallback. The campaign stops after F or earlier when the remaining measured bucket ceilings cannot close a row. It does not proceed into the assignment's kill list.

## 2. Correct denominators and attribution ledger

### 2.1 End-to-end wall targets

The brainstorm mixed the profiled pp512 total (878 ms) with the wall-rate target and therefore wrote a 53 ms gap (`docs/plans/2026-09-15-halo-prefill-levers.md:22-24`). Promotion uses wall time derived from the caller-supplied rates:

| Prompt | Hipfire now | Current wall, $P/r$ | Halogen target | Target wall, $P/r_t$ | Required wall reduction | Required time reduction |
|---|---:|---:|---:|---:|---:|---:|
| pp512 | 570 t/s | 898.246 ms | 620 t/s | 825.806 ms | **72.439 ms** | 8.06% |
| pp2048 | 554 t/s | 3.696751 s | 710 t/s | 2.884507 s | **812.244 ms** | 21.97% |
| pp32768 | 378 t/s | 86.687831 s | 566 t/s | 57.893993 s | **28.793838 s** | 33.22% |

These are the only row-clear denominators. Internal profiling serializes launches, so its tok/s is not valid promotion evidence (`docs/methodology/perf-benchmarking.md:154-159`).

### 2.2 Measured pp512 buckets

The caller supplied this one-chunk profile; the arithmetic below is exact:

| Bucket | Calls × average | Total |
|---|---:|---:|
| IU4 `full_set_occ3` | 272 × 1,729 µs | 470.288 ms |
| IU4 `full_add_occ3` | 128 × 1,685 µs | 215.680 ms |
| **IU4 total** | | **685.968 ms** |
| `gated_delta_net_q8_batch_seq` timer | 48 × 1,340 µs | 64.320 ms |
| `quantize_int4_mmq_ds128` | 256 × 154 µs | 39.424 ms |
| `fused_silu_mul_mq_rotate_batched` | 64 × 457 µs | 29.248 ms |
| `attention_q8_0_fa2_gqa_gfx11` | 16 × 512 µs | 8.192 ms |
| `conv1d_silu_split_f32_n` | 48 × 237 µs | 11.376 ms |
| Other profiled kernels to 878 ms | | 39.472 ms |
| Profile-to-production-wall reconciliation | | 20.246 ms |
| **Production wall** | | **898.246 ms** |

The model-specific default is exactly 512 only when `gpu.arch == "gfx1151"` and all dense weights are MQ4V2 (`crates/hipfire-arch-qwen35/src/qwen35/prefill.rs:425-462`). The outer loop then processes prompts in chunks through `forward_prefill_chunk` (`crates/hipfire-arch-qwen35/src/qwen35/prefill.rs:1334-1430`), so pp2048 has four outer chunks and pp32768 has 64.

### 2.3 Long-context attention denominator

The caller supplied 256 FA2 calls × 7,255 µs = 1.857280 s at pp8192. Sixteen outer chunks have causal endpoints 512, 1024, …, 8192, so their mean endpoint is 4,352, not 4,608. The kernel derives each workgroup's causal bound from `positions[]` and loops KT tiles to `gmax+1` (`kernels/src/attention_q8_0_fa2_gqa.gfx11.hip:167-206`). Two intentionally simple extrapolations bound, rather than measure, the other prompts:

- proportional work from the pp8192 sum gives 136.565 ms at pp2048 and 28.405459 s at pp32768;
- an affine per-call fit through the pp512 and pp8192 profile points gives 119.080 ms at pp2048 and 29.524582 s at pp32768.

Therefore this plan uses **0.12–0.14 s at pp2048** and **28.4–29.5 s at pp32768** as an attribution range. These are estimates. The unexplained wall residual is about 0.24–0.26 s at pp2048 and 4.0–5.1 s at pp32768 after the other pp512 buckets are scaled linearly; no unit may quietly spend that residual as savings.

## 3. Exact target call graph and shape audit

The loader derives dense shapes from `dim`, attention heads, linear-attention heads, and hidden width (`crates/hipfire-arch-qwen35/src/qwen35/load.rs:3487-3500`). It validates full-attention q/k/v/o and gate/up/down projections at the derived dimensions (`crates/hipfire-arch-qwen35/src/qwen35/load.rs:3543-3565`) and linear-attention qkv/z/a/b/out plus FFN projections separately (`crates/hipfire-arch-qwen35/src/qwen35/load.rs:3567-3591`). The checked Qwen3.8 shape is hidden 5120, intermediate 17408, 64 layers, 48 physical Q heads, 8 KV heads, 16 linear key heads, and 48 linear value heads (`crates/hipfire-arch-qwen35/src/qwen35/config.rs:1337-1350`).

The measured 48 LA + 16 FA layer mix yields the following 512-row IU4 launches per outer chunk:

| Family | M × K | Calls | 128×128 output blocks/call | Rounds of 80 resident blocks | Last round |
|---|---:|---:|---:|---:|---:|
| gate and up | 17408 × 5120 | 128 | 136 × 4 = 544 | 6.8 | 64/80 |
| FA q | 12288 × 5120 | 16 | 96 × 4 = 384 | 4.8 | 64/80 |
| LA qkv | 10240 × 5120 | 48 | 80 × 4 = 320 | 4.0 | full |
| LA z | 6144 × 5120 | 48 | 48 × 4 = 192 | 2.4 | 32/80 |
| FA k and v | 1024 × 5120 | 32 | 8 × 4 = 32 | 0.4 | 32/80 |
| down | 5120 × 17408 | 64 | 40 × 4 = 160 | 2.0 | full |
| LA/FA output | 5120 × 6144 or full-attention input width | 64 | 40 × 4 = 160 | 2.0 | full |

The IU4 host maps `[ceil(M/128), ceil(N/128)]` to a `[32,8]` block (`crates/rdna-compute/src/gemm.rs:18562-18580`). The kernel uses a 128×128 tile and eight wave32s (`kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip:43-50`), and its 31,744-byte LDS allocation permits two resident blocks on a 64 KiB CU. The gfx1151 hardware model records 64 KiB LDS/CU and 40-CU measurements are caller-supplied (`crates/rdna-compute/src/profiler.rs:67-73`). The table is therefore rounds of **blocks**, not a claim about scheduler wave timing.

The profile counts reconcile exactly: 272 set calls = 128 gate/up + 48 LA qkv + 48 LA z + 16 FA q + 16 FA k + 16 FA v; 128 add calls = 64 down + 64 output. Weighting the partially filled rounds by projection work gives a 3.16% structural block-capacity loss; that is an upper attribution, not measured savings.

The quantizer is already shared once per family input: qkvza launches it once then reuses the sidecar for qkv/z/beta/alpha (`crates/rdna-compute/src/gemm.rs:27975-28005`), q/k/v reuse one sidecar (`crates/rdna-compute/src/gemm.rs:28518-28530`), gate/up reuse one (`crates/rdna-compute/src/gemm.rs:29626-29639`), and residual creates one for its input (`crates/rdna-compute/src/gemm.rs:31148-31158`). C2 must therefore fuse producer→quantizer; it must not add another “quantize once” cache.

## 4. State ownership and global invariants

### 4.1 Outer prefill transaction

Before an outer chunk, `KvCache` and `DeltaNetState` are persistent caller state; `PrefillBatchScratch` is reusable transient storage. The chunk loop computes `chunk_n`, calls `forward_prefill_chunk`, commits optional hidden-ring staging only after that call succeeds, and then advances `chunk_start` (`crates/hipfire-arch-qwen35/src/qwen35/prefill.rs:1344-1430`). Every unit in this plan preserves that order.

After a successful outer chunk:

- K/V cache contains the appended positions;
- each LA layer's Q8 state/scales/EF represents the chunk-final recurrent state;
- scratch may contain dead intermediates and may be overwritten by the next layer/chunk;
- outputs and logits retain the same ownership as today.

No unit introduces a host-visible partial commit, rollback protocol, or persistent cache alias.

### 4.2 Invariants common to A/B/C2

A2, A1, B1, B2, and C2 are required to be bit-exact against the pre-unit target path. They may change when bytes are loaded or where identical bytes live, but they may not change:

- weight/activation decode;
- FMA/WMMA accumulation order;
- f32/f16 conversion points;
- output writer mapping;
- persistent KV or recurrent-state cadence.

A direct old/new kernel differential must compare output bytes and guard zones before a performance sample is admissible.

### 4.3 C1 is the sole numerical unit

C1 changes the algebraic evaluation order of the recurrent update. It does **not** change the persistent state representation or the number of Q8-EF commits per current outer prefill chunk. Its admission is numerical and state-parity scoped, not bit-exact. All other units remain bit-exact.

## 5. Unit A2 — IU4 X-tile stride 42

### 5.1 Source symbols and owner

Single owner: **A-lane composer**.

- `kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip`
  - `IU4_TILE_X_K` at `:48`;
  - `load_iu4_tile<FULL>` at `:204-243`;
  - `vec_dot_i4_x128` A-side b64 reads at `:264-281`;
  - all five `IU4_ENTRY` symbols at `:374-380`.
- `crates/rdna-compute/src/gemm.rs`
  - `Gpu::gemm_mq4g256v2_mmq_prequant_iu4` at `:18508-18597`;
  - host LDS mirror `MMQ_TILE_X_K` and `shared_mem` at `:18562-18572`.

The kernel and host constants form one ABI invariant and must change in the same commit.

### 5.2 Exact mechanism and proof

Today each A fragment starts at dword bank

$$b(r)=44r+c\pmod{32}=12r+c\pmod{32}.$$

For the 16 distinct `lane_row` values, rows $r$ and $r+8$ start on the same bank for every `int32x2_t` load. The addresses are exactly `row*IU4_TILE_X_K + h*16 + t*4` followed by two b64 loads (`kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip:264-281`).

Set `IU4_TILE_X_K = 42`. Then $42r=10r\pmod{32}$ visits all 16 even start banks once; each b64 occupies that even bank and its following odd bank, so the 16 unique fragment rows cover all 32 banks exactly once. Wave lanes `l` and `l+16` have the same `lane_row=l&15`, request the same address, and use LDS broadcast; the eliminated conflicts are distinct logical rows `r` and `r+8` mapping to the same banks. The stride remains even and holds all 32 nibble dwords plus eight `half2` header dwords; those layout constants are `IU4_X_QS_WORDS=32` and `IU4_X_DM_OFF=32` (`kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip:47-50`). Y's stride 18 already gives the same 16-even-start property.

LDS falls from

- `(128*18 + 128*44)*4 = 31,744 B`
- to `(128*18 + 128*42)*4 = 30,720 B`.

Two-block residency is unchanged. Weight bytes, header locations within each logical row, activation bytes, WMMA inputs, and accumulation order are unchanged.

### 5.3 Transition invariants

| Transition | Owner before | Operation | Owner after / invariant |
|---|---|---|---|
| global A/Xq → LDS | global tensors read-only; WG owns its dynamic LDS | same loaders write new physical row stride | WG owns identical logical dwords; no alias between 128 rows |
| LDS → WMMA fragments | WG LDS read-only during dot | same four t steps and same two b64 operands | identical `av0/av1/bv0/bv1` values |
| accumulator → Y | lane owns the same output fragment | unchanged store loop | one writer/output and add/set semantics unchanged (`:352-371`) |

### 5.4 Gate and abandon rule

Timebox: **one engineer hour**.

1. Compile old and stride-42 twin symbols in one JIT module; run the IU4 direct differential on set/add target shapes. Any output or red-zone byte mismatch: reject.
2. `hipfire profile --kernel gemm_mq4g256v2_residual_mmq_iu4 --json`: require `scratch_bytes == 0`, LDS 30,720 B, and two-block/CU LDS residency. Kernel metadata includes VGPR, LDS, scratch, and occupancy fields (`crates/rdna-compute/src/profiler.rs:199-205,327-350`).
3. On Halo, compare fresh-process medians for both `full_set_occ3` and `full_add_occ3`. Admit only if the 685.968 ms weighted IU4 bucket improves by **>1.0%**, neither symbol regresses by >1.0%, and the production pp512 wall gain exceeds the observed paired spread.
4. If the weighted gain is ≤1.0%, revert both constants and record “A-side LDS bank conflict is not on the gfx1151 critical path.” Do not try stride 46.

Conditional planning estimate after admission: **1–3% of the IU4 bucket** = 6.9–20.6 ms pp512, 27–82 ms pp2048, 0.44–1.32 s pp32768. Until the gate passes, bank 0 ms.

## 6. Unit A1 — IU4 single-LDS software pipeline

### 6.1 Corrected premise

The source really is serialized: it loads weight + activation half 0, synchronizes, dots, synchronizes, loads half 1, synchronizes, dots, and synchronizes for every K/256 group (`kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip:321-345`). Each WG reads 17,408 weight bytes and two 9,216-byte activation tiles per group, for 35,840 B/group. `load_iu4_tile` assigns 16 nibble dwords and one raw header dword to each thread (`kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip:204-243`); each activation half assigns nine dwords/thread (`:324-340`). A one-X + one-Y pending stage is therefore about 26 dwords/thread in addition to the 64-float accumulator (`:319`).

However, the caller's new IU4 twin experiment is stronger than the brainstorm's clock/CU inference: doubling the WMMA body produced 2.7× time with zero spills. Treat the current kernel as MAC/issue-bound until A1 measures otherwise. The brainstorm's 30% GEMM saving and its “56% non-WMMA” attribution (`docs/plans/2026-09-15-halo-prefill-levers.md:83-101`) are retired and may not be used in a forecast.

### 6.2 Frozen pipeline

Keep one physical LDS X tile and one physical LDS Y tile. Do **not** allocate a second LDS stage. After admitted A2 this remains exactly 30,720 B: 9,216 B for `tile_y`, 21,504 B for `tile_x`, and zero bytes for a second plane.

Introduce register-only `prefetch_*` and `commit_*` helpers which preserve the exact index and clamp expressions of `load_iu4_tile<FULL>` and the two activation loops. The schedule is:

1. **Prologue:** prefetch X(0) + Y(0,h0) into VGPRs, commit both to LDS, `__syncthreads`.
2. For current group `kb`:
   1. always issue global loads for Y(`kb`,h1) into pending registers; if `kb+1` exists, also issue X(`kb+1`) into disjoint pending registers;
   2. run `vec_dot_i4_x128(..., h=0)`;
   3. synchronize to prove no reader still uses current Y, commit pending Y(h1), synchronize for visibility;
   4. if `kb+1` exists, issue Y(`kb+1`,h0) into the now-free Y pending registers while X(next) remains live;
   5. run `vec_dot_i4_x128(..., h=1)`;
   6. if `kb+1` exists, synchronize to prove no reader still uses current X/Y, commit pending X(next)+Y(next,h0), and synchronize for visibility; otherwise exit without a final turnover barrier because no later LDS writer exists.
3. **Epilogue:** unchanged stores.

The steady-state body has exactly four workgroup barriers: read-complete/publish around the h0→h1 turnover, then read-complete/publish around the h1→next-h0 turnover. The prologue has one publish barrier; the final group omits only the next-group turnover pair. No asynchronous LDS primitive, second LDS plane, or changed barrier scope is allowed.

The host requires `K % 256 == 0` (`crates/rdna-compute/src/gemm.rs:18527-18531`), so every K/256 group has both 128-K activation halves. Preserve the logical order `kb0/h0`, `kb0/h1`, `kb1/h0`, … and do not alter `vec_dot_i4_x128`, its two WMMAs per t step, or the f32 fold (`kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip:245-295`).

### 6.3 State machine

| State | LDS ownership | Pending VGPR ownership | Legal next transition |
|---|---|---|---|
| `READY_H0(kb)` | X=kb, Y=2kb; read-only to all waves | empty | always issue Yh1; conditionally issue Xnext; dot h0 |
| `DOT_H0_DONE` | X=kb remains live; Yh0 no longer needed after barrier | Yh1 and optional Xnext | barrier, overwrite Y only |
| `READY_H1(kb)` | X=kb, Y=2kb+1; read-only | optional Xnext, then optional Ynext-h0 | dot h1 |
| `DOT_H1_DONE` | current X/Y dead after barrier | optional Xnext/Ynext-h0 | if next exists, commit both and publish; otherwise finish |
| `DONE` | no future LDS read/write | empty | unchanged output epilogue |

No commit is legal before all four waves that consume the old tile have crossed the read-complete barrier. Global A and Xq remain immutable for the launch. Earlier issue of a load cannot change its value.

### 6.4 Gate and abandon rule

Timebox: **one engineer day**, including the differential.

1. Gate the dominant `M=17408,K=5120,N=512` full-set shape first, 100 warm device-timed iterations, interleaved old/new. Require bit-identical output, zero guard corruption, `scratch_bytes == 0`, and unchanged two-block/CU resident capacity.
2. Continue to the in-model set/add profile only if the dominant-shape median is at least **10% faster** and larger than observed spread.
3. Admit only if the weighted 272-set/128-add IU4 bucket is at least **10% faster**, neither symbol regresses >2%, and pp512 wall transfers.
4. If any threshold fails, revert A1 completely and record “the IU4 MAC/issue ceiling survived software prefetch.” Do not try deeper VGPR staging, LDS double buffering, another WMMA-unroll twin, or scheduler-priority knobs.

Pre-gate forecast: **0–10%** of IU4 because the twin result points against the mechanism. Only after the ≥10% gate passes may planning use a 10–15% admitted range: 68.6–102.9 ms pp512, 274–412 ms pp2048, 4.39–6.59 s pp32768. The old 30% stretch is not bankable.

## 7. Unit B1 — FA2 even swizzle and LDS b64 loads

### 7.1 Source symbols and owner

Single owner: **B-lane composer**.

- `kernels/src/attention_q8_0_fa2_gqa.gfx11.hip`
  - `fa2_swiz` at `:99-102`;
  - Q8-K/fhwt3-K producers at `:208-293`;
  - V producer at `:294-338`;
  - K fragment reads at `:354-390`;
  - V fragment reads at `:450-471`;
  - tile-turnover barrier at `:475-477`;
  - Q8 and fwht3 entry symbols at `:502-524,526-600`.
- `crates/rdna-compute/src/attention.rs` is read-only for the production B1 change. It is touched only if the temporary direct-oracle twin launcher is needed; the production launcher remains `[ceil(batch/8),4]`, block 128, dynamic LDS 32,768 B (`:3918-3969`).

### 7.2 Exact swizzle

Replace the current `8*(c>>2)+(c&3)` swizzle with:

```c
int r = ((c & 1) << 3) | ((c & 2) << 1)
      | ((c & 4) >> 1) | ((c & 8) >> 3);
return 2 * r;
```

That is `2*bit_reverse4(c)`, with lane starts:

`0,16,8,24,4,20,12,28,2,18,10,26,6,22,14,30`.

For K, every fragment address is `key*128 + ((dc*8+i+R(key&15)) & 127)` (`kernels/src/attention_q8_0_fa2_gqa.gfx11.hip:361-372`). For `i={0,2,4,6}`, every start is 8-byte aligned; the 16 unique `ml` rows place one b64 on every even/odd bank pair exactly once. Wave lanes `l` and `l+16` have the same `ml=l&15`, deliberately request the same address, and use broadcast. Replace eight scalar dword loads with four aligned b64 loads, then unpack low/high dwords into the same `half16_t` order.

For V at KT32, `V_ROW=16` dwords (64 B) and `V_MASK=15` (`kernels/src/attention_q8_0_fa2_gqa.gfx11.hip:119-125`). The address masks the swizzle before adding the row base: `d*16 + ((ps+i+R(d&15)) & 15)`. Even `d` rows have bank base 0 and odd rows bank base 16. Within either parity, `R(d)&15` visits the eight even residues 0..14 once, so the 16 unique rows cover all even b64 starts 0..30 exactly once. The two wave lanes with the same `ml` also have the same `d=dc*16+ml` and broadcast the same address. Again use four b64 loads at `i={0,2,4,6}` and unpack in the same key order.

Both K producers already add `fa2_swiz` to their dword store address (`kernels/src/attention_q8_0_fa2_gqa.gfx11.hip:208-293`), and the V producer does the same (`:294-338`). Change producer and consumer together. The K plane remains dwords `[0,4096)` (bytes `[0,16384)`), V remains dwords `[4096,8192)` (bytes `[16384,32768)`), and the transient gmax/gmin mailbox remains the final two dwords before the first V fill overwrites it (`:138-140,167-200`).

Do not try b128: 16 distinct rows × four banks requires 64 bank slots in a 32-bank instruction.

### 7.3 FA2 state transitions

| Transition | Owner before | Operation | Owner after / invariant |
|---|---|---|---|
| init | Q/positions/KV caches caller-owned read-only; `out` caller-owned write target | compute waves zero `Ofr,m,l`; wave 0 publishes gmax/gmin mailbox | all lanes observe identical causal bounds (`:142-200`) |
| fill | all four waves own disjoint producer slots | dequantize K and V into the same logical matrices under the new permutation | K/V LDS planes immutable after fill barrier (`:208-339`) |
| QK/softmax/PV | waves 0..2 own 48 query rows; wave 3 is idle | read b64 fragments, preserve f16 element order and all WMMA/softmax operations | `m,l,Ofr` order and values bit-identical (`:341-473`) |
| turnover | compute waves may still read K/V | `__syncthreads` | all four waves may overwrite both planes (`:475-477`) |
| completion | each lane half owns even/odd output dimensions | unchanged normalize/store | exactly one writer per output (`:480-499`) |

For KMODE=3, Q is intentionally modified in place before the shared body, and the production route is direct/eager because a second launch would rotate it twice (`kernels/src/attention_q8_0_fa2_gqa.gfx11.hip:526-539`). The oracle must clone Q separately for incumbent, old-FA2, and new-FA2 arms; only the two FA2 arms compare the post-launch Q bytes.

### 7.4 Seven-shape oracle and gate

Timebox: **two engineer days**.

The historical “7-shape oracle PASS” is recorded only as a result (`CHANGELOG.md:11`); its harness and exact shapes were throwaway and are not in-tree. Rebuild the oracle around the deliberately public direct launchers `Gpu::attention_q8_0_fa2_gqa_gfx11` and `Gpu::attention_q8_0_fa2_gqa_fwht3k_gfx11` (`crates/rdna-compute/src/attention.rs:3842-3854,3991-4010`), plus the corresponding incumbent launchers. Compile old FA2 and candidate FA2 as distinct temporary symbols. Freeze these `(batch,start_position,max_position+1)` fixtures:

1. `(1,0,1)` — direct-launcher minimum;
2. `(7,26,33)` — partial eight-query WG, partial KT tile, nonzero position base;
3. `(64,0,64)` — production minimum;
4. `(128,384,512)` — two production batch tiles at a chunk boundary;
5. `(512,0,512)` — pp512 production chunk;
6. `(512,7680,8192)` — profiled long-context anchor;
7. `(512,32256,32768)` — maximum admitted context.

Run all seven for KMODE=0 and KMODE=3. For KMODE=3, clone Q per arm. First require old FA2 output bits to equal the incumbent and candidate output bits to equal both; then require candidate-FA2 vs old-FA2 post-launch Q bits (KMODE=3) and all guard zones to match. Any failure blocks performance work. The rebuilt list covers the direct launcher's H24/KV4/D256, batch 1..512, and context 1..32768 envelope (`crates/rdna-compute/src/attention.rs:3880-3904`); it does not claim to reconstruct the unrecorded historical fixtures.

Then require:

- emitted ISA actually contains b64 LDS reads for both fragment loops and halves the fragment LDS instruction count; if the compiler scalarizes them, reject;
- zero scratch/spills and no loss of two-WG/CU LDS residency;
- Halo `attention_q8_0_fa2_gqa_gfx11` median at pp8192 improves by **≥10%** and pp2048 transfers without >2% regression;
- because the source is used by the gfx11 allowlist (`crates/rdna-compute/src/attention.rs:3867-3878`), the same rebuilt oracle passes on gfx1100 before promotion. Performance is claimed only for gfx1151; if gfx1100 wall regresses >2%, split an exact-gfx1151 symbol rather than imposing the change cross-architecture.

Planning estimate after admission: 10–25% of attention = 0.8–2.0 ms pp512, 12–34 ms pp2048, 2.8–7.4 s pp32768.

## 8. Unit B2 — remove FA2 compiler memory clobbers

B2 remains owned by the B-lane composer and starts from admitted B1.

The two lines `asm volatile("" ::: "memory")` appear after every QK dc WMMA and every PV dc WMMA (`kernels/src/attention_q8_0_fa2_gqa.gfx11.hip:364-390,455-471`). They are compiler barriers, not device fences. K/V LDS is read-only from the post-fill barrier through the existing turnover barrier; Q is read-only in KMODE=0, and the KMODE=3 prologue completes its Q transform before the shared body. Remove both clobbers and change nothing else.

Timebox: **two engineer hours**.

- Re-run the exact B1 seven-shape, two-K-mode incumbent/old/new oracle.
- Require zero spills, no occupancy loss, and a real ISA scheduling difference.
- Admit only for **≥3% incremental** reduction in the post-B1 attention symbol at pp8192, transfer at pp2048, and no pp512 wall regression outside paired spread.
- If ISA is unchanged, resource usage worsens, or gain is <3%, revert B2. Do not invent a `sched_barrier` fallback: there is no source-backed need once the experiment fails.

Conditional estimate: 3–10% of the post-B1 attention bucket = 0.2–0.8 ms pp512, 3–13 ms pp2048, 0.6–2.7 s pp32768. It is multiplicative with B1.

## 9. Unit C2 — producer-emitted IU4 sidecars

### 9.1 Correct scope

The standalone quantizer emits a 72-byte block containing f32 scale, exact i32 sum, and 64 nibble bytes (`kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip:52-60`). It uses an amax reduction, eight MSE candidates, strict-`<` winner selection, exact `rintf`/clamp, f32 single-rounding FMAs, and an integer sum reduction (`:62-139`). C2 must run that same arithmetic. It removes a separate global reload/launch and, where safe, the producer's f32 store; it does **not** remove the quantization work.

The host entry points are `Gpu::fused_rmsnorm_rotate_mq_batched` (`crates/rdna-compute/src/gemv.rs:2778-2850`), `Gpu::fused_silu_mul_rotate_mq_batched` (`:2915-2977`), and the AWQ SwiGLU sibling `Gpu::fused_silu_mul_rotate_mq_awq_batched` (`:3055-3122`). Their runtime sources are wired by `FUSED_RMSNORM_MQ_ROTATE_{,AWQ}_SRC` and `FUSED_SILU_MUL_MQ_ROTATE_{,AWQ}_SRC` (`crates/rdna-compute/src/kernels.rs:1062-1089,1104-1105,1141-1144`).

The existing scratch allocator always launches the quantizer and lays blocks `[K/128,batch]` (`crates/rdna-compute/src/scratch.rs:1203-1294`). The current RMSNorm producer materializes its final eight f32 outputs just before storing them (`kernels/src/fused_rmsnorm_mq_rotate.hip:207-220`); the SwiGLU producer likewise owns exactly one 256-element group and its final eight outputs per lane (`kernels/src/fused_silu_mul_mq_rotate.hip:101-183`). Those are the only producer points C2 may use.

First tranche only:

- 128 RMSNorm/FWHT family inputs per outer chunk: 48 LA attention + 48 LA FFN + 16 FA attention + 16 FA FFN;
- 64 SwiGLU/FWHT down inputs;
- leave the 64 post-attention/output-projection inputs alone because they are produced by gated/scattered epilogues, not either contiguous producer.

The profile call count should move from 256 to **64** standalone quantizer launches per outer chunk if every first-tranche route is active.

### 9.2 Frozen type-state interface

Follow the existing prepared-pointer-view convention established by `Mq4v2Fp8Prepared` (`crates/rdna-compute/src/scratch.rs:20-33`) but make the new fields opaque, and add the consumed reservation state required by producer emission:

```rust
pub struct Int4MmqReservation {
    ptr: *mut c_void, k: usize, n: usize, generation: u64,
}
pub struct Int4MmqPrepared {
    ptr: *mut c_void, k: usize, n: usize, generation: u64,
}
```

Add `int4_mmq_generation: u64` beside the existing dedicated scratch fields. `reserve_int4_mmq` uses checked increment and returns `HipError` on the impossible overflow rather than wrapping and revalidating an ancient handle. Freeze these methods:

```rust
Gpu::reserve_int4_mmq(k, n) -> HipResult<Int4MmqReservation>
Gpu::fused_rmsnorm_rotate_mq_i4_batched(..., reservation, emit_f32: bool)
    -> HipResult<Int4MmqPrepared>
Gpu::fused_silu_mul_rotate_mq_i4_batched(..., reservation)
    -> HipResult<Int4MmqPrepared>
```

`reserve_int4_mmq` grows but does not launch; every reservation increments the generation. A producer consumes `Int4MmqReservation` and returns `Int4MmqPrepared` only after its launch is enqueued. Prepared consumers validate current generation, `k`, `n`, exact route, and pointer before launch. A later reservation invalidates every older handle. This strengthens the existing prepared-view lifetime because producer reservation and production are separate calls; it is required even when the allocation pointer does not change. `ensure_int4_mmq_x` itself always launches (`crates/rdna-compute/src/scratch.rs:1237-1294`), so the prepared route removes that call rather than interacting with the unrelated FP16/FP8 pointer caches. Same-stream launch ordering is the device-ready contract; no host synchronization is added.

Add prepared family entry points in `crates/rdna-compute/src/gemm.rs` which reuse the existing prequant consumers rather than duplicating them:

```rust
Gpu::gemm_qkvza_mq4g256v2_wmma_prepared_iu4(..., &Int4MmqPrepared)
Gpu::gemm_qkv_mq4g256v2_wmma_prepared_iu4(..., &Int4MmqPrepared)
Gpu::gemm_gate_up_mq4g256v2_wmma_prepared_iu4(..., &Int4MmqPrepared)
Gpu::gemm_mq4g256v2_residual_prepared_iu4(..., &Int4MmqPrepared)
```

They validate the same eager, batch≥128, batch%128, K%256, exact-arch, and IU4-flag predicates used by the current qkvza/qkv/gate-up/residual paths (`crates/rdna-compute/src/gemm.rs:27975-27984,28518-28526,29626-29634,31148-31156`). They call `gemm_mq4g256v2_mmq_{set,add}_prequant_iu4`; they never call `ensure_int4_mmq_x`.

### 9.3 Exact lane remap

Extract the current 128-element quant recipe into one runtime-JIT common source fragment, then concatenate that fragment into the standalone IU4 and specialized producer modules in `crates/rdna-compute/src/kernels.rs`. Do not maintain three independently edited copies of the MSE recipe. This C2 slice may take ownership of `kernels.rs` only after the in-flight `MmqLutU2b` work has landed; until then it is read-only.

For one 256-element producer group and wave lane `p`, the final registers are values `8p..8p+7`. Quant wave lane `l` must own the original standalone mapping `4l..4l+3` (`kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip:80-94`). For half `h∈{0,1}`:

- source producer lane is `16*h + (l >> 1)`;
- even `l` selects that producer lane's first final float4;
- odd `l` selects its second final float4;
- four `__shfl` operations materialize the same four f32 bit patterns;
- invoke the shared quant recipe and write sidecar index `(2*group+h)*N + token`.

Run h0 then h1. Preserve the current shuffle-reduction order 16,8,4,2,1, strict-`<` tie behavior, f32 FMA sites, and final nibble packing (`kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip:91-139`). RMSNorm's eight-wave block applies the same mapping independently within each wave/group; its current group loop and uniform tail break are at `kernels/src/fused_rmsnorm_mq_rotate.hip:122-149`.

Compile plain and AWQ-specialized producer symbols without changing the old symbols or their ABIs. RMS AWQ remains derived from the uniform source by `HIPFIRE_RMSNORM_AWQ`; SwiGLU AWQ remains the separate, structurally matched `kernels/src/fused_silu_mul_mq_rotate_awq.hip`. The RMS source has distinct exact multiply orders for AWQ and plain (`kernels/src/fused_rmsnorm_mq_rotate.hip:151-171`), and the SwiGLU AWQ divide occurs before signs/FWHT (`kernels/src/fused_silu_mul_mq_rotate_awq.hip:46-116`), so quantization consumes each variant's final post-scale values without refactoring its arithmetic.

### 9.4 Producer/store decisions and call sites

Exact gfx1151/MQ4V2/IU4/eager branches are inserted at these producer+consumer pairs:

- LA input RMSNorm at `crates/hipfire-arch-qwen35/src/qwen35/prefill.rs:4443-4458`, then qkvza at `:4470-4695`;
- LA FFN RMSNorm at `:5353-5404`, then gate/up at `:5406-5549`;
- FA input RMSNorm at `:5770-5792`, then qkv at `:5794-5972`;
- FA FFN RMSNorm at `:6534-6586`, then gate/up at `:6587-6727`;
- LA SwiGLU/FWHT at `:5597-5629`, then down epilogue at `:5633-5643`;
- FA SwiGLU/FWHT at `:6775-6800`, then down epilogue at `:6804-6813`.

Store policy:

- LA qkvza keeps the f32 `x_rot` store because beta/alpha small tails first try the f32 MW4 kernel before falling back to IU4 (`crates/rdna-compute/src/gemm.rs:27889-27945,27987-28004`).
- FA qkv, both gate/up families, and both down families may omit the f32 store only in the exact prepared branch, because their prepared consumer is the sole next reader.
- The original producer symbols always retain the f32 store. If calibration/capture/replay or any non-IU4 path is active, use the original producer and original dispatcher.

For `BatchEpilogue::Residual`, prepared down calls IU4 add into `pbs.x_batch`. For `BatchEpilogue::Partial`, zero the target view then use the same IU4 add consumer, preserving the existing partial contract (`crates/hipfire-arch-qwen35/src/qwen35/prefill.rs:60-87,214-315`). Put this in a dedicated `dispatch_mq4v2_iu4_prepared_epilogue`; do not widen the shared dispatcher with an optional pointer.

### 9.5 State transitions

| State | Scratch ownership | f32 output validity | Legal transition |
|---|---|---|---|
| `FREE` | allocator owns capacity; generation stale | none | reserve → `RESERVED(g,k,n)` |
| `RESERVED` | consumed reservation exclusively names sidecar | none or target f32 not yet written | producer launch on active stream |
| `PREPARED` | read-only to the named GEMM family until next reserve | valid only when `emit_f32=true` | one or more sibling set consumers, or one add consumer |
| `INVALID` | later reserve owns buffer | previous handle forbidden | validation error, never a launch |

After the final consumer, no persistent model state points at the sidecar. The next producer may reuse it.

### 9.6 Gates and abandon rules

Timebox: **one-hour arithmetic screen, then one day for SwiGLU and two days for RMSNorm**, after A and B admissions.

0. **C2.0 diagnostic:** build a throwaway standalone quantizer that evaluates only the first MSE candidate. Its output is numerically invalid and must never feed a model. Interleave it with the exact eight-candidate kernel at `(K,N)={(5120,512),(17408,512)}`. If the weighted median improves by ≤20%, launch/global traffic dominates and C2 proceeds. If it improves by >20%, candidate arithmetic is material and producer fusion will retain that cost: build only the smallest SwiGLU→sidecar→down chain prototype, and continue C2 only if that combined chain is already ≥5% faster than SwiGLU→standalone-quantizer→down. Otherwise abandon C2 before the RMS work. Delete the one-candidate symbol and output after the diagnosis in every outcome.
1. Refactor the standalone quantizer onto the shared recipe and prove its 72-byte blocks remain byte-identical before adding producers.
2. Producer oracle: K={5120,17408}, N={128,512}, plain+AWQ as applicable, deterministic random data plus all-zero, signed-zero, clipping, equal-MSE/tie, and final partial-grid fixtures. Compare all 72 bytes/block against producer→store→standalone-quantize. Any mismatch rejects C2.
3. Chain oracle: compare old producer→quantizer→IU4 projection with prepared producer→IU4 projection for qkvza/qkv/gate-up/down, set/add, and partial epilogue. Require f32 output bits where retained, projection output bits, and guards all equal.
4. Require zero spills. A no-store producer must not materialize or invalidate an f32 tensor that a later call reads.
5. Performance gate per tranche: the combined specialized producer + consumer time must beat the old producer + quantizer + consumer by **≥10% for that pair**, pp512 wall must improve beyond paired spread, and no untargeted kernel may regress >2%.
6. Final route gate: standalone quantizer call count must be 64/chunk, not merely renamed; if it is not, find the missed first-tranche call before measuring.
7. If either producer spills or saves <10% for its pair, revert that producer independently. If both fail, remove the common interface and C2 route entirely.

Planning estimate, not a ceiling: 10–30 ms pp512, 40–120 ms pp2048, 0.64–1.92 s pp32768. The upper bound assumes meaningful removal of global f32 traffic; no part of the measured 39.424 ms quantizer bucket is counted as “free” until the combined-chain timer proves it.

## 10. Unit C1 — Q8-EF GDN with transient F32 chunk carry

> **STATUS 2026-09-15 — C1 REJECTED at C1.0.** Halo, 48 heads, HD 128, T=512,
> warm median-of-21: shipping `gated_delta_net_q8_fast` 1260 us; existing F32
> chunked algebra (sum of per-chunk launches) CS16 15663 us, CS32 13247 us —
> ratio 10.5x, gate was ≤0.60. Parity held on all seven T/CS pairs (<2.3e-7).
> The F32 chunk kernel is one WG per head with host-serialized launches; it is
> a correctness oracle, not a speed vehicle. No Q8 bridge was written. A
> future C1 needs a new chunked kernel design (multi-WG per head, on-device
> chunk loop), which is outside this plan's four-day box.


### 10.1 Source audit correction

The profiled timer name is `gated_delta_net_q8_batch_seq`, but the default selector chooses `gated_delta_net_q8_fast` when per-token requant is disabled (`crates/rdna-compute/src/norm.rs:3000-3016`). That launch is `[n_heads,32,1]`, block 32 (`:3018,3041-3068`). The actual fast kernel dequantizes each four-row tile once, walks all tokens serially in FP32 LDS, and requantizes **once outside the token loop** (`kernels/src/gated_delta_net_q8_fast.hip:5-7,122-145,258-317`).

Therefore the brainstorm's proposed Q8 requant at each internal CS boundary would not preserve the shipping state cadence. The older boundary-EF experiment showed bounded ~7e-3 error, but explicitly changed cadence (`docs/plans/chunked-gdn.md:102-117,125-155`). This plan does not make that product change.

The algebra kernel already exists: one WG/head, host-serialized chunks, CS≤32, about 45.7 KiB LDS (`kernels/src/gated_delta_net_f32_chunked.hip:7-50`). Its inclusive cumsum, strict-lower solve, output, and state-carry equations are implemented at `:52-106,156-272`; the host clamps CS to 32 and launches one kernel per chunk (`crates/rdna-compute/src/norm.rs:3759-3874`).

### 10.2 Frozen implementation

Add one transient F32 state tensor to `PrefillBatchScratch`, sized `[linear_num_value_heads,128,128]`, allocated once and reused layer-by-layer. Existing scratch allocation is centralized in `crates/hipfire-arch-qwen35/src/qwen35/batch.rs:177-276,443-514`; no per-layer allocation is allowed.

Add `kernels/src/gated_delta_net_q8_chunked_bridge.hip` with two symbols:

```c
gated_delta_net_q8_dequant_state(... s_q8, s_scales, s_f32_tmp, n_heads)
gated_delta_net_q8_requant_state(... s_f32_tmp, s_q8, s_scales,
                                  s_ef_residual, frame, n_heads)
```

Before admission, route C1 only when `HIPFIRE_GFX1151_GDN_Q8_CHUNKED=1`; rejection deletes the arm. The ingress predicate is exact gfx1151, `BatchSemantics::Sequential`, `n>1`, `n_v_heads==48`, `head_dim==128`, Q8 state with EF present, default single-end requant cadence, and neither retained-replay recording nor graph capture. After C1.1, C1.2, and §13.5 all pass, flip only that ordinary single-GPU product route to default-on and retain `=0` as the documented rollback.

- Dequant maps every state cell exactly as fast-kernel load: `scale[row] * (float)code` (`kernels/src/gated_delta_net_q8_fast.hip:133-143`). It does not fold EF.
- Requant maps one wave/row with the same four elements/lane, folds the prior f16 EF before absmax, performs the same XOR max tree, scale, `rintf`, clamp, code store, scale store, and fresh EF update as `gated_delta_net_q8_fast` (`:258-297`).
- Route requires `ef_residual.is_some()` and default single-end cadence. Stochastic/no-EF and explicit per-token-requant modes stay on the existing path.

Refactor the private launch loop behind `Gpu::gated_delta_net_f32_chunked` so C1 can call it without nesting profile timers. Add:

```rust
Gpu::gated_delta_net_q8_chunked_f32_carry(
    q,k,v,gate,beta,s_q8,s_scales,s_f32_tmp,out,n,heads,128,ef,cs
)
```

The wrapper executes:

1. reserve the same `n_tokens` frame IDs as the current batch wrapper so global frame progression is unchanged (`crates/rdna-compute/src/norm.rs:3031-3039`);
2. dequant persistent Q8/scales → transient F32, leaving Q8/scales/EF untouched;
3. run the existing F32 chunk algebra over CS=16 and CS=32 candidates, advancing only transient F32 state and writing output;
4. after all internal chunks succeed, fold EF and requant transient F32 → persistent Q8/scales/EF exactly once.

### 10.3 State transaction

| Transition | Persistent Q8/scales/EF | Transient F32 | Output | Invariant |
|---|---|---|---|---|
| entry | caller-owned current state | dead | target | current outer-chunk boundary |
| dequant | unchanged | exact decoded state | unchanged | EF not folded early |
| internal chunk `ci` | unchanged | `S_ci → S_ci+1` in place | rows `ci*CS..` written | chunks serialized; no Q8 seam |
| pre-commit | unchanged | full chunk-final F32 state | all rows written | a failed internal launch has not committed persistent state |
| requant commit | Q8/scales/EF replaced once | dead after enqueue | unchanged | same outer-boundary cadence as fast kernel |
| success | caller owns new persistent state | reusable scratch | caller owns output | no scratch pointer escapes |

This is a logical transaction, not a rollback mechanism. A device failure during the final requant can still partially write persistent state, as a failure in the current kernel can; no stronger guarantee is claimed.

In `batch_chunk_delta_net_attn`, replace only the ordinary sequential Q8 call to `Gpu::gated_delta_net_q8_batch_seq` (`crates/hipfire-arch-qwen35/src/qwen35/prefill.rs:5287-5337`) when the frozen ingress predicate holds. `batch_chunk_delta_net_pre_gdn` has already expanded and normalized Q/K to `n_v_heads` (`:4942-4978`), so the existing F32 algebra's equal-head layout remains valid. Tree parents, independent/masked batches, Q4, FP32, n=1 AR, replay/capture, TP/EP-sharded head shapes, and speculative verify remain on their current kernels.

### 10.4 Gate ladder

Timebox: **two-hour screen, then at most four engineer days**.

**C1.0 — no-new-Q8-kernel screen.** Extend the throwaway `gdn_chunk_parity` benchmark to `N_HEADS=48`, T=512, CS={16,32}, and add a shipping `gated_delta_net_q8_batch_seq` arm initialized from the same deterministic state. Upload/initial-state cloning remains outside the timed interval. The harness already has the F32 sequential/chunk differential and warm-median structure (`crates/rdna-compute/examples/gdn_chunk_parity.rs:129-249`). Require:

- existing F32 output and state parity `<1e-4` for T/CS `(16,16),(32,16),(32,32),(37,16),(33,8),(512,16),(512,32)` (`crates/rdna-compute/examples/gdn_chunk_parity.rs:5-24,41-127`);
- the sum of existing F32 chunk launches at T512/CS32 is **≤60%** of the matched shipping Q8-fast median. This leaves a measured 15-point budget for the two bridge launches while still making the final ≥25% gate possible.

If either fails, abandon C1 before writing the Q8 bridge. Speed versus the FP32 sequential kernel is diagnostic only; the Q8-fast arm is the relevant denominator.

**C1.1 — Q8 state/numerical oracle.** Extend the harness with cloned Q8 codes/scales/f16 EF and T/CS pairs `(1,1),(16,16),(33,16),(128,16),(128,32),(512,16),(512,32)`, in both near-1 and mid-decay regimes. Compare candidate, shipping Q8-fast, and FP32 truth. Define output relative infinity error as `max_abs(candidate-truth)/max(max_abs(truth),1e-8)` and state relative L2 as `l2(candidate_state-truth_state)/max(l2(truth_state),1e-8)`. Require:

- no NaN/Inf and no guard corruption;
- for every fixture, candidate output relative-infinity error and dequantized final-state relative-L2 error are each no worse than the corresponding shipping Q8-fast error plus **5e-4**;
- at T512, candidate output relative-infinity error is ≤`8.0e-3` in the documented near-1 fixture and ≤`5.5e-3` in the mid-decay fixture; CS16 and CS32 candidate errors differ by ≤`5e-4` (the reference bands are `docs/plans/chunked-gdn.md:125-155`);
- exactly one persistent Q8/scales/EF commit per outer segment, proven by sentinel snapshots before the final bridge;
- global frame counter advances by the same `n_tokens` reservation as baseline.

Any failure rejects C1; do not loosen tolerances post hoc.

**C1.2 — performance.** Profile the combined wrapper, including dequant, all internal chunk launches, and requant. Admit only if:

- combined warm median is ≥25% faster than the matched shipping Q8-fast median at N512 (1,340 µs/call in the supplied profile), with zero spills in all three kernels;
- CS choice transfers at N128,256,512; select CS16 or CS32 solely by the best geometric mean, provided no tested N regresses >2%; no other CS is in scope;
- production pp512 and pp2048 wall improve beyond paired spread;
- full model state/quality gates in §13.5 pass.

The conditional planning range, only after C1.0 passes, is 50–70% of the GDN bucket: 32–45 ms pp512, 129–180 ms pp2048, 2.06–2.88 s pp32768. If measured gain is 25–49%, the code may still be useful but its measured value—not this range—enters the campaign ledger. If <25%, revert C1.

## 11. Gate F — prefill chunk 1024, only after FA2 and GDN compatibility

The existing developer override accepts values above 512 (`crates/hipfire-arch-qwen35/src/qwen35/prefill.rs:418-422`), but gfx1151's model default remains 512 (`:455-462`). The generic FA2 route admits only batch 64..512 step 16 and otherwise falls back to the incumbent (`crates/rdna-compute/src/attention.rs:3259-3286`); the direct launcher also rejects batch>512 (`:3889-3896`). Setting `HIPFIRE_PREFILL_MAX_BATCH=1024` today is therefore not evidence for a chunk win.

All of the following are mandatory before timing 1024:

1. **FA2 direct envelope:** extend the temporary direct launcher to N=1024. Prove one N1024 launch is bit-identical to two disjoint N512 launches for Q8-K and fwht3-K, including a Q clone for each fwht3 arm. Add `(1024,0,1024)`, `(1024,7168,8192)`, and `(1024,31744,32768)` to the B oracle. The kernel grid itself is `ceil(batch/8)` and uses per-WG positions (`kernels/src/attention_q8_0_fa2_gqa.gfx11.hip:15-17,167-206`), but that is only a reason to test, not proof of compatibility.
2. **Host envelope:** only after (1), extend both generic and direct gfx1151 gates to 1024. Other gfx11 architectures stay capped at 512 unless separately measured.
3. **Scratch/capacity:** construct `PrefillBatchScratch` at 1024 and pass every checked capacity path. The outer code derives owned PBS capacity and mins chunk size with PBS/hidden-ring capacity (`crates/hipfire-arch-qwen35/src/qwen35/prefill.rs:1329-1348`).
4. **GDN cadence:** preserve the current 512-token persistent-state boundary. A 1024 outer chunk must invoke either shipping Q8-fast twice on slices `[0,512)`/`[512,1024)`, or admitted C1 twice as `dequant→internal chunks→requant` macrosegments. One 1024-token fast launch or one C1 final requant would remove the former midpoint Q8-EF commit and is forbidden.
5. **Prepared-IU4 lifetime:** each 1024 producer reservation/consumer pair remains within a layer and generation-valid; no handle crosses a GDN macrosegment or layer.
6. **Numerical/state oracle:** compare N512 outer chunking with N1024 outer chunking through one full 2048-token prefill. A/B/C2 outputs must be bit-identical; Q8/scales/EF snapshots at positions 512,1024,1536,2048 and final logits must match exactly when C1 is off. With C1 on, each snapshot must satisfy the frozen C1 tolerances and use the same two commits per 1024.

Timebox after prerequisites: **two engineer hours**.

Run fresh-process ABBA at pp2048/8192/32768. Admit a gfx1151/all-MQ4V2 default of 1024 only if production wall improves **≥3% at all three prompts**, pp512 and tg1 remain within 2%, every target kernel stays on its intended route, and no state/quality gate fails. If <3%, restore 512 and remove the 1024 route. Bank **0 ms** before measurement; the brainstorm's 40–75 ms / 0.9–1.7 s estimates are not used because they omitted the GDN-cadence constraint.

## 12. Predicted deltas and row-clear checkpoints

All entries are conditional planning estimates after each unit's own admission gate. They are not additive measurements; each later row must be re-based on the accepted predecessor.

| Unit | Target bucket | pp512 | pp2048 | pp32768 | Admission floor |
|---|---|---:|---:|---:|---|
| A2 | IU4 | 6.9–20.6 ms | 27–82 ms | 0.44–1.32 s | >1% weighted IU4 |
| A1 | IU4 | 68.6–102.9 ms | 274–412 ms | 4.39–6.59 s | ≥10% weighted IU4 |
| B1 | attention | 0.8–2.0 ms | 12–34 ms | 2.8–7.4 s | ≥10% attention |
| B2 | post-B1 attention | 0.2–0.8 ms | 3–13 ms | 0.6–2.7 s | ≥3% incremental |
| C2 | producer/quant pair | 10–30 ms | 40–120 ms | 0.64–1.92 s | ≥10% per fused pair |
| C1 | GDN | 32–45 ms | 129–180 ms | 2.06–2.88 s | ≥25% measured; estimate assumes 50–70% |
| F | whole wall | 0 | **0 banked** | **0 banked** | ≥3% measured at 2K/8K/32K |

Conditional cumulative admitted ranges, excluding F:

| Checkpoint | pp512 saved | pp2048 saved | pp32768 saved |
|---|---:|---:|---:|
| after A2 | 6.9–20.6 ms | 27–82 ms | 0.44–1.32 s |
| after A1 | **75.5–123.5 ms** | 301–494 ms | 4.83–7.91 s |
| after B1/B2 | 76.5–126.3 ms | 316–541 ms | 8.23–18.01 s |
| after C2 | 86.5–156.3 ms | 356–661 ms | 8.87–19.93 s |
| after C1 | 118.5–201.3 ms | **485–841 ms** | **10.93–22.81 s** |

Row-clear policy:

- **pp512:** first possible checkpoint is after A1, because the minimum admitted A2+A1 estimate (75.5 ms) exceeds the exact 72.439 ms gap. It is cleared only when fresh production wall is ≤825.806 ms. If A1 is rejected, the first possible checkpoint moves to C2/C1 and is not guaranteed.
- **pp2048:** no conservative checkpoint clears 812.244 ms. The high end first crosses after C1 (841 ms). With every non-A1 unit at its high estimate, A1 still needs about 384 ms, or 14% of the IU4 bucket. Report the measured row; do not claim it from component sums.
- **pp32768:** the credible envelope does **not** clear 28.793838 s. Even the high estimate leaves about 5.98 s. With every non-A1 unit at its high estimate, A1 would need about 12.57 s—28.6% of the IU4 bucket—close to the retired 30% stretch and inconsistent with the new MAC/issue evidence. If the final wall remains above 57.893993 s, the plan-of-record result is “quadratic exact attention plus the 40-CU issue denominator remains,” not another speculative kernel.

## 13. Exact validation protocol (GPU owner runs on hipx)

### 13.1 Identity and measurement policy

All Halo commands run with:

```bash
cd /home/kaden/ClaudeCode/warpfront/wt-lloyd
export ROCR_VISIBLE_DEVICES=1
export HIPFIRE_GFX11_MQ4V2_IU4=1
export HIPFIRE_DPM_WARMUP_SECS=10
unset HIPFIRE_GFX11_FA2_PREFILL
unset HIPFIRE_PREFILL_MAX_BATCH
export MODEL=/absolute/path/to/qwen3.8-27b.mq4-xt
```

The registry identity is `qwen3.8:27b-mq4-xt`, file `qwen3.8-27b.mq4-xt`, default KV Q8 (`registry/v1.json:954-960`). Record commit/tree state, binary md5, model md5, deterministic prompt md5, GPU/ROCm identity, all `HIPFIRE_*` values, fresh/resident policy, warmups, and run order; those fields are mandatory evidence (`docs/methodology/perf-benchmarking.md:28-38`).

Promotion timing uses **three ABBA cycles** (six fresh processes/arm), raw samples archived, min/median/max reported. ABBA is on top of matched fresh processes, never a substitute for them (`docs/methodology/perf-benchmarking.md:87-96`). Run attribution profiles separately from production wall.

### 13.2 Per-kernel attribution

For each relevant prompt and arm in a fresh process:

```bash
cargo run --release -p saddle-lab --example profile_prefill_qwen35 \
  --features deltanet -- "$MODEL" --prefill 512 --warmup 3 --kv-mode q8
cargo run --release -p saddle-lab --example profile_prefill_qwen35 \
  --features deltanet -- "$MODEL" --prefill 2048 --warmup 3 --kv-mode q8
cargo run --release -p saddle-lab --example profile_prefill_qwen35 \
  --features deltanet -- "$MODEL" --prefill 8192 --warmup 3 --kv-mode q8
```

The profiler wraps one `forward_prefill_batch`, resets DeltaNet state for warmup and measured passes, and aggregates calls/total/average per symbol (`crates/saddle-lab/examples/profile_prefill_qwen35.rs:123-177,179-239`). Use pp512 for A/C2/C1, pp8192 for B, and pp2048 as the transfer bucket. Do not use profiled wall as throughput evidence.

For each new or changed symbol:

```bash
./target/release/hipfire profile qwen3.8:27b-mq4-xt \
  --kernel '<exact-symbol-substring>' --json
```

Archive VGPR, SGPR, LDS, scratch/private segment, and occupancy. `scratch_bytes != 0` is a hard rejection for every kernel in this plan. The established resource table treats zero scratch as the no-spill criterion (`docs/plans/hfq4g256-occupancy-ab-results.md:57-70`). FA2's reported byte count is the incumbent analytical upper bound, not measured FA2 traffic (`crates/rdna-compute/src/attention.rs:3953-3958`); use event duration, never its derived GB/s, for B gates.

### 13.3 Production matrix

Each ABBA sample is a fresh invocation with `--runs 1`; use identical binaries/config except the temporary unit selector:

```bash
./target/release/hipfire bench qwen3.8:27b-mq4-xt --matrix \
  --pp 512,2048,8192,32768 --ctx 128,2048,32768 \
  --runs 1 --warmups 10 --kv-mode q8
```

After each accepted unit, compare measured production wall against the exact targets in §2.1. A symbol win that fails to transfer is not a row clear.

### 13.4 Kernel-specific oracles

- **A2/A1:** temporary old/new IU4 twin differential for full set `M={1024,6144,10240,12288,17408}`, add `M=5120`, `K={5120,6144,17408}`, `N={128,512}`, initialized nonzero add destinations, guard zones, and bitwise f32 output. A1 also requires the dominant 100-iteration device-timed microbench.
- **B1/B2:** rebuilt exact seven shapes and both K modes from §7.4, with independent Q clones for fwht3. Bitwise incumbent/old-FA2/new-FA2 output, old/new post-transform Q, and guards.
- **C2:** exact sidecar/chain differential from §9.6, including all 72 bytes of every `block_i4_128`.
- **C1:** extended `gdn_chunk_parity` ladder from §10.4. This is the only tolerance-based kernel oracle.
- **F:** N1024-vs-two-N512 attention differential plus 2048-token full-state boundary snapshots from §11.

A failed oracle ends the unit before performance work.

### 13.5 Product behavior and state isolation

The serve harness resolves and prints concrete sampling/config without running via `--show-config`, and its battery records prefill/decode plus runaway/empty/coherence evidence (`scripts/serve_harness.py:3-25,3431-3583`). Run:

```bash
python3 scripts/serve_harness.py --model "$MODEL" --tag qwen3.8:27b-mq4-xt \
  --kv q8 --mtp off --dflash off --mode battery --sampling registry \
  --thinking off --max-tokens 512 --max-seq 32768 --seed 0 --show-config
python3 scripts/serve_harness.py --model "$MODEL" --tag qwen3.8:27b-mq4-xt \
  --kv q8 --mtp off --dflash off --mode battery --sampling registry \
  --thinking off --max-tokens 512 --max-seq 32768 --seed 0 \
  --out .redline-work/halo-prefill-<arm>-battery.json
```

Acceptance: 5/5 finish normally, no empty/runaway/attractor failure, and no decode regression >2%. For A/B/C2/F with C1 off, compare baseline/candidate deterministic transcripts byte-for-byte. For C1, product text need not be byte-identical; its numerical gate, WT2 quality gate, and coherent battery are authoritative.

Run the resident phase/state harness after the final admitted stack:

```bash
python3 scripts/redline_daemon_harness.py --model "$MODEL" \
  --daemon target/release/daemon --prefill 128 512 --decode-context 128 \
  --kv-mode q8 --state-quant q8 --max-seq 32768 \
  --capture-repeats 2 --measure-repeats 5 --shadow-iterations 4 \
  --out .redline-work/halo-prefill-redline.json \
  --log .redline-work/halo-prefill-redline.log
```

The harness explicitly treats Q8-EF state parity and nonzero replay evidence as required (`scripts/redline_daemon_harness.py:105-180`) and fails unless prefill/decode capture sequence, AQL contracts, and shadow parity pass (`:499-619`). No unit in this plan is admitted into capture/replay; this run proves those existing paths remained intact.

For C1, additionally run the repository's full-vocabulary WT2 KLD path and require candidate KLD ≤ baseline + 0.0005, matching the existing promotion margin cited for FA2 (`CHANGELOG.md:11`). Archive the command, reference digest, and raw output with the campaign ledger.

## 14. Composer slices, dependencies, and file ownership

No composer runs a project-wide formatter, linter, build, or test suite. Each runs only its unit oracle/smoke/profile; the main seat owns one shared validation pass after all accepted slices land.

### Slice A — IU4 A2 then A1

**Exclusive files while active**

- `kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip`
- `crates/rdna-compute/src/gemm.rs`
- temporary IU4 differential example, if needed

**Frozen output contract**

- same public GEMM APIs and symbols;
- A2 host/kernel LDS constants agree;
- A1 leaves `vec_dot_i4_x128` and output stores unchanged;
- no permanent developer flags after gate.

A2 is measured and either committed or reverted before A1 starts. A-lane sends the accepted source hash and measured weighted IU4 delta to the main seat.

### Slice B — FA2 B1 then B2

**Exclusive files while active**

- `kernels/src/attention_q8_0_fa2_gqa.gfx11.hip`
- `crates/rdna-compute/src/attention.rs` only for temporary twin launch support
- temporary FA2 oracle example

**Frozen output contract**

- production entry ABIs, grid, block, and 32,768-byte LDS unchanged;
- K/V plane byte offsets unchanged;
- B1 incumbent/old/new output bits and old/new post-transform Q bits identical for seven shapes × two K modes;
- B2 is an independent commit and can be reverted without reverting B1.

Source work may be prepared concurrently with Slice A because the ownership sets are disjoint, but the main seat performs promotion gates in A2→A1→B1→B2 order so every denominator has one predecessor.

### Slice C2 — producer sidecars

**Starts only after** Slice A is sealed and `MmqLutU2b` has released `crates/rdna-compute/src/kernels.rs`.

**Exclusive files while active**

- new common IU4 quant source fragment
- `kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip` for common-recipe extraction
- `kernels/src/fused_rmsnorm_mq_rotate.hip`
- `kernels/src/fused_silu_mul_mq_rotate.hip`
- `kernels/src/fused_silu_mul_mq_rotate_awq.hip`
- `crates/rdna-compute/src/kernels.rs`
- `crates/rdna-compute/src/scratch.rs`
- `crates/rdna-compute/src/gemv.rs`
- `crates/rdna-compute/src/gemm.rs`
- `crates/hipfire-arch-qwen35/src/qwen35/prefill.rs`
- temporary sidecar oracle

**Frozen output contract**

- `Int4MmqReservation → Int4MmqPrepared` generation state machine from §9.2;
- old producer/GEMM ABIs unchanged;
- prepared branch exact gfx1151/MQ4V2/IU4/eager only;
- qkvza retains f32 input for beta/alpha tails;
- standalone quantizer count 64/chunk.

### Slice C1 — Q8-EF GDN

**Starts only after** C2 is sealed because both own `kernels.rs` and `prefill.rs`.

**Exclusive files while active**

- new `kernels/src/gated_delta_net_q8_chunked_bridge.hip`
- `crates/rdna-compute/src/kernels.rs`
- `crates/rdna-compute/src/norm.rs`
- `crates/rdna-compute/examples/gdn_chunk_parity.rs`
- `crates/hipfire-arch-qwen35/src/qwen35/batch.rs`
- `crates/hipfire-arch-qwen35/src/qwen35/prefill.rs`
- `docs/env-vars.md` for the admitted gfx1151 kill switch

**Frozen output contract**

- transient F32 state never escapes PBS;
- one Q8/scales/EF commit per current outer chunk;
- route ordinary eager Q8-EF only;
- all other recurrent routes byte-untouched;
- tolerances and 25% admission threshold from §10.4 cannot be relaxed.

### Slice F — chunk 1024

**Starts only after** B, C2, and C1 final decisions.

**Exclusive files while active**

- `crates/rdna-compute/src/attention.rs`
- `crates/hipfire-arch-qwen35/src/qwen35/prefill.rs`
- `crates/hipfire-arch-qwen35/src/qwen35/batch.rs` only if a checked capacity fix is required
- temporary N1024 oracle

**Frozen output contract**

- exact gfx1151/all-MQ4V2 only;
- direct FA2 remains active at N1024;
- Q8-EF commits remain every 512 tokens;
- 1024 route removed if the ≥3% all-long-prompt gate fails.

## 15. Cross-architecture behavior and explicit non-goals

- **A2/A1:** current IU4 consumer admits exact gfx1100/gfx1151 (`crates/rdna-compute/src/gemm.rs:18518-18525`). The shared source must stay numerically correct on both. Halo owns the performance decision; if a shared change regresses gfx1100 >2%, split the optimized source/symbol by exact arch or reject it.
- **B1/B2:** current direct launcher accepts the gfx11 family (`crates/rdna-compute/src/attention.rs:3867-3878`). Both K modes remain correct across that family; the rebuilt oracle is mandatory on XTX and Halo. No performance result is generalized beyond the measured architecture.
- **C2/C1/F:** new default routes are exact gfx1151 only. gfx1100, gfx11 siblings, gfx12, gfx9/CDNA, and CPU paths retain existing symbols and decisions.
- **Not in scope:** decode (`n=1`), tree/multislot/independent speculative verify, replay/graph capture, distributed TP/EP transport or collective-topology changes (C2 must preserve the existing `BatchEpilogue::Partial` numerical contract if reached), KV format changes, quant recipe changes, f16 attention accumulation, halogen-server changes, or NPU offload.
- **No compatibility shims:** if an admitted interface replaces a call, every target caller migrates in the same slice and the obsolete experimental path is deleted.

## 16. Rejected and deferred levers

### Permanently rejected by assignment evidence

- IU8 register-resident redesign, MMQ round-2 ×3, `s_setprio`, NPU offload, gfx11 B=16 environment arms, KT64 FA2, and split-KV prefill are on the supplied kill list. They are not experiments in this plan.
- KT64 specifically already lost the KT32/KT64 ABBA and the host pins KT32 (`crates/rdna-compute/src/attention.rs:3918-3927`; kernel assumptions at `kernels/src/attention_q8_0_fa2_gqa.gfx11.hip:75-87`).
- Wave64 IU4 is a new kernel because the consumer uses `_w32` WMMA builtins (`kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip:276-281`).
- M-tile 256 would increase A-side LDS enough to reduce block residency; it is not compatible with the current 128×128, 31,744-byte design (`kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip:43-50,315-319`).
- Y-store vectorization attacks the unchanged output epilogue, not the measured read/issue path (`kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip:352-371`).
- f16 accumulator FA2 changes the stated f16-operands/f32-accumulator contract and is a numerical redesign (`kernels/src/attention_q8_0_fa2_gqa.gfx11.hip:57-58`).
- Prefill graph capture is out of scope: target IU4 family routes already reject replay/capture before preparation (`crates/rdna-compute/src/gemm.rs:27975-27980,28518-28523,29626-29631,31148-31153`).
- The `gfx1151` compact2/compact3/R8/R4X2/DPP GDN family is decode-only (`Gpu::gated_delta_net_q8_compact` at `crates/rdna-compute/src/norm.rs:2785-2829`). It is not a hidden batched-prefill competitor and is not pulled into C1.

### Deferred until a measured surviving bucket justifies them

- **A3 z half-height / qkv merge:** structural tail is only 3.16% of work and A2/A1 already own the source. Re-open only if A1 passes, z/k/v remain ≥5% of measured IU4 time, and a proposed variant has a ≥15% per-shape gate. The relevant grid and shape behavior are `crates/rdna-compute/src/gemm.rs:18562-18580`.
- **A4 occ2/occ3:** both symbol families already exist (`kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip:374-380`), while LDS caps actual block residency at two. Do not rerun without a ledger proving the Halo arm was never tested.
- **A5 grid-order/L2 reuse:** current grid is row tiles on x and batch tiles on y, matching `row0=blockIdx.x`, `col0=blockIdx.y` (`crates/rdna-compute/src/gemm.rs:18569-18578`; kernel `:308-309`). The cache-reuse claim is unmeasured and is abandoned unless a post-A profile attributes ≥5% stalls/traffic that this exact swap can address.
- **B3 pre-convert Q to f16:** Q is converted from four float4s inside every dc iteration (`kernels/src/attention_q8_0_fa2_gqa.gfx11.hip:373-385`). It adds a scratch layout/launch and is deferred until B1/B2 either fail or leave Q conversion as a measured ≥10% attention slice.
- **B4 helper-wave pipeline:** all four waves cooperatively fill K/V, only waves 0..2 compute, and the turnover barrier protects the single LDS planes (`kernels/src/attention_q8_0_fa2_gqa.gfx11.hip:127-140,208-339,475-477`). No design starts unless a fill-only measurement shows ≥20% of long-context attention time. If not, abandon.
- **C3 quantizer restructuring:** its amax/eight-candidate/final reductions are explicit (`kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip:91-129`) and launch geometry is `[ceil(K/1024),N]`, block 256 (`crates/rdna-compute/src/scratch.rs:1261-1278`). A one-candidate diagnostic may explain C2 results, but it is numerically invalid and cannot ship. No DPP/candidate-parallel rewrite starts unless post-C2 standalone quantization remains ≥3% of pp512 wall and the diagnostic attributes ≥30% of its time to candidate arithmetic.
- **Output-projection quant fusion:** the remaining 64 quantizer calls consume gated/scattered producers, not the contiguous RMSNorm/SwiGLU register groups. Re-open only after C2 passes and those 64 calls remain ≥5 ms pp512.
- **Conv1d:** it is an 11.376 ms pp512 bucket and has no same-owner producer/consumer fusion in this plan. It cannot close any row.

## 17. Global abandon and review gates

1. A unit that misses its predeclared correctness, resource, or speed threshold is reverted before continuing. Failed code is not retained for a later combined win.
2. After each admitted unit, update the wall ledger from fresh production samples. Never substitute component estimates for the observed row.
3. After C1, recompute remaining row gaps using measured wall. If the sum of **measured remaining target buckets** is smaller than a gap, stop that row immediately.
4. F is the final experiment. If the final stack does not beat 620/710/566 t/s, report the exact achieved rows and the measured residual attribution. Do not reopen killed or unmeasured ideas to manufacture a complete-looking roadmap.
5. Every composer provides its scoped oracle output, resource JSON, raw per-kernel samples, and fresh wall samples to the main seat. No composer self-admits.
6. A Fable/foreman/main-seat reviewer has final veto over state parity, source ownership, measurement validity, and promotion. Because this is Sol-authored, sending it to `@reviewer` does not satisfy the review gate.
