# gfx1201 FP8/W4 GEMM differential

## Bottom line

**Trace pair:** radiance reaches **221.279 TFLOP/s**, while the shipped hipfire FP8-v2 symmetric-fold GEMMs reach **143.040 TFLOP/s**: **1.547x**.

**Controlled standalone pair on Card-C at the exact M=8192 gate-up/down shapes:** radiance reaches **255.858 TFLOP/s aggregate**, while shipped hipfire symmetric-fold reaches **167.924 TFLOP/s aggregate**: **1.524x**, or **+52.37%**.

The principal differentiator is:

> **A fold-free, 16-chain per-wave tile enabled by MXFP4 scale folding, coupled to one 16-load scalar-address A clause.**

Register-direct A is not sufficient by itself. Radiance uses the freed A-LDS budget to do twice as many FP8 WMMAs per K16 drain and performs no K128 floating-point fold. Hipfire's killed direct-A IU4 kernel retained the expensive K128 quantization fold, did not cross an occupancy step, and compiled direct A into load-counter staircases inside the WMMA drain. The shipped FP8-v2 kernel has clean staged loads but still retains eight partial and eight running accumulator vectors plus 64 scale FMAs per K128, which constrains it to half radiance's WMMA work per K16 and three times the LDS operand traffic per WMMA.

## Measurement scope and method

- GPU for new measurements: Card-C, AMD Radeon AI PRO R9700, `gfx1201`.
- Card-B was not touched.
- Every standalone row uses five warm-up launches followed by 20 individually HIP-event-timed launches; the reported time is the median of launches 10 and 11 after sorting.
- Inputs were zero-filled, but buffers use each kernel's native packed-weight, activation, scale, and output representation.
- Exact logical shapes:
  - gate-up: `M=8192, N=34816, K=5120`, 2.92057776128 TFLOP;
  - down: `M=8192, N=5120, K=17408`, 1.46028888064 TFLOP.
- Gate-up's `N=34816` is the fused gate-plus-up width. Hipfire writes two 17408-wide outputs; radiance writes one 34816-wide output. Logical dense work is identical.
- These timings isolate the GEMM bodies. They are not a numerical-equivalence claim across MXFP4 and MQ4V2, and zero-filled synthetic data does not reproduce a serving cache history.

The trace headline and controlled standalone headline answer different questions. The **221.279/143.040** pair comes from the shipped serving traces. The **255.858/167.924** pair holds shape, device, warm-up, timing method, and launch count fixed.

## Exact radiance instantiation

The standalone source was compiled with:

```text
hipcc --offload-arch=gfx1201 -O3 --save-temps \
  -Rpass-analysis=kernel-resource-usage \
  -c radiance_mxfp4_fp8.hip -o radiance_mxfp4_fp8.o
```

The M=8192 gate-up/down dispatch selects:

```text
radiance_mxfp4_fp8_gemm_atiled<4, true, 64>
TM=4, TN=4, NS=4, NWAVE=8, LBK=64, WPERM=true, WSLOTS=4
block=(256,1,1)
workgroup output tile=(256 M rows, 128 N rows)
wave output tile=(64 M rows, 64 N rows)
```

Launch geometry:

| Shape | Grid | Workgroups |
|---|---:|---:|
| gate-up | `272 x 32` | 8704 |
| down | `40 x 32` | 1280 |

The gate grid independently proves that the dispatch width is the fused 34816 rows: `34816 / 128 = 272`.

## Resource and occupancy differential

| Kernel | Tile MxN | Threads / waves | Logical SGPR | Logical VGPR | LDS | Scratch/spill | Compiler waves/SIMD | Measured active blocks/CU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| radiance A-tiled `<4,true,64>` | 256x128 | 256 / 8 | 36 | 206 | 9344 B | 0 / 0 | 7 | **3** |
| killed hipfire IU4 A-tiled | 128x128 | 256 / 8 | 40 | 199 | 12288 B | 0 / 0 | 7 | **3** |
| shipped hipfire FP8 symfold gate | 128x128 | 256 / 8 | 32 | 186 | 18944 B | 0 / 0 | 8 | **3** |
| shipped hipfire FP8 symfold down | 128x128 | 256 / 8 | 30 | 191 | 18944 B | 0 / 0 | 8 | **3** |

Notes:

- The fresh compiler resource remark for the exact radiance symbol reports 36 SGPR, 206 VGPR, zero scratch/spill, seven waves/SIMD, and 9344 B static LDS.
- The serving profiler reports allocated radiance values of 208 VGPR and 9728 B LDS. Those allocation-rounded trace columns are consistent with the standalone compile; they are not a second logical register count.
- Hipfire's FP8 census field named `blocks_per_cu` contains compiler waves/SIMD. The table above separates that from the actual block occupancy queried with `hipOccupancyMaxActiveBlocksPerMultiprocessor`.
- **Occupancy is not the answer.** All three baseline designs admit three 256-thread workgroups/CU. Direct A reduced the killed IU4 candidate from 20480 to 12288 LDS bytes and from 208 to 200 allocated VGPR, but crossed no block-occupancy boundary.

## Per-K machine differential

Counts below are normalized to one wave unless a row explicitly says workgroup. FP8 WMMA consumes K16; IU4 WMMA consumes K32, so raw WMMA counts across those instruction types must be interpreted with that K-width difference.

| Property | radiance A-tiled FP8 | shipped hipfire FP8 symfold | killed hipfire direct-A IU4 |
|---|---|---|---|
| Runtime K slab | K64 | K64; scale fold every K128 | K64; quant fold every K128 |
| Output per wave | 64x64 | 64x32 | 64x32 |
| Accumulator organization | one `acc[4][4]`: 16 `v8` chains held across full K | `P[4][2]` partial plus `T[4][2]` running: transient 16 `v8` vectors | eight IU4 accumulator chains, then explicit K128 fold |
| WMMA per K16 / wave | **16 FP8** | **8 FP8** | n/a |
| WMMA per K64 / wave | **64 FP8** | **32 FP8** | **16 IU4** |
| Direct A global loads | 16 x `global_load_b64` per K64 in one `s_clause 0xf`; scalar base plus unsigned lane offset | none in drain; A is staged | 4 x `global_load_b64` per K32; first half split into two `s_clause 0x1`, next half one `s_clause 0x3` |
| Other global loads per lane/K64 | one preshuffled W `b128` plus one scale `b32`; row reference hoisted | two A `b128`, four packed-W `b32`; one scale `b16` per K128 half | two compact-W `b64` per K64 plus K128 metadata |
| LDS operand reads per K16 / wave | 4 W `ds_read_b64`; no A LDS | 4 A plus 2 W `ds_read_b64` | per K32: 2 W `ds_read_b64`; no A LDS |
| LDS operand-read ops / FP8 WMMA | **4/16 = 0.25** | **6/8 = 0.75** | different K32 instruction; not directly comparable |
| Workgroup LDS traffic per K64 | W: 8192 B written after expansion, 32768 B read; A: 0 | A: 8192 B written, 32768 B read; W: 8192 B written after expansion, 16384 B read | compact W: 4096 B written, 8192 B read; no A plane; separate DS/SZ planes |
| A global payload / workgroup/K64 | 32768 B issued; 16384 B logically unique because two N waves reuse each A fragment | 8192 B, then reused through LDS | direct packed IU4 fragments; duplicated across N waves |
| Wait structure | global load completion before W expansion/publish; LDS-scoped fence/barrier; drain waits only on `dscnt` for four W operands | prefetches next slab globally while computing current slab, then single-buffer publish/barrier; staged `dscnt` waits for six operands | `loadcnt` staircases occur inside each eight-WMMA drain: first half waits 4/3/2, second half 2/1/0, in addition to W `dscnt` |
| Inner-loop non-WMMA arithmetic | source/ISA estimate about **47 logical VALU per 64 WMMA = 0.734/WMMA** | gate **4.1719/WMMA**, down **4.0938/WMMA** from the shipped ISA census | K32 bundle itself has no arithmetic VALU, but the K128 fold is at least about 392 logical operations per 32 WMMA, **>=12.25/WMMA**, before address/control |
| K unroll | four K16 drains fully unrolled inside rolled `k0 += 64`; software A prefetch rejected because it raised VGPR to about 230 and was neutral/slower | K64 slab body; next slab global-prefetched, single LDS buffer | K32 bundle fully unrolled; outer group/half loops kept rolled to avoid more than 256 VGPR/spill |
| Scale application | MXFP4 block exponent folded into E4M3 weight bytes during W staging; one final row factor in epilogue | symmetric zero term removed, but per-K128 scale still multiplies 64 outputs/wave | full K128 IU4 fold remains, including conversion and scale/sum arithmetic |

### Why the 16-load A clause matters

Radiance's A producer writes WMMA-fragment order. Every wave can fetch all 16 A fragments for K64 through a single scalar-addressed clause, then spend the drain on W LDS reads and 64 WMMAs. The killed hipfire direct-A experiment proved that “global-to-register” alone does not guarantee that schedule: vector-addressed loads were split into short clauses and `loadcnt` dependencies surfaced among the WMMAs. Saving the A LDS plane therefore did not translate into a clean compute drain.

### Why the scale fold matters more than direct A

The shipped symmetric fold removes the zero-point term, but not the K128 scale boundary. It must preserve both eight partial vectors and eight running vectors while performing 64 scale FMAs per wave/K128. That register/dataflow contract caps the useful wave tile at 64x32.

Radiance's power-of-two block scale is absorbed while expanding the weights. Its 16 accumulators can remain live through the full K loop with no K128 output fold. It therefore issues 16 FP8 WMMAs per K16, versus hipfire's eight, while consuming only four LDS operand reads versus six. In normalized terms, radiance uses **one third the LDS read instructions per FP8 WMMA**.

The larger 256x128 workgroup tile also shares each staged W tile across twice as many M rows. It pays twice hipfire's direct/staged A bytes per output element, because A is loaded separately by two N waves, but eliminates A LDS entirely and halves W global bytes per output. On this large-M card, that trade wins.

## Standalone timings

### Baselines

| Shape | Kernel | Median us | Mean us | TFLOP/s | Grid | Active blocks/CU |
|---|---|---:|---:|---:|---:|---:|
| gate-up | radiance A-tiled | 12244.515 | 12376.859 | **238.521** | 272x32 | 3 |
| gate-up | hipfire shipped symfold 128x128 | 18059.408 | 17778.281 | **161.721** | 272x64 | 3 |
| down | radiance A-tiled | 4877.718 | 4873.636 | **299.380** | 40x32 | 3 |
| down | hipfire shipped symfold 128x128 | 8029.042 | 8037.906 | **181.876** | 40x64 | 3 |
| combined | radiance A-tiled | **17122.233** | — | **255.858** | — | — |
| combined | hipfire shipped symfold | **26088.450** | — | **167.924** | — | — |

Standalone radiance is 47.5% faster in gate-up, 64.6% faster in down, and **52.37% faster for the combined logical work**.

### One controlled hipfire variant: tile aspect only

The only hipfire variant changed the constant-area workgroup tile from 128x128 to 256x64. BK=64, eight waves, format, WMMA sequence, fold, build flags, inputs, and timing method remained fixed.

| Shape | Tile | Median us | TFLOP/s | Delta vs 128x128 | LDS | Active blocks/CU |
|---|---:|---:|---:|---:|---:|---:|
| gate-up | 256x64 | 16867.736 | 173.146 | **+7.07%** | 23296 B | 2 |
| down | 256x64 | 8654.510 | 168.732 | **-7.23%** | 23296 B | 2 |
| combined | 256x64 | 25522.246 | 171.649 | **+2.22%** | — | — |

This isolates tile aspect/W reuse as a secondary effect. It helps the very wide gate projection, hurts down, and recovers only 2.22% aggregate while dropping occupancy from three to two blocks/CU. It cannot explain the 52.37% controlled gap. Radiance's advantage comes from the larger tile **together with** its fold-free accumulator lifetime, preshuffled W path, and long direct-A clause—not from BM=256 alone.

## Relation to the killed direct-A experiment

The real-daemon experiment was correct and still lost:

- pp8192 throughput: **-5.201%** and **-4.807%** in two paired runs;
- traced GEMM set time: **+5.767%**;
- traced GEMM add time: **+10.846%**;
- producer time: **+2.545%** aggregate;
- candidate resources: 12288 B LDS, 200 allocated VGPR, three blocks/CU.

This is consistent with the ISA differential:

1. Direct A removed 8192 B of LDS but did not raise active blocks/CU.
2. It kept the 128x128 tile and expensive K128 IU4 fold, so it did not create radiance's 16-chain FP8 drain.
3. Its A loads remained on the load counter and were interleaved with the WMMA sequence rather than arriving as one 16-load scalar-address clause.
4. The new producer layout was byte-exact, so the loss is scheduling/dataflow cost rather than a correctness fallback.

The negative result does **not** refute producer-emitted A tiles in general. It refutes direct A without the numeric/dataflow properties that make the longer radiance drain possible.

## Trace reconciliation

The serving artifacts report:

| Trace | GEMM us/token | Logical GFLOP/token | TFLOP/s |
|---|---:|---:|---:|
| radiance | 220.088858 | 48.701112320 | **221.279318** |
| hipfire shipped FP8-v2 symfold | 340.487773 | 48.701112320 | **143.040000** |

The radiance trace contains 191 calls of `radiance_mxfp4_fp8_gemm_atiled<4,true,64>` totaling 641103.479 us/pass; the same trace also uses the narrower `<2,true,128>` and folded kernels for other shapes. The 221.279 TFLOP/s value is therefore the serving GEMM-group aggregate, not the isolated `<4,true,64>` speed alone.

A separate side-by-side trace table lists a different hipfire control at 238.260557 us/token and 204.402747 TFLOP/s. That control must not be substituted for the shipped FP8-v2 symmetric-fold result. The requested shipped value is the 143.040 TFLOP/s normalization in `FP8SYM_REPORT.json`.

## Reproduction

All created files are under this scratch directory. Source trees were not modified.

```text
export HOME=/home/kaden/.hipfire-homes/ab2
export ROCR_VISIBLE_DEVICES=GPU-085289909a86cc63
export HIPFIRE_KERNEL_CACHE=/home/kaden/.hipfire-homes/ab2/.hipfire_kernels

hipcc --offload-arch=gfx1201 -O3 bench_radiance.hip -o bench_radiance
./bench_radiance

hipcc --offload-arch=gfx1201 -O3 bench_symfold.hip -o bench_symfold_gate
hipcc --offload-arch=gfx1201 -O3 -DBENCH_RESIDUAL=1 \
  bench_symfold.hip -o bench_symfold_down
./bench_symfold_gate
./bench_symfold_down

hipcc --offload-arch=gfx1201 -O3 -DBENCH_BM=256 -DBENCH_BN=64 \
  bench_symfold.hip -o bench_symfold_gate_256x64
hipcc --offload-arch=gfx1201 -O3 -DBENCH_BM=256 -DBENCH_BN=64 \
  -DBENCH_RESIDUAL=1 bench_symfold.hip -o bench_symfold_down_256x64
./bench_symfold_gate_256x64
./bench_symfold_down_256x64
```

Primary evidence:

- `radiance_mxfp4_fp8.hip` and generated `radiance_mxfp4_fp8-hip-amdgcn-amd-amdhsa-gfx1201.s` in this directory;
- radiance kernel and launch sources under `ggz14/vllm-mxfp4/`;
- serving trace `rocm10-levers/kx/RADIANCE_TRACE/{RESULTS.txt,radiance_per_kernel.csv,groups_side_by_side.csv}`;
- shipped hipfire source `kernels/src/gemm_gate_up_mq4g256v2_wmma_fp8.gfx12.hip` and its generated gfx1201 assembly;
- shipped census `rocm10-levers/kx/FP8SYM_REPORT.json`;
- killed direct-A source, ISA, resources, occupancy, and paired daemon result under `ATiledProducers/`.
