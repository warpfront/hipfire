# gfx1201 iu4 prefill efficiency attribution

## Bottom line

The shipped pp8192 GEMM rate is **247.214 TOPS-equivalent / 197.000 us per token**, versus **695.085 TOPS / 70.065 us per token** for register-resident iu4 WMMA. The unexplained device-time gap is therefore **126.935 us/token**. Occupancy and DRAM bandwidth do not explain it.

The only defect unambiguously large enough to dominate that gap is the **K128 scale-fold dataflow**: it keeps both eight int32 partial accumulator vectors and eight fp32 running-sum vectors live, performs 192 logical VALU operations for every 32 WMMAs, and adds 20 logical metadata LDS reads. In the incremental hardware ladder, adding that dataflow costs **97.409 us/token**. The complete scratch ladder is 31.419 us/token slower than the exact production body, so an intentionally pessimistic reconciliation that assigns all of that difference to this one rung still leaves **65.990 us/token** attributable to the fold/dataflow bucket. The largest loss is thus **about 66--97 us/token**, not memory traffic or occupancy.

The next largest raw rung is the weight global-to-LDS pipeline and per-slab synchronization at **41.508 us/token**, followed by the output transpose/store at **12.497**, and activation staging/reads at **8.087**. Those are order-dependent upper bounds, not independently recoverable speedups. Only the first-place ranking is robust to the production reconciliation.

The production steady-state inner loop issues:

- **32** `wmma_i32_16x16x32_iu4` per wave per K128;
- **235 logical VALU instructions per K128**, or **7.34375 VALU/WMMA** (43 body + 192 fold);
- **0.750 operand LDS reads/WMMA** (0.25 W + 0.50 activation);
- **1.375 total logical LDS reads/WMMA** after including fold metadata.

The radiance comparison is 0.734 logical VALU/WMMA and 0.25 operand LDS reads/WMMA. Thus this kernel does **10.0x the non-WMMA VALU work** and **3x the operand LDS reads per matrix instruction**; including fold metadata, the LDS-read ratio is 5.5x. This is the hardware-level explanation for 35.5% of peak.

## Normalization

All rates use the campaign convention of two logical integer operations per MAC and the supplied dense-model denominator of 48.701112320 GOP/token:

```text
peak-equivalent time = 48.701112320 * 1000 / 695.085 = 70.064974 us/token
shipped rate          = 48.701112320 * 1000 / 197     = 247.213768 TOPS
measured gap          = 197 - 70.064974              = 126.935026 us/token
```

The register-only ladder below independently lands at 693.665 TOPS, within 0.2% of the 695.085-TOPS peak microbenchmark.

## Incremental ceiling ladder

### Construction

`scratch-iu4prof/iu4_ladder.hip` uses the actual two dominant projection shapes at B=8192:

- gate/up: grid 272 x 64, K128=40 (M=34816, K=5120);
- down: grid 40 x 64, K128=136 (M=5120, K=17408).

Every rung uses 256 threads and 20 KiB dynamic LDS. Each shape receives five warmups and 20 individually event-timed launches; the reported value is the mean of sorted samples 9 and 10. Gate/up and down are aggregated by their exact logical-operation counts, not averaged. The rungs are:

0. register-resident operands and WMMAs;
1. weight operand LDS reads (0.25 logical reads/WMMA), activation still in registers;
2. weight global/cache-to-LDS double-buffer staging and synchronization;
3. activation global/cache-to-LDS staging plus activation operand reads (another 0.50 reads/WMMA);
4. exact-shaped K128 metadata staging plus the symmetric scale fold and dual accumulator sets;
5. production-style LDS output transpose and coalesced global stores.

Build and run:

```bash
cd /home/kaden/ClaudeCode/warpfront/wt-iu4prof/scratch-iu4prof
hipcc --offload-arch=gfx1201 -O3 -Rpass-analysis=kernel-resource-usage \
  --save-temps iu4_ladder.hip -o iu4_ladder
HOME=/home/kaden/.hipfire-homes/ab0 \
ROCR_VISIBLE_DEVICES=GPU-9eb7aeda51c88ffd \
HIP_VISIBLE_DEVICES=GPU-9eb7aeda51c88ffd \
./iu4_ladder | tee ladder-card-a.txt
python3 ladder_analysis.py ladder-card-a.txt
```

Measured on card-A:

| Rung | Aggregate launch (us) | TOPS | Whole-model equiv. us/token | Raw marginal us/token |
|---|---:|---:|---:|---:|
| 0 register WMMA | 6,315.533 | 693.665 | 70.208 | -- |
| 1 + W operand LDS reads | 6,009.194 | 729.027 | 66.803 | -3.406 |
| 2 + W staging/sync | 9,743.011 | 449.642 | 108.311 | +41.508 |
| 3 + activation staging/reads | 10,470.498 | 418.401 | 116.398 | +8.087 |
| 4 + K128 fold/metadata/dual accumulators | 19,232.842 | 227.781 | 213.807 | **+97.409** |
| 5 + output transpose/store | 20,357.014 | 215.202 | 226.304 | +12.497 |

The apparent improvement in rung 1 is not a negative LDS cost. The inserted independent LDS instructions space the eight accumulator chains and the run can boost differently; it establishes that operand LDS latency alone is hidden at this occupancy. I assign it zero recoverable cost.

Compiler resources from the build command:

| Rung | VGPR | SGPR | compiler cap, waves/SIMD | spill bytes/lane | API active blocks/CU with 20 KiB LDS |
|---|---:|---:|---:|---:|---:|
| 0 | 69 | 12 | 16 | 0 | 3 |
| 1 | 76 | 12 | 16 | 0 | 3 |
| 2 | 86 | 12 | 16 | 0 | 3 |
| 3 | 108 | 22 | 12 | 0 | 3 |
| 4 | 180 | 30 | 8 | 0 | 3 |
| 5 | 183 | 32 | 8 | 0 | 3 |

Three 256-thread blocks are 24 waves/CU, or six waves/SIMD on four SIMDs. Even the fold rung's eight-wave/SIMD register cap is above the six waves supplied by the LDS-limited launch. The production kernel similarly reports 182 VGPR, 40 SGPR, no scratch/spills, and three active blocks/CU. The separate peak test sustaining 690.040 TOPS at eight blocks/CU is additional evidence that occupancy is not the missing 64.5%.

### Production bridge and bounds

The final scratch rung is deliberately readable and does not preserve every production scheduling decision. I therefore measured the exact shipped full-set body at both shapes:

```bash
cd /home/kaden/ClaudeCode/warpfront/wt-iu4prof/scratch-iu4prof
hipcc --offload-arch=gfx1201 -O3 -DBENCH_WP=0 \
  /home/kaden/ClaudeCode/warpfront/wt-iu4w/scratch-iu4w/bench_iu4.hip \
  -o bench_production
HOME=/home/kaden/.hipfire-homes/ab0 \
ROCR_VISIBLE_DEVICES=GPU-9eb7aeda51c88ffd \
HIP_VISIBLE_DEVICES=GPU-9eb7aeda51c88ffd ./bench_production
HOME=/home/kaden/.hipfire-homes/ab0 \
ROCR_VISIBLE_DEVICES=GPU-9eb7aeda51c88ffd \
HIP_VISIBLE_DEVICES=GPU-9eb7aeda51c88ffd ./bench_production down
```

```text
RESULT,baseline,gate_up,B=8192,M=34816,K=5120,median_us=12658.943,mean_us=12647.187,tflops=230.713,occ=3,lds=20480
RESULT,baseline,down,B=8192,M=5120,K=17408,median_us=4871.807,mean_us=4859.717,tflops=299.743,occ=3,lds=20480
```

The aggregate is 249.896 TOPS, equivalent to **194.885 us/token**. It is only 2.115 us/token faster than the exact daemon family total of 197 us/token. The arithmetic reconciliation is:

```text
scratch rung 5                         226.304 us/token
production overlap/address schedule   -31.419
exact production full-set body         194.885
full-add/context/run residual            +2.115
observed daemon family                 197.000
```

Consequently, the ladder determines ordering and scale, but its marginals must not be summed as prospective gains. A conservative sensitivity bound subtracts the entire 31.419-us production scheduling credit from one bucket at a time:

| Rank by raw marginal | Defect bucket | Raw measured marginal | Conservative range after reconciliation | Practical recoverability evidence |
|---:|---|---:|---:|---|
| 1 | K128 fold, metadata, dual accumulator sets | 97.409 | **65.990--97.409 us/token** | No quality-valid iu4 fold-free result yet |
| 2 | W staging plus per-slab synchronization | 41.508 | 10.089--41.508 | W preshuffle alone: +1.27% standalone, -0.32% daemon |
| 3 | Output LDS transpose/global store | 12.497 | 0--12.497 | Cannot be eliminated; only reduced by a larger useful wave tile |
| 4 | Activation staging plus operand reads | 8.087 | 0--8.087 | Already largely hidden; same 0.50 A reads/WMMA as the shipped layout |
| 5 | W operand LDS reads alone | approximately 0 | approximately 0 | Hidden in this experiment |

Ranks 2--4 overlap after reconciliation; rank 1 does not. The W-staging rung is an instruction/synchronization/cache-to-LDS cost, not DRAM bandwidth. The campaign weight-volume check is 14.99 GB / 8192 / 460 GB/s = 3.98 us/token, about 2% of the 197-us budget.

## Production ISA census

Rebuild the exact symmetric production translation unit and run the checked parser:

```bash
cd /home/kaden/ClaudeCode/warpfront/wt-iu4w/scratch-iu4w
hipcc --offload-arch=gfx1201 -O3 --save-temps -c compile_baseline.hip \
  -o compile_baseline.o
python3 /home/kaden/ClaudeCode/warpfront/wt-iu4prof/scratch-iu4prof/isa_census.py \
  compile_baseline-hip-amdgcn-amd-amdhsa-gfx1201.s
```

Output:

```text
steady WMMA/K128: 32
steady logical VALU/K128: 235 (body 43, fold 192)
steady logical VALU/WMMA: 7.34375
first-step-only logical VALU: 10
fold VALU classes: 64 cvt, 64 mul, 64 fmac/add
operand LDS reads/WMMA: 0.75000 (12 encoded dual-address b64 loads)
fold-metadata LDS reads/WMMA: 0.62500 (10 encoded dual-address b32 loads)
all logical LDS reads/WMMA: 1.37500
global loads/K128/lane: 10 (8 b64 payload, 2 b32 metadata)
```

A dual VALU encoding counts as two logical operations; a dual-address DS load counts as two logical reads. The 10 one-time VALU instructions initialize the first fold. Including that setup, the actual average is 7.35156 VALU/WMMA at K128=40 and 7.34605 at K128=136.

### Instruction mix and wait placement

Per steady K128/wave:

- four groups of eight WMMAs;
- each group has three encoded dual-address b64 operand loads, `dscnt 1` before its first two WMMAs, then `dscnt 0` before its final six: eight operand-DS waits per K128;
- eight b64 global payload loads are issued as four two-load clauses; two standalone b32 loads fetch the scale metadata;
- ten encoded dual-address b32 metadata reads feed the fold, with `dscnt` thresholds 4, 3, 2, 1, 0 staggered through the arithmetic;
- the fold itself is 64 int32-to-fp32 conversions, 64 scale multiplies, and 64 fused multiply-accumulates/adds;
- `loadcnt` waits are interleaved into the WMMA drains, but a combined load/DS wait starts the fold, a final load wait occurs near its end, and barrier signal/wait separates LDS-buffer ownership.

This explains why the problem is dataflow/issue pressure. Each K128 step spends six logical VALU instructions per WMMA on fold arithmetic alone, before the 1.34375 body VALU/WMMA, and the fold phase has a serial LDS-metadata dependency chain. The 64x32 wave tile reuses W at 0.25 LDS read/WMMA but reloads activation operands at 0.50. By comparison, radiance keeps 16 useful accumulator chains in a 64x64 tile and needs 0.25 operand LDS read/WMMA total.

## Hardware counters: explicitly unavailable on this gfx1201 stack

Profiler version:

```text
rocprofv3 1.3.5, ROCm 10.0.0, git 6b0e43f341195e203754e08f850e437ff2fc09f9
```

`rocprofv3 -L` exposes no WMMA/MFMA/matrix-utilization counter. It advertises derived VALU, LDS, occupancy, dependency-wait, and memory-unit metrics, so I collected them on the real daemon protocol, filtered to the two production iu4 kernels. The replay contains two pp8192 passes (16,384 prompt tokens total):

```bash
cd /home/kaden/ClaudeCode/warpfront/wt-iu4prof/scratch-iu4prof
rm -f /home/kaden/.hipfire-homes/ab0/.hipfire/daemon.pid
HOME=/home/kaden/.hipfire-homes/ab0 \
ROCR_VISIBLE_DEVICES=GPU-9eb7aeda51c88ffd \
HIP_VISIBLE_DEVICES=GPU-9eb7aeda51c88ffd \
HIPFIRE_KERNEL_CACHE=/home/kaden/.hipfire-homes/ab0/.hipfire_kernels \
HIPFIRE_MODELS_DIR=/home/kaden/.hipfire/models \
HIPFIRE_GRAPH=1 \
rocprofv3 --pmc SQ_BUSY_CYCLES GRBM_GUI_ACTIVE LDSBankConflict LdsUtil \
  MeanOccupancyPerActiveCU OccupancyPercent VALUBusy WAVE_DEP_WAIT MemUnitBusy \
  --kernel-include-regex 'gemm_mq4g256v2_residual_mmq_iu4_full_(set|add)_symfold' \
  -f csv -d prof-counters -o iu4 -- \
  /home/kaden/ClaudeCode/warpfront/wt-iu4w/target/release/daemon \
  < replay-iu4.jsonl > /dev/null
python3 counter_summary.py prof-counters/iu4_counter_collection.csv
```

Across 992 filtered dispatch records (736 full-set and 256 full-add), the only nonzero requested counters were `GRBM_GUI_ACTIVE` and `SQ_BUSY_CYCLES`:

```text
GRBM_GUI_ACTIVE: min=149061 median=7894386 max=13605247
SQ_BUSY_CYCLES:  min=2026054 median=251260712.5 max=434552741
LDSBankConflict: 0 for every record
LdsUtil: 0 for every record
MeanOccupancyPerActiveCU: 0 for every record
OccupancyPercent: 0 for every record
MemUnitBusy: 0 for every record
VALUBusy: 0 for every record
WAVE_DEP_WAIT: 0 for every record
```

Those zeroes are unavailable counters, not physical zeroes: the disassembly proves that these dispatches issue VALU and LDS instructions. A second collection using the raw dependencies (`SQC_LDS_BANK_CONFLICT`, `SQC_LDS_IDX_ACTIVE`, `SQ_WAVE_CYCLES`, `SQ_INST_CYCLES_VALU`, `SQ_WAIT_ANY`, `SQ_WAIT_INST_ANY`, `TA_TA_BUSY`, `SQ_INSTS_VALU`, `SQ_INSTS_LDS`, `SQ_WAVES`) aborts rocprofv3 on the first profiled dispatch with:

```text
Invalid Register used: hwip=37, ip_inst=48, base_idx=32.
Exception: array::at: __n (which is 37) >= _Nm (which is 37)
```

Therefore no bank-conflict percentage, VALU/WMMA utilization, stall split, or counter-derived occupancy claim is defensible on this stack. The report uses the executable ladder, compiler census, ISA, and HIP occupancy API instead.

## Target feasibility and work allocation

| Goal | GEMM budget | Required rate | Peak fraction | Saving from 197 us | Throughput uplift |
|---|---:|---:|---:|---:|---:|
| 4,000 tok/s | 165 us/token | 295.158 TOPS | 42.46% | 32 us | 19.39% |
| 5,000 tok/s | 115 us/token | 423.488 TOPS | 60.93% | 82 us | 71.30% |

### Approximately 165 us/token

This is structurally plausible, but **no currently measured, quality-cleared combination reaches it**. The primary fold/dataflow bucket alone is large enough: it must yield at least 32 us of its 66--97-us attributable range. Minor tuning cannot substitute:

- iu4 W preshuffle alone is worth at most about 2.5 us by its +1.27% standalone result and regresses in the daemon;
- transferring the fp8 fold-free+preshuffle uplift of +10.8% as an analogy would put 197 us at 177.8 us, still 12.8 us short;
- even subtracting the entire 12.5-us raw epilogue rung from that analogy only reaches 165.3 us with no margin, and the epilogue cannot actually disappear.

The viable research combination is therefore coupled, not additive: make the scale representation valid without a per-K128 fold, use the freed registers for 16 useful accumulator chains / a 64x64 wave tile, and preshuffle weights for that tile. The current pow2-half route is blocked by its measured quality gate, and there is not yet an iu4 fold-free throughput row. The first 4--5 agent-days should be spent on that numeric/dataflow contract and killed quickly if quality cannot pass; standalone layout or launch tuning is already receipted dead.

### Approximately 115 us/token

This is **not a credible near-term target from the measured fixes**. It requires recovering 82 us: essentially 84% of the raw 97.4-us primary rung, or the entire conservative 66-us lower bound plus at least 16 us from staging/epilogue. It demands all of the following simultaneously:

1. eliminate or radically amortize the K128 fold without violating the per-K128 activation-scale semantics;
2. convert the freed register budget into a 64x64/16-chain tile, reducing operand LDS traffic from 0.75 toward 0.25 reads/WMMA;
3. reduce weight staging/synchronization materially, despite standalone preshuffle showing no daemon gain;
4. retain a low-cost epilogue.

Nothing measured on iu4 demonstrates that combination, and the 221-TF fp8 comparison on the same card does not establish 61% of iu4 peak either. Treat 115 us as an architectural redesign threshold, not an optimization milestone.

## Decision

Prioritize one coupled investigation: **quality-valid fold removal/amortization plus the 64x64 dataflow it unlocks**. Its 66--97-us bucket is the only place with enough budget for 4,000 tok/s and the only conceivable path toward 5,000. Do not spend the next allocation on standalone W preshuffle, occupancy, DRAM tuning, or the already-receipted launch/LDS variants. If the numeric contract cannot eliminate the per-K128 fold, the evidence says 4,000 tok/s is not yet supported by the remaining measured levers; 5,000 is out of reach.
