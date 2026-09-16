# Halo IU4 low-footprint prefill GEMM: residency, not more output chains

Date: 2026-09-16. Planning-only specification for `/home/kaden/ClaudeCode/warpfront/wt-lloyd` (`mq4-lloyd`). No kernel/host changes, GPU execution, project build/test suite, formatter, linter, or commit were performed. A standalone **CPU-only gfx1151 compilation of shipping source** and CPU arithmetic/ownership checks were performed; their evidence is below. Local ISA is not the Halo shipping JIT object and is not a performance or numerical-parity result. Main owns hardware gates; the independent reviewer owns final veto.

## Ten-line summary

1. Select **M128 × N128 with 16 wave32s**, block `[32,16,1]`, four 16×16 subtiles/wave and `sum[32]`; retain A5 column-adjacent order.
2. Shipping's 190-register ISA ledger is **64 FP sums + 64 deferred integer results + 24 headers + 30 address/control + 8 fragments**; source `acc[8]` hides its actual lifetime cost.
3. Selected target is **≤96 allocated VGPR, zero scratch/spills, 30,720 B LDS, two WGs/CU = 32 waves/CU**, versus 16 waves today.
4. Fold each output subtile promptly through the W4.0-pinned DAG; retain one 8-register integer chain, not all four completed chains, and issue four A-row loads before their first wait.
5. Selected logical A/Xq traffic remains **189,399,040 / 200,540,160 B per N512 production launch**; no additional Xq rereads or change to the output tile grid.
6. M64×N128/eight waves can reach **three**, not four WGs/CU at ≤128 VGPR; its Xq reads double, and its optimistic wait-only gate gain is only 12%.
7. Conditional selected per-call model is **0.82–1.00× gate / 0.765–1.00× down baseline**; the ideal ends save 18%/23.5%, not measured results or guaranteed bounds against regression.
8. At 1,533/1,566 µs normalization only, those ranges are **1,257–1,533 / 1,198–1,566 µs**; these supplied family averages are not shape-specific timing baselines.
9. Admit only after metadata, exact bits/eval md5, a significant **SQ_WAIT_CNT_ANY decrease**, ≥15% per-call on both production shapes, and pp512/pp2048 model/bench transfer; two iterations, 16 active hours maximum.
10. Biggest risk: getting the complete pinned-fold kernel to **96 VGPR** without spills or crippling scheduling; more waves alone do not increase the selected tile's aggregate Xq-load window.

## 1. Evidence and scope

The latest STATUS blocks override the prospective WS4 body in `/home/kaden/ClaudeCode/warpfront/wt-lloyd/docs/plans/2026-09-15-halo-w2-iu4-redesign.md:5-35`:

- WS4 ended at 216 VGPR/zero spills, only −6.2% gate/−7.8% down, and 1-ulp fold drift; rejected. Earlier spilled 256-register variants are not candidates.
- W3 mc4: 188 VGPR, two WGs/CU, −2.7%/−5.1%; mc2: 151 VGPR but +14.5%/+8.9% time. More independent chains are not this design's mechanism.
- W5: only one of nine prefetched Xq dwords fit under the 192-register budget; ±1–3% noise, rejected. Supplied rocprof shipping wait fractions are gate **36%**, down **47%** of wave-cycles; barrier and LDS-conflict fractions approximately **2% each**. They are not independent fractions of kernel wall time.
- A5 is admitted. `/home/kaden/ClaudeCode/warpfront/wt-lloyd/docs/plans/2026-09-15-halo-gap-plan.md:154-163` records 1688→1534 /1643→1567 µs and pp2048 650 t/s on a quiet Halo. Keep A5; do not reuse pre-A5 timing denominators or its historical CPU-build-contaminated rejection.
- Current campaign profile supplied by Main: 3.11 s pp2048, IU4 79.4%, set 1533 µs ×1088, add 1566 ×512. Reaching 710 from 650 t/s needs **8.45% wall-time reduction**; if the 79.4% serialized-profile fraction were also the bench fraction, GEMM would need 10.64% reduction. That is only an Amdahl sensitivity calculation, not a throughput prediction.

**Non-goals:** change weight/Xq/quantization formats; introduce split-K, partial FP sums, alternate fold contraction, low-precision sums, extra LDS planes, producer-only waves, next-group prefetch, fewer barriers, new streams, chunk size/cadence, model state/KV/GDN/conv transactions, new graph/replay/retained eligibility, gfx1100 tuning, or a permanent flag/benchmark framework. Do not combine this experiment with W4 64×256 or resurrect WS4/W3/W5. TailA/B/C work is an independently fixed predecessor, never an additive speed credit. No edit to `/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/hipfire-arch-qwen35/src/qwen35/prefill.rs` is needed; HaloTailA remains its integrator.

## 2. Shipping ISA register ledger: measured offline, not guessed from C++

### 2.1 Reproducible artifact identity

The W2 STATUS names the local hipcc resource workflow, but its current document contains no full command; the full recipe is in W4.0 below. This plan executed its standalone equivalent in a **durable project workspace**, not on a GPU:

- Prelude: `/home/kaden/ClaudeCode/warpfront/wt-lloyd/kernels/src/block_i4_128_quant.hip`, SHA-256 `df39676b9a56d0f381a33c681c8b2ad5feba3f0ca2a2e2188505c59fc87c7ee0`.
- Kernel: `/home/kaden/ClaudeCode/warpfront/wt-lloyd/kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip`, SHA-256 `0ff99b708a5939f2e7dcdf3d1904134343e0a3755d9282c38e0acc935fd2c9ca`.
- Concatenate those sources, in that order, into `/home/kaden/ClaudeCode/warpfront/wt-lloyd/target/halo-iu4-lowfootprint/ledger-20260916/shipping.hip` (same composition as `/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/rdna-compute/src/kernels.rs:3403-3406`).
- Compiler: HIP `7.15.26333-0000000`, AMD clang `23.0.0git`, LLVM revision `8f497e0992fb7513f7f78a6f6b6f1056c375e961`, installed under `/opt/rocm/core-10.0/lib/llvm/bin`.

With working directory `/home/kaden/ClaudeCode/warpfront/wt-lloyd/target/halo-iu4-lowfootprint/ledger-20260916`, the exact compile command was:

```sh
/opt/rocm/core/bin/hipcc --genco --offload-arch=gfx1151 -O3 --no-offload-compress -save-temps -Rpass-analysis=kernel-resource-usage -o /home/kaden/ClaudeCode/warpfront/wt-lloyd/target/halo-iu4-lowfootprint/ledger-20260916/shipping-gfx1151.hsaco /home/kaden/ClaudeCode/warpfront/wt-lloyd/target/halo-iu4-lowfootprint/ledger-20260916/shipping.hip
```

No per-kernel/environment passthrough was added. Production's core options are `/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/rdna-compute/src/compiler.rs:995-1014`; Halo's actual compiler/options/object must still be archived and compared at W4.0.

**Every `ISA:Lx-Ly` citation below means** `/home/kaden/ClaudeCode/warpfront/wt-lloyd/target/halo-iu4-lowfootprint/ledger-20260916/shipping-hip-amdgcn-amd-amdhsa-gfx1151.s:x-y`, SHA-256 `d9ae3a9a618f539af9b9d045d108d32f18e5fa9fede2068fe8529f693b7e600d`. Object `/home/kaden/ClaudeCode/warpfront/wt-lloyd/target/halo-iu4-lowfootprint/ledger-20260916/shipping-gfx1151.hsaco` has SHA-256 `59d7cbe75094e3529fdeffc96d962c753dd4af7f6104e3d097909098a8caf90c`. Preserve these ignored build artifacts as evidence or archive them before cleaning target; the table and exact regeneration recipe remain in this document.

### 2.2 The complete 190-register partition

Representative entry: `gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3_col_gfx1151`, `ISA:L15850-L18169`. The following is a disjoint physical-register-role partition at the **last h0 integer subtile / start of the deferred fold**, not an assertion that a physical register has one role throughout the kernel. Temporaries reuse dead accumulator/fragment registers during the fold and fill.

| Role | Count | Physical registers at this phase | ISA evidence |
|---|---:|---|---|
| FP32 running sums | **64** | `v66`, `v96:v144`, `v146:v157`, `v161:v162` | Zero initialization `L15880-L15912`, loop-path initialization `L15961-L15988`; independent final additions throughout `L16378-L16721`, e.g. `v118`, `v162`, `v66`, `v96` |
| Integer WMMA results, including seven completed chains | **64**, not 8 | `v0:v63`; chain order `0:7`, `8:15`, `16:23`, `24:31`, `40:47`, `48:55`, `56:63`, `32:39` | Successive WMMA loops `L16142-L16337`; first floating-point fold begins only at `L16340`, after all eight integer chains |
| A packed half2 headers | **16** | `v72:v85`, `v88:v89` | Eight two-dword LDS loads `L16133-L16140`, reused for all four column subtiles |
| Xq d/s metadata | **8** | `v86:v87`, `v90:v95` | `L16141`, `L16182-L16184`, `L16235-L16239`, `L16288-L16290` |
| Address/index/control state | **30** | `v64:v65`, `v67:v71`, `v145`, `v158:v160`, `v163:v181` | Header/payload bases, lane/wave/index values `L15919-L15960`; LDS/Xq offsets `L16078-L16125`; cached column offsets `L16182-L16290` |
| Current A/B fragments (and later fold scratch) | **8** | `v182:v189` | Last chain's two b128-equivalent LDS loads and WMMAs `L16327-L16336`; these registers become scale/fold scratch at `L16340-L16382` |
| **Total used** | **190** | Exactly `v0:v189`, no omitted mystery bucket | Descriptor `L18192`; compiler metadata `L18230-L18243` |

The address/control count includes seven base/address words (`v64:v65,v67:v71`), five lane/wave/epilogue indices (`v145,v158:v160,v163`), and eighteen cached LDS/copy addresses (`v164:v181`). It is **not** thirty live 64-bit pointers. Header conversions do not require a second permanent temp bank: packed header values feed `v_fma_mix_f32`; conversions and scale scratch recycle `v0:v63,v182:v189`. At global fill, those same scratch slots carry payloads/addresses instead of live integer results.

Consequences:

1. Cutting only `sum[64]` to `sum[32]` is not a reliable `190−32` prediction. Halving subtiles also cuts the compiler's deferred integer bank **64→32**, and a 16-row wave needs only eight A header words instead of sixteen. A like-for-like deferred-fold screen is **190−32−32−8 = 118 used**, likely rounded to120: plausible for the M64×N128 three-WG tile, **not enough for selected ≤96**.
2. The selected design must also shorten completed-integer-result lifetimes: `32→8` saves another24, giving a **94-register sizing anchor** before changed address allocation. This is a compiler-risk estimate, not candidate metadata. Source braces alone did not stop shipping's fold sinking.
3. Do not add “eight acc registers” and then call the other56 unexplained compiler overhead. Those56 have identifiable unconsumed integer outputs in the actual ISA.

Local full entries, including A5 add/set, all report **190 VGPR, 21 SGPR, zero scratch/spills, eight waves/SIMD**. Add-column metadata: `ISA:L15787-L15843,L18620-L18629`; set-column: `L18173-L18243,L18659-L18677`. Wave32 is explicit. `VGPRBlocks:23` denotes **24×8=192 allocated VGPR**, not190. `group_segment_fixed_size=0`/`LDSByteSize:0` means dynamically supplied LDS, **not zero actual LDS**; host adds30,720 B. The local tail/base entry is195 used (`L3800-L3811`); do not silently apply full-entry occupancy claims to tails.

### 2.3 What the ISA says about memory and floating-point arithmetic

- A payload loop issues **two wave-wide `global_load_b32`** before waiting (`ISA:L16024-L16054`), repeats eight times/wave for sixteen rows. Header load is separate (`L16062-L16073`). There is substantial room to issue more A rows before the first wait once state is smaller.
- Xq already issues **nine** dword loads before its first VM wait (`L16096-L16117`, repeated h1 `L16735-L16755`). Freed registers do not automatically improve that existing window.
- Fragment reads `L16147-L16153`/`L16330-L16336` use unsigned A/signed B and one dependent integer chain; no integer reassociation is needed.
- One concrete h0 fold: `L16340-L16352` computes rounded scale×d into `v183`, converts C in`v50` and s in`v95`, multiplies scaled C, recomputes zp×d, and FMA-adds its s term into the product; `L16389` separately adds that term to running sum`v116`. Packed-half conversion/multiplication uses `v_fma_mix_f32` with `neg(0)`. The schematic DAG is `p=RN(RN(sc*d)*float(C)); term=FMA(RN(zp*d),float(s),p); sum=RN(sum+term)`, **subject to W4.0 confirming conversion, signed-zero, mixed instruction and denormal semantics on actual objects**. Do not assume the FMA takes `sum` as its addend.
- Local descriptor round modes0/denorm modes3 are at`L18195-L18198`. Shipping allows final register-only folds to cross a barrier (`L16717-L16721`); only LDS reads need retire before overwrite. The candidate may finish folds earlier, but cannot change each output's arithmetic DAG/order.

## 3. Candidate arithmetic and residency

Use the requested Halo model: **two SIMD/CU,1536 lane-register units/SIMD,16 wave32 slots/SIMD,64 KiB LDS/CU**, with resource/placement confirmation on the actual device. For even wave count W and allocated R:

`WGcap = min(floor(65536/LDS), floor(min(16,floor(1536/R))/(W/2)), hardware WG/thread caps)`.

Balanced placement is a capacity calculation, not observed residency. Check actual WGP/CU allocation policy and launch limits with the hardware owner; never infer residency from `occ3` or compiler occupancy alone.

**Important correction to the proposed examples:** eight-wave **three**-WG residency requires `R≤1536/12=128`; **four** requires `R≤1536/16=96`, not128. Sixteen-wave two-WG residency also requires≤96. At128 a sixteen-wave block permits only **one** WG/CU. Four128-register/eight-wave WGs would require2048 regs/SIMD and are impossible in this model.

All dimensions are M rows×N batch columns; `G=K/256`. A group=136 global bytes/row; Xq=72 bytes/column/K128. Formulas reused from W4:

`LDS=4*(42*Tm+18*Tn)`; `sum/lane=Tm*Tn/(32*W)`; `A_launch=M*G*136*(N/Tn)`; `Xq_launch=(K/128)*N*72*(M/Tm)`; `A+Xq per WG/group=136*Tm+144*Tn`.

| Candidate | Waves / sums per lane | LDS A+Xq = total B | Allocated VGPR target | WG/CU / waves/CU | A/Xq vs shipping | Decision |
|---|---|---|---:|---|---|---|
| Shipping128×128 |8 /64|21,504+9,216=30,720|192|2 /16|1× /1×|Reference, A5 retained|
| **Selected128×128** |**16 /32**|**21,504+9,216=30,720**|**≤96**|**2 /32**|**1× /1×**|No traffic tax; must retire integer results promptly|
|64×128 |8 /32|10,752+9,216=19,968|≤128|3 /24|1× /2×|59,904 B for3;4 needs79,872 B, impossible even at96 registers|
|64×64 |8 /16|10,752+4,608=15,360|≤96|4 /32|2× /2×|Genuine four-WG option, but doubles both input streams; not selected|
|128×64 |8 /32|21,504+4,608=26,112|≤128|2 /16|2× /1×|LDS prevents3; no extra waves, worse A traffic|
|32×128 |4 /32|5,376+9,216=14,592|≤192|4 /16|1× /4×|Four WGs is not more waves; reject Xq tax|

For selected, if allocated R is104–192, only **one16-wave WG** fits (same16 resident waves as today's two8-wave WGs), and the stated mechanism fails. For64×128, R136–192 gives only2 WGs/16 waves, also mechanism failure. No spilled capped allocation is accepted.

### 3.1 Traffic and launch counts, with A5 column adjacency

Production shapes are gate-set `(M,K,N)=(17408,5120,512)` and down-add `(5120,17408,512)`. Both have47,349,760 unique A bytes and91,268,055,040 dense integer ops. Unique Xq is1,474,560 /5,013,504 B respectively. Per-launch logical requested bytes below are identical for both shapes because M*K matches. They are **not DRAM measurements**.

| Tile | Gate column-adjacent grid / WGs | Down grid / WGs | A B/launch | Xq B/launch | Total B/launch | B/WG/group |
|---|---|---|---:|---:|---:|---:|
|128×128 (8 or16 waves)|`[4,136,1]` /544|`[4,40,1]` /160|189,399,040|200,540,160|389,939,200|35,840|
|64×128 /8 waves|`[4,272,1]` /1088|`[4,80,1]` /320|189,399,040|401,080,320|590,479,360|27,136|
|64×64 /8 waves|`[8,272,1]` /2176|`[8,80,1]` /640|378,798,080|401,080,320|779,878,400|17,920|
|128×64 /8 waves|`[8,136,1]` /1088|`[8,40,1]` /320|378,798,080|200,540,160|579,338,240|26,624|
|32×128 /4 waves|`[4,544,1]` /2176|`[4,160,1]` /640|189,399,040|802,160,640|991,559,680|22,784|

Selected retains four adjacent N tiles for each M tile; `col_tile=blockIdx.x`, `row_tile=blockIdx.y`. This preserves A5's walk, not guaranteed CU/cache placement. Gate output writes35,651,584 B, down writes10,485,760 B plus an equal residual read; selected does not change either. At literal N2048 every table launch-byte total and grid x dimension multiply by4: selected A756,596,160/Xq802,160,640 B. **Do not confuse a pp2048 prompt with a single N2048 launch:** current chunked in-model call counts must be observed and remain unchanged.

64×128 increases combined requested input51.43%;64×64 doubles it. A5's improved cache reuse makes extra A rereads particularly unattractive. L2-resident Xq is still costly to request/write into LDS; locality, transactions and memory service must be measured, not inferred by dividing all logical bytes by DRAM bandwidth.

### 3.2 Loads in flight: capacity and ISA issue windows, not promises

Define a “wave-load slot” as one wave-wide global dword instruction issued before the first wait in a fill packet. A full32-lane slot represents128 requested bytes; masks reduce this. Aggregate figures below assume resident WGs reach this phase together and queues permit it. They are **upper capacity screens**, not hardware counter results; cache merging can reduce actual transactions.

| Tile | Resident waves | A packet2 slots/CU (requested payload B) | Xq half fill slots/CU | Implication |
|---|---:|---:|---:|---|
|Shipping|16|32 (4096)|16×9=144 (18,432 B)|Observed ISA windows2 A /9 Xq per wave|
|64×128 /8 waves|24|48 (6144)|24×9=216 (27,648 B)|1.5× slots but twice Xq requests/launch|
|Selected128×128 /16 waves|32|64 (8192); **packet4 target128 (16,384)**|16×5+16×4=144 (18,432 B)|A window2× from waves or4× with packet4; Xq aggregate window **unchanged**|
|64×64 /8 waves|32|64 (8192)|16×5+16×4=144 (18,432 B)|Twofold wave residency, but twofold A/Xq bytes and unchanged Xq window|
|128×64 /8 waves|16|32 (4096)|8×5+8×4=72 (9216 B)|No residency benefit; Xq window halves|
|32×128 /4 waves|16|32 (4096)|16×18=288 (36,864 B), if compiler batches all18|Xq window2× but launch traffic4×; no extra waves|

Selected initial implementation deliberately changes **only the current A-fill packet** to four coalesced row-dword loads before their stores/first VM wait. Eight rows/wave become two packets; no pending next-group data survives into compute. Xq assigns four or five words/thread in one half, not nine per new wave. All higher window claims require emitted-ISA confirmation and rocprof; more slots do not imply more off-chip bandwidth or fourfold speed.

### 3.3 Conditional exposed-latency projection

Let `w` be supplied36%/47% wait fraction, `r` the resident-wave ratio, `q∈[0,1]` the fraction of that wait actually eligible for inverse-r concurrency hiding, and `p≥0` new exposed cost. A deliberately optimistic screening model is:

`Tnew/Told = 1 - q*w*(1-1/r) + p`.

This equates eligible wave-cycle stalls with exposed wall-time only as a hypothesis. `SQ_WAIT_CNT_ANY` includes non-global waits; overlapping stalls, more load issue, queue saturation, bandwidth and phase synchronization invalidate the simple model. **No latency projection is measured.** Ranges below set p=0; real regressions lie outside them.

| Tile / residency | r | Gate ratio / ideal time reduction | Down ratio / ideal time reduction | Screening decision |
|---|---:|---|---|---|
|64×128 /3 WG|1.5|0.880–1.000 /12.0%|0.8433–1.000 /15.67%|Even ideal gate misses15%; new Xq cost unpriced|
|**128×128 /16 waves /2 WG**|**2**|**0.820–1.000 /18.0%**|**0.765–1.000 /23.5%**|Only sum32 option here with unchanged traffic and a plausible15% gate margin|
|64×64 /4 WG|2|0.820–1.000 /18.0%|0.765–1.000 /23.5%|Same optimistic stall-only ceiling before2× input tax; reject for this unit|
|128×64 or32×128 at listed residency|1|1.000 before overhead /0%|1.000 before overhead /0%|No residency-based latency saving|

Selected needs `q≥0.15/0.18=0.8333` gate and`q≥0.15/0.235=0.6383` down even with p=0. With q=1, new overhead may consume only3.0/8.5 percentage points before the15% gate fails. Do **not** apply an extra4× latency credit for the A packet or add old W3/A5 savings. Xq's unchanged aggregate load window makes the down projection especially uncertain.

If the ideal fraction model held, normalized wait fractions would become `(.36/2)/(.64+.36/2)=21.95%` and`(.47/2)/(.53+.47/2)=30.72%`. For r1.5 they would be27.27%/37.15%. These are diagnostic predictions, not admission thresholds. For timing, use each shape's contemporary paired baseline T0. Supplied1533/1566 µs are **family aggregates**: multiplying gives1257–1533 /1198–1566 µs only as scale examples;15% gates at those normalizations would be1303.05/1331.10 µs. A direct hot-loop baseline must never be compared with an in-model denominator.

## 4. Frozen selected implementation contract

### 4.1 Symbols, architecture and per-lane ownership

After W4.0 admission, K exclusively owns `/home/kaden/ClaudeCode/warpfront/wt-lloyd/kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip`. Keep shipping macros/helpers/entries unchanged except the independently admitted pin. Add private `IU4_LF_` constants and helpers `load_iu4_tile_lf16_gfx1151`, `vec_dot_i4_x128_lf16_gfx1151`, `gemm_iu4_body_lf16_gfx1151`, under **exact `#if defined(__gfx1151__)`**, plus:

- `gemm_mq4g256v2_residual_mmq_iu4_full_set_lf16_col_gfx1151`
- `gemm_mq4g256v2_residual_mmq_iu4_full_add_lf16_col_gfx1151`

No non-Halo no-op exports. Seven arguments remain `(const char* A,const block_i4_128* Xq,float* Y,int M,int K,int N,int add)`; offsets **0,8,16,24,28,32,36**,40 user bytes. Same module `gemm_mq4g256v2_residual_mmq_iu4`, same existing concatenated source. Block`[32,16,1]`, grid`[N/128,M/128,1]`, dynamic LDS30,720 B. Use`__launch_bounds__(512,2)` as a compiler hint, **not** evidence of96 VGPR/two resident WGs; no forced register cap that spills.

For `w=threadIdx.y∈[0,15]`, `lane∈[0,31]`, `c∈[0,3]`, `l∈[0,7]`:

```
row0=128*blockIdx.y; col0=128*blockIdx.x;
wave_row=16*(w/2); wave_col=64*(w%2);
i=wave_row+2*l+lane/16;
j=wave_col+16*c+lane%16;
sum_index=8*c+l;
Y_index=(col0+j)*M+(row0+i);
```

Each wave owns16×64; four subtiles/eight outputs per lane/subtile give32 running sums. CPU enumeration covered all16,384 outputs exactly once. No split output, cross-wave reduction or atomic is permitted. The costed64×128 alternative uses the identical mapping with w0..7 and row0 stride64; it is **not a second implementation unit or fallback experiment**.

Per half and c, start one zero `int32x8_t`, execute t0..3 with two WMMAs in original low/high order at each t. A address is`tile_x+(wave_row+lane%16)*42+16*h+4*t`; B is`tile_y+(wave_col+16*c+lane%16)*18+2+4*t`. Flags remain`false,A,true,B,acc,false`. Complete eight integer steps then fold c through the exact pin into its own sums before the next c lifetime. Every output sees kb increasing,h0 thenh1; integer bound15,360 does not authorize reassociation. Retain final residual add once after all groups, never per half.

**Register schedule contract:** unroll c/l for static sum indexing; retain the ordered t loop. One current chain, current fragments, one output-column d/s pair, at most eight A half2 header words; no four-chain bundle or persistent staged next-half payload. Target sizing ledger:32 sums+8 C+8 fragments+10 packed-header/d/s words+26 address/index words+4 fold temps=88, leaving8 allocation/headroom slots to96. This is an optimization budget, not a second claim about generated registers. Recompute cheap LDS addresses locally instead of hoisting all column/header addresses across the group. A narrowly placed `__builtin_amdgcn_sched_barrier(0)` between completed folds may constrain instruction scheduling; it is **not** a synchronization primitive, and its presence does not prove SelectionDAG/live-range behavior. Inspect actual ISA and spend the second iteration only on a diagnosed violation. No noinline ABI, volatile/scratch workaround, inline assembly fold with unverified semantics, or compiler-option change.

### 4.2 LDS writers and exact phase transitions

LDS remains Xq `[0,9216)` (128×18 dwords) then A `[9216,30720)` (128×42). A contains both K128 halves; only one Xq half is live. Pads40/41 remain unread/uninitialized.

- A row payload: `i=i0+w`, i0=0,16,...112; lane copies `gp+8+4*lane` into`tile_x[42*i+lane]`, where`gp=A+136*((row0+i)*(K/256)+kb)` with64-bit addressing. Group four successive i0 rows into one bounded packet; each load still covers a full128-B row across a wave. 4096 payload words/WG,8/lane, unchanged global16,384 B.
- Header mapping is preserved: `i=16*w+lane/2`, `ksc=lane&1`, guard`i<128`; waves0..7 load, waves8..15 skip uniformly. Each of256 global header words has one writer, replicated into four slots`32+4*ksc+[0,3]`; all16 waves execute every barrier. Same half conversion path.
- Xq half: tid=`32*w+lane`; copy all `l=tid+512*q<2304` from`(const int*)(Xq+((2*kb+h)*N+col0))`. First256 threads own5 words, rest4; every dword one writer. No out-of-range load on the last loop iteration.
- A/B b64 starts use strides42/18; both visit all16 even start banks over lane%16. New wave bases are multiples of16 rows/64 columns and preserve bank phase. This CPU bank model is not proof of hardware LDS throughput/conflict behavior.

A/Xq are immutable caller/scratch-owned global inputs, Y is exclusively tiled output, LDS is WG-owned, sums lane-owned. All512 threads execute four barriers/group.

| Transition | Before | Action | After invariant |
|---|---|---|---|
|Entry→fill group0|Y unchanged; sums+0; LDS invalid|Fill disjoint A/Xq0 destinations|No consumer reads until publication|
|B0 publish|A(kb)/Xq(2kb) writes complete|`__syncthreads()`|All consumed cells visible; no writer touches either plane during half0|
|Half0→B1 release|Sums include exactly earlier groups|Eight ordered WMMAs/output and pinned fold h0; barrier|All Xq0/A reads retired; A remains immutable; Xq writable|
|Fill half1→B2 publish|Xq plane released; A(kb) retained|Write Xq(2kb+1), then barrier|Full half1 visible; no writer during consumption|
|Half1→B3 release|Each output includes current h0|Eight ordered WMMAs/output and pinned fold h1; barrier|All group reads complete; sums include groups0..kb|
|B3→next group/epilogue|No live LDS readers|Advance kb or store final sums|No next-group speculative read; all add destinations read once|
|Return|One owner/output|Set or one final add|Only output rectangle mutated; inputs unchanged|

K>0 and K%256==0 imply both halves exist. Do not remove barriers just because the compiler can finish a register-only fold on either side of one. Kernel failure propagates current errors; there is no rollback promise for partial Y. No persistent model state commits occur here.

## 5. Bit-exact prerequisite: W4.0, verbatim

The following section is copied **verbatim** from `/home/kaden/ClaudeCode/warpfront/wt-lloyd/docs/plans/2026-09-15-halo-w4-iu4-tile64x256.md:192-213`. Its W4/§7 references retain their original meaning; for this low-footprint campaign also apply the fresh oracle matrix in §7 below. Historical source line numbers are not current edit anchors. The copied recipe is not an instruction to establish working state in `/tmp`; use the durable workspace from §2 for this campaign.

## 5. W4.0: independently land the bit-exact fold prerequisite

**This is the first landing unit, before any tile, host, schedule or compiler-option change.** The inspected shipping source currently has an ordinary expression at kernel lines208-209, not a named fold macro. Introduce a single `IU4_FOLD_RN(sum_lvalue, sc, zp, d, acc_i32, s_i32)` contract and use it at that site; later W4 calls the identical pinned contract. No copy of the quantizer, no tolerance, no source-level algebra guess.

Owner F must:

1. Archive today’s shipping source, actual JIT/code-object identity, compiler/ROCm/options and existing eval output for **each** gfx1151/gfx1100 device, before modifying the fold. Reference artifacts must remain available after recompilation; a newly compiled “old source” is not automatically the previously shipping object.
2. Read disassembly of the current shipping full set/add `_occ3` entries, and the base/other full entries affected by the common helper. Trace conversion, multiplication, contraction and final running-sum addition for h0 and h1, including a nonzero running sum. Record the exact floating-point DAG, instruction modifiers, denormal mode and compiler flags. Do not infer it from `sc*d*float(acc)+zp*d*float(s)`.
3. Reproduce precisely that DAG using explicit `__fmul_rn`, `__fmaf_rn`, and `__fadd_rn` where those operations actually occur. Operand association and which add participates in an FMA matter. Explicit RN does not by itself specify FTZ/denormal mode: retain the shipping mode. Do not introduce a global `-ffp-contract=off` or fast-math change.
4. Inspect both architectures independently. If their shipping DAGs differ, make the smallest explicit compile-time architecture policy inside the fold contract; do not normalize one architecture to the other. If full/base/entry instances need distinct arithmetic policies, bind the verified policies explicitly to those existing paths. If the shipping behavior cannot be faithfully represented and demonstrated within W4.0, **stop W4**, rather than redefining “bit-exact.” Other gfx11 compilation targets retain their existing arithmetic; W4 adds no host route for them.
5. Reinspect emitted pinned ISA and compare full output bytes for today-vs-pinned shipping on the §7 fixtures, both modes/architectures. Parent runs the unchanged-reference `eval_hipfire` one-chunk md5 gate on gfx1151 **and** gfx1100. All mismatches block landing. The eval file contains sequence KLD aggregates, so md5 is an additional model gate, not a substitute for direct output-bit checks.
6. Land only this fold change and its measured evidence. No material per-call regression beyond paired noise is acceptable in this prerequisite. This admitted pin survives even if W4 tiling is later rejected.

Local offline ISA recipe for the **future owner**, not executed by the planner (run in a unique temporary directory; all source paths are absolute):

```sh
cat /home/kaden/ClaudeCode/warpfront/wt-lloyd/kernels/src/block_i4_128_quant.hip /home/kaden/ClaudeCode/warpfront/wt-lloyd/kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip > /tmp/halo-w4-fold.hip
hipcc --genco --offload-arch=gfx1151 -O3 --no-offload-compress -save-temps -Rpass-analysis=kernel-resource-usage -o /tmp/halo-w4-fold-gfx1151.hsaco /tmp/halo-w4-fold.hip
hipcc --genco --offload-arch=gfx1100 -O3 --no-offload-compress -save-temps -Rpass-analysis=kernel-resource-usage -o /tmp/halo-w4-fold-gfx1100.hsaco /tmp/halo-w4-fold.hip
```

Use the installed ROCm disassembler to read the saved objects and compare with the **actual shipping JIT object**, not just this recipe’s assembly. The production core flags are in `/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/rdna-compute/src/compiler.rs:995-1010`; archive passthrough flags as well. The recipe is a way to obtain evidence, not a claim that the current ISA has already been read. The measured DAG and per-device hashes are required W4.0 outputs, not placeholders to be guessed by the tile owner.


### 5.1 Current-source/A5 addition to the unchanged prerequisite

W4.0 also must cover **both A5 `_occ3_col_gfx1151` entries on gfx1151 and gfx1100**, present at `/home/kaden/ClaudeCode/warpfront/wt-lloyd/kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip:305-307`. Current common fold expression is`:208-213`, not the older line numbers in the copied text. The local ledger/DAG in §2 assists the owner but is not an archived Halo or gfx1100 JIT reference. Archive affected base/full/occ3/column entry behaviors before pinning; select arithmetic policy by existing path/architecture only if verified differences demand it. **W4.0 pins arithmetic, not a new schedule**; independently admit that unit before changing tile/waves/load packet. Fresh post-pin baseline metadata is the candidate's denominator.

## 6. Exact host integration and unchanged architecture/state routes

H exclusively owns `/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/rdna-compute/src/gemm.rs`, at `Gpu::gemm_mq4g256v2_mmq_prequant_iu4:19039-19146`. No exported API change. Re-run references against this worktree before editing; the available rust-analyzer returned no references for the visible wrapper and the mismatch was reported, so the current planning inventory was recovered from explicit source search.

- Keep architecture and K rejection at`:19049-19062`, binding/module/ABI/params/blob construction and timer/error handling.
- Preserve `full` and A5 `use_col` at`:19065-19072`. Candidate predicate: **exact gfx1151, positive M/K/N, existing K validity, full M/N128 alignment, and `use_col`**. Thus candidate is eager/full only; no new capture/recording eligibility. `force_blob` alone is not a fallback predicate.
- Select **symbol+block+grid+LDS together**. Candidate uses the two lf16 entries, `[32,16,1]`, `[N/128,M/128,1]`,30,720 B. All other cases retain the entire current branch at`:19073-19083,19119-19142`, including eager gfx1100's **A5 column entries**, despite their gfx1151 suffix. Fix the existing inaccurate comment if editing its scope, not its behavior.
- Tails/zero cases retain current behavior and eight-wave shipping grid; no new zero-size policy. gfx1100 never sees lf16. Other architectures remain rejected; build guards cannot leak unsupported symbols into routing. No new source module or registration, scratch reservation, producer or quantizer invocation.
- `gemm_mq4g256v2_mmq_set_prequant_iu4:19149-19158` and `...add...:19161-19170` forward unchanged. Every eligible caller inherits selection here; do not add per-family routing.

Current call sites, all in `/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/rdna-compute/src/gemm.rs`:

| Owner symbol | Set/add calls |
|---|---|
|`gemm_mq4g256v2_small_tail_set_iu4`|28495 fallback; preserve its existing small-M path|
|`gemm_qkvza_mq4g256v2_wmma` / `..._iu4_prepared`|28535-28551 /28706-28722|
|`gemm_qkv_mq4g256v2_wmma` / `..._iu4_prepared`|29123-29125 /29247-29249|
|`gemm_gate_up_mq4g256v2_wmma` / `..._iu4_prepared`|30254-30257 /30404-30407|
|`gemm_mq4g256v2_residual_wmma` / `..._iu4_prepared`|31797 /31948|

Prepared producer call chain in `/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/hipfire-arch-qwen35/src/qwen35/prefill.rs`: qkvza4591, LA gate/up5590, LA down5855, FA qkv6042, FA gate/up6895, FA down7150 in the inspected snapshot. Sibling edits can move these lines; symbols, not stale line numbers, own the contract. This plan requires **no prefill.rs changes**.

`Gpu::ensure_int4_mmq_x` at `/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/rdna-compute/src/dispatch.rs:2863-2886` delegates to scratch. `/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/rdna-compute/src/scratch.rs:1331-1425` always quantizes and invalidates prior prepared generation; `reserve_int4_mmq:1428-1444` instead reserves for an existing producer. Consumer accepts valid same-stream sidecars without reallocation, quantization or freshness inference from pointer identity. Before launch inputs are published on that stream; during execution they remain live/immutable; after completion existing owners may reuse them. Prepared-generation rejection stays fail-closed.

`Gpu::launch_maybe_blob`→`launch_maybe_blob_bound` at `/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/rdna-compute/src/dispatch.rs:2314-2331,2368-2492` remains the only launch path. Params and forced blob launch the same eager candidate. Recording/capture deliberately retain shipping symbol, `[32,8,1]`, row-major grid and30,720 LDS as today, not the new block. Tape owns recorded launch metadata; graph owns blob bytes through replay. Warm selected/fallback modules before capture, preserve allocation lifetime and add-start restoration, and do not patch new dimensions into old tapes. Test capture/recording through actual Gpu lifecycle APIs, not raw HIP capture while `graphs.capture_mode` remains false. `replay_recorded_hip_prefix:2497` remains diagnostic-only. Existing retained routes stay isolated; no candidate replay speed claim.

## 7. Fresh oracle and claim-scoped gate order

Main/hardware owner alone runs Halo gates on hipx, `ROCR_VISIBLE_DEVICES=1`, confirmed exact gfx1151. No GPU run is authorized for the planning/composer workers. Build/JIT **before** measurements; no cargo/CK/CPU build anywhere on that APU during sampling. Archive source/prelude/options/compiler/object IDs, model/reference/prompt IDs, route flags including unset values, clocks/power/temperature, raw counters/times and run order. gfx1100 owner handles its separate W4.0/isolation gate.

### G0 — complete object, no execution until it fits

For both final lf16 entries, require **allocated VGPR≤96 (eight-register rounding applied), zero VGPR/SGPR spills, zero scratch/private backing/dynamic stack, wave32,512 threads, exact30,720 dynamic and zero unexpected static LDS**. Archive SGPR usage,40-byte ABI offsets, both code objects and complete resource remarks. Expect compiler register occupancy16 waves/SIMD, then derive two WGs/CU from real LDS/thread/WG limits; missing resource fields are unknown, not zero. Check complete emitted body:32 independent sums, original eight WMMA operations/output/half, one live C chain at the planned boundary, pinned fold DAG, bounded current A packet4, four/five Xq words/thread, and all barriers. Dead-math/microkernel metadata cannot pass. Any allocated104 result fails the mechanism gate even with no spills. Recheck actual shipping JIT metadata after W4.0; do not assume the local190 result is its identity.

### G1 — fresh `tmp_halo_iu4_oracle`, every output bit and state boundary

O creates `/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/hipfire-runtime/examples/tmp_halo_iu4_oracle.rs` **fresh**; it is absent now. H owns any necessary example registration in `/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/hipfire-runtime/Cargo.toml`. Reuse upload/event/blob idioms from `pack_mq4g256v2`, `pack_i4_xq`, `launch_twin` in `/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/hipfire-runtime/examples/tmp_halo_iu4_calibrate.rs:772-827,991-1032`, not its narrow fixtures or add=0 launcher. No copied handwritten production twin or alternate quantizer may become the reference.

Frozen phases: `fold` (archived pre-pin vs pinned shipping), `bits` (pinned shipping vs explicit lf16), `dispatch` (real wrappers/sidecars/params/blob/capture/recording fallback), `timing` (opt-in only after bits/resources). Default is correctness-only. Raw reference uses actual A5 column entries/grid and eight-wave block for eager full shapes; raw candidate uses same grid, sixteen-wave block, same LDS. Use unique reference/candidate module/function identities to avoid cache collisions; public `Gpu::ensure_kernel_public` or archived-object loader follows established patterns. Restore add Y outside **each** timed sample.

Required observable coverage:

1. Gate `(17408,5120,512)` and down `(5120,17408,512)`, **both modes for bits**, gate-set/down-add for performance. Compare every`f32::to_bits()`, not an epsilon or digest-only comparison; guard A/Xq/Y and verify immutable input payloads.
2. Full boundaries M128/256/384,N128/256/512/1024,K256/512/768: distinct all sixteen wave row/column bands, K halves and groups; verify single writer and no five-word Xq-loop overrun.
3. Fallback M1/63/64/127/129 and N1/127/129/255/257/384 as appropriate, real dispatcher; full N384 is **eligible** because divisible128, not a256-tail. Include invalid-K rejection without launching raw invalid candidates. Zero cases remain unchanged, not newly exercised through unchecked raw entries.
4. Existing family set M1024/6144/10240/12288/17408,K5120; add M5120,K5120/6144/17408; N128/256/512 and1024. Nonzero/poison set destination, varied finite nonzero residual add starts.
5. Codes weight0..15, Xq−8..7, nonzero exact s, zero blocks/cancellation, different h0/h1 and per-group scale/zp, representable signed-zero/small/large finite scales under current FP mode, and deliberately non-power-of-two folds that distinguish contractions by1 ulp. Include data through the real quantizer and producer-emitted prepared sidecars, not only host-packed data.
6. Change X/sidecar contents in the **same allocation** between successive launches: results must change appropriately; stale prepared handles still reject. Eager params vs force-blob exact equality. Actual graph capture/recorded HIP replay must preserve shipping fallback entry/block/grid, output bytes, and ownership/lifetimes; reset add/input states between independent replays.
7. gfx1151 candidate passes all eligible cases; gfx1100 old-vs-pinned arithmetic and existing A5/tail/capture behavior remain exact, with no lf16 lookup. Other architectures: compile/eligibility isolation only, not unsupported numerical assertions.

On mismatch fail nonzero with architecture/object/symbol/mode/shape/seed, first `(row,column)` and both hex words, mismatch count and relevant headers. Missing hardware evidence blocks admission. Main's one-chunk `eval_hipfire` gate uses the identical model/reference/settings and requires Halo md5 **`cc7f5a26e98217f8139d85c1f7a37044` unchanged**, both after W4.0 and final integration. gfx1100 requires its own archived unchanged reference digest (same supplied digest only if verified as that device's reference). KLD aggregate md5 is supplemental to direct bit checks.

### G2 — rocprof mechanism: wait must drop, not just register count

After G0/G1, profile same-shape/mode A5-pinned baseline and candidate separately with identical rocprof counter sets, warmup, device state and kernel dispatch count. At least three paired interleaved counter captures per production shape; preserve raw counter numerators/denominators and profiler metric definitions, replay passes and object identity. Compare `SQ_WAIT_CNT_ANY` using the same **wave-cycle normalization** as the supplied36%/47%; also report raw wait cycles per output/work amount, achieved waves/residency when available, VM/LDS wait split if supported, cache hit/miss/bytes, barrier wait and LDS conflicts. No unavailable counter is fabricated.

**Mandatory mechanism gate: `SQ_WAIT_CNT_ANY` normalized wait fraction must decrease for BOTH shapes by more than run-to-run spread**, and normalization must not merely hide unchanged/increased wait work behind extra issued wave-cycles. Raw wait/work and the available global-memory counters/ISA must support the claimed A-load concurrency mechanism; investigate denominator/traffic changes rather than accepting an artificial ratio drop. Ideal22%/31% from §3 is not a required absolute target. Occupancy alone, fewer registers, or a timing win with unchanged wait does **not** pass this ticket. Increased barrier or LDS-conflict cost must be reported and included in end timing. If wait rises or cannot be measured, no admission.

### G3 — at least15% per-call reduction on each production shape

Isolation:100 paired interleaved HIP-event samples after matched warmup, alternate arm order, same allocations/data and add restoration outside intervals, no compile/quantize/download in timing. For gate-set **and** down-add, require median paired ratio≤0.85, improvement larger than paired spread; if using a confidence interval its upper bound must also≤0.85. Archive min/median/max and raw pairs. Profiled-counter timings do not substitute for uninstrumented event timing. No average may hide a failing shape.

In-model attribution: prebuilt `/home/kaden/ClaudeCode/warpfront/wt-lloyd/target/release/examples/profile_prefill_qwen35`, accepted model, `--prefill 512 --warmup 3 --kv-mode q8`, then separately2048. Three fresh-process ABBA cycles, accepted IU4/FA2/Q8-EF/cadence settings; only this kernel unit differs between arms. Require≥15% per-call reduction for **each target shape in both prompt profiles**, unchanged shape/call and producer/quantizer counts, no hidden other-shape regression. `/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/saddle-lab/examples/profile_prefill_qwen35.rs:179-195` groups category+kernel, not dimensions: H may temporarily add static shape-qualified timer labels at `/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/rdna-compute/src/gemm.rs:19117-19118`, distinguishing exact gate/down/mode from others, while archiving actual symbol/object/launch geometry. Do not infer either production shape's time from a family average. Remove diagnostic labels before final uninstrumented model runs.

### G4 — production transfer and reviewer gate

Prebuilt `/home/kaden/ClaudeCode/warpfront/wt-lloyd/target/release/hipfire bench qwen3.8:27b-mq4-xt --matrix --pp 512,2048,8192,32768 --ctx 128,2048,32768 --runs 1 --warmups 10 --kv-mode q8`, same three fresh-process ABBA cycles and accepted predecessor. **pp512 and pp2048 must improve beyond paired noise**, no>2% regression in any existing prompt/decode row. Record actual pp2048 t/s;710 is the campaign target, not a number earned by passing G3. A halogen headline additionally needs the matching HTTP done-field prompt measurement, not profile wall time.

Main owns unchanged-eval digest, actual serve deterministic-text battery and existing retained-path isolation gates in `/home/kaden/ClaudeCode/warpfront/wt-lloyd/docs/plans/2026-09-15-halo-gap-plan.md:318-369`. Those are shared admission checks, not composer suites. Independent reviewer can veto arithmetic, state, metadata, mechanism or evidence even when timings pass.

## 8. Ordered composer units, ownership and stop rule

Interfaces in §§4/6/7 are frozen before dispatch. No composer runs project-wide builds/tests, formatters, linters or GPU commands; Main schedules artifact-specific compilation/validation. Kernel-file mutations are serialized; disjoint oracle/host preparation can run concurrently after the pin. No sibling touches prefill.rs for this unit.

| Unit / owner | Exact files and change | Independent acceptance / handoff |
|---|---|---|
|**LF.0 = W4.0 / F**|Only the shipping IU4 HIP file/common fold, original and A5 entries; archive original objects and pin verified per-architecture/path DAG|Actual original→pinned ISA/bits/md5 on gfx1151/gfx1100, no material paired timing regression; no tile/schedule/host change. Stop before LF.1 if unavailable/failing|
|**LF.1 / K** (after LF.0)|Only same HIP file: complete guarded lf16 helpers/entries,32 sums, one-chain/fold lifetime, packet4 A fill,16-wave mapping|Submit full implementation, source/resource expectations and CPU ownership/fill proof to Main; Main G0 then G1 gates. Both modes complete, no placeholder or host route enabled on a failed object|
|**LF.2 / O** (concurrent with LF.1)|Only fresh oracle example; frozen names, launch geometry, phases and fixture/report contract from §7|Reference/candidate cache identities distinct; no add=0 shortcut; exact guard/bit/state output. Main runs after G0. Submit required manifest stanza to H, never edit H's file|
|**LF.3 / H** (preparation concurrent with LF.1/2; activation after G0/G1)|Only gemm.rs selector/launch tuple/doc comment; necessary Cargo example registration; temporary shape labels; no model caller or replay edits|Preserve every fallback/A5 branch, ABI and flags; submit routing matrix to O/Main. Diagnostic candidate build has G0/G1 proof before model execution|
|**LF.4 / Main hardware owner**|No speculative source edits: exact complete object→G1→G2→G3→G4|Archive raw artifact/bit/state/counter/timing/model evidence; classify bottleneck and pass/fail for both shapes independently|
|**LF.5 / existing file owners, Main**|At most one evidence-driven correction within same tile/ABI/barriers/pin/scope|Name one ISA/counter failure and predicted observable fix before editing; repeat complete gates on the new object. No third arm or switch to a costed alternative|
|**LF.6 / H + Main + reviewer**|After successful smoke/model proof: retain admitted route; remove oracle/registration/labels/candidate-only diagnostics; update this plan and existing campaign ledger with measured results|Reviewer final veto. If rejected, remove lf16 helpers/entries/host route and experiment scaffolding, retain independently admitted W4.0 and A5. Preserve measurement evidence, do not commit|

**Bounded iterations:** iteration1 is exactly selected128×128/16 waves/≤96/packet4/one-plane/four-barrier design. Iteration2 can fix **one** measured current-packet, address lifetime or fold-scheduling defect without changing arithmetic, tile, wave count, register cap, architecture or route envelope. It is not authorization for another shape, split-K, extra chain array, producer pipeline, noinline, relaxed bits or hidden compiler flags. A failed metadata iteration counts. If either shape still fails any required resource, exactness, wait-drop,≥15% isolation/in-model or bench transfer gate after iteration2, **kill**; inconclusive/noisy/unavailable required evidence is not admission.

**Timebox:16 active engineering hours total**, at most two booked hardware-owner sessions: prerequisite≤3h; concurrent LF.1/2/3 critical path≤4h; first artifact/hardware gates≤4h; one correction+regates≤4h; reviewer/admission handoff≤1h. Queue time does not authorize more iterations or performance claims. Stop earlier if the verified fold cannot be pinned, the96-register target needs forbidden techniques, or the mechanism has no measured wait reduction. Reviewers may stop a hopeless trade before the second iteration.

## 9. Risks and planning proof

1. **Compiler lifetime cliff:** naive sum32 still leaves deferred C32/headers/addresses at roughly118 used;96 requires actual prompt folding and shorter cached-address lifetimes. One extra allocated group104 drops selected residency from32 to16 waves. Explicit RN fixes arithmetic, not liveness.
2. **Wait-counter attribution:**36–47% wave waits are not36–47% exposed wall; extra waves may mask stalled waves without lowering their wait metric, or VM/LDS/issue limits may dominate. This ticket nevertheless requires a measured wait-fraction drop, so no fallback narrative can admit a different mechanism.
3. **Xq opportunity does not scale with waves:** selected has the same144 theoretical outstanding wave-load slots/CU for Xq; down's larger Xq footprint can defeat the optimistic23.5% projection. A packet4 may saturate memory queues/cache service rather than help.
4. **Finite grid/phase balance:** down160 WGs across40 CUs is only two scheduling rounds at two WGs/CU. Sixteen-wave rendezvous, header fill by only eight waves, and load-phase alignment can limit achieved residency; compiler's16 waves/SIMD is only capacity.
5. **Numerical hazard:** mixed half/FMA instructions, signed zero/denorm modes and contraction across variants/architectures previously caused real1-ulp drift. No tolerance waiver or inferred equivalent algebra.
6. **Inherited route scope:** current eager gfx1100 uses A5 column order; capture/recording deliberately do not. Mistaking old W4/W2 text for current dispatch can regress either route. Function/module cache identity and dynamic LDS must travel with the new512-thread block.

Planning evidence actually executed: standalone shipping gfx1151 compile returned190/21/zero spills/scratch for full entries; the saved `.s` was inspected to identify all190 physical register roles, deferred64-integer lifetime, load issue windows, fold operations and descriptor fields. CPU checks verified the disjoint190-register ledger, resource/traffic/projection arithmetic, selected16,384-output unique ownership,4096 A payload-word writers,256 header writers,2304 Xq-word writers/half,18/42-stride bank-address permutations, and four barriers/group for G1/2/3/20/68 with increasing per-output K order. These are **offline source/ISA/indexing facts**, not candidate resources, GPU parity, actual occupancy, stall reduction, eval digest or speed. No GPU result is claimed; zero performance credit is banked.
