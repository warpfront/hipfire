# gfx1201 iu4 fold overlap: issue model and bounded next experiment

2026-09-18. **Plan only; no GPU used, no production source changed.** Baseline is the `mq4-lloyd` state identified in the assignment as `cef913aba`. Source line landmarks below were read before `Iu4WaveSpec`'s concurrent kernel edits; the observed kernel SHA256 and the exact older, matching-v0b ISA input are recorded in the evidence ledger. They are not a claim that a concurrently edited checkout still contains those bytes.

**Decision:** try exact FP32 VOPD pairing first; treat wave specialization as a bounded, falsifiable experiment, not a presumed FA3 speedup. None of the three options has evidence supporting a standalone **gate/set <=400 us** promise. Two important corrections are required: the old census omits VOPD from its fold count, and AMD's RDNA4 matrix calculator says IU4 K32 cannot co-execute with VALU.

Paths: `KERNEL` means `kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx12.hip`; `ORACLE` means `crates/hipfire-runtime/examples/tmp_iu4_gfx12_oracle.rs`; `E` means `.codeinsight+research/scratch-2026-09-17/Iu4FoldPlan/`, all relative to `/home/kaden/ClaudeCode/warpfront/wt-lloyd`. New evidence is confined to `E`, never `/tmp`.

Primary references:

- **[ISA]** [AMD RDNA4 ISA Reference Guide, 7-April-2025](https://docs.amd.com/api/khub/documents/uQpkEvk3pv~kfAb2x~j4uw/content). Section/page numbers below are the guide's printed numbers, not PDF page indices.
- **[SYNC]** [LLVM AMDGPU Execution Synchronization, GFX12 and barrier-ID table](https://llvm.org/docs/AMDGPUExecutionSynchronization.html#gfx12).
- **[USAGE]** [LLVM AMDGPU Usage, Named Barriers](https://llvm.org/docs/AMDGPUUsage.html#named-barriers), [execution-barrier memory model](https://llvm.org/docs/AMDGPUUsage.html#amdgpu-amdhsa-execution-barriers-memory-model).
- **[MATRIX]** [AMD Matrix Instruction Calculator, `matrix_calculator.py`, version 1.3.2, RDNA4 IU4 K32 entry](https://github.com/ROCm/amd_matrix_instruction_calculator/blob/main/matrix_calculator.py#L2568-L2588); [execution-statistics printer](https://github.com/ROCm/amd_matrix_instruction_calculator/blob/main/matrix_calculator.py#L5222-L5259). Retrieved through GitHub `file_read`; moving-main line links are backed by the source ledger's observed values.
- **[SCHED]** [LLVM `SISchedule.td`, `GFX12SpeedModel` and `SISchedMachineModel`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/SISchedule.td). This is a compiler model, not a silicon timing measurement.

## 1. Current K-loop: count packets, then model service cycles

### 1.1 Source and measured baseline

`iu4_bundle<SB,FIRST>` at `KERNEL:189-222` performs eight WMMAs, with calls at `405-406,436-437`: **32 K32 instructions per wave per 128-K half**. The workgroup has eight waves; each owns 32 rows x64 tokens (`233-280`). It has eight independent 16x16 C fragments, each eight i32 registers/lane: **64 i32 C +64 FP32 sum registers/lane**, before staging and addresses (`278-328`). The integer chain restarts at each half, with `FIRST=true`, and does not span halves.

The exact fold is `IU4_FOLD_RN` (`163-179`), invoked at `512-532`. There are **64 scalar-output folds per lane**, not eight total: eight components times eight fragments. The graph for each output is:

```
t1 = RN(sc_h * d_h)
p = RN(t1 * float(C_h))
t2 = RN(zp_h * d_h)
term = RN_FMA(t2, float(s_h), p)
sum = RN_ADD(sum, term)
```

Existing metadata receipt `wt-iu4/scratch-2026-09-17/GemmV2Plan/iu4fold/vgpr-v0b.txt:1-3` reports 182 VGPR and zero spills/scratch for all three entries. The assignment reports measured 3 WG/CU. Existing `wt-lloyd/.../GemmV2Plan/iu4fold/gate4-time.log:92-96` records 479.7/495.1/504.4/474.1/394.3 us for gate/set, gate/add, down/add, qkvza/set, qkv/set. The assignment's earlier 475/495/504/470/392 us bracket is the forecast reference here; no baseline was rerun.

Attribution caveat: the report actually found at `wt-iu4/scratch-2026-09-17/GemmV2Plan/iu4attrib/REPORT.md:44-58,101-116` measured an older ~695-us, 239-VGPR kernel: fold 34.9%, staging 43.5%, WMMA 19.1%, store 11.0%, with explicitly nonadditive ablation marginals. Its scale-shuffle cost was subsequently removed. The assignment's current 35/25/20/10 split is useful as a **planning estimate**, not a newly measured decomposition of the 475-us kernel. Do not add those percentages or present them as a current roofline.

### 1.2 Corrected steady-state ISA census

CPU-only analysis of the existing `v0b-dis.txt`, `full_add` loop `0x826c..0x8f64`, reproduced the old 617-packet loop and expanded both sides of every `::` VOPD instruction. Evidence: `E/baseline-steady.isa`, `E/analyze.py`, `E/analysis.json`. The original `wt-iu4/.../iu4fold/isa_census.py:27-32` recognizes ordinary `v_mul/v_fmac/v_add`, **not `v_dual_*`**. Therefore its `F=242` and W/F alternation metric are not a complete fold issue census.

| Fold operation | Scalar operations, including VOPD halves | Singleton packets | Operations inside dual packets |
|---|---:|---:|---:|
| i32 -> f32 | 68 =64 C +4 reused token sums | 68 | 0 |
| FP32 multiply | 192 =3 x64 | 101 | 91 |
| FP32 fused multiply-add | 64 | 36 | 28 |
| FP32 add to running sum | 64 | 37 | 27 |
| **Total** | **388** | **242** | **146 in 73 packets** |

Thus the actual fold has **315 issue packets**, already saving 73 versus an all-singleton 388. There is no four-times missing fold factor to recover. The remaining loop also contains 47 classified load/store packets, four barrier signal/wait pairs, address/predicate instructions, scalar control and waits. The 315 count is fold arithmetic only, not all VALU, and a packet count is not elapsed cycles.

### 1.3 Matrix service versus VALU issue

- [ISA] §2.1 p9: wave32 issues an instruction once, unlike wave64's usual two passes. §7.8 pp82-84: VOPD executes two legal FP32 operations in parallel in wave32 only. It is one issued packet, not two serial packets.
- [ISA] §7.12 pp89-90 and §7.12.1 pp90-91 define K32 integer WMMA, signedness and dependencies. They do **not** provide a guarantee of an independent matrix pipe that runs concurrently with arbitrary VALU.
- [MATRIX] gives this exact RDNA4 instruction `cycles=8`, `coexec=False`, `coexec_delay=-1`. Its printer calls the field “Can co-execute with VALU.” Use eight matrix service cycles/instruction, **not** one issue cycle as the full cost, and do not replace matrix-plus-fold work with `max(matrix,fold)` without contrary device evidence.
- [SCHED] models ordinary FP32 ALU, FP32 conversions and FP32 FMA with five-cycle dependency latency and `IssueWidth=1`. **[INFERENCE/model]** use one ready ordinary/VOPD packet per SIMD issue cycle, with enough independent elements to cover latency. Five-cycle dependency latency is not five cycles of throughput per independent element. [ISA] §5.8 pp56-57 explains scheduling delays and their optional nature; §7.12.1 allows hardware stalls on dependent WMMA/VALU accesses.

Per original wave/half, the optimistic **shared-service** arithmetic model is therefore:

```
WMMA: 32 *8 =256 matrix service cycles
fold: 242 +73 =315 ready VALU issue cycles
arithmetic-only total: 571 SIMD service cycles
fold:matrix service ratio =315/256 =1.2305
```

This is a lower-bound planning model, excluding DS/global traffic, address VALU, dependency bubbles, barrier latency, store and occupancy. Multiplying it by wave count and dividing by available SIMD engines is a throughput accounting exercise, not a latency prediction. Other waves can cover memory/dependency bubbles; they do not manufacture extra arithmetic capacity on the same SIMD. Splitting roles across different SIMDs also partitions the available capacity and does not provide free overlap.

Existing `.../wmma_probe/run1.txt:1-8` measured 539.7 TOPS K32; the register-resident four-chain probe is at `.../wmma_probe/wmma_probe.hip:55-65,83-106`. This corroborates high saturated WMMA throughput, but measures neither WMMA/VALU coexecution nor ring overhead. `Iu4FoldSched`/`Iu4PingPong` receipts already show schedule alternation alone did not improve TIME; the double-i32-bank experiment spilled. Do not repeat those sweeps.

## 2. Wave specialization: a legal bounded design, not an assumed win

### 2.1 gfx1201 synchronization facts

**Split barriers exist; named/subset barriers do not.** [ISA] §5.6-5.6.2 pp48-50, Tables24-25, and [SYNC]'s ID-availability table agree:

| Primitive/object | gfx1201 availability | Use here |
|---|---|---|
| `s_barrier_signal -1`, `s_barrier_wait 0xffff` (workgroup ID -1) | Yes; all workgroup waves count | Initialization, final join and store phases only |
| ID -2 trap barrier | Privileged trap handler only | Never use from kernel |
| Named IDs0..16, init/join/leave | gfx12.5+, not gfx1201 | Unavailable; do not import gfx1250 examples |
| `s_sleep 1` | Yes, about64 clocks; `s_sleep 0` sleeps zero | Failed LDS-flag poll backoff |
| Dynamic per-role VGPR allocation | gfx120* metadata field reserved; gfx12.5+ feature | Unavailable |

`signal/wait` also works in an eight-wave/256-thread workgroup, but cannot become a four-wave producer barrier by passing a different count. It still includes every live member. [SYNC] explicitly says barriers synchronize execution, not memory visibility. Pair memory release/acquire with them; raw `s_barrier_signal` is not a store fence. Do not signal then abandon the corresponding wait or exit: [ISA] §5.6 warns that this can corrupt barrier state.

**Autonomous producer/consumer roles need workgroup-scoped LDS atomic sequence flags**, plus compiler ordering and DS completion. An LDS volatile flag alone is insufficient. For gfx12 the relevant wait is **`s_wait_dscnt 0`**, not legacy `lgkmcnt`: [ISA] §5.7.1.4 p55 says DS writes are complete when written into LDS and DS loads when data reaches VGPRs. Emit wave-uniform poll loops; a leader can poll then broadcast the result with readfirstlane, but acquire/release ordering must cover all participating lanes. A full-wave DS wait and compiler memory fence must precede the leader's publication; similarly all payload reads must finish before its credit release. Audit IR address-space3, workgroup syncscope and the resulting ISA; generic-pointer atomics that become global/flat traffic fail the gate. A narrowly encapsulated inline-ISA helper is acceptable only with explicit memory operands/clobber, full-wave ordering and reviewer approval, not as a way to ignore the language memory model.

After each failed poll, `s_sleep 1` and reload. Never indefinite sleep or wakeup-only progress. The sleep is a performance tradeoff, not a memory fence or a fairness proof. There must be no path where a producer waits for ring credit while the needed consumer is blocked at a default workgroup barrier. All waves must be resident/admitted together for the chosen launch, confirmed by the occupancy/runtime gate.

### 2.2 Frozen geometry, roles and LDS layout

Recommend **8 producer WMMA waves +8 consumer fold waves**, 512 threads, same128-row x128-token tile, 64-K staged slabs. This is the simplest ownership-preserving 1:1 mapping; role owner `u=wave&7`, producer `wave<8`, consumer otherwise. Keep the original owner mapping `row0=(u>>1)*32`, `tok0=(u&1)*64`. Producer lanes0..255 retain the existing staging maps at `KERNEL:281-314,335-367,383-500`; consumers do not stage A/W.

Why1:1: per equal output area, ideal work is256 matrix versus315 fold cycles; a hypothetical equally provisioned independent-service balance would want `F/P=1.23`. Eight/eight is close, maps ownership exactly, and producer staging/transfer work pushes the producer side upward. Four/eight underprovisions matrix/staging and doubles producer C storage; eight/four doubles consumer sum storage. Extra fold waves cannot improve the shared arithmetic-capacity bound. This is a resource/work-balance recommendation, **not** a claim that matrix/VALU will overlap on one SIMD.

A full half's output is128*128*4=**65536B**. Neither a one-slot full-half ring nor eight depth-two1024B fragment rings fit alongside19456B of staging. Instead split each16x16 fragment's eight per-lane components into two four-component packets:

| Byte range, half-open | Object | Ownership |
|---|---|---|
| `[0,19456)` | Unchanged A/W0/W1/DS0/DS1/SZ0/SZ1 from `KERNEL:139-150` | Producer cohort; metadata additionally read by consumers |
| `[19456,27648)` | `i32 ring[8][2][4][32]` | Eight SPSC channels, depth2,512B/slot |
| `[27648,27712)` | `u32 seq[8][2]` | Per-slot producer/consumer handshake |
| `[27712,27744)` | `u32 producer_epoch[8]` | One producer writer per counter; producer subset barrier |
| `[27744,27776)` | `u32 fold_done[8]` | One consumer writer per owner; metadata reclamation |
| `[27776,28160)` | Allocation padding | No payload |

**Launch LDS=28160B**; logical payload/control end27776. Even using the conservative64KiB/CU budget, `2*28160=56320<=65536`, leaving two-WG LDS headroom. [ISA] §12.1 p144 describes128KiB/WGP split into two32-bank sets; CU/WGP naming in occupancy tools must be recorded rather than silently equated. LDS fit is necessary, **not sufficient**, for >=2 WG/CU.

Packet index per owner is `q=16*kb+2*(4*rg+nb)+part`, `rg=0..1`, `nb=0..3`, `part=0..1`, `slot=q&1`. Payload component `j=0..3` is the exact i32 `cacc[nb][rg][4*part+j]` from the same lane. Address:

```
19456 + (((u*2 +slot)*4 +j)*32 +lane)*4
```

Each component plane touches all32 banks exactly once; no padding is needed for b32 SoA access. `E/analyze.py` checked all owner/slot/component banks and all16384 output coordinates for bijective coverage. This is a CPU layout proof, not an LDS-bandwidth measurement. Do not force a contiguous per-lane b128 store onto this SoA layout.

### 2.3 State machine and reclamation invariants

The all-wave prologue initializes `seq[u][slot]=slot`, `producer_epoch=0`, `fold_done=0`, stages half0/slab0 and publishes with a memory-ordered all-WG barrier. Then roles diverge in wave-uniform branches.

| Transition | Before / owner | Operations | After / invariant |
|---|---|---|---|
| Slab read -> overwrite permission | All producers can still read current A slab | Each producer drains DS reads, release-stores next epoch to its own counter, then acquire-waits every producer counter >= that epoch | No producer can still read overwritten bytes; consumers never read A/W |
| Slab write -> compute permission | Producers own disjoint stage writes | Global-load completion, LDS stores, DS completion, same producer-only epoch barrier | All producer waves see complete A/W slab before WMMA; keep the original four phase boundaries, not four all-WG barriers |
| C accumulation -> finished packet | Only producer owns64 C registers; same four K32 operations per half | After that fragment's last WMMA and required hazards, acquire-wait `seq[u][q&1]==q` | Slot belongs to producer; C for this packet is complete and has not been reused |
| Packet fill -> publication | Slot private to producer | All lanes write four i32 components; full-wave DS completion/release; leader publishes `q+1` | Consumer may acquire exactly this generation; no stale parity-only flag |
| Publication -> snapshot | Consumer waits `seq==q+1` | Acquire, load four components, wait DS; acquire corresponding DS/SZ metadata protected below | Consumer holds exact C and metadata values; producer still cannot overwrite payload |
| Snapshot -> credit | Consumer has drained all packet reads | Full-wave ordering; leader release-stores `q+2` | Producer can reuse slot for q+2 even while consumer folds its private registers |
| Snapshot -> fold | Consumer exclusively owns64 FP32 sums | Invoke original pinned scalar DAG for these four distinct outputs; each output processes kb in increasing order | Its sum equals original after that output's kb update; never accumulate partial FP32 sums in another wave |
| Completed half -> metadata credit | Consumer has folded all16 packets of kb and has no outstanding DS/SZ reads | Release-store `fold_done[u]=kb+1` | Producers can reclaim this metadata parity once all relevant consumers acknowledge |
| Metadata parity reuse | Before stage of half t, parity holds t-2 | For t>=2, acquire-wait all `fold_done>=t-1` before writing DS/SZ[t&1] | No consumer of t-2 can observe metadata from t; ring credit alone is NOT sufficient |
| Final drain -> store reuse | Producer has published all16*(K/128) packets; consumer may still fold | Producers finish without entering any additional subset barrier; all roles join once consumers finish; DS/fence + default workgroup barrier | All compute planes and ring are dead; no missing last-half fold |
| Store phase | Consumers own final sums | Eight consumers write original owner-indexed stride20 slots; all16 waves barrier; only256 producer threads perform original coalesced stream; all16 barrier again | One write/output, identical set or one final residual add; LDS may be reused next phase |

The producer epoch counters are monotone: use `>=phase`, never equality with a reset-to-zero barrier. A fast producer cannot skip a phase another producer has not reached. The ring uses exact per-slot sequence equality and depth2 advancement; no cross-dispatch state survives. The K contract is already i32 and divisible by256; with positive i32 K the last packet generation `K/8` remains below2^32, so this dispatch cannot wrap u32 sequence numbers. Tails follow the original zero/clamp rules and **do not remove lanes/waves from synchronization**. Only the original workgroup-uniform whole-tile out-of-range return is allowed.

Producer software barriers use the same memory-ordered flag primitives: publish epoch after full-wave DS completion, acquire-observe every producer, continue. No consumer participates in these barriers. A consumer never waits for an epoch that requires its own future `fold_done`; it obtains current-half metadata after packet publication. This breaks the common ring-credit/default-barrier deadlock cycle.

The first producer may begin publishing a fragment immediately after its fourth WMMA, instead of waiting for all32 instructions. However it cannot reuse that C register until its DS store has consumed the value, and it cannot begin next-half FIRST overwrites of unsent fragments. Producers do not retain a second C bank. Depth2 gives one queued fragment or two half-fragments, **not** a complete half of buffering. Backpressure will frequently couple the roles; report it rather than calling the design fully asynchronous.

### 2.4 Exactness, register budget and costs

Data movement preserves all32 C bits. Weight half scales and activation d/s remain tied to the same kb. Each consumer owns the same outputs for the entire K loop, initializes sum to+0, performs identical conversions/multiply/FMA/add nodes and rounds in the same mode. Interleaving independent outputs is legal; interleaving/reassociating two halves of one output is not. `KERNEL:163-179` and `ORACLE:217-250` remain the numerical specification. Unsigned weight x signed activation accumulation over128 terms has magnitude <=15360, so no i32 overflow; nevertheless transport i32 unchanged in this experiment. Per256 folding, FP16 intermediates, TF32, split-K floating reduction and packed-i16 ring compression are non-goals.

| Role | Mandatory live data | Planning allowance, not measured allocation |
|---|---|---|
| Producer | 64 i32 C; fragment operands; hoisted addresses; existing8 prefetch dwords | Aim112-128 total VGPR, zero FP32 running-sum bank |
| Consumer | 64 FP32 sums;4 C snapshot; current metadata and four DAG temporaries at a time | Aim96-120 total VGPR, no WMMA C bank |
| Whole kernel | Static role-uniform branch; allocator must reuse registers across disjoint roles | Common allocation target<=128 VGPR; zero VGPR/SGPR spills and scratch; actual occupancy>=2 |

Role counts do **not** create role-specific physical allocations: the code object's common maximum applies to every wave. [ISA] §3.3.2.1 p17 gives16-VGPR wave32 allocation granularity, or24 on1536-VGPR/SIMD devices. At24 granularity,128 rounds to144:32 resident role waves*144 equals24 baseline waves*192, so the target does not exceed the reported baseline's aggregate VGPR allocation. This is only an occupancy feasibility argument; register distribution, wave slots and hardware mode still need a real occupancy query.

Every half adds **64KiB LDS writes +64KiB LDS reads/workgroup**, despite using only8KiB ring storage. The unchanged input slabs stage16KiB/half. [ISA] §12.1's32 banks x4B, each1R/1W per clock, gives a best-case512 read cycles and512 write cycles per32-bank set for the ring (possibly overlapping each other), before flags, metadata, arbitration and routing. This is substantial and previously “LDS exonerated” measurements did not include it. With [MATRIX]'s no-coexecution constraint, specialization must win through measured latency/occupancy/staging effects large enough to pay these costs. Do not budget the35% fold wall time as removable.

The sync facts and ring recommendation were sent to `Iu4WaveSpec` through hub as soon as settled; the corrected coexecution constraint was sent immediately afterward. Kernel ownership remains with that worker.

Worker coordination update: `Iu4WaveSpec` selected a distinct 4-producer/4-consumer experiment with 256 threads, a union of 128 i32 C or 128 FP32 sums per lane, a 4x2x1024B ring, and catch-up before each workgroup barrier. That union avoids simultaneous 256-register accumulator liveness, but needs additional registers for operands/staging and transfers the same 128KiB/half. Its measured result may supersede the ranking; it is not the frozen 8+8 design above. Catch-up barriers limit overlap and require a separate progress proof. **Do not assume even/odd workgroup wave IDs guarantee placement on particular SIMDs**: [ISA] §2.3 p11 allows the workgroup's waves to run on any of the four SIMD32s. Placement-based claims require additional hardware evidence.

## 3. Exact dual-issue/packed fold alternative

### 3.1 Legal pairings and prohibited rewrites

[ISA] §7.8 pp82-84 and §15.3.7 pp188-190 list `V_DUAL_MUL_F32`, `V_DUAL_FMAC_F32`, `V_DUAL_ADD_F32`; **no dual i32->f32 conversion** exists. Ordinary VOPD preserves FP32 operation semantics, including the [ISA] §7.2.4 p73 round/denormal mode. Do not choose the DX9-zero multiply form or change mode bits.

For two independent output components a/b, all five arithmetic stages can pair:

| Stage | Candidate VOPD packet | Rounding constraint |
|---|---|---|
| t1 | `mul(sc_a,d) :: mul(sc_b,d)` | Two separate RN products |
| p | `mul(t1_a,Cf_a) :: mul(t1_b,Cf_b)` | No FMA contraction with later terms |
| t2 | `mul(zp_a,d) :: mul(zp_b,d)` | Still rounded before affine correction |
| term | `fmac(t2_a,sf,p_a) :: fmac(t2_b,sf,p_b)` | Destination initially contains p; each side is exactly one fused RN FMA |
| sum | `add(sum_a,term_a) :: add(sum_b,term_b)` | Separate final additions, no affine+sum FMA |

`t1_a` and `t2_a` may also pair if their registers meet the restrictions; independent cross-stage nodes from distinct outputs can pair. Never pair dependent t1->p, p->FMA or FMA->sum as though the second side consumes a newly produced first-side result. Source sharing of the same d/s register is allowed by the guide's same-register exception.

Physical legality must be demonstrated, not assumed from `float2`: opposite-parity destination VGPRs; SRC0/SRC1 paired reads in different banks modulo4 or the documented identical-register exception; FMAC's implicit SRC2/destination reads even/odd; SGPR/literal limits; no DPP; full wave32. The guide permits a limited asymmetric old-value source/destination overlap, but the proposed pair helper does not depend on it. Never hardcode reserved physical registers or introduce operand-copy traffic that cancels saved packets. Pairing intrinsics at the C++ level is an experiment; LLVM may choose not to encode VOPD.

[ISA] §7.7 pp80-82, Table37, and §15.3.6 list packed FP16/integer16 operations and mixed-precision dot/FMA forms, **not a native two-FP32 `v_pk_mul_f32`/`v_pk_add_f32`/`v_pk_fma_f32` route on gfx1201**. Converting the intermediates to half changes the DAG. FP16 header storage does not make `sc*d`, `zp*d`, C-derived p or sum FP16 quantities. CDNA packed-FP32 ideas and FP16 dot-product fusion are out of scope. “Packed” in this plan means two independent FP32 VOPD operations, not changing precision.

### 3.2 Actual remaining headroom

The320 FP32 operations can ideally occupy160 dual packets;68 conversions remain singleton. **Best arithmetic-only floor=228 packets**, versus current315, a saving of87/315=**27.62%**, not50%. The current kernel already uses73 of the possible160 dual packets. The idealized whole arithmetic service model falls571->484 cycles (15.24%), before other costs. Moves, bank restrictions and dependency stalls make228 an optimistic floor, not an expected compiler result.

Use a bounded compile experiment on two independent j components at a time. Proposed local interface, not an exported API: `iu4_fold_rn_pair(sum0,sum1,sc0,sc1,zp0,zp1,d,C0,C1,sf)`, two sum references, scalar FP32 metadata, i32 C values, and the exactly converted token sum `sf`. Stages are two t1 multiplies, two C conversions, two p multiplies, two t2 multiplies, two FMAs, two adds; all intermediates stay FP32. Keep sf reuse across both rg values (the baseline has only4 sf conversions per half). The existing scalar macro remains the specification and gfx1200 path; use a gfx1201-specific paired loop/helper only if it produces a measurable win. Do not keep an unused candidate or add a public compatibility abstraction.

Initially retain the current rg/nb traversal and all existing staging/synchronization. If a two-component source schedule makes the allocator lose already-existing cross-element VOPD or raises register pressure, it fails. No wide live temporary arrays, second accumulator bank, new scheduler profile or unrolled two-half window. Inline VOPD is a second bounded compile attempt only when the target assembler/constraints can guarantee valid pairing and preserve the pinned operations; assembler success alone does not establish numerical identity.

**[INFERENCE]** a useful candidate might reach245-275 packets:13-22% fewer fold slots. At an assumed35% exposed fold fraction, that predicts roughly438-454 us before new overhead; give a broader440-465 us planning range. The ideal228 bound predicts429.1 us under that same assumption. Reaching400 would require the exposed fold fraction to be>=57.17% at ideal pairing, or additional independently measured savings outside the fold. The older34.9% ablation does not justify that premise.

## 4. Tile/wave-shape alternative

The fold-to-WMMA arithmetic ratio is intrinsic to 128-K affine quantization, not the number of waves. For a fixed 128x128 tile, workgroup totals remain 256 WMMAs and **16384 scalar output elements** folded per half; dividing by 32 lanes gives 512 vector-component folds distributed across waves. Geometry changes register pressure, reuse, tails and barriers, not this arithmetic count.

| Shape, rows x tokens | Waves / per-wave output | Consequence |
|---|---|---|
| Current128x128 | 8 /32x64 | 64 C +64 sums per lane;32 WMMA/half |
| Same128x128 with4 waves | 4 /64x64 | 128 C +128 sums =256 VGPR minimum before any operands/addresses. Reject as a direct substitution; streaming/recomputation is a different algorithm |
| Same128x128 with16 waves | 16 /32x32, four row owners x four token owners | 32 C +32 sums;16 WMMA/half; roughly half fold work/wave, same work/WG. Worth one bounded occupancy experiment |
|128x64 with4 waves | 4 /64x32 or32x64 | Fits the old64+64 accumulator budget, but is a smaller tile: doubles token-direction WG count and weight staging per output area. Not a free “4 waves x64 rows” optimization |
|256x64 or64x256, same area | 8 with altered owner map | Input slab bytes rise25% because rows+tokens320 versus256. Estimated compute-plane LDS28160B or20480B respectively with current buffering; no fold-count reduction |

For the sixteen homogeneous-wave option choose a concrete map: `row0=(wave>>2)*32`, `tok0=(wave&3)*32`, two16-row groups x two16-token blocks. Each wave owns32 outputs/lane, so `cacc[2][2]`/`acc[2][2]`, not the current4x2 arrays. Each512-thread staging pass covers all128 slab rows once (`tid>>2`); do not leave the existing two256-thread rounds in place or the second round writes out of bounds. Store slots require16*16*20*4=20480B, slightly above19456B compute space; launch LDS must be `max(compute,store)=20480`, not silently left19456. Coalesced store mapping and every host/oracle block-size use must migrate together.

**[INFERENCE]** target common VGPR96-128 and zero spills; occupancy may improve despite the larger block, but must be measured. There is no ratio improvement: both matrix and fold service per wave halve. The upside is lower pressure/latency bubbles; the downside is more waves/barrier participants, more duplicate metadata work if reuse is lost, changed store phases and potentially no additional resident workgroups. Existing gfx1151 LF16 is an ownership-pattern reference only (`kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip:332-337,586-605`); do not transplant its K16 lane mapping or FP contract into gfx12.

## 5. Ranking, kill criteria and composer slices

### 5.1 Ranked forecasts, all unmeasured

These ranges are planning judgments, not confidence intervals or new results. Gate/set baseline=475 us. A measured counterexample supersedes the ranking.

| Rank | Option | Expected gate/set TIME [INFERENCE] | Confidence / reason | Kill criterion |
|---|---|---|---|---|
|1|Exact VOPD fold pairing|440-465 us; ideal35%-fold arithmetic estimate429 us|Medium confidence in legal slot headroom, low-to-medium confidence compiler can realize it; no new memory traffic|No net fold-packet decrease after two narrowly distinct pairing forms, spills/scratch, occupancy below baseline3, any oracle mismatch, or same-session improvement<3% |
|2|16 homogeneous waves, same128x128 tile|445-495 us|Low; helps register footprint, not issue ratio; may be neutral or lose|Not lower pressure at zero scratch, no residency/latency benefit, or TIME improvement<3%; reject4-wave same-area form immediately on256-register accumulator floor |
|3|8 WMMA +8 fold-wave ring|480-620 us; upside below475 only if latency/occupancy wins pay transfer tax|Low forecast precision, high confidence that the naive FA3 independent-pipe argument is invalid; shared service plus128KiB LDS/half|Any unsafe synchronization, >32KiB LDS, actual occupancy<2, spills, or no >=3% TIME win; no continuation based solely on role separation/static alternations |

**Target gate: <=400 us remains unmet and is not forecast credibly by any standalone option.** A400-us result is a stretch acceptance target, not permission to change the pinned DAG or compare against the old695-us baseline. If the winner produces a small real improvement, report it honestly as below target and let the caller decide whether it is worth landing; do not label the target achieved. Do not combine several losing experiments to avoid an abandon criterion. Combining individually proven wins would require a new isolated measurement, not multiplication of speedups.

For specialization, a small mixed-role, register-resident control could test the no-free-overlap premise before building a full ring, but the planning agent does not run it. It must normalize matrix and fold work, active SIMD count, VGPR allocation and occupancy against the unsplit control; wall time from a control whose work is optimized away is invalid. If mixed execution is consistent with shared service and the full ring has no measured latency benefit, abandon the approach on gfx1201. No new performance claim may be based on a static W/F alternation metric.

### 5.2 Frozen source/call-site contract

The winning composer slice is **exact FP32 VOPD pairing**, not a wave-ring implementation. Kernel ownership must first be handed back by `Iu4WaveSpec`; no simultaneous edits to the same file. Compose independent evidence work concurrently, and serialize only kernel integration/device ownership.

- Kernel numerical contract: `IU4_FOLD_RN`, `KERNEL:163-179`; fold call region `512-532`; owner/register definitions `233-328`. Preserve `iu4_bundle` at `189-222`, its four calls, staging `335-500` and store `537-579` unchanged for this experiment.
- Entry points: `gemm_mq4g256v2_residual_mmq_iu4` at585; `_full_add` at599; `_full_set` at609. Same seven arguments and ADD semantics; both template instantiations must be proved. Runtime-add wrapper calls body<true/false> at592/594; specialized entries at605/615.
- Host: `Gpu::gemm_mq4g256v2_mmq_prequant_iu4`, `crates/rdna-compute/src/gemm.rs:19167-19255`, gfx1201 launch arm at19193, symbol choices19199-19208, grid/block/LDS19226-19238. Direct wrappers call it at19357(set) and19369(add). **No host launch/API change for VOPD**.
- JIT source: `crates/rdna-compute/src/kernels.rs:3471-3474`, `GEMM_MQ4G256V2_RESIDUAL_MMQ_IU4_GFX12_SRC` concatenates quant prelude+kernel. No second copied production source.
- Oracle: source concatenation `ORACLE:26-32`; `launch_iu4:169-195`; CPU DAG198-256; compile-only478-480; TIME488-500; occupancy502-511; nine cases515-529. The existing CPU reference stays unchanged.
- LSP reference lookup was attempted for the sibling worktree and returned no references despite the two same-file callers; the limitation was reported. The call-site locations above were recovered by exact textual search/read, not inferred from that empty LSP result. Composer must rerun project-root-aware references before any exported-symbol or launch contract change.

**Cross-architecture:** runtime currently opts into this consumer only on exactgfx1201 (`gemm.rs:19177,19193`); gfx1100/gfx1151 retain their separate kernel, quantization and dispatch. The file also compiles undergfx1200 (`KERNEL:115`), but there is no measured gfx1200 speedup. Scope a paired candidate to `__gfx1201__` until a caller explicitly validates/promotes gfx1200; this is an architecture-specific implementation choice, not a public compatibility shim. Non-gfx12 stubs at618-634 remain unchanged. No new dispatch opt-in, quantization, KV/recurrent-state, retained-replay, scheduler-profile or decode-path change is authorized. `ensure_int4_mmq_x` and format/layout contracts remain untouched.

### 5.3 Independently executable composer slices

Shared contract for the batch: preserve baseline128x128/256-thread/19456B launch,64-element sum ownership, per128-K graph and seven kernargs; candidate artifacts use unique source/content identity. No mid-flight project-wide build, tests, lint or formatter. A single integration owner applies production edits after kernel ownership is released; the caller owns GPU ordering and shared validation. Reviewers own the final veto.

**Slice A — kernel pairing, sole kernel writer.** Target `KERNEL:163-179,512-532` only, with nearby local helper declarations if necessary. Implement the two-component helper described in§3, use constant j/j+1 indices and retain exact sf conversion/reuse. Never edit the quantizer, body staging, launch geometry or host flags. Acceptance: actual gfx1201 emitted DAG has the same388 scalar fold operations or a rigorously explained identical-value CSE, fewer net arithmetic packets including copy overhead, zero spills, and no speculative FP contraction. Produce candidate TU/ISA and source hash under`E/vopd/`; do not claim speedup. If compiler output is unchanged or worse after two bounded forms, return the negative result and remove the candidate instead of starting scheduler sweeps.

**Slice B — independent census/legality review, no production edits.** Input contract: baseline and candidate disassemblies plus metadata and exact source hashes fromA. Extend the scratch-only census to count both VOPD halves, packet counts, converters, bank/parity/source-port legality, actual innermost K loop and both ADD instantiations. No source-text test that merely asserts presence of a helper. Include the full_add/base/full_set role proof: all production waves still execute the unchanged32-WMMA compute and legal fold path; no accidental dead branch, dropped output or extra integer bank. For any WS contender, replace this with CFG proof of mutually exclusive producer/consumer paths and all final joins, not a sum over unreachable instructions. Acceptance: reviewers can trace every folded output and all emitted packet operands; an invalid pair or omitted fold vetoes the candidate before timing.

**Slice C — oracle/acceptance preparation, owns only scratch harness variants.** Reuse `ORACLE::launch_iu4`, `cpu_ref`, `run_case`, `time_case`; preserve the CPU DAG. Prepare targeted cases for half-specific sc/zp and d/s, cancellation, nonzero residual Y, M/N tails, K=256 and longK, and repeat identity; for a WS candidate add producer/consumer skew and >2 metadata-parity wraps. Keep launch geometry candidate-specific in the scratch harness, not an unchecked environment override shared with production. Acceptance: a frozen input set and output-hash comparator ready for the caller's sequential GPU run; no independent GPU use while another owner holds ordinal0. No new permanent tests just to demonstrate activity.

A andC can proceed independently;B can finish baseline analysis immediately and inspectA's artifact when available. All file-sharing contracts are frozen above. Kernel integration, GPU execution and final approval are dependency-ordered, not pretend parallelism.

### 5.4 Ordered gates and promotion evidence

1. **Oracle bit identity first.** Use the existing nine cases, CPU exact outputs for small/full cases and existing real-matrix stripe checks; candidate versus incumbent full-buffer bit hashes for real matrices, repeated identical launches, set and nonzero-seeded add. Additional targeted rounding/half-transition cases must assert bits, not tolerance. The runtime-add entry must agree with its specialized counterpart for both add values. FP8 comparison remains diagnostic, not an iu4 bit-identity oracle. Any mismatch stops promotion; no KLD waiver for changed integer/fold semantics.
2. **Metadata/residency.** Record compiler/JIT flags, wave32, raw and allocated VGPR, SGPR, VGPR/SGPR spills, private scratch, dynamic/static LDS, launch dimensions and exact occupancy for all used symbols. VOPD requires zero spills/scratch and baseline3 WG/CU; WS requires<=32768B and>=2; homogeneous16-wave requires its updated20480B launch and a justified residency benefit. `__launch_bounds__` is not occupancy evidence. Reject unsupported named-barrier or dynamic-VGPR instructions on gfx1201.
3. **ISA/DAG/role proof.** RunB's complete census; inspect RN multiply/FMA/add order, conversions and output ownership. Packet savings must survive register allocation and include newly introduced moves/waits. For WS additionally inspect DS release/acquire sequencing, per-role branches, no all-WG barrier in an autonomous role loop, no scratch-indexed accumulator arrays, and correct final drain. Static role separation proves role separation only, not matrix/VALU hardware overlap.
4. **TIME.** Caller runs same-session A/B/A/B at N512 with source/cache identity and exclusive ordinal0 ownership. Report all five rows in `ORACLE:490-499`, all raw medians, bracket drift and temperature/clock context. Winner must improve gate/set>=3% across brackets without>2% reproducible regression in another row; rows within noise are neutral. The task's headline stretch gate is gate/set<=400us. Record both the relative win and whether400 was achieved. Gate/set is one matrix; do not compare it directly with fused two-matrix FP8 gate_up without normalization.
5. **Pins, then end-to-end bench.** After1-4 pass, caller rebuilds the committed candidate once, records hipfire/daemon binary hashes and exact cached TU/HSACO identity. Reuse existing iu4fold gate4 recipes and unchanged expected pins: chunk1 `032ebad84f2c1dd5e2980fa805215ac0`, chunk2 `dc7e53181662271780374f0a85fd7732`, chunk24 `0821993b56021caf4505c1dc4a6b9990` (existing KLD0.063410). Any bit drift is a failure, not a reason to repin. Run bracketed OFF/ON bench over the four established prefill lengths, retaining raw JSON; do not select the single best ON row.
6. **Decode in a separate run, then serve.** Decode must have its own before/after bracket, not piggyback on a long-prefill thermal state. Earlier receipts reproduced28.6-28.8tok/s despite one unrepeated36.23 result; do not use the unrepeated maximum as the baseline or declare decode unaffected from source inspection. Then the established serve battery with measured prefill/decode, all turns, termination, repetition/attractor/empty/retrieval/stream errors. The caller adjudicates any existing baseline regression separately; this experiment must not worsen it or hide it.
7. **Final reviewer veto and cutover.** Only after the runtime gates, remove losing candidate switches/helpers and throwaway production scaffolding; preserve scratch receipts and update this plan with measured outcome. Host/oracle/architecture contracts are intentionally unchanged for VOPD. If a geometry contender instead wins, update every launch/occupancy/store use named above atomically and rerun1-6 for that exact integrated artifact. Do not leave a fallback maze, new scheduler-profile entry, or an unvalidated architecture promotion.

Planning proof already executed: `python3 E/analyze.py` (with E expanded) reproduced315 fold packets, the228 ideal floor,28160B allocation, bank coverage and complete output-coordinate bijection. These analytical checks support this plan only; they are not GPU numerical, synchronization, occupancy or TIME validation. No performance result for a new candidate is claimed.
