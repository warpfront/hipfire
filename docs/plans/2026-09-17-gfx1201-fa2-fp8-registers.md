# gfx1201 FA2 fp8 stage a: register architecture

Date: 2026-09-17. Status: **plan, not implementation or performance approval**.

## 1. Decision

**Keep the four-step QK loops fully unrolled. Stop carrying expanded K-LDS addresses across the entire KT64 loop. Rematerialize those addresses inside one `(dg, sub)` four-WMMA group, using an opaque, group-local byte coordinate with a dependency on the preceding score state.** Keep the fragment conversion, scales, scores, softmax, PV, fills, entry ABI, and 32,768-byte LDS layout unchanged.

The main recoverable live state in the supplied ISA is not four simultaneously decoded K fragments. It is **56 invariant K-load addresses: 30 resident VGPRs plus 26 scratch dwords**. They are computed before the KT64 loop, survive all QK groups, and remain live through PV for the next KT64 iteration. This also explains why scratch *reloads* appear in QK while an important pressure peak is in PV.

[INFERENCE] The counted lifetime reduction supports a **228–240 VGPR direct / approximately 229–240 partial** target, zero spills, rather than an impossibility verdict for f16 WMMA. These are allocation forecasts, not compiled results. The selected candidate has not been built or run. If its scoped metadata or interleaved performance gates fail after the bounded attempt below, abandon this stage-a implementation and return to stage b; do not land the rolled-QK version.

The scheduling dependency is deliberately at the **four-WMMA group boundary**, not between its four `u` steps. Its potential cost is loss of inter-group lookahead and integer address recomputation. It must earn its place in measurement. There is no claim that two-WG capacity alone offsets fp8 decode and header-load overhead.

## 2. Evidence and corrections to the premise

All relative source paths below are under `/home/kaden/ClaudeCode/warpfront/wt-fa2`. Persistent evidence root **`E`** is `/home/kaden/ClaudeCode/warpfront/wt-lloyd/.codeinsight+research/scratch-2026-09-17`; **`R`** is `E/Fa2RegPlan`. `K` means `kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip`; `I` means `E/fa2c/fp8-hip-amdgcn-amd-amdhsa-gfx1201.s`. Line numbers are the inspected snapshots, not the older legs plan's numbers. All `E/...` and `R/...` citations expand to those absolute persistent paths; evidence must not be stored in ephemeral temporary directories.

Inspected identities:

| Artifact | SHA-256 / relationship |
|---|---|
| `I` | `cb3b4bf525e4b2aecb0815450691d041d69daddf22b8ec1bbfd70eb44f2e96f7` |
| `E/fa2_backups/stagea_v1.hip` | `fac698632162432637b0b081d152e8c0b017d7771e8bd0a9028e93d6cd2e3d27` |
| inspected worktree `K` | `005564629c6e289a9b655436b23a6907a211f5ab04800108385b6eb2f61e376b` |
| `E/fa2c/v1fp8.hip` | exactly `#define HIPFIRE_FA2_FP8 1\n` plus `E/fa2_backups/stagea_v1.hip` |
| `E/fa2c/fp8.hip` | the v1 wrapper with the Q8 fill-loop closing brace moved inside its KMODE branch; fixes KMODE=3 compilation without changing KMODE=0 tokens |

`I`'s retained preprocessor output identifies `fp8.hip` and the ROCm clang-23 runtime wrapper. The **current worktree already contains the rejected rolled loops**, not the unrolled v1 scheduling: `K:551,610,669,728` say `#pragma unroll 1`. The source difference from `E/fa2_backups/stagea_v1.hip` is those four pragmas/comments and the brace fix. Use the corrected unrolled `E/fa2c/fp8.hip` as the numerical reference; preserve the brace fix at `K:415-417`. Do not restore the entire old snapshot over the worktree. `R/inspected-kernel.hip` preserves the line-numbered source snapshot used by this plan.

Corrections established from source/ISA:

* Shape is **H24 / Hkv4 / D256, GQA 6:1**, not D128 or 4:1. One WG covers eight positions and 48 query rows, packed into three 16-row compute waves plus one helper wave (`K:7-31,193-232,954`). The *key* tile is 64.
* K is 64×256 bytes = 16 KiB; V is two 8 KiB even/odd planes; total 32 KiB (`K:81-98,204-209`). No unused plane padding exists.
* There is **no K scale table** in v1. K scales are reloaded from the q8_0 block headers and applied **before WMMA**, as `f32(scale) * decoded_e4m3`, rounded to f16 (`K:105-145,558-565` and the three siblings). V uses the same input-rounding discipline (`K:150-166,833-856`). There is no extra f32 post-dot scale vector to remove.
* The supplied direct and partial ISA reports **Occupancy: 5**, not two waves/SIMD (`I:5756,12052`). The 256 architectural VGPR ceiling is real; the explanation “512 total divided by two waves” is not this target's resource model.
* Empty `asm("" : "+v"(dd))` statements do **not** enforce the helper comment's one-live-decoded-pair claim. `I:2191-2194` issues four decode instructions before the first multiply/pack. Their eight f32 outputs coexist. The previous `dd` is overwritten in the source; the next decode has no dependency on it.

Accepted parent measurements, not rerun here:

* v1 KLD: ON 0.048389 versus OFF 0.048659, delta −0.000270; contract ON ≤ OFF + 0.0005. The old 0.1/0.01 device tolerance is superseded. Device-screen gated differences are approximately 0.3391 / 0.0236 versus f16; those are not a candidate-v1 tolerance.
* v1 direct: 256 VGPR / 31 reported spills / 120 B scratch; partial: 256 / 33 / 128 B. Both launch-bounds variants and both optimization levels spilled. These resource fields are also present in `I:12367-12376,12478-12487`.
* Rejected rolled candidate: 243 VGPR, no spill; ON 1448.5 / 1383.4 / 1179.4 / 752.0 versus OFF 1481.1 / 1453.0 / 1343.8 / 1031.1 tok/s at pp512/2048/8192/32768. **The −27% 32K result is a real regression from the changed pipeline until a controlled OFF arm reproduces it.**
* Rejected per-block rewrite: approximately 220 VGPR, but real-data max-abs 2.35. Do not reuse that arithmetic to meet this plan's resource target.

Provenance: `agent://Fa2Fp8Verdict`, `agent://prefill2x-3`, `E/fa2_verdict/on.{O.f32,gate.f32,meta.txt}`, and `E/fa2_bench/{off1,on1}.json`. `R/isa-audit.py` is the reproducible read-only accounting tool; `R/isa-audit.json` records its results and inspected artifact hashes. Preserve evidence in this persistent tree rather than assuming shared filenames remain immutable.

## 3. ISA live-state accounting

### 3.1 Long-lived state

| State | Count per compute lane | Where and why live |
|---|---:|---|
| `Ofr[16]`, f32×8 | **128 VGPR dwords** | In this direct ISA: `v8:119` (112), `v128:135` (8), `v214:221` (8). PV destinations at `I:3941,4049,...,5448,3615`. Every value survives QK; previous KT64 output cannot die before the next softmax/PV update. |
| `sacc0..3`, f32×8 | **32** | `v136:143`, `v144:151`, `v152:159`, `v160:167`; first WMMAs at `I:1024,1194,1364,1534`, final QK WMMA at `I:3574`. All four persist through the dimension groups. The rolled softmax selector retains all four across its subtile loop (`I:3672-3709`), including PV. |
| Expanded K LDS addresses for dc9..15 | **56 = 4 subtiles × 7 dc × 2 words** | **30 resident:** `v4:7` and `v222:247`. **26 spilled:** scratch offsets 0,4,...,100. Prologue `I:208-399`; reloads at `I:2323-2328,2738-2743,2818-2823,3048...`; resident uses at `I:2971-2972,3551-3552`. No definitions of those 30 resident addresses in the tile body: they live across QK, PV, turnover, and the next tile. |
| Persistent state, pointers, fill/V addressing, masks, row coordinates | additional VGPR/SGPR state | Q base `v179:180`, position pointer `v181:182`, scale/header address state near `v194:197`, online state including `v193` and `v255`; K row bases and V/fill indices remain live for subsequent phases. Do not count uniform `do*` predicates or `qok` masks as four float8 vectors: EXEC/SGPR control is visible at `I:984,1154,1324,1494`. |

Why dc9..15? For the early dc values LLVM uses a common LDS base with `ds_load_2addr_b32` immediate offsets (for example `I:1679-1680,2189`). The circular row wrap of later fragments prevents that representation for every lane. LLVM expands both wrapped word addresses separately and hoists them to the prologue. The savings target is these **addresses**, not removing necessary decoded values or changing the swizzle.

Spill metadata is **not** a peak-liveness measurement. In particular, “256+31=287 simultaneously live VGPRs” is not justified: spill counts, scratch frame slots, physical allocation holes, and simultaneous lifetimes differ.

### 3.2 Fragment and scale state actually visible

* An e4m3 K fragment is two VGPR words, eight bytes. Its f16 WMMA A operand is **four**, not eight, VGPRs. Q is another four. The eight-wide accumulator is eight f32 VGPRs.
* `I:2191-2194` decodes to `v123:124`, `v168:169`, `v170:171`, `v121:122`: **eight decoded f32 values**, not a single f32×2 pair.
* `I:2208-2213` has the next Q fragment in `v168:171` while the current Q is `v203:206`; this is useful four-register lookahead. The current K operand is `v199:202`, scaled by `v198`. There are **not four complete f16 K operands** alive at this cut. Restricting the source to one local `kf` has already happened; repeating that source transformation is not a 12-register saving.
* `I:993-995` and its siblings show one block-scale header load/conversion before fragment construction. Scale movement is not just deletion of a hypothetical table or post-dot vector.
* The entire direct QK region `I:969-3574` contains **64 f16 WMMAs and 256 packed fp8-to-f32 conversion instructions**: 4 subtiles × 16 dc; four decodes per fragment. These are static instruction counts, not cycle estimates.

### 3.3 Counted pressure cuts and prediction arithmetic

A read-only host audit was run over the retained assembly. It computed CFG successor-union liveness for physical VGPR halves and scratch dword slots, accounting for the partial writes of `v_fma_mixlo/hi_f16` and d16 loads. It is an **ISA-cut reconstruction**, not LLVM's pre-allocation pressure report: EXEC lane masks and allocation constraints can alter the exact allocator demand. The explicit 56-address inventory above is independent of that reconstruction.

Results:

| Cut | Resident register names live | Scratch dword slots live | Sum of storage dwords |
|---|---:|---:|---:|
| Direct QK peak near `I:2209-2210` | 242 | 29 | **271** |
| Direct PV peak near `I:5221-5222` (also other PV dc) | 255 | 29 | **284** |
| Partial QK peak near `I:8157` | 241 | 31 | **272** |
| Partial PV peak near `I:11399` | 254 | 31 | **285** |

At direct QK `I:2210`, partition the 271 into **128 O + 32 scores + 56 expanded addresses + 55 other/transient dwords**. The resident part of the last category is 52; the other three are scratch offsets 104/108/112. Within those 52, current/next Q consume eight registers, current packed K four, its f32 scale one, and the last decoded component one. The rest is control/address/state/constant storage, not unexplained extra accumulators.

This explains overflow without inventing four coexisting K fragments. It also reveals an important qualification to the premise: the register pool is overcommitted across QK, but **counting QK scratch operations alone does not locate the true peak**. The hoisted addresses tax PV too.

Selected architecture budget, [INFERENCE]:

* Remove all **56** whole-tile address lifetimes.
* Permit at most **eight** expanded addresses for one four-step group, plus **one** opaque group-coordinate VGPR. Keep the four existing K row bases; their deletion is not credited.
* QK reconstructed demand: direct `271 − 56 + 9 = 224`; partial `272 − 56 + 9 = 225`.
* PV reconstructed demand: direct `284 − 56 = 228`; partial `285 − 56 = 229`. The group coordinate and its expanded addresses must be dead here.
* Thus the planning range **228–240 / 229–240**, allowing allocator/tuple/control overhead up to the 240 target. This is not obtained by subtracting 56 from capped metadata 256. A new maximum elsewhere, extra copies for asm constraints, or failed lifetime confinement invalidates the forecast; the emitted metadata decides.

## 4. Occupancy: what the remark proves and what it cannot prove

`I:5746-5756` gives 256 VGPR, fixed LDS 0, Occupancy 5; `I:5717` sets WGP mode. Fixed LDS is zero because the launch supplies LDS dynamically; it is **not evidence that LDS is free**.

The observed 239→6 waves and 243/256→5 waves match the gfx1201 physical-VGPR model:

```
architecturally addressable VGPRs/lane = 256
physical wave32 register-file accounting = 1536
allocation granule = 24 VGPRs
waves/SIMD = floor(1536 / round_up(VGPR, 24))
239 or 240 -> allocation 240 -> 6 waves/SIMD
243 or 256 -> allocation 264 -> 5 waves/SIMD
```

The encoding granule (`VGPRBlocks:31`) is not the physical allocation granule. Upstream LLVM explicitly separates them in `getVGPRAllocGranule`, `getVGPREncodingGranule`, and `getAddressableNumArchVGPRs`. Sources: [AMDGPUBaseInfo.cpp](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp), [AMDGPUTargetParser.cpp](https://github.com/llvm/llvm-project/blob/main/llvm/lib/TargetParser/AMDGPUTargetParser.cpp). These explain the model; the supplied local compiler's matching remarks remain the authoritative artifact for this attempt.

A physical RDNA CU has two SIMD32s; a **WGP has four and two CUs**. Do not mix the older “four SIMD per CU” model into the calculation. In this kernel's WGP mode, four wave32s/WG means one wave per WGP SIMD. Four 32-KiB WGs fit in the WGP's 128-KiB LDS block: **four WG/WGP = two WG/CU capacity**. This needs four waves/SIMD; the remark's five, or the target six, both clear it. Equivalently the 64-KiB-per-CU resource budget holds two 32-KiB WGs. See [LLVM target-parser LDS/SIMD model](https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/TargetParser/AMDGPUTargetParser.h).

`__launch_bounds__(128,1)` is a compiler minimum-residency hint, **not a runtime one-WG limit**. With the same zero-spill object resources and 32 KiB dynamic LDS it does not prevent two-WG/CU capacity. However changing it alone has **measured VGPR delta zero** for v1 and does not fix the 256 architectural cap or spilling. Keep `(128,2)` to document the intended capacity; do not present `(128,1)` as a solution.

The remark alone cannot assert achieved residency or throughput, because it does not include the dynamic launch LDS or workload underfill. Admission requires (a) remark at least four waves/SIMD in WGP mode, preferably six at ≤240 VGPR; (b) verified 32,768 dynamic bytes and no fixed LDS addition; (c) runtime occupancy query with the actual function, 128 threads, and 32,768 bytes, interpreted in the device API's CU/WGP units, demonstrating at least the two-WG/CU-equivalent capacity. An achieved-occupancy claim needs profiling, not this compile-time calculation.

## 5. Ranked options

Deltas below are **live-state estimates against the inspected ISA**, not promises of a recompilation result. Only launch-bounds/rolled results were already measured. Savings at disjoint phases must not be added twice.

| Rank | Option | ISA-counted delta and predicted consequence | v1 bit identity | Pipeline impact / disposition |
|---|---|---|---|---|
| **1** | **Local K-address rematerialization, unrolled four-step fragment/WMMA groups** | Remove 56 whole-tile addresses; replace by ≤9 group-local dwords. **−47 at QK, −56 at PV**. Forecast 228–240 direct, 229–240 partial, no spills. | Yes by address and operation-order construction; device check still mandatory. | Keeps all 4 `u` WMMAs unrolled and intra-group Q lookahead. Adds address ALU work; group dependency may constrain inter-group scheduling. **Chosen.** |
| 2 | Split the 64 keys into two 32-key score lifetimes, with a real lifetime boundary | **−16** scores (`v136:167` 32→16) certainly; up to **−28** of the 56 invariant addresses only if the second half's addresses cannot be hoisted. Combined −44 gives roughly **240/241** at the PV pressure cut before extra loop/control changes. Scores alone leave ~268/269. | Yes if QK per score remains dc0..15 and softmax/PV still sub0,1,2,3. Do not form 32-key joint softmax or accumulate/recombine partial dots. | Four `u` steps can remain unrolled. QK-half0→PV0/1→QK-half1→PV2/3 limits ILP and adds outer control. v1 already reloads Q per subtile, so no automatic 2× Q-traffic penalty relative to v1. Larger rewrite and little headroom; fallback experiment only, not the first patch. |
| 3 | Truly pair-local decode and immediate K-fragment consumption | Eight decoded f32 values → two: **up to −6** at the decode cut, not −24 or −12 from imaginary four-K-fragment staging. A second source scope around `kf` alone gives **0** guaranteed saving: ISA already consumes one K operand at a time. QK-only pair fencing does not remove the PV peak. | Yes if the identical f32-scale/f16-round sequence and WMMA order remain. Do not replace rounded multiplication with a different fp16 multiply. | A dependency through the *next packed input* could enforce pair serialization, unlike the existing overwritten-`dd` fence. It shortens the decode pipeline and may lose issue overlap. Insufficient stand-alone remedy; do not include it in the chosen first candidate. |
| 4 | Native fp8 QK WMMA, stage b | K operand **4→2** VGPR (−2); eliminates eight decoded K f32 temporaries at their cut; converting Q **4→2** per live fragment can save 2–4 more, but Q scales and per-q8-block dot accounting add state. Does **not** inherently delete the 56 address lifetimes or shrink `Ofr`/scores. No total VGPR forecast without its own ISA. | **No.** Native fp8 Q plus block-scale arithmetic changes rounding and reduction structure. v1 KLD certificate cannot transfer. | Real opcode/operand-density opportunity, not a proven fix. Next separate plan if selected stage-a attempt fails; PV stays f16. |
| Reject | Move K block scale to P·V or fold into online-softmax rescale | **0 nonexistent epilogue-vector registers** to reclaim. At most the current 1–2 scalar K-scale registers disappear, while correct per-block partial accumulations would add at least another f32×8 temporary. | **No, generally not even the same real-valued attention** if moved across softmax. K scales differ by key and by 32-dim block. A scalar `alpha` cannot undo changed logits. V scale movement into `prow` changes f16 rounding and depends on output-dimension block. | Not a register scheduling change. Reject for stage a. |
| Reject | Put the “128 B scale table” in LDS | V1 table-register count **0**. A complete K table is **64 keys × 8 blocks × 2 B = 1024 B**, likewise V, total 2048 B. Header-address/scale staging may change a few temporaries but cannot be credited without a design/ISA. | Raw f16 scale copies could preserve arithmetic, but this is a different staging transaction. | 32,768+2,048 exceeds the exact two-WG/CU LDS budget. Even +128 B misses four-WG/WGP capacity (at most three under WGP pooling). Mailbox bytes are occupied by V during compute. One-block tables require phased fill/barriers/plane reuse, not free space. Reject here. |
| Reject | Alias `sacc` storage with f16 K/Q fragment registers | **0 valid saving** while both operands and accumulator are live at the same WMMA. f32 scores remain needed for later dc and softmax. Packing scores to f16 would save 16 but changes the contract. | Aliasing live values is incorrect; lowering score precision is not bit-exact. | Lifetime splitting is option 2, not a C++ union/register trick. |
| Reject | `(128,1)` with 32 KiB LDS only | **Measured delta 0**; same 256 VGPR/spills. Two-WG/CU capacity is not prohibited by this bound. | Arithmetic unchanged. | Cannot repair spilling. |
| Reject | Roll the four QK `u` loops | Measured **256 with spills →243 without spills**, five waves/SIMD; still above the preferred 240 target. | Intended same arithmetic; bit check is still owed for any new object. | **Measured severe pipeline regression**, including 752 vs 1031 tok/s at 32K. Do not repeat or land as a fallback. |

For the rejected K-scale move, the invariant is `score(q,k) = WMMA_dc0..15(Q16, RNE16(sf[k,dc/2] * decode(K8)))`. Even if a scale were uniform over all dimensions of a key, multiplying the logit before exponentiation is not equivalent to multiplying that key's V afterward. The actual eight scales per key make the attempted move still less applicable. Moving V scaling into P also changes `RNE16(V*scale)` before multiplication into `RNE16(P*scale)`, so its KLD must be recertified even where the real-valued algebra factors.

## 6. Frozen implementation contract and exact edit regions

### 6.1 Ownership and interfaces

One composer owns **only `K`** for the production patch. No Rust ABI, symbol, flags, dispatch predicates, cache formats, scratch sizing, launch LDS, or graph ownership change. The internal helper is not exported; no C++ language server is available in this session. Its exact callers are the four lines `K:563,622,681,740`; update all four in the same slice.

Replace the address-input portion of the existing helper with this internal interface (name unchanged; no compatibility overload):

```
half8_t fa2_kfrag_fp8(const uint32_t* Kdw8,
                     unsigned key_row_dw, unsigned group_byte,
                     int u, float sf);
```

Contract:

* `key_row_dw = key * 64`; key remains `ml`, `16+ml`, `32+ml`, or `48+ml`.
* `group_byte` has the runtime value `dg*64 + kg*8 + 4*fa2_swiz(ml)`, but is **opaque to loop-invariant address expansion** after the boundary below.
* `u` is compile-time-unrolled 0,1,2,3. `x = (group_byte + 16*u) & 255`.
* The two reads are `Kdw8[key_row_dw + (x >> 2)]` and `Kdw8[key_row_dw + (((x >> 2)+1)&63)]`. Do not change the second read to an unconditionally contiguous 64-bit load: row wrapping is real.
* Decode and scale code `K:115-145` remains numerically identical. Retain KMODE=3's pure-conversion behavior. Do not add new per-pair serialization in this attempt.
* This signature changes **address construction only**. The scale header still uses the original `dc = dg*4+u` and `dc>>1`, the original `gkok` clamp, and the original scale conversion.

At the start of **each executed fp8 `(dg,sub)` block**, before its unrolled `u` loop, create a local `group_byte` and an empty tied-register asm dependency using the existing inline-asm idiom:

```
unsigned group_byte = dg*64u + kg*8u + 4u*fa2_swiz(ml);
asm volatile("" : "+v"(group_byte)
                 : "v"(sacc0), "v"(sacc1), "v"(sacc2), "v"(sacc3));
```

This is a compiler register-lifetime boundary, **not a workgroup synchronization barrier**. All score inputs already exist. Their values are not changed, rounded, inspected numerically, or written to memory. Unlike the current `dd` barrier, the **output is consumed** to form the next group's addresses. The inputs prevent the next boundary from moving above the preceding live score updates, including cases where a later subtile is disabled. The four `u` steps inside this boundary remain freely schedulable and fully unrolled; no individual WMMA-result dependency is added between them beyond their existing accumulator chain.

Do not add `"memory"` clobbers at each `u`, spill scores through LDS, make arrays volatile, or add `__syncthreads()` inside `if(compute)`/`if(do*)`. The existing group memory barrier at `K:768` and PV barrier at `K:873` remain. Empty asm does not magically guarantee a VGPR count: reject the candidate if the resulting ISA still carries the 56-address family across the tile, keeps several groups' expanded addresses alive at once, or inserts expensive copies/extra waits. This is the specific mechanism being tested, not an approved unmeasured optimization.

### 6.2 Regions to modify, preserve, and inspect

| Region | Required treatment |
|---|---|
| `K:105-114`, `fa2_kfrag_fp8` signature/address setup | Replace key/dc/kg-derived address expansion with the frozen interface above. Preserve wrap and K-plane indices. |
| `K:115-145` | Preserve conversion/scale operation order. Correct only comments claiming a single live decoded pair; existing ISA disproves that claim. |
| `K:538-566,597-625,656-684,715-743` | Add the group-coordinate lifetime boundary; use the new helper arguments; restore `#pragma unroll` at 551/610/669/728; remove rolled-loop claims. Keep qok/do/gkok and scale-header semantics. |
| `K:490-508,768-778` | Keep four score vectors, dg order, group boundary, and ordered softmax selection. No half-tile rewrite in this candidate. |
| `K:365-477`, especially 415-417 | No fill/layout change; keep the KMODE brace fix. |
| `K:150-166,786-874` | V decode, masks, max/sum chains, exp, f16 P packing, alpha/O updates, and PV unchanged. |
| `K:879,882-934` | Tile turnover and direct/partial completion unchanged. |
| `K:939,1049,1078` | Keep `(128,2)` on fp8 bodies. Base f16 entries stay `(128,1)`. |
| `K:999-1036,1120-1162` | Q preconvert and merge untouched. |

Line numbers must be re-grounded after any concurrent change. The acceptance is structural, not a blind patch against old numbers.

## 7. State ownership and transitions

| Transition | State before | Action / ownership | State after and invariant |
|---|---|---|---|
| Host Q preconvert → body | Caller Q, K/V, positions immutable; Gpu owns q16 scratch | Existing same-stream preconvert produces f16 Q; host scratch growth invalidates captured graphs first | Body gets identical q16 pointer/layout. No new allocation, cache write, retained state, or graph capture change. |
| WG init / bounds | No live tile data | Compute lanes own 16 O fragments and m/l; all waves participate in mailbox barrier (`K:215-281`) | O=0, m=−∞, l=0; bounds uniformized; transient mailbox can be overwritten by V fill. |
| KT64 fill | Prior tile readers have finished at turnover | All four waves cooperatively write exactly their current K/V bytes; existing barrier `K:477` publishes planes | K/V LDS immutable until all QK/PV consumers finish. q8 headers remain in immutable global cache. No scale table. |
| Start `(dg,sub)` QK group | O/m/l unchanged; four saccs contain their preceding dc reductions | Compute wave establishes opaque group coordinate from lane/key geometry and current scores | Only this group's expanded addresses may become live. No inter-wave or helper-wave participation required. |
| Each unrolled `u` | sacc for this subtile is the exact v1 prefix sum | Read identical Q, K byte pair and scale; decode, multiply/round to f16, execute identical f16 WMMA | Only that sacc advances by dc; K/Q/decode temps die as consumed, with useful within-group lookahead allowed. Other saccs and online state unchanged. |
| Group boundary | Four dc updates complete for that subtile | Last use of this group's coordinate/expanded addresses | These addresses must die, not carry through later groups or PV; score values survive unchanged. |
| QK → softmax sub0..3 | All four complete score vectors exist | Apply original causal masking and ordered f32 max/sum/exp; preserve kg shuffle | m/l exactly v1 after each subtile; packed f16 `prow` identical. K-address temporaries absent. |
| Rescale / PV | O is prior prefix output, alpha/prow belong to this query | Original O rescale and dc0..15 PV chain, with original V scale/decode | O and l are the exact v1 unnormalized prefix state. O storage remains 128 dwords. |
| Tile turnover | Last PV read completed for each compute wave | Existing unconditional WG barrier `K:879`; helper participates | Planes may be overwritten for next KT64. No barrier is inserted on a divergent path. |
| Direct / partial completion | Final O,m,l owned by compute lane | Direct divides by own l and writes only qok rows; partial writes unchanged `[m,l,O[256]]` stride258 | One writer per output component. Unused rows untouched; q16/Q/K/V/positions never modified. Merge behavior/partitioning unchanged. |
| Output gate | Kernel output is pre-gate | Existing engine epilogue applies sigmoid(gate), possibly fused with rotation | Gate remains post-attention. No attempt to hide numerical differences by moving the gate. |

The kernel is replay-idempotent because it still only reads inputs and overwrites designated outputs. Nothing in this patch owns KV append/rollback, recurrent state, slot binding, or retained replay lifetime. Those transactions remain wholly outside the change.

## 8. Exact call-site and architecture inventory

No caller changes are required. These are the boundaries the composer/reviewer must keep frozen:

* Kernel entries → shared body: direct `K:962-963` → `fa2_gqa_body<false>`; fwht3 `K:1072-1073` → the same; partial `K:1116-1117` → `<true>`. Four `fa2_kfrag_fp8` call sites listed in §6 affect both instantiations.
* `crates/rdna-compute/src/attention.rs:3569`, `Gpu::attention_q8_0_fa2_gqa_gfx1201`: symbol/source choice 3630-3650; q16 ownership 3652-3664; preconvert 3706-3714; body grid/block/dynamic LDS 3715-3722.
* Q8 production ingress `attention.rs:3282-3285` calls that launcher. `fa2_fp8_screen.rs:347-359` calls it directly. Synthetic oracle call: `crates/hipfire-runtime/examples/tmp_fa2_fp8_oracle.rs:154-155`.
* fwht3 launcher's symbol/source selection `attention.rs:3837-3843`, dynamic LDS/body 3921 onward; dispatch `crates/hipfire-dispatch/src/families/attention.rs:1698-1722` admits exact gfx1201, H24/KV4/D256, batch 64..512 divisible by16, context64..32768, no tree bias, Q8 V.
* `Gpu::attention_q8_0_fa2_gqa_split_gfx1201_bench`, `attention.rs:4401`: partial/merge symbol choices 4461-4481; q16 preparation 4483-4507; preconvert 4537-4545; partial launch 4546-4553; merge launch 4586. Still bench-only; do not add production split dispatch or lift context limits.
* Source strings `crates/rdna-compute/src/kernels.rs:5748-5772`: `ATTENTION_Q8_0_FA2_GQA_FP8_GFX1201_SRC`, `ATTENTION_Q8_0_FA2_GQA_FWHT3K_FP8_GFX1201_SRC`, partial and merge source aliases. Their defines/include contract is unchanged. Do not rename/export a second public route.
* Actual gate ownership moved since the older plan: `crates/hipfire-arch-qwen35/src/qwen35/prefill.rs:7060-7114`, `batch_chunk_full_attn_output_projection`. Ordinary path gates `fa_attn_out_batch` at **7114**; fused paths at 7079/7099 leave that f32 buffer pre-gate and produce rotated f16 separately. The screen uses `DflashFusionCtx::Off` at `fa2_fp8_screen.rs:283`. Its **direct `--kernel-out` output is always pre-gate**, independent of engine epilogue variation.

Cross-architecture scope:

* Only the existing fp8 compile branch on gfx1200/1201 changes; `K:62-66` continues rejecting other architectures. Runtime production routing remains its existing exact-gfx1201 route; do not broaden it just because compilation accepts gfx1200.
* Flag OFF/f16 bodies, gfx11 source objects (gfx1100/gfx1151), gfx10 fallback, non-admitted shapes, windows/slots/tree masks, and decode remain unchanged. Compare corresponding `.text` sections before/after under identical compiler flags; no byte-identity claim has been measured for this proposal yet.
* KMODE=3 gets identical **address-only** semantics with the same pure K conversion. The Q8 KLD certificate is not a fwht3 quality certificate. Preserve the existing guard/brace behavior and verify a same-variant differential if that fp8 entry is exposed; do not claim new fwht3 quality or speed from the Q8 screen.
* Non-goals: native fp8 QK, fp8 P/PV, cache-format changes, new scale staging, tensor shape generalization, tuning fast-math, partial partition policy, >32K routing, and host/replay/refcount changes.

Tool limitation: rust-analyzer from the active checkout returned no references for the external-worktree launcher although direct call sites above exist. That empty result is not a proof of no callers; the explicit inventory uses source matches. A composer changing any exported Rust API must re-run references in the correct workspace first. This plan does not require such a change.

## 9. Independently executable composer slices

Freeze before dispatch: the helper ABI/address formula in §6, unchanged exported entry ABI, same 32,768-byte launch LDS, and v1 reference identity in §2. All concurrent owners **skip builds/tests/lint/formatters**; Main owns validation after the patch is integrated. No agent writes wt-lloyd's other dirty files or hipfire-beta.

### R1 — production kernel owner

Target: only the listed regions of `wt-fa2/K`; no Rust, harness, or other kernel ownership.

Change: restore the four unrolled `u` pragmas; implement the group-coordinate boundary and new helper address inputs; preserve every arithmetic operation and state transition; remove inaccurate rolled-loop/single-pair-live comments. No alternate implementations or compatibility helper remain.

Acceptance artifact: candidate source hash and exact changed symbols/regions; an ISA-inspection checklist asserting the intended lifetime mechanism, explicitly marked unmeasured until Main compiles. Source preserves all contracts in §§6–8. R1 does not tune PV or scale placement if its forecast fails.

### R2 — reference and differential preparation owner, independent of R1

Target: **persistent `R` artifact directory only**, using the retained corrected v1 source and existing screen/oracle. No edits to R1's file, production Rust, or any unrelated workspace. Disposable harnesses may be removed after admission; reference evidence and receipts stay persistent.

Change: preserve the v1 source, supplied ISA/object, existing verdict output/meta, and a manifest of hashes, compiler flags, model hash, tokens/layer, resolved flags, and GPU identity. Prepare the two-arm screen invocation and byte-comparison commands below. Ensure reference and candidate process/kernel caches are isolated; the candidate must not bind a stale module under the same symbol.

Acceptance: runnable protocol consumes candidate source/binary only after R1 completes; baseline provenance refers to unrolled corrected v1, **not the current rolled worktree**. Input equality is part of the protocol, not assumed from matching CLI arguments. If existing artifacts lack input provenance, prepare a disposable harness copy that records Q/K/V/positions hashes at screen lines 291-331, or replays both kernel objects against the same tapped buffers. No permanent test/API is added merely for this task.

### R3 — static review/preflight owner, independent of R1/R2 preparation

Target: read-only source/ISA analysis; no shared-file edits or builds.

Change: independently check the 56-address inventory and finite wrap enumeration, asm output dependency, no divergent WG barriers, unchanged scale/WMMA/softmax ordering, and the CU/WGP unit conversion. Prepare claim-scoped metadata and unchanged-architecture comparisons. Do not infer performance from lower VGPRs.

Acceptance: explicit pass/veto checklist against the candidate when available; distinguish forecast, parent measurements, and newly obtained evidence. Final reviewer veto remains independent of the implementation owner.

Join: Main compiles the combined candidate once with the runtime flags, then runs §§10–11 on an assigned isolated GPU. These slices can prepare concurrently; only candidate-dependent validation is serialized. If R1 is rejected, the reference/artifact work remains useful for stage b rather than being coupled to an unfinished production API.

## 10. Correctness and metadata gates

### G0 — reference, source and build provenance

Use `crates/rdna-compute/src/compiler.rs:1006-1009` runtime flags (`--genco --offload-arch=gfx1201 -O3 --no-offload-compress`) with the same compiler/options as the admitted runtime; append resource remarks and save temps for inspection, not a new optimization mode. Match the actually cached source to the chosen snapshot. Both v1 and candidate must have distinct isolated cache/build provenance; changing a file does not prove an embedded `include_str!` or cached HSACO changed.

Preserve corrected-v1 reference output before integrating the kernel patch, or use a separate external lab wrapper/object. Do not temporarily overwrite R1's kernel to switch arms. Distinguish reference baseline regeneration from the already accepted KLD result.

### G1 — static resources, before GPU timing

For direct and partial entries separately (and the fwht3 entry if included):

1. `.vgpr_spill_count == 0`, `.sgpr_spill_count == 0`, `.private_segment_fixed_size == 0`, no `scratch_load/store` in body ISA.
2. **VGPR ≤240 target/admission budget** for this attempt. 241..256 is addressable and can have enough two-WG capacity but misses the promised headroom; do not call it the target achieved. A relaxation requires Main's explicit decision, not quiet acceptance.
3. Fixed LDS remains0; launch dynamic LDS is32768; wave32 and128-thread WG preserved; no extra static/shared scale array. Capacity calculation/query as §4, with explicit units; expected remark6 waves/SIMD at the target (at least4 required by four-WG/WGP residency).
4. Four `u` WMMAs remain unrolled. QK has64 f16 WMMAs and the same256 packed conversions in the straightforward unrolled body. No new inner loop/backedge or per-step global memory fence. Compiler may rearrange independent instructions, not reduce the numerical contract.
5. The 56 prologue address values are gone as whole-tile lifetimes. At most one group's expanded K addresses plus its seed survive between the four unrolled steps; **none survive into PV**. Do not require particular physical register numbers in the new ISA.
6. Check PV and completion too: removing QK reloads alone is insufficient. Confirm no new PV or output-store spills and no changed FP mode/opcode semantics.
7. Base gfx1201 f16 and gfx1100/gfx1151 corresponding `.text` unchanged under identical compilation. Preconvert and merge contracts unchanged. This is scoped compilation/artifact comparison, not a project-wide suite.

### G2 — bit identity to v1, not tolerance to f16

Existing real-data route: `crates/saddle-lab/examples/fa2_fp8_screen.rs --kernel-out PREFIX`; parser85-135, actual launch335-359, dumps360-381. It writes `PREFIX.O.f32` (pre-gate), `PREFIX.gate.f32`, and `PREFIX.meta.txt`; output has384×24×256 float32 = **9,437,184 bytes**. `--qstride` samples CPU math only, **not** the kernel output dump.

Canonical shape is layer35,4224 tokens,384-row final chunk. For each isolated reference/candidate executable, use the same model and device and these semantic settings:

```
R=/home/kaden/ClaudeCode/warpfront/wt-lloyd/.codeinsight+research/scratch-2026-09-17/Fa2RegPlan
ARM=v1  # repeat with ARM=candidate and its isolated executable/cache
env HIPFIRE_GFX12_FA2_FP8=1 HIPFIRE_GFX12_FA2_PREFILL=1 HIPFIRE_LLOYD_GFX12=1 \
  fa2_fp8_screen --model /home/kaden/.hipfire/models/qwen3.8-27b.mq4-xt \
  --layer 35 --tokens 4224 --qstride 4 --kernel-out "$R/$ARM"
```

`ARM` is `v1` or `candidate`; use each arm's explicitly selected `fa2_fp8_screen` executable. Device/HOME/kernel cache are explicitly assigned and isolated by Main, not inherited accidentally. Build features must include `deltanet,arch-qwen35`; the example itself forces graph0, q8 KV, prompt normalization0. Keep other unrelated feature flags identical. Existing `E/fa2_verdict/on.*` is a useful archived v1 pin, but do not silently substitute it for a controlled same-input run if the surrounding engine has changed.

**Same-input requirement:** two full prefills can generate different later-layer Q/K/V if some unrelated engine binary or flag differs. Record and compare Q, raw K, raw V, positions, gate and model/flag hashes, or have a disposable screen harness call v1 and candidate modules on the same tapped buffers. Gate equality alone is not proof of Q/K/V equality. The ordinary `--kernel-out` interface does not yet record input hashes; R2 must cover that gap without a production API change. In a two-module harness avoid the host's symbol-keyed function cache collision: bind module-specific function handles or rename only the lab wrapper's reference entry/preconvert symbols. No production symbol aliases.

Once input identity and finite values are established:

```
cmp "$R/v1.O.f32" "$R/candidate.O.f32"
cmp "$R/v1.gate.f32" "$R/candidate.gate.f32"
```

Both exit0; count mismatching float **bits** and report the first index if not. Compare metadata fields for layer/tokens/chunk/H/Hkv/D/flag. **Zero differing output bits** is the gate; max-abs0.34 relative to f16 is irrelevant here. Check pre-gate O first so sigmoid cannot conceal an error. With identical gate and identical downstream gate implementation, pre-gate bit identity implies post-gate identity; optionally compare separately gated dumps with the same implementation. Do not apply the gate twice to an engine buffer.

Required boundary smoke differential, same v1 inputs: the existing seven synthetic shape pairs at `crates/hipfire-runtime/examples/tmp_fa2_fp8_oracle.rs:70-78` are `(4,256),(8,512),(32,2048),(64,4096),(128,8192),(256,16384),(512,32768)`. Preserve its input-scale settings; it compares candidates to v1, not renewed fp8-vs-f16 tolerance. Exercise a truncated causal key tile (`seq_len` around15/16/17,31/32/33,63/64/65) and batch tails admitted by the direct lab launcher. The Q/K/V input bytes stay immutable. For partial, compare v1 versus candidate at identical split counts including1 and8: `[m,l,O]` records and merged outputs. **Do not require partial-vs-direct bit identity**, since they have a different reduction partition already.

Host address proof already run during planning: exhaustive key0..63, dc0..15, kg0..1 (**2048 fragment pairs**) checked

```
old = (4*(key*64 + (x>>2)), 4*(key*64 + (((x>>2)+1)&63)))
new = (key*256 + x, key*256 + ((x+4)&255))
x = (dc*16 + kg*8 + 4*fa2_swiz(key&15)) & 255
```

All matched, including row wrapping. This proves only address algebra, not generated scheduling, numeric parity, or throughput. The composer should retain a tiny regression only if its boundary/identity behavior cannot otherwise be protected; no source-text or metadata-default unit tests.

### G3 — numerical-certification decision

The proposed address-only schedule has identical f16 operands and each output's ordered WMMA/softmax/PV graph by construction. With G2 passing and unchanged arithmetic ISA semantics, the v1 KLD certification can transfer to this optimization. A finite real-data memcmp alone is not a universal mathematical proof; it supplements the operation/address proof.

Any intentional change to scale placement, f16 round points, fp8 format, Q conversion, score accumulation order, max/sum/exp order, partial partition, or P packing **invalidates transfer**. First fix unintended differences. A genuinely new arithmetic candidate owes a fresh matched24-chunk WT2 ON/OFF evaluation with ON≤OFF+0.0005; do not use stale global pins or the superseded screen tolerance to excuse it. Stage b necessarily takes this branch.

## 11. Performance gates, expectation and abandonment

No GPU, benchmark, compilation, formatter, linter, or project-wide tests were run for this planning assignment. Evidence newly produced here is the ISA accounting, source relationships and exhaustive host address proof.

[INFERENCE] Working expectation for the chosen attempt: recover the rolled implementation's large long-context loss, then seek a **measurable modest improvement over f16 OFF**, not merely improvement over752 tok/s. A useful success band is roughly +3–8% at pp8192/32768 with no material short-row regression. The older +5–10% occupancy-only forecast is not a result and is not guaranteed: decode, global scale loads, and new address ALU work still exist. At the supplied1031 tok/s OFF32K reference, +3–8% would be roughly1062–1114 tok/s, **an unmeasured hypothesis**, not an expected delivered number.

After G1/G2, Main runs interleaved OFF/ON/OFF/ON (reverse order once if needed) on the **same frozen candidate binary state**, one assigned GPU, identical graphs-ON/model/q8/other flags and isolated daemon/cache lifecycle. Use `--runs 3 --warmups 1`; report per-row raw samples/median, device/temperature/clock state, flags, daemon executable hash and source/HSACO hashes. Do not compare a candidate built after sibling edits against an earlier OFF binary.

Main's rule is binding: session spread is approximately±0.5%; **a prefill delta larger than about1.5% is not noise**. An adverse swing remains attributed to the change until the interleaved OFF arm also reproduces it. Admission: no >1.5% regression on any pp row, and a reproducible >1.5% gain at32K (preferably≥3%) rather than a tie. If only a tie is measured, stage a has not justified this complexity. Do not average away an individual long-context regression. Decode must remain at the accepted approximately36.4+ floor in matched thermal state; a28–29 tok/s decode row immediately after32K is the documented junction-throttle case: **rerun it, do not average it in and do not use it to excuse a prefill delta**.

**Bounded abandon criterion:** one address-rematerialization candidate plus at most one correction to the same lifetime mechanism. Abandon this selected stage-a approach if it cannot meet zero spill/≤240/32-KiB capacity, changes any output bit without an understood fix, loses the unrolled intra-group pipeline in ISA, or fails the interleaved performance gate. Do not serially combine scale movement, precision changes, rolled loops and unrelated tilings under the same certification. Option2 needs an explicit new experiment decision; otherwise return to the stage-b plan and fresh KLD. Native fp8 QK halves the K operand size but is **not guaranteed** to solve the address/PV pressure or to preserve v1 numerics.

The honest present conclusion is **not “f16 stage a is proven impossible”**: the ISA exposes a specific47–56-dword avoidable lifetime tax. It is also **not “stage a is solved”**: register forecasts, emitted scheduling and end-to-end performance remain to be measured. Reviewers have the final veto after the claim-scoped gates. On acceptance, the integration owner records actual resource/bit-identity/bench evidence in the plan/changelog, removes disposable comparison scaffolding, and leaves unrelated dirty work untouched.

### R1 verdict, 2026-09-18 — ABANDONED at G1 (agent Fa2RegR1)

Candidate: wt-fa2 branch `gfx12-fa2-fp8-a`, commit `eddc0c571`
(`kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip` sha256
`92e90dfd105a4937bb7c8c6cda923987ebe6c2136986da4b182a3bac319c0d4e`).
Changed regions only: `fa2_kfrag_fp8` signature/address setup (ex-105-146,
now `(Kdw8, key_row_dw, group_byte, u, sf)` with `x = (group_byte + 16u) & 255`)
plus the four fp8 `(dg,sub)` blocks (group_byte seed + score-tied asm boundary,
restored `#pragma unroll`, new call args). Fill brace fix, scales, softmax, PV,
ABI, launch bounds `(128,2)`, and 32,768 B dynamic LDS untouched.
Evidence root: `E/Fa2RegR1/` (compile logs, ISA, audit scripts, f16 diff).

G1 metadata (runtime flags `--genco --offload-arch=gfx1201 -O3
--no-offload-compress` + `-Rpass-analysis=kernel-resource-usage`, direct hipcc):

| Entry | VGPR | SGPR | spill V/S | scratch B/lane | waves/SIMD | fixed LDS |
|---|---|---|---|---|---|---|
| direct fp8 (KMODE=0) | 248 | 58 | 0/0 | 0 | 5 | 0 |
| partial fp8 (KMODE=0) | 244 | 61 | 0/0 | 0 | 5 | 0 |
| fwht3k fp8 (KMODE=3) | 239 | — | 0/0 | 0 | 6 | 0 |
| partial (KMODE=3) | 240 | — | 0/0 | 0 | 6 | 0 |
| preconvert / merge | 13 / 17 | 6 / 10 | 0/0 | 0 | 16 | 0 |

v1 baseline was 256/31 spills/120 B direct, 256/33/128 B partial, 5 waves.
ISA lifetime check (same CFG-reconstruction method as `R/isa-audit.py`):
80 WMMAs per entry (64 QK + 16 PV), 0 `scratch_load/store` file-wide,
address math rematerialized inline per unrolled group, no prologue-hoisted
56-address family. Reconstructed peaks: direct QK 228 (was 271), direct max 239
in PV (was 284); partial QK 226 (was 272), max 239 (was 285); scratch live in
0 rows. Base f16 gfx1201 `.text` byte-identical candidate-vs-HEAD for all four
functions (only the compile-UUID stamp differs); gfx11 file untouched.

Failing gate: **G1.2, VGPR ≤240 — direct 248 (miss by 8), partial 244 (miss by 4),
occupancy stays 5 waves.** Zero-spill and 32-KiB capacity pass. A no-barrier
diagnostic copy (`E/Fa2RegR1/exp_nobarrier.hip`, score inputs stripped) emits
byte-identical 248/244: the score-tied boundary costs 0 VGPR. The KMODE=3 module
at 239/239/240 + 6 waves proves the address mechanism met its budget; the
remaining ~8 VGPR in KMODE=0 are the per-fragment Q8 scale-multiply transients
inherent to the v1 arithmetic contract. Removing them is scale movement under a
new certification, explicitly out of scope — so no same-mechanism correction can
credibly close the gap and the one-correction budget was deliberately unspent.
G2 and §11 did not run (stop at first failure). Return to the stage-b plan and
fresh KLD; do not land this candidate. Branch left with the candidate commit.
