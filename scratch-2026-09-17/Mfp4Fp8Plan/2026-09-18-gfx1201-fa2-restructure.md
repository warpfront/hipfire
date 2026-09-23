# gfx1201 FA2: exact fragment-major LDS restructure

## 1. Per-phase packet attribution — what the retained ISA actually says

**Plan only; no GPU or compiler invocation, no measured candidate speedup or numerical admission.** Source root is `/home/kaden/ClaudeCode/warpfront/wt-lloyd`, HEAD **`af16ec4bd2714bf75704d5a3bf6c7defa08ca3b9`**. All file:line references below expand under that root. Abbreviations: `K = kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip`; `E = .codeinsight+research/scratch-2026-09-17`; `S = E/fa3ping/base_f16.s`; `G = E/fa3ping/base_gl.s`; `R = E/Fa2Plan`. The census was compiled at8f2f29480, but **K's bytes at this HEAD equal `E/fa3ping/base_f16.hip`**: SHA256 `6c3e7c4974545c4ae0aaa5ffc57e23612acf050fdd150347691d34ca6ed41d8e`. ISA hashes and exact instruction-to-source mapping are retained in `R/host_audit.json` and `R/attribution.tsv`.

### 1.1 Reproducing, and correcting the interpretation of, the census

`R/host_audit.py` reproduced **1941 static packets,267 VOPD packets,80 static WMMA**, with the original source-line phase attribution from `E/fa3ping/census_base.md:8-29`. One VOPD counts as one packet, not two operations. These are **static inventories**, not elapsed cycles or a dynamic trace. SALU and VALU columns in the original census are not an additive partition: VOPD contributes two VALU operations, waits overlap scalar categories, and `global_inv` is counted as GMEM although it is not a payload load.

The following replacement partition is additive. EXP is included in VALU; shuffles in LDS; barriers/invalidation in “other”. Line attribution is retained even when scheduling moves an instruction into another phase's physical region.

| Source-attributed phase | packets | WMMA | global loads | LDS incl. shuffle | VALU packets | waits | delay hints | scalar/control/other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Entry guards |58|0|0|0|3|1|3|51|
| Prologue/mailbox |178|0|1|12|114|24|13|14|
| K fill |359|0|8|8|216|40|10|77|
| V fill |209|0|8|8|109|29|6|49|
| QK |624|64|16|160|156|91|39|98|
| Softmax, one subtile |260|0|1|2|171|26|19|41|
| PV, one subtile |123|16|0|64|22|16|5|0|
| Epilogue |130|0|0|0|84|3|7|36|

The “other” epilogue includes32 global stores. The K-fill359 includes hoisted setup/O initialization and turnover instructions; its contiguous fill code is only236 static instructions (`S:334-569`), with four eight-iteration loops. V's contiguous fill is208 static instructions (`S:570-777`), with two eight-iteration loops. Thus **359+209 must not be charged once as dynamic fill cost**.

### 1.2 Exactly where QK's624 packets go

| Additive QK component | packets | Cause / retained ISA evidence |
|---|---:|---|
| WMMA |64|Four independent subtile score chains, each dc0..15. `K:345-456`; examples `S:829-837,856-864`.|
| K LDS reads |160|**96 `ds_load_2addr_b32` +64 `ds_load_b32`**. They deliver256 dwords =64 half8 operands. Twelve dc per subtile use two paired loads; the last four use four individual wrapped-address loads. `S:815-822,995-1002,1160-1178`; address/source map in R.|
| Q global reads |16|Already ideal aligned `global_load_b128`, reused by all four subtiles. `S:790,797,805,812`; `K:346-360`.|
| K address additions |72|`v_add_nc_u32`: materialize large subtile/group offsets for paired LDS reads; the paired instruction's small offsets cannot directly span the full plane. `S:839-854,883-898,927-942,1612-1618`. Additional wrapped address seeds are hoisted into the prologue (`S:175-239`).|
| Moves / zeros / register placement |69|27 scalar-VGPR moves +42 dual-move packets. Includes Q zero initialization for qok guards, and cold `doN=false` score initialization. `S:784-814,866-876,910-920,954-964`. Not69 half unpack conversions.|
| Wait instructions |91|60 LDS waits,4 combined load/LDS waits,27 dependency waits. Counted packets, not measured stall duration.|
| Delay hints |39|`s_delay_alu`; cannot count each as a known cycle cost.|
| Vector control |15|10 `v_cmp_ne_u32`,5 `v_cndmask`; subtile liveness/control, not numerical K scaling.|
| Scalar control |98|Exec save/restore, qok guards, branches and subtile predicates.|
| **Total** |**624**|**560 non-WMMA**,8.75 support packets/WMMA in this static attribution.|

**No QK q8 scale multiplication, f16 conversion or `ds_bpermute` exists.** Scale/conversion is in fill. `fa2_dw_t`'s half selection/half8 construction is bit placement already eliminated by the compiler; the expensive residue is **fragment-address/layout lowering and control**, not arithmetic unpacking. The K swizzle prevents a single aligned half8 read, especially at wrap, and sustains extra live address registers. Do not optimize nonexistent QK scale or shuffle work.

The full-valid/full-query QK path reconstructed from the branch targets is **562 packets,64 WMMA,498 support** (`S:784-1222`, with live outlined blocks `:1612-1785`). This excludes its four setup/exit-to-subtile packets at `S:1223-1226`. The static624 includes cold zero paths and some hoisted/misattributed instructions. It is not an exact dynamic624-per-tile measurement.

### 1.3 PV, softmax and fill attribution

**PV:**123 static packets/subtile =16 WMMA +64 `ds_load_b32` +16 LDS waits +5 delay hints +22 VALU packets. Those22 include5 exponentials and5 mask comparisons attributed here by inlining, plus10 address operations,1 mask and1 move. Four LDS reads build each half8 V operand (`S:1378-1381,1404-1407,1505-1553`); **zero shuffles, zero V scaling/conversion in PV**. Dynamic KT64 has64 PV WMMA and256 V-LDS reads.

**Softmax:** physical subtile work includes9 EXP, not4 or8:8 P exponentials +1 alpha (`S:1345-1375`; census `:21-22`). Source-attributed softmax has260 packets, of which66 are dual-multiply-led packets,35 e64 selects,7 e32 selects,8 half conversions,8 serial sum additions,5 max3 instructions,2 shuffles,1 denominator FMA, plus masking/control/address/waits. Runtime selection of one of four score vectors accounts for24 selects/subtile (`S:1231-1293`). The128 O multiplies/subtile are **already61 dual-mul pairs +1 multiply paired with address work +5 single multiplies**,67 packets carrying O work (`S:1362-1502`). Perfect pairing needs64 packets: at most three saved/subtile, before charging any displaced address work; it is not a512-packet windfall. Keep `sum`'s order (`S:1554-1568`), exp implementation, f16 P RNE, and existing denominator FMA semantics.

For one fully unmasked subtile, the retained control path has **383 packets**:16 WMMA,64 V loads,2 shuffles,193 VALU packets,43 waits,24 delay hints,41 scalar/control. It includes score selection, masking and loop control (`S:1576-1609` then`:1227-1575`). The positions load is skipped on that path. A masked subtile executes the extra load path. Four such subtiles plus QK and setup give **2098 compute packets/wave/KT64,128 WMMA**. No timing claim follows from that count.

**Fill:** all four waves execute. The host trace explicitly follows all-valid load guards and eight loop iterations; it is a control-flow reconstruction, not a general ISA simulator. Results:

| Full KT64, per participating wave | K fill | V fill |
|---|---:|---:|
| Dynamic packets, excluding publish/turnover |**1216**|**1160**|
| Payload/global scale loads |32 b32 +4 b16|32 b32 +4 b16|
| Signed-byte extraction |96 BFE +32 arithmetic shifts|96 BFE +32 arithmetic shifts|
| int32→f32 conversion |128|128|
| Scale multiply + f16 pack |128 `v_fma_mix{lo,hi}_f16`|128 same|
| f16-scale→f32 conversion |4|4|
| LDS b32 stores |64|64|
| LDS destination address core |64 adds +64 ANDs +64 shift/adds|64 adds +64 ANDs +96 shift/adds|
| All VALU packets |712|714|
| Wait / delay / scalar-control packets |144 /8 /252|140 /6 /200|

Representative exact dequantization is `S:361-384` for K and`:609-653` for V, matching `K:240-265,285-310`. Packing has already fused multiply+half rounding; there is no extra standalone f16-convert instruction to delete. q8→f16 requires numerical per-element work. A b128 store does not remove it.

### 1.4 Two important resource/synchronization corrections

1. **f16 K is32KiB and V is32KiB**, not16KiB each:64×256×2 per plane; `K:115-117`, launch `crates/rdna-compute/src/attention.rs:3714-3720`. Two f16 KT64 tiles require**128KiB**, not64KiB. The16+16KiB premise belongs to native fp8 stage b. At64KiB/WG the current f16 layout already has one-WG/CU-equivalent LDS capacity; six waves/SIMD is a register-capacity remark, not six resident waves guaranteed by the launch.128KiB would consume the entire WGP pool, assuming the per-WG limit even admitted it.
2. There are **three static barrier pairs in the direct body, but only two steady-state pairs per KT64**: mailbox publish once (`S:147,159`; `K:170`), fill publish each tile (`S:779-780`; `K:314`), turnover each tile (`S:312-313`; `K:543`). The backedge does not revisit mailbox setup. The census's “3 per tile” wording (`:37-39`) must not drive a3→1 saving forecast.

## 2. Decision, frozen exactness contract and ranked options

**Attempt fragment-major LDS first; then the matching K-fragment producer.** Keep one KT64 K/V tile, four waves,64 QK +64 PV WMMA per compute wave, and all arithmetic order. No measured winner yet. K one-deep prefetch was already bit-identical but flat/slightly worse and reverted (`agent://Fa2PingPong`; `E/fa3ping/gates_s1/time.log`). Do not repeat it or sell WMMA/VALU ping-pong: `E/wmma_probe/coexec_RESULTS.md:6-19` shows no separate co-execution pipe.

Exact means the **same f16 Q/K/V/P operand bits, ordered dc reductions, KT16 online max/sum/exp graph, f32 O rescale/WMMA order per output, physical partial records and final normalization**. Reordering independent memory loads, independent rows or independent O components is permitted. Moving a rounding point, contracting a new FMA, changing FTZ/FP mode, reordering a reduction or substituting exp is not. “Intended exact” becomes admitted exact only after the gates below.

| Rank / option | Expected removed packet work | Exactness | Decision |
|---|---|---|---|
| **1 A: fragment-major K/V LDS, native-vector consumers** |Per compute wave: K loads160→64 (−96),72 QK address adds targeted for elimination; V loads256→64 (−192). **−360/wave,−1080/WG** working target; keep existing fill arithmetic and budget zero fill gain initially.|Bit-identical by address bijection + unchanged arithmetic; GPU proof owed.|Primary exact candidate. f16 uses **b128**, not one b64: a half8 is16bytes. Native fp8 later uses b64.|
| **2 C: K producer in destination fragment order, b128 stores** |K stores64→16 per wave; fewer loop/address packets. Working net K-fill saving**320/wave** (sensitivity160..400) after extra scale loads; **−1280/WG** nominal.|Bit-identical if signed extraction, sf product, RNE and zero-tail bits unchanged.|Compose after A; measure separately. V initially keeps safe scalar scatter stores.|
| 3 D: compile-time subtile selection, same softmax/PV graph |Remove96 score-select packets/wave/KT64 plus some loop control; nominal total128/wave, not measured. Four expanded PV bodies increase code size.|Intended exact; compiler must retain each row's operation order.|Reserve candidate, not required before handing off stage b; one scoped trial only after A/C if a measured bottleneck remains.|
| 4 d: VOPD / packed-f32 softmax rescale only |O pass is already almost fully dual-issued: ceiling12 packets/compute wave/KT64 from perfect pairing before charging displaced address work; packing non-reduction f32 ops may add little.|Same f32 instructions/rounding can be exact. Packed **f16** state, reassociated sum, RTZ P packing are not.|Do not spend a kernel rewrite on this alone. Check existing ISA before claiming savings.|
| b: q8 K scale in final score epilogue |No valid one-scale-per-column transform exists for eight independent32-D q8 scales. Correct blockwise alternative adds partial dot/scale/reduction work; cost must be rebuilt.|**Not bit-identical, and the naive full-dot version is not even real-arithmetic equivalent.** Correct blockwise version is KLD-gated.|Reject from exact work. No transfer of e091/3834 pins.|
| c: K **and V** b128 fill “with no per-element VALU” |Impossible for decoded q8 f16 planes: each side currently needs128 extract +128 conversion +128 mul/round operations/wave. V's writer owns key pairs across dimensions, not eight keys of one dimension.|Vector stores alone exact; removing necessary decode wrong. V producer transpose can be exact but costs shuffles/loads/registers.|Do not credit a V vector-fill gain without a concrete transpose and emitted net count. Keep it out of A/C.|
| e: two-KT64-tile ring, barriers3→1 |Wrong storage and barrier premises. Legal f16 ring needs128KiB and steady baseline is2→1; new loader work cannot hide behind WMMA issue for free.|Potentially exact only with a full reuse/visibility proof; not an admitted candidate.|**Abandon under64KiB contract.** KT32 double buffering or compact q8 planes is a new tiling/decode experiment, not this optimization.|

Scale counterexample, host checked in R: half scale`0x3c01`=1.0009765625, code127, Q=1. Old f16 K is**127.125**; postponing scale produces**127.1240234375**. Even a common scale changes the f16 rounding point. With eight different block scales, `dot(Q,codes)*one_scale` is wrong before rounding is considered. Native fp8's per-head scale contract is different; see§7.

## 3. Frozen fragment layout and producer/consumer interfaces

Reuse the project's native vector-load idiom rather than inventing a packing framework: `kernels/src/gemm_gate_up_mq4g256v2_wmma_fp8.gfx12.hip:589-626` loads packed `v2i_t` directly from LDS into gfx12 WMMA. Its GEMM tiling/accumulation order is **not** transplanted into attention.

Let `F=8*element_bytes`, lane=`16*kg+ml`, sub0..3, dc0..15. Byte offsets within each plane:

```
Kfrag(sub,dc,lane) = ((dc*4 + sub)*32 + lane)*F
Vfrag(sub,dc,lane) = ((sub*16 + dc)*32 + lane)*F
K operand j = K[key=sub*16+ml, dim=dc*16+kg*8+j]
V operand j = V[key=sub*16+kg*8+j, dim=dc*16+ml]
```

Use tiny internal force-inline offset helpers with compile-time element width, no exported ABI/new dispatch flag. Keep K at byte0, V at byte32768 for f16; stage b V starts16384. Each fragment is aligned and contiguous; no circular wrap. Consumer offset formation is one bounded lane/sub base plus compile-time immediate offsets, not a permanently live array of64 addresses. Existing four-Q/four-K `dg` grouping and memory-clobber scheduling boundaries remain (`K:340-458`); do not roll the `u` loops or increase fragment lookahead. Q stays row-major half8, unchanged.

**Host proof already executed:** both element widths1/2, both planes, every64×256 element is a bijection; each of2048 fragments/plane is contiguous/aligned; the proposed K producer writes every element once. Wave-wide address-bank multiset has the expected four words/bank for a16-byte/lane transaction. This is **not** a measured bank-conflict/throughput proof: actual b128 read/write subtransactions, compiler lowering and fill scatter patterns remain GPU/ISA gates.

### A: first layout-only implementation

Change both producers' destinations and both consumers together, keeping their current work ownership and numerical decode. Existing q8 K slots and V key-pair slots remain (`K:231-313`), as does fwht3 K arithmetic (`K:196-226`). Scalar writes are acceptable for this first candidate. Replace four scalar/paired reads with one aligned half8/native-bit-vector load at all four QK blocks and PV. No change to global KV row layout or byte stride. Measure the whole tile: a cheaper consumer can be offset by a worse producer bank pattern.

### C: K writer refinement, same frozen layout

Give each loader wave its own KT16 sub=`wave`; lane owns key=`16*wave+ml` and half-dimension group kg. Iterate q8 block b0..7; retain its f32 scale for dc=`2*b,2*b+1`; for each dc load the eight codes at `block+2+16*(dc&1)+8*kg`, decode identically and immediately store one half8 at Kfrag(wave,dc,lane). Only one outgoing fragment lives at a time. Across the WG every K element still has one writer.

This issues16 b128 stores/wave instead of64 b32, without a lane transpose. Each wave now loads8 scales instead of4; kg halves duplicate a scale. Charge that cost (and changed cache/coalescing/bank behavior), do not hide it in the forecast. Eight-byte unaligned global code loads must be emitted safely via the existing memcpy idiom; never over-read a34-byte block or assume `blk+2` is naturally aligned. If the compiler splits b64 global loads, keep the actual count. Do not stage next-tile registers. For fwht3 either migrate the producer with unchanged cnorm/table arithmetic or keep its scalar writer using the same new offset helper; no forked compute body.

V remains the original key-pair producer, now writing Vfrag's inverse. A b128 V store would require one owner to gather eight keys of a dimension: that is not the existing producer's data. Do not add a shuffle transpose to C opportunistically.

### Bound publication: eliminate transient mailbox ownership

With a changing plane layout, do not assume the last eight LDS bytes stay unread/unwritten in a convenient order. Existing `K:142-174` has a publication barrier but no explicit reader-completion barrier before later V fill can reuse mailbox bytes. **The plan does not certify this reuse from apparent lockstep timing.** Use identical wave-local integer bound reductions: each wave's lanes0..7 read the same valid eight positions (`if(lane<8)` rather than`tid<8`), other lanes use the existing sentinel values, and retain the integer min/max reduction/readfirstlane semantics. All four waves derive identical gmin/gmax/seq_len from immutable positions. Remove mailbox stores/loads and their once-per-WG barrier. This adds three duplicate position-read waves once/WG, not per KT64; credit no hot-loop speedup. It avoids a new barrier or permanent LDS metadata and composes at exactly32KiB for fp8.

## 4. State ownership and transitions

| Transition / exact source | Before | Action | After / invariant |
|---|---|---|---|
| Q preparation: `K:630-699`; launcher`attention.rs:3643-3664,3701-3713` |Caller owns immutable f32 Q; scratch owner valid, old captures invalidated before growth.|Same existing f32→f16 (or fwht3 rotate→f16) preconvert, same stream.|Each half8 Q operand identical; Q never modified; no source-pointer-only activation reuse.|
| Bounds/init: `K:119-174` |No tile reader/writer; output state uninitialized.|Initialize O=0,m=−inf,l=0; every wave computes same integer bounds privately.|No live LDS mailbox. m/l/O remain lane-owned; helper has no live output state.|
| Fill tile t: `K:178-313` |For t>first, turnover barrier has released all prior plane readers. K/V global prefix immutable.|Unique writers fill bijective f16 planes; out-of-range keys write zero fragments with unchanged bits.|No consumer proceeds before unconditional fill publish. All four waves participate.|
| Publish: `K:314` |Producer writes may be outstanding.|Keep compiler-generated WG fence/barrier.|Every QK/PV consumer sees complete current-tile operands; no divergent barrier.|
| QK: `K:322-459` |K plane/Q immutable; four score chains zero; masks/liveness known.|Same dc0..15 WMMA order per score chain; shared Q loads; new vector K loads.|Same sacc0..3 bits. Drop K-fragment/address temporaries promptly; no next-tile overwrite.|
| Softmax: `K:462-510` |Prior-prefix m/l, current complete score vector.|Same sub0→3 order; causal masking before max; same exp, sum, partner shuffle, f16 P and denominator update.|Same m/l/P/alpha bits. Do not exchange V scale with l or move K scaling through softmax.|
| Rescale/PV: `K:512-539` |Lane-owned O, valid P and V fragment.|Same alpha multiply once/component then same WMMA for each dc; vector loads only.|Same physical f32 numerator. No cross-query shuffle or reordered key reduction.|
| Turnover: `K:541-543` |Last PV readers may still be active; helper may arrive early.|Keep unconditional WG barrier with its memory ordering.|Only now may any wave overwrite a plane for t+1; no ring or asynchronous loader state.|
| Direct/partial finish: `K:546-598` |Final m,l,O owned by compute lanes.|Same qok guarded normalization/direct stores or physical `[m,l,Onum256]` records, stride258.|Exactly one writer/component; zero/all-masked behavior unchanged. Partial comparisons use identical split count, not partial-vs-direct bit identity.|
| Merge/gate: `K:789-825`; existing engine epilogue |Physical partial records or raw O complete.|Merge unchanged; gate applied once at its existing engine site.|No normalization/gate movement to hide rounding differences.|
| Retained/captured replay |KV/Q/scratch owners survive dispatch; committed positions control visibility.|Do not change cache/recurrent/RNG/hidden-ring state or capture admission.|Same-format eager versus admitted replay state must agree. Current production FA2 ingress explicitly excludes capture/recording (`attention.rs:3273-3282`); a fallback pass is not proof of captured FA2.|

## 5. Honest time model and forecasts

### 5.1 Units and assumptions

No clock/occupancy/issue calibration was run. Count the **four fill waves and three compute waves**, not one of each. For a full unmasked tile, excluding prologue/epilogue/barrier exposure:

```
Pwg = 4*(1216+1160) + 3*(562+4+4*383) = 15798 packets
Cwg(c16) = Pwg + 3*128*(c16-1)
A saving = 3*(96 +72 +192) = 1080
C saving = 4*320 = 1280 nominal (640..1600 sensitivity)
A+C saving = 2360 nominal
Rwork = 1 - saving/Cwg
Rwall = u + (1-u)*Rwork                  # unchanged exposed time fraction u
```

c16 is an **assumed effective WMMA service cost** relative to a counted support packet; it is not measured cycles. SALU/VMEM/LDS overlap, wait lengths, bank conflicts, register residency, frequency and tails can invalidate a packet-proportional model. c16=4/8/16 gives base16950/18486/21558 issue-equivalents. Use16 for the nominal table, retain the other two as sensitivities, and assume u=0.25 only as a worked example. New bank stalls may make a candidate slower despite lower packet counts.

| Candidate | Rwork at c16=4 /8 /16 | Nominal Rwall (u=.25,c16=16) | Amortized ns/WG-tile at L8192 /32768 |
|---|---|---:|---:|
| A layout/consumers only |.936 /.942 /.950|.962|124 /139|
| C K fill saving alone, marginal against baseline |.924 /.931 /.941|.955|123 /138|
| **A+C nominal** |**.861 /.872 /.891**|**.918**|**118 /132**|
| d perfect O-rescale pairing ceiling |.998 /.998 /.998|.999|129 /144|
| D constant-subtile nominal |.977 /.979 /.982|.987|127 /142|

C depends on A's layout; its standalone row is marginal attribution, not a separately executable old-layout kernel. With c16=16 and C saving160..400/wave, A+C Rwall=.940..906; with all c16 sensitivities and u0..0.5 the wider scenario range is roughly.84..96. These are **planning scenarios, not confidence intervals**. Ring e has no legal64KiB f16 candidate and therefore no honest per-tile forecast. Scale option b has no equivalent one-epilogue candidate; a correct blockwise redesign requires a new operation model and KLD before any timing prediction. V vector fill without a transpose likewise has no credited saving.

### 5.2 Relating model units to existing measurements

Retained ship timings are batch512, positions at the end of L (`E/fa3ping/attrib_base/run.log:6,11-35`), not a complete pp8192/pp32768 model invocation. Counting actual causal WG/KT64 visits gives:

| L | Measured baseline kernel | WG-tile visits | Measured amortized time/visit | A+C nominal forecast |
|---:|---:|---:|---:|---:|
|1024|497.32us|3200|155.4ns|456us|
|8192|4112.07us|31872|129.0ns|**3774us**|
|32768|18757.27us|130176|144.1ns|**17217us**|

The ns columns are **GPU-wide throughput-normalized service per WG-tile**, not elapsed latency of an individual WG or wave. They include amortized launch/preconvert/tails. Their difference across L is evidence against a universal packet→nanosecond conversion. Nominal A+C seeks roughly8% attention-time reduction, with a conservative same-assumption6–9% band from K-fill uncertainty; it does not promise that result. The seven-shape TIME baseline is separately `E/fa3ping/gates_base/time.log:1-10`:37.80,69.81,260.87,521.67,1079.37,4307.95,19377.92us, plus three truncated rows20.31,247.46,616.57us.

At **pp8192**, supplied FA2 wall share s=.146 and R=.918 imply `speedup=1/(1-s+s*R)-1` = **+1.21%**, before any unrelated change. An attention-only reduction of10.1% is needed to reach about+1.5% E2E here. This exact attempt may be worthwhile at32K yet be near noise at8192; do not promise a double-digit end-to-end win.

At **pp32768**, the actual attention share was not provided. With the same R and *assumed* shares s=.30/.50/.70, E2E forecasts are **+2.53%/+4.28%/+6.10%**. Measure the true share; “dominates32K” is not a numeric denominator. The4112/18757us standalone timings and the supplied237us→2.7ms row-scaling observation belong to different invocation geometries; do not combine them as a single fit.

Roof sanity: three compute waves perform3×128×8192=3,145,728 matrix FLOPs/WG-tile.194.8TF/s f16 gives16.15ns/device-wide ideal matrix service;389TF/s fp8 gives8.09ns. Neither is a whole-tile latency prediction. `E/wmma_probe/coexec_RESULTS.md:6-19` forbids making softmax/fill VALU free by assigning it another wave.

## 6. Composer slices, exact touchpoints and claim-scoped gates

No production implementation is part of this document. **One kernel integration owner** serializes same-file edits. Evidence preparation and independent review can run concurrently; do not invent parallel K/V editors of K. All composers skip project-wide builds/tests/linters/formatters; Main owns those once at the join and authorizes GPU gates. Re-anchor after the fp8 pool changes shared files.

### Slice E — evidence/protocol preparation, independent of kernel editing

Own only `R/exact/` evidence and a disposable harness, not production K/launchers or the fp8 pool's screen file. Freeze baseline source/object/ISA and compiler options; use retained census and input fixtures as references. Prepare same-input two-module raw-O comparison, input/output hashes, canaries and timing records. Existing oracle is `crates/hipfire-runtime/examples/tmp_fa2_fp8_oracle.rs:70-78,138-171`; it writes **gated** O and has temporary-path defaults, so use explicit R output paths and a disposable raw-O differential extension. The formerly reported `tmp_fa2_pingpong_gates.rs` is absent from this inspected worktree; do not claim that nonexistent file is a runnable deliverable. Use its retained10-row receipts plus the existing oracle source to reconstruct the scoped harness under R.

Frozen harness interface: baseline/candidate module handles consume the **same** q16/K/V/positions allocations, same scalars, layout-independent public kernargs and dynamic65536; write disjoint canary-protected outputs. Handle symbols directly per module or rename only the lab reference wrapper, since the production function cache is symbol-keyed (`K:645-651`). Neither production flags nor KV dtype select “old layout” versus “new layout”. Publish per-row raw TIME samples, source/HSACO hashes and bit-diff report. Acceptance is a runnable protocol ready for Main, not an invented GPU pass.

### Slice A — exact shared layout owner

Own K only: update comments/helper near`:35-39,68-79`, plane contract`:115-117`, bounds`:142-174`, fwht3 destinations`:208-224`, q8 K destinations`:245-265`, V destinations`:295-310`, four QK operand loads`:363-379,387-403,411-427,435-451`, and PV load`:521-533`. Implement§3 A plus mailbox-free bounds. Keep numerical expressions, four-Q grouping, all do/qok guards, two tile barriers, softmax and epilogues unchanged. All producers/consumers switch together; remove unused swizzle code only when no remaining specialization uses it. No stale compatibility path.

Acceptance: host bijection/producer proof, matched source arithmetic, target K-LDS64/PV-LDS64 per KT64/compute wave, no new half conversions/shuffles in QK/PV, and no address-family lifetime extension. If high-level aligned vector loads fail to lower, at most one localized typed-vector/address-lifetime correction; no broad inline-ISA rewrite. Main runs gates G1–G3 before C; E can prepare concurrently.

### Slice C — exact K producer refinement, after A

Same kernel owner, only q8 K fill`:228-267` plus any local offset helper required by§3 C; fwht3 numerical decode stays untouched except its already-migrated destination. Keep all four waves as loaders and zero invalid keys; one outgoing half8 lifetime. No next-tile prefetch, extra LDS or V transpose. Publish emitted K fill counts, global bytes/scale loads and bank/resource effect; require actual net reduction, not “b128 present”. Main reruns the same output/metadata/TIME gates on A+C and retains A alone if C loses. No numerical tolerance downgrade.

### Slice D — optional, separately admitted softmax specialization

Not on the critical path for stage b. Only if A/C measurements identify remaining selector cost: specialize the four sub calls with compile-time sub and direct saccN arguments over`:461-539`; same scalar reductions and per-output PV order, same4-QK-chain lifecycle. Do not fold full-mask fast paths, packed-half arithmetic or exp changes into it. Compare the expanded emitted control/score-copy count, code footprint, VGPRs and TIME. One attempt; no indefinite unrolling search. It must beat the admitted A/C arm, not baseline, to survive.

### Existing symbols/call sites — intentionally unchanged by exact slices

- `fa2_gqa_body<PARTIAL>` at`K:89`, direct call`:626`, fwht3 call`:736`, partial call`:780`; merge`:787-825`. Same entry names/kernargs and partial258 ABI.
- Production q8 ingress `Gpu::attention_q8_0_flash_prefill_wmma` at`crates/rdna-compute/src/attention.rs:3255`, exact predicate/call`:3273-3286`; direct launcher`:3568`, preconvert/scratch`:3643-3664,3705`, launch65536`:3716-3734`. Fwht3 launcher`:3762`. No admission widening, capture enabling or Rust API change.
- JIT source constants `crates/rdna-compute/src/kernels.rs:5806-5807,5822-5825,5833-5857` all include K; compile all exposed specializations, do not validate only one symbol.
- Direct lab call `crates/hipfire-runtime/examples/tmp_fa2_fp8_oracle.rs:154`; real screen `crates/saddle-lab/examples/tmp_fa2_fp8_screen.rs:347-380`; attribution driver selected in`E/fa3ping/attrib_base/run.log:3-8`.
- Only rust-analyzer is available here; references on the absolute external-worktree launcher returned empty despite these explicit calls. Treat that as tooling scope, **not proof of no callers**. Any composer adding a Rust API change must query references in the implementation workspace first.

### G1 — compile/ISA/resource gate (Main; before timing)

Compile with the baseline gfx1201 flags (`E/fa3ping/census_base.md:3-6`); keep source/object hashes and metadata for direct, fwht3 and exposed partial variants. **0 VGPR spill,0 SGPR spill,0 scratch bytes** is mandatory. Target/admit≤240 VGPR for this exact attempt, preserving the baseline239/6-wave register budget; a241..256 candidate requires an explicit revised occupancy/performance decision, not a silent pass. Dynamic LDS65536, no additional static LDS, wave32/block128. Verify actual occupancy limits with dynamic LDS rather than equating compiler waves with resident WGs.

Require64 QK and64 dynamic PV f16 WMMA, no numerical-mode change, no new operand conversions or scratch traffic, unchanged scalar sum/exp/FMA semantics. Rebuild a **dynamic** full/tail phase census; verify b128 consumer lowering and store safety, bank effects and new fill costs. A aims at at least80 fewer QK LDS/address packets and128 fewer PV LDS packets/wave before expensive real runs; otherwise the claimed mechanism did not materialize.

### G2 — exact outputs, memory/state and tails (Main)

1. Real-data screen: layer35,4224 tokens,384-row final chunk, qstride4; `tmp_fa2_fp8_screen.rs:291-380` taps actual Q/K/V/positions and launches raw direct O. Use tree-built `fa2_fp8_screen --model /home/kaden/.hipfire/models/qwen3.8-27b.mq4-xt --layer 35 --tokens 4224 --qstride 4 --kernel-out <R/exact/arm>` with **`HIPFIRE_GFX12_FA2_FP8=0`**, PREFILL=1,LLOYD=1 and identical effective GEMM/model flags. For the frozen baseline inputs raw O is9,437,184bytes, md5 **`1f7a1183a751de7555066bcf5a1ca11c`** (`E/fa3ping/screen_base/md5.txt`). Require unchanged md5 and zero differing bits. Also compare gate and input hashes; changed upstream activations are not a license to rebaseline an exact kernel. Prefer same-buffer dual-module replay. Do not use the old stage-a tolerance oracle or double-apply sigmoid.
2. TIME + raw-output byte equality on all seven `(batch,ctx)` pairs: `(4,256),(8,512),(32,2048),(64,4096),(128,8192),(256,16384),(512,32768)`; also retained tails `(3,100),(6,2000),(8,5000)`. Preserve input-scale/RNG settings. Add seq boundaries15/16/17,31/32/33,63/64/65; batches1/7/8/9; duplicate/nonmonotone positions supported by the direct API; all-masked rows and empty partial splits. Compare same split counts1/8 and physical partial records, not split-vs-direct association.
3. Check canaries and immutable Q/K/V/positions before/after repeated calls. Compare fwht3 to its own baseline, including unmodified original Q; q8 equality is not fwht3 quality certification. Any output mismatch blocks timing/admission until explained and fixed.
4. Eager/graph/retained-replay claim gate: existing ingress intentionally falls back while recording/capturing; prove that path still does so. If directly capturing the lab launcher, compare repeated output and immutable-input/scratch lifetime, including growth invalidation; label it lab coverage, not newly admitted production capture. Required Redline route is `docs/VALIDATION.md:120-141`; do not infer FA2 execution merely from an overall parity pass.

### G3 — pins, end-to-end bench, decode and serve (Main/integration reviewer)

- **Exact arithmetic pins:** fresh tree-built `eval_hipfire`, q8/q8, graph0, prompt-normalization0, default fp8-GEMM route (V2=0, IU4=0; FP8_GATEUP/RESID/QKVZA/QKV=1,SLABS=2) per`E/GemmV2Plan/final/kld.md:3-14`. Model`/home/kaden/.hipfire/models/qwen3.8-27b.mq4-xt`, ref`/home/kaden/kldrefs/qwen3.8-27b.ref_wt2.bin`, `--kv-mode q8 --kv-v q8 --scoring-mode prefill --max-chunks 1|2`, explicit output under R. Require **`e091d74e5f290efb9bc343f107a716b4`** and **`3834e25a771ecfb9e4f1a02383aa3fe0`**, not the v2-GEMM ON pins. If running24, exact baseline is0.048659/md5`deff6ed3ac7c600c3ec5e623f58db5da`. These are not iu4-route pins: separately establish matched-arm equality for that performance route. Deliberate arithmetic changes invalidate pin transfer and owe fresh24-chunk KLD≤matched OFF+0.0005; they are outside A/C admission.
- **Interleaved bench:** frozen baseline/candidate binaries or module-cache manifests, same assigned GPU and flags, OFF/ON/OFF/ON plus reverse order if needed; compare **old f16 versus restructured f16**, not FA2-disabled versus FA2. Run `hipfire bench qwen3.8:27b-mq4-xt --matrix --pp 512,2048,8192,32768 --ctx 128 --tg 64 --spec off --runs 3 --warmups 1 --kv-mode q8 --json`, graph1/LLOYD1 and the same **iu4 route** in both arms for the supplied14.6% claim. Record exact binary, JIT source/object and full flag identity. Run a separately labeled default-fp8-GEMM arm if making that claim. Per-row raw samples/medians, temperature/clocks and FA2 hit attribution required; standalone TIME is not E2E proof.
- **Performance admission:** no>1.5% regression on any pp or TIME row; require reproducible>1.5% E2E gain atpp32768, with pp8192 at least nonregressing. Session spread about±0.5% does not excuse adverse>1.5% changes. A/C's forecast pp8192+1.2% may remain inconclusive; report that rather than rounding it into a win. Do not average away a bad long-context row.
- **Separate decode run:** repeat without preceding32K heat soak. Preserve approximately36.4tok/s accepted clean floor and matched-arm nonregression. The documented28–29tok/s post32K thermal dip (`E/GemmV2Plan/final/bench.md:35-37`) must be rerun, never averaged in or used to dismiss a pp regression. Decode dispatch must remain unchanged.
- **Serve:** exact model/settings, graph-on production daemon, `python3 scripts/serve_harness.py --mode battery` and`--mode chain`, `--tag qwen3.8:27b-mq4-xt --kv q8 --mtp off --dflash off --sampling registry --thinking off --max-tokens 256 --seed 0`, using this tree's flags (`E/GemmV2Plan/final/serve.md:3-6`, not an invented positional subcommand). Record daemon hash, read every decoded turn, cover cold load and related-turn prefix reuse/reset; no attractor/empty/stream/corrupted-state result. If claiming speculative/retained replay, additionally exercise the admitted spec route and joint KV/DN/EF/hidden/commit-position parity under`docs/VALIDATION.md`; AR battery alone cannot certify it.

### Bounded kill criteria / reviewer veto

One A implementation plus at most one lowering/address correction; one C refinement plus at most one same-mechanism correction. Abandon the offending leg for any unexplained bit difference, new spill/scratch, failed resource gate, widened live-fragment/address family, unsafe tail/global read, producer-bank regression that cancels the consumer saving, or failed repeated performance gate. Retain A without C only if A independently passes. If all exact changes tie/lose, ship the original f16 structure and give stage b that honest handoff; do not turn A into a precision experiment or retry ring/ping-pong under its name. D gets one independent trial only. Reviewers have the final veto after claim-scoped evidence, not after packet counts alone. Integration records actual results and removes disposable harness scaffolding only after smoke proof; retained R analysis/receipts remain evidence.

## 7. Native-fp8 stage-b composition — one schedule, no fork

Companion authority: `docs/plans/2026-09-18-gfx1201-fp8-kv-stage-b.md`, especially`:72-84` (updated BF16/VMM scope),`:115` (names),`:174-228` (Q/scale/O/LDS),`:302-345` (handoff/gates/slices). **Kaden's update admits contiguous and production VMM for both native fp8 and the existing BF16 mode; slots remain rejected.** This exact layout plan does not reinstate the superseded VMM exclusion or implement allocation changes. Logical fp8 row1032bytes/side and BF16 row2048bytes/side are independent of queried VMM page granularity; map the full live prefix before access, never derive row stride from page-rounded mapped byte size. Those F/L/O-owned changes and BF16 quality-control runs remain required by the companion plan.

Freeze this handoff after A/C admission: commit/source/HSACO hashes, accepted optional legs, fragment formulas/width, wave roles, two tile barriers, mailbox-free bounds, live-fragment budget, exact-output/pin/resource/TIME/bench verdict. B edits **the same K** only after its integration owner releases it; F/L/O may prepare independently. If A loses, B consumes the original measured baseline instead. No copied FA2 file or stage-a in-loop-decode revival.

1. **Q0 first:** `HIPFIRE_FA2_KMODE=8`, FP8 arithmetic OFF, distinct native-fp8-KV/f16 entry. Decode stored E4M3 with its stored f16 head scale at fill into the accepted **f16 fragment layout**, preserving f16 QK/PV and65536 LDS. This isolates storage error. Companion24-chunk gate≤0.049159 and≤matched q8+0.0005, plus BF16 control and same-format contiguous/VMM parity, must pass before stage b. The q8 exact O/pins do not transfer to a new cache format.
2. **Stage-b operands:** F=8 instead of16 in the same Kfrag/Vfrag formulas, plane16384bytes each, total**32768 exactly**, no static scale table or second tile. Q preconvert produces packed fp8 Q plus f32 sq once, using distinct symbols. Both QK/PV use native gfx12 fp8 K16 WMMA, same64+64 instruction count and same key/subtile order. Fill copies raw fp8 fragments; it does not re-expand them inside QK/PV. Native V transpose/scatter remains a producer concern, not another schedule.
3. **QK scale epilogue:** `score[j]=acc[j]*sk[key_j]*(sq[query]*scale_attn)` before mask/max. sk is per token/head and therefore separable in real arithmetic, unlike q8's eight scales; still KLD-gated due quantization/rounding. Scale ownership is accumulator key`sub*16+8*kg+j`, not the supplying K lane's ml. No persistent64-key scale VGPR array.
4. **PV and changing units:** denominator uses unweighted e. Pack `w=e*sv[key]` to p8 with positive block scale b; retain companion b-floor/zero-row rules. O state invariant`Ophysical=bprev*Ofr`; replace alpha rescale with`rho=(alpha*bprev)/bnew`, then fp8 WMMA, then bprev=bnew. Direct store restores bprev/l; partial stores **physical**bprev*Ofr in the unchanged258 record. No extra512 output-scale FMAs silently introduced. This is a new numerical contract, never exact f16.
5. **Rebuild stage-b packet model after restructure:** with admitted A, both half8 b128 and fp8 b64 are **one LDS instruction/fragment**. The companion plan's old D≈208 load-packet saving is therefore **not additive/transferable**. Matrix roof/service advantage and fill decode removal remain possible; fresh scale-feed/P-quantize/rho overhead N must be charged. Count actual f16 versus native-fp8 ISA and measure; do not claim another2x load-instruction cut or free VALU/WMMA overlap.

Cross-architecture/non-goals: exact work remains gfx1201 H24/Hkv4/D256, existing contiguous-address noslots attention semantics, context≤32768 and existing capture predicates. gfx11's separate source/KT32 schedule, gfx942/gfx1200, other shapes, scalar decode, global KV formats, recurrent/GDN state, hidden-ring/rollback algorithms, tree-FA2, slots, adaptive/compaction and>32K routing are unchanged here. Static VMM and BF16 **are in scope of the parallel native-KV project**, not exclusions; this fragment layout neither adds nor removes those backend admissions. No new numerical approximation, matrix-roof guarantee, occupancy guarantee or shipping performance claim is made.

## 8. Planning evidence delivered

Executed `python3 .codeinsight+research/scratch-2026-09-17/Fa2Plan/host_audit.py` successfully:1941/267 census reproduction; explicit valid-path1216/1160/562/383 counts; f16/fp8 K/V element bijections and2048 aligned fragments/plane; single-writer K mapping; scale-rounding counterexample; forecast arithmetic. Outputs:`R/host_audit.json`, `R/attribution.tsv`. This validates the **host accounting and layout algebra only**. No GPU, compiler, project-wide test, formatter or linter was run; all candidate codegen, bit identity, bank behavior, pins and performance remain the composer/Main/reviewer gates above.
