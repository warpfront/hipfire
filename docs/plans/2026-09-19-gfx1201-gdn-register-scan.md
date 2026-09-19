# gfx1201 GDN: a register-resident chunk-64 scan, with a measured solve budget

Date: 2026-09-19. **Plan only; no production implementation or dispatch change.**
Base: `mq4-lloyd` at `2c093bd4a`. Worktree: `/home/kaden/ClaudeCode/warpfront/wt-gdnreg`, branch `gfx1201-gdnreg-plan`.
All new experiment evidence is untracked under `/home/kaden/ClaudeCode/warpfront/wt-gdnreg/scratch-2026-09-17/GdnRegPlan/` (called **E** below). No writes were made to the base worktree or hipfire-beta.

## 1. Measured assumptions first

### 1.1 Result and limits

The strongest candidate is **64 value rows per WG, 256 threads/eight waves, f16 register state, f32 delta and solve, C=64**. Two WGs cover one value head. Within each WG, pairs of waves own the two key-dimension halves. This is not the old LDS-state scan.

The load-bearing experiment is `E/reg_probe.hip`, executable `E/reg_probe-unrolled`, assembly `E/reg_probe-unrolled.s`, compiler receipt `E/compile-unrolled.log`. It performs dense KK/QK packet quotas, S·K, S·Q, M·delta and carry with live state fragments, optionally a real 64-row f32 forward-substitution instruction shape. It is **not a correct GDN kernel**: its matrices are synthetic, its Grams do not construct its solve coefficients, and it omits real input staging, cross-wave partial sums, decay construction, output stores and Q8+EF commit. The numbers below are instruction-shape floors, not kernel predictions or quality evidence.

Physical ordinal 1 only, using:

- `HOME=/home/kaden/.hipfire-homes/ab1`
- `ROCR_VISIBLE_DEVICES=1`
- `HIPFIRE_MODELS_DIR=/home/kaden/.hipfire/models`
- `HIPFIRE_KERNEL_CACHE=/home/kaden/.hipfire-homes/ab1/.hipfire_kernels`

Timing start/end was announced on hub. Ordinals 2 and 3 were untouched. Device query: R9700, gfx1201, wave32, HIP `multiProcessorCount=32`, reported shared memory per multiprocessor 65,536 B, advertised clock 2,350,000 kHz. **“MP” below means that HIP-reported unit, not an independently counted physical CU.** Event timings are medians of seven batches of twenty launches, three warmups. No clocks were locked; initial cold/clock-transition observations varied substantially. The 128-chunk run warms the device and characterizes amortized issue cost; it does not authorize skipping 512-row commits.

**Accepted receipts:** `E/time-unrolled-128-0.txt`, `E/time-unrolled-8-1.txt`, `E/time-unrolled-8-2.txt`. All rows reserve 28,672 B dynamic LDS and have zero private scratch/spills.

| State/partition, solve included | State storage floor, VGPR/lane | Actual VGPR | Eight chunks, µs/launch, two warm runs | 128 chunks, µs/launch | HIP active WG/MP |
|---|---:|---:|---:|---:|---:|
| f32, full 128 rows | 64 | 189 | 301.55 / 301.29 | 4657.18 | 2 |
| f16, full 128 rows | 32 | 144 | 286.26 / 300.74 | 4566.40 | 2 |
| f32, split 2 | 32 | 141 | 305.85 / 309.06 | 4687.19 | 2 |
| **f16, split 2** | **16** | **124** | **219.41 / 222.20** | **3263.46** | **2** |
| f32, split 4 | 16 | 117 | 586.93 / 607.39 | 9158.33 | 2 |
| f16, split 4 | 8 | 114 | 591.12 / 612.06 | 9202.26 | 2 |

The selected split-2 number is **27.43–27.78 µs per C64 round for all 48 heads**, or **0.571–0.579 µs per head per round amortized over the grid**. The printed probe field divides by WGs, so multiply that field by two to normalize split-2 to heads. The long-run corresponding head-round cost is 3263.455/(48×128) = **0.5312 µs**. These are not isolated single-head latencies.

Without the solve, the same selected geometry takes **35.89–36.10 µs for eight rounds**, VGPR 98: roughly 0.0935–0.0940 µs/head/round. Therefore the solve, not raw f16 WMMA issue, is the first optimization constraint. Full f32 without solve already uses 170 VGPR. Arithmetic storage counts alone do not prove allocation.

The compiler partially unrolls the requested 64-row solve by 32, executing its inner loop twice. This materially changes allocation and timing relative to the bounded-loop probe. The bounded-loop census-valid alternative is retained in `E/reg_probe-bounded.hip`, `E/reg_probe-bounded.s`, `E/time-distinct-*`; it uses 141 VGPR for f16 split-2 and does not meet the target. Its full-f32 solve is faster but uses 216 VGPR. Do not select variants from different compiler schedules without their resource receipts.

**Rejected evidence:** `time-initial.txt` and every `time-final-*` predate the distinct-chain fix. Compiler CSE emitted only 84/60 rather than 144/96 WMMAs per wave/chunk. They are not valid packet floors. Final distinct chains emit exactly 144 (full), 96 (split-2), or 72 (split-4) WMMAs, checked in ISA; outputs keep every state fragment live. No instruction-PMC claims are made.

### 1.2 Important correction from the actual shipped ISA

The serial source declares `S_f4` in LDS, but the **shipping compiler already register-promotes its four-row state inside the token loop**. Do not sell this redesign as removing that kernel's per-token state LDS traffic.

The cached source `/home/kaden/.hipfire-homes/ab1/.hipfire_kernels/gfx1201/gated_delta_net_q8_fast.ecd733273d0cfbc2.hip` was compared byte-for-byte with the base source plus the exact `GATED_DELTA_NET_Q8_FAST_SRC` defines (`HIPFIRE_GDN_DPP_REDUCE`, `HIPFIRE_GDN_PREFETCH`): equal. Its actual cached HSACO was unbundled and disassembled, not rebuilt as a stand-in. Binary SHA256: `7d271a1d691887798633f76bb48a6becbbd174ad8e96367884444ff92b0e1b62`.

`E/serial-shipped.isa`, `E/serial-steady-loop.isa`, and `E/isa-census.json` show:

- Steady token path 0x2400–0x2a00, excluding final-token zero-prefetch arm 0x253c–0x2574: **256 instruction packets per four-row wave/token**. A VOPD packet is counted once.
- 127 vector-prefixed and 120 scalar-prefixed packets, eight global f32 loads, one global b128 output store; **zero LDS loads/stores in that hot path**.
- Eight `v_permlanex16_b32`, 28 DPP adds, four DPP moves, plus readfirstlane/reduction dependencies.
- Cached resource metadata: 60 VGPR, 55 SGPR, no scratch/spills. Source LDS is 2048 B and only touched around the promoted recurrence/commit.
- Thus one head/C64 costs about **256×32×64 = 524,288 serial wave-packets**, not an LDS bandwidth floor.

The prior failed scan remains decisive: `/home/kaden/ClaudeCode/warpfront/wt-gdnscan/scratch-2026-09-17/gdnscanfix/metadata-fixed.txt` reports 61,952 B LDS and zero spill; 32,768 B is its 64×128 f32 state. That limits it to one WG per reported 64-KiB unit. Its oracle receipt separately reports serial 477 µs versus scan 4037 µs in that vehicle; the assigned matched-profile baseline is ~570 µs and ~7.4× regression. Do not mix those timing denominators. Register placement prevents repeating this mistake; it does not by itself explain a speedup over shipping serial.

## 2. Decision, forecast, and abandon numbers

**Authorize a bounded kernel experiment, not integration or default enablement.** Primary: f16 state, split-2, register delta, f32 lower solve, compact Gram construction followed by expanded column-major solve storage. Full f32 remains a quality reference, not the ≤128-VGPR implementation. Split-4 is **abandoned now**: its measured floor already exceeds the ~570-µs incumbent in the final schedule. Wave-specialized ping-pong is excluded: prior measured PING/SERIAL 1.06–1.13 and absent WMMA/VALU co-execution give it no budget.

The production kernel does not exist. The following is an explicitly conditional forecast, not a result:

| Quantity | Conditional forecast |
|---|---:|
| Selected packet+solve floor, 512 rows | 219–222 µs/launch, measured |
| Missing input/state/output traffic, layout, exp, barriers, pair reduction, commit | **120–198 µs/launch allowance**, unmeasured |
| Complete scan | **340–420 µs/512-row launch**, central 380 |
| GDN contribution at pp8192 | **31.88–39.38 µs/token**, central 35.63 |
| GDN contribution at pp32768 | **same 31.88–39.38 µs/token**, assuming unchanged 512 commits and stationary layer timings |
| Saving against 570 µs × 48/512 = 53.44 | **14.06–21.56 µs/token**, central 17.81 |
| iu4 wall at pp8192, calibrated to ~375 baseline | **353.4–360.9 µs/token**, central 357.2 |
| default fp8-v2 wall at pp8192, calibrated to ~470 baseline | **448.4–455.9 µs/token**, central 452.2 |

At pp32768 there is no measured wall baseline in this experiment. If unchanged FA2 cost per token scales linearly with context from 76 to 304 µs and all other components stay constant, the calibrated iu4 baseline is 375+228=603, and candidate **581.4–588.9 µs/token**; fp8-v2 candidate **676.4–683.9**. This is a scaling model, not a pp32768 benchmark. More robustly: subtract 14.06–21.56 from the matched measured pp32768 wall. Component profile sums and bench wall are different quantities; the supplied component estimates sum above 375, so this table uses deltas rather than summing them into a fictitious baseline.

**This GDN plan alone cannot achieve 290 µs/token.** Even deleting all 53.44 would leave about 321.6 on the iu4 arm. The projected contribution still needs roughly 63–71 µs/token from the other plans to reach 290.

Hard gates before host integration:

1. Candidate metadata: **≤128 allocated VGPR**, **zero scratch and zero register spills**, static+dynamic LDS **≤30,720 B**; runtime occupancy API must report **≥2 active 256-thread WGs per reported MP**. No S state array in LDS, including compiler-induced private spills. If two bounded layouts cannot achieve these, abandon the ≤128-VGPR design rather than quietly use 62 KiB LDS.
2. Census: 1536 WMMA/head/C64 for the conservative split-2 schedule; total steady wave-packets **≤150,000/head/C64** after real staging. Probe is 112,704; >150k means the assumed packet reduction has substantially evaporated.
3. Complete, initialized kernel median **≤427.5 µs** (0.75×570) and **≤0.75×its matched serial median**, five alternating batches, repeatable within 3%. >427.5 or <25% speedup abandons integration. Do not lower this threshold after observing a candidate.
4. Matched bench should save **≥10 µs/token at pp8192**, decode regress **<1%**, pp32768 must not reverse the saving. Kernel-only speed is not sufficient.
5. Any quality/commit/negative-route failure below kills the candidate. No repinning flag-OFF, no accepting a better isolated timing to waive KLD.

## 3. Frozen data ownership and kernel math

### 3.1 Interface and admitted shape

Proposed new source: `/home/kaden/ClaudeCode/warpfront/wt-gdnreg/kernels/src/gdn_register_scan_q8.gfx1201.hip`.
Entry: `gated_delta_net_q8_register_scan_gfx1201`. C ABI is exactly the existing fast 13-argument order:

`(q:f32*, k:f32*, v:f32*, gate:f32*, beta:f32*, s_q8:i8*, s_scales:f32*, output:f32*, n_tokens:i32, n_heads:i32, head_dim:i32, frame:u32, s_ef_residual:f16*)`.

Grid `[48,2,1]`, block `[256,1,1]`, one WG per `(value_head, value_row_half)`. No new global workspace, allocator, state owner, stream, or replay payload. Accept only exact gfx1201, dense Qwen3.8-27B shape H48/D128 and Q8+EF ordinary sequential prefill, single-end requant, 64≤N≤512. N<64 and other unsupported inputs use the unchanged existing route. Ragged final C64 uses masks, not an extra requant. No launch crosses a 512-row commit boundary.

**Physical input layout matters:** model has 16 QK heads, but this prefill path has already normalized/scaled/interleaved Q/K to `[N,48,128]`. Base `prefill.rs:6298-6300` uses `v_dim` views, and the shipping wrapper does not set `HIPFIRE_GDN_QK_HEAD_DIV=3`. New ABI reads expanded head `h`, not `h/3`. Compact-QK conversion/sharing is a separate non-goal.

### 3.2 Register map and arithmetic

For split-2, `wave = tid/32`, `ml=lane&15`, `kg=lane>>4`:

- `value_group = wave/2` (0..3), `key_half = wave%2` (0..1).
- Value row = `split*64 + value_group*16 + ml`.
- State fragment `S[b][j]` = key `key_half*64 + b*16 + kg*8+j`, b=0..3, j=0..7.
- Each wave owns 16×64 state elements. Per lane **32 f32 =32 VGPR**, or **32 f16 packed =16 VGPR**. Both key halves jointly own exactly 64×128 state elements, without duplication.
- Delta fragments `delta[t][j]`, t=0..3, represent token `t*16+kg*8+j` for that lane's value row: **32 f32 registers**, duplicated across the key-half wave pair only after partial S·K reduction.

Full rows / split-2 / split-4 state-only budgets at 256 lanes:

| State type | 128×128 | 64×128 | 32×128 |
|---|---:|---:|---:|
| f32 | 64 VGPR | 32 VGPR | 16 VGPR |
| packed f16 | 32 VGPR | 16 VGPR | 8 VGPR |

The split-2 target live budget is S16 + delta32 + f32 accumulator8 + two half8 operands8 + gate/address/mask/solve temporaries up to64 = **128**. The measured complete synthetic solve uses124, so only four registers of numerical headroom are demonstrated. Real staging must reuse phase-dead registers; do not assume the probe proves final allocation. f32 split-2 is141 even before real staging, so simply changing the state type does not fit.

Use the existing gfx12 fragment convention from `/home/kaden/ClaudeCode/warpfront/wt-gdnreg/kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:25-39`: operand row `ml`, K subrange `kg*8+j`, accumulator row `kg*8+j`, column `ml`. Compute S products as `K*S^T` / `Q*S^T`, and carry as `K^T*delta`; this makes the state accumulator layout directly reusable as the next B operand without transposing S through LDS. Store/load streamed operands in fragment order, not conflicting row-major half8 gathers.

For a chunk, G is inclusive sum of log-gates, e_i=exp(G_i), d_ij=exp(G_i-G_j):

- `L[i,j] = beta_i*d_ij*(k_i·k_j)` for j<i, zero otherwise.
- `rhs_i = beta_i*(v_i - e_i*S_in*k_i)`.
- Solve `(I+L)*delta = rhs` exactly once; never form an inverse and solve a second time.
- `M[i,j] = d_ij*(q_i·k_j)` for j≤i, zero otherwise.
- `out_i = e_i*S_in*q_i + Σ_j M[i,j]*delta_j`.
- `S_out = exp(G_last)*S_in + Σ_j exp(G_last-G_j)*delta_j*k_j^T`.

WMMA inputs are f16, accumulators f32. Beta, G, exp/decay, L/M storage and forward substitution remain f32. Delta is rounded to half only when consumed as a WMMA operand. The primary rounds S to packed f16 once per C64 carry, **not per token and not after each K16 WMMA**. Promote a state fragment for all four carry K16 steps, then round once. Commit folds EF in f32 after promoting final S. No f16 triangular coefficients or reciprocal/exp substitutions are proposed.

CPU f64 algebra was executed for T129/D128/R64/C64, including a one-row tail: maximum output error 4.996e-16, final-state error 3.331e-16 (`E/algebra.json`). This proves the written decomposition, not its f16 numerical suitability.

### 3.3 Transitions and invariants

| Transition | Before | Work and ownership | After / required invariant |
|---|---|---|---|
| Host admission | S codes/scales/EF and conv state belong to one DeltaNetState layer/slot | Select only ordinary route; reserve exactly N frames; retain same pointers | No frame reset, state clone, or hidden slot stride |
| Load | Global Q8 codes/scales are last committed S; EF is deferred | Each WG reads only its value half; each wave its key half; reconstruct scale×code then f16 | **Do not add EF at entry**; every S element has one owner |
| KK / G prepare | S unchanged in VGPR | Stage K, construct compact strict-lower L and G/beta/decays; expand L only after K plane is dead | L has unit diagonal implicit; all barriers WG-uniform |
| S·K | S unchanged; L read-only | Wave pair produces key-half partials, exchanges f32 accumulator fragments in LDS, sums once | Both waves receive identical full RHS/delta for their 16 value rows |
| Solve | All RHS registers valid; L stable | 64 forward rank updates; solved delta broadcast between `kg` lanes, coefficients broadcast from column-major L | No S write; no WG barrier inside wave-local solve; full delta stays f32 in registers |
| QK / S·Q / output | S still chunk-entry state; delta solved | Reuse dead L arena for compact M then expand; pair-sum S·Q; both waves may compute identical M·delta | Only key_half=0 writes output; **do not sum two copies of M·delta** |
| Carry | All output reads of old S complete | Update each disjoint S fragment with decayed delta/K; f32 accumulate then one f16 round | S is chunk-exit state; no global state commit; reuse temporary LDS only after producer/consumer barrier |
| Next chunk | All carry fragments complete | Increment c0 by64 inside same launch | No EF fold, frame reservation, host launch, or Q8 round trip here |
| Commit | Last chunk S, no more delta/Gram use | Promote S; add old f16 EF exactly once; max over all128 key columns using lane and wave-pair reductions; same max/127, rint/clamp and f16 residual arithmetic | One owner writes each code/residual and one owner each row scale; **zero-row scale is1**, inv_scale0, same as fast source |
| Return/cache | Codes/scales/EF fully committed on existing stream | Existing prefix snapshots and decode consume the same owners | Conv history unchanged by scan; snapshot/restore must include existing full tuple; no new mutable cache object |

The quantizer arithmetic is `/home/kaden/ClaudeCode/warpfront/wt-gdnreg/kernels/src/gated_delta_net_q8_fast.hip:427-465`. Preserve its f32 multiply/subtract rounding choices, not only its formula. A changed reduction tree is harmless to finite absmax but the numerical scan state is not bit-identical. Candidate commit must be byte-identical to the shipping epilogue **given the identical precommit f32 S and EF**, including all-zero rows and clamp endpoints.

## 4. LDS, occupancy, packets, and bytes

### 4.1 A phase-overlaid LDS arena, not an S arena

One arena, maximum **30,720 B**, and no allocation for S or full delta:

| Phase | Storage live | Bytes |
|---|---|---:|
| KK | full K f16 16,384 + strict-lower L f32 8,064 + gate vectors1,024 | 25,472 |
| QK | full K f16 16,384 + lower-including-diagonal M f32 8,320 + Q16 panel4,096 + vectors1,024 | 29,824 |
| S·K or S·Q | expanded L/M f32 16,384 + Q/K16 panel4,096 + pair partial exchange8,192 + vectors1,024 | 29,696 |
| Solve | expanded column-major L16,384 + vectors1,024 | 17,408 |
| Carry/commit | dead Gram space reused for operand panels / row maxima | ≤29,696 |

Compact construction writes strict triangular elements; expansion writes a *disjoint* 16-KiB destination over the now-dead K plane, then synchronizes before reclaiming the compact source. It is not a racy in-place triangular expansion. During QK, one Q16 panel gives four useful tile-computing waves; other waves do not overlap WMMA with supposedly free VALU. The packet-floor probe does not model this utilization loss; it is charged to the missing-phase allowance and complete-kernel gate.

Freeze the two expansion layouts separately: L is column-major `[source_token][target_token]` for the coalesced solve broadcast; M is native A-fragment order `[output_tile][source_tile][lane][j]` with f32 elements, converted to half only in operand registers. The two layouts occupy the same16-KiB arena at different times. Reusing L's column-major layout as strided half8 M gathers is not admitted. K uses token-row fragments for Grams/pulls; carry stages the transposed K fragments into phase-dead panel space. These are operand panels, never an S spill.

Exchange8,192 = eight waves × 32 lanes × eight f32 accumulator elements. The tempting 4-KiB estimate is wrong. Phase reuse must be expressed with explicit offsets and barriers; padding is permitted only within the 30,720 cap.

Occupancy arithmetic: floor(65,536/30,720)=2 WGs; three need92,160 B and cannot fit. Probe at28,672 reports2 active WGs with124 VGPR; its zero-LDS no-solve counterpart reports7, so the candidate proof is not “VGPR alone gives two.” Real resource metadata and occupancy API at the **final** arena size remain the admission gate. Compile-time waves/SIMD remarks are not a substitute for LDS-aware occupancy. Two resident WGs is capacity, not achieved utilization:96 WGs over32 reported MPs imply at least two scheduling rounds at capacity64, with a partial second round.

### 4.2 Packet budget

Dense algebraic WMMA counts per head/C64:

| Product | Full-state geometry | Conservative split-2 |
|---|---:|---:|
| KK + QK | 256 | 512 (duplicated between value WGs) |
| S·K | 256 | 256 |
| S·Q | 256 | 256 |
| M·delta | 128 | 256 (key-pair duplicate, one output writer) |
| carry K^T·delta | 256 | 256 |
| **Total** | **1152** | **1536** |

Probe selected dynamic census includes the compiler's two32-row solve iterations: **7044 packets/wave/C64**, of which96 are WMMA,2048 f32 FMA,2112 ds_bpermute and64 ds_load_2addr_b32. Across16 waves/head this is **112,704 packets**, versus shipping serial524,288. The ratio4.65× is a packet-count ratio, **not a speedup claim**. The solve consumes most packets; counts that stop at the96 WMMAs are misleading. Scratch traffic is zero. Actual staging, gate exponentials, paired reductions and epilogue must be added and timed; source-level operation counting cannot waive the150k packet gate.

The solve broadcasts each column from two coalesced LDS vectors, rather than retaining32 per-element LDS addresses and masks. The earlier naive unrolled solve hit256 VGPR and hundreds of scratch bytes/lane. This is the probe's demonstrated reason to bound operand/selection liveness. Do not mechanically copy the unrolled source without checking emitted trip counts and ISA.

### 4.3 Byte budget

For one H48/N512 launch with expanded Q/K, actual unique tensor bytes are:

- Q/K/V read and output write: 50,331,648 B.
- Gate/beta:196,608 B.
- Q8+scale+EF read/write:4,767,744 B.
- **Total55,296,000 B**, before any repeated panels or cache-line amplification.

The ~85-GiB/s assignment figure is analytical effective bandwidth, not a DRAM counter. At an assumed400–500 GB/s effective unique traffic,55.3 MB costs111–138 µs; this is an assumption, not a measured roof here. Extra panel fetches can be cache hits but still issue packets.

Shipping ISA requests eight global32 Q/K loads ×32 lanes ×4 B ×1536 waves ×512 tokens =805,306,368 B at the instruction interface, before scalar V/gate, output and state; these are **not**805 MB of measured DRAM. Split-2 stages shared full K for Gram construction and16-row panels for state products, avoiding the serial32fold value-tile reread. Freeze an implementation accounting gate of **≤256 MiB global-request bytes/512-row launch** and report the actual repeated panels separately from55.3MB unique bytes. No extra global T/U/W/delta workspace is allowed. A layout that reads one pair of global operands per WMMA instead of staging can lose this reduction and is not the proposed design.

## 5. Exact source integration and cross-architecture boundaries

All source paths here are rooted at `/home/kaden/ClaudeCode/warpfront/wt-gdnreg/`.

- `crates/rdna-compute/src/kernels.rs:6697`: existing `GATED_DELTA_NET_Q8_FAST_SRC` remains unchanged. Add `GDN_REGISTER_SCAN_Q8_GFX1201_SRC` for the new source; do not modify shared fast source or compact/decode macros.
- `crates/rdna-compute/src/norm.rs:3060`: existing `Gpu::gated_delta_net_q8_batch_seq` remains unchanged, including `reserve_gdn_requant_frames` at3113, launch_maybe_blob and its thirteen-argument packing. Add separate `Gpu::gated_delta_net_q8_register_scan_gfx1201` with the same Rust argument list, required EF, resource constants above, bind/ensure/launch/profile conventions, and exactly one N-frame reservation. No “try candidate, then serial after launch failure” mutation path.
- Use the existing experimental `hipfire_config::developer_var` convention used by `norm.rs:gdn_chunked` (line65): new default-OFF predicate `gdn_register_scan()` for `HIPFIRE_GFX12_GDN_REGISTER_SCAN=1`. This experimental plan does not introduce a second persistent FeatureFlags/config schema. Promotion into stable config requires a separate decision after admission.
- `crates/hipfire-arch-qwen35/src/qwen35/prefill.rs:6096` `batch_chunk_delta_net_attn`: add a final `register_scan_admitted: bool` authority parameter. Only its Q8 sequential branches at6309/6325 may select the new wrapper; retain unsliced S/scales/EF and exactly the current per512 views. Never infer authority in the generic norm wrapper.
- Compute that authority inside `forward_batch_chunk_impl` near10878 using Standard workload, Sequential semantics, dense target shape, no hidden ring, per-token hidden capture, tape, tree, band, routed output, mask override, max-layer limit, pre-upload/pre-embedded vehicle, or DFlash fusion; require no capture and no retained recording plus exact gfx1201 and Q8+EF/single-end mode. The candidate is selected only when the actual segment is64..512 rows. Pass it at11007.
- The paired caller `prefill.rs:10372` passes **false** in this plan; preserve the pair's two legacy commits. The TP caller `qwen35/forward.rs:5513` passes **false**. These are all three located callers of `batch_chunk_delta_net_attn`; they must be migrated together. `forward_batch_chunk_impl` itself needs no new parameter, so its batch/EP caller ABI stays unchanged.
- Generic wrapper callers remain intentionally unchanged: `prefill.rs:8937` MoE, `forward_slots.rs:980,1293`, `speculative.rs:1851`. Independent masked/full paths, tree/tape paths, compact-QK decode, replay/AQL, and F32/Q4 state routes remain serial/current.

Reference search was attempted with the ready rust-analyzer but returned no references for the known norm symbol in this worktree; discrepancy reported to the tool. Text search established the call sites above. Composer must obtain a functioning worktree-scoped LSP reference set before exported-symbol changes; do not treat the empty result as proof of no callers.

Cross-architecture: gfx1100/gfx1151/gfx1200/gfx942 and unknown devices never compile/dispatch the gfx12-only intrinsic kernel through this route. Existing kernels, graphs, launch ABIs, numerical pins and replay state machines are untouched there. EF absent, per-token requant enabled, decode N1, tree, independent slots, TP/EP, retained recording/capture, banded/hidden-ring/verification modes all stay on the established route. Prefix-cache storage format and rollback snapshots remain unchanged; the scan is not admitted into retained transactions. Restoring a scan-produced committed state and continuing serial decode must work through the existing tuple ownership.

Non-goals: preamble fusion, GEMM/FA changes, compact QK production, f16 L/M, approximate exp/rcp, explicit inverse, two-pass global workspace, multi-stream scheduling, kernel fusion across layers, dropping/widening a512 commit, new state formats, or reproducing competitor code. The competitor shape is an algorithmic hint only.

## 6. Composer slices with frozen interfaces

These are bounded **one-file ownership** slices, not permission to run shared builds mid-flight. All workers skip formatters, linters and project-wide tests. Integration owner stages shared dependencies once and runs validation after the batch. Kernel/resource and oracle work may execute independently; host integration waits for the kernel gate rather than consuming an unmeasured promise.

| Slice | Exactly owned file | Deliverable and mechanically checkable gate |
|---|---|---|
| K | `kernels/src/gdn_register_scan_q8.gfx1201.hip` | One thirteen-argument entry, fixed primary geometry, transitions/arena above. Standalone compile+ISA metadata ≤128 VGPR,0 scratch,≤30,720 LDS; initialized kernel ≤0.75×serial and≤427.5µs. No host edits. |
| O | `scratch-2026-09-17/GdnRegComposer/oracle.py` | Throwaway host oracle/driver with frozen ABI, independently compile/load K when available; f64 algebra, f16 shadow, Q8+EF seam/snapshot checks below. No test-suite boilerplate or production dispatch. |
| R | `crates/rdna-compute/src/kernels.rs` | One new include_str constant with exact frozen name; existing fast/compact source constants byte-unchanged. Integration owner supplies K before shared build. |
| N | `crates/rdna-compute/src/norm.rs` | New default-OFF developer predicate and separate launcher, same public argument shape and frame reservation. Existing batch_seq unchanged. One grid[48,2,1],block256,arena30,720 launch; no allocation/workspace or fallback after mutation. |
| P | `crates/hipfire-arch-qwen35/src/qwen35/prefill.rs` | Authority parameter and exact two sequential dispatch locations; caller11007 derives eligibility, pair10372 false.512 views/owners unchanged. Existing noncandidate branches remain behaviorally identical. |
| T | `crates/hipfire-arch-qwen35/src/qwen35/forward.rs` | Caller5513 supplies false to the frozen new last parameter; no TP behavior change. |

R/N/P/T can be prepared against frozen interfaces without editing the same file. Their combined integration is conditional on K/O, not a default enablement. No placeholder source is landed to make a partial build compile. Reviewers own the final veto. The docs owner updates this plan with measured admission/rejection and removes only throwaway composer artifacts after proof; planning receipts in E remain evidence.

## 7. Correctness and claim-scoped validation

### 7.1 Algebra, precision, and state boundaries

1. CPU f64 recurrence versus chunk algebra at N1,63,64,65,129,511,512; orthogonal, repeated and nearly collinear normalized K; beta0/1; zero/strongly negative gates; nonzero initial state. Check every output and precommit S. The prior double-transform bug is defended by the delta equation residual, not only a loose output tolerance.
2. GPU candidate versus a CPU/FP32 shadow with **the candidate's explicit half conversion points**. Check finite values, f16 overflow/underflow statistics, output normalized RMSE≤0.002 and max absolute error≤0.01×max(1,max|serial output|), analogous precommit-state checks. These are proposed screening thresholds, **not passed results** and not a substitute for KLD. Do not require “no worse than f32 serial versus f64”: that is generally incompatible with intentionally half-rounded state.
3. Commit given the same f32 precommit S/old EF must exactly reproduce Q8 codes, scales and f16 EF from the shipping epilogue. Include zero rows, saturation/ties, nonzero EF, and row-max ownership across both key halves. A value half may not use a half-row max.
4. Launch512 twice versus the same two512 view calls from a widened1024/4096 parent: identical candidate outputs and all S/scales/EF bytes at every512 seam; frame counter increases1024 in either case. An eight-C64 internal loop must not fold EF eight times. Compare commit against independently quantizing the downloaded precommit S, not merely comparing two candidate runs.
5. Guard regions and adjacent heads/splits/slots unchanged; ragged masked positions do not alter state. Snapshot/restore S/scales/EF+conv, then continue candidate and serial decode from the restored tuple: restored continuation must match its own unsnapshotted route byte-for-byte. No prefix-cache owner changes or retained capture side effects.

### 7.2 Model admission

Use the exact same model/eval binary/tokenization/cache settings in ON/OFF arms; do not substitute old document pins. Assignment's current flag-free pins:

- WT2 c1 `9fd99341d2e044b83de8e463c41e0491`
- WT2 c2 `1cadeca9bf2adb72b7e59922fa87f33c`
- WT2 c24 baseline KLD **0.048028**
- iu4 route c24 baseline KLD **0.063890**

Flag-OFF must retain its established pins exactly. Flag-ON is **KLD-gated, never md5-exact**: WT2 24-chunk KLD≤matched route OFF+0.0005 on both routes, hence expected ceilings **0.048528 fp8-v2** and **0.064390 iu4** when matched OFF reproduces those baselines. Record ON c1/c2 outputs but do not replace OFF pins. Any baseline mismatch is investigated as stack provenance, not repinned. f16 S and f16 delta WMMA casts add more rounding than the old f32-state scan; its oracle/KLD pass cannot be inherited.

Negative-route exactness checks: ordinary flag-OFF, N1 decode, EF-off, per-token requant, gfx1151/gfx1100, tree/tape verify, independent inactive lanes, TP caller, pair caller, capture/retained recording. Compare actual dispatch/output/state, not source strings. Run only selected hardware-supported cases; unavailable cross-architecture runtime gates remain explicitly unpassed, with reviewer veto rather than inferred success.

### 7.3 Performance claim gates

Actual HSACO ISA + HIP-event kernel-only A/B + matched unprofiled wall are required. Instruction PMCs reading zero on gfx1201 are not evidence. Trace must confirm768 scan launches at pp8192 (48 LA layers×16 commits),3072 at pp32768, no extra workspace kernels, and the identical commit cadence. Record VGPR/LDS/SGPR/scratch, residency API, all timing distributions, input dimensions and artifact hashes. No raw-WMMA floor, packet ratio, occupancy capacity, or f64 algebra number is a full-model speed/correctness claim.

No full-model evaluation, project-wide tests, formatter, linter, or project build was run for this planning deliverable. Executed evidence is limited to the isolated GPU probes, shipped-ISA extraction/census, CPU algebra, and resource/byte arithmetic documented above.
