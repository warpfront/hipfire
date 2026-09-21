# gfx1201 register-resident Q prefill attention

## Design note (written before implementation)

### Target and incumbent launch

The shipped artifact reports 24 query heads, 4 KV heads, and `head_dim=256`, hence GQA 6. At pp8192 the current gfx1201 packet launcher divides each equal-length 512-row run into `ceil(512*6/128)=24` x workgroups, 4 KV-head y workgroups, and 16 z runs. Each workgroup has 256 threads (8 wave32s), owns 128 query-head rows (21 1/3 query positions), and streams KV in 64-position tiles.

### Implemented tile decomposition

The candidate will use 768 threads (24 wave32s) and own 384 query-head rows: exactly 64 query positions times all six query heads associated with one KV head. The grid is `[ceil(min(batch,512)/64), 4, ceil(batch/512)]`. It retains a 64-position KV tile for the first correctness/performance prototype, so one K/V fill is shared by three times as many query-head rows as the incumbent. Each wave still owns one 16-row WMMA output tile.

Each lane keeps its owned Q row's sixteen 16-dimension E4M3 fragments in registers for the entire KV loop. In the current lane-pair mapping this is sixteen `v2i_t` values, or 32 VGPRs/thread. The sixteen `float8` output fragments cost 128 VGPRs/thread. The explicit long-lived state is therefore 160 VGPRs plus the row scale, softmax state, and short-lived score/fill temporaries. The resource target is at most 255 VGPRs with zero private-segment bytes and zero spills; a spill is a hard failure. The incumbent's 237 VGPR count includes the wave-private V-transpose and bounded Q reload temporaries. The candidate makes room for the 32-VGPR Q tile by limiting transpose ownership to the first eight waves, separating fill and compute lifetimes, removing all Q scratch address/reload state from the KV loop, and replacing the eager scale update's live temporary set with deferred scaling state. If compilation still spills, the Q ownership will be reduced rather than accepting a spilling artifact.

K and V remain in LDS. The candidate keeps the incumbent's exact V transpose, but only waves 0..7 stage/transpose the common KV tile; all 24 waves consume the resulting shared fragments. LDS stays 49,408 B: 16,384 B K plane, 16,384 B V plane, eight 2,048 B transpose regions, and 256 B K/V scale headers. A K64/V64 iteration reads 32,768 code bytes plus 256 scale-header bytes = 33,024 B/workgroup. The raw-V handoff is wave-local because each transpose wave consumes only its own private SCR region. It therefore needs one wave barrier plus two full-workgroup barriers per KV tile (completed K/V plus headers ready, then tile turnover), down from the incumbent's three full-workgroup barriers. Two one-time full-workgroup barriers reduce the 64-query position bounds before the KV loop; they are not per-tile costs.

The f32 Q ABI requires an amax pass and a conversion pass. The candidate retains each converted fragment directly instead of writing/re-reading global Q8 scratch. Q reads are therefore `384*256*4*2 = 786,432 B` (768 KiB) per workgroup per pass, independent of context length, with zero inner-loop Q reads. At pp8192 that is 21.7x fewer instruction bytes per workgroup than the incumbent's 16.25 MiB; at equal 384-row output coverage, three incumbent workgroups read 48.75 MiB, a 65x reduction.

### Resource-driven implementation

The first 768-thread/384-row object fully unrolled all fill and compute loops. It compiled at 240 VGPR, 27 SGPR, 9 VGPR spills, and a 40-byte private segment, so that object was rejected. Intermediate no-spill 576- and 640-thread shapes lost the intended wide-workgroup benefit. The final implementation retains 768 threads and 384 rows but bounds live load-address temporaries: K staging is unrolled four ways, raw-V staging and transpose four ways, and memory compiler barriers separate consecutive QK/PV WMMA fragments. The final object is 216 VGPR, 28 SGPR, 0 VGPR spills, 0 SGPR spills, and 0 private bytes. Its runtime occupancy query reports one block/CU.

### Deferred softmax scaling

The candidate maintains a stable softmax reference maximum and a persistent FP8 probability/value scale across K16 recurrences. Scores are exponentiated against that stable reference. When the reference maximum advances, the denominator and persistent scale are multiplied by the matching factor rather than eagerly touching all 128 output values. The accumulator is rescaled only when the weighted-V range can no longer fit E4M3; ordinary K16 steps add directly without the incumbent's unconditional sixteen `float8` multiplies. This is lazy/deferred online-softmax scaling rather than the incumbent eager per-K16 rescale.

## Implementation and receipts

The opt-in route is `HIPFIRE_ATTN_QRESIDENT=1` / `kernel.attn_qresident=true`; default is off. Admission requires gfx1201, 24 query heads, 4 KV heads, head dimension 256, native FP8 KV, and batch/context within the candidate launcher limits. The regular packet route is unchanged when the flag is absent.

Real-slab exactness used layer 35 of the captured pp8192 input and a CPU FP32 attention reference over four sampled query rows (96 query heads total). The control packet kernel measured max-abs 1.269074440 and relative RMS 0.03222988321; the candidate measured max-abs 1.273281574 and relative RMS 0.03232714207. Candidate versus control was max-abs 0.06932961941, relative RMS 0.003622857630, with zero non-finite outputs in all comparisons.

The standalone full pp8192 launch screen (five warmups, median of 20 HIP-event samples) measured control 11,177.706 us total / 1.364466 us per token and candidate 9,110.523 us total / 1.112124 us per token: 1.226901x faster. This is a compile/exactness screen; the daemon receipts below decide retention.

Final code-object metadata is 216 VGPR, 28 SGPR, 49,408 B LDS, zero VGPR/SGPR spills, zero private bytes, and one resident block/CU. Candidate-symbol ISA has 32 FP8 WMMA instructions, zero F16 WMMA instructions, 68 packed FP8 conversions, 48 global loads, 32 global stores, zero scratch references, and four barrier signal/wait instruction pairs.

### Quality gates

| Route | c2 KLD | c24 KLD | c24 limit | Result |
|---|---:|---:|---:|---|
| IU4 prefill | 0.061176 | 0.078594 | 0.0815 | pass |
| fp8v2 prefill | 0.037087 | 0.045574 | 0.0465 | pass |

The corrected candidate passes both full quality gates. The four-row slab comparison also has no non-finite values and differs from the packet control by 0.003623 relative RMS.

### In-daemon performance and profile

On card-A, a fresh build was warmed once through the first 8192-token pass. The typed config flag was explicitly enabled for the candidate replay; the default-off replay was the control.

| Route | cold/warm tok/s | warm ms | warm delta |
|---|---:|---:|---:|
| packet control | 3522.6 / 3637.3 | 2252.19 | baseline |
| Q-resident | 3679.9 / 3686.2 | 2222.32 | +1.344% |

Contemporaneous `rocprofv3 --kernel-trace --stats` over two pp8192 passes confirms selection of the candidate symbol. The packet control used 396,793,841 ns over 32 calls, or 24.218 us/token after normalizing by 16,384 tokens. The Q-resident kernel used 332,164,926 ns over 32 calls, or 20.274 us/token: 16.29% less attention time, but well above the 13 us/token retention alternative.

### Verdict

**KILL for production retention.** Q residency and the wider workgroup work mechanically: equal-output Q traffic falls 65x, the final object is spill-free, quality passes, and the isolated body is 1.227x faster. The daemon gain is only 1.344%, below the required 1.5%, while profiled attention remains 20.274 us/token rather than at most 13 us/token. The limiting factor is therefore no longer Q rereads alone: the retained K64/V64 LDS staging, V transpose, full-workgroup synchronization, one-block/CU residency, and 16 per-layer launches dominate the remaining gap.

The matrix, TTFT, and serve battery were not run after both performance retention alternatives failed; this branch remains opt-in and default-off. Evidence is retained under `scratch-attnq/`, including the final KLD sequences, profiler CSVs, code-object ISA, and resource metadata.
