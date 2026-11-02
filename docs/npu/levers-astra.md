# Plan A: V9 resource model and exact-GEMM levers (historical planning)

> **Historical research note: superseded by the 0.4.2 serving route.** This file records an independent planning analysis (no hardware run; arithmetic evaluated locally) of the V9 resource model and exact-GEMM levers from the 2026-10 NPU research cycle. It is not current guidance, and any figure in it is a dated fixture-bound measurement or an explicitly labelled historical estimate, not a product claim. The shipped 0.4.2 NPU route is the Flash-Next gfx1151 trunk input projection described in [npu-trunk.md](npu-trunk.md): `HIPFIRE_QWEN4_NPU_TRUNK=off|auto|on|emulate` (default `auto`, effectively `off` on every architecture except gfx1151), calibration schema 4, BF16-direct only. There is **no I32 serving contract** (no int32 partial, combine or fold exists in the product route). The out-projection program exists but is **not wired** into serving; `qsa.o` and the GDN o-projection are **not shipped**. Environment-variable names and flag syntax below are historical proposals or experiment switches, not supported configuration.

Planning only, 2026-10-02. Sources were read from a research checkout; no code, driver, host, or hardware changes and no project tests were run. Arithmetic below was evaluated locally, not measured on an NPU. Source paths are relative to this repository. `report.md` measurements take precedence over source comments that still describe older experiments. TOPS counts useful `2*M*N*K`, including neither padding nor repeated timing-probe work (`gemm_array.rs:1448-1453`). MiB means 2^20 bytes; GB/s means 10^9 bytes/s (`crates/npu-tools/src/bin/npu-bw.rs:9-15`).

## 1. What is established, and what is not

The latest exact V9 results are gate_up **25.36 busy / 25.19 wall TOPS**, down **23.93 / 23.46**, and the unpadded 4096×2560×2560 control **33.41 / 32.77** (`report.md:434-442`). V9 no-compute results are **31.21 / 25.19 TOPS-equivalent**, not equal to the gate_up compute result. The equality between no-compute and compute belongs to **V8**, not current V9 (`report.md:329-338,371-374,442`). Thus “DDR alone binds everything” is not an adequate description of current V9. Down is close to its data-movement floor; gate_up has a substantial extra overlap/core penalty.

The measured read-only sweep plateau is **55.4 GB/s**, versus **53.8 at 4 columns ×2 channels**, **53.7 at 8×1**, and **11.8 for one channel**. Write-only is **54.2**; one simultaneous equal-volume point is **33.6 read +33.6 write** (`report.md:329-331`). These establish a tested path plateau, not the physical location of that bottleneck and not a universal mixed-read/write bandwidth function. The backing-path track is testing backing types, page sizes and shim attributes. Until its table lands, “NPU fabric port” remains a plausible explanation rather than a demonstrated exclusion of every software/backing limitation.

Important inconsistency: `gemm_array.rs:1511-1527` and `bw_probe.rs:17-24` model a channel at **4 B/AIE cycle**. At measured **1.80 GHz**, that is **7.2 GB/s**, below the measured single shim channel **11.8 GB/s**. The model is therefore **not a valid hard shim-channel ceiling**. Distinguish the shim measurement, the assumed core-stream rate, and actual bank/stream arbitration. Do not infer clock-domain widths from a BD word being four bytes. Exact memtile-port and stream-switch silicon ceilings are not present in the inspected evidence; inventing them would make this plan less rigorous.

## 2. Exact traffic algebra at the two expert shapes

Source geometry: a core computes 128×64, K chunks are 64, and 32 cores cover 512×512 per wave (`crates/peacemaker-npu/src/kernels/gemm_core.rs:77-82`; `gemm_array.rs:38-85,513-549`). Let `MW=ceil(M/512)`, `NW=ceil(N/512)`, `kc=K/64`, `W=MW*NW`, `Q=W*kc`. V9 executes N-wave outer and replays B across all M-waves (`gemm_array.rs:87-126,525-536,953-974`).

Exact bytes per submit:

- DDR A: `8*4096*Q`; DDR B: `8*4096*kc*NW`.
- DDR C: `32*8192*W`; this includes padded C, even though `unpack_c` discards it.
- Memtile→array: A `8*4096*Q`, B the same; each injected stream broadcasts to four core DMA consumers. Therefore the 32 core DMAs receive four times the injected A/B bytes, not four times the DDR bytes.
- Memtile DMA memory accesses: ingress A+B, replay A+B, ingress C, drain C = `DDR_read + 2*(8*4096*Q) + 2*DDR_C`. This is a byte-accounting identity, **not** a claim those requests share one port.

These follow `Geometry::{a_segment_bytes,b_segment_bytes,c_bytes,pack_a_pair,pack_b,pair_c_offset}`, `pair_column_circuits`, and `emit_shim` (`gemm_array.rs:523-555,574-598,647-674,773-833,1291-1334`).

| Derived quantity | gate_up: 4096×1280×2560 | down: 4096×2560×640 |
|---|---:|---:|
| Useful operations | 26,843,545,600 | 13,421,772,800 |
| MW / NW / kc / W / Q | 8 / 3 / 40 / 24 / 960 | 8 / 5 / 10 / 40 / 400 |
| Padded N; useful compute fraction | 1536; 5/6 | 2560; 1 |
| A / B / C DDR MiB | 30 / 3.75 / 6 | 12.5 / 1.5625 / 10 |
| DDR read / write B per useful op | 0.001318359375 / 0.000234375 | 0.0010986328125 / 0.00078125 |
| Useful ops per DDR-read byte | 758.5185 | 910.2222 |
| Memtile→array A+B MiB | 60 | 25 |
| Core DMA A+B received MiB, after broadcast | 240 | 100 |
| Total memtile DMA memory-access MiB | 105.75 | 59.0625 |
| Same memory accesses B/useful op | 0.004130859375 | 0.0046142578125 |

Per column, per wave, the memtile sources `kc*4096` A and the same B, accepts four 8192-byte C tiles, and drains two 16384-byte C streams. At the **assumed** four-byte core-stream rate, the A and B channels each require 1024 cycles/chunk; each core C channel requires 2048 cycles/wave; each memtile C drain requires 4096 cycles/wave. Dedicated circuits prevent master collisions, but broadcast backpressure couples consumers (`gemm_array.rs:773-833`, especially `814-817`). More buffering can reduce that coupling; it does not change any byte count above.

Core-local traffic is much larger than DMA input: each later chunk has 512×64-byte A vector loads, 1024×64-byte B vector loads, 512×64-byte C-quarter loads and 512×64-byte C-quarter stores: **160 KiB/core/chunk**; the first chunk omits 32 KiB C loads. One epilogue reads 32 KiB and writes 8 KiB. Derivation: the `group=0..32`, `dm=0..4`, `k=0..8`, `quarter=0..4` loops in `gemm_core.rs:413-445`; 64-byte source vectors and quarter sizing in `docs/gemm_i8.md:9-12` and `gemm_core.rs:798-819`. Repeated A loads by both pair cores contend on the same physical owner, despite DDR multicast savings.

## 3. Resource roofs, and the resource actually evidenced as binding

### DDR/backing path

For a measured envelope `Br,Bw`, a necessary condition is `T >= max(R/Br,C/Bw)`. A mixed envelope must be measured at the GEMM's read/write ratio; do not add the independent maxima or assume the equal-volume mixed point is a global constraint.

| Derived roof/requirement | gate_up | down |
|---|---:|---:|
| Read-only plateau roof `55.4e9*ops/R` | 42.02 TOPS | 50.43 TOPS |
| Write-only plateau roof `54.2e9*ops/C` | 231.25 TOPS | 69.38 TOPS |
| At 50 useful TOPS: required read +write | 65.918 +11.719 GB/s | 54.932 +39.063 GB/s |
| At observed exact busy rate: read +write | 33.434 +5.944 GB/s | 26.290 +18.695 GB/s |
| At observed no-compute rate: read +write | 41.146 +7.315 GB/s | 27.675 +19.680 GB/s |

All rows are arithmetic from §2 and `report.md:330,436-442`. The directional roofs assume the existing measured plateau continues to apply. Their being above the measured no-compute rates demonstrates overhead/backpressure; it does not identify where it occurs. Treating 67.2 GB/s as a *hypothetical* shared total would give 43.28/35.75 TOPS, but the equal-volume measurement does **not** justify that bound for either actual ratio.

### Core and array cycles

Peak arithmetic is `32*512*2*1.8e9 = 58.9824 TOPS` (rounded 59.0 in `report.md:308`; source `gemm_array.rs:720-732`). Current gate_up geometry therefore has an unavoidable **49.152 useful TOPS** MAC ceiling. **50 is impossible without eliminating/reassigning its padding work**, even with infinite DDR bandwidth and zero synchronization overhead.

The checked-in straight-through pair schedule averages `(1131+1123)/2=1127` cycles/chunk, plus a 530-cycle int8 epilogue. The existing source estimate is `W*(kc*1127+530)+4096` for these shapes (`gemm_core.rs:699-705`; `gemm_array.rs:1528-1547`). It excludes startup, bank stalls and most lock waits; it is not a guaranteed achievable rate or a strict silicon bound.

| Derived cycles/rate | gate_up | down |
|---|---:|---:|
| Pure MAC / assumed input-stream cycles | 983,040 | 409,600 |
| Existing optimistic issue-model cycles | 1,098,736 | 476,096 |
| Existing optimistic issue-model TOPS | 43.98 | 50.74 |
| Observed busy cycles at 1.80 GHz | 1,905,299 | 1,009,578 |
| Observed no-compute cycles | 1,548,170 | 959,079 |
| Cycle budget at 50 useful TOPS | 966,368 | 483,184 |

The measured repeat-compute slope of about **1357 cycles** is a compute-region marginal cost, not full normal chunk time (`report.md:413-421`). Multiplying only this slope by Q gives optimistic compute-only 37.09/44.51 TOPS. Adding an estimated 60 cycles of pair control (`1127-1067`, `gemm_core.rs:11-13,699-702`), epilogue and drain gives an illustrative 35.09/40.80, **not an independent measurement**. After eliminating gate_up padding to 20 full useful waves, the same issue model becomes 52.73 TOPS, and its chunk-period budget at 50 is only **1189.59 cycles**, versus **1144.72** for down. Both require getting close to the no-stall issue schedule.

The report's `1470*Q +5800*W` fit used **two different designs/shapes** (`report.md:457-460`). It is descriptive arithmetic, not independent identification of a chunk cost and a wave cost. Do not call 5800 cycles of overhead removable or extrapolate it into a promised gain. A fixed-variant two-dimensional M/K sweep is needed before using that fit causally.

### Attribution summary

- **DDR read/backing:** measured aggregate plateau; V8 dominated by it, current down remains near the complete DMA-path floor. A faster read path alone cannot remove down's output/mixed-traffic and array constraints.
- **Shim channels:** V9 already uses all eight A MM2S1 and eight B MM2S0 channels (`emit_shim`). 8×1 and 4×2 nearly saturate the read sweep, so simply increasing channel count is not a 2× lever. No valid 4 B/cycle hard shim ceiling is established.
- **Memtile DMA ports/banks:** replay does not remove array-side bytes. V10 down removes 10 MiB DDR read yet is slower (**22.99** vs V9 **23.91** same-window) and no-compute reaches only **29.5** (`report.md:444-460`). There is an array/startup/overlap limit, but these measurements do not distinguish memtile SRAM arbitration, lock gating or downstream backpressure. Total-access demands at 50 would be 206.54/230.71 GB/s spread across eight memtiles; this is not a measured SRAM limit.
- **Stream switch:** isolated lane utilization looks compatible with arithmetic peak under the source's four-byte model, but the slowest broadcast receiver stalls its peers. Existing simulator issue timing does not establish silicon backpressure/bank performance (`gemm_core.rs:54-70`). Route-only changes move a coupling/overlap term, not the arithmetic or DDR-byte roofs.
- **Core:** measured bank/layout improvement is real (+11% gate_up); 1357-cycle slope still exceeds the 1067/1070 issue region. Eight single-port banks and peer-owner sharing are sourced in `gemm_core.rs:37-70`; precise AIE2P 512-bit beat occupancy is explicitly undocumented there. It is not defensible to calculate a unique exact physical bank ceiling from the load-only compiler scoreboard.

## 4. Ranked five levers and kill experiments

Ranking is expected information/value, not a promise of gain. Every hardware experiment is a **single ≤90 s window**, compiled and simulator-gated beforehand, with a hard external deadline; no historical ≤240 s M4 permission is inherited. Any mismatch, hang, unproved memory overlap, or invalid rule-gate program kills the candidate immediately. Exact results must include first/final CPU comparisons; timing-only probes must be labelled non-GEMM.

### 1. Resolve backing-path bandwidth; adopt a faster path only if measured

The backing-path track owns SHMEM versus HIP VMM host dma-buf versus device allocations migrated to GTT, userptr/page-size and shim-attribute probes. Frozen payload contract: identical packed A/B/C bytes, same V9 PDI/TXN/core, backing selection only. Relevant boundaries: `bw_probe::BwConfig/design` (`bw_probe.rs:85-107,317-329`), `npu-bw::run_config` (`npu-bw.rs:51-121`), `Device::import_dmabuf` (`crates/railgun/src/npu/mod.rs:342-380`), and argument creation in `crates/npu-tools/src/bin/npu-gemm.rs:563-590`.

**Ceiling moved:** measured DDR read/write envelope, not bytes/op, padding, core cycles or broadcast coupling. With unchanged V9, a new path must deliver at least the §3 target pair; even then gate_up cannot reach 50. If only A changes backing, B and C remain bottlenecks and allocation/copy/packing time belongs in end-to-end cost.

**Kill:** one bounded backing comparison on pure read, exact writes, relevant mixed ratios, and V9 real/no-compute, followed by same-clock readback. Reject an apparent win that only changes page warmup, bypasses correctness, migrates to GTT under a different label, or disappears under the exact workload. No new driver patch without approval.

**Branch:** faster existing path → use it and remeasure overlap; driver change required → source-supported proposal only, continue levers 2–5; all paths plateau → stop backing hunting and prioritize traffic/reuse plus overlap. Single-channel anomaly must be reconciled in the bandwidth model rather than hidden.

### 2. Reduce broadcast backpressure with deeper A staging, then a three-slot core ring

First bounded experiment: V9-only memtile A depth four, unchanged host layout/core/routes. At gate kc40, existing resident memory ends at `0x64000`, leaving **112 KiB** (28 four-KiB regions); source `R_B_OFF`, two B slots, `V9_MAX_KC` (`gemm_array.rs:317-333`). Do not use this capacity without deriving limits at other K. Four slots can retain A offsets 0/0x2000 and add two tail slots if available. Explicit A producer/consumer BD rings need IDs not occupied by B replay, C or the opposite parity bank; `nw_outer_operand_descriptors`, `resident_c_descriptors`, `memtile_locks`, `ring_tasks` (`953-1207`) own them. Fall back/reject at unsupported kc/MW; never overlap C/B.

Then, independently, three core A+B slots: add A at 0x1000 and B at 0x3000 to layout 0's existing map (`gemm_core.rs:129-134`), consuming its last eight KiB. **This puts third B into the A double bank**, so a deeper queue might worsen conflicts. Three full per-slot A/B/peer lock sets plus C need 20 locks, exceeding the 16-lock core namespace (`gemm_core.rs:102-120`); use ordered counting semaphores, not imaginary extra locks. Keep C at 0x8000 and Cout at 0x6000, same numerics and output order.

**Ceiling moved:** waiting/overlap only. Neither depth change alters §2 traffic or the §3 read roof. If only compute overlap improves and the no-compute path stays fixed, the present empirical target is at most **31.21/25.19 equivalent TOPS**, not 50. Since the ring itself can change no-compute time, that number is a comparison target, not a hard future ceiling. Any new target still obeys the 42.02/50.43 read roofs, 49.152/58.982 MAC roofs and actual core cost.

**Kill:** simulator delayed-consumer and odd/even/modulo-three restart cases first; then one window compare baseline, memtile-depth-four, core-depth-three separately, each exact and no-compute. Drop any candidate with no repeatable reduction in exact submit time beyond same-window baseline variation, even if its timing-only probe improves. Do not combine knobs before attribution.

### 3. Remove core bank conflicts by phased ownership/scheduling, not another blind base sweep

Boundaries: `wavefront`, `chunk`, `pair_a_bases`, `pair_acquire`, `pair_release`, `program_pair` (`gemm_core.rs:413-517,709-778,855-896`). Preserve increasing-K signed accumulation, first-chunk overwrite and final floor/saturate SRS. Candidate is a **balanced** peer phase/order change: lower/upper visit different independent output blocks while retaining the same published A lifetime, reducing simultaneous owner-bank requests. A one-time startup delay is not evidence of steady-state staggering: peer barriers can resynchronize it every chunk. Reordering block visits must restore the existing C memory/stream order.

**Ceiling moved:** marginal chunk compute cost from measured ~1357 toward issue-region ~1067, not below the VMAC work. Existing no-stall full-model caps remain 43.98/50.74 before geometry changes; memory-path ceilings still apply. Pure bank gains cannot alone bridge 25→50. Exact 512-bit bank-beat modeling is an unresolved input, so simulator collision counts are filters, not predicted speedups.

**Kill:** simulator exact plus issue/rule/bank-trace checks; a single window real + repeat=2/3 on fixed V9 and layout0, with repeat timing clearly non-GEMM. Abandon if marginal cycles do not fall and exact wall time does not improve; avoid a program-size increase beyond the existing **16 KiB** guard (`gemm_array.rs:1676-1678`).

### 4. Shape-aware reuse and padding removal: necessary for 50, higher risk

Down already fits V10: `kc*(2+NW)=10*7=70<=112`; A+B falls to **4.0625 MiB**, B/op **0.0003173828125**, target-50 read demand **15.869 GB/s**, with write still **39.063 GB/s**. Its read-only roof becomes **174.55 TOPS**, irrelevant because array/core/output bind. Existing V10 measured slower, so adopt only after overlap/startup is improved and exact end-to-end time wins (`report.md:444-460`; `gemm_array.rs:128-174,1656-1669`).

Gate_up cannot use current V10: `40*(2+3)=200>112`. Hypothetical all-once read is **13.75 MiB**, but it is **not a legal current design**. A two-N-wave grouped scheme would read A twice rather than three times: **23.75 MiB A+B**, read roof **59.72 TOPS**, with the existing padded C. Capacity is severe: one 160 KiB A tile +two 160 KiB B tiles +one 32 KiB C slot = **512 KiB exactly**. Whole-slot fill/replay serializes A and removes double-C buffering; that naive variant is not a recommended implementation. A per-chunk ordered release/refill scheme or new tile layout needs an explicit BD/queue/lock proof before it earns silicon time.

Padding removal must use useful work on the otherwise idle tail cores, not merely stop half of them. A two-full-wave plus 1024×256 tail remap gives 16+4=**20** fully useful waves instead of 24: pure arithmetic ceiling **58.9824**, optimistic old-period ceiling **52.73**. A stays **30 MiB**, B becomes **3.125**, C **5**: read roof only **42.81**, so memory/reuse must improve too. The tail doubles unique A per wave from 32 to **64 KiB/K64**; eight existing A injection lanes would each carry eight KiB, or **2048 cycles** at the assumed stream rate. That can erase the theoretical compute saving. Any tail proposal must prove the extra A delivery route/ownership, not just count cores.

A homogeneous 1024×256 V9 tiling is worse for current DDR: A×5 +B once = **53.125 MiB**, read roof **26.70 TOPS**. The report's **62.5 MiB** example concerns V8 repeated B (`report.md:340-346`), not V9. Reject a narrow-wave rewrite justified only by removal of padding.

**Kill:** simulator-only capacity/traffic/route/ownership proof first. For grouped reuse, reject if its serialized stage lower bound already exceeds V9's exact time. For tail remapping, reject absent a feasible A bandwidth budget. Only a surviving design gets one baseline/candidate exact+no-compute ≤90 s window; require improvements in useful TOPS and total bytes, not padded TOPS. This lever is required for the 50 aspiration but is not a near-term measured result.

### 5. GPU producer→NPU work scheduling using the existing real zero-copy path

This is a **system throughput/latency lever**, not an NPU roof increase. Existing proof is GPU `hipMemcpyHtoD` into HIP VMM host allocation, synchronize, export/import and exact NPU consumption; it is not a GPU producer-kernel bandwidth measurement (`report.md:376-401`; `hip_dmabuf.rs:202-268`). Baseline concurrency: GPU prefill **5563.3→6126.3 ms (+10.1%)**, NPU **25.60→22.16 TOPS** (`report.md:462-475`). These are empirical baselines, not forecasts for a faster NPU schedule.

Frozen interface: packed A layout is exactly `ArrayDesign::pack_in` / `Geometry::pack_a_pair`, shape/epilogue/version recorded with each buffer; one live imported `Bo` per exported dma-buf. GPU producer completion precedes NPU submit; NPU syncobj completion precedes reuse/overwrite; all consumers precede destruction. CPU coordination is acceptable initially and must be measured. Import alone is not an implicit GPU↔NPU execution fence. `Device::import_dmabuf` owns GEM handle/mapping but not caller fd (`crates/railgun/src/npu/mod.rs:342-380`); HIP storage/VA/fd lifetime extends beyond imported BO and contexts (`hip_dmabuf.rs:17-24,290-305`).

**Ceiling moved:** host staging/producer cost and scheduling contention. Zero-copy alone was **22.65 vs22.70** TOPS in the earlier layout, not a speedup (`report.md:397-400`). A real producer/pack kernel may remove host copies but still writes repeated packed-A bytes; it does not remove the NPU's repeated DDR reads. Measure GPU packing, host coordination, NPU busy+wall, GPU prefill p50/p95 and completed useful expert work. Do not add non-equivalent GPU and int8 NPU TOPS, or claim full-model exactness from standalone GEMM exactness.

**Kill:** a preloaded single ≤90 s window interleaves short GPU-alone/NPU-alone/concurrent observations; use bounded in-flight buffer slots. Reject schedules that fail reuse/coherency checks or do not improve actual completed application work/latency under an explicit GPU-slowdown policy. The report's “241 TOPS break-even” is explicitly inference, not an application acceptance test.

## 5. Frozen state invariants and composer ownership

Common interface stays `design_v9(m,n,k,epi,ctl) -> ArrayDesign`, arguments `[packed A, packed B, packed C]`, signed int8 inputs, int32 accumulation, `clamp(c >> shift,-128,127)` output. CLI callers are `crates/npu-tools/src/bin/npu-gemm.rs:493-497`; `design_pair` calls `program_pair`, builds CDO/TXN and args (`gemm_array.rs:1674-1694`). An integration owner must expose experimental options explicitly and preserve existing V8/V9/V10 selections; no accidental global change. Each slice can be evaluated against baseline independently; integrate only measured winners.

| Slice / owner boundary | Before → transition → after; invariant | Scoped acceptance |
|---|---|---|
| M: backing probe | GPU/CPU producer owns writable input → completed producer and visibility handoff → NPU read-only ownership until timeline completion. Imported BO and exporter remain alive. | Backing-identical traffic; exact write probe; exact GEMM; no migration confused with a bandwidth win. The backing-path track owns this work. |
| A: memtile staging | Empty count D, full0, producer/consumer index0 → ordered fill reserves empty, writes whole chunk, publishes full → ordered replay reserves full, finishes all reads, returns empty. `empty=D-fills_started+reads_finished`; `full=fills_finished-reads_started`. | Existing V9 host/core contract unchanged; byte-exact outputs and per-channel totals under asymmetric backpressure; no C/B overlap or wrong BD bank. |
| C: core ring | A-empty/B-empty=D, A/B-full and peer counters0; sequence0 → fill publishes same sequence to local core; local core advertises peer-ready; both consume same slot `seq%D` → B returned after own last read; A returned only after own last read **and** peer acknowledgement. | One producer and ordered consumer per operand; peer notification neither skipped nor double-consumed; no slot reused early; compare multiwave modulo-D and repeated-context cases. Shared counters rely on strict order, not just aggregate counts. |
| K: phased core schedule | A/B acquired and peer readiness established → compute reads only live input slots and updates the correct C microtiles → all reads/accumulator writes retire before peer-empty/C-full publication. | Exact C including saturation/floor boundary inputs; no stale accumulator from prior wave; branch alignment and five delay-slot rules; program-size bound. |
| G: grouped/tail topology | Output ownership is a disjoint cover of useful M×N; each retained operand has a known consumer count → final use releases storage → next fill may overwrite. | Layout, descriptors, routes, wave coordinates, pack/unpack and output-range coverage change atomically under one owner. No duplicated or missing C, no premature rounding between K segments. |
| H: GPU/NPU orchestration | Free slot → GPU-writing → producer-complete/NPU-readable → NPU-in-flight → NPU-complete/consumer-readable → free only after last consumer. | No overwrite/drop during flight; failed/timed-out work is not marked reusable; preserve current CPU-fenced path as a valid explicit mode, not a silent fallback. |

**Submit/restart is part of every DMA change:** after prior completed submit, halt/reset cores, reset persistent core/memtile channels (not shim), rewrite every stateful resident descriptor because BD iteration `current` survives channel reset, initialize all locks/indices, set PC0, queue rings then finite shim tasks, enable cores last, wait for **both C channels across all eight columns**. Sources: `gemm_array.rs:176-191,1337-1408`. No token may acknowledge a still-writing C stream. No residual stream words may be allowed to prefix the next submit (`bw_probe.rs:32-48`). A timeout requires quarantine/teardown, not assuming completion or rewriting live storage.

Simulator gates reuse `crates/peacemaker-npu/tests/array.rs::{v9_shape,v9_reuses_context_with_odd_and_even_kc_times_waves,v10_shape,v10_reuses_context_with_odd_and_even_kc_times_waves}` (`194-252,266-315`), plus the pair tests and existing ISA rule gate. Extend only behaviorally meaningful scenarios: modulo-three boundaries, backpressure, changed input each submit, poisoned outputs, signed extrema, shift/floor/saturation, non-divisible M/N, max memory/BD constraints. Functional simulation proves exactness and ownership, not silicon speed. Reviewers own veto; caller owns any shared validation after integration.

## 6. Architecture scope and explicit non-goals

Only measured **Strix Halo XDNA2/AIE2P 8×4** behavior is in scope. LLVM's AIE2P subtarget reuses AIE2 bank address-space information (`ref/llvm-aie/llvm/lib/Target/AIE/aie2p/AIE2PSubtarget.h:33-38`), but that is not evidence of identical physical beat arbitration on other AIE generations. `ref/llvm-aie/.../aie2ps/AIE2PSInstrInfo.h:127` specifies 16-byte target alignment; the actual AIE2P silicon rule is enforced locally and measured in `report.md:320-327,423-428`. Do not re-enable hardware ZOL or remove the alignment/delay-slot gates for performance.

On HIP host-VMM-capable Halo use the proven exported-host allocation contract; ordinary `hipMalloc` may migrate to GTT, and unsupported allocation/export combinations must report unsupported rather than silently advertise zero-copy. Driver reference `ref/amdxdna/linux-source-7.0.0/drivers/accel/amdxdna/amdxdna_gem.c:644-677` imports an attachment SG table as SHMEM; `amdxdna_gem.h:74-77` selects userptr under PASID versus DMA address otherwise. Thus the report's host-VA account is the observed configuration, not a universal cross-platform addressing rule.

No vendor compiler; no driver/IOMMU/firmware/clock policy changes without approval; no reduced precision, approximate output, sparsity-based TOPS, K splitting with intermediate int8 saturation, fake DMA-only GEMM, changing expert shapes to improve the headline, or claims of model-quality parity. Retaining expert weights across submissions needs a separate lifetime/invalidation protocol and saves only V9's B fraction; it is not a hidden 2× lever.

## 7. Recommended next 48 hours

- **0–6 h:** finish the backing-path track's matrix and mixed-ratio accounting; reconcile shim-rate assumption; lock baseline exact/traffic/clock evidence. In parallel prepare depth-four memtile and three-slot core state proofs, plus a balanced peer-schedule candidate. No hardware until simulator-exact.
- **6–18 h:** independent ≤90 s kill windows for staging, core rings, and peer schedule; use real/no-compute/repeat probes only for their stated claims. Keep winners only. A faster backing path changes the DDR envelope; a driver-only path produces an approval proposal, not a host change.
- **18–30 h:** remeasure composition of winners and re-fit fixed-variant wave/chunk behavior. Re-evaluate V10 down after overlap changes. Simulator-only grouped-reuse and tail bandwidth/BD feasibility; abandon if the capacity/critical-path proof fails.
- **30–48 h:** integrate only surviving exact candidates; bounded GPU producer/orchestration experiment and concurrency acceptance on completed useful work. Reviewers veto numerical/lifetime/silicon claims. Report useful busy/wall TOPS at both original shapes and GPU prefill impact; if gates leave no legal measured path to 50, say so rather than rounding 49.152 up or crediting a hypothetical memory bandwidth.

## 8. Reconciliation after reading Plan B

This plan was cross-read against [levers-fable.md](levers-fable.md) (Plan B) and critiques were exchanged. The final decision/order is in [levers-joint.md](levers-joint.md); the independent proposals above remain visible rather than silently rewritten into agreement.

- **Adopt for feasibility work:** Plan B's 128×80 core tile / 256×1280 full-array wave removes both gate_up padding and A rereads. Conditional A/B/C traffic is gate_up **10/3.125/5 MiB**, down **2.5/1.5625/10 MiB** (down requires A replay across its two N-waves). That is materially better than Plan A's tail/grouped alternatives. It is not yet a legal implemented design or a forecast.
- **New mandatory wide-wave gates:** each column needs **two independent 5 KiB B streams/chunk**; serializing them on one four-byte lane would cost 2560 cycles and cap arithmetic throughput at **29.4912 TOPS**. Gate memtile B400+C80 leaves only **32 KiB**, not 112, before A staging. With four A injectors, down needs **40 KiB A per injector per M-wave**; double A80+B160+C80=320 KiB fits. Core C40+A8+B10=58 KiB fits only with output overlap or a new output scheme; C-read/compact-store safety does not prove next-wave overwrite safety. Require explicit `C_EMPTY` ownership and a bank-conflict map.
- **Wide-wave startup is material:** whole-B-ready startup at the measured read plateau is **59.148 µs gate /29.574 µs down**. For down, even MAC409600 +startup53233 +epilogue lower bound20480 +last C-drain5120 = **488433 cycles**, above the **483184-cycle** budget at 50 (assumed four-byte C lane). Prefix-ready B or a separately proved retained-B lifecycle, and/or a changed output schedule, would be needed to escape that particular bound. No free startup/epilogue is assumed.
- **Prefer Plan B's iteration-stepped memtile ring over an explicit many-BD ring:** one ordered producer and one ordered consumer can share counting empty/full locks. At kc40, a contiguous depth-eight/sixteen ring fits in the 112 KiB tail. This is different from two readers sharing ready credits.
- **Reject Plan B's proposed `A_FULL += 2` two-reader shortcut:** one fast core can read slot0, slot1, then consume the slow peer's remaining slot0 credit as a fresh generation, before DMA refills it. A local credit-state enumeration reproduced that stale-generation trace; this is a logical counterexample, not a hardware experiment. Keep per-reader notification or prove generation gating. The existing acquire/release functions contain **eight lock operations**, not nine; the nine-step description includes compute (`gemm_core.rs:719-778`).
- **Adopt the concrete balanced-order candidate:** swap the Upper role's E/O A visit order and the matching C row load/store ownership, preserving physical C order. Treat it as a schedule change needing exact and bank-trace proof, not “four safe lines.” A one-time phase delay remains inadequate.
- **Retain disagreement:** Plan B prefers wide geometry over three-slot core buffering; Plan A agrees to demote the latter to a baseline-geometry backup. Link-rate isolation becomes a first-window gate, but four-byte links do **not** logically exclude buffering gains: 7.66 B/cycle required is below two lanes' nominal eight, albeit tightly; per-lane rates and slow-consumer effects must be measured.
- **Scope remains unchanged:** no int4/reference change, unsupported firmware readback, unapproved host changes, or inherited 240-second M4 window. Plan B's 40–45/35–40 TOPS expectations are unmeasured estimates, not joint promised outcomes. GPU orchestration is last in execution order but remains a required deliverable because the goal explicitly includes concurrent prefill.
- **Final peer agreement:** Plan B withdrew shared `+2 FULL`, the physical-width inference, the unsafe firmware-probe safety statement, and the inherited long hardware window. TN80 is now a **gate_up-only** candidate; down should first revisit existing TN64 V10 with earlier B readiness, preserving its separate Cout. For down, the guarded first B pass covers `NW*kc=50` chunks, followed by `MW-1=7` unguarded whole-segment replays only after the first pass proves full residency. TN80's two B streams need separate readiness credits. Both plans accept the joint top-five order; Plan B would spend wide-geometry effort earlier and expects 40–45 gate_up only as an unmeasured estimate.
