> Consolidated from the local `review/vmm-fixes` / `review/vmm-validate` worktrees into the
> VMM multi-slot branch. Times are server UTC unless noted. Working log; the code is the authority.

# VMM paged-KV fix log (review/vmm-fixes, base 01ecc10f)
Times in server UTC unless noted.

## Hypothesis H1 (14:00): hipMemRelease→hipMemCreate race; quarantine 20 ms
- Probe ~/work/probe/vmmclear.cpp: release then create, kernel writes later read 0x0 (mode 4: 80-190/100 bad iterations); sleep before creates, handle reuse, never release: clean. hipMalloc clean.
- Patch: hip-bridge ffi.rs VMM_RELEASE_QUARANTINE (uncommitted; unit tests pass).
- RESULT 14:19-14:21: FAILED. F7H (vmm+hip seq:30) 42/60 wrong (SB2 control 9/60); F7 (vmm pm4 seq:30) 32/60 wrong, shorts wrong too. NOT committed.
- Interpretation: corrupt value is 0x0 (wipe-on-release poison would be non-zero) => clear-on-ALLOC fence of a new BO lands after user-queue writes. Release timing only changes whether the new blocks are dirty.

## H2 (current): wait for the new allocation clear fence via dma-buf poll(POLLOUT)
- Probe mode 128 = export handle as dmabuf fd, poll POLLOUT before first use. Next: run probe modes 4/516/132.

## 14:30-14:45 probe results (vmmclear, 100 iters x 8 maps x 2 MiB, mode bits in source)
- dmabuf poll(POLLOUT) before use (132): still bad, poll returns at once => no pending BO fence. H2 dropped.
- Fresh VA every iteration + release (1028): CLEAN. Same VA, handles leaked, never released (2052): CLEAN via same VA.
- Same VA reused, no release, readback through a FRESH alias VA of the new handle (6148): 736/800 bad, all zero.
  => kernel writes after hipMemUnmap+hipMemMap of a REUSED VA go to the OLD physical page (stale GPU translation),
     the new allocation never sees them. Not a clear/wipe race.
- Fresh VA + alias readback (5124): clean.
- Remedies that do NOT help: device sync after unmap, unmap+remap twice, dummy VMM map, re-SetAccess, hipHostRegister.
- Time helps: >=200 us sleep after map clean; hipMalloc(64MB+)/hipFree after map clean (likely just time).
## H3 (current): reused-VA remap leaves stale GPU TLB/PTE for some time (async invalidation on gfx11 KFD).
- Serve test queue15: T7 trace timestamps, G7/G7H with HIPFIRE_DIAG_VMM_NO_UNMAP=1 (never unmap => never remap a VA).

## 2026-10-28 14:50 UTC (18:50 Dubai) — no-unmap confirmed; mechanism narrowed
- T7 (vmm pm4, trace): 24/60 wrong. Granules 1,2 unmap->map every request (idle reclaim between seq requests), unmap->map gap 0.146-1.1 s.
- G7 (vmm pm4, HIPFIRE_DIAG_VMM_NO_UNMAP=1): 60/60 correct. G7H (vmm+hip, NO_UNMAP): 60/60 correct. kfaults=0.
- Probe (vmmclear_busy, 100 iters x 8 x 2MiB):
  - busy GPU (non-blocking spinner) does NOT stop healing: busy+sleep-after-map 1/5/50 ms => 0 bad; busy+hipMalloc/free => 0.
  - deferred release (release old handles one iteration later) => still bad (111). reuse handles (mode 36) => 0. leak (2052) => 0. fresh VA (1028) => 0.
  - settle after map: 100/200/500 us (nmap 8), 200/1000 us (nmap 32), busy 200-2000 us => all 0 bad.
  => Condition: new handle on recycled physical VRAM mapped at a previously-used VA; first writes lost (zeros) until a short wall-clock delay after map. Serve hits it because forward starts right after map (map->use gap ~0); unmap->map gap is irrelevant.
  Likely async clear/PTE update ordering in amdgpu/KFD on the PT fast path. Needs driver review.
- FIX DESIGN (fix1): sync() keeps free granules mapped (cache), unmaps only when budget would refuse; trim() on idle returns VRAM; any map at a previously-mapped VA sleeps REMAP_SETTLE=5ms before return.
- Diag patch saved to ~/work/diag_vmm_trace.diff (reverted from tree).

## Update (UTC ~15:10)
- COMMITTED fdcc7d58c "kernels: census the vmm_paged twins and count multi-slot modules" (priority 3). `cargo test -p rdna-compute --lib kernel_registry`: 63 passed (logs/regtest.log).
- fix1 (keep free granules mapped as cache; trim at idle; 5 ms REMAP_SETTLE) is NOT sufficient. queue16:
  - X1 seq:30 vmm+pm4: long correct 10/30 (wrong: d9c6x9 48c4x5 9109x4 43e5x2), short 29/30. kfaults=0.
  - X1H vmm+hip: long 10/30 (48c4x14 d9c6x5). Not PM4/tape specific.
  - X5 127x4 x20: rep 3 all four lanes wrong/different, other 19 reps identical. Short single-page requests also hit.
  - XR reclaimv:3: outputs stable (ce70f0f9edf0), VRAM 19317 -> 19291 MB.
- H6: long lane sometimes emits exactly the short lane answer (9109c8c47831) => cross-request KV visibility. NO_UNMAP (G7/G7H) 60/60 correct.
  Pool state logic reviewed (vmm_page_pool.rs used_granules/sync/unmap_free; CacheOnly/ReclaimPending count as used): no double-allocation found by inspection.
  Leading hypothesis: hipMemUnmap+Release then hipMemCreate+Map at a previously used VA leaves stale GPU translations / VA-tagged cache lines, so the VA can alias recycled physical memory owned by another granule. Needs driver/human review.
- Priority 4 (attempt=): `attempt=` is printed by hipfire-client TypedDaemonError Display from a process-global correlation id (serve/complete.rs:1190 next_attempt_id). Upstream code, not changed by this branch; the id must stay globally unique (commit/cancel/abort key). Making it per-request is a wire/display contract change => deferred to human decision.
- Next: queue17 Y1 NO_TRIM / Y2 SETTLE 300 ms / Y3 KICK discriminates time vs flush vs design fix.

## 2026-10-28 ~15:20 UTC (19:20 Dubai) — queue17/18 results (recovered from cfg_Y*.log) and decision
Diag tree (fix1 + DIAG hooks), expected long e5381e96ed4d / short 9109c8c47831, kfaults 0 everywhere:
| run | setting | reqs | long ok | short ok |
|---|---|---|---|---|
| Y0 | control (5 ms settle) | seq:20 | 7/20 | 17/20 |
| Y1 | NO_TRIM | seq:10 | 10/10 | 10/10 |
| Y2 | SETTLE 300 ms | seq:10 | 10/10 | 10/10 |
| Y3 | KICK | seq:10 | 10/10 | 10/10 |
| Y4 | KICK, settle 0 | seq:20 | 20/20 | 20/20 |
| Y5 | SETTLE 50 ms | seq:20 | 11/20 | 11/20 (periodic) |
Conclusion: any remap of a used VA onto a new handle is unsafe; time-based settles are not a fix (50 ms fails although the probe healed in ~200 us).
DECISION: stable binding. VmmPageMapper::sync never unmaps; a VA committed by a successful sync keeps its allocation until teardown
(bound_granules guard poisons on any attempted rebind); budget overflow = capacity refusal; idle trim removed.
Added check_table_ownership (Private page in at most one live slot table, no Free/ReclaimPending/out-of-range entries) per sync -> poisons on violation.
fix1 discarded from worktree (saved ~/work/fix1.diff). Build OK 15:26 UTC, vmm_page_pool unit tests 6/6.

### 2026-10-28 15:40 UTC — stable binding committed + first queue19 results
- Commit a75283756 "vmm: never remap a used VA; bind each KV granule for the pool lifetime" (on fdcc7d58c).
- Smokes (logs/smoke_sb.log): vmm_page_mapper_smoke, vmm_paged_pool_smoke, vmm_paged_reclaim_smoke, kernel_parity (Q8_WRITE/LDS/TILE/WMMA) all PASS.
- Full rdna-compute suite (logs/test_rc.log, GPU hidden): lib 1181 pass / 0 fail / 30 ignored; bins ok;
  tests/all_launcher_abi 64 pass / 2 FAIL (inventory_matches_the_changed_kernel_catalogue, changed_kernel_coverage_is_closed).
  cargo stopped there, so later integration binaries did not run. Investigating (logs/test_abi.log); rerun with --no-fail-fast after.
- queue19 (stable binding, invariants on where noted):
  | run | load | result | kfaults |
  |---|---|---|---|
  | Z1H | seq x30, hip backend | long 30/30 e5381e96ed4d, short 30/30 9109c8c47831 | 0 |
  | Z1 | seq x30, pm4 | long 30/30, short 30/30 (same shas) | 0 |
  | Z5 | 127x4 lanes x30, invariants | every lane 30/30 79079223b841 | 0 |
  (cfg_X1 at fix1 had many wrong seq iterations; cfg_D5 6/20 wrong per lane.)
- Uncommitted, not yet built (waiting for queue19 to stop using the binary):
  forward_slots.rs paged_replay_input_error() pre-replay validation + 4 unit tests;
  serve_engine.rs MoE-prefill/decode slot-disjointness check + explicit device_synchronize between the
  HIP-eager isolated MoE prefill and a possibly PM4-replayed decode remainder.

### 2026-10-28 16:05 UTC (20:05 Dubai) — queue19 complete, EQ attribution (queue20/21), ABI fix
- queue19 rest (stable binding a75283756), kfaults 0 on every run:
  | run | load | result |
  |---|---|---|
  | Z6 / Z6H | 127+7681 x20, pm4 / hip | 127 20/20 79079223b841, 7681 20/20 a35841f454d6 (both) |
  | Z2 / Z3 / Z4 | 127,1025,3841,257 x30/x20/x20 (LP / allowlist / no-LP) | 127 and 257 all correct; 1025 and 3841 vary (see EQ attribution) |
  | ZR | reclaimv:3, 4x12000 | all ce70f0f9edf0, VRAM flat 19811 MB (no idle trim by design) |
  | ZMX | maxctx 16300 / 2x8000 | ok; 16300+200 cleanly refused (validation 400) |
  | ZEX | exhaust, 40 pages | clean 503 "capacity exhausted while reserving KV pages", post requests correct |
  | ZSIX | 6 slots | all lanes stable across reps |
  | ZP | perf | 1025 3915/192.1, 3841 2872/184.0, 7681 2009/171.1, conc4 247.3 tok/s (= 01ecc10f D within noise) |
- EQ (eq.py, solo vs 4-way concurrent, 9 lengths, 36 rounds=144 lane results):
  | run | backend / setting | mismatches | where |
  |---|---|---|---|
  | ZEQ | vmm pm4 | 6 | only 1151-1153 |
  | ZEQA | arena pm4 | 7 | only 1151-1153 |
  | ZEQH | vmm hip replay | 4 | only 1153 |
  | ZEQS / ZEQF | crossover forced scalar / flash | 13 / 10 | only 1151-1153 |
  | ZEQB | BASE 54b635cb multi-slot | 144 | every length (base never matches solo when concurrent) |
  | ZEQ4K | vmm pm4, PREFILL_CHUNK=4096 | **0** | - |
  Cause: the shared per-step row budget (multi_slot_prefill_rows = max(prefill_chunk, verify rows), decode rows first, then
  prefill by rotating cursor; hipfire-runtime/src/scheduler.rs allocate_rows ~L290) splits a >1024-token prompt at a timing-dependent
  boundary when other lanes are present. The alternate outputs ARE the outputs of other chunkings: unchunked solo 1152 = 10f68f4f7896
  (the ZEQ/ZEQA "mismatch" hash), unchunked solo 1153 = ecf139ac0ce5 (the ZEQH "mismatch" hash, also base's solo hash).
  With boundaries pinned (chunk 4096) concurrent == solo 144/144. So: no memory corruption left; residual is chunk-boundary numerics,
  pre-existing in base's scheduler design. Not changed (scheduler policy; see report).
- all_launcher_abi: commit 24ec96695 lists the two gfx1100 MoE _slots kernels in hip_sources; inventory test passes and
  live_launchers_match_declared_kernel_abi now checks their launchers (65 pass). Remaining failure changed_kernel_coverage_is_closed
  (qwen4_moe_sym_{nt1_,}gemm "nk is not an if/else block") reproduces on BASE 54b635cb (logs/test_abi_base.log) => pre-existing.
Watchdog 2026-10-28T16:08:04Z: started fresh worker from main chat (see the next entries)

### 2026-10-28 16:09 UTC — state on fresh-worker start
- Commits since last entry: 2df51d447 (tape: pre-replay validation of rows/ctx bucket/block tables, 4 unit tests), ed2727abf (serve: MoE prefill isolation disjointness + device sync before decode remainder). test_tape.log / test_perf_full.log: hipfire-arch-qwen35 lib 488 pass / 0 fail / 25 ignored.
- Uncommitted: forward_slots.rs n_tiles gate active_slots > 1 -> >= 1 (paged prefill WMMA for single prefilling slot; stretch goal 6). Built in ~/work/target-perf (build_perf.sh) but NOT GPU-tested.
- **queue22 Y6 (127+7681 x20, vmm+pm4, invariants, binary = ed2727abf tree) GPU WEDGE at 16:07:17 UTC on iteration 9** (iters 0-8 correct).
  Daemon: HIPFIRE_REDLINE_QUEUE_TIMEOUT queue_id=32 read_index=9 write_index=10 (PM4 replay of live_slots=2 tape f5637977974b439b), then HIP 700.
  Kernel: GCVM_L2_PROTECTION_FAULT, UTCL2 client SQC (data), PERMISSION_FAULTS 0x3, page at address 0x0, pid 104183; MES failed REMOVE_QUEUE/SUSPEND; queue reset; "device wedged".
  Same QUEUE_TIMEOUT signature as the earlier arena-only fault (logs/C_arena.log:93) => PM4 replay path, not VMM remap.
  After the wedge amd-smi shows GFX_ACTIVITY 100% with VRAM 25 MB and no hipfire process: device needs a reboot. Y5 (127x4 x30) before it was 120/120 correct.
- Per hard rules: GPU work STOPPED. Only CPU-only work (GPU-hidden unit tests, code review) until Ghazni reboots.

### Paused by Ghazni 2026-10-28T16:11:07Z: GPU work on hold until he resumes
- Ghazni is debugging the stuck GPU from the Proxmox host. No GPU jobs (no amd-smi resets, daemons, stress or GPU tests) until he resumes.
- Running (CPU-only, GPU hidden): ~/work/test_full.sh -> logs/full_rc.log (rdna-compute all targets, --no-fail-fast), then logs/full_ws.log (workspace, --no-fail-fast); summary in logs/test_full.out. Tree = ed2727abf clean.
- Prefill WMMA gate patch saved as ~/work/prefill_wmma.diff and reverted from the tree (NOT committed, untested on GPU).
- Lead for the Y6 / C_arena PM4 fault (unproven): SQC(data) fault at VA 0 right after a fresh prepare of the live_slots=2 tape, same QUEUE_TIMEOUT signature on arena => PM4 replay/prepare path (retained kernargs pool=vram, host-patched, published by one BAR readback in replay.rs:5278 / runtime.rs:2710), not the VMM remap. Tapes are re-captured every request (3 prepares/iter) — check the prepare path publishes kernargs before the first replay, and test HIPFIRE_PM4_KERNARG_POOL=host.
- Next GPU steps once resumed (one at a time, invariants on, stop at first fault):
  1. Health check: amd-smi GFX_ACTIVITY idle, journalctl -k clean, smoke_sb.sh.
  2. Y6 repeat (127+7681 x20 pm4) at ed2727abf; then the same with HIPFIRE_PM4_KERNARG_POOL=host; then at a75283756 (Z6 tree) to see whether ed2727abf/2df51d447 changed anything.
  3. 4x12000 (D_edge reclaim) pm4 and hip; 127+7681 hip.
  4. Rest of queue22 (YEQ, YEQ4K, Y2, YR, YEX, YMX, YP), then prefill_wmma.diff: EQ + perf vs 54b635cb/01ecc10f before any commit.
  5. Regression tests (concurrent==solo, 127/128/129 and page multiples +-1, pool saturation, teardown after fault).

### 2026-10-28 16:43 UTC — CPU-only test results at ed2727abf (GPU hidden), GPU still paused
- rdna-compute all targets (logs/full_rc.log): lib 1181 pass/0 fail/30 ign; all_launcher_abi 65 pass / 1 FAIL (changed_kernel_coverage_is_closed, qwen4_moe_sym gemm "nk" sites — reproduces on base 54b635cb per test_abi_base.log => pre-existing). Goal-1 inventory failure is FIXED by 24ec96695.
  NEW to this log (not seen before because cargo stopped at all_launcher_abi earlier):
  - flagship_pack_coverage 3 FAIL (gfx1100/1151/1201): measured pack sha256 != source for gated_delta_net_q8_fast (+ compact3_b2/_rows/_ml/_ml_from). The branch modifies kernels/src/gated_delta_net_q8_fast.hip (+79 lines vs 54b635cb) => likely branch-introduced stale pack receipts; check on base/01ecc10f.
  - kernel_registry_coverage every_runtime_hip_module_is_registered_or_allowlisted: 34 modules (railgun_window *_pm_gfx1201/gfx1151, dflash_selector_greedy_exact, ...) not registered/allowlisted. Branch edits lazy-kernel-allowlist.tsv (4 +-1) => check on base/01ecc10f.
- workspace --no-fail-fast (logs/full_ws.log, still running under nohup at write time, 3600 s cap): so far 11048 pass / 3 fail / 177 ign: hipfire-registry receipts::every_quote_is_verbatim_in_its_source_and_prints_the_number; hipfire-tui ui_settings golden_l1_quality_160x48, golden_s4_search_kv_80x24. Attribution vs base pending.
- Next CPU step (when allowed): run the same failing test binaries on ~/work/hipfire-base and 01ecc10f to split pre-existing vs branch-introduced.
Resumed by Ghazni 2026-10-28T18:20:52Z: GPU recovered, worker started from main chat

## 2026-10-28 resume (Dubai 22:19; server UTC below)
- 18:20-18:27 UTC health: GPU idle, no amdgpu faults in boot 0 (started 18:18:52). Smoke (logs/smoke_resume.log): ALL PASS incl. vmm_paged_kernel_parity Q8_WRITE/LDS/TILE/WMMA.
- Fault analysis (boot -1): 16:07:17 UTC (pid 104183, queue 32, read_index=9) and 17:30:03 UTC (pid 285461, queue 89, read_index=29, step m=[0,1,1,0] pos [166,7711]).
  BOTH on the same 2-slot decode tape tape_hash f5637977974b439b (1037 launches, resource_independent=281) in steady-state decode.
  Kernel: sq_intr then gfxhub page fault addr 0x0, client SQC (data) = scalar load through null, STATUS 0x00801431 (read, permission), vmid 8; MES REMOVE_QUEUE fails -> wedge.
  Ghazni's 16:50-18:07 UTC repros on the same binary: ~800+ clean Y6 iterations -> rate ~1 per several hundred reps; x20 A/B cannot discriminate configs.
  Hypotheses: (a) intra-tape race missed by Resource wait policy (scalar-loaded pointer/index read before written), (b) kernarg segment read as zero, (c) HIP-stream work overlapping the PM4 replay.
- Diagnostic build (uncommitted, worktree, logs/build_diag.log): HIPFIRE_REDLINE_TAPE_DUMP_DIR dumps each prepared launch list + kernarg hex;
  HIPFIRE_REDLINE_BOUNDARY_STAMPS=1 builds the gfx11 boundary-stamped tape (stamp after each DISPATCH and CS_PARTIAL_FLUSH); on timeout prints
  HIPFIRE_REDLINE_PM4_STAMP_PROGRESS leading_nonzero=N -> names the stalled window. Z1 (3 reps) checks the dump; then long soak.
Watchdog 2026-10-28T19:10Z: started fresh worker from main chat after interruption
- 19:08 UTC Z2 (127+7681 x400, stamped tape, invariants): 400/400 correct, kfaults=0 (cfg_Z2.log). NOT significant on its own: baseline unstamped rate is ~2 faults / ~850 Y6 iters (~1 per 17-19k 2-slot replays), so expected faults in 400 iters ~1.
- Analysis (no GPU): 2-slot tape (1037 dispatches) vs 1-slot (834) have the same no-wait pair structure (xbatch||xbatch 110, router||silu 80, residual_sigmoid||gate_up 40, alpha_gate||conv1d 30, rmsnorm 10, kv K||V 10); none of the elided pairs can produce a NULL pointer (MoE expert ids guarded, expert table static, descs null-checked). GDN slots kernarg has slot2/slot3 = NULL and conv1d lanes has NULL for unused lanes (valid only if blockIdx.y/z never exceeds lanes). No dynamic grids in the tape.
  Fault waves: wgp 0 on both SHs (several SEs), wave_ids 0/5/6 => one multi-WG dispatch with a uniform null scalar base. Leading candidates: SH state lost across MES preemption with stateful SH elision (e.g. RSRC2 user-SGPR count / USER_DATA), or a VRAM kernarg segment read as zero. No kernel-log events before either fault (no eviction trace), no gpupool jobs.
- 19:24 UTC queue24 A1 started: accelerated soak 127+7681 gen=1024 x140 (~1000 2-slot replays/iter, ~7 baseline-expected faults) with HIPFIRE_REPLAY_PM4_STATEFUL=legacy HIPFIRE_PM4_KERNARG_POOL=host. run_cfg.sh now passes GEN (backup run_cfg.sh.bak24).

### 2026-10-28 20:05 UTC (00:05 Dubai) — A1 result, step-5 commit, A2 started
- **A1 (legacy SH emission + host kernarg pool, 127+7681 gen=1024 x140): CLEAN.** 140/140 iterations, 280 requests, ~137,620 two-slot decode replays, kfaults=0, VRAM back to 25 MB (queue24.out 19:57:38, logs/A1_legacy_hostka.log, repro/cfg_A1.log).
  Baseline: 2 faults in ~34k two-slot replays (~850 Y6 iters); both faults were in steady-state decode (pos [166,7711], read_index 9/29), so the per-replay rate applies: expected ~7-8, P(0|baseline) ~0.05-2% (Poisson / Jeffreys). Caveat: if the trigger were per-request, expected only ~0.3 (inconclusive) — but both faults were mid-decode, not on a prepare.
- **A2 started 19:58:05 UTC**: legacy SH emission + VRAM kernarg pool (default), same workload x140 (repro/queue25.sh, cfg_A2.log, logs/A2_legacy_vramka.log). Discriminates register elision vs kernarg pool. ETA ~20:31 UTC.
- Code lead: on gfx10/11 PM4 "static" (default) maps to Gfx10Pm4CommandBuffer::new_stateful() = elide EVERY repeated SH register (PGM, RSRC1/2, USER_DATA kernarg ptr, NUM_THREAD) within the IB; on gfx12 "static" = new_static_stateful (only COMPUTE_RESOURCE_LIMITS cached). If A2 is clean the fix is to give gfx10/11 the gfx12 static-only semantics.
- **Step 5 committed cd8c3382c** on review/vmm-fixes (CPU-only, GPU hidden, --release --locked):
  | test | verdict | action |
  |---|---|---|
  | flagship_pack_coverage x3 | branch-introduced (GDN source sha drift) | fixed: q8_fast.hip byte-identical to base, slots entry in gated_delta_net_q8_fast_slots.hip; 9/9 pass (logs/t5b.log) |
  | railgun-cert recording inventory | branch-introduced | fixed; 12/12 |
  | hipfire-tui settings goldens x2 | branch-introduced (3 new config fields) | snapshots updated; 356 pass |
  | inventory_matches_the_changed_kernel_catalogue | branch-introduced | already fixed by 24ec96695 |
  | changed_kernel_coverage_is_closed | pre-existing (fails on base) | none |
  | kernel_registry_coverage (34 rows) | pre-existing: identical 34-row set on base and 01ecc10f | none |
  | hipfire-registry receipts quote | pre-existing (base) | none |
  | qsa_bf16_scores | needs GPU (env) | none |
- Diagnostic files (replay.rs, aql/replay.rs, pm4_gfx10.rs: tape dump + boundary stamps) remain uncommitted.

### 2026-10-28 20:20 UTC (00:20 Dubai) — A2 FAULTED: register elision is NOT the cause
- A2 = old binary (bin_hipfire_pre_fix_1832 tree build), HIPFIRE_REPLAY_PM4_STATEFUL=legacy (full SH emission), retained kernarg pool = VRAM,
  127+7681 gen=1024, invariants on. Iterations 0-~92 correct (cfef9419a1e0 / fbecc76894dc), then fault at 20:17:42 UTC:
  GCVM_L2_PROTECTION_FAULT page 0x0, UTCL2 client SQC (data), PERMISSION_FAULTS 0x3, vmid 8, sq_intr wgp0 sh0+sh1 (same signature as Y6/C_arena),
  MES REMOVE_QUEUE/SUSPEND fail, queue reset, "device wedged"; kernel MODE1 reset 20:18:05, "GPU reset(3) succeeded", VRAM lost.
  Iterations 93-98 HTTP 500. queue25 run() status=GPU_FAULT kfaults=19, ABORT set. Log: repro/cfg_A2.log, logs/A2_legacy_vramka.log, journalctl -k 20:17-20:18 UTC.
- queue26 (fixed-binary soaks/queue22) was gated on a clean A2 and did NOT start. All my GPU processes stopped (none left). ABORT file set.
- Post-reset state 20:18:45 UTC: amd-smi GFX_ACTIVITY 1 %, USED_VRAM 25 MB (unlike the 16:07 wedge, which stuck at 100 %).
- Conclusion: legacy SH (A2) faults, host kernarg pool (A1, 137,620 two-slot replays) clean => the discriminator is the VRAM retained kernarg pool,
  not SH-register elision. The uncommitted static-register-policy change (pm4_gfx10.rs new_static_stateful + replay.rs mapping) is NOT a fix; kept
  uncommitted for now. Fault reproduction budget is now used up: no more fault-risky GPU runs without Ghazni.
- Mechanism lead: a null SQC (scalar) data read = a kernel's s_load from a pointer read out of its kernarg segment that is 0 => kernarg bytes
  stale/zero when the dispatch reads them (host BAR write to VRAM not yet visible through HDP / stale K$ line), not a bad kernarg base register.
Watchdog 2026-10-28T20:22Z: A2 GPU fault + MODE1 reset seen; no worker restarted; holding for Ghazni decision on further fault-risky runs

## 2026-10-28 20:23 UTC (00:23 Dubai)
Ghazni OK 00:22 Dubai: fix + soak, resets acceptable

## 2026-10-28 20:35 UTC (00:35 Dubai) — kernarg visibility fix implemented (HDP flush), A3 soak running
- Root cause (vendor-confirmed policy): retained PM4 kernargs live in a VRAM pool patched by host stores through the
  write-combined BAR (since 3bf6a0e96, 2026-09-30, present in base). Publish was fence + 1-byte BAR readback only.
  AMD ROCclr (rocsettings.cpp setKernelArgImpl, SWDEV-451594) uses readback-only ONLY on MI200/MI300 ("works only ...
  because of the strict guarantee on ordering of stores in those ASICs"); every other PCIe agent with a valid HDP
  flush register uses DeviceKernelArgsHDP: `*hdpMemFlushCntl = 1; read back`. Without the HDP flush, BAR stores can
  sit in the Host Data Path and a dispatch reads stale/zero kernarg pointers -> SQC scalar load at VA 0x0 -> fault ->
  MES REMOVE_QUEUE timeout. Matches A1 (host pool clean, 137,620 replays) vs A2 (VRAM pool faulted, ~iter 92).
- Fix (redline-rocr): query HSA_AMD_AGENT_INFO_HDP_FLUSH (0xA00E) in discover_host_writable_device_local; fail closed
  to the host pool if unavailable/null. publish_host_writes = SeqCst fence; BAR readback of last byte (all earlier
  posted stores reached HDP); write 1 to HDP_MEM_FLUSH_CNTL + read back; SeqCst fence. Global flush covers all
  segments + IB patches written before it. No timing/sleeps. HIPFIRE_PM4_KERNARG_POOL=host remains the fallback.
- Dropped the uncommitted PM4 static-register-policy change (falsified as fault cause by A2); saved to
  ~/work/static_policy.diff as a separate semantic note (40c2360ea folds Static into full elision).
- CPU: redline-rocr lib 77 passed; vmm_page_pool 9 passed. Committed tests separately.
- queue27 (repro/queue27.sh/out): A3 = HEAD+HDP, default PM4 policy, VRAM kernargs, 127+7681 GEN=1024 x280 (~2x A1)
  started 20:25:04 UTC; log line confirms "pool=vram (publish: BAR readback + HDP flush)"; outputs identical to A1 hashes.
Watchdog 2026-10-28T20:41:18Z: started fresh worker from main chat after Tailscale re-auth

### 20:43 UTC — fresh worker state
- HDP kernarg publish fix is NOT yet committed (worktree: redline-rocr abi.rs/runtime.rs, rdna-compute replay.rs). scripts/check_vmm_paged_regressions.py untracked (REG run in queue27).
- A3 at iter 78/280 clean, ~12.3 s/iter -> ETA ~21:23 UTC. queue27 then runs F2, YR, S2H, S2M, YEQ, YEQ4K, Y2, YEX, YMX, REG, YP, YPH, YP01 automatically.
- Built ~/work/probe/kpub (publish cost microbench: readback-only vs readback+HDP, 1037 VRAM segments); run in a GPU gap.

## 2026-10-28 22:51 UTC (02:51 Dubai)
SSH restored 02:50 Dubai after Tailscale node re-auth; A3 found GPU_FAULT

### 2026-10-28 22:56 UTC — GPU health, A3 analysis, fix plan
- GPU recovered: MODE1 reset at 20:46:02 UTC succeeded (VRAM lost). 22:50 UTC amd-smi GFX 1%, VRAM 25 MB; smoke (logs/smoke_postA3.log, 22:51-22:52 UTC): page_mapper / paged_pool / paged_reclaim / paged_kernel_parity Q8_WRITE/LDS/TILE/WMMA all PASS. Watchdog untouched.
- A3 (HEAD+HDP flush, default static SH policy, VRAM kernargs, 127+7681 gen1024 x280): iters 0-98 correct (hashes = A1), fault at iter ~99 (~97k replays), 20:45:48 UTC. Signature identical to A2/Y6: sq_intr wgp0 waves 0/4/5/6, gfxhub page fault VA 0x0, SQC(data), STATUS 0x00801431 PERMISSION_FAULTS 0x3, vmid 8; MES REMOVE_QUEUE fail -> wedged -> MODE1. Daemon QUEUE_TIMEOUT queue 401 read_index 272, PM4_PROGRESS end_tick=0, steady-state 2-slot decode m=[1,1,0,0] pos [409,7954]. => HDP flush does NOT fix it.
- Stats caveat: on THIS workload VRAM faults ~1 per ~93k replays (A2 ~90k, A3 ~97k). A1 host clean 137,620 => P(0 | same rate) ~23%, not conclusive alone. Host x280 (~275k) => P~5%; combined with A1 ~1.2%.
- Analysis: kernarg pointers are written once at prepare; per-replay patches are only u32 scalars, written while the queue is quiescent (replay_and_wait). A torn/late scalar patch cannot yield a null pointer, so overwrite race / pure HDP visibility is unlikely.
- Hypothesis H-K (unproven): every prepare allocates a fresh KernargPool and frees each VRAM segment in KernargBuffer::drop; tapes are re-prepared ~3x/iter and the tape hash recurs with period ~5 iters, so VRAM kernarg VAs are freed and re-bound continually. Same gfx1100 stale-translation class as root cause 1 (VA re-bound to new physical page, GPU reads old/cleared page -> zero pointer). Host pool is system memory -> not affected.
- Fix (uncommitted in worktree): default HIPFIRE_PM4_KERNARG_POOL=host on all targets (no target has a long-soak-proven VRAM path); vram opt-in keeps the HDP flush; new experimental vram-recycle = VRAM segments retired to a runtime-wide recycle list (exact size) and freed only at runtime shutdown, to test H-K. Schema/CONFIG/REDLINE/R3/CHANGELOG updated.
- Plan: queue28 S1 = host default x280 (gate commit), S2 = vram-recycle x280 (H-K test), then perf YP host / YPV vram / YP01, then rest of queue27 with stop-on-first-fault.

### 2026-10-28 23:42 UTC — worker resumed (box SSH restored)
- S1 running (queue28 pid 114866 -> run_cfg S1), 187/280 iters clean at 23:42 UTC, ~13.9 s/iter, ETA ~00:05 UTC. queue28b (pid 119037) waits on queue28, gates on "done S1 status=ok .*kfaults=0" in cfg_S1.log, then YP/YPV/YPR/YP01/F2/YR/S2H/S2M/YEQ/YEQ4K/Y2/YEX/YMX/REG, then S2 (vram-recycle) last, stop on first fault. Not relaunching anything.
- Worktree diff reviewed (host default, HDP flush on vram, vram-recycle); build + config/redline-rocr tests passed (logs/build_hostka.log). Commit after S1 clean.

### 2026-10-29 00:06 UTC — S1 CLEAN, host-default committed 129ab13ee
- S1 (host kernargs default, 127+7681 gen1024 x280, invariants): 280/280 iters, every request 200 cfef9419a1e0 / fbecc76894dc, kfaults=0, 22:58:27-00:03:23 UTC (3890 s), VRAM back to 25 MB. ~275k two-slot replays (2x A1). Log logs/S1_hostka_default.log, repro/cfg_S1.log.
- Commit 129ab13ee "redline: retained PM4 kernargs default to the host pool; HDP-flush the VRAM publish" (host default + HDP flush on vram + vram-recycle). Not pushed.
- Perf (queue28b, same binary, decode tok/s run1): host 1025 164.7 / 3841 160.0 / 7681 150.2 / conc4 206.8; vram 192.8 / 183.7 / 170.6 / 244.9; vram-recycle 192.2 / 184.0 / 170.4 / 244.1. Prefill equal (~3970/2867/2007). Host pool costs ~13-15% decode. => S2 (vram-recycle soak) matters: if clean it recovers the speed.

### 2026-10-29 00:55 UTC (04:55 Dubai) — queue28b complete; S2 (vram-recycle) FAULTED -> keep host default
- Core runs (HEAD 129ab13ee, host kernargs default, invariants on), all status=ok kfaults=0:
  | run | load | result |
  |---|---|---|
  | F2 | 127+7681 x300 gen48 | 127 300/300 79079223b841, 7681 300/300 a35841f454d6 |
  | YR | reclaimv:3 4x12000 pm4 | 12/12 ce70f0f9edf0, post 127 9109c8c47831; VRAM 19313 -> 19807 MB (+494 MB stays bound) |
  | S2H | reclaimv:3 4x12000 hip replay | 12/12 ce70f0f9edf0; VRAM 19268 -> 19763 MB (+495 MB) |
  | S2M | 127+7681 x20 hip replay | 20/20 both lanes |
  | YEQ | EQ 36 rounds default chunk | 8/144 mismatches, only len 1151/1153 (chunk-boundary class) |
  | YEQ4K | EQ 36 rounds PREFILL_CHUNK=4096 | 0/144 |
  | Y2 | 127,1025,3841,257 x30 LP | 127 and 257 30/30; 1025/3841 vary (chunk-boundary, as Z2) |
  | YEX | exhaust 40 pages | clean 503 capacity exhausted, post requests correct |
  | YMX | maxctx | 16300+84 ok, 16300+200 400 validation, 2x8000 ok |
  | REG | scripts/check_vmm_paged_regressions.py | REGCHK exit 1: 6 concurrent!=solo, all at len 1151/1152 (default chunk -> chunk-boundary class); saturate/abort rechecks ok |
- **S2 = vram-recycle kernargs, 127+7681 gen1024 x280: GPU FAULT at iter 19 (~19k two-slot replays), 00:47:54 UTC.**
  Daemon log shows "pool=vram (publish: BAR readback + HDP flush; recycle=true)". HIPFIRE_REDLINE_QUEUE_TIMEOUT queue_id=82 read_index=178,
  PM4_PROGRESS end_tick=0, HIP 700. Kernel: sq_intr wgp0 sh0/sh1, gfxhub page fault VA 0x0, SQC(data), STATUS 0x00801431, PERMISSION_FAULTS 0x3, vmid 8;
  MES REMOVE_QUEUE/SUSPEND fail -> wedged -> MODE1 reset 00:48:08 succeeded. Iters 19-24 HTTP 500, ABORT set. GPU idle 25 MB at 00:49.
  => H-K (VRAM kernarg VA rebinding) FALSIFIED: never-freed VRAM segments fault like the plain VRAM pool (A2 ~92, A3 ~99, S2 19 iters).
  Decision: keep host default (129ab13ee). vram and vram-recycle stay opt-in, documented as fault-prone on gfx1100. No VRAM default commit.
  Logs: repro/cfg_S2.log, logs/S2_vramrecycle.log, journalctl -k 00:47-00:48 UTC.

### 2026-10-29 01:15 UTC (05:15 Dubai) — queue29/30 (prefill WMMA) done; committed 5ba922332 + 7b1be5929
- Prefill patch (forward_slots.rs n_tiles gate `active_slots > 1` -> `>= 1`), binary ~/work/hipfire-prefill (target-perf, built 00:54 UTC, ~/work/build_perf.sh, logs/build_perf28.out; qwen35 lib 488 pass). Invariants on, all status=ok kfaults=0:
  | run | load | result |
  |---|---|---|
  | PP | perf | prefill 1025/3841/7681 = 4999/4443/3760 tok/s (HEAD 3970/2869/2007, +26/+55/+87%); decode 164/158/150.5; conc4 215.1 |
  | PEQ | EQ default chunk | 13/144 mismatches, only len 1151-1153 (chunk-boundary class) |
  | PEQ4K | EQ PREFILL_CHUNK=4096 | 0/144 |
  | PF | 127+7681 x40 | 40/40 stable; 127 -> 36eee1858597 (was 79079223b841, WMMA numerics), 7681 a35841f454d6 unchanged |
  | PY2 | 127/1025/257/3841 x20 | 127, 1025, 257 20/20 (1025 was unstable on HEAD); 3841 varies |
  | PR | reclaimv:3 4x12000 | split 8f60863d5e4b / 985fa056f18e; maxctx ok; post-127 ok |
  | REG4K | regression script, HEAD binary, chunk 4096 | PASS, REGCHK exit 0 |
  | PS12 (q30) | solo 12000 x3 + reclaimv:2, default chunk | solo 3/3 985fa056f18e; concurrent mostly 985f, one 8f60 |
  | PS12K (q30) | same at chunk 4096 | solo 2/2 ce70f0f9edf0 (= HEAD); concurrent mix of ce70/985f/8f60; VRAM 21491 MB (bigger chunk scratch) |
  | HS12 (q30) | HEAD solo 12000 x2 | 2/2 ce70f0f9edf0 |
  => The PR split is chunk-schedule numerics, not corruption. Solo output is deterministic for a fixed chunk size. The output changes with the chunk size (985f at 1024, ce70 at 4096). All variants are coherent with the same prefix. Concurrent lanes get different chunk splits and so different hashes. HEAD's per-row path hid this at 12000, but HEAD shows the same class at 1151-1153, 1025 and 3841.
- Committed (local only, not pushed): 5ba922332 (prefill gate, numerics caveat in the message) and 7b1be5929 (scripts/check_vmm_paged_regressions.py).
- Logs: repro/cfg_{PP,PEQ,PEQ4K,PF,PY2,PR,REG4K}29.log, repro/cfg_{PS12,PS12K,HS12}30.log, server logs logs/<label>.log. Queues repro/queue29.sh, queue30.sh. edge.py gained a "solo:<len>:<n>" mode (backup edge.py.bak29).

## FINAL REPORT — review/vmm-fixes (2026-10-29 01:15 UTC / 05:15 Dubai)

### Commits (branch review/vmm-fixes on 01ecc10fd; local, never pushed)
| hash | fixes |
|---|---|
| fdcc7d58c | kernel census covers the vmm_paged twins and counts multi-slot modules |
| a75283756 | **RC1 fix**: never remap a used VA; each KV granule is bound for the pool lifetime (stale-memory corruption) |
| 24ec96695 | ABI inventory covers the two gfx1100 MoE _slots kernels |
| 2df51d447 | tape validates rows, ctx bucket and block tables before a paged PM4 replay |
| ed2727abf | MoE prefill isolation is asserted disjoint, with a sync before the decode remainder |
| cd8c3382c | fixes the test failures this branch introduced |
| cccc2b31a | vmm_page_pool regression tests: page edge, saturation, teardown after fault |
| 129ab13ee | **RC2 workaround**: retained PM4 kernargs default to the host pool; HDP flush on the VRAM publish; vram-recycle is opt-in |
| 5ba922332 | paged single-slot prefill uses WMMA (fixes the 16-49% prefill regression) |
| 7b1be5929 | adds scripts/check_vmm_paged_regressions.py |

### Root causes
1. **VA remap reads stale memory (gfx1100, ROCm VMM).** After unmap/release and remap, a VA reads old physical contents, which corrupted KV. Fix: lifetime binding, so a used VA is never remapped. Repro: ~/work/probe/vmmclear.cpp (+ vmmclear_busy.cpp; control mallocclear.cpp).
2. **PM4 retained kernargs in a VRAM pool fault the GPU.** Signature: SQC(data) page fault at VA 0x0, STATUS 0x00801431, PERMISSION_FAULTS 0x3, vmid 8, sq_intr wgp0. Then MES REMOVE_QUEUE/SUSPEND fails, the device wedges, and a MODE1 reset follows. It faults with legacy SH (A2), with HDP flush (A3), and with never-freed recycled segments (S2, iter 19). So it is not register elision, HDP visibility, or VA rebinding (H-K falsified). The root cause is unresolved. Host kernargs are clean: S1 280/280 (~275k replays), A1 ~137k.
3. **Prefill regression (perf).** A paged pool never takes the single_slot reduction. The `active_slots > 1` gate therefore sent solo and isolated-MoE prefill to the per-row decode attend.

### Before/after
Correctness and faults (27B q8, 4 slots, ctx 16384, invariants on):
| | 01ecc10f-era vmm | HEAD (host) | HEAD vram / vram-recycle |
|---|---|---|---|
| stale-KV corruption | yes (RC1) | none observed | n/a |
| GPU faults in soak | A2/A3 fault at ~92-99 iters | 0 (S1 280/280, F2 300/300, all queues) | fault (A2/A3; S2 at 19 iters) |
| EQ at chunk 4096 | — | 0/144 (prefill patch 0/144) | — |
| EQ at default chunk | — | 8/144, only 1151-1153 (base the same) | — |

Perf (prefill tok/s at 1025/3841/7681 | decode tok/s at the same lengths | conc4 aggregate tok/s):
| config | prefill | decode | conc4 |
|---|---|---|---|
| base 54b635cb non-paged multi-slot | 4862/4480/3884 | 82/60/44 | 111 |
| base 54b635cb arena | 4956/4546/3905 | 82/60/44 | 190 |
| 01ecc10f (vmm) | 3980/2863/2006 | 191/183/170 | 245 |
| HEAD 129ab13ee host (default) | 3970/2869/2007 | 164.7/160.0/150.2 | 206.8 |
| HEAD vram (opt-in, faults) | same | 192.8/183.7/170.6 | 244.9 |
| HEAD vram-recycle (opt-in, faults) | same | 192.2/184.0/170.4 | 244.1 |
| 5ba922332 host (default) | 4999/4443/3760 | 164/158/150.5 | 215.1 |
- Versus 54b635cb: prefill is back to parity and decode is ~2-3.4x faster. Versus 01ecc10f: prefill is +26/+55/+87%, but decode is 13-15% slower because of host kernargs.

VRAM cost of lifetime binding:
| | loaded | after 4x12000 |
|---|---|---|
| HEAD pm4 | 19313 MB | 19807 MB (+494 MB, held until teardown) |
| HEAD hip replay | 19268 MB | 19763 MB |
| 01ecc10f-era (unmaps) | — | back to 19291 MB |
| base 54b635cb non-paged | 21569 MB | 21585 MB |
| base arena | 21540 MB | — |
- Worst case is ~675 MB for 4x16384 (~10 KB/token), still about 1.7 GB below base. PREFILL_CHUNK=4096 adds ~1.7 GB of scratch (21491 MB).

### Open risks
- RC2 is not understood. The host default costs 13-15% decode. vram and vram-recycle stay opt-in and fault-prone on gfx1100.
- Chunk-boundary numerics: concurrent and solo outputs differ when chunk splits differ (1151-1153, 1025/3841, 12000 under the prefill patch). Base has the same class. A deterministic check needs PREFILL_CHUNK=4096 (or a pinned schedule).
- Lifetime binding keeps the high-water KV bound until pool teardown, so there is no reclaim under memory pressure.
- The attempt= id work is deferred to Ghazni.
- Test failures that predate this branch: changed_kernel_coverage_is_closed, kernel_registry_coverage, and the hipfire-registry quote.
- After any fault, the MES wedge needs a MODE1 reset. All in-flight requests fail (HTTP 500) until the watchdog resets the GPU.

### Possible upstream ROCm/amdgpu bugs
1. VMM: a VA remapped after unmap/release returns stale physical memory (gfx1100). Repro: ~/work/probe/vmmclear.cpp.
2. MES: after an SQC data page fault, REMOVE_QUEUE/SUSPEND fails and the device wedges until a MODE1 reset (kernel log 00:47:54-00:48:08 UTC, 2026-10-29).
3. PM4 kernargs in device VRAM fault (SQC VA 0x0, PERMISSION_FAULTS 0x3) even with BAR readback + HDP flush and never-freed segments. Host-memory kernargs never fault.

### Commands
- Build: ~/work/build_perf.sh (-> ~/work/target-perf; tree ~/work/hipfire-prefill), ~/work/build_hostka.sh.
- Runs: cd ~/work/repro && ./run_cfg.sh <label> <tree> <port> <lp> $M $V [env...] -- (M/V from queue7.sh), driven by queue28b.sh / queue29.sh / queue30.sh. Edge modes: EDGE="solo:L:n reclaimv:n maxctx exhaust six seq:n".
- Regression: python3 scripts/check_vmm_paged_regressions.py with HIPFIRE_SERVE_MULTI_SLOT_PREFILL_CHUNK=4096 (REG4K PASS).
- Logs: ~/work/repro/cfg_*.log (client side), ~/work/logs/<label>.log (daemon), journalctl -k (faults).

## 2026-10-29 (Asia/Dubai, UTC+4): follow-up for Ghazni: remap question, decode cost

### Why the same VA is remapped (design, not an accident)
- Each full-attention layer reserves ONE VA arena for K and one for V, once, with hipMemAddressReserve
  (hip-bridge/src/vmm.rs:695, called from VmmPageMapper::new, vmm_page_pool.rs:240-300).
- A KV position's byte address is a pure function of its physical page index
  (kernels/src/kv_slot_desc_vmm_paged.h:60-81: granule*granule_bytes + within*page_tokens*per_pos + page_off*per_pos).
  Arena bases are baked into kernargs and PM4 tapes, so page index N always lives at arena+f(N).
- Backing on demand: map_at (vmm.rs:982) = hipMemCreate :1035, hipMemMap :1042, hipMemSetAccess :1071/:1082.
  Returning memory: unmap_at (vmm.rs:1312) = hipDeviceSynchronize :1343, hipMemUnmap :1356, hipMemRelease :1363.
  At base, the idle trim (serve_engine.rs:5003-5009, then sync at vmm_page_pool.rs:447-449) unmapped free granules. Later reuse
  of that page index meant Create+Map of a NEW handle at the SAME VA.
- This is the intended way to use the VMM API: reserve once, then change the physical backing behind the reserved range. CUDA VMM,
  vAttention and PyTorch expandable_segments all do it. Nothing else can get that VA while the reservation exists;
  the VA is ours. The API contract after Unmap+Map(new handle) is that accesses hit the NEW physical memory.

### Decisive probe: ~/work/probe/vmmstale2.cpp (hipcc -O2 --offload-arch=gfx1100)
Usage: ./vmmstale2 iters nmap release reuse sleep_us [thrash]. All VAs come from one big reservation that is never reused, so old VAs cannot alias
across iterations. Per granule (2 MiB): A mapped at R and at alias VA_A, filled PA via R; Unmap(R) (A kept or released);
B created, mapped at its own VA_B, filled PB, synced; B mapped at R (reuse=1) or a fresh R2 (reuse=0); a kernel reads R and writes PC
through R. Checks: read@R==PB, B@VA_B==PC, A@VA_A==PA. The GPU was idle and there were 0 kernel faults afterwards.

| args | meaning | bad granules | what was observed |
|---|---|---|---|
| 40 8 0 1 0 | reuse VA, A alive | 313/320 | read@R returned A's PA; write via R landed in A; B untouched |
| 100 1 0 1 0 | 1 granule/iter, A alive | 100/100 | same |
| 40 8 0 1 1000 / 50000 | +1 ms / 50 ms sleep, A alive | 313/320 | sleep does not heal |
| 40 8 0 1 0 1 | + 4 GiB TLB-thrash kernel before read | 313/320 | not a cached-TLB effect; page table itself stale |
| 40 8 1 1 0 | A released (freed page) | 68/320 | read/write a FREED physical page |
| 100 1 1 1 0 | 1 granule, A released | 99/100 | same |
| 40 8 1 1 200000 | A released, 200 ms sleep | 0/320 | heals eventually once A is freed |
| 40 8 0 0 0, 40 8 1 0 0, 100 1 0 0 0 | control: B at a fresh VA | 0 | correct |

Verdict: after hipMemUnmap(R) + hipMemMap(R, B) + SetAccess, GPU reads and writes through R still go to A's OLD physical page.
That page is either a different live allocation or a freed page. It is NOT "non-zeroed memory" (B was pre-filled with PB, and the read returned PA).
It is a stale GPU page-table translation in ROCm 7.x/amdgpu on this VM. That makes it a memory-isolation bug (use-after-free / cross-allocation write).
Correction to earlier wording: "re-mapped address reads stale memory" was vague. Also, "~200 us heals" is only true when the old handle is released;
with the old handle alive, 50 ms does not heal. I have not filed or reduced this to a minimal upstream report yet (vmmstale2 is ~150 lines and close).

### Decode cost (host vs VRAM retained kernargs)
- Tapes (logs/tapes/tape_9297_*): 1-slot decode = 834 dispatches/token, 50,064 B kernargs (avg 60 B; 30 x 304 B), 1.15M workgroups.
  2-slot = 1037 dispatches, 70,816 B.
- Measured (129ab13ee): 164.7 vs 192.8, 160.0 vs 183.7, 150.2 vs 170.6 tok/s. That is 0.885/0.806/0.796 ms per token, or ~0.95-1.06 us per dispatch.
  Bandwidth is trivial (~10 MB/s), so this is latency: the host pool is FINE_GRAINED|KERNARG_INIT system memory (runtime.rs:2055), which is uncached in GPU L2.
  So each dispatch's first scalar kernarg load is a PCIe round trip (~1 us via the VM's IOMMU), on the critical path behind CS_PARTIAL_FLUSH.
  The VRAM pool is coarse-grained device-local (runtime.rs:2146-2219), which is L2-resident.
- Kernarg preload (args in SGPRs): LLVM supports it on gfx90a/gfx940-942/gfx950/gfx1250 only, NOT gfx1100.
- HIP microbench (probe/kalat, kalatg; HIP_FORCE_DEV_KERNARG=0/1): no difference, because HIP's ~3 us/dispatch AQL floor hides a 1 us effect. Inconclusive by design.

## 2026-10-29 (Dubai 12:15-14:10, UTC+4; server UTC 08:15-10:10): kernarg decode-cost recovery prototypes
Branch review/kernarg-recovery (worktree ~/work/hipfire-kernarg, off review/vmm-fixes 7b1be5929), commit c3712f397 (local, not pushed).
Opt-in HIPFIRE_PM4_KERNARG_STAGE=off|pack|gpu-copy|l2-prefetch (default off = unchanged host pool). Applies only to single-queue Gfx11
PM4 tapes with host kernargs + system entry acquire (decode). All kernargs are repacked into one contiguous host arena (slab views;
CPU patches still land in the host master).
- pack: dispatch from the packed host arena (contiguity control).
- gpu-copy: after the entry acquire, CP DMA_DATA host->VRAM mirror (TC_L2, CP_SYNC, write confirm) + K$/L0 acquire; dispatches read VRAM. No CPU BAR writes per replay.
- l2-prefetch: arena in the CPU-agent COARSE_GRAINED pool (flags 0x4); after the entry GL2 wb+inv, DMA_DATA DST=NOWHERE warms GL2.
Code: runtime.rs:2143 discover_host_coarse, :2906 slab field, :3020 slab_view; pm4_gfx10.rs:29,674,681,685 DMA_DATA;
replay.rs:303 emit_kernarg_stage, :3670-3705 mode/staging/stage_retained_kernargs, ~6657 gate, :6754 emit, :6852 dispatch addr, :7325;
hipfire-config lib.rs:3937; docs/env-vars.md, docs/CONFIG.md rows. Tests: hipfire-config 357 ok, redline-rocr 77 ok.

Decode tok/s 1025/3841/7681 | conc4 agg (Nex-N2.5-mini MQ4R q8, 4 slots vmm, same binary):
| mode | decode | conc4 |
|---|---|---|
| host (default) KP_host | 163.5/157.8/149.9 | 214.8 |
| vram (opt-in, faults) KP_vram | 193.3/183.7/171.6 | 256.5 |
| pack | 160.7/155.0/146.5 | 212.6 |
| gpu-copy | 194.8/185.3/172.6 | 241.3 |
| l2-prefetch | 185.2/175.1/164.0 | 231.9 |
EQ4K (EQ=36, chunk 4096, 144 cases): gpu-copy 0 mismatch / 0 errors; l2-prefetch 0 / 0.
Soaks (127+7681 gen1024, invariants on):
- KS_gpu-copy x280: FAULT at iter 37 (08:33:59 UTC). Signature identical to RC2: sq_intr wgp0, gfxhub page fault VA 0x0, SQC(data),
  STATUS 0x00801431, PERMISSION_FAULTS 0x3, vmid 8 -> MES REMOVE_QUEUE/SUSPEND fail -> wedged -> MODE1 reset (succeeded, VRAM lost).
  Daemon: QUEUE_TIMEOUT queue_id=153 read_index=11 write_index=12, PM4_PROGRESS end_tick=0. Iters 0-36 hashes correct.
- KS_l2-prefetch x280: 280/280 ok, 0 kfaults, all iterations hash ff4438867140/b8a625bc7b2f.
- KS2_gpu-copy x150: 150/150 ok, 0 kfaults, same hashes.
- KH_host x5 (default): same hashes ff4438867140/b8a625bc7b2f => staged modes match host output.
Findings:
- Contiguity is not the cost (pack ~2% slower than host). Kernarg location is: VRAM-resident kernargs recover the full 13-15%.
- RC2 is NOT caused by CPU BAR writes: gpu-copy has no per-replay CPU writes to VRAM and still faulted with the RC2 signature (1 of 2 soaks;
  37 iters, ~fault rate similar to vram). Points at VRAM-resident kernarg reads by dispatch (or something about device-local kernarg
  addresses) rather than CPU->VRAM visibility. Unproven.
- l2-prefetch recovers ~63-70% of the gap (185/175/164 vs 163/158/150 host, 193/184/172 vram), clean over 280 iters. Coherence of coarse
  CPU pool under CPU patches between replays: passed EQ + soak, not proven in principle (GPU may read non-snooped).

---

## Re-validation of feat/vmm-paged-kv-pm4-moe + review/vmm-fixes (2026-10-30)

Worktree `~/work/hipfire-validate`, branch `review/vmm-validate` (local only, not pushed):
`7b1be5929` (base HEAD) → `40d9d0401` gfx11 CWSR PRIV fix → `4034e3264` debug knobs
(FORCE_EVICT_MS, PM4_NO_CWSR_PRIV) → `8f9368f3d` revert of `129ab13ee` (retained PM4
kernargs default back to `vram`). NOTE: the revert also removes the HDP-flush on VRAM
publish and the vram-recycle path that 129ab13ee added. Release build into
`~/work/target-perf` (exit 0). Every run logs `retained PM4 kernargs: pool=vram`, `priv=` knob off.
Model Nex-N2.5-mini MQ4R, kv q8, 4 slots, ctx 16384. All GPU runs idle-gated, ≤10 min, kernel log checked.

### 1. Unit tests (--release, GPU hidden) — VALIDATED (no new failures)
| target | result |
|---|---|
| redline-rocr | 78/78 pass |
| rdna-compute --lib | 1184 pass / 0 fail / 30 ignored |
| kernel_registry_coverage | 20/21; `every_runtime_hip_module_is_registered_or_allowlisted` fails, 34 failures / 21 modules — identical on base |
| all_launcher_abi | 65/66; `changed_kernel_coverage_is_closed` fails on qwen4_moe_sym_nt1_gemm + qwen4_moe_sym_gemm ("`nk` is not an if/else block") — identical on base (line offsets only) |

### 2. Correctness — VALIDATED (no cross-request contamination)
EQ=36 rounds × 4 lanes = 144 cases/run, lengths 127..1153, gen 32, invariants on.
| run | config | mismatches |
|---|---|---|
| VC1/VC2/VC3 | PM4 + VRAM pool, default chunk | 18 / 18 / 16 (all at 1151-1153) |
| VCH1/VCH2 | HIP replay (no PM4) A/B control | 19 / 13 (all at 1151-1153, same hashes) |
| VC4K1-3 | PM4 + VRAM pool, chunk 4096 | 0 / 0 / 0 |
| VREG | check_vmm_paged_regressions.py, chunk 4096 | PASS |
| VB1/VB2, VBA1 | base 54b635cb non-paged / arena | 144/144 each (deterministic, base batched path ≠ solo; not investigated) |
PM4 12.0% vs HIP 11.1% at the chunk boundary → not PM4-specific; the suspected missing
HIP-stream→PM4 wait (~forward_slots.rs:4351) is NOT implicated by this A/B. Mismatch outputs
never equal another lane's solo output → chunk-boundary numerics class, not contamination.

### 3. Stress — VALIDATED (0 faults)
- VS12k: 12 cycles of 4×12000 concurrent, 48/48 HTTP 200, klog 0.
- VSPF1-3: 127+7681 gen1024, 36 iters each, every lane hash stable (ff4438867140 / b8a625bc7b2f, = host-pool reference), klog 0.
- VE1/VE2: forced eviction every 10 ms (≥11000 drops each), PRIV fix on, 243 iters, 486/486 200, klog 0.
  Rare 1025-lane variant 5c533f15009a at 3/122 and 1/121 (same rate as earlier fix-on runs).
- No fault occurred, so no PRIV-off A/B was run on this branch (dev-stack A/B: PRIV off 6/6 faults).

### 4. Speed — 3 interleaved runs/arm, per-run medians of 3 reps, mean shown
| arm | prefill 1025/3841/7681 | decode 1025/3841/7681 | conc4 agg |
|---|---|---|---|
| validate (VMM+PM4, vram) | 4866 / 4368 / 3718 | 192.0 / 183.6 / 170.5 | 257.7 |
| base 54b635cb non-paged | 4930 / 4490 / 3901 | 81.8 / 59.5 / 43.5 | 189.1 |
| base 54b635cb arena | 4934 / 4477 / 3900 | 81.6 / 59.4 / 43.5 | 188.8 |
Decode ×2.35/×3.09/×3.92 vs base; conc4 +36%; prefill −1.3%/−2.7%/−4.7% vs base.
Matches earlier HEAD-vram / KP_vram numbers (192.8/183.7/170.6, 244.9-256.5) and is well
above the host-pool default being reverted (164.7/160.0/150.2, 206.8).

### 3b. Extended stress (added after link drop; queueV2)
- VSPF4/VSPF5: 127+7681 gen1024, 50 iters each (540 s each), lane hashes 100% stable (same as VSPF1-3), invariants on, klog 0.
- 1025-lane variant 5c533f15009a, 127+1025 gen64, 120 s:
  | run | config | iters | 5c53 |
  |---|---|---|---|
  | VE1/VE2 | PM4, force-evict 10 ms | 122 / 121 | 3 / 1 |
  | VN1 | PM4, no eviction | 165 | 3 |
  | VNH1 | HIP replay, no eviction | 141 | 4 |
  It shows up at the same ~2-3% rate with no eviction and with HIP replay, so it is not
  caused by eviction, the PRIV fix, or PM4. Timing-dependent batching numerics
  (same class as the chunk-boundary mismatches), not contamination. All requests 200, klog 0.

### Verdict
| item | verdict |
|---|---|
| branch + build (PRIV fix, knobs, revert 129ab13ee → vram default) | VALIDATED (build exit 0; runtime logs pool=vram) |
| unit tests | VALIDATED: no new failures; the 2 failing tests fail identically on base |
| correctness / cross-contamination | VALIDATED: no contamination; default-chunk 1151-1153 mismatch is identical under HIP replay (not PM4); EQ4K 0/432; REG PASS |
| stress 4×12k, 127+7681, forced eviction | VALIDATED: 0 faults, klog 0 in every run; PRIV-off A/B not run because nothing faulted |
| speed | VALIDATED: decode ×2.35-3.92, conc4 +36% vs base; prefill −1.3..−4.7% vs base |
| not tested | the host-pool path after the revert; PRIV-off on this branch; models other than Nex-N2.5-mini MQ4R |
Nothing pushed; ~/github/hipfire not touched.

---

## 2026-10-30 (later): re-validation after merging beta bd61265fc, and a concurrent-vs-solo finding

The branch was merged forward to `upstream/beta` `bd61265fc` (`8ee106324`). Re-validated on
gfx1100 (7900 XTX, HIP 7.16 / ROCm 10.1), branch HEAD `07bf59e07`.

### What the merge itself broke, and what was fixed

| item | result |
|---|---|
| `cargo build --release -p hipfire-daemon -p hipfire-cli` | exit 0 |
| `cargo clippy --workspace --all-targets`, `cargo-fmt --all -- --check`, `scripts/check-crate-maps.py --check` | clean (a merge-induced `SseSink` initializer in a CLI test, four crate-map rows and three rustfmt hunks were branch fallout, fixed in `07bf59e07`) |
| `redline-rocr` 78, `rdna-compute` vmm_page_pool 9 / kv_slots 15, `hipfire-arch-qwen35` lib 502 | pass |
| `rdna-compute --test all_launcher_abi` 66, `--test flagship_pack_coverage` 12, `peacemaker-isa --test mq4_rows_x4` 19 | pass |
| `kernel_registry_coverage` `every_runtime_hip_module_is_registered_or_allowlisted` | **pre-existing upstream**: the identical 40-finding set (railgun_window/tp_peer_push/vit_*/stale allowlist rows) fails on beta `bd61265fc` itself |
| `kernels/mq4_rows_x4_pm_gfx1100.{s,co,hxaco,sha256}` | re-emitted from merged source (beta's gfx1100 SGPR change + this branch's dual-issue); gfx1151 artifacts re-emit byte-identically and the receipts (`.m7.json`, `shape.d/*`) are semantically unchanged, so they were left as committed |
| `scripts/pre-commit-ci.sh upstream/beta` | all checks pass after the fixes above |

### The finding: mixed-length concurrent prefill does not reproduce solo output

Route: `HIPFIRE_SERVE_MULTI_SLOT=1 HIPFIRE_SERVE_MULTI_SLOT_PAGED=1
HIPFIRE_SERVE_MULTI_SLOT_PAGED_BACKEND=vmm HIPFIRE_SERVE_MULTI_SLOT_PREFILL_CHUNK=4096
HIPFIRE_SERVE_MULTI_SLOT_SLOTS=4 HIPFIRE_SERVE_MULTI_SLOT_CTX=16384`, `--kv-mode q8`,
`hipfire serve --idle-timeout 0`, non-streaming chat completions, `temperature: 0`,
`chat_template_kwargs.enable_thinking: false`, 32 generated tokens, prompts built by
`check_vmm_paged_regressions.py`'s `filler_prompt` (calibrated to an exact `prompt_tokens`).

`python3 scripts/check_vmm_paged_regressions.py --port 11435 --model <9b.mq4>` — with the
chunk pinned, which is what makes the 1151-1153 class disappear — fails 24 of 144 cases, and
every failure is at the **page edges** 127/128/129/255/256/257, not at 1151-1153:

| lanes | result (9B, gfx1100) | solo reference |
|---|---|---|
| `[127,128]` | 127 -> `a0ce0c1f` | 127 `9eb2c1af` |
| `[127,128,129]` | 127 -> `a0ce0c1f`, 128 -> `9eb2c1af` (= **127's answer**) | 128 `c4c6d94c` |
| `[127,128,129,130]` | 127 -> `a0ce0c1f`, 128 -> `9eb2c1af`, 129 -> `c4c6d94c` (= **128's answer**) | 129 `9d8da3f5` |
| `[127,127,127,127]` | all four -> `a0ce0c1f`; a repeat run of the **same composition** gave 3x `a0ce0c1f` + 1x `9eb2c1af` | 127 `9eb2c1af` |
| `[127,128,257,300]`, `[255,256,257,300]`, `[127,128,255,256]` | no mismatch in that run | - |

Every prompt, token count and lane order was fixed across runs, so this is not a client-side
mix-up: the server reports `prompt_tokens` 127/128/129/130 for those lanes and returns a
completion that is byte-identical to a *different* length's solo completion. With four
*identical* prompts in one batch, the four lanes can disagree with each other inside a single
run, which also contradicts "output is deterministic for a fixed batch composition" for this
backend.

Dense 27B (`qwen3.8-27b.mq4p`, same flags) shows the same class at a lower rate: `[127,128,129,130]`
returned 130 -> `a6e0dc5a`, which is solo 127's answer; `[127,128,129]` and `[255,256,257,300]`
each returned one unique wrong answer. `[127,127,127,127]` and `[128,128,128,128]` were clean.

### What it is not

| candidate | test | outcome |
|---|---|---|
| the beta merge | same probes on pre-merge tip `38cb54fa7` (3 runs) | fails identically |
| the shared-prefill commit `fb573685e` | same probes on its parent `8f9368f3d` (3 runs) | fails identically |
| beta's batched lm_head (default on) | `HIPFIRE_BATCHED_HEAD=0` | still fails |
| the VMM page mapper | `HIPFIRE_SERVE_MULTI_SLOT_PAGED_BACKEND=arena` (same codes, arena backing) | still fails |
| client mixing | distinct-answer marker prompts, 129/130/257/256 tokens (off the page edges), 3 rounds | every lane returned its own marker, identical to solo |
| device state | `journalctl -k` | 0 GCVM / page faults; all requests HTTP 200 |

So the engine keeps per-lane request state (the marker probe passes), and the failure is in the
**batched prefill row/token accounting at page-edge lengths**: a lane's prompt is executed as a
prefix of itself, which makes its decode continue from another length's prompt.
The marker probe's lengths were not page edges, which is why it stayed clean.

Not confined to dense: the A3B `.mq4r` route (arch 6 `qwen3_5_moe`, retained PM4 armed by the
loader — `[redline] slots: enabling fail-closed retained default on gfx1100 (...) transport=pm4`),
where MoE prompt chunks run as isolated single-slot `_vmm_paged` steps, was byte-identical to solo
for two equal-length lanes across 3 rounds, and matched again for a 360-token code prompt; but a
mixed pair (360 + 25 tokens, `enable_thinking: false`, 160 greedy tokens) differed once in two
rounds on the short lane. Coherent text throughout, 0 kernel faults.

### Status

Open. Not fixed in this branch: it predates the last two commits, needs the batched-prefill
row/slot/position accounting traced against the paged block table, and no fix can be claimed
without repeating the EQ matrix, the page-edge matrix above and the 4x12000 soak. The route
stays **experimental and not an admission route** (see `docs/VALIDATION.md`), and
`check_vmm_paged_regressions.py` is the reproduction: it is expected to report the page-edge
mismatches on gfx1100 until this is fixed.
