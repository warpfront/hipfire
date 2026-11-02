# railgun for the NPU (XDNA2 / AIE2P, Strix Halo NPU5)

> **Lifecycle (0.4.2): design note for the dark `railgun::npu` runtime; §7 describes
> the transport under the Flash-Next trunk route.** §§0–6 (prepared commands, lean TXN,
> persistent ring, grouped experts, tape) are research designs that no serving path
> calls. §7 is the transport that the default-`auto` Flash-Next gfx1151 NPU trunk uses
> for its **input projections** (`GdnIn` / `QsaIn`, calibration schema 4, BF16-direct
> only; there is no I32 contract). The out-projection shapes (`GdnOut` / `QsaO`)
> exist in the runtime and in the CPU packers and programs, but **no serving route
> submits them**; they are not shipped, and `qsa.o` / GDN o-proj are not offloaded.
> For the current route, schema and environment see [`npu-trunk.md`](npu-trunk.md)
> and the root `CHANGELOG.md`. Timings in this file are fixture-bound or historical
> measurements from the dates stated, not current performance claims.

Retained, certified, low-overhead dispatch for our own amdxdna runtime (`railgun::npu`, DRM ioctls only), plus the
CPU-free GPU→NPU work ring that GPU+NPU co-op needs. Everything here is Rust; no driver or firmware change.

Measured results and logs: `report.md`, section "railgun for the NPU". This note is the design: the four
mechanisms, the patch inventory, the hash-keyed program cache / step tape, and the exact interface the GPU side
(hipfire railgun PM4) uses.

## 0. Driver facts this design relies on (linux 7.0 `drivers/accel/amdxdna`)
| fact | source |
|---|---|
| `ERT_CMD_CHAIN = 19`; payload `{u32 command_count, submit_index, error_index, reserved[3]; u64 data[count]}` (CMD BO handles) | `amdxdna_ctx.h:15-58` |
| chain = one 4 KiB firmware cmdbuf, one mailbox message, one response; 112 B per START_CU slot ⇒ ≤ 36 commands | `aie2_message.c:960-1030`, `aie2_msg_priv.h:364,384-396` |
| failure reports `error_index`; sub-command headers are not updated (only the chain BO); on error the driver memsets the (first) cmd payload to 0xff | `aie2_ctx.c:229-282`, `amdxdna_ctx.c:140-160` |
| `force_cmdlist = true`: every single START_CU already travels as a chain of 1 | `aie2_ctx.c:26` |
| the driver sets the cmd state to NEW itself at job run, so a retained cmd BO needs no header rewrite | `aie2_ctx.c:313` |
| EXEC_CMD takes exactly one cmd BO; arg BOs (every BO any sub-command touches) ≤ 4095, pinned per job | `amdxdna_ctx.c:592-617` |
| job timeout 60 s (`HWCTX_MAX_TIMEOUT`) | `aie2_ctx.c:30,538` |
| UMQ / doorbell fields exist in the uapi `CreateHwctx` but the aie2 driver never reads them: no user-mode queue | uapi `amdxdna_accel.h`; no `umq`/`doorbell` use under `drivers/accel/amdxdna` |

## 1. Prepared programs and in-place patching (`railgun::npu`)
`HwCtx` + `config_cu(PDI)` + the instruction (TXN) DEV BO are built once per design/shape. A `Prepared` is one
retained ERT_START_CU command BO (17 words) plus its retained EXEC_CMD handle array:

| word | content | hot path |
|---|---|---|
| 0 | header `NEW | 16<<12 | START_CU<<23 | ERT_CU<<28` | never (driver resets state) |
| 1 | cu_mask = 1 | never |
| 2, 3 | TXN opcode 3, 0 | never |
| 4, 5, 6 | insts device address lo/hi, insts bytes | never (a different program = a different `Prepared`, see §5) |
| 7+2i, 8+2i | arg i device address lo/hi (i < 5) | **patched** by `patch_arg(i, &Bo)` only when the BO changes; only that cache line is flushed |

`Chain` = one retained ERT_CMD_CHAIN BO over N ≤ 36 `Prepared` (payload never rewritten; the deduplicated arg
handle array is rebuilt only when a sub-command's handle changed). `HwCtx::submit_prepared` / `submit_chain`
issue one EXEC_CMD with no encoding; one syncobj wait per chain.

## 2. Lean TXN (`crates/peacemaker-npu/src/kernels/gemm_array_lean.rs`, `ArrayDesign::lean_insts`)
The vendor-style dynamic TXN of the V8/V9/V10 pair designs re-initialises the whole array every submit (core
reset, 176 channel resets, all memtile descriptors, 512 core + 512 memtile lock writes, ring tasks; ≈ 51 KB,
1746 ops). Measured on silicon that costs ≈ 150 µs per GEMM of firmware time. The lean TXN keeps the array
configured across submits and repairs only what a completed submit leaves in a non-initial state, derived
analytically per shape from completed BD loads per channel (`J = kc·waves`, `W = waves`, `NW`, `MW` parities;
table in the module doc) and proven in the simulator by `state_snapshot()` equality with a full submit
(locks, every BD register including iteration `current`, channel engine positions, stream FIFOs).

Contract: valid only directly after a *completed* full or lean submit of the same PDI on the same hwctx (no other
hwctx may run on the partition in between). The first submit after `config_cu` is always the full TXN.

| design / shape | full TXN | lean TXN | lean ops per column |
|---|---:|---:|---|
| V9 gate_up 4096×1280×2560, down 4096×2560×640 | 51,056 B | 11,056 B | 4 cores × (halt/reset, release, PC, enable); shim BDs+patch+tasks; reset+requeue memtile S2MM4 / MM2S1; B_EMPTY lock; clear `current` of 8 B-replay BDs |
| V9 periodic (NW even) | 48,368 B | 7,792 B | cores + shim only |
| V10 down | 48,560 B | 7,984 B | cores + shim + requeue the finite B fill |
| V10 `--b-prefix` down / 160×2560×640 | 49,136 / 48,944 B | 8,816 / 15,824 B | cores + shim + requeue the finite fill / guarded-replay groups in order (+ BD 36 when MW > 1), clear their iteration `current` |

## 3. Persistent work ring P1 (`crates/peacemaker-npu/src/kernels/ring.rs`)
One START_CU whose TXN runs S slot-runs back to back with no CPU in between. The NPU-side "controller" is the
firmware's TXN interpreter itself: the shim DMA cannot branch on data, so a self-looping shim BD mirrors the slot's
`seq` word from DDR into memtile memory and the TXN `MASKPOLL`s it.

### 3.1 Shared memory (frozen, version 1)
One buffer (CPU `shmem_bo` in the CPU-producer harnesses; in co-op, anonymous host pages pinned by amdxdna as a userptr BO
(`Device::userptr_bo`) and `hipHostRegister`ed for the GPU — NOT a HIP VMM dma-buf export, which stops aliasing after the
import: report.md amendment 2026-10-03), 64-byte lines, little endian. `nslots` is a power of two.

| line | offset | writer | content |
|---|---|---|---|
| header | 0 | host, once | `u32 magic 0x504E4752 ('RGNP')`, `u32 version 1`, `u32 nslots`, `u32 64`, `u32 sentinel 0xFFFFFFFF` (byte 16) |
| slot s | `64·(1+s)` | **producer (GPU)** | `u32 seq` (byte 0, written LAST), `u32 prog`, `u32 m`, `u32 flags`, `u64 a_off`, `u64 b_off`, `u64 c_off` (P1 reads only `seq`; the rest is informational) |
| done s | `64·(1+nslots+s)` | **NPU only** | `u32 done_seq` (byte 0), `u32 slot`, `u32 0x454E4F44 ('DONE')`, zeros — one 64-byte DMA write |

Run k (global, monotonic) uses slot `s = k mod nslots` and `seq = k + 1` (never 0, never the sentinel).
Within a retained persistent command of S runs started at `seq0`, run j has `seq = seq0 + j`.

### 3.2 NPU side, per run j (slot s)
1. POLL: reset+release memtile(0) S2MM4; `WRITE32 X := 0`; queue shim(0) MM2S0 BD P (1 word ← slot line s, `next = P`)
   and memtile(0) S2MM4 BD Q (1 word → X, `next = Q`); `MASKPOLL X == seq`.
2. DRAIN: one `WRITE32` flips P's `next` to the finite sentinel BD SENT; `MASKPOLL X == 0xFFFFFFFF` (in-order
   stream ⇒ every poll word has landed, nothing stray reaches the GEMM's B channel).
3. BODY: the V9 GEMM TXN with shim DDR patches advanced by the slot's fixed arena offsets; ends with both C SYNCs
   (all 16 C channels joined by the firmware; design gate 2). `persistent_v9_lean`: run 0 of a command takes the full
   body, runs 1.. the lean body (§2) plus a forced reset/requeue of memtile(0) S2MM4 and MM2S0, the two channels POLL
   and DONE borrow (136 KB vs 416 KB of TXN per 8 slots).
4. DONE: reset+release memtile(0) MM2S0 (strictly after the C SYNC); 16 `WRITE32` stage {seq, s, 'DONE'} at memtile
   Y; memtile MM2S0 BD DQ → shim S2MM0 BD D (token) writes the 64-byte done line; `SYNC(shim 0 S2MM0)`.

Spare resources (asserted against V9's own at the maximal topology): shim BDs P=4, SENT=5, D=6; memtile BDs Q=10,
DQ=11; memtile scratch X=0x1000, Y=0x3000 (never touched by any V9 DMA). Latch argument for the single-word flip:
`ring.rs` module doc.

Limits of P1 (stated, not hidden): one shape/PDI per persistent command; each slot's operand offsets are fixed when
the command is built (the GPU writes operands *into* the slot's arena, it does not pass arbitrary addresses);
multi-round retained resubmission needs `S mod nslots == 0`; a whole command is bounded by the 60 s job timeout.
Dynamic per-slot addresses would need the NPU to rewrite shim BD registers from DDR (control packets from a
controller core or from the shim DMA to its own TileCtrl port) — a P2 option, not built.

### 3.3 CPU role
The CPU never sits between a producer write and the NPU start, or between the NPU done and the consumer. It
re-arms the queue: per round it `patch_seq`s the retained insts BO (the declared words only) and submits the same
cmd BO; rounds can be queued ahead (the driver keeps them in order), so re-arming never synchronises with the GPU.

### 3.4 Grouped expert launch (`crates/peacemaker-npu/src/kernels/experts.rs`, `grouped_v9`)
One START_CU (optionally one ring slot) runs E experts of one projection back to back: one PDI / ConfigKey for every
expert with M ≤ 512 (one 512-row wave), expert 0 the full body, experts 1..E−1 the lean body, each with its own
A rows / B (expert weights) / C arena patches. No per-expert full TXN, no per-expert submit. Experts are serialized
by completion (each body ends with its C SYNCs). Size: 147 KB (E=8), 485 KB (E=32), 935 KB (E=64), all run on silicon.
A design that wants to join (e.g. G80) exposes `append_run_body(txn, arena_base)`, `append_lean_run_body(txn,
arena_base, extra_requeue)` with its case in the lean derivation, `pack_in`/`unpack_out`/`reference`, and its wave M.

## 4. GPU-side interface (hipfire railgun PM4)
Addresses are GPU VAs of the shared host pages (`HipRuntime::register_host`): `ring` (ring base), `a_arena`, `b_arena`,
`c_arena`, and the per-slot offsets fixed when the persistent command was built (`slot_a_off[s]`, `slot_c_off[s]`).
Measured implementation: `npu_tools::coop_gpu` (PM-native kernels, a polling kernel instead of PM4 packets) and
`npu-coop`; report.md "coop-w2" (publish → NPU done → GPU observe p50 3.13 µs, exact GEMM slots).

Producer, run k (slot `s = k mod nslots`, `seq = k + 1`):
1. Slot free: if `k ≥ nslots`, `WAIT_REG_MEM` (memory space, function `==`, mask 0xFFFFFFFF, reference `seq − nslots`)
   on `ring + 64·(1+nslots+s)`. Equality, never ≥: wrap-safe and cannot pass on a skipped slot.
2. The producing kernel writes A (and any per-slot operand) to `a_arena + slot_a_off[s]`.
3. Make those writes visible in memory before the publish. Measured: on the userptr / `hipHostRegister` pages, stores
   issued `glc slc dlc` and completed (`s_waitcnt_vscnt 0`) are seen by the NPU (and the CPU); the A copy kernel's end
   of kernel precedes the publish kernel in the same stream.
4. Optional slot body: `WRITE_DATA` of words 1..15 of `ring + 64·(1+s)` (prog/m/flags/offsets; P1 ignores them).
5. Publish: `WRITE_DATA` (dst = memory, `WR_CONFIRM` = 1) of `seq` to `ring + 64·(1+s)`.

Consumer, run k: `WAIT_REG_MEM` (memory, `==`, mask 0xFFFFFFFF, reference `seq`) on `ring + 64·(1+nslots+s)` — measured
form: a poll loop of `glc dlc` loads — before reading C at `c_arena + slot_c_off[s]` (loads `glc dlc`).

Ordering guarantees from the NPU side: the done line is written only after the firmware's SYNC on every C channel's
task-complete token, so every C byte was handed to DDR before the done write is issued; the done line itself is a
single 64-byte DMA write.

Schedule (`npu-coop`, measured): producer and consumer are two in-order GPU queues, not one. The consumer, after
reading C of run k, stores `seq` to a GPU-only `consumed[s]` line; the producer's slot-free wait is
`consumed[s] == seq − nslots` (stricter than step 1: it also covers the C arena) and then runs up to `nslots` runs
ahead, so seq(k+1) is published while the NPU computes run k. On one queue the copy / publish of run k+1 waited for
the C read of run k and the NPU idled between runs (gemm 512×1280×2560: 271 → 230 µs per run).

Rounds are queued one ahead in `npu-coop`: round r+1's re-armed command (`Bodies::LeanFirst`, two retained command /
insts copies) and its GPU work are issued before the CPU waits for round r, so each command's ~45–50 µs start-up
(measured on the empty ring, independent of TXN and BO size) overlaps the previous round. With one shared B on G80
(`persistent_with(.., b_shared = true)`), lean runs keep the whole-K B resident in the memtile: no shim B stream, no
memtile fill, the B ready locks are written with the `kc` credits the fill would have released.

## 5. Hash-keyed program cache and step tape (`crates/railgun/src/npu/tape.rs`)
Mirrors the GPU railgun replay tape (`replay_sequence_hash`, FNV-1a 64).
- `ProgramKey` = FNV-1a64 over `'P' ‖ len ‖ PDI ‖ 'T' ‖ len ‖ TXN template ‖ 'D' ‖ count ‖ {word index u64 ‖ kind tag}`.
  The TXN template has every declared dynamic TXN word (ring `Seq`) zeroed, so current values never change the key,
  but moving one word between static and dynamic does. Kind tags: ArgAddress 0x01‖arg, InstsAddress 0x02,
  Seq 0x03‖run u32, Custom 0x04‖id u16.
- `ConfigKey` = FNV-1a64 over `'C' ‖ PDI`: the array-configuration identity.
- `ProgramCache`: key → retained insts BO; every hit byte-compares the stored canonical bytes, a mismatch is a
  "hash collision" error, never a reuse.
- Step `Tape`: ordered entries (program key, config key, binding kinds); `sequence_hash` covers keys and binding
  identities but not values, so per-step address/seq changes replay the same tape. `check_replay` detects a stale
  key (program bytes changed since recording).
- Lean TXN is keyed here: an entry whose ConfigKey equals the previous entry's (same hwctx, array still
  configured) takes the lean program; otherwise the full one. Gates: collision + stale-key negatives, replay
  byte-equal to a fresh encode for every entry (`verify_fresh`), every changed word declared (`verify_patch`),
  outputs exact vs eager.

## 6. Certification
| gate | how | negative control |
|---|---|---|
| exact vs CPU | first + last (and `--verify-every`: every) submit/slot vs the CPU reference | — |
| byte-exact replay | prepared/chain/lean/ring C buffers byte-equal to an eager full-TXN submit of the same operands | — |
| patch inventory | snapshot every retained BO, patch, diff: changed words ⊆ declared sites, result == fresh encode | stale patch (skip one arg patch): caught by the fresh-encode diff and by the output check |
| ring done protocol | per-slot done lines, equality waits | skipped done word (`Neg::SkipDone`): the missing done line is reported; stale seq patch: 0 of 16 words patched and no new done lines |
| ring overrun | producer guard `producer_may_publish` (done[s] must equal seq − nslots) | guard refuses; forced in the simulator ⇒ MASKPOLL deadlock reported |
| AIE2P control rules | `crates/peacemaker-npu/src/isa/rules.rs` in `Program::finish` (unchanged core programs) | its own unit tests |

## 7. Trunk BF16 transport (0.4.2)

`railgun::npu::trunk::NpuTrunkRuntime::open(Vec<PanelProgram>, Vec<LayerPanels>, banks, rows_max, pci)`
accepts one `build_panel_program_rowmajor(rows_max, shape.pitch(), 0)` PDI/TXN
per shape and resident `pack_b_bf16` panels including Sm/Se. The shape programs
share the same G80 PDI/context but retain distinct pitch-specific instruction BOs.
The out shapes (`Shape::GdnOut` / `Shape::QsaO`, K=6144, five 512-column panels,
`build_panel_program_n512(rows_max, 2560, 0, 6144)`) are supported by the runtime but are
**not requested by the 0.4.2 serving route** (out-projection is unwired; see the
lifecycle note at the top). They run a different PDI: the relay creates one
`HwCtx` + `config_cu` per distinct PDI (input shapes share one, both out shapes share one),
submits serially from its single thread, and routes each chain to its shape's context. Each
bank additionally reserves `a_out` (`rows*6912` B, `a_out_arena(bank)`) and BF16 out C arenas
(`rows*2560*2` B); padding for out shapes is 512-row waves. Whether two contexts coexist and
what a context switch costs is measured by `npu-trunk-transport --coexist` (an
investigation probe, not a serving gate).
GdnIn/QsaIn use 13/11 START_CUs in a retained chain per (layer, shape, bank).
The program declares its padded row extent; rows never patch the program.
The caller zeros all padding codes/scales. A is one `pack_a_bf16` stream, not
13 copies. Each bank owns separate fused row-major C arenas: at 8192 rows,
A=23,592,960 B, GDN C=272,629,760 B, QSA C=230,686,720 B.
Panel p binds C at `arena + 2*p*1280` bytes. Bounds check the actual addressed
span `2*((padded_rows-1)*pitch+1280)`, rather than incorrectly adding the
full fused arg2 declaration to each panel offset. B residency is unchanged.
`a_arena(bank)`/`c_arena(bank, shape)` return the BO handle/NPU address,
anonymous host pointer, HIP device pointer and logical length. The BO owner
stays on the relay thread. No command/TXN words are changed after open.

The existing dlopen'd `npu::hip_runtime` is used rather than adding hip-bridge to railgun's
runtime dependency graph. HIP events/registrations remain !Send; only immutable entry points
and event handles cross to the relay, which selects the original GPU on its own thread.
`unsafe kick(layer, shape, bank, rows, raw_hip_stream)` records an explicit
`hipEventReleaseToSystem` event after A production. Bounded event queries precede NPU submit.
The chain's transaction drains all C channels. A chain-only watchdog defaults to
300 ms (`CHAIN_WATCHDOG`, five times the measured ~60 ms expected chain); explicit
`open_with_watchdog` budgets must be positive and at most 3 s. GPU A-ready polling
has a separate 3 s deadline. COMPLETED is checked before CPU-alias invalidation.
`wait_host(Seq)` propagates submission, event, timeline and watchdog errors.
`unsafe wait_gpu(Seq, stream)` immediately queues `hipStreamWaitValue32(>= seq)`
and a system-fenced event before C loads: **no host-completion check or wait**.
Every kicked generation gets a terminal release-store/CLFLUSH, including NPU
errors, watchdog expiry, enqueue failures, relay unwinding and runtime shutdown.
Failure is sticky and recorded before releasing its GPU wait; a late success
cannot overwrite it. The signal means terminal, not necessarily successful.
Retirement checks known runtime errors without host waiting, and the pair waits
for its final already-enqueued chains and checks errors before head/capture work
or state commits. Failed pairs return an error; their speculative results are
never published or used. Raw signal waits bypassing this contract are unsupported.
Per-bank sequences never wrap (reopen at i32::MAX).
`unsafe retire_bank(Seq, stream)` records the last consumer's event. Before
the next GPU producer writes A, `unsafe prepare_bank(bank, stream)` queues
`hipStreamWaitEvent` on that retirement event; the following kick's A-ready
event then orders the next NPU submit. Callers that do not prepare still need
completed retirement before kick. `Completion::{submit_ns,done_ns}` are relay
CLOCK_MONOTONIC timestamps (submit issue and completed timeline observation).
Serving records retirement after mixer Core (all direct arena readers), before
the out projection (which stays on the GPU in 0.4.2). `prepare` waits retirement on the producer stream, A production follows,
and kick's ready event follows both; the relay must observe ready before submit.
This orders **all old C reads before the next C DMA write**, not merely A reuse.
`close()` first cancels outstanding generations and releases all their GPU waits,
then joins the relay (context before BOs), checks GPU retirement, and HIP
unregisters before unmapping. It also reports the sticky NPU failure. On
unretired/stuck GPU work, teardown reports an error and quarantines
registrations/pages until process exit instead of freeing live operands.

Host-side reuse protection remains at `prepare_bank`/`kick` so a bank's ready
event is not re-recorded before its old NPU job observes it. These bounded waits
depend only on a previously issued producer/chain, not an unqueued GPU fragment
or the just-enqueued `G` fragment. For `prepare/kick(l+1,A)`, the old-chain wait
is for NPU(l,A), whose terminal signal is a prerequisite of GPU G(l,A).
Consequently this reuse check resolves no later than G(l,A)'s start; the host
can enqueue wait(l,B)+G(l,B) while the GPU still runs G(l,A). Retirement is only
a stream-enqueued event dependency, not a host wait for G(l,A).

Visibility is deliberately stronger than a release-value fence:
[coop-flashnext §3](coop-flashnext.md#3-buffer-ownership-and-visibility) requires compute
completion/GL2 writeback before A publication and acquire/GL2 invalidate before C loads;
[§4](coop-flashnext.md#4-per-layer-synchronization-timeline) places these boundaries at each
handoff. The [HIP event contract](https://github.com/ROCm/hip/blob/amd-staging/include/hip/hip_runtime_api.h)
defines system sequentially-consistent event fences with cache writeback/invalidation
(`hipEventDisableSystemFence` explicitly disables them; this runtime never sets it).
Producer and post-wait consumer events request system release, never device release or
timing-only fence suppression. CPU CLFLUSH does **not** replace the GPU boundary.
The memory class is anonymous pages + amdxdna userptr + default coherent hipHostRegister:
never exported HIP VMM (snapshot-only), never imported-BO SYNC_BO.

A and C are flushed **once at open**, after anonymous-page initialization and
registration/userptr setup. In production the CPU never reads or writes either
arena after that flush, so it holds no arena cache lines. GPU system release
publishes A writes to DRAM before the relay submits NPU DMA reads. NPU DMA writes
C to DRAM; the transaction tail completes all C channels before timeline done.
The GPU alone consumes C, through registered memory after the terminal device
wait and system acquire. Thus per-chain A/C CLFLUSH is unnecessary in production;
it cannot replace either GPU fence. The terminal signal's release store and
64-B CLFLUSH remain on every completion.

`NpuTrunkRuntime::open_with_options` explicitly opts CPU oracle users into
`OpenOptions { cpu_reads: true, ..Default::default() }`: A is flushed before
submit and C after completion, before publishing the terminal word. The
transport gate uses this option and compares both CPU arena aliases against the
GPU-produced/downloaded bytes. Production `open` defaults to device-only access.

Hardware gate (run on a Halo host): `target/release/npu-trunk-transport --pci <npu-bdf>`, where `<npu-bdf>` is the NPU's PCI address on that host (the flag has a built-in default for one reference host's enumeration).
It uses default shim AXI, two production-size banks, GPU shader copies of CPU-packed A,
and both row-major shapes (13 GDN / 11 QSA panels), 20 alternating rounds each.
Every live BF16 value at `2*(row*pitch+panel*1280+col)` is compared against CPU
integer dots + `bf16_direct`; padding columns are checked for zero. A CPU
reader-window gate checks all four consumer families terminate before padding
(GDN 16480..16640, QSA 13952..14080). The diagnostic full-C shader download
intentionally includes padding; production Core readers do not. The separate
`npu_trunk_consumers_check` poison oracle exercises actual arena kernels.
C is poisoned before each round. Each shape's 20 exact rounds and `EXACT`
p50/p99/max, busy-envelope/wall summary run **before** its negative controls.
The deterministic missing-wait negative then reads
fresh poison without submitting a replacement job; a racing ungated read could legitimately
finish after NPU completion and is not a reliable negative. A deliberately unsubmitted host
Seq must error immediately in process (no DRM wait or relay request). The former impossible
DRM timeline probe was removed after a reported receive timeout hid all rounds.
Every exact round asserts `gpu_wait_enqueue_ns < npu_done_ns`. A timing event
immediately after the terminal wait/system acquire measures NPU done to GPU
resume; the `RESUME` p50/p99/max summary includes the host/HIP clock-anchor
bracket uncertainty. A third negative opens a separate valid zero-operand runtime
with a deliberately impossible 1 ns chain budget, exercising the real watchdog
path after EXEC_CMD (no production fault-injection flags). Its post-wait event
must complete within the normal 300 ms watchdog; `wait_host`, sticky error checks,
retirement when already failed, and checked teardown must surface the failure.
The failed C arena is never read. Expected output per shape: 20 enqueue-ahead
exact round lines, `EXACT` and `RESUME` summaries, three negative-control lines,
and `PASS shape=GdnIn|QsaIn rounds=20 negatives=3 consumer_padding_bounds=checked`.
Busy is the host submit-to-timeline
envelope, not an AIE hardware counter; wall includes CPU verification. Any mismatch/error
exits 1 and identifies `stage=... round=...` (`round=none` for setup/negatives/teardown).
The first-light transport route passed on Halo; the enqueue-ahead ordering, forced
watchdog release and resume-latency numbers of this updated gate are not recorded in this
document (see the root `CHANGELOG.md` for the recorded 0.4.2 exactness results). No hardware is opened by the CPU unit tests:
`cargo test -p railgun --features npu --lib` and
`cargo test -p npu-tools --bin npu-trunk-transport`.
