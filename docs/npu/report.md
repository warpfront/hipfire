# NPU (XDNA2 / AIE2P) on Strix Halo — railgun::npu + peacemaker-npu bring-up

> **Lifecycle: historical evidence log (2026-10-02 onward), not a current-behavior
> reference.** This file records bring-up windows, probes and dated measurements on one
> Strix Halo test host. Every number is bound to the date, commit, shape and fixture it
> names; none is a current performance claim or a serving admission. Statuses such as
> "PASS", "KILL", "not run" and "Fast core still failing" describe that day and are
> frequently superseded by later entries below. Designs measured here that were killed
> or never wired (int32 epilogue / `I32` partials with a GPU fold, out-projection
> programs, co-op epilogue folds, the GPU↔NPU co-op ring) are **not** shipped serving
> paths. The 0.4.2 Flash-Next gfx1151 NPU trunk route is input projections only,
> calibration schema 4, BF16-direct only (no I32 contract), default `auto`; for the
> current route, schema and `HIPFIRE_QWEN4_NPU_TRUNK` / `fusions.npu_trunk` /
> `fusions.npu_trunk_calibration` behaviour see [`npu-trunk.md`](npu-trunk.md) and
> the root `CHANGELOG.md`. Logs referenced as `logs/…` live in the original NPU
> workspace and are not part of this repository.

Written 2026-10-02 onward by the NPU bring-up workstream. The NPU workspace repository (git `main`) was built on a
build host; the Halo test host ran from clean git worktrees cloned from it (no rsync). Raw window logs were collected
under `logs/` of that workspace (not copied into this repository). Milestones M1–M4 below follow the 2026-10-02 assignment (M1 hello completion, M2 single-core
GEMM on hardware, M3 multi-core GEMM at expert shapes, M4 GPU+NPU concurrency).

Status as of 2026-10-02 09:00 UTC (historical): **no GO/NO-GO verdict** — per the project directive of that date, none is given until our own int8 kernel runs.
No vendor compiler/runtime was used to measure the NPU.

## Phase 0 — host inventory (Halo test host, read-only)

| item | value |
|---|---|
| kernel | 7.0.0-38-generic, `amdxdna` in-tree (`drivers/accel/amdxdna`, DRM driver 0.7.0) |
| device | `<npu-bdf>` `1022:17f0` rev `0x11`, `vbnv=RyzenAI-npu5`, PCIe link 16 GT/s x16 (internal) |
| firmware | `amdnpu/17f0_11/npu_7.sbin` loaded at boot; sysfs `fw_version=1.1.2.65` |
| node | `/dev/accel/accel0` `crw-rw---- root:render` (the test user is in group `render`) |
| XRT | Debian `libxrt*` 2.21.75 + `xrt-smi` installed (not used by our path) |
| IOMMU | `iommu=pt` on the kernel cmdline; default domain Passthrough; NPU iommu group 37 type `identity` |
| geometry | 8 columns (xrt-smi platform); rows: shim 0, mem tile 1, core 2..5 (QUERY_AIE_METADATA below) |
| memlock | ssh sessions get RLIMIT_MEMLOCK 8 MiB despite `/etc/security/limits.d/{90-memlock,99-amdxdna}.conf` = unlimited; our wrapper raises it per process with `prlimit` (not a host change) |

Fabric clock control (standard amdgpu sysfs, kernel docs for `power_dpm_force_performance_level` / `pp_dpm_fclk`): for each amdgpu device
with `pp_dpm_fclk`, write `manual` to `power_dpm_force_performance_level`, write the top level index to
`pp_dpm_fclk`, then poll up to 30 s until the read-back shows `manual` and exactly one `*` on the top
line; `release` writes `auto`. Our `tools/npu/npu-window.sh` applies the same method to the Strix Halo iGPU
only (fclk levels 400…2000 MHz), refuses to start unless it owns the exclusive
host timing-lock hold and the incoming level is `auto`, and always restores `auto`. The pin / verify /
restore code is `crates/railgun/src/npu/fclk.rs` (`npu-tools fclk pin|status|restore`, called by the script); concurrent
GPU+NPU binaries (`npu-coop`, `npu-tools m4`) refuse to start unless the pin is held (`NPU_FCLK_GUARD`).

## Specs read (not run)

- ISA: llvm-aie `llvm/lib/Target/AIE/aie2p/*.td` + `MCTargetDesc/aie2p/AIE2PMCCodeEmitterGen.inc`.
- Registers: aie-rt `release/main_aig` `driver/src/global/xaie2pgbl_params.h`.
- Transaction stream: mlir-aie `include/aie/Runtime/TxnEncoding.h`, aie-rt `common/xaie_txn.h`.
- CDO/PDI: bootgen `cdo-driver/`, `versal/include/imageheadertable-versal.h`.
- Driver/UAPI: kernel 7.0 `include/uapi/drm/amdxdna_accel.h`, `drivers/accel/amdxdna/aie2_message.c`.

## Milestones

### M1 — npu-hello: syncobj completion and clean teardown, 3/3 — **PASS** (2026-10-02 09:57 UTC)

Run 1 (08:42 UTC, `ec06e20`) landed 64/64 words but timed out on completion (missing shim TILE_CTRL→SOUTH0
token route) and its teardown logged `aie2_hwctx_restart: Config cu failed, ret -22` (PDI BO closed before
the context). Fixed in `1a3eeac`. Run 2 never ran (killed while queued); batches `18eda6b` and `bba86ed`
got windows at 09:17/09:19 but failed `Permission denied` (binaries scp'd without the exec bit — fixed:
`chmod 755` after copy + preflight `test -x` in `tools/npu/npu-batch.sh`, which now also exits non-zero when any
step fails). Measured run: batch `2320869` (binaries built from clean `bba86ed`, sha256 hello `c182306e…941c`,
gemm `7cbcac1a…2f8e`), log `logs/batch-2320869.log`.

| check | measured |
|---|---|
| window | fclk pinned after 5.5 s (`manual`, top level `2000Mhz *`), pin still held after the batch, restored to `auto` and verified |
| completion | 3/3 fresh hardware contexts: `wait=Ok(4)` (ERT_STATE_COMPLETED) at 161.6 / 124.6 / 129.4 µs submit→completion |
| data | 3/3: 64/64 words `0xC0DE0000+i`, 0 mismatches |
| teardown | kernel log during the window: **no amdxdna/amdgpu lines** (no hwctx restart, no Config-cu failure) |

### M2 — single-core int8 GEMM on hardware, CPU-exact — **PASS** (same window, batch `2320869`)

`npu-gemm M M M --iters 3` runs `gemm_i8::design` (one core, col 0 row 2; nop-padded scalar-scheduled 8×8
microkernel; A/B packed per 8×8 output microtile and streamed through the memtile; two DDR-patched args).
Fresh hwctx per iteration; time = submit→syncobj completion; exact compare against the CPU int8×int8→i32.
Job limit raised from 256 to 1024 microtiles (`bba86ed`) so 256³ runs as one design.

| shape | iter 0 / 1 / 2 (µs) | useful GOPS (best) | mismatches |
|---|---|---|---|
| 64³ | 200.3 / 165.2 / 144.4 | 3.63 | 0/4096 ×3 |
| 128³ | 237.3 / 217.0 / 204.7 | 20.5 | 0/16384 ×3 |
| 256³ | 524.7 / 516.9 / 486.9 | 68.9 | 0/65536 ×3 |

These are latency-dominated single-core numbers (one core, every 8×8 output re-streams its full A row and
B column: 4 MiB of input for 256³); they prove the encoder → PDI/TXN → railgun::npu path end to end, not speed.
SMU metrics (current decoder `npu-tools soc-metrics`, read-only `gpu_metrics` v3.0: DRAM/NPU MB/s, NPU clocks) were sampled at
20 ms during the batch (`logs/metrics-2320869.txt`); the SMU fields are running averages that only ramped
(ipuclk 7→315 MHz) over the 0.3 s batch, so they say nothing about short runs.

### M3 — 32-core int8 GEMM at Flash-Next expert shapes — **PASS on hardware with the Serial core** (2026-10-02 15:42 UTC); Fast (VLIW) core still failing

Shapes (from the Flash-Next parent model `config.json`: hidden 2560, moe_intermediate 640,
512 experts, top-10): gate_up = [M×2560]·[2560×1280], down = [M×640]·[640×2560]. (The assignment's
"hidden 4096-class" is 2560 in this config.)

Offline (all gates pass on `main`):

| piece | result |
|---|---|
| slice C hazard model (`isa/sched.rs`, scheduler-checker slice) folded `d3c3cc1` | 1387 vendor ELFs / 461 311 bundles: zero physical hazards |
| slice E exact PDI/TXN interpreter (simulator slice) folded `00a7d86` | hello + single-core design CPU-exact from the exact submitted bytes |
| core kernel `kernels/gemm_core.rs` (scheduler-checker slice) v2 `9c90adb` | 128×64 output tile per core, C resident in core memory, K streamed in 64-chunks; 4-accumulator VLIW wavefront (AIE2P has only dm0..dm4) + zero-overhead loop; **1067 issue cycles per later K64 chunk for 1024 VMACs (0.96 VMAC/issue), 1070 for chunk 0**; zero decoded physical hazards (kc=40, tiles=24); CPU-exact in the timed core simulator (kc 1/2/10/40 × tiles 1/2/3); 14 048 B program |
| array deployment `kernels/gemm_array.rs` (ArrayDesign) | vendor whole-array topology (A broadcast per core row from memtile r, B broadcast per column, C joined in each column's memtile → shim); host-prepacked 8×8 blocks; 3 args; PDI 477 600 B, TXN 38 112 B |
| whole-array simulator (simulator slice) folded `c5ca1dd` | executes the exact array PDI+TXN: 512×512×64/256, 512×2560×640, 512×1280×2560, 1024×1280×2560 CPU-exact |

Hardware run 1 (batch `array-fe6c711`, 12:29 UTC, kernel v1): **FAILED** — `npu-gemm 512 512 64 --array` and
`512 512 256`: syncobj wait ETIME after 30 s, C buffer untouched (all words still the 0xA5 poison), kernel log
empty; the 90 s window expired in the third command. The queued v2 batch was cancelled (same deployment
path). Difference from the hardware-proven single-core design: ours is the first TXN that writes core and
memtile registers (1 100+ CORE_CONTROL / DMA-ctrl / lock / PC / memtile task-queue writes) and ends in one
SYNC with ncol=8; the vendor whole-array TXN touches only shim tiles and ends in 8 per-column SYNCs
(current decoder: `npu-tools txn-dump`).

Hardware run 2 = bisect (batch `bisect-5a1818d`, 13:29 UTC, kernel v2, `tools/npu/npu-bisect.sh`, log
`logs/bisect-5a1818d.log`, dumps `logs/bisect-5a1818d/*.bin`): **all 7 variants ETIME (3 s), C untouched**, so
the TXN shape is not the (only) cause — V1 single tile, V2 single tile via memtile, V3 one column, V4 one row,
V5 full array with static CDO + vendor-shaped shim-only TXN + 8 per-column SYNCs, V6, V6b all hang. On timeout
npu-gemm now dumps `DRM_AMDXDNA_QUERY_AIE_STATUS`; `npu-tools aie-status` decodes it (layout from XRT
`info_aie2.cpp`, status v1.1: per column 4×80 B core, 136 B memtile, 48 B shim = 504 B; bit names from aie-rt).
Measured state, identical on every core of V2/V3/V5:

| field | value |
|---|---|
| core_status | `0x100000` CoreDone (program reached `done`); PC/SP/LR fields read 0 |
| sticky core events | VECTOR, LOAD, STORE, LOCK_ACQUIRE_REQ, LOCK_RELEASE_REQ, MEMORY/LOCK stall, SRS_OVERFLOW (48) |
| core locks L0..L9 | `1 0 0 0 1 0 1 0 0 0` (A-ping-empty and B-ping-empty released; L6 = 1; **C-full L9 = 0**) |
| core DMA | S2MM0 on BD1 starved and S2MM1 on BD3 starved (A and B ping chunks fully delivered); MM2S0 BD4 **LockAcqStall** on C-full |
| memtile / shim | every C input channel starved on its first BD — no C word left the cores |

So data delivery, routes for A/B, and the computation all ran on silicon; the C drain never starts because
C-full is never observed by the core DMA. The simulator and a source/TableGen audit (scheduler-checker slice: no register
clobbers, branch delay slots and ZOL constraints met, raw BM stores, no SRS) do not explain it — **root cause
not yet proven**. V1 additionally violates the AIE2P memtile rule that cardinal South↔North pass-through must keep
the lane (aie-rt `xaie_ss_aieml.c` PortVerify; V1 used South0→North4) and logged a stream-switch parity-error
event; fixed on `npu/m3-array` (South0→North0 / South1→North1).

Hardware run 3 = bisect2 (batch `bisect2-8e63fff`, 14:47 UTC, `tools/npu/npu-bisect2.sh`, log
`logs/bisect2-8e63fff.log`, dumps `logs/bisect2-8e63fff/*.bin`, status dumped after every wait). Core variants from
`gemm_core::program_variant`: `Serial` (same algorithm, one op per bundle, 7-nop gaps, no ZOL), `LockOnly` (same
lock/DMA protocol, no compute, 0xC0DE pattern in the first 64 C words), `Fast` (v2 VLIW wavefront + ZOL):

| command | result | time (fresh ctx, µs) |
|---|---|---|
| gemm_i8 64³ (control) | PASS, 0/4096 mismatches | 213.7 |
| V2 128×64×64 LockOnly | PASS, 0/64 pattern words | 248.6 |
| V2 128×64×64 Serial | **PASS, 0/8192 CPU-exact** | 223.8 |
| V2 128×64×64 Fast | ETIME (3 s), same state as run 2 | — |
| V5 512×512×64 LockOnly (8×4) | PASS, 0/2048 pattern words | 2124.3 |
| V5 512×512×64 Serial (8×4) | **PASS, 0/262144 CPU-exact — first 32-core result on silicon** | 2186.1 |
| V1 128×64×64 LockOnly (lane fix) | PASS, 0/64 | 170.8 |
| V1 128×64×64 Fast | ETIME | — |

So the lock/BD/route contract for C (and the V1 lane fix) is correct on hardware; the failure is in the Fast
core program only. Decoded difference Fast vs Serial (V2, same tile): Serial ends with locks
`1 0 0 0 1 0 0 0 1 0` (C drained, C-empty returned) and memory event DMA_MM2S_0_FINISHED_BD; Fast ends with
`1 0 0 0 1 0 1 0 0 0` and additionally raises core events MEMORY_STALL, GROUP_ERRORS_0/1 and SRS_OVERFLOW plus
DM bank-2/3 conflicts. Offline register diff vs the vendor whole-array GEMM `032eb944` (core (0,2) BD4 word5
vendor `0x26049fe5` acq 5/rel 4, ours `0x26051fe9` acq 9/rel 8, identical to the working gemm_i8; vendor core
releases with `REL_mLockId_imm #53, r9`, ours `#57, r1`, same instruction form) found no C-path configuration
difference that matters, consistent with LockOnly/Serial passing.

Hardware run 4 (batch `m3-a29f1af` — worktree advanced to `4bd73df` before the window started; binary
sha256 `fee14aec…29ec`; 15:42 UTC; fclk pinned 2000 MHz and restored to auto; log `logs/m3-a29f1af.log`).
Deployed design V6 (dynamic TXN, one hardware context, 3 submits each, **every submit CPU-exact**), core
`Serial`; time = submit→syncobj completion; useful TOPS = 2·M·N·K / time:

| shape (M×N×K) | submit 0 / 1 / 2 (µs) | best useful TOPS | mismatches |
|---|---|---|---|
| 512×512×64 | 1516.6 / 270.3 / 264.9 | 0.127 | 0/262144 ×3 |
| gate_up 512×1280×2560 | 5009.8 / 3748.5 / 3805.2 | 0.895 | 0/655360 ×3 |
| down 512×2560×640 | 2943.3 / 1701.7 / 1692.0 | 0.992 | 0/1310720 ×3 |
| gate_up 4096×1280×2560 | 30003.2 / 29066.8 / 29008.1 | **0.925** | 0/5242880 ×3 |
| down 4096×2560×640 | 13920.5 / 13162.6 / 13267.4 | **1.020** | 0/10485760 ×3 |

These are the slow, serialised core (one instruction per bundle, 7 NOPs after each, no ZOL): the 32-core
array, routes, DMA rings, lock protocol, multi-wave TXN restarts and context reuse are proven on silicon;
the per-core MAC rate is not. Fast-core bisect in the same window (V2 128×64×64):

| variant | result | meaning |
|---|---|---|
| `FastSlowCtl` (Fast compute: VLIW + ZOL; Serial-discipline control: 7-NOP lock spacing, nothing in branch delay slots) | **completes**, C delivered, 5120/8192 wrong; no SRS_OVERFLOW / error events | the hang (and the error events) come from Fast's control code; the wrong data from the compute |
| `LockOnlyZol` (LockOnly with its 64-store loop as a zero-overhead loop, `MOVXM lc,#64`) | completes, word 0 correct, words 1..63 never written | **the ZOL body runs once and never loops back on silicon** (simulator loops) |

Hardware run 5 (batch `m3-5d95c79`, worktree advanced to `2aa1e4a` before start, binary sha256
`98374afc…8f63`, 17:57 UTC, fclk pinned/restored, log `logs/m3-5d95c79.log`):

**Array clock (measured).** `ClockProbe` (LockOnly plus N × 4014 known issue cycles of NOP16 + JNZ before
C-full, V2, fresh context, 3 submits each): N = 0 / 1000 / 2000 / 4000 → best 179.7 / 2413.6 / 4635.7 /
9073.3 µs. Slope 1000→4000: (9073.3 − 2413.6) µs / (3000 × 4014) cycles = 0.553 ns → **1.81 GHz**
(0→4000: 1.80 GHz). So the AIE array runs at the 1.8 GHz H-clock, not the 1267 MHz MP-NPU clock.
**Peak** for the int8 8×8×8 VMAC (512 MAC/cycle/core) on 32 cores: 32 × 512 × 2 × 1.80 GHz = **59.0 TOPS**
(the advertised ~50 TOPS corresponds to ~1.5 GHz). The Serial-core numbers above are 1.6–1.7 % of it.

**ZOL.** `LockOnlyZol` with the llvm-aie form `add.nc lc, r1, #63` still stores only word 0, and
`FastSlowCtl` (ZOL via `add.nc lc`) still gets exactly 3072/8192 words right — the count that results when
the 128-bundle wavefront body runs once instead of 6× (periods 0..7 and 28..31 of 32). Both vendor ZOLs we
decoded (`032eb944` core (0,2) at 0x630–0x660 and 0x7a0–0x7c0) use the same `movxm ls/le` + `add.nc lc`
sequence, 16-byte-aligned LS/LE and ≥112 B setup distance as ours; the remaining difference is unknown.
`Fast` (original control code) still hangs. Decision: no hardware ZOL — the Fast wavefront now closes its
six-trip loop with `jnz r12` in the free LNG slot of body bundle 122 (its 5 delay slots are bundles
123..127, so the back edge is still free) and `add r12,-1` in the ALU slot of bundle 100 (`8cccd8b`;
simulator CPU-exact for every array test, 0 physical hazards).

Cycle model for the Fast core (ESTIMATE, `ArrayDesign::estimate`: A-channel-bound 2048 cycles per K64 chunk,
kernel 1067 issue cycles/chunk, serialized C drain 32768 cycles per wave): gate_up M=4096 2 752 512 cycles,
down M=4096 2 129 920 cycles → at the measured 1.80 GHz 17.6 / 11.3 useful TOPS (30 % / 19 % of peak).

### M4 — GPU pp8192 + NPU GEMM loop concurrency — NOT RUN (waiting on the Fast core; Serial-core M4 deliberately not spent)

Prepared M4 orchestration (current port: `npu-tools m4`; the ≤240 s hold approval was historical; MemAvailable ≥ 16 GiB check): one Flash-Next
load, 1 warm + 3 GPU-alone pp8192, NPU loop (`npu-gemm … --array --loop S`, first submit verified before
the GPU starts) + 3 concurrent pp8192, NPU loop alone, unload; SMU DRAM/NPU bandwidth sampled at 20 ms with
phase marks. Reference (earlier Halo MoE-bytes run `plain1`, Oct 1; not in this repository): pp8192 5632–5855 ms per prefill after warm-up.

**Handoff (2026-10-02 ~18:10 UTC).** Window 6 was queued on the test host through a queue gate
(a script that queued the hold only when ≤ 4 Halo holds waited, rechecking every 10 min;
output `logs/m3-b3f5c55.log`, binary sha256 `fab4d77b…ecf8`).
It runs `tools/npu/npu-m3-shapes.sh` with `NPU_M3_CORE=FastSlowCtl`: V2 FastSlowCtl and Fast (JNZ back edge),
V5 FastSlowCtl, then the V6 expert shapes (M = 512/4096, 3 submits each). If FastSlowCtl is exact there:
make its control code the only `Fast` (delete the variant), then run M4 with
`NPU_M4_CMD="./target/release/npu-gemm 4096 1280 2560 --array --loop 30 --timeout-ms 10000"`,
`NPU_M4_ALONE_CMD="… --loop 10 …"` under one ≤ 240 s hold (`NPU_WINDOW_TIMEOUT=240`). Next throughput step
(user target ~50 TOPS): the A-bound dataflow (A 8 KiB/chunk on one 4 B/cycle stream = 2048 cycles against
1067 compute cycles) is being redesigned (vertical A sharing through neighbour memory, plan in progress by
the A-split plan, a workspace-local note not in this repository), plus a second memtile→shim C channel.

## Obstacles (not kills)

1. Timing-lock contention: fold gates have priority; NPU windows waited 35 min to 2 h 20 min.
2. ssh sessions on the test host get an 8 MiB memlock limit; handled per process with `prlimit`.
3. The token-route requirement wasn't stated in anything we read; it was found on hardware.
4. Exec bit lost on scp cost two windows (fixed with chmod + preflight).
5. The whole-array deployment (and every bisect variant) does not complete on hardware: cores finish but C-full is never seen by the core DMA (M3); root cause unproven, kernel log silent.
6. `peacemaker-npu` test `vendor_decode_oracle_roundtrip` cannot run here: the oracle `/tmp/aie2p-llvm-objdump` lost
   `libLLVM.so` (environment, oracle only); all other peacemaker-npu / peacemaker-npu-sim gates pass.

Host changes: **none** (no driver, firmware, IOMMU, kernel-parameter or persistent limit change; no reboot).

## Handoff for M3 (multi-core int8 GEMM at expert shapes) and M4 (GPU+NPU concurrency)

**State.** The NPU workspace repo `main` holds railgun::npu, peacemaker-npu (encoder, decoder,
VLIW packer, CDO/PDI/TXN, DMA/route, `kernels/gemm_i8.rs` incl. single-core `design()` and the 8×4
`ArrayPlan` staging), peacemaker-npu-sim, npu-hello, npu-gemm, `tools/npu/npu-window.sh`, `tools/npu/npu-batch.sh`.
Folded since: scheduler-checker slice (`d3c3cc1`), simulator slice exact-PDI/TXN + whole-array sim (`00a7d86`, `c5ca1dd`), gemm_core v2
(`646a1fb`), gemm_array + bisect variants (`9232c8a`). Open branches: `npu/m3-kernel` e96b835 (Serial/LockOnly
core variants, gated), `npu/m3-array` 7e83786 (bisect2 READY, merged e96b835 + main 5f05360: `--core Fast|Serial|LockOnly`, `--dump-status` on success and failure incl. single-core, Rust status decoder, V1 same-lane fix, `tools/npu/npu-bisect2.sh`; not yet merged to main or run on hardware). Vendor corpus:
a local vendor-artifact cache (gitignored, not in this repository).

**Next window (bisect2, < 60 s):** gemm_i8 64³ with status dump (decoder reference); V2 with LockOnly, Serial,
Fast; V5 with LockOnly, Serial. LockOnly passing ⇒ lock/DMA contract fine, kernel at fault (Serial passing ⇒
VLIW/ZOL schedule); LockOnly failing ⇒ C-full lock/DMA encoding. Then M3 expert shapes (`npu-gemm M 1280 2560
--array --reuse-ctx`, `M 2560 640`), then M4 (current port: `npu-tools m4`; historical 240 s approval; MemAvailable ≥ 16 G).

**How to run on the test host (rules: free ≥ 40 G only for >4 G jobs; one ≤ 90 s window per exclusive hold; no host changes).**
1. Commit on the build host; on the test host fetch `main` into a bare mirror and `git worktree add --detach <worktree>/<sha> <sha>`; check `git status --porcelain` is empty.
2. Build on the build host (same glibc 2.43) and scp only `target/release/{npu-hello,npu-gemm,npu-tools}` (plus any other binary the batch runs) into the worktree's `target/release/` (record sha256 both sides) — the build lock queue on the test host is long; `npu-window.sh` refuses (exit 7) without `npu-tools`.
3. Launch detached so the lock wait survives the client:
   `(setsid nohup <timing-lock-wrapper> timing -- env NPU_WINDOW_LOG=logs/npu-window.log NPU_WINDOW_TIMEOUT=90 ./tools/npu-window.sh ./tools/npu-batch.sh > logs/batch-<sha>.log 2>&1 < /dev/null &)`
   then wait on `end rc=` in that log with a background waiter. Never `pkill -f` with a pattern that also matches your own ssh command line (it kills the session).

**Gate order for M3.** (a) `npu-gemm 64 64 64` CPU-exact on hardware (queued in batch `18eda6b`).
(b) Single-core shapes up to the kernel's limits (K 64..256, M,N multiples of 8) CPU-exact.
(c) Replace the nop-padded microkernel with a software-pipelined VLIW body (`isa/bundle.rs` packing,
scheduler-checker slice latencies; the vendor matmul ELFs show the loads of iteration i+1 overlapping VMAC of i).
(d) Multi-core: 4 core rows × 8 columns via mem-tile broadcast of B / split of A (`ArrayPlan`), shim
DMA 3-D BDs (`dma.rs`). Expert shapes: gate_up N=1280, K=2560; down N=2560, K=640; M = 128…4096.
Each needs ≤ 5 DDR-patched args (firmware translates only bo0..bo4) or arg_plus offsets into one buffer.
(e) Report useful TOPS = 2·M·N·K / submit→completion time, against an exact CPU result, per shape.

**M4 plan.** GPU side: a workspace-local MoE-bytes runner (Flash-Next pp8192, daemon
from a local build, Flash-Next MQ4 model, ~125 GB:
check `free -g` ≥ 40 G first; `pmc` mode gives FETCH_SIZE/WRITE_SIZE for DRAM bytes). Run 3× GPU-alone
and 3× with the NPU looping the best M3 kernel in a second process, inside ONE pinned window each
(the pp8192 run is longer than 90 s → request a longer exclusive hold; the pin must cover the
whole overlap). Verdict GO only if NPU useful ops added > GPU ops lost (GPU slowdown × GPU useful ops).

**Open questions.** Which clock drives the AIE array (MP-NPU 1267 MHz vs H-clock 1800 MHz)?
Does the firmware accept a partition smaller than 8 columns (num_tiles < 32) so the NPU can be shared?
Does reusing one context across submits work for designs whose cores loop forever (avoids the
~ms context-creation cost per run; prior raw-dispatch measured ~100 µs warm submit on vendor PDIs)?

## 2026-10-02 21:01 UTC — V8 on silicon (npu main 773e1e4, log logs/v8b-773e1e4-2101.log)
One npu-window hold (≤90 s, fclk pinned then restored to auto).

| Variant / control | Shape | Result | Busy TOPS | Wall TOPS |
|---|---|---|---:|---:|
| V8 FastSlowCtl, int8 epi, --loop 3 s | 4096×1280×2560 (gate_up) | PASS (first and last submit verified) | **19.28** | **15.75** |
| V8 FastSlowCtl, int8 epi, --loop 3 s | 4096×2560×640 (down) | PASS | **18.87** | 11.52 |
| V8 FastSlowCtl, int8 epi, 3 verified submits | gate_up / down | PASS, 0 mismatches | 14.1 / 10.5 | 3.6 / 0.7 |
| V8 FastSlowCtl, i32 epi, 3 verified submits | gate_up / down | PASS | 11.4 / 7.3 | 2.8 / 0.7 |
| V8 FastSlowCtl small checks (i32, int8 K=64/256) | 512² | PASS | — | — |
| V8 Fast, i32 | 512×512×64 | FAIL: syncobj timeout (ETIME), 0 C bytes written | — | — |
| V8 Fast, int8 | 512×512×64 | FAIL: completes, 126201/262144 mismatches | — | — |
| V6 array, Fast | 512×512×64 | FAIL: syncobj timeout | — | — |

Takeaways:
- V8 + FastSlowCtl is exact on silicon at expert shapes: up from Serial's 0.93–1.02 TOPS to 19.3 TOPS busy / 15.8 wall (gate_up), 33% of the 59.0 peak. The cycle model predicted 41.1 TOPS for this control (int8, Slow), so silicon reaches ~47% of the model.
- The Fast core (JNZ back-edge fix 8cccd8b) is still wrong on silicon in both V6 and V8. i32 hangs; int8 completes with ~48% wrong words. This is a separate bug from the ZOL one.
- Wall vs busy gap on down (11.5 vs 18.9): per-submit host overhead at the smaller shape.

### 21:02 UTC: shape sweep, V8 FastSlowCtl int8, 2 s verified loops (logs/sweep-773e1e4.log)
| M×N×K | busy TOPS | wall TOPS | ms per submit (busy) |
|---|---:|---:|---:|
| 4096×1280×640 | 13.2 | 8.5 | 0.507 |
| 4096×1280×1280 | 17.1 | 12.6 | 0.783 |
| 4096×1280×2560 | 19.7 | 16.0 | 1.365 |
| 4096×1280×5120 | — (no summary line; not yet diagnosed) | | |
| 4096×2560×2560 | **25.4** | **20.2** | 2.112 |
| 4096×640×2560 | 13.7 | 11.5 | 0.979 |
| 8192×1280×2560 | 21.7 | 17.2 | 2.466 |
| 2048×1280×2560 | 17.3 | 14.7 | 0.776 |

Derived (arithmetic on the rows above):
- **Padding waste.** V8 pads N to 512-wide waves: N=1280 runs as 1536 (17% waste), N=640 as 1024 (37.5%). The unpadded N=2560 reaches 25.4 TOPS. Scaling N=1280 by 1536/1280 gives 23.6, the same rate as the K slope. Lever: a wave width that divides 1280 and 640 (gate_up +~20%, down more).
- **K slope.** About 23–24 TOPS on the padded work, plus a fixed ~0.23 ms per submit at M=4096, N=1280 that scales with output size (C drain / per-wave setup). The model predicted 41 TOPS for this control, so the core/stream rate, not host overhead, binds at ~40% of peak.
- **Wall vs busy** narrows with bigger submits (20.2 / 25.4 at N=2560), so host overhead is secondary at expert shapes.

## 2026-10-02 21:17–21:27 UTC — Fast core fixed (branch-target alignment), K-slope = DDR traffic, K=5120 (npu main 1f3dd24)
Two npu-window holds (≤90 s each, fclk pinned then restored to auto). Logs: `logs/probe1-aaa165d.log`
(bw sweep, Fast-control bisect, K-slope probes) and `probe2-1f3dd24.log` (after the fix). Busy TOPS = useful ops / summed
submit→completion time; wall includes host overhead; peak 59.0 TOPS at the measured 1.80 GHz.

| variant | shape | busy TOPS | wall TOPS | % peak (busy) | binding limit | exactness |
|---|---|---:|---:|---:|---|---|
| V8 Fast (aligned targets), int8 | gate_up 4096×1280×2560 | **19.64** | 15.95 | 33% | NPU↔DDR traffic (60 MiB read/submit, 46.6 GB/s) | exact, 1192 submits, first+last verified |
| V8 Fast (aligned), int8 | down 4096×2560×640 | **18.95** | 11.45 | 32% | DDR traffic (25.3 MiB read + 10 MiB write) | exact |
| V8 Fast (aligned), int8 | 4096×1280×5120 | **21.73** | 19.20 | 37% | DDR traffic | exact (MAX_KC 40 → 160) |
| V8 FastSlowCtl, int8 | 4096×1280×5120 | 21.83 | 19.29 | 37% | DDR traffic | exact |
| V8 Fast (aligned), i32 | gate_up / down, 3 verified submits | 12.3 / 7.2 | — | — | C drain (i32) | exact |
| V6 Fast (aligned), i32 | gate_up / down, 2 s loop | 17.48 / 11.05 | 9.99 / 4.38 | 30% / 19% | A stream + i32 C | exact |
| V8 FastSlowCtl int8 (21:18) | gate_up / down | 19.88 / 18.93 | 16.03 / 11.29 | 34% / 32% | DDR traffic | exact |

**Fast-core silicon bug — root cause: unaligned branch targets.** Bisect (V8 512×512×64, i32 + int8, one knob at a time
moving Fast's control code towards FastSlowCtl): lock spacing 3→7, MOVXM gaps, ALU gaps, no delay-slot work, 7-NOP branch
gaps each still hung or corrupted (or passed the small shape only to hang at the expert shape), while **16-byte-aligned
branch targets alone made Fast exact** (small checks and gate_up/down loops). llvm-aie enforces the same rule:
`AIEMachineAlignment` pads every jump-target block to `getMachineBlockAlignmentBytes() == 16` (`aie2ps/AIE2PSInstrInfo.h:127`)
and asserts "Jump Candidate Alignment is wrong". FastSlowCtl worked by accident: its targets were unaligned but the bytes
before them in the same 16-byte line were serial-gap NOPs. Fix `1f3dd24`: `Assembler::bind` pads every label to 16 bytes,
`finish` asserts every branch target aligned (all programs: V8 pair, V6 Fast, Serial, LockOnly, ClockProbe).

**K-slope binding limit = NPU↔DDR traffic** (measured, not modelled):
- `npu-bw --sweep`: shim DDR read ceiling 55.4 GB/s (8 cols × 2 MM2S; 4×2 53.8, 8×1 53.7, one channel 11.8), write 54.2 GB/s,
  read+write together 33.6 + 33.6 GB/s. Strix Halo LPDDR5X peaks far higher, so this is the NPU's fabric port.
- Timing probe `--ctl-probe nocompute` (identical locks/DMA, no MACs): gate_up 19.68 TOPS-equivalent vs 19.64 with
  compute; down 18.87 vs 18.95. The data movement alone takes the whole submit.
- V8 moves 64 KiB from DDR per 32-core K64 chunk step (A 512×64 + B 64×512): gate_up 60 MiB read per submit (A read once
  per N-wave ×3, B once per M-wave ×8) at 46.6 GB/s read + 4.7 GB/s write.
- Compute rate on silicon (`--ctl-probe repeat=2/3`, slope over extra chunk computes): **~1480–1520 cycles per 128×64×64 chunk
  per core vs 1070 issue cycles** of the wavefront (+40% stalls, suspected data-memory bank conflicts: both pair cores read the
  same A banks, B ping/pong share one bank, VST/VLDA bm hit the same C bank). Compute-only ceiling at that rate ≈ 39 TOPS.

**Padding (priority 1) re-evaluated with the measured limit.** Only gate_up is padded (N 1280→1536); down (N=2560) is not
(the earlier "N=640 → 1024" row is the 4096×640×2560 sweep shape, not the expert down). Under the DDR limit, narrower waves
cost more bytes: a 1024×256 wave tiling (no N padding) moves A×5 + B×4 = 62.5 MiB per gate_up submit vs 60 MiB now, and
128-wide waves (the only width dividing both 1280 and 640) A×10 ≈ 103 MiB. So no narrow-wave tiling was built; the
expected 19.7 → 23.6 assumed a compute-bound rate that the nocompute probe disproves. The lever is fewer DDR bytes per op:
V9 (in progress) keeps B resident in each memtile for one N-wave and replays it across all M-waves (nw-outer order):
gate_up 60 → 33.75 MiB read, down 25.3 → 14.1 MiB (model).

**K=5120 (priority 4):** the sweep's missing line was `Geometry::new` rejecting K > 2560 (`MAX_KC = 40`, an arbitrary
limit; npu-gemm panicked before printing). Raised to `MAX_KC = 160` (K ≤ 10240; |c| ≤ 128·128·10240 < 2³¹), simulator-exact
(512×512×5120 Fast/Slow) and exact on silicon (row above).

## 2026-10-02 21:43 UTC — V9 (resident B) on silicon: exact, gate_up 22.7 / down 21.6 TOPS (npu main 8a8d88e)
One hold (log `logs/probe3-8a8d88e.log`). V9 = V8 with each N-wave's B tile resident in the column memtile
and replayed for all M-waves (nw-outer wave order; replay BDs toggle the two B slots through the BD iteration field;
compact int8 C). DDR read per submit: gate_up 60 → 33.75 MiB, down 25.3 → 14.1 MiB.

| variant | shape | busy TOPS | wall TOPS | % peak (busy) | binding limit | exactness |
|---|---|---:|---:|---:|---|---|
| **V9 Fast int8** | **gate_up 4096×1280×2560** | **22.70** | **18.00** | 38% | core compute + DMA overlap (DMA-only probe 30.5) | exact, 1345 submits, first+last verified |
| V9 FastSlowCtl int8 | gate_up | 22.26 | 17.72 | 38% | same | exact |
| V9 Fast, `--ctl-probe nocompute` | gate_up | 30.47 eq. | 23.20 | — | DDR (33.75 MiB read in 0.88 ms) | timing only |
| **V9 Fast int8** | **down 4096×2560×640** | **21.58** | **12.42** | 37% | DMA: 14.1 MiB read + 10 MiB C write (DMA-only probe 23.8) | exact, 1870 submits |
| V9 FastSlowCtl int8 | down | 20.90 | 12.13 | 35% | same | exact |
| V9 Fast, nocompute | down | 23.81 eq. | 13.21 | — | DDR read+write | timing only |
| V9 Fast int8 | 8192×1280×2560 | 26.18 | 20.45 | 44% | compute/DMA | exact |
| V9 Fast int8 | 4096×1280×3456 (V9 K limit) | 24.16 | 19.93 | 41% | compute/DMA | exact |
| V9 Fast int8 | 4096×2560×2560 | 30.10 | 23.20 | 51% | compute/DMA | exact |
| V8 Fast int8 (same window, reference) | gate_up / down | 19.76 / 18.84 | 16.12 / 11.34 | 33% / 32% | DDR | exact |
| V9 small checks | 512²×64, 1024²×128, 1536×1024×192 (3 submits, one context) | — | — | — | — | exact |

Reading: gate_up is no longer DDR-bound (DMA alone would allow 30.5) but runs at 22.7 — 1.18 ms per submit against 0.88 ms
of DMA and ~0.80 ms of compute at the measured ~1500 cycles/chunk, i.e. compute and data movement overlap poorly (the A
broadcast couples all cores of a row lane, and only two A/B slots per core). Next levers: the core stalls (bank layout,
in progress) and deeper core-side buffering. Down is close to its DMA floor (its int8 C alone is 10 MiB per submit).

## 2026-10-02 21:46 UTC — GPU→NPU zero-copy via dma-buf: SUPPORTED (exact on silicon; npu main 817a9cd) — **AMENDED 2026-10-03 02:45 UTC: only bytes written BEFORE the export alias; see "Amendment: HIP VMM dma-buf export does not live-alias"**
Question (M4 prep): can `amdxdna` import a dma-buf exported from a HIP allocation on the Halo iGPU so the NPU
reads GPU-written activations in place? Source chain (full citations in `docs/npu/gemm_i8.md`, "GPU→NPU dma-buf"):
amdxdna has a PRIME import (`amdxdna_pci_drv.c:249` → `amdxdna_gem.c:644-677`, imported object = `AMDXDNA_BO_SHMEM` with
the attachment's SG table); its device address is the host VA after an mmap of the imported GEM object (identity/SVA,
`amdxdna_gem.h:74-77`, HMM registration `amdxdna_gem.c:248-249,342-346`), and `EXEC_CMD` takes it like any argument BO.
On the exporter side amdgpu refuses to keep VRAM BOs for a non-P2P importer and migrates them to GTT (a copy;
`amdgpu_dma_buf.c:127-157,188-209` in 7.0.x), so `hipMalloc` / fine-grained / uncached device allocations would be
"supported with a copy"; `hipHostMalloc` memory is not exportable (ROCr rejects portable export of CPU-agent memory).
The in-place path is a **HIP VMM host-backed physical allocation** (`hipMemCreate` pinned, location Host, mapped
GPU-accessible) exported with `hipMemGetHandleForAddressRange` → `Device::import_dmabuf` (`railgun::npu`).

Hardware window (log `logs/probe4-817a9cd.log`): `npu-gemm --a-from-hip` allocates the packed A in that
memory, the **GPU writes it** (`hipMemcpyHtoD` + `hipDeviceSynchronize`), exports the dma-buf, the NPU imports it as arg0
(B and C are normal BOs), and every checked submit is compared with the CPU result:

| run | result | busy TOPS |
|---|---|---:|
| V8 512×512×64, fresh context | exact 0/262144 | — |
| V8 512×512×64, 3 submits one context | exact ×3 | — |
| V9 512×512×256, fresh context ×2 (re-import) | exact ×2 | — |
| V9 gate_up 4096×1280×2560, 2 s loop, A from the GPU allocation | exact (first+last), 1342 submits | 22.65 (= normal BO 22.70) |
| V9 down 4096×2560×640, 2 s loop | exact, 1842 submits | 21.25 (normal 21.58) |

Verdict: **supported, zero-copy**, provided the producer allocates with HIP VMM host memory (not `hipMalloc`). Not
measured: GPU kernel write bandwidth into that host-VMM buffer vs VRAM-carve-out memory. Disclosure: during
implementation a helper ran a HIP allocate/copy/export smoke on the build host's local GPU (not the NPU test host, no NPU access) against
its instructions.

## 2026-10-02 21:56–22:08 UTC — core bank layout (+11%), AIE2P rule gate, wall ≈ busy (npu main d8f0314)
Windows: `logs/probe5-02f7cbd.log` (layout sweep) and `probe6-d8f0314.log` (new defaults).

**Core data-memory banks.** AIE-ML data memory = 8 single-port 8 KiB banks, even/odd interleaved at 16 B into four
16 KiB double banks, round-robin arbitration among core / neighbour core / DMA (AM020 "AIE-ML Memory Module", UG1603;
model and citations in `gemm_core.rs` module doc). In the V6 map the C accumulator (0x6000..) shared a double bank with
B (0x4000..), so every B load and C load collided. Four maps were tried on silicon (V9 loops, exact):

| pair layout (current id) | map (A / B / C / Cout) | V9 gate_up busy | V9 down busy | chunk compute (repeat slope) |
|---|---|---:|---:|---:|
| 1 (old V6 map) | 0,0x2000 / 0x4000,0x5000 / 0x6000 / 0xE000 | 22.69 | 21.28 | ~1530 cycles |
| **0 (new default)** | 0,0x2000 / 0x4000,0x5000 / **0x8000 / 0x6000** | **25.24** | **22.47** | **~1357 cycles** |
| 2 | 0,0x1000 / 0x4000,0x6000 / 0x8000 / 0x2000 | 25.04 | 22.64 | — |
| 3 | 0,0x5000 / 0x4000,0x1000 / 0x8000 / 0x6000 | 24.88 | 22.64 | ~1304 cycles |

The wavefront issues 1070 cycles per chunk; ~290 stall cycles remain (both pair cores read the same A banks; DMA writes
share banks with core reads).

**AIE2P rule gate (`crates/peacemaker-npu/src/isa/rules.rs`, asked in review).** `Program::finish` now rejects, with the rule and pc:
unaligned branch targets (unless the 16-byte line before the target is whole NOP bundles — the form FastSlowCtl and
the single-core `gemm_i8` ran exact with), any `ls`/`le`/`lc` write (hardware ZOL never looped on silicon), lock requests
< 4 bundles apart, and branches/DONE/locks or landing targets inside the 5 delay slots. One violating unit test per rule;
each row cites its silicon log. FastSlowCtl's 7-NOP gaps and NOP-only delay slots were measured to be conservative, not
required. All 54 existing designs (V1–V9 × cores, gemm_i8) assemble to byte-identical PDI/TXN; `LockOnlyZol` was removed.

**Loop harness.** `--loop` no longer poisons and cache-flushes the whole C buffer before every submit (only before the
first and one final verified submit). Wall now tracks busy, and busy itself rose for down (the 10 MiB host memset competed
for DRAM):

| variant | shape | busy TOPS | wall TOPS | % peak (busy) | binding limit | exactness |
|---|---|---:|---:|---:|---|---|
| **V9 Fast int8, layout 0** | **gate_up 4096×1280×2560** | **25.36** | **25.19** | 43% | DMA/compute overlap (DMA-only 31.2, compute-only ~37 est.) | exact, first+final of 1884 |
| **V9 Fast int8, layout 0** | **down 4096×2560×640** | **23.93** | **23.46** | 41% | DDR (DMA-only 25.2) | exact, 3534 submits |
| V9 | 8192×1280×2560 | 29.14 | 28.60 | 49% | | exact |
| V9 | 4096×2560×2560 | 33.41 | 32.77 | 57% | | exact |
| V8 Fast int8, layout 0 | gate_up / down | 20.02 / 19.66 | 19.74 / 18.85 | 34% / 33% | DDR | exact |
| V8 i32, layout 0 | gate_up, 3 verified submits | 13.75 | — | | i32 C drain | exact |
| V9 nocompute probe | gate_up / down | 31.21 / 25.19 eq. | | | DDR | timing only |

## 2026-10-02 22:11 UTC — V10 (A per M-wave + all of B resident): exact, not faster for down (npu main 78b96f3)
Hold log `logs/probe7-78b96f3.log`. V10 reads A and B from DDR once per submit (down: 4.06 MiB read vs
V9's 14.1 MiB). Exact everywhere (512²×64, 1536×1024×192 three submits on one context, 1024×4096×64 NW=8, loops).

| variant | shape | busy TOPS | wall TOPS | binding limit | exactness |
|---|---|---:|---:|---|---|
| V10 Fast int8 | down 4096×2560×640 | 22.99 | 22.51 | core-side data movement / per-wave overhead (DMA-only 29.5) | exact |
| V10 FastSlowCtl / layout 3 | down | 22.35 / 22.94 | 21.43 / 22.05 | same | exact |
| V10 `--a-from-hip` (A imported from the GPU allocation) | down | 23.05 | 22.16 | same | exact |
| V10 | 8192×2560×640 | 29.40 | 27.28 | | exact |
| V10 | 2048×2560×640 / 4096×1280×640 | 15.93 / 15.21 | 15.64 / 14.94 | | exact |
| V9 (same window) | down | 23.91 | 23.03 | DDR | exact |

V10 removes the DDR bound for down (DMA-only probe 25.2 → 29.5) but the GEMM got no faster: the remaining limit is on
the array side. Fitting the DMA-only probes (V9 gate_up 960 chunk steps in 0.86 ms, V10 down 400 in 0.455 ms) to
`waves·F + chunks·c` gives c ≈ 1470 cycles per K64 chunk (the streams carry 1024 cycles of data) and F ≈ 5800 cycles per
wave (arithmetic on two measurements, not a model). V9 stays the best down design.

## 2026-10-02 22:12–22:14 UTC — M4: NPU GEMM loop concurrent with GPU pp8192 (one ≤240 s hold; npu main 3dbe227)
M4 orchestration (then Python, current port: `npu-tools m4`; one Flash-Next load, fclk pinned, MemAvailable 28 GB ≥ 16 GB), NPU = V9 Fast int8 gate_up
4096×1280×2560 `--loop 25` concurrent / `--loop 10` alone. Logs `logs/m4-3dbe227.log` and `m4-3dbe227/`.

| phase | GPU pp8192 ms (3 runs) | GPU mean | NPU busy TOPS | NPU wall TOPS | NPU exactness |
|---|---|---:|---:|---:|---|
| GPU alone | 5549.4 / 5563.4 / 5577.0 | 5563.3 | — | — | — |
| GPU + NPU loop | 6035.2 / 6103.5 / 6240.3 | 6126.3 (**+10.1%**) | 22.16 | 22.14 | first + final of 20626 exact |
| NPU alone (after) | — | — | 25.60 | 25.57 | first + final of 9532 exact |

Verdict arithmetic: over the three concurrent prefills (18.4 s) the GPU lost 0.30 prefills of work (1.69 s of GPU time) while
the NPU delivered 22.16 TOPS × 18.4 s ≈ 408 int8 TOP. The trade is net positive unless the GPU's useful prefill rate exceeds
408 / 1.69 ≈ 241 TOPS, far above anything the iGPU sustains on this workload [INFERENCE: GPU useful prefill TOPS not measured
here]. **GO** by the plan's criterion; the NPU itself loses 13% (25.6 → 22.2) to the shared fabric.

## 2026-10-02 23:14 UTC — Levers window 1: same-day baselines, E0b no-compute kc sweep, zero-code E0a (npu levers 61cb611)
One hold, 21 s, every exact step `mismatches=0` (first+final of each loop), no kernel-log lines. Log
`logs/levers-w1-61cb611.log` (copy `logs/levers/`). Plan: `docs/npu/levers-joint.md` §7.

| run | busy TOPS | wall TOPS | µs/submit | exactness |
|---|---:|---:|---:|---|
| V9 gate_up 4096×1280×2560 | 25.03 | 24.85 | 1072.6 | exact, 1859 submits |
| V9 down 4096×2560×640 | 23.87 | 23.37 | 562.4 | exact, 3520 submits |
| V10 down | 23.01 | 22.51 | 583.4 | exact, 3394 submits |
| V9 gate_up / V9 down / V10 down `nocompute` | 31.25 / 25.25 / 29.52 eq. | | 859.0 / 531.5 / 454.7 | timing only |
| V9 gate_up `repeat=2` / `repeat=3` | 13.37 / 10.96 | | 2007.8 / 2450.3 | timing only |

**E0b (V9 no-compute, N=2560, M ∈ {2048, 4096}, K ∈ {64,128,320,640}).** Per K, the two M values give the per-wave cost
(20 extra waves) and the intercept. µs per submit (busy):

| K (kc) | V9 M=4096 | V9 M=2048 | V9 per wave | V9 intercept | V10 M=4096 | V10 M=2048 | V10 per wave | V10 intercept |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 (1) | 401.1 | 307.7 | 4.67 µs | 214.2 | 400.7 | 310.6 | 4.50 µs | 220.5 |
| 128 (2) | 409.5 | 314.6 | 4.74 | 219.8 | 403.7 | 311.1 | 4.63 | 218.5 |
| 320 (5) | 465.2 | 341.3 | 6.20 | 217.3 | 424.3 | 333.9 | 4.52 | 243.5 |
| 640 (10) | 534.0 | 396.8 | 6.86 | 259.5 | 456.0 | 361.1 | 4.75 | 266.2 |

Readings (arithmetic on the rows): (1) **a fixed ≈215 µs per submit** independent of M and kc (both variants within
6 µs for kc ≤ 2). It is 20% of gate_up's 1073 µs and 38% of down's 562 µs. The railgun slice's no-op bench (test host
`railgun-small-3dcfe06.log`) splits it: no-op EXEC_CMD round trip 59 µs single / ~20 µs chained, V9 512×512×64
214 µs single / 167 µs chained → ~150 µs is execution of our ~50 KB dynamic TXN. Lean TXN is owned by the railgun slice.
(2) At kc ≤ 2 the per-wave cost (4.5–4.7 µs) equals the 256 KiB C write at 55–56 GB/s. (3) The one-formula fit
`T = T0 + W·F + W·kc·c` has residuals up to ±22k cycles (12 µs): per-wave cost is not additive in kc (V9 per-wave grows
4.67 → 6.86 µs, V10 stays 4.5–4.75 µs); it is not used as a hardware law. (4) V9 gate_up no-compute minus 215 µs is
644 µs for 33.75 MiB read = 54.9 GB/s, i.e. the measured shim read plateau (55.4); the real kernel spends 858 µs.

**E0a, zero-code form (V10 no-compute, same grid).** V10 reads A and B from DDR once, so per-wave cost is memtile→core
delivery plus the C write. Per-wave cost stays 4.5–4.75 µs from kc=1 to kc=10 — C-write bound — so at kc=10 the
two-slot ring delivers each core's 4 KiB A + 4 KiB B chunk in ≤ 4.75 µs/10 = 855 cycles: **≥ 4.8 B/AIE-cycle per core
S2MM channel** (lower bound; the link was not the binding resource). The finite release-only sink probe was not built:
the zero-code bound already exceeds the source's 4 B/cycle assumption and the 1024-cycle A/B stream model.

## 2026-10-02 23:20 UTC — Lever 2 (balanced pair A order, `--ctl-probe swap`): KILLED (npu levers c88a6fb)
Mechanism: the upper core of each pair walks its own O half with the row-0 cursor and the lower core's E half with the
row-1 cursor, C row cursors swapped to match (same C bytes). Simulator: pair core exact (all layouts, both controls and
epilogues, kc 1/2/3/40), whole-array V8/V9/V10 exact incl. gate_up/down shapes and three-submit same-context runs; bank
model peer-A collisions 1024 → 0, compute-only split/shared stall slots 512/748 → 0/236 (model). Silicon window (34 s,
all exact, log `logs/levers-w2-c88a6fb.log`, interleaved base/swap):

| run | base busy / wall | swap busy / wall |
|---|---|---|
| V9 gate_up, pass 1 | 23.58 / 23.35 | 25.50 / 25.26 |
| V9 gate_up, pass 2 | 25.73 / 25.49 | 25.55 / 25.31 |
| V9 down, pass 1 / 2 | 23.83 / 23.15, 23.71 / 23.02 | 23.74 / 23.06, 23.77 / 23.07 |
| V10 down | (W1: 23.01 / 22.51) | 22.87 / 22.22 |
| repeat 2/3/4 slope, V8 gate_up | **1336** cycles/chunk | **1369** |
| repeat 2/3/4 slope, V9 gate_up | **1323** | **1345** |

Verdict: slope ≥ 1330 with swap and no exact wall win beyond same-window variation (gate_up base itself spans
23.35–25.49 wall) → **killed**; code reverted (854e74c). Measured consequence: the ~255–290 stall cycles per chunk above
the 1067-cycle issue schedule are **not** same-cycle peer-A bank collisions, contrary to the bank model's attribution.

## 2026-10-02 23:36 UTC — Lever 3: memtile A ring depth KILLED, V10 prefix-ready B kept dark (npu levers df94050)
Simulator first (`crates/peacemaker-npu/tests/levers3.rs`, 19 tests: exact C, unchanged inputs, exact per-channel traffic, Fast and
Slow, three-seed same-context runs with odd/even `kc*waves` and non-multiple-of-D chunk counts, 88-chunk prefix boundary);
default V9/V10 bytes pinned by the new `golden_bytes.rs` (38 rows generated before the edit, green after). One hold, 42 s,
all 50 checked submits exact (first + final of every loop), log `logs/levers-w3-df94050.log`.

| run (two interleaved passes) | gate_up busy / wall | down busy / wall | no-compute µs gate_up / down | repeat slope gate_up |
|---|---|---|---|---:|
| V9 A depth 2 (default) | 25.35 / 25.01, 24.50 / 24.19 | 23.86 / 22.99, 23.83 / 22.89 | 857.8 / 531.6 | 1336 cycles/chunk |
| V9 A depth 8 | 25.73 / 25.40, 25.66 / 25.33 | 24.09 / 23.13, 24.05 / 23.12 | 866.0 / 533.5 | 1337 |
| V9 A depth 16 | 25.54 / 25.20, 25.41 / 25.08 | 24.12 / 23.17, 24.10 / 23.17 | 870.5 / 528.8 | 1334 |
| V10 down | — | 22.96 / 22.12, 22.94 / 22.12 | — / 453.1 | 1435 (2 points) |
| **V10 down `--b-prefix`** | — | **24.12 / 23.22, 24.06 / 23.13** | — / 441.4 | (anomalous point, see below) |

* **3A (A ring depth 8/16): killed.** The plan's gate needs ≥ 10% on no-compute and a real wall win: no-compute is
  unchanged or slower (W1 had already put it at the 55 GB/s read plateau), gate_up wall stays inside the baseline's own
  spread (24.19–25.49 across W2/W3), and the compute slope is unchanged. Down gains ~1% wall, below the gate. Code removed.
* **3B (V10 prefix-ready B): passes.** +4.5% wall / +5.0% busy over same-window V10 (gate ≥ 3%), no-compute −2.6%, and it
  beats the same-window best V9 down by 1.0% wall (min 23.13 > V9 max 22.99). Kept behind `--variant V10 --b-prefix`
  (`design_v10_prefix`) for integration; defaults unchanged.
* Repeat probes in 0.4 s loops produced two anomalous points (V9 depth-2 `repeat=4` 3801 µs/submit vs ~3140 expected;
  V10 prefix `repeat=3` 1413 µs vs ~1150), so the V9 depth-2 slope uses repeat 2→3; the prefix slope was re-taken below.

**3B confirmation, 23:38 UTC** (npu levers b60716d, log `logs/levers-w4-b60716d.log`, 42 s, all 27 checked
submits exact). Four interleaved passes, 1.5 s loops, down 4096×2560×640:

| down variant | median busy / wall TOPS | wall per pass | repeat 2 / 3 / 4 µs/submit (2 s loops) | cycles per accumulate chunk |
|---|---|---|---|---|
| V9 | 23.82 / 23.16 | 23.07, 23.16, 23.16, 23.16 | — | — |
| V10 | 22.94 / 22.31 | 22.34, 22.36, 22.28, 22.27 | 866.2 / 1140.3 / 1400.9 | fit 1337 (pairwise 1371, 1303) |
| **V10 `--b-prefix`** | **24.12 / 23.46** | 23.48, 23.46, 23.45, 23.43 | 831.1 / 1122.2 / 1395.9 | fit 1412 (pairwise 1456, 1368) |

Prefix: +1.3% wall over V9 (every prefix pass above every V9 pass), +5.1% over V10; fit residuals ≤ 6 µs. **The gain is
in the startup intercept** (B no longer waits for the whole segment; repeat=2 831 vs 866 µs), not in chunk compute. Lean
TXN (railgun slice) and the persistent ring attack the same per-submit intercept, so 3B's margin over V9 must be re-measured
once lean TXN lands.

## 2026-10-02 23:44 UTC — Shipped-shape baselines: Flash-Next routed experts at M = 64…512 (npu levers 805559f)
From now on every lever verdict carries the shapes we would ship (project directive, 23:44 UTC): Flash-Next routed experts
at M = 64/128/160/256/512 (160 = 8192·10/512 average rows, `fn-halo-2500.md:35`) and the dense Qwen3.8-27B IU4 GEMMs at
M = 8192 (from the dense-27B study, `docs/npu/dense27b.md` §1). The M = 4096 rows above remain for continuity.
One hold, 34 s, all 52 checked submits exact (simulator-exact first: `v9_exact_expert_rows_m160`,
`v10_exact_expert_down_m160`, `prefix_exact_expert_down_m160`). Log `logs/levers-w5-805559f.log`.
One GEMM per submit, 1 s loops, busy TOPS (µs per submit; wall within 0.5% of busy):

| M | gate_up ×1280×2560, V9 | down ×2560×640, V9 | down, V10 | down, V10 `--b-prefix` |
|---:|---|---|---|---|
| 64 | 1.16 (360.6) | 0.75 (281.4) | 0.72 (293.0) | 0.76 (276.6) |
| 128 | 2.32 (361.0) | 1.51 (277.6) | 1.42 (295.4) | 1.52 (276.7) |
| 160 (two passes) | 2.90, 2.91 (361) | 1.87, 1.75 (280.5, 299.2) | 1.76, 1.76 (298) | 1.61, 1.63 (326.6, 321.4) |
| 256 | 4.66 (359.8) | 2.96 (283.7) | 2.83 (296.8) | 2.62 (319.7) |
| 512 | 9.27 (361.9) | 5.25 (319.7) | 5.78 (290.1) | 5.97 (281.1) |

Readings: every variant is nearly flat at 280–360 µs per submit across M = 64…512, so at expert row counts the
per-submit intercept (≈215 µs TXN + submit, plus one wave's fill and drain) is ≥ 75% of the time; no core or memtile lever
moves these rows. 3B is mixed here (−7 to −9% at M = 160/256, +3 to +7% elsewhere), so its M = 4096 win does not carry to
the shipped expert shapes. Dense 27B: every shape has K ≥ 5120 (kc ≥ 80), outside V9 (`kc ≤ 54`) and V10
(`kc·(2+NW) ≤ 112`), so levers 2/3A/3B have no dense row; the dense-27B study's V8 baselines are 26.9–27.5 busy TOPS at M = 4096
slices (its log `logs/dense27b/w1-be0be55.log`).

## 2026-10-02 23:20–23:57 UTC — railgun for the NPU: prepared / chained submit, lean TXN, persistent work ring (npu railgun 3dcfe06…4cdc078)
Design note: `docs/npu/railgun-npu.md`. Code: `railgun::npu` (`Prepared`, `Chain`, `tape`), `crates/peacemaker-npu/src/kernels/gemm_array_lean.rs`
(lean TXN), `crates/peacemaker-npu/src/kernels/ring.rs` (persistent ring P1), benches `npu-railgun` / `npu-ring`. Every silicon window
below was followed by a clean V9 health window (exact, 24.9–25.7 TOPS, empty kernel log). All results exact.

**Per-submit overhead split** (no-op = TXN with one WRITE32, same hwctx and arg BOs; `railgun-small-3dcfe06.log`): a
single EXEC_CMD round trip is 59 µs p50; pipelined (prepared ×8/×32) or ERT_CMD_CHAIN ×8/×32 costs 20–25 µs per command.
A V9 512²×64 GEMM (≈1 µs of compute) costs 214 µs single and 167 µs chained ⇒ ≈150 µs per GEMM is the firmware
executing the 51 KB full TXN; the lean TXN (11–14 KB) cuts that to ≈50 µs.

**Measured table** (V9 Fast int8; a step = 8 GEMMs; overhead = step time at a ≈1–2 µs-compute shape, 512²×64 for full
TXN / 512²×128 for lean; TOPS at G = 8 per step; logs `logs/`):

| mode | per-step overhead (8 GEMMs) | gate_up 4096×1280×2560 busy / wall TOPS | down 4096×2560×640 busy / wall TOPS | exactness | log |
|---|---:|---|---|---|---|
| eager (encode + submit + wait each), full TXN | 1865 µs | 25.54 / 23.39 (1051 µs/GEMM) | 23.89 / 17.22 (562) | first+last exact | `railgun-{small,big}-3dcfe06.log` |
| prepared (retained cmd BOs, arg-word patch, queue 8, one wait), full | 1408 µs | 26.92 / 25.62 (997) | 25.90 / 22.04 (518) | exact, byte = eager, inventory 0 undeclared | same |
| ERT_CMD_CHAIN ×8, full | 1338 µs | 25.85 / 24.56 (1039); 26.21 / 24.92 in lean window | 25.41 / 21.59 (528) | exact, byte = eager | same, `railgun-lean-1e59815.log` |
| persistent ring P1, full body (8 slots / cmd) | 1290 µs | 27.06 / 27.03 (992 µs/slot) | 26.02 / 25.98 (516) | every slot of every round byte = eager | `ring-gemm-e4b3f61.log`, `ring-lean-4cdc078.log` |
| prepared + **lean TXN** | 589 µs | **30.18** / 28.71 (889) | **32.88** / 28.01 (408) | exact; `--verify-every`: 328/328, 120/120, 800/800 | `railgun-lean-1e59815.log`, `railgun-leanverify-1dcc22c.log` |
| chain ×8 + lean TXN | 573 µs | 29.19 / 27.75 (920) | 32.96 / 28.05 (407) | exact | `railgun-lean-1e59815.log` |
| **persistent ring + lean** (run 0 full, 1..7 lean) | 648 µs | **29.93 / 29.90** (897 µs/slot); paced 29.92 / 29.92 | **31.87 / 31.80** (421) | every slot of every round byte = eager (1440/1440, 1552/1552) | `ring-lean-4cdc078.log` |
| no-op floor (chain ×8, 1-write TXN) | 203 µs | — | — | — | `railgun-small-3dcfe06.log` |

Wall < busy for eager/prepared/chain because their loops do certification snapshots and C poison/verify outside the
timed region; the ring harness times publish→done, so its wall ≈ busy. Lean vs eager per GEMM: gate_up −15% (1051 →
889 µs), down −27% (562 → 408 µs).

**Persistent ring latency** (CPU playing the GPU producer; poll + done only, no GEMM; `ring-probe2-e4b3f61.log`,
`ring-gemm-e4b3f61.log`): publish seq → done line observed p50 4.5 µs, min 1.4–2.6 µs, p99 51–73 µs (the p99 is each
round's first slot, which waits for the submit); prepublished done→done 5.3 µs; submit → first done 50–55 µs (first ever
on a fresh context 1.6 ms). The slot-to-start and done-to-observe parts were not separated (no NPU timestamp); each is
bounded by the 4.5 µs round trip. With GEMMs, paced (publish only after the previous done is observed) runs at the
same µs/slot as prepublished (897 vs 897 lean, 991 vs 991 full): the CPU-free hand-off costs nothing measurable.

**Certification** (all detected / passing on silicon): byte-exact replay vs eager for prepared, chain, lean, tape and
every ring slot; patch inventory (0 undeclared words, retained BO == fresh encode, every step); negatives: stale arg
patch caught by the fresh-encode diff AND by the output check; ring skipped done word (missing done line of run 4
reported, other 7 present); stale seq patch (0/16 words patched, no new done lines); ring overrun refused by the host
guard (and deadlocks the MASKPOLL in the simulator); tape stale key (cache miss + `check_replay` mismatch, nothing
submitted); hash collision (unit test). Simulator gates: `ring` 11, `lean` 8, `array` 59, `golden_bytes`, `levers3`,
peacemaker-npu 97 + oracle, railgun::npu 18.

## 2026-10-03 00:15–00:16 UTC — railgun grouped expert launch: one TXN per E experts (npu railgun 9b3ec13 / 8058f2c)
`crates/peacemaker-npu/src/kernels/experts.rs` (`grouped_v9`): one START_CU whose TXN runs E experts of one projection back to back on
one V9 PDI (one ConfigKey: every M ≤ 512 is one 512-row wave, so the design is identical for all experts): expert 0 the
full body, experts 1..E−1 the lean body, each with its own A/B/C arena patches (distinct weights per expert). Optional
single ring poll/done around the whole group (sim-proven; silicon runs below are ring=off). Simulator: 14 `experts` gates
(per-expert exact + byte-equal eager, end state == all-full group). Silicon (`logs/experts-gate_up-9b3ec13.log`,
`experts-down-8058f2c.log`; health after both clean): every expert's C byte-equal to its eager full-TXN oracle on the
first and last round (`--verify-every` on the sanity run), all 32 runs PASS; E=64 is a 935 KB TXN and runs fine.

| projection (N×K) | E | µs / expert (M = 64 / 128 / 160 / 256 / 512) | eager µs / expert, same process | useful TOPS at M = 64 / 128 / 160 / 256 / 512 |
|---|---:|---|---|---|
| gate_up 1280×2560 | 8 | 216 / 234 / 217 / 217 / 217 | 563–597 | 1.95 / 3.58 / 4.83 / 7.74 / 15.49 |
| gate_up | 32 | 209 / 209 / 210 / 210 / 208 | 422–428 | 2.01 / 4.01 / 5.00 / 7.99 / 16.16 |
| gate_up | 64 | 201 / 203 / 201 / 202 / 201 | 396–404 | 2.08 / 4.13 / 5.22 / 8.32 / 16.69 |
| down 2560×640 | 8 | 129 / 130 / 130 / 131 / 130 | 496–516 | 1.62 / 3.22 / 4.02 / 6.39 / 12.95 |
| down | 32 | 113 / 113 / 116 / 113 / 114 | 332–346 | 1.85 / 3.71 / 4.53 / 7.42 / 14.70 |
| down | 64 | 116 / 115 / 116 / 115 / 115 | 309–317 | 1.81 / 3.64 / 4.53 / 7.28 / 14.60 |

Per expert the time is flat in M: V9 computes the full 512-row wave for every expert (at M = 160 that is 3.2× padding).
Versus the shipped-M eager baseline of the levers window (gate_up M=160 368 µs = 2.85 TOPS, down 288 µs = 1.82 TOPS,
`levers-w5-805559f.log`) grouping is 1.8× (gate_up) / 2.5× (down) per expert. Next lever: a smaller-M wave (G80's
256-row waves; contract handed to the G80 lever slice).

## 2026-10-03 00:18 UTC — lean TXN for V10 and V10 `--b-prefix` (npu railgun 087dd68)
Lean derivation extended to the prefix design (finite fill / guarded-replay groups requeued in order, BD 36 when MW > 1,
group iteration `current` cleared; sim: 27 lean gates incl. every `levers3` prefix shape and four omission negatives).
Silicon (`logs/railgun-prefix-087dd68.log`, health after clean), G = 8 prepared, full TXN first then lean with every
GEMM of every step poisoned and verified (96/96 and 1600/1600 per run):

| design | down 4096×2560×640 µs/GEMM full → lean | lean busy TOPS | 160×2560×640 µs/GEMM full → lean | lean TXN |
|---|---|---:|---|---:|
| V9 | 518 → 419 | 32.02 | 230 → 132 | 11,056 / 14,096 B |
| V10 | 538 → 433 | 31.02 | 248 → 149 | 7,984 / 15,184 B |
| **V10 `--b-prefix`** | 511 → **412** | **32.56** | 239 → 131 | 8,816 / 15,824 B |

## 2026-10-03 01:02 UTC — Lever 4 (G80, 256×1280 wave, 128×80 core tile): EXACT, faster at every gate_up shape (npu levers 60c2648)
Paper gates `docs/npu/g80-feasibility.md`; simulator `crates/peacemaker-npu/tests/g80.rs` 51/51 (gate_up 4096×1280×2560, expert M 64/160/200/513,
odd/even resubmits, full/lean interleaves, grouped E=3), golden_bytes / levers3 / lean / ring green. Declared risky, approved by
run alone: 34 s, all 36 checked submits exact (first + final of every loop), V9 health check after (25.39 / 25.04, exact),
rc=0, accel0 present. Log `logs/levers-w6-60c2648.log`. Single GEMM per submit, busy / wall TOPS (µs/submit):

| gate_up M×1280×2560 | V9 | G80 | G80 vs V9 |
|---|---|---|---|
| M=64 | 1.17 / 1.17 (358.7) | 1.45 / 1.45 (289.4) | +24% |
| M=160 | 2.91 / 2.91 (360.1) | 3.59 / 3.58 (292.3) | +23% |
| M=256 | 4.61 / 4.60 (364.0) | 5.69 / 5.69 (294.6) | +24% |
| M=512 | 9.19 / 9.17 (365.3) | 10.07 / 10.04 (333.3) | +10% |
| M=4096, pass 1 / 2 | 25.51 / 25.27, 25.26 / 25.02 | **28.68 / 28.37, 28.75 / 28.42** | +13% wall |
| M=4096 no-compute | 858.0 µs (31.3 eq.) | 516.7 µs (52.0 eq.) | DDR bytes 39.75 → 18.1 MiB |
| repeat 2/3/4 slope | 1335 cycles per 1024-VMAC chunk | 1776 per 1280-VMAC chunk (pairwise 1798, 1753) | 1.39 vs 1.30 cycles/VMAC |

G80 is now compute-bound at M=4096 (16·40 chunks × 1776 cycles = 632 µs of the 935 µs submit; the rest is the per-submit
intercept). Its per-VMAC rate is ~7% worse than V9's pair core (the +3.5% bank-model risk had the right sign). Expert rows
are still intercept-bound one-per-submit; grouped G80 vs grouped V9 (`npu-experts --design G80`) is the next, separate window.
Kept dark behind `--variant G80` / `npu-experts --design G80`; defaults unchanged (golden_bytes).

## 2026-10-03 02:08 UTC — coop-w1: grouped G80 vs grouped V9 experts with lean tails (npu coop bd3ac0e)
Declared new design (two-wave G80 group for M = 512: `design_g80(512, ..)` passed to `experts::grouped` with
`wave_m = 512`; simulator `g80.rs` 4 new two-wave gates, 55/55), approved for a solo hold, run alone: 36 s, 33/33 runs PASS, every
expert's C byte-equal to its eager full-TXN oracle on the first and last round (every round in the two `--verify-every`
sanity runs), CPU sample exact; V9 health check after 25.24 / 24.91 TOPS exact, rc=0, kernel log empty. One-wave cells ran
first, the two-wave M = 512 cells only after all passed. Log `logs/coop-w1b-bd3ac0e.log`
(`logs/coop/`). gate_up 1280×2560, µs per expert (useful TOPS), V9 → G80:

| E | M=64 | M=128 | M=160 | M=256 | M=512 (G80 two waves) |
|---:|---|---|---|---|---|
| 8 | 219.0 (1.92) → 141.7 (2.96) | 223.3 (3.76) → 140.2 (5.98) | 217.0 (4.83) → 140.8 (7.45) | 219.9 (7.63) → 140.4 (11.95) | 217.6 (15.42) → 172.3 (19.47) |
| 32 | 209.3 (2.00) → 124.7 (3.36) | 210.5 (3.98) → 122.4 (6.85) | 208.6 (5.03) → 128.7 (8.15) | 209.8 (8.00) → 125.7 (13.35) | 208.1 (16.12) → 154.8 (21.68) |
| 64 | 201.1 (2.09) → 124.0 (3.38) | 201.3 (4.17) → 124.2 (6.76) | 201.7 (5.20) → 124.1 (8.45) | 201.3 (8.33) → 124.0 (13.53) | 200.1 (16.77) → 155.1 (21.64) |

G80 is 36–40 % faster per expert at M ≤ 256 (one 256-row wave instead of V9's 512) and 22–26 % at M = 512; still flat in
M (intercept-bound) at ~124 µs per grouped expert. Kept dark behind `npu-experts --design G80`; defaults unchanged.

## 2026-10-03 02:42–02:45 UTC — coop-w2: GPU-driven persistent ring, CPU-free GPU → NPU → GPU (npu coop 0fe69f9, bab5276)
PM-native gfx1151 kernels (`crates/npu-tools/native/coop_ring_gfx1151.s`, PM source from the 0.4.1 beta snapshot; `npu_tools::coop_gpu`):
`coop_copy` (the GPU writes A into the slot arena / reads C out), `coop_publish` (seq store `glc slc dlc` + `vscnt 0`, the
WRITE_DATA-with-confirm equivalent, then a done-line poll with a realtime stamp before every load). The whole round's GPU
stream is queued before the wait; the NPU side is the unchanged `persistent_v9_lean` / `empty_persistent`; the CPU only
arms / re-arms (`patch_seq`) and verifies. Shared ring / A / C memory: anonymous host pages pinned by amdxdna as a userptr
BO (`Device::userptr_bo`, `amdxdna_gem.c:594-639`) and `hipHostRegister`ed for the GPU (see the amendment below for why
not a HIP VMM export). Declared new design, approved, run alone; both windows rc=0, kernel log empty, no CPU rescue, V9
health 25.33 / 25.02 and 25.36 / 25.00 TOPS exact. Logs `logs/coop-w2-{0fe69f9,bab5276}.log` (`logs/coop/`);
numbers below are 0fe69f9, bab5276 reproduces them within 0.2 µs / 1 %. GPU realtime 99.81 MHz
(`hipDeviceAttributeWallClockRate`):

| run (S8 / R4) | GPU publish → NPU done → GPU observe | done → GPU observe (upper bound) | one uncached poll load | run-to-run (GPU clock) | per run, wall | exactness |
|---|---|---|---|---|---|---|
| empty ring, 8 rounds (64 runs) | p50 **3.13 µs**, min 2.73 | p50 0.56 µs | 0.24 µs | 5.77 µs | — | 64/64 records, 32/32 done lines |
| GEMM 512×1280×2560, 4 rounds | 195.9 µs (incl. lean V9 GEMM) | 0.56 µs | 0.28 µs | 251.3 µs | 276 µs = 12.1 TOPS (GPU A copy + GEMM + GPU C read, serial) | GPU-read C == eager 32/32, A arena == GPU copy 16/16, CPU 16/16 |
| GEMM 512×2560×640, 4 rounds | 109.6 µs | 0.56 µs | 0.28 µs | 152.0 µs | 175 µs = 9.6 TOPS | 32/32, 16/16, 16/16 |

The GPU-driven hand-off (3.1 µs) is faster than the CPU-driven ring's 4.5 µs; "done → observe" is bounded by the last
missed poll's issue time, i.e. one uncached load (0.24 µs) to two. The first run of each round carries the NPU
command start (≤ 1.6 ms) and is the `max` of every row.

## Amendment (2026-10-03 02:45 UTC): HIP VMM dma-buf export does not live-alias
The 2026-10-02 21:46 UTC result ("GPU→NPU zero-copy via dma-buf: SUPPORTED, exact") and everything built on it tested only
bytes written **before** `hipMemGetHandleForAddressRange` exported the allocation (`HipDmabufA::create_kind` writes with
`hipMemcpyHtoD`, then exports; every later access in those runs was NPU-only and verified through the import's own CPU
mapping). Measured on Halo (no NPU command in these checks): for a HIP VMM host allocation, pinned or uncached, 64 KiB to
16 MiB, bytes written before the export are seen through the amdxdna import; after the import, GPU writes (HtoD and
GPU-kernel stores) are **not** seen through the dma-buf (import mapping or direct dma-buf mmap), writes through the dma-buf
are **not** seen by the GPU (DtoH and GPU-kernel loads), and a second export still shows the pre-export bytes. Logs
`logs/coop-w2-{2df862e,c6193a6,45aef0b}.log` (`logs/coop/`). [INFERENCE] the dma-buf pages and the GPU's VA
mapping diverge after the export/import; the driver/ROCr mechanism is not identified (`amdgpu_amdkfd_gpuvm_export_dmabuf`
exports the KFD BO itself and `amdgpu_dma_buf_map` maps its TT pages, so the split is not in those two functions).

Consequences:
* **Invalid:** "zero-copy GPU producer in place" — a GPU kernel writing into an exported HIP VMM buffer after the NPU
  imported it, or reading NPU results from it (the co-op case). `docs/npu/railgun-npu.md` §3.1 / §4 named that transport for
  the ring; corrected there to the userptr transport. `docs/npu/levers-joint.md` §2 lever 5 / §4 H inherited the claim.
* **Intact as measured (write-once-before-export, NPU-only afterwards):** probe4-817a9cd (`--a-from-hip` V8/V9 exact);
  probe7-78b96f3 V10 `--a-from-hip`; the memory-path slice's vmm-host / vmm-host-uncached DMA bandwidth sweeps, safe AXI sweeps and
  V9 no-compute / full-exact threshold runs (the NPU read/wrote the dma-buf's pages and verified through the import's
  mapping; bandwidth to those system pages is unaffected; the separately measured GPU bandwidth on the HIP VA is also
  intact). What these do **not** show is that the GPU and the NPU share one live buffer.
* **Unaffected:** the dense-27B study's own windows (w1 partials, m4 derate) use native NPU BOs, not `--a-from-hip`
  (`tools/npu/npu-dense27b-{w1,m4}.sh`); M4 uses native BOs.
* **Replacement transport (measured, live in both directions):** anonymous host pages pinned by amdxdna (userptr BO) and
  `hipHostRegister`ed for the GPU. Permanent gate `npu-coop --mode alias` (first step of `tools/npu/npu-coop-w2.sh`, no NPU
  command): the three-way check (CPU write → GPU kernel load, GPU kernel store → CPU, NPU mapping == CPU, NPU mapping →
  CPU) must PASS on userptr and must be DETECTED on the old export path. coop-w2-bab5276: userptr all ok; VMM export
  cpu→gpu, gpu→cpu and npu-map==cpu MISMATCH → detected.

## Levers table (levers-joint §7 order; busy/wall TOPS at the expert shapes, exact unless noted)
| lever | gate_up busy / wall | down busy / wall | exactness | kill verdict | log (`logs/`) |
|---|---|---|---|---|---|
| baseline V9 (W1) | 25.03 / 24.85 | 23.87 / 23.37 | exact | — | `levers-w1-61cb611.log` |
| baseline V10 down (W1) | — | 23.01 / 22.51 | exact | — | `levers-w1-61cb611.log` |
| E0b fit | — | — | timing only | fixed ≈215 µs/submit (→ railgun-slice lean TXN) | `levers-w1-61cb611.log` |
| E0a (zero-code, V10 no-compute) | — | — | timing only | memtile→core ≥ 4.8 B/cycle/channel, not binding | `levers-w1-61cb611.log` |
| 2: Upper A/C half-order swap | 25.50 / 25.26, 25.55 / 25.31 | 23.74 / 23.06, 23.77 / 23.07 | exact | **killed**: slope 1345 (V9) / 1369 (V8) ≥ 1330, no wall win | `levers-w2-c88a6fb.log` |
| 3A: V9 memtile A ring depth 8 / 16 | 25.73 / 25.40, 25.66 / 25.33 (d8) | 24.12 / 23.17, 24.10 / 23.17 (d16) | exact | **killed**: no-compute 858 → 866 / 871 µs, wall within baseline spread | `levers-w3-df94050.log` |
| 3B: V10 down prefix-ready B | — | **24.12 / 23.46** (median of 4; V9 23.82 / 23.16) | exact | **pass** (+1.3% wall vs V9, +5.1% vs V10; gain is startup intercept); merged behind `--b-prefix` | `levers-w3-df94050.log`, `levers-w4-b60716d.log` |
| shipped experts, M = 160 (W5) | V9 2.90 / 2.90 | V9 1.87 / 1.87, 1.75 / 1.75; prefix 1.61 / 1.60, 1.63 / 1.63 | exact | 3B **no win** at expert M (intercept-bound, ~280–360 µs/submit) | `levers-w5-805559f.log` |
| 3B under lean TXN (railgun slice, `npu-railgun --lean`, G=8 prepared, npu/railgun 087dd68) | — | prefix 412 µs = 32.56 TOPS; V9 419 µs = 32.0; V10 433 µs = 31.0 | exact | margin holds under lean: +1.7% vs V9, +5.1% vs V10 | `railgun-prefix-087dd68.log` (measured by the railgun slice) |
| 4: G80 gate_up 256×1280 wave | **28.68 / 28.37, 28.75 / 28.42** (V9 25.51 / 25.27, 25.26 / 25.02); M=160 3.59 / 3.58 (V9 2.91) | — (gate_up only) | exact | **pass**: +13% wall at M=4096, +10–24% at expert M=64–512; slope 1776 cycles / 1280 VMAC; dark behind `--variant G80` | `levers-w6-60c2648.log` |

## Final table (best exact variant per expert shape, measured)
| shape | variant | busy TOPS | wall TOPS | % of 59.0 peak | binding limit |
|---|---|---:|---:|---:|---|
| gate_up 4096×1280×2560 | G80 Fast int8 (opt-in `--variant G80`; V9 25.51 / 25.27 same window) | 28.75 | 28.42 | 49% | core compute (1776 cycles / 1280-VMAC chunk) + per-submit intercept; DMA-only 52.0 eq. |
| down 4096×2560×640 | V10 Fast int8 `--b-prefix` (opt-in; V9 23.82 / 23.16 same window) | 24.12 | 23.46 | 41% | per-submit intercept (~215 µs TXN + submit) + C write; DMA-only 30.4 eq. (441 µs) |
| gate_up, railgun | V9 + lean TXN, prepared ×8 / persistent ring | 30.18 / 29.93 | 28.71 / 29.90 | 51% | lean TXN ≈50 µs/GEMM left; DMA/compute overlap |
| down, railgun | V9 + lean TXN, prepared ×8 / persistent ring; V10 `--b-prefix` + lean prepared ×8 | 32.88 / 31.87; 32.56 | 28.01 / 31.80 | 56% | lean TXN; C write |
| start of session (V8 FastSlowCtl) | gate_up / down | 19.7 / 18.9 | 16.0 / 11.5 | 33% / 32% | DDR traffic |

Open: ≥50 TOPS not reached. Remaining limits, measured: fixed ≈215 µs per submit (TXN execution ~150 µs + submit round trip;
railgun-slice lean TXN / persistent ring); core chunk compute ~1323–1337 cycles vs 1067 issue, with the stall **not** caused by
same-cycle peer-A bank collisions (lever 2) nor by memtile A depth (lever 3A); gate_up no-compute at the ~55 GB/s DDR
read plateau; down's 10 MiB int8 C write per submit. Lever 4 (G80 wide gate_up geometry) in simulator feasibility.

Open: ≥50 TOPS not reached. Remaining limits, measured: core chunk compute ~1357 cycles vs 1070 issue (bank conflicts,
both pair cores read the same A banks); array-side data movement ~1470 cycles per chunk + ~5800 per wave (2-slot core
buffers, coupled broadcasts); DDR port ~55 GB/s. Next steps: 3-slot core A/B rings with counting locks (core memory has
2×4 KiB free at 0x1000/0x3000), a deep memtile A ring for V9 (worker branch `npu/aring` unfinished, not merged), and
the GPU writing activations straight into the dma-buf-imported A (supported, exact).

## 2026-10-02 — shim AXI and memory-path experiment: source/simulator gate

The `npu/axi-wip` cutover retains the software default AXI tuple
`(burst, AxCACHE, AxQoS) = (3, 2, 0)`. AIE2P (not AIE2PS) has burst in BD
word 4 bits 31:30, cache in word 5 bits 27:24, and QoS in bits 23:20.
Burst encodings 0/1/2/3 mean 64/128/256/512 bytes:
aie-rt `driver/src/global/xaie2pgbl_params.h:16347-16394`,
`driver/src/dma/xaie_dma_aie2p.c:40-51`, `xaie_dma.c:765-782`;
[AIE2p target model](https://github.com/Xilinx/mlir-aie/blob/main/lib/Dialect/AIE/IR/AIETargetModel.cpp#L1744-L1758).
`ArrayDesign::set_shim_axi` changes every shim BD in the submitted TXN
in-place; a default call performs no parsing or allocation. The simulator
latches/observes these exact fields but does not model AXI performance or
host cache coherence.

Offline smoke: emitted default and `(1,3,15)` V9 512×512×64 images each ran
twice through the submitted-PDI/TXN simulator, with freshly poisoned C:
262144 CPU-exact outputs, 252364 nonzero reference values, unchanged inputs,
131072 read bytes and 524288 write bytes cumulatively per context.
Default-byte comparison against `dc261e13c5faa4e0ea06e1ff888c9ea7f8f0f72e`:
896 valid designs across V1–V10/core/shape/control/epilogue combinations,
single-core GEMM and bandwidth probes produced **zero PDI/TXN differences**;
171 invalid cases were rejected identically. Each side had 1792 files
(253066016 PDI bytes, 22863968 TXN bytes), sorted SHA256-manifest digest
`b179a7be53c6f1f3ba37b5bb46171d3ff89b8b149b1213177400d081d0543356`.
478 setter transition/atomicity/idempotence cases and 768 nondefault
bandwidth builds changed only attribute bits; a burst-2 negative control
changed 256 TXNs. Throwaway proof harnesses/artifacts were removed.

The initial source-only candidates were default `(3,2,0)`, burst 0/1/2
with cache 2, cache 0/3 with burst 3, and QoS 15 with burst 3/cache 2.
**This safety inference was wrong:** register encodability does not imply
Halo NoC support. The corrected automatic sweep keeps vendor word 5
(cache 2, QoS 0) and varies only burst. Nonvendor word-5 settings require
an explicit hazardous-probe opt-in and advance declaration; no automatic
sweep may include them. Allocating cache modes `0xb/0xf` are not called
safe merely because their bitfields encode.
`npu-bw --memory native|vmm-host|vmm-host-uncached` selects both argument
buffers; V9 takes the same `--memory` plus `--burst/--axcache/--axqos`.
Read-only DMA and `nocompute` explicitly report timing-only, not GEMM
exactness; writes retain poisoned first/final full-pattern comparisons.
The GPU reference checks read XORs (collision-capable checksum, not
full-byte exactness) and full write patterns outside the timed kernel.
No new silicon bandwidth result is claimed by this source-gate entry.

Integrated release gates: 210 tests passed across the workspace packages,
and workspace release build passed. The first oracle invocation failed
only because `/tmp/aie2p-llvm-objdump` could not locate `libLLVM.so`;
the failed oracle passed with `LD_LIBRARY_PATH=/tmp`, then the previously
unrun simulator/driver/doc gates passed. No host configuration changed.

## 2026-10-02 22:47–23:02 UTC — initial silicon evidence, probe correction and QoS incident

Source `3e36d9a`, remote detached Git worktree verified clean before each
window. Every hold used the exclusive timing lock, timeout 90 s, incoming
performance `auto`, fabric pinned/read back at 2000 MHz, MemAvailable
at least 26 GiB, then restored `auto`.

* Native full bandwidth sweep read ceiling: **56.032 GB/s** (8×2,
  16 MiB/channel; 64 MiB/channel 55.605). All nine write configurations
  had first submit exact but final submit **1024 bad words**, first at
  byte 0: `0x00cd0cd0` vs `0xb0000000`. The log does not establish a
  contiguous damaged region. **All write bandwidth from this window is
  invalid as an exact result**, including 54.400 GB/s and mixed
  33.741+33.741. Log:
  `logs/mempath2-3e36d9a-sweep-native.log`.
* Native V9 DMA-only gate_up/down default: 30.625/25.090 TOPS-equivalent
  (read/write 40.375/7.178 and 27.564/19.601 GB/s). Burst 0/1/2 and cache
  0/3 completed; their paired V9 512²×128 three-submit checks were all
  CPU-exact. Timing-only submits are **not** GEMM exactness, despite the
  old summary's misleading `verify: ... exact` label (now corrected).
  Log: `logs/mempath2-3e36d9a-v9-native.log`.
* **AxQoS 15 wedged the NPU fabric.** First gate_up submit timed out,
  zero C bytes changed, and one firmware status query timed out. A later,
  distinct default-QoS host-VMM scenario failed device-open `GET_INFO`
  with errno 22; kernel `aie2_smu_exec: smu cmd 4 failed, 0xff`,
  `aie2_smu_init: Access power failed, ret -22`, and resume failed -22.
  Logs above plus `...-v9-vmm-host-default.log`; preserved dump:
  `logs/mempath2-3e36d9a-qos15-status.bin`.
  NPU use was halted and the test host was rebooted externally at 22:54:48 UTC.
  The first post-recovery health window was exact V8 FastSlowCtl
  512² and gate_up, 19.88 busy/16.17 wall TOPS. This agent performed no
  reboot, reload, or kernel/host policy change.
* The corrected emitter rejects nonzero QoS/nonvendor cache by default;
  a distinct explicit per-probe path is required for hazardous attributes.
  Such a run must be declared, alone/last in its own window, and followed
  by stop/report if it wedges. Register-width tests are encoding evidence,
  not platform-safety evidence.
* The write pattern arena is intentionally relocated to memtile
  `0x14000/0x18000`, established resident-B memory in V9, rather than low
  RAM. This changes the bandwidth probe's PDI/addresses, not default GEMM
  bytes or AXI encoding. A modeled low-word-0 mutation fails before and
  passes after; mutating the actual source reproduces exactly 1024 bad
  words at a 16384-byte stride for 16 MiB. Firmware contains the exact
  `0x00cd0cd0` constant once at offset `0x4d430` of decompressed
  `/lib/firmware/amdnpu/17f0_11/npu_7.sbin.zst` (429680 bytes).
  **[INFERENCE]** Firmware maintenance of low RAM is consistent with
  the signature; the actual writer is not source-proven.

GPU reference (1 GiB, 20 launches, best rate; read XOR checksum and full
write pattern checked outside timing):

| GPU backing | Read GB/s | Write GB/s | Verification | log under `logs/` |
|---|---:|---:|---|---|
| VRAM carve-out | 240.77 | 222.23 | read checksum / write exact | `mempath2-3e36d9a-gpu-vram-isolated.log` |
| HIP VMM Host pinned | 241.22 | 222.49 | read checksum / write exact | `mempath2-3e36d9a-gpu-host-isolated.log` |
| HIP VMM Host uncached | 240.83 | 222.38 | read checksum / write exact | same |

Initial GPU launch failed because the gfx1151-only fatbin was registered
against every visible GPU, including unrelated gfx1100. The diagnostic
explicitly requested gfx1100 (`...-gpu-diagnostic.log`). Isolating Halo
with process-start `HIP_VISIBLE_DEVICES=1` fixed it; no code-object ABI
change was needed. The selected device was gfx1151.

Repair integration gate: **213 release tests passed**, workspace release
build and runner/build-script syntax checks passed. Fresh post-guard
comparison against `3e36d9a` emitted 50 V6/V8/V9/V10 designs spanning
small/multiwave/expert shapes, Fast/Serial/FastSlowCtl/LockOnly/ClockProbe,
int8/i32 epilogues: all 100 PDI/TXN files (23371968 bytes/side) were
byte-identical, SHA256-manifest
`867f81b4b802161d8b54a2f3a5406887f09dcd2dfe29f896cf014a419ce3b983`.
Default strict and explicitly hazardous bandwidth builders emitted
identical bytes on the same corrected arena/geometry in offline smoke.
The old/new bandwidth bytes intentionally differ at the source arena,
not because default AXI encodings changed. Byte-proof scaffolds removed.

## 2026-10-02 23:03–23:42 UTC — corrected source, complete safe backing/AXI matrix

All measurements below use clean Git snapshot `5721bd9`, the corrected high
source arena, cache=2/QoS=0, exclusive ≤90 s windows and MemAvailable ≥16 GiB.
Each bandwidth case has six submits. All 27 write cases in the geometry
matrix and all 12 in the AXI matrix checked fresh poison on first and final
submit and were exact. Pure reads and V9 no-compute are timing-only.

### DMA-only geometry matrix

Decimal GB/s; each cell is read/write. Bytes are per active channel.
Channels are per column, as in the original bandwidth probe.

| Cols | R/W channels/col | MiB/channel | Native R/W | Host pinned R/W | Host uncached R/W |
|---:|---:|---:|---:|---:|---:|
| 1 | 1/0 | 16 | 12.772/0.000 | 13.025/0.000 | 13.048/0.000 |
| 1 | 2/0 | 16 | 23.832/0.000 | 25.075/0.000 | 24.934/0.000 |
| 2 | 1/0 | 16 | 24.456/0.000 | 25.310/0.000 | 24.961/0.000 |
| 2 | 2/0 | 16 | 45.461/0.000 | 46.320/0.000 | 46.196/0.000 |
| 4 | 1/0 | 16 | 44.791/0.000 | 45.495/0.000 | 45.987/0.000 |
| 4 | 2/0 | 16 | 55.266/0.000 | 54.380/0.000 | 54.719/0.000 |
| 8 | 1/0 | 16 | 54.754/0.000 | 54.226/0.000 | 54.918/0.000 |
| 8 | 2/0 | 16 | 55.466/0.000 | 55.719/0.000 | 56.235/0.000 |
| 1 | 0/1 | 16 | 0.000/13.463 | 0.000/13.607 | 0.000/13.494 |
| 1 | 0/2 | 16 | 0.000/27.194 | 0.000/27.364 | 0.000/27.375 |
| 2 | 0/1 | 16 | 0.000/27.328 | 0.000/26.664 | 0.000/27.360 |
| 2 | 0/2 | 16 | 0.000/46.976 | 0.000/46.860 | 0.000/47.282 |
| 4 | 0/1 | 16 | 0.000/51.566 | 0.000/52.277 | 0.000/52.590 |
| 4 | 0/2 | 16 | 0.000/52.646 | 0.000/54.208 | 0.000/54.414 |
| 8 | 0/1 | 16 | 0.000/54.024 | 0.000/46.162 | 0.000/54.819 |
| 8 | 0/2 | 16 | 0.000/55.280 | 0.000/51.697 | 0.000/55.860 |
| 8 | 2/2 | 16 | 33.667/33.667 | 32.736/32.736 | 34.388/34.388 |
| 8 | 2/0 | 1 | 43.612/0.000 | 42.445/0.000 | 44.539/0.000 |
| 8 | 2/0 | 64 | 55.579/0.000 | 55.567/0.000 | 55.467/0.000 |

Logs: `logs/mempath2-5721bd9-sweep-{native,vmm-host,vmm-host-uncached}.log`
(19 cases each). Peak read: **55.579 / 55.719 / 56.235 GB/s**; peak write:
**55.280 / 54.208 / 55.860 GB/s**. Backing did not raise the plateau.

### Safe shim burst sweep

| Burst field / bytes | Native read/write GB/s | Host pinned read/write | Host uncached read/write |
|---|---:|---:|---:|
| 3 / 512 | 54.349/54.795 | 53.479/54.892 | 55.778/53.994 |
| 0 / 64 | 42.836/26.476 | 43.376/26.420 | 46.116/26.458 |
| 1 / 128 | 52.155/48.420 | 54.219/43.915 | 53.395/46.653 |
| 2 / 256 | 53.641/49.798 | 52.203/47.669 | 54.245/49.870 |

Eight columns, two channels/column, 16 MiB/channel. Logs:
`mempath2-5721bd9-axi-{native,vmm-host,vmm-host-uncached}.log`.
Smaller bursts generally reduce throughput; automatic sweeps never change word5.

### V9 no-compute traffic and exact consumer checks

Each DMA probe runs one second. Cells are busy/wall **equivalent** TOPS, not
computed GEMM throughput. Each tuple also ran V9 512×512×128 for three submits,
all CPU-exact (36 submits total).

| Burst | Shape | Native busy/wall | Host pinned busy/wall | Host uncached busy/wall |
|---:|---|---:|---:|---:|
| 3 | gate_up | 31.2268/31.1156 | 30.3045/30.1668 | 31.4391/31.2716 |
| 3 | down | 25.4033/25.2611 | 24.8567/24.6487 | 25.1147/24.9134 |
| 0 | gate_up | 22.0499/21.9183 | 24.2808/24.1378 | 18.1708/18.0568 |
| 0 | down | 20.4486/20.2805 | 20.4927/20.3368 | 20.6272/20.4604 |
| 1 | gate_up | 29.6703/29.5036 | 29.7043/29.5521 | 30.5086/30.3572 |
| 1 | down | 23.4259/23.2122 | 23.1117/22.9235 | 23.0644/22.8785 |
| 2 | gate_up | 31.3418/31.1947 | 31.3647/31.2082 | 31.0987/30.9379 |
| 2 | down | 23.2392/23.0643 | 23.4279/23.2407 | 23.3041/23.1253 |

Logs: `mempath2-5721bd9-v9-{native,vmm-host,vmm-host-uncached}.log` (12 runs
each). Relative to default native burst3, no imported configuration is >10%
faster. Same-slow-burst0 Host gate_up is +10.12% versus native burst0, but
still slower than default. Full exact expert-shape runs below cover that
literal threshold; their same-config gain is below 10%.

### Rust-only tooling source cutover

`npu-tools` contains `aie-status`, `txn-dump`, `soc-metrics`, `m4`, `tblgen2rs`,
and `gpu-bw`. Five Python tools are removed; callers use Rust. Actual compiled
decoder stdout was byte-identical on 20 existing dump invocations (seven AIE
dumps, default/255 bitmaps, three TXNs, full/summary). Additional worker parity
covered five AIE bitmap boundaries and 32 metrics frames. The metadata
generator emitted 881 instructions, 77 formats and 109 operand kinds,
byte-identical except deliberately updated Rust-generator attribution.

M4 was exercised by running the actual Rust program with scripted external
daemon/producer/sampler processes, including error cleanup; the full model
was not rerun after the port. Its historical ≤240 s exception does not permit
new >90 s windows. The inherited cache-drop side effect was preserved but not
invoked on the test host under the no-host-change rule.

GPU kernels are PM-native gfx1151 assembly and ELF, loaded by Rust through
shared lazy HIP module/VMM ABI; no hipcc, LLVM assembler or vendor NPU runtime.
Regeneration uses only the permitted PM source
(hipfire beta snapshot `5b4664e3b`), locked Cargo
output outside that tree, then `PEACEMAKER=... crates/npu-tools/native/build-gpu-bw.sh`.
Emission repeated byte-identically; M7 lifted both kernels exactly with zero
obligations. All three backing modes passed target read/write launches; old
HIP source and hipcc build script are removed.

A delegated helper performed an unauthorized local Navi48 allocation/event/
export smoke (no kernel launch), despite its offline-only assignment. This
was reported and is not target acceptance evidence. Subsequent GPU/NPU
hardware work is restricted to locked test-host windows.

### Additional concurrency probe — simulator-exact before silicon

`npu-bw --sweep-concurrency` enumerates all **405** combinations: active
channels/direction N={1,2,4,8,16}, queued tasks Q={1,2,4}, BD sizes 4 KiB..1 MiB
powers of two, read/write/mixed. Channels spread column-first over eight columns.
Q tasks have independent BDs and DDR ranges; only final task returns completion.
Mixed N16/Q4 uses all sixteen shim BDs/column.

Primary limits: aie-rt `driver/src/global/xaie2pgbl_reginit.c` AIE2P
`StartQSizeMax=4`, `NumBds=16`. Legacy bandwidth and all default GEMM designs
are unchanged. New simulator coverage: all 45 N/Q/direction geometries at
4 KiB, poisoned resubmits, exact channel/DRAM counts, global-source wrapping
through 1 MiB, maximum-BD mixed case and invalid/unsafe inputs. All 16 scoped
simulator tests and nine builder tests passed; standalone N16/Q4/4 KiB and
N1/Q4/1 MiB each completed two exact submits with expected cumulative traffic.
Silicon is split into three explicit 135-case direction-filtered windows;
their union is the full grid.

Integrated source gate: **222 release tests passed in 23 suites**, workspace
release build passed, runner/native-build script syntax passed. Actual Rust
dispatcher help ran successfully. Default byte-identity evidence above remains
applicable: additions do not alter any default GEMM emitter or legacy BW design.

### Source verdict: memory backing and no-migration VRAM import

Citations below are to the Linux 7.0 kernel source (`drivers/accel/amdxdna/`, `drivers/dma-buf/`, `drivers/gpu/drm/amd/amdgpu/`); the local reference checkout they were read from is not part of this repository.

**No, this unmodified importer cannot retain the GPU VRAM carve-out backing.**
`amdxdna_gem.c:644–677`
uses plain `dma_buf_attach`, imports a SHMEM SG table, and marks the object
SHMEM. Plain attach supplies no importer operations, so peer2peer remains
false (`dma-buf.c:1033–1040,1069–1072`).
The exporter removes VRAM eligibility for non-peer attachments, chooses GTT
unless peer2peer is allowed, and maps TT pages; the VRAM mapping branch is
separate (`amdgpu_dma_buf.c:137–143,192–232`).
Thus successful `hipMalloc` import is not proof of a no-copy VRAM path.

Native SHMEM and imported Host VMM still use the same NPU address mechanism:
PASID mode chooses `mem.userptr`, not a special amdgpu fabric route
(`amdxdna_gem.h:74–77`).
Default `force_iova` is false
(`amdxdna_iommu.c:13–15,137–139`).
Uncached Host is a real HIP allocation mode, not a relabeled native BO;
both its GPU access and NPU import were exercised.

No numeric ~55 GB/s software clamp was found in the reviewed clock/QoS path:
`aie2_ctx.c:461–466`
copies `dma_bandwidth`, but the solver uses GOPs/fps/latency and chooses maximum
DPM when those QoS inputs are absent
(`aie2_solver.c:50–73,100–133`).
NPU5 uses the NPU4 table and setter
(`npu5_regs.c:65–68,95–96`);
that table tops out at HCLK 1800 MHz
(`npu4_regs.c:79–88`).
This is a bounded source review, not proof that no firmware/fabric limit exists.

**Proposal only; no driver/module/host changes applied:** a genuine no-migration
VRAM experiment would need dynamic attach with peer eligibility and move
notification, verified topology and reservation/pinning lifetime, plus an
import/addressing path that accepts VRAM bus SG addresses rather than assuming
SHMEM pages and a faultable PASID userptr. Merely setting `allow_peer2peer` is
not sufficient or safe. Coherency and exact read/write behavior must be proved
before timing. There is no measured claim that this would improve NPU bandwidth.

## 2026-10-02 23:55–2026-10-03 00:03 UTC — PM-native GPU target acceptance

Clean Git snapshot `cfeec35`; gfx1151, 20 reported CUs.
Each operation streams 1 GiB for 20 timed launches. Read verifies per-thread
four-lane XOR checksums (collision-capable); write verifies all bytes before
timing and after final fresh poison. Every check passed; all three ≤90 s
windows ended rc=0 with no kernel log errors and fabric policy restored.

| Backing | Best read GB/s | Median read | Best write GB/s | Median write |
|---|---:|---:|---:|---:|
| VRAM | 239.43 | 237.65 | 222.01 | 219.22 |
| VMM Host pinned | 238.68 | 237.47 | 221.84 | 219.23 |
| VMM Host uncached | 238.79 | 237.94 | 226.20 | 220.79 |

Logs: `logs/mempath2-cfeec35-gpu-{vram,vmm-host,vmm-host-uncached}.log`.
This reproduces the earlier HIP reference bandwidth with PM-native code
objects and a Rust launcher. GPU bandwidth near 240 GB/s while all accessible
NPU backing modes plateau near 55 GB/s rules out a universal LPDDR bandwidth
ceiling, not every possible NPU-side software or fabric restriction.

## 2026-10-03 00:05–00:10 UTC — all 405 concurrency cases measured

Snapshot `cfeec35`, native SHMEM, burst3/cache2/QoS0, four submits/case.
Three ≤90 s direction-filtered windows cover the complete requested grid.
All 270 write-bearing cases checked first and final fresh poison exactly.
Every window ended rc=0, zero failed cases, no kernel log errors, fabric
restored. Read remains timing-only.

**Ceiling-cause verdict:** no pure-read case exceeded 60 GB/s. Peak read
was **51.407 GB/s** at N16/Q4/1 MiB BD (64 MiB total); legacy long-BD
ceiling remained 55–56 GB/s. Within this complete N/Q/BD grid, the old
plateau is not explained by our channel/task concurrency cap. Classify as
an accessible **NPU port/path plateau**, not proof of physical port width
or absence of every software/firmware restriction.

Tables are best warm decimal GB/s. Columns are BD KiB; N is active
channels/direction, Q tasks/channel. Mixed cells are **per direction**:
read and write bytes/rates are equal by construction; total is twice the cell.
Raw logs: `logs/mempath2-cfeec35-concurrency-native-{read,write,mixed}.log`.

### Read concurrency matrix

| N | Q | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0.059 | 0.130 | 0.259 | 0.513 | 0.986 | 1.860 | 3.243 | 4.802 | 7.270 |
| 1 | 2 | 0.131 | 0.258 | 0.518 | 1.011 | 1.838 | 3.188 | 4.839 | 7.292 | 9.483 |
| 1 | 4 | 0.263 | 0.494 | 0.988 | 1.968 | 3.253 | 4.933 | 7.265 | 8.777 | 11.187 |
| 2 | 1 | 0.128 | 0.255 | 0.526 | 1.022 | 1.961 | 3.859 | 6.862 | 10.457 | 14.179 |
| 2 | 2 | 0.256 | 0.515 | 0.999 | 1.981 | 3.649 | 6.338 | 9.752 | 13.996 | 18.296 |
| 2 | 4 | 0.506 | 0.986 | 1.930 | 3.729 | 5.476 | 9.382 | 14.203 | 18.874 | 21.650 |
| 4 | 1 | 0.249 | 0.495 | 1.029 | 2.017 | 3.813 | 7.078 | 10.903 | 19.470 | 26.413 |
| 4 | 2 | 0.509 | 1.075 | 1.983 | 3.787 | 7.078 | 10.952 | 19.146 | 26.120 | 34.051 |
| 4 | 4 | 0.937 | 1.894 | 3.622 | 6.800 | 11.195 | 17.572 | 25.621 | 33.202 | 39.275 |
| 8 | 1 | 0.508 | 0.972 | 1.846 | 3.542 | 6.623 | 11.029 | 20.044 | 28.818 | 37.011 |
| 8 | 2 | 0.923 | 1.829 | 3.423 | 6.623 | 10.771 | 18.219 | 27.241 | 36.720 | 45.421 |
| 8 | 4 | 1.748 | 3.289 | 5.356 | 10.297 | 17.836 | 27.863 | 35.886 | 44.056 | 48.916 |
| 16 | 1 | 0.901 | 1.748 | 3.317 | 5.492 | 11.303 | 17.360 | 26.495 | 36.163 | 44.441 |
| 16 | 2 | 1.620 | 3.156 | 5.371 | 10.272 | 17.134 | 26.061 | 35.858 | 43.859 | 50.216 |
| 16 | 4 | 2.549 | 5.083 | 10.008 | 15.996 | 24.918 | 35.886 | 43.291 | 49.083 | 51.407 |

### Write concurrency matrix

| N | Q | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0.044 | 0.115 | 0.240 | 0.483 | 0.971 | 1.841 | 3.238 | 4.990 | 7.194 |
| 1 | 2 | 0.130 | 0.259 | 0.512 | 1.005 | 1.881 | 3.260 | 4.986 | 7.223 | 9.766 |
| 1 | 4 | 0.262 | 0.506 | 0.977 | 1.915 | 3.254 | 4.968 | 7.392 | 9.841 | 11.578 |
| 2 | 1 | 0.126 | 0.261 | 0.563 | 1.010 | 1.987 | 3.633 | 6.420 | 9.570 | 14.875 |
| 2 | 2 | 0.256 | 0.527 | 1.025 | 1.986 | 3.696 | 6.329 | 9.873 | 14.442 | 19.168 |
| 2 | 4 | 0.506 | 0.997 | 1.952 | 3.555 | 6.356 | 9.993 | 14.958 | 19.536 | 23.263 |
| 4 | 1 | 0.247 | 0.503 | 1.036 | 1.961 | 3.848 | 7.046 | 12.760 | 18.807 | 27.835 |
| 4 | 2 | 0.489 | 0.970 | 1.925 | 4.119 | 6.920 | 10.951 | 19.572 | 27.579 | 36.968 |
| 4 | 4 | 0.954 | 1.815 | 3.559 | 6.524 | 10.959 | 18.620 | 27.741 | 36.662 | 41.642 |
| 8 | 1 | 0.461 | 0.939 | 1.822 | 3.627 | 6.540 | 9.803 | 18.199 | 27.628 | 36.771 |
| 8 | 2 | 0.933 | 1.873 | 3.665 | 6.458 | 11.417 | 17.870 | 27.893 | 37.061 | 43.222 |
| 8 | 4 | 1.758 | 3.392 | 5.388 | 10.811 | 19.434 | 28.059 | 35.809 | 43.146 | 47.127 |
| 16 | 1 | 0.865 | 1.746 | 3.362 | 6.302 | 10.501 | 17.173 | 28.506 | 36.736 | 43.411 |
| 16 | 2 | 1.614 | 3.179 | 5.765 | 10.487 | 16.790 | 26.265 | 36.867 | 43.595 | 48.184 |
| 16 | 4 | 2.612 | 5.211 | 9.307 | 15.970 | 25.693 | 35.052 | 42.358 | 47.197 | 49.319 |

### Mixed concurrency matrix — GB/s per direction

| N | Q | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0.035 | 0.117 | 0.237 | 0.487 | 0.925 | 1.709 | 3.087 | 4.760 | 6.908 |
| 1 | 2 | 0.122 | 0.241 | 0.466 | 0.939 | 1.762 | 3.034 | 4.728 | 6.949 | 9.073 |
| 1 | 4 | 0.236 | 0.516 | 0.922 | 1.748 | 3.143 | 4.543 | 6.853 | 9.091 | 10.814 |
| 2 | 1 | 0.124 | 0.240 | 0.466 | 0.960 | 1.772 | 3.374 | 6.133 | 8.825 | 13.093 |
| 2 | 2 | 0.239 | 0.460 | 0.929 | 1.783 | 3.253 | 5.285 | 8.779 | 12.863 | 16.795 |
| 2 | 4 | 0.435 | 0.915 | 1.748 | 3.209 | 5.938 | 8.789 | 12.691 | 16.804 | 19.509 |
| 4 | 1 | 0.234 | 0.458 | 0.943 | 1.814 | 3.494 | 6.510 | 10.460 | 16.503 | 23.240 |
| 4 | 2 | 0.467 | 0.935 | 1.822 | 3.529 | 6.482 | 10.433 | 16.278 | 23.070 | 28.572 |
| 4 | 4 | 0.881 | 1.648 | 3.336 | 5.524 | 8.771 | 15.734 | 22.292 | 28.049 | 32.372 |
| 8 | 1 | 0.445 | 0.879 | 1.719 | 3.340 | 5.298 | 9.552 | 15.095 | 21.599 | 27.532 |
| 8 | 2 | 0.882 | 1.660 | 2.680 | 5.483 | 9.615 | 15.216 | 21.674 | 26.995 | 31.637 |
| 8 | 4 | 1.321 | 2.619 | 4.893 | 8.777 | 14.252 | 20.469 | 26.467 | 31.086 | 33.152 |
| 16 | 1 | 0.704 | 1.323 | 2.768 | 4.666 | 8.920 | 14.104 | 19.725 | 24.985 | 29.010 |
| 16 | 2 | 1.253 | 2.555 | 4.747 | 8.404 | 12.725 | 19.044 | 24.446 | 27.918 | 30.822 |
| 16 | 4 | 1.987 | 4.003 | 7.335 | 11.794 | 18.078 | 23.842 | 27.652 | 30.304 | 32.364 |

## 2026-10-03 00:13–00:20 UTC — full exact GEMM threshold/control proof

Snapshot `cfeec35`, V9 Fast int8, both expert shapes, one-second reuse loops.
Every row below passed full CPU reference checks on first and last submit,
fresh poison on final; all three windows ended rc=0, no kernel log errors.
This also exercises the unified Rust HIP VMM/export/import helper on both
Host modes, not merely the standalone GPU allocator.

| Backing | Burst | Shape | Submits | Busy TOPS | Wall TOPS |
|---|---:|---|---:|---:|---:|
| Host pinned | 0 | gate_up 4096×1280×2560 | 883 | 23.8581 | 23.5226 |
| Host pinned | 0 | down 4096×2560×640 | 1515 | 20.6819 | 19.9095 |
| Native | 0 | gate_up | 847 | 22.8823 | 22.5693 |
| Native | 0 | down | 1476 | 20.1936 | 19.3918 |
| Host uncached | 3 | gate_up | 941 | 25.3880 | 25.0700 |
| Host uncached | 3 | down | 1745 | 23.8552 | 22.9308 |

Logs: `mempath2-cfeec35-v9-full-{vmm-host-burst0,native-burst0,vmm-host-uncached-burst3}.log`.
The literal +10.12% slow-burst DMA row does **not** become a >10% exact GEMM
win: pinned Host/native burst0 busy gains are 4.26% gate_up and 2.42% down.
Both are slower than the established default native best (25.36/25.19
gate_up, 23.93/23.46 down busy/wall). Uncached Host/default burst3 likewise
does not improve that established best meaningfully.

Final reachable verdict: amdgpu-managed Host backing and tested safe AXI
attributes do not unlock a faster NPU memory route; pure-read ceiling remains
55–56 GB/s, versus ~239 GB/s on the same Halo GPU. Full concurrency expansion
does not cross 60 GB/s. No-migration VRAM requires an unimplemented driver/
addressing change, not a HIP allocation switch. No host/driver change was made.

### Final Rust live-metrics smoke

Clean Git snapshot `8dcb230`, locked test-host window 00:26:49–00:26:58 UTC:
`npu-tools soc-metrics --once` read the real Halo metrics table and emitted
the expected 22-column numeric row, including fabric 2000 MHz and idle NPU
clocks/counters. rc=0, no kernel log errors, fabric restored to auto.
Log: `logs/mempath2-8dcb230-metrics.log`. The redundant metrics
argument-vector allocation was removed without changing either CLI mode.

## 2026-10-03 06:52 UTC — fabric-clock guard smoke (npu `0e082b7`)

`railgun::npu::fclk::FabricClockGuard` on silicon, one exclusive hold (`tools/npu/npu-fclk-smoke.sh`, 06:52:39–06:52:40
UTC, binaries sha256 npu-coop `b585e2d0…1ea3`, npu-tools `26c85872…9fac`), log `logs/fclk/fclk-smoke-0e082b7.log`:

| step | result |
|---|---|
| incoming `npu-tools fclk status` | perf `auto`, not pinned (rc 1) |
| `npu-coop --mode alias`, default `require`, unpinned | refused before any GPU/NPU work: `fclk guard: refusing concurrent GPU+NPU work (npu-coop) … not pinned`, rc 1 |
| `NPU_FCLK_GUARD=pin npu-coop --mode alias` | guard pinned (`manual`, `7: 2000Mhz *`), alias PASS (userptr all ok, VMM export detected), guard restored `auto` on drop, rc 0 |
| `npu-window.sh npu-coop --mode alias` | window pinned via `npu-tools fclk pin` in 13 ms; guard saw the pin (`require`, held as found); alias PASS; pin still held after cmd; `fclk restore` → `auto`; kernel log empty; rc 0 |
| final status | perf `auto` |

Under `auto`, `pp_dpm_fclk` may also show the `*` on the top level (every `auto` read in this window; an earlier
read the same morning showed no `*`), so the guard requires `manual` as well as the single top `*`.

## 2026-10-03 — co-op epilogue fold, gate_up / down NPU-column tail (npu coop-epi-fuse 5e0ba76)
Details: `docs/npu/coop-epi-fuse.md`. PM twins of the incumbent V2B gate_up SiLU / down ADD (code bytes equal to the
vendored incumbent up to dealloc+endpgm) plus a flag-gated, work-claimed NPU-column epilogue; merged wide tails as the
alternative. GPU-only windows WF1–WF3, co-op windows WC1–WC2: 3 fresh processes per point, the 3 tails ABBA in-process,
all exact (GPU columns 0 mismatches, NPU h == SiLU-DAG(oracle g,u), C == oracle), V9 health exact after each, rc=0.
Delta vs GPU-alone, ms/layer: gate_up J=3 fold −0.89/−0.67/−0.68 (sep −0.39/−0.46/−0.36); J=2 fold −0.14/−0.29/+0.41;
J=1 fold +0.36/+0.18/+0.27; down n=1024 wide +1.10/+1.03/+1.07 (L42 +1.01), fold +1.30/+1.27/+1.30. The fold removes
the 2.77 ms tail but the window grows ~2.4 ms (one workgroup per WGP). Step E: gate_up −0.6…−0.8 %, down +1.0 %.
**KILL** wiring gate_up co-op as an opt-in hipfire route. Logs `logs/coopepi/` (workspace-local).
