# gfx11 retained PM4: missing COMPUTE_PGM_RSRC1.PRIV faults under CWSR

**Lifecycle:** `historical` (per [`docs/investigations/README.md`](README.md))
**Date:** 2026-10-29 (master A/B added 2026-10-30)
**Status:** fix `measured` on gfx1100 against `master` @ `561c37926` (single-slot default PM4 replay) and on a development stack; see "Not tested".
**Code:** `crates/redline-rocr/src/pm4_gfx10.rs` (`with_gfx11_cwsr_priv`), `crates/rdna-compute/src/replay.rs` (`gfx11_cwsr_priv`, `debug_force_evict_start`).

## Summary

On GFX11 the AQL dispatch microcode sets `COMPUTE_PGM_RSRC1.PRIV` (bit 20) for
every dispatch, and CWSR (compute wave save/restore) relies on it. Redline's
raw-PM4 encoder copied the code object's `compute_pgm_rsrc1` unchanged, so the
bit stayed clear. When KFD preempted the PM4 queue in the middle of a retained
dispatch, the GPU took a page fault at VA 0x0 and the queue wedged.

The fix sets the bit on GFX11 PM4 dispatches only. With KFD queue evictions
forced every 10 ms on `master`'s default retained-PM4 route, the unfixed build
faulted in 4 of 4 runs and the fixed build was clean in 4 of 4 (one-sided
Fisher exact p = 1/70). A development stack showed the same split (7/7 vs 0/5).

## Symptom and signature

Hardware: AMD Radeon RX 7900 XTX (gfx1100), Ubuntu 26.10, kernel
7.3.0-9-generic (in-tree amdgpu), `amdgpu.cwsr_enable=1`, ROCm 10.1.0, HIP
runtime 7.16.26385.

Kernel log, in this order:

1. `sq_intr` lines from wgp0 (seen on the development stack).
2. `gfxhub` page fault: ring 24, page VA `0x0`, client `SQC (data)` (a scalar
   load), vmid 8, `GCVM_L2_PROTECTION_FAULT_STATUS 0x00801431` (read,
   `PERMISSION_FAULTS 0x3`), 6 page-fault lines per fault.
3. `MES failed to respond` for `REMOVE_QUEUE` and `SUSPEND`, then a MODE1
   reset.

Daemon side: HTTP 500 for every request after the fault, and
`HipError(700) ... illegal memory access` on shutdown. On the development
stack the replay reported `HIPFIRE_REDLINE_QUEUE_TIMEOUT`.

## Master A/B (2026-10-30)

Base: `master` @ `561c37926` plus the fix and the test-only knobs of this
branch (same tree as the rebased branch). Model: Nex-N2.5-mini MQ4R
(`model_arch=qwen3_5_moe`). Default single-slot serve; no PM4 or kernarg env
overrides. The daemon log of every arm showed:

```
[redline] enabling fail-closed retained default on gfx1100 (model_arch=qwen3_5_moe, drafter=off, transport=pm4)
[redline] retained PM4 kernargs: pool=vram
```

```bash
cargo build --release --locked -p hipfire-daemon -p hipfire-cli
# server
HIPFIRE_DEBUG_FORCE_EVICT_MS=10 \
  ./target/release/hipfire serve --model nex-n2.5-mini.mq4r --kv-mode q8 --spec off --idle-timeout 0
# negative control: add HIPFIRE_DEBUG_PM4_NO_CWSR_PRIV=1
# load: two concurrent chat requests (127- and 1025-token prompts, 64
# generated tokens each), repeated until the 120 s cap
```

Arms were interleaved (off, on, off, on, ...), each started only with the GPU
idle (`GFX_ACTIVITY < 10`), and each was scored from the kernel log for its
own window.

| Arm | Result | Successful requests | Kernel log |
|---|---|---|---|
| Fix off (`NO_CWSR_PRIV=1`), 4 runs | 4/4 page fault at ~15 s | 0 | 6 page-fault lines, `0x00801431`, vmid 8, 2 `MES failed`, 1 MODE1 each |
| Fix on, 4 runs | 4/4 clean to 120 s | 122-123 per run (more than 11,000 forced evictions each) | no fault lines |

One-sided Fisher exact: p = 1/70 (about 0.014).

Fix on, no forced eviction, single-slot: a 60 s smoke run was clean
(86 iterations).

Rebased-branch sanity run (same procedure, binaries built from this branch
after rebasing onto `master` @ `561c37926`; the code tree is identical to the
A/B tree, only docs differ): fix off 2/2 page fault (same signature), fix on
2/2 clean to 120 s (122 iterations each, no failed requests).

`master`'s multi-slot serve (`HIPFIRE_SERVE_MULTI_SLOT=1`) did not engage
retained PM4 replay in a smoke run (no `[redline]` lines), so it was not part
of this A/B.

## Development-stack results (not master)

Before the master A/B, the fault was found and A/B'd on a development stack
with VMM-paged multi-slot serving and retained PM4 tapes, plus local debug
commits (a per-dispatch GPU-clock breadcrumb tape and kernel-side guards).
That route does not exist on `master`; the details below describe that stack
only.

- Plain serving: roughly 1 fault per 100 iterations; unpatched builds also
  faulted with no trigger at all (after 185 s, 315 s and 390 s in three runs).
- `HIPFIRE_DEBUG_FORCE_EVICT_MS=10`: fix off 7/7 page fault, fix on 0/5 clean
  (one-sided Fisher p = 1/792).
- `rocprofv3 --hsa-core-trace` as the trigger: fix off 2/2 page fault, fix on
  0/3 clean (p = 1/10). `--kernel-trace` alone and `--hsa-amd-trace` stayed
  clean for 90 s.
- Control, same forced eviction with the HIP replay backend
  (`HIPFIRE_REPLAY_BACKEND=hip`) instead of PM4: 3/3 clean to 120 s, no fault
  lines. The fault needs the raw-PM4 path.
- Kernel-side null-pointer and slot-index early-return guards, fix off,
  forced eviction: 3/3 still faulted at ~15 s with the same signature. The
  guards do not prevent it.
- umr wave dumps at real faults (5 captures, 144-176 waves each): every wave
  was on the PM4 queue (vmid 8), halted with `TRAPSTS.SAVECTX=1` and
  `UTC_ERROR=1`, PC in the trap handler. The waves were being context-saved
  when they faulted.
- Fix on, no forced eviction: a 300 s run was clean, and a 3 x 300 s soak was
  clean.

Arms that started while the GPU sat at a stale 100% after a reset (four runs)
hit an MES `REMOVE_QUEUE` hang with no page fault and were excluded as
confounded; no arm that started idle had an MES-only hang.

## How the trigger works

`HIPFIRE_DEBUG_FORCE_EVICT_MS=N` (test-only, default off) registers a 2 MiB
host buffer with HIP (a userptr BO) and drops one of its pages every N ms. Each
drop fires the MMU notifier, and KFD evicts and restores the process's queues,
which runs CWSR save/restore on resident waves. `rocprofv3 --hsa-core-trace`
has the same effect. That fits AMD's guidance in the references: on GFX11, PM4
dispatches need `PRIV=1` for CWSR.

`HIPFIRE_DEBUG_PM4_NO_CWSR_PRIV=1` (test-only) skips the fix, for the negative
control; the daemon logs
`[redline] DEBUG negative control: gfx11 COMPUTE_PGM_RSRC1.PRIV not set`.

## Ruled out earlier (development stack)

These were tested before the PRIV candidate was found and did not change the
fault: eliding redundant SH register writes, HDP flushes, kernarg VA
rebinding, CPU BAR writes to kernargs (a GPU-copy kernarg path still faulted),
the GCR bits on entry acquire, publish fencing, and kernel-side null and slot
early-return guards.

## Not tested

- **Mechanism:** why a CWSR save/restore without `PRIV` ends as an SQC read of
  page 0. The fix follows AMD's documented requirement; the path from the
  missing bit to address 0 was not traced.
- **Unforced fault on master:** `master` without forced eviction or a
  profiler was not soaked long enough to see a natural fault (the unforced
  faults were on the development stack).
- **Other routes on master:** the host kernarg pool, dense models, spec
  decode with a drafter, and multi-slot serving were not A/B'd.
- **Other GPUs:** only gfx1100 was measured. Other GFX11 parts (gfx1101,
  gfx1102, gfx1151) were not run. GFX10 and GFX12 are unchanged by design:
  AMD says the workaround is GFX11-only.
- **Throughput cost** of `PRIV=1` on master: not measured. On the development
  stack, decode tok/s with the fix was within noise of fix off (for example
  183.3 vs 184.1 at 3841 context).
- **CWSR area and control-stack sizing**, the second issue in the AMD gist:
  not checked. Redline creates its queues through `hsa_queue_create`, so ROCr
  sizes them.

## References

- AMD (fxkamd), "Solutions to problems with BERT training with tinygrad on AMD
  GPUs": GFX11 PM4 dispatch must set `COMPUTE_PGM_RSRC1.PRIV=1` for CWSR; the
  AQL dispatch packet does it implicitly; other GPUs should not.
  <https://gist.github.com/fxkamd/ecaaab72694696602b6f350edaab9d8d>
- KFDTest commit 507637ed, "kfdtest: Update COMPUTE_PGM_RSRC1 for software
  trap" ("For software trap in GFX11, COMPUTE_PGM_RSRC1 must have PRIV = 1").
  <https://github.com/ROCm/ROCT-Thunk-Interface/commit/507637ed5b82197eecbf483cdc1234939766549a>
- tinygrad `runtime/ops_amd.py`: ORs `1 << 20` into rsrc1 for gfx11 targets
  only ("Set rsrc1.priv=1 on gfx11 to workaround cwsr").
  <https://github.com/tinygrad/tinygrad/blob/master/tinygrad/runtime/ops_amd.py>
