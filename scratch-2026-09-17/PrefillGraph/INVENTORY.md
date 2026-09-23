# PrefillGraph slice 1 — launch inventory (widened 4096 chunk, gfx1201 default arm)

Date: 2026-09-19. Worktree `wt-pgraph`, branch `gfx1201-prefill-graph`, base `mq4-lloyd@5cffd8a7d`.
Model `qwen3.8-27b.mq4-xt`, fp8 KV (`new_gpu_fp8_filtered`), Q8 DN+EF, all-MQ4V2,
all gfx12 flags on. Tool `crates/hipfire-runtime/examples/tmp_prefill_chunk_census.rs`.

## Method

Tape recording (`ReplayController`, manual-pm4 vehicle) around two consecutive
4096-row `forward_prefill_batch` calls (resume pattern: shared KV/DN/scratch,
caller-owned PBS per call), plus `hip_bridge::launch_counters` deltas.
Requires `HIPFIRE_GFX12_PREFILL_GRAPH=1`: without it, `is_recording()` forces
legacy 512 cadence (see below) and the census observes the wrong stream.

## Headline (widened, flag on)

- **1828 launches / 4096-row chunk**, 21 unique kernels, both chunks identical
  `seq_hash=991b54b2a51c8bf9`, zero grid/block/LDS/kernarg-len mismatches.
- **Raw (non-blob) launches: 0.** Every launch goes through `launch_maybe_blob`
  or the scratch shared helper (same blob/record gate, parity by construction).
- **Untyped (no ABI/binding_layout): 0.** Slice-1 re-encodes every segment.
- Non-kernel ops per chunk: **htod=2** (tokens 16384 B + positions 16384 B, single
  uploads — pre-uploadable via existing `upload_prefill_batch_inputs`),
  **dtod=1** (final-logits last-row 20480 B), dtoh=0, memset=0, syncs=0,
  **dsync=3 first / 1 steady** (sites unnamed — open item).
- Per-kernel histogram (chunk): FA2 body+preconvert 128+128, kv_write 256,
  deinterleave/rope/sigmoid 16 each, qkv 16, GDN fast 384 (48 LA × 8 commits),
  gdn_pre 48, gated_norm 48, qkvza 48, gate_up 64, residual 128, convert 256,
  fused_rmsnorm 128, rotate 64, silu 64, rmsnorm 17, embedding 1, gemv 1, mq_rotate 1.

## Variant-scalar census (chunk0 vs chunk1 kernargs)

- **Every differing byte is a device-pointer byte** (8-byte slots at 8-spaced
  kernarg offsets; low bytes equal = same interior view offsets; differing
  mid/high bytes = PBS-slide arenas; paired ±deltas; 1-byte arena tags
  0xb0↔0x30, 0x4c↔0xb2). Zero non-pointer diffs — **except GDN frames** (next).
- **GDN stochastic frames are bit-load-bearing and monotonic.**
  `gated_delta_net_q8_fast` bakes `fr = reserve_gdn_requant_frames(1)` (norm.rs).
  Census cross-call diff at frame byte (off=78): 0x06→0x09 in all 384 commits.
  Consequence: a chunk captured at counter F replays bytes that match eager
  ONLY at counter F. Cross-chunk or cross-request replay without frame
  patching is inexact. The tape's `GdnFrameU32`/checkpoint discipline
  (redline.rs shadow harness) is the model; production needs per-replay
  absolute-frame patching (new binding kind) with chunk-k base =
  request_base + A·k (A = deterministic per-chunk advance; verify no
  interleaved reservers — norm.rs:3266/3445 reserve batch_size, confirm path).
- No start_pos/seq_pos/row-count scalars anywhere: positions flow via the
  `pbs.positions` device buffer; FA tiles use tile-local (start+off, +stride);
  grids are shape-invariant at full 4096 rows.

## Classification counts (1828 launches)

- Capturable as-is (stable kernargs, device-buffer inputs): **1444**
  (everything except the 384 GDN commits).
- Needs param patch per replay: **384** (GDN frame slots only).
- Needs launcher change: **0**.
- Needs host-side handling: 2 H2D (pre-upload, exists) + 1 D2D last-row
  (must be inside the replay unit — verify PM4 copy-node support) + 1 steady
  dsync (identify; keep outside or eliminate).

## Backend decision: Redline tape (not hipGraph)

1. No capture-mode restrictions: recording observes the true eager stream
   (hipGraph would force `max_ctx_len=physical_cap`, multirow-off, pair-off,
   f16-off divergences plus H2D/sync-node surgery).
2. Pointer slides (per-call PBS) are natively re-encoded from ReplayBindings.
3. Frame patching on tape = host-side kernarg re-encode per replay (already
   the slice-1 architecture); on hipGraph = 384
   `hipGraphExecKernelNodeSetParams` per replay (≈ the saved launch cost).
4. Decode precedent: retained PM4 route + shadow harness + frame
   checkpoint/restore discipline all exist.

## Trap log

- Recording (or capture_mode) forces legacy cadence by design
  (`wide_candidate` excludes both). Census with the recorder armed but flag
  off observes 8×512 legacy chunks (8352 launches) — NOT the widened stream.
  Fixed by the flag (this slice).
- `launch_maybe_blob_bound` is not the only funnel: 2560 launches/chunk
  (converts + silu sidecars) go through the scratch shared helper in
  scratch.rs — same blob/record gate, parity invariant holds. Kernel-issue
  markers must cover both.
- Single 4096 chunk = 1828 launches < default 4k tape cap. The 8k/16k caps
  were needed only for the legacy-stream artifact.

## Open items (next slice)

1. Name the 1 steady + 2 cold dsyncs (interleave trace, needs temp
   instrumentation again — reverted this slice).
2. PM4 D2D-copy-node support for the last-row copy (or split tape at lm-head).
3. Absolute-frame binding kind + request_base checkpoint; single-call 8192
   census (shared PBS, continued frames) to verify chunk1-vs-chunk0 frame math.
4. Then: capture driver keyed by (rung, route, layer-set) behind the flag;
   eval-md5 ×3, profiler CSV, interleaved bench, decode/serve gates.
## Slice 2 (2026-09-19 eve)

### Single-call-8192 census (production truth: shared PBS, continued frames)

- `--single`: one 8192 forward, tape=3656, split M=1828 by kernel repetition.
- Chunk1-vs-chunk0 diff: **only** (a) GDN frame byte (off=78, all 384 commits,
- 0x06->0x09 = +196608 = 384 commits x 512 frames; `frames = max(1, nt*grid.z)`
- per dispatch, nt=512), (b) 6 layer-0-head singleton pointer bytes
- (idx 2,3,15,16,19; small arena-local deltas) = **scratch growth during the
- first widened chunk** (sidecar scratch sized by pre-widen history, replaced
- mid-chunk0; the 2 cold dsyncs are its sync+free).
- Zero pointer diffs otherwise (shared PBS => identical addresses).
- Zero start_pos/row-count scalars. Confirmed: **frames are the ONLY
- per-chunk variant**, and they are already handled (next).

### dsync answer: 0 steady-state in-chunk syncs

- The 1 steady dsync/chunk was the census's own `device_synchronize`
- (inside the counter window, before `finish_capture`). The chunk body has
- **zero** syncs. The 2 cold ones are scratch-growth replacement (above).
- Production: capture after warmup/JIT/growth, fail open to eager on any
- growth inside the capture window (existing refresh/stale machinery).

### Frame exactness: existing machinery suffices (no new binding kind)

- Prepare auto-attaches `GdnFrameU32 { offset: 76, frames: 512 }` to every
- `gated_delta_net_q8_fast` launch; replay reserves FRESH frames per launch.
- Fresh == eager-chunk-k frames **by counter lockstep**: capture and eager
- consume identically (384x512/chunk), so replay-time reservation lands on
- exactly the frames eager would have used. Counter stays in lockstep
- automatically. Deterministic per process => identical-runs gate holds.
- `synthesize_position_bindings` already exempts offset 76. Nothing to build.

### D2D decision: split lm-head out (no PM4 copy nodes exist)

- redline-dispatch has no D2D-copy packet support. The chunk's only D2D is
- the final-logits last-row copy (20480 B).
- Split: truncate the recorded tape after the last layer kernel (assert tail
- == `[rmsnorm_f32@[1,1,1], mq_rotate_x, gemv_mq4g256v2_multirow_r2]`, drop 3;
- the D2D is never recorded). Replay = 1825 launches; then run
- `batch_chunk_final_logits` eager. Exact by construction; also skips the
- wasted lm-head on intermediate chunks (less work than eager).

### Driver spec (build-ready, not yet implemented)

- Secondary `ReplayController` + prepared state + key in `Qwen35Scratch`
- (model-session lifetime; zero `Gpu` changes). Key = (rung 4096, route
- fingerprint, full-stack layer-set). Fail open to eager on mismatch/growth.
- Capture: first admitted full 4096 chunk runs eager under secondary
- recording, truncate lm tail, prepare (AQL first, PM4 fallback per
- forward.rs:1725/1738), store. Retain one dedicated 4096 PBS for the graph
- route (per-call PBS is freed; cross-request replay needs stable addresses).
- Replay: H2D tokens+positions, `replay_linear_aql(0)`, eager final_logits.
- Tails (<4096) eager. No counter management (lockstep). Transport from
- `ReplayTransport::from_config`, logged in the receipt.
