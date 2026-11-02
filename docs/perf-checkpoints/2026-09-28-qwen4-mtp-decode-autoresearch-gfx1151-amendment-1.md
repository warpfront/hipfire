# Amendment 1 — Qwen4 native MTP decode autoresearch (gfx1151) — 2026-09-28

**Lifecycle:** `historical`

**Amends:** [2026-09-28-qwen4-mtp-decode-autoresearch-gfx1151.md](2026-09-28-qwen4-mtp-decode-autoresearch-gfx1151.md)
(unchanged). Same fixture, prompts, harness and gate; this records the
continuation of the campaign on the same branch (`69fd81923` → `d554080d1`).

**Disposition:** measured local deltas on `gfx1151`. Not a G5 admission, not a
retained-replay certification, not a cross-architecture result, not a product
speed-floor update. External `llama-server` processes kept the GPU busy during
the whole continuation; small deltas were confirmed with back-to-back paired
runs of both binaries.

## Result

| | start (run 328, `69fd81923`) | end (run 343, `d554080d1`) |
|---|---:|---:|
| MTP decode, code | 54.7 | **60.9** |
| MTP decode, prose | 37.8 | **41.5** |
| geomean | 45.5 | **50.2** (+10%) |

Acceptance is unchanged (code 3.48 tokens per window, prose 2.12): every change
below is bitwise, so drafts and verify rows are identical and MTP ids equal AR
ids on every run. Few-row forward (`qwen4_rows`, 232-token prefix) 4 rows:
51.5 → 48.5 ms mean; dispatches 1990 → ~1290 (rocprofv3 count 1430 at run 340,
then 48 + 36 + 60 fewer by construction).

## Validation (end state, daemon md5 `ca3d7172f23e7faffedab5546936ee95`)

- Breadth: the same 9 committed prompts at 200 tokens: MTP ids equal AR ids
  on all; MTP/AR 1.35x-2.01x (was 1.12x-1.74x).
- Serve route: `scripts/serve_harness.py --mode battery --sampling greedy
  --mtp on --thinking off --max-tokens 400 --kv q8 --max-seq 2048`: 5/5 turns
  `finish=stop`, no runaway/empty/attractor; answers read and correct (merge
  function, 210 miles, axial tilt, four-sentence story, five practices).
- Unit tests: `hipfire-arch-qwen4` 56, `hipfire-dispatch` 286, `rdna-compute`
  286 pass.

## What changed

- **Verify launches.** One MQ6 launch for the grouped GDN/QSA projections;
  the HC read writes its consumer's FWHT basis for any row count (GDN, QSA and
  the grouped MoE); the batched GDN gate writes the out-projection rotation
  (head pairs); convolution and gate parameters, and the QSA prologue (norms,
  RoPE, cache append, index-key copy) run in one launch each for few rows.
- **Grouped MoE route.** The router rounds its logits and stores rounded
  weights itself, so the separate round trips go (including a full-capacity
  weight pass per layer); the router projects in the shared selector/gate/up
  launch for few rows; unscatter, SwiGLU and the 128-wide rotation, plus the
  shared-expert activation, are one launch; the scatter's padded prefix sum is
  a block scan (12.6 → 3.8 µs).
- **MTP head.** Its single-row attention uses the batched kernel's parallel
  body (was a per-thread serial dot, ~190 µs at position 300); the reused
  selection length is tracked on the host (no blocking readback); a blocking
  default-stream `hipMemset` per MoE became a stream-ordered kernel; the last
  step of each chain (full-accept commit, replay tail, prompt catch-up)
  appends only K/V and index keys.
- **State copies.** Snapshot, restore and few-row rollback issue one
  `copy_regions_u32` launch per 64 regions instead of one blit per tensor.

## Rejected (measured)

A few-token grouped gate/up kernel: two verify rows share almost no experts
(19 of 20 distinct), and the grouped kernel already streams ~280 GB/s with
MALL hits. MQ6 rows kernel with 2-byte packed loads: no change (the compiler
already merged them).
