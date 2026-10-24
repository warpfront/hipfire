# Issue #925: real-model position separation (lab only)

Base: upstream `beta` at `8c148af7861c624bba28ed0805bddc4379c84aad`.
Builds on the [first storage slice](2026-10-21-tiered-kv-prototype.md).
Raw results and measured limits: [checkpoint](../../benchmarks/results/2026-10-22-tiered-kv-model/README.md).
This is a prerequisite experiment, **not a serving admission or a resolution
of #925**. Production callers still supply no position override.

This remains strict opt-in: an explicit `lab` feature build and invocation of
the diagnostic executable are required; `bounded` also requires an explicit
mode argument (the model oracle defaults to ordinary `baseline`). There is no
new `hipfire serve` configuration or automatic tiered route.

## What changed

The existing `lab` feature exposes `forward_scratch_positioned_lab` for
single-request, dense Qwen35-family AR with legacy Q8/Q8 KV. It separates:

| Domain | Meaning |
|---|---|
| Absolute position | Original token index, used for Q/K RoPE |
| Packed write slot | Current token's row in the selected-history device slab |
| Packed frontier | Exactly `write_slot + 1` initialized rows visible to attention |

DeltaNet still advances over every logical token. Stored post-RoPE K and V
are copied unchanged. No compaction, rotary re-basing or recurrent reset is
used. The folded FA KV writer takes one position input, so the lab override
uses the existing separate writer instead; default callers keep the fold.
The hook refuses VMM, adaptive/other KV codecs, MoE/EP, graph/Redline and CPU
offload before touching model state. Legacy's mapping helper is a no-op, so
the hook also checks the actual K/V backing extents before dispatch.

## Four independent processes, one binary and one card

`qwen35_tiered_kv_oracle` uses an ordinary full batched prefill for every arm:

1. `baseline`: existing ordinary AR, all history resident.
2. `identity`: lab positional hook, absolute position equals packed position,
   all history resident. Every step's complete logit hash, token, final valid
   KV, hidden and recurrent/conv/scales/error-feedback state must equal AR.
3. `bounded`: pinned-host archive of completed 128-token prompt blocks;
   keep the first block plus the most recent selected blocks on the device,
   followed immediately by any partial prompt block and a generation reserve.
   Absolute RoPE continues at the original position, not the packed slot.
4. `wrong-rope`: deliberately uses packed positions for RoPE too, as a
   negative control. Never a product arm or a quality result.

Before freeing the original KV, the bounded arms compare **every byte of
every selected block and partial tail, on every FA K and V side**, against
the corresponding original ordinary backing. This reference does not read
the archive or trust its map. CPU negative controls corrupt blocks, tails
and sides and must fail. First-new-FA-K matching alone cannot prove that
history was staged correctly: that K is produced before this layer reads
the history. The separate check is essential.

The first generated token is sampled from common full-prefill logits.
Only subsequent logits exercise the selected history. The first new K at
the first FA layer must match AR with correct absolute RoPE and differ
with deliberately wrong RoPE. Later bounded tokens need not equal full
history: selection intentionally changes attention, not just storage.

Receipts use `hipfire.tiered_kv_model.v2`, including actual model and binary
SHA-256, input hashes, full per-token logit hashes, stage-verification counts,
effective route and memory counters. The comparator rejects old v1 receipts,
mixed binaries, missing/false byte verification and insufficient long decode.
Do not relabel an old receipt to v2. A run's `pass=true` means only that the
individual oracle completed; cross-arm checks belong to the comparator.

## Reproduction

The explicit `HIPFIRE_KV_BACKEND=legacy` below is intentional. Current
gfx1100 split-verify defaults admit only VMM, so the incompatible verifier
must be disabled for this lab route. The initial refusal is not a GPU bug.
Use an otherwise idle card and the existing per-card lock. Example for the
evidence host's GPU0; replace UUID, visibility index and BDF together.

```sh
cargo build --release --locked -p hipfire-arch-qwen35 --features lab \
  --example qwen35_tiered_kv_oracle

source scripts/gpu-lock.sh
gpu_acquire tiered-model-lab GPU-dee503dba9717041
export ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0
export HIPFIRE_EMU=rx7900xt HIPFIRE_GRAPH=0 HIPFIRE_AR_GRAPH=0
export HIPFIRE_REPLAY_BACKEND=hip HIPFIRE_KV_BACKEND=legacy
export HIPFIRE_KV_ADAPTIVE=off HIPFIRE_FORWARD_LOWERED=1
export HIPFIRE_FA_KVWRITE_FOLD=0 HIPFIRE_DN_STATE_EF=1
export HIPFIRE_PREFILL_CHUNK_ROWS=1024 HIPFIRE_PREFILL_MAX_BATCH=1024
export HIPFIRE_GFX1100_FA2_SPLIT_VERIFY=0
# Create a NEW directory; every oracle/comparison output refuses overwrites.
mkdir results/tiered-model-reproduction
for arm in baseline identity bounded wrong-rope; do
  python3 scripts/tiered_kv_vram_guard.py --pci-bdf 0000:53:00.0 \
    --out "results/tiered-model-reproduction/${arm}.vram.json" -- \
    target/release/examples/qwen35_tiered_kv_oracle \
      --model "$HOME/.hipfire/models/qwen3.8-27b.mq4-xts" \
      --mode "$arm" --decode-tokens 1024 --prefix-records 288 \
      --window-blocks 8 \
      --prompt-file benchmarks/prompts/mq4v2_k5120_long_ar.txt \
      --out "results/tiered-model-reproduction/${arm}.json" \
      > "results/tiered-model-reproduction/${arm}.log" 2>&1 || break
done
gpu_release
python3 scripts/compare_tiered_kv_oracles.py \
  --baseline results/tiered-model-reproduction/baseline.json \
  --identity results/tiered-model-reproduction/identity.json \
  --bounded results/tiered-model-reproduction/bounded.json \
  --wrong-rope results/tiered-model-reproduction/wrong-rope.json \
  --guard baseline=results/tiered-model-reproduction/baseline.vram.json \
  --guard identity=results/tiered-model-reproduction/identity.vram.json \
  --guard bounded=results/tiered-model-reproduction/bounded.vram.json \
  --guard wrong-rope=results/tiered-model-reproduction/wrong-rope.vram.json \
  --out results/tiered-model-reproduction/comparison.json
```

`rx7900xt` is the existing **20 GiB allocation-capacity profile**, running
on the physical gfx1100 without compute emulation or identity spoofing.
The independent sysfs guard defaults to a 24 GiB device-wide cap minus
512 MiB safety margin. Managed limits do not include every driver/JIT
allocation; sampled physical peaks can miss short spikes. Neither is proof
of the issue reporter's complete 24 GB deployment.

## Limits and next work

- Initial prefill still holds full KV and its scratch. It is **not bounded
  prefill**, so this slice does not make a 256K real-model input fit.
- The fixed generation reserve is sized for the requested decode budget.
  There is no decode-time restaging or publication of new generated blocks.
- Sink/recent selection is a reference policy, not query-based retrieval.
  Stored byte preservation cannot establish answer quality after selection.
- Per-step full-logit downloads/hashing deliberately perturb execution;
  elapsed time is **not decode-throughput evidence**.
- The executable is a one-shot lab process, not an engine resource owner;
  early failure relies on process teardown for remaining allocations.
- Serving/session integration, streamed bounded prefill, retrieval quality,
  speculative commit/rollback, graph/PM4, vision and multi-request ownership
  remain separate, unvalidated work.

### Next implementation seam: bounded chunked prefill

Read-only planning, **not implemented or validated**: in
`qwen35/prefill.rs`, `forward_batch_chunk_impl` currently shares `start_pos`
between mapped-capacity checks, uploaded KV positions, FA route admission and
`logical_max_ctx`. The minimal next lab seam needs separate `absolute_start`
and `packed_start`: KV positions become `packed_start + i`, while a distinct
RoPE position buffer holds `absolute_start + i`. FA preparation must explicitly
select that RoPE buffer; merely pre-uploading it is insufficient because the
current preparation reads it only in the tree-context route. Do not borrow
tree mode or deprecated `compact_offset` as a sparse-history contract.

FA attend/`execute_fa_attend_step` must use packed starts/frontiers, so row
`i` sees exactly the selected history plus current rows through `i`, never a
padding hole or future chunk row. Admission and actual backing bounds must
use that same packed extent. Keep ordinary callers on `None`, and first
require multi-chunk all-history identity against the original prefill.

Two bypasses need explicit treatment: lean `PrefillBatchScratch` does not
allocate an independent `rope_positions` buffer (it aliases a zero-length
sentinel), so the first hook must allocate non-lean scratch with checked
`n * 4`-byte position buffers. Also the gfx1201 Q8 F5 route
`batch_chunk_fa_pm_qprep_prep` uses `pbs.positions` directly for K RoPE and Q
angles, bypassing ordinary FA prepare. Refuse it in the initial lab seam or
split its RoPE input too. Q8 admission alone does not exclude that path.
Initially refuse merged chunk segmentation and per-token fallbacks; handle
a one-token tail through the already-tested positional decode hook.

Archive publication occurs only after a chunk has completed every layer and
synchronized. Publish newly completed absolute 128-token blocks using the
correct packed source rows; carry the <128-row partial block separately before
restaging. The last emitted-but-not-forwarded token is not KV and must never
be archived. Restaging and generation-reserve reuse require this explicit
committed frontier before any serving or DFlash transaction work.

CPU checks:

```sh
cargo test --locked -p hipfire-arch-qwen35 --features lab --lib positioned_lab_
cargo test --locked -p hipfire-arch-qwen35 --features lab \
  --example qwen35_tiered_kv_oracle
cargo test --locked -p hipfire-runtime --features lab --lib tiered_kv::tests
python3 scripts/test_compare_tiered_kv_oracles.py
python3 scripts/test_tiered_kv_vram_guard.py
```
