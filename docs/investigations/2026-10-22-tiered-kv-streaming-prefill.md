# Lab-only bounded chunked prefill for #925

This extends the opt-in storage/position prototype in PR #939. It is not a
production serving mode, a complete solution to #925, or a quality/performance
admission. It requires the `lab` feature and an explicitly invoked example.
Ordinary load, prefill and serving calls keep their existing routes.

## What is bounded

`load_qwen35_bundle_bounded_lab` takes an explicit physical KV row capacity,
independent of the logical context limit. The initial Q8 K/V allocations and
decode attention scratch use that physical capacity. It does not allocate a
full logical-context KV cache and compact it afterward, and it does not use
CASK. Only dense, text-only, single-GPU legacy Q8/Q8 loads are admitted.

`forward_prefill_chunk_positioned_lab` forwards one caller-owned PBS chunk
(at most 512 rows). Absolute positions drive RoPE; packed positions drive KV
writes and each row's causal frontier. The eager DeltaNet path processes all
new input rows once, in order, regardless of which completed FA blocks remain
selected. Unsupported fallback, VMM, adaptive KV, capture/replay, MoE and
speculative paths are refused rather than silently changing the contract.

For block size `B`, selected capacity `W` rows, chunk limit `C` and requested
decode `D`, the example allocates `W + max(C,D) + B - 1` KV rows from load.
There is a short installation overlap of **two physical-cap-sized KV slabs**:
the capped loader slab and the archive window. Both are included in the
managed cap and external sampled VRAM evidence. Neither is a full-history slab
in the bounded arm. PBS size is fixed, not proportional to logical context.

After each successful whole-model chunk:

1. Save the incomplete block's encoded bytes before changing the window.
2. Publish only newly completed 128-token blocks to immutable pinned history.
3. Stage the diagnostic sink/recent selection, in chronological order.
4. Read back every selected K/V side against the archive.
5. Restore and verify the unpublished carry immediately after the **actual**
   selected rows, with no padding gap.
6. Publish the new logical/packed frontiers only after all checks succeed.

An error terminates the one-shot owner; it never retries a partially advanced
DeltaNet chunk. A failed final GPU drain retains the device owners instead of
freeing memory still potentially in use. The last emitted decode token is not
forwarded, following ordinary AR semantics.

## Independent evidence and limits

- CPU row-label simulation checks actual archive/stage/carry content against
  absolute row identities, not merely a second copy of the planner formula.
  It covers repeated saturation, non-aligned chunks and partial blocks.
- `baseline` uses ordinary explicit-PBS prefill; `identity` uses the new
  position-buffer branch with all history retained. Each uses the same chunk
  and singleton schedule. Each full-history chunk must leave its entire
  just-computed GPU KV/DN/hidden prefix unchanged across archive/stage/carry,
  independently of the archive's source plan. Cross-arm hashes must also agree.
- `bounded` must contain a real history gap. Its archived/staged encoded bytes
  and carry are verified. Its later logits, recurrent state and text need not
  equal full-history attention: deleting FA history changes the computation.
- Full-history reference arms deliberately allocate enough for all history;
  they are not reported as memory-saving bounded arms.
- Managed-allocation limits are enforced, while sysfs VRAM observation is
  sampled and can miss short driver/JIT spikes. Point-in-time ledger readings
  are not an allocator high-water mark.
- Per-chunk readback and state/logit hashing make this a diagnostic, **not a
  speed benchmark**. Lossless encoded-byte movement does not imply quality
  neutrality of sink/recent attention.

Still absent: production serve/session ownership, query-driven selection,
decode-time restaging/publication, checkpoint/resume transactions, DFlash
accepted-row transactions, vision and asynchronous overlap.

## Reproduce

Pinned implementation base: `beta@3a5ee848c161b87f28d2437c3cb977de736b4a9c`.
The exact tested model and executable SHA-256 are recorded in each JSON.
Use a fresh output directory; oracle and guard receipts refuse overwrites.

```sh
cargo build --release --locked -p hipfire-arch-qwen35 --features lab \
  --example qwen35_tiered_prefill_oracle

source scripts/gpu-lock.sh
gpu_acquire tiered-prefill-lab GPU-dee503dba9717041
export HIP_VISIBLE_DEVICES=0
export HIPFIRE_EMU=rx7900xt HIPFIRE_GRAPH=0 HIPFIRE_AR_GRAPH=0
export HIPFIRE_REPLAY_BACKEND=hip HIPFIRE_KV_BACKEND=legacy
export HIPFIRE_KV_ADAPTIVE=off HIPFIRE_DN_STATE_EF=1
export HIPFIRE_FORWARD_LOWERED=1 HIPFIRE_FA_KVWRITE_FOLD=0
export HIPFIRE_PREFILL_MAX_BATCH=512 HIPFIRE_GFX1100_FA2_SPLIT_VERIFY=0
mkdir results/tiered-prefill-reproduction
for arm in baseline identity bounded; do
  python3 scripts/tiered_kv_vram_guard.py --pci-bdf 0000:53:00.0 \
    --out "results/tiered-prefill-reproduction/$arm.vram.json" -- \
    target/release/examples/qwen35_tiered_prefill_oracle \
      --model "$HOME/.hipfire/models/qwen3.8-27b.mq4-xts" \
      --prompt-file benchmarks/prompts/mq4v2_k5120_long_ar.txt \
      --mode "$arm" --prefix-records 288 --window-blocks 8 \
      --chunk-tokens 257 --decode-tokens 1024 \
      --out "results/tiered-prefill-reproduction/$arm.json" \
      > "results/tiered-prefill-reproduction/$arm.log" 2>&1 || break
done
gpu_release

python3 scripts/compare_tiered_prefill_oracles.py \
  --baseline results/tiered-prefill-reproduction/baseline.json \
  --identity results/tiered-prefill-reproduction/identity.json \
  --bounded results/tiered-prefill-reproduction/bounded.json \
  --guard baseline=results/tiered-prefill-reproduction/baseline.vram.json \
  --guard identity=results/tiered-prefill-reproduction/identity.vram.json \
  --guard bounded=results/tiered-prefill-reproduction/bounded.vram.json \
  --out results/tiered-prefill-reproduction/comparison.json
```

Adapt the PCI BDF/UUID/visibility to the actual free card. The legacy-only
diagnostic explicitly disables the gfx1100 VMM-only FA2 verifier; a default
incompatible load must refuse, not change backend. On gfx1201 a strict
full-history arithmetic comparison additionally needs both arms to disable
the PM qprep shortcut (`HIPFIRE_FA_PREFILL_PM_QPREP=0`), since the new explicit
absolute-position path intentionally does not use its slot-based RoPE bypass.

## Results

Local gfx1100 W7900s, Qwen3.8-27B MQ4-XTS, HIP 7.15, Q8/Q8 legacy KV,
Q8+EF recurrent state, eager HIP, fixed PBS 257 rows. All five arms requested
and generated 1,024 tokens, finishing by length. The oracle executable SHA-256
was `d0a64188858d24f6c453660df158f1f3f50c533d43cea74e816325d6efed535c`.

| Arm | Prompt tokens | Physical KV rows | KV payload bytes | Sampled whole-device peak bytes |
|---|---:|---:|---:|---:|
| Full-history ordinary, GPU0 | 6,175 | 7,423 | 258,439,168 | 16,477,356,032 |
| Full-history positioned identity, GPU0 | 6,175 | 7,423 | 258,439,168 | 16,518,991,872 |
| Bounded, GPU0 | 6,175 | 2,175 | 75,724,800 | 16,385,163,264 |
| Bounded, GPU1 | 32,383 | 2,175 | 75,724,800 | 16,751,845,376 |
| Bounded, GPU1 | 129,151 | 2,175 | 75,724,800 | 16,793,690,112 |

The full-history ordinary/identity pair agrees exactly on every chunk's
logits, original/staged KV/DN/hidden state, all 1,024 decode logits/tokens,
first new FA K and final state. The independent three-arm comparator passes,
including all guard receipts. The bounded arm has a real absolute/packed gap;
its later output is deliberately not compared for equality with full history.

The same GPU1 fixed KV payload is **72.2168 MiB** at both 32,383 and 129,151
input tokens. Host archive payload grows from 1,123,024,896 to 4,492,099,584 B.
Sampled whole-device peaks are 15.6014 and 15.6403 GiB respectively. These runs
show bounded device KV and a completed long-input model path under a 20 GiB
managed-allocation cap, not constant total host memory, a hard physical 16 GiB
fit claim, or an end-to-end speedup. GPU0 and GPU1 peaks are not a matched
performance comparison. The simple lighthouse outputs were read and remained
coherent; no LongBench/quality-neutrality claim is made.

Normal non-lab regression: the native serve battery passed all five prompts
(114/326/70/112/78 generated tokens, natural stops, no empty/attractor/runaway
or stream errors). The separate Redline harness passed stable capture
(659 dispatches, 16 kernels, hash `b84cfd51ff978677`), AQL contracts and
16-position HIP/PM4 logits/KV/recurrent/GDN-frame parity. These are normal-route
regressions, **not** tiered-mode serving or replay admission.

Focused CPU checks: runtime storage/planner 12, load carrier 13, positioned
prefill admission/geometry 4, example source-copy checks 3, recording inventory
6, VRAM guard 19, previous positional comparator 25 and streaming comparator
29 tests passed (111 total). The pre-commit GPU-free checks passed; the prior
Draft PR's separately documented upstream full-workspace failures are not
reclassified as green.

Two failed preliminary diagnostics are retained: an initial load correctly
refused the VMM-only verifier on legacy KV, and the first new comparator
incorrectly treated normal post-exit guard cleanup signals as a memory trip.
The latter was corrected to the existing guard's completed/exit-code/peak
contract and covered by a CPU regression. No raw receipt was rewritten.

Raw artifacts and source identities are preserved in
[`benchmarks/results/2026-10-22-tiered-kv-streaming/README.md`](../../benchmarks/results/2026-10-22-tiered-kv-streaming/README.md).

### Complete CPU-suite follow-up

The current source's complete workspace-library rerun finished all 58 targets:
8,292 passed, 2 failed, 86 ignored. Both failures are the unchanged tensor-ops
export/index closure cases, reproduced on clean pinned beta. The original
no-GPU script fails on those same cases; every later phase was also executed
separately. The full-source ABI check passes 62/62, CLI 770/770, and the entire
Python list passes 818/818 in an isolated system-Python environment. Initial
Anaconda import failures are preserved and explained, not erased.

The old loader failure is configuration-sensitive, not a current tiered-KV
regression: the same test binary passes with isolated `HIPFIRE_HOME`, fails
with the inherited `asym3` user configuration, and all 186 loader tests pass
in the isolated full run. These findings correct the earlier attribution
without changing the historical logs. Full-suite **pass** claims remain
blocked by the two registry failures. See the
[complete CPU checkpoint](../../benchmarks/results/2026-10-22-tiered-kv-full-ci/README.md).
