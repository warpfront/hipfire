# Issue #925: bounded device KV with host history — first storage slice

Experimental design and evidence, not a product admission or a benchmark claim.
Base: upstream `beta` at `8c148af7861c624bba28ed0805bddc4379c84aad`.

## Scope

`hipfire-runtime::tiered_kv` and `tiered_kv_storage_oracle` are gated by the
existing `lab` feature. No config key, user command, default, model-forward
route, quant format, or production kernel changes. This is an independently
testable prerequisite, **not an implementation of the whole #925 request**.

- The archive stores complete **committed** token-major blocks in pinned host
  memory, preserving encoded K/V bytes, including scales and post-RoPE K.
- The device owner has fixed buffers for each full-attention layer and side,
  with a selected-history prefix and a separate generation reserve. It owns
  its slot map; maps cannot be attached to newly allocated or reordered slabs.
- A selection is unique, chronological, bounded and belongs to one archive.
  Block identity includes archive owner, absolute start and revision. Only
  changed physical slots cross H2D; unchanged selections copy zero bytes.
- Copy failure invalidates the entire view. A divergent block-aligned truncate
  invalidates its device view and never rewinds block revisions. Partial-block
  truncation is refused in this first slice.
- Transfers are synchronous reference operations, not an optimized transport.
  Retrieval scoring and recent/sink preservation policy are NOT implemented.

For the issue's Qwen shape (16 FA layers, 4 KV heads, D=256), Q8 has
`4 * (256 / 32) * 34 = 1088` bytes per token **per side per layer**.
256K host history = 9,126,805,504 bytes (8.5 GiB). A 32K selected prefix plus
8K generation reserve = 1,426,063,360 device KV bytes (1.328125 GiB). These
are payload counts, not total model/driver/process VRAM.

## Capacity validation on a larger GPU

Two independent mechanisms are needed:

1. The oracle installs the existing process-local managed-allocation ledger
   **before GPU allocation**, capped at 24 GiB minus 512 MiB. It checks that
   an oversized request fails with the adapter's budget-refusal message, not
   merely an indistinguishable raw HIP OOM. It records managed snapshots.
2. `scripts/tiered_kv_vram_guard.py` samples device-wide sysfs VRAM at 50 ms,
   retaining every raw sample, baseline, peak and limit. It stops only its own
   child process group on excess or observation failure. This is **not a hard
   physical cap**: it can miss transient peaks, and driver/JIT allocations are
   outside the managed ledger. Another user's allocation is counted, never
   terminated. Use an otherwise idle, locked card.

The optional resident reservation is **synthetic allocation pressure**, not
actual model weights, vision, DFlash or measured scratch. Passing under it
does not prove that a 27B serving configuration fits or works on a 24 GB card.
No artificial compute/clock/bandwidth derating is performed.

```sh
cargo test --locked -p hipfire-runtime --lib --features lab tiered_kv::tests
python3 scripts/test_tiered_kv_vram_guard.py
cargo build --release --locked -p hipfire-runtime --features lab \
  --example tiered_kv_storage_oracle

# First acquire the selected card with scripts/gpu-lock.sh; pin visibility to
# that card. Replace the BDF with that same physical card's PCI address. Each
# output path MUST be new (existing evidence is never overwritten).
python3 scripts/tiered_kv_vram_guard.py --pci-bdf 0000:9e:00.0 \
  --out storage-256k.vram.json -- \
  target/release/examples/tiered_kv_storage_oracle \
    --history-tokens 262144 --window-tokens 32768 \
    --resident-reserve-bytes 19327352832 --out storage-256k.json
```

Do not combine this oracle's direct ledger installation with `HIPFIRE_EMU`.
The generic external guard can wrap ordinary `HIPFIRE_EMU` experiments too.

The synthetic oracle validates every side's byte preservation and generation
reserve canaries; Q8 attention on the first/last FA layer is compared with
independent CPU f64 softmax over the original selected rows. Output is NaN
poisoned before each dispatch. It exercises a contiguous selection, a gapped
selection, unchanged/delta transfers, invalid selections and a changed-payload
divergent suffix. Synthetic encoded K rows are stand-ins for post-RoPE bytes:
**this does not validate model RoPE, recurrent state or speculative semantics**.

## Next slices and non-negotiable boundaries

1. Separate `absolute_position`, `packed_write_slot` and
   `packed_visible_length` in Qwen35 legacy Q8 AR. Do not misuse deprecated
   CASK compaction or tree-verification APIs as a sparse-position contract.
2. Add an explicit reference selector and all-history parity on a small model
   before trying approximate retrieval. DN continues processing every token;
   dropping FA history must never reset or skip recurrent updates.
3. Chunked prefill must preserve absolute RoPE positions and use packed causal
   frontiers. Add real flagship battery/chain plus long-context quality checks.
4. DFlash needs transactional history publication: only seed + accepted
   forwarded rows enter the archive, not rejected suffix or an unforwarded
   bonus token. Rollback/terminal repair includes mapping and archive marks,
   alongside DN/conv/scales/error-feedback. A short-window draft alone does
   not prove compatibility.
5. VMM, multiple requests, graph/Redline and vision are separate admissions.
   Unsupported combinations must be refused, not silently restored to a
   full-history device allocation.

No default or PR admission is requested by this experiment. A retrieved subset
can miss important evidence even if every stored byte survives; quality must be
measured independently, not inferred from storage byte parity.
