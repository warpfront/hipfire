# Tiered KV storage prototype — synthetic gfx1100 evidence

This is a lab-only storage/attention test, **not** a 27B inference run, a
retrieval-quality evaluation, DFlash validation, a performance claim, or a
product admission. No production route is enabled.

## Identity

- Upstream base: `beta@8c148af7861c624bba28ed0805bddc4379c84aad`.
- GPU: W7900 Dual Slot, gfx1100, PCI `0000:9e:00.0`, physical GPU1.
- HIP: `7.15.26333-0000000`; physical VRAM `48,301,604,864` bytes.
- Oracle binary MD5: `8793710d8afcec456a7df7a40a575522`.
- `tiered_kv.rs` SHA256: `57f0527263f90cebfa8e5e8643c7cede7d209672b75669bf17b3bc3f8d2d7073`.
- `tiered_kv_storage_oracle.rs` SHA256: `42b0ce2b91eda8dbd7d7f45eb4109434ff0b341a14b97818c52aca0778958d2a`.
- `tiered_kv_vram_guard.py` SHA256: `1c2eaf915a87b62ff16804e6e8ea8413c9446242038784473b1b34aba7f00b61`.
- Test source SHA256: `1169e391c432e6f5b3eb9139b6a9f533e7ea6ea7dfa348ec49bb351ce94cac9b`.
- Synthetic encoded Q8: 16 FA layers, Hq24/Hkv4/D256, block128, 8K generation
  reserve, 1088 bytes/token/side/layer. No model artifact or prompt was used.
- Each accepted run used one locked card, `ROCR_VISIBLE_DEVICES=1`,
  `HIP_VISIBLE_DEVICES=0`, `HIPFIRE_EMU` and `HIPFIRE_DEVICES` unset.

## Measurements (raw bytes; GiB = 2^30 bytes)

| Run | Host history tokens | Selected tokens | Host archive B | Fixed GPU KV B | Synthetic resident B | Sampled actual VRAM peak B |
|---|---:|---:|---:|---:|---:|---:|
| smoke-v2 | 8192 | 4096 | 285212672 | 427819008 | 0 | 793182208 |
| storage-256k | 262144 | 32768 | 9126805504 | 1426063360 | 19327352832 | 21155815424 |

The large run holds 8.5 GiB history in host RAM and 1.328125 GiB KV on the GPU,
including generation reserve. Its external sampled peak is 19.70288849 GiB
with 18 GiB **synthetic** resident pressure. The managed allocation cap and
external guard threshold are each 25,232,932,864 B (23.5 GiB). The managed
phase-snapshot maximum is 20,765,896,716 B; driver/JIT memory is not included
in that ledger. Both receipts must be read, not substituted for one another.

Single runs only: no repeatability/performance conclusion. Elapsed durations
include CPU fixture generation, transfers, JIT, f64 reference and cleanup;
they are not prefill/decode timing.

## Findings and limits

1. Every archived/staged side was byte-exact, and generation-reserve canaries
   survived all selections. History growth from 8K through 256K did not grow
   the fixed GPU KV owner. The archive's complete 9,126,805,504 B charge was
   released after use; the HIP layer retained its unrelated 8-byte pinned
   staging allocation. Device VRAM and KFD processes returned to pre-run
   baseline after process exit.
2. All three large-run selections passed independent CPU f64 attention on
   FA layers 0 and 15: prefix, gapped selection including the last 256K block,
   and divergent suffix with **different payload bytes**. Maximum absolute
   error was `8.381903171539307e-9`, below `1e-4`; output was NaN-poisoned
   before every dispatch and non-finite output fails the oracle.
3. Repeating an unchanged selection transferred 0 B. Replacing one block
   transferred 4,456,448 B (4.25 MiB across every layer and side), rather
   than the large selection's 1,140,850,688 B. An invalidated divergent
   window recopied the full selected prefix as intended.
4. Duplicate/uncommitted selections and unaligned truncation were refused.
   The oversize-allocation negative control required the managed adapter's
   refusal message, with no pending charge; a raw physical HIP OOM does not
   count. CPU fault injection separately checks whole-map invalidation after
   a partial copy and full reload on retry, stale revisions and request IDs.
5. This proves only a storage primitive and synthetic attention seam. It does
   not test actual RoPE production, DN state, a query-driven selector, vision,
   batch, graph/Redline or speculative commit/rollback. Next: single-request
   legacy-Q8 AR with explicit absolute/packed position separation, then model
   parity and quality evaluation before DFlash integration.

The `initial-failed-smoke.*` files preserve the first harness failure verbatim.
The old oracle incorrectly required **all** managed pinned memory to reach zero,
including an unrelated HIP 8-byte staging owner. It exited nonzero; no valid
result JSON was produced. The corrected oracle checks the archive's exact
before/after release delta, then was rebuilt and rerun from fresh processes.
No raw logs or receipts were edited to turn this failure into a pass.

CPU/build checks: tiered storage unit tests **7/7**, guard tests **19/19**,
release oracle build, targeted lab clippy, and final
`scripts/pre-commit-ci.sh origin/beta` all passed. Raw CPU/clippy/CI outputs
are attached as lossless `.log.gz` files (existing upstream warnings retained).
The first pre-commit attempt found the new module declaration out of rustfmt
order; it was moved, then the complete pre-commit suite passed. Full workspace
unit/no-GPU CI, production serve and Redline routes were not run for this
lab-only slice and are not claimed.

## Reproduction

Build and CPU checks:

```sh
cargo test --locked -p hipfire-runtime --lib --features lab tiered_kv::tests
python3 scripts/test_tiered_kv_vram_guard.py
cargo build --release --locked -p hipfire-runtime --features lab \
  --example tiered_kv_storage_oracle
```

On this host, after a read-only availability check, with new output names:

```sh
source scripts/gpu-lock.sh
GPU_LOCK_TIMEOUT=1 gpu_acquire tiered-kv-256k GPU-ecb57a743c42951e
env -u HIPFIRE_EMU -u HIPFIRE_DEVICES ROCR_VISIBLE_DEVICES=1 HIP_VISIBLE_DEVICES=0 \
  python3 scripts/tiered_kv_vram_guard.py --pci-bdf 0000:9e:00.0 \
  --out new-256k.vram.json -- \
  target/release/examples/tiered_kv_storage_oracle \
    --history-tokens 262144 --window-tokens 32768 \
    --resident-reserve-bytes 19327352832 --out new-256k.json
gpu_release
```

For the smoke use `--history-tokens 8192 --window-tokens 4096` and omit the
resident reservation. Replace device ordinal, UUID and BDF together on another
host. The guard records raw samples at 50ms and can miss transient peaks; it
is not a physical hardware quota or a substitute for a real 24GB model run.
