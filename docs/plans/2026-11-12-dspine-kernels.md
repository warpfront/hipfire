# DSpine bridge HIP kernel contracts (M4b)

Companion to [2026-11-12-dspine-drafter.md](2026-11-12-dspine-drafter.md) and the
[semantics map](2026-11-12-dspine-semantics-map.md) §3/§4. Pins the seven HIP
kernels to concrete signatures, memory layouts, and Tier C acceptance before
implementation starts.

## Placement and build pattern

- HIP sources: `kernels/src/dspine_bridge.gfx1100.hip` (single translation
  unit, gfx1100-only, same convention as `dflash_draft_collapse.gfx1100.hip`).
- Rust launchers: `crates/rdna-compute/src/dspine_bridge.rs` — `include_str!`
  the source, every launcher goes through `launch_maybe_blob` + `KernargBlob`
  so the fast path stays hipGraph-capturable.
- The five-layer DFlash decoder trunk (attention GEMMs, MWMA/`gemm_dispatch`,
  norms, MLP) is **not** re-implemented: the bridge kernels only cover the
  code-bridge stages and the serial refinement. The trunk forward calls the
  existing dflash draft kernels unchanged.

## Shared conventions

- Activations are BF16, matching the upstream contract; reductions
  (RMS stats, top-k score sums) accumulate in F32 and store F32.
- `C = code_size (1024)`, `R = bridge_rank (512)`, `H = hidden (5120)`,
  block `T = 16`, `K = head_topk (16)`.
- Kernel names carry the `dspine_` prefix; each launcher validates shape
  invariants (fail-closed) before enqueue.

## 1. `dspine_prepare_read`

```
in  x            [m, H]   bf16   (hidden + residual, pre-add is done host-side
                                  or in-kernel — pinned here: in-kernel add of
                                  the residual pointer, arg 2)
out raw          [m, H]   bf16
out rms          [m]      f32    (RMS over H, eps from config, F32 accumulate)
```
Grid: one workgroup per row (512 threads over H=5120 → 10 elems/thread),
block reduction in LDS. Rows are independent.

## 2. `dspine_normalize_codes`

```
in  projected    [m, C]   bf16   (hidden @ read_weight / early_read_weights[i])
in  codebook     [V, C]   bf16   (GPU-resident, 508 MB)
in  anchor_ids   [m]      u32    (per-block first real token id)
out codes        [m, C]   bf16
```
Row 0 of each block: `codes = codebook[anchor_id] * sqrt(C)` (identity path,
no projection). Other rows: L2-normalize `projected` to `sqrt(C)` scale,
F32 accumulate for the norm. Grid: row-per-workgroup, C=1024 over 256
threads × 4.

## 3. `dspine_feature_message`

```
in  packed       [m, 3R]  bf16   (codes @ packed_weight; segments
                                  query=packed[..R], sender=packed[R..2R],
                                  value=packed[2R..3R] of each row)
in  gate_bias    [R]      bf16
out message      [m, R]   bf16
```
`gate = sigmoid(query_i + sender_{i-1} + bias)`, `message_i = gate *
value_{i-1}`. Block-first rows (`i % T == 0`) write 0. Pure row-neighborhood
shift: `sender/value` segments come from the *previous row within the block*;
rows are laid out block-major so the predecessor is `packed[i-1]` for non-first
rows. Elementwise over R=512; one workgroup per row.

Precondition: `packed_weight` = concat(gate.weight, predecessor_gate.weight,
prev.weight) built at load time as one [C, 3R] GEMM weight, so the packed
GEMM is a plain draft GEMM dispatch (MQ4 or MW16 by tier), not a kernel here.

## 4. `dspine_feature_finish`

```
in  raw          [m, H]   bf16   (kernel 1 output, current layer)
in  rms          [m]      f32
in  delta        [m, H]   bf16   (message @ feature_write_weights[layer] —
                                  plain GEMM dispatch [m,R]@[R,H], not here)
out output       [m, H]   bf16   (raw + rms*delta on non-first rows; raw as-is
                                  on block-first rows)
out normed       [m, H]   bf16   (next layer's input RMSNorm fused; last layer
                                  uses the final norm weight)
```
Grid: row-per-workgroup; the fused next-input RMSNorm uses `rms_norm_eps` from
config. The `(raw, rms, packed)` triple of the last layer is stashed to the
`last_write_stash` buffer by the launcher (device memory, consumed by the
refinement kernels).

## 5. `dspine_head_topk`

```
in  head_logits  [n, V]   f32    (target lm_head GEMM output — the perf-critical
                                  [n,H]@[H,V] GEMM rides the existing lm-head
                                  dispatch; this kernel only selects)
out topk_ids     [n, K]   u32
```
Two-stage chunked top-k over V=248320 (tile the vocab into chunks that fit
LDS, merge). K is a power of two in [2,64] from the contract; the kernel
specializes K=16 (fast path) and falls back to a generic path otherwise.

## 6. `dspine_serial_table`

```
in  base         [n, K]   f32    (lm_head·normalized-hidden scores of the K
                                  candidates)
in  rms          [n]      f32
in  query        [n, R]   bf16   (last_write_stash query segment)
in  gate_b       [R]      bf16
in  TSV_gate     [V, R]   bf16   (precomputed at load: [G_p; A] @ sqrt(C)·codebook)
in  TSV_val      [V, R]   bf16
in  TG           [V, R]   bf16   (bf16(lm_head·norm_w) @ W_last)
in  sid          [n]      u32    (sender id chain: anchor row = anchor token id,
                                  then each row's selected candidate)
out table        [n, K]   f32    (selection scores; argmax per row = serial pick)
```
`score(c) = base[c] + rms * (sigmoid(query + TSV_gate[sid] + gate_b) ·
TSV_val[sid]) @ TG[cand]` — the inner product over R=512 in F32. One
workgroup per (row, K-candidate) tile.

## 7. `dspine_serial_chain`

```
in  table        [n, K]   f32    (kernel 6 output)
in  cand_ids     [n, K]   u32    (kernel 5 output)
in  anchor_id    [n]      u32
out draft_ids    [n, T]   u32    (greedy draft sequence, T=16 rows)
```
Walk from the anchor row: `sid_0 = anchor_id`, each subsequent row picks
`argmax(table[row])`, updates `sid` to the selected candidate's token id.
Sequential over n rows in one workgroup (the chain is inherently serial);
T=16 steps of K=16 argmax.

## Sampled arm (v1 scope note)

The sampled path (`serial_rs_table` / `serial_sample_chain`) recomputes the
exact message/write/norm/head per candidate pair in F32 and emits q-values for
the existing rejection-sampling verifier. Upstream keeps only the greedy path
in a CUDA graph; v1 implements the greedy arm first (Tier C/P/S), the sampled
arm follows as its own slice with the same contract style (explicit F32
accumulation; HIP has no TF32 to disable).

## Tier C acceptance (test_kernels)

Each kernel gets a `test_kernels` case with a CPU oracle over small shapes
(m ∈ {1, 17, 256}, the block-boundary cases included): bit-exact BF16 passthrough
stages (2 identity path, 3 block-first zeros), F32-reduction tolerance gates for
1/4/5/6 (the oracle mirrors the f32 accumulation order), and a full
5-layer+bridge mini-drafter parity check on random weights against the
`raon_oracle`-style CPU model. gfx1100 only (contract tiers elsewhere defer to
the generic path the same way dflash does).

## Out of scope for this doc

- The target-side embedding tap plumbing (semantics map §5, small item, lands
  with the speculator build).
- `admissions.yml` (product-refused by plan) and registry packaging (M5).
