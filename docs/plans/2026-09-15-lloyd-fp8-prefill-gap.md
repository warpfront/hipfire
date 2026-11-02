# Lloyd FP8 prefill gap: per-slab SGPR LUT select for gate_up + qkv (gfx1201)

Date: 2026-09-15. Scope: MQ4G256V2-Lloyd (qt=52) FP8-LUT GEMM prefill on exact gfx1201.
Worktree: `/home/kaden/ClaudeCode/warpfront/wt-lloyd` (branch `mq4-lloyd`).
No source edits in this plan. All line numbers verified against the worktree this date.

## 0. Ground truth and honest ceiling

Parent-measured, pp512 on gfx1201, per-kernel serialized totals for a 512-token
prefill over 64 layers (BEFORE qkvza S2BT8 re-admission; uniform total ~360 ms):

| Symbol (LUT) | Lloyd | Uniform | Delta |
|---|---|---:|---:|
| `gemm_gate_up_mq4g256v2_wmma_fp8_gfx12_s2bt8_lut` | 128.1 ms | 124.4 ms | **+3.0%** |
| `gemm_mq4g256v2_residual_wmma_fp8_gfx12_s2bt8_lut` | 92.0 ms | 93.0 ms | −1% (noise/parity) |
| `gemm_qkvza_mq4g256v2_wmma_fp8_gfx12_bt12_lut` | 60.9 ms | 43.1 ms (uniform s2bt8) | +41%, **already fixed** |
| `gemm_qkv_mq4g256v2_wmma_fp8_gfx12_s2bt8_lut` | 14.2 ms | 13.3 ms | **+7%** |

After the qkvza fix the bench gap is ~−3% (pp512 1374 vs 1412 tok/s;
pp8192 1250 vs 1286 — Lloyd slower).

Amdahl ceiling (denominator = ~360 ms uniform prefill total):
gate_up +3.0% × 128.1 ms ≈ +3.8 ms ≈ **~1.1% of total**;
qkv +7% × 14.2 ms ≈ +1.0 ms ≈ **~0.3% of total**.
Honest ceiling for this change: **~1.5–2% prefill-time recovery, i.e. near-parity,
not a win past uniform.** Residual and qkvza contribute nothing further (already
at parity / fixed). If post-change profile shows gate_up/qkv at parity but the
bench gap persists, the remainder is outside this kernel (X-prep, launch
geometry, or non-GEMM prefill stages) — do not chase it here.

## 1. Where the remaining ~3% comes from (per-symbol table)

Single source, 16 symbols: `kernels/src/gemm_gate_up_mq4g256v2_wmma_fp8.gfx12.hip`,
via `-DHIPFIRE_FP8_LUT_ARG` (`crates/rdna-compute/src/kernels.rs:3532-3654`;
gate_up BT12/BT8/BT4/S2BT8 LUT at 3532/3540/3548/3556, residual ×4 at
3564/3572/3580/3588, qkvza ×4 at 3596/3604/3612/3620, qkv ×4 at 3628/3636/3644/3652).
BV/SLABS/RESIDUAL/QKVZA/QKV are all compile-time `#if`s in the one `.hip`, so
one edit covers all BT12/BT8/BT4/S2BT8 variants of a family. SLABS=2 (S2BT8) is
the pp512 hot path: every `*_prepared_lloyd` launcher in
`crates/rdna-compute/src/gemm.rs` selects `slabs2` whenever
`HIPFIRE_GFX12_MQ4V2_FP8_SLABS != "1" && batch_size % 128 == 0`
(gate_up 29401-29408, residual 30129-30136, qkvza 30342-30349, qkv 30615-30622);
512 % 128 == 0, so pp512 runs S2BT8 for all four families.

Mechanism. Uniform decodes each nibble pack with one `mq4_lut8` call whose four
table dwords are SGPR immediates (`gemm_gate_up_mq4g256v2_wmma_fp8.gfx12.hip:97-109`;
uniform K-loop decode at 560-563 for SLABS=2, 656-657 for SLABS=1). The LUT-ARG
twin receives table dwords as kernel args (`:88-95` `mq4_lut8(sel,C0..C3)`,
`:113-119` `mq4_qpk_to_e4m3`, arg lists at `:204-207` qkvza / `:222` residual /
`:243-245` qkv / `:263-264` gate_up). Keeping kernel-arg dwords in SGPRs across
the K-loop is the whole game:

| Symbol group | Decode behavior today | Expected gap source |
|---|---|---|
| gate_up `*_s2bt8_lut` (hot) | SLABS=2 K-loop evaluates **2 candidates per nibble pack** (`mq4_lut8` on LG* and on LU*) and `mq4_pick2`-selects per lane by `lut_side[s]` — `:549-556`; selectors hoisted at `:530-537`, side index per lane-row at `:337-346` | 2× `v_perm_b32` decode vs uniform 1× → measured **+3.0%** on the largest bucket (128.1 ms) |
| gate_up `*_bt{12,8,4}_lut` (cold: N%128!=0 or SLABS=1) | Same 2-candidate eval, SLABS=1 form at `:649-652`, side at `:391-395` | Same mechanism; off the pp512 path |
| qkv `*_s2bt8_lut` (hot) | SLABS=2 K-loop evaluates **3 candidates per pack** (LQ*/LK*/LV*) + `mq4_pick3` — `:540-547`; side at `:341-342` | 3× decode vs 1× → measured **+7%** on a 14.2 ms bucket |
| qkv `*_bt{12,8,4}_lut` (cold) | Same 3-candidate eval, SLABS=1 at `:644-647`, side at `:391-392` | Same mechanism; off-path |
| qkvza `*_s2bt8_lut` (hot, fixed) | Per-slab SGPR select keyed on wave-uniform slab start row `rs + 16*s` — `:327-335`; K-loop decodes **once** via SLC arrays — `:522-526` | Gap closed by construction; **do not touch** |
| qkvza `*_bt{12,8,4}_lut` (cold) | Single-LUT select on wave-uniform tile start `rs` — `:385-389`; single decode — `:632-634` | Already SGPR-clean; **do not touch** |
| residual `*lut` all 4 (hot s2bt8 at parity) | Single LC* LUT used directly at decode — `:522-526` (S2BT8), `:632-634` (SLABS=1); trivial SGPR copy prologue `:314-319` / `:379-380` | No candidate eval → measured −1% (parity). **Do not touch** |

Helpers: `mq4_pick2`/`mq4_pick3` at `:133-138` (comment at `:128-132` states the
intent: kernel-arg dwords consumed directly, per-lane decision collapses to VALU
selects). The prologue comment at `:308-313` describes exactly this scheme; the
`#else` branch (`:336-347`) is what gate_up/qkv still use and what this plan
replaces.

Bit-exactness note: the change only selects *which* 4 SGPR dwords feed the
existing `mq4_lut8`; the decode math and FMA order are untouched, so output is
bit-identical by construction (same bytes decoded per row). State this in the
composer task; still verify by differential ( §4 ).

## 2. Implementation spec (composer-ready, no open design decisions)

Goal: give gate_up and qkv the exact per-slab SGPR select qkvza already has,
keyed on the wave-uniform slab start row, with host-side `% 16` fail-closed
admission. Kernel-arg ABI is unchanged (LG*/LU*/LQ*/LK*/LV* still passed);
only the internal select changes.

### Task A — kernel: gate_up + qkv per-slab SGPR select (one file)

File: `kernels/src/gemm_gate_up_mq4g256v2_wmma_fp8.gfx12.hip`.

A1. SLABS=2 prologue (`:336-347`, the `#else` of the
`#if HIPFIRE_FP8_RESIDUAL || HIPFIRE_FP8_QKVZA` at `:322`). Replace the
`lut_side[2]` computation with per-slab LUT-dword arrays mirroring the qkvza
block at `:327-335`, keyed on the SLAB START ROW `rs + 16*s` (wave-uniform from
`blockIdx`, hence SGPR — this is the load-bearing detail; do NOT key on `srx[s]`,
the lane row, which is VGPR):

```c
// gate_up (HIPFIRE_FP8_QKV==0 arm):
unsigned int SLC0[2], SLC1[2], SLC2[2], SLC3[2];
#pragma unroll
for (int s = 0; s < 2; s++) {
    const int srs = rs + 16 * s;
    if (srs < gate_m) { SLC0[s]=LG0; SLC1[s]=LG1; SLC2[s]=LG2; SLC3[s]=LG3; }
    else { SLC0[s]=LU0; SLC1[s]=LU1; SLC2[s]=LU2; SLC3[s]=LU3; }
}
// qkv (HIPFIRE_FP8_QKV==1 arm):
unsigned int SLC0[2], SLC1[2], SLC2[2], SLC3[2];
#pragma unroll
for (int s = 0; s < 2; s++) {
    const int srs = rs + 16 * s;
    if (srs < q_m) { SLC0[s]=LQ0; SLC1[s]=LQ1; SLC2[s]=LQ2; SLC3[s]=LQ3; }
    else if (srs < q_m + k_m) { SLC0[s]=LK0; SLC1[s]=LK1; SLC2[s]=LK2; SLC3[s]=LK3; }
    else { SLC0[s]=LV0; SLC1[s]=LV1; SLC2[s]=LV2; SLC3[s]=LV3; }
}
```

Keep the surrounding `#if HIPFIRE_FP8_QKV` structure; delete `lut_side[2]` for
these two arms only. Residual (`:314-319`) and qkvza (`:320-335`) arms:
untouched. Update the comment at `:308-313` to say all four arms select per
slab (residual trivially).

A2. SLABS=2 K-loop decode (`:527-557`). Replace the `#else` candidate-eval
block (`:528-557`, selectors at `:530-537`, `mq4_pick3` at `:540-547`,
`mq4_pick2` at `:549-556`) with the single-decode form used by
residual/qkvza at `:522-526`:

```c
v2i_t q0a = mq4_qpk_to_e4m3(pk0.x, SLC0[0], SLC1[0], SLC2[0], SLC3[0]);
v2i_t q0b = mq4_qpk_to_e4m3(pk0.y, SLC0[0], SLC1[0], SLC2[0], SLC3[0]);
v2i_t q1a = mq4_qpk_to_e4m3(pk1.x, SLC0[1], SLC1[1], SLC2[1], SLC3[1]);
v2i_t q1b = mq4_qpk_to_e4m3(pk1.y, SLC0[1], SLC1[1], SLC2[1], SLC3[1]);
```

i.e. extend the `:522` `#if` condition to cover gate_up/qkv (or collapse the
`#if/#else` entirely — after A1 all four arms own SLC arrays, so the guarded
block at `:522-526` can become unconditional under `HIPFIRE_FP8_LUT_ARG`).
Delete `mq4_pick2`/`mq4_pick3` call sites; keep the helper definitions
(`:133-138`) only if any caller remains (after this change: none — delete the
helpers too, so a stale caller fails compile, never silently).

A3. SLABS=1 prologue (`:390-396`, inside `:374-397`). Replace `lut_side` with
single LUT dwords keyed on wave-uniform tile start `rs`, mirroring qkvza at
`:385-389`:

```c
// gate_up:
unsigned int LC0, LC1, LC2, LC3;
if (rs < gate_m) { LC0=LG0; LC1=LG1; LC2=LG2; LC3=LG3; }
else { LC0=LU0; LC1=LU1; LC2=LU2; LC3=LU3; }
// qkv:
unsigned int LC0, LC1, LC2, LC3;
if (rs < q_m) { LC0=LQ0; LC1=LQ1; LC2=LQ2; LC3=LQ3; }
else if (rs < q_m + k_m) { LC0=LK0; LC1=LK1; LC2=LK2; LC3=LK3; }
else { LC0=LV0; LC1=LV1; LC2=LV2; LC3=LV3; }
```

This covers the BT12/BT8/BT4 (`HIPFIRE_FP8_BV` 12/8/4) symbols, which share the
SLABS=1 path. Residual (`:379-380`) and qkvza (`:381-389`): untouched.

A4. SLABS=1 K-loop decode (`:635-654`). Replace selectors (`:638-641`) and
candidate eval (`:644-652`) with the direct form at `:632-634`:

```c
v2i_t q8a = mq4_qpk_to_e4m3(pk2.x, LC0, LC1, LC2, LC3);
v2i_t q8b = mq4_qpk_to_e4m3(pk2.y, LC0, LC1, LC2, LC3);
```

Same collapse as A2: after A3 every arm owns LC*, so `:631-654` becomes the
`:632-634` body unconditionally under `HIPFIRE_FP8_LUT_ARG`.

Correctness invariant (composer must preserve): no 16-row slab may straddle two
source tensors, else the slab-start select decodes rows on the wrong codebook.
Enforced host-side (Task B). The `sr`-clamped tail row (`:351-352`,
`(mr < tm) ? mr : (tm - 1)`) can still alias the last row — harmless: with the
`% 16` contract the clamp only fires on the final partial tile whose rows all
belong to the last source... EXCEPT a `tm % 16 != 0` total can leave the last
tile straddling: the tile's `rs` selects one source while clamped lanes read the
other. Therefore the host check must ALSO require `tm % 16 == 0` OR the kernel
must keep the clamped-lane behavior safe. Simplest frozen decision: require each
`m % 16 == 0` (as qkvza does at `gemm.rs:30334`) — for gate_up, `gate_m` and
`up_m` individually `% 16 == 0` implies `tm % 16 == 0`, so no straddle is
possible including the clamp lane. Same for qkv (`q_m`, `k_m`, `v_m` each
`% 16`). This exactly matches the qkvza precedent; do not invent a weaker
check.

### Task B — host: `% 16` fail-closed admission for gate_up + qkv (one file)

File: `crates/rdna-compute/src/gemm.rs`. Template: the qkvza check at
`:30327-30341` (comment + error text). Copy it verbatim in style.

B1. `gemm_gate_up_hfq4g256_wmma_gfx12_mq4v2_fp8_bt12_prepared_lloyd`
(`:29350-29400` guards). After the `batch_size % 64` check (`:29389-29394`),
insert:

```rust
// The LUT kernels key the per-slab codebook on the slab start row
// (wave-uniform -> SGPR), which requires every 16-row slab to lie in one
// source: gate_m and up_m must both be multiples of 16 (hence tm too, so the
// clamped tail lane in the kernel cannot straddle). Any other shape fails
// closed here, never a straddled tile decoding on the wrong codebook.
if gate_m % 16 != 0 || up_m % 16 != 0 {
    return Err(hip_bridge::HipError::new(
        0,
        &format!(
            "gemm_gate_up mq4v2 fp8 lloyd: every projection m must be a multiple of 16 (gate={gate_m} up={up_m})"
        ),
    ));
}
```

B2. `gemm_qkv_hfq4g256_wmma_gfx12_mq4v2_fp8_prepared_lloyd` (`:30560-30614`
guards). After the `batch_size % 64` check (`:30603-30608`), insert the
analogous check with
`"gemm_qkv mq4v2 fp8 lloyd: every projection m must be a multiple of 16 (q={q_m} k={k_m} v={v_m})"`.

B3. Update the stale doc comments that say "the kernel selects per row-source":
gate_up prepared-launcher doc at `:29342-29348`, qkv at `:30556-30558` — change
to "the kernel selects per 16-row slab from the wave-uniform slab start row;
all m must be multiples of 16 (fail-closed)". Qkvza (`:30263-30266`) and
residual (`:30078-30080`) docs: untouched.

B4. No changes to: `kernels.rs` (same sources, same symbols — the edited `.hip`
recompiles under the existing 16 `*_LUT_SRC` constants at `:3532-3654`; the
composer must confirm `ensure_kernel` keys the code-object cache on source
content so the new machine code cannot alias the old — if it keys on symbol
name alone, add a cache-bust and say so in the diff), the `*_lloyd` convenience
wrappers (pad path unchanged), `hipfire-dispatch` (no new keys; qt=52 still has
no fused keys by design — `crates/hipfire-dispatch/src/families/fused_qkv.rs:77-90`
and `crates/hipfire-dispatch/src/families/gemm.rs:251-276` refuse uniform
routing, and `forward_slots.rs:358-360,374-376` panics on fused-key requests
for Lloyd — all of that stays).

### Task C — shape audit (read-only, part of Task B's acceptance)

Verify from the loader (do not change): with the Qwen3.8-27B dense config
(`crates/hipfire-arch-qwen35/src/qwen35/config.rs:1339-1351`: hidden 5120,
intermediate 17408, n_heads 48, n_kv_heads 8, head_dim 128, linear key heads 16
× 128, linear value heads 48 × 128):

- gate/up: `gate_m = up_m = hidden_dim = 17408` (`load.rs:3511-3512`,
  `:3546-3547`); 17408 = 16 × 1088. ✓
- FA qkv: `q_rows = n_heads*head_dim*2 = 12288`, `kv_dim = n_kv*head_dim = 1024`
  (`load.rs:3502-3507`); all % 16 == 0. ✓
- LA qkvza: `qkv_dim = 2*key_dim+value_dim = 10240`, `z = value_dim = 6144`,
  `alpha = beta = value_heads = 48` (`load.rs:3452-3458`, `:3526-3537`;
  `weights.rs:29-32`); all % 16 == 0. ✓ (this is why the qkvza check never
  fires on this model)
- Residual/o_proj/down: `m = dim = 5120` or `hidden` — single-LUT, no check
  needed regardless.
- TP sharding caveat: `dense_tp_rank_layouts` splits `ffn_hidden_range`
  (TP2: 8704/8704 at `config.rs:1360-1365`; TP5: 3584/3584/3584/3328/3328 at
  `:1516-1520`) — all observed splits are % 16 == 0, and GQA exact-cover
  (`config.rs:236-243`) rejects head splits that are not whole units. Single-GPU
  (TP1) is the measured config and needs no split analysis. If a future TP
  split ever violates % 16, the launcher fails closed with the B1/B2 message
  and prefill errors loudly — that is the intended behavior, not a fallback.

## 3. Odd-tail padding: quantified, off the hot path

Wrappers: `pad_f32_batch_to_64` at `gemm.rs:29324-29340` (1 alloc + 1 D2D of
`n*k*4` + zero of the tail), `pad_prefill_batch` at `:30043-30056`,
`unpad_prefill_ys` at `:30060-30076` (one D2D per Y of `n*m*4`).

- N=512 (pp512): `512 % 64 == 0` → all four `*_lloyd` wrappers take the direct
  prepared route (`gemm.rs:29540-29546` gate_up, `:30238-30243` residual,
  `:30530-30536` qkvza, `:30777-30783` qkv): **zero pad copies, zero temp
  allocs.** The pad wrappers are dead code on this path.
- Odd tails (e.g. N=511, K=5120): gate_up does 1 X D2D (~511·5120·4 ≈ 10.5 MB)
  + 20 KB memset + 2 Y D2Ds back (~511·17408·4 ≈ 35.6 MB each) at
  `:29547-29566`; residual additionally preloads Y into the temp
  (`:30247-30252`, required for `+=` semantics — see comment `:30244-30246`);
  qkvza/qkv go through `pad_prefill_batch`/`unpad_prefill_ys` (`:30537-30553`,
  `:30784-30794`). Cost is linear in the tail chunk and paid once per odd-sized
  chunk, not per layer-row — negligible next to the ~360 ms GEMM total and
  irrelevant to the pp512/pp8192 numbers (both % 64 == 0).
- Frozen decision: **do not touch the pad path in this change.** No hoisting,
  no persistent pad buffers, no N'-caching. If a future odd-N workload shows
  pad copies in `profile_prefill_qwen35`, that is a separate task with its own
  measurement.

## 4. Verification recipe (parent runs; composer does not)

GPU work is the parent's (GPUs busy during planning). No GPU command was run
for this plan. Env: exact gfx1201, eager only (all four prepared launchers
reject capture/replay — `gemm.rs:29371-29376, 30099-30104, 30297-30302,
30585-30590`), `HIPFIRE_GFX12_MQ4V2_FP8_SLABS` unset (default S2BT8 at N=512).

V1. Build-only (composer may run; pure Rust, no GPU): `cargo build -p
rdna-compute` equivalent scoped check that the edited `.hip` string still
compiles into `kernels.rs` — actually the `.hip` compiles at runtime via
`ensure_kernel`, so the composer cannot compile-check it without a GPU.
Compensate: meticulous `#if/#endif` balance review of the A1–A4 edits against
the qkvza arms (`:320-335`, `:385-389`, `:522-526`, `:632-634`) which are the
character-for-character template. At least one reviewer must diff the new
gate_up/qkv SLC blocks against the qkvza SLC blocks and confirm identical
structure modulo tensor names.

V2. Kernel differential vs LUT GEMV (correctness gate, parent runs on gfx1201):
for each of gate/up/q/k/v projections, compare the FP8-LUT GEMM output row(s)
against the per-tensor LUT GEMV reference `gemv_mq4g256v2_lloyd`
(`crates/rdna-compute/src/gemv.rs:7824-7873`, source
`kernels::GEMV_MQ4G256V2_LUT_SRC` at `kernels.rs:1578-1581`) driven with the
same `lloyd_lut_f16` sidecar (`lloyd_luts_from_levels` at
`crates/hipfire-runtime/src/lloyd_lut.rs:95-100`; E4M3 codebook at `:88-93`).
Bit-exactness is expected (same decode bytes, same FMA order — only the LUT
*select* moved from per-lane VALU to per-slab SGPR). Any mismatch → the slab
select is wrong (straddle or mis-keyed bound), not a numerics question.

V3. 1-chunk WT2 KLD prefill vs per-token (quality gate): run the Lloyd artifact
through the KLD harness the repo uses for quant promotion
(`crates/saddle-lab/examples/eval_hipfire_fullvocab.rs --ref <hfkldr.bin>
--max-chunks N`, header at `:34`) — one full prefill chunk through the changed
gate_up/qkv path vs the per-token decode path; KLD must not move (bit-exact
change ⇒ KLD delta exactly 0; nonzero delta means wrong-codebook rows, fail the
change). Cherry-pick an odd-N chunk too (exercises the untouched pad path,
§3, as a regression tripwire).

V4. `profile_prefill_qwen35` before/after (perf gate, the parent's numbers are
the baseline): `crates/saddle-lab/examples/profile_prefill_qwen35.rs`
(usage `--prefill N`, `:23-24`, `:48-71`) at `--prefill 512` (and 8192 for the
second bench pair), ≥3 fresh-process runs each side with prompt md5 + binary
md5 recorded and decoded output eyeballed per the backend-engineer evidence
rules (fresh processes: within-session A/B drifts 10–15% from DPM/thermal
state). Expect: gate_up s2bt8_lut 128.1 → ≈124–125 ms, qkv s2bt8_lut 14.2 →
≈13.3 ms, residual/qkvza unchanged; bench gap −3% → ≈−1%..parity. If gate_up
lands at parity but tok/s does not move, the bottleneck is outside this kernel
(§0 ceiling) — accept and stop.

V5. Validation authority: `docs/VALIDATION.md` selector tables (§"Retired
coherence-gate scripts": the `scripts/coherence-gate-*.sh` family is historical
reproduction only, never acceptance) plus the perf methodology at
`docs/methodology/perf-benchmarking.md`. Do not invent a new harness; V2–V4 use
only the claim-scoped tools named above.

## 5. Rejected alternatives (do not revisit without new data)

R1. Per-lane LUT copy into VGPRs (one `uint4` per lane, then single decode).
Rejected: burns VGPRs across the entire K-loop for all 32 lanes to save the
candidate VALU selects; qkvza already proved the SGPR select is sufficient and
at parity. Higher register pressure for zero additional gain.

R2. Hoisting LUT dwords into LDS / a constant buffer. Rejected: kernel args
already land in SGPRs for free; an LDS staging adds a load + barrier for
nothing. The bug was never LUT *delivery*, it was the per-lane *select*.

R3. Pre-decoding weights to E4M3 offline (store dequantized bytes). Rejected:
breaks the 136 B/row serialized MQ4v2 contract shared with qt=44 (same stride
— mis-routes already run at full speed and return noise, which is exactly why
`hipfire-dispatch` fail-closes Lloyd out of uniform keys at
`families/gemm.rs:251-276` and `families/fused_qkv.rs:77-90`); doubles VRAM for
the largest bucket; kills the shared quantizer path.

R4. Unifying gate+up (or q/k/v) codebooks to a single LUT. Rejected: changes
quantization quality (per-tensor Lloyd fits exist because they measure better);
saves nothing once the select is per-slab SGPR anyway.

R5. Padding `m` to 16 instead of fail-closed. Rejected: a straddled tile would
silently decode rows on the wrong codebook at full speed — the same silent-noise
failure class the dispatch guards exist to prevent. Loud error > wrong answer.

R6. Converting residual to the same scheme. Rejected: nothing to convert —
single LUT already decodes once (measured −1%, parity). Touching it risks the
`+=` epilogue (`:700-701`, `:727-728`) for zero ceiling.

R7. Rewriting the BT12/BT8/BT4 (SLABS=1) forms separately or leaving them on
candidate-eval. Rejected/included: A3–A4 convert them in the same edit because
they share the `#else` branches — no separate task, no divergent decode logic
between slab counts. N ≤ 256 odd shapes still route there via the pad path
(`:29415-29426`, `:30629-30640`).

R8. Fusing a Lloyd gate_up/qkv/qkvza FP8 kernel key into `hipfire-dispatch`
fused keys. Rejected: out of scope — prefill calls the `*_fp8_lloyd` launchers
directly from `prefill.rs` (`:4634` qkvza, `:5506`/`:6685` gate_up, `:5922`
qkv, `:190`/`:290`/`:9064` residual); fused keys remain decode-only by design
(`forward_slots.rs:358-360,374-376`).

## 6. What this plan is NOT doing

- No change to X-preparation (`prepare_mq4v2_fp8_x`), FWHT rotation, scales, or
  zero-point math.
- No change to decode GEMVs (`fused_*_mq4g256v2_lloyd` at
  `gemm.rs:32248-32450`, `GEMV_MQ4G256V2_LUT_SRC` family).
- No change to qkvza or residual kernels/launchers (done / at parity).
- No new env flags, no kill switches (bit-exact by construction; if V2 shows
  otherwise the change is wrong, not flaggable).
- No arch expansion beyond exact gfx1201 (launchers stay fail-closed).
- No pad-path, dispatch, registry, docs, or test-suite work beyond V1–V5.
