# Maple-Preview 20B-A1B — hipfire port (native ternary, arch 15) — design

**Status of every claim below is marked.** `measured` = read off the published
artifact by this branch's author, with the method stated. `published` = asserted
by the model card or `config.json`, not independently checked. `planned` =
intent, nothing implemented. Nothing here is `branch-implemented` yet.

**Branch:** `quant/maple-preview`, stacked on `quant/quality` (PR #599). Base
commit `673ea3ae0`. Push to `warpfront`, never the fork.

**Goal:** serve `deepgrove/maple-preview` — a 20B-A1B natively-ternary MoE — on
gfx1151, as a new arch crate (`hipfire-arch-maple`, **arch_id 15**), with the
ternary weights carried **losslessly**.

## Why this stacks on #599

Three things this port needs live only on `quant/quality`:

1. **The ternary formats themselves.** `TQ2G128` is qt40 there (renumbered from
   qt38 in #597). Master has no ternary at all.
2. **`quantize_mq2g256_lloyd_k3`** (`crates/hipfire-quantize/src/quant_mq.rs`) —
   the "MQ1.58 probe": a K=3 Lloyd codebook packed into the MQ2-Lloyd container
   with slot 3 duplicating slot 2, explicitly so it "runs on the existing
   MQ2G256Lloyd kernel with NO new kernel". This is arm B's container.
3. **The MoE kernel suite**, including the exact router and activation Maple
   needs (see "What already exists").

#599 also refactored the code this port must touch: `hipfire-quantize/src/main.rs`
is now split into `hfq.rs` / `pipeline.rs` / `pipeline_gguf.rs` / `quant_mq.rs`.
Work written against #597's layout would conflict structurally, not textually.
`quant/quality` contains `feat/ternary-bonsai-27b` outright (verified with
`git merge-base --is-ancestor`), so stacking on #597 would buy nothing and defer
the same merge.

This branch must not modify #599's qt40/qt41/qt44/qt45 internals, nor the
MQ2-Lloyd container layout. Ownership audit, same form #610 used:

```
git diff -G'TQ2G128|BQ1G128|MQ2G256Lloyd|qt.?40|qt.?41|72 B/group' 673ea3ae0 -- crates
```

Expected to show only *additive* dtype arms at review time, no edits to existing
packers or the 72 B/group layout.

## The model

`published` (`config.json`, `architectures: ["MapleForCausalLM"]`,
`model_type: maple`):

- 24 layers, hidden 2048, `intermediate_size 4096` (**unused** — `MapleMLP` is
  only ever constructed with `moe_intermediate_size`).
- GQA 16 heads / 4 KV, `head_dim 128`, `use_qk_norm: true`,
  `partial_rotary_factor 0.5`, θ=10000, `rms_norm_eps 1e-6`.
- **3:1 attention pattern**, `layer_types` = `[S,S,S,G] × 6`, `sliding_window 512`,
  and `nope_on_global_attention: true`.
- MoE on **every** layer: 256 experts, top-8, `moe_intermediate_size 512`,
  `num_shared_experts: 0`, `norm_topk_prob: true`, `router_dtype: fp32`,
  `moe_router_enable_expert_bias: false`.
- vocab 151936 (Qwen tokenizer), bos 151643, eos 151645,
  `tie_word_embeddings: false`, `max_position_embeddings 131072`.
- `quantize: true` and `preaffine: false` — **dead keys**, `measured`: neither
  string appears in `modeling_maple.py`, `configuration_maple.py` or `fa3.py`.
  There is no quantization path in the shipped modeling code to replicate; it is
  plain `nn.Linear` throughout. The ternary structure lives in the *values*, not
  in any code.

`measured` (HTTP range reads of the safetensors headers, no full download):
9 shards, **40.4 GB of BF16**, 1845 tensors in shard 1 alone. The card's
"5.31 GB checkpoint" is not what is published — **the repo is the dequantized
master.** Same pattern as the PrismML unpacked masters; see
`prismml_unpacked_masters_are_dequantized`.

Two details from `modeling_maple.py` that are load-bearing and easy to miss:

- **Clamped SwiGLU**: `silu(clamp(gate, max=7.0)) * clamp(up, min=-7.0, max=7.0)`.
- **RoPE only on sliding layers.** `apply_rotary_pos_emb` is called under
  `if self.sliding_window is not None`. Global layers get no positional signal
  at all. QK-norm is applied *before* RoPE, on both branches.
- Embedding tensor is `model.word_embeddings`, not `embed_tokens`.

## The key property: per-row ternary — `measured`

Every linear weight is exactly `{-s_r, 0, +s_r}` with **one bf16 scale per output
row**. Method: range-fetch the tensor bytes, reinterpret BF16→F32, and for each
row assert `unique(|w|)\{0}` has cardinality 1.

| tensor | rows checked | rows not ternary | nonzero frac | row-scale range |
|---|---:|---:|---:|---|
| `layers.0.mlp.experts.0.gate_proj` | 512 | **0** | 0.613 | 0.0193–0.0366 |
| `layers.0.self_attn.q_proj` | 1953 | **0** | 0.612 | 0.00873–0.137 |
| `layers.0.self_attn.o_proj` | 1953 | **0** | 0.607 | 0.0168–0.248 |

Not ternary, and to be carried at higher precision: `mlp.gate.weight` (router),
`word_embeddings`, `lm_head`, and all norms — each ~1000+ distinct values per
row and 100% nonzero.

**Consequence.** A container with a per-block scale or a per-block codebook can
reproduce these weights *exactly*, because every block within a row sees only
three distinct values. This is not a quantization problem. It is a packing
problem, and the acceptance bar is bit-exactness, not KLD.

All relevant K dimensions (2048 for gate/up/q/k/v/o, 512 for down) are divisible
by both 128 and 256, so neither container needs a tail case.

Parameter accounting (`measured` shapes, arithmetic ours): 815,792,128 ternary
params/layer × 24 = **19.58 B ternary**, plus 622 M for embeddings + lm_head
⇒ ~20.2 B total, ~0.87 B active/token excluding lm_head. Consistent with the
published "20B-A1B".

## Two container arms — B first

Both arms are **bit-exact reconstructions of the same weights**. This is
therefore not a quality experiment; it is a performance and engineering one,
with a free correctness oracle (below).

### Arm B (first) — MQ2-Lloyd container, K=3, no FWHT

Take `quantize_mq2g256_lloyd_k3` and drop the rotation. With only three distinct
values per 256-block, Lloyd converges on them at zero distortion; centroids round
to fp16, and Maple's bf16 `s_r` (0.0087–0.25 measured) is exactly representable
there. Slot 3 duplicates slot 2 and is never indexed.

- 72 B / 256 = 2.25 bpw ⇒ **5.51 GB** for the ternary part.
- **Zero new kernels.** Output DType stays `MQ2G256Lloyd`, which already has
  indexed gate_up, indexed down, batched k4/k8 and grouped GEMM across
  gfx1151 / gfx12 / gfx942 / gfx1030.

The obstacle: `quantize_mq2g256_lloyd_k3` calls `cpu_fwht_256`
**unconditionally**. FWHT is orthogonal so the *math* is transparent, but it
destroys the three-value structure, which is precisely what makes the K=3
codebook exact. Unrotated weights then require the runtime to skip
`rotate_x_mq` / `rotate_x_mq_batched` for these tensors.
`quantize_mq2g256_lloyd_no_fwht` already exists in `diagnostics.rs` as the
unrotated packer precedent.

### Spike B0 — RESOLVED, arm B is viable

**Kernel: rotation-agnostic** — `branch-implemented`, read from
`kernels/src/gemv_mq2g256_lloyd_moe_gate_up_indexed.hip`. The kernel is a pure
codebook lookup plus dot product; there is no FWHT inside it. Its header comment
("X must be FWHT-pre-rotated by the caller") confirms rotation is purely a
caller-side convention. Unrotated weights + unrotated x therefore compute the
correct dot product on the existing kernel, unmodified.

**Packing: value-exact on real weights** — `measured`. A no-FWHT K=3 packer
mirroring the tree's exact byte layout, run against published Maple tensors:

| tensor | max distinct / 256-block | max abs err | value-exact |
|---|---:|---:|---|
| `experts.0.gate_proj` | 3 | **0** | yes |
| `experts.0.down_proj` | 3 | **0** | yes |
| `self_attn.q_proj` | 3 | **0** | yes |
| `lm_head` (negative control) | 246 | 0.468 | **no**, as required |

The negative control confirms the test discriminates rather than passing
vacuously. Row scales also survive bf16→fp16→f32 unchanged (1953 scales,
0.00873–0.137), which is the precondition for fp16 codebook entries being exact.

**One real caveat: signed zeros.** ~19% of weights are `-0.0` in the source and
come back `+0.0`. Numerically identical (max err is exactly 0) and irrelevant to
a GEMV, but the round-trip is therefore **value-exact, not bitwise-identical**.
Both arms emit `+0.0` from a zero code, so they still agree with each other and
the A-vs-B differential oracle is unaffected.

**What B still needs (the one code change): a new DType tag.**
`needs_x_rot_local` is derived *purely from DType*
(`crates/hipfire-dispatch/src/families/moe.rs:247`) — `routed_gate_up_mq2lloyd`
forces rotation on, and the resolver's own comment warns that unrotated x into
rotated weights is "a silent garbage-output failure". So reusing `MQ2G256Lloyd`
as-is would wrongly rotate. Arm B adds a tag — say `MQ2G256LloydU` — that:

- shares the 72 B/group layout byte-for-byte, so **existing kernels bind unchanged**;
- joins `CODEBOOK_INDEXABLE` and the `routed_dtype_indexable` ORs;
- is **excluded** from the `needs_x_rot_local` ORs;
- satisfies the resolver's stated SAFETY INVARIANT — (a) an indexed gate_up GEMV
  arm, (b) an atomic self-combining down GEMV arm, (c) membership in
  `routed_down_self_combines`. Missing (c) "double-counts every MoE layer,
  silently". All three are satisfied by pointing at the identical kernels.

This is additive dispatch wiring plus a packer variant. **No new HIP.**

### Arm A — TQ2G128 (qt40)

The format #597/#599 already ship, and the one whose GEMV/GEMM kernels were just
optimised (1.77× decode, 4.1× prefill).

- 34 B / 128 = 2.125 bpw ⇒ **5.20 GB** for the ternary part. Slightly smaller
  than B.
- **But TQ2G128 has no MoE kernels.** Zero hits in `tables/moe_table.rs` and
  `families/moe.rs`, and `MIXED_SUPPORTED_TIERS = [MQ4G256, MQ6G256, ParoQ4G128]`.
  #597's ternary kernels are dense-only. Arm A must add indexed gate_up, indexed
  down, k8-batched variants and a grouped GEMM — mechanical (templates exist for
  six other dtypes) but not small.

Note the storage difference is ~0.3 GB and, per
`hipfire_lowbit_gemv_not_bandwidth_bound`, low-bit decode GEMVs are x-load/ALU
bound rather than weight-bandwidth bound — so bpw is **not** the figure of merit
here. Compare ms/launch.

### Why both — the differential oracle

If both arms are exact, they must emit **identical logits**. Any divergence is a
kernel bug, not a quality difference. That is a far sharper acceptance test than
KLD, and it validates arm A's new kernels against arm B's mature ones. Per
`feedback_assert_the_event_not_the_proxy`, every arm still gets a generation
smoke — a KLD number alone has lied on this exact class of work before.

Embeddings / lm_head / router / norms use the same precision policy in both arms
so they cancel out of the comparison.

## Shared work — arm-independent, and most of the project

**`hipfire-arch-maple`, arch_id 15.** Confirmed free: `MODEL_TYPE_TO_ARCH_ID`
(`crates/hipfire-runtime/src/arch_mapping.rs:28`) tops out at 14 for primaries;
22/23 are drafter sidecars. Add `("maple", 15)`.

**Attention template = `hipfire-arch-muse-glimmer` (14), not cohere2moe.**
Glimmer is already 3:1 sliding/full with **NoPE on the full layers**, head_dim
128, QK-norm and untied lm_head. Maple's deltas: partial rotary 0.5 (Glimmer has
none), plain pre-norm instead of sandwich norm, no logit softcap, no attention
gate, and MoE instead of dense. Cohere2MoE (12) is the fallback reference for
the sliding/global *plumbing* but differs more (parallel block, sigmoid router,
dense layer-0, tied embeddings).

**Convert path.** Stream safetensors shard-by-shard: download, verify per-row
ternary, pack, delete. Peak disk ~11 GB rather than 46 GB — worth it, `/` is at
100% and `/data` has ~101 GB. Must handle 18,432 expert tensors
(256 × 3 × 24). The per-row ternary check should be a **hard gate** that refuses
to write a non-ternary row rather than silently falling back to lossy
quantization — the failure mode we want is a loud refusal, following
`check_ternary_pack_health`'s precedent.

## What already exists (and must not be rewritten)

Maple's MoE decode pipeline is, kernel-for-kernel, already in `quant/quality`:

| Maple needs | Kernel present |
|---|---|
| softmax → top-8 router | `moe_router_softmax_topk_k8_wave64{,_exact}.hip` |
| `norm_topk_prob: true` | `moe_topk_renorm_k8.hip` |
| **clamped** SwiGLU | `moe_unscatter_silu_clamp_k8.hip` |
| scatter / permute / combine | `moe_scatter_*`, `moe_down_combine_*` |

k=8 throughout, which is Maple's `num_experts_per_tok` exactly. The gap is the
weight dtype, nothing else.

## Verification

Following #610's template: `scripts/coherence-gate-maple.sh` +
`_coherence_runner.py`, a `registry_gen.py` entry with a matching test.

1. **Value-exactness** — packed→dequantized weights compared against the source
   BF16 tensors. `max |err|` must be exactly 0, not "close". Bitwise equality is
   *not* the bar: signed zeros legitimately differ (B0). The assertion is
   "every differing word is a zero on both sides" — anything else fails.
   This is the arm gate.
2. **Differential** — arm A vs arm B logits on a fixed prompt set. Must be
   identical.
3. **Reference** — per-layer hidden-state cosine against `modeling_maple.py` on
   CPU, the method that localised the Bonsai double-norm bug to layer 0.
4. **Coherence smoke** — real generation, both arms. Non-negotiable regardless of
   what the numbers say.

### Measured — arm B, 2026-08-22, gfx1151

Convert of the full published checkpoint (9 shards, 18,651 tensors, structurally
validated against `model.safetensors.index.json` first):

```
18,528 ternary tensor(s), 19,579,011,072 weights, 5.13 GiB packed (2.250 bpw)
   123 high-precision tensor(s), 1.18 GiB
   per-tensor nonzero fraction 0.1714..0.6154
   -> maple-preview.hfq, 6.79 GB
```

**Zero "not ternary" refusals across every weight in the model.** The per-row
ternary claim is therefore verified on 100% of the checkpoint, not a sample —
the converter refuses rather than falling back, so a clean run IS the proof.

1. Value-exactness: `examples/maple_verify_pack` — 12 layers sampled across all
   9 shards, plus 20 tensors covering every projection type, all `max|err| = 0`,
   with every bitwise difference accounted for as a signed zero (counts EQUAL,
   nothing unexplained).
2. Differential vs arm A: not run — arm A is not implemented (out of scope).
3. Reference per-layer cosine: not run. Tooling is in place
   (`HIPFIRE_MAPLE_DUMP_HIDDEN` + `python3 -m tools.models.maple.compare_hidden`); it was not
   needed because the coherence gate passed on the first attempt, and it is the
   localisation tool for when it does not.
4. Coherence: **PASS.** Three prompts, coherent and technically correct, correct
   `<think>` framing, clean EOS. ~130-134 tok/s decode; 136 tok/s prefill once
   kernels are compiled (the harness prefilled one token per step at the time —
   superseded by the batched prefill in item 5).
5. **Batched prefill (`forward_batch`), 2026-08-22.** Now wired into
   `maple_coherence` and `MapleCarrier::bench_prefill`. Measured as a
   CONTROLLED A/B — both arms from one binary on one machine, back to back,
   selected by `HIPFIRE_MAPLE_PER_TOKEN_PREFILL`, rather than against a figure
   from an earlier session under unknown load:

   | 3,059-token prompt | prefill | prefill tok/s | decode tok/s |
   |---|---|---|---|
   | per-token (previous path) | 25.19 s | 121.4 | 118.3 |
   | **batched (`forward_batch`, chunk 256)** | **2.00 s** | **1,531.9** | 118.5 |

   **12.6x faster prefill**, decode unchanged (118.3 vs 118.5 — the change does
   not touch the decode path). Load average was 3.0-4.2 during both arms; the
   per-token arm reproduces the 24.37 s / 125.5 tok/s reference within 3%, which
   is what bounds the contention error on the ratio. Same output text on both
   arms.

   The **sliding window is proven live**, not assumed: with
   `HIPFIRE_MAPLE_FORCE_FULL_CAUSAL=1` on a 1,200-token prompt (past the 512
   window) parity cosine collapses from 0.996 to **-0.725**. Note that the
   argmax bar still passed 5/5 in that run — argmax alone would NOT catch a
   dropped window, which is why cosine stays a reported metric.

   Batched-vs-per-token argmax flips at small chunk sizes (B=1, B=17) were
   diagnosed with `maple_prefill_parity --near-tie` and are **F16-vs-F32
   near-ties, not a defect**: the median reference top1-top2 gap is 0.086 at
   flips vs 1.116 at matches (13x), the largest perturbation any flip required
   (0.451) is inside the measured deciding-margin noise (max 0.996), and the
   flip set is not reproducible run to run. B=256 — the shipped chunk size —
   shows no flips.

6. **Dedicated dense qt51 WMMA GEMM: not needed, closed out.** At 1,532 tok/s
   prefill the tile-id indirection and BLOCK_M padding are not worth a new
   kernel. Driving the grouped MoE kernel as a single-expert case keeps the
   "no new HIP kernels" property across the whole arch.

Two real defects were found by this verification rather than by inspection, both
silent-by-construction:

- The convert reordered tensors through a `BTreeMap` after spilling them
  sequentially, so `write_hfq` — which reads the spill in slice order — gave
  most tensors another tensor's bytes. Container valid, sizes right, weights
  wrong. Layer 0 decoded exactly, layers 2-23 gave `max|err| ~3412`.
- `run_moe_decode` FWHT-rotated the intermediate before the unrotated qt51 down
  GEMV (see Implementation findings).

Not done: a `registry/models.json` entry. It carries a published HF repo path
and measured serving figures for a downloadable artifact; the `.hfq` is local
only, so an entry now would point at a repo that does not exist.

## Open questions

- ~~**B0 (blocking):** is unrotated MQ2-Lloyd decodable by the existing
  kernels?~~ **Resolved: yes.** See "Spike B0" above.
- Precision policy for `word_embeddings` / `lm_head` (622 M params). BF16 costs
  1.24 GB; Q8 halves it. Deepgrove's 5.31 GB implies they quantize these; we
  need not match that to ship.
- Does the 131072 context interact badly with SWA-512 + NoPE-global on
  hipfire's KV paths? Glimmer runs SWA 2048; Maple's 512 is tighter.
- 256 experts × 24 layers of *tiny* (512×2048) expert matrices is an unusual
  shape for these kernels; per-launch overhead may dominate regardless of arm.
- **Follow-up: batched-prefill scratch is allocated eagerly, even decode-only.**
  `MapleState::new_with_max_seq` allocates all batch scratch at load time,
  sized from `MAPLE_PREFILL_MAX_B` (512), not from the chunk actually used
  (`MAPLE_PREFILL_CHUNK`, 256). Measured, for hidden 2048 / moe_inter 512 /
  k_top 8 / 256 experts:

  | group | size |
  |---|---|
  | grouped-MoE scratch (`moe_m_total_max` = 7,936) | **93.0 MiB** |
  | ├ `b_y_down` | 62.0 MiB |
  | ├ `b_y_gate_up` | 31.0 MiB |
  | batch scratch (`b_h`, `b_q/k/v`, `b_act`, F16 mirrors, …) | **39.0 MiB** |
  | **total** | **132.0 MiB** |

  A decode-only process pays all 132 MiB and uses none of it. Two independent
  reductions are available and neither was taken now (deliberately out of scope
  for the integration task): allocate lazily on first `forward_batch`, and size
  from the chunk rather than the cap — `moe_grouped_m_total_bound` at B=256
  gives 5,888 rows, cutting the grouped-MoE block from 93.0 to **69.0 MiB**.

## Implementation findings

Recorded during the arm-B build, because each one is a trap the next reader
would otherwise re-discover the hard way.

**The shared MoE executor cannot serve this arch.** `run_moe_decode`'s
gate-side unconditionally runs the shared-expert gate/up GEMVs, and Maple has
`num_shared_experts: 0`. Worse, its gate→down step used
`fused_silu_mul_rotate_mq_batched`, which FWHT-rotates the intermediate before
the down GEMV — correct for MQ2-Lloyd (qt19), and silent garbage for the
UNROTATED qt51. Widening the gate_up/down/self-combine arms to accept qt51 was
not sufficient; that step was left behind. Fixed in the executor (a named
`gate_down_skips_rotation` predicate, with a test requiring the rotated and
unrotated dtypes to disagree on both rotation flags), but `hipfire-arch-maple`
drives the indexed MQ2-Lloyd kernels directly regardless, because it also needs
the clamped SwiGLU and has no shared expert. This mirrors `cohere2moe`, which
is likewise a no-shared-expert MoE and likewise bypasses the executor.

**The RoPE convention is a three-way trap.** Maple needs pairs
`(i, i + n_rot/2)` *within* the first `n_rot = 64` dims, frequency denominator
`n_rot`. hipfire has three near-neighbours:

| kernel | pairing | denominator | fits Maple? |
|---|---|---|---|
| `rope_partial_halfsplit_f32` | `(i, i+n_rot/2)` | `n_rot` | **yes** |
| `rope_partial_halved_f32` | `(i, i+head_dim/2)` | `head_dim` | no — Gemma-4 proportional |
| `rope_partial_interleaved` (kernel) | `(2i, 2i+1)` | `n_rot` | no |

The correct one is reached through the Rust wrapper named
`rope_partial_interleaved_f32`, which dispatches the *half-split* kernel by
default (the name is stale; the interleaved kernel is behind
`HIPFIRE_ROPE_INTERLEAVED_LEGACY=1`). Picking by name gets this wrong.

**The clamped SwiGLU already exists.** `deepseek4_silu_mul_clamp_f32_batched`
is byte-for-byte Maple's math — gate capped from above only, `up` clamped both
ways — with the limit as a runtime parameter. DeepSeek passes 10.0, Maple
passes 7.0. No new HIP kernel; the "no new kernels" premise survived the whole
arch, not just the format.

**The HFQ metadata envelope is load-bearing.** The runtime reads the source
config from a `config` key and the tokenizer from a `tokenizer` STRING holding
tokenizer.json verbatim. Emitting the bare config.json converts cleanly and
then fails at load — after a 40 GB round trip.

**There is no BF16 embedding-lookup kernel.** `word_embeddings` is widened to
F32 at load (~620 MB over the BF16 bytes). That is the cheap answer to the
"precision policy" open question above for now; a Q8 embedding export would
halve it and needs no new code, since the Q8 lookup path is already wired.

**Packing verified against the real checkpoint.** The independent-packer parity
fixture (`python3 -m tools.models.maple.make_parity_fixture`, now committed) reproduces the
Rust packer byte-for-byte on `layers.0.mlp.experts.0.gate_proj` (1,048,576
weights, 61.3% nonzero) and `layers.0.self_attn.q_proj` (4,194,304 weights,
61.2% nonzero), both at 2.2500 bpw with `max|err| = 0`.

## Non-goals

- Reproducing Deepgrove's training or their 5.31 GB packing.
- True 1.58-bpw storage. Both arms are ~2.1–2.25 bpw containers; the `_k3`
  comment already flags dense ternary packing as a mechanical follow-up.
- Matching the published "200+ tok/s on M4" claim, which is `published` and
  unverified by us on unrelated hardware.
- Vision, drafter/MTP sidecars, EP/multi-GPU.
