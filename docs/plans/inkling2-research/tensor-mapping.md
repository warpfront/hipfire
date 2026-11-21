# Inkling-2-Mini tensor mapping: safetensors ↔ GGUF ↔ HFQ

Read-only research (2026-11-21). No `crates/*` code was changed for this doc.
Sources, all read verbatim:

- Checkpoint: `D:\hipfire-models\Inkling-2-Mini\` (`config.json`,
  `model.safetensors.index.json`, `hf_quant_config.json`,
  `model-00001-of-00001.safetensors` — shapes below measured with
  `safetensors.safe_open`, not inferred from names).
- GGUF F32: `D:\hipfire-models\Inkling-2-Mini\inkling2-f32.gguf` (shapes read
  from the GGUF header with a struct parser; 704 tensors).
- HFQ: `D:\hipfire-models\inkling2-mini-mq4.hfq` (13.9 GB, arch 24,
  **6790 tensors**; header/index parsed directly).
- Converter: `llama-inkling2/conversion/inkling2.py` (`modify_tensors`,
  `_dequant_expert_chunk`, `Inkling2LoopModel.modify_tensors`).
- GGUF schema: `llama-inkling2/gguf-py/gguf/{tensor_mapping.py,constants.py}`
  + `gguf_reader.py`.
- Runtime shapes: `llama-inkling2/src/models/{inkling2.cpp,inkling2-loop.cpp}`
  (`create_tensor` calls).
- hipfire side: `hipfire-arch-inkling2/src/{weights.rs,forward.rs,oracle.rs}`,
  `hipfire-quantize/src/{pipeline.rs,dequant.rs,model_filter.rs}`
  (worktree `hipfire-inkling2`).

## 0. Geometry (all verified against `config.json`)

`hidden_size` (`n_embd`) = **2304** (not 4096). `num_hidden_layers` = 24
physical, loop core 6–18 (`core_len` = 12) × 3 passes = 48 visits.
`num_attention_heads` = 24, `num_key_value_heads` = 8, `head_dim` = 128,
`d_rel` (`n_rel`) = 16, `rel_extent` = 1024. MoE: 128 experts, top-8,
`intermediate_size` = 1152, `route_scale` = 8, gate bias present.
`gate_rank` = 64. `sconv_kernel_size` = 4. Full-attention layers =
5, 11, 17, 23 (one per 6-block); the other 20 are SWA-1024.
`n_layers_per_attn_res_block` = 6. Vocab 201024, unpadded 200058.
`log_scaling_{floor,alpha}` = 8192 / 0.1. Clips 22.6274 / 8.0.

Derived widths used below: `eattn` = 24·128 = **3072**, `ekv` = 8·128 =
**1024**, `erel` = 24·16 = **384**, `2·n_embd` = **4608**,
`2·inter` = **2304**, `2·core_len` = **24**.

## 1. Axis-order conventions (read this first)

- **Safetensors / HFQ / hipfire forward**: torch row-major logical order.
  A linear `y = Wx` with `W [out, in]` is stored `[out, in]`.
- **GGUF file order is REVERSED**: `gguf-dump` / header dims list the fastest
  axis first. E.g. GGUF `blk.0.attn_q.weight dims=(2304,3072)` is torch
  logical `(3072,2304)` = `wq_du`. The llama.cpp `create_tensor` shapes
  (`{n_embd, n_head*n_embd_head}` etc.) are also file order (ggml `ne[]`).
- **gguf.cpp runtime consumes GGUF tensors with no transpose** (ggml native
  order), so GGUF-stored bytes are already correct for ggml and need
  un-reversing for torch row-major consumers.
- **HFQ stores torch-logical shapes and row-major bytes** (same as
  safetensors), except experts (split, see §4) and quant *values*.

## 2. Full table (per layer `L` = 0..23 unless noted)

### 2.1 Attention projections (all 24 layers, MQ4 in HFQ)

| safetensors (`model.llm.layers.L.attn.*`) | shape / dtype | GGUF (`blk.L.*.weight`, F32) | HFQ name / qt / shape |
|---|---|---|---|
| `wq_du.weight` | `[3072, 2304]` BF16 | `attn_q`, file `(2304,3072)` | same name, qt44, `[3072,2304]` |
| `wk_dv.weight` | `[1024, 2304]` BF16 | `attn_k`, file `(2304,1024)` | same, qt44, `[1024,2304]` |
| `wv_dv.weight` | `[1024, 2304]` BF16 | `attn_v`, file `(2304,1024)` | same, qt44, `[1024,2304]` |
| `wr_du.weight` | `[384, 2304]` BF16 | `attn_rel`, file `(2304,384)` | same, qt44, `[384,2304]` |
| `wo_ud.weight` | `[2304, 3072]` BF16 | `attn_output`, file `(3072,2304)` | same, qt44, `[2304,3072]` |
| `gate_attn.weight` | `[3072, 2304]` BF16 | `attn_gate`, file `(2304,3072)` | same, qt44, `[3072,2304]` |
| `attn_sink_proj.weight` | `[24, 2304]` BF16 | `attn_sink_proj`, file `(2304,24)` | same, qt1 (F16 keep), `[24,2304]` |
| `rel_logits_proj.proj` | `[16, 1024]` BF16 | `attn_rel_bias`, file `(16,1024)` | same, qt1, `[16,1024]` — **no transpose** (differs from GGUF, see §5.3) |
| `q_l2_clip.scale` | `[24, 128]` BF16 | `attn_q_clip`, file `(128,24)` | same, qt1, `[24,128]` |
| `rel_l2_clip.scale` | `[24, 16]` BF16 | `attn_rel_clip`, file `(16,24)` | same, qt1, `[24,16]` |

Semantics: `wq/wk/wv/wr` fan out from the (normed, controlled) Q/K/V inputs;
`wr` projects the *same Q input* to per-head relative dims (`[n_head·d_rel]`);
`attn_sink_proj` maps the Q input to one scalar per Q head; `gate_attn`
sigmoid-gates the attention output; `wo_ud` maps back to `n_embd`.
`q_l2_clip` is per-Q-head per-dim `[n_head, head_dim]`; `rel_l2_clip`
`[n_head, d_rel]`. The `.scale` suffix is a bare learned parameter, **not**
an FP8 sibling (see §6.2).

### 2.2 Norms (all 24 layers, F16 in HFQ)

SWA layers (20×: all except 5/11/17/23) carry `attn_norm.weight [2304]`
→ GGUF `attn_norm`. Full layers (4×) carry **three** input norms instead and
have **no** `attn_norm`:

| safetensors | shape | GGUF | HFQ |
|---|---|---|---|
| `q_input_norm.weight` (full only) | `[2304]` | `attn_q_inp_norm`, file `(2304,)` | qt1, same shape |
| `k_input_norm.weight` (full only) | `[2304]` | `attn_k_inp_norm` | qt1 |
| `v_input_norm.weight` (full only) | `[2304]` | `attn_v_inp_norm` | qt1 |
| `attn_norm.weight` (SWA only) | `[2304]` | `attn_norm` | qt1 |
| `mlp_norm.weight` (all 24) | `[2304]` | `ffn_norm` | qt1 |

`weights.rs` validation relies on exactly this split (SWA→`attn_norm`,
full→`q_input_norm`).

### 2.3 Short convolutions (all 24 layers, F16 in HFQ)

SWA layers use plain static convs; full layers (5/11/17/23) use
`static_w` + input-dependent `weight_proj` taps (dynamic path deferred in the
hipfire CPU forward — static weights only so far).

| safetensors | shape / dtype | GGUF (F32) | HFQ |
|---|---|---|---|
| `attn.k_sconv.weight` (SWA, 20×) | `[1024, 1, 4]` BF16 | `attn_k_conv`, file `(4,1024)` (squeezed to torch `(1024,4)`) | qt1, `[1024,1,4]` verbatim |
| `attn.v_sconv.weight` (SWA, 20×) | `[1024, 1, 4]` BF16 | `attn_v_conv`, file `(4,1024)` | qt1, verbatim |
| `attn.k_sconv.static_w` (full, 4×) | `[4, 1024]` BF16 | `attn_k_conv`, file `(4,1024)` = `flip(0).T` of checkpoint (torch `(1024,4)`) | qt1, `[4,1024]` verbatim (**no** flip at quantize time) |
| `attn.v_sconv.static_w` (full, 4×) | `[4, 1024]` BF16 | `attn_v_conv`, same transform | qt1, verbatim |
| `attn.k_sconv.weight_proj` (full, 4×) | `[32, 2304]` BF16 | `attn_k_conv_proj`, file `(2304,32)` (torch `(32,2304)`, **no transpose**) | qt1, `[32,2304]` |
| `attn.v_sconv.weight_proj` (full, 4×) | `[32, 2304]` BF16 | `attn_v_conv_proj`, file `(2304,32)` | qt1 |
| `attn_sconv.weight` (all 24) | `[2304, 1, 4]` BF16 | `attn_post_conv`, file `(4,2304)` (torch `(2304,4)`) | qt1, `[2304,1,4]` verbatim |
| `mlp_sconv.weight` (all 24) | `[2304, 1, 4]` BF16 | `ffn_post_conv`, file `(4,2304)` | qt1, verbatim |

`weight_proj` rows = `n_conv·n_head_kv` = 4·8 = 32 (one 4-tap kernel per KV
head). `load_conv` in `forward.rs` transposes 3-D `[ch,1,k]`→`[k,ch]` and
un-flips 2-D `static_w` along the kernel axis at load.

### 2.4 Controllers (core layers 6–17 only: 12 layers × 12 tensors = 144)

Stored checkpoint layout is **`down [rank, 2·n_embd] = [64, 4608]`**,
**`gain/forget [n_embd, rank] = [2304, 64]`** — the plan's old
"`down [2·n_embd, rank]`" phrasing is ggml file order, not storage order.

| safetensors (`...controllers.{attn,k,v,mlp}.{down,forget,gain}_weight`) | shape | GGUF (`blk.6–17.loop_{q,k,v,ffn}_ctrl_{down,gain,forget}`, F32) | HFQ |
|---|---|---|---|
| `attn.down_weight` | `[64, 4608]` BF16 | file `(4608,64)` (= torch `(64,4608)`, no transpose) | qt1, `[64,4608]` |
| `attn.gain_weight` / `attn.forget_weight` | `[2304, 64]` BF16 | file `(64,2304)` | qt1, `[2304,64]` |
| `k/v/mlp` × same three | same shapes | `loop_{k,v,ffn}_ctrl_*`, same pattern | qt1, same shapes |

Math (`oracle::control`, eps 1e-6): `gate = down·[RMS(cur);RMS(msg)]`
(`[rank]`), `out = cur − tanh(forget·gate)·cur + tanh(gain·gate)·msg`,
i.e. `matvec(down, gate_in, rank, 2·n_embd)` expects row-major `[64, 4608]`
— exactly the stored order. No transpose anywhere on the HFQ/Raw path.

### 2.5 Cross-pass message queries (globals, F32 in checkpoint → F16 in HFQ)

| safetensors | shape | GGUF | HFQ |
|---|---|---|---|
| `model.llm.middle_cycle_cross_pass_queries` | `[24, 2304]` F32 | **exploded** to per-core-layer `blk.6–17.loop_msg_query`, file `(2304,4)` (torch `(4,2304)`), row order `[q2i, kv2i, kv2i+1, q2i+1]` (see §5.5) | qt1, `[24,2304]` verbatim (NOT exploded) |
| `model.llm.middle_cycle_cross_pass_kv_queries` | `[24, 2304]` F32 | same explosion (kv rows of the same stack) | qt1, `[24,2304]` verbatim |

`forward.rs::pack_message_queries` reassembles the two `[24, 2304]` HFQ
globals; the `GgufF32` shim reassembles them from the 12 GGUF stacks.

### 2.6 AttnRes queries (F32 in checkpoint → F16 in HFQ)

| safetensors | shape | GGUF | HFQ |
|---|---|---|---|
| `model.llm.attn_res.attn_queries.{i}`, i ∈ 0..=22 except {4,10,16,22} (19×) | `[2304]` F32 | `blk.{i+1}.attn_res_query`, file `(2304,)` — **index shifts by one** (§5.4) | same name, qt1, `[2304]` |
| `attn_queries.{4,10,16,22}` (feed full layers 5,11,17,23) | `[3, 2304]` F32 | `blk.{5,11,17,23}.attn_res_query`, file `(2304,3)` (torch `(3,2304)`) | same name, qt1, `[3,2304]` |
| `model.llm.attn_res.mlp_queries.{i}`, i = 0..23 (24×) | `[2304]` F32 | `blk.{i}.ffn_res_query`, file `(2304,)` — **no shift** | same name, qt1, `[2304]` |
| `model.llm.attn_res.output_query` | `[2304]` F32 | `output_res_query`, file `(2304,)` | same name, qt1, `[2304]` |

Layer 0 has no `attn_res_query` in GGUF (nothing maps to `blk.0`);
`forward.rs` reads `attn_queries.{phys}` with an `if let Ok` fallback to
unmixed inputs — see §6.4.

### 2.7 MoE (all 24 layers; NVFP4 0–19, MXFP8 20–23)

Router (all layers, shared by experts):

| safetensors (`...mlp.gate.*`) | shape | GGUF | HFQ |
|---|---|---|---|
| `gate.weight` | `[128, 2304]` BF16 | `ffn_gate_inp`, file `(2304,128)` (torch `(128,2304)`) | qt44, `[128,2304]` |
| `gate.bias` (aka `e_score_correction_bias` in GGUF mapping) | `[128]` F32 | `exp_probs_b`, file `(128,)` (TENSOR_NOT_REQUIRED upstream, present in F32 file) | qt1, `[128]` |

Experts — packed in checkpoint, **split per expert** in GGUF and HFQ:

| safetensors (`...mlp.experts.*`, E=128) | stored shape | GGUF (F32, grouped `[gate…; up…]`) | HFQ (split, grouped) |
|---|---|---|---|
| `w13_weight` + `_scale` (+`_amax`, `_outer_scale_denominator` if NVFP4) | NVFP4 L0–19: weight `[128, 2304, 1152]` u8-packed (2 nibbles/col), scale `[128, 2304, 144]` E4M3, amax `[128]` F32, denom `[1]` F32. MXFP8 L20–23: weight `[128, 2304, 2304]` E4M3, scale `[128, 2304, 72]` u8 (UE8M0) | `ffn_gate_up_exps`, file `(2304,2304,128)` (torch `(128,2304,2304)` = `[E, 2·inter, hidden]`) | `...mlp.experts.{E}.gate_up_proj.weight`, qt13 (MQ4G256), `[2304, 2304]` = `[2·inter, hidden]`, gate rows first |
| `w2_weight` + `_scale` (+ amax/denom if NVFP4) | NVFP4: weight `[128, 2304, 576]` u8, scale `[128, 2304, 72]`; MXFP8: weight `[128, 2304, 1152]` E4M3, scale `[128, 2304, 36]` | `ffn_down_exps`, file `(1152,2304,128)` (torch `(128,2304,1152)` = `[E, hidden, inter]`) | `...mlp.experts.{E}.down_proj.weight`, qt13, `[2304, 1152]` = `[hidden, inter]` |

Stored-block geometry (validates `dequant.rs`): NVFP4 logical
`w13 [E, 2·inter=2304, hidden=2304]` packed to `cols/2 = 1152`, scale
`cols/16 = 144` → block `(1,16)`; `w2 [E, hidden=2304, inter=1152]`
packed to 576, scale 72 → block `(1,16)`. MXFP8 `w13 [E,2304,2304]`,
scale 72 = `cols/32` → block `(1,32)`; `w2 [E,2304,1152]`, scale 36 →
block `(1,32)`. The `hf_quant_config.json` MXFP8 `weight_block_shape
[32,32]` conflicts with the stored `(1,32)` block — the converter warns and
trusts the stored shapes; so do we.

The 8 companions per NVFP4 expert tensor (`_scale`, `_amax`,
`_outer_scale_denominator` × w13/w2) are consumed at quantize time and
absent from HFQ; MXFP8 has only `_scale` (an `_amax`/denominator on an MXFP8
tensor is refused at ingest).

### 2.8 Embed / norm / head

| safetensors | shape | GGUF (F32) | HFQ |
|---|---|---|---|
| `model.llm.embed.weight` | `[201024, 2304]` BF16 | `token_embd`, file `(2304,201024)` | qt44, `[201024, 2304]` |
| `model.llm.embed_norm.weight` | `[2304]` BF16 | `token_embd_norm`, file `(2304,)` | qt1, `[2304]` |
| `model.llm.norm.weight` | `[2304]` BF16 | `output_norm` | qt1 |
| `model.llm.unembed.weight` | `[201024, 2304]` BF16 | `output`, file `(2304,201024)` | qt1 in `hf_quant_config` cast group… **but qt44 in the shipped HFQ** (`[201024,2304]`). The quantizer quantizes it; the runtime must slice rows `0..200058` (GGUF `vocab_size_unpadded`, `get_vocab_base` truncates the same way) |

## 3. HFQ census reconciliation (6790 tensors)

- Per-expert: 24 layers × 128 × 2 = **6144**, qt13 (MQ4G256),
  `...experts.{E}.{gate_up_proj,down_proj}.weight`.
- Quantized dense: 24 × {wq,wk,wv,wr,wo,gate} = 144 + 24 routers +
  embed + unembed = **170**, qt44 (MQ4G256V2).
- F16/small (qt1, **476**): 24 sink + 20+4 k-conv + 20+4 v-conv + 8
  proj + 24 rel-tables + 20 attn_norm + 24 attn_sconv + 4×3 input norms +
  24 mlp_norm + 24 mlp_sconv + 48 clips + 24 gate.bias + 144 controllers +
  48 attn_res (23 + 24 + 1) + 2 cross-pass + embed_norm + norm.
- 6144 + 170 + 476 = **6790** ✓. Vision + `mtp.safetensors` excluded
  (Phase-1 refusal).

## 4. Ingest transforms (checkpoint → GGUF vs checkpoint → HFQ)

| # | Transform | GGUF (`modify_tensors`) | HFQ (`pipeline.rs`) |
|---|---|---|---|
| 1 | Expert dequant (NVFP4: low nibble first, E2M1 → ×E4M3 scale → ×`amax/denominator`; MXFP8: E4M3 → ×`2^(u8−127)` per 32 cols) | `_dequant_expert_chunk`, grouped output | `dequantize_nvfp4_amax_denominator` / `dequantize_mxfp8_e8m0_strict` per 2-D expert slice, same math; strict NaN/finiteness refusal |
| 2 | w13 gate/up interleave → group | packed path emits grouped directly; F32 path `reshape(E,inter,2,hidden).transpose(1,2).reshape` (interleaved→grouped) | `grouped[j]=f32[2j]`, `grouped[inter+j]=f32[2j+1]` (stored rows are `[g0,u0,g1,u1,…]`) |
| 3 | Per-expert split | stacked 3-D `gate_up_exps/down_exps`, expert on GGUF axis 2 (file order) | per-expert 2-D tensors `experts.{E}.gate_up_proj/down_proj.weight` (+`.weight` suffix added) |
| 4 | `static_w` convs | `flip(0).T`: `[4,1024]`→torch `(1024,4)` | **verbatim** `[4,1024]`; `forward.rs::load_conv` un-flips at load |
| 5 | `rel_logits_proj.proj` | `.T`: `(16,1024)`→torch `(1024,16)` | **verbatim** `(16,1024)`; forward indexes `rel_table[rr·1024]` i.e. `[n_rel, n_extent]` directly |
| 6 | `attn_res.attn_queries.{i}` | → `blk.{i+1}.attn_res_query` (shift) | **verbatim names** (no shift, no `.weight` suffix) |
| 7 | `mlp.gate.bias` | renamed `e_score_correction_bias` → `exp_probs_b` | **verbatim** `mlp.gate.bias` |
| 8 | Cross-pass globals | exploded to 12× `loop_msg_query [n_embd,4]` | **verbatim** two `[24,2304]` globals |
| 9 | 3-D conv squeeze | `squeeze(1)`: `[C,1,K]`→torch `(C,K)` | **verbatim** 3-D `[C,1,K]`; `load_conv` transposes at load |
| 10 | Convs/clips/queries/rel-bias/scales/controllers | forced F32 in GGUF | keep-F16 (`is_inkling2_keep_f16`: `.controllers.`, `res_query`, `attn_rel_bias`, `q_l2_clip`, `rel_l2_clip`, `attn_sink_proj`, `conv`, `scale`) — plus 1-D `gate.bias` lands qt1 |

## 5. Cross-pass packing detail

Checkpoint globals are `[2·core_len, n_embd] = [24, 2304]`. The converter
requires `q.shape[0] == 2·(end−start)` and `kv == q.shape`, then per core
layer `i` (global layer `start+i`) stacks torch rows
`[q[2i], kv[2i], kv[2i+1], q[2i+1]]` → `loop_msg_query [4, n_embd]`
(Q/K/V/FFN). `forward.rs::pack_message_queries` inverts this from the two
HFQ globals.

## 6. Known traps (all verified in the sources above)

1. **GGUF dims read backwards.** `gguf-dump` order is `(fastest … slowest)`;
   torch order is the reverse. `attn_q_clip` file `(128,24)` = torch
   `[24,128]`. Never compare dump dims to checkpoint shapes without
   reversing.
2. **`gguf_to_f32_bin.py` dims/data mismatch.** `ReaderTensor.shape` is file
   order but `.data` is reshaped to *reversed* (torch-logical) order, so the
   `.bin` carries file-order dim labels with torch-order bytes. The
   `GgufF32` shim in `weights.rs` is only correct where it ignores the
   labels: plain linears (explicit `matvec` dims), `loop_msg_query`
   reassembly, clips. It is **wrong** wherever it trusts the labels —
   controllers (spurious transpose → scramble, the `4608×64 vs 64×4608`
   bug), experts (`(i·cols+j)·nexp+e` indexes a torch-`[E,…]` buffer as
   expert-last → scramble), and full-layer `attn_res_query`
   (`shape[0]` = 2304 taken as `query_rows` instead of 3). The HFQ/Raw paths
   need no transpose anywhere. Fix the shim by reversing bin dims at read
   (or re-exporting with torch-order labels); do not "fix" the forward.
3. **Controller orientation.** Stored `down` is `[64, 4608]`, gain/forget
   `[2304, 64]` — ggml file order (`{4608,64}`/`{64,2304}`) is the transpose.
   `oracle::control` wants row-major `[rank, 2·n_embd]`; pass through.
4. **AttnRes off-by-one (verify before trusting).** GGUF puts query `i` on
   block `i+1` (`blk.0` has none); `GgufF32` maps `attn_queries.{i}` →
   `blk.{i+1}`. But `decode_visit` reads `attn_queries.{phys}` for the
   current visit — giving block 0 block 1's query and block 23 nothing
   (silent `if let Ok` fallback). The `mlp_queries` side is unshifted in all
   three representations and correct. Reconcile the attn side (expected:
   `phys == 0` → none, else `attn_queries.{phys−1}`) and re-run parity.
5. **`rel_logits_proj` transpose asymmetry.** GGUF runtime sees torch
   `(1024,16)` (converter `.T`); HFQ/forward sees `(16,1024)` indexed
   `[rr, extent]`. Both are self-consistent; do not "harmonize".
6. **`.scale` sibling rule.** `q/rel_l2_clip.scale` are bare parameters with
   no `.weight` stem. The quantizer's generic "FP8 sibling of a `.weight`"
   rule must check stem existence first (already fixed — the 48 clips are in
   the artifact; a regression drops them silently).
7. **`static_w` flip lives in different places.** GGUF bakes `flip(0).T` at
   conversion; HFQ stores verbatim and `load_conv` un-flips at load. A
   forward that reads HFQ `static_w` as `(1024,4)` without un-flipping, or
   double-flips, breaks full-layer convs (layers 5/11/17/23 only).
8. **Expert layout.** Stored `w13` rows interleave `[g0,u0,…]`; GGUF and HFQ
   both store grouped `[g…;u…]`. `pipeline.rs`, `RawWeights::f32` and the
   converter agree; the qt44↔GGUF 0.995 per-weight correlation supports it.
9. **`block_count` = 24, never 48.** GGUF stamps 24 physical blocks + loop
   keys; visits alias weights. An artifact stamped 48 (or an unroll) doubles
   VRAM and breaks every imatrix/calibration key.
10. **Forward scaffolding gaps (not mapping bugs, but they dominate parity).**
    Single-token `decode_visit` gathers `rel_table` at distance 0 only
    (`rel_table[rr·n_extent]`); full-layer dynamic conv taps are deferred
    (static weights); these plus the uniform-MQ4 quality gap (SQNR ~20 dB,
    §“Numeric oracle” in the implementation log) explain residual divergence
    after the mapping is right.
