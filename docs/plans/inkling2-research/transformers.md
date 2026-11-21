# Inkling-2 research: HuggingFace transformers oracle (transformers)

Scope: what the hipfire CPU reference forward needs from the HF transformers
reference implementation. Primary math oracle is
`src/transformers/models/inkling2/modeling_inkling2.py` (+
`configuration_inkling2.py`) on **main, fetched 2026-11-21** (post-2026-11-20),
**plus** the open PR **huggingface/transformers#51050 "Add Inkling2 Mini"**
(branch `add-inkling2-mini`, head `a10ebc3`, 69 commits, created 2026-11-20 by
`guarin`, still open) which adds `src/transformers/models/inkling2_loop/` and
refactors `modeling_inkling2.py`. The main-branch file is **Small-only**
(48 flat layers, no loop); everything loop-specific (controllers, messages,
passes) lives only in the PR branch.

Companion report (other seat): `inkling2-research-transformers-ninfer-all.md`
— transformers Small operators + ninfer-all loop port. This file is the
transformers-side record with verbatim excerpts.

Cross-checked against (no re-derivation, per brief):
- llama.cpp @ merge commit `29b52d398` (`C:\...\llama-inkling2\src\models\inkling2*.cpp`,
  `conversion/inkling2.py`) — the verified oracle.
- Handoff log `docs\plans\local\2026-11-21-inkling2-implementation-log.md`
  (already-verified hipfire oracle ops: `rmsnorm, clip, tau, matvec, control,
  mix, convolve_static, relative_bias, situ, sigmoid_topk, softcap`).

## 0. Which source to trust for what

- Attention / conv / clip / tau / gate / sink / softcap / MoE core math:
  transformers `modeling_inkling2.py` == llama.cpp `inkling2.cpp` (modulo the
  renorm wording in §5 — they agree numerically, see below).
- Loop schedule / prelude / bank / controllers / message packing: **only**
  PR #51050's `modular_inkling2_loop.py` + llama.cpp `inkling2-loop.cpp`.
  Main-branch transformers has `block_states` accumulation but no passes.
- Weight-name mapping checkpoint→transformers: PR #51050 `conversion_mapping.py`
  (quoted in §3/§4).
- Secondary (structure only, not quoted as math): vLLM PR
  `vllm-project/vllm#69028`, SGLang PR `sgl-project/sglang#51926`
  (see §7).

## 1. Attention

### 1a. Clips (main-branch `modeling_inkling2.py`)

```python
class Inkling2L2Clip(nn.Module):
    """Scales each head vector down to `threshold` when its L2 norm is larger; shorter vectors are unchanged."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        norm = torch.linalg.vector_norm(hidden_states, dim=-1, keepdim=True)
        return (hidden_states * (self.threshold / (norm + 1e-6)).clamp(max=1.0)).to(input_dtype)


class Inkling2ScaledL2Clip(nn.Module):
    """`Inkling2L2Clip` applied after a learned scale."""

    def __init__(self, num_heads: int, head_dim: int, threshold: float):
        self.threshold = threshold
        self.scale = nn.Parameter(torch.ones(num_heads, head_dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float() * self.scale.float()
        norm = torch.linalg.vector_norm(hidden_states, dim=-1, keepdim=True)
        return (hidden_states * (self.threshold / (norm + 1e-6)).clamp(max=1.0)).to(input_dtype)
```

Math: `y = x * min(1, threshold / (||x|| + 1e-6))`, all in fp32.
Placement (both `Inkling2SlidingAttention` and `Inkling2FullAttention`):
Q = scaled-clip, K = scaleless clip, R = scaled-clip, V = unweighted RMSNorm
(`Inkling2UnweightedRMSNorm`, eps = `rms_norm_eps`). Thresholds:
`qk_l2_clip_threshold = 22.627416997969522`, `rel_l2_clip_threshold = 8.0`.

### 1b. Relative-bias projection/gather (main-branch)

```python
class Inkling2RelativeLogits(nn.Module):
    def __init__(self, d_rel: int, rel_extent: int):
        self.rel_extent = rel_extent
        self.proj = nn.Parameter(torch.empty(d_rel, rel_extent))

    def forward(self, relative_states, query_positions, key_positions) -> torch.Tensor:
        # relative_states: [batch, q_len, num_heads, d_rel] -> bias: [batch, num_heads, q_len, kv_len]
        rel_logits = (relative_states @ self.proj).transpose(1, 2)
        distance = (query_positions[:, None] - key_positions[None, :])[None, None, :, :]
        gather_index = distance.clamp(0, self.rel_extent - 1).expand(*rel_logits.shape[:2], -1, -1)
        position_bias = rel_logits.gather(-1, gather_index)
        return position_bias.masked_fill((distance < 0) | (distance >= self.rel_extent), 0.0)
```

Math: per-head `rel = R @ proj` → `[B, H, q, rel_extent]`; bias =
`rel[..., clamp(q-k, 0, extent-1)]`, zeroed outside `0 <= d < rel_extent`.
Config: `d_rel = 16`, `rel_extent = 1024` (full layers; sliding layers use
`sliding_window_size = 1024` as the extent). PR #51050 refactors this into
`Inkling2RelativePositionBias` (linear + norm + proj in one module, optional
log-scaling of R) and adds `get_relative_distance()`; math unchanged.

### 1c. Sink subtraction + output gate + softcap (main-branch)

```python
# eager_attention_forward:
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if position_bias is not None:
        attn_weights = attn_weights + position_bias
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    if sinks is not None:
        # The sink is one extra logit per query with a zero value: it only absorbs probability mass
        combined_logits = torch.cat([attn_weights, sinks], dim=-1).float()
        attn_weights = F.softmax(combined_logits, dim=-1)[..., :-1].to(query.dtype)
```

`sink_proj: hidden → num_heads` (no bias); sink is one extra logit column per
query, softmaxed jointly, then dropped (zero-value column). `scaling =
head_dim**-0.5`, head_dim 128 both layer types. Output gate (both attentions):

```python
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = (attn_output.float() * torch.sigmoid(gate.float())).to(attn_output.dtype)
        attn_output = self.o_proj(attn_output)
```

(PR #51050 moves this into `Inkling2GatedOutputProjection`: gate =
`sigmoid(gate_proj(h))` computed in fp32, multiply, then `linear`. Same math.)
Final-logit softcap (`Inkling2ForCausalLM` / conditional-generation):

```python
class Inkling2Softcap(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.tanh(hidden_states / self.softcap) * self.softcap
# ...
        logits = self.lm_head(hidden_states).float()
        logits = self.logit_softcap(logits)          # final_logit_softcapping = 30.0
        logits = logits[..., : self.config.unpadded_vocab_size]   # 200058 of 201024
```

### 1d. Tau (log-scale, FULL layers only — main-branch)

```python
        # log scaling of the queries and relative states, in fp32 as in the original
        tau = 1.0 + self.config.log_scaling_alpha * torch.log(
            ((kwargs["position_ids"] + 1).float() / self.config.log_scaling_n_floor).clamp(min=1.0)
        )
        tau = tau[..., None, None]

        query_states = self.q_l2_clip(query_states.view(hidden_shape))
        query_states = (query_states.float() * tau).to(query_states.dtype).transpose(1, 2)
        ...
        relative_states = self.rel_l2_clip(relative_states.view(*input_shape, self.num_heads, -1))
        relative_states = (relative_states.float() * tau).to(relative_states.dtype)
```

`tau = 1 + 0.1 * log(max((pos+1)/floor, 1))`, multiplies clipped Q **and**
clipped R, in fp32. Sliding layers: no tau. Floors: Small 128000, Mini 8192
(`Inkling2LoopTextConfig.log_scaling_n_floor = 8192`). PR #51050 factors this
into `Inkling2LogScale` / `Inkling2LogScaledL2Clip`; math unchanged.
llama.cpp equivalent (quoted from verified source, `inkling2.cpp`
`build_inputs`): log-scaled positions multiply q and r of full layers.

## 2. Short convs (unified recurrent state)

Static (`Inkling2ShortConvolution`, k/v of sliding layers AND attn-out/FFN-out
of every layer — conv_idx 0..3):

```python
        hidden_states = hidden_states.float()

        residual = hidden_states
        hidden_states = apply_mask_to_padding_states(hidden_states, conv_mask)
        ...
        hidden_states = hidden_states.transpose(1, 2)
        # Different from Inkling which has no output scale
        hidden_states = (hidden_states + residual) / math.sqrt(self.conv_kernel_size + 1)
        return hidden_states.to(dtype=input_dtype)
```

Dynamic (`Inkling2DynamicShortConvolution`, k/v of FULL layers only):

```python
        # tap index is the backward distance: `static_weight[0]` weights the current token
        self.static_weight = nn.Parameter(torch.zeros(conv_kernel_size, hidden_size))
        self.weight_proj = nn.Linear(input_size, num_heads * conv_kernel_size, bias=False)
# ...
        weights = self.static_weight.float().T.view(self.num_heads, -1, self.conv_kernel_size) + dynamic_weights
        ...
        # windows[..., lag] is the token `lag` steps back
        windows = context.unfold(-1, self.conv_kernel_size, 1).flip(-1).transpose(1, 2)
        windows = windows.reshape(batch_size, seq_len, self.num_heads, -1, self.conv_kernel_size)
        hidden_states = (windows * weights).sum(-1).reshape(batch_size, seq_len, hidden_size)
        hidden_states = (hidden_states + residual) / math.sqrt(2 * self.conv_kernel_size + 1)
```

Per-operator summary: depthwise causal conv, kernel 4, no bias, no activation;
static taps are learned per-channel weights; dynamic taps add per-token,
per-KV-head projected weights (`weight_proj(dynamic_input)`, same hidden
input that feeds k/v proj). Post-scale `1/sqrt(n_conv+1)` static,
`1/sqrt(2*n_conv+1)` dynamic. Padding tokens zeroed before conv
(`apply_mask_to_padding_states`), residual added AFTER conv. Matches verified
llama.cpp `convolve()`: `out / sqrt((dynamic ? 2 : 1)*n_conv + 1)`.
PR #51050 adds `cache_offset` (per-pass cache layers, §6) and a `seq_idx`
same-sequence mask to the dynamic path; math unchanged.

## 3. Controllers (message gates — PR #51050 `modular_inkling2_loop.py` ONLY)

Main-branch transformers has no controllers. Verbatim from the PR branch:

```python
class Inkling2LoopMessageGate(nn.Module):
    """Mixes hidden states and message as  `(1 - forget_gate) * hidden_states + message_gate * message`"""

    def __init__(self, config: Inkling2LoopTextConfig):
        super().__init__()
        self.norm = Inkling2LoopUnweightedRMSNorm(eps=config.rms_norm_eps)
        self.down_proj = nn.Linear(2 * config.hidden_size, config.loop_gate_rank, bias=False)
        self.forget_gate_proj = nn.Linear(config.loop_gate_rank, config.hidden_size, bias=False, dtype=torch.float32)
        self.message_gate_proj = nn.Linear(config.loop_gate_rank, config.hidden_size, bias=False, dtype=torch.float32)

    def forward(self, hidden_states: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        hidden_states, message = hidden_states.float(), message.float()
        gate_hidden_states = torch.cat([self.norm(hidden_states), self.norm(message)], dim=-1)
        gate_hidden_states = self.down_proj(gate_hidden_states.to(self.down_proj.weight.dtype)).float()

        forget_gate = torch.tanh(self.forget_gate_proj(gate_hidden_states))
        message_gate = torch.tanh(self.message_gate_proj(gate_hidden_states))
        return (1 - forget_gate) * hidden_states + message_gate * message
```

Math: `gate = down(RMSNorm(cur) ‖ RMSNorm(msg))` (rank 64), `out =
(1−tanh(F·gate))·cur + tanh(G·gate)·msg`, gate projections in fp32.
Exactly the brief's formula. llama.cpp identical (verified source,
`inkling2.cpp::control`):

```cpp
ggml_tensor * gate = ggml_concat(ctx0, ggml_rms_norm(ctx0, cur, 1e-6f), ggml_rms_norm(ctx0, message, 1e-6f), 0);
gate = build_lora_mm(loop.ctrl_down[ctrl], gate);
ggml_tensor * gain   = ggml_tanh(ctx0, build_lora_mm(loop.ctrl_gain[ctrl],   gate));
ggml_tensor * forget = ggml_tanh(ctx0, build_lora_mm(loop.ctrl_forget[ctrl], gate));
return ggml_add(ctx0, ggml_sub(ctx0, cur, ggml_mul(ctx0, forget, cur)), ggml_mul(ctx0, gain, message));
```

Four gates per loop layer (q/k/v/mlp). SWA loop layer applies gates to
Q, K, V inputs (each gated independently, then shared `input_layernorm`);
full loop layer gates pre-normed q/k/v streams (`attention_input` outputs).
MLP input gated in both. Gate 0 of the merged query tensor = Q message.
Checkpoint names (PR `conversion_mapping.py`):
`controllers.{attn,k,v,mlp}.down_weight` →
`{q,k,v,mlp}_message_gate.down_proj.weight`;
`...gain_weight` → `..._message_gate.message_gate_proj.weight`;
`...forget_weight` → `..._message_gate.forget_gate_proj.weight`.

## 4. mix() (PR #51050, `Inkling2LoopMessageAttentionResidual`)

```python
class Inkling2LoopMessageAttentionResidual(Inkling2BlockAttentionResidual):
    """Retrieves messages from the block states of the previous block using learned queries."""

    def __init__(self, config: Inkling2LoopTextConfig):
        super().__init__(config)
        # one query per message (query, key, value and mlp) for each loop layer
        shape = (config.loop_layer_index_end - config.loop_layer_index_start, 4, config.hidden_size)
        self.query = nn.Parameter(torch.zeros(*shape, dtype=torch.float32))

    def forward(self, block_states, partial_block_states, loop_layer_idx) -> torch.Tensor:
        stacked_states = torch.cat([block_states, partial_block_states.unsqueeze(-2)], dim=-2).float()
        stacked_states = self.norm(stacked_states)
        ...
        query_states = self.query[loop_layer_idx].float()
        query_states = query_states.expand(*stacked_states.shape[:-2], -1, -1)
        ...
        attn_output, _ = attention_interface(
            self, query_states, stacked_states, stacked_states, None, scaling=hidden_size**-0.5
        )
        return attn_output.transpose(1, 2)
```

Base class (`Inkling2BlockAttentionResidual`, main-branch): learned queries
attend over RMSNormed `[block_states..., cur]` with scale `hidden**-0.5`
(SDPA, unbind per query). Loop version: `message_input` holds one
`(num_loop_layers, 4, hidden)` query tensor; `messages =
message_input(prev_blocks, prev_out, loop_layer_idx)` →
`[B, T, 4, H]`; `messages.unbind(-2)` = q/k/v/mlp messages (§3).
llama.cpp `mix(blocks, cur, query)` is the same op (concat blocks+cur,
rms_norm, `Q @ states` softmax `/sqrt(n_embd)` back-project).
Bank (§6): `previous_block_states` = prev pass blocks after the first core
block + prev output — i.e. pass-1 blocks + pass-2+ blocks appended
(`num_pre_loop_blocks + 1` skip) + `previous_hidden_states`. NOTE: each pass
re-seeds `block_states` from the prelude snapshot, so `message_input` sees
ONLY the previous pass's blocks — `mix()` never spans two passes back.

Query packing (verbatim, PR `core_model_loading.py`
`Inkling2LoopMergeMessageQueries`):

```python
        # (num_loop_layers, 2, hidden_size): [query, mlp] and [key, value] of each loop layer
        queries = queries.unflatten(0, (-1, 2))
        kv_queries = kv_queries.unflatten(0, (-1, 2))
        return {target_patterns[0]: torch.cat([queries[:, :1], kv_queries, queries[:, 1:]], dim=1)}
```

Checkpoint stores `middle_cycle_cross_pass_queries` ([q, mlp] per layer) and
`middle_cycle_cross_pass_kv_queries` ([k, v] per layer), each
`(2*num_loop_layers, hidden)`; merged to `(num_loop_layers, 4, hidden)` in
**[q, k, v, mlp] order**. Correction to the brief's item 6: the order is
q/kv-interleave in the STORED tensors, merged q,k,v,mlp — there is no
`[q[2i], kv[2i], kv[2i+1], q[2i+1]]` packing in either transformers or
llama.cpp; llama.cpp `msg_query` is `[n_embd, 4]` (4 = Q,K,V,FFN).

## 5. MoE (main-branch; loop reuses via `Inkling2LoopTopKRouter` passthrough)

```python
class Inkling2TopKRouter(nn.Module):
    def __init__(self, config: Inkling2TextConfig):
        self.top_k = config.num_experts_per_tok            # 8
        self.num_experts = config.num_local_experts         # n_routed_experts: 384 Small / 128 Mini
        self.score_fn = ACT2FN[config.scoring_func]         # "sigmoid"
        self.routed_scaling_factor = config.routed_scaling_factor  # 8.0
        self.e_score_correction_bias = nn.Buffer(torch.zeros(self.num_experts, dtype=torch.float32))

    def forward(self, hidden_states):
        # routing runs in fp32, as in the original
        hidden_states = hidden_states.float()
        flat = hidden_states.reshape(-1, self.hidden_dim)
        logits = F.linear(flat, self.weight)
        scores = self.score_fn(logits)
        indices = torch.topk(scores + self.e_score_correction_bias, self.top_k, dim=-1, sorted=False).indices
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return logits, weights * self.routed_scaling_factor, indices
```

Experts (`Inkling2Experts` + `Inkling2SoftcappedSwiGLU`):

```python
    def forward(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        input_dtype = gate.dtype
        gate, up = gate.float(), up.float()
        return (self.gate_softcap(gate) * torch.sigmoid(gate) * self.up_softcap(up)).to(input_dtype)
# gate_softcap = Inkling2Softcap(config.mlp_gate_softcap)   # 4.0
# up_softcap   = Inkling2Softcap(config.mlp_up_softcap)     # 25.0
# accumulate the top-k expert outputs in fp32, as in the original
```

Per-operator summary: sigmoid top-8 with auxiliary-loss-free bias
(`e_score_correction_bias`, init 0), weights = top-8 sigmoid scores gathered
then **divided by their sum** and scaled by `route_scale = 8.0`.
CORRECTION TO THE BRIEF ("no renorm"): transformers DOES renorm
(`weights / weights.sum`). llama.cpp agrees numerically — its
`LG_EXPERT_GATING_FUNC_TYPE_SIGMOID` path applies the same
renorm-then-`route_scale`; the "no renorm" wording in the hipfire oracle's
`sigmoid_topk` test name refers to no SOFTMAX, not no rescale. No shared
experts. Expert combine accumulates in fp32.

SiTU equivalence (proof, not re-derivation): llama.cpp converter
(`conversion/inkling2.py:234-235`)
`w.add_activation_situ_beta(hp["mlp_gate_softcap"])` /
`w.add_activation_situ_linear_beta(hp["mlp_up_softcap"])` — i.e. SiTU
beta/linear_beta ARE the 4.0/25.0 softcaps, and
`softcap(g)*sigmoid(g)*softcap(u)` is exactly SiTU's formula. Gate/up are
one packed `gate_up_proj` (`w13`, Interleave dim=1 on load), `down_proj` is
`w2`.

## 6. Loop geometry (PR #51050 `Inkling2LoopTextModel.forward`, verbatim skeleton)

```python
        loop_start = self.config.loop_layer_index_start      # 6
        loop_end = self.config.loop_layer_index_end            # 18
        num_loop_layers = loop_end - loop_start                # 12

        if cache is not None and len(cache.layers) == self.config.num_hidden_layers:
            # Append extra caches for loop layers
            loop_cache_layers = cache.layers[loop_start:loop_end]
            cache.layers += [copy.deepcopy(layer) for layer in loop_cache_layers * (self.config.loop_num_passes - 1)]
        ...
        # Pre loop layers
        for decoder_layer in self.layers[:loop_start]: ...    # 0..6 once
        pre_loop_hidden_states, pre_loop_block_states = hidden_states, block_states
        num_pre_loop_blocks = pre_loop_block_states.shape[-2]

        # First pass over loop layers                       # pass 0: no messages
        for decoder_layer in self.layers[loop_start:loop_end]: ...

        # Remaining passes restart from the pre loop states and receive messages from the previous pass
        for loop_idx in range(1, self.config.loop_num_passes):   # passes 1, 2
            # +1 to skip the states appended by the first loop pass
            previous_block_states = block_states[..., num_pre_loop_blocks + 1 :, :]
            previous_hidden_states = hidden_states
            hidden_states, block_states = pre_loop_hidden_states, pre_loop_block_states

            # Each pass uses its own copies of the loop cache layers, appended after the model's layers
            cache_offset = self.config.num_hidden_layers + (loop_idx - 1) * num_loop_layers - loop_start
            for loop_layer_idx, decoder_layer in enumerate(self.layers[loop_start:loop_end]):
                messages = self.message_input(previous_block_states, previous_hidden_states, loop_layer_idx)
                new_hidden_states = decoder_layer(
                    hidden_states, block_states, messages, ...,
                    cache=cache, cache_offset=cache_offset, ...)

        # Post loop layers
        for decoder_layer in self.layers[loop_end:]: ...      # 18..24 once
```

`layer_phys(il) = il < n_phys ? il : core_start + (il - n_phys) %
(core_end - core_start)` (llama.cpp `inkling2-loop.cpp:3-5`, verified).
Mini: 24 phys layers, core [6,18) (block-size 6 aligned), 3 passes →
**48 visits** (`total_num_hidden_layers` property). Each pass's cache layers
are deep-copied siblings (`cache_offset` threads through attn/conv calls);
doc note: "`output_hidden_states` and `output_attentions` return one entry
per layer visit". Varloop research checkpoint (`TML-d24-varloop-research`):
same arch, `loop_num_passes` overridable via `text_config` at load
(`__post_init__` reads `loop_config`); vLLM additionally supports
`loop_share_cache`.

## 7. Secondary targets: vLLM / SGLang (structure only)

- vLLM: `vllm-project/vllm#69028` (day-0 blog 2026-11-20). Per vLLM blog +
  recipe: core layers 6–17 × 3 passes with shared weights; each pass keeps
  its own KV + conv-state cache registered as ordinary cache layers (48 of
  each) so prefix caching / CUDA graphs work; the four message reads per
  layer fused into one attention-residual op; the four controllers' weights
  stacked for batched GEMM; bundled DSpark speculator
  (`num_speculative_tokens: 7`); NVFP4 experts cannot be TP-sharded
  (activation scales cover full expert width) — keep EP == TP.
- SGLang: `sgl-project/sglang#51926` (cookbook "Inkling-2-Mini"). fa4
  attention backend; DSpark draft reads decoder visits 5, 23, 41 + final
  normalized state 48; `--page-size 128`; `--reasoning-parser inkling`,
  `--tool-call-parser inkling2`.
- Neither was code-read for this report; useful for kernel structure and
  tensor naming only. The math above stands on transformers + llama.cpp.

## 8. Deltas vs the already-verified log (no re-derivation — what's NEW here)

1. `cache_offset` threading: every attn/conv/cache call in the PR branch
   takes `layer_idx + cache_offset`; per-pass deep-copied cache layers.
   The hipfire carrier/forward must implement pass-indexed KV + conv caches
   (48 slots), not 24.
2. Controller tensor names (checkpoint → hipfire loader): see §3 mapping;
   merged `message_input.query` is `(12, 4, 2304)` in [q,k,v,mlp] order —
   split from the two `(24, 2304)` stored tensors, NOT a `[q2i,kv2i..]`
   interleave (brief item 6 wording corrected).
3. PR-branch refactor modules (`Inkling2QueryProjection`,
   `Inkling2KeyValueProjection`, `Inkling2GatedOutputProjection`,
   `Inkling2RelativePositionBias`, `Inkling2LogScale`) are renames of
   main-branch math — no numeric change; hipfire oracle ops already cover
   them.
4. MoE: transformers renorms top-8 sigmoid by sum THEN × 8.0; keep the
   hipfire `sigmoid_topk` semantics identical to llama.cpp (gather →
   sum-divide → × route_scale), not pure top-8 without rescale.
5. NVFP4/MXFP8 dequant recipe now has a second independent witness (PR
   `Inkling2LoopDequantizeExperts`): E2M1 nibble order (even = low nibble),
   fp8 scale per 16 values, per-expert `amax / outer_scale_denominator`
   factor; MXFP8 `2**(scale-127)` per 32 values. Agrees with the merged
   llama.cpp converter the log verified.
6. Mini config pins (PR `Inkling2LoopTextConfig`): hidden 2304, heads 24/8,
   layers 24, experts 128 × top-8 × 1152, `log_scaling_n_floor` 8192,
   core [6,18), passes 3, gate rank 64, vocab 201024 / unpadded 200058,
   image_token 200054. NVFP4 layers 0–19, MXFP8 20–23 (vLLM recipe).

