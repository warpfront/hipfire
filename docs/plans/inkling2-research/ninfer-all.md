# Inkling-2-Mini research: transformers oracle + ninfer-all loop port (ninfer-all)

Scope: what the hipfire CPU reference forward needs. Primary math oracle is
HuggingFace `transformers` `modeling_inkling2.py` (post-2026-11-20, `main`) for the
Small (non-looped) operators, **plus** llama.cpp `inkling2*.cpp` @ `29b52d398`
(verified in `docs/plans/local/2026-11-21-inkling2-implementation-log.md`) and
ninfer-all `tools/reference/inkling2.py` + `src/models/inkling2/*` (PR #385,
fixes #386–#388) for the Mini loop. Reason: the transformers file in this repo
is **Inkling-2-Small, not Mini** — 48 flat layers, `Inkling2BlockAttentionResidual`,
no `layer_phys`/prelude/bank. The loop geometry only exists in llama.cpp-loop
and ninfer-all.

Config baseline (transformers `configuration_inkling2.py`, Mini values confirmed
in `docs/plans/local/2026-11-20-inkling2-mini-support.md` §8 from live
`thinkingmachines/Inkling-2-Mini/config.json`):
`hidden 4608(Small)/2304(Mini)`, heads 48/8 (SWA same 48/8 on Small), head_dim 128,
`d_rel 16`, `rel_extent 1024`, SWA window 1024, `qk_clip 22.627416997969522`,
`rel_clip 8.0`, `log_floor 128000(Small)/8192(Mini per plan)`, `log_alpha 0.1`,
`rms_eps 1e-6`, sconv kernel 4, res-block 6, MoE 384x(top-8,1152)/128x(top-8,1152 Mini),
`route_scale 8.0`, `gate_softcap 4.0`, `up_softcap 25.0`, `final_softcap 30.0`,
`scoring sigmoid`, loop Mini `core [6,18) x3 passes, rank 64, is_varloop=false`,
48 visits, unpadded vocab 200058 (embeddings sized 201024, head sliced).

## 0. Which source to trust for what

- Attention / conv / clip / tau / gate / sink / softcap / MoE core math:
  transformers `modeling_inkling2.py` == llama.cpp `inkling2.cpp` == ninfer-all
  reference/executor, up to renorm wording (see §5).
- Loop schedule / prelude / bank / controllers / message packing: **only**
  llama.cpp `inkling2-loop.cpp` + `conversion/inkling2.py` + ninfer-all
  `reference/inkling2.py` + `executor.cpp`. Transformers `Inkling2TextModel`
  has `block_states` accumulation but no passes.
- Secondary (vLLM/SGLang): day-0 Mini support exists (vLLM blog 2026-11-20,
  SGLang cookbook + PR `sgl-project/sglang#51926`, vLLM PRs #67325/#67880),
  useful for kernel structure/tensor naming only. Not quoted here beyond the
  controller formula (identical): `out = (1-tanh(W_forget·h))·input +
  tanh(W_gain·h)·message`. Skipped deep vLLM/SGLang code read (see tail).

## 1. Attention

Math (all three agree):
`s = Q·Kᵀ/√d + rel_bias + mask`, sink joins softmax denominator as an extra
logit with zero value, `out = softmax([s, sink])[:-1]·V`, then
`out *= sigmoid(gate_proj(h))`, `o_proj`. Q has per-channel scale then clip;
K clips scaleless; R clips with per-channel scale; V is unweighted RMSNorm.
Full layers multiply Q **and** R by `tau`. Sliding layers: static conv on K/V,
extent = window 1024. Full layers: dynamic conv on K/V, extent 1024.

Verbatim — transformers `Inkling2RelativeLogits` + `eager_attention_forward`:
```python
rel_logits = (relative_states @ self.proj).transpose(1, 2)
distance = (query_positions[:, None] - key_positions[None, :])[None, None, :, :]
gather_index = distance.clamp(0, self.rel_extent - 1).expand(*rel_logits.shape[:2], -1, -1)
position_bias = rel_logits.gather(-1, gather_index)
return position_bias.masked_fill((distance < 0) | (distance >= self.rel_extent), 0.0)
...
combined_logits = torch.cat([attn_weights, sinks], dim=-1).float()
attn_weights = F.softmax(combined_logits, dim=-1)[..., :-1].to(query.dtype)
```
Verbatim — transformers clips/tau/gate (`Inkling2FullAttention.forward`,
`Inkling2SlidingAttention.forward` identical minus tau):
```python
tau = 1.0 + self.config.log_scaling_alpha * torch.log(
    ((kwargs["position_ids"] + 1).float() / self.config.log_scaling_n_floor).clamp(min=1.0))
query_states = self.q_l2_clip(query_states.view(hidden_shape))
query_states = (query_states.float() * tau).to(query.dtype).transpose(1, 2)
relative_states = (relative_states.float() * tau).to(relative_states.dtype)
...
attn_output = (attn_output.float() * torch.sigmoid(gate.float())).to(attn_output.dtype)
attn_output = self.o_proj(attn_output)
```
Verbatim — llama.cpp `inkling2.cpp` relative + sink (authoritative order: sink
subtracted **before** gather):
```cpp
ggml_tensor * table = ggml_mul_mat(ctx0, ggml_pad(ctx0, layer.attn_rel_bias, 0, 1, 0, 0), Rcur);
table = ggml_sub(ctx0, table, ggml_reshape_3d(ctx0, sinks, 1, n_head, n_tokens));
...
ggml_tensor * kq_b = ggml_get_rows(ctx0, table, rel_pos);
```
Verbatim — ninfer-all `tools/reference/inkling2.py:attention` (same order):
```python
bias_rows = r[:, h, :] @ table.T  # [n, extent]
s = s + np.where(inside, np.take_along_axis(bias_rows, np.clip(dist,0,extent-1), axis=1), 0.0)
top = np.maximum(s.max(-1, keepdims=True), sinks[:, h:h+1])
p = np.exp(s - top)
denominator = p.sum(-1, keepdims=True) + np.exp(sinks[:, h:h+1] - top)
```

## 2. Short convs (unified recurrent state)

Math: one recurrent state per visit holds K-window, V-window, attn-out-window,
FFN-out-window at channel offsets `0, E_k, 2·E_k, 2·E_k+E` (`E_k = n_head_kv*d`,
`E = n_embd`). Static taps (SWA + post convs) vs dynamic taps (full-layer K/V:
`static [C,L] + weight_proj(h) [n, heads·L]` shared per KV head). Post-scale
`1/sqrt((dynamic?2:1)·L+1)`, L=4 → `1/sqrt(5)` static, `1/sqrt(9)=1/3` dynamic.

Verbatim — llama.cpp `graph::convolve`:
```cpp
ggml_tensor * out = ggml_ssm_conv(ctx0, x, weight);
...
return ggml_scale(ctx0, ggml_add(ctx0, cur, out), 1.0f/sqrtf((dynamic ? 2 : 1)*n_conv + 1));
```
Verbatim — transformers `Inkling2DynamicShortConvolution.forward` (same denominator):
```python
hidden_states = (windows * weights).sum(-1).reshape(batch_size, seq_len, hidden_size)
hidden_states = (hidden_states + residual) / math.sqrt(2 * self.conv_kernel_size + 1)
# static path: / math.sqrt(self.conv_kernel_size + 1)
```
Verbatim — ninfer-all `kernels.cu:kv_post_kernel` (same scale, fused clip/norm):
```cpp
const float divisor = rsqrtf(float((dynamic ? 2 : 1) * kTaps + 1));
const float k = convolve(...) * divisor;
const float v = convolve(...) * divisor;
```

## 3. Controllers

Math: per core layer 4 controllers (Q/K/V/FFN): `h = down·[rms(cur);rms(msg)]`,
`down [2E,rank]`, `gain/forget [rank,E]`, rank 64,
`out = cur − tanh(F·h)·cur + tanh(G·h)·msg`, eps 1e-6, no norm learned weight.
SWA looped layers control all 3 attention inputs then per-input norm; FFN
message applied at MoE input. FP32, FMA-contraction off in ninfer-all.

Verbatim — llama.cpp `graph::control`:
```cpp
ggml_tensor * gate = ggml_concat(ctx0, ggml_rms_norm(ctx0, cur, 1e-6f), ggml_rms_norm(ctx0, message, 1e-6f), 0);
gate = build_lora_mm(loop.ctrl_down[ctrl], gate);
ggml_tensor * gain   = ggml_tanh(ctx0, build_lora_mm(loop.ctrl_gain[ctrl],   gate));
ggml_tensor * forget = ggml_tanh(ctx0, build_lora_mm(loop.ctrl_forget[ctrl], gate));
return ggml_add(ctx0, ggml_sub(ctx0, cur, ggml_mul(ctx0, forget, cur)), ggml_mul(ctx0, gain, message));
```
Verbatim — ninfer-all `reference/inkling2.py:control`:
```python
def control(x, message, parts):
    down, gain, forget = parts
    h = np.concatenate([rms(x, 1e-6), rms(message, 1e-6)], -1) @ down.T
    return x - np.tanh(h @ forget.T) * x + np.tanh(h @ gain.T) * message
```
Verbatim — ninfer-all `kernels.cu:control_apply_kernel`:
```cpp
y = make_float4(x.x - tanhf(f.x) * x.x + tanhf(g.x) * m.x, ...);
```

## 4. mix()

Math: depth attention over `[blocks..., cur]` (cur last), each RMSNormed
eps 1e-6, scores `q·s/√E` softmaxed, weighted sum. SWA: 1 query; full: 3
queries (Q/K/V slices). Bank = previous pass's `blocks[prelude_len+1:]` + its
output `cur`; prelude `blocks/cur` restored at each later pass start
(skip-first-core-block detail: `+1`).

Verbatim — llama.cpp `graph::mix`:
```cpp
ggml_tensor * states = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);
for (ggml_tensor * block : blocks)
    states = ggml_concat(ctx0, ggml_reshape_3d(ctx0, block, n_embd, 1, n_tokens), states, 1);
states = ggml_rms_norm(ctx0, states, 1e-6f);
ggml_tensor * scores = ggml_mul_mat(ctx0, states, queries);
scores = ggml_soft_max_ext(ctx0, scores, nullptr, 1.0f/sqrtf(n_embd), 0.0f);
return ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, states)), scores);
```
Verbatim — llama.cpp `inkling2-loop.cpp` bank/prelude:
```cpp
auto * prelude = cur;
const auto prelude_blocks = blocks;
...
bank.assign(blocks.begin() + prelude_blocks.size() + 1, blocks.end());
bank_out = cur; cur = prelude; blocks = prelude_blocks;
...
cur = build_layer(il, ip, cur, blocks, mix(bank, bank_out, loop.msg_query), &loop);
```
Verbatim — ninfer-all `reference/inkling2.py:mix` + pass driver:
```python
def mix(states, query):
    s = np.stack([rms(state, 1e-6) for state in states])  # [S, n, H], cur last
    scores = np.einsum("snh,qh->qns", s, query) / math.sqrt(s.shape[-1])
    ...
if core and ip == c.core[0]:
    if k == 0: prelude_blocks, prelude_cur = list(blocks), cur
    else: bank = blocks[len(prelude_blocks)+1:]; bank_last = cur; blocks, cur = list(prelude_blocks), prelude_cur
if core and k > 0: messages = mix(bank + [bank_last], message_queries(w, c, ip))
```

## 5. MoE

Math: sigmoid router + selection-only bias (`e_score_correction_bias` /
`exp_probs_b.bias`, init 0), top-8 by `score+bias` but weights from bare
`sigmoid(logit)`, **renormalized** (`w/sum`, eps 1e-20) then `×route_scale 8`.
Experts: `[gate;up]` rows with per-expert scale, SiTU
`gate_cap·tanh(g/gate_cap)·sigmoid(g) · up_cap·tanh(u/up_cap)`, caps 4/25.
Down with per-expert scale. Accumulate FP32. No shared experts.

Verbatim — transformers `Inkling2TopKRouter` + `Inkling2SoftcappedSwiGLU`:
```python
scores = self.score_fn(logits)  # sigmoid
indices = torch.topk(scores + self.e_score_correction_bias, self.top_k, dim=-1, sorted=False).indices
weights = scores.gather(1, indices)
weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
return logits, weights * self.routed_scaling_factor, indices
...
return (self.gate_softcap(gate) * torch.sigmoid(gate) * self.up_softcap(up)).to(input_dtype)
# gate_softcap = tanh(g/4)*4, up_softcap = tanh(u/25)*25
```
Verbatim — ninfer-all `reference/inkling2.py:moe` (same renorm):
```python
probs = sigmoid(logits)
selection = probs + (w.vector(g + "exp_probs_b.bias") if ... else 0)
chosen = np.argsort(-selection, axis=-1, kind="stable")[:, :c.top_k]
weights = (probs[tokens] / (probs...sum(-1) + 1e-20) * c.route_scale)
act = (c.gate_cap*np.tanh(gate/c.gate_cap)*sigmoid(gate) * c.up_cap*np.tanh(up/c.up_cap))
```
Verbatim — ninfer-all `situ.h:situ` (bit-exact portability notes):
```cpp
const float gate = s::mul(s::mul(gate_cap, portable_tanh(portable_div(g, gate_cap))), portable_sigmoid(g));
return s::mul(gate, s::mul(up_cap, portable_tanh(portable_div(u, up_cap))));
```
Correction to brief: "no renorm" is wrong vs all three codebases — every one
divides by the top-8 sum then scales by 8. llama.cpp passes `LLM_FFN_SITU` with
`expert_weights_scale` through the same `build_moe_ffn` renorm path.

## 6. Loop geometry

`layer_phys(il) = il<n_phys ? il : core_start+(il−n_phys)%(core_end−core_start)`;
`layers[il]=layers[phys]` (weight sharing, per-slot KV/conv state);
`n_slot = 24+2·12 = 48`; validation fixed-loop ≥2 passes,
`0<start<end≤n_phys`, block-aligned to 6, `rank≥1`, `n_nextn==0`.
Cross-pass queries: globals `q,kv` each `[2·core_len,E]` → per core layer `i`
GGUF `loop_msg_query [E,4]` stacks `[q[2i], kv[2i], kv[2i+1], q[2i+1]]` →
`[Q,K,V,FFN]`. Full visits 8 (global), sliding 40, per the otis cross-check in
the plan. `attn_res_query` absent on layer 0; SWA 1 query, full 3.

Verbatim — llama.cpp `inkling2-loop.cpp:layer_phys` + validation:
```cpp
return il < n_phys ? il : core_start + (il - n_phys) % (core_end - core_start);
if (n_passes < (varying ? 1u : 2u) || !gate_rank || !core_start || core_start >= core_end || ...) throw ...;
```
Verbatim — llama.cpp `conversion/inkling2.py` packing:
```python
query = torch.stack([q[2 * i], kv[2 * i], kv[2 * i + 1], q[2 * i + 1]])
yield self.format_tensor_name(gguf.MODEL_TENSOR.LOOP_MSG_QUERY, start + i, ".weight"), query
```
Verbatim — ninfer-all `tools/convert/inkling2.py:message_queries` (both layouts):
```python
if upstream_layout(gguf): return _direct(gguf, f"blk.{layer}.loop_msg_query.weight", (4, h))
q = ...reshape(2*core, h); kv = ...reshape(2*core, h)
rows_ = np.stack([q[2*i], kv[2*i], kv[2*i+1], q[2*i+1]])
```

## 7. ninfer-all fixes after the base PR (what changed)

- #386 (`6d470d8`): DSpark drafts by default over expert parallelism
  (drafter on rank 0, ≤8-token verify crosses devices once per MoE block).
- #387 (`5798d89`): 1-line MSVC fix — `component_is` compares the JSON string
  value, not `json == string_view` (C2666).
- #388 (`16aa690`, `b6399a1`): bench receipts vs llama.cpp (RTX 3090 1.7–2×,
  16 rented cards, host/disk expert tables). No math change.
- Converter covers both GGUF layouts (`inkling2_loop` 24-block and Unsloth
  `inkling2` 48-unrolled metadata) to byte-identical payloads; the header
  differs only in source path/artifact id.

## 8. Deltas vs hipfire's verified log (no re-derivation needed)

Agree with `2026-11-21-inkling2-implementation-log.md` + plan §8: NVFP4 nibble
order/scale companions, MXFP8 1×32, controller/mix/clip/convolve/tau/relative
order, MoE SiTU, `block_count=24` trap, `is_varloop` refusal. Two wording fixes
for the Rust oracle: (a) transformers Small ≠ Mini loop — cite loop files for
schedule, not `modeling_inkling2.py`; (b) MoE **does** renorm-then-scale in
every implementation.

Skipped: full vLLM/SGLang source read (blogs + PR/file lists only), vision
tower + DSpark internals beyond naming, perf-benchmark reproduction.
Risk: transformers `main` drifts (file is auto-generated from
`modular_inkling2.py`); re-pin SHA before citing numbers in code comments.
