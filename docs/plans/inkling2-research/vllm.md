# Inkling-2 research: vLLM implementation (PR #69028)

Scope note: per the orchestrator brief, vLLM is a **secondary target** — useful
for kernel structure and tensor naming, **not** authoritative on math. The math
oracle remains `modeling_inkling2.py` (transformers) / the merged llama.cpp PR
#32694 (`29b52d398`). This report quotes vLLM verbatim and flags every place
vLLM's numerics need checking against the primary oracle.

- PR #69027 = recipe YAML only (`recipes/models/thinkingmachines/Inkling-2-Mini.yaml`).
  No model code. Not covered beyond this line.
- PR #69028 = the model (`[Model] Add Inkling 2 Mini`, open, branch
  `WoosukKwon:inkling2-mini`). All excerpts below are from that branch at
  `b3c0ff7` (plus `main` for shared kernels that #69028 reuses unchanged:
  `ops/attention.py`, `sconv_swa_attn.py`). Builds on the Inkling-2 Small
  support (#67325, #67880, #68761).
- Validation quoted in the PR (GB200): `tests/models/inkling2` all pass;
  GSM8K-200 97.5% (DSpark, MAL 3.55); TP4 NIAH 550/550 to 1M; per-layer-visit
  hidden states differ from the HF reference by about as much as the
  reference's own FA4-vs-eager paths differ from each other; **MoE matches
  bitwise given the same inputs**; numerics are same-dtype FP32 message merges,
  only reduction order differs.

## 0. Checkpoint facts (HF `thinkingmachines/Inkling-2-Mini` config.json)

Verbatim key fields of `text_config` (Mini is `model_type: inkling2_loop`,
arch `Inkling2LoopForConditionalGeneration`):

```json
"hidden_size": 2304, "num_hidden_layers": 24,
"vocab_size": 201024, "unpadded_vocab_size": 200058,
"num_attention_heads": 24, "num_key_value_heads": 8,
"head_dim": 128, "d_rel": 16, "rel_extent": 1024,
"log_scaling_n_floor": 8192, "log_scaling_alpha": 0.1,
"qk_l2_clip_threshold": 22.627416997969522, "rel_l2_clip_threshold": 8.0,
"local_layer_ids": [0,1,2,3,4, 6,7,8,9,10, 12,13,14,15,16, 18,19,20,21,22],
"sconv_kernel_size": 4, "sliding_window_size": 1024,
"n_layers_per_attn_res_block": 6, "attn_res_normalize_values": true,
"final_logit_softcapping": 30.0,
"n_routed_experts": 128, "num_experts_per_tok": 8, "n_shared_experts": 0,
"intermediate_size": 1152,
"mlp_gate_softcap": 4.0, "mlp_up_softcap": 25.0,
"route_scale": 8.0, "use_gate_bias": true,
"loop_config": {"core_start": 6, "core_end": 18, "num_passes": 3,
                "gate_rank": 64, "is_varloop": false}
```

Derived, all confirmed in code below: 20 local + 4 full layers
(full = 5, 11, 17, 23 — one per AttnRes block of 6); loop core = layers
6..17 (12 layers) x 3 passes; 24 + 2x12 = **48 layer visits**.
`mtp_config.target_layer_ids = [5, 23, 41, 48]` = visits 5/23/41 + final
norm output (48 = `num_layer_visits`). `mlp_gate_softcap/up_softcap` =
**the brief's SiTU `beta=4, linear_beta=25`** (see §5).
Note: checkpoint `log_scaling_n_floor` is **8192**, not the vLLM config
default 128000 — the file on disk wins; do not copy the default.

## 1. Attention

Architecture quote (`nvidia/attention.py`, module docstring):

```python
"""Inkling2 attention: relative-bias GQA with short-conv K/V, L2-clipped
q/k/r, gated output and per-token attention sinks."""
```

Full vs local split (`Inkling2Attention`, docstring + `split_sizes`):

```python
    """Local layers project all of q/k/v from one input; full layers project the
    query side (q, r, gate, sink) from `x_q`, k from `x_k` and v from
    `x_v`, and use dynamic K/V convolutions and log-scaled q/r. Local layers
    given separate `x_k` / `x_v` (loop passes after the first) project them
    as full layers do."""
        # Query-side outputs first so full layers can project them in one GEMM.
        self.split_sizes = [
            q_size,                        # q
            self.num_heads * self.d_rel,   # r
            q_size,                        # gate
            self.num_heads,                # sinks (one scalar per head)
            kv_size,                       # k
            kv_size,                       # v
        ]
```

So the checkpoint's `qkvr` projection packs, in order: **q, r, gate,
sink, k, v** (full layers do the query side `q+r+gate+sink` in one GEMM,
k/v separately; local layers project all six from one input). Tensor
naming for the loader: `{prefix}.qkvr`, `{prefix}.wo_ud`,
`rel_logits_proj` `[d_rel, rel_extent]`, per-head scales `q_scale`
`[H, D]`, `r_scale` `[H, d_rel]`.

Relative-bias projection/gather. Table is pre-shifted at load
(`process_weights_after_loading`):

```python
    def process_weights_after_loading(self) -> None:
        self.rel_proj_shifted = shift_rel_proj(self.rel_logits_proj, self.is_local)
```

```python
def shift_rel_proj(rel_proj: torch.Tensor, is_local: bool) -> torch.Tensor:
    """Return the `(8, d_rel + 1, extent + 512)` table `attention_prologue`
    reads the `(d_rel, extent)` relative projection from, so that every
    sheared output column block is an aligned slice of one of 8 shifts.
    `out[s, :d_rel, y]` is `rel_proj[:, d]` for distance `d = s + extent
    + 256 - y`, or zero out of range; row `d_rel` is the padding added to
    the logits: -inf for negative distances and, if `is_local`, those past
    the extent, else zero."""
```

Math summary: raw table `[16, n_rel]` (d_rel=16, n_rel=`rel_extent`=1024,
window 1024 for local layers) is expanded to 8 shift-variants so each
sheared FA4 column block is one aligned slice. Per-token relative logits
are `r_clipped @ rel_proj[dist]` + pad row; distances < 0 get -inf, and on
local layers distances past the extent get -inf too (else 0). Output is
**sheared** for tml_FA4 (`(T+128, H, extent+256)`, column `c` of token `t`
holds key `c + cdiv(past + tile_end, 128)*128 - (E+256)` over 128-row
query tiles). FA4 adds bias *before* scaling:

```python
    """tml_fa4 paged relative attention with fp32 output and the natural-log LSE.
    The relative bias is added before scaling: `(qk + bias / scale) * scale`."""
```

QK + rel L2 clips + tau (fused in `attention_prologue`, `ops/sconv.py`
docstring; kernel `qr_clip_rel_tokens`, `ops/norm.py`):

```python
    """Writes the L2-clipped `q * q_scale` to `q_out` and the relative logits
    of the clipped `r * r_scale` to `rel_out` (with `tau`, clipped values
    are rounded, then scaled by the fp32 per-token `tau`). ... Runs the
    K/V short convolutions, clips K, RMS-normalizes V and inserts both into
    the KV cache (tokens with `kv_slot_mapping < 0` skip the write)."""
```

```python
    """For tokens `[t0, t0 + num_t)`: per-head L2 clipping of `q * q_scale`
    (`chunk < 0`) or of `r * r_scale` and the `BLOCK_E` columns `chunk`
    of the relative logits `r @ rel_proj` (fp32 accumulate ...). Clipped values are
    rounded, then scaled by the fp32 per-token `tau`."""
```

```python
def _l2_clip(X, Scale, heads, cols, mask, threshold, eps, tau, HAS_TAU):
    x = tl.load(X + heads * cols.shape[1] + cols, mask, other=0).to(tl.float32)
    x = x * tl.load(Scale + heads * cols.shape[1] + cols, mask, other=0).to(tl.float32)
    sum_sq = tl.sum(x * x, axis=1)
    norm = tl.where(sum_sq > 0, sum_sq * rsqrt(sum_sq), 0.0)
    y = x * tl.minimum(tl.div_rn(threshold, norm + eps), 1.0)[:, None]
    if HAS_TAU:
        y = y.to(X.dtype.element_ty).to(tl.float32) * tau
    return y.to(X.dtype.element_ty)
```

Math summary: per-head `y = (v*s) * min(threshold / (||v*s|| + eps), 1)`
with thresholds `qk_l2_clip_threshold = 22.627... (= 16*sqrt(2))` for q
(and K in the prologue) and `rel_l2_clip_threshold = 8.0` for r; then, on
full layers only, multiply by per-token `tau`. Tau
(`compute_log_scaling_tau`, `ops/attention.py`):

```python
    """Return fp32 `1 + alpha * log(max((positions + 1) / n_floor, 1))`."""
```

with checkpoint `n_floor = 8192`, `alpha = 0.1`. Full-layers-only is
explicit at the call site (`nvidia/model.py`):

```python
        return self.attn(
            positions, inputs[0], inputs[1], inputs[2],
            None if self.is_local else tau,   # tau scaling on full layers only
            out, pass_idx,
        )
```

Output gate + sink subtraction (`gate_sink`, `ops/attention.py`):

```python
    """Write bf16 `output * sigmoid(gate) * sigmoid(lse - sink)` per head to
    `result`. `gate` and `result` are `(T, H * D)`, `sinks` and `lse` are
    `(T, H)` and `output` is `(T, H, D)`."""
```

Math summary: per head, `out = attn_out * sigmoid(gate) *
sigmoid(LSE - sink)` where `sink` is the projected per-head scalar and LSE
is the attention log-sum-exp. The SM90 fallback kernel does the identical
gated combine inline (`ops/attention_sm90.py`:
`o = o * sigmoid(gate); o = o * sigmoid(lse - sink)`).

Softcap 30 is at the LM head, not in attention (`nvidia/model.py`):

```python
        self.logits_processor = processor(
            text_config.padded_vocab_size,
            org_vocab_size=text_config.vocab_size,
            soft_cap=text_config.final_logit_softcapping,   # 30.0
        )
```

## 2. Short convs (unified recurrent state)

One paged cache per decoder layer holds all four streams
(`nvidia/sconv.py` docstring; layout from `sconv_swa_attn.py`):

```python
"""Each decoder layer owns one `InklingConvState` ... holding its four conv
streams (K, V, attn-out, FFN-out) ... K/V are TP-sharded with the KV heads;
the attn-/mlp-output streams are replicated full-width on every rank
(`replicate_hidden=True`)."""
        # Packed per-head width: K + V + attn-output chunk + mlp-output chunk,
        # padded to a power of two ...
        # Per-head D-sub-range (offset, width) for each stream. Streams share
        # the cache; each writes/reads its own width across all H heads.
        self.stream_ranges = (
            (0, head_dim),                              # _K
            (head_dim, head_dim),                       # _V
            (2 * head_dim, hidden_per_head),            # _ATTN
            (2 * head_dim + hidden_per_head, hidden_per_head),  # _MLP
        )
```

Stream ids: `K_STREAM, V_STREAM, ATTN_STREAM, MLP_STREAM = range(4)`.
Tokenizer-facing note: hipfire needs no such page unifier; the math to
port is the per-stream causal conv below.

Static vs dynamic taps (`ops/sconv.py` docstring — matches the brief's
formulae exactly):

```
    Static convolution (fp32 accumulate, oldest tap first)::
        y = (sum_k w[c, k] * x[t - (W - 1) + k] + x[t]) * (W + 1) ** -0.5
    Dynamic convolution (newest tap first, per-token per-head tap weights)::
        y = (sum_l (static_w[l, c] + dyn[t, head(c), l]) * x[t - l] + x[t])
            * (2 * W + 1) ** -0.5
    `y` is rounded to the input dtype. The K/V variant then L2-clips K and
    RMS-normalizes V per head and writes both to the paged KV cache.
```

Code scale factors confirming the post-scale `1/sqrt((dynamic?2:1)*W+1)`:

```python
        (width * (2 if dynamic else 1) + 1) ** -0.5,   # attention_prologue: W=4 -> 5**-0.5 / 9**-0.5
```

```python
            5**-0.5,   # ops/conv_mix.py: hidden-stream static W=4 conv fused with AttnRes mix
```

Tap ordering in `_tap_sum`: `lag = k if dyn else W-1-k` — static goes
oldest-first, dynamic newest-first. Taps inside the forward come from `x`,
earlier taps from the paged cache, taps before position 0 are zero.

Which streams are dynamic: K/V convs on **full** layers only
(`nvidia/attention.py`):

```python
        self.k_sconv, self.v_sconv = (
            Inkling2ShortConv(
                kv_size, config.sconv_kernel_size, conv_state, stream,
                dynamic_input_size=None if is_local else config.hidden_size,
                num_heads=self.num_kv_heads,
            )
            for stream in (K_STREAM, V_STREAM)
        )
```

Dynamic taps = `static_w [W, dim] + weight_proj(x)` with
`weight_proj: [num_heads*W, hidden]` projected from the stream input
(`dynamic_taps`: `linear_k_groups` fp32 1024-wide K-group path when wide,
else `nn.functional.linear`). Attn/mlp-out streams are always static
(`assert not self.dynamic`).

vLLM perf note (structure, not math): from >=2048 tokens the q/r prologue
runs in its own launch beside the K/V convs (`_prologue_qr_tokens`;
"an 8K prefill at 51K context: 120.5 -> 116.1 ms on GB200").

## 3. Controllers (loop message gates)

Per-stream reference (`nvidia/loop.py`):

```python
class Inkling2MessageController(nn.Module):
    """Blends a loop message into a core sublayer input: from the rank-`r`
    projection `h` of the RMS-normalized `[dest, message]`,
    `(1 - tanh(forget(h))) * dest + tanh(gain(h)) * message` in fp32."""
    def forward(self, dest, message):
        """fp32 `(T, D)` `dest` and `message` -> fp32 `(T, D)`."""
        pair = torch.stack((dest, message), dim=1)
        pair = pair * torch.rsqrt(pair.square().mean(-1, keepdim=True) + 1e-6)
        normalized = pair.to(torch.bfloat16).flatten(1)
        hidden = linear_k_groups(normalized, self.down_weight, torch.bfloat16)
        gain = torch.mm(hidden, self.gain_weight.t(), out_dtype=torch.float32)
        forget = torch.mm(hidden, self.forget_weight.t(), out_dtype=torch.float32)
        return (1.0 - torch.tanh(forget)) * dest + torch.tanh(gain) * message
```

Shipped path batches the 4 streams (`attn, k, v, mlp`): weights stacked
as `controller_down [S, r, 2D]` and `controller_gain_forget [S, 2D, r]`
(bf16, `r = gate_rank = 64`), `pair_rms_norm` = weightless per-stream
RMS of dest and message to bf16 `[S, T, 2D]`, then batched bmm; the merge
runs fused with the sublayer RMSNorm (`blend_rms_norm`) **without FMA
contraction, as the reference**:

```python
            # The blend is unfused (no FMA), as the reference.
            enable_fp_fusion=False,
```

and the blend output feeds the sublayer's RMSNorm in the same kernel
(`blend()` returns "bf16 `(S, T, D)`" post-norm). Controller norm eps is
hard `1e-6`, not the layer eps.

## 4. mix() — block attention residual

(`ops/attn_res.py` docstring + kernel; this is the brief's `mix()` and
the message-read primitive.)

```python
"""Block attention residual: softmax-weighted mix of residual stream blocks.
For each token, sources `x_n` (stored blocks, then the partial block plus an
optional addend, all in fp32) are scored as `(q . x_n) * rsqrt(mean(x_n^2) +
eps)` per pseudo-query `q`; the output is `sum_n softmax_n * x_n`, with
`x_n` RMS-normalized when `normalize_values` is set. The fp32 mix then
goes through the following sublayer's RMSNorm in the same kernel, or is
returned as is (fp32) without a norm weight."""
```

Queries (`nvidia/model.py`): `attn_queries` = 1 query per layer boundary
for local layers, **3 for full layers** (q/k/v inputs each get their own
mix); `mlp_queries` = 1 per layer; plus one `output_query`. All fp32,
pre-scaled by `hidden**-0.5` at load (`_load_query ... .mul_(self.scale)`),
and `attn_res` asserts fp32. Kernel detail: three passes (logits+inv-RMS,
mix-RMS, normalized mix); fp32 throughout, `enable_fp_fusion=False`;
small batches use a row-register kernel. `NORMALIZE` =
`attn_res_normalize_values` (true for Mini/loops: `probs *= inv_rms`).

Queries per visit for the loop's message reads (`nvidia/loop.py`):

```python
        # Message pseudo-queries (raw), two per core layer: attn/mlp and k/v.
        shape = (2 * len(self.loop_core), config.hidden_size)
        self.middle_cycle_cross_pass_queries ...
        self.middle_cycle_cross_pass_kv_queries ...
    def process_weights_after_loading(self):
        # (attn, k, v, mlp) = (queries[0], kv[0], kv[1], queries[1]).
        stacked = torch.stack(
            (queries[:, 0], kv_queries[:, 0], kv_queries[:, 1], queries[:, 1]), 1
        )
        self.message_queries.copy_(stacked * self.hidden_size**-0.5)
```

i.e. the brief's cross-pass packing **[q[2i], kv[2i], kv[2i+1],
q[2i+1]]**, scaled by D^-0.5. Message read itself (`_run_layer`):

```python
            messages = attn_res(
                bank[:1], bank[1], None,
                self.message_queries[i - self.loop_core.start], None, 0.0, True)
```

raw fp32 (no norm weight), over the 2-block bank; then each stream blends
`messages[0..3]` into its dest (attn streams share the dest broadcast for
local layers: `dests.expand(3, ...)`).

## 5. MoE

Router (`ops/moe.py`): selection on `sigmoid(logit) + bias`, top-8 by an
order-preserving bit key (ties prefer the lower expert index):

```python
def _selection_keys(logits, bias, experts, E):
    """Sort keys ... order-preserving bits of sigmoid(logit) + bias in the
    high word and the reversed expert index in the low bits, so ties prefer
    the lower expert index."""
    sigmoid = tl.div_rn(1.0, 1.0 + tldevice.exp(-logits))
    selection = sigmoid + bias
```

Routing weights — **read carefully, discrepancy vs the brief** (§7, flag
F1):

```python
def _routing_weights(log_probs, route_scale, TOPK):
    """Renormalized sigmoid probabilities [R, TOPK] of the selected experts
    from their log-sigmoids."""
    max_log_probs = tl.max(log_probs, 1)
    shifted = tldevice.exp(log_probs - max_log_probs[:, None])
    total = _sum8(shifted) if TOPK == 8 else tl.sum(shifted, 1)
    logsumexp = max_log_probs + tldevice.log(total)
    return tldevice.exp(log_probs - logsumexp[:, None]) * route_scale
```

i.e. `w_i = sigmoid(l_i)/sum_j(sigmoid(l_j)) * 8` — renormalized to sum to
`route_scale`. Gate: `Inkling2Gate` asserts `use_gate_bias`, fp32 bias;
`route_scale = 8.0`, `topk = 8`, `n_routed_experts = 128` (Mini;
`n_shared_experts = 0` asserted). The routing weight is applied **inside
the activation as gamma, before the down projection** ("The routing weight
is applied inside the SwiGLU, before the down projection"), and the local
combine is fp32 rounded once to bf16 before the TP all-reduce.

SiTU (`ops/activation.py`, `_situ` — the docstring says "SwiGLU" but the
code is SiTU; `swiglu_quantize` docstring confirms the formula):

```python
    """Quantized (NVFP4 with `weight_amax`, else MXFP8) bf16
    `softcap(gate) * sigmoid(gate) * softcap(up) * gamma`, computed in fp32."""
```

verbatim kernel body (`_situ`):

```python
    sigmoid = _sigmoid_approx(gate)
    gate = _softcap(gate, GATE_CAP)
    up = _softcap(up, UP_CAP)
    return (((gate * sigmoid) * up) * gamma).to(tl.bfloat16).to(tl.float32)
```

with `_softcap(x, CAP) = CAP * tanh.approx(x / CAP)` (exact division via
rounded-reciprocal `1/4` and `1/25` FMA correction) and Mini's
`GATE_CAP = mlp_gate_softcap = 4.0`, `UP_CAP = mlp_up_softcap = 25.0` —
**this is the brief's "SiTU (beta=4, linear_beta=25)"**: `situ(g, u) =
softcap4(g)*sigmoid(g)*softcap25(u)`, times routing weight. No other
beta/linear constants exist in the kernel. Sigmoid is an `ex2.approx`
fast path. Experts: NVFP4 (Blackwell/Hopper) or MXFP8, gate/up GEMM then
quantize-then-down-GEMM; Hopper uses Humming bf16-activation GEMMs.

## 6. Loop geometry

Config (`configs.py`): `loop_config = {core_start, core_end, num_passes,
gate_rank, is_varloop}` (ints + bool, validated); Mini = core 6..18, 3
passes, rank 64, fixed. Serving overrides via
`--hf-overrides '{"text_config": {"loop_num_passes": N,
"loop_share_cache": true}}'` — **rejected on fixed-loop checkpoints**;
DSpark requires the checkpoint's own pass count (`num_loop_passes ==
loop["num_passes"]`, else error).

Visits (`configs.py` + `loop.py`):

```python
    @property
    def num_layer_visits(self) -> int:
        extra_passes = self.num_loop_passes - 1
        return self.num_hidden_layers + extra_passes * len(self.loop_core_layers)
    def layer_visit_ids(self, layer_id: int) -> list[int]:
        core, passes = self.loop_core_layers, self.num_loop_passes
        if not core or layer_id < core.start: return [layer_id]
        if layer_id >= core.stop:
            return [layer_id + (passes - 1) * len(core)]
        return [layer_id + p * len(core) for p in range(passes)]
```

```python
        # (layer, pass) visits; the prelude and coda run as pass 0.
        visits = [(i, 0) for i in range(core.start)]
        visits += [(i, p) for p in range(num_passes) for i in core]
        visits += [(i, 0) for i in range(core.stop, len(self.layers))]
```

Mini: 6 prelude + 3x12 core + 6 coda = 48 visits; `block_size =
n_layers_per_attn_res_block = 6`; full layers 5/11/17/23 sit at block
boundaries. Per-pass state: extra `Inkling2AttentionPassCache` KV layers
(registered as real cache layers, so "the KV cache manager and prefix
caching need no changes") + per-pass conv states
(`conv_states[pass_idx]`); `loop_share_cache` shares one KV cache across
passes (conv states stay per-pass). Routed-expert capture has one row per
layer visit (`visit_ids`).

Prelude/bank mechanics (`loop.py forward` + class docstring):

```python
"""The core layers `[core_start, core_end)` run `num_passes` times: a prelude,
then each pass from the AttnRes state after the prelude, then the coda from
the last pass's state. Each pass of a core layer has its own KV and
conv-state caches. In passes after the first, every core sublayer input
(attention q/k/v and MLP) is blended with a message: a normalized-value
AttnRes read of the previous pass's blocks, merged by a gated low-rank
controller."""
```

```python
            elif i == core.start:
                # Every pass restarts from the prelude's state, with the
                # previous pass's blocks (after the prelude's) and partial as
                # the messages' bank.
                bank = stack[prelude_blocks + 1 : num_blocks + 1].clone()
                num_blocks = prelude_blocks
                stack[num_blocks].copy_(prelude_partial)
```

`prelude_dests`
(the first core visit's attn inputs) is computed once and reused for every
pass restart; "Messages blend into fp32 mixes; a pass restart discards"
the normed mix (`next_norm = None if next_pass else ...`). Coda continues
the last pass's state (no reset after the final pass).

DSpark hooks: draft reads target hidden states at `target_layer_ids`
visits (`[5, 23, 41, 48]`); `draft_block_size = 8` stored, `dspark_block_size
= 7` masks ("each predicting the token at its own position"),
`sample_from_anchor = False`, `is_causal = False`; SM12 draft KV forced to
BF16 ("The target's clipped, unit-scale FP8 cache does not describe the
draft's unnormalized values").

Fused kernels (`csrc/inkling2/`, `ops/conv_mix.py`,
`ops/allreduce.py`): `conv_mix`, `allreduce_conv_mix`, `seq_parallel`
now also support hidden 2304 (Mini) besides 4608; they "return the FP32
mix before the RMSNorm for message blends"; the loop model uses them at
every TP size but never sequence-parallel. TP path: all-reduce fused with
conv+mix (one-/two-shot), else reduce then `conv_mix`.

## 7. Cross-check vs the verified baseline + flags for the orchestrator

Agreements with llama.cpp #32694 (`inkling2.cpp`, `inkling2-loop.cpp`)
and the handoff log (`docs/plans/local/2026-11-21-inkling2-implementation-log.md`):

- A1. Controller formula identical: llama `control()` =
  `(1-forget)*cur + gain*message` with tanh on both projections of the
  RMS-normalized pair; hipfire oracle `control` matches.
- A2. Bank = "the previous pass's blocks after the first core block, and
  its output"; every pass restarts from the prelude output; llama visit
  map `il -> il < n_phys ? il : core_start + (il-n_phys) % core_len`.
- A3. Post-conv scales `5**-0.5` (static W=4) / `9**-0.5` (dynamic)
  confirmed in two independent call sites; hipfire `convolve_static`
  covers the static path, dynamic-tap path still open (log §Next steps 2).
- A4. Query packing `[q[2i], kv[2i], kv[2i+1], q[2i+1]]` confirmed verbatim
  in `process_weights_after_loading`.
- A5. `sigmoid_topk` oracle exists in hipfire (log: "no renorm") — see
  flag F1.
- A6. `situ` oracle exists — vLLM's kernel confirms the formula
  `softcap4(g)*sigmoid(g)*softcap25(u)*gamma`; hipfire `situ` should be
  diffed against exactly this (incl. the ex2-approx sigmoid question:
  vLLM uses `ex2.approx` fast sigmoid in the fused kernel and tanh.approx
  in softcap — hipfire CPU oracle should use exact transcendentals and
  note the expected ~1e-3-level deviation on served numerics).

Flags (vLLM differs from the brief, or underdetermines it — resolve
against `modeling_inkling2.py`):

- **F1. MoE weight renormalization.** Brief says "sigmoid top-8 gate with
  bias, route_scale=8, no renorm". vLLM `_routing_weights` **renormalizes**:
  `w = exp(logsig - logsumexp(logsig_top8)) * 8` (sums to 8). Plain
  `sigmoid*8` vs renormalized differ. One of {brief, vLLM} misdescribes
  the reference — the transformers agent's excerpt decides; hipfire's
  `sigmoid_topk` currently implements the brief's reading.
- **F2. tau floor.** vLLM default `log_scaling_n_floor = 128000`, Mini
  checkpoint says **8192**. Any hardcoded 128000 is wrong for Mini.
- **F3. MoE numerics scope.** vLLM's "bitwise MoE" claim is conditioned
  on "given the same inputs" through quantized NVFP4/MXFP8 experts; the
  hipfire plan quantizes experts to MQ4/MQ4L, so hipfire-side MoE parity
  must be measured post-quantization, not assumed from vLLM's claim.
- **F4. Gated-output path.** vLLM prefills through FA4 + separate
  `gate_sink`; hipfire AR decode should implement the fused math
  `attn*sigmoid(gate)*sigmoid(lse-sink)` directly — no FA4 needed.
- **F5. Not in vLLM (do not look for it here):** GGUF/HFQ tensor layout,
  dequant semantics, CPU reference forward — those live in the llama and
  hipfire agents' lanes.

## 8. Tensor-name map (vLLM name -> checkpoint/HF name, for the loader lane)

From `nvidia/model.py`, `nvidia/attention.py`, `nvidia/moe.py`,
`nvidia/loop.py` (`{prefix}` = `model.llm.layers.N` style + suffix):

| vLLM module / param | Checkpoint tensor (HF name) |
|---|---|
| `attn.qkvr` (+ `split_sizes` q,r,gate,sink,k,v) | `...self_attn.qkvr.weight` (packed; full layers: q-side rows then k, v rows) |
| `attn.wo_ud` | `...self_attn.wo_ud.weight` |
| `attn.rel_logits_proj` `[16, 1024]` | `...self_attn.rel_logits_proj` (raw table; vLLM shifts at load) |
| `attn.q_scale [H,128]`, `attn.r_scale [H,16]` | `...self_attn.q_scale` / `r_scale` |
| `attn_res.attn_queries[i]` (1 or 3 rows) | per-boundary attn pseudo-queries |
| `attn_res.mlp_queries[i]`, `attn_res.output_query` | per-layer + final pseudo-queries |
| `attn_norm` (1 or 3 stacked norms), `mlp_norm` | `...input_layernorm`-family (3 stacked on full layers) |
| `attn_sconv` / `mlp_sconv` (`conv_weight [D,4]`) | hidden-stream static conv weights |
| `k_sconv.static_w [4,dim]` + `weight_proj [Hkv*4, hidden]` | dynamic K/V conv taps (full layers only) |
| `mlp.gate.weight [E,H]` + `mlp.gate.bias [E]` | `...mlp.gate...` router + bias (fp32) |
| `mlp.experts.{w13,w2}_{weight,weight_scale,...}` | NVFP4/MXFP8 expert tensors (+ `_amax`, `_outer_scale_denominator`) |
| `loop.controller_down [4,64,2D]`, `loop.controller_gain_forget [4,2D,64]` | per-core-layer stream-stacked controller weights |
| `loop_model.middle_cycle_cross_pass_queries` / `..._kv_queries` `[2*12, D]` | raw cross-pass pseudo-queries (vLLM scales by D^-0.5 at load) |
| `model.embed_tokens`, final `norm`, lm head | vocab 201024 / unpadded 200058, BF16 unembed + FP32 output (`head_dtype`) |

Weight-loader caveat: several vLLM params are TP-sharded or
load-rescaled views (`_load_query` multiplies by `hidden**-0.5`;
`message_queries` likewise; `q_side` GEMM pads N to 8 with discarded
columns). The on-disk bytes are unscaled/unpadded — apply scale/pad in
the loader, not in the stored HFQ.

## 9. File inventory (vLLM PR #69028, fork `WoosukKwon:inkling2-mini`)

New files most relevant to the port, largest first:

- `vllm/models/inkling2/nvidia/loop.py` (403 lines) — loop driver, messages,
  controllers, prelude/bank/coda. Quoted §§3,4,6.
- `vllm/models/inkling2/nvidia/ops/loop.py` (132) — `pair_rms_norm`,
  `blend_rms_norm` Triton kernels. Quoted §3.
- `vllm/models/inkling2/nvidia/ops/moe_sm120.py` (234), `moe_humming.py`,
  `moe_sm90.py` — SM-specific experts; routing core shared in `ops/moe.py`.
- `vllm/models/inkling2/nvidia/ops/gemv.py` (113) — low-M BF16 GEMV (SM120).
- `vllm/models/inkling2/configs.py` (306) — `Inkling2ModelConfig`
  (all §0 defaults), `loop_core_layers`, `layer_visit_ids`,
  `Inkling2DSparkConfig`. Quoted §§0,6.
- Reused from Small/main: `ops/attention.py` (FA4 + `gate_sink` +
  `compute_log_scaling_tau`), `ops/attn_res.py` (306-line Triton mix),
  `ops/sconv.py` (prologue + convs), `ops/activation.py` (SiTU),
  `ops/conv_mix.py` + `csrc/inkling2/conv_mix.cu` (fused hidden conv+mix),
  `model.py` (layer/model driver, §6 call sites).

Test oracles in-tree: `tests/models/inkling2/test_core_ops.py` (65 pass),
`test_attention.py` (147), `test_model.py` (21), `test_moe.py` — useful as
a second numeric cross-check after the transformers excerpt lands, not as
a primary oracle.
