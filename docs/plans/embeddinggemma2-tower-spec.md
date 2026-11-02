# EmbeddingGemma-2 tower spec — formula-level ground truth

Extracted from transformers 5.19.0 `models/embedding_gemma2/*`; towers are the
`models/gemma4` implementations (`gemma4_vision` / `gemma4_audio` — same archs the
checkpoint config declares). This is the implementation source for `image.rs`,
`vision.rs`, `audio.rs`; every constant below is checkpoint-verified unless noted.

## VISION TOWER

Config: hidden 768, inter 3072, 16 layers, 12 heads == 12 kv heads, head_dim 64,
act gelu_pytorch_tanh, rms_eps 1e-6, patch 16, pooling_kernel_size 3,
position_embedding_size 10240, rope axial theta 100, use_clipped_linears False,
standardize False.

### Image processor (image_processing_pil_gemma4.py)
- max_patches = 280 * 3² = 2520; side_mult = 3 * 16 = 48.
- Resize (aspect-preserving): `factor = sqrt((2520*256)/(H*W))`;
  `target_h = floor(factor*H/48)*48`, same for w. Zero-dim edge cases: set the
  zero side to 48, the other to `min(floor(ratio)*48, 280*48)`. Only when dims
  changed.
- Resample = **torchvision's `tvF.resize(..., BICUBIC, antialias=True)` on a
  uint8 tensor bit-for-bit**. The processor backend is
  `Gemma4ImageProcessor` (torchvision), so the resize is
  `torch.nn.functional.interpolate` on uint8, which ATen serves from its own
  separable fixed-point kernel
  (`aten/src/ATen/native/cpu/UpSampleKernel.cpp`,
  `HelperInterpCubic::_compute_index_ranges_int16_weights` +
  `basic_loop_separable_1d_{horizontal,vertical}<uint8_t>`); it is NOT
  `PIL.Image.resize(..., BICUBIC)` and NOT `image`'s `CatmullRom`.
  Per axis: `scale = in/out`,
  `support = (scale >= 1) ? 2*scale : 2` (Catmull-Rom, `a = -0.5`),
  `max_interp_size = ceil(support)*2 + 1`, kernel centre
  `scale*(i + 0.5) - 0.5` in sample coordinates, indices
  `[trunc(center - support + 0.5), trunc(center + support + 0.5))` clamped to
  `[0, in]`, `invscale = (scale >= 1) ? 1/scale : 1`. Weights are computed and
  normalized in **double**, and the whole axis keeps its largest normalized
  weight `wt_max`; the quantization precision is the first `p in 0..22` with
  `(int)(0.5 + wt_max*2^(p+1)) >= 2^15` (22 if none), and each weight becomes
  `(int)(±0.5 + w*2^p)` as **int16**. Each output is
  `clamp( (2^(p-1) + Σ pixel_i · k_i) >> p, 0, 255 )` in i32.
  The **width pass runs first**, into a uint8 intermediate, then the height
  pass. A single-pass f32 resampler with the same kernel lands up to 25/255
  away on textured images, which the tower amplifies past the media parity
  band.
- Rescale ×1/255; NO normalization. Pixels in [0,1].
- Patchify: NPH=H/16, NPW=W/16; per-patch vector index `((dy*16)+dx)*3 + c`
  (h→w→channel); rows row-major over (NPH, NPW). Positions `[x=col, y=row]`.
- `N_tokens = (target_h/16 * target_w/16) // 9` — DYNAMIC ≤ 280 (1024² → 256;
  the processor pads patches to 2520 with 0 and positions with -1).

### Patch embedder (modeling_gemma4.py L579–622)
`h = (2*(patches - 0.5)) @ W_in.T` (`input_proj [768,768]`, no bias);
`pos = table[0][x] + table[1][y]` (plain lookup, lower-clamp only; pad rows forced
to 0 vector); `h += pos`.

### Layer (L904–1015) — sandwich norms, attention scale 1.0
```
r = h; h = rmsnorm(h, input_layernorm)
q = rmsnorm(q_proj(h).view(L,12,64), q_norm[64]);  q = rope2d(q)
k = rmsnorm(k_proj(h).view(L,12,64), k_norm[64]);  k = rope2d(k)
v = rmsnorm_noscale(v_proj(h).view(L,12,64), v_norm[64])   # scale-less
attn = softmax(q@kᵀ * 1.0, fp32)                   # pad keys -inf; NO 1/sqrt(d)
h = r + rmsnorm(o_proj(attn@v), post_attention_layernorm)
r = h; h = rmsnorm(h, pre_feedforward_layernorm)
h = down(gelu_tanh(gate(h)) * up(h))
h = r + rmsnorm(h, post_feedforward_layernorm)
```
RMSNorm: fp32 upcast, `h * pow(mean_sq + eps, -0.5)` (no rsqrt), × weight.

### RoPE (L707–771, L848–901)
inv_freq (16) = `100^(-2i/32)`, i=0..15 — same for both axes.
Angles: H-pair i ← `x * inv_freq[i]`, W-pair i ← `y * inv_freq[i]`.
Table layout H-H-W-W: `recomposition_frequencies` returns
`cat([h, h, w, w])` (16 + 16 + 16 + 16), and `apply_multidimensional_rope`
splits that into the two 32-dim **halves** and rotates each with `rotate_half`
— so the x half (dims 0..31) pairs `(j, j + 16)`, `j = 0..15`, and the y half
(dims 32..63) pairs `(32 + j, 48 + j)`. q AND k get rope; v does not.
The `rope_2d_halfsplit_f32` kernel instead pairs `(d, d + head_dim/2)`, so the
Q/K head dims are uploaded through `vision::axial_perm`, which puts the gemma
pair on the kernel pair (kernel quarters come out `[x, y, x, y]`). The kernel
reads one angle per side; `axial_perm` is what makes the two sides agree.

### Pooler + tail (L624–689, L2008–2050)
Zero pad patches; `bin = floor(x/3) + (patch_w/3)*floor(y/3)`;
`tokens[bin] = mean of its 9 patches` (one-hot/9 matrix product, fp32);
**scale by sqrt(768)**; drop bins whose 9 patches were all padding → exactly
N_tokens rows. NO final tower norm. Then the embedder:
`rmsnorm_scaleless(h, eps=1e-6) @ W_proj.T` → `[N,512]`
(`embed_vision.embedding_projection [512,768]`, no bias).

## AUDIO TOWER

Config: hidden 1024, 12 layers, 8 heads × 128, act silu, conv channels [128,32],
lconv kernel 5, residual_weight 0.5, chunk 12, context_left 13, right 0,
logit cap 50, invalid logit −1e9, use_clipped_linears TRUE, rms_eps 1e-6,
gradient_clipping 1e10, output_proj_dims 1536. clip = min(1e10, dtype max).

### Mel frontend (feature_extraction_gemma4.py L50–296)
pad_left 160 (semicausal, not centered); frames: unfold(size 321, step 160),
drop last sample → 320; periodic Hann `0.5 − 0.5·cos(2πn/320)`;
rfft n=512 (right zero-pad); **magnitude** (not power);
mel = mag @ filterbank (257→128, HTK, norm None, 0–8000 Hz);
`log(mel + 1e-3)`; no per-bin normalization; padded frames zeroed.
**`L' = ceil(L/128)·128`** (the extractor's `pad_to_multiple_of=128` zero-pads
the waveform FIRST — omitting it loses a frame; e.g. 0.5 s: 8000 → 8064 gives
nf 50, not 49). Then `nf = (L' + 160 − 321)//160 + 1`. Frame validity:
mask[i*160 + 320]. Variable lengths: `L'` only.

### Conv subsample (L357–413)
Input `(T,128)` as `(1,1,T,128)` (H=time, W=freq). Two stages of
`Conv2d(k=3,s=2,p=1,bias=False) → LN(affine, no bias, eps 1e-6) → ReLU`,
time mask applied pre-conv and halved each stage. 128→64→32 freq.
Flatten `[T', F'*C]` with `index = f*32 + c` → 1024;
`input_proj_linear` (1024→1024, no bias), **no activation**.
`M = valid subsampled tokens` (all-valid: `t=nf; t=(t-1)/2+1 twice`) —
reference-verified: 0.1 s → 3, 0.5 s → 13, 1.0 s → 25, 2.0 s → 50,
11.2 s → 280, 15 s → 375. **No cap** (`audio_seq_length: 280` is metadata, not
a truncation).

### Relative positions (L218–246)
context = 12+13−1+0 = 24; 512 timescales, inv = `exp(i · −ln(10000)/511)`;
pos_ids = [12..0] (13 values); embed = `[sin(pos·inv) | cos(pos·inv)]` (13,1024);
project once via `relative_k_proj` (1024→1024, no bias).

### Attention (L249–355)
q = `(q_proj(h)·head_dim^-0.5·softplus(per_dim_scale[128]))·ln2⁻¹` — stored
scales: q_scale = `head_dim^-0.5/ln2`, k_scale = `ln(1+e)/ln2`; per_dim_scale
through softplus. Block the seq by chunk 12 (zero-pad tail); K/V context = pad
(12 left, 11 right), unfold 24; logits `ac + rel_shift(bd)`; softcap
`w/50 → tanh → ×50`; invalid = −1e9; softmax fp32. rel_shift = Transformer-XL
skew (pad 25, slice 12·24).

**The mask is INVERTED, and the inversion is load-bearing.** The 5-D mask is
DEFINED as `valid_key ∧ 0 ≤ q − kv < attention_context_left − 1` (keys
kv ∈ [q−11, q] of the valid rows; `create_bidirectional_mask(...)` with
`and_mask_function=sliding_window_mask_function((12, 0))`, `valid` = the frame
mask after its two `[::2]` halvings), but `Gemma4AudioAttention.forward` (L344)
applies it with the Bool-mask idiom
`masked_fill(attention_mask.logical_not(), −1e9)`, and the reference generator
forces `attn_implementation="eager"`, whose mask interface
(`masking_utils.eager_mask`, L541: "the masks for eager attention are simply
boolean mask from sdpa, casted to 0 and −inf") converts the boolean mask to an
ADDITIVE float one before the attention sees it. `logical_not` then fires on
exactly the positions the mask means to INCLUDE. A slot therefore keeps its
score iff

```
0 <= kv < t1                              # inside the 4-D mask's real range
  AND NOT (kv < M AND 0 <= q - kv < 12)   # the mask's inclusion, INVERTED
```

where `t1` is the whole subsampled length and `M` the valid prefix of it
(`_convert_4d_mask_to_blocked_5d` pads with `value=False`, which is 0 in the
additive mask, so `kv < 0`, the tail block's `kv >= t1` and the 12-wide context
pads are still filled). The mask's QUERY axis contributes nothing — a dumped
4-D row is `valid_key ∧ window` for the padded-tail rows too, not an
all-invalid (or all-valid) row. Consequences, all reference-verified:
- a valid query reads its FUTURE keys (any `kv > q`) and every past key older
  than 11 — but nothing in between — and it reads the padded TAIL rows,
  `kv ∈ [M, t1)`, unconditionally (no window can include them);
- an INVALID (tail) query inverts to "the tail rows alone", and the tower must
  still compute it: that row becomes the next layer's key/value row. So the
  tower runs all `t1` rows and returns the first `M` — truncating to `M` before
  the forward is ~15 % rel-L2 wrong on the clips where `M < t1` (≈1 clip length
  in 4; 1 409 samples and 8 449 samples repro it, the latter with `M = 13 <
  t1 = 14`, where a query-validity term in the rule is 2e-3..5e-3 wrong);
- the only relative-position row that can reach a survivor of a valid query is
  `R[0]` (position/distance `ctx/2 = 12`): every shorter past distance is
  inside the window, a valid future survivor sits at `j < i` where rel_shift
  gives 0, and the tail survivors are at `dist <= -M`;
- a query in the first block can have NO survivor at all (`q = 11` of a
  13-row clip: all 12 valid slots are inside the window) — the softmax is then
  uniform over the block context, and the out-of-sequence values it averages
  are the context padding's ZEROS;
- `sdpa` (the standalone default) does NOT invert: it keeps the window. A
  standalone `Gemma4AudioModel(...)` run — the obvious way to build an oracle —
  is therefore NOT the reference, and disagrees with it by 3e-2…1.4e-1. Build
  it with `attn_implementation="eager"`, or use `examples/audio_lab.rs`, which
  runs this crate's frontend + tower for one WAV straight from the checkpoint
  (safetensors dir or `.vl` pack, no daemon, no text tower) and dumps the
  log-mel and the rows (`--layers N` for a layer-by-layer bisect).

### Layer (L525–577)
```
h = macaron_ffn1(h)            # each ffn: clamp → rmsnorm → linear → silu
                               #   → linear → clamp → rmsnorm → ×0.5 → +residual
r = h; h = clamp→rmsnorm(norm_pre_attn); h = attention(h)
h = clamp→rmsnorm(norm_post_attn) + r
h = lconv1d(h)                 # clamp→rmsnorm→linear_start(2048)→glu(a·σ(b))
                               #   →causal dwconv k5 (left pad 4)→clamp→rmsnorm
                               #   →silu→linear_end→+residual
h = macaron_ffn2(h)
h = clamp→rmsnorm(norm_out)    # NOT residual-added
```
Tail: `output_proj` (1024→1536, bias) — no activation — then the audio embedder
(`rmsnorm_scaleless(1536)` → `embed_audio.embedding_projection [512,1536]`).
Placeholder text run: `boa(256000) + audio_token(258881)×M + eoa(258883)`.

## Cross-cutting
- Every clippable linear: clamp input to [input_min,input_max], output to
  [output_min,output_max] (audio only; vision tensors are absent).
- Vision `use_clipped_linears=False`; audio True — gate on config, not hardcode.
- Image markers: `boi(255999) + image_token(258880)×N + eoi(258882)`.
- Video: the SAME vision tower, one sequence per FRAME (`attention_mask =
  ~padding_positions` — frames never attend each other), resized against the
  VIDEO processor's budget (`video_processor.max_soft_tokens = 140` →
  `max_patches = 1260`, NOT the image processor's 280/2 520).
- Video markers: one `<|video|>` (258884) per VIDEO (not per frame), expanding
  to `boi(255999) + video_token(258884)×S + eoi(258882)` PER FRAME, `S` =
  that frame's soft tokens, blocks concatenated with no timestamps
  (`add_timestamps=false`) and no separators.
- Video sampling: a decoded frame list carries no capture rate, so HF skips
  its `fps` pass (`metadata.fps is None` → `fps=None`) and applies only the
  `max_frames` budget: ≤32 frames all kept, more reduced by
  `np.linspace(0, total-1, 32, dtype=int)` (`overflow_strategy "uniform"`).
  The caller's frames keep whatever rate they were extracted at; the
  32-frame budget is the checkpoint's, not the caller's.
- Invalid-audio rows are dropped BEFORE splice: placeholder count == valid M.
- Mean-pool covers all rows incl. media rows (`include_prompt: true`).
