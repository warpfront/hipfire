# One `GatedMlp` op for the dense gated FFN forms (#666 follow-up)

Status: **planned**. This is a design for maintainer review. No code changes until the note is accepted.
Inputs: beta `1ee6815c5`; PR heads #795 `b45a9ef6c` (Qwen3.5), #877 `5be1920a4` (LFM2),
#884 `e8828aa43` (MiniMax), #881 `a9483b4c9` and #886 `52ed1ace3` (Glimmer).

## Answer

Yes. The four forms can become one operation without changing any kernel or reduction order,
**provided the op names its projection ladder in an explicit field**.

One op gives one declaration, one validation, one scratch contract, one `Step` variant and
one `OpKind`. It also fixes the stage order: optional pre-norm, gate/up, activation, down,
output epilogue. What it cannot unify is the kernel ladder that the batched rows take. The
existing ladders give different bits for the same weight format:

| Format, `rows > 1` | sandwich `Ladder::Wide` / `Route` (Gemma, Glimmer) | LFM2 lane batch (`batched_proj`) | Qwen3.5 rows (`SwigluFfnBatch`) |
|---|---|---|---|
| MQ4G256 / HFQ4G256 | `gemm_hfq4g256_batched_lmhead`: FP16 WMMA on gfx11/gfx12, documented as not identical to the scalar GEMM | scalar `gemm_hfq4g256` (byte-exact with AR) | fused gate/up keys, IU4/A8/FP8 producers |
| Q8_0 | `Wide`: gfx1151 4-warp WMMA from 64 rows. `Route`: `gemm_q8_0_batched_chunked` | `gemm_q8_0_batched_chunked` | `FusedGateUpQ8_0` when `q8_wmma_arch` |

Qwen4's `project_weights` is a fourth ladder: a shared F16 rotation for MQ4/MQ6 V2 above 8 rows,
x4 GEMVs for 2–8 verify rows, and gfx942 BF16 MFMA. A single body that picks kernels only from
dtype, arch and rows would change kernels for at least LFM2 lanes, Gemma rows and Qwen3.5 rows,
which the constraint rules out. The ladder therefore stays a field. Each `MlpLadder` arm is an
existing body, moved verbatim. Collapsing two ladders into one is separate work, done one pair at
a time behind a parity gate (see "Collapse candidates").

**Correction to the issue's inventory:** MiniMax (#884) has no dense gated FFN. Every MiniMax layer
is attention + MoE. The gate/up → SiLU → down chains in `hipfire-arch-minimax/src/forward.rs` are
routed-expert bodies: the EP hand MoE `minimax_moe_block_ep` and the `forward_batch` expert loops.
They belong to sealed MoE (G14), not to `GatedMlp`.

## The four forms

### 1. `layer_ops::DenseFfnOp` (beta; Qwen4)

- **Users:** trunk layers (`gpu_forward.rs`), the MTP head (`MtpFfnCall::Dense`) and the NPU trunk.
  Only the developer profile `tiny-dense-dev` has a dense FFN. No production Qwen4 checkpoint has one
  (`reference_oracle/QWEN4_PROFILES.md`).
- **Pre-norm:** none. `HyperRead` produces the input (`hc_mixed`).
- **Gate/up:** one `project_weights` call over both weights.
- **Activation:** `silu_f32`, then `mul_f32`.
- **BF16 recipe** (`DenseFfnRecipe::SiluBf16Source`): BF16 round trip after gate, after up, after SiLU,
  after the product, and after down.
- **Down:** `project_weight` into `output`. No residual; `HyperWrite` consumes the output.
- **Rows:** `rows`. The route comes from `gpu.route_rows(rows)`: decode, 2–8 verify rows, prefill.

### 2. `sandwich::SandwichMlpOp` (beta: Gemma 4; #881/#886: + Muse Glimmer)

Equation: `x = residual + post_norm(down(act(gate(n)) * up(n)))`, `n = pre_norm(x)`.

- **`SandwichRoute::Wide` (Gemma):**
  - Decode, MQ4G256 pair, not calibrating: `fused_rmsnorm_rotate_mq` + `fused_gate_up_hfq4g256`.
    Otherwise `rmsnorm_f32` + two `gemv`.
  - Rows: `rmsnorm_batched`, then gfx1100's opt-in `FusedGateUpQ8_0`, else `gemm_rows(Ladder::Wide)`.
  - Under greedy EAGLE, Q8 rows take the scalar kernel and the fused post-norm residual is off
    (`EAGLE_STRICT`).
- **`SandwichRoute::Steps(StepsRoute)` (Glimmer):**
  - Decode: `RmsnormAutomatic` + two `Step::Gemv` in one `execute_steps` window, so the dispatch
    fusion table (`GATE_UP2`) groups them.
  - `shared_rotation` fuses norm + FWHT and feeds `Prerotated` input.
  - Rows: norm (fused with the rotation when shared), then `gemm_rows(Ladder::Route(route))`. On gfx12
    the residual WMMA GEMMs serve MQ4/HFQ4 prefill.
- **Activation:**
  - `GeluTanh`: `gelu_tanh_f32` + `mul_f32`.
  - `Silu`: `silu_mul_f32`. With `fused_silu_rotate` and an MQ4G256/MQ6G256 down weight, it uses
    `RotationVariant::WithSwiGLU` instead (fused SiLU·mul + FWHT, AWQ-aware, batched arm).
- **Down and output:** `gemv` / `gemv_prerotated` / `gemm_rows` into `out`, then `post_norm_residual`:
  - decode: fused `rmsnorm_residual_add_f32`;
  - rows: post-norm, residual restore, add;
  - without a post-norm: a plain add;
  - optional output stream `out` (no residual copy).
- **Embedding:** `ParallelMoeMlpOp` (Gemma 26B-A4B) embeds the op. It runs the body only and
  combines with the routed experts itself.
- **Rows:** `SandwichStream { rows, route: RowsRoute }`. The route alone picks decode or batched kernels.

### 3. `hybrid::SwigluFfnOp` (#795 head only; beta keeps this code in `hipfire-arch-qwen35`)

- **Users:** dense layers of the Qwen3.5/3.6/3.8 hybrids, the MTP head's dense FFN, and the
  super-op `ForwardBindings`. The super-op calls the stages separately: `PROJ_GATE_UP` → `project`,
  and the residual opcode → `down`.
- **Decode gate/up:** `normed_projection`, i.e. `RmsnormAutomatic` with the first weight's rotation
  plan (FWHT, AWQ, Givens) plus two `Step::Gemv`. The fusion table rewrites them.
- **Decode down:** `swiglu_down_residual`. Arms:
  - MQ formats: `fused_silu_mul_rotate_mq{_awq}` + GEMV `WithSwiGLUResidual` (`x += W·h`);
  - HFQ formats: `silu_mul_f32` + GEMV `WithResidual`;
  - everything else: `silu_mul_f32` + GEMV + `add_inplace_f32` on a per-call temporary;
  - a CPU-offload arm (`host_mapped_cpu_capable`).
- **Rows (`SwigluFfnBatch`):**
  - Norm producer ladder: A8, IU4, gfx12 fused quant, gfx12 FP8 stream, exact-F16 rotate, F32 fused
    norm + rotate.
  - Packed-MQ4 opt-in.
  - F1-lite: the gate/up GEMM emits `h = silu(gate)·up` (`FfnGateOutput::{A8H, Iu4H, Fp8H}`), or it
    emits the down projection's A4 sidecar (`Iu4A4`).
  - `DenseBatchMath::SingletonWmma`: exact verify math.
  - Down: producers, then a residual GEMM or `BatchEpilogue::Partial` (tensor-parallel partial).

### 4. LFM2 primitive chain (#877)

- **Decode:** `RmsnormAutomatic(RotationPlan::None)`, `Gemv(Raw)` w1, `Gemv(Raw)` w3,
  `Step::SiluMul`, `Step::GemvResidual` w2. The input stays `Raw` on purpose: `Prerotated` admits the
  fusion table and changes numerics (HIPF-14).
- **Prefill and verify:** per-token decode (`spec_impl::verify_block` runs sequentially).
- **Multi-lane batch** (`forward_decode_batch_prepared_lfm`, still hand code, HIPF-75):
  `rmsnorm_batched`, then `batched_proj`, then `silu_mul_f32`, then `batched_proj`, then
  `add_inplace_f32`. `batched_proj` arms:
  - Q8: chunked GEMM;
  - HFQ4: scalar `gemm_hfq4g256`;
  - MQ4: rotate + scalar `gemm_hfq4g256`;
  - BF16: MFMA, with the staging hoisted once for w1/w3.

## Why the op spans pre-norm to output

Existing routes fuse across every stage boundary:

| Fused boundary | Where |
|---|---|
| norm + rotation (+ quant producer) for gate/up | Gemma decode MQ4, Glimmer `shared_rotation`, Qwen3.5 decode and every row producer |
| gate/up GEMM + activation (emits `h`, no `gate`/`up` planes) | Qwen3.5 F1-lite, gfx12 FP8 `silu_h` |
| gate/up GEMM epilogue + down input producer | Qwen3.5 `Iu4A4` |
| activation + rotation (+ quant) for down | Qwen3.5 decode and rows, Glimmer `fused_silu_rotate` |
| down + residual | Qwen3.5 `WithSwiGLUResidual` / residual GEMMs, LFM2 `GemvResidual` |
| post-norm + residual | Gemma/Glimmer decode `rmsnorm_residual_add_f32` |

`Gemv` + `SiluMul` + `Gemv` steps can express LFM2 only. One op that owns the whole chain can
express all four forms, because each ladder decides internally where the boundaries fall. Qwen4 is
the case with no pre-norm and a plain output.

## Proposed op

The op lives in `hipfire-dispatch/src/pipeline/gated_mlp.rs` (planned). `Activation` moves there
from `sandwich`.

```rust
pub struct GatedMlpOp<'a> {
    pub rows: usize,
    /// Decode (exactly one row), Prefill or Verify. With `ladder`, picks the kernels.
    pub route: RowsRoute,
    pub hidden: usize,
    pub intermediate: usize,
    /// Residual stream (or, without a pre-norm, the FFN input): F32 `[rows, hidden]`.
    pub x: &'a GpuTensor,
    /// RMS pre-norm folded into the gate/up producer; `None` = `x` is the input (Qwen4).
    pub pre_norm: Option<RmsNorm<'a>>,          // { weight, eps }
    pub gate: WeightRef<'a>,
    pub up: WeightRef<'a>,
    pub down: WeightRef<'a>,
    pub activation: Activation,                 // GeluTanh | Silu
    /// Storage boundaries of the source equation.
    pub storage: MlpStorage,                    // F32 | Bf16Source
    pub output: MlpOutput<'a>,
    pub ladder: MlpLadder<'a>,
    pub scratch: GatedMlpScratch<'a>,
}

pub enum MlpOutput<'a> {
    /// `out = mlp(x)`; the caller combines (Qwen4 HyperWrite, Gemma parallel dense+MoE).
    Write(&'a GpuTensor),
    /// `x += mlp(x)` through the ladder's residual-accumulating down kernel
    /// (Qwen3.5 `WithSwiGLUResidual` / residual GEMMs, LFM2 `GemvResidual`).
    Residual,
    /// Tensor-parallel partial (`BatchEpilogue::Partial`): zeroed `partial` += down(h).
    Partial(&'a GpuTensor),
    /// Sandwich epilogue: `dst = x + post_norm(mlp(x))`, `dst` = `stream` or `x`
    /// (via `residual`). `norm: None` is a plain add after a plain down projection.
    PostNorm {
        norm: Option<&'a GpuTensor>,
        eps: f32,
        out: &'a GpuTensor,
        residual: &'a GpuTensor,
        stream: Option<&'a GpuTensor>,
    },
}

pub enum MlpLadder<'a> {
    /// `layer_ops::project_weights` (shared F16 rotation, x4 verify GEMVs, gfx942 MFMA).
    Trunk,
    /// `SandwichRoute::Wide`: one-row fused MQ4 gate/up and post-norm residual,
    /// `gemm_rows(Ladder::Wide)` for rows, `EAGLE_STRICT` inside.
    SandwichWide,
    /// `SandwichRoute::Steps`: decode as one generic `Gemv` step window (fusion table),
    /// rows on `gemm_rows(Ladder::Route(route))`.
    Steps(StepsRoute),
    /// The hybrid product ladder; `Some` exactly when `route.batched()`.
    Product(Option<ProductRows<'a>>),       // x_rot_f16, hidden_f16, chain_verify,
                                            // q8_wmma_arch, lean, math: DenseBatchMath
    /// LFM2 lane batch: scalar batched GEMMs (`batched_proj`). Rows only.
    Lanes,
}

pub struct GatedMlpScratch<'a> {
    /// `[rows, intermediate]` each; `act` may alias `gate` (the Trunk recipe runs in place).
    pub gate: &'a GpuTensor,
    pub up: &'a GpuTensor,
    pub act: &'a GpuTensor,
    /// `[rows, hidden]`: the normed (plain) activation; may alias `x_rot` on rows.
    pub normed: &'a GpuTensor,
    /// `[rows, max(hidden, intermediate)]`: FWHT basis.
    pub x_rot: &'a GpuTensor,
    /// BF16 staging `[rows, max k]` for BF16 weights on rows; `None` allocates per call
    /// (the existing sandwich behavior).
    pub bf16: Option<&'a GpuTensor>,
}
```

`Residual` and `PostNorm { norm: None }` both compute `x + mlp(x)`, but with different kernels.
LFM2 lets the down GEMV accumulate (`GemvResidual`); Glimmer's drafter adds after a plain GEMV.
Each keeps its own launches, so they stay separate variants.

### Admission (checked by `validate_for_gpu`, before the first launch)

| Ladder | `pre_norm` | `activation` | `storage` | `output` | routes |
|---|---|---|---|---|---|
| `Trunk` | `None` | `Silu` | `Bf16Source` | `Write` | Decode, Prefill, Verify |
| `SandwichWide` | `Some` | `GeluTanh`, `Silu` | `F32` | `PostNorm`, `Write` (parallel MoE body) | Decode, Prefill, Verify |
| `Steps` | `Some` | `GeluTanh`, `Silu` | `F32` | `PostNorm`, `Write`, `Residual` (decode only; new arm that issues LFM2's `Step::GemvResidual`) | Decode, Prefill, Verify |
| `Product` | `Some` | `Silu` | `F32` | `Residual`; `Partial` on rows | Decode, Prefill, Verify |
| `Lanes` | `Some` | `Silu` | `F32` | `Residual` (separate add, as today) | batched only |

Any other combination is refused. No arm gets a kernel or a fallback it does not have today.
Extent checks are taken over from `DenseFfnOp::validate_layout` and `SandwichMlpOp::validate_for_gpu`.
`SwigluFfnOp` has none today; it gets the same checks. The implicit scratch stays implicit in the
first cut: Qwen3.5 decode's `gpu.scratch.mq_x_rot` and the per-call temporary of its generic
GEMV + add arm. Declaring them is a follow-up.

`GatedMlpOp::{project, down}` stay public for the `Product` decode arm, because Qwen3.5's super-op
`ForwardBindings` (EP/TP decode) runs the two halves as separate opcodes.

## Route map

| Family, route | Ladder | `pre_norm` | `activation` / `storage` | `output` | Fields the arch crate sets |
|---|---|---|---|---|---|
| Qwen4 trunk, MTP, NPU trunk: decode, verify, prefill | `Trunk` | `None` | `Silu` / `Bf16Source` | `Write(ffn_output)` | gate/up/down, `rows`, `intermediate`, `rotation` → `x_rot`, `moe_{gate,up}_batch` → `gate`/`up`, `act = gate` |
| Gemma 4 dense layers: decode, prefill, EAGLE verify | `SandwichWide` | `Some(pre_ffn_norm)` | `GeluTanh` / `F32` | `PostNorm { Some(post_ffn_norm), post_eps }` | stream tensors as today |
| Gemma 4 26B-A4B dense half | `SandwichWide` | `Some` | `GeluTanh` / `F32` | `Write(mlp.out)` inside `ParallelMoeMlpOp` | unchanged |
| Glimmer target: decode, prefill, DFlash verify | `Steps { shared_rotation, fused_silu_rotate }` (both env, default on) | `Some` | `Silu` / `F32` | `PostNorm { Some(post_feedforward_layernorm) }`, x/out ping-pong `stream` | as today |
| Glimmer drafter | `Steps { false, false }` | `Some` | `Silu` / `F32` | `PostNorm { None, stream: None }` | as today |
| Qwen3.5 dense layers + MTP head: decode | `Product(None)` | `Some(ffn_norm / attn_post_norm)` | `Silu` / `F32` | `Residual` | `plain` → `normed`, `x_rot`, `gate_ffn`/`up`/`ffn_hidden` |
| Qwen3.5: prefill, DFlash chain verify, TP | `Product(Some(..))` | `Some` | `Silu` / `F32` | `Residual`, or `Partial` on TP | `ProductRows` from `SwigluFfnBatch` |
| LFM2 decode (prefill and verify are per-token decode) | `Steps { false, false }` | `Some(ffn_norm)` | `Silu` / `F32` | `Residual` | `tmp` → `normed`, `dense_{gate,up,act}` |
| LFM2 multi-lane batch (HIPF-75) | `Lanes` | `Some` | `Silu` / `F32` | `Residual` | `ffn_tmp`, `proj_rot`, `bf16_scratch` |

A MiniMax row is deliberately absent; see the correction under "Answer".

For LFM2 decode on `Steps { false, false }` with `Residual`, the launch sequence must be
`RmsnormAutomatic` (plain = out = `normed`), `Gemv(Raw)` ×2 in one window, `silu_mul_f32`
(the kernel `Step::SiluMul` runs), and `Step::GemvResidual`. That is today's LFM2 sequence.
The launch-trace oracle below has to show it.

## Step graph and mesh split

- Append `OpKind::GatedMlp` with the next free ID at landing time. Beta ends at 26; DS4 #883/#889
  will take more IDs first.
- `DenseFfn = 18` and `SandwichMlp = 20` are retired, never reused (frozen-ID rule in
  `step_graph.rs`). `SwigluFfn` never had an ID: #795 predates the step graph.
- `mesh_split.rs` tests that build `OpKind::DenseFfn` move to `GatedMlp`.
- The `SplitRule` stays `Opaque`. A tensor-parallel rule is the natural next step and is the reason
  `Partial` is part of the op, but it is not part of this design. Under that rule, gate/up split by
  column, down by row, and the output is a partial.

## Migration (one PR per row; each one deletes the old op/variant, no shims)

| Slice | Precondition | Change |
|---|---|---|
| 1. Sandwich → `SandwichWide` / `Steps` | #881/#886 on beta | add `GatedMlpOp`, move `dense_mlp` / `steps_mlp` / `activate` / `post_norm_residual` arms; Gemma, Glimmer target and drafter, `ParallelMoeMlpOp` |
| 2. Qwen4 → `Trunk` | slice 1 | move `execute_dense_ffn`; delete `DenseFfnOp`, `DenseFfnRecipe` |
| 3. LFM2 decode → `Steps` + `Residual` | #877 on beta, slice 1 | add the decode `Residual` arm to `Steps`; replace `ffn_steps`' five dense steps |
| 4. Qwen3.5 → `Product` | #795 rebased and on beta | move `SwigluFfnOp` decode + `execute_swiglu_ffn_batched`; delete `SwigluFfnOp`/`SwigluFfnBatch` |
| 5. LFM2 lanes → `Lanes` | HIPF-75 | move `batched_proj`'s dense FFN use |

## Parity oracles a migration needs

Every slice is a code move. The primary gate is **launch-trace identity**:
`rdna_compute::profile::start/stop` per-kernel name sequence plus grid, per (route × weight format),
on beta and on the migrated head. A trace difference means a kernel changed, and the slice stops.
Output digests come second.

| Family | Oracle | Fixtures (`~/.hipfire/models` on halo, sha256) | Not coverable on halo (gfx1151) |
|---|---|---|---|
| Gemma 4 | HIPF-86 `gemma/` rows: greedy transcripts + top-16 logit traces, 13 rows, incl. long prompts on the gfx1151 Q8 4-warp ladder; EAGLE strict on 12B | E2B q8 `gemma4-eseries/gemma4-e2b-it-pr439-q8.hfq` `4815a332…a260` (4975368774 B); `gemma4-12b.mq4` `4ceb57b5…81c3` (8914591328 B); `gemma4-26b-a4b-requant.mq4` `7b416c67…f1c2` (15632064572 B); `gemma4-26b-a4b.q8-experts-reference.hfq` `ed82786b…668f` (26848688188 B); drafter `gemma4-12b-assistant.q8f16.hfq` `2907b117…bb0f` | gfx1100 `FusedGateUpQ8_0` opt-in |
| Muse Glimmer | `rows_parity_dump.rs` route oracle (HIPF-77): 130 digests (prefill through 1,280/2,048, decode, verify with partial accepts, device capture, five drafter windows); once more with `HIPFIRE_GLIMMER_SHARED_ROT=0 HIPFIRE_GLIMMER_FUSED_SILU_ROTATE=0` for the plain arms | `muse-glimmer-30b.mq4` `87bc776a…c052` (18614785024 B); `.mq4r` `47ccfccd…7503` (16261223424 B); `muse-glimmer-30b-dflash.mq4` `6fc0988e…3d2d` (1357990400 B) | gfx12 residual / fused QKVZA rows GEMMs |
| LFM2 | `lfm2_route_dump.rs` (HIPF-86 `parity-lfm-mm/`): tokens, logits, conv state, KV, eager + hipGraph; lanes (slice 5) need a B ∈ {2, 4, 8} lane-digest harness | `lfm2.5-350m.q8` `7cd38949…bc3c` (382103280 B); `lfm2.5-350m.mq4` `4885d1ce…a60f` (229474032 B); `lfm2.5-1.2b.mq4` `a7aeba87…6533` (698641072 B); `lfm2.5-8b-a1b.mq4` `07da9942…4148` (4659505584 B, dense layers); `lfm2.5-vl-3b.mq4v2.hfq` `ce009cd0…d5d2` (2307116928 B) | BF16 teachers (gfx942 MFMA) |
| Qwen3.5 family | HIPF-9 A/B rows: decode, prefill, DFlash chain verify, MTP head, VL; `SingletonWmma` batched == singleton under `HIPFIRE_CB_VERIFY_FP8`; record which producer arms resolve on gfx1151 from the trace | `qwen3.5-9b.mq4` `ba83acf5…28ad` (5313750016 B); `qwen3.8-27b.mq4-xt` `9f91556f…eeb7` (14980361216 B); `qwen38-27b-dflash-mq4.hfq` `d0a74a23…01bd` (1209603072 B); `qwen3.8-27b.mtp` `f0d46d07…d32e` (225716224 B) | gfx12 FP8 stream / fused quant, gfx1100 packed MQ4 and V2C, TP `Partial` (multi-GPU): hiptrx/hipx runbooks |
| Qwen4 | `tiny-dense-dev` per-op dumps (`reference_oracle/generate_fixtures.py --tiny-profile tiny-dense-dev --seed 3801 --screen-qsa-ties`, plus `--no-ple`); FFN output per layer via `chunk_dump` on beta vs head; MTP step 0 | **fixture-missing on halo**: generated by the pinned CPU venv named in `QWEN4_PROFILES.md` | gfx942 BF16 MFMA |

Each row records head SHA, binary md5 and the command under `~/mesh-validation/<ISSUE>/` (halo-hearth
evidence rules).

## What this does not change

No kernel, no launch count, no performance. The ladder count stays five. The gain is structural:

- one `Step` variant replaces three plus LFM2's five-step chain;
- one validation and one scratch contract;
- one `OpKind`, the place a later tensor-parallel `SplitRule` attaches;
- arch crates stop sequencing FFN stages (LFM2 decode, then Qwen3.5 and LFM2 lane hand code).

## Collapse candidates (after the migration; each one changes or removes a ladder, so each is its own parity-gated PR)

- **`Lanes` → `gemm_rows` with a new `Ladder::Scalar`.** Verified from code: Q8 is the same kernel
  (`gemm_q8_0_batched_chunked`) on both, and the MQ4/HFQ4 arm (scalar `gemm_hfq4g256`) would become
  the new variant. BF16 differs only in the staging count: one hoisted conversion vs one per weight,
  so the bytes are the same and the trace has one launch more.
- **Gemma `SandwichWide` decode MQ4 pair vs `Steps { shared_rotation: true }`.** Unverified: is the
  fusion table's `FusedGateUpHfq4G256` the same launch as the direct `fused_gate_up_hfq4g256`, behind
  the same norm producer? If the launch trace is identical, Wide decode gate/up can go.

## Open questions for the maintainer

1. Accept `ladder` as an op field that names today's ladders, rather than waiting until the ladders
   are collapsed?
2. Retire `OpKind` 18/20 for one new `GatedMlp` ID?
3. LFM2 lanes are neither prefill nor verify. Should they get a `RowsRoute::Lanes`, or should the
   `Lanes` ladder only require `route.batched()`?
4. Out of scope unless asked: Gemma's `PerLayerInput` (`gelu(gate(x)) · ple`, no up projection),
   DS4's shared expert (`pipeline::mla::ExpertFfn`), and the hand dense FFNs in
   `hipfire-runtime/src/llama.rs`, `hipfire-arch-qwen2` and `hipfire-arch-cohere2moe`, plus the
   vision-tower MLPs.
