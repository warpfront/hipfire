# Quantization

How hipfire stores weights and KV cache (design / math). For the operator
workflow (`hipfire quantize` / `hipfire-quantize`), see [QUANTIZE.md](QUANTIZE.md).
Format-specific specs: [quant-formats/mq4-v2.md](quant-formats/mq4-v2.md),
[quant-formats/mq-v2-family.md](quant-formats/mq-v2-family.md),
[quant-formats/ladder.md](quant-formats/ladder.md),
[quant-formats/qt-register.txt](quant-formats/qt-register.txt),
[quant-formats/hfp4.md](quant-formats/hfp4.md).

Truth labels follow [INDEX.md](INDEX.md). Mutable inventories (CLI flags, registry
KV recommendations, arch admissions) are owned by their source pages — this file
describes the math and wire contracts.

## Weight families at a glance

| Family | Rotation | Element | Group | Role |
|---|---|---|---|---|
| **HFQ** | none | INT asymmetric | G256 (G128 variants exist) | Dense Llama / Mistral / Gemma / plain Qwen |
| **MQ (MagnumQuant)** | offline FWHT-256 | INT asymmetric (or Lloyd codebook) | G256 | Qwen 3.5+ hybrid path; FWHT baked into weights |
| **HFP / MFP** | optional offline FWHT | OCP E2M1 FP4 | G32 rows | RDNA-oriented FP4; MFP = HFP + FWHT |
| **PARO** | pairwise Givens | AWQ W4 | G128 or G64 (`ParoQ4G128` / `ParoQ4G64`, from `group_size`) | Imported ParoQuant/AWQ probe path |
| **Q8F16** | none | INT8 + f16 scale | G32 | Embeddings, routers, precision-sensitive 2D |

Production defaults for day-to-day use:

| CLI / extension | Wire type | Nominal bits | Stored bpw (bytes × 8 ÷ weights) | Bytes / 256 el | When |
|---|---|---:|---:|---|---|
| `mq4` / `.mq4` | **`MQ4G256V2` (qt 44)** | 4 | 4.25 | 136 (8 hdr dual-fp16 per-128 + 128 data) | Qwen 3.5+ safetensors default; **`--format mq4` → V2** |
| `mq4v1` / `mq4g256` / `magnum` | `MQ4G256` (qt 13) | 4 | 4.25 | 136 (8 hdr f32 scale+min + 128 data) | Legacy Magnum v1 only — not the `mq4` alias |
| `mq6` / `.mq6` | `MQ6G256` (qt 15) | 6 | 6.25 | 200 | Qwen 3.5+ higher quality (v1 body) |
| `mq6v2` … `mq2v2` | V2 family qt 47/48/49/50 | 6…2 | 6.25 / 5.25 / 3.25 / 2.25 | 200/168/104/72 | Dense Magnum V2 ladder bodies — see below |
| `hf4` / `.hf4` | `HFQ4G256` (qt 6) | 4 | 4.25 | 136 | Dense GGUF default (**GGUF weight input deprecated since 0.4.0, removal in 0.5.0**); Llama-style dense |
| `hf6` / `.hf6` | `HFQ6G256` (qt 8) | 6 | 6.25 | 200 | Dense, higher quality |
| `mq3` / `.mq3` | `MQ3G256` (qt 17) | 3 | 3.25 | 104 (8 hdr + 96 data) | Sub-4-bit bandwidth; quality-sensitive |
| `q8` | `Q8F16` (qt 3) | 8 | 8.5 | **34 B / 32 el** (2 B f16 scale + 32×i8; 272 B / 256 el) | Reference / debug / protected tensors |

**Nominal bitwidth is not bpw.** The “bits” column is the payload width only.
Stored bpw is the exact group arithmetic and includes the per-group header:
every G256 affine row above spends 8 header bytes per 256 weights (+0.25 bpw
over nominal), and Q8F16 spends 2 scale bytes per 32 weights (+0.5 bpw). These are
wire/codec costs only. Whole-model bpw is higher still (embeddings, lm_head and
role lifts sit above the base width; see [ladder.md](quant-formats/ladder.md)), and
none of these figures is a performance or quality claim.

Under standard MQ/HF/HFP/MFP recipes, embeddings are forced to `Q8F16`
(Q4-grade embedding error compounds). 1D norms / scales stay F16. Direct
recipes such as `q4k-all` intentionally bypass that embedding rule.

### Magnum V2 family (qt 44 / 47–50 / 52), MQ4C (qt 45) and MQ4G128V2 (qt 53)

Magnum V2 uses a neutral-size **8 B dual half `fp16 scale + fp16 zero` header
per 128 weights**, with payload packing unchanged from the matching v1 bit
width. Group strides: bits **2/3/4/5/6 → 72/104/136/168/200** B per 256
elements. MQ4C is separate: one per-256 fp16 scale+zero pair at `[0..4)`,
zero padding at `[4..8)`, and the 4-bit payload at `+8`. qt=52
`MQ4G256V2L` / `DType::MQ4G256V2Lloyd` keeps the qt=44 136 B group container
and adds a per-tensor F32[16] `lloyd_levels` sidecar (see
[mq4-v2.md](quant-formats/mq4-v2.md) Lloyd section).

| `--format` | qt | `DType` | B/group | Stored bpw | Product note |
|---|---:|---|---:|---:|---|
| `mq4` *(alias)*, `mq4v2` | **44** | `MQ4G256V2` | 136 | 4.25 | Default MQ4 body; dense Qwen3.8 ladder evidence |
| `mq4c` | **45** | `MQ4CG256` | 136 | 4.25 | v1.5 pad (fp16 header @+0, zero pad +4..8, nibbles @+8) |
| `mq6v2` | **47** | `MQ6G256V2` | 200 | 6.25 | Dense product candidate (ladder) |
| `mq5v2` | **48** | `MQ5G256V2` | 168 | 5.25 | Dense product candidate (ladder) |
| `mq3v2` | **49** | `MQ3G256V2` | 104 | 3.25 | Dense product candidate (ladder) |
| `mq2v2` | **50** | `MQ2G256V2` | 72 | 2.25 | **Wire + runtime supported; product quality-rejected** |
| `mq4v2-lloyd` / tiers `mq4l-xt` · `mq4l` · `mq4l-pro` | **52** | `MQ4G256V2Lloyd` (`QuantType::MQ4G256V2L`) | 136 + `lloyd_levels` | 4.25 + per-tensor 64 B codebook | V2 wire + per-tensor 16-level Lloyd codebook; product tiers vs AWQ'd uniform |
| *(Qwen4 producer; no thin-CLI `--format`)* | **53** | `MQ4G128V2` | 68 per 128 | 4.25 | Row-local per-128 group (`ceil(K/128)` groups per row, 68 B each: fp16 scale, fp16 zero, 64 B nibbles; final partial group zero-padded before the FWHT). FWHT-128 rotation, sign seeds 43 / 1043. Written by the Flash-Next producer for routed-expert `down_proj` (K = 640, which a 256-wide group cannot tile); never interpret it as qt 44 |

Stored bpw counts the group header only. Qwen4 routed experts therefore store at
4.25 bpw in both qt 44 (gate/up, `K % 256 == 0`) and qt 53 (down, row-local
128-wide groups); whole-model bpw is higher.

**Wire implementation ≠ product admission.** MQ2V2 loads, decodes, and passes
parity, but its measured Qwen3.8 ladder KLD is catastrophic (12.44–14.20 on WT2
and v6sel in the [ladder.md](quant-formats/ladder.md) table; that table owns the KLD
protocol and unit).
**Do not call MQ2V2 a shipping product body.** MQ3–MQ6 V2 cells are measured
candidates on the product ladder, not automatic registry cards. Full contracts:
[mq4-v2.md](quant-formats/mq4-v2.md), [mq-v2-family.md](quant-formats/mq-v2-family.md),
[ladder.md](quant-formats/ladder.md).

Qwen3.x MoE runtime support covers routed `MQ4G256V2` and `MQ6G256V2`
experts for indexed decode, batched decode, grouped prefill, shared experts,
slots, and expert-parallel execution. Uniform and frozen mixed V1/V2 expert
tables preserve the exact per-projection dtype tag. This runtime support does
not relax the quantizer's direct-MoE output gate for non-MQ4 V2 formats.
Qwen4 (Flash-Next) routed experts use a different pairing, qt 44 for gate/up
and qt 53 for down; see [Shipped XTS artifacts](#shipped-xts-artifacts).

Product names (`mq4`, `mq4-xt`, `mq4 pro`, …) are **bpw-class + role-lift**
labels owned by the ladder, separate from codec generation (`V2`, `C`) and
from wire ids. A product row must satisfy measured model bpw ∈ [N, N+1).

### Shipped XTS artifacts

Two MQ4 XTS artifacts are the canonical pinned fixtures. Both are identified by
file size and SHA-256, never by file name; the pin tables live in
[AGENTS.md](../AGENTS.md) §5 (“Pinned Hugging Face bench fixture” and “Pinned
Flash-Next bench fixture”) and the registry rows in [MODELS.md](MODELS.md).

| Artifact | Registry tag | Canonical file | Size (B) | SHA-256 |
|---|---|---|---:|---|
| Dense Qwen3.8-27B | `qwen3.8:27b-mq4-xts` | `qwen3.8-27b.mq4-xts` | 14987185152 | `3e38ccbae3776470eb5a89344d300e9279d6b9ab6c31fd40ca1758c4f7c6f8ae` |
| Qwen3.8-Flash-Next (arch 16) | `qwen3.8:flash-next-mq4-xts` (also `qwen3.8:flash-next`, `-mq4`, `-gptq3`) | `qwen3.8-flash-next.mq4-xts` | 125288544792 | `8b15b6fede7d7c5bfed0db4720a8295bedda51bc93e545fa242bd50d0f200972` |

- **Dense XTS** is the symmetric MQ4V2 XT body (qt 44) of the H2 checkpoint. The
  file `qwen3.8-27b.mq4-xt` has exactly the same size, so only the digest tells
  them apart.
- **Flash-Next XTS** per its pin: trunk and language head MQ6G256V2 (qt 47), routed
  experts MQ4G256V2 (qt 44, gate/up) + MQ4G128V2 (qt 53, down), Q8F16 (qt 3)
  embedding, MTP attention and PLE n-gram rows, with the raw-I64 metadata
  records tagged qt 54. The old wire file name `qwen3.8-flash-next-gptq3.mq4`
  is a byte-identical legacy alias of the canonical name; clients normalize the
  exact repo/size/SHA-256 pin to `qwen3.8-flash-next.mq4-xts`.
  Its zab trunk tier ships as a separate `<model>.trunk-overlay` sidecar and its
  PLE rows can be read from `.ple-q4` / `.ple-q8` sidecars (see
  [QUANTIZE.md](QUANTIZE.md#qwen4-flash-next-streaming-producer)).

- **Raon-1-mini-preview XTS** (arch 19, `raon-1-mini-preview.mq4-xts`) by tensor
  class: the attention classes (MLA and RDA projections, including the RDA gates and
  the MLA/RDA LoRA factors) are promoted to Q8F16 (qt 3); `lm_head` is Q8; the routed
  experts are symmetric MQ4V2 XT (qt 44) with the XTS refit (per-group AWQ alpha plus
  GPTQ scale refit), so the int4-activation prefill and sealed MoE routes apply. The
  always-active FFN (the 24 shared experts and the dense layer-0 MLP, which every token
  reads, unlike a routed expert) is Q8F16 with no AWQ sidecar: it consumes the plain
  activation, and a qt 44 shared gate/up would otherwise carry the routed experts'
  `moe_in` AWQ scale (the runtime feeds routed and shared gate/up from one rotated
  activation; the loader rejects a qt 44 file whose shared gate/up sidecar differs).
  Shared and dense down projections are stored with physical K 1280 / 12032 (zero
  padding). The `-v2` file is the XTS file with exactly those 75 tensors re-quantized
  from the BF16 source by `hipfire-quantize --raon-shared-ffn q8f16 --input <xts.hfq>
  --raon-source <checkpoint>` (everything else byte-identical; `--raon-shared-ffn-verify`
  proves it); `HIPFIRE_RAON_SHARED_FFN=q8f16|bf16|qt44` selects the same class in a full
  conversion. KLD vs the BF16 teacher: WT2 0.074, code24 0.0116 (the previous XTS file
  with qt 44 shared/dense FFN: 0.081 / 0.0138; the RTN control with the same Q8
  attention: 0.102 / 0.019). Source-exact BF16 for the same tensors measured no better
  (0.075 / 0.0118) and costs 0.8 GB more device memory, so it is not shipped.
  Prefill activation precision resolves once at load via
  `HIPFIRE_RAON_PREFILL_PRECISION` (per-site policy; gfx1201 default: int8 activations at
  every site, W4A8 FFN + W8A8 attention class, table in
  `crates/hipfire-arch-raon/src/precision.rs`; `f16` is the kill switch, `<site>=f16` a
  per-site one).

**Historical sweep vs current registry.** The `mq4-xt` / `mq4` / `mq4 pro` ladder
cells in [ladder.md](quant-formats/ladder.md) are the 2026-08 candidate sweep on
dense Qwen3.8-27B (a measurement grid, not registry cards). The shipped XTS
files above are the current registry pins; a file in the sweep that happens to
share a product name or byte size with a pinned file is not comparable unless its
digest matches. [ladder.md](quant-formats/ladder.md) owns the sweep results and
the current-quality-methodology note for shipped XTS trunks.

### Bonsai (ternary / 1-bit)

| Type | qt | Bytes/group | Stored bpw | Notes |
|---|---:|---|---:|---|
| `TQ2G128` | 40 | 34 per 128 | 2.125 | PrismML Bonsai ternary — passthrough; `[fp16 d][32 B 2-bit codes]` |
| `BQ1G128` | 41 | 18 per 128 | 1.125 | PrismML Bonsai 1-bit — passthrough; `[fp16 d][16 B sign bits]`; won qt=41 adjudication 2026-08-18 |

### Sub-4-bit and research formats (v1 / Lloyd / FP4)

“Stored bpw” is group bytes × 8 ÷ weights (header included); rows of the FP4/E8
families add a 16 B row header (128 / K bpw) and `MFP4G32Lloyd` a 32 B per-tensor
codebook, so their per-weight cost depends on K.

| Type | qt | Bytes/group | Stored bpw | Truth state (INDEX) |
|---|---|---|---:|---|
| `MQ3G256` | 17 | 104 | 3.25 | **shipped / ref-pinned** encoder + decode/prefill paths; quality degrades below ~9B (historical local battery — see below) |
| `MQ2G256` | 18 | 72 | 2.25 | **shipped / ref-pinned** encoder; reserved product use — uniform 4-level codebook collapses; needs `--allow-mq2` / `HIPFIRE_ALLOW_MQ2=1` (research opt-in qualifier) |
| `MQ2G256Lloyd` | 19 | 72 | 2.25 | **shipped / ref-pinned** implementation; research opt-in (`--allow-mq2-lloyd`) — not a product default |
| `MQ3G256Lloyd` | 20 | 112 | 3.5 | **shipped / ref-pinned** implementation; research opt-in (`--allow-mq3-lloyd`) — not a product default |
| `MQ4G256Lloyd` | 30 | 160 | 5.0 | **shipped / ref-pinned** implementation; research opt-in (`--allow-mq4-lloyd`) — qt renumbered 21→30; pre-renumber artifacts must be re-quantized |
| `MQ5G256` | 31 | 168 | 5.25 | **shipped / ref-pinned** encoder present (5-bit FWHT) |
| `MQ2G256GL` / `MQ3G256GL` | 38 / 39 | 66 / 98 (GL SoA: indices + fp16 scale) | 2.0625 / 3.0625 | **shipped / ref-pinned** Magnum GL passthrough; one tensor-global codebook, not counted here |
| `MQ2G256LloydU` | 51 | 72 | 2.25 | passthrough; Maple native-ternary: byte layout of qt 19 but **unrotated** (no FWHT at pack time, so the runtime must not rotate `x` for it) |
| `HFP4G32` | 21 | 17 per 32 | 4.25 + 128 / K | **shipped / ref-pinned** FP4 family (see hfp4.md) |
| `MFP4G32` | 24 | 17 per 32 | 4.25 + 128 / K | **shipped / ref-pinned** drop-in MQ4-class FP4+FWHT |
| `MFP4G32Lloyd` / `MFP4G32P` / `MFP4G32E8` / SoA / `MFP3G32E8` / `MFP2G32E8` | 32–37 | 17 / 17 / 17 / 17 / 13 / 9 per 32 | 4.25 / 4.25 / 4.25 / ≈4.25 / 3.25 / 2.25 (each + row overhead) | **shipped / ref-pinned** encoders + selective kernels; experimental opt-in qualifier; MoE cold-tier recipes — not CLI defaults. The SoA layout (qt 35) pads its scale block to 16 B per row |
| `MFP4G32E8G128` | 42 | 17 per 32 | 4.25 + 128 / K | arch-loaded (Qwen4 expert load path): wire layout byte-identical to qt 34, but the encoder rotates in 128-wide segments (sign seeds 43 / 1043) because a 256-wide segment cannot tile K = 640; the rotation width is not in the bytes |
| `PARO4G128` / `PARO4G128T` | 28 / 29 | Paro buffers | varies | **shipped / ref-pinned** load/probe path for ParoQuant imports |
| `MQ2G256V2` | 50 | 72 | 2.25 | **wire/runtime supported; product quality-rejected** — see V2 family section |

**Historical quality note (not a live admission):** uniform MQ3/MQ2 local
battery on **gfx1100** (7900 XTX), **Qwen 3.5 / 3.6**, master ref **`c448d5e`**
(2026-04-30). MQ3 was fluent on-topic at 9B and 27B across a 4-prompt coherence
set; 4B was partial collapse; 0.8B gibberish. Uniform MQ2 collapsed at every
size tested. Treat as scoped **historical** provenance, not a product floor.
MQ3-Lloyd / MQ2-Lloyd remain research-gated until product validation. Separately,
the **2026-08-20** dense Qwen3.8 MQ V2 ladder rejects **MQ2V2** on KLD while
keeping wire support.

### Prefill / WMMA runtime batch eligibility (weights)

`is_batchable_la(dtype, arch)` is **runtime batch eligibility only** — it is
**not** product admission and is **not** an empty/full admission registry.
llama and qwen35 each own their own predicate and are **not** currently in
lockstep (`crates/hipfire-runtime/src/llama.rs`,
`crates/hipfire-arch-qwen35/src/qwen35.rs`).

**Always batch-eligible (both owners):** `MQ4G256`, `HFQ4G256`, `MQ6G256`,
`HFQ6G256`, `Q8_0`. (qwen35 also treats `ParoQ4G128` / `F32` as always-ok for
the PARO probe path; llama admits `ParoQ4G64`, whose prefill rotates each
projection's rows and runs `gemm_hfq4g64`, while llama `ParoQ4G128` stays
per-token.)

| DType | `llama::is_batchable_la` | `qwen35::is_batchable_la` |
|---|---|---|
| `MQ3G256` | WMMA on gfx1100/1101/1102/1150/1151/1200/1201; scalar batched on gfx101x/103x; else not | Same plus **gfx1103** and **gfx1152** on the WMMA set; same gfx10 scalar set |
| `HFP4G32`, `MFP4G32` | WMMA arches only (llama set above) | WMMA arches only (qwen35 set, includes gfx1103/1152) |
| `MQ3G256Lloyd` | **Not** batch-eligible | WMMA on gfx1100/1101/1102/1150/1151/1200/1201 |
| `MQ4G256Lloyd` | **Not** batch-eligible | WMMA on gfx1100/1101/1102/1151/1200/1201 (no gfx1150) |
| Other Lloyd / E8 / research dtypes | Not batch-eligible here | Additional arms (e.g. E8) may exist behind their own env gates — see source |

Uniform MQ3 ships WMMA residual / fused GEMM kernels on RDNA3+ and is
batch-eligible on the arches above. Do **not** claim “no MQ3 WMMA.” Lloyd
batch eligibility is owner-specific (qwen35 yes on listed arches; llama no).

## FWHT rotation (the M in MQ)

Each 256-wide group is transformed by **two** fixed `±1` sign diagonals and a
Walsh–Hadamard butterfly, then scaled by `1/√256 = 1/16`. Encoder and runtime
share the same contract (`cpu_fwht_256` / `fwht_forward_256` /
`rotate_x_mq`):

```
R = D2 · H · D1 / 16
  signs1 → Hadamard butterfly → ×(1/16) → signs2

quant:       w′ = R w   →  quantize w′
inference:   x′ = R x   (same R, not an inverse)
             y  = (w′)ᵀ x′  = wᵀ x     in exact arithmetic
```

Sign tables use fixed PRNG seeds **42** (`D1` / signs1) and **1042** (`D2` /
signs2), shared by quantizer and runtime. Cancellation holds in exact
arithmetic; **fp32** mixed-precision paths accumulate rounding drift (typically
below per-block quantization error, but not bit-exact identity).

**Why it matters:** outliers spread across the group, narrowing dynamic range
before 4-bit (or lower) quantization. The rotation was designed around Qwen
3.5 / 3.6 (DeltaNet hybrid) weight distributions, which is why MQ is the
recipe for that family.

On Llama-style dense models the math still cancels, but the rotation adds
`rotate_x_mq` cost on the activation side. Recipe guidance: **MQ for Qwen
3.5+; HFQ for other dense families** — treat this as a starting point, not a
guarantee: whether MQ helps or HFQ suffices is model-specific and needs
validation per model. CLI defaults follow the guidance (`safetensors` → `mq4`;
GGUF → `hf4`, GGUF weight input is **deprecated since 0.4.0, removal in
0.5.0**; see [QUANTIZE.md](QUANTIZE.md)).

## Header layout (HFQ/MQ G256 affine blocks)

### v1 (qt 6/13/15/17/18/…)

Per 256 elements, 8-byte header + packed payload. HFQ4 and MQ4 v1 both store an
**f32 affine scale** and an **f32 minimum** (not FP16-in-f32):

- scale: f32
- zero / min: f32
- data: `bitwidth × 256 / 8` bytes (128 for 4-bit, 96 for 3-bit, 192 for 6-bit, …)

### V2 (qt 44 / 47–50 / 52) and MQ4C (qt 45)

Same 8-byte header budget and payload offset **+8**, different header encoding:

- **MQ\*G256V2:** fp16 scale+zero for half 0 (weights 0–127) at `[0..4)`, half 1
  at `[4..8)`; payload packing identical to the matching v1 bit width.
- **MQ4CG256:** one packed fp16 scale+zero at `[0..4)`, mandatory zero pad
  `[4..8)`, 128 B nibbles at `[8..136)` (v1-compatible geometry).
- **MQ4G256V2L (qt 52):** byte-identical 136 B group to qt=44; nibble `q`
  decodes through a per-tensor F32[16] Lloyd codebook sidecar
  (`<tensor>.lloyd_levels`), with loader zero-point rewrite
  `zp' = f16(zp + 7.5·sc)` and centered levels `C = L − 7.5`.

Legacy Lloyd variants (qt 19/20/30) replace the uniform grid with a per-block
(or per-tensor) fp16 codebook plus packed indices; byte sizes differ (e.g.
MQ3-Lloyd 112 B/group, MQ4-Lloyd 160 B/group).

## KV cache

K and V are quantized differently: K enters softmax (errors exponentiate); V is
already attention-weighted. Split modes store **V as Q8** (or Lloyd V via
`--kv-v`) and compress **K**; native `fp8`/`bf16` store K and V in one layout:

| Mode | K | V | Notes |
|---|---|---|---|
| `q8` | Q8_0 (G32) | Q8_0 | `auto` default wherever native fp8 does not apply (gfx1100, gfx1151, …); DFlash-safe |
| `fp8` | FP8 E4M3 | FP8 E4M3 | Qwen-only native pair; `auto` default on native-eligible exact gfx1201 single GPU (H24/Hkv4/D256 dense or H16/Hkv2/D256 A3B, D256, no adaptive/CASK/slots/TP/PP split). Native `bf16` stays H24 dense-only |
| `fwht4` / `fwht3` / `fwht2` | FWHT-rotated K at 4/3/2-bit | Q8_0 | Optional headroom modes (Qwen3.5 family, any arch), not a default. Same byte layout family as legacy asym; FWHT basis matches MQ drafts |
| `asym4` / `asym3` / `asym2`, `turbo` / `turbo2` / `turbo3` / `turbo4` | **Deprecated spellings** (since 0.4.0, removal in 0.5.0; spell `fwhtN` or `q8`) | Q8_0 | On Qwen they are aliases that resolve to the supported FWHT K path: `asymN`/`turboN` → `fwhtN`, `turbo` → `fwht3`. Only the llama HFQ loader still maps bare `asym3`/`asym4`/`turbo4` to the legacy Givens constructors (deprecated route, removal in 0.5.0) |
| `legacy-asym4` / `legacy-asym3` / `legacy-asym2` | Givens + Lloyd-Max K | Q8_0 | **Deprecated since 0.4.0, removal in 0.5.0** (fwht3 supersedes Givens asym KV); Qwen `--kv-k` only |

**Default resolution** (`load_params` in `crates/hipfire-cli/src/main.rs`,
then the load site's policy in `crates/hipfire-runtime/src/kv_mode.rs`):

1. CLI `--kv-mode`
2. `HIPFIRE_KV_MODE` env, then per-model / global config `kv_cache`
3. Registry `default_kv_mode` when no user layer sets `kv_cache` and the
   card's value is not `q8` (a `q8` card value is left to `auto`)
4. Else `auto`, resolved by the architecture: Qwen native **fp8** on
   native-eligible exact gfx1201 single GPU (H24/Hkv4/D256 dense or
   H16/Hkv2/D256 A3B; no adaptive/CASK/slots/TP/PP split), **q8** on
   every other Qwen load (gfx1100, gfx1151, …); other families keep
   their own default (Maple bf16, DeepSeek V4 f32, others q8)

There is **no** live per-arch `archDefaults` table (removed). Compressed modes
are opt-in via registry cards or explicit config — not guessed from GPU VRAM.
Override: `hipfire config set kv_cache <mode>` or `HIPFIRE_KV_MODE=<mode>`.

Optional **adaptive** downshift (`kv_adaptive`: `off` | `conservative` |
`balanced` | `aggressive` | `advanced:k=…,v=…`) can lower V and K tiers as
context grows; see [CONFIG.md](CONFIG.md). Works best when K starts at a higher
fwht tier.

## HFQM container (`.hfq` / `.mq4` / `.hf4` / …)

Extensions are naming convenience; the on-disk magic is always **`HFQM`**:

```
0x00  "HFQM"              (4 bytes)
0x04  version             (u32 LE = 1)
0x08  arch_id             (u32 LE — see architecture-ids.md)
0x0C  n_tensors           (u32 LE)
0x10  metadata_offset     (u64 LE)
0x18  data_offset         (u64 LE — 4096-aligned)

[metadata_offset .. data_offset):
  metadata_json (UTF-8: config, tokenizer, gguf_meta, source, …)
  tensor_index:
    n_tensors: u32 LE
    for each tensor:
      name_len: u16 LE
      name: utf-8
      quant_type: u8      (see QuantType table below)
      n_dims: u8
      shape: u32 LE × n_dims
      group_size: u32 LE
      data_size: u64 LE

[data_offset .. EOF]: tensor bytes in index order
```

`HfqFile::open` mmaps the file; tensor slices feed HIP upload without an extra
host copy. Bundled `.mq4-mtp` files append an MTP HFQM section + `HFBNDMTP`
trailer; trunk loaders ignore the trailer and see only the trunk.

### QuantType IDs (encoder / loader contract)

Source of truth: [`quant-formats/qt-register.txt`](quant-formats/qt-register.txt).

| ID | Name | Notes |
|---:|---|---|
| 0 | Q4F16G64 | Legacy |
| 1 | F16 | |
| 2 | F32 | |
| 3 | Q8F16 | Embeddings / protected |
| 4 | Q4K | GGUF-family |
| 5 | Q8HFQ | |
| 6 | HFQ4G256 | Production dense 4-bit (unrotated) |
| 7 | HFQ4G128 | |
| 8 | HFQ6G256 | |
| 9–12 | HFQ2/3 G256/G128 | |
| 13 | MQ4G256 | Legacy Magnum 4-bit v1 (`mq4v1` / `mq4g256` / `magnum`) |
| 14 | MQ8G256 | |
| 15 | MQ6G256 | Magnum 6-bit v1 |
| 16 | BF16 | Vision / lossless |
| 17 | MQ3G256 | Uniform 3-bit MQ v1 — **shipped / ref-pinned** |
| 18 | MQ2G256 | Uniform 2-bit v1 — **shipped / ref-pinned** encoder; reserved product / research opt-in |
| 19 | MQ2G256Lloyd | **shipped / ref-pinned** impl; research opt-in |
| 20 | MQ3G256Lloyd | **shipped / ref-pinned** impl; research opt-in |
| 21 | HFP4G32 | Canonical HFP4 — **shipped / ref-pinned** |
| 22 | TidI32 | DeepSeek tid2eid tables (**reassigned** from former HFP4G16 reservation) |
| 23 | *(reserved)* | Former HFP4G64 ablation slot — still reserved, not built |
| 24 | MFP4G32 | HFP4 + offline FWHT — **shipped / ref-pinned** |
| 25–27 | *(reserved)* | Former MX/NV/HFP8 interop slots — still reserved, not built |
| 28 | PARO4G128 | ParoQuant native — **shipped / ref-pinned** load path |
| 29 | PARO4G128T | Engine-tiled Paro — **reassigned** from former `MFP4G32R` online-R reservation |
| 30 | MQ4G256Lloyd | Research MQ Lloyd — **reassigned** (was 21 pre-renumber off HFP4) |
| 31 | MQ5G256 | 5-bit MQ v1 — **shipped / ref-pinned** encoder |
| 32–37 | MFP4Lloyd / MFP4P / E8 / SoA / MFP3E8 / MFP2E8 | **shipped / ref-pinned** encoders + selective kernels; experimental opt-in |
| 38 | MQ2G256GL | Magnum GL 2-bit — passthrough |
| 39 | MQ3G256GL | Magnum GL 3-bit — passthrough |
| 40 | TQ2G128 | Bonsai ternary — passthrough |
| 41 | BQ1G128 | Bonsai 1-bit — passthrough |
| 42 | MFP4G32E8G128 | mfp4-E8 with a 128-wide FWHT rotation group — arch-loaded (Qwen4 expert load path); wire layout byte-identical to qt 34 (16 B header + (K/32) × 17 B), 4.25 bpw + 128 / K; split from qt 34 because its 256-wide segments cannot tile K = 640 |
| 44 | MQ4G256V2 | **Default `mq4` body** — dual-fp16 per-128 header, 136 B (4.25 bpw) |
| 45 | MQ4CG256 | MQ4 v1.5 pad — 136 B |
| 47 | MQ6G256V2 | Magnum 6-bit V2 — 200 B |
| 48 | MQ5G256V2 | Magnum 5-bit V2 — 168 B |
| 49 | MQ3G256V2 | Magnum 3-bit V2 — 104 B |
| 50 | MQ2G256V2 | Magnum 2-bit V2 — 72 B; **wire OK, product rejected** |
| 51 | MQ2G256LloydU | Maple native-ternary: unrotated MQ2-Lloyd, same 72 B layout as qt 19 — passthrough |
| 52 | MQ4G256V2L | **MQ4 V2 + per-tensor Lloyd codebook** — 136 B group + `lloyd_levels` F32[16] sidecar; product tiers `mq4l-xt` / `mq4l` / `mq4l-pro` |
| 53 | MQ4G128V2 | Row-local MQ4 V2, per-128 groups of 68 B (4.25 bpw) — passthrough; Qwen4 routed-expert `down_proj` |
| 54 | QwenI64 | **Reserved in the register; not a quantized weight and no `QuantType` variant.** Raw signed-I64 metadata record (the three PLE hash arrays: multipliers, vocab sizes, prefix offsets) written by the Qwen4 producer |
| 131 | MQ3I4G128R256 | MQ3-I4 gate/up — planar `[M·K·3/8 B codes][M·K/128 f16 scales]`, R256 FWHT (seeds 42/1042); **typed-qwen4 expert-slab only, gfx1151/gfx1201** |
| 132 | MQ3I4G128R128 | MQ3-I4 down — same planar W3 layout, R128 FWHT (seeds 43/1043); **typed-qwen4 expert-slab only, gfx1151/gfx1201** |
| 133 | MQ2I4G128R256 | MQ2-I4 gate/up — planar `[M·K/4 B codes][M·K/128 f16 scales]`, R256 FWHT; **typed-qwen4 expert-slab only, serving refused on every arch** |
| 134 | MQ2I4G128R128 | MQ2-I4 down — same planar W2 layout, R128 FWHT; **typed-qwen4 expert-slab only, serving refused on every arch** |
| 135 | E8PI4G128R256 | E8P-I4 gate/up — `[M·K/8 LE u16 words][M·K/128 f16 scales]`, R256 FWHT; **typed-qwen4 expert-slab only, gfx1100/gfx1151/gfx1201** |
| 136 | E8PI4G128R128 | E8P-I4 down — same planar E8P layout, R128 FWHT; **typed-qwen4 expert-slab only, gfx1100/gfx1151/gfx1201** |

**Never-reuse reserved wire IDs:** 23 and 25–27 (deliberately unassigned). qt 54 is
held in the register with disposition `reserved` for the Qwen4 I64 metadata
record described above; it is not an encoded weight type. IDs **22, 29, and 30
were reassigned** (TidI32, PARO4G128T, MQ4G256Lloyd). qt 42 and 53 are now live
rows (above); qt **43 and 46** have no register row (43 is burned by the retired
SEL codebook, 46 is the design-only hierarchical `mq4_k`). Do not invent slots —
edit `qt-register.txt` to claim a number.

IDs **131–134** (MQ3-I4 / MQ2-I4) are fresh allocations above qt130 (compact
Hessian records, not a `QuantType`). They are planar (codes plane, then f16
scale plane; codec in `crates/hipfire-runtime/src/mq_i4.rs`), valid only as
per-expert entries of a Qwen4 artifact declaring `qwen4.expert_slabs` v1
(`crates/hipfire-arch-qwen4/src/expert_slabs.rs`).
The CPU byte/arithmetic oracles in `hipfire_runtime::mq_i4_reference` define
signed A4 unpacking, ascending-K128 fused IU4 folds, the frozen wave32 F4
decode tree and original-slot scatter. Their module documentation is the
rounding/order contract for the corresponding PeaceMaker kernels.

**Producing the artifact.** `qwen4_i4_slabs` (`crates/hipfire-quantize/src/bin/qwen4_i4_slabs.rs`, writer
`hipfire_quantize::i4_slabs::write_i4_expert_slabs`) packs the pinned MQ4-XTS Flash-Next HFQ plus R1
per-expert NPZ records (`L{L}/E{e}.npz`: `gds/gc/dds/dc/sd/bits/grid/scale_rule`) into the slab container:
trunk expert entries first (layer, expert, gate_up then down; qt 131/132, 133/134 or the unchanged 44/53
slices of the published aggregates, chosen per expert by an allocation point or `--uniform w3|w2|mq4`), then
every other source tensor byte-identically (legacy MTP aggregate experts included), plus the
`qwen4.expert_slabs` and `qwen4.expert_slabs_provenance` metadata keys. Import accepts only `grid=uniform`,
`sd` exactly 1.0 and the scale rules `mqi4-canonical-v1-rne-tree128`, `mqi4-recover-s1-v1`,
`mqi4-recover-s2-v1`; scales must be exact f16 values. Verification cannot be disabled: every expert is
decoded back to its codes and scale bits, the output is reopened and every payload plus the whole file is
re-hashed, and `--cross-check-packs-w3/-w2` additionally compares each slab byte-for-byte with the Python R1
pack. `--receipt` writes the JSON receipt. An existing `--out` is never overwritten.

**Pruned coder split bases.** `--expert-keep keep.json` accepts
`hipfire.coder.keep.v1` through `ExpertKeep::from_coder_keep_file`. Expert ids
remain in the original 512-id space; sorted, ragged layer lists are allowed
(at least top-10 per layer). Absent experts have no index entries or payload.
The loader synthesizes their zero-byte slab slots. Allocation rows may remain
dense (absent entries are ignored) or contain only kept experts in keep-list
order. Optional `tiers` is null or an object keyed by original layer and expert
ids, e.g. `{"0":{"3":"w3","17":"mq4"}}`. A tier for an absent expert is refused,
as are missing records for selected W3/W2 experts. Receipt tier counts cover
kept experts only.
Optional `mtp` selects compact MTP aggregate rows; absent/null preserves MTP.
`qwen4.expert_keep` records membership and its canonical SHA-256.

Use the shipped `hipfire ple split` path on the MQ4 source first (or reuse an
already verified MQ4 split base). Packing that base preserves its
`hipfire_ple_rows` marker unchanged and never adds PLE rows. This binds the
pruned base to the same shared `.ple-q4`/`.ple-q8` row identity, independent of
its new base size/index hash. The split command verifies the Q8 source rows;
do not copy a marker from an unrelated model.

```sh
hipfire ple split "$SOURCE_MQ4" --out "$SOURCE_BASE_MQ4" --keep-original
qwen4_i4_slabs --source "$SOURCE_BASE_MQ4" \
  --alloc "$ALLOC" --point "$POINT" --records-w3 "$W3_RECORDS" \
  --expert-keep "$KEEP_JSON" --out "$CODER_BASE" \
  --receipt "$RECEIPT" --threads 8
qwen4_slab_admit_check "$CODER_BASE"
```

Install the verified shared row files under the runtime aliases
`$CODER_BASE.ple-q4` / `$CODER_BASE.ple-q8` (links to the shared files, not copies).
They must have the same Q8 source-row digest and geometry as the split marker.
Add `--size-only` to the pack command to print exact container bytes and
kept-only routed/non-expert payload totals without writing output or receipt;
use identical source, records paths, allocation, and keep file for packing.
The checker verifies membership SHA, table counts, exclusion of absent expert
tensors, all W3/W2 payloads, and the production CPU admission refusal.

The CPU loader builds an O(index entries) `expert_slab_table::ExpertSlabTable`
from the actual spans: 48 × 512 gate/up-down pairs first, canonical names and
order, frozen shapes/qtype pairs/group sizes/byte sizes, contiguous aligned
slabs, no trunk aggregates or stray I4 entries. The sealed slab route admits
MQ4 (codec 0) with E8P (codec 3) on exact gfx1100, MQ4 with W3 (codec 1) or E8P
on exact gfx1151 (E8P through the `SLAB_ROUTE_GFX1151_E8P_RECEIPTS_LANDED`
admission switch, `true`; its earlier Halo byte-oracle receipts are pre-rc5 history and do
not by themselves qualify the current bounded images), and MQ4 with W3 or E8P on exact gfx1201. Each
layer has at most
one cold codec; W3 and E8P cannot share its cold bucket. W2 is refused
everywhere, as are W3 on gfx1100 and all other slab
architectures. Missing or malformed v1 metadata fails closed.
`validate_expert_slab_scales` separately checks W3/W2 payloads through the shared
codec, including finite non-negative f16 scales, rejection of -0, and canonical
midpoint codes for zero-scale groups; it is not run on the default load path.
Legacy aggregate artifacts retain their existing admission checks and ordering.

The two execution buckets remain cold codec then MQ4. The planner's u8
table is a cold-membership flag (W3/E8P = 1, MQ4 = 0), not a codec ID;
the real codec IDs remain available for the prefill tile masks. Kernel
entries are selected by architecture and the layer's cold codec. gfx1100
uses its own K1/K2/K3 objects and tile-mask image, with AR only (MTP off).
Existing gfx1201 W3/MQ4 kernel bytes, the gfx1201-only EP long-prefill staging
default, and VMM/MTP refusals are unchanged. gfx1151 (Strix Halo) keeps its
beta MQ4/W3 route (W3 K2 NT4 launcher, hipcc tile mask) and serves E8P layers
with its own PeaceMaker E8P K1/K2 images (`e8p_i4_moe_gemv_pm_gfx1151`,
`qwen4_moe_iu4_sym_e8p_pm_gfx1151`), admitted by the single
`SLAB_ROUTE_GFX1151_E8P_RECEIPTS_LANDED` constant
(`rdna-compute/src/slab_admission.rs`); that is an admission statement, not a GPU
qualification claim for the rc5 bounded images (fresh Halo gates are separate). Its slab
kernel preload is unconditional (planner, mask and all four K1 handles, even
for an MQ4-only layer), the one exception to the no-cold-kernel-preload rule
that gfx1100 and gfx1201 keep; it is not a W3 bucket launch.

The gfx1100 E8P K2 wiring is CPU-prepared; promotion remains gated on its
XTX K2 G1. This admission description is not a GPU qualification claim.

The rc5 bounded E8P ABI is shared by gfx1100 and gfx1151: K1 uses a
96-byte kernarg segment (`n_experts: u32` at byte 88, reserved zero at
byte 92; 17 kernel parameters), and K2 uses a 64-byte segment
(`n_experts: u32` at byte 60). Both kernels reject expert IDs with an
unsigned, uniform bounds check before any expert pointer read, prefetch,
or barrier. Invalid IDs perform no dereference or output store; the
gfx1151 port preserves rc5's E8P NaN containment.

#### Expert capability table

One table, `rdna_compute::expert_capabilities`, answers "does this arch run this expert codec on
this route" for every gate (slab K1/K2 selection, arena and tier-plan `check_executable`,
artifact load, dispatch pair/bucket selection, expert stage defaults). The key is the exact
on-disk quant-tag pair read from the tensor table (`ExpertQt`: MQ4 `(44,53)`, W3 `(131,132)`,
W2 `(133,134)`, E8P `(135,136)`), never a model or XTS label. `generation_label` (v1 qt 13,
v2 candidates qt 45/52, v3 qt 44/53 and the planar I4 tags) is display-only and never read by
admission; the "V2" in `MQ4G256V2`/`MQ4G128V2` names a codec-layout revision, not the MQ
generation. The qt 44/53 kernels need the symmetric grid, which the tag cannot show; the
existing header check (`routed_headers_symmetric`, `qwen4_moe_sym_check`) stays the prerequisite.

Routes: `PrefillK2`, `DecodeGemv`, `Stage` (slab-arena staging, needs K1 and K2), `TierStage`
(`ExpertTierPlan`; MQ4 is arch-neutral), `D2CpuSplit` (CPU decoder inventory supplied by
dispatch). A cell is admitted only when its raw kernel inventory (read from the real K1/K2
tables, never from admission) is present and no named hold refuses it. Beta holds, all in
`HOLD_CELLS`: tier plans refuse E8P on gfx1100/gfx1151/gfx1201 and W3 on gfx1151; the gfx1151
E8P cells (K2, K1, stage) hold only while `SLAB_ROUTE_GFX1151_E8P_RECEIPTS_LANDED` is false. The
per-arch artifact receipts flags gate the load only. W2 and gfx1100 W3 have no kernel. Stage
defaults (gfx1201 always on and 2048 rows; narrow-link policy elsewhere; the gfx1100 8192-row
narrow-link chunk) are `stage_policy` metadata read together with the measured link predicate.

### PARO4G128 probe payload

`quant_type=28` is **not** row-major HFQ4/MQ4. Each tensor stores ParoQuant
buffers in order:

```
qweight        int32 [K, M/8]
qzeros         int32 [K/128, M/8]
scales         f16   [K/128, M]
pairs          int16 [8, K]
theta          f16   [8, K/2]
channel_scales f16   [K]
```

Runtime contract:

```
x_rot = rotate(x, pairs, theta, channel_scales)
y     = awq_w4_gemv(x_rot, qweight, qzeros, scales)
```

Verify imports with `python3 scripts/astrea.py paro-oracle --source PARO_SAFE_DIR
--hfq MODEL.hfq --module MODULE --pretty` before quality or perf claims.

## Why a custom format

hipfire stores weights in its own HFQM container with in-place GEMV kernels
and engineering-level runtime paths rather than a new quantization algorithm class:

1. **Fused projections** — QKV / QKVZA / gate+up fused launches for HFQ4/MQ4
   (and siblings) reduce the number of kernel launches on decode.
2. **WMMA prefill** — batched GEMM on RDNA3+ WMMA for MQ/HFQ (and MQ3/HFP on
   eligible arches) amortizes dequant across prompt tokens.

Published speedups belong in measured owners ([BENCHMARKS.md](BENCHMARKS.md),
perf-checkpoints); do not promote undated multiples here.

## Related

- Operator quantize: [QUANTIZE.md](QUANTIZE.md)
- Magnum V2 / family / ladder / register:
  [quant-formats/mq4-v2.md](quant-formats/mq4-v2.md),
  [quant-formats/mq-v2-family.md](quant-formats/mq-v2-family.md),
  [quant-formats/ladder.md](quant-formats/ladder.md),
  [quant-formats/qt-register.txt](quant-formats/qt-register.txt)
- HFP4 wire spec: [quant-formats/hfp4.md](quant-formats/hfp4.md)
- Config / env KV knobs: [CONFIG.md](CONFIG.md), [env-vars.md](env-vars.md)
- Models / registry: [MODELS.md](MODELS.md)
- Arch IDs: [architecture-ids.md](architecture-ids.md)
- Sources: `crates/hipfire-quantize/src/` (`QuantType`, encoders, pipeline),
  `crates/hipfire-runtime/src/{hfq,llama,weight_backend}.rs`,
  `crates/hipfire-cli/src/main.rs`, `crates/hipfire-config/src/lib.rs`
