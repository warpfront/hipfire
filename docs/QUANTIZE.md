# Quantize

`hipfire quantize` is the user-facing wrapper around the CPU-only
`hipfire-quantize` binary. It converts HuggingFace safetensors or a local
model directory into hipfire’s HFQM container (extensions `.mq4`, `.hf4`,
`.mq6`, …). The daemon mmaps the result directly. GGUF **weight** input is
deprecated since 0.4.0 (removal in 0.5.0): GGUF→mqN is lossy double quantization; quantize from the original safetensors instead. A GGUF-format
`imatrix.gguf` (`--imatrix`) stays supported.

Design / math / wire IDs: [QUANTIZATION.md](QUANTIZATION.md).
Magnum V2 specs: [quant-formats/mq4-v2.md](quant-formats/mq4-v2.md),
[quant-formats/mq-v2-family.md](quant-formats/mq-v2-family.md).
Product ladder: [quant-formats/ladder.md](quant-formats/ladder.md).
Register: [quant-formats/qt-register.txt](quant-formats/qt-register.txt).
HFP4 family detail: [quant-formats/hfp4.md](quant-formats/hfp4.md).

## Pick a format

| Format | Wire | Bitwidth | Rotation | Typical use |
|---|---|---|---|---|
| **`mq4`** *(alias)* / `mq4v2` | **qt 44 `MQ4G256V2`** | 4 | FWHT | Qwen 3.5+ default body. **`--format mq4` means V2, not qt 13.** |
| `mq4v1` / `mq4g256` / `magnum` | qt 13 `MQ4G256` | 4 | FWHT | Legacy Magnum v1 only (explicit opt-in) |
| `mq4c` | qt 45 `MQ4CG256` | 4 | FWHT | V1.5 pad geometry; `hipfire-quantize` direct |
| `mq6` | qt 15 `MQ6G256` | 6 | FWHT | Qwen 3.5+ v1 when you can spare size for quality |
| `mq{6,5,3,2}v2` | qt 47/48/49/50 | 6…2 | FWHT | Dense Magnum V2 ladder bodies — **`hipfire-quantize` direct** |
| `hf4` (`hfq4`, `hfq4g256`) | qt 6 | 4 | none | Dense Llama / Mistral / Gemma / older Qwen. **Default for GGUF.** |
| `hf6` (`hfq6`, `hfq6g256`) | qt 8 | 6 | none | Dense, higher quality |
| `q8` / `q8f16` | qt 3 | 8 | none | Reference / debug (safetensors path) |
| `mq3` | qt 17 | 3 | FWHT | Sub-4-bit v1 — **`hipfire-quantize` direct only** (thin CLI rejects it) |
| `tq2` / `bq1` | qt 40 / 41 | ternary / 1 | none | Bonsai — `hipfire-quantize` direct |
| `hfp4` / `mfp4` / E8 / Lloyd | varies | see docs | Advanced; usually via `hipfire-quantize` directly |

FWHT (the “M” in MQ) applies the two-sign rotation
`R = D2 · H · D1 / 16` (seeds 42 / 1042) before quantization. Qwen 3.5+ kernels
apply the **same** `R` to activations (`rotate_x_mq`), not an inverse. On
Llama-style dense models MQ is correct math but adds rotate cost with little
quality benefit — prefer HFQ.

**Thin CLI allowlist** (`hipfire quantize`): still only `mq4`, `mq6`, `q8`,
`q8f16`, and `hf4`/`hf6` aliases. **Unchanged surface** — the wrapper does not
grow V2/`mq3`/Lloyd/HFP flags. Because the binary maps `mq4` → qt 44, a thin
CLI `hipfire quantize … --format mq4` produces **MQ4G256V2**. **GGUF input**
further narrows to `hf4`, `hf6`, `mq4`, `mq6`. Formats such as `mq3`,
`mq{2,3,5,6}v2`, `mq4v1`, `mq4c`, HFP/MFP, Lloyd, Bonsai, and graded MoE need
`hipfire-quantize` directly.

**Not produced by the thin CLI path:** graded per-expert MoE recipes
(`mq4p`, tiered Lloyd, imatrix/Hessian GPTQ-E8, REAP overlays), product-tier
ladders (`--tier` / `--fixed-tier`), and research formats. Those need extra
flags on `hipfire-quantize` directly. See the binary’s `--help` and
QUANTIZATION.md.

**MQ2V2 product rejection:** `--format mq2v2` is wire-legal and runtime-loaded,
but dense Qwen3.8 ladder KLD rejects it as a product body. Prefer `mq3v2` and
above for any artifact you intend to serve; see
[ladder.md](quant-formats/ladder.md).

Research / reserved formats require explicit opt-in on the binary:

| Flag / env | Opens |
|---|---|
| `--allow-mq2` / `HIPFIRE_ALLOW_MQ2=1` | uniform `mq2` v1 (reserved — collapses) |
| `--allow-mq2-lloyd` | `mq2-lloyd` |
| `--allow-mq3-lloyd` | `mq3-lloyd` |
| `--allow-mq4-lloyd` | `mq4-lloyd` |

### FLUX component packs (no quantization)

`--flux-pipe` does not quantize — it *re-containers* a diffusers FLUX.1 or
FLUX.2 Klein pipe into per-component HFQ files with the dtype policy the
diffusion loaders dispatch on. The packs are the only form the daemon loads; a
pipe directory is the packer's input, never a model path:

```bash
hipfire-quantize --flux-pipe <pipe_dir> --output <base>.hfq
#   <base>-transformer.hfq   arch 40  (F16 weights / F32 bias+scale)
#   <base>-t5.hfq            arch 41  (T5-XXL; shared across variants)
#   <base>-clip.hfq          arch 42  (CLIP-L)
#   <base>-vae.hfq           arch 43  (shared across variants)
hipfire-quantize --flux-pipe <pipe_dir> --output t5-xxl.hfq --flux-component t5
```

The family is detected from `transformer/config.json`. A FLUX.2 (Klein) pipe
packs as `<base>-transformer.hfq` (arch 45), `<base>-qwen3.hfq` (arch 46, the
Qwen3 text encoder) and `<base>-vae.hfq` (arch 43). Component ids:
`docs/architecture-ids.md` § Image-generation component ids.

## From HuggingFace

```bash
hipfire quantize Jackrong/Qwopus3.5-4B-v3 \
    --both \
    --upload schuttdev/hipfire-qwopus-4b \
    --create-repo \
    --install \
    --register qwopus:4b
```

Downloads into `~/.hipfire/hf-cache/`, quantizes once per `--format`, optionally
uploads, copies into `~/.hipfire/models/`, and registers a local alias.

## From a local safetensors directory

```bash
hipfire quantize ./my-finetune/ --format mq4 -o my-finetune.mq4
```

Directory needs `config.json` plus one or more `.safetensors` files. The
quantizer accepts many architectures; inference only runs if a carrier/loader
exists for that `arch_id` (see [architecture-ids.md](architecture-ids.md)).

## From GGUF (deprecated, removal in 0.5.0)

**Deprecated since 0.4.0, removal in 0.5.0: GGUF→mqN is lossy double quantization; quantize from the original safetensors instead.**
A GGUF weight input prints
`warning: GGUF weight input is deprecated and will be removed in 0.5.0; … (imatrix.gguf input stays supported)`.
Quantize from the original safetensors instead. `--imatrix <file.gguf>` is not affected.

```bash
hipfire quantize ./tinyllama.Q4_K_M.gguf \
    --install --register tinyllama:1b-gguf
# default --format hf4 → …hf4

hipfire quantize ./qwen3.5.Q4_K_M.gguf --format mq4 \
    --install --register q35:9b-gguf
```

GGUF tensor names are rewritten to HuggingFace-style names so
`load_weights_hfq` works. Tokenizer fields (`tokenizer.ggml.tokens` / merges /
bos/eos / model) are preserved under `gguf_meta` in the metadata blob;
`Tokenizer::from_hfq_metadata` reads them — the original GGUF need not stay on
disk.

Per-tensor selection in the GGUF pipeline:

| Tensor | Format |
|---|---|
| 1D norm / scale | F16 |
| embedding (`token_embd`) | Q8F16 |
| 2D weight, `K % 256 == 0` | chosen `--format` |
| 2D weight, K not multiple of 256 | HFQ4-G128 fallback (no rotation) |

Supported source GGUF dequant types: `Q4_0`, `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K`,
`F16`, `BF16`, `F32`. Unsupported (quantizer errors on encounter): `Q5_0`,
`Q5_1`, and IQ* families.

### Quality caveat (GGUF)

GGUF is **double quantization**: dequant already-lossy weights, then requantize.
Expect worse quality than full-precision safetensors of the same model. Prefer
HF6/MQ6 if disk allows, or quantize from original safetensors.

## CLI flags (`hipfire quantize`)

| Flag | Purpose |
|---|---|
| `--format <fmt>` | Repeatable. Default: `mq4` → **qt 44 MQ4G256V2** (safetensors); `hf4` (GGUF). Thin allowlist only. |
| `--both` | Shorthand for `--format mq4 --format mq6` |
| `-o, --output <path>` | Single-format output path |
| `--output-dir <dir>` | Multi-format output directory |
| `--stem <name>` | Override output basename |
| `--upload <owner/repo>` | Push outputs to HuggingFace |
| `--create-repo` | Create HF repo if missing |
| `--install` | Copy into `~/.hipfire/models/` |
| `--register <tag>` | Local alias for `hipfire run <tag>` |

Build the binary if missing:

```bash
cargo build --release -p hipfire-quantize
```

The CLI searches `target/release/hipfire-quantize` and
`~/.hipfire/bin/hipfire-quantize`.

## Direct `hipfire-quantize` (advanced)

Full alias set and product-ladder controls. Common extras:

| Flag | Role |
|---|---|
| `--input` / `--output` | Paths |
| `--format` | Full set: `mq4`/`mq4v2` → qt44; `mq4v1`/`mq4g256`/`magnum` → qt13; `mq4c`; `mq{6,5,3,2}v2`; `mq3`; `hf4`/`hf6`; `hfp4`/`mfp4`/`mfp4e8`; Lloyd; `tq2`/`bq1`; … |
| `--tier <xt\|base\|pro>` | Product rung (Qwen3.8 ladder): `xt` keeps lm_head at base codec; `base` lifts embed+lm_head; `pro` also lifts `ssm_out`. embed/conv1d stay Q8; structural tensors stay F16. See [ladder.md](quant-formats/ladder.md). |
| `--fixed-tier <spec>` | Per-class codec overrides, e.g. `lm_head:mq6v2,ssm_out:mq6v2`. Classes: `lm_head,embed,router,attn,ssm_out` (plus `attn_full` where applicable). Dtypes: `q8,mq2v2,mq3v2,mq4v2,mq5v2,mq6v2` (+ limited legacy). Env: `HIPFIRE_FIXED_TIER`. |
| `--imatrix <gguf>` | GGUF-format imatrix for activation-aware recipes |
| `--hessian-dir <dir>` | GPTQ-E8 Hessians |
| `--awq` / `--awq-alpha` / `--awq-a4-aware` | AWQ pre-scale (default alpha 0.55). `--awq-a4-aware` is MQ4V2-only and selects an alpha per shared activation group from 0.35/0.45/0.55/0.65/0.75 by sampled W4A4 block-output error, using the imatrix RMS profile and the runtime-exact four-candidate `block_i4_128` fake quantizer. Fused qkvza/gate-up siblings share the anchor projection's alpha so their single runtime inverse scale remains valid. |
| `--kmap-dense` / `--kmap-mode` / `--no-kmap` / `--uniform` | K-map promotion policy |
| `--q8-router` / `--no-q8-conv1d` | Protect routers / conv1d |
| `--reap-overlay` / `--reap-bake` / `--reap-out` / `--reap-arch` | REAP plan paths |
| `--arch-id` / `--force-arch-id` | Override written arch_id (qwen3* pillar guarded) |
| `--threads` / `HIPFIRE_QUANT_THREADS` | CPU thread cap (default ~80% cores) |
| `--include-vision` / `--vision-quant` | Vision tensors |
| `--allow-mq2` / `--allow-mq*-lloyd` | Research gates (v1 mq2 / Lloyd) |
| `--qwen4-flash-next` | Native Qwen4 / Qwen3.8-Flash-Next streaming producer; see [Qwen4 Flash-Next streaming producer](#qwen4-flash-next-streaming-producer). Reads no legacy recipe flag. |
| `--qwen4-component-mode production\|compact-fixture` | Only with `--qwen4-flash-next`; default `production`. `compact-fixture` is a non-production bounded fixture mode. |
| `ple-hot-rank` / `ple-hot-pack` / `trunk-overlay` | Standalone subcommands (first argument), not flags: PLE hot-row sidecars and the trunk overlay sidecar; see the sections below. |

**Dense-only gate:** `mq{2,3,5,6}v2` refuse MoE `arch_id`s. Use legacy
`mq{2,3,5,6}` or `mq4`/`mq4v2`/`mq4c` for MoE, or a dense checkpoint.

**K-map Promote6 follows the MQ4 container family on audited loaders.** With
`mq4`/`mq4v2` or V2 Lloyd MQ4 on the audited loader families (Llama
`arch_id` 0/1, Qwen3.5 dense/MoE 5/6, Qwen2 7/8), promoted attention,
routed experts, shared experts and shared-down weights use MQ6G256V2
(qt47). Loader families whose qt47 support is not verified (MiniMax 10,
LFM2-MoE 11, Cohere2-MoE 12, Gemma4 13/22, Glimmer 14/23, and any other
unaudited `arch_id`) keep their existing encoder routes: ordinary
`mq4`/`mq4v2` and nonexpert V2 Lloyd Promote6 weights retain MQ6G256
(qt15), but V2 Lloyd stacked experts remain unpromoted MQ4G256 (qt13).
Legacy MQ4 and compact `mq4c` also keep MQ6G256 (qt15); GGUF promotions
remain qt15. Non-256-aligned weights still fall back to Q8. This does
not rewrite existing artifacts or alter their loader routes.
MQ6 V2 is not a layout-only conversion: its two fp16 affine grids per
256-weight group differ from V1's single f32 grid. Both occupy 200 bytes;
requantization can change decoded tokens.

Example — Qwen3.8 product ladder cell (base MQ4V2):

```bash
hipfire-quantize --input ./qwen3.8-27b --output qwen3.8-27b.mq4v2.base.hfq \
  --format mq4v2 --tier base
# mq2 pro style fixed-tier (illustrative; MQ2V2 is not a product body):
#   --format mq2v2 --tier pro --fixed-tier lm_head:mq6v2,ssm_out:mq6v2
```

### Multi-slot (`serve.multi_slot`) recipe admissibility

The experimental multi-slot engine admits projections PER SITE, not per
layer: a fused group (QKVZA / QKV / gate+up) keeps its fused launch when
every member shares a container and otherwise falls back to per-weight
plain GEMMs; residual roles (wo, w_down) and the lm_head dispatch on their
own dtype. Admitted projection containers: `Q8_0`, `MQ4G256`/`HFQ4G256`,
`MQ4G256V2`, `MQ4CG256`, `MQ6G256`/`HFQ6G256`, `MQ6G256V2`, `MQ5G256V2`,
`MQ3G256`, `MQ3G256V2`, `MQ3G256Lloyd`, `MQ2G256V2` (the MQ3/MQ6-v1/Lloyd
members are uniform-group-only — no plain per-projection key). lm_head adds
`HFQ3G256` and `F16`. Every `--tier` and `--fixed-tier` recipe on this page
that stays inside those containers loads on slots — including the mixed
`--tier pro` body (MQ4G256V2 projections + Q8_0 wo/embed/lm_head/conv1d).
Embedding lookup admits only the loaded formats `Q8_0`, `F32`, `HFQ4G256`
and `HFQ4G128`. Supported host-decoded embeddings arrive as `F32`: F16,
BF16, TQ2G128, BQ1G128 and MQ*G256V2 (quant types 44/45/47–50); other
embedding formats still refuse with a named error.
Refused projection containers: Lloyd-V2 (qt52), PARO, E8/MFP4 and
G128-rotated containers. fp32 DeltaNet state also refuses with a named
error. AWQ sidecars are refused inside a mixed-dtype group (each scale
needs its own rotated input). KV tiers: q8 (auto/unset default on every
GPU), fwht{2,3,4} (`asymN`/`turboN` aliases; bare `turbo` = fwht3),
legacy-asym{2,3,4}, native bf16/f16 (f16 is slots-only), and explicit fp8
on exact gfx1201 only — `--kv-k`/`--kv-v` splits are not supported by slots.

### Vision-tower sidecar (`qwen3.8-27b-vision.hfq`)

The Qwen3.8-27B vision tower ships as a shared sidecar beside the trunk so
every text quant tier serves images
without requantizing the trunk. Build it from the tower HF dir (needs
`config.json` + the `model.visual.*` shard, e.g.
`model-00001-of-00018.safetensors`):

```bash
hipfire-quantize --input ./qwen3.8-27b-tower \
    --include-vision --include-prefix model.visual. \
    --output ~/.hipfire/models/qwen3.8-27b-vision.hfq
# shorthand for the same two flags:
# hipfire-quantize --input <tower-dir> --vision-only \
#     --output ~/.hipfire/models/qwen3.8-27b-vision.hfq
```

Pack contract (pinned by `cargo test -p hipfire-quantize --lib vision`):

- ONLY the 333 `model.visual.*` tensors: `patch_embed.proj.{weight,bias}`
  (2), `pos_embed.weight` (1), 12 per block × 27, merger 6.
- Matrices → F16 (qt=1, 111 tensors); norms/biases/`pos_embed` → F32
  (qt=2, lossless widen — 222 tensors). `--format` is ignored for vision:
  `should_quantize` is false for the whole group, so tower tensors never
  enter an MQ/HFQ branch. The loader's `load_f16_gpu` / `load_f32_*` arms
  consume qt=1/qt=2 directly.
- `arch_id` 5, `has_vision: true`, `config.vision_config` carried from the
  source `config.json` (metadata is built before the include-prefix filter
  runs, so the filter cannot strip the config) plus pixel-budget keys merged
  from `preprocessor_config.json` when present.

Census the artifact (CPU-only, no GPU):

```bash
cargo build -p hipfire-quantize --example hfq_dump --release
target/release/examples/hfq_dump ~/.hipfire/models/qwen3.8-27b-vision.hfq | head -5
# arch_id   : 5, n_tensors : 333, metadata carries has_vision + config.vision_config
target/release/examples/hfq_dump ~/.hipfire/models/qwen3.8-27b-vision.hfq | grep -c 'qt=1 '
# 111 (F16 matrices)
target/release/examples/hfq_dump ~/.hipfire/models/qwen3.8-27b-vision.hfq | grep -c 'qt=2 '
# 222 (F32 norms/biases/pos_embed)
target/release/examples/hfq_dump ~/.hipfire/models/qwen3.8-27b-vision.hfq \
  | grep 'qt=' | grep -cv 'model\.visual\.'
# 0 tensor rows outside the tower
```

Graded MoE and E8 recipes are intentionally outside the thin `hipfire quantize`
help surface so accidental low-quality artifacts are harder to produce.

### Qwen4 developer tiny checkpoints (`--qwen4-dev-tiny`, developer build only)

`hipfire-quantize --qwen4-dev-tiny source-exact|packed --input <dir> --output <file>`
converts a **local** safetensors directory (never `hf://`) into one complete HFQM
artifact of a synthetic Qwen4 developer profile. The route exists only in a build
with the explicit Cargo feature `qwen4-dev-profiles` (off by default, never part of
a release build): `cargo build -p hipfire-quantize --features qwen4-dev-profiles`.
It is exclusive with `--qwen4-flash-next`, `--qwen4-component-mode` and every legacy
recipe flag, and reads only `--input` and `--output`. The input `config.json` must
carry the explicit `hipfire_profile` marker (the S0 tiny-MoE / tiny-dense fixtures
in `crates/hipfire-arch-qwen4/reference_oracle/QWEN4_PROFILES.md`); the pinned
Flash-Next profile is refused here and is produced only by `--qwen4-flash-next`.

- `source-exact` stores every tensor, the PLE tables included, as its BF16 source
  bytes (recipe id `synthetic-source-bf16`).
- `packed` (recipe id `synthetic-packed`) stores each role in the Flash-Next format
  its own reduction width `K` tiles, and BF16 otherwise: token embedding Q8F16
  (`K % 32`), language head MQ6G256V2 (`K % 256`, else Q8F16 if `K % 32`), trunk
  attention/GDN projections Q8F16 (`K % 128`), dense FFN gate/up/down Q8F16
  (`K % 32`), PLE rows Q8F16 (row width `% 32`), routed experts gate/up MQ4G256V2 +
  down MQ4G128V2 **together** (hidden `% 256` and expert width `% 128`, else both
  BF16). Router, shared expert, hyper-connection mixers, PLE projections, MTP
  connectors, norms and convolutions stay BF16 exactly as in Flash-Next. A profile
  in which no role tiles any format (the width-16 tiny profiles) is refused with the
  unsupported roles named; it is never written as a "packed" artifact, so use
  `source-exact` there. The width-5120 profiles do pack.

The transaction is the Flash-Next one: the index is planned from the typed manifest
and the safetensors headers, a temporary file is written with bounded chunks,
fsynced, source-identity checked, reopened, admitted through
`hipfire_arch_qwen4::admit_hfqm_artifact` (which must accept exactly the planned
tags and byte extents; the predicted and actual file length must be equal), and only
then published without replacing an existing file. The recipe is recorded in the
metadata as `qwen4_recipe.id`; the runtime selects the packed manifest by that id
and by nothing else. Synthetic artifacts are developer evidence only: they do not
establish Qwen4-27B compatibility or GPU execution support.

## Qwen4 Flash-Next streaming producer

`hipfire-quantize --qwen4-flash-next --input <source> --output <file>` writes the
native Qwen4 (`arch_id` 16) HFQM artifact. It is `hipfire-quantize` direct only: the
thin `hipfire quantize` wrapper does not expose it. Source: `crates/hipfire-quantize/src/cli.rs`
(flags, `--help`) and `crates/hipfire-quantize/src/qwen4.rs` (producer).

- **Input** (`--input`): a local directory of `.safetensors` shards, a single local
  `.safetensors` file, or the **immutable** remote form
  `hf://OWNER/REPO@<40-hex-commit>`. The revision must be exactly 40 hexadecimal
  characters: `hf://OWNER/REPO` without a revision, a branch such as `main`, a tag,
  or a short SHA is rejected, so an artifact can never depend on mutable
  repository state. A remote source is read with bounded HTTP range requests; each
  response's reported revision must equal the pinned one and the source identity
  is re-checked before publication. The endpoint is `HIPFIRE_HF_BASE`, else
  `HF_ENDPOINT`, else `https://huggingface.co`; a bearer token is taken from
  `HF_TOKEN`, else `HUGGING_FACE_HUB_TOKEN` (an empty variable counts as unset).
  A token holding a space, newline, quote or non-ASCII character is refused
  before any request, with an error naming the variable.
- **Production admission** is the pinned checkpoint: exactly 131 safetensors
  shards and 1,658 source tensors (1,655 BF16 + 3 I64); the 333 vision-tower tensors
  are not written, so the output has 1,325 entries.
- **`--qwen4-component-mode`**: `production` (default) keeps those pinned counts.
  `compact-fixture` is explicitly non-production: a bounded local fixture mode for
  exercising the full transactional writer, with the production admission counts
  (shards, source tensors, output entries) not applied.
- **No legacy recipe flags.** The recipe is fixed by the Qwen4 manifest. The route
  accepts only `--input`, `--output`, `--threads` and `--qwen4-component-mode`; any
  other flag given explicitly on the command line (`--format`, `--imatrix`, AWQ,
  symmetric-requant, `--tier`, `--fixed-tier`, `--hessian-dir`, …) is refused, not
  ignored, with `--<flag> has no effect with --qwen4-flash-next; remove it`. It also
  conflicts with `--flux-pipe`.
- **Always** includes typed PLE metadata, all 128 PLE shards, and the native MTP
  experts.
- **Per-role formats** (`Qwen4Manifest::build`, `crates/hipfire-arch-qwen4/src/weights.rs`,
  shared by producer and loader): routed experts gate/up MQ4G256V2 (qt 44, hidden
  `K % 256 == 0`) and down MQ4G128V2 (qt 53, row-local per-128 groups; K = 640
  cannot tile 256); `lm_head` MQ6G256V2 (qt 47); `embed_tokens` and the rank-2 trunk
  attention/GDN projections Q8F16 (qt 3; developer env `HIPFIRE_QWEN4_TRUNK_TIER=mq6`
  declares the trunk at MQ6G256V2 instead, see [env-vars.md](env-vars.md)); the 128
  PLE n-gram shards Q8F16 rows; hyper-connection mixers and the MTP connector
  matrices stay BF16; the three PLE hash arrays are raw I64 records (qt 54, not a
  quantized weight). Exact stored bpw per format is in
  [QUANTIZATION.md](QUANTIZATION.md#shipped-xts-artifacts); it counts group headers
  only, not whole-model bpw.
- **Transaction.** The index is planned from the safetensors headers, a temporary
  file is written with bounded chunks, fsynced, source-identity checked, reopened
  and validated against the plan, then published without replacing an existing
  file (an existing `--output` is refused). A destination-capacity preflight
  runs first; `HIPFIRE_QWEN4_CAPACITY_BYTES` overrides the available-bytes figure it
  compares against. On success one `qwen4: wrote …` summary line (entries, experts,
  PLE shards, resident / external PLE bytes, predicted size, scratch high-water) goes
  to stderr.

The direct producer does not claim to reproduce the shipped XTS file: the pinned
Flash-Next XTS artifact is defined by size and SHA-256 in
[AGENTS.md](../AGENTS.md) §5 (its experts are GPTQ-solved, a calibrated recipe this
route does not read). Check any candidate against that pin before comparing results.

### PLE row sidecars (`.ple-q4`, `.ple-q8`)

The Flash-Next per-layer-embedding (PLE) n-gram table is 128 shard tensors of
2,500,012 rows × 160 weights. The loader can read those rows from the artifact
or from a sidecar next to the model file:

| File | Rows | Stored bpw (group bytes × 8 ÷ weights) |
|---|---|---:|
| `<model>.ple-q8` | the exact Q8F16 rows (34 B per 32 weights) | 8.5 |
| `<model>.ple-q4` | 4-bit Lloyd-Max rows derived from them; per 32-weight block an f16 scale + 16 B of labels (18 B) | 4.5 |

A sidecar is bound to its base file (sha256 of the header and tensor index, and file
size); the loader refuses a mismatched sidecar. Command surface (`crates/hipfire-cli/src/ple.rs`;
full behavior in [CLI.md](CLI.md#flash-next-split-ple-layout)):

```bash
hipfire ple derive [--format q4] [--rank PATH] <model>   # write <model>.ple-q4 from the Q8 rows (q8 is not derivable)
hipfire ple split  [--out PATH] [--keep-original] <model> # monolithic model -> <base> without PLE shards + <base>.ple-q8
hipfire ple rebind --old-base PATH --new-base PATH --source PATH  # carry a q4 sidecar to a repacked base with identical embedded PLE rows
hipfire pull <tag> --ple auto|q4|q8|all|none              # which sidecar(s) to download for a split registry entry
```

**Format selection** is `memory.ple_format` (`auto` \| `q4` \| `q8`, compat env
`HIPFIRE_PLE_FORMAT`):

- `auto` (default) resolves to **q4 only when the GPU architecture is exactly
  `gfx1151`**, and to **q8 on every other architecture** (an unknown architecture
  resolves to q8). `auto` never derives and never downloads: if the preferred file
  is absent it uses the other format when present and prints a note, so a split
  base whose only sidecar is `.ple-q4` loads with it; the q4 table is never
  derived automatically. The q8 preference off gfx1151 is deliberate: q4 is
  lossy and its current evidence is a gfx1151 WT2 point estimate (KLD 0.104865
  q4 vs 0.103252 q8), not a paired confidence interval on the XTX (gfx1100,
  code24), and no performance bottleneck there is traced to the PLE format.
- `q4` always uses the 4-bit rows. A load never derives `<model>.ple-q4`: when it is
  missing the load is refused with both searched locations and the remedies
  `hipfire pull <tag> --ple q4` / `hipfire ple derive --format q4 <model>` (the
  explicit step that derives it from the local Q8 rows).
- `q8` always uses the exact rows. This is the opt-out from the q4 default on
  gfx1151: set `HIPFIRE_PLE_FORMAT=q8` or `memory.ple_format = "q8"`. A split base
  then needs `<model>.ple-q8` (`hipfire pull <tag> --ple q8`); a missing file is
  refused with that hint, never substituted.

`hipfire pull --ple none` only skips the download; it does not change which format
the loader selects. Env and config reference:
[CONFIG.md](CONFIG.md#qwen4-ple-row-tables), [env-vars.md](env-vars.md)
(`HIPFIRE_PLE_FORMAT`).

At run time the q4 sidecar's pages are warmed into the page cache by
`memory.ple_prewarm_bytes` (`auto` by default). `auto` follows the host RAM left
after the model is loaded (`MemAvailable` − CMA region − 8 GiB, capped at the
table): it is additionally capped at 15 GiB on unified-memory GPUs and on
discrete GPUs whose experts are all resident in VRAM, and uncapped only on a
discrete GPU with host-placed experts. An explicit byte count is never capped
at 15 GiB. See [CONFIG.md](CONFIG.md#qwen4-ple-row-tables).

### Trunk overlay sidecar (`.trunk-overlay`)

The zab trunk tier of the stock Flash-Next XTS artifact ships as
`qwen3.8-flash-next.mq4-xts.trunk-overlay` (305,543,168 B) in
`hipfire-models/qwen3.8-flash-next`, pinned in the client (not a registry schema
change) to the stock base by exact size and SHA-256. `hipfire pull` fetches and verifies it next to the base
for the four stock tags (`qwen3.8:flash-next`, `-mq4`, `-gptq3`, `-mq4-xts`); `hipfire rm`
removes it, and listings never show it as a model. It is an HFQM container of
pre-encoded symmetric QT44 (MQ4G256V2, qt 44) tensors that replace the base's MQ6
`linear_attn.in_proj_{z,a,b}` matrices at load (108 tensors) and arm the trunk A4 default
its manifest names, `gfx1151-zab-v1`.

`HIPFIRE_QWEN4_TRUNK_OVERLAY` selects the policy: `auto` (default, also unset) applies a
present sidecar **only on exact gfx1151**; `0` never applies it (kill switch: the
stock base weights run); `1` applies it on any architecture (the A4 route itself
still arms by default only on gfx1151); any other value fails the load. A sidecar
that fails binding is refused with one log line and the base weights run.
`HIPFIRE_QWEN4_TRUNK_IU4=0` disables the A4 route but keeps the overlaid weights.
No speed or quality figure is claimed here; see [CONFIG.md](CONFIG.md#qwen4-trunk-overlay-sidecar),
[CLI.md](CLI.md#flash-next-trunk-overlay) and [env-vars.md](env-vars.md).

Build one from a base model and a directory of pre-encoded QT44 payloads
(`<tensor name>.bin`, exactly the QT44 size `rows × (K/256) × 136` B; this command does
not encode them):

```bash
hipfire-quantize trunk-overlay --base <base.hfq> --qt44-payload-dir <dir> \
    --suffix <tensor-name-suffix> [--suffix ...] \
    --trunk-iu4-default gfx1151-zab-v1 --output <base>.trunk-overlay
```

Base tensors whose names end with any `--suffix` are replaced. The sidecar binds to the
base's file length and header/index digest, records the base's full sha256, is
re-bound and re-read as a self-check, and refuses to overwrite an existing output; the
report is printed as JSON on stdout. Source: `crates/hipfire-quantize/src/trunk_overlay.rs`,
`crates/hipfire-runtime/src/trunk_overlay.rs`.

### PLE hot-row tools

Two offline, no-GPU subcommands build the optional hot-row store inputs
(runtime budget: `memory.ple_hot_store_bytes`, off by default; see
[CONFIG.md](CONFIG.md#memory)):

```bash
hipfire-quantize ple-hot-rank --model <MODEL.mq4> --out <MODEL.mq4>.plehot \
    [--window 8192] [--threads N] [--counts-in FILE.u32 ...] <corpus.jsonl>...
hipfire-quantize ple-hot-pack --model <MODEL.mq4> [--rank <MODEL.mq4>.plehot] \
    [--out <rows>.plehotrows] [--ple-format q4|q8] [--gib 40] [--tier-group 5]
```

- `ple-hot-rank` tokenizes a JSONL corpus (`{"text": …}` encoded raw, or
  `{"messages": […], "tools": […]?}` rendered with the model's chat template with thinking
  off; unparseable lines are counted and skipped), scores each PLE row by how many
  `--window`-token windows touch it, and writes the 256-bucket `<model>.plehot` rank
  sidecar (bucket 0 = hottest). It needs at least one corpus file or `--counts-in`
  (precomputed u32 little-endian per-row counts, summed into the score), and works on a
  split base because it reads only metadata.
- `ple-hot-pack` copies the hottest rows verbatim into `<rows source>.plehotrows`, in
  tiers of `--tier-group` buckets (2–256, default 5). `--gib` (default 40) sets the
  data budget; the number of kept buckets is the largest K ≤ 256 whose rows fit.
  `--ple-format q4|q8` cuts from the `.ple-q4` / `.ple-q8` sidecar instead of the
  model's embedded rows (it never derives; run `hipfire ple derive` first, and a
  rank-order q4 table is its own pack and is refused).

`hipfire rm` removes `.plehot` / `.plehotrows` with the model; they are never listed or
served as models. Sources: `crates/hipfire-quantize/src/ple_hot_rank.rs`,
`crates/hipfire-quantize/src/ple_hot_pack.rs`.

## Runtime cost

Quantization is CPU- and memory-bandwidth bound. Wall time and peak memory depend on
tensor size, target format, calibration, and the thread cap, so no fixed numbers
are given; run on a small model first to gauge cost, and lower the thread cap if
the machine becomes memory-bandwidth constrained.

## Deprecated: CASK sidecar (removal in 0.5.0)

CASK / TriAttention eviction is deprecated and will be removed in 0.5.0; it is
not a post-quantize step and not supported. Until then,
`hipfire sidecar-gen <model> --corpus <file>` still writes
`<model>.triattn.bin` (with a deprecation warning); nothing
attaches it unless `cask_sidecar` or `cask_auto_attach=true` is set. See
[CONFIG.md](CONFIG.md).

## Related

- Formats, KV, QuantType IDs: [QUANTIZATION.md](QUANTIZATION.md)
- Magnum V2 / ladder / register: [quant-formats/mq4-v2.md](quant-formats/mq4-v2.md),
  [quant-formats/mq-v2-family.md](quant-formats/mq-v2-family.md),
  [quant-formats/ladder.md](quant-formats/ladder.md),
  [quant-formats/qt-register.txt](quant-formats/qt-register.txt)
- Models / aliases: [MODELS.md](MODELS.md)
- CLI surface: [CLI.md](CLI.md) (`hipfire ple`, `hipfire pull --ple`, trunk overlay pull)
- Runtime controls for PLE sidecars and the trunk overlay: [CONFIG.md](CONFIG.md#qwen4-ple-row-tables),
  [CONFIG.md](CONFIG.md#qwen4-trunk-overlay-sidecar), [env-vars.md](env-vars.md)
- Sources: `crates/hipfire-cli/src/main.rs`, `crates/hipfire-quantize`
