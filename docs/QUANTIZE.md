# Quantize

`hipfire quantize` is the user-facing wrapper around the CPU-only
`hipfire-quantize` binary. It converts HuggingFace safetensors, a local
model directory, or a GGUF file into hipfire’s HFQM container (extensions
`.mq4`, `.hf4`, `.mq6`, …). The daemon mmaps the result directly.

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

## From GGUF

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
| `--imatrix <gguf>` | llama.cpp imatrix for activation-aware recipes |
| `--hessian-dir <dir>` | GPTQ-E8 Hessians |
| `--awq` / `--awq-alpha` | AWQ pre-scale (default alpha 0.55) |
| `--kmap-dense` / `--kmap-mode` / `--no-kmap` / `--uniform` | K-map promotion policy |
| `--q8-router` / `--no-q8-conv1d` | Protect routers / conv1d |
| `--reap-overlay` / `--reap-bake` / `--reap-out` / `--reap-arch` | REAP plan paths |
| `--arch-id` / `--force-arch-id` | Override written arch_id (qwen3* pillar guarded) |
| `--threads` / `HIPFIRE_QUANT_THREADS` | CPU thread cap (default ~80% cores) |
| `--include-vision` / `--vision-quant` | Vision tensors |
| `--allow-mq2` / `--allow-mq*-lloyd` | Research gates (v1 mq2 / Lloyd) |

**Dense-only gate:** `mq{2,3,5,6}v2` refuse MoE `arch_id`s. Use legacy
`mq{2,3,5,6}` or `mq4`/`mq4v2`/`mq4c` for MoE, or a dense checkpoint.

Example — Qwen3.8 product ladder cell (base MQ4V2):

```bash
hipfire-quantize --input ./qwen3.8-27b --output qwen3.8-27b.mq4v2.base.hfq \
  --format mq4v2 --tier base
# mq2 pro style fixed-tier (illustrative; MQ2V2 is not a product body):
#   --format mq2v2 --tier pro --fixed-tier lm_head:mq6v2,ssm_out:mq6v2
```

### Vision-tower sidecar (`qwen3.8-27b-vision.hfq`)

The Qwen3.8-27B vision tower ships as a shared sidecar (llama.cpp
mmproj-style) beside the trunk so every text quant tier serves images
without requantizing the trunk. Build it from the tower HF dir (needs
`config.json` + the `model.visual.*` shard, e.g.
`model-00001-of-00018.safetensors`):

```bash
hipfire-quantize --input /home/kaden/.hipfire/hf/Qwen3.8-27B-tower \
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

## Runtime cost

CPU- and memory-bandwidth bound. Rough desktop wall times (order-of-magnitude,
not a guarantee):

| Model size | Wall time |
|---|---|
| ~1B | tens of seconds |
| ~4B | ~1 min |
| ~9B | a few minutes |
| ~27B | several–tens of minutes |

Peak RAM is roughly `max(tensor elements) × 4` (one tensor dequantized to f32).

## After quantizing: CASK sidecar

For long-context CASK eviction calibration:

```bash
hipfire sidecar-gen my-finetune.mq4 --corpus /path/to/corpus.txt
# or via registered tag:
hipfire sidecar-gen finetune:1b --corpus /path/to/corpus.txt
```

Writes `my-finetune.mq4.triattn.bin` beside the model by default. The daemon
does not attach it by default. Set `cask_sidecar` to the exact path, or opt into
sibling discovery with `cask_auto_attach=true`. Set `cask=true` separately only
when core-aware m-folding is intended. See [CONFIG.md](CONFIG.md).

## Related

- Formats, KV, QuantType IDs: [QUANTIZATION.md](QUANTIZATION.md)
- Magnum V2 / ladder / register: [quant-formats/mq4-v2.md](quant-formats/mq4-v2.md),
  [quant-formats/mq-v2-family.md](quant-formats/mq-v2-family.md),
  [quant-formats/ladder.md](quant-formats/ladder.md),
  [quant-formats/qt-register.txt](quant-formats/qt-register.txt)
- Models / aliases: [MODELS.md](MODELS.md)
- CLI surface: [CLI.md](CLI.md)
- Sources: `crates/hipfire-cli/src/main.rs`, `crates/hipfire-quantize`
