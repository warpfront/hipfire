# Models

**Owner:** registry-backed model surface (`docs/INDEX.md`).
**Machine sources:** curated `registry/models.json`; generated and bundled
`registry/v1.json` (loaded by `hipfire-registry`).
**Last checked:** 2026-10-13 against `registry/models.json` (90 tags, 82 aliases) on the 0.4.2 beta line.

This page projects **registry availability**: tags, default artifact filenames, declared download size, and declared VRAM floor. It is **not** a product admission table and **not** a guarantee that every GPU/route runs every tag.

| Concept | Meaning |
|---|---|
| Registry tag | Pull/list name resolved through the bundled v1 registry or a refreshed remote/cache registry (+ aliases). |
| Default artifact | Validated registry `file` field — what `hipfire pull <tag>` fetches into `~/.hipfire/models/`; upgraded clients normalize the exact Flash-Next XTS pin as described below. |
| Runtime support | Whether the daemon/loader/arch crate can load and run the artifact shape (`arch_id`, kernels, Cargo features). Source-of-truth: runtime crates + [`architecture-ids.md`](architecture-ids.md). |
| Admission | Explicit product decision in [`admissions.yml`](admissions.yml). Schema v2 holds exactly one evidence-bound record; no inferred admissions beyond that row. |

`hipfire list -r` prints the live registry plus local availability. Prefer that command when sizes change; this page is a checked narrative, not a second registry.

The binary embeds `registry/v1.json` and may refresh the remote registry into
`~/.hipfire/registry.cache.json` (24-hour TTL). Offline or invalid remote
payloads fall back to cache then the bundle; `HIPFIRE_NO_REGISTRY_FETCH=1`
forces the bundle. The Flash-Next pin-bound normalization applies to bundled,
remote and cached entries alike.

---

## Pull and run

```bash
hipfire pull qwen3.5:9b
hipfire run qwen3.5:9b "hello"
hipfire list -r
```

Default serve pre-warm tag is `qwen3.5:9b` (`CONFIG.md` → `default_model`). Per-tag sampling defaults come from registry `recommended_settings` only (applied by the native CLI request resolver). Cards may set intentional `reasoning_effort` (semantic prompt strength only — never a budget). Effort-native families (Qwen3.8, Ornith 1.5, DeepSeek V4, Muse Glimmer) omit `thinking_budget`; absence means no implicit cap. Qwen3.8 and Ornith 1.5 accept hipfire's explicit integer Qwen continuation cap. DeepSeek V4 and Muse Glimmer cap fields are dropped+warned rather than translated into hand-written tokenizer closure. Registry `sampling` blocks are legacy metadata and are not promoted by the native request resolver. Full contract: [`CONFIG.md`](CONFIG.md); HTTP examples: [`SERVE.md`](SERVE.md).

---

## Registry tags (from `registry/models.json`)

Fields: **Tag**, **File** (`file`), **Size GB** (`size_gb`), **Min VRAM GB** (`min_vram_gb`), **Default KV** (registry `default_kv_mode` when set; else empty), **Notes** (`desc`, condensed). `default_kv_mode = "q8"` is the universal fallback and is **not** lowered into config: the global `kv_cache=auto` resolves per [`CONFIG.md`](CONFIG.md#mode-k-and-v) (Qwen: native `fp8` on eligible exact gfx1201, otherwise q8). A non-q8 registry value (DeepSeek V4 `f32`, Maple-Preview `bf16`) is lowered to `kv_cache`. KV names `asymN` / `turboN` / `legacy-asymN` are **deprecated since 0.4.0, removal in 0.5.0**; use `fwhtN` or `q8`.

### Qwen 3.5 dense / hybrid

| Tag | File | Size GB | Min VRAM | Default KV | Notes |
|---|---|---:|---:|---|---|
| `qwen3.5:0.8b` | `qwen3.5-0.8b.mq4` | 0.55 | 2.0 | q8 | MQ4 default small |
| `qwen3.5:0.8b-mq6` | `qwen3.5-0.8b.mq6` | 0.67 | 2.2 | q8 | MQ6 |
| `qwen3.5:2b` | `qwen3.5-2b.mq4` | 1.29 | 2.8 | | MQ4 (registry desc still mentions legacy HF4 naming) |
| `qwen3.5:2b-hf6` | `qwen3.5-2b.hf6` | 1.6 | 3 | | HF6 |
| `qwen3.5:2b-mq3` | `qwen3.5-2b.mq3` | 1.16 | 2.7 | | MQ3 |
| `qwen3.5:2b-mq6` | `qwen3.5-2b.mq6` | 1.63 | 3.1 | | MQ6 |
| `qwen3.5:4b` | `qwen3.5-4b.mq4` | 2.59 | 4.1 | q8 | MQ4 |
| `qwen3.5:4b-mq3` | `qwen3.5-4b.mq3` | 2.25 | 3.8 | q8 | MQ3 |
| `qwen3.5:4b-mq6` | `qwen3.5-4b.mq6` | 3.48 | 5.0 | q8 | MQ6 |
| `qwen3.5:9b` | `qwen3.5-9b.mq4` | 5.31 | 6.8 | q8 | MQ4; common default |
| `qwen3.5:9b-mq3` | `qwen3.5-9b.mq3` | 4.57 | 6.1 | q8 | MQ3 alpha (gfx11/gfx12 noted in desc) |
| `qwen3.5:9b-mq6` | `qwen3.5-9b.mq6` | 7.3 | 8.8 | q8 | MQ6 |
| `qwen3.5:27b` | `qwen3.5-27b.mq4` | 15.0 | 16 | q8 | MQ4 |
| `qwen3.5:27b-mq3` | `qwen3.5-27b.mq3` | 10.7 | 12 | | MQ3 alpha |
| `qwen3.5:27b-mq6` | `qwen3.5-27b.mq6` | 21.4 | 24 | | MQ6 |

### Qwen 3.5 / 3.6 MoE (A3B)

Sizes below are **registry declarations**, not a substitute for runtime MoE layout checks.

| Tag | File | Size GB | Min VRAM | Default KV | Notes |
|---|---|---:|---:|---|---|
| `qwen3.5:35b-a3b` | `qwen3.5-35b-a3b.mq4` | 19.7 | 22 | q8 | 35B / 3B-active |
| `qwen3.6:35b-a3b` | `qwen3.6-35b-a3b.mq4p` | 19.8 | 22 | q8 | Default graded mq4p SKU |
| `qwen3.6:35b-a3b-mq2` | `qwen3.6-35b-a3b.mq2` | 11.6 | 14 | | Floor SKU |
| `qwen3.6:35b-a3b-mq3p` | `qwen3.6-35b-a3b.mq3p` | 17.2 | 20 | | MQ3+P graded |
| `qwen3.6:35b-a3b-mq4p` | `qwen3.6-35b-a3b.mq4p` | 19.8 | 22 | | MQ4+P graded |
| `qwen3.6:35b-a3b-mfp4` | `qwen3.6-35b-a3b.mfp4` | 20.2 | 22 | | MFP4-E8 |
| `qwen3.6:35b-a3b-mq4r` | `qwen3.6-35b-a3b.mq4r` | 18.7 | 22 | | Uniform MQ4G256V1/qt13 Redline speed SKU; zero qt44/graded experts. Dated tok/s in registry text is not a live baseline. |
| `qwen3.6:35b-a3b-mq5` | `qwen3.6-35b-a3b.mq5` | 23.7 | 26 | | Quality SKU |
| `qwen3.6:35b-a3b-mq6` | `qwen3.6-35b-a3b.mq6` | 27.7 | 30 | | Max quality |

The `qwen3.6:35b-a3b*` entries and both `ornith-1.5:35b-a3b*` entries carry an `mtp.file` sidecar name (`qwen3.6-35b-a3b.mtp`, `ornith-1.5-35b-a3b.mtp`). MTP enablement is config/runtime gated (`mtp_mode`, env); registry presence alone is not admission.

### Qwen 3.6 dense

| Tag | File | Size GB | Min VRAM | Default KV | Notes |
|---|---|---:|---:|---|---|
| `qwen3.6:27b` | `qwen3.6-27b.mq4` | 15.0 | 16 | q8 | Registry also lists a `triattn` sidecar (deprecated CASK, removal in 0.5.0: `pull` fetches it, nothing attaches it by default); template not effort-native — `reasoning_effort` dropped+warned, never converted to a cap |
| `qwen3.6:27b-mq3` | `qwen3.6-27b.mq3` | 10.7 | 12 | | MQ3 alpha |

### Qwen 3.8 dense

| Tag | File | Size GB | Min VRAM | Default KV | Notes |
|---|---|---:|---:|---|---|
| `qwen3.8:27b-mq3-xt` | `qwen3.8-27b.mq3-xt` | 11.78 | 13 | q8 | MQ3V2 XT |
| `qwen3.8:27b-mq3` | `qwen3.8-27b.mq3` | 12.62 | 14 | q8 | MQ3V2 base |
| `qwen3.8:27b-mq3-pro` | `qwen3.8-27b.mq3-pro` | 13.18 | 15 | q8 | MQ3V2 Pro |
| `qwen3.8:27b-mq4-xt` | `qwen3.8-27b.mq4-xt` | 14.99 | 16 | q8 | MQ4V2 XT (speed; supersedes legacy `.mq4r`) |
| `qwen3.8:27b-mq4-xts` | `qwen3.8-27b.mq4-xts` | 14.99 | 16 | q8 | MQ4V2 XT, symmetric (H2); int4-activation prefill on gfx1201/gfx1100/gfx1151; canonical dense fixture ([`AGENTS.md`](../AGENTS.md) §5) |
| `qwen3.8:27b` | `qwen3.8-27b.mq4` | 15.66 | 17 | q8 | MQ4V2 base; default; effort-native (`low`/`medium`/`xhigh`, default `xhigh`); uncapped think span unless explicit integer cap |
| `qwen3.8:27b-mq4-pro` | `qwen3.8-27b.mq4-pro` | 16.46 | 18 | q8 | MQ4V2 Pro |
| `qwen3.8:27b-mq5-xt` | `qwen3.8-27b.mq5-xt` | 18.18 | 20 | q8 | MQ5V2 XT |
| `qwen3.8:27b-mq5` | `qwen3.8-27b.mq5` | 18.71 | 20 | q8 | MQ5V2 base |
| `qwen3.8:27b-mq5-pro` | `qwen3.8-27b.mq5-pro` | 19.32 | 21 | q8 | MQ5V2 Pro |
| `qwen3.8:27b-mq6-xt` | `qwen3.8-27b.mq6-xt` | 21.39 | 23 | q8 | MQ6V2 XT |
| `qwen3.8:27b-mq6` | `qwen3.8-27b.mq6` | 21.75 | 23 | q8 | MQ6V2 base |
| `qwen3.8:27b-mq6-pro` | `qwen3.8-27b.mq6-pro` | 22.17 | 24 | q8 | MQ6V2 Pro |

MQ2V2 is not registered. Explicit `qwen3.8:27b-mq4` aliases to `qwen3.8:27b`. Legacy `qwen3.8:27b-fast` / `qwen3.8:fast` alias to `qwen3.8:27b-mq4-xt`. Reasoning contract: [`CONFIG.md`](CONFIG.md) (effort is semantic only; named `thinking_budget` dropped on this family).

### Qwen 3.8 Flash-Next (MQ4 XTS)

| Tag | File (wire registry) | Size GB | Min VRAM | Notes |
|---|---|---:|---:|---|
| `qwen3.8:flash-next-mq4-xts` | `qwen3.8-flash-next.mq4-xts` | 125.3 | 128 | MQ4 XTS; canonical pinned arch 16 fixture |
| `qwen3.8:flash-next`, `qwen3.8:flash-next-mq4`, `qwen3.8:flash-next-gptq3` | `qwen3.8-flash-next-gptq3.mq4` | 125.3 | 128 | Compatibility names for the same MQ4 XTS bytes |

The short user-facing name is `qwen3.8:flash`, an alias for `qwen3.8:flash-next`.
For example: `hipfire pull qwen3.8:flash` and
`hipfire run qwen3.8:flash "hello" --stats`.

The canonical filename is `qwen3.8-flash-next.mq4-xts`. Upgraded clients
normalize only entries from `hipfire-models/qwen3.8-flash-next` with size
`125288544792` and SHA-256
`8b15b6fede7d7c5bfed0db4720a8295bedda51bc93e545fa242bd50d0f200972`.
The old filename and tags remain valid through 0.4.x and at least 90 days;
existing tags keep the old wire filename so old clients do not re-download.
The old HF object remains until explicit EOL approval.

Installed legacy bytes are SHA-verified and linked to the canonical local
name automatically, preserving the old link and external storage targets.
Unsupported/read-only linking uses the verified old path without a download
or payload copy; a conflicting canonical file fails rather than being replaced.
The historical RTN-asymmetric `qwen3.8-flash-next.mq4`
(`qwen3.8:flash-next-rtn-asym`) is a different artifact and is never migrated.
Fixture identity and recipe history: [`AGENTS.md`](../AGENTS.md#pinned-flash-next-bench-fixture).

Other Flash-Next registry entries, retained for provenance and not the default:

| Tag | File | Size GB | Min VRAM | Notes |
|---|---|---:|---:|---|
| `qwen3.8:flash-next-mq4-xts-zab` | `qwen3.8-flash-next.mq4-xts-zab` | 125.1 | 128 | **Local-only** (empty `repo`): not uploaded, not pullable |
| `qwen3.8:flash-next-rtn-asym` | `qwen3.8-flash-next.mq4` | 125.3 | 128 | Previous canonical rung (RTN-asymmetric experts), superseded by MQ4 XTS |
| `qwen3.8:flash-next-mq6q8-pleq8` | `qwen3.8-flash-next.mq6q8-pleq8` | 125.5 | 128 | Earlier canonical rung, superseded |

### DFlash draft artifacts (registry)

| Tag | File | Size GB | Min VRAM | Pairs with (by name) |
|---|---|---:|---:|---|
| `qwen3.5:9b-draft` | `qwen35-9b-dflash-mq4.hfq` | 0.55 | 6 | `qwen3.5:9b` |
| `qwen3.5:27b-draft` | `qwen35-27b-dflash-mq4.hfq` | 0.92 | 16 | `qwen3.5:27b` |
| `qwen3.5:27b-draft-mq3` | `qwen35-27b-dflash-mq3.hfq` | 0.67 | 12 | `qwen3.5:27b` (mq3 draft) |
| `qwen3.6:27b-draft` | `qwen36-27b-dflash-mq4.hfq` | 0.92 | 16 | `qwen3.6:27b` |
| `qwen3.6:27b-draft-mq3` | `qwen36-27b-dflash-mq3.hfq` | 0.67 | 12 | `qwen3.6:27b` |
| `qwen3.8:27b-draft-mq3` | `qwen38-27b-dflash-mq3.hfq` | 0.98 | 16 | `qwen3.8:27b*` (same-bit alt) |
| `qwen3.8:27b-draft-mq4` | `qwen38-27b-dflash-mq4.hfq` | 1.21 | 16 | `qwen3.8:27b*` (recommended controller) |
| `qwen3.8:27b-draft-mq5` | `qwen38-27b-dflash-mq5.hfq` | 1.43 | 16 | `qwen3.8:27b*` (same-bit alt) |
| `qwen3.8:27b-draft-mq6` | `qwen38-27b-dflash-mq6.hfq` | 1.66 | 16 | `qwen3.8:27b*` (same-bit alt) |
| `muse-glimmer:draft` | `muse-glimmer-30b-dflash.mq4` | 1.36 | 26 | `muse-glimmer` / `muse-glimmer:fast` (no filename auto-pairing: attach with `HIPFIRE_DFLASH_DRAFT`) |

DSpark draft sidecars are declared on the DeepSeek V4 Flash entries (`deepseek-v4-flash` → `deepseek-v4-flash-0731-dspark.mq2r`, `deepseek-v4-flash:mq2lloyd` → `deepseek-v4-flash-0731-dspark.mq2lloyd`, `deepseek-v4-flash-preview` → `deepseek-v4-flash-dspark.mq2lloyd`); the preview entry also declares an MTP sidecar (`deepseek-v4-flash-mtp.mq2lloyd`). `speculation` accepts `dspark` ([`CONFIG.md`](CONFIG.md#speculative-decode)); see [`GLOSSARY.md`](GLOSSARY.md) for the DSpark path.

Draft **loading** is registry-driven: `hipfire pull <tag>` fetches the draft sidecar alongside the target, and `dflash_mode` / `speculation` ([`CONFIG.md`](CONFIG.md), [`env-vars.md`](env-vars.md)) decides what happens next — the schema default `dflash_mode` is **`off`** (pull ≠ enable) with one registry-scoped exception: the Qwen3.8-27B tiers that declare a paired draft (`qwen3.8:27b` and the `mq3*` / `mq4*` tiers) get `auto` from the registry layer, so the draft is used when its file is present and the load keeps MTP `auto` when it is absent (an explicit `dflash_mode=off` wins; [`CONFIG.md`](CONFIG.md#qwen38-27b-paired-dflash-default)). Elsewhere `auto` uses the sidecar when present (AR otherwise); `on` requires it and fails the load without it. Override the sidecar with `developer.dflash_draft` / `HIPFIRE_DFLASH_DRAFT` or `run --model-draft`. Watch for `DFlash draft loaded:` in load output. Several tags may share one `dflash.file`; `hipfire rm` keeps that file while another installed declarer still needs it ([`CLI.md`](CLI.md)).

Vision-tower **loading** is registry-driven the same way: every `qwen3.8:27b*` tier declares the shared `vision.file` (`qwen3.8-27b-vision.hfq`, llama.cpp mmproj-style), so `hipfire pull <tag>` fetches the tower once and every text quant tier serves images without requantizing the trunk. The loader opens the sidecar as a separate pack and applies it with the trunk's vision config. `vision_mode` ([`CONFIG.md`](CONFIG.md#vision-tower), [`env-vars.md`](env-vars.md)) decides what happens next — default **`off`** never loads the tower (even an explicit `--vision` is skipped, the daemon's `dflash_mode=off`-style hard override, so text loads pay no tower VRAM); `auto` uses the registry/sibling sidecar when present and stays silently text-only when absent; `on` requires the declared sidecar and fails the load closed without it. A trunk with an embedded tower declares no sidecar and is unaffected by this key. Override per load with `run --vision` / `serve --vision` or `HIPFIRE_VISION_SIDECAR` (empty opts out); a `<trunk-stem>-vision.hfq` file beside the trunk is also discovered. `hipfire rm` keeps the shared file while another installed declarer still needs it ([`CLI.md`](CLI.md)).

MTP-head **loading** is registry-driven too. Every Qwen3.8-27B trunk tier above (`mq3*`, `mq4*`, `mq5*`, `mq6*`) declares the shared `mtp.file` `qwen3.8-27b.mtp` (225,716,224 B, sha256 `f0d46d07ded75abc095ebfdbccc167c68418a6515b187bf810395faa460bd32e`), so `hipfire pull <tag>` fetches it once for all tiers. It is the native `mtp.*` layer of the public `Qwen/Qwen3.8-27B` checkpoint, packed by `hipfire-quantize`'s `mtp_extract --quant mq4` (arch 21). It carries no trunk weights: the head reuses the trunk's embeddings and `lm_head`, which is why one file serves every tier. The DFlash drafts declare no head.
- `speculation` and `mtp_mode` ([`CONFIG.md`](CONFIG.md#speculative-decode)) both default to `auto`, so a resolved head runs on every request. The load logs `MTP head loaded (sidecar …)` and each request logs `drafter=mtp`. `off` never loads the head, and `on` fails the load without it.
- Lookup order: `mtp.file` in the models directory, then `<file>.mtp` beside the model path as typed, then beside the canonical (symlink-resolved) trunk. A head placed beside a symlinked trunk is therefore found.
- The loader refuses a head whose hidden size or vocab differs from the trunk's (for example a 35B-A3B head beside a 27B trunk). `auto` falls back to AR with one log line, and `on` fails the load with the two sets of dimensions.
- On gfx1201 with H2 (`qwen3.8:27b-mq4-xts`), MTP decode against AR measured 1.09–1.65× on six of eight task genres, 0.96× on a tool answer and 0.95× on code editing. Greedy MTP output matches its own verifier's argmax, but it can differ from greedy AR output.
- `hipfire rm` keeps the shared head while another installed tier still declares it ([`CLI.md`](CLI.md)).

Flash-Next trunk-overlay **loading** is pinned in the client, not declared in the registry. The four stock Flash-Next MQ4 XTS tags (`qwen3.8:flash-next`, `-mq4`, `-gptq3`, `-mq4-xts`) resolve to one base artifact (sha256 `8b15b6fe…`, 125288544792 B), and the client carries a compiled-in pin for it naming `qwen3.8-flash-next.mq4-xts.trunk-overlay` (305543168 B, sha256 `df49bc0a…`, always the canonical base file name plus `.trunk-overlay`, whichever legacy spelling the wire entry uses) in `hipfire-models/qwen3.8-flash-next`. `hipfire pull <tag>` fetches it next to the base. There is no registry schema change: older clients keep parsing the published `registry/v1.json`. It is the zab tier as an overlay: an HFQM container with the 108 GDN `in_proj_{z,a,b}` projections encoded as symmetric QT44, bound to the base by file size and index SHA-256, that the Qwen4 loader substitutes for the base's MQ6 tensors and then arms the `gfx1151-zab-v1` A4 trunk route. The bytes are the ones the developer map `HIPFIRE_QWEN4_REQUANT="...=qt44:<dir>"` applies.
- `HIPFIRE_QWEN4_TRUNK_OVERLAY` unset or `auto` applies the overlay only on gfx1151 (the exact architecture of the `gfx1151-zab-v1` trunk default); `0` never applies it (kill switch, the stock base weights load); `1` applies it on any architecture (the A4 route itself still arms by default only on gfx1151; elsewhere `HIPFIRE_QWEN4_TRUNK_IU4` must select it); any other value is a load error.
- `hipfire rm` removes the overlay with its base; the local-only `qwen3.8:flash-next-mq4-xts-zab` tier is the same bytes baked into one file and declares no overlay.

### Qwen3 (non-3.5) dense HF4

| Tag | File | Size GB | Min VRAM | Notes |
|---|---|---:|---:|---|
| `qwen3:0.6b` | `qwen3-0.6b.hf4` | 0.4 | 1 | standard attention |
| `qwen3:8b` | `qwen3-8b.hf4` | 4.1 | 6 | standard attention |

### Fine-tunes on Qwen 3.5 / 3.6 families

| Tag | File | Size GB | Min VRAM | Notes |
|---|---|---:|---:|---|
| `carnice:9b` | `carnice-9b.mq4` | 5.0 | 6 | Hermes tool-use; `default_tool_format=hermes` |
| `carnice:9b-mq6` | `carnice-9b.mq6` | 7.3 | 8 | Hermes MQ6 |
| `carnice:27b` | `carnice-27b.mq4` | 15.0 | 16 | Hermes 27B |
| `carnice:27b-mq6` | `carnice-27b.mq6` | 21.4 | 24 | Hermes 27B MQ6 |
| `qwopus:4b` | `qwopus-4b.mq4` | 2.6 | 4 | Qwopus3.5 v3 |
| `qwopus:4b-mq6` | `qwopus-4b.mq6` | 3.8 | 5 | |
| `qwopus:9b` | `qwopus-9b.mq4` | 5.3 | 6 | |
| `qwopus:9b-mq6` | `qwopus-9b.mq6` | 7.3 | 8 | |
| `qwopus:27b` | `qwopus-27b.mq4` | 15.0 | 16 | |
| `qwopus:27b-mq6` | `qwopus-27b.mq6` | 21.4 | 24 | |
| `qwopus3.6:27b-coder` | `qwopus3.6-27b-coder.mq4` | 15.0 | 16 | registry `default_kv_mode` q8 (`auto` still applies); agentic coder finetune |
| `nex-n2:mini` | `nex-n2-mini.mq4p` | 19.82 | 22 | registry `default_kv_mode` q8 (`auto` still applies); Qwen3.5-35B-A3B agentic MoE finetune |
| `ornith-1.5:35b-a3b` | `ornith-1.5-35b-a3b.mq4` | 19.02 | 22 | registry `default_kv_mode` q8 (`auto` still applies); MQ4G256V2 quality trunk with selective MQ6/Q8 protection; semantic `low`/`medium`/`xhigh` effort (default `xhigh`), uncapped unless an explicit integer cap is set |
| `ornith-1.5:35b-a3b-mq4r` | `ornith-1.5-35b-a3b.mq4r` | 18.70 | 22 | registry `default_kv_mode` q8 (`auto` still applies); uniform MQ4G256V2 Redline SKU, 20,871 qt44 and zero qt13/qt15; same effort contract as the quality trunk. Speed SKU aliases: `ornith-1.5:fast` / `ornith-1.5:35b-a3b-fast` → this tag (Muse/Qwen3.8 `:fast` pattern) |

### Other families (registry)

| Tag | File | Size GB | Min VRAM | Notes |
|---|---|---:|---:|---|
| `deepseek-v4-flash` | `deepseek-v4-flash-0731.mq2r` | 82 | 96 | Current 0731 release, MQ2R (default SKU); KV `f32`; target-aligned DSpark sidecar; default low `reasoning_effort`, uncapped think span (no named budget) |
| `deepseek-v4-flash:mq2lloyd` | `deepseek-v4-flash-0731.mq2lloyd` | 86.2 | 96 | Previous default (MQ2-Lloyd), retained; KV `f32`; DSpark sidecar |
| `deepseek-v4-flash-preview` | `deepseek-v4-flash.mq2lloyd` | 82 | 96 | Prior preview package, retained under an explicit preview identity; declares MTP and DSpark sidecars |
| `minimax-m2.7` | `MiniMax-M2.7.mq2` | 79.2 | 96 | arch_id=10 Mixtral-style MoE |
| `north-mini-code` | `north-mini-code.mq4.hfq` | 16 | 24 | Cohere2-MoE arch_id=12; registry `sampling` block is **inert metadata** today |
| `vibethinker:3b` | `vibethinker-3b.mq4.hfq` | 1.82 | 3.5 | Qwen2 MQ4 |
| `vibethinker:3b-mq6` | `vibethinker-3b.mq6.hfq` | 2.51 | 5.0 | Qwen2 MQ6 |
| `muse-glimmer` | `muse-glimmer-30b.mq4` | 18.61 | 26 | 30B dense + perception encoder; MQ4 quality trunk; always-on Onyx reasoning with strength dial; default uncapped think span |
| `muse-glimmer:fast` | `muse-glimmer-30b.mq4r` | 16.26 | 24 | MQ4R speed SKU (MQ4 body and attention, Q8 lm_head); same Onyx reasoning contract as `muse-glimmer` |

### Bonsai 27B (1-bit / ternary)

| Tag | File | Size GB | Min VRAM | Default KV | Notes |
|---|---|---:|---:|---|---|
| `bonsai:27b-bq1` | `bonsai-27b.bq1` | 3.8 | 5 | q8 | BQ1G128, 1.14 bpw; smallest 27B in the registry |
| `bonsai:27b-tq2` | `bonsai-27b.tq2` | 7.16 | 8 | q8 | TQ2G128 ternary, 2.125 bpw; the `bonsai` alias target |

Registry `desc` carries dated KLD and tok/s figures for these tags; they are registry text, not a live baseline.

### Maple-Preview (natively ternary MoE)

| Tag | File | Size GB | Min VRAM | Default KV | Notes |
|---|---|---:|---:|---|---|
| `maple-preview` | `maple-preview.mq2lloydu` | 6.5 | 12 | bf16 | 20B-A1B natively ternary MoE (arch 15); MQ2G256LloydU packs the body value-exactly; optional `q4k` / `bf16` head packs; reasoning model |

### Image generation (FLUX.1, arch 40)

| Tag | File | Size GB | Min VRAM | Notes |
|---|---|---:|---:|---|
| `flux.schnell:1` | `flux-schnell-transformer.hfq` | 23.8 | 24 | Step-distilled (4 default steps); pull fetches the trunk plus T5 / CLIP / VAE sidecars |
| `flux.dev:1` | `flux-dev-transformer.hfq` | 23.8 | 24 | **Local-only placeholder**: FLUX.1-dev packs are not redistributable and are never uploaded |

Served by `hipfire img <tag>` and `/v1/images/generations`; see [`IMAGEGEN.md`](IMAGEGEN.md) and [`CLI.md`](CLI.md).

### LFM2.5 (registry)

| Tag | File | Size GB | Min VRAM | Notes (registry `desc`) |
|---|---|---:|---:|---|
| `lfm2.5:350m` | `lfm2.5-350m.q8` | 0.38 | 1.9 | 350M dense; default artifact is **Q8** file |
| `lfm2.5:1.2b` | `lfm2.5-1.2b.mq4` | 0.7 | 2.2 | 1.2B Instruct dense |
| `lfm2.5:1.2b-thinking` | `lfm2.5-1.2b-thinking.mq4` | 0.7 | 2.2 | 1.2B Thinking dense |
| `lfm2.5:8b-a1b` | `lfm2.5-8b-a1b.mq4` | 4.66 | 6.2 | 8B-A1B MoE |

Registry `recommended_settings` for LFM tags is low temperature (0.05–0.2) with `repeat_penalty` 1.05 — applied by the CLI resolver. Do not treat a registry `sampling` field as active defaults.

### LFM2.5-24B-A2B — loadable, **no published artifact**

`LiquidAI/LFM2.5-24B-A2B` (40 layers, 64 experts top-4, hidden 2048, 128k vocab,
`conv_L_cache` 3, `num_dense_layers` 2, `use_expert_bias` true) loads on arch 11.
There is **no registry tag and no `hipfire pull` target**: no artifact has been
published, so add nothing to `registry/models.json` until one is.

Recipe that works today (local, `--format mq4`):

```bash
hipfire-quantize --input <LFM2.5-24B-A2B dir> \
                 --output ~/.hipfire/models/lfm2.5-24b-a2b.mq4 --format mq4
# 12,758,948,048 B output: experts + dense projections MQ4G256, embedding
# MQ4G256V2, 38 expert_bias tensors F32, norms Q8F16.
```

Because the model is ~98% routed experts by bytes (11.6 GiB of 12.7 GiB), it does
not fit VRAM on a 12 GB card (or a 16 GB one) whole; it needs the partial-offload
expert tier:

```bash
HIPFIRE_GPU_LAYER_BUDGET=auto HIPFIRE_MOE_EXPERT_BUDGET=auto \
  hipfire run ~/.hipfire/models/lfm2.5-24b-a2b.mq4 "..."
# measured on gfx1101 (12.9 GB): "routed experts host on 13 layers ... device
# weights 9781 MiB of 9947 MiB available", 3366 MiB pinned host RAM.
```

**Verified on this branch** (gfx1100 RX 7900 XT, HIP 7.15): a locally quantized
`lfm2.5-24b-a2b.mq4` (12,758,948,048 B, md5 `542f7998…f422`, quantized with
`--format mq4` from `LiquidAI/LFM2.5-24B-A2B` bf16 safetensors) loads fully
resident and generates coherently — `serve_harness battery` passes all five
turns (think-disabled, ~78 tok/s decode, daemon md5 `b9c04329…ab05`), and a
forced expert-tier spill (`HIPFIRE_MOE_EXPERT_BUDGET=16`, 24 MoE layers
host-placed, 6,732 MiB pinned host) stays coherent. Earlier gfx1101 work on
this branch also ran the model under `gpu_layer_budget=auto` +
`moe_expert_budget=auto` on a 12.9 GB card. A spill changes no numerics —
greedy decode is byte-identical across placements on the fixtures where a
non-degenerate comparison is possible (see
`crates/hipfire-arch-lfm2moe/tests/offload_parity.rs`: 2.6B dense and 8B-A1B
MoE both match a resident baseline byte-for-byte, whole-layer and
routed-expert tiers, with a prompt-variety guard that rejects a saturated
oracle).

**Resolved blocker — LFM2.5 MoE incoherence.** On gfx1100/gfx1101 the published
`lfm2.5:8b-a1b` artifact used to emit degenerate text (saturated to 1–3 distinct
tokens under greedy) while the dense 2.6B stayed coherent, and the 24B showed
the same class of failure. Root cause was
`kernels/src/gemv_hfq4g256_moe_down_k8_indexed_batched_expanded.hip`: the
atomic-free MoE down GEMV was hardcoded to K=512 (two 256-weight groups),
summing only the first 512 columns of each expert's down projection. LFM2's
`moe_inter` (1792 on the 8B, 1536 on the 24B) dropped 5/7–6/7 of the dot
product. The kernel now keeps the K=512 fast path verbatim and loops
`groups_per_row = K/256` groups otherwise; the gfx1151 buffer/row variants were
already `k == 512`-gated in `gemv.rs` and are untouched. Verified on gfx1100:
per-stage intermediates (router sigmoid, top-k, gate/up, rot, down-expanded,
combine) match a NumPy replay of the HFQ weights to cos=1.0000 on every MoE
stage, `prompt_probe` emits 36–41 distinct tokens per 48 greedy steps, and
`hipfire run` on `lfm2.5-8b-a1b.mq4` produces coherent code + reasoning text.
The same K-hardcoding affected every consumer with `moe_inter != 512` — e.g.
MiniMax-M2 (`moe_inter` 1536) reads through this kernel too.

**Reasoning + tool calls (arch 11, sequential + VL paths).** LFM2.5 is a pure
reasoning model — its chat template ends the generation prompt with a literal
`<think>` opener, so generated tokens begin *inside* the reasoning block and
only `</think>` is generated. The arch-11 generate paths detect that opener
(`render_tail_opens_think`) and re-emit `<think>` as display-only text before
the first generated token, giving downstream consumers a well-formed
`<think>…</think>…` block: `serve` splits `reasoning_content`/`content`
(matching vLLM's deepseek_r1-class reasoning parser semantics), the run/chat
stripper works, and `finish_reason` reports `length` (not `stop`) when a
turn is still thinking at max_tokens. Tool calls use LFM's dedicated special
tokens (`\x3C\x7Ctool_call_start\x7C\x3E` / `\x3C\x7Ctool_call_end\x7C\x3E`)
wrapping a Pythonic `[f(k=v, ...)]` call list. Both generate paths suppress
the span at the token level (markers and body never reach the client
stream), parse the body into `tool_calls` on span close
(`emit_text::parse_lfm_tool_call_body`, mirroring vLLM's `pythonic` tool
parser: kwargs → JSON object, positional args dropped, malformed body →
treat-as-text fallback), and emit the calls through the standard
`{"type":"tool_calls"}` event + `finish_reason: "tool_calls"`. An unclosed
span at EOS/max_tokens is re-emitted as raw text and never fabricates a
call. Verified on gfx1100: `serve` returns
`reasoning_content` + `tool_calls[0].function.arguments =
"{\"city\":\"Tokyo\"}"` with `finish_reason: "tool_calls"` for a
system-prompt tool list; streaming emits `reasoning_content` deltas.
### Gemma4 (registry — no published artifact yet)

> **No hipfire-quantized Gemma4 `.hfq`/`.mq4` artifact has been published.**
> The current `registry/models.json` intentionally lists **zero** `gemma4:`
> tags so that no registry row can point at a 404. The architecture notes below
> describe the runtime crate only.

**Architecture ground truth** (`crates/hipfire-arch-gemma4/src/config.rs`
and `crates/hipfire-arch-gemma4/src/gemma4.rs`):

- **Hybrid 5:1 sliding:global** — every 6th layer is global (full) attention.
  The per-layer `layer_types` array is authoritative; on the 12B dense text
  config (`hidden_size=3840`, `num_hidden_layers=48`, `vocab_size=262144`) that
  is 40 sliding + 8 full. Do not assume the period in code.
- **Sliding-window layers:** `window = 1024` (`sliding_window`), `head_dim = 256`
  (`sliding_head_dim` / `head_dim`), `RoPE θ = 10_000` (`sliding_rope_theta`),
  `RopeType::Default` on full head_dim, `attention scale = 1.0` (kernels bake
  `1/√d` so decode pre-scales Q by `√head_dim`), Q/K per-head RMSNorm.
- **Global (full) layers:** `head_dim = 512` (`global_head_dim` / `full_head_dim`),
  `window = 0` (full causal), `RoPE θ = 1_000_000` (`full_rope_theta`),
  `RopeType::Proportional` with `partial_rotary_factor = 0.25` (only first 64 of
  512 dims rotate; remainder NoPE), `attention_k_eq_v = true` — **V shares the
  pre-`k_norm` output of `k_proj`** (no `v_proj` on those layers; `v_norm` is
  weight-less, implemented with a ones-filled scratch buffer).
- **KV head counts (variant-dependent):** 12B text — `num_attention_heads = 16`,
  `num_key_value_heads = 8` (sliding) and `num_global_key_value_heads` defaults
  to sliding when absent; 31B target — `n_heads = 32`,
  `sliding_n_kv_heads = 16`, `full_n_kv_heads = 4`, `sliding_head_dim = 256`,
  `full_head_dim = 512`, `hidden_dim = 21504`. The 12B `hidden_dim` is
  `intermediate_size = 15360`, `dim = 3840`.
- **FFN:** SwiGLU with `gelu_pytorch_tanh` activation, `intermediate_size`
  (above) per-layer.
- **Norms:** Sandwich RMSNorm — `input_layernorm`, `post_attention_layernorm`,
  `pre_feedforward_layernorm`, `post_feedforward_layernorm` per layer plus a
  learned scalar `layer_scalar [1]` at layer end. Gemma4 uses plain `x * w`
  (weights init 1.0); the `HIPFIRE_GEMMA4_NORM_PLUS_ONE=1` toggle bakes the
  Gemma-2/3 `x * (1+w)` form at load time.
- **Embeddings:** `embed_scale = sqrt(hidden_size)` multiplied on every lookup;
  **`lm_head` is TIED to `embed_tokens`** (single GPU allocation, aliased
  `WeightTensor`; `free` skips the head to avoid double-free).
- **Output:** `final_logit_softcapping = 30.0` — `tanh(logits/30)*30` before
  sampling. `norm_eps = 1e-6`, `max_position_embeddings = 262144` (12B).
- **Stop / EOS:** HF `eos_token_id` is the **list** `[1, 106]` (`<eos>` = 1 and
  `<end_of_turn>` / `<turn|>` = 106); parsing it as a scalar drops 106 and lets
  decode loop on `<turn|>` forever. The loader resolves `eos_tok` against
  `["<end_of_turn>", …]` and masks both.
- **Sampling defaults (Gemma card):** `temperature 1.0`, `top_p 0.95`,
  `top_k 64` — the registry will carry these in `recommended_settings` /
  `sampling_profiles` when tags land (today no Gemma rows exist to carry them).
- **Thinking:** boolean only (official default **off**). No native
  `reasoning_effort`; unsupported effort/named-budget values are dropped with a
  warning. See [`CONFIG.md`](CONFIG.md) / [`SERVE.md`](SERVE.md).
- **Crate:** `hipfire-arch-gemma4` (`Gemma4Config`, `Gemma4Weights`,
  `Gemma4State`), `Gemma4Bundle { config, weights, state, eos_tok }` in
  `hipfire-loader`. `Gemma4Carrier` claims arch ids **13** (`gemma4_text`) and
  **22** (`gemma4_unified_assistant` EAGLE drafter) — see
  [`architecture-ids.md`](architecture-ids.md).

The Gemma4 EAGLE drafter (arch 22) is currently gated off (`HIPFIRE_GEMMA4_EAGLE=1` to enable; see [`architecture-ids.md`](architecture-ids.md)).

No Gemma4 rows are registered until a quantized artifact is actually on the Hub and passes `scripts/registry_gen.py` probing: that script checks LFS presence and `size_gb` against the real file and fails the run on a mismatch.

---

## Aliases

String redirects in `registry/models.json` → `aliases` (not separate
downloads). **Partial table** — for the complete surface read that file or run
`hipfire list -r`.

| Alias | Resolves to |
|---|---|
| `qwen3.5` | `qwen3.5:4b` |
| `qwen3.5:latest` | `qwen3.5:9b` |
| `qwen3.5:small` | `qwen3.5:0.8b` |
| `qwen3.5:large` | `qwen3.5:27b` |
| `qwen3.6` / `qwen3.6:a3b` | `qwen3.6:35b-a3b` |
| `ornith` / `ornith-1.5` / `ornith1.5` / `ornith1.5:35b-a3b` | `ornith-1.5:35b-a3b` |
| `ornith-1.5:fast` / `ornith-1.5:35b-a3b-fast` | `ornith-1.5:35b-a3b-mq4r` |
| `qwen3.8` / `qwen3.8:latest` | `qwen3.8:27b` |
| `qwen3.8:fast` / `qwen3.8:27b-fast` | `qwen3.8:27b-mq4-xt` |
| `qwen3.8:27b-mq4` | `qwen3.8:27b` |
| `qwen3.8:draft` / `qwen3.8:27b-draft` | `qwen3.8:27b-draft-mq4` |
| `muse-glimmer:latest` / `muse-glimmer:quality` / `muse-glimmer:30b` | `muse-glimmer` |
| `qwen3` | `qwen3:8b` |
| `carnice` | `carnice:9b` |
| `qwopus` | `qwopus:9b` |
| `qwopus:{4b,9b,27b}-{mq4,hf4}` | matching primary `qwopus:{4b,9b,27b}` tag |
| `deepseek4` / `deepseek-v4` / `deepseek-v4-flash-0731` / `ds4` (also `:0731`) | `deepseek-v4-flash` |
| `deepseek4:mq2r` / `deepseek-v4:mq2r` / `deepseek-v4-flash:mq2r` / `ds4:mq2r` (also `-0731:mq2r`, `:0731-mq2r`) | `deepseek-v4-flash` (MQ2R is the default SKU) |
| `deepseek4:mq2lloyd` / `deepseek-v4:mq2lloyd` / `deepseek-v4-flash-0731:mq2lloyd` / `ds4:mq2lloyd` (also `:0731-mq2lloyd`) | `deepseek-v4-flash:mq2lloyd` |
| `deepseek4:preview` / `deepseek-v4:preview` / `ds4:preview` | `deepseek-v4-flash-preview` |
| `bonsai` | `bonsai:27b-tq2` |
| `qwen3.8:flash` | `qwen3.8:flash-next` |
| `muse-glimmer:30b-draft` / `muse-glimmer:30b:draft` / `muse-glimmer-30b:draft` | `muse-glimmer:draft` |
| `vibethinker` | `vibethinker:3b` |
| `qwen3.5:*-mq4` / `*-hf4` / several `*-hf6` | same-size primary or mq6 tag (see registry) |
| `qwen3.5:9b:draft` etc. | matching `*-draft` tags |

---

## Runtime family map (source, not registry)

Runtime dispatch uses HFQ `arch_id` ([`architecture-ids.md`](architecture-ids.md)). Summary for operators:

| Family | arch_id | Crate | Registry examples |
|---|---:|---|---|
| LLaMA / Mistral / plain Qwen3 path | 0 / 1 | `hipfire-arch-llama` | `qwen3:8b`, `qwen3:0.6b` (HF4 dense) |
| Qwen3.5 dense hybrid | 5 | `hipfire-arch-qwen35` | `qwen3.5:*`, `qwen3.6:27b`, `qwen3.8:27b*`, carnice/qwopus dense |
| Qwen3.5 / 3.6 MoE A3B | 6 | `hipfire-arch-qwen35` | `*:35b-a3b*`, `ornith-1.5:35b-a3b*`, `nex-n2:mini` |
| Qwen2 | 7 | `hipfire-arch-qwen2` | `vibethinker:3b`, `vibethinker:3b-mq6` (support, not admission) |
| DeepSeek V4 Flash | 9 | `hipfire-arch-deepseek4` | `deepseek-v4-flash`, `deepseek-v4-flash:mq2lloyd`, `deepseek-v4-flash-preview` |
| MiniMax-M2 | 10 | `hipfire-arch-minimax` | `minimax-m2.7` |
| LFM2.5 dense **and** MoE | 11 | `hipfire-arch-lfm2moe` | all `lfm2.5:*` |
| Cohere2-MoE | 12 | `hipfire-arch-cohere2moe` | `north-mini-code` |
| Gemma4 text (dense) | 13 | `hipfire-arch-gemma4` | *(no registry tag yet)* |
| Gemma4 unified-assistant (EAGLE drafter) | 22 | `hipfire-arch-gemma4` `drafter` | *(no registry tag; drafter gated off)* |
| Muse Glimmer (dense text) | 14 | `hipfire-arch-muse-glimmer` | `muse-glimmer`, `muse-glimmer:fast` (draft: arch 23) |
| Maple-Preview (ternary MoE) | 15 | `hipfire-arch-maple` | `maple-preview` |
| Qwen3.8-Flash-Next / Qwen4 experimental | 16 | `hipfire-arch-qwen4` | `qwen3.8:flash-next*` |
| FLUX.1 MMDiT | 40 | `hipfire-arch-diffusion` | `flux.schnell:1`, `flux.dev:1` |

**Dense LFM2.5 is supported on arch_id 11.** The LFM config parser treats `num_experts == 0` as dense SwiGLU on every layer (`crates/hipfire-arch-lfm2moe/src/config.rs`). Do **not** claim dense LFM is unsupported.

`hipfire-arch-lfm2moe` is a **non-optional** dependency of `hipfire-loader` / daemon load paths on this tree (see crate `Cargo.toml` graphs). Feature flags on `hipfire-runtime` default set do not list a separate `arch-lfm2moe` toggle the way some other arches do — loader always links the crate.

Capability features (DFlash, PP, MTP, batched prefill, n-gram) are **per-path and often narrower than “model loads”**. Speculation capability inventory: [`speculation-support-inventory.md`](speculation-support-inventory.md) (current inventory; capability is not admission). Product claims need source + [`admissions.yml`](admissions.yml).

### LFM optimized prefill — branch-only, not in this tree

**Branch-only; not shipped.** This tree has no environment-selectable batched LFM prefill path and no LFM2 prefill-batch, prefill-chunk, or gfx1201 decode-fusion environment variables: nothing under `crates/` reads them. LFM prefill and decode use the eager per-token path on every GPU.

Audited branch wording allowed for the optimized path that exists only on a separate branch (and nothing broader):

- Exact cohort: the **350M dense MQ4** fixture (`lfm2.5-350m.mq4`), **not** a generic “all LFM” claim.
- GPU: **gfx1201** only.
- Pin when citing the branch implementation: `lfm-redline@692a726dde53508cb53de1a74c720e75a7c9f33e` (or later branch commits only if re-grounded).

**Planned (not implemented claims here):** Q8-first generic completion of the optimized path, wider LFM cohorts (1.2B / 8B-A1B), multi-GPU, and Phase-4 default-on.
**Admitted (exact one row):** [`admissions.yml`](admissions.yml) schema v2 admits only the sealed gfx1201 LFM2.5-350M MQ4 retained-PM4 plain-AR product route; nothing else. That admission is the sole authority and does not imply a batched-prefill product path.
**Not a current baseline:** any exploratory tok/s tables in designs/plans.

---

## Bring your own

### HuggingFace → quantize → register

```bash
hipfire quantize Jackrong/Qwopus3.5-4B-v3 \
  --format mq4 --install --register qwopus:4b
```

See [`QUANTIZE.md`](QUANTIZE.md) and [`QUANTIZATION.md`](QUANTIZATION.md).

### Local safetensors directory

Requires `config.json` + `.safetensors`. Architectures the **engine** loads are those with arch crates / loaders above; the quantizer may accept more shapes than inference can run.

### GGUF (deprecated, removal in 0.5.0)

GGUF weight input is deprecated since 0.4.0: GGUF→mqN is lossy double quantization; use llama.cpp for GGUF.

```bash
hipfire quantize ./model.Q4_K_M.gguf --install --register my:tag
```

Dequant path support is format-specific (common Q4_0 / Q8_0 / Q4_K / Q6_K / F16 / BF16 / F32). Unsupported GGUF quants fail closed in the quantizer.

---

## On-disk layout

```text
~/.hipfire/models/
  <registry file names>
  optional sibling drafts (DFlash / DSpark / MTP / vision sidecars); legacy .triattn*.bin sidecars (CASK, deprecated since 0.4.0, removal in 0.5.0)
```

Extension hints (loader recognizes several): `.mq4`, `.mq6`, `.mq4p`, `.mq4r`, `.mq2`, `.mq2lloyd`, `.mq2r`, `.mfp4`, `.hf4`, `.hf6`, `.hfq`, `.q8`, and related graded names as produced by quant tooling. Exact dtype routing is loader/kernel source, not this table. Selecting the MQ4R and DeepSeek V4 MQ2R routes by file extension is **deprecated since 0.4.0, removal in 0.5.0** (0.5.0 selects by HFQ metadata; no runtime warning).

---

## Thinking / chat framing

Reasoning is a **three-axis contract** (mode / semantic effort / hard cap) owned
by config and the OpenAI request layer — not by inventing per-tag field meanings:

- Axes, defaults, and family table — [`CONFIG.md`](CONFIG.md)
- HTTP fields, warn+drop metadata, curl examples — [`SERVE.md`](SERVE.md)
- Chat template overrides — `chat_template`, `default_chatml` / env in [`env-vars.md`](env-vars.md)

Registry cards may publish `reasoning_effort` defaults. Effort-native tags should
**not** pin a hipfire named `thinking_budget`; absence means uncapped. Ornith 1.5
uses Qwen3.8-compatible `low`/`medium`/`xhigh` semantic prompt steering (default
`xhigh`) — never a budget reinterpretation. Qwen `<think>` framing is not
universal — Gemma uses boolean thinking (default off), Glimmer uses Onyx channel
strength, DeepSeek uses `thinking.type` + its own effort ladder.

---

## Related

| Topic | Owner |
|---|---|
| Config keys / defaults | [`CONFIG.md`](CONFIG.md) |
| Env vars | [`env-vars.md`](env-vars.md) |
| CLI pull/run/list | [`CLI.md`](CLI.md) |
| Arch IDs | [`architecture-ids.md`](architecture-ids.md) |
| Admissions | [`admissions.yml`](admissions.yml) (schema v2; exactly one earned record) |
| Validation routes | [`VALIDATION.md`](VALIDATION.md) |
| Dated benches | [`BENCHMARKS.md`](BENCHMARKS.md) (tables remain **historical** regardless of admission; admission and measurement classification are independent) |
