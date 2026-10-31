# Models

**Owner:** registry-backed model surface (`docs/INDEX.md`).
**Machine sources:** curated `registry/models.json`; generated and bundled
`registry/v1.json` (loaded by `hipfire-registry`).
**Last checked:** 2026-10-16 against `registry/models.json` (92 tags, 83 aliases) on the 0.4.2 beta line.

This page projects **registry availability**: tags, default artifact filenames, declared download size, and declared VRAM floor. It is **not** a product admission table and **not** a guarantee that every GPU/route runs every tag.

| Concept | Meaning |
|---|---|
| Registry tag | Pull/list name resolved through the bundled v1 registry or a refreshed remote/cache registry (+ aliases). |
| Default artifact | Validated registry `file` field — what `hipfire pull <tag>` fetches into `~/.hipfire/models/`; the Flash-Next MQ4 XTS tags name a split base, and the compiled-in grandfathering of the old monolithic file is described below. |
| Runtime support | Whether the daemon/loader/arch crate can load and run the artifact shape (`arch_id`, kernels, Cargo features). Source-of-truth: runtime crates + [`architecture-ids.md`](architecture-ids.md). |
| Admission | Explicit product decision in [`admissions.yml`](admissions.yml). Schema v2 holds exactly one evidence-bound record; no inferred admissions beyond that row. |

`hipfire list -r` prints the live registry plus local availability. Prefer that command when sizes change; this page is a checked narrative, not a second registry.

The binary embeds `registry/v1.json` and may refresh the remote registry into
`~/.hipfire/registry.cache.json` (24-hour TTL). Offline or invalid remote
payloads fall back to cache then the bundle; `HIPFIRE_NO_REGISTRY_FETCH=1`
forces the bundle. The Flash-Next grandfathered monolithic pin is compiled into
the client, so it applies whichever registry source is in use.

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

Optional split-base PLE map (`ple`, an object keyed by format token: lowercase letters and digits such as `q4` = 4-bit Lloyd-Max rows and `q8` = the exact Q8F16 rows; each value is `{file, sha256, size_bytes}`) names the PLE row sidecars of a split entry; `registry_gen.py` pins every one from the live Hugging Face tree (an LFS sha256 is required). The entry's `file` must be a split base (`<name>-base.<ext>`). A sidecar's `file` is either the per-base `<file>.ple-<key>` or, for the Flash-Next split family (`qwen3.8-flash-next-base.<ext>`) only, the shared `qwen3.8-flash-next.ple-<key>`, which every entry naming it must pin identically. An unknown key (a future `q2`) parses and validates, so a newer registry stays readable; a client plans only the formats it can load (today `q4` and `q8`) and ignores the rest with a note. `q8` is the exact reference: a verified grandfathered monolithic embeds it, and it is the fallback when a lossy format a client wants is unpublished; an explicit `--ple q8` never falls back. `hipfire pull` and the on-demand load download a shared file once, verify it against the pin, and install `<base file>.ple-<key>` (the name the loader opens) as a hard link to it, never a second copy; an existing file under that name that is not the verified shared file is left untouched and reported. `hipfire rm` removes the shared file with the last base that links it. Entries without a `ple` map are unchanged. (This replaces the earlier fixed `ple_q4` / `ple_q8` fields.)

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
| `qwen3.6:35b-a3b-mq4rs` | `qwen3.6-35b-a3b-d53.mq4rs` | 18.7 | 22 | q8 | Symmetric-expert MQ4 (qt44 gate/up + qt53 down, MQ4G256V2/MQ4G128V2). Loads on gfx1201 only, where it takes the int4-activation expert prefill; other GPUs refuse it unless `HIPFIRE_MOE_SYM_IU4=0` accepts the qt53 F32 prefill fallback (about 5x slower than the IU4 route on gfx1201). No MTP sidecar registered. |
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
| `qwen3.8:flash-next`, `qwen3.8:flash-next-mq4`, `qwen3.8:flash-next-gptq3`, `qwen3.8:flash-next-mq4-xts` | `qwen3.8-flash-next-base.mq4-xts` | 70.9 | 72 | MQ4 XTS split base; PLE rows from the shared sidecar pair (below) |
| `qwen3.8:flash-next-mq3-xts` | `qwen3.8-flash-next-base.mq3-xts` | 59.1 | 64 | MQ3 XTS split base: W3 GPTQ on 16,839 of 24,576 routed experts, PLE rows from the shared sidecar (below) |
| `qwen3.8:flash-next-mq2-xts` | `qwen3.8-flash-next-base.mq2-xts` | 46.9 | 48 | MQ2 XTS split base: 6,306 hot MQ4 XTS experts, E8P 2-bit on the other 18,270, PLE rows from the shared sidecar (below) |
| `qwen3.8:flash-next-mq2-xts-4500` | `qwen3.8-flash-next-base.mq2-xts-4500` | 44.5 | 48 | MQ2 XTS split base: 4,500 hot MQ4 XTS experts, E8P 2-bit on the other 20,076, PLE rows from the shared sidecar (below) |

The short user-facing name is `qwen3.8:flash`, an alias for `qwen3.8:flash-next`.
For example: `hipfire pull qwen3.8:flash` and
`hipfire run qwen3.8:flash "hello" --stats`.

All four MQ4 tags and the `qwen3.8:flash` alias name the split base
`qwen3.8-flash-next-base.mq4-xts` in `hipfire-models/qwen3.8-flash-next`
(70,888,270,360 B, SHA-256
`eafaff94fb9c6bb51282942e0ea6441cbdc266b634bb003b5913cfe2ccb90f45`). It has no
PLE rows of its own; they come from the shared sidecar pair described below.
It is the `hipfire ple split` of the previous monolithic file: 1,197 tensors
are byte-identical to it, and only the 128 PLE n-gram shards were removed (a
`hipfire_ple_rows` marker was added). The base plus the shared q8 rows is
therefore the same model as the monolithic file; with the q4 rows the rows are
not exact. The previous monolith is `qwen3.8-flash-next.mq4-xts` (also
published as `qwen3.8-flash-next-gptq3.mq4`; 125,288,544,792 B, SHA-256
`8b15b6fede7d7c5bfed0db4720a8295bedda51bc93e545fa242bd50d0f200972`).

Existing monolithic installs are grandfathered, with no automatic local split.
The client carries a compiled-in pin of that file, under both spellings
(`qwen3.8-flash-next.mq4-xts` and `qwen3.8-flash-next-gptq3.mq4`). Under every
Flash-Next MQ4 tag, a verified local copy (SHA-256, as for any pinned file) is
served when the split base is absent. For such an install `hipfire pull <tag>`
downloads no base: a planned q8 is skipped, because the monolith embeds the q8
rows, and a planned q4 (gfx1151 `auto`, or `--ple q4`) fetches the shared q4 and
links it as `<monolithic file>.ple-q4`, which binds the monolith natively. Its
overlay and hot set come from the monolith-bound pins below, installed beside
it. `hipfire pull <tag> --force` pulls the split base instead, and
`hipfire ple split <tag>` remains the manual converter.

The four tag names are unchanged, and the registry no longer references the
monolithic files. v0.4.2.1 clients reject a registry that names the shared
sidecar pair and fall back to their cached or bundled registry, which still
points at the monolithic files. No Hugging Face files are deleted or altered
by the tag retirement below.
The historical RTN-asymmetric `qwen3.8-flash-next.mq4`
(`qwen3.8:flash-next-rtn-asym`) is a different artifact and is never migrated.
Fixture identity and recipe history: [`AGENTS.md`](../AGENTS.md#pinned-flash-next-bench-fixture).

**Known limit: host RAM on a 16 GB card.** Measured sizing finding: an RX 9070 XT (16 GB) host with 64 GB RAM (TTM cap 31 GiB) cannot load Qwen3.8 Flash-Next, whose 38–61 GB of host-mapped routed experts exceed that cap. Flash-Next on a 16 GB card needs at least 128 GB of host RAM. (Details and the `HIPFIRE_EMU=rx9070xt` capacity profile: [`EMU.md`](EMU.md#measured-sizing-finding-flash-next-on-a-16-gb-card).)

Other Flash-Next identities, retained for provenance and not the default:

| Tag | File | Size GB | Min VRAM | Notes |
|---|---|---:|---:|---|
| `qwen3.8:flash-next-rtn-asym` | `qwen3.8-flash-next.mq4` | 125.3 | 128 | **Retired**; previous RTN-asymmetric rung; existing verified local file still supported |
| `qwen3.8:flash-next-mq6q8-pleq8` | `qwen3.8-flash-next.mq6q8-pleq8` | 125.5 | 128 | **Retired**; earlier rung; existing verified local file still supported |

Only the two tags marked **Retired** are retired. New clients refuse
`hipfire pull <retired-tag>` and name `qwen3.8:flash-next` as the replacement;
this also applies when the old file is already installed or `--force` is used.
`run` and `serve` accept the old tag, bare filename or installed model path
after SHA-256 and size verification, without redownloading it. An explicit
path to a copy elsewhere remains a plain-file load, not a name-based retired
identity. A missing retired file produces a clear error, never an automatic
download. Local `hipfire list` marks installed models `[retired]` (JSON:
`"retired": true`); the new client's bundled and loaded available views
exclude them.

A compiled `RETIRED_MODELS` table preserves the repository, filename, SHA-256
and size independently of the available catalog. Neither retired artifact had
sidecar, overlay or hotset pins; none are invented. Local
verification reuses the existing `VerifiedCache` in
`<HIPFIRE_HOME>/cache/verified-artifacts.json`; unchanged verified files avoid
another full hash. Retirement does not invent overlay or hotset identities
where none existed, migrate the old bytes, or delete or alter any Hugging Face
object.

Both retired entries are removed from the raw remote registry as well as the
new client's bundled/loaded available view; no retirement wire fields are
added. Source inspection shows that 0.4.2 rejects the new generic `ple` schema
and falls back to its cache or bundle, not that it accepts the current wire
registry. This is code-derived behavior, not an exercised old-binary test.
The monolithic Hugging Face files **must stay published while 0.4.2 clients
exist**, because their bundled entries still point to them. The split-base
MQ4 aliases and the MQ2/MQ3 tiers above are unchanged. The obsolete local-only
`qwen3.8:flash-next-mq4-xts-zab` entry is removed outright with no retired pin;
the overlay pins and the `gfx1151-zab-v1` default stay as they are.

**Split bases and the shared PLE pair.** The Flash-Next MQ4 XTS base
(`qwen3.8-flash-next-base.mq4-xts`, 70.9 GB, the four MQ4 tags above) and the
MQ3 XTS base (`qwen3.8-flash-next-base.mq3-xts`, 59.1 GB,
`qwen3.8:flash-next-mq3-xts`) are split bases with no PLE rows of their own;
the MQ2 XTS bases follow below. The MQ3 base has 16,839 of the 24,576 trunk
routed experts as 3-bit GPTQ (W3), its 108 linear-attention input projections
use 4-bit QT44 encoding instead of 6-bit MQ6, and its other 7,737 experts and
every other tensor are the MQ4 XTS bytes. Sidecars use
the existing `hipfire pull <tag> --ple q4|q8|auto` pick: a plain pull follows
`memory.ple_format` (`auto` = q4 on gfx1151, q8 elsewhere; on a host with
several GPU archs, such as a gfx1100 next to a gfx1151, `--ple auto` fetches
the union of what each card defaults to, here q4 and q8), `--ple q4` pulls
the q4 rows explicitly, and `--ple q8` opts in to the exact Q8 rows, which
`memory.ple_format = q8` then requires. All split bases share one published
pair of row files in `hipfire-models/qwen3.8-flash-next`, `qwen3.8-flash-next.ple-q4`
(29,120,146,432 B, 29.12 GB, 4-bit Lloyd-Max rows in rank order) and
`qwen3.8-flash-next.ple-q8` (54,400,269,312 B, 54.4 GB, exact Q8F16 rows), bound
by the base marker's row geometry and the q8 source-rows SHA-256 (`18e2fef1…`,
not derived). The shared q8 rows are byte-identical to the monolithic MQ4 XTS
file's embedded rows, which binds natively and needs no sidecar.

| Tier | Base | Download with q4 (gfx1151 default) | Download with q8 |
|---|---:|---:|---:|
| MQ4 XTS | 70.9 GB | 100.0 GB (100,008,416,792 B) | 125.3 GB (125,288,539,672 B) |
| MQ3 XTS | 59.1 GB | about 88.23 GB | about 113.51 GB |
| MQ2 XTS | 46.9 GB | about 76.0 GB | about 101.3 GB |
| MQ2 XTS 4500 | 44.5 GB | about 73.7 GB | about 98.9 GB |

The old monolithic MQ4 XTS file was 125.3 GB with the q8 rows embedded. The q4
rows save about 25.28 GB of disk and download, not VRAM (the PLE rows are
external). They are not exact: a fresh pull on gfx1151 previously ran the
embedded Q8 rows and now runs q4 rows (WT2 KLD 0.104865 with q4 rows against
0.103252 with q8 rows, a v0.4.2 measurement); `hipfire pull qwen3.8:flash-next
--ple q8` keeps exact rows. Loader gates on hipfire a6f241ce2's split-base
loader (MQ3 tier): greedy AR token ids equal the `qwen4_kld` reference on 8
prompts, MTP output equals AR, and the q4 and q8 sidecars both load.
Row-table semantics: [`CONFIG.md`](CONFIG.md#qwen4-ple-row-tables).

**MQ3 XTS on Strix Halo (gfx1151), native.** All 59.1 GB stay resident; W3
experts run the slab route (K1 decode GEMV with F32 activations, K2 IU4
prefill with A4 activations). Measured on hipfire ff859c360 with the q4 rows
(row order) against the published BF16 teacher refs (WT2
`bf16src-wt2-c512x32`, code24 `fn-bf16-code24-24x512.v3`). The matched
reference is the MQ4 XTS monolith with its `.trunk-overlay`, which gives it
the same QT44 linear-attention trunk as the MQ3 base, so the difference is
the W3 experts alone. Prefill (all-row) KLD, MQ4 → W3: WT2 0.1073 → 0.1369
(+0.030, 95% chunk bootstrap [+0.021, +0.038]); code24 0.1075 → 0.1393 (+0.032
[+0.021, +0.043]). Decode (`--decode`, 8 chunks), MQ4 → W3: WT2 0.0750 →
0.1150; code24 0.0754 → 0.1358. The overlay alone moves the plain monolith's
code24 prefill KLD by +0.0072 (95% t [+0.0001, +0.0143]). Tool calls (12
cases) and reasoning answers (18 problems, up to 6.3K reasoning tokens) passed
with every one of seeds 1–3 on both models; reasoning length shows no
detectable difference on the long set (10 problems × 3 seeds; mean W3 844 vs
MQ4 931 tokens, W3/MQ4 ratio 0.91, 95% CI 0.75–1.17). Decode throughput:
greedy W3 34.3 vs MQ4 33.8 tok/s, but sampled (T=1, top-k 20, top-p 0.95) W3
30.0 vs MQ4 32.1 tok/s. The K1 slab GEMV matches its CPU twin byte for byte on
real layers 0, 31 and 47 and over every wide R=1–8 slab cell, with C1 on and off.

**MQ2 XTS tier (split bases).** `qwen3.8:flash-next-mq2-xts`
(`qwen3.8-flash-next-base.mq2-xts`, 46.9 GB) keeps 6,306 hot routed experts as
MQ4 XTS bytes; the other 18,270 trunk routed experts are E8P 2-bit (an
E8-lattice codebook with one FP16 scale per 128 weights), solved on a
chat-mix calibration set. `qwen3.8:flash-next-mq2-xts-4500`
(`qwen3.8-flash-next-base.mq2-xts-4500`, 44.5 GB) keeps 4,500 hot experts and
runs E8P on the other 20,076. In both, the 108 linear-attention input
projections use 4-bit QT44 encoding instead of 6-bit MQ6. Both carry no PLE
rows and use the same shared
`qwen3.8-flash-next.ple-q4` / `.ple-q8` pair and the same `--ple q4|q8|auto`
behaviour as the MQ3 XTS tier above. The download is about 76.0 GB with the q4
rows and 101.3 GB with the q8 rows for `-mq2-xts`, and about 73.7 GB and
98.9 GB for `-mq2-xts-4500`. Supported on the 7900 XTX (gfx1100) and R9700
(gfx1201) only; Strix Halo (gfx1151) refuses these tags cleanly in 0.4.3.

**Grandfathered monolithics.**
The client carries a compiled-in pin for the Flash-Next MQ4 XTS split base
(`qwen3.8-flash-next-base.mq4-xts`, sha256 `eafaff94…`, 70,888,270,360 B), a
byte-exact `hipfire ple split` of the monolithic MQ4 XTS file (sha256
`8b15b6fe…`, 125,288,544,792 B): when a registry entry carries that split
identity, a verified local monolithic (canonical or `-gptq3.mq4` name) keeps
serving its tags, `hipfire pull` downloads no base for it, and the exact q8 rows
it embeds satisfy `q8`. The four published MQ4 tags carry that split identity,
so a stock install with the monolithic keeps serving it. A
resolved monolithic is hashed once: the result is cached by path, device,
inode, size and mtime in `<HIPFIRE_HOME>/cache/verified-artifacts.json`, so
later loads skip the 125 GB read until the file changes.

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

Draft **loading** is registry-driven: `hipfire pull <tag>` fetches the draft sidecar alongside the target, and `dflash_mode` / `speculation` ([`CONFIG.md`](CONFIG.md), [`env-vars.md`](env-vars.md)) decides what happens next — the schema default `dflash_mode` is **`off`** (pull ≠ enable) with one registry-scoped exception: the Qwen3.8-27B tiers that declare a paired draft (`qwen3.8:27b` and the `mq3*` / `mq4*` tiers) get `auto` from the registry layer, so the draft is used when its file is present and the load keeps MTP `auto` when it is absent (which is AR for this Qwen35 producer, since it declares no measured AR floor; `--spec mtp` opts in; an explicit `dflash_mode=off` wins; [`CONFIG.md`](CONFIG.md#qwen38-27b-paired-dflash-default)). Elsewhere `auto` uses the sidecar when present (AR otherwise); `on` requires it and fails the load without it. Override the sidecar with `developer.dflash_draft` / `HIPFIRE_DFLASH_DRAFT` or `run --model-draft`. Watch for `DFlash draft loaded:` in load output. Several tags may share one `dflash.file`; `hipfire rm` keeps that file while another installed declarer still needs it ([`CLI.md`](CLI.md)).

Vision-tower **loading** is registry-driven the same way: every `qwen3.8:27b*` tier declares the shared `vision.file` (`qwen3.8-27b-vision.hfq`, llama.cpp mmproj-style), so `hipfire pull <tag>` fetches the tower once and every text quant tier serves images without requantizing the trunk. The loader opens the sidecar as a separate pack and applies it with the trunk's vision config. `vision_mode` ([`CONFIG.md`](CONFIG.md#vision-tower), [`env-vars.md`](env-vars.md)) decides what happens next — **`off`** never loads the tower (even an explicit `--vision` is skipped, the daemon's `dflash_mode=off`-style hard override, so those loads pay no tower VRAM); **`auto`** (the default) uses the registry/sibling sidecar when present and stays silently text-only when absent — the Qwen3.5/3.8-VL tower is loaded on demand from host/disk and streamed per block through two small VRAM staging buffers during each image encode, so a text load pays no resident tower VRAM; **`resident`** uploads the whole tower into VRAM at load (~0.94 GiB on the 27B) for the lowest first-image latency, with the same sidecar-missing behaviour as `auto`; `on` streams like `auto` but requires the declared sidecar and fails the load closed without it. A trunk with an embedded tower declares no sidecar and is unaffected by this key. Override per load with `run --vision` / `serve --vision` or `HIPFIRE_VISION_SIDECAR` (empty opts out); a `<trunk-stem>-vision.hfq` file beside the trunk is also discovered. `hipfire rm` keeps the shared file while another installed declarer still needs it ([`CLI.md`](CLI.md)).

MTP-head **loading** is registry-driven too. Every Qwen3.8-27B trunk tier above (`mq3*`, `mq4*`, `mq5*`, `mq6*`) declares the shared `mtp.file` `qwen3.8-27b.mtp` (225,716,224 B, sha256 `f0d46d07ded75abc095ebfdbccc167c68418a6515b187bf810395faa460bd32e`), so `hipfire pull <tag>` fetches it once for all tiers. It is the native `mtp.*` layer of the public `Qwen/Qwen3.8-27B` checkpoint, packed by `hipfire-quantize`'s `mtp_extract --quant mq4` (arch 21). It carries no trunk weights: the head reuses the trunk's embeddings and `lm_head`, which is why one file serves every tier. The DFlash drafts declare no head.
- `speculation` and `mtp_mode` ([`CONFIG.md`](CONFIG.md#speculative-decode)) both default to `auto`. For the Qwen3.8-27B head (a Qwen35 producer) `auto` does **not** arm it: native MTP `auto` arms only a producer that declares a measured AR floor (today only Flash-Next / Qwen4), so the load keeps AR, loads no head and logs `MTP: off by default (no measured AR floor); opt in with --spec mtp`. `--spec mtp` / `speculation.mtp=on` is the explicit opt-in: the load then logs `MTP head loaded (sidecar …)` and each request logs `drafter=mtp`. `off` never loads the head, and `on` fails the load without it.
- Lookup order: `mtp.file` in the models directory, then `<file>.mtp` beside the model path as typed, then beside the canonical (symlink-resolved) trunk. A head placed beside a symlinked trunk is therefore found.
- The loader refuses a head whose hidden size or vocab differs from the trunk's (for example a 35B-A3B head beside a 27B trunk). `auto` falls back to AR with one log line, and `on` fails the load with the two sets of dimensions.
- On gfx1201 with H2 (`qwen3.8:27b-mq4-xts`), MTP decode against AR measured 1.09–1.65× on six of eight task genres, 0.96× on a tool answer and 0.95× on code editing. Greedy MTP output matches its own verifier's argmax, but it can differ from greedy AR output.
- `hipfire rm` keeps the shared head while another installed tier still declares it ([`CLI.md`](CLI.md)).

Flash-Next trunk-overlay **loading** is pinned in the client, not declared in the registry. The four stock Flash-Next MQ4 XTS tags (`qwen3.8:flash-next`, `-mq4`, `-gptq3`, `-mq4-xts`) resolve to the split base (sha256 `eafaff94…`, 70888270360 B), and the client carries a compiled-in pin for it naming the re-bound overlay `qwen3.8-flash-next-base.mq4-xts.trunk-overlay` (305543168 B, sha256 `e3e352ca…`, always the base file name plus `.trunk-overlay`) in `hipfire-models/qwen3.8-flash-next`. Its payload is identical to the pre-split overlay; only the base binding changed. `hipfire pull <tag>` fetches it next to the base on a host with a gfx1151 GPU or no detected GPU (the loader applies it only on gfx1151; gfx1100 and gfx1201 hosts skip the download, and `HIPFIRE_QWEN4_TRUNK_OVERLAY=1` keeps it). The client also keeps the pre-split pin, bound to the monolithic file (sha256 `8b15b6fe…`, 125288544792 B): `qwen3.8-flash-next.mq4-xts.trunk-overlay` (sha256 `df49bc0a…`), installed beside a grandfathered monolithic install instead. There is no registry schema change: older clients keep parsing the published `registry/v1.json`. It is the zab tier as an overlay: an HFQM container with the 108 GDN `in_proj_{z,a,b}` projections encoded as symmetric QT44, bound to the base by file size and index SHA-256, that the Qwen4 loader substitutes for the base's MQ6 tensors and then arms the `gfx1151-zab-v1` A4 trunk route. The bytes are the ones the developer map `HIPFIRE_QWEN4_REQUANT="...=qt44:<dir>"` applies.
- `HIPFIRE_QWEN4_TRUNK_OVERLAY` unset or `auto` applies the overlay only on gfx1151 (the exact architecture of the `gfx1151-zab-v1` trunk default); `0` never applies it (kill switch, the stock base weights load); `1` applies it on any architecture (the A4 route itself still arms by default only on gfx1151; elsewhere `HIPFIRE_QWEN4_TRUNK_IU4` must select it); any other value is a load error.
- `hipfire rm` removes the overlay with its base. The obsolete local-only `qwen3.8:flash-next-mq4-xts-zab` baked tier is removed from the registry (no retired pin); the overlay on the stock tags supersedes it.

Flash-Next expert-hotset **fetching** is likewise pinned in the client, not declared in the registry. The same four stock tags resolve through the split base identity (sha256 `eafaff94…`, 70888270360 B) to the re-bound `qwen3.8-flash-next-base.mq4-xts.expert-hotset` (217675 B, ~213 KiB, sha256 `91143074…`, always the base file name plus `.expert-hotset`) in `hipfire-models/qwen3.8-flash-next`. The pre-split pin, bound to the monolithic file (sha256 `8b15b6fe…`, 125288544792 B; `qwen3.8-flash-next.mq4-xts.expert-hotset`, 217676 B, sha256 `7f6760e1…`), is kept for grandfathered monolithic installs. `hipfire pull <tag>` selects it by default; the interactive `hipfire pull` prompt offers it as an independent default-on yes/no **Expert hot set (R9700 decode, 213 KB)** choice (the "213 KB" is the rounded display label; the file is about 217.7 KB), shown in the summary only when selected; `--yes` and the TUI's Models-tab pull keep the default. It is verified by SHA-256 and stored at the sidecar path of the local base file, which may be a renamed local file that matches a pinned base identity. `registry/v1.json` is unchanged. At runtime it applies by default on a single R9700 (gfx1201) when present and bound to the base by size and index SHA-256. `hipfire rm` removes it with its base, and `*.expert-hotset` files are sidecars, never listed or served as models. See [`CLI.md`](CLI.md#flash-next-expert-hotset).

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
| `decider:4b` | `decider-4b.mq4` | 2.6 | 4.1 | registry `default_kv_mode` q8 (`auto` still applies); dense Qwen3.5-4B decision judge for `POST /v1/systemone` (upstream Mapika/decider-4b, Apache-2.0); symmetric MQ4G256V2, uncalibrated — no calibration profile ships with the tag |
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

### Raon-1-mini-preview (arch 19)

| Tag | File | Size GB | Min VRAM | Notes |
|---|---|---:|---:|---|
| `raon-1-mini-preview` | `raon-1-mini-preview-v2.mq4-xts` | 17.9 | 20 | KRAFTON Raon-1-mini-preview, 28.8B / 4.3B-active MoE (arch 19; RDA gated-delta + MLA + mHC, SiTU experts). MQ4 XTS v2: routed experts symmetric MQ4V2 (qt 44) with AWQ + GPTQ refit; the always-active shared-expert and dense layer-0 FFN, attention and `lm_head` Q8F16; 20 GB-card fit pending real-hardware measurement (32K+ context admitted under 7900 XT capacity emulation). Reasoning model (thinking on by default), XML `<tool_call>` tools. Registry sampling defaults: temperature 1.0, top_p 0.95, top_k 20, min_p 0, presence_penalty 1.5, repeat_penalty 1.0 (the model card's coding / math setting is presence_penalty 0; presence applies to generated tokens only). |

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

### EmbeddingGemma 2 (arch 18 — local-only rows, no Hub artifact)

Two `embeddinggemma-2:` rows are registered with an **empty `repo`**, which the
registry reads as *local-only*: `hipfire pull` refuses them with "place \<file\>
in \<models dir\>" instead of 404ing, and `hipfire embed`/`list` resolve them
from your models directory. Digests are of the bytes the command below
reproduces (a re-quantized artifact carries different ones). Nothing needs to be
published for this to work.

| Tag | File | Size GB | Min VRAM | Notes |
|---|---|---:|---:|---|
| `embeddinggemma-2:q8f16` | `embeddinggemma-2-q8f16.hfq` | 0.33 | 2 | Arch-18 default tier: worst fixture 2.0e-2 rel-L2 against the float32 reference (band 3e-2). Also reachable as the `embeddinggemma-2` alias |
| `embeddinggemma-2:mq4v2` | `embeddinggemma-2-mq4v2.hfq` | 0.25 | 2 | MQ4G256V2 trunk, worst fixture 2.8e-1 (band 5e-1) |

**Media needs the tower pack as a sibling.** Both rows declare their
`<trunk>.vl` pack in the registry's `vision` slot (963 tensors: the checkpoint's
vision + audio towers), and the loader also finds a `<trunk>.vl` sibling on its
own. `vision_mode` defaults to `auto`, so a present pack is loaded and media is
served; `hipfire config set vision_mode off` keeps the load text-only (media
refused per modality, naming the knob).

```bash
# One command, from the HF snapshot: quantize, install both files, register the tag.
hipfire quantize ~/.cache/huggingface/hub/models--google--embeddinggemma-2/snapshots/<rev> \
  --format q8f16 --install --register embeddinggemma-2:q8f16
hipfire embed embeddinggemma-2:q8f16 --kind query "what is in this picture <|image|>" \
  --image photo.png -j
```

`--format mq4v2` produces the smaller tier (`--install --register
embeddinggemma-2:mq4v2`). `--install` copies the trunk **and** the `.vl` pack it
was written beside into the models directory — the originals stay in the working
directory, so run it somewhere disposable or delete them afterwards. `--format
q4k` is refused at load: Q4K (qt=4) is a GGUF/Maple-side container outside the
dense HFQ decode set, which this encoder's host decoder does not implement.

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
| `decider` | `decider:4b` |
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

#### GGUF-native serving (developer tools, gfx1201, default off)

With `HIPFIRE_GGUF_NATIVE=1`, `eval_hipfire`, `bench_qwen35_mq4` and `gguf_native_gen` load a dense Qwen3.5 / 3.6 / 3.8 `.gguf` as it is: the ggml blocks (IQ4_XS, Q4_K, Q5_K, Q6_K, Q8_0) are uploaded unchanged, never re-quantised, and the value-head order the exporter changed is undone on the host (row permutations) or carried with the weight (the `out_proj` K axis). Decode runs the `ggml.a8q81` route (activations quantised per 32 to q8_1, integer dot, f32 scale fold); prefill dequantises each weight to F16 and uses the F16 WMMA GEMM. The load log prints the model's bits per weight (summed over every tensor), the route and `kld_class=UNMEASURED` until a KLD run against a pinned teacher exists for the file. A GGUF whose MTP block holds a type outside the admitted set loads trunk-only with the reason named. The product CLI, daemon and `hipfire serve` do not load GGUF yet, and the legacy Q4_K / Q6_K kernels used by the quantizer path are unchanged. Env switches: [env-vars.md](env-vars.md) "GGUF-native".

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
