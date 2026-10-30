# CLI reference

Audience: operators driving the native Rust `hipfire` binary
(`crates/hipfire-cli`).
Run `hipfire <cmd> --help` (or `hipfire help <cmd>`) for the live flag text. This page indexes subcommands and file locations; it does **not** duplicate the full model registry or every config key.

Bare `hipfire` (no subcommand) is equivalent to `hipfire tui`: it launches the terminal UI and **requires** the `hipfire-tui` binary (`HIPFIRE_TUI_BIN`, `~/.hipfire/bin/hipfire-tui`, the workspace `target/release`, or `PATH`). When that binary is missing the command exits with an error naming the build command; it does not print a command list (use `hipfire help`). Config keys: [CONFIG.md](CONFIG.md). Env vars: [env-vars.md](env-vars.md). Models: [MODELS.md](MODELS.md).

## Model lifecycle

| Command | Purpose |
|---|---|
| `hipfire pull <tag>` | Download a registry model into `~/.hipfire/models/` and fetch its declared DFlash (`dflash.file`), vision (`vision.file`) and MTP (`mtp.file`) sidecars. The schema `dflash_mode` default is **`off`**; Qwen3.8-27B tiers with a declared draft (`qwen3.8:27b`, `-mq3*`, `-mq4*`) carry registry-layer `auto` ([CONFIG.md](CONFIG.md#qwen38-27b-paired-dflash-default)). A present draft enables eligible DFlash; an absent file leaves native MTP governed by its separate `auto` policy, which arms native MTP only for a producer that declares a measured AR floor (today only Flash-Next / Qwen4), so a pulled Qwen35 MTP head stays unused until `--spec mtp` / `speculation.mtp=on`. `dflash_mode=on` / `speculation.dflash=on` refuses without a DFlash draft, even on an MTP-only tier; it never aliases to MTP. Use `--spec mtp` / `speculation.mtp=on` for explicit native MTP. A native MTP producer that arms under `auto` declines a resolved host-mapped routed-expert spill on either execution seam. Uniform MQ6G256 V1 (qt15) Qwen35 MoE on gfx1200/gfx1201 with FP16 WMMA enabled refuses an attaching native MTP head (an explicit `on`, since Qwen35 `auto` no longer attaches); use `--spec off` / `speculation.mtp=off` for AR or `HIPFIRE_FP16=0` for explicit MTP. Other explicit attachment is unchanged. `dflash_mode=off` beats registry `auto`. Draft path override: `developer.dflash_draft` / `HIPFIRE_DFLASH_DRAFT` / `run --model-draft`; vision override: `run --vision` / `serve --vision` / `HIPFIRE_VISION_SIDECAR`. See [MODELS.md](MODELS.md#dflash-draft-artifacts-registry). |
| `hipfire pull <tag> --force` | Replace an existing target after downloading and verifying a new copy. |
| `hipfire pull <tag>` (on a terminal) | Asks only what the registry leaves open, then confirms: (1) the **tier**, when the registry has other tags `<tag>-<suffix>` of the same `arch_id` (aliases resolve first, so `qwen3.8:flash` offers the `qwen3.8:flash-next` tiers; same-artifact tags, a split entry's monolithic sibling, drafts and local-only entries are not tiers), each with its size, the first sentence of its `desc` and whether `min_vram_gb` fits the detected VRAM or VRAM+RAM; (2) the **PLE rows** of a split entry (published q4/q8, both, or none; preselected: what `--ple auto` fetches, or q4 when host RAM cannot hold the base plus the q8 rows); (3) the registry's **draft/MTP sidecars** (MTP, DSpark, DFlash; default yes); (4) a **summary** with every file, the total download and the free disk space (default no when it does not fit). Enter takes the default. A flag skips its question: name the tier's tag, `--ple`, `--draft` / `--no-draft`. A single-variant model with nothing to ask pulls without a prompt. `-y` / `--yes`, or a stdin that is not a terminal, asks nothing and takes the registry defaults (the tag as given, `--ple auto`, every declared sidecar), so scripts and CI are unchanged. |
| `hipfire pull <tag> --no-draft` | Skip the registry's draft/MTP sidecars (MTP, DSpark, DFlash); `--draft` fetches them without asking (the default). Other declared sidecars are always fetched. |
| `hipfire pull <tag> --ple auto\|q4\|q8\|all\|none` | For a split Flash-Next entry, choose which PLE row sidecar is downloaded next to the base (default `auto`: `memory.ple_format`, else q4 on gfx1151 / q8 elsewhere, the union over every detected GPU arch on a mixed host such as gfx1100 + gfx1151). See [Flash-Next split PLE layout](#flash-next-split-ple-layout). |
| `hipfire pull <flash-next tag>` (trunk overlay) | The four stock Flash-Next MQ4 XTS tags also fetch the `qwen3.8-flash-next.mq4-xts.trunk-overlay` sidecar (~305.5 MB) next to the base, from a pin compiled into the client, only on a host with a gfx1151 GPU (or no detected GPU); a gfx1100/gfx1201 host skips it. See [Flash-Next trunk overlay](#flash-next-trunk-overlay). |
| `hipfire pull <flash-next tag>` (expert hotset) | The four stock Flash-Next MQ4 XTS tags also fetch the `qwen3.8-flash-next.mq4-xts.expert-hotset` sidecar (217676 B, ~213 KiB) next to the base, selected by default (the interactive pull prompt asks it as the default-on **Expert hot set (R9700 decode, 213 KB)** yes/no choice; `--yes` and the TUI's Models-tab pull keep it), from a pin compiled into the client. See [Flash-Next expert hotset](#flash-next-expert-hotset). |
| `hipfire ple derive [--format q4] [--rank PATH] <model>` / `hipfire ple split [--out PATH] [--keep-original] <model>` | Derive `<model>.ple-q4` from the Q8 rows, or split a monolithic model into a PLE-less base + `<base>.ple-q8`. `<model>` is a path or registry tag. |
| `hipfire ple rebind --old-base PATH --new-base PATH --source PATH` | Copy an existing q4 PLE sidecar to `<new-base>.ple-q4` with a new base binding, only after proving both monolithic bases have identical embedded PLE descriptors and Q8 bytes. The source sidecar is never modified. |
| `hipfire list [-r\|--remote] [-j\|--json]` | Local models; `-r` also lists pullable registry tags; user aliases from `quantize --register` appear separately. |
| `hipfire rm <tag\|path> [-y\|--yes]` | Delete the weight file and sibling sidecars (`.triattn*.bin`, `*.mtp`, matching DSpark). A declared DFlash draft is **shared**: several tags can name the same `dflash.file` (e.g. `qwen3.8:27b` / `qwen3.8:27b-mq4-pro` / `qwen3.8:27b-mq4-xt` → one `qwen38-27b-dflash-mq4.hfq`). If any *other* registry entry that declares the same draft still has its own target file on disk, `rm` **keeps** the sidecar and prints one stderr line; otherwise the draft is removed with the target. The `vision` tower sidecar and the `mtp` head follow the same shared-keeper rule (every `qwen3.8:27b*` tier declares the one `qwen3.8-27b-vision.hfq`, and every trunk tier the one `qwen3.8-27b.mtp`). Confirms unless `-y`. |
| `hipfire ps [-j\|--json]` | Running daemon / quantize / upload processes and whether the configured serve port is busy (process scan is Linux-oriented). |
| `hipfire remove <tag\|path>` | Alias of `rm`. |
| `hipfire registry status\|list\|update [-j\|--json]`, `hipfire registry show <tag> [-j]`, `hipfire registry verify [PATH] [-j]` | Inspect the registry migration surface. `status` prints the source in use (bundled, cache or network), schema version, revision, cache path and warnings; `list` prints the registered tags; `show` prints one canonical entry; `update` loads the registry and prints the same report as `status` (the cache is refetched only when older than 24 h); `verify` validates the bundled registry or an explicit v1 JSON file. |

Tags resolve through the dynamic registry + aliases. Authoritative live list:
`hipfire list -r`. The native binary embeds `registry/v1.json`; the curated
editing surface is `registry/models.json`. At startup it may refresh the v1
payload into `~/.hipfire/registry.cache.json` (24h). Offline or invalid remote
payloads fall back to cache then the embedded registry. Pin the bundle with
`HIPFIRE_NO_REGISTRY_FETCH=1`.

Flash-Next MQ4 XTS tags (short name `qwen3.8:flash`, alias for
`qwen3.8:flash-next`) automatically migrate the exact SHA-verified installed
`qwen3.8-flash-next-gptq3.mq4` to `qwen3.8-flash-next.mq4-xts` by a no-clobber
hardlink, preserving the old name. External-storage symlinks are preserved;
when linking is unsupported or read-only, the verified old path is used.
This does not download or copy the payload. Conflicting canonical files fail
clearly; the historical RTN `.mq4` artifact is never migrated. See
[MODELS.md](MODELS.md#qwen-38-flash-next-mq4-xts) for the compatibility window.

### Retired Flash-Next tags

`qwen3.8:flash-next-rtn-asym` (`qwen3.8-flash-next.mq4`) and
`qwen3.8:flash-next-mq6q8-pleq8` (`qwen3.8-flash-next.mq6q8-pleq8`) are
retired. `hipfire pull` refuses either tag or bare filename before auth/network,
including an already installed copy and `--force`, with this one-line error:
`<tag> is retired; the file still runs if you have it; replacement qwen3.8:flash-next`.

`hipfire run` and `hipfire serve` accept the old tag, bare filename or installed
model path after SHA-256 and size verification, without redownloading it. A
copy elsewhere loaded by explicit path remains a plain-file load, not a
name-based retired identity. A missing retired file errors instead of pulling
(serve: HTTP 404). `hipfire list` marks installed models `[retired]` (JSON:
`"retired": true`); the available catalog (including `list --remote`) omits them.

Compiled `RETIRED_MODELS` pins preserve the original repo/file/SHA-256/size;
neither artifact had sidecar, overlay or hotset pins. Verification reuses
`<HIPFIRE_HOME>/cache/verified-artifacts.json` (`VerifiedCache`). This does not
create new overlay/hotset identities or delete or alter any Hugging Face file.
Both entries are removed from the raw remote registry and new bundled/loaded
available views, without adding retirement wire fields. Code inspection shows
that 0.4.2 rejects the generic `ple` schema and falls back to cache or bundle;
this is not an exercised old-binary test. Monolithic Hugging Face files must
stay published while 0.4.2 clients exist: their bundles still reference them.
The obsolete local-only `qwen3.8:flash-next-mq4-xts-zab` entry is removed
outright from the registry with no retired pin; its trunk tier ships as the
default overlay on the stock tags. The split `flash-next` / `-mq4` / `-gptq3` /
`-mq4-xts` aliases, MQ2/MQ3 tiers and overlay pins are unchanged.


### Flash-Next split PLE layout

A split registry entry (`file` = `<name>-base.<ext>`, e.g.
`qwen3.8-flash-next-base.mq4-xts`) ships the model without its 128 PLE n-gram
shards plus swappable row files next to it: `<base>.ple-q8` (exact Q8F16 rows,
~54.4 GB) and `<base>.ple-q4` (4-bit Lloyd-Max rows, ~28.8 GB, bit-identical in
decode to the q4 `hipfire ple derive` makes locally from the q8 rows).

- **Pull.** `hipfire pull <tag> [--ple auto|q4|q8|all|none]` downloads the base
  plus the chosen sidecar(s) after printing the plan (files, sizes). `auto`
  (default) follows `memory.ple_format` (`HIPFIRE_PLE_FORMAT`), whose own `auto`
  is q4 on gfx1151 and q8 elsewhere; an unknown GPU arch (or no GPU) resolves to
  q8. On a host with several GPU archs `auto` fetches the union of every
  detected arch's default (a gfx1100 + gfx1151 host plans q4 and q8). Unknown
  registry formats (a future `q2`) are ignored with a note; an unpublished
  lossy format falls back to the published q8, an explicit `--ple q8` never
  falls back. `none` fetches the base only. All files are SHA-256 verified; a
  failed sidecar fails the pull. Without `--ple`, a pull on a terminal asks
  which rows to fetch (preselecting what `auto` fetches, both on a mixed host,
  or q4 when host RAM cannot hold the base plus the q8 rows); `--yes` or a
  non-terminal stdin takes `auto`.
- **On-demand.** `run`, `bench` and `serve` download a missing sidecar before
  loading (`ple: downloading <file> (<GB>) for memory.ple_format=<..>`): q8 when
  q8 is requested; q4 only when neither `<base>.ple-q4` nor `<base>.ple-q8` is on
  disk. With only the q8 rows present, a `q4` load is refused until you run
  `hipfire ple derive --format q4 <base>`: a load never derives. `auto` uses whichever of the two
  is present and downloads the arch's preferred format only when neither is. A
  serve *request* naming such a model downloads only when
  `serve.allow_request_pull` is on; otherwise the load is refused with the pull
  command.
- **No forced migration (grandfathered monolithic).** A user who already has the
  monolithic `qwen3.8-flash-next.mq4-xts` (or its legacy `…-gptq3.mq4` name)
  keeps serving it under the split tags: a compiled-in pin (the split base's
  identity, sha256 `eafaff94…`) lets a SHA-verified local monolithic serve the
  tag while the split base is absent, so `hipfire pull <tag>` downloads no base
  (`Already satisfied (monolithic): <path> (serves <tag> without the split base)`,
  plan line `present (already satisfied by the monolithic; no base download)`).
  The exact q8 rows are embedded in it, so q8 is never fetched for it; a planned
  q4 is linked beside it, and the monolith-bound overlay and hot set are
  fetched when missing. One note says `hipfire ple split <tag>` or
  `hipfire pull <tag> --force` gets the split layout. Verification is cached
  (`<HIPFIRE_HOME>/cache/verified-artifacts.json`: path, device, inode, size,
  mtime, sha256): an unchanged file is not hashed again on later loads.
- **`hipfire ple derive [--format q4] [--rank PATH] <model>`** writes
  `<model>.ple-q4` from the model's Q8 rows (`<model>.ple-q8` when present, else
  the embedded rows), ordered by `--rank` (default `<model>.plehot` when present,
  else row order). `--format q8` is an error: q8 rows are the model's own rows or
  the file `hipfire ple split` writes. Progress (percent, GB, MB/s) prints at
  most every ~2 s; the summary lists output, sizes, elapsed time and the
  `payload_sha256` / `q8_rows_sha256` digests.
- **`hipfire ple split [--out PATH] [--keep-original] <model>`** converts a
  monolithic model into `<base>` (default: `-base` inserted before the
  extension) + `<base>.ple-q8`, verifies both outputs, then removes the original
  (and its hardlinked legacy spelling) unless `--keep-original`. It refuses to
  overwrite existing outputs and a non-monolithic input. `<model>.plehot*` files
  belong to the monolithic layout and are left in place with a note. The summary
  prints `base_index_sha256` and `q8_rows_sha256`.
- **`hipfire ple rebind --old-base PATH --new-base PATH --source PATH`** is for a repacked monolithic base whose non-PLE weights changed but whose embedded PLE rows did not. It validates the source q4 sidecar against the old base, compares both bases' PLE descriptors and Q8 payloads, verifies every q4 segment digest, and copies the unchanged q4 payload with only the new base-size/index binding. After readback and attachment validation it atomically replaces `<new-base>.ple-q4` with a separate inode, including when that destination was a hardlink to the source. The source file is never rewritten. Changed PLE bytes/layouts or corrupt digests are refused; split bases are not supported. A stock sidecar must not simply be hardlinked to a smaller requantized base: its binding will correctly fail the loader.
- **`rm` and listings.** `rm` also removes `<model>.ple-q4`, `<model>.ple-q8`,
  their `.plehotrows` packs and `<model>.plehot`. `*.ple-q4`, `*.ple-q8`,
  `*.plehot` and `*.plehotrows` files are sidecars: never listed or served as
  models.

### Flash-Next trunk overlay

The four stock Flash-Next MQ4 XTS tags (`qwen3.8:flash-next`, `-mq4`, `-gptq3`,
`-mq4-xts`) share one base artifact (sha256 `8b15b6fe…`, 125288544792 B). The
client carries a compiled-in pin for that base, naming the trunk overlay
`qwen3.8-flash-next.mq4-xts.trunk-overlay` (the canonical base file name plus
`.trunk-overlay`; 305543168 B, sha256 `df49bc0a…`) in the
`hipfire-models/qwen3.8-flash-next` repo. There is no registry schema change, so
older clients keep parsing the published registry. The overlay holds the 108 GDN
`in_proj_{z,a,b}` projections as pre-encoded symmetric QT44, bound to the base by
size and index SHA-256. The Qwen4 loader substitutes them for the base's MQ6
tensors on gfx1151 by default; `HIPFIRE_QWEN4_TRUNK_OVERLAY=0` restores the base
weights and `=1` applies the overlay on any architecture.

- **Pull.** `hipfire pull <tag>` downloads and SHA-256 verifies the pinned
  overlay with the other sidecars; a valid file already on disk is not
  downloaded again, a stale one is refreshed. The loader applies the overlay
  only on gfx1151, so the pull fetches it only when a detected GPU is a gfx1151
  (a mixed host with one counts) or none is detected (the pull cannot know where
  the model will run); on a host whose GPUs are all other architectures it
  prints `Skipping Trunk overlay sidecar` and downloads nothing for it, and the
  interactive summary does not list it. `HIPFIRE_QWEN4_TRUNK_OVERLAY=1` (apply
  on any architecture) keeps the download everywhere. After adding a gfx1151
  card, run `hipfire pull <tag>` again to fetch it. As for other sidecars, a failed
  overlay download prints a warning and the pull still succeeds (the base loads
  unchanged). Any other registry entry (e.g. the RTN-asym file) has no
  overlay.
- **`rm` and listings.** `rm <tag>` removes the overlay with the base
  (`<file>.trunk-overlay`). `*.trunk-overlay` files are sidecars: never listed
  or served as models.

### Flash-Next expert hotset

The same four stock Flash-Next MQ4 XTS tags (one base artifact, sha256
`8b15b6fe…`, 125288544792 B) also have an expert-hotset sidecar. As for the
trunk overlay, the client carries the pin, not the registry: it names
`qwen3.8-flash-next.mq4-xts.expert-hotset` (the canonical base file name plus
`.expert-hotset`; 217676 B, ~213 KiB, sha256 `7f6760e1…`) in the
`hipfire-models/qwen3.8-flash-next` repo, matched by the base's exact size and
SHA-256. `registry/v1.json` is unchanged, so older clients keep parsing the
published registry.

- **Pull.** `hipfire pull <tag>` selects the hotset by default. On a terminal
  the interactive pull prompt has an independent default-on yes/no choice,
  **Expert hot set (R9700 decode, 213 KB)**, shown in the summary only when
  selected (`--yes` and the TUI's Models-tab pull take the default); the "213 KB" is
  the rounded display label, the pinned file is 217676 B (~213 KiB). The hotset
  is downloaded and SHA-256 verified with the other sidecars to the sidecar path
  of the local base file, which may be a renamed local file as long as it matches
  the pinned base identity; a valid file already on disk is not downloaded again,
  a stale one is refreshed. As for other sidecars, a failed hotset download
  prints a warning and the pull still succeeds. Any other registry entry (e.g.
  the RTN-asym file) has no hotset.
- **Runtime.** The hotset applies by default on a single R9700 (gfx1201) when
  the sidecar is present and bound to the base by size and index SHA-256.
- **`rm` and listings.** `rm <tag>` removes the hotset with the base
  (`<file>.expert-hotset`). `*.expert-hotset` files are sidecars: never listed
  or served as models.

## Inference

| Command | Purpose |
|---|---|
| `hipfire run <model> [flags] [prompt...]` | One-shot generate. Model = registry tag, alias, or path. Uses a healthy `serve` over HTTP when present; otherwise spawns a one-shot daemon. Forces local spawn when `HIPFIRE_LOCAL=1` or a load-time override such as `--kv-mode`/`--image` cannot safely reuse the resident model. Recognized missing registry tags auto-pull. |
| `hipfire img <model> [flags] <prompt...>` | One-shot txt2img on a diffusion checkpoint (arch 40 FLUX.1, arch 45 FLUX.2 Klein). Model = an HFQ trunk pack (`<base>-transformer.hfq`, sidecars next to it) or a registry tag that resolves to one (`flux.schnell:1`); build packs with `hipfire-quantize --flux-pipe` below. Writes `hipfire-<seed>.png` in the working directory (or the `-o` path) and prints the path. |
| `hipfire embed <model> [flags] [text...]` | Encode text and/or images with an embedding checkpoint (arch 17, the Qwen3.5 multi-vector encoder — e.g. `pplx-embed-v2-late-0.6b`; or arch 18, EmbeddingGemma-2, one pooled vector). Prints tokens×dims; `--query`/`--doc` score the pair by late-interaction MaxSim; `--image`/`--audio`/`--video` add media (arch 18: `--video` takes one clip per flag, as its decoded frames); `--json` emits the token-embedding matrix. See [`hipfire embed` flags](#hipfire-embed-flags). |
| `hipfire chat [model] [-t <f>] [--top-p <f>] [-n <n>] [--system <text>] [--no-color]` | Interactive multi-turn line client over the HTTP service (not the TUI; for the TUI use `hipfire tui`). The model defaults to `serve.default_model`. `-t` is `0..=2`, `--top-p` is `(0,1]`, `-n` is `1..=393216`. See [CHAT.md](CHAT.md). |
| `hipfire serve [model] [host] [port] [flags]` | OpenAI-compatible HTTP server. See [SERVE.md](SERVE.md). |
| `hipfire restart [serve flags...]` | `stop --force` semantics then start with the same flags. |
| `hipfire stop [port] [--force] [--all]` | Stop the tracked background serve (`~/.hipfire/serve.pid`). `--force` reaps orphan daemons and frees the port; `--all` also reaps orphan quantize jobs. |
| `hipfire tui [flags...]` | Launch `hipfire-tui` (Home / Dashboard / Chat / Models / Settings / System / Logs). Trailing arguments are forwarded to `hipfire-tui` (e.g. `--check`, `--settings`). `?` shows each tab's keys; Esc only closes or cancels, `q` quits (it asks first while a Models-tab pull is running). The Models tab pulls with `hipfire pull <tag> --yes` (never the interactive prompt): `p` opens a small chooser for PLE rows (auto/q4/q8 → `--ple`) and draft sidecars (→ `--no-draft`) when the entry has them, and `c` cancels a running pull. |

### `hipfire run` flags
Flags may appear before or after the model. CLI help and the native typed schema list stored globals; **effective `run` sampling** is not “global default unless overridden.” The Rust request resolver transmits only explicit CLI flags, per-model overlay values, or registry `recommended_settings`. Otherwise the field is **omitted** so daemon/HFQ/arch fallback applies. Global-only `temperature` / `top_p` / `repeat_penalty` are not sent. **Chat** is the exception (global session snapshot — [CHAT.md](CHAT.md)).

| Flag | Purpose |
|---|---|
| `-t, --temp <float>` | Temperature when set on the CLI (`0` = greedy). Stored global default is `0.3` but is not auto-sent. |
| `--top-p <float>` | Nucleus sampling when set. Stored global default `0.8` is not auto-sent. |
| `--repeat-penalty <float>` | Repeat penalty when set. Stored global default `1.05` is not auto-sent. |
| `-n, --max-tokens <int>` | Generation cap (config default `4096`). |
| `--kv-mode <m>` | Whole-cache KV preset for this load: `auto`, `q8`, `fp8`, `bf16`, `fwht4`/`3`/`2`; legacy spellings `asym4`/`3`/`2`, `turbo`… (see [KV load flags](#kv-load-flags-run--serve--bench)). |
| `--kv-backend <legacy\|vmm>` | KV allocation backend. Omitted = automatic (prefer VMM). `contiguous` is **rejected** (renamed to `legacy`). |
| `--kv-k <name>` | Qwen-family K-axis override (`q8`, `fwhtN`; deprecated since 0.4.0, removal in 0.5.0: legacy `asymN`/`turboN` → `fwhtN`; legacy Givens `legacy-asymN`). |
| `--kv-v <name>` | Qwen-family V-axis override (`q8`, `lloyd2`/`3`/`4`). |
| `--head <name>` | Select a published `lm_head` variant from the registry's `heads` (e.g. `--head q4k`). The overlay shadows the model's own head at load time; omitted = the head baked into the model file. |
| `--tp <N>` | Tensor/expert-parallel degree for this load, `1..=64` (same admission as `serve --tp` / `bench --tp`). Only values above 1 are forwarded and force a local daemon. |
| `--max-seq <n>` | KV context for this load, within the `memory.max_seq` bounds (512–1048576); sent as the load's `max_seq` and replaces any configured `memory.max_seq`, as `bench --max-seq` does. Unset: eligible growing Qwen VMM KV defaults to min(model trained context, measured card capacity); legacy and other owners retain their bounds. Forces a local daemon (see below). |
| `--spec <m>` / `--speculation <m>` | Spec mechanism: `off` \| `auto` \| `ngram` \| `dflash` \| `mtp` \| `dspark` (config default `auto`). |
| `-md, --model-draft <path>` | DFlash draft path; implies `--spec dflash` when `--spec` is not given. |
| `--draft-max <N>` (alias `--draft`) | Draft window, `1..=32`. With `--spec ngram` it sets the n-gram window, with `--spec mtp` the MTP window, otherwise both. |
| `--dspark-conf-threshold <f>` | DSpark confidence cutoff in `[0,1]` (qwen3 + deepseek4). |
| `--system <text>` | System prompt. |
| `--image <path>` | Vision input (when the model supports it). |
| `--vision <path>` | Vision-tower sidecar for this load; wins over the registry `vision` slot and `HIPFIRE_VISION_SIDECAR`. Skipped while `vision_mode=off`; required when `on`; used under the default `auto` and under `resident`. Also accepted by `serve` (where `serve -d` does not forward it to the detached child). |
| `-j, --json` | Machine-readable output. |
| `--no-stream` | Buffer full response. |
| `--stats` | Emit one statistics footer on stderr after successful generation (local or HTTP, streaming or `--no-stream`); generated stdout is unchanged. Suppressed with `--json` and on failed/aborted generation. |

Resolution ladder for speculation: **CLI flag > env > per-model > global > registry card > built-in default**. On `run`, `--spec` replaces the selector the config and environment layers resolved (`hipfire config explain speculation.mode` prints the winning source).
`dflash_mode` itself defaults to **`off`** in the schema — see [CONFIG.md](CONFIG.md#speculative-decode) — except that Qwen3.8-27B tiers with a registry-declared draft resolve `auto` through the registry layer ([CONFIG.md](CONFIG.md#qwen38-27b-paired-dflash-default)); an explicit `dflash_mode=off` (per-model, global or env) wins. Pulling a draft for any other model does not enable DFlash.

`--stats` illustrative output (values are placeholders, not a benchmark): `[stats] tokens=256 decode=123.30 tok/s ttft=42.1 ms tau=7.15`.
`tokens` counts committed daemon completion IDs, including reasoning, not
printed characters. `decode` is decode throughput, not wall-inclusive `tok_s`;
`ttft` is time to first token. `tau` is included only for speculation when
available. Missing, invalid or nonfinite metrics show `n/a`, never a substituted
wall rate or zero. A continuous-batch response may legitimately have `decode=n/a`.

```bash
hipfire run qwen3.5:9b "What's 2+2?"
hipfire run qwen3.5:9b "What's 2+2?" --stats
hipfire run qwen3.5:9b --spec ngram "Repeat verbatim: ..."
hipfire run qwen3.5:27b -md ~/.hipfire/models/qwen35-27b-dflash-mq4.hfq "..."
HIPFIRE_LOCAL=1 hipfire run qwen3.5:4b "..."   # skip HTTP; always local spawn
```

While a long prompt prefills, `hipfire run` draws one self-overwriting line on stderr, only when stderr is a terminal and never with `--json`: `  prefill 512/2048 tokens (25%), ETA 6s`. It redraws at most every 500 ms, the ETA extrapolates the prefill rate since the request was sent, and the line is erased before the first token is written. Through a healthy `serve` the line follows the stream's `: prefill <done>/<total>` comments (sent only once the stream has committed, see [SERVE.md](SERVE.md)), so it can appear later than with a local daemon.

Local-forcing (skip a healthy serve): `HIPFIRE_LOCAL` truthy, `--image`, `--kv-mode`, `--kv-backend`, `--kv-k`, `--kv-v`, `--max-seq`, `--head`, `--tp` above 1, `--spec`/`--speculation`, `--model-draft`, `--vision`, `--draft-max`, or `--dspark-conf-threshold` (exact list: `run_should_force_local` in `crates/hipfire-cli/src/main.rs`). JSON and non-streaming responses are supported by the native HTTP service and do not by themselves force a local daemon.

### Load summary

Every model load prints exactly one INFO line on the daemon's stderr (which `hipfire run` and `hipfire serve` inherit, so it appears once in the terminal or serve log; `hipfire bench` is unchanged). It states what is on or off and why. Observed on one R9700 (Qwen3.8-27B, a config with `speculation.mtp`/`dflash`/`ngram` = `off` and `memory.max_seq = 262144`):

```text
$ hipfire run qwen3.8:27b "Hi"
hipfire load summary: qwen3.8-27b.mq4r (qwen3_5) | spec: off (DFlash off: speculation.dflash=off; MTP off: speculation.mtp=off; ngram off: speculation.ngram=off) | kv: fp8 (vmm backend, automatic) | context: 262144 (user-set; model 262144, card 504465; measured fit unchanged) | placement: fully resident | prefill: per-prompt chunking
$ hipfire run qwen3.8:27b --spec dflash --max-seq 16384 "Hi"
hipfire load summary: qwen3.8-27b.mq4r (qwen3_5) | spec: dflash (draft qwen38-27b-dflash-mq4.hfq, block 16) | kv: fp8 (vmm backend, automatic) | context: 16384 (user-set; model 262144, card 447002; measured fit unchanged) | placement: fully resident | prefill: per-prompt chunking
```

A Flash-Next load whose experts spill to host memory reads, for example, `spec: off (MTP off: <N> expert tensors host-mapped on gfx1100 …)` and `placement: experts: layers 0..<k> in VRAM, <N> tensors (<X> GiB) host-mapped`.

| Facet | Meaning |
|---|---|
| `spec` | The speculation route that armed (`mtp K=…`, `dflash (draft <file>, block N)`, `dspark (block N)`, `ngram K=… min=…`, `qwen4 native mtp K=…`, plus the MTP n-gram copy lane verdict) or `off (<reasons>)`. Reasons are recorded by the engine at the decision that declined each route: `speculation.dflash=off` / `speculation.mtp=off` / `speculation.ngram=off` (config or `--spec off`), no draft paired, no MTP head in the file, no measured AR floor under `speculation.mtp=auto`, host-mapped experts, `.mq4r` retained replay path, pp/ep topology, `kv_adaptive`, DSpark/DFlash load failure; with no recorded reason it reads `off (no speculator armed)`. |
| `kv` | The KV layout the carrier resolved (`fp8`, `K=q8 V=q8`, …; the requested `auto` is resolved), the backend (`vmm` / `legacy`) and whether the backend was `automatic` or `explicit`. |
| `context` | Admitted `max_seq` and what bound it (`auto: min(model N, card N)`, `user-set`), plus the measured post-load fit verdict for Qwen VMM loads (`measured fit unchanged`, `measured fit <requested> -> <fit>`). |
| `placement` | Where the weights live: `fully resident`, the host-mapped expert census (layers in VRAM, tensors and GiB host-mapped), or the pipeline/expert-parallel topology. |
| `prefill` | Load-time prefill decisions when the arch makes them (Qwen4 chunk rows, long-prefill expert staging, NPU trunk route); otherwise `per-prompt chunking`, meaning the ceiling is chosen per prompt. |

The reasons are the engine's own decision outputs, not a second policy evaluation. The same facets ride the `loaded` ack as a `load_summary` object (`line`, `spec`, `kv`, `context`, `placement`, `prefill`); `hipfire serve` publishes it on the resident model's `/v1/models` entry ([SERVE.md](SERVE.md)).

### KV load flags (`run` / `serve` / `bench`)

Shared by `hipfire run`, `hipfire serve`, and `hipfire bench` (including `--matrix`). Full config surface: [CONFIG.md](CONFIG.md#kv-cache).

| Flag | Values / default | Notes |
|---|---|---|
| `--kv-backend` | `legacy` \| `vmm`; **default automatic prefer VMM** | Accepts only those two spellings. Old `contiguous` is rejected with a migration error that names `legacy` (e.g. use `--kv-backend legacy` or `memory.kv_backend = "legacy"`). Selecting legacy (explicit or automatic fallback) prints one stderr warning containing the stable token `HIPFIRE_KV_BACKEND=legacy`. Explicit `vmm` on an unsupported combination fails closed before teardown. |
| `--kv-mode` | `auto` (default when unset), `q8`, `fp8`, `bf16`, `fwht2`/`3`/`4`; legacy spellings `asym2`/`3`/`4`, `turbo`/`turbo2`/`3`/`4`; … | Whole-cache preset. Native `fp8`/`bf16` encode K and V together, without a separate V mode. `fwhtN` is an optional headroom mode, not a default. |
| `--kv-k` / `--kv-v` | Qwen-only axis overrides; omitted when unset | Orthogonal to mode. On supported Qwen sites the legacy spellings `asymN` and `turboN` (bare `turbo` = `turbo3`) mean **`fwhtN`**; `legacy-asymN` selects the legacy Givens asym K. `asymN`, `turboN` and `legacy-asymN` are deprecated since 0.4.0 (removal in 0.5.0) and warn at load. V names: `q8`, `lloyd2`/`3`/`4`. Non-Qwen carriers refuse these axes before teardown. |
| `--max-seq` | int (512–1048576) when set; else automatic | On eligible growing Qwen VMM KV, default = **min(model trained context, measured card capacity)** after weights load; legacy and other owners retain their existing bounds. An explicit `--max-seq` replaces any configured `memory.max_seq` for that load (per process for `serve`). `bench` sizes its own load instead ([below](#bench-context-max_seq)). |

**Qwen `auto` / unset mode:** native `fp8` (both K and V) on exact `gfx1201` single GPU when the load is native-eligible — H24/Hkv4/D256 dense or H16/Hkv2/D256 A3B attention, no PP/TP/EP/slots split, no `kv_adaptive`, no CASK sidecar; `q8`/`q8` everywhere else, including gfx1100 and gfx1151. Native `bf16` stays H24 dense-only. Native modes report `kv_mode=fp8` or `bf16` in loaded/diag/bench JSON; no synthetic `V=q8` axis is reported. Non-Qwen family defaults are unchanged (e.g. Maple BF16, DeepSeek compressor F32, Gemma layered policy).

**K/V precedence** (split modes only): CLI flag **>** per-model config **>** global config **>** registry / arch default. Mode seeds the pair; `--kv-k` overrides K and `--kv-v` overrides V. An authored `--kv-mode fp8`/`bf16` refuses every axis override, including `--kv-v q8`; an auto-selected native pair can be replaced only when both axes are supplied. Full refusal rules: [CONFIG.md](CONFIG.md#kv-cache).

**Backend precedence** (forced choices): CLI `--kv-backend` **>** per-model `memory.kv_backend` **>** global `memory.kv_backend`. Omitted backend stays automatic VMM-preferring policy, not an authored pin.

### `hipfire serve` flags

| Flag / arg | Purpose |
|---|---|
| up to three positionals: `[model]` `[host]` `[port]` or `host:port` | Optional model, then bind (config defaults `127.0.0.1` and `11435`). **No TLS; authentication is optional** (`--api-key-file`) — `0.0.0.0` exposes the API on every interface; a key is recommended off loopback, and use only a trusted/firewalled network or a **TLS-terminating reverse proxy**. |
| `--api-key-file <absolute-path>` | Optional printable-ASCII bearer key file (overrides `serve.api_key_file`); see [SERVE.md](SERVE.md#security-bearer-authentication-and-https-proxies). |
| `--model <tag>` | Model tag/path to pre-warm for this process. |
| `-d`, `--detach`, `--background` | Background; log `~/.hipfire/serve.log`, pid `~/.hipfire/serve.pid`. |
| `--kv-mode <m>` | KV mode preset for this process (same contract as `run`). |
| `--kv-backend <legacy\|vmm>` | KV allocation backend (same as `run`; `contiguous` rejected → use `legacy`). |
| `--multi-slot-paged` | Multi-slot KV: one shared `PagePool` of 128-token pages with per-slot block tables, refcounted prefix sharing and COW, instead of fixed per-slot slabs. Implied by `--multi-slot-paged-pages` and by `--kv-backend vmm` precedence. Forwarded by `serve -d`. |
| `--multi-slot-paged-pages <N>` | Physical pages in the shared paged pool. `0` (default) auto-sizes from post-weight free VRAM after the prefill scratch, leaving the 512 MiB map reserve. Forwarded by `serve -d`. |
| `--kv-k <name>` / `--kv-v <name>` | Qwen-family K/V axis overrides (same as `run`). |
| `--max-seq <n>` | KV context for every ordinary model load of this service (512–1048576); replaces the configured/automatic `memory.max_seq` (same contract as `run`). With `serve.multi_slot` it sets the per-slot context (overriding `serve.multi_slot_ctx`), so total KV scales with the slot count (`serve.multi_slot_slots`). Forwarded by `serve -d`. |
| `--spec <m>` / `--speculation <m>` | Speculation mode for every model this service loads: `off` \| `auto` \| `ngram` \| `dflash` \| `mtp` \| `dspark` (other values fail at parse). Applied per load after the config selector, so it overrides `speculation.mode` and per-model speculation; `dflash` / `auto` also resolve the registry draft. `--spec off` turns MTP and DFlash off (plain AR), e.g. for an A/B. Unset: config (default `auto`). Forwarded by `serve -d`. |
| `--idle-timeout <s>` | Unload after idle seconds (`0` = never; max `86400`). |
| `--swap-host-bytes <BYTES>` | Multi-slot route: host-RAM ceiling for parked swap snapshots (`0` disables the host tier). Overrides `serve.swap_host_bytes`. Forwarded by `serve -d`; with `-d` against an already-running server the flag is not applied and a warning is printed. |
| `--swap-disk-bytes <BYTES>` | Multi-slot route: disk ceiling for parked swap snapshots (`0` disables disk spill). Overrides `serve.swap_disk_bytes`; LRU snapshots are evicted at the ceiling. Forwarded by `serve -d`; not applied (with a warning) if a server is already running. |
| `--swap-dir <DIR>` | Multi-slot route: swap spill directory. Overrides `serve.swap_dir` (default `$XDG_RUNTIME_DIR/hipfire/swap/<pid>`, falling back to `$TMPDIR/hipfire-swap-<pid>`). The directory is created `0700`, must be owned by this user, is locked to one daemon, and may not be a symlink; an existing owned directory has group/other access removed. Only stale `*.snap`/`*.snap.tmp` files are removed at startup; other contents are preserved. Forwarded by `serve -d`; not applied (with a warning) if a server is already running. |
| `--no-prewarm` | Do not load the configured default model before accepting requests (lazy-load on first request). |
| `--tp N` | Tensor/expert-parallel degree, `1..=64` (supported paths only). |
| `--continuous-batch-size N` | Number of eligible concurrent batched lanes, `1..=256` (`1` preserves sequential behavior), subject to memory admission; same range as the `serve.continuous_batch_size` config key. The runtime batch width is dynamic, and the 128/63 chunk sizes are not lane limits. Forwarded by `serve -d`. |
| `--ui` | Serve the embedded chat UI at `/ui` on the serve listener. |
| `--open` | Open the chat UI in the default browser once serving; implies `--ui`. With `-d` the browser launch happens in the launching process and the detached child only enables the route. |
| `--vision <path>` | Vision-tower sidecar wired into every model load of this process. Skipped while `vision_mode=off`. **Foreground serve only:** `serve -d` does not forward it to the detached child, so set `HIPFIRE_VISION_SIDECAR` or the registry `vision` slot for a detached service. |

### `hipfire img` flags

Run in a fresh daemon process; the prompt is the positional text (a built-in sample prompt when omitted). `--steps`
defaults to the model's own default (4 for step-distilled `flux.schnell:1`,
28 for guidance-distilled `flux.dev:1`).

| Flag | Purpose |
|---|---|
| `-o, --out <path>` | Output PNG path (default `hipfire-<seed>.png` in the working directory). |
| `--width` / `--height` | Pixel size, `1..=8192` and divisible by the VAE compression factor. Default 1024×1024; with `--image` the default is the reference image's own size. |
| `--steps <n>` | Denoise steps, `1..=128`; omit for the architecture default. |
| `--seed <n>` | Noise seed (default `0`). Same model + seed → byte-identical PNG. |
| `--backend <cpu\|gpu>` | `gpu` = the HIP transformer forward, `cpu` = the f32 reference oracle; defaults to `gpu` when available, else `cpu`. Other values are rejected. |
| `--image <path>` | Reference image for FLUX.2 Klein edit (arch 45 only; a FLUX.1 pipe refuses it). Repeatable, up to 4. Give every `--image` **before** the prompt words: the prompt is a greedy trailing positional. The CLI reads each file and sends its bytes; the daemon never opens a client-named path. |
| `-j, --json` | Print one JSON result object (`path`, `bytes`, `width`, `height`, `steps`, `seed`, `ms`, `model`) instead of the progress lines and the PNG path. |

Model resolution accepts an HFQ **component pack** (trunk file) — the
`t5`/`clip`/`vae` sidecar packs are discovered next to it by sibling name
(`<stem>-t5.hfq` etc., or shared `t5-xxl.hfq` / `clip-l.hfq` / `vae.hfq`).
`hipfire pull flux.schnell:1` fetches the trunk + all three sidecars at once.

### `hipfire embed` flags

Runs in a fresh daemon process. The production target is an embedding
checkpoint loaded as an **HF safetensors directory** (e.g. a
`pplx-embed-v2-late-0.6b` snapshot), or a raw-dtype `.hfq` built from one
with `hipfire-quantize` (the quantizer keeps the norms, the 128-d Dense head
and the vision tower raw, folds `1_Dense/model.safetensors` into the trunk,
and auto-includes the tower).

EmbeddingGemma-2 (arch 18) snapshots — e.g. a `google/embeddinggemma-2`
directory, whose quantized form writes a `.vl` tower sidecar — are served by
the same command. They answer with **one pooled vector** (`n_tokens: 1`)
instead of arch 17's token matrix, and `--image` / `--audio` / `--video` reach
their vision and audio towers (video drives the vision tower frame by frame)
when the sidecar is resident.

| Flag | Purpose |
|---|---|
| `--query <text>` | Query-side text (its marker is `"[Q] "`). |
| `--doc <text>` | Document-side text (marker `"[D] "`); with `--query`, prints the MaxSim score. |
| `--kind <query\|document>` | Side for the positional text; default `document`. Sets the marker and truncation limit. |
| `--image <path>` | Encode an image (PNG/JPEG) with the checkpoint's vision tower. Repeatable — the images are appended after the text, in flag order; text and images may be combined. An arch-18 checkpoint without its `.vl` sidecar, or a host where `vision_mode` is off, refuses the image instead of dropping it. |
| `--audio <path>` | Encode an audio clip with the checkpoint's audio tower (arch 18). Repeatable — clips are appended after the text and images, in flag order. Takes 16 kHz mono/stereo **WAV/RIFF** (PCM16 or float32); any other container is refused with the `ffmpeg` conversion to run. |
| `--video <frame>[,<frame>…]` | Encode a video with the checkpoint's vision tower (arch 18), which runs once per frame. **One clip per flag**: give that clip's decoded **frames**, comma separated, in display order — the daemon applies the video processor's own sampler (`max_frames` 32: ≤32 frames are all used, more are strided uniformly) itself, so pass a clip at whatever rate you extracted it. Repeat the flag for several clips; each clip is spliced at its own `<|video|>` marker (one marker per video, never per frame). Frames are PNG/JPEG; a container (MP4/MKV/…) is refused with the `ffmpeg` extraction to run. |
| `-j, --json` | Print the full JSON result — the `tokens×dims` matrix for a single encode, or `{score, query_tokens, doc_tokens, dim}` for a pair. |

Token limits are per side (`query_length` 1024, `document_length` 4096 on the
0.6B); text is truncated from the end with the marker always kept. The
document side drops punctuation token vectors (the ColBERT skiplist); the
query side keeps them. Scores are late interaction (`Σ max dot`), so they are
not cosine similarities and are only meaningfully comparable between the same
query across documents.

```bash
hipfire embed ~/.hipfire/models/pplx-embed-v2-late-0.6b \
  --query "what is the capital of France" --doc "Paris is the capital of France"
hipfire embed <model> "encode this document" --kind document --json
hipfire embed <model> --image chart.png "the caption" --json
```

The same surface is exposed over HTTP as `POST /v1/embeddings` (OpenAI-shaped;
the `embedding` array is the flat `n_tokens × dim` matrix and each data object
carries `n_tokens`, with an optional `"kind": "query" | "document"`). `input`
takes a string, a one-element list, the object form (`{text, image_base64}`, or
`{text, images_base64: […]}` for several — the same spelling the JSONL `embed`
verb takes), or the typed content array
(`[{"content": [{"type": "text", "text": …}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,…"}}]}]`)
— one content object per request, text parts concatenated in order, image
parts decoded at the route. A bare content-part array without the
`{"content": …}` wrapper is refused with the wrapper it should have used.
Interleave media by putting one `<|image|>` marker per image in the text
(arch 18); a text with no marker gets its image runs appended before EOS, and
a marker count that disagrees with the image count is refused. Audio takes the
OpenAI `input_audio` part
(`{"type": "input_audio", "input_audio": {"data": "data:audio/wav;base64,…", "format": "wav"}}`,
WAV/RIFF only, `format` optional) and the JSONL `audios_base64` list, with one
`<|audio|>` marker per clip under the same count rule; a text with no marker
gets its audio runs appended after the image runs. Audio needs the
checkpoint's audio tower (the same `.vl` sidecar as vision): a vision-only load
refuses a clip by name rather than encoding the text alone.

Video takes the llama.cpp-shaped `input_video` part
(`{"type": "input_video", "input_video": {"data": ["data:image/png;base64,…", …]}}`,
one entry per frame in display order, `format` optional and PNG/JPEG only) or
the JSONL `videos_base64` field — **one frame list per video**, e.g.
`{"text": "… <|video|> …", "videos_base64": [[frame0, frame1, …]]}`. One
`<|video|>` marker per video under the same count rule, and the daemon samples
the frames itself (the video processor's `max_frames` 32 budget, uniform
stride) before running the vision tower once per frame; a frame that is not a
PNG/JPEG image is refused with the `ffmpeg` extraction to run.

## Configuration

| Command | Purpose |
|---|---|
| `hipfire config` | Global interactive TUI → `~/.hipfire/config.toml` (`hipfire-tui --settings`: opens on the Settings tab). |
| `hipfire config <tag>` | Print resolved per-model policy and override provenance. |
| `hipfire config list\|get\|set\|reset ...` | Scriptable global ops (`-j/--json` on list/get; `reset` with no key removes every override). |
| `hipfire config explain [<key>] [--model <tag>] [-j]` | With a key: type, scope, default, effective value and provenance. With no key: every effective setting with its source, plus an audit of the `HIPFIRE_*` variables set in the environment (applies / bootstrap / runtime / deprecated / unknown with a did-you-mean). |
| `hipfire config schema [-j]` | Print the authoritative typed configuration schema. |
| `hipfire config migrate` | Convert legacy `config.json` to sparse `config.toml` without deleting the JSON. |
| `hipfire config profile [set <name>\|create <name>]` | `set` merges a built-in (`default`, `dev`, `hip`, `redline`) or custom profile into the global sparse config (profile keys overwrite, every other key is kept; `default` leaves per-arch/registry defaults such as `memory.kv_cache` and `speculation.dflash` at auto) and prints how many keys it set; `create` snapshots the current global config as a new custom profile; with no action it opens the profile wizard in `hipfire-tui`. `migrate`, `schema` and `profile` are global-only. |
| `hipfire config <tag> list\|get\|set\|reset ...` | Same, scoped to per-model keys. |

Do not inventory every key here — [CONFIG.md](CONFIG.md) owns defaults and ranges. Notable defaults from source: `dflash_mode=off` (schema default; Qwen3.8-27B tiers with a paired draft resolve `auto` through the registry layer), `speculation=auto`, `thinking=on`, `reasoning_effort=auto` (no preference: an explicit non-`auto` config/CLI/HTTP value wins; when every layer is unset/`auto`, Qwen3.8 and Ornith 1.5 use hipfire's `medium` default, so `reasoning_effort = "xhigh"` restores the official-template default), schema default `thinking_budget=med` (legacy named-cap route only; dropped+warned on effort-native models), `max_tokens=4096`, `idle_timeout=300`. Qwen Jinja contracts additionally accept an explicit integer think cap; other families do not synthesize a force-close mechanism. Reasoning axes and family examples: [CONFIG.md](CONFIG.md), [SERVE.md](SERVE.md).

## Quantization and calibration

| Command | Purpose |
|---|---|
| `hipfire quantize <hf-id\|dir\|file.gguf> [flags]` | CPU quantize via `hipfire-quantize`. GGUF weight input is deprecated since 0.4.0 (removal in 0.5.0): GGUF→mqN is lossy double quantization; quantize from the original safetensors checkpoint instead. GGUF input accepts only `hf4`, `hf6`, `mq4` and `mq6`. |
| `hipfire-quantize --flux-pipe <pipe_dir> --output <base.hfq>` | Pack a FLUX.1 or FLUX.2 Klein diffusers pipe into per-component HFQ files (`<base>-transformer.hfq` plus `-t5.hfq`, `-clip.hfq`, `-vae.hfq` for FLUX.1, or `-qwen3.hfq`, `-vae.hfq` for Klein; arch ids 40–46). `--flux-component` packs one. The packs are the only form the daemon loads. |
| `hipfire sidecar-gen <model> [flags]` | **Deprecated since 0.4.0 (CASK), removal in 0.5.0.** Build a `.triattn.bin` next to the model (does not pull); prints a deprecation warning. |
| `hipfire calibrate-npu-trunk --from <measurements.json> --out <cal.json>` | Write a keyed NPU trunk calibration from offline measurement inputs. Every measured cost is mandatory; the record's `key.kernel.uname_release` must equal the current `uname -r`, otherwise the command fails. |

### `quantize` (summary)

| Flag | Purpose |
|---|---|
| `--format <fmt>` | Repeatable: `mq4`, `mq6`, `q8`, `q8f16`, `hf4`, `hf6` (`hfq4`/`hfq4g256` map to `hf4`, `hfq6`/`hfq6g256` to `hf6`). With no `--format` and no `--both` the default is **`mq4`** for safetensors input and **`hf4`** for a `.gguf` file. |
| `--both` | Add `mq4` and `mq6` to the requested formats (suppresses the default format). |
| `-o` / `--output` | Exact output path; valid with exactly one format. |
| `--output-dir`, `--stem` | Output directory (default: working directory) and filename stem (default: the input's file/directory name, `.gguf` removed); files are `<stem>.<format>`. |
| `--install` | Copy into `~/.hipfire/models/`. |
| `--register <tag>` | Local alias in `~/.hipfire/models.toml`, pointing at the `mq4` artifact when produced, else the first one. |
| `--upload <owner/repo>`, `--create-repo` | Optional HF publish through the `hf` CLI (`--create-repo` creates the model repo first). |

Supported CLI formats include `mq4`, `mq6`, `q8`/`q8f16`, `hf4`/`hf6` and hfq aliases. Graded MoE recipes need the quantizer binary directly — [QUANTIZE.md](QUANTIZE.md).

### `sidecar-gen` (summary; deprecated since 0.4.0, removal in 0.5.0)

| Flag | Purpose |
|---|---|
| `--corpus PATH` | Calibration text (else builtin seeds). |
| `--max-tokens N` | Default `4000`. |
| `--chunk-len N` | Default `256`. |
| `--gpu-calib` / `--cpu-calib` | Device (GPU is the usual default path). |
| `-o PATH` | Output path (default `<model-path>.triattn.bin`). |
| `--skip-validation` | Skip post-gen validation. |

## Benchmarks and diagnostics

| Command | Purpose |
|---|---|
| `hipfire bench <model> [opts] [prompt]` | Prefill/decode timing. Every bench load is text-only (`vision_mode=off` is pinned whatever `vision.mode` resolves to: bench has no media input, and the default `auto` would otherwise add the ~0.9 GiB tower and shrink the load's measured capacity). `--runs N` (default 5), `--json`, `--exp` (RDNA2 variant sweep), `--max-seq N` ([context](#bench-context-max_seq)). `--prompt-file PATH` reads the prompt verbatim; JSON records `prompt_tokens`/`prompt_md5`/`prompt_chars`/`warnings` (short prompts warn that `prefill_tok_s` is launch overhead), the load's `max_seq`, and the GPU power and memory state of every measured run ([below](#bench-power-and-memory-state)). |
| `hipfire bench <model> --matrix ...` | Synthetic PP/context/TG matrix (`--pp`, `--ctx`, `--tg`, `--sustained-tg`, `--sustained-ctx`, `--warmups`, `--kv-mode`, `--kv-backend`, `--kv-k`, `--kv-v`, `--max-seq`, `--redline`). Same KV contract as `run`/`serve`, except that the load's `max_seq` is sized to the matrix ([context](#bench-context-max_seq)); `--json` surfaces the effective loaded `kv_mode` and backend fields and the loaded `topology` (`{pp,tp,ep,cards[]}`, [below](#bench-matrix-topology)), and each row carries a `power` entry per run. |
| `hipfire bench <model> --spec <m> ...` | Every bench generate sends greedy sampling with the neutral `repeat_penalty 1.0` (any other value routes the daemon to the plain AR decoder), so `--spec mtp\|dflash\|ngram\|dspark\|auto` measures the speculative decoder and `--spec off` the AR decoder on the same workload. JSON records `spec_requested`, a per-run `spec_route` (`"mtp"`, `"mtp+ngram"`, `"dflash"`, a daemon `spec` name, or `null` when AR ran) and per-run `spec_tau`; when `--spec` is not `off` but no run speculated, the bench prints a `warning:` and adds it to `warnings`. The `load_speculation` block records `mtp_mode` as requested beside `mtp_effective`, the daemon's `loaded` ack `mtp_mode_effective`: `auto` on a producer without a measured AR floor (Qwen3.5-class, DeepSeek V4) is effective `off`, so a `null` `spec_route` there is the capability gate rather than a failed attach, and `--spec mtp` measures the head (`mtp_effective` is `null` when the daemon or route reports none, as on EP loads). `spec_tau` averages over every window, including native MTP's AR floor tokens, which draft nothing, so a request the floor retired reads near zero whatever its drafts accepted. The per-run `spec_mtp` (`null` unless native MTP ran) splits the windows: `windows_mtp` (windows before retirement: native and n-gram windows plus the floor's timed AR tokens), `windows_floor_ar` (those timed AR tokens: the calibration token and any retirement probes), `windows_ar_retired` (head-free AR tokens after retirement), `retired_at` (index of the first retired window, `null` if the request never retired), `accepted` (the daemon's `done.accepted`) and `tau_mtp` = `accepted / (windows_mtp - windows_floor_ar)`, the acceptance of the windows that drafted (`null` from a daemon without `accepted` or `floor_ar_windows`). |
| `hipfire bench <model> --concurrency 1,2,4,8 ...` | Closed-loop serving sweep over the daemon wire: at each K, K lanes drain the same N requests, after a discarded warmup at that K. Reports output tok/s, steady-window tok/s, TTFT/TPOT p50/p99, achieved concurrency, τ, finish reasons and routes ([below](#bench-concurrency---concurrency)). `--json` is not supported on this path. |
| `hipfire bench <model> --emu <id\|off> ...` | SKU application-capacity profile for **every** spawned bench daemon arm (standard, `--matrix`, `--ttft`, `--concurrency`, `--exp`): sets `developer.emu` in each child's process snapshot, overrides `HIPFIRE_EMU` including `--emu off`, validates the id before anything spawns, and logs the effective value on stderr (`emu: effective profile ...`; silent when no profile is requested). The host still runs at full speed; only application memory capacity is capped. With a profile active the bench JSON `gpu` object (the daemon `diag`) gains an `emu` object (budgeted free/total, ledger); the `vram_free_mb`/`vram_total_mb` beside it stay physical. See [EMU.md](EMU.md). |
| `hipfire emu list [-j\|--json]` | Print the SKU emulation registry: id, host gfx, target gfx, target CUs, memory, peak bandwidth, IC/L2, VGPR file, fidelity class. CPU-only. |
| `hipfire emu validate <id> [-j\|--json] [--mib N ...]` | Start a fresh model-free daemon with `developer.emu=<id>` and run the `emu_validate` verb (finite eager/graph traffic and topology checks, ledger checks; `--mib` buffer sizes, 1..=2048, repeatable). Prints a summary plus the pretty receipt, or one raw JSON line with `--json`. Exits nonzero when the capacity, host-topology or graph section fails (or the receipt status is `fail`); `unavailable` is reported, not failed. The daemon refuses a request whose id differs from the installed profile, or a loaded model. Requires a compatible host (see [EMU.md](EMU.md)); GPU runs belong in the fabric scheduler. |
| `hipfire emu estimate <id> --input FILE [-j\|--json]` | CPU-only derating **estimate** of the profile's target from a measured `DeratingInput` JSON (schema v1: run identity, host/target resource rates, per-kernel observations). Never starts a daemon, never touches HIP. The report keeps `measured` (input), `estimate` (model output; `estimated_wall_ns` is `null` with reasons when time closure, rates or identity are missing) and `target_profile` (registry metadata) apart. Exit 0 also when the estimate is `unavailable`; nonzero for unreadable or malformed input. |
| `hipfire profile [model] [--kernel substr] [--json]` | Live daemon roofline and compiled-kernel VGPR/SGPR/LDS/occupancy report. Use `hipfire-atlas` for measured ISA-fit and workload analysis. |
| `hipfire diag [-j\|--json]` | Static device/runtime checks plus a live HIP arch, version, and VRAM probe when the daemon is available. |
| `hipfire --version` | Concise semver + build commit + source ref identity. |
| `hipfire version [--json]` | Detailed build/source match, checkout state, target, config schema, and installed-daemon SHA-256. |
| `hipfire update [REF]` | **Linux managed installs:** advance the current branch or auto-detect and install a branch/tag/commit; a leading `@` on `REF` is optional. |
| `hipfire update --branch NAME` | Switch to and track a remote branch such as `master` or `beta`. |
| `hipfire update --tag TAG` / `--commit SHA` | Install an immutable detached revision. A later explicit selector moves away from the pin. |
| `hipfire setup --source PATH [flags]` | Install or repair this machine's runtime from a source checkout (normally run by `scripts/install.sh`; `--source` is required and must contain `Cargo.toml`). Flags: `--rocm-root PATH`, `--hipcc PATH` (also `HIPFIRE_HIPCC`), `--strict-rocm` (also `HIPFIRE_ROCM_STRICT=1`), `--gpu-arch ARCH`, `--profile auto\|hip\|redline` (`auto` leaves `replay.backend=auto` so `.mq4r` models select Redline), `-y`/`--yes` (alias `--non-interactive`), `--compile-kernels` (compile locally with `hipcc` even when a release kernel pack exists), `--kernel-pack-url URL` (directory or URL holding the kernel-pack assets; default: the tag's GitHub release). A release tag selects that release's prebuilt kernel pack, which needs no device compiler. |
| `hipfire kernel-pack install --tag TAG --arch ARCH --source PATH [--dest DIR] [--url URL] [--rocm-version V]` | Download, verify and install the prebuilt kernel pack for `TAG` and `ARCH`. `--source` is the checkout the installed daemon was built from; its `HEAD` must be the pack's commit. `--dest` defaults to `~/.hipfire/bin/kernels/compiled/<ARCH>`; `--rocm-version` (default: the resolved ROCm root's `.info/version`) is checked against the pack's supported range. A successful install also writes the pack's verified `manifest.json` beside the installed directory as `<ARCH>.manifest.json` (for example `~/.hipfire/bin/kernels/compiled/gfx1151.manifest.json`), through a temporary file and a rename. `hipfire doctor --npu` reads it to name the installed kernel set. |
| `hipfire calibrate [--show\|--check\|--clear] [--json]` | Measure this host's PCIe link, CPU read bandwidth and GPU/CPU offload rates in at most 60 s and record them, with the hardware fingerprint they were measured on, in `config.toml` `[calibration]`; the engine then skips its load-time probes while the record is fresh. `--show` prints the record and the routes it selects (nothing measured), `--check` exits 0 when a fresh record covers every detected GPU, `--clear` removes the record. Refuses while a daemon runs. See [below](#hipfire-calibrate). |
| `hipfire doctor [-j\|--json]` | GPU-free runtime/driver/version preflight: amdgpu driver binding, `/dev/kfd` and render-node access, KFD topology, ROCm root, HIP/HSA runtime, device libraries for each GPU arch, `HSA_OVERRIDE_GFX_VERSION`, kernel pack and daemon build match. Prints one `fix:` line per failed or warned check; exit 1 when any check fails. `-j`/`--json` prints one report (schema `hipfire-doctor-runtime-1`) with no hostname, user, path or PCI address. Linux only. See [below](#hipfire-doctor-runtime-preflight). |
| `hipfire doctor --npu [--yes] [-j\|--json] [--apply]` | Strix Halo (gfx1151 + XDNA2) NPU trunk fitness check: GPU-free checks, a tier, then (after confirmation) a GPU-only vs `auto` prefill A/B and a greedy `on`/`emulate` exactness check, about 6–7 minutes. `-j`/`--json` prints one report with no hostname, user, path or PCI address; `--apply` sets `npu_trunk off` on a measured loss or an exactness mismatch and changes nothing else. Linux only. See [below](#hipfire-doctor---npu). |
| `hipfire doctor --npu --soak [--hours H] [--out DIR] [--yes]` | Stability soak instead of the A/B: cold long-prompt prefills with the NPU and GPU busy together for `H` hours (default 4, at most 24), with a kernel-log watch and a heartbeat. Stops at the first fatal kernel line, hang, daemon exit or low memory. Writes only into `DIR` (default `npu-soak-<UTC timestamp>` in the current directory). |
| `hipfire doctor --npu --soak-report DIR` | Summarize a soak directory, including the previous boot's kernel log after a freeze and reboot. Runs no checks and does not need the GPU or NPU. |
| `hipfire fit <model> [--ctx N] [--kv-mode M] [--kv-k K] [--kv-v V] [--spec S] [--no-fetch] [-j\|--json]` | Say, for this machine, whether `<model>` loads at context `N` and what the load would choose (device, topology, KV backend and mode, context bound, weight placement, decode route and why, host-link routes), plus the expected speed class from published receipts. `--kv-mode`, `--kv-k`, `--kv-v` and `--spec` take the same one-shot overrides as `hipfire run`, so the answer is for that exact load. When it does not fit, it names the term that fails and the change that fixes it. No HIP call, no daemon start, no weight or draft download; machine facts come from sysfs, `/proc`, the KFD topology and the `hipfire calibrate` record. Exit 0 fits, 1 does not fit, 2 undetermined. `<model>` may be a registry tag that is not installed (only its header is fetched; `--no-fetch` forbids that). See [below](#hipfire-fit). |

Perf claim protocol (warmup, fresh-process, noise): [methodology/perf-benchmarking.md](methodology/perf-benchmarking.md). Published tables are measured/historical: [BENCHMARKS.md](BENCHMARKS.md).

`hipfire bench` flags (summary; `hipfire bench --help` is authoritative):

| Flag | Purpose |
|---|---|
| `--runs N` | Measured runs; default `5`, must be positive. |
| `--tp N` | Tensor/expert-parallel degree for the bench daemon, `1..=64`. |
| `-j`, `--json` | JSON report; not supported with `--exp` or `--concurrency`. |
| `--exp` | RDNA2 kernel-variant sweep in isolated daemon processes; cannot combine with `--matrix`/`--redline`, `--json` or `--reasoning-on`. |
| `--matrix` / `--redline` | Synthetic token-id rows; neither takes a prompt. Runs on every topology `serve` admits (single card, `--tp` / `topology.pp` PP, TP, EP) and reports the loaded `topology` ([below](#bench-matrix-topology)). `--redline` additionally selects the Redline replay backend (PM4 transport, graph AR/forward). |
| `--pp`, `--ctx` | Comma lists for the matrix; default `128,512,2048,4096,8192,20000`. |
| `--tg N` | Matrix decode length; default `128`. |
| `--sustained-tg N`, `--sustained-ctx` | Sustained-decode rows run when `--sustained-tg` is set; `--sustained-ctx` defaults to `128,8192`. |
| `--warmups N` | Discarded warmup requests before the measured runs of the standard bench, `--ttft` and `--matrix`; default `10`. A standard-bench warmup is a full request with the same prompt and `--max-tokens` as a measured run, and `0` skips warmups entirely. The standard `--json` output records the count actually run as `warmups_run`. Not used by `--concurrency`, which always runs its own discarded warmup per K ([below](#bench-concurrency---concurrency)). |
| `--ttft` | Client-side time to first token: per run, generate one token on the standard prompt and time request-send to first streamed token; reports `prompt_tokens` and TTFT. Standard path only (not with `--matrix`). |
| `--max-tokens N` | Generated tokens per measured request: default `128` on the standard bench, `256` on the `--concurrency` `steady` shape. |
| `--reasoning-on` | Let a reasoning model think during the bench (off by default, because a think span usually cannot close inside the benchmark's token budget and the daemon then fails that turn closed); pair with a large enough `--max-tokens`. |
| `--spec <m>`, `--kv-mode`, `--kv-backend`, `--kv-k`, `--kv-v`, `--max-seq` | Same contracts as on `run`/`serve`. |
| `--prompt-file PATH` | Read the standard-bench prompt verbatim (raw bytes); mutually exclusive with positional prompt words. |
| `--emu <id\|off>` | SKU capacity-emulation profile for every spawned bench daemon; overrides `HIPFIRE_EMU`, including `off`. Unknown ids and UMA-knob conflicts fail before any daemon starts. See [EMU.md](EMU.md). |
| `--concurrency`, `--backend`, `--workload`, `--bench-shape`, `--num-prompts`, `--temperature` | Concurrency sweep controls ([below](#bench-concurrency---concurrency)). |

### Bench concurrency (`--concurrency`)

`hipfire bench <model> --concurrency 1,2,4,8 [--backend batch|noslots|slots|both] [--workload stateless|multiturn|both] [--bench-shape <shape>] [--num-prompts N] [--temperature T] [--reasoning-on] [--runs N]` measures serving throughput with a closed-loop sweep at each concurrency level K (`--bench-shape` defaults to `steady`; `hipfire bench --help` lists the other shapes and the flags they need):

- **Closed loop at K.** K client lanes drain one FIFO of N requests. A lane sends its next request the moment its previous one ends, so there is no burst barrier and no lane idles until the queue is empty. Request i is prompt `i mod len` of the shape's list, so every K sends the same multiset in the same order.
- **Warmup discarded.** Before every sample, 8 separate warmup prompts run closed-loop at the same K with 64-token outputs. No measured prompt is ever a warmup prompt.
- **No prefix-cache hits.** Each request's first message starts with a `Q<run>-<i>.` tag, unique per `--runs` round and request. Every K of a round therefore sends the same tokens, and a later round's repeat never hits the daemon's prompt cache. `cached` reports any prompt tokens the daemon served from cache.
- **Chat turns, neutral sampling.** Requests carry `messages` (plus `prompt.system`, as `hipfire serve` injects it), thinking off unless `--reasoning-on`, and neutral penalties (`repeat_penalty` 1.0, `presence_penalty` 0, `min_p` 0).
- **Levels are interleaved** across `--runs` (`1,2,4,8,1,2,4,8,…`), and the table reports each metric's median over the runs.

| Backend | What it loads and sends |
|---|---|
| `batch` | The VMM continuous-batching route `hipfire serve` runs with `serve.vmm_batch` on: `continuous_batch_size` = max K (at least 2) plus the `serve_vmm_batch` load object built from `serve.batch_spec`, `serve.batch_nonexact`, `serve.max_batch_tokens` and `serve.prefill_min_tokens`. Requests are the chat requests serve forwards, with `serve_continuous_batch`. The driver refuses a daemon that did not stage `continuous_batch_route: "vmm"`. A lone request takes the singleton route, as under serve. |
| `noslots` | No batching. The daemon serves one request at a time and the other lanes wait client-side, as serve queues them with batching off. The wait counts in TTFT. |
| `slots` | The experimental multi-slot engine (`experimental_multi_slot`). |
| `both` | All three, one after another. Only one daemon is resident at a time. |

The `multiturn` workload makes each request a two-turn chat. The follow-up ("Explain that again in one sentence.", with the first answer as the assistant turn) is sent on the same lane as soon as the first turn completes, and both turns count. On `slots`, a sample whose follow-ups recorded no prefix-cache hit is reported `[invalid]` and dropped.

| Shape | Requests | Output length | Sampling |
|---|---|---|---|
| `steady` (default) | 64 long-answer code prompts built into the binary (`--num-prompts N` cycles them), measured-set md5 `f0b077a766edc166a0c3f18cf81c169c` | `--max-tokens`, default 256 | greedy |

`--temperature` overrides the shape's temperature (default `0` for `steady`). The daemon has no `ignore_eos`, so `steady` fixes the output length by prompt choice: its prompts ask for answers longer than 256 tokens. Check that the `finish` column reads `length` only. Sampled requests (temperature > 0) are not batched: the VMM route serves greedy requests only, so a sampled sweep measures the singleton route queued K deep.

Each sample prints one line on stderr: output tok/s, steady tok/s, TTFT and TPOT p50/p99, in-flight, τ, ok/total, tokens and duration, finish reasons, routes (`execution_mode/batched=`), cached prompt tokens, mean prompt tokens and the md5 of the decoded texts. The final table has one row per (backend, workload, K):

- **out tok/s**: output throughput, output tokens ÷ (last completion − first send). The median of the runs, with min–max in brackets.
- **steady tok/s**: tokens whose arrival falls in the steady window ÷ the window. The window runs from the moment all of the first K requests have a token to the first completion after the last send.
- **TTFT**: first token − send. **TPOT**: (completion − first token) ÷ (tokens − 1). Both are the medians of the per-sample p50/p99.
- **in-flight**: achieved concurrency, Σ request latency ÷ duration.
- **τ**: output tokens per verify window, token-weighted over the sample.
- **tokens**: median output tokens per sample.
- **finish**, **batched**, **cached**, **errors**: summed over all samples. A failed request counts no tokens.

A daemon that sends no frame for 300 s fails the sweep. Its in-flight requests are cancelled first.

### Bench context (`max_seq`)

`--max-seq N` sets the benchmark load's KV context, within the `memory.max_seq` bounds (512–1048576), and replaces any configured `memory.max_seq`.

Without it, `--matrix` and `--redline` size the load to the run instead of using the configured or automatic context. The run needs the largest `--pp`, or the largest `--ctx` plus `--tg` (and `--sustained-ctx` plus `--sustained-tg` when sustained rows run), whichever is larger, plus the 32 positions of headroom the daemon's probes keep. That is raised to the `memory.max_seq` minimum of 512 and capped at the model's declared context (`max_position_embeddings` in its metadata). `--pp 8192 --ctx 128 --tg 32` loads `max_seq` 8224.

- A shape past the model's declared context fails before the daemon starts, naming the run's need and the model's limit. This applies with or without `--max-seq`.
- A `--max-seq` below a matrix run's need fails the same way, naming both numbers.
- Past the declared context, a `--max-seq` that holds the run is sent as given, and the daemon's admission is the bound. For example, Qwen4 refuses anything above its admitted 262144.

The standard, `--ttft`, `--exp` and `--concurrency` paths have no fixed shape, because the prompt is tokenized by the daemon. Without `--max-seq` they keep the configured or automatic context.

Every JSON report has a `max_seq` object:

- `effective`: what the daemon loaded.
- `requested`: the value sent, or `null`.
- `source`: `flag`, `run_shape`, `config` (an authored `memory.max_seq`) or `automatic`.
- `run_need`: the matrix need, or `null`.
- `model_limit`: the declared context, or `null`.

The bench also prints a `max_seq:` line.

### Bench matrix topology

`--matrix` measures the model on whatever topology the load admitted: one card, pipeline-parallel (`topology.pp`, 2 or more stages), dense tensor-parallel (`--tp` / `topology.tp`) or expert-parallel (EP). Admission is `serve`'s own, so a load `serve` refuses is refused by the bench with the same message. The rows are synthetic: fixed token ids on the plain prefill route and a primed synthetic decode context. They time the loaded topology's forward, not a prompt, speculation or the HTTP path, and a multi-card load's rows carry that load's cross-card traffic. Time a real prompt with `--prompt-file` (see the standard bench).

The `--matrix` JSON gains a `topology` object, taken from the daemon's own `prefill_result` / `decode_result` answers and not derived from the flags:

| Field | Meaning |
|---|---|
| `pp` | Pipeline stages (`1` when not pipelined). |
| `tp` | Dense tensor-parallel ranks (`1` when none). |
| `ep` | Expert-parallel ranks (`1` when none). |
| `cards` | Identifiers of the cards that actually hold the model, as a non-empty array of strings (HIP device ordinals in mesh rank order, each card listed once). |

A single-card load is `{"pp":1,"tp":1,"ep":1,"cards":["0"]}`; a PP3 load `{"pp":3,"tp":1,"ep":1,"cards":["0","1","2"]}`; `tp` and `ep` are reported separately, never merged. All rows of one run must report the same topology (including the card list), and `topology.pp` must equal the report's `pp` (the loaded value), otherwise the bench fails instead of labeling the run. A daemon that reports no topology yields `"topology": null`, never an assumed single card. Every other `--matrix` field is unchanged.

### Bench daemon lifetime

`hipfire bench` owns the daemon it spawns (and, through it, the machine-wide per-GPU `flock` of the cards the load reserves). The daemon is killed and reaped when the bench returns or fails, and when the bench receives SIGINT, SIGTERM or SIGHUP; an interrupted bench prints `hipfire bench: interrupted; terminated N owned daemon(s)` and exits 130. Reaping completes before the CLI exits, so a harness that waits for `hipfire bench` can start the next arm on the same cards immediately. Only daemons this bench process spawned are touched. `SIGKILL` of the CLI itself cannot be intercepted and leaves the daemon running, so harnesses should send SIGTERM to the CLI first and escalate to SIGKILL (on the whole process group) only after a grace period.

There is no load timeout in the CLI, the client or the daemon: a cold multi-GPU load (first-use kernel compilation, 100+ GB of weights) runs until it finishes, and any deadline is the caller's. Size one for a cold kernel cache, or warm `kcache` with an earlier run first.

### Bench power and memory state

`hipfire bench` samples the GPU every 0.5 s during each measured run (`--ttft` and `--matrix` too; not `--exp` or `--concurrency`). It reads only world-readable sysfs, so it needs no root, and it never fails a bench: whatever it cannot read is reported as the string `"unavailable"`.

- **`power_device`** (once per report): the card sampled, `card` (`/sys/class/drm/cardN`), `pci`, `arch`, `selected_by` and `sample_period_ms`. It is the daemon's logical device 0: the card named by `hardware.devices` (`HIPFIRE_DEVICES`), else the first entry of an inherited `ROCR_VISIBLE_DEVICES`, else the first ROCr agent of the daemon's arch. With `--tp` above 1 only that first card is sampled. If no card resolves, it holds `status: "unavailable"` and a `reason`.
- **`power`**: one object per measured run, in the order of the `samples` arrays (top level for the standard and `--ttft` paths, per row for `--matrix`):

| Field | Source | Meaning |
|---|---|---|
| `samples`, `duration_s` | — | Samples taken (one at run start, every 0.5 s, one at run end) and the span they cover. |
| `gpu_metrics_format` | `gpu_metrics` header | `"3.0"` is decoded (Strix Halo and other `gpu_metrics_v3_0` APUs). Any other `format.content` (e.g. `"1.3"` on RDNA dGPUs), `"malformed"`, or `"unavailable"` (no file) leaves every `gpu_metrics` field below `"unavailable"`. |
| `gfxclk_mhz` `{avg,min,max}` | `gpu_metrics` | `average_gfxclk_frequency`. |
| `gfx_maxfreq_mhz_min` | `gpu_metrics` | Lowest `current_gfx_maxfreq` the SMU allowed during the run. |
| `socket_power_w`, `gfx_power_w` `{avg,max}` | `gpu_metrics` | `average_socket_power`, `average_gfx_power`. |
| `gfx_temp_c_max` | `gpu_metrics` | Highest `temperature_gfx`. |
| `fclk_mhz_avg` | `gpu_metrics` | Mean `average_fclk_frequency`. |
| `throttle_pct` | `gpu_metrics` | Percent of the run's wall time at each limiter: `fast_ppt`, `slow_ppt`, `stapm` (the SMU's SPL counter), `thm_gfx`, `thm_core`, `thm_soc`, `prochot`. The SMU counters are cumulative milliseconds; the percentage is their increase between the first and last sample over the time between them, capped at 100. |
| `tctl_c_max` | k10temp hwmon | Highest CPU package `Tctl` (meaningful on APUs, where the GPU shares the package). |
| `perf_level` | `power_dpm_force_performance_level` | Read at run start. |
| `od_sclk` `{min_mhz,max_mhz,range_min_mhz,range_max_mhz}` | `pp_od_clk_voltage` | The `OD_SCLK` levels and the `OD_RANGE` SCLK bounds, at run start. |
| `hwmon` | the card's amdgpu hwmon | Every labelled input, keyed by its label: `temp_c` (max; e.g. `edge`, `junction`, `mem`), `power_w` (`powerN_average`, else `powerN_input`; e.g. `ppt`) and `freq_mhz` (e.g. `sclk`, `mclk`), with avg/min/max. This is all that remains on a card whose `gpu_metrics` format is not decoded. |
| `memory` | `mem_info_{vram,gtt}_{total,used}`, `/proc/meminfo` | At run start, in MiB: `vram_total_mib` (the UMA carve-out on an APU), `vram_used_mib`, `vram_free_mib`, the same three for `gtt`, and the host's `mem_total_mib` and `mem_available_mib`. Shrinking the BIOS carve-out and serving from GTT changes these, so tag baselines with them. |

`gpu_metrics` values are the SMU's time-filtered averages, so the first second of a run lags the load.

When any limiter's residency exceeds 10 % of a measured run, the bench prints one `warning:` line naming the worst residency per limiter, and adds it to the JSON `warnings`. The number is then power- or thermal-limited and is comparable only with runs whose power fields match.

**High-TDP APU thermal note.** On a high-TDP Strix Halo box at stock power limits, long prefill (e.g. pp8192) can be limited first by fast PPT and then by GPU temperature, which reaches the thermal point within seconds of prefill start. Capping the four SMU limits (STAPM, fast PPT, slow PPT, APU slow PPT) to a common value around 130 W (`ryzenadj --stapm-limit=130000 --fast-limit=130000 --slow-limit=130000 --apu-slow-limit=130000`, or the BIOS power mode) can keep the GPU below its thermal limit at some cost in burst prefill throughput; lower caps trade more throughput for energy efficiency. The effect depends on chassis, cooling and firmware, so measure the trade-off on your own machine. The power limits need root and are not visible to the bench, so record them with the result.

### `hipfire doctor` (runtime preflight)

`hipfire doctor` with no `--npu` checks the driver, ROCm and hipfire install of this machine before any model loads, and prints one `fix:` line for every failed or warned check. It is GPU-free (no HIP call, no daemon start) and needs no network. It is Linux only. The only flag is `-j`/`--json`; `--yes`, `--apply`, `--soak`, `--hours`, `--out` and `--soak-report` belong to `--npu` and are errors without it.

Each check prints `PASS`, `FAIL`, `WARN` or `INFO` with what it found. A `FAIL` blocks: the last line is `result: blocked (N failed)` and the exit status is 1. `WARN` and `INFO` never block (`result: ok`, exit 0). Like the NPU report, nothing printed carries a hostname, user name, filesystem path or PCI address; GPUs are named `GPU<i> gfxNNNN (1002:<device id>)`, with `<i>` the position in PCI address order. `hipfire fit` runs the same checks first.

| Check id | Blocks? | Reads | Fix |
|---|---|---|---|
| `amdgpu_driver` | yes | `/sys/bus/pci/devices/*` (AMD display devices: vendor `0x1002`, class `0x0300`/`0x0380`), each device's `driver` link, `/sys/module/amdgpu/version` (present on out-of-tree DKMS builds) and the kernel release. `FAIL` when no AMD GPU is on the bus, or a GPU has no driver or a driver other than `amdgpu`. | Unbound: check `journalctl -k -b \| grep -iE 'amdgpu\|firmware'`; a "failed to load firmware" line means install the current linux-firmware package, no amdgpu line means the kernel predates the GPU (boot a newer kernel or install amdgpu-dkms from the same ROCm release as userspace). `vfio-pci`: remove the GPU from the `ids=` list and reboot. `radeon`: boot with `radeon.si_support=0 radeon.cik_support=0 amdgpu.si_support=1 amdgpu.cik_support=1`. Another driver: unbind it and let amdgpu bind. |
| `kfd` | yes | `/dev/kfd`: exists, is a character device, and is readable and writable by this user. | Missing: `journalctl -k -b \| grep -i kfd`, `CONFIG_HSA_AMD=y`, current linux-firmware (MES/MEC). Not accessible: `sudo usermod -aG render,video $USER`, then log out and back in. |
| `kfd_gpus` | yes | The KFD topology (`/sys/class/kfd/kfd/topology/nodes`): every amdgpu-bound GPU must have a KFD node with the same PCI address. A topology that exists but cannot be read is a `FAIL` naming the read error, not "KFD did not register"; the arch-dependent checks are then omitted. A missing topology directory is a known absence. | The `kfd` journalctl and firmware fix; an unreadable topology: check `/sys/class/kfd/kfd/topology` is readable, reload amdgpu or reboot. |
| `render_node` | yes | The DRM render node of each KFD GPU (`/sys/bus/pci/devices/<gpu>/drm/renderD*`, `/dev/dri/renderD*`): present and readable and writable. | Missing: `journalctl -k -b \| grep -i amdgpu`. Not accessible: the `usermod` line above. |
| `firmware` | no (always `INFO`) | `/sys/bus/pci/devices/<gpu>/fw_version/*` per amdgpu-bound GPU, for example `mes 0x86, mec 0x12`; "not exposed by this kernel" when the directory is absent. The values [docs/KNOWN-ISSUES.md](KNOWN-ISSUES.md) asks you to attach. | None. |
| `rocm_root` | yes | The ROCm root hipfire resolves (`HIPFIRE_ROCM_PATH`, `ROCM_PATH`, `HIP_PATH`, `/opt/rocm`, ...) and its `.info/version`. `FAIL` when none is found or several complete installs are present with none selected. | None found: the install guidance (AMD's live install selector). Ambiguous: `export HIPFIRE_ROCM_PATH=/opt/rocm-<version>`. |
| `hip_runtime` | yes | `libamdhip64` and `libhsa-runtime64` (`lib`, `lib64`) in the selected root. | Install the ROCm HIP runtime packages for this ROCm version. |
| `rocm_isa` | yes, or `WARN` | Per KFD arch, `oclc_isa_version_<arch>.bc` in the root's device libraries (`amdgcn/bitcode`, `lib/llvm/amdgcn/bitcode`, `llvm/amdgcn/bitcode`, `lib/llvm/lib/clang/*/lib/amdgcn/bitcode`, `*/lib/llvm/amdgcn/bitcode`). No bitcode directory at all (runtime-only install): `WARN`. A bitcode directory without this arch's file means this ROCm predates the GPU: `FAIL`, or `WARN` when an installed kernel pack admits this ROCm (registry kernels still run). | Runtime-only: install the ROCm HIP SDK, or `hipfire kernel-pack install --tag <tag> --arch <arch> --source ~/.hipfire/src`. Predates the GPU: install a newer ROCm (6.4 for gfx1200/gfx1201, 7.2 for gfx1150/gfx1151/gfx1152, 6.0 otherwise). |
| `rocm_version_floor` | no (`WARN`) | The root's version against hipfire's documented minimum for each KFD arch (6.4 gfx1200/gfx1201, 7.2 gfx1150/gfx1151/gfx1152, 6.0 otherwise), compared numerically. `INFO` when the version is unknown. Capability itself is gated by `rocm_isa`. | Install ROCm at or above the minimum for that arch. |
| `hsa_override` | no (`WARN`) | `HSA_OVERRIDE_GFX_VERSION`. Unset: `PASS`. Equal to every KFD arch: `INFO` (redundant). Different from a KFD arch, or unparseable: `WARN`; hipfire honours the override, so it builds and runs the overridden arch's kernels on the GPU. | `unset HSA_OVERRIDE_GFX_VERSION` (and remove it from your shell profile or service unit) unless you rely on it for a GPU ROCm does not support. |
| `kernel_pack` | yes, or `WARN` | Per KFD arch, the kernel pack manifest recorded beside the daemon (`kernels/compiled/<arch>.manifest.json`). No manifest: `INFO` (kernels JIT-compile on first use). The ROCm version outside the pack's `[rocm_min, rocm_max_exclusive)`: `FAIL`. A pack built from a different commit than the daemon: `WARN`; the daemon rejects every object whose recorded key (kernel source, flags, toolchain, cache ABI) no longer matches and recompiles it with hipcc on first use, and without a device compiler it fails as an "unverified kernel image". | Out of range: `hipfire setup --source ~/.hipfire/src --compile-kernels` (needs the ROCm HIP SDK), or reinstall a ROCm in the pack's range. Other commit: `hipfire update`, or `hipfire kernel-pack install --tag <tag> --arch <arch> --source ~/.hipfire/src`. |
| `daemon` | yes, or `WARN` | The found daemon binary and its `--version` line (`hipfire-daemon <version> <commit>`, 5 s timeout) against this CLI's version and commit. Not installed: `FAIL`. No answer, or a different version or commit: `WARN`. | Not installed or no answer: `hipfire setup --source ~/.hipfire/src`. Different builds: `hipfire update` (or re-run the installer). |
| `kho_scratch` | no (`WARN`) | Only on unified-memory APUs: the Kexec HandOver scratch reservation (`/proc/meminfo`, `/proc/zoneinfo`), warned at 1 GiB or more. `INFO` ("KHO scratch unknown") when either file cannot be read; a successful read with no reservation is `PASS`. | Boot with `kho=off` ([KNOWN-ISSUES.md](KNOWN-ISSUES.md)). |

Checks that need a ROCm root or a KFD GPU (`hip_runtime`, `rocm_isa`, `rocm_version_floor`, `render_node`, `kernel_pack`, `kho_scratch`, `firmware`) are omitted when there is none; the failing `rocm_root`, `kfd` or `kfd_gpus` check says why.

`-j`/`--json` prints one object, schema `hipfire-doctor-runtime-1` (same privacy policy), and the exit status is the same:

```json
{
  "schema": "hipfire-doctor-runtime-1",
  "hipfire": {"version": "0.4.3", "commit": "<first 12 hex>"},
  "result": "ok" | "blocked",
  "host": {
    "kernel": "<osrelease>",
    "amdgpu": {"source": "in-tree" | "dkms", "version": "<dkms version>" | null},
    "rocm": {"version": "<.info/version>" | null},
    "gpus": [{"index": 0, "arch": "gfx1201" | null, "pci_id": "1002:7551",
              "driver": "amdgpu" | null, "firmware": {"mes": "0x86"}}]
  },
  "checks": [{"id": "kfd", "status": "pass" | "fail" | "warn" | "info",
              "detail": "...", "fix": "..." }]
}
```

`fix` is present only on `fail` and `warn` checks; several fixes of one check are separated by newlines.

### `hipfire doctor --npu`

`hipfire doctor --npu` answers one question for a Strix Halo host: can the NPU trunk prefill route run here, and does `npu_trunk = auto` beat the GPU-only route on this machine? Pass `--npu` to select it (plain `hipfire doctor` is the [runtime preflight](#hipfire-doctor-runtime-preflight)); the flags are:

| Flag | Purpose |
|---|---|
| `--npu` | Select this check: the Strix Halo (gfx1151 + XDNA2) NPU trunk route. Without it, `hipfire doctor` runs the [runtime preflight](#hipfire-doctor-runtime-preflight). |
| `--yes` | Run the calibration (or soak) without asking first. Needs `--npu`. |
| `-j`, `--json` | Print the report as one JSON object on stdout. |
| `--apply` | Set `npu_trunk` to `off` when the calibration measures a loss or an exactness mismatch. Cannot combine with `--soak` or `--soak-report`. Needs `--npu`. |
| `--soak` | Run the stability soak instead of the calibration. Needs `--npu`. |
| `--hours H` | Soak duration in hours; default 4; a positive number, at most 24. Needs `--soak`. |
| `--out DIR` | Soak output directory; default `npu-soak-<UTC timestamp>` (for example `npu-soak-20261016T120000Z`) in the current directory. It must not exist or must be empty. Needs `--soak`. |
| `--soak-report DIR` | Summarize the soak in `DIR`. Cannot combine with `--soak` or `--apply`. Needs `--npu`. |

The command is Linux only. It reads Linux sysfs, proc and device nodes, so on another OS the first checks fail and the report is `blocked`. The checks open neither the GPU nor the NPU; only the calibration and the soak do.

**Checks.** Every check prints `PASS`, `FAIL`, `WARN` or `INFO` with what it found. A failed check blocks: nothing is run and the exit status is 1. `WARN` and `INFO` checks never block and never change the tier. A failed or warned check prints a `fix:` line.

| Check id | Blocks? | Reads | Fix |
|---|---|---|---|
| `gpu_arch` | yes | The KFD topology (`/sys/class/kfd/kfd/topology/nodes`): a `gfx1151` GPU. | None; the route needs a Strix Halo GPU. |
| `npu_xdna2` | yes | `/sys/class/accel/accel0/device`: driver `amdxdna`, PCI vendor `0x1022`, device `0x17f0`. | Boot a kernel with the `amdxdna` driver (in-tree since Linux 6.14) and enable the NPU (IPU) in the firmware setup. |
| `accel_node` | yes | `/dev/accel/accel0` is a character device that this user can read and write. | `sudo usermod -aG render $USER`, then log out and in. A missing node points back to `npu_xdna2`. |
| `npu_firmware` | yes | The `accel0` driver link and a non-empty `fw_version`. | Install `linux-firmware` with `amdnpu/17f0_*` and reload the driver: `sudo modprobe -r amdxdna && sudo modprobe amdxdna`. |
| `memlock` | yes | `RLIMIT_MEMLOCK` (soft and hard), `CAP_IPC_LOCK` and the process's `VmPin`. Passes when the soft or the hard limit (unlimited counts) covers `VmPin` plus the bytes the armed trunk pins, or when `CAP_IPC_LOCK` is held. | A bounded limit sized from the requirement, not `infinity`: the fix line gives the systemd `LimitMEMLOCK`, the `limits.d` and the `ulimit -l` values. See [Host memlock limit](npu/npu-trunk.md#host-memlock-limit). |
| `rocm_runtime` | yes | A ROCm root that holds the HIP runtime library (`libamdhip64`). | The ROCm install guidance, printed in the fix line. |
| `daemon` | yes | Finds `hipfire-daemon` and runs `hipfire-daemon --version` (5 s limit). Passing needs only that the daemon is found; the detail is its version line when it answers. | `hipfire setup` |
| `model` | yes | The local Flash-Next models (HFQ architecture 16). A model passes when its NPU trunk index digest (`model_trunk_index_sha256`) equals a calibration record: the built-in records for the `stock` and `overlay:gfx1151-zab-v1` routes, or the override `npu-trunk/calibration.json`. | `hipfire pull qwen3.8:flash-next` |
| `gpu_free` | yes | Whether `hipfire serve` is running (`serve.pid`) and a non-blocking probe of the GPU's lock file. Reports `INFO` and skips when there is no gfx1151 GPU. | `hipfire stop` |
| `npu_trunk_config` | no | The effective `fusions.npu_trunk`. | None. |
| `fclk` | no | The fabric clock state of the GPU (pinned or automatic) and whether it could be pinned without a password (writable attributes or `sudo -n`). Serving runs it on automatic. | None. |
| `power` | no | Whether the GPU is running below full power: the system power mode (`/sys/firmware/acpi/platform_profile`, else `/sys/class/platform-profile/*/profile`), the GPU power profile (`pp_power_profile_mode`), the power limit against its default (`power1_cap`, `power1_cap_default`; `power1_cap_max` is only shown), the GPU performance level, and the GPU clock ceiling against the hardware maximum (`pp_dpm_sclk`, `pp_od_clk_voltage`), with the current clock for reference. `WARN` when the system mode is `low-power`, `quiet`, `cool` or `balanced`, the GPU profile is `POWER_SAVING`, the power limit is under 80 % of its default (not judged when the card reports no default), the clock ceiling is under 90 % of the hardware maximum, or the performance level is `low`. hipfire's published speeds assume full power. `INFO` when nothing could be read or there is no gfx1151 GPU. A current clock below its maximum is not a warning: an idle GPU clocks down. | The warning names each cause and gives one fix line for each, such as `powerprofilesctl set performance`; raise a lowered power limit in the BIOS or your vendor's power tool. |

**Tiers and profiles.** The tier follows from the checks and a table of validated host profiles compiled into the CLI:

| Tier | Meaning |
|---|---|
| `reference` | No check failed and every profile field equals one validated row. |
| `compatible-unvalidated` | No check failed, but at least one field differs from every row. The report lists each difference as `field: found X (reference Y)` against the nearest row (fewest differing fields; the earlier row wins a tie). |
| `blocked` | At least one check failed. |

A profile row holds these fields. The report compares each against the host:

| Field | Host value | Reference row `hipx` |
|---|---|---|
| `kernel` | Kernel release (`/proc/sys/kernel/osrelease`). | `7.3.0-rc7-perfopt+` |
| `iommu.gpu` | IOMMU group domain type of the GPU. | `identity` |
| `iommu.npu` | IOMMU group domain type of `accel0`. | `identity` |
| `amdxdna` | The out-of-tree driver version, or `in-tree`. | `in-tree` |
| `npu_fw` | The NPU firmware version (`fw_version`). | `1.1.2.65` |
| `model` | The NPU trunk index digest of the matched model (`model_trunk_index_sha256`: index-only, equal for a monolithic MQ4 and its split XTS base). | `098bd75fd3a6a5174d06033f2b3ec7febeb7777e7eba4696b64cecca5e3418b6` |
| `kernels` | The kernel set the daemon will run (below). | `pack:v0.4.3` (the v0.4.3 release pack, at its release commit). |

The kernel set is `pack:<tag>@<commit12>` when the install holds a kernel pack with a recorded `<arch>.manifest.json` for gfx1151 (written by `hipfire kernel-pack install`), or `jit:<commit12>` from the third word of `hipfire-daemon --version` when it does not; `commit12` is the first 12 hex digits of the commit. It is `unknown` when neither is available. A row lists the kernel sets validated on it: an entry `pack:<tag>` matches that release's pack at any commit (a release commit cannot name its own hash), and any other entry matches only itself. A kernel set that no row lists makes the host `compatible-unvalidated` by itself, even when every other field matches; JIT kernels (`jit:<commit12>`) are never the reference on `hipx`, which is pinned to the v0.4.3 release pack. The profile id and differences are in the report as `profile` and `profile_diff`.

**Confirmation and exit status.** After the checks the command prints what it is about to do, then asks `Run it now? [y/N]` on a terminal; only `y` or `yes` (any case) runs it. `--yes` skips the question. Without a terminal on stdin and without `--yes` it refuses, runs nothing and exits 2. Answering no exits 0 and still prints the report. Exit status: 0 ok, 1 blocked, a failed calibration or a failed soak, 2 refused. With `--json`, stdout carries only the report; the plan, the prompt, the progress lines, the soak's request lines and summary, and the `--apply` note go to stderr.

**Calibration.** The plan it prints before asking is:

```text
one daemon with npu_trunk=auto alternates GPU-only and auto prefills at 16K and 32K rows, 2 reps each, on prompts that never hit the prompt cache
then npu_trunk=on and npu_trunk=emulate daemons each prefill the same prompt and generate 16 tokens greedily; the two outputs must match
about 6–7 minutes on a Strix Halo (three model loads plus about 2 minutes of prefills)
uses the GPU and the NPU
changes no setting and writes no file
```

- **A/B.** One warm `npu_trunk=auto` daemon takes an unmeasured warm-up pair (GPU-only, then auto, 16384 rows), then two passes over 16384 and 32768 rows. Each size runs once on the GPU-only route and once on `auto` per pass, in alternating order (GPU first on the first pass, `auto` first on the second), so clock and thermal drift reaches both arms alike. The GPU-only arm is the diagnostic route ([below](npu/npu-trunk.md#diagnostic-gpu-only-route)); the daemon's `npu_trunk=auto` setting and its fault latch are not changed. Every prompt is synthetic text with a different leading line, so no request is served from the prompt cache; each generates one token, and the time is the time to first token. A request that was partly served from the cache, a GPU-only request that did not log the diagnostic route line, an `auto` request that did, or a prompt shorter than 9000 rows fails the calibration.
- **Per size**, from the two samples of each arm: `saved %` = (median off − median auto) / median off × 100, so a loss is negative; `spread %` is the larger of the two arms' (max − min) / median × 100. The verdict is `gain` when `saved %` is above the spread, `loss` when it is below minus the spread, and `no measurable gain` otherwise. The rows shown are the median of the prompt tokens the daemon reported for that size.
- **Also reported** when `auto` armed: the trunk route, the rate scales the estimator reached (`gpu_x`, `npu_x`, `frozen`, generation), the NPU wait per pair (median and max over the `auto` requests' pairs) and the number of guard trips. When `auto` did not arm (for example the memlock limit is short), the report quotes the daemon's reason and runs no A/B and no exactness check.
- **Exactness.** A daemon with `npu_trunk=on` and one with `npu_trunk=emulate` each prefill the same 9319-row prompt (one 8192-row chunk plus a 1127-row tail, the smallest prompt that forms an NPU pair) and generate 16 tokens greedily. `exact` means the two outputs are identical; `MISMATCH` names the index of the first token that differs; `not exercised` means `on` ran no NPU pair, so the comparison proves nothing. The daemons' own stderr is not echoed; only their `npu-trunk:` lines are used.

**Recommendation.** The report ends with one recommendation, computed from the calibration and the effective `npu_trunk`:

| Measured | `npu_trunk` is not `off` | `npu_trunk` is `off` |
|---|---|---|
| No calibration ran (blocked, declined, refused or failed) | none | none |
| Exactness `MISMATCH` | `set off`: `hipfire config set npu_trunk off`; the reason asks you to report a bug and attach the `--json` report | none (already off) |
| `auto` did not arm | none; the daemon's reason | none; the daemon's reason |
| `loss` at some size and `gain` at none | `set off`: `hipfire config set npu_trunk off` | none (already off) |
| `gain` at some size and `loss` at none | `keep auto` | `set auto`: `hipfire config set npu_trunk auto` |
| `gain` and `loss` at different sizes | `keep auto`; the guard bounds any loss | none; stays off |
| No size beyond the spread | `keep auto`; the guard bounds any loss | none; stays off |

**`--apply`.** Only the `set off` recommendation is applied, by the same write as `hipfire config set npu_trunk off`; the report then says `applied: npu_trunk is now off` (`applied: true` in JSON). In every other case it prints `--apply: no change` with the reason, including when the host is blocked, the question was declined or refused, or the calibration failed. It never sets `auto`; the `set auto` recommendation prints its command for you to run. A `MISMATCH` means the route's `on` and `emulate` outputs disagree, which is a bug; attach the `--json` report to the bug report.

**`--json`.** One object, schema `hipfire-doctor-npu-1`. It carries no hostname, user name, filesystem path or PCI address; check details and the model `tag` (set only for registry models) are written without them. Top-level fields:

| Field | Content |
|---|---|
| `schema` | `hipfire-doctor-npu-1`. |
| `hipfire` | `version` and `commit` (first 12 hex digits) of the CLI. |
| `tier` | `reference`, `compatible-unvalidated` or `blocked`. |
| `profile` | Id of the matched or nearest profile row; absent when blocked. |
| `profile_diff` | List of `{field, reference, found}`; empty for `reference` and `blocked`. |
| `host` | `gpu_arch`, `npu`, `kernel`, `iommu` (`gpu`, `npu`), `amdxdna`, `npu_fw`, `kernels`, `memlock` (`soft`, `hard`, `required`, `cap_ipc_lock`; `null` is unlimited) and `fclk` (`pinned`, `pinnable`). |
| `checks` | List of `{id, status, detail}` with `fix` on a failure or a warning; `status` is `pass`, `fail`, `warn` or `info`. |
| `model` | `tag`, `model_trunk_index_sha256` and `calibration` (`builtin` or `file`, the record that admitted it); absent when no model matched. |
| `npu_trunk` | The effective `fusions.npu_trunk`. |
| `calibration` | Absent when none ran. `armed`, `off_reason`, `trunk_route`, `sizes` (per size: `rows`, `off` and `auto` as `samples_ms` and `median_ms`, `saved_pct`, `spread_pct`, `verdict`), `rate_scales`, `npu_wait_per_pair`, `guard_trips`, `exactness` (`verdict`, `prompt_rows`, `tokens`, `first_difference`) and `duration_s`. |
| `calibration_skipped` | Why no calibration ran: blocked, `declined`, not confirmed (no terminal), soak mode or failed. |
| `recommendation` | `action` (`keep-auto`, `set-off`, `set-auto` or `none`), `reason`, and `command` when there is one. |
| `applied` | `true` only when `--apply` changed `npu_trunk`. |

**`--soak`.** `hipfire doctor --npu --soak` replaces the calibration with a long run of the route under load, to find stability faults that the minutes-long A/B cannot. It runs the same checks, prints its plan, asks the same question, and then makes no recommendation. One `npu_trunk=auto` daemon (`max_seq` 65536) takes a loop of cold prompts of 9000, 16000, 32000 and 57000 rows, one token each, with 20 s idle between bursts so the clocks ramp each time. The run ends after `--hours`, or earlier at the first of:

- a fatal kernel log line, checked between requests and every second while idle;
- a request that does not complete within 1200 s (the daemon is killed), a failed request, or the daemon exiting;
- `MemAvailable` below 4 GiB;
- the kernel-log follower exiting (the soak then cannot watch the kernel).

The kernel log comes from `journalctl -k -f`, or from `sudo -n dmesg -w -T --since now` when the journal is not readable; with neither the soak does not start (add yourself to the `systemd-journal` group, or allow passwordless `sudo dmesg`). A line is fatal when it is an `amdgpu` line with `page fault`, `timeout`, `GPU reset`, `device lost`, `MES failed`, `sq_intr`, `Fence fallback` or a ring reset; any other `amdxdna` line; or a `BUG:`, `Oops`, `hung_task`, `soft lockup`, `hard LOCKUP`, `Out of memory`, `watchdog: ` or RCU stall line. The two `amdxdna` messages `amdxdna_insert_pages: Failed to mmap dma buf -22` and `amdxdna_gem_obj_mmap: Failed insert pages, ret -22` (hipfire 0.4.2.x on Linux 7.1 or later, see the changelog) are counted as benign and never stop the run.

A heartbeat line (`<UTC time> load=<1-minute load> memavail_kb=<kB>`) is written and fsync'd every 10 s, so after a freeze the last line shows when the machine stopped. The soak writes only into `--out` (which must be a new or empty directory) and changes no setting. It exits 0 on PASS and 1 on FAIL.

| File in `--out` | Content |
|---|---|
| `heartbeat.log` | The heartbeat; its first line is the run start. |
| `kmsg.log` | The followed kernel log, as read. It still holds the host name and PCI addresses. |
| `requests.tsv` | One row per request: index, UTC time, requested rows, ok, wall seconds, prompt tokens. |
| `npu-trunk.log` | The daemon's `npu-trunk:` lines, rewritten after every request. |
| `summary.txt` | Verdict, route lines, per-size median wall time, kernel-line counts, heartbeat gaps over 30 s and, on FAIL, the kernel log tail (as read, like `kmsg.log`). |
| `summary.json` | Schema `hipfire-soak/1`: `verdict`, `reason`, `hours`, `requests`, `failed_requests`, `per_size`, `benign_count`, `fatal_count`, `heartbeat_gaps`, `fatal_line` and `host` (kernel, IOMMU, driver and firmware facts; no host name). `reason` and `fatal_line` have the host name and PCI addresses removed. |

After a freeze and reboot, `hipfire doctor --npu --soak-report DIR` reads the start time from the first line of `DIR/heartbeat.log`, collects the previous boot's kernel log since then (`journalctl -k -b -1`; run it as a user that can read that journal, which a volatile journal does not keep), and writes it to `DIR/prev-boot-kmsg.log` and a summary to `DIR/report.txt`. The summary, also printed, gives the last heartbeat, the last request, the kernel-line counts, up to 20 fatal lines and the last 40 kernel lines. It exits 0 and does not run the checks; it fails when `DIR` has no `heartbeat.log`.

### `hipfire calibrate`

`hipfire calibrate` measures the rates the engine's load-time selectors otherwise probe for, once, and stores them in `config.toml` under `[calibration]` next to a fingerprint of this host. It never prompts and makes no network calls; everything it reads and writes is local. With no flag it measures; the flags are:

| Flag | Purpose |
|---|---|
| `--show` | Print the stored record, whether it is fresh for each detected GPU (with the reason when not), the engine routes it selects and why, and, per device, the D2 CPU-split verdict the Flash-Next load would take from the stored `splice` rates. Opens no GPU and measures nothing. |
| `--check` | Print one line and exit 0 when a fresh record covers every detected GPU (or no GPU is detected and a record exists), exit 1 otherwise. The installers use it. |
| `--clear` | Remove the `[calibration]` table; other config keys are untouched. |
| `-j`, `--json` | Machine-readable output for the measuring run and for `--show`. |

`--show`, `--check` and `--clear` are mutually exclusive.

**What it detects.** From sysfs, proc and the KFD topology, without running a kernel: per GPU, the architecture, VRAM size, PCI address (BDF) and the PCIe link speed and width (the narrowest link on the path from the root port to the GPU); per host, the CPU model, logical and physical core counts, the ISA flags (AVX2, FMA, F16C, AVX-512 variants), total RAM, and whether each GPU is unified-memory (an APU) or discrete.

**What it measures.** The measuring run covers, in order:

1. CPU read bandwidth on the pinned splice pool (one worker per physical core), the pool the CPU-executed offload engine runs on.
2. Per HIP-visible GPU: the pinned host-to-device copy rate (the qwen4 host-link probe itself); the zero-copy read rate, a GPU kernel reading host-mapped memory over the link; and the `offload_split` engine-rate probe, a CPU GEMV against a GPU GEMV over the same host-mapped bytes, for each CPU-decodable dense format of the `hipfire offload-bench` matrix (`pool = "global"`). The D2 measured default is not recorded: it needs the loaded model's real CPU expert arm, so the first eligible not-yet-measured shape in a process is measured live at load and later matching loads in that process reuse the successful measurement (see below). A unified-memory GPU has no link to probe. A GPU the kernel lists but HIP cannot open in this run is recorded as unmeasured.

The whole run is bounded at 60 s. An optional probe that would start past the deadline is skipped and reported, and the record is written with what completed. The run ends by printing the summary, the routes the record selects and `calibrated in X.X s`. On hipx (7900 XTX on PCIe 4.0 x4, 16-core Strix Halo host) the whole run took 6.5 s.

**Refuses while a daemon runs.** The probes need an idle GPU, so the command exits with an error while a hipfire daemon is live (the same check `hipfire offload-bench` makes). Stop the daemon first.

**How the engine uses the record.** At a qwen4 model load on a discrete GPU, a fresh record replaces the load-time copy probe. The host-link class, and with it the G2 expert-staging default, the 8192-row narrow-link prefill chunk and the stage minimum rows, is taken from the stored copy rate, and the load log says `[calibrated <time>]`. A calibrated load still sets up the probe's copy path with one 4 KiB copy, so it sees the same free VRAM as a probed load and places the same expert layers. A stale record (a different GPU, PCIe link, RAM, CPU or probe version) or a missing one makes the engine probe at load exactly as before, and the log names the reason. Developer env overrides still win. Run `hipfire calibrate --show` to see the stored values, freshness and chosen routes. The stored schema and staleness rules are in [CONFIG.md](CONFIG.md#calibration-machine-written).

**D2 CPU split (Flash-Next, unset `memory.offload_exec`).** The measured D2 default does **not** read the record. It predicts the per-layer decode cost of D2 (the production CPU arm timed on the model's real host-resident experts, plus the current copy-transport handoff of 806.1 µs) against PCIe (bytes over a live zero-copy link rate, plus a launch floor), from 5 warmed samples of each engine, pessimistically for D2: the slowest CPU arm sample against the fastest link sample and the minimum floor. It arms D2 only when that worst case is still at least 15% faster than that best case, however wide the spread; a worst case that does not clear, a noisy inconclusive run and a failed or zero sample stay on PCIe (there is no spread threshold, the medians and spreads are logged). A stored CPU or link rate alone can never enable it: the record's engine rates are dense, best-of figures with no spread, so they say nothing about this model's arm or this host's noise. The first eligible, not-yet-measured shape (device, resident codecs, expert shape) in a process gets live evidence; later loads of the same shape in that process reuse the successful measurement (the log says so), and failures are not cached. `hipfire calibrate` accordingly no longer writes `pool = "splice"` engine entries, and an entry an earlier version wrote is ignored. The load logs one line with both decision costs, their medians, the inputs, the sample spreads and the probe's own wall cost: `qwen4 D2 default (N host split layers): ON|off (live samples (…): worst-case D2 … us vs best-case PCIe … us …)`. The explicit `memory.offload_exec`, `developer.qwen4_d2_default`, `HIPFIRE_MOE_CPU_EXPERTS=0`, a load with no host split layers and a unified-memory GPU decide before any probe. The probe is a few milliseconds of CPU and 5 short link reads, resolves after the load's free-VRAM sizing reads and frees its device buffers, so it does not change expert placement; it does not pause the PLE prewarm (contention can only turn D2 off). Rules: [CONFIG.md](CONFIG.md#partial-gpu-offload-memorygpu_layer_budget--memorymoe_expert_budget--memoryoffload_exec).

**D2 line in `--show`.** For each device whose entry is fresh, `--show` adds one line, `D2 default    decided at model load …`: the decision needs the loaded model's real CPU expert arm and live link samples, which a stored record cannot supply, so `--show` states that and never prints an `ON` or `off` verdict it cannot make (`off (unified memory: no link to beat)` on a unified-memory GPU, which never probes). It is produced by the same `d2_default::decide` the load uses, with no model arm to measure. A stale, missing or unmeasured device has no D2 line, only the freshness reason. With `--json` each device carries `d2_default`: `null` or an object `{status, text}`, where `status` is `load` (decided at model load) or `off` (unified memory). Fields and rules: [CONFIG.md](CONFIG.md#calibration-machine-written).

### `hipfire fit`

`hipfire fit <model> [--ctx N] [--kv-mode M] [--kv-k K] [--kv-v V] [--spec S] [--no-fetch] [-j|--json] [-v|--verbose]` answers, before you pull or load, whether a model runs on this machine and how. It reuses the engine's own admission and placement code (`hipfire_loader::admission` and, for Flash-Next, `hipfire_arch_qwen4::expert_residency`) with the same load parameters `hipfire run` would build from your config; it has no sizing formula of its own. It opens no GPU, never loads the HIP runtime and starts no daemon (the runtime preflight only asks `hipfire-daemon --version`). When no device selection is configured or inherited, fit names every KFD GPU in `ROCR_VISIBLE_DEVICES` in its own process, so admission's device count is the one the daemon's HIP would report without fit loading HIP.

`<model>` is an installed tag, alias, file name or path, or a registry tag that is not installed. Nothing is pulled.

| Flag | Purpose |
|---|---|
| `--ctx N` | The context to admit, `512..=1048576` (the range of `memory.max_seq`). Omitted: whatever the load would choose, the configured `memory.max_seq` when you set one, else automatic. |
| `--kv-mode M` | One-shot KV format for this load, the same values and validation as `hipfire run --kv-mode` (it overrides `memory.kv_cache`). The plan's `kv` and `capacity` lines are for that mode. |
| `--kv-k K`, `--kv-v V` | One-shot K and V axis formats, as `hipfire run --kv-k` / `--kv-v` (Qwen only). A combination the engine refuses (for example `--kv-mode fp8` with `--kv-k q8`) is the `admission` finding, with the engine's own refusal text. |
| `--spec S` (alias `--speculation`) | One-shot speculation selector, `off`, `auto`, `ngram`, `dflash`, `mtp` or `dspark`, as `hipfire run --spec` (it overrides `speculation.mode`). The `decode` line reports the route it selects. Fit never downloads the draft: a DFlash draft that is declared but not installed is the `draft` finding. |
| `--no-fetch` | Never touch the network. A registry tag that is not installed is then reported `UNDETERMINED` with `hipfire pull <tag>` as the fix. |
| `-j`, `--json` | Print one JSON object on stdout instead of text. |
| `-v`, `--verbose` | Also print the source-cited derivation lines (which engine terms a bound includes or omits, with file and line). `--json` always carries them as `details`. |

**What it reads.** The model file's header (installed file, or for a tag that is not installed a sparse replica fetched by HTTP Range under `<HIPFIRE_HOME>/cache/fit-headers/`: the header, metadata and tensor index, plus the few payload records admission itself reads; about 14 to 37 MB per model, and for Flash-Next on a GPU whose PLE rows are q4 (gfx1151) also the q4 table's embedded rank section, about 305 MiB, because admission verifies it. Replicas are reused from the cache without network), your resolved config and per-model overrides, the KFD topology and sysfs for each GPU (architecture, VRAM total and used, PCIe link), `/proc/meminfo`, the TTM pool and GTT limit, the `hipfire calibrate` record when it is fresh, and the bundled receipts table `registry/receipts-v1.json`. Free device memory is computed the way the engine's `device_mem_info` reports it, without HIP (discrete: VRAM total minus used; unified-memory APU: the smaller of the GTT headroom and available RAM less CMA plus the TTM pool). Once a process opens the GPU, HIP reports slightly less; the report says so in its notes.

**Overrides.** `--kv-mode`, `--kv-k`, `--kv-v` and `--spec` layer a one-shot override over your config exactly as the same flags on `hipfire run` do. Fit builds the load parameters with the one helper `run` uses for them (config, `--kv-mode`, the `--kv-k`/`--kv-v` axes, the effective speculation selector, and the DFlash draft that selector keeps), so admission sees the overridden values and every refusal is the engine's own text in an `admission` finding: an unsupported KV mode or axis pair on this architecture, `--spec mtp` on a profile with no MTP head, `--spec dflash` on an artifact with no declared draft. The plan header names the flags (``plan (what `hipfire run --kv-mode q8 --spec mtp` would load on GPU0)``) and the JSON echoes them as `overrides`. Fit never downloads weights: the header replica above is its only network use, and `--spec dflash` does not fetch the draft.

**Plan lines.** `basis` (where the numbers come from: `pre-load estimate from file headers + sysfs; the load re-measures free VRAM after weights, drafts and scratch (max_seq fit) and may admit less`; every line below is that estimate unless it says `decided at load`), `topology` (single GPU, or pipeline/expert parallel across the devices your config selects), `kv` (backend and mode), `context` (the number admitted and its bound: `user`, `config`, or automatic, with the trained window), `capacity` (tagged `(pre-load estimate: ...)`; for Qwen3.5/3.6 the admission's own projection of the largest context the card holds, with the free memory, weights, prefill scratch and rounding terms and the bytes per token. When the weights and prefill scratch leave no room at all it reads `weights + prefill scratch exceed free memory (2.7 GiB free of 31.9 GiB; 29.2 GiB held by other processes or the desktop)` instead of `up to 0 tokens`. A capacity above the model's trained window is shown as the window with the raw KV room noted (`up to 262144 tokens (model limit; KV room for 1.68M)`), in the per-GPU rows, this line, `best GPU`, the findings and the tier table alike; a capacity within the window reads as before. Fit reads sysfs counters only, so it cannot say which process holds the rest), `best GPU` (printed under the plan when more than one GPU is visible and another one projects a larger context than the one the plan loads on: `GPU2 (31.8 GiB free): up to 262144 tokens (model limit; KV room for 1.68M) — select it with HIP_VISIBLE_DEVICES=2 (or `hipfire config set hardware.devices <uuid>`)`; the same admission projection is evaluated on every GPU in `this machine`. When `hardware.devices` is set it alone selects, so only the config form is named. Ties go to the lowest index; the candidates are the GPUs the plan's own device selection leaves visible, so a `hardware.devices` pinned to one GPU leaves no alternative to name; `hipfire fit` has no GPU flag of its own and `hipfire run` has none either), `placement` (fully resident, or how many Flash-Next expert layers stay in VRAM and how many bytes are host-mapped), `decode` (the route the params select, with a second line saying why) and `routes` (the host-link route choices and why, when a calibration record is fresh). `decode` is one of `ar` (the reason names what kept it AR: `--spec off`, `speculation.mode = off (config)`, or MTP `auto` on an architecture that declares no measured AR floor, where `--spec mtp` opts in), `dflash (draft <file>)`, `dflash (draft <file> not installed)` (the load refuses), `dspark`, `mtp` (explicit: also over host-mapped experts), `mtp (auto)` (the architecture declares a measured AR floor; it stays AR when the head is absent or experts are host-mapped), `mtp / ar (auto; decided at load)` (the carrier is only known per rank) or `ngram`. The load installs the first armed of dspark, dflash, mtp, ngram; lower-priority routes that are also enabled are listed in the reason. With more than one visible GPU each GPU row in `this machine` also states the model's projected capacity there (`capacity: up to N tokens`, or `capacity: weights don't fit`).

**Findings.** Each is `FITS`, `UNDETERMINED` or `DOES NOT FIT`, with a `fix:` line when it fails, and when it is undetermined and there is something to change. Ids: `preflight` (the `hipfire doctor` runtime checks; Linux only), `model` (the file was found, or fetched), `vram` (a GPU is present and has room), `admission` (the engine's own admission accepted or refused the load, for example an unsupported KV backend on this platform or a refused `--kv-mode`/`--kv-k`/`--spec` combination), `draft` (a registry DFlash draft the final mode needs is declared but not installed: `DOES NOT FIT` for an installed model under `--spec dflash` or `speculation.dflash = on`, because the load refuses; `UNDETERMINED` for a model fit only read the header of, since `hipfire pull <tag>` fetches both; under `auto` it is a note and the plan runs without the draft; the size shown is the registry's), `weights`, `context` (the context fits the card. When the projection leaves room for fewer than 4096 tokens, or none, and the request does not fit, the finding is `DOES NOT FIT` with `the model does not fit in this GPU's currently free memory`, the free and total GiB and how much is `held by other processes or the desktop`, and the fix names freeing GPU memory, another GPU (the best one, when a visible GPU holds more) or a smaller tier; it never suggests lowering `memory.max_seq`, because no useful context fits. When at least 4096 tokens fit but fewer than requested, the fix is `lower --ctx to at most N` (`N` at least 512)), `experts` (Flash-Next expert placement: how many expert layers stay resident), `host_ram` and `gtt` (Flash-Next host-mapped experts against available RAM and the GTT cap).

**Verdict and exit status.** The verdict is the fold of every finding: `DOES NOT FIT` if any finding does not, else `UNDETERMINED` if any is, else `FITS`. Exit status 0 is `FITS`, 1 is `DOES NOT FIT` (including a blocked runtime preflight), 2 is `UNDETERMINED`.

**Decided at load.** Terms only the GPU can size at load are not guessed. Flash-Next's load reserve needs a live device for some terms, so fit uses a sound lower bound: the resident-expert count it reports is an upper bound and the host-mapped bytes a lower bound; a host gate that already fails on that bound is a definite `DOES NOT FIT`, and a placement that depends on the omitted terms is reported as decided at load. Multi-GPU (pipeline or expert) capacity is sized per rank at load (`UNDETERMINED`, with each device's free memory listed in the notes), and an architecture with no pre-load KV projection reports its context as sized at load. The report assumes the HIP virtual memory manager is available (ROCm on native Linux provides it); admission still applies its own platform refusals.

**Quant tiers.** When the tag is a family root with more than one tier, every tier is evaluated the same way and a table (tag, size, verdict, capacity or placement) is printed with `largest tier that fits at <ctx>: <tag>`. For Qwen3.5/3.6 tiers the row states the largest context the tier holds on the best visible GPU (`up to 262144 tokens on GPU2 (model limit; KV room for 1.62M)`, or `up to 100000 tokens on GPU2` when the room is within the window; `on GPUn` only when more than one GPU is visible), or `weights don't fit` when its weights and prefill scratch exceed the free memory there, and the verdict and the `largest tier that fits` line are that best GPU's (the row carries `best_gpu` in JSON). A tier's verdict is the model's own fit; machine-level blockers (the runtime preflight, device visibility) are reported once above and apply to every tier. `hipfire pull <family>` still defaults to the root tier. The requested model's resolved process configuration (the settings that apply to the whole process, as `hipfire run` installs them) is installed once, before the machine facts and admission are computed, and every tier is evaluated under that one process policy. A tier whose own resolved process settings differ from it is still given its verdict, but its summary says `per-model settings differ: run hipfire fit <tier> for its exact answer`.

**Expected speed.** Receipts for the same tag, GPU architecture and topology, and whose placement is the plan's placement (`resident` or `host-experts`; a receipt whose source does not state its placement never matches), each with its tokens, route, date, class (`fast`, `interactive`, `usable`, `slow`, from fixed tok/s bands) and source. Rows marked historical (an older engine, or an artifact the registry no longer serves, such as the Flash-Next numbers measured on the retired monolithic file) stay in the table but are never applied. A tag with no matching receipt prints `no published receipt for <tag> on <arch> with this placement; measure with hipfire bench <tag>`. Receipts are never carried across tag, architecture, topology or a placement the source states and the plan does not have.

**JSON.** Schema `hipfire-fit-1`: `schema`, `hipfire` (`version`, `commit`), `input` (the model name), `overrides` (`kv_mode`, `kv_k`, `kv_v`, `spec`; `null` when the flag was not given), `model` (`tag`, `file`, `installed`, `replica`, `size_bytes`, `arch_id`, `quant`), `machine` (devices, RAM, GTT, link facts; each device also carries `capacity`, the raw tokens of KV room the model projects on it, `0` when its weights do not fit, and `capacity_shown`, that capped at the trained window), `plan` (`basis` (`pre-load`), `device_index`, `topology`, `kv_backend`, `kv_backend_reason`, `kv_mode`, `max_seq`, `bound`, `model_ctx`, `projection`, `best_device` (`index`, `free_bytes`, `capacity`, `capacity_shown`: the visible GPU with the largest projected capacity, present when more than one GPU is visible), `placement`, `decode_route` (a string), `decode_reason`, `routes`), `findings` (`id`, `verdict`, `detail`, `fix`), `verdict`, `tiers` (`tag`, `size_bytes`, `verdict`, `summary`, `best_gpu`), `receipts`, `notes`, `details`. Like the doctor report it carries no hostname, user name, home path or PCI address; the model is named by file name only.

## Where files live

| Path | Role |
|---|---|
| `~/.hipfire/bin/` | native `hipfire`, `daemon`, optional `hipfire-tui`, tools |
| `~/.hipfire/bin/kernels/compiled/<arch>/` | Precompiled / JIT kernels |
| `~/.hipfire/models/` | Weight files and co-located sidecars |
| `~/.hipfire/config.toml` | Sparse typed global config (`config.json` is migration input) |
| `~/.hipfire/models.toml` | Local aliases, paths, registry identities, and per-model overrides |
| `~/.hipfire/models.json` | Legacy catalog migration input |
| `~/.hipfire/per_model_config.json` | Legacy overlay; folded into the catalog on refresh |
| `~/.hipfire/registry.cache.json` | Dynamic registry cache |
| `~/.hipfire/serve.pid` / `serve.log` | Detached serve pid record and log |
| `~/.hipfire/hf-cache/` | HF download cache used by quantize |
| `~/.hipfire/src/` | Source tree used by `update` / some installs |

Override catalog path with `HIPFIRE_MODELS_CATALOG_PATH` if needed.

## Dynamic registry

1. Bundled `registry/v1.json` (embedded in the binary).
2. Optional network fetch of `registry/v1.json` → 24h cache.
3. Invalid remote payloads are rejected wholesale.
`hipfire diag` reports which source the process used. Entry metadata may include size, `arch_id` ([architecture-ids.md](architecture-ids.md)), quant, and hashes — registry presence is not a runtime admission ([INDEX.md](INDEX.md), [admissions.yml](admissions.yml)).

## Useful environment overrides

Single-invocation knobs (non-exhaustive; full list in [env-vars.md](env-vars.md)):

| Variable | Effect |
|---|---|
| `HIPFIRE_LOCAL=1` | `run` skips HTTP serve and spawns a local daemon (also forced by load-time overrides such as `--kv-mode` or `--image`). |
| `HIPFIRE_HOME=...` | Override the state/config root (default `~/.hipfire`). |
| `HIPFIRE_MODELS_DIR=...` | Override model discovery, pull, list, and TUI model paths. |
| `HIPFIRE_KV_MODE=...` | Override KV layout preset (`auto` → Qwen fp8 on native-eligible exact gfx1201, else q8/q8; non-Qwen unchanged). |
| `HIPFIRE_KV_BACKEND=...` | Not a second default plane for ordinary loads; the stable **warning token** `HIPFIRE_KV_BACKEND=legacy` appears on stderr when legacy storage is selected. Config/CLI use `legacy`\|`vmm` only. |
| `HIPFIRE_SPECULATION=...` | Speculation selector from the environment: overrides per-model and global config; a CLI `--spec` still wins. |
| `HIPFIRE_DFLASH_DRAFT=...` | Explicit draft path. |
| `HIPFIRE_VISION_SIDECAR=...` | Explicit vision-tower sidecar path; empty opts out. Skipped while `vision_mode=off`. |
| `HIPFIRE_VISION_MODE=...` | Tower sidecar gate: `auto` (default) / `resident` / `on` / `off`. |
| `HIPFIRE_DFLASH_MODE=...` | Daemon-side mode (CLI default config is still `off`). |
| `HIPFIRE_NO_REGISTRY_FETCH=1` | Pin bundled registry. |
| `HIPFIRE_REGISTRY_URL=...` | Alternate registry URL. |
| `HIPFIRE_NORMALIZE_PROMPT=0` | Opt out of `\n{3,}` collapse (config `prompt_normalize` default on). |
| `HIPFIRE_DEBUG=1` | Full stacks on CLI failure. |
| `--no-color` (chat) | Accepted by `hipfire chat` for compatibility; the line client never emits ANSI color. |

## Errors and next steps

| Problem | Action |
|---|---|
| Unknown command | `hipfire help` |
| Unknown config key | `hipfire config set` prints valid keys; see [CONFIG.md](CONFIG.md) |
| Model missing (`model not found: ... expected <models>/<file>`) | `hipfire pull <tag>`; a local-only entry: place the named file in the models dir; a path: check it or use a tag from `hipfire list -r`; an unknown tag: the did-you-mean list, then `hipfire pull <closest>` |
| Daemon missing (`daemon binary not found`) | `cargo build --release -p hipfire-daemon`, install it at `~/.hipfire/bin/daemon`, or set `HIPFIRE_DAEMON_BIN` |
| Draft missing (`DFlash draft not found`) | Registry tag: `hipfire pull <tag>` and drop `--model-draft`; otherwise pass an existing `-md <path>` or run with `--spec off` |
| Vision sidecar missing (`vision sidecar not found`) | Pass an existing `--vision <path>`, `hipfire pull <tag>` for the registry tower, or turn vision off: `HIPFIRE_VISION_MODE=off` / `hipfire config set vision.mode off` |
| Serve already running | `hipfire stop` or `restart` |
| Stale pid / port busy (`port N is in use by another process`) | `hipfire stop --force`; inspect `serve.log`; or pick another port: `hipfire serve [model] 127.0.0.1 <port>` (default: `hipfire config set serve.port <port>`) |
| `update` on non-Linux | Re-run `install.ps1 -Branch/-Tag/-Commit` or use a source build |
| `update` from a detached pin | Choose the next target explicitly, e.g. `hipfire update @beta` |
| Need HTTP API detail | [SERVE.md](SERVE.md) |
| Need chat UX | [CHAT.md](CHAT.md) |
| Need onboarding | [GETTING_STARTED.md](GETTING_STARTED.md) |
