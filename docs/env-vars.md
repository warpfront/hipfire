# Environment variables

**Owner:** environment-variable inventory (`docs/INDEX.md`).
**Machine sources:** `HIPFIRE_*` token scan of Rust, Python, and shell sources under the repo (excluding `target/`, `.git/`, etc.).
**Config schema owner:** [`CONFIG.md`](CONFIG.md) (`crates/hipfire-config/src/lib.rs` + `crates/hipfire-runtime/src/config.rs`).
**Last checked:** 2026-07-31.

For persistent user configuration, prefer `hipfire config set ...` and
`~/.hipfire/config.toml`. Schema-declared environment variables are retained as
one-shot/compatibility overrides. The native CLI resolves stable typed fields
and experimental `[developer]` values into one versioned daemon snapshot; see
[`CONFIG.md`](CONFIG.md#process-wide-hardware-and-kernel-policy).

This page has two layers:

1. **Manual (normative for operators)** — precedence, product-facing knobs, LFM/Redline branch scope, and pointers. Defaults here are grounded in CLI/runtime source cited inline.
2. **Generated inventory** — exhaustive name → example source path table from the scan. Presence in the inventory means the token appears in source; it does **not** mean the knob is supported, stable, or admitted.

Detailed procedures belong in domain owners ([`CONFIG.md`](CONFIG.md), [`SERVE.md`](SERVE.md), [`multi-gpu.md`](multi-gpu.md), [`REDLINE.md`](REDLINE.md), [`VALIDATION.md`](VALIDATION.md), methodology pages). This file does not host validation matrices or benchmark floors.

Coverage check for top-level docs: `scripts/check-env-docs.py` (every
`HIPFIRE_*` in `AGENTS.md` / `README.md` / `CONTRIBUTING.md` must appear
somewhere in this file).

---

## Manual — precedence

1. **Built-in typed defaults** in `hipfire-config` / `RuntimeConfig`.
2. **Registry card** `recommended_settings` from `registry/v1.json`.
3. **Global** `~/.hipfire/config.toml`.
4. **Per-model overlay** from `~/.hipfire/models.toml`.
5. **Compatibility environment layer** for schema-declared `HIPFIRE_*` names.
6. **One-shot CLI flags** for the current command.

Legacy `config.json`, `models.json`, and `per_model_config.json` are read only
as migration inputs when their TOML successors do not exist. Invoking the
daemon binary directly still resolves local TOML plus compatibility variables
before GPU initialization, but does not add registry/per-model policy that was
not supplied by a native client.

Ladder for speculation selector (product path): `HIPFIRE_SPECULATION` / `--spec` > per-model > global > default `auto`.

Effective sampling send order (run/serve): CLI flags > per-model > registry `recommended_settings` > daemon/HFQ/arch fallback (field omitted when only a bare global default would apply). Global-only sampling edits are not sent; registry `sampling` blocks are inert metadata — see [`CONFIG.md`](CONFIG.md) / [`MODELS.md`](MODELS.md). **Chat** keeps a global session snapshot exception ([CHAT.md](CHAT.md)).

### TOML replacement for experimental variables

Production engine code no longer reads the ambient environment for kernel,
architecture, replay, or diagnostic behavior. Stable controls use typed schema
keys. Residual experiments use a flat global-only namespace:

```toml
[developer]
gfx1151_gate_up_wave64 = true
deepseek4_attn = "twin"
```

The generic compatibility rule is `HIPFIRE_FOO=value` →
`developer.foo = "value"` at startup. This preserves existing harnesses while
making TOML the persistent and navigable surface. `[developer]` values are
process-scoped, excluded from registry and per-model policy, and frozen before
GPU initialization. The only direct production environment reads left are
bootstrap locations needed to discover config or launch the process (for
example `HIPFIRE_HOME`, binary paths, registry URL, kernel cache, spill path,
and quant diagnostic output).

---

## Manual — product / operator knobs

Values and defaults below match `hipfire-config`, the native CLI, and/or `RuntimeConfig` as of the check date. Full key tables: [`CONFIG.md`](CONFIG.md).

### Paths and process

| Variable | Role | Notes |
|---|---|---|
| `HIPFIRE_HOME` | Native state/config root | Default `~/.hipfire`; used by config, registry cache, CLI, and TUI. |
| `HIPFIRE_DIR` | Script/harness compatibility | Not read by the native CLI; prefer `HIPFIRE_HOME`. |
| `HIPFIRE_MODELS_DIR` | Model discovery/lifecycle root | Overrides list/pull/remove/pre-warm and TUI model paths. |
| `HIPFIRE_MODEL` | Serve/run model tag or path | Also `default_model` config. |
| `HIPFIRE_DAEMON_BIN` | Daemon binary override | |
| `HIPFIRE_LOCK_DIR` | Shared per-GPU lock directory | Absolute writable path; daemon and `gpu-lock.sh` must use the same setting to contend. Default on Linux/WSL `/run/lock/hipfire` if writable, otherwise `/tmp/hipfire-locks`; on native Windows `%ProgramData%\hipfire\locks`, otherwise `%TEMP%\hipfire-locks` (`LockFileEx`, same file names and holder-PID reporting). Different directories do not see each other's locks. Files are `gpu-GPU-<uuid>.lock`, or `gpu-pci-<dddd:bb:dd.f>.lock` for cards without a UUID. |
| `HIPFIRE_TUI_BIN` | TUI binary | |
| `HIPFIRE_ROCM_PATH` | hipfire-specific ROCm SDK root override | Highest priority (`HIPFIRE_ROCM_PATH` > `ROCM_PATH` > `HIP_PATH`). Must provide the runtime, headers, and `hipcc`. Authoritative: no fallback to another install or bare soname. |
| `ROCM_PATH` / `HIP_PATH` | ROCm/HIP compatibility root overrides | Used only when `HIPFIRE_ROCM_PATH` is unset (`ROCM_PATH` before `HIP_PATH`). `HIP_PATH=<root>/hip` normalizes to `<root>`. Multiple equally eligible roots without an override are refused — set `HIPFIRE_ROCM_PATH`. |
| `HSA_USERPTR_FOR_PAGED_MEM` | ROCm (libhsakmt) host-allocation backing | Linux, in a process that sets `HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS` (host-mapped Qwen4 experts), or a daemon started for a Qwen4 model whose cards are all discrete GPUs (`hipfire run`, `bench`, and `serve`'s pre-warm model; the arch of the cards comes from the KFD topology): hipfire sets `0` before the HIP runtime loads, unless it is already set, and logs it on a `[hip-bridge] … host memory out of reclaim` line. Other processes keep ROCm's default. With `0`, `hipHostMalloc` memory (the host-mapped routed experts, clr's staging buffers) is GTT, outside the kernel's reclaim. Under ROCm's default (non-zero) it is pageable userptr: when reclaim takes those pages, KFD evicts the process's GPU queues, and host-memory pressure can stall the GPU with one core spinning in `hipMemcpy` (ROCm/rocm-systems#12528). GTT is capped by TTM's `pages_limit` (`/sys/module/ttm/parameters/pages_limit`, in pages; half of RAM by default). A Qwen4 load whose host-mapped experts do not fit beside the GTT that amdgpu devices already hold is refused before it allocates. When the process exits, amdgpu keeps its freed GTT pages in TTM's page pool (up to `/sys/module/ttm/parameters/page_pool_size`, half of RAM by default). `MemAvailable` does not count the pool, but the next GTT allocation takes pages from it and memory pressure shrinks it, so the Qwen4 host-RAM check adds an estimate of it (see `HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS`). Operators can return the pool to the kernel at once with `echo 2 \| sudo tee /proc/sys/vm/drop_caches`; `sudo cat /sys/kernel/debug/ttm/page_pool` shows its size (last line, in pages). Opt out with `HSA_USERPTR_FOR_PAGED_MEM=1`, which brings the stall exposure back. |
| `GPU_PINNED_MIN_XFER_SIZE` | ROCm (clr) pageable-copy threshold, MB | Linux, in the same processes as `HSA_USERPTR_FOR_PAGED_MEM` above: hipfire sets `100000` before the HIP runtime loads, unless it is already set, so every copy to or from pageable host memory goes through clr's pinned staging buffer. That covers weight uploads from the mapped model file and Qwen4's per-forward PLE rows. At or above the threshold, clr pins a copy's pageable source in place as a userptr, which carries the same queue-eviction stall under host-memory pressure. Other processes keep clr's default: set globally, the two switches together slowed H2's gfx1201 weight upload from 1.0 s to 1.2 s. Opt out by setting it explicitly: copies at or above the value you set are pinned in place again. |
| `HIPFIRE_LOCAL=1` | Skip attaching to running serve | One-shot local daemon. |
| `HIPFIRE_REGISTRY_URL` | Dynamic registry fetch URL | |
| `HIPFIRE_NO_REGISTRY_FETCH=1` | Pin bundled registry | |

### KV / attention / prompt

| Variable | Default / sense | Source |
|---|---|---|
| `HIPFIRE_KV_MODE` | From config; **`auto` → registry non-q8 `default_kv_mode`, else Qwen-family native `fp8` on exact gfx1201 when native-eligible (H24/Hkv4/D256, single GPU, no adaptive/CASK), else Qwen Q8/Q8** (gfx1100, gfx1151 and every other load; explicit `--kv-mode q8` still honored). `fwht2`/`fwht3`/`fwht4` are optional headroom modes that `auto` does not select (only the kill switch below does); `asymN`/`turboN` are legacy spellings (Qwen aliases of `fwhtN`). Qwen4 / Flash-Next (arch 16) accepts only `auto`, `bf16`, `fp8` and `q8`: `auto` is fp8 QSA K/V on exact gfx1201 and `bf16` elsewhere; `q8` (opt-in, gfx11/gfx12) stores the QSA K/V as Q8_0 blocks of 32 ([`CONFIG.md`](CONFIG.md#mode-k-and-v)). Non-Qwen families keep their own defaults (Maple BF16, Gemma layered, DeepSeek compressor). | CLI / pair resolver; **not** a legacy hard-coded fwht-per-arch table |
| `HIPFIRE_KV_ADAPTIVE` | off unless set / param | Loader/CLI |
| `HIPFIRE_KV_PHYSICAL_CAP` | optional physical slot cap | Daemon |
| `HIPFIRE_KV_V` | **developer-only** V-axis override (e.g. `lloyd2`/`lloyd3`/`lloyd4`); **lower precedence** than an authored `--kv-v` or `memory.kv_v` | Qwen carrier (`developer_var`); not a second user config plane — prefer CLI/TOML |
| `HIPFIRE_QWEN_KV_DEFAULT_Q8` | default **ON** (implicit Qwen Q8/Q8 wherever `auto` does not pick native fp8). **`=0`** is the emergency kill switch: restores the prior *implicit* HFQ/PaRo defaults on every load that does not get native fp8, gfx1201 included (HFQ `auto`/unset and PaRo `"auto"` → FWHT3/Q8; PaRo raw unset stays Q8). Does **not** override authored `--kv-mode`/`--kv-k`/`--kv-v` or `memory.kv_*`, native gfx1201 fp8, or non-Qwen families. | Loader admission (`qwen_default_q8_enabled`); sampled once per load |
| `HIPFIRE_ATTN_FLASH` | from `flash_mode` (`auto`/`always`/`never`) | CLI → daemon |
| `HIPFIRE_NORMALIZE_PROMPT` | on unless `0`/`false`/`off`/`no` | `RuntimeConfig` |
| `HIPFIRE_PROMPT_TOKEN_HEAT=1` | dump BPE heat | RuntimeConfig |
| `HIPFIRE_PROMPT_HEAT_JSON=1` | JSON heat | RuntimeConfig |
| `HIPFIRE_PROMPT_HEAT_LIMIT` | default **64** | RuntimeConfig |
| `HIPFIRE_LM_HEAD_F16` | default **`auto`** | Qwen loader compatibility alias for `kernel.lm_head_f16` |

### Speculation / DFlash / MTP / n-gram / DSpark

| Variable | Default / sense | Notes |
|---|---|---|
| `HIPFIRE_SPECULATION` | `off`/`auto`/`ngram`/`dflash`/`mtp`/`dspark` | Canonical selector |
| `HIPFIRE_DFLASH_DRAFT` | explicit draft path (overrides the registry sidecar); empty opts out | Legacy `developer.dflash_draft` read; still appears in legacy gate scripts. |
| `HIPFIRE_VISION_SIDECAR` | explicit vision-tower sidecar path (overrides `params.vision`); empty opts out | Daemon load; validated at admission (arch 5\|6 + tower tensor), loaded by the Qwen35 carrier |
| `HIPFIRE_DFLASH_WINDOW` | **unset → the draft artifact's declared `sliding_window`** (DFlash2 drafts: all layers sliding; DFlash drafts declaring n−1 sliding + last full: SWA on layers 0..n−2, last layer full). `<rows>` overrides (warns on mismatch); **`0` forces legacy contiguous** | Windowed draft context: draft VRAM pins at W and past-W requests degrade τ instead of falling back to AR. Legacy contiguous applies only when the draft declares no window, `HIPFIRE_DFLASH_WINDOW=0`, or CASK eviction is active. **`=0` (explicit legacy contiguous) is deprecated since 0.4.0, removal in 0.5.0** and warns |
| `HIPFIRE_DFLASH_CTX_CAP` | **8192**; `0` = uncapped | **Deprecated since 0.4.0, removal in 0.5.0** (legacy contiguous DFlash; setting it warns). **Legacy contiguous mode only** (no draft window, `HIPFIRE_DFLASH_WINDOW=0`, or CASK eviction): caps draft-side context storage; over-cap requests fall back to AR. Ignored when the draft runs windowed |
| `HIPFIRE_DFLASH_MODE` | RuntimeConfig default **`off`** | Distinct from config `dflash_mode` apply path — product CLI also uses load params |
| `HIPFIRE_DFLASH_NGRAM_BLOCK` | set/clear from config | |
| `HIPFIRE_DFLASH_ADAPTIVE_B` | **kill switch**; `=0` forces the fixed full block, any other value or unset defers to the setting | Read through the `[developer]` namespace (lifecycle `developer`; it is **not** a schema alias — the schema key is `speculation.dflash_adaptive_b`, stable, default `false` = fixed block, listed in the [`CONFIG.md`](CONFIG.md#lifecycle-status) table with no env column). `=0` overrides an enabled `dflash_adaptive_b` load param/setting; it cannot turn adaptive on. Adaptive is also auto-suppressed on retained-PM4 verify loads (replay needs the fixed B=16 shape). |
| `HIPFIRE_DFLASH_CKPT_RESUME` / `HIPFIRE_CACHE_CKPT_*` | checkpointing | Qwen DFlash and MTP divergent-render resume |
| `HIPFIRE_SPEC_WINDOW_ROLLBACK` | on unless `0` | Enables retained pre-window repair for strict-prefix speculative terminals; `0` keeps the conservative reset path. |
| `HIPFIRE_DFLASH_VERIFY_PM4` | **unset / off**; `1` opts in | Retained-PM4 route for the fixed B=16 DFlash2 chain target-verify forward. Admitted only on exact gfx1201, single GPU, dense recurrent Qwen3.5-family target, Q8 KV + Q8 DeltaNet state, DFlash2 selector + dynamic-conv draft, `target_layer_ids == [5,19,33,47,61]`, no DDTree. Every other configuration reports a specific `disabled` reason and runs the unchanged HIP/HipGraph path. |
| `HIPFIRE_DFLASH_LEGACY_PREFILL` | **unset / off**; `1` opts out | Qwen35 DFlash target prompt prefill (cold seed, prompt-cache suffix, forced tokens). Default: the seed plans its chunks exactly as AR's ordinary prefill does (`ordinary_prefill_chunk_limit`, then `ordinary_serve_prefill_chunk_len`) and each chunk takes AR's route (widened chunk with `commit_stride`, GDN chunk scan, whole-chunk FA2), so after the prompt the target's KV, DeltaNet state and last-token logits are byte-identical to AR's prefill. Chunks wider than the 256-row ring staging write their hidden rows straight to the ring; this is eager only, and verify and captured forwards keep staging. `1` restores the previous route: 256-row chunks through staging, outside the widened chunk and the chunk scan. Both routes reuse one prefill scratch across chunks. |
| `HIPFIRE_DN_SNAPSHOT_BULK_OFF` | **unset**; `1` opts out | DeltaNet snapshot save/restore as one descriptor-driven byte-copy launch each (`dflash_state_bulk_copy_gfx1100`, pure byte copy) instead of one `hipMemcpy` per tensor. Default on exact gfx1100 and exact gfx1201 (railgun E0 port); DFlash and MTP share the snapshot. `1` restores the memcpy loops. |
| `HIPFIRE_GDN_REPLAY_ML_OFF` | **unset**; `1` opts out | Exact gfx1201: the DFlash/MTP GDN tape replay runs every LinearAttention layer in two launches (`dflash_gdn_replay_pre_ml` + `gated_delta_net_q8_fast_ml`, byte-identical) instead of 4 launches per layer. Q8 state with the fast (single-end requant) kernel only; declines while a Redline recording is open. `1` restores the per-layer launches. |
| `HIPFIRE_SELECT_REGRID_OFF` | **unset**; `1` opts out | Exact gfx1201: `topk_values_batched_f32` (DFlash2 selector) and `argmax_f32_batched` run 1024-thread rows with wave32 reductions (`select_regrid.hip`, byte-identical including ties; tie rows rerun the shipping top-K body). `1` restores the shipping one-block-per-row kernels. |
| `HIPFIRE_VERIFY_ATTN` | **on** on exact gfx1201, gfx1100 and gfx1151 (`kernel.verify_attn`); `0` opts out | VerifyAttn: speculative-verify attention (DFlash / MTP / n-gram verify blocks of 1..=32 rows, non-tree, H2 head_dim 256 / GQA 6) runs one K/V scan per kv head for 8 rows x 6 q heads over a fixed, context-independent split grid (graph-capture-stable) instead of `attention_flash_{fp8_e4m3,q8_0}_tile_batched`'s per-(row, head, tile) grid. On gfx1100 (Q8 KV) it also replaces the eager multi-row `attention_flash_q8_0_rows{8,4}_d8` above `HIPFIRE_FA_PERTOKEN_MIN_CTX` with a twin of its online-softmax arithmetic. On gfx1151 (Q8 KV) it replaces the single-slot WMMA flash prefill (`attention_q8_0_flash_prefill_wmma`, one wave per (q head, 16 rows) walking the whole context) with a context-parallel S = Q.K^T launch (f16 S tiles in the flash partials, one K dequant per kv head) and a per-16-dim-chunk online-softmax + P.V walk (one V dequant per kv head), also over context-independent grids. Byte-identical output (and gfx1201/gfx1100 partials); `0` restores the tile_batched / multi-row + reduce pairs and the WMMA prefill. |
| `HIPFIRE_DN_SNAPSHOT_FLIP` | **on** (exact gfx1201); `0` opts out | Railgun D8, exact gfx1201 DFlash chain spec: the rollback drops the DeltaNet restore copy. On a full accept with the EF residual on and a HIP/HipGraph verify, the verify-advanced state is kept; on any shorter accept the GDN tape replay reads the pre-verify state from the snapshot and writes the live buffers (`dflash_gdn_replay_pre_ml_from` + `gated_delta_net_q8_fast_ml_from`). Byte-identical to restore + replay; the snapshot stays the pre-window state, so terminal repair is unchanged. Needs the two-launch replay (`HIPFIRE_GDN_REPLAY_ML_OFF` unset); the no-tape path and every other decline restore as before. Introduced opt-in in 0.4.0, default on since 0.4.1; `0` restores the copy. |
| `HIPFIRE_DRAFT_MAX` | routes to active mech window | CLI |
| `HIPFIRE_DRAFT_F16` | on unless `0` | RuntimeConfig |
| `HIPFIRE_NGRAM_DRAFT` | `1` forces n-gram | Loader always honors |
| `HIPFIRE_NGRAM_DRAFT_K` / `HIPFIRE_NGRAM_MIN_COUNT` | n-gram params | |
| `HIPFIRE_NGRAM_LOOP_THRESHOLD` | default **0 (off)** | RuntimeConfig |
| `HIPFIRE_NGRAM_WINDOW` | default 256 | RuntimeConfig |
| `HIPFIRE_MTP_MODE` / `HIPFIRE_MTP_K` | auto / 3 | Config + RuntimeConfig |
| `HIPFIRE_MTP_NGRAM` | off | `speculation.mtp_ngram` (`on`/`off`/`auto`, also `1`/`0`; `auto` = off): MTP + ngram-mod for greedy, thinking-off requests |
| `HIPFIRE_MTP_SAMPLED` | **on**; `0` opts out | `speculation.mtp_sampled` (stable; introduced as an experimental opt-in and flipped to default on in 0.4.1): Qwen4 (Flash-Next) native MTP also serves sampled (temperature > 0) requests by speculative rejection sampling — a draft drawn from `q` (the request's truncation applied to the draft head's 8 exactly re-scored candidates) is accepted with probability `min(1, p/q)`, a rejection emits a draw from `(p − q)+`, an all-accepted window emits its bonus from `p`, where `p` is exactly the AR sampler's distribution (`llama::sample_top_k_p`: temperature, top_k with absent = 20 and 0 = 64 candidates, min_p, top_p). Lossless against AR in distribution, not byte-identical to AR for a seed; one seed replays the same text. Greedy requests are unchanged. `0`: sampled requests run AR. Non-neutral repeat/presence/frequency penalties still run AR. |
| `HIPFIRE_MTP_OWN_PREFILL` | **unset / off**; `1` opts out | Qwen35 MTP prompt fill. Default: the trunk prefills the prompt through AR's own route (same outer chunks, widened chunk, GDN chunk scan, standard dispatch) and hands its hidden rows to the MTP head, so the prompt's KV, DeltaNet state and first-token logits match AR's. `1` restores MTP's previous route: 512-row trunk chunks captured as a speculative verify (sequential GDN recurrence; on Q8 KV, no query16 flash prefill). Resolved once per MTP load (`hipfire_config::mtp_own_prefill`); serve.log prints `qwen35 MTP prompt fill route: …`. |
| `HIPFIRE_QWEN35_MTP` / `HIPFIRE_QWEN35_MTP_K` | Qwen35 MTP opt-in gate | Loader — separate from DeepSeek MTP |
| `HIPFIRE_DEEPSEEK4_SPEC_DECODE` / `HIPFIRE_DEEPSEEK4_SPEC_K` | DeepSeek MTP legacy | |
| `HIPFIRE_DEEPSEEK4_DSPARK` / `HIPFIRE_DEEPSEEK4_DSPARK_CONF_THRESHOLD` | DSpark | |
| `HIPFIRE_QWEN3_DSPARK_CONF_THRESHOLD` / `HIPFIRE_QWEN35_DSPARK_CONF_THRESHOLD` | per-arch conf | |
| `HIPFIRE_DDTREE_BUDGET` / `HIPFIRE_DDTREE_TOPK` | tree draft | Runtime defaults 256/8 if env-only; CLI config defaults 0/4 |
| `HIPFIRE_DDTREE_*` | research/diag family | See inventory; not product defaults |

### Qwen4 / Qwen3.8 Flash-Next (arch 16)

Read only by the Qwen4 carrier and its kernels; no other model reads them.

| Variable | Default / sense | Notes |
|---|---|---|
| `HIPFIRE_QWEN4_F16_WMMA` | on unless `0` | Prefill F16 WMMA arms (grouped MoE gate/up and down, BF16 dense projections through an F16 shadow, HC read, full-window QSA, chunked GDN) from 512 rows. Not bit-exact; admitted by KLD against the BF16 source. `0` keeps the bit-exact F32 arms. |
| `HIPFIRE_QWEN4_F16_WMMA_GFX1201` | gfx1201: on unless `0` | Exact gfx1201, eager-only: runs the `HIPFIRE_QWEN4_F16_WMMA` BF16-projection and HC-read arms on gfx12 WMMA kernels. The M=1 shared selector stays on exact multirow (U1a owns fusion). X rounds F32→F16 RNE; BF16 weights convert to F16 (exact for in-range normals); WMMA F32 16-element substeps and fixed-order split-K reduction change summation order, so it is not bit-exact against the multirow/SIMT arms. Admitted by Flash-Next KLD against the BF16 teacher on an R9700 (WikiText-2 −0.0033 [−0.0082, +0.0006], code +0.0047 [−0.0030, +0.0131] vs `=0`); pp8192 2,169 → 3,598 tok/s with `HIPFIRE_QWEN4_HC_FUSE=1` and `HIPFIRE_QWEN4_HC_UP_TILE`. `0` keeps gfx1201 on the multirow/SIMT arms; this is the kill switch. Recording/capture and other architectures retain their existing routes. Empty split-K work launches nothing. |
| `HIPFIRE_QWEN4_PROJ_REGIONS` | **opt-in**; off unless `1` | F16 WMMA projection route only: virtually concatenate router/shared gate/up/selector rows of K=2560 into one LDS launch; gfx11 HC-down (320×10240) can use it alone. Ascending 16-element K steps are unchanged. gfx1201 keeps its exact short selector and split-K=4 HC-down separate; this flag does not admit the F16 route itself. PLE, shared-down and HC-write dispatch are unchanged. Default off pending G0 and timing approval. |
| `HIPFIRE_QWEN4_SHARED_DOWN_EPI` | **opt-in**; off unless `1` | Fuses BF16 shared-down GEMM and row-scaled residual add for BF16 recipes on the existing F16 WMMA route (512+ rows, M >= 1024, eager-only). Rounds the value, row scalar, residual, product and sum independently to BF16, storing BF16-rounded F32; preserves the incumbent WMMA accumulation. gfx1201 also requires its F16 WMMA route (`HIPFIRE_QWEN4_F16_WMMA_GFX1201`, default on); it does not admit that route itself. Unset/`0`, recording/capture, other dtypes and non-BF16 recipes retain the separate kernels. |
| `HIPFIRE_QWEN4_MQ6_X4_GFX1201` | Qwen4 prefill chunks of >= 512 rows: on; set, on only for `1` | Unset, it applies only inside the Qwen4 (Flash-Next) forward, on prefill chunks of >= 512 rows (decode keeps the GEMV); `1` forces it for every MQ6G256V2 GEMM at every row count, any other value (`0`) disables it. Exact gfx1201, eager-only MQ6G256V2 trunk projections: BT8 four-row-tile X-LDS WMMA overwrite replaces memset + residual; shared rotation produces F16 directly, and chunked GDN qkv may store BF16 bits with RNE. Keeps ascending K tiles and the baseline F16 dequantization. Capture/retained recording and other architectures retain their existing routes. Bitwise-identical to the memset + residual route it replaces; `0` keeps that route. |
| `HIPFIRE_QWEN4_MQ6_X4_TILE` | Qwen4 forward on gfx1151: `auto` | Exact gfx1151 MQ6G256V2 X-LDS overwrite tile: unset, the Qwen4 (Flash-Next) forward uses the measured per-shape table in `docs/quant-formats/mq-v2-family.md`; `BVxRWxprefetch` (`BV` 8/12/16, `RW` 4/8, prefetch 1/2) forces one tile, and `auto` the table, for every MQ6G256V2 GEMM, including the a/b/z region fold and the HC-write epilogue launch (those two have BV8/BV16 twins only; BV12 keeps their incumbent; the table routes them only at their measured N=8192/2048 rows); `0` keeps the incumbent BT8 x4 kernels. Every tile keeps each output's ascending-K WMMA chain and epilogue expression, so the bytes are identical. Other arches ignore it. |
| `HIPFIRE_QWEN4_MQ6_X4_REGIONS` | Qwen4 forward on gfx1151: on; else off unless `1` | Exact; unset, on only inside the Qwen4 (Flash-Next) forward on gfx1151, `1` on gfx1151/gfx1201 everywhere: folds the chunked GDN a/b/z MQ6 F32 projections (sharing one prepared F16 rotation) into one row-region launch with independent accumulators; bytes identical to three launches. `0` keeps three launches. |
| `HIPFIRE_QWEN4_QSA_WMMA_GATHER` | gfx1151, gfx1201: on unless `0` | Runs QSA prefill attention for chunks of >= 512 rows that the full-window dense route does not take (gfx1151: the past-budget chunks of the F32 state and every such chunk of the opt-in q8 state; gfx1201: every such chunk) on the gathered F16 WMMA kernels (`indexed_attention_gathered_wmma.gfx1151.hip` on the gfx1151 F32 and q8 states, `indexed_attention_gathered_wmma.gfx1201.hip` on the gfx1201 fp8 state) instead of `indexed_attention_attention_*_batched_hg4`. Each call first writes the layer's cache rows `[0, end)` as F16 into the route's own workspace (2 KiB per context token: 128 MiB at 64K, 512 MiB at 262K). The load reserves its address for the whole `max_seq` before any capture (logged as `qwen4 QSA gather workspace`). With VMM QSA state (`kv_backend` resolved to `vmm`, gfx1151) each prefill forward maps only the prefix `[0, end)` it reads, and `HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS=auto` charges the first chunk's bytes; with legacy state the load commits all of it and the reserve charges all of it. Not bit-exact. Acceptance bar: its error against an f64 reference of the same quantized source is no worse than storing Q/K/V/P as BF16. On the worst-error 1536-row snapshot per arch (training-storage calibration) the gathered route measured max |err| 1.93e-3 (gfx1151) / 2.78e-3 (gfx1201) against 1.58e-2 / 1.68e-2 for BF16 storage, p99 9.8e-5 / 1.24e-4 against 1.01e-3 / 1.23e-3. The bar compares aggregate max and p99 on these two snapshots, not each element (93.4–93.5% of elements are within their BF16-storage error) and not a model-global bound. `0` keeps every launch of the incumbent route. Other arches and state formats ignore it; `HIPFIRE_QWEN4_F16_WMMA=0` also disables it. |
| `HIPFIRE_QWEN4_QSA_PM` | gfx1151, gfx1201: on unless `0` | Runs the gathered route's (`HIPFIRE_QWEN4_QSA_WMMA_GATHER`) F16 K/V producer and attention from the embedded certified builder (PeaceMaker) module (`kernels/qsa_gather_pm_gfx{1151,1201}.hxaco`) instead of the JIT-compiled hipcc kernels: same ABI, grid, block, LDS layout and output bytes. The module has no producer for the opt-in q8 QSA state: q8 always converts with the hipcc `indexed_attention_kv_f16vb_q8` and attends through the module. A cache whose F16 scratch or token stride would overflow the module's 32-bit buffer offsets (or `v_mad_u32_u24` operands) keeps the hipcc kernels. `0` keeps the hipcc kernels everywhere; with the gathered route off it does nothing. |
| `HIPFIRE_QWEN4_QSA_SCORE_PM` | gfx1151: on unless `0` | Runs the live QSA selector pair's rows16 F32 score kernel from the certified builder (PeaceMaker) module `kernels/qsa_select_pm_gfx1151.hxaco` instead of hipcc. Exact gfx1151 only: >= 512 rows, 4 index heads, dimension 128, <= 130816 pooled blocks, budget <= 512 and buffer extents within the module's 32-bit offsets. Selection is byte-identical. Recorder/graph capture, decode/verify, unsupported extents and other architectures keep hipcc; `0` keeps the hipcc score kernel independently of the select flag. Defined in `crates/rdna-compute/src/tensor_ops.rs`; emitter `crates/hipfire-isa/src/kernels/qsa_score.rip.rs`, aggregator `qsa_select.rs`, contract `crates/hipfire-isa/kernels/qsa_select.gfx1151.score.contract.json`. |
| `HIPFIRE_QWEN4_QSA_SELECT_PM` | gfx1151: on unless `0` | Runs the live QSA selector pair's select-from-scores kernel from the same certified builder module `kernels/qsa_select_pm_gfx1151.hxaco`, with the same eligibility and byte-identical selection as `HIPFIRE_QWEN4_QSA_SCORE_PM`. Recorder/graph capture, decode/verify and other architectures keep hipcc; `0` keeps the hipcc select kernel independently of the score flag. Defined in `crates/rdna-compute/src/tensor_ops.rs`; emitter `crates/hipfire-isa/src/kernels/qsa_topk.rip.rs`, aggregator `qsa_select.rs`, contract `crates/hipfire-isa/kernels/qsa_select.gfx1151.select.contract.json`. |
| `HIPFIRE_QWEN4_MQ6_X4_PM` | gfx1201: on unless `0` | Runs the exact-gfx1201 MQ6G256V2 F32 overwrite trunk GEMM (`HIPFIRE_QWEN4_MQ6_X4_GFX1201` route, non-BF16 output) from the embedded certified builder (PeaceMaker) module (`kernels/qwen4_mq6_x4_pm_gfx1201.hxaco`, entries `qwen4_mq6_x4_pm_gfx1201_w4`/`_w8`) instead of the JIT-compiled hipcc `gemm_mq6g256v2_wmma_gfx12_bt8_x4`: same arguments, 256-token batch tile (the module's `n_tile`; the hipcc BT8 tile is 128) and output bytes. Rows <= 64 use the 64-row 128-thread entry, larger M the 128-row 256-thread entry. BF16 output, the residual, regions and HC-write twins keep the hipcc kernels, as does every other architecture. `0` keeps the hipcc kernel (same-binary control). |
| `HIPFIRE_QWEN4_QSA_SELECT_EXACT` | **opt-in**; off unless `1` | `1` runs the batched QSA selector (`indexed_attention_select_{f32,bf16}_batched_exact`, `indexed_attention_select_exact.hip`) for launches with <= 2048 complete pooled blocks and <= 512 budget blocks outside a recorder or graph capture: 256-element tile sorts plus a fixed-order top-512 merge replace the all-pairs ranks, with the incumbent's score association, score-descending / index-ascending order, `-1` fill, tails and mirror. Selected bytes are byte-identical to the incumbent. Every other launch (longer contexts, recording, capture, the serial kernel) keeps the incumbent selector; unset or `0` keeps it everywhere. |
| `HIPFIRE_QWEN4_MOE_SYM_IU4` | gfx1151 and gfx1201: on unless `0` | gfx1151 and gfx1201 only: runs Qwen4 prefill chunks of 512+ rows on the grouped symmetric IU4 MoE route (stable grouping, A4 gate/down activations, IU4 gate/up and down) for layers whose every routed-expert QT44/QT53 header is verified symmetric (`zp == -8*sc`) on the device at load. Only a symmetric requant artifact (GPTQ3) takes it: shipped asymmetric artifacts fail the check and stay on the F16 WMMA route automatically, and the load log prints `N/48 layers verified symmetric`. Trade on gfx1151 (Strix Halo, GPTQ3 artifact): pp8192 ~1,775 tok/s vs ~1,459 on the F16 route, BF16-reference KLD 0.1027 vs 0.0661. Trade on gfx1201 (R9700, `auto` placement, GPTQ3 artifact): pp8192 2,171.5 vs 1,449.1 tok/s (median of 3 each), BF16-teacher KLD WikiText-2 0.0877 vs 0.0572 (+0.0306) and code 0.1018 vs 0.0733 (+0.0285), the same deltas as the Halo's. Needs the default two-candidate A4 producers (gfx1151: `HIPFIRE_GFX11_A4_CANDIDATES` unset or `2`; gfx1201: `HIPFIRE_G12_A4C2` unset or `1`). The GEMMs are certified builder (PeaceMaker) expert-run tiles: four 16-slot tiles per expert weight stream, eight on gfx1201 for layers whose experts are host-mapped (spilled past the VRAM budget); every width is byte-identical to the 16-slot tile. Not bit-exact against the F16 WMMA route. `0` keeps the whole F16 route (scatter, producers, GEMMs) on symmetric artifacts too; decode, MTP and smaller prefill always keep it. Every other arch ignores it. |
| `HIPFIRE_QWEN4_EXPERT_STAGE` | on for exact gfx1201, off elsewhere; `1`/`0` force | Qwen4 long prefill with host-mapped experts (`HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS` below the layer count): each forward of at least `HIPFIRE_QWEN4_EXPERT_STAGE_MIN_ROWS` rows DMA-copies every host layer's unchanged QT44/QT53 expert bytes into VRAM ahead of its MoE, through static pointer tables and compute/copy events, instead of reading them over PCIe. Placement is unchanged: the two stages borrow the expert VRAM of the last two resident layers once their own MoE has run, and those layers are copied back from pinned host copies (2 x 1,275 MiB, taken at load) before the forward returns, so decode, short prompts and MTP verify run on today's residency and mapped reads. The restore overlaps the last host layer and the LM head; what is left is charged to that chunk. Byte-identical: no arithmetic changes. Off (one `qwen4 long-prefill expert staging: off` line) when the host RAM or GTT checks refuse the restore copies, fewer than two layers are resident, or the artifact is not the Flash-Next QT44/QT53 geometry; a stream capture keeps the mapped path. |
| `HIPFIRE_QWEN4_EXPERT_STAGE_MIN_ROWS` | `2048` | Smallest forward (prefill chunk) that stages under `HIPFIRE_QWEN4_EXPERT_STAGE`; values below 2 keep the default. A layer copy takes 23.5 ms on gfx1201, which a 2048-row layer already hides (pp2048 1057 -> 1377 tok/s on card B, 18 ms of restore left exposed per chunk). |
| `HIPFIRE_QWEN4_MOE_SYM_PM` | `1` | gfx1151 only, when `HIPFIRE_QWEN4_MOE_SYM_IU4` applies: `0` runs the route's two grouped IU4 GEMMs from the hipcc module (`qwen4_moe_iu4_sym_gfx1151`) instead of the embedded builder bundle. Both produce the same bytes; this is a same-binary control. gfx1201 has only the builder bundle and ignores it. |
| `HIPFIRE_QWEN4_TRUNK_IU4` | **opt-in**; unset / `0` = off | Exact gfx1151 only. Selects which trunk projections run the dense IU4 route (A4 activations) on a Qwen4 trunk whose tensors may mix symmetric MQ4G256V2 (QT44, `zp == -8*sc`) and MQ6G256V2 (per tensor, e.g. `HIPFIRE_QWEN4_REQUANT="pat=qt44:DIR;..."`). Value: `1` or `all` (every family, every layer) or a comma list of `family[@lo-hi]` tokens; `lo-hi` are inclusive 0-based full-layer indices (`@n` = one layer, no `@` = every layer). Families: `gdn.qkv gdn.z gdn.a gdn.b gdn.out qsa.q qsa.k qsa.v qsa.idx qsa.o`; groups `gdn` (the five GDN), `qsa` (the five QSA), `all`. Examples: `all`; `gdn,qsa.o@3-15` (all GDN projections everywhere plus the QSA output projection on layers 3..15); `gdn.qkv@0-20,gdn.out`. An unparsable value (including a `+` sign or a non-digit in a layer index) or a band outside the trunk (`lo` >= the layer count, or an explicit `hi` >= the layer count) fails the model load with an error naming the bad token. At load every MQ4G256V2 trunk matrix is checked symmetric on the device and the log prints `qwen4 IU4 trunk: mask=<canonical> MQ4 <v>/<n> verified symmetric, A4 <a> projections, MQ6/other <o>` (`a` = MQ4 and in the mask); the route arms iff at least one MQ4 trunk tensor is in the mask (`a > 0`) and all MQ4 ones verified. Prefill chunks of 512+ rows in whole 128-row tiles (eager, no recorder or graph capture) then run each masked MQ4 projection on the dense IU4 route: one A4 producer per shared input (`mq_rotate_x_i4`, FWHT-256 + `block_i4_128`) feeding the dense SET GEMMs (V2B / PeaceMaker `pm_v2b` where the grid is eligible, X5 / symfold otherwise); a GDN layer whose Z, `in_proj_b` and `in_proj_a` are all MQ4 and masked folds them into one `z + 256`-row V2B SET whose epilogue scatters the three outputs (a load-time copy of those rows, ~8.7 MB per layer). MQ4 projections outside the mask keep exact activations (F16 BT4 WMMA, one shared rotation per input) on chunks of 96+ rows; chunks under 96 rows, and any chunk under a recorder or graph capture, run unmasked MQ4 projections on the generic MQ4 GEMM (`gemm_mq4g256v2`), which on gfx1151 at 64..95 rows is itself the IU4 (A4) prefill. The whole-128-row-tile rule for the masked A4 route is unchanged. MQ6 (and every other tier) projections are untouched and keep the exact F16 route, including the MQ6 a/b/z row-region fold (only the group's original a/b/z, never qkv), the HC-write fusion of the MQ6 output projection and the BF16 qkv store. Not bit-exact against the F16 route: A4 is a 4-bit activation quantization (opt-in, KLD-gated by the caller); the native MTP head never takes it. Needs `HIPFIRE_IU4_PREFILL` and `HIPFIRE_IU4_SYMFOLD` on (their defaults). |
| `HIPFIRE_QWEN4_GDN_CONV_QKNORM` | gfx1151: on unless `0`; ignored elsewhere | Exact gfx1151 only, on the chunked GDN prefill route (>= 512 rows, eager, F16 WMMA route on): makes the GDN convolution launch also store the Q/K the recurrence reads already normalized (`gated_delta_conv_qknorm_bf16_f32_batched_k4` in `tensor_ops.hip`), and skips the separate `gated_delta_qk_norm_bf16_batched` launch. The normalized Q/K and every other output are bytewise what the two-launch pair stores; the convolution output's Q/K columns are not written (the recurrence reads only V from it). Needs 16 key heads of 128 in q\|k\|v channel order, `channels % 256 == 0`, a 3-row ring and a BF16 convolution output; any other shape keeps the two launches. Never taken under a recorder, a graph capture, or a row capture, and gfx1100/gfx1201 (whose GDN prefill is the persistent route) ignore it. `0` keeps every launch of the incumbent route; this is the kill switch. |
| `HIPFIRE_QWEN4_GDN_PIPE` | gfx1201: on unless `0`; ignored elsewhere | Exact gfx1201 only, on the batched GDN recurrence without a state capture (prefill; not verify or rollback): runs `gated_delta_step_pipe128_{f32,q8}` (`tensor_ops.hip`, 128-thread blocks, one value column per thread with its 128 state values in registers, no cross-thread handoff per row) instead of `gated_delta_step_halves_state128_persistent256_{f32,q8}`. The output and the final F32 / Q8 state (codes and scales) are bytewise the persistent256 kernel's. `0` keeps the persistent256 kernel; this is the kill switch. |
| `HIPFIRE_QWEN4_GDN_Q8_INLINE` | **opt-in**; off unless `1` | Exact gfx1151 only, on the chunked GDN prefill route with a Q8 GDN state: `1` runs the recurrence as `gated_delta_chunk_gate_q8_wmma` (module `gated_delta_chunk_q8_wmma`), which decodes the Q8 slot to F32 on load and requantizes it (seeded by the last row, `position + rows - 1`) on store, instead of the `gdn_state_q8_to_f32` / `gdn_state_f32_to_q8` launches and F32 scratch around `gated_delta_chunk_gate_wmma`. The gated output and the Q8 slot are bytewise the conversion arm's. An F32 state, a slot not 16-byte aligned, and every other arch keep the incumbent arm. Unset or `0` keeps every launch of the incumbent route; this is the kill switch. |
| `HIPFIRE_QWEN4_HC_FUSE` | gfx1151: `3`; gfx1201: `1` | Exact gfx1151 and gfx1201, on the F16 WMMA HC read (>= `QWEN4_F16_WMMA_MIN_TOKENS` rows, eager, no recorder or graph capture; gfx1201 also needs `HIPFIRE_QWEN4_F16_WMMA_GFX1201`, default on, which enables that read). Fusion level of the HC read/write pair around each Qwen4 mixer; every level leaves the HC streams bytewise what the unfused sequence stores. `1`: the HC read's norm launch also projects its paired HC write's gates (`hyper_norm_gate_outputs`), so the write skips its own norm + gate launch. `2`: also the MQ6G256V2 attention output projection (GDN `output`, QSA `o_proj`) applies the HC write in its GEMM epilogue (`gemm_mq6g256v2_wmma_gfx11_bt8_x4_hcw`; gfx1201 `gemm_mq6g256v2_wmma_gfx12_bt8_hcw`, >= 64 rows); the attention output tensor is not materialized. `3`: also, on gfx1151, a BF16 shared-expert down on the F16 WMMA route folds the BF16 scaled add and the HC write into its epilogue (`gemm_wmma_lds_128_256_32_64_k64_hcsd`); the routed `moe_output` rows are then NOT rewritten with the shared-down sum (they are dead after the write). Levels 2 and 3 measured slower than 1 on gfx1201, which defaults to 1. A level that a given layer's shapes or routes do not admit runs the next lower one. An unparsable value keeps the arch default. `0` keeps every launch of the incumbent route; this is the kill switch. |
| `HIPFIRE_QWEN4_HC_UP_TILE` | gfx1151 / gfx1201: on unless `0` | Exact gfx1151 and gfx1201, on the F16 WMMA HC read: runs the up projection + branch mix on the retiled operand-swapped entries (`hyper_read_up_wmma_bf16_swap` on gfx1151; `hyper_read_up_wmma_bf16_gfx1201_t128` on gfx1201 when `low_rank % 64 == 0`), bytewise the baseline entries' output. `0` keeps the baseline entries; this is the kill switch. |
| `HIPFIRE_QWEN4_MOE_COMBINE_ZINIT` | gfx1151 / gfx1201: on unless `0`; ignored elsewhere | Exact gfx1151 and gfx1201, sealed Qwen4 MoE grouped prefill whose combine takes the BF16-row path 2 arm with the shared down after the combine: starts the combine from +0.0 (`moe_down_combine_grouped_top10_bf16in_zinit`) and drops the `moe_output` zero fill that precedes the call. Bytewise the zero fill + the unchanged combine. The fill stays whenever any condition is unmet (other route, recorder / retained tape / graph capture, target not exactly the cleared tensor); an absorbed fill the combine did not consume fails the call instead of leaving the target uncleared. Also taken on a G2-staged prefill (host-mapped experts), which runs the whole step program in one interpreter call with the stage waits and refills as MoE hooks. `0` keeps the separate fill; this is the kill switch. |
| `HIPFIRE_QWEN4_HC_ROW_FOLD` | gfx1151: on unless `0`; ignored elsewhere | Exact gfx1151 only, with `HIPFIRE_QWEN4_HC_FUSE=3` and `HIPFIRE_QWEN4_MOE_COMBINE_ZINIT=1` (both gfx1151 defaults) on the sealed Qwen4 MoE grouped prefill whose combine takes the BF16-row path 2 arm, eager, BF16 HC streams, when the MoE block's HC write is directly followed by the next layer's HC read on the same streams and the F16 WMMA read route: runs the shared-down GEMM as a plain BF16 store (`gemm_wmma_lds_128_256_32_64_k64_bf16st`) and replaces the combine, the shared-down fold + HC write epilogue and the next read's `hyper_norm_gate_outputs` with one per-token row kernel (`hc_row_fold_norm_gate`) that combines the ten expert rows in rank order, folds the shared row, writes the four HC streams and emits the next read's F16 normalized row and its paired write's gate logits; the next read then skips its norm-and-gate launch. Bytewise the unfused chain (streams, F16 row, gate logits). Idea from Gufo upstream (gufo-org/gufo @1071b361, MIT). `0` keeps the existing launches; this is the kill switch. |
| `HIPFIRE_QWEN4_ROUTER_FAST` | gfx1151 / gfx1201: on unless `0`; ignored elsewhere | Exact gfx1151 and gfx1201, Qwen4 E512/top-10 grouped prefill router with more than one token, eager (no recorder or graph capture): runs `moe_router_softmax_top10_f32_fast` (one wave32 per token, logits register-resident, no workgroup barriers) instead of the one-workgroup-per-token `moe_router_softmax_top10_f32`. Bytewise the same expert ids and weights (same stable top-ten order, same left-to-right float32 denominator, same expf calls, same early-return rules; a token with any non-finite logit runs the incumbent's selection verbatim). Decode (one token), recording and capture keep the incumbent symbol. `0` keeps the incumbent; this is the kill switch. (Idea from Gufo upstream, gufo-org/gufo @1071b361, MIT.) |
| `HIPFIRE_QWEN4_PLE_FUSE` | gfx1151 / gfx1201: on unless `0`; ignored elsewhere | Exact gfx1151 and gfx1201, Qwen4 PLE block (once per prefill, layer 1) on BF16-stored HC streams: replaces the `hc_state_bf16_to_f32` / `grouped_gate_bf16` / `grouped_norm_bf16` / `grouped_depthwise_conv_silu_add_bf16` / `hc_state_bf16_add_f32` chain with three launches (`ple_gate_rows_bf16s`, `ple_norm_inv`, `ple_conv_add_bf16s` in `grouped_ops.hip`) that keep the incumbent's serial reduction orders and every F32 rounding point, so the HC streams and the convolution state are bytewise the incumbent's for every non-NaN value (where both chains produce NaN, the NaN payload bits may differ). With BF16 key/value weights on the F16 WMMA route the PLE rows are also gathered straight to F16 (`grouped_gather_convert_bf16_f16`), dropping the F32 rows and both F32 -> F16 conversions; the 4-tap dilation-3 PLE convolution runs a register-window kernel (`ple_conv_add_bf16s_k4d3`). The gate scalars and norm inverses live in the `ple_normed` scratch prefix; the F32 widened query, gated, normed and conv-output tensors are not written. Never taken under a recorder, retained tape or graph capture, with F32 streams, more than 4 branches or `hidden % 32 != 0`; those keep the incumbent chain. `0` keeps the incumbent chain; this is the kill switch. (Idea from Gufo upstream, gufo-org/gufo @1071b361, MIT.) |
| `HIPFIRE_QWEN4_HC_DOWN_TILE` | gfx1151: on unless `0`; ignored elsewhere | Exact gfx1151 only, on the F16 WMMA HC read's down projection (`gemm_bf16_xf16_f16_wmma`, 320×10240, >= 2048 rows, eager, no recorder or graph capture): runs it on the 160×64 pipelined LDS tile `gemm_wmma_lds_160_64_32_64_k64_p` (five 32×64 waves, 256 workgroups) instead of 64×64 (640 workgroups). Every output keeps the 64×64 tile's single ascending 16-element K chain and store, so the F32 `low` is bytewise the baseline's. Smaller batches, other shapes and arches keep the baseline tile. `0` keeps the 64×64 tile; this is the kill switch. |
| `HIPFIRE_MTP_INCREMENTAL` | **unset**: per-window choice | Native MTP verify route. Unset picks, per window, a batched `(K+1)`-row verify at the depth that maximizes expected tokens per cost, or the interleaved route (one target row per draft, stop at the first rejection). `0` forces batched at the full `mtp_k`; `1` forces interleaved. Both emit AR's greedy tokens. |
| `HIPFIRE_MTP_SAMPLED_MODE` | `leviathan` | Sampled (temperature > 0) Qwen4 native MTP verifier, with `speculation.mtp_sampled` on. `leviathan`: speculative rejection sampling (drafts drawn from the head's 8 re-scored candidates; output equals AR in distribution). `naive`: SpecInfer naive sampling (argmax drafts, one AR-sampler draw per verify row, accept iff equal); a seeded request emits AR's exact tokens. Any other value fails sampled requests. |
| `HIPFIRE_MTP_DRAFT_HEAD` | `mq2r` | Draft ranking head: an `mq2`..`mq6` copy of the LM head (`r` suffix = re-score its top 8 exactly against the head's own Q8_0 or MQ6G256V2 rows). |
| `HIPFIRE_MTP_PAIRING` | head state | Draft-step conditioning experiment: `aligned-head` or `aligned`. |
| `HIPFIRE_MTP_TRACE` / `HIPFIRE_MTP_PHASE_TIMING` | off; `1` enables | Per-window MTP trace / per-phase `hipEvent` timing to stderr (diagnostic). |
| `HIPFIRE_QWEN4_MTP_BATCHED_FILL` | gfx1151: on unless `0`; ignored elsewhere | Exact gfx1151 only, Qwen4 native MTP prompt fill (`mtp_prefill`): runs the head's KV-only Append step over each prefill chunk as one row-batched pass (sub-chunks of up to 1024 rows, about 17 launches each) instead of one ~20-launch single-row `forward_token` per prompt row. Every operator is the single-row kernel with a rows grid, or a multi-row kernel whose per-row reduction order is the single-row kernel's, and never the F16 WMMA route, so the head's QSA state (`full_keys`, `full_values`, `raw_index_keys`, `pooled_keys`, lengths), the draft request state and the pending hidden row are built to match the per-row fill byte for byte; the (token p, hidden p) pairing is unchanged. Read at each `mtp_prefill`. Allocates about 260 MB of row-batched scratch per drafter at the first fill (counted in `native_mtp_device_bytes`). `0` keeps the per-row loop and allocates nothing; other arches always take the per-row loop. |
| `HIPFIRE_QWEN4_PLE_WIDE_READERS` | on unless `0` | Qwen4 SSD-resident PLE rows: a row request with more than 1,024 scattered page reads (a prefill chunk's ~110K) splits them over up to 256 concurrent reader threads (one per 64 reads) instead of 16; smaller requests (decode, MTP verify, short prompts) keep 16. Bytes, order and the row-store cache are unchanged; only the read queue depth differs. Read when the model loads. `0` keeps 16 readers for every request. |
| `HIPFIRE_QWEN4_TRUNK_TIER` | Q8F16 declaration | `mq6` declares the rank-2 trunk attention/GDN matrices at MQ6G256V2 for the quantizer; the loader admits both tiers from the file. |
| `HIPFIRE_QWEN4_MTP_TIER` | recipe tier | `source` keeps the rank-2 MTP matrices at BF16 (quantizer and loader comparison knob). |
| `HIPFIRE_QWEN4_REQUANT` | unset | Load-time precision experiment: `pat=fmt;...` requantizes resident rank-2 weights whose name contains `pat` to `mq2`..`mq6` or `q8`, or (`qt44:DIR`) replaces each with the pre-encoded QT44 bytes of `DIR/<tensor name>.bin` where that file exists (an offline GPTQ solve; the file must hold exactly the matrix). |
| `HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS` | unset: every routed expert in VRAM where it fits, else `auto` | Discrete-GPU capacity knob. Unset on a discrete GPU keeps every routed expert in VRAM only when all of them, the non-expert weights and the `auto` reserve below fit in free VRAM, and resolves to `auto` otherwise (a 32 GB card); unified memory (Strix Halo) stays fully resident. An explicit value always wins. The load logs the choice on one `qwen4 expert placement` line. Set or unset, a daemon started for a Qwen4 model on discrete GPUs gets the two host-memory switches below (`HSA_USERPTR_FOR_PAGED_MEM`, `GPU_PINNED_MIN_XFER_SIZE`), which are decided before the HIP runtime loads; a daemon started for another model (serve switching models) gets them only with the variable set, and its Qwen4 load warns without them. `N` keeps the routed experts of trunk layers `0..N` in VRAM; later layers' and the MTP layer's routed experts are fulfilled into pinned, device-mapped host RAM and read over PCIe (zero-copy) by the unchanged sealed MoE kernels. `auto` picks the largest `N` that leaves a computed reserve of VRAM beyond the non-expert weights: 6.5 GiB (measured without MTP at `max_seq` 32768 with legacy QSA state, a full-context prefill included; N=16 on a 32 GB R9700), plus the trunk QSA arenas' committed bytes past those 32768 legacy tokens, plus, when native MTP stays attached (the default on exact gfx1201, `--spec mtp` elsewhere), the MTP head's committed device bytes (its QSA state and the MQ2 draft copy of the language head) and the larger of the draft copy's F32 requant scratch (vocab × hidden × 4 B, returned to the device at attach) and what the first speculative request allocates (the verify hidden rows and the `K+1`-row GDN capture rings in the state's GDN format). QSA context arenas are charged as committed, never by their virtual extent: legacy state commits all of `max_seq`; VMM state (`kv_backend` `vmm`) commits only the pages the first prefill chunk maps, so the admitted context does not grow the VMM reserve and later growth maps at forward time (a growth that does not fit fails the request before its position advances). On the R9700 at `max_seq` 16384 the attached head costs two expert layers (layers `0..15` in VRAM without it, `0..13` with it). The load logs the reserve on a `qwen4 auto expert placement` line, and refuses before allocating when the free VRAM cannot hold the non-expert weights plus the reserve even at N=0 (for example beside another process's model). `0` puts every routed expert (~65 GB) in host RAM. The load refuses before allocating when `MemAvailable`, plus (with GTT-backed host memory, `HSA_USERPTR_FOR_PAGED_MEM=0`) an estimate of the freed GTT pages in TTM's page pool, is below the pinned bytes plus 4 GiB. The pool is readable only by root, so the estimate is the host memory outside every `/proc/meminfo` counter, minus the GTT amdgpu devices hold and 4 GiB for other drivers, capped at TTM's `page_pool_size`; the load log prints both numbers. `echo 2 \| sudo tee /proc/sys/vm/drop_caches` empties the pool. Refused on UMA. With any expert host-mapped, native MTP stays on under `auto` on exact gfx1201 (greedy MTP text equals AR; `--spec off` / `speculation.mtp = "off"` keeps AR) and is opt-in elsewhere (`--spec mtp` / `speculation.mtp = "on"`; `auto` keeps AR). |
| `HIPFIRE_QWEN4_ROUTE_TRACE` | unset | Diagnostic: append one line per single-token HIP forward to this file, `position` followed by every layer's routed expert ids. Synchronizes the device each token. |

### Vision tower sidecar

| Variable | Default / sense | Notes |
|---|---|---|
| `HIPFIRE_VISION_SIDECAR` | explicit vision-tower path (overrides the registry sidecar); empty opts out | Read via `developer_var` (env beats `developer.vision_sidecar`); wired into the daemon load as `params["vision"]`. Skipped while `vision_mode=off`. |
| `HIPFIRE_VISION_MODE` | tower sidecar gate: `off` (default) / `auto` / `on` | Env-compat for config `vision.mode`; projected into load params as `vision_mode` and enforced daemon-side. |
| `HIPFIRE_IMAGE_DECODE` | VL image JPEG decode path: `cpu` (default) / `vcn` / `auto` | Env-compat for config `image.decode`; read via process snapshot in `hipfire-arch-qwen35-vl`. The standard daemon build compiles the `vcn-jpeg` path in (default feature). Runtime default remains `cpu`, which never enters the VCN prepass. `vcn` and `auto` attempt shared VCN JPEG decode and fall back to CPU for unsupported inputs, platforms where VCN is unavailable, or recoverable decode failure. A failed terminal GPU completion fails closed by quarantining the shared VA session, emitting a request error, and exiting the daemon nonzero (restart required) instead of unsafe same-device CPU fallback. When VCN runs, pooled decode surfaces stay leased until GPU preprocessing completes; the learned vision tower is unchanged. |
| `HIPFIRE_VL_FILE` | explicit `.vl` tower file for the multi-slot engine | Checked before the `<stem>.vl` sibling probe in `hipfire-runtime` `sidecar::resolve_vl_sidecar`; a set-but-missing path is reported, not silently ignored. |
| `HIPFIRE_VIT_ATTN` | `naive` forces the per-(head, query) ViT attention kernel | Rollback/A-B switch for the Q-tiled ViT attention kernel. Head dims outside the Q-tiled contract (`head_dim % 16 != 0` or `> 128`) use the naive kernel regardless. |

### Graph / MMQ / prefill

| Variable | Notes |
|---|---|
| `HIPFIRE_GRAPH` | hipGraph capture opt-in (quality caveats; AR-oriented) |
| `HIPFIRE_GEMMA4_GRAPH` | Gemma4 per-model decode graph: `1` enables, `0` disables (kill switch). Hand carrier defaults on. Lowered carrier: this switch wins, then `HIPFIRE_GRAPH=0/1`, then the arch default, which is on for exact gfx1201 and off for every other arch (gfx1100, gfx1151, gfx1200, …). Lowered captures after one eager warm-up, stages position on-device before replay, and re-captures when KV/weight/scratch allocations or shapes change. Dense bodies and indexed MoE with HFQ4G128 expert down are admitted; CPU expert fallback and atomic Q8 expert down remain eager. |
| `HIPFIRE_GRAPH_MOE` | MoE graph opt-in |
| `HIPFIRE_VERIFY_GRAPH` / `_TIMING` / `_TREE` | verify-side graph diag |
| `HIPFIRE_MMQ` / `HIPFIRE_WO_MMQ` | MMQ activation |
| `HIPFIRE_MMQ_SCREEN` / `HIPFIRE_MMQ_SCREEN_THRESHOLD` / `HIPFIRE_MMQ_MIN_BATCH` | screening |
| `HIPFIRE_PREFILL_BATCHED` / `HIPFIRE_PREFILL_CHUNK_ROWS` | batched prefill knobs (config mirrors). PFlash (`speculation.prefill.*`, `HIPFIRE_PFLASH_*`) is **deprecated since 0.4.0, removal in 0.5.0** |
| `HIPFIRE_PREFILL_BATCHED` | RuntimeConfig: on unless `0` (Qwen-style batched prefill gate — **not** the LFM flag) |
| `HIPFIRE_FLASH_PREFILL` | Developer override for Q8 WMMA flash prefill: `0` forces off, `1` forces on; unset uses the architecture/workload envelope. |
| `HIPFIRE_GFX1100_PACKED_MQ4_PREFILL` | Experimental, default off, exact gfx1100: `1` runs the dense 27B FFN gate/up and down (17408x5120 / 5120x17408) of uniform MQ4G256 models through the packed MQ4 GEMM (144-byte DS4 activation, native per-32 Q8 scale) for 256-multiple chunks. Full (not lean) PBS; graph capture and retained recording stay on the native route. |
| `HIPFIRE_GFX1100_MQ4_WIDE_PREFILL` | Experimental, default off, exact gfx1100: `1` admits dense 27B models whose projections are all uniform MQ4G256 to widened ordinary prefill (`HIPFIRE_PREFILL_CHUNK_ROWS` 512..8192, memory admission may reduce it) on native MMQ and full PBS, keeping 512-row recurrent commits. Rejects unaudited dispatch overrides. With `HIPFIRE_GFX1100_PACKED_MQ4_PREFILL=1`, complete 512-row chunks coalesce and irregular tails keep their legacy grouping, so each row's quantization route is unchanged. |
| `HIPFIRE_FLASH_PREFILL_FIXED_HD` | Developer ablation: fixed-head-dimension specialization is on unless `0`. |
| `HIPFIRE_FLASH_PREFILL_PREFETCH_V` | Developer ablation: gfx12 V prefetch is on unless `0`. |
| `HIPFIRE_GFX11_FA2_PREFILL` | GQA-fused FA2 prefill on gfx1100/gfx1151 (Qwen NH24/NKV4/HD256, N 64..512 step 16, ctx 64..32768) — default ON (`kernel.gfx11_fa2_prefill`); `=0` opts out toward the byte-identical incumbent |
| `HIPFIRE_GFX11_Q8_FA2_WIDE` | Whole-chunk Q8/Q8 FA2 prefill (B 64..8192, above 512 aligned to 512; Qwen NH24/NKV4/HD256) — **auto ON on exact gfx1100 and exact gfx1151** (`kernel.gfx11_q8_fa2_wide`, experimental); other arches off; `=0` opts out. Requires `kernel.gfx11_fa2_prefill`. gfx1151 was enabled deliberately in commit `ae9c5cf9d12a5a63ae60630637d22d00a4fc964e` through the exact-gfx1151 FA2 twin (`HIPFIRE_GFX1151_FA2_TWIN`: CU mode, heaviest q-tiles first) with the same arithmetic, byte-identical output. Evidence (Strix Halo, H2 layer; unreleased [`CHANGELOG.md`](../CHANGELOG.md) entry): pp8192 ABBA +2.13 %, BAAB +2.11 %, captured layer −27.5 % / −29.7 %, KLD pins unchanged. |
| `HIPFIRE_FA2_FILL` | Warp-specialized K/V fill in that FA2 kernel on gfx1100/gfx1151 (bit-exact; helper waves dequantize the next K/V tile while compute waves run QK/PV) — default ON; `=0` restores the all-wave per-tile fill |
| `HIPFIRE_GFX1100_FA2_R3` | Exact-gfx1100 variant of that FA2 fill body (bit-exact; CU mode, bank-conflict-free helper plane stores, O rescale skipped when alpha is exactly 1, heaviest q tiles first; symbols `attention_q8_0_fa2_gqa_gfx1100` / `attention_fa2_q_preconvert_gfx1100`) — default ON; `=0` restores the shared gfx11 body |
| `HIPFIRE_GFX1151_FA2_TWIN` | Exact-gfx1151 twin of that FA2 fill kernel (CU mode, heaviest q-tile first, conflict-free helper V stores; bit-exact) — default ON; `=0` restores the gfx11 module |
| `HIPFIRE_GFX12_FA2_PREFILL` | GQA-fused FA2 prefill on exact gfx1201 (same Qwen NH24/NKV4/HD256 envelope) — default ON (`kernel.gfx12_fa2_prefill`); `=0` opts out toward the byte-identical incumbent |
| `HIPFIRE_GFX12_FA_PACKET` | Packet-minimal Q128 FA2 body on exact gfx1201 (same Qwen envelope as `HIPFIRE_GFX12_FA2_PREFILL`) — default ON (`kernel.gfx12_fa_packet`); `=0` opts out to the byte-identical route-N body |
| `HIPFIRE_ATTN_QRESIDENT_V2` | Bit-exact v2 schedule of the gfx1201 register-resident-Q FA2 prefill kernel (same Qwen envelope; only where `kernel.attn_qresident` selects the Q-resident route) — default ON (`kernel.attn_qresident_v2`); `=0` restores the byte-identical v1 Q-resident kernel |
| `HIPFIRE_GFX12_FA_PREP_FUSED` | Exact gfx1201 FA Q/K norm and RoPE fusion (`kernel.gfx12_fa_prep_fused`); default ON only on gfx1201, `=0` restores separate launches |
| `HIPFIRE_GFX12_FA_PREP_FP8Q` | Preconvert Q to E4M3 codes for gfx1201 Q-resident v2 attention (`kernel.gfx12_fa_prep_fp8q`); default ON only on gfx1201, `=0` retains F32 Q; requires fused prep and Q-resident v2 |
| `HIPFIRE_FP8_DECODE_ATTN_GQA` | Exact-gfx1201 native-fp8 **decode** attention (head_dim 256, GQA group 6, tile 128, no output gate: H2): GQA-shared flash tile (one 256-thread workgroup per kv head and 128-key tile serves its six q heads, so K/V are read once) + head-dim-split reduce (`attention_flash_fp8_e4m3_tile_gqa_gfx1201` / `attention_flash_reduce_dsplit_gfx1201`); byte-identical partials and output — default ON; `=0` restores `attention_flash_fp8_e4m3_tile` + `attention_flash_q8_0_reduce` |
| `HIPFIRE_GFX1100_DECODE_ATTN_GQA` | Exact-gfx1100 Q8_0 **decode** attention with the gated AWQ MQ-rotating epilogue (head_dim 256, GQA group 6, tile 128, full causal: H2): GQA-shared flash tile (`attention_flash_q8_0_tile_gqa_gfx1100`, one 256-thread workgroup per kv head and 128-key tile, K/V read once for the six q heads, every load issued at entry) + load-ahead reduce/gate/rotate (`attention_flash_q8_0_reduce_gated_mq_rotate_awq_dec_gfx1100`, one 1024-thread workgroup per head); byte-identical partials and output — default ON; `=0` restores `attention_flash_q8_0_tile` + `attention_flash_q8_0_reduce_gated_mq_rotate_awq_gfx1100` |
| `HIPFIRE_GFX1151_Q8_DECODE_ATTN_GQA` | Exact-gfx1151 Q8_0 **decode** attention (head_dim 256, GQA group 6, tile 128, full causal, no output gate: H2): GQA-shared flash tile (one 256-thread workgroup per kv head and 128-key tile serves its six q heads, so K/V are read once) + head-dim-split reduce (`attention_flash_q8_0_tile_gqa_gfx1151` / `attention_flash_reduce_dsplit_gfx1151`); byte-identical partials and output — default ON; `=0` restores `attention_flash_q8_0_tile` + `attention_flash_q8_0_reduce` |
| `HIPFIRE_CALIB_BF16` | Calibration-only: keep native-BF16 teachers in BF16 (`kernel.calib_force_bf16`, default off; shipped inference unaffected) |
| `HIPFIRE_GFX12_MQ4V2_FP8_GATEUP` / `_RESID` / `_QKVZA` / `_QKV` | gfx1201 FP8-WMMA MQ4v2 prefill route — default ON on exact gfx1201 (widened prefill chunk 4096 via `prefill.chunk_rows`); `=0` on any one opts out toward the F16 path (chunk 384). `=1` forces on; launchers stay exact-gfx1201-only, so other arches are unchanged |
| `HIPFIRE_GFX12_MQ4V2_FP8_SLABS` | Two-slab S2BT8 FP8 symbols by default; `=1` selects the single-slab symbols |
| `HIPFIRE_GFX12_MQ4V2_FP8_V2` | gfx1201 FP8-WMMA MQ4v2 staged-tile v2 route — default ON on exact gfx1201 (`kernel.gfx12_mq4v2_fp8_v2`); `=0` restores the s2bt8/BT symbols. The four family flags remain prerequisites |
| `HIPFIRE_FP8_SYMFOLD` | Developer opt-out for centered FP8-v2 GEMM twins on exact gfx1201 symmetric MQ4V2 artifacts (`mq4v2.symmetric`): default enabled when the artifact marker and v2 route are both active; `=0` restores the asymmetric v2 entries. Non-symmetric artifacts are unchanged |
| `HIPFIRE_GEMMA4_PLE_BRANCH_BATCHED_PREFILL` | Exact-arithmetic batched Gemma 4 E-series PLE branch projections (`kernel.gemma4_ple_branch_batched_prefill`, experimental) — **auto ON on exact gfx1100 and gfx1201**; other arches (gfx1101, gfx1102, gfx1151, gfx1200) off; `=0` opts out. Independent of `HIPFIRE_GEMMA4_PLE_BATCHED_PREFILL` (model projection, default off). Benchmark history: [`GEMMA4_ESERIES_GFX1100_PREFILL.md`](GEMMA4_ESERIES_GFX1100_PREFILL.md), [`GEMMA4_ESERIES_LOGIT_STABILITY.md`](GEMMA4_ESERIES_LOGIT_STABILITY.md). |
| `HIPFIRE_GFX12_MQ4V2_FP8_V2_GEOM` | v2 tile geometry: `128x128` (default on exact gfx1201, the measured pin), `64x256`, `128x64`, `256x64` (prior default, still selectable) |
| `HIPFIRE_GFX12_GDN_PRE_FUSED` | gfx1201 batched-prefill GDN preamble fusion (sigmoid+conv+qknorm 3→1, byte-exact) — default ON on exact gfx1201 (`kernel.gfx12_gdn_pre_fused`); `=0` restores the 3-launch sequence |
| `HIPFIRE_GFX1151_GDN_SCAN` | Exact-gfx1151 twin of the GDN chunk scan (`gdn_chunk_scan_gfx1151`: pipelined chunk loop, K staged transposed, one value half per 256-thread workgroup; byte-identical `out` and state) — default ON; `=0` restores `gdn_chunk_scan` |
| `HIPFIRE_V2B_A4_EPI` | Developer A4-fusion arm of the exact-gfx1151 V2B gate/up SiLU launch (eager prefill only; admitted only where the certified V2B tile is: `gate_m == up_m`, `gate_m % 256 == 0`, `batch % 128 == 0`, `k % 256 == 0`; graph capture and Redline recording keep the unfused path). Unset, `1` and any other value (default ON on exact gfx1151): the fused kernel `gemm_mq4g256v2_gate_up_silu_a4_iu4_pm_v2b_gfx1151` also writes the w_down A4 sidecar (72-byte `block_i4_128` records) into a dedicated scratch slot, so `h` is not written and the hin producer is skipped; byte-identical. `0`: opt-out, the V2B SiLU + `fused_silu_hin_rotate_mq_i4_batched` path. `retile`: the stage-1 m512 twin replaces the V2B SiLU launch (same `h`, same downstream producer). Read on every call, so one process can switch arms. `HIPFIRE_V2B_PM=0` and `HIPFIRE_F1LITE=0` also disable it; other arches are unaffected |
| `HIPFIRE_V2B_A4_PM_BUNDLE` | Developer: `=<path>` loads the A4-fusion builder bundle (`gemm_mq4g256v2_gate_up_silu_a4_iu4_pm_v2b_gfx1151.hxaco`) from a file instead of the embedded image, for every `HIPFIRE_V2B_A4_EPI` arm except `0`. Read once; an unreadable file fails every A4-fusion launch |
| `HIPFIRE_GFX12_FP8_STREAM` | gfx1201 RMSNorm+rotate producer → MQ4v2 FP8 pre-pass fusion (byte-identical `prepare_mq4v2_fp8_x_f32` outputs for the qkvza/gate_up/qkv inputs; standalone pack launch disappears) — default ON on exact gfx1201 (`kernel.gfx12_fp8_stream`); `=0` opts out; other arches off |
| `HIPFIRE_G12_NORM` | gfx1201 `_v2` RMSNorm and gated-norm int4 producers (batched sum-of-squares loads + one-reciprocal RTN codes; one wave per gated-norm group; bit-identical) — default ON on exact gfx1201 (`kernel.g12_norm`); `=0` restores the incumbent `_gfx12` symbols |
| `HIPFIRE_G12_DEC_NORM` | gfx1201 and gfx1151 decode norms as multi-workgroup grids (f32 AWQ RMSNorm+FWHT: K/256 workgroups, each redoing the row's reduction and rotating one group; out-of-place single-row `rmsnorm_f32`: n/256 workgroups; half-split partial RoPE: one workgroup per head; bit-identical) — default ON on exact gfx1201 and exact gfx1151 (`kernel.g12_dec_norm`); `=0` restores the single-workgroup launches |
| `HIPFIRE_GFX1100_DEC_NORM` | gfx1100 decode norms as multi-workgroup grids (the `HIPFIRE_G12_DEC_NORM` twins built for gfx1100: f32 AWQ RMSNorm+FWHT K/256 workgroups, out-of-place single-row `rmsnorm_f32` n/256 workgroups; bit-identical) — default ON on exact gfx1100 (`kernel.gfx1100_dec_norm`); `=0` restores the single-workgroup launches |
| `HIPFIRE_G12_A4C2` | gfx1201 int4 producers search two activation scales ({5,7}, as gfx11's `-DIU4_A4_CANDIDATES=2`) instead of RTN d = amax/7; one-pass producer-layout search, bit-identical to that flag — default ON on exact gfx1201 (`kernel.g12_a4c2`; appends `-DIU4_A4_CANDIDATES=2` to the gfx1201 JIT flags); `=0` restores RTN |
| `HIPFIRE_PREFILL_CHUNK_ROWS` | Widened ordinary-prefill chunk ceiling (`prefill.chunk_rows`; default 4096 on exact gfx1201, 512 elsewhere; explicit `HIPFIRE_PREFILL_MAX_BATCH` wins; VRAM admission may admit a smaller rung). Qwen4 (Flash-Next) reads it as its prefill chunk ceiling, rounded down to 256 rows: default 8192 on exact gfx1151, 4096 on exact gfx1201, 1536 elsewhere. The PLE row staging follows the chunk, and load admission steps down 4096/2048/1536 until the chunk scratch fits free VRAM with 1 GiB to spare |

### LFM (arch 11) — lowered forward and evidence scope

| Variable | Status | Scope |
|---|---|---|
| `HIPFIRE_LFM2_GRAPH` | LFM graph experiments | Source in LFM crate/daemon |
| `HIPFIRE_FORWARD_LOWERED` | default on (LFM and several other arches); `=0` opts out | Shared lowered forward escape hatch. For LFM: **unset or any value other than `0`** keeps the lowered path; **`=0`** forces the legacy hand loop. Recorded on the sealed LFM [`admissions.yml`](admissions.yml) evidence row; **not** an automatic Redline runtime-default opt-out. |
| Other `HIPFIRE_LFM*` | diag/trace | Inventory |

`HIPFIRE_LFM2_PREFILL_BATCH`, `HIPFIRE_LFM2_PREFILL_MAX_BATCH`, and `HIPFIRE_LFM2_GFX1201_DECODE_FUSION` have **no reader under `crates/`** in this tree: setting them changes nothing, LFM prefill is the eager loop, and there is no batched-prefill or decode-fusion selector. They appear only in design/planning docs, the sealed [`admissions.yml`](admissions.yml) evidence row (historical `explicit_opt_outs`, not runtime opt-outs), and branch-pinned wording (`lfm-redline@692a726dde53508cb53de1a74c720e75a7c9f33e`, absent from `origin/beta@202282de…`). [`admissions.yml`](admissions.yml) remains the sole product-admission authority (schema v2; exactly one earned retained-PM4 product row).

### CASK (deprecated, removal in 0.5.0) / serve / multi-GPU

CASK / TriAttention eviction is deprecated and will be removed in 0.5.0; it is
not supported. `HIPFIRE_CASK_SIDECAR` is the legacy env alias of
`memory.cask.sidecar` and goes with it; setting it (or any CASK key) makes the
daemon print a deprecation warning at
load. `HIPFIRE_CASK_OFF` is a retired compatibility name that the Rust control
plane does not consume; CASK is already off unless `memory.cask.sidecar` is
set or `memory.cask.auto_attach=true`. The old name remains only in developer
harness exports pending their cleanup.

| Variable | Notes |
|---|---|
| `HIPFIRE_IDLE_TIMEOUT` | Serve idle unload seconds |
| `HIPFIRE_MAX_REQUEST_BYTES` | Body cap |
| `HIPFIRE_SERVE_MAX_QUEUE` / `HIPFIRE_SERVE_QUEUE_TIMEOUT_MS` | Admission queue |
| `HIPFIRE_EXPERIMENTAL_BUDGET_ALERT` | Research budget nudge |
| `HIPFIRE_FA_PERTOKEN_MIN_CTX` | Context length past which an exact-gfx1100, exact-gfx1201 or (opt-in) exact-gfx1151 Q8 small-batch (n = 4..32, head_dim 128/256, sequential non-tree, HIP graph capture off, retained replay recording off) attend step leaves the batched flash kernel for the multi-row tile; default `4096` on gfx1100/gfx1201, unset on gfx1151 (setting any value > 0 opts gfx1151 in), `0` disables the route. Other arches, KV modes, shapes, and semantics retain the batched route. |
| `HIPFIRE_RCCL_LIB` | Explicit `librccl.so` path, tried before the ROCm root. For distributions whose ROCm prefix does not carry RCCL (nixpkgs: `rocmtoolkit-merged` has HIP/HSA, `librccl` is a separate store path). |
| `HIPFIRE_DEVICES` / `HIPFIRE_DEVICE` / `HIPFIRE_TP` / `HIPFIRE_TP_USE_RCCL` | Multi-GPU / TP. `HIPFIRE_DEVICES` (singular alias `HIPFIRE_DEVICE`; conflicting values fail) is the compatibility spelling of `hardware.devices`: comma-separated, in logical order, each entry a PCI-order index (`rocm-smi` order, not ROCr/HIP ordinals), `gfxNNNN` (first free card of that arch; repeat for more), `GPU-<uuid>`, or PCI address `[DDDD:]BB:DD.F` (required for no-UUID cards). Resolved from the KFD topology without touching a GPU, reserved, lowered to ROCr UUIDs/ordinals plus HIP `0..N-1`, and checked against HIP's arch and PCI bus ID after init. See [multi-gpu.md](multi-gpu.md#device-selection). |
| `HIPFIRE_EMULATE_GPUS` | **Developer-only logical GPU emulation.** A successfully parsed integer `>=2` enables; missing, malformed, `0`, and `1` disable. Requested logical IDs are aliased modulo the loaded physical count; this switch does not choose the EP rank count and is not a product admission. |
| `HIPFIRE_EP_PEER_ALLREDUCE_DECODE=1` | Explicit peer all-reduce selector for logical EP decode proofs; do not set it to `0`. |
| `HIPFIRE_EP_PEER_ALLREDUCE=1` | Explicit peer all-reduce selector for logical EP batched prefill/tick proofs; do not set it to `0`. |
| `HIPFIRE_ALLOW_MIXED_ARCH=1` | Mixed arch pairs |
| `HIPFIRE_PP_LAYERS` / `HIPFIRE_PP_PFLASH` | Pipeline parallel |
| `HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB` | Uniform init tolerance |
| `HIPFIRE_SLOTS_PAGED` / `HIPFIRE_SLOTS_PAGED_PAGES` | Multi-slot engine: `1`/`on`/`true` selects the paged KV pool instead of fixed per-slot slabs (implied by `serve.prefix_cache`); `_PAGES` overrides the physical page count (clamped to ≥ 1). |
| `HIPFIRE_VL_SEQUENTIAL` | Multi-slot engine: `1`/`on`/`true` forces the legacy one-image-at-a-time VL prefill. Refused together with the paged pool. |
| `HIPFIRE_PAGE_EVICTION` | `0` keeps model pages in the host page cache after upload (default: evict only on UMA). Shared by the sequential loader and the multi-slot engine. |
| `HIPFIRE_BENCH_MULTI_SLOT` / `HIPFIRE_BENCH_MULTI_SLOT_SLOTS` / `HIPFIRE_BENCH_MULTI_SLOT_CTX` | `hipfire bench`: load the model on the multi-slot engine with the given slot count and per-slot context. |
| `HIPFIRE_MTP_TRACE=1` | Per-cycle MTP draft/verify trace on stderr. |
| `HIPFIRE_DEBUG_POOL_INVARIANTS=1` | Multi-slot engine: assert page-pool / slot-lease invariants after every scheduler tick (debug; slow). |
| `HIPFIRE_FAULT_HIP` / `HIPFIRE_FAULT_PREFIX_PUBLISH` / `HIPFIRE_FAULT_MTP_FULL_REJECT` | Test-only fault injection used by `test_serve_prefix_cache`: `HIP` fails one `upload`/`launch`/`sync` below the HIP bridge, `PREFIX_PUBLISH=1` fails the first prefix-cache publication, `MTP_FULL_REJECT=1` forces every MTP draft to be rejected. Never set in production. |

#### Logical GPU emulation (developer-only)

`HIPFIRE_EMULATE_GPUS` is an environment-only diagnostic switch captured in
the startup snapshot. It is deliberately not a user-facing CLI or persistent
TOML/config key. The value is parsed as `usize`: only a successfully parsed
integer `>=2` enables emulation. Missing, malformed (including negative
values), `0`, and `1` leave emulation disabled.

When enabled, each requested logical device ID is resolved modulo the loaded
physical-device count:

| Physical devices loaded | Requested logical IDs | Resolved physical IDs |
|---|---|---|
| 1 | `[0, 1, 2, 3]` | `[0, 0, 0, 0]` |
| 2 | `[0, 1, 2, 3]` | `[0, 1, 0, 1]` |

The switch is an enable switch, not a rank-count setting. `Gpus::init_ep(2,
...)` owns a two-logical-rank EP layout, and `Gpus::init_ep(4, ...)` owns a
four-logical-rank layout. Duplicate physical IDs remain rejected in ordinary
mode; the snapshotted emulation state is the explicit diagnostic exception for
sealed, load, and batch admission.

For a one-device proof, leave `HIPFIRE_DEVICES` unset and set
`HIP_VISIBLE_DEVICES=0`; set both peer selectors above explicitly to `1`.
Leave `HIPFIRE_TP_USE_RCCL` unset—`=0` is not a host all-reduce fallback.
The complete route-oracle workflow and evidence boundary are in
[`multi-gpu.md`](multi-gpu.md#logical-gpu-emulation-developer-only).

### Redline / retained replay

Policy owner: [`REDLINE.md`](REDLINE.md) (**shipped / ref-pinned**). Timing is not route-certified without same-report timed-arm proof. Registry admission is not canonical certification.

| Variable | Notes |
|---|---|
| `HIPFIRE_REPLAY_BACKEND` | `hip` / `off` / `shadow` / `auto`. Unset may select `auto` only from the automatic product defaults in `retained_redline_default`: `mq4r_redline_default` — exact GPU arch `gfx1100`/`gfx1151`/`gfx1201` + case-insensitive `.mq4r` + pp=tp=1 (model-family agnostic; no `arch_id` gate; `gfx1200` and all other arches remain opt-in); Qwen3.5 dense (`qwen3_5`, any weight format) plain-AR decode on exact `gfx1201` with pp=tp=1 and no drafter (retained PM4; byte-identical to the HIP AR graph); Qwen4 (Flash-Next, `qwen4`) plain-AR decode on exact `gfx1201` with pp=tp=1 and no drafter, i.e. an MTP-off load (retained PM4, unpaced; greedy text identical to the HIP route); and DeepSeek4 `.mq2r` AR on gfx1151. Existing LFM `.mq4` registry evidence is **not** automatically selected because it is not `.mq4r`; any usable non-default retained route is explicit opt-in and must still prove route support. The sealed LFM [`admissions.yml`](admissions.yml) row is registry evidence/admission only and does not wire runtime defaults. Runtime default ≠ Redline certification/registry admission. Built-in `hip` config profile, another explicit backend selection, `replay.backend = "hip"` or `=hip` disables the automatic default. |
| `HIPFIRE_GFX1201_PM4_PACING` | NOP pacing of the retained gfx1201 Qwen3.5-dense decode PM4 tape (`replay.gfx1201_pm4_pacing`): `auto` (default) = one 64-body-dword `NOP` after every `DISPATCH_DIRECT`; `off`/`0` = unpaced tape; `nop:N` = N-body-dword NOP. Applied only when the Redline default admits exact gfx1201 + `qwen3_5` (not MQ4R, not MoE, not gfx11). NOPs write no register or memory, so decode is byte-identical. railgun's gfx1201 lowering (`HIPFIRE_RAILGUN_BACKEND=railgun`) paces the same way by its per-arch default; an explicit `off`/`nop:N` applies to it too |
| `HIPFIRE_REPLAY_TRANSPORT` | `pm4` / AQL family |
| `HIPFIRE_UNSAFE_WSL_REDLINE` | **Unsafe until certified.** `replay.unsafe_wsl_redline`, default off. Under WSL2/ROCDXG (`/dev/dxg` present, `/dev/kfd` absent) the retained Redline PM4 default falls back to the HIP graph with one `[redline] retained default refused` log line, and an explicit `replay.backend = "redline"`/`"shadow"` makes the daemon exit at startup. `1` lifts both. No effect on native Linux; native Windows (no ROCr) stays refused. |
| `HIPFIRE_UNSAFE_WSL_VMM_KV` | **Unsafe until certified.** `memory.unsafe_wsl_vmm_kv`, default off. Under WSL2/ROCDXG automatic KV selects `legacy` (reason in `kv_backend_reason`) and an explicit `kv_backend = "vmm"` is refused, because WDDM VA growth may alias earlier KV segments as it does on native Windows. `1` lifts it. Native Windows stays legacy-only. |
| `HIPFIRE_REPLAY_MANUAL_CAPTURE` | Manual capture delimiters |
| `HIPFIRE_REPLAY_PM4_*` | PM4 research knobs — inventory |
| `HIPFIRE_REDLINE_GAP_TIMING` | Developer-only diagnostic: `1` = per-token host breakdown of the Qwen3.5 plain-AR decode step (embedding, position copy, retained-PM4 patch, submit+wait with its in-IB GPU span, or the HIP-graph launch, and the host time between consecutive steps), one `[gap-timing]` stderr summary per run of consecutive decode positions. Unset reads no clocks. |
| `HIPFIRE_GFX1100_PM4_EXPERIMENTS` | Developer-only diagnostic: `1` lets the gfx1151 `HIPFIRE_GFX1151_PM4_INITIATOR` / `_INTERLEAVE` / `_RESOURCE_LIMITS` knobs apply to gfx1100 too. Unset, gfx1100 keeps its legacy encoding byte for byte. |
| `HIPFIRE_RAILGUN_SHADOW` | Developer-only railgun M1 shadow: a `railgun-jit-corpus` directory. railgun authors its program next to every Redline recording from the corpus facts and diffs its prepared PM4 lowering against the gfx12 tape; never submitted, route unchanged. See REDLINE.md §5 "railgun shadow diff". |
| `HIPFIRE_RAILGUN_SHADOW_OUT` | Directory for the shadow's JSON reports (one per prepare). |
| `HIPFIRE_RAILGUN_INVENTORY` | `railgun-cert recording-inventory json` output; the matched route's recording-dependent decisions go into the shadow report. |
| `HIPFIRE_RAILGUN_CACHE_TABLE` | railgun M2: a G4 probe `cache-table.<arch>.json` (`cargo run -p railgun --features g4-probe --example g4_probe`). Attaches silicon receipts to the cache-table rows whose tier-D rung the probe ran clean for ≥2²⁰ trials; a row whose own rung the probe saw stale fails the prepare. |
| `HIPFIRE_RAILGUN_BACKEND` | railgun M2, developer-only: `railgun` makes the gfx12 single-IB prepared tape execute railgun's lowering (needs `HIPFIRE_RAILGUN_SHADOW`); any refusal fails the prepare closed to HIP, never to the Redline planner. |
| `HIPFIRE_RAILGUN_CHECK` | railgun M2 check mode, developer-only: `off` (default) / `sample(N)` / `always`. Runs the prepared Qwen3.5 or Qwen4 PM4 lowering and its HIP twin on the same state and byte-compares every surface: `always` = every live `State` allocation of the `hip_bridge::registry` allocator registry (the immutable `Weights`, which both loaders register under that role, digested before and after the first checked step of each prepared program), `sample(N)` = one step in N, the program's written allocations only. A difference poisons the route; the twin's result stands. The registry records allocations only while this (or `HIPFIRE_RAILGUN_DIGEST=1`) is set. |
| `HIPFIRE_RAILGUN_CHECK_OUT` | railgun M2: file receiving one JSON line per replayed step (check result, phase timings, digests). |
| `HIPFIRE_RAILGUN_NEGATIVE_CONTROL` | railgun M2 check-mode negative controls, developer-only: `flip_byte` XORs one byte of one surface into the PM4 arm's result at every checked step, which must come back unequal on exactly that byte (`negative_control.caught` in `HIPFIRE_RAILGUN_CHECK_OUT`); `drop_boundaries` executes railgun's lowering without any mid-segment wait/acquire (with `HIPFIRE_RAILGUN_BACKEND=railgun`), so `HIPFIRE_RAILGUN_CHECK` must report a difference. Never for serving. |
| `HIPFIRE_RAILGUN_DIGEST` | railgun M2: `1` = after every replayed step, digest every live `State` allocation (paired by allocation order) into `HIPFIRE_RAILGUN_CHECK_OUT`; the G2 third arm compares these between a railgun and a pre-railgun process. |
| `HIPFIRE_REPLAY_ROUTE_PROOF_LOG` | Developer-only / one-shot compat for `diagnostic.replay.route_proof_log`. When `1`/`true`/`on` (or TOML `true`), the daemon emits one post-generate retained-route proof marker per successful request: `HIPFIRE_REPLAY_ROUTE_PROOF transport=<name> position=<n> request_id=<id> replays=<count>`. Off by default; product coherence smoke enables it only via temporary serve_harness `config.toml`, not ambient env. |

### Chat template

| Variable | Notes |
|---|---|
| `HIPFIRE_CHAT_TEMPLATE_FILE` | External jinja path |
| `HIPFIRE_DEFAULT_CHATML` | Only load-bearing at `0` (disable ChatML fallback) |
| `HIPFIRE_JINJA_CHAT` | Serve jinja path toggle |
| `HIPFIRE_QWEN35_GRAMMAR` | Tool grammar |

### Perf / diag (common)

| Variable | Notes |
|---|---|
| `HIPFIRE_DPM_WARMUP_SECS` | DPM warmup before timing |
| `HIPFIRE_HOST_TIMING=1` | Host timing JSON |
| `HIPFIRE_PROFILE` / `HIPFIRE_PROFILE_DECODE` / `HIPFIRE_PROFILE_CYCLES` | Profiling |
| `HIPFIRE_DETERMINISTIC` | Determinism toggles in dispatch |
| `HIPFIRE_DS4_DENSE_ACT_DIR` | DeepSeek4 calibration-only dump of P1 projection inputs in `collect_e8_hessian` format; direct evaluator flag `--dump-dense-acts` is preferred. |
| `HIPFIRE_HIPCC_EXTRA_FLAGS` | Compatibility alias for `diagnostic.compiler.hipcc_extra_flags` |
| `HIPFIRE_KERNEL_CACHE` | Kernel cache dir (`var_os`) |
| `HIPFIRE_NO_DEVICE_COMPILER=1` | Require verified installed kernel objects instead of JIT; a missing/stale index, wrong symbol/source/flags/profile/ABI/toolchain identity or object SHA-256 fails before HIP loads it. Hot JIT keys remain toolchain-specific. |
| `HIPFIRE_*_DUMP` / `*_TRACE` / `*_PROFILE` | Diagnostic families — see inventory |


For a full indexed install, run `scripts/compile-kernels.sh gfx1201` from the
matching source revision with hipcc available, and install the resulting
`kernels/compiled/gfx1201/` directory beside the daemon executable. Native
installers and container/Nix builds perform this registry packaging directly.
Do not copy bare `.hsaco` or `.hash` files from older installations.

To build a compiler-free `gfx1201` RMSNorm package for the production
`Gpu::rmsnorm_f32` route, run
`hipfire-kernel-pack --arch gfx1201 --output <daemon-bin-dir>/kernels/compiled/gfx1201 --extra-flags '-DIU4_A4_CANDIDATES=2' --kernel rmsnorm_f32:rmsnorm_f32:kernels/src/rmsnorm.hip`
with the selected ROCm hipcc installed. The tool writes `rmsnorm_f32.hsaco`,
`rmsnorm_f32.hash` (portable cold key), and `rmsnorm_f32.index.json` (versioned
source/flags/profile/ABI/symbol/toolchain/object-SHA record). Compiler flags
must match the daemon's active `FeatureFlags` (including arch defaults);
the exact-source registry exporter emits these flags for `--registry` builds.
A compiler-free `daemon --precompile --module rmsnorm_f32` probes the real
`Gpu::rmsnorm_f32` kernel load and verifies numerical output. The installed
objects must be beside the actual daemon executable, not merely in the CWD;
the cache override controls only writable hot JIT entries.
Kernel-selector and arch-specific `HIPFIRE_GFX*` / `HIPFIRE_RDNA*` /
`HIPFIRE_MOE_*` levers are **research/power-user**. Centralized
`FeatureFlags` controls now have typed TOML keys under `kernel` or
`diagnostic.kernel`; defaults and safety remain source-defined and none are
registry-authorized.

---

## Manual — top-level doc references

The canonical documentation checker requires every `HIPFIRE_*` token in `AGENTS.md`, `README.md`, and `CONTRIBUTING.md` to appear in this file. Tokens historically routed from those surfaces (keep listed even if a root file is later thinned):

`HIPFIRE_ATTN_FLASH`, `HIPFIRE_DDTREE_` (prefix family; concrete vars in inventory), `HIPFIRE_DFLASH_DRAFT`, `HIPFIRE_GRAPH`, `HIPFIRE_HOST_TIMING`, `HIPFIRE_KV_MODE`, `HIPFIRE_LM_HEAD_F16`, `HIPFIRE_LOCAL`, `HIPFIRE_NORMALIZE_PROMPT`, `HIPFIRE_PROMPT_HEAT_JSON`, `HIPFIRE_PROMPT_HEAT_LIMIT`, `HIPFIRE_PROMPT_TOKEN_HEAT`, `HIPFIRE_VERIFY_GRAPH`.

---

## Manual — TOML key → legacy environment compatibility

Compatibility map from the `hipfire-config` schema. TOML is the supported
persistent surface. The native CLI sends typed policy directly to the daemon;
it does not project these values into the child environment. The names below
are accepted only as compatibility/one-shot input to the resolver. Message-
field-only keys (temperature, thinking, …) have no compatibility alias.

Copyable user, developer, and retained-PM4 TOML profiles are in
[`docs/configs/`](configs/README.md).

| Config key | Env |
|---|---|
| `kv_cache` | `HIPFIRE_KV_MODE` |
| `flash_mode` | `HIPFIRE_ATTN_FLASH` |
| `prompt_normalize` | `HIPFIRE_NORMALIZE_PROMPT` |
| `dflash_ngram_block` | `HIPFIRE_DFLASH_NGRAM_BLOCK` |
| `experimental_budget_alert` | `HIPFIRE_EXPERIMENTAL_BUDGET_ALERT` |
| `max_total_think_tokens` | `HIPFIRE_MAX_TOTAL_THINK_TOKENS` |
| `memory.gpu_layer_budget` | `HIPFIRE_GPU_LAYER_BUDGET` |
| `memory.offload_exec` | `HIPFIRE_OFFLOAD_EXEC` |
| `mtp_mode` / `mtp_k` | `HIPFIRE_MTP_MODE` / `HIPFIRE_MTP_K` |
| `speculation.mtp_ngram` | `HIPFIRE_MTP_NGRAM` |
| `speculation.mtp_sampled` | `HIPFIRE_MTP_SAMPLED` |
| `chat_template` | `HIPFIRE_CHAT_TEMPLATE_FILE` |
| `default_chatml=false` | `HIPFIRE_DEFAULT_CHATML=0` |
| `speculation` | `HIPFIRE_SPECULATION` |
| `default_model` / serve model | `HIPFIRE_MODEL` |
| `idle_timeout` | `HIPFIRE_IDLE_TIMEOUT` |
| `max_request_bytes` | `HIPFIRE_MAX_REQUEST_BYTES` |
| `serve_max_queue` | `HIPFIRE_SERVE_MAX_QUEUE` |
| `serve_queue_timeout_ms` | `HIPFIRE_SERVE_QUEUE_TIMEOUT_MS` |
| `serve.local` | `HIPFIRE_LOCAL` |
| `serve.multi_slot` | `HIPFIRE_SERVE_MULTI_SLOT` |
| `serve.multi_slot_slots` | `HIPFIRE_SERVE_MULTI_SLOT_SLOTS` |
| `serve.multi_slot_ctx` | `HIPFIRE_SERVE_MULTI_SLOT_CTX` |
| `serve.multi_slot_prefill_chunk` | `HIPFIRE_SERVE_MULTI_SLOT_PREFILL_CHUNK` |
| `prefill_*` | matching `HIPFIRE_PREFILL_*` |
| `mmq_screen*` | `HIPFIRE_MMQ_SCREEN*` |
| `hardware.devices` | `HIPFIRE_DEVICES` / `HIPFIRE_DEVICE`; resolves index / `gfxNNNN` / `GPU-<uuid>` / PCI entries to physical cards and synchronizes `ROCR_VISIBLE_DEVICES=<their ROCr selectors>` with `HIP_VISIBLE_DEVICES=0..N-1` before GPU initialization |
| `hardware.allow_mixed_arch` | `HIPFIRE_ALLOW_MIXED_ARCH` |
| `hardware.tp_use_rccl` | `HIPFIRE_TP_USE_RCCL` |
| `hardware.uniform_vram_tolerance_gb` | `HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB` |
| `generation.loop_guard_threshold` / `generation.loop_guard_window` | `HIPFIRE_NGRAM_LOOP_THRESHOLD` / `HIPFIRE_NGRAM_WINDOW` |
| `kernel.flash_partials_batch` | `HIPFIRE_FLASH_PARTIALS_BATCH` |
| `kernel.lm_head_f16` | `HIPFIRE_LM_HEAD_F16` |
| `diagnostic.prompt_*` | matching `HIPFIRE_PROMPT_*` |
| `diagnostic.kernel.*` / `diagnostic.compiler.*` | schema-declared kernel/compiler compatibility aliases; enumerate with `hipfire config schema --json` |

---

## Manual — NPU research stack (dark, not `HIPFIRE_*`)

Read only by `pm-npu`, `railgun` (feature `npu`) and `npu-tools`; nothing on the daemon/engine path reads them. See [`npu/README.md`](npu/README.md).

| Variable | Reader | Meaning |
|---|---|---|
| `NPU_FCLK_GUARD` | `railgun::npu::fclk` (npu-coop, coop27) | `require` (unset; refuse concurrent GPU+NPU work unless the Halo iGPU fabric clock is pinned), `pin` (pin for the process, restore on exit), `off` (skip, loud warning) |
| `NPU_IGPU_PCI` | `railgun::npu::fclk` (fclk sysfs device, npu-coop GPU) | PCI bus id of the Halo iGPU, default `0000:bf:00.0` (hipx); other Halo boxes enumerate it elsewhere (e.g. `0000:c5:00.0`) |
| `NPU_M4_CMD`, `NPU_M4_ALONE_CMD` | `npu-tools m4` | NPU loop command run concurrently with the GPU prefills (required) and alone (defaults to `NPU_M4_CMD`) in the M4 derate measurement |
| `AIE2P_VENDOR_CORPUS` | `pm-npu` tests | Vendor `vendor-artifacts` directory for byte-oracle cross-checks; tests skip when unset |
| `AIE2P_OBJDUMP` | `pm-npu` `isa_decode` (ignored test) | llvm-aie `llvm-objdump` used as the decode oracle (default `/tmp/aie2p-llvm-objdump`) |
| `NPU_HELLO_ARTIFACT_DIR`, `NPU_HELLO_PROGRAM` | `pm-npu` sim tests | Dumped `npu-hello` artifacts for the exact-PDI hello gates; skipped when unset |
| `GOLDEN_PRINT`, `IEF15_PROBE_DUMP` | `pm-npu` `golden_bytes` / `ief15_probe` tests | `GOLDEN_PRINT`: also print the computed golden rows; `IEF15_PROBE_DUMP=<path>`: write the expected probe blob |
| `NPU_WINDOW_TIMEOUT`, `NPU_WINDOW_LOG`, `NPU_FCLK_TOOL`, `NPU_PROBE_OUT`, `NPU_BATCH_BASIC`, `NPU_METRICS_OUT`, `NPU_BISECT_OUT`, `NPU_BISECT2_OUT`, `NPU_RAILGUN_PHASE`, `NPU_RING_PHASE`, `NPU_EXPERTS_*`, `NPU_M3_*`, `NPU_V8_OUT`, `NPU_LEVELS`, `NPU_LOOP_S` | `tools/npu/*.sh` | Silicon-window harness knobs (timeouts, log/output dirs, phases); documented in each script header |

---

## Manual — Cargo features (not env)

`crates/hipfire-runtime/Cargo.toml` default features (checked 2026-07-19):

`arch-qwen35`, `arch-qwen35-vl`, `arch-llama`, `arch-qwen2`, `arch-deepseek4`, `arch-cohere2moe`, `arch-dots-ocr`, `deltanet`.

`hipfire-arch-lfm2moe` / `hipfire-arch-minimax` are linked as ordinary dependencies on the loader/runtime graph (not named in that default feature list). Feature toggles are build-time, not `HIPFIRE_*` env.

---

## Lifecycle status

Every variable in the generated inventory below carries one lifecycle status,
derived by `scripts/check-lifecycle.py` (wired into `scripts/no-gpu-ci.sh`):

| Status | Meaning |
|---|---|
| `stable` | Compatibility alias of a non-experimental schema key ([`CONFIG.md`](CONFIG.md#lifecycle-status)), or a bootstrap path/URL variable read before config exists. Supported. |
| `experimental` | Compatibility alias of an experimental schema key. May change without notice. |
| `developer` | Read by product code through the `[developer]` namespace (`HIPFIRE_FOO` → `developer.foo`). Unsupported, for experiments only. |
| `harness` | Only referenced by scripts, examples, tests or tools, never by the product binaries. |
| `deprecated` | Deprecated since 0.4.0, removed in 0.5.0: the alias of a deprecated schema key, a deprecated knob, or only referenced by deprecated code. Using it prints a one-line warning. |

A `; …` suffix records deprecated **values** of a supported variable
(`HIPFIRE_KV_MODE=asymN|turbo*`, `HIPFIRE_DFLASH_WINDOW=0`). The 0.4.0
deprecations are listed in [`CHANGELOG.md`](../CHANGELOG.md).

---

## Generated inventory

**Do not hand-edit rows below**; regenerate with `python3 scripts/check-lifecycle.py --write`.
Presence in the inventory means the token appears in source; it does **not** mean the knob is supported, stable, or admitted.

<!-- env-inventory:begin (generated by scripts/check-lifecycle.py --write) -->

**Generation method:** token scan over tracked `*.rs`, `*.py`, and `*.sh` (`scripts/check-lifecycle.py --write`).
**Columns:** variable; up to two lexical source paths; lifecycle status (see [Lifecycle status](#lifecycle-status)).
**Count:** 1408

| Variable | Example source path(s) | Lifecycle |
|---|---|---|
| `HF_ENDPOINT` | crates/hipfire-cli/src/main.rs, crates/hipfire-config/src/lib.rs | stable |
| `HIPFIRE_6409_EVIDENCE` | crates/radiowave/src/recipes.rs | developer |
| `HIPFIRE_9B_MODEL` | scripts/bisect_9b_decode.sh | harness |
| `HIPFIRE_A4_ATTN_EPI` | crates/hipfire-arch-qwen35/src/qwen35/prefill.rs, crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_A4_FA_GATE_IL` | crates/hipfire-arch-qwen35/src/qwen35/prefill.rs | developer |
| `HIPFIRE_A4_HIN_TOKFAST` | crates/rdna-compute/src/gemm.rs, crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_A4_RMS_FDIV` | crates/hipfire-arch-qwen35/src/qwen35/load.rs, crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_A4_SLAB` | crates/hipfire-arch-qwen35/src/qwen35/prefill.rs, crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_A8_APF_K32` | crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_A8_FUSED_PROD` | crates/rdna-compute/src/feature_flags.rs | developer |
| `HIPFIRE_A8_PREFILL` | crates/rdna-compute/src/feature_flags.rs | developer |
| `HIPFIRE_ABORT_EVIDENCE_DIR` | scripts/test-ds4-heterogeneous-abort-resume.sh | harness |
| `HIPFIRE_ADAPTIVE_B_DOWN` | crates/hipfire-runtime/examples/dflash_spec_demo.rs | harness |
| `HIPFIRE_ADAPTIVE_B_UNSAFE` | crates/hipfire-runtime/examples/dflash_spec_demo.rs | harness |
| `HIPFIRE_ADAPTIVE_B_UP` | crates/hipfire-runtime/examples/dflash_spec_demo.rs | harness |
| `HIPFIRE_AGENTIC_GATE_NO_VRAM_CHECK` | scripts/agentic-gate.sh | harness |
| `HIPFIRE_AGENTIC_GATE_OUT` | scripts/agentic-gate.sh | harness |
| `HIPFIRE_ALLOW_BF16_QUANTIZED_TARGET` | crates/hipfire-arch-qwen4/src/artifact.rs | developer |
| `HIPFIRE_ALLOW_FMT` | scripts/no-cargo-fmt-guard.py, scripts/test-no-cargo-fmt-guard.py | harness |
| `HIPFIRE_ALLOW_MIXED_ARCH` | crates/hipfire-config/src/lib.rs, crates/hipfire-runtime/src/config.rs | stable |
| `HIPFIRE_ALLOW_MQ2` | crates/hipfire-quantize/src/pipeline.rs | developer |
| `HIPFIRE_ALLOW_MQ2_LLOYD` | crates/hipfire-quantize/src/pipeline.rs | developer |
| `HIPFIRE_ALLOW_MQ3_LLOYD` | crates/hipfire-quantize/src/pipeline.rs | developer |
| `HIPFIRE_ALLOW_MQ4_LLOYD` | crates/hipfire-quantize/src/pipeline.rs | developer |
| `HIPFIRE_ALLOW_UNIT_IMATRIX` | crates/hipfire-quantize/src/pipeline.rs | developer |
| `HIPFIRE_API_KEY` | scripts/lmx_continuous_batch.py | harness |
| `HIPFIRE_AR_GRAPH` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_AR_GRAPH_TRACE` | crates/hipfire-arch-qwen35/src/qwen35/forward.rs, scripts/vmm_kv_matrix.py | developer |
| `HIPFIRE_ATTENTION_REDUCE_GATED_MQ_AWQ` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_ATTENTION_REDUCE_GATED_MQ_KERNEL` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_ATTN_FLASH` | crates/hipfire-arch-qwen35/src/qwen35/forward.rs, crates/hipfire-config/src/lib.rs | stable |
| `HIPFIRE_ATTN_QRESIDENT` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | stable |
| `HIPFIRE_ATTN_QRESIDENT_V2` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | stable |
| `HIPFIRE_ATTN_TILE_SIZE` | crates/hipfire-arch-qwen35/src/qwen35/forward.rs, crates/hipfire-runtime/src/llama.rs | developer |
| `HIPFIRE_AWQ_EXPERTS` | crates/hipfire-quantize/src/pipeline.rs | developer |
| `HIPFIRE_AWQ_F1_ONLY` | crates/hipfire-quantize/src/calibration.rs, scripts/awq_alpha_sweep.sh | developer |
| `HIPFIRE_A_OUT` | scripts/ab-dispatch-validation.sh | harness |
| `HIPFIRE_A_REF` | scripts/ab-dispatch-validation.sh | harness |
| `HIPFIRE_BASELINE_ARCH` | crates/hipfire-runtime/examples/coherence_probe.rs, scripts/kernel_atlas.py | harness |
| `HIPFIRE_BASELINE_FORMATS` | scripts/baseline_quant_smoke.sh | harness |
| `HIPFIRE_BASELINE_KV_MODE` | scripts/baseline_quant_smoke.sh | harness |
| `HIPFIRE_BASELINE_MAX_GEN` | scripts/baseline_quant_smoke.sh | harness |
| `HIPFIRE_BASELINE_MODEL` | scripts/baseline_quant_smoke.sh | harness |
| `HIPFIRE_BASELINE_NAME` | scripts/baseline_quant_smoke.sh | harness |
| `HIPFIRE_BASELINE_OUT` | scripts/baseline_quant_smoke.sh | harness |
| `HIPFIRE_BASELINE_PROMPT` | scripts/baseline_quant_smoke.sh | harness |
| `HIPFIRE_BASELINE_PROMPT_MODE` | scripts/baseline_quant_smoke.sh | harness |
| `HIPFIRE_BASELINE_WIDE_MARGIN` | scripts/baseline_quant_smoke.sh | harness |
| `HIPFIRE_BATCHED_PREFILL` | crates/hipfire-arch-gemma4/examples/prefill_parity_gemma4.rs, crates/hipfire-arch-gemma4/src/lowered.rs | developer |
| `HIPFIRE_BENCH_AB` | benchmarks/scripts/bench_dflash_27b_gfx906.sh | harness |
| `HIPFIRE_BENCH_MAX` | scripts/adaptive_b_bench.sh, scripts/qwen36_bench.sh | harness |
| `HIPFIRE_BENCH_N` | crates/rdna-compute/examples/bench_indexed_moe_keystone.rs | harness |
| `HIPFIRE_BENCH_RUNS` | scripts/adaptive_b_bench.sh, scripts/bench_qwen36_ar_dflash.sh | harness |
| `HIPFIRE_BF16_H_A4` | crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_BF16_H_FP8` | crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_BIN` | scripts/calibrate_multigpu.sh, scripts/install.sh | harness |
| `HIPFIRE_BLOB_FORCE` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/dispatch.rs | experimental |
| `HIPFIRE_BLOCK_I4_128_QUANT_NO_STANDALONE` | crates/rdna-compute/examples/qwen4_moe_sym.rs, crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_BLOCK_I8_128_QUANT_NO_STANDALONE` | crates/rdna-compute/src/gemv.rs | developer |
| `HIPFIRE_BQ1G128_XBATCH` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_BQ1G128_XBATCH_KERNEL` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_BQ1G128_XBATCH_MAX` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_BQ1_MODEL` | benchmarks/quality-baselines/harness/spe_ablation.sh | harness |
| `HIPFIRE_BRANCH` | scripts/mi300x_bootstrap.sh | harness |
| `HIPFIRE_BUILDER_GIT_SHA` | crates/hipfire-isa/src/lib.rs | developer |
| `HIPFIRE_BUILD_COMMIT` | crates/hipfire-cli/build.rs, crates/hipfire-cli/src/main.rs | developer |
| `HIPFIRE_BUILD_COMMIT_OVERRIDE` | crates/hipfire-cli/build.rs | harness |
| `HIPFIRE_BUILD_DIRTY` | crates/hipfire-cli/build.rs, crates/hipfire-cli/src/main.rs | developer |
| `HIPFIRE_BUILD_REF` | crates/hipfire-cli/build.rs, crates/hipfire-cli/src/main.rs | developer |
| `HIPFIRE_BUILD_REF_OVERRIDE` | crates/hipfire-cli/build.rs | harness |
| `HIPFIRE_BUILD_TARGET` | crates/hipfire-cli/build.rs, crates/hipfire-cli/src/main.rs | developer |
| `HIPFIRE_BUILD_VERSION` | crates/hipfire-cli/build.rs, crates/hipfire-cli/src/main.rs | developer |
| `HIPFIRE_B_REF` | scripts/ab-dispatch-validation.sh | harness |
| `HIPFIRE_C2M_DUMP_PROMPT` | crates/hipfire-generate/src/dense.rs | developer |
| `HIPFIRE_C2M_EMPTY_TURN_GUARD` | crates/hipfire-arch-cohere2moe/src/spec_emit.rs, crates/hipfire-generate/src/dense.rs | developer |
| `HIPFIRE_C2M_NORMDUMP` | crates/hipfire-arch-cohere2moe/src/forward.rs | developer |
| `HIPFIRE_CACHE_CKPT_INTERVAL` | crates/hipfire-arch-qwen35/src/dflash_spec.rs, crates/hipfire-arch-qwen35/tests/mtp_cache_byte_identity.rs | developer |
| `HIPFIRE_CACHE_CKPT_MAX` | crates/hipfire-arch-qwen35/src/dflash_spec.rs, crates/hipfire-config/src/lib.rs | developer |
| `HIPFIRE_CACHE_CKPT_RESUME` | crates/hipfire-generate/src/ar.rs | developer |
| `HIPFIRE_CALIB_BF16` | crates/hipfire-arch-lfm2moe/src/lfm2moe.rs, crates/hipfire-arch-muse-glimmer/src/batch.rs | experimental |
| `HIPFIRE_CALIB_F64_AUDIT` | crates/hipfire-runtime/src/calibration.rs | developer |
| `HIPFIRE_CALIB_HESSIAN_STORAGE` | crates/hipfire-runtime/src/calibration.rs | developer |
| `HIPFIRE_CALIB_PROFILE` | crates/hipfire-runtime/examples/triattn_validate.rs | harness |
| `HIPFIRE_CANARY_MODEL` | scripts/gfx906_fallback_canary.sh | harness |
| `HIPFIRE_CANARY_PREFILL` | scripts/gfx906_fallback_canary.sh | harness |
| `HIPFIRE_CANARY_RUNS` | scripts/gfx906_fallback_canary.sh | harness |
| `HIPFIRE_CASK_OFF` | scripts/redline_daemon_harness.py, scripts/serve_harness.py | deprecated |
| `HIPFIRE_CASK_SIDECAR` | crates/hipfire-config/src/lib.rs | deprecated |
| `HIPFIRE_CHATML` | crates/saddle-lab/examples/probe_argmax_agreement.rs | harness |
| `HIPFIRE_CHAT_CURRENT_DATE` | crates/hipfire-runtime/src/prompt_frame.rs | developer |
| `HIPFIRE_CHAT_TEMPLATE_FILE` | crates/hipfire-config/src/lib.rs, crates/hipfire-runtime/examples/dump_embedded_template.rs | stable |
| `HIPFIRE_CK_PREFIX` | scripts/install-ck-runtime.sh | harness |
| `HIPFIRE_CK_TARGET_GFX1100` | experiments/flash-attn-ck-sidecar/build_sidecar.sh | harness |
| `HIPFIRE_CK_TARGET_GFX1151` | experiments/flash-attn-ck-sidecar/build_sidecar.sh | harness |
| `HIPFIRE_CK_TARGET_GFX1201` | experiments/flash-attn-ck-sidecar/build_sidecar.sh | harness |
| `HIPFIRE_CLI_BIN` | benchmarks/scripts/mq4v2_k5120_abba.sh, crates/hipfire-config/src/lib.rs | stable |
| `HIPFIRE_COHERE2MOE_Q8_SCALAR` | crates/hipfire-arch-cohere2moe/src/forward.rs | developer |
| `HIPFIRE_COHERENCE_MAX_SEQ` | scripts/coherence-gate-cohere2moe.sh | deprecated |
| `HIPFIRE_COHERENCE_OUT` | scripts/coherence-gate-cohere2moe.sh, scripts/coherence-gate-deepseek4-mtp.sh | deprecated |
| `HIPFIRE_COHERENCE_TIMEOUT` | scripts/coherence-gate-deepseek4-mtp.sh, scripts/coherence-gate-deepseek4-recall.sh | deprecated |
| `HIPFIRE_COHERE_DEBUG` | crates/hipfire-arch-cohere2moe/src/forward.rs | developer |
| `HIPFIRE_COMPILER_FLAGS` | autoresearch/ar/certify/cross_arch.py, crates/rdna-compute/src/attention.rs | developer |
| `HIPFIRE_COMP_DUMP` | crates/hipfire-arch-deepseek4/src/forward.rs | developer |
| `HIPFIRE_CONTAINER` | scripts/container-gate.sh | harness |
| `HIPFIRE_CONTINUOUS_BATCH` | crates/hipfire-arch-qwen35/src/carrier.rs | developer |
| `HIPFIRE_CONTINUOUS_BATCH_SIZE` | crates/hipfire-config/src/lib.rs | stable |
| `HIPFIRE_CONV_QKNORM` | crates/hipfire-arch-qwen35/src/qwen35/forward.rs | developer |
| `HIPFIRE_CONV_QKNORM_SHAPE` | crates/hipfire-arch-qwen35/src/qwen35/forward.rs, crates/rdna-compute/src/norm.rs | developer |
| `HIPFIRE_CONV_SCALAR_PREP` | crates/hipfire-arch-qwen35/src/qwen35/forward.rs | developer |
| `HIPFIRE_CPU_EXEC_TRACE` | crates/hipfire-dispatch/src/cpu_exec.rs, docs/perf-checkpoints/data-2026-09-27-cpu-exec-mq3-avx2/cpu_exec_ab.sh | developer |
| `HIPFIRE_CQN_BLOCK` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_CQN_KERNEL` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_CQN_SCALAR_PREP` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_DAEMON` | scripts/test_pr228_spiral_check.sh | harness |
| `HIPFIRE_DAEMON_BIN` | autoresearch/ar/certify/serve_runner.py, autoresearch/ar/gate/serve_probe.py | stable |
| `HIPFIRE_DAEMON_FI_BIN` | scripts/device_mesh_matrix.py | harness |
| `HIPFIRE_DAEMON_NAME` | scripts/serve_harness.py | harness |
| `HIPFIRE_DAEMON_STDERR` | docs/investigations/evidence/ds4-mi300x-cdna-test-fail/raw/a1-m0/04-profile-feed.py, docs/investigations/evidence/ds4-mi300x-cdna-test-fail/raw/a1-m0/04-run-profile-direct.sh | harness |
| `HIPFIRE_DDTREE_ASSERT_MASK` | crates/hipfire-arch-qwen35/src/speculative.rs | developer |
| `HIPFIRE_DDTREE_BUDGET` | crates/hipfire-arch-llama/src/spec_impl.rs, crates/hipfire-arch-qwen35/src/dflash_spec.rs | experimental |
| `HIPFIRE_DDTREE_DUMP_PQ` | crates/hipfire-runtime/examples/ddtree_pq_sim.rs | harness |
| `HIPFIRE_DDTREE_FORCE_SLOW` | crates/hipfire-arch-qwen35/src/speculative.rs | developer |
| `HIPFIRE_DDTREE_LOGW_CUTOFF` | crates/hipfire-arch-qwen35/src/speculative.rs, crates/hipfire-config/src/lib.rs | experimental |
| `HIPFIRE_DDTREE_PATH_C_VERBOSE` | scripts/path-c-smoke.sh | harness |
| `HIPFIRE_DDTREE_TAPE_DUMP` | crates/hipfire-arch-qwen35/src/speculative.rs | developer |
| `HIPFIRE_DDTREE_TOPK` | crates/hipfire-config/src/lib.rs, crates/hipfire-runtime/src/loader_api.rs | experimental |
| `HIPFIRE_DDTREE_TOPK_DIRECT_OFF` | crates/rdna-compute/src/feature_flags.rs | developer |
| `HIPFIRE_DDTREE_TREE_LA` | crates/hipfire-arch-qwen35/src/speculative.rs, crates/hipfire-config/src/lib.rs | experimental |
| `HIPFIRE_DDTREE_VERIFY` | crates/hipfire-arch-qwen35/src/speculative.rs, crates/rdna-compute/src/feature_flags.rs | developer |
| `HIPFIRE_DEBUG_BATCH` | crates/hipfire-arch-qwen35/src/mtp_spec.rs, crates/hipfire-arch-qwen35/src/qwen35/prefill.rs | developer |
| `HIPFIRE_DEBUG_POOL_INVARIANTS` | crates/hipfire-arch-qwen35/src/serve_engine.rs | developer |
| `HIPFIRE_DEEPSEEK4_AR` | crates/hipfire-arch-deepseek4/examples/dspark_bench.rs | harness |
| `HIPFIRE_DEEPSEEK4_ATTN` | crates/hipfire-arch-deepseek4/examples/deepseek4_chat.rs, crates/hipfire-arch-deepseek4/src/config_cache.rs | developer |
| `HIPFIRE_DEEPSEEK4_ATTN_DEBUG_BISECT` | crates/hipfire-arch-deepseek4/src/forward.rs | developer |
| `HIPFIRE_DEEPSEEK4_ATTN_ILP4` | crates/rdna-compute/src/attention.rs | developer |
| `HIPFIRE_DEEPSEEK4_ATTN_PER_POS` | crates/hipfire-arch-deepseek4/src/forward.rs | developer |
| `HIPFIRE_DEEPSEEK4_ATTN_SCOREGRID_LARGE_SERIAL` | crates/rdna-compute/src/attention.rs | developer |
| `HIPFIRE_DEEPSEEK4_ATTN_SCOREGRID_XLANE` | crates/rdna-compute/src/attention.rs | developer |
| `HIPFIRE_DEEPSEEK4_ATTN_TOPK_DIRECT` | crates/hipfire-arch-deepseek4/src/forward.rs | developer |
| `HIPFIRE_DEEPSEEK4_ATTN_TWIN` | crates/hipfire-arch-deepseek4/src/forward.rs | developer |
| `HIPFIRE_DEEPSEEK4_ATTN_WARP` | crates/rdna-compute/src/attention.rs | developer |
| `HIPFIRE_DEEPSEEK4_BATCH_HEAD` | crates/hipfire-arch-deepseek4/src/forward.rs | developer |
| `HIPFIRE_DEEPSEEK4_BENCH_EXPERTS_PER_TOK` | crates/hipfire-arch-deepseek4/examples/dspark_bench.rs | harness |
| `HIPFIRE_DEEPSEEK4_BENCH_RAW` | crates/hipfire-arch-deepseek4/examples/dspark_bench.rs | harness |
| `HIPFIRE_DEEPSEEK4_CACHE_TRACE` | crates/hipfire-generate/src/common.rs, crates/hipfire-generate/src/dense.rs | developer |
| `HIPFIRE_DEEPSEEK4_CACTUS` | crates/hipfire-arch-deepseek4/examples/dspark_bench.rs | harness |
| `HIPFIRE_DEEPSEEK4_CHAT_RAW` | crates/hipfire-arch-deepseek4/examples/deepseek4_chat.rs, crates/hipfire-arch-deepseek4/scripts/ds4_mq2r_longctx_scan.sh | harness |
| `HIPFIRE_DEEPSEEK4_COMP_F16_WMMA` | crates/hipfire-arch-deepseek4/src/arch.rs, crates/hipfire-arch-deepseek4/src/deepseek4.rs | developer |
| `HIPFIRE_DEEPSEEK4_COMP_ROPE_POS` | crates/hipfire-arch-deepseek4/src/forward.rs | developer |
| `HIPFIRE_DEEPSEEK4_DSA_WMMA` | crates/hipfire-arch-deepseek4/src/forward.rs | developer |
| `HIPFIRE_DEEPSEEK4_DSPARK` | crates/hipfire-arch-deepseek4/examples/dspark_bench.rs, crates/hipfire-runtime/src/loader_api.rs | developer |
| `HIPFIRE_DEEPSEEK4_DSPARK_CONF_THRESHOLD` | crates/hipfire-arch-deepseek4/src/dspark_speculator.rs | developer |
| `HIPFIRE_DEEPSEEK4_DSPARK_VERIFY_AQL` | crates/hipfire-arch-deepseek4/src/spec_impl.rs | developer |
| `HIPFIRE_DEEPSEEK4_DSPARK_VERIFY_GRAPH` | crates/hipfire-arch-deepseek4/src/spec_impl.rs | developer |
| `HIPFIRE_DEEPSEEK4_DSPARK_VERIFY_GRAPH_BATCH` | crates/hipfire-arch-deepseek4/src/spec_impl.rs | developer |
| `HIPFIRE_DEEPSEEK4_DSPARK_VERIFY_PM4` | crates/hipfire-arch-deepseek4/src/spec_impl.rs | developer |
| `HIPFIRE_DEEPSEEK4_DUMP_INDEXER` | crates/hipfire-arch-deepseek4/src/forward.rs | developer |
| `HIPFIRE_DEEPSEEK4_DUMP_INDEXER_LAYERS` | crates/hipfire-arch-deepseek4/src/forward.rs | developer |
| `HIPFIRE_DEEPSEEK4_DUMP_PROMPT` | crates/hipfire-generate/src/dense.rs, crates/hipfire-generate/src/qwen.rs | developer |
| `HIPFIRE_DEEPSEEK4_DUMP_STATE` | crates/hipfire-arch-deepseek4/src/forward.rs | developer |
| `HIPFIRE_DEEPSEEK4_DUMP_TOPK` | crates/hipfire-dispatch/src/families/moe.rs, crates/hipfire-dispatch/src/pipeline/mod.rs | developer |
| `HIPFIRE_DEEPSEEK4_E8_ATTN_PACK_B3` | crates/hipfire-arch-deepseek4/src/forward.rs | developer |
| `HIPFIRE_DEEPSEEK4_E8_BATCHED_GEMV` | crates/hipfire-arch-deepseek4/src/config_cache.rs, crates/hipfire-arch-deepseek4/src/forward.rs | developer |
| `HIPFIRE_DEEPSEEK4_E8_BATCHED_PAIR_B3` | crates/hipfire-arch-deepseek4/src/forward.rs | developer |
| `HIPFIRE_DEEPSEEK4_E8_PREFILL_B2` | crates/hipfire-arch-deepseek4/src/config_cache.rs | developer |
| `HIPFIRE_DEEPSEEK4_E8_PREFILL_B4` | crates/hipfire-arch-deepseek4/src/config_cache.rs | developer |
| `HIPFIRE_DEEPSEEK4_E8_U4` | crates/hipfire-arch-deepseek4/src/config_cache.rs | developer |
| `HIPFIRE_DEEPSEEK4_E8_WO_GROUPED` | crates/hipfire-arch-deepseek4/src/config_cache.rs | developer |
| `HIPFIRE_DEEPSEEK4_EXPERT_LAYER_END` | crates/hipfire-arch-deepseek4/src/arch.rs | developer |
| `HIPFIRE_DEEPSEEK4_F32_TRACE` | crates/hipfire-arch-deepseek4/src/forward.rs | developer |
| `HIPFIRE_DEEPSEEK4_FFN_OVERLAP` | crates/hipfire-arch-deepseek4/src/config_cache.rs, crates/hipfire-arch-deepseek4/src/forward.rs | developer |
| `HIPFIRE_DEEPSEEK4_FUSED_ROUTE_ACT` | crates/hipfire-arch-deepseek4/src/config_cache.rs | developer |
| `HIPFIRE_DEEPSEEK4_FUSED_UNSCATTER_SILU` | crates/hipfire-dispatch/src/pipeline/mod.rs | developer |
| `HIPFIRE_DEEPSEEK4_GEN_TOKENS` | crates/hipfire-arch-deepseek4/examples/deepseek4_chat.rs, crates/hipfire-arch-deepseek4/scripts/ds4_mq2r_longctx_scan.sh | harness |
| `HIPFIRE_DEEPSEEK4_GFX1100_INDEXER_TOPK_TWOSTAGE` | crates/hipfire-arch-deepseek4/src/config_cache.rs | developer |
| `HIPFIRE_DEEPSEEK4_GFX1151_INDEXER_TOPK_TWOSTAGE` | crates/hipfire-arch-deepseek4/src/config_cache.rs | developer |
| `HIPFIRE_DEEPSEEK4_GFX1201_E8_WIDE` | crates/hipfire-arch-deepseek4/src/forward.rs | developer |
| `HIPFIRE_DEEPSEEK4_GFX1201_HC_FUSED24` | crates/hipfire-arch-deepseek4/src/forward.rs | developer |
| `HIPFIRE_DEEPSEEK4_GFX1201_INDEXER_ROPE_HEADS` | crates/hipfire-arch-deepseek4/src/config_cache.rs | developer |
| `HIPFIRE_DEEPSEEK4_GFX1201_INDEXER_TOPK_TWOSTAGE` | crates/hipfire-arch-deepseek4/src/config_cache.rs | developer |
| `HIPFIRE_DEEPSEEK4_GFX1201_TOPK_GATHER_TILED` | crates/hipfire-arch-deepseek4/src/forward.rs | developer |
| `HIPFIRE_DEEPSEEK4_GFX942_COMPRESSOR_GATE` | crates/hipfire-arch-deepseek4/src/config_cache.rs | developer |
| `HIPFIRE_DEEPSEEK4_GFX942_E8_ROCBLAS` | crates/rdna-compute/examples/test_mfp4e8_soa_rocblas_gfx942.rs, crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_DEEPSEEK4_GFX942_E8_WO_GROUPED` | crates/hipfire-arch-deepseek4/src/config_cache.rs | developer |
| `HIPFIRE_DEEPSEEK4_GFX942_FFN_OVERLAP` | crates/hipfire-arch-deepseek4/src/config_cache.rs | developer |
| `HIPFIRE_DEEPSEEK4_GFX942_HC_FINALIZE_FUSED` | crates/hipfire-arch-deepseek4/src/config_cache.rs | developer |
| `HIPFIRE_DEEPSEEK4_GFX942_INDEXER_TOPK_BOUNDED` | crates/hipfire-arch-deepseek4/scripts/ds4_mq2r_cap_campaign.sh, crates/hipfire-arch-deepseek4/src/config_cache.rs | developer |
| `HIPFIRE_DEEPSEEK4_GFX942_INDEXER_TOPK_PARALLEL` | crates/hipfire-arch-deepseek4/src/config_cache.rs, docs/investigations/evidence/ds4-mi300x-cdna-test-fail/raw/a1-m0/run-abba.sh | developer |
| `HIPFIRE_DEEPSEEK4_GFX942_INDEXER_TOPK_TWOSTAGE` | crates/hipfire-arch-deepseek4/src/config_cache.rs | developer |
| `HIPFIRE_DEEPSEEK4_GRAPH` | crates/hipfire-arch-deepseek4/examples/ds4_longctx_probe.rs, crates/hipfire-arch-deepseek4/src/deepseek4.rs | developer |
| `HIPFIRE_DEEPSEEK4_HC_CONTROL_FINALIZE_FUSED` | crates/hipfire-arch-deepseek4/src/config_cache.rs | developer |
| `HIPFIRE_DEEPSEEK4_HC_CONTROL_RSQRT_ONCE` | crates/hipfire-arch-deepseek4/src/config_cache.rs | developer |
| `HIPFIRE_DEEPSEEK4_HC_FINALIZE_FUSED` | crates/hipfire-arch-deepseek4/src/config_cache.rs | developer |
| `HIPFIRE_DEEPSEEK4_HC_FINALIZE_INPUT_MAP` | crates/hipfire-arch-deepseek4/src/config_cache.rs | developer |
| `HIPFIRE_DEEPSEEK4_HC_PINGPONG` | crates/hipfire-arch-deepseek4/src/config_cache.rs, crates/hipfire-arch-deepseek4/src/forward.rs | developer |
| `HIPFIRE_DEEPSEEK4_HFQ4_WMMA` | crates/hipfire-arch-deepseek4/src/forward.rs | developer |
| `HIPFIRE_DEEPSEEK4_INDEXER_TOPK_BLOCK1024` | crates/rdna-compute/src/attention.rs | developer |
| `HIPFIRE_DEEPSEEK4_INDEXER_TOPK_BOUNDED` | crates/rdna-compute/src/attention.rs | developer |
| `HIPFIRE_DEEPSEEK4_INDEXER_TOPK_SERIAL` | crates/rdna-compute/src/attention.rs | developer |
| `HIPFIRE_DEEPSEEK4_INDEXER_TOPK_TWOSTAGE_MIN` | crates/hipfire-arch-deepseek4/src/config_cache.rs, crates/rdna-compute/examples/test_indexer_top_k_buf.rs | developer |
| `HIPFIRE_DEEPSEEK4_INDEXER_TOPK_UNROLLED` | crates/rdna-compute/src/attention.rs | developer |
| `HIPFIRE_DEEPSEEK4_INDEXER_WMMA` | crates/hipfire-arch-deepseek4/src/forward.rs | developer |
| `HIPFIRE_DEEPSEEK4_LAYER_NORM` | crates/hipfire-arch-deepseek4/examples/ds4_prod_vs_parent_trace.rs, crates/hipfire-arch-deepseek4/src/forward.rs | developer |
| `HIPFIRE_DEEPSEEK4_LOAD_DSPARK` | crates/hipfire-runtime/src/loader_api.rs | developer |
| `HIPFIRE_DEEPSEEK4_LOAD_MTP` | crates/hipfire-arch-deepseek4/src/arch.rs, crates/hipfire-arch-deepseek4/src/deepseek4.rs | developer |
| `HIPFIRE_DEEPSEEK4_MAX` | crates/hipfire-arch-deepseek4/examples/dspark_bench.rs | harness |
| `HIPFIRE_DEEPSEEK4_MAX_COMPRESS_POS` | crates/hipfire-arch-deepseek4/src/deepseek4.rs, docs/investigations/evidence/ds4-mi300x-cdna-test-fail/raw/a1-m0/run-cliff.sh | developer |
| `HIPFIRE_DEEPSEEK4_MODEL` | crates/hipfire-arch-deepseek4/examples/deepseek4_chat.rs, crates/hipfire-arch-deepseek4/examples/deepseek4_prefill_bench.rs | harness |
| `HIPFIRE_DEEPSEEK4_MOE` | crates/hipfire-arch-deepseek4/src/config_cache.rs, crates/hipfire-arch-deepseek4/src/forward.rs | developer |
| `HIPFIRE_DEEPSEEK4_MOE_8W` | crates/hipfire-arch-deepseek4/examples/deepseek4_prefill_bench.rs, crates/hipfire-dispatch/src/pipeline/mod.rs | developer |
| `HIPFIRE_DEEPSEEK4_MOE_CND` | crates/hipfire-arch-deepseek4/examples/deepseek4_prefill_bench.rs, crates/hipfire-dispatch/src/pipeline/mod.rs | developer |
| `HIPFIRE_DEEPSEEK4_MOE_DETERMINISTIC` | crates/hipfire-arch-deepseek4/src/forward.rs, crates/hipfire-dispatch/src/pipeline/mod.rs | developer |
| `HIPFIRE_DEEPSEEK4_MOE_DOWN_BATCHED_K8ALL` | crates/hipfire-dispatch/src/pipeline/mod.rs | developer |
| `HIPFIRE_DEEPSEEK4_MOE_GATE_UP_BATCHED_K4096_LDS` | crates/hipfire-dispatch/src/pipeline/mod.rs | developer |
| `HIPFIRE_DEEPSEEK4_MOE_GROUPED` | crates/hipfire-dispatch/src/pipeline/mod.rs | developer |
| `HIPFIRE_DEEPSEEK4_MOE_GROUPED_GATE` | crates/hipfire-dispatch/src/families/moe.rs, crates/hipfire-dispatch/src/pipeline/mod.rs | developer |
| `HIPFIRE_DEEPSEEK4_MOE_LLOYD_4W` | crates/hipfire-dispatch/src/pipeline/mod.rs | developer |
| `HIPFIRE_DEEPSEEK4_MOE_MMQLOAD` | crates/hipfire-arch-deepseek4/examples/deepseek4_prefill_bench.rs, crates/hipfire-dispatch/src/pipeline/mod.rs | developer |
| `HIPFIRE_DEEPSEEK4_MOE_N32` | crates/hipfire-arch-deepseek4/examples/deepseek4_prefill_bench.rs, crates/hipfire-dispatch/src/pipeline/mod.rs | developer |
| `HIPFIRE_DEEPSEEK4_MOE_NOSYNC` | crates/hipfire-arch-deepseek4/examples/deepseek4_prefill_bench.rs, crates/hipfire-dispatch/src/pipeline/mod.rs | developer |
| `HIPFIRE_DEEPSEEK4_MQ2_PERM` | crates/hipfire-dispatch/src/pipeline/mod.rs | developer |
| `HIPFIRE_DEEPSEEK4_MTP_ADDON` | crates/hipfire-arch-deepseek4/src/arch.rs, crates/hipfire-loader/src/carriers.rs | developer |
| `HIPFIRE_DEEPSEEK4_MTP_HEAD_HC` | crates/hipfire-arch-deepseek4/src/config_cache.rs, crates/hipfire-arch-deepseek4/src/mtp.rs | developer |
| `HIPFIRE_DEEPSEEK4_MTP_SKIP_HEAD` | crates/hipfire-arch-deepseek4/src/forward.rs, crates/hipfire-arch-deepseek4/src/mtp.rs | developer |
| `HIPFIRE_DEEPSEEK4_PARENT_MODEL` | crates/hipfire-ds4-parent/examples/ds4_parent_loader_oracle.rs | harness |
| `HIPFIRE_DEEPSEEK4_PARENT_POST_SCALE` | crates/hipfire-ds4-parent/src/hc.rs | developer |
| `HIPFIRE_DEEPSEEK4_POST_SCALE` | crates/hipfire-arch-deepseek4/examples/ds4_prod_vs_parent_trace.rs, crates/hipfire-arch-deepseek4/src/forward.rs | developer |
| `HIPFIRE_DEEPSEEK4_PP_BATCH` | crates/hipfire-arch-deepseek4/examples/deepseek4_chat.rs, crates/hipfire-arch-deepseek4/examples/deepseek4_prefill_bench.rs | developer |
| `HIPFIRE_DEEPSEEK4_PROMPT` | crates/hipfire-arch-deepseek4/examples/dspark_bench.rs, crates/hipfire-arch-deepseek4/examples/dspark_forward_smoke.rs | harness |
| `HIPFIRE_DEEPSEEK4_Q8_4W` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | stable |
| `HIPFIRE_DEEPSEEK4_Q8_WMMA` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | stable |
| `HIPFIRE_DEEPSEEK4_QNORM_ROTATE_FUSED` | crates/hipfire-arch-deepseek4/src/config_cache.rs | developer |
| `HIPFIRE_DEEPSEEK4_REAP_KEEPMAP` | crates/hipfire-arch-deepseek4/examples/deepseek4_perplexity.rs, crates/hipfire-arch-deepseek4/src/deepseek4.rs | developer |
| `HIPFIRE_DEEPSEEK4_REDLINE_FFN_SPLIT` | crates/hipfire-arch-deepseek4/src/config_cache.rs | developer |
| `HIPFIRE_DEEPSEEK4_REDLINE_HIP_BLOB` | crates/hipfire-arch-deepseek4/src/forward.rs | developer |
| `HIPFIRE_DEEPSEEK4_RETAINED_EMBEDDING` | crates/hipfire-arch-deepseek4/src/config_cache.rs | developer |
| `HIPFIRE_DEEPSEEK4_ROUTE_SCALE` | crates/hipfire-arch-deepseek4/examples/ds4_prod_vs_parent_trace.rs, crates/hipfire-arch-deepseek4/examples/ds4_quant_plog.rs | developer |
| `HIPFIRE_DEEPSEEK4_SEED` | crates/hipfire-arch-deepseek4/examples/deepseek4_chat.rs, crates/hipfire-generate/src/dense.rs | developer |
| `HIPFIRE_DEEPSEEK4_SKIP_FFN` | crates/hipfire-arch-deepseek4/src/config_cache.rs | developer |
| `HIPFIRE_DEEPSEEK4_SPEC_DECODE` | crates/hipfire-arch-deepseek4/examples/deepseek4_chat.rs, crates/hipfire-loader/src/carriers.rs | developer |
| `HIPFIRE_DEEPSEEK4_SPEC_K` | crates/hipfire-arch-deepseek4/examples/deepseek4_chat.rs, crates/hipfire-arch-deepseek4/examples/dspark_bench.rs | developer |
| `HIPFIRE_DEEPSEEK4_STAGE_LAYERS` | crates/hipfire-arch-deepseek4/src/forward.rs | developer |
| `HIPFIRE_DEEPSEEK4_STAGE_NORM` | crates/hipfire-arch-deepseek4/src/forward.rs | developer |
| `HIPFIRE_DEEPSEEK4_TEMP` | crates/hipfire-arch-deepseek4/examples/deepseek4_chat.rs, crates/hipfire-arch-deepseek4/examples/dspark_bench.rs | harness |
| `HIPFIRE_DEEPSEEK4_TOP_K` | crates/hipfire-arch-deepseek4/examples/deepseek4_chat.rs, crates/hipfire-arch-deepseek4/examples/dspark_bench.rs | developer |
| `HIPFIRE_DEEPSEEK4_TOP_P` | crates/hipfire-arch-deepseek4/examples/dspark_bench.rs | harness |
| `HIPFIRE_DEEPSEEK4_UPLOAD_EXPERTS` | crates/hipfire-arch-deepseek4/src/arch.rs, crates/hipfire-arch-deepseek4/src/forward.rs | developer |
| `HIPFIRE_DEEPSEEK4_WARMUP` | crates/hipfire-arch-deepseek4/examples/dspark_bench.rs | harness |
| `HIPFIRE_DEEPSEEK4_WO_MULTIROW` | crates/hipfire-arch-deepseek4/src/forward.rs | developer |
| `HIPFIRE_DEEPSEEK4_WO_Q8_WMMA` | crates/rdna-compute/src/gemv.rs | developer |
| `HIPFIRE_DEFAULT_CHATML` | crates/hipfire-config/src/lib.rs | stable |
| `HIPFIRE_DEMOTE_MQ6` | crates/hipfire-runtime/examples/hfq_splice_attn.rs | harness |
| `HIPFIRE_DENSE_FIXTURE` | crates/hipfire-generate/tests/dense_rollback_gpu.rs, scripts/device_mesh_matrix.py | harness |
| `HIPFIRE_DENSE_TP` | crates/hipfire-arch-qwen35/examples/qwen_dense_tp2_parity.rs | harness |
| `HIPFIRE_DETECTED_ARCH` | scripts/_detect-gpu.sh, scripts/test-kernels.sh | harness |
| `HIPFIRE_DETECTED_NAME` | scripts/_detect-gpu.sh | harness |
| `HIPFIRE_DETECTED_VRAM_GB` | scripts/_detect-gpu.sh | harness |
| `HIPFIRE_DETERMINISTIC` | autoresearch/ar/certify/serve_runner.py, crates/hipfire-arch-qwen35/examples/qwen_dense_tp2_parity.rs | experimental |
| `HIPFIRE_DEVICE` | crates/hipfire-config/src/lib.rs | developer |
| `HIPFIRE_DEVICES` | crates/hipfire-arch-qwen35/tests/mtp_takeover_fill.rs, crates/hipfire-cli/src/main.rs | stable |
| `HIPFIRE_DFLASH_ADAPTIVE_B` | crates/hipfire-arch-qwen35/src/dflash_spec.rs, crates/hipfire-daemon/src/main.rs | developer |
| `HIPFIRE_DFLASH_CHAT` | crates/hipfire-generate/src/ar.rs | developer |
| `HIPFIRE_DFLASH_CKPT_RESUME` | crates/hipfire-arch-qwen35/src/dflash_spec.rs, crates/hipfire-config/src/lib.rs | developer |
| `HIPFIRE_DFLASH_CTX_CAP` | crates/hipfire-arch-qwen35/src/dflash_slot.rs, crates/hipfire-arch-qwen35/src/dflash_spec.rs | deprecated |
| `HIPFIRE_DFLASH_DRAFT` | crates/hipfire-arch-muse-glimmer/src/drafter.rs, crates/hipfire-arch-qwen35/src/serve_engine.rs | developer |
| `HIPFIRE_DFLASH_FAST_SAMPLE` | crates/hipfire-arch-qwen35/src/speculative.rs, crates/hipfire-config/src/lib.rs | experimental |
| `HIPFIRE_DFLASH_LEGACY_PREFILL` | crates/hipfire-arch-qwen35/src/dflash_spec.rs, crates/hipfire-arch-qwen35/src/speculative.rs | developer |
| `HIPFIRE_DFLASH_LOGIT_DUMP` | crates/hipfire-arch-qwen35/src/speculative.rs | developer |
| `HIPFIRE_DFLASH_LOOP_BREAK` | crates/hipfire-runtime/examples/dflash_spec_demo.rs | harness |
| `HIPFIRE_DFLASH_LOOP_BREAK_MAX_ESCALATIONS` | crates/hipfire-runtime/examples/dflash_spec_demo.rs | harness |
| `HIPFIRE_DFLASH_LOOP_BREAK_RECOVERY` | crates/hipfire-runtime/examples/dflash_spec_demo.rs | harness |
| `HIPFIRE_DFLASH_LOOP_BREAK_RP_MAX` | crates/hipfire-runtime/examples/dflash_spec_demo.rs | harness |
| `HIPFIRE_DFLASH_LOOP_BREAK_RP_STEP` | crates/hipfire-runtime/examples/dflash_spec_demo.rs | harness |
| `HIPFIRE_DFLASH_LOOP_BREAK_STOP_AFTER` | crates/hipfire-runtime/examples/dflash_spec_demo.rs | harness |
| `HIPFIRE_DFLASH_LOOP_BREAK_TEMP` | crates/hipfire-runtime/examples/dflash_spec_demo.rs | harness |
| `HIPFIRE_DFLASH_MODE` | crates/hipfire-config/src/lib.rs, scripts/benchlocal_campaign.py | stable |
| `HIPFIRE_DFLASH_MOE_DRAFT_FFN_GRAPH` | crates/hipfire-arch-qwen35/src/speculative.rs | developer |
| `HIPFIRE_DFLASH_MOE_VERIFY_GRAPH_LMHEAD` | crates/hipfire-arch-qwen35/src/speculative.rs | developer |
| `HIPFIRE_DFLASH_NGRAM_BLOCK` | crates/hipfire-arch-qwen35/src/speculative.rs, crates/hipfire-config/src/lib.rs | stable |
| `HIPFIRE_DFLASH_OFF` | scripts/serve-loop-gate.sh | harness |
| `HIPFIRE_DFLASH_Q8_LMHEAD_WMMA` | crates/hipfire-arch-qwen35/src/speculative.rs, crates/hipfire-config/src/lib.rs | experimental |
| `HIPFIRE_DFLASH_REFERENCE` | scripts/dflash_ref_spec_test.py, scripts/dflash_spec_debug.py | harness |
| `HIPFIRE_DFLASH_SEED_ORACLE` | crates/hipfire-arch-qwen35/src/speculative.rs, scripts/seed_oracle_collect.sh | developer |
| `HIPFIRE_DFLASH_TEMP_SPEC` | crates/hipfire-generate/src/ar.rs, crates/hipfire-generate/src/batch.rs | developer |
| `HIPFIRE_DFLASH_TREE` | crates/hipfire-config/src/lib.rs, crates/hipfire-runtime/src/dflash_generic.rs | experimental |
| `HIPFIRE_DFLASH_VERIFY_PM4` | crates/hipfire-arch-qwen35/src/dflash_spec.rs, crates/hipfire-generate/src/redline.rs | developer |
| `HIPFIRE_DFLASH_WINDOW` | crates/hipfire-arch-qwen35/src/dflash_slot.rs, crates/hipfire-arch-qwen35/src/dflash_spec.rs | developer; `0` (legacy contiguous DFlash) deprecated, removal 0.5.0 |
| `HIPFIRE_DFLASH_ZLAB_SAFETENSORS` | scripts/dflash_spec_debug.py | harness |
| `HIPFIRE_DIR` | .agents/skills/hipfire-autoheal/triage.sh, scripts/ab-dispatch-validation.sh | harness |
| `HIPFIRE_DIVERGENCE_PREFILL` | scripts/gfx906_logit_divergence.sh | harness |
| `HIPFIRE_DIVERGENCE_TOL` | scripts/gfx906_logit_divergence.sh | harness |
| `HIPFIRE_DN_REQUANT_PER_TOKEN` | crates/hipfire-arch-qwen35/src/qwen35/prefill.rs, crates/hipfire-runtime/examples/tmp_gfx1201_chunk_exactness.rs | developer |
| `HIPFIRE_DN_SNAPSHOT_BULK_OFF` | crates/hipfire-arch-qwen35/src/speculative.rs, crates/rdna-compute/src/feature_flags.rs | developer |
| `HIPFIRE_DN_SNAPSHOT_FLIP` | crates/hipfire-arch-qwen35/src/speculative.rs, crates/rdna-compute/src/feature_flags.rs | developer |
| `HIPFIRE_DN_STATE_EF` | autoresearch/ar/certify/serve_runner.py, crates/hipfire-arch-qwen35/src/qwen35/ep_batch.rs | developer |
| `HIPFIRE_DOT2_GEMV` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_DOTS_FAULT` | crates/hipfire-generate/tests/vision_lifecycle_tests.rs | harness |
| `HIPFIRE_DOTS_IMAGE` | crates/hipfire-generate/tests/vision_lifecycle_tests.rs | harness |
| `HIPFIRE_DOTS_OCR_BF16_RESIDUAL` | crates/hipfire-arch-dots-ocr/src/dots_ocr.rs | developer |
| `HIPFIRE_DOTS_OCR_DUMP_DIR` | crates/hipfire-arch-dots-ocr/src/dots_ocr.rs, scripts/diff_dots_ocr_stages.py | developer |
| `HIPFIRE_DOTS_OCR_FIXTURE` | crates/hipfire-generate/tests/vision_lifecycle_tests.rs | harness |
| `HIPFIRE_DOTS_OCR_TRACE` | crates/hipfire-arch-dots-ocr/src/dots_ocr.rs | developer |
| `HIPFIRE_DPM_WARMUP_SECS` | benchmarks/scripts/mq4v2_k5120_abba.sh, crates/hipfire-arch-cohere2moe/examples/infer.rs | developer |
| `HIPFIRE_DRAFT_COLLAPSE_OFF` | crates/hipfire-arch-qwen35/examples/test_dflash_draft_collapse_gfx1100.rs, crates/hipfire-arch-qwen35/src/speculative.rs | developer |
| `HIPFIRE_DRAFT_F16` | crates/hipfire-config/src/lib.rs, crates/hipfire-runtime/src/config.rs | stable |
| `HIPFIRE_DRAFT_GEMM_DUMP` | crates/hipfire-config/src/lib.rs, crates/hipfire-runtime/src/config.rs | experimental |
| `HIPFIRE_DRAFT_SUBPHASE` | crates/hipfire-config/src/lib.rs, crates/hipfire-runtime/src/config.rs | experimental |
| `HIPFIRE_DRAM_GBS` | crates/rdna-compute/examples/test_vae_lds.rs | harness |
| `HIPFIRE_DS4_ALLOW_NON_GFX942` | crates/hipfire-arch-deepseek4/examples/ds4_prod_vs_parent_trace.rs, crates/hipfire-ds4-parent/examples/ds4_parent_residual_content.rs | harness |
| `HIPFIRE_DS4_DENSE_ACT_DIR` | crates/hip-bridge/examples/collect_e8_hessian_rocblas.rs, crates/hipfire-arch-deepseek4/examples/deepseek4_perplexity.rs | developer |
| `HIPFIRE_DS4_EXPERT_OVERLAP` | crates/rdna-compute/src/gemv.rs | developer |
| `HIPFIRE_DS4_GATHER_TILED` | crates/hipfire-arch-deepseek4/src/forward.rs | developer |
| `HIPFIRE_DS4_MOE_UNSCATTER_ONLY` | crates/rdna-compute/examples/bench_moe_e8_prefill.rs | harness |
| `HIPFIRE_DS4_OWNER_WORKER_SAMPLES` | crates/hipfire-runtime/examples/ds4_gfx1201_owner_worker_transport.rs | harness |
| `HIPFIRE_DS4_OWNER_WORKER_WARMUPS` | crates/hipfire-runtime/examples/ds4_gfx1201_owner_worker_transport.rs | harness |
| `HIPFIRE_DS4_QUANT_PLOG_ALLOW_NON_GFX942` | crates/hipfire-arch-deepseek4/examples/ds4_quant_plog.rs, crates/hipfire-arch-deepseek4/examples/ds4_quant_plog_multi.rs | harness |
| `HIPFIRE_DS4_REPLAY_INVENTORY` | crates/hipfire-arch-deepseek4/src/forward.rs, crates/hipfire-arch-deepseek4/src/spec_impl.rs | developer |
| `HIPFIRE_DS4_REPLAY_SEQUENCE` | crates/hipfire-arch-deepseek4/src/spec_impl.rs | developer |
| `HIPFIRE_DS4_ROUTE_DUMP` | crates/hipfire-arch-deepseek4/src/forward.rs | developer |
| `HIPFIRE_DSPARK_ADAPTIVE_BLOCK` | crates/hipfire-arch-qwen35/src/dflash_spec.rs, crates/hipfire-runtime/src/dflash_adaptive_block.rs | developer |
| `HIPFIRE_DSPARK_DEBUG` | crates/hipfire-runtime/src/dspark_core.rs | developer |
| `HIPFIRE_DSPARK_FIXTURE` | crates/hipfire-arch-deepseek4/src/arch.rs | developer |
| `HIPFIRE_DSPARK_HFQ4_WMMA` | crates/hipfire-runtime/src/dspark_core.rs | developer |
| `HIPFIRE_DSPARK_KERNEL_PROFILE_POSITION` | crates/hipfire-runtime/src/dspark_core.rs | developer |
| `HIPFIRE_DSPARK_PROFILE` | crates/hipfire-runtime/src/dspark_core.rs | developer |
| `HIPFIRE_DSPARK_Q8_4W` | crates/hipfire-runtime/src/dspark_core.rs | developer |
| `HIPFIRE_DSPARK_Q8_WMMA` | crates/hipfire-config/src/lib.rs, crates/hipfire-runtime/src/dspark_core.rs | developer |
| `HIPFIRE_DSPARK_ZERO_CTX` | crates/hipfire-runtime/src/dspark_core.rs | developer |
| `HIPFIRE_DTOD_DUMP` | crates/rdna-compute/src/dispatch.rs, scripts/analysis/ds4-gfx1151-roofline/dtod2.sh | developer |
| `HIPFIRE_DTOH_DUMP` | crates/hip-bridge/src/ffi.rs | developer |
| `HIPFIRE_DUMP_HIDDEN` | crates/hipfire-arch-qwen35/src/qwen35/forward.rs, crates/hipfire-dispatch/src/pipeline/mod.rs | developer |
| `HIPFIRE_DUMP_HIDDEN_POS` | crates/hipfire-dispatch/src/pipeline/mod.rs | developer |
| `HIPFIRE_E8_DGPU_TWIN` | crates/rdna-compute/src/gemv.rs, crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_E8_GFX12` | crates/hipfire-arch-qwen35/src/qwen35/prefill.rs, crates/hipfire-dispatch/src/families/moe.rs | developer |
| `HIPFIRE_E8_GROUPED_GFX1201_ONLY` | crates/rdna-compute/examples/bench_e8_soa_correctness.rs | harness |
| `HIPFIRE_E8_HESSIAN_DIR` | crates/hipfire-quantize/src/reap_overlay.rs, scripts/reap/build_deepseek4_e8_bucket_overlay.sh | developer |
| `HIPFIRE_E8_IMATRIX` | crates/hipfire-quantize/src/quant_e8.rs, scripts/reap/build_deepseek4_e8_bucket_overlay.sh | developer |
| `HIPFIRE_E8_LDSX` | crates/rdna-compute/src/gemv.rs, crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_E8_PACK_CORRECTNESS_ONLY` | crates/rdna-compute/examples/bench_e8_soa_correctness.rs | harness |
| `HIPFIRE_E8_PREFILL_COOP_BATCH` | crates/rdna-compute/examples/bench_e8_soa_correctness.rs | harness |
| `HIPFIRE_E8_PREFILL_COOP_ONLY` | crates/rdna-compute/examples/bench_e8_soa_correctness.rs | harness |
| `HIPFIRE_E8_PREFILL_GFX1201_BATCH` | crates/rdna-compute/examples/bench_e8_soa_correctness.rs | harness |
| `HIPFIRE_E8_PREFILL_GFX1201_ONLY` | crates/rdna-compute/examples/bench_e8_soa_correctness.rs | harness |
| `HIPFIRE_E8_PREFILL_GFX1201_TRIALS` | crates/rdna-compute/examples/bench_e8_soa_correctness.rs | harness |
| `HIPFIRE_E8_ROW_GATE` | crates/saddle-lab/examples/collect_e8_hessian_native.rs | harness |
| `HIPFIRE_E8_SOA_EXPERTS` | crates/hipfire-arch-qwen35/src/qwen35/load.rs, crates/rdna-compute/src/gemv.rs | developer |
| `HIPFIRE_E8_STRIP` | crates/rdna-compute/src/gemv.rs, crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_EMIT_TOKEN_IDS` | autoresearch/ar/certify/serve_runner.py, crates/hipfire-arch-cohere2moe/src/spec_emit.rs | developer |
| `HIPFIRE_EMULATE_GPUS` | crates/hipfire-arch-qwen35/tests/route_oracle_mesh.rs, crates/hipfire-runtime/src/config.rs | developer |
| `HIPFIRE_EMU_BF16_H` | crates/rdna-compute/src/gemv.rs | developer |
| `HIPFIRE_EP_DECODE_TIMING` | crates/hipfire-arch-deepseek4/src/ep.rs, crates/hipfire-arch-minimax/src/forward.rs | developer |
| `HIPFIRE_EP_DUMP_IDX` | crates/hipfire-arch-deepseek4/src/ep.rs | developer |
| `HIPFIRE_EP_DUMP_POS` | crates/hipfire-arch-deepseek4/src/ep.rs | developer |
| `HIPFIRE_EP_FAIL_RANK` | crates/hipfire-loader/src/lib.rs | developer |
| `HIPFIRE_EP_KV_MODE` | crates/hipfire-runtime/examples/ep_decode_parity.rs | harness |
| `HIPFIRE_EP_KV_SEQ` | crates/hipfire-runtime/examples/ep_decode_parity.rs | harness |
| `HIPFIRE_EP_ORACLE_DUMP` | crates/hipfire-arch-qwen35/tests/route_oracle_mesh.rs | harness |
| `HIPFIRE_EP_PEER_ALLREDUCE` | crates/hipfire-arch-qwen35/src/qwen35/ep_batch.rs | developer |
| `HIPFIRE_EP_PEER_ALLREDUCE_DECODE` | crates/hipfire-runtime/src/ep.rs | developer |
| `HIPFIRE_EP_PREFILL` | crates/hipfire-runtime/examples/ep_decode_parity.rs | harness |
| `HIPFIRE_EP_PREFILL_TIMING` | crates/hipfire-arch-qwen35/src/qwen35/ep_batch.rs | developer |
| `HIPFIRE_EP_PROMPT_REPEAT` | crates/hipfire-runtime/examples/ep_decode_parity.rs | harness |
| `HIPFIRE_EP_SKIP_ALLREDUCE` | crates/hipfire-arch-qwen35/src/qwen35/ep_batch.rs | developer |
| `HIPFIRE_EVENT_LOG` | docs/investigations/evidence/ds4-mi300x-cdna-test-fail/raw/a1-m0/04-profile-feed.py, docs/investigations/evidence/ds4-mi300x-cdna-test-fail/raw/a1-m0/04-run-profile-direct.sh | harness |
| `HIPFIRE_EXPERIMENTAL_BUDGET_ALERT` | crates/hipfire-config/src/lib.rs, crates/hipfire-runtime/src/config.rs | experimental |
| `HIPFIRE_EXPERIMENTAL_MODE` | crates/hipfire-config/src/lib.rs | developer |
| `HIPFIRE_F1LITE` | crates/hipfire-arch-qwen35/src/qwen35/prefill.rs, crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_FA2_A4_EPILOGUE` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_FA2_FILL` | crates/rdna-compute/src/attention.rs, crates/rdna-compute/src/kernel_registry.rs | developer |
| `HIPFIRE_FA2_FP8` | crates/rdna-compute/src/attention.rs, crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_FA2_GFX1100` | crates/rdna-compute/src/attention.rs, crates/rdna-compute/src/kernel_registry.rs | developer |
| `HIPFIRE_FA2_KMODE` | crates/hipfire-runtime/examples/tmp_fa2_attrib.rs, crates/rdna-compute/src/attention.rs | developer |
| `HIPFIRE_FA2_KT` | crates/hipfire-runtime/examples/tmp_fa2_attrib.rs, crates/rdna-compute/src/attention.rs | developer |
| `HIPFIRE_FA2_PACKET` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_FA2_Q16` | crates/rdna-compute/src/attention.rs, crates/rdna-compute/src/kernel_registry.rs | developer |
| `HIPFIRE_FA2_QRESIDENT` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_FA2_QRESIDENT_V2` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_FA2_QRESIDENT_V2_Q8` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_FAST_SAMPLE` | crates/hipfire-generate/src/ar.rs, crates/hipfire-generate/src/batch.rs | developer |
| `HIPFIRE_FAULT_HIP` | crates/hip-bridge/src/ffi.rs, crates/hipfire-runtime/examples/test_serve_prefix_cache.rs | developer |
| `HIPFIRE_FAULT_MTP_FULL_REJECT` | crates/hipfire-runtime/examples/test_serve_prefix_cache.rs | harness |
| `HIPFIRE_FAULT_PREFIX_PUBLISH` | crates/hipfire-arch-qwen35/src/serve_engine.rs, crates/hipfire-runtime/examples/test_serve_prefix_cache.rs | developer |
| `HIPFIRE_FA_BATCH_FUSE_OFF` | crates/hipfire-arch-qwen35/src/qwen35/prefill.rs, crates/rdna-compute/src/feature_flags.rs | developer |
| `HIPFIRE_FA_PERTOKEN_MIN_CTX` | crates/hipfire-arch-qwen35/src/qwen35/prefill.rs, crates/railgun-cert/src/recording.rs | developer |
| `HIPFIRE_FIXED_TIER` | crates/hipfire-quantize/src/model_filter.rs, crates/hipfire-quantize/src/pipeline.rs | developer |
| `HIPFIRE_FLASH_ATTN_CK_LIB` | crates/hipfire-config/src/lib.rs, crates/railgun-cert/src/recording.rs | experimental |
| `HIPFIRE_FLASH_ATTN_CK_TEST_LIB` | crates/rdna-compute/src/flash_attn_ck.rs | developer |
| `HIPFIRE_FLASH_ATTN_CK_WORKSPACE_BYTES` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_FLASH_PARTIALS_BATCH` | crates/hipfire-arch-qwen35/src/qwen35/forward.rs, crates/hipfire-arch-qwen35/src/qwen35/prefill.rs | experimental |
| `HIPFIRE_FLASH_PREFILL` | crates/hipfire-arch-qwen35/src/forward_slots.rs, crates/hipfire-arch-qwen35/src/qwen35/prefill.rs | developer |
| `HIPFIRE_FLASH_PREFILL_BC` | crates/hipfire-dispatch/src/families/attention.rs | developer |
| `HIPFIRE_FLASH_PREFILL_BR` | crates/hipfire-dispatch/src/families/attention.rs, crates/rdna-compute/src/kernel_registry.rs | developer |
| `HIPFIRE_FLASH_PREFILL_FIXED_HD` | crates/rdna-compute/examples/bench_glimmer_verify_attention.rs, crates/rdna-compute/src/attention.rs | developer |
| `HIPFIRE_FLASH_PREFILL_KERNEL` | crates/hipfire-arch-muse-glimmer/src/forward.rs, crates/hipfire-arch-qwen35/src/forward_slots.rs | developer |
| `HIPFIRE_FLASH_PREFILL_MIN_CTX` | crates/hipfire-dispatch/src/families/attention.rs | developer |
| `HIPFIRE_FLASH_PREFILL_PREFETCH_V` | crates/rdna-compute/examples/bench_glimmer_verify_attention.rs, crates/rdna-compute/src/attention.rs | developer |
| `HIPFIRE_FLASH_PREFILL_SPLITQ` | crates/rdna-compute/examples/bench_glimmer_verify_attention.rs, crates/rdna-compute/src/attention.rs | developer |
| `HIPFIRE_FLUX_ATTN` | crates/hipfire-arch-diffusion/src/flux_gpu.rs, crates/rdna-compute/src/attention.rs | developer |
| `HIPFIRE_FLUX_ATTN_GRID` | crates/rdna-compute/src/attention.rs | developer |
| `HIPFIRE_FLUX_F16_ACT` | crates/hipfire-arch-diffusion/examples/gpu_flux_block_parity.rs, crates/hipfire-arch-diffusion/examples/gpu_flux_forward.rs | developer |
| `HIPFIRE_FLUX_GEMM_LDS` | crates/hipfire-arch-diffusion/src/flux_gpu.rs | developer |
| `HIPFIRE_FLUX_GEMM_PIPE` | crates/rdna-compute/examples/test_gemm_wide_lds_parity.rs, crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_FLUX_GEMM_WIDE` | crates/hipfire-arch-diffusion/src/flux_gpu.rs | developer |
| `HIPFIRE_FLUX_GUIDANCE` | crates/hipfire-arch-diffusion/src/pipeline.rs | developer |
| `HIPFIRE_FLUX_MOD_GEMV` | crates/hipfire-arch-diffusion/src/flux_gpu.rs | developer |
| `HIPFIRE_FLUX_ROPE_FAST` | crates/rdna-compute/src/norm.rs | developer |
| `HIPFIRE_FLUX_WPAD` | crates/hipfire-arch-diffusion/examples/gpu_flux_forward.rs, crates/hipfire-arch-diffusion/src/flux_gpu.rs | developer |
| `HIPFIRE_FOO` | crates/hipfire-cli/src/main.rs | developer |
| `HIPFIRE_FORCE_ANSWER_SECS` | scripts/test-qwen35-think-cap.sh | harness |
| `HIPFIRE_FORCE_REBUILD` | crates/hipfire-cli/src/main.rs | developer |
| `HIPFIRE_FORCE_UNFUSED` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_FORWARD_LOWERED` | crates/hipfire-arch-deepseek4/examples/ds4_longctx_probe.rs, crates/hipfire-arch-deepseek4/examples/ds4_prod_vs_parent_trace.rs | developer |
| `HIPFIRE_FORWARD_ORACLE` | crates/hipfire-dispatch/src/pipeline/superop.rs | developer |
| `HIPFIRE_FP16` | crates/hipfire-arch-gemma4/examples/infer_gemma4_spec.rs, crates/hipfire-cli/src/serve/complete.rs | stable |
| `HIPFIRE_FP16_LAYER_MAX` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_FP16_LAYER_MIN` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_FP8_BV` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_FP8_DECODE_ATTN_GQA` | crates/rdna-compute/src/attention.rs | developer |
| `HIPFIRE_FP8_G3_PAD_TAIL` | crates/hipfire-runtime/examples/eval_hipfire.rs | harness |
| `HIPFIRE_FP8_GATEUP_KERNEL` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_FP8_LUT_ARG` | crates/hipfire-runtime/src/llama.rs, crates/hipfire-runtime/src/lloyd_lut.rs | developer |
| `HIPFIRE_FP8_PROD_INREG` | crates/hipfire-arch-qwen35/src/qwen35/prefill.rs, crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_FP8_PROD_SHORT` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_FP8_QKV` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_FP8_QKVZA` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_FP8_RESIDUAL` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_FP8_ROW_SCALE_SHIFT` | crates/rdna-compute/src/gemv.rs | developer |
| `HIPFIRE_FP8_SILU_H` | crates/rdna-compute/src/gemm.rs, crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_FP8_SILU_H_BF16` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_FP8_SLABS` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_FP8_STREAM` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_FP8_SYMFOLD` | crates/rdna-compute/src/gemm.rs, crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_FP8_V2_BK` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_FP8_V2_BM` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_FP8_V2_BN` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_FP8_V2_TILE` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_FP8_V2_WAVES` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_FP8_WMMA` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/examples/test_gemm_hfp4g32_fp8.rs | experimental |
| `HIPFIRE_FUSED_GATE_UP_K1024` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_FUSED_GATE_UP_K5120` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_FUSED_GATE_UP_KERNEL` | crates/rdna-compute/src/kernel_registry.rs, crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_FUSED_GATE_UP_PAIR_KERNEL` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_FUSE_QKV_BIAS` | crates/hipfire-config/src/lib.rs, crates/hipfire-dispatch/src/pipeline/steps.rs | stable |
| `HIPFIRE_FUSE_QKV_BIAS_DEBUG` | crates/hipfire-config/src/lib.rs, crates/hipfire-dispatch/src/pipeline/steps.rs | experimental |
| `HIPFIRE_G12_A4C2` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | stable |
| `HIPFIRE_G12_DEC_NORM` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | stable |
| `HIPFIRE_G12_FP8_F2_BUNDLE` | crates/rdna-compute/src/gemm.rs, crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_G12_FP8_ISA` | crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_G12_FP8_ISA_COVERAGE` | crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_G12_FP8_ISA_FORCE_SMALL_N` | crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_G12_IU4_B1S_BUNDLE` | crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_G12_IU4_B1_BUNDLE` | crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_G12_IU4_B1_CONTROL` | crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_G12_IU4_GDN_COVERAGE` | crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_G12_IU4_ISA` | crates/rdna-compute/src/feature_flags.rs | developer |
| `HIPFIRE_G12_IU4_V3` | crates/rdna-compute/src/gemm.rs, crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_G12_NORM` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | stable |
| `HIPFIRE_G12_RASTER` | crates/rdna-compute/src/gemm.rs, crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_GATED_NORM_FP8_AWQ` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_GATED_NORM_FP8_KERNEL` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_GATED_NORM_MQ_ROTATE` | crates/hipfire-arch-qwen35/src/qwen35/forward.rs | developer |
| `HIPFIRE_GATED_NORM_MQ_ROTATE_AWQ` | crates/rdna-compute/src/gemv.rs, crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_GATED_NORM_MQ_ROTATE_KERNEL` | crates/rdna-compute/src/gemv.rs, crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_GATED_NORM_WAVE_GROUP` | crates/rdna-compute/src/gemv.rs, crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_GATED_NORM_X_BF16` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_GATEUP_LDSSTAGE` | crates/rdna-compute/src/feature_flags.rs, crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_GATE_MODEL` | scripts/gates.sh | harness |
| `HIPFIRE_GATE_UP_BT` | crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_GATE_UP_NOSYNC` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/examples/bench_gate_up_nosync.rs | experimental |
| `HIPFIRE_GATE_UP_PAIR2` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_GATE_UP_VARIANT` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_GATE_WORK_DIR` | scripts/gates.sh | harness |
| `HIPFIRE_GCN5_WAVE64_HYBRID` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_GDN_BLK` | crates/rdna-compute/src/norm.rs | developer |
| `HIPFIRE_GDN_BLOCK_SIZE` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_GDN_CHUNKED` | crates/rdna-compute/src/kernels.rs, crates/rdna-compute/src/norm.rs | developer |
| `HIPFIRE_GDN_CHUNK_SIZE` | crates/rdna-compute/src/norm.rs | developer |
| `HIPFIRE_GDN_COMPACT2` | crates/hipfire-arch-qwen35/src/qwen35/forward.rs | developer |
| `HIPFIRE_GDN_COMPACT2_SHAPE` | crates/rdna-compute/src/norm.rs | developer |
| `HIPFIRE_GDN_COMPACT3` | crates/hipfire-arch-qwen35/src/qwen35/forward.rs | developer |
| `HIPFIRE_GDN_DPP_REDUCE` | crates/rdna-compute/src/dflash_gdn_replay.rs, crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_GDN_KERNEL` | crates/rdna-compute/src/dflash_gdn_replay.rs, crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_GDN_KKT_BATCHED` | crates/rdna-compute/src/norm.rs | developer |
| `HIPFIRE_GDN_KKT_GFX1100` | crates/rdna-compute/src/norm.rs | developer |
| `HIPFIRE_GDN_LAYER_TABLE` | crates/rdna-compute/src/dflash_gdn_replay.rs | developer |
| `HIPFIRE_GDN_MIN_BLOCKS` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_GDN_PREFETCH` | crates/rdna-compute/src/dflash_gdn_replay.rs, crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_GDN_PREP_FUSED` | crates/rdna-compute/src/gemm.rs, crates/rdna-compute/src/norm.rs | developer |
| `HIPFIRE_GDN_PREP_GFX11` | crates/rdna-compute/src/norm.rs | developer |
| `HIPFIRE_GDN_PRE_FUSE_OFF` | crates/rdna-compute/src/feature_flags.rs | developer |
| `HIPFIRE_GDN_QK_HEAD_DIV` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_GDN_REPLAY_ML_OFF` | crates/railgun-cert/src/recording.rs, crates/rdna-compute/src/dflash_gdn_replay.rs | developer |
| `HIPFIRE_GDN_SCAN_MSEG` | crates/rdna-compute/src/kernels.rs, crates/rdna-compute/src/norm.rs | developer |
| `HIPFIRE_GDN_SCAN_OUT` | crates/hipfire-arch-qwen35/src/qwen35/prefill.rs, crates/rdna-compute/src/norm.rs | developer |
| `HIPFIRE_GDN_SCAN_OUT_EMU` | crates/rdna-compute/src/kernels.rs, crates/rdna-compute/src/norm.rs | developer |
| `HIPFIRE_GDN_STATE_SRC` | crates/rdna-compute/src/dflash_gdn_replay.rs | developer |
| `HIPFIRE_GDN_TILE_ROWS` | crates/rdna-compute/src/dflash_gdn_replay.rs, crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_GDN_WAVES_PER_BLOCK` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_GEMMA4_ATTN_VERIFY` | crates/hipfire-arch-gemma4/src/lowered.rs | developer |
| `HIPFIRE_GEMMA4_BASELINE_ATTN` | crates/hipfire-arch-gemma4/src/forward.rs | developer |
| `HIPFIRE_GEMMA4_BATCHED_EMBEDDING_PREFILL` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_GEMMA4_DUMP` | crates/hipfire-arch-gemma4/examples/prefill_parity_gemma4.rs, crates/hipfire-arch-gemma4/src/lowered.rs | developer |
| `HIPFIRE_GEMMA4_EAGLE` | crates/hipfire-arch-gemma4/src/forward.rs, crates/hipfire-generate/src/dense.rs | developer |
| `HIPFIRE_GEMMA4_FUSED_ATTN_NORM` | crates/hipfire-arch-gemma4/src/forward.rs | developer |
| `HIPFIRE_GEMMA4_FUSED_FFN` | crates/hipfire-arch-gemma4/src/forward.rs | developer |
| `HIPFIRE_GEMMA4_FUSED_POSTNORM` | crates/hipfire-arch-gemma4/src/forward.rs | developer |
| `HIPFIRE_GEMMA4_FUSED_PROJ` | crates/hipfire-arch-gemma4/src/forward.rs | developer |
| `HIPFIRE_GEMMA4_FUSED_QK` | crates/hipfire-arch-gemma4/src/forward.rs | developer |
| `HIPFIRE_GEMMA4_FUSED_QK_ROPE` | crates/hipfire-arch-gemma4/src/forward.rs | developer |
| `HIPFIRE_GEMMA4_GEMM_VERIFY` | crates/hipfire-arch-gemma4/src/lowered.rs | developer |
| `HIPFIRE_GEMMA4_GRAPH` | crates/hipfire-arch-gemma4/examples/infer_gemma4.rs, crates/hipfire-arch-gemma4/src/forward.rs | developer |
| `HIPFIRE_GEMMA4_LOGIT_TRACE_DIR` | crates/hipfire-generate/src/dense.rs, scripts/diag-gemma4-logit-routes.sh | developer |
| `HIPFIRE_GEMMA4_LOGIT_TRACE_FULL_STEPS` | crates/hipfire-generate/src/dense.rs, scripts/diag-gemma4-logit-routes.sh | developer |
| `HIPFIRE_GEMMA4_LOGIT_TRACE_MAX_STEPS` | crates/hipfire-generate/src/dense.rs, scripts/diag-gemma4-logit-routes.sh | developer |
| `HIPFIRE_GEMMA4_LOGIT_TRACE_TOPK` | crates/hipfire-generate/src/dense.rs | developer |
| `HIPFIRE_GEMMA4_NORM_PLUS_ONE` | crates/hipfire-arch-gemma4/src/config.rs, crates/hipfire-arch-gemma4/src/drafter.rs | developer |
| `HIPFIRE_GEMMA4_PLE_ACTIVATION_FUSED_PREFILL` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_GEMMA4_PLE_BATCHED_PREFILL` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_GEMMA4_PLE_BRANCH_BATCHED_PREFILL` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_GEMMA4_PREFILL_BATCH` | crates/hipfire-generate/src/dense.rs, scripts/eval_gemma4_eseries.py | developer |
| `HIPFIRE_GEMMA4_Q8_FUSED_PREFILL` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_GEMM_DUMP` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_GEMV_DP4A` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | stable |
| `HIPFIRE_GEMV_PREFETCH` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | stable |
| `HIPFIRE_GEMV_ROWS` | crates/hipfire-cli/src/serve/complete.rs, crates/hipfire-config/src/lib.rs | experimental |
| `HIPFIRE_GEN` | crates/saddle-lab/examples/a3b_multiturn_oneshot.rs | harness |
| `HIPFIRE_GEN_STEPS` | crates/saddle-lab/examples/oracle_xcheck.rs | harness |
| `HIPFIRE_GEN_TIMEOUT` | docs/investigations/evidence/ds4-mi300x-cdna-test-fail/raw/a1-m0/04-profile-feed.py, docs/investigations/evidence/ds4-mi300x-cdna-test-fail/raw/a1-m0/04-run-profile-direct.sh | harness |
| `HIPFIRE_GFX1100_ASYM3_Q8_PAIR` | crates/rdna-compute/src/attention.rs | developer |
| `HIPFIRE_GFX1100_DECODE_ATTN_GQA` | crates/rdna-compute/src/attention.rs | developer |
| `HIPFIRE_GFX1100_DEC_NORM` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | stable |
| `HIPFIRE_GFX1100_DENSE_GATE_UP_DOT_REFORM` | crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_GFX1100_DENSE_GATE_UP_LANE0_HEADERS` | crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_GFX1100_DENSE_GATE_UP_PAIR` | crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_GFX1100_DENSE_GATE_UP_PAIR2` | crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_GFX1100_DENSE_GATE_UP_QUAD_PREFETCH` | crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_GFX1100_DENSE_GATE_UP_SETPRIO` | crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_GFX1100_DENSE_GATE_UP_STAGE_X32` | crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_GFX1100_FA2_R3` | crates/rdna-compute/src/attention.rs | developer |
| `HIPFIRE_GFX1100_FA_PREP` | crates/hipfire-arch-qwen35/src/qwen35/prefill.rs | developer |
| `HIPFIRE_GFX1100_GATED_NORM_V2` | crates/rdna-compute/src/gemv.rs, crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_GFX1100_MQ4V2_NINEPATH_RPB8` | crates/hipfire-runtime/examples/mq4v2_fused_parity.rs | harness |
| `HIPFIRE_GFX1100_MQ4_WIDE_PREFILL` | crates/hipfire-arch-qwen35/src/qwen35/prefill.rs | developer |
| `HIPFIRE_GFX1100_PACKED_MQ4_PREFILL` | crates/hipfire-arch-qwen35/src/qwen35/prefill.rs, crates/rdna-compute/src/feature_flags.rs | developer |
| `HIPFIRE_GFX1100_PM4_EXPERIMENTS` | crates/rdna-compute/src/replay.rs | developer |
| `HIPFIRE_GFX1100_PM_BUNDLE` | crates/rdna-compute/src/gemm.rs, crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_GFX1100_PM_GEMM` | crates/rdna-compute/src/gemm.rs, crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_GFX1100_ROUTER_W64` | crates/hipfire-dispatch/src/pipeline/moe_program.rs | developer |
| `HIPFIRE_GFX1151_ATTENTION_TILE_DPP` | crates/rdna-compute/src/attention.rs | developer |
| `HIPFIRE_GFX1151_ATTENTION_TILE_DPP_REDUCE` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_GFX1151_ATTN_SCOREGRID_LARGE_SERIAL` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_GFX1151_ATTN_SCOREGRID_XLANE` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_GFX1151_CUMODE_MODULES` | crates/rdna-compute/src/compiler.rs | developer |
| `HIPFIRE_GFX1151_DOWN_HYBRID_BUFFER` | crates/rdna-compute/src/gemv.rs | developer |
| `HIPFIRE_GFX1151_DOWN_ROW1_BUFFER` | crates/rdna-compute/src/gemv.rs | developer |
| `HIPFIRE_GFX1151_DOWN_ROW2_BUFFER` | crates/rdna-compute/src/gemv.rs | developer |
| `HIPFIRE_GFX1151_DOWN_ROW2_CLUSTERED` | crates/rdna-compute/src/gemv.rs | developer |
| `HIPFIRE_GFX1151_DOWN_ROW8` | crates/rdna-compute/src/gemv.rs | developer |
| `HIPFIRE_GFX1151_DOWN_TIGHT_GRID` | crates/rdna-compute/src/gemv.rs | developer |
| `HIPFIRE_GFX1151_E8_BUFFER` | crates/rdna-compute/src/feature_flags.rs | developer |
| `HIPFIRE_GFX1151_FA2_TWIN` | crates/rdna-compute/src/attention.rs, crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_GFX1151_FA_PREP` | crates/hipfire-arch-qwen35/src/qwen35/prefill.rs | developer |
| `HIPFIRE_GFX1151_GATE_UP_ALL_BUFFER` | crates/rdna-compute/src/gemv.rs | developer |
| `HIPFIRE_GFX1151_GATE_UP_HYBRID_BUFFER` | crates/rdna-compute/src/gemv.rs | developer |
| `HIPFIRE_GFX1151_GATE_UP_K2048` | crates/rdna-compute/src/gemv.rs | developer |
| `HIPFIRE_GFX1151_GATE_UP_LOW_VGPR` | crates/rdna-compute/src/gemv.rs | developer |
| `HIPFIRE_GFX1151_GATE_UP_PAIRED_WAVES` | crates/rdna-compute/src/gemv.rs | developer |
| `HIPFIRE_GFX1151_GATE_UP_PAIR_ALL_BUFFER` | crates/rdna-compute/src/gemv.rs | developer |
| `HIPFIRE_GFX1151_GATE_UP_PAIR_BUFFER` | crates/rdna-compute/src/gemv.rs | developer |
| `HIPFIRE_GFX1151_GATE_UP_PAIR_VGPR` | crates/rdna-compute/src/gemv.rs | developer |
| `HIPFIRE_GFX1151_GATE_UP_PERSISTENT_RANK8` | crates/rdna-compute/src/gemv.rs | developer |
| `HIPFIRE_GFX1151_GATE_UP_ROUTE_ALL_BUFFER` | crates/rdna-compute/src/gemv.rs | developer |
| `HIPFIRE_GFX1151_GATE_UP_SPLIT` | crates/rdna-compute/src/gemv.rs | developer |
| `HIPFIRE_GFX1151_GATE_UP_TIGHT_GRID` | crates/rdna-compute/src/gemv.rs | developer |
| `HIPFIRE_GFX1151_GATE_UP_WAVE64` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/gemv.rs | developer |
| `HIPFIRE_GFX1151_GDN_DPP` | crates/rdna-compute/src/norm.rs | developer |
| `HIPFIRE_GFX1151_GDN_R4X2` | crates/rdna-compute/src/norm.rs | developer |
| `HIPFIRE_GFX1151_GDN_R8` | crates/rdna-compute/src/norm.rs | developer |
| `HIPFIRE_GFX1151_GDN_SCAN` | crates/rdna-compute/src/kernels.rs, crates/rdna-compute/src/norm.rs | developer |
| `HIPFIRE_GFX1151_LM_HEAD_ALL_BUFFER` | crates/rdna-compute/src/gemv.rs | developer |
| `HIPFIRE_GFX1151_LM_HEAD_BUFFER` | crates/rdna-compute/src/gemv.rs | developer |
| `HIPFIRE_GFX1151_LM_HEAD_CPOL` | crates/rdna-compute/src/gemv.rs | developer |
| `HIPFIRE_GFX1151_LM_HEAD_DOT2` | crates/rdna-compute/src/gemv.rs | developer |
| `HIPFIRE_GFX1151_LM_HEAD_HYBRID_BUFFER` | crates/rdna-compute/src/gemv.rs, crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_GFX1151_LM_HEAD_K2048` | crates/rdna-compute/src/gemv.rs, crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_GFX1151_LM_HEAD_X_BUFFER` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_GFX1151_MOE_DOWN_HYBRID_BUFFER` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_GFX1151_MOE_GATE_UP_HYBRID_BUFFER` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_GFX1151_MOE_TIGHT_GRID` | crates/rdna-compute/src/gemv.rs | developer |
| `HIPFIRE_GFX1151_PM4_ENTRY_ACQUIRE` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/replay.rs | experimental |
| `HIPFIRE_GFX1151_PM4_INITIATOR` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/replay.rs | experimental |
| `HIPFIRE_GFX1151_PM4_INTERLEAVE` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/replay.rs | experimental |
| `HIPFIRE_GFX1151_PM4_RESOURCE_LIMITS` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/replay.rs | experimental |
| `HIPFIRE_GFX1151_Q8_DECODE_ATTN_GQA` | crates/rdna-compute/src/attention.rs | developer |
| `HIPFIRE_GFX1151_QKVZA_ALL_BUFFER_CPOL` | crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_GFX1151_QKVZA_HYBRID_BUFFER` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_GFX1151_QKVZA_K2048_HOIST` | crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_GFX1151_QKVZA_LDSX8_BUFFER` | crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_GFX1151_QKVZA_PAIR_BUFFER` | crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_GFX1151_QKVZA_R2` | crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_GFX1151_QKVZA_R2_BUFFER` | crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_GFX1151_QKVZA_R4_STREAM` | crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_GFX1151_QKVZA_WAVE64` | crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_GFX1151_QKVZA_WAVE64_SHARE_X` | crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_GFX1151_QKVZA_X_BUFFER_LARGE` | crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_GFX1151_QKV_ALL_BUFFER_CPOL` | crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_GFX1151_QKV_X_BUFFER` | crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_GFX1151_REDLINE_CU_COUNT` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/replay.rs | experimental |
| `HIPFIRE_GFX1151_RESIDUAL_HYBRID_BUFFER` | crates/rdna-compute/src/gemv.rs, crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_GFX1151_RESIDUAL_K4096` | crates/rdna-compute/src/gemv.rs, crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_GFX1151_RESIDUAL_MULTIROW_R2` | crates/rdna-compute/src/gemv.rs | developer |
| `HIPFIRE_GFX1151_RESIDUAL_ROW1` | crates/rdna-compute/src/gemv.rs | developer |
| `HIPFIRE_GFX1151_RESIDUAL_ROW_SERIAL` | crates/rdna-compute/examples/test_mq4v2_residual_row_serial_gfx1151.rs, crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_GFX1151_RESIDUAL_TIGHT_GRID` | crates/rdna-compute/src/gemv.rs | developer |
| `HIPFIRE_GFX1151_RESIDUAL_WAVE64` | crates/rdna-compute/src/gemv.rs | developer |
| `HIPFIRE_GFX1151_WEIGHT_BUFFER_DOWN` | crates/rdna-compute/src/gemv.rs | developer |
| `HIPFIRE_GFX1151_WEIGHT_BUFFER_GATE_UP` | crates/rdna-compute/src/gemv.rs | developer |
| `HIPFIRE_GFX1151_WEIGHT_BUFFER_LOADS` | crates/rdna-compute/src/gemm.rs, crates/rdna-compute/src/gemv.rs | developer |
| `HIPFIRE_GFX1151_WEIGHT_BUFFER_QKVZA` | crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_GFX1151_WEIGHT_BUFFER_RESIDUAL` | crates/rdna-compute/src/gemv.rs | developer |
| `HIPFIRE_GFX1151_WEIGHT_BUFFER_SIGMOID` | crates/rdna-compute/src/gemv.rs | developer |
| `HIPFIRE_GFX11_A4_CANDIDATES` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_GFX11_FA2_PREFILL` | crates/hipfire-config/src/lib.rs, crates/hipfire-dispatch/src/families/attention.rs | stable |
| `HIPFIRE_GFX11_IU4_GRIDSPEC` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | stable |
| `HIPFIRE_GFX11_IU4_SHAPE` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_GFX11_LEAN_PBS` | crates/hipfire-arch-qwen35/src/qwen35/prefill.rs, crates/hipfire-config/src/lib.rs | experimental |
| `HIPFIRE_GFX11_MMQ_X128` | crates/rdna-compute/src/dispatch.rs, crates/rdna-compute/src/feature_flags.rs | developer |
| `HIPFIRE_GFX11_PRODUCER_QUANT_FUSED` | crates/hipfire-arch-qwen35/src/qwen35/prefill.rs, crates/hipfire-config/src/lib.rs | stable |
| `HIPFIRE_GFX11_Q8_FA2_WIDE` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_GFX11_WEIGHT_LOAD_POLICY` | benchmarks/scripts/mq4v2_k5120_abba.sh, crates/hipfire-config/src/lib.rs | experimental |
| `HIPFIRE_GFX1201_PM4_PACING` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/replay.rs | stable |
| `HIPFIRE_GFX1201_ROUTER_W64` | crates/hipfire-dispatch/src/pipeline/moe_program.rs | developer |
| `HIPFIRE_GFX12_FA2_FP8` | crates/hipfire-runtime/examples/tmp_fa2_fp8_oracle.rs | harness |
| `HIPFIRE_GFX12_FA2_PREFILL` | crates/hipfire-config/src/lib.rs, crates/hipfire-dispatch/src/families/attention.rs | stable |
| `HIPFIRE_GFX12_FA2_SPLIT_COUNT` | crates/hipfire-arch-qwen35/src/qwen35/prefill.rs | developer |
| `HIPFIRE_GFX12_FA2_SPLIT_VERIFY` | crates/hipfire-arch-qwen35/src/qwen35/prefill.rs, crates/rdna-compute/src/attention.rs | developer |
| `HIPFIRE_GFX12_FA_PACKET` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | stable |
| `HIPFIRE_GFX12_FA_PREP_FP8Q` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | stable |
| `HIPFIRE_GFX12_FA_PREP_FUSED` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | stable |
| `HIPFIRE_GFX12_FP8_STREAM` | crates/hipfire-config/src/lib.rs, crates/hipfire-runtime/examples/fp8stream_producer_oracle.rs | stable |
| `HIPFIRE_GFX12_GDN_CHUNK_SCAN` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | stable |
| `HIPFIRE_GFX12_GDN_PRE_FUSED` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | stable |
| `HIPFIRE_GFX12_MQ4V2_FP8_GATEUP` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | stable |
| `HIPFIRE_GFX12_MQ4V2_FP8_QKV` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | stable |
| `HIPFIRE_GFX12_MQ4V2_FP8_QKVZA` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | stable |
| `HIPFIRE_GFX12_MQ4V2_FP8_RESID` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | stable |
| `HIPFIRE_GFX12_MQ4V2_FP8_SLABS` | crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_GFX12_MQ4V2_FP8_V2` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | stable |
| `HIPFIRE_GFX12_MQ4V2_FP8_V2_GEOM` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_GFX12_PRODUCER_QUANT_FUSED` | crates/hipfire-arch-qwen35/src/qwen35/prefill.rs, crates/hipfire-config/src/lib.rs | stable |
| `HIPFIRE_GFX12_SILU_QUANT_FUSED` | crates/hipfire-arch-qwen35/src/qwen35/prefill.rs, crates/hipfire-config/src/lib.rs | stable |
| `HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE` | crates/rdna-compute/examples/bench_vgpr_cap_sweep.rs, crates/rdna-compute/examples/mq4c_parity.rs | developer |
| `HIPFIRE_GFX12_WEIGHT_LOAD_POLICY` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_GFX942_GEMV_V2` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_GFX942_GEMV_V3` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_GFX942_LDS_GEMV` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_GFX942_MFMA_PREFILL` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_GFX942_MQ2_DOWN_ALLRANKS` | docs/investigations/evidence/ds4-mi300x-cdna-test-fail/raw/a1-m0/run-det.sh | harness |
| `HIPFIRE_GFX942_RMSNORM_SPLIT` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_GFX942_ROTATE_VALIDATE_LIVE` | crates/rdna-compute/src/gemv.rs | developer |
| `HIPFIRE_GITHUB_URL` | scripts/install.sh, scripts/test_install_revision.py | harness |
| `HIPFIRE_GLIMMER` | crates/hipfire-runtime/examples/build_kld_ref_native_glimmer.rs, crates/hipfire-runtime/examples/eval_hipfire_glimmer.rs | harness |
| `HIPFIRE_GLIMMER_ACCEPT_DIAG` | crates/hipfire-arch-muse-glimmer/src/drafter.rs | developer |
| `HIPFIRE_GLIMMER_BATCHED_LM_HEAD` | crates/hipfire-arch-muse-glimmer/src/forward.rs | developer |
| `HIPFIRE_GLIMMER_CACHE_TRACE` | crates/hipfire-generate/src/dense.rs, scripts/serve_harness.py | developer |
| `HIPFIRE_GLIMMER_CTX_CAP` | crates/hipfire-arch-muse-glimmer/src/drafter.rs, crates/hipfire-generate/src/dense.rs | developer |
| `HIPFIRE_GLIMMER_DEVICE_CAPTURE` | crates/hipfire-loader/src/carriers.rs, crates/hipfire-runtime/examples/dflash_spec_demo.rs | developer |
| `HIPFIRE_GLIMMER_DEVICE_CAPTURE_AUDIT` | crates/hipfire-arch-muse-glimmer/src/glimmer.rs, crates/hipfire-generate/src/dense.rs | developer |
| `HIPFIRE_GLIMMER_FLASH_DECODE` | crates/hipfire-arch-muse-glimmer/src/forward.rs | developer |
| `HIPFIRE_GLIMMER_FLASH_FULL` | crates/hipfire-arch-muse-glimmer/src/forward.rs | developer |
| `HIPFIRE_GLIMMER_FUSED_POSTNORM` | crates/hipfire-arch-muse-glimmer/src/forward.rs | developer |
| `HIPFIRE_GLIMMER_FUSED_QK_ROPE` | crates/hipfire-arch-muse-glimmer/src/forward.rs | developer |
| `HIPFIRE_GLIMMER_FUSED_SILU_ROTATE` | crates/hipfire-arch-muse-glimmer/src/forward.rs | developer |
| `HIPFIRE_GLIMMER_GATE_UP_K6656` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_GLIMMER_GPU_SAMPLE` | crates/hipfire-generate/src/dense.rs | developer |
| `HIPFIRE_GLIMMER_KV_VMM` | crates/hipfire-arch-muse-glimmer/src/glimmer.rs, crates/hipfire-runtime/examples/build_kld_ref_native_glimmer.rs | developer |
| `HIPFIRE_GLIMMER_MUSE_GEMM` | crates/hipfire-arch-muse-glimmer/src/forward.rs | developer |
| `HIPFIRE_GLIMMER_NO_ATTN_GATE` | crates/hipfire-arch-muse-glimmer/src/forward.rs | developer |
| `HIPFIRE_GLIMMER_NO_CENTERED_NORM` | crates/hipfire-arch-muse-glimmer/src/glimmer.rs | developer |
| `HIPFIRE_GLIMMER_NO_EMBED_NORM` | crates/hipfire-arch-muse-glimmer/src/forward.rs | developer |
| `HIPFIRE_GLIMMER_NO_FLASH` | crates/hipfire-arch-muse-glimmer/src/forward.rs | developer |
| `HIPFIRE_GLIMMER_NO_OUTMUL` | crates/hipfire-arch-muse-glimmer/src/forward.rs | developer |
| `HIPFIRE_GLIMMER_NO_QK_NORM` | crates/hipfire-arch-muse-glimmer/src/forward.rs | developer |
| `HIPFIRE_GLIMMER_NO_QK_SCALE` | crates/hipfire-arch-muse-glimmer/src/forward.rs | developer |
| `HIPFIRE_GLIMMER_NO_SOFTCAP` | crates/hipfire-arch-muse-glimmer/src/forward.rs | developer |
| `HIPFIRE_GLIMMER_O_RESIDUAL` | crates/hipfire-arch-muse-glimmer/src/forward.rs | developer |
| `HIPFIRE_GLIMMER_PREFILL_CHUNK` | crates/hipfire-arch-muse-glimmer/src/forward.rs | developer |
| `HIPFIRE_GLIMMER_PROMPT_CACHE` | crates/hipfire-generate/src/dense.rs | developer |
| `HIPFIRE_GLIMMER_QKVG_K6656` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_GLIMMER_ROPE_ALL` | crates/hipfire-arch-muse-glimmer/src/forward.rs | developer |
| `HIPFIRE_GLIMMER_ROPE_INTERLEAVED` | crates/hipfire-arch-muse-glimmer/src/forward.rs | developer |
| `HIPFIRE_GLIMMER_SHARED_ROT` | crates/hipfire-arch-muse-glimmer/src/drafter.rs, crates/hipfire-arch-muse-glimmer/src/forward.rs | developer |
| `HIPFIRE_GLIMMER_SPEC_DIAG` | crates/hipfire-generate/src/dense.rs, crates/hipfire-runtime/examples/dflash_spec_demo.rs | developer |
| `HIPFIRE_GLIMMER_SPEC_FALLBACK` | crates/hipfire-generate/src/dense.rs | developer |
| `HIPFIRE_GLIMMER_SPEC_PERTURB` | crates/hipfire-generate/src/dense.rs | developer |
| `HIPFIRE_GLIMMER_SPEC_PROFIT_GUARD` | crates/hipfire-generate/src/dense.rs | developer |
| `HIPFIRE_GLIMMER_SWA_PREFETCH_V` | crates/rdna-compute/examples/bench_glimmer_verify_attention.rs, crates/rdna-compute/src/attention.rs | developer |
| `HIPFIRE_GLIMMER_TAP_LAYERS` | crates/hipfire-generate/src/dense.rs | developer |
| `HIPFIRE_GLIMMER_TIMING` | crates/hipfire-arch-muse-glimmer/src/drafter.rs, crates/hipfire-arch-muse-glimmer/src/forward.rs | developer |
| `HIPFIRE_GLIMMER_WMMA_FULL` | crates/hipfire-arch-muse-glimmer/src/forward.rs | developer |
| `HIPFIRE_GPTQ_BLOCK` | crates/hipfire-quantize/src/gptq.rs | developer |
| `HIPFIRE_GPTQ_DAMPING` | crates/hipfire-quantize/src/quant_mq.rs | developer |
| `HIPFIRE_GPUS` | scripts/ab-dispatch-validation.sh | harness |
| `HIPFIRE_GPU_0_BIG` | scripts/ab-dispatch-validation.sh | harness |
| `HIPFIRE_GPU_LAYER_BUDGET` | crates/hipfire-arch-qwen35/src/qwen35/config.rs, crates/hipfire-config/src/lib.rs | stable |
| `HIPFIRE_GPU_LOCKFILE` | autoresearch/ar/swarm.py, scripts/container-gate.sh | harness |
| `HIPFIRE_GPU_LOCK_OWNER` | scripts/gpu-lock.sh, scripts/pp-gate.sh | harness |
| `HIPFIRE_GPU_LOCK_PATH` | scripts/gpu-lock.sh | harness |
| `HIPFIRE_GPU_TOPK` | crates/saddle-lab/examples/infer_qwen35.rs | harness |
| `HIPFIRE_GQA_CHUNK` | crates/rdna-compute/src/attention.rs | developer |
| `HIPFIRE_GQA_FUSED` | crates/hipfire-arch-qwen2/src/qwen2.rs, crates/hipfire-dispatch/src/families/kv_tier.rs | developer |
| `HIPFIRE_GRAPH` | benchmarks/scripts/bench_pp_gfx906.sh, crates/hipfire-arch-gemma4/src/lowered.rs | experimental |
| `HIPFIRE_GRAPH_MOE` | crates/hipfire-arch-qwen35/src/qwen35/forward.rs, crates/hipfire-config/src/lib.rs | experimental |
| `HIPFIRE_GRAPH_PREFILL` | crates/hipfire-runtime/examples/bench_qwen35_mq4.rs | harness |
| `HIPFIRE_GRID_TOKFAST` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_GTT_CEILING_GIB` | scripts/run-bounded.sh | harness |
| `HIPFIRE_HAVE_2_GPU` | crates/hipfire-arch-qwen35/tests/pp_parity.rs, crates/hipfire-arch-qwen35/tests/route_oracle_mesh.rs | harness |
| `HIPFIRE_HC_CTRL_T1024` | crates/rdna-compute/src/attention.rs, scripts/analysis/ds4-gfx1151-roofline/hcctrl.sh | developer |
| `HIPFIRE_HFQ` | benchmarks/vision/run_bench.sh | harness |
| `HIPFIRE_HFQ3_DP4A` | crates/hipfire-config/src/lib.rs, crates/hipfire-runtime/examples/verify_hfq3_batched.rs | experimental |
| `HIPFIRE_HFQ3_MMQ` | crates/hipfire-config/src/lib.rs, crates/hipfire-runtime/examples/bench_hfq3_mmq_sweep.rs | experimental |
| `HIPFIRE_HFQ3_MMQ_LAYER_MAX` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_HFQ3_MMQ_LAYER_MIN` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_HFQ4G128_MMQ` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | stable |
| `HIPFIRE_HFQ4G256_K2048` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_HFQ4G256_KERNEL` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_HFQ4G256_LDSSTAGE` | crates/rdna-compute/src/feature_flags.rs, crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_HFQ4G256_XBATCH` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_HFQ4G256_XBATCH_KERNEL` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_HFQ4G256_XBATCH_MAX` | crates/rdna-compute/src/gemv.rs, crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_HFQ4_DOT_REFORM` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_HFQ4_LANE0_HEADERS` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_HFQ4_MMQ_GFX906_Y64` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_HFQ4_MMQ_RDNA2` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_HFQ4_QUAD_PREFETCH` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_HFQ4_SETPRIO` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_HFQ4_STAGE_X32` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_HFQ6_EXPERT_BINS` | crates/rdna-compute/examples/test_moe_grouped_wmma_hfq6.rs | harness |
| `HIPFIRE_HFQ6_REAL_K` | crates/rdna-compute/examples/test_moe_grouped_wmma_hfq6.rs | harness |
| `HIPFIRE_HFQ6_REAL_LABEL` | crates/rdna-compute/examples/test_moe_grouped_wmma_hfq6.rs | harness |
| `HIPFIRE_HFQ6_REAL_M` | crates/rdna-compute/examples/test_moe_grouped_wmma_hfq6.rs | harness |
| `HIPFIRE_HFQ6_REAL_M_TOTAL` | crates/rdna-compute/examples/test_moe_grouped_wmma_hfq6.rs | harness |
| `HIPFIRE_HFQ6_REAL_X_ROW_DIV` | crates/rdna-compute/examples/test_moe_grouped_wmma_hfq6.rs | harness |
| `HIPFIRE_HF_BASE` | crates/hipfire-cli/src/main.rs, crates/hipfire-config/src/lib.rs | stable |
| `HIPFIRE_HIDDEN_SCATTER_FUSE_OFF` | crates/hipfire-arch-qwen35/examples/test_dflash_hidden_scatter_gfx1100.rs, crates/rdna-compute/src/feature_flags.rs | developer |
| `HIPFIRE_HIPCC` | crates/hipfire-cli/src/main.rs, crates/hipfire-cli/src/setup.rs | developer |
| `HIPFIRE_HIPCC_EXTRA_FLAGS` | .agents/skills/hipfire-autoheal/triage.sh, crates/hipfire-config/src/lib.rs | experimental |
| `HIPFIRE_HIP_WAIT` | crates/rdna-compute/src/dispatch.rs | developer |
| `HIPFIRE_HOME` | crates/hipfire-config/src/lib.rs, crates/hipfire-registry/src/lib.rs | stable |
| `HIPFIRE_HOST` | crates/hipfire-config/src/lib.rs | stable |
| `HIPFIRE_HOST_TIMING` | crates/hipfire-generate/src/qwen.rs, crates/hipfire-runtime/examples/dflash_spec_demo.rs | developer |
| `HIPFIRE_IDLE_TIMEOUT` | crates/hipfire-config/src/lib.rs | stable |
| `HIPFIRE_IMAGE` | scripts/container-gate.sh | harness |
| `HIPFIRE_IMAGE_DECODE` | crates/hipfire-arch-qwen35-vl/src/image.rs, crates/hipfire-config/src/lib.rs | stable |
| `HIPFIRE_IMG_COND_CACHE` | crates/hipfire-arch-diffusion/src/pipeline.rs, crates/hipfire-generate/src/img.rs | developer |
| `HIPFIRE_IMG_PROFILE` | crates/hipfire-arch-diffusion/src/flux_gpu.rs, crates/hipfire-arch-diffusion/src/pipeline.rs | developer |
| `HIPFIRE_INSTALL_REF` | scripts/install.sh, scripts/test_install_revision.py | harness |
| `HIPFIRE_ISA_MODULE` | crates/hipfire-runtime/examples/tmp_iu4_gfx12_v3_oracle.rs | harness |
| `HIPFIRE_ISA_SYMBOL_SUFFIX` | crates/hipfire-runtime/examples/tmp_iu4_gfx12_v3_oracle.rs | harness |
| `HIPFIRE_ISA_TILE_ROWS` | crates/hipfire-runtime/examples/tmp_iu4_gfx12_v3_oracle.rs | harness |
| `HIPFIRE_IU4_BAFOLD` | crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_IU4_ONEPASS` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_IU4_PREFILL` | crates/hipfire-arch-qwen35/src/qwen35/prefill.rs, crates/hipfire-config/src/lib.rs | stable |
| `HIPFIRE_IU4_RTN_RCP` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_IU4_SIDECAR` | crates/hipfire-isa/src/kernels/iu4_v2b_a4.rs, crates/rdna-compute/src/gemv.rs | developer |
| `HIPFIRE_IU4_SLAB` | crates/rdna-compute/src/kernels.rs, crates/rdna-compute/src/scratch.rs | developer |
| `HIPFIRE_IU4_SYMFOLD` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_IU4_V2B` | crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_IU4_V2C` | crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_IU4_X5` | crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_JINJA_CHAT` | crates/hipfire-config/src/lib.rs, crates/hipfire-engine/src/prompt.rs | stable |
| `HIPFIRE_JINJA_TOOLS_DRAFTER` | scripts/agentic-gate-jinja-tools.sh | harness |
| `HIPFIRE_JINJA_TOOLS_MODEL` | scripts/agentic-gate-jinja-tools.sh | harness |
| `HIPFIRE_KERNEL_CACHE` | benchmarks/scripts/mq4v2_k5120_abba.sh, crates/hipfire-config/src/lib.rs | stable |
| `HIPFIRE_KLD_NGL` | crates/hipfire-runtime/examples/build_kld_ref.rs, crates/hipfire-runtime/examples/eval_gguf.rs | harness |
| `HIPFIRE_KLD_TEACHER` | benchmarks/quality-baselines/harness/spe_ablation.sh | harness |
| `HIPFIRE_KV` | crates/saddle-lab/examples/oracle_xcheck.rs | harness |
| `HIPFIRE_KV_ADAPTIVE` | crates/hipfire-arch-qwen35/src/carrier.rs, crates/hipfire-config/src/lib.rs | stable |
| `HIPFIRE_KV_BACKEND` | crates/hipfire-loader/src/admission.rs, scripts/guard_gfx1201_baseline.py | developer |
| `HIPFIRE_KV_BF16` | crates/hipfire-dispatch/src/families/attention.rs, crates/hipfire-dispatch/src/families/kv_tier.rs | developer |
| `HIPFIRE_KV_FP8_E4M3` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_KV_MODE` | benchmarks/scripts/bench_pp_gfx906.sh, crates/hipfire-arch-qwen35/src/serve_engine.rs | stable; `asymN`/`turbo*` values deprecated, removal 0.5.0 |
| `HIPFIRE_KV_PHYSICAL_CAP` | crates/hipfire-runtime/src/loader_api.rs | developer |
| `HIPFIRE_KV_SEQ` | crates/hipfire-arch-gemma4/src/carrier.rs, crates/hipfire-arch-gemma4/src/lowered.rs | developer |
| `HIPFIRE_KV_SLOT_PAGED` | crates/rdna-compute/src/kernel_registry.rs, crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_KV_V` | crates/hipfire-arch-qwen35/src/carrier.rs, crates/hipfire-loader/src/admission.rs | developer |
| `HIPFIRE_LABEL` | docs/investigations/evidence/ds4-mi300x-cdna-test-fail/raw/a1-m0/04-profile-feed.py, docs/investigations/evidence/ds4-mi300x-cdna-test-fail/raw/a1-m0/04-run-profile-direct.sh | harness |
| `HIPFIRE_LDS_EPI_DIRECT` | crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_LFM2_CAPTURE_POSTMIXER` | crates/hipfire-arch-lfm2moe/examples/dump_lfm2moe_hidden_states.rs, crates/hipfire-arch-lfm2moe/src/forward.rs | developer |
| `HIPFIRE_LFM2_GRAPH` | crates/hipfire-arch-lfm2moe/examples/graph_parity_lfm2moe.rs, crates/hipfire-arch-lfm2moe/src/forward.rs | developer |
| `HIPFIRE_LLOYD_FORCE_BASELINE` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/examples/test_gemv_mq4g256_lloyd_tail.rs | experimental |
| `HIPFIRE_LLOYD_K3` | crates/hipfire-quantize/src/pipeline.rs, crates/hipfire-quantize/src/quant_mq.rs | developer |
| `HIPFIRE_LLOYD_MB4` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/examples/test_gemm_mq4g256_lloyd_residual_wmma.rs | experimental |
| `HIPFIRE_LLOYD_MMQ_OFF` | crates/hipfire-arch-qwen35/src/qwen35/prefill.rs, crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_LM_HEAD_F16` | crates/hipfire-cli/src/serve/complete.rs, crates/hipfire-config/src/lib.rs | stable |
| `HIPFIRE_LM_HEAD_OVERWRITE` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_LM_HEAD_WMMA` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | stable |
| `HIPFIRE_LOAD_TIMEOUT` | docs/investigations/evidence/ds4-mi300x-cdna-test-fail/raw/a1-m0/04-profile-feed.py, docs/investigations/evidence/ds4-mi300x-cdna-test-fail/raw/a1-m0/04-run-profile-direct.sh | harness |
| `HIPFIRE_LOAD_TRACE` | crates/hipfire-arch-qwen35/src/qwen35/load.rs | developer |
| `HIPFIRE_LOCAL` | benchmarks/scripts/mq4v2_k5120_abba.sh, crates/hipfire-cli/src/main.rs | stable |
| `HIPFIRE_LOCK_DIR` | crates/hipfire-daemon/src/gpu_lock.rs, crates/npu-tools/src/m4.rs | developer |
| `HIPFIRE_LOG` | crates/hipfire-daemon/src/main.rs | developer |
| `HIPFIRE_LOG_FORMAT` | crates/hipfire-daemon/src/main.rs, scripts/check-env-docs.py | developer |
| `HIPFIRE_LOWBIT_WMMA_WAVES` | crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_MAGIC` | crates/hipfire-runtime/examples/build_kld_ref.rs, crates/hipfire-runtime/examples/build_kld_ref_native_gemma4.rs | harness |
| `HIPFIRE_MAPLE_DOWN` | crates/hipfire-arch-maple/src/forward.rs | developer |
| `HIPFIRE_MAPLE_DUMP_HIDDEN` | crates/hipfire-arch-maple/src/forward.rs, tools/models/maple/compare_hidden.py | developer |
| `HIPFIRE_MAPLE_DUMP_ROUTER` | crates/hipfire-arch-maple/src/forward.rs | developer |
| `HIPFIRE_MAPLE_FORCE_FULL_CAUSAL` | crates/hipfire-arch-maple/examples/maple_prefill_parity.rs, crates/hipfire-arch-maple/src/forward.rs | developer |
| `HIPFIRE_MAPLE_PER_TOKEN_PREFILL` | crates/hipfire-arch-maple/examples/maple_coherence.rs | harness |
| `HIPFIRE_MAX_REQUEST_BYTES` | crates/hipfire-config/src/lib.rs | stable |
| `HIPFIRE_MAX_TOKENS` | docs/investigations/evidence/ds4-mi300x-cdna-test-fail/raw/a1-m0/04-profile-feed.py, docs/investigations/evidence/ds4-mi300x-cdna-test-fail/raw/a1-m0/04-run-profile-direct.sh | harness |
| `HIPFIRE_MAX_TOTAL_THINK_TOKENS` | crates/hipfire-config/src/lib.rs, crates/hipfire-runtime/src/config.rs | stable |
| `HIPFIRE_MEMSET_DUMP` | crates/hip-bridge/src/ffi.rs, crates/rdna-compute/src/dispatch.rs | developer |
| `HIPFIRE_MEM_CAP` | scripts/run-bounded.sh, scripts/serve_concurrency_gate.sh | harness |
| `HIPFIRE_MEM_MARGIN_GIB` | scripts/run-bounded.sh | harness |
| `HIPFIRE_META_MAX` | scripts/ddtree_meta_sweep.sh | harness |
| `HIPFIRE_META_RUNS` | scripts/ddtree_meta_sweep.sh | harness |
| `HIPFIRE_MINIMAX_BATCH_PREFILL` | crates/hipfire-generate/src/dense.rs | developer |
| `HIPFIRE_MINIMAX_CAPTURE_POSTATTN` | crates/hipfire-arch-minimax/src/forward.rs | developer |
| `HIPFIRE_MINIMAX_ENABLE_DOWN_AWQ` | crates/hipfire-arch-minimax/src/minimax.rs | developer |
| `HIPFIRE_MINIMAX_EXPERT_MQ2L` | crates/hipfire-quantize/src/pipeline.rs | developer |
| `HIPFIRE_MINIMAX_EXPERT_MQ3L` | crates/hipfire-quantize/src/pipeline.rs | developer |
| `HIPFIRE_MINIMAX_EXPERT_MQ6` | crates/hipfire-quantize/src/pipeline.rs | developer |
| `HIPFIRE_MINIMAX_GATE_OUT` | scripts/coherence-gate-minimax.sh | deprecated |
| `HIPFIRE_MINIMAX_GRAPH` | crates/hipfire-arch-minimax/src/forward.rs, crates/hipfire-generate/src/dense.rs | developer |
| `HIPFIRE_MINIMAX_MODEL` | scripts/coherence-gate-minimax.sh | deprecated |
| `HIPFIRE_MINIMAX_PROMOTE_MQ4` | crates/hipfire-quantize/src/pipeline.rs | developer |
| `HIPFIRE_MINIMAX_PROMOTE_MQ6` | crates/hipfire-quantize/src/pipeline.rs | developer |
| `HIPFIRE_MMQ` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | stable |
| `HIPFIRE_MMQ_DIAG_QUANTIZE_ONLY` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_MMQ_DUMP` | crates/rdna-compute/examples/test_gfx906_mmq_realdata.rs | harness |
| `HIPFIRE_MMQ_LUT` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_MMQ_LUT_PAIRTAB` | crates/rdna-compute/src/gemm.rs, crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_MMQ_MIN_BATCH` | benchmarks/scripts/bench_dflash_27b_gfx906.sh, crates/hipfire-config/src/lib.rs | experimental |
| `HIPFIRE_MMQ_SCREEN` | crates/hipfire-config/src/lib.rs, crates/hipfire-daemon/src/main.rs | stable |
| `HIPFIRE_MMQ_SCREEN_THRESHOLD` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | stable |
| `HIPFIRE_MODEL` | crates/hipfire-config/src/lib.rs, docs/investigations/evidence/ds4-mi300x-cdna-test-fail/raw/a1-m0/04-profile-feed.py | stable |
| `HIPFIRE_MODELS_DIR` | autoresearch/ar/gate/run.py, benchmarks/quality-baselines/harness/spe_ablation.sh | stable |
| `HIPFIRE_MODEL_PATH` | autoresearch/ar/census.py | harness |
| `HIPFIRE_MODEL_STORE` | scripts/baseline_quant_smoke.sh | harness |
| `HIPFIRE_MOE_AWQ` | crates/hipfire-arch-qwen35/src/qwen35/load.rs | developer |
| `HIPFIRE_MOE_BUCKETED` | crates/hipfire-arch-gemma4/src/lowered.rs | developer |
| `HIPFIRE_MOE_BYPASS` | crates/hipfire-arch-gemma4/src/lowered.rs | developer |
| `HIPFIRE_MOE_CODEBOOK_BATCHED` | crates/hipfire-arch-qwen35/src/qwen35/prefill.rs, crates/hipfire-dispatch/src/families/moe.rs | developer |
| `HIPFIRE_MOE_COMBINE_NEXT_RMS` | crates/hipfire-arch-qwen35/src/qwen35/forward.rs | developer |
| `HIPFIRE_MOE_COMBINE_NEXT_RMS_RENORM` | crates/hipfire-arch-qwen35/src/qwen35/forward.rs | developer |
| `HIPFIRE_MOE_COMBINE_RMSNORM_MQ_KERNEL` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_MOE_DOWN_COMBINE_VEC4` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_MOE_DOWN_CPOL` | crates/rdna-compute/src/gemv.rs | developer |
| `HIPFIRE_MOE_DOWN_FUSED` | crates/rdna-compute/src/gemv.rs, crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_MOE_DOWN_KERNEL` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_MOE_DOWN_LAST_COMBINE` | crates/hipfire-arch-qwen35/src/qwen35/forward.rs, crates/hipfire-dispatch/src/pipeline/moe_program.rs | developer |
| `HIPFIRE_MOE_DOWN_MQ5` | crates/hipfire-quantize/src/pipeline.rs | developer |
| `HIPFIRE_MOE_DOWN_MQ6` | crates/hipfire-quantize/src/pipeline.rs | developer |
| `HIPFIRE_MOE_DOWN_ROW2_CLUSTERED` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_MOE_DOWN_ROW2_KERNEL` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_MOE_DOWN_TIGHT_GRID` | crates/rdna-compute/src/gemv.rs | developer |
| `HIPFIRE_MOE_EXPERTS_MQ5` | crates/hipfire-quantize/src/pipeline.rs | developer |
| `HIPFIRE_MOE_EXPERTS_MQ6` | crates/hipfire-quantize/src/pipeline.rs | developer |
| `HIPFIRE_MOE_EXPERT_STATS` | crates/hipfire-arch-qwen35/src/qwen35/forward.rs | developer |
| `HIPFIRE_MOE_EXPERT_STATS_OUT` | crates/hipfire-arch-qwen35/src/qwen35/forward.rs, crates/hipfire-runtime/examples/eval_hipfire.rs | developer |
| `HIPFIRE_MOE_GATE_UP_CPOL` | crates/rdna-compute/src/gemv.rs | developer |
| `HIPFIRE_MOE_GATE_UP_FIXED_GROUPS` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_MOE_GATE_UP_FUSED` | crates/rdna-compute/src/gemv.rs, crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_MOE_GATE_UP_KERNEL` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_MOE_GATE_UP_LOW_VGPR` | crates/rdna-compute/src/gemv.rs, crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_MOE_GATE_UP_MIN_BLOCKS` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_MOE_GATE_UP_PAIR_VGPR` | crates/rdna-compute/src/gemv.rs, crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_MOE_GATE_UP_RANK_INTERLEAVE` | crates/rdna-compute/src/gemv.rs, crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_MOE_GATE_UP_TIGHT_GRID` | crates/rdna-compute/src/gemv.rs | developer |
| `HIPFIRE_MOE_GATE_UP_WG2` | crates/rdna-compute/src/gemv.rs | developer |
| `HIPFIRE_MOE_GATE_UP_WG_WAVES` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_MOE_GRADED` | crates/hipfire-quantize/src/pipeline.rs, crates/hipfire-quantize/src/quant_mq.rs | developer |
| `HIPFIRE_MOE_GROUPED_4W` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_MOE_GROUPED_GEMM` | crates/hipfire-arch-qwen35/src/qwen35/batch.rs, crates/hipfire-arch-qwen35/src/qwen35/prefill.rs | stable |
| `HIPFIRE_MOE_GROUPED_I8` | crates/hipfire-config/src/lib.rs, crates/hipfire-dispatch/src/families/moe.rs | experimental |
| `HIPFIRE_MOE_GROUPED_I8_K4` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_MOE_GROUPED_I8_K4_GFX12` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_MOE_GROUPED_I8_K8` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_MOE_GROUPED_M2` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/examples/test_moe_grouped_wmma_m2.rs | experimental |
| `HIPFIRE_MOE_HFQ6_I8` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/examples/test_moe_grouped_wmma_hfq6.rs | experimental |
| `HIPFIRE_MOE_HFQ6_V2` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/examples/test_moe_grouped_wmma_hfq6_v2.rs | experimental |
| `HIPFIRE_MOE_HOT_FRAC` | crates/hipfire-quantize/src/pipeline.rs | developer |
| `HIPFIRE_MOE_MQ6_ADMIT` | crates/hipfire-arch-qwen35/src/forward_slots.rs, crates/hipfire-arch-qwen35/src/qwen35/prefill.rs | developer |
| `HIPFIRE_MOE_NINEPATH` | crates/hipfire-dispatch/src/pipeline/moe_program.rs | developer |
| `HIPFIRE_MOE_PAIRED_WAVES` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_MOE_PARO_I8` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_MOE_PARO_I8_K8` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_MOE_PROJECTION` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_MOE_PROJECTION_KERNEL` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_MOE_ROUTER_SHARED_FUSE` | crates/hipfire-dispatch/src/pipeline/moe_program.rs | developer |
| `HIPFIRE_MOE_TIER_MAP` | crates/hipfire-quantize/src/pipeline.rs | developer |
| `HIPFIRE_MQ` | crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_MQ2_DOWN_ROWS` | crates/hipfire-arch-maple/src/forward.rs, crates/rdna-compute/src/gemv.rs | developer |
| `HIPFIRE_MQ3_DOWN_ROWS` | crates/rdna-compute/src/gemv.rs, crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_MQ3_MB4` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/examples/test_gemm_hfq3g256_wmma.rs | experimental |
| `HIPFIRE_MQ4C_RESIDUAL_KERNEL` | crates/rdna-compute/examples/bench_vgpr_cap_sweep.rs, crates/rdna-compute/examples/mq4c_parity.rs | harness |
| `HIPFIRE_MQ4C_VGPR_CAP` | crates/rdna-compute/examples/bench_vgpr_cap_sweep.rs | harness |
| `HIPFIRE_MQ4G256V2_K4096` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_MQ4G256V2_K512` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_MQ4G256V2_KERNEL` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_MQ4G256V2_LUT` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_MQ4G256V2_RESIDUAL_EPILOGUE` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_MQ4G256V2_RESIDUAL_SIGMOID_SCALED_EPILOGUE` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_MQ4G256V2_XBATCH_KERNEL` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_MQ4G256V2_XBATCH_MAX` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_MQ4V2_DOWN_TIGHT_GRID` | crates/rdna-compute/src/gemv.rs | developer |
| `HIPFIRE_MQ4V2_GATEUP_K5120` | benchmarks/scripts/mq4v2_k5120_abba.sh, crates/hipfire-config/src/lib.rs | stable |
| `HIPFIRE_GFX12_MQ4V2_GATEUP_K5120` | Opt-in exact gfx1201 MQ4V2 gate/up decode, K5120 and 17408 rows per output; typed `kernel.gfx12_mq4v2_gateup_k5120`, snapshotted at GPU initialization. Default off; preserves DEV/RT cache policy. | experimental |
| `HIPFIRE_MQ4V2_GATE_UP_KERNEL` | crates/rdna-compute/examples/mq4v2_moe_parity.rs, crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_MQ4V2_GATE_UP_NOLDS` | crates/rdna-compute/examples/mq4v2_moe_parity.rs, crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_MQ4V2_GATE_UP_TIGHT_GRID` | crates/rdna-compute/src/gemv.rs | developer |
| `HIPFIRE_MQ4V2_GFX1201_QKV_BT` | crates/rdna-compute/examples/test_mq4v2_qkv_bt_gfx1201.rs | harness |
| `HIPFIRE_MQ4V2_NINEPATH_KERNEL` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_MQ4V2_NINEPATH_RPB` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_MQ4V2_RESIDUAL_R1` | crates/hipfire-runtime/examples/mq4v2_parity.rs, crates/rdna-compute/src/gemv.rs | developer |
| `HIPFIRE_MQ4V2_RESIDUAL_R1_PARITY` | crates/hipfire-runtime/examples/mq4v2_parity.rs | harness |
| `HIPFIRE_MQ4V2_SHARED_DOWN_FUSED` | crates/hipfire-dispatch/src/pipeline/mod.rs | developer |
| `HIPFIRE_MQ6G256V2_RESIDUAL_EPILOGUE` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_MQ6G256V2_XBATCH` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_MQ6G256V2_XBATCH_KERNEL` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_MQ6G256V2_XBATCH_MAX` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_MQ6V2_DOWN_TIGHT_GRID` | crates/rdna-compute/src/gemv.rs | developer |
| `HIPFIRE_MQ6V2_GATE_UP_TIGHT_GRID` | crates/rdna-compute/src/gemv.rs | developer |
| `HIPFIRE_MQV2_GFX11_SCREEN` | crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_MQV2_GFX11_WMMA` | crates/hipfire-arch-qwen35/src/qwen35/prefill.rs, crates/hipfire-runtime/src/llama.rs | developer |
| `HIPFIRE_MQ_F16_PROJECTION_OFF` | crates/rdna-compute/src/feature_flags.rs, crates/rdna-compute/src/mq_f16_producers.rs | developer |
| `HIPFIRE_MQ_F16_RESIDUAL_OFF` | crates/rdna-compute/src/feature_flags.rs | developer |
| `HIPFIRE_MQ_PROLOGUE_FUSE_OFF` | crates/rdna-compute/src/feature_flags.rs | developer |
| `HIPFIRE_MTP_BYTE_IDENTITY_HEAD` | crates/hipfire-arch-qwen35/tests/mtp_cache_byte_identity.rs, crates/hipfire-arch-qwen35/tests/mtp_step_oracle.rs | harness |
| `HIPFIRE_MTP_BYTE_IDENTITY_MODEL` | crates/hipfire-arch-qwen35/tests/mtp_cache_byte_identity.rs, crates/hipfire-arch-qwen35/tests/mtp_step_oracle.rs | harness |
| `HIPFIRE_MTP_DEVICE_TOKEN_CHAIN` | crates/hipfire-arch-qwen35/src/mtp_spec.rs | developer |
| `HIPFIRE_MTP_DRAFT_HEAD` | crates/hipfire-arch-qwen4/src/mtp_gpu.rs, crates/hipfire-arch-qwen4/tests/greedy_mtp_identity_hw.rs | developer |
| `HIPFIRE_MTP_GPU_ACCEPT` | crates/hipfire-arch-qwen35/src/mtp_spec.rs | developer |
| `HIPFIRE_MTP_HEAD_LMHEAD_WMMA` | crates/hipfire-arch-qwen35/src/mtp_head.rs | developer |
| `HIPFIRE_MTP_IDENTITY_ARM` | crates/hipfire-arch-qwen4/tests/greedy_mtp_identity_hw.rs | harness |
| `HIPFIRE_MTP_IDENTITY_CASE` | crates/hipfire-arch-qwen4/tests/greedy_mtp_identity_hw.rs | harness |
| `HIPFIRE_MTP_IDENTITY_MODEL` | crates/hipfire-arch-qwen4/tests/greedy_mtp_identity_hw.rs, crates/hipfire-arch-qwen4/tests/sampled_mtp_distribution_hw.rs | harness |
| `HIPFIRE_MTP_IDENTITY_OUT` | crates/hipfire-arch-qwen4/tests/greedy_mtp_identity_hw.rs | harness |
| `HIPFIRE_MTP_INCREMENTAL` | crates/hipfire-arch-qwen4/src/mtp_spec.rs, crates/hipfire-arch-qwen4/tests/greedy_mtp_identity_hw.rs | developer |
| `HIPFIRE_MTP_K` | crates/hipfire-config/src/lib.rs, crates/hipfire-loader/src/carriers.rs | stable |
| `HIPFIRE_MTP_MODE` | crates/hipfire-config/src/lib.rs, crates/hipfire-runtime/src/config.rs | stable |
| `HIPFIRE_MTP_NGRAM` | crates/hipfire-config/src/lib.rs, crates/hipfire-generate/src/qwen.rs | stable |
| `HIPFIRE_MTP_NGRAM_K` | scripts/serve_harness.py | harness |
| `HIPFIRE_MTP_OWN_PREFILL` | crates/hipfire-arch-qwen35/src/mtp_spec.rs, crates/hipfire-arch-qwen35/src/mtp_speculator.rs | developer |
| `HIPFIRE_MTP_PAIRING` | crates/hipfire-arch-qwen4/src/mtp_spec.rs | developer |
| `HIPFIRE_MTP_PHASE_TIMING` | crates/hipfire-arch-qwen4/src/mtp_spec.rs | developer |
| `HIPFIRE_MTP_PROPOSAL_GRAPH` | crates/hipfire-arch-qwen35/src/mtp_spec.rs | developer |
| `HIPFIRE_MTP_P_MIN` | crates/hipfire-arch-qwen35/src/mtp_spec.rs | developer |
| `HIPFIRE_MTP_Q8_VERIFY_WMMA` | crates/hipfire-arch-qwen35/src/mtp_spec.rs | developer |
| `HIPFIRE_MTP_SAMPLED` | crates/hipfire-arch-qwen4/src/mtp_spec.rs, crates/hipfire-arch-qwen4/tests/sampled_mtp_distribution_hw.rs | stable |
| `HIPFIRE_MTP_SAMPLED_MODE` | crates/hipfire-arch-qwen4/src/mtp_spec.rs, crates/hipfire-arch-qwen4/tests/sampled_mtp_distribution_hw.rs | developer |
| `HIPFIRE_MTP_SMOKE_HEAD` | crates/hipfire-arch-qwen35/examples/mtp_head_smoke.rs | harness |
| `HIPFIRE_MTP_SMOKE_TRUNK` | crates/hipfire-arch-qwen35/examples/mtp_head_smoke.rs | harness |
| `HIPFIRE_MTP_SNAPSHOT_OVERLAP` | crates/hipfire-arch-qwen35/src/mtp_spec.rs | developer |
| `HIPFIRE_MTP_TAPE_REPLAY` | crates/hipfire-arch-qwen35/src/mtp_spec.rs | developer |
| `HIPFIRE_MTP_TRACE` | crates/hipfire-arch-qwen35/src/mtp_spec.rs, crates/hipfire-arch-qwen4/src/mtp_spec.rs | developer |
| `HIPFIRE_MTP_VERIFY_DECOUPLE` | crates/hipfire-arch-qwen35/src/qwen35/prefill.rs | developer |
| `HIPFIRE_MUSE_QKVG_ALLOW_BITDIFF` | crates/rdna-compute/examples/test_gemm_qkvza_hfq4g256.rs | harness |
| `HIPFIRE_MUSE_QKVG_TOL_ABS` | crates/rdna-compute/examples/test_gemm_qkvza_hfq4g256.rs | harness |
| `HIPFIRE_MUSE_QKVG_TOL_REL` | crates/rdna-compute/examples/test_gemm_qkvza_hfq4g256.rs | harness |
| `HIPFIRE_NGRAM_DRAFT` | crates/hipfire-arch-qwen2/src/spec_impl.rs, crates/hipfire-config/src/lib.rs | stable |
| `HIPFIRE_NGRAM_DRAFT_K` | crates/hipfire-config/src/lib.rs, crates/hipfire-runtime/src/config.rs | stable |
| `HIPFIRE_NGRAM_LOOP_THRESHOLD` | crates/hipfire-config/src/lib.rs, crates/hipfire-generate/src/ar.rs | stable |
| `HIPFIRE_NGRAM_MIN_COUNT` | crates/hipfire-config/src/lib.rs, crates/hipfire-runtime/src/config.rs | stable |
| `HIPFIRE_NGRAM_MOD_N_MATCH` | crates/hipfire-config/src/lib.rs, scripts/serve_harness.py | developer |
| `HIPFIRE_NGRAM_MOD_N_MAX` | crates/hipfire-config/src/lib.rs, scripts/serve_harness.py | developer |
| `HIPFIRE_NGRAM_MOD_N_MIN` | crates/hipfire-config/src/lib.rs, scripts/serve_harness.py | developer |
| `HIPFIRE_NGRAM_WINDOW` | crates/hipfire-config/src/lib.rs, crates/hipfire-generate/src/ar.rs | stable |
| `HIPFIRE_NORMALIZE_PROMPT` | crates/hipfire-config/src/lib.rs, crates/hipfire-runtime/examples/build_kld_ref_native_gemma4.rs | stable |
| `HIPFIRE_NO_DEVICE_COMPILER` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/compiler.rs | experimental |
| `HIPFIRE_NO_Q8_ROUTER` | crates/hipfire-quantize/src/pipeline.rs | developer |
| `HIPFIRE_NO_REGISTRY_FETCH` | crates/hipfire-config/src/lib.rs, crates/hipfire-registry/src/lib.rs | stable |
| `HIPFIRE_NO_SPILL` | crates/hipfire-quantize/src/pipeline.rs | developer |
| `HIPFIRE_NPU_SPILLOVER` | crates/hipfire-config/src/lib.rs, crates/hipfire-runtime/src/config.rs | experimental |
| `HIPFIRE_OFFLOAD_DEBUG` | crates/hipfire-arch-qwen35/src/qwen35/load.rs, crates/rdna-compute/src/dispatch.rs | developer |
| `HIPFIRE_OFFLOAD_EXEC` | crates/hipfire-config/src/lib.rs, crates/hipfire-dispatch/src/cpu_exec.rs | stable |
| `HIPFIRE_OOM_GUARD` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/kv_slots.rs | stable |
| `HIPFIRE_ORACLE_DIVERGE` | crates/hipfire-arch-qwen35/tests/route_oracle_mesh.rs | harness |
| `HIPFIRE_ORACLE_DUMP` | crates/saddle-lab/examples/tmp_gemm_v2_oracle.rs | harness |
| `HIPFIRE_ORACLE_KV_F32` | crates/hipfire-runtime/examples/gemma4_oracle.rs | harness |
| `HIPFIRE_ORACLE_MAX` | scripts/seed_oracle_collect.sh | harness |
| `HIPFIRE_ORACLE_STATE_FP32` | crates/hipfire-arch-qwen35/tests/route_oracle_mesh.rs | harness |
| `HIPFIRE_ORNITH15_MODEL` | scripts/coherence-gate-ornith15.sh | deprecated |
| `HIPFIRE_ORNITH_FIXTURE` | crates/hipfire-arch-qwen35/src/qwen35/load.rs, crates/hipfire-arch-qwen35/tests/route_oracle_mesh.rs | developer |
| `HIPFIRE_PAGE_EVICTION` | crates/hipfire-arch-qwen35/src/serve_engine.rs, crates/hipfire-loader/src/carriers.rs | developer |
| `HIPFIRE_PARENT_ROUTE_SCALE` | crates/hipfire-arch-deepseek4/scripts/ds4_parent_route_scale_probe.sh, crates/hipfire-ds4-parent/src/moe.rs | developer |
| `HIPFIRE_PARITY_EXTRA_MODEL` | crates/hipfire-arch-qwen35/tests/gpu_gemv_parity.rs | harness |
| `HIPFIRE_PARITY_EXTRA_QT` | crates/hipfire-arch-qwen35/tests/gpu_gemv_parity.rs | harness |
| `HIPFIRE_PARITY_MAX_TOK` | scripts/forward-lowered-parity.sh | harness |
| `HIPFIRE_PARITY_OUT` | scripts/forward-lowered-parity.sh | harness |
| `HIPFIRE_PARO_BATCHED` | crates/hipfire-arch-qwen35/src/qwen35/prefill.rs, crates/hipfire-dispatch/src/families/moe.rs | developer |
| `HIPFIRE_PATH_C_OUT` | scripts/path-c-smoke.sh | harness |
| `HIPFIRE_PFLASH_DEBUG` | crates/hipfire-generate/src/ar.rs | deprecated |
| `HIPFIRE_PFLASH_DRAFTER` | scripts/pflash-gate.sh | deprecated |
| `HIPFIRE_PFLASH_DRAFTER_KV` | crates/hipfire-config/src/lib.rs, crates/hipfire-pflash/src/pflash.rs | deprecated |
| `HIPFIRE_PFLASH_SCORE_LAYER` | crates/hipfire-config/src/lib.rs, crates/hipfire-pflash/src/pflash.rs | deprecated |
| `HIPFIRE_PFLASH_TARGET` | scripts/pflash-gate.sh | deprecated |
| `HIPFIRE_PING_TIMEOUT` | docs/investigations/evidence/ds4-mi300x-cdna-test-fail/raw/a1-m0/04-profile-feed.py, docs/investigations/evidence/ds4-mi300x-cdna-test-fail/raw/a1-m0/04-run-profile-direct.sh | harness |
| `HIPFIRE_PM4_KERNARG_POOL` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/replay.rs | experimental |
| `HIPFIRE_PORT` | crates/hipfire-config/src/lib.rs | stable |
| `HIPFIRE_POST_LATCH_ANSWER_TOKENS` | crates/hipfire-generate/src/ar.rs, crates/hipfire-generate/src/qwen.rs | developer |
| `HIPFIRE_PP_DEVICES` | crates/hipfire-generate/tests/qwen35_reset_hw.rs | harness |
| `HIPFIRE_PP_DFLASH` | crates/hipfire-daemon/src/main.rs | developer |
| `HIPFIRE_PP_GATE_HETEROGENEOUS` | scripts/pp-gate.sh | harness |
| `HIPFIRE_PP_GATE_INCLUDE_IGPU` | scripts/pp-gate.sh | harness |
| `HIPFIRE_PP_GATE_MODEL` | scripts/pp-gate.sh | harness |
| `HIPFIRE_PP_GATE_REQUIRE_SYSFS` | scripts/pp-gate.sh | harness |
| `HIPFIRE_PP_GATE_TEST_NO_SYSFS` | scripts/pp-gate.sh | harness |
| `HIPFIRE_PP_LAYERS` | crates/hipfire-loader/src/carriers.rs | developer |
| `HIPFIRE_PP_PARITY_MODEL` | crates/hipfire-arch-qwen35/tests/pp_parity.rs | harness |
| `HIPFIRE_PP_PFLASH` | crates/hipfire-daemon/src/main.rs | deprecated |
| `HIPFIRE_PP_RESET_MODEL` | crates/hipfire-generate/tests/qwen35_reset_hw.rs | harness |
| `HIPFIRE_PREFILL_BATCHED` | crates/hipfire-arch-qwen35/src/qwen35/prefill.rs, crates/hipfire-config/src/lib.rs | stable |
| `HIPFIRE_PREFILL_CHUNK` | crates/hipfire-arch-gemma4/examples/prefill_parity_gemma4.rs, crates/hipfire-runtime/examples/ep_decode_parity.rs | harness |
| `HIPFIRE_PREFILL_CHUNK_ROWS` | crates/hipfire-arch-qwen35/src/qwen35/prefill.rs, crates/hipfire-arch-qwen4/src/gpu_forward.rs | stable |
| `HIPFIRE_PREFILL_MAX_BATCH` | crates/hipfire-arch-qwen35/src/qwen35/ep_batch.rs, crates/hipfire-arch-qwen35/src/qwen35/prefill.rs | developer |
| `HIPFIRE_PREFILL_REUSE_PBS` | crates/hipfire-arch-qwen35/src/qwen35/forward.rs, crates/hipfire-arch-qwen35/src/speculative.rs | developer |
| `HIPFIRE_PROBE_ALIGN` | crates/hipfire-runtime/src/hfq.rs | developer |
| `HIPFIRE_PROBE_MODEL` | crates/hipfire-arch-qwen4/src/artifact.rs, crates/hipfire-runtime/src/hfq.rs | developer |
| `HIPFIRE_PROBE_READS` | crates/hipfire-runtime/src/hfq.rs | developer |
| `HIPFIRE_PROFILE` | crates/hipfire-arch-diffusion/examples/gpu_flux_forward.rs, crates/hipfire-arch-diffusion/src/flux_gpu.rs | developer |
| `HIPFIRE_PROFILE_CYCLES` | crates/hipfire-runtime/examples/dflash_spec_demo.rs, crates/saddle-lab/examples/mtp_only_demo.rs | harness |
| `HIPFIRE_PROFILE_DECODE` | crates/hipfire-runtime/examples/bench_qwen35_mq4.rs, scripts/kernel_atlas.py | harness |
| `HIPFIRE_PROFILE_MAX` | scripts/ddtree_verify_profile.sh | harness |
| `HIPFIRE_PROFILE_RUNS` | scripts/ddtree_verify_profile.sh | harness |
| `HIPFIRE_PROFILE_SELF` | docs/investigations/evidence/ds4-mi300x-cdna-test-fail/raw/a1-m0/04-profile-feed.py | harness |
| `HIPFIRE_PROMPT` | docs/investigations/evidence/ds4-mi300x-cdna-test-fail/raw/a1-m0/04-profile-feed.py, docs/investigations/evidence/ds4-mi300x-cdna-test-fail/raw/a1-m0/04-run-profile-direct.sh | harness |
| `HIPFIRE_PROMPT_CACHE_CAP` | crates/hipfire-config/src/lib.rs, crates/hipfire-loader/src/lib.rs | stable |
| `HIPFIRE_PROMPT_CACHE_UNBOUNDED` | crates/hipfire-config/src/lib.rs, crates/hipfire-loader/src/lib.rs | experimental |
| `HIPFIRE_PROMPT_HEAT_JSON` | crates/hipfire-config/src/lib.rs, crates/hipfire-runtime/src/config.rs | experimental |
| `HIPFIRE_PROMPT_HEAT_LIMIT` | crates/hipfire-config/src/lib.rs, crates/hipfire-runtime/src/config.rs | experimental |
| `HIPFIRE_PROMPT_TOKEN_HEAT` | crates/hipfire-config/src/lib.rs, crates/hipfire-runtime/src/config.rs | experimental |
| `HIPFIRE_Q8_BATCHED_LEGACY` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_Q8_CLASSES` | crates/hipfire-quantize/src/diagnostics.rs, crates/hipfire-quantize/src/model_filter.rs | developer |
| `HIPFIRE_Q8_FLASH_TILE` | crates/hipfire-arch-maple/src/maple.rs, crates/rdna-compute/src/attention.rs | developer |
| `HIPFIRE_Q8_PREFILL_WMMA` | crates/hipfire-arch-qwen35/src/qwen35/prefill.rs | developer |
| `HIPFIRE_QA_KV_MODES` | crates/saddle-lab/examples/test_inferenceQA.rs | harness |
| `HIPFIRE_QKVZA_BLOCK_SIZE` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_QKVZA_CPOL` | crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_QKVZA_KERNEL_NAME` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_QKVZA_MIN_BLOCKS_PER_CU` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_QKVZA_SCALAR_PREP` | crates/hipfire-arch-qwen35/src/qwen35/forward.rs, crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_QKVZA_SPLIT_TAIL` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_QKVZA_WAVES_PER_BLOCK` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_QKV_KERNEL_NAME` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_QKV_WITH_BIAS` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_QUANTIZE` | scripts/stage_models.sh | harness |
| `HIPFIRE_QUANT_DIAG_PATH` | crates/hipfire-config/src/lib.rs, crates/hipfire-quantize/src/diagnostics.rs | stable |
| `HIPFIRE_QUANT_THREADS` | crates/hipfire-quantize/src/cli.rs, crates/hipfire-quantize/tests/cli_contract.rs | developer |
| `HIPFIRE_QWEN2_VERIFY_SEQ` | crates/hipfire-arch-dots-ocr/src/spec_impl.rs, crates/hipfire-arch-qwen2/src/spec_impl.rs | developer |
| `HIPFIRE_QWEN35_A3B_RESET_MODEL` | crates/hipfire-generate/tests/qwen35_reset_hw.rs | harness |
| `HIPFIRE_QWEN35_DSPARK_CONF_THRESHOLD` | crates/hipfire-loader/src/lib.rs | developer |
| `HIPFIRE_QWEN35_FA_EPILOGUE_FUSE` | crates/hipfire-arch-qwen35/src/qwen35/forward.rs | developer |
| `HIPFIRE_QWEN35_FA_PREP_FUSE` | crates/hipfire-arch-qwen35/src/qwen35/forward.rs | developer |
| `HIPFIRE_QWEN35_FA_PREP_KERNEL` | crates/rdna-compute/src/kernel_registry.rs, crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_QWEN35_FINITE_TRACE` | crates/hipfire-arch-qwen35/src/qwen35/prefill.rs | developer |
| `HIPFIRE_QWEN35_FIXTURE` | crates/hipfire-arch-qwen35/src/qwen35/load.rs, crates/hipfire-arch-qwen35/tests/route_oracle_mesh.rs | developer |
| `HIPFIRE_QWEN35_GRAMMAR` | crates/hipfire-daemon/src/slots.rs, crates/hipfire-generate/src/ar.rs | developer |
| `HIPFIRE_QWEN35_NGRAM_LEN_MIN` | crates/hipfire-arch-qwen35/src/grammar_config.rs | developer |
| `HIPFIRE_QWEN35_NGRAM_MIN_REPEATS` | crates/hipfire-arch-qwen35/src/grammar_config.rs | developer |
| `HIPFIRE_QWEN35_RESET_MODEL` | crates/hipfire-generate/tests/qwen35_reset_hw.rs | harness |
| `HIPFIRE_QWEN3_BENCH_MODE` | crates/hipfire-arch-llama/examples/qwen3_dspark_bench.rs | harness |
| `HIPFIRE_QWEN3_DSPARK_CONF_THRESHOLD` | crates/hipfire-arch-llama/examples/qwen3_dspark_bench.rs, crates/hipfire-loader/src/carriers.rs | developer |
| `HIPFIRE_QWEN3_MAX` | crates/hipfire-arch-llama/examples/qwen3_dspark_bench.rs | harness |
| `HIPFIRE_QWEN3_MODEL` | crates/hipfire-arch-llama/examples/qwen3_dspark_bench.rs | harness |
| `HIPFIRE_QWEN3_NGRAM_K` | crates/hipfire-arch-llama/examples/qwen3_dspark_bench.rs | harness |
| `HIPFIRE_QWEN3_PROMPT` | crates/hipfire-arch-llama/examples/qwen3_dspark_bench.rs | harness |
| `HIPFIRE_QWEN3_RAW` | crates/hipfire-arch-llama/examples/qwen3_dspark_bench.rs | harness |
| `HIPFIRE_QWEN3_TEMP` | crates/hipfire-arch-llama/examples/qwen3_dspark_bench.rs | harness |
| `HIPFIRE_QWEN3_TOP_K` | crates/hipfire-arch-llama/examples/qwen3_dspark_bench.rs | harness |
| `HIPFIRE_QWEN3_TOP_P` | crates/hipfire-arch-llama/examples/qwen3_dspark_bench.rs | harness |
| `HIPFIRE_QWEN3_WARMUP` | crates/hipfire-arch-llama/examples/qwen3_dspark_bench.rs | harness |
| `HIPFIRE_QWEN4_CAPACITY_BYTES` | crates/hipfire-quantize/src/qwen4.rs | developer |
| `HIPFIRE_QWEN4_EXPERT_STAGE` | crates/hipfire-arch-qwen4/src/gpu_forward.rs, crates/railgun-cert/src/recording.rs | developer |
| `HIPFIRE_QWEN4_EXPERT_STAGE_MIN_ROWS` | crates/hipfire-arch-qwen4/src/gpu_forward.rs, crates/railgun-cert/src/recording.rs | developer |
| `HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS` | crates/hip-bridge/src/ffi.rs, crates/hipfire-loader/src/admission.rs | developer |
| `HIPFIRE_QWEN4_F16_WMMA` | crates/hipfire-arch-qwen4/examples/qwen4_qsa_ctx.rs, crates/rdna-compute/examples/bench_qsa_indexed.rs | developer |
| `HIPFIRE_QWEN4_F16_WMMA_GFX1201` | crates/rdna-compute/examples/bench_qwen4_hc_wmma.rs, crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_QWEN4_GDN_CONV_QKNORM` | crates/hipfire-dispatch/src/pipeline/layer_ops.rs, crates/rdna-compute/src/feature_flags.rs | developer |
| `HIPFIRE_QWEN4_GDN_PIPE` | crates/rdna-compute/src/tensor_ops.rs | developer |
| `HIPFIRE_QWEN4_GDN_Q8_INLINE` | crates/rdna-compute/src/feature_flags.rs, crates/rdna-compute/src/kernel_registry.rs | developer |
| `HIPFIRE_QWEN4_HC_DOWN_TILE` | crates/railgun-cert/src/recording.rs, crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_QWEN4_HC_FUSE` | crates/hipfire-dispatch/src/pipeline/layer_ops.rs, crates/hipfire-dispatch/src/pipeline/moe_program.rs | developer |
| `HIPFIRE_QWEN4_HC_ROW_FOLD` | crates/hipfire-dispatch/src/pipeline/layer_ops.rs, crates/hipfire-dispatch/src/pipeline/moe_program.rs | developer |
| `HIPFIRE_QWEN4_HC_UP_TILE` | crates/rdna-compute/src/feature_flags.rs, crates/rdna-compute/src/tensor_ops.rs | developer |
| `HIPFIRE_QWEN4_MOE_COMBINE_ZINIT` | crates/hipfire-dispatch/src/pipeline/moe_program.rs, crates/hipfire-dispatch/src/pipeline/qt44_qt53_prefill.rs | developer |
| `HIPFIRE_QWEN4_MOE_SYM_DOWN_RR` | crates/rdna-compute/examples/qwen4_moe_sym.rs, crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_QWEN4_MOE_SYM_IU4` | crates/hipfire-arch-qwen4/src/gpu_forward.rs, crates/hipfire-dispatch/src/pipeline/qt44_qt53_prefill.rs | developer |
| `HIPFIRE_QWEN4_MOE_SYM_PM` | crates/rdna-compute/examples/qwen4_moe_sym.rs, crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_QWEN4_MQ6_X4_GFX1201` | crates/rdna-compute/src/feature_flags.rs, crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_QWEN4_MQ6_X4_PM` | crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_QWEN4_MQ6_X4_REGIONS` | crates/rdna-compute/src/feature_flags.rs, crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_QWEN4_MQ6_X4_TILE` | crates/rdna-compute/src/feature_flags.rs, crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_QWEN4_MTP_BATCHED_FILL` | crates/hipfire-arch-qwen4/examples/qwen4_mtp_fill.rs, crates/hipfire-arch-qwen4/src/mtp_spec.rs | developer |
| `HIPFIRE_QWEN4_MTP_TIER` | crates/hipfire-arch-qwen4/src/weights.rs | developer |
| `HIPFIRE_QWEN4_ORACLE_CACHE` | crates/hipfire-arch-qwen4/reference_oracle/upstream.py | harness |
| `HIPFIRE_QWEN4_PLE_FUSE` | crates/hipfire-arch-qwen4/src/gpu_forward.rs, crates/hipfire-dispatch/src/pipeline/layer_ops.rs | developer |
| `HIPFIRE_QWEN4_PLE_WIDE_READERS` | crates/hipfire-arch-qwen4/src/bundle.rs | developer |
| `HIPFIRE_QWEN4_PROFILE_CHECKPOINT` | crates/hipfire-arch-qwen4/src/state_parity.rs | developer |
| `HIPFIRE_QWEN4_PROFILE_SOURCE_CALLBACK` | crates/hipfire-arch-qwen4/src/state_parity.rs | developer |
| `HIPFIRE_QWEN4_PROJ_REGIONS` | crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_QWEN4_QSA_PM` | crates/hipfire-isa/src/kernels/qsa_gather.rip.rs, crates/rdna-compute/examples/qsa_pm_check.rs | developer |
| `HIPFIRE_QWEN4_QSA_SCORE_PM` | crates/rdna-compute/src/tensor_ops.rs | developer |
| `HIPFIRE_QWEN4_QSA_SELECT_EXACT` | crates/railgun-cert/src/recording.rs, crates/rdna-compute/src/tensor_ops.rs | developer |
| `HIPFIRE_QWEN4_QSA_SELECT_PM` | crates/rdna-compute/src/tensor_ops.rs | developer |
| `HIPFIRE_QWEN4_QSA_WMMA_GATHER` | crates/hipfire-arch-qwen4/examples/qwen4_qsa_ctx.rs, crates/hipfire-arch-qwen4/src/bundle.rs | developer |
| `HIPFIRE_QWEN4_REQUANT` | crates/hipfire-arch-qwen4/src/weights.rs | developer |
| `HIPFIRE_QWEN4_ROUTER_FAST` | crates/railgun-cert/src/recording.rs, crates/rdna-compute/src/feature_flags.rs | developer |
| `HIPFIRE_QWEN4_ROUTE_TRACE` | crates/hipfire-arch-qwen4/src/gpu_forward.rs | developer |
| `HIPFIRE_QWEN4_SHARED_DOWN_EPI` | crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_QWEN4_TRUNK_IU4` | crates/hipfire-arch-qwen4/src/gpu_forward.rs, crates/hipfire-arch-qwen4/src/weights.rs | developer |
| `HIPFIRE_QWEN4_TRUNK_TIER` | crates/hipfire-arch-qwen4/src/weights.rs | developer |
| `HIPFIRE_QWEN_CACHE_TRACE` | crates/hipfire-daemon/src/main.rs, crates/hipfire-generate/src/ar.rs | developer |
| `HIPFIRE_QWEN_KV_DEFAULT_Q8` | crates/hipfire-loader/src/admission.rs, crates/hipfire-runtime/src/loader_api.rs | developer |
| `HIPFIRE_QWEN_MOE_FINAL_NORM_RAW` | scripts/test_pr228_spiral_check.sh | harness |
| `HIPFIRE_QWEN_MTP` | scripts/benchlocal_campaign.py, scripts/serve_harness.py | harness |
| `HIPFIRE_QWEN_PROMPT_CACHE` | crates/hipfire-arch-qwen4/src/bundle.rs, crates/hipfire-generate/src/ar.rs | developer |
| `HIPFIRE_RAILGUN_BACKEND` | crates/rdna-compute/src/replay.rs, crates/rdna-compute/src/replay/railgun_shadow.rs | developer |
| `HIPFIRE_RAILGUN_CACHE_TABLE` | crates/rdna-compute/src/replay/railgun_shadow.rs | developer |
| `HIPFIRE_RAILGUN_CHECK` | crates/hip-bridge/src/registry.rs, crates/hipfire-arch-qwen4/src/gpu_forward.rs | developer |
| `HIPFIRE_RAILGUN_CHECK_OUT` | crates/rdna-compute/src/railgun_check.rs | developer |
| `HIPFIRE_RAILGUN_DIGEST` | crates/hip-bridge/src/registry.rs, crates/rdna-compute/src/railgun_check.rs | developer |
| `HIPFIRE_RAILGUN_INVENTORY` | crates/rdna-compute/src/replay/railgun_shadow.rs | developer |
| `HIPFIRE_RAILGUN_NEGATIVE_CONTROL` | crates/rdna-compute/src/railgun_check.rs, crates/rdna-compute/src/replay/railgun_shadow.rs | developer |
| `HIPFIRE_RAILGUN_SHADOW` | crates/rdna-compute/src/railgun_check.rs, crates/rdna-compute/src/replay.rs | developer |
| `HIPFIRE_RAILGUN_SHADOW_OUT` | crates/rdna-compute/src/replay/railgun_shadow.rs | developer |
| `HIPFIRE_RCCL_LIB` | crates/hip-bridge/src/rccl.rs | developer |
| `HIPFIRE_RDNA2_VARIANT` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_RDNA3` | crates/hipfire-runtime/examples/tmp_halo_iu4_calibrate.rs | harness |
| `HIPFIRE_RDNA3_HFQ4_LM_HEAD_K2048` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_RDNA3_HFQ4_MOE_GATE_UP_K2048` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_RDNA3_HFQ4_QKVZA_2WAVE` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_RDNA3_HFQ4_QKVZA_HOIST_X32` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_RDNA3_HFQ4_QKVZA_K2048` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_RDNA3_HFQ4_QKVZA_LDSX8` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_RDNA3_HFQ4_QKVZA_REDUCE_CHAIN` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_RDNA3_HFQ4_QKVZA_WAVEPACK4` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_RDNA3_HFQ4_QKV_WAVE64` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_RDNA3_HFQ4_RESIDUAL_K2048` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_RDNA3_HFQ4_RESIDUAL_STAGE_X32` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_RDNA3_HFQ4_SIGMOID_BUFFER` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_RDNA3_HFQ4_SIGMOID_HOIST_X16` | crates/rdna-compute/src/gemv.rs | developer |
| `HIPFIRE_RDNA3_HFQ4_SIGMOID_K512` | crates/rdna-compute/src/gemv.rs | developer |
| `HIPFIRE_RDNA3_HFQ4_SIGMOID_ROWS4` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_RDNA3_HFQ4_SIGMOID_TIGHT_GRID` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_RDNA3_MOE_GATE_UP_K2048` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_RDNA3_MOE_GATE_UP_ROUTE_BUFFER` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_RDNA3_MOE_GATE_UP_X_BUFFER` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_RDNA3_QKVZA_HOIST_X32` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_RDNA3_QKVZA_K2048` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_RDNA3_QKVZA_LDSX8` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_RDNA3_QKVZA_PAIR_BUFFER` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_RDNA3_QKVZA_R2` | crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_RDNA3_QKVZA_REDUCE_CHAIN` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_RDNA3_QKVZA_X_BUFFER` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_RDNA3_QKV_K2048` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_RDNA3_QKV_X_BUFFER` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_RDNA3_RESIDUAL_BUFFER_CONSUME` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_RDNA3_RESIDUAL_K2048` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_RDNA3_RESIDUAL_MATRIX_RSRC` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_RDNA3_RESIDUAL_STAGE_X32` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_RDNA3_RMSNORM_SIGN_CONST` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_RDNA3_RMSNORM_SIGN_LDS` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_RDNA3_RMSNORM_SPLIT` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_RDNA3_RMSNORM_VECSUM` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_RDNA3_RMSNORM_VECSUM_KERNEL` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_RDNA3_RMSNORM_WAVEGRID` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_RDNA3_SIGMOID_HOIST_X16` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_RDNA3_SIGMOID_K512` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_RDNA3_SIGMOID_M2048` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_RDNA3_SIGMOID_ROWS4` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_REAP_PLAN` | crates/hipfire-arch-deepseek4/src/deepseek4.rs, crates/hipfire-arch-lfm2moe/src/config.rs | developer |
| `HIPFIRE_REDLINE_DISPATCH_PROFILE` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/replay.rs | experimental |
| `HIPFIRE_REDLINE_GAP_TIMING` | crates/rdna-compute/src/gap_timing.rs | developer |
| `HIPFIRE_REDLINE_IB_POOL` | crates/rdna-compute/src/replay.rs | developer |
| `HIPFIRE_REDLINE_PM4_PROGRESS` | crates/redline-dispatch/src/aql/replay.rs | developer |
| `HIPFIRE_REDLINE_POOL_DEBUG` | crates/hipfire-config/src/lib.rs, crates/redline-rocr/src/runtime.rs | experimental |
| `HIPFIRE_REDLINE_PROBE_VMEMAP` | crates/redline-rocr/src/runtime.rs | developer |
| `HIPFIRE_REDLINE_QUEUE_TIMEOUT` | crates/redline-rocr/src/runtime.rs | developer |
| `HIPFIRE_REGISTRY_URL` | crates/hipfire-cli/src/main.rs, crates/hipfire-config/src/lib.rs | stable |
| `HIPFIRE_REMOTE` | scripts/mi300x_bootstrap.sh | harness |
| `HIPFIRE_REPLAY_BACKEND` | crates/hipfire-config/src/lib.rs, crates/hipfire-daemon/src/main.rs | stable |
| `HIPFIRE_REPLAY_BINDINGS_VERIFY` | crates/rdna-compute/src/replay.rs | developer |
| `HIPFIRE_REPLAY_DIAGNOSTIC_SPECIALIZED_MOE_CAPTURE` | crates/hipfire-dispatch/src/pipeline/sealed_moe.rs | developer |
| `HIPFIRE_REPLAY_GRAPH` | crates/hipfire-arch-qwen35/src/speculative.rs | developer |
| `HIPFIRE_REPLAY_MANUAL_CAPTURE` | crates/hipfire-config/src/lib.rs, crates/hipfire-dispatch/src/pipeline/sealed_moe.rs | experimental |
| `HIPFIRE_REPLAY_PM4_ACQUIRE_POLICY` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/replay.rs | experimental |
| `HIPFIRE_REPLAY_PM4_DS4_FFN_BRANCH_CHAINS` | crates/rdna-compute/src/replay.rs | developer |
| `HIPFIRE_REPLAY_PM4_DYNAMIC_GRID` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/dispatch.rs | experimental |
| `HIPFIRE_REPLAY_PM4_GCR_TRIM` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/replay.rs | experimental |
| `HIPFIRE_REPLAY_PM4_GFX1010_DEPENDENCY` | crates/rdna-compute/src/replay.rs | developer |
| `HIPFIRE_REPLAY_PM4_GFX11_VMEM_ACQUIRE` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/replay.rs | experimental |
| `HIPFIRE_REPLAY_PM4_GFX12_VMEM_ACQUIRE` | crates/rdna-compute/src/replay.rs | developer |
| `HIPFIRE_REPLAY_PM4_MAX_PARALLEL_PHASES` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/replay.rs | experimental |
| `HIPFIRE_REPLAY_PM4_MIN_PARALLEL_WIDTH` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/replay.rs | experimental |
| `HIPFIRE_REPLAY_PM4_MIN_PARALLEL_WORKGROUPS` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/replay.rs | experimental |
| `HIPFIRE_REPLAY_PM4_NAME` | scripts/lmx_redline_campaign.py, tools/redline/product_bench.py | harness |
| `HIPFIRE_REPLAY_PM4_NATIVE_PHASES` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/replay.rs | experimental |
| `HIPFIRE_REPLAY_PM4_QUEUES` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/replay.rs | experimental |
| `HIPFIRE_REPLAY_PM4_SINGLE_IB_REORDER` | crates/rdna-compute/src/replay.rs, tools/redline/tests/test_product_bench.py | developer |
| `HIPFIRE_REPLAY_PM4_STATEFUL` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/replay.rs | experimental |
| `HIPFIRE_REPLAY_PM4_STREAM_ACCOUNTING` | crates/rdna-compute/src/replay.rs | developer |
| `HIPFIRE_REPLAY_PM4_WAIT_POLICY` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/replay.rs | experimental |
| `HIPFIRE_REPLAY_ROUTE_PROOF` | crates/rdna-compute/src/replay.rs, tools/redline/product_bench.py | developer |
| `HIPFIRE_REPLAY_ROUTE_PROOF_LOG` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/replay.rs | experimental |
| `HIPFIRE_REPLAY_TRANSPORT` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/replay.rs | experimental |
| `HIPFIRE_REPO` | scripts/quantize-dspark.sh | harness |
| `HIPFIRE_REPO_ROOT` | benchmarks/quality-baselines/harness/spe_ablation.sh | harness |
| `HIPFIRE_REQUIRE_REAP_OVERLAY` | scripts/reap/run_deepseek4_e8_layer_screen.sh, scripts/reap/run_deepseek4_e8_phase_quality.sh | harness |
| `HIPFIRE_RESET_ROUNDS` | crates/hipfire-generate/tests/qwen35_reset_hw.rs | harness |
| `HIPFIRE_RESIDUAL_CPOL` | crates/rdna-compute/src/gemv.rs | developer |
| `HIPFIRE_RESIDUAL_KERNEL` | crates/rdna-compute/examples/bench_vgpr_cap_sweep.rs, crates/rdna-compute/examples/mq4c_parity.rs | developer |
| `HIPFIRE_RESIDUAL_KSPLIT_OFF` | crates/rdna-compute/examples/test_mq4v2_residual_ksplit_gfx1100.rs, crates/rdna-compute/src/dflash_draft_fusion.rs | developer |
| `HIPFIRE_RESIDUAL_LDSSTAGE` | crates/rdna-compute/src/dflash_draft_fusion.rs, crates/rdna-compute/src/feature_flags.rs | developer |
| `HIPFIRE_RESIDUAL_MULTIROW_R2_KERNEL` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_RESIDUAL_MULTIROW_R4_KERNEL` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_RESIDUAL_MULTIROW_R8_KERNEL` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_RESULT_JSON` | docs/investigations/evidence/ds4-mi300x-cdna-test-fail/raw/a1-m0/04-profile-feed.py, docs/investigations/evidence/ds4-mi300x-cdna-test-fail/raw/a1-m0/04-run-profile-direct.sh | harness |
| `HIPFIRE_RMSNORM_AWQ` | crates/rdna-compute/src/gemv.rs, crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_RMSNORM_AWQ_RCP` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_RMSNORM_FOLD` | crates/rdna-compute/src/kernel_registry.rs, crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_RMSNORM_FP8_STRIDED` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_RMSNORM_GROUP_GRID` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_RMSNORM_KERNEL` | crates/rdna-compute/src/gemv.rs, crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_RMSNORM_MQ_TIGHT_LDS` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_RMSNORM_P1A_BATCHED` | crates/rdna-compute/src/gemv.rs, crates/rdna-compute/src/kernel_registry.rs | developer |
| `HIPFIRE_RMSNORM_P1A_DRAIN` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_RMSNORM_P1A_MAX8` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_ROCBLAS_ALL_ARCHS` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/dispatch.rs | experimental |
| `HIPFIRE_ROCBLAS_MIN_BATCH` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/dispatch.rs | experimental |
| `HIPFIRE_ROCBLAS_OFF` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/examples/test_mfp4e8_soa_rocblas_gfx942.rs | stable |
| `HIPFIRE_ROCM_PATH` | crates/hipfire-cli/src/setup.rs, crates/hipfire-config/src/rocm.rs | developer |
| `HIPFIRE_ROCM_ROOT` | crates/hipfire-cli/src/setup.rs, scripts/build-kernel-pack.sh | developer |
| `HIPFIRE_ROCM_STRICT` | crates/hipfire-cli/src/main.rs, crates/hipfire-cli/src/setup.rs | developer |
| `HIPFIRE_ROCPROF_BIN` | scripts/rocprof-daemon-wrap.sh | harness |
| `HIPFIRE_ROCPROF_CSV` | crates/hipfire-runtime/examples/bench_qwen35_mq4.rs, scripts/coverage-audit.py | harness |
| `HIPFIRE_ROCPROF_DAEMON_TARGET` | scripts/rocprof-daemon-wrap.sh | harness |
| `HIPFIRE_ROCPROF_OUTPUT_DIR` | scripts/rocprof-daemon-wrap.sh | harness |
| `HIPFIRE_ROOT` | crates/hipfire-arch-deepseek4/scripts/ds4_gate6_teacher_kld.sh, crates/hipfire-arch-deepseek4/scripts/ds4_mq2r_cap_campaign.sh | harness |
| `HIPFIRE_ROPE_HALFSPLIT` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_ROPE_INTERLEAVED_LEGACY` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_ROTATE_AWQ` | crates/rdna-compute/src/gemv.rs, crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_ROTATE_FP8_AWQ` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_ROTATE_FP8_GATE_IL` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_ROTATE_FP8_KERNEL` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_ROTATE_FP8_SIGMOID_GATE` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_ROTATE_GATE_IL` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_ROTATE_KERNEL` | crates/rdna-compute/src/gemv.rs, crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_ROTATE_SIGMOID_GATE` | crates/rdna-compute/src/gemv.rs, crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_ROUTED_GL` | crates/hipfire-quantize/src/pipeline.rs | developer |
| `HIPFIRE_ROUTER_EXACT_KERNEL` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_ROUTER_SHARED_SILU_MQ_ROTATE` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_ROUTE_ORACLE_KV_B` | crates/hipfire-arch-qwen35/tests/route_oracle_single.rs | harness |
| `HIPFIRE_S4_FLAG_PROBE_UNSET_OFF` | crates/hipfire-config/src/lib.rs | developer |
| `HIPFIRE_S4_FLAG_PROBE_UNSET_ON` | crates/hipfire-config/src/lib.rs | developer |
| `HIPFIRE_SAMPLED_MTP_ARM` | crates/hipfire-arch-qwen4/tests/sampled_mtp_distribution_hw.rs, crates/hipfire-arch-qwen4/tests/sampled_mtp_distribution_hw.rs | harness |
| `HIPFIRE_SAMPLED_MTP_MIN_P` | crates/hipfire-arch-qwen4/tests/sampled_mtp_distribution_hw.rs, crates/hipfire-arch-qwen4/tests/sampled_mtp_distribution_hw.rs | harness |
| `HIPFIRE_SAMPLED_MTP_OUT` | crates/hipfire-arch-qwen4/tests/sampled_mtp_distribution_hw.rs, crates/hipfire-arch-qwen4/tests/sampled_mtp_distribution_hw.rs | harness |
| `HIPFIRE_SAMPLED_MTP_TEMP` | crates/hipfire-arch-qwen4/tests/sampled_mtp_distribution_hw.rs, crates/hipfire-arch-qwen4/tests/sampled_mtp_distribution_hw.rs | harness |
| `HIPFIRE_SAMPLED_MTP_TOP_K` | crates/hipfire-arch-qwen4/tests/sampled_mtp_distribution_hw.rs, crates/hipfire-arch-qwen4/tests/sampled_mtp_distribution_hw.rs | harness |
| `HIPFIRE_SAMPLED_MTP_TOP_P` | crates/hipfire-arch-qwen4/tests/sampled_mtp_distribution_hw.rs, crates/hipfire-arch-qwen4/tests/sampled_mtp_distribution_hw.rs | harness |
| `HIPFIRE_SAMPLED_MTP_TRIALS` | crates/hipfire-arch-qwen4/tests/sampled_mtp_distribution_hw.rs, crates/hipfire-arch-qwen4/tests/sampled_mtp_distribution_hw.rs | harness |
| `HIPFIRE_SAMPLE_COMPARE` | crates/hipfire-runtime/src/llama.rs, crates/saddle-lab/examples/infer_qwen35.rs | developer |
| `HIPFIRE_SAMPLE_FAST` | crates/rdna-compute/examples/sample_parallel_stable_parity.rs, crates/rdna-compute/src/sampling.rs | developer |
| `HIPFIRE_SAMPLE_PARALLEL` | crates/rdna-compute/examples/sample_accept_parity.rs, crates/rdna-compute/src/sampling.rs | developer |
| `HIPFIRE_SCHED_PROFILE` | crates/rdna-compute/src/compiler.rs | developer |
| `HIPFIRE_SELECT_REGRID_OFF` | crates/rdna-compute/src/feature_flags.rs, crates/rdna-compute/src/select_regrid.rs | developer |
| `HIPFIRE_SERVE_ALLOW_INCOHERENT` | scripts/serve_harness.py | harness |
| `HIPFIRE_SERVE_ALLOW_REQUEST_PATHS` | crates/hipfire-config/src/lib.rs | stable |
| `HIPFIRE_SERVE_ALLOW_REQUEST_PULL` | crates/hipfire-config/src/lib.rs | stable |
| `HIPFIRE_SERVE_GATE_DFLASH` | scripts/serve-multiturn-gate.sh | harness |
| `HIPFIRE_SERVE_GATE_OUT` | scripts/serve-multiturn-gate.sh | harness |
| `HIPFIRE_SERVE_GATE_PORT` | scripts/serve-loop-gate.sh | harness |
| `HIPFIRE_SERVE_HARNESS_GRACEFUL_CLEANUP` | scripts/serve_harness.py | harness |
| `HIPFIRE_SERVE_HARNESS_PID_FILE` | scripts/lmx_continuous_batch.py, scripts/serve_harness.py | harness |
| `HIPFIRE_SERVE_HARNESS_SELFTEST` | scripts/serve_harness.py | harness |
| `HIPFIRE_SERVE_LOG` | scripts/test-ds4-heterogeneous-abort-resume.sh, scripts/test-qwen35-abort-resume.sh | harness |
| `HIPFIRE_SERVE_MAX_BATCH_TOKENS` | crates/hipfire-config/src/lib.rs, crates/hipfire-daemon/src/slots.rs | experimental |
| `HIPFIRE_SERVE_MAX_QUEUE` | crates/hipfire-config/src/lib.rs, crates/hipfire-daemon/src/slots.rs | stable |
| `HIPFIRE_SERVE_MAX_QUEUE_BYTES` | crates/hipfire-config/src/lib.rs, crates/hipfire-daemon/src/slots.rs | experimental |
| `HIPFIRE_SERVE_MULTI_SLOT` | crates/hipfire-config/src/lib.rs, scripts/serve_concurrency_gate.sh | stable |
| `HIPFIRE_SERVE_MULTI_SLOT_CTX` | crates/hipfire-config/src/lib.rs, scripts/scs_suite.py | stable |
| `HIPFIRE_SERVE_MULTI_SLOT_PREFILL_CHUNK` | crates/hipfire-config/src/lib.rs | stable |
| `HIPFIRE_SERVE_MULTI_SLOT_SLOTS` | crates/hipfire-config/src/lib.rs | stable |
| `HIPFIRE_SERVE_PREFILL_MIN_TOKENS` | crates/hipfire-config/src/lib.rs, crates/hipfire-daemon/src/slots.rs | experimental |
| `HIPFIRE_SERVE_PREFIX_CACHE` | crates/hipfire-config/src/lib.rs, crates/hipfire-daemon/src/slots.rs | experimental |
| `HIPFIRE_SERVE_PREFIX_CACHE_MAX_BYTES` | crates/hipfire-config/src/lib.rs | experimental |
| `HIPFIRE_SERVE_QUEUE_TIMEOUT_MS` | crates/hipfire-config/src/lib.rs, crates/hipfire-daemon/src/slots.rs | stable |
| `HIPFIRE_SERVE_RETRY_BACKOFF_MS` | crates/hipfire-config/src/lib.rs | experimental |
| `HIPFIRE_SERVE_RETRY_ENABLED` | crates/hipfire-config/src/lib.rs | experimental |
| `HIPFIRE_SERVE_SPEC_OFF` | scripts/scs_suite.py | harness |
| `HIPFIRE_SERVE_STREAM_STALL_TIMEOUT_MS` | crates/hipfire-config/src/lib.rs | experimental |
| `HIPFIRE_SERVE_STRUCTURED_JUMP_FORWARD` | crates/hipfire-config/src/lib.rs, crates/hipfire-daemon/src/slots.rs | experimental |
| `HIPFIRE_SERVE_UI` | crates/hipfire-cli/src/serve/mod.rs, crates/hipfire-config/src/lib.rs | stable |
| `HIPFIRE_SILU_FP8_AWQ` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_SILU_FP8_H_BF16` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_SILU_FP8_KERNEL` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_SILU_HIN` | crates/hipfire-isa/src/kernels/iu4_v2b_a4.rs, crates/rdna-compute/src/gemv.rs | developer |
| `HIPFIRE_SILU_H_BF16` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_SILU_MQ_ROTATE_KERNEL` | crates/rdna-compute/src/gemv.rs, crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_SKIP_AGENTIC_GATE` | scripts/agentic-gate.sh | harness |
| `HIPFIRE_SKIP_BUILD` | scripts/container-gate.sh | harness |
| `HIPFIRE_SKIP_LLAMA_COMMIT_CHECK` | crates/hipfire-runtime/src/eval_common.rs | developer |
| `HIPFIRE_SLOTS_ATTN_CROSSOVER` | crates/hipfire-arch-qwen35/src/forward_slots.rs, crates/rdna-compute/src/feature_flags.rs | developer |
| `HIPFIRE_SLOTS_DECODE_GRAPH` | crates/rdna-compute/src/feature_flags.rs | developer |
| `HIPFIRE_SLOTS_PAGED` | crates/hipfire-arch-qwen35/src/serve_engine.rs | developer |
| `HIPFIRE_SLOTS_PAGED_PAGES` | crates/hipfire-arch-qwen35/src/serve_engine.rs | developer |
| `HIPFIRE_SLOT_TRACE` | crates/rdna-compute/src/feature_flags.rs, scripts/serve_concurrency_gate.sh | developer |
| `HIPFIRE_SMOKE_KV` | crates/hipfire-arch-qwen35/src/qwen35/forward.rs, crates/saddle-lab/examples/a3b_smoke_forward.rs | developer |
| `HIPFIRE_SMOKE_KV_SEQ` | crates/saddle-lab/examples/a3b_smoke_forward.rs | harness |
| `HIPFIRE_SMOKE_MODE` | crates/hipfire-arch-qwen35/src/qwen35/forward.rs, crates/saddle-lab/examples/a3b_smoke_forward.rs | developer |
| `HIPFIRE_SMOKE_PROMPT` | crates/hipfire-arch-qwen35/src/qwen35/forward.rs, crates/saddle-lab/examples/a3b_smoke_forward.rs | developer |
| `HIPFIRE_SMOKE_STEPS` | crates/hipfire-arch-qwen35/src/qwen35/forward.rs, crates/saddle-lab/examples/a3b_smoke_forward.rs | developer |
| `HIPFIRE_SPECULATION` | crates/hipfire-config/src/lib.rs, docs/investigations/evidence/ds4-mi300x-cdna-test-fail/raw/a1-m0/04-run-profile-direct.sh | stable |
| `HIPFIRE_SPEC_PHASES` | crates/hipfire-arch-qwen35/src/speculative.rs | developer |
| `HIPFIRE_SPEC_WINDOW_ROLLBACK` | crates/hipfire-config/src/lib.rs | developer |
| `HIPFIRE_SPE_OUT_ROOT` | benchmarks/quality-baselines/harness/spe_ablation.sh | harness |
| `HIPFIRE_SPILL_DIR` | crates/hipfire-config/src/lib.rs, crates/hipfire-quantize/src/pipeline.rs | stable |
| `HIPFIRE_STATE_QUANT` | crates/hipfire-arch-qwen4/examples/qwen4_kld.rs, crates/hipfire-arch-qwen4/examples/qwen4_qsa_ctx.rs | harness |
| `HIPFIRE_SWEEP_MAX` | scripts/ddtree_budget_sweep.sh | harness |
| `HIPFIRE_SWEEP_OUT` | scripts/mq3-mq2-sweep.sh, scripts/spec_decode_genre_sweep.sh | harness |
| `HIPFIRE_SWEEP_PROMPTS_DIR` | scripts/mq3-mq2-sweep.sh | harness |
| `HIPFIRE_SWEEP_RUNS` | scripts/ddtree_budget_sweep.sh | harness |
| `HIPFIRE_T5_GPU` | crates/hipfire-arch-diffusion/examples/gpu_pipeline_parity.rs, crates/hipfire-arch-diffusion/src/pipeline.rs | developer |
| `HIPFIRE_TARGET_ARCH` | crates/rdna-compute/src/dispatch.rs, scripts/kernel_atlas.py | developer |
| `HIPFIRE_TEMP` | docs/investigations/evidence/ds4-mi300x-cdna-test-fail/raw/a1-m0/04-profile-feed.py, docs/investigations/evidence/ds4-mi300x-cdna-test-fail/raw/a1-m0/04-run-profile-direct.sh | harness |
| `HIPFIRE_TEST_MODEL` | scripts/test-ds4-heterogeneous-abort-resume.sh, scripts/test-qwen35-abort-resume.sh | harness |
| `HIPFIRE_TEXT_OUT` | docs/investigations/evidence/ds4-mi300x-cdna-test-fail/raw/a1-m0/04-profile-feed.py, docs/investigations/evidence/ds4-mi300x-cdna-test-fail/raw/a1-m0/04-run-profile-direct.sh | harness |
| `HIPFIRE_THINK_CONTINUATION` | crates/hipfire-arch-qwen35/src/spec_emit.rs, crates/hipfire-daemon/src/main.rs | developer |
| `HIPFIRE_TIER_RATIO` | crates/hipfire-quantize/src/cli.rs | developer |
| `HIPFIRE_TOPK8_RESCORE` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_TOPK_FIXUP_MARKED` | crates/rdna-compute/src/select_regrid.rs | developer |
| `HIPFIRE_TP` | crates/hipfire-arch-qwen35/examples/qwen_dense_tp2_parity.rs | harness |
| `HIPFIRE_TP4_GRAPH_REPLAYS` | crates/hipfire-runtime/examples/tp4_cross_device_graph_barrier.rs | harness |
| `HIPFIRE_TP_BENCH_ITERS` | crates/hip-bridge/examples/rccl_smoke.rs, crates/hipfire-runtime/examples/tp_allreduce_smoke.rs | harness |
| `HIPFIRE_TP_BENCH_N` | crates/hip-bridge/examples/rccl_smoke.rs, crates/hipfire-runtime/examples/tp_allreduce_smoke.rs | harness |
| `HIPFIRE_TP_BENCH_WARMUP` | crates/hipfire-runtime/examples/tp_allreduce_smoke.rs | harness |
| `HIPFIRE_TP_GRAPH_RANKS` | crates/hipfire-runtime/examples/tp4_cross_device_graph_barrier.rs | harness |
| `HIPFIRE_TP_PARITY_PROMPT_FILE` | crates/hipfire-arch-qwen35/examples/qwen_dense_tp2_parity.rs | harness |
| `HIPFIRE_TP_PARITY_STEPS` | crates/hipfire-arch-qwen35/examples/qwen_dense_tp2_parity.rs | harness |
| `HIPFIRE_TP_PARITY_TP` | crates/hipfire-arch-qwen35/examples/qwen_dense_tp2_parity.rs | harness |
| `HIPFIRE_TP_PEER_DIRECT` | crates/hipfire-arch-qwen35/src/qwen35/forward.rs | developer |
| `HIPFIRE_TP_USE_RCCL` | crates/hipfire-config/src/lib.rs, crates/hipfire-runtime/src/config.rs | stable |
| `HIPFIRE_TQ2G128_XBATCH` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_TQ2G128_XBATCH_KERNEL` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_TQ2G128_XBATCH_MAX` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_TQ2_MODEL` | benchmarks/quality-baselines/harness/spe_ablation.sh | harness |
| `HIPFIRE_TRANSFORMERS_CHECKOUT` | crates/hipfire-arch-qwen4/reference_oracle/upstream.py | harness |
| `HIPFIRE_TUI_BIN` | crates/hipfire-cli/src/main.rs, crates/hipfire-config/src/lib.rs | stable |
| `HIPFIRE_UNIFORM_GATE_UP` | crates/hipfire-runtime/examples/hfq_splice_attn.rs | harness |
| `HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB` | crates/hipfire-cli/src/serve/complete.rs, crates/hipfire-config/src/lib.rs | stable |
| `HIPFIRE_UNSAFE_WSL_REDLINE` | crates/hipfire-config/src/devices.rs, crates/hipfire-config/src/lib.rs | experimental |
| `HIPFIRE_UNSAFE_WSL_VMM_KV` | crates/hipfire-config/src/devices.rs, crates/hipfire-config/src/lib.rs | experimental |
| `HIPFIRE_V2B_A4_EPI` | crates/hipfire-arch-qwen35/src/qwen35/prefill.rs, crates/rdna-compute/examples/test_mq4v2_gate_up_a4_gfx1151.rs | developer |
| `HIPFIRE_V2B_A4_PM_BUNDLE` | crates/rdna-compute/src/gemm.rs, crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_V2B_ADDEPI` | crates/hipfire-arch-qwen35/src/qwen35/prefill.rs, crates/rdna-compute/src/dispatch.rs | developer |
| `HIPFIRE_V2B_DOWN_SWZ` | crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_V2B_PM` | crates/rdna-compute/src/gemm.rs, crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_V2B_PM_BUNDLE` | crates/rdna-compute/src/gemm.rs, crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_V2B_ZBA_SCATTER` | crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_V2C_ADDEPI` | crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_VAE_CONFIG_ONLY` | crates/hipfire-arch-diffusion/examples/flux_txt2img.rs, crates/hipfire-arch-diffusion/examples/gpu_flux_golden_latent.rs | developer |
| `HIPFIRE_VAE_CONV` | crates/hipfire-arch-diffusion/src/vae_gpu.rs | developer |
| `HIPFIRE_VAE_FUSE_NORM` | crates/hipfire-arch-diffusion/src/vae_gpu.rs | developer |
| `HIPFIRE_VAE_GPU` | crates/hipfire-arch-diffusion/src/pipeline.rs | developer |
| `HIPFIRE_VAE_IM2COL_MAP` | crates/rdna-compute/src/vae.rs | developer |
| `HIPFIRE_VAE_IM2COL_MB` | crates/hipfire-arch-diffusion/src/vae_gpu.rs | developer |
| `HIPFIRE_VAE_IM2COL_TILE` | crates/rdna-compute/src/vae.rs | developer |
| `HIPFIRE_VAE_PROFILE` | crates/hipfire-arch-diffusion/src/vae_gpu.rs | developer |
| `HIPFIRE_VAE_TRANSPOSE` | crates/rdna-compute/src/vae.rs | developer |
| `HIPFIRE_VCN_DEBUG` | crates/va-bridge/src/interop.rs, crates/va-bridge/src/lib.rs | developer |
| `HIPFIRE_VCN_DRM_NODE` | crates/va-bridge/src/lib.rs | developer |
| `HIPFIRE_VCN_LIBVA_PATH` | crates/va-bridge/src/ffi.rs | developer |
| `HIPFIRE_VERIFY_ATTN` | crates/hipfire-config/src/lib.rs, crates/railgun-cert/src/recording.rs | stable |
| `HIPFIRE_VERIFY_GRAPH` | crates/hipfire-arch-qwen35/src/mtp_probe.rs, crates/hipfire-arch-qwen35/src/speculative.rs | developer |
| `HIPFIRE_VERIFY_GRAPH_TIMING` | crates/hipfire-arch-qwen35/src/speculative.rs | developer |
| `HIPFIRE_VERIFY_GRAPH_TREE` | crates/hipfire-arch-qwen35/src/speculative.rs, scripts/tree_graph_bench.sh | developer |
| `HIPFIRE_VERSION` | crates/hipfire-runtime/examples/build_kld_ref.rs, crates/hipfire-runtime/examples/build_kld_ref_native_gemma4.rs | harness |
| `HIPFIRE_VISION_FIXTURE` | crates/hipfire-generate/tests/vision_lifecycle_tests.rs | harness |
| `HIPFIRE_VISION_IMAGE` | crates/hipfire-generate/tests/vision_lifecycle_tests.rs | harness |
| `HIPFIRE_VISION_LIFECYCLE_CHILD` | crates/hipfire-generate/tests/vision_lifecycle_tests.rs | harness |
| `HIPFIRE_VISION_MODE` | crates/hipfire-config/src/lib.rs | stable |
| `HIPFIRE_VISION_SIDECAR` | crates/hipfire-cli/src/main.rs, crates/hipfire-cli/src/serve/mod.rs | developer |
| `HIPFIRE_VIT_ATTN` | crates/hipfire-arch-qwen35-vl/src/qwen35_vl.rs | developer |
| `HIPFIRE_VLLM_CHECKOUT` | crates/hipfire-arch-qwen4/reference_oracle/upstream.py | harness |
| `HIPFIRE_VL_DUMP_DIR` | crates/hipfire-arch-qwen35-vl/src/qwen35_vl.rs, crates/saddle-lab/examples/infer.rs | developer |
| `HIPFIRE_VL_FILE` | crates/hipfire-daemon/src/slots.rs, crates/hipfire-loader/src/carriers.rs | developer |
| `HIPFIRE_VL_MAX_PIXELS` | crates/hipfire-arch-qwen35-vl/src/image.rs | developer |
| `HIPFIRE_VL_SEQUENTIAL` | crates/hipfire-arch-qwen35/src/serve_engine.rs, crates/hipfire-runtime/src/scheduler.rs | developer |
| `HIPFIRE_VMM_ACCESS_DEVICE` | crates/hip-bridge/examples/vmm_arena_smoke.rs | harness |
| `HIPFIRE_VMM_CHUNK_BYTES` | crates/rdna-compute/examples/vmm_tensor_smoke.rs | harness |
| `HIPFIRE_VMM_FIRST_BYTES` | crates/hip-bridge/examples/vmm_arena_smoke.rs | harness |
| `HIPFIRE_VMM_SECOND_BYTES` | crates/hip-bridge/examples/vmm_arena_smoke.rs | harness |
| `HIPFIRE_VMM_SMOKE_DEVICE` | crates/hip-bridge/examples/vmm_arena_smoke.rs, crates/rdna-compute/examples/vmm_tensor_smoke.rs | harness |
| `HIPFIRE_VRAM_BUDGET_BYTES` | crates/rdna-compute/examples/q8_batched_attn_microbench.rs | harness |
| `HIPFIRE_WEIGHT_BUFFER_LOADS_FLAT_GEMV_OPT_IN` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_WEIGHT_BUFFER_LOADS_OPT_IN` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_WEIGHT_CACHE_FLAT_GEMV` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_WEIGHT_CPOL_AUX` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_WIDENED_PBS_GROW_ONLY` | crates/hipfire-arch-qwen35/src/qwen35/prefill.rs | developer |
| `HIPFIRE_WMMA_FA` | benchmarks/results/wmma-fa-probe-gfx1100.sh, benchmarks/results/wmma-fa-probe.sh | developer |
| `HIPFIRE_WMMA_FA_MIN_BATCH` | crates/rdna-compute/src/attention.rs | developer |
| `HIPFIRE_WMMA_PREFILL` | crates/hipfire-arch-gemma4/examples/prefill_parity_gemma4.rs, crates/hipfire-arch-gemma4/src/lowered.rs | developer |
| `HIPFIRE_WO_MMQ` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_WO_WMMA_VARIANT` | crates/hipfire-config/src/lib.rs, crates/hipfire-runtime/examples/test_wmma_correctness.rs | experimental |
| `HIPFIRE_XDNA_A` | crates/hipfire-xdna/src/lib.rs | developer |
| `HIPFIRE_XDNA_B` | crates/hipfire-xdna/src/lib.rs | developer |
| `HIPFIRE_XDNA_CREF` | crates/hipfire-xdna/src/lib.rs | developer |
| `HIPFIRE_XDNA_INSTS` | crates/hipfire-xdna/src/lib.rs | developer |
| `HIPFIRE_XDNA_MKN` | crates/hipfire-xdna/src/lib.rs | developer |
| `HIPFIRE_XDNA_PDI` | crates/hipfire-xdna/src/lib.rs | developer |

<!-- env-inventory:end -->

---

## Maintenance

```bash
# Independently verify concrete Rust HIPFIRE_* reads against table rows.
scripts/regen-env-vars-doc.sh

# Shipped env/docs coverage and production-ownership check.
python3 scripts/check-env-docs.py
```

Also match `std::env::var_os("HIPFIRE_…")` (e.g. `HIPFIRE_KERNEL_CACHE` in `compiler.rs`) — token scan catches the name regardless of `var` vs `var_os`.

When adding a user-facing knob:

1. Prefer a typed field + validation in `crates/hipfire-config/src/lib.rs` ([`CONFIG.md`](CONFIG.md)).
2. Add the env name to product docs only if operators must set it outside config.
3. Re-scan so the generated inventory stays complete.
4. Do not document unearned or widened LFM defaults here (multi-cohort, path/extension selection of `.mq4`, automatic runtime default for non-`.mq4r`, or generic default-on beyond the exact sealed [`admissions.yml`](admissions.yml) evidence row). That LFM row is registry evidence without current automatic runtime wiring; only `retained_redline_default` auto-selects (`.mq4r` + exact GPU arch + pp=tp=1, Qwen3.5 dense and Qwen4 MTP-off plain-AR on exact gfx1201, DeepSeek4 `.mq2r` AR on gfx1151). Planned broader admissions may not be documented as shipped.

**Last inventory verification:** 2026-07-29.

## Batched attention (SP1)

| Variable | Default | Meaning |
|---|---|---|
| `HIPFIRE_ATTN_TILE_SIZE` | `128` | Tile size for the batched attention tile+reduce path. Must be a positive multiple of 32; anything else falls back to 128. Resolved once via `Gpu::attn_tile_size()`. **Raising it is safe; lowering it increases `max_tiles` and therefore the `partials` bytes per query row, which can exceed buffers sized elsewhere against the 128 default.** |
| `HIPFIRE_VRAM_BUDGET_BYTES` | 32 GiB | Deployment-target VRAM ceiling used by the SP1 benchmark harnesses' preflight. Read by `examples/`, not by production code. |
| `HIPFIRE_CPU_EXEC_TRACE` | unset (off) | Developer diagnostic for `memory.offload_exec=cpu`. With `=1`, the CPU-exec seam prints one line per **distinct step shape** — `m`, `k`, quant, whether the activation was rotated, whether a residual was accumulated, whether an AWQ sidecar applied — at that shape's first call and then at every doubling of its call count, each with the running counters: steps run on the CPU and **host-mapped steps that stayed on the GPU**. The second number must be `0` for a model the CPU covers (a non-zero value means a step shape never reached the seam). Each line's D2H / GEMV / H2D split is the mean over *that shape's* calls so far, so the `calls=1` line is the cold first step and the later lines (…32, 64, 128 calls) are its steady state — the numbers that are worth quoting. A value that does not move with the call count is the shape's own cost, not a process-wide average. Read via `hipfire_config::developer_var`, so it works as an environment variable or a `[developer]` TOML key. |
| `HIPFIRE_GPU_LAYER_BUDGET` | unset (fully resident) | Typed key `memory.gpu_layer_budget`: the resident-layer budget for partial GPU offload. `N` keeps the last `N` layers on the GPU and spills the prefix `[0 .. n_layers-N)` to host RAM. The number counts layers **on** the GPU, not layers offloaded — `3` on a 64-layer model spills 61. `auto`/`-1` defers placement to the engine, which currently keeps every layer resident; unset, empty and unparseable values also resolve to fully resident, so a bad value never forces an offload. Placement is resolved once at load and never thrashes per request; `token_embd`/`output_norm`/`lm_head` are always resident. Spilled weights are `hipHostMalloc(hipHostMallocMapped)` system RAM; *who multiplies* them is `memory.offload_exec`. Full write-up: [`plans/partial-gpu-offload-design.md`](plans/partial-gpu-offload-design.md). |
| `HIPFIRE_OFFLOAD_EXEC` | `pcie` | Typed key `memory.offload_exec`, the sibling of `memory.gpu_layer_budget`. Decides **who multiplies** a spilled layer's weights: `pcie` (default) runs the GPU kernels against host-mapped weights over the link, `cpu` executes those GEMVs on the CPU (`crates/hipfire-cpu`). Unset, empty and unknown all fail closed to `pcie`. Placement, VRAM accounting and KV residency are unchanged; with nothing spilled it prints one informational line and changes nothing. Refused at load together with a retained-replay (Redline) backend. Per-step diagnostics: `HIPFIRE_CPU_EXEC_TRACE=1`. Full write-up: [`plans/partial-gpu-offload-design.md`](plans/partial-gpu-offload-design.md#621-cpu-execution-of-the-spilled-weight-ops-memoryoffload_execcpu). |
| `HIPFIRE_OFFLOAD_EXEC` | crates/hipfire-config/src/lib.rs |
| `HIPFIRE_CPU_EXEC_TRACE` | crates/hipfire-dispatch/src/cpu_exec.rs |
| `HIPFIRE_OOM_GUARD` | `auto` | Typed key `memory.oom_guard`. Gates **only** the host `MemAvailable` headroom half of `kv_slots::preflight_alloc` / `SlotPool` / CLI bench-sweep preflight. The R9700 deployment-target VRAM-budget check **always runs**. `auto`: on for unified-memory APU archs (gfx1035/1036/1103/1150/1151/1152), off for recognized discrete GPUs, and for processes with no known GPU arch by host swap state (no swap → on; unreadable → on). `1`/`true`/`0`/`false` force either way. The auto decision is logged once to stderr with its reason. `scripts/run-bounded.sh` remains the hard backstop. Full write-up: [`CONFIG.md`](CONFIG.md#memoryoom_guard). |
| `HIPFIRE_MEM_CAP` | `24G` | Read by `scripts/run-bounded.sh`, not by the binaries: cgroup `MemoryMax` for a gated run. Exit 137 means the cap fired — shrink the configuration rather than raising it. |
