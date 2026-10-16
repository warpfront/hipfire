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
| `HIPFIRE_SERVE_API_KEY_FILE` | Serve bearer credential file | Optional absolute path; maps to `serve.api_key_file` (default unset = no authentication). Contains UTF-8, >=32 bytes after trimming trailing whitespace; Unix mode must exclude group/other (`chmod 600`). Windows ACLs are not checked yet and no key is auto-generated. Recommended on a non-loopback bind; use HTTPS for remote traffic. Only the path, never secret contents, belongs in config/env. See [SERVE.md](SERVE.md#security-bearer-authentication-and-https-proxies). |
| `HIPFIRE_DAEMON_BIN` | Daemon binary override | |
| `HIPFIRE_LOCK_DIR` | Shared per-GPU lock directory | Absolute writable path; daemon and `gpu-lock.sh` must use the same setting to contend. Default on Linux/WSL `/run/lock/hipfire` if writable, otherwise `/tmp/hipfire-locks`; on native Windows `%ProgramData%\hipfire\locks`, otherwise `%TEMP%\hipfire-locks` (`LockFileEx`, same file names and holder-PID reporting). Different directories do not see each other's locks. Files are `gpu-GPU-<uuid>.lock`, or `gpu-pci-<dddd:bb:dd.f>.lock` for cards without a UUID. |
| `HIPFIRE_TUI_BIN` | TUI binary | |
| `HIPFIRE_ROCM_PATH` | hipfire-specific ROCm SDK root override | Highest priority (`HIPFIRE_ROCM_PATH` > `ROCM_PATH` > `HIP_PATH`). Must provide the runtime, headers, and `hipcc`. Authoritative: no fallback to another install or bare soname. |
| `ROCM_PATH` / `HIP_PATH` | ROCm/HIP compatibility root overrides | Used only when `HIPFIRE_ROCM_PATH` is unset (`ROCM_PATH` before `HIP_PATH`). `HIP_PATH=<root>/hip` normalizes to `<root>`. Multiple equally eligible roots without an override are refused — set `HIPFIRE_ROCM_PATH`. |
| `HIPFIRE_TEST_REQUIRE_ROCM` | ROCm oracle test availability | Defaults to required: ISA/PeaceMaker oracle tests use `ROCM_PATH`, else `/opt/rocm`, and fail on missing tools. Set exactly `0` only for explicitly no-ROCm hosts (GPU-free CI). This does not change runtime SDK discovery. |
| `HIPFIRE_TEST_ROCM_LLVM_BIN` | Historical ROCm oracle LLVM-bin override (harness; no longer read) | Older oracle tests used this directory instead of `/opt/rocm/core-10.0/lib/llvm/bin`. Current tests use `ROCM_PATH` and `HIPFIRE_TEST_REQUIRE_ROCM`; this spelling does not affect runtime SDK discovery. |
| `HSA_USERPTR_FOR_PAGED_MEM` | ROCm (libhsakmt) host-allocation backing | Linux, in a process that sets `HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS` (host-mapped Qwen4 experts), or a daemon started for a Qwen4 model whose cards are all discrete GPUs (`hipfire run`, `bench`, and `serve`'s pre-warm model; the arch of the cards comes from the KFD topology): hipfire sets `0` before the HIP runtime loads, unless it is already set, and logs it on a `[hip-bridge] … host memory out of reclaim` line. Other processes keep ROCm's default. With `0`, `hipHostMalloc` memory (the host-mapped routed experts, clr's staging buffers) is GTT, outside the kernel's reclaim. Under ROCm's default (non-zero) it is pageable userptr: when reclaim takes those pages, KFD evicts the process's GPU queues, and host-memory pressure can stall the GPU with one core spinning in `hipMemcpy` (ROCm/rocm-systems#12528). GTT is capped by TTM's `pages_limit` (`/sys/module/ttm/parameters/pages_limit`, in pages; half of RAM by default). A Qwen4 load whose host-mapped experts do not fit beside the GTT that amdgpu devices already hold is refused before it allocates. When the process exits, amdgpu keeps its freed GTT pages in TTM's page pool (up to `/sys/module/ttm/parameters/page_pool_size`, half of RAM by default). `MemAvailable` does not count the pool, but the next GTT allocation takes pages from it and memory pressure shrinks it, so the Qwen4 host-RAM check adds an estimate of it (see `HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS`). Operators can return the pool to the kernel at once with `echo 2 \| sudo tee /proc/sys/vm/drop_caches`; `sudo cat /sys/kernel/debug/ttm/page_pool` shows its size (last line, in pages). Opt out with `HSA_USERPTR_FOR_PAGED_MEM=1`, which brings the stall exposure back. |
| `GPU_PINNED_MIN_XFER_SIZE` | ROCm (clr) pageable-copy threshold; explicit values in MiB | Linux, in **every** process: hipfire sets `100000` before the HIP runtime loads, unless it is already set or `HIPFIRE_PINNED_XFER_DEFAULT=0`, so every copy to or from pageable host memory goes through clr's pinned staging buffer and the copy *source* is never registered as a reclaimable KFD userptr. That source exists on every weight load, so this is the switch that covers every model — the mapped model file, and Qwen4's per-forward PLE rows. Historical gfx1201 / Qwen3.8-27B MQ3-Pro measurements (warm cache, three reps each): the weight sweep was 1190–1198 ms armed vs 1204–1210 ms unarmed; these are not new performance evidence. At or above the threshold, clr pins a copy's pageable source in place as a userptr, which carries the queue-eviction stall under host-memory pressure (see `HSA_USERPTR_FOR_PAGED_MEM`). Set `HIPFIRE_PINNED_XFER_DEFAULT=0` before process start for the exact opt-out (see its row): hipfire then leaves `GPU_PINNED_MIN_XFER_SIZE` unset and clr keeps its own default. Setting `GPU_PINNED_MIN_XFER_SIZE=0` instead is not an opt-out: it pins every pageable copy source in place, including copies below clr's built-in threshold, so it does **not** restore the previous unset CLR default. Explicit threshold values are in **MiB**; the built-in threshold can differ by CLR version and need not be expressible in whole MiB (current CLR: 64 KiB; older CLR: 128 MiB). **Undocumented clr knob** — absent from HIP's environment-variable reference and from AMD's env-var page (checked 2026-10-03 against ROCm 7.2); the name is a string in `libamdhip64.so` next to `GPU_PINNED_XFER_SIZE`, read once at runtime initialization, with no published default. Treat it as internal, and note that on APUs it is *not* ignored the way `HSA_USERPTR_FOR_PAGED_MEM` is. |
| `HIPFIRE_PINNED_XFER_DEFAULT` | exact opt-out from the process-wide `GPU_PINNED_MIN_XFER_SIZE` default; unset: default armed | Linux, read before the HIP runtime loads (installed process config, else the ambient environment). Exactly `0`: hipfire does **not** set `GPU_PINNED_MIN_XFER_SIZE`, so clr keeps its own unset default, and logs one `[hip-bridge] HIPFIRE_PINNED_XFER_DEFAULT=0: GPU_PINNED_MIN_XFER_SIZE left unset (clr default)` line. This is exact restoration of the pre-default behavior, unlike `GPU_PINNED_MIN_XFER_SIZE=0`, and it re-exposes the reclaim-stall exposure described on the `GPU_PINNED_MIN_XFER_SIZE` row. Any other value or unset keeps the `100000` default; a `GPU_PINNED_MIN_XFER_SIZE` already set in the environment is always respected, with or without this variable. It does not touch `HSA_USERPTR_FOR_PAGED_MEM` or the Qwen4 host-mapping tier. |
| `HSA_CU_MASK` / `ROC_GLOBAL_CU_MASK` | ROCm CU affinity masks; **refused on RDNA** | Linux, every process: before ROCr (`Runtime::initialize`) or the HIP runtime (`HipRuntime::load`) loads, hipfire reads the real process environment. A non-blank value is refused with an error when any KFD GPU node is RDNA (gfx10/gfx11/gfx12), or when the topology cannot be read or lists no GPU, because no RDNA device can then be ruled out. It deliberately ignores `ROCR_VISIBLE_DEVICES`. Only all-gfx9 (GCN/CDNA) hosts pass. On 2026-10-14 a CU-masked queue probe hung the R9700 (gfx1201) MES scheduler, forced mode1 resets and took the host down; the mask stays banned until a safe method is approved. The old `200× under ROC_GLOBAL_CU_MASK=0x1` stress recipes are withdrawn. The retained-PM4 queue mask entry points (`AqlQueue::create_with_cu_mask`, `QueueSet::create_with_cu_masks`/`set_cu_mask`) refuse RDNA devices too, and `HIPFIRE_GFX1151_REDLINE_CU_COUNT` accepts only `all`. |
| `GPU_MAX_HW_QUEUES` | ROCm hardware-queue cap | hipfire never sets it. Read by ROCm before HIP loads, process-wide. With native MTP opted in over a `pcie` host-mapped expert spill on `gfx1200`/`gfx1201`, `GPU_MAX_HW_QUEUES=1` avoids a reported random decode-throughput loss; it is not measured on any other route, so it stays an operator choice. |
| `HIPFIRE_LOCAL=1` | Skip attaching to running serve | One-shot local daemon. |
| `HIPFIRE_REGISTRY_URL` | Dynamic registry fetch URL | |
| `HIPFIRE_NO_REGISTRY_FETCH=1` | Pin bundled registry | |

### KV / attention / prompt

| Variable | Default / sense | Source |
|---|---|---|
| `HIPFIRE_KV_MODE` | From config; **`auto` → registry non-q8 `default_kv_mode`, else Qwen-family native `fp8` on exact gfx1201 when native-eligible (H24/Hkv4/D256, single GPU, no adaptive/CASK), else Qwen Q8/Q8** (gfx1100, gfx1151 and every other load; explicit `--kv-mode q8` still honored). `fwht2`/`fwht3`/`fwht4` are optional headroom modes that `auto` does not select (only the kill switch below does); `asymN`/`turboN` are legacy spellings (Qwen aliases of `fwhtN`). Qwen4 / Flash-Next (arch 16) accepts only `auto`, `bf16`, `fp8` and `q8`: `auto` is fp8 QSA K/V on exact gfx1201 and `bf16` (the exact F32 state) elsewhere; an explicit `fp8` is also served on exact gfx942 (FNUZ-hardware E4M3 rows with F32 index keys, 1,672 B per layer-token on Flash-Next; gfx942 `auto` never selects it); `q8` (opt-in, gfx11/gfx12) stores the QSA K/V as Q8_0 blocks of 32 ([`CONFIG.md`](CONFIG.md#mode-k-and-v)). Non-Qwen families keep their own defaults (Maple BF16, Gemma layered, DeepSeek compressor). | CLI / pair resolver; **not** a legacy hard-coded fwht-per-arch table |
| `HIPFIRE_KV_ADAPTIVE` | off unless set / param | Loader/CLI |
| `HIPFIRE_KV_PHYSICAL_CAP` | optional physical slot cap | Daemon |
| `HIPFIRE_KV_V` | **developer-only** V-axis override (e.g. `lloyd2`/`lloyd3`/`lloyd4`); **lower precedence** than an authored `--kv-v` or `memory.kv_v` | Qwen carrier (`developer_var`); not a second user config plane — prefer CLI/TOML |
| `HIPFIRE_QWEN_KV_DEFAULT_Q8` | default **ON** (implicit Qwen Q8/Q8 wherever `auto` does not pick native fp8). **`=0`** is the emergency kill switch: restores the prior *implicit* HFQ/PaRo defaults on every load that does not get native fp8, gfx1201 included (HFQ `auto`/unset and PaRo `"auto"` → FWHT3/Q8; PaRo raw unset stays Q8). Does **not** override authored `--kv-mode`/`--kv-k`/`--kv-v` or `memory.kv_*`, native gfx1201 fp8, or non-Qwen families. | Loader admission (`qwen_default_q8_enabled`); sampled once per load |
| `HIPFIRE_ATTN_FLASH` | from `flash_mode` (`auto`/`always`/`never`) | CLI → daemon |
| `HIPFIRE_YARN_FACTOR` | unset = checkpoint `rope_parameters`; `>1` YaRN factor over the original context; `1` disables | `attention.yarn_factor`; `RuntimeConfig` → `hipfire_runtime::rope` (Qwen4) |
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
| `HIPFIRE_DFLASH_MODE` | RuntimeConfig default **`off`**; Qwen3.8-27B tiers with a registry-declared draft resolve `auto` through the registry layer; `off` beats it | Schema alias of `speculation.dflash` ([`CONFIG.md`](CONFIG.md#qwen38-27b-paired-dflash-default)). `on` requires a DFlash draft and refuses without one; it never aliases to native MTP. Use `--spec mtp` / `speculation.mtp=on` for native MTP. |
| `HIPFIRE_DFLASH_NGRAM_BLOCK` | set/clear from config | |
| `HIPFIRE_DFLASH_ADAPTIVE_B` | **kill switch**; `=0` forces the fixed full block, any other value or unset defers to the setting | Read through the `[developer]` namespace (lifecycle `developer`; it is **not** a schema alias — the schema key is `speculation.dflash_adaptive_b`, stable, default `false` = fixed block, listed in the [`CONFIG.md`](CONFIG.md#lifecycle-status) table with no env column). `=0` overrides an enabled `dflash_adaptive_b` load param/setting; it cannot turn adaptive on. Adaptive is also auto-suppressed on retained-PM4 verify loads (replay needs the fixed B=16 shape). |
| `HIPFIRE_DFLASH_BLOCK` | **`auto`**; `auto` or `4`–`16`; **`16` is the kill switch** (previous fixed width) | Schema alias of `speculation.dflash_block` ([`CONFIG.md`](CONFIG.md#mechanism-knobs)): the DFlash2 proposal width L of one load, chosen per serving configuration and never per live batch, so batched VMM lanes and singletons of the same load run the same L and stay byte-identical. `auto` = 8 on a `serve.vmm_batch` serve with `continuous_batch_size` ≥ 4, else 16 (resolved by the daemon from the staged route). Legacy DFlash drafts keep their declared width; the retained-PM4 verify route and dense TP2 need 16; experimental multi-slot admits only 16. |
| `HIPFIRE_DFLASH_ONLINE_TUNE` | **unset / off**; `stats` or `on` | Developer-only online draft tuning (`crates/hipfire-runtime/src/dflash_online.rs`). Covers Qwen3.5 chain DFlash on the greedy batched LM-head path, and the generic (llama-family) DFlash chain at greedy and temp>0. Takes the draft's top-16 per row on device and re-ranks it on the host with a per-request model trained from tokens the target already verified. Features: chain n-grams, suffix match, stutter, neighbour-row tokens. Verify is unchanged. `stats` keeps argmax proposals and prints acceptance-ceiling stats. `sweep` drafts a token the target never picks, so a dump records the draft's top-K at every position for `--simulate`. Learner weights carry across requests; session text statistics reset per request. `HIPFIRE_DFLASH_ONLINE_HP=lr=..,acc0=..,margin=..,inject=..,carry=0|1` overrides learner settings. `HIPFIRE_DFLASH_ONLINE_DUMP=<file>` writes records for `examples/dflash_online_replay.rs`. |
| `HIPFIRE_DFLASH_CKPT_RESUME` / `HIPFIRE_CACHE_CKPT_*` | checkpointing | Qwen DFlash and MTP divergent-render resume |
| `HIPFIRE_SPEC_WINDOW_ROLLBACK` | on unless `0` | Enables retained pre-window repair for strict-prefix speculative terminals; `0` keeps the conservative reset path. |
| `HIPFIRE_SPEC_RETIRE` | **unset → `1.5`**; `off`/`0` disables; `<base>[,<per_1k_ctx>]` sets the break-even line | Per-request spec → AR retirement (`crates/hipfire-runtime/src/spec_retire.rs`) for greedy requests on Qwen3.5-family DFlash and generic (llama-family) DFlash. Every 16 spec cycles the request's mean committed tokens per cycle is compared with `base + per_1k_ctx × position / 1024`; two failing windows in a row switch the rest of the request to target-only AR tokens (draft context kept in sync). Token counts only, so a greedy request retires at the same cycle on every run. The default retires drafters that commit under 1.5 tokens per cycle, below the smallest measured cycle/AR cost ratio (gfx1151 Qwen3.8-27B: 1.53–1.73 up to 32K). `done` carries `spec_retired`, `spec_retired_at` and `ar_tokens` when a request retired; serve exports `hipfire_spec_retired_total`. Malformed values turn retirement off with a log line. |
| `HIPFIRE_DFLASH_VERIFY_PM4` | **unset / off**; `1` opts in | Retained-PM4 route for the fixed B=16 DFlash2 chain target-verify forward. Admitted only on exact gfx1201, single GPU, dense recurrent Qwen3.5-family target, Q8 KV + Q8 DeltaNet state, DFlash2 selector + dynamic-conv draft, `target_layer_ids == [5,19,33,47,61]`, no DDTree. Every other configuration reports a specific `disabled` reason and runs the unchanged HIP/HipGraph path. |
| `HIPFIRE_GFX1151_XLDS` | **on** for exact gfx1151; `0` opts out (developer) | Exact-gfx1151 1..=16-row MQ4G256V2 WMMA GEMMs (gate_up, qkvza, qkv, residual, including the verify lm_head; K % 256 == 0) launch `gemm_*_mq4g256v2_wmma_gfx1151_xlds` (4 waves x 16 rows per block, X slab staged in LDS once per block) instead of the one-wave base kernels. Per-output WMMA chain, K order, dequant values and epilogue are the base kernels', so outputs are byte-identical. Eager and HipGraph capture (DFlash verify graph) take the route; Redline replay recording keeps the base kernels. Other arches are unchanged. `0` restores the base kernels. Oracle: `rdna-compute/examples/test_mq4v2_xlds_gfx1151` (`--features lab`). |
| `HIPFIRE_CB_VERIFY_CHUNK128` | **unset / off**; `1` opts in (developer) | VMM continuous-batching DFlash verify only (not MTP): admits exact 64..=128-row aggregated verify chunks on exact gfx1201, dense MQ4G256V2 targets, through the explicit `gemm_*_verify_exact` kernels (128 rows per chunk instead of 63). Needs `HIPFIRE_WMMA_BATCH_TILES`. Byte-identical to the 63-row route. Unset/`0` keeps every existing route. Oracles: `cb_verify_gemm_oracle` (kernel bytes) and `cb_vmm_state_oracle` (state). |
| `HIPFIRE_CB_VERIFY_PM` | **on** inside `HIPFIRE_CB_VERIFY_CHUNK128`; `0` opts out (developer) | Selects the embedded PeaceMaker twins for the 64..=128-row exact verify kernels; `0` runs the byte-identical hipcc `_k32` twins. No effect unless the wide verify route is on. Never selects IU4/FP8; a missing PM image is a load error, not a fallback. |
| `HIPFIRE_CB_VERIFY_FP8` | **unset / off**; `1` opts in (developer) | 27B Qwen3.5-hybrid target verify on exact gfx1201 with symmetric MQ4G256V2 weights (DFlash and MTP; singleton = 16 rows, continuous batching = up to 128 rows), including the verify lm_head: the dense verify GEMMs run the embedded PeaceMaker FP8 family (`mq4_fp8_verify_*_pm_gfx1201_bt{BT}w{W}`: weights dequantized exactly to E4M3, per-row E4M3 activations, `v_wmma_f32_16x16x16_fp8_fp8`) whose per-output reduction order is fixed for every row count 1..=128, so batched rows are byte-identical to singleton rows by construction. Not byte-identical to the F16 verify routes (FP8 activation rounding). Fails closed: the scope refuses off-arch / asymmetric artifacts, a missing bundle or symbol is an error, and F16-input GEMM entries error inside the scope. `HIPFIRE_CB_VERIFY_PM` does not apply. Unset/`0` keeps every existing route. Route receipt: `Gpu::fp8_verify_receipt` and one stderr line `[cb-verify-fp8] FP8 verify GEMM route active`. |
| `HIPFIRE_CB_VERIFY_IU4` | **unset = per-load default**: on when `speculation.dflash_block` resolves to 8 (a batching serve: `serve.vmm_batch` with `continuous_batch_size` ≥ 4) and `HIPFIRE_CB_VERIFY_FP8` is off, else off; **`0` is the kill switch**; `1` forces it on (developer) | DFlash target verify with 4-bit (A4) activations. **Numerics differ from the F16 verify; quality is tied on task evals** (Qwen3.8-27B, paired greedy DFlash: HumanEval 157 → 156, GSM8K 5-shot 1282 → 1282, MMLU-Pro subset 871 → 871). F16 == IU4 is not promised; batched == singleton stays byte-identical within the IU4 configuration. Needs exact gfx1201, a symmetric MQ4G256V2 artifact, a dense MQ4G256V2 trunk and head, IU4 prefill and both gfx1201 A4 producer fusions (`HIPFIRE_GFX12_PRODUCER_QUANT_FUSED`, `HIPFIRE_GFX12_SILU_QUANT_FUSED`). The singleton chain verify, the VMM continuous-batching verify trunk (any aggregate row count, including the `HIPFIRE_CB_VERIFY_CHUNK128` 128-row chunk) and the batched rollback replay run every projection through the gfx1201 A4 producer (AWQ RMSNorm/FWHT, gated norm, sigmoid, SwiGLU / F1-lite h) and its IU4 GEMM (`_b1s` SET/ADD, F1-lite gate/up+SwiGLU), whose output rows do not depend on the row count. The lm_head stays F16 and the GDN scan plane F32. The IU4 verify runs eager (no HipGraph verify) and `HIPFIRE_DFLASH_VERIFY_PM4` is disabled with it. `1` together with `HIPFIRE_CB_VERIFY_FP8=1` fails the DFlash load; with FP8 on and this unset, FP8 wins. The load logs `DFlash verify IU4: on|off`. Tree (DDTree) verify is unchanged. Oracles: `iu4_verify_oracle` (row-count byte oracle of every producer + GEMM), `cb_vmm_state_oracle` (IU4-aware at block 8). |
| `HIPFIRE_DFLASH_LEGACY_PREFILL` | **unset / off**; `1` opts out | Qwen35 DFlash target prompt prefill (cold seed, prompt-cache suffix, forced tokens). Default: the seed plans its chunks exactly as AR's ordinary prefill does (`ordinary_prefill_chunk_limit`, then `ordinary_serve_prefill_chunk_len`) and each chunk takes AR's route (widened chunk with `commit_stride`, GDN chunk scan, whole-chunk FA2), so after the prompt the target's KV, DeltaNet state and last-token logits are byte-identical to AR's prefill. Chunks wider than the 256-row ring staging write their hidden rows straight to the ring; this is eager only, and verify and captured forwards keep staging. `1` restores the previous route: 256-row chunks through staging, outside the widened chunk and the chunk scan. Both routes reuse one prefill scratch across chunks. |
| `HIPFIRE_DFLASH_PREFILL_DEVICE` | **on**; `0` opts out (developer) | Qwen35 DFlash cold prompt seed (singleton and VMM lane). Default: extracted hidden rows stay in the device ring between the ring-sized trunk pieces and feed the final exact prompt scatter; a temporary host shadow is captured (per-piece D2H, CPU interleave, full-prompt append) only when the windowed long-reach backfill reads it (`w < w_full` and prompt longer than `w`). The paired all-sliding DFlash2 draft and Legacy drafts never capture. Decode always reads the device-resident draft context; no persistent host shadow exists. `0` restores the per-piece capture for every configuration. No arithmetic, chunk-boundary or GPU state-byte changes. |
| `HIPFIRE_DFLASH_PREFILL_SCATTER_FAST` | **on**; `0` opts out (developer) | Qwen35 DFlash cold and cached-suffix prompt hidden scatter only. Copies each extraction layer into the draft's interleaved target-hidden state with stream-ordered pitched D2D copies, split at source and destination ring wraps, instead of synchronous per-row copies. Identical bytes and ring metadata; no change to projection arithmetic, trunk prefill, decode or terminal repair. |
| `HIPFIRE_DFLASH_ORDERED_DRAFT` | **on**; `0` opts out (developer) | DFlash draft body: the residual-save and context/noise K/V D2D copies are enqueued on the execution stream ahead of their consumer kernels instead of a synchronous null-stream `hipMemcpy` per copy, and the batched lane draft uploads its key positions before the body instead of between the lane fc GEMMs. Same bytes and ranges (byte-identical output); the ordered body is also graph-capturable. `0` restores the synchronous copies. |
| `HIPFIRE_DRAFT_COLLAPSE_OFF` | **unset**; `1` opts out (developer) | DFlash draft GEMM launch collapse. Exact gfx1100: the S7 rotate-to-F16 + overwrite GEMMs and fused draft norms. Exact gfx1201: every MQ4G256V2 draft-body projection and the draft lm_head whose route ends in the one-tile `gemm_mq4g256v2_residual_wmma_gfx12` or 128-K `_vt` kernel (2..=63 rows; no fp8/IU4 verify scope, capture or recording) launches an overwrite twin (`gemm_mq4g256v2_overwrite_wmma_gfx12[_vt*]`, store `acc + 0.0f`, bit-identical to the `hipMemset` pre-zero + `Y += acc` pair) instead of the fill + residual kernel. `1` restores the fill + residual launches on both arches. |
| `HIPFIRE_DRAFT_SHARED_QKV_ROTATION` | **on** (exact gfx1100); `0` opts out | `kernel.draft_shared_qkv_rotation`. DFlash draft on exact gfx1100, MQ4G256V2 draft weights (#838): the noise Q/K/V input is FWHT-rotated once and the three projection GEMMs read that rotation, instead of one identical rotation per projection. Admitted eager only (no replay recording or graph capture), 2..=16 rows, one `k` (multiple of 256) across the group, no AWQ sidecar, no `HIPFIRE_DRAFT_GEMM_DUMP`; anything else, or `0`, rotates per projection. Same GEMM route per projection (the S7 F16 split-K overwrite GEMM, or the batched lm_head GEMM with `HIPFIRE_DRAFT_COLLAPSE_OFF=1`); byte-identical output. Ignored on every other arch. |
| `HIPFIRE_DRAFT_SHARED_FFN_ROTATION` | **on** (exact gfx1100); `0` opts out | `kernel.draft_shared_ffn_rotation`. Same as `HIPFIRE_DRAFT_SHARED_QKV_ROTATION` for the draft FFN gate/up pair (one rotation for both). |
| `HIPFIRE_DRAFT_SHARED_CTX_KV_ROTATION` | **on** (exact gfx1100); `0` opts out | `kernel.draft_shared_ctx_kv_rotation`. Same as `HIPFIRE_DRAFT_SHARED_QKV_ROTATION` for the incremental context K/V pair (one rotation of the newly accepted target-hidden rows for both; a one-row step keeps the GEMV tail). |
| `HIPFIRE_CB_DRAFT_ROWS64` | **on**; `0` opts out (developer) | Batched DFlash draft (continuous batching, greedy lanes): a draft forward packs up to 64 rows instead of 48, so C8 at 8-row lanes drafts in one forward and one draft-head GEMM instead of 48 + 16. Exact gfx1201 only, with the draft overwrite route on (`HIPFIRE_DRAFT_COLLAPSE_OFF` unset), batch-tile kernels, the F16 WMMA head chain and no LDS-staged or A8 arm: a full 64-row chunk's shared GEMMs and head run, inside a scope only that forward opens, on the `_vt4` overwrite twin, which is per-row identical to the one-tile kernel (the legacy ≥ 64-row A8 / IU4 / bt-tile arms are not). Per-lane fc and context-fill GEMMs keep their singleton route. `0` restores the 48-row cap. |
| `HIPFIRE_DFLASH_SELECTOR_LANES` | **on**; `0` opts out (developer) | Batched greedy DFlash2 draft on exact gfx1201 (draft overwrite route on): the candidate selector scores every lane of a draft chunk through ONE selector projection GEMM over the chunk's rows (row-independent overwrite route; inside the 64-row scope for a full chunk), ONE top-K and ONE download each of the projection, ids and values, then each lane's host scoring on its slices (`propose_candidates_lanes_device`), instead of a projection, top-K and three synchronous downloads per lane (3 host syncs per chunk instead of 24 at C8). Proposals byte-identical. `0` restores the per-lane calls. |
| `HIPFIRE_DRAFT_NOISE_EMBED_BATCH` | **on**; `0` opts out (developer) | Batched DFlash draft on exact gfx1201, Q8_0 token embeddings: the draft block's noise embeddings (each lane's seed, then the mask token) are looked up for all rows of the chunk in one `embedding_q8_batched` launch, after one upload of the row ids. Previously it was one `embedding_q8` launch per row (64 at C8). Both kernels compute each element as `f16 scale × int8` with the same striding, so rows are byte-identical. gfx1100 keeps its fused-collapse builder. `0` restores the per-row loop. |
| `HIPFIRE_CB_DRAFT_LANE_CONV` | **on**; `0` opts out (developer) | Batched DFlash2 draft on exact gfx1201 (no capture, replay recording or G0 observation; ≤ 16 lanes): each prepare/finish dynamic causal convolution runs as one `dynamic_causal_conv_f32_lanes` launch over all lanes. That kernel is the `dynamic_conv_f32` body per lane, with grid.z = lane, inputs rebased to the lane's rows and causal guards on the lane-local row. It replaces one launch per lane (160 → 20 per C8 step on the 27B draft). Every output byte is identical. `0` restores the per-lane launches. |
| `HIPFIRE_CB_DRAFT_LANE_CTX` | **on**; `0` opts out (developer) | Batched DFlash2 draft on exact gfx1201 (ordered draft, no capture, replay recording or G0 observation; ≤ 9 new rows per lane, ≤ 72 packed rows, MQ4G256V2 fc/wk/wv). Each lane's new target-hidden fc rows and per-layer context K/V rows are packed by their ORIGINAL segment's arithmetic class. ≥ 2-row segments go into shared overwrite-WMMA chunks of 2..=63 rows, per-row identical to the one-tile route. 1-row segments go into one `gemv_mq4g256v2_multirow_r2_lanes` launch, the R2 GEMV body per z-plane. One norm and one RoPE run over the packed rows, then one gather/scatter. This replaces the per-lane rotate/GEMM/norm/RoPE launches. Every ring and state byte is identical; otherwise (or with `0`) the per-lane loops run. |
| `HIPFIRE_DN_SNAPSHOT_BULK_OFF` | **unset**; `1` opts out | DeltaNet snapshot save/restore as one descriptor-driven byte-copy launch each (`dflash_state_bulk_copy_gfx1100`, pure byte copy) instead of one `hipMemcpy` per tensor. Default on exact gfx1100, exact gfx1151 and exact gfx1201 (railgun E0 port); DFlash and MTP share the snapshot. `1` restores the memcpy loops on all three. |
| `HIPFIRE_HIDDEN_SCATTER_FUSE_OFF` | **unset**; `1` opts out | DFlash hidden-ring scatter as one kernel launch each (`dflash_hidden_commit5_gfx1100` for the staging-to-ring commit, `dflash_hidden_scatter5_gfx1100` for the interleaved target-hidden scatter; five F32 extraction layers, pure F32 byte copies with one writer per destination element) instead of the per-row `memcpy_dtod_at` loops. Default on exact gfx1100, exact gfx1151 and exact gfx1201; never captured or recorded (the kernels bake the current ring head). On exact gfx1151 a finite-window scatter with more rows than `dst_modulus` (overlapping destination rows) keeps the sequential loop to preserve byte parity; gfx1100 and gfx1201 are unchanged. `1` restores the row-copy loops on all three. |
| `HIPFIRE_CB_TAPE_COPY_BATCH` | **on**; `0` opts out (developer) | VMM continuous-batching DFlash verify trunk (`forward_prefill_batch_multi`): each DeltaNet layer's GDN rollback-tape capture (pre-conv1d qkv rows + post-sigmoid α/β rows of every lane) runs as one `tape_copy_batched` launch (descriptor table in the kernarg segment, pure byte copy) after the layer's per-lane recurrence, instead of three `hipMemcpyDtoD` per lane per layer (1,152 → 48 launches per C8 step on the 27B). Tape bytes identical by construction. `0` restores the per-lane copies. |
| `HIPFIRE_CB_LANE_GDN_BATCH` | **on**; `0` opts out (developer) | Exact gfx1201 VMM continuous-batching DFlash verify trunk: when every request of a chunk is a ChainVerify lane with a rollback tape and the same row count (at most 16 lanes, fast Q8 GDN kernel, head dim 128, no capture/recording), each DeltaNet layer runs one launch per stage for all lanes instead of four per lane: the row-local sigmoid/alpha gate and Q/K norm-interleave over every row, `conv1d_silu_split_f32_lanes` (each lane's conv body verbatim over its rows and ring) and the pointer-table recurrence `gated_delta_net_q8_fast_ml` (single-lane body per lane, frames `base + lane * n`). 1,536 → 192 launches per C8 step on the 27B; every output and state byte identical. Needs `HIPFIRE_CB_TAPE_COPY_BATCH` on. `0` restores the per-lane launches. |
| `HIPFIRE_CB_LANE_GDN_VARROWS` | **on**; `0` opts out (developer) | Widens the lane-batched DeltaNet verify (`HIPFIRE_CB_LANE_GDN_BATCH`) and the save-free rollback (`HIPFIRE_CB_DN_SAVE_FREE`) to chunks whose lanes have different row counts, each 1..=16 (2..=16 for save-free). This is the clipped verify window near a request's end. The recurrence table carries each lane's `n_tokens` and stochastic-rounding `frame_off` (prefix sum), so each lane gets the same frames as the per-lane launch order. The host reserves the chunk's total rows once. Byte-identical to the per-lane route. `0` restores the equal-row-count requirement in both admissions. |
| `HIPFIRE_VERIFY_TAPE_COPY_BATCH` | **on**; `0` opts out (developer) | Singleton DFlash chain verify (sequential, non-tree, HIP-graph captured or eager): each DeltaNet layer's GDN rollback-tape capture (pre-conv1d qkv rows + post-sigmoid α/β rows) runs as one `tape_copy_batched` launch with the same three descriptors the continuous-batching route queues, at the original copy point, instead of three `hipMemcpyDtoD`. 27B B=16 verify graph: 144 copy nodes → 48 kernel nodes per cycle. Tape bytes identical by construction. `0` restores the three copies. |
| `HIPFIRE_CB_LANE_FA_BATCH` | **on**; `0` opts out (developer) | Exact gfx1201 VMM continuous-batching DFlash verify trunk: when every lane of a chunk is a ChainVerify lane on the fp8 KV VerifyAttn route (≤ 16 lanes, ≤ 32 rows each, no segment twin / fused prep / multi-row arm, no capture or recording), each FullAttention layer runs the row-local Q deinterleave+norm, K norm and RoPE once over all rows, then one fp8 K and one V cache write (`kv_cache_write_fp8_e4m3_lanes`), one `attention_verify_gqa_fp8_lanes_gfx1201` (the `va_verify_gqa` template per lane, grid.z = lane) and one `attention_verify_reduce_lanes_gfx1201` (verbatim reduce body per lane) instead of seven launches per lane. Lane partials live in a dedicated, grown-on-demand scratch (≤ 1 GiB, else the per-lane route). Byte-identical outputs and KV rows. `0` restores the per-lane launches. |
| `HIPFIRE_CB_DN_SAVE_FREE` | **on**; `0` opts out (developer) | Exact gfx1201 VMM continuous-batching DFlash verify: the per-lane DeltaNet SAVE before verify (≈ 120 MB of S / scales / EF / conv state per lane per step on the 27B) is dropped. Each lane's live and snapshot buffer owners are exchanged on the host, so the old live state becomes the snapshot. The verify then reads it as its source and writes the new live bank: the conv ring is byte-copied A → B inside the layer's existing `tape_copy_batched` launch and runs the unchanged in-place `conv1d_silu_split_f32_lanes`; the recurrence is `gated_delta_net_q8_fast_lanes_varrows_from`. Commit and terminal repair are unchanged: the D8 replay reads the snapshot. The snapshot-source tables are cached in two slots, so the alternating banks re-upload nothing. Admitted only when every lane of the chunk qualifies: the lane-batched GDN admission, each lane's row count 2..=16 (equal across lanes unless `HIPFIRE_CB_LANE_GDN_VARROWS`), D8 replay admitted at that lane's count, Q8 + EF state, equal-size HIP-owned banks at least the logical extent, pairwise distinct owners, no Redline/replay graph. Otherwise the save runs as before. Every output and state byte is identical. `0` restores the save. |
| `HIPFIRE_CB_DN_SAVE_FREE_DIAG` | **off**; `1` opts in (developer) | Investigative save-free DeltaNet admission diagnostic, captured once per process: emits one stderr line per packed chunk naming its first failing predicate, lane row counts/temperatures and reason-specific details (replay readiness, bank extents/ownership or aliases). Does not change admission; off performs no diagnostic formatting, allocation or extra family walk. The save-free route admits equal-sized pool-rounded EF banks at least as large as the logical extent (like scales/conv); S remains exact-sized. `HIPFIRE_CB_DN_SAVE_FREE=0` remains the save-free kill switch, independent of this diagnostic. Never enable diagnostic output for timing runs. |
| `HIPFIRE_DFLASH_SPARSE_TARGET` | **on**; `0` opts out (developer) | Sampled DFlash (temperature > 0), singleton and continuous-batching lanes: the FAST_SAMPLE target rows and the selector q rows are held sparse instead of as `[rows × vocab]` f32 buffers. The device compacts each verify row's kept set `p >= tau` (`sampled_rows_compact_f32`, per-row `tau`, ascending ids, 256 per row) so only counts, ids and values come back; rows with `tau <= 0` or more than 256 kept entries download their dense row. The selector q keeps its top-K support as (id, value) pairs with the dense row's last-write rule. The host accept loop, residual, bonus draw and degenerate argmax then walk the ascending union of the supports, performing the same f32 operations in the same order as the dense loop, and the PLD point-mass q is one sparse entry. CACTUS stays on the dense arithmetic. Identical decisions and RNG state: 10k host cases each for accept, residual / categorical and point-mass residual. `0` restores the dense `[rows × vocab]` download and dense q rows. |
| `HIPFIRE_DFLASH_DEVICE_ACCEPT` | **on**; `0` opts out (developer) | Sampled DFlash singleton and continuous-batching lanes (temperature > 0, FAST_SAMPLE on, sparse DFlash2 selector or PLD q): the accept / residual / bonus decision runs in ONE device kernel (`dflash_sampled_accept_sparse_q_f32`) over resident `softmax_temp_topp_batched_f32` rows (`probs`, `tau`, `Z`), with reusable session scratch (`VerifyScratch.device_accept`, grown on demand) and only a 16-byte decision readback instead of target-row D2H / compaction. The host uploads proposed tokens, last-write sparse q, `max_accept` and exactly `b` uniforms per window (one per proposal plus one residual-or-bonus draw). The normalized support is `p >= tau` (full row when `tau <= 0`, including identity / overflow rows in place); accept tests use `u * max(q, FLT_MIN) <= p_t`, with first rejection bounded by `max_accept` and bonus from the boundary row. Residual / bonus sampling uses ascending ids and a thread-count-independent fixed 256-segment CDF order with no atomics; CDF fall-off returns the last support id. This deliberately differs from the host loop's data-dependent RNG draws and f32 CDF association. CB lanes decide in original lane order during verify, from the packed exact head's lane view or their private singleton Product-b head, then consume that decision at commit without another RNG draw; these lanes are excluded from host target-row batching. The same seed uses the same sampled law as C1. Greedy, CACTUS, dense q (`HIPFIRE_DFLASH_SPARSE_TARGET=0`), `HIPFIRE_DFLASH_FAST_SAMPLE=0`, non-identical head sources and the non-selector C8 `chain_accept_spec_f32` branch retain their existing paths. `0` restores host acceptance for singleton and CB lanes. Production kernel: the certified builder (PeaceMaker) module `kernels/dflash_sampled_accept_pm_gfx1201.hxaco` (`dflash_sampled_accept_sparse_q_f32_pm_gfx1201`, emitter `crates/peacemaker-isa/src/kernels/dflash_accept.rs`, contract `crates/peacemaker-isa/kernels/dflash_sampled_accept.gfx1201.sparse_q.contract.json`, M7 obligation-free), **gfx1201 only**: other architectures keep the host accept loop (there is no hipcc production fallback). The hipcc kernel `kernels/src/dflash_sampled_accept_sparse_q.hip` is retained as a non-production oracle (`Gpu::dflash_sampled_accept_sparse_q_f32_hip_oracle`). Needs `n >= 1` and `(n + 1) * vocab * 4 < 2^32` bytes of probs. |
| `HIPFIRE_CB_SAMPLED_PACKED_HEAD` | **on**; `0` opts out (developer) | Exact gfx1201 VMM continuous-batching DFlash verify, chunks with at least one sampled (temperature > 0) lane: the chunk's rows share ONE explicit exact F16 head (`DenseBatchMath::SingletonWmma`, the `gemm_mq4g256v2_lmhead_verify_exact` family) and one argmax download instead of each sampled lane running a private singleton-b Product head. The exact head reproduces every lane's singleton Product-b logits at any packed row count (Product dispatch is not row-count independent above 63 rows); each lane's target rows are still normalized in the original lane order through `dflash_target_rows_into`. Admitted when the machine's normal F16 head chain is on (`mq4_verify_singleton_chain`), the head is MQ4G256V2 with `k = dim` and `m = vocab`, 1..=128 rows fit the shared head scratch, every lane verifies 2..=16 rows, and there is no FP8 / IU4 verify scope, A8 route at a lane's row count, capture, recording or G0 observation. Otherwise each sampled lane keeps its private Product-b head. `0` restores the private heads. |
| `HIPFIRE_CB_SAMPLED_TARGET_BATCH` | **on**; `0` opts out (developer) | Continuous-batching DFlash verify with the packed sampled head (`HIPFIRE_CB_SAMPLED_PACKED_HEAD`) and sparse target rows on: each run of consecutive sampled lanes with bitwise-equal sampling configs builds its target rows with ONE `softmax_temp_topp_batched_f32` and ONE `sampled_rows_compact_f32` over all the run's rows of the packed logits, plus one download each of tau / Z / counts / ids / values. The scratch is persistent and grown on demand. Both kernels compute each row in its own block from its own row only, so every lane's rows are bit-identical to its own per-lane call. Cactus lanes, unpacked heads, the singleton and the dense path keep the per-lane call. `0` restores the per-lane calls. |
| `HIPFIRE_CB_SAMPLED_DRAFT_BATCH` | **on**; `0` opts out (developer) | Exact gfx1201 VMM continuous-batching sampled DFlash2 draft: contiguous runs of eligible sampled lanes share one draft forward and target head, while each lane keeps its selector, RNG and q receipt in original traversal order. Also requires the `HIPFIRE_CB_DFLASH_DRAFT_BATCH` master switch (on unless `0`) and at least `max(draft_min_lanes, 2)` lanes. Every lane has positive temperature, `3 <= b <= block_size`, a selector and receipt, shared draft weights/config/layout, and normal F16 overwrite admission for every MQ4G256V2 draft/head weight at the packed row count and each lane's original `b` and `b - 1`; no FP8/IU4/A8/LDS, capture, recording or G0 route. Ineligible sampled lanes and greedy lanes end a run; refused or short runs keep the singleton selector draft. `0` restores singleton sampled drafts without disabling greedy batching. |
| `HIPFIRE_CB_UPLOAD_COALESCE` | **on**; `0` opts out (developer) | VMM continuous-batching serve step: fewer synchronous host→device uploads per step. Each blocks the host until the device drains. (1) The per-KV-layer slot-descriptor tables (`descs_dev`) are uploaded only when their bytes differ from what the device already holds: a host mirror of the last successful upload, so 16 → 0 uploads per steady step on the 27B, re-uploading only on mapping/owner changes. (2) The multi-request verify trunk uploads every lane's RoPE/KV positions in one copy instead of one per lane (8 → 1 at C8). Identical bytes at identical addresses before every reader. `0` restores the per-step and per-lane uploads. |
| `HIPFIRE_GDN_REPLAY_ML_OFF` | **unset**; `1` opts out | Exact gfx1201: the DFlash/MTP GDN tape replay runs every LinearAttention layer in two launches (`dflash_gdn_replay_pre_ml` + `gated_delta_net_q8_fast_ml`) instead of 4 launches per layer. Q8 state with the fast (single-end requant) kernel only; declines while a Redline recording is open. The replay tables (and the `HIPFIRE_DN_SNAPSHOT_FLIP` snapshot-source tables) are armed when the tape is built, so an allocation failure refuses the lane or request at provision instead of changing routes later; an admitted replay never falls back to the per-layer launches. `1` is an explicit diagnostic opt-in to the per-layer launches, which are **not byte-identical** to the default route on the production model (different committed DeltaNet scales / EF residual bytes). |
| `HIPFIRE_SELECT_REGRID_OFF` | **unset**; `1` opts out | Exact gfx1201: `topk_values_batched_f32` (DFlash2 selector) and `argmax_f32_batched` run 1024-thread rows with wave32 reductions (`select_regrid.hip`, byte-identical including ties; tie rows rerun the shipping top-K body). `1` restores the shipping one-block-per-row kernels. |
| `HIPFIRE_VERIFY_ATTN` | **on** on exact gfx1201, gfx1100 and gfx1151 (`kernel.verify_attn`); `0` opts out | VerifyAttn: speculative-verify attention (DFlash / MTP / n-gram verify blocks of 1..=32 rows, non-tree, H2 head_dim 256 / GQA 6) runs one K/V scan per kv head for 8 rows x 6 q heads over a fixed, context-independent split grid (graph-capture-stable) instead of `attention_flash_{fp8_e4m3,q8_0}_tile_batched`'s per-(row, head, tile) grid. On gfx1100 (Q8 KV) it also replaces the eager multi-row `attention_flash_q8_0_rows{8,4}_d8` above `HIPFIRE_FA_PERTOKEN_MIN_CTX` with a twin of its online-softmax arithmetic. On gfx1151 (Q8 KV) it replaces the single-slot WMMA flash prefill (`attention_q8_0_flash_prefill_wmma`, one wave per (q head, 16 rows) walking the whole context) with a context-parallel S = Q.K^T launch (f16 S tiles in the flash partials, one K dequant per kv head) and a per-16-dim-chunk online-softmax + P.V walk (one V dequant per kv head), also over context-independent grids. Byte-identical output (and gfx1201/gfx1100 partials); `0` restores the tile_batched / multi-row + reduce pairs and the WMMA prefill. |
| `HIPFIRE_DN_SNAPSHOT_FLIP` | **on** (exact gfx1201); `0` opts out | Railgun D8, exact gfx1201 DFlash chain spec: the rollback drops the DeltaNet restore copy. On a full accept with the EF residual on and a HIP/HipGraph verify, the verify-advanced state is kept; on any shorter accept the GDN tape replay reads the pre-verify state from the snapshot and writes the live buffers (`dflash_gdn_replay_pre_ml_from` + `gated_delta_net_q8_fast_ml_from`). Byte-identical to restore + replay; the snapshot stays the pre-window state, so terminal repair is unchanged. Needs the two-launch replay (`HIPFIRE_GDN_REPLAY_ML_OFF` unset); the no-tape path and every other decline restore as before. Introduced opt-in in 0.4.0, default on since 0.4.1; `0` restores the copy. |
| `HIPFIRE_FN_GDN_DENSE_SCAN` | **on** (exact gfx1201 FN prefill); `0` opts out | Flash-Next's admitted eager prefill uses the fused dense C64 GDN scan (`fn_gdn_dense`, `gdn_chunk_kkt_solve_batched` and `gdn_chunk_scan_bf16_mseg`) instead of the pipe128 recurrence. This route is KLD-gated, **not bit-exact** versus pipe128. It uses RNE exit Q8 quantization with no error-feedback (EF) residual. Short tails, decode, verify/capture and other declined calls keep the incumbent route; `0` restores pipe128 prefill. |
| `HIPFIRE_DRAFT_MAX` | routes to active mech window | CLI |
| `HIPFIRE_DRAFT_F16` | on unless `0` | RuntimeConfig |
| `HIPFIRE_NGRAM_DRAFT` | `1` forces n-gram | Loader always honors |
| `HIPFIRE_NGRAM_DRAFT_K` / `HIPFIRE_NGRAM_MIN_COUNT` | n-gram params | |
| `HIPFIRE_NGRAM_LOOP_THRESHOLD` | default **0 (off)** | RuntimeConfig |
| `HIPFIRE_NGRAM_WINDOW` | default 256 | RuntimeConfig |
| `HIPFIRE_MTP_MODE` / `HIPFIRE_MTP_K` | auto / auto (`mtp_k`: 7 for native Qwen4 MTP on gfx1151, else 3; an explicit 1–10 overrides, `4` restores the previous gfx1151 cap) | Config + RuntimeConfig. On gfx1151 the adaptive chooser picks each window's depth up to the cap. Qwen35 MoE native MTP `auto` declines to AR on a resolved host-mapped routed-expert spill under either `pcie` or `cpu`; `on` / `--spec mtp` opts in. Uniform MQ6G256 V1 (qt15) Qwen35 MoE on gfx1200/gfx1201 with FP16 WMMA enabled is refused before serving/generation whenever an MTP head would attach, including resident `auto`. This is metadata-based, not filename-based. `--spec off` / `speculation.mtp=off` and auto-declined spill remain AR; `HIPFIRE_FP16=0` permits explicit MTP. Mixed/graded MQ4+MQ6, MQ6V2, gfx1151 and dense remain allowed. Other resident attachment is unchanged; no root WMMA fix is claimed. |
| `HIPFIRE_MTP_NGRAM` | **auto** (on for native Qwen4 MTP on gfx1151 / gfx1201); `off`/`0` opts out | `speculation.mtp_ngram` (`on`/`off`/`auto`, also `1`/`0`; env beats TOML beats default): MTP + ngram-mod takeover windows. `auto` arms native Qwen4 MTP (arch 16: Flash-Next) on gfx1151 and gfx1201 and leaves every other producer off; `on`/`1` arms every MTP producer that supports it; `off`/`0` is the kill switch. Eligible requests are greedy **and** sampled, thinking on or off (Qwen4 and Qwen3.5/3.x). A sampled takeover verifies each n-gram candidate with the point-mass draft distribution δ(candidate) through the shared rejection verify, with the request's penalties applied to every candidate row as to native drafts: lossless in distribution, not bit-identical to MTP-only output (greedy takeovers are exact per window). The pool is request-local: cleared and reseeded from the full prompt every request, so a request's candidates never depend on earlier requests. On Qwen4 a pool hit takes the window only when the head's own first draft (row-0 argmax, before any sampled draw) equals the first candidate and the takeover's decayed emitted tokens per window over its verify cost are at least the native chooser's best expected rate; otherwise the window is native. Qwen4 takeover verifies `[seed, candidates..]` on the same batched verify executor as native MTP windows (the interleaved one-row route under `HIPFIRE_MTP_INCREMENTAL=1`), rolls the target back to the consumed rows, then appends exactly those rows to the MTP head teacher-forced (token p, target hidden p). With `HIPFIRE_MTP_NGRAM_COPY=off` (the default) the Qwen4 source is beta's 5/3/3 `MtpNgramContext` pool exactly: the head's row-0 confirmation runs on every takeover, the yield gate and the pool's low-acceptance clear stay, a takeover ends with the AR floor retiring the request, and load admission is sized by `max_k`. With `HIPFIRE_MTP_NGRAM_COPY=on` (the copy lane, off by default) the Qwen4 source is instead a verbatim-copy matcher (`MtpCopyContext`: keys of 32, 16, 8 and `n_match` tokens, longest first, each hash → the position past the key's most recent occurrence, the match checked token for token, the proposal copying the history from there; tables never cleared, a stale slot never proposes); on a CPU proxy of the agentic code-edit chain (`release-0.4.3/ngram-copy-research`, a proxy, not a hardware measurement) conditional per-depth acceptance was 0.991–0.997 against 0.90–0.96 for the earlier 5-token chain. It is not cleared on low acceptance (the take length backs off instead). On Qwen4 each window with a pool proposal takes `d` candidates, `d` in `n_min..=min(proposal, n_max, row-exact verify rows − 1 = 7, remaining budget)`, the `d` maximizing expected emitted tokens per cost from this request's own shadow-measured per-depth n-gram acceptance and measured takeover window costs, when that rate is at least the native chooser's best; otherwise the window is native. The head's own first draft (row-0 argmax, before any sampled draw) must equal the first candidate before the takeover runs, except for a take of at least 5 candidates once the request's depth-1 n-gram acceptance is at least 0.9 over at least 4 windows of evidence and the last window's first candidate was accepted (constants checked against agentic-chain replays; a divergence costs at most one blind window). A request the floor retired keeps the copy lane: only the AR calibration bars the pool, a take must beat AR itself (1.0 token per AR step), the MTP head is caught up (the rows the retired AR tokens owe, appended teacher-forced) before each takeover, and the floor keeps pricing takeovers and disarms the lane only when they cost at least AR per token; the request stays retired. Load admission sizes the verify rows and GDN row capture for the take cap plus the seed row where the n-gram source arms. Load prints `qwen4 native MTP n-gram: on\|off (n_match=.. n_min=.. n_max=..)`, plus `qwen4 native MTP n-gram copy lane: on (take_cap=N)` only when the n-gram source is on and the copy lane is armed; `HIPFIRE_MTP_TRACE=1` window events carry `source` (`ngram`\|`mtp`) and `verify_route`, and with the copy lane on also `ngram_plan` (proposal length, take, confirm, takeover and native rate, depth-1 acceptance, planning time, `retired`) and `ngram_confirm` (the row-0 comparison) events. |
| `HIPFIRE_MTP_NGRAM_COPY` | **auto** (off on every GPU for now); `on`/`1` arms the copy lane | `speculation.mtp_ngram_copy` (`on`/`off`/`auto`, also `1`/`0`; env beats TOML beats default): the native Qwen4 MTP n-gram copy lane (verbatim-copy matcher `MtpCopyContext`, triple 5/1/63, adaptive take of up to the row-exact verify rows minus one, row-0 confirmation only until the copy source is established, shadow-measured acceptance, kept on floor-retired requests). `on`/`1` arms it wherever the native Qwen4 n-gram source is armed (`HIPFIRE_MTP_NGRAM`); `off`/`0` and `auto` (off on every GPU until a GPU's own receipts flip it) keep beta's 5/3/3 `MtpNgramContext` pool with row-0 confirmation on every takeover, the yield gate, the low-acceptance pool clear and takeovers ending with floor retirement. |
| `HIPFIRE_NGRAM_MOD_N_MATCH` / `HIPFIRE_NGRAM_MOD_N_MIN` / `HIPFIRE_NGRAM_MOD_N_MAX` | **5 / 3 / 3** on admitted Qwen4 targets (the 5/3/3 pool, copy lane off); **5 / 1 / 63** with `HIPFIRE_MTP_NGRAM_COPY=on`; **24 / 48 / 64** elsewhere | Developer overrides for the ngram-mod triple (match length / min draft / max draft), applied per component over the per-arch default. With the copy lane off, `n_max` on Qwen4 is capped at 63 (the 64-row verify holds the seed plus up to 63 candidates). With the copy lane on, `n_match` is the shortest key of the copy matcher (the longer keys are 8, 16 and 32), `n_max` is capped at 63 (the 64-row spec verify capacity holds the seed plus up to 63 candidates) and is the proposal and observation length, not the take cap: a takeover takes at most `min(n_max, 7)` candidates, 7 being one below the rows a verify forward runs bitwise like the single-row decode (the per-GPU row-exact verify rows, `rdna_compute::prefill_invariant::qwen4_row_exact_verify_rows`, 8 on gfx1151 and gfx1201 today; a wider forward runs prefill kernels). |
| `HIPFIRE_MTP_SAMPLED` | **on**; `0` opts out | `speculation.mtp_sampled` (stable; introduced as an experimental opt-in and flipped to default on in 0.4.1): Qwen4 (Flash-Next) native MTP also serves sampled (temperature > 0) requests by speculative rejection sampling — a draft drawn from `q` (the request's truncation applied to the draft head's 8 exactly re-scored candidates) is accepted with probability `min(1, p/q)`, a rejection emits a draw from `(p − q)+`, an all-accepted window emits its bonus from `p`, where `p` is exactly the AR sampler's distribution (`llama::sample_top_k_p`: temperature, top_k with absent = 20 and 0 = 64 candidates, min_p, top_p). Lossless against AR in distribution, not byte-identical to AR for a seed; one seed replays the same text. Greedy requests are unchanged. `0`: sampled requests run AR. Repeat/presence/frequency penalties stay on MTP (0.4.1): every target row — the first token, each verify row and the bonus — is penalized exactly as the host AR sampler penalizes it, against the trailing `repeat_window` tokens of the full prompt + emitted tokens + the window's earlier drafts, before truncation; the draft `q` is penalized the same way. Temperature-0 requests with non-neutral penalties run the Qwen4 AR producer (penalize-then-argmax). |
| `HIPFIRE_MTP_OWN_PREFILL` | **unset / off**; `1` opts out | Qwen35 MTP prompt fill. Default: the trunk prefills the prompt through AR's own route (same outer chunks, widened chunk, GDN chunk scan, standard dispatch) and hands its hidden rows to the MTP head, so the prompt's KV, DeltaNet state and first-token logits match AR's. `1` restores MTP's previous route: 512-row trunk chunks captured as a speculative verify (sequential GDN recurrence; on Q8 KV, no query16 flash prefill). Resolved once per MTP load (`hipfire_config::mtp_own_prefill`); serve.log prints `qwen35 MTP prompt fill route: …`. |
| `HIPFIRE_MTP_FILL_FAST` | **on**; `0` opts out | Qwen35 MTP prompt fill only (native decode and the MTP auto sidecar/AR prompt route are unchanged). Default: the head's KV-only Append writes K/V with two same-stream `hipMemcpy2DAsync` pitched D2D copies instead of 2n synchronous per-row concat memcpys, omits the unused Q branch (including its residual), runs the identical K-only RoPE kernel with `n_heads_q = 0`, and reuses the same FWHT-rotated activation for eligible wk/wv (MQ4G256, no AWQ). Fill preparation zeroes the head's KV buffers on device (ordered `hipMemsetD32`) instead of host-side zero vectors with H2D uploads; a cold or missing predecessor's zero also uses `hipMemsetD32`, and the valid prefix carry-copy is unchanged. On eligible gfx1201 Q8-KV heads it uses lean fill-only scratch (borrowing inactive fields) and larger head slices (default 512 → 1,688 rows; 384 → 1,264), capped so scratch does not exceed the previous full scratch plus rotation bytes. The separate `HIPFIRE_MTP_FILL_PM_GEMM` switch controls the exact head GEMM replacement. `0` restores the prior qwen35 MTP prompt fill. |
| `HIPFIRE_MTP_FILL_PM_GEMM` | **on**; `0` opts out | Qwen35 MTP fast prompt fill only, exact gfx1201 and MQ4G256-v1 eh_proj/wk/wv (including shared-rotation wv). Uses the byte-identical PeaceMaker four-weight-row GEMM, sharing each activation load across rows. Requires `HIPFIRE_MTP_FILL_FAST`; other dtypes, architectures and callers retain the incumbent. `0` restores `gemm_hfq4g256` without changing fill slicing or rotation. Resolved once per process through the developer environment reader. |
| `HIPFIRE_QWEN35_MTP` / `HIPFIRE_QWEN35_MTP_K` | Retired; no runtime readers | These former Qwen35-specific controls no longer affect loading. Use unified `speculation.mtp` (`auto` by default): the loader uses a bundled MTP head first, then a `.mtp` sidecar. Use `speculation.mtp_k` for the draft cap; see [Speculative decode](CONFIG.md#speculative-decode). |
| `HIPFIRE_DEEPSEEK4_SPEC_DECODE` / `HIPFIRE_DEEPSEEK4_SPEC_K` | DeepSeek MTP legacy | |
| `HIPFIRE_DEEPSEEK4_DSPARK` / `HIPFIRE_DEEPSEEK4_DSPARK_CONF_THRESHOLD` | DSpark | |
| `HIPFIRE_QWEN3_DSPARK_CONF_THRESHOLD` / `HIPFIRE_QWEN35_DSPARK_CONF_THRESHOLD` | per-arch conf | |
| `HIPFIRE_DDTREE_BUDGET` / `HIPFIRE_DDTREE_TOPK` | tree draft | Runtime defaults 256/8 if env-only; CLI config defaults 0/4 |
| `HIPFIRE_DDTREE_*` | research/diag family | See inventory; not product defaults |

### Qwen4 / Qwen3.8 Flash-Next (arch 16)

Read only by the Qwen4 carrier and its kernels; no other model reads them.

| Variable | Default / sense | Notes |
|---|---|---|
| `HIPFIRE_QWEN4_WIDE_TWIN_MQ4` | **off** (developer); `1` opts in | Exact-wide MQ4G256V2 x4 GEMV candidate on gfx1151/gfx1201: admitted actual rows 9..16 select `_wide_r16`, 17..32 `_wide_r32`. Unset/`0` and unsupported shapes retain S1's exact <=8-row launch groups. Does not raise `Gpu::few_row_limit()` or admit prefill. Default stays off pending architecture-local byte/state/resource/timing gates. |
| `HIPFIRE_QWEN4_WIDE_TWIN_MQ6` | **off** (developer); `1` opts in | Same admission and fallback as `HIPFIRE_QWEN4_WIDE_TWIN_MQ4`, for MQ6G256V2 single/x4 exact-wide GEMVs; independent of the prefill MQ6 WMMA/PM switches. |
| `HIPFIRE_QWEN4_WIDE_TWIN_BF16` | **off** (developer); `1` opts in | Same admission and fallback, for ordinary BF16 x4 and HC k4 wide candidates. Unsupported scalar-domain shapes retain exact row tiling. HC quarter folds/activation and the existing <=8-row entries are unchanged. Tiny HC gates may retain the exact tiled specialization. |
| `HIPFIRE_QWEN4_WIDE_TWIN_Q8` | **off** (developer); `1` opts in | Same admission and fallback, for Q8 K2560 single/x2, K320, small-K four-chain and HC k8 wide candidates. Uses the existing decode copies and each shape's decode arithmetic recipe; does not create copies or reinterpret BF16 router/selector weights as Q8. |
| `HIPFIRE_QWEN4_WIDE_TWIN_K1` | **off** (developer); `1` opts in | Sealed-slab K1 lane-union candidates on gfx1151/gfx1201: admitted lanes 9..16 select `_c16`, 17..32 `_c32`; <=8 keep narrow entries. Otherwise S1 retains exact <=8-lane groups. Descriptor capacity 32 reserves 7688 bytes before capture; the 512-thread planner preserves first-occurrence order. Capacity is not admission: the shared per-architecture `Gpu::few_row_limit()` still controls routing. The twins exist only for the shared MQ4/W3 K1 images: layers with an E8P cold bucket (gfx1201) and gfx1100 keep the S1 exact <=8-lane groups. |
| `HIPFIRE_QWEN4_F16_WMMA` | on unless `0` | Prefill F16 WMMA arms (grouped MoE gate/up and down, BF16 dense projections through an F16 shadow, HC read, full-window QSA, chunked GDN) from 512 rows. Not bit-exact; admitted by KLD against the BF16 source. `0` keeps the bit-exact F32 arms. |
| `HIPFIRE_QWEN4_F16_WMMA_GFX1201` | gfx1201: on unless `0` | Exact gfx1201, eager-only: runs the `HIPFIRE_QWEN4_F16_WMMA` BF16-projection and HC-read arms on gfx12 WMMA kernels. The M=1 shared selector stays on exact multirow (U1a owns fusion). X rounds F32→F16 RNE; BF16 weights convert to F16 (exact for in-range normals); WMMA F32 16-element substeps and fixed-order split-K reduction change summation order, so it is not bit-exact against the multirow/SIMT arms. Admitted by Flash-Next KLD against the BF16 teacher on an R9700 (WikiText-2 −0.0033 [−0.0082, +0.0006], code +0.0047 [−0.0030, +0.0131] vs `=0`); pp8192 2,169 → 3,598 tok/s with `HIPFIRE_QWEN4_HC_FUSE=1` and `HIPFIRE_QWEN4_HC_UP_TILE`. `0` keeps gfx1201 on the multirow/SIMT arms; this is the kill switch. Recording/capture and other architectures retain their existing routes. Empty split-K work launches nothing. |
| `HIPFIRE_QWEN4_MOE_X_LDS` | `auto` (developer): exact gfx1151, on where the F16 WMMA MoE route runs; `1` forces, `0` kills | Values `auto`, `1` or `0`. Exact gfx1151 F16 WMMA routed-expert GEMMs: gate/up SiLU takes `gemm_mq4g256v2_moe_grouped_wmma_k2_silu_bf16out_x4` (M % 64 == 0, K % 256 == 0) and the BF16-output down GEMM `gemm_mq4g128v2_moe_grouped_wmma_gfx1151_x8_bf16out` (M % 128 == 0): four/eight row-tile waves share each expert run's X through LDS, output bytes equal to the per-wave entries'. `auto` (unset) arms exactly when a layer's MoE runs on the F16 WMMA route, i.e. its symmetric IU4 arm is not selected (`HIPFIRE_QWEN4_MOE_SYM_IU4=0`, or experts that fail the symmetry check such as the asymmetric `qwen3.8-flash-next.mq4`); the default IU4 route of the GPTQ3 artifact never reaches these GEMMs and is unchanged. `1` forces the entries wherever admissible (the same route and arch); `0` keeps the per-wave entries and launch geometry everywhere. F32 output, decode and other architectures are unchanged. These entries are HIP, not PeaceMaker-emitted. Measured on Strix Halo, F16 route (`qwen3.8-flash-next-gptq3.mq4` with `HIPFIRE_QWEN4_MOE_SYM_IU4=0`, `hipfire bench --matrix` synthetic stream, 3 fresh processes per arm): pp512 +6.6 %, pp1024 +7.4 %, pp8192 +9.4 % over the per-wave entries, decode unchanged. The gfx1151 byte oracle is `qwen4_moe_f16_lds_oracle --mode check` (build with `--features lab`). |
| `HIPFIRE_QWEN4_MOE_DOWN_TWIN` | on unless `0` (developer): exact gfx1151 | Exact gfx1151, Qwen4 grouped MoE down GEMM (M = 2560, K = 640, one X row per top-10 slot) on batches of >= 9 tokens below the F16 WMMA window (the >= 512-token WMMA arm is checked first): runs `gemm_mq4g128v2_moe_grouped_top10_o8_r16_xs4_gfx1151` (module `gemm_mq4g128v2_moe_grouped_top10_o8_r16_xs`; four waves, 32 rows per block, each group's X chunk staged in LDS once per block) instead of `gemm_mq4g128v2_moe_grouped_top10_o8_r16_gfx1151`. Bitwise the O8×R16 output: same per-output decode, accumulation order and wave32 reduction; only where X is loaded from changes. `0` restores O8×R16 for every row count. |
| `HIPFIRE_QWEN4_HC_UP_TWIN` | on unless `0` (developer): exact gfx1151 | Exact gfx1151, Qwen4 HC read tail `hyper_read_up_fused` on batches of >= 16 rows: runs `hyper_read_up_fused_f32_sx` (module `hyper_read_up_fused_twin`) instead of `hyper_read_up_fused_f32`. Same kernel arguments, grid, block and dynamic LDS; bitwise the incumbent's output: only the output stores move (held in registers until the last pass), so the wave-uniform X loads become scalar loads. Batches below 16 rows and every other architecture keep the incumbent. `0` restores `hyper_read_up_fused_f32` for every row count. |
| `HIPFIRE_QWEN4_MOE_GATEUP_TWIN` | on unless `0` (developer): exact gfx1151 | Exact gfx1151, Qwen4 grouped MoE gate/up GEMM (M = 1280, K = 2560, ten X rows per token) on batches of more than eight tokens below the F16 WMMA window (the >= 512-token WMMA arm is checked first): runs `gemm_mq4g256v2_moe_grouped_top10_o4_r8_pipe_gfx1151` (module `gemm_mq4g256v2_moe_grouped_top10_o4_r8_pipe`) instead of `gemm_mq4g256v2_moe_grouped_top10_o4_r8_x4`. Bitwise the O4×R8 X4 output: same per-output arithmetic; only the scheduling changes (fully unrolled ten-group walk, waves per block, prefetch depth). Batches of eight tokens or fewer keep the O4×R2 incumbent. `0` restores O4×R8 X4 for every row count. |
| `HIPFIRE_QWEN4_PROJ_REGIONS` | **opt-in**; off unless `1` | F16 WMMA projection route only: virtually concatenate router/shared gate/up/selector rows of K=2560 into one LDS launch; gfx11 HC-down (320×10240) can use it alone. Ascending 16-element K steps are unchanged. gfx1201 keeps its exact short selector and split-K=4 HC-down separate; this flag does not admit the F16 route itself. PLE, shared-down and HC-write dispatch are unchanged. Default off pending G0 and timing approval. |
| `HIPFIRE_QWEN4_SHARED_DOWN_EPI` | **opt-in**; off unless `1` | Fuses BF16 shared-down GEMM and row-scaled residual add for BF16 recipes on the existing F16 WMMA route (512+ rows, M >= 1024, eager-only). Rounds the value, row scalar, residual, product and sum independently to BF16, storing BF16-rounded F32; preserves the incumbent WMMA accumulation. gfx1201 also requires its F16 WMMA route (`HIPFIRE_QWEN4_F16_WMMA_GFX1201`, default on); it does not admit that route itself. Unset/`0`, recording/capture, other dtypes and non-BF16 recipes retain the separate kernels. |
| `HIPFIRE_QWEN4_MQ6_X4_GFX1201` | Qwen4 prefill chunks of >= 512 rows: on; set, on only for `1` | Unset, it applies only inside the Qwen4 (Flash-Next) forward, on prefill chunks of >= 512 rows (decode keeps the GEMV); `1` forces it for every MQ6G256V2 GEMM at every row count, any other value (`0`) disables it. Exact gfx1201, eager-only MQ6G256V2 trunk projections: BT8 four-row-tile X-LDS WMMA overwrite replaces memset + residual; shared rotation produces F16 directly, and chunked GDN qkv may store BF16 bits with RNE. Keeps ascending K tiles and the baseline F16 dequantization. Capture/retained recording and other architectures retain their existing routes. Bitwise-identical to the memset + residual route it replaces; `0` keeps that route. |
| `HIPFIRE_QWEN4_MQ6_X4_TILE` | Qwen4 forward on gfx1151: `auto` | Exact gfx1151 MQ6G256V2 X-LDS overwrite tile: unset, the Qwen4 (Flash-Next) forward uses the measured per-shape table in `docs/quant-formats/mq-v2-family.md`; `BVxRWxprefetch` (`BV` 8/12/16, `RW` 4/8, prefetch 1/2) forces one tile, and `auto` the table, for every MQ6G256V2 GEMM, including the a/b/z region fold and the HC-write epilogue launch (those two have BV8/BV16 twins only; BV12 keeps their incumbent; the table routes them only at their measured N=8192/2048 rows); `0` keeps the incumbent BT8 x4 kernels. Every tile keeps each output's ascending-K WMMA chain and epilogue expression, so the bytes are identical. Other arches ignore it. |
| `HIPFIRE_QWEN4_MQ6_X4_REGIONS` | Qwen4 forward on gfx1151: on; else off unless `1` | Exact; unset, on only inside the Qwen4 (Flash-Next) forward on gfx1151, `1` on gfx1151/gfx1201 everywhere: folds the chunked GDN a/b/z MQ6 F32 projections (sharing one prepared F16 rotation) into one row-region launch with independent accumulators; bytes identical to three launches. `0` keeps three launches. |
| `HIPFIRE_QWEN4_NPU_TRUNK` | `auto` (gfx1151 only; effectively `off` elsewhere) | Flash-Next family-B MQ6 input-projection trunk: `off|auto|on|emulate`, config `fusions.npu_trunk` (alias `npu_trunk`). Auto arms only with an open NPU and matching schema-4 BF16-direct calibration selecting a nonempty family set. The planner admits rebalanced pairs and row-adaptive 256-row-wave chains above the stored crossover; at most four distinct wave counts are built per load, and a pair needing a fifth runs GPU-native. There is no I32 serving contract; QSA and GDN out projections are not shipped on this route. Any auto load refusal logs one reason and stays GPU-only (`WARN` when a qualifying load fails to arm the NPU); an NPU fault at request time turns the `auto` route off for the rest of the process (one `WARN`) and the faulted request reruns GPU-only, while `on` fails the request closed (see [NPU trunk](npu/npu-trunk.md), Faults). The armed trunk pins 1,053,843,456 B of userptr arenas against `RLIMIT_MEMLOCK`; grant a bounded limit of 1,318,060,032 B (`LimitMEMLOCK=1318060032`, `ulimit -l 1287168`), see [host memlock limit](npu/npu-trunk.md#host-memlock-limit). `off` is the kill switch. Off and non-gfx1151 auto do no NPU/index/calibration I/O, allocations or logging. Forced on/emulate refuse other architectures; emulate is GPU-only. See [NPU trunk](npu/npu-trunk.md). |
| `HIPFIRE_QWEN4_NPU_TRUNK_CAL` | `<hipfire home>/npu-trunk/calibration.json` | Config `fusions.npu_trunk_calibration` (alias `npu_trunk_calibration`); empty uses the default. Schema 4 requires `epilogue = "bf16"` and has no `glue.fold_ms_per_col_8192` I32 field. A file takes precedence over compiled-in measured Halo overlay/stock calibration; only an absent file falls back. A file requires identical GPU PCI address/NPU firmware/build/uname/model-index/epilogue/trunk-route keys; the built-in record binds on arch/build/model-index/epilogue/trunk-route only; armed logs include `cal_source=file|builtin`. Invalid/unreadable files never fall back. Read only in effective auto/on/emulate. |
| `HIPFIRE_QWEN4_NPU_TRUNK_FAMILIES` | unset (developer) | Comma list of `qkv`, `zab`, `q`, `kvidx`, or `all` (names are the calibrated family groups: GDN `qkv`; GDN `z`+`a`+`b`; QSA `q`; QSA `k`+`v`+`idx`). Read once at load; forces the NPU trunk to run exactly that subset in every mode (`auto`, `on`, `emulate`), replacing the calibration-selected family set. Unset or empty keeps the mode's own selection (`auto`: calibration-selected; `on`/`emulate`: all families); `all` selects every family. Only effective when the trunk is armed (read like `HIPFIRE_QWEN4_NPU_TRUNK_TRACE`, never in effective `off`). |
| `HIPFIRE_QWEN4_NPU_TRUNK_FAULT_CHAIN` | unset (developer) | Developer fault injection: a 1-based relay chain number whose chain watchdog is forced to 1 ns, the real `NPU chain watchdog timed out` path, to exercise fault containment (`auto`: the route latches off, one `WARN npu-trunk: NPU pair fault ...` line, and the faulted request finishes GPU-only; `on`: the request fails closed). Counts every chain the relay runs, in relay order. Unset or empty injects nothing; a value that is not a number >= 1 (including `0`) logs one `npu-trunk: HIPFIRE_QWEN4_NPU_TRUNK_FAULT_CHAIN=... ignored` line and injects nothing. Only read when the NPU runtime opens (never in `off` or `emulate`). |
| `HIPFIRE_QWEN4_NPU_TRUNK_SAMPLE_LOG` | unset (developer) | `1` prints one `npu-trunk: pair-sample ...` line per executed pair under `auto`: the measured NPU chain time, stall-free GPU window, GPU wait on the NPU and pair time, next to what the request was planned with (`planned(genN)`: model NPU, window, wait, pair, GPU-only time and saving), the built-in prior and the scales after the sample. Reads only the always-on pair events; adds no synchronization. |
| `HIPFIRE_QWEN4_NPU_TRUNK_TRACE` | unset (developer) | Append per-(layer, bank) JSONL for successful physical pairs. GPU prepare/rotation/pack/wait/mixer events are resolved after the pair against a monotonic anchor; relay submit/done/terminal-signal timestamps use CLOCK_MONOTONIC. No mid-pipeline synchronization. Unset or empty writes nothing; ignored when the route is not armed. See [NPU trunk](npu/npu-trunk.md). |
| `HIPFIRE_QWEN4_QSA_WMMA_GATHER` | gfx1151, gfx1201: on unless `0` | Runs QSA prefill attention for chunks of >= 512 rows that the full-window dense route does not take (gfx1151: the past-budget chunks of the F32 state and every such chunk of the opt-in q8 state; gfx1201: every such chunk) on the gathered F16 WMMA kernels (`indexed_attention_gathered_wmma.gfx1151.hip` on the gfx1151 F32 and q8 states, `indexed_attention_gathered_wmma.gfx1201.hip` on the gfx1201 fp8 state) instead of `indexed_attention_attention_*_batched_hg4`. Each call first writes the layer's cache rows `[0, end)` as F16 into the route's own workspace (2 KiB per context token: 128 MiB at 64K, 512 MiB at 262K). The load reserves its address for the whole `max_seq` before any capture (logged as `qwen4 QSA gather workspace`). With VMM QSA state (`kv_backend` resolved to `vmm`, gfx1151 and gfx1201) each prefill forward maps only the prefix `[0, end)` it reads, and `HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS=auto` charges the first chunk's bytes; with legacy state the load commits all of it and the reserve charges all of it. Not bit-exact. Acceptance bar: its error against an f64 reference of the same quantized source is no worse than storing Q/K/V/P as BF16. On the worst-error 1536-row snapshot per arch (training-storage calibration) the gathered route measured max |err| 1.93e-3 (gfx1151) / 2.78e-3 (gfx1201) against 1.58e-2 / 1.68e-2 for BF16 storage, p99 9.8e-5 / 1.24e-4 against 1.01e-3 / 1.23e-3. The bar compares aggregate max and p99 on these two snapshots, not each element (93.4–93.5% of elements are within their BF16-storage error) and not a model-global bound. `0` keeps every launch of the incumbent route. Other arches and state formats ignore it; `HIPFIRE_QWEN4_F16_WMMA=0` also disables it. |
| `HIPFIRE_QWEN4_QSA_F16_ARENA` | **opt-in**; off unless `1` (gfx1151 F32 QSA K/V, head_dim 256 only) | BF-1, exact relative to today's gathered prefill attend: each QSA layer keeps a persistent F16 K and block-transposed V copy, converting only invalid/new rows (at most three valid rows reread for four-token alignment) with the existing producer before gathered WMMA attention. Resolved once at load; requires the gathered route enabled. Adds 2 KiB/token/layer for 2 KV heads, rounded to four-token capacity, charged in admission and state accounting. On unified-memory (UMA) archs the context admission charges the QSA growth including this arena, the resident PLE prewarm/load-fill budget and the existing headroom, so an explicit `max_seq` that no longer fits is refused at load under the default auto OOM guard (`card_cap` reports the largest fitting context with current free memory; the limit moves with free host memory and the PLE prewarm budget); a discrete GPU is unchanged. Both VMM demand mapping and legacy storage are supported. Rollback/truncation clamp the valid prefix; reset, restored/replaced prefixes and failed forwards invalidate it. Decode, MTP verify and the MTP head keep F32 K/V. `0` restores whole-prefix per-chunk conversion in the shared workspace. Hardware oracle: `qwen4_qsa_arena_check` (`reference-parity`), `QSA_ORACLE_COMPARE=1` runs fresh off/on processes; `QSA_ORACLE_KV_BACKEND=legacy` also exercises legacy storage. |
| `HIPFIRE_QWEN4_QSA_POOLED_BF16` | gfx1151: on unless `0` (exact gfx1151, F32 QSA state, index_dim 128, PM BF16 score module present, `HIPFIRE_QWEN4_QSA_SCORE_PM` not `0`; ignored elsewhere) | Developer switch, read once at load; `0` is the kill switch. Each QSA layer keeps a derived BF16 mirror of its pooled index keys (`[pooled block][128]`, the high 16 bits of each F32 key, exact because `qsa_pool_rope` stores BF16-rounded values) so the PM BF16 score kernel replaces the F32 one on mirror-covered select calls; selected indices, attention, state bytes and logits are unchanged. Module `qsa_pooled_bf16_mirror` converts only the blocks past a per-layer block watermark after pooling; the canonical F32 pooled arena stays authoritative and the mirror is never snapshotted. Rollback/truncation clamp the watermark to the retained complete blocks; reset, restored/replaced prefixes and failed forwards invalidate it; a select call whose mirror is not fully valid uses the F32 keys. Adds 256 B per pooled block per QSA layer (2 MiB/layer at 32K tokens, 16 MiB/layer at 256K, compress 4), charged in admission and state accounting. Direct launches only: under recording/capture the sync neither launches nor advances the watermark. Hardware oracle: `qwen4_qsa_arena_check` with the variable set must match the off arm. |
| `HIPFIRE_QWEN4_QSA_PM` | gfx1151, gfx1201: on unless `0` | Runs the gathered route's (`HIPFIRE_QWEN4_QSA_WMMA_GATHER`) F16 K/V producer and attention from the embedded certified builder (PeaceMaker) module (`kernels/qsa_gather_pm_gfx{1151,1201}.hxaco`) instead of the JIT-compiled hipcc kernels: same ABI, grid, block, LDS layout and output bytes. The module has no producer for the opt-in q8 QSA state: q8 always converts with the hipcc `indexed_attention_kv_f16vb_q8` and attends through the module. A cache whose F16 scratch or token stride would overflow the module's 32-bit buffer offsets (or `v_mad_u32_u24` operands) keeps the hipcc kernels. `0` keeps the hipcc kernels everywhere; with the gathered route off it does nothing. |
| `HIPFIRE_QWEN4_MQ6_X4_PM` | gfx1151, gfx1201: on unless `0` | Runs the Qwen4 MQ6G256V2 overwrite trunk GEMM from the embedded certified builder (PeaceMaker) module instead of the JIT-compiled hipcc kernel, with the same arguments and output bytes and a 256-token batch tile (the module's `n_tile`). gfx1201: the F32 overwrite (`HIPFIRE_QWEN4_MQ6_X4_GFX1201` route, non-BF16 output; `kernels/qwen4_mq6_x4_pm_gfx1201.hxaco`, entries `qwen4_mq6_x4_pm_gfx1201_w4`/`_w8`) replaces `gemm_mq6g256v2_wmma_gfx12_bt8_x4`; rows <= 64 use the 64-row 128-thread entry, larger M the 128-row 256-thread entry; BF16 output, the residual, regions and HC-write twins keep hipcc. gfx1151 is selective (`kernels/qwen4_mq6_x4_pm_gfx1151.hxaco`): only the plain F32 overwrite, the BF16-output overwrite and the a/b/z regions fold take the module's 128-row 256-thread entries (`qwen4_mq6_x4_pm_gfx1151_w8`, `_w8_bf16out`, `_w8_regions`, replacing the U3 twin or `gemm_mq6g256v2_wmma_gfx11_bt8_x4*`), and only for launches of >= 2560 rows (regions: summed rows) and >= 2048 tokens. The 64-row `_w4` entries, the HC-write (`hcw`) launch, smaller launches and the residual route keep U3 / hipcc. Every other architecture keeps hipcc. `0` keeps the hipcc kernels on both (same-binary control). |
| `HIPFIRE_QWEN4_QSA_SCORE_PM` | gfx1151: on unless `0` | Runs the live QSA selector pair's rows16 F32 score kernel from the certified builder (PeaceMaker) module `kernels/qsa_select_pm_gfx1151.hxaco` instead of hipcc. Exact gfx1151 only: at least one row (prefill, decode and verify calls alike; `HIPFIRE_QWEN4_QSA_SELECT_PM_DECODE=0` removes the calls below 512 rows), 4 index heads, dimension 128, <= 130816 pooled blocks, budget <= 512 and buffer extents within the module's 32-bit offsets. Selection is byte-identical. On BF16 index arenas (the opt-in q8 QSA state) the same calls run the BF16-pooled score kernel of `kernels/qsa_select_bf16_pm_gfx1151.hxaco` (contract `crates/peacemaker-isa/kernels/qsa_select.gfx1151.score_bf16.contract.json`) followed by the select kernel, in place of the fused hipcc `indexed_attention_select_bf16_batched`; `0` keeps the fused kernel for those arenas. Recorder/graph capture, unsupported extents and other architectures keep hipcc; `0` keeps the hipcc score kernel independently of the select flag. Defined in `crates/rdna-compute/src/tensor_ops.rs`; emitter `crates/peacemaker-isa/src/kernels/qsa_score.rip.rs`, aggregator `qsa_select.rs`, contract `crates/peacemaker-isa/kernels/qsa_select.gfx1151.score.contract.json`. |
| `HIPFIRE_QWEN4_QSA_SELECT_PM` | gfx1151: on unless `0` | Runs the live QSA selector pair's select-from-scores kernel from the same certified builder module `kernels/qsa_select_pm_gfx1151.hxaco`, with the same eligibility and byte-identical selection as `HIPFIRE_QWEN4_QSA_SCORE_PM`, including decode and verify calls unless `HIPFIRE_QWEN4_QSA_SELECT_PM_DECODE=0`. On BF16 index arenas it applies only while `HIPFIRE_QWEN4_QSA_SCORE_PM` is on; `0` there selects with the hipcc `indexed_attention_select_from_scores`. Recorder/graph capture and other architectures keep hipcc; `0` keeps the hipcc select kernel independently of the score flag. Defined in `crates/rdna-compute/src/tensor_ops.rs`; emitter `crates/peacemaker-isa/src/kernels/qsa_topk.rip.rs`, aggregator `qsa_select.rs`, contract `crates/peacemaker-isa/kernels/qsa_select.gfx1151.select.contract.json`. |
| `HIPFIRE_QWEN4_QSA_SELECT_PM_DECODE` | gfx1151: on unless `0` | Developer switch, read once, for the builder (PeaceMaker) QSA score/select pair of `HIPFIRE_QWEN4_QSA_SCORE_PM` and `HIPFIRE_QWEN4_QSA_SELECT_PM` on calls below 512 rows (decode and verify; admission uses the original or invariant-profile row count, whichever is larger). `0` disables the pair for those calls, which then keep the hipcc F32 score/select pair or, on BF16 arenas, the fused hipcc selector; calls of 512 or more rows are unaffected. Unset, the builder pair runs at every admitted row count (>= 1) on exact gfx1151 outside recording and graph capture while the PM score/select flags are on. Defined in `crates/rdna-compute/src/tensor_ops.rs`. |
| `HIPFIRE_QWEN4_QSA_SELECT_EXACT` | **opt-in**; off unless `1` | `1` runs the batched QSA selector (`indexed_attention_select_{f32,bf16}_batched_exact`, `indexed_attention_select_exact.hip`) for launches with <= 2048 complete pooled blocks and <= 512 budget blocks outside a recorder or graph capture: 256-element tile sorts plus a fixed-order top-512 merge replace the all-pairs ranks, with the incumbent's score association, score-descending / index-ascending order, `-1` fill, tails and mirror. Selected bytes are byte-identical to the incumbent. Every other launch (longer contexts, recording, capture, the serial kernel) keeps the incumbent selector; unset or `0` keeps it everywhere. |
| `HIPFIRE_QWEN4_QSA_BF16_SPLIT` | exact gfx1201: on unless `0`; every other architecture: not used | Exact gfx1201 only, BF16 pooled index keys (one row of four heads, dimension <= 128, budget <= 512 blocks). Replaces the fused `indexed_attention_select_bf16_batched` selector with the exact hipcc BF16 score kernels (`kernels/src/qsa_bf16_scores.hip`, rows8 up to 8 rows else rows16, scores bit-identical to the fused per-block score) followed by the exact sampled filter trio (`qsa_filter_select.hip`; select-from-scores never beat it, so `HIPFIRE_QWEN4_QSA_FILTER_SELECT=0` keeps the fused selector instead). Selected indices, causal tails and the final-row mirror are byte-identical to the fused selector. From 2048 pooled blocks for calls of at most 256 rows (decode, verify) and 8192 above (prefill), measured on an R9700 as graph-replayed GPU time per call: at 2048 blocks fused 64.6 / 64.3 / 67.6 us vs 40.1 / 43.6 / 65.8 us for 1 / 8 / 16 rows, at 65536 blocks (256K tokens) 552 / 546 / 545 us vs 313 / 263 / 211 us; below the crossover the fused selector stays. Safe for the retained decode tape: recorded, the crossover reads the pooling capacity (`shape_blocks`) instead of the live count, the score stride, score grid (`ceil(shape_blocks / 256)`) and filter compact grid (`ceil(shape_blocks / 4096)`) come from it, the block count and `position_start` of every launch are declared to the recorder, and the score/filter scratch lives in the dedicated selector scratch slot, which `Qwen4Bundle::attach_forward` reserves (about 80 MiB) before any capture. hipGraph capture, `HIPFIRE_QWEN4_QSA_SELECT_EXACT=1`, gfx1151 and every other architecture keep the fused selector. `0` is the kill switch and always keeps the fused selector; only `0` disables, any other value keeps the default. Read once. Defined in `crates/rdna-compute/src/tensor_ops.rs`. |
| `HIPFIRE_QWEN4_QSA_FILTER_SELECT` | gfx1151: on unless `0`; other gfx11+: off unless `1` | Eager gfx11+ SIMT score/select calls with at most 512 budget blocks replace select-from-scores with deterministic sampled threshold, tiled candidate compaction and exact finish (`qsa_filter_select.hip`) when the call has more pooled blocks than the measured crossover of the select it replaces: 4096 against the hipcc select; against the gfx1151 PM select 24576 for calls of at most 16 rows (decode, verify) and 49152 otherwise. Underfilled or overflowing candidate rows repair exactly in-kernel; ordered selected indices, causal tails, padding and final-row mirror remain byte-identical. Uses 2048 samples, target 1024 and candidate capacity 4096. Recording/capture and uncovered calls retain the incumbent; score arithmetic and PM score admission are unchanged. |
| `HIPFIRE_QWEN4_QSA_FILTER_FEW_ROWS32` | **opt-in** (developer); off unless `1` | Candidate admission only, default off: with `1`, exact-wide verify calls of 17..=32 rows (`Gpu::few_row_limit` above 8, eager verify forward; never prefill, recording or capture) use the gfx1151 PM few-row filter crossover (24576 pooled blocks) instead of the 49152 many-row one, matching a one-row decode's threshold. Selection is exact either way (each row selects over its own block count, same filter trio and in-kernel repair, byte-identical indices, tails, padding and mirror); only the select-versus-filter performance choice changes, so keep it off until the PM-select versus filter crossover at 16/32 rows is measured. Unset or `0` keeps the existing thresholds (4096 hipcc, 24576 for at most 16 rows, 49152 above). Read once; no effect without `HIPFIRE_QWEN4_QSA_FILTER_SELECT` admission. |
| `HIPFIRE_QWEN4_QSA_SCORE_ROWS32` | **opt-in** (developer); off unless `1` | Candidate admission only, default off: with `1`, exact-wide verify calls of 17..=32 rows (`Gpu::few_row_limit` above 8, eager verify forward; never prefill, recording or capture), index width 4 x 128, score with one rows32 tile (`indexed_attention_select_scores_rows32_{f32,bf16}_pm_gfx1151` on the PM arms, `indexed_attention_select_scores_rows32_{f32,bf16}` on the hipcc arms) instead of two rows16 tiles. Same score bits and selection either way; the grid-y divisor always matches the symbol's tile. Extent checks (`HIPFIRE_QWEN4_QSA_EXTENT_CHECK`) treat the PM tile as reading all pooled blocks of the call and the hipcc tile as reading only up to its last row's visible blocks. Stays off until the rows32 tile is measured to beat two rows16 tiles. Read once. |
| `HIPFIRE_QWEN4_QSA_DECODE_SPLITK` | exact gfx1151: on unless `0` (kill switch); ignored on every other architecture | KLD-gated, **not bit-exact**. Runs QSA decode attention as split-K: `qsa_decode_splitk_partial_f32` cuts each (row, KV group) selected list into 32 contiguous slices (one workgroup each, per-slice softmax partials in the QSA per-launch scratch) and `qsa_decode_splitk_combine_f32` merges them in slice order (module `qsa_decode_splitk`, `qsa_decode_splitk.hip`), so decode fills the GPU at a few heads and rows. Scoring is the 32-way KV-group layout: one workgroup per KV head scores its 12 query heads together, reading each K/V row once for all 12 heads. The slice merge and the grouped dot reassociate the softmax and dot sums, so output differs from the per-head kernel in the last bits and the route is accepted by KLD, not byte equality (decode KLD against the BF16 reference, split-K minus per-head route: code24 +0.0011, WT2 +0.0009). Exact gfx1151, F32 K/V, head_dim 256, 12 query heads per KV head, 1..=`Gpu::few_row_limit` rows (8; up to 32 inside an eager exact-wide verify forward, where each row partitions its own selected length and the same kernels run with grid z = rows), eager launches only: no recorder, no graph capture; every other shape, format, arch and recorded or captured launch keeps the one-row per-head kernels, and `HIPFIRE_QWEN4_QSA_DECODE_SPLITK=0` keeps them everywhere. This is one policy (`qsa_decode_attention_policy`, tensor_ops.rs) for every admitted call: architecture, flag, K/V format, head_dim, head grouping, the LDS fit of the context-derived longest selected list, and eager versus recorded/captured launch decide split-K versus per-head; the only row-count term is the `few_row_limit` admission, so an admitted call never takes the batch-size-only hg4, hg12 or WMMA attention routes (n rows reproduce n one-row decodes of the same mode). Calls past the limit (prefill, or 9..=32 rows outside an eager verify) keep the row-routed choice. AR decode in a recorded replay or graph (per-head) and an eager verify (split-K) are different arithmetic: `qsa_decode_attention_eager_differs_from_recorded` reports it. The workspace is reserved once at 32 rows x 48 splits and only the launch's actual `rows x heads x splits` records are mapped. Split count 32 and scoring layout 2 are the constants `QSA_DECODE_SPLITK_SPLITS` and `QSA_DECODE_SPLITK_SCORING_LAYOUT` in `crates/rdna-compute/src/tensor_ops.rs`, selected from an R9700 (gfx1201) sweep (lowest summed time over the 1131- and 8188-token cases; S16 is fastest at 1131 tokens, S32 at 8188), pending Strix Halo KLD and performance. The kernels accept up to 48 splits and the direct test entry also reaches the per-head layouts 0 and 1. |
| `HIPFIRE_QWEN4_QSA_EXTENT_CHECK` | off (developer); exactly `1` or `2` enables | Parsed once into the GPU flags. `1` checks QSA kernel byte extents against VMM mapped prefixes or current allocation sizes and aborts with arena, logical shape, byte limit, required end, depth and launch site on overflow. `2` also calls `hipDeviceSynchronize` immediately after each checked launch and aborts with that site's diagnostics on error. Every other value is off; the disabled path only tests the flag. This is fault-localization instrumentation, not a serving setting; mode 2 serializes launches. |
| `HIPFIRE_QWEN4_QSA_CODE_CHECK` | off (developer); exactly `1` enables | Parsed once into the GPU flags. Snapshots non-writable loaded-code ranges through verified CPU aliases after module load, then compares once after the first host-side HIP error 700; no GPU copies or synchronization. Unsupported aliases are unavailable, and aborts before a host-side 700 produce no capture. A mismatch shows persistent CPU-visible corruption at capture time, not its writer; equality does not establish fault-time bytes or exclude instruction-cache, PTE/translation, transient corruption or a bad PC. Every other value is off: no enumeration, HSA lookup or hook, only the flag test. The `qsa_code_check_selftest` example checks startup alias/baseline eligibility through the same CPU snapshot path without consuming the first-700 slot; build/run instructions are in its source. See docs/KNOWN-ISSUES.md. |
| `HIPFIRE_QWEN4_MOE_SYM_IU4` | gfx1151 and gfx1201: on unless `0` / `false` | gfx1151 and gfx1201 only: runs Qwen4 prefill chunks of 512+ rows on the grouped symmetric IU4 MoE route (stable grouping, A4 gate/down activations, IU4 gate/up and down) for layers whose every routed-expert QT44/QT53 header is verified symmetric (`zp == -8*sc`) on the device at load. Only a symmetric requant artifact (GPTQ3) takes it: shipped asymmetric artifacts fail the check and stay on the F16 WMMA route automatically, and the load log prints `N/48 layers verified symmetric`. Trade on gfx1151 (Strix Halo, GPTQ3 artifact): pp8192 ~1,775 tok/s vs ~1,459 on the F16 route, BF16-reference KLD 0.1027 vs 0.0661. Trade on gfx1201 (R9700, `auto` placement, GPTQ3 artifact): pp8192 2,171.5 vs 1,449.1 tok/s (median of 3 each), BF16-teacher KLD WikiText-2 0.0877 vs 0.0572 (+0.0306) and code 0.1018 vs 0.0733 (+0.0285), the same deltas as the Halo's. Needs the default two-candidate A4 producers (gfx1151: `HIPFIRE_GFX11_A4_CANDIDATES` unset or `2`; gfx1201: `HIPFIRE_G12_A4C2` unset or `1`). The GEMMs are certified builder (PeaceMaker) expert-run tiles: four 16-slot tiles per expert weight stream, eight on gfx1201 for layers whose experts are host-mapped (spilled past the VRAM budget); every width is byte-identical to the 16-slot tile. Not bit-exact against the F16 WMMA route. A typed boolean (`developer.qwen4_moe_sym_iu4`): `true` / `1` requests the route and `false` / `0` keeps the whole F16 route (scatter, producers, GEMMs) on symmetric artifacts too; any other spelling is rejected with `expected a boolean: true, false, 1 or 0` rather than ignored. Decode, MTP and smaller prefill always keep it. Every other arch ignores it. |
| `HIPFIRE_QWEN4_APPEND_WMMA` | gfx1151: on unless `0` (developer) | gfx1151 only. Flash-Next prefill tiles of `HIPFIRE_QWEN4_APPEND_WMMA_MIN_ROWS`..511 rows take the >= 512-row kernel family instead of the F32 SIMT routes. These are agent-turn appends, tool results and short prompt tails. The tile gets the canonical routing of `HIPFIRE_PREFILL_PM_INVARIANT`, applied to that tile only: F16 WMMA dense and MQ6, WMMA QSA, BF16 HC streams and WY GDN. Its MoE takes the F16 WMMA grouped arm, with symmetric IU4 suppressed for the tile. This changes the logits of those tiles. Strix Halo, 32K depth: 145 / 273 / 401-row appends take 428 / 665 / 900 ms -> 282 / 359 / 434 ms. KLD against the BF16 teacher, with every scored row from appends of at most 145 / 273 / 401 rows: WT2 -0.0011 / -0.0000 / -0.0006, code24 +0.0032 / +0.0013 / +0.0008. `0` keeps those tiles on the F32 routes. Decode, MTP verify, tiles of 512+ rows and the `HIPFIRE_PREFILL_PM_INVARIANT` profile are unchanged. |
| `HIPFIRE_QWEN4_APPEND_WMMA_MIN_ROWS` | `48` (developer) | Smallest prefill tile row count the append route (`HIPFIRE_QWEN4_APPEND_WMMA`) takes. Values below 2 are ignored. On Strix Halo the route loses at 33 rows (211 -> 217 ms) and wins at 55 (279 -> 238) and 81 (337 -> 252). |
| `HIPFIRE_QWEN4_COMPACT_SCRATCH` | **opt-in** (developer); off unless `1` | Exact gfx1201, storage-only: selects the compact Flash-Next forward-scratch plan (`Qwen4ForwardScratchMode::FnSymGfx1201`) instead of the full one. Taken only when the load proves it from the artifact payload before expert placement: exact gfx1201, grouped MoE GEMM on, the symmetric IU4 route requested (`HIPFIRE_QWEN4_MOE_SYM_IU4` not `0`), every trunk routed-expert QT44/QT53 header on the symmetric grid (`zp == -8*sc`, finite; read from the bytes, not from metadata labels), and BF16 PLE key/value weights (no `HIPFIRE_QWEN4_REQUANT` rule on them). The plan sizes the grouped-MoE and shared-down buffers for the BF16 grouped route (with a 512-row F32 fallback) and aliases phase-disjoint PLE, HC and GDN buffers: the 8192-row chunk's scratch is 4,240.125 MiB smaller and no larger than today's 4096-row chunk, so `auto` expert placement (`HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS`) is charged the compact plan and can keep today's resident expert layers at chunk 8192. No kernel, route, chunk partition or arithmetic changes: outputs are byte-identical. Any other arch, flag or artifact (asymmetric or non-finite headers, grouped MoE off, IU4 off) keeps the full plan; a compact scratch refuses a >512-row MoE invocation that would need the full buffers before its first launch. `0` or unset keeps the full plan; the load log prints the plan when it is compact. |
| `HIPFIRE_QWEN4_EXPERT_STAGE` | on for exact gfx1201, off elsewhere; `1`/`0` force | Qwen4 long prefill with host-mapped experts (`HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS` below the layer count): each forward of at least `HIPFIRE_QWEN4_EXPERT_STAGE_MIN_ROWS` rows DMA-copies every host layer's unchanged QT44/QT53 expert bytes into VRAM ahead of its MoE, through static pointer tables and compute/copy events, instead of reading them over PCIe. Placement is unchanged: the two stages borrow the expert VRAM of the last two resident layers once their own MoE has run, and those layers are copied back from pinned host copies (2 x 1,275 MiB, taken at load) before the forward returns, so decode, short prompts and MTP verify run on today's residency and mapped reads. The restore overlaps the last host layer and the LM head; what is left is charged to that chunk. Byte-identical: no arithmetic changes. Off (one `qwen4 long-prefill expert staging: off` line) when the host RAM or GTT checks refuse the restore copies, fewer than two layers are resident, or the artifact is not the Flash-Next QT44/QT53 geometry; a stream capture keeps the mapped path. |
| `HIPFIRE_QWEN4_EXPERT_STAGE_MIN_ROWS` | `min(2048, effective prefill chunk rows)`; exact gfx1201 keeps `2048` | Smallest forward (prefill chunk) that stages under `HIPFIRE_QWEN4_EXPERT_STAGE`; values below 2 keep the default, an explicit value is authoritative. The default is capped at the forward's prefill chunk (`HIPFIRE_PREFILL_CHUNK_ROWS` or the arch default, as admitted at load), so a full chunk always qualifies where the chunk is below 2048 (gfx1100: 1536 rows) instead of the stage never engaging. A layer copy takes 23.5 ms on gfx1201, which a 2048-row layer already hides (pp2048 1057 -> 1377 tok/s on card B, 18 ms of restore left exposed per chunk), so gfx1201 stays at 2048. The load log line reports the effective value and its source (`env` or `default`). |
| `HIPFIRE_QWEN4_EXPERT_STAGE_CAPTURE` | on unless `0` | Allows Qwen4 long prefill with wide-hidden output capture (native-MTP prompt fill) to use `HIPFIRE_QWEN4_EXPERT_STAGE`, subject to its existing placement and `HIPFIRE_QWEN4_EXPERT_STAGE_MIN_ROWS` admission. This is output capture, not HIP graph capture: hidden rows are copied after the trunk program on the compute stream. `0` restores the old exclusion, keeping wide-hidden forwards on mapped expert reads. Decode, below-threshold verify/short chunks, the QSA parity observer and HIP stream capture keep their existing exclusions. No measured identity or performance result is claimed for this extension. |
| `HIPFIRE_QWEN4_MOE_SYM_PM` | `1` | gfx1151 only, when `HIPFIRE_QWEN4_MOE_SYM_IU4` applies: `0` runs the route's two grouped IU4 GEMMs from the hipcc module (`qwen4_moe_iu4_sym_gfx1151`) instead of the embedded builder bundle. Both produce the same bytes; this is a same-binary control. gfx1201 has only the builder bundle and ignores it. |
| `HIPFIRE_QWEN4_TRUNK_IU4` | marked zab artifact on exact gfx1151: `gdn.z,gdn.a,gdn.b`; otherwise off | Only a `qwen4_trunk_iu4_default=gfx1151-zab-v1` marker requests the zab default, from the artifact's HFQM metadata or from an applied `<model>.trunk-overlay` sidecar (`HIPFIRE_QWEN4_TRUNK_OVERLAY`); canonical stock aliases without the sidecar and other architectures are unchanged. An explicit value overrides the artifact default: `0` disables the trunk A4 route but **retains the baked or overlaid QT44 weights**; for byte-exact weight rollback load the stock XTS artifact without its `.trunk-overlay` sidecar (or set `HIPFIRE_QWEN4_TRUNK_OVERLAY=0`). Values: `1` / `all`, or comma-separated `family[@lo-hi]` tokens (inclusive 0-based full-layer indices; `@n` = one layer). Families: `gdn.qkv gdn.z gdn.a gdn.b gdn.out qsa.q qsa.k qsa.v qsa.idx qsa.o`; groups: `gdn`, `qsa`, `all`. Invalid tokens or layer bands fail the load. Every QT44 trunk matrix must pass the device symmetry check (`zp == -8*sc`); the load log reports the canonical mask, symmetric counts and A4 projection count (zab: 108, other trunk projections: 132). Eligible eager prefill chunks of at least 512 rows in whole 128-row tiles share A4 activation producers and dense SET GEMMs; jointly selected QT44 z/b/a fold into one projection. Recording/capture and smaller chunks keep existing fallback dispatch. MQ4 outside the mask keeps exact F16 activations from 96 rows; generic MQ4 at 64..95 rows can itself use A4. MQ6 projections and the native MTP head are unchanged. Not bit-exact against stock or QT44/F16: this is a numerical artifact/activation change, quality-gated separately. Needs `HIPFIRE_IU4_PREFILL` and `HIPFIRE_IU4_SYMFOLD` on (their defaults). |
| `HIPFIRE_QWEN4_TRUNK_OVERLAY` | `auto` | Read once per load. Whether a present `<model>.trunk-overlay` sidecar (pre-encoded QT44 `linear_attn.in_proj_{z,a,b}` tensors, bound to the exact base file by length and header/index sha256) replaces the base's MQ6 tensors and arms the trunk A4 default its manifest names (`gfx1151-zab-v1`: zab on exact gfx1151). `auto` (also unset) applies it only on the exact architecture of that default (gfx1151); `0` never applies it (kill switch: the stock base weights run, one log line names the sidecar); `1` applies it on any architecture (the A4 route default itself still arms only on its exact architecture; elsewhere `HIPFIRE_QWEN4_TRUNK_IU4` must select it). Any other value fails the load. A sidecar that does not bind to the base, fails a payload sha256, names an unknown default, is not for a Flash-Next profile, or sits next to a base that already carries a default marker is **refused with one `REFUSED` log line and the base weights run**; the load never fails because of the sidecar. An absent sidecar logs nothing. `HIPFIRE_QWEN4_TRUNK_IU4=0` keeps the overlay weights but disables the A4 route. `HIPFIRE_QWEN4_REQUANT` still applies after the overlay. |
| `HIPFIRE_QWEN4_TRUNK_IU4_RAGGED` | on unless `0` | Read once. Extends the trunk A4 route only to jointly masked symmetric QT44 GDN z/b/a on eager gfx1151 prefill with >=512 rows not divisible by 128. Pads to a 256-row V2B tile, zeros only unused input rows, and writes padded outputs in place in dedicated full-capacity F32 scratch. Real rows are byte-identical to aligned IU4. Insufficient capacity, compact/aliased scratch, invariant prefill, recording/capture, decode and MTP verify keep the previous path. Other families never take ragged IU4. `0` restores exact-activation fallback for ragged chunks; aligned chunks are unchanged. |
| `HIPFIRE_QWEN4_GDN_CONV_QKNORM` | gfx1151: on unless `0`; ignored elsewhere | Exact gfx1151 only, on the chunked GDN prefill route (>= 512 rows, eager, F16 WMMA route on): makes the GDN convolution launch also store the Q/K the recurrence reads already normalized (`gated_delta_conv_qknorm_bf16_f32_batched_k4` in `tensor_ops.hip`), and skips the separate `gated_delta_qk_norm_bf16_batched` launch. The normalized Q/K and every other output are bytewise what the two-launch pair stores; the convolution output's Q/K columns are not written (the recurrence reads only V from it). Needs 16 key heads of 128 in q\|k\|v channel order, `channels % 256 == 0`, a 3-row ring and a BF16 convolution output; any other shape keeps the two launches. Never taken under a recorder, a graph capture, or a row capture, and gfx1100/gfx1201 (whose GDN prefill is the persistent route) ignore it. `0` keeps every launch of the incumbent route; this is the kill switch. |
| `HIPFIRE_QWEN4_GDN_PIPE` | gfx1201 and gfx942: on unless `0`; ignored elsewhere | Exact gfx1201 (F32 and Q8 GDN state) and exact gfx942 (F32 GDN state only), on the batched GDN recurrence without a state capture (prefill; not verify or rollback): runs `gated_delta_step_pipe128_{f32,q8}` (`tensor_ops.hip`, 128-thread blocks, one value column per thread with its 128 state values in registers, no cross-thread handoff per row) instead of `gated_delta_step_halves_state128_persistent256_{f32,q8}`. The output and the final F32 / Q8 state (codes and scales) are bytewise the persistent256 kernel's; on gfx942 the kernel uses persistent256's pinned `__fmul_rn` / `__fmaf_rn` / `__fsub_rn` F32 expressions. The Q8 kernel exists on gfx1201 only (gfx942 refuses a Q8 GDN state). `0` keeps the persistent256 kernel; this is the kill switch. |
| `HIPFIRE_QWEN4_GDN_Q8_INLINE` | **opt-in**; off unless `1` | Exact gfx1151 only, on the chunked GDN prefill route with a Q8 GDN state: `1` runs the recurrence as `gated_delta_chunk_gate_q8_wmma` (module `gated_delta_chunk_q8_wmma`), which decodes the Q8 slot to F32 on load and requantizes it (seeded by the last row, `position + rows - 1`) on store, instead of the `gdn_state_q8_to_f32` / `gdn_state_f32_to_q8` launches and F32 scratch around `gated_delta_chunk_gate_wmma`. The gated output and the Q8 slot are bytewise the conversion arm's. An F32 state, a slot not 16-byte aligned, and every other arch keep the incumbent arm. Unset or `0` keeps every launch of the incumbent route; this is the kill switch. |
| `HIPFIRE_FN_LANE_PM_ROWS` | gfx1151: on unless `0`; ignored elsewhere | Exact gfx1151 only, inside the Flash-Next (Qwen4) lane step (`Gpu::lane_rows_scope`, set by the lane-step driver), outside a recorder or graph capture: routes the four incumbent rows launchers at `rows` 2..=8 to the embedded certified builder module `kernels/qwen4_lane_rows_pm_gfx1151.hxaco` (module `qwen4_lane_rows_pm`): `gemm_mq6g256v2_f32_rows` (K % 256 == 0) to `gemv_mq6g256v2_xbatch_pm`, `gemm_mq6g256v2_f32_rows_x4` to `gemv_mq6g256v2_x4_xbatch_pm`, `gemv_q8_0_k8_rows` (K % 2048 == 0) to `gemv_q8_0_k8_xbatch_pm`, and `gemv_q8_0_staged_rows` at K = 320 to `gemv_q8_0_k320_xbatch_pm`. Each column keeps the arithmetic DAG of the 1-row incumbent (byte-identical to it and to the incumbent rows kernel) while sharing the activation loads across weight rows; operands whose byte offsets exceed 32 bits keep the incumbent. `0` keeps every incumbent launch; this is the kill switch. Byte oracle: `cargo run -p rdna-compute --features lab --example lane_rows_pm_oracle -- [--bench N]`. |
| `HIPFIRE_QWEN4_HC_FUSE` | gfx1151: `3`; gfx1201: `1` | Exact gfx1151 and gfx1201, on the F16 WMMA HC read (>= `QWEN4_F16_WMMA_MIN_TOKENS` rows, eager, no recorder or graph capture; gfx1201 also needs `HIPFIRE_QWEN4_F16_WMMA_GFX1201`, default on, which enables that read). Fusion level of the HC read/write pair around each Qwen4 mixer; every level leaves the HC streams bytewise what the unfused sequence stores. `1`: the HC read's norm launch also projects its paired HC write's gates (`hyper_norm_gate_outputs`), so the write skips its own norm + gate launch. `2`: also the MQ6G256V2 attention output projection (GDN `output`, QSA `o_proj`) applies the HC write in its GEMM epilogue (`gemm_mq6g256v2_wmma_gfx11_bt8_x4_hcw`; gfx1201 `gemm_mq6g256v2_wmma_gfx12_bt8_hcw`, >= 64 rows); the attention output tensor is not materialized. `3`: also, on gfx1151, a BF16 shared-expert down on the F16 WMMA route folds the BF16 scaled add and the HC write into its epilogue (`gemm_wmma_lds_128_256_32_64_k64_hcsd`); the routed `moe_output` rows are then NOT rewritten with the shared-down sum (they are dead after the write). Levels 2 and 3 measured slower than 1 on gfx1201, which defaults to 1. A level that a given layer's shapes or routes do not admit runs the next lower one. An unparsable value keeps the arch default. `0` keeps every launch of the incumbent route; this is the kill switch. |
| `HIPFIRE_QWEN4_HC_UP_TILE` | gfx1151 / gfx1201: on unless `0` | Exact gfx1151 and gfx1201, on the F16 WMMA HC read: runs the up projection + branch mix on the retiled operand-swapped entries (`hyper_read_up_wmma_bf16_swap` on gfx1151; `hyper_read_up_wmma_bf16_gfx1201_t128` on gfx1201 when `low_rank % 64 == 0`), bytewise the baseline entries' output. `0` keeps the baseline entries; this is the kill switch. |
| `HIPFIRE_QWEN4_MOE_COMBINE_ZINIT` | gfx1151 / gfx1201: on unless `0`; ignored elsewhere | Exact gfx1151 and gfx1201, sealed Qwen4 MoE grouped prefill whose combine takes the BF16-row path 2 arm with the shared down after the combine: starts the combine from +0.0 (`moe_down_combine_grouped_top10_bf16in_zinit`) and drops the `moe_output` zero fill that precedes the call. Bytewise the zero fill + the unchanged combine. The fill stays whenever any condition is unmet (other route, recorder / retained tape / graph capture, target not exactly the cleared tensor); an absorbed fill the combine did not consume fails the call instead of leaving the target uncleared. Also taken on a G2-staged prefill (host-mapped experts), which runs the whole step program in one interpreter call with the stage waits and refills as MoE hooks. `0` keeps the separate fill; this is the kill switch. |
| `HIPFIRE_QWEN4_HC_ROW_FOLD` | gfx1151: on unless `0`; ignored elsewhere | Exact gfx1151 only, with `HIPFIRE_QWEN4_HC_FUSE=3` and `HIPFIRE_QWEN4_MOE_COMBINE_ZINIT=1` (both gfx1151 defaults) on the sealed Qwen4 MoE grouped prefill whose combine takes the BF16-row path 2 arm, eager, BF16 HC streams, when the MoE block's HC write is directly followed by the next layer's HC read on the same streams and the F16 WMMA read route: runs the shared-down GEMM as a plain BF16 store (`gemm_wmma_lds_128_256_32_64_k64_bf16st`) and replaces the combine, the shared-down fold + HC write epilogue and the next read's `hyper_norm_gate_outputs` with one per-token row kernel (`hc_row_fold_norm_gate`) that combines the ten expert rows in rank order, folds the shared row, writes the four HC streams and emits the next read's F16 normalized row and its paired write's gate logits; the next read then skips its norm-and-gate launch. Bytewise the unfused chain (streams, F16 row, gate logits). Idea from Gufo upstream (gufo-org/gufo @1071b361, MIT). `0` keeps the existing launches; this is the kill switch. |
| `HIPFIRE_QWEN4_ROUTER_FAST` | gfx1151 / gfx1201: on unless `0`; ignored elsewhere | Exact gfx1151 and gfx1201, Qwen4 E512/top-10 grouped prefill router with more than one token, eager (no recorder or graph capture): runs `moe_router_softmax_top10_f32_fast` (one wave32 per token, logits register-resident, no workgroup barriers) instead of the one-workgroup-per-token `moe_router_softmax_top10_f32`. Bytewise the same expert ids and weights (same stable top-ten order, same left-to-right float32 denominator, same expf calls, same early-return rules; a token with any non-finite logit runs the incumbent's selection verbatim). Decode (one token), recording and capture keep the incumbent symbol. `0` keeps the incumbent; this is the kill switch. (Idea from Gufo upstream, gufo-org/gufo @1071b361, MIT.) |
| `HIPFIRE_QWEN4_DECODE_ZBA_X3` | gfx1151: on unless `0`; ignored elsewhere | Exact gfx1151 (default 2-row GEMV), eager one-row (`route_rows` 1) F32 FN GDN decode input projection group of an MQ6G256V2 qkv and MQ4G256V2 a/b/z on the same F32 rotated input and K (not the 248,320 x 2,048 lm-head shape), with no per-call trunk IU4 route and no projection already covered; a stored A4 mask or ZBA fold may be present. Replaces three `gemv_mq4g256v2_multirow_r2` launches with `gemv_mq4g256v2_multirow_r2_x3`. Each matrix uses a verbatim mirror of the incumbent r2 arithmetic, accumulation chains and shuffle tree; the incumbent source stays byte-identical to preserve accepted gfx1201 PM pins. Qkv and its preceding shared FWHT remain unchanged. Removes two launches per GDN layer (72 per 36-layer token). Gated by `!is_recording && !capture_mode`; recording, graph capture, other arches/routes and `0` keep the incumbent launches; this is the kill switch. Hardware check: `cargo test -p rdna-compute --release --lib mq4v2_multirow_r2_x3_tests -- --ignored --nocapture`. |
| `HIPFIRE_QWEN4_DECODE_HEAD_FUSE` | gfx1151 only: on unless `0`; ignored elsewhere | Exact gfx1151, eager Qwen4 QT44/QT53 grouped single-token decode, K = 2560, BF16 router/selector and Q8_0 shared gate/up with equal row counts and no AWQ. No recording, graph capture, activation/Hessian taps, active MoE dump, hidden dump, EP or EP row gather. Replaces `gemv_bf16_xf32_x4` plus `gemv_q8_0_k2560_staged_pair` with `gemv_bf16_xf32_x2_q8_0_k2560_staged_pair` in standalone module `qwen4_decode_head_fuse`. Verbatim incumbent device-body mirrors preserve per-row arithmetic and reduction order. Saves one launch/layer (48/token); all gate misses and `=0` retain incumbent launches. Hardware bitwise check: `cargo test -p rdna-compute --release --test qwen4_decode_head_fuse -- --ignored`. |
| `HIPFIRE_QWEN4_PLE_FUSE` | gfx1151 / gfx1201: on unless `0`; ignored elsewhere | Exact gfx1151 and gfx1201, Qwen4 PLE block (once per prefill, layer 1) on BF16-stored HC streams: replaces the `hc_state_bf16_to_f32` / `grouped_gate_bf16` / `grouped_norm_bf16` / `grouped_depthwise_conv_silu_add_bf16` / `hc_state_bf16_add_f32` chain with three launches (`ple_gate_rows_bf16s`, `ple_norm_inv`, `ple_conv_add_bf16s` in `grouped_ops.hip`) that keep the incumbent's serial reduction orders and every F32 rounding point, so the HC streams and the convolution state are bytewise the incumbent's for every non-NaN value (where both chains produce NaN, the NaN payload bits may differ). With BF16 key/value weights on the F16 WMMA route the PLE rows are also gathered straight to F16 (`grouped_gather_convert_bf16_f16`), dropping the F32 rows and both F32 -> F16 conversions; the 4-tap dilation-3 PLE convolution runs a register-window kernel (`ple_conv_add_bf16s_k4d3`). The gate scalars and norm inverses live in the `ple_normed` scratch prefix; the F32 widened query, gated, normed and conv-output tensors are not written. Never taken under a recorder, retained tape or graph capture, with F32 streams, more than 4 branches or `hidden % 32 != 0`; those keep the incumbent chain. `0` keeps the incumbent chain; this is the kill switch. (Idea from Gufo upstream, gufo-org/gufo @1071b361, MIT.) |
| `HIPFIRE_QWEN4_HC_DOWN_TILE` | gfx1151 / gfx1201: on unless `0`; ignored elsewhere | The F16 WMMA HC read's down projection (`gemm_bf16_xf16_f16_wmma`, 320 rows). Exact gfx1151 (K = 10240, >= 2048 rows, eager, no recorder or graph capture): runs it on the 160×64 pipelined LDS tile `gemm_wmma_lds_160_64_32_64_k64_p` (five 32×64 waves, 256 workgroups) instead of 64×64 (640 workgroups). Every output keeps the 64×64 tile's single ascending 16-element K chain and store, so the F32 `low` is bytewise the baseline's. Smaller batches, other shapes and arches keep the baseline tile. Exact gfx1201 (K % 256 == 0, every batch, eager and recorded): runs the split-K GEMM on `qwen4_wmma_lds_160_80_32_80_k32_p_s4` (five 32×80 waves, K staged 32, 15360 B of LDS per workgroup) instead of `qwen4_wmma_lds_64_128_32_64_k64_p_s4` (24576 B). It keeps the same four K splits, ascending 16-element substeps and fragment map, so the partials and the reduced `low` are bytewise the baseline's. `0` keeps the baseline tile on both; this is the kill switch. |
| `HIPFIRE_MTP_INCREMENTAL` | **unset**: per-window choice | Native MTP verify route. Unset picks, per window, a batched `(K+1)`-row verify or the interleaved route (one target row per draft, stop at the first rejection); the batched depth comes from the AR floor's chooser (measured depths plus fitted neighbours, at the depth that minimizes cost per emitted token). `0` forces batched at the full `mtp_k`; `1` forces interleaved. Both emit AR's greedy tokens. |
| `HIPFIRE_MTP_AR_FLOOR` | on (unset) | Qwen4 native MTP same-request AR floor (a measured target-only token prices AR; a request whose speculative options never beat it retires to head-free AR). A retired request copies each AR token's target hidden row to the spec hidden capture and owes the head that row; the head appends the owed rows teacher-forced, as an n-gram takeover appends its kept rows (one batched decode fill where it runs, else one Append per row), every 1024 tokens and before the terminal pending-seed flush, so a retired turn still ends with head and target together and the conversation's next turn continues live. `0` disables the floor and restores the pre-floor per-window chooser exactly: no calibration token, no probe windows, no retirement, no floor timing syncs; each window routes by the batched-depth chooser (interleaved when no depth pays). Any other value keeps the floor on. Read once per request. Diagnostic / A-B use; distinct from `HIPFIRE_MTP_INCREMENTAL`, whose forced routes win when both are set. |
| `HIPFIRE_MTP_FLOOR_AR_PROBE` | `6` | Fresh ordinary AR tokens a singleton Qwen4 native MTP request's floor times at a would-be retirement, to re-price AR where the request is now instead of by its single first decode token (under the Flash-Next expert hot set that token's cost is a property of its routed experts: 7–26% off the request's mean AR token on R9700), at most once per `HIPFIRE_MTP_FLOOR_MIN_WINDOWS` windows. With the probes on, a retirement must also lose to the slow-memory (~25 windows) realized per-token cost of all the request's windows, so a streak of rejected drafts no longer retires a request that wins. The probe tokens are real, exact AR tokens (the calibration step: head kept synced). An integer ≥ 0; `0` is the kill switch and restores the single-calibration-token floor; anything unparsable keeps the default. Continuous-batching lanes are unchanged. Route only: greedy output is byte-identical at every value and a sampled request stays distribution-exact. Read once per request. |
| `HIPFIRE_MTP_FLOOR_MIN_WINDOWS` | `16` | Speculative windows a singleton Qwen4 native MTP request observes before its AR floor may retire it; measured routing still starts after the 3-window probe budget, and inside the grace a losing request keeps drafting on its cheapest measured depth. An integer ≥ 1, anything else keeps the default; `3` restores retirement at the probe budget. Continuous-batching lanes are unchanged. Route only: greedy output is byte-identical at every value. Read once per request. |
| `HIPFIRE_MTP_FLOOR_TRACE` | off; `1` enables | One `QWEN4_FLOOR_TRACE` JSON line per calibration / AR-probe / AR token and per native or external window on stderr (a native window when the next window is planned): wall us (start to next plan), emitted tokens, drafted depth, the AR price, every chooser candidate's us per emitted token (`"route":"capN"` for routing at depth N, with its mean window `win_us`, realized `emit`, mean `drafted` depth, and `"modeled":true` when it is priced only from a neighbouring cap), the fitted per-row cost `row_us`, the 8-window and slow realized cost, probes left, retirement. A drafter's first batched window at a drafted depth traces as event `native_batched_first_use`. Diagnostic. |
| `HIPFIRE_QWEN4_PENALTY_PREPASS` | on (unset) | Qwen4 / Flash-Next repeat, presence and frequency penalties on the GPU. A penalized greedy AR request penalizes each decode logit row with a GPU prepass (`logit_penalty_table_rows`) and keeps the device argmax; a penalized sampled native MTP request penalizes its target verify rows with the same prepass (all rows of a batched verify in one launch) before downloading them. The host computes every per-token factor and subtrahend with the CPU policy's arithmetic, so each row is bit-identical to `sampler::apply_logit_policy_cpu`; draft `q`s keep the host policy over the same histories. `0` restores the host path (download the row, apply the policy on the CPU). Read once per request. Diagnostic / A-B use. |
| `HIPFIRE_MTP_SAMPLED_MODE` | `leviathan` | Sampled (temperature > 0) Qwen4 native MTP verifier, with `speculation.mtp_sampled` on. `leviathan`: speculative rejection sampling (drafts drawn from the head's 8 re-scored candidates; output equals AR in distribution). `naive`: SpecInfer naive sampling (argmax drafts, one AR-sampler draw per verify row, accept iff equal); a seeded request emits AR's exact tokens. Any other value fails sampled requests. |
| `HIPFIRE_QWEN4_MTP_DEVICE_ACCEPT` | off; `1` enables | Flash-Next sampled Leviathan MTP only: resident parallel target distributions plus ordered device acceptance, with reusable host draft-q staging and an 8-byte result readback. Requires neutral penalties (no history window/prepass), finite positive temperature, exact gfx1151 or gfx1201 with device sampler support, at least one draft, pool `1..=64` and cap `1..=pool`; unsupported, penalized, naive, greedy and target-only AR-floor paths remain host routes. Covers native batched/incremental and n-gram point-mass takeover. Unset (or any value but `1`) keeps the unchanged host path. Enabled target law uses lower-token-id ties and fixed-tree sums, not host byte identity; accept law and consumed `SpecRng` chronology stay unchanged. gfx1201 end-to-end A/B ties the host path (+1.83 %, CS95 −2.76 … +6.63 %); default stays off pending the per-seed stability decision. |
| `HIPFIRE_QWEN4_MTP_SAMPLED_PICKS` | off; `1` enables | Flash-Next sampled MTP, only on the `HIPFIRE_QWEN4_MTP_DEVICE_ACCEPT=1` Leviathan route: skips the unused greedy verify-pick D2H through private resident verify/capture entrypoints, keeping the same heads, argmax arithmetic, logits and target/head transaction state. Covers native and n-gram batched/incremental windows; the public `SpecTarget::verify_block` contract still returns real picks. Unset keeps pick downloads. Trace or phase timing keeps real picks; greedy, naive, penalty, unsupported and target-only AR-floor paths stay unchanged. No separate speed claim. |
| `HIPFIRE_MTP_DRAFT_HEAD` | `mq2r` | Draft ranking head: an `mq2`..`mq6` copy of the LM head (`r` suffix = re-score its top 8 exactly against the head's own Q8_0 or MQ6G256V2 rows). |
| `HIPFIRE_MTP_PAIRING` | head state | Draft-step conditioning experiment: `aligned-head` or `aligned`. |
| `HIPFIRE_MTP_TRACE` / `HIPFIRE_MTP_PHASE_TIMING` | off; `1` enables | Per-window MTP trace / per-phase `hipEvent` timing to stderr (diagnostic). Qwen4 `QWEN4_MTP_PHASE` lines carry `source` (`mtp`\|`ngram`), `verify_route`, `k` (drafts verified), `accepted`, `emitted` and `window_us`, wall time after a device synchronize (timing mode only); a takeover confirmed by the head's first draft adds `probe_us`. Batched and takeover windows split the verify forward (`verify`) from the target-row work (`accept`: penalty prepass, row downloads, host distributions). Qwen3.x single-GPU native MTP prints `QWEN35_MTP_PHASE` lines with `source`, `k`, `accepted`, `emitted`, synced `window_us`, and the host time and launch/upload counts of its penalty prepass (`penalty_host_us`, `penalty_launches`, `penalty_uploads`). |
| `HIPFIRE_QWEN4_MTP_APPEND_DET` | exact gfx1151: on unless `0`; ignored elsewhere | Default on (exact gfx1151), **prompt fill only** (`mtp_prefill` and lane prompt tiles): routes all seven projections of the head's batched KV-only Append through fixed-row native kernels instead of the incumbent multirow / K4 / staged-Q8 kernels: `fc_embedding`, `fc_hidden` (`rows * hc_count` rows), HC down, HC up (BF16 weights, `gemm_bf16_xf32_mtp_dense_det_pm`) and the Q8_0 index, K and V projections (`mtp_q8_det_pm_launch`, after one shared F32 to F16 conversion of the mixed row). Each is the PM F16 WMMA with F32 accumulate and one ascending-K chain per output (no split-K), so a row's bytes do not depend on its batch, sub-chunk size or position in the pass. HC down is the raw WMMA output followed by the existing `hc_activation_fused_f32` (scale `1/hc_count`, in place) instead of the K4 GEMV epilogue. **Numerics are explicitly not the incumbent's (G1) and are not byte-equal to it or to the per-row Append:** BF16 weights go through a model-lifetime F16 shadow (BF16 widened to F32, rounded to F16 RNE, once) and X is rounded F32 to F16 RNE on every call, so F16's narrower exponent range can underflow or overflow values BF16/F32 hold, and the gfx11 WMMA's internal dot order is its own. The head's QSA rows, draft state and pending hidden therefore differ from the incumbent's bytes (deterministic run to run); a session snapshot carries whichever arithmetic built it. Target outputs are unchanged (speculative verification): on Strix Halo greedy output is byte-identical to `0`, TTFT is −1.67% at 16K and −2.79% at 9.3K tokens, decode is not slower, and only the draft acceptance (τ) moves slightly per prompt. Never taken by the decode takeover step's head fill (explicitly `DecodeFill`, always the incumbent kernels) or by any other arch, nor on gfx1201. A lone prompt row (a one-token tail) takes the same rows kernels instead of the per-row `append_token` (with `0` a lone row stays per-row). The batched-fill scratch is allocated (and charged in admission) even with `HIPFIRE_QWEN4_MTP_BATCHED_FILL=0`; there DET runs one row per pass, byte-equal to the batched passes by row invariance. Under expert parallelism the drafter is replicated and `mtp_prefill` fans out to every rank, each rank running this same prompt fill on its own GPU (the head's Append step runs no MoE and no EP exchange), so a gfx1151 mesh would take DET identically on every rank; EP itself is gfx1201-only today, where DET is never taken. `0` is the kill switch (the incumbent kernels); read at every `mtp_prefill`. |
| `HIPFIRE_QWEN4_MTP_BATCHED_FILL` | gfx1151 or gfx1201: on unless `0`; ignored elsewhere | Exact gfx1151 or exact gfx1201 only (gfx1200 and every other arch keep the per-row loop), Qwen4 native MTP prompt fill (`mtp_prefill`): runs the head's KV-only Append step over each prefill chunk as one row-batched pass (sub-chunks of up to 1024 rows, about 17 launches each) instead of one ~20-launch single-row `forward_token` per prompt row. Every operator is the single-row kernel with a rows grid, or a multi-row kernel whose per-row reduction order is the single-row kernel's, and never the F16 WMMA route, so the head's QSA state (`full_keys`, `full_values`, `raw_index_keys`, `pooled_keys`, lengths), the draft request state and the pending hidden row are built to match the per-row fill byte for byte; the (token p, hidden p) pairing is unchanged. Read at each `mtp_prefill`. On gfx1201 the per-row references are the singleton `gemv_bf16_xf32` (embedding, HC up), `gemv_bf16_xf32_x4_rows_r4` (hidden projection), the singleton K4 down projection and the unstaged `gemv_q8_0`; gfx1151 keeps its own references. Allocates about 247.77 MiB (259,805,184 bytes) of row-batched scratch per drafter at the first fill (1024 rows, 4096 token IDs), counted in the per-request bytes of `native_mtp_device_bytes` (the MTP reserve is max(build scratch, request bytes)). The same gate feeds expert placement, so on gfx1151 and gfx1201 the auto VRAM reserve can rise by up to that amount and, because resident expert layers are floored to whole layers, can leave one routed-expert layer fewer in VRAM depending on the remaining slack. `0` keeps the per-row loop, allocates nothing and charges no batched-fill reserve, so reserve and placement return to the per-row values; other arches always take the per-row loop. The byte-identity statements above describe the incumbent kernels only: on exact gfx1151 the prompt fill uses the DET projections by default (`HIPFIRE_QWEN4_MTP_APPEND_DET`; `0` restores these kernels; DET is not byte-equal to the per-row fill), and the scratch is allocated and charged even with `0`. |
| `HIPFIRE_QWEN4_MTP_REUSE_PREFILL_IDS` | off unless `1` | Qwen4 native MTP prompt fill on the batched path (`HIPFIRE_QWEN4_MTP_BATCHED_FILL`, exact gfx1151 or gfx1201): the head's batched Append embeds each sub-chunk from the token ids the same chunk's trunk forward already uploaded to its device `token_ids` buffer (rows `off..off + n`, read in place) instead of uploading them again. Taken only when that forward uploaded the ids whole from the host outside graph capture and its host mirror equals the sub-chunk's tokens; a HIP single-row body, a device-argmax id or any mismatch uploads the head's own copy. The kernel sequence, syncs and output bytes are unchanged, and the MTP decode-window append always uploads. Read at every `mtp_prefill`. Default off (the warm path was not output-identical on the RC4c gate); unset or `0` keeps the head's own upload. |
| `HIPFIRE_QWEN4_MTP_SKIP_INTERMEDIATE_PICK` | on unless `0` | Qwen4 native MTP prompt fill (`mtp_prefill`): a prefill chunk that does not end the prompt (a capture-split or natural chunk) skips the final hyper, LM head, argmax and host readback, because only the prompt's last chunk's pick is read (the seed, and the last row's logits for a sampled draw). Such a chunk runs the trunk with `Qwen4OutputPolicy::None` but keeps the full wide hidden capture, so target and head state, the hidden rows the head consumes, the prefix-checkpoint capture boundaries (end of prompt is final and keeps its pick) and the seed are unchanged; the final chunk keeps full logits and readback. Same tiles and trunk route, no new kernel or threshold. It only helps prompts filled in more than one chunk (a capture split or a prompt longer than `prefill_rows`); a one-chunk fill (for example a short cache-hit suffix) has no non-final chunk and is unchanged. Read at each `mtp_prefill`. `0` keeps the head and argmax readback on every chunk. |
| `HIPFIRE_QWEN4_PLE_ASYNC_UPLOAD` | on unless `0` | Qwen4 prefill of more than one token whose PLE rows are staged after layer 0 (no device token, no graph capture): the host-staged rows upload with an asynchronous copy instead of a blocking `hipMemcpy`, so the host no longer waits for the layers already enqueued. The single reusable host staging buffer is fenced by a hip event recorded behind the copy; the next write to it, the teardown and the drop all retire the fence first (an enqueue or fence failure falls back to a device drain, and a drop with a copy pending leaks the bytes instead of freeing them). Bytes, order, kernels and the decode device-token branch are unchanged. Read when the forward is built. Default on (=0 disables the fenced asynchronous deferred-PLE upload and restores the blocking one; =1 forces on). Graph capture and the device-token path stay blocking. Developer knob, not a stable config key. Default flipped in 0.4.1.1 after identity gate (warm/CQ outputs byte-identical); no performance claim. |
| `HIPFIRE_QWEN4_PLE_WIDE_READERS` | on unless `0` | Qwen4 SSD-resident PLE rows: a row request with more than 1,024 scattered page reads (a prefill chunk's ~110K) splits them over up to 256 concurrent reader threads (one per 64 reads) instead of 16; smaller requests (decode, MTP verify, short prompts) keep 16. Bytes, order and the row-store cache are unchanged; only the read queue depth differs. Read when the model loads. `0` keeps 16 readers for every request. |
| `HIPFIRE_QWEN4_TRUNK_TIER` | Q8F16 declaration | `mq6` declares the rank-2 trunk attention/GDN matrices at MQ6G256V2 for the quantizer; the loader admits both tiers from the file. |
| `HIPFIRE_QWEN4_MTP_TIER` | recipe tier | `source` keeps the rank-2 MTP matrices at BF16 (quantizer and loader comparison knob). |
| `HIPFIRE_QWEN4_REQUANT` | unset | Load-time precision experiment: `pat=fmt;...` requantizes resident rank-2 weights whose name contains `pat` to `mq2`..`mq6` or `q8`, or (`qt44:DIR`) replaces each with the pre-encoded QT44 bytes of `DIR/<tensor name>.bin` where that file exists (an offline GPTQ solve; the file must hold exactly the matrix). |
| `HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS` | unset: every routed expert in VRAM where it fits, else `auto` | Discrete-GPU capacity knob. Unset on a discrete GPU keeps every routed expert in VRAM only when all of them, the non-expert weights and the `auto` reserve below fit in free VRAM, and resolves to `auto` otherwise (a 32 GB card); unified memory (Strix Halo) stays fully resident. An explicit value always wins. The load logs the choice on one `qwen4 expert placement` line. Set or unset, a daemon started for a Qwen4 model on discrete GPUs gets the two host-memory switches below (`HSA_USERPTR_FOR_PAGED_MEM`, `GPU_PINNED_MIN_XFER_SIZE`), which are decided before the HIP runtime loads; a daemon started for another model (serve switching models) gets them only with the variable set, and its Qwen4 load warns without them. `N` keeps the routed experts of trunk layers `0..N` in VRAM; later layers' and the MTP layer's routed experts are fulfilled into pinned, device-mapped host RAM and read over PCIe (zero-copy) by the unchanged sealed MoE kernels. `auto` picks the largest `N` that leaves a computed reserve of VRAM beyond the non-expert weights. The reserve is the larger of two formulas. The historical floor is 6.5 GiB (measured without MTP at `max_seq` 32768 with legacy QSA state, a full-context prefill included; N=16 on a 32 GB R9700), plus the trunk QSA arenas' committed bytes past those 32768 legacy tokens, plus, when native MTP stays attached (the default on exact gfx1201, `--spec mtp` elsewhere), the MTP head's committed device bytes (its QSA state and the MQ2 draft copy of the language head) and the larger of the draft copy's F32 requant scratch (2048 rows × hidden × 4 B, 20 MiB on Flash-Next, returned to the device at attach) and what the first speculative request allocates (the verify hidden rows and the `K+1`-row GDN capture rings in the state's GDN format). The census is bottom-up for the first full-context request and is logged term by term on the `qwen4 auto expert placement` line: the trunk QSA context at the admitted `max_seq` as the ACTUAL backend maps it (a VMM arena maps ahead to `max_seq`, rounded to the driver granularity; legacy commits all of `max_seq`), the rest of the request state (GDN recurrent and conv with the snapshot mirror, QSA fixed arenas, PLE conv, hyper feedback), the chunk forward at the largest requested prefill chunk, the lazy F16 weight shadows of the BF16 hyper-connection down mixes, router, shared-expert gate/up/down and PLE key/value projections the long-prefill F16 WMMA arms read (exact gfx1201, queried from the manifest, so a quantized trunk charges nothing), the kernels' pool workspaces (itemised by the engine), the QSA split selector's score/filter scratch, native MTP and the QSA gather workspace. Whole-layer placement (`auto`, and the unset check that decides fully resident, on a discrete card) then keeps a 512 MiB safety margin free beyond whichever formula wins and, on a slab-arena load whose long-prefill expert staging is on (`HIPFIRE_QWEN4_EXPERT_STAGE`, exact gfx1201 by default; not expert-parallel), the two dedicated stage buffers allocated at load, `2 × (largest layer payload + 4096 B)` (it scales with the artifact: twice 1,060 MiB plus the pad, 2120 MiB, for a 1,060 MiB-per-layer artifact), because they are allocated even when every prompt is shorter than the staging threshold. The expert hot set keeps its own margin and tier-plan stage accounting instead. QSA context arenas of the historical floor are charged as committed, never by their virtual extent: legacy state commits all of `max_seq`. A discrete card's floor charges legacy storage of `max_seq` whichever `kv_backend` is selected (the census charges the actual backend's full map); unified memory's floor with VMM state (`kv_backend` `vmm`) charges only the pages the first prefill chunk maps. On the R9700 at `max_seq` 16384 the attached head costs one expert layer (layers `0..15` in VRAM without it, `0..14` with it). The load logs the reserve on a `qwen4 auto expert placement` line, and refuses before allocating when the free VRAM cannot hold the non-expert weights plus the reserve, margin and stage buffers even at N=0 (for example beside another process's model). `0` puts every routed expert (~65 GB) in host RAM. The load refuses before allocating when `MemAvailable` minus the CMA region (the larger of `CmaFree` and the CMA pageblocks: `CmaTotal` or `/proc/zoneinfo` `cma`; only movable allocations can use them, never the pinned host-mapped bytes), plus (with GTT-backed host memory, `HSA_USERPTR_FOR_PAGED_MEM=0`) an estimate of the freed GTT pages in TTM's page pool, is below the pinned bytes plus 4 GiB. The pool is readable only by root, so the estimate is the host memory outside every `/proc/meminfo` counter, minus the GTT amdgpu devices hold and 4 GiB for other drivers, capped at TTM's `page_pool_size`; the load log prints both numbers. `echo 2 \| sudo tee /proc/sys/vm/drop_caches` empties the pool. Refused on UMA. With any expert host-mapped, native MTP stays on under `auto` on exact gfx1201 (greedy MTP text equals AR; `--spec off` / `speculation.mtp = "off"` keeps AR) and is opt-in elsewhere (`--spec mtp` / `speculation.mtp = "on"`; `auto` keeps AR). |
| `HIPFIRE_QWEN4_EXPERT_HOTSET` | unset: on for exact gfx1201 when `<model>.expert-hotset` exists, else off | Developer switch, read once per load: the Lever1 expert hotset, a decode-ranked per-expert VRAM placement of a Qwen4 aggregate-MQ4 base that replaces whole-layer expert placement on a discrete gfx1201 card. **Unset:** the loader looks for `<model file name>.expert-hotset` beside the model (`hipfire_config::expert_hotset_sidecar_path`) and applies it when it exists and applies; no sidecar means today's whole-layer placement, silently. **`0`:** off (kill switch): the sidecar is not looked at and whole-layer placement runs. **An absolute path:** that file is the sidecar, instead of the sibling one. **A relative or empty value:** a policy error, the only refusal this variable makes. The sidecar is JSON (`kind` `hipfire.qwen4.expert_hotset`, `version` 1). It ranks every routed trunk expert by decode routing count (`rank`: each global id `layer * experts_per_layer + expert` exactly once, `counts` descending, ties by global id ascending; `counts` indexed by global id) and carries `layers`, `experts_per_layer`, `top_k`, `capture_tokens`, `counts_sha256`, `format_map` and `format_map_sha256` (full lowercase 64-hex digests). It is bound to its base by `base_size` (`u64`, the base file's length in bytes) and `base_index_sha256` (lowercase hex sha256 of the base's HFQ header and index), both required; there is no whole-artifact hash. The parser (`expert_hotset::HotsetSidecar`) validates kind, version, dimensions, hashes, counts length and rank ordering and refuses unknown keys. **Anything that does not apply keeps whole-layer placement and never fails the load:** an absent, unreadable or malformed sidecar, a sidecar whose `base_size` or `base_index_sha256` does not match the loaded base or whose dimensions do not match the model, another architecture than gfx1201, unified memory, expert parallelism, or an explicit `HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS`; the load logs why on one line. The byte budget is derived at load, never read from the file: free VRAM less the non-expert weights, the `auto` reserve, the frozen expert tables and a 512 MiB margin. |
| `HIPFIRE_QWEN4_EP` | unset | Developer kill switch for Qwen4 expert parallelism: `0` forces `topology.ep = 1` (single device) whatever the config says; any other value leaves `topology.ep` in charge. `topology.ep` (`HIPFIRE_TOPOLOGY_EP`, default 1, allowed `1..=8`) shards the routed experts `e % N` (uneven per-rank counts when N does not divide 512, e.g. 86/86/85/85/85/85 for N = 6) across the first N visible cards, which must be one exact arch, gfx1100 or gfx1201 (a mixed set is refused), with peer access between every pair; every card runs the unchanged single-device program. Under the developer-only `HIPFIRE_EMULATE_GPUS` the ranks alias physical devices (rank `r` on device `r % count`, see the logical GPU emulation section below). EP > 1 admits native MTP only when requested explicitly (`--spec mtp` / `speculation.mtp = "on"`; `auto` stays AR): the MTP drafter is replicated on every card, rank 0 owns the window chooser, and `HIPFIRE_MTP_SAMPLED_MODE=naive` is refused (its draws come from the process-wide AR sampler). EP > 1 still refuses DFlash/DSpark/n-gram/DDTree, adaptive KV, drafters, CASK, PFlash, lane batching and the retained Redline default. |
| `HIPFIRE_QWEN4_EP_TRANSPORT` | unset (`auto`) | Harness only, read by `ep_transport_oracle --devices` (feature `lab`); no product path reads it. Selects the oracle's transport when `--transport` is absent: `peer` (P2P reads of peer VRAM after `hipDeviceEnablePeerAccess`), `host` (shared mapped-pinned host banks: each rank publishes its owned rows/tiles into the bank, every rank reads them back; peer access is never enabled and every kernel pointer is checked against the rank's own allocations and its bank alias), or `auto` (`host` when any directed device pair lacks peer access, else `peer`). |
| `HIPFIRE_EP_BALANCED_OWNERS` | on under EP on gfx1201 at 2..=4 ranks; `0` disables | Flash-Next (Qwen4) expert parallelism only (`topology.ep` 2..=8), exact gfx1201 at 2..=4 ranks only (the per-call owner-assign kernel ships for that arch and covers at most 4 ranks; on gfx1100 a rank logs `qwen4 EP rank r/n: balanced ownership not admitted (needs gfx1201); static split`, and at 5..=8 ranks `... (needs at most 4 ranks); static split`, and runs the exact static split): balanced per-call expert ownership on the host-mapped trunk MoE layers (the layers whose experts are read zero-copy from pinned host memory). Instead of the static `e % ranks` split, each call deals the distinct routed experts to the ranks alternately in ascending expert-id order, so every rank reads (nearly) the same number of host experts and none waits for a rank that drew more. One small device kernel per host layer rewrites the rank's live expert pointer tables from the routed ids; every routed expert is still computed by the unchanged kernel from its own bytes on exactly one rank and copied bit-for-bit to the others, so output is byte-identical to the static split and to a single card. Applies to one-row decode and few-row forwards (verify, replay, takeover, up to 8 rows); resident layers, chunk forwards and the MTP head keep the static split. With `0`, nothing is built: no tables, no buffer, no extra stage and no extra kernel (EP and single-card bytes unchanged). Logs one `qwen4 EP rank r/n: balanced ownership on N host-mapped MoE layers` line per rank at attach. |
| `HIPFIRE_EP_OWNER_ROUTED_PREFILL` | on under EP (2..=8 ranks); exact `0` disables | Flash-Next (Qwen4), `topology.ep` 2..=8, exact gfx1201 (symmetric IU4 route) or exact gfx1100 (F16 WMMA grouped route). Default on under EP whenever every admission condition holds; only the exact value `0` is off (unset or any other value is on, like `HIPFIRE_EP_BALANCED_OWNERS`); single card is unchanged: prefill chunks of at least 512 rows run grouped gate/up and down only on owned expert tiles (`owner_table[e] = e % ranks`). A separate copy of tile IDs masks non-owned tiles to `-1`; the scatter output is unchanged. The gfx1201 route needs every MoE layer verified symmetric on every rank; the gfx1100 route runs the BF16-out WMMA gate/up (SwiGLU epilogue) and down on QT44/QT53 experts, needs no symmetric headers (the per-layer verdict is the expert formats, which every rank must agree on) and takes the stable grouping the symmetric route uses (the fused atomic scatter orders rows within an expert non-deterministically, so ranks could not share a grouped layout). Certified PeaceMaker mask/pull kernels copy non-owned BF16 grouped down rows from their owners before the unchanged combine (approximately 105 MB/layer/rank on EP2 or 157 MB on EP4 before grouping padding). Ready event, host barrier, all-peer ready waits, pull, done event, host barrier; every peer's done is waited before the next grouped-row writer. EP2 retains its 48-byte pull specialization behind the same API; EP3..EP8 take the eight-source pull launch. Needs the arch's route and non-compact scratch; collective verdict mismatch or unsupported rank/device/route refuses the lever for the entire load. Tail chunks under 512 rows, few-row forwards, decode and the MTP head keep today's routes. Without an active stream event waits are host synchronizations. Off allocates nothing, lowers no stage and loads no kernel. |
| `HIPFIRE_CHUNK_DUMP_DIR` | unset | Diagnostic (developer): digest the prefill chunk boundaries of selected layers under `$DIR/rank-<device>/chunk-<start>-<rows>/layer-<L>/digest.json` (SHA-256 of the routed ids, the layer's FFN output, the hyper-connection streams and every live tile of `y_down_grouped`). The chunk forward runs in segments ending after each selected layer and synchronizes the device there; `fn_ep_oracle --chunk-dump` sets it per arm and `fn-ep/coal/s1/compare_chunk.py` compares arms. `HIPFIRE_CHUNK_DUMP_LAYERS` (default `0,2,3,31,47`) selects layers; `HIPFIRE_CHUNK_DUMP_RAW=1` also writes the raw bytes (large; failure diagnosis only). |
| `HIPFIRE_EP_EXPERT_STAGE` | on under EP when owner-routed prefill is admitted on gfx1201; exact `0` disables | Admit Flash-Next long-prefill expert staging under EP (it requires the owner-routed prefill to be admitted, so it is never created when `HIPFIRE_EP_OWNER_ROUTED_PREFILL=0`, and never on gfx1100: an EP rank there logs `qwen4 EP rank r/n: expert staging not admitted (needs gfx1201)` and builds nothing). Only the exact value `0` is off; unset or any other value is on, like `HIPFIRE_EP_BALANCED_OWNERS`. Single card is unchanged. `HIPFIRE_QWEN4_EXPERT_STAGE` remains the pipeline kill switch. Stage only when every trunk MoE binds the compact owner-routed route; full-table EP forwards never stage because they read peer donor VRAM. Each rank copies its owned compact host experts through its last two resident layers (637.5 MiB/layer/rank at EP2, 318.75 MiB at EP4); owned expert `e` uses donor slot `e / ranks`. Two pinned donor mirrors/rank, each charged at the largest rank's share, are charged for all ranks against one host's RAM/GTT. Every rank drains restored donors and crosses the transport's host barrier before any full-table chunk begins. All-resident loads need no host-layer stage; decode row exchange keeps host event waits independently of the staging stream. |
| `HIPFIRE_QWEN4_ROUTE_TRACE` | unset | Diagnostic: append one line per single-token HIP forward to this file, `position` followed by every layer's routed expert ids. Synchronizes the device each token. |
| `HIPFIRE_QWEN4_PREFILL_ROUTE_DUMP` | unset | Diagnostic: append, for every MoE layer of every multi-row prefill chunk, one record to this file: four little-endian u32 (`layer`, `rows`, `topk`, `start_pos`) then `rows * topk` little-endian i32 routed expert ids in slot order (`slot = row * topk + j`). Synchronizes the device per layer. Off by default; bytes are unchanged when unset. |
| `HIPFIRE_QWEN4_CUT_LAYERS` | unset | Developer only: comma-separated absolute zero-based trunk layer indices (1 to 7, strictly increasing, each `>= 1`, below the layer count, and after the PLE layer when the profile has one). Executes the Qwen4 single-card trunk step program as separate interpreter runs split before each listed layer: the cut-safe Single schedule of the pipeline-parallel parity gate (no HC fusion or gate state crosses a cut; the PLE split is kept). Eager only: disables retained replay. Refuses the chunk dump, the QSA tap, `HIPFIRE_QWEN4_ROUTE_TRACE`, `HIPFIRE_QWEN4_PREFILL_ROUTE_DUMP`, expert-parallel forwards and the NPU trunk pair route with an error naming this variable; an invalid list fails runtime construction. Bytes and launches are unchanged when unset. |
| `HIPFIRE_SESSION_CACHE_BYTES` | `8589934592` (8 GiB); `0` disables | Compat env for `memory.session_cache_bytes` (int 0–1 TiB). Byte ceiling, not a reservation, of one loaded Qwen4 or Gemma 4 model's engine-owned session cache; `0` attaches no cache and every turn prefills cold. Gemma 4 snapshots Q8 K/V every 1024 prompt tokens and at the prompt rounded down to the prefill width (never decoded rows), byte-identical to cold prefill. Qwen4 snapshots fixed GDN/PLE/hyper/selections and MTP head/draft metadata whole, and append-only QSA/index/head rows as parent-linked deltas; committed snapshots are shared across sessions, evicted LRU over leaves, and restored from the deepest matching prefix. AR and MTP routes never cross. Qwen4 snapshots and live continuations are **session-exact, not a cold-exact guarantee**: boundaries after a tail/resident restore and later live turns remain eligible for pool capture; cold-exact consumers must not treat those snapshots as cold-prefill references. Snapshots live in device memory (VRAM or UMA) and optionally the disk tier. Every allocation keeps the context's growth reserve plus headroom: UMA `clamp(MemTotal / 16, 2 GiB, 8 GiB)`, discrete GPU 256 MiB; captures skip or evict when this guard fails. Live continuation skips the committed consumed history when the canonical prompt strictly extends it on the same route, prefilling only the suffix, with ties preferring live over snapshots. **Short side requests:** one longer conversation can be parked when divergence is within the first 1024 tokens and prompt + maximum output + write lookahead is at most 8192 rows. Parking copies fixed state plus the overwritten row prefixes (cold reset rewrites from row zero), retaining the rest in place, under the same budget/memory guard. The next strict extension of the parked main history restores it byte-exactly when its match is longer than the side history's; another divergence, context pressure or unload frees the parked copy. Long requests, missing bounds or insufficient memory use ordinary reset/snapshot fallback. For Flash-Next native MTP, the fixed state is about 33 MiB; row cost depends on QSA format (q8 versus F32) and includes the F32 MTP head. Serve reports reused tokens as `cached_tokens`. |
| `HIPFIRE_SESSION_CACHE_DISK_BYTES` | `0` (off) | Compat env for `memory.session_cache_disk_bytes` (int 0–64 TiB). Byte budget of the session cache's disk tier in `HIPFIRE_SESSION_CACHE_DIR`, shared by every model and build whose files are there. A snapshot evicted from memory is written to one file instead of freed, and model unload writes every published snapshot; a restore that hits a disk snapshot loads it back into memory (or a temporary buffer when the memory budget is full) and deletes the file. Disk eviction is LRU over leaves and survives restarts through file mtimes. Files are bound to the daemon executable's SHA-256: another build's files are never restored and are evicted first. Snapshots still in memory at daemon kill or crash are lost. |
| `HIPFIRE_SESSION_CACHE_DIR` | empty = `$HIPFIRE_HOME/session-cache` | Compat env for `memory.session_cache_dir`: directory of the session cache's disk tier (`~` expanded). Locked by one process at a time; a second process logs `session cache disk tier off` and keeps only the memory tier. |
| `HIPFIRE_PLE_HOT_STORE_BYTES` | `0` (off); `auto` opts in | Compat env for `memory.ple_hot_store_bytes` (`auto` or int 0–1 TiB). Prewarm budget of the Qwen4 PLE hot-row store. The store needs the frequency-rank sidecar `<model>.plehot` (`hipfire-quantize ple-hot-rank`) and the hot-row pack `<model>.plehotrows` cut from it (`hipfire-quantize ple-hot-pack`): the rows of the hottest buckets, copied verbatim from the model file and grouped by rank so the hottest rows share OS pages. Rows the pack holds are read from the pack, everything else from the model file, and both decode through the same row decoder, so output is byte-identical. Before serving, a worker thread checks the sidecar hash and compares 4,096 pack rows with the model. After load it reads (prewarms) the first N bytes of the pack into the page cache, hottest tier first and then tier 0 once more so its pages are promoted to the active list, while no request runs: `auto` = `MemAvailable` minus the CMA region minus 4 GiB at ready, an explicit N is that many bytes, both capped at the pack. The pack's pages are ordinary clean page cache, counted in `MemAvailable`, dropped (never swapped) under pressure and re-read from the pack when needed. Unified-memory GPUs only. |
| `HIPFIRE_PLE_FORMAT` | `auto`; `q4` or `q8` pins | Compat env for `memory.ple_format` (`auto` \| `q4` \| `q8`). Which PLE row table a Flash-Next (Qwen4) load reads. `<model>.ple-q8` holds the exact Q8 rows; `<model>.ple-q4` holds 4-bit Lloyd-Max rows derived from them, decode equal to the tested q4 table. `auto` = q4 on gfx1151 (unified-memory Strix Halo), q8 elsewhere; `auto` never derives or downloads and falls back to the other available format with a stderr note. `q4` reads `<model>.ple-q4`; a load never derives it, and a missing file is refused naming both searched locations (the model path as given, then its canonical symlink target), `hipfire pull <tag> --ple q4` and `hipfire ple derive --format q4 <model>` (about 1 minute from the local Q8 rows, progress on stderr). `q8` on a split base (`hipfire ple split <model>`: `<model-stem>-base.<ext>` + `.ple-q8`) needs `<model>.ple-q8` (`hipfire pull <tag> --ple q8`; `serve` downloads it on demand for registry models); the monolithic `qwen3.8:flash-next-mq4-xts` keeps its embedded Q8 rows. A sidecar is bound to its base file (base index sha256 + size) and refused on mismatch. See [`CONFIG.md`](CONFIG.md#qwen4-ple-row-tables). |
| `HIPFIRE_PLE_PREWARM_BYTES` | `auto`; `0` disables | Compat env for `memory.ple_prewarm_bytes` (`auto` or int 0–1 TiB). Idle-only page-cache prewarm of the q4 PLE table `<model>.ple-q4` after the model is ready (rank-ordered: hottest tier first, then tier 0 again; row-ordered: sequential). `auto` = 15 GiB, capped at `MemAvailable` minus the CMA region minus 8 GiB at ready and at the table; `0` = off; an explicit N is that many bytes. The pages are ordinary clean page cache, dropped (never swapped) under memory pressure and not charged in admission. See [`CONFIG.md`](CONFIG.md#qwen4-ple-row-tables). |
| `HIPFIRE_PLE_LOAD_FILL_BYTES` | `auto`; `0` disables | Compat env for `memory.ple_load_fill_bytes` (`auto` or int 0–1 TiB). Bytes of a rank-ordered q4 PLE table `<model>.ple-q4` read into the page cache while the model loads, hottest rows first, at most 0.2 GB/s, re-reading pages the weight upload's cache traffic evicted (hottest first) until ready; ready never waits for it, and the prewarm after ready continues at the end of the resident prefix. `auto` = 14 GiB; any value is capped at the table, at an explicit `memory.ple_prewarm_bytes`, and at the host RAM available at load start (`MemAvailable` minus the CMA region, plus TTM's page pool) minus the weights the load puts in host RAM minus 8 GiB; `0` = off. See [`CONFIG.md`](CONFIG.md#qwen4-ple-row-tables). |
| `HIPFIRE_PLE_HOT_PACK_FILE` | `<model>.plehotrows` | Path of the PLE hot-row pack, overriding the file next to the model. Empty or unset uses `<model file>.plehotrows`. |
| `HIPFIRE_PLE_HOT_RANK_FILE` | `<model>.plehot` | Path of the PLE hot-rank sidecar, overriding the file next to the model. Empty or unset uses `<model file>.plehot`. |
| `HIPFIRE_PLE_HOT_DEBUG` | off; `1` enables | Developer diagnostic: one stderr line per PLE fetch of at least 4,096 rows (a prefill chunk). It gives the pages and rows served from the hot-row pack, the cache hits and misses, the source read groups, the inline-read and reader counts and the timing of each phase. |
| `HIPFIRE_PLE_INLINE_FETCH` | on unless `0` | Developer kill switch for small PLE fetches (at most 64 source-page groups plus hot-pack runs): try page-cache-only `preadv2(RWF_NOWAIT)` reads inline before dispatching misses to blocking readers. Completed reads use the same row decoder and source-page cache publication. `0` restores the previous blocking/threaded fetch path. |
| `HIPFIRE_QWEN4_TAIL_SNAPSHOT` | on unless `0` | Kill switch, read once per Qwen4 bundle at assembly (needs the session cache and tokenizer `<|im_end|>` id). A cold or chunk-snapshot-restored prefill of a long prompt also splits before its last `<|im_end|>` and keeps fixed state there as a lineage-only resident (no pool rows). A request extending that final message or repeating the prompt restores it and prefills only its new rows and template. Session-exact, not cold-exact: the extra tile changes the forward schedule. Descendant regular boundaries now enter the session-exact pool, so losing the resident no longer loses the whole lineage. Turns restored from a tail do not split at their own tail. `0` restores the chunk-only schedule. |
| `HIPFIRE_QWEN4_TAIL_SNAPSHOT_MIN_GAP` | `1024` (rows) | Read once with the switch above. The tail split is taken only when the tail index lies at least this many rows above the later of the prefill start and the preceding chunk boundary (below it the replay it would save is cheaper than the extra tile). |
| `HIPFIRE_QWEN4_TAIL_SNAPSHOT_MIN_PROMPT` | `16384` (tokens) | Read once with the switch above. Prompts shorter than this never take the tail split, so short cold prompts pay no extra tile. |
| `HIPFIRE_QWEN4_TOKEN_READBACK_EARLY` | on unless `0` | Developer kill switch for Qwen4's next-token PLE readback: enqueue a pinned-host 4-byte token copy on the readback stream after it waits for argmax, then synchronize that stream once when staging the next token. `0` restores event synchronization followed by copy and stream synchronization. Token selection and vocabulary validation are unchanged. |
| `HIPFIRE_QWEN4_TURN_SNAPSHOTS` | off; `1` enables | Developer switch, read once per Qwen4 bundle (needs the session cache and a tokenizer `<|im_end|>` id). Also splits every prefill (cold, snapshot-restored and live-continued turns) right after each `<|im_end|>` (every message end) and captures a session-cache snapshot there, published at client commit and reusable across sessions like the chunk snapshots. These snapshots are session-exact, not cold-exact (a live turn's state descends from decode, and the extra split changes the cold forward schedule); their scope carries `/turns`, so they never share a domain with flag-off snapshots. Off: only prefill-chunk multiples are snapshotted and a live turn captures nothing. |
| `HIPFIRE_QWEN4_TOOL_ARG_REPLAY` | on unless `0` | Opt-out for the shared Qwen4 AR/native-MTP and dense Qwen AR/spec repair (the flag name is retained): restore original producer parameter order on the last echoed assistant tool turn of the active committed conversation, preserving typed values. Byte-identical echoes are unchanged. Accept only an exact committed-prefix extension with a nonempty suffix; parse/render doubt falls back to the original safe miss. No lexical sorting, cross-session guarantee, or retired-MTP guard change. |
| `HIPFIRE_QWEN_PROMPT_CACHE` | on unless `0` | Conversation-reuse kill switch of the Qwen3.5/3.6 and dense generation paths (LCP prompt cache, DFlash/MTP conversation plan). **No longer affects Qwen4**: Qwen4 reuse is the session cache above (`HIPFIRE_SESSION_CACHE_BYTES`). The RC3 Qwen4 prefix cache (end-of-prompt checkpoint, multi-entry radix cache, shared-turn anchors, periodic captures) is removed; only the live continuation described above remains. |
| `HIPFIRE_QWEN_RADIX_CACHE` | (removed) | Read by nothing. It selected the removed RC3 Qwen4 multi-entry radix cache; setting it has no effect. Use `HIPFIRE_SESSION_CACHE_BYTES=0` to turn Qwen4 prompt reuse (snapshots and live continuation) off. |
| `HIPFIRE_QWEN_CACHE_TRACE` | unset | Set `1` to trace Qwen3.5/3.6 and dense/AR serve-path cache decisions to stderr. Qwen4 prints one `[qwen4-session] route=… prompt=… prior=… lcp=… reused=… new=… source=live\|snapshot\|cold` line per request: `prior` is the previous turn's committed history, `lcp` its common prefix with the new canonical prompt and `reused` the tokens the session cache skips; when the prompt diverges from that history a `[qwen4-session ids]` line shows the token ids around the first difference. Tail capture prints `session cache: tail resident at P`; a plan using it prints `session cache: using tail resident at P`. The RC3 `[qwen4-radix]` / `[qwen4-prefix]` lines are gone; a skipped snapshot prints `session cache: skipped N-token snapshot (reason)` regardless of this switch. |
| `HIPFIRE_QWEN4_CACHE_QUALITY_OUT` | unset | Oracle/diagnostic only. Absolute path of a JSONL file that `hipfire_arch_qwen4::cache_quality::cache_quality_event` appends to (one compact JSON line per event, flushed per line; the variable is read once; unset or empty disables capture, a relative path warns and disables it). It is the opt-in production ID capture sink consumed by `quality_driver.py`; no daemon or generate code path calls it in this tree. |
| `HIPFIRE_CQ_SESSION_BUDGET_BYTES` | unset: unbounded | Oracle only: `u64` byte budget of the `SessionCache` the `qwen4_parity --mode cache-quality` hit-vs-cold logits oracle builds (reference parity of the session cache; also honours `HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS`). Not read by the daemon. |

Assistant-turn token splicing also stores complete call-less `length` bodies,
including open `<think>` spans, keyed by the unchanged content/tool fingerprint
and producer reasoning text. Echoing `content` and `reasoning_content` replays
the emitted IDs without a synthetic `</think>`; the history template still
appends its per-turn `<|im_end|>`. Truncated/malformed tool output, suppressed
hidden calls, stop-sequence, abort and fail-closed terminals remain unstored.
See [Prompt / assistant-turn cache](CONFIG.md#prompt--assistant-turn-cache).

For Flash-Next native MTP with Q8 GDN, short-request parking's fixed parts
occupy 34,180,608 bytes (32.60 MiB). Each token adds 21,632 bytes with Q8 QSA
or 61,568 bytes with F32 QSA, including the F32 head and amortized pooled
index rows. At the 8192-row bound this is 169 MiB or 481 MiB of rows,
201.60 MiB or 513.60 MiB total (plus retained resident buffers and host
metadata). Shorter or non-four-row-aligned requests pay per-part 256-byte
packing and pooled-row rounding; only rows valid in the parked main state
are copied. This is a copy/restore, not a numerical conversion.
Explicit F32 GDN raises the fixed copy to 118,230,528 bytes (112.75 MiB);
with F32 QSA the full 8192-row park is then 593.75 MiB.
Lineage-derived pool snapshots use a separate `/session-exact` domain, so
they do not alias cold-exact reference snapshots. Cache tracing labels a
parked-main restore `source=live`; `prior` and `lcp` still describe the
immediately preceding request, which may be the side conversation.

For Flash-Next/Qwen4 pipeline parallelism (`topology.pp` 2..=8; see
[multi-gpu.md](multi-gpu.md#flash-next-pipeline-parallelism-host-staged)),
`memory.session_cache_bytes` is an independent opportunistic device-cache budget
**per band**, not a mesh-wide reservation. Captures retain the same
free-memory/context-growth guards as Single. Every band captures its own state at
the root's common prefill boundaries; only client-committed, all-band snapshots
are reusable. Aggregate records hold pins on every local piece, are scoped to
placement and mesh/load epoch, and a missing piece makes the whole record a cold
miss. The bounded aggregate index retains at most 64 records. Resident-only
lineage/tail snapshots do not enter that index because their row streams depend
on mutable live state. A load with native MTP (`--spec mtp`) attaches no session
cache on the bands (MTP sessions are not distributed); AR loads keep theirs.
`memory.session_cache_disk_bytes` above 0 and `HIPFIRE_QWEN4_TURN_SNAPSHOTS=1` are
refused at load under PP; Single and EP session behavior is unchanged.

### Embedding (arch 17)

| Variable | Default / sense | Notes |
|---|---|---|
| `HIPFIRE_EMBED_VISION_OFFLOAD` | `1`/`on`/`true` forces the qwen3.5 embedding model's SigLIP tower into host RAM (GPU-addressable, read over PCIe); `0`/`off`/`false` forces it device-resident; unset auto-decides | The tower is ~250 MB of the arch's ~740 MB and is only touched by an image request, so it is the component to spill when the card is busy. **Auto (the default)** keeps it resident only when it fits in the VRAM free at load plus a 64 MB margin, and spills otherwise — a machine whose desktop already holds most of the card then loads the model and serves text instead of OOMing, at the cost of PCIe reads on image requests. Same tradeoff `HIPFIRE_OFFLOAD_EXEC=pcie` makes for spilled text layers; applied per-model here because the embedding arch can offload the tower whole. Read via `developer_var` in `hipfire-arch-qwen35-embed`. |
| `HIPFIRE_EMBED_GEMM` | `f32` (or `mb8`) forces the f32-activation `gemm_f16_wmma_mb8` on every projection | A/B and bit-fidelity switch for the LDS-tiled macro-tile GEMM (`gemm_f16_x_f16_wmma_lds_auto`) the encoder uses for large shapes. Measured on gfx1101 at n=4096: mb8 runs the trunk's GEMMs at ~1.6 TFLOP/s (2% of the f16 peak) because it is a register-blocked 32-thread block with no LDS staging — right for the small batches it was written for (the vision tower), wrong here; the tiled kernel reaches ~25 TFLOP/s and took the GEMM from 1227 ms to ~80 ms. Its X operand is f16, so activations narrow: ~1e-3 relative, two orders below what 4-bit weights already cost, and the embedding error is unchanged to four digits on the validation fixture. Small shapes (`out < 256`, the GQA `a`/`b` projections) and any `k % 64 != 0` keep mb8 either way. |
| `HIPFIRE_EMBED_ATTN` | `naive` = masked reference per-(query, head) kernel; `no-gemm` (or `tiled`) = the Q-tiled SIMT kernel everywhere | Attention has three paths, selected by shape and this knob. Default: the **eager decomposition** (`S = Q·Kᵀ` as a WMMA GEMM, a row softmax, then `P·V` as a second GEMM) on a wave32-WMMA arch when `n % 64 == 0` and `n >= 256` — the tiled GEMM asserts `k % 64 == 0` and the second GEMM contracts over `n` — and `head_dim % 64 == 0`. Otherwise the Q-tiled SIMT kernel, which handles any `n` and is cheap at small `n` (its cost is O(n²)). To put inputs of any length on the eager path, sequences of 256+ tokens are **padded up to a multiple of 64** (`padded_rows`); padding goes at the end, which the architecture tolerates because GDN is causal (a padded row cannot influence a real row's state), the eager full-attention layers mask the padded keys in a masked f16 row-softmax, and the Q-tiled kernel bounds its score loop by `n_valid` so the padded keys never enter the softmax — and `naive` masks them with an f32 key vector (an unmasked path would let the zeroed pad rows take a softmax share and dilute every output row). Inputs below 256 tokens are not padded and stay on the tiled kernel. Measured: `embed_medium` (2209 tokens) 390.8 -> **165.3 ms** once padded onto the eager path. `naive` reaches the masked reference kernel for A/B. Measured on gfx1101 at 4096 tokens: tiled 1056 ms -> eager 342 ms (**3.09x**), because the tiled kernel's scalar f32 FMA runs at ~2.05 TFLOP/s (6.9% of f32 peak) while the trunk's WMMA GEMM holds 5.5-6.25 TFLOP/s at exactly this shape. Costs f16 Q/K/P for the attention operands; the embedding error is unchanged to four digits (1.51e-4 eager vs 1.40e-4 tiled against the NumPy oracle). |
| `HIPFIRE_EMBED_PHASE` | `1` prints the per-phase timing of one `embed` request | `assemble \| encode \| mask+normalize \| serialize+write`, on stderr. **Its own knob, deliberately not `HIPFIRE_EMBED_DEBUG`**: that one also arms the encoder's per-layer finite `trace()`, which syncs the GPU every layer and inflated the same measurement from 334 ms to 678 ms — the first reading of it was wrong for exactly that reason. |
| `HIPFIRE_EMBED_DEBUG` | `1` traces the embedding encoder | Prints a per-layer finite-value trace of the residual stream (absmax + non-finite count) and the VRAM free/total after load. Downloads one tensor per call, so it is strictly a debugging path — it is what localized a weight-format bug that NaN'd the last rows of every projection. |

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
| `HIPFIRE_GFX1100_FA2_SPLIT_VERIFY` | Experimental, default **off** on every arch (`kernel.gfx1100_fa2_split_verify`); `1` opts in on exact gfx1100 only. Replaces the plain-Q8 K-and-V DFlash verifier's eager R4/R8 attention with the FA2 split-KV S8 route only for dense Qwen H24/NKV4/HD256 sequential batches 4..32 on **VMM KV** once the **live logical context** exceeds `HIPFIRE_FA_PERTOKEN_MIN_CTX` (4,096 by default; smaller non-zero overrides are clamped to the measured 4,096 crossover for this route). VMM-only: an opted-in load of an in-envelope model whose plain-Q8 K and V resolve to the legacy backend is refused at load (`memory.kv_backend=vmm` or `=0` to proceed), and a legacy cache is never admitted at runtime. `HIPFIRE_VERIFY_ATTN=0` remains the parent opt-out. HipGraph/retained recording additionally require the compiler-free precompiled kernel set, fully materialized fixed-address Q16 scratch, and enough `flash_partials` capacity for the S8 record layout; a deliberately small `HIPFIRE_FLASH_PARTIALS_BATCH` therefore fails closed to the established batched route. The Redline/PM4 product admission remains B=16 and inherits its existing single-GPU/state guards. Not a supported default: ON is a different-numerics route (split-KV partial/merge reduction order), not greedy-byte-identical to OFF; on gfx1100 with a 20.7K-token prompt, greedy DFlash text stays coherent but diverges from OFF after about 700 characters. Promoting it to a default would need KLD, a task tie and maintainer approval. |
| `HIPFIRE_GFX1151_FA2_TWIN` | Exact-gfx1151 twin of that FA2 fill kernel (CU mode, heaviest q-tile first, conflict-free helper V stores; bit-exact) — default ON; `=0` restores the gfx11 module |
| `HIPFIRE_PM_DECODE` | Oracle-accepted PeaceMaker twins of the five W1 decode projection modules (QKVZA, QKV, gate/up, multirow-r2, residual MQ4v2) — **auto ON on exact gfx1201** (`kernel.pm_decode`); `=0` restores the hipcc modules, which stay in the pack. Embedded images are digest-pinned and refused on mismatch; outputs are byte-identical (27B retained PM4/graph traces, MTP state, KLD pins). Other arches ignore it. |
| `HIPFIRE_GFX12_FA2_PREFILL` | GQA-fused FA2 prefill on exact gfx1201 (same Qwen NH24/NKV4/HD256 envelope) — default ON (`kernel.gfx12_fa2_prefill`); `=0` opts out toward the byte-identical incumbent |
| `HIPFIRE_GFX12_FA_PACKET` | Packet-minimal Q128 FA2 body on exact gfx1201 (same Qwen envelope as `HIPFIRE_GFX12_FA2_PREFILL`) — default ON (`kernel.gfx12_fa_packet`); `=0` opts out to the byte-identical route-N body |
| `HIPFIRE_ATTN_QRESIDENT_V2` | Bit-exact v2 schedule of the gfx1201 register-resident-Q FA2 prefill kernel (same Qwen envelope; only where `kernel.attn_qresident` selects the Q-resident route) — default ON (`kernel.attn_qresident_v2`); `=0` restores the byte-identical v1 Q-resident kernel |
| `HIPFIRE_GFX12_FA_PREP_FUSED` | Exact gfx1201 FA Q/K norm and RoPE fusion (`kernel.gfx12_fa_prep_fused`); default ON only on gfx1201, `=0` restores separate launches |
| `HIPFIRE_GFX12_FA_PREP_FP8Q` | Preconvert Q to E4M3 codes for gfx1201 Q-resident v2 attention (`kernel.gfx12_fa_prep_fp8q`); default ON only on gfx1201, `=0` retains F32 Q; requires fused prep and Q-resident v2 |
| `HIPFIRE_FA_RAGGED_PACKET` | Developer kill switch (default ON, read through `developer_bool`). Exact gfx1201 native-fp8 KV, Qwen NH24/NKV4/HD256, full fp8q eligibility (fused prep, `HIPFIRE_GFX12_FA_PREP_FP8Q`, Q-resident and Q-resident v2 attention, no tree/independent/verify scope), ordinary widened prefill (`commit_stride` exactly 512) with `512 < n <= 32768` whose tail `n % 512` is 64..=511 and whose context ends exactly at the chunk end (`start_pos + n == max_ctx_len <= physical_cap`): the chunk's fp8q prep output is attended through the whole-`n` Raw Q (code plane then scale plane) as a whole-512-row prefix launch plus a tail launch, with one whole-`n` K/V write; the A4 attention epilogue is not used for such chunks and the existing whole-`n` sigmoid output producer still runs. `=0` restores the incumbent admission exactly (F32 prep and segmented attention for ragged chunks). Aligned chunks, tails below 64 rows and every other architecture, tier or mode are unchanged. |
| `HIPFIRE_FP8_DECODE_ATTN_GQA` | Exact-gfx1201 native-fp8 **decode** attention (head_dim 256, GQA group 6, tile 128, no output gate: H2): GQA-shared flash tile (one 256-thread workgroup per kv head and 128-key tile serves its six q heads, so K/V are read once) + head-dim-split reduce (`attention_flash_fp8_e4m3_tile_gqa_gfx1201` / `attention_flash_reduce_dsplit_gfx1201`); byte-identical partials and output — default ON; `=0` restores `attention_flash_fp8_e4m3_tile` + `attention_flash_q8_0_reduce` |
| `HIPFIRE_GFX1100_DECODE_ATTN_GQA` | Exact-gfx1100 Q8_0 **decode** attention with the gated AWQ MQ-rotating epilogue (head_dim 256, GQA group 6, tile 128, full causal: H2): GQA-shared flash tile (`attention_flash_q8_0_tile_gqa_gfx1100`, one 256-thread workgroup per kv head and 128-key tile, K/V read once for the six q heads, every load issued at entry) + load-ahead reduce/gate/rotate (`attention_flash_q8_0_reduce_gated_mq_rotate_awq_dec_gfx1100`, one 1024-thread workgroup per head); byte-identical partials and output — default ON; `=0` restores `attention_flash_q8_0_tile` + `attention_flash_q8_0_reduce_gated_mq_rotate_awq_gfx1100` |
| `HIPFIRE_GFX1151_Q8_DECODE_ATTN_GQA` | Exact-gfx1151 Q8_0 **decode** attention (head_dim 256, GQA group 6, tile 128, full causal, no output gate: H2): GQA-shared flash tile (one 256-thread workgroup per kv head and 128-key tile serves its six q heads, so K/V are read once) + head-dim-split reduce (`attention_flash_q8_0_tile_gqa_gfx1151` / `attention_flash_reduce_dsplit_gfx1151`); byte-identical partials and output — default ON; `=0` restores `attention_flash_q8_0_tile` + `attention_flash_q8_0_reduce` |
| `HIPFIRE_CALIB_BF16` | Calibration-only: keep native-BF16 teachers in BF16 (`kernel.calib_force_bf16`, default off; shipped inference unaffected) |
| `HIPFIRE_GFX12_MQ4V2_FP8_GATEUP` / `_RESID` / `_QKVZA` / `_QKV` | gfx1201 FP8-WMMA MQ4v2 prefill route — default ON on exact gfx1201 (widened prefill chunk 4096 via `prefill.chunk_rows`); `=0` on any one moves that family (its GEMM and its `HIPFIRE_GFX12_FP8_STREAM` producer) to the F16 path (chunk 384). `=1` forces on; launchers stay exact-gfx1201-only, so other arches are unchanged |
| `HIPFIRE_GFX12_MQ4V2_FP8_SLABS` | Two-slab S2BT8 FP8 symbols by default; `=1` selects the single-slab symbols |
| `HIPFIRE_GFX12_MQ4V2_FP8_V2` | gfx1201 FP8-WMMA MQ4v2 staged-tile v2 route — default ON on exact gfx1201 (`kernel.gfx12_mq4v2_fp8_v2`); `=0` restores the s2bt8/BT symbols. The four family flags remain prerequisites |
| `HIPFIRE_FP8_SYMFOLD` | Developer opt-out for centered FP8-v2 GEMM twins on exact gfx1201 symmetric MQ4V2 artifacts (`mq4v2.symmetric`): default enabled when the artifact marker and v2 route are both active; `=0` restores the asymmetric v2 entries. Non-symmetric artifacts are unchanged |
| `HIPFIRE_GEMMA4_PLE_BRANCH_BATCHED_PREFILL` | Exact-arithmetic batched Gemma 4 E-series PLE branch projections (`kernel.gemma4_ple_branch_batched_prefill`, experimental) — **auto ON on exact gfx1100 and gfx1201**; other arches (gfx1101, gfx1102, gfx1151, gfx1200) off; `=0` opts out. Independent of `HIPFIRE_GEMMA4_PLE_BATCHED_PREFILL` (model projection, default off). Benchmark history: [`GEMMA4_ESERIES_GFX1100_PREFILL.md`](GEMMA4_ESERIES_GFX1100_PREFILL.md), [`GEMMA4_ESERIES_LOGIT_STABILITY.md`](GEMMA4_ESERIES_LOGIT_STABILITY.md). |
| `HIPFIRE_GFX12_MQ4V2_FP8_V2_GEOM` | v2 tile geometry: `128x128` (default on exact gfx1201, the measured pin), `64x256`, `128x64`, `256x64` (prior default, still selectable) |
| `HIPFIRE_GFX12_GDN_PRE_FUSED` | gfx1201 batched-prefill GDN preamble fusion (sigmoid+conv+qknorm 3→1, byte-exact) — default ON on exact gfx1201 (`kernel.gfx12_gdn_pre_fused`); `=0` restores the 3-launch sequence |
| `HIPFIRE_GFX1151_GDN_SCAN` | Exact-gfx1151 twin of the GDN chunk scan (`gdn_chunk_scan_gfx1151`: pipelined chunk loop, K staged transposed, one value half per 256-thread workgroup; byte-identical `out` and state) — default ON; `=0` restores `gdn_chunk_scan` |
| `HIPFIRE_V2B_A4_EPI` | Developer A4-fusion arm of the exact-gfx1151 V2B gate/up SiLU launch (eager prefill only; admitted only where the certified V2B tile is: `gate_m == up_m`, `gate_m % 256 == 0`, `batch % 128 == 0`, `k % 256 == 0`; graph capture and Redline recording keep the unfused path). Unset, `1` and any other value (default ON on exact gfx1151): the fused kernel `gemm_mq4g256v2_gate_up_silu_a4_iu4_pm_v2b_gfx1151` also writes the w_down A4 sidecar (72-byte `block_i4_128` records) into a dedicated scratch slot, so `h` is not written and the hin producer is skipped; byte-identical. `0`: opt-out, the V2B SiLU + `fused_silu_hin_rotate_mq_i4_batched` path. `retile`: the stage-1 m512 twin replaces the V2B SiLU launch (same `h`, same downstream producer). Read on every call, so one process can switch arms. `HIPFIRE_V2B_PM=0` and `HIPFIRE_F1LITE=0` also disable it; other arches are unaffected |
| `HIPFIRE_V2B_A4_PM_BUNDLE` | Developer: `=<path>` loads the A4-fusion builder bundle (`gemm_mq4g256v2_gate_up_silu_a4_iu4_pm_v2b_gfx1151.hxaco`) from a file instead of the embedded image, for every `HIPFIRE_V2B_A4_EPI` arm except `0`. Read once; an unreadable file fails every A4-fusion launch |
| `HIPFIRE_GFX12_FP8_STREAM` | gfx1201 RMSNorm+rotate producer → MQ4v2 FP8 pre-pass fusion (byte-identical `prepare_mq4v2_fp8_x_f32` outputs for the qkvza/gate_up/qkv/residual inputs; standalone pack launch disappears) — default ON on exact gfx1201 (`kernel.gfx12_fp8_stream`); `=0` opts out; other arches off. A producer runs only for a family whose `HIPFIRE_GFX12_MQ4V2_FP8_*` route is on (uniform MQ4v2) |
| `HIPFIRE_G12_NORM` | gfx1201 `_v2` RMSNorm and gated-norm int4 producers (batched sum-of-squares loads + one-reciprocal RTN codes; one wave per gated-norm group; bit-identical) — default ON on exact gfx1201 (`kernel.g12_norm`); `=0` restores the incumbent `_gfx12` symbols |
| `HIPFIRE_G12_DEC_NORM` | gfx1201 and gfx1151 decode norms as multi-workgroup grids (f32 AWQ RMSNorm+FWHT: K/256 workgroups, each redoing the row's reduction and rotating one group; out-of-place single-row `rmsnorm_f32`: n/256 workgroups; half-split partial RoPE: one workgroup per head; bit-identical) — default ON on exact gfx1201 and exact gfx1151 (`kernel.g12_dec_norm`); `=0` restores the single-workgroup launches |
| `HIPFIRE_GFX1100_DEC_NORM` | gfx1100 decode norms as multi-workgroup grids (the `HIPFIRE_G12_DEC_NORM` twins built for gfx1100: f32 AWQ RMSNorm+FWHT K/256 workgroups, out-of-place single-row `rmsnorm_f32` n/256 workgroups; bit-identical) — default ON on exact gfx1100 (`kernel.gfx1100_dec_norm`); `=0` restores the single-workgroup launches |
| `HIPFIRE_G12_A4C2` | gfx1201 int4 producers search two activation scales ({5,7}, as gfx11's `-DIU4_A4_CANDIDATES=2`) instead of RTN d = amax/7; one-pass producer-layout search, bit-identical to that flag — default ON on exact gfx1201 (`kernel.g12_a4c2`; appends `-DIU4_A4_CANDIDATES=2` to the gfx1201 JIT flags); `=0` restores RTN |
| `HIPFIRE_PREFILL_CHUNK_ROWS` | Widened ordinary-prefill chunk ceiling (`prefill.chunk_rows`; default 4096 on exact gfx1201, 512 elsewhere; explicit `HIPFIRE_PREFILL_MAX_BATCH` wins; VRAM admission may admit a smaller rung). Qwen4 (Flash-Next) reads it as its prefill chunk ceiling, rounded down to 256 rows: default 8192 on exact gfx1151, 4096 on exact gfx1201, 1536 elsewhere. The PLE row staging follows the chunk, and load admission steps down 4096/2048/1536 until the chunk scratch fits free VRAM with 1 GiB to spare |

### LFM (arch 11) — lowered forward and evidence scope

| Variable | Status | Scope |
|---|---|---|
| `HIPFIRE_LFM2_GRAPH` | LFM graph experiments | Source in LFM crate/daemon |
| `HIPFIRE_FORWARD_LOWERED` | default on (LFM and several other arches); `=0` opts out | Shared lowered forward escape hatch. For LFM: **unset or any value other than `0`** keeps the lowered path; **`=0`** forces the legacy hand loop. Qwen3.5-family single-token forwards automatically use the CPU-splice-capable hand executor when the loaded model owns CPU-executed host weights, including enabled MTP forced advances and terminal flushes; no override is required. Device-resident and PCIe-offloaded Qwen models retain the lowered route. Recorded on the sealed LFM [`admissions.yml`](admissions.yml) evidence row; **not** an automatic Redline runtime-default opt-out. |
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
| `HIPFIRE_SERVE_VMM_BATCH` | Compatibility spelling of `serve.vmm_batch` (default off, experimental): the VMM continuous-batching route for concurrent requests. Also gates the Flash-Next (qwen4, arch 16) exact AR fn-lanes route, which additionally needs `serve.continuous_batch_size >= 2` (greedy text chat only; see [SERVE.md](SERVE.md#flash-next-exact-lane-batching-qwen4-arch-16)). `0` stages no lane store and keeps the serial singleton route. |
| `HIPFIRE_EXPERIMENTAL_BUDGET_ALERT` | Research budget nudge |
| `HIPFIRE_FA_PERTOKEN_MIN_CTX` | Context length past which an exact-gfx1100, exact-gfx1201 or (opt-in) exact-gfx1151 Q8 small-batch (n = 4..32, head_dim 128/256, sequential non-tree, HIP graph capture off, retained replay recording off) attend step leaves the batched flash kernel for the multi-row tile; default `4096` on gfx1100/gfx1201, unset on gfx1151 (setting any value > 0 opts gfx1151 in), `0` disables the route. Other arches, KV modes, shapes, and semantics retain the batched route. |
| `HIPFIRE_RCCL_LIB` | Explicit `librccl.so` path, tried before the ROCm root. For distributions whose ROCm prefix does not carry RCCL (nixpkgs: `rocmtoolkit-merged` has HIP/HSA, `librccl` is a separate store path). |
| `HIPFIRE_DEVICES` / `HIPFIRE_DEVICE` / `HIPFIRE_TP` / `HIPFIRE_TP_USE_RCCL` | Multi-GPU / TP. `HIPFIRE_DEVICES` (singular alias `HIPFIRE_DEVICE`; conflicting values fail) is the compatibility spelling of `hardware.devices`: comma-separated, in logical order, each entry a PCI-order index (`rocm-smi` order, not ROCr/HIP ordinals), `gfxNNNN` (first free card of that arch; repeat for more), `GPU-<uuid>`, or PCI address `[DDDD:]BB:DD.F` (required for no-UUID cards). Resolved from the KFD topology without touching a GPU, reserved, lowered to ROCr UUIDs/ordinals plus HIP `0..N-1`, and checked against HIP's arch and PCI bus ID after init. See [multi-gpu.md](multi-gpu.md#device-selection). |
| `HIPFIRE_EMULATE_GPUS` | **Developer-only logical GPU emulation.** A successfully parsed integer `>=2` enables; missing, malformed, `0`, and `1` disable. Requested logical IDs are aliased modulo the loaded physical count; this switch does not choose the EP rank count and is not a product admission. Flash-Next (Qwen4) EP honors it too: with `topology.ep = N` (2..=8), rank `r` runs on physical device `r % count`, one visible device is enough, each rank's free VRAM is its device's free VRAM divided by the ranks aliased onto it, and same-device peer checks are skipped. |
| `HIPFIRE_EMU` | **Developer-only SKU application-capacity profile** (`developer.emu`; default unset = off; process start only, read from the startup snapshot). `<id>` is an exact lower-case registry id from `hipfire emu list` (`rx9070xt`, `rx9070`, `rx9060xt`, `rx9060xt-8g`, `rx7900xt`, `rx7800xt`, `rx7700xt`, `rx7600`, `strix-point`, `krackan`); unset, empty or `off` disables; an unknown id fails, it never silently disables. The profile caps this process's *managed application allocation* (device, pinned-host, VMM and ROCr kernarg backing) to the target SKU's memory capacity and routes the sizing queries through the same budget. It does **not** emulate CUs, clocks, power or the target ISA: the host runs at its own speed on all its CUs, and the profile is refused on a host whose detected gfx, integrated/discrete kind, physical CUs or total memory cannot stand in for it. CU masks are never used (refused on RDNA). `hipfire bench --emu <id\|off>` sets it for every bench daemon and overrides this variable, including `off`. Unrelated to `HIPFIRE_EMULATE_GPUS` (logical multi-GPU) and `HIPFIRE_EMU_BF16_H` (kernel switch). See [EMU.md](EMU.md). |
| `HIPFIRE_EMU_HOST_BYTES` / `HIPFIRE_EMU_CARVEOUT_BYTES` / `HIPFIRE_EMU_HOST_RESERVE_BYTES` / `HIPFIRE_EMU_GTT_BYTES` | **UMA scenario overrides for `HIPFIRE_EMU=strix-point\|krackan`** (`developer.emu_host_bytes`, `developer.emu_carveout_bytes`, `developer.emu_host_reserve_bytes`, `developer.emu_gtt_bytes`). Each is a checked unsigned decimal byte count; empty = default. `_HOST_BYTES` is the emulated laptop RAM H (default 32 GiB = `34359738368`), `_CARVEOUT_BYTES` the BIOS/display carve-out C (default 1 GiB = `1073741824`), `_HOST_RESERVE_BYTES` the OS/application headroom R (default 4 GiB = `4294967296`), `_GTT_BYTES` the aggregate virtual GTT limit G, which is the emulated device capacity (default 16 GiB = `17179869184`). Managed shared-host allowance S = H − C − R (default 27 GiB); `C + R < H` and `0 < G <= S` are required. Setting any of them without an active profile, or with a discrete profile, fails. These are explicit scenario assumptions, not AMD limits. They are unrelated to `HIPFIRE_VRAM_BUDGET_BYTES` (examples only). |
| `HIPFIRE_EP_PEER_ALLREDUCE_DECODE=1` | Explicit peer all-reduce selector for logical EP decode proofs; do not set it to `0`. |
| `HIPFIRE_EP_PEER_ALLREDUCE=1` | Explicit peer all-reduce selector for logical EP batched prefill/tick proofs; do not set it to `0`. |
| `HIPFIRE_TP_PEER_DIRECT` / `HIPFIRE_TP_PEER_ALTERNATING` / `HIPFIRE_TP_FULL_GRAPH` | Developer-only dense-TP2 residual reduction controls; every route computes the same rank-ordered `x + (p0 + p1)`. `HIPFIRE_TP_PEER_DIRECT=0` replaces the default symmetric direct-peer read with the rooted fold. `HIPFIRE_TP_PEER_ALTERNATING` and `HIPFIRE_TP_FULL_GRAPH` are tri-state (`1` forces on, `0` forces off, unset takes the default); **both default on for an exact gfx1201 pair and off everywhere else**, so `HIPFIRE_TP_FULL_GRAPH=0` is the gfx1201 kill switch. `HIPFIRE_TP_PEER_ALTERNATING` uses the fact that the attention and FFN partials live in different buffers: each reduction waits only on the peer's producer event, without the consumer-complete reverse wait; unset follows `HIPFIRE_TP_FULL_GRAPH`, and `=0` also disables the whole-token graph. `HIPFIRE_TP_FULL_GRAPH` admits an exact homogeneous gfx1100 pair or an exact homogeneous gfx1201 pair (TP=2, peer access, AR decode; mixed or other arches are not admitted) and captures each rank's whole decode token as one hipGraph, with device-resident signal barriers and a host-mapped coherent payload exchange; enabling it turns on the alternating protocol unless that is set to `0`. See [multi-gpu.md](multi-gpu.md#dense-tp2-reduction-protocol-developer). |
| `HIPFIRE_ALLOW_MIXED_ARCH=1` | Mixed arch pairs |
| `HIPFIRE_PP_LAYERS` | Pipeline-parallel band layer counts: `a,b,…` lists the trunk layer count of each band in **logical band order** (the first count is band 0, the first layers; the last count is the last band; it is not the card order), `pp` entries, none zero, summing to the trunk layer count. Read once at admission; unset or empty gives the uniform split (remainder on the first bands). **Flash-Next/Qwen4** (`topology.pp` 2..=8) also needs band 0 to hold at least 2 layers (the PLE layer) and runs the last band on the output card, physical card 0; a bad list is refused naming `topology.pp` and `HIPFIRE_PP_LAYERS`. The generic Qwen3.5/3.6 route (`params.pp`) skips the uniform free-VRAM delta check when it is set. See [multi-gpu.md](multi-gpu.md#band-placement-hipfire_pp_layers). |
| `HIPFIRE_TOPOLOGY_PP` | Compat env for `topology.pp` (experimental; integer `1..=8`, default `1`): the pipeline degree `hipfire run`, `serve` and `bench` send as the load's `pp`. There is **no load-time `--pp` CLI flag** (`hipfire bench --pp` is the prompt-size list). An explicit `params.pp` above `1` on a daemon load wins and this value is not read; a request of `1` or none uses it. An invalid configured value is refused naming `1..=8` only when it is the value in effect. Flash-Next/Qwen4 admits `2..=8` with `tp = ep = 1` and host-staged edges; generic Qwen3.5/3.6 HFQ uses the same degree over peer memcpys. See [multi-gpu.md](multi-gpu.md#flash-next-pipeline-parallelism-host-staged). |
| `HIPFIRE_MESH_TRANSPORT` | **Developer-only.** How the admitted pipeline mesh moves each band edge's HC streams; read once at admission, only for Flash-Next (Qwen4) `topology.pp` 2..=8: `auto` (default; also unset or empty), `host` or `peer`. `auto` and `host` are host-staged: two pinned host banks per edge, peer access is never enabled. The decision is per directed edge and made before any allocation or peer enable; there is no runtime fallback. `peer` needs pre-side-effect byte-qualification evidence for every edge; none exists yet, so `peer` is refused at admission naming `topology.pp`, and `auto` resolves to host-staged edges. Any other value is refused (`is not one of auto, peer, host`). The generic Qwen3.5/3.6 PP route does not read it: its band edges are peer memcpys. |
| `HIPFIRE_PP_PFLASH` / `HIPFIRE_PP_DFLASH` | **Experimental, generic Qwen3.5/3.6 PP only** (`pp>1`; `HIPFIRE_PP_PFLASH` is deprecated with PFlash). `=1` makes the daemon accept a PFlash compose or a DFlash `draft` that it otherwise refuses with `pp>1` (DFlash: cross-card speculative generate is not fully implemented). Neither is an admission, and neither changes Flash-Next PP, which still refuses PFlash and DFlash. |
| `HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB` | Uniform init tolerance |
| `HIPFIRE_SLOTS_PAGED` / `HIPFIRE_SLOTS_PAGED_PAGES` | Multi-slot engine: `1`/`on`/`true` selects the paged KV pool instead of fixed per-slot slabs (implied by `serve.prefix_cache`); `_PAGES` overrides the physical page count (clamped to ≥ 1). |
| `HIPFIRE_VL_SEQUENTIAL` | Multi-slot engine: `1`/`on`/`true` forces the legacy one-image-at-a-time VL prefill. Refused together with the paged pool. |
| `HIPFIRE_PAGE_EVICTION` | `0` keeps model pages in the host page cache after upload (default: evict only on UMA). Shared by the sequential loader and the multi-slot engine. |
| `HIPFIRE_BENCH_MULTI_SLOT` / `HIPFIRE_BENCH_MULTI_SLOT_SLOTS` / `HIPFIRE_BENCH_MULTI_SLOT_CTX` | `hipfire bench`: load the model on the multi-slot engine with the given slot count and per-slot context. |
| `HIPFIRE_MTP_TRACE=1` | Per-cycle MTP draft/verify trace on stderr. |
| `HIPFIRE_DEBUG_POOL_INVARIANTS=1` | Multi-slot engine: assert page-pool / slot-lease invariants after every scheduler tick (debug; slow). |
| `HIPFIRE_FAULT_HIP` / `HIPFIRE_FAULT_PREFIX_PUBLISH` / `HIPFIRE_FAULT_MTP_FULL_REJECT` | Test-only fault injection used by `test_serve_prefix_cache`: `HIP` fails one `upload`/`launch`/`sync` below the HIP bridge, `PREFIX_PUBLISH=1` fails the first prefix-cache publication, `MTP_FULL_REJECT=1` forces every MTP draft to be rejected. Never set in production. |
| `HIPFIRE_MODULE_LOAD_AUDIT=1` | Developer-only module-load audit (default off): one `HIPFIRE_MODULE_LOAD_AUDIT` stderr line per HIP module load (module, symbol, source SHA-256, recipe flags), per `hipModuleLoad`/`hipModuleLoadData` (handle, path or image FNV-1a; embedded PeaceMaker images included) and per resolved symbol, for offline route-closure audits of `kernel_registry::route_entries` and of the load-time barrier (a closed route shows every load before `loaded` and none after). Never set in production or timing runs. |

Flash-Next/Qwen4 PP native MTP is greedy-only; sampled/penalized requests
decode AR through the same PP mesh instead of failing the request. The route
decision logs once per process. This does not change Single native-MTP
sampling support or admit other speculative engines under PP.

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
four-logical-rank layout; Flash-Next (Qwen4) EP takes its rank count from
`topology.ep` (2..=8) and maps rank `r` to physical device `r % count`, with
the arch still checked (exact gfx1100 or gfx1201, one arch), same-device peer
checks skipped, and each rank's free VRAM divided by the ranks aliased onto its
device. Duplicate physical IDs remain rejected in ordinary
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
| `HIPFIRE_REPLAY_BACKEND` | `hip` / `off` / `shadow` / `auto`. Unset may select `auto` only from the automatic product defaults in `retained_redline_default`: `mq4r_redline_default` — exact GPU arch `gfx1100`/`gfx1151`/`gfx1201` + case-insensitive `.mq4r` + pp=tp=1 (model-family agnostic; no `arch_id` gate; `gfx1200` and all other arches remain opt-in); Qwen3.5 dense (`qwen3_5`, any weight format) plain-AR decode on exact `gfx1201` with pp=tp=1 and no drafter (retained PM4; byte-identical to the HIP AR graph); Qwen4 (Flash-Next, `qwen4`) plain-AR decode on exact `gfx1201` with pp=tp=1 and no drafter, i.e. an MTP-off load (retained PM4, unpaced; greedy text identical to the HIP route); and DeepSeek4 `.mq2r` AR on gfx1151. A Qwen3.5-family process with a CPU-executed partial-offload spill (`memory.offload_exec=cpu` plus a manual layer/routed-expert budget or explicit `auto`/`-1` that resolves to a spill) gets no retained default — its spilled GEMVs run on the host between GPU launches, which a retained tape does not record — and an explicit non-`hip` backend there is refused at load. Unset/null budgets never request a spill. Existing LFM `.mq4` registry evidence is **not** automatically selected because it is not `.mq4r`; any usable non-default retained route is explicit opt-in and must still prove route support. The sealed LFM [`admissions.yml`](admissions.yml) row is registry evidence/admission only and does not wire runtime defaults. Runtime default ≠ Redline certification/registry admission. Built-in `hip` config profile, another explicit backend selection, `replay.backend = "hip"` or `=hip` disables the automatic default. |
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
| `HIPFIRE_PACK_JOBS` | Kernel-pack builder only: positive worker count, bounded by CPU affinity and available RAM (~1.5 GiB/job). Defaults to `available_parallelism()` minus a reserve of at least one core (one per eight cores). `build-kernel-pack.sh` treats this as the total budget shared by concurrent architectures. |
| `HIPFIRE_JIT_JOBS` | Load-time JIT and synchronous batch compiles: positive worker count. Default `available_parallelism()` minus one core per eight (at least one); always clamped to CPU affinity and to `(MemAvailable − CMA − reserve) / 1.5 GiB`, honoring a tighter cgroup limit; unknown RAM admits one job. Reserve is 1 GiB, plus 2 GiB of loader staging headroom while compiles overlap the weight load. `1` disables concurrency but keeps the pre-ready barrier (it never restores post-ready lazy JIT). `0` or a non-integer fails the load before any GPU allocation. Not read by the pack builder (`HIPFIRE_PACK_JOBS`). |
| `HIPFIRE_ROUTE_PLAN=0` | Developer kill switch for the load-time kernel barrier: every load keeps lazy per-kernel loading (`kernels: … [no closed route plan: HIPFIRE_ROUTE_PLAN=0]`). `scripts/route_plan_walk.py` sets it so a walk observes the route's whole lazy closure, not a checked-in plan's preload. Unset or `1` uses closed plans. |
| `HIPFIRE_NO_DEVICE_COMPILER=1` | Require verified installed kernel objects instead of JIT; a missing/stale index, wrong symbol/source/flags/profile/ABI/toolchain identity or object SHA-256 fails before HIP loads it. Hot JIT keys remain toolchain-specific. |
| `HIPFIRE_*_DUMP` / `*_TRACE` / `*_PROFILE` | Diagnostic families — see inventory |


For a full indexed install, run `scripts/compile-kernels.sh gfx1201` from the
matching source revision with hipcc available, and install the resulting
`kernels/compiled/gfx1201/` directory beside the daemon executable. Native
installers and container/Nix builds perform this registry packaging directly.
Do not copy bare `.hsaco` or `.hash` files from older installations.

Pack compilation uses one compiler per worker and leaves source, flags, objects
and index contents unchanged. `hipfire-kernel-pack --print-jobs` prints the
current host budget; `HIPFIRE_PACK_JOBS=1` selects serial compilation.
For release tarballs, `scripts/build-kernel-pack.sh --tag TAG --jobs 3
gfx1201 gfx1100 gfx1151` builds up to three architectures concurrently, splitting
the total CPU/RAM budget between them rather than launching three full pools.

**Load-time kernel barrier (limited routes).** For a load whose kernel route
is closed by `kernel_registry::route_entries` — Qwen3.6-27B MQ4G256V2 XTS on
exact gfx1201 (single GPU, native fp8 K/V, Q8 DeltaNet state, AR or native
MTP), and Qwen3.8-Flash-Next on exact gfx1201 (the canonical GPTQ3 header
layout, single GPU, the walked QSA/GDN/KV-backend axes, default MTP K,
`max_seq` up to the walked 131072, AR or native MTP with n-gram), both with
default feature flags — the common loader starts the
route's missing hipcc compiles during the weight load (`HIPFIRE_JIT_JOBS`
workers), then, before the first model or load-transform dispatch, loads every
planned HIP and embedded PeaceMaker module and binds every planned symbol. The
`loaded` acknowledgment waits for weights and kernels; afterwards an unplanned
kernel request fails instead of compiling or loading between dispatches (27B),
or, on the open Flash-Next plan, loads lazily with one
`kernels: WARNING unplanned kernel …` line and a count. If
no worker fits alongside the weight load but one fits before it, the whole
plan is compiled and loaded before weights; if none fits, the load fails
before any allocation. Verified installed packs still satisfy the plan with
zero compiler processes. Every other route (other models/arches, DFlash,
DDTree, DSpark, n-gram, CASK, adaptive KV, vision, non-default flags, pp>1,
EP, a Flash-Next trunk overlay or retained Redline route) logs `kernels: no
closed route plan (...)` and keeps lazy per-kernel JIT; staging Flash-Next
fn-lanes (`serve.vmm_batch`) returns a planned load to lazy loading.
This is a cold-start/readiness change, not a steady-state speedup.

The Flash-Next plan is generated, not written by hand:
`scripts/route_plan_walk.py` drives the real dispatch through the route's
request axes (every prompt row count to 300 and the chunk/staging tails,
decode contexts to 120K, greedy and nine sampling variants, n-gram copies,
session resumes, long-context QSA) under `HIPFIRE_MODULE_LOAD_AUDIT=1`, and
`scripts/route_plan_build.py` turns the walks into
`crates/rdna-compute/src/route_plans/<family>.<arch>.tsv`. Each row records
the effective source or image SHA-256; a row that no longer matches the
registry refuses the plan (lazy loading) and fails
`cargo test -p rdna-compute --test fn_route_plan`, which also checks that
every planned HIP module is a release-pack entry. Regenerate the plan after a
change to the Flash-Next dispatch or its kernel sources.

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

### gfx1100 decode-campaign levers (developer, default off)

Evidence and certification for everything in this table:
[`docs/perf-checkpoints/2026-10-09-gfx1100-campaign-reverify.md`](perf-checkpoints/2026-10-09-gfx1100-campaign-reverify.md)
and the 2026-08-22/23 perf-checkpoint chain. Unless noted, each lever is an
explicit `=1` opt-in, off by default, and certified bit-exact on the stated
shapes before measurement.

| Variable | Notes |
|---|---|
| `HIPFIRE_FA_KVWRITE_FOLD=1` | Fold the single-token Q8_0 K/V cache write into the FA-prep epilogue (16Q/4K, head_dim 256, non-compact decode). Bit-exact (probe + greedy e2e). Decode: +0.5-1% (gfx1100 fixture record); +0.19% paired, 10/15 wins (gfx1101, 2026-10-09 verify). |
| *(removed 2026-10-09)* | The v1-only levers were discarded with the legacy quant family: the BT2 prefill family (`HIPFIRE_BT2_DISABLE` + the per-GEMM ``*_BT2_FORCE``/``KS2`` arms), the qkvza/qkv/gate_up fusednorm negative oracles, and the residual-family oracles (`RESIDUAL_PERSIST_R2`/`_SCOPE`, `RESIDUAL_LUT`, `RESIDUAL_DUALROW`). They were reachable only from the legacy HFQ4G256/MQ4G256 GEMM and residual kernels; Magnum V2 dtypes take their own GEMM/GEMV families (several with batch-tiled arms that already ship upstream). The measurements, retractions and removal rationale are recorded in [`docs/perf-checkpoints/2026-10-09-gfx1100-campaign-reverify.md`](perf-checkpoints/2026-10-09-gfx1100-campaign-reverify.md). |
| `HIPFIRE_LM_HEAD_HFQ4=1` / `HIPFIRE_LM_HEAD_HFQ3=1` / `HIPFIRE_LM_HEAD_HFQ2=1` | Research-only load-time requant of a Q8_0 output projection to HFQ{4,3,2}G256 (disk file untouched). Lossy: output diverges from the Q8_0 head's tokens. HFQ3 decode: +~10% (gfx1100 record); +13.5% (gfx1101, 2026-10-09 verify). Excluded from campaign headline numbers by scope ruling. |
| ~~`HIPFIRE_XLDS_R`~~ | **Removed 2026-10-09.** The LDS-staged activation residual probe was invalid: its kernel paired nibbles 4..7 of each lane's word with x offsets 0..3 (it needed a second `float4` at +4), so it never computed the GEMV it claimed, and its launch requested 0 bytes of dynamic LDS for a kernel that stages `k` floats in `extern __shared__`. The 2026-08 amendment-2k `-16%..-37%` table and its "BITEXACT" claim are void; see [`docs/perf-checkpoints/2026-10-09-gfx1100-campaign-reverify.md`](perf-checkpoints/2026-10-09-gfx1100-campaign-reverify.md). |
| `HIPFIRE_AWQ_NORM_WAVEGRID=1` | Multi-CU AWQ norm on gfx1100 (null result; infrastructure kept). **Not bit-exact:** on Qwen3.5-4B MQ4 the decode logit bits differ from the default route (256 greedy tokens identical; xtx, 2026-10-13). Leave unset for exact runs. |
| `HIPFIRE_GFX1101_GFX1100_CAMPAIGN=1` | **Experimental**: admit the gfx1100-certified fusions (conv scalar-prep, `fa_prep`, `fa_epilogue`, FA kvwrite fold) on gfx1101 (same RDNA3 ISA). Diagnostic only, not a certified path; `probe_fa_prep` shows 1-ulp rope-branch diffs on gfx1101. |
| `HIPFIRE_SLOW_TOKEN_LOG=1` | bench_qwen35_mq4 prints any decode token slower than 50 ms (environmental-contention detector). |

---

## Manual — tokenizer differential test (dev-only, not runtime)

Read only by `crates/hipfire-runtime/tests/hf_tokenizer_differential.rs` (see [`CONTRIBUTING.md`](../CONTRIBUTING.md#tokenizer-differential-test-hf-oracle)); the pure-Rust HF `tokenizers` oracle is a `[dev-dependencies]` entry and never ships.

| Variable | Default | Role |
|---|---|---|
| `HIPFIRE_TOKENIZER_FUZZ_CASES` | `50000` | adversarial cases per tokenizer |
| `HIPFIRE_TOKENIZER_FUZZ_SEED` | built-in | u64 seed (decimal or `0x` hex), mixed with a per-family hash |
| `HIPFIRE_TOKENIZER_DIFF_REPORT_DIR` | `<temp>/hipfire_tokenizer_differential` | JSONL/summary report directory |
| `HIPFIRE_TOKENIZER_DIFF_MIN_PROBES` | `5000` | minimisation probe budget per unique signature |
| `HIPFIRE_TOKENIZER_DIFF_MIN_LIMIT` | `2000` | max signatures minimised per family |
| `HIPFIRE_TOKENIZER_DIFF_PRINT_LIMIT` | `50` | root signatures printed per family |

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
| `memory.moe_expert_budget` | `HIPFIRE_MOE_EXPERT_BUDGET` |
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
| `topology.ep` | `HIPFIRE_TOPOLOGY_EP` |
| `hardware.uniform_vram_tolerance_gb` | `HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB` |
| `generation.loop_guard_threshold` / `generation.loop_guard_window` | `HIPFIRE_NGRAM_LOOP_THRESHOLD` / `HIPFIRE_NGRAM_WINDOW` |
| `kernel.flash_partials_batch` | `HIPFIRE_FLASH_PARTIALS_BATCH` |
| `kernel.lm_head_f16` | `HIPFIRE_LM_HEAD_F16` |
| `diagnostic.prompt_*` | matching `HIPFIRE_PROMPT_*` |
| `diagnostic.kernel.*` / `diagnostic.compiler.*` | schema-declared kernel/compiler compatibility aliases; enumerate with `hipfire config schema --json` |

---

## Manual — NPU research stack (dark, not `HIPFIRE_*`)

Read only by `peacemaker-npu`, `railgun` (feature `npu`) and `npu-tools`; nothing on the daemon/engine path reads them. See [`npu/README.md`](npu/README.md).

| Variable | Reader | Meaning |
|---|---|---|
| `NPU_FCLK_GUARD` | `railgun::npu::fclk` (npu-coop, coop27) | `require` (unset; refuse concurrent GPU+NPU work unless the Halo iGPU fabric clock is pinned), `pin` (pin for the process, restore on exit), `off` (skip, loud warning) |
| `NPU_IGPU_PCI` | `railgun::npu::fclk` (fclk sysfs device, npu-coop GPU) | PCI bus id of the Halo iGPU, default `0000:bf:00.0` (reference enumeration); other Halo systems enumerate it elsewhere (e.g. `0000:c5:00.0`) |
| `NPU_M4_CMD`, `NPU_M4_ALONE_CMD` | `npu-tools m4` | NPU loop command run concurrently with the GPU prefills (required) and alone (defaults to `NPU_M4_CMD`) in the M4 derate measurement |
| `AIE2P_VENDOR_CORPUS` | `peacemaker-npu` tests | Vendor `vendor-artifacts` directory for byte-oracle cross-checks; tests skip when unset |
| `AIE2P_OBJDUMP` | `peacemaker-npu` `isa_decode` (ignored test) | llvm-aie `llvm-objdump` used as the decode oracle (default `/tmp/aie2p-llvm-objdump`) |
| `NPU_HELLO_ARTIFACT_DIR`, `NPU_HELLO_PROGRAM` | `peacemaker-npu` sim tests | Dumped `npu-hello` artifacts for the exact-PDI hello gates; skipped when unset |
| `GOLDEN_PRINT`, `IEF15_PROBE_DUMP` | `peacemaker-npu` `golden_bytes` / `ief15_probe` tests | `GOLDEN_PRINT`: also print the computed golden rows; `IEF15_PROBE_DUMP=<path>`: write the expected probe blob |
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

`HIPFIRE_IU4_C2_ONEPASS` below is a compile-time kernel macro, not an
environment-variable control. The HIN source defines it only for gfx1100 with
an explicitly selected two-candidate quantizer; setting it in the process
environment does not enable or disable this path.

**Do not hand-edit rows below**; regenerate with `python3 scripts/check-lifecycle.py --write`.
Presence in the inventory means the token appears in source; it does **not** mean the knob is supported, stable, or admitted.

<!-- env-inventory:begin (generated by scripts/check-lifecycle.py --write) -->

**Generation method:** token scan over tracked `*.rs`, `*.py`, and `*.sh` (`scripts/check-lifecycle.py --write`).
**Columns:** variable; up to two lexical source paths; lifecycle status (see [Lifecycle status](#lifecycle-status)).
**Count:** 1648

| Variable | Example source path(s) | Lifecycle |
|---|---|---|
| `HF_ENDPOINT` | crates/hipfire-arch-qwen4/reference_oracle/rent_setup.py, crates/hipfire-cli/src/main.rs | stable |
| `HIPFIRE_6409_EVIDENCE` | crates/radiowave/src/recipes.rs | developer |
| `HIPFIRE_9B_MODEL` | scripts/bisect_9b_decode.sh | harness |
| `HIPFIRE_A4_ATTN_EPI` | crates/hipfire-arch-qwen35/src/qwen35/prefill.rs, crates/rdna-compute/src/kernel_registry.rs | developer |
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
| `HIPFIRE_ARCH` | scripts/kernel-tune-loop.sh | harness |
| `HIPFIRE_AR_GRAPH` | crates/hipfire-arch-qwen35/examples/shared_weight_state_oracle.rs, crates/hipfire-arch-qwen35/tests/pm_decode_twins_hw.rs | experimental |
| `HIPFIRE_AR_GRAPH_TRACE` | crates/hipfire-arch-qwen35/src/qwen35/forward.rs, scripts/vmm_kv_matrix.py | developer |
| `HIPFIRE_ATTENTION_REDUCE_GATED_MQ_AWQ` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_ATTENTION_REDUCE_GATED_MQ_KERNEL` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_ATTN_FLASH` | crates/hipfire-arch-qwen35/src/qwen35/forward.rs, crates/hipfire-config/src/lib.rs | stable |
| `HIPFIRE_ATTN_QRESIDENT` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | stable |
| `HIPFIRE_ATTN_QRESIDENT_V2` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | stable |
| `HIPFIRE_ATTN_TILE_SIZE` | crates/hipfire-arch-qwen35/src/qwen35/forward.rs, crates/hipfire-runtime/src/llama.rs | developer |
| `HIPFIRE_AWQ_EXPERTS` | crates/hipfire-quantize/src/pipeline.rs | developer |
| `HIPFIRE_AWQ_F1_ONLY` | crates/hipfire-quantize/src/calibration.rs, scripts/awq_alpha_sweep.sh | developer |
| `HIPFIRE_AWQ_NORM_WAVEGRID` | crates/hipfire-runtime/examples/probe_awq_wavegrid.rs, crates/rdna-compute/src/gemv.rs | developer |
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
| `HIPFIRE_BENCH_AB` | benchmarks/scripts/bench_dflash_27b_gfx906.sh | harness |
| `HIPFIRE_BENCH_BACKEND` | scripts/kernel-tune-loop.sh | harness |
| `HIPFIRE_BENCH_MAX` | scripts/adaptive_b_bench.sh, scripts/qwen36_bench.sh | harness |
| `HIPFIRE_BENCH_MAX_TOKENS` | scripts/kernel-tune-loop.sh | harness |
| `HIPFIRE_BENCH_N` | crates/rdna-compute/examples/bench_indexed_moe_keystone.rs | harness |
| `HIPFIRE_BENCH_RUNS` | scripts/adaptive_b_bench.sh, scripts/bench_qwen36_ar_dflash.sh | harness |
| `HIPFIRE_BENCH_WARMUPS` | scripts/kernel-tune-loop.sh | harness |
| `HIPFIRE_BENCH_WORKLOAD` | scripts/kernel-tune-loop.sh | harness |
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
| `HIPFIRE_BUILDER_GIT_SHA` | crates/peacemaker-isa/src/lib.rs | developer |
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
| `HIPFIRE_CASK_OFF` | crates/hipfire-arch-qwen35/tests/pm_decode_twins_hw.rs, scripts/redline_daemon_harness.py | deprecated |
| `HIPFIRE_CASK_SIDECAR` | crates/hipfire-config/src/lib.rs | deprecated |
| `HIPFIRE_CB_DFLASH_DRAFT_BATCH` | crates/hipfire-arch-qwen35/examples/cb_vmm_state_oracle.rs, crates/hipfire-arch-qwen35/src/dflash_cb.rs | developer |
| `HIPFIRE_CB_DN_SAVE_FREE` | crates/hipfire-arch-qwen35/examples/cb_vmm_state_oracle.rs, crates/hipfire-arch-qwen35/src/dflash_cb.rs | developer |
| `HIPFIRE_CB_DN_SAVE_FREE_DIAG` | crates/hipfire-arch-qwen35/src/dflash_cb.rs, crates/railgun-cert/src/recording.rs | developer |
| `HIPFIRE_CB_DRAFT_LANE_CONV` | crates/hipfire-runtime/src/dflash/lanes.rs, crates/railgun-cert/src/recording.rs | developer |
| `HIPFIRE_CB_DRAFT_LANE_CTX` | crates/hipfire-runtime/src/dflash.rs, crates/hipfire-runtime/src/dflash/lanes.rs | developer |
| `HIPFIRE_CB_DRAFT_ROWS64` | crates/hipfire-arch-qwen35/src/speculative.rs | developer |
| `HIPFIRE_CB_LANE_FA_BATCH` | crates/hipfire-arch-qwen35/src/qwen35/prefill_multi.rs | developer |
| `HIPFIRE_CB_LANE_GDN_BATCH` | crates/hipfire-arch-qwen35/src/dflash_cb.rs, crates/hipfire-arch-qwen35/src/qwen35/prefill_multi.rs | developer |
| `HIPFIRE_CB_LANE_GDN_VARROWS` | crates/hipfire-arch-qwen35/src/dflash_cb.rs, crates/hipfire-arch-qwen35/src/qwen35/prefill_multi.rs | developer |
| `HIPFIRE_CB_PHASES` | crates/hipfire-arch-qwen35/src/forward_slots/vmm/dflash.rs, crates/hipfire-generate/src/batch.rs | developer |
| `HIPFIRE_CB_SAMPLED_DRAFT_BATCH` | crates/hipfire-arch-qwen35/src/dflash_cb.rs | developer |
| `HIPFIRE_CB_SAMPLED_PACKED_HEAD` | crates/hipfire-arch-qwen35/src/dflash_cb.rs, crates/railgun-cert/src/recording.rs | developer |
| `HIPFIRE_CB_SAMPLED_TARGET_BATCH` | crates/hipfire-arch-qwen35/src/dflash_cb.rs | developer |
| `HIPFIRE_CB_SEG_TWINS` | crates/hipfire-arch-qwen35/examples/cb_vmm_state_oracle.rs, crates/hipfire-arch-qwen35/src/qwen35/prefill_multi.rs | developer |
| `HIPFIRE_CB_TAPE_COPY_BATCH` | crates/hipfire-arch-qwen35/src/dflash_cb.rs, crates/hipfire-arch-qwen35/src/qwen35/prefill_multi.rs | developer |
| `HIPFIRE_CB_UPLOAD_COALESCE` | crates/hipfire-arch-qwen35/src/forward_slots/vmm.rs, crates/hipfire-arch-qwen35/src/qwen35/prefill_multi.rs | developer |
| `HIPFIRE_CB_VERIFY_CHUNK128` | crates/hipfire-arch-qwen35/examples/cb_vmm_state_oracle.rs, crates/hipfire-arch-qwen35/src/dflash_cb.rs | developer |
| `HIPFIRE_CB_VERIFY_FP8` | crates/hipfire-arch-qwen35/examples/pm_fp8_verify_kld.rs, crates/hipfire-arch-qwen35/examples/pm_fp8_verify_oracle.rs | developer |
| `HIPFIRE_CB_VERIFY_IU4` | crates/hipfire-arch-qwen35/src/dflash_spec.rs, crates/hipfire-arch-qwen35/src/qwen35/prefill.rs | developer |
| `HIPFIRE_CB_VERIFY_PM` | crates/hipfire-arch-qwen35/examples/cb_verify_gemm_oracle.rs, crates/hipfire-arch-qwen35/examples/cb_vmm_state_oracle.rs | developer |
| `HIPFIRE_CHATML` | crates/saddle-lab/examples/probe_argmax_agreement.rs | harness |
| `HIPFIRE_CHAT_CURRENT_DATE` | crates/hipfire-arch-qwen4/examples/qwen4_greedy_ids.rs, crates/hipfire-runtime/src/prompt_frame.rs | developer |
| `HIPFIRE_CHAT_TEMPLATE_FILE` | crates/hipfire-config/src/lib.rs, crates/hipfire-runtime/examples/dump_embedded_template.rs | stable |
| `HIPFIRE_CHUNK_DUMP_DIR` | crates/hipfire-arch-qwen4/src/chunk_dump.rs, crates/hipfire-arch-qwen4/src/gpu_forward.rs | developer |
| `HIPFIRE_CHUNK_DUMP_LAYERS` | crates/hipfire-arch-qwen4/src/chunk_dump.rs | developer |
| `HIPFIRE_CHUNK_DUMP_RAW` | crates/hipfire-arch-qwen4/src/chunk_dump.rs | developer |
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
| `HIPFIRE_CONTAINER` | scripts/container-gate.sh, scripts/test_container_gate.py | harness |
| `HIPFIRE_CONTINUOUS_BATCH` | crates/hipfire-arch-qwen35/src/carrier.rs | developer |
| `HIPFIRE_CONTINUOUS_BATCH_SIZE` | crates/hipfire-config/src/lib.rs | stable |
| `HIPFIRE_CONV_QKNORM` | crates/hipfire-arch-qwen35/src/qwen35/forward.rs | developer |
| `HIPFIRE_CONV_QKNORM_SHAPE` | crates/hipfire-arch-qwen35/src/qwen35/forward.rs, crates/rdna-compute/src/norm.rs | developer |
| `HIPFIRE_CONV_SCALAR_PREP` | crates/hipfire-arch-qwen35/src/qwen35/forward.rs | developer |
| `HIPFIRE_CPU_EXEC_TRACE` | crates/hipfire-arch-qwen35/src/qwen35/prefill.rs, crates/hipfire-arch-qwen35/src/serve_engine.rs | developer |
| `HIPFIRE_CQN_BLOCK` | crates/rdna-compute/src/kernels.rs, crates/rdna-compute/src/rows_batched.rs | developer |
| `HIPFIRE_CQN_KERNEL` | crates/rdna-compute/src/kernels.rs, crates/rdna-compute/src/rows_batched.rs | developer |
| `HIPFIRE_CQN_SCALAR_PREP` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_CQ_SESSION_BUDGET_BYTES` | crates/hipfire-arch-qwen4/src/cache_quality.rs | developer |
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
| `HIPFIRE_DEVICE` | crates/hipfire-arch-qwen35/tests/mtp_takeover_fill.rs, crates/hipfire-config/src/lib.rs | developer |
| `HIPFIRE_DEVICES` | crates/hipfire-arch-qwen35/examples/prefill_ragged_oracle.rs, crates/hipfire-arch-qwen35/examples/shared_weight_state_oracle.rs | stable |
| `HIPFIRE_DFLASH_ADAPTIVE_B` | crates/hipfire-arch-qwen35/examples/cb_vmm_state_oracle.rs, crates/hipfire-arch-qwen35/src/dflash_spec.rs | developer |
| `HIPFIRE_DFLASH_BLOCK` | crates/hipfire-config/src/lib.rs | stable |
| `HIPFIRE_DFLASH_CHAT` | crates/hipfire-generate/src/ar.rs, crates/hipfire-generate/src/batch.rs | developer |
| `HIPFIRE_DFLASH_CKPT_RESUME` | crates/hipfire-arch-qwen35/examples/cb_vmm_state_oracle.rs, crates/hipfire-arch-qwen35/src/dflash_spec.rs | developer |
| `HIPFIRE_DFLASH_CTX_CAP` | crates/hipfire-arch-qwen35/examples/cb_vmm_state_oracle.rs, crates/hipfire-arch-qwen35/src/dflash_slot.rs | deprecated |
| `HIPFIRE_DFLASH_DEVICE_ACCEPT` | crates/hipfire-arch-qwen35/src/speculative.rs, crates/hipfire-arch-qwen35/tests/sampled_dflash_device_accept_hw.rs | developer |
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
| `HIPFIRE_DFLASH_ONLINE_DUMP` | crates/hipfire-runtime/examples/dflash_online_replay.rs, crates/hipfire-runtime/src/dflash_online.rs | developer |
| `HIPFIRE_DFLASH_ONLINE_HP` | crates/hipfire-runtime/src/dflash_online.rs, scripts/dflash_online_tune/online_ab.sh | developer |
| `HIPFIRE_DFLASH_ONLINE_TUNE` | crates/hipfire-runtime/examples/dflash_online_replay.rs, crates/hipfire-runtime/src/dflash.rs | developer |
| `HIPFIRE_DFLASH_ORDERED_DRAFT` | crates/hipfire-arch-qwen35/examples/cb_vmm_state_oracle.rs, crates/hipfire-arch-qwen35/examples/dflash_draft_graph_replay.rs | developer |
| `HIPFIRE_DFLASH_PREFILL_DEVICE` | crates/hipfire-arch-qwen35/src/dflash_spec.rs, crates/hipfire-arch-qwen35/tests/dflash_prefill_scatter.rs | developer |
| `HIPFIRE_DFLASH_PREFILL_SCATTER_FAST` | crates/hipfire-arch-qwen35/src/speculative.rs, crates/hipfire-arch-qwen35/tests/dflash_prefill_scatter.rs | developer |
| `HIPFIRE_DFLASH_Q8_LMHEAD_WMMA` | crates/hipfire-arch-qwen35/examples/cb_vmm_state_oracle.rs, crates/hipfire-arch-qwen35/src/speculative.rs | experimental |
| `HIPFIRE_DFLASH_REFERENCE` | scripts/dflash_ref_spec_test.py, scripts/dflash_spec_debug.py | harness |
| `HIPFIRE_DFLASH_SCATTER_ORACLE_OUT` | crates/hipfire-arch-qwen35/tests/dflash_prefill_scatter.rs | harness |
| `HIPFIRE_DFLASH_SEED_ORACLE` | crates/hipfire-arch-qwen35/src/speculative.rs, scripts/seed_oracle_collect.sh | developer |
| `HIPFIRE_DFLASH_SELECTOR_LANES` | crates/hipfire-arch-qwen35/src/speculative.rs | developer |
| `HIPFIRE_DFLASH_SPARSE_TARGET` | crates/hipfire-arch-qwen35/src/speculative.rs | developer |
| `HIPFIRE_DFLASH_TEMP_SPEC` | crates/hipfire-generate/src/ar.rs, crates/hipfire-generate/src/batch.rs | developer |
| `HIPFIRE_DFLASH_TREE` | crates/hipfire-config/src/lib.rs, crates/hipfire-runtime/src/dflash_generic.rs | experimental |
| `HIPFIRE_DFLASH_VERIFY_PM4` | crates/hipfire-arch-qwen35/src/dflash_spec.rs, crates/hipfire-generate/src/redline.rs | developer |
| `HIPFIRE_DFLASH_WINDOW` | crates/hipfire-arch-qwen35/examples/cb_vmm_state_oracle.rs, crates/hipfire-arch-qwen35/src/dflash_slot.rs | developer; `0` (legacy contiguous DFlash) deprecated, removal 0.5.0 |
| `HIPFIRE_DFLASH_ZLAB_SAFETENSORS` | scripts/dflash_spec_debug.py | harness |
| `HIPFIRE_DIR` | .agents/skills/hipfire-autoheal/triage.sh, scripts/ab-dispatch-validation.sh | harness |
| `HIPFIRE_DIVERGENCE_PREFILL` | scripts/gfx906_logit_divergence.sh | harness |
| `HIPFIRE_DIVERGENCE_TOL` | scripts/gfx906_logit_divergence.sh | harness |
| `HIPFIRE_DN_REQUANT_PER_TOKEN` | crates/hipfire-arch-qwen35/src/qwen35/prefill.rs, crates/hipfire-runtime/examples/tmp_gfx1201_chunk_exactness.rs | developer |
| `HIPFIRE_DN_SNAPSHOT_BULK_OFF` | crates/hipfire-arch-qwen35/src/speculative.rs, crates/rdna-compute/src/feature_flags.rs | developer |
| `HIPFIRE_DN_SNAPSHOT_FLIP` | crates/hipfire-arch-qwen35/examples/cb_vmm_state_oracle.rs, crates/hipfire-arch-qwen35/src/speculative.rs | developer |
| `HIPFIRE_DN_STATE_EF` | autoresearch/ar/certify/serve_runner.py, benchmarks/scripts/mq4v2_k5120_gfx1201_validate.sh | developer |
| `HIPFIRE_DOT2_GEMV` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_DOTS_FAULT` | crates/hipfire-generate/tests/vision_lifecycle_tests.rs | harness |
| `HIPFIRE_DOTS_IMAGE` | crates/hipfire-generate/tests/vision_lifecycle_tests.rs | harness |
| `HIPFIRE_DOTS_OCR_BF16_RESIDUAL` | crates/hipfire-arch-dots-ocr/src/dots_ocr.rs | developer |
| `HIPFIRE_DOTS_OCR_DUMP_DIR` | crates/hipfire-arch-dots-ocr/src/dots_ocr.rs, scripts/diff_dots_ocr_stages.py | developer |
| `HIPFIRE_DOTS_OCR_FIXTURE` | crates/hipfire-generate/tests/vision_lifecycle_tests.rs | harness |
| `HIPFIRE_DOTS_OCR_TRACE` | crates/hipfire-arch-dots-ocr/src/dots_ocr.rs | developer |
| `HIPFIRE_DPM_WARMUP_SECS` | benchmarks/scripts/mq4v2_k5120_abba.sh, crates/hipfire-arch-cohere2moe/examples/infer.rs | developer |
| `HIPFIRE_DRAFT_COLLAPSE_OFF` | crates/hipfire-arch-qwen35/examples/dflash_draft_graph_replay.rs, crates/hipfire-arch-qwen35/examples/test_dflash_draft_collapse_gfx1100.rs | developer |
| `HIPFIRE_DRAFT_F16` | crates/hipfire-config/src/lib.rs, crates/hipfire-runtime/src/config.rs | stable |
| `HIPFIRE_DRAFT_GEMM_DUMP` | crates/hipfire-config/src/lib.rs, crates/hipfire-runtime/src/config.rs | experimental |
| `HIPFIRE_DRAFT_NOISE_EMBED_BATCH` | crates/hipfire-arch-qwen35/src/speculative.rs | developer |
| `HIPFIRE_DRAFT_SHARED_CTX_KV_ROTATION` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | stable |
| `HIPFIRE_DRAFT_SHARED_FFN_ROTATION` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | stable |
| `HIPFIRE_DRAFT_SHARED_QKV_ROTATION` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | stable |
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
| `HIPFIRE_DSA_HW_TRIALS` | crates/hipfire-arch-qwen35/tests/sampled_dflash_device_accept_hw.rs | harness |
| `HIPFIRE_DSA_HW_VOCAB` | crates/hipfire-arch-qwen35/tests/sampled_dflash_device_accept_hw.rs | harness |
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
| `HIPFIRE_E1_FIXTURE_DIR` | crates/hipfire-runtime/src/e8p_i4_reference.rs | developer |
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
| `HIPFIRE_EMBED_ATTN` | crates/hipfire-arch-qwen35-embed/src/forward.rs | developer |
| `HIPFIRE_EMBED_BATCH_WINDOW_US` | crates/hipfire-arch-qwen35-embed/src/slot_engine.rs | developer |
| `HIPFIRE_EMBED_DEBUG` | crates/hipfire-arch-qwen35-embed/src/carrier.rs, crates/hipfire-arch-qwen35-embed/src/forward.rs | developer |
| `HIPFIRE_EMBED_ENGINE_TRACE` | crates/hipfire-arch-qwen35-embed/src/slot_engine.rs | developer |
| `HIPFIRE_EMBED_GEMM` | crates/hipfire-arch-qwen35-embed/src/forward.rs | developer |
| `HIPFIRE_EMBED_PHASE` | crates/hipfire-arch-qwen35-embed/examples/batch_parity.rs, crates/hipfire-generate/src/embed.rs | developer |
| `HIPFIRE_EMBED_VISION_OFFLOAD` | crates/hipfire-arch-qwen35-embed/src/lib.rs, crates/hipfire-arch-qwen35-embed/src/vision.rs | developer |
| `HIPFIRE_EMIT_TOKEN_IDS` | autoresearch/ar/certify/serve_runner.py, crates/hipfire-arch-cohere2moe/src/spec_emit.rs | developer |
| `HIPFIRE_EMU` | crates/hipfire-arch-deepseek4/src/heterogeneous.rs, crates/hipfire-arch-qwen35-embed/src/vision.rs | developer |
| `HIPFIRE_EMULATE_GPUS` | crates/hipfire-arch-qwen35/tests/route_oracle_mesh.rs, crates/hipfire-arch-qwen4/src/ep.rs | developer |
| `HIPFIRE_EMU_BF16_H` | crates/hipfire-config/src/emu.rs, crates/rdna-compute/src/gemv.rs | developer |
| `HIPFIRE_EMU_CARVEOUT_BYTES` | crates/hipfire-config/src/emu.rs | developer |
| `HIPFIRE_EMU_GTT_BYTES` | crates/hipfire-cli/src/emu.rs, crates/hipfire-config/src/emu.rs | developer |
| `HIPFIRE_EMU_HOST_BYTES` | crates/hipfire-config/src/emu.rs | developer |
| `HIPFIRE_EMU_HOST_RESERVE_BYTES` | crates/hipfire-config/src/emu.rs | developer |
| `HIPFIRE_EP_BALANCED_OWNERS` | crates/hipfire-arch-qwen4/src/ep.rs, crates/hipfire-arch-qwen4/src/gpu_forward.rs | developer |
| `HIPFIRE_EP_DECODE_TIMING` | crates/hipfire-arch-deepseek4/src/ep.rs, crates/hipfire-arch-minimax/src/forward.rs | developer |
| `HIPFIRE_EP_DUMP_IDX` | crates/hipfire-arch-deepseek4/src/ep.rs | developer |
| `HIPFIRE_EP_DUMP_POS` | crates/hipfire-arch-deepseek4/src/ep.rs | developer |
| `HIPFIRE_EP_EXPERT_STAGE` | crates/hipfire-arch-qwen4/src/gpu_forward.rs | developer |
| `HIPFIRE_EP_FAIL_RANK` | crates/hipfire-loader/src/lib.rs | developer |
| `HIPFIRE_EP_KV_MODE` | crates/hipfire-runtime/examples/ep_decode_parity.rs | harness |
| `HIPFIRE_EP_KV_SEQ` | crates/hipfire-runtime/examples/ep_decode_parity.rs | harness |
| `HIPFIRE_EP_ORACLE_DUMP` | crates/hipfire-arch-qwen35/tests/route_oracle_mesh.rs, tools/ep_bisect.py | harness |
| `HIPFIRE_EP_ORACLE_DUMP_PRE_STEP` | crates/hipfire-arch-qwen35/tests/route_oracle_mesh.rs | harness |
| `HIPFIRE_EP_OWNER_ROUTED_PREFILL` | crates/hipfire-arch-qwen4/src/ep.rs, crates/hipfire-arch-qwen4/src/gpu_forward.rs | developer |
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
| `HIPFIRE_FABRIC_ADMISSION` | crates/hipfire-arch-qwen35/src/carrier.rs, crates/hipfire-daemon/src/gpu_lock.rs | developer |
| `HIPFIRE_FAST_SAMPLE` | crates/hipfire-generate/src/ar.rs, crates/hipfire-generate/src/batch.rs | developer |
| `HIPFIRE_FAULT_HIP` | crates/hip-bridge/src/ffi.rs, crates/hipfire-runtime/examples/test_serve_prefix_cache.rs | developer |
| `HIPFIRE_FAULT_MTP_FULL_REJECT` | crates/hipfire-runtime/examples/test_serve_prefix_cache.rs | harness |
| `HIPFIRE_FAULT_PREFIX_PUBLISH` | crates/hipfire-arch-qwen35/src/serve_engine.rs, crates/hipfire-runtime/examples/test_serve_prefix_cache.rs | developer |
| `HIPFIRE_FA_BATCH_FUSE_OFF` | crates/hipfire-arch-qwen35/src/qwen35/prefill.rs, crates/rdna-compute/src/feature_flags.rs | developer |
| `HIPFIRE_FA_KVWRITE_FOLD` | crates/hipfire-arch-qwen35/src/qwen35/forward.rs, crates/railgun-cert/src/recording.rs | developer |
| `HIPFIRE_FA_PERTOKEN_MIN_CTX` | crates/hipfire-arch-qwen35/src/qwen35/prefill.rs, crates/hipfire-dispatch/src/families/attention.rs | developer |
| `HIPFIRE_FA_RAGGED_PACKET` | crates/hipfire-arch-qwen35/src/qwen35/prefill.rs | developer |
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
| `HIPFIRE_FLOOR_ORACLE` | crates/hipfire-arch-qwen4/examples/qwen4_kld.rs | harness |
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
| `HIPFIRE_FN_GDN_DENSE_SCAN` | crates/railgun-cert/src/recording.rs, crates/rdna-compute/src/feature_flags.rs | developer |
| `HIPFIRE_FN_LANE_PM_ROWS` | crates/hipfire-arch-qwen4/src/gpu_forward_lanes.rs, crates/railgun-cert/src/recording.rs | developer |
| `HIPFIRE_FOO` | crates/hipfire-cli/src/main.rs | developer |
| `HIPFIRE_FORCE_ANSWER_SECS` | scripts/test-qwen35-think-cap.sh | harness |
| `HIPFIRE_FORCE_REBUILD` | crates/hipfire-cli/src/main.rs | developer |
| `HIPFIRE_FORCE_UNFUSED` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_FORWARD_LOWERED` | crates/hipfire-arch-deepseek4/examples/ds4_longctx_probe.rs, crates/hipfire-arch-deepseek4/examples/ds4_prod_vs_parent_trace.rs | developer |
| `HIPFIRE_FORWARD_ORACLE` | crates/hipfire-dispatch/src/pipeline/superop.rs | developer |
| `HIPFIRE_FP16` | crates/hipfire-arch-qwen35/examples/cb_vmm_state_oracle.rs, crates/hipfire-arch-qwen35/src/qwen35/prefill_multi.rs | stable |
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
| `HIPFIRE_FP8_REFS_DIR` | crates/hipfire-arch-qwen4/examples/qwen4_kld.rs | harness |
| `HIPFIRE_FP8_RESIDUAL` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_FP8_ROW_SCALE_SHIFT` | crates/rdna-compute/src/gemm_fp8_verify.rs, crates/rdna-compute/src/gemv.rs | developer |
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
| `HIPFIRE_FUSED_GATE_UP_KERNEL` | crates/rdna-compute/examples/pm_decode_twins.rs, crates/rdna-compute/src/kernel_registry.rs | developer |
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
| `HIPFIRE_G3_BUILD_ID` | crates/hipfire-arch-qwen35/examples/shared_weight_state_oracle.rs | harness |
| `HIPFIRE_G942_TRUNK_GEMV` | crates/hipfire-dispatch/src/pipeline/layer_ops.rs | developer |
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
| `HIPFIRE_GDN_LANE_VARROWS` | crates/rdna-compute/src/lane_gdn.rs | developer |
| `HIPFIRE_GDN_LAYER_TABLE` | crates/rdna-compute/src/dflash_gdn_replay.rs, crates/rdna-compute/src/lane_gdn.rs | developer |
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
| `HIPFIRE_GDN_STATE_SRC` | crates/rdna-compute/src/dflash_gdn_replay.rs, crates/rdna-compute/src/lane_gdn.rs | developer |
| `HIPFIRE_GDN_TILE_ROWS` | crates/rdna-compute/src/dflash_gdn_replay.rs, crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_GDN_WAVES_PER_BLOCK` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_GEMMA4_BATCHED_EMBEDDING_PREFILL` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_GEMMA4_EAGLE` | crates/hipfire-dispatch/src/pipeline/sandwich.rs, crates/hipfire-generate/src/dense.rs | developer |
| `HIPFIRE_GEMMA4_GRAPH` | crates/hipfire-arch-gemma4/examples/infer_gemma4.rs, crates/hipfire-arch-gemma4/src/forward.rs | developer |
| `HIPFIRE_GEMMA4_LOGIT_TRACE_DIR` | crates/hipfire-generate/src/dense.rs, scripts/diag-gemma4-logit-routes.sh | developer |
| `HIPFIRE_GEMMA4_LOGIT_TRACE_FULL_STEPS` | crates/hipfire-generate/src/dense.rs, scripts/diag-gemma4-logit-routes.sh | developer |
| `HIPFIRE_GEMMA4_LOGIT_TRACE_MAX_STEPS` | crates/hipfire-generate/src/dense.rs, scripts/diag-gemma4-logit-routes.sh | developer |
| `HIPFIRE_GEMMA4_LOGIT_TRACE_TOPK` | crates/hipfire-generate/src/dense.rs | developer |
| `HIPFIRE_GEMMA4_NORM_PLUS_ONE` | crates/hipfire-arch-gemma4/src/config.rs, crates/hipfire-arch-gemma4/src/drafter.rs | developer |
| `HIPFIRE_GEMMA4_PLE_ACTIVATION_FUSED_PREFILL` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_GEMMA4_PLE_BATCHED_PREFILL` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_GEMMA4_PLE_BRANCH_BATCHED_PREFILL` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_GEMMA4_PREFILL_BATCH` | crates/hipfire-arch-gemma4/src/bundle.rs, scripts/eval_gemma4_eseries.py | developer |
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
| `HIPFIRE_GFX1100_FA2_SPLIT_VERIFY` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_GFX1100_FA_PREP` | crates/hipfire-arch-qwen35/src/qwen35/prefill.rs | developer |
| `HIPFIRE_GFX1100_GATED_NORM_V2` | crates/rdna-compute/src/gemv.rs, crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_GFX1100_MQ4V2_NINEPATH_RPB8` | crates/hipfire-runtime/examples/mq4v2_fused_parity.rs | harness |
| `HIPFIRE_GFX1100_MQ4_WIDE_PREFILL` | crates/hipfire-arch-qwen35/src/qwen35/prefill.rs | developer |
| `HIPFIRE_GFX1100_PACKED_MQ4_PREFILL` | crates/hipfire-arch-qwen35/src/qwen35/prefill.rs, crates/rdna-compute/src/feature_flags.rs | developer |
| `HIPFIRE_GFX1100_PM4_EXPERIMENTS` | crates/rdna-compute/src/replay.rs | developer |
| `HIPFIRE_GFX1100_PM_BUNDLE` | crates/rdna-compute/src/gemm.rs, crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_GFX1100_PM_GEMM` | crates/rdna-compute/src/gemm.rs, crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_GFX1100_ROUTER_W64` | crates/hipfire-dispatch/src/pipeline/moe_program.rs | developer |
| `HIPFIRE_GFX1101_GFX1100_CAMPAIGN` | crates/hipfire-arch-qwen35/src/qwen35/forward.rs, crates/rdna-compute/src/norm.rs | developer |
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
| `HIPFIRE_GFX1151_Q8_DECODE_ATTN_GQA_GEMMA` | crates/rdna-compute/examples/gemma4_kernel_f64_check.rs, crates/rdna-compute/src/attention.rs | developer |
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
| `HIPFIRE_GFX1151_REDLINE_CU_COUNT` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/replay.rs | deprecated |
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
| `HIPFIRE_GFX1151_XLDS` | crates/railgun-cert/src/recording.rs, crates/rdna-compute/src/feature_flags.rs | developer |
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
| `HIPFIRE_GFX12_FP8_STREAM` | crates/hipfire-arch-qwen35/src/qwen35/prefill.rs, crates/hipfire-config/src/lib.rs | stable |
| `HIPFIRE_GFX12_GDN_CHUNK_SCAN` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | stable |
| `HIPFIRE_GFX12_GDN_PRE_FUSED` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | stable |
| `HIPFIRE_GFX12_MQ4V2_FP8_GATEUP` | crates/hipfire-arch-qwen35/src/qwen35/prefill.rs, crates/hipfire-config/src/lib.rs | stable |
| `HIPFIRE_GFX12_MQ4V2_FP8_QKV` | crates/hipfire-arch-qwen35/src/qwen35/prefill.rs, crates/hipfire-config/src/lib.rs | stable |
| `HIPFIRE_GFX12_MQ4V2_FP8_QKVZA` | crates/hipfire-arch-qwen35/src/qwen35/prefill.rs, crates/hipfire-config/src/lib.rs | stable |
| `HIPFIRE_GFX12_MQ4V2_FP8_RESID` | crates/hipfire-arch-qwen35/src/qwen35/prefill.rs, crates/hipfire-config/src/lib.rs | stable |
| `HIPFIRE_GFX12_MQ4V2_FP8_SLABS` | crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_GFX12_MQ4V2_FP8_V2` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | stable |
| `HIPFIRE_GFX12_MQ4V2_FP8_V2_GEOM` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_GFX12_MQ4V2_GATEUP_K5120` | benchmarks/scripts/mq4v2_k5120_abba.sh, benchmarks/scripts/mq4v2_k5120_gfx1201_validate.sh | experimental |
| `HIPFIRE_GFX12_PRODUCER_QUANT_FUSED` | crates/hipfire-arch-qwen35/src/qwen35/prefill.rs, crates/hipfire-config/src/lib.rs | stable |
| `HIPFIRE_GFX12_SILU_QUANT_FUSED` | crates/hipfire-arch-qwen35/src/qwen35/prefill.rs, crates/hipfire-config/src/lib.rs | stable |
| `HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE` | crates/rdna-compute/examples/bench_vgpr_cap_sweep.rs, crates/rdna-compute/examples/mq4c_parity.rs | developer |
| `HIPFIRE_GFX12_WEIGHT_LOAD_POLICY` | benchmarks/scripts/mq4v2_k5120_gfx1201_validate.sh, crates/hipfire-config/src/lib.rs | experimental |
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
| `HIPFIRE_GPU_LAYER_BUDGET` | crates/hipfire-arch-qwen35-embed/src/vision.rs, crates/hipfire-config/src/lib.rs | stable |
| `HIPFIRE_GPU_LOCKFILE` | autoresearch/ar/swarm.py, scripts/container-gate.sh | harness |
| `HIPFIRE_GPU_LOCK_OWNER` | scripts/gpu-lock.sh, scripts/pp-gate.sh | harness |
| `HIPFIRE_GPU_LOCK_PATH` | scripts/gpu-lock.sh | harness |
| `HIPFIRE_GPU_TOPK` | crates/saddle-lab/examples/infer_qwen35.rs | harness |
| `HIPFIRE_GQA_CHUNK` | crates/rdna-compute/src/attention.rs | developer |
| `HIPFIRE_GQA_FUSED` | crates/hipfire-arch-qwen2/src/qwen2.rs, crates/hipfire-dispatch/src/families/kv_tier.rs | developer |
| `HIPFIRE_GRAPH` | benchmarks/scripts/bench_pp_gfx906.sh, crates/hipfire-arch-maple/examples/maple_decode_profile.rs | experimental |
| `HIPFIRE_GRAPH_MOE` | crates/hipfire-arch-qwen35/src/qwen35/forward.rs, crates/hipfire-arch-qwen4/tests/common/hw_identity.rs | experimental |
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
| `HIPFIRE_HFQ4G256_LDSSTAGE` | crates/hipfire-arch-qwen35/examples/cb_vmm_state_oracle.rs, crates/hipfire-arch-qwen35/src/qwen35/prefill.rs | developer |
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
| `HIPFIRE_HIPCC_EXTRA_FLAGS` | .agents/skills/hipfire-autoheal/triage.sh, benchmarks/scripts/mq4v2_k5120_gfx1201_validate.sh | experimental |
| `HIPFIRE_HIP_WAIT` | crates/rdna-compute/src/dispatch.rs | developer |
| `HIPFIRE_HOME` | benchmarks/scripts/mq4v2_k5120_gfx1201_validate.sh, crates/hipfire-cli/src/main.rs | stable |
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
| `HIPFIRE_IU4_C2_ONEPASS` | crates/rdna-compute/src/kernels.rs, scripts/test_gfx1100_xts_scope.py | developer |
| `HIPFIRE_IU4_ONEPASS` | crates/rdna-compute/src/kernels.rs, scripts/test_gfx1100_xts_scope.py | developer |
| `HIPFIRE_IU4_PREFILL` | crates/hipfire-arch-qwen35/examples/prefill_ragged_oracle.rs, crates/hipfire-arch-qwen35/src/qwen35/prefill.rs | stable |
| `HIPFIRE_IU4_RTN_RCP` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_IU4_SIDECAR` | crates/peacemaker-isa/src/kernels/iu4_v2b_a4.rs, crates/rdna-compute/src/gemv.rs | developer |
| `HIPFIRE_IU4_SLAB` | crates/rdna-compute/src/kernels.rs, crates/rdna-compute/src/scratch.rs | developer |
| `HIPFIRE_IU4_SYMFOLD` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_IU4_V2B` | crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_IU4_V2C` | crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_IU4_X5` | crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_JINJA_CHAT` | benchmarks/prompts/mtpfloor/generate.py, crates/hipfire-arch-qwen4/examples/qwen4_kld.rs | stable |
| `HIPFIRE_JINJA_TOOLS_DRAFTER` | scripts/agentic-gate-jinja-tools.sh | harness |
| `HIPFIRE_JINJA_TOOLS_MODEL` | scripts/agentic-gate-jinja-tools.sh | harness |
| `HIPFIRE_JIT_JOBS` | crates/hipfire-loader/src/lib.rs, crates/rdna-compute/src/compile_jobs.rs | developer |
| `HIPFIRE_KERNEL_CACHE` | benchmarks/scripts/mq4v2_k5120_abba.sh, benchmarks/scripts/mq4v2_k5120_gfx1201_validate.sh | stable |
| `HIPFIRE_KLD_NGL` | crates/hipfire-runtime/examples/build_kld_ref.rs, crates/hipfire-runtime/examples/eval_gguf.rs | harness |
| `HIPFIRE_KLD_TEACHER` | benchmarks/quality-baselines/harness/spe_ablation.sh | harness |
| `HIPFIRE_KV` | crates/saddle-lab/examples/oracle_xcheck.rs | harness |
| `HIPFIRE_KV_ADAPTIVE` | crates/hipfire-arch-qwen35/src/carrier.rs, crates/hipfire-config/src/lib.rs | stable |
| `HIPFIRE_KV_BACKEND` | crates/hipfire-loader/src/admission.rs, scripts/guard_gfx1201_baseline.py | developer |
| `HIPFIRE_KV_BF16` | crates/hipfire-dispatch/src/families/attention.rs, crates/hipfire-dispatch/src/families/kv_tier.rs | developer |
| `HIPFIRE_KV_FP8_E4M3` | crates/hipfire-runtime/src/kv_mode.rs, crates/rdna-compute/src/attention.rs | developer |
| `HIPFIRE_KV_MODE` | benchmarks/scripts/bench_pp_gfx906.sh, benchmarks/scripts/mq4v2_k5120_gfx1201_validate.sh | stable; `asymN`/`turbo*` values deprecated, removal 0.5.0 |
| `HIPFIRE_KV_PHYSICAL_CAP` | crates/hipfire-runtime/src/loader_api.rs | developer |
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
| `HIPFIRE_LM_HEAD_F16` | crates/hipfire-arch-qwen35/src/shared_weights.rs, crates/hipfire-cli/src/serve/complete.rs | stable |
| `HIPFIRE_LM_HEAD_HFQ2` | crates/hipfire-arch-qwen35/src/qwen35/load.rs, crates/hipfire-runtime/src/weight_backend.rs | developer |
| `HIPFIRE_LM_HEAD_HFQ3` | crates/hipfire-arch-qwen35/src/qwen35/load.rs, crates/hipfire-runtime/src/weight_backend.rs | developer |
| `HIPFIRE_LM_HEAD_HFQ4` | crates/hipfire-arch-qwen35/src/qwen35/load.rs, crates/hipfire-runtime/src/weight_backend.rs | developer |
| `HIPFIRE_LM_HEAD_OVERWRITE` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_LM_HEAD_WMMA` | crates/hipfire-arch-qwen35/examples/cb_vmm_state_oracle.rs, crates/hipfire-arch-qwen35/src/qwen35/prefill_multi.rs | stable |
| `HIPFIRE_LOAD_TIMEOUT` | docs/investigations/evidence/ds4-mi300x-cdna-test-fail/raw/a1-m0/04-profile-feed.py, docs/investigations/evidence/ds4-mi300x-cdna-test-fail/raw/a1-m0/04-run-profile-direct.sh | harness |
| `HIPFIRE_LOAD_TRACE` | crates/hipfire-arch-qwen35/src/qwen35/load.rs, crates/hipfire-arch-qwen35/src/qwen35/weights.rs | developer |
| `HIPFIRE_LOCAL` | benchmarks/scripts/mq4v2_k5120_abba.sh, crates/hipfire-cli/src/main.rs | stable |
| `HIPFIRE_LOCK_DIR` | crates/hipfire-cli/src/doctor/probe.rs, crates/hipfire-cli/src/main.rs | developer |
| `HIPFIRE_LOG` | crates/hipfire-daemon/src/process_init.rs | developer |
| `HIPFIRE_LOG_FORMAT` | crates/hipfire-daemon/src/process_init.rs, scripts/check-env-docs.py | developer |
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
| `HIPFIRE_MESH_TRANSPORT` | crates/hipfire-loader/src/admission.rs, scripts/fn-ep-gate.sh | developer |
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
| `HIPFIRE_MODULE_LOAD_AUDIT` | crates/hip-bridge/src/ffi.rs, crates/hipfire-arch-qwen4/src/mtp_spec.rs | developer |
| `HIPFIRE_MOE_AWQ` | crates/hipfire-arch-qwen35/src/qwen35/load.rs | developer |
| `HIPFIRE_MOE_COALESCED_WEIGHTS` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_MOE_CODEBOOK_BATCHED` | crates/hipfire-arch-qwen35/src/qwen35/prefill.rs, crates/hipfire-dispatch/src/families/moe.rs | developer |
| `HIPFIRE_MOE_COMBINE_NEXT_RMS` | crates/hipfire-arch-qwen35/src/qwen35/forward.rs | developer |
| `HIPFIRE_MOE_COMBINE_NEXT_RMS_RENORM` | crates/hipfire-arch-qwen35/src/qwen35/forward.rs | developer |
| `HIPFIRE_MOE_COMBINE_RMSNORM_MQ_KERNEL` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_MOE_CPU_EXPERTS` | crates/hipfire-arch-qwen35/src/qwen35/load.rs, crates/hipfire-arch-qwen35/src/qwen35/weights.rs | developer |
| `HIPFIRE_MOE_CPU_ORACLE` | crates/hipfire-arch-qwen35/src/qwen35/forward.rs, crates/hipfire-dispatch/src/cpu_exec.rs | developer |
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
| `HIPFIRE_MOE_DUMP_DIR` | crates/hipfire-arch-qwen35/src/qwen35/forward.rs, crates/hipfire-arch-qwen35/tests/route_oracle_mesh.rs | developer |
| `HIPFIRE_MOE_EXPERTS_MQ5` | crates/hipfire-quantize/src/pipeline.rs | developer |
| `HIPFIRE_MOE_EXPERTS_MQ6` | crates/hipfire-quantize/src/pipeline.rs | developer |
| `HIPFIRE_MOE_EXPERT_BUDGET` | crates/hipfire-config/src/lib.rs | stable |
| `HIPFIRE_MOE_EXPERT_STATS` | crates/hipfire-arch-qwen35/src/qwen35/forward.rs | developer |
| `HIPFIRE_MOE_EXPERT_STATS_OUT` | crates/hipfire-arch-qwen35/src/qwen35/forward.rs, crates/hipfire-runtime/examples/eval_hipfire.rs | developer |
| `HIPFIRE_MOE_GATE_UP_CPOL` | crates/rdna-compute/src/gemv.rs | developer |
| `HIPFIRE_MOE_GATE_UP_FIXED_GROUPS` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_MOE_GATE_UP_FUSED` | crates/rdna-compute/src/gemv.rs, crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_MOE_GATE_UP_KERNEL` | crates/rdna-compute/src/gemm.rs, crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_MOE_GATE_UP_LOW_VGPR` | crates/rdna-compute/src/gemv.rs, crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_MOE_GATE_UP_MIN_BLOCKS` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_MOE_GATE_UP_PAIR_VGPR` | crates/rdna-compute/src/gemv.rs, crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_MOE_GATE_UP_RANK_INTERLEAVE` | crates/rdna-compute/src/gemv.rs, crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_MOE_GATE_UP_TIGHT_GRID` | crates/rdna-compute/src/gemv.rs | developer |
| `HIPFIRE_MOE_GATE_UP_WG2` | crates/rdna-compute/src/gemv.rs | developer |
| `HIPFIRE_MOE_GATE_UP_WG_WAVES` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_MOE_GRADED` | crates/hipfire-quantize/src/pipeline.rs, crates/hipfire-quantize/src/quant_mq.rs | developer |
| `HIPFIRE_MOE_GRADED_PARITY_FIXTURE` | crates/hipfire-arch-qwen35/tests/gpu_moe_cpu_parity.rs | harness |
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
| `HIPFIRE_MOE_PARITY_FIXTURE` | crates/hipfire-arch-qwen35/src/qwen35/load.rs, crates/hipfire-arch-qwen35/tests/gpu_moe_cpu_parity.rs | developer |
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
| `HIPFIRE_MQ4G256V2_RESIDUAL_EPILOGUE` | crates/rdna-compute/examples/pm_residual_xbatch_oracle.rs, crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_MQ4G256V2_RESIDUAL_SIGMOID_SCALED_EPILOGUE` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_MQ4G256V2_XBATCH_KERNEL` | crates/rdna-compute/examples/pm_residual_xbatch_oracle.rs, crates/rdna-compute/examples/pm_xbatch_oracle.rs | developer |
| `HIPFIRE_MQ4G256V2_XBATCH_MAX` | crates/rdna-compute/examples/pm_residual_xbatch_oracle.rs, crates/rdna-compute/examples/pm_xbatch_oracle.rs | developer |
| `HIPFIRE_MQ4V2_DOWN_TIGHT_GRID` | crates/rdna-compute/src/gemv.rs | developer |
| `HIPFIRE_MQ4V2_GATEUP_K5120` | benchmarks/scripts/mq4v2_k5120_abba.sh, crates/hipfire-config/src/lib.rs | stable |
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
| `HIPFIRE_MQV2_BT_SCREEN_BITS3` | crates/rdna-compute/examples/test_mqv2_bt_gfx11.rs | harness |
| `HIPFIRE_MQV2_GFX11_SCREEN` | crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_MQV2_GFX11_WMMA` | crates/hipfire-arch-qwen35/src/qwen35/prefill.rs, crates/hipfire-runtime/src/llama.rs | developer |
| `HIPFIRE_MQ_F16_PROJECTION_OFF` | crates/rdna-compute/src/feature_flags.rs, crates/rdna-compute/src/mq_f16_producers.rs | developer |
| `HIPFIRE_MQ_F16_RESIDUAL_OFF` | crates/rdna-compute/src/feature_flags.rs | developer |
| `HIPFIRE_MQ_PROLOGUE_FUSE_OFF` | crates/rdna-compute/src/feature_flags.rs | developer |
| `HIPFIRE_MTP_AR_FLOOR` | crates/hipfire-arch-qwen4/src/mtp_lanes.rs, crates/hipfire-arch-qwen4/src/mtp_spec.rs | developer |
| `HIPFIRE_MTP_BYTE_IDENTITY_HEAD` | crates/hipfire-arch-qwen35/tests/mtp_cache_byte_identity.rs, crates/hipfire-arch-qwen35/tests/mtp_step_oracle.rs | harness |
| `HIPFIRE_MTP_BYTE_IDENTITY_MODEL` | crates/hipfire-arch-qwen35/tests/mtp_cache_byte_identity.rs, crates/hipfire-arch-qwen35/tests/mtp_step_oracle.rs | harness |
| `HIPFIRE_MTP_CPU_ORACLE_ARM` | crates/hipfire-arch-qwen35/tests/mtp_verify_graph_oracle.rs | harness |
| `HIPFIRE_MTP_CPU_ORACLE_OUT` | crates/hipfire-arch-qwen35/tests/mtp_verify_graph_oracle.rs | harness |
| `HIPFIRE_MTP_CPU_ORACLE_REF_TOKS` | crates/hipfire-arch-qwen35/tests/mtp_verify_graph_oracle.rs | harness |
| `HIPFIRE_MTP_CPU_ORACLE_REQUIRE` | crates/hipfire-arch-qwen35/tests/mtp_verify_graph_oracle.rs | harness |
| `HIPFIRE_MTP_DEVICE_TOKEN_CHAIN` | crates/hipfire-arch-qwen35/src/mtp_spec.rs | developer |
| `HIPFIRE_MTP_DRAFT_HEAD` | crates/hipfire-arch-qwen4/src/mtp_gpu.rs, crates/hipfire-arch-qwen4/tests/dev_mtp_hw.rs | developer |
| `HIPFIRE_MTP_FILL_FAST` | crates/hipfire-arch-qwen35/src/mtp_head.rs, crates/hipfire-arch-qwen35/tests/mtp_takeover_fill.rs | developer |
| `HIPFIRE_MTP_FILL_ORACLE_A` | crates/hipfire-arch-qwen35/tests/mtp_takeover_fill.rs | harness |
| `HIPFIRE_MTP_FILL_ORACLE_B` | crates/hipfire-arch-qwen35/tests/mtp_takeover_fill.rs | harness |
| `HIPFIRE_MTP_FILL_ORACLE_DIR` | crates/hipfire-arch-qwen35/tests/mtp_takeover_fill.rs | harness |
| `HIPFIRE_MTP_FILL_ORACLE_OUT` | crates/hipfire-arch-qwen35/tests/mtp_takeover_fill.rs | harness |
| `HIPFIRE_MTP_FILL_PM_GEMM` | crates/hipfire-arch-qwen35/src/mtp_head.rs, crates/hipfire-arch-qwen35/tests/mtp_takeover_fill.rs | developer |
| `HIPFIRE_MTP_FLOOR_AR_PROBE` | crates/hipfire-arch-qwen4/src/mtp_spec.rs | developer |
| `HIPFIRE_MTP_FLOOR_MIN_WINDOWS` | crates/hipfire-arch-qwen4/src/mtp_spec.rs | developer |
| `HIPFIRE_MTP_FLOOR_TRACE` | crates/hipfire-arch-qwen4/src/mtp_spec.rs | developer |
| `HIPFIRE_MTP_GPU_ACCEPT` | crates/hipfire-arch-qwen35/src/mtp_spec.rs | developer |
| `HIPFIRE_MTP_HEAD_LMHEAD_WMMA` | crates/hipfire-arch-qwen35/src/mtp_head.rs | developer |
| `HIPFIRE_MTP_IDENTITY_ARM` | crates/hipfire-arch-qwen4/tests/greedy_mtp_identity_hw.rs | harness |
| `HIPFIRE_MTP_IDENTITY_CASE` | crates/hipfire-arch-qwen4/tests/greedy_mtp_identity_hw.rs | harness |
| `HIPFIRE_MTP_IDENTITY_MODEL` | crates/hipfire-arch-qwen4/tests/greedy_mtp_identity_hw.rs, crates/hipfire-arch-qwen4/tests/mtp_takeover_fill_hw.rs | harness |
| `HIPFIRE_MTP_IDENTITY_OUT` | crates/hipfire-arch-qwen4/tests/greedy_mtp_identity_hw.rs | harness |
| `HIPFIRE_MTP_INCREMENTAL` | crates/hipfire-arch-qwen4/examples/fn_lane_oracle.rs, crates/hipfire-arch-qwen4/src/mtp_spec.rs | developer |
| `HIPFIRE_MTP_K` | crates/hipfire-arch-qwen4/tests/mtp_takeover_fill_hw.rs, crates/hipfire-config/src/lib.rs | stable |
| `HIPFIRE_MTP_MODE` | crates/hipfire-config/src/lib.rs, crates/hipfire-runtime/src/config.rs | stable |
| `HIPFIRE_MTP_NGRAM` | crates/hipfire-arch-qwen35/tests/mtp_takeover_fill.rs, crates/hipfire-arch-qwen4/tests/greedy_mtp_identity_hw.rs | stable |
| `HIPFIRE_MTP_NGRAM_COPY` | crates/hipfire-arch-qwen4/tests/greedy_mtp_identity_hw.rs, crates/hipfire-arch-qwen4/tests/sampled_mtp_distribution_hw.rs | stable |
| `HIPFIRE_MTP_NGRAM_K` | scripts/serve_harness.py | harness |
| `HIPFIRE_MTP_OWN_PREFILL` | crates/hipfire-arch-qwen35/src/mtp_spec.rs, crates/hipfire-arch-qwen35/src/mtp_speculator.rs | developer |
| `HIPFIRE_MTP_PAIRING` | crates/hipfire-arch-qwen4/src/mtp_spec.rs, crates/hipfire-arch-qwen4/tests/mtp_takeover_fill_hw.rs | developer |
| `HIPFIRE_MTP_PHASE_TIMING` | crates/hipfire-arch-qwen35/src/mtp_cb.rs, crates/hipfire-arch-qwen35/src/mtp_spec.rs | developer |
| `HIPFIRE_MTP_PROPOSAL_GRAPH` | crates/hipfire-arch-qwen35/src/mtp_spec.rs | developer |
| `HIPFIRE_MTP_P_MIN` | crates/hipfire-arch-qwen35/src/mtp_spec.rs | developer |
| `HIPFIRE_MTP_Q8_VERIFY_WMMA` | crates/hipfire-arch-qwen35/src/mtp_spec.rs | developer |
| `HIPFIRE_MTP_SAMPLED` | crates/hipfire-arch-qwen35/tests/sampled_mtp_penalty_hw.rs, crates/hipfire-arch-qwen4/src/mtp_spec.rs | stable |
| `HIPFIRE_MTP_SAMPLED_MODE` | crates/hipfire-arch-qwen4/src/mtp_spec.rs, crates/hipfire-arch-qwen4/tests/mtp_takeover_fill_hw.rs | developer |
| `HIPFIRE_MTP_SMOKE_HEAD` | crates/hipfire-arch-qwen35/examples/mtp_head_smoke.rs | harness |
| `HIPFIRE_MTP_SMOKE_TRUNK` | crates/hipfire-arch-qwen35/examples/mtp_head_smoke.rs | harness |
| `HIPFIRE_MTP_SNAPSHOT_OVERLAP` | crates/hipfire-arch-qwen35/src/mtp_spec.rs | developer |
| `HIPFIRE_MTP_TAKEOVER_FILL_ARM` | crates/hipfire-arch-qwen4/tests/mtp_takeover_fill_hw.rs | harness |
| `HIPFIRE_MTP_TAPE_REPLAY` | crates/hipfire-arch-qwen35/src/mtp_spec.rs | developer |
| `HIPFIRE_MTP_TRACE` | crates/hipfire-arch-qwen35/src/mtp_spec.rs, crates/hipfire-arch-qwen4/src/mtp_spec.rs | developer |
| `HIPFIRE_MTP_VERIFY_DECOUPLE` | crates/hipfire-arch-qwen35/src/qwen35/prefill.rs | developer |
| `HIPFIRE_MTP_VERIFY_GRAPH_ORACLE_ARM` | crates/hipfire-arch-qwen35/tests/mtp_verify_graph_oracle.rs | harness |
| `HIPFIRE_MTP_VERIFY_GRAPH_ORACLE_OUT` | crates/hipfire-arch-qwen35/tests/mtp_verify_graph_oracle.rs | harness |
| `HIPFIRE_MTP_VERIFY_GRAPH_ORACLE_OUT_ROOT` | crates/hipfire-arch-qwen35/tests/mtp_verify_graph_oracle.rs | harness |
| `HIPFIRE_MUSE_QKVG_ALLOW_BITDIFF` | crates/rdna-compute/examples/test_gemm_qkvza_hfq4g256.rs | harness |
| `HIPFIRE_MUSE_QKVG_TOL_ABS` | crates/rdna-compute/examples/test_gemm_qkvza_hfq4g256.rs | harness |
| `HIPFIRE_MUSE_QKVG_TOL_REL` | crates/rdna-compute/examples/test_gemm_qkvza_hfq4g256.rs | harness |
| `HIPFIRE_NGRAM_DRAFT` | crates/hipfire-arch-qwen2/src/spec_impl.rs, crates/hipfire-config/src/lib.rs | stable |
| `HIPFIRE_NGRAM_DRAFT_K` | crates/hipfire-config/src/lib.rs, crates/hipfire-runtime/src/config.rs | stable |
| `HIPFIRE_NGRAM_LOOP_THRESHOLD` | crates/hipfire-config/src/lib.rs, crates/hipfire-generate/src/ar.rs | stable |
| `HIPFIRE_NGRAM_MIN_COUNT` | crates/hipfire-config/src/lib.rs, crates/hipfire-runtime/src/config.rs | stable |
| `HIPFIRE_NGRAM_MOD_N_MATCH` | crates/hipfire-arch-qwen4/tests/greedy_mtp_identity_hw.rs, crates/hipfire-config/src/lib.rs | developer |
| `HIPFIRE_NGRAM_MOD_N_MAX` | crates/hipfire-arch-qwen4/tests/greedy_mtp_identity_hw.rs, crates/hipfire-config/src/lib.rs | developer |
| `HIPFIRE_NGRAM_MOD_N_MIN` | crates/hipfire-arch-qwen4/tests/greedy_mtp_identity_hw.rs, crates/hipfire-config/src/lib.rs | developer |
| `HIPFIRE_NGRAM_WINDOW` | crates/hipfire-config/src/lib.rs, crates/hipfire-generate/src/ar.rs | stable |
| `HIPFIRE_NORMALIZE_PROMPT` | crates/hipfire-config/src/lib.rs, crates/hipfire-runtime/examples/build_kld_ref_native_gemma4.rs | stable |
| `HIPFIRE_NO_DEVICE_COMPILER` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/examples/mtp_q8_det_byte_oracle.rs | experimental |
| `HIPFIRE_NO_Q8_ROUTER` | crates/hipfire-quantize/src/pipeline.rs | developer |
| `HIPFIRE_NO_REGISTRY_FETCH` | crates/hipfire-cli/tests/run_stats.rs, crates/hipfire-config/src/lib.rs | stable |
| `HIPFIRE_NO_SPILL` | crates/hipfire-quantize/src/pipeline.rs | developer |
| `HIPFIRE_NPU_SPILLOVER` | crates/hipfire-config/src/lib.rs, crates/hipfire-runtime/src/config.rs | experimental |
| `HIPFIRE_NPU_TRUNK_TEST_MODEL` | crates/hipfire-arch-qwen4/tests/npu_trunk_model_sanity.rs | harness |
| `HIPFIRE_OFFLOAD_DEBUG` | crates/rdna-compute/src/dispatch.rs | developer |
| `HIPFIRE_OFFLOAD_EXEC` | crates/hipfire-arch-qwen35-embed/src/vision.rs, crates/hipfire-arch-qwen35/tests/gpu_gemv_parity.rs | stable |
| `HIPFIRE_OFFLOAD_PASSBACK_SHARE` | crates/hipfire-config/src/lib.rs | stable |
| `HIPFIRE_OOM_GUARD` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/kv_slots.rs | stable |
| `HIPFIRE_ORACLE_DIVERGE` | crates/hipfire-arch-qwen35/tests/route_oracle_mesh.rs | harness |
| `HIPFIRE_ORACLE_DUMP` | crates/saddle-lab/examples/tmp_gemm_v2_oracle.rs | harness |
| `HIPFIRE_ORACLE_KV_Q8` | crates/hipfire-runtime/examples/gemma4_oracle.rs | harness |
| `HIPFIRE_ORACLE_MAX` | scripts/seed_oracle_collect.sh | harness |
| `HIPFIRE_ORACLE_STATE_FP32` | crates/hipfire-arch-qwen35/tests/route_oracle_mesh.rs | harness |
| `HIPFIRE_ORNITH15_MODEL` | scripts/coherence-gate-ornith15.sh | deprecated |
| `HIPFIRE_ORNITH_FIXTURE` | crates/hipfire-arch-qwen35/src/qwen35/load.rs, crates/hipfire-arch-qwen35/tests/route_oracle_mesh.rs | developer |
| `HIPFIRE_PACK_JOBS` | crates/rdna-compute/src/bin/hipfire-kernel-pack.rs, scripts/build-kernel-pack.sh | developer |
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
| `HIPFIRE_PINNED_XFER_DEFAULT` | crates/hip-bridge/src/ffi.rs | developer |
| `HIPFIRE_PLE_FORMAT` | crates/hipfire-arch-qwen4/examples/fn_lane_oracle.rs, crates/hipfire-arch-qwen4/examples/qwen4_parity.rs | stable |
| `HIPFIRE_PLE_HOT_DEBUG` | crates/hipfire-config/src/lib.rs, crates/hipfire-runtime/src/external_rows.rs | developer |
| `HIPFIRE_PLE_HOT_PACK_FILE` | crates/hipfire-config/src/lib.rs | developer |
| `HIPFIRE_PLE_HOT_RANK_FILE` | crates/hipfire-config/src/lib.rs, crates/hipfire-runtime/src/ple_convert.rs | developer |
| `HIPFIRE_PLE_HOT_STORE_BYTES` | crates/hipfire-config/src/lib.rs, scripts/perf_config_receipt.py | stable |
| `HIPFIRE_PLE_INLINE_FETCH` | crates/hipfire-runtime/src/external_rows.rs | developer |
| `HIPFIRE_PLE_LOAD_FILL_BYTES` | crates/hipfire-config/src/lib.rs | stable |
| `HIPFIRE_PLE_PREWARM_BYTES` | crates/hipfire-config/src/lib.rs | stable |
| `HIPFIRE_PM4_KERNARG_POOL` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/replay.rs | experimental |
| `HIPFIRE_PM_DECODE` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/examples/draft_lane_rows_oracle.rs | experimental |
| `HIPFIRE_PORT` | crates/hipfire-config/src/lib.rs | stable |
| `HIPFIRE_POST_LATCH_ANSWER_TOKENS` | crates/hipfire-generate/src/ar.rs, crates/hipfire-generate/src/qwen.rs | developer |
| `HIPFIRE_PP_DEVICES` | crates/hipfire-generate/tests/qwen35_reset_hw.rs | harness |
| `HIPFIRE_PP_DFLASH` | crates/hipfire-daemon/src/main.rs | developer |
| `HIPFIRE_PP_GATE_HETEROGENEOUS` | scripts/pp-gate.sh | harness |
| `HIPFIRE_PP_GATE_INCLUDE_IGPU` | scripts/pp-gate.sh | harness |
| `HIPFIRE_PP_GATE_MODEL` | scripts/pp-gate.sh | harness |
| `HIPFIRE_PP_GATE_REQUIRE_SYSFS` | scripts/pp-gate.sh | harness |
| `HIPFIRE_PP_GATE_TEST_NO_SYSFS` | scripts/pp-gate.sh | harness |
| `HIPFIRE_PP_LAYERS` | crates/hipfire-arch-qwen4/src/admission.rs, crates/hipfire-loader/examples/fn_ep_oracle.rs | developer |
| `HIPFIRE_PP_PARITY_MODEL` | crates/hipfire-arch-qwen35/tests/pp_parity.rs | harness |
| `HIPFIRE_PP_PFLASH` | crates/hipfire-daemon/src/main.rs | deprecated |
| `HIPFIRE_PP_RESET_MODEL` | crates/hipfire-generate/tests/qwen35_reset_hw.rs | harness |
| `HIPFIRE_PREFILL_BATCHED` | crates/hipfire-arch-qwen35/src/qwen35/prefill.rs, crates/hipfire-config/src/lib.rs | stable |
| `HIPFIRE_PREFILL_CHUNK` | crates/hipfire-runtime/examples/ep_decode_parity.rs | harness |
| `HIPFIRE_PREFILL_CHUNK_ROWS` | crates/hipfire-arch-qwen35/examples/cb_vmm_state_oracle.rs, crates/hipfire-arch-qwen35/examples/prefill_ragged_oracle.rs | stable |
| `HIPFIRE_PREFILL_MAX_BATCH` | crates/hipfire-arch-qwen35/examples/prefill_ragged_oracle.rs, crates/hipfire-arch-qwen35/src/qwen35/ep_batch.rs | developer |
| `HIPFIRE_PREFILL_PM_INVARIANT` | crates/hipfire-arch-qwen4/examples/fn_prefill_g0.rs, crates/hipfire-arch-qwen4/src/gpu_forward.rs | developer |
| `HIPFIRE_PREFILL_REUSE_PBS` | crates/hipfire-arch-qwen35/src/qwen35/forward.rs, crates/hipfire-arch-qwen35/src/speculative.rs | developer |
| `HIPFIRE_PROBE_ALIGN` | crates/hipfire-runtime/src/hfq.rs | developer |
| `HIPFIRE_PROBE_MODEL` | crates/hipfire-arch-qwen4/src/artifact.rs, crates/hipfire-runtime/src/hfq.rs | developer |
| `HIPFIRE_PROBE_READS` | crates/hipfire-runtime/src/hfq.rs | developer |
| `HIPFIRE_PROBE_ROWS` | crates/hipfire-runtime/src/ple_q4.rs, scripts/check-env-docs.py | developer |
| `HIPFIRE_PROFILE` | crates/hipfire-arch-diffusion/examples/gpu_flux_forward.rs, crates/hipfire-arch-diffusion/src/flux_gpu.rs | developer |
| `HIPFIRE_PROFILE_CYCLES` | crates/hipfire-runtime/examples/dflash_spec_demo.rs, crates/saddle-lab/examples/mtp_only_demo.rs | harness |
| `HIPFIRE_PROFILE_DECODE` | crates/hipfire-runtime/examples/bench_qwen35_mq4.rs, scripts/kernel-tune-loop.sh | harness |
| `HIPFIRE_PROFILE_MAX` | scripts/ddtree_verify_profile.sh | harness |
| `HIPFIRE_PROFILE_RUNS` | scripts/ddtree_verify_profile.sh | harness |
| `HIPFIRE_PROFILE_SELF` | docs/investigations/evidence/ds4-mi300x-cdna-test-fail/raw/a1-m0/04-profile-feed.py | harness |
| `HIPFIRE_PROMPT` | docs/investigations/evidence/ds4-mi300x-cdna-test-fail/raw/a1-m0/04-profile-feed.py, docs/investigations/evidence/ds4-mi300x-cdna-test-fail/raw/a1-m0/04-run-profile-direct.sh | harness |
| `HIPFIRE_PROMPT_CACHE_CAP` | crates/hipfire-config/src/lib.rs, crates/hipfire-loader/src/lib.rs | stable |
| `HIPFIRE_PROMPT_CACHE_UNBOUNDED` | crates/hipfire-config/src/lib.rs, crates/hipfire-loader/src/lib.rs | experimental |
| `HIPFIRE_PROMPT_FILE` | scripts/kernel-tune-loop.sh | harness |
| `HIPFIRE_PROMPT_HEAT_JSON` | crates/hipfire-config/src/lib.rs, crates/hipfire-runtime/src/config.rs | experimental |
| `HIPFIRE_PROMPT_HEAT_LIMIT` | crates/hipfire-config/src/lib.rs, crates/hipfire-runtime/src/config.rs | experimental |
| `HIPFIRE_PROMPT_TOKEN_HEAT` | crates/hipfire-config/src/lib.rs, crates/hipfire-runtime/src/config.rs | experimental |
| `HIPFIRE_Q8_BATCHED_LEGACY` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_Q8_CLASSES` | crates/hipfire-quantize/src/diagnostics.rs, crates/hipfire-quantize/src/model_filter.rs | developer |
| `HIPFIRE_Q8_FLASH_TILE` | crates/hipfire-arch-maple/src/maple.rs, crates/rdna-compute/src/attention.rs | developer |
| `HIPFIRE_Q8_PREFILL_GQA_WMMA` | crates/rdna-compute/src/attention.rs | developer |
| `HIPFIRE_Q8_PREFILL_WMMA` | crates/hipfire-arch-qwen35/src/forward_slots.rs, crates/hipfire-arch-qwen35/src/qwen35/prefill.rs | developer |
| `HIPFIRE_QA_KV_MODES` | crates/saddle-lab/examples/test_inferenceQA.rs | harness |
| `HIPFIRE_QCAL_BREAKDOWN` | crates/hipfire-runtime/tests/hf_tokenizer_encode_bench.rs | harness |
| `HIPFIRE_QKVZA_BLOCK_SIZE` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_QKVZA_CPOL` | crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_QKVZA_KERNEL_NAME` | crates/rdna-compute/examples/pm_decode_twins.rs, crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_QKVZA_MIN_BLOCKS_PER_CU` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_QKVZA_SCALAR_PREP` | crates/hipfire-arch-qwen35/src/qwen35/forward.rs, crates/rdna-compute/examples/pm_decode_twins.rs | developer |
| `HIPFIRE_QKVZA_SPLIT_TAIL` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_QKVZA_WAVES_PER_BLOCK` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_QKV_KERNEL_NAME` | crates/rdna-compute/examples/pm_decode_twins.rs, crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_QKV_WITH_BIAS` | crates/rdna-compute/examples/pm_decode_twins.rs, crates/rdna-compute/src/kernels.rs | developer |
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
| `HIPFIRE_QWEN36_MOE_FIXTURE` | crates/hipfire-arch-qwen35/tests/route_oracle_single.rs | harness |
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
| `HIPFIRE_QWEN4_APPEND_WMMA` | crates/hipfire-arch-qwen4/src/gpu_forward.rs, crates/rdna-compute/src/dispatch.rs | developer |
| `HIPFIRE_QWEN4_APPEND_WMMA_MIN_ROWS` | crates/rdna-compute/src/prefill_invariant.rs | developer |
| `HIPFIRE_QWEN4_CACHE_QUALITY_OUT` | crates/hipfire-arch-qwen4/src/cache_quality.rs | developer |
| `HIPFIRE_QWEN4_CAPACITY_BYTES` | crates/hipfire-quantize/src/qwen4.rs | developer |
| `HIPFIRE_QWEN4_COMPACT_SCRATCH` | crates/hipfire-arch-qwen4/src/gpu_forward.rs | developer |
| `HIPFIRE_QWEN4_CUT_LAYERS` | crates/hipfire-arch-qwen4/src/gpu_forward.rs, crates/hipfire-loader/examples/fn_ep_oracle.rs | developer |
| `HIPFIRE_QWEN4_DECODE_HEAD_FUSE` | crates/railgun-cert/src/recording.rs, crates/rdna-compute/src/feature_flags.rs | developer |
| `HIPFIRE_QWEN4_DECODE_ZBA_X3` | crates/railgun-cert/src/recording.rs, crates/rdna-compute/src/feature_flags.rs | developer |
| `HIPFIRE_QWEN4_DEV_ARTIFACT` | crates/hipfire-arch-qwen4/tests/dev_mtp_hw.rs, crates/hipfire-arch-qwen4/tests/dev_profiles_hw.rs | harness |
| `HIPFIRE_QWEN4_DEV_EXPECT_REFUSAL` | crates/hipfire-arch-qwen4/tests/dev_mtp_hw.rs, crates/hipfire-arch-qwen4/tests/dev_profiles_hw.rs | harness |
| `HIPFIRE_QWEN4_DEV_MTP_REF` | crates/hipfire-arch-qwen4/tests/dev_mtp_hw.rs | harness |
| `HIPFIRE_QWEN4_DEV_PLE_ARTIFACT` | crates/hipfire-arch-qwen4/tests/dev_ple_rows_cpu.rs | harness |
| `HIPFIRE_QWEN4_DEV_REF` | crates/hipfire-arch-qwen4/tests/dev_profiles_hw.rs | harness |
| `HIPFIRE_QWEN4_DEV_REPORT` | crates/hipfire-arch-qwen4/tests/dev_mtp_hw.rs, crates/hipfire-arch-qwen4/tests/dev_profiles_hw.rs | harness |
| `HIPFIRE_QWEN4_EP` | crates/hipfire-arch-qwen4/src/mtp_spec.rs, crates/hipfire-config/src/lib.rs | developer |
| `HIPFIRE_QWEN4_EP_TRANSPORT` | crates/hipfire-runtime/examples/ep_transport_oracle.rs | harness |
| `HIPFIRE_QWEN4_EXPERT_HOTSET` | crates/hipfire-arch-qwen4/src/expert_hotset.rs, crates/hipfire-arch-qwen4/src/expert_residency.rs | developer |
| `HIPFIRE_QWEN4_EXPERT_STAGE` | crates/hipfire-arch-qwen4/src/gpu_forward.rs, crates/railgun-cert/src/recording.rs | developer |
| `HIPFIRE_QWEN4_EXPERT_STAGE_CAPTURE` | crates/hipfire-arch-qwen4/src/gpu_forward.rs | developer |
| `HIPFIRE_QWEN4_EXPERT_STAGE_MIN_ROWS` | crates/hipfire-arch-qwen4/src/expert_residency.rs, crates/hipfire-arch-qwen4/src/gpu_forward.rs | developer |
| `HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS` | crates/hip-bridge/src/ffi.rs, crates/hipfire-arch-qwen4/examples/fn_lane_oracle.rs | developer |
| `HIPFIRE_QWEN4_F16_WMMA` | crates/hipfire-arch-qwen4/examples/qwen4_qsa_arena_check.rs, crates/hipfire-arch-qwen4/examples/qwen4_qsa_ctx.rs | developer |
| `HIPFIRE_QWEN4_F16_WMMA_GFX1201` | crates/rdna-compute/examples/bench_qwen4_hc_wmma.rs, crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_QWEN4_GDN_CONV_QKNORM` | crates/hipfire-dispatch/src/pipeline/layer_ops.rs, crates/rdna-compute/src/feature_flags.rs | developer |
| `HIPFIRE_QWEN4_GDN_PIPE` | crates/hipfire-runtime/examples/qwen4_tierc/gdn.rs, crates/rdna-compute/src/tensor_ops.rs | developer |
| `HIPFIRE_QWEN4_GDN_PREFILL` | crates/hipfire-arch-qwen4/examples/fn_prefill_g0.rs, crates/rdna-compute/src/prefill_invariant.rs | developer |
| `HIPFIRE_QWEN4_GDN_Q8_INLINE` | crates/rdna-compute/src/feature_flags.rs, crates/rdna-compute/src/kernel_registry.rs | developer |
| `HIPFIRE_QWEN4_HC_DOWN_TILE` | crates/railgun-cert/src/recording.rs, crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_QWEN4_HC_FUSE` | crates/hipfire-dispatch/src/pipeline/layer_ops.rs, crates/hipfire-dispatch/src/pipeline/moe_program.rs | developer |
| `HIPFIRE_QWEN4_HC_ROW_FOLD` | crates/hipfire-dispatch/src/pipeline/layer_ops.rs, crates/hipfire-dispatch/src/pipeline/moe_program.rs | developer |
| `HIPFIRE_QWEN4_HC_UP_TILE` | crates/rdna-compute/src/feature_flags.rs, crates/rdna-compute/src/tensor_ops.rs | developer |
| `HIPFIRE_QWEN4_HC_UP_TWIN` | crates/rdna-compute/src/tensor_ops.rs | developer |
| `HIPFIRE_QWEN4_MOE_COMBINE_ZINIT` | crates/hipfire-dispatch/src/pipeline/moe_program.rs, crates/hipfire-dispatch/src/pipeline/qt44_qt53_prefill.rs | developer |
| `HIPFIRE_QWEN4_MOE_DOWN_TWIN` | crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_QWEN4_MOE_GATEUP_TWIN` | crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_QWEN4_MOE_SYM_DOWN_RR` | crates/rdna-compute/examples/qwen4_moe_sym.rs, crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_QWEN4_MOE_SYM_IU4` | crates/hipfire-arch-qwen4/src/gpu_forward.rs, crates/hipfire-dispatch/src/pipeline/qt44_qt53_prefill.rs | developer |
| `HIPFIRE_QWEN4_MOE_SYM_PM` | crates/rdna-compute/examples/qwen4_moe_sym.rs, crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_QWEN4_MOE_X_LDS` | crates/rdna-compute/examples/qwen4_moe_f16_lds_oracle.rs, crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_QWEN4_MQ4_ROWS_PM` | crates/rdna-compute/examples/mq4_rows_oracle.rs, crates/rdna-compute/src/gemv.rs | developer |
| `HIPFIRE_QWEN4_MQ4_X4_PM` | crates/hipfire-arch-qwen4/tests/lm_head_long_output_hw.rs, crates/railgun-cert/src/recording.rs | developer |
| `HIPFIRE_QWEN4_MQ6_X4_GFX1201` | crates/rdna-compute/src/feature_flags.rs, crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_QWEN4_MQ6_X4_PM` | crates/rdna-compute/src/gemm.rs, crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_QWEN4_MQ6_X4_REGIONS` | crates/rdna-compute/src/feature_flags.rs, crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_QWEN4_MQ6_X4_TILE` | crates/rdna-compute/src/feature_flags.rs, crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_QWEN4_MTP_APPEND_DET` | crates/hipfire-arch-qwen4/src/mtp_gpu.rs, crates/hipfire-arch-qwen4/src/mtp_spec.rs | developer |
| `HIPFIRE_QWEN4_MTP_BATCHED_FILL` | crates/hipfire-arch-qwen4/examples/qwen4_mtp_fill.rs, crates/hipfire-arch-qwen4/src/mtp_spec.rs | developer |
| `HIPFIRE_QWEN4_MTP_DEVICE_ACCEPT` | crates/hipfire-arch-qwen4/src/mtp_spec.rs | developer |
| `HIPFIRE_QWEN4_MTP_REUSE_PREFILL_IDS` | crates/hipfire-arch-qwen4/src/mtp_spec.rs, crates/railgun-cert/src/recording.rs | developer |
| `HIPFIRE_QWEN4_MTP_SAMPLED_PICKS` | crates/hipfire-arch-qwen4/src/mtp_spec.rs | developer |
| `HIPFIRE_QWEN4_MTP_SKIP_INTERMEDIATE_PICK` | crates/hipfire-arch-qwen4/src/mtp_spec.rs | developer |
| `HIPFIRE_QWEN4_MTP_TIER` | crates/hipfire-arch-qwen4/src/weights.rs | developer |
| `HIPFIRE_QWEN4_NPU_TRUNK` | crates/hipfire-arch-qwen4/src/npu_trunk/config.rs, crates/hipfire-config/src/lib.rs | experimental |
| `HIPFIRE_QWEN4_NPU_TRUNK_CAL` | crates/hipfire-config/src/lib.rs, scripts/perf_config_receipt.py | experimental |
| `HIPFIRE_QWEN4_NPU_TRUNK_FAMILIES` | crates/hipfire-arch-qwen4/src/npu_trunk/forward.rs | developer |
| `HIPFIRE_QWEN4_NPU_TRUNK_FAULT_CHAIN` | crates/hipfire-arch-qwen4/src/npu_trunk/forward.rs | developer |
| `HIPFIRE_QWEN4_NPU_TRUNK_SAMPLE_LOG` | crates/hipfire-arch-qwen4/src/npu_trunk/forward.rs, crates/hipfire-cli/src/doctor/calibrate.rs | developer |
| `HIPFIRE_QWEN4_NPU_TRUNK_TRACE` | crates/hipfire-arch-qwen4/src/npu_trunk/forward.rs | developer |
| `HIPFIRE_QWEN4_ORACLE_CACHE` | crates/hipfire-arch-qwen4/reference_oracle/generate_fixtures.py, crates/hipfire-arch-qwen4/reference_oracle/test_route_floor.py | harness |
| `HIPFIRE_QWEN4_PENALTY_PREPASS` | crates/hipfire-arch-qwen4/src/bundle.rs, crates/hipfire-arch-qwen4/src/mtp_spec.rs | developer |
| `HIPFIRE_QWEN4_PLE_ASYNC_UPLOAD` | crates/hipfire-arch-qwen4/src/gpu_forward.rs, crates/hipfire-arch-qwen4/src/ple_stage.rs | developer |
| `HIPFIRE_QWEN4_PLE_FUSE` | crates/hipfire-arch-qwen4/src/gpu_forward.rs, crates/hipfire-dispatch/src/pipeline/layer_ops.rs | developer |
| `HIPFIRE_QWEN4_PLE_WIDE_READERS` | crates/hipfire-arch-qwen4/src/bundle.rs | developer |
| `HIPFIRE_QWEN4_PREFILL_ROUTE_DUMP` | crates/hipfire-arch-qwen4/src/gpu_forward.rs | developer |
| `HIPFIRE_QWEN4_PROFILE_CHECKPOINT` | crates/hipfire-arch-qwen4/src/state_parity.rs | developer |
| `HIPFIRE_QWEN4_PROFILE_SOURCE_CALLBACK` | crates/hipfire-arch-qwen4/src/state_parity.rs | developer |
| `HIPFIRE_QWEN4_PROJ_REGIONS` | crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_QWEN4_QSA_BF16_SPLIT` | crates/hipfire-arch-qwen4/src/bundle.rs, crates/railgun-cert/src/recording.rs | developer |
| `HIPFIRE_QWEN4_QSA_CODE_CHECK` | crates/rdna-compute/examples/qsa_code_check_selftest.rs, crates/rdna-compute/src/feature_flags.rs | developer |
| `HIPFIRE_QWEN4_QSA_DECODE_SPLITK` | crates/rdna-compute/src/feature_flags.rs, crates/rdna-compute/src/kernel_registry.rs | developer |
| `HIPFIRE_QWEN4_QSA_EXTENT_CHECK` | crates/rdna-compute/src/feature_flags.rs | developer |
| `HIPFIRE_QWEN4_QSA_F16_ARENA` | crates/hipfire-arch-qwen4/examples/qwen4_qsa_arena_check.rs, crates/hipfire-arch-qwen4/src/state.rs | developer |
| `HIPFIRE_QWEN4_QSA_FILTER_FEW_ROWS32` | crates/rdna-compute/src/tensor_ops.rs | developer |
| `HIPFIRE_QWEN4_QSA_FILTER_SELECT` | crates/rdna-compute/src/tensor_ops.rs | developer |
| `HIPFIRE_QWEN4_QSA_PM` | crates/peacemaker-isa/src/kernels/qsa_gather.rip.rs, crates/rdna-compute/examples/qsa_pm_check.rs | developer |
| `HIPFIRE_QWEN4_QSA_POOLED_BF16` | crates/hipfire-arch-qwen4/examples/qwen4_qsa_arena_check.rs, crates/hipfire-arch-qwen4/src/state.rs | developer |
| `HIPFIRE_QWEN4_QSA_SCORE_PM` | crates/rdna-compute/examples/qsa_select_pm_check.rs, crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_QWEN4_QSA_SCORE_ROWS32` | crates/rdna-compute/src/tensor_ops.rs | developer |
| `HIPFIRE_QWEN4_QSA_SELECT_EXACT` | crates/hipfire-runtime/examples/qwen4_tierc/qsa.rs, crates/railgun-cert/src/recording.rs | developer |
| `HIPFIRE_QWEN4_QSA_SELECT_PM` | crates/rdna-compute/examples/qsa_select_pm_check.rs, crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_QWEN4_QSA_SELECT_PM_DECODE` | crates/rdna-compute/src/tensor_ops.rs | developer |
| `HIPFIRE_QWEN4_QSA_WMMA_GATHER` | crates/hipfire-arch-qwen4/examples/qwen4_qsa_ctx.rs, crates/hipfire-arch-qwen4/src/bundle.rs | developer |
| `HIPFIRE_QWEN4_REQUANT` | crates/hipfire-arch-qwen4/src/artifact.rs, crates/hipfire-arch-qwen4/src/bundle.rs | developer |
| `HIPFIRE_QWEN4_ROUTER_FAST` | crates/railgun-cert/src/recording.rs, crates/rdna-compute/src/feature_flags.rs | developer |
| `HIPFIRE_QWEN4_ROUTE_TRACE` | crates/hipfire-arch-qwen4/src/gpu_forward.rs | developer |
| `HIPFIRE_QWEN4_SHARED_DOWN_EPI` | crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_QWEN4_SHARED_GU_PM` | crates/hipfire-dispatch/src/pipeline/qt44_qt53_prefill.rs, crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_QWEN4_TAIL_SNAPSHOT` | crates/hipfire-arch-qwen4/src/bundle.rs, crates/hipfire-arch-qwen4/tests/session_cache_hw.rs | developer |
| `HIPFIRE_QWEN4_TAIL_SNAPSHOT_MIN_GAP` | crates/hipfire-arch-qwen4/src/bundle.rs | developer |
| `HIPFIRE_QWEN4_TAIL_SNAPSHOT_MIN_PROMPT` | crates/hipfire-arch-qwen4/src/bundle.rs | developer |
| `HIPFIRE_QWEN4_TOKEN_READBACK_EARLY` | crates/hipfire-arch-qwen4/src/gpu_forward.rs | developer |
| `HIPFIRE_QWEN4_TOOL_ARG_REPLAY` | crates/hipfire-generate/src/qwen.rs | developer |
| `HIPFIRE_QWEN4_TRUNK_IU4` | crates/hipfire-arch-qwen4/src/gpu_forward.rs, crates/hipfire-arch-qwen4/src/weights.rs | developer |
| `HIPFIRE_QWEN4_TRUNK_IU4_RAGGED` | crates/rdna-compute/examples/qwen4_trunk_zba_ragged.rs, crates/rdna-compute/src/gemm.rs | developer |
| `HIPFIRE_QWEN4_TRUNK_OVERLAY` | crates/hipfire-arch-qwen4/src/trunk_overlay.rs, crates/hipfire-loader/src/carriers.rs | developer |
| `HIPFIRE_QWEN4_TRUNK_TIER` | crates/hipfire-arch-qwen4/src/weights.rs | developer |
| `HIPFIRE_QWEN4_TURN_SNAPSHOTS` | crates/hipfire-arch-qwen4/src/bundle.rs, crates/hipfire-arch-qwen4/src/bundle/pp.rs | developer |
| `HIPFIRE_QWEN4_WIDE_TWIN_BF16` | crates/rdna-compute/examples/bf16_rows_oracle.rs, crates/rdna-compute/examples/bf16_rows_pm_oracle.rs | developer |
| `HIPFIRE_QWEN4_WIDE_TWIN_K1` | crates/rdna-compute/src/gemv.rs | developer |
| `HIPFIRE_QWEN4_WIDE_TWIN_MQ4` | crates/rdna-compute/examples/mq4_rows_oracle.rs, crates/rdna-compute/src/gemv.rs | developer |
| `HIPFIRE_QWEN4_WIDE_TWIN_MQ6` | crates/rdna-compute/examples/mq6_lm_head_rows.rs, crates/rdna-compute/src/gemv.rs | developer |
| `HIPFIRE_QWEN4_WIDE_TWIN_Q8` | crates/rdna-compute/src/gemv.rs | developer |
| `HIPFIRE_QWEN4_WIDE_VERIFY_ROWS` | crates/hipfire-arch-qwen4/src/gpu_forward.rs, crates/rdna-compute/src/prefill_invariant.rs | developer |
| `HIPFIRE_QWEN_CACHE_TRACE` | crates/hipfire-daemon/src/main.rs, crates/hipfire-generate/src/ar.rs | developer |
| `HIPFIRE_QWEN_KV_DEFAULT_Q8` | crates/hipfire-arch-qwen35/examples/cb_vmm_state_oracle.rs, crates/hipfire-arch-qwen35/examples/prefill_ragged_oracle.rs | developer |
| `HIPFIRE_QWEN_MOE_FINAL_NORM_RAW` | scripts/test_pr228_spiral_check.sh | harness |
| `HIPFIRE_QWEN_MTP` | scripts/benchlocal_campaign.py, scripts/serve_harness.py | harness |
| `HIPFIRE_QWEN_PROMPT_CACHE` | crates/hipfire-generate/src/ar.rs, crates/hipfire-generate/src/decide/session.rs | developer |
| `HIPFIRE_RAILGUN_BACKEND` | crates/hipfire-arch-qwen35/tests/pm4_staging_hold_hw.rs, crates/rdna-compute/src/replay.rs | developer |
| `HIPFIRE_RAILGUN_CACHE_TABLE` | crates/rdna-compute/src/replay/railgun_shadow.rs | developer |
| `HIPFIRE_RAILGUN_CHECK` | crates/hip-bridge/src/registry.rs, crates/hipfire-arch-qwen35/tests/pm4_staging_hold_hw.rs | developer |
| `HIPFIRE_RAILGUN_CHECK_OUT` | crates/rdna-compute/src/railgun_check.rs | developer |
| `HIPFIRE_RAILGUN_DIGEST` | crates/hip-bridge/src/registry.rs, crates/hipfire-arch-qwen35/tests/pm4_staging_hold_hw.rs | developer |
| `HIPFIRE_RAILGUN_INVENTORY` | crates/rdna-compute/src/replay/railgun_shadow.rs | developer |
| `HIPFIRE_RAILGUN_NEGATIVE_CONTROL` | crates/rdna-compute/src/railgun_check.rs, crates/rdna-compute/src/replay/railgun_shadow.rs | developer |
| `HIPFIRE_RAILGUN_SHADOW` | crates/hipfire-arch-qwen35/tests/pm4_staging_hold_hw.rs, crates/rdna-compute/src/railgun_check.rs | developer |
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
| `HIPFIRE_RELEASES_LATEST_API` | scripts/install.sh, scripts/test_install_revision.py | harness |
| `HIPFIRE_REMOTE` | scripts/mi300x_bootstrap.sh | harness |
| `HIPFIRE_REPLAY_BACKEND` | crates/hipfire-arch-qwen35/tests/pm4_staging_hold_hw.rs, crates/hipfire-arch-qwen35/tests/pm_decode_twins_hw.rs | stable |
| `HIPFIRE_REPLAY_BINDINGS_VERIFY` | crates/rdna-compute/src/replay.rs | developer |
| `HIPFIRE_REPLAY_DIAGNOSTIC_SPECIALIZED_MOE_CAPTURE` | crates/hipfire-dispatch/src/pipeline/sealed_moe.rs | developer |
| `HIPFIRE_REPLAY_GRAPH` | crates/hipfire-arch-qwen35/src/speculative.rs, crates/rdna-compute/src/dflash_gdn_replay.rs | developer |
| `HIPFIRE_REPLAY_MANUAL_CAPTURE` | crates/hipfire-arch-qwen35/tests/pm4_staging_hold_hw.rs, crates/hipfire-arch-qwen35/tests/pm_decode_twins_hw.rs | experimental |
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
| `HIPFIRE_REPLAY_TRANSPORT` | crates/hipfire-arch-qwen35/tests/pm4_staging_hold_hw.rs, crates/hipfire-arch-qwen4/examples/qwen4_g2_replay_gate.rs | experimental |
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
| `HIPFIRE_ROUTE_PLAN` | crates/hipfire-loader/src/lib.rs, scripts/route_plan_walk.py | developer |
| `HIPFIRE_ROWS_CQN` | crates/rdna-compute/src/rows_batched.rs | developer |
| `HIPFIRE_ROWS_FA_PREP` | crates/rdna-compute/src/rows_batched.rs | developer |
| `HIPFIRE_ROWS_GATED_NORM` | crates/rdna-compute/src/rows_batched.rs | developer |
| `HIPFIRE_ROWS_GDN` | crates/rdna-compute/src/rows_batched.rs | developer |
| `HIPFIRE_S4_FLAG_PROBE_UNSET_OFF` | crates/hipfire-config/src/lib.rs | developer |
| `HIPFIRE_S4_FLAG_PROBE_UNSET_ON` | crates/hipfire-config/src/lib.rs | developer |
| `HIPFIRE_SAMPLED_MTP_ARM` | crates/hipfire-arch-qwen35/tests/sampled_mtp_penalty_hw.rs, crates/hipfire-arch-qwen4/tests/sampled_mtp_distribution_hw.rs | harness |
| `HIPFIRE_SAMPLED_MTP_CONFIG` | crates/hipfire-arch-qwen35/tests/sampled_mtp_penalty_hw.rs | harness |
| `HIPFIRE_SAMPLED_MTP_COPY_LONG_ARMS` | crates/hipfire-arch-qwen4/tests/sampled_mtp_distribution_hw.rs | harness |
| `HIPFIRE_SAMPLED_MTP_FREQUENCY_PENALTY` | crates/hipfire-arch-qwen35/tests/sampled_mtp_penalty_hw.rs, crates/hipfire-arch-qwen4/tests/sampled_mtp_distribution_hw.rs | harness |
| `HIPFIRE_SAMPLED_MTP_GREEDY_TOKENS` | crates/hipfire-arch-qwen35/tests/sampled_mtp_penalty_hw.rs | harness |
| `HIPFIRE_SAMPLED_MTP_LATE_TOKENS` | crates/hipfire-arch-qwen4/tests/sampled_mtp_distribution_hw.rs | harness |
| `HIPFIRE_SAMPLED_MTP_MIN_P` | crates/hipfire-arch-qwen35/tests/sampled_mtp_penalty_hw.rs, crates/hipfire-arch-qwen4/tests/sampled_mtp_distribution_hw.rs | harness |
| `HIPFIRE_SAMPLED_MTP_OUT` | crates/hipfire-arch-qwen35/tests/sampled_mtp_penalty_hw.rs, crates/hipfire-arch-qwen4/tests/sampled_mtp_distribution_hw.rs | harness |
| `HIPFIRE_SAMPLED_MTP_PRESENCE_PENALTY` | crates/hipfire-arch-qwen35/tests/sampled_mtp_penalty_hw.rs, crates/hipfire-arch-qwen4/tests/sampled_mtp_distribution_hw.rs | harness |
| `HIPFIRE_SAMPLED_MTP_PROMPT` | crates/hipfire-arch-qwen35/tests/sampled_mtp_penalty_hw.rs, crates/hipfire-arch-qwen4/tests/sampled_mtp_distribution_hw.rs | harness |
| `HIPFIRE_SAMPLED_MTP_RENDER` | crates/hipfire-arch-qwen35/tests/sampled_mtp_penalty_hw.rs | harness |
| `HIPFIRE_SAMPLED_MTP_REPEAT_PENALTY` | crates/hipfire-arch-qwen35/tests/sampled_mtp_penalty_hw.rs, crates/hipfire-arch-qwen4/tests/sampled_mtp_distribution_hw.rs | harness |
| `HIPFIRE_SAMPLED_MTP_REPEAT_WINDOW` | crates/hipfire-arch-qwen35/tests/sampled_mtp_penalty_hw.rs, crates/hipfire-arch-qwen4/tests/sampled_mtp_distribution_hw.rs | harness |
| `HIPFIRE_SAMPLED_MTP_SHARD` | crates/hipfire-arch-qwen4/tests/sampled_mtp_distribution_hw.rs | harness |
| `HIPFIRE_SAMPLED_MTP_TEMP` | crates/hipfire-arch-qwen35/tests/sampled_mtp_penalty_hw.rs, crates/hipfire-arch-qwen4/tests/sampled_mtp_distribution_hw.rs | harness |
| `HIPFIRE_SAMPLED_MTP_THINK_SEEDS` | crates/hipfire-arch-qwen4/tests/sampled_mtp_distribution_hw.rs | harness |
| `HIPFIRE_SAMPLED_MTP_THINK_TOKENS` | crates/hipfire-arch-qwen4/tests/sampled_mtp_distribution_hw.rs | harness |
| `HIPFIRE_SAMPLED_MTP_TOKENS` | crates/hipfire-arch-qwen35/tests/sampled_mtp_penalty_hw.rs | harness |
| `HIPFIRE_SAMPLED_MTP_TOP_K` | crates/hipfire-arch-qwen35/tests/sampled_mtp_penalty_hw.rs, crates/hipfire-arch-qwen4/tests/sampled_mtp_distribution_hw.rs | harness |
| `HIPFIRE_SAMPLED_MTP_TOP_P` | crates/hipfire-arch-qwen35/tests/sampled_mtp_penalty_hw.rs, crates/hipfire-arch-qwen4/tests/sampled_mtp_distribution_hw.rs | harness |
| `HIPFIRE_SAMPLED_MTP_TRIALS` | crates/hipfire-arch-qwen35/tests/sampled_mtp_penalty_hw.rs, crates/hipfire-arch-qwen4/tests/sampled_mtp_distribution_hw.rs | harness |
| `HIPFIRE_SAMPLE_COMPARE` | crates/hipfire-runtime/src/llama.rs, crates/saddle-lab/examples/infer_qwen35.rs | developer |
| `HIPFIRE_SAMPLE_FAST` | crates/hipfire-runtime/examples/sample_batch_exact_oracle.rs, crates/rdna-compute/examples/sample_parallel_stable_parity.rs | developer |
| `HIPFIRE_SAMPLE_PARALLEL` | crates/hipfire-runtime/examples/sample_batch_exact_oracle.rs, crates/hipfire-runtime/src/batch_sampler.rs | developer |
| `HIPFIRE_SCHED_PROFILE` | crates/rdna-compute/src/compiler.rs | developer |
| `HIPFIRE_SEGS_ATTN_FP8` | crates/rdna-compute/src/verify_twins.rs | developer |
| `HIPFIRE_SELECT_REGRID_OFF` | crates/rdna-compute/src/feature_flags.rs, crates/rdna-compute/src/select_regrid.rs | developer |
| `HIPFIRE_SERVE_ALLOW_INCOHERENT` | scripts/serve_harness.py | harness |
| `HIPFIRE_SERVE_ALLOW_REQUEST_PATHS` | crates/hipfire-config/src/lib.rs | stable |
| `HIPFIRE_SERVE_ALLOW_REQUEST_PULL` | crates/hipfire-config/src/lib.rs | stable |
| `HIPFIRE_SERVE_API_KEY_FILE` | crates/hipfire-cli/src/serve/mod.rs, crates/hipfire-config/src/lib.rs | stable |
| `HIPFIRE_SERVE_BATCH_NONEXACT` | crates/hipfire-arch-qwen35/src/forward_slots/vmm.rs, crates/hipfire-config/src/lib.rs | experimental |
| `HIPFIRE_SERVE_BATCH_SAMPLED` | crates/hipfire-config/src/lib.rs | experimental |
| `HIPFIRE_SERVE_BATCH_SPEC` | crates/hipfire-config/src/lib.rs, crates/hipfire-generate/src/fn_batch.rs | experimental |
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
| `HIPFIRE_SERVE_VMM_BATCH` | crates/hipfire-config/src/lib.rs | experimental |
| `HIPFIRE_SESSION_CACHE_BYTES` | crates/hipfire-config/src/lib.rs, experiments/qwen4-yarn-longctx/niah.sh | stable |
| `HIPFIRE_SESSION_CACHE_DIR` | crates/hipfire-config/src/lib.rs | stable |
| `HIPFIRE_SESSION_CACHE_DISK_BYTES` | crates/hipfire-config/src/lib.rs | stable |
| `HIPFIRE_SESSION_CACHE_DISK_WRITE_PHASE` | crates/hipfire-arch-qwen4/tests/session_cache_hw.rs | harness |
| `HIPFIRE_SESSION_CACHE_MODEL` | crates/hipfire-arch-qwen4/tests/session_cache_hw.rs | harness |
| `HIPFIRE_SHARED_VMM_DEVICE` | crates/hip-bridge/examples/shared_vmm_fd_probe.rs | harness |
| `HIPFIRE_SILU_FP8_AWQ` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_SILU_FP8_H_BF16` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_SILU_FP8_KERNEL` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_SILU_HIN` | crates/peacemaker-isa/src/kernels/iu4_v2b_a4.rs, crates/rdna-compute/src/gemv.rs | developer |
| `HIPFIRE_SILU_H_BF16` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_SILU_MQ_ROTATE_KERNEL` | crates/rdna-compute/src/gemv.rs, crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_SKIP_AGENTIC_GATE` | scripts/agentic-gate.sh | harness |
| `HIPFIRE_SKIP_BUILD` | scripts/container-gate.sh, scripts/test_container_gate.py | harness |
| `HIPFIRE_SKIP_LLAMA_COMMIT_CHECK` | crates/hipfire-runtime/src/eval_common.rs | developer |
| `HIPFIRE_SLOTS_ATTN_CROSSOVER` | crates/hipfire-arch-qwen35/src/forward_slots.rs, crates/rdna-compute/src/feature_flags.rs | developer |
| `HIPFIRE_SLOTS_DECODE_GRAPH` | crates/rdna-compute/src/feature_flags.rs | developer |
| `HIPFIRE_SLOTS_PAGED` | crates/hipfire-arch-qwen35/src/serve_engine.rs | developer |
| `HIPFIRE_SLOTS_PAGED_PAGES` | crates/hipfire-arch-qwen35/src/serve_engine.rs | developer |
| `HIPFIRE_SLOT_TRACE` | crates/rdna-compute/src/feature_flags.rs, scripts/serve_concurrency_gate.sh | developer |
| `HIPFIRE_SLOW_TOKEN_LOG` | crates/hipfire-runtime/examples/bench_qwen35_mq4.rs | harness |
| `HIPFIRE_SMOKE_KV` | crates/hipfire-arch-qwen35/src/qwen35/forward.rs, crates/saddle-lab/examples/a3b_smoke_forward.rs | developer |
| `HIPFIRE_SMOKE_KV_SEQ` | crates/saddle-lab/examples/a3b_smoke_forward.rs | harness |
| `HIPFIRE_SMOKE_MODE` | crates/hipfire-arch-qwen35/src/qwen35/forward.rs, crates/saddle-lab/examples/a3b_smoke_forward.rs | developer |
| `HIPFIRE_SMOKE_PROMPT` | crates/hipfire-arch-qwen35/src/qwen35/forward.rs, crates/saddle-lab/examples/a3b_smoke_forward.rs | developer |
| `HIPFIRE_SMOKE_STEPS` | crates/hipfire-arch-qwen35/src/qwen35/forward.rs, crates/saddle-lab/examples/a3b_smoke_forward.rs | developer |
| `HIPFIRE_SPECULATION` | crates/hipfire-config/src/lib.rs, docs/investigations/evidence/ds4-mi300x-cdna-test-fail/raw/a1-m0/04-run-profile-direct.sh | stable |
| `HIPFIRE_SPEC_PHASES` | crates/hipfire-arch-qwen35/examples/cb_vmm_state_oracle.rs, crates/hipfire-arch-qwen35/src/speculative.rs | developer |
| `HIPFIRE_SPEC_RETIRE` | crates/hipfire-generate/src/qwen.rs | developer |
| `HIPFIRE_SPEC_WINDOW_ROLLBACK` | crates/hipfire-config/src/lib.rs | developer |
| `HIPFIRE_SPE_OUT_ROOT` | benchmarks/quality-baselines/harness/spe_ablation.sh | harness |
| `HIPFIRE_SPILL_DIR` | crates/hipfire-config/src/lib.rs, crates/hipfire-quantize/src/pipeline.rs | stable |
| `HIPFIRE_STATE_QUANT` | crates/hipfire-arch-qwen4/examples/fn_lane_oracle.rs, crates/hipfire-arch-qwen4/examples/fn_prefill_g0.rs | harness |
| `HIPFIRE_SWEEP_MAX` | scripts/ddtree_budget_sweep.sh | harness |
| `HIPFIRE_SWEEP_OUT` | scripts/mq3-mq2-sweep.sh, scripts/spec_decode_genre_sweep.sh | harness |
| `HIPFIRE_SWEEP_PROMPTS_DIR` | scripts/mq3-mq2-sweep.sh | harness |
| `HIPFIRE_SWEEP_RUNS` | scripts/ddtree_budget_sweep.sh | harness |
| `HIPFIRE_T5_GPU` | crates/hipfire-arch-diffusion/examples/gpu_pipeline_parity.rs, crates/hipfire-arch-diffusion/src/pipeline.rs | developer |
| `HIPFIRE_TARGET_ARCH` | crates/rdna-compute/src/dispatch.rs, scripts/kernel_atlas.py | developer |
| `HIPFIRE_TEMP` | docs/investigations/evidence/ds4-mi300x-cdna-test-fail/raw/a1-m0/04-profile-feed.py, docs/investigations/evidence/ds4-mi300x-cdna-test-fail/raw/a1-m0/04-run-profile-direct.sh | harness |
| `HIPFIRE_TEST_MODEL` | scripts/test-ds4-heterogeneous-abort-resume.sh, scripts/test-qwen35-abort-resume.sh | harness |
| `HIPFIRE_TEST_REAL_HIPCC` | crates/rdna-compute/src/compiler.rs, scripts/check-env-docs.py | developer |
| `HIPFIRE_TEST_REQUIRE_ROCM` | crates/peacemaker-cdna/src/oracle.rs, crates/peacemaker-cdna/tests/enc_sop.rs | developer |
| `HIPFIRE_TEXT_OUT` | docs/investigations/evidence/ds4-mi300x-cdna-test-fail/raw/a1-m0/04-profile-feed.py, docs/investigations/evidence/ds4-mi300x-cdna-test-fail/raw/a1-m0/04-run-profile-direct.sh | harness |
| `HIPFIRE_THINK_CONTINUATION` | crates/hipfire-arch-qwen35/src/spec_emit.rs, crates/hipfire-daemon/src/main.rs | developer |
| `HIPFIRE_TIER_RATIO` | crates/hipfire-quantize/src/cli.rs | developer |
| `HIPFIRE_TOKBENCH_ONLY` | crates/hipfire-runtime/tests/hf_tokenizer_encode_bench.rs | harness |
| `HIPFIRE_TOKBENCH_REPS` | crates/hipfire-runtime/tests/hf_tokenizer_encode_bench.rs | harness |
| `HIPFIRE_TOKENIZER_DIFF_MIN_LIMIT` | crates/hipfire-runtime/tests/hf_tokenizer_differential.rs | harness |
| `HIPFIRE_TOKENIZER_DIFF_MIN_PROBES` | crates/hipfire-runtime/tests/hf_tokenizer_differential.rs | harness |
| `HIPFIRE_TOKENIZER_DIFF_PRINT_LIMIT` | crates/hipfire-runtime/tests/hf_tokenizer_differential.rs | harness |
| `HIPFIRE_TOKENIZER_DIFF_REPORT_DIR` | crates/hipfire-runtime/tests/hf_tokenizer_differential.rs | harness |
| `HIPFIRE_TOKENIZER_FUZZ_CASES` | crates/hipfire-runtime/tests/hf_tokenizer_differential.rs | harness |
| `HIPFIRE_TOKENIZER_FUZZ_SEED` | crates/hipfire-runtime/tests/hf_tokenizer_differential.rs | harness |
| `HIPFIRE_TOPK8_RESCORE` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_TOPK_FIXUP_MARKED` | crates/rdna-compute/src/select_regrid.rs | developer |
| `HIPFIRE_TOPOLOGY_EP` | crates/hipfire-config/src/lib.rs, scripts/fn-ep-gate.sh | experimental |
| `HIPFIRE_TOPOLOGY_PP` | crates/hipfire-config/src/lib.rs, scripts/fn-ep-gate.sh | experimental |
| `HIPFIRE_TP` | crates/hipfire-arch-qwen35/examples/qwen_dense_tp2_parity.rs | harness |
| `HIPFIRE_TP2_GRAPH_REPLAYS` | crates/hipfire-runtime/examples/tp2_cross_device_graph_reduce.rs | harness |
| `HIPFIRE_TP4_GRAPH_REPLAYS` | crates/hipfire-runtime/examples/tp4_cross_device_graph_barrier.rs | harness |
| `HIPFIRE_TP_BENCH_ITERS` | crates/hip-bridge/examples/rccl_smoke.rs, crates/hipfire-runtime/examples/tp_allreduce_smoke.rs | harness |
| `HIPFIRE_TP_BENCH_N` | crates/hip-bridge/examples/rccl_smoke.rs, crates/hipfire-runtime/examples/tp_allreduce_smoke.rs | harness |
| `HIPFIRE_TP_BENCH_WARMUP` | crates/hipfire-runtime/examples/tp_allreduce_smoke.rs | harness |
| `HIPFIRE_TP_FULL_GRAPH` | crates/hipfire-arch-qwen35/src/qwen35/forward.rs, crates/hipfire-arch-qwen35/tests/route_oracle_mesh.rs | developer |
| `HIPFIRE_TP_GRAPH_RANKS` | crates/hipfire-runtime/examples/tp4_cross_device_graph_barrier.rs | harness |
| `HIPFIRE_TP_PARITY_PROMPT_FILE` | crates/hipfire-arch-qwen35/examples/qwen_dense_tp2_parity.rs | harness |
| `HIPFIRE_TP_PARITY_STEPS` | crates/hipfire-arch-qwen35/examples/qwen_dense_tp2_parity.rs | harness |
| `HIPFIRE_TP_PARITY_TP` | crates/hipfire-arch-qwen35/examples/qwen_dense_tp2_parity.rs | harness |
| `HIPFIRE_TP_PEER_ALTERNATING` | crates/hipfire-runtime/src/multi_gpu.rs | developer |
| `HIPFIRE_TP_PEER_DIRECT` | crates/hipfire-runtime/src/multi_gpu.rs | developer |
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
| `HIPFIRE_VERIFY_GRAPH` | crates/hipfire-arch-qwen35/examples/cb_vmm_state_oracle.rs, crates/hipfire-arch-qwen35/src/mtp_probe.rs | developer |
| `HIPFIRE_VERIFY_GRAPH_TIMING` | crates/hipfire-arch-qwen35/src/speculative.rs | developer |
| `HIPFIRE_VERIFY_GRAPH_TREE` | crates/hipfire-arch-qwen35/src/speculative.rs, scripts/tree_graph_bench.sh | developer |
| `HIPFIRE_VERIFY_TAPE_COPY_BATCH` | crates/hipfire-arch-qwen35/src/qwen35/prefill.rs | developer |
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
| `HIPFIRE_VMM_PROMOTE_AT` | crates/hipfire-generate/src/batch.rs | developer |
| `HIPFIRE_VMM_SECOND_BYTES` | crates/hip-bridge/examples/vmm_arena_smoke.rs | harness |
| `HIPFIRE_VMM_SMOKE_DEVICE` | crates/hip-bridge/examples/vmm_arena_smoke.rs, crates/rdna-compute/examples/vmm_tensor_smoke.rs | harness |
| `HIPFIRE_VQ_BOOKS_JSON` | crates/hipfire-runtime/src/e8p_i4_reference.rs | developer |
| `HIPFIRE_VRAM_BUDGET_BYTES` | crates/rdna-compute/examples/q8_batched_attn_microbench.rs | harness |
| `HIPFIRE_WEIGHT_BUFFER_LOADS_FLAT_GEMV_OPT_IN` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_WEIGHT_BUFFER_LOADS_OPT_IN` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_WEIGHT_CACHE_FLAT_GEMV` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_WEIGHT_CPOL_AUX` | crates/rdna-compute/src/kernels.rs | developer |
| `HIPFIRE_WEIGHT_HOST` | crates/hipfire-arch-qwen35/examples/shared_weight_state_oracle.rs, crates/hipfire-arch-qwen35/src/carrier.rs | developer |
| `HIPFIRE_WEIGHT_HOST_SOCKET` | crates/hipfire-arch-qwen35/examples/shared_weight_state_oracle.rs, crates/hipfire-arch-qwen35/src/carrier.rs | developer |
| `HIPFIRE_WH_FD_ISOLATED_CHILD` | crates/hipfire-runtime/src/weight_host.rs | developer |
| `HIPFIRE_WIDENED_PBS_GROW_ONLY` | crates/hipfire-arch-qwen35/src/qwen35/prefill.rs | developer |
| `HIPFIRE_WMMA_BATCH_TILES` | crates/hipfire-arch-qwen35/examples/cb_vmm_state_oracle.rs, crates/hipfire-arch-qwen35/src/qwen35/prefill_multi.rs | developer |
| `HIPFIRE_WMMA_FA` | benchmarks/results/wmma-fa-probe-gfx1100.sh, benchmarks/results/wmma-fa-probe.sh | developer |
| `HIPFIRE_WMMA_FA_MIN_BATCH` | crates/rdna-compute/src/attention.rs | developer |
| `HIPFIRE_WO_MMQ` | crates/hipfire-config/src/lib.rs, crates/rdna-compute/src/feature_flags.rs | experimental |
| `HIPFIRE_WO_WMMA_VARIANT` | crates/hipfire-config/src/lib.rs, crates/hipfire-runtime/examples/test_wmma_correctness.rs | experimental |
| `HIPFIRE_XDNA_A` | crates/hipfire-xdna/src/lib.rs | developer |
| `HIPFIRE_XDNA_B` | crates/hipfire-xdna/src/lib.rs | developer |
| `HIPFIRE_XDNA_CREF` | crates/hipfire-xdna/src/lib.rs | developer |
| `HIPFIRE_XDNA_INSTS` | crates/hipfire-xdna/src/lib.rs | developer |
| `HIPFIRE_XDNA_MKN` | crates/hipfire-xdna/src/lib.rs | developer |
| `HIPFIRE_XDNA_PDI` | crates/hipfire-xdna/src/lib.rs | developer |
| `HIPFIRE_YARN_FACTOR` | crates/hipfire-config/src/lib.rs, crates/hipfire-runtime/src/config.rs | stable |

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
| `HIPFIRE_CPU_EXEC_TRACE` | unset (off) | Developer diagnostic for `memory.offload_exec=cpu`. With `=1`, the CPU-exec seam prints one line per **distinct step shape** — `m`, `k`, quant, whether the activation was rotated, whether a residual was accumulated, whether an AWQ sidecar applied — at that shape's first call and then at every doubling of its call count, each with the running counters: steps run on the CPU and **host-mapped steps that stayed on the GPU**. The second number must be `0` for a model the CPU covers (a non-zero value means a step shape never reached the seam). Each line's D2H / GEMV / H2D split is the mean over *that shape's* calls so far, so the `calls=1` line is the cold first step and the later lines (…32, 64, 128 calls) are its steady state — the numbers that are worth quoting. A value that does not move with the call count is the shape's own cost, not a process-wide average. Read via `hipfire_config::developer_var`, so it works as an environment variable or a `[developer]` TOML key. It also prints, on the *global* step count's doubling schedule, one `cpu exec: idle N% — window ending at step S covers K steps: …ms wall, …ms on CPU (…us CPU/step)` line per window: the share of that decode window's wall spent inside CPU-executed steps. A CPU step is a host sync point (the blocking D2H, the multiplication, the H2D) and prefill never enters the seam, so that share is a **lower bound on the GPU's idle fraction** — the headroom any scheme handing part of a spilled step back to the GPU would be spending. It is not a correctness signal. The routed-expert splice prints its own `moe expert splice` line, keyed by `(quant, dim, mi, k)`, splitting the D2H into the first download (which carries the GPU wait) and the rest, and the GEMV into gate_up and down, plus coarse `setup` (expert resolve), `middle` (whole SiLU+FWHT loop), `combine` (whole weighted-combine loop), and `total` (splice wall excluding the trace print) means. Per-phase sums reconcile within a splice, but per-quant attribution does not: `quant` is the first selected expert's gate_up format only, `graded=true` collapses gate_up-mixed, down-mixed, and gate_up≠down mixes into one flag, the down quant is never printed, and no worker count is printed — discard the per-quant split when picking a structural fix. |
| `HIPFIRE_GPU_LAYER_BUDGET` | unset/null (fully resident on every model) | Typed key `memory.gpu_layer_budget`: the resident-layer budget for partial GPU offload. `N` keeps the last `N` layers on the GPU and spills the prefix `[0 .. n_layers-N)` to host RAM. Counts layers **on** the GPU — `3` on a 64-layer model spills 61. Explicit `auto`/`-1` places by fit capacity: routed experts first, whole layers only when the expert tier cannot free enough; a model that fits stays fully resident. Unset/null, empty and unparseable values do not request a spill, including on Qwen MoE. Manual `N` is a hard pin, never silently widened. Placement resolves once at load; supported offload paths are single-GPU Qwen3.5-family text trunks, not paged-expert / REAP-keep-map / multi-GPU / VL loads. `token_embd`/`output_norm`/`lm_head` stay resident. Spilled weights use `hipHostMalloc(hipHostMallocMapped)` RAM; `memory.offload_exec` selects execution, not placement. See [`CONFIG.md`](CONFIG.md#partial-gpu-offload-memorygpu_layer_budget--memorymoe_expert_budget--memoryoffload_exec) and [`plans/partial-gpu-offload-design.md`](plans/partial-gpu-offload-design.md). |
| `HIPFIRE_MOE_EXPERT_BUDGET` | unset/null (fully resident on every model) | Typed key `memory.moe_expert_budget`, the routed-expert sibling of `memory.gpu_layer_budget`. `N` keeps the last `N` layers' **routed experts** on the GPU and spills the rest; only routed experts move, while attention, router, shared expert and KV stay resident. Explicit `auto`/`-1` opts into fit-based placement, routed experts before whole layers. Unset/null or empty never implies Qwen MoE auto-fit. **Requires the packed expert layout.** Uniform layers require a shared dtype and stride per projection: MQ4/MQ4V2/MQ4C (qt 13/44/45) on any placement, plus MQ6/MQ2-Lloyd/MQ5/MFP4-E8 (qt 15/19/31/34) on host placement. Non-uniform layers use the graded packer, not a uniform-pack refusal. Graded PCIe layers require tag-mapped gate/down pairs without AWQ/LUT/GL sidecars; tag-branched kernels read per-expert strides. No-AWQ graded CPU layers use the real host packed bytes, each expert's own gate/down dtype and checked bucket-owner offset. Unsupported packing fails with the layer named; it never falls back to per-expert host tensors. Draft weights stay device-resident and are charged to fit capacity. Native MTP `auto` declines a resolved host-mapped routed-expert spill on either seam; `--spec mtp` opts in subject to the metadata-based uniform-MQ6 gfx12 FP16 refusal (see `HIPFIRE_MTP_MODE`). Resident narrow `SpeculativeVerify` retains beta's legacy grouped route; narrow Path1 is host-mapped routed-projection-only, with the coalesced gfx1201 host-weight narrow kernel retained. No effect on dense; configured unsupported spills fail closed. |
| `HIPFIRE_OFFLOAD_EXEC` | `pcie` | Typed key `memory.offload_exec`, the sibling of the layer/expert budgets. Selects **who multiplies**, never what spills: `pcie` (unchanged default; also unset/empty/unknown) uses GPU kernels over host-mapped weights; `cpu` executes supported GEMVs on the CPU (`crates/hipfire-cpu`). `passback` is **scheduled co-inference** — both engines run the same spilled step concurrently (the GPU arm is the `pcie` path, the CPU arm is the `cpu` path), divided by output rows per `memory.offload_passback_share`. Pass-back output is **not reproducible run to run**, by design: `cpu` and `pcie` execute a layer's GEMV on different engines, so their arithmetic need not agree, and pass-back's row split is scheduled per process from the seeding probe and the online arm timings — two identical greedy runs mix the engines differently. Pin `memory.offload_passback_share` to a number, or to `0` (byte-identical to `memory.offload_exec=cpu`), before diffing pass-back output against a reference. On a routed-expert spill, `cpu` computes the selected experts' whole FFNs for AR single-token decode and every `<=4`-row batched forward: speculative verify, tape-disabled accepted-prefix rollback and very short prompt prefill. The sealed GPU program keeps router/shared expert/residual, omits the routed FFN and binds no dummy expert pointer. Wider prompt/verify batches keep real pointer tables and GPU grouped PCIe reads. The splice groups repeated expert slots for cache reuse, preserves token rank and per-format accumulation order, reuses one f32 partial sum across qt13/15 k-tiles, and adds no warm per-call allocation or activation quantization. No-AWQ graded packed experts are supported per tier with checked bucket-owner offsets under `HIPFIRE_MOE_CPU_EXPERTS`; AWQ and unsupported host formats remain refused. Native MTP `auto` declines any resolved host-mapped routed-expert spill; `speculation.mtp=on` / `--spec mtp` opts in subject to the uniform-MQ6 gfx12 FP16 refusal. `dflash_mode=on` never aliases to MTP. Other resident MTP attachment is unchanged. `HIPFIRE_MOE_CPU_ORACLE` / `HIPFIRE_MOE_EXPERT_STATS` fail closed with CPU-deferred experts because `down_expanded` is unwritten (`explicit GPU routed expert oracle/statistics unavailable with CPU-deferred experts; use PCIe/resident`). Nothing spilled means no execution change. CPU execution disables graph capture and is refused with retained replay (Redline); single-token CPU-spilled MTP uses the CPU-splice-capable executor without `HIPFIRE_FORWARD_LOWERED=0`. Diagnostics: `HIPFIRE_CPU_EXEC_TRACE=1`. See [`plans/partial-gpu-offload-design.md`](plans/partial-gpu-offload-design.md#621-cpu-execution-of-the-spilled-weight-ops-memoryoffload_execcpu). |
| `HIPFIRE_OFFLOAD_PASSBACK_SHARE` | `auto` | Typed key `memory.offload_passback_share`, read only when `memory.offload_exec=passback`. How a spilled step's output rows are divided between the two engines: the GPU takes the first rows (the `pcie` arm) while the CPU runs the rest (the `cpu` arm). `auto` (default) schedules the split from this host's own measured engine rates and refines it online; a number in `(0, 0.5]` pins it; `0` collapses the split onto the CPU engine alone, so the mode is byte-identical to `memory.offload_exec=cpu` (the A/B baseline). Out-of-range numbers and unknown spellings are rejected by the schema. Set in another mode it is inert and prints one informational line at load. Measure or override with `hipfire offload-bench` (`--write` persists its recommendation). Full write-up: [`plans/partial-gpu-offload-design.md`](plans/partial-gpu-offload-design.md#622-pass-back-scheduled-co-inference-of-the-spilled-layers-memoryoffload_execpassback). |
| `HIPFIRE_OFFLOAD_EXEC` | crates/hipfire-config/src/lib.rs |
| `HIPFIRE_CPU_EXEC_TRACE` | crates/hipfire-dispatch/src/cpu_exec.rs |
| `HIPFIRE_OOM_GUARD` | `auto` | Typed key `memory.oom_guard`. Gates **only** the host-memory headroom half of `kv_slots::preflight_alloc` / `SlotPool` / CLI bench-sweep preflight (Linux `MemAvailable` minus the CMA region (the larger of `CmaFree` and the CMA pageblocks: `CmaTotal` or `/proc/zoneinfo` `cma`, which includes KHO scratch), since only movable allocations can use them, not pinned GTT or kernel memory; in a unified-memory process the TTM page-pool estimate is added on top, unchanged; CLI on Windows uses available physical memory). The R9700 deployment-target VRAM-budget check **always runs**. `auto`: on for unified-memory APU archs (gfx1035/1036/1103/1150/1151/1152), off for recognized discrete GPUs, and for processes with no known GPU arch by host swap state (Linux `SwapTotal`; Windows pagefile approximation: commit limit minus physical total, saturating). Zero or unavailable swap/pagefile → on; positive → off. `1`/`true`/`0`/`false` force either way. The auto decision is logged once to stderr with its reason. `scripts/run-bounded.sh` remains the hard backstop. Full write-up: [`CONFIG.md`](CONFIG.md#memoryoom_guard). |
| `HIPFIRE_MEM_CAP` | `24G` | Read by `scripts/run-bounded.sh`, not by the binaries: cgroup `MemoryMax` for a gated run. Exit 137 means the cap fired — shrink the configuration rather than raising it. |
