# Configuration

**Owner:** daemon / user config keys (`docs/INDEX.md`).
**Machine sources:**

- Native schema + resolution: `crates/hipfire-config/src/lib.rs`
- Versioned CLI/daemon handoff: `hipfire_config::ProcessConfig`
- Compact runtime snapshots: `hipfire_runtime::config::RuntimeConfig` and
  `rdna_compute::feature_flags::FeatureFlags`

**Last checked:** 2026-09-05.

Persistent stores under `~/.hipfire/`:

1. **Global** — sparse typed `config.toml`; missing keys inherit schema defaults.
2. **Per-model overlay (primary)** — `models.toml`: aliases, local paths,
   registry identities, and sparse per-model overrides.
3. **Migration inputs** — `config.json`, `models.json`, and
   `per_model_config.json`. Migration writes TOML and preserves the JSON files
   for rollback.

Copyable sparse profiles for users, kernel developers, and retained-PM4 work
live in [`docs/configs/`](configs/README.md). They include direct mappings from
historical `HIPFIRE_*` inputs to their persistent TOML keys. Apply a whole
bundle with the dedicated profile command:

```bash
hipfire config profile set default
hipfire config profile set dev
hipfire config profile set redline
hipfire config profile create lab
hipfire config profile              # interactive wizard
```

Selection replaces the complete sparse global `config.toml` with the chosen
built-in or custom profile layer (no stale keys remain). Custom snapshots live
under `~/.hipfire/profiles/<name>.toml`. Profile names are not schema fields and
are never written into `config.toml`. The Settings TUI and CLI share the
hipfire-config helpers. Profiles are global-only.

Edit global settings interactively with `hipfire config`. Per-model overlays use
the typed commands: `hipfire config <tag> list|get|set|reset ...`. Global
non-interactive edits use `hipfire config list|get|set|reset ...`.

**Precedence (operator view):** one-shot CLI values **>** legacy-compatible
process env **>** per-model override **>** global TOML **>** registry card
(`recommended_settings`) **>** built-in defaults. `hipfire config explain`
prints the winning source and shadowed candidates. Direct daemon invocation
resolves the same local TOML plus compatibility environment before GPU startup.

TOML is the supported persistent surface. Environment variables named in the
schema are compatibility/one-shot overrides, not a second persistent config
system. The native CLI serializes a versioned, schema-validated
`ProcessConfig` to the daemon before the GPU is initialized. The daemon
revalidates it and lowers it directly into `RuntimeConfig` and `FeatureFlags`;
it does not materialize TOML values as process environment variables. Sparse
architecture-sensitive fields remain absent so existing compiled defaults are
unchanged.

Process and diagnostic keys are global-only. `hipfire config <model> set ...`
rejects them because the daemon snapshots these values once; claiming a
per-model override inside a long-lived serve process would be misleading.

This page is the normative **key/default/enum** table. CASK/TriAttention is
deprecated (removal in 0.5.0) and PFlash is retained legacy research (see their
sections); multi-GPU topology
lives in its linked owner rather than a duplicated matrix here.

---

## Process-wide hardware and kernel policy

Use the native config commands; the flat spelling shown in `hipfire config
list` remains accepted as a migration alias:

```bash
hipfire config set kernel.mmq auto
hipfire config set kernel.fp16 true
hipfire config set fusions.qkv_bias true
hipfire config set diagnostic.gemm_dump false
```

`auto` means the existing architecture/model policy wins. These keys default
to `auto` and do not materialize an environment value until explicitly set:

| Family | Keys |
|---|---|
| GEMV / quant | `kernel.gemv_dp4a`, `kernel.gemv_prefetch`, `kernel.gfx942_lds_gemv`, `kernel.hfq3_dp4a`, `kernel.hfq3_mmq`, `kernel.hfq4_mmq_rdna2`, `kernel.gcn5_wave64_hybrid`, `kernel.mmq`, `kernel.gfx942_gemv_v2` |
| MoE | `kernel.moe_grouped_i8`, `kernel.moe_paro_i8`, `kernel.moe_paro_i8_k8` |
| Certified arch routes | `kernel.rdna3_hfq4_qkvza_k2048`, `kernel.rdna3_hfq4_residual_stage_x32`, `kernel.rdna3_hfq4_sigmoid_buffer`, `kernel.rdna3_rmsnorm_vecsum`, `kernel.gfx942_rmsnorm_split` |

Stable/default-on and safety controls:

| Key | Default | Purpose |
|---|---:|---|
| `hardware.allow_mixed_arch` | `false` | Permit heterogeneous multi-GPU topology. |
| `hardware.tp_use_rccl` | `true` | RCCL tensor-parallel all-reduce. |
| `kernel.prefill_batched` | `true` | Batched prefill kernels. |
| `speculation.draft_f16` | `true` | FP16 DFlash draft activations. |
| `kernel.fp16` | `true` | Allow FP16 routes. |
| `kernel.lm_head_wmma` | `true` | Allow WMMA LM-head dispatch. |
| `kernel.hfq4g128_mmq` | `true` | Allow HFQ4G128 MMQ dispatch. |
| `kernel.moe_grouped_gemm` | `true` | Grouped-GEMM MoE prefill. |
| `kernel.deepseek4_q8_wmma` | `true` | DeepSeek4 Q8 WMMA prefill. |
| `kernel.deepseek4_q8_4w` | `true` | DeepSeek4 four-wave Q8 tile. |
| `speculation.ddtree_tree_la` | `true` | DDTree linearized-ancestor tape. |
| `speculation.dflash_fast_sample` | `true` | GPU DFlash sampled verification. |
| `speculation.dflash_q8_lmhead_wmma` | `true` | Q8 WMMA LM-head in DFlash verify. |
| `fusions.qkv_bias` | `true` | Fold supported QKV bias launches. |
| `kernel.rocblas_off` | `false` | Disable rocBLAS dispatch. |
| `fusions.force_unfused` | `false` | Force supported projection paths unfused. |
| `speculation.dflash_tree` | `false` | Enable DDTree tree-SWOR verification. |
| `memory.oom_guard` | `auto` | Memory preflight OOM guard — see [Memory](#memory). Gates only the host `MemAvailable` headroom check; the R9700 deployment-target VRAM-budget check always runs. Env: `HIPFIRE_OOM_GUARD`. |

The following default-off keys are experimental kernel-route overrides. They
are typed booleans, process-scoped, and visible in `hipfire config list` with
per-key help:

| Family | Keys |
|---|---|
| RDNA3 QKV/ZA | `kernel.rdna3_hfq4_qkv_wave64`, `kernel.rdna3_hfq4_qkvza_2wave`, `kernel.rdna3_hfq4_qkvza_wavepack4`, `kernel.rdna3_hfq4_qkvza_ldsx8`, `kernel.rdna3_hfq4_qkvza_reduce_chain`, `kernel.rdna3_hfq4_qkvza_hoist_x32` |
| RDNA3 residual / sigmoid / head | `kernel.rdna3_hfq4_residual_k2048`, `kernel.rdna3_hfq4_sigmoid_tight_grid`, `kernel.rdna3_hfq4_sigmoid_rows4`, `kernel.rdna3_hfq4_lm_head_k2048`, `kernel.rdna3_hfq4_moe_gate_up_k2048` |
| RMSNorm | `kernel.rmsnorm_mq_tight_lds`, `kernel.rdna3_rmsnorm_wavegrid`, `kernel.rdna3_rmsnorm_split`, `kernel.rdna3_rmsnorm_sign_lds`, `kernel.rdna3_rmsnorm_sign_const` |
| MoE | `kernel.moe_grouped_i8_k8`, `kernel.moe_grouped_i8_k4`, `kernel.moe_grouped_i8_k4_gfx12`, `kernel.moe_grouped_m2`, `kernel.moe_grouped_4w`, `kernel.moe_down_combine_vec4`, `kernel.moe_hfq6_i8`, `kernel.moe_hfq6_v2` |
| Other kernel routes | `kernel.fp8_wmma`, `kernel.dot2_gemv`, `kernel.wo_mmq`, `kernel.lm_head_overwrite`, `kernel.hfq4_mmq_gfx906_y64`, `kernel.gate_up_nosync`, `kernel.qkvza_split_tail`, `kernel.gfx942_gemv_v3`, `kernel.deterministic`, `kernel.q8_batched_legacy`, `kernel.rope_interleaved_legacy`, `kernel.rocblas_all_archs`, `kernel.lloyd_force_baseline` |

Diagnostic booleans all default off: `diagnostic.prompt_token_heat`,
`diagnostic.prompt_heat_json`, `diagnostic.draft_gemm_dump`,
`diagnostic.draft_subphase`, `diagnostic.mmq_quantize_only`,
`diagnostic.blob_force`, `diagnostic.gemm_dump`, and
`diagnostic.qkv_bias`.

Process-wide scalar policy is typed as well:

| Key | Default | Values / purpose |
|---|---:|---|
| `hardware.devices` | `null` | Comma-separated device list in logical order: PCI-order index (`rocm-smi` order), `gfxNNNN` (first free card of that arch; repeat for more), `GPU-<uuid>`, or PCI address. Resolved from the KFD topology, reserved, installed as `ROCR_VISIBLE_DEVICES` UUIDs/ordinals plus HIP `0..N-1` before initialization, and checked against HIP after it. See [multi-gpu.md](multi-gpu.md#device-selection). |
| `hardware.uniform_vram_tolerance_gb` | `null` | Free-VRAM spread override; unset uses the compiled default. |
| `generation.loop_guard_threshold` | `0` | Repeated 4-gram count that forces EOS; zero disables. |
| `generation.loop_guard_window` | `256` | Token window inspected by the loop guard. |
| `kernel.flash_partials_batch` | `null` | Prefill flash-attention scratch multiplier. |
| `kernel.lm_head_f16` | `"auto"` | `auto`, `native`, `f16`, `f32`, `fp32`, `legacy`, plus numeric compatibility aliases `1`/`0`. |
| `diagnostic.prompt_heat_limit` | `64` | Maximum token-heat rows. |

Developer-only scalar and variant controls live below `diagnostic.kernel` so
they remain visible and typed without becoming registry-authorized product
defaults:

| Key | Values when set |
|---|---|
| `diagnostic.kernel.gemv_rows` | `"1"`, `"2"`, `"4"`, `"8"` |
| `diagnostic.kernel.fp16_layer_min`, `diagnostic.kernel.fp16_layer_max` | non-negative layer index |
| `diagnostic.kernel.hfq3_mmq_layer_min`, `diagnostic.kernel.hfq3_mmq_layer_max` | non-negative layer index |
| `diagnostic.kernel.mmq_min_batch`, `diagnostic.kernel.rocblas_min_batch` | non-negative batch threshold |
| `diagnostic.kernel.ddtree_logw_cutoff` | non-negative number |
| `diagnostic.kernel.lloyd_mb4`, `diagnostic.kernel.mq3_mb4` | `"1"`, `"2"`, `"4"` |
| `diagnostic.kernel.gate_up_variant` | `ldsx`, `k4`, `ldscoop`, `2tile` |
| `diagnostic.kernel.gfx11_weight_load_policy` | `buffer`, `global`, `flat-buffer` |
| `diagnostic.kernel.gfx12_weight_load_policy` | `rt`, `global`, `ht`, `nt-rt`, `nt-ht` |
| `diagnostic.kernel.gfx942_mfma_prefill` | `"1"`, `"2"`, `"3"`, `"4"` |
| `diagnostic.kernel.rdna2_variant` | integer 1–5 |
| `diagnostic.kernel.wo_wmma_variant` | `ksplit`, `ksplit_det`, `k2`, `k2x32`, `k4`, `wmma`, `wmma2` |
| `diagnostic.compiler.hipcc_extra_flags` | advanced local compiler-flag string |

Every field currently read by the centralized `RuntimeConfig` and
`FeatureFlags` snapshots has a schema mapping and crosses the CLI/daemon
boundary in the versioned `ProcessConfig`; no child-environment projection is
used.

### Experimental developer controls

The long tail of kernel experiments and diagnostics is available in one flat,
global-only TOML namespace instead of as hundreds of ambient process switches:

```toml
[developer]
gfx1151_gate_up_wave64 = true
deepseek4_attn = "twin"
replay_pm4_queues = 4
```

The CLI accepts the same keys (`hipfire config set
developer.gfx1151_gate_up_wave64 true`) and preserves the supplied scalar
spelling. A legacy `HIPFIRE_FOO=value` one-shot maps to
`developer.foo = "value"` during startup resolution. Developer values are
snapshotted before GPU initialization, sent in `ProcessConfig`, and read only
from that immutable snapshot. They are never registry-authorized or valid in a
per-model overlay. Prefer a typed schema key whenever one exists.

Direct production reads of `HIPFIRE_*` are restricted to bootstrap paths that
must be known before config discovery or process launch: home/model roots,
binary/registry locations, kernel cache, spill directory, and quant diagnostic
output. Examples and tests may still inject compatibility variables before
startup.

---

## Generation / sampling

| Key | Default | Validated range / enum | Notes |
|---|---|---|---|
| `temperature` | `0.30` | number 0.0–2.0 | Stored default only; see send path below. |
| `top_p` | `0.80` | number (0, 1] | Stored default only; see send path below. |
| `top_k` | `20` | int 1–100000 | Registry/TOML/request top-K cutoff; the built-in is omitted so daemon/model fallback remains authoritative. |
| `min_p` | `0.0` | number 0.0–1.0 | Relative-probability cutoff; zero disables it. |
| `presence_penalty` | `0.0` | number 0.0–2.0 | OpenAI-style flat penalty over the repeat window; zero disables it. |
| `repeat_penalty` | `1.05` | number 1.0–3.0 | Kept low; higher values harm some MQ4 greedy paths (source comment). Stored default only; see send path below. |
| `max_tokens` | `4096` | int 1–393216 | Per-turn generation cap for run / OpenAI fallback; 384 Ki tokens matches DeepSeek V4 Flash 0731's documented maximum output length. |
| `max_seq` | automatic on growing Qwen VMM KV | int 512–1048576 when set | Default: min(checkpoint `config.text_config.max_position_embeddings`, card capacity after weights, admitted prefill scratch, and VRAM headroom). Explicit user values override either bound, including the trained window. |
| `thinking` | `"on"` | `on` \| `off` | Thinking **mode** (on/off). Independent of effort and of any hard think-token cap. When off wins, effort and cap controls are dropped with a warning. |
| `reasoning_effort` | `"auto"` | `auto` \| `none` \| `low` \| `medium` \| `high` \| `xhigh` \| `max` | **Semantic** effort only — prompt / framing strength. Field meaning never changes by family and is **never** reinterpreted as a token budget. Unsupported or cross-family values are dropped with a warning (see below). |
| `thinking_budget` | `"med"` | `low` \| `med` \| `high` \| `xhigh` \| `max` \| `uncapped` | Legacy **named cap preset** only on models that still honor that config route. Not an effort dial. On effort-native models the string is dropped with a warning (no implicit cap). |
| `max_think_tokens` | *(absent)* | int 0–393216 when set | Explicit **integer** think-span cap (`reasoning.max_tokens`) on Qwen Jinja contracts. `0` or absence = uncapped. Positive N force-closes through the validated Qwen continuation path. DeepSeek, Gemma, Glimmer, and unsupported contracts drop this field with a warning rather than inventing model-specific closure behavior. Independent of mode and effort. |
| `max_total_think_tokens` | `0` | int 0–1000000 | Cross-reopen total `<think>` budget; `0` = off. |

### Reasoning contract (three independent axes)

Thinking controls are three orthogonal axes. A field's meaning never depends on
model family:

| Axis | Keys | Meaning |
|---|---|---|
| Mode | `thinking` / `reasoning.mode`; HTTP `chat_template_kwargs.enable_thinking`, `thinking.type` | Whether thinking is on or off for the turn |
| Effort | `reasoning_effort` / `reasoning.effort`; HTTP `reasoning.effort` / `reasoning_effort` | Semantic prompt strength only — **never** a budget |
| Cap | `max_think_tokens` / `reasoning.max_tokens`; legacy named `thinking_budget` / `reasoning.budget` | Hard think-token limit on contracts that support one; otherwise dropped+warned |

**Thinking budget map** (legacy named presets → integer, only when the loaded
model's contract still accepts the named-cap route):

| Preset | Tokens |
|---|---:|
| `low` | 512 |
| `med` | 2048 |
| `high` | 8192 |
| `xhigh` | 24576 |
| `max` | 32768 |
| `uncapped` | 0 (unlimited) |

**Rules:**

- Mode, effort, and cap are independent. Setting `reasoning_effort=low` does
  **not** invent a think-token cap; setting `thinking_budget=high` does **not**
  change effort.
- **Effort-native** models (Qwen3.8, Ornith 1.5, DeepSeek V4, Muse Glimmer) have **no**
  registry or built-in think-token cap by default. Qwen3.8 and Ornith 1.5 accept the
  explicit integer cap through the validated Qwen continuation path. DeepSeek V4 and Muse
  Glimmer expose effort but no parent-defined independent think-token cap, so
  integer and named cap fields are dropped with a visible warning.
- Registry cards may keep intentional `reasoning_effort` defaults. Omitting a
  budget field is the correct uncapped signal — do not pin effort-native tags
  to a hipfire named cap.
- Thinking disabled wins: effort and cap are dropped with a warning.
- Recognizable unsupported / cross-family values are normalized or dropped with
  `[WARN: INVALID CONFIG]` (server log + OpenAI response metadata under
  `hipfire.reasoning` / `hipfire.config_warnings`). Malformed types and
  out-of-range integers remain hard errors.
- Template probing only checks whether semantic effort is native; it never
  changes field meaning.

**Family examples** (full HTTP surface: [`SERVE.md`](SERVE.md)):

| Family | Mode | Effort | Cap |
|---|---|---|---|
| **Qwen3.8** | `enable_thinking=false` → native empty closed think | `low` \| `medium` \| `xhigh` (default `xhigh`); system-turn semantic prompt steering only — **not** a budget | Positive `max_think_tokens` force-closes; 0/absent uncapped. Named `thinking_budget` dropped+warned |
| **Ornith 1.5** | same Qwen on/off (`enable_thinking=false` → empty closed think) | Qwen3.8-compatible `low` \| `medium` \| `xhigh` semantic prompt steering (default `xhigh`; medium unsteered; **not** a budget reinterpretation) | Same Qwen integer continuation cap as Qwen3.8; 0/absent uncapped. Named `thinking_budget` dropped+warned |
| **Qwen3.6** (non-effort-native template) | same on/off | Effort **dropped+warned** — never converted to a cap. Set `thinking_budget` or `max_think_tokens` explicitly if you want a limit | Named preset or integer as requested |
| **DeepSeek V4** | `thinking.type` enabled/disabled (default on) | `low` \| `high` \| `max`; aliases `medium`/`xhigh`→`high`. | No parent-defined independent cap; integer/named cap fields dropped+warned |
| **Gemma4** | Boolean thinking; official default **off**; Plain/Gemma channel framing | No native effort — unsupported effort/cap dropped+warned | Unsupported; dropped+warned |
| **Muse Glimmer** | Always reasons (Onyx `to=self`); off dropped+warned | `low` \| `medium` \| `high` \| `xhigh` strength; no Qwen `<think>` framing | No parent-defined independent cap; integer/named cap fields dropped+warned |
| **Unsupported** | Controls dropped+warned | dropped+warned | dropped+warned |

**Effective sampling send order** (`request_f64` / `request_u64` / `run`): explicit CLI or HTTP request **>** per-model TOML overlay **>** registry `recommended_settings` **>** daemon/HFQ/arch fallback. This applies to `temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`, and `repeat_penalty`. Bare global sampling values are deliberately not transmitted on the current run/serve path; if one shadows a registry recommendation, the card value is recovered for that model. Registry entry `sampling` blocks are legacy metadata — the resolver reads `recommended_settings`; see [`MODELS.md`](MODELS.md).

`prompt.system` (legacy spelling `system_prompt`) is the corresponding request-scoped framing default. A client-supplied `system` or `developer` message wins; otherwise a per-model TOML value or registry-card `system_prompt` is inserted before the first user message.

---

## KV cache

| Key | Default | Validated values |
|---|---|---|
| `kv_cache` | `"auto"` | `auto`, `q8`, `fp8`, `bf16`, `fwht4`, `fwht3`, `fwht2`, `f32`/`f16` (DeepSeek V4 only); legacy spellings `asym4`, `asym3`, `asym2`, `turbo`, `turbo4`, `turbo3`, `turbo2`, `legacy-asym3` (family-dependent; see below) |
| `kv_k` | empty (unset) | Qwen-only: `q8`, `fwht2`–`fwht4`, `asym2`–`asym4`, `turbo`/`turbo2`–`turbo4`, `legacy-asym2`–`legacy-asym4` |
| `kv_v` | empty (unset) | Qwen-only: `q8`, `lloyd2`–`lloyd4` |
| `kv_adaptive` | `"off"` | `off`, `conservative`, `balanced`, `aggressive`, or `advanced:k=<fwht4\|fwht3\|fwht2>,v=<lloyd4\|lloyd3\|lloyd2>` |
| `kv_backend` | automatic (prefer VMM) | `legacy` \| `vmm` |

### Mode, K, and V

`memory.kv_cache` / `--kv-mode` is the whole-cache preset: quantized modes
seed a split *(K, V)* pair, while native `fp8`/`bf16` are one K+V layout
without an independent V axis. Typed `memory.kv_k` / `memory.kv_v` (CLI
`--kv-k` / `--kv-v`) are **Qwen-only** axis overrides, empty by default and
omitted from IPC when unset. They replace their respective axis of a split
pair; see the native override refusal below.

**Qwen K names** (shared table for `--kv-mode` preset K and `--kv-k`):

- `q8` → Q8 K
- `fwht2` \| `fwht3` \| `fwht4` → signed FWHT K at that bit width
- `asym2` \| `asym3` \| `asym4` and `turbo` \| `turbo2` \| `turbo3` \| `turbo4`
  → **FWHT** K (`turbo` ≡ `fwht3`, `turboN`/`asymN` ≡ `fwhtN`). These are
  legacy spellings kept as aliases; they do not build the old Givens-Asym
  constructors. **Deprecated since 0.4.0, removal in 0.5.0**: spell `fwhtN`.
- `legacy-asym2` \| `legacy-asym3` \| `legacy-asym4` → old Givens-Asym K
  (**legacy**; explicit rollback spelling, `--kv-k` / `memory.kv_k` only —
  `memory.kv_cache` does not accept it). **Deprecated since 0.4.0, removal in
  0.5.0**: fwht3 supersedes Givens asym KV.

Naming any deprecated KV value at load prints one line:
`[hipfire-daemon] warning: KV name '<name>' is deprecated and will be removed in 0.5.0 (…; use fwhtN or q8)`.

`fwht3` (and `fwht2`/`fwht4`) is an optional **headroom** mode on the
Qwen3.5-family sites of every arch: it shrinks K to trade quality for
context/VRAM. No default selects it (only the developer kill switch below).

Qwen3 dense (flat constructor) accepts only `q8` K plus `legacy-asym3` /
`legacy-asym4` at head_dim 256; bare `asymN`/`turboN`/`fwhtN` are refused.
Outside Qwen, the llama-family **HFQ** loader keeps the pre-migration alias
table: bare `asym3`/`asym4`/`turbo4` still build the legacy Givens
constructors, while `asym2`/`turbo2`/`turbo`/`turbo3`/`fwhtN` are unsupported
and fall back to `q8` with a warning. The llama **Dir** loader reads names
through the Qwen table, so bare `asymN`/`turboN` mean FWHT there and fall back
to `q8` with a warning. Those llama spellings are deprecated the same way.

**Qwen V names:** `q8`, `lloyd2`, `lloyd3`, `lloyd4`. Lloyd V **requires**
FWHT K (including `asymN`/`turboN` aliases that resolve to FWHT). Lloyd V
with `q8` or `legacy-asymN` K is rejected before teardown.

**Native pairs are indivisible.** An authored `--kv-mode fp8` or `bf16`
(or the same via config) cannot be combined with any authored K or V axis
override — even when the override would equal the native axis. Adaptive
plus a fixed K/V override is likewise refused. An `auto` selection that
landed on eligible gfx1201 native FP8 is **not** an authored `fp8` mode:
both axes together may replace that pair when the constructor supports
them; a single axis against native FP8 refuses and advises `--kv-mode q8`
or both axes (never FP8-K/Q8-V).

Unsupported K/V by topology, head dimension, or arch; static CASK with
FWHT/Lloyd-V; and explicit native FP8/BF16 on unsupported arch/PP all fail
**before** destructive teardown with requested/effective values and
supported alternatives. No silent fallback on unsupported Qwen sites.

**Resolution of `auto` / unset (Qwen family):** native **fp8** (one K+V
layout) on exact **gfx1201** when the load is native-eligible: attention
geometry H24 / Hkv4 / D256, single GPU (`pp = tp = 1`), no `kv_adaptive`,
no CASK sidecar. Every other Qwen load resolves **q8/q8**: gfx1100, gfx1151
and every other arch; gfx1201 with a different geometry, with adaptive or
CASK, or on PP / dense TP / MoE EP (those policies have no native
constructor); and Qwen3 dense (flat constructor). gfx1200 / gfx11 / gfx94x
never inherit fp8. Registry Qwen cards leave mode as `auto` (a registry
`default_kv_mode = "q8"` is not lowered into config).

**Non-Qwen families:** Maple keeps BF16/BF16 (registry and direct-path
auto; explicit `--kv-mode q8` still works). Gemma4 eager stays Q8/Q8.
Gemma4 lowered keeps the sliding Q8 ring; its full-attention tier follows
`kv_cache`: `auto`/`q8` → Q8 on every arch (no lowered Gemma4 attend site
admits native FP8), `legacy-asym3` → the previous Givens Asym3 K + Q8 V,
`fp8`/`bf16` → load error, anything else → Q8 with a warning. llama HFQ,
MiniMax and LFM2-MoE resolve `auto` to q8. DeepSeek4 keeps its F32
compressor default (or explicit F16); V is not independently selectable
there. Other carriers keep their existing site policy (llama, MiniMax and
LFM2-MoE resolve `auto` to q8).

**Qwen3.8-Flash-Next / Qwen4 (arch 16)** has its own table: `kv_cache`
sets the storage of the QSA full-attention K/V arenas. The GatedDeltaNet
recurrent state is separate: Qwen3.5's Q8 DeltaNet format by default on every
arch, `fp32` through the `state_quant` load parameter.

| `kv_cache` | gfx1201 | gfx1100, gfx1151 (Halo), other arches |
|---|---|---|
| `auto` / unset | `fp8` | `bf16` |
| `bf16` | exact reference state (F32 K/V arenas, BF16-valued index keys) | same |
| `fp8` | E4M3 K/V with one f16 scale per head and token; indexer raw/pooled keys stored as BF16 (the same values) | refused |
| anything else (`q8`, `fwhtN`, `f16`, …) | refused | refused |

`auto` picks fp8 only on exact gfx1201 and only for a head geometry the
kernels implement (head_dim 256, even KV-head count; Flash-Next qualifies);
otherwise it stays `bf16`. The native MTP layer keeps the exact state. On
Flash-Next, fp8 cuts each trunk QSA layer's context state from 4,736 to
1,352 bytes per token, and the Q8 GDN state is 3.88× smaller than F32;
`memory.max_seq` goes up to the native 262,144.

**Kill switch:** `HIPFIRE_QWEN_KV_DEFAULT_Q8=0` (developer variable) restores
the prior *implicit Qwen* per-site default wherever `auto` does not pick
native fp8: HFQ `auto`/unset → FWHT3 K + Q8 V, PaRo `auto` → FWHT3/Q8,
PaRo unset → Q8/Q8, other sites Q8/Q8. Default is ON (Q8/Q8). It must not
override any authored CLI/config mode or K/V, must not touch the eligible
gfx1201 fp8 path, and must not affect non-Qwen families.

**Precedence** (each key independently — mode, K, V, backend):

CLI flag **>** one-shot env **>** per-model TOML **>** global TOML **>**
registry preset / arch default.

Mode supplies the initial layout; axis overrides apply to split modes
regardless of which layer authored the mode. Effective resolution logs
`requested mode=…, effective K=…, effective V=…` for split layouts or
`requested mode=…, effective KV=fp8` (respectively `bf16`) for native.

`kv_adaptive` is opt-in and is a *changing tier* path (starts FWHT4/Q8,
downshifts toward FWHT/Lloyd floors), not a fixed static pair. With
adaptive on, `max_seq` is the context guaranteed at the floor tier. Daemon
param overrides `HIPFIRE_KV_ADAPTIVE` when set through the CLI load path.
Adaptive plus authored `--kv-k`/`--kv-v` is rejected.

### `kv_backend` (allocation backend)

Accepted values are **`legacy`** and **`vmm` only**. The former spelling
`contiguous` is **rejected** with an actionable migration message naming
`legacy` (e.g. use `--kv-backend legacy` or
`memory.kv_backend = "legacy"`). It is not a deprecated alias.

Omitted `memory.kv_backend` / `--kv-backend` is **automatic**. Automatic
selects on-demand HIP VMM mapping on certified Qwen layout/device/topology
combinations (single-GPU and dense TP with per-rank VMM), covering static
modes, adaptive, and native `fp8`/`bf16`. Otherwise it falls back to
**legacy** once per load, logging the reason and the effective backend.
Absolute HFQ/safetensors paths and registry tags resolve identically for
the same source/mode/topology; registry cards set neither backend nor
`memory.max_seq`, but may still set `generation.max_tokens`. On
gfx1100/gfx1151 the certified Qwen VMM route is q8 only; unvalidated modes
retain the existing legacy fallback.

For single-card **dense** Qwen HFQ VMM loads, omission of `max_seq` first
preflights projected model fit before tearing down a resident model. After
weights actually load, the Qwen carrier measures free VRAM immediately before
KV construction. It subtracts only the minimum viable 512-row prefill scratch,
the existing 1 GiB transient headroom and VMM page rounding to derive card
capacity; the preflight weight projection never caps the effective context.
Only full-attention layers carry KV (16 of 64 for Qwen3.8-27B), so card capacity
divides that remaining memory by the selected K+V token stride. VMM reserves
virtual space to `max_seq` and maps physical pages on demand; without eviction,
`physical_cap == max_seq`, while deprecated CASK eviction can make `physical_cap < max_seq`.
Larger prefill PBS allocations are lazy: each request chooses the widest rung
that fits currently free VRAM after mapped (not virtually reserved) KV. When
future KV growth needs the memory, widened PBS reuse is released between
requests, allowing later chunks to narrow rather than making growth OOM.
The loaded diagnostic reports `max_seq`, its `model`/`card`/`user` bound,
`model_ctx`, `card_cap` and KV mode. Qwen MoE keeps its prior 32768 built-in
default (with a loaded diagnostic reason) until its distinct prefill scratch can
be budgeted; other carriers keep their existing context behavior until their
different cache ownership can use the same allocation accounting.

**Explicit backend override precedence** (forced choices, not automatic
policy): CLI `--kv-backend` **>** per-model TOML `memory.kv_backend` **>**
global TOML `memory.kv_backend`. Explicit `legacy` always wins and never
emits a misleading VMM marker. Explicit `vmm` on an unsupported combination
fails with an actionable capability error **before** resident-model teardown
and never silently falls back (refusal may suggest `--kv-backend legacy`).

Unsupported automatic cases (for example MoE EP/PP, non-Qwen carriers without a
matching owner, uncertified device/OS, or missing HIP VMM symbols) keep
legacy service with a logged reason. The deprecated adaptive→CASK handoff **requires**
VMM: if VMM is unavailable that combination is refused rather than handed off as
invalid legacy. Private speculative draft caches (DFlash/MTP) are owned
separately from the trunk KV backend and do not relabel it; their actual backend
is logged on its own.

On Windows (ROCm 7.2), VMM maps the full reservation upfront for correctness —
do not expect on-demand VRAM savings there. There is no env-based second default
for the backend (`memory.kv_backend` has no `HIPFIRE_*` compat binding).

One-shot aliases: `HIPFIRE_KV_MODE` for **mode** only; see also
`HIPFIRE_QWEN_KV_DEFAULT_Q8` above and [`env-vars.md`](env-vars.md).

---

## Attention

| Key | Default | Values |
|---|---|---|
| `flash_mode` | `"auto"` | `auto` \| `always` \| `never` |

Lowered directly into the daemon snapshot. `HIPFIRE_ATTN_FLASH` remains a
legacy one-shot alias. The Qwen3.5 MTP head inherits the trunk
`flash_mode` / `attention_flash_mode` (it does not pin a separate non-flash path).

---

## Speculative decode

Only **one** mechanism runs. Canonical selector:

| Key | Default | Values |
|---|---|---|
| `speculation` | `"auto"` | `off` \| `auto` \| `ngram` \| `dflash` \| `mtp` \| `dspark` |

- **`off`** — AR only.
- **`auto`** — cascade by availability / eligibility; legacy mode knobs filter each mechanism. Under auto, DSpark can win over in-trunk MTP when a DSpark sidecar is present (CLI comments).
- Forced mechanism names bypass heuristics; missing prerequisites fall back to AR with a warning.

Legacy one-shot alias: `HIPFIRE_SPECULATION`. CLI: `--spec`.

### Mechanism knobs

| Key | Default | Values / range | Notes |
|---|---|---|---|
| `dflash_mode` | `"off"` | `on` \| `off` \| `auto` | **Default off.** `auto` enables on dense Qwen3.5-class targets and skips known-loss A3B cases. |
| `vision_mode` | `"off"` | `on` \| `off` \| `auto` | **Default off.** Tower sidecar gate — see [Vision tower](#vision-tower). |
| `dflash_adaptive_b` | `false` | bool | **Opt-in.** Adaptive verify-block width: follows the trailing 8-cycle acceptance depth (τ̂+2), full below 2k context. Not output-identical (window boundaries move; per-position sampling stays target-lossless). Auto-suppressed on retained-PM4 verify loads; `HIPFIRE_DFLASH_ADAPTIVE_B=0` forces fixed. |
| `dflash_ngram_block` | `"auto"` | `true` \| `false` \| `"auto"` | Verify-path n-gram defense; auto size-gates. |
| `mtp_mode` | `"auto"` | `off` \| `on` \| `auto` | Built-in MTP when a head is present: the DeepSeek V4 trunk's MTP layer, or for Qwen a bundled `.mq4-mtp` trailer or a `.mtp` sidecar (the registry `mtp` slot; Qwen3.8-27B ships one — [MODELS.md](MODELS.md#dflash-draft-artifacts-registry)). `auto` uses a present head; `on` fails the load without one. |
| `mtp_k` | `3` | int 1–10 | |
| `dspark_conf_threshold` | `null` | `null` or number 0.0–1.0 | `null` ⇒ per-arch carrier default (qwen3 0.1 / deepseek4 0.3 in comments). |
| `ngram_mode` | `"off"` | `off` \| `on` \| `auto` | Model-free; byte-identical to AR when used. |
| `ngram_k` | `12` | int 2–32 | |
| `ngram_min_count` | `2` | int 1–10 | |
| `ddtree_budget` | `0` | int 0–64 | `0` = chain DFlash (no tree). |
| `ddtree_topk` | `4` | int 1–8 | |

Legacy compatibility input still wins at the top of the startup ladder for
the corresponding knobs, but the engine receives only the resolved immutable
snapshot. Full aliases: [`env-vars.md`](env-vars.md).

---
## Vision tower

| Key | Default | Values / range |
|---|---|---|
| `vision_mode` | `"off"` | `off` \| `auto` \| `on` |

- **`off`** (default) — never load a tower sidecar. The registry/sibling file is not wired, and even an explicit `run --vision` / `serve --vision` / `HIPFIRE_VISION_SIDECAR` path is skipped with one stderr line. The daemon enforces the same hard override for non-CLI clients, mirroring `dflash_mode=off`. Text loads pay no tower VRAM (~1 GB).
- **`auto`** — use the registry `vision` slot (or the `<trunk-stem>-vision.hfq` sibling beside the trunk) when present; silently text-only when absent.
- **`on`** — require the declared sidecar: the load fails closed with a pull hint when it cannot be resolved. A trunk with an embedded tower declares no sidecar and is unaffected by this key.

```bash
hipfire config set vision_mode auto
hipfire config qwen3.8:27b set vision_mode auto   # per-model overlay
HIPFIRE_VISION_MODE=auto hipfire run qwen3.8:27b  # one-shot
```

Loading detail: [`MODELS.md`](MODELS.md). Env inventory: [`env-vars.md`](env-vars.md).

---

## MMQ screening

| Key | Default | Values / range |
|---|---|---|
| `mmq_screen` | `"auto"` | `off` \| `on` \| `auto` |
| `mmq_screen_threshold` | `0.10` | number (0, 1] |

MMQ policy is configured with `kernel.mmq = "auto" | true | false` and the
experimental weight-only route with `kernel.wo_mmq`. Screening only matters
when MMQ is on. Daemon arch-gates the sweep (RDNA3/3.5 family in source
comments). Legacy `HIPFIRE_MMQ` / `HIPFIRE_WO_MMQ` values remain one-shot
compatibility overrides.

---

## CASK / TriAttention eviction (deprecated, removal in 0.5.0)

CASK / TriAttention KV eviction (including FlashCASK under DFlash) is
deprecated since 0.4.0 and will be removed in 0.5.0. The keys below still
load, but the path is **not supported**, not a recommended setting, and not an
acceptance route. Every load that sets `cask_sidecar` or `cask=true` prints one
warning line:
`[hipfire-daemon] warning: CASK is deprecated and will be removed in 0.5.0; not supported`.
All defaults leave it off, and a downloaded `.triattn*.bin` sidecar never
attaches unless `cask_sidecar` or `cask_auto_attach=true` is set.

| Key | Default | Range |
|---|---|---|
| `cask_sidecar` | `""` | path string; empty = disabled |
| `cask` | `false` | bool (m-fold vs plain drop) |
| `cask_budget` | `512` | int 64–65536 |
| `cask_beta` | `128` | int 0–65536 |
| `cask_handoff_tokens` | `0` | int 0–1048576 |
| `cask_core_frac` | `0.5` | number 0.0–1.0 |
| `cask_fold_m` | `2` | int 1–16 |
| `cask_auto_attach` | `false` | bool |

Sidecar generation (deprecated with CASK): `hipfire sidecar-gen` — [`CLI.md`](CLI.md).

---

## Prompt processing

| Key | Default | Values |
|---|---|---|
| `prompt_normalize` | `true` | bool |

Collapse `\n{3,}` → `\n\n` at engine entry. Legacy alias:
`HIPFIRE_NORMALIZE_PROMPT` (`0`/`false`/`off`/`no` disable).

---

## PFlash speculative prefill (deprecated, removal in 0.5.0)

| Key | Default | Range / values |
|---|---|---|
| `prefill_compression` | `"off"` | `off` \| `auto` \| `always` |
| `prefill_threshold` | `32768` | int 0–1048576 |
| `prefill_keep_ratio` | `0.05` | (0, 1] |
| `prefill_alpha` | `0.85` | [0, 1] |
| `prefill_min_keep` | `2048` | int 0–1048576 |
| `prefill_sink` | `256` | int 0–65536 |
| `prefill_recent` | `1024` | int 0–65536 |
| `prefill_block` | `128` | int 1–4096 |
| `prefill_drafter` | `""` | path |
| `prefill_drafter_device` | `-1` | int −1–15 (`-1` = same device as target) |
| `prefill_profile` | `false` | bool |
| `prefill_sparse_threshold` | `32768` | int 0–1048576 |

**Status:** deprecated since 0.4.0, removal in 0.5.0; not supported. Prefix
caching supersedes it. Setting `prefill_compression` to anything but `off`
prints one line at load:
`[hipfire-daemon] warning: PFlash is deprecated and will be removed in 0.5.0; not supported, prefix caching supersedes it (prefill_compression=<mode> set at load)`.

Off by default. Typed fields cover product PFlash policy; remaining experiments
live under `[developer]` and retain matching `HIPFIRE_PREFILL_*` one-shot
aliases. Detailed bypass reasons and serve wiring: [`SERVE.md`](SERVE.md) /
runtime PFlash module — not restated here.

---

## Server / serve admission

| Key | Default | Range |
|---|---|---|
| `host` | `"127.0.0.1"` | non-empty hostname/IP, no whitespace, ≤255 (`0.0.0.0` = all interfaces) |
| `port` | `11435` | int 1–65535 |
| `idle_timeout` | `300` | int 0–86400 seconds (`0` = never unload) |
| `default_model` | `"qwen3.5:9b"` | non-empty tag/path string |
| `serve.local` | `false` | Force the current command to use a locally spawned daemon. |
| `max_request_bytes` | `67108864` (64 MiB) | int 4096–4GiB |
| `serve_max_queue` | `64` | int 0–100000 (`0` = uncapped depth) |
| `serve_queue_timeout_ms` | `600000` (10 min) | int 0–3600000 (`0` = no wait timeout). Serve runs one generation at a time by default, so this must cover a full generation. |
| `serve.allow_request_pull` | `false` | bool. Let a chat request that names a registry model not on disk download it; off → 404 "run `hipfire pull`". Env: `HIPFIRE_SERVE_ALLOW_REQUEST_PULL`. |
| `serve.allow_request_paths` | `false` | bool. Let a chat request load any readable file it names; off → only installed models (models directory, catalog paths, the pre-warm model). Env: `HIPFIRE_SERVE_ALLOW_REQUEST_PATHS`. |
| `serve.max_queue_bytes` | `268435456` | int 1–1TiB. Multi-slot route: total request bytes allowed to wait in the admission queue |
| `serve.max_batch_tokens` | `4096` | int 1–1048576 (global trunk-row budget) |
| `serve.prefill_min_tokens` | `1` | int 1–1048576 (prefill quantum) |
| `serve.prefix_cache` | `false` | bool — experimental; off until route admission |
| `serve.prefix_cache_max_bytes` | `0` | int 0–1TiB (0 = no retained cache) |
| `serve.structured_jump_forward` | `false` | bool — experimental |
| `serve.stream_stall_timeout_ms` | `30000` | int 0–3600000. Multi-slot route: a streaming client that leaves its response channel full (or a final frame unflushed) this long is aborted and its queue permit released |
| `experimental_budget_alert` | `false` | bool |
| `serve.multi_slot` | `false` | Serve concurrent requests on the multi-slot engine instead of one at a time. Needs `memory.kv_cache = "q8"` set explicitly: the slot engine refuses every other value, including the default `auto`. Concurrent requests can produce different greedy text than serial requests at ≥4 slots; output is deterministic for a fixed batch composition. 2 slots matched serial in earlier testing; on 35B-A3B MoE checkpoints, 2 slots also diverged from serial on 1–4 of 4 prompts. A pre-warm that fails (for example, slot-engine allocations that do not fit beside another process on the card) exits serve with the load error rather than serving lazily ([`SERVE.md`](SERVE.md) § Lifecycle). |
| `serve.multi_slot_slots` | `4` | int 1–64 concurrent slots. |
| `serve.multi_slot_ctx` | `8192` | int 512–1048576 per-slot context capacity (tokens). |
| `serve.multi_slot_prefill_chunk` | `1024` | int 1–1048576. Prefill tokens taken from one slot per multi-slot step; batch scratch is sized `n_slots ×` this. Env: `HIPFIRE_SERVE_MULTI_SLOT_PREFILL_CHUNK`. |

Serve HTTP surface: [`SERVE.md`](SERVE.md). The corresponding `HIPFIRE_MODEL`,
`HIPFIRE_IDLE_TIMEOUT`, `HIPFIRE_MAX_REQUEST_BYTES`,
`HIPFIRE_SERVE_MAX_QUEUE`, `HIPFIRE_SERVE_QUEUE_TIMEOUT_MS`, and
`HIPFIRE_SERVE_MULTI_SLOT*` names are legacy one-shot aliases.

---

## Memory

### `memory.oom_guard`

| Key | Default | Values |
|---|---|---|
| `memory.oom_guard` | `auto` | `auto` \| `true`/`on`/`1` \| `false`/`off`/`0` |

Compat env: `HIPFIRE_OOM_GUARD`. Used by `kv_slots::preflight_alloc`, the
`SlotPool` arena check, and the CLI bench-sweep headroom path.

Two checks exist; only one is gated:

- **Host `MemAvailable` headroom** — gated by `memory.oom_guard`. Default
  `auto` turns it **on** for unified-memory APU arches (`gfx1035` / `gfx1036` /
  `gfx1103` / `gfx1150`–`gfx1152`: GPU allocations come from system RAM, so an
  overshoot is a desktop-killing OOM), **off** for discrete GPUs (overshoot is
  a failed `hipMalloc`), and for GPU-less processes by host swap state (no
  swap → on). Explicit `true`/`false` force either way; `auto` logs its
  decision once.
- **R9700 deployment-target VRAM budget** (32 GiB class ceiling in
  `preflight_alloc`) — **always runs**, on every arch, whether the host
  headroom guard is active or not. A configuration that does not fit the
  deployment target is refused regardless of this box's RAM.

`scripts/run-bounded.sh` (`HIPFIRE_MEM_CAP`) remains the hard cgroup backstop.

### Prompt / assistant-turn cache

| Key | Default | Values |
|---|---|---|
| `memory.prompt_cache_capacity` | `32` | int ≥0; maximum cached assistant-turn tokenizations (`0` keeps none). Env: `HIPFIRE_PROMPT_CACHE_CAP`. |
| `memory.prompt_cache_unbounded` | `false` | Remove the capacity bound. Env: `HIPFIRE_PROMPT_CACHE_UNBOUNDED`. |

Qwen AR and DFlash multi-turn reuse store each completed assistant turn as the
**verbatim generated token span** (whole envelope: full body tokens, plus
producer reasoning text when the turn thought). On the next turn, Jinja history
replay splices that span through the model's trained template framing so the
LCP prefix matches the prior bake. Unedited rich `reasoning_content` history
hits; edited or mismatched history falls back to a plain retokenized render
(cold or checkpoint path) instead of replaying stale tokens.

Multi-turn DFlash and the prefix cache: when a DFlash turn ends on EOS (or the
think cap) mid-window, **RepairForTerminal** restores the pre-window recurrent
state and replays only the consumed prefix so the prompt/prefix cache stays
warm. The next turn prefills only the new suffix instead of a full cold
prefill (the previous fail-closed path reset and invalidated the cache).

---

## Chat template overrides

| Key | Default | Validation |
|---|---|---|
| `chat_template` | `""` | empty or existing file path (`~/` expanded); existence + `isFile` only — no readability/access check |
| `default_chatml` | `true` | bool |

Lowered directly into the startup/load policy. `HIPFIRE_CHAT_TEMPLATE_FILE`
and `HIPFIRE_DEFAULT_CHATML` remain legacy one-shot aliases.

---

## Per-model overlay

```bash
hipfire config qwen3.5:9b set dflash_mode off
hipfire config qwen3.5:9b          # list resolved model policy
```

Only explicitly set keys are stored; others inherit global. Primary path:
`~/.hipfire/models.toml`. Legacy `models.json` and `per_model_config.json` are
migration inputs and are not native write targets.

---

## Legacy-compatible one-shot env overrides (examples)

Full inventory: [`env-vars.md`](env-vars.md). Prefer `hipfire config set` for
persistent policy. Parent-process values still take precedence for one-shot
bisects and compatibility:

```bash
HIPFIRE_KV_MODE=q8
HIPFIRE_ATTN_FLASH=never
HIPFIRE_NORMALIZE_PROMPT=0
HIPFIRE_SPECULATION=off
HIPFIRE_DFLASH_DRAFT=/path/to/draft.hfq
HIPFIRE_LOCAL=1
HIPFIRE_MODEL=qwen3.5:9b
```

---

## Typed daemon snapshots

The CLI resolves TOML, registry, one-shot flags, and compatibility input into a
versioned `ProcessConfig` before the GPU is initialized. The daemon revalidates
that envelope and lowers stable runtime values through
`RuntimeConfig::from_process_config`; `FeatureFlags` performs the corresponding
kernel-policy lowering. Direct daemon invocation resolves local TOML plus the
compatibility environment into the same `ProcessConfig` first. Neither path
uses ambient variables in engine hot paths.

| Field | TOML key | Legacy env | Default behavior |
|---|---|---|---|
| `normalize_prompt` | `prompt.normalize` | `HIPFIRE_NORMALIZE_PROMPT` | true unless `0`/`false`/`off`/`no` |
| `prompt_token_heat` | `diagnostic.prompt_token_heat` | `HIPFIRE_PROMPT_TOKEN_HEAT` | off unless `1` |
| `prompt_heat_json` | `diagnostic.prompt_heat_json` | `HIPFIRE_PROMPT_HEAT_JSON` | off unless `1` |
| `prompt_heat_limit` | `diagnostic.prompt_heat_limit` | `HIPFIRE_PROMPT_HEAT_LIMIT` | 64 |
| `dflash_mode` | `speculation.dflash` | `HIPFIRE_DFLASH_MODE` | `"off"` |
| `vision_mode` | `vision.mode` | `HIPFIRE_VISION_MODE` | `"off"` |
| `draft_f16` | `speculation.draft_f16` | `HIPFIRE_DRAFT_F16` | true unless `0` |
| `draft_gemm_dump` | `diagnostic.draft_gemm_dump` | `HIPFIRE_DRAFT_GEMM_DUMP` | off unless `1` |
| `draft_subphase` | `diagnostic.draft_subphase` | `HIPFIRE_DRAFT_SUBPHASE` | off unless `1` |
| `ddtree_budget` | `speculation.ddtree_budget` | `HIPFIRE_DDTREE_BUDGET` | 256 |
| `ddtree_topk` | `speculation.ddtree_topk` | `HIPFIRE_DDTREE_TOPK` | 8 |
| `prefill_batched` | `kernel.prefill_batched` | `HIPFIRE_PREFILL_BATCHED` | true unless `0` |
| `flash_partials_batch` | `kernel.flash_partials_batch` | `HIPFIRE_FLASH_PARTIALS_BATCH` | unset |
| `tp_use_rccl` | `hardware.tp_use_rccl` | `HIPFIRE_TP_USE_RCCL` | unset → RCCL default on; `0`/`false` opt out |
| `ngram_loop_threshold` | `generation.loop_guard_threshold` | `HIPFIRE_NGRAM_LOOP_THRESHOLD` | **0 (off)** |
| `ngram_window` | `generation.loop_guard_window` | `HIPFIRE_NGRAM_WINDOW` | 256 |
| `ngram_draft` | `speculation.ngram` | `HIPFIRE_NGRAM_DRAFT` | off |
| `ngram_k` | `speculation.ngram_k` | `HIPFIRE_NGRAM_DRAFT_K` | 12 |
| `ngram_min_count` | `speculation.ngram_min_count` | `HIPFIRE_NGRAM_MIN_COUNT` | 2 |
| `prompt_cache_capacity` | `memory.prompt_cache_capacity` | `HIPFIRE_PROMPT_CACHE_CAP` | 32 |
| `prompt_cache_unbounded` | `memory.prompt_cache_unbounded` | `HIPFIRE_PROMPT_CACHE_UNBOUNDED` | false |
| `devices` | `hardware.devices` | `HIPFIRE_DEVICES` (alias `HIPFIRE_DEVICE`) | unset |
| `allow_mixed_arch` | `hardware.allow_mixed_arch` | `HIPFIRE_ALLOW_MIXED_ARCH` | false unless `1` |
| `uniform_vram_tolerance_gb` | `hardware.uniform_vram_tolerance_gb` | `HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB` | unset |
| `mtp_mode` | `speculation.mtp` | `HIPFIRE_MTP_MODE` | `"auto"` |
| `mtp_k` | `speculation.mtp_k` | `HIPFIRE_MTP_K` | 3 |

`kernel.lm_head_f16` maps the live Qwen3.5/3.6 loader compatibility control
`HIPFIRE_LM_HEAD_F16`; the duplicate unused `RuntimeConfig` member was removed.
The similarly unused `RuntimeConfig::dflash_draft` member was removed rather
than promoted. Draft selection remains part of the typed speculation/load
policy and filename/registry discovery.

Note: CLI `ddtree_budget` default is `0` while bare `RuntimeConfig` default is `256` when only env/runtime path is used — CLI load params are the product path for `hipfire run`/`serve`.

Redline eligibility helpers `mq4r_redline_default` and `retained_redline_default` (same file) are **narrow runtime-default predicates** for replay backend selection: exact GPU arch `gfx1100`/`gfx1151`/`gfx1201` + case-insensitive `.mq4r` + pp=tp=1 (model-family agnostic, no `arch_id` gate; `gfx1200` and all other arches remain opt-in), plus Qwen3.5 dense (`qwen3_5`) plain-AR decode on exact `gfx1201` with pp=tp=1 and no drafter. They are not general config keys and not Redline certification/registry admission. `replay.backend = "hip"`, the built-in `hip` config profile or another explicit backend selection disables the automatic default. Policy: [`REDLINE.md`](REDLINE.md).

---

## Lifecycle status

Every schema key carries one lifecycle status. The table is generated from
`crates/hipfire-config/src/lib.rs` by `scripts/check-lifecycle.py --write`, and
`scripts/no-gpu-ci.sh` fails when it drifts. The status is `deprecated` when the
field carries `// lifecycle: deprecated since 0.4.0, removal 0.5.0 — <reason>`,
otherwise `experimental` or `stable` from the schema's `experimental` flag.
Deprecated keys still load in 0.4.0 and warn once when they switch a deprecated
feature on; their defaults are inert, so default configs never reach
deprecated code. Env statuses live in [`env-vars.md`](env-vars.md#lifecycle-status).

Deprecated since 0.4.0, removal in 0.5.0:

- `memory.cask.*` (CASK / TriAttention eviction).
- `speculation.prefill.*` and `diagnostic.pflash.score_layer` (PFlash; prefix caching supersedes it).
- KV **values** `asym2|asym3|asym4`, `turbo|turbo2|turbo3|turbo4` (`kv_cache`, `kv_k`) and `legacy-asym2|3|4` (`kv_k`): Givens asym KV and its aliases; use `fwhtN` or `q8`.

<!-- lifecycle-table:begin (generated by scripts/check-lifecycle.py --write) -->

**Count:** 270 schema keys, plus the `developer.<name>` namespace.

| Key | Legacy key | Env | Lifecycle |
|---|---|---|---|
| `attention.ck_runtime_lib` | `ck_runtime_lib` | `HIPFIRE_FLASH_ATTN_CK_LIB` | experimental |
| `attention.ck_workspace_bytes` | `ck_workspace_bytes` | `HIPFIRE_FLASH_ATTN_CK_WORKSPACE_BYTES` | experimental |
| `attention.flash` | `flash_mode` | `HIPFIRE_ATTN_FLASH` | stable |
| `diagnostic.blob_force` | `blob_force` | `HIPFIRE_BLOB_FORCE` | experimental |
| `diagnostic.compiler.hipcc_extra_flags` | `hipcc_extra_flags` | `HIPFIRE_HIPCC_EXTRA_FLAGS` | experimental |
| `diagnostic.compiler.no_device_compiler` | `no_device_compiler` | `HIPFIRE_NO_DEVICE_COMPILER` | experimental |
| `diagnostic.draft_gemm_dump` | `draft_gemm_dump` | `HIPFIRE_DRAFT_GEMM_DUMP` | experimental |
| `diagnostic.draft_subphase` | `draft_subphase` | `HIPFIRE_DRAFT_SUBPHASE` | experimental |
| `diagnostic.gemm_dump` | `gemm_dump` | `HIPFIRE_GEMM_DUMP` | experimental |
| `diagnostic.kernel.ddtree_logw_cutoff` | `ddtree_logw_cutoff` | `HIPFIRE_DDTREE_LOGW_CUTOFF` | experimental |
| `diagnostic.kernel.fp16_layer_max` | `fp16_layer_max` | `HIPFIRE_FP16_LAYER_MAX` | experimental |
| `diagnostic.kernel.fp16_layer_min` | `fp16_layer_min` | `HIPFIRE_FP16_LAYER_MIN` | experimental |
| `diagnostic.kernel.gate_up_variant` | `gate_up_variant` | `HIPFIRE_GATE_UP_VARIANT` | experimental |
| `diagnostic.kernel.gemv_rows` | `gemv_rows` | `HIPFIRE_GEMV_ROWS` | experimental |
| `diagnostic.kernel.gfx11_weight_load_policy` | `gfx11_weight_load_policy` | `HIPFIRE_GFX11_WEIGHT_LOAD_POLICY` | experimental |
| `diagnostic.kernel.gfx12_weight_load_policy` | `gfx12_weight_load_policy` | `HIPFIRE_GFX12_WEIGHT_LOAD_POLICY` | experimental |
| `diagnostic.kernel.gfx942_mfma_prefill` | `gfx942_mfma_prefill` | `HIPFIRE_GFX942_MFMA_PREFILL` | experimental |
| `diagnostic.kernel.hfq3_mmq_layer_max` | `hfq3_mmq_layer_max` | `HIPFIRE_HFQ3_MMQ_LAYER_MAX` | experimental |
| `diagnostic.kernel.hfq3_mmq_layer_min` | `hfq3_mmq_layer_min` | `HIPFIRE_HFQ3_MMQ_LAYER_MIN` | experimental |
| `diagnostic.kernel.lloyd_mb4` | `lloyd_mb4` | `HIPFIRE_LLOYD_MB4` | experimental |
| `diagnostic.kernel.mmq_min_batch` | `mmq_min_batch` | `HIPFIRE_MMQ_MIN_BATCH` | experimental |
| `diagnostic.kernel.mq3_mb4` | `mq3_mb4` | `HIPFIRE_MQ3_MB4` | experimental |
| `diagnostic.kernel.rdna2_variant` | `rdna2_variant` | `HIPFIRE_RDNA2_VARIANT` | experimental |
| `diagnostic.kernel.rocblas_min_batch` | `rocblas_min_batch` | `HIPFIRE_ROCBLAS_MIN_BATCH` | experimental |
| `diagnostic.kernel.wo_wmma_variant` | `wo_wmma_variant` | `HIPFIRE_WO_WMMA_VARIANT` | experimental |
| `diagnostic.mmq_quantize_only` | `mmq_diag_quantize_only` | `HIPFIRE_MMQ_DIAG_QUANTIZE_ONLY` | experimental |
| `diagnostic.pflash.score_layer` | `pflash_score_layer` | `HIPFIRE_PFLASH_SCORE_LAYER` | deprecated |
| `diagnostic.prompt_heat_json` | `prompt_heat_json` | `HIPFIRE_PROMPT_HEAT_JSON` | experimental |
| `diagnostic.prompt_heat_limit` | `prompt_heat_limit` | `HIPFIRE_PROMPT_HEAT_LIMIT` | experimental |
| `diagnostic.prompt_token_heat` | `prompt_token_heat` | `HIPFIRE_PROMPT_TOKEN_HEAT` | experimental |
| `diagnostic.qkv_bias` | `fuse_qkv_bias_debug` | `HIPFIRE_FUSE_QKV_BIAS_DEBUG` | experimental |
| `diagnostic.replay.dispatch_profile` | `replay_dispatch_profile` | `HIPFIRE_REDLINE_DISPATCH_PROFILE` | experimental |
| `diagnostic.replay.gfx1151_cu_count` | `gfx1151_redline_cu_count` | `HIPFIRE_GFX1151_REDLINE_CU_COUNT` | experimental |
| `diagnostic.replay.gfx1151_entry_acquire` | `gfx1151_pm4_entry_acquire` | `HIPFIRE_GFX1151_PM4_ENTRY_ACQUIRE` | experimental |
| `diagnostic.replay.gfx1151_initiator` | `gfx1151_pm4_initiator` | `HIPFIRE_GFX1151_PM4_INITIATOR` | experimental |
| `diagnostic.replay.gfx1151_interleave` | `gfx1151_pm4_interleave` | `HIPFIRE_GFX1151_PM4_INTERLEAVE` | experimental |
| `diagnostic.replay.gfx1151_resource_limits` | `gfx1151_pm4_resource_limits` | `HIPFIRE_GFX1151_PM4_RESOURCE_LIMITS` | experimental |
| `diagnostic.replay.manual_capture` | `replay_manual_capture` | `HIPFIRE_REPLAY_MANUAL_CAPTURE` | experimental |
| `diagnostic.replay.pm4_acquire_policy` | `replay_pm4_acquire_policy` | `HIPFIRE_REPLAY_PM4_ACQUIRE_POLICY` | experimental |
| `diagnostic.replay.pm4_dynamic_grid` | `replay_pm4_dynamic_grid` | `HIPFIRE_REPLAY_PM4_DYNAMIC_GRID` | experimental |
| `diagnostic.replay.pm4_gcr_trim` | `replay_pm4_gcr_trim` | `HIPFIRE_REPLAY_PM4_GCR_TRIM` | experimental |
| `diagnostic.replay.pm4_kernarg_pool` | `replay_pm4_kernarg_pool` | `HIPFIRE_PM4_KERNARG_POOL` | experimental |
| `diagnostic.replay.pm4_max_parallel_phases` | `replay_pm4_max_parallel_phases` | `HIPFIRE_REPLAY_PM4_MAX_PARALLEL_PHASES` | experimental |
| `diagnostic.replay.pm4_min_parallel_width` | `replay_pm4_min_parallel_width` | `HIPFIRE_REPLAY_PM4_MIN_PARALLEL_WIDTH` | experimental |
| `diagnostic.replay.pm4_min_parallel_workgroups` | `replay_pm4_min_parallel_workgroups` | `HIPFIRE_REPLAY_PM4_MIN_PARALLEL_WORKGROUPS` | experimental |
| `diagnostic.replay.pm4_native_phases` | `replay_pm4_native_phases` | `HIPFIRE_REPLAY_PM4_NATIVE_PHASES` | experimental |
| `diagnostic.replay.pm4_queues` | `replay_pm4_queues` | `HIPFIRE_REPLAY_PM4_QUEUES` | experimental |
| `diagnostic.replay.pm4_register_policy` | `replay_pm4_stateful` | `HIPFIRE_REPLAY_PM4_STATEFUL` | experimental |
| `diagnostic.replay.pm4_wait_policy` | `replay_pm4_wait_policy` | `HIPFIRE_REPLAY_PM4_WAIT_POLICY` | experimental |
| `diagnostic.replay.pool_debug` | `replay_pool_debug` | `HIPFIRE_REDLINE_POOL_DEBUG` | experimental |
| `diagnostic.replay.route_proof_log` | `replay_route_proof_log` | `HIPFIRE_REPLAY_ROUTE_PROOF_LOG` | experimental |
| `experimental.budget_alert` | `experimental_budget_alert` | `HIPFIRE_EXPERIMENTAL_BUDGET_ALERT` | experimental |
| `experimental.graph.ar` | `graph_ar` | `HIPFIRE_AR_GRAPH` | experimental |
| `experimental.graph.forward` | `graph_forward` | `HIPFIRE_GRAPH` | experimental |
| `experimental.graph.moe` | `graph_moe` | `HIPFIRE_GRAPH_MOE` | experimental |
| `fusions.force_unfused` | `force_unfused` | `HIPFIRE_FORCE_UNFUSED` | experimental |
| `fusions.policy` | `fusion_policy` |  | stable |
| `fusions.qkv_bias` | `fuse_qkv_bias` | `HIPFIRE_FUSE_QKV_BIAS` | stable |
| `generation.loop_guard_threshold` | `ngram_loop_threshold` | `HIPFIRE_NGRAM_LOOP_THRESHOLD` | stable |
| `generation.loop_guard_window` | `ngram_window` | `HIPFIRE_NGRAM_WINDOW` | stable |
| `generation.max_tokens` | `max_tokens` |  | stable |
| `generation.min_p` | `min_p` |  | stable |
| `generation.presence_penalty` | `presence_penalty` |  | stable |
| `generation.repeat_penalty` | `repeat_penalty` |  | stable |
| `generation.temperature` | `temperature` |  | stable |
| `generation.top_k` | `top_k` |  | stable |
| `generation.top_p` | `top_p` |  | stable |
| `hardware.allow_mixed_arch` | `allow_mixed_arch` | `HIPFIRE_ALLOW_MIXED_ARCH` | stable |
| `hardware.deepseek4_compute_placement` | `deepseek4_compute_placement` |  | stable |
| `hardware.devices` | `devices` | `HIPFIRE_DEVICES` | stable |
| `hardware.tp_use_rccl` | `tp_use_rccl` | `HIPFIRE_TP_USE_RCCL` | stable |
| `hardware.uniform_vram_tolerance_gb` | `uniform_vram_tolerance_gb` | `HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB` | stable |
| `image.decode` | `image_decode` | `HIPFIRE_IMAGE_DECODE` | stable |
| `kernel.attn_qresident` | `attn_qresident` | `HIPFIRE_ATTN_QRESIDENT` | stable |
| `kernel.attn_qresident_v2` | `attn_qresident_v2` | `HIPFIRE_ATTN_QRESIDENT_V2` | stable |
| `kernel.calib_force_bf16` | `calib_force_bf16` | `HIPFIRE_CALIB_BF16` | experimental |
| `kernel.deepseek4_q8_4w` | `deepseek4_q8_4w` | `HIPFIRE_DEEPSEEK4_Q8_4W` | stable |
| `kernel.deepseek4_q8_wmma` | `deepseek4_q8_wmma` | `HIPFIRE_DEEPSEEK4_Q8_WMMA` | stable |
| `kernel.deterministic` | `deterministic` | `HIPFIRE_DETERMINISTIC` | experimental |
| `kernel.dot2_gemv` | `dot2_gemv` | `HIPFIRE_DOT2_GEMV` | experimental |
| `kernel.flash_partials_batch` | `flash_partials_batch` | `HIPFIRE_FLASH_PARTIALS_BATCH` | experimental |
| `kernel.fp16` | `fp16` | `HIPFIRE_FP16` | stable |
| `kernel.fp8_wmma` | `fp8_wmma` | `HIPFIRE_FP8_WMMA` | experimental |
| `kernel.g12_a4c2` | `g12_a4c2` | `HIPFIRE_G12_A4C2` | stable |
| `kernel.g12_dec_norm` | `g12_dec_norm` | `HIPFIRE_G12_DEC_NORM` | stable |
| `kernel.g12_norm` | `g12_norm` | `HIPFIRE_G12_NORM` | stable |
| `kernel.gate_up_nosync` | `gate_up_nosync` | `HIPFIRE_GATE_UP_NOSYNC` | experimental |
| `kernel.gcn5_wave64_hybrid` | `gcn5_wave64_hybrid` | `HIPFIRE_GCN5_WAVE64_HYBRID` | experimental |
| `kernel.gemma4_batched_embedding_prefill` | `gemma4_batched_embedding_prefill` | `HIPFIRE_GEMMA4_BATCHED_EMBEDDING_PREFILL` | experimental |
| `kernel.gemma4_ple_activation_fused_prefill` | `gemma4_ple_activation_fused_prefill` | `HIPFIRE_GEMMA4_PLE_ACTIVATION_FUSED_PREFILL` | experimental |
| `kernel.gemma4_ple_batched_prefill` | `gemma4_ple_batched_prefill` | `HIPFIRE_GEMMA4_PLE_BATCHED_PREFILL` | experimental |
| `kernel.gemma4_ple_branch_batched_prefill` | `gemma4_ple_branch_batched_prefill` | `HIPFIRE_GEMMA4_PLE_BRANCH_BATCHED_PREFILL` | experimental |
| `kernel.gemma4_q8_fused_prefill` | `gemma4_q8_fused_prefill` | `HIPFIRE_GEMMA4_Q8_FUSED_PREFILL` | experimental |
| `kernel.gemv_dp4a` | `gemv_dp4a` | `HIPFIRE_GEMV_DP4A` | stable |
| `kernel.gemv_prefetch` | `gemv_prefetch` | `HIPFIRE_GEMV_PREFETCH` | stable |
| `kernel.gfx1100_dec_norm` | `gfx1100_dec_norm` | `HIPFIRE_GFX1100_DEC_NORM` | stable |
| `kernel.gfx11_a4_candidates` | `gfx11_a4_candidates` | `HIPFIRE_GFX11_A4_CANDIDATES` | experimental |
| `kernel.gfx11_fa2_prefill` | `gfx11_fa2_prefill` | `HIPFIRE_GFX11_FA2_PREFILL` | stable |
| `kernel.gfx11_iu4_gridspec` | `gfx11_iu4_gridspec` | `HIPFIRE_GFX11_IU4_GRIDSPEC` | stable |
| `kernel.gfx11_iu4_shape` | `gfx11_iu4_shape` | `HIPFIRE_GFX11_IU4_SHAPE` | experimental |
| `kernel.gfx11_iu4_symfold` | `gfx11_iu4_symfold` | `HIPFIRE_IU4_SYMFOLD` | experimental |
| `kernel.gfx11_lean_pbs` | `gfx11_lean_pbs` | `HIPFIRE_GFX11_LEAN_PBS` | experimental |
| `kernel.gfx11_producer_quant_fused` | `gfx11_producer_quant_fused` | `HIPFIRE_GFX11_PRODUCER_QUANT_FUSED` | stable |
| `kernel.gfx11_q8_fa2_wide` | `gfx11_q8_fa2_wide` | `HIPFIRE_GFX11_Q8_FA2_WIDE` | experimental |
| `kernel.gfx12_fa2_prefill` | `gfx12_fa2_prefill` | `HIPFIRE_GFX12_FA2_PREFILL` | stable |
| `kernel.gfx12_fa_packet` | `gfx12_fa_packet` | `HIPFIRE_GFX12_FA_PACKET` | stable |
| `kernel.gfx12_fa_prep_fp8q` | `gfx12_fa_prep_fp8q` | `HIPFIRE_GFX12_FA_PREP_FP8Q` | stable |
| `kernel.gfx12_fa_prep_fused` | `gfx12_fa_prep_fused` | `HIPFIRE_GFX12_FA_PREP_FUSED` | stable |
| `kernel.gfx12_fp8_stream` | `gfx12_fp8_stream` | `HIPFIRE_GFX12_FP8_STREAM` | stable |
| `kernel.gfx12_gdn_chunk_scan` | `gfx12_gdn_chunk_scan` | `HIPFIRE_GFX12_GDN_CHUNK_SCAN` | stable |
| `kernel.gfx12_gdn_pre_fused` | `gfx12_gdn_pre_fused` | `HIPFIRE_GFX12_GDN_PRE_FUSED` | stable |
| `kernel.gfx12_mq4v2_fp8_gateup` | `gfx12_mq4v2_fp8_gateup` | `HIPFIRE_GFX12_MQ4V2_FP8_GATEUP` | stable |
| `kernel.gfx12_mq4v2_fp8_qkv` | `gfx12_mq4v2_fp8_qkv` | `HIPFIRE_GFX12_MQ4V2_FP8_QKV` | stable |
| `kernel.gfx12_mq4v2_fp8_qkvza` | `gfx12_mq4v2_fp8_qkvza` | `HIPFIRE_GFX12_MQ4V2_FP8_QKVZA` | stable |
| `kernel.gfx12_mq4v2_fp8_resid` | `gfx12_mq4v2_fp8_resid` | `HIPFIRE_GFX12_MQ4V2_FP8_RESID` | stable |
| `kernel.gfx12_mq4v2_fp8_v2` | `gfx12_mq4v2_fp8_v2` | `HIPFIRE_GFX12_MQ4V2_FP8_V2` | stable |
| `kernel.gfx12_producer_quant_fused` | `gfx12_producer_quant_fused` | `HIPFIRE_GFX12_PRODUCER_QUANT_FUSED` | stable |
| `kernel.gfx12_silu_quant_fused` | `gfx12_silu_quant_fused` | `HIPFIRE_GFX12_SILU_QUANT_FUSED` | stable |
| `kernel.gfx942_gemv_v2` | `gfx942_gemv_v2` | `HIPFIRE_GFX942_GEMV_V2` | experimental |
| `kernel.gfx942_gemv_v3` | `gfx942_gemv_v3` | `HIPFIRE_GFX942_GEMV_V3` | experimental |
| `kernel.gfx942_lds_gemv` | `gfx942_lds_gemv` | `HIPFIRE_GFX942_LDS_GEMV` | experimental |
| `kernel.gfx942_rmsnorm_split` | `gfx942_rmsnorm_split` | `HIPFIRE_GFX942_RMSNORM_SPLIT` | experimental |
| `kernel.hfq3_dp4a` | `hfq3_dp4a` | `HIPFIRE_HFQ3_DP4A` | experimental |
| `kernel.hfq3_mmq` | `hfq3_mmq` | `HIPFIRE_HFQ3_MMQ` | experimental |
| `kernel.hfq4_mmq_gfx906_y64` | `hfq4_mmq_gfx906_y64` | `HIPFIRE_HFQ4_MMQ_GFX906_Y64` | experimental |
| `kernel.hfq4_mmq_rdna2` | `hfq4_mmq_rdna2` | `HIPFIRE_HFQ4_MMQ_RDNA2` | experimental |
| `kernel.hfq4g128_mmq` | `hfq4g128_mmq` | `HIPFIRE_HFQ4G128_MMQ` | stable |
| `kernel.iu4_prefill` | `iu4_prefill` | `HIPFIRE_IU4_PREFILL` | stable |
| `kernel.lloyd_force_baseline` | `lloyd_force_baseline` | `HIPFIRE_LLOYD_FORCE_BASELINE` | experimental |
| `kernel.lm_head_f16` | `lm_head_f16` | `HIPFIRE_LM_HEAD_F16` | stable |
| `kernel.lm_head_overwrite` | `lm_head_overwrite` | `HIPFIRE_LM_HEAD_OVERWRITE` | experimental |
| `kernel.lm_head_wmma` | `lm_head_wmma` | `HIPFIRE_LM_HEAD_WMMA` | stable |
| `kernel.mmq` | `mmq` | `HIPFIRE_MMQ` | stable |
| `kernel.moe_down_combine_vec4` | `moe_down_combine_vec4` | `HIPFIRE_MOE_DOWN_COMBINE_VEC4` | experimental |
| `kernel.moe_grouped_4w` | `moe_grouped_4w` | `HIPFIRE_MOE_GROUPED_4W` | experimental |
| `kernel.moe_grouped_gemm` | `moe_grouped_gemm` | `HIPFIRE_MOE_GROUPED_GEMM` | stable |
| `kernel.moe_grouped_i8` | `moe_grouped_i8` | `HIPFIRE_MOE_GROUPED_I8` | experimental |
| `kernel.moe_grouped_i8_k4` | `moe_grouped_i8_k4` | `HIPFIRE_MOE_GROUPED_I8_K4` | experimental |
| `kernel.moe_grouped_i8_k4_gfx12` | `moe_grouped_i8_k4_gfx12` | `HIPFIRE_MOE_GROUPED_I8_K4_GFX12` | experimental |
| `kernel.moe_grouped_i8_k8` | `moe_grouped_i8_k8` | `HIPFIRE_MOE_GROUPED_I8_K8` | experimental |
| `kernel.moe_grouped_m2` | `moe_grouped_m2` | `HIPFIRE_MOE_GROUPED_M2` | experimental |
| `kernel.moe_hfq6_i8` | `moe_hfq6_i8` | `HIPFIRE_MOE_HFQ6_I8` | experimental |
| `kernel.moe_hfq6_v2` | `moe_hfq6_v2` | `HIPFIRE_MOE_HFQ6_V2` | experimental |
| `kernel.moe_paro_i8` | `moe_paro_i8` | `HIPFIRE_MOE_PARO_I8` | experimental |
| `kernel.moe_paro_i8_k8` | `moe_paro_i8_k8` | `HIPFIRE_MOE_PARO_I8_K8` | experimental |
| `kernel.npu_spillover` | `npu_spillover` | `HIPFIRE_NPU_SPILLOVER` | experimental |
| `kernel.prefill_batched` | `prefill_batched` | `HIPFIRE_PREFILL_BATCHED` | stable |
| `kernel.q8_batched_legacy` | `q8_batched_legacy` | `HIPFIRE_Q8_BATCHED_LEGACY` | experimental |
| `kernel.qkvza_split_tail` | `qkvza_split_tail` | `HIPFIRE_QKVZA_SPLIT_TAIL` | experimental |
| `kernel.rdna3_hfq4_lm_head_k2048` | `rdna3_hfq4_lm_head_k2048` | `HIPFIRE_RDNA3_HFQ4_LM_HEAD_K2048` | experimental |
| `kernel.rdna3_hfq4_moe_gate_up_k2048` | `rdna3_hfq4_moe_gate_up_k2048` | `HIPFIRE_RDNA3_HFQ4_MOE_GATE_UP_K2048` | experimental |
| `kernel.rdna3_hfq4_qkv_wave64` | `rdna3_hfq4_qkv_wave64` | `HIPFIRE_RDNA3_HFQ4_QKV_WAVE64` | experimental |
| `kernel.rdna3_hfq4_qkvza_2wave` | `rdna3_hfq4_qkvza_2wave` | `HIPFIRE_RDNA3_HFQ4_QKVZA_2WAVE` | experimental |
| `kernel.rdna3_hfq4_qkvza_hoist_x32` | `rdna3_hfq4_qkvza_hoist_x32` | `HIPFIRE_RDNA3_HFQ4_QKVZA_HOIST_X32` | experimental |
| `kernel.rdna3_hfq4_qkvza_k2048` | `rdna3_hfq4_qkvza_k2048` | `HIPFIRE_RDNA3_HFQ4_QKVZA_K2048` | experimental |
| `kernel.rdna3_hfq4_qkvza_ldsx8` | `rdna3_hfq4_qkvza_ldsx8` | `HIPFIRE_RDNA3_HFQ4_QKVZA_LDSX8` | experimental |
| `kernel.rdna3_hfq4_qkvza_reduce_chain` | `rdna3_hfq4_qkvza_reduce_chain` | `HIPFIRE_RDNA3_HFQ4_QKVZA_REDUCE_CHAIN` | experimental |
| `kernel.rdna3_hfq4_qkvza_wavepack4` | `rdna3_hfq4_qkvza_wavepack4` | `HIPFIRE_RDNA3_HFQ4_QKVZA_WAVEPACK4` | experimental |
| `kernel.rdna3_hfq4_residual_k2048` | `rdna3_hfq4_residual_k2048` | `HIPFIRE_RDNA3_HFQ4_RESIDUAL_K2048` | experimental |
| `kernel.rdna3_hfq4_residual_stage_x32` | `rdna3_hfq4_residual_stage_x32` | `HIPFIRE_RDNA3_HFQ4_RESIDUAL_STAGE_X32` | experimental |
| `kernel.rdna3_hfq4_sigmoid_buffer` | `rdna3_hfq4_sigmoid_buffer` | `HIPFIRE_RDNA3_HFQ4_SIGMOID_BUFFER` | experimental |
| `kernel.rdna3_hfq4_sigmoid_rows4` | `rdna3_hfq4_sigmoid_rows4` | `HIPFIRE_RDNA3_HFQ4_SIGMOID_ROWS4` | experimental |
| `kernel.rdna3_hfq4_sigmoid_tight_grid` | `rdna3_hfq4_sigmoid_tight_grid` | `HIPFIRE_RDNA3_HFQ4_SIGMOID_TIGHT_GRID` | experimental |
| `kernel.rdna3_rmsnorm_sign_const` | `rdna3_rmsnorm_sign_const` | `HIPFIRE_RDNA3_RMSNORM_SIGN_CONST` | experimental |
| `kernel.rdna3_rmsnorm_sign_lds` | `rdna3_rmsnorm_sign_lds` | `HIPFIRE_RDNA3_RMSNORM_SIGN_LDS` | experimental |
| `kernel.rdna3_rmsnorm_split` | `rdna3_rmsnorm_split` | `HIPFIRE_RDNA3_RMSNORM_SPLIT` | experimental |
| `kernel.rdna3_rmsnorm_vecsum` | `rdna3_rmsnorm_vecsum` | `HIPFIRE_RDNA3_RMSNORM_VECSUM` | experimental |
| `kernel.rdna3_rmsnorm_wavegrid` | `rdna3_rmsnorm_wavegrid` | `HIPFIRE_RDNA3_RMSNORM_WAVEGRID` | experimental |
| `kernel.rmsnorm_mq_tight_lds` | `rmsnorm_mq_tight_lds` | `HIPFIRE_RMSNORM_MQ_TIGHT_LDS` | experimental |
| `kernel.rocblas_all_archs` | `rocblas_all_archs` | `HIPFIRE_ROCBLAS_ALL_ARCHS` | experimental |
| `kernel.rocblas_off` | `rocblas_off` | `HIPFIRE_ROCBLAS_OFF` | stable |
| `kernel.rope_interleaved_legacy` | `rope_interleaved_legacy` | `HIPFIRE_ROPE_INTERLEAVED_LEGACY` | experimental |
| `kernel.verify_attn` | `verify_attn` | `HIPFIRE_VERIFY_ATTN` | stable |
| `kernel.wo_mmq` | `wo_mmq` | `HIPFIRE_WO_MMQ` | experimental |
| `memory.cask.auto_attach` | `cask_auto_attach` |  | deprecated |
| `memory.cask.beta` | `cask_beta` |  | deprecated |
| `memory.cask.budget` | `cask_budget` |  | deprecated |
| `memory.cask.core_fraction` | `cask_core_frac` |  | deprecated |
| `memory.cask.enabled` | `cask` |  | deprecated |
| `memory.cask.fold` | `cask_fold_m` |  | deprecated |
| `memory.cask.handoff_tokens` | `cask_handoff_tokens` |  | deprecated |
| `memory.cask.sidecar` | `cask_sidecar` | `HIPFIRE_CASK_SIDECAR` | deprecated |
| `memory.gpu_layer_budget` | `gpu_layer_budget` | `HIPFIRE_GPU_LAYER_BUDGET` | stable |
| `memory.kv_adaptive` | `kv_adaptive` | `HIPFIRE_KV_ADAPTIVE` | stable |
| `memory.kv_backend` | `kv_backend` |  | stable |
| `memory.kv_cache` | `kv_cache` | `HIPFIRE_KV_MODE` | stable |
| `memory.kv_k` | `kv_k` |  | stable |
| `memory.kv_v` | `kv_v` |  | stable |
| `memory.max_seq` | `max_seq` |  | stable |
| `memory.mmq.screen` | `mmq_screen` | `HIPFIRE_MMQ_SCREEN` | stable |
| `memory.mmq.screen_threshold` | `mmq_screen_threshold` | `HIPFIRE_MMQ_SCREEN_THRESHOLD` | stable |
| `memory.offload_exec` | `offload_exec` | `HIPFIRE_OFFLOAD_EXEC` | stable |
| `memory.oom_guard` | `oom_guard` | `HIPFIRE_OOM_GUARD` | stable |
| `memory.prompt_cache_capacity` | `prompt_cache_capacity` | `HIPFIRE_PROMPT_CACHE_CAP` | stable |
| `memory.prompt_cache_unbounded` | `prompt_cache_unbounded` | `HIPFIRE_PROMPT_CACHE_UNBOUNDED` | experimental |
| `memory.unsafe_wsl_vmm_kv` | `unsafe_wsl_vmm_kv` | `HIPFIRE_UNSAFE_WSL_VMM_KV` | experimental |
| `model.deepseek4_experts_per_token` | `deepseek4_experts_per_token` |  | stable |
| `prefill.chunk_rows` | `prefill_chunk_rows` | `HIPFIRE_PREFILL_CHUNK_ROWS` | stable |
| `prompt.chat_template` | `chat_template` | `HIPFIRE_CHAT_TEMPLATE_FILE` | stable |
| `prompt.default_chatml` | `default_chatml` | `HIPFIRE_DEFAULT_CHATML` | stable |
| `prompt.jinja_chat` | `jinja_chat` | `HIPFIRE_JINJA_CHAT` | stable |
| `prompt.normalize` | `prompt_normalize` | `HIPFIRE_NORMALIZE_PROMPT` | stable |
| `prompt.system` | `system_prompt` |  | stable |
| `reasoning.budget` | `thinking_budget` |  | stable |
| `reasoning.effort` | `reasoning_effort` |  | stable |
| `reasoning.max_tokens` | `max_think_tokens` |  | stable |
| `reasoning.max_total_tokens` | `max_total_think_tokens` | `HIPFIRE_MAX_TOTAL_THINK_TOKENS` | stable |
| `reasoning.mode` | `thinking` |  | stable |
| `replay.backend` | `replay_backend` | `HIPFIRE_REPLAY_BACKEND` | stable |
| `replay.gfx1201_pm4_pacing` | `gfx1201_pm4_pacing` | `HIPFIRE_GFX1201_PM4_PACING` | stable |
| `replay.pm4_gfx11_vmem_acquire` | `replay_pm4_gfx11_vmem_acquire` | `HIPFIRE_REPLAY_PM4_GFX11_VMEM_ACQUIRE` | experimental |
| `replay.transport` | `replay_transport` | `HIPFIRE_REPLAY_TRANSPORT` | experimental |
| `replay.unsafe_wsl_redline` | `unsafe_wsl_redline` | `HIPFIRE_UNSAFE_WSL_REDLINE` | experimental |
| `serve.allow_request_paths` | `allow_request_paths` | `HIPFIRE_SERVE_ALLOW_REQUEST_PATHS` | stable |
| `serve.allow_request_pull` | `allow_request_pull` | `HIPFIRE_SERVE_ALLOW_REQUEST_PULL` | stable |
| `serve.continuous_batch_size` | `continuous_batch_size` | `HIPFIRE_CONTINUOUS_BATCH_SIZE` | stable |
| `serve.default_model` | `default_model` | `HIPFIRE_MODEL` | stable |
| `serve.host` | `host` | `HIPFIRE_HOST` | stable |
| `serve.idle_timeout_seconds` | `idle_timeout` | `HIPFIRE_IDLE_TIMEOUT` | stable |
| `serve.local` | `local` | `HIPFIRE_LOCAL` | stable |
| `serve.max_batch_tokens` | `max_batch_tokens` | `HIPFIRE_SERVE_MAX_BATCH_TOKENS` | experimental |
| `serve.max_queue` | `serve_max_queue` | `HIPFIRE_SERVE_MAX_QUEUE` | stable |
| `serve.max_queue_bytes` | `max_queue_bytes` | `HIPFIRE_SERVE_MAX_QUEUE_BYTES` | experimental |
| `serve.max_request_bytes` | `max_request_bytes` | `HIPFIRE_MAX_REQUEST_BYTES` | stable |
| `serve.multi_slot` | `multi_slot` | `HIPFIRE_SERVE_MULTI_SLOT` | stable |
| `serve.multi_slot_ctx` | `multi_slot_ctx` | `HIPFIRE_SERVE_MULTI_SLOT_CTX` | stable |
| `serve.multi_slot_prefill_chunk` | `multi_slot_prefill_chunk` | `HIPFIRE_SERVE_MULTI_SLOT_PREFILL_CHUNK` | stable |
| `serve.multi_slot_slots` | `multi_slot_slots` | `HIPFIRE_SERVE_MULTI_SLOT_SLOTS` | stable |
| `serve.port` | `port` | `HIPFIRE_PORT` | stable |
| `serve.prefill_min_tokens` | `prefill_min_tokens` | `HIPFIRE_SERVE_PREFILL_MIN_TOKENS` | experimental |
| `serve.prefix_cache` | `prefix_cache` | `HIPFIRE_SERVE_PREFIX_CACHE` | experimental |
| `serve.prefix_cache_max_bytes` | `prefix_cache_max_bytes` | `HIPFIRE_SERVE_PREFIX_CACHE_MAX_BYTES` | experimental |
| `serve.queue_timeout_ms` | `serve_queue_timeout_ms` | `HIPFIRE_SERVE_QUEUE_TIMEOUT_MS` | stable |
| `serve.retry_backoff_ms` | `retry_backoff_ms` | `HIPFIRE_SERVE_RETRY_BACKOFF_MS` | experimental |
| `serve.retry_enabled` | `retry_enabled` | `HIPFIRE_SERVE_RETRY_ENABLED` | experimental |
| `serve.stream_stall_timeout_ms` | `stream_stall_timeout_ms` | `HIPFIRE_SERVE_STREAM_STALL_TIMEOUT_MS` | experimental |
| `serve.structured_jump_forward` | `structured_jump_forward` | `HIPFIRE_SERVE_STRUCTURED_JUMP_FORWARD` | experimental |
| `serve.ui` | `serve_ui` | `HIPFIRE_SERVE_UI` | stable |
| `speculation.ddtree_budget` | `ddtree_budget` | `HIPFIRE_DDTREE_BUDGET` | experimental |
| `speculation.ddtree_topk` | `ddtree_topk` | `HIPFIRE_DDTREE_TOPK` | experimental |
| `speculation.ddtree_tree_la` | `ddtree_tree_la` | `HIPFIRE_DDTREE_TREE_LA` | experimental |
| `speculation.dflash` | `dflash_mode` | `HIPFIRE_DFLASH_MODE` | stable |
| `speculation.dflash_adaptive_b` | `dflash_adaptive_b` |  | stable |
| `speculation.dflash_fast_sample` | `dflash_fast_sample` | `HIPFIRE_DFLASH_FAST_SAMPLE` | experimental |
| `speculation.dflash_ngram_block` | `dflash_ngram_block` | `HIPFIRE_DFLASH_NGRAM_BLOCK` | stable |
| `speculation.dflash_q8_lmhead_wmma` | `dflash_q8_lmhead_wmma` | `HIPFIRE_DFLASH_Q8_LMHEAD_WMMA` | experimental |
| `speculation.dflash_tree` | `dflash_tree` | `HIPFIRE_DFLASH_TREE` | experimental |
| `speculation.draft_f16` | `draft_f16` | `HIPFIRE_DRAFT_F16` | stable |
| `speculation.dspark_confidence` | `dspark_conf_threshold` |  | experimental |
| `speculation.mode` | `speculation` | `HIPFIRE_SPECULATION` | stable |
| `speculation.mtp` | `mtp_mode` | `HIPFIRE_MTP_MODE` | stable |
| `speculation.mtp_k` | `mtp_k` | `HIPFIRE_MTP_K` | stable |
| `speculation.mtp_ngram` | `mtp_ngram` | `HIPFIRE_MTP_NGRAM` | stable |
| `speculation.ngram` | `ngram_mode` | `HIPFIRE_NGRAM_DRAFT` | stable |
| `speculation.ngram_k` | `ngram_k` | `HIPFIRE_NGRAM_DRAFT_K` | stable |
| `speculation.ngram_min_count` | `ngram_min_count` | `HIPFIRE_NGRAM_MIN_COUNT` | stable |
| `speculation.prefill.alpha` | `prefill_alpha` |  | deprecated |
| `speculation.prefill.block` | `prefill_block` |  | deprecated |
| `speculation.prefill.drafter` | `prefill_drafter` |  | deprecated |
| `speculation.prefill.drafter_device` | `prefill_drafter_device` |  | deprecated |
| `speculation.prefill.drafter_kv` | `prefill_drafter_kv` | `HIPFIRE_PFLASH_DRAFTER_KV` | deprecated |
| `speculation.prefill.keep_ratio` | `prefill_keep_ratio` |  | deprecated |
| `speculation.prefill.min_keep` | `prefill_min_keep` |  | deprecated |
| `speculation.prefill.mode` | `prefill_compression` |  | deprecated |
| `speculation.prefill.profile` | `prefill_profile` |  | deprecated |
| `speculation.prefill.recent` | `prefill_recent` |  | deprecated |
| `speculation.prefill.sink` | `prefill_sink` |  | deprecated |
| `speculation.prefill.sparse_threshold` | `prefill_sparse_threshold` |  | deprecated |
| `speculation.prefill.threshold` | `prefill_threshold` |  | deprecated |
| `vision.mode` | `vision_mode` | `HIPFIRE_VISION_MODE` | stable |
| `developer.<name>` | | `HIPFIRE_<NAME>` | developer |

<!-- lifecycle-table:end -->

## Related

| Topic | Owner |
|---|---|
| Copyable TOML profiles | [`configs/`](configs/README.md) |
| Env inventory | [`env-vars.md`](env-vars.md) |
| Models / registry sampling | [`MODELS.md`](MODELS.md) |
| Serve API | [`SERVE.md`](SERVE.md) |
| CLI | [`CLI.md`](CLI.md) |
| Multi-GPU | [`multi-gpu.md`](multi-gpu.md) |
