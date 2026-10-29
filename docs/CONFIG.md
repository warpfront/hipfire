# Configuration

**Owner:** daemon / user config keys (`docs/INDEX.md`).
**Machine sources:**

- Native schema + resolution: `crates/hipfire-config/src/lib.rs`
- Versioned CLI/daemon handoff: `hipfire_config::ProcessConfig`
- Compact runtime snapshots: `hipfire_runtime::config::RuntimeConfig` and
  `rdna_compute::feature_flags::FeatureFlags`

**Last checked:** 2026-10-23.

Persistent stores under `~/.hipfire/`:

1. **Global** — sparse typed `config.toml`; missing keys inherit schema defaults.
2. **Per-model overlay (primary)** — `models.toml`: aliases, local paths,
   registry identities, and sparse per-model overrides.
3. **Migration inputs** — `config.json`, `models.json`, and
   `per_model_config.json`. Migration writes TOML and preserves the JSON files
   for rollback.
4. **Tuning state** — `tuning.toml`: which preset owns which `config.toml`
   keys, and the single undo level (see [Tuning presets](#tuning-presets)).

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

Selection **merges** the chosen built-in or custom profile into the global
`config.toml`: every key the profile sets is written (overwriting that key's old
value) and every other key (devices, ports, default model, kernel flags, ...) is
kept. The `default` profile pins only process-level policy with a single fixed
default (loop guard, prompt-cache size, `serve.*`); it never sets keys the
registry card or the GPU architecture resolves on its own (`memory.kv_cache`,
`memory.max_seq`, `speculation.*`, `reasoning.*`, `attention.flash`,
`prompt.*`, `generation.max_tokens`), so those stay at auto. The profile wizard
lists the changes and asks `y/n` before writing. A profile counts as applied
when all its keys are present with the same values. Custom snapshots live
under `~/.hipfire/profiles/<name>.toml`. Profile names are not schema fields and
are never written into `config.toml`. The Settings TUI and CLI share the
hipfire-config helpers. Profiles are global-only.

Edit global settings interactively with `hipfire config`. Per-model overlays use
the typed commands: `hipfire config <tag> list|get|set|reset ...`. Global
non-interactive edits use `hipfire config list|get|set|reset ...`.

In the TUI Settings tab every non-deprecated key is editable inline: its editor
kind (cycle an enum, toggle a boolean, type a number or text) is derived from
the schema's value rule, so the TUI offers exactly what `hipfire config set`
accepts. `memory.kv_cache` offers `auto`, `q8`, `fwht4`, `fwht3`, `fwht2`, `fp8`,
`bf16`, `f16` and `f32` (the deprecated `asymN`/`turboN` aliases stay settable
only from the CLI). `developer.qwen4_moe_sym_iu4` is written as a TOML boolean.
The tab reads the same layers as `hipfire config explain` (built-in defaults,
the model's registry card, `config.toml`, `models.toml`, the environment) and
shows, per row, the value predicted for the active model on your GPU and the
layer that set it. A value set by an environment variable is locked in the
TUI: unset the variable to change it.

### Reading settings: `list`, `explain`, `overrides`

`hipfire config list` groups every key by what it does for you, in the same six
groups the Settings tab uses (**Quality**, **Speed**, **Memory & context**,
**Speculation**, **Hardware & devices**, **Advanced**). Each row shows what the
key will do *now* on this GPU and who set it:

```text
Global configuration: /home/u/.hipfire/config.toml
GPU: gfx1201 (install.json)

Quality
  kernel.iu4_prefill                          true             default            serve restart
  reasoning.effort                            low              you                next message

Memory & context
  memory.kv_cache                             fp8 (auto)       default · gfx1201  next model load
```

- **Using now** is the *predicted* effective value: an `auto` (or an
  arch-gated default such as `fp8` KV on gfx1201) is resolved to the concrete
  value for the GPU in `~/.hipfire/install.json` and, in the model form, the
  model's registry card. ` (auto)` marks a value an arch rule or `auto` chose.
  serve decides at load; see its log. Without `gpu_arch` in `install.json` the
  header says `GPU: unknown (no gpu_arch in install.json)` and arch rules fall
  back to their arch-neutral answer.
- **From** names the layer that set the value: `default`, `default · gfx1201`
  (an arch rule), `registry`, `profile dev`, `you` (config.toml), `this model`
  (models.toml) or `env HIPFIRE_*`. **Applies** says when a change takes effect
  (`next message`, `new chat`, `next model load`, `serve restart`).
- A row ends with `(not used by this model)` / `(not used on this GPU)` when the
  key has no effect there.

Flags (all combine; they also apply to `--json`):

| Flag | Effect |
|---|---|
| `--group G` | one group: `quality`, `speed`, `memory` (`memory-context`), `speculation`, `hardware` (`hardware-devices`), `advanced`; case-insensitive; an unknown name errors with the valid list |
| `--changed` | only rows something other than the defaults or the model card set (`profile`, `preset`, `you`, `this model`, `env`) |
| `--search TEXT` | only rows matching `TEXT` across key, legacy key, help, env var, category, scope (the Settings tab search) |
| `--model M` | list for one model; same as `hipfire config M list` |
| `--json` | machine-readable output |

`hipfire config MODEL list [--group G] [--changed] [--search TEXT] [--json]`
shows the same rows for one model: registry card and `models.toml` values apply,
and "from" says `this model` for a per-model value.

`--json` keeps the earlier per-key fields (`legacy_key`, `value`, `default`,
`source`, `overridden`) and adds `group` (kebab-case group name), `effective`
(`{"value", "reason"}`), `origin` (`default`, `registry`, `profile`, `preset`,
`you`, `model`, `env`, `one-shot`), `from` (the label above) and `applies`; the
top level gains `gpu_arch`.

`hipfire config explain KEY [--model M]` (also `hipfire config MODEL explain KEY`)
adds four lines after `source:`: `effective:` (shown when the predicted value
differs from the resolved one or carries a reason, for example
`fp8 (auto: fp8 on exact gfx1201, one GPU, Qwen)`), `from:`, `group:` and
`applies:`. Its JSON adds `effective`, `origin`, `from`, `group` and `applies`.

`hipfire config explain [--model M] [-j]` with **no key** prints every effective
setting with the layer that set it, grouped like `list`, then an `environment`
audit and the hint `explain one key: hipfire config explain <key>`. The header
names `config.toml` once (and, with `--model`, the per-model catalog file
(`models.toml`, or a legacy `models.json` not yet migrated), resolved exactly as
`hipfire config M list` does). Sources: `default`, `default · <arch>`,
`registry <tag>`, `profile <name>`, `config.toml`, `<catalog file> [<model>]`,
`env HIPFIRE_X`, `cli`. `developer.*` keys appear only when set. The
`environment` section lists each `HIPFIRE_*` variable set in the current
process with one of: `applies <config key>` (schema alias or `developer.*`
mapping), `bootstrap` (read before config exists, e.g. `HIPFIRE_HOME`,
`HIPFIRE_DEVICES`), `runtime` (documented in [`env-vars.md`](env-vars.md) but not
a config key), `DEPRECATED since 0.4.0, removed in 0.5.0: <what to do instead>`, or
`UNKNOWN: not a hipfire variable (typo? did you mean HIPFIRE_Y)` (nearest known
name within edit distance 3). Values of names containing `KEY`, `TOKEN`,
`SECRET` or `PASS` print as `(set)`. `-j` prints an object with `settings`
(array of `{key, value, effective, source:{kind, label, env?, path?}}`) and
`env` (array of `{name, value, status, label}`).

`hipfire config overrides [--json]` lists every per-model value in
`~/.hipfire/models.toml` next to the all-models value it overrides:

```text
  qwen3:8b               memory.max_seq                   4096       (all models: 32768)
  qwen3:8b               reasoning.effort                 high       (all models: low)

2 overrides on 1 model.
```

With none, it prints `No per-model overrides (~/.hipfire/models.toml has none).`
JSON carries `count`, `models` and `overrides[]` (`model`, `key`, `value`,
`all_models`).

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

The TUI Settings tab edits per-model values with the same functions
(`hipfire_config::set_model_override` / `remove_model_override`): `m` picks
what Settings edits (all models, or one model), and in a model's scope every
change writes that model's entry in `models.toml`, byte-identical to
`hipfire config <model> set KEY VALUE`. `Del` removes that model's value (the
row follows "all models" again, like `hipfire config <model> reset KEY`) and
`R` removes all of its values. Global-only keys read `all models only
(locked)` there. `o` lists every per-model value next to the all-models value
(the screen form of `hipfire config overrides`); the Models tab shows the
count as `N own values`.

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
| RDNA4 MQ4V2 gate/up decode | `kernel.gfx12_mq4v2_gateup_k5120` (exact gfx1201, K=5120, 17408 rows per output; default off) |
| GEMV / quant | `kernel.gemv_dp4a`, `kernel.gemv_prefetch`, `kernel.gfx942_lds_gemv`, `kernel.hfq3_dp4a`, `kernel.hfq3_mmq`, `kernel.hfq4_mmq_rdna2`, `kernel.gcn5_wave64_hybrid`, `kernel.mmq`, `kernel.gfx942_gemv_v2` |
| MoE | `kernel.moe_grouped_i8`, `kernel.moe_paro_i8`, `kernel.moe_paro_i8_k8` |
| Certified arch routes | `kernel.rdna3_hfq4_qkvza_k2048`, `kernel.rdna3_hfq4_residual_stage_x32`, `kernel.rdna3_hfq4_sigmoid_buffer`, `kernel.rdna3_rmsnorm_vecsum`, `kernel.gfx942_rmsnorm_split` |

Stable/default-on and safety controls:

| Key | Default | Purpose |
|---|---:|---|
| `hardware.allow_mixed_arch` | `false` | Permit heterogeneous multi-GPU topology. |
| `topology.ep` | `1` | Qwen4 (Flash-Next) expert-parallel degree on gfx1100 or gfx1201; opt-in, experimental. Env `HIPFIRE_TOPOLOGY_EP`. The schema validates an integer 1–8 and a load accepts the same range (an N that does not divide 512 gives uneven per-rank expert counts). Routed experts shard `e % N` across N visible cards of one exact arch (gfx1100 or gfx1201, no mixing) with peer access between every pair. `HIPFIRE_QWEN4_EP=0` forces `1`. Admission rules and the support matrix live with the owner, [multi-gpu.md](multi-gpu.md); the config-exposed EP and NPU controls are listed in [Qwen4 expert-parallel and NPU controls](#qwen4-expert-parallel-and-npu-controls). |
| `topology.pp` | `1` | Pipeline-parallel degree: 1 (single device) or an integer 2–8; opt-in, experimental. Env `HIPFIRE_TOPOLOGY_PP`. An explicit load request with `pp` above 1 overrides it; otherwise this value applies. Details and the combination rules are in [Qwen4 expert-parallel and NPU controls](#qwen4-expert-parallel-and-npu-controls) and [multi-gpu.md](multi-gpu.md). |
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
| `fusions.npu_trunk` | `"auto"` | Flash-Next (Qwen4) trunk projection engine: `off\|auto\|on\|emulate` (case-insensitive). `auto` (default) resolves silently to `off` on every arch except gfx1151, with no NPU I/O or allocation. On gfx1151 it arms only when the load is eligible (Flash-Next, K=2560, stock MQ6G256V2 trunk projections, a single device without EP, the full 8192-row prefill scratch rung and enough free memory for it), the NPU firmware is readable, and the calibration matches the device/build/model identity and selects a nonempty family set (the built-in calibration is the normal path and applies on any Halo kernel/firmware/GPU PCI address, keyed by arch/build/model/epilogue/route; under `auto` each request picks its NPU family subset and pair plan from GPU/NPU rate scales learned live in this process, see [live rate scales](npu/npu-trunk.md#live-rate-scales-auto)); any refusal logs one `npu-trunk: off (reason)` line (prefixed `WARN` when the load qualifies but the NPU fails to arm) and the load continues GPU-only. The armed trunk pins 1,053,843,456 B of userptr arenas against `RLIMIT_MEMLOCK`; grant a bounded limit of 1,318,060,032 B (`LimitMEMLOCK=1318060032`, `ulimit -l 1287168`), see [host memlock limit](npu/npu-trunk.md#host-memlock-limit). `off` is the kill switch. `on` requires the NPU and `emulate` runs the bit-identical GPU emulation of the NPU numerics: both refuse an ineligible load or arch instead of falling back, and both tolerate a missing or mismatched calibration (logged as `calibration=none (forced)`; every family is used). |
| `fusions.npu_trunk_calibration` | `""` | Empty resolves to `<hipfire home>/npu-trunk/calibration.json`; alias `npu_trunk_calibration`, env `HIPFIRE_QWEN4_NPU_TRUNK_CAL` (a non-empty path replaces the default). A calibration is `schema_version` 4 JSON that rejects unknown fields. A file at the path wins; only a **missing** file uses the compiled-in measured Halo calibration (stock route, or the overlay route of the load's trunk overlay), so an unreadable or invalid file is an error and never falls back to the built-in. A calibration file is an optional strict override and passes a strict key check (arch, GPU PCI address, NPU firmware, build, kernel release, model-index hash, epilogue, trunk route); `auto` treats a mismatch as GPU-only and never recalibrates. No file is required: the built-in calibration binds on the hard key fields only (arch, build, model-index hash, epilogue, trunk route) and applies on any Halo kernel, NPU firmware and GPU PCI address at margin 0.0; when those differ from the record the load logs one `npu-trunk: built-in Halo calibration (measured on kernel <k>, NPU firmware <fw>, GPU <bdf>) applies by arch/model/route; this host ...` line. Live rate scales correct its speed for the host (see [NPU trunk](npu/npu-trunk.md#live-rate-scales-auto)). Armed logs include `cal_source=file\|builtin`. No calibration/index I/O when effectively off. See [NPU trunk](npu/npu-trunk.md). |
| `speculation.dflash_tree` | `false` | Enable DDTree tree-SWOR verification. |
| `memory.oom_guard` | `auto` | Memory preflight OOM guard — see [Memory](#memory). Gates the host available-memory headroom check (`MemAvailable` minus the CMA region) and, only when explicitly `off`, the load-time VRAM fit refusal of an explicit single-card Qwen VMM `max_seq`; the R9700 deployment-target VRAM-budget check always runs. Env: `HIPFIRE_OOM_GUARD`. |

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

`kernel.mq4v2_gateup_k5120` (exact gfx1100, K=5120, 17408 rows per output) is default on since 0.4.1: that gate/up decode runs on a fixed-group-count kernel whose output is byte-identical to the generic one. Set `hipfire config set kernel.mq4v2_gateup_k5120 false` to restore the generic route; the one-shot compatibility override is `HIPFIRE_MQ4V2_GATEUP_K5120=0` / `1`. This process-wide setting is snapshotted at GPU initialization; restart an existing daemon after changing it. Other architectures, shapes, and quantization formats keep their existing routes. Compiler-less installations need a kernel pack containing `fused_gate_up_mq4g256v2_k5120_gfx1100` (the 0.4.1 gfx1100 pack carries it).

Architecture-gated default-on kernel routes (exact arch only; other arches keep their prior routes; all are process-wide and snapshotted at GPU initialization, so restart an existing daemon after changing them):

| Key | Default on | Opt out | Notes |
|---|---|---|---|
| `kernel.gfx1100_fa2_split_verify` | exact gfx1100 | `false` / `HIPFIRE_GFX1100_FA2_SPLIT_VERIFY=0`; parent `kernel.verify_attn=false` / `HIPFIRE_VERIFY_ATTN=0` | Uses FA2 split-KV S8 only inside the existing dense Qwen H24/KV4/HD256, Q8/Q8 VMM-KV, sequential B4..32 long-context verifier envelope; admission, capture and Redline guards are unchanged. Its split partial/merge reduction order is not byte-identical to the incumbent route; promotion is backed by logit KLD, a hard30 task tie and canonical-XTS DFlash A/B evidence. |
| `kernel.gemma4_ple_branch_batched_prefill` | `auto`: exact gfx1100 and gfx1201 | `false` / `HIPFIRE_GEMMA4_PLE_BRANCH_BATCHED_PREFILL=0` | Exact-arithmetic batched PLE branch projections; gfx1101, gfx1102, gfx1151, gfx1200 stay off. Separate from `kernel.gemma4_ple_batched_prefill` (default off). |
| `kernel.gfx11_q8_fa2_wide` | `auto`: exact gfx1100 and gfx1151 (not gfx1100-only) | `false` / `HIPFIRE_GFX11_Q8_FA2_WIDE=0` | Whole-chunk Q8/Q8 FA2 prefill; requires `kernel.gfx11_fa2_prefill`. gfx1151 enabled deliberately in commit `ae9c5cf9d12a5a63ae60630637d22d00a4fc964e` via the twin CU-mode dispatch (same arithmetic, byte-identical). |
| `kernel.gfx12_fp8_stream` | exact gfx1201 | `false` / `HIPFIRE_GFX12_FP8_STREAM=0` | RMSNorm+rotate producer → MQ4v2 FP8 pre-pass fusion; byte-identical outputs; other arches off. |
| `kernel.pm_decode` | `auto`: exact gfx1201 | `false` / `HIPFIRE_PM_DECODE=0` | Loads the embedded, oracle-accepted PeaceMaker twins of the five W1 decode projection modules (`fused_qkvza_hfq4g256_mq4v2`, `fused_qkv_hfq4g256_mq4v2`, `fused_gate_up_hfq4g256_mq4v2`, `gemv_hfq4g256_multirow_default_mq4v2`, `gemv_hfq4g256_residual_mq4v2`) instead of compiling them with hipcc. Each image is pinned by bundle and ELF SHA-256 and refused on mismatch. Byte-identical outputs; the hipcc modules stay in the kernel pack for the opt-out; other arches never load the twins. |

See [`env-vars.md`](env-vars.md) for the full per-variable rows.

The experimental gfx1201 twin is independently controlled by `kernel.gfx12_mq4v2_gateup_k5120` (`HIPFIRE_GFX12_MQ4V2_GATEUP_K5120=1` / `0`). It does not change the gfx1100 default above. Restart the daemon after changing this process-scoped setting. Compiler-less installations need a pack containing `fused_gate_up_mq4g256v2_k5120_gfx1201`. The twin preserves DEV/RT weight loads, quantization, arithmetic and resident weight layout. Initial R9700 AR ABBA showed only +0.37%; this is not a statistically established E2E speedup.

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
| `kernel.flash_partials_batch` | `null` | `null` or int 0–65536; prefill flash-attention scratch multiplier. |
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

### Qwen4 expert-parallel and NPU controls

Typed schema keys (process-scoped):

| Key | Default | Values | Notes |
|---|---|---|---|
| `topology.ep` | `1` | integer 1–8 in the schema; a load accepts the same range | Env `HIPFIRE_TOPOLOGY_EP`. `HIPFIRE_QWEN4_EP=0` forces `1` and wins over any configured value, even an invalid one. A load needs pp = tp = 1, N visible devices that are all the same exact arch (gfx1100 or gfx1201; a mixed set is refused) and peer access between every pair of ranks, or it is refused naming the missing requirement. Under the developer-only `HIPFIRE_EMULATE_GPUS` the ranks alias physical devices instead (one visible device is enough). |
| `topology.pp` | `1` | integer 1–8 | Env `HIPFIRE_TOPOLOGY_PP`. Pipeline-parallel degree: the model's layers split into N contiguous stages across the first N visible devices (`hardware.devices` / `HIPFIRE_DEVICES`). An explicit load request with `pp` above 1 wins and the configured value is not read; a request of 1 or none uses `topology.pp`. An invalid configured value is refused naming `1..=8` only when it is the value in effect. PP does not combine with `tp` > 1 or with `topology.ep` > 1. |
| `fusions.npu_trunk` | `"auto"` | `off` \| `auto` \| `on` \| `emulate` | gfx1151, single device only; see the stable-controls table above. |
| `fusions.npu_trunk_calibration` | `""` | path string | See the stable-controls table above. |
| `kernel.npu_spillover` | `false` | bool; env `HIPFIRE_NPU_SPILLOVER` | Opt-in, gfx1151-only XDNA NPU spillover sidecar. Needs a pulled, verified `.xdna.zip` sidecar declared by the registry entry: with the flag on and no declared sidecar the load stays GPU-only (one log line); a declared sidecar that is missing or fails verification fails the load with a `hipfire pull` hint. Never active on other architectures. |

With `topology.ep` > 1 the loader refuses DFlash (including a draft path), DSpark, n-gram drafting, DDTree, adaptive KV, an EAGLE drafter, CASK, PFlash, an experts-per-token override and a non-`single` compute placement. Native MTP attaches only on an explicit request (`--spec mtp` / `speculation.mtp = "on"`; `auto` stays AR). Lane entry points, state snapshots and `HIPFIRE_MTP_SAMPLED_MODE=naive` are unsupported under Qwen4 EP.

The EP owner-routed prefill and the expert-staging levers have no typed schema key. They are `[developer]` keys (a legacy `HIPFIRE_FOO=value` maps to `developer.foo`), read from the process snapshot at load; restart the daemon after changing them. For the three `ep_*` levers only the exact value `0` turns the lever off.

| Developer key | Env | Default | Effect |
|---|---|---|---|
| `developer.fa_prefill_pm` | `HIPFIRE_FA_PREFILL_PM` | Q8_0 and native FP8: unset = on where qualified; explicit `true` / `1` or `false` / `0` wins | Codec-aware PeaceMaker full-causal prefill on exact gfx1201, H16/KV2/D256, 64..8192 query rows and context 1..262144. Q8_0 K/V and native FP8 E4M3 K/V use their own direct bodies after shared Q16 preconvert. FP8 token rows contain 512 codes and two f16 head scales (516 bytes); `FP8_KV_QUALIFIED` names exact gfx1201 for both PM routes. Unsupported codec/layout, replay recording and graph capture keep the incumbent route. Explicit `0` / `false` kills the route for that codec; this flag does not change auto KV selection. |
| `developer.fa_prefill_pm_qprep` | `HIPFIRE_FA_PREFILL_PM_QPREP` | Q8_0 and native FP8: unset = on where qualified; explicit `true` / `1` or `false` / `0` wins | Exact F5 Q-prep fold nested under both flags resolved for the actual cache codec. FP8 requires both flags enabled; disabling PM retains the incumbent regardless of this flag. Requires whole PM admission, 64 rotary dimensions, non-legacy half-split RoPE and a plain eager full-causal single-chunk layer. The codec must match both tier-plan entries: the Q8 or FP8 batched cache writer and corresponding masked attention. K then V are written with the matching production writer before `fa_prefill_pm_direct_codec`. FP8 has its own qualified rows; Q8 qualification never enables FP8. |
| `developer.moe_combine_v2` | `HIPFIRE_MOE_COMBINE_V2` | on where qualified (exact gfx1201) | Admitted grouped BF16-input combine v2; `false` / `0` restores the incumbent. Ordered top-k arithmetic is retained. |
| `developer.moe_iu4_wide_tiles` | `HIPFIRE_MOE_IU4_WIDE_TILES` | on where qualified (exact gfx1201) | Admitted VRAM-resident symmetric IU4 gate/up runs use the existing NT8 pair when checked slot occupancy permits; `false` / `0` keeps incumbent tiles. |
| `developer.prefill_iu4_producer_fused` | `HIPFIRE_PREFILL_IU4_PRODUCER_FUSED` | on where qualified (exact gfx1201) | Existing prefill RMSNorm/FWHT and SwiGLU/FWHT producers retain the exact eight-candidate IU4 search while avoiding separate preparation; `false` / `0` keeps the incumbent. |
| `developer.moe_router_fused_prefill` | `HIPFIRE_MOE_ROUTER_FUSED_PREFILL` | on where qualified (exact gfx1201) | Prefill-only exact softmax/top-8/renormalization fusion; all existing shape/execution prerequisites apply. `false` / `0` keeps separate operations. |
| `developer.moe_router_topk_multirow` | `HIPFIRE_MOE_ROUTER_TOPK_MULTIROW` | on where qualified (exact gfx1201) | Wave-per-row fused router (one wave32 per row, 8 rows per workgroup, any row count), byte-identical to the one-row fused router; requires fused-router admission. `false` / `0` restores the incumbent. |
| `developer.moe_shared_prepared_iu4` | `HIPFIRE_MOE_SHARED_PREPARED_IU4` | on where qualified (exact gfx1201) | Reuses one prepared shared-expert IU4 activation for selector and gate/up consumers; `false` / `0` keeps per-consumer preparation. |
| `developer.moe_rotate_i4_pm` | `HIPFIRE_MOE_ROTATE_I4_PM` | on where qualified (exact gfx1201) | Exact-gfx1201 PeaceMaker down-projection rotate/quantize twin of `qwen4_moe_rotate128_i4` (byte-identical); existing runtime-K, layout and execution admission remains required. `false` / `0` restores the HIP incumbent. |
| `developer.prefill_iu4_c8_pm` | `HIPFIRE_PREFILL_IU4_C8_PM` | on where qualified (exact gfx1201) | PeaceMaker eight-candidate IU4 rotate/RMSNorm-rotate twins (byte-identical), nested beneath the resolved producer-fusion route; preserves rotated f32 side output and complete activation records. `false` / `0` restores the HIP producers. |
| `developer.gdn_chunk_prep_pm` | `HIPFIRE_GDN_CHUNK_PREP_PM` | on where qualified (exact gfx1201) | PeaceMaker GDN chunk-prep twin (byte-identical) nested beneath the admitted MoE chunk-scan route; unsupported cases retain incumbent preparation. `false` / `0` restores the incumbent. |
| `developer.moe_shared_down_sigmoid_epi` | `HIPFIRE_MOE_SHARED_DOWN_SIGMOID_EPI` | on where qualified (exact gfx1201) | Shared-expert sigmoid-gate epilogue in the existing HIP `full_add_v3` entry, not a PeaceMaker route. Retains the admitted FMA/residual arithmetic. `false` / `0` restores the separate gated residual add. |
| `developer.gdn_qk_indexed` | `HIPFIRE_GDN_QK_INDEXED` | on where qualified (exact gfx1201) | Sequential Q8-state MoE prefill indexes key-head Q/K by value-head ratio instead of repeating Q/K; `false` / `0` restores repeat/copy. |
| `developer.gdn_moe_pre_fused` | `HIPFIRE_GDN_MOE_PRE_FUSED` | on where qualified (exact gfx1201) | Admitted MoE prefill uses the existing fused sigmoid/conv/norm/repeat preamble; `false` / `0` keeps separate launches. |
| `developer.gdn_moe_chunk_scan` | `HIPFIRE_GDN_MOE_CHUNK_SCAN` | on where qualified (exact gfx1201) | Admitted MoE prefill uses the existing GDN chunk scan with Q8 state and EF residual; `false` / `0` restores sequential processing. |
| `developer.ep_owner_routed_prefill` | `HIPFIRE_EP_OWNER_ROUTED_PREFILL` | on under EP (2..=8 ranks) | Owner-routed chunk MoE: a prefill chunk of at least 512 rows computes only its own experts' tiles and pulls the other ranks' tiles over P2P; every other chunk keeps the full-table route. Admitted on gfx1201 (every layer verified symmetric on every rank, the symmetric IU4 MoE route requested, below) and on gfx1100 (the F16 WMMA grouped MoE route, QT44/QT53 experts on every layer), always with the Flash-Next profile, `kernel.moe_grouped_gemm` on and the full (non-compact) scratch plan; a refusal logs `qwen4 EP rank r/n: owner-routed prefill not admitted (reason)`. |
| `developer.qwen4_moe_sym_iu4` | `HIPFIRE_QWEN4_MOE_SYM_IU4` | unset = requested on gfx1100 / gfx1151 / gfx1201 | Grouped symmetric IU4 MoE prefill route for Qwen4. A typed boolean: `true` / `1` requests it, `false` / `0` keeps the whole incumbent route. `hipfire config set developer.qwen4_moe_sym_iu4 false`, a TOML boolean (or `"false"`, `0`) and `HIPFIRE_QWEN4_MOE_SYM_IU4=false` all disable it, exactly like `0`; any other spelling (`on`, `yes`, `2`, an empty value) is rejected with `invalid value for developer.qwen4_moe_sym_iu4: expected a boolean: true, false, 1 or 0`, never ignored. The request also needs the IU4 producers built with exactly the two-candidate activation search: on gfx1201 that is `kernel.g12_a4c2` (default `true`), on gfx1151 `kernel.gfx11_a4_candidates = "2"` (default), and any other `-DIU4_A4_CANDIDATES` value in `diagnostic.compiler.hipcc_extra_flags` withdraws it. Layers whose routed-expert headers fail the symmetry check (asymmetric artifacts) stay on the incumbent route. On gfx1100 the route is requested by default like the other two (the shipped NT1 gate/up and down module with the K3 down producer; **TODO-evidence: default-on decision record, see the CHANGELOG entry**): unset and `true` / `1` request it, `false` / `0` kills it everywhere, and the regular prefill and the S1 `AggregateSymIu4` window route (receipt `aggregate-sym-iu4`, resolved with a symmetric MQ4 artifact and exactly the two-candidate activation search `-DIU4_A4_CANDIDATES=2` in `diagnostic.compiler.hipcc_extra_flags`) read the same request, so they always agree; any other candidate value withdraws it, and a forced-but-unsupported `=1` prints one diagnostic and keeps the incumbent. On gfx1201, owner-routed prefill therefore has no single request key: it follows this developer lever plus those typed keys; on gfx1100 owner-routed prefill stays on the F16 WMMA grouped route (this lever gates only the single-device IU4 arm and the S1 aggregate window). |
| `developer.moe_sym_iu4` | `HIPFIRE_MOE_SYM_IU4` | unset = on by capability | Symmetric IU4 grouped routed-expert prefill route for qwen35 MoE artifacts whose loader-verified routed experts are symmetric qt44 gate/up + symmetric qt53 down (gfx1201 PeaceMaker objects). A typed boolean like `developer.qwen4_moe_sym_iu4`; `false` / `0` is the kill switch and loads such an artifact with a WARNING onto its ~3.7x slower qt53 F32 prefill fallback. Without it, an arch lacking the objects (every arch but gfx1201 today) or headers that fail verification refuse the load. Independent of `developer.qwen4_moe_sym_iu4`; an explicit `HIPFIRE_IU4_PREFILL=0` also disables it. Numerics: BF16 slot boundaries, A4 activations (Qwen KLD +0.0205 vs the fallback). See `docs/env-vars.md`. |
| `developer.qwen4_moe_x_lds` | `HIPFIRE_QWEN4_MOE_X_LDS` | `auto` | Exact gfx1151 F16 WMMA MoE route: the grouped gate/up SiLU and down GEMMs take X-through-LDS entries, bytewise the per-wave entries. `auto` arms exactly when a layer's MoE runs on the F16 WMMA route (its symmetric IU4 arm, above, is not selected); the default IU4 route is unchanged. `1` forces it wherever admissible, `0` keeps the per-wave entries. See `docs/env-vars.md`. |
| `developer.ep_balanced_owners` | `HIPFIRE_EP_BALANCED_OWNERS` | on under EP (gfx1201 at 2..=4 ranks only) | Per-call balanced expert ownership on the host-mapped trunk MoE layers of an EP rank. Admitted on exact gfx1201 at 2..=4 ranks only (the certified owner-assign kernel's range); on gfx1100 the rank logs `balanced ownership not admitted (needs gfx1201); static split`, at 5..=8 ranks `... (needs at most 4 ranks); static split`, and runs the exact static `e % N` split. Off builds nothing. |
| `developer.ep_expert_stage` | `HIPFIRE_EP_EXPERT_STAGE` | on (gfx1201 only) | Admits long-prefill expert staging under EP. Applies only when the owner-routed route is installed on gfx1201 (a gfx1100 rank logs `expert staging not admitted (needs gfx1201)` and builds nothing) and every trunk MoE layer in the forward binds the compact table. |
| `developer.qwen4_expert_stage` | `HIPFIRE_QWEN4_EXPERT_STAGE` | on for gfx1201 and, for the borrowed-donor (packed) stage, on any discrete device whose measured host-to-device link is narrow (< 16 GB/s); off for unified memory and an unmeasured link; `1` / `0` force | Long-prefill host-expert staging: a long forward borrows the VRAM of the last two resident expert layers as DMA stages (no dedicated VRAM) and restores them before it returns, so decode residency is unchanged. The restore copies are pinned host RAM on top of the placement's pinned experts (2 x 1,275 MiB on the Flash-Next packed artifact), reported separately in the `qwen4 long-prefill expert staging` log line. With the variable unset the load measures the link once per device (64 MiB pinned `hipMemcpyAsync` H2D, one warm-up then the best of 5 event-timed copies, GB/s = 10^9 B/s); on gfx1100 this happens before the expert placement is sized, because the narrow-link default prefill chunk (`prefill.chunk_rows`) sizes it, and fast arches and an explicit `HIPFIRE_PREFILL_CHUNK_ROWS` never probe for the chunk. The stage charge the placement applies reads only the probe-free switch (explicit value, else gfx1201), so a measured link changes which expert layers stay resident only through that 8192-row chunk. Slab and tier layouts, whose stage buffers are dedicated VRAM, stage only under that probe-free switch (explicit `1`, or gfx1201) and otherwise keep the zero-copy mapped reads. A failed probe is "unmeasured" and never narrow. Unified-memory devices are never copy-probed. gfx1201 (an R9700 in a Gen5 x16 slot included) keeps its existing default whatever the probe returns. |
| `developer.qwen4_expert_stage_min_rows` | `HIPFIRE_QWEN4_EXPERT_STAGE_MIN_ROWS` | `min(2048, prefill chunk rows)`; a fixed `768` on a measured narrow discrete link off gfx1201; gfx1201 `2048` | Forwards with at least this many rows stage (integer ≥ 2, authoritative when set); smaller ones, decode and verify keep the mapped reads. The default is capped at the prefill chunk so a full chunk qualifies where the chunk is below 2048 (gfx1100 on a fast link: 1536); the narrow-link 768 is not lowered to the chunk, so a chunk admitted below 768 never stages. |
| `developer.qwen4_expert_stage_capture` | `HIPFIRE_QWEN4_EXPERT_STAGE_CAPTURE` | on | `0` keeps the native-MTP prompt-fill tiles on the mapped reads. |
| `developer.qwen4_decode_share` | `HIPFIRE_QWEN4_DECODE_SHARE` | `on` (unset or `auto`; `DECODE_SHARE_DEFAULT`) | Decode-share mode: `on` (default on admitted loads), `off` / `0` (and any unrecognised value), `bind`, `bank`, `fixed`. Lends the long-prefill scratch to a donor tail of the coldest resident experts' VRAM (mirrored in pinned host RAM, read through a BorrowPrefill expert view while borrowed, restored after the prompt) and, in `on`, charges only the dedicated forward in the expert-placement reserve so the hot budget grows by the difference. Scope: exact gfx1201, single device, ranked hot set, compact scratch plan; refused (silently unless set explicitly) with EP, a pp band, the NPU trunk, S1, DFlash, a CPU expert split, continuous batching or a chunk of 512 rows or fewer. An explicit `developer.qwen4_expert_hot_budget_mib` does not grow, so `on` becomes `off` for that load. `bind` (second expert view only), `bank` (owned test scratch) and `fixed` (real donor tail, plain budget) are A/B controls. Disable with `HIPFIRE_QWEN4_DECODE_SHARE=off`. Admission, the placement charge, the loan protocol and the log lines are in the `HIPFIRE_QWEN4_DECODE_SHARE` row of [env-vars.md](env-vars.md). |
| `developer.qwen4_decode_share_verify` | `HIPFIRE_QWEN4_DECODE_SHARE_VERIFY` | off | `1` verifies the donor tail against its host mirror right after materialization and, in `fixed` / `on`, after every donor restore (debug oracle; a mismatch fails the load or poisons the loan). |
| `developer.qwen4_decode_share_sync` | `HIPFIRE_QWEN4_DECODE_SHARE_SYNC` | off | `1` host-waits for the donor restore right after it is enqueued (A/B timing; puts the restore inside TTFT). `fixed` / `on` only. |
| `developer.qwen4_decode_share_log` | `HIPFIRE_QWEN4_DECODE_SHARE_LOG` | off | `1` logs one line per donor borrow epoch (tiles, class span, zeroed and restored MiB, host wait ms). `fixed` / `on` only. |
| `developer.qwen4_decode_share_shadows` | `HIPFIRE_QWEN4_DECODE_SHARE_SHADOWS` | on | `0` keeps the F16 weight shadows and the shared-expert INT8 shadows resident. Default: an admitted `on` / `fixed` gfx1201 load keeps them in the start of the donor tail (rebuilt per borrow epoch; read only by borrowed prefill tiles), so their ~1,477 MiB of reserve becomes resident experts under `on`. |
| `developer.qwen4_d2_default` | `HIPFIRE_QWEN4_D2_DEFAULT` | unset = measured selector | Typed boolean (`true` / `1`, `false` / `0`; any other spelling is rejected, never ignored). The per-load D2 measured default of an unset `memory.offload_exec` (see the "D2 measured default" bullet of the partial-offload section): unset runs the measured decode-cost comparison (the production CPU arm on the model's real host experts plus the copy-transport handoff, against PCIe from live link samples; D2 only when at least 15% faster), `true` turns the D2 CPU split on for every host-placed Flash-Next layer it can serve (hot-set tier layers, un-tiered host-mapped aggregate QT44/QT53 layers and host-mapped expert-slab layers whose present experts all have a CPU decoder) without the comparison (lab and receipt runs), `false` never arms it. An explicit `memory.offload_exec` always wins over both values, and `true` does not bypass `HIPFIRE_MOE_CPU_EXPERTS=0`, unified memory or an unsupported layer. Decode and `<=4`-row verify only; prefill, placement and host allocations are unchanged. |
| `developer.qwen4_mtp_head_d2` | `HIPFIRE_QWEN4_MTP_HEAD_D2` | unset = follows the trunk's D2 | Typed boolean (`true` / `1`, `false` / `0`; any other spelling is rejected, never ignored), read once per load when the native MTP head attaches. Unset and `true` run the head's routed experts (`mtp.layers.0.mlp.experts.*`, host-mapped, 512 experts / top-10) on a rows=1 D2 CPU split executor instead of reading them over the link, **only when the trunk's D2 split is armed on the same load** (a fast-link card whose D2 default is off keeps the PCIe-read head) and the bundle is not expert-parallel; a head with a keep mask or a refused split admission keeps the GPU route. `false` / `0` never arms (the kill) and keeps the pre-L7 GPU head byte-for-byte; the only difference is one `off` load line. Return/ready/join transports are the trunk's process snapshot (`HIPFIRE_QWEN4_D2_RETURN/READY/JOIN`); there is no head-specific transport knob. Only draft numerics move (CPU arm vs the GPU grouped kernel, ULP level): the exact verify fixes every target token. Cost: +40 B device (`masked_indices`) and 112,760 B pinned host; the trunk's executors, capacity rows and census are unchanged. Every load prints one line, `qwen4 MTP head D2: armed (rows=1, ...)` or `qwen4 MTP head D2: off (<reason>)`. The head step is eager-only: under graph capture an armed head returns an error, never a GPU fallback. |
| `developer.qwen4_slowlink_prefill` | `HIPFIRE_QWEN4_SLOWLINK_PREFILL` | unset = AUTO default selector: automatic candidate selection from 1,537 rows (first genuinely multi-chunk request, one row past the 1,536-row tile bound); `0` kill switch; `1` forces the shipping-path candidate | **S1 slow-link layer-major prefill; AUTO is the default selector.** A typed boolean (`true`/`1`, `false`/`0`; any other spelling is a load error). Unset runs the automatic candidate selector on a load and request that satisfy every check below; `1` forces the shipping-path candidate (a capacity shortfall fails the load with its exact byte deficit instead of falling back to the incumbent); `0` is the kill switch and forces the incumbent, and the automatic selector never probes the link on a killed load. Considered only on exact gfx1100 with a discrete device whose measured pinned host-to-device link is narrow (< 16 GB/s; the existing cached probe, never the card's own sysfs width), the qualified MQ4 packed Flash-Next recipe, a single-device whole-model auto-placed load and a long enough context, with a request of at least 1,537 rows (automatic and forced share that floor); a short context, a fast link, unified memory or an unmeasured link never selects it under either selector, and neither does expert parallelism, a pipeline band, a hot set, `whole-first`, an explicit `developer.qwen4_expert_vram_layers`, the CPU expert splice, an expert-keep artifact, a selected NPU trunk or externally captured prefill; each of these is the incumbent before any reserve or mutation. An explicit `developer.qwen4_expert_stage` = `0` is an opt-out of this route too. When selected, the load admits placement for it first: the purpose's mixer tile (768 rows for the product profile that every state, headroom, greedy and perf cell runs on; 512 rows for the primitive self-test profile that proves kernel correctness only), the prompt arena, a dedicated double-buffered expert ring plus one next-layer lookahead page store and the session boundary state are charged together against the actual post-initialization free device memory, and the placement keeps the largest trunk-layer resident count that fits (ten preferred, never a floor; zero is valid when the base budget fits and a host layer remains). Each host layer's routed experts then cross the link once per admitted window (layer by layer, the next layer's weights prefetched during the current layer's compute) instead of once per prefill chunk; route-dependent work is never run early. It is not bitwise the incumbent's 8,192-row manifest: the gate requires same-route determinism, exact greedy ids against the beta autoregressive route and the paired code24 decode KLD verdict. Receipts (`Qwen4Bundle::slowlink_receipt`) report the route actually run, its windows, bytes and spans. Measured base on the XTX at pp16384 is 1,738.9-1,750.2 tok/s (accepted release, ~1.87x beta); 2,000 tok/s is not a release block. The intended followup is the quality-gated PLACE + IU4 candidate (ordered resident-layer placement, `developer.qwen4_slowlink_resident_layers`, plus the gfx1100 explicit-opt-in IU4 window route `AggregateSymIu4`, above); QSA is excluded (fails 4K, repaired separately, never part of this stack) and no qualified performance beyond the measured base is claimed. The validation contract and its current evidence are in [`VALIDATION.md`](VALIDATION.md#s1-slow-link-layer-major-prefill-claim-scoped). |
| `developer.qwen4_sp` | `HIPFIRE_QWEN4_SP` | unset = on for exact gfx1100, off elsewhere; `0` kills | Typed boolean (`true` / `1`, `false` / `0`; any other spelling is rejected, never ignored), read once per Qwen4 load. Unset runs hybrid SP on exact gfx1100 (the default) and not on other architectures; `false` / `0` is the kill switch (touched-only SP then runs alone) and `true` / `1` asks for it on any arch (all but gfx1100 decline by name). SP, the short-prompt per-window mode of the S1 slow-link prefill (top-k\* experts stream, the rest run on the CPU, one canonical combine), when every load-time admission rule holds: an admitted S1 profile, exact gfx1100, the `Aggregate` window route (or `AggregateSymIu4`, whose SP windows run the F16 `Aggregate` route by a per-window override; see `docs/env-vars.md` `HIPFIRE_QWEN4_SP`), 512 experts, CPU quants for the host layers, a splice pool, the route-region fit (`sp_max_rows = min(5,376, fit)`, at least 2,048), the pinned-host fit and `serve.prefix_cache` off. A failed rule is a sticky named `sp: <reason>` decline that leaves S1 unchanged. Adds no VRAM: resident layers and the S1 census are identical with SP armed. When a rule declines the hybrid mode, touched-only SP (`developer.qwen4_sp_touched`, default on) runs instead and the load says why. Windows of 5..=`sp_max_rows` rows with no interior session cut, no one-row tile and capture `None`/`Bounded` take SP; every other window runs today's sweep. See [env-vars.md](env-vars.md). | With `developer.qwen4_sp_touched` also on, this hybrid mode wins and touched-only is its fallback.
| `developer.qwen4_sp_touched` | `HIPFIRE_QWEN4_SP_TOUCHED` | unset = on for exact gfx1100, off elsewhere; `0` kills | Typed boolean (`true` / `1`, `false` / `0`; any other spelling is rejected, never ignored), read once per Qwen4 load. Unset runs SP's touched-only mode on exact gfx1100 (the default, exact: byte-identical to the stream-all sweep) and not on other architectures; `false` / `0` is the kill switch and `true` / `1` asks for it on any arch (all but gfx1100 decline by name). The mode: the S1 slow-link prefill runs its own per-panel pipeline on the same short windows as `developer.qwen4_sp`, but streams only the experts each host layer routes to (one selective ring schedule, no CPU expert, no K2-ized F32-tail tile, no pinned owner, no new VRAM), byte-identical to the stream-all sweep. Admission needs an admitted S1 profile, exact gfx1100, the `Aggregate` or `AggregateSymIu4` window route, an S1 window of at least 2,048 rows and a ring with one page per 32 experts; a failed rule is a sticky named `sp: <reason>` decline that leaves S1 unchanged. With `developer.qwen4_sp` also on, the hybrid mode wins. Full semantics in `docs/env-vars.md` (`HIPFIRE_QWEN4_SP_TOUCHED`). |
| `developer.qwen4_expert_vram_layers` | `HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS` | unset (automatic placement) | `N` keeps the routed experts of trunk layers before `N` in VRAM and places layers at or past `N` in pinned host RAM. |
| `developer.qwen4_slowlink_resident_layers` | `HIPFIRE_QWEN4_SLOWLINK_RESIDENT_LAYERS` | unset or `front` = leading layers `0..N` | PLACE: which trunk layers an admitted S1 slow-link load keeps resident; the COUNT `N` stays the admission's decision. `front` (default, case-insensitive) is the incumbent placement; `spread` is the evenly-spaced set; otherwise a comma list of exactly `N` distinct layer indices. Any other value fails the S1 load naming the variable; MTP-layer experts stay host-mapped. Full semantics in `docs/env-vars.md` (`HIPFIRE_QWEN4_SLOWLINK_RESIDENT_LAYERS`). |
| `developer.qwen4_expert_hotset` | `HIPFIRE_QWEN4_EXPERT_HOTSET` | unset: `<model>.expert-hotset` is used when it exists (gfx1201) | Lever1 expert hotset, a decode-ranked per-expert VRAM placement (sidecar `kind` `hipfire.qwen4.expert_hotset`, version 1; format and binding in `docs/env-vars.md` and below). Unset applies the sidecar beside the model when one exists; `0` is the kill switch (whole-layer placement); an absolute path names the sidecar explicitly; a relative or empty value is a policy error. A sidecar that does not apply (missing, unreadable, bound to another base, wrong dimensions, another architecture, explicit `developer.qwen4_expert_vram_layers`) keeps whole-layer placement and never fails the load. The expert budget keeps a 512 MiB VRAM margin. |
| `developer.qwen4_expert_hotset_policy` | `HIPFIRE_QWEN4_EXPERT_HOTSET_POLICY` | unset: `ranked` | Developer-only, default-off placement policy for the Qwen4 expert hotset on exact gfx1201. `ranked` (the default, also when unset) is the current ranked per-expert hotset, unchanged. `whole-first` runs the same baseline whole layers as `developer.qwen4_expert_hotset` = `0` (same placement, materialization and route; no partial or tiered layers, no mixed experts, hot experts = 0), does not consult the sidecar even when `developer.qwen4_expert_hotset` names a path, and keeps the baseline long-prefill staging, charged to the same budget. Any other value is a load error naming the variable; where no hot set could apply (pipeline-parallel band, expert parallelism, explicit `developer.qwen4_expert_vram_layers`, unified memory) it is ignored with one log line, and gfx1100 ignores it silently. It preserves decode safety but may lose the ranked hotset's long-prefill gain, and decode and prefill may not both fit in the available memory; no new performance claim. Details in `docs/env-vars.md`. |
| `developer.ep_peer_allreduce_decode` | `HIPFIRE_EP_PEER_ALLREDUCE_DECODE` | unset (canonical) | `1` selects the legacy unrooted peer all-reduce diagnostic, `0` selects RCCL. |

**S1 session-cache pressure host snapshots (opt-in with S1; no setting of its own).** When the S1 slow-link profile is attached to a load, the session cache is armed with a generic, bounded pressure-selected pinned-host snapshot policy; with no S1 profile nothing is armed and the non-S1 cache and admission behave byte for byte as before, with no extra allocation. The policy needs no new variable or config key and no architecture, card or format gate: its host budget is the cache's existing `memory.session_cache_bytes` (the gate fixture configures 2 GiB), and its device budget is the admitted profile's `device_snapshot_budget_bytes`, the admitted remaining free device memory after the non-expert and resident-layer bytes and the joint reserve (which already includes the safety margin). That budget is discretionary cache capacity above the full S1 peak plus safety, not extra reserve and not safety credit. Under device memory pressure a capture is placed in pinned host memory instead of being skipped: the part is copied directly into its final pinned allocation (never through a full device clone), the host tier is bounded by the host budget with normal eviction across pool, pending, resident, deferred and parked owners, restores copy straight back into the live context, and the existing device guard (and the 512 MiB floor for S1 device-policy allocations) still applies. Lineage, delta ranges and abort semantics are unchanged. Host ownership is reported beside the device-only fields in `SessionCacheUsage` (`host_budget_bytes`, `host_pool_bytes`, `host_pending_bytes`, `host_resident_bytes`, `host_deferred_release_bytes`, `host_parked_bytes`, `host_total_bytes()`, and the `host_captures_total`, `host_restores_total`, `host_evictions_total` and `host_skips_total` counters); the device fields and `device_total_bytes` stay device-only. The unchanged 512 MiB whole-product minimum-free gate and the 768-row product tile are not relaxed (the resident layer count is adaptive, ten preferred), and this storage tier does not by itself recover the active-compute headroom the historical hold missed. The envelope figure is a historical source-derived estimate with no GPU pass and no final base CI/serve PASS claimed: a source-derived host envelope estimate (S1 host census 52,930,156,544 B + the 2 GiB host pool + a 620,021,760 B pending ceiling + a 2 GiB driver allowance = 57,845,145,600 B against the 54 GiB declaration of 57,982,058,496 B, 136,912,896 B to spare) is an estimate, not a measurement, and no GPU pass is claimed.

**S1 bounded load-lifetime weight and F16-shadow arenas (S1 only; no setting of its own).** When, and only when, the S1 slow-link profile is selected on a load, the resident non-expert weights are carried in a small number of exact-size device roots, and the BF16 weights that already get an F16 shadow have their shadows laid out in a second family of roots, instead of one allocation per weight. The layout is one plan per load: members are placed by bytewise name order at 256 B alignment with deterministic next-fit, each root is bounded by the largest allocation the load would already have made for an eligible weight (never below 256 B), and the same plan is used by the admission census, by the fulfilment that uploads the weights and by the F16 shadow cache, so the bytes charged are the bytes allocated. Weight roots are allocated once at load, exactly the planned size, and live until the model is freed (a rollback or a failed construction releases them only after every view over them is gone). Shadow roots are allocated lazily, exactly the planned size, the first time a planned shadow is converted, and stay pointer-stable; a request that fits inside a member's reserved span never overruns it, and a shadow source that is not in the plan (for example a separately accounted native head) keeps the existing individual allocation. Allocation-granule padding is therefore charged once per root instead of once per weight, which lowers the device bytes the admission charges and so the pressure that reaches the host snapshot tier described above; the host-tier policy and session semantics themselves are unchanged by the arenas. There is no new variable or config key, no architecture, card, format or model-name gate (the plan is built from the weight roles, never from a quant number), and no change to the mixer tile (768 product, 512 primitive), the adaptive resident layer count (the largest that fits, ten preferred), the 1 GiB runtime reserve, the 512 MiB safety margin or the unchanged whole-product external + HIP 512 MiB minimum-free gate. Non-S1 loads, the fast, unified-memory and unmeasured-link paths and the incumbent route build no plan and keep every individual allocation and teardown path exactly as before. Source-derived fixture projection (a real-model CPU census, not a measurement and not a product constant): 1,096 core weights, 291 shadow owners, a largest eligible allocation of 675,430,400 B and a canonical plan of 9 weight roots plus 2 shadow roots (the 9+2 shape is a fixture the receipt tests build; no check selects on it), with a projected recovery of 1,010,670,848 B, which leaves a projected 48,509,588 B of slack at N = 10 / tile 768 after the 1 GiB runtime reserve and the 512 MiB safety margin, and 113,608,212 B at the primitive profile's tile 512 with its 217,821,184 B diagnostic. These are admission projections from source; **no GPU run has confirmed the recovery and none is claimed** (the recorded failed hold's receipts stay as they were). The gate's `load.placement` detail additively reports `arena_census`, and the run records `arena_census_end` once the lazy shadows exist. Per family it reports the member and root counts, alignment and root cap, the PLANNED payload, root-request and granule-rounded backing bytes with their alignment and granule padding, and the ACTUAL live roots, capacities and backing taken from the allocators' own capacities; `capacity_proof` is `exact` only when every root is live at exactly its requested capacity and the backing equals the plan's. A planned lazy root is never reported as actual (`planned_is_actual` is always false). A root whose capacity differs from its request, a census of the wrong length, capacities with no plan, a weight root that is not live after load, or an S1 profile without a plan fails the receipt (`load.placement`, or `load.arena_capacity` after the run); the non-gating residual diagnostic, the seven primitive checks and the strict product headroom are unchanged.

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
| `max_seq` | `32768` schema value; automatic on growing Qwen VMM KV | int 512–1048576 when set | Only an authored value (CLI, per-model or global TOML, one-shot env) is sent to the daemon; when omitted the daemon chooses per family. Automatic: min(checkpoint `config.text_config.max_position_embeddings`, card capacity after weights, admitted prefill scratch, and VRAM headroom). Explicit user values override either bound, including the trained window. |
| `thinking` | `"on"` | `on` \| `off` | Thinking **mode** (on/off). Independent of effort and of any hard think-token cap. When off wins, effort and cap controls are dropped with a warning. |
| `reasoning_effort` | `"auto"` | `auto` \| `none` \| `low` \| `medium` \| `high` \| `xhigh` \| `max` | **Semantic** effort only — prompt / framing strength. Field meaning never changes by family and is **never** reinterpreted as a token budget. `auto` means no preference: any resolved non-`auto` value from user config, per-model config, CLI, or an HTTP request wins; only when every layer is unset or `auto` does hipfire's default apply, which is `medium` on the effort-native Qwen3.8 and Ornith 1.5 contracts (the official Qwen3.8 template itself defaults to `xhigh`, so `medium` is a deliberate hipfire choice, not the template default). `reasoning_effort = "xhigh"` restores the previous behavior. Other contracts keep their own `auto` behavior. Unsupported or cross-family values are dropped with a warning (see below). |
| `thinking_budget` | `"med"` | `off` \| `low` \| `med` \| `high` \| `xhigh` \| `max` \| `uncapped` | Legacy **named cap preset** only on models that still honor that config route. Not an effort dial. On effort-native models the string is dropped with a warning (no implicit cap). `off` is not a token cap: on a contract that honors the named route it disables thinking like `thinking = "off"`. |
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
| `off` | no cap; thinking disabled (same as `thinking = "off"`) |
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
- Thinking disabled wins: effort and cap are dropped with a warning. On Qwen3.8 and Ornith 1.5 that includes the hipfire-resolved `medium` default.
- Qwen3.8 and Ornith 1.5 resolve effort through the existing CLI/config resolution layer: an explicit non-`auto` value from user config, per-model config, CLI, or HTTP wins; if every layer is unset or `auto`, the hipfire default is `medium`. The official Jinja templates are unchanged and still default to `xhigh` when no effort reaches them. This default never implies a think-token cap.
- Recognizable unsupported / cross-family values are normalized or dropped with
  `[WARN: INVALID CONFIG]` (server log + OpenAI response metadata under
  `hipfire.reasoning` / `hipfire.config_warnings`). Malformed types and
  out-of-range integers remain hard errors.
- Template probing checks the **loaded template**, not the model name: effort
  must change rendered prompt text. The registry Qwen3.8-27B MQ6 template
  supports `low`, `medium`, and `xhigh`. The ordinary loader prefers a
  configured/per-model template override; experimental multislot loads the
  embedded HFQ template directly. Both probe their effective template.
  Before 0.4.3, experimental multislot incorrectly reported `false` / `[]`
  without probing, causing valid effort settings to be dropped. Check `/health`:
  `reasoning_effort_native: false` means native effort was not established.
  An empty `reasoning_efforts` can also mean the baseline template render failed,
  or the tokenizer/template was missing, not that the model family lacks effort.
  `/health.reasoning_effort_probe_error` is `null` when the probe succeeds,
  including a template that simply ignores effort; otherwise it contains the
  failure reason. A load-time probe-error log also identifies the effective
  template source (including the path for a user override).
  For a genuinely non-effort-native Qwen template, use `thinking_budget` or
  `max_think_tokens` for a cap instead. Effort is never converted to a cap.
- `native` is **not** a `reasoning_effort` value or an alias for `auto`.
  Use a supported value from `/health.reasoning_efforts` (Qwen3.8:
  `low`, `medium`, `xhigh`). Unsupported strings can fall back to the
  configured/default effort; they do not enable missing template support.

**Family examples** (full HTTP surface: [`SERVE.md`](SERVE.md)):

| Family | Mode | Effort | Cap |
|---|---|---|---|
| **Qwen3.8** | `enable_thinking=false` → native empty closed think | `low` \| `medium` \| `xhigh`; when all layers are unset/`auto`, hipfire uses `medium` (official template default is `xhigh`; set `xhigh` explicitly to restore it); system-turn semantic prompt steering only — **not** a budget | Positive `max_think_tokens` force-closes; 0/absent uncapped. Named `thinking_budget` dropped+warned |
| **Ornith 1.5** | same Qwen on/off (`enable_thinking=false` → empty closed think) | Qwen3.8-compatible `low` \| `medium` \| `xhigh` semantic prompt steering (loader injects the same Qwen3.8 semantic block); when all layers are unset/`auto`, hipfire uses `medium` (medium unsteered; official default `xhigh`; the registry coding profile keeps `xhigh`); **not** a budget reinterpretation | Same Qwen integer continuation cap as Qwen3.8; 0/absent uncapped. Named `thinking_budget` dropped+warned |
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
| `kv_backend` | schema `"vmm"`; automatic when not authored | `legacy` \| `vmm` |
| `state_quant` | `"auto"` | `auto`, `q8`/`int8`, `q8_ef` (Qwen3.8-Flash-Next only), `fp32`/`f32`, `q4`/`int4` (Qwen3.5 only). GDN/DeltaNet recurrent-state storage (`memory.state_quant`, env `HIPFIRE_STATE_QUANT`); `auto` = Flash-Next F32 on exact gfx1151, Q8 on every other arch, and never selects `q8_ef`; explicit values honored or refused, never rewritten; semantics in the Qwen4 paragraph below the Flash-Next table |

### Mode, K, and V

`memory.kv_cache` / `--kv-mode` is the whole-cache preset: quantized modes
seed a split *(K, V)* pair, while native `fp8`/`bf16` are one K+V layout
without an independent V axis. Typed `memory.kv_k` / `memory.kv_v` (CLI
`--kv-k` / `--kv-v`) are **Qwen-only** axis overrides, empty by default and
omitted from IPC when unset. They replace their respective axis of a split
pair; see the native override refusal below.

DeepSeek V4 `f16` on gfx1151, measured on a Ryzen AI Max+ PRO 495 (single
device, MQ2R 0731 artifact, DSpark sidecar): loads and generates coherent
text; decode is neutral (25.4 -> 25.8 tok/s short-context with DSpark,
drafter acceptance 24% -> 25%) while pp16384 pays for the storage convert
(163.5 -> 139.3 tok/s; single runs each, the f32 reference measured on the
same host and shape). Pick `f16` on gfx1151 for long-context capacity, not
throughput; the capacity delta itself is not yet measured on this host.

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
layout) on exact **gfx1201** single GPU (`pp = tp = 1`) when the load is
native-eligible: attention geometry H24 / Hkv4 / D256 (dense 27B-class) or
H16 / Hkv2 / D256 (A3B MoE-class), no `kv_adaptive`, no CASK sidecar, no
slots, no TP/PP split. Native `bf16` stays H24 dense-only: `auto` never
resolves an A3B load to `bf16`. Every other Qwen load resolves **q8/q8**:
gfx1100, gfx1151 and every other arch; gfx1201 with a different geometry,
with adaptive or CASK, on slots, or on PP / dense TP / MoE EP (those
policies have no native constructor); and Qwen3 dense (flat constructor).
gfx1200 / gfx11 / gfx94x never inherit fp8. Registry Qwen cards leave mode
as `auto` (a registry `default_kv_mode = "q8"` is not lowered into config).

**Non-Qwen families:** Maple keeps BF16/BF16 (registry and direct-path
auto; explicit `--kv-mode q8` still works). Gemma4 keeps its sliding tier
Q8 on every variant; its full-attention tier follows `kv_cache`: `auto`/`q8`
→ Q8 on every arch (no Gemma4 attend site admits native FP8),
`legacy-asym3` → the previous Givens Asym3 K + Q8 V,
`fp8`/`bf16` → load error, anything else → Q8 with a warning. Only Q8/Q8
loads take batched prefill and the session cache. llama HFQ,
MiniMax and LFM2-MoE resolve `auto` to q8. DeepSeek4 keeps its F32
compressor default (or explicit F16); V is not independently selectable
there. Other carriers keep their existing site policy (llama, MiniMax and
LFM2-MoE resolve `auto` to q8).

**Qwen3.8-Flash-Next / Qwen4 (arch 16)** has its own table: `kv_cache`
sets the storage of the QSA full-attention K/V arenas. The GatedDeltaNet
recurrent state is separate and follows `memory.state_quant` (below the
table): `auto` is F32 on exact gfx1151 and Qwen3.5's Q8 DeltaNet format on
every other arch; `q8` is the kill switch, `fp32` the explicit opt-in and
`q8_ef` the explicit-only error-feedback variant of `q8`.

| `kv_cache` | gfx1201 | gfx1151 (Halo), gfx1100 (7900 XTX) | other gfx11/gfx12 | gfx942 (MI300X) | other arches |
|---|---|---|---|---|---|
| `auto` / unset | `fp8` | `q8` (Q8_0 K/V, split-KV decode attention) | `bf16` | `bf16` (never `fp8`) | `bf16` |
| `bf16` | exact reference state (F32 K/V arenas, BF16-valued index keys) | exact F32 state (opt-in); F32 split-K decode attention on gfx1151 only | same as gfx1201 | same | same |
| `fp8` | E4M3 K/V with one f16 scale per head and token; indexer raw/pooled keys stored as BF16 (the same values) | refused | refused | explicit opt-in: the same row (E4M3 codes plus one f16 scale per head and token) in the gfx942 FNUZ-hardware format; indexer raw/pooled keys stay F32 | refused |
| `q8` | Q8_0 K/V: the dense q8 KV row (32-element blocks of one f16 scale and 32 int8 codes, encoded as `kv_cache_write_q8_0`); indexer keys stored as BF16 (the same values) | same (the `auto` default) | same (explicit opt-in) | refused | refused |
| anything else (`fwhtN`, `asymN`, `f16`, …) | refused | refused | refused | refused |

`auto` picks fp8 only on exact gfx1201 and only for a head geometry the
kernels implement (head_dim 256, even KV-head count; Flash-Next qualifies);
otherwise it stays `bf16`. On exact gfx1151 and exact gfx1100 `auto`/unset picks `q8` where the
model has head_dim 256 (Flash-Next qualifies) and falls back to the exact
`bf16` state where it does not; an explicit `kv_cache = "q8"` with another
geometry is refused rather than rewritten. `kv_cache = "bf16"` is the way back
to the exact F32 state on every arch, gfx1151 and gfx1100 included. On exact gfx942
`auto`/unset stays the exact `bf16` state; fp8 is served only by an explicit
`kv_cache = "fp8"` (same geometry requirement, refused rather than rewritten
when the model does not meet it). On every other arch `auto` never picks
`q8`; an explicit `q8` needs head_dim 256.
On gfx1151 and gfx1100 the decode attention follows the K/V format under the
`HIPFIRE_QWEN4_QSA_DECODE_SPLITK` switch (`=0` disables both routes): split-KV
for q8 K/V (exact gfx1151 and exact gfx1100), split-K for the F32 state
(exact gfx1151 only). Both are KLD-gated, not bit-exact against the per-head
kernel.
The native MTP layer keeps the exact state. On Flash-Next, each trunk QSA
layer's context state is 4,736 bytes per token at `bf16`, 1,352 at gfx1201
fp8, 1,672 at gfx942 fp8 and 1,408 at q8 (12 layers at 262,144 tokens:
14,208, 4,056, 5,016 and 4,224 MiB). The gfx942 fp8 charge is the K and V
rows (2 × 516 = 1,032 B; a row is 2 KV heads × (256 E4M3 codes + one f16
scale)), the F32 raw index key (512 B) and the F32 pooled key (one 512 B row
per four tokens, 128 B); the index state is the same F32 state as `bf16`.
gfx942 uses legacy (fully allocated) context storage, so admission charges
the whole `max_seq`. The gfx942 FNUZ hardware encoder stores code bytes with
one f16 scale per head and token (the decoder doubles the scale for the FNUZ
bias). Stored bytes equal gfx1201's except that `0x80` is stored as `0x00`
(+0) for both input negative zero and negative underflow to zero. FNUZ has no
negative zero (`0x80` is its NaN), so a stored K/V row never contains `0x80`;
every other finite code is stored unchanged. Decoded values equal the gfx1201
fp8 row's up to the sign of zero; full byte identity is not claimed. The
Q8 GDN state is 3.88× smaller than F32. `memory.max_seq` goes up to the
native 262,144. Index keys hold the same values in every format, so
selection is unchanged; only the attention over the selected K/V rows sees
the quantization.

**GDN recurrent state: `memory.state_quant`.** Storage of the
GatedDeltaNet/DeltaNet recurrent state, separate from `kv_cache`. Values
`auto` (default), `q8`/`int8`, `q8_ef`, `fp32`/`f32`, `q4`/`int4`; legacy key
`state_quant`, env `HIPFIRE_STATE_QUANT`, CLI `hipfire config set state_quant
<value>`. `auto` is the per-model/arch default: on Qwen3.8-Flash-Next it is
**F32 on exact gfx1151 (Halo)** and Q8 on gfx1100, gfx1201 and every other
arch (each promoted independently); on the stock-FP8 profile (gfx942) F32 is
pinned and `q8` and `q8_ef` are refused; Qwen3.5 keeps Q8. `q4`/`int4` is
Qwen3.5-only and `q8_ef` is Qwen3.8-Flash-Next-only (Qwen3.5 refuses it). An
explicit value is honored or refused for the model/arch, never rewritten, and
there is no free-memory fallback from F32 to Q8. The Q8 and `q8_ef` formats
need 128×128 heads; Flash-Next qualifies.

**Kill switch:** `hipfire config set state_quant q8`, `state_quant = "q8"` in
the TOML, or `HIPFIRE_STATE_QUANT=q8` restores the Q8 recurrent state on
gfx1151.

**`q8_ef` (explicit opt-in).** `q8_ef` is plain Q8 recurrent state plus one
**live f16 error-feedback residual** per GDN layer per live owner (the request
state and each lane owner): the rounding error of each Q8 requantization is
kept in the residual and added back into the next step's state, so the
per-token requantization error is fed forward instead of dropped. It is never
selected by `auto`, by any default, or on any arch: it is used only when
`memory.state_quant = "q8_ef"`, `hipfire config set state_quant q8_ef` or
`HIPFIRE_STATE_QUANT=q8_ef` asks for it explicitly, and it is refused on the
stock-FP8 profile, on Qwen3.5 and without 128×128 heads. It is also refused on
every arch when the `HIPFIRE_QWEN4_GDN_Q8_INLINE=1` developer flag is set:
no GPU has run a `q8_ef` inline-WY chunk kernel, so none ships, and a `q8_ef`
chunked prefill always runs on its F32 reconstruction; unset one of them. Only
the live residual
carries EF: the capture ring, the transaction mirror and every session-cache,
prefix and disk snapshot hold **plain Q8** codes and scales, byte-identical in
layout to `q8`, so no snapshot or cache entry grows. The residual is zeroed,
in stream order, on initial reset, keep-0 rollback, partial rollback (the Q8
ring is replayed with the incumbent Q8 rollback kernel and the residual is then
zeroed), transaction restore, session/prefix/disk restore and lane reinit; an
uninterrupted AR step, a prefill launch and a full all-accept commit keep the
final residual. No historical residual is stored or restored.

**`q8_ef` numerics and scope.** A restored or rolled-back continuation
**intentionally differs** from the uninterrupted EF continuation, because the
residual restarts from zero after the restore or rollback. **MTP == AR is not
guaranteed** under `q8_ef`: native MTP resets the residual on rejection and
plain AR does not, so greedy IDs and live state may differ between routes, and
cold versus restored continuations may differ; those cross-route differences
are reported as diagnostics, while same-route, same-history determinism is
required. `q8_ef` is not bit-identical to `q8` or `fp32`. It uses its own
session-cache scope, `q8_ef-reset`; a `q8_ef` snapshot is never restored into a
`q8` or `fp32` scope and the reverse. The cost is **+54 MiB per live owner**
on Flash-Next (36 layers × 1,572,864 B, the storage size of the residual, not a
measured cost), independent of the capture-ring rows R and with no growth of
the ring, mirror or any cache entry; each reset also writes that plane once.
Admission charges the live plane once per owner.

**q8_ef: sampled requests use AR until the sampled-MTP law is verified.**
Under `q8_ef`, temperature > 0 requests run AR instead of native MTP (the
native-MTP drafter of a `q8_ef` load has sampled verification off, the same
route `speculation.mtp_sampled = false` takes), and the load logs `qwen4 native
MTP sampled: off for GDN q8_ef (sampled requests use AR until the sampled-MTP
law is verified)`. Greedy native MTP is unchanged, and `q8`, `fp32` and the
default keep `speculation.mtp_sampled`. The rejection-sampling law under the EF
reset history has not been verified on a GPU.

**`q8_ef` measurements.** Decode KLD against the BF16 reference on code24
(24 chunks × 512, paired per chunk, 95% bootstrap CI, graph off):

| Card | Default | `q8` | F32 | `q8_ef` | `q8_ef` − `q8` | `q8_ef` − F32 |
|---|---|---:|---:|---:|---|---|
| Strix Halo (gfx1151) | F32 | 0.078394 | 0.071546 | 0.071987 | −0.006406 (−0.008538 to −0.004098) | +0.000441 (−0.001301 to +0.002306) |
| R9700 (gfx1201) | `q8` | 0.082302 | 0.074518 | 0.073598 | −0.008704 (−0.011854 to −0.005680) | −0.000919 (−0.004398 to +0.002255) |

The R9700 arms ran with `HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS=14` (routed experts
of layers 0–14 in VRAM, the rest host-mapped). The GDN state oracle has 0 hard
mismatches for `q8`, F32 and `q8_ef` on both cards, and each native-MTP route
(AR, batched, incremental, adaptive, full-head) gives identical IDs and live
state in two fresh processes (on Strix Halo the adaptive route's wall-clock
chooser decided differently between two runs, which is not required to
repeat). On gfx1201 the walked route plans cover the default `q8` state only:
a `q8_ef` load has no closed route plan (`gdn = q8_ef is outside the walked
["q8"]`) and compiles its kernels on first use. Timing for `q8_ef` has **not
been measured**, so nothing here is a performance claim.

**Numerical change.** F32 state has no per-token requantization. The gfx1151
default output is therefore **not bit-identical** to the previous Q8 default;
explicit `q8` is the previous route. Switching the format between loads changes
the session-cache scope (`Q8` vs `F32` in the state-format part of the key), and
a snapshot of one format is never restored into the other (the fixed-part
lengths differ, and the restore refuses a length mismatch).

Measured on Strix Halo (Flash-Next GPTQ3), F32 − Q8 decode KLD against the
BF16 reference, paired per chunk: −0.006848 (95% CI −0.009145 to −0.004703) on
code24 (24 chunks) and −0.011792 (95% CI −0.014177 to −0.009682) on a
WikiText-2 10-chunk subset (first 10 of 24 chunks × 2048 tokens); these and the
decode cost (−1.46% without speculation, +1.65% with native MTP, one process
per arm) come from gate binaries built from beta `c80b9e3e7`, before the
K1-C1/Cn gfx1151 decode GEMV and the WY GDN fix `0d06c380b`. On the release
16K prefill lane (6 interleaved fresh processes × 3 requests, beta `0d06c380b`
plus this change) F32 measures 2,389.98 tok/s and Q8 2,394.77 tok/s (paired
Q8 − F32 mean +2.5 tok/s, +0.10%). Details are in the CHANGELOG.

**Memory owners (storage sizes of the format, not measured costs).** For
Flash-Next (36 GDN layers, 48 value heads, 128×128): 811,008 B (Q8) or
3,145,728 B (F32) per layer, 27.84375 or 108 MiB per recurrent set, so each
live set is **+80.15625 MiB** under F32. The transaction (snapshot) mirror is
one more set: live + mirror 55.6875 → 216 MiB. Native MTP replaces the live
set by a ring of 2R slots (R = the actual capture-ring rows, logged at load,
not inferred from K) plus one separately allocated mirror, so ring + mirror
grows by 160.3125·R + 80.15625 MiB (R = 4: 250.59375 → 972 MiB; R = 8: Q8 ring
445.5 → F32 1728 MiB; R = 32: 1782 → 6912 MiB). Every session-cache or prefix
entry holds one fixed set: +80.15625 MiB per entry whatever its row count.
Conv state and the QSA K/V arenas do not change with the format. Lanes
multiply these owners, pipeline-parallel bands charge their own GDN layers, and
expert-parallel ranks each replicate the recurrent state (it is never divided
by the rank count). Admission and the placement reserve charge the F32 bytes
(live + mirror and the full capture ring); hipfire never lowers context, cache
budget or lane count to make F32 fit, and where a limit is already derived from
free memory (the unified-memory context cap) the F32 bytes enter that
arithmetic.

**Capture-ring growth (a bound, not charged).** The native-MTP reserve charges
the ring at the R a singleton request uses, R = max(K + 1, the GPU's row-exact
verify rows: 8 on gfx1151), and that request creates the ring at R on first
use, so with the defaults (K = 7 on gfx1151) the ring never grows. Only
experimental multislot (`serve.multi_slot`) can hand the resident state a
smaller ring: a promoted MTP lane's state carries a `k + 1`-row ring (lane
k = min(K, 7)), and a later singleton request on that state grows it to R when
K ≠ 7. Growth builds the new ring before it frees the old one (whose first
slot is the live state), so for that moment the recurrent owners exceed the
reserve by ring(R_old) minus one live set: at most 1,642.8 MiB (F32) or
440.4 MiB (Q8) for an 8-row old ring (K = 8–10), and 1,423.9 or 381.9 MiB for
a 7-row one (K = 6). Admission does not charge this transient, and the 2 GiB
Qwen4 admission headroom is sized for the first long request's other lazy
allocations (about 1.9 GiB measured on Strix Halo), so it does not cover it
either. A growth allocation that fails fails that request; hipfire does not
lower the context or fall back to Q8.

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

The schema default is `vmm`, but the CLI forwards a backend to the daemon only when one is authored (`--kv-backend`, per-model or global TOML); built-in and registry values are not sent. Omitted `memory.kv_backend` / `--kv-backend` is therefore **automatic**. Automatic
selects on-demand HIP VMM mapping on certified Qwen layout/device/topology
combinations (single-GPU and dense TP with per-rank VMM), covering static
modes, adaptive, and native `fp8`/`bf16`. Otherwise it falls back to
**legacy** once per load, logging the reason and the effective backend.
Absolute HFQ/safetensors paths and registry tags resolve identically for
the same source/mode/topology; registry cards set neither backend nor
`memory.max_seq`, but may still set `generation.max_tokens`. On
gfx1100/gfx1151 the certified Qwen3.5-family VMM KV pairs are Q8/Q8 and, on
exact gfx1100 TP1 only, FWHT2/3/4 K with Q8 or Lloyd2/3/4 V. Existing
pipeline-parallel VMM restrictions still apply. Unsupported combinations
retain the existing legacy fallback,
except on gfx1151 Qwen3.5-family loads (arch 5/6): a nonempty `kv_k` override
selecting FWHT2/3/4 (including typed aliases) with an unset/`auto`/`q8` mode
is refused unless `--kv-backend legacy` is explicit. There is no certified
gfx1151 VMM FWHT path, so this refusal also applies when VMM runtime or
platform support is unavailable. `kv_mode = "fwht3"` without a K override
retains the ordinary automatic legacy fallback; gfx1100 TP/Givens fallback,
gfx1201 selection and defaults are unchanged.

Qwen4 (Flash-Next, arch 16) owns its own QSA context layout. Automatic
selects **vmm** exactly on gfx1151 or gfx1201 with a certified platform (not Windows,
not WSL without its unsafe override) and a HIP VMM runtime; elsewhere it
selects legacy and logs the reason. Explicit `legacy` is always honoured, and
explicit `vmm` is refused only where that capability check fails. Under VMM,
the target and MTP QSA arenas (full K/V, raw and pooled index keys) and the
gathered-attention workspace reserve address space for `max_seq` and map
pages as forwards touch rows, before any capture or Redline record. Committed
context then tracks the tokens touched, not `max_seq`: the load logs
`qwen4 QSA context: … MiB virtual …, … MiB committed at load`, and each
growth logs `qwen4 QSA context mapped for N tokens`. Mapped pages stay owned
until unload, and VMM is not eviction.

On gfx1151 two exact QSA selection levers are on by default (kill switches in
[env-vars](env-vars.md)): each layer keeps a BF16 mirror of its pooled index
keys (`HIPFIRE_QWEN4_QSA_POOLED_BF16`; 256 bytes per pooled block, i.e.
16 MiB per QSA layer at 256K, charged in admission and never snapshotted),
and long-context score/select calls use the sampled filter select
(`HIPFIRE_QWEN4_QSA_FILTER_SELECT`). Selection, attention and logits are
byte-identical with either off.

Qwen4 `max_seq` on a **unified-memory APU** (gfx1151): omission is
**automatic** under VMM (legacy storage refuses it and needs an explicit
value). The context is the model's `max_position_embeddings` (at most
262,144), capped by **card capacity**: after every load allocation the carrier
measures free device memory and finds the largest context whose remaining VMM
growth (target and MTP QSA pages, gathered workspace), native MTP request
buffers and a 2 GiB headroom fit. The load logs one `qwen4 memory admission`
line with those numbers, and the daemon logs `max_seq: N (bound=model|card;
model_ctx=…, card_cap=…)`. An explicit `max_seq` that cannot grow refuses at
load, naming the largest one that fits (with `memory.oom_guard` off it loads
and only warns). Before any allocation, a fully resident load that does not
fit free device memory refuses the same way. The session cache is not
charged: its snapshots yield to context growth (see
`memory.session_cache_bytes`). On a discrete GPU an omitted Qwen4 `max_seq`
stays 32768 and an explicit one is not checked at load.

**Capping `max_seq` for 8K-class work on a slow-link 7900 XTX.** On a 7900 XTX
(gfx1100, 24 GB, PCIe x4) running Flash-Next, a smaller `max_seq` frees KV and
gather-workspace memory that expert placement turns into resident expert
layers. A load capped at `max_seq` 8448 admits 10 resident expert layers
instead of 6 at 262,144, and measured +5.4 to +5.7% 8K prefill (1,035 / 1,002
tok/s against 982 / 948, same-index cells, n=2). A 33,792 cap admitted 9
resident layers but no clean 32K gain (+0.7% prefill, 1,720 against 1,708
tok/s, noise-level). Each resident-layer count is its own numerical identity:
greedy token IDs diverge between counts (first at token 53 at 32K, 6 against 9
layers), so this is a placement choice, not an exact equivalence. Cap
`max_seq` for 8K-class workloads; leave it at the default for long context.

**Free device memory on unified memory.** Every free-memory sizing site
(this admission, expert placement, the dense Qwen card bound below, VMM KV
provisioning, prefill rungs, the session cache) reads `Gpu::device_mem_info`
(or `Gpu::device_mem_info_for` for another device; both apply the same clamp).
A discrete GPU, or an APU whose device pool is its BIOS carve-out, gets
`hipMemGetInfo` unchanged. An APU that allocates from GTT (device pool larger
than `mem_info_vram_total`, e.g. Strix Halo with a 1 GiB carve-out and a
120 GiB `ttm.pages_limit`) gets `min(hipMemGetInfo free, MemAvailable - CMA
+ TTM page-pool estimate)`: `hipMemGetInfo` there is TTM's cap minus live GTT
and ignores host RAM pressure, while freed GTT pages parked in TTM's pool are
outside `MemAvailable` but are what the next GTT allocation takes first.
The CMA region (the larger of `CmaFree` and the CMA pageblocks, `CmaTotal` or
the sum of `/proc/zoneinfo` `cma` lines; `MemAvailable` counts both free pages
and page cache in them, but they serve only movable allocations, never pinned
GTT or kernel memory) is subtracted, saturating at 0, even when `CmaTotal` is
0: kernels with Kexec HandOver (KHO) enabled by default release its scratch
memory (~13 GiB on a 128 GB Strix Halo) as CMA pageblocks outside any
registered CMA area. A missing `CmaFree`/`CmaTotal` counts as 0, unreadable
zoneinfo falls back to `CmaTotal`. Loading on a unified-memory GPU prints
`KHO scratch reserves N GiB unusable by the GPU; boot with kho=off to reclaim
it` once when the scratch is at least 1 GiB, and `hipfire diag` reports it
as a warning. The TTM pool addition is unchanged. Every host-RAM sizing site
(the OOM guard, host-mapped expert and offload placement, the PLE prewarm,
hot-row and load-fill budgets, the JIT worker count) reads the same
`MemAvailable - CMA` figure through `rdna_compute::uma_memory`
(`meminfo_available_bytes`, `mem_available_bytes`, `host_total_bytes` for
`MemTotal`). An unreadable `MemAvailable` fails the query (callers refuse
rather than size blind).

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
The carrier's cap is provisional (it also charges the DFlash/MTP draft reserve):
it sizes the VMM reservation and the draft structures before the drafts, DeltaNet
state and scratch exist. So the automatic context is **re-measured after the
whole load** (KV prefix, DeltaNet state, scratch, DFlash/MTP/DSpark, eviction):
the loader finds the largest context whose remaining KV mapping plus the
per-request reserve fits the VRAM actually free. The reserve is the minimum
prefill reserve (the minimum PBS, its 1 GiB headroom and 128 MiB of page
rounding), plus the owners a request creates lazily after the load: the
DeltaNet checkpoint rings (one snapshot per ring, for the AR ring and, with a
DFlash/MTP speculator, its own ring; further checkpoints up to
`HIPFIRE_CACHE_CKPT_MAX`, default 8, are taken only while free VRAM still
covers the admitted context's KV growth plus the reserve, else the ring recycles
its oldest snapshot) and, with
native MTP, the per-request `MtpSpecState` (head KV, scratch, verify buffers,
trunk snapshot and PBS) and the prompt-fill transient. The loader logs one
`max_seq fit: measured after full load: free=… transient=… (prefill … +
checkpoints … + mtp …) kv_mapped_growth_per_token≈… drafts=… ; context
A -> B` line, and lowers the published context (and `card_cap`) to it; below 512
the load is refused. An explicit `max_seq` that does not fit is refused at load,
naming the largest fitting value; with `memory.oom_guard` off it loads and only
warns.
Larger prefill PBS allocations are lazy: each request chooses, once per prompt,
the widest rung that fits currently free VRAM after mapped (not virtually
reserved) KV minus the KV the same prompt will still map (the AR, vision,
native-MTP and DFlash prompt paths deduct it; with no remaining growth the
choice is unchanged). A long prompt on a tight card therefore runs narrower
chunks instead of letting an early wide PBS take memory its own later KV needs.
When future KV growth needs the memory, widened PBS reuse is released between
requests, allowing later chunks to narrow rather than making growth OOM.
The loaded diagnostic reports `max_seq`, its `model`/`card`/`user` bound,
`model_ctx`, `card_cap` and KV mode. Qwen MoE uses the same automatic sizing on
unified-memory GPUs (gfx1151): its minimum prefill scratch, including the
shared-expert, expanded-expert and grouped-GEMM buffers, is charged through
`PrefillBatchScratch::projected_allocation_bytes`, so `qwen3.6-35b-a3b` and
`ornith-1.5-35b-a3b` load at their 262144-token trained window on Strix Halo.
On a discrete GPU Qwen MoE keeps its prior 32768 built-in default (with a loaded
diagnostic reason). Other carriers keep their existing context behavior until
their different cache ownership can use the same allocation accounting.

A legacy `config.json` that carries `max_seq: 32768` (the old TypeScript CLI's
built-in default) loads and migrates without it, with a warning, so the
migrated `config.toml` does not pin carriers that size their own context. A
`config.toml` written by an earlier migration keeps its explicit value; remove it
with `hipfire config reset memory.max_seq`.

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
| `yarn_factor` | `null` | `null` or number 1.0–64.0 |

Lowered directly into the daemon snapshot. `HIPFIRE_ATTN_FLASH` remains a
legacy one-shot alias. The Qwen3.5 MTP head inherits the trunk
`flash_mode` / `attention_flash_mode` (it does not pin a separate non-flash path).

`yarn_factor` (`attention.yarn_factor`, env `HIPFIRE_YARN_FACTOR`) overrides
the checkpoint's RoPE scaling at load. `null` uses the checkpoint's own
`rope_parameters` / `rope_scaling` (`rope_type: "yarn"` is honored; any
`rope_type` other than `default` or `yarn` fails the load); a value above 1
applies YaRN with that factor over the checkpoint's
`original_max_position_embeddings` (else `max_position_embeddings`), which
extends the admitted context to `factor x original`; `1` disables a
checkpoint's YaRN. An override ignores the checkpoint's declared
`attention_factor`, which was tuned for the declared factor. The effect equals
editing the model card's `rope_parameters`: for Qwen3.8-Flash-Next,
`yarn_factor = 4` admits `max_seq` up to 1,048,576 (the Qwen4 maximum
context), subject to what card memory can hold. Static YaRN also changes
outputs at short context, so set it only for long-context loads.
Implemented by the engine (`rdna_compute::rope`, `kernels/src/rope_yarn.h`);
the value reaches the engine through the versioned `ProcessConfig` into
`RuntimeConfig::yarn_factor`. It is currently wired for Qwen4 (Flash-Next)
and for the CLI's declared-context calculation. Other families ignore it.

---

## Speculative decode

Only **one** mechanism runs. Canonical selector:

| Key | Default | Values |
|---|---|---|
| `speculation` | `"auto"` | `off` \| `auto` \| `ngram` \| `dflash` \| `mtp` \| `dspark` |

- **`off`** — AR only.
- **`auto`** — cascade by availability / eligibility; legacy mode knobs filter each mechanism. Under auto, DSpark can win over in-trunk MTP when a DSpark sidecar is present (CLI comments). Native MTP in the cascade arms only for a producer that declares a measured AR floor (today only Qwen4; see `mtp_mode`); every other producer's head needs `--spec mtp` / `speculation.mtp = "on"`.
- **Forced mechanisms require their own prerequisites.** `--spec dflash` / `speculation.mode=dflash` forces `speculation.dflash=on` and refuses the load without a DFlash draft; it never aliases to native MTP. Use `--spec mtp` / `speculation.mode=mtp` or `speculation.mtp=on` to request native MTP explicitly, subject to the safety refusal in the `mtp_mode` row.

Legacy one-shot alias: `HIPFIRE_SPECULATION`. CLI: `--spec`.

### Mechanism knobs

| Key | Default | Values / range | Notes |
|---|---|---|---|
| `dflash_mode` | `"off"` | `on` \| `off` \| `auto` | **Schema default off; one registry-scoped exception.** `auto` enables on dense Qwen3.5-class targets and skips known-loss A3B cases. The registry layer (below the global / per-model / one-shot config, above the schema default) lowers `speculation.dflash = "auto"` for the Qwen3.8-27B trunk tiers that declare a paired `dflash` draft (`qwen3.8:27b`, `qwen3.8:27b-mq3*`, `qwen3.8:27b-mq4*`; not the `-draft-*` sidecar tags, not tiers without a `dflash` declaration). See [Qwen3.8-27B paired DFlash default](#qwen38-27b-paired-dflash-default). **Non-WMMA GPUs (gfx1030 and other pre-RDNA3 arches):** a DFlash draft over an MQ V2 / MQ3 output layer (e.g. MQ4G256V2, qt=44) is refused at load admission (#771) because DFlash's batched lm_head admission for those formats is validated only on the gfx11/gfx12 WMMA set (a generic non-WMMA x-batch scalar kernel exists but is not validated for DFlash verify); the refusal names the GPU and the format and says to set `dflash_mode off` (or `--spec off`). Autoregressive decode and Q8/HFQ4/MQ4G256 output layers are unaffected. `on` requires a DFlash draft and fails the load without one, even when an MTP head is present. It never enables MTP; use `--spec mtp` / `speculation.mtp=on` instead. |
| `vision_mode` | `"auto"` | `auto` \| `resident` \| `on` \| `off` | **Default auto** (was `off` before 0.4.4). Tower sidecar gate and residency — see [Vision tower](#vision-tower). |
| `dflash_adaptive_b` | `false` | bool | **Opt-in.** Adaptive verify-block width: follows the trailing 8-cycle acceptance depth (τ̂+2), full below 2k context. Not output-identical (window boundaries move; per-position sampling stays target-lossless). Auto-suppressed on retained-PM4 verify loads. `HIPFIRE_DFLASH_ADAPTIVE_B=0` is a kill switch read through the `[developer]` namespace (not a schema env alias): it forces the fixed block even when this is `true`, and cannot enable adaptive. |
| `dflash_block` | `"auto"` | `"auto"` \| int 4–16 | DFlash2 proposal width L (`speculation.dflash_block`, env `HIPFIRE_DFLASH_BLOCK`). Chosen **per serving configuration, never per live batch**: every request of a load (singleton or batched VMM lane) runs the same L, so batched output stays byte-identical to the singleton and text never depends on server load. **Default `auto`** = 8 when `serve.vmm_batch` is on with `continuous_batch_size` ≥ 4 (a full C8 verify step stays at 64 rows instead of 128), else 16. A lone request on a batching serve runs slower at 8 than at 16. `16` (or `HIPFIRE_DFLASH_BLOCK=16`) restores the previous fixed width. Greedy text depends on L (different window boundaries). Legacy DFlash drafts always run their declared width. Retained-PM4 verify and dense TP2 need 16. The experimental multi-slot route admits only 16 (`auto` resolves to 16 there). |
| `dflash_lookup` | `"auto"` | `on` \| `off` \| `auto` (also `1` / `0`) | Dense Qwen3.5-class DFlash2 verbatim-copy lookup composed into the existing singleton verify window (`speculation.dflash_lookup`, env `HIPFIRE_DFLASH_LOOKUP`; env beats TOML beats default). A **Copy** window verifies lookup candidates in place of the learned candidates; a **Tail** window keeps the first K learned candidates and appends a lookup tail (K+L ≤ 15, ≤ 16 verify rows, never above the load's block width). The lookup is a request-local `MtpCopyContext` (triple 5/1/63, never indexes speculative tokens) and a window is taken only when the request's own measured acceptance per measured window price beats the incumbent window; a request with no lookup proposal runs the incumbent window byte-for-byte. **`auto` (default) and `on`/`1` arm it** only on a singleton DFlash2 selector-chain load (no DDTree, no retained-PM4 verify, no VMM/CB lane, no dense-TP mesh, no CACTUS) and log `[dflash] lookup: armed (…)` or `[dflash] lookup: off (<reason>)` at load; `off`/`0` disables it. R9700: copy-heavy decode +39.8%; non-copy requests take no lookup windows and run the incumbent window (CHANGELOG 0.4.3). Lookup rows are verified with point-mass draft probability (sampled windows accept with p(candidate)); not a new greedy-AR identity claim. Developer `HIPFIRE_SPEC_LOOKUP_TAIL=0` disables only the Tail windows. |
| `dflash_ngram_block` | `"auto"` | `true` \| `false` \| `"auto"` | Verify-path n-gram defense; auto size-gates. |
| `mtp_mode` | `"auto"` | `off` \| `on` \| `auto` | Built-in MTP when a head is present: the DeepSeek V4 trunk's MTP layer, or for Qwen a bundled `.mq4-mtp` trailer or a `.mtp` sidecar (the registry `mtp` slot; Qwen3.8-27B ships one — [MODELS.md](MODELS.md#dflash-draft-artifacts-registry)). **`auto` arms native MTP only for a producer that declares a measured AR floor** (a per-request AR-vs-draft pricing that retires drafting when it loses to AR). Today only Qwen4 (Flash-Next, arch 16) declares one. Qwen3.5-class (arch 5/6, dense and MoE), DeepSeek V4 (its native MTP was audited and has no measured floor) and Gemma 4 declare none, so `auto` stays AR there even with a head present, loads no head and logs `MTP: off by default (no measured AR floor); opt in with --spec mtp`. `on` / `--spec mtp` is the explicit opt-in and still arms any present head on every producer, failing the load without one; `off` never arms. DSpark and DFlash selection is separate and unchanged. A producer that does arm under `auto` still declines over a host-mapped routed-expert spill (Qwen4 attaches over a spill only on its admitted arch), on either `memory.offload_exec=pcie` or `cpu`, logging the count, seam and `opt in with --spec mtp`. `on` / `--spec mtp` opts in, except that uniform MQ6G256 V1 (qt15) Qwen35 MoE with FP16 WMMA enabled on gfx1200/gfx1201 is refused before serving/generation whenever a native MTP head would attach (now only an explicit `on`, since Qwen35 `auto` no longer attaches). The guard uses tensor metadata, not filenames. `--spec off` / `speculation.mtp=off` or an auto-declined spill stays AR; `HIPFIRE_FP16=0` permits explicit MTP. Mixed/graded MQ4+MQ6, MQ6V2, gfx1151 and dense models are outside this refusal. Other explicit attachment behavior is unchanged; this is admission protection, not a fix for the underlying WMMA defect. |
| `mtp_k` | `"auto"` | `"auto"` \| int 1–63 | MTP draft cap (`speculation.mtp_k`, env `HIPFIRE_MTP_K`). **Default `auto`** for native Qwen4 MTP (Flash-Next, arch 16) = `min(GPU row-exact verify rows, window of the CPU routed-expert mode that owns layers) − 1` (capability rule: the cap is derived from the GPU's row-exact verify rows, not an architecture list), bounded by executor capacity 63; else 3 for every other model. With no CPU-owned experts that is 7 on every GPU today (8 rows); a load whose routed experts run on the CPU (D1 splice, or any D2 tier split including the measured per-load default; 4-row window) resolves 3 after placement. **On current beta a host-offloaded Flash-Next load whose CPU beats the host link arms the D2 measured default, so its effective `auto` cap is 3, not 7** (diagnostic j1408, R9700 gfx1201: CPU 153.9 GB/s versus link 26.2 GB/s, D2 on 48/48 tier layers, native MTP K=3, receipt `/home/kaden/qcal/release-0.4.3/mtp-kcap/run-r9700-selector/`); 7 applies where D2 does not own experts (`memory.offload_exec=pcie`, `HIPFIRE_QWEN4_D2_DEFAULT=0`, a link the CPU does not beat, or a fully resident load). `hipfire config` cannot see placement, so it shows 7 for arch 16 and the load-time `auto` value can be lower. The adaptive chooser picks each window's depth up to the cap, starting a request with no measured history at `min(3, cap)` and growing to the cap on evidence. Before the AR floor retires a request whose priced native options are all deeper than 3, it first re-probes shallow (K ≤ 3) native depths with bounded windows (3 priced windows, 6 in all) and retires only if those also lose to AR; otherwise the request keeps its best measured shallow depth and climbs again only on 3 consecutive better-priced windows (no setting for this). An explicit 1–63 is accepted by the shared schema; native Qwen4 carries it separately from `auto`, while qualified-row and CPU-window hard caps still apply and other producers retain their own existing limits. The incumbent floor's runtime depth limit remains 10. The developer-only `HIPFIRE_QWEN4_MTP_DERIVED_K=1` policy (default off) may select beyond the incumbent/default K, bounded only by an explicit user K if any, the dynamic qualified bundle row limit minus the seed (including CPU ownership), executor 63 and remaining budget; see [`env-vars.md`](env-vars.md). `4` restores the previous gfx1151 cap. Evidence and limits: one R9700 (gfx1201) hold on the PCIe / no-D2 route (base `0a3b67cb1`, before the D2 default existed) measured code +10.17% and prose +0.30% decode tok/s over the previous K3 with AR-identical output (receipts under `/home/kaden/qcal/release-0.4.3/mtp-kcap/`, see the changelog). That gain is conditional on that route and is not a claim about the current default, whose D2 route runs K3. That hold exercised neither a CPU-owned (K3) load nor the shallow re-probe/hold states on hardware; both are covered by CPU tests only, and gfx1151 and gfx1100 were not measured. |
| `mtp_sampled` | `true` | bool | Default true. Native Qwen4 MTP uses distribution-preserving speculative rejection sampling for temperature >0, including supported repeat/presence/frequency penalties: target verification and draft probabilities use the same AR history, penalty window and sampling policy. False/`HIPFIRE_MTP_SAMPLED=0` leaves sampled requests on AR; greedy MTP is unchanged. Under `topology.pp` (Flash-Next pipeline mesh) the same Leviathan verify serves sampled requests with neutral penalties; penalized sampled requests, `HIPFIRE_MTP_SAMPLED_MODE=naive` and this key off decode AR on the mesh. Under `memory.state_quant = "q8_ef"` sampled requests use AR until the sampled-MTP law is verified, whatever this key says (see [`q8_ef`](#kv-cache)). This is distribution preservation, not identical seeded sampled token streams. DFlash/other drafters without this policy still fall back to AR for unsupported penalty requests. |
| `mtp_ngram` | `"auto"` | `on` \| `off` \| `auto` (also `1` / `0`) | **Default `auto` (0.4.1).** N-gram takeover windows inside native MTP (ngram-mod). `auto` arms native Qwen4 MTP (arch 16, Flash-Next) on gfx1151 and gfx1201 and leaves every other producer off; `on`/`1` arms every MTP producer that supports it; `off`/`0` is the opt-out (kill switch). Env `HIPFIRE_MTP_NGRAM` beats TOML beats default. Eligible requests are greedy and sampled, thinking on or off (Qwen4 and Qwen3.5/3.x). A sampled takeover verifies each candidate with the point-mass draft distribution δ(candidate) through the shared rejection verify: lossless in distribution, not bit-identical to MTP-only output. The pool is request-local (cleared and reseeded from the full prompt every request), so candidates never depend on earlier requests. Triple (`HIPFIRE_NGRAM_MOD_N_MATCH` / `_N_MIN` / `_N_MAX`, overridable per component): 5/3/3 on admitted Qwen4 targets with the copy lane off (5/1/63 with it on), 24/48/64 elsewhere; on Qwen4 `n_max` is capped at 63 (the 64-row verify). On Qwen4 a pool hit takes the window only when the MTP head's own first draft equals the first candidate and the takeover's measured yield (decayed emitted tokens per takeover window over its verify cost) is at least the native window's best expected rate; otherwise the window is native. The Qwen4 takeover uses the same batched verify executor as native MTP windows (interleaved one-row route under `HIPFIRE_MTP_INCREMENTAL=1`) and appends the consumed rows to the MTP head teacher-forced (token p, target hidden p - 1), matching the draft steps' pairing. Load marker: `qwen4 native MTP n-gram: on\|off (n_match=.. n_min=.. n_max=..)`; `HIPFIRE_MTP_TRACE=1` window events carry `source` / `verify_route`. The 5/3/3 pool described here is the copy lane's `off` state; `speculation.mtp_ngram_copy` / `HIPFIRE_MTP_NGRAM_COPY` (`on` arms the 5/1/63 copy lane; default `auto` arms it on gfx1151 / gfx1201) switches it. |
| `mtp_ngram_copy` | `"auto"` | `on` \| `off` \| `auto` (also `1` / `0`) | Native Qwen4 MTP n-gram copy lane (verbatim-copy matcher, adaptive take of up to the row-exact verify rows minus one, row-0 confirmation only until the copy source is established, kept on floor-retired requests). on/1 arms it wherever the native Qwen4 n-gram source is armed (triple 5/1/63); off/0 keeps the 5/3/3 pool with row-0 confirmation on every takeover (kill switch); auto (default) arms it for native Qwen4 MTP (arch 16) on gfx1151 and gfx1201 and leaves every other GPU off. Env `HIPFIRE_MTP_NGRAM_COPY` beats TOML beats default. |
| `dspark_conf_threshold` | `null` | `null` or number 0.0–1.0 | `null` ⇒ per-arch carrier default (qwen3 0.1 / deepseek4 0.3 in comments). |
| `ngram_mode` | `"off"` | `off` \| `on` \| `auto` (also `1` / `0`) | Model-free; byte-identical to AR when used. |
| `ngram_k` | `12` | int 2–32 | |
| `ngram_min_count` | `2` | int 1–10 | |
| `ddtree_budget` | `0` | int 0–64 | `0` = chain DFlash (no tree). The runtime flag reader treats `0` as unset, so a call site falls back to its own default (the dense path uses 8 nodes when `speculation.dflash_tree` is on; the Qwen3.5 path stays chain-only). |
| `ddtree_topk` | `4` | int 1–8 | |

**DFlash IU4 verify at `dflash_block` 8.** When a DFlash load's proposal width resolves to 8 (the `auto` width of a batching serve), its target verify runs on gfx1201 with 4-bit (A4) activations: the IU4 verify. **Numerics differ from the F16 verify.** F16 == IU4 is not promised; batched == singleton stays byte-identical within the configuration. A lone request on that serve runs slower than at width 16. Width-16 loads keep the F16 verify. `HIPFIRE_CB_VERIFY_IU4=0` is the kill switch, `=1` forces it on at any width; `HIPFIRE_CB_VERIFY_FP8=1` takes precedence over the default and cannot be combined with `=1` ([`env-vars.md`](env-vars.md)).

Legacy compatibility input still wins at the top of the startup ladder for
the corresponding knobs, but the engine receives only the resolved immutable
snapshot. Full aliases: [`env-vars.md`](env-vars.md).

### Qwen3.8-27B paired DFlash default

The built-in `speculation.dflash` default stays `off` and `speculation.mtp` stays `auto` (`auto` arms MTP only for a producer with a declared measured AR floor, so today only Qwen4; the registry layer never changes this). The registry config layer is the only place that changes this, and only for Qwen3.8-27B trunk tags (`qwen3.8:27b` and `qwen3.8:27b-*`, never a longer lookalike) whose registry entry declares a `dflash` sidecar; it writes `speculation.dflash = "auto"`. Explicit global, per-model and one-shot config rank above the registry layer.

- **Paired draft file present** (the registry-declared `dflash.file` next to the model, or a `developer.dflash_draft` / `HIPFIRE_DFLASH_DRAFT` / `run --model-draft` path): the load wires it and DFlash runs.
- **Paired draft file absent:** nothing fails and nothing is wired. The load keeps MTP `auto` (the registry `mtp` head); `auto` arms native MTP only for a producer that declares a measured AR floor, which Qwen35 does not, so the 27B then runs AR unless `--spec mtp` / `speculation.mtp = "on"` opts in (see [`mtp_mode`](#speculative-decode)).
- **Kill switch:** an explicit `dflash_mode off` wins over the registry layer: `hipfire config qwen3.8:27b set dflash_mode off` (per model), `hipfire config set dflash_mode off` (global), or `HIPFIRE_DFLASH_MODE=off`. The canonical key is `speculation.dflash = "off"`.
- **Unchanged:** every other model keeps the schema `off` (Qwen3.5/3.6, Flash-Next / `arch_id=16`, the `-draft-*` and sidecar tags, and 27B tiers with no `dflash` declaration such as `mq5*` / `mq6*`). `dflash_block` stays `auto`: 16 on a singleton load, 8 on a `serve.vmm_batch` serve with `continuous_batch_size` ≥ 4, and the IU4 verify gates ([above](#mechanism-knobs)) are untouched.
- **Output:** DFlash greedy decode is not promised byte-identical to AR (or to MTP): near an argmax tie the multi-row verify can pick the other token, and the text then diverges from that point. Batched output stays byte-identical to the singleton within a configuration.

Acceptance evidence for this default (fresh-process determinism, the batching C1 == C8 state oracle, MTP-vs-DFlash task-eval quality, the serve battery and chain text) is collected outside the repository; no pass is recorded here until a public, dated receipt exists.

---
## Vision tower

| Key | Default | Values / range |
|---|---|---|
| `vision_mode` | `"auto"` | `auto` \| `resident` \| `on` \| `off` |

- **`off`** — never load a tower sidecar. The registry/sibling file is not wired, and even an explicit `run --vision` / `serve --vision` / `HIPFIRE_VISION_SIDECAR` path is skipped with one stderr line, and an image request is refused with an error that names this knob. The daemon enforces the same hard override for non-CLI clients, mirroring `dflash_mode=off`. Loads pay no tower VRAM. Use it to keep the tower's VRAM for KV on a tight card.
- **`auto`** (default since 0.4.4; it was `off`) — use the registry `vision` slot (or the `<trunk-stem>-vision.hfq` sibling beside the trunk) when present; silently text-only when absent (no download, no message). For the Qwen3.5/3.8-VL tower the tower is loaded on demand from host/disk and streamed per block through two small VRAM staging buffers during each image encode, with byte-identical math: no resident tower VRAM. Measured on `qwen3.8-27b.mq4-xts` + `qwen3.8-27b-vision.hfq` at idle (`max_seq` 32768): +127.6 MiB over `off` on an R9700 and +142.2 MiB on a 7900 XTX (2 x 45.0 MiB staging, 4.9 MiB small resident tensors, allocator overhead), against +958 MiB for `resident`. Every tower load (`auto`, `on`, `resident`) also leaves the activation scratch of the largest admitted image out of the automatic `max_seq` (327 MiB at the default 2 MP pixel budget; `vision_reserve=` in the `max_seq budget:` log line): on a 24 GB XTX the automatic `max_seq` is 223,323 for `off`, 210,453 for `auto` and 184,612 for `resident`. The container pages are pulled into the page cache in the background at load, so the first image reads RAM; the cost over `resident` at the HTTP level (1024x1024 image, median of fresh processes) is about +3-9 ms on an R9700, +16 ms on Strix Halo and +36 ms on a 7900 XTX behind a PCIe 4.0 x4 link (+68 ms there for the smaller 692x1024 `scene_1.jpg`, where the copy is no longer hidden under tower compute). When the page cache was dropped after the load the first image reads the tower from disk: +30-46 ms on the R9700 host, +250 ms on Strix Halo, +510 ms on the XTX host; the next image is back to the numbers above. `hipfire bench` pins `off` (it has no media input).
- **`resident`** — same sidecar discovery and silent text-only fallback as `auto`, but the whole tower is uploaded into VRAM at load (~0.94 GiB for the 27B tower). It gives the lowest first-image latency at the cost of that VRAM for the life of the load.
- **`on`** — like `auto` (streamed), but require the declared sidecar: the load fails closed with a pull hint when it cannot be resolved. A trunk with an embedded tower declares no sidecar and is unaffected by this key.
- **Scope** — only the Qwen3.5/3.8-VL tower streams. EmbeddingGemma 2, qwen35-embed, LFM2-VL and dots towers load as before for every non-`off` mode (`resident` behaves like `auto` there).
- **EmbeddingGemma 2 (arch 18, `hipfire embed`)** reads the same ladder against its own discovery: the quantizer's `<trunk>.vl` pack beside the checkpoint, which carries **every** bundled tower group (`vision_tower.*`, `audio_tower.*`, `embed_vision.*`, `embed_audio.*`) — one pack, one decision. Under `auto` (the default), `resident` or `on` the pack is loaded and image/audio/video input is served on the embed verb; under `off` the load is text-only and any input carrying media is refused per modality, naming the knob. The `serve.multi_slot` slot route applies the same ladder (its engine is told the resolved tower path, never left to rediscover one).

```bash
hipfire config set vision_mode off                # never load the tower (pre-0.4.4 behaviour)
hipfire config qwen3.8:27b set vision_mode off    # per-model overlay
HIPFIRE_VISION_MODE=off hipfire run qwen3.8:27b   # one-shot
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
| `prefill_drafter_kv` | `"q8"` | `q8` \| `fwht4` \| `fwht3` \| `fwht2` (`speculation.prefill.drafter_kv`, env `HIPFIRE_PFLASH_DRAFTER_KV`) |
| `diagnostic.pflash.score_layer` | `null` | `null` or int 0–65535 (env `HIPFIRE_PFLASH_SCORE_LAYER`) |

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
| `serve.multi_slot` | `false` | Experimental opt-in: serve concurrent requests on the multi-slot engine instead of one at a time. Per-site KV policy: `memory.kv_cache = "auto"` or unset stays `q8` on every GPU. Explicit modes admit `q8`, rotated `fwht2`/`fwht3`/`fwht4` (`asymN`/`turboN` aliases; bare `turbo` = `fwht3`), legacy `legacy-asym2`/`legacy-asym3`/`legacy-asym4`, native `bf16`/`f16`, and `fp8` only on exact gfx1201 (refused elsewhere). `--kv-k`/`--kv-v` splits are not supported by slots. Concurrent requests can produce different greedy text than serial requests at ≥4 slots; output is deterministic for a fixed batch composition. 2 slots matched serial in earlier testing; on 35B-A3B MoE checkpoints, 2 slots also diverged from serial on 1–4 of 4 prompts. A pre-warm that fails (for example, slot-engine allocations that do not fit beside another process on the card) exits serve with the load error rather than serving lazily ([`SERVE.md`](SERVE.md) § Lifecycle). |
| `serve.multi_slot_slots` | `4` | int 1–64 concurrent slots. |
| `serve.multi_slot_ctx` | `8192` | int 512–1048576 per-slot context capacity (tokens). |
| `serve.multi_slot_prefill_chunk` | `1024` | int 1–1048576. Prefill tokens taken from one slot per multi-slot step; batch scratch is sized `n_slots ×` this. Env: `HIPFIRE_SERVE_MULTI_SLOT_PREFILL_CHUNK`. |
| `serve.swap_host_bytes` | `17179869184` (16 GiB) | int 0–1TiB. Multi-slot route: host-RAM ceiling for parked swap snapshots. `0` disables the host tier so snapshots spill to disk. This key does not govern host-pinned weight admission. |
| `serve.swap_disk_bytes` | `8589934592` (8 GiB) | int 0–1TiB. Multi-slot route: disk ceiling for parked swap snapshots. `0` disables disk spill. A snapshot larger than the ceiling is refused without touching the cached snapshots; otherwise the least-recently-used snapshot is evicted and its session re-prefills on the next turn. |
| `serve.swap_dir` | `""` | Path or empty. Multi-slot route: swap spill directory; empty uses `$XDG_RUNTIME_DIR/hipfire/swap/<pid>`, falling back to `$TMPDIR/hipfire-swap-<pid>`. Created `0700`, must be owned by this user, is locked to one daemon, and may not be a symlink; an existing owned directory has group/other access removed. Stale `*.snap`/`*.snap.tmp` files are removed at start; other files are left untouched. |
| `serve.multi_slot_session_ttl_secs` | `3600` | int 0–86400. Multi-slot route: close a parked (swapped or cold) session after this many seconds idle; `0` disables the idle sweep (the hard cap still applies). The engine wakes at most every 30 s while idle when this is non-zero. |
| `serve.multi_slot_max_sessions` | `0` | int 0–4096. Multi-slot route: hard cap on session-table entries; `0` = auto `max(64, 16 × serve.multi_slot_slots)`. Only parked sessions are closed to enforce it. |

Multi-slot parked sessions are swept after `serve.multi_slot_session_ttl_secs`
idle and capped at `serve.multi_slot_max_sessions` non-resident entries
(oldest first). Closing one only forces a cold re-prefill on its next turn;
resident sessions are never swept.

Serve HTTP surface: [`SERVE.md`](SERVE.md). The corresponding `HIPFIRE_MODEL`,
`HIPFIRE_IDLE_TIMEOUT`, `HIPFIRE_MAX_REQUEST_BYTES`,
`HIPFIRE_SERVE_MAX_QUEUE`, `HIPFIRE_SERVE_QUEUE_TIMEOUT_MS`,
`HIPFIRE_SERVE_MULTI_SLOT*`, and `HIPFIRE_SWAP_*` names are legacy one-shot
aliases.

---

## Memory

### `memory.oom_guard`

| Key | Default | Values |
|---|---|---|
| `memory.oom_guard` | `auto` | `auto` \| `true`/`on`/`1` \| `false`/`off`/`0` |

Compat env: `HIPFIRE_OOM_GUARD`. Used by `kv_slots::preflight_alloc`, the
`SlotPool` arena check, and the CLI bench-sweep headroom path.

Three checks exist; only two are gated:

- **Host available-memory headroom** (`MemAvailable` minus the CMA region) — gated
  by `memory.oom_guard`. Default
  `auto` turns it **on** for unified-memory APU arches (`gfx1035` / `gfx1036` /
  `gfx1103` / `gfx1150`–`gfx1152`: GPU allocations come from system RAM, so an
  overshoot is a desktop-killing OOM), **off** for discrete GPUs (overshoot is
  a failed `hipMalloc`), and for GPU-less processes by host swap state (no
  swap → on). Explicit `true`/`false` force either way; `auto` logs its
  decision once. The CMA region (larger of `CmaFree` and the CMA pageblocks,
  `CmaTotal` or `/proc/zoneinfo` `cma`) is subtracted because only movable
  allocations can use it, not pinned GTT or kernel memory. In a unified-memory
  process the room it checks is `MemAvailable - CMA` plus TTM's page-pool
  estimate (freed GTT pages the next GTT allocation takes first; this
  addition is unchanged).
- **R9700 deployment-target VRAM budget** (32 GiB class ceiling in
  `preflight_alloc`) — **always runs**, on every arch, whether the host
  headroom guard is active or not. A configuration that does not fit the
  deployment target is refused regardless of this box's RAM.
- **VRAM fit of an explicit single-card Qwen VMM `max_seq`** — after the whole
  load (weights, drafts, scratch) the loader measures the largest context that
  fits free VRAM with the per-request reserve (see the dense Qwen automatic
  context above); an explicit `max_seq` above
  it is refused at load naming that value. Gated only by an explicit `off`
  (`false`/`0`), which prints the same message as a warning
  (`memory.oom_guard off: loading anyway`) and keeps the requested context;
  `auto` and `on` both refuse.

`scripts/run-bounded.sh` (`HIPFIRE_MEM_CAP`) remains the hard cgroup backstop.

#### Windows host memory and `hipfire diag`

- **Host memory.** `hipfire diag --json` carries an additive `host_memory`
  object (`null` when the platform has no host-memory source). On Linux it
  is parsed from `/proc/meminfo`. On Windows it comes from
  `GlobalMemoryStatusEx`: `available_physical_bytes` is *physical* RAM
  (`ullAvailPhys`), whereas the commit figures (`commit_limit_bytes` /
  `commit_available_bytes`, from `ullTotalPageFile` / `ullAvailPageFile`)
  count RAM plus pagefile. `swap_total_bytes` on Windows is the **pagefile
  approximation** `commit limit − total physical`, saturating at 0; it is not a
  direct pagefile-size query. The bench-sweep headroom check compares the
  model against available *physical* memory on every platform.
- **GPU memory budget.** On Windows only, `hipfire diag` adds a
  `gpu_memory_budget` JSON key and a text section *GPU memory budget (DXGI,
  this process)*: per hardware adapter, identity (vendor/device id, LUID, AMD
  vs non-AMD), dedicated/shared sizes, and for the local and non-local
  segments the `Budget`, `CurrentUsage`, `AvailableForReservation` and
  `CurrentReservation` from `IDXGIAdapter3::QueryVideoMemoryInfo` (node 0).
  A failed query is reported as `error`, not as an empty list. Software
  adapters (`DXGI_ADAPTER_FLAG_SOFTWARE`) are skipped.
- **Not an admission budget.** These numbers are what the *calling process*
  (the CLI) sees on node 0, at one instant. They are diagnostics. Admission of
  a load must query DXGI from inside the engine process itself.
- **GPU-less `auto` on Windows.** With no GPU arch known, `auto` decides by
  host swap state. On Windows that state is the pagefile approximation above:
  approximation `> 0` counts as swap and **disables** the guard; `0` or an
  unavailable query **enables** it. The arch overrides apply unchanged
  (unified APU → on, discrete → off).

### Spill warning

On Linux with amdgpu, `hipfire serve`, `hipfire run` and `hipfire bench`
(standard path) watch for VRAM spilling into system RAM (GTT), which keeps
the model running but 2-3x slower with no error. The card is the one the
bench's `power_device` report names (`hardware.devices`, else
`ROCR_VISIBLE_DEVICES`, else the first ROCr agent of the daemon's arch). A
baseline of `mem_info_gtt_used` is taken right after the model loads; after
each finished request (each generation for `run`/`bench`) the monitor warns
once on stderr when GTT grew by at least 256 MiB over it while
`mem_info_vram_used` is within 512 MiB of `mem_info_vram_total`:

```
WARNING: GPU memory is spilling to system RAM: GTT +N MiB since load while VRAM is full (U/T MiB). Decode/prefill can be 2-3x slower. Lower memory.max_seq, use a smaller KV mode (kv_cache), a smaller tier, or turn speculation off.
```

It repeats only after GTT grows another 256 MiB. `hipfire bench --json` adds a
per-run `spill` array of objects (`baseline_gtt`, `gtt`, `vram_used`,
`vram_total` in bytes, and `spilled`) when the card is known. Unreadable
counters (no amdgpu, no permission) disable the monitor silently; it is a
no-op off Linux.

### Partial GPU offload (`memory.gpu_layer_budget` / `memory.moe_expert_budget` / `memory.offload_exec`)

| Key | Default | Values |
|---|---|---|
| `memory.gpu_layer_budget` | unset/null (fully resident on every model) | int 0–65536 \| `auto`/`-1` |
| `memory.moe_expert_budget` | unset/null (fully resident on every model) | int 0–65536 \| `auto`/`-1` |
| `memory.offload_exec` | unset (`pcie`, except the measured Flash-Next D2 default below) | `pcie` \| `cpu` \| `passback` |
| `memory.offload_passback_share` | `auto` | `auto` \| number in `(0, 0.5]` \| `0` |

Compat env: `HIPFIRE_GPU_LAYER_BUDGET`, `HIPFIRE_MOE_EXPERT_BUDGET`, `HIPFIRE_OFFLOAD_EXEC`, `HIPFIRE_OFFLOAD_PASSBACK_SHARE`. All four are `ModelLoad`-scoped (`hipfire config set memory.*`, per-model overlay, or env); there is no `--offload` CLI flag.

- **Counting.** Both budgets count layers kept **on** the GPU, not offloaded — `3` on a 64-layer model spills 61. `gpu_layer_budget` spills whole-layer prefixes (`[0 .. n-N)`); `moe_expert_budget` spills only routed experts. On an expert-only spill, attention, router and shared expert stay resident; embedding, `lm_head` and KV stay resident in either case.
- **Explicit auto-fit.** `auto`/`-1` opts into fit-based placement: the smallest spill that loads, **routed experts first**, whole layers only when the expert tier cannot free enough; a model that already fits stays fully resident. Unset/null budgets leave weights device-resident on every model, including Qwen MoE; there is no implicit spill. Manual `N` budgets are hard pins and are never silently widened — a pinned placement that does not fit refuses the load naming the knob.
- **Scope.** Explicit auto-fit runs only on spillable single-GPU text-trunk loads (`pp == 1`, no paged experts, no REAP keep-map); unset stays resident everywhere. Offload is implemented for Qwen3.5 dense/MoE (arch 5/6) and LFM2.5 dense/MoE (arch 11) — any other arch, `tp > 1`, `pp > 1`, EP, a VL tower, paged experts or a REAP keep-map with a configured spill fails the load closed.
- **LFM2.5 (arch 11) specifics.** The `moe_expert_budget` tier is the one that matters here: LFM2.5-24B-A2B is ~98% routed experts by bytes (11.6 GiB of its 12.7 GiB), so the whole-layer tier alone frees almost nothing and `auto` correctly takes the expert tier first. `memory.offload_exec=cpu`/`passback` is refused for arch 11 — its projections launch through the llama-crate GEMV helpers rather than `hipfire-dispatch`'s `execute_steps`, so the CPU-exec seam could not honour the request; only `pcie` runs. Routed experts are packed into one blob per projection per layer on **both** tiers: one `hip_malloc` per expert would waste ~90 MiB per MoE layer to allocation granularity (the placement census reads stored bytes, so the over-count made the search over-place and OOM mid-load), and per-expert host tensors would pay the 1 MiB host tail pad 128 times per layer. A spilled load keeps `embed_tokens` resident, and a `qt` 44/16/1 embedding is expanded to an F32 gather table plus a second copy for the tied `lm_head` — ~1.1 GiB of resident globals on the 24B, not the ~130 MiB its stored size suggests.
- **Execution.** `memory.offload_exec` decides **who multiplies**, never what spills. `pcie` (default; unset/empty/unknown fall back to it) runs the GPU kernels against host-mapped weights over the link. `cpu` executes those GEMVs on the host, and on a `moe_expert_budget` spill recomputes each selected routed expert's whole FFN on the CPU for AR single-token decode and any `<=4`-row batched forward (narrow speculative verify and accepted-prefix rollback included; wider prefill keeps the GPU grouped PCIe read). `passback` runs both engines at once on a spilled dense step, split by output rows per `memory.offload_passback_share`; routed experts on a `moe_expert_budget` spill take the same CPU splice as `cpu`. `cpu` and `passback` disable hipGraph capture for the model's lifetime and are refused together with a retained-replay (Redline) backend.
- **Flash-Next (qwen4) CPU splice (D1, whole layer, no hotset).** `memory.offload_exec=cpu` (env `HIPFIRE_OFFLOAD_EXEC`) also applies to Flash-Next host-placed routed experts (placed by `HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS` / its auto residency, not by the `memory.*` budgets above). It is opt-in and the `pcie` default is unchanged. When armed, AR single-token decode and any `<=4`-row batched forward (speculative verify included) recompute each selected host-placed routed expert's FFN on the CPU; wider prefill and wider verify keep the GPU path over PCIe. Arming is decided once at load and is fixed for the loaded model, so session snapshots never mix PCIe and CPU execution modes within that load. An armed load runs eager HIP only: `auto` retained replay (Redline) is disabled, and `replay.backend=shadow`, active stream capture, the graph flag and replay recording are refused at attachment. Speculation is width-capped to the 4-row splice: native MTP proposal capacity and `k` and the n-gram/copy lane cap at 3 drafts, MTP lanes refuse draft `k>3` at admission, any direct non-prefill verify or replay above 4 rows is refused before state mutation, and a terminal repair wider than 4 rows declines repair and falls back to reset. The splice emulates the BF16 storage boundaries of the GPU path, but its dot accumulation order is its own, so output is still not bit-identical to resident or PCIe execution. Expert parallelism (EP), the expert slab, the expert tier, AWQ experts and unsupported host dtypes are excluded and stay on PCIe. This whole-layer behavior is what a load **without** an applied expert hotset gets, and it is unchanged; it is architecture-generic (no arch gate). A load **with** an applied hotset is never routed through it, and takes the D2 mode in the next bullet (on whatever arch the loader applied the hotset) or stays on PCIe. `HIPFIRE_MOE_CPU_EXPERTS=0` is the kill-switch. Existing native-MTP admission over host-mapped experts (the `mtp_mode` row) is unchanged. A load that arms the splice prints `qwen4 routed experts: CPU splice armed (offload_exec=cpu) …`.
- **Flash-Next (qwen4) CPU split (D2; `memory.offload_exec=cpu`, or the measured per-load default when it is unset — see the next bullet; architecture-generic).** Hot-set-cold experts on the CPU, hot-set-hot experts on the GPU, for `<=4` rows; the same executor also serves un-tiered host-mapped aggregate layers (every host expert cold). It has no CPU-side arch gate: it arms on every layer of a tier plan the loader admitted. The loader's hotset rule is unchanged: the default `<model>.expert-hotset` sibling sidecar applies on exact gfx1201 only, and an explicit absolute `HIPFIRE_QWEN4_EXPERT_HOTSET` sidecar path applies on any architecture (gfx1100 included, which never applies the sibling sidecar implicitly), subject to the usual hotset skips (unified memory, EP, non-aggregate layouts, an explicit `HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS`) and the tier plan's per-arch format check. The routed-expert CPU splice must not be disabled (`HIPFIRE_MOE_CPU_EXPERTS` not `0`) and `passback` must not be selected. Anything else keeps the D1 or `pcie` behavior above. No variable other than the typed `developer.qwen4_d2_default` selects it. The fixed hotset (its tier digest) and this execution mode are part of the loaded model's cache/session identity (`cpu-tier-split-v1/<tier digest>/<armed-layer bitmap>`, and `cpu-whole-split-v1/<bitmap>` for un-tiered layers; a load with no CPU arithmetic armed keeps the empty, unchanged identity), so a D2 session never reuses state built under another hotset or mode. It is **eager-only**: it refuses graph capture, specialized (verify/replay) capture, retained replay, shadow validation and AQL/PM4 routing. Expert parallelism (EP), pipeline parallelism (PP), AWQ experts, stock FP8 and non-uniform expert layouts are unsupported (a host-mapped expert-slab layer arms only when every present expert's format has a CPU decoder; see the measured-default bullet): a layer on such a route has no masked route or cold-slot installation, does not arm, and the load logs one `[moe-cpu-split] qwen4: not armed on <arch>: <reason>` line per reason (a forced `HIPFIRE_QWEN4_EXPERT_STAGE_MIN_ROWS` of at most 4 rows is refused the same way, because a staged expert table has no masked route). The decode route is the same fixed top-10 QT44/QT53 indexed route on every arch (the symmetric IU4 arm needs at least 512 routing rows, so it never sees a `<=4`-row window). Every call inside a prefill tile (`prefill_scope`, including a prompt of at most 4 rows and bench prefill), any window wider than 4 rows and any 1–4-row tail of a long prefill keep the GPU route (the prefill seal binds an executor-less view of the layer's weights, `Qwen4MoeLayerRuntime::routed_view`).
- **Flash-Next (qwen4) D2 measured default (unset `memory.offload_exec`).** With `memory.offload_exec` unset, a Flash-Next load that places routed experts in host memory decides **once per load** whether to run the D2 CPU split, by predicting the per-layer decode cost of each route from live measurements of **this model on this host**. For one full-cold decode row of one host layer (`top_k` distinct cold experts of the layer's real resident codecs, `B` bytes in all): `PCIe = B / measured link rate + launch floor` and `D2 = the production CPU arm + CURRENT_COPY_TRANSPORT_HANDOFF_NS`, and D2 turns on only when `D2 <= 85% of PCIe` (at least 15% faster; exact equality selects it). **The comparison reads a robust statistic of the CPU arm and is optimistic for PCIe:** the decision's D2 is the **median** of the CPU arm samples (the upper median, `sorted[len / 2]`, so an even or half-slow set is never read as fast) plus the handoff, and its PCIe is the **fastest** of the 5 link samples plus the **minimum** of the 5 launch-floor samples. D2 is chosen when that D2 clears 85% of that best case, **however wide the sample spread** (an arm disturbed 3× on a 7900 XTX, 0.39 ms × 3 + 0.806 ms against 26 MB at 7 GB/s, is still a clear win). A single slow sample can no longer decide the load: an earlier rule read the *slowest* of the 5 samples and refused D2 on one 7,316 µs outlier among ~410 µs samples (worst-case D2 8,122 µs against PCIe 3,691 µs, a load that decoded at ~5 tok/s; the median D2 was 1,216 µs). **Re-probe:** a CPU arm pass whose slowest sample is more than 200% of its median (`CPU_OUTLIER_PERCENT`, strict) is measured **once more** and the two passes are pooled (10 samples; the upper median is the 6th smallest). A pass that is not dispersed is never re-probed and the re-probe never repeats. **The margin is unchanged (85%):** a genuinely slow CPU (every sample slow, or a majority: three or more of 5, five or more of 10 pooled) moves the median and ends on PCIe, and an inconclusive run ends on PCIe. Link samples keep the optimistic fastest sample (a disturbed sample can only be slow; an implausibly fast one keeps PCIe: a lost D2 win, never a wrong D2). There is no spread threshold: the slowest sample, the single-slowest D2 it would have been, medians and spreads (max/min and max/median) are logged for information only. **CPU arm:** `moe_cpu_host_slot_outputs`, the very call the D2 join makes (gate/up GEMV, BF16 boundaries, SwiGLU, down FWHT and GEMV, all on the splice's pinned physical-core pool), over the **actual resident expert spans** of the host layers (hot-set tier owners or the host-mapped aggregate allocations, the same borrowed views the layer build uses), a non-zero deterministic activation row, one decode row. One untimed warm batch (thread pool, scratch) is followed by 5 timed batches of `top_k` experts; every batch takes *different* cold experts, round-robin across the host layers (`(1 + 5) × top_k` = 60 experts, about 150 MB for Flash-Next, beyond a typical last-level cache, so one sample is not served from the previous sample's cache lines). A model whose host layers have fewer cold experts than that fails closed. This is a few milliseconds of CPU and reads no dense 192 MiB weight. The arm takes each expert's actual CPU codec and span lengths as input, and every timed batch carries its own byte count (`DecodeMeasurement::batch_bytes`): `B` is the median batch's bytes and the CPU samples are scaled to it, so layers whose experts have different codecs and byte counts compare as time per byte through the same interface. A host-mapped expert-slab arena (the mq2 split) holds a codec per expert (MQ4, W3, E8P) and the GPU decode reads each routed slot in place in that slot's own codec, so every slab slot carries its own GPU read bytes, resolved from the slot's plan format (E8P gate/up + down 1,305,600 B, MQ4 2,611,200 B at the mq2 shape) and never from its CPU span lengths, the arena size or one constant; a one-codec owner (hot-set tier, un-tiered aggregate) reads the CPU payload bytes. A slab layer whose arena is device-resident or whose present experts include a format with no CPU decoder is not D2 eligible and stays on PCIe. **Link:** 5 live zero-copy reads (a GPU kernel reading 32 MiB of host-mapped memory over the link, timed launch to synchronize) and 5 launch + synchronize timings of a one-element read (the floor); a link sample includes one launch, so PCIe is counted slightly conservatively. **`CURRENT_COPY_TRANSPORT_HANDOFF_NS` = 806,100 ns**: the cost of D2's current copy transport around the CPU arm per layer, one named constant derived from the j1779 timeline (`xtx-decode50/PLAN.md` §0; 43 host layers, 16 measured tokens): readiness wait (stream-sync API wall) 497.7 µs + post-CPU return (cold H2D API entry to copy start) 225.9 µs + result copy 18.2 µs + copy end to next kernel 56.9 µs + join residual 7.4 µs. It is deliberately conservative: the readiness term is the whole stream-sync wait, which also contains GPU prerequisites a PCIe route waits for too. A mapped-result or CPU-wait transport that is measured cheaper lowers this constant (and nothing else in the rule). **Failure fails closed to PCIe:** a zero or non-positive sample, byte count or floor, a failed link probe, a span that does not resolve, too few cold experts, a unified-memory device (never probed) and `HIPFIRE_MOE_CPU_EXPERTS=0` all stay on the link. **PLE prewarm:** the probe does not wait for or pause the PLE sidecar page-cache prewarm (no reusable pause guard exists on this base, and waiting for a full warm would add load latency); a prewarm running during the probe slows or scatters the CPU samples, a minority of which is absorbed by the median (and the dispersed-pass re-probe) and a sustained slowdown resolves to PCIe (a lost D2 win on a contended load, never a wrong D2). There is no arch or card-name rule: a link the GPU reads at 4–7 GB/s (a 7900 XTX over PCIe 4.0 x4) loses to a 0.4–0.6 ms CPU arm plus the handoff, a link at 26–55 GB/s (an R9700 in a Gen5 x16 slot, healthy or contended: CPU 119.6 / link 54.8 and 47.7 / 55.4 GB/s) does not, even against a CPU that streams dense data faster than the link. The load prints one `qwen4 D2 default (N host split layers): ON|off (live samples (…): D2 … us (median CPU arm of N samples[; first pass was dispersed, re-probed once and pooled]) vs best-case PCIe … us per full-cold decode layer (…x; D2 needs <= 85%) [single slowest sample would give D2 … us; median PCIe … us]: CPU arm median … us / slowest … us [spread x… max/min, x… max/median, … GB/s median over … MB of experts] + … us copy-transport handoff; link fastest … / median … GB/s [spread x…] + launch floor … us min / … us median; real-arm probe … ms)` line (the probe wall covers source resolution, warm-up, both sample sets and the one-time compile of the read kernel); a successful measurement is reused by a later load of the same shape in the same process, a failed one is measured again. **What it arms:** the layers the split can serve that place experts in host memory — hot-set tier layers (as above) and, with no hot set at all, **un-tiered aggregate layers whose two expert allocations are host-mapped QT44/QT53** (the shipped placement). The QT44/QT53 pair is a real backend limit, not a CPU decoder limit: the D2 GPU masked route exists only for the QT44/QT53 grouped route (`cpu_split_decode_conflict`) and the executor requires dim % 256 == 0, intermediate % 128 == 0, top-10 and 4 rows. For an un-tiered layer every host-placed expert is cold: the CPU computes each selected expert and the GPU keeps the router, shared expert and canonical combine, through the same D2 executor and join (not the D1 splice's arithmetic). Its identity is `cpu-whole-split-v1/<layer bitmap>` (no tier digest or sidecar: nothing to freeze), the arm line reads `[moe-cpu-split] qwen4: N whole-layer host aggregate layers armed …`. A host-mapped expert-slab layer (the mq2 split's MQ4/W3/E8P mix) is the third span owner: the CPU multiplies each present expert straight out of the owning arena's pinned host bytes in that expert's own codec (checked per-expert `(gate_up, down)` slices, nothing copied; the slab K1 planner and kernels skip the masked `-1` ids), its identity is `cpu-slab-split-v1/<layer bitmap>` and the load log names the CPU kernel of every format it decodes. A layer the split cannot serve (EP, AWQ, FP8, non-QT44/QT53, a forced `HIPFIRE_QWEN4_EXPERT_STAGE_MIN_ROWS` of at most 4 rows, an expert-slab layer with a device-resident arena or a format with no CPU decoder) is refused or skipped as before. **Scope:** the default requests **only** the D2 split (decode and `<=4`-row verify); it never selects D1, never sets `memory.offload_exec`, `cpu_exec_enabled()` or `passback`, never latches process state (each load decides for itself), never changes expert placement (the owners are views over the loader's existing pinned host bytes, so no expert is copied or allocated twice), and every call the split may not serve keeps the GPU route: any call inside a prefill tile (the forward's `prefill_scope`, set for ordinary prefill including a prompt of at most 4 rows and bench prefill; verify runs with it off), any window wider than 4 rows, and any 1–4-row tail of a long prefill whose routing window is the canonical chunk. The D2 executor gate sees only the row count, so the qwen4 prefill seal binds those calls to an executor-less view of the layer's weights (`Qwen4MoeLayerRuntime::routed_view`, decided by `d2_executor_admitted` from the phase and `cpu_split::cpu_splice_window`). **Override:** an explicit `memory.offload_exec` (`HIPFIRE_OFFLOAD_EXEC`, any value) always wins, both ways — `pcie` or `passback` turns the default off, `cpu` keeps its existing D1/D2 behavior unchanged (D1 on un-tiered layers) — and `HIPFIRE_MOE_CPU_EXPERTS=0` also turns it off. The typed developer key `developer.qwen4_d2_default` (`HIPFIRE_QWEN4_D2_DEFAULT`) takes `false` = never and `true` = force on (skipping only the cost comparison; lab and receipt runs). The unset `memory.offload_exec` is not carried in the process config, so an explicit `pcie` is distinguishable from the default. **Hardware receipts (7900 XTX, gfx1100, PCIe 4.0 x4, in a Strix Halo host; candidate `7ce8af5bc` on beta `61788435d`; taken with the first, dense-rate selector, which this cost comparison replaces and which selected D2 on that card for the same reason: a 4.2 GB/s link).** H1 `/home/kaden/qcal/release-0.4.3/d2-default-gates/hold-results/d2-mq4-7ce8af5bc-h1` (j1769; mq4-xts split base, q4 PLE, QSA Q8 / GDN Q8, two interleaved ABBA pairs, one sample per cell): decode tg64 **5.906×** (30.043 against 5.087 tok/s), pp512 **0.9992×**, pp8192 **1.0018×**; D2 armed 43 layers (44 under MTP) with the first selector's lines CPU 107.5 / 109.7 / 113.5 GB/s against 4.2 GB/s over the link; two candidate processes give identical greedy IDs on 8 prompts × 64 tokens (exact). H2 `/home/kaden/qcal/release-0.4.3/d2-default-gates/hold-results/d2-kld-7ce8af5bc-h2` (j1770): paired decode KLD (D2 minus PCIe), code24 seq 0:10, n = 10, mean **+0.000554**, paired t 95% CI [−0.002191, +0.003299], bootstrap [−0.001667, +0.002874] (PASS: lower bound ≤ 0). The selector assumes no generic host bandwidth; the figures are this host's own, and another host or link is decided by its own measurement at load, not by these receipts. Opt-outs: `memory.offload_exec=pcie` or `HIPFIRE_QWEN4_D2_DEFAULT=0`. Three limits are accepted (the "Accepted limits" bullet below).
  - **Calibration record (D2 default).** The `hipfire calibrate` record ([Calibration](#calibration-machine-written)) supplies **no** figure of the D2 decision and never replaces a live one. Its engine rates are dense, alone-and-best-of figures with no sample spread: they cannot say what the production arm costs on this model's experts or how noisy this host is right now, so a stored CPU or link rate alone can never enable D2. The first eligible, not-yet-measured shape (device, resident codecs, expert shape) in a process takes live real-arm and link samples whatever the record says; later loads of the same shape in that process reuse that successful measurement from the in-process cache (the load log says so), and a failed or invalid measurement is never cached, so the next load measures again (the pinned H2D and zero-copy figures the record stores still feed the other host-link policies, not this one). `hipfire calibrate` therefore no longer measures or writes `pool = "splice"` engine entries; an entry an earlier version wrote stays readable in the record and is ignored by the selector. `hipfire calibrate --show` prints one `D2 default    decided at model load …` line per fresh device (JSON `d2_default.status` = `load`; `off` on a unified-memory device, which never probes), produced by the selector's own `d2_default::decide` with no model arm, so it can say the decision is made at load and never claims an `ON` it cannot make.
  - **Measured probe overhead:** R9700 hold `j1417`, three fresh default loads: 88.3 / 92.2 / 85.4 ms wall, including CPU-arm setup and live link probes; calibration records do not skip this work. All three chose PCIe despite CPU-arm sample spreads of 1.28× / 2.32× / 2.59×. Matched tg64 AR means were 48.98 / 48.92 / 48.96 tok/s versus D2-off 48.92 (pooled +0.07%, within-process spread about 0.20%); all 28 requests had identical 64-token IDs. Receipt: `release-0.4.3/d2-selector-evidence/run-r9700-gate/GATE-RESULT.json` (one 419.9 s exclusive hold, max-seq 8192). No new XTX GPU result is claimed.
  - **Speculative width and eager scope (default only).** Unlike D1 and the explicit `cpu` D2, the measured default is not load-wide host arithmetic (`Qwen4GpuForward::has_cpu_deferred_experts` is false for it), so that predicate imposes nothing on it, and it asks for eager HIP only on the calls that execute the split (`call_executes_host_arith`: a whole `<=4`-row non-prefill window). A verify of at most 4 rows runs the split, row-exact with the D2 AR decode; a wider verify and every prefill call keep the unarmed GPU route over PCIe and its graph/replay rules, so those rows carry the PCIe arithmetic rather than the D2 decode's. **The native MTP `auto` draft cap does follow an armed default**, because it is read independently of `has_cpu_deferred_experts`: load resolves it after placement as `min(GPU row-exact verify rows, CPU-owned expert window) − 1` (`mtp_k_default`, `Qwen4Bundle::native_mtp_cpu_window_rows`, which reads D1 and any armed D2 tier split, the measured per-load default included). An armed D2 load therefore has a 4-row window and `auto` = 3, so it never drafts past the rows that run CPU arithmetic, while a load where D2 is not armed (`memory.offload_exec=pcie`, `HIPFIRE_QWEN4_D2_DEFAULT=0`, a link the selector keeps on PCIe, a fully resident load) resolves 7 (8 GPU rows). An explicit `mtp_k` is not rewritten by the `auto` resolution (its separate limits are in the `mtp_k` row of [Mechanism knobs](#mechanism-knobs)). Before commit `1638dd485` the `auto` cap was 7 only on gfx1151 and 3 elsewhere, and this paragraph said the default capped no speculative width; that is no longer the behavior. Retained decode replay (Redline) is off on an armed load because its decode runs host arithmetic. No automatic-MTP timing is claimed for the selector change (the `j1417` comparison above ran with speculation off).
  - **Accepted limits (default).** (1) A native-MTP verify of more than 4 rows takes the PCIe route, so its arithmetic is the PCIe arithmetic, not the D2 decode's. The `auto` cap no longer produces such windows on an armed load (cap 3 verifies at most 4 rows); only an explicit `mtp_k` above 3 or a longer n-gram take can. On an R9700, where the pessimistic selector keeps the load on PCIe, D2 is not armed and `auto` resolves 7. (2) D2 arithmetic is placement-dependent, and this still holds whenever D2 is ON under the current selector: native MTP's head reserve shifts one resident layer, changing which experts run on the CPU, so a native-MTP load and an AR load can have different cold sets and the MTP output can differ from the D2 AR output. No automatic AR = MTP identity is claimed for the default; compare AR and MTP only at an equal, pinned placement (identical armed `partition_digest`, see the pinned retest prerequisite below). HISTORICAL (the H1 numbers only, taken with the first dense-rate selector on the XTX; not current-selector receipts): 44 armed layers and 56.1 GiB pinned host under MTP, 43 and 54.9 GiB for AR, so native MTP differed from the D2 AR decode on 2 of 8 prompts (fiction_lighthouse at token 39, rust_csv at token 26) and the H1 hold's MTP-equals-AR, equal-placement and equal-stage-state gates were FAIL for that reason; D2 AR itself differed from the PCIe beta on 4 of 8 prompts (tokens 14, 39, 26, 42), since the arithmetic differs by design. Those H1 receipts used the old selector and the old `auto` cap (3 on gfx1100, before `1638dd485`) and are not receipts for the current selector or cap. (3) The lab KLD loaders use their own context placement (37 host layers in H2), identically on both arms, not the product serve placement.
- **D2 partition, transport and join.** The router and the original ids/weights are never rewritten. Per layer and row, ownership is a fixed function of the hotset: each routed slot whose expert is hot stays on the GPU grouped route, each cold slot is computed on the CPU, and a slot with an invalid id belongs to neither arm (the existing router-fault handling still sees it). The partition is therefore deterministic for a given route. Per layer the engine downloads the rotated F32 activation `[rows, 2560]` plus the original I32 ids and F32 weights (top-10) — 10,320 B/row-layer, no residual — and uploads a separate masked-ids tensor (40 B/row, cold slots set to `-1`) so the GPU kernels skip them, then, after the GPU down projection on the same stream, uploads each cold slot's raw F32 expert output (10,240 B per cold slot). Decode total is 10,360 + 10,240 × cold_slots B/layer, i.e. 10,360 B (all hot) to 112,760 B (all 10 cold). The original canonical BF16 combine then runs once over the merged slots, shared by both arms. The original GPU kernels may write masked slots as zero, so the cold upload must (and, by same-stream order, does) land after them. The CPU arm is a synchronous scoped splice pool: it runs inside the join while the GPU tail executes, and no CPU job outlives the join. The `<=4`-row cap matches D1 and uses the same arithmetic for verify, copy and replay; the native MTP `auto` draft cap follows that window on every armed CPU mode, the measured default included (`min(GPU row-exact rows, 4) − 1` = 3 on an armed D2 load, see the speculative-width bullet above), while an explicit `mtp_k` is not rewritten by `auto` and is admitted per its existing caps. The CPU arm keeps its own dot-accumulation order, so output is not claimed bit-identical to resident or PCIe execution.
  - **D2 transport default and kill switch (`HIPFIRE_QWEN4_D2_RETURN`, `HIPFIRE_QWEN4_D2_JOIN`).** The cold-result return (`copy` | `mapped`) and the join (`host` | `cpwait`) are chosen per load. On exact **gfx1100 (7900 XTX)** an unset return requests `mapped` and an unset join requests `cpwait`: the CPU writes the cold slots into one mapped coherent allocation that the unchanged canonical combine reads directly, and the combine is enqueued behind a command-processor wait that the CPU releases (byte-exact: same CPU arm bytes, same combine; 7900 XTX Flash-Next 32K, ABBAAB n=3/arm, 33.3 -> 34.6 tok/s, +3.9%, ids identical; 131K and 250K confirmation pending). Per-layer capability routing still decides: only all-cold (whole-layer host) layers qualify; a mixed hot/cold or tier layer keeps `copy` / `host` and the refusal is logged at load. Every other arch keeps `copy` / `host`. An explicit value always wins, and `HIPFIRE_QWEN4_D2_RETURN=copy HIPFIRE_QWEN4_D2_JOIN=host` restores the previous transport exactly on any arch (the kill switch); `HIPFIRE_QWEN4_D2_RETURN=copy` alone also keeps the join `host` because the CP-wait join needs the mapped return. `HIPFIRE_QWEN4_D2_READY` is unchanged (default `sync`). The load summary reports the effective per-layer modes and `transport request: default (gfx1100)` when the arch default applied. The measured D2 default selector's handoff price is still the copy-transport constant, so the decision to arm D2 is the same on every transport.
- **D2 status and lab gate.** Prior tiered D2 (hot-set placement) quality and native product gates PASS, per `offload-r9700-arm/RESULT.md` lines 52–70 (final section; supersedes the earlier pending and invalid status). Paired per-chunk decode KLD delta (D2 minus PCIe): R9700 WT2 (32 chunks) +0.002442, bootstrap 95% CI [+0.000217, +0.004942] (upper ≤ +0.010, PASS); R9700 code24 (24 chunks) +0.001178, CI [-0.001942, +0.004334] (includes zero, PASS); XTX code24 seq0:10 (n=10) -0.001441, bootstrap CI [-0.005204, +0.001578] (includes zero, PASS), per `ssh://hipx/home/kaden/pm-wave/offload-prefill/d2quality-xtx/runs/FINAL-xtx-d2q-INDEX.md`. R9700 product AR=MTP 8/8 per route: `offload-r9700-arm/runs/final-20261017T090726Z-f91114cb1c20-r9700/final-verdict.md`. These are tiered results. The whole-layer (un-tiered) D2 measured default has its own XTX receipts, H1 (ABBA timing, IDs, determinism) and H2 (paired KLD), in the measured-default bullet above; its speed-up is measured on that card and PCIe 4.0 x4 link only. The correctness/timing probe is the lab-only `qwen4_cpu_split_parity` example of `hipfire-arch-qwen4` (feature `lab`, no new variable). Whole-layer mode: `HIPFIRE_QWEN4_D2_DEFAULT=1` (typed `developer.qwen4_d2_default`) with `HIPFIRE_OFFLOAD_EXEC` unset (an explicit value is refused) and no `--hot-budget-mib`/`--hotset`: `HIPFIRE_QWEN4_D2_DEFAULT=1 cargo run --release -p hipfire-arch-qwen4 --features lab --example qwen4_cpu_split_parity -- MODEL.mq4 TEXT.txt --untiered` [same options, no `--hot-budget-mib`/`--hotset`]. Tiered mode: `HIPFIRE_OFFLOAD_EXEC=cpu cargo run --release -p hipfire-arch-qwen4 --features lab --example qwen4_cpu_split_parity -- MODEL.mq4 TEXT.txt --hot-budget-mib N [--hotset /abs/SIDECAR] [--layers 4|0,1,2|all] [--offset 0] [--prompt-rows 32] [--steps 16] [--rows 1..4] [--greedy] [--timing-rounds 2] [--warmup-steps 2] [--no-endpoints] [--no-timing] [--json OUT.json]`. `--hot-budget-mib` fills the device hot set from the sidecar's ranking (every other trunk expert stays in the pinned host tier); the sidecar is the explicit `--hotset` path or `HIPFIRE_QWEN4_EXPERT_HOTSET` on any arch, and gfx1201 also takes the default sibling sidecar. `--layers` selects the first N armed layers, a list, or all; `--rows` feeds 1–4 teacher-forced tokens per narrow forward (`--greedy` is rows 1 only); `--prompt-rows` (≥ 5) is prefilled on the wide route first. It compares same-input captures per layer call (D2 joined target against the all-CPU and all-GPU targets, bitwise hot/cold-slot and all-hot/all-cold endpoint gates, an overlapped-vs-serial timing run), exits 0 only if every gate passes, 1 on a gate failure and 2 on an error; max-abs/max-rel figures are measurements, not gates.
  - **Row-count audit (source baseline `70fd77671`).** The CPU-first regression oracles pass before any arithmetic change: `d2_window_rows_are_byte_identical_to_one_row_ar_for_every_position`, `canonical_join_multirow_windows_match_one_row_ar_for_sampled_masks`, `canonical_join_two_row_window_is_byte_identical_for_all_1024_masks` (`crates/hipfire-dispatch/src/cpu_exec.rs`) and `gemv_shared_window_rows_match_one_row_at_every_position` (`crates/hipfire-cpu/src/gemv.rs`). They cover widths 2–4 at every position, changing cold sets, repeated experts, invalid slots and scratch reuse; no CPU row-count defect was reproduced.
  - Width one selects indexed decode and widths 2–4 select the grouped protocol (`crates/hipfire-arch-qwen4/src/program.rs:903–917`), but different protocol names do not establish different arithmetic. Indexed and grouped gate/up retain the same four accumulation chains and reduction (`kernels/src/qwen4_gemv_mq4g256v2_moe_gate_up_k8_indexed_batched.hip:275–305`; `kernels/src/gemm_mq4g256v2_moe_grouped_top10_o4_r8_x4.gfx1151.hip:193–271`). Few-row grouped down re-enters the indexed down (`crates/hipfire-dispatch/src/pipeline/qt44_qt53_prefill.rs:1660–1667,1744–1745`), and both protocols use `moe_down_combine_top10_batched` (`pipeline/mod.rs:4949–4961`; `qt44_qt53_prefill.rs:1871–1872,1904–1913`). Masking preserves original flat slots, and only routed stages see masked IDs (`cpu_split.rs:494–524`; `pipeline/moe_program.rs:474–504`); the join uses the original route. This is source evidence, not a hardware bitwise claim.
  - Verify and accepted-prefix replay retain the existing row-exact policy: verify uses `prefill=false` (`crates/hipfire-arch-qwen4/src/bundle.rs:1419–1432`), and partial acceptance either retains captured rows or calls `spec_forward_rows` on the prefix (`mtp_spec.rs:5711–5730`, generic commit `:752–761`). No arithmetic rewrite or speculative-width reduction is justified by the CPU oracle and source audit.
  - **Pinned retest prerequisite.** The original R9700 AR/MTP receipt used **6850 versus 6479 hot experts**, so it did not hold the numerical partition constant. A sidecar fixes ranking, not count (`crates/hipfire-arch-qwen4/src/expert_hotset.rs:10–12`); the loader's available budget subtracts the MTP head reserve (`crates/hipfire-loader/src/carriers.rs:1918–1934`; `expert_residency.rs:326–331`). Set the same explicit `HIPFIRE_QWEN4_EXPERT_HOT_BUDGET_MIB` on both loads, with the same model, sidecar, PLE and other settings; an unavailable pin fails instead of shrinking. Require identical armed `partition_digest` values **before** comparing IDs. A differing digest is an invalid identity comparison, not evidence of a verify arithmetic bug.
  - The numerical identity already includes the frozen partition: `Qwen4GpuForward::host_exec_identity` passes `lease.logical_digest()` (`gpu_forward.rs:5839–5861`), whose plan hash covers each expert's tier (`expert_tier_plan.rs:201`); `digest_covers_placement_format_source_artifact_and_window` (`:783–796`) proves that changing a placement bit changes it. `gpu_forward/cpu_splice.rs:499–502,1040–1068` includes that digest, armed-layer bitmap and sidecar hash in the D2 identity and checks distinct digests cannot collide at the identity-construction layer. The armed log now exposes the same plan hash as `partition_digest=<64 lowercase hex>`.
  - **GPU acceptance (HISTORICAL tiered-D2 gate; not a current instruction).** While D2 was the tiered, opt-in route, D2 was kept default off and this gate applied: the product eight-prompt, 128-greedy-token AR/MTP comparison on R9700 (`gfx1201`), then XTX (`gfx1100`, explicit sidecar), with identical pinned partition digests; require the IDs, tau receipts and a read of the generated text. Sampled generation is not an acceptance gate. Cross-route byte equality of every wide-verify row/state is not a prerequisite where arithmetic legitimately differs. The historical wide-verify failure is classified as a shared-base GDN rounding-route difference, not a proven D2 race; retain native capture-free bitwise bug checks.
  - **Armed KLD gate (tiered and whole-layer gates PASS).** Keep the existing WT2/code24 `qwen4_kld s11ref --decode` S11 fixtures unchanged. The tiered gate used the lab-only `--hotset /abs/SIDECAR --hot-budget-mib N` admission levers on both `HIPFIRE_OFFLOAD_EXEC=cpu` (D2 on) and `HIPFIRE_OFFLOAD_EXEC=pcie` (D2 off), with the same model, sidecar and budget. It required actual armed D2 on the CPU arm and equal frozen-plan digests across the CPU/PCIe pair; an environment-only attempt that never arms D2, or OOMs, is not quality evidence. It computed paired per-CHUNK/sequence KLD deltas (D2 on minus off) and a two-sided 95% CI `[lo, hi]` separately for WT2 and code24: pass if `lo <= 0`; only a significant positive regression (`lo > 0`) must satisfy the upper-bound budget `hi <= +0.010`. Results are in 1140; the tiered proof does NOT carry over to the whole-layer route. **Whole-layer (current) gate (run as H2):** `qwen4_kld s11ref --decode --untiered d2|pcie` (no `--hot-budget-mib`/`--hotset`): `d2` needs `HIPFIRE_QWEN4_D2_DEFAULT=1` with `HIPFIRE_OFFLOAD_EXEC` unset; `pcie` needs `HIPFIRE_OFFLOAD_EXEC=pcie` with the force unset. Both arms use the same model and the same placement env: `HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS` unset (or `auto`) resolves the product loader's automatic placement through the one shared `expert_residency::incumbent_auto_resident_layers` (the lab loaders `qwen4_kld`, `qwen4_cpu_split_parity` and `qwen4_greedy_ids` place exactly what the product loader would for the same inputs), and a fixed `N` is only an optional diagnostic override; the receipt records which one produced the placement as `placement_source` (`env-fixed`, `product-resolver` or `product-resolver-resident`). Both arms must show the same resolved placement digest. The d2 arm must arm every host-placed trunk layer as a whole-layer owner (`cpu-whole-split-v1/<bitmap>`); the pcie arm must arm nothing. Each run writes `OUTPUT.d2.json`. Apply the same paired per-CHUNK CI rule to the paired code24 decode runs (n=10). H2 on the XTX (code24 seq 0:10, n = 10, both arms on the same 37-host-layer lab placement, QSA Q8 / GDN Q8): paired mean +0.000554, t 95% CI [−0.002191, +0.003299], bootstrap [−0.001667, +0.002874], PASS (receipt `/home/kaden/qcal/release-0.4.3/d2-default-gates/hold-results/d2-kld-7ce8af5bc-h2`). The lab loader sizes its own context and places its own layers (37 host layers, against 43 in the product serve load of H1), so this measures D2 against PCIe at that placement.
  - **KV/GDN pins and runtime proof.** Hold the KV and recurrent-state formats identical between the baseline and D2 arms, including benchmarks. For both manual lab loaders (`qwen4_kld` and `qwen4_cpu_split_parity`), set `HIPFIRE_KV_MODE=fp8 HIPFIRE_STATE_QUANT=q8` on R9700 (`gfx1201`), and `HIPFIRE_KV_MODE=q8 HIPFIRE_STATE_QUANT=q8` on XTX (`gfx1100`), in every process. For product daemon loads, send explicit JSON `load_params` fields `"kv_mode":"fp8","state_quant":"q8"` on R9700 or `"kv_mode":"q8","state_quant":"q8"` on XTX. The wire key is `kv_mode`, not `kv_cache`; the settings key is `memory.kv_cache`. The daemon does not read `HIPFIRE_STATE_QUANT`, so manual-loader environment pins are not a substitute for its wire fields.
  - Retain actual resolved `kv_mode`, `qsa_state` and `gdn_state` proof for every baseline/D2 load, alongside the partition digest: each manual tool must emit `[qwen4_kld] resolved kv_mode=… qsa_state=… gdn_state=…` or `[qwen4_cpu_split_parity] resolved kv_mode=… qsa_state=… gdn_state=…` from the final allocated state/getters (or resolved constructor inputs only where no fallback can change them). Requested environment values alone do not prove the actual formats. For daemon loads, retain the actual `qwen4 state: QSA fp8|q8 K/V (<backend> context storage), GDN q8 recurrent` load line. The separate F32 MTP head is not the trunk QSA/GDN format.
  - **Historical-source boundary.** Formats inferred from old source's `auto` resolution are source-derived, not runtime-observed, and old runs cannot be retroactively labelled explicitly pinned. The old `4363` XTX F32-source binaries cannot certify current beta's Q8-default math: freeze the new source and rerun the matched Q8 baseline/D2 gate, rather than forcing old `bf16` to evade the beta-default gate. R9700 FP8 math is unchanged, but record the admission-census delta with the new source identity and frozen partition instead of treating admission as unchanged. These requirements establish no GPU quality pass.
  - **Timing repeat receipts.** `qwen4_cpu_split_parity` must report per-run output hashes for fresh same-route Timing-versus-Timing repeats at the same pinned partition, not merely compare overlapped Timing against serial execution. The old Timing hashes were not retained, so same-route determinism remains unresolved until new repeat receipts pass; component parity or cross-route arithmetic comparisons cannot substitute for that gate. Record the pinned digest with each repeat and retain truthful failures rather than claiming a pass from missing hashes.
  - **Full-versus-Timing source finding; reference-only repair.** The Full probe replayed the unmasked routed tail before `SharedDown` (`moe_program.rs:1433–1438,1471–1473`) with the original fused-shared descriptor still live. Decode (`pipeline/mod.rs:4494–4522`, `mq_rotate_x_128_v2.hip:106–112`) and grouped prefill (`qt44_qt53_prefill.rs:1575–1608`, `moe_gate_up_unscatter_silu_top10.hip:94–105`) can therefore apply sigmoid to the BF16 shared scalar again; `qwen4_gemv_bf16_xf32.hip:477–494` consumes that scalar directly. The reference replay uses copied parameters with `skip_shared=true` and no shared prelude, preserving original routed ids and arithmetic. Production Timing has no extra replay; same-stream FIFO and pinned-buffer lifetime (`cpu_split.rs:659–665`, `ffi.rs:3353–3402`) do not support a transfer-race diagnosis. This repair adds no barrier or production performance change and is not a corrected GPU pass; same-route Timing determinism still needs new repeat receipts.
- **Load lines.** A spill prints `partial offload: {host} of {n} layers host-placed[, routed experts host on {k} layers], device weights {X} MiB of {Y} MiB available` plus the pinned-host report and the `kv reserved:` accounting line; a fit prints `partial offload: model fits - every layer resident`. A `cpu` load adds the `cpu exec: {covered}/{spilled} spilled layers fully covered …; uncovered quants: …` coverage line.
- **Spec.** Native MTP is opt-in on a host-mapped expert spill except on exact gfx1100 and gfx1201, where `auto` keeps the head — see the `mtp_mode` row. Full write-up: [`env-vars.md`](env-vars.md) offload rows and [`plans/partial-gpu-offload-design.md`](plans/partial-gpu-offload-design.md).
- **Narrow verify routing.** Resident MoE `SpeculativeVerify` batches retain beta's legacy grouped route. Narrow Path1 applies only to host-mapped routed projections; the coalesced gfx1201 host-weight narrow kernel remains available.
- **CPU execution route.** On a CPU-spilled Qwen3.5-family model, single-token decode (MTP forced advances and terminal flushes included) runs the same step program as a resident model; its GEMV steps take the CPU splice. Where the fused gated-norm MQ rotation admits the shape (A3B), the DeltaNet output projection's CPU step reads the GPU-rotated activation instead of rotating on the CPU, so `cpu` output can differ from earlier builds in the last bits. This execution routing does not change the MTP `auto` admission policy above.
**`passback` output is not reproducible run to run, by design.** `cpu` and `pcie` run a layer's GEMV on different engines, so their arithmetic need not agree, and `passback`'s row split is scheduled per process from the seeding probe and the online arm timings — two identical greedy runs mix the engines differently. Within a mode at a fixed split the output is deterministic, and `memory.offload_passback_share = 0` is byte-identical to `memory.offload_exec = cpu`. Pin the share before diffing `passback` output against a reference. Design: [`plans/partial-gpu-offload-design.md`](plans/partial-gpu-offload-design.md#622-pass-back-scheduled-co-inference-of-the-spilled-layers-memoryoffload_execpassback).

### Prompt / assistant-turn cache

| Key | Default | Values |
|---|---|---|
| `memory.prompt_cache_capacity` | `32` | int 0–1048576; maximum cached assistant-turn tokenizations (`0` keeps none). Env: `HIPFIRE_PROMPT_CACHE_CAP`. |
| `memory.prompt_cache_unbounded` | `false` | Remove the capacity bound. Env: `HIPFIRE_PROMPT_CACHE_UNBOUNDED`. |
| `memory.session_cache_bytes` | `8589934592` | int 0–1 TiB; byte budget (a ceiling, not a reservation) of the Qwen4 and Gemma 4 engine-owned session cache (`0` = off, cold prefill every turn). Qwen4 keeps live continuations and session-exact snapshots, including boundaries descended from a tail/resident restore; these are not a cold-prefill identity guarantee. Gemma 4 resumes a follow-up turn from the previous prompt rounded down to the prefill width, byte-identical to a cold prefill. Qwen4 can park one longer live conversation across an early-diverging short request: LCP at most 1024 tokens and total prompt + maximum output + write lookahead at most 8192 rows. It copies fixed state and only overwritten row prefixes, not the full context; the next strict main-history extension restores those bytes and continues live when it matches further than the side history. A longer request, unavailable write bound, or insufficient guarded memory uses ordinary cache/reset behavior. Parking and snapshots yield to context growth. Free device memory keeps headroom for that growth and the allocation in flight: on a unified-memory APU `clamp(MemTotal / 16, 2 GiB, 8 GiB)`, on a discrete GPU 256 MiB; disk staging applies it too. Env: `HIPFIRE_SESSION_CACHE_BYTES` ([`env-vars.md`](env-vars.md)). |
| `memory.session_cache_disk_bytes` | `0` | int 0–64 TiB; byte budget of the session cache's disk tier (one budget for the whole `memory.session_cache_dir`, every model and build). Snapshots evicted from memory, and every snapshot at model unload, are written there and restored on demand, surviving daemon restarts of the same build; LRU over leaves, other builds' files evicted first. `0` = off. Env: `HIPFIRE_SESSION_CACHE_DISK_BYTES` ([`env-vars.md`](env-vars.md)). |
| `memory.session_cache_dir` | `""` | Directory of the session cache's disk tier; empty = `$HIPFIRE_HOME/session-cache`. One process uses it at a time (a second one runs without the disk tier). Env: `HIPFIRE_SESSION_CACHE_DIR`. |
| `memory.ple_hot_store_bytes` | `0` | `auto` or int 0–1 TiB; prewarm budget of the Qwen4 PLE hot-row store. The most frequently used rows of the per-layer-embedding table, ranked by the `<model>.plehot` sidecar (`hipfire-quantize ple-hot-rank`), are copied verbatim from the model file into the `<model>.plehotrows` hot-row pack (`hipfire-quantize ple-hot-pack`), grouped by rank. Rows the pack holds are read from it through the page cache and decoded by the same row decoder, so output is byte-identical to the model-file path. After load the first N bytes of the pack are read into the page cache (hottest tier first, then tier 0 once more so its pages are promoted to the active list) while no request is in flight (paused for each whole request, resumed 5 s after the last): `auto` = `MemAvailable` minus the CMA region minus 4 GiB at ready, an explicit N is that many bytes, both capped at the pack. The pack is clean page cache: the kernel drops it under memory pressure instead of swapping. Off by default (`0`); unified-memory GPUs only. Env: `HIPFIRE_PLE_HOT_STORE_BYTES`, paths `HIPFIRE_PLE_HOT_RANK_FILE` / `HIPFIRE_PLE_HOT_PACK_FILE` ([`env-vars.md`](env-vars.md)). |
| `memory.ple_format` | `auto` | `auto` \| `q4` \| `q8`; which PLE row table a Flash-Next (Qwen4) load reads. `q8` = the exact Q8F16 rows (the `<model>.ple-q8` sidecar of a split base, which is the stock MQ4 XTS layout, or the rows embedded in a grandfathered monolithic MQ4 XTS install); `q4` = the 4-bit Lloyd-Max rows `<model>.ple-q4`, decode equal to the tested q4 table, a load never derives it: a missing file is refused naming both searched locations (the model path as given, then its canonical symlink target), `hipfire pull <tag> --ple q4` and `hipfire ple derive --format q4 <model>` (derives it from the local Q8 rows, about 1 minute, progress on stderr). `auto` = q4 on gfx1151 (unified-memory Strix Halo), q8 elsewhere; `auto` never derives or downloads and falls back to the other available format with a note. An explicit `q8` on a split base needs `<model>.ple-q8` (`hipfire pull <tag> --ple q8`; `serve` downloads it on demand for registry models). See [Qwen4 PLE row tables](#qwen4-ple-row-tables). Env: `HIPFIRE_PLE_FORMAT` ([`env-vars.md`](env-vars.md)). |
| `memory.ple_prewarm_bytes` | `auto` | `auto` or int 0–1 TiB; page-cache prewarm of the q4 PLE table, paused for the whole of every request (resumes 5 s after the last one) `<model>.ple-q4` after the model is ready (a rank-ordered table: hottest tier first, then tier 0 again, the hot-row store's prewarm; a row-ordered one: sequentially). `auto` is derived from the host RAM left once the model is loaded: the base budget is `MemAvailable` − CMA region − 8 GiB, sampled once after the pinned experts and the rest of the model are loaded (so the weights are already subtracted and the reserve is never charged twice). On a unified-memory GPU and on a discrete GPU whose experts are all resident in VRAM, `auto` = min(15 GiB, that budget) (the hottest rows, where the value per GB of warm rows is highest). Only on a discrete GPU with host-placed experts (pinned or host-mapped expert tensors) is `auto` uncapped, so a host with more free RAM keeps more of the table warm, hottest rows first. Every `auto` budget is capped at the table's data bytes; an unreadable `MemAvailable` resolves `auto` to 0 (off). `0` = off (kill switch); an explicit N is that many bytes, is never capped at 15 GiB or otherwise rewritten, and overrides the RAM derivation (the table still caps it). The pages are clean, reclaimable page cache: dropped, never swapped, under memory pressure, and not charged in admission. See [Qwen4 PLE row tables](#qwen4-ple-row-tables). Env: `HIPFIRE_PLE_PREWARM_BYTES` ([`env-vars.md`](env-vars.md)). |
| `memory.ple_load_fill_bytes` | `auto` | `auto` or int 0–1 TiB; bytes of a rank-ordered q4 PLE table `<model>.ple-q4` read into the page cache **while the model loads**, hottest rows first, at most 0.2 GB/s so the weight upload keeps the drive. The upload streams the model file through the same cache and evicts the oldest pages first, so the fill re-reads evicted pages of its prefix (found with `mincore`), hottest first, until ready. Ready never waits for it; the resident prefix serves the first request, and the prewarm after ready (`memory.ple_prewarm_bytes`) continues at its end. `auto` = 14 GiB on every GPU (the smallest prefix whose modelled prose and agentic 8K prefill is within 1% of the 15 GiB prewarm's); any value is capped at the table, at an explicit `memory.ple_prewarm_bytes`, and at the host RAM available at load start (`MemAvailable` minus the CMA region, plus TTM's page pool) minus the weights the load puts in host RAM minus 8 GiB. `0` = off. Clean, reclaimable page cache, not charged in admission. Env: `HIPFIRE_PLE_LOAD_FILL_BYTES` ([`env-vars.md`](env-vars.md)). |
| `memory.state_quant` | `auto` | `auto` \| `q8` \| `int8` \| `q8_ef` \| `fp32` \| `f32` \| `q4` \| `int4`. GDN/DeltaNet recurrent-state storage; `auto` per-model/arch default (Qwen3.8-Flash-Next: F32 on exact gfx1151, Q8 elsewhere; Qwen3.5: Q8) and never `q8_ef`. Explicit values honored or refused, never rewritten; q4/int4 Qwen3.5-only; `q8_ef` (explicit opt-in) Qwen3.8-Flash-Next-only: Q8 plus a live f16 error-feedback residual zeroed on rollback/restore, own `q8_ef-reset` cache scope, restored/rolled-back continuation differs from uninterrupted EF, MTP == AR not guaranteed, sampled requests use AR until the sampled-MTP law is verified, refused with `HIPFIRE_QWEN4_GDN_Q8_INLINE=1`. Kill switch: `hipfire config set state_quant q8` or `HIPFIRE_STATE_QUANT=q8`. Env: `HIPFIRE_STATE_QUANT` ([`env-vars.md`](env-vars.md)). See [KV cache](#kv-cache) for the Qwen4 format, numerical and memory-owner semantics. |

Qwen AR, MTP and DFlash multi-turn reuse store each completed assistant turn as the
**verbatim generated token span** (whole envelope: full body tokens, plus
producer reasoning text when the turn thought). A `length` terminal with a complete
call-less body (open-think length included) is stored the same way, under the same
content/tool fingerprint and producer reasoning; the template's per-turn `im_end`
framing is kept and no synthetic think close is inserted. Truncated or malformed
tool output, suppressed hidden completed calls (they cannot be described in the
echo key), stop-sequence, abort and fail-closed terminals are never stored. On the next turn, Jinja history
replay splices that span through the model's trained template framing so the
LCP prefix matches the prior bake. Unedited rich `reasoning_content` history
hits; edited or mismatched history falls back to a plain retokenized render
(cold or checkpoint path) instead of replaying stale tokens.

Tool argument replay is on by default (`HIPFIRE_QWEN4_TOOL_ARG_REPLAY=0` opts out).
It restores original producer argument ordering, not lexical sorting, on the last
echoed Qwen4 or dense Qwen assistant tool turn in the active committed conversation. Typed
values and byte-identical echoes are unchanged. A repaired render is accepted
only if it strictly extends every committed token; otherwise the original safe
miss remains. This does not enable retired-MTP continuation or cross-session reuse.

Multi-turn DFlash and the prefix cache: when a DFlash turn ends on EOS (or the
think cap) mid-window, **RepairForTerminal** restores the pre-window recurrent
state and replays only the consumed prefix so the prompt/prefix cache stays
warm. The next turn prefills only the new suffix instead of a full cold
prefill (the previous fail-closed path reset and invalidated the cache).

### Qwen4 PLE row tables

The Flash-Next (Qwen4) per-layer-embedding (PLE) n-gram table is 128 shard
tensors of `[2,500,012, 160]` Q8F16 rows (54.4 GB). A load can read it from
two interchangeable row files, chosen by `memory.ple_format`:

- **`<model>.ple-q8`** — the exact Q8F16 rows.
- **`<model>.ple-q4`** — 4-bit Lloyd-Max rows (28.8 GB) derived from the Q8
  rows. Each 32-weight block is an f16 scale plus 16 bytes of 4-bit labels;
  the decoded row equals the tested q4 table bit for bit.

`memory.ple_format` (env `HIPFIRE_PLE_FORMAT`):

- `auto` (default) — q4 on gfx1151 (the unified-memory Strix Halo), q8
  elsewhere. `auto` never derives and never downloads at load: when the
  preferred file is absent it uses the other available format and prints a
  note on stderr naming how to get the preferred one. The on-demand download
  of a missing sidecar resolves `auto` with the arch of the card the load
  targets (`hardware.devices`, else `ROCR_VISIBLE_DEVICES`, else ROCr agent
  order; the first detected GPU when the CLI cannot tell), while
  `hipfire pull --ple auto` fetches the union of every detected GPU arch's
  default, so a gfx1100 + gfx1151 host gets both q4 and q8 (q4 only when a
  verified grandfathered monolithic already embeds the q8 rows).
- `q4` — always 4-bit rows. A load never derives `<model>.ple-q4`: when it is
  missing the load is refused with both searched locations and the remedies
  `hipfire pull <tag> --ple q4` or `hipfire ple derive --format q4 <model>`
  (derives it from the local Q8 rows, about 1 minute, progress on stderr).
- `q8` — always exact rows (the kill switch for the q4 default on gfx1151). A split base needs `<model>.ple-q8`
  (`hipfire pull <tag> --ple q8`; `hipfire serve` downloads it on demand for
  registry models); a missing file is refused with that hint, never
  substituted.

**Sidecar lookup.** For every `<model>.ple-*` file the loader probes the model
path exactly as you selected it first, then the canonical target when that path
is a symlink (`<target>.ple-*`). A models directory that symlinks to an artifact
stored elsewhere therefore finds the sidecars published beside the link, or
beside the real file. A missing file is refused naming both locations searched.

The previous stock `qwen3.8:flash-next-mq4-xts` artifact, a monolithic file
(a grandfathered install; the stock MQ4 tags now name the split base
`qwen3.8-flash-next-base.mq4-xts`), keeps its PLE rows embedded: with `q8` (or
`auto` off gfx1151) the embedded Q8 rows are read exactly as before; `q4` adds
`<model>.ple-q4` next to it, which `hipfire pull` links from the shared q4 file.

**Split layout.** `hipfire ple split <model>` rewrites a monolithic model as
`<model-stem>-base.<ext>` (everything except the 128 PLE shards, about
70.9 GB, with a `hipfire_ple_rows` marker in its metadata) plus
`<model-stem>-base.<ext>.ple-q8` (the exact Q8 rows), e.g.
`qwen3.8-flash-next.mq4-xts` becomes `qwen3.8-flash-next-base.mq4-xts` and
`qwen3.8-flash-next-base.mq4-xts.ple-q8`. A sidecar is bound to its base
file: its header records the sha256 of the base's file header and tensor
index (the bytes before the data section) and the base's file size, and a
load refuses a sidecar whose binding does not match the base it sits next
to (delete it and re-derive, or re-pull). A `.ple-q4` derived from a base
is bound to that base the same way. A split base cannot load without the
sidecar for the selected format.

**MQ3 XTS split tier (`qwen3.8:flash-next-mq3-xts`).** This tier is a split
base (`qwen3.8-flash-next-base.mq3-xts`) that carries no PLE rows. Its row
tables come from sidecars chosen by the same `--ple q4|q8|auto` pick as every
split entry, with no tier-specific default. A plain `hipfire pull` (`--ple
auto`) follows `memory.ple_format`, whose own `auto` is q4 on gfx1151 and q8
elsewhere, so on other architectures a plain pull selects q8. To get the q4
rows on any architecture, pull them explicitly:
`hipfire pull qwen3.8:flash-next-mq3-xts --ple q4`. The exact Q8 rows are an
explicit opt-in the same way: `--ple q8` fetches `.ple-q8` and nothing else.

At load, `memory.ple_format = q8` (or `HIPFIRE_PLE_FORMAT=q8`) requires the
`.ple-q8` sidecar and refuses to load without it; it is never substituted
with q4. `auto` takes the architecture's preferred format if that file is
present and the other one if it is absent (it never downloads), so after a
q8-only pull `auto` loads the q8 rows.

To go back to q4 once a `.ple-q8` file is on disk, set
`memory.ple_format = q4` or `HIPFIRE_PLE_FORMAT=q4`, with the `.ple-q4`
sidecar present. Pulling with `--ple q4` does not remove the q8 sidecar and
does not change which format a later load selects, so with both files present
`auto` still follows the architecture's preference (q8 off gfx1151), not the
pull. Shared row files: the split bases of the Flash-Next family share one
published pair of row files in `hipfire-models/qwen3.8-flash-next`,
`qwen3.8-flash-next.ple-q4` (29,120,146,432 B) and
`qwen3.8-flash-next.ple-q8` (54,400,269,312 B). A split registry entry names
its row files in the `ple` map (`"ple": {"q4": {...}, "q8": {...}}`, keyed by
format token; this replaces the fixed `ple_q4` / `ple_q8` fields, and an
unknown key such as a future `q2` parses and is ignored by a client that
cannot load it) and may name these shared file names. The CLI then downloads
each shared file once and gives
every base a hardlink alias at that base's `<base>.ple-q4` / `<base>.ple-q8`
path, which is what the loader reads. A split base binds a sidecar by the base
marker's row geometry and the SHA-256 of the q8 source rows, not by the base's
file size and header/index hash, so a base with the same marker geometry and
q8 source rows can use the same shared file; a mismatch is refused. This
applies to split bases only: a monolithic model (a grandfathered MQ4 XTS install
included) keeps the file-size plus header/index binding described above and gets
no shared-file slot; the shared q4 that `hipfire pull` links beside such an
install as `<monolithic file>.ple-q4` binds the monolith natively. The MQ3 XTS
base file itself is not
published yet, pending its release gates. With the q4 rows the MQ3 XTS download
is about 88.37 GB (base plus `.ple-q4`), with the q8 rows about 113.65 GB; the
q4 rows are about 25.28 GB smaller on disk and in the download. The PLE rows are
read from the external file, not held in VRAM, so this is not a VRAM reduction.
No GPU gate result is claimed here.

`memory.ple_hot_store_bytes` (above) also applies to the sidecars: the
hot-row pack is cut from, and verified against, the rows file in use.
With the q4 sidecar and no hot-row pack attached, the sidecar's data is
read into the page cache after the model is ready, while no request runs
(a rank-ordered table hottest tier first); `memory.ple_prewarm_bytes` (env
`HIPFIRE_PLE_PREWARM_BYTES`) sets the budget: `auto` (default) is derived from
`MemAvailable` minus the CMA region minus 8 GiB, sampled once after the model
(pinned experts included) is loaded, capped at the sidecar's data bytes. On a
unified-memory GPU and on a discrete GPU with every expert resident in VRAM it
is additionally capped at 15 GiB (the smaller of the two); only a discrete GPU
with host-placed experts takes the uncapped figure. An unreadable
`MemAvailable` gives 0 (off). `0` = off, an integer = that many bytes (an
explicit override of the RAM derivation, never capped at 15 GiB, still capped
at the table). A rank-ordered table's
hottest prefix is already read during the load, throttled, by
`memory.ple_load_fill_bytes` (env `HIPFIRE_PLE_LOAD_FILL_BYTES`, `auto` =
14 GiB, `0` = off), so the first request after ready finds those rows
warm; the prewarm continues after it. The pages are ordinary clean page
cache, dropped (never swapped) under memory pressure and not charged in
admission.

**Prewarm yields to whole requests.** Every Qwen4 request, on any route
(ordinary and speculative generate, `serve`, `bench`, and every
continuous-batch lane), holds the PLE row store busy from admission to its
terminal event and releases it on every exit, cancellation and error
included. At each chunk boundary, both prewarm workers pause while any
request is in flight, host-expert compute gaps between PLE fetches included.
An already-started buffered read (at most 8 MiB) may drain; there is no
explicit request wait or I/O cancellation. The prewarm resumes 5 s after the
last request ended, from where it stopped. There is no setting for this.

**`auto` format policy is unchanged.** `memory.ple_format = auto` stays q4 on
gfx1151 and q8 elsewhere, and a split base already uses the `.ple-q4` sidecar
when it is the only one present (a load never derives it automatically). The
q8 default on other architectures, and the q8 monolith, are kept on purpose:
q4 is lossy, and the current q4 evidence is a gfx1151 WT2 point estimate (KLD
0.104865 q4 vs 0.103252 q8), not a paired confidence interval on the XTX
(`gfx1100`, code24). Moving that default would ship an unqualified
quantization change, and the xtx-rootcause findings show no PLE-format
performance bottleneck that would justify it.

### Qwen4 trunk overlay sidecar

`<model>.trunk-overlay` (for the stock XTS split base,
`qwen3.8-flash-next-base.mq4-xts.trunk-overlay`, ~305 MB) holds pre-encoded QT44
tensors that replace the base's MQ6 `linear_attn.in_proj_{z,a,b}` matrices at
load and arm the trunk A4 route default its manifest names
(`gfx1151-zab-v1`: the GDN Z, alpha and beta projections on exact gfx1151). It
is an HFQM container whose metadata records the base it was built for, and a
load binds it before using it: same architecture id, same base file size, same
sha256 over the base's header and tensor index (the bytes before the data
section), and every listed tensor present in the base with the recorded
quant type, size and shape. Each payload's sha256 is verified when it is read
for upload. A sidecar that fails any check, names an unknown default, is not
for a Flash-Next profile, or sits next to a base that already carries a trunk
default marker is refused with one `REFUSED` line on stderr and the base
weights run; an absent sidecar logs nothing. Like the PLE sidecars, it is looked
up next to the model file `hipfire serve`/`run` resolve (symlinks followed), so
a symlinked model needs its sidecar beside the link target.

Env `HIPFIRE_QWEN4_TRUNK_OVERLAY` selects the policy: `auto` (default; also
unset) applies the sidecar only on the architecture of its default (gfx1151),
`0` never applies it (kill switch: stock base weights, one log line), `1`
applies it on any architecture (the A4 route still arms by default only on
its exact architecture; elsewhere set `HIPFIRE_QWEN4_TRUNK_IU4`). Any other
value fails the load. `HIPFIRE_QWEN4_TRUNK_IU4=0` keeps the overlay weights
and disables the A4 route. `hipfire-quantize trunk-overlay` builds the file
from a directory of QT44 payloads (`--base`, `--qt44-payload-dir`, `--suffix`
(repeatable), `--trunk-iu4-default`, `--output`).

### Qwen4 expert hotset sidecar

`<model>.expert-hotset` (JSON, `kind` `hipfire.qwen4.expert_hotset`, version
1) is the decode-ranked per-expert VRAM placement of a Qwen4 aggregate-MQ4
base. On an exact gfx1201 discrete card it replaces whole-layer expert
placement by keeping the highest-ranked experts of every trunk layer in VRAM
and the rest in pinned host RAM. It is bound to its base by two required
fields, `base_size` (`u64`, the base file's length in bytes) and
`base_index_sha256` (lowercase hex sha256 of the base's HFQ header and
index); there is no whole-artifact hash. Like the trunk overlay it is looked
up next to the model file `hipfire serve`/`run` resolve.

Env `HIPFIRE_QWEN4_EXPERT_HOTSET` (`developer.qwen4_expert_hotset`) selects
the policy: unset (default) applies `<model>.expert-hotset` when it exists
(exact gfx1201 only), `0` never applies it (kill switch: whole-layer
placement), an absolute path names the sidecar explicitly and applies on any
architecture, and a relative or empty value is a policy error.
A sidecar that does not apply (absent, unreadable or malformed, bound to a
different base, dimensions that do not match the model, the unset default on
an architecture other than gfx1201,
unified memory, expert parallelism, or an explicit
`HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS`) keeps whole-layer placement and never
fails the load. The expert byte budget is derived at load, never read from the
file: free VRAM less the non-expert weights, the `auto` reserve, the frozen
expert tables and a 512 MiB margin.

---

## Shared-weight host (developer, Linux, default off)

Two hipfire processes — typically two different builds with different
kernels — can decode at the same time on one card from **one** imported copy
of a model's weights. A separate process, `hipfire-weight-host`, loads the
weights once with the ordinary loader, turns every final weight owner into a
read-only, exportable VMM allocation, and leases the allocation file
descriptors to tenant daemons over a private Unix socket. Each tenant maps
them read-only and keeps its own KV, DeltaNet state, scratch, graphs and
kernels. Implementation: `crates/hipfire-daemon/src/bin/hipfire-weight-host.rs`
(host) and `weight_host_tenant.rs` (tenant); the validation route is in
[`VALIDATION.md`](VALIDATION.md) ("Shared-weight host import").

There are no TOML keys. The switches are bootstrap environment variables,
read before the process config exists because they select the GPU claim:

| Variable | Values | Effect |
|---|---|---|
| `HIPFIRE_WEIGHT_HOST` | unset / `0` / `1` | `1` opts a daemon in: weights are imported, never loaded privately. Unset or `0` (the kill switch, which wins over any socket setting) is the incumbent private load and exclusive claim, unchanged. Any other value refuses to start. `0` also refuses to start `hipfire-weight-host`. |
| `HIPFIRE_WEIGHT_HOST_SOCKET` | absolute path | Required with `HIPFIRE_WEIGHT_HOST=1`: the host's socket. |
| `HIPFIRE_FABRIC_ADMISSION` | `<state json>#<entry id>` | Admission token exported by a developer GPU-admission wrapper (not shipped in this repository) for a correct-class job. Required by the host and every tenant; same-user trust for development, not security isolation. |

**Envelope (v1).** Linux, exact `gfx1201`, one visible card, a dense Qwen3.5
HFQ (arch 5), `tp = pp = 1`, text only. The load must send
`mtp_mode`, `dflash_mode` and `dspark_mode` as `"off"` (the CLI's
`speculation` selector `off` does) and no draft, drafter, prefill drafter,
head overlay or vision tower; the experimental slot engine is refused.
Anything else is refused before the prior model is torn down — there is no
private-load fallback while opted in. Retained replay is not part of v1: use
`HIPFIRE_REPLAY_BACKEND=hip` (with `HIPFIRE_GRAPH=0` or `1`).

**Running it.** Host and tenants each run as their own admitted job on the
same card, with the admission wrapper declaring each job's peak memory: the
host about the weight set plus the largest owner during conversion (≈17 GB for
the 27B), each tenant only its private state. Without such a wrapper there is
no way to obtain the admission token, so the shared claim is refused.

```
mkdir -m 0700 /path/wh
# host, under the admission wrapper (exports HIPFIRE_FABRIC_ADMISSION):
hipfire-weight-host --model /models/qwen3.8-27b.mq4-xts --socket /path/wh/host.sock
# each tenant, under the same wrapper, with its own build and HIPFIRE_KERNEL_CACHE:
HIPFIRE_WEIGHT_HOST=1 HIPFIRE_WEIGHT_HOST_SOCKET=/path/wh/host.sock \
  <driver of target/release/daemon>
```

`hipfire-weight-host` flags: `--model`, `--socket` (its directory must exist,
belong to you and be mode 0700), `--drain-timeout-secs` (default 600),
`--handshake-timeout-ms` (default 10000). It hashes the read-only source once
and writes `<socket>.source-cert` (mode 0444) beside the socket; tenants only
`fstat` the model file against it. It prints `WEIGHT_HOST_READY epoch=…
layout_sha256=… unique_bytes=…` when the socket is published, one
`WEIGHT_HOST_LEASE` line per lease transition, and `WEIGHT_HOST_CLOSED` after a
clean drain. A tenant prints `WEIGHT_HOST_IMPORT epoch=… layout_sha256=…
imported_bytes=… ack=ok` and, on every Qwen3.5 load, `QWEN35_LOAD_PHASES
route=<private|import> weights_ms=… kv_ms=… dn_ms=… scratch_ms=…`.

**Admission and locking.** Host and tenants take a *shared* `flock` on the
machine-wide `/run/lock/hipfire/gpu-<id>.lock` (the file exclusive daemons
lock), never write it, and write no `~/.hipfire/daemon-*.pid`. A shared claim
is accepted only when `HIPFIRE_FABRIC_ADMISSION` names a running correct-class
entry, the process is in that job's session, and the admission wrapper holds
that card's lock shared. The host also checks each tenant's token (by `SO_PEERCRED`
pid) and that it is for the same card. A tenant whose card, source digest,
loader ABI or layout recipe differs is rejected before any descriptor is sent.

**Identity.** The loader ABI digest is generated by
`crates/hipfire-arch-qwen35/build.rs` from the full source of the loader
closure (`hfq.rs`, `model_load.rs`, `weight_backend.rs`, `llama.rs`,
`weight_host.rs`, Qwen3.5 `config.rs`/`load.rs`/`weights.rs`/`shared_weights.rs`),
the `DType` enum and top-level constants of `rdna-compute/src/dispatch.rs`,
the `rustc -vV` output, target, opt-level and enabled features. Forward and
JIT kernel sources are not inputs, so two builds that differ only in kernels
share a generation; a change to any input refuses import until the host is
rebuilt from the same loader.

**Allocation layout.** One physical allocation per unique final owner (1,347
for the 27B). Each is `round_up(logical + 64 KiB, minimum granularity)` bytes,
with the bytes past the logical end zeroed before the allocation is sealed
read-only. The 64 KiB tail exists because forward GEMV/attention kernels read
a few bytes past a tensor (a phantom group header or a vector tail load, see
`HOST_TAIL_PAD_BYTES` in `rdna-compute/src/dispatch.rs`): a private
`hipMalloc` absorbs that in allocator slack, an exact-fit VMM mapping faults.
64 KiB is far beyond one group header and costs about 88 MB for the 27B.
Granularity is the runtime *minimum* (4 KiB on gfx1201, exercised by G0)
rather than the recommended 2 MiB: for 1,347 owners the per-owner rounding to
2 MiB adds 2.32 GB (2.89 GB with a 1 MiB tail) against 0.69 MB of rounding at
4 KiB (S3 owner census: 15,008,626,688 logical bytes, 15,097,597,952 physical).
Small-page mappings may cost TLB reach in weight-streaming decode; the G3
timing diagnostic compares import against private decode before that is
claimed either way.

**Lifecycle.** The host leases a generation identified by a random 128-bit
epoch. A tenant acknowledges an import only after mapping it, and sends
`Release` only after its unload freed the imported owners, dropped graphs and
reported clean VMM teardown; otherwise the lease closes unacknowledged and the
host keeps it charged until that tenant's process (pid and start time) is
gone. On SIGTERM/SIGINT/SIGHUP the host stops admission (unlinks the socket),
half-closes every lease, waits until every lease is released or its process
has exited, frees the generation and exits 0; past `--drain-timeout-secs` it
exits 3 and the generation stays occupied by the listed tenants (their own
mappings keep it alive). A tenant whose host went away unloads the imported
model before its next command and fails requests until reloaded against a live
host; there is no transparent rebind.

---

## Chat template overrides

| Key | Default | Validation |
|---|---|---|
| `chat_template` | `""` | empty or existing file path (`~/` expanded); existence + `isFile` only — no readability/access check |
| `default_chatml` | `true` | bool |

Lowered directly into the startup/load policy. `HIPFIRE_CHAT_TEMPLATE_FILE`
and `HIPFIRE_DEFAULT_CHATML` remain legacy one-shot aliases.

### Opt-in community Qwen template

The official Qwen chat template stays the default. The community
[froggeric Qwen chat template](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates)
(v22.5, Apache-2.0) can be used as an opt-in override. Download its `.jinja`
file, then pick one of:

- `hipfire config set prompt.chat_template <path>`
- `HIPFIRE_CHAT_TEMPLATE_FILE=<path>` in the environment
- `~/.hipfire/templates/<model-basename>.j2` for a single model (used when no
  configured path is set)

It changes the prompt, not the engine: default reasoning effort is medium,
reasoning history is preserved by default, `<|think_on|>` / `<|think_off|>`
toggle thinking per message, and the tool prompt carries extra guidance.

Overriding the template changes prompt tokens. That can move speculative-decode
acceptance and task scores, so re-measure before comparing against numbers taken
with the official template.

---

## Additional schema keys

Remaining typed keys that are not covered by a section above. Defaults and
ranges are the schema's; every key also appears in the generated
[lifecycle table](#lifecycle-status) with its legacy spelling and env alias.

| Key | Default | Values / notes |
|---|---|---|
| `memory.gpu_layer_budget` | `null` | `null`, `-1` (`auto`) or int 0–65536. Resident-layer budget for partial GPU offload: `N` keeps the **last N layers on the GPU** and spills the rest to system RAM (3 on a 64-layer model spills 61); unset keeps every layer on the GPU; `-1` defers placement to the engine, which currently keeps every layer on the GPU. Env `HIPFIRE_GPU_LAYER_BUDGET`. |
| `memory.offload_exec` | unset (`"pcie"`, except the measured Flash-Next D2 default) | `pcie` \| `cpu` \| `passback`. Which engine executes the ops that read a spilled layer's weights: GPU kernels over the link (`pcie`) or the CPU. Never changes placement; the KV cache stays in VRAM. On Qwen3.5-family spills `cpu` runs the spilled GEMVs and routed-expert FFNs on the host; on Flash-Next (qwen4) it applies only to host-placed routed experts, for decode and `<=4`-row verify (wide prefill stays PCIe), not bit-identical. Unset keeps `pcie` except for the per-load Flash-Next D2 default (hot-set tier layers with host experts, CPU engine measured strictly faster than the host link; see the D2 measured default bullet above); an explicit value, `pcie` included, always wins. Env `HIPFIRE_OFFLOAD_EXEC`. |
| `model.deepseek4_experts_per_token` | `null` | `null` or int 1–6; DeepSeek V4 routed experts per token (`null` keeps the checkpoint default). |
| `hardware.deepseek4_compute_placement` | `"single"` | `single` or `dense-expert-split(dense=<selector>,experts=<selector>)` with two distinct device selectors. |
| `serve.vmm_batch` | `false` | bool; route eligible concurrent requests through the VMM continuous-batching executor (per-request VMM KV, one resident target). Off keeps the VMM singleton route. Env `HIPFIRE_SERVE_VMM_BATCH`. |
| `serve.continuous_batch_size` | `1` | int 1–256; number of eligible concurrent batch lanes (`1` = sequential), subject to memory admission. The runtime batch width is dynamic: it follows the lanes in flight and is not fixed at this value, and the 128-row/63-token chunk sizes are not lane limits. Env `HIPFIRE_CONTINUOUS_BATCH_SIZE`; the `serve --continuous-batch-size` flag takes the same range. |
| `serve.batch_spec` | `true` | bool; cross-request speculative verify rows on the VMM batch route (`false` disables batched speculation only). |
| `serve.batch_nonexact` | `false` | bool; permit explicitly labeled non-bit-exact batched arithmetic on the VMM batch route. |
| `serve.batch_sampled` | `true` | bool, applies only with `serve.vmm_batch`; admits sampled requests to the batch route only in the lane mode the singleton route selects for them. Flash-Next fn-lanes admit sampled requests with neutral penalties (AR lanes draw with the singleton AR sampler and request seed; MTP lanes verify naively, so seeded output equals the AR singleton's where logits match). `false` keeps them on the serialized singleton route. |
| `serve.ui` | `false` | bool; serve the embedded chat UI at `/ui` on the serve listener. |
| `serve.retry_enabled` | `false` | bool; server-owned single retry on typed transient daemon failures. |
| `serve.retry_backoff_ms` | `50` | int 0–60000; backoff before that retry. |
| `prompt.jinja_chat` | `true` | bool; render chat prompts through the model Jinja template (`HIPFIRE_JINJA_CHAT=0` selects the hand-rolled scaffold). |
| `image.decode` | `"cpu"` | `cpu` \| `vcn` \| `auto` (VCN when probed, else CPU); VL image JPEG decode path. |
| `prefill.chunk_rows` | `512` | int 2–1048576; widened ordinary-prefill chunk ceiling in rows. The registered `512` is a schema fallback only: it is not exported to the process config, so it is not a global runtime default, and an unset value resolves per model. Architecture default 8192 on exact gfx1100/gfx1151/gfx1201, 512 elsewhere (the Qwen35 dense 27B widened route); per-device VRAM admission may pick a smaller rung. **Qwen3.5-class resident MoE (A3B/A10B/A17B):** the widened route used to admit only the dense 27B shape, so MoE kept the incumbent per-arch chunk (512 on exact gfx1100, 384 or 512 under the full FP8 path on exact gfx1201, 256 elsewhere). Those are the dense MQ4V2 gate-up sweet spots measured on those cards and 512 matches the dense 512-row commit seam; they are not a measured MoE optimum. Unset, a resident MoE now defaults to one 4096-row chunk (never an intermediate 1024/2048 rung) only when the model is fully resident on one device, every layer passes batched-prefill producer admission, the routed and shared experts are uniform MQ4G256 or MQ4G256V2, the DeltaNet state is Q8 with its EF owners, and the device has the WMMA grouped producer, and only when the projected full prefill scratch (grouped MoE scratch plus the DeltaNet `dn_z_fold_batch`/`dn_normed_batch` buffers the legacy projection omits, the projection prelude, and the existing 1 GiB transient/driver headroom over requested tensor payload) fits the actual free VRAM; otherwise the incumbent chunk applies. This is a default-admission change onto the existing explicit `HIPFIRE_PREFILL_MAX_BATCH=4096` path (the j1405 reference set both `HIPFIRE_PREFILL_MAX_BATCH=4096` and `HIPFIRE_PREFILL_CHUNK_ROWS=4096`; `prefill.chunk_rows` alone never raised the legacy MoE ceiling); no MoE kernel or commit code changed, and the receipt's generic `commit_stride=512` does not prove the MoE kernels commit per 512 rows (the dense 512-row commit is unchanged). Explicit values keep precedence: `HIPFIRE_PREFILL_MAX_BATCH` wins and leaves the incumbent; a `prefill.chunk_rows` below 4096 opts the MoE out; 4096 or more requests the single 4096 rung. The log line `prefill_chunk: requested=R admitted=A commit_stride=S` records the decision. 8192 was measured and rejected as the MoE default: it lost four same-hold comparisons to 4096 (-0.03% to -1.80%, 3 samples per shape) and raised peak VRAM by about 2.1 GiB. Offloaded, TP, EP, dense and Qwen4 loads are unaffected. Qwen4 rounds it down to 256 rows. Qwen4 sizes the auto expert placement reserve (and so how many expert layers stay in VRAM) from this same chunk: the explicit value, else 8192 on exact gfx1151, 4096 on exact gfx1201, and 1536 elsewhere. Exact gfx1100 on a measured narrow discrete host link (< 16 GB/s, probed once at load before placement is sized; see `developer.qwen4_expert_stage`) defaults to 8192 rows, so its placement is sized for the 8192-row reserve and cedes resident expert layers for the wider chunk; gfx1100 on a fast, unmeasured or unified-memory link keeps 1536, and no other arch probes. An explicit value is authoritative for placement and never probes: `HIPFIRE_PREFILL_CHUNK_ROWS=1536` opts out of the narrow-link default, and `=8192` charges the 8192-row reserve on any link (one gfx1100 XTX hold, single run, measured pp8192 931.7 vs 90.1 tok/s = 10.3x, but tg64 5.09 vs 5.71 tok/s = -11% and pp512 62.0 vs 72.2 tok/s = -14% against the 1536-row placement, with 5 instead of 10 resident layers). Env `HIPFIRE_PREFILL_CHUNK_ROWS`. |
| `fusions.policy` | `"safe"` | `safe` \| `off`; certified fusion policy (individual kernel selection stays compiled). |
| `attention.ck_runtime_lib` | `null` | `null` or path; optional exact-architecture CK runtime sidecar (load or capability failure keeps native attention). |
| `attention.ck_workspace_bytes` | `0` | int 0–17179869184; preallocated bytes for the optional CK staged attention path. |
| `experimental.graph.forward` | `"auto"` | `auto` \| bool; hipGraph forward capture (`auto` follows architecture policy). Env `HIPFIRE_GRAPH`. |
| `experimental.graph.ar` | `true` | bool; allow autoregressive forward graph capture when otherwise eligible. **Windows gfx1100 default is off (opt-in)**: AR graph replay silently produced wrong output there (token soup, #655), so an unset value resolves to `false` on Windows exact gfx1100 only; Linux gfx1100 and every other architecture keep `true`. An explicit `true`/`false` (config, `hipfire config set`, or env `HIPFIRE_AR_GRAPH`) always wins. Env `HIPFIRE_AR_GRAPH`. |
| `experimental.graph.moe` | `true` | bool; allow graph capture for supported MoE forward paths. Env `HIPFIRE_GRAPH_MOE`. |
| `kernel.calib_force_bf16` | `false` | bool; keep native-BF16 calibration teachers in BF16 (calibration only). |
| `kernel.gemma4_q8_fused_prefill` | `false` | bool; fuse eligible Gemma 4 Q8 prefill projections on gfx1100. |
| `kernel.gemma4_ple_batched_prefill` | `false` | bool; batch the Gemma 4 E-series PLE model projection on gfx1100. |
| `kernel.gemma4_batched_embedding_prefill` | `"auto"` | `auto` \| bool; `auto` enables validated gfx1100/gfx1201. |
| `kernel.gemma4_ple_activation_fused_prefill` | `"auto"` | `auto` \| bool; `auto` enables validated gfx1100/gfx1201. |
| `memory.unsafe_wsl_vmm_kv` | `false` | bool. UNSAFE until certified: allow the VMM KV backend under WSL/ROCDXG, where WDDM VA growth may alias earlier KV pages. Off: automatic KV selects legacy and an explicit `kv_backend = "vmm"` is refused. |
| `replay.unsafe_wsl_redline` | `false` | bool. UNSAFE until certified: allow Redline PM4/retained replay under WSL/ROCDXG. Off: the retained default falls back to the HIP graph and an explicit `replay.backend = "redline"`/`"shadow"` is refused. |

The Qwen dense uniform `MQ4G256V2` Exact route also admits exact gfx1100
with Q8 VMM KV. On gfx1100 each target lane executes the complete singleton
layer program; target projections are not merged across requests. This first
route supports AR lanes only: cross-request DFlash/MTP verification is not
installed, while the existing singleton speculative path remains available.
gfx1201 retains its batched projection and speculative-lane routes.
Q8 recurrent state requires the default-on error feedback; with it disabled,
gfx1100 Exact serve staging keeps the existing singleton route because the
legacy stochastic state's singleton hipGraph is not byte-stable.

### Architecture-gated default-on kernel routes

All are process booleans that default to `true` and take effect only on the
architecture named (other architectures keep their prior route); opt out with
`false` or the `HIPFIRE_*` spelling set to `0` (see the lifecycle table for the
exact name). They are snapshotted at GPU initialization, so restart an existing
daemon after changing them.

| Scope | Keys |
|---|---|
| Exact gfx1201 | `kernel.gfx12_mq4v2_fp8_gateup`, `kernel.gfx12_mq4v2_fp8_resid`, `kernel.gfx12_mq4v2_fp8_qkvza`, `kernel.gfx12_mq4v2_fp8_qkv` (the four family flags), `kernel.gfx12_mq4v2_fp8_v2` (staged-tile v2; the four family flags remain prerequisites), `kernel.gfx12_gdn_pre_fused`, `kernel.gfx12_silu_quant_fused`, `kernel.gfx12_producer_quant_fused`, `kernel.g12_norm`, `kernel.g12_a4c2`, `kernel.gfx12_fa2_prefill`, `kernel.gfx12_fa_packet`, `kernel.attn_qresident` (exact H24/KV4/D256 native-fp8-KV shapes), `kernel.attn_qresident_v2` (only where the Q-resident route is selected), `kernel.gfx12_fa_prep_fused`, `kernel.gfx12_fa_prep_fp8q` |
| Exact gfx1100 / gfx1151 / gfx1201 | `kernel.gfx12_gdn_chunk_scan`, `kernel.iu4_prefill`, `kernel.verify_attn` |
| Exact gfx1100 / gfx1151 | `kernel.gfx11_fa2_prefill`, `kernel.gfx11_producer_quant_fused`, `kernel.gfx11_lean_pbs`, `kernel.gfx11_iu4_symfold` (symmetric artifacts only) |
| Exact gfx1201 / gfx1151 | `kernel.g12_dec_norm` |
| Exact gfx1100 | `kernel.gfx1100_dec_norm`, `kernel.gfx11_iu4_shape` |

`kernel.gfx11_a4_candidates` (default `"2"`; `"1"`, `"2"`, `"4"` or `"8"`) sets
the number of activation-quantization candidates searched inside the fused
gfx1100/gfx1151 IU4 producers; `"8"` restores the full-grid search.

`kernel.gfx11_iu4_gridspec` (default `true`; opt out with `false` or
`HIPFIRE_GFX11_IU4_GRIDSPEC=0`) splits partial-N gfx11 IU4 GEMMs so full
interior tiles use the unchecked specialization and only the final tile uses
guarded loads and stores.

### Opt-in gfx11 8-bit activation prefill

`kernel.gfx11_a8_prefill` (default `false` on every architecture; opt in with
`true` or `HIPFIRE_GFX11_A8_PREFILL=1`, exact gfx1100/gfx1151 only) moves the
MQ4V2 prefill projections from IU4 W4A4 to W4A8: the fused producers emit int8
per-128 activations and an iu8-WMMA GEMM consumes them. It applies to
symmetric MQ4V2, eager prefill (no Redline recording or graph capture) with
batch >= 64 and K % 256 == 0, and takes precedence over IU4 there; everything
else stays on IU4. On Strix Halo (gfx1151, qwen3.8-27b) 8-bit activation
prefill gives about 43% lower KLD (WT2 0.0394 against 0.0695), and prompts
process about 1.5–1.6× slower (pp8192 604 against 974 tok/s, pp32000 521
against 777 tok/s). Enable it with `hipfire config set kernel.gfx11_a8_prefill true`
or `HIPFIRE_GFX11_A8_PREFILL=1`; with the key off, prefill is unchanged.

### Retained replay

Policy and certification: [`REDLINE.md`](REDLINE.md). `replay.backend`
(default `"auto"`; `auto` \| `hip` \| `redline` \| `shadow` \| `off`, env
`HIPFIRE_REPLAY_BACKEND`) selects the launch backend; the runtime default is
distinct from certification or admission. `replay.transport` (default `"auto"`;
`auto` \| `aql` \| `pm4` \| `pm4_ib` \| `ib`, env `HIPFIRE_REPLAY_TRANSPORT`)
selects the retained transport. `replay.pm4_gfx11_vmem_acquire` (default
`"auto"`, which selects gfx1151) and `replay.gfx1201_pm4_pacing` (default
`"auto"` = a 64-dword NOP after every dispatch; `off`/`0`; `nop:N`) are process
keys. The remaining retained-replay controls are developer diagnostics:

| Key | Default | Values |
|---|---|---|
| `diagnostic.replay.route_proof_log` | `false` | bool; one post-generate retained-route proof marker per request |
| `diagnostic.replay.manual_capture` | `false` | bool; arm recording manually instead of the model lifecycle |
| `diagnostic.replay.pool_debug` | `false` | bool; report the pool backing the retained indirect buffer |
| `diagnostic.replay.dispatch_profile` | `false` | bool; per-dispatch GPU clock write (changes the tape identity) |
| `diagnostic.replay.pm4_min_parallel_width` | `2` | int 2–1024 |
| `diagnostic.replay.pm4_min_parallel_workgroups` | `0` | int ≥ 0 |
| `diagnostic.replay.pm4_max_parallel_phases` | `null` | `null` (unlimited) or int ≥ 0 |
| `diagnostic.replay.pm4_native_phases` | `false` | bool |
| `diagnostic.replay.pm4_queues` | `"1"` | `"1"`, `"2"`, `"4"`, `"auto"` |
| `diagnostic.replay.pm4_register_policy` | `"static"` | `legacy` \| `static` \| `stateful` |
| `diagnostic.replay.pm4_kernarg_pool` | `"vram"` | `vram` \| `host` |
| `diagnostic.replay.pm4_wait_policy` | `"resource"` | `allowlist` \| `resource-audit` \| `resource` |
| `diagnostic.replay.pm4_acquire_policy` | `"required-only"` | `conservative`, `entry-only`, `required-only`, `without-repeat-interleave`, `without-fused-silu-rotate`, `without-mq-rotate`, `without-rope` |
| `diagnostic.replay.pm4_gcr_trim` | `true` | bool |
| `diagnostic.replay.pm4_dynamic_grid` | `false` | bool |
| `diagnostic.replay.gfx1151_initiator` | `"legacy"` | `legacy` \| `order` \| `radv` |
| `diagnostic.replay.gfx1151_interleave` | `"inherit"` | `inherit`, `off`, `64`, `128`, `256`, `512` |
| `diagnostic.replay.gfx1151_resource_limits` | `"legacy"` | `legacy` \| `simd-always` \| `radv` |
| `diagnostic.replay.gfx1151_cu_count` | `"all"` | deprecated; `all` only (`inherit`/`40` accepted as aliases). Any other value is refused at PM4 replay preparation: CU masks on RDNA hung the R9700 MES on 2026-10-14 |
| `diagnostic.replay.gfx1151_entry_acquire` | `"system"` | `system` \| `agent` \| `vmem` \| `none` |
| `diagnostic.compiler.no_device_compiler` | `false` | bool; treat the device compiler as absent so precompiled code objects are used verbatim |

---

## Decide calibration

Per-model temperatures for `POST /v1/systemone` answers (spec
`docs/specs/2026-09-28-jev-decide-design.md` §13). An answer's
probabilities are `softmax(label logits / T)`, with one `T` per question
type.

| Key | Default | Range | Scope |
|---|---|---|---|
| `decide.calibration.choice` | `1.0` | 0.05–20 | request (per-model overlay) |
| `decide.calibration.score` | `1.0` | 0.05–20 | request (per-model overlay) |
| `decide.calibration.noul` | `1.0` | 0.05–20 | request (per-model overlay) |

`1.0` is the raw readout. T > 1 makes answers less confident and T < 1 more
confident; the chosen option never changes. Set them per model, from a fit
(`scripts/jev_eval/calibrate.py fit`). This is the actual shipped fit for
`qwen3.5-4b.mq4` (`bench/jev/qwen3.5-4b/calibration.json`; all three question
types cleared the ship rule, spec §13.4):

```toml
[models."qwen3.5-4b.mq4".overrides.decide.calibration]
choice = 1.653
score = 1.48
noul = 1.363
```

or `hipfire config qwen3.5-4b.mq4 set decide.calibration.choice 1.653`. Serve
resolves them on every decide, so a change applies without a reload. They
are not in the Jev response body; a non-identity calibration shows in the
`x-hipfire-timing` header. A hipfire binary built before these keys existed
refuses a `models.toml` that contains them.

---

## Tuning presets

`hipfire config tuning` trades speed for answer quality in one step, per GPU.
A preset is a bundle of **global** `config.toml` writes (never `models.toml`);
nothing is written until you confirm.

```bash
hipfire config tuning                       # open the Settings TUI on the preset chooser
hipfire config tuning status [--json]       # which preset is applied, which keys it owns
hipfire config tuning show quality-first    # preview: lines, measured trade-offs, net change
hipfire config tuning show quality-first --arch gfx1201 --json
hipfire config tuning set quality-first     # preview, then [y/N]
hipfire config tuning set quality-first --skip flash-next-ple --yes
hipfire config tuning undo [--yes]          # single level
```

Presets: `balanced`, `quality-first` (alias `quality`), `low-power`. Line ids,
for `--skip`: `prompt-processing`, `flash-next-experts`, `flash-next-ple`,
`shorter-thinking`. `set` and `undo` print the preview and ask `[y/N]` on a
terminal. Without a terminal they need `--yes`; otherwise they print `error:
refusing to change config without confirmation; pass --yes`, exit 2 and write
nothing. `--arch` only previews another GPU's plan; `set --arch X` is refused
unless X is this machine's GPU (`install.json`). `--json` carries `lines[]`
(`id`, `keys`, `from`, `to`, `ticked`, `requirement`, `evidence`) and
`net_diff`.

| Line | gfx1151 (Strix Halo) | gfx1201 (R9700 / RX 9070) | gfx1100 (RX 7900) |
|---|---|---|---|
| **Quality-first** · Prompt processing (dense MQ4V2) | `kernel.gfx11_a8_prefill = true` (the opt-in W4A8 route; IU4 W4A4 stays the default) | `kernel.iu4_prefill = false` (native fp8) | `kernel.iu4_prefill = false` (A8 is not offered: no gfx1100 quality run) |
| **Quality-first** · Flash-Next experts | `developer.qwen4_moe_sym_iu4 = false` | `developer.qwen4_moe_sym_iu4 = false` | nothing |
| **Quality-first** · Flash-Next word table | `memory.ple_format = "q8"`, ticked only when every installed Flash-Next split base has its `.ple-q8` (or the shared `qwen3.8-flash-next.ple-q8`); a monolithic embeds the q8 rows | nothing (auto is already q8) | nothing |
| **Low-power** · Shorter thinking | `reasoning.effort = "low"` | same | same |
| **Low-power** · Prompt processing | nothing (IU4 W4A4, the fastest prompt math, is already the default) | nothing | nothing |
| **Balanced** | writes nothing; gives back the keys the applied preset still owns | same | same |

Any other GPU gets only arch-neutral lines and the note "no measured quality
switches for <arch> yet". The Prompt processing line is read from the same
choices as the Settings virtual row (`hipfire_config::effective::
quality_choices_for`). `kernel.gfx11_a8_prefill` is detected by schema
(`hipfire_config::field("kernel.gfx11_a8_prefill")`), never by version: were its
default ever turned on, Quality-first would write nothing for prompts on
gfx1151 and Low-power would write it `false`. Without the key, Quality-first
writes `kernel.iu4_prefill = false` instead. The
sym IU4 key is a typed boolean and is written as a TOML boolean. "Shorter
thinking" relies on a chat request without `reasoning_effort` forwarding the
global `reasoning.effort` to an effort-native model (pinned by a `hipfire-cli`
test).

Every line shows its measured trade-off from `hipfire_config::evidence` (for
example "43% closer (WT2 KLD 0.0695 -> 0.0394), prompts 1.6× slower, typing
unchanged"); an unmeasured number is said to be unmeasured. Not changed by any
preset: KV cache (fp8 vs exact within noise on Flash-Next), speculation
(designed lossless), and context, devices, offload and model choices.

**In the TUI.** `p` on the Settings tab (or bare `hipfire config tuning`,
which launches `hipfire-tui --tuning`) opens the chooser: Balanced,
Quality-first and Low-power with a one-line description each and `●` on the
applied one. `Enter` opens the preview, which lists every line with `[x]` /
`[ ]`, its `from → to` change and the measured numbers from
`hipfire_config::evidence` ("not measured on this GPU" when absent). `Space`
ticks or unticks a line; a line whose requirement is unmet (a missing
`.ple-q8`) cannot be ticked and `Space` says why and what fixes it. The Checks
section flags an environment variable that overrides a written key, a
`models.toml` value that shadows it, and a running serve that still has the old
values. `y` writes exactly the ticked net diff and records ownership in
`tuning.toml`; `Y` does the same and then restarts serve (the Dashboard `R`
path, no second prompt). `config.toml` is byte-identical until `y`. Balanced
lists the keys it gives back. `u` (Settings or the chooser) shows the keys
undo restores and `y` applies it. At 160×48 and up the three presets are cards
over one full-width preview table (`←/→` switch preset). The Settings header
reads `Tuning: <preset>` (`Quality-first (edited)` after a hand edit).

**State.** `~/.hipfire/tuning.toml` (a control-plane file beside `config.toml`,
never inside it) records what the applied preset wrote, so the Settings "From"
column can say `preset` and switching is exact:

```toml
schema_version = 1

[applied]                       # what is in effect now
preset = "quality-first"
arch = "gfx1201"
applied_at = 1760630000

[applied.set]                   # keys this preset owns, exact values
"kernel.iu4_prefill" = false
"developer.qwen4_moe_sym_iu4" = false

[applied.base]                  # what to give back when the preset leaves
"kernel.iu4_prefill" = { unset = true }

[undo]                          # restores the state before the last action
previous_preset = "balanced"
[undo.prior]                    # per touched key: value before the action
"kernel.iu4_prefill" = { unset = true }
[undo.after]                    # per touched key: value the action left
"kernel.iu4_prefill" = { value = false }
```

Applying a preset first gives back the keys the old preset still owns (a key
you edited since is kept and reported), then merges the ticked lines;
re-applying the same preset is a no-op that keeps your undo. `undo` restores
each touched key that still has the value the action left, or removes it, and
hands ownership back to the previous preset. `status` reads `Balanced`,
`<Preset>`, `<Preset> (edited)` (an owned key changed or went away) or `Custom`
(no preset owns anything, but a key a preset manages was set by hand to a value
no preset writes, such as `reasoning.effort = "high"`; a hand-set value a
preset would also write, such as `"low"`, still reads `Balanced`). The
write order is `tuning.toml` to a temp file, then `config.toml`, then the
rename, so an interrupted run never leaves ownership claiming keys that were
not written. The preview warns when an environment variable overrides a key the
preset writes, or a `models.toml` entry sets it for one model.

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
| `vision_mode` | `vision.mode` | `HIPFIRE_VISION_MODE` | `"auto"` |
| `draft_f16` | `speculation.draft_f16` | `HIPFIRE_DRAFT_F16` | true unless `0` |
| `draft_gemm_dump` | `diagnostic.draft_gemm_dump` | `HIPFIRE_DRAFT_GEMM_DUMP` | off unless `1` |
| `draft_subphase` | `diagnostic.draft_subphase` | `HIPFIRE_DRAFT_SUBPHASE` | off unless `1` |
| `ddtree_budget` | `speculation.ddtree_budget` | `HIPFIRE_DDTREE_BUDGET` | 0 (chain DFlash; the flag reader treats `0` as unset) |
| `ddtree_topk` | `speculation.ddtree_topk` | `HIPFIRE_DDTREE_TOPK` | 4 |
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
| `session_cache_bytes` | `memory.session_cache_bytes` | `HIPFIRE_SESSION_CACHE_BYTES` | 8589934592 (8 GiB; `0` off) |
| `ple_hot_store_bytes` | `memory.ple_hot_store_bytes` | `HIPFIRE_PLE_HOT_STORE_BYTES` | 0 (off; `auto` = prewarm `MemAvailable` − CMA region − 4 GiB of the hot-row pack) |
| `ple_format` | `memory.ple_format` | `HIPFIRE_PLE_FORMAT` | `auto` (q4 on gfx1151, q8 elsewhere; see [Qwen4 PLE row tables](#qwen4-ple-row-tables)) |
| `ple_prewarm_bytes` | `memory.ple_prewarm_bytes` | `HIPFIRE_PLE_PREWARM_BYTES` | `auto` (prewarm `MemAvailable` − CMA region − 8 GiB of the q4 PLE table, sampled after the model is loaded and capped at the table; also capped at 15 GiB on unified memory and on a discrete GPU with every expert in VRAM, uncapped only with host-placed experts on a discrete GPU; unknown → 0; an explicit N is never capped at 15 GiB; `0` off) |
| `ple_load_fill_bytes` | `memory.ple_load_fill_bytes` | `HIPFIRE_PLE_LOAD_FILL_BYTES` | `auto` (read the hottest 14 GiB of a rank-ordered q4 PLE table during load at ≤ 0.2 GB/s; `0` off) |
| `session_cache_disk_bytes` | `memory.session_cache_disk_bytes` | `HIPFIRE_SESSION_CACHE_DISK_BYTES` | 0 (off) |
| `session_cache_dir` | `memory.session_cache_dir` | `HIPFIRE_SESSION_CACHE_DIR` | `$HIPFIRE_HOME/session-cache` |
| `state_quant` | `memory.state_quant` | `HIPFIRE_STATE_QUANT` | `"auto"` (Flash-Next: F32 on exact gfx1151, Q8 elsewhere; Qwen3.5: Q8; `q8_ef` only when explicit) |
| `devices` | `hardware.devices` | `HIPFIRE_DEVICES` (alias `HIPFIRE_DEVICE`) | unset |
| `allow_mixed_arch` | `hardware.allow_mixed_arch` | `HIPFIRE_ALLOW_MIXED_ARCH` | false unless `1` |
| `uniform_vram_tolerance_gb` | `hardware.uniform_vram_tolerance_gb` | `HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB` | unset |
| `mtp_mode` | `speculation.mtp` | `HIPFIRE_MTP_MODE` | `"auto"` (arms only a producer that declares a measured AR floor: Qwen4 today; Qwen35 / DeepSeek V4 / Gemma 4 stay AR; `on` / `--spec mtp` always arms; `off` never) |
| `mtp_k` | `speculation.mtp_k` | `HIPFIRE_MTP_K` | `"auto"` (native Qwen4 MTP: `min(GPU row-exact verify rows, CPU-owned-expert window) − 1`, bounded by executor 63 = 7 with no CPU-owned experts, 3 on a CPU-placed load today, resolved at load after placement; `hipfire config` shows 7; other models 3; explicit 1–63 accepted, producer/qualified-row/CPU limits remain; `4` restores the previous gfx1151 cap) |
| `mtp_sampled` | `speculation.mtp_sampled` | `HIPFIRE_MTP_SAMPLED` | true |
| `mtp_ngram` | `speculation.mtp_ngram` | `HIPFIRE_MTP_NGRAM` | `"auto"` (on for Qwen4 native MTP on gfx1151 / gfx1201; `off`/`0` opts out) |
| `mtp_ngram_copy` | `speculation.mtp_ngram_copy` | `HIPFIRE_MTP_NGRAM_COPY` | `"auto"` (on for native Qwen4 MTP on gfx1151 / gfx1201, off elsewhere; `off`/`0` is the kill switch) |
| `dflash_lookup` | `speculation.dflash_lookup` | `HIPFIRE_DFLASH_LOOKUP` | `"auto"` (on; `off`/`0` disables the DFlash2 verbatim-copy lookup) |

`kernel.lm_head_f16` maps the live Qwen3.5/3.6 loader compatibility control
`HIPFIRE_LM_HEAD_F16`; the duplicate unused `RuntimeConfig` member was removed.
The similarly unused `RuntimeConfig::dflash_draft` member was removed rather
than promoted. Draft selection remains part of the typed speculation/load
policy and filename/registry discovery.

Note: `ddtree_budget` and `ddtree_topk` are `FeatureFlags` overrides, not `RuntimeConfig` fields. The schema defaults (`0` and `4`) are what the CLI sends for `hipfire run`/`serve`. A zero budget reads as unset, so each call site falls back to its own default (the dense path uses 8 nodes when `speculation.dflash_tree` is on; the Qwen3.5 path stays chain-only); the non-zero `ddtree_topk` default is always sent and overrides a call site's own fanout.

Redline eligibility helpers `mq4r_redline_default` and `retained_redline_default` (same file) are **narrow runtime-default predicates** for replay backend selection: exact GPU arch `gfx1100`/`gfx1151`/`gfx1201` + case-insensitive `.mq4r` + pp=tp=1 (model-family agnostic, no `arch_id` gate; `gfx1200` and all other arches remain opt-in), plus the A3B symmetric MQ4R file (`qwen3_5_moe`, `.mq4rs`) plain-AR on exact `gfx1201` with pp=tp=1 and no drafter, Qwen3.5 dense (`qwen3_5`) and Qwen4 (Flash-Next, `qwen4`; MTP-off loads) plain-AR decode on exact `gfx1201` with pp=tp=1 and no drafter. A Qwen3.5-family process with a CPU-executed partial-offload spill (`memory.offload_exec=cpu` plus a manual layer/routed-expert budget or explicit `auto`/`-1` that resolves to a spill) gets no retained default, and an explicit non-`hip` backend there is refused at load. Unset/null budgets never request a spill. They are not general config keys and not Redline certification/registry admission. `replay.backend = "hip"`, the built-in `hip` config profile or another explicit backend selection disables the automatic default. Policy: [`REDLINE.md`](REDLINE.md).

---

## Calibration (machine-written)

`config.toml` can carry one more top-level table, `[calibration]`, that `hipfire calibrate` writes and nothing else does. It records what this host measured (PCIe link, CPU memory read rate, per-GPU offload engine rates) together with the hardware it was measured on, so the engine can skip the equivalent load-time probe. It is **not** a settable key namespace: it has no schema keys, `hipfire config set calibration.*` is rejected as an unknown key, and `hipfire config list|explain` do not show it. The loader skips it like `schema_version`, and every writer of `config.toml` (`hipfire config set` / `reset`, the Settings TUI, profile selection, tuning presets) preserves it untouched. Per-model overlays and profiles never carry it. See [`hipfire calibrate`](CLI.md#hipfire-calibrate) for the command.

This is the record `hipfire calibrate` wrote on hipx (7900 XTX on a PCIe 4.0 x4 link, Strix Halo host; hold j1760). Seven of the nine engine entries are left out here:

```toml
[calibration]
duration_s = 6.4798244480000005
hipfire_version = "0.4.3 (8cef2bdd47d5; feat/calibrate-043)"
measured_at = "2026-10-18T00:37:16Z"
probe_version = 1

[calibration.cpu_read]
bytes = 4294967296
gbps = 120.02258219024576
reps = 3
threads = 16

[[calibration.devices]]
arch = "gfx1100"
pci_bdf = "0000:67:00.0"
pinned_h2d_gbps = 7.115572651289664
unified = false
vram_bytes = 25753026560
zero_copy_gbps = 7.167447390604741

[[calibration.devices.engine]]
cpu_gbps = 102.85508765691702
format = "mq4g256v2"
gpu_gbps = 4.248585554299781
k = 5120
m = 74018
pool = "global"
reps = 5

[[calibration.devices.engine]]
cpu_gbps = 121.58500365366845
format = "q8_0"
gpu_gbps = 6.571798732768647
k = 5120
m = 37009
pool = "global"
reps = 5

[calibration.devices.link]
bottleneck_bdf = "0000:65:00.0"
current_gts = 16.0
gts = 16.0
width = 4

[calibration.host]
cpu_model = "AMD RYZEN AI MAX+ 395 w/ Radeon 8060S"
isa = ["sse4.2", "avx", "avx2", "fma", "f16c", "avx512f", "avx512bw", "avx512vl", "avx512vnni", "avx512bf16", "avxvnni"]
logical_cpus = 32
physical_cores = 16
ram_total_bytes = 131217301504

[[calibration.unmeasured]]
arch = "gfx1151"
pci_bdf = "0000:c0:00.0"
reason = "not visible to HIP in this run, or it failed to open"
```

Fields marked **fingerprint** decide whether the record still describes this host. The other fields are information only.

| Field | Meaning |
|---|---|
| `probe_version` | **Fingerprint.** Version of the probe semantics (currently `1`). Any other version makes the whole record stale. |
| `measured_at` | RFC 3339 UTC time of the run. |
| `hipfire_version` | Version of the `hipfire` binary that wrote the record. |
| `duration_s` | Wall time of the whole `hipfire calibrate` run, in seconds. |
| `host.cpu_model` | **Fingerprint.** CPU model: the `model name` line of `/proc/cpuinfo`, else the CPUID brand string. |
| `host.logical_cpus` | **Fingerprint.** Logical CPU count (`processor` lines of `/proc/cpuinfo`). |
| `host.physical_cores` | Physical core count, when cpuinfo reports core ids. |
| `host.isa` | x86 ISA features detected at run time. Empty on other architectures. |
| `host.ram_total_bytes` | **Fingerprint.** `MemTotal` in bytes. A drift of up to 1 GiB counts as the same host. |
| `cpu_read.gbps` | CPU read bandwidth in GB/s (10^9 bytes/s): a parallel streaming sum over `bytes` on the splice pool, one pinned worker per physical core, median of `reps` timed passes. |
| `cpu_read.threads`, `bytes`, `reps` | Workers, buffer bytes (a quarter of `MemAvailable`, at most 4 GiB, at least 256 MiB) and timed passes. |
| `devices[].pci_bdf` | PCI address of the GPU (lowercase `0000:67:00.0`). It is the key that matches an entry to a device. |
| `devices[].arch` | **Fingerprint.** GPU architecture (`gfx…`). |
| `devices[].vram_bytes` | **Fingerprint.** Total VRAM in bytes: sysfs `mem_info_vram_total` when readable, else the HIP total. Compared exactly. |
| `devices[].unified` | `true` for a unified-memory GPU (APU). No link probe runs on one. |
| `devices[].link.gts` | **Fingerprint.** Lowest `max_link_speed` (GT/s) over every PCI device on the sysfs path from the root port to the GPU. It is the capability, so power management cannot flap it. |
| `devices[].link.width` | **Fingerprint.** Lowest `current_link_width` over that path. A card behind an internal switch reports x16 at its own function, while the slot link above it may be x4. |
| `devices[].link.current_gts` | Lowest `current_link_speed` over the path. Power management lowers it at idle (an R9700 reads 2.5 GT/s idle on a 32 GT/s link), so it is information only. |
| `devices[].link.bottleneck_bdf` | The downstream device of the narrowest link. |
| `devices[].pinned_h2d_gbps` | Pinned host-to-device copy rate in GB/s: the qwen4 host-link probe itself. It is the figure the host-link selectors read. |
| `devices[].zero_copy_gbps` | Rate at which a GPU kernel reads host-mapped memory over the link, in GB/s. Information only. |
| `devices[].engine[]` | One two-engine probe per CPU-decodable dense format (`pool = "global"`): the `offload_split` probe of the `hipfire offload-bench` matrix with the CPU arm on the rayon global pool (`k` 5120, `m` = rows of a 192 MiB weight). Each entry holds `format`, `k`, `m`, `pool`, `reps`, and the median `cpu_gbps` / `gpu_gbps` (CPU GEMV against GPU GEMV over the same host-mapped bytes). They feed the dense pass-back split only. An earlier version also wrote `pool = "splice"` entries for the D2 default; they stay readable in the record and are ignored (the D2 default no longer reads any engine rate, see below). |
| `unmeasured[]` | GPUs the kernel listed that this run could not measure: not visible to HIP (for example under `HIPFIRE_DEVICES`), or they failed to open. `pci_bdf`, `arch` and `reason` are recorded. |

**Staleness.** The engine uses a record only while it still describes the host. Otherwise it measures at load exactly as it does without a record. A record is stale when any of these differ from the live host, checked in this order:

- `probe_version`;
- the CPU model or the logical CPU count;
- total RAM, by more than 1 GiB;
- for the device being loaded: `arch`, total VRAM, `link.gts` (to 0.01 GT/s) or `link.width`.

A device with no entry is also measured at load. That covers a GPU added later and one listed under `unmeasured`. `hipfire calibrate --show` prints the reason for each stale or missing entry. `hipfire calibrate --check` exits 1 until the record is fresh for every detected GPU; a GPU listed as unmeasured with the same address and architecture counts as covered, because a re-run could not measure it either. A model load never rewrites the record; only `hipfire calibrate` does.

**Selectors that read it.** At a qwen4 (Flash-Next) model load on a discrete GPU, the host-link class takes a fresh `devices[].pinned_h2d_gbps` in place of its load-time copy probe. The load log names the source: `[calibrated <measured_at>]`, `[measured; calibration stale: <reason>]` or `[measured; <reason>]`. The link class drives the same decisions as a measured one:

- the G2 long-prefill expert-staging default on a narrow discrete link (below 16 GB/s);
- the 8192-row prefill chunk default that gfx1100 takes on a narrow discrete link, which also sizes the expert placement reserve;
- the 768-row stage minimum on a narrow link.

A calibrated load still runs the probe's copy sequence once, with a 4 KiB copy and no timing. The HIP runtime keeps about 146 MiB of VRAM for its first pinned copy on a new stream, and the expert placement reads free VRAM right after the link is resolved. So a calibrated load and a probed load see the same free VRAM and place the same expert layers.

The developer overrides `HIPFIRE_QWEN4_EXPERT_STAGE`, `HIPFIRE_PREFILL_CHUNK_ROWS` and `HIPFIRE_QWEN4_EXPERT_STAGE_MIN_ROWS` still win over a calibrated link, exactly as over a measured one. A unified-memory GPU is classified from the live device and never reads the record. `hipfire calibrate --show` lists these routes and the reason for each.

**D2 default (Flash-Next, unset `memory.offload_exec`).** The record supplies nothing to the D2 measured default ([Partial GPU offload](#partial-gpu-offload-memorygpu_layer_budget--memorymoe_expert_budget--memoryoffload_exec), the "D2 measured default" and "Calibration record" bullets): the decision needs this model's real CPU expert arm and live link samples, which a stored dense, best-of rate cannot provide. The first eligible, not-yet-measured shape (device, resident codecs, expert shape) in a process therefore gets live evidence (a few milliseconds of CPU and 5 short link reads; the cost is logged) with no calibration record involved; later loads of the same shape in that process reuse the successful in-process measurement, and a failed or invalid one is not cached. No stored CPU or link figure can enable D2. `hipfire calibrate` therefore does not write `splice`-pool entries any more, and `hipfire calibrate --show` prints, per fresh device, `D2 default    decided at model load …` (JSON `d2_default.status` = `load`), or `off (unified memory: no link to beat)` on a unified-memory GPU. An explicit `memory.offload_exec`, `developer.qwen4_d2_default`, `HIPFIRE_MOE_CPU_EXPERTS=0`, a load with no host split layers and a unified-memory GPU decide before any probe.

**Removing it.** `hipfire calibrate --clear` deletes the `[calibration]` table and leaves every other key alone. The engine then measures at load again.

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

Removed keys. A key that shipped in a release and was later removed from the
schema stays in `RETIRED_CONFIG_KEYS` (`crates/hipfire-config/src/lib.rs`)
through its deprecation window: a `config.toml` that still sets it loads, the
value is ignored, and run/serve/chat and `hipfire config show` print
`warning: ignored <key> in <file>: removed in <version> (…)`. Custom profiles
drop it the same way. `hipfire config set` refuses it. A key that never shipped
in a release (for example one that only existed on an experiment branch) or a
misspelled key is unknown and fails the load with
`unknown configuration key '<key>'`.

| Removed key | Legacy key | Shipped | Removed in | Ignored with a warning until |
| --- | --- | --- | --- | --- |
| `kernel.mw16` | `mw16` | 0.3.1 | 0.4.1 | 0.5.0 |

<!-- lifecycle-table:begin (generated by scripts/check-lifecycle.py --write) -->

**Count:** 310 schema keys, plus the `developer.<name>` namespace.

| Key | Legacy key | Env | Lifecycle |
|---|---|---|---|
| `attention.ck_runtime_lib` | `ck_runtime_lib` | `HIPFIRE_FLASH_ATTN_CK_LIB` | experimental |
| `attention.ck_workspace_bytes` | `ck_workspace_bytes` | `HIPFIRE_FLASH_ATTN_CK_WORKSPACE_BYTES` | experimental |
| `attention.flash` | `flash_mode` | `HIPFIRE_ATTN_FLASH` | stable |
| `attention.yarn_factor` | `yarn_factor` | `HIPFIRE_YARN_FACTOR` | stable |
| `decide.calibration.choice` | `decide_calibration_choice` |  | stable |
| `decide.calibration.noul` | `decide_calibration_noul` |  | stable |
| `decide.calibration.score` | `decide_calibration_score` |  | stable |
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
| `diagnostic.replay.gfx1151_cu_count` | `gfx1151_redline_cu_count` | `HIPFIRE_GFX1151_REDLINE_CU_COUNT` | deprecated |
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
| `fusions.npu_trunk` | `npu_trunk` | `HIPFIRE_QWEN4_NPU_TRUNK` | experimental |
| `fusions.npu_trunk_calibration` | `npu_trunk_calibration` | `HIPFIRE_QWEN4_NPU_TRUNK_CAL` | experimental |
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
| `kernel.draft_shared_ctx_kv_rotation` | `draft_shared_ctx_kv_rotation` | `HIPFIRE_DRAFT_SHARED_CTX_KV_ROTATION` | stable |
| `kernel.draft_shared_ffn_rotation` | `draft_shared_ffn_rotation` | `HIPFIRE_DRAFT_SHARED_FFN_ROTATION` | stable |
| `kernel.draft_shared_qkv_rotation` | `draft_shared_qkv_rotation` | `HIPFIRE_DRAFT_SHARED_QKV_ROTATION` | stable |
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
| `kernel.gfx1100_fa2_split_verify` | `gfx1100_fa2_split_verify` | `HIPFIRE_GFX1100_FA2_SPLIT_VERIFY` | stable |
| `kernel.gfx11_a4_candidates` | `gfx11_a4_candidates` | `HIPFIRE_GFX11_A4_CANDIDATES` | experimental |
| `kernel.gfx11_a8_prefill` | `gfx11_a8_prefill` | `HIPFIRE_GFX11_A8_PREFILL` | experimental |
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
| `kernel.gfx12_mq4v2_gateup_k5120` | `gfx12_mq4v2_gateup_k5120` | `HIPFIRE_GFX12_MQ4V2_GATEUP_K5120` | experimental |
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
| `kernel.mq4v2_gateup_k5120` | `mq4v2_gateup_k5120` | `HIPFIRE_MQ4V2_GATEUP_K5120` | stable |
| `kernel.npu_spillover` | `npu_spillover` | `HIPFIRE_NPU_SPILLOVER` | experimental |
| `kernel.pm_decode` | `pm_decode` | `HIPFIRE_PM_DECODE` | experimental |
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
| `memory.moe_expert_budget` | `moe_expert_budget` | `HIPFIRE_MOE_EXPERT_BUDGET` | stable |
| `memory.offload_exec` | `offload_exec` | `HIPFIRE_OFFLOAD_EXEC` | stable |
| `memory.offload_passback_share` | `offload_passback_share` | `HIPFIRE_OFFLOAD_PASSBACK_SHARE` | stable |
| `memory.oom_guard` | `oom_guard` | `HIPFIRE_OOM_GUARD` | stable |
| `memory.ple_format` | `ple_format` | `HIPFIRE_PLE_FORMAT` | stable |
| `memory.ple_hot_store_bytes` | `ple_hot_store_bytes` | `HIPFIRE_PLE_HOT_STORE_BYTES` | stable |
| `memory.ple_load_fill_bytes` | `ple_load_fill_bytes` | `HIPFIRE_PLE_LOAD_FILL_BYTES` | stable |
| `memory.ple_prewarm_bytes` | `ple_prewarm_bytes` | `HIPFIRE_PLE_PREWARM_BYTES` | stable |
| `memory.prompt_cache_capacity` | `prompt_cache_capacity` | `HIPFIRE_PROMPT_CACHE_CAP` | stable |
| `memory.prompt_cache_unbounded` | `prompt_cache_unbounded` | `HIPFIRE_PROMPT_CACHE_UNBOUNDED` | experimental |
| `memory.session_cache_bytes` | `session_cache_bytes` | `HIPFIRE_SESSION_CACHE_BYTES` | stable |
| `memory.session_cache_dir` | `session_cache_dir` | `HIPFIRE_SESSION_CACHE_DIR` | stable |
| `memory.session_cache_disk_bytes` | `session_cache_disk_bytes` | `HIPFIRE_SESSION_CACHE_DISK_BYTES` | stable |
| `memory.state_quant` | `state_quant` | `HIPFIRE_STATE_QUANT` | stable |
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
| `serve.api_key_file` | `api_key_file` | `HIPFIRE_SERVE_API_KEY_FILE` | stable |
| `serve.batch_nonexact` | `serve_batch_nonexact` | `HIPFIRE_SERVE_BATCH_NONEXACT` | experimental |
| `serve.batch_sampled` | `serve_batch_sampled` | `HIPFIRE_SERVE_BATCH_SAMPLED` | experimental |
| `serve.batch_spec` | `serve_batch_spec` | `HIPFIRE_SERVE_BATCH_SPEC` | experimental |
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
| `serve.multi_slot_max_sessions` | `multi_slot_max_sessions` | `HIPFIRE_SERVE_MULTI_SLOT_MAX_SESSIONS` | experimental |
| `serve.multi_slot_prefill_chunk` | `multi_slot_prefill_chunk` | `HIPFIRE_SERVE_MULTI_SLOT_PREFILL_CHUNK` | stable |
| `serve.multi_slot_session_ttl_secs` | `multi_slot_session_ttl_secs` | `HIPFIRE_SERVE_MULTI_SLOT_SESSION_TTL_SECS` | experimental |
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
| `serve.swap_dir` | `swap_dir` | `HIPFIRE_SWAP_DIR` | experimental |
| `serve.swap_disk_bytes` | `swap_disk_bytes` | `HIPFIRE_SWAP_DISK_BYTES` | experimental |
| `serve.swap_host_bytes` | `swap_host_bytes` | `HIPFIRE_SWAP_HOST_BYTES` | experimental |
| `serve.ui` | `serve_ui` | `HIPFIRE_SERVE_UI` | stable |
| `serve.vmm_batch` | `serve_vmm_batch` | `HIPFIRE_SERVE_VMM_BATCH` | experimental |
| `speculation.ddtree_budget` | `ddtree_budget` | `HIPFIRE_DDTREE_BUDGET` | experimental |
| `speculation.ddtree_topk` | `ddtree_topk` | `HIPFIRE_DDTREE_TOPK` | experimental |
| `speculation.ddtree_tree_la` | `ddtree_tree_la` | `HIPFIRE_DDTREE_TREE_LA` | experimental |
| `speculation.dflash` | `dflash_mode` | `HIPFIRE_DFLASH_MODE` | stable |
| `speculation.dflash_adaptive_b` | `dflash_adaptive_b` |  | stable |
| `speculation.dflash_block` | `dflash_block` | `HIPFIRE_DFLASH_BLOCK` | stable |
| `speculation.dflash_fast_sample` | `dflash_fast_sample` | `HIPFIRE_DFLASH_FAST_SAMPLE` | experimental |
| `speculation.dflash_lookup` | `dflash_lookup` | `HIPFIRE_DFLASH_LOOKUP` | stable |
| `speculation.dflash_ngram_block` | `dflash_ngram_block` | `HIPFIRE_DFLASH_NGRAM_BLOCK` | stable |
| `speculation.dflash_q8_lmhead_wmma` | `dflash_q8_lmhead_wmma` | `HIPFIRE_DFLASH_Q8_LMHEAD_WMMA` | experimental |
| `speculation.dflash_tree` | `dflash_tree` | `HIPFIRE_DFLASH_TREE` | experimental |
| `speculation.draft_f16` | `draft_f16` | `HIPFIRE_DRAFT_F16` | stable |
| `speculation.dspark_confidence` | `dspark_conf_threshold` |  | experimental |
| `speculation.mode` | `speculation` | `HIPFIRE_SPECULATION` | stable |
| `speculation.mtp` | `mtp_mode` | `HIPFIRE_MTP_MODE` | stable |
| `speculation.mtp_k` | `mtp_k` | `HIPFIRE_MTP_K` | stable |
| `speculation.mtp_ngram` | `mtp_ngram` | `HIPFIRE_MTP_NGRAM` | stable |
| `speculation.mtp_ngram_copy` | `mtp_ngram_copy` | `HIPFIRE_MTP_NGRAM_COPY` | stable |
| `speculation.mtp_sampled` | `mtp_sampled` | `HIPFIRE_MTP_SAMPLED` | stable |
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
| `topology.ep` | `ep` | `HIPFIRE_TOPOLOGY_EP` | experimental |
| `topology.pp` | `pp` | `HIPFIRE_TOPOLOGY_PP` | experimental |
| `vision.mode` | `vision_mode` | `HIPFIRE_VISION_MODE` | stable |
| `developer.<name>` | | `HIPFIRE_<NAME>` | developer |

<!-- lifecycle-table:end -->

## Serve bearer credential file

`serve.api_key_file` is an optional process-scoped string, empty/unset by default
(no authentication). Set an absolute path, never the secret itself.
`hipfire serve --api-key-file` overrides it for that process;
`HIPFIRE_SERVE_API_KEY_FILE` is the environment override. CLI/TUI local HTTP
calls load the same configured credential.
The file must be UTF-8 with at least 32 bytes after trimming trailing whitespace;
Unix group/other permissions are refused (`chmod 600`). Windows ACLs are not
checked yet, and Windows never auto-generates a key. A key is recommended, not
required, on a non-loopback bind, and remote bearer traffic requires an HTTPS
proxy for confidentiality. See [SERVE.md](SERVE.md#security-bearer-authentication-and-https-proxies).

## Related

| Topic | Owner |
|---|---|
| Copyable TOML profiles | [`configs/`](configs/README.md) |
| Env inventory | [`env-vars.md`](env-vars.md) |
| Models / registry sampling | [`MODELS.md`](MODELS.md) |
| Serve API | [`SERVE.md`](SERVE.md) |
| CLI | [`CLI.md`](CLI.md) |
| Multi-GPU | [`multi-gpu.md`](multi-gpu.md) |
