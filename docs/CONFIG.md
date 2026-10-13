# Configuration

**Owner:** daemon / user config keys (`docs/INDEX.md`).
**Machine sources:**

- Native schema + resolution: `crates/hipfire-config/src/lib.rs`
- Versioned CLI/daemon handoff: `hipfire_config::ProcessConfig`
- Compact runtime snapshots: `hipfire_runtime::config::RuntimeConfig` and
  `rdna_compute::feature_flags::FeatureFlags`

**Last checked:** 2026-10-13.

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
| RDNA4 MQ4V2 gate/up decode | `kernel.gfx12_mq4v2_gateup_k5120` (exact gfx1201, K=5120, 17408 rows per output; default off) |
| GEMV / quant | `kernel.gemv_dp4a`, `kernel.gemv_prefetch`, `kernel.gfx942_lds_gemv`, `kernel.hfq3_dp4a`, `kernel.hfq3_mmq`, `kernel.hfq4_mmq_rdna2`, `kernel.gcn5_wave64_hybrid`, `kernel.mmq`, `kernel.gfx942_gemv_v2` |
| MoE | `kernel.moe_grouped_i8`, `kernel.moe_paro_i8`, `kernel.moe_paro_i8_k8` |
| Certified arch routes | `kernel.rdna3_hfq4_qkvza_k2048`, `kernel.rdna3_hfq4_residual_stage_x32`, `kernel.rdna3_hfq4_sigmoid_buffer`, `kernel.rdna3_rmsnorm_vecsum`, `kernel.gfx942_rmsnorm_split` |

Stable/default-on and safety controls:

| Key | Default | Purpose |
|---|---:|---|
| `hardware.allow_mixed_arch` | `false` | Permit heterogeneous multi-GPU topology. |
| `topology.ep` | `1` | Qwen4 (Flash-Next) expert-parallel degree on gfx1201; opt-in. Env `HIPFIRE_TOPOLOGY_EP`. The schema validates an integer 1–4; a load accepts `1`, `2` or `4` and refuses `3`. Routed experts shard `e % N` across N visible gfx1201 cards with peer access. `HIPFIRE_QWEN4_EP=0` forces `1`. Admission rules and the support matrix live with the owner, [multi-gpu.md](multi-gpu.md); the config-exposed EP and NPU controls are listed in [Qwen4 expert-parallel and NPU controls](#qwen4-expert-parallel-and-npu-controls). |
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
| `fusions.npu_trunk` | `"auto"` | Flash-Next (Qwen4) trunk projection engine: `off\|auto\|on\|emulate` (case-insensitive). `auto` (default) resolves silently to `off` on every arch except gfx1151, with no NPU I/O or allocation. On gfx1151 it arms only when the load is eligible (Flash-Next, K=2560, stock MQ6G256V2 trunk projections, a single device without EP, the full 8192-row prefill scratch rung and enough free memory for it), the NPU firmware is readable, and the calibration matches the device/build/model identity and selects a nonempty family set; any refusal logs one `npu-trunk: off (reason)` line and the load continues GPU-only. `off` is the kill switch. `on` requires the NPU and `emulate` runs the bit-identical GPU emulation of the NPU numerics: both refuse an ineligible load or arch instead of falling back, and both tolerate a missing or mismatched calibration (logged as `calibration=none (forced)`; every family is used). |
| `fusions.npu_trunk_calibration` | `""` | Empty resolves to `<hipfire home>/npu-trunk/calibration.json`; alias `npu_trunk_calibration`, env `HIPFIRE_QWEN4_NPU_TRUNK_CAL` (a non-empty path replaces the default). A calibration is `schema_version` 4 JSON that rejects unknown fields. A file at the path wins; only a **missing** file uses the compiled-in measured Halo calibration (stock route, or the overlay route of the load's trunk overlay), so an unreadable or invalid file is an error and never falls back to the built-in. File and built-in calibrations pass the same strict key check (arch, PCI BDF, NPU firmware, build, kernel release, model-index hash, epilogue, trunk route); `auto` treats a mismatch as GPU-only and never recalibrates. Armed logs include `cal_source=file\|builtin`. No calibration/index I/O when effectively off. See [NPU trunk](npu/npu-trunk.md). |
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

`kernel.mq4v2_gateup_k5120` (exact gfx1100, K=5120, 17408 rows per output) is default on since 0.4.1: that gate/up decode runs on a fixed-group-count kernel whose output is byte-identical to the generic one. Set `hipfire config set kernel.mq4v2_gateup_k5120 false` to restore the generic route; the one-shot compatibility override is `HIPFIRE_MQ4V2_GATEUP_K5120=0` / `1`. This process-wide setting is snapshotted at GPU initialization; restart an existing daemon after changing it. Other architectures, shapes, and quantization formats keep their existing routes. Compiler-less installations need a kernel pack containing `fused_gate_up_mq4g256v2_k5120_gfx1100` (the 0.4.1 gfx1100 pack carries it).

Architecture-gated default-on kernel routes (exact arch only; other arches keep their prior routes; all are process-wide and snapshotted at GPU initialization, so restart an existing daemon after changing them):

| Key | Default on | Opt out | Notes |
|---|---|---|---|
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
| `topology.ep` | `1` | integer 1–4 in the schema; a load accepts `1`, `2`, `4` | Env `HIPFIRE_TOPOLOGY_EP`. `HIPFIRE_QWEN4_EP=0` forces `1` and wins over any configured value, even an invalid one. A load needs pp = tp = 1, gfx1201 on every rank, N visible devices and peer access between every pair of ranks, or it is refused naming the missing requirement. |
| `fusions.npu_trunk` | `"auto"` | `off` \| `auto` \| `on` \| `emulate` | gfx1151, single device only; see the stable-controls table above. |
| `fusions.npu_trunk_calibration` | `""` | path string | See the stable-controls table above. |
| `kernel.npu_spillover` | `false` | bool; env `HIPFIRE_NPU_SPILLOVER` | Opt-in, gfx1151-only XDNA NPU spillover sidecar. Needs a pulled, verified `.xdna.zip` sidecar declared by the registry entry: with the flag on and no declared sidecar the load stays GPU-only (one log line); a declared sidecar that is missing or fails verification fails the load with a `hipfire pull` hint. Never active on other architectures. |

With `topology.ep` > 1 the loader refuses DFlash (including a draft path), DSpark, n-gram drafting, DDTree, adaptive KV, an EAGLE drafter, CASK, PFlash, an experts-per-token override and a non-`single` compute placement. Native MTP attaches only on an explicit request (`--spec mtp` / `speculation.mtp = "on"`; `auto` stays AR). Lane entry points, state snapshots and `HIPFIRE_MTP_SAMPLED_MODE=naive` are unsupported under Qwen4 EP.

The EP2 owner-routed prefill and the expert-staging levers have no typed schema key. They are `[developer]` keys (a legacy `HIPFIRE_FOO=value` maps to `developer.foo`), read from the process snapshot at load; restart the daemon after changing them. For the three `ep_*` levers only the exact value `0` turns the lever off.

| Developer key | Env | Default | Effect |
|---|---|---|---|
| `developer.ep_owner_routed_prefill` | `HIPFIRE_EP_OWNER_ROUTED_PREFILL` | on under EP2 | Owner-routed chunk MoE: a prefill chunk of at least 512 rows whose layers verified symmetric on every rank computes only its own experts' tiles and pulls the other ranks' tiles over P2P; every other chunk keeps the full-table route. Admitted only on gfx1201 with the Flash-Next profile, `kernel.moe_grouped_gemm` on, the symmetric IU4 MoE route requested (below) and the full (non-compact) scratch plan; a refusal logs `qwen4 EP rank r/n: owner-routed prefill not admitted (reason)`. **EP4 is refused at admission** (`EP4 owner-routed prefill not yet gated`; the code is rank-generic but the EP4 exactness cells have not run), so EP4 chunks keep the full-table route. |
| `developer.qwen4_moe_sym_iu4` | `HIPFIRE_QWEN4_MOE_SYM_IU4` | unset = requested on gfx1151 / gfx1201 | Grouped symmetric IU4 MoE prefill route for Qwen4. `1` requests it, `0` keeps the whole incumbent route. The request also needs the IU4 producers built with exactly the two-candidate activation search: on gfx1201 that is `kernel.g12_a4c2` (default `true`), on gfx1151 `kernel.gfx11_a4_candidates = "2"` (default), and any other `-DIU4_A4_CANDIDATES` value in `diagnostic.compiler.hipcc_extra_flags` withdraws it. Layers whose routed-expert headers fail the symmetry check (asymmetric artifacts) stay on the incumbent route. Owner-routed prefill therefore has no single request key: it follows this developer lever plus those typed keys. |
| `developer.ep_balanced_owners` | `HIPFIRE_EP_BALANCED_OWNERS` | on under EP | Per-call balanced expert ownership on the host-mapped trunk MoE layers of an EP rank. Off builds nothing. |
| `developer.ep_expert_stage` | `HIPFIRE_EP_EXPERT_STAGE` | on | Admits long-prefill expert staging under EP. Applies only when the owner-routed route is installed (so never on EP4) and every trunk MoE layer in the forward binds the compact table. |
| `developer.qwen4_expert_stage` | `HIPFIRE_QWEN4_EXPERT_STAGE` | on for gfx1201 only; `1` / `0` force | Long-prefill host-expert staging: a long forward borrows the VRAM of the last two resident expert layers as DMA stages and restores them before it returns, so decode residency is unchanged. |
| `developer.qwen4_expert_stage_min_rows` | `HIPFIRE_QWEN4_EXPERT_STAGE_MIN_ROWS` | `2048` | Forwards with at least this many rows stage (integer ≥ 2); smaller ones, decode and verify keep the mapped reads. |
| `developer.qwen4_expert_stage_capture` | `HIPFIRE_QWEN4_EXPERT_STAGE_CAPTURE` | on | `0` keeps the native-MTP prompt-fill tiles on the mapped reads. |
| `developer.qwen4_expert_vram_layers` | `HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS` | unset (automatic placement) | `N` keeps the routed experts of trunk layers before `N` in VRAM and places layers at or past `N` in pinned host RAM. |
| `developer.ep_peer_allreduce_decode` | `HIPFIRE_EP_PEER_ALLREDUCE_DECODE` | unset (canonical) | `1` selects the legacy unrooted peer all-reduce diagnostic, `0` selects RCCL. |

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
- Template probing only checks whether semantic effort is native; it never
  changes field meaning.

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

| `kv_cache` | gfx1201 | gfx1100, gfx1151 (Halo), other gfx11/gfx12 | other arches |
|---|---|---|---|
| `auto` / unset | `fp8` | `bf16` | `bf16` |
| `bf16` | exact reference state (F32 K/V arenas, BF16-valued index keys) | same | same |
| `fp8` | E4M3 K/V with one f16 scale per head and token; indexer raw/pooled keys stored as BF16 (the same values) | refused | refused |
| `q8` (opt-in) | Q8_0 K/V: the dense q8 KV row (32-element blocks of one f16 scale and 32 int8 codes, encoded as `kv_cache_write_q8_0`); indexer keys stored as BF16 (the same values) | same | refused |
| anything else (`fwhtN`, `asymN`, `f16`, …) | refused | refused | refused |

`auto` picks fp8 only on exact gfx1201 and only for a head geometry the
kernels implement (head_dim 256, even KV-head count; Flash-Next qualifies);
otherwise it stays `bf16`. `auto` never picks `q8`; it needs head_dim 256.
The native MTP layer keeps the exact state. On Flash-Next, each trunk QSA
layer's context state is 4,736 bytes per token at `bf16`, 1,352 at fp8 and
1,408 at q8 (12 layers at 262,144 tokens: 14,208, 4,056 and 4,224 MiB). The
Q8 GDN state is 3.88× smaller than F32. `memory.max_seq` goes up to the
native 262,144. Index keys hold the same values in every format, so
selection is unchanged; only the attention over the selected K/V rows sees
the quantization.

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
gfx1100/gfx1151 the certified Qwen VMM route is q8 only; unvalidated modes
retain the existing legacy fallback.

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

**Free device memory on unified memory.** Every free-memory sizing site
(this admission, expert placement, the dense Qwen card bound below, VMM KV
provisioning, prefill rungs, the session cache) reads `Gpu::device_mem_info`.
A discrete GPU, or an APU whose device pool is its BIOS carve-out, gets
`hipMemGetInfo` unchanged. An APU that allocates from GTT (device pool larger
than `mem_info_vram_total`, e.g. Strix Halo with a 1 GiB carve-out and a
120 GiB `ttm.pages_limit`) gets `min(hipMemGetInfo free, MemAvailable + TTM
page-pool estimate)`: `hipMemGetInfo` there is TTM's cap minus live GTT and
ignores host RAM pressure, while freed GTT pages parked in TTM's pool are
outside `MemAvailable` but are what the next GTT allocation takes first. An
unreadable `MemAvailable` fails the query (callers refuse rather than size
blind).

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
- **`auto`** — cascade by availability / eligibility; legacy mode knobs filter each mechanism. Under auto, DSpark can win over in-trunk MTP when a DSpark sidecar is present (CLI comments).
- **Forced mechanisms require their own prerequisites.** `--spec dflash` / `speculation.mode=dflash` forces `speculation.dflash=on` and refuses the load without a DFlash draft; it never aliases to native MTP. Use `--spec mtp` / `speculation.mode=mtp` or `speculation.mtp=on` to request native MTP explicitly, subject to the safety refusal in the `mtp_mode` row.

Legacy one-shot alias: `HIPFIRE_SPECULATION`. CLI: `--spec`.

### Mechanism knobs

| Key | Default | Values / range | Notes |
|---|---|---|---|
| `dflash_mode` | `"off"` | `on` \| `off` \| `auto` | **Schema default off; one registry-scoped exception.** `auto` enables on dense Qwen3.5-class targets and skips known-loss A3B cases. The registry layer (below the global / per-model / one-shot config, above the schema default) lowers `speculation.dflash = "auto"` for the Qwen3.8-27B trunk tiers that declare a paired `dflash` draft (`qwen3.8:27b`, `qwen3.8:27b-mq3*`, `qwen3.8:27b-mq4*`; not the `-draft-*` sidecar tags, not tiers without a `dflash` declaration). See [Qwen3.8-27B paired DFlash default](#qwen38-27b-paired-dflash-default). `on` requires a DFlash draft and fails the load without one, even when an MTP head is present. It never enables MTP; use `--spec mtp` / `speculation.mtp=on` instead. |
| `vision_mode` | `"off"` | `on` \| `off` \| `auto` | **Default off.** Tower sidecar gate — see [Vision tower](#vision-tower). |
| `dflash_adaptive_b` | `false` | bool | **Opt-in.** Adaptive verify-block width: follows the trailing 8-cycle acceptance depth (τ̂+2), full below 2k context. Not output-identical (window boundaries move; per-position sampling stays target-lossless). Auto-suppressed on retained-PM4 verify loads. `HIPFIRE_DFLASH_ADAPTIVE_B=0` is a kill switch read through the `[developer]` namespace (not a schema env alias): it forces the fixed block even when this is `true`, and cannot enable adaptive. |
| `dflash_block` | `"auto"` | `"auto"` \| int 4–16 | DFlash2 proposal width L (`speculation.dflash_block`, env `HIPFIRE_DFLASH_BLOCK`). Chosen **per serving configuration, never per live batch**: every request of a load (singleton or batched VMM lane) runs the same L, so batched output stays byte-identical to the singleton and text never depends on server load. **Default `auto`** = 8 when `serve.vmm_batch` is on with `continuous_batch_size` ≥ 4 (a full C8 verify step stays at 64 rows instead of 128), else 16. A lone request on a batching serve runs slower at 8 than at 16. `16` (or `HIPFIRE_DFLASH_BLOCK=16`) restores the previous fixed width. Greedy text depends on L (different window boundaries). Legacy DFlash drafts always run their declared width. Retained-PM4 verify and dense TP2 need 16. The experimental multi-slot route admits only 16 (`auto` resolves to 16 there). |
| `dflash_ngram_block` | `"auto"` | `true` \| `false` \| `"auto"` | Verify-path n-gram defense; auto size-gates. |
| `mtp_mode` | `"auto"` | `off` \| `on` \| `auto` | Built-in MTP when a head is present: the DeepSeek V4 trunk's MTP layer, or for Qwen a bundled `.mq4-mtp` trailer or a `.mtp` sidecar (the registry `mtp` slot; Qwen3.8-27B ships one — [MODELS.md](MODELS.md#dflash-draft-artifacts-registry)). `auto` uses an eligible present head; `on` fails the load without one. On Qwen3.5-class MoE, `auto` declines to AR after placement resolves a host-mapped routed-expert spill, on either `memory.offload_exec=pcie` or `cpu`, logging the count, seam and `opt in with --spec mtp`. `on` / `--spec mtp` opts in, except that uniform MQ6G256 V1 (qt15) Qwen35 MoE with FP16 WMMA enabled on gfx1200/gfx1201 is refused before serving/generation whenever a native MTP head would attach, including resident `auto` attachment. The guard uses tensor metadata, not filenames. `--spec off` / `speculation.mtp=off` or an auto-declined spill stays AR; `HIPFIRE_FP16=0` permits explicit MTP. Mixed/graded MQ4+MQ6, MQ6V2, gfx1151 and dense models are outside this refusal. Other resident attachment behavior is unchanged; this is admission protection, not a fix for the underlying WMMA defect. |
| `mtp_k` | `"auto"` | `"auto"` \| int 1–10 | MTP draft cap (`speculation.mtp_k`, env `HIPFIRE_MTP_K`). **Default `auto`** = 4 for native Qwen4 MTP (Flash-Next, arch 16) on gfx1151, else 3. An explicit 1–10 overrides on every model; `3` restores the previous gfx1151 cap. |
| `mtp_sampled` | `true` | bool | Default true. Native Qwen4 MTP uses distribution-preserving speculative rejection sampling for temperature >0, including supported repeat/presence/frequency penalties: target verification and draft probabilities use the same AR history, penalty window and sampling policy. False/`HIPFIRE_MTP_SAMPLED=0` leaves sampled requests on AR; greedy MTP is unchanged. This is distribution preservation, not identical seeded sampled token streams. DFlash/other drafters without this policy still fall back to AR for unsupported penalty requests. |
| `mtp_ngram` | `"auto"` | `on` \| `off` \| `auto` (also `1` / `0`) | **Default `auto` (0.4.1).** N-gram takeover windows inside native MTP (ngram-mod). `auto` arms native Qwen4 MTP (arch 16, Flash-Next) on gfx1151 and gfx1201 and leaves every other producer off; `on`/`1` arms every MTP producer that supports it; `off`/`0` is the opt-out (kill switch). Env `HIPFIRE_MTP_NGRAM` beats TOML beats default. Eligible requests are greedy and sampled, thinking on or off (Qwen4 and Qwen3.5/3.x). A sampled takeover verifies each candidate with the point-mass draft distribution δ(candidate) through the shared rejection verify: lossless in distribution, not bit-identical to MTP-only output. The pool is request-local (cleared and reseeded from the full prompt every request), so candidates never depend on earlier requests. Triple (`HIPFIRE_NGRAM_MOD_N_MATCH` / `_N_MIN` / `_N_MAX`, overridable per component): 5/3/3 on admitted Qwen4 targets, 24/48/64 elsewhere; on Qwen4 `n_max` is capped at 63 (the 64-row verify). On Qwen4 a pool hit takes the window only when the MTP head's own first draft equals the first candidate and the takeover's measured yield (decayed emitted tokens per takeover window over its verify cost) is at least the native window's best expected rate; otherwise the window is native. The Qwen4 takeover uses the same batched verify executor as native MTP windows (interleaved one-row route under `HIPFIRE_MTP_INCREMENTAL=1`) and appends the consumed rows to the MTP head teacher-forced (token p, hidden p). Load marker: `qwen4 native MTP n-gram: on\|off (n_match=.. n_min=.. n_max=..)`; `HIPFIRE_MTP_TRACE=1` window events carry `source` / `verify_route`. |
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

The built-in `speculation.dflash` default stays `off` and `speculation.mtp` stays `auto`. The registry config layer is the only place that changes this, and only for Qwen3.8-27B trunk tags (`qwen3.8:27b` and `qwen3.8:27b-*`, never a longer lookalike) whose registry entry declares a `dflash` sidecar; it writes `speculation.dflash = "auto"`. Explicit global, per-model and one-shot config rank above the registry layer.

- **Paired draft file present** (the registry-declared `dflash.file` next to the model, or a `developer.dflash_draft` / `HIPFIRE_DFLASH_DRAFT` / `run --model-draft` path): the load wires it and DFlash runs.
- **Paired draft file absent:** nothing fails and nothing is wired. The load keeps MTP `auto` (the registry `mtp` head, as before this default).
- **Kill switch:** an explicit `dflash_mode off` wins over the registry layer: `hipfire config qwen3.8:27b set dflash_mode off` (per model), `hipfire config set dflash_mode off` (global), or `HIPFIRE_DFLASH_MODE=off`. The canonical key is `speculation.dflash = "off"`.
- **Unchanged:** every other model keeps the schema `off` (Qwen3.5/3.6, Flash-Next / `arch_id=16`, the `-draft-*` and sidecar tags, and 27B tiers with no `dflash` declaration such as `mq5*` / `mq6*`). `dflash_block` stays `auto`: 16 on a singleton load, 8 on a `serve.vmm_batch` serve with `continuous_batch_size` ≥ 4, and the IU4 verify gates ([above](#mechanism-knobs)) are untouched.
- **Output:** DFlash greedy decode is not promised byte-identical to AR (or to MTP): near an argmax tie the multi-row verify can pick the other token, and the text then diverges from that point. Batched output stays byte-identical to the singleton within a configuration.

Acceptance evidence for this default (fresh-process determinism, the batching C1 == C8 state oracle, MTP-vs-DFlash task-eval quality, the serve battery and chain text) is collected outside the repository; no pass is recorded here until a public, dated receipt exists.

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
  decision once. In a unified-memory process the room it checks is
  `MemAvailable` plus TTM's page-pool estimate (freed GTT pages the next GTT
  allocation takes first).
- **R9700 deployment-target VRAM budget** (32 GiB class ceiling in
  `preflight_alloc`) — **always runs**, on every arch, whether the host
  headroom guard is active or not. A configuration that does not fit the
  deployment target is refused regardless of this box's RAM.

`scripts/run-bounded.sh` (`HIPFIRE_MEM_CAP`) remains the hard cgroup backstop.

### Partial GPU offload (`memory.gpu_layer_budget` / `memory.moe_expert_budget` / `memory.offload_exec`)

| Key | Default | Values |
|---|---|---|
| `memory.gpu_layer_budget` | unset/null (fully resident on every model) | int 0–65536 \| `auto`/`-1` |
| `memory.moe_expert_budget` | unset/null (fully resident on every model) | int 0–65536 \| `auto`/`-1` |
| `memory.offload_exec` | `pcie` | `pcie` \| `cpu` \| `passback` |
| `memory.offload_passback_share` | `auto` | `auto` \| number in `(0, 0.5]` \| `0` |

Compat env: `HIPFIRE_GPU_LAYER_BUDGET`, `HIPFIRE_MOE_EXPERT_BUDGET`, `HIPFIRE_OFFLOAD_EXEC`, `HIPFIRE_OFFLOAD_PASSBACK_SHARE`. All four are `ModelLoad`-scoped (`hipfire config set memory.*`, per-model overlay, or env); there is no `--offload` CLI flag.

- **Counting.** Both budgets count layers kept **on** the GPU, not offloaded — `3` on a 64-layer model spills 61. `gpu_layer_budget` spills whole-layer prefixes (`[0 .. n-N)`); `moe_expert_budget` spills only routed experts. On an expert-only spill, attention, router and shared expert stay resident; embedding, `lm_head` and KV stay resident in either case.
- **Explicit auto-fit.** `auto`/`-1` opts into fit-based placement: the smallest spill that loads, **routed experts first**, whole layers only when the expert tier cannot free enough; a model that already fits stays fully resident. Unset/null budgets leave weights device-resident on every model, including Qwen MoE; there is no implicit spill. Manual `N` budgets are hard pins and are never silently widened — a pinned placement that does not fit refuses the load naming the knob.
- **Scope.** Explicit auto-fit runs only on spillable single-GPU text-trunk loads (`pp == 1`, no paged experts, no REAP keep-map); unset stays resident everywhere. Offload is implemented for Qwen3.5 dense/MoE (arch 5/6) and LFM2.5 dense/MoE (arch 11) — any other arch, `tp > 1`, `pp > 1`, EP, a VL tower, paged experts or a REAP keep-map with a configured spill fails the load closed.
- **LFM2.5 (arch 11) specifics.** The `moe_expert_budget` tier is the one that matters here: LFM2.5-24B-A2B is ~98% routed experts by bytes (11.6 GiB of its 12.7 GiB), so the whole-layer tier alone frees almost nothing and `auto` correctly takes the expert tier first. `memory.offload_exec=cpu`/`passback` is refused for arch 11 — its projections launch through the llama-crate GEMV helpers rather than `hipfire-dispatch`'s `execute_steps`, so the CPU-exec seam could not honour the request; only `pcie` runs. Routed experts are packed into one blob per projection per layer on **both** tiers: one `hip_malloc` per expert would waste ~90 MiB per MoE layer to allocation granularity (the placement census reads stored bytes, so the over-count made the search over-place and OOM mid-load), and per-expert host tensors would pay the 1 MiB host tail pad 128 times per layer. A spilled load keeps `embed_tokens` resident, and a `qt` 44/16/1 embedding is expanded to an F32 gather table plus a second copy for the tied `lm_head` — ~1.1 GiB of resident globals on the 24B, not the ~130 MiB its stored size suggests.
- **Execution.** `memory.offload_exec` decides **who multiplies**, never what spills. `pcie` (default; unset/empty/unknown fall back to it) runs the GPU kernels against host-mapped weights over the link. `cpu` executes those GEMVs on the host, and on a `moe_expert_budget` spill recomputes each selected routed expert's whole FFN on the CPU for AR single-token decode and any `<=4`-row batched forward (narrow speculative verify and accepted-prefix rollback included; wider prefill keeps the GPU grouped PCIe read). `passback` runs both engines at once on a spilled dense step, split by output rows per `memory.offload_passback_share`; routed experts on a `moe_expert_budget` spill take the same CPU splice as `cpu`. `cpu` and `passback` disable hipGraph capture for the model's lifetime and are refused together with a retained-replay (Redline) backend.
- **Load lines.** A spill prints `partial offload: {host} of {n} layers host-placed[, routed experts host on {k} layers], device weights {X} MiB of {Y} MiB available` plus the pinned-host report and the `kv reserved:` accounting line; a fit prints `partial offload: model fits - every layer resident`. A `cpu` load adds the `cpu exec: {covered}/{spilled} spilled layers fully covered …; uncovered quants: …` coverage line.
- **Spec.** Native MTP is opt-in on any host-mapped expert spill — see the `mtp_mode` row. Full write-up: [`env-vars.md`](env-vars.md) offload rows and [`plans/partial-gpu-offload-design.md`](plans/partial-gpu-offload-design.md).
- **Narrow verify routing.** Resident MoE `SpeculativeVerify` batches retain beta's legacy grouped route. Narrow Path1 applies only to host-mapped routed projections; the coalesced gfx1201 host-weight narrow kernel remains available.
- **CPU execution route.** Once MTP is enabled on a CPU-spilled Qwen3.5-family model, single-token decode, forced advances and terminal flushes automatically use the CPU-splice-capable executor. No `HIPFIRE_FORWARD_LOWERED=0` override is required. This execution routing does not change the MTP `auto` admission policy above.
**`passback` output is not reproducible run to run, by design.** `cpu` and `pcie` run a layer's GEMV on different engines, so their arithmetic need not agree, and `passback`'s row split is scheduled per process from the seeding probe and the online arm timings — two identical greedy runs mix the engines differently. Within a mode at a fixed split the output is deterministic, and `memory.offload_passback_share = 0` is byte-identical to `memory.offload_exec = cpu`. Pin the share before diffing `passback` output against a reference. Design: [`plans/partial-gpu-offload-design.md`](plans/partial-gpu-offload-design.md#622-pass-back-scheduled-co-inference-of-the-spilled-layers-memoryoffload_execpassback).

### Prompt / assistant-turn cache

| Key | Default | Values |
|---|---|---|
| `memory.prompt_cache_capacity` | `32` | int 0–1048576; maximum cached assistant-turn tokenizations (`0` keeps none). Env: `HIPFIRE_PROMPT_CACHE_CAP`. |
| `memory.prompt_cache_unbounded` | `false` | Remove the capacity bound. Env: `HIPFIRE_PROMPT_CACHE_UNBOUNDED`. |
| `memory.session_cache_bytes` | `8589934592` | int 0–1 TiB; byte budget (a ceiling, not a reservation) of the Qwen4 engine-owned session cache (cold-exact prefill snapshots shared across sessions, plus live continuation of the previous turn; `0` = off, cold prefill every turn). Snapshots take free device memory only while the context does not need it: they are skipped or evicted before VMM context growth would run short. Env: `HIPFIRE_SESSION_CACHE_BYTES` ([`env-vars.md`](env-vars.md)). |
| `memory.session_cache_disk_bytes` | `0` | int 0–64 TiB; byte budget of the session cache's disk tier (one budget for the whole `memory.session_cache_dir`, every model and build). Snapshots evicted from memory, and every snapshot at model unload, are written there and restored on demand, surviving daemon restarts of the same build; LRU over leaves, other builds' files evicted first. `0` = off. Env: `HIPFIRE_SESSION_CACHE_DISK_BYTES` ([`env-vars.md`](env-vars.md)). |
| `memory.session_cache_dir` | `""` | Directory of the session cache's disk tier; empty = `$HIPFIRE_HOME/session-cache`. One process uses it at a time (a second one runs without the disk tier). Env: `HIPFIRE_SESSION_CACHE_DIR`. |
| `memory.ple_hot_store_bytes` | `0` | `auto` or int 0–1 TiB; prewarm budget of the Qwen4 PLE hot-row store. The most frequently used rows of the per-layer-embedding table, ranked by the `<model>.plehot` sidecar (`hipfire-quantize ple-hot-rank`), are copied verbatim from the model file into the `<model>.plehotrows` hot-row pack (`hipfire-quantize ple-hot-pack`), grouped by rank. Rows the pack holds are read from it through the page cache and decoded by the same row decoder, so output is byte-identical to the model-file path. After load the first N bytes of the pack are read into the page cache (hottest tier first, then tier 0 once more so its pages are promoted to the active list) while no request runs: `auto` = `MemAvailable` minus 4 GiB at ready, an explicit N is that many bytes, both capped at the pack. The pack is clean page cache: the kernel drops it under memory pressure instead of swapping. Off by default (`0`); unified-memory GPUs only. Env: `HIPFIRE_PLE_HOT_STORE_BYTES`, paths `HIPFIRE_PLE_HOT_RANK_FILE` / `HIPFIRE_PLE_HOT_PACK_FILE` ([`env-vars.md`](env-vars.md)). |
| `memory.ple_format` | `auto` | `auto` \| `q4` \| `q8`; which PLE row table a Flash-Next (Qwen4) load reads. `q8` = the exact Q8F16 rows (embedded in the monolithic `qwen3.8:flash-next-mq4-xts`, or the `<model>.ple-q8` sidecar of a split base); `q4` = the 4-bit Lloyd-Max rows `<model>.ple-q4`, decode equal to the tested q4 table, derived from the Q8 rows on first load when missing (about 1 minute, progress on stderr; `hipfire ple derive --format q4 <model>` does it ahead of time). `auto` = q4 on gfx1151 (unified-memory Strix Halo), q8 elsewhere; `auto` never derives or downloads and falls back to the other available format with a note. An explicit `q8` on a split base needs `<model>.ple-q8` (`hipfire pull <tag> --ple q8`; `serve` downloads it on demand for registry models). See [Qwen4 PLE row tables](#qwen4-ple-row-tables). Env: `HIPFIRE_PLE_FORMAT` ([`env-vars.md`](env-vars.md)). |
| `memory.ple_prewarm_bytes` | `auto` | `auto` or int 0–1 TiB; idle-only page-cache prewarm of the q4 PLE table `<model>.ple-q4` after the model is ready (a rank-ordered table: hottest tier first, then tier 0 again, the hot-row store's prewarm; a row-ordered one: sequentially). `auto` = 15 GiB (the hottest rows, where the value per GB of warm rows is highest), capped at `MemAvailable` minus 8 GiB at ready and at the table; `0` = off (kill switch); an explicit N is that many bytes. The pages are clean, reclaimable page cache: dropped, never swapped, under memory pressure, and not charged in admission. See [Qwen4 PLE row tables](#qwen4-ple-row-tables). Env: `HIPFIRE_PLE_PREWARM_BYTES` ([`env-vars.md`](env-vars.md)). |
| `memory.ple_load_fill_bytes` | `auto` | `auto` or int 0–1 TiB; bytes of a rank-ordered q4 PLE table `<model>.ple-q4` read into the page cache **while the model loads**, hottest rows first, at most 0.2 GB/s so the weight upload keeps the drive. The upload streams the model file through the same cache and evicts the oldest pages first, so the fill re-reads evicted pages of its prefix (found with `mincore`), hottest first, until ready. Ready never waits for it; the resident prefix serves the first request, and the prewarm after ready (`memory.ple_prewarm_bytes`) continues at its end. `auto` = 14 GiB (the smallest prefix whose modelled prose and agentic 8K prefill is within 1% of the 15 GiB prewarm's); any value is capped at the table, at an explicit `memory.ple_prewarm_bytes`, and at the host RAM available at load start (`MemAvailable` plus TTM's page pool) minus the weights the load puts in host RAM minus 8 GiB. `0` = off. Clean, reclaimable page cache, not charged in admission. Env: `HIPFIRE_PLE_LOAD_FILL_BYTES` ([`env-vars.md`](env-vars.md)). |

Qwen AR and DFlash multi-turn reuse store each completed assistant turn as the
**verbatim generated token span** (whole envelope: full body tokens, plus
producer reasoning text when the turn thought). On the next turn, Jinja history
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
  elsewhere. `auto` never derives and never downloads: when the preferred
  file is absent it uses the other available format and prints a note on
  stderr naming how to get the preferred one.
- `q4` — always 4-bit rows. If `<model>.ple-q4` is missing it is derived from
  the local Q8 rows on first load (about 1 minute, progress on stderr).
  `hipfire ple derive --format q4 <model>` does the same ahead of time.
- `q8` — always exact rows (the kill switch for the q4 default on gfx1151). A split base needs `<model>.ple-q8`
  (`hipfire pull <tag> --ple q8`; `hipfire serve` downloads it on demand for
  registry models); a missing file is refused with that hint, never
  substituted.

The monolithic `qwen3.8:flash-next-mq4-xts` artifact keeps its PLE rows
embedded: with `q8` (or `auto` off gfx1151) the embedded Q8 rows are read
exactly as before; `q4` adds `<model>.ple-q4` next to it.

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

`memory.ple_hot_store_bytes` (above) also applies to the sidecars: the
hot-row pack is cut from, and verified against, the rows file in use.
With the q4 sidecar and no hot-row pack attached, the sidecar's data is
read into the page cache after the model is ready, while no request runs
(a rank-ordered table hottest tier first); `memory.ple_prewarm_bytes` (env
`HIPFIRE_PLE_PREWARM_BYTES`) sets the budget: `auto` (default) = 15 GiB,
capped at `MemAvailable` minus 8 GiB at ready and at the sidecar's data
bytes, `0` = off, an integer = that many bytes. A rank-ordered table's
hottest prefix is already read during the load, throttled, by
`memory.ple_load_fill_bytes` (env `HIPFIRE_PLE_LOAD_FILL_BYTES`, `auto` =
14 GiB, `0` = off), so the first request after ready finds those rows
warm; the prewarm continues after it. The pages are ordinary clean page
cache, dropped (never swapped) under memory pressure and not charged in
admission.

### Qwen4 trunk overlay sidecar

`<model>.trunk-overlay` (for the stock XTS artifact,
`qwen3.8-flash-next.mq4-xts.trunk-overlay`, ~305 MB) holds pre-encoded QT44
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

---

## Additional schema keys

Remaining typed keys that are not covered by a section above. Defaults and
ranges are the schema's; every key also appears in the generated
[lifecycle table](#lifecycle-status) with its legacy spelling and env alias.

| Key | Default | Values / notes |
|---|---|---|
| `memory.gpu_layer_budget` | `null` | `null`, `-1` (`auto`) or int 0–65536. Resident-layer budget for partial GPU offload: `N` keeps the **last N layers on the GPU** and spills the rest to system RAM (3 on a 64-layer model spills 61); unset keeps every layer on the GPU; `-1` defers placement to the engine, which currently keeps every layer on the GPU. Env `HIPFIRE_GPU_LAYER_BUDGET`. |
| `memory.offload_exec` | `"pcie"` | `pcie` \| `cpu`. Which engine executes the ops that read a spilled layer's weights: GPU kernels over the link (`pcie`) or the CPU. Never changes placement; the KV cache stays in VRAM. Env `HIPFIRE_OFFLOAD_EXEC`. |
| `model.deepseek4_experts_per_token` | `null` | `null` or int 1–6; DeepSeek V4 routed experts per token (`null` keeps the checkpoint default). |
| `hardware.deepseek4_compute_placement` | `"single"` | `single` or `dense-expert-split(dense=<selector>,experts=<selector>)` with two distinct device selectors. |
| `serve.vmm_batch` | `false` | bool; route eligible concurrent requests through the VMM continuous-batching executor (per-request VMM KV, one resident target). Off keeps the VMM singleton route. Env `HIPFIRE_SERVE_VMM_BATCH`. |
| `serve.continuous_batch_size` | `1` | int 1–256; number of eligible concurrent batch lanes (`1` = sequential), subject to memory admission. The runtime batch width is dynamic: it follows the lanes in flight and is not fixed at this value, and the 128-row/63-token chunk sizes are not lane limits. Env `HIPFIRE_CONTINUOUS_BATCH_SIZE`; the `serve --continuous-batch-size` flag takes the same range. |
| `serve.batch_spec` | `true` | bool; cross-request speculative verify rows on the VMM batch route (`false` disables batched speculation only). |
| `serve.batch_nonexact` | `false` | bool; permit explicitly labeled non-bit-exact batched arithmetic on the VMM batch route. |
| `serve.batch_sampled` | `true` | bool, applies only with `serve.vmm_batch`; admits sampled requests to the batch route only in the lane mode the singleton route selects for them. `false` keeps them on the serialized singleton route. |
| `serve.ui` | `false` | bool; serve the embedded chat UI at `/ui` on the serve listener. |
| `serve.retry_enabled` | `false` | bool; server-owned single retry on typed transient daemon failures. |
| `serve.retry_backoff_ms` | `50` | int 0–60000; backoff before that retry. |
| `prompt.jinja_chat` | `true` | bool; render chat prompts through the model Jinja template (`HIPFIRE_JINJA_CHAT=0` selects the hand-rolled scaffold). |
| `image.decode` | `"cpu"` | `cpu` \| `vcn` \| `auto` (VCN when probed, else CPU); VL image JPEG decode path. |
| `prefill.chunk_rows` | `512` | int 2–1048576; widened ordinary-prefill chunk ceiling in rows. Architecture default 8192 on exact gfx1100/gfx1151/gfx1201, 512 elsewhere; per-device VRAM admission may pick a smaller rung. Qwen4 rounds it down to 256 rows. Env `HIPFIRE_PREFILL_CHUNK_ROWS`. |
| `fusions.policy` | `"safe"` | `safe` \| `off`; certified fusion policy (individual kernel selection stays compiled). |
| `attention.ck_runtime_lib` | `null` | `null` or path; optional exact-architecture CK runtime sidecar (load or capability failure keeps native attention). |
| `attention.ck_workspace_bytes` | `0` | int 0–17179869184; preallocated bytes for the optional CK staged attention path. |
| `experimental.graph.forward` | `"auto"` | `auto` \| bool; hipGraph forward capture (`auto` follows architecture policy). Env `HIPFIRE_GRAPH`. |
| `experimental.graph.ar` | `true` | bool; allow autoregressive forward graph capture when otherwise eligible. Env `HIPFIRE_AR_GRAPH`. |
| `experimental.graph.moe` | `true` | bool; allow graph capture for supported MoE forward paths. Env `HIPFIRE_GRAPH_MOE`. |
| `kernel.calib_force_bf16` | `false` | bool; keep native-BF16 calibration teachers in BF16 (calibration only). |
| `kernel.gemma4_q8_fused_prefill` | `false` | bool; fuse eligible Gemma 4 Q8 prefill projections on gfx1100. |
| `kernel.gemma4_ple_batched_prefill` | `false` | bool; batch the Gemma 4 E-series PLE model projection on gfx1100. |
| `kernel.gemma4_batched_embedding_prefill` | `"auto"` | `auto` \| bool; `auto` enables validated gfx1100/gfx1201. |
| `kernel.gemma4_ple_activation_fused_prefill` | `"auto"` | `auto` \| bool; `auto` enables validated gfx1100/gfx1201. |
| `memory.unsafe_wsl_vmm_kv` | `false` | bool. UNSAFE until certified: allow the VMM KV backend under WSL/ROCDXG, where WDDM VA growth may alias earlier KV pages. Off: automatic KV selects legacy and an explicit `kv_backend = "vmm"` is refused. |
| `replay.unsafe_wsl_redline` | `false` | bool. UNSAFE until certified: allow Redline PM4/retained replay under WSL/ROCDXG. Off: the retained default falls back to the HIP graph and an explicit `replay.backend = "redline"`/`"shadow"` is refused. |

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
| `diagnostic.replay.gfx1151_cu_count` | `"all"` | `all` or an even CU count below 40 |
| `diagnostic.replay.gfx1151_entry_acquire` | `"system"` | `system` \| `agent` \| `vmem` \| `none` |
| `diagnostic.compiler.no_device_compiler` | `false` | bool; treat the device compiler as absent so precompiled code objects are used verbatim |

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
| `ple_hot_store_bytes` | `memory.ple_hot_store_bytes` | `HIPFIRE_PLE_HOT_STORE_BYTES` | 0 (off; `auto` = prewarm `MemAvailable` − 4 GiB of the hot-row pack) |
| `ple_format` | `memory.ple_format` | `HIPFIRE_PLE_FORMAT` | `auto` (q4 on gfx1151, q8 elsewhere; see [Qwen4 PLE row tables](#qwen4-ple-row-tables)) |
| `ple_prewarm_bytes` | `memory.ple_prewarm_bytes` | `HIPFIRE_PLE_PREWARM_BYTES` | `auto` (prewarm 15 GiB of the q4 PLE table, capped at `MemAvailable` − 8 GiB; `0` off) |
| `ple_load_fill_bytes` | `memory.ple_load_fill_bytes` | `HIPFIRE_PLE_LOAD_FILL_BYTES` | `auto` (read the hottest 14 GiB of a rank-ordered q4 PLE table during load at ≤ 0.2 GB/s; `0` off) |
| `session_cache_disk_bytes` | `memory.session_cache_disk_bytes` | `HIPFIRE_SESSION_CACHE_DISK_BYTES` | 0 (off) |
| `session_cache_dir` | `memory.session_cache_dir` | `HIPFIRE_SESSION_CACHE_DIR` | `$HIPFIRE_HOME/session-cache` |
| `devices` | `hardware.devices` | `HIPFIRE_DEVICES` (alias `HIPFIRE_DEVICE`) | unset |
| `allow_mixed_arch` | `hardware.allow_mixed_arch` | `HIPFIRE_ALLOW_MIXED_ARCH` | false unless `1` |
| `uniform_vram_tolerance_gb` | `hardware.uniform_vram_tolerance_gb` | `HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB` | unset |
| `mtp_mode` | `speculation.mtp` | `HIPFIRE_MTP_MODE` | `"auto"` |
| `mtp_k` | `speculation.mtp_k` | `HIPFIRE_MTP_K` | `"auto"` (4 for native Qwen4 MTP on gfx1151, else 3; an explicit 1–10 overrides) |
| `mtp_sampled` | `speculation.mtp_sampled` | `HIPFIRE_MTP_SAMPLED` | true |
| `mtp_ngram` | `speculation.mtp_ngram` | `HIPFIRE_MTP_NGRAM` | `"auto"` (on for Qwen4 native MTP on gfx1151 / gfx1201; `off`/`0` opts out) |

`kernel.lm_head_f16` maps the live Qwen3.5/3.6 loader compatibility control
`HIPFIRE_LM_HEAD_F16`; the duplicate unused `RuntimeConfig` member was removed.
The similarly unused `RuntimeConfig::dflash_draft` member was removed rather
than promoted. Draft selection remains part of the typed speculation/load
policy and filename/registry discovery.

Note: `ddtree_budget` and `ddtree_topk` are `FeatureFlags` overrides, not `RuntimeConfig` fields. The schema defaults (`0` and `4`) are what the CLI sends for `hipfire run`/`serve`. A zero budget reads as unset, so each call site falls back to its own default (the dense path uses 8 nodes when `speculation.dflash_tree` is on; the Qwen3.5 path stays chain-only); the non-zero `ddtree_topk` default is always sent and overrides a call site's own fanout.

Redline eligibility helpers `mq4r_redline_default` and `retained_redline_default` (same file) are **narrow runtime-default predicates** for replay backend selection: exact GPU arch `gfx1100`/`gfx1151`/`gfx1201` + case-insensitive `.mq4r` + pp=tp=1 (model-family agnostic, no `arch_id` gate; `gfx1200` and all other arches remain opt-in), plus Qwen3.5 dense (`qwen3_5`) and Qwen4 (Flash-Next, `qwen4`; MTP-off loads) plain-AR decode on exact `gfx1201` with pp=tp=1 and no drafter. A Qwen3.5-family process with a CPU-executed partial-offload spill (`memory.offload_exec=cpu` plus a manual layer/routed-expert budget or explicit `auto`/`-1` that resolves to a spill) gets no retained default, and an explicit non-`hip` backend there is refused at load. Unset/null budgets never request a spill. They are not general config keys and not Redline certification/registry admission. `replay.backend = "hip"`, the built-in `hip` config profile or another explicit backend selection disables the automatic default. Policy: [`REDLINE.md`](REDLINE.md).

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

**Count:** 295 schema keys, plus the `developer.<name>` namespace.

| Key | Legacy key | Env | Lifecycle |
|---|---|---|---|
| `attention.ck_runtime_lib` | `ck_runtime_lib` | `HIPFIRE_FLASH_ATTN_CK_LIB` | experimental |
| `attention.ck_workspace_bytes` | `ck_workspace_bytes` | `HIPFIRE_FLASH_ATTN_CK_WORKSPACE_BYTES` | experimental |
| `attention.flash` | `flash_mode` | `HIPFIRE_ATTN_FLASH` | stable |
| `attention.yarn_factor` | `yarn_factor` | `HIPFIRE_YARN_FACTOR` | stable |
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
| `serve.vmm_batch` | `serve_vmm_batch` | `HIPFIRE_SERVE_VMM_BATCH` | experimental |
| `speculation.ddtree_budget` | `ddtree_budget` | `HIPFIRE_DDTREE_BUDGET` | experimental |
| `speculation.ddtree_topk` | `ddtree_topk` | `HIPFIRE_DDTREE_TOPK` | experimental |
| `speculation.ddtree_tree_la` | `ddtree_tree_la` | `HIPFIRE_DDTREE_TREE_LA` | experimental |
| `speculation.dflash` | `dflash_mode` | `HIPFIRE_DFLASH_MODE` | stable |
| `speculation.dflash_adaptive_b` | `dflash_adaptive_b` |  | stable |
| `speculation.dflash_block` | `dflash_block` | `HIPFIRE_DFLASH_BLOCK` | stable |
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
