# hipfire-config — map

> **Status:** `production` (vocabulary owned by [`docs/GLOSSARY.md`](../../docs/GLOSSARY.md)).
> **Layer:** Operator — typed TOML config + migration and deterministic layer resolution, side-DAG; see the layering table in [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md).

## Purpose

Typed configuration schema and deterministic layer resolution for the operator plane. Owns the public vocabulary, legacy-key compatibility, `~/.hipfire/config.toml` + env + CLI layer merge, validation, field-level provenance, and `rocm` sub-tables. Deliberately independent of GPU/model crates so runtime crates consume resolved values instead of parsing hot-path config. See the `//!` docs in [`src/lib.rs`](src/lib.rs).

## Gotchas

- Resolution is deterministic layer-order (global → catalog → env → CLI profile) — out-of-order merges will mis-report provenance.
- DS4-specific knobs must never alter Qwen or the process-wide mixed-arch policy; respect the per-field scoping in `lib.rs`.
- `mtp_mode_admits` / `mtp_effective_mode[_name]` / `mtp_no_floor_decline_line` are the one pure producer-capability gate for `speculation.mtp = auto`: `auto` arms only a producer whose crate declares `MTP_MEASURED_AR_FLOOR = true` (Qwen4); `on` always arms, `off` never. The host-spill policy (`mtp_host_spill_policy`) only narrows an already admitted mode. Keep the gate arch-blind: no tag or card names.
- The registered `prefill.chunk_rows` value (`512`) is a schema fallback that the process config does not export; per-arch and per-model defaults (Qwen35 dense 8192, resident Qwen35 MoE memory-aware 4096) live in the arch crates.
- A3B prefill's 10 carried keys are registered in [`src/lib.rs`](src/lib.rs) in `TYPED_BOOL_DEVELOPER_KEYS`; [`src/intent.rs`](src/intent.rs) classifies their operator intent. They accept `true`/`false`/`1`/`0` (invalid values are rejected when the layer is built). An unset `developer_typed_bool` lookup is `None`, not a schema default; consumers resolve it through [`rdna-compute/src/prefill_route_capabilities.rs`](../rdna-compute/src/prefill_route_capabilities.rs), then apply their existing inventory/shape/state gates. Existing typed keys, including `developer.qwen4_slowlink_prefill`, are unchanged.

  | Typed developer key | Environment flag | Consumer route / prerequisite |
  | --- | --- | --- |
  | `developer.gdn_qk_indexed` | `HIPFIRE_GDN_QK_INDEXED` | `GdnQkIndexed`; raw-Q/K indexed sequential Q8 GDN launch |
  | `developer.gdn_moe_pre_fused` | `HIPFIRE_GDN_MOE_PRE_FUSED` | `GdnMoePreFused`; fused MoE GDN preamble |
  | `developer.gdn_moe_chunk_scan` | `HIPFIRE_GDN_MOE_CHUNK_SCAN` | `GdnMoeChunkScan`; generalized MoE scan with common and per-layer admission |
  | `developer.moe_router_fused_prefill` | `HIPFIRE_MOE_ROUTER_FUSED_PREFILL` | `MoeRouterFusedPrefill`; sealed fused router |
  | `developer.moe_shared_prepared_iu4` | `HIPFIRE_MOE_SHARED_PREPARED_IU4` | `MoeSharedPreparedIu4`; same-activation router/shared token reuse |
  | `developer.fa_prefill_pm` | `HIPFIRE_FA_PREFILL_PM` | `FaPrefillPm`; PM attention, resolved per KV codec |
  | `developer.moe_router_topk_multirow` | `HIPFIRE_MOE_ROUTER_TOPK_MULTIROW` | `MoeRouterTopkMultirow` (F4); nested beneath the admitted fused router |
  | `developer.moe_combine_v2` | `HIPFIRE_MOE_COMBINE_V2` | `MoeCombineV2` (F1); admitted IU4 expert combine |
  | `developer.fa_prefill_pm_qprep` | `HIPFIRE_FA_PREFILL_PM_QPREP` | `FaPrefillPmQprep` (F5); nested beneath PM attention for the same codec |
  | `developer.moe_shared_down_sigmoid_epi` | `HIPFIRE_MOE_SHARED_DOWN_SIGMOID_EPI` | `MoeSharedDownSigmoidEpi`; shared-down HIP V3 epilogue |

- All routes have one accepted exact-gfx1201 `QUALIFIED` row; explicit `0` kills and `1` opts in only where admission permits. PM attention and F5 resolve via `Gpu::fa_prefill_pm_flag` / `fa_prefill_pm_qprep_flag` for the actual codec: Q8 keeps its qualification and the existing two `FP8_KV_QUALIFIED` rows enable H16/KV2/D256 native-FP8 PM attention and Qprep on exact gfx1201. These rows qualify F16 Q with FP8 KV, not FP8-quantized Q. A parent PM `0` always refuses F5. The four late-H routes keep their nesting beneath their parent routes. These keys do not select KV storage; KV selection and FA2 selection remain independent and stackable, with all existing admission guards retained.

## Crate map

<!-- crate-map:generated:begin -->

Modules, exported symbols and dependencies are generated, not committed: see
`map.generated.md` in this directory (`cargo xtask regen` writes it).

<!-- crate-map:generated:end -->
