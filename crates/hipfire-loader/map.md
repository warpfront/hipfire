# hipfire-loader — map

> **Status:** `production` (vocabulary owned by [`docs/GLOSSARY.md`](../../docs/GLOSSARY.md)).
> **Layer:** Composition root — top of the DAG's load path, above `hipfire-runtime`; arch crates hang off it via `Carrier`; see the layering table in [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md) and the carrier table in [`docs/architecture-ids.md`](../../docs/architecture-ids.md).

## Purpose

Single `load_model` dispatch for the daemon. Owns `LoadedModel` / `ModelState` closed enum, the object-safe `Carrier` trait and `REGISTRY` in [`src/carriers.rs`](src/carriers.rs) / [`src/lib.rs`](src/lib.rs) (impls stay in `carriers.rs` because `Carrier::load` returns `LoadedModel` and arch crates must not depend on the loader), `ModelSource::from_path` source × carrier routing (HFQ vs `SafetensorsSource` dir, `UNCLAIMED_ARCH_ID` sentinel), `batch_staging`, and `spec_build` RAII glues. Arch-agnostic callers destructure only via `as_arch_model{,_mut}`.

## Gotchas

- `load_model` requires exactly one matching carrier (0 → error; 2 → ambiguous) — add a carrier only after reserving its `arch_id` in `docs/architecture-ids.md`.
- Per-arch load bodies now live in the arch crates as `load_<arch>_bundle(src, ctx)`; `carriers.rs` is the thin registry that calls them — do not pull weight/config parsing back into the loader.
- `LoadedModel.ep` holds expert-parallel rank state separately from `state`; EP staging reclaims completed ranks on failure but a mid-constructor failure can leak that rank's partial allocs.
- Native MTP `auto` is gated by the producer capability `Carrier::mtp_measured_ar_floor` (default `false`; Qwen4 overrides with its crate's `MTP_MEASURED_AR_FLOOR = true`): `LoadedModel::effective_mtp_mode` resolves `auto` to `off` for every other carrier, and `qwen4_native_mtp_for_floor` keeps the Qwen4 table. A new carrier whose MTP lacks a measured AR floor must leave the default.
- The Qwen4 carrier reads `memory.session_cache_bytes` once at load and, when it is above 0, attaches the engine-owned session cache (keyed by the model's `CacheDomain`) to the bundle after the native MTP head is built, logging `qwen4 session cache: N MiB budget, N MiB per C-token snapshot; snapshots yield to context growth`. `0` attaches nothing, which also disables the bundle's live continuation; no prefix/radix cache is attached any more. With `memory.session_cache_disk_bytes` above 0 the cache also gets its disk tier in `memory.session_cache_dir` (logged as `session cache disk tier: …` or `… off: <reason>`).
- The Qwen4 carrier opens the PLE hot-row store (`memory.ple_hot_store_bytes` / `HIPFIRE_PLE_HOT_STORE_BYTES`, default `0` = off) before the weight sweep (`start_ple_hot_store`). It needs unified memory, Q8F16 PLE shards, the `<model>.plehot` sidecar and the `<model>.plehotrows` pack (paths overridable via `HIPFIRE_PLE_HOT_RANK_FILE` / `HIPFIRE_PLE_HOT_PACK_FILE`). Its worker verifies the pack against the model during the load; the row store reads the pack's rows from the pack once verified. Before admission the carrier snapshots the prewarm budget (`ple_hot_prewarm_budget`): an explicit N, or `auto` = `MemAvailable` − `PLE_HOT_PREWARM_RESERVE_BYTES` (4 GiB), capped at the pack. Unified-memory admission reserves this resident budget even though the pack is reclaimable page cache. After admission and session-cache attach, the carrier sets that same budget; the worker reads that much of the pack into the page cache (hottest tier first, then tier 0 a second time to promote it to the active list) while no PLE fetch ran for 5 s.
- With a rank-ordered q4 PLE table (`<model>.ple-q4`), the Qwen4 carrier starts the table's worker before the weight sweep (`start_ple_table_worker`): it reads the table's hottest prefix into the page cache at ≤ `PLE_LOAD_FILL_BYTES_PER_SEC` (0.2 GB/s) while the weights load, and re-reads pages of the prefix the weight file's cache traffic evicted, hottest first (`HotRowStore::load_fill`, `mincore` view). Size: `memory.ple_load_fill_bytes` / `HIPFIRE_PLE_LOAD_FILL_BYTES` (`auto` = `PLE_LOAD_FILL_AUTO_BYTES`, 14 GiB; `0` = off), capped by `ple_load_fill_bytes` at the table, an explicit `memory.ple_prewarm_bytes`, and `uma_memory::host_available_bytes` (`MemAvailable − CMA` + TTM pool: a previous process's freed GTT pages sit in the pool, outside `MemAvailable`) minus the weights the load puts in host RAM (unified memory: every non-external weight; discrete: host-mapped experts) minus `PLE_PREWARM_RESERVE_BYTES`; logged as `qwen4 ple load fill: …`. At ready the carrier sets the prewarm budget (`ple_prewarm_budget`), which ends the fill (ready never waits for it); the idle prewarm continues at the end of the resident prefix, and the worker is attached to the bundle (`attach_ple_table_worker`).
- Qwen4 context admission (unified-memory archs only; a discrete GPU keeps `QWEN4_DEFAULT_CONTEXT` and no sequence): an omitted `max_seq` resolves in `admission.rs` to a pending sequence at the model context (VMM only). After every load allocation the carrier measures free device memory (`Gpu::device_mem_info`) and runs `admission::qwen4_memory_admission` (target+MTP QSA growth including the opt-in F16 arena (`HIPFIRE_QWEN4_QSA_F16_ARENA`, 2 KiB/token/layer at 2 KV heads, rounded to four tokens), gather workspace, MTP request buffers, the resident PLE prewarm/load-fill budget, `QWEN4_MEMORY_HEADROOM_BYTES`), logging `qwen4 memory admission: …; card_cap N tokens` with its current-free-memory dependence. Automatic contexts take `min(model_ctx, card_cap)`; an explicit `max_seq` that cannot grow refuses at load under the default auto OOM guard, with `card_cap` the largest fitting context with current free memory: the limit moves with free host memory and the PLE prewarm budget (warns with `memory.oom_guard` off). A discrete GPU is unchanged. The session cache is never charged. Before allocating, a unified-memory fully resident load refuses (guard on) when weights + reserve exceed that free figure.
- Every free-memory sizing site uses `Gpu::device_mem_info` (or `Gpu::device_mem_info_for` for another device; both apply the same clamp), never `hip.get_vram_info` directly: on a GTT-backed APU it clamps `hipMemGetInfo` free to `MemAvailable − CMA` + TTM pool (`rdna_compute::uma_memory`; CMA pageblocks serve only movable allocations, so the CMA region, the larger of `CmaFree` and `CmaTotal`/`/proc/zoneinfo` `cma` (KHO scratch included), is subtracted, saturating, a missing field counting as 0; the load warns once when KHO scratch is ≥ 1 GiB).

- Qwen4 EP admission preflights geometry, physical devices and peer access and plans one `Qwen4EpPlan::seal` before rank 0 allocates. The seal binds the mesh epoch, rank/device origins, expert plans and source generation (base file, resolved PLE rows and trunk overlay). `Qwen4EpPlan::rank(rank)` supplies the `Qwen4EpRankSeal` accepted by `load_qwen4_rank`; peers open their own resolved source with `open_qwen4_rank_source`. Each rank checks its source generation before allocation, fulfills on the sealed mesh and binds the assembled bundle to that seal; an independent placement is not the EP load API.
- Private FN native-MTP behavior remains additive to that EP contract. Eligible single-rank loads compose one verified, settled `measured_load_identity` shared by `Arc` with the speculator and the frozen `measured_resource_snapshot`; chooser policy does not change numerical identity. Missing verified evidence, PP bands and EP ranks carry no measured descriptor and retain the incumbent route with a named reason. Resource snapshots are conservative allocation-plan/owner unions, not measured peaks or wider-execution qualification.
- Private `qwen4_preload_resource_plan` remains a metadata-only, single-rank, source-planned resource union at the already-resolved context: no weight fulfillment, source symmetry scan, sidecar verification/hash, GPU allocation, fit declaration or qualification. Missing metadata and checked overflow return named errors, not fabricated zeroes.
- FN placement and settled admission share the owned table/mask census: tier/runtime aliases count once, while stage tables remain disjoint. AUTO and hotset selection reserve those costs before choosing resident experts; staging's fixed point includes its table charge. The final capacity and RAM/GTT refusals remain authoritative.

## Crate map

<!-- crate-map:generated:begin -->

_Generated by `scripts/check-crate-maps.py` from the tree — do not edit inside the markers._

### Modules and exported symbols

- [`src/admission.rs`](src/admission.rs)
  - `admit_mesh`
  - `admit_qwen4_source`
  - `admit_source`
  - `admit_source_with_options`
  - `classify_vision`
  - `EdgeAdmission`
  - `effective_ep`
  - `EffectiveTopology`
  - `fn`
  - `from_override`
  - `growth`
  - `hipfire_arch_qwen4`
  - `hipfire_runtime`
  - `KvBackendHints`
  - `KvBackendRequest`
  - `LEGACY_NUMERICAL_POLICY`
  - `legacy_warning`
  - `MESH_TRANSPORT_ENV`
  - `normalize_mesh_shape`
  - `normalize_pp_bands`
  - `parse_mesh_transport`
  - `ple_shrunk`
  - `PpBands`
  - `qwen35_card_projection`
  - `qwen35_measured_fit`
  - `QWEN35_MIN_AUTO_CONTEXT`
  - `Qwen35CardProjection`
  - `QWEN4_EP_ARCHS`
  - `qwen4_ep_device_refusal`
  - `qwen4_ep_preflight`
  - `qwen4_ep_rank_devices`
  - `qwen4_kv_adaptive_requested`
  - `qwen4_memory_admission`
  - `QWEN4_MEMORY_HEADROOM_BYTES`
  - `QWEN4_PP_NUMERICAL_POLICY`
  - `qwen4_request_admission`
  - `Qwen4MemoryAdmission`
  - `Qwen4SourceAdmission`
  - `qwen_default_q8_enabled`
  - `RankBinding`
  - `required`
  - `resolve_qwen4_ple_rows`
  - `resolve_qwen4_ple_rows_hfq`
  - `SequenceResolution`
  - `SourceAdmission`
  - `SourceAdmissionOptions`
  - `visible_ranks`
- [`src/batch_staging.rs`](src/batch_staging.rs)
  - `BatchStaging`
  - `qwen_batch_weight_formats_supported`
  - `stage_continuous_batch`
  - `VmmStagingRequest`
- [`src/bench_topology.rs`](src/bench_topology.rs)
  - `bench_topology`
  - `bench_topology_facts`
  - `BenchRoute`
  - `BenchTopology`
  - `classify`
  - `EpKind`
  - `from_gpus`
  - `is_mesh`
  - `MeshFacts`
  - `PP_EP_EXCLUSIVE_MESSAGE`
  - `to_json`
  - `TopologyFacts`
  - `uses_carrier_hooks`
- [`src/carriers.rs`](src/carriers.rs)
  - `attach_qwen4_ep_peers`
  - `Cohere2MoeCarrier`
  - `Deepseek4Carrier`
  - `DotsOcrCarrier`
  - `EmbeddingGemma2Carrier`
  - `FluxDiffusionCarrier`
  - `Gemma4Carrier`
  - `Lfm2MoeCarrier`
  - `LlamaCarrier`
  - `load_qwen4_rank`
  - `MapleCarrier`
  - `MinimaxCarrier`
  - `MuseGlimmerCarrier`
  - `open_qwen4_rank_source`
  - `prepare_host_memory_for`
  - `Qwen2Carrier`
  - `Qwen35Carrier`
  - `Qwen35EmbedCarrier`
  - `qwen4_ep_alias_free`
  - `qwen4_ep_free_vram`
  - `qwen4_ep_vram_layers`
  - `qwen4_preload_resource_plan`
  - `Qwen4Carrier`
  - `Qwen4EpPlan`
  - `Qwen4RankLoad`
  - `Qwen4RankShared`
  - `Qwen4SessionSeed`
  - `rank`
  - `RaonCarrier`
- [`src/decide.rs`](src/decide.rs)
  - `DecideSnapshot`
  - `free`
- [`src/gguf_native.rs`](src/gguf_native.rs)
  - `admitted`
  - `admitted_dtype`
  - `ADMITTED_TYPES`
  - `analyze`
  - `bpw`
  - `census`
  - `gguf`
  - `GGUF_NATIVE_ARCH_ID`
  - `gguf_native_enabled`
  - `GGUF_NATIVE_ENV`
  - `GGUF_ROUTE`
  - `GgufAnalysis`
  - `GgufBadge`
  - `GgufCensus`
  - `GgufManifest`
  - `GgufManifestEntry`
  - `GgufNativeModel`
  - `GgufSourceMeta`
  - `inspect_report`
  - `is_mapping`
  - `line`
  - `load_qwen35_weights`
  - `mapping_refusals`
  - `mtp_refusal`
  - `of`
  - `open`
  - `open_gated`
  - `payload_bytes`
  - `Refusal`
  - `refusal_counts`
  - `RefusalKind`
  - `require_gguf_native`
  - `source_meta`
  - `tensor_bytes`
  - `TypeStat`
  - `weights`
- [`src/gguf_native/test_writer.rs`](src/gguf_native/test_writer.rs)
  - `base_kvs`
  - `base_tensors`
  - `bytes`
  - `DIM`
  - `FF`
  - `Kv`
  - `mtp_tensors`
  - `new`
  - `params`
  - `Spec`
  - `VOCAB`
  - `write_gguf`
- [`src/gguf_native/tests.rs`](src/gguf_native/tests.rs)
- [`src/lib.rs`](src/lib.rs)
  - `ack_dims`
  - `admission`
  - `applies_request_penalties`
  - `arch_label`
  - `as_arch_model`
  - `as_arch_model_mut`
  - `AsstTurnCache`
  - `batch_staging`
  - `bench_decode_route`
  - `bench_topology`
  - `BenchDecodeRoute`
  - `beta`
  - `budget`
  - `can_rewind_session_to`
  - `capture_row_elems`
  - `Carrier`
  - `carrier_for`
  - `carriers`
  - `clear`
  - `cohere2moe`
  - `cohere2moe_mut`
  - `contains_key`
  - `continuous_batch_route`
  - `ContinuousBatchRoute`
  - `decide`
  - `deepseek4`
  - `deepseek4_heterogeneous_mut`
  - `deepseek4_mut`
  - `Deepseek4HeterogeneousBundle`
  - `device_hidden_capture_enabled`
  - `device_hidden_log_end`
  - `dots_ocr`
  - `dots_ocr_mut`
  - `drafter_ctx_capacity`
  - `effective_mtp_mode`
  - `embeddinggemma2`
  - `embeddinggemma2_mut`
  - `ensure_vmm_ready_for_load`
  - `ep_admission`
  - `ep_eos_route`
  - `ep_prompt_route`
  - `EpArch`
  - `EpEosRoute`
  - `EpPromptRoute`
  - `EpState`
  - `Eviction`
  - `flux_pipe`
  - `flux_pipe_mut`
  - `free_gpu`
  - `gemma4`
  - `GEMMA4_EAGLE_DRAFT_LEN`
  - `gemma4_eagle_spec_len`
  - `gemma4_mut`
  - `generation_early_route`
  - `GenerationEarlyRoute`
  - `get`
  - `get_for_reasoning`
  - `gguf_native`
  - `GlimmerDrafterBundle`
  - `has_audio_encoder`
  - `has_video_route`
  - `has_vision_encoder`
  - `hipfire_arch_qwen4`
  - `host_capture_rows`
  - `img_route`
  - `ImgRoute`
  - `insert`
  - `is_diffusion`
  - `is_empty`
  - `is_generate`
  - `lfm2_vision`
  - `lfm2moe`
  - `lfm2moe_mut`
  - `llama`
  - `llama_mut`
  - `load_admitted_with_gemma4_drafter`
  - `load_model`
  - `load_model_ep`
  - `load_model_ep_admitted`
  - `load_model_ep_with_compressor_cache`
  - `load_model_ep_with_kv_mode`
  - `load_model_with_gemma4_drafter`
  - `load_model_with_kv_backend`
  - `LoadedModel`
  - `maple`
  - `maple_mut`
  - `maybe_evict`
  - `minimax`
  - `minimax_mut`
  - `model_dims`
  - `mtp_measured_ar_floor`
  - `muse_glimmer`
  - `muse_glimmer_mut`
  - `MuseGlimmerBundle`
  - `n_tokens`
  - `new_from_env`
  - `ple_serving_guard`
  - `qwen2_mut`
  - `qwen35`
  - `qwen35_embed_mut`
  - `qwen35_ep_moe_topology_refusal`
  - `qwen35_mut`
  - `qwen4`
  - `qwen4_mesh_mut`
  - `qwen4_mut`
  - `raon`
  - `raon_mut`
  - `registry_carriers`
  - `reset_session_state`
  - `rewind_session_to`
  - `rewrite_muse_glimmer_onyx_template`
  - `skeleton`
  - `skeleton_pp`
  - `slot_preflight`
  - `spec_build`
  - `unload_model`
  - `vision_config`
  - `vision_config_mut`
  - `vision_route`
  - `vision_weights`
  - `vision_weights_mut`
  - `VisionRoute`
  - `wire_verb`
  - `WireVerb`
- [`src/qwen4_slab.rs`](src/qwen4_slab.rs)
- [`src/spec_build.rs`](src/spec_build.rs)
  - `build_speculator`
  - `is_qwen35_dense_tp`
  - `model_slot`
  - `mtp_requested_k`
  - `Qwen35DenseTpGuard`
  - `Qwen35SlotGuard`
  - `take`

### Dependencies (from `Cargo.toml`)

#### path

- `hip-bridge`
- `hipfire-arch-cohere2moe`
- `hipfire-arch-deepseek4`
- `hipfire-arch-diffusion`
- `hipfire-arch-dots-ocr`
- `hipfire-arch-embeddinggemma2`
- `hipfire-arch-gemma4`
- `hipfire-arch-lfm2-vl`
- `hipfire-arch-lfm2moe`
- `hipfire-arch-llama`
- `hipfire-arch-maple`
- `hipfire-arch-minimax`
- `hipfire-arch-muse-glimmer`
- `hipfire-arch-qwen2`
- `hipfire-arch-qwen35`
- `hipfire-arch-qwen35-embed`
- `hipfire-arch-qwen35-vl`
- `hipfire-arch-qwen4`
- `hipfire-arch-raon`
- `hipfire-config`
- `hipfire-dispatch`
- `hipfire-runtime`
- `rdna-compute`
- `saddle-core`

#### external

- `serde_json`

#### dev

- `hipfire-arch-qwen4`
- `sha2`
- `tempfile`

#### build

—

### Reverse dependencies

Workspace crates with a path dependency on this crate:

- `hipfire-arch-qwen4`
- `hipfire-cli`
- `hipfire-daemon`
- `hipfire-engine`
- `hipfire-generate`
- `hipfire-runtime`

<!-- crate-map:generated:end -->
