# S1 over the mixed-codec slab layer (mq3-xts) — design (not implemented)

Why: mq3-xts has no S1 ("slow-link prefill (S1): not selected (manifest has no packed routed gate/up and down expert payload)"),
so every 768-row chunk re-streams the host layers over the 7 GB/s link: pp16384 1,157 tok/s and TTFT 7 s (1105 tokens) against mq4's
S1 (1,892 / 2.3 s); S1 also frees VRAM (mq4: 7 resident layers vs 4 without S1; mq3: 5 without S1).

## What blocks it today (3 independent refusals)
1. Admission sizing. `expert_residency::routed_layer_payload_split` (line ~344) only sums `.mlp.experts.gate_up_proj` / `.down_proj`
   entries; a slab artifact has `*slab_arena` entries instead -> "manifest has no packed routed gate/up and down expert payload"
   -> S1 declines before any other check. `slowlink_admission::window_route_of` already maps arenas to `WindowRoute::SlabIu4`.
2. Layer check. `gpu_forward.rs::s1_layer_check` (~1723): `experts_all_gate_up_mq4` and "mixed-codec slab layer" (any slab codec != CODEC_MQ4).
3. Tile lowering. `hipfire-dispatch/src/pipeline/qt44_qt53_prefill.rs` ~3658 (`slowlink tile` classifier): the slab tile must be
   `SlabBuckets::Uniform(bucket != 0)` ("the layer is not uniform MQ4"); the ordinary (non-S1) tile already handles `SlabBuckets::Table`
   (cold K2 + MQ4 K2 over `qwen4_slab_tile_masks`) and `Uniform(0)` (`cold_gate_up`).

## Plan
- (1) Add a slab arm to the payload split: per layer, expert slot bytes = the arena's largest per-expert extent (MQ4 hot expert,
  2,611,200 B gate_up+down), so `PanelPageLayout` page bytes equal the mq4 page (83.6 MB) and the ring/lookahead charge is unchanged;
  copies still go by touched extents of the real (smaller, W3) payload, so link bytes shrink with the codec. Needs the arena's pointer
  table offsets in the S1 page-fill (`slowlink_load.rs` / `slowlink_prefill.rs` page filler reads packed per-expert offsets today).
- (2) Relax `s1_layer_check` to: slab codecs all in {MQ4, W3} AND `slab_prefills(arch, codec)` (rdna-compute, added by this branch)
  for every codec present. E8P keeps its own refusal (mq2: `mixed_source_priced_mq2r2_e8p` pre-mutation decline stays).
- (3) Add `SlowlinkTile::SlabK2Table` (and `Uniform(0)` cold-only): lower through the same `cold_gate_up/cold_down` + MQ4 K2 pair with
  tile masks as the non-S1 table arm, reading the binding's GLOBAL-id pointer tables (S1 already binds global-id tables and masks tile
  ids per binding, so the existing masks apply per page). The K1 few-row cohort (`slowlink_k1_panel`, 3 rows) uses
  `slab_route::k1_gate_up(layer, ..)`, which is already codec-table-aware.
- Gate: S1 changes only the route, not the kernels -> the S1-on prefill must be BYTE-IDENTICAL to the mq3 non-S1 (K2) prefill
  (qwen4_kld all-row logits digest, `--decode` twin) before any timing; then pp16384 + code-1k TTFT via timing-job.sh.
- Cost estimate: 1-2 agent-days incl. the S1 ledger/census unit tests (`slowlink_admission` has ~8.6k lines of tests that pin
  the SlabIu4 route numbers) and one identity card job. Touches hot files: slowlink_admission.rs (not hot-listed), gpu_forward.rs,
  expert_residency.rs, carriers.rs? (no), dispatch qt44_qt53_prefill.rs (hot-listed: claim first).
- Risk: S1 is default-on for mq4 on gfx1100; every edit must keep the mq4 route byte-identical (aggregate-only paths untouched).
