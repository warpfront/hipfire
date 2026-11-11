# SP (short-prompt hybrid prefill) over mixed MQ4+W3 slab layers: design

Goal: mq3-xts short prompts (128..1,536 rows) stop paying the constant S1 sweep (about 5.9 s: 38 host layers x 1.1 GB / 7.1 GB/s). mq4 gets pp1105 837 tok/s from SP: per host layer the planner streams the top-k* experts (by routed-token count) and the CPU computes the rest (E1), one canonical combine.

## What already carries over
- CPU arm (E1): `moe_cpu_host_slot_outputs` / `moe_cpu_grouped_rows_bf16` take `CpuMoeExpert { gate_up_quant, gate_up, down_quant, down }` PER EXPERT, and hipfire-cpu `gemv_shared_sourced` runs jobs of different `CpuQuant`. The D2 slab owner (`gpu_forward/cpu_slab_owner.rs`) already maps arena experts to CPU quants (W3 planar R256/R128, MQ4G256V2/G128V2) for decode. So the CPU arm is codec-agnostic at its interface; `sp.rs` builds its experts from ONE `(gate_up, down)` dtype per layer, which must become per expert (arena codec).
- GPU arm: `execute_panel_rows` -> `panel_gemms`, now codec-split for slab (this branch); S1 pages/sources for slab arenas work (card-proven).
- hipfire-cpu: `gemm` (per output element dot, parallel over rows x tokens; AVX2 + AVX-512 dot paths in `simd/x86*.rs`, rayon pool, `splice_pool`). Planar W3 dot exists (D2 decode). No shared-decode multi-token GEMM; E1 groups tokens per expert (`gemv_shared_sourced` row-major jobs), so a weight row is read once per expert job.

## Gaps (the work)
1. Admission (`slowlink_sp.rs` `admit_mode`): hybrid accepts Aggregate and AggregateSymIu4 only; add SlabIu4 with a `slab_hybrid_ready` fact (like `f16_route_ready`). Window route stays SlabIu4 (no per-window override).
2. CPU arm input: E1 reads the window's rotated F16 activations `x_f16`; the slab route has A4 int4 sidecars instead, no F16 plane. Add a rotated-F16 plane to the slab window (hidden x rows x 2 B = 5 KB/row, 5.6 MB at 1105 rows; charged in the route region) produced by `rotate_x_mq_batched` before the window's A4 quantization.
3. Per-expert CPU quant: `SpCpuExpert` built from `slab.codecs[id]` (`expert_arena::codec_slot_bytes` already gives the extents).
4. Planner cost (`plan_layer_split`, `SplitCost::sp_hook_xtx`): stream cost per expert is one constant (0.405 ms = 2.61 MB at 6.45 GB/s); make it per expert (W3 1.92 MB = 0.30 ms, MQ4 0.405 ms) and the E1 table per codec (D2 shows the W3 CPU arm slightly faster per layer than MQ4: 265 vs 314 us).
5. Row table / combine: the hybrid's CPU rows (BF16) and the canonical combine are route independent, but the F32-tail-tile class change (tail tiles K2-ized) does not apply to slab (K1 tail <= 8 rows stays K1): verify the K1 cohort path coexists with the compact row table.
6. Tests: unit oracles per step (extent/quant per expert, planner with mixed costs, admission truth table, CPU-vs-CPU determinism, CPU reference == decoder oracle (hipfire-cpu planar W3 vs a scalar reference), panel pipeline vs masks). Row-exactness against the GPU K2 is NOT claimed (D2-style CPU arithmetic contract); the KLD gate follows later.

## Expected gain and honest limits
mq4's SP at 1K rows: TTFT 8.05 s -> 2.1 s. For mq3 the same structure gives roughly 2.1 s x (1.92/2.61 stream) on the streamed share and the CPU share (W3 decode costs about the same per byte as MQ4) => TTFT near 1.6-2.2 s at 1105 rows (500-690 tok/s); the constant sweep is the ceiling only if the planner streams everything. Not measured: all of it is a model until the card run.

## ETA
3-4 agent-days implementation (steps 1-5, unit oracles each) + 1 card correctness job (S==hybrid determinism, 3 repeats) + 1 timing job. Highest risk: step 2 (new F16 plane inside the charged slab route region) and the SP planner calibration for W3.
