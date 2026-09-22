# KvI8s: existing `Fp8` vs `Q8` KV on gfx1201 (card-B)

Worktree `wt-kvi8s` @ `34ccae74b` (branch `kvi8s`, off `wt-lloyd`).
Artifact `/home/kaden/qcal/qat/v25/qwen3.8-27b.mq4v2.xt.sym-a035.qat-r7s200.hfq`
(14,987,185,152 B; first two eval cells used the resolving symlink
`~/.hipfire/models/qwen3.8-27b.mq4-xt`, all later cells the absolute path —
same file).
GPU `GPU-e475645fe0200397` (card-B), HOME `ab1`.
Evaluator `cargo build --release -p hipfire-runtime --example eval_hipfire --features deltanet`;
daemon+CLI built in-worktree; `HIPFIRE_DAEMON_BIN=<wt>/target/release/daemon`.
No source change (no kernel edited). ab1 `config.toml` was varied for one
diagnostic pair only, then restored byte-identical (diff-verified).

## 1. Mechanism — what each mode actually does today

Both modes are selectable with no code change (`FULL_LADDER`).

| | `Q8` | `Fp8` (native E4M3FN) |
|---|---|---|
| Per-element encoding | symmetric int8, **no zero point**: `q = clamp(rn(x·127/amax), −127, 127)` | OCP E4M3FN codes via HW `cvt_pk_fp8_f32` (RNE, finite saturation) |
| Scale granularity | **per 32-element block** (8 blocks/head at D256): `scale = amax/127` stored f16 | **per (token, KV head)** over the full 256-elem head: `s = amax/448` stored f16 (zero row → 1.0; subnormal floor bump + next-up guard so `448·s ≥ amax`) |
| Zero point stored | No (zero maps to zero) | No (scale-only) |
| Row layout | 34 B/block = `[f16 scale][32×int8]`; 1088 B/side/token/layer at Hkv4/D256 | `[Hkv×D codes][Hkv f16 scales]` = **1032 B**/side/token/row at Hkv4/D256 |
| K vs V | K = Q8_0; V = Q8_0 by default (`v_mode: VMode::Q8`), **independently re-encodable** to lloyd2/3/4 via `--kv-v`/`HIPFIRE_KV_V` | **Indivisible pair: fp8 on both sides.** Non-q8 V is rejected at load and at dispatch. `--kv-v q8` is only the neutral spelling |
| VRAM | 2×1088 B/token/layer over the 16 KV layers; VMM `mapped_prefix=1927`, `physical_cap=max_seq=262144` | 2×1032 B/token/layer over the 16 KV layers; VMM `mapped_prefix=2032`, `physical_cap=max_seq=262144` |
| Decode FA (short ctx, scalar) | **Dequant first**: `q·(scale·code)` accumulated in f32 FMA | **Dequant first**: `q·(ks·e4m3_to_f32(code))` in f32 FMA |
| Decode FA (long ctx, flash tile) | **Dequant first**: wave-cooperative f32 dot + 2-pass softmax | **Dequant first**: same structure, E4M3→f32 decode per element |
| Prefill FA2 (gfx1201) | **Dequant at fill**: K/V decoded once per 64-key tile into fragment-major **f16** LDS planes, then **f16 WMMA** body (Q/K/V/P f16, accumulators f32). Admission narrow: batch 64..=512, batch%16==0, **eager-only** (never under graph capture) | **In place, native dtype**: stage-b copies E4M3 codes verbatim into fp8 fragment planes (no decode, no f16 rounding; scales read header-direct) and runs **fp8 (E4M3) QK + PV WMMA legs** with f32 scores/softmax-state/O. Default route is qresident (first in dispatch order), batch 64..32768, capture-safe |

File:line citations (all in `wt-kvi8s`):

- Q8 quant rule (symmetric, per-32, no ZP): `kernels/src/kv_cache_write_q8_0_batched.hip:49-56`
  (`scale = amax/127`, `q = max(-127, min(127, q))`, `out[0..2]=f16 scale`).
- Q8 layout math: `crates/saddle-core/src/kv.rs:1380-1384`
  (`total_blocks = n_kv_heads·(head_dim/32)`, `cache_bytes = physical_cap·total_blocks·34`).
- Q8 V default + VMode axis: `kv.rs:1430` (`v_mode: VMode::Q8`); `kv.rs:258-264` (VMode enum);
  `--kv-v` → V-only realloc: `crates/hipfire-runtime/examples/eval_hipfire.rs:588-596`;
  carrier gate (Lloyd-V needs FWHT-K): `crates/hipfire-arch-qwen35/src/carrier.rs:246-258`.
- fp8 scale/code rule: `kernels/src/kv_cache_write_q8_0_batched.hip:186-210`
  (`s16 = amax/448`, zero→1.0, `0x0001` floor, next-up bump, `cvt_pk_fp8_f32`).
- fp8 row layout: `kv.rs:2856-2859` (`[Hkv×D codes, head-major][Hkv f16 scales]`, 1032 @ Hkv4/D256);
  constructor: `kv.rs:2948-2969`; `quant_fp8=true`, empty scales: `kv.rs:3001`.
- fp8 indivisible K/V: `kv.rs:28-34` (enum doc), `kv.rs:2066-2070`
  (`set_v_mode_realloc` refuses fp8/bf16), `kv.rs:2947` ("K/V pair is indivisible"),
  `carrier.rs:322-326` (native tiers require neutral V),
  dispatch re-check: `crates/hipfire-dispatch/src/families/kv_tier.rs:344-351`;
  eval enforcement: `eval_hipfire.rs:449-454`.
- Q8 scalar decode (f32 FMA): `kernels/src/attention_q8_0_kv.hip:75-83`.
- fp8 scalar decode (f32 FMA after E4M3→f32): same file `:258-286`
  (comment `:253-257`: "ks factored out of the dot").
- Flash tile = f32 dot for both: `kernels/src/attention_flash_q8_0_tile.hip:5`
  ("sequential dot product + 2-pass softmax"); fp8 branch `:407+` (same order, E4M3 decode).
- Q8 FA2 = f16 planes + f16 WMMA: `kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:9-10`
  ("K and V are each dequantized exactly once per 64-key tile into two
  fragment-major f16 LDS planes"); KMODE doc `:48-60`; launcher comment
  `crates/rdna-compute/src/attention.rs:3644-3646` ("q8 cache always runs the f16 FA2 body").
- Q8 FA2 admission (narrow, eager-only): `attention.rs:3633-3642`.
- fp8 stage-b = in-place fp8 WMMA: `attention.rs:4461-4474` ("cop
...[18313 chars truncated]