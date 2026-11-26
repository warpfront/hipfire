### Performance: Inkling-2-Mini split-KV FA2 decode attention (gfx1201)

- **New: split-KV FA2 decode attention for Inkling-2-Mini on gfx1201
  (arch 24).** `rdna-compute/src/inkling2_attention.rs` adds
  `Gpu::inkling2_attn_decode_fa2`, a specialization of the shared gfx1201
  FA2 partial+merge family for H24/KV8/D128 (GQA3). It reads the same F16
  KV head planes, SWA ring slots and biased scores (relative bias minus
  sink, unchanged clamp) as the scalar kernel, feeds Q and P to the WMMA as
  hi/lo f16 pairs, and adds the sink once at the merge. The GPU forward
  selects it through `Inkling2GpuState::attn_fa2` (default off). Replay on
  R9700: 29x combined attention at 2K context, max |FA2 - scalar| 8.0e-6
  under a beta-calibrated budget of 3.5e-5, planted faults caught.
