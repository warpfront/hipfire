### Performance: Inkling-2-Mini blocked prefill, F32 projections and router as tiled WMMA GEMMs (gfx1201)

- **Faster: Inkling-2-Mini blocked prompt prefill on gfx1201 (arch 24), about
  1.56x at pp512/pp2048.** The loop-controller, sink and dynamic-tap F32
  projections no longer run a per-row GEMV. They run as hi/lo-F16 WMMA GEMMs
  (`rdna_compute::inkling2_hilo`): the controller down projections share one
  launch, and gain/forget run fused with the blend tail. Error is F32-class,
  below a sequential F32 sum against f64. On a rocprof pp2048 trace these
  projections fall from 432 ms to about 20 ms.
- The router now runs `gemm_f16_wmma_mb8` with one wave per 16x16 tile
  (`Gpu::gemm_f16_wmma_cols_gfx12`, `KernelKey::GemmF16WmmaColsGfx12`). Its
  output is bit-identical, and its time on the trace falls from 115 ms to 2.7 ms.
- Daemon prefill on R9700, 4 fresh loads per arm against Stage 1: pp128 808 to
  1285 tok/s, pp512 1413 to 2215, pp2048 1427 to 2233. Paired prefill-scoring
  KLD is neutral (wt2 −2.8e-5, code24 −8.8e-5; both 95% CIs include 0).
  Row-exact mode keeps the per-row GEMV and stays bitwise.
