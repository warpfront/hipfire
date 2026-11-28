### Performance: Inkling-2-Mini blocked prefill keeps its big buffers per load and decodes embeddings on the GPU (gfx1201)

- **Inkling-2-Mini blocked prefill (arch 24, gfx1201 typed-XTS loads) no longer
  allocates its largest per-visit buffers per visit.** The routed-expert
  producer rows, the grouped gate_up and down outputs and the loop-message mix
  now live in one `PrefillScratch` allocated at load for the product block size
  and counted in the load's VRAM summary (`prefill scratch … GiB (512 rows)`,
  0.157 GiB; the measured load delta grows by the same 162 MiB). These buffers outgrow the pool's 64 MiB parking limit at
  wider blocks, where each visit paid a `hipMalloc` and a synchronizing
  `hipFree`.
- **The prompt's embedding rows are decoded on the GPU.** The host gathers each
  block's packed qt44 rows (1,224 bytes per row) and a new kernel
  (`Gpu::inkling2_embed_qt44_rows`) decodes them bit for bit as the host
  decoder did (checked over all 201,024 rows of the artifact's table), instead
  of decoding 2,304 F32 values per row on the host and uploading them.
- Prefill logits are bitwise unchanged. pp2048 at the product 512-row block on
  R9700, 4 fresh loads per arm: 3,770 to 3,782 tok/s. At 2048-row blocks
  (developer flag) pp2048 goes from 466 to 425 ms: 0 `hipMalloc` / `hipFree`
  per call (was 168 `hipFree`) and a 4.9 ms lead-in (was 15.5 ms).
- `inkling_kld bench --block-rows N` (developer) runs the blocked prefill at
  `N` rows per block after re-sizing the load's prefill scratch
  (`Inkling2GpuState::resize_prefill_scratch`); `--n` may exceed one
  reference chunk.
