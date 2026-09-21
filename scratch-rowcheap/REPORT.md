# Cheap row-global IU4 activation scale

## Verdict

**KILL.** The premise was refuted before the performance gates: coarsening the activation scale hurts the genuine IU4 route. The previously reported `0.037143 / 0.045510` row-global result came from an FP8 route selected by a stale predecessor binary, not from grouped-IU4 activation quantization. There is therefore no quality win for the per-tile-amax construction to preserve.

The route-fixed ladder is recorded by `A4K512` at correction commit `f8a16acef` (implementation commit `5ade50219`):

| activation scale | c2 KLD | c24 KLD | result versus K128 |
|---|---:|---:|---|
| K128 control | 0.064864 | 0.081199 | control |
| K512 | 0.072867 | 0.096131 | worse |
| row-global | 0.118151 | 0.158940 | much worse; fails the c24 hard budget of 0.10 |

No performance matrix was run after this refutation. Main explicitly stopped further gates.

## Route evidence and refutation

A fresh card-B trace of the current route-fixed row-global path is in `scratch-rowcheap/trace-current-row/bench_results.db`. For one 1024-token quality chunk it contains:

- 512 `quantize_int4_mmq_grouped` launches;
- 736 `gemm_mq4g256v2_residual_mmq_iu4_full_symfold` launches;
- 256 `gemm_mq4g256v2_residual_mmq_iu4_full_add_symfold` launches;
- zero FP8 projection GEMMs or FP8 activation packers.

That genuine IU4 run measured c1 KLD `0.091611`; the matching route-fixed row-global run measured c2/c24 `0.118151 / 0.158940`. The current K128 control reproduced c2/c24 `0.064864 / 0.081199`, and K512 measured `0.072867 / 0.096131`.

For comparison, rerunning the old `wt-rowa` binary produced the attractive c1 `0.030996`, but `scratch-rowa/trace-row/bench_results.db` contains FP8 producers and FP8 projection GEMMs, with zero grouped quantizer launches and zero IU4 projection GEMMs. Its low score is the route inversion, not a row-global IU4 result. `InversionDiag` independently confirmed there was no post-fix KLD run coupling the old low score to IU4 symbols.

## Producer and quantizer boundary map

| Producer | Workgroup ownership | Existing amax / quantization boundary | Per-tile construction mapped here |
|---|---|---|---|
| RMSNorm + AWQ + FWHT | one 256-thread workgroup owns a row; its eight waves loop over K256 groups | the shipping sidecar immediately reduces and emits two K128 `block_i4_128` blocks per K256 group | each wave iteration can publish its already-held K256 maximum; a later row quantizer consumes all tile maxima |
| SwiGLU + AWQ + FWHT | one 32-thread workgroup owns one K256 group for one row | eight final values per lane cover K256; the shipping sidecar reduces/quantizes two K128 blocks in the producer | publish one K256 maximum to `[row, tile]`, then quantize at the consumer boundary |
| Plain/AWQ rotate | one 32-thread workgroup owns one K256 group for one row | rotation leaves eight final values per lane and immediately emits two K128 blocks | publish one K256 maximum, preserving the existing F32 rotated row for the second stage |
| Sigmoid + AWQ rotate | same K256 ownership as plain rotate, with sigmoid folded before rotation | same immediate K128 sidecar boundary | publish one K256 maximum, then row quantize |
| Gated RMSNorm + AWQ + FWHT | one 64-thread workgroup owns two K128 heads (K256); both waves normalize, wave 0 performs the K256 rotation | wave 0 holds the final eight values per lane and emits two K128 blocks | wave 0 publishes one K256 maximum, then row quantize |
| Standalone int4 quantizer | one 256-thread workgroup covers K1024 in the K128 path; grouped mode owns one `(row, K-window)` | grouped row-global scans the F32 row for amax, then scans it again to pack | proposed second stage instead reduces only 20 maxima at K5120 or 68 at K17408, then reads the F32 row once to pack |

The ordinary producers have no grid-wide barrier, so a row maximum cannot be consumed inside their K256 workgroups. The actual code shape is therefore a producer-to-quantizer boundary: producers write a tiny scale plane, and one workgroup per row assembles the scale and packs the activation.

## Unused implementation plumbing

A complete default-off prototype remains in this branch solely as implementation evidence; it must not be enabled or shipped because the quality lever is dead.

- typed gate `kernel.a4_rowglobal_cheap` / `HIPFIRE_A4_ROWGLOBAL_CHEAP`, default false;
- per-reservation `[N, K/256]` F32 amax scratch plane;
- separate cheap producer helper that reduces the eight final values per lane and publishes one K256 maximum;
- cheap variants for RMSNorm, SwiGLU, rotate, sigmoid-rotate, and gated-norm producers;
- `quantize_int4_mmq_rowglobal_cheap`, one workgroup per row, reducing only the scale plane before one F32 packing pass;
- forced F32 producer output only under the gate, because the second stage needs the materialized row.

The shipping K128 helper and grouped-quantizer source were restored byte-for-byte to the inherited implementation. The release `eval_hipfire` build passed after that cleanup. The prototype was not taken through cost, wall-clock, decode, or battery gates after the quality premise was refuted.
