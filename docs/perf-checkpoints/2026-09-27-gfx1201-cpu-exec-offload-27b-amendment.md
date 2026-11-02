# Amendment — 27B MQ3-xt footprint recipe and qt 49 coverage — 2026-09-27

**Lifecycle:** `historical`

**Amends:** [`2026-09-27-gfx1201-cpu-exec-offload.md`](2026-09-27-gfx1201-cpu-exec-offload.md)
(unchanged; this file adds a fixture and a footprint recipe it does not cover).
Same host, same binaries as that record unless stated.

**Disposition:** exploratory addition for the `qwen3.8-27b.mq3-xt` fixture —
**every one of its 497 projection tensors is qt 49 (`MQ3G256V2`)**, which the
base record does not exercise (its fixtures are qt 13/15/8/20). Not a product
baseline, not an admission, not a `docs/BENCHMARKS.md` claim.

## Fixture

- `qwen3.8-27b.mq3-xt` at `/path/to/models/qwen3.8-27b.mq3-xt`, 11,777,616,896 B,
  64 layers, `arch_id` 5. Non-AWQ quant types in the index: **qt 49 × 497**,
  qt 3 × 49, qt 1 × 305. Layer bytes ≈ 155 MB (first-8 average 154.6 MB).
- Card: gfx1201, 16304 MB total; host 28 GB RAM, THP `[always]`.

## Coverage under CPU execution (measured)

| `HIPFIRE_GPU_LAYER_BUDGET` | spilled | load | coverage line |
|---:|---:|---|---|
| 48 | 16 | **stalled twice**, at load layer 62/64, with 0 GB RAM free and host load ~6–10 from unrelated work; the KV line had not printed | `16/16 spilled layers fully covered; uncovered quants: none` (reached on the earlier run that did finish loading) |
| 56 | 8 | completes | `8/8 spilled layers fully covered; uncovered quants: none` |

**Recipe for this fixture on this host: budget 56 (8 spilled) or smaller.**
The two stalls were host-capacity: 16 spilled layers pin ~2.5 GB of
`hipHostMalloc` memory that the kernel cannot reclaim, on a host already holding
17 GB in `buff/cache` with 0 GB free. Nothing in the stall pointed at the CPU
path — the coverage line and KV allocation both complete at the smaller
footprint, and the same model at budget 48 loaded and decoded 160 tokens earlier
the same day while the host was quieter.

With the trace enabled (`HIPFIRE_CPU_EXEC_TRACE=1`) the spilled projections do
run on the CPU — all distinct step shapes report `quant=Mq3G256V2` and
`0 host-mapped steps still on GPU`, e.g. `m=12288 k=5120` at **5.70 ms/step**
(after the per-group fp16-header hoist; 19.95 ms before it, same shape).

## Numerical evidence for qt 49

qt 49 has no canonical `dequantize_to_f32` arm, so its oracle is the production
launcher on **real 27B tensors**:

```
HIPFIRE_PARITY_EXTRA_MODEL=/path/to/models/qwen3.8-27b.mq3-xt HIPFIRE_PARITY_EXTRA_QT=49 \
  cargo test -p hipfire-arch-qwen35 --release --test gpu_gemv_parity -- --ignored
  qwen3.8-27b.mq3-xt qt=49 q_proj  m=12288 k=5120  rel=2.391e-7
  qwen3.8-27b.mq3-xt qt=49 q_proj  m=12288 k=5120  rel=3.530e-7
```

Both arms of the rotation contract were re-measured on the build that carries
the per-group header hoist (`74b08860b`) and are unchanged from the pre-hoist build:
qt 44 8.931e-8, qt 47 1.933e-7, qt 48 2.857e-7, qt 49 7.187e-8 (synthetic),
qt 45/50 bit-exact.

## Known gap in this amendment

**No decoded text was read on this fixture.** Four attempts: one turn completed
the load and 160 tokens but failed the daemon's reasoning-budget gate
(`open think span at end of generation`), and on a failed turn no text is
released; two stalled in load under host memory pressure (above); the fourth
loaded at budget 56 and failed the same reasoning gate at 600 tokens. Reading
the 27B's text needs either `reasoning.effort=none` (a config change this record
does not make) or the reasoning-off `hipfire bench` route, which reports rates
but discards text. qt 49's correctness therefore rests on the real-tensor parity
above plus the coverage/trace evidence, not on an eyeballed completion.
