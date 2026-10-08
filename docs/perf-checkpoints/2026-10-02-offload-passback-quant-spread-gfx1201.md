# Offload pass-back: quant spread at one size (mq3 / mq4 / mq6) — gfx1201 — 2026-10-02

**Lifecycle:** `historical`

**Disposition:** the same-size, different-quant companion to
[`2026-10-02-offload-passback-model-spread-gfx1201.md`](2026-10-02-offload-passback-model-spread-gfx1201.md).
It is not a baseline, not an admission, and not a
[`docs/BENCHMARKS.md`](../BENCHMARKS.md) claim.

## Fixture (measured)

- Host: 1 × Radeon RX 9070 XT `gfx1201`, PCIe 4.0 ×16, HIP 7.2; Ryzen 7 7800X3D,
  16 cores, 28 GB RAM. Load uncontrolled.
- Source HEAD `289b38ad531013dddcdb808338cc518fc942a479`;
  `target/release/daemon` md5 `b35c25f032f80882d757512d0f8ee98a`;
  `target/release/hipfire` md5 `ee24d2671dc3480ac8949a43df71b156`.
- Prompt `benchmarks/prompts/gpu_offload_probe.txt`, md5
  `5835c71e471849b4a72e1dc8e39695e7`. `--spec off --backend noslots
  --workload stateless`, greedy, 128 tokens, one fresh process per run.

| tag | file | bytes | sha256 (registry) |
|---|---|---:|---|
| `qwen3.5:9b-mq3` | `qwen3.5-9b.mq3` | 4,569,785,344 | `c379dbbc90d7faf5e7281f01310b4e3f3e76587a6e951a0c7b6a809eeae5550b` |
| `qwen3.5:9b` (mq4) | `qwen3.5-9b.mq4` | 5,313,750,016 | `ba83acf5bfd5d4e334b0afc26d779734e31623bb7f74e807c3581dfecb3128ad` |
| `qwen3.5:9b-mq6` | `qwen3.5-9b.mq6` | 7,296,132,096 | `69b0e3b2be99a7fcab17f82bae2a2f1342ac32ee96d421347726815a69e78ce4` |

All three are qwen3.5 dense, 32 layers. **Pre-flight** (each loaded under passback):
`8/8 spilled layers fully covered`, `8/8 spilled layers splittable`, `uncovered
quants: none` for mq3 and mq6 — so a missing CPU decoder is *not* the reason for
anything below.

## Result (measured) — mq3

5 interleaved fresh-process rounds per point, arm order rotated by round:

| spilled | `cpu` | `passback` | `pcie` | passback vs cpu | pcie vs cpu | paired passback−cpu |
|---:|---:|---:|---:|---:|---:|---|
| 8 / 32 | 27.00 | **31.10** | 27.80 | **+15.2 %** | +3.0 % | 5/5 positive, +12.5…+25.7 %, median +14.1 % |
| 16 / 32 | 16.50 | **19.70** | 16.20 | **+19.4 %** | −1.8 % | 5/5 positive, +15.1…+26.7 %, median +18.7 % |

Against the same-size mq4 points from the model-spread record (identical fixture
and method):

| spilled | quant | `cpu` | `passback` | `pcie` | passback vs cpu | pcie vs cpu |
|---:|---|---:|---:|---:|---:|---:|
| 8 / 32 | mq3 (4.57 GB) | 27.00 | 31.10 | 27.80 | +15.2 % | **+3.0 %** |
| 8 / 32 | mq4 (5.31 GB) | 28.10 | 33.60 | 23.80 | +19.6 % | **−15.3 %** |
| 16 / 32 | mq3 | 16.50 | 19.70 | 16.20 | +19.4 % | **−1.8 %** |
| 16 / 32 | mq4 | 16.50 | 19.90 | 13.20 | +20.6 % | **−20.0 %** |

## mq6: no measurement — the build cannot decode it

**`qwen3.5:9b-mq6` does not decode in this build, in any mode** (not a pass-back
result, and not a download problem). It fails identically under `passback`,
`cpu`, and fully resident with no offload budget:

```
hipfire: daemon error: [internal retryable=false rolled_back=true attempt=1]
  forward_scratch decode: HipError(0): HIP error: HipError(0):
  unsupported gemv.swiglu_residual for / (hipError=0) (hipError=0)
```

Root cause, from the code. The model's tensors are the **legacy `MQ6G256`** (the
trace prints `quant=Mq6G256`; not `MQ6G256V2`). The table's own plan for that
dtype says the SwiGLU-residual should run **unfused** —
`KernelKey::gemv_steps(MQ6G256, WithSwiGLUResidual)` returns
`[SiluMulRotate, GemvResidual]` (`crates/hipfire-dispatch/src/types.rs:1028`) —
and the unfused `GemvResidual` arm for `MQ6G256` **is** wired (`gemv.rs:622`,
`MQ6G256 => gpu.gemv_hfq6g256_residual(…)`). So a working kernel and routing
already exist for this dtype. What fails is that the runtime's
`weight_gemv_swiglu_residual` (`crates/hipfire-runtime/src/llama.rs:1543`) takes
the **fused** launcher path, and `dispatch_swiglu_residual`
(`gemv.rs:637`) has **no `MQ6G256` arm** — it lists `MQ5G256` and `MQ6G256V2` but
not `MQ6G256` — so it falls to the catch-all. **This is a plan-vs-fused routing
mismatch, not missing kernel coverage.**

Plausible fixes, neither done here: (a) add
`MQ6G256 => hip!(gpu.gemv_hfq6g256_residual(w.buf, x_in, residual, m, k))` to
`dispatch_swiglu_residual`, mirroring the shipped non-swiglu `MQ6G256` arm
(`gemv.rs:622`) — i.e. reuse the kernel the codebase already routes this dtype's
residual to, not a guess; or (b) make the fused path consult `gemv_steps` and take
the decomposed route for dtypes whose plan is unfused.

Three further defects this exposed:

- **Phantom coverage.** `KernelKey::GemvMq6G256SwiGLUResidual` is declared
  (`types.rs:332`), mapped by `for_gemv_swiglu_residual`, and registered
  (`gemv_table.rs:160`), yet the fused launcher has no arm for it. That is a
  three-way inconsistency between `types.rs` / `gemv_table.rs` / `gemv.rs`, and
  `coverage_tests.rs` cannot catch it because it checks key↔table agreement, never
  that a dispatch arm (or a backing `.hip`) exists.
- **The error names no dtype.** The `_ =>` arms build
  `UnsupportedVariant { arch: "", quant: "" }`, so the message reads
  `unsupported gemv.swiglu_residual for /`. That omission is why 30 failed runs and
  a manual code read were needed; filling `quant` from `{:?}` of the dtype is a
  one-line diagnostic fix.
- **The registry advertises a tag the shipped build cannot decode**
  (`qwen3.5:9b-mq6`).

## Reading

1. **Quant changes the winner.** On mq4 the ordering is passback > cpu > pcie
   (+19.6 %, `pcie` −15.3 %). On mq3 it is passback > pcie ≈ cpu: `pcie` goes from
   −15.3 % to **+3.0 %** over `cpu` at 8 spilled. So "which single-engine route is
   best" is a function of the quant as well as the size — the same
   baseline-dependent shape seen in the model spread.
2. **Pass-back's own gain is smaller on mq3** (+15.2 % vs +19.6 % at 8 spilled,
   +19.4 % vs +20.6 % at 16) — but mq3's `cpu` arm is also *lower* (27.00 vs
   28.10) despite the smaller file, so this is not a simple "fewer bytes" effect
   (the 3-bit CPU decoder costs more per byte). The gain difference is small and
   the mechanism is **not** established by this record.
3. mq6 is unmeasurable here for a build reason, *not* a performance result — do
   not read it as "pass-back doesn't help mq6".

## Not measured / out of scope

- One host, one prompt, 128 tokens, greedy, `noslots`/`stateless`, uncontrolled
  load; a session re-run would move the absolutes.
- **Output identity is not verified for these quant arms** — and the reason is a
  property of the mode, not a gap in this record: `passback` is **not
  bit-reproducible across processes** (its scheduled share varies with the arm
  timings), so two identical greedy invocations differ; `cpu` and `pcie` are
  deterministic. Controls in
  [`2026-10-02-offload-passback-model-spread-gfx1201-amendment-1.md`](2026-10-02-offload-passback-model-spread-gfx1201-amendment-1.md) § 5.
- The mq6 dispatch gap is diagnosed, not fixed.
