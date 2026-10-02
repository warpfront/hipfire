# CPU SIMD coverage — hipfire-cpu

Coverage map for the `hipfire-cpu` matmul core's AVX2 row-dot kernels, against the
scalar decode that is now the ARM and non-AVX2 fallback (and the reference the
vector path is tested against). Companion to
[`qt-register.txt`](qt-register.txt) (the authoritative qt-number allocation).

**Every format the crate can decode has a vector kernel.** This file exists to say
which kernel, under which gate, and what the kernel still does not cover.

Source of truth in-tree:
- Dispatch gate `simd::row_dot_enabled(q, requested)` and the format→kernel map
  `simd::row_dot_avx2` — both exhaustive over `CpuQuant`, so a new format does not
  compile until it has a kernel and a stated feature requirement.
  (`crates/hipfire-cpu/src/simd/mod.rs`)
- Kernels in `crates/hipfire-cpu/src/simd/x86.rs` — AVX2 + FMA (+ F16C where
  noted), gated on `#[cfg(target_arch = "x86_64")]`. **ARM is all-scalar** — no
  NEON path.
- Format table: `crates/hipfire-cpu/src/quant.rs` (`CpuQuant`, `from_quant_type`,
  `group_elems`, `group_bytes`, `is_fwht_g256`).
- Row driver: `crates/hipfire-cpu/src/gemv.rs` (`dot_row_simd` picks the kernel
  when the gate says so, else `dot_row_scalar`).

## How the kernels are organised

Every kernel computes `Σ_j W[row][j]·x[j]` and differs only in how a group's
metadata and payload are laid out, so `x86.rs` is a table of layouts over one
decode core:

| piece | what it does |
|---|---|
| `codes8::<BITS>` | eight `BITS`-wide codes as `i32` lanes. Eight codes are exactly `BITS` bytes, so 1/2/3/4-bit chunks are one broadcast + one `vpsrlvd` + one mask, and the 40/48-bit 5/6-bit chunks use 64-bit lanes with the low dwords compacted back into eight lanes. |
| `codes_dot_sum::<BITS, PAYLOAD>` | `(Σ c·x, Σ x)` over a run of chunks — two vector accumulators, one horizontal sum each per group. |
| `uniform_group_dot::<BITS, PAYLOAD, CHUNKS>` | `scale·Σ(c·x) + zero·Σx` for one `(scale, zero)` header. |
| `v2_group_dot::<BITS>` | the V2 family's per-128 `fp16` header, each half scored then affinely corrected. |
| `codebook_lookup::<CB>` | the Lloyd tier: `vpermd` over a widened codebook (two tables plus a blend on the index's top bit for the 16-entry book). |
| `dense_row_dot!`, `affine_row_dot!`, `row_dot!` | the row drivers, monomorphized per format because a `#[target_feature]` body's features do not propagate into a closure or through a function pointer. |

Two consequences worth keeping in mind when adding a format:

- **Rotation is the caller's business.** The activation arrives already
  FWHT-rotated for the rotated formats, so a rotated format and its byte-identical
  unrotated twin share one kernel (qt 6 with 13, 8 with 15, 11 with 17, 9 with 18,
  51 with 19).
- **A group is `scale·Σ(c·x) + zero·Σx`, never a per-element `fma`.** That is why
  the TQ2/BQ1 pair needs no kernel of its own: `(code-1)·d` and `bit ? +d : -d` are
  the affine form with `(d, -d)` and `(2d, -d)`.

`load_le` reads *exactly* a chunk's `BITS` bytes — a wider load would be one
instruction cheaper and would run up to three bytes past the group on its last
chunk, which the caller's row slice licenses only for rows that are not the
tensor's last.

`Mq4G256` (qt 13) keeps a hand-unrolled nibble group (32 codes per 16-byte load)
rather than the `codes8` path; its recorded measurements were taken with it.

## Coverage — every `CpuQuant`

Header = per-group affine metadata. Payload = weight encoding, `BITS` bits per code
in `BITS`-byte chunks. "rot" = FWHT-256 baked into the weights (runtime must rotate
`x`). B/group at K=256; G128 formats are per-128-block. `—` in the B column = not a
grouped quant (plain tensor, 256-element "group").

| qt | name | B/256 | header | payload | rot | kernel | ISA gate |
|---:|---|---:|---|---|:---:|---|---|
| 1 | F16 | — | none | `f16` (2 B/elt) | no | `f16_row_dot` | AVX2 + FMA + F16C |
| 2 | F32 | — | none | `f32` (4 B/elt) | no | `f32_row_dot` | AVX2 + FMA |
| 3 | Q8F16 | 34/32 | `fp16` scale | `i8` codes | no | `q8f16_row_dot` | AVX2 + FMA + F16C |
| 6 | HFQ4G256 | 136 | f32 `scale`+`zero` | nibbles | no | `hfq4g256_row_dot` | AVX2 + FMA |
| 7 | HFQ4G128 | 72/128 | f32 `scale`+`zero` | nibbles | no | `hfq4g128_row_dot` | AVX2 + FMA |
| 8 | HFQ6G256 | 200 | f32 `scale`+`zero` | 6-bit packs | no | `hfq6g256_row_dot` | AVX2 + FMA |
| 9 | HFQ2G256 | 72 | f32 `scale`+`zero` | 2-bit packs | no | `hfq2g256_row_dot` | AVX2 + FMA |
| 10 | HFQ2G128 | 40/128 | f32 `scale`+`zero` | 2-bit packs | no | `hfq2g128_row_dot` | AVX2 + FMA |
| 11 | HFQ3G256 | 104 | f32 `scale`+`zero` | 3-bit packs | no | `hfq3g256_row_dot` | AVX2 + FMA |
| 12 | HFQ3G128 | 56/128 | f32 `scale`+`zero` | 3-bit packs | no | `hfq3g128_row_dot` | AVX2 + FMA |
| 13 | MQ4G256 | 136 | f32 `scale`+`zero` | nibbles | yes | `mq4g256_row_dot` | AVX2 + FMA |
| 15 | MQ6G256 | 200 | f32 `scale`+`zero` | 6-bit packs | yes | `mq6g256_row_dot` | AVX2 + FMA |
| 16 | Bf16 | — | none | `bf16` weights | no | `bf16_row_dot` | AVX2 + FMA |
| 17 | MQ3G256 | 104 | f32 `scale`+`zero` | 3-bit packs | yes | `mq3g256_row_dot` | AVX2 + FMA |
| 18 | MQ2G256 | 72 | f32 `scale`+`zero` | 2-bit packs | yes | `mq2g256_row_dot` | AVX2 + FMA |
| 19 | MQ2G256Lloyd | 72 | 4-entry `fp16` codebook | 2-bit indices | yes | `mq2g256lloyd_row_dot` | AVX2 + FMA + F16C |
| 20 | MQ3G256Lloyd | 112 | 8-entry `fp16` codebook | 3-bit indices | yes | `mq3g256lloyd_row_dot` | AVX2 + FMA + F16C |
| 30 | MQ4G256Lloyd | 160 | 16-entry `fp16` codebook | nibble indices | yes | `mq4g256lloyd_row_dot` | AVX2 + FMA + F16C |
| 31 | MQ5G256 | 168 | f32 `scale`+`zero` | 5-bit packs | yes | `mq5g256_row_dot` | AVX2 + FMA |
| 40 | TQ2G128 | 34/128 | `fp16` `d` | 2-bit codes, `(code-1)·d` | no | `tq2g128_row_dot` | AVX2 + FMA + F16C |
| 41 | BQ1G128 | 18/128 | `fp16` `d` | sign bits, `±d` | no | `bq1g128_row_dot` | AVX2 + FMA + F16C |
| 44 | MQ4G256V2 | 136 | per-128 `fp16` `[s z]×2` | nibbles | yes | `mq4g256v2_row_dot` | AVX2 + FMA + F16C |
| 45 | MQ4CG256 | 136 | packed `fp16` pair + 4 B pad | nibbles | yes | `mq4cg256_row_dot` | AVX2 + FMA + F16C |
| 47 | MQ6G256V2 | 200 | per-128 `fp16` `[s z]×2` | 6-bit packs | yes | `mq6g256v2_row_dot` | AVX2 + FMA + F16C |
| 48 | MQ5G256V2 | 168 | per-128 `fp16` `[s z]×2` | 5-bit packs | yes | `mq5g256v2_row_dot` | AVX2 + FMA + F16C |
| 49 | MQ3G256V2 | 104 | per-128 `fp16` `[s z]×2` | 3-bit packs | yes | `mq3g256v2_row_dot` | AVX2 + FMA + F16C |
| 50 | MQ2G256V2 | 72 | per-128 `fp16` `[s z]×2` | 2-bit packs | yes | `mq2g256v2_row_dot` | AVX2 + FMA + F16C |
| 51 | MQ2G256LloydU | 72 | 4-entry `fp16` codebook | 2-bit indices | **no** (unrotated) | `mq2g256lloydu_row_dot` | AVX2 + FMA + F16C |

### Gates

Two of them, and the format's kernel decides which applies
(`simd::features_for`):

- **AVX2 + FMA** — every kernel.
- **AVX2 + FMA + F16C** — the kernels that widen `fp16` metadata or `fp16`
  weights (`vcvtph2ps`): the V2 family, qt 45, the Lloyd tier, TQ2/BQ1, Q8F16 and
  F16. Every AVX2 part in practice has F16C, but "in practice" is not a hardware
  guarantee, so it is detected.

A forced `Some(true)` on hardware lacking the feature falls back to scalar rather
than faulting (`simd::use_avx2`), which is what keeps the predicate testable on
any runner.

### Not covered, and why

- **Formats with no `CpuQuant` variant** never reach the SIMD dispatch, because
  the crate has no decode for them at all (so a step over one stays on the GPU
  over PCIe): qt 0/4 (`Q4F16G64`/`Q4K`, GGUF-side), qt 5 (`Q8HFQ`, padded rows the
  group model cannot express), qt 14 (`MQ8G256`, `RotationPlan::Mq8Internal` —
  an int8-quantized activation, not a plain FWHT of f32), qt 21/24/32-37
  (HFP4/MFP4 family: per-row 16 B header + per-32 block scales), qt 28/29
  (PARO4G128(-T), a Givens rotation on the *activation*), qt 38/39
  (`MQ{2,3}G256GL`, MoE-indexed only). See `quant.rs`'s test module for the
  itemised reasoning.
- **Non-x86_64.** `avx2_available()` is `false`, so every format stays scalar;
  there is no NEON path.
- **MoE / non-qwen35 arches** are out of scope for the CPU-offload *work* (the
  kernels are arch-agnostic and would serve them, but nothing exercises them
  there).

### Which models this is exercised on

qwen35 dense, as before: `qwen3.5-2b.mq4` (qt 13), `qwen3.5-2b.mq3` (**qt 20**,
not 17 — the `-mq3` tags ship the Lloyd-Max 3-bit tier), `qwen3.5-2b.mq6`
(qt 15), `qwen3.5-2b.hf6` (qt 8), the `mq4v2` bodies (qt 44), plain qt 17, and
qt 49 for `qwen3.8-27b.mq3-xt`.

## Verification

`cargo test -p hipfire-cpu` (26 tests, GPU-free — the crate is a leaf):

- `every_cpu_decodable_format_has_a_kernel` — the kernel table in `simd::tests`
  names exactly the formats `from_quant_type` maps, and nothing else. A format
  that gains a kernel but not a row here fails.
- `row_dot_enabled_matches_each_kernels_feature_gate` — per format, the gate is
  exactly its kernel's feature requirement, a forced request never faults, and
  forced-off always loses.
- `avx2_and_scalar_agree_within_tolerance` — every format, four shapes up to
  k=12288, AVX2 against the forced scalar path (differing accumulation orders) at
  `1e-4` relative: f32 accumulation noise, not a decode bound. Measured worst
  case 2.3e-5; a real decode error is orders of magnitude larger (dropping a
  codebook table half: 7.1e-1; moving a payload offset: 3.5e+1). The fixture
  rewrites each family's metadata with distinct non-power-of-two values — the
  shipped fixture headers, codebooks and element values are powers of two, which
  would make every product exact and the comparison vacuous.
- `gemv_matches_the_f64_reference` — the whole public GEMV path, AVX2 active,
  against an independent f64 oracle over every format.

Beyond the crate, `cargo test -p hipfire-arch-qwen35 --test gpu_gemv_parity --
--ignored` (needs the GPU; run here on gfx1201) compares the **production GPU
launcher** against this crate's GEMV with the vector path active: real fixture
tensors (`qwen3.5-2b.mq4`, `qwen3.5-2b.mq6`, `qwen3.5-2b.mq3` — qt 13/15/**20** —
and `qwen3.5-9b.mq4`, with AWQ and pre-rotated arms, m up to 12288, k up to 6144)
plus an 18-check pre-rotated arm, and synthetic buffers for the rest — 23 formats
in its per-format summary. **Worst relative error is 9.3e-7** (`Mq4G256`, the
`+awq` real-tensor arm at m=6144 k=2048); every other format is ≤ 2.9e-7
(`Mq5G256V2` 2.9e-7, `Hfq6G256` 2.3e-7, `Mq6G256` 2.3e-7, `Mq3G256Lloyd` 2.2e-7,
`Mq6G256V2` 1.9e-7), then `Mq4G256V2` 8.9e-8 and `Mq3G256V2` 7.2e-8 — and the
remaining 15 of the 23 report exactly 0.

The test is pre-existing — what this work adds is that the formats it exercises
now take a kernel instead of the scalar decode. Two honest gaps: qt 11/12
(`Hfq3G256`/`Hfq3G128`) report `SKIP — production launcher has no kernel (…) on
this arch` on gfx1201, so their parity could not run (the crate's
canonical-table oracle still covers them); and F16/F32/Bf16 are absent from the
test's synthesised table, so those three rest on the crate tests alone.

Kernel-level microbench (m=4096, k=5120, same rayon pool both arms, best of 3 —
not a serve-path measurement), speedup of the AVX2 row dot over the scalar decode:

| qt | speedup | qt | speedup | qt | speedup | qt | speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20.2× | 9 | 14.0× | 19 | 21.6× | 45 | 13.0× |
| 2 | 13.2× | 10 | 8.8× | 20 | 8.6× | 47 | 14.4× |
| 3 | 7.9× | 11 | 7.7× | 30 | 16.5× | 48 | 20.2× |
| 6 | 13.6× | 12 | 6.9× | 31 | 19.8× | 49 | 7.7× |
| 7 | 13.9× | 13 | 13.4× | 40 | 10.7× | 50 | 13.3× |
| 8 | 6.6× | 15 | 6.6× | 41 | 10.2× | 51 | 21.1× |
| 16 | 15.4× | 17 | 7.2× | 44 | 15.9× | — | — |
| — | — | 18 | 12.8× | — | — | — | — |

No format regresses (the vector path is a throughput choice; the scalar path stays
reachable and exact).

## History

The first version of this map listed the formats as tiers ("Tier 0 — V2 family
skeleton", "Tier 1 — legacy affine", "Tier 2 — Lloyd codebooks", "Tier 3 —
asymmetric single-header") and ranked them by reuse × shipment. That listing was
never committed — it lived in the branch's development checkout — so the tier
names are not in this repository's history; what is, is the executed result, one
commit per group in the order the tiers set: among the tiered formats qt 49 came
first, then qt 44/47/48/50 (the V2 family), the affine formats (6-31, 40, 41,
45), the Lloyd tier (19/20/30/51), and the element/block formats (1/2/3/16).
qt 13's kernel predates this work (the module docs explain why it is not the
`codes8` path).
