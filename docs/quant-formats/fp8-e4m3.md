# FP8 E4M3 weight formats (qt=55 `FP8E4M3B128`, qt=56 `FP8E4M3PT`)

Stock OCP-FP8 checkpoints (Qwen4 / Flash-Next, arch 16) carried into HFQM
**byte-for-byte**: e4m3fn codes and `weight_scale_inv` bits are stored verbatim,
so a payload is verifiable against the source safetensors manifest. Only gfx942
(MI300X, wave64) has kernels for them; every other arch refuses the artifact.

| qt | name | disposition | `DType` | `QuantType` |
|---|---|---|---|---|
| 55 | `FP8E4M3B128` | passthrough (`RAW_CODECS` row) | `DType::FP8E4M3B128` | `QuantType::FP8E4M3B128` |
| 56 | `FP8E4M3PT` | arch-loaded (Qwen4 PLE row store) | `DType::FP8E4M3PT` | `QuantType::FP8E4M3PT` |

Registered in [`qt-register.txt`](qt-register.txt). CPU reference (decode, layout
helpers, scans): `rdna_compute::fp8_e4m3` (`crates/rdna-compute/src/fp8_e4m3.rs`).
Neither format is FWHT-rotated: they consume activations in the natural basis.

## Element code: OCP e4m3fn

Byte `b`: sign `b>>7`, `E = (b>>3)&15`, `m = b&7`, bias 7.

- `E == 15 && m == 7` (`0x7F`, `0xFF`): **NaN**. Never stored; converters and
  decoders refuse it.
- `E == 0`: `±m · 2^-9` (subnormal; `0x00` = `+0`, **`0x80` = `-0`**).
- otherwise: `±(1 + m/8) · 2^(E-7)` (max `0x7E` = 448).

## qt=55 `FP8E4M3B128` layout

Logical shape `[E, R, K]` (`E = 1` allowed, i.e. `[R, K]`). The payload is `E`
consecutive expert blocks. Block `e`:

```
codes   u8[R*K]                         row-major, checkpoint bytes verbatim
s_inv   bf16[ceil(R/128) * ceil(K/128)] row-major, raw bits LE, verbatim weight_scale_inv
pad     zero bytes up to a multiple of 256 total
```

Block stride `S(R,K) = round_up(R*K + 2*ceil(R/128)*ceil(K/128), 256)`; codes of
expert `e` start at `e*S`, its scales at `e*S + R*K`; tensor bytes `= E*S`.
Scale `(i, j)` covers rows `128i..128i+127` and columns `128j..128j+127` (edge
blocks are ragged, never padded in the codes).

Helpers: `fp8_e4m3::{b128_block_stride, b128_tensor_bytes, b128_scales_offset,
b128_scale_count, b128_scale_dims}`.

Qwen4 routed experts (512 per layer):

| tensor | R | K | scale grid | `S` |
|---|---|---|---|---|
| `gate_up` | 1280 (rows 0..639 gate_proj, 640..1279 up_proj) | 2560 | 10 x 20 (scale rows 0..4 gate, 5..9 up; exact because 640 % 128 == 0) | 3,277,312 |
| `down` | 2560 | 640 | 20 x 5 | 1,638,656 |

## qt=56 `FP8E4M3PT` layout

Logical `[rows, K]`, one scale for the whole tensor:

```
header  16 B: bf16 scale bits (LE u16) at byte 0, bytes 2..16 zero
codes   u8[rows*K] row-major
```

Tensor bytes `16 + rows*K`; row `r` at `16 + r*K`. Per-element decode is the qt=55
formula with the single scale. Qwen4 PLE shards: 128 tensors of
`[2,500,012, 160]`. Helpers: `fp8_e4m3::{pt_tensor_bytes, pt_row_offset,
pt_header, pt_header_scale, dequant_row_pt}`. `DType::row_bytes(k)` is `Some(k)`
(code bytes of one row; the header belongs to the reader).

## Decode formula (numerical contract)

```
W_bf16 = bf16_rne( f32(e4m3fn(q)) * f32(bf16 s_inv) )
bf16_rne(x) = (u + 0x7FFF + ((u >> 16) & 1)) >> 16,   u = bits(x) as u32
```

This is the twin of `subq4/calib2/fp8_teacher.py:dequant_e4m3_block`. The F32
product is exact (4 x 8 significant bits) for every finite code and every scale
whose product stays F32-normal; decoders refuse non-finite scales, products that
would not be exact, and bf16 overflow, instead of rounding twice. Zero scale is
allowed (signed zeros out). Accumulation downstream is F32 in fixed K order, no
split-K, no atomics.

## FNUZ fold (gfx942 hardware decode)

gfx942 `v_cvt_f32_fp8` reads **e4m3fnuz** (bias 8, subnormal `m * 2^-10`, max 240,
no inf, `0x80` = NaN, no `-0`).

**Theorem.** Reinterpreting any OCP byte `b` as FNUZ yields exactly `fn(b)/2`:

- normal (`E` in 1..15): `fn = 2^(E-7)(1+m/8)`, `fnuz = 2^(E-8)(1+m/8)`: identical
  bits, half the value (`E = 15` lands at 240, in range);
- subnormal (`E = 0`): `fn = m * 2^-9`, `fnuz = m * 2^-10`;
- `0x00` is `+0` in both.

Hence with the doubled scale `s' = 2 * s_inv` (exact in BF16/F32 unless `s_inv`
is within a factor of 2 of the F32 maximum) the product `f32(fn(b)/2) * f32(2s)
= f32(fn(b)) * f32(s)` is the same exact F32 value, and
`bf16_rne(cvt(b) * 2s)` is **bit-identical** to the teacher for every finite
code. The kernel therefore decodes `cvt_f32_fp8(b) * (2 * s)` then rounds RNE to
BF16.

Exceptions, both handled by the decoder rather than by rewriting the artifact:

1. **`0x80` (`-0`).** FNUZ reads it as NaN. The kernel must decode it explicitly
   as `-0.0f` (not remap `0x80 -> 0x00` first). The teacher computes
   `f32(-0) * s = -0` (for `s > 0`), whose BF16 bits are `0x8000`; a `0x80 -> 0x00`
   remap gives `+0` = `0x0000`. The values are equal but the **BF16 bits differ**,
   and the identity gate is on BF16 bits. The artifact stores OCP bytes verbatim,
   including `0x80`.
2. **`0x7F` / `0xFF` (NaN).** FNUZ reads them as finite `+-240`, so they would be
   silently wrong; they are refused at conversion (`fp8_e4m3::scan_codes`
   reports counts; the Python teacher refuses them too).

CPU model of the hardware path: `fp8_e4m3::fnuz_fold_decode(b, s_bf16)`
(`e4m3fnuz_to_f32` + explicit `0x80` + multiply by `f32(2*s)` + `bf16_rne`).

## gfx942 QSA K/V rows (explicit `memory.kv_cache = "fp8"`)

Not an artifact format: the Qwen4 QSA context K/V arenas of `QsaKvFormat::Fp8Fnuz`.
A K or V row is `kv_heads * (head_dim + 2)` bytes: `kv_heads * head_dim` E4M3
codes, then one f16 scale per head (516 B at Hkv 2, D256; K+V 1,032 B per
token). The raw and pooled index keys stay F32. Per layer-token, 1,032 + 512 +
128 = 1,672 B (20,064 B over Flash-Next's 12 QSA layers).

The encoder feeds `(v / s) * 0.5` to the FNUZ convert, so the stored byte is the
OCP e4m3fn code of `v / s` (the Theorem above) and the decoder multiplies by
`2 * s`. The sole deviation from the OCP code is the negative-zero code: FNUZ
has no `-0` (`0x80` is NaN), so the quantized code `0x80` is stored as `0x00`.
That covers an input of exactly `-0` and a negative nonzero input that rounds
(RNE) to zero. Every other finite OCP code is stored unchanged, and a stored K/V
row never contains `0x80`.

## Verification

`cargo test -p rdna-compute --lib fp8` asserts BF16-bit identity of
`dequant_b128_block`, `dequant_row_pt` and `fnuz_fold_decode` against
`crates/rdna-compute/tests/fixtures/fp8_e4m3/golden.bin`: all 254 finite codes
(including `0x80`) x 49 BF16 scales (1.0, 2^-20, 2^-9, 0.0017, 2^-100, 2^10,
2^100, zero, plus 32 seeded random), generated from the teacher itself by
`gen_golden.py` in the same directory (`python3 gen_golden.py`, deterministic).
NaN-code refusal, non-finite-scale refusal, layout geometry (Qwen4 strides above)
and ragged-block scale selection are covered by the same module.
