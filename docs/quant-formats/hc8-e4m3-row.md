# HC8E4M3Row (qt = 57)

Sidecar-only 8-bit format for the Flash-Next (Qwen4) hyper-connection mixer
weights: OCP E4M3 codes with one power-of-two f32 scale per output row. It is
never part of a base artifact (base validation refuses qt 57); it lives in the
`<model>.hc8` sidecar and, resident on the device, as `rdna_compute::DType::HC8E4M3Row`.

## Targets

Tensors whose name contains `hyper_connection` and ends with
`.input_mix_weight_down.weight` (`[320, 10240]`) or
`.input_mix_weight_up.weight` (`[10240, 320]`): the 48 layers' attn and mlp hyper
connections, the final `hyper_connection_mixer`, and the MTP ones when present.
The base tensors are BF16 (qt 16). A weight needs `m % 16 == 0` and `k % 32 == 0`.

## Quantization

For an `[m, k]` BF16 weight, per row `r`:

- `amax = max |w[r, :]|`; `s[r] = 2^ceil(log2(amax / 448))` (an all-zero row has
  `s = 1`), an exact power of two;
- `q[r, c] = E4M3_RNE(w[r, c] / s[r])`, OCP E4M3 (bias 7, max 448), round to nearest
  even, subnormals kept.

Decode is `q * s`; the dequantized weight is exactly a BF16 value (the scale only
shifts the exponent).

## Payload layout

`data_size = m * k + 4 * m` bytes:

1. `q`, swizzled, `m * k` bytes;
2. `s[m]`, little-endian f32.

The swizzle tiles `q` into 16-row by 32-K tiles of 512 B. Tile `(mt, kt)` (rows
`16*mt..16*mt+16`, K `32*kt..32*kt+32`) starts at byte `(mt * (K / 32) + kt) * 512`.
Lane `l` (0..32) of a wave reads 16 contiguous bytes at `l * 16` of the tile. Inside a
tile, byte `l * 16 + step * 8 + j` (`step` 0..2, `j` 0..8) holds the element at

- row `16 * mt + (l & 15)`,
- K offset `kk = 16 * step + 4 * (l >> 4) + (j & 3) + 8 * (j >> 2)` within the tile
  (absolute column `32 * kt + kk`).

Reference implementations: `rdna_compute::hc8::swizzle_hc8` (and its inverse
`unswizzle_hc8`), `rdna_compute::hc8::quantize_e4m3_rows`,
`rdna_compute::hc8_weight::hc8_payload_bytes` / `hc8_swizzled_offset`.

## Sidecar container (`<model>.hc8`)

An HFQM container (`write_hfqm_package_streaming`) whose `arch_id` is the base's.
Every tensor has the shape of the base tensor it replaces, `quant_type` 57,
`group_size` 32 (the K tile), `data_size = m * k + 4 * m`, in base index order.

The metadata JSON is exactly one object key, `hipfire_hc8_sidecar`:

```json
{"hipfire_hc8_sidecar": {
  "version": 1,
  "format": "e4m3_row_pow2",
  "base": {"file_bytes": 0, "index_sha256": "…", "sha256": "…"},
  "tensors": [{"name": "…", "sha256": "…", "base_quant_type": 16, "base_data_size": 0}]
}}
```

Binding (`hipfire_runtime::hc8_sidecar::bind_hc8_sidecar`, the cheap load-time check):
arch id, base file length, sha256 of the base bytes `[0, data_offset)`, manifest and
index name agreement, and per tensor: quant type 57, base tensor BF16 with the bound
size, equal shape, `data_size` equal to the payload size. Payload sha256s are verified
when the payloads are read for upload (`BoundHc8Sidecar::read_verified`). The base's
whole-file `sha256` is recorded at build time and not recomputed at load.

Build: `hipfire-quantize hc8 build --model <base> [--out <base>.hc8] [--force]`.
