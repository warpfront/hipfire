# S7 receipt — stage-b CPU screen (`tmp_fa2_fp8_screen.rs`)

Base: wt-fa2a `gfx1201-fa2-fragorder` @ `e555cc08f` (stage-b base merge).
File: `crates/saddle-lab/examples/tmp_fa2_fp8_screen.rs` (sole S7 file).
Plan: `docs/plans/2026-09-18-gfx1201-fa2-stage-b.md` §8, §10/S7, §14 (unchanged S7 contract).
Model: `/home/kaden/.hipfire/models/qwen3.8-27b.mq4-xt`, gfx1201 ordinal 1 (ab1 home).
Env: `HIPFIRE_NORMALIZE_PROMPT=0 HIPFIRE_GRAPH=0 HIPFIRE_KV_MODE=q8`.

## 1. Arm-1 bit-identity gate — PASS

`--kernel-out` bundle, layer 35 / 4224 tokens / qstride 4 / default prompt:

| file | md5 | retained ref (`fa3ping/screen_base/md5.txt`) |
|---|---|---|
| `arm1/layer35/kern.O.f32` | `1f7a1183a751de7555066bcf5a1ca11c` | match |
| `arm1/layer35/kern.gate.f32` | `8c0287afddd1a3aef7f43496c3127c23` | match |

Engine buffer is POST-gate at this HEAD (gate probe: raw 13.5, gated-once 2.4e-7;
residual = host-vs-device sigmoid rounding, not a tap mismatch).

## 2. Per-arm table vs arm 1 (pre-gate), layer 35

Default prompt (ids 0..4224), `layer35.log`:

| arm | max-abs | tail-1% |
|---|---:|---:|
| arm2 cpu-q8ref (self-check-B mate) | 9.538e-4 | 1.207e-5 |
| arm3 Q0 format-only | 7.964e-1 | 1.939e-2 |
| arm4 fp8-QK + exact weighted PV | 1.916e0 | 3.179e-2 |
| arm5 stage-b route N | 1.915e0 | 3.209e-2 |
| arm6 stage-b route Q | 2.042e0 | 3.301e-2 |

Second prompt (`--prompt-range 4096`, ids 4096..8320), `layer35_p4096.log`:

| arm | max-abs | tail-1% |
|---|---:|---:|
| arm2 | 2.718e-4 | 1.494e-5 |
| arm3 | 7.915e-1 | 2.280e-2 |
| arm4 | 2.559e0 | 4.197e-2 |
| arm5 | 2.559e0 | 4.229e-2 |
| arm6 | 2.461e0 | 4.158e-2 |

Score max|Δ| vs exact — default: q0 3.341e-1, qk/sb 4.373e-1, rq 4.544e-1;
p4096: q0 3.308e-1, qk/sb 4.204e-1, rq 4.346e-1.
arm1 production gap vs exact: 2.205e-1 / 3.369e-3 (default), 3.765e-1 / 5.039e-3 (p4096).
(|O_ref| p99 here is 2.34 pre-gate; the old screen_base 0.73 was post-gate —
different measurement point [INFERENCE], arm-1 O itself is bit-identical.)

## 3. P-underflow stop gate (≤0.1% of positive valid weighted entries) — PASS both routes, both prompts

- Default: N 1838/9287424 = 0.0198%, Q 1853/9287424 = 0.0200%, lost mass 0.0000%.
- p4096: N 1696/9287424 = 0.0183%, Q 1685/9287424 = 0.0181%, lost mass 0.0000%.
- True-f32-exp zeros: 0; all-zero V-code (q,h): 0.

## 4. Tap census (§8.1(B) tool): raw pre-quant K/V per FA layer

Clip fractions under the per-256 native row scale (denominator = all dims).

| layer | plane | amax | p99.9 | p99 | lost-to-zero | subnormal |
|---|---|---:|---:|---:|---:|---:|
| 3 | K | 1.057e1 | 5.674 | 3.600 | 0.0007% | 0.0115% |
| 3 | V | 1.048e1 | 3.592 | 2.299 | 0.0007% | 0.0093% |
| 19 | K | 1.808e1 | 4.978 | 3.834 | 0.0006% | 0.0093% |
| 19 | V | 1.555e1 | 5.724 | 2.893 | 0.0010% | 0.0118% |
| 35 | K | 1.958e1 | 7.370 | 4.797 | 0.0015% | 0.0138% |
| 35 | V | 3.734e1 | 6.556 | 3.420 | 0.0008% | 0.0133% |
| 35 p4096 | K | 1.950e1 | 7.029 | 4.734 | 0.0011% | 0.0151% |
| 35 p4096 | V | 3.721e1 | 6.538 | 3.258 | 0.0010% | 0.0133% |

Layers 5, 20 are **not** FullAttention (linear/DN layers project into
`dn_qkv_batch`, never `fa_k/fa_v`): no own FA K/V taps exist, census N/A.
Their tap buffers hold stale rows from the last FA layer ≤ want in the
truncated prefix (layer 5 → layer 3's rows, layer 20 → layer 19's rows —
amax cross-confirmed against the FA-layer census above). Depth coverage for
the attribution tool therefore uses FA layers 3/19/35 instead.

## 5. Route-Q requant (§2.3)

Header-only `s_row` over all 4224 rows × 4 heads, `448*f32(s_row) >= amax_row`
asserted per row (33,792 assertions, zero failures):
default s_row ∈ [3.881e-3, 8.337e-2]; p4096 ∈ [3.717e-3, 8.307e-2].
Codec self-test pins the rule (zero row → 1; uniform ±127; skewed argmax;
zero-scales-with-codes rejected).

## 6. Self-checks

- A (stored-q8 vs raw taps, all rows): default worst 1.468e-1 tie-adj 11;
  p4096 (via replay) worst 1.415e-1 tie-adj 8 — taps OK.
- B (arm2 vs arm1): 9.538e-4 default, 2.718e-4 p4096 (WMMA-tolerance).

## 7. Replay protocol — reproduces capture exactly

`--replay dump35` and `--replay dump35_p4096` (GPU-free) print identical arm
tables, underflow numerators and census values to the capturing runs
(`replay35.log`, `replay35_p4096.log`). Replay additionally runs tap census
and self-check A from the bundle.

## 8. Prompt note (per assignment)

The screen takes `--prompt-ids FILE` / `--prompt-range START` (no text
prompt — no tokenizer on this path). No ag-corpus token file exists in the
receipts (`Fp8Kv/q0/ag/` holds KLD logs only), so an ag-faithful prompt
cannot be run from here; the second prompt above is an id-offset prompt
reported as such. WT2-style = default 0..4224.

## Files (all under `scratch-2026-09-17/StageB/Screen/`, untracked)

`layer35.log`, `layer35_p4096.log`, `replay35.log`, `replay35_p4096.log`,
`census3_19.log`, `census5_20.log`, `arm1/layer35/`, `arm1_p4096/layer35/`,
`dump35/layer35/`, `dump35_p4096/layer35/`, `release_build.log`, this receipt.
Binaries used: debug for the default-prompt run, release for p4096/replays
(per Main: debug CPU math 481 s → release ~2–3 min for the same work).
