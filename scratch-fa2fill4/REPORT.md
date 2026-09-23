# gfx11 FA2 q8 prefill: warp-specialized K/V fill — KEEP on gfx1100 and gfx1151

**Verdict: KEEP.** The production FA2 kernel built with `HIPFIRE_FA2_FILL=1` hides the per-tile K/V fill almost entirely. The output is bit-exact on both cards. The change is gated to gfx1100/gfx1151, and `HIPFIRE_FA2_FILL=0` restores the all-wave fill.

## Change

On the Q16 tile (256 threads: 6 compute waves and 2 helper waves), the fill is split by phase:

- While the compute waves run QK(t), the helpers dequantize V(t) into the V plane.
- While the compute waves run softmax/PV(t), the helpers dequantize K(t+1) into the K plane.

The helpers hold the raw global bytes one phase ahead in registers, in their own loop. The compute waves' registers are therefore not affected. Dequant uses exact packed f16: the code is biased via `v_perm`, and `v_pk_mul_f16` rounds `s*c` once with RNE, which is the same value as the f32 product cast to f16. LDS stays at 32 KiB and each tile still has two barriers. VGPR use drops from 218 to 188.

Files: `kernels/src/attention_q8_0_fa2_gqa.gfx11.hip`, `crates/rdna-compute/src/attention.rs` (gate `q16_tile && developer_bool("HIPFIRE_FA2_FILL", true)`), `docs/env-vars.md`.

## Standalone screen

The screen used `scratch-fa2fill/` from `42b6fa385`: `fa2_bench`, NaN-poisoned output, every output bit compared with production, and median HIP-event times. Binaries were built locally, with no compile on hipx.

| Variant | gfx1100 `xtx` B8192 | gfx1100 `halo` | gfx1100 `odd` | gfx1151 `halo` | gfx1151 `odd` |
|---|---:|---:|---:|---:|---:|
| prod (ms) | 21.207 | 21.658 | 5.744 | 50.494 | 12.481 |
| **prodfill** | 16.275 (x1.303) | 17.709 (x1.223) | 4.855 (x1.183) | 38.456 (x1.313) | 9.740 (x1.282) |
| splitpk | 16.800 (x1.262) | — | 4.812 (x1.194) | 40.330 (x1.252) | 9.784 (x1.276) |
| nofill (inexact ceiling) | 15.525 (x1.366) | — | — | 38.039 (x1.327) | — |

- All variants except the ablations had 0 mismatches and 0 non-finite values.
- `q32splitpk` and `q32pk` (32-query tile) produced **478 mismatches** on the `odd` tails and were rejected.
- `split4pk`, `vpf2pk` and `f0pk` were no better than production.

## Production A/B

Setup: branch `gfx11-fa2fill-prod`, base official `4bd33ce40`.

- **pp8192 ROCprof:** the traced (second, uncached) request only. Both arms ran the same binaries in fresh processes, with the off arm set to `HIPFIRE_FA2_FILL=0`. Q8 VMM was asserted. The FA2 `VGPR_Count` was 224 off and 192 on, which confirms each arm ran its intended kernel. Full table: `evidence/rocprof-pp8192-summary.txt`.

| | FA2 off → on | Summed GPU off → on | Request wall |
|---|---:|---:|---:|
| gfx1100 | 347.37 → 268.69 ms (−22.6%) | 3076.55 → 3013.55 ms (−63.0, −2.05%) | 3148.0 → 3075.6 ms |
| gfx1151 | 830.55 → 616.97 ms (−25.7%) | 7639.43 → 7451.85 ms (−187.6, −2.46%) | 7688.0 → 7500.8 ms |

- **WT2 c24 q8/q8** (`run_wt2.py`, exact mode) is byte-identical to the `gate-first` baselines of `4bd33ce40`:
  - gfx1100: KLD 0.077271, sha256 `fbbaa065…`
  - gfx1151: KLD 0.075973, sha256 `79bbff9e…`
  - See `evidence/wt2-and-hygiene.txt`.
- **Hygiene:** Stacker2 leased both cards for the screen and for the A/B. Before and after, rocm-smi showed only gpusentry PID 2970. Daemons were stopped gracefully with `hipfire stop`. Binaries were built locally and copied with scp; there was no ABBA.

## Files

- `run_profile.py`: the pp8192 ROCprof arm runner. The `off` arm sets `HIPFIRE_FA2_FILL=0`.
- `summarize.py`: kernel table for the traced request, off vs on.
- `evidence/`: the summaries quoted above.
