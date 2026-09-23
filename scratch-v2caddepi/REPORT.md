# V2CAddEpi: gfx1100 V2C ADD epilogue

**Verdict: KEEP the touch, KILL the fold. The change is exact.** The kill switch is `HIPFIRE_V2C_ADDEPI=0`, which restores the plain `_add`.
The production commit is `dda3a79fe` on stack-1 `1fcb53d4d`. The same change on the `mq4-lloyd` base is `gfx11-v2c-addepi` `75666ac35`.

## Lever

`gemm_mq4g256v2_residual_iu4_v2c_add_touch_gfx11` touches the tile's residual starting 16 epochs before the end of the K loop: one dword per 64-byte line, one line per thread per epoch, over epochs E-16..E-13. These loads ride ahead of the staging packet, so the epilogue's residual read hits cache. The output bytes match `_add`. The router in `gemm.rs` (V2C block, exact gfx1100) picks `_add_touch` for every V2C ADD unless the kill switch is set.

ISA was checked on gfx1100, ignoring address offsets. The `set`, `add` and `gate_up_silu` entries disassemble identically to stack-1. `_add_touch` disassembles identically to the screened `l64d16` object, at 183 VGPR against 179.

The fold is AddEpilogue's `726b3410f`: SET into a delta buffer, then the AWQ-i4 RMSNorm adds it in. On gfx1100 it is exact but loses to the touch. At K6144, N8192:

| arm | time |
|---|---|
| ADD + norm | 4.41–4.44 ms |
| SET + folded norm | 4.12 ms |
| touch + plain norm | 3.13 + 0.95 = 4.08 ms |

At K17408 the fold is flat. The folded norm costs +0.12 ms per call (1.08 vs 0.95 ms). The fold and the touch cannot be combined, so the fold was not ported.

## Evidence (hipx gfx1100, Stacker2 leases)

| check | result |
|---|---|
| S1 standalone oracle (`logs/S1-*`), non-periodic oracle inputs, N8192 | 886,046,720 values, 0 differing: 8 touch variants plus the fold, against production V2C ADD + norm |
| S1 timing, N8192, fwd/rev | K6144: 3.51/3.50 → 3.15/3.13 ms (−10.4%). K17408: 8.68/8.68 → 8.60/8.64 ms (−0.9/−0.5%). **Both ADD shapes combined: −3.6/−3.3%**, above the 3% kill line |
| S2 timing, N4096 (the production chunk) | l64d16, K6144: 1.64/1.71 → 1.62/1.61 ms. K17408: 4.42 → 4.34/4.33 ms. The l64d32 variant was 0.03–0.04 ms per call better at K6144 (untested follow-up) |
| pp8192 rocprof A (off) vs B (on), same binaries (`trace/`) | GPU 3100.27 → 3091.43 ms (**−8.85 ms, −0.29%**). Wall 3162.9 → 3152.9 ms. V2C ADD 755.89 → 744.09 ms (−11.8 ms). K6144: 1.584 → 1.525 ms/call. K17408: 4.321 → 4.288 ms/call |
| J1: JIT object (trace-B cache) `_add_touch` vs hipcc stack-1 `_add` | 298,844,160 values, 0 differing |
| WT2 c24 q8/q8 on stack port (`gfx1100/wt2/`) | 0.077271 off and on, kldseq byte-identical (sha256 fbbaa065…) |
| WT2 c24 q8/q8 on base branch (`gfx1100/wt2-base/`) | 0.076879 off and on, sha256 8f2b94bb…, byte-identical to the mq4-lloyd baseline kldseq (F1-lite B) |

The standalone norm objects are built without `-DIU4_A4_CANDIDATES=2`. Both oracle arms use them, so the fold comparison is self-consistent, but it does not reproduce the production norm bytes.

## Reproduce

On hipx, create a worktree at this commit and copy in release binaries built locally. Then run
`scratch-v2caddepi/proof.sh trace jit wt2` and `scratch-v2caddepi/run_standalone.sh <tag> oracle F R`.
