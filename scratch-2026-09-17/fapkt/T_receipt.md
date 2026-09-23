# Slice T receipt — FaPacket transpose gate (ordinal 2, gfx1201)
## Exception granted by Main (post-T)
Main overruled the per-WG packet stop: the fill measures -27.5% in time and
is byte-exact; the per-WG packet gate was set on a per-wave assumption the
Q128 geometry changes (per-wave 635 vs 1184 is the like-for-like number).
Proceed to K1; the real gate is the integrated per-wave census <= 2,880 and
the end-to-end bench. (Recorded as instructed.)

Branch `gfx1201-fapkt` @ `eae7627e5` (plan cherry-picked from `61bad9f4e`).
Source: `scratch-2026-09-17/fapkt/transpose.hip` (untracked evidence).
Compiler: HIP 7.15.26333, AMD clang 23.0.0git, LLVM `8f497e0992fb7513f7f78a6f6b6f1056c375e961`
(plan provenance match). Build: `hipcc -O3 --offload-arch=gfx1201`.
Device: ordinal 2 only (`ROCR_VISIBLE_DEVICES=2`, `HOME=ab2`). Ordinal 3 untouched.
Run log: `T_run.log`. ISA: `transpose.s`. Resources: `T_build.log`.

## Functional: PASS, byte-exact
72/72 cases zero diffs: `compare: 72 cases, 0 diffs (K=0 V=0 S=0 R=0)`.
Coverage: kv_h {0,3} x ktile {0,64} x seq_len
{0,1,2,7,8,9,15,16,17,31,32,33,63,64,65,127,128,129} (tails 0..65, 127..129,
empty, single-key, odd/even, tile-boundary, 2-tile).
K pairs through frozen map: 0 diffs. V pairs through frozen map + transpose4
selectors: 0 diffs. Shared f16 headers as f32: 0 diffs. LDS poison 0xDEADBEEF:
no unwritten bytes. Dump redzones (2048 words): intact, both modules.

## Corrections used (allowance: implementation + 1 lowering correction)
1. p-loop ownership (correctness, not lowering): plan text
   `p=2t+(lane>>4), t 0..3` covers half the pairs — a wave's scratch holds
   only its key-half, so only consumer lanes with `(lane>>4)==(wave&1)`
   produce pairs; `p=t` over 8 iterations. Without this, V=8192/plane wrong.
2. Lowering: branchy transpose byte-selects (~640 cndmask/shift packets) ->
   hoisted constant `v_perm_b32` selectors (project `mq4_lut8` convention:
   `perm(hi,lo,sel)`, sel 0..3 lo / 4..7 hi; selectors
   0x06020400/0x03070105/0x05040100/0x03020706). Static 1494 -> 635.
   (A no-effect rbase hoist was also tried; compiler had done it.)

## Timing: PASS (+27.5%, gate >=20%)
HIP-event fill-only, ktile=0, seq_len=40000 (all-valid fast path),
200 iters, interleaved old/new, best of 5 rounds each side:
`fill best: old=6.732 new=4.883 us/fill (27.5% reduction)`.
Old module = shipped loops verbatim (128T/32768 LDS). New = paired fill
(256T/49408 LDS). Same binary, same clocks, same window.

## ISA/resources: MIXED
- No global byte-gather V path (8B vector loads): PASS.
- No decode/encode (0 WMMA, 0 cvt, 0 exp in `t_new_fill` ISA): PASS.
- 0 spills/scratch all four entries; fill-only VGPR 24: PASS.
- Fill packets: **MISS**. Exact dynamic path census (all blocks execute on
  the full tile; unrolled transpose issues under exec mask regardless):
  per-wave dynamic = 635 packets (K/stage prologue+loops 253 + transpose
  382, incl 93 `s_wait_dscnt` + 27 `s_delay_alu`); **WG total = 8*635 =
  5080 vs gate <=2560 (1.98x over)**. Same-method old WG = 4736
  (4*1184, plan S4). Per-wave 635 vs old 1184 (46% under); per live row
  5080/128=39.7 vs old 4736/48=98.7 (2.5x better) — but the absolute WG
  ceiling is missed: transpose waits/addr (~48/pair vs ~15 minimum) plus
  guard diamonds cannot reach 320/wave without grouped-read rescheduling
  (a second rewrite) and even optimistic grouping lands ~2800/WG. No
  required wait/delay was deleted to chase the number.

## Verdict: STOP per plan S10
T abandon rule fires ("V fill cannot meet <=2560"). K1 is NOT commissioned.
No production source was changed (K/A/D/J untouched); nothing to commit
beyond the plan cherry-pick `eae7627e5`.

## Main's transposed-score idea (evaluated, NOT adopted)
Checked against census before writing any QK leg: per-query max/sum/wmax
over 16 keys needs a 4-step 16-lane reduction x3 = 12 exchanges/KT16 vs
frozen 3 xor16, plus a packed-P cross-lane transpose for PV B (currently
zero-cost) and 8x Q traffic per lane; saves only ~7 cached header sk
loads/KT16. Net negative; also breaks frozen geometry (needs reviewer
approval to reopen). Sent to Main over hub.
