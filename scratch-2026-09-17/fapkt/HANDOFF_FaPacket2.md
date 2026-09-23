# FaPacket → FaPacket2 handoff (2026-09-19 ~05:40 UTC)

## State: everything committed, tree clean
All FaPacket work is in commit `cc5018a69` (WIP FA packet-minimal Q128 body)
on branch `gfx1201-fapkt`, on top of `eae7627e5` (plan) on top of `61bad9f4e`.
`git status` is clean except untracked `scratch-2026-09-17/`. No uncommitted
code changes exist. I am stopping touching the paths below as of this note.

## Changed paths (all in cc5018a69, +1148 lines, nothing else touched)
- `kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip` (+477): packet region
  (`fa2_stageb_packet_body`, direct/partial/merge entries) behind
  `HIPFIRE_FA2_KMODE==8 && HIPFIRE_FA2_FP8 && HIPFIRE_FA2_PACKET`.
- `crates/rdna-compute/src/attention.rs` (+414): packet direct launcher,
  stage-b split + packet split bench launchers, shared impl.
- `crates/rdna-compute/src/kernels.rs` (+16): PACKET SRC const.
- `crates/rdna-compute/src/feature_flags.rs` (+6): `gfx12_fa_packet`
  (default off) + one pre-existing missing `for_test` default
  (`gfx12_mq4v2_fp8_v2`, stale at HEAD) needed to compile.
- `crates/hipfire-config/src/lib.rs` (+9): `kernel.gfx12_fa_packet`.
- `crates/hipfire-dispatch/src/families/attention.rs` (+29): packet branch.
- `crates/hipfire-runtime/examples/tmp_fa_packet_oracle.rs` (+197, new).

## Gate status
- Slice T (transpose): GREEN — 72/72 byte-exact, fill −27.5%, 0 spills.
  Receipt: `scratch-2026-09-17/fapkt/T_receipt.md` (+ Main's exception).
- G1 objects: GREEN — K0/K3/K8/K8fp8/fp8 code-identical modulo `__hip_cuid`.
- Resources: direct/partial 237 VGPR (<240 hard, >224 target), 6-wave class,
  0 spills. Merge 17 VGPR.
- Integrated census: ~2894/wave static estimate (0.5% over 2880, −36% vs old
  4543). K1's one lowering correction NOT spent.
- Slice E oracle: **RED**. Direct b=1 c=15: 132/6144 diffs, first idx 6 =
  (query0,head0,dim6) with OLD=NaN vs NEW=0.0. Split diffs at s=1,8 +
  `packet_partial` memory fault (groups [1,4,8], LDS 49408, batch-1-like
  case) → process panic. Full log: see Main's copy of bg_31 output.

## Defect analysis (not isolated — prime leads, no fix applied)
Signature reads as NEW fully-masked/zero vs OLD valued+NaN on (query0,head0).
Static audit finds all mask inputs identical across arms (qok, sub_full,
pos=14, key_g, dos, gmax/gmin=14, Q/K/V/scale bytes via T-proven maps,
permlane==xor16 per the plan's own probe). Unresolved contradiction, so:
1. **NaN e4m3 codes in oracle KV**: the oracle uses full-range random bytes
   incl 0x7f/0xff. At valid keys, old provably yields NaN via
   e=NaN→l=NaN→inv=0, Ofr=NaN, NaN·0=NaN. If new masks anywhere old doesn't,
   new=0.0. FIRST EXPERIMENT: rerun with NaN codes masked out; if diffs
   vanish, the issue is NaN-propagation parity (exactness then REQUIRES
   reproducing old's NaN·0=NaN, not "fixing" new to 0.0 — needs Main ruling).
2. Resident-Q or paired-load misindex visible only on real data (T proved
   fill bytes, not consumer addressing under the QK/PV loops).
3. Partial fault: all global accesses analyze guarded; grid/LDS correct.
   Needs device debug on the exact faulting case (streaming harness will
   pinpoint it; old harness printed post-hoc only).
Note: Main's ruling ("packet defective; fix finite-guard l, mask before
max, OOR rows write 0.0") is recorded but static audit shows completion
and masking already mirror the old body line-for-line — apply any guard
change ONLY if device data supports it, else you diverge from the oracle.

## Your orders (from Main, relayed)
- Rewrite oracle device-resident (upload largest Q/K/V/pos once, cases as
  launch-param views, device compare kernel → per-case diff count, one
  stderr line/case, ≤5 min wall). Never rerun the upload/download loop.
- Oracle timings rejected (confounded); bitwise verdict only. All timing
  from profiler CSV + interleaved bench. Put that verbatim in E receipt.
- Then E → K1 → pins → bench → decode → serve. Layer-35 compare, flag-free
  pins must be bit-identical if exact, else abandon with numbers.
