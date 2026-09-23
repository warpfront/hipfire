# Mq4e8Combo evidence record — combined-constraint KLD on the existing iu4 kernel

Worktree /home/kaden/ClaudeCode/warpfront/wt-mq4e8 @ 6344f3a94 (branch mq4e8-study).
GPU ordinal 0, HOME=/home/kaden/.hipfire-homes/ab0.

- scripts/check_fixture.sh --sha: OK (14987185152 B, sha256 verified)
- eval binary target/release/examples/eval_hipfire md5 a92a527fa9c1f1509aedba26a330cbbf
  (rebuild from tree; identical to slice-b activation-arm evaluator)
- hipfire CLI target/release/hipfire md5 63d6e7a544514dc18c04c7df25ad3d07
- daemon target/release/daemon md5 53fe3b08c4af3fff2b0f4cd5f11ba76a
- models: fixture 1a48b45ebf5f3ef852efcf8ee5d8084b,
  C1 3676676ea4e6f135926675a93e267075, C2 6582bc9b0d8cb4915c877711bc2a1d02,
  C4 e5a522a0a27a779c3db15949010e0178
- refs: WT2 8a21364051d844b97c122e2c895f56d8, AG b63d3bc13e3ea294e6c938050a07d36f

Per Main steers: arm (a) fixture+XMASTER and the pin rerun dropped from scope
(slice-b/xm_c24.bin 0.070464 retained); arms (f) C1-fp8 + (g) C1-iu4 added.
The base-iu4 pin rerun finished before the steer (0.063410 exact) and is kept
as a bonus row. Partial arm-(a) rerun (killed at 17/24, no KLD/bin) deleted.

All KLD rows: fresh processes, PROVENANCE header per log, EXIT_CODE 0.
See run_sweep_fg.sh / run_sweep_rest.sh / run_short_bench.sh / run_bench_only.sh.
