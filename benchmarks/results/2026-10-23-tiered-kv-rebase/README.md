# PR #939: beta rebase, separate from historical GPU evidence

Rebased implementation source: `7592074d5203fdaddceeae90cde3a0ca22a20127`.
Comparison base: `beta@8949a6bea32f84d30ce682674fbbe5c1cf3dda0b`.
No GPU, model inference, long-context, serving, Redline or speed run was
performed for this rebase. All test processes hide HIP/ROCR devices.

## What changed in the conflict resolution

Upstream extracted the former Qwen-local attention/FFN bodies into shared
operators. The lab decode and prefill hooks now use those operators; the
deleted production bodies and tests were not restored. Ordinary entry points
forward `None` for the position override and retain the upstream parameters,
including batched `tap`, fusion and commit stride.

Only the lab route separates absolute RoPE from packed KV write/attention
positions. It still refuses graph/recording, VMM/adaptive and unsupported
model configurations. The old combined-position PM qprep bypass remains
ineligible with an override. Read-only admission additionally rejects the
new IU4 verify scope and checks the C64-padded raw-Q workspace before uploads.
The oracle initializes the newly required `LoadCtx::batch_request` to its
default (off); no continuous-batch route is enabled.

## New-base CPU checks

| Check | Result |
|---|---|
| Workspace examples, default features | PASS |
| Both Qwen lab examples, `--features lab` | PASS after the required `LoadCtx` adaptation |
| Positioned decode/prefill tests | 8 passed |
| Qwen carrier tests | 13 passed |
| Tiered storage/streaming planner | 12 passed |
| Streaming example/verifier tests | 3 passed |
| Both comparison scripts and sampled guard | 73 passed |
| Shared dispatch library | 611 passed, 1 existing ignored |
| Kernel registry | 56 passed |
| Recording certificate library | FAIL: 5 passed, 1 failed; clean-beta control reproduces all six missing inventory sites below |
| Full workspace libraries (`--locked --no-fail-fast`) | FAIL, exit 101: all 58 library targets complete; 8,878 passed, 1 failed, 95 existing ignored. Only `railgun-cert --lib` fails with the same six inventory diagnostics. |
| `scripts/pre-commit-ci.sh origin/beta` | PASS; existing warnings are not a strict `-D warnings` pass |

No official full `scripts/no-gpu-ci.sh` rerun is claimed here. Its historical
run is retained at the old checkpoint, not relabelled as a new-base run.

### Recording inventory failure on clean beta

Both the rebased branch and a fresh `git archive` of the comparison base
fail `every_recording_predicate_in_the_tree_has_an_inventory_row` for:

- `rdna-compute/src/gemm.rs`: `hc_down_fused_selected`,
  `qwen4_hc_down_fused_applies`, `qwen4_moe_front_fuse_applies`;
- `rdna-compute/src/hc_row_fold.rs`: `hc_row_fold_gfx1201_applies`;
- `rdna-compute/src/tensor_ops.rs`: `hc_up_v2_selected`,
  `hyper_read_up_wmma_launch`.

These source files are unchanged by this PR. This is a failing check, not a
waiver or a classification of those upstream paths; no scanner exclusion,
predicate classification, threshold or assertion was weakened. The earlier
base's two registry-export failures do not describe this new base: its
current 56 registry tests pass.

Clean-control file blobs were independently checked against the pinned beta:
`recording.rs` = `b91dfc1580ba9ed750db7fec32d101660c6a9bdd`,
`gemm.rs` = `b20de72853facd024a35c6fc11e4ce603cf7342d`,
`hc_row_fold.rs` = `ba1e6dfd2675bc4dea88cd43d26c51c11400af0f`, and
`tensor_ops.rs` = `b60ca71e6f3b5777b37b166e95a651dad819a14d`.

## Evidence identity and failed attempts

The driver started at rebased tip `8bfad97b0` and first found a missing
`batch_request` in the lab example. The one-line adaptation is committed as
`7592074d5`; the successful final lab-example check and all subsequent tests
use that same final source. The first failed build is retained separately
from the successful rerun. The raw driver status retains that initial
failure and the inventory failure; it is not edited to report an all-green
wrapper.

The rebase checkpoint retains the command driver, raw outputs, clean-beta
control and the source delta (`crates/` and `scripts/`) against the comparison
base. SHA-256:

- `cpu-checks-raw.tar.gz`:
  `c65b1ac78ecb7011299bc331821de4ce34aab5ecdc43b55609e88a34ac95cebb`.
- `source.patch.gz`:
  `43e30d2a36a61988a561a54dffec70173acd40f868262fc0930e05ff31854ec4`.

## Historical GPU results are still historical

All four previous tiered-KV evidence directories are byte-identical to
`b942a258af5cb4940dd798a7d5499af25a0d97c2` before this rebase. In particular,
the 6,175 / 32,383 / 129,151 input-token streaming matrix, 1,024-token decode,
sampled VRAM and normal serving/Redline checks were measured on source
`638a350fb619477a5dda563e04942af2ddd4cb23`, based on
`3a5ee848c161b87f28d2437c3cb977de736b4a9c`. Older storage/model and ABBA
results retain their own original pins too.

Conflict resolution and these CPU checks are **not** a fresh GPU/state,
quality, long-context or performance qualification of the rebased head.
Lab-only opt-in scope is unchanged; this does not complete #925.
