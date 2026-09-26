# Qwen4 shared token-batched prefill — implementation record

Date: 2026-09-18

This is the mutable implementation/progress record for
[`qwen4-shared-token-batched-prefill.md`](qwen4-shared-token-batched-prefill.md).
The plan remains immutable. This record is updated with concrete ownership,
changed files, evidence, and blockers as work lands.

## Contract accepted before implementation

CodeGraph was attempted first (`which codegraph` / `command -v codegraph`) and
is unavailable on this checkout. The repository rules, kernel-tuning workflow,
`docs/VALIDATION.md`, `docs/ARCHITECTURE.md`, and
`docs/design/sealed-granular-moe.md` were read before source work.

The production contract is:

1. **Family binding only in `hipfire-arch-qwen4`.** Qwen4 owns typed layer and
   weight declarations, resource binding, fixed-capacity batch scratch, state
   handles/marks, PLE epoch/lease ownership, and requested output-row policy.
   It does not own a second prefill interpreter or a token×layer sequencing
   loop after migration.
2. **Common lowering/execution in `hipfire-dispatch`.** The existing
   `pipeline::Step` / `execute_steps` chokepoint remains the only ordered
   executor. It already owns GEMV/GEMM/attention lowering and sealed MoE
   granular stages. Qwen4's new layer program will use semantic, row-batched
   steps (not one public operation per kernel launch); the dispatch side owns
   launch selection, shape checks, alias/lifetime checks, and stateful launch
   order. No callback hides an architecture-owned launch interpreter.
3. **MoE reuses the existing sealed call.** Qwen4 binds
   `MoePrefillParams` and enters `seal_prefill`/`seal_prefill_ep`, then the
   existing granular `Step::Moe*` lowering. The only new admission is the
   exact Qwen4 geometry: `n_exp=512`, `k_top=10`, `hidden=2560`,
   `mi=640`, routed gate/up `MQ4G256V2` (QT44), routed down `MQ4G128V2`
   (QT53), with QT53 allowed only in the already-audited Qwen4 placements.
   Route stamps, source/device/lease identity, complete-call preflight,
   fences, dirty-scratch initialization, canonical slot order, and weighted
   combine remain mandatory. Other families keep their existing semantics.
4. **Execution shape is genuinely layer-major.** A bounded chunk is embedded
   into reusable `[B,*]` buffers. For every layer, all independent projections
   use shared batched GEMM/GEMV machinery. GDN recurrence/convolution, QSA
   append/selection/attention, and PLE depthwise/history updates execute in
   causal row order inside that layer; no future row is visible. This is an
   ordered stateful kernel region surrounded by real matrix work, not a scalar
   loop wrapped in `Step`s.
5. **Outputs and MTP stay explicit.** Ordinary AR prefill computes only the
   requested final logits row unless a caller requests all rows or an
   intermediate/wide capture. Native MTP remains a separate token-shaped
   state owner, but reuses common projection/activation/attention operations
   where applicable and does not silently claim batched MTP.
6. **Bounded resources.** Scratch is allocated once at attach time for the
   configured chunk cap; QSA/GDN/PLE state arenas are not widened to context
   or duplicated per token. PLE uses its existing bounded reader, lease, and
   epoch cleanup protocol.

## Concrete file ownership

### Existing shared owners to extend

- `crates/hipfire-dispatch/src/pipeline/steps.rs`: add only the semantic
  row-batched steps required by Qwen4 plus their complete-call preflight and
  launch arms; keep `Step::Moe*` as the sealed MoE authority.
- `crates/hipfire-dispatch/src/pipeline/mod.rs`: shared Qwen4 lowering helpers
  and exact QT44/QT53 grouped down selection; generic MoE rejection remains
  fail-closed outside the exact Qwen4 predicate.
- `crates/hipfire-dispatch/src/families/gemm.rs` and existing rotation/
  attention families: reuse, do not duplicate, batch projection dispatch.
- `crates/hipfire-dispatch/src/families/moe.rs`,
  `pipeline/moe_program.rs`, and `pipeline/sealed_moe.rs`: preserve and, where
  necessary, complete the exact Qwen4 top-10/QT44/QT53 shared route.
- `crates/rdna-compute/src/qwen4.rs` and `kernels/src/qwen4_ops.hip`: add
  row-batched state/pointwise wrappers only where existing scalar Qwen4
  operations cannot express the required bounded layer program. Quantized
  projection kernels remain in their existing shared GEMM/MoE owners.

### Qwen4 family binders and state owners

- `crates/hipfire-arch-qwen4/src/gpu_forward.rs`: reduce to typed resource
  binding plus invocation of the shared layer program; retain public
  `forward_token`/`forward_chunk` and scalar compatibility through the same
  program, not a second interpreter.
- `crates/hipfire-arch-qwen4/src/bundle.rs`: preserve the attach/unload owner
  and expose the bounded chunk/output-row contract without per-call device
  allocation.
- `crates/hipfire-arch-qwen4/src/state.rs`: retain GPU state allocation,
  snapshot/restore, and marks; expose only typed state views/counters needed by
  the shared executor.
- `crates/hipfire-arch-qwen4/src/ple_rows.rs`: retain bounded source reader,
  leases, epochs, and cleanup; the shared program consumes a borrowed lease at
  the configured layer boundary.
- `crates/hipfire-arch-qwen4/src/projection.rs`: retain Qwen4 QT44/QT53
  logical/encoded view geometry and route matching; use shared batch dispatch
  for projections.
- `crates/hipfire-arch-qwen4/src/mtp_gpu.rs` and `mtp_spec.rs`: preserve
  native-MTP ownership and refusal/lifecycle semantics; migrate only reusable
  numerical operations, not MTP sequencing.

### Tests/evidence record owners

- Existing Qwen4 CPU equation/shape tests and dispatch tests remain the
  narrow no-GPU contract tests. New tests defend chunk tiling, requested-row
  output, causal state transitions, exact Qwen4 admission/refusal, and
  preflight-before-effect behavior—not enum/source wiring.
- GPU/model validation belongs to the delegated baseline/validation worker and
  the main session. Do not overwrite baseline binaries or caches before the
  baseline-saved signal.

## Pre-existing dirty tree (preserved, not authored by this implementation)

At record creation, `git status --short` reported:

- `crates/hipfire-arch-qwen4/examples/qwen4_parity.rs`
- `crates/hipfire-arch-qwen4/reference_oracle/equations.py`
- `crates/hipfire-arch-qwen4/reference_oracle/upstream.py`
- `crates/hipfire-arch-qwen4/src/gpu_forward.rs`
- `crates/hipfire-arch-qwen4/src/mtp_gpu.rs`
- `crates/hipfire-dispatch/src/pipeline/sealed_moe.rs`
- `crates/rdna-compute/src/moe.rs`
- `crates/rdna-compute/src/qwen4.rs`
- `kernels/src/moe_router_softmax_top10_f32.hip`
- `kernels/src/qwen4_ops.hip`
- `.slim/`
- this design plan

The implementation must layer on these changes and never reset, checkout, or
otherwise destroy them. The two Qwen4 source/kernel files above already carry
uncommitted correctness fixes; edits will be narrow and preserve their intent.

## Progress

- Contract reported to `Main`; implementation authorization received.
- Required design/rules/validation reads complete.
- No source edits or builds performed yet in this session.
- Baseline binary overwrite guard remains active: wait for the baseline-saved
  signal before any build that can replace retained outputs.

## Evidence and blockers

- CodeGraph: unavailable on host; repository exploration used narrow `read`,
  `grep`, and `glob` calls instead.
- Existing oracle failures reported by the user (GDN and routed index 16) are
  named blockers/ground truth. They must not be rerun merely for confirmation
  or hidden by tolerance changes.
- Physical EP2/EP4 and any unavailable Qwen4/G5 admission hardware remain
  validation blockers, not implementation fallbacks.
- No numerical or performance claim is made by this record until the matched
  baseline/candidate routes are run and recorded by the main validation flow.

## Changed files (implementation-owned)
## Source checkpoint — 2026-09-18

The exact Qwen4 prefill route now has a cohesive shared lowering module
(`crates/hipfire-dispatch/src/pipeline/qwen4_prefill.rs`) and the sealed
selection/route producer reaches it only for the exact replicated
512-expert/top-10/QT44/QT53 geometry.  Qwen4 EP remains rejected before GPU
work.  QT53 shared-down uses a dense row-batched projection rather than a
per-row scalar loop; shared G128 rotation is performed once before that
projection.  Grouped QT53 down rows are BF16-rounded before weighted combine,
matching the scalar route's per-expert output boundary.  The Qwen4 shared
residual fold uses a row-batched source-exact BF16 product/add operation.

Central dispatch ownership is now represented by
`crates/hipfire-dispatch/src/pipeline/qwen4_program.rs`, registered from
`pipeline/mod.rs`.  Its typed descriptors (`Qwen4LayerDescription`,
`Qwen4ProgramDims`, state views, scratch views, and `Qwen4MoeBinding`) and
`execute_layer` batch BF16 HC/GDN/QSA projections while keeping recurrent/QSA
state transitions ordered and routing batched MoE through `seal_prefill` and
`Step::Moe`.  Family binding/refactor into this executor remains the next
integration boundary; the module is not yet an end-to-end product path.

Scoped evidence:

- `cargo check -p hipfire-dispatch --features deltanet` passed after the
  centralized module and QT53 batched additions.
- Earlier `cargo check -p hipfire-arch-qwen4` passed before the central module
  was introduced; no Qwen4 family wiring claim is made from that result.

Changed files authored by this implementation (append-only):

- `crates/hipfire-dispatch/src/pipeline/qwen4_prefill.rs`
- `crates/hipfire-dispatch/src/pipeline/qwen4_program.rs`
- `crates/hipfire-dispatch/src/pipeline/mod.rs`
- `crates/hipfire-dispatch/src/pipeline/moe_program.rs`
- `crates/hipfire-dispatch/src/pipeline/sealed_moe.rs`
- `crates/hipfire-dispatch/src/families/gemv.rs`
- `crates/rdna-compute/src/gemm.rs`
- `crates/rdna-compute/src/kernels.rs`
- `crates/rdna-compute/src/qwen4.rs`
- `kernels/src/gemm_mq4g128v2_batched.hip`
- `kernels/src/qwen4_ops.hip`

None yet. This list is append-only for files changed by this implementation;
pre-existing dirty files above remain separately identified.

## Source checkpoint — post-proof cleanup

The immutable plan remains unchanged. The shared execution implementation now
includes the Qwen4 typed layer program, bounded chunk/final-row output policy,
complete-call MoE preflight, QT44/QT53 grouped routing, and exact BF16
multi-row dense projection. Temporary stage dumps, input replay, and the
unused `Qwen4LayerScratch::moe_input` view were removed after validation.

Validation evidence retained outside the source tree:

- QT53 dense probe: `.codeinsight+research/qwen4/shared-token-prefill/fixed-cut-current/runs/qt53-dense-probe.json` (report SHA256 prefix `f0a84d3422`; source SHA256
  `6fc79077de05b580c99abfba8586d19d8c8449e98a14ed1faf8b4d9013da9d5f`; exact
  real layer-0 expert and synthetic cancellation comparisons for N=2/3/5).
- Immutable fixed BF16 oracle binary SHA256
  `5cb6b4a3269a9006788c88e046e3a6a6188504b7c84b6e62ea34c6665fe8be3a`;
  cap-4/cap-8 final-row and state checks are exact.
- Fresh AR/MTP smoke reports and logs are under
  `.codeinsight+research/qwen4/shared-token-prefill/fixed-cut-current/runs/`;
  preserved fixed binaries are separate from the later source cleanup.
- `cargo build --release --workspace --all-targets --locked` passed on the
  post-cleanup source; log:
  `.codeinsight+research/qwen4/shared-token-prefill/final-verification/build-release-workspace-all-targets-locked-final.log`.

The current acceptance blocker is a fresh warm AR context-291 transcript
difference between the baseline and candidate despite identical request bytes
and settings. Validation is isolating exact tokenizer-produced token IDs at
the first divergent token across scalar, cap-128, and cap-4 executions. The
small retained oracle and QT53 channel probes do not substitute for this
long-state boundary evidence; no final product-equivalence or performance
claim is made until that isolation completes.

## Verification checkpoint — host gates

The post-cleanup source passed `cargo build --release --workspace
--all-targets --locked` and the explicit touched-Rust `rustfmt --check`;
the logs are retained under `final-verification/`.  The full workspace
library-test run recorded in
`test-lib-workspace-locked.log` ended with passing test results before the
later mechanical launcher `Vec`-to-fixed-array edits.  Its post-array rerun
was intentionally canceled to release the GPU lock for the context-291
isolation oracle; the post-array release all-target build is the compile
proof for those edits.

The no-GPU command was run as:

`LD_LIBRARY_PATH=/nix/store/1xw5xccqqh1xw3mvd70hyil6x418wxcm-gcc-14.3.0-lib/lib scripts/no-gpu-ci.sh`

The NixOS `libstdc++` path allowed the Rust check and no-GPU unit-test
sections to complete.  Python collected 387 tests: 381 passed and 6
failed.  The evidence is exact: one failure is the host's absent
`/bin/bash` in the KernelAtlas shell-chain test; five are
`tests/test_mq4c_repack.py` failures because its imported `mq4c_repack`
module lacks `HfqmError`, `main`, and `parse_hfqm_index`.  This cutover did
not edit that test, its provider module, or their API.  Because the script
stops at that Python section, the remaining phases were run individually
with the same `LD_LIBRARY_PATH`: Redline's 179-test log records the
MQ4R registry-card hash mismatch and mocked cargo-argv failures in
`tools/redline/tests/test_golden.py`; install revision has one host
failure from a temporary fake-cargo `#!/bin/bash` shebang; uninstall
passes, and the environment/docs check passes.  This cutover did not edit
the registry card, `tools/redline/tests/test_golden.py`, or the installer
tests/scripts.  Exact logs are `no-gpu-ci-libstdcxx.log`,
`no-gpu-redline-unittest.log`, `no-gpu-install-revision.log`,
`no-gpu-uninstall.log`, and `no-gpu-env-docs.log`.

The reachable layering/registry checks were also run independently:
quant registry and arch-dispatch checks pass.  The frozen layering check
reports `hipfire-arch-qwen4` absent from `scripts/layering.txt`; the
dispatch-bypass check reports the same crate's three calls absent from
`docs/governance/debt-dispatch-bypass.txt`.  The user-provided branch
scope identifies that Qwen4 governance entry as a prerequisite from before
this task; it is separate from the API cutover, and neither governance
file was edited.  A narrow `git diff --name-only` over those files and the
failing test/provider/install paths is empty; the captured evidence is
`untouched-gate-scope.log`.  The complete `leanup-ratchets.sh` result is
retained in `leanup-ratchets.log`: its seven threshold failures are exactly
`daemon_arch_id`, `daemon_arch_refs`, `daemon_lines`, `ungated_examples`,
`layer_unlisted_crates`, `bypass_unlisted`, and `bypass_total`.  No ratchet
budget inflation or governance edit is made here.  The Qwen4 exported-step
seams themselves pass the focused `qwen4_program` (2 tests) and
`sealed_moe` (34 tests) suites.

Qwen4 product PM4 admission remains out of scope, and physical EP2/EP4
validation is unavailable.  The canonical qwen3.8-27b.mq4-xt dense-trunk
retained regression harness is locally runnable on gfx1151; because the
common Step interpreter changed, that regression proof remains owed to the
validation worker after context-291 isolation.  The warm context-291 scalar
versus batched diagnostic is still active; this record is provisional and
makes no completion, product-equivalence, performance, or marketing claim.

## Verification checkpoint — shared retained host regression (2026-09-18)

The post-cleanup host verification is now recorded without changing the
immutable plan or any prior checkpoint.  The canonical dense-trunk fixture was
the already-verified
`~/.hipfire/models/qwen3.8-27b.mq4-xt` (SHA256
`9f91556f7e0431a077d03756a7102d0154108757289e6e5fe9a2d204c0c9eeb7`), loaded
as `qwen3_5` on `gfx1151`.  The harness used a private `HOME` and kernel cache,
`HIP_VISIBLE_DEVICES=0`, manual capture, and `HIPFIRE_REPLAY_BACKEND=shadow`.
No Qwen4 retained admission or product claim is made.

Reachable scoped retained evidence:

- The exact AQL route was run with
  `scripts/redline_daemon_harness.py --skip-prefill --decode-context 32
  --shadow-iterations 15`.  Capture was stable at 1,139 launches, 17 unique
  kernels, sequence hash `7a1072b2b002bce1`; the AQL contract probe reported
  17 kernels.  Fifteen-position HIP/AQL/blob shadow parity passed:
  `bit_exact`, `blob_bit_exact`, logits, KV, recurrent state, GDN state, and
  `gdn_frame_exact` were all true; dispatches `1139`, packets `1140`, and
  `queue_id=2`.  The report is
  `.codeinsight+research/qwen4/shared-token-prefill/final-retained/dense-trunk-aql-context32-final.json`
  (SHA256
  `40f0a1269ad6dce1e18e3643c44b0e7311f039aa409e1de2b4b7553aa6065715`);
  raw daemon and stdout logs are retained beside it.  This is scoped AQL
  discovery/correctness evidence, not a full default-route acceptance or
  timed-arm Redline certification.
- The same route with `--pm4` reached stable capture and the same 17-kernel
  AQL contract probe, then failed closed during PM4 preparation:
  `gemv_mq4g256v2_residual: GFX10/GFX11 PM4 dispatch does not yet support
  scratch (private=20, dynamic_callstack=false)`.  The raw PM4 daemon/stdout
  logs are
  `.codeinsight+research/qwen4/shared-token-prefill/final-retained/dense-trunk-pm4-context32-final.log`
  and `.stdout.log`; no PM4 pass or promotion claim is made.
- The canonical default `--decode-context 128` attempt failed before capture
  during Qwen prefill prime with
  `valid_lane_mask: max_batch must be 1..64`.  Context 32 was used only to
  reach the bounded diagnostic path; it is not full default-route coverage.
  Source provenance is retained in
  `.codeinsight+research/qwen4/shared-token-prefill/final-retained/lane-mask-provenance.txt`:
  `valid_lane_mask` is unchanged from commit `8cf5eb0389` (2026-08-15), the
  sequential `valid_lane_mask(n)` call is from `27abd53e46` (2026-09-16), and
  the current working diff only changes `execute_steps` arguments from
  `&[Step]` to `&mut [Step]`.  The failure therefore predates this Step
  mutability cutover; no unrelated source edit was made.

Final host test proof:

- `flock -w3600 /tmp/hipfire-gpu.lock` plus
  `flock -w3600 /tmp/hipfire-build.lock`, with `RUST_TEST_THREADS=1`,
  `cargo test --lib --workspace --locked` completed with exit code 0.  The
  raw log is
  `.codeinsight+research/qwen4/shared-token-prefill/final-verification/test-lib-workspace-locked-post-harness.log`.
  Warnings were existing dead-code/unused-variable warnings; no test failed.
- The current daemon binary used by the retained harness has SHA256
  `e0112c3b95f5a44528be1dc164a60874251dc3f09485bc4616da18830c8341ec`.

The warm context-291 scalar-versus-batched diagnostic remains active, the
scalar mismatch remains an acceptance blocker, and this checkpoint makes no
completion, product-equivalence, performance, PM4-admission, or marketing
claim.

## Verification checkpoint — source-matched scalar parity (2026-09-18)

The preserved old-source checkout was reconstructed from commit
`96553691be5eac650039adc7b3b376ff58606070` plus the tracked patch
`f54e646e9b39b738e1c80f3b6078b89b9df41e49606056b7ff33727b96d290a`.
Its rebuilt daemon is byte-identical to the saved baseline daemon
(`sha256=a2134f43cc7b7f34ac7fef19f60b096b84f70cece71c52d398a6a4e08babb025`,
`md5=a3fc2bd319cb0871c1f477460473e456`).  The model identity is
`sha256=7dcbceb4f501a66abef81cc624b86a4da250850ba79acf8dcc6a5204a5c8303f`,
`md5=fda74d3760dc803e778e9b30a2fe0ebd`.

The exact 291-token manifest is shared by old and candidate
(`payload sha256=e0dd3b389ed16741371b44229b355896105d83d657703fd6242aa6b75a13dbee`);
the source-matched serve request remains
`prompt_md5=973900074bfd15d4adeeecdff3359082`,
`request_md5=c634a65ab0a325960028c431c3cd7534`.  A fresh old-source scalar
cap-1 oracle and the current candidate cap-1 oracle are bit-identical for
all 291 full-vocabulary logits (`sha256=87394feac971f9a20b88d6d59511663c4f9faef05233a2401063923f81bc8f1e`)
and for captured states at positions 128, 256, and 291
(`13f452e641f322142aa8d52142caf1e3fade720be517720d8654ee18a1e56a0d`,
`b8cd2c9c486a98d636093c37876fc321f3555ad7c4b708a2cb6e76970ac5bbbd`,
`362aef3fcd5fb5abf499c668ccfc489bc6a39f3cd22ea66be5efaf37989ee382`).
There is no differing row or compared boundary tensor.  The compact
provenance/equality record is
`.codeinsight+research/qwen4/shared-token-prefill/baseline-source-matched/scalar-parity-summary.json`.

This establishes scalar/cap-1 parity, not product continuation equivalence:
the warm product continuation/transcript mismatch is not yet resolved.
`qwen4-completion-check` owns the remaining product caller/current-binary
investigation; no product-equivalence or performance claim is made.

## Verification checkpoint — post-HC host validation (2026-09-18)

The HC-specific source fix restored the missing `hc_read_batch(gpu, dims,
&layer.attn_hyper.read, &streams, scratch, rows)?` immediately before the
attention lowering in `crates/hipfire-dispatch/src/pipeline/qwen4_program.rs`.
The semantic source identity used for the numerical/product build was
`a553fe861610e79776c9b9856fcd92329e5200a887cb798c99d76a8a178847b8`.  After
the requested formatter-only collapse of that call to one rustfmt line, the
final `qwen4_program.rs` SHA256 is
`f1f37bfd5f19aea3301f5de7ff455e710097b66197e6436b67f3cf4cf13377b3`.
No semantic source change was made by the formatter correction; the product
binaries below were built from the semantic source and retain their identity.

Host gates, with raw logs retained under
`.codeinsight+research/qwen4/shared-token-prefill/final-verification/`:

- `flock -w 3600 /tmp/hipfire-build.lock cargo build --release --workspace
  --all-targets --locked` passed without taking the GPU lock.  The exact
  product paths are `/home/bjoern/hipfire/target/release/daemon` (MD5
  `cda02f6fc9688e215ec8ad95e4f3c9dc`, SHA256
  `bb346923ed1d974282d2b0068a6df16bcac6230499800f89ea9ed931671de53f`) and
  `/home/bjoern/hipfire/target/release/hipfire` (MD5
  `12553c586915590ccc2f355c4d513446`, SHA256
  `4e9211dc6f28fb6a671e045034a645c2619f493d2521223ed76cc946ebb633de`).
  Log: `post-hc-build-release-workspace-all-targets-locked.log`.
- With both `/tmp/hipfire-build.lock` and `/tmp/hipfire-gpu.lock`, `RUST_TEST_THREADS=1
  cargo test --lib --workspace --locked` passed with exit code 0; every
  workspace library suite in the log passed.  Log:
  `post-hc-test-lib-workspace-locked.log`.
- With `/tmp/hipfire-build.lock` and no GPU work, `cargo clippy --workspace
  --all-targets --locked` passed with exit code 0.  The log records the
  repository's existing warnings; no warning suppression or `-D warnings`
  change was used.  Log: `post-hc-clippy-workspace-all-targets-locked.log`.
- Scoped `rustfmt --edition 2021 --check --config skip_children=true` now
  passes all 33 touched Rust paths.  The 13 post-baseline introduced files
  were formatted only in their owned hunks, and the one additional
  `qwen4_bf16_scaled_add_batched` call hunk inside initially dirty
  `crates/rdna-compute/src/qwen4.rs` was formatted narrowly.  Initial dirty
  baseline hunks were otherwise preserved.  Classification and final proof:
  `post-hc-fmt-classification.log`, `post-hc-fmt-owned-rust.log`,
  `post-hc-fmt-classification-final.log`, `post-hc-fmt-final.log`.
- The existing gfx1151 cache identity is recorded in
  `post-hc-kernel-cache-identity.log`: 342 `.hash` files, sorted content
  manifest SHA256
  `a03ddd17a2b356569a8a9caac58c1ed18dd1e1eeed2b3a45ed7fceefab58e863`.

Post-fix HC numerical evidence on gfx1151 is now exact for the scoped oracle
routes:

- The fresh cap-1 two-token report
  `fixed-cut-current/runs/post-fix-cap1-two.json` (SHA256
  `a1ca46140142ef73d0dc80e3325ae7d4c7407f209d7fbad9556ad7c60e3b3438`)
  matches the old scalar rows exactly: row hashes
  `f4acc642299657321271b9db58aea0fb04f7f2d23f00fd88e3352a5de55769dc` and
  `23a42e3aec9723bfd740eb62ae9b07968c8f5a155a27a58b1882cd77e2584e90`.
- The cap-1 versus natural cap-128 first-129 comparison passed at boundaries
  `[128, 129]`; the shared raw comparison SHA256 is
  `94703015d94d910050bd6d166b4a05b503ca8700dfb1d209423e08837d2d28c0`.
  Reports are `fixed-cut-current/runs/post-fix-cap1-first129.json` (SHA256
  `02caf97a238b670a5750d4378e47a0be0a0919914c350d732db0bc8c71f3d62e`) and
  `fixed-cut-current/runs/post-fix-cap128-first129-natural.json` (SHA256
  `0f43a3c5f5195fcae7e99154bb28417b0219a17a6308317a3727af88b99d44f1`).
- The natural final-only slices `[0,128]`, `[128,256]`, `[256,291]` passed
  in `fixed-cut-current/runs/post-fix-cap128-final-api.json` (SHA256
  `30e7bbd3ca037bbc7139ec00777ff6ca36c2c678312b1611b4f2c66c416ba530`):
  all 291 rows were finite with `max_abs=0.0`, and the final state comparison
  was exact.
- Continuation/reset passed for 17 rows with full/partition/token logits and
  final state `max_abs=0.0`; report
  `fixed-cut-current/runs/post-fix-continuation-reset.json` (SHA256
  `10a9ba002840f899e673b6255cd896394f59b8e0cc84945f20f5693ff2212bc8`).
- The repeated dirty-combine probe passed for three tokens × hidden 2560:
  repeated expert IDs and nonzero residuals were exercised, grouped/indexed
  one-pass and two-pass outputs were CPU bit-exact, and every compared
  `max_abs` was `0.0`.  Report:
  `fixed-cut-current/runs/post-fix-dirty-combine.json` (SHA256
  `75fabbb03397a1366d29bb2e066030d6521780896cc197613a4b338030ca881f`).

This checkpoint is HC-specific oracle and host-gate evidence.  It does not
rerun dense retained replay, does not change the previously documented AQL
context-32 / PM4 scratch and default-128 lane-limit evidence, and makes no
Qwen4 retained admission, PM4 promotion, or performance claim.  The
performance worker now owns the clean product-CLI continuation and profile
routes using the exact binary identities above.

## Verification checkpoint — current implementation and compiler-enabled product smoke (2026-09-18)

The implementation checkpoint is now source-matched to the current product
build.  The Qwen4 family crate remains a typed binding/descriptor layer; the
shared lowering and execution choke point is `Step`/`execute_steps` in
`hipfire-dispatch`.  Qwen4 execution is layer-major over bounded slices:
shared matrix work uses the bounded slice while stateful GDN/QSA/PLE transitions
retain row order.  The sealed MoE route retains the exact expert geometry
(`n_exp=512`, `top_k=10`, `hidden=2560`, `mi=640`, QT44 gate/up, QT53 down)
and its route/lease/fence/dirty-initialization/slot-order contract.  The
final-only API returns the final row and final state; `bundle.reset` is the
reset boundary used by the continuation proof.  Native MTP remains a separate
token-shaped route and is not folded into the ordinary AR final-row policy.

The HC correction is the current-stream attention preparation immediately
before GDN/QSA:
`hc_read_batch(gpu, dims, &layer.attn_hyper.read, &streams, scratch, rows)?`.
It prepares the layer attention input from the current streams before GDN/QSA
consumes `moe_input`; this is not a claim that attention follows the same
layer's later MLP/MoE sequence.  The semantic source identity used for the
product build was `a553fe861610e79776c9b9856fcd92329e5200a887cb798c99d76a8a178847b8`;
the formatter-only final source identity is
`f1f37bfd5f19aea3301f5de7ff455e710097b66197e6436b67f3cf4cf13377b3`.

The historical PLE failure is distinct from the HC omission.  In the old
max-128 path, the layer-1 PLE application was guarded by
`ple.lease.is_none()`, so row 0 acquired/applied the PLE lease while later
rows skipped it; the first divergence was row 2 and the state divergence was
`gdn[1].recurrent`.  The corrected bounded reader/lease/epoch cleanup and the
current HC read are both exercised by the post-fix reports; no tolerance
relaxation, cache fallback, special-case input, or production debug hook was
added.

Compiler-enabled product smoke used fresh private caches and the current
source; `HIPFIRE_NO_DEVICE_COMPILER` was not set.  The compiler identity was
HIP 7.2.53211-9999 / clang 22.0.0 under
`/nix/store/lqklrnx2bc9k765jyxc0d8q6h15wlybb-clr-7.2.3`.  The final AR16
result is
`.codeinsight+research/qwen4/shared-token-prefill/fixed-cut-current/final-ar16-compiler/result.json`
(SHA256 `cab493e281e2a089f1a64acfc084c8b15df813b45affcb124eb14bbb51e60940`);
it is `ctx=291`, `gen=16`, `finish=length`, with one terminal frame and the
current-source-matched no-speculation text
`The text you provided appears to be a mix of a **pangram**`.  Its explicit
scalar comparison record is
`.codeinsight+research/qwen4/shared-token-prefill/fixed-cut-current/final-ar16-compiler/scalar-comparison.json`
(SHA256 `077263f8d8de098214b1728a1b871314cd6eaf7d22ece712ceec75fe41d190b6`).
That record now carries a direct scalar-vs-batched greedy continuation
proof; it no longer relies on the teacher-forced prefix/state comparison for
the 16-token product transcript.

The native MTP smoke is
`.codeinsight+research/qwen4/shared-token-prefill/fixed-cut-current/final-mtp-smoke-compiler/result.json`
(SHA256 `d7c4cca76b155a4d1b9f6d6eb0546abb3f1b9cc3d53ff7ca389a2e8ac565c6c3`):
`ctx=42`, `gen=16`, `finish=length`, `mtp=true`, `cycles=6`, `tau=1.5`, and
the expected Paris continuation.  Both product runs use daemon SHA256
`bb346923ed1d974282d2b0068a6df16bcac6230499800f89ea9ed931671de53f`,
CLI SHA256 `4e9211dc6f28fb6a671e045034a645c2619f493d2521223ed76cc946ebb633de`,
and model SHA256
`7dcbceb4f501a66abef81cc624b86a4da250850ba79acf8dcc6a5204a5c8303f`.

The prior source-matched checkpoint's statement that product continuation
equivalence remained unresolved is historical/provisional for this final
source-matched smoke: it must not be replaced with the old product
multirow-PLE-bug transcript.  The exact continuation fixture below is the
authoritative sampled-id/logit/state comparison.

## Verification checkpoint — direct scalar-vs-batched greedy continuation (2026-09-18)

The direct fixture
`.codeinsight+research/qwen4/shared-token-prefill/fixed-cut-current/runs/greedy16-scalar-batched/result.json`
(SHA256 `b5eefc6ad17bdfccbe6cd69ecefe929acb5711d105e17d35ba31e517d8e1b3bc`)
loads one current bundle, resets it, runs scalar `forward_token` prefill over
the exact 291-token manifest, and takes the final-row greedy argmax.  It then
resets the same loaded bundle, runs the natural `[0,128]`, `[128,256]`,
`[256,291]` final-only path, and applies the identical standard 16-token
greedy schedule (prefill final-row argmax plus 15 `forward_token`
transitions).  This is a direct full-16 continuation proof, not an inference
from the earlier 17-row teacher-forced reset test.

The result is `status=pass`: scalar and natural-128 prefill final logits and
state are bit-exact; all 16 emitted IDs, all 16 logits rows, and all 16
captured per-step states are bit-exact.  Both ID sequences are
`[760,1414,488,3766,7701,310,381,264,6311,314,264,2972,79,512,2319,332]`,
which decode to
`The text you provided appears to be a mix of a **pangram**`, exactly the
final AR16 transcript.  The scratch oracle identities are binary
`52c9eef230b082d32736a191659cae74404b334eedb2a781a16efcd40854c507`,
source `1899c7d8f1990edf5bdef839438a83305883db0d017a46af34608ec674f77846`,
`Cargo.toml` `aaf5600d29f73f3e8696bcab346f420aafac8062f608ce0884d6c2d8e71b99b1`,
and `Cargo.lock`
`974133b46fd6bba9571c55202a7f9d6eea828b9f0589b977a7ef3558217dc50d`.
The compiler-enabled private cache contains 88 files with listing SHA256
`dbf4e11d6d6550f95464c5786ccf37a2b074c2b52d35b6d262aba317935dfd64` and
content SHA256
`90b2d76079620bf715c87b10ab04eef4bd03eb35502dfcbafa701ae504d0ec5c`.

Remaining boundaries are unchanged: physical EP2/EP4 validation is
unavailable, Qwen4 retained admission/PM4 promotion is out of scope, dense
retained replay was not rerun, and no performance/marketing claim is made by
this checkpoint.  Clean product benchmark/profile evidence remains owned by
the performance worker; the final scoped measurement follows below as a separate dated checkpoint.

## Verification checkpoint — final scoped product measurements (2026-09-18)

The final dated measurement record is
[`docs/perf-checkpoints/2026-09-18-qwen4-shared-token-batched-prefill-final-measurement.md`](../perf-checkpoints/2026-09-18-qwen4-shared-token-batched-prefill-final-measurement.md).
It is historical, fixture-bound evidence only.  It records six fresh
compiler-enabled product CLI samples on `gfx1151` with q8 contiguous KV,
speculation off, noslots/stateless workload, `max_tokens=16`, ten warmups, and
one measured run per fresh private home:

- 42-token short prompt, three raw samples: prefill median `18.5 tok/s`,
  decode median `8.4 tok/s`, TTFT median `2273.3 ms`.  The CLI explicitly
  labels this prompt's prefill number launch-overhead-only.
- 291-token prompt, three raw samples: prefill median `5.5 tok/s`, decode
  median `1.8 tok/s`, TTFT median `53160.3 ms`.  These are observations under
  unknown external GPU contention, not an A/B or speedup claim.

The separate serve smoke returned no completed visible generation
(`ctx=0`, `gen=0`, `finish=null`) and is excluded from those medians.  The
rocprof route is blocked at startup/compiler probing: the daemon retry,
corrected direct oracle profile, and tiny rocprof smoke produced no CSV or
runtime markers.  The `HIPFIRE_PROFILE=1` fallback completed a normal short
bench but emitted no usable profile markers.  Consequently this checkpoint
claims no grouped benefit, launch count, kernel count, ISA attribution, or
dynamic-memory result; memory scope is **STATIC only**.  The historical
old-PLE output-equivalence caveat and all raw paths/hashes are in the dated
record.

## Six-lever tuning lineage — fixture-bound (2026-09-19)

The gfx1151 campaign covers six staged tuning records, not only the final HC
rows/grid-Y candidate. Each record freezes a distinct source/cache/product
identity; these are scoped engineering evidence, not six product promotions:

| # | staged lever | immutable evidence and disposition |
|---:|---|---|
| 1 | QSA selection-only parallelism | [`selection-candidate/manifest.json`](../../.codeinsight+research/qwen4/perf-targets-20260918/baseline/selection-candidate/manifest.json) — `parallel-qsa-select-only`, frozen before the attention edit; no standalone product claim. |
| 2 | QSA selection-plus-attention parallelism | [`attention-candidate/manifest.json`](../../.codeinsight+research/qwen4/perf-targets-20260918/baseline/attention-candidate/manifest.json) — `parallel-qsa-select-and-attention`, immutable pre-sentinel evidence; no promotion claim. |
| 3 | QSA selection sentinel | [`selection-sentinel/manifest.json`](../../.codeinsight+research/qwen4/perf-targets-20260918/baseline/selection-sentinel/manifest.json) — accepted pre-GDN route with exact oracle/profile comparisons; not a product baseline. |
| 4 | GDN shared exact-128 BF16 QK norm | [`gdn-shared-norm128/manifest.json`](../../.codeinsight+research/qwen4/perf-targets-20260918/baseline/gdn-shared-norm128/manifest.json) — accepted pre-N8 route with exact oracle/profile comparisons; not a product baseline. |
| 5 | gfx1151 N8 measured-prefill multirow allowlist | [`n8-gfx1151/manifest.json`](../../.codeinsight+research/qwen4/perf-targets-20260918/baseline/n8-gfx1151/manifest.json) — accepted kernel optimization and frozen A comparison; no product-win claim. |
| 6 | HC rows/grid-Y batching | [`hc-rows-gridy-20260919-v3/manifest.json`](../../.codeinsight+research/qwen4/perf-targets-20260918/baseline/hc-rows-gridy-20260919-v3/manifest.json) — current final candidate; no product-target claim. |

The first two rows are deliberately separate QSA source/cache stages, while
the selection sentinel is the accepted combined route that gates the GDN
stage. The six records therefore preserve the full tuning lineage without
reinterpreting any earlier historical checkpoint.

## Verification checkpoint — HC rows/grid-Y final candidate (2026-09-19)

The immutable dated record is
[`docs/perf-checkpoints/2026-09-19-qwen4-hc-rows-gridy-final-measurement.md`](../perf-checkpoints/2026-09-19-qwen4-hc-rows-gridy-final-measurement.md).
It supersedes no prior historical record and makes no product target or
promotion claim.

The explicit ROCTX natural-marker HC counts are **112,326** for frozen N8
(56,163 norm + 28,227 read + 27,936 write) and **1,158** for the current HC
candidate (579 + 291 + 288). The candidate kernel trace confirms grid-Y=1 for
scalar and batched continuation, and grid-Y=128/35 for the natural 128/128/35
chunks. The withdrawn heuristic `109024` split is not used.

The focused primitive probe is exact for rows 1, 2, 35, and 128; the fresh
greedy-16 scalar-vs-natural oracle and cross-version N8/original comparisons pass
with zero step mismatches. A separate deterministic Paris correctness fixture
passes ordinary AR and active native MTP independently: both return `Paris`,
finish normally, emit content, see `done`, and have no runaway or stream error;
MTP reports `tau=1.0`, one cycle, and `mtp=true`. This fixture is correctness
evidence only, not a performance observation.

The preflight-gated fresh product ABBAAB used rebuilt candidate binaries:
prefill medians are A `26.0` vs B `28.8` tok/s, decode medians A `8.8` vs B
`8.9` tok/s, and the B/A decode ratio is `1.011364` (not a meaningful decode
win). User thresholds `prefill >500` and `decode >=25` are both unmet and
remain open/blocked; further uncontended profiling is required. The record
does not assign the gap to contention or promise that an idle GPU would solve
it. The stale-binary ABBAAB is preserved as invalid.

Fresh release build, workspace library tests, workspace all-target clippy, and
scoped Rust formatting checks all completed with exit 0; raw host-gate logs
and all GPU evidence are retained at the paths in the dated record. Physical
EP2/EP4, retained replay/PM4 admission, and product promotion remain outside
this checkpoint.

## Verification checkpoint — current typed Qwen4 ownership (2026-09-19)

This dated section records the ownership cutover after a real gfx1151 short
smoke. Earlier checkpoints remain historical and append-only; their source
paths and evidence are not rewritten here.

### Current ownership and path map

- `crates/hipfire-arch-qwen4/src/program.rs` owns the Qwen4 program
  dimensions, fixed-capacity scratch layout, typed layer descriptors, and the
  architecture-bound MoE binding/sealing entry. Qwen4 supplies its typed
  dimensions and resident weight formats; shared sealing still applies the
  immutable canonical QT44/QT53 grouped capability checks.
- `crates/hipfire-arch-qwen4/src/gpu_forward.rs` owns GPU scratch allocation,
  row-bounded typed `Step` construction, the logical `[rows, experts]` router
  view used by both MoE preflight and execution, and whole-program preflight
  before token upload, embedding, stream setup, or PLE effects. QSA scalar
  metadata is copied from request-local operation state only after the
  validated program and final-output path succeed.
- `crates/hipfire-dispatch/src/pipeline/layer_ops.rs` owns neutral
  `HyperRead`, `HyperWrite`, `GatedDeltaNet`, `IndexedAttention`,
  `GroupedDepthwise`, and `Clear` contracts plus their shape/dtype
  validation and execution. Reusable max-chunk arenas remain capacity-checked;
  exact-row views are passed to wrappers whose low-level contracts infer
  logical row counts. GDN A-log and dt-bias validation follows the BF16
  `gated_delta_params` wrapper contract.
- `crates/hipfire-dispatch/src/pipeline/steps.rs` owns typed schedule order and
  composite/MoE preflight. Legacy scalar `Step` variants retain their
  established launch-time validation and are not widened by this cutover.
  `crates/hipfire-dispatch/src/pipeline/qt44_qt53_prefill.rs` owns the exact
  QT44/QT53 grouped format stages, while
  `pipeline/sealed_moe.rs` and `pipeline/moe_program.rs` remain the shared
  sealed route and lowering owners.
- `crates/rdna-compute/src/tensor_ops.rs` and
  `crates/rdna-compute/src/grouped_ops.rs` own neutral tensor/kernel wrappers
  and their executable dtype contracts. Fixed Qwen4 HIP helpers live in
  `crates/hipfire-arch-qwen4/src/gpu_ops.rs` and
  `crates/hipfire-arch-qwen4/src/qwen4_specific.hip`.
- The superseded shared Qwen4-specific interpreter/helper paths
  (`crates/hipfire-dispatch/src/pipeline/qwen4_program.rs`,
  `crates/hipfire-dispatch/src/pipeline/qwen4_prefill.rs`,
  `crates/rdna-compute/src/qwen4.rs`, and
  `crates/rdna-compute/src/qwen4_ple_ops.rs`) are removed from the current
  production path. No second architecture-owned interpreter or duplicate
  Qwen4 sequencing loop remains.

The current typed sequence uses the architecture-owned fixed inline capacity
of 384 steps; the configured 48-layer sequence consumes 337 composite/MoE
steps at its maximum. This is a bounded scheduling resource, not a
prompt-sized allocation.

### Scoped evidence and boundary

| Check | Current evidence |
|---|---|
| GDN metadata contract | Host fake-tensor regression accepts BF16 A-log/dt-bias and rejects F32 metadata; `cargo test -p hipfire-dispatch --lib pipeline::layer_ops::tests::gdn_accepts_bf16_metadata_and_rejects_f32_metadata --locked` passed. |
| Architecture binding compile | `cargo check -p hipfire-arch-qwen4 --lib --locked` passed after the shaped router-view repair. |
| Real-device short smoke | `.codeinsight+research/qwen4/architecture-seam-20260919/runs/post-refactor/oracle/short-two-token-moefixed.json` reports `status=pass`, `exact_bits=true`, `ids_equal=true`, gfx1151, and one continuation step. The earlier HyperRead and flat-router failures remain preserved in their dated stderr artifacts. |
| Full natural oracle and frozen comparisons | `.codeinsight+research/qwen4/architecture-seam-20260919/runs/post-refactor/oracle/full-result.json` reports `status=pass`, `exact_bits=true`, `ids_equal=true`, gfx1151, 16 continuation steps, and natural chunks `[128,128,35]`; `.codeinsight+research/qwen4/architecture-seam-20260919/runs/post-refactor/oracle/parity-check.json` also reports `status=pass`, exact prefill state, exact per-step outputs, and equality with the immutable original, frozen HC, and frozen N8 references. |
| AR/MTP serve smoke | `.codeinsight+research/qwen4/architecture-seam-20260919/runs/post-refactor/serve-final3/corrected-validation.json` reports `status=pass`: both arms return `Paris`, `finish=stop`, `saw_done=true`, nonempty output, no runaway, and no stream error; AR reports `mtp=null`, while native MTP reports `mtp=true` (`.codeinsight+research/qwen4/architecture-seam-20260919/runs/post-refactor/serve-final3/mtp/harness.json` records `tau=1.0`, `cycles=1`). The earlier malformed-environment attempt remains preserved under `.codeinsight+research/qwen4/architecture-seam-20260919/runs/post-refactor/serve/ar/`. |
| Final host gates | `.codeinsight+research/qwen4/architecture-seam-20260919/host-checks/rustfmt-check-scoped-bounded-views-final.log`, `.codeinsight+research/qwen4/architecture-seam-20260919/host-checks/cargo-build-release-workspace-all-targets-locked-bounded-views-final.log`, `.codeinsight+research/qwen4/architecture-seam-20260919/host-checks/cargo-test-release-workspace-lib-locked-bounded-views-final.log`, and `.codeinsight+research/qwen4/architecture-seam-20260919/host-checks/cargo-clippy-release-workspace-all-targets-locked-bounded-views-final.log` all record exit 0; the workspace library suites include rdna-compute 253/253 and hipfire-dispatch 285/285. |

This cutover is an ownership, ordering, shape, and dtype correction. It does
not admit a replay/PM4 route, physical EP2/EP4, retained admission, throughput
promotion, or quality/marketing claim. The exact oracle, serve, and host
artifacts above are correctness and integration evidence only.

## Verification checkpoint — HC bounded BF16 boundary batching (2026-09-19, rejected)

This was a candidate-only gfx1151 experiment in
`crates/hipfire-dispatch/src/pipeline/layer_ops.rs`.  It replaced the three
row-wise uses of the existing `rdna_compute::tensor_ops::bf16_roundtrip_f32`
boundary in shared `execute_hyper_read` with contiguous chunks bounded by the
already-owned `bf16_scratch.numel()` capacity.  The candidate added no kernel,
scratch allocation, precision/reduction change, architecture gate, or GDN
change.  The candidate source diff and snapshot remain preserved in
[`candidate-git-diff.patch`](../../.codeinsight+research/qwen4/current-pp-profile-20260919-v1/metadata/candidate-git-diff.patch)
and
[`candidate-source-snapshot.json`](../../.codeinsight+research/qwen4/current-pp-profile-20260919-v1/metadata/candidate-source-snapshot.json).

The candidate profile established a genuine launch-count reduction on the
natural `[128,128,35]` path (`observed_target_natural_launches=4656`) while
leaving the scalar `grid512` count unchanged at `84681`; see
[`candidate-profile-summary.json`](../../.codeinsight+research/qwen4/current-pp-profile-20260919-v1/candidate/profile/rocprof/current-marker-20260919/candidate-profile-summary.json).
The host tail/row-crossing probe was exact for rows `1`, `35`, and `128`, but
is explicitly a host simulation:
[`chunk-tail-row-crossing-probe.json`](../../.codeinsight+research/qwen4/current-pp-profile-20260919-v1/metadata/chunk-tail-row-crossing-probe.json).
The candidate full GPU oracle remained exact (`exact_bits=true`,
`state_exact=true`, `ids_equal=true`) and the AR/native-MTP Paris smoke passed,
but these are correctness and integration evidence, not throughput promotion:
[`result.json`](../../.codeinsight+research/qwen4/current-pp-profile-20260919-v1/candidate/oracle/result.json)
and
[`summary.json`](../../.codeinsight+research/qwen4/current-pp-profile-20260919-v1/candidate/serve-smoke-20260919-paris/summary.json).

For the natural marker, the all-symbol raw-trace ranking (not the
target-only hot-symbol filter) sums `4,715.890965 ms` of device duration over
`107,195` dispatches and `39` symbols inside the actual ROCTX bounds
(`7,142.418151 ms` marker wall time).  The top five are
`gemm_mq4g256v2_moe_grouped_top10_simt` (`1,109.758875 ms`),
`gemm_mq4g128v2_moe_grouped_top10_multirow_gfx1151` (`836.797373 ms`),
`gemm_bf16_xf32_multirow` (`740.026696 ms`),
`gemm_bf16_xf32_multirow_n8_gfx1151` (`718.992121 ms`), and
`gated_delta_step_shared_norm128_gfx1151` (`493.483439 ms`).  This profile
attribution is retained for a better next-lever choice and is not a native
throughput claim:
[`natural-marker-all-symbol-top5.json`](../../.codeinsight+research/qwen4/current-pp-profile-20260919-v1/metadata/natural-marker-all-symbol-top5.json).

The valid native primary measurement used fixed products, private
`max_seq=2048` configuration, DPM warmup `10s`, the exact 291-token prompt,
and six manually ordered ABBAAB arms.  The standard non-matrix native bench
ignores `--warmups`; its one internal `Hello` warmup is not a matching
291-token warmup.  The declared protocol therefore used native `--runs 11`,
discarded zero-based samples `0..9`, and retained only sample `10`; all raw
arrays of length `11` are preserved.  The cooperative GPU lock serialized
these arms, but unrelated GPU clients were preserved, so this is not an
uncontended measurement.  The durable record is
[`primary-summary.json`](../../.codeinsight+research/qwen4/current-pp-profile-20260919-v1/candidate/perf-v3-abbaab-20260919/primary-summary.json):
A prefill median was `51.4 tok/s`, B was `53.4 tok/s` (`+2.0 tok/s`,
`+3.9%`), but that delta was below the maximum observed arm spread
(`2.2 tok/s`); decode medians were equal at `13.2 tok/s`.  This is
`no_clear_win`, not a performance promotion.  The earlier invalid
`v1` configuration run and cold exploratory `v2` timings are excluded from
the decision; their artifacts remain preserved.

The separately run 24-token France fixture is exploratory only: the bench
itself warns that its prefill number is launch overhead, not throughput.
Its A/B raw records are retained under
[`second-prompt-20260919`](../../.codeinsight+research/qwen4/current-pp-profile-20260919-v1/candidate/perf-v3-abbaab-20260919/second-prompt-20260919/),
but this short prompt cannot repair the primary no-clear-win result or serve
as transfer evidence.

Decision: reject and do not retain this HC chunk lever.  The authored source
was surgically restored to the pre-experiment identity
`30984594a8bd6055e8deba1e487422a43a86292f514b0cbcc8e697043d06b496`;
candidate binaries, raw traces, oracle/smoke proofs, and corrected warmup
protocol evidence remain available for archaeology.  No 500-token/s prefill
or product throughput claim is made, and no replay/PM4, retained admission,
QT53 policy, or other route was changed.

## Verification checkpoint — grouped-down O4×R16 and dense LM-head dispatch (2026-09-20)

Two gfx1151 changes landed on top of the HC activation fusion candidate and
were verified against the frozen full-model oracle.

**QT53 grouped down O4×R16.** The routed down projection
(`m=2560, k=640`) moves from one wave covering two output rows and sixteen
route slots (`gemm_mq4g128v2_moe_grouped_top10_o2_r16_gfx1151`) to one wave
covering four output rows and sixteen route slots with a single accumulator
chain per slot — the same 64 accumulators, four-row batched weight loads,
vector four-K loads with a peeled K tail, and a per-slot gated wave32
reduction.  Each expert's weight rows are still fetched once per padded tile
while the number of X-reading waves halves.  The O2×R16 kernel, its registry
const and its launcher were removed on the same change; every non-gfx1151
path is untouched.  Evidence: 23/23 focused-driver cases bitwise equal, a
real-layer-0-input driver showing 0/409600 differing raw cells against the
production kernel blob, and the full oracle reporting
`comparison.exact_bits=true` with prefill logits `max_abs=0.0` on both the
291-token fixture and the 513-token/511-chunk boundary fixture.  Focused
driver timing on the production-shaped case: 5.521 ms vs 8.957 ms, 1.62×.

**Dense BF16 multirow LM-head dispatch.** A 14-shape focused measurement found
the r16 variant fastest on every production shape and also on the
`(248320, 2560, *)` logits family, which had been dispatched to the plain
four-row kernel and would have used the dominated n8 route at 128 rows.
`(248320, 2560)` was added to the gfx1151 r16 allowlist and the now-unreachable
n8 branch, its const and its kernel body were removed.  Measured
`(248320,2560,291)`: 400.8 ms plain, 266.9 ms n8, 164.6 ms r16.  This is a
latent-path and quality fix, not a change to the measured 291-token prefill
number: the current 291-token production trace contains no `(248320, 2560)`
multirow dispatch.

**Methodology correction (load-bearing for this thread).** The scratch
per-kernel drivers had been compiling kernels with `-ffast-math` while
production compiles without it.  Under `-ffast-math` clang may reassociate
`acc += a + b + c + d`, so such a driver does not compare the production tree;
one candidate was pinned to a non-production association and reported 18965 of
409600 raw output cells differing.  Production `-O3` without `-ffast-math`
reproduces the shipped kernel-cache blob exactly (the `-ffast-math` build
differs by 296 instructions on that kernel).  Focused drivers for this
workstream must be built with production flags.  The full-model oracle always
used production flags and was never affected.

**Measured effect.** Three fresh native processes of the landed tree,
byte-identical 291-token prompt (md5 `973900074bfd15d4adeeecdff3359082`),
`q8`/`contiguous`, `--spec off`, `noslots`, unrelated llama servers paused for
the whole measurement: prefill **114.0 / 119.2 / 113.0** tok/s (median 114.0),
decode 12.6–12.7 tok/s.  The preceding candidate measured 105.5 / 105.7 / 108.1
(prefill `hipfire` md5 `ed15c814`, `daemon` md5 `8949e51a`).  This is a
measured local delta under a paused-GPU protocol, not a product admission, a
speed-floor update, or a 500-token/s claim.  The pp500 target remains unmet:
MoE expert-weight traffic for one ≥291-token chunk is the full 512-expert
union (≈64.2 GB), which at the measured 223.9 GB/s streaming ceiling is
≈287 ms before any dense, attention, GDN or HC work, and the 500 tok/s budget
at 291 prompt tokens is 582 ms total.  The grouped gate kernel already runs at
about 90% of that ceiling, so gate retiling has little headroom; the remaining
headroom is concentrated in the dense BF16 projections (≈2.1 TFLOP/s, ≈12% of
FP32 peak) and the grouped down GEMM.

Rejected on the same day and not retained: a single-row gate O1×R16 tile
(104.0 vs 105.5–108.1 tok/s — the O2×R8 gate is already bandwidth bound, so
halving weight fetches is cancelled by doubled X traffic) and a half-slotted
gate O2×R16 tile that failed bitwise parity.  Both are recorded with raw
numbers in
[`REJECTED-TILE-EXPERIMENTS-20260920.md`](../../.codeinsight+research/qwen4/pp500-kernel-check/REJECTED-TILE-EXPERIMENTS-20260920.md).

### 2026-09-20 second pass — dense kernel rewrite, gate liveness elision, and the half-slotted defect

**Dense BF16 multirow rewrite.** `gemm_bf16_xf32_multirow_r16_gfx1151` was
rebuilt in place: one wave now owns sixteen output rows and two token rows
(grid `[ceil(N/2), ceil(M/16), 1]`, token tile on `blockIdx.x` so a row group's
weight rows stay cached), which takes activation bytes per FMA from 4.0 to
0.25 and load instructions per FMA from 0.258 to 0.078.  Measured per-shape
medians against the previous kernel: `(10240,2560,291)` 7.132 → 2.838 ms
(2.51×), `(12288,2560,291)` 8.517 → 3.403 ms, `(2560,6144,291)` 3.667 →
1.783 ms, `(320,10240,291)` 0.636 → 0.367 ms, `(10240,2560,512)` 11.957 →
4.964 ms, `(248320,2560,291)` 163.157 → 75.116 ms; required-shape geomean
2.21×, all twenty measured shapes 1.94×.  Every arm bitwise equal to the
four-row reference.  VGPR 241, no spills.  The r16 allowlist gained the
measured `(6144,2560)` entry (1.763× on that shape, 0.128 s of the plain path);
`(10240,320)` stays on the plain path because the same measurement showed
0.713× there.

**Gate liveness elision.** In
`gemm_mq4g256v2_moe_grouped_top10_o2_r8.gfx1151.hip`, a subtile with no live
slot now elides its whole K walk (`group_walk = any_live ? groups_per_row : 0`)
instead of decoding weights that are multiplied by nothing.  The fold, the
shuffle and the store loop still run and still write the zero every sentinel
slot owes, so the output is bit-identical; VGPR, symbol, grid and metadata
contract are unchanged.  Focused driver 7/7 bitwise equal; real-workload gate
device time −5.7%.

**Located defect (recorded because a silent structural bug is worth the
space).** The earlier half-slotted gate O2×R16 candidate failed full-model
parity with prefill logits `max_abs` 12.77 and no associativity explanation.
The cause was accumulator *lifetime*, not ordering: the design shared one
4-chain × 8-slot × 2-row accumulator set across both slot halves and never
re-zeroed or folded it between them, so half 0's partial sums survived into
half 1's epilogue and half 1's stores wrote `half0 + half1` for the second
eight slots — the normal production case being a second half that is entirely
sentinel, where slots 8..15 simply received slots 0..7's values (reproduced
exactly as `slot 8 = slot 0`, `slot 9 = slot 1`, … in five independent focused
cases).  Re-zeroing and folding per half makes the same design raw-F32 bitwise
equal, which confirms the diagnosis; both the half-slotted and the
128-accumulator O2×R16 designs nevertheless measured slower than production
(1.20× and 1.07× on the production-shaped case), so only the liveness elision
was kept.

**Final measured state.** Four fresh native processes of the landed tree,
byte-identical 291-token fixture (md5 `973900074bfd15d4adeeecdff3359082`),
unrelated llama servers paused for the whole measurement:
**143.3 / 137.7 / 148.9 / 142.8** tok/s (median 142.8), decode 12.6–12.8.
Full-model oracle on the same tree: `exact_bits=true`, `ids_equal=true`,
`state_exact=true`, prefill logits `max_abs=0.0` on both the 291-token and the
513-token/511-chunk fixtures.  The AR `serve_harness.py` battery route on the
same binaries finishes `Paris` with one terminal event and zero post-terminal
bytes.  Day progression at the pinned fixture: 105.7 → 114.0 → 141.8 → 142.8
tok/s median.  `hipfire` md5 `1c336a8ba4ae01fa79f6c7852c046a6f`, `daemon` md5
`134a263ab37fd823831d3c9b120d12c7`.

The pp500 target is **not** met and the remaining distance is structural, not
incremental.  The batched-prefill window is now 1.58 s of which the grouped
gate is 0.43 s, the dense projections 0.54 s, the grouped down 0.23 s and the
batchable attention/GDN/HC remainder ~0.34 s; roughly 0.47 s of the 2.03 s
time-to-first-token is outside that window (embedding, PLE gather, final-row
LM head, sampling, host setup).  MoE expert-weight traffic for one
≥291-token chunk is the full 512-expert union (48 × 1.337 GB ≈ 64.2 GB), which
at the measured 223.9 GB/s streaming ceiling is ≈0.29 s before any other work,
against a 0.58 s budget for 500 tok/s at 291 prompt tokens.  Reaching 500 tok/s
from here would need the remaining grouped and dense GEMMs to run several
times closer to their ceilings simultaneously, which the bitwise-parity
contract (no tensor cores, no reassociation) and the measured gate behaviour
at ~90% of the streaming ceiling do not presently allow.  No promotion,
admission, or speed-floor claim is made; these are measured local deltas under
a paused-GPU protocol.

### Certified identities for the 2026-09-20 final state

| artifact | digest |
|---|---|
| `target/release/hipfire` | md5 `a9be601fc3824da20a5dde8c829a58db` |
| `target/release/daemon` | md5 `91cfdcf51dd825e64084b8f5fdaf3ad1` |
| oracle `qwen4_greedy16_continuation` | md5 `a0b071d5e88f4a9af68fdc6baf4343f5` |
| `kernels/src/gemm_mq4g256v2_moe_grouped_top10_o2_r8.gfx1151.hip` | sha256 `353dcc2d236170a824545cd013c405483182b85c43316668fee65a7ba9b3987c` |
| `kernels/src/gemm_bf16_xf32_multirow_r16.gfx1151.hip` | sha256 `151ace64c9301556727eb82647ef7a5eb557276067264c9008fe2983793e9fc4` |
| `kernels/src/gemm_mq4g128v2_moe_grouped_top10_o4_r16.gfx1151.hip` | sha256 `26409eaba0f059d006e125d58a9e88dceb2fe1bd312854c2cd5c802c159959ce` |
| `crates/rdna-compute/src/gemm.rs` | sha256 `8ce64ce241b9f5946de5c32bdfb4a2278c4a6a30ba3850b545bebe993db3ce39` |
| `crates/rdna-compute/src/kernels.rs` | sha256 `9c1827ad6b9beb1eb53012cd8d18c6c8f656b009523522f1aa79324659375964` |

These binaries were rebuilt after the final `rustfmt` pass so the certified
binary corresponds to the exact source text above; the oracle re-run on them is
`FINAL2-oracle/result.json` (`exact_bits=true`, `max_abs=0.0`,
`state_exact=true`).  Seven fresh native processes across the two final passes
give **143.3 / 137.7 / 148.9 / 142.8 / 147.4 / 146.4 / 143.3** tok/s (median
143.3, best 148.9).  The serialized full-library workspace suite passes on this
tree (including `rdna-compute` 253 tests), and every touched Rust file is
`rustfmt`-clean.

### 2026-09-20 third pass — PLE read granularity (the largest remaining host cost)

Product-path attribution of the ~0.45-0.56 s that sat outside the batched GPU
window found a single dominant item: **492 ms of GPU idle on the PLE lease
wait**.  The cause was a chosen userspace page size, not an artifact
requirement: `PLE_PAGE_TARGET_BYTES` was 2 MiB rounded down to rows
(`PLE_PAGE_BYTES` 2,096,960 B), and because that value is both the cache unit
and the read unit, one 320-byte PLE row cost a full 2,096,960-byte positioned
read.  PLE reads sixteen rows per token scattered across 128 shards, so a
291-token chunk pulled ~4,656 rows / **2,998 distinct windows ≈ 6.29 GB for
1.49 MB of needed bytes (4,219× amplification)**, measured at 21.3 GB/s of
page-cached `pread` and ~19 GB/s of subsequent copy.

The fix reduces `PLE_PAGE_TARGET_BYTES` to 4 KiB (`PLE_PAGE_BYTES` 3,840 B =
12 rows), which bounds the amplification to ≤16×, and replaces the page
cache's recency scan with a `BTreeMap<stamp, PageKey>` index because a 4 KiB
page yields ~70k cache entries for which the previous `min_by_key` scan was not
viable.  LRU semantics are preserved exactly (`cache_is_bounded_and_evicts_lru`
unchanged), and a new regression test
(`read_window_stays_within_the_row_page`) pins the read window: a 64-row
prefetch must read one window per distinct page and at most 16× the requested
bytes; it fails on the old 2 MiB window with 6,553×.  Untouched: no mmap, no
pinned fd, 256 MiB page-cache budget, two-buffer staging pool, ticket/epoch and
cancellation semantics, artifact format, descriptor validation.

Evidence: `.codeinsight+research/qwen4/ple-window/` (read accounting, mark
tables for both arms, oracle runs, three candidate runs plus an in-session
control, serve smoke, summary).  Oracle on the landed tree: `exact_bits=true`,
`ids_equal=true`, `state_exact=true`, prefill logits `max_abs=0.0` on both the
291-token and the 513-token/511-chunk fixtures.  AR serve route: `Paris`,
`finish=stop`, one terminal event, zero post-terminal bytes.

**Effect.** Four fresh native processes: **184.6 / 184.0 / 184.9 / 183.9** tok/s
(median 184.0), decode 13.3-13.4.  Two things changed at once: the level rose
25.7% over the previous median 146.4, and the fresh-process spread fell from
±4-5% to **0.7%** — the page-cache/I/O variance that had been polluting every
earlier measurement is gone, so the remaining deltas on this thread can now be
resolved much more finely.

Day progression at the pinned fixture: 105.7 → 114.0 → 141.8 → 146.4 → **184.0**
tok/s median; 88.6 → 184.0 (~2.08×) across the session.

**Identity caveat, corrective.** Binary md5s in this checkout are not a tree
fingerprint: rebuilding an untouched crate twice produced `60d5835d` then
`e29322e2` for identical sources, and the `hipfire` CLI md5 stayed `a9be601f`
across the PLE change because that code is not in the CLI's dependency graph.
Source sha256 remains the reliable identity; binary md5s in this document
should be read as "the binary that produced the recorded numbers", not as
reproducible fingerprints.  `crates/hipfire-arch-qwen4/src/ple_rows.rs` md5
`e7f55a70c4204937ce7e8dd72384c6d4` (landed), `crates/hipfire-arch-qwen4/src/gpu_forward.rs`
carries only the pre-existing HC-batch change.

**Where the remaining time is.** The batched window is now 1.495 s against a
1.576 s time-to-first-token, so out-of-window overhead is down to ~81 ms: gate
0.399 s, dense r16 0.329 s, grouped down 0.223 s, plain dense 0.156 s, GDN
0.117 s, attention 0.074 s.  The 500 tok/s budget at 291 prompt tokens is
0.582 s.  Even setting every GEMM to zero would only reach ~0.47 s, and the
MoE expert-weight traffic alone (the full 512-expert union, ≈64 GB, at the
measured 223.9 GB/s streaming ceiling) is ≈0.29 s of unavoidable traffic.  A
realistic floor with this kernel family is therefore ≈1.1-1.2 s (≈250 tok/s),
not 0.58 s.  No promotion, admission, or speed-floor claim is made.

## Disposition — pp500 prefill target closed (2026-09-20)

The Qwen4 gfx1151 prefill workstream is closed at its measured result rather
than at the pp500 (500 tok/s) target.

Reached: **189.3 tok/s median** on the pinned 291-token fixture (fresh
processes: 189.3 / 189.0 / 191.3), up from 88.6 at the start of the session —
**2.14×** — with exact bitwise parity preserved end to end (`comparison.exact_bits=true`,
`state_exact=true`, prefill logits `max_abs=0.0` on both the 291-token and the
513-token/511-chunk fixtures), the AR serve route verified (`Paris`, one
terminal event, zero post-terminal bytes), the serialized workspace library
suite green, and every touched crate `rustfmt`/`clippy` clean. Landed as
commit `34d79c453` (grouped and dense kernel work, gate liveness elision and
the measured dispatch entries) and `2c175b7c2` (PLE read-granularity fix).

Not reached: 500 tok/s, still 2.6× away, and closed as **not achievable under
the byte-parity contract**, not as an implementation gap left open. The bound
is structural: MoE expert-weight traffic for one ≥291-token chunk is the full
512-expert union (≈64 GB) which is ≈0.29 s against a 0.582 s total budget, and
the grouped kernels are now decode/issue-bound rather than bandwidth-bound, so
further tiling buys single-digit percentages. Estimated practical ceiling for
this kernel family: ≈250-400 tok/s.

What would change the answer, for whoever picks this up next: relaxing the
bitwise-parity requirement for the prefill path so WMMA/MFMA-class kernels
become admissible, which needs a new acceptance definition (numerical tolerance
rather than exact bits) and re-validation of every affected kernel; or measuring
a different, longer fixture, where fixed launch and read costs amortize — that
raises the reported number without making a 291-token prompt any faster.

No promotion, admission, or speed-floor change is claimed by any of this, and
the 500 tok/s target remains unmet.
