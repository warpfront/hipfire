# Explicit native MQ6 railgun programs

The native gfx1151 lane-row MQ6 single/x4 kernels can be authored and executed
as a fixed-binding railgun program through
`rdna_compute::mq6_railgun::PreparedMq6Program`. This is the railgun follow-up
to the native MQ6 capture work in #941. It supplies an explicit reusable
operator API; automatic model capture integration remains a separate step.

`railgun::mq6::Mq6Facts::from_hit` requires the canonical bundle's actual
schema-9 A1/G8 receipt from a verified `JitCorpus` lookup. The author derives
memory effects from those facts through `kernel::author`, then the runtime
executes the resulting `plan::Pm4Plan` with ROCr's existing PM4 transport.
No name-based MQ6 `pointer_effects` entry is added to the Redline recorder.

The kernels are `gemv_mq6g256v2_xbatch_pm` and
`gemv_mq6g256v2_x4_xbatch_pm`. Their loader kernarg segments are 36 and 96 bytes,
respectively. The single-kernel launch record is zero-padded to 48 bytes; this
padding is not part of its loader segment. The code checks the receipt,
embedded bundle, extracted device ELF and loaded ABI before preparing an IB.

## Execution and ownership

- Exact gfx1151, F32 activation/output, MQ6G256V2 weights, K positive and
  divisible by 256, batch 1 through 8. An inactive x4 slot is `None`, encoded
  as null weight/output pointers with M=0. Shapes and bindings are constants.
- Prepare probes ordinary, same-device HIP allocations and subviews. It
  refuses out-of-bounds or unaligned views, conflicting allocation identities,
  and same-node output/input aliases. VMM and host-mapped allocations need
  their own range/lifetime proof and are refused here.
- Multiple nodes may form a dependency chain. The program uses conservative
  allocation-wide M1 effects and a private WaitIdle + System cache table for
  all hazards. This does not qualify global gfx1151 cache rows or claim A2
  span proofs. IF/WHILE is unnecessary for this straight-line API; conditional PM4 operations are explicitly refused before submission.
- Replay joins the current HIP producer stream, submits the retained planned
  PM4 IB and synchronously waits for terminal completion before returning to
  a HIP consumer. The existing PM4 transport is reused; no Redline recording
  or graph-capture admission is enabled by this API.
- Prepare/replay are unsafe: the caller retains every original allocation,
  including the owners behind subviews, at the same addresses until the
  program is quiesced and destroyed. Shallow tensor views do not own memory.
  Other-stream producers must be joined onto the current HIP stream.
- An execution failure poisons the program. Unknown quiescence quarantines
  its owned transport, code and kernargs; external allocations must remain
  live until GPU reset or process exit. A new address or shape requires a new
  prepare. Existing model routes and Redline defaults are unchanged.

## Reproduce the numerical and replay oracle

Use a private Cargo target directory for this checkout. No model is needed.
Create new output directories rather than overwriting previous evidence:

```bash
cargo build --release --locked -p railgun-cert
cargo build --release --locked -p rdna-compute --features lab \
  --example mq6_railgun_oracle
cargo test --release --locked -p railgun --lib mq6::tests

"$CARGO_TARGET_DIR/release/railgun-cert" certify --out "$CERT_OUT" \
  kernels/qwen4_lane_rows_pm_gfx1151.hxaco
python3 scripts/railgun/mq6_native_corpus.py \
  --receipt "$CERT_OUT/receipts/gfx1151/bc34ee389bd990386bdfc93d94b66752affcd3fbd33bec212a6b4edcaa51a963.receipt.json" \
  --out "$CORPUS_OUT"

# Acquire the machine's GPU locks before running this command.
"$CARGO_TARGET_DIR/release/examples/mq6_railgun_oracle" \
  --corpus "$CORPUS_OUT" --out "$ORACLE_OUT" \
  --source-head "$(git rev-parse HEAD)"
```

The packing helper only constructs a native fixture corpus from the real
certificate. It does not certify an object, invoke hipcc or admit a default
program. The CPU fixture contains the complete actual bundle receipt, with
negative mutations tested separately.

The GPU oracle compares against independently launched incumbent HIP singleton
rows, including full output guards. It covers single/x4 B1..8, ragged M,
inactive slots, native GDN/QSA projection shapes, and two-node RAW chains.
Each prepared program receives changing inputs A/B/A without changing its IB
or kernargs. No-write, poison, wrong-binding and real device-corruption
controls must be detected. Raw per-case program/plan/ABI identities and a
summary are written to the requested directory. This is component correctness
and retained-replay evidence; it does not measure model quality or speed.

## Review follow-up: order the poison producer (2026-10-25)

The reported `x4_empty_first B=1 part=3: unwritten active word c=0 r=0`
failure was reproduced after integrating beta
`6c53fbf0f9104864d750a90e0d56ba445f631416`. The unfixed build at
`41bd815b3b814213d00026134fd5b66bbd9e2c66` failed in the first of three
fresh processes; its next two processes passed. Its frozen binary SHA-256 is
`afd84bea03d2e11f81048c35aa2922c9f8d3fcd6e4737ce55413cadfa197568e`.
The older frozen PR binary had also passed three new processes. A passing
single process was insufficient to expose this ordering error.

`Output::reset` submitted `hipMemset` to the default stream. A base device
allocation's memset may complete asynchronously (see the
[ROCm implementation](https://github.com/ROCm/clr/blob/develop/hipamd/src/hip_memory.cpp),
`ihipMemset` / `hipMemset_common`). Replay joins the current active HIP stream
before submitting its independent ROCr PM4 IB. With no new active-stream
operation joining that default-stream fill, the earlier device sync in
`Fixture::reset` only drained work submitted before the poison write. The
later poison could race with and overwrite the PM4 result.

The oracle now queues `hipMemsetAsync` on the same active producer stream that
replay waits for; the no-active-stream branch keeps the matching default
stream path. Comparison errors also identify the HIP reference versus replay
output. The MQ6 kernel objects, shapes, grid, kernarg ABI, receipt, runtime
handoff and Redline defaults were not changed by this fix.

Verified source: `8352551d32f68b5fa776bcbe49d6774fd9eac4d1`. Five fresh
processes each passed all 40 shape/batch cases and 120 exact changing-input
replays: **200 cases / 600 replays**, including empty-first B1..8, both RAW
chains, guards and every existing negative control. No mandatory case was
skipped. Frozen fixed binary SHA-256:
`17bbe71339546005d260fd41d0e6920d89db18c13cae6788e191e51aab78d0d3`.

On that source the affected CPU suites passed: railgun 23; rdna-compute
1,125 (29 existing ignored); railgun-cert 6, including the inventory test
whose prior beta omissions have since been fixed upstream. The six required
static checks and env-docs/lifecycle passed. Named-file rustfmt and diff
whitespace checks passed. No full model or performance claim is added.

Raw evidence under the same Nahida output root:

- `review-reproduction-v1/`: the older frozen PR binary, three new processes.
- `review-latest-beta-unfixed-v1/`: the reproduced failure, source/binary pins
  and all three fresh-process logs.
- `review-fixed-stream-v2/`: the fixed source/binary, five full oracle reports
  and process-retirement records; both GPU locks held.
- `review-cpu-v1/`: affected CPU suites and required checks.

## Initial validation (before the poison-ordering fix)

Tested implementation: `3e5f176d71d19e803f7a0a1a7b7cf82906faf689`, based on
beta `6d81107a55c6a8e7748635edd9d5e0a0529eda8d`. The checkout was clean when
the hardware binary was built and frozen. Hardware: gfx1151, PCI
`0000:c5:00.0`, HIP 7.16; both machine GPU locks held.

The oracle passed 40 shape/batch cases and 120 exact changing-input replays:
38 single-node cases plus two two-node RAW-chain cases. The chain stayed in
one PM4 segment with a planned WaitIdle + System RAW boundary; both the device
intermediate and final output matched the independent HIP reference. Each of
seven base fixtures detected no-write, one-word poison, wrong-X binding and
actual device-output corruption controls. No mandatory oracle case was skipped.

```json
{
  "status": "PASS",
  "shape_width_cases": 40,
  "exact_replays": 120,
  "single_node_cases": 38,
  "raw_chain_cases": 2,
  "raw_chain_replays": 6,
  "raw_chain_nodes": 2,
  "raw_chain_segments": 1,
  "raw_wait_idle_system": true,
  "mandatory_skipped": 0,
  "no_write_controls": 7,
  "one_word_controls": 7,
  "wrong_binding_controls": 7,
  "device_corruption_controls": 7
}
```

The frozen oracle binary SHA-256 was
`a35ecc1233da4d7b35bcaf137bc8dccec90a75564406a75ce1933338d215b310`.
The real bundle receipt SHA-256 was
`c343fa457e16e95a9d76e144dcd31d25d81d66cd1a3b25a3ff3cedf36fa8c285`;
both target kernels were A1 Proven and G8 pass. Source, process retirement,
binary identity, certificate, full per-case program/plan/ABI reports and raw
logs are retained on Nahida under
`/home/husrcf/Code/hipfireQC/tmpOutput/hipfire-mq6-railgun-20261025/`.

CPU checks: `cargo test --release --locked -p railgun --lib` passed 23 tests;
the equivalent rdna-compute run passed 1,120 with 29 pre-existing ignored
tests. The targeted lab example and certifier built successfully. The six
required static checks (leanup ratchets, crate maps, scratch growth, released
changelog, expectation ratchet diff and supply-chain bans/licenses/sources)
passed, as did env-docs, lifecycle and named-file rustfmt checks.

`cargo test --release --locked -p railgun-cert --lib` passed 5 tests and failed
the tree inventory test. Its six missing sites are all in beta's existing
Qwen4 decode-share code. The frozen baseline certifier scanning a `git archive`
of the exact beta base produced byte-identical errors to the candidate
certifier scanning the candidate tree. The new explicit-MQ6 refusal site is
inventoried; this PR does not repair the unrelated baseline omissions.

These results cover the explicit component API. Automatic model capture,
model quality and end-to-end performance have no result or claim here.
