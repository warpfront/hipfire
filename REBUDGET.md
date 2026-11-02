# Qwen4 gfx1201 expert hot-set rebudget

## Decision

`HIPFIRE_QWEN4_EXPERT_HOTSET_POLICY=whole-first` is an opt-in,
conservative decode-safe policy. It keeps the ordinary whole-layer load's
placement, owners, pointer tables, decode route and G2 prefill staging. It
creates **zero mixed/tier-owned layers and zero additional hot experts**.
Unset retains beta's existing ranked-hotset path, logs and byte accounting;
gfx1100 is unchanged. This is not a default-flip proposal supported by new
performance measurements.

The guarantee is structural equality with the same load's `hotset=0` plan,
not an assertion that wall-clock execution is immune to measurement noise.
A mixed layer has not been established to cost no more than a host layer,
so adding even a small residual-budget hot prefix cannot meet that guarantee.
The implementation therefore deliberately leaves residual bytes unspent.

## Receipt and range correction

Receipt: `/home/kaden/qcal/release-0.4.3/offload-r9700-regression/runs/beta18b-component-bisect-cardE/summary.md`,
lines 15–95 (placement), 98–115 (timings), beta `18b216848`.
Source evidence below is anchored to branch base `e6541e6`; implementation
additions can shift line numbers without changing these cited functions.

**The placement receipt table has an off-by-one interpretation.** The log's
`layers 0..14` is a half-open range: **14 layers, 0–13**, not 15 layers.
`place_routed_experts` retains a trunk entry only when `layer < vram_layers`
(`crates/hipfire-arch-qwen4/src/expert_residency.rs:1609–1625`).
`carriers.rs:2434–2435,2460–2464` applies and prints that count. The staging
log naming donors 12 and 13 corroborates the 14-layer count. Likewise the tag's
`0..15` means 15 layers, 0–14, not 16. Do not size a replacement against the
incorrect inclusive interpretation in the receipt table.

| Arm | Whole trunk layers | Device hot experts | Device expert MiB | Dedicated stage MiB | Borrowed capacity / pinned restore MiB |
|---|---:|---:|---:|---:|---:|
| beta default | 0 | 6,847 | 17,050 | 973 + 961 | 0 |
| beta hotset=0 | 14 (0–13) | 0 | 17,850 | 0 | 2,550 / 2,550 |
| beta stage=0 | 0 | 7,624 | 18,985 | 0 | 0 |
| v0.4.2.1 | 15 (0–14) | 0 | 19,125 | 0 | 2,550 / 2,550 |

The receipt's host totals include MTP/other placement bookkeeping; they are
44,149 MiB for the default tier load, 42,214 MiB for stage=0, and 43.6 GiB
routed host payload for hotset=0. These are not additional device budgets.

| Cell (tokens/s) | beta default | hotset=0 | stage=0 | both off |
|---|---:|---:|---:|---:|
| pp512 | 1112.8 | 1177.3 | 1184.0 | 1182.4 |
| pp8192 | 4681.7 | 3789.2 | 2979.6 | 2823.7 |
| pp16384 | 4597.5 | 3777.1 | 2948.4 | 2784.6 |
| tg64@128 | 32.13 | 35.17 | 32.81 | 35.17 |
| tg64@2048 | 32.21 | 33.54 | 32.93 | 33.57 |

The default is 8.64% below hotset=0 at tg64@128 (the reverse comparison is a
9.46% improvement), and 5.48% below at pp512. Its pp8192 is **23.55% above**
hotset=0; equivalently hotset=0 is 19.06% below default. The often quoted
“+19%” uses the reverse denominator. Stage=0 still loses short decode despite
placing 777 more experts in VRAM. Both-off and hotset=0 have essentially the
same short decode, so staging itself is not a decode-time operation.

## Capacity diagnosis

The receipt logs 32,550 MiB free, 5,115 MiB non-expert weights, and a base
reserve of 7,935 MiB. The reserve is the larger of a historical floor and
6,294 MiB census, **not** their sum. Both placement families also retain a
512 MiB safety margin. Tier ownership adds 417,792 bytes (0.3984375 MiB) of
frozen pointer/codec tables:
`expert_residency.rs:1475–1505` (`EXPERT_TIER_TABLE_BYTES`,
`EXPERT_HOTSET_VRAM_MARGIN_BYTES`, `expert_hotset_budget`).

The logged ranked budget before stage is 18,986 MiB, and after the 1,934 MiB
stage charge is 17,052 MiB. `carriers.rs:2115–2147` recomputes a fixed point:
shrinking the expert budget can grow the host-only stage, so it replans until
that stage fits. `expert_residency.rs:1522–1538` sums the largest host payload
per layer parity, plus tail padding. These are real allocations made by
`gpu_forward/slab_stage.rs:524–575`, not a speculative reserve.

One expert is 2,611,200 bytes; one 512-expert layer is exactly 1,275 MiB.
The 14 baseline resident layers need 17,850 MiB. The 17,052 MiB post-stage
budget cannot retain them: it is short by about 798 MiB. Even the pre-stage
budget has only about 1,136 MiB left after those layers, less than the existing
1,934 MiB tier stage. (Receipt budgets are integer-MiB rounded.) Fifteen
whole layers need 19,125 MiB and already exceed the pre-stage budget. Thus
retaining the baseline layers **and** the same dedicated ranked-hotset G2
buffers is incompatible with this load's budget.

The stage=0 prefix's extra 777 device experts occupy 1,934.912 MiB of payload.
That recovers only 32.81 vs 32.13 tg64@128, still below the whole-layer arm's
35.17. Stage capacity displacement is a contributor, not a sufficient
explanation of the observed decode regression.

## Decode route diagnosis: same indexed schedule, different ownership

The source does **not** support blaming an extra GPU tier indirection or an
extra per-layer expert kernel:

- `crates/hipfire-arch-qwen4/src/program.rs:897–906,1016` sends one-row
  non-FP8 calls through `build_moe_decode` / `seal_decode`, with batch size 1.
- `crates/hipfire-dispatch/src/pipeline/moe_program.rs:1885–1933` selects the
  decode route from quant dtypes, top-k, shapes, EP mode, architecture and
  route policy; neither the host bit nor tier membership is a route selector.
  The QT44/QT53 shapes and top-k=10 match in all three placement cases.
- `gpu_forward.rs:2584–2587,2741–2744,2807–2840` uses the tier view's
  per-expert placement/address tables for tier layers and ordinary indexed
  pointer tables for non-tier layers. Both supply one gate/up pointer table
  and one down pointer table. They are not an additional table wrapped around
  the ordinary tables.
- `moe_program.rs:1406–1531` lowers the same routed `GateUp`, `Activation`,
  `Down`, and (for the grouped route) `Combine` stages. There is no
  residency-dependent extra stage. This is a count of logical routed stages,
  **not** a claim that every stage always means exactly one GPU launch.
- Tier-bound sealing additionally validates generation/digest, placement,
  codec and table identity (`pipeline/sealed_moe.rs:1358–1365,3400–3475,4559–4596`),
  and gives the retained route a tier-identity key suffix
  (`sealed_moe.rs:1296–1313,4074–4075`). This is CPU validation/cache identity,
  not a different expert kernel. These citations do not establish how much
  validation executes per replayed token, so no per-token latency is assigned.
- Host bits do affect **prefill** admission and grouped launch flags
  (`moe_program.rs:2199–2203,2313–2321`;
  `pipeline/qt44_qt53_prefill.rs:1430–1466`), a separate route from one-row
  decode. Decode does not consume the temporary prefill stage pointer tables
  (`families/moe.rs:1808–1809`).

Consequently a fully device layer has 512 device expert addresses and no
host expert reads; a fully host layer has 512 host-mapped addresses; a mixed
layer has both in the **same** indexed route. The default has potential
host-mapped access in 48/48 layers, versus 34/48 layers in beta hotset=0
(and 33/48 in the tag), while losing all 14 fully device layers. Tier CPU
identity validation applies to 48 tier layers versus zero baseline layers.
Those are source/placement counts, not measured latency multipliers.

The regression is therefore a placement/data-movement/ownership regression,
not evidence of a tier-only kernel launch or an extra GPU table dereference.
The receipt establishes the net loss but cannot separate host-access latency,
work distribution, locality, route-key handling and CPU sealing costs. In
particular it does **not** measure a mixed layer against the same fully host
layer in isolation; claiming mixed is always slower (or faster) would exceed
the evidence. Exact baseline ownership is the defensible decode-safe choice.

## Sidecar coverage is not a decode guarantee

The bound sidecar is
`/home/kaden/qcal/models/qwen3.8-flash-next-gptq3.mq4.expert-hotset`,
SHA-256 `7f6760e11a86e4ffe0b05c050ce64e6eb772ab748b6b11c8b2e6dbc566e57086`.
Its header declares 3,705 capture tokens, top-k 10, 48 × 512 expert ids,
code-prompt counts weighted ×2, and 25 prompts disjoint from the eight bench
prompts and warmup. Counts sum to 2,616,000 **weighted** selections; they are
not 3,705 × 48 × 10 unweighted selections.

Summing `counts[id]` over its stored ranked prefixes (CPU JSON arithmetic):

| Placement | Weighted covered selections | Weighted in-sample coverage | Device experts per layer |
|---|---:|---:|---:|
| 6,847 ranked experts | 1,698,581 | 64.930466% | 121–164, all 48 layers mixed |
| 7,624 ranked experts | 1,796,152 | 68.660245% | 133–192, all 48 layers mixed |
| 14 whole layers | 763,000 | 29.166667% | 512 in layers 0–13, 0 otherwise |

The sidecar's high weighted in-sample hit rate is genuine. It does not predict
bench token routes, selected-expert memory overlap, cache behavior, or the
critical path of ten experts dispatched together. It therefore cannot prove
that replacing 14 fully device layers and 34 fully host layers with 48 mixed
layers is faster. No per-layer timing or bench-route hit trace is present in
the supplied receipt; attributing a precise latency share to PCIe access or
CPU validation would be unsupported.

## Why the G2 prefill gain cannot simply be retained

The packed whole-layer G2 path borrows the last two resident layer payloads,
stages complete host layers, then restores those resident owners from pinned
copies before returning to decode (`gpu_forward.rs:406–431`; the receipt's
hotset=0 staging line names layers 14..48 and donors 12/13).

The tier G2 path copies **only host-tier experts** into two dedicated parity
buffers. Device experts keep their original owner addresses; only host
pointer-table entries are rebased, owners are never overwritten and no donor
restore is required (`gpu_forward/slab_stage.rs:25–35,187–189,436–439,524–532,631–645,720–755`).
That is a genuinely different prefill data movement plan, not merely the same
buffers under another placement label. Source explains the eliminated restore
and partial-layer copies; it does not by itself apportion the measured 892.5
tokens/s pp8192 difference among them. The source comments describing slab
staging's older whole-host PCIe slowdown are not measurements of this arm.

Forwards below 2,048 rows do not use the default gfx1201 staging path
(`gpu_forward.rs:410–427,456`), matching the pp512 regression despite strong
long-prefill results. Keeping only “stage enabled” cannot preserve the ranked
plan's measured speed when the placement and copy schedule change.

`whole-first` keeps the **baseline** long-prefill optimization, not the
incompatible ranked-hotset gain. Packed artifacts retain borrowed donors,
zero incremental stage VRAM and pinned restore copies. Slab artifacts retain
the ordinary dedicated stages charged by
`expert_residency.rs:1541–1605` (`dedicated_slab_stage`, `PlacementCharges`).
The same `auto_placement_layers` and placement mutation are used, so neither
stages nor safety reserve can steal a baseline layer. No new kernel route,
prefill/decode owner swap, or unsafe owner alias is introduced.

## Implementation and expected placement summary

The additive policy parser/eligibility resolver is in
`expert_hotset.rs:139–217`; unset and `ranked` are the legacy selection.
Empty or unknown values are load errors naming the variable.
The policy is inactive and silent outside gfx1201. On gfx1201, placement
already chosen by UMA, EP/agreed split, pipeline bands or explicit
`HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS` is left unchanged with an ignored-reason log.
This creates no new PP refusal.

`carriers.rs:2078–2101` bypasses sidecar preflight only for the active opt-in;
`carriers.rs:2441–2468` passes the same baseline inputs and charges to
`WholeFirstPlan::auto`. The helper calls `auto_placement_layers` rather than
reimplementing its sizing formula (`expert_residency.rs:1653–1800`).
`carriers.rs:2503–2506,2523–2531` prints the opt-in summary. These line numbers
refer to the implementation, not the base-source anchors above.

For the supplied receipt's packed-artifact geometry, the expected additional
line is:

```text
  qwen4 expert hot set: policy=whole-first, whole layers 0..14 (14 of 49 MoE layers), hot experts 0, expert resident 17850 MiB, stage allocated 0 MiB, borrowed 2550 MiB from resident layers 12 and 13 (2550 MiB pinned host restore copies); hotset=0 whole-layer placement, sidecar not consulted
```

The denominator 49 includes the MTP MoE layer; the 48-layer trunk's resident
range is still 0..14. Actual current-beta census/free memory can change the
count; the same-load hotset=0 arm must agree. Dedicated slab stages instead
print allocated/charged stage MiB and zero borrowed MiB. Disabled staging
prints zero for both. Existing ordinary staging lines still describe runtime
stage availability. This summary is planned capacity, not a new GPU receipt.

CPU tests cover absent/ranked selection, gfx1100 inactivity, ignored placement
overrides, whole-layer count equality across budget sweeps, exact receipt
geometry, dedicated/borrowed/disabled stage accounting and refusal parity.
They use pure inputs rather than mutating process-wide environment variables.

## Measurement handoff and default decision

Measure three otherwise identical fresh-process arms on an R9700: unset
(beta ranked default), `HIPFIRE_QWEN4_EXPERT_HOTSET_POLICY=whole-first`, and
`HIPFIRE_QWEN4_EXPERT_HOTSET=0`. Keep the same model, stage switch, sequence
capacity, corpus, runs and warmup as the receipt. Inspect the opt-in policy
summary and the unchanged ordinary placement/staging lines. Use the branch's
current beta base, not the old receipt's SHA, for its same-load comparison.

No GPU run was performed for this implementation. The old receipt supports
expecting whole-first to match hotset=0 structurally; it does not constitute a
measurement of this branch or justify flipping the default. A default change
must explicitly trade the existing long-prefill benefit against decode, and
requires the requested matched R9700 arm results first.

## CPU gate receipt

All cargo commands used `ROCR_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1`,
`nice -n 10`, and
`CARGO_TARGET_DIR=/home/kaden/qcal/release-0.4.3/target-hotset-rebudget`.
Logs are in `/home/kaden/qcal/release-0.4.3/hotset-rebudget-gates/`.

- `cargo test --release -p hipfire-arch-qwen4 --lib`: 864 passed, 2 ignored.
- `cargo test --release -p hipfire-loader --lib`: 148 passed.
- `cargo check --release -p hipfire-daemon`: passed.
- `scripts/pre-commit-ci.sh origin/beta`: passed, including workspace
  `cargo clippy --workspace --all-targets --locked`, env-docs, crate-maps,
  changelog, lifecycle, format, ratchets and dependency checks.
- Explicit env-docs and changelog checkers also passed. The first crate-map
  check found the new public exports missing from its generated inventory;
  regenerating the Qwen4 map resolved that failure. Final precommit passed
  with no source changes or test reruns required.

No GPU benchmark, model execution or fabric job was run by this worker.
