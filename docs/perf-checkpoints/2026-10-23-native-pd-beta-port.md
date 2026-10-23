# Native Prefill/Decode port onto the newer beta

## Integration decisions

- Preserve beta's K640 R4 production dispatch and G1 handling. The earlier
  literal-K640 down selector is not installed over that dispatch. Its retained
  explicit API is a scoped validation/benchmark surface, not a selected product
  route in this port.
- Preserve beta's fixed-eight and wide MQ6 routes, grouped-position down image
  selection, runtime `k_top` ABI, and existing route planner/certification logic.
- Retain narrow native MQ6 PM capture contracts, the small-batch inverse GU
  selector, streamed prefill GU, scalar MQ6 specializations, and prepared-bank
  completion reuse where their admission predicates apply.
- Keep gfx1201-only HC x8 out of gfx11 registry entries. Keep the existing
  gfx1201 implementation and its fixed module-count guard.
- Adapt the stream8 emitter to beta's GP/blocks interface. The source/image
  freshness test passes after this merge; the committed image needs no change.

## Evidence boundaries

The adjacent October 21 checkpoints and compressed receipts describe their
original experimental revisions. They are historical evidence, not performance
or correctness certification of this newer-beta port.

An independently built comparison against the older pinned beta observed:

| NPU | Decode mode | Prefill A → B (tok/s) | Decode A → B (tok/s) |
|---|---|---:|---:|
| auto | AR | 2229.4 → 2242.8 | 34.2 → 34.3 |
| auto | MTP | 2186.0 → 2202.9 | 52.9 → 54.2 |
| off | AR | 2029.2 → 2033.2 | 34.1 → 34.2 |
| off | MTP | 1995.9 → 2007.3 | 54.1 → 54.3 |

These are four single A/B pairs, after eight excluded prime processes, using
matched default-product feature/profile multisets. They do not establish a
repeatable improvement or a gain over this newer beta. Full source/product
identities, commands, outputs and analysis live in the external review bundle:

`/home/husrcf/Code/hipfireQC/tmpOutput/hipfire-native-pd-pr-20261022/`

The older candidate passed all 17 required CPU checks and 13 of 14 hardware
checks. Its remaining strict AR/MTP Seasons identity check failed identically
on the older baseline: all five corresponding A/B routes produced equal
256-token ID sequences. The failed cross-route check remains a failure.

The 130 thinking-on QC requests used the pre-repair candidate. All 65 A/B
output/reasoning hashes and token counts matched; LongBench scored 17/30 for
each AR/MTP arm. LongDecode has no gold scorer and semantic review remains
pending. Those QC results are not relabeled as final-port validation.

## New-port validation status

Eight scoped static checks and stream8 source/image freshness passed during
integration. Fresh, separate A/B default products, feature/profile comparison,
and the candidate's PR-required CPU checks are the next validation stage.
New-port hardware, thinking-on QC and formal AB results are pending. No claim
of complete validation or merge readiness is made here.
