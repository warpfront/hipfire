# Amendment 2: own-start simulator and final online-tuning policy

Lifecycle: **historical**. This amends the unchanged
[2026-10-05 record](2026-10-05-gfx1151-dflash-online-draft-tuning.md) and
[2026-10-06 amendment](2026-10-06-gfx1151-dflash-online-draft-tuning-amendment.md).
The route is still developer-only (`HIPFIRE_DFLASH_ONLINE_TUNE=on`) and off by default.

## Evaluation method that replaces fixed-start replay

`HIPFIRE_DFLASH_ONLINE_TUNE=sweep` drafts a token the target never picks. Every cycle
then commits one target token, and the dump holds the draft's top-K at every position of
the greedy text.

`examples/dflash_online_replay.rs --simulate` walks any policy over such a dump with its
own block starts: it accepts the longest matching prefix and starts the next block after
the bonus. This is what happens online, minus batched-verify near-tie text flips.

Validation:
- Simulated argmax windows equal shipping windows: q9 code_edit 80 = 80, and every
  session is within a few percent.
- For the earlier policy, simulated ratios were q9 ×1.024 and q8 ×1.088; online gave
  ×1.036 and ×1.095.
- Per-prompt residual between online and simulation: 2.8% SD.
- Both arms share one text, so the comparison carries none of the online A/B's
  text-divergence noise (about ±1% per 10-prompt pair).

## Policy changes selected with the simulator

Selection set: 10 `online_tune` prompts × 2 pairs.

| change | simulated tokens/window ratio |
|---|---|
| baseline | ×1.0554 |
| suffix-match injection threshold 4 → 1 | ×1.0631 |
| neighbour-row draft probabilities | ×1.0681 |
| learning rate 0.2 → 0.3 | ×1.0723 |
| one weight set for all row depths (no depth buckets) | ×1.0833 |

Fixed-start replay had rejected the first two and favoured depth buckets.

Rejected in simulation, each flat or worse: shift features, gap interactions, previous-block view,
off-policy rows, single and combined feature drops, history-window constants.

Held-out check (8 other committed prompts × 2 pairs, never used for selection):
×1.0102 → ×1.0186. The single weight set is the largest step; lr 0.3 alone is −0.13 pp
there.

## Online confirmation (final policy)

Method: 20 cold single-request runs (10 prompts × 2 pairs), tuned against cached shipping
DFlash, harness at `8b6cacde6`.

| pair | tokens/window | decode tok/s |
|---|---|---|
| Qwen3.5-9B + qwen35 DFlash | ×1.044 | ×1.072 |
| Qwen3-8B + generic DFlash | ×1.146 | ×1.139 |
| all | **×1.094** | **×1.105** |

The previous policy measured ×1.0655 tokens/window and ×1.073 tok/s. The simulator
predicted +2.6% for the change; online measured +2.7%.

## Warm daemon (weights carried across requests)

Simulated, each session after the other sessions of its set:

| set | pair | cold → carried |
|---|---|---|
| held-out | q9 | ×1.0095 → ×1.0254 |
| held-out | q8 | ×1.0278 → ×1.0385 |
| online_tune (long) | q9 | ×1.032 → ×1.036 |
| online_tune (long) | q8 | ×1.137 → ×1.133 |

Carrying the weights pays on short requests, which otherwise end before the learner warms up.
