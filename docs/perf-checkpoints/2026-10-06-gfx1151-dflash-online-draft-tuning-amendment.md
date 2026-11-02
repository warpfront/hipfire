# Amendment: online DFlash draft tuning, second pair and replay correction

Lifecycle: **historical**. This amends the unchanged
[2026-10-05 record](2026-10-05-gfx1151-dflash-online-draft-tuning.md). The route is still
developer-only (`HIPFIRE_DFLASH_ONLINE_TUNE=on`) and off by default.

## Corrections to the original record

1. **Replay overstates changes that extend acceptance.** The fixed-start replay
   (`examples/dflash_online_replay.rs`) scores a policy at the block starts of the
   recorded argmax session. A policy that accepts more moves later block starts onto
   harder positions, and replay cannot see that.
   - Example (Qwen3-8B pair, code_inventory): the same policy scores τ 1.89 at the
     argmax session's starts but 1.43 at its own online starts. At those online starts,
     argmax alone would score 1.08 rather than 1.39.
   - Replaying a policy over its *own* online dump reproduces online τ exactly
     (picked τ 1.479 / 0.482 both ways), so the replay code is correct; only its
     evaluation premise is biased.
   - The replay "carry leave-one-out ×1.062" and "first 32 cycles ×1.058" figures in the
     original record are therefore upper bounds, not online estimates.
   - The original record's online numbers (36 paired runs: τ ×1.049, decode ×1.034) stand.
2. **Two changes rejected online.** An injection threshold of 1 and neighbour-row draft
   probabilities scored +5% in replay. A 20-prompt online A/B (both arms tuned) showed
   tokens/window ×0.989 on Qwen3-8B and ×0.996 on Qwen3.5-9B. Both were reverted.
3. **A held-out-fitted weight prior** scored +8% in replay on Qwen3-8B (cold). Online over
   20 runs it gave ×1.0655 → ×1.0698, within noise. Not kept.

## Second pair: Qwen3-8B, generic DFlash (weak draft)

Fixture:
- Target `qwen3-8b.mq4`; draft `qwen3-8b-dflash.hfq` (not registry-managed). Both are
  set through an isolated `HIPFIRE_HOME` whose config sets `developer.dflash_draft` and
  `speculation.dflash = "on"`.
- Shipping DFlash on this pair: prose τ 0.40–0.55, ~10–12 tok/s; code τ 1.2–1.4,
  ~17 tok/s; AR 46.5 tok/s. DFlash loses to AR here with or without tuning.

Shipping defect found and fixed (`cb0f1f5ae`):
- Generic DFlash panicked the daemon (`dflash_generic.rs:473`) whenever a request's
  remaining budget made the verify block 1–3 rows.
- Cause: the llama verify per-token fallback captured no hidden rows.
- Fix: it now uses the per-token capture forward.
- Repro: `hipfire bench --spec dflash --max-tokens 64` on this pair panicked before the
  fix and completes after.

The tuner is wired into the generic chain. Its drafts are deterministic, so both the greedy
accept and the temp>0 naive-sampling verify stay exact.

## Online results (final policy, build `8b6cacde6` + `cb0f1f5ae`)

Method:
- Tokens per verify window, tuned vs shipping DFlash (tuner unset).
- 10 prompts (`benchmarks/prompts/online_tune/`) × 2 pairs; one cold request per fresh
  daemon.
- Greedy, so the figure is deterministic per build (run #487 = run #488 exactly).
- The shipping arm is cached; tok/s therefore compares runs taken at different times.

| pair | tokens/window | decode tok/s |
|---|---|---|
| Qwen3.5-9B (strong draft) | ×1.036 (code ×1.05–1.09, prose ×0.99–1.04) | ×1.042 |
| Qwen3-8B (weak draft) | ×1.095 (code ×1.09–1.21, mixed ×1.17, prose ×1.03–1.12) | ×1.104 |
| all 20 | ×1.066 | ×1.073 |

An earlier 6-prompt paired run on the Qwen3-8B pair (shipping arm measured alongside)
showed τ ×1.237 and decode ×1.109, with all 6 pairs faster.
