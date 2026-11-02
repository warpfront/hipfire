# Amendment 1 to `2026-10-11-cpu-mtp-extended-16cell-narrow.md` — 2026-10-11

**Lifecycle:** `historical`
Amends `2026-10-11-cpu-mtp-extended-16cell-narrow.md` (the original is unchanged). Corrects unproved claims and records math-prompt output-identity evidence.

## Corrections (original line numbers refer to the amended file)

- **Backend inversion attribution (§ Backend change, § Bottom line).** The sentence attributing the CPU-beats-PCIe inversion causally to the day's CPU optimization stack (async copies, SIMD, Rayon) is withdrawn as a causal claim: no isolated A/B of those optimizations was run in this session. Mark it **[INFERENCE]**: the inversion is measured (8/8 cells, medians over 5 fresh processes each), the optimization stack is the plausible context, but credit to any specific optimization is unproved. The `invalidate_fp16_cache` correctness fix changes no arithmetic and is not a speed explanation.
- **"All of it acceptance" (prompt-change section).** The claim that the merge→math MTP drop is entirely an acceptance effect is unproved as a timing cause: acceptance fell (82→68%) and throughput fell (−3 to −9%), but no per-step timing decomposition isolates the two. Mark **[INFERENCE]**.
- **"1-LSB close-argmax flip" (acceptance-rate note).** The mechanism for the τ backend gap (a 1-LSB logit flip turning an accept into a reject) was never observed — no per-row logits were dumped on the live path. It is a hypothesis, not a finding. The τ gap itself (merge 2.46 vs 2.39, math 2.04 vs 2.00, all-5 flat per prompt×backend) is measured. τ differences across backends sit under the existing float numerical contract (CPU vs GPU results need not be byte-equal); the required contract is same-backend AR/MTP parity (see § Identity below).
- **No acceptance-criteria recommendation.** The original's "future work" paragraph on relaxing the exact-argmax accept rule is withdrawn in full: it would violate the user's bit-identity constraint. The exact reject/argmax rules are preserved; no tuning is proposed or performed.
- **MTP-vs-AR scope (spec-change section).** Read parity pairs as PCIe +13.9/+22.4% and CPU +33.4/+43.8%, honest (auto) pairs as PCIe +10.6/+18.2% and CPU +27.7/+41.3%. The Bottom line's "+14–22% pcie" shorthand is corrected to these per-prompt figures; no all-8 single range is claimed.
- **Source identity (header).** Code commit `099193aa3` ("fix(qwen35): restore MTP with CPU-offloaded MoE experts", 4 files, +6/−33, no GPU launchers changed) does **not** carry the entire measured source: the pre-existing dirty-tree CPU optimizations (async offload returns, six-bit AVX2 unpack, affine-sum reuse, nested Rayon; see handoff `.codeinsight+research/CPU-MOE-OFFLOAD-HANDOFF.md`) are excluded from that commit and remain uncommitted working-tree state. Measured source = `f20a1f230cc8` + gate deletions + `invalidate_fp16_cache()` ×2 + that dirty optimization set. Binaries as pinned (CLI `1420653877697b0baab7e726d387f92d`, daemon `fd635fbfe94ed68fb02815d6ace11a25`).

## Math-prompt output identity (new evidence, same final binaries)

`hipfire bench` emits no decoded text, so identity was checked with `hipfire run --stats` smokes at the exact suite params (budget24, q8/vmm, max_seq 4096 via scoped config, K3, thinking off via scoped config, greedy `-t 0`, pinned math prompt md5 `e527794265ce421b9cf4bc7cff631490`), `-n 256`, `-j`. All four arms hit `finish:length` at the 256 cap (the worked math answer exceeds it) — comparison is over the 256-token prefix, which is the valid identity signal (any divergence inside the cap is real; truncation is identical across arms).

| Pair | Result | Receipt |
| ---- | ------ | ------- |
| CPU AR vs CPU MTP (math) | **byte-identical: True** | `mathcheck/ar-cpu.json` vs `mtp-cpu.json` |
| CPU AR vs PCIe AR (math) | False (expected under float contract) | same dir |
| PCIe AR vs PCIe MTP (math) | **False — reproducible divergence**: first diff at char 127 (widespread wording-level, ratio 0.68 over ~900 chars); rerun pair r2 reproduces both sides exactly (AR r1==r2 True, MTP r1==r2 True) while AR-r2==MTP-r2 stays False | `mathcheck/{ar-pcie,mtp-pcie,ar-pcie-r2,mtp-pcie-r2}.json/.stderr` |
| CPU AR vs CPU MTP (merge, prior) | True (166 tokens, `finish:stop`) | `narrowcheck/narrow-mtp.json` vs `smoke-nothink-ar.json` |

MTP acceptance on the math smokes: CPU τ=2.00, PCIe τ=2.04 — matching the bench cells. Raw: `.codeinsight+research/cpu-mtp-smoke-20261010-gatefree/mathcheck/` (4 + 2 rerun JSON/stderr pairs).

**Reading:** the required same-backend AR/MTP parity holds on CPU for both prompts and on PCIe for merge, but **fails deterministically on PCIe math** — a genuine, reproducible same-backend divergence on the PCIe arm only, at matched K3/greedy/thinking-off settings. It is not length-truncation (divergence at char 127 of ~900), not sampling noise (both sides self-reproduce), and not the CPU fix (PCIe path untouched by it). Scope: one prompt, one backend; merge-strict byte-identity holds 2/2 backends, math holds 1/2. No root cause is claimed here; the exact-argmax accept rule means any verify-logit row difference surfaces as a committed-token difference, so this bounds the PCIe verify path, not the draft path. Flagged for parent triage — not a bench-data invalidation (throughput cells stand as measured), but a parity gap that must close before any PCIe-math MTP output is treated as AR-equivalent.

No git operations performed for this amendment (per instruction). No tuning, no rebuild, no suite rerun.
