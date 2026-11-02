# Amendment 1 — pass-back mode A/B (current HEAD) — 2026-10-02

**Lifecycle:** `historical`

**Amends (does not modify):**
[`2026-10-02-offload-passback-mode-ab-gfx1201.md`](2026-10-02-offload-passback-mode-ab-gfx1201.md).
That file is unchanged, per [`README.md`](README.md).

Two corrections to a record that is superseded anyway by
[`2026-10-02-offload-passback-model-spread-gfx1201.md`](2026-10-02-offload-passback-model-spread-gfx1201.md)
(which re-ran the 9B at 5 rounds); this amendment fixes the original rather than
leaving the wrong reading in the ledger.

## 1. The outlier's cause was asserted, not supported

The original's Reading says the passback arm's wide range "is expected: it is the
only arm that uses both engines and so is the most contention-sensitive." That is
not supported by the record's own data. The outlier is passback's 30.5 in **round
1, position 2** (round 1 order was `cpu, passback, pcie`); the run *in the cold
slot* (round 1, position 0, `cpu`) read 28.5, in line with its other rounds
(28.4 / 28.8 / 29.0). So the dip does not line up with the post-`pkill` cold slot,
and the "contention-sensitive" attribution is withdrawn. A first-use warm-up of
the split path is a candidate, but **this record does not establish it** — no
causal claim is made.

## 2. Paired per-round statistic (the honest form of the result)

Medians alone hide the pairing. passback−cpu, per round, same round:

- **8 / 32 spilled**: +7.0 %, +17.2 %, +20.4 %, +20.1 % — **4/4 positive**, range
  +7.0 %..+20.4 %, median **+18.7 %**.
- **16 / 32 spilled**: +14.5 %, +15.0 %, +22.0 %, +20.5 % — **4/4 positive**,
  range +14.5 %..+22.0 %, median **+17.7 %**.

So the sign is unanimous and the magnitude spread is the real uncertainty; the
single-median "+19.0 % / +17.7 %" should be read as its central value.

## Superseded

The 5-round 9B spread in
[`2026-10-02-offload-passback-model-spread-gfx1201.md`](2026-10-02-offload-passback-model-spread-gfx1201.md)
replaces this record's numbers for current-HEAD figures (+19.6 / +20.6 / +21.6 %
at 8 / 16 / 24 of 32 spilled, 5/5 paired positive, passback's own spread tight).
This amendment exists to correct the record, not to keep it current.
