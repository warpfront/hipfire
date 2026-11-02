# Amendment 1 — MoE offload manual-vs-auto (2026-10-04 record)

**Lifecycle:** `historical`
**Amends (without modifying):**
[`2026-10-04-moe-offload-manual-vs-auto.md`](2026-10-04-moe-offload-manual-vs-auto.md)
**Disposition:** correction + retained-evidence supplement. The original
record is unchanged; this file carries the fixes.

## 1. Previously unretained refusal artifacts — now committed

The original record quotes frontier-boundary refusals whose `.err` files
were gitignored scratch at commit time. They are committed here, in
`data-2026-10-04-moe-offload-manual-vs-auto-amendment-1/` next to this file:

| quoted figure (original L60-62, L38-39) | artifact |
|---|---|
| N28: needs 13904 vs 13095 (809 over) | `probe_N28.err` |
| N34: needs 16289 vs 13095 (3194 over) | `q35_N34_pcie.err` |
| N40: needs 18827 vs 13067 (5760 over) | `probe_N40.err` |
| N0: 16.0 GiB pinned + 4 GiB headroom vs MemAvailable 16.8 GiB | `manual_moe_N0.err` |
| graded trunk resident-impossible, 18827 needed vs 13356 at `max_seq 2048` | `probe_resident_s2048.err` |

(`probe_N34.err`, same dir, is a second N34 probe reading 16365 vs 13095 —
3270 over — taken under a different KV-budget revision than the quoted
`q35_N34_pcie.err` line; the record's number is the `q35_N34_pcie.err` one.
Both retained so the discrepancy is auditable, not silently resolved.)

## 2. Llama-side anchor artifacts — now committed

`llama_probe_ngl0.err` (Vulkan device probe transcript) and
`llama_anchor.png` are committed in the same amendment data dir. The
fit-auto `(41, 19, ATTN)` tuple remains session observation with no
retained transcript, as the original record already states.

## 3. Method correction: N22 is noise, and runs are in-process repeats

`manual_pcie_N22.json` decode samples are `[56.3, 52.0, 48.7, 56.1, 53.6]`
(σ2.82) — the only row breaking the otherwise tight ladder (N26/N16/N10
spreads ≤0.4 tok/s). Do not derive a per-N scaling ratio or a "PCIe beats
CPU by X" figure from any pair involving N22; the cpu-exec verdict
(+19–26 tok/s at every spill) stands on the other four rows.

Further: each cell's 5 samples are repeated runs inside one bench process
(`--runs 5`), not ≥3 fresh-process runs per AGENTS.md §5. Treat every rate
in the original record as single-process evidence.

## 4. Coverage caveat: GPU parity is same-qt only, and on other trunks

Neither parity arm in the record exercises cross-qt (`gate_qt != down_qt`)
dispatch on hardware: the mq4p graded buckets are all same-qt pairs
(13/13 tag 2, 15/15 tag 0, 20/20 tag 3), and the qt44 arm is a fourth
same-qt point (qt44/qt44) on `ornith-1.5-35b-a3b.mq4` — not the benchmarked
`qwen3.5-35b-a3b.mq4` trunk. GPU-level cross-qt tag dispatch is covered
only by the `mixed_expert_tag` unit tests (tags 0–18, GL-reject, no
V2→V1 collapse). "Graded mixed-dtype offload" therefore means
per-expert-tier placement verified on hardware for same-qt tiers, with
cross-qt tag resolution verified in unit tests.

## 5. Dead plan link

The disposition links `../methodology/cpu-offload-head-to-head-plan.md`,
which is gitignored (`.gitignore` excludes that path) and absent in a clean
clone. The plan lives in the author's worktree only; the record's tables
and reproduction section are self-contained without it.
