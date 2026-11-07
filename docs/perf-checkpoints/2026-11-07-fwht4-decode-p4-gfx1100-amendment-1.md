# Amendment 1 to `2026-11-07-fwht4-decode-p4-gfx1100.md`: lloyd4 verdict holds on the P1-included tree

**Lifecycle:** `historical`. Amends (does not modify)
`docs/perf-checkpoints/2026-11-07-fwht4-decode-p4-gfx1100.md`; the parent's
numbers and parked disposition stand unchanged.

## Why

The parent measured the GQA-shared decode tile against lloyd4 V on a tree
without P1 (FA2 WMMA prefill for lloyd V). Since P1 changes lloyd handling
on the prefill path, the lloyd4 verdict was re-checked with P1 present: a
throwaway merge of `feat/fa2-lloyd-v` @ `0fa66a69` into the parked
occupancy-v2 tree (branch `tmp/p1-gqa-lloyd`, deleted after measurement;
never a merge candidate).

## New evidence (RX 7900 XTX, HIP 7.16, mq4l fwht4/lloyd4, spec off,
noslots, GRAPH=0, TEMP env gate, md5-verified binary)

| Ctx | Scalar (gate off) | GQA-shared (gate on, kernel verified bound in-run) |
|---|---|---|
| Short (24 tokens), 3 fresh runs/side | 48.9 (sd 0.05) | 47.8 (sd 0.05), -2.2% |

An earlier ungated pair on the same merge measured 48.8 vs 49.0 and is
VOID (the parked tree carries no gate; both legs ran scalar — caught by
the missing kernel-bind line in the serve log, not by the numbers).

## Disposition

Unchanged: lloyd4 V GQA stays parked. The P1 tree changes nothing about
the decode-side verdict (same -2% as without P1, no overlap). The q8 ship
path (`docs/perf-checkpoints/2026-11-07-fwht4-gqa-q8-gfx1100.md`) is
unaffected. A prior void leg pair is recorded here so nobody re-runs it.