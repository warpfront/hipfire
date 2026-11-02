# Amendment 1 — MTP 9-prompt Path-1 sweep (2026-10-04 record)

**Lifecycle:** `historical`
**Amends (without modifying):** `2026-10-04-mtp-9prompt-path1-sweep.md`
**Disposition:** correction. The original record stands; this file carries what re-measurement overturned.

## 1. The 63.0 K=1 cell is retracted

Two single-rep cells (61.1 @128tok merge_sort, 63.0 @512tok code_python, same md5 `3874f31c`) did not reproduce. Length-controlled re-run, gate-live binaries `7184efff`/`b392507e`, `BUDGET=20`, 2 fresh-process reps each:

| | max=128 | max=512 |
|---|---|---|
| AR | 60.7 | 59.4 |
| MTP K=1 (τ≈0.8) | 41.0 | 42.5 |

No length effect (AR flat, MTP flat). Replicable K=1 is **~42 tok/s (0.70×)**, not 61–63. Do not cite the 61–63 cells.

## 2. Aggregate accept (replaces the code_python cherry-pick)

Sweep-mean τ/K over all 9 prompts: **K=1 0.681, K=2 0.494, K=3 0.363** (n=9 each). Head is ~16% weaker than llama's at depth 1 too (.681 vs .812), so the engine gap quoted in the original §1 is an upper bound; head quality is an unresolved confound, not a cleared one.

## 3. Short-run caveat

Cells that EOS-terminated early (translation 17 tok, creative_short 34, summarize 47) never amortize spec warm-up (proposal graph, tape alloc; the 10-token warmup row reads ~33 tok/s on every cell). Their K-losses overstate the steady-state deficit. The length A/B above shows no 128-vs-512 effect on a full-length prompt, so the caveat is confined to early-EOS cells, but those cells should not anchor genre claims.

## 4. Standing result (replicable)

AR ~60; MTP K=1 ~42 (0.70×), K=2 ~41–57 (noisy), K=3 ~48 (0.80×). Path-1 gate stands as a ~2× loss-reduction at K=3 (25→48, interleaved, redline parity pass). MTP≡AR token identity unrun — neither route matched AR textually (0.63–0.93), so throughput numbers are on non-identical decodes until proven otherwise.
