# MTP 9-prompt sweep: narrow-verify Path 1 gate + hipfire-vs-llama chart — 2026-10-04

**Lifecycle:** `historical`
**Branch:** `feature/qwen35-moe-offload`, commits `7112ec1e2` (phase instrument) + `d6b22bc2d` (narrow-verify Path-1 gate).
**Binaries:** `hipfire` md5 `7184efff`, daemon md5 `b392507e`.
**Host:** 1× RX 9070 XT `gfx1201` (16 GiB), HIP 7.2. Load 3–6 during sweep (shared desktop).
**Fixture:** `ornith-1.5-35b-a3b.mq4` (md5 `f6cb95300d29c0c4c10d017180c12ab8`) + `.mtp` (`0cf41c5a36430c3570c509643db102b5`), `HIPFIRE_MOE_EXPERT_BUDGET=20`, `HIPFIRE_GRAPH=0`, `--kv-mode q8`, `--backend noslots --workload stateless`.
**Prompts:** `benchmarks/prompts/mtp-bench/*.txt` (byte-identical to `mtp-bench.py` literals):

| file | md5 (8) |
|---|---|
| code_python.txt | 3874f31c |
| code_cpp.txt | 194e470a |
| explain_concept.txt | dca47b0d |
| summarize.txt | c7c9abe9 |
| qa_factual.txt | fcee0d10 |
| translation.txt | 9564b5d9 |
| creative_short.txt | e8467086 |
| stepwise_math.txt | e5277942 |
| long_code_review.txt | 0b86c16d |

**Method:** 9 prompts × (AR + MTP K=1/2/3) × 1 rep, `--runs 1 --warmups 1 --max-tokens 512`, one fresh process per cell (36 loads). Single-rep: τ/K points are noisy (±0.05 at ~70 windows). Raw `--json` + `.err` (τ lines, phase lines) in `data-mtp-9prompt-sweep/` next to this record.
**Caveats (do not compare past these):** llama cells come from `mtp-bench.py` over `/v1/chat/completions` (chat-templated, content-matched not byte-identical; sampling unresolved server-side vs greedy hipfire; 3 server launches for K=1/2/3); hipfire is greedy argmax-match. Tok/s only as ratio-to-own-AR; accept only as τ/K vs draft fraction.

## Result (decode tok/s median, τ, τ/K, ×AR)

| prompt | AR | K1 / τ/K / ×AR | K2 / τ/K / ×AR | K3 / τ/K / ×AR |
|---|---|---|---|---|
| code_python | 57.6 | 41.6 .83 .72 | 50.4 .59 .88 | 41.6 .46 .72 |
| code_cpp | 59.7 | 40.9 .82 .69 | 40.7 .61 .68 | 37.6 .46 .63 |
| explain_concept | 59.4 | 38.3 .66 .64 | 35.5 .44 .60 | 31.5 .33 .53 |
| summarize | 58.0 | 40.1 .77 .69 | 39.6 .60 .68 | 35.6 .47 .61 |
| qa_factual | 59.9 | 39.7 .71 .66 | 40.3 .56 .67 | 35.8 .41 .60 |
| translation | 59.8 | 31.6 .55 .53 | 31.1 .30 .52 | 27.6 .20 .46 |
| creative_short | 60.8 | 32.4 .42 .53 | 35.1 .38 .58 | 29.6 .25 .49 |
| stepwise_math | 60.4 | 42.2 .81 .70 | 42.9 .62 .71 | 38.5 .46 .64 |
| long_code_review | 60.4 | 35.5 .56 .59 | 32.9 .37 .54 | 28.0 .24 .46 |

llama-server, same box, Qwen3.6-35B-A3B UD-Q4_K_M GGUF, K=1/2/3 across 3 launches: accept .812/.714/.630, tok/s ~50/~56/~52 vs AR ~44.5 (1.12×/1.26×/1.17×).

## Reading

1. **K=1 isolates engine cost:** code_python accept .83 vs llama .812 (matched), yet 0.72×AR vs ~1.1×AR. ~40% gap is verify row cost, not the head.
2. **Depth decay is ours:** hipfire τ/K 0.90→0.68→0.45 (merge_sort; sweep medians similar) vs llama .81→.71→.63. Depth-1 runs the same 16K compressed sidecar at 0.90, so the ranking head is cleared; the decay lives in the chained `t_mtp_out` input at k≥1 (structural: no trunk hidden exists pre-verify), not a pairing bug.
3. **No hipfire cell wins.** Best is K=2 stepwise_math 0.71×. The narrow-verify Path-1 gate (verify 84→44ms at K=3) halved the worst loss; the rest is rows/token × row cost against τ the lossy chain can't supply.
4. **Head lineage:** ornith-1.5 is a Qwen3.6-35B-A3B fine-tune (same family as llama's trunk); the `.mtp` ships `has_compressed_lm_head_draft: true`, cvs 16384, so K>1 drafts chain lossily. `use_full_vocab` is derived (sidecar present ⇒ lossy), not a knob.
