# Data licences for `bench/jev/<model>/probs/`

`scripts/jev_eval/calibrate.py freeze` turns each reported row into
`{"source", "type", "probs", "gold"}` — the served probabilities and a label
*index*, no dataset text and no option keys (spec §13.6). Before any of
those per-source files are staged into git, this table asks whether the
*source dataset's own* licence clearly permits redistributing that kind of
derived data. Where it does, the file is committed gzip-compressed
(`<name>.jsonl.gz`). Where it does not, or the terms are unclear, the row
file is left on disk (produced by `freeze` from the read-only eval
worktree) but excluded from git by `.gitignore`; `report.py` and
`calibrate.py` simply do not find it in a fresh checkout and skip that
source (§ note in `report.md`).

This is a per-dataset judgement call, not a legal opinion. Re-check before
adding a new source.

## jev-bench tasks (`jevbench_<task>.jsonl`)

| Task | Dataset | Licence | Committed | Source |
|---|---|---|---|---|
| `banking77` | PolyAI Banking77 | CC BY 4.0 | yes | [HF dataset card](https://huggingface.co/datasets/PolyAI/banking77) |
| `clinc150` | CLINC150 | CC BY 3.0 | yes | [HF dataset card](https://huggingface.co/datasets/clinc/clinc_oos) |
| `massive-en`, `massive-it` | Amazon MASSIVE | CC BY 4.0 | yes | [HF dataset card](https://huggingface.co/datasets/AmazonScience/massive) |
| `ledgar` | LEDGAR (LexGLUE) | CC BY 4.0 | yes | [HF dataset card](https://huggingface.co/datasets/coastalcph/lex_glue) |
| `sms-spam` | UCI SMS Spam Collection | CC BY 4.0 | yes | [UCI ML Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) |
| `doc-yesno` | BoolQ | CC BY-SA 3.0 | yes, attributed here | [HF dataset card](https://huggingface.co/datasets/google/boolq) |
| `ag-news` | AG News (di.unipi.it corpus) | Custom, restricts use to "data mining... and any other **non-commercial** activity" | **no** | [original notice](http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html), quoted on [HF dataset card](https://huggingface.co/datasets/fancyzhx/ag_news) |
| `duplicates` | Quora Question Pairs (via GLUE QQP) | Quora/Kaggle terms: non-commercial, redistribution of raw examples discouraged | **no** (named explicitly in the human decision for this task) | [GLUE benchmark](https://gluebenchmark.com/), [Quora dataset release](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs) |
| `offensive` | TweetEval (OffensEval) | Listed "Undefined" on the dataset card; bound to Twitter's ToS, which restricts bulk redistribution of tweet-derived content | **no** | [HF dataset card](https://huggingface.co/datasets/cardiffnlp/tweet_eval) |
| `sentiment-it` | CardiffNLP tweet_sentiment_multilingual | CC BY 3.0 base, but the card requires compliance with Twitter/Twitter API ToS on top | **no** (Twitter ToS is the binding, more restrictive layer) | [HF dataset card](https://huggingface.co/datasets/cardiffnlp/tweet_sentiment_multilingual) |
| `yelp-stars` | Yelp Review Full | Yelp Dataset Terms of Use: non-commercial academic use only; explicitly forbids "redistribut[ing] summaries, metrics, or derived statistics" without Yelp's written consent | **no** (named explicitly in the human decision for this task) | [Yelp Dataset Terms of Use](https://s3-media0.fl.yelpcdn.com/assets/srv0/engineering_pages/f64cb2d3efcc/assets/vendor/Dataset_User_Agreement.pdf) |

## Calibration sets (`cal_<name>.jsonl`)

| Set | Dataset | Licence | Committed | Source |
|---|---|---|---|---|
| `synth` | jev-ood-calibration synthetic support tickets | CC0 1.0 (public domain) | yes | `data/LICENSE` in [scienthoon/jev-ood-calibration](https://github.com/scienthoon/jev-ood-calibration) (local read-only clone) |
| `openbookqa` | OpenBookQA | Apache License 2.0 (repo `LICENSE`) | yes | [allenai/OpenBookQA `LICENSE`](https://github.com/allenai/OpenBookQA/blob/main/LICENSE) (HF card lists "unknown"; the source repo's own licence file governs) |
| `commonsense_qa` | CommonsenseQA | MIT | yes | [HF dataset card](https://huggingface.co/datasets/tau/commonsense_qa) |
| `hellaswag` | HellaSwag | HF dataset card lists MIT, **but** the upstream GitHub repo (`rowanz/hellaswag`) is currently blocked under an active DMCA takedown | **no** | GitHub returns `{"message":"Repository access blocked","block":{"reason":"dmca", "html_url":"https://github.com/github/dmca/blob/master/2026/09/2026-09-14-wikihow.md"}}` for `rowanz/hellaswag` as of 2026-09-29. The notice (wikiHow, Inc., 2026-09-14) asserts that repositories including this one host "large-scale, unauthorized copies of [wikiHow's] copyrighted content, which were obtained by illicitly scraping [wikiHow's] websites" (HellaSwag partly derives its contexts from WikiHow). This is new since the HF licence tag was set and is a materially different, more restrictive situation than "MIT": withheld pending resolution. |

## `jev-bench` and `jev-ood-calibration`'s own repo licences

- `jev-bench` ([Running-Dolphins/jev-bench](https://github.com/Running-Dolphins/jev-bench), local read-only clone): MIT. Its own `predictions/*.jsonl` (label, top probability, correctness — no text) are explicitly "Committed" by that repo under its MIT licence.
- `jev-ood-calibration` ([scienthoon/jev-ood-calibration](https://github.com/scienthoon/jev-ood-calibration), local read-only clone): code MIT, synthetic data (`data/`) CC0 (see `synth` row above).
- **What this repo takes from them:** `report.py` reads their committed `predictions/*.jsonl` and `results/jev_*.jsonl` *locally, from the read-only clones*, only to compute a handful of aggregate numbers (n, accuracy, ECE) printed into `report.md` (the "Jev acc" / "Jev ECE" columns). None of their files, and no per-row Jev data, is copied into this repository. This is well inside their MIT terms even before considering that only derived summary statistics — not rows — are written here.

## What SOURCES.json is

`bench/jev/<model>/probs/SOURCES.json` (committed) lists, for every source
`freeze` processed (including the withheld ones), the file path in the
read-only eval worktree and a SHA-256 of its contents. A hash is not
reversible and a local path reveals no dataset content, so this manifest is
committed in full regardless of a source's licence outcome — it is
provenance metadata, not derived data.

## How to keep this honest

`.gitignore` excludes `bench/jev/*/probs/*.jsonl` (the plain form `freeze`
always writes locally). Only the `.jsonl.gz` for a licence-clear source is
ever `git add`ed, by explicit path — never a wildcard add of the `probs/`
directory. If a new source is added to `freeze`'s inputs, its rows stay
local-only (plain `.jsonl`, gitignored) until this table is updated with a
"yes" and a citation.
