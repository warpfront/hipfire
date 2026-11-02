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
| `doc-yesno` | BoolQ | CC BY-SA 3.0 (share-alike, not Apache-2.0 compatible) | **no** | [HF dataset card](https://huggingface.co/datasets/google/boolq) |
| `ag-news` | AG News (di.unipi.it corpus) | Custom, restricts use to "data mining... and any other **non-commercial** activity" | **no** | [original notice](http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html), quoted on [HF dataset card](https://huggingface.co/datasets/fancyzhx/ag_news) |
| `duplicates` | Quora Question Pairs (via GLUE QQP) | Quora/Kaggle terms: non-commercial, redistribution of raw examples discouraged | **no** (named explicitly in the human decision for this task) | [GLUE benchmark](https://gluebenchmark.com/), [Quora dataset release](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs) |
| `offensive` | TweetEval (OffensEval) | Listed "Undefined" on the dataset card; bound to Twitter's ToS, which restricts bulk redistribution of tweet-derived content | **no** | [HF dataset card](https://huggingface.co/datasets/cardiffnlp/tweet_eval) |
| `sentiment-it` | CardiffNLP tweet_sentiment_multilingual | CC BY 3.0 base, but the card requires compliance with Twitter/Twitter API ToS on top | **no** (Twitter ToS is the binding, more restrictive layer) | [HF dataset card](https://huggingface.co/datasets/cardiffnlp/tweet_sentiment_multilingual) |
| `yelp-stars` | Yelp Review Full | Yelp Dataset Terms of Use: non-commercial academic use only; explicitly forbids "redistribut[ing] summaries, metrics, or derived statistics" without Yelp's written consent | **no** (named explicitly in the human decision for this task) | [Yelp Dataset Terms of Use](https://s3-media0.fl.yelpcdn.com/assets/srv0/engineering_pages/f64cb2d3efcc/assets/vendor/Dataset_User_Agreement.pdf) |

### `doc-yesno` (BoolQ): withheld, CC BY-SA 3.0

BoolQ is CC BY-SA 3.0 (share-alike). Derived files under a share-alike
licence cannot be carried inside this Apache-2.0 repository, so none ship
here (maintainer decision, 2026-10-13, superseding the earlier decision to
keep them under BoolQ's own licence). `scripts/jev_eval/licences.py` lists
`doc-yesno` as restricted, so it is handled like every other withheld
source: `heldout.py run` skips it unless named with `--allow-restricted`,
`calibrate.py fit` refuses its rows, and `report.py` never prints it.
`noul` temperatures are therefore fitted on sms-spam and synth-noul only.

To evaluate on BoolQ locally, `heldout.py build` downloads it (through
jev-bench's dataset loader) from the [HF dataset](https://huggingface.co/datasets/google/boolq) (CC BY-SA 3.0,
attribution: Clark et al., "BoolQ: Exploring the Surprising Difficulty of
Natural Yes/No Questions", NAACL 2019) into `~/.cache/hipfire-jev-calib`;
any rows you derive from it stay local and carry BoolQ's licence.

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
- **What this repo takes from them:** the task definitions, prompts and
  example rows that the eval scripts read *locally, from the read-only
  clones*, to drive hipfire. No file from either repository, and none of their
  own results, is copied into this repository or published in `report.md`.

## What SOURCES.json is

`bench/jev/<model>/probs/SOURCES.json` (committed) lists, for every source
`freeze` processed (including the withheld ones), the file's path *relative
to `--src`* (the eval worktree's `bench/jev/<model>` directory, e.g.
`jevbench/raw/banking77.jsonl`) and a SHA-256 of its contents — never an
absolute path, so the manifest carries no local filesystem layout. A hash is
not reversible and a relative path reveals no dataset content, so this
manifest is committed in full regardless of a source's licence outcome — it
is provenance metadata, not derived data.

## How to keep this honest

`.gitignore` excludes `bench/jev/*/probs/*.jsonl` (the plain form `freeze`
always writes locally). Only the `.jsonl.gz` for a licence-clear source is
ever `git add`ed, by explicit path — never a wildcard add of the `probs/`
directory. If a new source is added to `freeze`'s inputs, its rows stay
local-only (plain `.jsonl`, gitignored) until this table is updated with a
"yes" and a citation.
