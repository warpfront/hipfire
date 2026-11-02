# Decide evaluation results

Accuracy and calibration of `POST /v1/systemone` (spec
`docs/specs/2026-09-28-jev-decide-design.md`) on the jev-bench tasks and
jev-ood-calibration sets. Only hipfire's own numbers are published here.
Harness and instructions: `scripts/jev_eval/README.md`.

## Runs (2026-09-28, STARLING, Strix Halo gfx1151, ROCm 7.2.2)

| Directory | Model | Coverage |
|---|---|---|
| `qwen3.5-4b/` | `qwen3.5-4b.mq4` | Short run: jev-bench banking77 (n=500) plus the 900 synthetic tickets |
| `qwen3.8-27b-mq4-xts/` | `qwen3.8-27b.mq4-xts` | Full run: all 12 jev-bench tasks, 6 experiments and 4 calibration sets were run; published here: 6 tasks and 3 sets (the rest withheld — see Licences; experiments not frozen) |

### Build

Both runs used a throwaway local build (not pushed): `feat/jev-decide` at
the decide v1 fixes (f52015942), merged with warpfront/hipfire PR #768
`mq4-lloyd` (head fde062757) for its faster gfx1151 prefill. The merged build
was `eval/jev-decide-mq4-lloyd` at 237bb7bba. Serve ran with the user's
normal config: KV q8, CASK on with budget 16384. Every request carried one
question, so the answers come from one plain prefill per question.

## Licences

Results for seven datasets are withheld from this repo because their licences
restrict redistribution of derived statistics, are currently unclear, or are
not compatible with the repository's Apache-2.0 licence: AG News, Quora
Question Pairs (`duplicates`), Yelp Review Full (`yelp-stars`), TweetEval
offensive (`offensive`), CardiffNLP Italian tweet sentiment (`sentiment-it`),
HellaSwag (`hellaswag`; source repo DMCA-blocked) and BoolQ (`doc-yesno`;
CC BY-SA 3.0 share-alike). See `DATA-LICENSES.md` for the per-dataset review.

## Latency: decide vs generate (`latency/`)

These timings compare a decide (`POST /v1/systemone`, one question) with the
same model answering the same question through `/v1/chat/completions`
(greedy, thinking off, `max_tokens` 12, prompt asks for the option name).
Both use the same examples and the same build. Requests were sequential,
with one warm-up request excluded. Figures are wall-clock at the HTTP
client, in ms (script: `scripts/jev_eval/latency.py`).

| Model | Task (n) | decide p50 / p95 | generate p50 / p95 |
|---|---|---:|---:|
| qwen3.5-4b.mq4 | banking77 (100) | 348 / 371 | 543 / 586 |
| qwen3.8-27b.mq4-xts | banking77 (100) | 1106 / 1127 | 1451 / 1769 |

Decide is 1.3–1.6× faster than generating the answer at p50 on banking77, and
it has a much tighter tail, because it has no decode loop and its answer
length doesn't vary.

## Metrics

- **Accuracy:** share of examples where the top-probability answer matches
  the gold label.
- **ECE:** top-probability expected calibration error, 10 equal-width bins.
  Computed by `report.py`, identically to jev-bench's `jevbench.py` `ece()`.

## Caveats

- The public QA sets (OpenBookQA, CommonsenseQA, HellaSwag) are likely in
  both models' training data.
- On the merged PR #768 build, Qwen3.8-27B answers vary by up to about
  1 log-prob depending on how the prompt is split across prefill calls. That
  build's default-on gfx1151 prefill routes do not give the same result
  regardless of batching (spec §10, gate 1a note). The numbers above are
  what that build serves.
- Calibration (spec §13): each report's "Fit" table gives the per-type
  temperatures fitted on held-out rows only (`heldout/`, answered by the
  same eval build; `../heldout/manifest.json` lists the sources and proves
  no overlap). The "ECE cal" / "NLL cal" columns apply them to the saved
  reported answers (`probs/`); nothing was re-run. The raw columns are the
  served probabilities.
  - Temperatures are fitted on licence-clear held-out sources only
    (`scripts/jev_eval/licences.py`; see Licences above and
    `scripts/jev_eval/README.md`). Every restricted `score` source
    (yelp-stars, sentiment-it) is withheld from the fit, so `score` T comes
    from synth-score alone — one source, not a pool across several. BoolQ
    (`doc-yesno`) is withheld too, so `noul` T is fitted on sms-spam and
    synth-noul.
  - The bootstrap resamples *rows*, not sources (`calibrate.py
    bootstrap_ci`). A pooled type's rows come from several sources —
    2,400 `choice` rows from 8 sources, 600 `noul` rows from 2 sources
    (`score` is a single source, synth-score, and is not pooled; spec §13
    "Licence-clear fitting") — and a resample can still draw
    disproportionately from one of them; the reported 95% interval for T is
    therefore narrower than the true cross-domain uncertainty, which
    row-level resampling cannot capture.
  - No bar (spec §13.6 "No bar"): a shipped T can make a set's ECE worse
    than raw, and that is reported as found rather than refitted. As
    measured here: qwen3.8-27b.mq4-xts's clinc150 ECE goes 0.044 (raw) →
    0.074 (calibrated) and its openbookqa ECE 0.015 → 0.046; qwen3.5-4b.mq4's
    synth-score ECE goes 0.091 → 0.125. All three still ship under §13.4's
    rule, which is scored on NLL, not ECE.

## Applying calibration

`calibrate.py fit` printed these `models.toml` snippets (`docs/CONFIG.md`
has the general key shape and range). **Neither is written to
`~/.hipfire/models.toml` by anything in this branch** — applying one is a
separate, human step, because a hipfire binary built before
`decide.calibration.*` existed refuses to load a catalog that contains it
(spec §13.9). Paste a snippet only once every hipfire that reads that
catalog — the installed `~/.hipfire/bin` included — is at or past this
branch.

```toml
[models."qwen3.5-4b.mq4".overrides.decide.calibration]
choice = 1.653
score = 1.48
```

```toml
[models."qwen3.8-27b.mq4-xts".overrides.decide.calibration]
choice = 1.391
```

`qwen3.5-4b.mq4`'s `noul` and `qwen3.8-27b.mq4-xts`'s `score` and `noul` did
not clear the ship rule (§13.4: held-out NLL improves ≥ 1% relative, and the
bootstrap 95% interval for T excludes 1) on licence-clear held-out data, so
they stay at T = 1 and are omitted from the snippets; see each model's
`calibration.json` and `report.md` "Fit" table for the numbers.
