# Decide (Jev-compatible) evaluation

External benchmarks are cloned outside the repo:

```bash
git clone https://github.com/Running-Dolphins/jev-bench ~/repos/jev-evals/jev-bench
git clone https://github.com/scienthoon/jev-ood-calibration ~/repos/jev-evals/jev-ood-calibration
```

1. GPU gates (must all PASS before merge). Build the daemon, CLI and the
   gate 1b noise-floor probe first:
   ```bash
   cargo build --release -p hipfire-daemon -p hipfire-cli
   cargo build --release -p hipfire-generate --features lab --example split_prefill_probe
   python3 scripts/jev_eval/gates.py --model ~/.hipfire/models/qwen3.5-4b.mq4
   python3 scripts/jev_eval/gates.py --model ~/.hipfire/models/qwen3.5-4b.mq4 --cask
   ```
   The same runs include the session-mode gates S1–S5 (spec §12.8). They
   need no extra flags; under `--cask` every session start is `cold` and S1
   is INCONCLUSIVE by design (the chat prompt cache is off under eviction).
   Gate S6 (session decide with a DFlash speculator loaded: read-only,
   keeps the assistant-turn cache, no leak) runs alone on a model with a
   drafter:
   ```bash
   python3 scripts/jev_eval/gates.py --model ~/.hipfire/models/qwen3.6-27b.mq4 \
     --draft ~/.hipfire/models/qwen36-27b-dflash-mq4.hfq --leak-n 120
   ```
2. Start serve on the repo daemon (one daemon per machine):
   `HIPFIRE_DAEMON_BIN=$PWD/target/release/daemon target/release/hipfire serve`

   **Hard requirement for every step below that collects answers (3, 4, 7's
   `heldout.py run`): the model must be raw — no `decide.calibration.*`
   configured for it (`docs/CONFIG.md`).** These are the answers the fit and
   the report are built from; a calibrated answer collected here would feed
   already-calibrated numbers back into both. `run_jevbench.py`,
   `run_calibration.py` and `heldout.py run` all enforce this themselves:
   each checks the `x-hipfire-timing` response header (`raw_guard.py`) and
   exits with an error the moment it sees a `calibration` entry there, so a
   misconfigured model fails loudly on the first request rather than
   quietly poisoning the run.
3. jev-bench, all 12 tasks and 6 experiments, at n=500 (aligned with Jev's committed rows):
   `python3 scripts/jev_eval/run_jevbench.py --model qwen3.5:4b --out bench/jev/qwen3.5-4b/jevbench all x-oos x-language x-options x-order x-repeat x-descriptions`
4. Calibration sets:
   `python3 scripts/jev_eval/run_calibration.py --model qwen3.5:4b --out bench/jev/qwen3.5-4b/calibration synth openbookqa commonsense_qa hellaswag`
5. Freeze the reported rows into text-free `{source, type, probs, gold}`
   files (spec §13.6), checked against the saved predictions:
   `python3 scripts/jev_eval/calibrate.py freeze --src bench/jev/qwen3.5-4b --out bench/jev/qwen3.5-4b/probs`

   `freeze` always writes the plain `.jsonl` form. Only a licence-clear
   source (`bench/jev/DATA-LICENSES.md`) gets committed, and only its
   gzip-compressed form, by explicit path — never a wildcard add of
   `probs/` or `heldout/`:
   ```bash
   gzip -nk bench/jev/qwen3.5-4b/probs/jevbench_banking77.jsonl   # -n: no name/mtime, reproducible; -k: keep the plain .jsonl too
   git add -f bench/jev/qwen3.5-4b/probs/jevbench_banking77.jsonl.gz
   ```
6. Report (raw vs calibrated, from the frozen rows and `calibration.json`
   if fitted): `python3 scripts/jev_eval/report.py --out bench/jev/qwen3.5-4b`
7. Calibration (spec §13; CPU unless noted). Held-out rows never overlap
   the reported rows, and the fit never reads reported rows.
   ```bash
   # once: freeze the reported rows (text-free) from the eval run's raw answers
   python3 scripts/jev_eval/calibrate.py freeze --src ~/repos/hipfire-jev-eval/bench/jev/qwen3.5-4b --out bench/jev/qwen3.5-4b/probs
   # once: build the held-out set (network); text stays in ~/.cache/hipfire-jev-calib
   python3 scripts/jev_eval/heldout.py build && python3 scripts/jev_eval/heldout.py verify
   # GPU: answer it on the build that served the reported rows (serve on :11435; raw is a hard requirement, see step 2)
   python3 scripts/jev_eval/heldout.py run --model ~/.hipfire/models/qwen3.5-4b.mq4 --out bench/jev/qwen3.5-4b/heldout
   # fit per question type, then report raw vs calibrated on the reported rows
   python3 scripts/jev_eval/calibrate.py fit --heldout bench/jev/qwen3.5-4b/heldout --model-id qwen3.5-4b.mq4 --build <sha> --out bench/jev/qwen3.5-4b/calibration.json
   python3 scripts/jev_eval/report.py --out bench/jev/qwen3.5-4b
   ```
   Committing a `heldout/<source>.jsonl.gz` follows the same `gzip -nk` +
   explicit-path `git add -f` step as `probs/` above (step 5).
   The fit prints the `models.toml` snippet (`decide.calibration.*`; see
   `docs/CONFIG.md`). GPU check that the daemon applies T exactly as the
   fitter does: `python3 scripts/jev_eval/calib_live_check.py --model ~/.hipfire/models/qwen3.5-4b.mq4`.

Not every frozen source is committed: a source dataset's own licence has to
clearly permit redistributing derived per-row data first — see
`bench/jev/DATA-LICENSES.md`. `report.py` and `calibrate.py fit` work from
whatever is present, committed or local, and simply skip a source that is
neither.

Calibration temperatures are fitted on licence-clear held-out sources only
(human decision). `licences.py` holds the one restricted-source list
(ag-news, duplicates, yelp-stars, offensive, sentiment-it, hellaswag):
`heldout.py run` skips those by default (naming one needs
`--allow-restricted`), and `calibrate.py fit` refuses any row from them.
Score-type T therefore comes from synth-score alone.

Jev's own latencies include the network and are not comparable. Compare
decide latency against the same model generating the answer via
`/v1/chat/completions` (see "Latency" in the PR description).
