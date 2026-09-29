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
3. jev-bench, all 12 tasks and 6 experiments, at n=500 (aligned with Jev's committed rows):
   `python3 scripts/jev_eval/run_jevbench.py --model qwen3.5:4b --out bench/jev/qwen3.5-4b/jevbench all x-oos x-language x-options x-order x-repeat x-descriptions`
4. Calibration sets:
   `python3 scripts/jev_eval/run_calibration.py --model qwen3.5:4b --out bench/jev/qwen3.5-4b/calibration synth openbookqa commonsense_qa hellaswag`
5. Freeze the reported rows into text-free `{source, type, probs, gold}`
   files (spec §13.6), checked against the saved predictions:
   `python3 scripts/jev_eval/calibrate.py freeze --src bench/jev/qwen3.5-4b --out bench/jev/qwen3.5-4b/probs`
6. Report (raw vs calibrated, from the frozen rows and `calibration.json`
   if fitted): `python3 scripts/jev_eval/report.py --out bench/jev/qwen3.5-4b`

Not every frozen source is committed: a source dataset's own licence has to
clearly permit redistributing derived per-row data first — see
`bench/jev/DATA-LICENSES.md`. `report.py` and `calibrate.py fit` work from
whatever is present, committed or local, and simply skip a source that is
neither.

Jev's own latencies include the network and are not comparable. Compare
decide latency against the same model generating the answer via
`/v1/chat/completions` (see "Latency" in the PR description).
