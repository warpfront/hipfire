# MTP parity fixture and acceptance ratchet

Pinned prompts, a cell matrix, and a ratchet that fails when per-position MTP
acceptance, tokens per step, or window cost regress beyond pre-registered noise.
It compares hipfire EP2/PP2 MTP against another engine's `/stats draft_positions`
on **identical prompt bytes**, and it is the durability gate for MTP lanes.

- `fixture.json`: six prompts (story, code, prose_long, agentic, math,
  multilingual), md5 + size pinned, 28 cells (`<prompt>-C<n>-<t1|greedy>`).
  `story` and `code` are the core pair at C1/C4/C8; the four breadth classes run
  at C1/C8. A pass needs **every** gated cell of **every** class: acceptance
  that holds on two prompts and collapses on agentic is not durable.
- `ratchet.json`: the pre-registered `rule` (thresholds and the measured spread
  they come from) and one `baseline` per cell. A `null` baseline means no
  registration campaign has run yet; `check` then fails closed (exit 3) unless
  `--allow-unregistered`.
- `prompts/`: the four prompt files that did not already exist (byte-identical
  to the `diverse.json` texts the other engine runs) and the frozen agentic
  transcript. The two core prompts stay at `benchmarks/prompts/`.

Definitions match MtpTrain's `a_d` and the `/stats draft_positions` counters:
`reached_d` = windows that offered >= d rows and accepted the first d-1;
`a_d = accepted_d / reached_d`; `yield` = committed tokens per lane-window.

```
python3 scripts/check-mtp-parity-ratchet.py fixture                 # prompt bytes
python3 scripts/check-mtp-parity-ratchet.py extract-trace LOG --cell story-C1-t1 --build SHA > rec.json
python3 scripts/check-mtp-parity-ratchet.py from-radiance cell.json > rec.json
python3 scripts/mtp-parity-drive.py --base http://127.0.0.1:PORT --out cells/ --cells c1 --rep 0   # drive a running serve
python3 scripts/check-mtp-parity-ratchet.py from-driver cells/story-C1-t1-r0.json --trace serve.log > rec.json   # singleton (HIPFIRE_MTP_TRACE=1)
python3 scripts/check-mtp-parity-ratchet.py from-driver cells/story-C4-t1-r0.json > rec.json   # lanes: /stats draft_positions delta
python3 scripts/check-mtp-parity-ratchet.py register recs/          # >= 3 fresh-process reps per cell
python3 scripts/check-mtp-parity-ratchet.py check recs/             # exit 1 on regression
python3 scripts/check-mtp-parity-ratchet.py guard --base origin/beta
```

`guard` is the technical half of the ratchet, like `scripts/ratchet-diff.sh`: a
commit that lowers a baseline, widens a tolerance, or un-gates a cell must say
so with a `RATCHET-RAISE: <cell|rule> <reason>` line in its message.

`register` refuses a cell with yield cv > 0.15 across reps or fewer than 50 pooled reached trials at any depth
(a prompt that ends early cannot carry a baseline). Single-stream cells are the noisy ones: on the other engine's
fixture run, `prose_long-C1-t1` measured cv 0.16 at 3 reps, so C1 cells may need 5 reps.

Cells at C > 1 are `gate: false` (with a `pending` note) until native-MTP lanes are on beta: only the singleton
route has a per-position count today. `check` skips un-gated cells without a baseline and enabling a gate is always allowed.

A hipfire baseline on this fixture does not exist yet. The card job that
produces records is `HIPFIRE_MTP_TRACE=1` serve runs (singleton) or, for EP2/PP2
lanes, per-depth lane counters that the daemon does not export yet; see
`docs/perf-checkpoints/2026-11-01-mtp-parity-ratchet-registration.md`.
