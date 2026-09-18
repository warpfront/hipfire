# iu4port gate 5c — serve battery ON (ordinal 1, fresh daemon + iu4 flag)
# `python3 scripts/serve_harness.py --mode battery --model qwen3.8:27b-mq4-xt
# --tag qwen3.8:27b-mq4-xt --devices 0`, env IU4=1 (pass-through to daemon).
# Raw: bench/serve-on (JSON, 5 turns) + bench/serve-on.log.
# Daemon: wt-iu4/target/release/daemon (fresh build, new kernel embedded).

| turn | kind | finish | runaway/empty | text check |
|---|---|---|---|---|
| 1 | code | stop | no/no | correct merge_sorted implementation |
| 2 | reasoning | stop | no/no | correct arithmetic (150+60=210 miles) |
| 3 | factual | stop | no/no | correct (axial tilt causes seasons) |
| 4 | prose | stop | no/no | coherent story continuation |
| 5 | instruct | stop | no/no | sensible style guidelines |

5/5 healthy stop completions; short-prompt prefill 521-638 tok/s, decode 36.5 tok/s.
Verdict: PASS (no runaways, no empties, no corruption).
