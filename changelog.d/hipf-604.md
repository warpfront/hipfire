### Fixes: engine_loop_ab G-spec

- **`scripts/engine_loop_ab.py` G-spec no longer fails on MTP tau noise or lane window policy.** On a `spec_noise` route the A/B G-spec is report-only, and the run is `pass-partial` unless a concurrency k=1 cell gates tau (bound − 2 %). Concurrency k>1 cells report tau (both means and n) without gating it; the k>1 throughput/TTFT targets gate a draft loss.
