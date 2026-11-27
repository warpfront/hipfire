### Fixes: engine_loop_ab G-spec

- **`scripts/engine_loop_ab.py` G-spec no longer fails on MTP tau noise or lane window policy.** On a `spec_noise` route the A/B G-spec is report-only; the concurrency k=1 cells gate tau. Concurrency k>1 cells report tau (both means and n) without gating it; the k>1 throughput/TTFT targets gate a draft loss.
