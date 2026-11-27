### Fixes: engine_loop_ab G-spec

- **`scripts/engine_loop_ab.py` G-spec no longer fails on MTP tau noise or lane window policy.** On a `spec_noise` route the A/B G-spec is report-only, and the run is `pass-partial` unless a concurrency k=1 cell gates tau (bound − 2 %). Concurrency k>1 cells are not judged against the singleton `ref` arm.
- **New k>1 lane tau gate.** With `conc.lane_tau`, each k>1 engine cell must report tau on every request, and its mean over ≥ 3 runs must stay within − 15 % of a stored baseline of the accepted engine build. `--lane-tau-baseline` writes that baseline from `--conc` receipts. The Qwen4 MTP route and its baseline (`g12-poc` `f43ba4736`) are in `benchmarks/engine-loop-ab/`.
