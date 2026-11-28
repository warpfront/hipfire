### Fixes: pre-commit hook skips data files

- **Fixed: `.githooks/pre-commit` no longer starts the GPU gates for data files
  (HIPF-630).** The `HOTSPOT` query now excludes `benchmarks/` and `*.json` with
  Git pathspecs, so quoted non-ASCII names also skip the gates. A commit of only
  `benchmarks/engine-loop-ab/*mq4*.json` exits before the speed gate and the
  agentic gate. `scripts/test-precommit-hotspot.sh` tests the routing.
