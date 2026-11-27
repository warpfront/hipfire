### Fixes: pre-commit hook skips data files

- **Fixed: `.githooks/pre-commit` no longer starts the GPU gates for data files
  (HIPF-630).** The `HOTSPOT` match now ignores `benchmarks/` and `*.json`. A
  commit of only `benchmarks/engine-loop-ab/*mq4*.json` exits before the speed
  gate and the agentic gate.
