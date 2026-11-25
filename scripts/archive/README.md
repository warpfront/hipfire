# scripts/archive

Unmaintained lab scripts, kept for history.

These are one-off benches, calibration and quant-quality campaigns, rental-pod
helpers, training experiments and debug harnesses that nothing in the tree calls
any more (the D9.0 classification, signed off 2026-11-25). They moved here from
`scripts/` with their relative layout intact (`scripts/qat/x.py` is now
`scripts/archive/qat/x.py`).

- Nothing here is run by CI, the hooks or a release workflow, and none of it is
  tested. It may not run against the current tree.
- Repo-root and `_detect-gpu.sh` lookups were re-pointed one level deeper so the
  scripts resolve the same files as before; nothing else was changed.
- Released CHANGELOG sections and `CREDITS.md` still name the old `scripts/<name>`
  paths; they are immutable records.
- To revive a script, move it back out of `archive/` with `git mv` and restore its
  caller or test.
