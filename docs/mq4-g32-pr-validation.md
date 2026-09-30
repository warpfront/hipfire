# Packed G32 PR validation status

## Final quality attachment

See the immutable [September30 quality checkpoint](perf-checkpoints/2026-09-30-gfx1100-mq4-g32-quality.md)
and `benchmarks/results/mq4-g32-final-quality-evidence.tar.gz`.
That checkpoint records frozen implementation `be6ad1de4`: hard30 30/30 text
parity, five long-generation text matches, +16.28% median PP8192 and neutral
median TG1024. It is not silently relabelled as validation of a later binary.

## Structural cleanup

- The two new oracle/benchmark examples now require explicit `lab` features.
- The four arch-side packed calls route through `GemmFamily::run_key` and
  registered format/epilogue keys. The original GPU methods still enforce
  opt-in, exact gfx1100, admitted shape and no-capture guards.
- Default dtype resolution does not select the experimental keys. There is
  no checkpoint format, kernel source or quantization arithmetic change.
- Counts return to clean beta levels: ungated examples62, bypass calls304.
  The existing thresholds/ledger are not relaxed. Their pre-existing global
  violations remain disclosed, not marked as passes.
- Touched crate maps are mechanically regenerated; unrelated maps are untouched.

Oracle build after the feature gates:

```sh
cargo build --release --locked -p rdna-compute \
  --features lab --example bench_packed_mq4_shapes
cargo build --release --locked -p hipfire-arch-qwen35 \
  --features lab --example mq4_prefill_state_oracle
cargo test --release --locked -p hipfire-dispatch --lib packed_gemm
```

Dispatch cleanup commit: `ad337d202`. New daemon SHA256:
`82a74ce541cbd2e1aab64604980cda0b1d9e45e0d8fa7f282de81e971159cbd2`.
New-versus-frozen GPU state comparisons at512 and8193 rows both passed all646
records exactly. Packed CPU tests passed9/9; dispatch all-lib tests passed279,
with1 ignored. New-build timing completed: native786.2 -> packed919.7 tok/s
(+16.98%), decode36.7 ->36.8 tok/s,1024-token texts identical. This is one
supplemental fresh-process pair, not a replacement for the previous three-run
statistics. Raw evidence and state comparison reports are attached in
`benchmarks/results/mq4-g32-registry-validation.tar.gz`. Earlier long quality
evidence remains pinned above, not misattributed to this new executable.

## Check disposition

Product release builds, dedicated numerical tests, warmed Redline hardware
checks, general test_kernels16/16, installation/uninstall tests and touched-map
checks passed on the recorded numerical version.

Full workspace/structural checks are not green. Clean beta reproduces the
missing scratch-source example build failure, loader unsupported-VMM assertion,
five mq4c_repack Python failures, and existing ratchet/env-doc/map failures.
An initial baseline Redline Python run had extra sandbox socket errors; an
unsandboxed rerun has the same2 failures/3 errors as the candidate. Both raw
runs are retained separately. The kernel-registry source-hash test was also
reproduced independently on clean beta after the historical checkpoint was
written; that checkpoint remains unchanged.

No unavailable canonical MQ4V2/AWQ result is claimed. No generated game code
has been executed. No locked performance floor is raised.
