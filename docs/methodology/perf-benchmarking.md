# Perf benchmarking methodology

The protocol for measuring kernel-level tok/s honestly. Read this
before claiming any win in a commit message; the gates assume you've
followed it.

## The within-session noise band

On gfx1100 (7900 XTX) the within-session A/B noise band on a fresh
process is **±10–15%** depending on DPM state, thermal headroom, and
firmware version. This is BIG. A "+8%" win measured by changing some
code in one shell session and re-running is **inside the noise**.

Sources of within-session drift, ranked by impact:

1. **Stale build cache**. The speed-gate's `ensure_build()` is a
   no-op when the bench binary already exists. A "stash and re-bench"
   flow leaves the post-change binary in place, so both runs measure
   the same code. Always `rm target/release/examples/<bench>` before
   re-running.
2. **DPM state**. The GPU clocks ramp up over the first ~5 seconds of
   sustained load; benchmarks that include warmup catch this, ones
   that don't measure cold clocks for the first run and hot clocks
   for subsequent runs. Use `cat /sys/class/drm/card*/device/pp_dpm_sclk`
   to inspect.
3. **Firmware shadowing**. `/lib/firmware/updates/amdgpu` overrides
   the kernel's bundled firmware; if the dkms-installed firmware
   doesn't match the kernel-installed firmware, you get a SMU IF
   mismatch and ~50% prefill cratering. Fix:
   `sudo mv /lib/firmware/updates/amdgpu /lib/firmware/updates/amdgpu.bak
   && sudo reboot`. Symptoms in `dmesg | tail -40`.
4. **Thermal throttle**. After 10+ minutes of sustained DFlash runs
   with the case closed, the 7900 XTX throttles. Check
   `cat /sys/class/drm/card*/device/hwmon/hwmon*/temp*_input`.

## Cross-process verification

To trust a perf delta as real:

```bash
# 1. Pin to the candidate commit
git checkout <candidate>

# 2. Rebuild from clean
cargo clean -p rdna-compute
rm -f target/release/examples/bench_qwen35_mq4
cargo build --release --features deltanet -p hipfire-runtime \
    --example bench_qwen35_mq4

# 3. Run the gate
./scripts/speed-gate.sh --fast

# 4. Same procedure for the baseline
git checkout <baseline>
# ... (same build + bench)
```

Or via the harness directly:

```bash
./scripts/probe_commits.sh <baseline-sha> <candidate-sha>
```

`probe_commits.sh` does this end-to-end with fresh process per commit
+ a multi-run median. A delta that survives this protocol is real;
one that doesn't probably isn't.

## Speed-gate baselines

`tests/speed-baselines/<arch>.txt` records the "ground floor" decode
+ prefill numbers per arch. The pre-commit hook runs
`scripts/speed-gate.sh --fast` automatically when the staged diff
touches kernel / dispatch / forward-pass files. Tolerance is ±5% from
the committed baseline.

If a legitimate change trades a small regression on the baseline arch
for a much bigger win on another arch (rare, but happens), update the
baseline in the same commit:

```bash
./scripts/speed-gate.sh --update-baselines
git add tests/speed-baselines/
git commit
```

The baseline change stays in the same commit so a reviewer sees the
trade-off explicitly. Don't sneak baseline updates into separate
"chore" commits.

**Do not use `--no-verify` to bypass the gate.** A regression that
the gate catches is information you need; bypassing produces a commit
that masks the issue from `git bisect` later. Authorized exceptions
must be explicitly stated in writing by the maintainer for that
specific change.

## Prompt structure matters

Two prompts that tokenize to the same number of tokens but with
different whitespace patterns produce dramatically different DFlash τ.
Same model, same flags, same binary md5:

```
PEP-8 strict (\n\n\n between top-level defs):    27B-3.5  τ=8.07  → 161 tok/s
Single-blank (\n\n between top-level defs):      27B-3.5  τ=9.42  → 184 tok/s
```

A 14% perf swing from a whitespace cleanup is invisible in code
review but catastrophic for benchmarking. **All cross-session perf
comparisons MUST use byte-identical prompts**:

- Embed prompts as committed files (not heredocs in scripts that
  editors reformat).
- Record the prompt md5 alongside the result.
- If you can't verify the prompt md5, treat the comparison as
  unreliable.

The engine collapses `\n{3,}` → `\n\n` at prompt entry by default
(`prompt_normalize=true`). This eliminates the whitespace-variance
source for normal use, but bench scripts that bypass the engine entry
point still need the prompt-md5 discipline.

## DFlash speed gate

`scripts/coherence-gate-dflash.sh` runs the spec-decode coherence
battery: a fixed (model × prompt × spec-mode) matrix that catches
token-attractor regressions (output that passes the perf gate while
emitting `[1734 2357 2733 283 869 1734 2357 ...]` for 1500 tokens).

Three-tier thresholds:

| Tier | Window | Hard fail if |
|---|---|---|
| 1 | First 128 tokens | unique_token_ratio < 0.15 OR max_token_freq > 0.50 |
| 2 | Last 128 tokens | unique_token_ratio < 0.30 OR max_token_freq > 0.50 |
| 3 | Full output | 3-gram repetition density > 50% in second half |

Why both ends: single-token attractors show up in the first 128;
block-level structural loops (5+ token sequences repeating) appear
later. The CASK m-fold + DFlash 2026-04-26 case (τ=8.98, tight
stddev) passed the first-128 gate but emitted 76+ block-loop reps in
the last 1000 tokens.

Any DFlash perf bench that lacks the unique-token + max-frequency
checks AND visual eyeballing of output is unreliable. **Tight stddev
on a spec-decode bench is actively suspicious, not reassuring**:
real acceptance noise is wider; tight stddev correlates with
deterministic attractors.

## What the gates do NOT catch

- Per-lane WMMA mapping bugs (see commit `b7ac66a` — the gfx11 C-mapping
  was silently wrong for 6 weeks while the gates were green).
- Output drift that's coherent but worse on a benchmark task hipfire
  doesn't measure. The coherence gate is unique-token / repetition; it
  doesn't measure HumanEval pass-rate.
- Long-context drift past the prompts the gates exercise.

For correctness on a new arch port: run
`crates/hipfire-runtime/examples/test_kernels.rs` on the target hardware. It
runs each kernel through small golden cases against a CPU reference.
That's the load-bearing correctness gate; the perf and coherence
gates are optimization gates.
