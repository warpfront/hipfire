#!/usr/bin/env bash
# Run the fast GPU-free CI checks locally before committing.
#
# Same commands as .github/workflows/ci.yml (gates, supply-chain, fmt, clippy)
# and the static checks of scripts/no-gpu-ci.sh. Skipped as slow or not
# runnable locally: the release `build` job, every test step, cargo-deny
# `advisories` (network fetch; continue-on-error in CI), and hw-gate (GPU +
# reviewer seats). no-gpu-ci's `cargo check --workspace --examples` is covered
# by clippy --all-targets.
#
# Usage: scripts/pre-commit-ci.sh [BASE_REF]
#   BASE_REF  the branch the PR targets, e.g. origin/beta; forwarded to
#             ratchet-diff.sh (its default is origin/master).
#
# rustfmt: the whole workspace must be clean (`cargo-fmt --all -- --check`;
# beta is rustfmt-clean since the 0.4.3 sweep, and the generated AIE2P tables
# are #[rustfmt::skip]), plus CI's per-file check on the staged + unstaged .rs
# files vs HEAD, which also covers files outside the workspace (tools/).
# `cargo-fmt` is called directly because .cargo/config.toml shadows `cargo fmt`.
# Runs every check, then exits 1 if any failed. Missing cargo-deny is an unmet
# prerequisite, not a supply-chain finding; report how to install it and fail
# closed.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 2

failed=()
run() {
    echo "== $* =="
    "$@" || failed+=("$*")
}

run bash scripts/leanup-ratchets.sh
run python3 scripts/check-crate-maps.py --check
run python3 scripts/check-scratch-growth.py
run python3 scripts/check-changelog.py
run python3 scripts/check-conflict-markers.py
run bash scripts/ratchet-diff.sh ${1:+"$1"}
if cargo deny --version >/dev/null 2>&1; then
    run cargo deny check bans licenses sources
else
    echo "== cargo deny check bans licenses sources =="
    echo "ERROR: cargo-deny is unavailable; install it with: cargo install --locked cargo-deny" >&2
    failed+=("cargo deny check bans licenses sources (cargo-deny unavailable)")
fi
run cargo-fmt --all -- --check
run bash scripts/ci-rustfmt-changed.sh
run cargo clippy --workspace --all-targets --locked
run python3 scripts/check-env-docs.py
run python3 scripts/check-lifecycle.py
run python3 scripts/check-selector-receipts.py

echo
if [[ ${#failed[@]} -gt 0 ]]; then
    echo "pre-commit-ci: ${#failed[@]} check(s) failed:" >&2
    printf '  %s\n' "${failed[@]}" >&2
    exit 1
fi
echo "pre-commit-ci: all checks passed."
