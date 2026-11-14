#!/usr/bin/env bash
# Run the fast GPU-free CI checks locally before committing.
#
# The structural gates are scripts/ci-gates.sh, the same runner the ci.yml
# `gates` job calls; this adds ci.yml's supply-chain, fmt and clippy jobs.
# Skipped as slow or not runnable locally: the release `build` job, every test
# step (scripts/no-gpu-ci.sh), cargo-deny `advisories` (network fetch;
# continue-on-error in CI), and hw-gate (GPU + reviewer seats).
#
# Usage: scripts/pre-commit-ci.sh [BASE_REF]
#   BASE_REF  the branch the PR targets (default origin/beta); forwarded to
#             ci-gates.sh for its base-relative gates.
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

run bash scripts/ci-gates.sh "${1:-origin/beta}"
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

echo
if [[ ${#failed[@]} -gt 0 ]]; then
    echo "pre-commit-ci: ${#failed[@]} check(s) failed:" >&2
    printf '  %s\n' "${failed[@]}" >&2
    exit 1
fi
echo "pre-commit-ci: all checks passed."
