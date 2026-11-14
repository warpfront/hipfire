#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
#
# The GPU-free structural gates, in one list. The ci.yml `gates` job and
# scripts/pre-commit-ci.sh both run this script, so a gate added here runs in
# both places. Every gate either fails on regression or is deleted; a gate
# nothing runs fails on nothing.
#
# Usage: scripts/ci-gates.sh [BASE_REF]
#   BASE_REF  the branch the change targets, e.g. origin/beta. When given, the
#             base-relative gates also run: ratchet-diff.sh (no committed
#             expectation loosened; honours RATCHET_RAISE_REQUIRE_LABEL and
#             RATCHET_RAISE_LABEL_PRESENT) and the MTP parity ratchet guard.
#
# Runs every gate, then exits 1 if any failed.

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
run python3 scripts/check-selector-receipts.py
run python3 scripts/check-env-docs.py
run python3 scripts/check-lifecycle.py
run python3 scripts/check-mtp-parity-ratchet.py --self-test
run python3 scripts/check-mtp-parity-ratchet.py fixture
run python3 scripts/check-pm-code-end.py --self-test
if [[ -n "${1:-}" ]]; then
    run bash scripts/ratchet-diff.sh "$1"
    run python3 scripts/check-mtp-parity-ratchet.py guard --base "$1"
fi

echo
if [[ ${#failed[@]} -gt 0 ]]; then
    echo "ci-gates: ${#failed[@]} gate(s) failed:" >&2
    printf '  %s\n' "${failed[@]}" >&2
    exit 1
fi
echo "ci-gates: all gates passed."
