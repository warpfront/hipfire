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
# Developer env-flag call sites must match tests/golden/flags-defaults/developer-callsites.tsv
# (the GPU-free CI shard's flags_defaults_golden). Regenerate with scripts/regen-generated.sh;
# it also rebuilds the crate maps and the lifecycle/env-inventory tables checked here.
run python3 scripts/flags-golden-extract.py
# POSIX-only libc externs (mmap, dlopen, clock_gettime, ...) that Windows
# compiles: `cargo xwin check` never links, so they break only at an MSVC link.
run python3 scripts/check-unix-externs.py --self-test
run python3 scripts/check-unix-externs.py
# Qwen4 QSA pooled mRoPE position rule (host C++ build of the kernel header,
# two planted mutants must be rejected).
run python3 scripts/test_qsa_mrope_position.py
# Fixture and quality pins (fixtures/pins.toml): schema, provenance, prompt
# md5s, mirrored digests, and every `pin:<kind>/<id>` citation resolves.
run python3 scripts/check-pins.py --self-test
run python3 scripts/check-pins.py
run python3 scripts/check-lifecycle.py
run python3 scripts/check-mtp-parity-ratchet.py --self-test
run python3 scripts/check-mtp-parity-ratchet.py fixture
run python3 scripts/check-pm-code-end.py --self-test
# Generated CDNA tables vs their committed generator/receipt/XML pins, and the
# .ignore == linguist-generated list (no AMD XML or ROCm needed).
run python3 scripts/check-pm-generated-pins.py --self-test
run python3 scripts/check-pm-generated-pins.py
# Promotion lane (docs/VALIDATION.md#promotion-lane). Each self-test plants
# the defect its rule exists for and fails if the checker lets it through:
# (a) skip is not pass (hw-gate fixtures, matrix rows, lane verdicts),
# (c) a required row exercises forced advance, append and rollback; rule (b)
# (no-caller vs production callers) is kernel_registry_coverage in ci.yml.
run python3 scripts/hw-gate/run.py --self-test
run python3 scripts/device_mesh_matrix.py --self-test
run python3 scripts/promotion-lane/lane.py --self-test
run python3 scripts/promotion-lane/lane.py check
run python3 scripts/check-attribution.py --self-test
if [[ -n "${1:-}" ]]; then
    run bash scripts/ratchet-diff.sh "$1"
    run python3 scripts/check-mtp-parity-ratchet.py guard --base "$1"
    run python3 scripts/check-attribution.py --range "$1..HEAD"
fi

echo
if [[ ${#failed[@]} -gt 0 ]]; then
    echo "ci-gates: ${#failed[@]} gate(s) failed:" >&2
    printf '  %s\n' "${failed[@]}" >&2
    exit 1
fi
echo "ci-gates: all gates passed."
