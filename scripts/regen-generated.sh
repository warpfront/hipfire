#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
#
# Regenerate every generated file that a tool derives from the tree and a gate
# checks for drift. On a clean tree it changes nothing.
#
#   scripts/regen-generated.sh            fast set (Python, <30s)
#   scripts/regen-generated.sh --all      fast set + the cargo-built goldens
#   scripts/regen-generated.sh --staged   fast set, plus the cargo goldens whose
#                                         inputs the staged diff touches; prints the
#                                         changed files (the pre-commit hook stages them)
#
# Fast set                                     tool                                      drift gate
#   crates/*/map.md generated blocks           check-crate-maps.py <crate>...            check-crate-maps.py --check
#   tests/golden/flags-defaults/
#     developer-callsites.tsv                  flags-golden-extract.py --write           flags-golden-extract.py
#   docs/CONFIG.md lifecycle table,
#   docs/env-vars.md env inventory             check-lifecycle.py --write                check-lifecycle.py
# Cargo set (needs a build; under /home/kaden/qcal/locks/build-slot.sh when present)
#   tests/golden/flags-defaults/*.tsv (rest),
#   tests/golden/flags-legacy-spelling/*.tsv   flags_defaults_golden / legacy_spelling   cargo test (ci.yml)
#   crates/hipfire-tui/tests/snapshots/*.txt   UPDATE_SNAPSHOTS=1 cargo test golden_    cargo test (ci.yml)
#
# NOT regenerated here, because each is a human decision a tool must not make:
# ratchet ceilings (scripts/leanup-thresholds.txt), the selector-receipts
# baseline (shrink-only), lifecycle labels (`// lifecycle:` marks and statuses),
# fixture/quality pins (fixtures/pins.toml), the kernel-registry census
# allowlist and flagship/route-plan source pins, the generated CDNA tables
# (need AMD's XML; check-pm-generated-pins.py), the hand-written prose of
# docs/env-vars.md (check-env-docs.py) and CHANGELOG entries.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 2

MODE="${1:-}"
case "$MODE" in ''|--all|--staged) ;; *) echo "usage: $0 [--all|--staged]" >&2; exit 2 ;; esac

FAST_OUTPUTS=(crates/*/map.md tests/golden/flags-defaults/developer-callsites.tsv docs/CONFIG.md docs/env-vars.md)
FLAGS_OUTPUTS=(tests/golden/flags-defaults tests/golden/flags-legacy-spelling)
TUI_OUTPUTS=(crates/hipfire-tui/tests/snapshots)

flags_cargo=0
tui_cargo=0
if [ "$MODE" = --all ]; then
    flags_cargo=1
    tui_cargo=1
elif [ "$MODE" = --staged ]; then
    staged="$(git diff --cached --name-only --diff-filter=ACMR)"
    grep -qE '^crates/(hipfire-config/|hipfire-runtime/src/config|rdna-compute/src/(feature_flags|arch_caps)\.rs)' <<<"$staged" && flags_cargo=1
    grep -qE '^crates/(hipfire-tui|hipfire-config)/' <<<"$staged" && tui_cargo=1
fi

outputs=("${FAST_OUTPUTS[@]}")
[ "$flags_cargo" = 1 ] && outputs+=("${FLAGS_OUTPUTS[@]}")
[ "$tui_cargo" = 1 ] && outputs+=("${TUI_OUTPUTS[@]}")
before="$(git ls-files -mo --exclude-standard -- "${outputs[@]}" | xargs -r sha1sum 2>/dev/null)"

failed=()
step() {
    echo "== ${*:1:6} ==" >&2
    # Per-crate "already up to date" lines are noise on every commit.
    "$@" 2>&1 | grep -v ': already up to date$' >&2
    [ "${PIPESTATUS[0]}" -eq 0 ] || failed+=("$*")
}

mapfile -t crates < <(ls -d crates/*/map.md | cut -d/ -f2)
step python3 scripts/check-crate-maps.py "${crates[@]}"
step python3 scripts/flags-golden-extract.py --write
step python3 scripts/check-lifecycle.py --write

SLOT=/home/kaden/qcal/locks/build-slot.sh
slot=()
[ -x "$SLOT" ] && slot=("$SLOT")
if [ "$flags_cargo" = 1 ]; then
    step "${slot[@]}" cargo test -p hipfire-runtime --test flags_defaults_golden -- --ignored regenerate_flags_defaults_golden
    step "${slot[@]}" cargo test -p hipfire-runtime --lib -- --ignored regenerate_legacy_spelling_golden
fi
if [ "$tui_cargo" = 1 ]; then
    step env UPDATE_SNAPSHOTS=1 "${slot[@]}" cargo test -p hipfire-tui golden_
fi

if [ "${#failed[@]}" -gt 0 ]; then
    echo "regen-generated: ${#failed[@]} generator(s) failed:" >&2
    printf '  %s\n' "${failed[@]}" >&2
    exit 1
fi

after="$(git ls-files -mo --exclude-standard -- "${outputs[@]}" | xargs -r sha1sum 2>/dev/null)"
# Files this run changed: modified/untracked now, and not with the same content before.
comm -13 <(sort <<<"$before") <(sort <<<"$after") | awk 'NF {print $2}'
