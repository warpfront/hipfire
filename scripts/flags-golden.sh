#!/usr/bin/env bash
# Flags migration gates GB (defaults golden) and GC (legacy spelling golden), CPU only.
#
#   scripts/flags-golden.sh                       run every GB/GC check
#   scripts/flags-golden.sh --regen               rewrite the goldens (review the diff!)
#   scripts/flags-golden.sh --regen-names LEDGER.csv [CENSUS.csv]
#                                                 rewrite names.tsv from the migration ledger
#
# Checks:
#   GB  cargo test -p hipfire-runtime --test flags_defaults_golden   (FeatureFlags, ArchCaps,
#       RuntimeConfig, schema defaults for the 20 gfx strings and an empty config)
#   GB  scripts/flags-golden-extract.py                              (developer_* call-site defaults)
#   GC  cargo test -p hipfire-runtime --lib legacy_spelling          (old readers vs
#       tests/golden/flags-legacy-spelling/{grammars,names}.tsv)
#
# Cargo runs under /home/kaden/qcal/locks/build-slot.sh when it exists and honours
# CARGO_TARGET_DIR. Goldens live in tests/golden/flags-defaults and
# tests/golden/flags-legacy-spelling.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 2

SLOT=/home/kaden/qcal/locks/build-slot.sh
slot=()
[[ -x $SLOT ]] && slot=("$SLOT")
cargo_run() { "${slot[@]}" cargo "$@"; }

failed=()
step() {
    echo "== $* =="
    "$@" || failed+=("$*")
}

case "${1:-}" in
--regen)
    cargo_run test -p hipfire-runtime --test flags_defaults_golden -- --ignored regenerate_flags_defaults_golden || exit 1
    python3 scripts/flags-golden-extract.py --write || exit 1
    cargo_run test -p hipfire-runtime --lib -- --ignored regenerate_legacy_spelling_golden || exit 1
    exit 0
    ;;
--regen-names)
    [[ -n ${2:-} ]] || { echo "usage: $0 --regen-names LEDGER.csv [CENSUS.csv]" >&2; exit 2; }
    exec python3 scripts/flags-golden-names.py --ledger "$2" ${3:+--census "$3"} --write
    ;;
"") ;;
*)
    echo "unknown argument: $1" >&2
    exit 2
    ;;
esac

step cargo_run test -p hipfire-runtime --test flags_defaults_golden
step python3 scripts/flags-golden-extract.py
step cargo_run test -p hipfire-runtime --lib legacy_spelling

echo
if [[ ${#failed[@]} -gt 0 ]]; then
    echo "flags-golden: ${#failed[@]} check(s) failed:" >&2
    printf '  %s\n' "${failed[@]}" >&2
    exit 1
fi
echo "flags-golden: GB and GC passed."
