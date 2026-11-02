#!/usr/bin/env bash
# The 0.4.3 one-time workspace rustfmt sweep, as a deterministic function of a tree.
#
#   sweep.sh <worktree>
#
# 1. `cargo-fmt --all` (rustfmt 1.9.0-stable, no rustfmt.toml), repeated until
#    `cargo-fmt --all -- --check` is clean: rustfmt is not idempotent on three
#    files (a `return Err(format!(..))` match arm needs a second pass).
# 2. Every other tracked .rs file (standalone tools/ crates, scripts/reap,
#    trybuild pass fixtures, the orphan hipfire-dispatch model_ext/) is
#    formatted with CI's per-file command (scripts/ci-rustfmt-changed.sh:
#    owning crate's edition, skip_children), again until its check is clean.
# Excluded (left byte-identical):
#   - crates/peacemaker-npu/src/isa/{gen,sched}.rs: generated, #[rustfmt::skip]
#   - any foo.rs with a sibling foo.stderr: trybuild compile_fail fixture whose
#     expected stderr carries line:col
#   - docs/**: recorded evidence
# Exit 0 = tree formatted and both checks clean.
set -euo pipefail
cd "$1"
export ROCR_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 HIPFIRE_DEVICES=-1

excluded() {
  case "$1" in
    crates/peacemaker-npu/src/isa/gen.rs|crates/peacemaker-npu/src/isa/sched.rs|docs/*) return 0 ;;
  esac
  [[ -f "${1%.rs}.stderr" ]]
}
source <(sed -n '/^file_edition()/,/^}/p' scripts/ci-rustfmt-changed.sh)

for pass in 1 2 3 4; do
  nice -n 10 cargo-fmt --all
  if cargo-fmt --all -- --check >/dev/null 2>&1; then break; fi
  [[ "$pass" -lt 4 ]] || { echo "cargo-fmt did not converge in 4 passes" >&2; exit 1; }
done
echo "cargo-fmt --all: clean after pass $pass"

per_file_check() { rustfmt --edition "$(file_edition "$1")" --check --config skip_children=true "$1" >/dev/null 2>&1; }
per_file_fmt() { rustfmt --edition "$(file_edition "$1")" --config skip_children=true "$1" 2>/dev/null; }

leftover=0
while IFS= read -r f; do
  excluded "$f" && continue
  per_file_check "$f" && continue
  leftover=$((leftover + 1))
  for pass in 1 2 3 4; do
    per_file_fmt "$f"
    per_file_check "$f" && break
    [[ "$pass" -lt 4 ]] || { echo "rustfmt did not converge on $f" >&2; exit 1; }
  done
  echo "per-file: $f"
done < <(git ls-files '*.rs')
echo "per-file leftovers formatted: $leftover"

# Step 2 may not have disturbed step 1's fixpoint.
cargo-fmt --all -- --check >/dev/null
echo "sweep: done"
