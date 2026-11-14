#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
#
# Ratchets for docs/governance/2026-08-15-hipfire-leanup-map.md § 4.
#
# Run from the repository root.
#
# Every metric this script emits is ASSERTED by scripts/leanup-thresholds.txt:
# a violation exits non-zero, and so does an emitted metric with no threshold
# (a number nobody asserts is not a gate). `--report` prints without asserting.
set -uo pipefail
REPORT_ONLY=0
if [ "${1:-}" = "--report" ]; then REPORT_ONLY=1; shift; fi
cd "${1:-.}" || exit 1

declare -A METRIC
p() { printf '%-26s %s\n' "$1" "$2"; METRIC["$1"]="$2"; }

DAEMON=crates/hipfire-daemon/src/main.rs
p daemon_lines "$(wc -l < $DAEMON)"
p daemon_arch_id "$(grep -cE 'arch_id *==' $DAEMON)"
p daemon_arch_refs "$(grep -coE 'hipfire_arch_[a-z0-9_]+' $DAEMON)"
# Scoped to the daemon manifest on purpose: the product must build without a
# feature incantation. Tree-wide `required-features` is a different thing
# entirely and is a GOOD signal -- it is how archived probes stay out of the
# default build. See `ungated_examples`.
p required_features_daemon "$(grep -c 'required-features' crates/hipfire-daemon/Cargo.toml)"
# `daemon_arch_refs` greps `hipfire_arch_*`, which `ModelState::Qwen35` does NOT match:
# ModelState is a LOADER-owned enum wrapping arch bundles. The daemon therefore reported
# 0 arch refs while still doing a 7-way architecture dispatch (main.rs:1732-1751). Count
# the laundered form too, or the gate certifies a decoupling that has not happened.
p daemon_modelstate "$(grep -co 'ModelState::' $DAEMON)"
# Split code from prose. ModelState is deleted; the remaining mentions are
# history in comments, and a gate that cannot tell those apart would either fail
# on a comment or pass on a reintroduction.
p loader_modelstate_code "$(grep -rhn 'ModelState::' crates/hipfire-loader/src 2>/dev/null | grep -vE '^[0-9]+:[[:space:]]*(//|///|\*)' | grep -c 'ModelState::')"
# `runtime_examples` used to count [[example]] declarations in hipfire-runtime.
# Gating an archived probe ADDS a declaration, so the number rose while the
# thing it meant to track improved. Direction inverted; removed. `ungated_examples`
# below measures the intent -- how many examples the default build still pays for.
p grammar_copies "$(find crates/hipfire-arch-*/src -name grammar.rs 2>/dev/null | wc -l)"
# Engine substrate (saddle-core, hipfire-engine, hipfire-dispatch) must not
# name an architecture.
leak=$(grep -roE 'hipfire_arch_[a-z0-9_]+' crates/saddle-core/src crates/hipfire-engine/src \
        crates/hipfire-dispatch/src 2>/dev/null | wc -l)
p substrate_clean_arch_refs "$leak"

# --- ungated research probes -------------------------------------------------
# Examples compile on every `cargo build --all-targets`. Archived one-question
# GPU probes are gated behind `--features lab` so an answer obtained in April is
# not type-checked in August. This counts the ones still UNGATED so the number
# cannot silently drift back up -- which is exactly how the dead golden CASES
# table accumulated five models, four of which do not exist.
#
# Counts example NAMES that have a required-features line inside their own
# [[example]] block. An earlier version counted required-features LINES per
# manifest and reported 16 where the answer was 32; a metric that is wrong is
# worse than no metric, because it is believed.
ungated_examples() {
  python3 - <<'PYEOF'
import glob, os, re
total = {(p.split("/")[1], os.path.basename(p)[:-3]) for p in glob.glob("crates/*/examples/*.rs")}
gated = set()
for tm in glob.glob("crates/*/Cargo.toml"):
    crate = tm.split("/")[1]
    for blk in re.split(r'(?=\[\[example\]\])', open(tm, encoding='utf8').read()):
        if not blk.startswith("[[example]]"):
            continue
        blk = re.split(r'\n\[(?!\[example)', blk)[0]
        nm = re.search(r'name\s*=\s*"([^"]+)"', blk)
        rf = re.search(r'required-features\s*=\s*\[([^\]]*)\]', blk)
        if nm and rf and rf.group(1).strip():
            gated.add((crate, nm.group(1)))
print(len(total - gated))
PYEOF
}
p ungated_examples "$(ungated_examples)"

# --- layering, derived from the Cargo graph -------------------------------
# scripts/check-layering.py computes the arch band from the real dependency
# graph rather than from a declared rule. The Phase 3 scope asserted that arch
# crates must not depend on saddle-core or hipfire-dispatch; measuring showed 11
# such edges already exist and are correct -- those crates sit at layers 4-5,
# BELOW the arch band at 6-7. A hardcoded rule would have failed on legitimate
# structure.
while read -r _k _v; do
  case "$_k" in [a-z]*) p "$_k" "$_v" ;; esac
done < <(python3 scripts/check-layering.py 2>/dev/null | grep -E '^[a-z_]+[[:space:]]+[0-9]+$')

# --- arch-key dispatch (string form of arch_id ==) ---------------------------
# arch_id == is asserted at 0, but the same coupling written as a string match
# is invisible to it. `match arch { "gemma4" => .. }` was written into
# hipfire-runtime during Phase 3B and reverted; nothing here would have objected.
while read -r _k _v; do
  case "$_k" in [a-z]*) p "$_k" "$_v" ;; esac
done < <(python3 scripts/check-arch-dispatch.py 2>/dev/null | grep -E '^[a-z_]+[[:space:]]+[0-9]+$')

# --- quant-type registry parity ----------------------------------------------
# The wire qt space is declared in hipfire-quantize::QuantType and consumed via
# hipfire-runtime::RAW_CODECS, with rdna-compute::DType as the execution twin.
# Nothing binds them, and every consumer arm is fail-open (`_ => None/Err`), so a
# half-registered format compiles green -- cf. cf061b7ed, where the encoder,
# GEMV and is_mq arms all landed and the model simply would not load.
while read -r _k _v; do
  case "$_k" in [a-z]*) p "$_k" "$_v" ;; esac
done < <(python3 scripts/check-quant-registry.py 2>/dev/null | grep -E '^[a-z_]+[[:space:]]+[0-9]+$')

# --- dispatch-bypass debt ledger ---------------------------------------------
# 222 arch-crate call sites reach Gpu::gemv_*/gemm_*/fused_* directly instead of
# resolving through hipfire_dispatch::KernelRegistry, which has existed since
# e822b319e. This is a debt TABLE, not a purity check: the rows record what is
# owed per crate, growth fails, and paying down is a one-line edit. The holster
# objective draws this ledger to zero over many commits, not one.
while read -r _k _v; do
  case "$_k" in [a-z]*) p "$_k" "$_v" ;; esac
done < <(python3 scripts/check-dispatch-bypass.py 2>/dev/null | grep -E '^[a-z_]+[[:space:]]+[0-9]+$')

# --- multi-card arch-id arms -------------------------------------------------
# Multi-GPU routing is a carrier declaration (Carrier::topology_caps), not an
# arch-id match. check-topology-arch-arms.sh counts the arch-id match forms
# that remain in the multi-card front-door files; the ceilings are
# topology_arch_arms_* in scripts/leanup-thresholds.txt.
while read -r _k _v; do
  case "$_k" in [a-z]*) p "$_k" "$_v" ;; esac
done < <(bash scripts/check-topology-arch-arms.sh --metrics 2>/dev/null | grep -E '^[a-z_]+[[:space:]]+[0-9]+$')

# --- assertion -------------------------------------------------------------
# The point of the file. A metric named in scripts/leanup-thresholds.txt must
# satisfy its threshold or this exits non-zero.
THRESH="scripts/leanup-thresholds.txt"
if [ "$REPORT_ONLY" -eq 1 ]; then
  echo
  echo "(--report: thresholds not asserted)"
  exit 0
fi
if [ ! -f "$THRESH" ]; then
  echo
  echo "leanup-ratchets: FAIL — $THRESH is missing."
  echo "  Without it nothing is asserted, which is the state this gate exists to end."
  exit 1
fi

fails=0
checked=0
declare -A THRESHOLDED
while read -r metric op want; do
  case "$metric" in ''|'#'*) continue ;; esac
  got="${METRIC[$metric]:-}"
  if [ -z "$got" ]; then
    echo "leanup-ratchets: FAIL — threshold names unknown metric '$metric'."
    echo "  A threshold on a metric that is not emitted silently protects nothing."
    fails=$((fails+1)); continue
  fi
  THRESHOLDED["$metric"]=1
  checked=$((checked+1))
  case "$op" in
    "==") [ "$got" = "$want" ] || { echo "leanup-ratchets: FAIL $metric = $got, must be $want"; fails=$((fails+1)); } ;;
    "<=") if [ "$got" -gt "$want" ] 2>/dev/null; then
            echo "leanup-ratchets: FAIL $metric = $got, ceiling $want"; fails=$((fails+1))
          fi ;;
    *) echo "leanup-ratchets: FAIL — unknown operator '$op' for $metric"; fails=$((fails+1)) ;;
  esac
done < "$THRESH"
for metric in "${!METRIC[@]}"; do
  if [ -z "${THRESHOLDED[$metric]:-}" ]; then
    echo "leanup-ratchets: FAIL — metric '$metric' is emitted but has no threshold in $THRESH."
    fails=$((fails+1))
  fi
done

echo
if [ "$fails" -gt 0 ]; then
  echo "leanup-ratchets: $fails violation(s) across $checked asserted metric(s)."
  echo "  Lowering a ceiling after an improvement is expected. Raising one needs a"
  echo "  sentence in the commit saying what was traded for it."
  exit 1
fi
echo "leanup-ratchets: OK — $checked metric(s) asserted, 0 violations."

