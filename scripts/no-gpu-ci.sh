#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Never touch a GPU or the caller's hipfire state, whoever calls this: hide
# every device, drop the caller's HIPFIRE_* overrides, and give the run a
# private lock dir and HIPFIRE_HOME. HOME stays as is (rustup/cargo live there).
while IFS= read -r var; do unset "$var"; done < <(compgen -e | grep '^HIPFIRE_' || true)
export ROCR_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1
HIPFIRE_LOCK_DIR="$(mktemp -d)"
HIPFIRE_HOME="$(mktemp -d)"
export HIPFIRE_LOCK_DIR HIPFIRE_HOME
trap 'rm -rf "$HIPFIRE_LOCK_DIR" "$HIPFIRE_HOME"' EXIT

if [[ "${1:-}" == "--self-check" ]]; then
    echo "ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES"
    echo "HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"
    echo "HIPFIRE_LOCK_DIR=$HIPFIRE_LOCK_DIR"
    echo "HIPFIRE_HOME=$HIPFIRE_HOME"
    exit 0
fi

echo "== Rust check =="
cargo check --workspace --examples

echo "== Rust no-GPU tests the ci.yml test shards do not run =="
# ci.yml already runs every --lib unit test (lib-peacemaker-isa, with the
# peacemaker-isa `toolchain` feature; lib-rest), all_launcher_abi
# (rdna-compute-abi), hipfire-cli (serve) and hipfire-tui's bin tests (int-rest).
# Left here: the hipfire-config bin and peacemaker-author's integration tests.
cargo test -p hipfire-config --bins
# PM typed core and its runtime driver, including the trybuild compile-fail
# race tests: their pinned rustc diagnostics catch API changes that alter them.
cargo test -p peacemaker-author --test '*'

echo "== Python CPU tests =="
# Explicit paths only: scripts/*_test.py includes torch+CUDA scripts.
PYTEST_PATHS=(
    tests scripts/test_astrea.py autoresearch/ar/tests
    scripts/hw-gate/tests
    scripts/test_registry_gen_ornith.py scripts/test_registry_gen_vision.py
    scripts/test_registry_gen_coverage.py scripts/test_registry_gen_decider.py
    scripts/test_lmx_corpus.py scripts/test_ck_runtime_bundle.py
    scripts/test_check_changelog.py scripts/test_check_conflict_markers.py scripts/test_kernel_idproof.py
    scripts/test_agentic_gate_driver.py
    scripts/test_post_land_smoke.py scripts/test_post_land_smoke_orchestrator.py
    scripts/test_check_crate_maps.py scripts/test-pm-rdna-xml.py
    scripts/test_registry_gen_flash_next.py scripts/test_registry_gen_ple.py
    scripts/test_route_plan_build.py scripts/test_route_plan_walk.py
    scripts/test_ep_hot_units.py scripts/test_k0rank.py scripts/test_fn_ep_gate_chunk.py
    scripts/test_gfx1100_xts_scope.py scripts/test_q8_overlay_writer.py scripts/test_subq4_g2_carrier.py
    scripts/slowlink-gap/quality_test_scorer.py scripts/slowlink-gap/quality_test_campaign.py
)
if python3 -c 'import pytest, numpy' 2>/dev/null; then
    python3 -m pytest "${PYTEST_PATHS[@]}"
elif command -v uv >/dev/null 2>&1; then
    uv run --with pytest --with numpy python -m pytest "${PYTEST_PATHS[@]}"
else
    echo "no-gpu-ci: pytest/numpy missing and uv unavailable" >&2
    exit 1
fi
python3 -m unittest tools.redline.tests.test_product_bench tools.redline.tests.test_golden tools.redline.tests.test_serve_diff tools.redline.tests.test_lower tools.redline.tests.test_dispatch_profile
python3 -m unittest discover -s scripts/jev_eval -p 'test_*.py'
python3 scripts/test_install_revision.py
python3 scripts/test_client_build_stamp.py
python3 scripts/test_uninstall.py
bash scripts/test-gpu-lock.sh
bash scripts/test-fmt-sweep-rebase.sh
python3 scripts/test-no-cargo-fmt-guard.py

# The structural gates (env-docs, lifecycle, MTP parity, ...) are
# scripts/ci-gates.sh, run by the ci.yml `gates` job.

echo "== Leaked hipfire processes =="
# A test that execs a workspace-built hipfire/daemon must not leave it running.
if leaked=$(pgrep -af "(^| )$ROOT/target/[^ ]*/(hipfire|daemon)( |$)"); then
    echo "no-gpu-ci: leaked hipfire processes:" >&2
    echo "$leaked" >&2
    exit 1
fi
