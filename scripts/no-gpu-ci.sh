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

echo "== Rust no-GPU unit tests =="
cargo test -p rdna-compute --lib
cargo test -p hipfire-cpu --lib
cargo test -p hipfire-arch-qwen35 --lib moe_prefill
cargo test -p hipfire-config -p hipfire-registry -p hipfire-client -p hipfire-cli -p hipfire-tui
# peacemaker audit: lds_store_unwaited_at_barrier on committed code objects (no ROCm).
cargo test -p hipfire-isa --features toolchain --lib audit::
# Advisory Peacemaker schedule lint: deterministic CPU-only calibration, no GPU.
cargo test -j 8 -p hipfire-isa --features toolchain --lib cost_lint::
# PM typed core and its runtime driver, including the trybuild compile-fail
# race tests: their pinned rustc diagnostics catch API changes that alter them.
cargo test -p peacemaker-author

echo "== Python CPU tests =="
# Explicit paths only: scripts/*_test.py includes torch+CUDA scripts.
PYTEST_PATHS=(
    tests scripts/test_astrea.py autoresearch/ar/tests
    scripts/hw-gate/tests
    scripts/test_registry_gen_ornith.py scripts/test_registry_gen_vision.py
    scripts/test_registry_gen_coverage.py
    scripts/test_lmx_corpus.py scripts/test_ck_runtime_bundle.py
    scripts/test_check_changelog.py scripts/test_kernel_idproof.py
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
python3 scripts/test_install_revision.py
python3 scripts/test_uninstall.py
bash scripts/test-gpu-lock.sh

echo "== Env/docs drift check =="
python3 scripts/check-env-docs.py

echo "== Lifecycle check (deprecated surfaces opt-in; env/config lifecycle status) =="
python3 scripts/check-lifecycle.py

echo "== Leaked hipfire processes =="
# A test that execs a workspace-built hipfire/daemon must not leave it running.
if leaked=$(pgrep -af "(^| )$ROOT/target/[^ ]*/(hipfire|daemon)( |$)"); then
    echo "no-gpu-ci: leaked hipfire processes:" >&2
    echo "$leaked" >&2
    exit 1
fi
