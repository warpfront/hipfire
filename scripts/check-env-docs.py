#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "env-vars.md"
REFERENCE_DOCS = [ROOT / "AGENTS.md", ROOT / "README.md", ROOT / "CONTRIBUTING.md"]
DIRECT_ENV = re.compile(
    r'\b(?:std::)?env::var(?:_os)?\(\s*"(HIPFIRE_[A-Z0-9_]+)"\s*\)',
    re.MULTILINE,
)
BOOTSTRAP_ENV = {
    "HIPFIRE_HOME",
    "HIPFIRE_MODELS_DIR",
    "HIPFIRE_DAEMON_BIN",
    "HIPFIRE_TUI_BIN",
    "HIPFIRE_CLI_BIN",
    "HIPFIRE_HF_BASE",
    "HIPFIRE_REGISTRY_URL",
    "HIPFIRE_NO_REGISTRY_FETCH",
    "HIPFIRE_KERNEL_CACHE",
    "HIPFIRE_SPILL_DIR",
    "HIPFIRE_QUANT_DIAG_PATH",
    # Test-harness marker read by the shared ROCm resolver (crates/rocm.rs);
    # only #[test] code calls it, it never affects runtime behaviour.
    "HIPFIRE_TEST_REQUIRE_ROCM",
    # Test-only real-compiler path for the bounded-executor CPU gate in
    # rdna-compute compiler.rs #[cfg(test)]; never read by runtime code.
    "HIPFIRE_TEST_REAL_HIPCC",
    # Test-only probe inputs of the #[ignore]d real-rows check in
    # hipfire-runtime ple_q4.rs #[cfg(test)]; never read by runtime code.
    "HIPFIRE_PROBE_MODEL",
    "HIPFIRE_PROBE_ROWS",
    # Test-only HF tokenizer.json path of the #[ignore]d Raon tokenizer parity
    # check in hipfire-runtime tokenizer.rs #[cfg(test)]; never read by runtime code.
    "HIPFIRE_RAON_TOKENIZER",
    # Test-only fixture path of the #[ignore]d packed-host-expert parity check in
    # hipfire-arch-qwen35 qwen35/load.rs #[cfg(test)]; never read by runtime code.
    "HIPFIRE_MOE_PARITY_FIXTURE",
    # Test-only inputs of the #[ignore]d emulated/physical TP oracle in
    # hipfire-arch-qwen35 qwen35/mesh_tp.rs #[cfg(test)]; never read by runtime code.
    "HIPFIRE_QWEN35_TP_MODEL",
    "HIPFIRE_QWEN35_TP_PARTS",
    "HIPFIRE_QWEN35_TP_ROWS",
    "HIPFIRE_QWEN35_TP_CHUNK",
    # Same for the Glimmer TP oracle in hipfire-arch-muse-glimmer mesh_tp.rs
    # #[cfg(test)]; never read by runtime code.
    "HIPFIRE_GLIMMER_TP_MODEL",
    "HIPFIRE_GLIMMER_TP_PARTS",
    "HIPFIRE_GLIMMER_TP_ROWS",
    "HIPFIRE_GLIMMER_TP_CHUNK",
    "HIPFIRE_GLIMMER_TP_GREEDY",
    # Read by the daemon's init_tracing() before the CLI has sent the process
    # config; going through developer_var there installs the local fallback
    # snapshot and makes the real install fail ("already initialized").
    "HIPFIRE_LOG_FORMAT",
    # Read by GpuLocks::open() before install_process_config() — the lock
    # directory must be resolved before the config snapshot exists, and a
    # developer_var read would install the local fallback early. Two daemons
    # contending for a card must read the same ambient dir before either has
    # its own config.
    "HIPFIRE_LOCK_DIR",
    # M7 certification receipt cache root, read by peacemaker-isa
    # cert_cache.rs. The peacemaker crates build standalone and deliberately
    # have no hipfire-config dependency, so this is an ambient developer read;
    # a directory selects the root, `0` disables the cache. Storage only: it
    # never changes a receipt's contents.
    "HIPFIRE_CERT_CACHE",
    # Test-only harness inputs of the M7 cache cases in pm_check.rs
    # #[cfg(test)]: the isolated child's case name and fixture root (the cache
    # directory reaches it as HIPFIRE_CERT_CACHE above). Never read by runtime
    # code.
    "HIPFIRE_M7_TEST_CASE",
    "HIPFIRE_M7_TEST_ROOT",
    # Shared-weight host opt-in, its socket and the dev-fabric admission
    # token: read before install_process_config() because they select the
    # GPU claim kind (shared vs exclusive), exactly like HIPFIRE_LOCK_DIR.
    "HIPFIRE_WEIGHT_HOST",
    "HIPFIRE_WEIGHT_HOST_SOCKET",
    "HIPFIRE_FABRIC_ADMISSION",
}
CENTRAL_CONFIG_READERS = {
    "crates/hipfire-config/src/lib.rs",
    "crates/hipfire-runtime/src/config.rs",
    "crates/rdna-compute/src/feature_flags.rs",
}


def env_vars(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return set(re.findall(r"\bHIPFIRE_[A-Z0-9_]+\b", text))


def main() -> int:
    canonical = DOC.read_text(encoding="utf-8", errors="ignore")
    missing: list[tuple[str, str]] = []
    for path in REFERENCE_DOCS:
        for name in sorted(env_vars(path)):
            if name not in canonical:
                missing.append((path.relative_to(ROOT).as_posix(), name))

    if missing:
        print("docs/env-vars.md is missing HIPFIRE_* vars referenced by top-level docs:")
        for path, name in missing:
            print(f"  {path}: {name}")
        return 1

    direct_reads: list[tuple[str, str]] = []
    for path in (ROOT / "crates").rglob("*.rs"):
        relative = path.relative_to(ROOT).as_posix()
        if (
            relative in CENTRAL_CONFIG_READERS
            or path.name == "build.rs"
            or "examples" in path.parts
            or "tests" in path.parts
        ):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for name in DIRECT_ENV.findall(text):
            if name not in BOOTSTRAP_ENV:
                direct_reads.append((relative, name))

    if direct_reads:
        print("production HIPFIRE_* reads must use ProcessConfig/TOML:")
        for path, name in sorted(direct_reads):
            print(f"  {path}: {name}")
        return 1

    print(
        "env-docs: references covered; production HIPFIRE_* reads are config-owned"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
