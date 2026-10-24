# Complete no-GPU follow-up for PR #939

Measured source: `638a350fb619477a5dda563e04942af2ddd4cb23`.
Pinned comparison base: `beta@3a5ee848c161b87f28d2437c3cb977de736b4a9c`.
This checkpoint adds evidence only, not production code or a gate change.

## Results (not all green)

| Route | Result |
|---|---|
| `scripts/no-gpu-ci.sh` | **FAIL, exit 101**. Workspace/example check passes; `rdna-compute --lib` reports 1,020 passed, 2 failed, 28 ignored, then the script correctly stops. |
| `cargo test --workspace --lib --locked --no-fail-fast` | **FAIL, exit 101**. All 58 library targets finish: **8,292 passed, 2 failed, 86 ignored**; only `rdna-compute --lib` fails. |
| Clean pinned-beta registry control | Same two failures; **23 passed, 2 failed** in the 25 registry tests. |
| Explicit continuation of every stage after the stopped registry command | All stages pass except the first Python run under the user's Anaconda environment; its full clean-system-Python rerun passes. The original failure remains recorded. |
| Launcher ABI / coverage | **62/62**, including both full-source checks; 1,056.24 seconds. |
| CLI bin tests inside control-plane run | **770/770**. Client/config/registry/TUI and included integration/doc tests also pass. |
| Complete prescribed Python pytest list, clean environment | **818/818**, 105.26 seconds. No `-k`, deselection, changed assertion or test patch. |
| Remaining Python/script stages | Redline 179, JEV 45, install 17 and uninstall 6 tests pass; GPU-lock, format-sweep, env/lifecycle and leaked-process checks all exit 0. |

The 86 ignored library cases are the repository's existing ignored tests, not
new exclusions. Do not add counts from the separate commands: feature
unification and overlapping tests make them different runs, not disjoint sets.
The normal-route GPU and lab-specific CPU evidence remains at the
[streaming checkpoint](../2026-10-22-tiered-kv-streaming/README.md).

## Remaining failures: unchanged registry export/index closure

Both candidate and clean beta fail:

- `kernel_registry::tests::source_export_evaluator_reproduces_the_sealed_gfx1100_object_and_index_gap`
- `kernel_registry::tests::tensor_ops_index_is_the_source_export_closure_per_arch`

The source now exports `hyper_write_bf16x8_gfx1201`, but the tensor-ops registry
and sealed export expectation do not contain it. It is emitted without an
architecture guard, so even the gfx1100 source-export comparison sees it.
This PR does not change those source, index or fixture files.

The clean beta snapshot's relevant Git blob identities match pinned beta:

- `crates/rdna-compute/src/kernel_registry.rs`:
  `4c8586500ceb010c69549a605c3420e26538afce`.
- `kernels/src/tensor_ops.hip`:
  `98eae94c968c0c221be2a3254f88eb6c037e244a`.
- `crates/hipfire-loader/src/admission.rs`:
  `1b80b5f0cda5a37c3d88914843377c6e5603976a`.

These are failures, not waivers. The complete no-GPU and workspace-library
**pass** checkboxes remain unchecked. No registry fixture, admission,
assertion, scanner exclusion or tolerance is weakened to get green results.

## Correction to the earlier loader-failure attribution

The current full-workspace run passes all **186 loader tests**. The former
`gfx1151_fwht_k_override_refuses_instead_of_legacy_fallback` failure is also
tested alone using the **same binary**, changing only configuration isolation:

- Fresh `HIPFIRE_HOME`: **passes**.
- Inherited user configuration (`kv_cache = "asym3"`): **fails**, reproducing
  the earlier `unwrap_err()` on `Ok(())` result.

Test-binary SHA-256:
`f6fb2ea45eb80c972ea953a4977cdbbf36a313302160b158378e417c2497ef38`.

Thus the earlier result should not be listed as a current intrinsic loader
or tiered-KV regression. The test's implicit mode consults process/user
configuration; the prescribed no-GPU isolation is important. User settings
were read but never modified. Both outcomes are preserved, not rewritten.

## Python environment control

The first full pytest run under Anaconda Python 3.12.7 reports **796 passed,
22 failed**:

- 21 hw-gate subprocess tests import the repository's `select.py` instead of
  the standard-library extension module, then fail at `select.select`.
- One optional Torch parity case cannot import Torch because the installed
  amdsmi Python binding requests a symbol absent from the active
  `/opt/rocm/lib/libamd_smi.so`.

The complete **same** test list is rerun in an isolated `/usr/bin/python3`
3.12.3 venv with pytest 8.4.1 and NumPy 2.1.3 (plus their dependencies).
On that interpreter `select` is built in, matching the Ubuntu-style CI
environment, and no global Anaconda/ROCm packages are changed. All 818 cases
pass. Torch is not installed there: its existing optional `ImportError`
return is taken; this does **not** establish Torch CPU-parity coverage or fix
the local Torch/amdsmi installation. The environment-sensitive code and
tests are unchanged from pinned beta.

All GPUs are hidden (`ROCR_VISIBLE_DEVICES=-1`, `HIP_VISIBLE_DEVICES=-1`).
The full-library run, continuation and clean Python rerun each use separate
temporary configuration/lock directories. Local ephemeral listeners are
permitted outside the filesystem/network sandbox, so CLI failures are not
artifacts of denied local HTTP access.

## Raw artifacts and reproduction

`full-ci-raw.tar.gz` preserves the official failed wrapper, complete workspace
output, clean-base control, both loader configurations, original Anaconda
pytest failure, successful clean pytest rerun and all continuation logs with
per-stage exit codes. It also contains the exact diagnostic continuation
script. SHA-256:
`e4a0e555ca1b7f0f4769c36342f821b976a7b49f6631d5f8e6c68bc5516b3be3`.

The continuation is explicitly **not a replacement gate**. It runs every
command after the failed registry command separately to expose remaining
failures; its results never convert the original script's exit 101 to pass.
Its original driver exits 1 for the Anaconda pytest failure. The subsequent
clean pytest log is a new run, not an edited old log. No raw log is hand-edited.

Run the two original routes on the pinned source with isolated config and all
GPUs hidden. Allow local ephemeral HTTP listeners:

```sh
scripts/no-gpu-ci.sh

# Use fresh task-specific directories, not the caller's existing config.
validation_home=$(mktemp -d /tmp/hipfire-lib-validation-home.XXXXXX)
validation_locks=$(mktemp -d /tmp/hipfire-lib-validation-locks.XXXXXX)
ROCR_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 HIPFIRE_TEST_REQUIRE_ROCM=0 \
  HIPFIRE_HOME="$validation_home" HIPFIRE_LOCK_DIR="$validation_locks" \
  cargo test --workspace --lib --locked --no-fail-fast
```

For Python use the exact `PYTEST_PATHS` from `scripts/no-gpu-ci.sh`, with the
isolated system interpreter above. No source changes or optional hardware
validation are implied by this CPU-only checkpoint. The PR remains Draft;
no human approval or merge-ready CI result is claimed.
