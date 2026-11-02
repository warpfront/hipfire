<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev> -->

# radiowave — vendored crate

This crate is a **vendored copy**, not hipfire-original code.

| | |
|---|---|
| Upstream | https://github.com/Kaden-Schutt/redline |
| Upstream path | `crates/radiowave` |
| Vendored at commit | `f4a0994e2315645b74eef8dec9305c58d252e699` |
| Vendored on | 2026-07-25 |
| Branch | `ds4-gfx1151-opt` |
| License | Apache-2.0 (NOT hipfire's workspace MIT) |

## Why vendored rather than a git dependency

hipfire's CI runs `cargo build --release --workspace --all-targets --locked`
with no network guarantee. A git dependency on a second repository would make
that gate depend on network reachability and on the upstream repo's visibility.
Vendoring keeps the required CI build self-contained.

The cost is drift: there are now two copies. Treat **upstream as the source of
truth**. Land changes there first, then re-vendor.

## Manifest divergence from upstream

The upstream `Cargo.toml` inherits seven keys from redline's
`[workspace.package]`. hipfire's workspace cannot supply them, so this copy
pins them explicitly:

| key | upstream (inherited) | here (explicit) | why |
|---|---|---|---|
| `edition` | 2024 | 2024 | hipfire workspace is 2021 |
| `license` | Apache-2.0 | Apache-2.0 | hipfire workspace is MIT — must not inherit |
| `rust-version` | 1.85 | 1.85 | not defined in hipfire workspace |
| `authors` / `repository` / `homepage` | redline | redline | not defined in hipfire workspace |
| `version` | 0.1.0 (redline) | 0.1.0 | keeps upstream version visible; hipfire is 0.4.2 |

Except for the scoped additions listed in the next section, everything under
`src/`, `include/` and `tests/` is byte-identical to upstream at the pinned
commit `f4a0994…` (the `Vendored at commit` row above is **not** advanced by a
scoped addition). If you change a file here, record it in this table or the
next re-vendor will silently revert it.

## Scoped additive re-vendors (not a whole-crate re-pin)

| Added / changed here | Upstream source | Notes |
|---|---|---|
| `src/union.rs` (new), `tests/union_identity.rs` (new), `README.md` | upstream commit `e073038d6bcfac3934e3c1a62f9ec2659c3abaf2` (`feat(radiowave): add union identity records`), plus the upstream test-compat fix commit `274c196359dcc4a5f76ea41e5226cf720f5f93e4` (`test(radiowave): keep union_identity to the upstream CompileManifest API`), made **upstream first** in the private worktree `pm-union/UnionU0-upstream`, branch `pm-union-u0-20261011`, parented on upstream commit `98d35126f3b5574d845d55ad5ca01efb939a1e72`; not pushed to a remote | Byte-copied from that commit; `cmp`-identical when copied. `README.md` here was identical to the upstream README before the commit, so it re-vendors whole. |
| `tests/downstream_peacemaker_manifest.rs` (new, **downstream-only**) | none; not in upstream | Upstream `CompileManifest` has no `peacemaker` field and upstream has no `PeacemakerRecord`, so the Peacemaker round-trip/masquerade checks live here instead of the shared `tests/union_identity.rs`. Do not send upstream; keep it when re-vendoring. |
| `src/lib.rs`: one added line, `pub mod union;` | same upstream commit adds the identical line | **`src/lib.rs` was NOT re-vendored.** This copy still differs from upstream parent `98d3512` in unrelated ways, as do `src/main.rs`, `src/oracle.rs`, `src/recipes_fp8.rs` and three `tests/artifacts/*.json`; those differences are pre-existing and untouched. The whole crate is therefore **not** claimed identical to any single upstream commit. |
| `src/lib.rs`: Windows executable-suffix tool lookup and its CPU-only regression test (#835, original commit `2284a98cd825415ae2391b6bfcd24b077fa3bf93`) | Upstream port not established by this review | Downstream lookup tries the bare name then the host suffix in each SDK directory. The corresponding fix still needs to land in `Kaden-Schutt/redline`; preserve it during re-vendoring until then. |

The union schema is a separate serde module: `CompileManifest`,
`PeacemakerRecord` and `OracleReceipt` (and every existing downstream user:
`peacemaker-isa`, `railgun-cert`, `rdna-compute`) are unchanged. The module is
CPU-only and adds no dependency.

To re-vendor the union module later, copy `src/union.rs`,
`tests/union_identity.rs` and `README.md` from the upstream commit above (or its
descendant) and re-add `pub mod union;` to this copy's `src/lib.rs` if it is
absent; do not copy upstream `src/lib.rs` over this one without reconciling the
pre-existing differences first.

## Re-vendoring

```bash
SRC=../redline-pub                 # or a fresh clone of the upstream repo
rm -rf crates/radiowave/{src,include,tests,README.md}
cp -r "$SRC/crates/radiowave"/{src,include,tests,README.md} crates/radiowave/
# then update the commit hash above and in crates/radiowave/Cargo.toml
```

Do **not** overwrite `Cargo.toml` or this file — both carry hipfire-local
divergence.

## What this crate is for here

Radiowave owns the policy boundary between HIP source and LLVM/AMDGPU: it
injects reviewed source-level lowering rules, invokes hipcc, inspects the
emitted code object, and records reproducible build evidence.

`rdna-compute` now uses Radiowave's existing-code-object certification path to
bind gfx1151 JIT artifacts to exact hashes and fail-closed mutable-read cache
classes before retained replay consumes them. Recipe selection remains future
work — in particular moving the per-stride cross-lane lowering table into
`radiowave::recipes` so new arch ports inherit it instead of re-deriving it.
