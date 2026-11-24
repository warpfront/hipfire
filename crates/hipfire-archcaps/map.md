# hipfire-archcaps — map

> **Status:** `production`
> **Layer:** 0 — data-only GPU architecture identity; no dependencies. See the layering table in [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md).

## Purpose

`GfxArch`: the single exact GPU architecture identity (`name()`, `parse()`, `ALL`). Identity only: capabilities and family groupings (`is_rdna3`, `is_cdna3`, …) stay in `rdna-compute` until they move with the facts table.

## Gotchas

- `parse` is exact: `gfx1201:xnack-` and near neighbours (`gfx1200` vs `gfx1201`) do not alias.
- `hipfire-kernel-abi`, `peacemaker-ir` and `peacemaker-cdna` are layer 0 and cannot depend on this crate; they keep their own `Arch` enums, tied to `GfxArch` by a name-agreement test elsewhere.

## Crate map

<!-- crate-map:generated:begin -->

Modules, exported symbols and dependencies are generated, not committed: see
`map.generated.md` in this directory (`cargo xtask regen` writes it).

<!-- crate-map:generated:end -->
