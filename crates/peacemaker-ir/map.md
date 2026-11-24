# peacemaker-ir — map

> **Status:** `production` / `research` / `legacy` — pick exactly one
> (vocabulary owned by [`docs/GLOSSARY.md`](../../docs/GLOSSARY.md)).
> **Layer:** see the layering table in
> [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md) — do not restate it here.

## Purpose

<!-- hand-written: what this crate owns and why it exists. Prefer pointing at
     the `//!` docs in `src/lib.rs` over copying prose. -->

## Gotchas

<!-- hand-written: what not to do here; traps a reader will hit. -->

The complete RDNA XML catalogs are `isa/{gfx1100,gfx1151,gfx12}.xml.jsonl`.
Regenerate with `scripts/pm-rdna-xml.py --xml-dir <AMD XML directory>
--llvm-mc <ROCm llvm-mc>`. Each catalog begins with the source SHA-256,
toolchain version, encoding bitmaps, and condition expressions. Subsequent rows
retain every source instruction encoding, including duplicate XML entries:
`variant` is the original instruction-local encoding ordinal. Operand input,
output, implicit, size, and format information is retained; memory operands and
branch/terminator flags are explicit.

The catalogs feed the variant-aware runtime field codec through
`build.rs`, which projects them into `$OUT_DIR/*.runtime.tbl` at build time
(rerun only when a catalog changes). The original census tables
and their ignored-field exceptions remain the byte-identical regression set.
An oracle status other than `verified` is a disagreement or an unavailable
candidate, never proof of LLVM non-support. `isa/sop1-overrides.json` records
LLVM-derived corrections and exact restricted source domains while preserving
the malformed XML condition. Numeric semantics remain separate: XML field
decoding does not qualify a new variant for emulator execution.
VOPD selection combines independently resolved halves only when their shared
fields have identical physical layouts; an inherent literal extends the packet
without changing the peer's row key. Reconstruction derives both halves' effects
from the named fields, not from provenance. Typed/census diagnostics take
precedence over failed generic decoding. DPP8FI selector `0xea` remains refused
by instruction name until a matching LLVM-verified selector specimen exists.

## Crate map

<!-- crate-map:generated:begin -->

Modules, exported symbols and dependencies are generated, not committed: see
`map.generated.md` in this directory (`cargo xtask regen` writes it).

<!-- crate-map:generated:end -->
