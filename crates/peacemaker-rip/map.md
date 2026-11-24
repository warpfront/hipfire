# peacemaker-rip — map

> **Status:** `production` (vocabulary owned by [`docs/GLOSSARY.md`](../../docs/GLOSSARY.md)).
> **Layer:** see the layering table in
> [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md) — do not restate it here.

## Purpose

The standalone `.rip` front end: Rust-shaped kernel scripts parsed with `syn`
and executed by a tree-walking interpreter against the checked
`peacemaker_isa` builder, with no rustc and no cargo rebuild. Owns the script
language and its diagnostics (`diag`), the builder vocabulary it exposes
(`native`), LDS/barrier/wait transitions through `peacemaker-author`'s sealed
driver, emission, and the agent-facing facts verbs (`features`: resources,
`cost_lint` schedule features, emulator executability). See `src/lib.rs` for
the source contract and the evaluated-language subset.
gfx942 source uses the same Workgroup driver but lowers through the checked CDNA
builder. `validate_cdna` checks native CDNA footprints, final byte-definedness,
hazards and CFG waits. `run` uses the CDNA wave64 simulator for that target.

## Gotchas

- The committed `.rip` kernels are byte-gated against their Rust twins
  (`qsa_*.rip.rs`): emitted assembly must stay identical, so a vocabulary or
  language change is only safe if `build` output is unchanged. `build --certify`
  covers assembly, link, M7 and the receipt cache.
- `crates/peacemaker-rip/tests/vocabulary.rs` is the executable inventory of
  what is still unreachable from a script, and why. Add there before adding a
  binding from memory.
- `const VARIANTS` / `const ARCHES` are reserved for `check` target discovery;
  a script's own tables must use another name (see `qsa_topk.rip`'s
  `HIST_VARIANTS`).
- `control` is the one instruction path that bypasses the front-end
  `validate::instruction` layer, because symbolic branch labels do not
  round-trip through text. Do not "fix" that by parsing labels.
- Emulator executability is decided by `peacemaker-ir`'s `semantics.tbl`, not
  by handler code; a table entry without a handler (or the reverse) is a bug
  there.

## Crate map

<!-- crate-map:generated:begin -->

Modules, exported symbols and dependencies are generated, not committed: see
`map.generated.md` in this directory (`cargo xtask regen` writes it).

<!-- crate-map:generated:end -->
