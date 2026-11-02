# Railgun `SealedTable`: certifying kernels that read through a loaded pointer table

Source-path convention: crate references omit the leading `crates/`. Facts below were produced by the code in this change; the evidence objects are the exact gfx1201 code objects the Raon railgun K0 probe dumped (`railgun-cert/tests/fixtures/moe_*.gfx1201.hsaco`).

## 1. The gap

The Raon K0 probe (`raon-railgun/PROBE.md` §3–4) could not certify two decode kernels:

* `gemv_mq4g256v2_moe_gate_up_k8_indexed_batched`
* `gemv_mq4g256v2_moe_down_k8_indexed_batched_expanded`

Both do

```c
int e = topk_indices[...];                  // mutable data, changes every step
if ((unsigned)e >= (unsigned)n_experts) { /* zero the output */ return; }
const char* A = (const char*)expert_ptrs[e];   // 64-bit pointer loaded from a table
... only loads through A ...
```

The unguarded analysis (`peacemaker_ir::passes::kernargs::analyze`) reports `Unknown`, rule `base-memory-derived`, on every access through `A` (14 in gate/up, 24 in down). The probe turned that into `proven` with a forced status. That is not a certificate. `IndexedBind` (design A §A9) does not apply: its index is an *immutable `I32`* feeding `pointer argument + displacement`, and here the *pointer itself* is loaded.

`SealedTable` is a separate conditional certificate for exactly this shape. It is model-agnostic: no kernel or model name appears in any rule or API (Flash-Next reuses it unchanged).

## 2. The certificate

A kernel is certified `Conditional`, witness kind `sealed_table`, when all of the following hold. Anything else stays `Unknown` with the exact rule.

For each **entry load** `L` (a 64-bit `s_load_b64` / `global_load_b64`):

1. **Shape.** Its address is *exactly* `T + (ext(i) << 3)`: `T` a pointer kernel argument taken whole, `i` a 32-bit value that is either the value one instruction loaded (`Load(inst)`) or a kernarg scalar, extended (sign or zero) to 64 bits, scaled by the 8-byte entry. Any other term, scale, offset, second pointer, or half-recombination refuses (`sealed-entry-shape`, `sealed-entry-index`, `sealed-entry-not-64bit`).
2. **Guard.** On **every feasible path from the kernel entry to `L`**, a scalar *unsigned* compare established `i < n`, where `n` is a kernarg scalar or a constant. (`sealed-index-unguarded` otherwise.)
3. **Not in a loop.** `L` and every block that can reach it lie outside every CFG cycle, so the paths are finite and `i` names one dynamic value (`sealed-entry-in-cycle`, `sealed-guard-cyclic-prefix`).

For each **access derived from an entry** (any memory access whose base is, as a whole 64-bit pointer, a value derived from some `L`):

4. It is a plain load, `vmem_load` or `smem_load` (global, flat, buffer, scalar). Stores, atomics and LDS-DMA refuse (`sealed-table-write`, `sealed-table-access-class`).
5. The loaded pointer is **never**: stored as data, used in an LDS address, used in a non-table address (even as an offset), or combined with a different base (an argument pointer, a PC-relative base, a loaded value) (`sealed-table-value-stored`, `sealed-table-value-in-address`, `sealed-table-value-in-offset`, `sealed-table-base-mixed`).
6. A value loaded *through* the pointer is data. Using it as an address base is ordinary `base-memory-derived`, still `Unknown` (no `P[Q[i]]`).

Finally every access of the kernel must resolve: the other (argument-based) accesses are the ordinary analysis, and any unresolved access keeps the whole kernel `Unknown`.

### 2.1 How the guard is proved (not guessed)

LLVM often materialises the guard as a flag: `ok = i < n ? -1 : 0; ... s_and_b32 t, ok, exec_lo; s_cselect_b32 t, 1, 0; s_cmp_lg_u32 t, 1; s_cbranch_scc1 exit`. A plain dominance test on the compare edge fails (the diamond rejoins), and the join of the flag loses it. The analyzer therefore **replays each entry-to-load path** with its own transfer function (`step`), from the entry state, one path at a time:

* The state carries exact constants through scalar logic (`s_and/or/xor/andn/orn`, including `x & 0` and `x | -1`, so a flag that is zero on the unguarded path stays zero through `& exec_lo`), SCC from those results and from compares, `s_cselect` with a decided SCC, and `vcc_lo` constants (wave32 `vccz`/`vccnz`; the compiler already relies on VCCZ reflecting `vcc_lo`).
* A conditional branch whose condition the replay decides keeps only its feasible edge. The unguarded path of the flag diamond is infeasible, so it does not reach the load.
* A scalar `s_cmp_{lt,gt,le,ge}_u32` leaves an exact witness of its SCC (`Node::Ltu{a, b, neg}`); the SCC edge of a branch on it adds the fact `a < b` for the edge it implies. Signed compares, vector compares and `readfirstlane`d indices add nothing (they stay unguarded).
* At the load, the facts about **its own index value** (`Load(inst)`/`K32` identity, by hash-consed expression) whose other side is a kernarg scalar or a constant are intersected over all paths. The bound recorded is one that holds on all of them; the compares are recorded as `guard_pcs`.
* Budgets (4096 paths, 400k steps), an undecodable instruction on a path, or an unreachable load refuse. Nothing is widened.

All of this is only enabled in `analyze_sealed`; the unguarded and guarded analyses are byte-for-byte unchanged (their transfer functions are not touched when `sealed` is off).

### 2.2 Provenance

`Val` gains `tbl`, a bit-set naming the qualified entry load(s) a value may be a derived copy of (it joins and mixes like `roots`, and is dropped by multiplies/shifts/bit-field extraction, i.e. the existing `AddressArithmetic` assumption), and a `TBL` influence flag that survives every operation (used for the sink checks in rule 5). `resolve_base` yields `Base::Table(mask)` only when both address halves carry exactly the same mask and nothing else.

Analysis is two runs: a collecting run finds shape-correct entry loads; each is proved (§2.1); a second run with the qualified loads tagged attributes every access. Unqualified entry loads keep producing plain `base-memory-derived` plus a precise `sealed-*` rule.

## 3. Receipt (schema 9, section 3)

`railgun-corpus/src/receipt.rs`:

* `RECEIPT_SCHEMA_VERSION` 8 → **9**, `RAILGUN_SECTION_VERSION` 2 → **3**. Admission refuses every other version; an older receipt (even one that happens to carry the new fields) is never promoted (`an_older_schema_receipt_is_never_promoted_to_sealed`). Every committed receipt/fixture was regenerated (`railgun/tests/fixtures/native_mq6_hit.json`).
* `KernelRecord.sealed_tables: Vec<SealedTableRecord>` (omitted from serialisation when empty, so unrelated receipts are unchanged apart from the version):

  ```
  SealedTableRecord { table_arg_offset, bound_kind: "kernarg_u32" | "const_u32", bound_value,
                      entry_bytes: 8, entries: [SealedEntryRecord { pc, class, path, cache, width,
                      index_expr: "load@<pc>" | "karg32:<off>", guard_pcs }],
                      accesses: [SealedAccessRecord { pc, class, path, cache, width, entry_pcs }] }
  ```
* `required_witnesses` is exactly the kinds the case fields imply, sorted (`indexed_bind` for `indexed_effects`, `sealed_table` for `sealed_tables`); unknown kinds are rejected (`KNOWN_WITNESSES`, also checked by `WindowManifest::validate`).
* `validate_case` (fail closed): a conditional record needs either guards + indexed effects or sealed tables (a sealed-only record has *no* guards); proven/unknown records carry none of it; every sealed table passes `SealedTableRecord::validate` (known bound kind and alignment, 8-byte entries, every entry guarded, every access a read whose `entry_pcs` name this table's entries); no duplicate `(arg, bound)`.
* `case_digest` is now `railgun-case-v2`: it binds object SHA, device ELF SHA, arch, symbol, status, kernarg size, guards, indexed effects, **sealed tables**, required witnesses. Changing any of them, the object, the device ELF or the arch fails admission (`railgun-cert/tests/sealed_table.rs`, `railgun-corpus/tests/conditional_receipt.rs`).
* The receipt states one more assumption (`SealedTable: …`, §5.2).

`railgun-cert` runs `analyze_sealed` on every kernel whose unguarded analysis is `Unknown` (no symbol table; it grants nothing: the status comes from the proof). If the proof fails but an entry load had the right shape, the receipt's `unresolved` carries the exact `sealed-*` rule instead of the generic one. Obligations: `SealedTableWitness`.

## 4. Runtime: facts, author, witness

`railgun/src/kernel.rs` parses a sealed-only conditional record into `ConditionalFacts.tables` and `KernelNode.table` (separate from `indexed`; it is never folded into `IndexedArgEffect`). `author` adds the entry-load read classes to the table argument's resource (writes make the node `HipDirect`), requires the table pointer and the bound to be real arguments, and leaves the node `Lowering::Conditional { "sealed_table witness not consumed" }`: a non-PM4 full barrier. The cached facts are never mutated; `assume_witnesses(&[…])` derives a program that lowers a node as PM4 only if every kind it needs is assumed.

`railgun/src/sealed_table.rs` holds the witness machinery (see §6 for the owner's calls):

* `SealedTable { allocation, entries, entry_width: 8, content_sha256, pointees, generation }`.
* `SealedTableRegistry::seal(..)` verifies at seal time: byte length = `entries * 8`; pointees non-empty, immutable, disjoint, disjoint from the table allocation; **every entry** is non-null and inside exactly one pointee allocation; no overlapping seal; generation not below the floor a retire raised. It records the SHA-256 of the exact table bytes.
* `bind_sealed_table(inputs, &dyn TableContent)` is the only constructor of `SealedTableWitness`. For each sealed node it checks: the table argument's resource is a sealed table by **exact** base and bytes; the kernarg pointer in the encoded image is the allocation base; the allocation generation is the sealed generation; the bound (read from the encoded kernarg image, or the constant) is `<= entries`; the live table digest (read through `TableContent`) equals the sealed digest; and, for the whole program, **no node writes** the table allocation or any pointee allocation (store, atomic, copy/fill destination, partial overlap included; a node with unknown effects is refused). Pointees are read-only union effects: they are not program resources and add no hazard edge; the bind's refusal is what makes that sound.
* `PreparedWindow::attach_sealed_table_witness` consumes the witness; the plan refuses `NotPm4` until it does; `submit` re-checks generation, extent, window/shape generation, kernarg image and grid, then leases the tables so `retire` is refused while a window is in flight; unknown quiescence quarantines the table.
* `SealedTableWitness::discharged_cases()` yields `(node, case_digest, "sealed_table")`, the triples `WindowManifest::admit_with` expects (`ConditionalWitness`).

## 5. Trust

### 5.1 What the proof establishes (machine-checked)

For the exact object (receipt keyed by object SHA + device ELF SHA + arch + case digest): the kernel reads `table[i]` only with `i < n` on every path, and everything it derives from an entry is only read through; no entry-derived value is written, stored, recombined or re-indexed; every other access resolves ordinarily.

### 5.2 What the loader/owner must guarantee (not provable by the kernel, stated in the receipt)

1. **Weights are immutable for as long as the table is sealed.** Neither the table allocation nor any pointee allocation is written by *anything*: not by this program (the bind checks that), not by another queue, host mapping or later program. `retire` is refused while a window is in flight and raises the generation, so earlier witnesses are stale; nothing in the runtime can see an outside writer, so the owner must not enqueue one.
2. **The table content is the sealed content.** The sealed digest is over the exact bytes at seal time; every bind re-reads the live bytes (`TableContent`) and refuses a difference. (Submit re-checks generation and extent, not the digest: a write to the table after bind is a violation of 1.)
3. **Every entry points into a registered pointee**, checked at seal. The kernel's own assumption (named in the receipt) is that its accesses stay inside the pointee the entry names (the same `InBoundsNoAlias` style assumption every allocation-wide effect rests on) and that halves of entries are never recombined across entries.
4. **The bound `n` the launch passes is `<=` the entry count.** Checked at bind from the encoded kernarg image.

The bound is a hazard-freedom argument, not a memory-safety one: entries `>= n` are never read, so an unused (or stale) table tail cannot be reached, and a stale `topk_indices` only selects among valid pointees or skips.

## 6. API a model owner calls

At load time (once per pointer table):

```rust
use railgun::sealed_table::{Pointee, SealedTableRegistry};
use railgun::Binding;

// weights: immutable allocations; table: the uploaded pointer table.
let registry = SealedTableRegistry::new();           // one per process, cloned into windows
let sealed = registry.seal(
    table_allocation,                                 // Binding { base, bytes }
    n_entries,                                        // u32
    &table_bytes,                                     // the exact bytes uploaded (LE u64 pointers)
    &[Pointee::immutable(weights_a), Pointee::immutable(weights_b)],
    allocation_generation,                            // the allocation registry generation
)?;                                                   // refuses an entry outside every pointee, null, overlap...
```

Per window (after `PreparedWindow::new`/bind of the program, before submit):

```rust
// `TableContent` downloads the live table bytes (the only place a GPU is touched).
prepared.attach_sealed_table_witness(&mut bound, &registry, &content)?;
// bound.sealed_table_witness().discharged_cases() feeds WindowManifest::admit_with.
```

At unload: `registry.retire(table_allocation.base)?` (refused while a window using it is in flight; advances the generation).

`railgun::sealed_table::tests::owner_seals_a_pointer_table_and_binds` and the `SealedTableRegistry::seal` doctest are the unit-tested examples. Sealing the real Raon expert tables, implementing `TableContent` over the device, and issuing the witness in the probe is a later slice.

## 7. Evidence

`railgun-cert coverage` over the Raon manifest (`run/objs/coverage.tsv`): both MoE kernels are `conditional` with `required_witnesses = [sealed_table]`; the 24 other previously `proven` rows are unchanged. Gate/up: table arg +0, bound kernarg +52 (`n_experts`), one entry load, guarded by one compare, 14 reads; down: bound +44, 24 reads (buffer loads through a V# built from the entry). Analyzer tests (synthetic llvm-mc kernels) cover the positive shapes, the flag diamond and each refusal: store through the base, atomic through the base, unguarded, guard on another value, signed and reversed guards, bypassed guard, a broken flag diamond, base reused as an index, pointer stored, pointer mixed with an argument, stride/offset the certificate does not cover, entry in a loop.

Regression check against the probe's receipts: of the 39 kernel records common to the probe run and this one, the 37 other records (all `proven`) are JSON-identical to the probe's; only the two MoE records changed (`unknown` → `conditional`). The 24 rows the probe certified `proven` stay `proven`.

## 8. Signed guards, null guards and entry-or-argument bases (second slice)

`ffn_situ_rows_f32` (awq mode 2) needed three more exact facts; each is a proof, none a guess.

**Flag algebra.** The sealed analysis keeps an exact boolean witness of SCC and of lane-uniform masks: atoms `a <u b`, `a <s b`, `a == b` over exact 32-bit operands and `w == 0` over a matched 64-bit pair (`Ent(load)` for a table entry), combined by `not`/`or`/`and`; a value holding a flag is `Mask(f)` (`s_cselect s, -1, 0`), `Bit(f)` (`s_cselect s, 1, 0`) or `NzFlag(f)` (`exec_lo & Mask(f)`). A decided branch edge (SCC, or wave32 VCC through `NzFlag`) adds the atoms the edge implies (`not`, a true `and`, a false `or`); nothing else is concluded.

**EXEC known nonzero.** `exec_lo & Mask(f)` is nonzero exactly when `f` holds only if EXEC is nonzero. `State.exec_nz` is set on a decided `s_cbranch_execz` fall-through / `s_cbranch_execnz` taken edge (wave32), survives only `s_or_b32 exec_lo, exec_lo, x`, is cleared by every other EXEC write, and a saved copy `s_mov_b32 s, exec_lo` of a nonzero EXEC is a nonzero value, so `s_or exec_lo, exec_lo, s` restores it. Joins keep it only when both sides know it.

**Signed guard.** `0 <=s e` and `e <s n` on every path give `e <u n`: the entry's bound is `n` (a kernarg scalar or constant), the compares of both halves are recorded as `guard_pcs`. Either half missing, a guard on another register, or no EXEC fact leaves it unguarded.

**Entry or argument.** A read whose base may be an entry on one path and an argument pointer on another (`s_mov_b64 s[4:5], awq_a` in mode 1, the table entry in mode 2) is both: an ordinary read of the argument and a sealed read of the entry (`Base::Both`). Stores through it still refuse. The receipt's `SealedTable` assumption adds that an entry is never added to another pointer.

**Null guard (`null_guarded`).** Per table, proved after the final run: every sealed read of the table is on a path where the exact `entry == 0` atom is false (`s_cmp_lg_u64 entry, 0`, directly or through `Mask`/`Bit`/`NzFlag`), for the entry load whose pointer the base carries on that path. It needs the access's ancestors outside every cycle; reads inside loops (the MoE kernels) are not null guarded. `SealedTableRecord.null_guarded` (omitted when false) is bound by the case digest. Runtime: `SealedTableRegistry::seal_with(.., NullEntries::Allow)` accepts null entries (every other entry still has to lie in a pointee) and records `has_null_entries`; the bind refuses `NullEntryUnguarded` unless the node's table fact is `null_guarded`. `seal(..)` is `seal_with(.., Refuse)`. The owner passes `Allow` only from the receipt's `null_guarded` of the kernels that will read the table.

**Vacuous launches.** `bind_sealed_table` evaluates each table's recorded bound first (the encoded image's kernarg scalar, or the receipt constant). A bound of 0 makes the certified guard (`i <u 0`, or `0 <=s i <s 0`) unsatisfiable, so no entry load can execute: the case is vacuous for that launch and needs no seal, no allocation equality and no content digest (the node is still discharged under its case digest; `BoundTableNode.vacuous` lists those tables, and the argument's own read stays an ordinary effect). This is the "plain vector" launch of a kernel whose table argument is a vector in another mode. The bound lives in the kernarg image the witness digests, so raising it after the bind voids the witness at submit; every nonzero bound is bound strictly (seal, exact allocation, base pointer, bound <= entries, digest).

Not covered: `ffn_swiglu_clamp_rows_f32` (same source family, other register allocation): its mode-0 path reaches the shared read with the kernarg-segment pointer still in the low half, so the path-insensitive provenance mixes it (`sealed-table-base-mixed`). Proving that needs per-access path-sensitive provenance, not a wider rule.

## 9. Not covered (stays `Unknown`)

Tables indexed by vector values (no per-lane facts), `P[Q[i]]`, entries with a non-8-byte stride, an entry load inside a loop, a table with null entries read by a kernel without a proved null guard, pointer comparisons feeding addresses, kernarg-preloaded indices or bounds (no exact identity). Each refuses with its own rule; widening any of them needs its own proof.
