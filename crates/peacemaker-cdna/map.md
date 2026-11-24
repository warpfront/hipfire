# peacemaker-cdna — map

> **Status:** `research` (default-off: compiled in transitively, never selected at runtime).
> **Layer:** Kernel authoring (build time / CPU emulation) — the CDNA instruction model beneath the `cdna` arm of [`peacemaker-isa`](../peacemaker-isa/map.md); see the layering table in
> [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md).

## Purpose

CDNA (CDNA3 and later; gfx942 / MI300X first, wave64) instruction model for PeaceMaker: the `Arch` variant enum with a per-variant capability table, typed instructions with an encoder/decoder and an `llvm-mc`-compatible printer per format, FP8/BF8 (FNUZ and OCP) conversions, the gfx90a+ kernel descriptor layout, and a pure-CPU wave64 workgroup emulator (`sim`). AMD's complete CDNA3 and CDNA4 XML encoding/operand tables supplement the hand-written typed API. gfx942 supports encoding; gfx950 is decode-only. Numeric semantics remain hand-written and a generated instruction without them refuses simulation by name.

## Gotchas

- Opt-in only: direct authoring consumers are `peacemaker-isa`'s `cdna` arm (`peacemaker_isa::cdna`, `Arch::Gfx942`) and `peacemaker-rip`'s native text validation and CPU runner. `rdna-compute` depends transitively on the model, but serving does not select a new kernel or dispatch path as a result.
- `oracle` shells out to ROCm `llvm-mc` / `llvm-objdump` for tests only; nothing in the production path runs a ROCm tool.
- `text::parse_line` is the typed text assembler (canonical `Display` and llvm-mc spellings into `Inst`, validated by the real encoder). It owns no opcode numbers: mnemonics resolve through the `*Op::ALL` tables, so a mnemonic missing from a table is an `unknown mnemonic` error until the table gains the row. Branch labels are the caller's.
- The emulator is a test oracle, not hardware evidence: the on-card harness is `peacemaker-isa`'s `cdna_card_tierc` example ([runbook](../peacemaker-isa/tools/CDNA-CARD-RUNBOOK.md)) and its receipts are under [`docs/pm-cdna/card`](../../docs/pm-cdna/card).
- `Inst::decode` tries the typed tables first; only an opcode the typed table has no entry for (typed refusals of opcodes it owns stand) falls back to the generated tables as `Inst::Xml(XmlInst)`. A generated instruction carries a footprint, memory class (`MemKind`, including `SmemStore`/`SmemAtomic`/`SmemAtomicRtn`) and branch offset derived from its XML row, never a default; the emulator refuses it by name (`SimError::Unsupported`), `Inst::unsupported()` gives the same text to other consumers, and `Inst::verification(arch)` reports whether the XML row of any instruction (typed or generated) is LLVM-verified: emission and seals must refuse an unverified row. gfx950 (`Arch::Gfx950`) is decode-only: all its instructions decode as `Inst::Xml`, nothing encodes, `text::parse_line` refuses it.
- XML regeneration: `scripts/generate-cdna-tables.py` in this crate reads `amdgpu_isa_cdna3.xml` and `amdgpu_isa_cdna4.xml`; generated tables pin their source SHA-256. `tools/crosscheck_xml.py --xml PATH --arch gfx942 --output tools/cdna3-llvm.json` (and gfx950/CDNA4) checks every row against LLVM independently. The committed receipts cover 2,465 CDNA3 rows (2,435 verified) and 2,568 CDNA4 rows (2,536 verified). Every other row remains explicitly unverified with the LLVM diagnostic: it can be recognized but cannot be emitted or sealed. Promotion requires a hardware receipt, not a relaxed parser or an allowlist. A receipt promotes instances, never the row: a `FieldReceipt` one proven field combination (the MI300X oracle receipt path and sha256 it records), an `AliasReceipt` the canonical instances of a typed model when `llvm-mc` assembles the row's opcode under another mnemonic (gfx942 `V_MAD_MIX_F32` is llvm's `v_fma_mix_f32`: VGPR/one SGPR/inline sources, `NEG_HI` = 0, swept exhaustively against llvm-mc in `tests/enc_vop3p.rs` and recording the MI300X `cdna_mix_hw_oracle` receipt path and sha256 like a `FieldReceipt`). `|src|` (`NEG_HI`), literal, AGPR and two-scalar mix forms, `V_MAD_MIXLO_F16`/`V_MAD_MIXHI_F16` and gfx950 stay unverified.
- Generated outputs: the row manifests `xml-manifest-cdna{3,4}.json` are not committed. Nothing reads them. `scripts/generate-cdna-tables.py --xml-dir <XML dir> --manifest-dir DIR` reproduces them byte for byte, and each `src/generated_cdna*.rs` header pins their sha256 (`// Manifest`). `../../scripts/check-pm-generated-pins.py` runs in CI and needs no XML. It fails when a table's generator, receipt or XML pin no longer matches the committed inputs. The generated tables and receipts are `linguist-generated` (`.gitattributes`) and excluded from ripgrep/agent search (`.ignore`). Their paths are unchanged.
- gfx942 mixed FMA: `V_MAD_MIX_F32` is LLVM's `v_fma_mix_f32`, not an unfused MAD. The instance receipt is bound to `peacemaker-isa/tests/fixtures/cdna/gfx942_fma_mix_f32_receipt.json` and its strict MI300X oracle. Half-selected float inline 1.0 materializes as `0x00003c00` (high half zero); integer inline constants remain sign-extended B32. NaNs follow source order/invalid-product precedence and MODE.IEEE, and output flush uses tininess after unbounded-exponent RNE, including the exact normal-boundary tie. These measured rules apply only to gfx942 `FmaMixF32`; mixlo/mixhi, NEG_HI/abs, literals, special scalars and gfx950 are not promoted.
- gfx942 scalar `Sop2Op::NorB64` uses the existing SOP2 encoder, wide operand validation, footprint and text/Builder path. Its full-width `!(S0 | S1)` result overwrites SCC with `result != 0`, including when EXEC is empty. The table-wide LLVM codec sweep and scalar edge/random reference tests cover it; `peacemaker-isa/tests/fixtures/cdna/gfx942_nor_b64_receipt.json` binds real MI300X register/result/SCC and destination-alias evidence. It does not widen gfx950 emission support or promote unrelated XML rows.


## Crate map

<!-- crate-map:generated:begin -->

Modules, exported symbols and dependencies are generated, not committed: see
`map.generated.md` in this directory (`cargo xtask regen` writes it).

<!-- crate-map:generated:end -->
