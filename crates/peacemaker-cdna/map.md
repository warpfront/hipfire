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
- gfx942 mixed FMA: `V_MAD_MIX_F32` is LLVM's `v_fma_mix_f32`, not an unfused MAD. The instance receipt is bound to `peacemaker-isa/tests/fixtures/cdna/gfx942_fma_mix_f32_receipt.json` and its strict MI300X oracle. Half-selected float inline 1.0 materializes as `0x00003c00` (high half zero); integer inline constants remain sign-extended B32. NaNs follow source order/invalid-product precedence and MODE.IEEE, and output flush uses tininess after unbounded-exponent RNE, including the exact normal-boundary tie. These measured rules apply only to gfx942 `FmaMixF32`; mixlo/mixhi, NEG_HI/abs, literals, special scalars and gfx950 are not promoted.
- gfx942 scalar `Sop2Op::NorB64` uses the existing SOP2 encoder, wide operand validation, footprint and text/Builder path. Its full-width `!(S0 | S1)` result overwrites SCC with `result != 0`, including when EXEC is empty. The table-wide LLVM codec sweep and scalar edge/random reference tests cover it; `peacemaker-isa/tests/fixtures/cdna/gfx942_nor_b64_receipt.json` binds real MI300X register/result/SCC and destination-alias evidence. It does not widen gfx950 emission support or promote unrelated XML rows.


## Crate map

<!-- crate-map:generated:begin -->

_Generated by `scripts/check-crate-maps.py` from the tree — do not edit inside the markers._

### Modules and exported symbols

- [`src/arch.rs`](src/arch.rs)
  - `ALL`
  - `Arch`
  - `Caps`
  - `caps`
  - `encodes`
  - `Fp8Encoding`
  - `has_mai`
  - `name`
- [`src/enc/ds.rs`](src/enc/ds.rs)
  - `accesses_lds`
  - `decode`
  - `Ds`
  - `encode`
  - `footprint`
  - `from_opcode`
  - `has_addr`
  - `mem_kind`
  - `name`
  - `offset`
  - `opcode`
  - `pair_offsets`
  - `split_offset`
  - `two_addr`
  - `widths`
- [`src/enc/flat.rs`](src/enc/flat.rs)
  - `decode`
  - `encode`
  - `Flat`
  - `FlatSeg`
  - `footprint`
  - `from_opcode`
  - `mem_kind`
  - `name`
  - `opcode`
  - `widths`
- [`src/enc/mubuf.rs`](src/enc/mubuf.rs)
  - `data_dwords`
  - `decode`
  - `encode`
  - `footprint`
  - `from_opcode`
  - `is_store`
  - `mem_kind`
  - `Mubuf`
  - `name`
  - `opcode`
- [`src/enc/smem.rs`](src/enc/smem.rs)
  - `buffer`
  - `data_dwords`
  - `decode`
  - `encode`
  - `footprint`
  - `from_opcode`
  - `has_address`
  - `mem_kind`
  - `name`
  - `opcode`
  - `Smem`
  - `SmemOffset`
- [`src/enc/sop.rs`](src/enc/sop.rs)
  - `branch_offset`
  - `decode`
  - `encode`
  - `footprint`
  - `from_opcode`
  - `hwreg_name`
  - `name`
  - `opcode`
  - `Sop1`
  - `Sop2`
  - `Sopc`
  - `Sopk`
  - `Sopp`
  - `waitcnt_fields`
  - `waitcnt_imm`
  - `width`
  - `widths`
- [`src/enc/vop.rs`](src/enc/vop.rs)
  - `decode`
  - `Dpp`
  - `DppCtrl`
  - `encode`
  - `float_compare`
  - `footprint`
  - `from_opcode`
  - `is_cmpx`
  - `name`
  - `opcode`
  - `reads_vcc`
  - `Sdwa`
  - `SdwaSel`
  - `SdwaUnused`
  - `src_dwords`
  - `Vop1`
  - `Vop2`
  - `Vop2K`
  - `Vopc`
  - `VopExt`
  - `widths`
  - `writes_vcc`
- [`src/enc/vop3.rs`](src/enc/vop3.rs)
  - `all`
  - `decode`
  - `encode`
  - `footprint`
  - `from_opcode`
  - `is_vop3b`
  - `mods`
  - `name`
  - `op_sel_bits`
  - `opcode`
  - `Vop3`
  - `Vop3Mods`
  - `Vop3Op`
  - `widths`
- [`src/enc/vop3p.rs`](src/enc/vop3p.rs)
  - `decode`
  - `encode`
  - `footprint`
  - `from_opcode`
  - `Mai`
  - `MfmaShape`
  - `MfmaType`
  - `name`
  - `opcode`
  - `shape`
  - `Vop3p`
  - `widths`
- [`src/float.rs`](src/float.rs)
  - `bf16_to_f32`
  - `f16_to_f32`
  - `f32_to_bf16_rne`
  - `f32_to_f16_rne`
  - `f32_to_f16_rtz`
  - `f32_to_f8_rne`
  - `f32_to_f8_sr`
  - `F8`
  - `f8_to_f32`
  - `ocp_e4m3_to_fnuz_half`
- [`src/generated_cdna3.rs`](src/generated_cdna3.rs)
  - `TABLE`
- [`src/generated_cdna4.rs`](src/generated_cdna4.rs)
  - `TABLE`
- [`src/inst.rs`](src/inst.rs)
  - `assembly`
  - `branch_offset`
  - `classify`
  - `decode`
  - `decode_all`
  - `encode`
  - `encode_all`
  - `Footprint`
  - `footprint`
  - `Format`
  - `format`
  - `Inst`
  - `MemKind`
  - `memory`
  - `new`
  - `of`
  - `overlaps`
  - `read`
  - `read_file`
  - `RegFile`
  - `RegSpan`
  - `RowStatus`
  - `unsupported`
  - `verification`
  - `words`
  - `write`
  - `write_file`
  - `xml`
- [`src/kd.rs`](src/kd.rs)
  - `accum_offset_field`
  - `dx10_clamp`
  - `enable_private_segment`
  - `enable_sgpr_dispatch_id`
  - `enable_sgpr_dispatch_ptr`
  - `enable_sgpr_flat_scratch_init`
  - `enable_sgpr_kernarg_segment_ptr`
  - `enable_sgpr_private_segment_buffer`
  - `enable_sgpr_private_segment_size`
  - `enable_sgpr_queue_ptr`
  - `enable_trap_handler`
  - `enable_vgpr_workitem_id`
  - `enable_wavefront_size32`
  - `enable_workgroup_id`
  - `enable_workgroup_info`
  - `float_denorm_mode_16_64`
  - `float_denorm_mode_32`
  - `float_round_mode_16_64`
  - `float_round_mode_32`
  - `fp16_overflow`
  - `from_bytes`
  - `granulated_wavefront_sgpr_count`
  - `granulated_workitem_vgpr_count`
  - `ieee_mode`
  - `KD_BYTES`
  - `KernelDescriptor`
  - `tg_split`
  - `to_bytes`
  - `user_sgpr_count`
- [`src/lib.rs`](src/lib.rs)
  - `ALL`
  - `arch`
  - `ds`
  - `enc`
  - `flat`
  - `float`
  - `inst`
  - `kd`
  - `mubuf`
  - `operand`
  - `oracle`
  - `Result`
  - `sim`
  - `smem`
  - `sop`
  - `text`
  - `vop`
  - `vop3`
  - `vop3p`
  - `WAVE`
  - `xml`
  - `xml_inst`
- [`src/operand.rs`](src/operand.rs)
  - `ALL`
  - `code`
  - `constant_bits`
  - `f16_bits`
  - `f32_bits`
  - `f64_bits`
  - `fmt`
  - `from_code`
  - `InlineFloat`
  - `is_agpr`
  - `is_inline`
  - `is_literal`
  - `is_sreg`
  - `is_vgpr`
  - `LITERAL`
  - `literal`
  - `Src`
  - `text`
- [`src/oracle.rs`](src/oracle.rs)
  - `bytes`
  - `disabled`
  - `mc_disassemble`
  - `mc_encode`
  - `normalize`
  - `objdump`
  - `tool`
- [`src/sim/load.rs`](src/sim/load.rs)
  - `load_code_object`
- [`src/sim/mem.rs`](src/sim/mem.rs)
  - `drain`
  - `issue`
  - `wait`
- [`src/sim/mfma.rs`](src/sim/mfma.rs)
  - `AccumModel`
  - `accumulate`
  - `accumulate_f64`
  - `accumulate_typed`
  - `exec`
  - `mfma_reference`
- [`src/sim/mod.rs`](src/sim/mod.rs)
  - `at`
  - `check_ready`
  - `Config`
  - `Counter`
  - `decode`
  - `Decoded`
  - `exec_one`
  - `fault`
  - `Flow`
  - `init_wave`
  - `KernelImage`
  - `lane_active`
  - `Launch`
  - `load`
  - `map`
  - `map_readonly`
  - `mem`
  - `Memory`
  - `mfma`
  - `Mode`
  - `new`
  - `Pending`
  - `Program`
  - `read`
  - `read_s`
  - `read_s64`
  - `read_v`
  - `read_v64`
  - `Region`
  - `region`
  - `Retire`
  - `run`
  - `salu`
  - `SimError`
  - `SimResult`
  - `Stats`
  - `Ty`
  - `unmap`
  - `unsupported`
  - `valu`
  - `Wave`
  - `WaveState`
  - `write`
  - `write_s`
  - `write_s64`
  - `write_v`
  - `write_v64`
  - `xlane`
- [`src/sim/salu.rs`](src/sim/salu.rs)
  - `exec`
- [`src/sim/valu.rs`](src/sim/valu.rs)
  - `exec`
- [`src/sim/xlane.rs`](src/sim/xlane.rs)
  - `dpp_lane_enabled`
  - `dpp_read`
  - `dpp_source_lane`
  - `ds_bpermute`
  - `ds_permute`
  - `ds_swizzle`
  - `sdwa_dst`
  - `sdwa_src`
- [`src/text.rs`](src/text.rs)
  - `parse_line`
- [`src/xml.rs`](src/xml.rs)
  - `ALL`
  - `assemble`
  - `can_encode`
  - `covered`
  - `DataComponent`
  - `DataField`
  - `DataFormat`
  - `decode`
  - `Decoded`
  - `dwords`
  - `encode`
  - `Encoding`
  - `EncodingCondition`
  - `Expr`
  - `extract`
  - `Field`
  - `field`
  - `field_at`
  - `fields`
  - `Form`
  - `form`
  - `forms`
  - `from_name`
  - `FunctionalGroup`
  - `gfx`
  - `holds`
  - `InstructionMeta`
  - `is_verified`
  - `Isa`
  - `isa`
  - `IsaTable`
  - `literal`
  - `mask`
  - `mnemonic`
  - `name`
  - `new`
  - `Operand`
  - `operand_value`
  - `operand_violations`
  - `OperandKind`
  - `OperandTypeMeta`
  - `PredefinedValue`
  - `Provenance`
  - `set`
  - `table`
  - `Verification`
  - `verified`
- [`src/xml_inst.rs`](src/xml_inst.rs)
  - `alias_receipt`
  - `alias_receipt_of`
  - `ALIAS_RECEIPTS`
  - `AliasReceipt`
  - `arch`
  - `branch_offset`
  - `covers`
  - `decode`
  - `decoded`
  - `encode`
  - `encoding`
  - `field`
  - `FIELD_RECEIPTS`
  - `FieldReceipt`
  - `footprint`
  - `form`
  - `format`
  - `memory`
  - `name`
  - `opcode`
  - `receipt`
  - `receipt_of`
  - `unsupported`
  - `V_CVT_PK_F32_FP8_SDWA_WORD1`
  - `V_MAD_MIX_F32_AS_FMA_MIX_F32`
  - `verification`
  - `verification_of`
  - `verified`
  - `XmlInst`
- [`src/xml_legality.rs`](src/xml_legality.rs)

### Dependencies (from `Cargo.toml`)

#### path

—

#### external

—

#### dev

—

#### build

—

### Reverse dependencies

Workspace crates with a path dependency on this crate:

- `peacemaker-isa`
- `peacemaker-lift`
- `peacemaker-rip`

<!-- crate-map:generated:end -->
