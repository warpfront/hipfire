// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! XML-derived structured codec for CDNA: encoding, field and operand metadata
//! generated from AMD's machine-readable ISA XML, plus a generic
//! table-driven decoder/encoder with named (not raw-word) fields.
//!
//! `src/xml.rs` is the hand-written runtime: the shared data model
//! ([`Isa`], [`Encoding`], [`Form`], [`Operand`], [`Decoded`]...), the
//! condition-expression evaluator, and the generic `decode`/`encode` drivers.
//! The per-generation tables (`generated_cdna3.rs`, `generated_cdna4.rs`) are
//! produced by `scripts/generate-cdna-tables.py`; see the header of either
//! generated file for the pinned source XML SHA-256 and the LLVM receipt used
//! for [`Verification`].
//!
//! Bit model: a field value joins its `BitLayout/Range` parts LSB part first
//! (`Range/Order` ascending). A `Padding` element is low operand-value bits
//! that are not in the encoding (`SBASE`: 6 bits, 1 zero pad bit = an
//! aligned SGPR pair); [`Decoded::operand_value`] returns raw `<<` pad bits
//! `|` pad value. "Reserved"
//! bits (covered by no field of the encoding) must be zero on decode and are
//! always zero on encode, so a structured decode re-encodes bit-exact.
//! Decode and encode validate every XML bit (identifier mask, the XML
//! `EncodingCondition` predicates, opcode, reserved bits). Decoded operand
//! codes are preserved verbatim; those outside the XML `OperandType` value
//! sets are reported by [`Decoded::operand_violations`] but not rejected,
//! because the XML sets omit encodings LLVM accepts (e.g. FLAT `SADDR =
//! 0x7f` "off").
//!

use crate::Result;

/// Which CDNA XML the tables and codec below describe.
// NB kept independent of `arch::Arch` (owned by the integration side); the
// generated modules likewise depend only on this module.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Isa {
    /// AMD CDNA 3 (`amdgpu_isa_cdna3.xml`), gfx942. Encode + decode.
    Cdna3,
    /// AMD CDNA 4 (`amdgpu_isa_cdna4.xml`), gfx950. Decode only.
    Cdna4,
}

impl Isa {
    /// Every ISA variant, in [`forms`] order of precedence.
    pub const ALL: &'static [Isa] = &[Isa::Cdna3, Isa::Cdna4];
    /// Table backing this ISA.
    pub fn table(self) -> &'static IsaTable {
        match self {
            Isa::Cdna3 => &crate::generated_cdna3::TABLE,
            Isa::Cdna4 => &crate::generated_cdna4::TABLE,
        }
    }
    /// `.amdgcn_target` processor name.
    pub fn gfx(self) -> &'static str {
        self.table().gfx
    }
    /// Whether [`encode`] accepts this ISA. gfx950 is decode-only.
    pub fn can_encode(self) -> bool {
        matches!(self, Isa::Cdna3)
    }
}

impl std::fmt::Display for Isa {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.gfx())
    }
}

/// Provenance of one generated table: which XML, which LLVM receipt, which
/// generator run. Recorded so a stale or mismatched table fails loudly.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Provenance {
    pub xml_file: &'static str,
    pub xml_sha256: &'static str,
    /// `<schema> <release-date> <arch-name> #<arch-id>`.
    pub xml_version: &'static str,
    pub arch_id: u32,
    pub rows: u32,
    pub encodings: u16,
    pub receipt_file: &'static str,
    pub receipt_sha256: &'static str,
    pub llvm_version: &'static str,
    pub generator: &'static str,
    pub generator_sha256: &'static str,
}

/// One architecture's complete table set (generated).
#[derive(Clone, Copy, Debug)]
pub struct IsaTable {
    pub isa: Isa,
    pub gfx: &'static str,
    pub provenance: Provenance,
    pub encodings: &'static [&'static Encoding],
    pub instructions: &'static [InstructionMeta],
    pub forms: &'static [Form],
    /// `(encoding_index, condition_index, opcode)` sorted; indexes `forms`.
    pub form_keys: &'static [(u16, u16, u32)],
    pub form_rows: &'static [u32],
    pub operand_types: &'static [OperandTypeMeta],
    pub data_formats: &'static [DataFormat],
    pub functional_groups: &'static [FunctionalGroup],
    pub functional_subgroups: &'static [&'static str],
}

impl IsaTable {
    /// Look up the single form addressed by encoding, condition and opcode.
    pub fn form(&self, enc: u16, cond: u16, opcode: u32) -> Option<&'static Form> {
        self.form_keys
            .binary_search(&(enc, cond, opcode))
            .ok()
            .map(|i| &self.forms[self.form_rows[i] as usize])
    }
}

/// One XML `Encoding`: identifier test, conditional sub-forms and fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Encoding {
    pub index: u16,
    pub name: &'static str,
    pub order: i32,
    pub description: &'static str,
    /// Dwords (`bit_count / 32`).
    pub dwords: u8,
    pub bit_count: u16,
    /// Identifier bits: `word & id_mask == id_value` for exactly one row.
    pub id_mask: u128,
    pub id_values: &'static [u128],
    pub conditions: &'static [EncodingCondition],
    pub fields: &'static [Field],
    /// Bit offset of the `OP` field (`OP` is always a single range).
    pub op_offset: u8,
    pub op_width: u8,
    /// Every covered bit except single-bit padding.
    pub covered: u128,
    /// `((1 << bit_count) - 1) & !covered`: must read zero.
    pub reserved: u128,
}

/// One named XML `EncodingCondition` with its compiled predicate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EncodingCondition {
    pub index: u16,
    pub name: &'static str,
    pub expr: Expr,
}

/// Compiled `CondtionExpression` predicate over the encoding's fields.
/// Only the operators the CDNA XMLs use are representable (`==`/`!=`/`&&`
/// against literals, `.notwithin` against a 3-constant set); the generator
/// fails on anything else, so an unknown future operator can never reach a
/// silently-wrong runtime.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Expr {
    Lit(u64),
    /// Raw field value (multi-range joined LSB part first, padding dropped).
    Field(u8),
    Eq(u8, u64),
    Ne(u8, u64),
    Both(&'static Expr, &'static Expr),
    NotWithin(u8, &'static [u64]),
}

impl Expr {
    /// Predicate over the raw field values (encoding order) of the encoding
    /// under test.
    pub fn holds(self, values: &[u64]) -> bool {
        match self {
            Expr::Lit(v) => v != 0,
            Expr::Field(i) => values[i as usize] != 0,
            Expr::Eq(i, v) => values[i as usize] == v,
            Expr::Ne(i, v) => values[i as usize] != v,
            Expr::Both(a, b) => a.holds(values) && b.holds(values),
            Expr::NotWithin(i, vs) => !vs.contains(&values[i as usize]),
        }
    }
}

/// One `MicrocodeFormat/BitMap/Field`: name, bit ranges, padding, and the
/// predefined values `OPR_HWREG` and friends narrow to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Field {
    pub name: &'static str,
    pub description: &'static str,
    /// LSB part first; parts are disjoint and ordered by `Range/Order`.
    pub ranges: &'static [(u8, u8)],
    /// Raw width: the sum of the `ranges` widths (padding excluded).
    pub bits: u8,
    pub pad_bits: u8,
    pub pad_value: u8,
    pub predefined: &'static [PredefinedValue],
}

impl Field {
    /// Raw value: multi-range chunks joined LSB part first, padding dropped.
    pub fn extract(self, word: u128) -> u64 {
        let mut v = 0u64;
        let mut shift = 0u32;
        for &(off, count) in self.ranges {
            v |= (((word >> off) & ((1u128 << count) - 1)) as u64) << shift;
            shift += u32::from(count);
        }
        v
    }
    /// `(1 << bits) - 1` over the raw (unpadded) width.
    pub fn mask(self) -> u64 {
        if self.bits >= 64 {
            u64::MAX
        } else {
            (1u64 << self.bits) - 1
        }
    }
    /// `(1 << (bits + pad)) - 1 << ...` over the covered positions.
    pub fn covered(self) -> u128 {
        let mut m = 0u128;
        for &(off, count) in self.ranges {
            m |= ((1u128 << count) - 1) << off;
        }
        m
    }
}

/// One `OperandPredefinedValues/PredefinedValue` row.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PredefinedValue {
    pub name: &'static str,
    pub description: &'static str,
    pub value: u32,
}

/// One `OperandType`: the valid range set of an operand kind, plus (for the
/// partitioned types like `OPR_HWREG`) the sub-field layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OperandTypeMeta {
    pub name: &'static str,
    pub description: &'static str,
    pub partitioned: bool,
    pub subtypes: &'static [&'static str],
    pub values: &'static [PredefinedValue],
    /// Valid `(low, high)` inclusive spans in operand-code space.
    pub ranges: &'static [(u32, u32)],
    pub fields: &'static [Field],
}

/// One XML `DataFormat`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DataFormat {
    pub name: &'static str,
    pub description: &'static str,
    pub data_type: &'static str,
    pub bit_count: u32,
    pub component_count: u32,
    pub components: &'static [DataComponent],
}

/// One `DataAttributes/BitMap` component of a [`DataFormat`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DataComponent {
    pub order: u32,
    pub fields: &'static [DataField],
}

/// One field of a [`DataComponent`] bit map.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DataField {
    pub name: &'static str,
    pub signedness: &'static str,
    pub ranges: &'static [(u32, u32)],
}

/// One `FunctionalGroup` (with the generator recording only the groups and
/// subgroups actually referenced by instructions; the XML registry itself is
/// also available via [`IsaTable::functional_groups`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FunctionalGroup {
    pub name: &'static str,
    pub description: &'static str,
}

/// One XML `Instruction`: the shared description/flags behind every encoding
/// row of the same mnemonic.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct InstructionMeta {
    pub name: &'static str,
    pub description: &'static str,
    pub is_branch: bool,
    pub is_conditional_branch: bool,
    pub is_indirect_branch: bool,
    pub is_program_terminator: bool,
    pub is_immediately_executed: bool,
    pub group: &'static str,
    pub subgroups: &'static [&'static str],
    pub aliases: &'static [&'static str],
}

/// LLVM receipt state of one [`Form`] (see `tools/cdna*-llvm.json`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Verification {
    /// `llvm-mc` disassembled the seed bytes to this mnemonic and reassembled
    /// them byte-identically (modulo its canonicalization of ignored bits).
    Verified,
    /// Kept in the tables, but [`encode`] refuses the row: `xml-only`
    /// (LLVM rejects the mnemonic) or `unresolved` (LLVM round-trip failed).
    Unverified(&'static str),
}

impl Verification {
    pub fn is_verified(self) -> bool {
        matches!(self, Verification::Verified)
    }
}

/// One `(Instruction, EncodingName, EncodingCondition)` row: a single opcode
/// under a single conditional form of a single encoding.
#[derive(Clone, Copy)]
pub struct Form {
    pub isa: Isa,
    /// Index into `IsaTable::forms` (also the LLVM receipt row index).
    pub row: u32,
    pub inst_index: u32,
    /// XML `InstructionName`, e.g. `"DS_ADD_U32"`.
    pub name: &'static str,
    /// Lowercase mnemonic, e.g. `"ds_add_u32"`.
    pub mnemonic: &'static str,
    pub encoding: &'static str,
    pub encoding_index: u16,
    pub condition: &'static str,
    pub condition_index: u16,
    pub opcode: u32,
    pub operands: &'static [Operand],
    pub fields: &'static [Field],
    pub is_branch: bool,
    pub is_conditional_branch: bool,
    pub is_indirect_branch: bool,
    pub is_program_terminator: bool,
    pub is_immediately_executed: bool,
    pub group: &'static str,
    pub subgroups: &'static [&'static str],
    pub verification: Verification,
    /// LLVM-assembled words of this row (little-endian dwords) from the
    /// receipt; empty exactly when the row is [`Verification::Unverified`].
    /// Every verified row has one, and it decodes to this very row.
    pub sample: &'static [u32],
    pub enc: &'static Encoding,
}

// Rows are unique per ISA: identity is `(isa, row)`, not a deep compare of the
// whole encoding the row points at.
impl PartialEq for Form {
    fn eq(&self, other: &Self) -> bool {
        self.isa == other.isa && self.row == other.row
    }
}

impl Eq for Form {}

impl std::hash::Hash for Form {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.isa.hash(state);
        self.row.hash(state);
    }
}

impl std::fmt::Debug for Form {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Form({} {} {}:{} op {})",
            self.isa, self.name, self.encoding, self.condition, self.opcode
        )
    }
}

impl Form {
    pub fn mnemonic(&self) -> &'static str {
        self.mnemonic
    }
    pub fn is_verified(&self) -> bool {
        self.verification.is_verified()
    }
}

/// One `Operands/Operand` row, in XML `Order`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Operand {
    pub order: u8,
    /// `None` for implicit operands (VCC, EXEC, SCC, M0, memory...).
    pub field: Option<&'static str>,
    pub field_index: Option<u8>,
    pub kind: OperandKind,
    pub data_format: &'static str,
    pub size_bits: u16,
    pub input: bool,
    pub output: bool,
    pub implicit: bool,
    pub microcode_required: bool,
}

/// One XML `OperandType` (union of the CDNA3 and CDNA4 XMLs).
// BEGIN operand-kinds (checked against both XMLs by scripts/generate-cdna-tables.py)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum OperandKind {
    Accvgpr,
    Dsmem,
    FlatScratch,
    Gpumem,
    Hwreg,
    Label,
    Pc,
    Sdst,
    SdstExec,
    SdstM0,
    Sendmsg,
    Simm16,
    Simm32,
    Simm4,
    Simm8,
    SmemOffset,
    Src,
    SrcAccvgpr,
    SrcNolds,
    SrcNolit,
    SrcSimple,
    SrcVgpr,
    SrcVgprOrAccvgpr,
    SrcVgprOrAccvgprOrConst,
    Sreg,
    SregNovcc,
    Ssrc,
    SsrcLanesel,
    SsrcNolit,
    SsrcSpecialScc,
    Vcc,
    Vgpr,
    VgprOrAccvgpr,
    VgprOrLds,
    Waitcnt,
}

impl OperandKind {
    /// Every kind, in declaration order.
    pub const ALL: &'static [OperandKind] = &[
        OperandKind::Accvgpr,
        OperandKind::Dsmem,
        OperandKind::FlatScratch,
        OperandKind::Gpumem,
        OperandKind::Hwreg,
        OperandKind::Label,
        OperandKind::Pc,
        OperandKind::Sdst,
        OperandKind::SdstExec,
        OperandKind::SdstM0,
        OperandKind::Sendmsg,
        OperandKind::Simm16,
        OperandKind::Simm32,
        OperandKind::Simm4,
        OperandKind::Simm8,
        OperandKind::SmemOffset,
        OperandKind::Src,
        OperandKind::SrcAccvgpr,
        OperandKind::SrcNolds,
        OperandKind::SrcNolit,
        OperandKind::SrcSimple,
        OperandKind::SrcVgpr,
        OperandKind::SrcVgprOrAccvgpr,
        OperandKind::SrcVgprOrAccvgprOrConst,
        OperandKind::Sreg,
        OperandKind::SregNovcc,
        OperandKind::Ssrc,
        OperandKind::SsrcLanesel,
        OperandKind::SsrcNolit,
        OperandKind::SsrcSpecialScc,
        OperandKind::Vcc,
        OperandKind::Vgpr,
        OperandKind::VgprOrAccvgpr,
        OperandKind::VgprOrLds,
        OperandKind::Waitcnt,
    ];
    /// XML spelling, e.g. `"OPR_VGPR_OR_ACCVGPR"`.
    pub fn name(self) -> &'static str {
        match self {
            OperandKind::Accvgpr => "OPR_ACCVGPR",
            OperandKind::Dsmem => "OPR_DSMEM",
            OperandKind::FlatScratch => "OPR_FLAT_SCRATCH",
            OperandKind::Gpumem => "OPR_GPUMEM",
            OperandKind::Hwreg => "OPR_HWREG",
            OperandKind::Label => "OPR_LABEL",
            OperandKind::Pc => "OPR_PC",
            OperandKind::Sdst => "OPR_SDST",
            OperandKind::SdstExec => "OPR_SDST_EXEC",
            OperandKind::SdstM0 => "OPR_SDST_M0",
            OperandKind::Sendmsg => "OPR_SENDMSG",
            OperandKind::Simm16 => "OPR_SIMM16",
            OperandKind::Simm32 => "OPR_SIMM32",
            OperandKind::Simm4 => "OPR_SIMM4",
            OperandKind::Simm8 => "OPR_SIMM8",
            OperandKind::SmemOffset => "OPR_SMEM_OFFSET",
            OperandKind::Src => "OPR_SRC",
            OperandKind::SrcAccvgpr => "OPR_SRC_ACCVGPR",
            OperandKind::SrcNolds => "OPR_SRC_NOLDS",
            OperandKind::SrcNolit => "OPR_SRC_NOLIT",
            OperandKind::SrcSimple => "OPR_SRC_SIMPLE",
            OperandKind::SrcVgpr => "OPR_SRC_VGPR",
            OperandKind::SrcVgprOrAccvgpr => "OPR_SRC_VGPR_OR_ACCVGPR",
            OperandKind::SrcVgprOrAccvgprOrConst => "OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST",
            OperandKind::Sreg => "OPR_SREG",
            OperandKind::SregNovcc => "OPR_SREG_NOVCC",
            OperandKind::Ssrc => "OPR_SSRC",
            OperandKind::SsrcLanesel => "OPR_SSRC_LANESEL",
            OperandKind::SsrcNolit => "OPR_SSRC_NOLIT",
            OperandKind::SsrcSpecialScc => "OPR_SSRC_SPECIAL_SCC",
            OperandKind::Vcc => "OPR_VCC",
            OperandKind::Vgpr => "OPR_VGPR",
            OperandKind::VgprOrAccvgpr => "OPR_VGPR_OR_ACCVGPR",
            OperandKind::VgprOrLds => "OPR_VGPR_OR_LDS",
            OperandKind::Waitcnt => "OPR_WAITCNT",
        }
    }
    /// Parse the XML spelling.
    pub fn from_name(name: &str) -> Option<OperandKind> {
        OperandKind::ALL.iter().copied().find(|k| k.name() == name)
    }
}
// END operand-kinds

/// A structured decoded instruction: its [`Form`] plus one raw value per
/// field of the encoding, in encoding order.
#[derive(Clone)]
pub struct Decoded {
    pub form: &'static Form,
    values: [u64; 24],
}

impl PartialEq for Decoded {
    fn eq(&self, other: &Self) -> bool {
        self.form == other.form && self.values == other.values
    }
}

impl Eq for Decoded {}

impl std::hash::Hash for Decoded {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.form.hash(state);
        self.values.hash(state);
    }
}

impl std::fmt::Debug for Decoded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Decoded")
            .field("form", &self.form.name)
            .field("encoding", &self.form.encoding)
            .field("condition", &self.form.condition)
            .field("opcode", &self.form.opcode)
            .field("values", &self.fields().collect::<Vec<_>>())
            .finish()
    }
}

impl Decoded {
    /// Blank slate for `form`: identifier/`ENCODING`/`OP` fields and simple
    /// `field == literal` conditions prefilled; fill the rest with [`set`].
    pub fn new(isa: Isa, form: &'static Form) -> Self {
        debug_assert_eq!(form.isa, isa);
        let mut d = Decoded {
            form,
            values: [0; 24],
        };
        d.prefill();
        d
    }
    fn prefill(&mut self) {
        let enc = self.form.enc;
        let ids = enc.id_values;
        let opcode = self.form.opcode;
        let op = enc.op_offset;
        let ow = enc.op_width;
        let id = ids
            .iter()
            .find(|id| ((**id >> op) & ((1u128 << ow) - 1)) as u32 == opcode)
            .copied()
            .unwrap_or(ids[0]);
        for (i, fl) in enc.fields.iter().enumerate() {
            // Fields fully covered by the identifier mask take the value.
            let mut m = 0u128;
            for &(off, count) in fl.ranges {
                m |= ((1u128 << count) - 1) << off;
            }
            if m != 0 && m & !enc.id_mask == 0 {
                self.values[i] = fl.extract(id);
            }
        }
        self.satisfy(enc.conditions[usize::from(self.form.condition_index)].expr);
    }
    fn satisfy(&mut self, expr: Expr) {
        // Seed simple `field == literal` conjuncts (DPP `SRC0 == 250`,
        // literal `SSRC0 == 255`, ...); anything fancier is the caller's.
        match expr {
            Expr::Both(a, b) => {
                self.satisfy(*a);
                self.satisfy(*b);
            }
            Expr::Eq(i, v) if (i as usize) < 24 => self.values[i as usize] = v,
            _ => {}
        }
    }
    /// Which ISA this decoded value belongs to.
    pub fn isa(&self) -> Isa {
        self.form.isa
    }
    /// Raw value of the named field (`None` if the form has no such field).
    /// Multi-range fields join LSB part first; padding is excluded (see
    /// [`operand_value`]).
    pub fn field(&self, name: &str) -> Option<u64> {
        self.form
            .fields
            .iter()
            .position(|fl| fl.name == name)
            .map(|i| self.values[i])
    }
    /// Raw value of the `i`-th field of the encoding.
    pub fn field_at(&self, i: usize) -> Option<u64> {
        (i < self.form.fields.len()).then_some(self.values[i])
    }
    /// `(name, value)` pairs in encoding order.
    pub fn fields(&self) -> impl Iterator<Item = (&'static str, u64)> + '_ {
        self.form
            .fields
            .iter()
            .enumerate()
            .map(|(i, fl)| (fl.name, self.values[i]))
    }
    /// Set a field by name; errors on unknown fields and over-wide values.
    pub fn set(&mut self, name: &str, value: u64) -> Result<()> {
        let (i, fl) = self
            .form
            .fields
            .iter()
            .enumerate()
            .find(|(_, fl)| fl.name == name)
            .ok_or_else(|| format!("{} has no field {name}", self.form.name))?;
        if value & !fl.mask() != 0 {
            return Err(format!(
                "{} field {name} value {value:#x} exceeds {} bits",
                self.form.name, fl.bits
            ));
        }
        self.values[i] = value;
        Ok(())
    }
    /// The `LITERAL` dword, when this form carries one.
    pub fn literal(&self) -> Option<u32> {
        self.field("LITERAL").map(|v| v as u32)
    }
    /// Dwords this instruction occupies.
    pub fn dwords(&self) -> usize {
        usize::from(self.form.enc.dwords)
    }
    /// The operand's numeric code in XML value space
    /// (`raw << pad_bits | pad_value`); `None` for implicit operands.
    pub fn operand_value(&self, op: &Operand) -> Option<u32> {
        let i = op.field_index? as usize;
        let fl = self.form.fields.get(i)?;
        debug_assert_eq!(Some(fl.name), op.field);
        Some(((self.values[i] << fl.pad_bits) | u64::from(fl.pad_value)) as u32)
    }
    /// Operand codes outside the XML `OperandType` predefined-value set.
    /// Advisory only (see the module docs): [`decode`] accepts these.
    pub fn operand_violations(&self) -> Vec<String> {
        let table = self.isa().table();
        operand_violations_inner(table, self.form, &self.values)
    }
    /// Assemble the dwords; checks identifier, condition, opcode, widths and
    /// reserved bits, and requires a re-decode to land on the same form.
    /// Policy gates (ISA encode rights, [`Verification`]) are the caller's:
    /// see [`encode`].
    pub fn assemble(&self) -> Result<([u32; 4], usize)> {
        assemble_inner(self.form, &self.values)
    }
}

/// Compute [`Decoded::operand_violations`] without borrowing a `Decoded`.
fn operand_violations_inner(table: &IsaTable, form: &Form, values: &[u64; 24]) -> Vec<String> {
    let mut out = Vec::new();
    for op in form.operands {
        let Some(i) = op.field_index.map(usize::from) else {
            continue;
        };
        let Some(fl) = form.fields.get(i) else {
            continue;
        };
        let v = ((values[i] << fl.pad_bits) | u64::from(fl.pad_value)) as u32;
        let Some(meta) = table
            .operand_types
            .iter()
            .find(|t| t.name == op.kind.name())
        else {
            continue;
        };
        if meta.values.is_empty() || meta.partitioned {
            continue;
        }
        let w = fl.bits + fl.pad_bits;
        let modulo = if w >= 32 {
            u64::from(u32::MAX) + 1
        } else {
            1u64 << w
        };
        // `v` is valid when some value-set member projects onto it through
        // the field's width: `m == v (mod modulo)` for some range `(lo, hi)`.
        // Codes the XML value set omits are reported (e.g. FLAT `SADDR =
        // 0x7f` "off" is a violation although LLVM encodes it): hence advisory.
        let ok = meta.ranges.iter().any(|&(lo, hi)| {
            let (lo, hi) = (u64::from(lo), u64::from(hi));
            // Smallest `t >= lo` with `t == v (mod modulo)`.
            let v = u64::from(v) % modulo;
            let t = lo + (v + modulo - lo % modulo) % modulo;
            t <= hi
        });
        if !ok {
            out.push(format!(
                "{} operand {} ({}={v:#x}) outside {}",
                form.name,
                op.order,
                op.field.unwrap_or("implicit"),
                op.kind.name()
            ));
        }
    }
    out
}

/// Compute [`Decoded::assemble`] without borrowing a `Decoded`.
fn assemble_inner(form: &Form, values: &[u64; 24]) -> Result<([u32; 4], usize)> {
    let enc = form.enc;
    let mut word = 0u128;
    for (i, fl) in enc.fields.iter().enumerate() {
        let v = values[i];
        if v & !fl.mask() != 0 {
            return Err(format!(
                "{} field {} value {v:#x} exceeds {} bits",
                form.name, fl.name, fl.bits
            ));
        }
        let mut rest = v;
        for &(off, count) in fl.ranges {
            word |= ((rest & ((1u64 << count) - 1)) as u128) << off;
            rest >>= count;
        }
    }
    if !enc.id_values.iter().any(|id| word & enc.id_mask == *id) {
        return Err(format!(
            "{} assembled word {word:#x} matches no {:?} identifier",
            form.name, enc.name
        ));
    }
    let mut cond_values = [0u64; 24];
    for (i, fl) in enc.fields.iter().enumerate() {
        cond_values[i] = fl.extract(word);
    }
    if !enc.conditions[usize::from(form.condition_index)]
        .expr
        .holds(&cond_values)
    {
        return Err(format!(
            "{} assembled word {word:#x} violates condition {}",
            form.name, form.condition
        ));
    }
    let op = ((word >> enc.op_offset) & ((1u128 << enc.op_width) - 1)) as u32;
    if op != form.opcode {
        return Err(format!(
            "{} assembled opcode {op} != form opcode {}",
            form.name, form.opcode
        ));
    }
    if word & enc.reserved != 0 {
        return Err(format!(
            "{} assembled word {word:#x} sets reserved bits {:#x}",
            form.name,
            word & enc.reserved
        ));
    }
    let n = usize::from(enc.dwords);
    let mut out = [0u32; 4];
    for (i, o) in out.iter_mut().enumerate().take(n) {
        *o = (word >> (32 * i)) as u32;
    }
    let (back, used) = decode(form.isa, &out[..n])?;
    if used != n || !std::ptr::eq(back.form, form) {
        return Err(format!(
            "{} assembled word {word:#x} re-decodes as {}",
            form.name, back.form.name
        ));
    }
    Ok((out, n))
}

/// Every [`Form`] of one ISA, in row order (== LLVM receipt index order).
pub fn forms(isa: Isa) -> &'static [Form] {
    isa.table().forms
}

/// Decode one instruction: match the identifier mask, the XML condition and
/// the opcode; reject reserved encodings and truncated words. Returns the
/// structured value and the dwords consumed.
pub fn decode(isa: Isa, words: &[u32]) -> Result<(Decoded, usize)> {
    if words.is_empty() {
        return Err("no words to decode".to_string());
    }
    let table = isa.table();
    let mut word = 0u128;
    for (i, &w) in words.iter().enumerate().take(4) {
        word |= u128::from(w) << (32 * i);
    }
    let mut hit: Option<(&'static Form, usize)> = None;
    for enc in table.encodings {
        if !enc.id_values.iter().any(|id| (word & enc.id_mask) == *id) {
            continue;
        }
        // A short slice still matches the identifier (missing words read
        // zero); conditions decide before length does.
        let n = usize::from(enc.dwords);
        let masked = word & (u128::MAX >> (128 - enc.bit_count));
        let mut values = [0u64; 24];
        for (i, fl) in enc.fields.iter().enumerate() {
            values[i] = fl.extract(masked);
        }
        let opcode = ((masked >> enc.op_offset) & ((1u128 << enc.op_width) - 1)) as u32;
        let mut found = None;
        for (ci, cond) in enc.conditions.iter().enumerate() {
            if !cond.expr.holds(&values) {
                continue;
            }
            if let Some(f) = table.form(enc.index, ci as u16, opcode) {
                found = Some(f);
                break;
            }
        }
        if let Some(form) = found {
            if let Some((prev, _)) = hit {
                // Distinct XML rows can claim the same words (the CDNA4
                // `ENC_VOP3PX2` identifiers `0xd3ad/0xd3ae` are ordinary
                // `VOP3P_MFMA` first dwords). A receipt-verified row beats an
                // unverified one; two of the same standing are ambiguous.
                match (prev.is_verified(), form.is_verified()) {
                    (true, false) => continue,
                    (false, true) => {}
                    _ => {
                        return Err(format!(
                            "ambiguous decode: matches both {} and {}",
                            prev.name, form.name
                        ))
                    }
                }
            }
            hit = Some((form, n));
        }
    }
    let (form, n) = hit.ok_or_else(|| format!("no {isa} encoding matches {:#x}", words[0]))?;
    if words.len() < n {
        return Err(format!(
            "{} needs {n} dwords, only {} present",
            form.name,
            words.len()
        ));
    }
    if word & form.enc.reserved != 0 {
        return Err(format!(
            "{} sets reserved bits {:#x}",
            form.name,
            word & form.enc.reserved
        ));
    }
    let masked = word & (u128::MAX >> (128 - form.enc.bit_count));
    let mut d = Decoded {
        form,
        values: [0; 24],
    };
    for (i, fl) in form.enc.fields.iter().enumerate() {
        d.values[i] = fl.extract(masked);
    }
    Ok((d, n))
}

/// Encode one structured instruction; appends its dwords to `out`.
/// Refuses decode-only ISAs (gfx950) and [`Verification::Unverified`] rows:
/// decode recognizes those, the seal must not emit them.
pub fn encode(d: &Decoded, out: &mut Vec<u32>) -> Result<()> {
    let isa = d.isa();
    if !isa.can_encode() {
        return Err(format!("{isa} is decode-only"));
    }
    if let Verification::Unverified(reason) = d.form.verification {
        return Err(format!("unverified encoding {}: {reason}", d.form.name));
    }
    let (words, n) = d.assemble()?;
    out.extend_from_slice(&words[..n]);
    Ok(())
}

/// Is `isa`'s [`Form`] `row` receipt-verified? (`row` = index in
/// [`forms`].)
pub fn verified(isa: Isa, row: usize) -> bool {
    isa.table().forms.get(row).is_some_and(|f| f.is_verified())
}
