// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#![forbid(unsafe_code)]
//! CDNA instruction model for PeaceMaker: CDNA3 and later, wave64.
//!
//! One crate for every CDNA generation from CDNA3 on. A generation is a
//! variant of [`arch::Arch`] with a row in the capability table
//! ([`arch::Caps`]: FP8 encoding FNUZ vs OCP, MFMA shapes and types, LDS
//! size, register files); encoders, decoders and the emulator take the
//! variant and consult its row instead of hard-coding gfx942. Only gfx942
//! (CDNA3, MI300X) is populated; gfx950 (CDNA4: OCP FP8, FP4/FP6 MFMA, 160 KiB
//! LDS) arrives as a new variant and table row, not a new crate.
//!
//! Layering mirrors the RDNA side, where `peacemaker-ir` owns the codec and the
//! wave32 interpreter and `peacemaker-isa` owns the checked builder: this crate
//! owns the CDNA instruction model and its CPU emulator; `peacemaker-isa`'s
//! `cdna` module owns the checked wave64 builder, the wait ledger rules, the
//! hazard pass and code-object emission.
//!
//! - [`arch`]: the variant enum and per-variant capability table.
//! - [`operand`]: the 9-bit source / 7-bit scalar destination operand space
//!   (CDNA3 ISA §5.2 Table 13 p.25-26) plus VGPR/AGPR operands.
//! - [`inst`]: [`Inst`], one variant per microcode format, format
//!   classification of a first dword (CDNA3 ISA §13), and the per-instruction
//!   register footprint the builder, ledger and hazard pass consume.
//! - [`enc`]: one module per format family: the opcode table (`*Op::ALL`),
//!   per-variant opcodes, `encode`/`decode`, and a `Display` that prints
//!   exactly what `llvm-mc -show-encoding` / `llvm-objdump` print.
//! - [`float`]: FP8/BF8 in FNUZ (CDNA3) and OCP encodings, F16 and BF16
//!   conversions, bit-exact (CDNA3 ISA §7.2 p.50-52).
//! - [`kd`]: the 64-byte gfx90a+ kernel descriptor layout.
//! - [`sim`]: the wave64 workgroup emulator.
//! - [`oracle`]: ROCm `llvm-mc` / `llvm-objdump` invocation for tests (test
//!   oracle only; nothing in the production path runs a ROCm tool).
//!
//! Citations `CDNA3 ISA §x.y p.N` refer to "AMD Instinct MI300 Instruction Set
//! Architecture Reference Guide", 5 August 2025 (PDF sha256
//! `0cec4237cd93ce7dd76ee8502771429eb2308ab47dfa389f5bcc34f5903a6e2a`), page N
//! as printed ("N of 553").

/// Declare one opcode table: a fieldless enum (variant = CamelCase of the
/// mnemonic without its `s_`/`v_`/`ds_`/... prefix and encoding suffix) plus
/// `ALL`, the table in declaration order. Tables are frozen contracts shared
/// by the encoders, the emulator and the builder: add a variant only through
/// the crate owner.
macro_rules! op_table {
    ($(#[$m:meta])* $name:ident { $($v:ident),+ $(,)? }) => {
        $(#[$m])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub enum $name { $($v),+ }
        impl $name {
            /// Every opcode of the table, in declaration order.
            pub const ALL: &'static [$name] = &[$($name::$v),+];
        }
    };
}

pub mod arch;
pub mod inst;
pub mod operand;
pub mod enc {
    //! One module per CDNA microcode format family (CDNA3 ISA §13).
    pub mod ds;
    pub mod flat;
    pub mod mubuf;
    pub mod smem;
    pub mod sop;
    pub mod vop;
    pub mod vop3;
    pub mod vop3p;
}
pub mod float;
pub mod kd;
pub mod oracle;
pub mod sim;

pub use arch::{Arch, Caps};
pub use inst::{decode_all, encode_all, Format, Inst};
pub use operand::Src;

pub type Result<T, E = String> = std::result::Result<T, E>;

/// Lanes per wavefront on every CDNA variant this crate models.
pub const WAVE: usize = 64;
