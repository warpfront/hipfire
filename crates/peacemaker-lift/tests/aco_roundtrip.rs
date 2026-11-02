// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! P1 gate for the raw (ACO) binary lift: every executable word round-trips from typed fields,
//! CFG and entry facts are exact, and every kind of tampering is rejected. The instruction
//! words are the pinned encodings of `text/parse_tests.rs` and the codec tests, not
//! hand-assembled: `v_dual_mov_b32`/`v_dual_add_f32`, `v_dual_fmamk_f32` with a literal,
//! `s_cbranch_scc1`, `v_cvt_f32_f16`, `s_endpgm`, and the pinned decode-negative words.
//! The real producer binary is a later gate (`pm-raw-census`).

use peacemaker_ir::cfg::{Cond, Terminator};
use peacemaker_ir::edit::entry_undefined;
use peacemaker_ir::inst::Program;
use peacemaker_ir::inst::{
    Abi, Arch, Axis, EntryFact, EntryLoc, EntryValue, ExecExtent, FpState, Frontend, KernelOrigin,
    LocalIdField, RawKernelSpec, RawResources, Relocation, Setting, SymbolId, Target, Wave,
};
use peacemaker_ir::operand::Special;
use peacemaker_ir::reg::{Kind, RegRef};
use peacemaker_ir::state::{Lifted, StreamProof};
use peacemaker_lift::raw::{parse_spec_file, scan_stream, verify_sidecar};
use peacemaker_lift::{emit, lift_raw, LiftError, Options, Rule};
use serde_json::json;
use sha2::{Digest, Sha256};

const ARCH: Arch = Arch::Gfx1201;
const ACO: Options = Options {
    frontend: Frontend::Aco,
};
const SIDECAR: &[u8] = b"{\"schema\":\"pm-aco-abi-v1\"}";
const HEADER: usize = 16;
const TRAILER: usize = 8;

// v_dual_mov_b32 v1, v2 :: v_dual_add_f32 v4, v5, v6
const VOPD: [u32; 2] = [0xca08_0102, 0x0104_0d05];
// v_dual_fmamk_f32 v1, v2, 0x3f800000, v3 :: v_dual_mul_f32 v4, v5, v6
const VOPD_LITERAL: [u32; 3] = [0xc886_0702, 0x0104_0d05, 0x3f80_0000];
// s_cbranch_scc1 +1 (pinned form: s_cbranch_scc1 -2 is 0xbfa2fffe)
const CBRANCH_SCC1_PLUS1: u32 = 0xbfa2_0001;
// v_cvt_f32_f16_e32 v1, v2.h
const CVT: u32 = 0x7e02_1782;
const ENDPGM: u32 = 0xbfb0_0000;

/// VOPD, `s_cbranch_scc1` over one `v_cvt`, VOPD with a literal, `s_endpgm`.
fn valid_words() -> Vec<u32> {
    let mut words = VOPD.to_vec();
    words.push(CBRANCH_SCC1_PLUS1);
    words.push(CVT);
    words.extend(VOPD_LITERAL);
    words.push(ENDPGM);
    words
}

fn sha(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
fn code_of(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn reg(kind: Kind, base: u16, len: u8) -> EntryLoc {
    EntryLoc::Reg(RegRef { kind, base, len })
}
fn fact(loc: EntryLoc, value: EntryValue) -> EntryFact {
    EntryFact { loc, value }
}

/// Exactly what `valid_words` reads: EXEC, SCC, v2..v3, v5..v6, plus the workgroup id and the
/// packed local id the pilot ABI declares.
fn entry_facts() -> Vec<EntryFact> {
    vec![
        fact(EntryLoc::Special(Special::Exec), EntryValue::Defined),
        fact(EntryLoc::Special(Special::Scc), EntryValue::Defined),
        fact(reg(Kind::V, 2, 2), EntryValue::Defined),
        fact(reg(Kind::V, 5, 2), EntryValue::Defined),
        fact(reg(Kind::Ttmp, 9, 1), EntryValue::WorkgroupId(Axis::X)),
        fact(
            reg(Kind::V, 0, 1),
            EntryValue::LocalId(vec![LocalIdField {
                axis: Axis::X,
                shift: 0,
                bits: 10,
                max: 31,
            }]),
        ),
    ]
}

struct Fixture {
    bytes: Vec<u8>,
    spec: RawKernelSpec,
}

impl Fixture {
    fn new(words: &[u32]) -> Self {
        let code = code_of(words);
        let mut bytes = vec![0xa5; HEADER];
        bytes.extend(&code);
        bytes.extend([0x5a; TRAILER]);
        let spec = RawKernelSpec {
            target: Target {
                arch: ARCH,
                xnack: Setting::Any,
                sramecc: Setting::Any,
                abi_version: 0,
            },
            symbol: SymbolId("aco_pilot".into()),
            wave: Wave::Wave32,
            extent: ExecExtent {
                offset: HEADER as u64,
                len: code.len() as u64,
            },
            binary_sha256: sha(&bytes),
            exec_sha256: sha(&code),
            resources: RawResources {
                lds_bytes: 0,
                scratch_bytes_per_lane: 0,
                vgprs_actual: 8,
                vgprs_allocated: 8,
                sgprs_actual: 0,
                sgprs_allocated: 8,
            },
            fp: FpState::default(),
            relocations: vec![],
            entry: entry_facts(),
            sidecar_sha256: sha(SIDECAR),
        };
        Self { bytes, spec }
    }
    fn code(&self) -> &[u8] {
        &self.bytes[HEADER..self.bytes.len() - TRAILER]
    }
    fn lift(&self) -> Result<Lifted<Program>, LiftError> {
        lift_raw(&self.bytes, &self.spec, ACO)
    }
}

fn rule(error: &LiftError) -> Rule {
    match error {
        LiftError::Rejected { rule, .. } => *rule,
    }
}
fn reason(error: &LiftError) -> &str {
    match error {
        LiftError::Rejected { reason, .. } => reason,
    }
}
fn offset(error: &LiftError) -> u64 {
    match error {
        LiftError::Rejected { offset, .. } => *offset,
    }
}

fn expect_rejected(result: Result<Lifted<Program>, LiftError>, wanted: Rule) -> LiftError {
    match result {
        Ok(_) => panic!("expected a {wanted:?} rejection, but the lift succeeded"),
        Err(error) => {
            assert_eq!(rule(&error), wanted, "{error}");
            error
        }
    }
}

#[test]
fn every_word_round_trips_with_exact_cfg_and_entry_facts() {
    let fx = Fixture::new(&valid_words());
    let lifted = fx.lift().unwrap_or_else(|e| panic!("{e}"));
    lifted.validate().unwrap();
    let program = &lifted.program;
    assert!(
        program.source.is_none(),
        "a raw program has no fabricated HSA envelope"
    );
    assert!(matches!(
        emit::module(program),
        Err(emit::EmitError::NoEnvelope)
    ));
    let kernel = &program.kernels[0];
    assert_eq!(kernel.symbol, fx.spec.symbol);
    assert_eq!(kernel.wave, Wave::Wave32);
    assert_eq!(kernel.abi, Abi::Raw(fx.spec.abi()));
    assert_eq!(
        kernel.origin,
        KernelOrigin::Frontend {
            kind: Frontend::Aco,
            object_sha256: fx.spec.binary_sha256,
            entry_va: HEADER as u64,
            size: fx.spec.extent.len,
        }
    );
    assert_eq!(lifted.proof.input_sha256, sha(&fx.bytes));
    assert_eq!(
        lifted.proof.streams,
        vec![StreamProof {
            entry: HEADER as u64,
            size: fx.spec.extent.len,
            stream_sha256: sha(fx.code())
        }]
    );
    assert_eq!(lifted.proof.sidecar_sha256, Some(sha(SIDECAR)));

    // Exact CFG: [VOPD, s_cbranch_scc1] -> {[v_cvt], [VOPD+literal, s_endpgm]}; v_cvt falls into the join.
    let body = &kernel.body;
    assert_eq!(body.layout.len(), 5);
    let ranges: Vec<(usize, usize)> = body.blocks.iter().map(|b| b.range).collect();
    assert_eq!(ranges, vec![(0, 2), (2, 3), (3, 5)]);
    assert!(matches!(
        body.blocks[0].term,
        Terminator::Branch {
            cond: Cond::Scc1,
            ..
        }
    ));
    assert_eq!(body.blocks[1].term, Terminator::FallThrough);
    assert_eq!(body.blocks[2].term, Terminator::EndPgm);
    let succ = |block: usize| {
        let mut v: Vec<usize> = body.blocks[block].succs.iter().map(|b| b.0).collect();
        v.sort();
        v
    };
    assert_eq!((succ(0), succ(1), succ(2)), (vec![1, 2], vec![2], vec![]));

    // Entry facts are exact: nothing the code reads is left undefined.
    assert!(entry_undefined(program, &fx.spec.symbol)
        .unwrap()
        .is_empty());

    // Re-encoding reproduces the executable range from typed fields alone ...
    assert_eq!(emit::bytes(kernel, ARCH).unwrap(), fx.code());
    assert_eq!(emit::words(kernel, ARCH).unwrap(), valid_words());
    // ... even with every provenance word deleted or poisoned.
    let mut poisoned = program.clone();
    let ids = poisoned.kernels[0].body.layout.clone();
    for (n, id) in ids.into_iter().enumerate() {
        let inst = poisoned.kernels[0].body.insts.get_mut(id).unwrap();
        inst.prov.bytes = if n % 2 == 0 {
            None
        } else {
            Some([0xdead_beef; 3])
        };
        inst.prov.pc = None;
    }
    assert_eq!(emit::bytes(&poisoned.kernels[0], ARCH).unwrap(), fx.code());
    assert_eq!(
        emit::bytes(&poisoned.kernels[0], ARCH).unwrap(),
        emit::bytes(kernel, ARCH).unwrap()
    );
}

#[test]
fn relocations_must_name_an_instruction_literal() {
    let mut fx = Fixture::new(&valid_words());
    // words: VOPD 0-1, s_cbranch 2, v_cvt 3, VOPD+literal 4-6 (literal at word 6), s_endpgm 7
    fx.spec.relocations = vec![Relocation {
        offset: 24,
        r_type: 1,
        symbol: "rodata".into(),
        addend: 0,
    }];
    let lifted = fx.lift().unwrap_or_else(|e| panic!("{e}"));
    let Abi::Raw(raw) = &lifted.program.kernels[0].abi else {
        panic!("raw ABI")
    };
    assert_eq!(raw.relocations, fx.spec.relocations);
    for bad in [0u64, 16, 28] {
        fx.spec.relocations = vec![Relocation {
            offset: bad,
            r_type: 1,
            symbol: "rodata".into(),
            addend: 0,
        }];
        let e = expect_rejected(fx.lift(), Rule::Spec);
        assert!(reason(&e).contains("literal"), "{e}");
    }
    fx.spec.relocations = vec![Relocation {
        offset: 2,
        r_type: 1,
        symbol: "rodata".into(),
        addend: 0,
    }];
    expect_rejected(fx.lift(), Rule::Spec);
    fx.spec.relocations = vec![Relocation {
        offset: 4096,
        r_type: 1,
        symbol: "rodata".into(),
        addend: 0,
    }];
    expect_rejected(fx.lift(), Rule::Spec);
}

#[test]
fn unknown_reserved_and_dangerous_bits_are_decode_rejections() {
    // v_cmpx_gt_i32_e64 s7, v4 with a reserved bit flipped (codec test `implicit EXEC`).
    let cmpx = [0xd4c4_007e_u32, 0x0202_0807];
    let fx = Fixture::new(&[cmpx[0] ^ 1, cmpx[1], ENDPGM]);
    let e = expect_rejected(fx.lift(), Rule::Decode);
    assert_eq!(offset(&e), HEADER as u64);
    assert!(reason(&e).contains("implicit EXEC"), "{e}");

    // v_add_co_u32 with the literal-selector fill in the unused src2 field (codec test `dangerous-fill`).
    let mut add = [0xd700_6a01_u32, 0x0202_020e];
    add[1] = (add[1] & !(0x1ff << 18)) | (0xff << 18);
    let e = expect_rejected(Fixture::new(&[add[0], add[1], ENDPGM]).lift(), Rule::Decode);
    assert!(reason(&e).contains("dangerous-fill"), "{e}");

    // The structured census names the same first failing word and does not hide it.
    let scan = scan_stream(&code_of(&[cmpx[0] ^ 1, cmpx[1], ENDPGM]), ARCH);
    let first = &scan.failures[0];
    assert_eq!(
        (first.offset, first.word, first.rule),
        (0, cmpx[0] ^ 1, Rule::Decode)
    );
    assert!(scan_stream(&code_of(&valid_words()), ARCH)
        .failures
        .is_empty());
    assert_eq!(scan_stream(&code_of(&valid_words()), ARCH).decoded.len(), 5);
}

#[test]
fn an_instruction_overrunning_the_executable_range_is_rejected() {
    // The VOPD literal word is cut off by the end of the executable range.
    let fx = Fixture::new(&[CVT, VOPD_LITERAL[0], VOPD_LITERAL[1]]);
    let e = expect_rejected(fx.lift(), Rule::Decode);
    assert_eq!(offset(&e), HEADER as u64 + 4);
}

#[test]
fn branches_must_land_on_instruction_starts_inside_the_stream() {
    // s_cbranch_scc1 +1 lands on the VOPD's literal word (middle of the instruction).
    let into_middle = Fixture::new(&[
        CBRANCH_SCC1_PLUS1,
        VOPD_LITERAL[0],
        VOPD_LITERAL[1],
        VOPD_LITERAL[2],
        ENDPGM,
    ]);
    expect_rejected(into_middle.lift(), Rule::BranchTarget);
    // s_branch +100 leaves the stream.
    let outside = Fixture::new(&[0xbfa0_0064, ENDPGM]);
    expect_rejected(outside.lift(), Rule::BranchTarget);
    // A path that falls off the end without s_endpgm.
    let no_end = Fixture::new(&[CVT]);
    expect_rejected(no_end.lift(), Rule::ControlFlow);
}

#[test]
fn target_wave_frontend_and_extent_mismatches_are_rejected() {
    // Target: s_add_f32 exists on gfx1151 and not on gfx1100 (codec test), so one stream, one
    // sidecar, and only the declared target decides.
    let mut s_add_f32 = Fixture::new(&[0xa000_0201, ENDPGM]);
    s_add_f32.spec.target.arch = Arch::Gfx1151;
    s_add_f32.spec.entry = vec![fact(reg(Kind::S, 0, 4), EntryValue::Defined)];
    s_add_f32.spec.resources = RawResources {
        sgprs_actual: 4,
        sgprs_allocated: 8,
        ..Default::default()
    };
    s_add_f32.lift().unwrap_or_else(|e| panic!("{e}"));
    s_add_f32.spec.target.arch = Arch::Gfx1100;
    expect_rejected(s_add_f32.lift(), Rule::Decode);
    s_add_f32.spec.target.arch = Arch::Gfx1030;
    expect_rejected(s_add_f32.lift(), Rule::UnsupportedArch);

    // Wave: declared wave64 with only the low EXEC half defined; the vector code reads EXEC_HI.
    let mut wave64 = Fixture::new(&valid_words());
    wave64.spec.wave = Wave::Wave64;
    wave64.spec.entry[0] = fact(EntryLoc::Special(Special::ExecLo), EntryValue::Defined);
    let e = expect_rejected(wave64.lift(), Rule::Entry);
    assert!(reason(&e).contains("exec_hi"), "{e}");
    // Wave32 has no EXEC_HI to declare.
    let mut wave32 = Fixture::new(&valid_words());
    wave32.spec.entry[0] = fact(EntryLoc::Special(Special::ExecHi), EntryValue::Defined);
    expect_rejected(wave32.lift(), Rule::Spec);

    // Frontend: the raw importer is the ACO importer.
    let fx = Fixture::new(&valid_words());
    for frontend in [
        Frontend::Hipcc,
        Frontend::Triton,
        Frontend::Builder,
        Frontend::Ctor,
    ] {
        let e = expect_rejected(
            lift_raw(&fx.bytes, &fx.spec, Options { frontend }),
            Rule::Spec,
        );
        assert!(reason(&e).contains("ACO"), "{e}");
    }

    // Extent outside the binary.
    let mut extent = Fixture::new(&valid_words());
    extent.spec.extent.len += 64;
    let e = expect_rejected(extent.lift(), Rule::Spec);
    assert!(reason(&e).contains("outside"), "{e}");
    let mut unaligned = Fixture::new(&valid_words());
    unaligned.spec.extent.offset += 2;
    expect_rejected(unaligned.lift(), Rule::Spec);
}

#[test]
fn hash_and_sidecar_tampering_is_rejected() {
    let fx = Fixture::new(&valid_words());
    verify_sidecar(SIDECAR, &fx.spec).unwrap();
    let e = verify_sidecar(b"{\"schema\":\"tampered\"}", &fx.spec).unwrap_err();
    assert_eq!(rule(&e), Rule::Spec);
    assert!(reason(&e).contains("sidecar"), "{e}");

    // Trailing data outside the executable range is covered by the binary hash.
    let mut tampered = Fixture::new(&valid_words());
    *tampered.bytes.last_mut().unwrap() ^= 1;
    let e = expect_rejected(tampered.lift(), Rule::Spec);
    assert!(reason(&e).contains("binary SHA-256"), "{e}");

    let mut exec_hash = Fixture::new(&valid_words());
    exec_hash.spec.exec_sha256[0] ^= 1;
    let e = expect_rejected(exec_hash.lift(), Rule::Spec);
    assert!(reason(&e).contains("executable SHA-256"), "{e}");

    let mut binary_hash = Fixture::new(&valid_words());
    binary_hash.spec.binary_sha256[31] ^= 1;
    expect_rejected(binary_hash.lift(), Rule::Spec);

    // A different sidecar hash is a different proof, never silently the same one.
    let mut other = Fixture::new(&valid_words());
    other.spec.sidecar_sha256 = sha(b"another sidecar");
    let proof = other.lift().unwrap().proof;
    assert_eq!(proof.sidecar_sha256, Some(sha(b"another sidecar")));
    assert_ne!(
        proof.sidecar_sha256,
        fx.lift().unwrap().proof.sidecar_sha256
    );
}

#[test]
fn reads_of_undeclared_entry_state_are_rejected() {
    // EXEC is not seeded for raw kernels: every vector instruction reads it.
    let mut no_exec = Fixture::new(&valid_words());
    no_exec.spec.entry.remove(0);
    let e = expect_rejected(no_exec.lift(), Rule::Entry);
    assert!(reason(&e).contains("exec_lo"), "{e}");
    assert_eq!(offset(&e), HEADER as u64, "the first vector instruction");

    // SCC feeds s_cbranch_scc1.
    let mut no_scc = Fixture::new(&valid_words());
    no_scc.spec.entry.remove(1);
    let e = expect_rejected(no_scc.lift(), Rule::Entry);
    assert!(reason(&e).contains("scc"), "{e}");
    assert_eq!(offset(&e), HEADER as u64 + 8, "s_cbranch_scc1");

    // VGPR sources.
    let mut no_v5 = Fixture::new(&valid_words());
    no_v5.spec.entry.remove(3);
    let e = expect_rejected(no_v5.lift(), Rule::Entry);
    assert!(reason(&e).contains("v5"), "{e}");

    // Declaring a location does not define its neighbours.
    let mut narrow = Fixture::new(&valid_words());
    narrow.spec.entry[3] = fact(reg(Kind::V, 5, 1), EntryValue::Defined);
    let e = expect_rejected(narrow.lift(), Rule::Entry);
    assert!(reason(&e).contains("v6"), "{e}");
}

#[test]
fn sgpr_counts_use_the_physical_basis_and_s106_is_never_ordinary() {
    // s_add_f32 s0, s1, s2 (gfx1151 pinned word): touches ordinary s0..s2.
    let mut fx = Fixture::new(&[0xa000_0201, ENDPGM]);
    fx.spec.target.arch = Arch::Gfx1151;
    fx.spec.entry = vec![fact(reg(Kind::S, 0, 4), EntryValue::Defined)];
    // Driver counts that include VCC_LO/VCC_HI (106, 107) are kept verbatim.
    fx.spec.resources = RawResources {
        sgprs_actual: 107,
        sgprs_allocated: 108,
        ..Default::default()
    };
    fx.spec.validate().unwrap();
    fx.lift().unwrap_or_else(|e| panic!("{e}"));
    let raw = fx.spec.abi();
    assert_eq!(
        (raw.resources.sgprs_actual, raw.resources.sgprs_allocated),
        (107, 108)
    );
    assert_eq!(
        (
            raw.resources.ordinary_sgprs_actual(),
            raw.resources.ordinary_sgprs_allocated()
        ),
        (106, 106)
    );
    // The ordinary-only basis (VCC not counted) is equally valid and equally strict.
    fx.spec.resources = RawResources {
        sgprs_actual: 4,
        sgprs_allocated: 106,
        ..Default::default()
    };
    fx.lift().unwrap_or_else(|e| panic!("{e}"));
    fx.spec.resources = RawResources {
        sgprs_actual: 107,
        sgprs_allocated: 108,
        ..Default::default()
    };
    // The total never exceeds the physical limit.
    fx.spec.resources.sgprs_allocated = 109;
    expect_rejected(fx.lift(), Rule::Spec);
    // The ordinary part bounds the code: s0..s2 do not fit a count of 2.
    fx.spec.resources = RawResources {
        sgprs_actual: 2,
        sgprs_allocated: 8,
        ..Default::default()
    };
    let e = expect_rejected(fx.lift(), Rule::Spec);
    assert!(reason(&e).contains("ordinary SGPRs"), "{e}");
    // An entry fact cannot name s106/s107 as ordinary registers, whatever the total says.
    fx.spec.resources = RawResources {
        sgprs_actual: 107,
        sgprs_allocated: 108,
        ..Default::default()
    };
    fx.spec.entry = vec![fact(reg(Kind::S, 106, 1), EntryValue::Defined)];
    expect_rejected(fx.lift(), Rule::Spec);
    // VCC is a special: declare it through `Special`, not as s106.
    fx.spec.entry = vec![
        fact(reg(Kind::S, 0, 4), EntryValue::Defined),
        fact(EntryLoc::Special(Special::VccLo), EntryValue::Defined),
    ];
    fx.lift().unwrap_or_else(|e| panic!("{e}"));
}

#[test]
fn wg_round_robin_is_a_gfx12_only_declaration() {
    let mut fx = Fixture::new(&valid_words());
    fx.spec.fp.wg_round_robin = true;
    let lifted = fx.lift().unwrap_or_else(|e| panic!("{e}"));
    let Abi::Raw(raw) = &lifted.program.kernels[0].abi else {
        panic!("raw ABI")
    };
    assert!(raw.fp.wg_round_robin);
    let mut gfx11 = Fixture::new(&[ENDPGM]);
    gfx11.spec.target.arch = Arch::Gfx1100;
    gfx11.spec.entry = vec![];
    gfx11.spec.resources = RawResources::default();
    gfx11.lift().unwrap_or_else(|e| panic!("{e}"));
    gfx11.spec.fp.wg_round_robin = true;
    let e = expect_rejected(gfx11.lift(), Rule::Spec);
    assert!(reason(&e).contains("wg_round_robin"), "{e}");
}

#[test]
fn malformed_declarations_and_register_overruns_are_rejected() {
    let mut overlap = Fixture::new(&valid_words());
    overlap
        .spec
        .entry
        .push(fact(reg(Kind::V, 3, 1), EntryValue::Defined));
    expect_rejected(overlap.lift(), Rule::Spec);

    let mut outside_allocation = Fixture::new(&valid_words());
    outside_allocation
        .spec
        .entry
        .push(fact(reg(Kind::V, 8, 1), EntryValue::Defined));
    expect_rejected(outside_allocation.lift(), Rule::Spec);

    let mut exact_pair = Fixture::new(&valid_words());
    exact_pair
        .spec
        .entry
        .push(fact(reg(Kind::S, 0, 2), EntryValue::Exact(0)));
    expect_rejected(exact_pair.lift(), Rule::Spec);

    // The code touches v6; declaring 4 actual VGPRs is a false declaration.
    let mut few = Fixture::new(&valid_words());
    few.spec.resources.vgprs_actual = 4;
    let e = expect_rejected(few.lift(), Rule::Spec);
    assert!(reason(&e).contains("VGPRs"), "{e}");
}

#[test]
fn spec_json_schema_parses_exactly_and_rejects_unknown_or_missing_keys() {
    let fx = Fixture::new(&valid_words());
    let mut doc = json!({
        "schema": "pm-raw-kernel-spec-v1",
        "target": {"arch": "gfx1201", "xnack": "any", "sramecc": "any", "abi_version": 0},
        "symbol": "aco_pilot", "wave": 32,
        "binary": "shader.bin", "binary_sha256": hex(&fx.spec.binary_sha256),
        "exec": {"offset": HEADER, "len": fx.spec.extent.len}, "exec_sha256": hex(&fx.spec.exec_sha256),
        "sidecar": "shader.abi.json", "sidecar_sha256": hex(&fx.spec.sidecar_sha256),
        "resources": {"lds_bytes": 0, "scratch_bytes_per_lane": 0, "vgprs_actual": 8, "vgprs_allocated": 8, "sgprs_actual": 0, "sgprs_allocated": 8},
        "fp": {"round32": 0, "round16_64": 0, "denorm32": 0, "denorm16_64": 0, "dx10_clamp": false, "ieee_mode": false,
               "fp16_overflow": false, "wgp_mode": false, "mem_ordered": false, "forward_progress": false, "wg_round_robin": false, "exceptions": 0},
        "relocations": [],
        "entry": [
            {"loc": {"kind": "special", "name": "exec"}, "value": {"kind": "defined"}},
            {"loc": {"kind": "special", "name": "scc"}, "value": {"kind": "defined"}},
            {"loc": {"kind": "vgpr", "base": 2, "len": 2}, "value": {"kind": "defined"}},
            {"loc": {"kind": "vgpr", "base": 5, "len": 2}, "value": {"kind": "defined"}},
            {"loc": {"kind": "ttmp", "base": 9, "len": 1}, "value": {"kind": "workgroup_id", "axis": "x"}},
            {"loc": {"kind": "vgpr", "base": 0, "len": 1}, "value": {"kind": "local_id", "fields": [{"axis": "x", "shift": 0, "bits": 10, "max": 31}]}}
        ]
    });
    let parsed = parse_spec_file(&doc).unwrap();
    assert_eq!(parsed.spec, fx.spec);
    assert_eq!(
        (parsed.binary.as_str(), parsed.sidecar.as_str()),
        ("shader.bin", "shader.abi.json")
    );
    lift_raw(&fx.bytes, &parsed.spec, ACO).unwrap_or_else(|e| panic!("{e}"));

    let mut extra = doc.clone();
    extra["fp"]["flush_denorms"] = json!(true);
    assert!(parse_spec_file(&extra).unwrap_err().contains("unknown key"));
    let mut missing = doc.clone();
    missing["resources"]
        .as_object_mut()
        .unwrap()
        .remove("scratch_bytes_per_lane");
    assert!(parse_spec_file(&missing)
        .unwrap_err()
        .contains("missing key"));
    let mut bad_kind = doc.clone();
    bad_kind["entry"][0]["value"] = json!({"kind": "guess"});
    assert!(parse_spec_file(&bad_kind).is_err());
    doc["wave"] = json!(16);
    assert!(parse_spec_file(&doc).is_err());
}
