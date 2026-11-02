// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! CPU gate of the ACO→HSA adapter (`peacemaker_isa::aco_hsa`), no GPU and no ROCm tool.
//!
//! The raw shader is a small gfx1201 program in the shape of the pilot (RADV entry: a 64-bit
//! pointer in `s[0:1]`, a push constant in `s2`, `ttmp9` work-group id, packed `v0`), encoded by
//! the typed codec from instruction text and lifted by `lift_raw`. The gate:
//!
//! * the emitted HSA object relifts (`lift_object`, `Frontend::Aco`), re-emits byte for byte
//!   (`emit::module`), and its words are exactly prologue ++ raw body;
//! * the body extracted from the ELF by the reported range equals the raw executable range;
//! * the prologue's temporaries come from liveness and the whole object passes the shared
//!   certifier with no open obligation;
//! * deliberate kernarg / entry-fact / FP / resource / hash / aperture mutations are refused,
//!   each with the causal error class, never a permissive emit;
//! * the static import chain (artifact.json → files → rawspec/sidecar/binary/SPIR-V) refuses a
//!   tampered file, a missing entry and missing provenance.
use peacemaker_ir::codec::gfx12;
use peacemaker_ir::inst::{
    Abi, Arch, Axis, EntryFact, EntryLoc, EntryValue, ExecExtent, FpState, Frontend, LocalIdField,
    RawKernelSpec, RawResources, Setting, SymbolId, Target, Wave,
};
use peacemaker_ir::operand::Special;
use peacemaker_ir::reg::{Kind, RegRef};
use peacemaker_isa::aco_cert::{certify_program, CertError, IdentityDomain, LaunchContract};
use peacemaker_isa::aco_hsa::{
    adapt_to_hsa, verify_artifact_dir, AcoError, AcoHipContract, AcoHsaEmission, Aperture,
    BindingSource, EntryBinding, HipArg, HipArgKind, ShapeDomain, CANDIDATE_SCHEMA,
};
use peacemaker_isa::native;
use peacemaker_isa::plan::Access;
use peacemaker_lift::raw::lift_raw;
use peacemaker_lift::text::parse_line;
use peacemaker_lift::{emit, lift_object, Options};
use serde_json::json;
use sha2::{Digest, Sha256};

const ARCH: Arch = Arch::Gfx1201;
const ACO: Options = Options {
    frontend: Frontend::Aco,
};
const SIDECAR: &[u8] = b"{\"schema\":\"pm-aco-abi-v1\"}";
const HEADER: usize = 16;
const TRAILER: usize = 8;

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
fn sgpr(base: u16, len: u8) -> RegRef {
    RegRef {
        kind: Kind::S,
        base,
        len,
    }
}

/// The shader: `if (wg*32 >= N) return; store32(P[lane], wg*32)`; s4 and v1/v2 are its temporaries.
const BODY: [&str; 7] = [
    "s_lshl_b32 s4, ttmp9, 5",
    "s_cmp_ge_u32 s4, s2",
    "s_cbranch_scc1 6",
    "v_lshlrev_b32_e32 v1, 2, v0",
    "v_mov_b32_e32 v2, s4",
    "global_store_b32 v1, v2, s[0:1]",
    "s_wait_storecnt 0x0",
];

fn encode(lines: &[&str]) -> Vec<u32> {
    let mut out = Vec::new();
    for line in lines {
        let inst = parse_line(line, ARCH).unwrap_or_else(|e| panic!("{line}: {e}"));
        let words = gfx12::encode_for(ARCH, &inst).unwrap_or_else(|e| panic!("{line}: {e}"));
        out.extend_from_slice(&words);
    }
    out
}

fn body_words() -> Vec<u32> {
    let mut words = encode(&BODY);
    words.extend(encode(&["s_endpgm"]));
    words
}

struct Fx {
    binary: Vec<u8>,
    spec: RawKernelSpec,
    words: Vec<u32>,
}

impl Fx {
    fn exec(&self) -> Vec<u8> {
        code_of(&self.words)
    }
}

fn fx_plain() -> Fx {
    fixture(|_| {})
}

/// The raw binary (header and trailer data around the executable range) and its spec; `tweak`
/// edits the spec's non-hashed declarations before it is returned.
fn fixture(tweak: impl FnOnce(&mut RawKernelSpec)) -> Fx {
    let words = body_words();
    let code = code_of(&words);
    let mut binary = vec![0xaa; HEADER];
    binary.extend(&code);
    binary.extend([0x55; TRAILER]);
    let mut spec = RawKernelSpec {
        target: Target {
            arch: ARCH,
            xnack: Setting::Any,
            sramecc: Setting::Any,
            abi_version: 0,
        },
        symbol: SymbolId("main".into()),
        wave: Wave::Wave32,
        extent: ExecExtent {
            offset: HEADER as u64,
            len: code.len() as u64,
        },
        binary_sha256: sha(&binary),
        exec_sha256: sha(&code),
        resources: RawResources {
            lds_bytes: 0,
            scratch_bytes_per_lane: 0,
            vgprs_actual: 3,
            vgprs_allocated: 8,
            sgprs_actual: 5,
            sgprs_allocated: 16,
        },
        fp: FpState {
            round32: 0,
            round16_64: 0,
            denorm32: 3,
            denorm16_64: 3,
            dx10_clamp: false,
            ieee_mode: false,
            fp16_overflow: false,
            wgp_mode: false,
            mem_ordered: true,
            forward_progress: true,
            wg_round_robin: false,
            exceptions: 0,
        },
        relocations: Vec::new(),
        entry: vec![
            fact(reg(Kind::S, 0, 2), EntryValue::Defined),
            fact(
                reg(Kind::S, 2, 1),
                EntryValue::Bounded {
                    min: 0,
                    max: 1 << 20,
                },
            ),
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
            fact(EntryLoc::Special(Special::Exec), EntryValue::Defined),
        ],
        sidecar_sha256: sha(SIDECAR),
    };
    tweak(&mut spec);
    Fx {
        binary,
        spec,
        words,
    }
}

fn contract(fx: &Fx) -> AcoHipContract {
    AcoHipContract {
        target: fx.spec.target,
        raw_symbol: fx.spec.symbol.clone(),
        hip_symbol: "toy_store".into(),
        wave: Wave::Wave32,
        args: vec![
            HipArg {
                name: "P".into(),
                offset: 0,
                size: 8,
                kind: HipArgKind::GlobalPointer {
                    access: Some(Access::WriteOnly),
                    extent_bytes: 128,
                },
            },
            HipArg {
                name: "N".into(),
                offset: 8,
                size: 4,
                kind: HipArgKind::ByValue,
            },
        ],
        kernarg_size: 16,
        bindings: vec![
            EntryBinding {
                dst: sgpr(0, 2),
                source: BindingSource::KernargPointer { arg: 0 },
            },
            EntryBinding {
                dst: sgpr(2, 1),
                source: BindingSource::KernargScalar { arg: 1 },
            },
        ],
        fp: fx.spec.fp,
        resources: fx.spec.resources,
        dynamic_lds_max: 0,
        shape: ShapeDomain {
            block: [32, 1, 1],
            grid: [4, 1, 1],
            scalars: vec![(1, 4096)],
            odd_tail_evidence_sha256: None,
        },
        aperture: Aperture::Global64,
        artifact_sha256: [1; 32],
        rawspec_sha256: [3; 32],
        spirv_sha256: [2; 32],
        raw_binary_sha256: fx.spec.binary_sha256,
        raw_exec_sha256: fx.spec.exec_sha256,
        sidecar_sha256: fx.spec.sidecar_sha256,
    }
}

fn launch(c: &AcoHipContract) -> LaunchContract {
    LaunchContract {
        grid: c.shape.grid,
        block: c.shape.block,
        dynamic_lds_max: 0,
        bound_inputs: vec![("P".into(), 128)],
    }
}

fn adapt(fx: &Fx, c: &AcoHipContract) -> Result<AcoHsaEmission, AcoError> {
    let lifted = lift_raw(&fx.binary, &fx.spec, ACO).expect("the raw binary lifts");
    let cert = certify_program(lifted, &fx.spec.symbol, &launch(c))
        .expect("the raw program certifies with no open obligation");
    adapt_to_hsa(&cert, c)
}

fn emitted() -> (Fx, AcoHipContract, AcoHsaEmission) {
    let fx = fx_plain();
    let c = contract(&fx);
    let e = adapt(&fx, &c).unwrap_or_else(|e| panic!("adapt_to_hsa: {e}"));
    (fx, c, e)
}

#[test]
fn emitted_object_relifts_exactly_and_preserves_the_body() {
    let (fx, _, e) = emitted();
    let exec = fx.exec();

    // The extracted body is the raw executable range, byte for byte, and as hashed.
    let body = &e.elf[e.body.offset as usize..(e.body.offset + e.body.len) as usize];
    assert_eq!(body, exec.as_slice());
    assert_eq!(body, &fx.binary[HEADER..HEADER + exec.len()]);
    assert_eq!(e.body.sha256, fx.spec.exec_sha256);
    assert_eq!(
        e.body.offset,
        e.prologue.offset + e.prologue.len,
        "the body follows the prologue with nothing between"
    );
    let prologue =
        &e.elf[e.prologue.offset as usize..(e.prologue.offset + e.prologue.len) as usize];
    assert_eq!(e.prologue.sha256, sha(prologue));

    // The object lifts back as an ACO-frontend HSA program and re-emits exactly.
    let relifted = lift_object(&e.elf, ACO).expect("the emitted object lifts");
    assert_eq!(emit::module(&relifted.program).expect("re-emit"), e.elf);
    let kernel = &relifted.program.kernels[0];
    assert!(matches!(kernel.abi, Abi::Hsa { .. }));
    assert_eq!(kernel.symbol.0, "toy_store");
    assert_eq!(relifted.program.target.abi_version, 6);
    let relifted_words = emit::words(kernel, ARCH).expect("typed re-encode");
    assert_eq!(code_of(&relifted_words), [prologue, body].concat());
    assert_eq!(
        &relifted_words[e.proof.prologue_words..],
        fx.words.as_slice()
    );
    assert_eq!(e.proof.body_words, fx.words.len());

    // Branch layout is unchanged: raw block ends shifted by the prologue.
    let raw = lift_raw(&fx.binary, &fx.spec, ACO).unwrap();
    let raw_ends: Vec<usize> = raw.program.kernels[0]
        .body
        .blocks
        .iter()
        .map(|b| b.range.1 + e.proof.prologue_insts)
        .collect();
    let back_ends: Vec<usize> = kernel
        .body
        .blocks
        .iter()
        .map(|b| b.range.1)
        .filter(|&end| end != e.proof.prologue_insts)
        .collect();
    assert_eq!(raw_ends, back_ends);

    // The HIP offload bundle carries the same object.
    let bundled = lift_object(&e.bundle, ACO).expect("the bundle lifts");
    assert_eq!(
        emit::words(&bundled.program.kernels[0], ARCH).unwrap(),
        relifted_words
    );

    // Shared-analysis certificate of the HSA program: closed, ACO-labelled, HSA domain.
    let report = &e.hsa_certificate.report;
    assert_eq!(report.domain, IdentityDomain::Hsa);
    assert_eq!(report.frontend, Frontend::Aco);
    assert!(report.obligations.is_empty() && e.hsa_certificate.analyzed.obligations.is_empty());
    assert_eq!(report.input_sha256, sha(&e.elf));
    assert_eq!(e.elf_sha256, sha(&e.elf));
}

#[test]
fn prologue_binds_every_entry_register_with_liveness_chosen_temporaries() {
    let (_, c, e) = emitted();
    assert_eq!(
        e.proof.prologue_text,
        [
            "s_mov_b64 s[4:5], s[0:1]",
            "s_load_b64 s[0:1], s[4:5], 0x0",
            "s_load_b32 s2, s[4:5], 0x8",
            "s_wait_kmcnt 0x0",
        ]
    );
    // s0/s1 are bound entry registers, s2 too; s4/s5 are the lowest pair neither bound nor live at the body entry.
    assert!(e.proof.kernarg_base_copied);
    assert_eq!(e.proof.kernarg_base, sgpr(4, 2));
    assert_eq!(e.proof.temps, [sgpr(4, 2)]);
    // The pointer is loaded whole (two dwords), never truncated.
    assert!(e
        .proof
        .prologue_text
        .iter()
        .any(|l| l.starts_with("s_load_b64 s[0:1]")));
    // Hardware-identical entry state is recorded, not re-created.
    for want in [
        "ttmp9 = work-group id x",
        "v0 = packed local ids",
        "exec_lo = launch mask",
    ] {
        assert!(e.proof.hardware_entry.iter().any(|h| h == want), "{want}");
    }
    // Every prologue write is a temporary or a bound destination.
    for w in &e.proof.prologue_writes {
        assert!(
            e.proof.temps.contains(w) || c.bindings.iter().any(|b| b.dst == *w),
            "{w}"
        );
    }
}

#[test]
fn descriptor_and_metadata_agree_with_the_contract() {
    let (fx, c, e) = emitted();
    let relifted = lift_object(&e.elf, ACO).unwrap();
    let Abi::Hsa {
        descriptor,
        metadata,
    } = &relifted.program.kernels[0].abi
    else {
        panic!("HSA ABI")
    };
    assert_eq!(descriptor.kernarg_size, c.kernarg_size);
    assert_eq!(
        descriptor.group_segment_fixed_size,
        fx.spec.resources.lds_bytes
    );
    assert_eq!(descriptor.private_segment_fixed_size, 0);
    assert!(descriptor.kernel_code_properties.wave32());
    // user SGPRs: the kernarg segment pointer only; system SGPR work-group id x only; v0 holds x.
    let rsrc2 = descriptor.compute_pgm_rsrc2.0;
    assert_eq!(rsrc2 >> 1 & 0x1f, 2);
    assert_eq!(rsrc2 >> 7 & 1, 1);
    assert_eq!(rsrc2 >> 8 & 3, 0);
    assert_eq!(rsrc2 >> 11 & 3, 0);
    // The raw shader's FP state, field for field.
    let rsrc1 = descriptor.compute_pgm_rsrc1.0;
    assert_eq!(rsrc1 >> 12 & 3, u32::from(fx.spec.fp.round32));
    assert_eq!(rsrc1 >> 14 & 3, u32::from(fx.spec.fp.round16_64));
    assert_eq!(rsrc1 >> 16 & 3, u32::from(fx.spec.fp.denorm32));
    assert_eq!(rsrc1 >> 18 & 3, u32::from(fx.spec.fp.denorm16_64));
    assert_eq!(rsrc1 >> 26 & 1, u32::from(fx.spec.fp.fp16_overflow));
    assert_eq!(rsrc1 >> 29 & 1, u32::from(fx.spec.fp.wgp_mode));
    assert_eq!(rsrc1 >> 30 & 1, u32::from(fx.spec.fp.mem_ordered));
    assert_eq!(rsrc1 >> 31 & 1, u32::from(fx.spec.fp.forward_progress));
    assert_eq!(rsrc1 >> 21 & 1, u32::from(fx.spec.fp.wg_round_robin));
    // Launch ABI preserved: the original arguments at the original offsets.
    let meta = &metadata.parsed;
    assert_eq!(meta.kernarg_segment_size, 16);
    assert_eq!(meta.max_flat_workgroup_size, 32);
    let args: Vec<(&str, u32, u32, &str)> = meta
        .args
        .iter()
        .map(|a| (a.name.as_str(), a.offset, a.size, a.value_kind.as_str()))
        .collect();
    assert_eq!(
        args,
        [("P", 0, 8, "global_buffer"), ("N", 8, 4, "by_value")]
    );
    assert!(
        meta.vgpr_count >= 3 && meta.sgpr_count >= 6,
        "{} VGPRs / {} SGPRs cover prologue and body",
        meta.vgpr_count,
        meta.sgpr_count
    );
}

#[test]
fn candidate_manifest_binds_every_identity() {
    let (fx, c, e) = emitted();
    for image in [&e.elf, &e.bundle] {
        let m = e.candidate_manifest(image);
        assert_eq!(m["schema"], CANDIDATE_SCHEMA);
        assert_eq!(m["symbol"], "toy_store");
        assert_eq!(m["object_sha256"], hex(&sha(image)));
        assert_eq!(m["spirv_sha256"], hex(&c.spirv_sha256));
        assert_eq!(
            m["aco_binary_sha256"],
            hex(&fx.spec.binary_sha256),
            "whole shader.bin, not the executable range"
        );
        assert_eq!(m["aco_exec_sha256"], hex(&fx.spec.exec_sha256));
        assert_eq!(m["aco_sidecar_sha256"], hex(&fx.spec.sidecar_sha256));
        assert_eq!(m["body_sha256"], hex(&fx.spec.exec_sha256));
        assert_eq!(m["wave_size"], 32);
        assert_eq!(m["block"], json!([32, 1, 1]));
        assert_eq!(m["kernarg_size"], 16);
        // Pointer slot zeroed, by-value N (4096 at offset 8) retained: the captured launch ABI.
        let mut template = [0u8; 16];
        template[8..12].copy_from_slice(&4096u32.to_le_bytes());
        assert_eq!(m["kernarg_abi_sha256"], hex(&sha(&template)));
        assert_eq!(m["shape_support"]["fixture"], true);
        assert_eq!(m["shape_support"]["odd_tail"], false);
        assert_eq!(m["adapter"]["contract_sha256"], hex(&c.sha256()));
        assert_eq!(m["adapter"]["artifact_sha256"], hex(&c.artifact_sha256));
        assert_eq!(m["adapter"]["rawspec_sha256"], hex(&c.rawspec_sha256));
    }
    assert_ne!(
        fx.spec.binary_sha256, fx.spec.exec_sha256,
        "the fixture binary has data around the executable range"
    );
}

#[test]
fn odd_tail_is_reported_only_with_evidence() {
    let fx = fx_plain();
    let mut c = contract(&fx);
    c.shape.odd_tail_evidence_sha256 = Some([7; 32]);
    let e = adapt(&fx, &c).expect("evidence-backed claim");
    let m = e.candidate_manifest(&e.elf);
    assert_eq!(m["shape_support"]["odd_tail"], true);
    assert_eq!(
        m["shape_support"]["odd_tail_evidence_sha256"],
        hex(&[7; 32])
    );
}

#[test]
fn native_link_of_encoded_words_matches_the_text_assembler() {
    // The encoded-word entry and the text front end share the descriptor/metadata/ELF writer:
    // the same kernel through both must be the same object.
    let words = body_words();
    let directives: Vec<(String, String)> = [
        ("group_segment_fixed_size", "0"),
        ("private_segment_fixed_size", "0"),
        ("kernarg_size", "8"),
        ("user_sgpr_count", "2"),
        ("user_sgpr_dispatch_ptr", "0"),
        ("user_sgpr_queue_ptr", "0"),
        ("user_sgpr_kernarg_segment_ptr", "1"),
        ("user_sgpr_dispatch_id", "0"),
        ("user_sgpr_private_segment_size", "0"),
        ("wavefront_size32", "1"),
        ("uses_dynamic_stack", "0"),
        ("enable_private_segment", "0"),
        ("system_sgpr_workgroup_id_x", "1"),
        ("system_sgpr_workgroup_id_y", "0"),
        ("system_sgpr_workgroup_id_z", "0"),
        ("system_sgpr_workgroup_info", "0"),
        ("system_vgpr_workitem_id", "0"),
        ("next_free_vgpr", "8"),
        ("next_free_sgpr", "8"),
        ("reserve_vcc", "0"),
        ("float_round_mode_32", "0"),
        ("float_round_mode_16_64", "0"),
        ("float_denorm_mode_32", "3"),
        ("float_denorm_mode_16_64", "3"),
        ("fp16_overflow", "0"),
        ("workgroup_processor_mode", "1"),
        ("memory_ordered", "1"),
        ("forward_progress", "1"),
        ("round_robin_scheduling", "0"),
        ("exception_fp_ieee_invalid_op", "0"),
        ("exception_fp_denorm_src", "0"),
        ("exception_fp_ieee_div_zero", "0"),
        ("exception_fp_ieee_overflow", "0"),
        ("exception_fp_ieee_underflow", "0"),
        ("exception_fp_ieee_inexact", "0"),
        ("exception_int_div_zero", "0"),
    ]
    .iter()
    .map(|(n, v)| (format!(".amdhsa_{n}"), (*v).to_owned()))
    .collect();
    let yaml = "amdhsa.kernels:\n  - .args:\n      - \n        .name: p\n        .offset: 0\n        .size: 8\n        .value_kind: by_value\n    .group_segment_fixed_size: 0\n    .kernarg_segment_align: 8\n    .kernarg_segment_size: 8\n    .max_flat_workgroup_size: 32\n    .name: k\n    .private_segment_fixed_size: 0\n    .sgpr_count: 8\n    .sgpr_spill_count: 0\n    .symbol: k.kd\n    .uniform_work_group_size: 1\n    .uses_dynamic_stack: false\n    .vgpr_count: 8\n    .vgpr_spill_count: 0\n    .wavefront_size: 32\n    .workgroup_processor_mode: 1\namdhsa.target: amdgcn-amd-amdhsa--gfx1201\namdhsa.version: [1, 2]\n";
    let linked = native::link_encoded(
        &[native::EncodedKernel {
            symbol: "k",
            words: &words,
            directives: &directives,
        }],
        yaml,
        peacemaker_isa::Arch::Gfx1201,
    )
    .expect("link");
    let mut source = String::from(".amdgcn_target \"amdgcn-amd-amdhsa--gfx1201\"\n.amdhsa_code_object_version 6\n.text\n.protected k\n.globl k\n.p2align 8\n.type k,@function\nk:\n");
    for line in BODY {
        source.push_str(&format!(
            "\t{}\n",
            line.replace("s_cbranch_scc1 6", "s_cbranch_scc1 .Lend")
        ))
    }
    source.push_str(".Lend:\n\ts_endpgm\n.Lk_end:\n.size k, .Lk_end-k\n.section .rodata,\"a\",@progbits\n.p2align 6\n.amdhsa_kernel k\n");
    for (name, value) in &directives {
        if name == ".amdhsa_round_robin_scheduling" {
            source.push_str("\t.amdhsa_inst_pref_size ((instprefsize(.Lk_end-k)<<4)&4080)>>4\n")
        }
        source.push_str(&format!("\t{name} {value}\n"));
    }
    source.push_str(".end_amdhsa_kernel\n.text\n.amdgpu_metadata\n---\n");
    source.push_str(yaml);
    source.push_str("...\n.end_amdgpu_metadata\n");
    let assembled = native::assemble(&source, peacemaker_isa::Arch::Gfx1201).expect("assemble");
    assert_eq!(linked, assembled);
}

// ---- refusals -----------------------------------------------------------------------

fn refused(fx: &Fx, c: &AcoHipContract) -> AcoError {
    match adapt(fx, c) {
        Ok(_) => panic!("the mutated cell was accepted"),
        Err(e) => e,
    }
}

#[test]
fn kernarg_mutations_are_refused() {
    let fx = fx_plain();
    let base = contract(&fx);

    // A truncated (4-byte) pointer arg.
    let mut c = base.clone();
    c.args[0].size = 4;
    assert!(matches!(refused(&fx, &c), AcoError::Contract(m) if m.contains("full 64 bits")));

    // A misaligned pointer arg.
    let mut c = base.clone();
    c.args[0].offset = 4;
    c.args[1].offset = 12;
    assert!(matches!(refused(&fx, &c), AcoError::Contract(_)));

    // Kernarg segment smaller than its arguments.
    let mut c = base.clone();
    c.kernarg_size = 8;
    assert!(matches!(refused(&fx, &c), AcoError::Contract(_)));

    // Overlapping arguments.
    let mut c = base.clone();
    c.args[1].offset = 4;
    assert!(matches!(refused(&fx, &c), AcoError::Contract(m) if m.contains("overlaps")));

    // A pointer fact bound as a by-value scalar, and the reverse.
    let mut c = base.clone();
    c.bindings[0].source = BindingSource::KernargScalar { arg: 1 };
    assert!(matches!(refused(&fx, &c), AcoError::Contract(_)));
    let mut c = base.clone();
    c.bindings[1].source = BindingSource::KernargPointer { arg: 0 };
    assert!(matches!(refused(&fx, &c), AcoError::Contract(_)));

    // A binding moved off the entry fact it must satisfy: s2 is defined but nothing initialises it.
    let mut c = base.clone();
    c.bindings[1].dst = sgpr(3, 1);
    assert!(matches!(refused(&fx, &c), AcoError::Entry(m) if m.contains("s2")));

    // A fact with no binding at all.
    let mut c = base.clone();
    c.bindings.pop();
    assert!(matches!(refused(&fx, &c), AcoError::Entry(_)));

    // A binding whose value the entry fact does not admit: the fixed N is outside Bounded.
    let mut c = base.clone();
    c.shape.scalars = vec![(1, 1 << 21)];
    assert!(matches!(refused(&fx, &c), AcoError::Entry(_)));

    // A by-value arg without a fixed shape value.
    let mut c = base;
    c.shape.scalars.clear();
    assert!(matches!(refused(&fx, &c), AcoError::Contract(_)));
}

#[test]
fn unbound_entry_sgprs_are_accepted_only_when_dead_at_body_entry() {
    // s3 is declared defined (like RADV's unused ring_offsets slot) but the body never reads it:
    // no binding is demanded and nothing is fabricated into it.
    let fx = fixture(|s| s.entry.push(fact(reg(Kind::S, 3, 1), EntryValue::Defined)));
    let c = contract(&fx);
    let e = adapt(&fx, &c).expect("a dead unbound declared entry SGPR is not a rejection");
    assert!(e
        .proof
        .hardware_entry
        .iter()
        .any(|h| h.starts_with("s3 =") && h.contains("dead")));
    assert!(
        !e.proof
            .prologue_writes
            .iter()
            .any(|r| r.base <= 3 && 3 < r.base + u16::from(r.len)),
        "the prologue does not write s3"
    );
    // The same slot read by the body (s2 without its binding) is refused with the liveness reason.
    let fx = fx_plain();
    let mut c = contract(&fx);
    c.bindings.pop();
    assert!(matches!(refused(&fx, &c), AcoError::Entry(m) if m.contains("read by the body")));
}

#[test]
fn entry_facts_without_a_source_are_refused() {
    let base_fx = fx_plain();
    let c = contract(&base_fx);
    for (what, extra) in [
        (
            "ttmp8",
            fact(EntryLoc::Special(Special::Ttmp(8)), EntryValue::Defined),
        ),
        (
            "vgpr other than v0",
            fact(reg(Kind::V, 1, 1), EntryValue::Defined),
        ),
        (
            "vcc",
            fact(EntryLoc::Special(Special::Vcc), EntryValue::Defined),
        ),
        (
            "m0",
            fact(EntryLoc::Special(Special::M0), EntryValue::Defined),
        ),
    ] {
        let fx = fixture(|s| s.entry.push(extra));
        let err = refused(&fx, &c);
        assert!(matches!(err, AcoError::Entry(_)), "{what}: {err}");
    }
    // Packed local ids at a position HSA does not pack them.
    let fx = fixture(|s| {
        s.entry[3] = fact(
            reg(Kind::V, 0, 1),
            EntryValue::LocalId(vec![LocalIdField {
                axis: Axis::X,
                shift: 2,
                bits: 10,
                max: 31,
            }]),
        );
    });
    assert!(matches!(refused(&fx, &c), AcoError::Entry(m) if m.contains("packs")));
    // A local-id bound smaller than the launch block.
    let fx = fixture(|s| {
        s.entry[3] = fact(
            reg(Kind::V, 0, 1),
            EntryValue::LocalId(vec![LocalIdField {
                axis: Axis::X,
                shift: 0,
                bits: 10,
                max: 15,
            }]),
        );
    });
    assert!(matches!(refused(&fx, &c), AcoError::Entry(m) if m.contains("bound")));
    // A block with items along an axis the ids do not carry.
    let fx = fx_plain();
    let mut c = contract(&fx);
    c.shape.block = [16, 2, 1];
    assert!(matches!(refused(&fx, &c), AcoError::Entry(_)));
}

#[test]
fn fp_mutations_are_refused() {
    let fx = fx_plain();
    let base = contract(&fx);
    // The contract's FP state must be the shader config's, bit for bit.
    for tweak in [
        (|f: &mut FpState| f.denorm32 = 0) as fn(&mut FpState),
        |f| f.denorm16_64 = 0,
        |f| f.round32 = 1,
        |f| f.fp16_overflow = true,
        |f| f.mem_ordered = false,
        |f| f.forward_progress = false,
        |f| f.wg_round_robin = true,
        |f| f.wgp_mode = true,
        |f| f.exceptions = 1,
    ] {
        let mut c = base.clone();
        tweak(&mut c.fp);
        assert!(matches!(refused(&fx, &c), AcoError::Mismatch(m) if m.contains("FP")));
    }
    // State the gfx12 descriptor cannot express is refused even when contract and shader agree.
    for (what, tweak) in [
        (
            "dx10_clamp",
            (|f: &mut FpState| f.dx10_clamp = true) as fn(&mut FpState),
        ),
        ("ieee_mode", |f| f.ieee_mode = true),
        ("address-watch exception", |f| f.exceptions = 0x80),
        ("memory-violation exception", |f| f.exceptions = 0x100),
    ] {
        let fx = fixture(|s| tweak(&mut s.fp));
        let c = contract(&fx);
        assert!(
            matches!(refused(&fx, &c), AcoError::Unsupported(_)),
            "{what}"
        );
    }
}

#[test]
fn resource_mutations_are_refused() {
    let fx = fx_plain();
    let base = contract(&fx);
    for tweak in [
        (|r: &mut RawResources| r.lds_bytes = 4096) as fn(&mut RawResources),
        |r| r.vgprs_allocated = 16,
        |r| r.vgprs_actual = 4,
        |r| r.sgprs_allocated = 32,
        |r| r.scratch_bytes_per_lane = 16,
    ] {
        let mut c = base.clone();
        tweak(&mut c.resources);
        assert!(matches!(refused(&fx, &c), AcoError::Mismatch(m) if m.contains("resources")));
    }
    // Scratch declared by the shader itself: nothing certifies scratch bounds.
    let scratch = fixture(|s| s.resources.scratch_bytes_per_lane = 16);
    assert!(
        matches!(refused(&scratch, &contract(&scratch)), AcoError::Unsupported(m) if m.contains("scratch"))
    );
    // Dynamic LDS has no certified bound.
    let mut c = base.clone();
    c.dynamic_lds_max = 1024;
    assert!(matches!(refused(&fx, &c), AcoError::Unsupported(m) if m.contains("dynamic LDS")));
    // A 32-bit or scratch aperture is not representable on HIP VMM.
    for aperture in [
        Aperture::Pointer32 {
            high_dword: 0xffff_8000,
        },
        Aperture::FlatScratch,
    ] {
        let mut c = base.clone();
        c.aperture = aperture;
        assert!(matches!(refused(&fx, &c), AcoError::Unsupported(m) if m.contains("aperture")));
    }
    // Wave64 and a different target.
    let mut c = base.clone();
    c.wave = Wave::Wave64;
    assert!(matches!(refused(&fx, &c), AcoError::Unsupported(_)));
    let mut c = base;
    c.target.arch = Arch::Gfx1100;
    assert!(matches!(refused(&fx, &c), AcoError::Mismatch(_)));
}

#[test]
fn identity_mutations_are_refused() {
    let fx = fx_plain();
    let base = contract(&fx);
    let mut c = base.clone();
    c.raw_exec_sha256[0] ^= 1;
    assert!(matches!(refused(&fx, &c), AcoError::Body(_)));
    let mut c = base.clone();
    c.raw_binary_sha256[0] ^= 1;
    assert!(matches!(refused(&fx, &c), AcoError::Mismatch(m) if m.contains("whole-binary")));
    let mut c = base.clone();
    c.sidecar_sha256[0] ^= 1;
    assert!(matches!(refused(&fx, &c), AcoError::Mismatch(m) if m.contains("sidecar")));
    let mut c = base.clone();
    c.raw_symbol = SymbolId("other".into());
    assert!(matches!(refused(&fx, &c), AcoError::Mismatch(_)));
    let mut c = base;
    c.hip_symbol = "bad symbol".into();
    assert!(matches!(refused(&fx, &c), AcoError::Contract(_)));
}

#[test]
fn certifier_names_the_frontend_and_refuses_unbounded_launches() {
    let fx = fx_plain();
    let c = contract(&fx);
    let lifted = lift_raw(&fx.binary, &fx.spec, ACO).unwrap();
    let mut bad = launch(&c);
    bad.dynamic_lds_max = 64;
    assert!(matches!(
        certify_program(lifted.clone(), &fx.spec.symbol, &bad),
        Err(CertError::Launch(_))
    ));
    let mut bad = launch(&c);
    bad.block = [2048, 1, 1];
    assert!(matches!(
        certify_program(lifted.clone(), &fx.spec.symbol, &bad),
        Err(CertError::Launch(_))
    ));
    assert!(matches!(
        certify_program(lifted.clone(), &SymbolId("absent".into()), &launch(&c)),
        Err(CertError::NoKernel(_))
    ));
    let cert = certify_program(lifted, &fx.spec.symbol, &launch(&c)).unwrap();
    assert_eq!(cert.report.frontend, Frontend::Aco);
    assert_eq!(cert.report.domain, IdentityDomain::Raw);
    assert_eq!(cert.report.sidecar_sha256, Some(fx.spec.sidecar_sha256));
    assert!(cert.report.obligations.is_empty());
    // An HSA certificate is not a raw certificate: the adapter refuses it as input.
    let e = adapt(&fx, &c).unwrap();
    assert!(matches!(
        adapt_to_hsa(&e.hsa_certificate, &c),
        Err(AcoError::Certificate(_))
    ));
}

#[test]
fn contract_json_round_trips_and_rejects_unknown_or_missing_keys() {
    let fx = fx_plain();
    let c = contract(&fx);
    let fp = &fx.spec.fp;
    let r = &fx.spec.resources;
    let doc = json!({
        "schema": "pm-aco-hip-contract-v1",
        "target": {"arch": "gfx1201", "xnack": "any", "sramecc": "any", "abi_version": 0},
        "raw_symbol": "main", "hip_symbol": "toy_store", "wave": 32, "kernarg_size": 16,
        "args": [
            {"name": "P", "offset": 0, "size": 8, "kind": "global_pointer", "access": "write_only", "extent_bytes": 128},
            {"name": "N", "offset": 8, "size": 4, "kind": "by_value", "access": null, "extent_bytes": null},
        ],
        "bindings": [
            {"dst": {"kind": "sgpr", "base": 0, "len": 2}, "source": {"kind": "kernarg_pointer", "arg": 0}},
            {"dst": {"kind": "sgpr", "base": 2, "len": 1}, "source": {"kind": "kernarg_scalar", "arg": 1}},
        ],
        "fp": {"round32": fp.round32, "round16_64": fp.round16_64, "denorm32": fp.denorm32, "denorm16_64": fp.denorm16_64,
            "dx10_clamp": fp.dx10_clamp, "ieee_mode": fp.ieee_mode, "fp16_overflow": fp.fp16_overflow, "wgp_mode": fp.wgp_mode,
            "mem_ordered": fp.mem_ordered, "forward_progress": fp.forward_progress, "wg_round_robin": fp.wg_round_robin, "exceptions": fp.exceptions},
        "resources": {"lds_bytes": r.lds_bytes, "scratch_bytes_per_lane": r.scratch_bytes_per_lane, "vgprs_actual": r.vgprs_actual,
            "vgprs_allocated": r.vgprs_allocated, "sgprs_actual": r.sgprs_actual, "sgprs_allocated": r.sgprs_allocated},
        "dynamic_lds_max": 0,
        "shape": {"block": [32, 1, 1], "grid": [4, 1, 1], "scalars": [{"arg": 1, "value": 4096}], "odd_tail_evidence_sha256": null},
        "aperture": "global64",
        "artifact_sha256": hex(&[1; 32]), "rawspec_sha256": hex(&[3; 32]), "spirv_sha256": hex(&[2; 32]),
        "raw_binary_sha256": hex(&fx.spec.binary_sha256), "raw_exec_sha256": hex(&fx.spec.exec_sha256), "sidecar_sha256": hex(&fx.spec.sidecar_sha256),
    });
    let parsed = AcoHipContract::from_json(&doc.to_string()).expect("parses");
    assert_eq!(parsed, c);
    assert_eq!(parsed.sha256(), c.sha256());
    parsed.validate().unwrap();

    let mut extra = doc.clone();
    extra["surprise"] = json!(1);
    assert!(
        matches!(AcoHipContract::from_json(&extra.to_string()), Err(AcoError::Contract(m)) if m.contains("surprise"))
    );
    let mut missing = doc.clone();
    missing.as_object_mut().unwrap().remove("fp");
    assert!(
        matches!(AcoHipContract::from_json(&missing.to_string()), Err(AcoError::Contract(m)) if m.contains("fp"))
    );
    let mut wrong = doc;
    wrong["shape"]["odd_tail_evidence_sha256"] = json!("not a hash");
    assert!(matches!(
        AcoHipContract::from_json(&wrong.to_string()),
        Err(AcoError::Contract(_))
    ));

    // Every contract field is in the identity hash.
    let mut changed = c.clone();
    changed.shape.grid = [8, 1, 1];
    assert_ne!(changed.sha256(), c.sha256());
    let mut changed = c.clone();
    changed.artifact_sha256[0] ^= 1;
    assert_ne!(changed.sha256(), c.sha256());
    let mut changed = c.clone();
    changed.rawspec_sha256[0] ^= 1;
    assert_ne!(changed.sha256(), c.sha256());
}

// ---- static import chain ------------------------------------------------------------

fn artifact_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("aco_hsa_artifact_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let files: [(&str, &[u8]); 5] = [
        ("shader.spv", b"spirv"),
        ("shader.bin", b"binary"),
        ("shader.abi.json", SIDECAR),
        (
            "shader.rawspec.json",
            b"{\"schema\":\"pm-raw-kernel-spec-v1\"}",
        ),
        ("shader.aco.dump", b"dump"),
    ];
    let mut listed = serde_json::Map::new();
    for (name, bytes) in files {
        std::fs::write(dir.join(name), bytes).unwrap();
        listed.insert(name.into(), json!({"sha256": hex(&sha(bytes))}));
    }
    let artifact = json!({
        "schema": "pm-aco-artifact-v1",
        "files": listed,
        "provenance": {"compiler_sha256": hex(&[4; 32]), "mesa_patch_sha256": hex(&[5; 32]), "mesa_commit": "deadbeef"},
        "binary": {"sha256": hex(&sha(b"binary")), "executable": {"offset": 0, "len": 4, "sha256": hex(&sha(b"bina"))}},
    });
    std::fs::write(
        dir.join("artifact.json"),
        serde_json::to_vec_pretty(&artifact).unwrap(),
    )
    .unwrap();
    dir
}

#[test]
fn artifact_import_chain_binds_every_file_hash_and_provenance() {
    let dir = artifact_dir("ok");
    let binding = verify_artifact_dir(&dir).expect("a consistent artifact verifies");
    assert_eq!(binding.spirv_sha256, sha(b"spirv"));
    assert_eq!(binding.binary_sha256, sha(b"binary"));
    assert_eq!(binding.sidecar_sha256, sha(SIDECAR));
    assert_eq!(
        binding.rawspec_sha256,
        sha(b"{\"schema\":\"pm-raw-kernel-spec-v1\"}")
    );
    assert_eq!(
        binding.artifact_sha256,
        sha(&std::fs::read(dir.join("artifact.json")).unwrap())
    );
    assert_eq!(binding.compiler_sha256, [4; 32]);
    assert_eq!(binding.mesa_patch_sha256, [5; 32]);
    assert_eq!(binding.mesa_commit, "deadbeef");
    assert_eq!(
        (binding.exec_offset, binding.exec_len, binding.exec_sha256),
        (0, 4, sha(b"bina"))
    );
    assert_eq!(binding.files, 5);

    // A hand-edited rawspec (entry/resource/FP declarations) no longer matches the recorded hash.
    std::fs::write(
        dir.join("shader.rawspec.json"),
        b"{\"schema\":\"pm-raw-kernel-spec-v1\",\"edited\":true}",
    )
    .unwrap();
    assert!(
        matches!(verify_artifact_dir(&dir), Err(AcoError::Artifact(m)) if m.contains("shader.rawspec.json"))
    );
    // Same for the sidecar, binary and SPIR-V.
    for name in [
        "shader.abi.json",
        "shader.bin",
        "shader.spv",
        "shader.aco.dump",
    ] {
        let dir = artifact_dir(&format!("tamper_{name}"));
        std::fs::write(dir.join(name), b"tampered").unwrap();
        assert!(
            matches!(verify_artifact_dir(&dir), Err(AcoError::Artifact(m)) if m.contains(name)),
            "{name}"
        );
    }
    // A missing file entry, missing provenance and an unknown schema are refused.
    let dir = artifact_dir("missing");
    let mut artifact: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("artifact.json")).unwrap()).unwrap();
    artifact["files"]
        .as_object_mut()
        .unwrap()
        .remove("shader.spv");
    std::fs::write(
        dir.join("artifact.json"),
        serde_json::to_vec(&artifact).unwrap(),
    )
    .unwrap();
    assert!(
        matches!(verify_artifact_dir(&dir), Err(AcoError::Artifact(m)) if m.contains("shader.spv"))
    );
    let dir = artifact_dir("provenance");
    let mut artifact: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("artifact.json")).unwrap()).unwrap();
    artifact["provenance"]
        .as_object_mut()
        .unwrap()
        .remove("mesa_patch_sha256");
    std::fs::write(
        dir.join("artifact.json"),
        serde_json::to_vec(&artifact).unwrap(),
    )
    .unwrap();
    assert!(
        matches!(verify_artifact_dir(&dir), Err(AcoError::Artifact(m)) if m.contains("mesa_patch_sha256"))
    );
    let dir = artifact_dir("schema");
    let mut artifact: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("artifact.json")).unwrap()).unwrap();
    artifact["schema"] = json!("pm-aco-artifact-v0");
    std::fs::write(
        dir.join("artifact.json"),
        serde_json::to_vec(&artifact).unwrap(),
    )
    .unwrap();
    assert!(
        matches!(verify_artifact_dir(&dir), Err(AcoError::Artifact(m)) if m.contains("schema"))
    );
    // The recorded executable range must be the hashed range of shader.bin.
    let dir = artifact_dir("exec");
    let mut artifact: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("artifact.json")).unwrap()).unwrap();
    artifact["binary"]["executable"]["sha256"] = json!(hex(&[9; 32]));
    std::fs::write(
        dir.join("artifact.json"),
        serde_json::to_vec(&artifact).unwrap(),
    )
    .unwrap();
    assert!(
        matches!(verify_artifact_dir(&dir), Err(AcoError::Artifact(m)) if m.contains("executable"))
    );
    // A file the artifact does not list is refused.
    let dir = artifact_dir("unlisted");
    std::fs::write(dir.join("extra.bin"), b"x").unwrap();
    assert!(
        matches!(verify_artifact_dir(&dir), Err(AcoError::Artifact(m)) if m.contains("extra.bin"))
    );

    // The contract must carry exactly the artifact's identities.
    let dir = artifact_dir("contract");
    let binding = verify_artifact_dir(&dir).unwrap();
    let fx = fx_plain();
    let mut c = contract(&fx);
    c.artifact_sha256 = binding.artifact_sha256;
    c.rawspec_sha256 = binding.rawspec_sha256;
    c.spirv_sha256 = binding.spirv_sha256;
    c.raw_binary_sha256 = binding.binary_sha256;
    c.raw_exec_sha256 = binding.exec_sha256;
    c.sidecar_sha256 = binding.sidecar_sha256;
    binding
        .check_contract(&c)
        .expect("a contract built from the artifact matches it");
    for tweak in [
        (|c: &mut AcoHipContract| c.artifact_sha256[0] ^= 1) as fn(&mut AcoHipContract),
        |c| c.rawspec_sha256[0] ^= 1,
        |c| c.spirv_sha256[0] ^= 1,
        |c| c.raw_binary_sha256[0] ^= 1,
        |c| c.raw_exec_sha256[0] ^= 1,
        |c| c.sidecar_sha256[0] ^= 1,
    ] {
        let mut m = c.clone();
        tweak(&mut m);
        assert!(matches!(
            binding.check_contract(&m),
            Err(AcoError::Artifact(_))
        ));
    }
    // The raw spec must be the artifact's: the fixture spec is a different binary.
    assert!(matches!(
        binding.check_raw_spec(&fx.spec),
        Err(AcoError::Artifact(_))
    ));
}

#[test]
fn wg_round_robin_is_mapped_exactly_not_dropped() {
    let fx = fixture(|s| s.fp.wg_round_robin = true);
    let c = contract(&fx);
    let e = adapt(&fx, &c).expect("a representable gfx12 scheduling bit is emitted");
    let relifted = lift_object(&e.elf, ACO).unwrap();
    let Abi::Hsa { descriptor, .. } = &relifted.program.kernels[0].abi else {
        panic!("HSA ABI")
    };
    assert_eq!(descriptor.compute_pgm_rsrc1.0 >> 21 & 1, 1);
    // The contract must state it too.
    let mut off = contract(&fx);
    off.fp.wg_round_robin = false;
    assert!(matches!(refused(&fx, &off), AcoError::Mismatch(m) if m.contains("FP")));
}
