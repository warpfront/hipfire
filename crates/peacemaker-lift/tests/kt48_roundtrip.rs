//! C7: KT48 end-to-end round-trip gates (core.md §8, T1–T10) plus the F2/iu4 bundle round
//! trips, exercised only through public APIs: `lift_object` → typed program → `emit`.
//! Errata (Main): KT48's selected kernel has 25 args (11 explicit + 14 hidden); the sound
//! CFG-aware replay yields 124 DS-store source obligations; KMcnt units in program order
//! are [2,1,1,1,2,2]; T4 counts 57 distinct `Label` targets over 67 branches.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use peacemaker_ir::cfg::{BlockId, Terminator};
use peacemaker_ir::codec::gfx12;
use peacemaker_ir::inst::{Abi, Arch, Form, FormFields, Frontend, Inst, Kernel, KernelOrigin, Program, Wave};
use peacemaker_ir::operand::{Half, Operand};
use peacemaker_ir::state::Lifted;
use peacemaker_lift::{emit, lift_object, LiftError, Options, Rule};
use sha2::{Digest, Sha256};

const SELECTED: &str = "attention_fp8_e4m3_fa2_gqa_qresident_v2_q8_gfx1201";
const SELECTED_VA: u64 = 0x7f00;
const SELECTED_SIZE: u64 = 10_604;
const ARCH: Arch = Arch::Gfx1201;

fn load(relative: &str, sha256: &str) -> Vec<u8> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(relative);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("fixture {}: {e}", path.display()));
    assert_eq!(hex(&Sha256::digest(&bytes)), sha256, "{} is not the pinned fixture", path.display());
    bytes
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn kt48_co() -> Vec<u8> {
    load("tests/fixtures/kt48/hipcc.co", "f876e2ce520ba8d97358b34c903a48dafc5cff2e4cc552d7ffc775361cbcb6bc")
}

fn kt48_hsaco() -> Vec<u8> {
    load("tests/fixtures/kt48/hipcc.hsaco", "3f968bb9dd13a79c4a1b7c5cd805c8d8b2bd1e8dc93821230acabe19df0d0e3a")
}

fn lift(bytes: &[u8], frontend: Frontend) -> Lifted<Program> {
    lift_object(bytes, Options { frontend }).unwrap_or_else(|e| panic!("lift: {e}"))
}

fn selected(program: &Program) -> &Kernel {
    program.kernels.iter().find(|k| k.symbol.0 == SELECTED).expect("selected KT48 kernel")
}

fn selected_mut(program: &mut Program) -> &mut Kernel {
    program.kernels.iter_mut().find(|k| k.symbol.0 == SELECTED).expect("selected KT48 kernel")
}

fn name(inst: &Inst) -> &'static str {
    inst.op.name(ARCH).expect("table opcode")
}

fn layout(kernel: &Kernel) -> Vec<&Inst> {
    kernel.body.layout.iter().map(|&id| kernel.body.insts.get(id).expect("live instruction")).collect()
}

fn width(inst: &Inst) -> usize {
    peacemaker_ir::passes::cfg::dwords_of(inst, peacemaker_ir::inst::Arch::Gfx1201)
}

/// File range of the selected kernel's code, from the lifted envelope.
fn selected_file_range(program: &Program) -> std::ops::Range<usize> {
    use peacemaker_lift::elf::EnvelopeCodec;
    let elf = &program.source.as_ref().expect("lifted from an object").elf;
    let start = elf.file_offset(SELECTED_VA, SELECTED_SIZE).expect("selected kernel lies in .text");
    start..start + SELECTED_SIZE as usize
}

/// T1: `emit(lift(x)) == x` for the 42,208-byte ELF and the 46,304-byte bundle, and both
/// inputs lift to the same six typed kernels.
#[test]
fn module_unedited_emits_identical_bytes() {
    let co = kt48_co();
    let hsaco = kt48_hsaco();
    let from_co = lift(&co, Frontend::Hipcc);
    let from_bundle = lift(&hsaco, Frontend::Hipcc);
    assert_eq!(emit::module(&from_co.program).unwrap(), co);
    assert_eq!(emit::module(&from_bundle.program).unwrap(), hsaco);
    assert_eq!((co.len(), hsaco.len()), (42_208, 46_304));
    assert_eq!(from_co.program.kernels.len(), 6);
    assert!(from_co.program.source.as_ref().unwrap().bundle.is_none());
    assert!(from_bundle.program.source.as_ref().unwrap().bundle.is_some());
    assert_eq!(from_bundle.program.kernels, from_co.program.kernels, "the bundle's device ELF is hipcc.co");
    assert_eq!(hex(&from_co.proof.input_sha256), "f876e2ce520ba8d97358b34c903a48dafc5cff2e4cc552d7ffc775361cbcb6bc");
    let stream = from_co.proof.streams.iter().find(|s| s.entry == SELECTED_VA).unwrap();
    assert_eq!(stream.size, SELECTED_SIZE);
    assert_eq!(stream.stream_sha256, <[u8; 32]>::from(Sha256::digest(&co[selected_file_range(&from_co.program)])));
    assert_eq!(from_co.program.target.abi_version, 6, "EI_ABIVERSION 4 = code object v6");
    let kernel = selected(&from_co.program);
    assert_eq!(kernel.body.layout.len(), 1696);
    assert!(matches!(kernel.origin, KernelOrigin::Frontend { kind: Frontend::Hipcc, entry_va: SELECTED_VA, size: SELECTED_SIZE, .. }));
}

/// T2: the stream is computed from typed fields (provenance words masked), a typed
/// `op_sel` flip changes exactly that instruction's words, and a don't-care set to its other
/// benign value changes bytes but not text.
#[test]
fn selected_kernel_stream_is_reencoded_not_copied() {
    let co = kt48_co();
    let lifted = lift(&co, Frontend::Hipcc);
    let range = selected_file_range(&lifted.program);
    let original = &co[range.clone()];

    let mut masked = lifted.program.clone();
    for kernel in &mut masked.kernels {
        let ids = kernel.body.layout.clone();
        for id in ids {
            kernel.body.insts.get_mut(id).unwrap().prov.bytes = None;
        }
    }
    assert_eq!(emit::bytes(selected(&masked), ARCH).unwrap(), original);
    assert_eq!(emit::module(&masked).unwrap(), co, "module identity without provenance words");

    // op_sel on one v_mov_b16_e64: the typed flip is the op_sel bit and the half it selects.
    let kernel = selected(&masked);
    let (index, id) = kernel.body.layout.iter().enumerate()
        .find(|(_, id)| name(kernel.body.insts.get(**id).unwrap()) == "v_mov_b16_e64")
        .map(|(i, id)| (i, *id)).expect("KT48 has v_mov_b16_e64");
    let start: usize = layout(kernel)[..index].iter().map(|inst| 4 * width(inst)).sum();
    let end = start + 4 * width(layout(kernel)[index]);
    let mut flipped = masked.clone();
    let inst = selected_mut(&mut flipped).body.insts.get_mut(id).unwrap();
    assert_eq!(inst.mods.op_sel & 1, 1, "{}", inst.text(ARCH).unwrap());
    inst.mods.op_sel ^= 1;
    let Some(Operand::Half(_, half)) = inst.operands.get_mut(1) else { panic!("v_mov_b16_e64 source is a half") };
    *half = Half::Lo;
    let changed = emit::bytes(selected(&flipped), ARCH).unwrap();
    let diffs: Vec<usize> = (0..changed.len()).filter(|&i| changed[i] != original[i]).collect();
    assert!(!diffs.is_empty());
    assert!(diffs.iter().all(|i| (start..end).contains(i)), "op_sel flip changed bytes outside [{start}, {end}): {diffs:?}");
    let module = emit::module(&flipped).unwrap();
    assert!((0..co.len()).filter(|&i| module[i] != co[i]).all(|i| (range.start + start..range.start + end).contains(&i)));

    // A VOP3b unused src2 set to its other benign value: bytes change, text does not.
    let kernel = selected(&masked);
    let (index, id) = kernel.body.layout.iter().enumerate()
        .find(|(_, id)| matches!(kernel.body.insts.get(**id).unwrap().fields, FormFields::Vop3b { .. }))
        .map(|(i, id)| (i, *id)).expect("KT48 has VOP3b instructions");
    let before = kernel.body.insts.get(id).unwrap().clone();
    let FormFields::Vop3b { src2_unused } = before.fields else { unreachable!() };
    let mut filled = masked.clone();
    let inst = selected_mut(&mut filled).body.insts.get_mut(id).unwrap();
    inst.fields = FormFields::Vop3b { src2_unused: if src2_unused == 0x80 { 0x00 } else { 0x80 } };
    assert_eq!(inst.text(ARCH).unwrap(), before.text(ARCH).unwrap());
    assert_eq!(peacemaker_lift::text::canonical(inst, ARCH).unwrap(), peacemaker_lift::text::canonical(&before, ARCH).unwrap());
    let refilled = emit::words(selected(&filled), ARCH).unwrap();
    let words = emit::words(selected(&masked), ARCH).unwrap();
    let at: usize = layout(selected(&masked))[..index].iter().map(|inst| width(inst)).sum();
    let differing: Vec<usize> = (0..words.len()).filter(|&i| words[i] != refilled[i]).collect();
    assert_eq!(differing, vec![at + 1], "only the VOP3b second dword changes");
    assert_eq!((words[at + 1] ^ refilled[at + 1]) & !(0x1ff << 18), 0, "only the src2 field changes");
}

/// F2 builder bundle (12 kernels) and the iu4 token/slab bundles (5 kernels
/// each) round trip through `lift_object`/`emit` byte for byte.
#[test]
fn builder_bundles_round_trip() {
    for (relative, sha, kernels, bytes) in [
        ("../../kernels/gemm_mq4g256v2_wmma_fp8_gfx12_b1.hxaco", "6a2dbe591a58680abbeef680be82a986dc0aee0beb2f46ef7cc74f4b3bfb9111", 12, 362_048),
        ("../../kernels/gemm_mq4g256v2_residual_mmq_iu4_gfx12_b1.hxaco", "8edd7565f6a442b67224cad950b3532a07386c9543b7c04c65e024a8dd4df467", 5, 83_192),
        ("../../kernels/gemm_mq4g256v2_residual_mmq_iu4_gfx12_b1s.hxaco", "cc0288f8162aeb1329f0e1a2d9d0fbd88650b501e5be04e77c1205b8f7a1bab3", 5, 83_472),
    ] {
        let input = load(relative, sha);
        assert_eq!(input.len(), bytes);
        let lifted = lift(&input, Frontend::Builder);
        assert_eq!(lifted.program.kernels.len(), kernels, "{relative}");
        assert!(lifted.program.source.as_ref().unwrap().bundle.is_some());
        assert_eq!(emit::module(&lifted.program).unwrap(), input, "{relative}");
        for kernel in &lifted.program.kernels {
            let KernelOrigin::Frontend { size, .. } = kernel.origin else { panic!("lifted origin") };
            assert_eq!(emit::bytes(kernel, ARCH).unwrap().len() as u64, size, "{}", kernel.symbol.0);
            assert_eq!(kernel.wave, Wave::Wave32);
        }
    }
}

/// T4 (erratum: true basic blocks): 106 blocks; 67 branches naming 57 distinct in-bounds
/// `Label` targets; entry + targets = 58 leaders = hipcc's 57 `.LBB5_*` labels plus the
/// kernel symbol; one `s_endpgm` ending the last block and no `Unreachable` block; every
/// `s_clause` window (14) and `s_delay_alu` reach (150) inside one block.
#[test]
fn cfg_matches_hipcc_labels() {
    let lifted = lift(&kt48_co(), Frontend::Hipcc);
    let kernel = selected(&lifted.program);
    let body = &kernel.body;
    assert_eq!(body.blocks.len(), 106);
    let mut branches = 0;
    let mut targets = BTreeSet::new();
    for inst in layout(kernel) {
        for operand in &inst.operands {
            if let Operand::Label(BlockId(target)) = operand {
                branches += 1;
                assert!(*target < body.blocks.len());
                targets.insert(*target);
            }
        }
    }
    assert_eq!(branches, 67);
    assert_eq!(targets.len(), 57);
    targets.insert(0);
    assert_eq!(targets.len(), 58, "entry + targets = hipcc's labels (entry is not a target)");
    let endpgm: Vec<usize> = layout(kernel).iter().enumerate().filter(|(_, i)| name(i) == "s_endpgm").map(|(i, _)| i).collect();
    assert_eq!(endpgm, vec![body.layout.len() - 1], "one s_endpgm, last in layout");
    assert_eq!(body.blocks.last().unwrap().term, Terminator::EndPgm);
    assert!(body.blocks.iter().all(|b| b.term != Terminator::Unreachable), "no padding inside the symbol");
    let windows = peacemaker_ir::passes::windows::check_windows(body).expect("windows stay inside blocks");
    assert_eq!(windows.clauses.len(), 14);
    assert_eq!(windows.delays.len(), 150);
    let block_of = |id| peacemaker_ir::passes::cfg::containing_block(body, id).unwrap();
    for clause in &windows.clauses {
        assert!(clause.members.iter().all(|&m| block_of(m) == block_of(clause.opener)));
    }
    for delay in &windows.delays {
        assert_eq!(block_of(delay.hint), delay.block);
        for consumer in delay.consumers().into_iter().flatten().flatten() {
            assert_eq!(block_of(consumer), delay.block);
        }
    }
}

/// `s_delay_alu` reading (M1 C6 finding, resolved against RDNA4 §16.5 p270 and LLVM's
/// `AMDGPUInsertDelayAlu`): each INSTID counts back from the instruction it applies to,
/// so INSTID1 counts from the consumer INSTSKIP selects, and `VALU_DEP_n` counts
/// non-TRANS VALUs (`TRANS32_DEP_n` TRANS ones). LLVM emits a hint only for a real
/// dependency, so on all six KT48 kernels every resolved producer writes a register its
/// consumer reads. Each rejected reading (INSTID1 counted from the hint; TRANS counted
/// as VALU) names a non-producer for some of hipcc's hints. Two hints select a
/// `ds_load_b128` as INSTID1's consumer: hipcc counted an empty inline-asm statement
/// (`;;#ASMSTART`/`;;#ASMEND` in the `.s`, no bytes) into INSTSKIP, so in the machine
/// code their second dependency lands on a non-ALU instruction (performance-only).
#[test]
fn delay_hints_name_the_producers_their_consumers_read() {
    use peacemaker_ir::effects::ImplicitSet;
    use peacemaker_ir::passes::windows::{counts_for_valu_dep, is_trans};
    let lifted = lift(&kt48_co(), Frontend::Hipcc);
    let reads = |consumer: &Inst, producer: &Inst| {
        producer.effects.defs.iter().any(|d| consumer.effects.uses.iter().any(|u| u.overlaps(*d)))
            || producer.effects.implicit.writes & consumer.effects.implicit.reads & ImplicitSet::VCC != 0
    };
    let (mut checked, mut instid1, mut from_hint_misses, mut trans_as_valu_misses) = (0, 0, 0, 0);
    let mut non_alu_consumers = Vec::new();
    for kernel in &lifted.program.kernels {
        let body = &kernel.body;
        let inst = |id| body.insts.get(id).expect("live");
        let pos = |id| body.layout.iter().position(|x| *x == id).expect("laid out");
        // The n-th instruction before layout position `at` that `counts` (straight line).
        let nth_before = |at: usize, n: u8, counts: &dyn Fn(&Inst) -> bool| {
            body.layout[..at].iter().rev().copied().filter(|&id| counts(inst(id))).nth(usize::from(n) - 1)
        };
        let windows = peacemaker_ir::passes::windows::check_windows(body).unwrap();
        for fact in &windows.delays {
            let (Some(producers), Some(consumers)) = (fact.producers(), fact.consumers()) else { continue };
            let hint = inst(fact.hint).mods.delay.expect("hint");
            for (slot, value) in [hint.instid0, hint.instid1].into_iter().enumerate() {
                if !(1..=7).contains(&value) { continue; }
                let (producer, consumer) = (producers[slot].expect("resolved"), consumers[slot].expect("resolved"));
                if !matches!(inst(consumer).form, Form::Vop1 | Form::Vop2 | Form::Vop3 | Form::Vop3p | Form::Vopd | Form::Vopc | Form::Sop1 | Form::Sop2 | Form::Sopc) {
                    non_alu_consumers.push((kernel.symbol.0.clone(), inst(fact.hint).prov.pc, slot, name(inst(consumer))));
                    continue;
                }
                assert!(reads(inst(consumer), inst(producer)), "{}: hint {:?} slot {slot} value {value}: {} does not read {}",
                    kernel.symbol.0, fact.hint, name(inst(consumer)), name(inst(producer)));
                assert_eq!(is_trans(inst(producer)), value >= 5, "TRANS32_DEP names a TRANS, VALU_DEP a non-TRANS VALU");
                checked += 1;
                if value > 4 { continue; }
                let miss = |p: Option<peacemaker_ir::cfg::InstId>| p.is_none_or(|p| !reads(inst(consumer), inst(p)));
                let any_valu = |i: &Inst| counts_for_valu_dep(i) || is_trans(i);
                if miss(nth_before(pos(consumer), value, &any_valu)) { trans_as_valu_misses += 1; }
                if slot == 1 {
                    instid1 += 1;
                    if miss(nth_before(pos(fact.hint), value, &counts_for_valu_dep)) { from_hint_misses += 1; }
                }
            }
        }
    }
    eprintln!("delay slots checked: {checked} ({instid1} INSTID1 VALU_DEP); misses: INSTID1 from the hint {from_hint_misses}, TRANS counted as VALU {trans_as_valu_misses}");
    assert!(instid1 > 0 && from_hint_misses > 0 && trans_as_valu_misses > 0, "the corpus separates both rejected readings");
    assert_eq!(non_alu_consumers, [(SELECTED.to_owned(), Some(0x95c8), 1, "ds_load_b128"), (SELECTED.to_owned(), Some(0x9604), 1, "ds_load_b128")],
        "only the two inline-asm-miscounted INSTSKIPs");
}

/// T5: descriptor and metadata values of core.md §0 (errata: 25 args, 11 explicit + 14
/// hidden); re-serialised descriptor and metadata bytes equal the originals.
#[test]
fn descriptor_and_metadata_fields() {
    use peacemaker_lift::kd::DescriptorCodec;
    let co = kt48_co();
    let lifted = lift(&co, Frontend::Hipcc);
    let kernel = selected(&lifted.program);
    let Abi::Hsa { descriptor, metadata } = &kernel.abi else { panic!("hipcc kernels are HSA") };
    assert_eq!(descriptor.group_segment_fixed_size, 0);
    assert_eq!(descriptor.private_segment_fixed_size, 0);
    assert_eq!(descriptor.kernarg_size, 328);
    assert_eq!(descriptor.kernel_code_entry_byte_offset, 23_744);
    assert_eq!(descriptor.compute_pgm_rsrc1.0, 0xe00f_001d);
    assert_eq!(descriptor.compute_pgm_rsrc2.0, 0x384);
    assert_eq!(descriptor.compute_pgm_rsrc3.0, 0x530);
    assert_eq!(descriptor.kernel_code_properties.0, 0x408);
    assert_eq!(kernel.wave, Wave::Wave32);
    let meta = &metadata.parsed;
    assert_eq!((meta.vgpr_count, meta.sgpr_count, meta.wavefront_size), (238, 30, 32));
    assert_eq!((meta.kernarg_segment_size, meta.max_flat_workgroup_size), (328, 768));
    assert!(meta.workgroup_processor_mode);
    assert_eq!(meta.symbol, format!("{SELECTED}.kd"));
    assert_eq!(meta.args.len(), 25);
    assert_eq!(meta.args.iter().filter(|a| a.value_kind.starts_with("hidden_")).count(), 14);
    // Descriptor re-serialises to the 64 bytes at its `.kd` VA (file offset == VA in .rodata).
    use peacemaker_lift::elf::EnvelopeCodec;
    let elf = &lifted.program.source.as_ref().unwrap().elf;
    let slot = elf.kernels.iter().find(|s| s.name == SELECTED).unwrap();
    assert_eq!(slot.kd_va, 0x2240);
    let kd_at = elf.file_offset(slot.kd_va, 64).unwrap();
    assert_eq!(descriptor.to_bytes().as_slice(), &co[kd_at..kd_at + 64]);
    // Metadata re-serialises from typed fields to the verbatim note slice.
    let reserialised = peacemaker_lift::metadata::serialize_kernel(metadata).unwrap();
    assert_eq!(reserialised, metadata.raw_msgpack);
    assert_eq!(co.windows(reserialised.len()).filter(|w| *w == reserialised.as_slice()).count(), 1, "one slice of the note");
}

fn rejection(result: Result<Lifted<Program>, LiftError>) -> (Option<String>, u64, Rule, String) {
    match result {
        Ok(_) => panic!("a program was produced"),
        Err(LiftError::Rejected { kernel, offset, rule, reason }) => (kernel, offset, rule, reason),
    }
}

/// Pinned second-opinion toolchain: `PEACEMAKER_ROCM` must hold the tools (fail, never
/// skip); `PEACEMAKER_NO_ROCM` forces the committed-fixture path; otherwise the default
/// install is used when present.
struct Tools {
    objdump: PathBuf,
    mc: PathBuf,
}

fn toolchain() -> Option<Tools> {
    let at = |root: &Path| Tools { objdump: root.join("lib/llvm/bin/llvm-objdump"), mc: root.join("lib/llvm/bin/llvm-mc") };
    if let Some(root) = std::env::var_os("PEACEMAKER_ROCM") {
        let tools = at(Path::new(&root));
        assert!(tools.objdump.is_file() && tools.mc.is_file(), "PEACEMAKER_ROCM={root:?} holds no llvm-objdump/llvm-mc");
        return Some(tools);
    }
    if std::env::var_os("PEACEMAKER_NO_ROCM").is_some() {
        return None;
    }
    let tools = at(Path::new("/opt/rocm/core-10.0"));
    (tools.objdump.is_file() && tools.mc.is_file()).then_some(tools)
}

#[derive(Debug, PartialEq)]
struct ObjdumpLine {
    addr: u64,
    words: Vec<u32>,
    text: String,
    /// `<sym+0x…>` annotation, as an offset from the symbol.
    target: Option<u64>,
}

/// `llvm-objdump -d` instruction lines: `\t<text> // <addr>: <WORDS…> [<sym+0x…>]`.
fn parse_objdump(stdout: &str) -> Vec<ObjdumpLine> {
    stdout.lines().filter_map(|line| {
        let (text, comment) = line.strip_prefix('\t')?.split_once("//")?;
        let text = text.trim();
        if text.is_empty() {
            return None;
        }
        let mut tokens = comment.split_whitespace();
        let addr = u64::from_str_radix(tokens.next()?.strip_suffix(':')?, 16).ok()?;
        let mut words = Vec::new();
        let mut target = None;
        for token in tokens {
            if let Some(annotation) = token.strip_prefix('<') {
                let offset = annotation.trim_end_matches('>').rsplit_once("+0x").map(|(_, hex)| hex);
                target = Some(offset.map_or(0, |hex| u64::from_str_radix(hex, 16).expect("hex annotation")));
                break;
            }
            words.push(u32::from_str_radix(token, 16).expect("encoding word"));
        }
        Some(ObjdumpLine { addr, words, text: text.to_owned(), target })
    }).collect()
}

/// The selected symbol's objdump lines: live from the pinned tool (which must equal the
/// committed fixture), else the committed fixture.
fn objdump_selected() -> Vec<ObjdumpLine> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/kt48");
    let pinned = parse_objdump(&std::fs::read_to_string(root.join("hipcc.objdump.txt")).expect("objdump fixture"));
    let lines = match toolchain() {
        Some(tools) => {
            let output = std::process::Command::new(&tools.objdump)
                .args(["--mcpu=gfx1201", "-d", &format!("--disassemble-symbols={SELECTED}")])
                .arg(root.join("hipcc.co"))
                .output()
                .expect("run pinned llvm-objdump");
            assert!(output.status.success(), "llvm-objdump: {}", String::from_utf8_lossy(&output.stderr));
            let live = parse_objdump(&String::from_utf8_lossy(&output.stdout));
            assert!(live == pinned, "committed objdump fixture is stale");
            live
        }
        None => pinned,
    };
    lines.into_iter().filter(|l| (SELECTED_VA..SELECTED_VA + SELECTED_SIZE).contains(&l.addr)).collect()
}

/// T3: boundaries, words and canonical text of all 1,696 instructions equal the pinned
/// objdump's; every branch's `Label` block starts at objdump's annotated target (67
/// branches, 57 distinct targets).
#[test]
fn decoder_agrees_with_pinned_objdump() {
    let lifted = lift(&kt48_co(), Frontend::Hipcc);
    let kernel = selected(&lifted.program);
    let body = &kernel.body;
    let lines = objdump_selected();
    assert_eq!(lines.len(), 1696);
    let insts = emit::insts(kernel, ARCH).unwrap();
    let texts = emit::text(kernel, ARCH).unwrap();
    let pc = |index: usize| u64::from(body.insts.get(body.layout[index]).unwrap().prov.pc.expect("lifted pc"));
    let mut mismatches = Vec::new();
    let mut branches = 0;
    let mut targets = BTreeSet::new();
    for (index, line) in lines.iter().enumerate() {
        assert_eq!(line.addr, pc(index), "boundary of instruction {index}");
        assert_eq!(line.words, gfx12::encode(&insts[index]).unwrap().to_vec(), "words at {:#x}", line.addr);
        if texts[index] != line.text {
            mismatches.push(format!("{:#x}: ours {:?}, objdump {:?}", line.addr, texts[index], line.text));
        }
        let label = layout(kernel)[index].operands.iter().find_map(|op| match op { Operand::Label(b) => Some(*b), _ => None });
        if let Some(block) = label {
            branches += 1;
            let target = pc(body.blocks[block.0].range.0) - SELECTED_VA;
            assert_eq!(line.target, Some(target), "branch target at {:#x}", line.addr);
            targets.insert(target);
        }
    }
    assert!(mismatches.is_empty(), "{} text mismatches:\n{}", mismatches.len(), mismatches[..mismatches.len().min(10)].join("\n"));
    assert_eq!((branches, targets.len()), (67, 57));
}

fn position(kernel: &Kernel, id: peacemaker_ir::cfg::InstId) -> usize {
    kernel.body.layout.iter().position(|x| *x == id).expect("laid out")
}

/// T6: wait census; 124 obligations (erratum), all DS-store source timing on a
/// `ds_store_2addr_b64` source; KMcnt units [2,1,1,1,2,2] in program order, each SMEM
/// load retired by an `s_wait_kmcnt 0x0`; an input whose first `s_wait_kmcnt 0x0` is
/// rewritten to `0x1` still lifts and yields a SMEM RAW hazard obligation.
#[test]
fn wait_facts_and_obligations() {
    use peacemaker_ir::effects::MemClass;
    use peacemaker_ir::passes::waits::{census, replay};
    use peacemaker_ir::state::ObligationKind;
    use peacemaker_ir::wait::Counter;
    let co = kt48_co();
    let lifted = lift(&co, Frontend::Hipcc);
    let kernel = selected(&lifted.program);
    let body = &kernel.body;
    let counts = census(body, ARCH).unwrap();
    assert_eq!(counts.kmcnt, vec![0; 7]);
    assert_eq!((counts.loadcnt, counts.dscnt, counts.loadcnt_dscnt, counts.alu), (16, 28, 3, 142));

    let facts = replay(body, ARCH).unwrap();
    assert_eq!(facts.obligations.len(), 124);
    let ds_stores: Vec<_> = facts.events.iter()
        .filter(|e| name(body.insts.get(e.inst).unwrap()) == "ds_store_2addr_b64")
        .collect();
    for obligation in &facts.obligations {
        assert_eq!(obligation.kind, ObligationKind::SrcReadTiming(MemClass::DsStore), "{}", obligation.text);
        let [consumer] = obligation.insts[..] else { panic!("one site per obligation: {:?}", obligation.insts) };
        let defs = &body.insts.get(consumer).unwrap().effects.defs;
        assert!(
            ds_stores.iter().any(|e| e.src_locks.0.iter().any(|lock| defs.iter().any(|d| d.overlaps(*lock)))),
            "{consumer:?} redefines no ds_store_2addr_b64 source"
        );
    }
    assert!(!facts.obligations.iter().any(|o| o.kind == ObligationKind::Hazard));

    let mut smem: Vec<_> = facts.events.iter().filter(|e| e.class == MemClass::SmemLoad).collect();
    smem.sort_by_key(|e| position(kernel, e.inst));
    let units: Vec<u8> = smem.iter().map(|e| e.units[Counter::Km as usize]).collect();
    assert_eq!(units, vec![2, 1, 1, 1, 2, 2]);
    for event in &smem {
        assert!(facts.facts.iter().any(|fact| fact.satisfies.contains(&event.id)
            && name(body.insts.get(fact.wait).unwrap()) == "s_wait_kmcnt"), "{:?} is never retired by kmcnt", event.inst);
    }

    let wait = layout(kernel).into_iter().find(|i| name(i) == "s_wait_kmcnt").unwrap();
    let at = selected_file_range(&lifted.program).start + (u64::from(wait.prov.pc.unwrap()) - SELECTED_VA) as usize;
    assert_eq!(co[at..at + 4], 0xbfc7_0000u32.to_le_bytes(), "s_wait_kmcnt 0x0");
    let mut mutated = co.clone();
    mutated[at..at + 4].copy_from_slice(&0xbfc7_0001u32.to_le_bytes());
    let weakened = lift(&mutated, Frontend::Hipcc);
    let body = &selected(&weakened.program).body;
    let facts = replay(body, ARCH).unwrap();
    let load = body.insts.get(smem[0].inst).unwrap();
    assert_eq!(name(load), "s_load_b128");
    let stranded: Vec<_> = facts.obligations.iter()
        .filter(|o| o.kind == ObligationKind::Hazard && o.rule_id == "wait-raw-smem-load")
        .collect();
    assert!(!stranded.is_empty(), "kmcnt 0x1 leaves the s_load_b128 destination pending at its use");
    for obligation in stranded {
        let effects = &body.insts.get(obligation.insts[0]).unwrap().effects;
        let touched = effects.uses.iter().chain(&effects.defs).any(|r| load.effects.defs.iter().any(|d| d.overlaps(*r)));
        assert!(touched, "{} names no access to the s_load_b128 destination", obligation.text);
    }
}

/// Path-sensitivity pin (C5 erratum): the first-iteration sites 956/957/973/974 are
/// obligations; 917/925 (reachable only through a later block) are not.
#[test]
fn replay_follows_feasible_paths_only() {
    let lifted = lift(&kt48_co(), Frontend::Hipcc);
    let kernel = selected(&lifted.program);
    let facts = peacemaker_ir::passes::waits::replay(&kernel.body, ARCH).unwrap();
    let sites: BTreeSet<usize> = facts.obligations.iter().map(|o| position(kernel, o.insts[0])).collect();
    for site in [956, 957, 973, 974] {
        assert!(sites.contains(&site), "feasible first-iteration site {site}");
    }
    for site in [917, 925] {
        assert!(!sites.contains(&site), "infeasible site {site}");
    }
}

/// Replace the selected kernel's leading instructions with `synth`, padded with `s_nop 0`
/// to a whole number of original instructions, so the rest of the stream stays aligned.
fn with_prefix(co: &[u8], program: &Program, synth: &[u32]) -> Vec<u8> {
    const S_NOP: u32 = 0xbf80_0000;
    let mut covered = 0;
    for inst in layout(selected(program)) {
        if covered >= synth.len() {
            break;
        }
        covered += width(inst);
    }
    let mut words = synth.to_vec();
    words.resize(covered, S_NOP);
    let start = selected_file_range(program).start;
    let mut out = co.to_vec();
    for (i, word) in words.iter().enumerate() {
        out[start + 4 * i..start + 4 * i + 4].copy_from_slice(&word.to_le_bytes());
    }
    out
}

/// T7: synthetic streams are rejections naming the kernel, the offending VA and the
/// rule, never programs. Words are pinned `llvm-mc -mcpu=gfx1201` encodings
/// (`s_waitcnt` from `-mcpu=gfx1100`; 0xbfff0000 is invalid for the pinned disassembler).
#[test]
fn rejects_what_it_cannot_model() {
    let co = kt48_co();
    let lifted = lift(&co, Frontend::Hipcc);
    let kernel = selected(&lifted.program);
    let insts = layout(kernel);
    let pcs: Vec<usize> = insts.iter().scan(0, |pc, inst| { let at = *pc; *pc += width(inst); Some(at) }).collect();
    let wide = (1..insts.len()).find(|&i| pcs[i] >= 2 && width(insts[i]) >= 2).unwrap();
    // s_branch at pc 0 lands at 1 + simm16: one dword into instruction `wide`.
    let mid_instruction = pcs[wide] as u32;
    let cases = [
        ("undefined opcode word", vec![0xbfff_0000], Rule::Decode),
        ("branch to mid-instruction", vec![0xbfa0_0000 | mid_instruction], Rule::BranchTarget),
        ("branch past the symbol end", vec![0xbfa0_7fff], Rule::BranchTarget),
        // s_clause 0x1; s_load_b32 s4, s[0:1], 0x0; s_load_b32 s5, s[0:1], 0x4; s_cbranch_scc1 -3
        ("clause window crossing a target", vec![0xbf85_0001, 0xf400_0100, 0xf800_0000, 0xf400_0140, 0xf800_0004, 0xbfa2_fffd], Rule::Window),
        ("s_waitcnt vmcnt(0) lgkmcnt(0) on gfx12", vec![0xbf89_0007], Rule::Decode),
    ];
    for (what, synth, rule) in cases {
        let input = with_prefix(&co, &lifted.program, &synth);
        let (kernel, offset, got, reason) = rejection(lift_object(&input, Options { frontend: Frontend::Hipcc }));
        assert_eq!((kernel.as_deref(), offset, got), (Some(SELECTED), SELECTED_VA, rule), "{what}: {reason}");
    }
}

fn example(row: &peacemaker_ir::isa::OpRow) -> Vec<u32> {
    row.encoding.split_whitespace().map(|w| u32::from_str_radix(w, 16).expect("table dword")).collect()
}

/// Word and shift of a table don't-care field in the codec's form layouts: whole-word
/// extras carry absolute masks, unused source selectors and `omod` field-relative values.
fn placement(name: &str) -> (usize, u32) {
    match name {
        "w0_extra" | "wait_unused" => (0, 0),
        "w1_extra" | "vsrc_unused" => (1, 0),
        "src1_unused" => (1, 9),
        "src2_unused" => (1, 18),
        "omod" => (1, 27),
        other => panic!("table don't-care field {other} has no known placement"),
    }
}

/// Every combination a row declares on its example words: each ignored field over its
/// benign values, and for literal forms a second literal value.
fn declared_variants(row: &peacemaker_ir::isa::OpRow, words: &[u32], literal: bool) -> Vec<Vec<u32>> {
    let mut variants = vec![words.to_vec()];
    for rule in row.fields.iter().filter(|r| r.class == peacemaker_ir::isa::FieldClass::Ignored) {
        let (index, shift) = placement(rule.name);
        let mask = rule.mask << shift;
        variants = variants.into_iter().flat_map(|base| rule.allowed.iter().map(move |&value| {
            let mut variant = base.clone();
            variant[index] = (variant[index] & !mask) | (value << shift);
            variant
        })).collect();
    }
    if literal {
        let other: Vec<Vec<u32>> = variants.iter().map(|v| {
            let mut v = v.clone();
            *v.last_mut().expect("literal word") ^= 0x5a5a_5a5a;
            v
        }).collect();
        variants.extend(other);
    }
    variants
}

/// T8: for every table row and every don't-care/literal combination it declares, the
/// words decode to a typed instruction `i` with `encode(i) ==` the words and
/// `decode(encode(i)) == i`.
#[test]
fn codec_is_total_over_the_table() {
    let mut combinations = 0;
    for row in peacemaker_ir::isa::gfx12() {
        let words = example(row);
        let (inst, _) = gfx12::decode(&words).unwrap_or_else(|e| panic!("{}: {e}", row.name));
        for variant in declared_variants(row, &words, inst.literal.is_some()) {
            let (inst, n) = gfx12::decode(&variant).unwrap_or_else(|e| panic!("{} {variant:08x?}: {e}", row.name));
            assert_eq!(n, variant.len(), "{}", row.name);
            let encoded = gfx12::encode(&inst).unwrap_or_else(|e| panic!("{} {variant:08x?}: {e}", row.name));
            assert_eq!(encoded.as_slice(), variant.as_slice(), "{}", row.name);
            assert_eq!(gfx12::decode(&encoded).unwrap().0, inst, "{}", row.name);
            combinations += 1;
        }
    }
    assert!(combinations > peacemaker_ir::isa::gfx12().len(), "don't-care combinations were exercised");
}

/// A VMEM VADDR byte the instruction does not use (scratch with SVE clear,
/// `global_*_addtid`) has no typed home, so only its zero encoding decodes.
#[test]
fn unused_vmem_vaddr_byte_rejects() {
    for (canonical, extra) in [([0xED05_C07C, 0, 0x0001_1000], 0x01), ([0xEE0A_401C, 0x7790_0000, 0xFFFF_F800], 0x80)] {
        let (inst, _) = gfx12::decode(&canonical).unwrap();
        assert_eq!(gfx12::encode(&inst).unwrap().as_slice(), canonical.as_slice());
        let mut dirty = canonical;
        dirty[2] |= extra;
        assert!(gfx12::decode(&dirty).is_err(), "{dirty:08x?} would re-encode without its VADDR byte");
    }
}

proptest::proptest! {
    #![proptest_config(proptest::test_runner::Config::with_cases(1000))]
    /// T8, fuzz property (the `fuzz/decode` target's body, uninstrumented): arbitrary words
    /// and bit-flipped table examples either reject or decode to an instruction whose
    /// encoding is exactly the consumed words.
    #[test]
    fn decode_is_exact_or_rejects(
        row in 0..peacemaker_ir::isa::gfx12().len(),
        random in proptest::prelude::any::<[u32; 3]>(),
        flips in proptest::collection::vec(0usize..96, 0..4),
        raw in proptest::prelude::any::<bool>(),
    ) {
        let mut words = if raw { random.to_vec() } else { example(&peacemaker_ir::isa::gfx12()[row]) };
        words.resize(3, random[2]);
        if !raw {
            for bit in flips {
                words[bit / 32] ^= 1 << (bit % 32);
            }
        }
        if let Ok((inst, n)) = gfx12::decode(&words) {
            proptest::prop_assert!(n > 0 && n <= words.len());
            let encoded = gfx12::encode(&inst).unwrap();
            proptest::prop_assert_eq!(encoded.as_slice(), &words[..n]);
        }
    }
}

/// Assemble canonical lines with the pinned `llvm-mc`; one encoding per line.
fn mc(tools: &Tools, lines: &[String]) -> Vec<Vec<u32>> {
    let path = std::env::temp_dir().join(format!("pm-c7-mc-{}-{}.s", std::process::id(), lines.len()));
    std::fs::write(&path, lines.join("\n") + "\n").unwrap();
    let output = std::process::Command::new(&tools.mc)
        .args(["-arch=amdgcn", "-mcpu=gfx1201", "-show-encoding"])
        .arg(&path)
        .output()
        .expect("run pinned llvm-mc");
    std::fs::remove_file(&path).unwrap();
    assert!(output.status.success(), "llvm-mc: {}", String::from_utf8_lossy(&output.stderr));
    let encodings: Vec<Vec<u32>> = String::from_utf8_lossy(&output.stdout).lines().filter_map(|line| {
        let bytes = line.split_once("; encoding: [")?.1.trim_end_matches(']');
        let bytes: Vec<u8> = bytes.split(',').map(|b| u8::from_str_radix(b.trim().trim_start_matches("0x"), 16).unwrap()).collect();
        Some(bytes.chunks_exact(4).map(|w| u32::from_le_bytes(w.try_into().unwrap())).collect())
    }).collect();
    assert_eq!(encodings.len(), lines.len());
    encodings
}

/// T9: `mc(emit::text(k)) == emit::bytes(k)` for all 1,696 KT48 instructions (the
/// required set is the whole stream, the reported set is empty); a VOP3b with its unused
/// `src2` at the other benign value prints identically, and `llvm-mc`'s bytes differ from
/// its encoding only in that field with identical semantics: reported, not failed.
#[test]
fn assembler_parity_where_syntax_is_lossless() {
    let lifted = lift(&kt48_co(), Frontend::Hipcc);
    let kernel = selected(&lifted.program);
    let lines = emit::text(kernel, ARCH).unwrap();
    let insts = emit::insts(kernel, ARCH).unwrap();
    let index = insts.iter().position(|i| matches!(i.fields, FormFields::Vop3b { .. })).expect("KT48 has VOP3b");
    let FormFields::Vop3b { src2_unused } = insts[index].fields else { unreachable!() };
    let synth = Inst { fields: FormFields::Vop3b { src2_unused: if src2_unused == 0x80 { 0x00 } else { 0x80 } }, ..insts[index].clone() };
    let synth_text = peacemaker_lift::text::canonical(&synth, ARCH).unwrap();
    assert_eq!(synth_text, lines[index], "the unused src2 is invisible in canonical text");
    let synth_words = gfx12::encode(&synth).unwrap();
    let Some(tools) = toolchain() else {
        eprintln!("T9: no pinned toolchain; assembler parity not run");
        return;
    };
    let assembled = mc(&tools, &lines);
    let mismatched: Vec<String> = insts.iter().zip(&assembled).zip(&lines)
        .filter(|((inst, words), _)| gfx12::encode(inst).unwrap().as_slice() != words.as_slice())
        .map(|(_, line)| line.clone())
        .collect();
    assert!(mismatched.is_empty(), "required set not byte-equal (reported set must be empty for KT48): {mismatched:?}");
    assert_eq!(assembled.concat(), emit::words(kernel, ARCH).unwrap());

    let reassembled = &mc(&tools, std::slice::from_ref(&synth_text))[0];
    assert_ne!(reassembled.as_slice(), synth_words.as_slice());
    assert_eq!(reassembled.len(), synth_words.len());
    assert_eq!(reassembled[0], synth_words[0]);
    assert_eq!((reassembled[1] ^ synth_words[1]) & !(0x1ff << 18), 0, "only the unused src2 field differs");
    let (back, _) = gfx12::decode(reassembled).unwrap();
    assert_eq!((back.op, back.form, &back.operands, &back.mods, back.literal), (synth.op, synth.form, &synth.operands, &synth.mods, synth.literal));
    assert_ne!(back.fields, synth.fields, "reported: the don't-care differs, the semantics do not");
}

/// T10: the profiler entry script — kernarg extension 328 → 352 with three appended args,
/// the `.__pm_profile` rename, and an entry `Insert` that loads the extension and copies
/// the record pointer into claimed `s[28:29]` — as one transaction on the lifted program,
/// then its inverse: stream and module identity hold again. The inserted instructions are
/// decoded from pinned `llvm-mc -mcpu=gfx1201` words, so they carry the codec's own fields.
#[test]
fn edit_inverse_restores_bytes() {
    use peacemaker_ir::edit::{analyze, Cursor, DescriptorChange, Edit, MetaChange};
    use peacemaker_ir::inst::SymbolId;
    use peacemaker_ir::metadata::Kernarg;
    use peacemaker_ir::reg::{ClaimOwner, Kind, RegClaim, RegRef, Scope};
    let co = kt48_co();
    let lifted = lift(&co, Frontend::Hipcc);
    let symbol = SymbolId(SELECTED.into());
    let base = analyze(lifted.program, &symbol).unwrap();
    let original = emit::bytes(selected(&base.program), ARCH).unwrap();
    // s_load_b128 s[4:7], s[0:1], 0x148; s_load_b64 s[8:9], s[0:1], 0x158;
    // s_wait_kmcnt 0x0; s_mov_b32 s28, s4; s_mov_b32 s29, s5.
    let words: [&[u32]; 5] = [&[0xf400_4100, 0xf800_0148], &[0xf400_2200, 0xf800_0158], &[0xbfc7_0000], &[0xbe9c_0004], &[0xbe9d_0005]];
    let header: Vec<Inst> = words.iter().map(|w| gfx12::decode(w).unwrap().0).collect();
    let header_bytes: Vec<u8> = words.concat().iter().flat_map(|w| w.to_le_bytes()).collect();
    let arg = |name: &str, offset| Kernarg { name: name.into(), size: 8, offset, value_kind: "by_value".into(), address_space: None };
    let args = vec![arg("pm_profile_records", 328), arg("pm_profile_stride", 336), arg("pm_profile_grid", 344)];
    let claims = vec![RegClaim { name: "pm_pointer".into(), reg: RegRef { kind: Kind::S, base: 28, len: 2 }, scope: Scope::Whole, owner: ClaimOwner::Builder }];
    let profiled = format!("{SELECTED}__pm_profile");
    let entry = selected(&base.program).body.layout[0];
    let script = Edit::Batch(vec![
        Edit::Descriptor { kernel: symbol.clone(), change: DescriptorChange::KernargSize(352) },
        Edit::Metadata { kernel: symbol.clone(), change: MetaChange::AppendArgs(args.clone()) },
        Edit::Insert { at: Cursor::before(BlockId(0), entry), insts: header.clone(), claims: claims.clone() },
        Edit::Metadata { kernel: symbol.clone(), change: MetaChange::Rename(profiled.clone()) },
    ]);
    // Without `.sgpr_count` 30 -> 32 (s29 + VCC) the resource check opens an obligation.
    let (short, _) = base.edit(&symbol, script.clone()).unwrap();
    assert!(short.obligations.iter().any(|o| o.rule_id == "resource-sgpr-metadata"), "claimed s[28:29] exceed .sgpr_count 30");
    let Edit::Batch(mut steps) = script else { unreachable!() };
    steps.insert(2, Edit::Metadata { kernel: symbol.clone(), change: MetaChange::SetSgprCount(32) });
    let script = Edit::Batch(steps);
    let (edited, delta) = base.edit(&symbol, script).unwrap();
    assert_eq!(delta.kernel.0, profiled);
    let kernel = edited.program.kernels.iter().find(|k| k.symbol.0 == profiled).expect("renamed kernel");
    assert_eq!(emit::bytes(kernel, ARCH).unwrap(), [header_bytes, original.clone()].concat(), "the original stream follows the header unchanged");
    let Abi::Hsa { descriptor, metadata } = &kernel.abi else { unreachable!() };
    assert_eq!((descriptor.kernarg_size, metadata.parsed.args.len(), metadata.parsed.sgpr_count), (352, 28, 32));
    assert_eq!(&metadata.parsed.args[25..], args.as_slice());
    let count = |a: &peacemaker_ir::state::Analyzed<Program>| a.obligations.len();
    assert_eq!(count(&edited), count(&base), "the script opens no obligation");

    let (restored, _) = edited.undo(&delta).unwrap();
    assert_eq!(emit::bytes(selected(&restored.program), ARCH).unwrap(), original, "stream identity after the inverse");
    assert_eq!(emit::module(&restored.program).unwrap(), co, "module identity after the inverse");
}
