// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The gfx1151 K1-C1 one-lane slab decode GEMV module `mq_i4_moe_gemv_c1_pm_gfx1151`
//! (`mq_i4_moe_gemv_c1_{r256_gate_up,r128_down}`, both codecs per launch): symbols, the frozen 80-byte kernarg ABI
//! (`n_experts@72`, `reserved@76`), the fixed LDS, exact-arch gating, the code-end guard, the unsigned `eid >= n_experts`
//! exit (emitted order and, on the peacemaker CPU interpreter, invalid ids `-1`, `i32::MIN`, `n`, `n + 1` against valid `n - 1`
//! on both paths and both codecs), the schedule census (every weight load of a codec body
//! precedes its barrier, the `vmcnt` ladder), the per-codec instruction census of the frozen numerical tree (the
//! operations, chain assignment, fold and reduction of K1), the derived shape contract, the committed bundle and
//! contract files, and (feature `toolchain`) wait replay and M7.
use peacemaker_isa::emit::CODE_END_TAIL;
use peacemaker_isa::kernels::mq_i4_moe_gemv_c1::{self as c1, Kind, Spec};
use peacemaker_isa::{native, Arch};

const SYMBOLS: [&str; 2] = [
    "mq_i4_moe_gemv_c1_r256_gate_up",
    "mq_i4_moe_gemv_c1_r128_down",
];
const MODULE: &str = "mq_i4_moe_gemv_c1_pm_gfx1151";
/// Frozen ABI: `(offset, size, name)` in kernarg order.
const ARGS: [(u32, u32, &str); 14] = [
    (0, 8, "expert_weight_ptrs"),
    (8, 8, "topk"),
    (16, 8, "codec_table"),
    (24, 8, "X"),
    (32, 8, "Y0"),
    (40, 8, "Y1"),
    (48, 4, "M"),
    (52, 4, "K"),
    (56, 4, "uniform_codec"),
    (60, 4, "x_row_div"),
    (64, 4, "x_row_stride"),
    (68, 4, "y_slot_stride"),
    (72, 4, "n_experts"),
    (76, 4, "reserved"),
];
const S_CODE_END: [u8; 4] = 0xbf9f_0000u32.to_le_bytes();

fn field(text: &str, key: &str) -> Vec<u32> {
    text.lines()
        .filter_map(|l| l.trim().strip_prefix(key))
        .map(|v| v.trim().parse().unwrap())
        .collect()
}
fn mnemonic(line: &str) -> &str {
    line.split_whitespace().next().unwrap_or("")
}
fn count(lines: &[&str], mnemonic_name: &str) -> usize {
    lines
        .iter()
        .filter(|l| mnemonic(l) == mnemonic_name)
        .count()
}
/// Operand `n` (0-based) of an instruction line, split on `", "`.
fn operand(line: &str, n: usize) -> String {
    line.trim()
        .split_once(' ')
        .map(|(_, rest)| rest.split(", ").nth(n).unwrap_or("").trim().to_owned())
        .unwrap_or_default()
}
fn vreg(token: &str) -> u32 {
    token
        .trim_start_matches('v')
        .parse()
        .unwrap_or_else(|_| panic!("not a VGPR: {token}"))
}
fn imm_offset(line: &str) -> u32 {
    line.split("offset:")
        .nth(1)
        .map_or(0, |v| v.split_whitespace().next().unwrap().parse().unwrap())
}
fn arch_specs() -> Vec<Spec> {
    c1::module_specs()
}
fn tag(kind: Kind) -> &'static str {
    kind.tag()
}

/// The trimmed lines of the W3 body (after `.Lc1_<tag>_w3:` up to `_mq4:`) and of the MQ4 body (up to `_exit:`).
fn bodies(text: &str, kind: Kind) -> (Vec<&str>, Vec<&str>) {
    let lines: Vec<&str> = text.lines().map(str::trim).collect();
    let at = |name: &str| {
        lines
            .iter()
            .position(|l| *l == format!(".Lc1_{}_{name}:", tag(kind)))
            .unwrap_or_else(|| panic!("label {name}"))
    };
    let (w3, mq4, exit) = (at("w3"), at("mq4"), at("exit"));
    assert!(w3 < mq4 && mq4 < exit);
    (lines[w3 + 1..mq4].to_vec(), lines[mq4 + 1..exit].to_vec())
}

#[test]
fn module_exports_are_the_two_frozen_symbols_in_order() {
    let specs = c1::module_specs();
    assert_eq!(specs, [Spec::new(Kind::GateUp), Spec::new(Kind::Down)]);
    assert_eq!(specs.iter().map(Spec::symbol).collect::<Vec<_>>(), SYMBOLS);
    assert_eq!(c1::SYMBOLS, SYMBOLS);
    assert_eq!(Spec::module(), MODULE);
    let (emitted, text, proof) = c1::emit_module().unwrap();
    assert_eq!(emitted.len(), SYMBOLS.len());
    assert_eq!((proof.module.as_str(), proof.arch), (MODULE, Arch::Gfx1151));
    assert_eq!(proof.kernels.len(), SYMBOLS.len());
    assert_eq!(text.matches(".amdhsa_kernel ").count(), SYMBOLS.len());
    for symbol in SYMBOLS {
        assert!(
            text.contains(&format!(".amdhsa_kernel {symbol}\n")),
            "{symbol}: missing from module source"
        );
        assert!(
            text.contains(&format!(".globl {symbol}\n")),
            "{symbol}: not global"
        );
        assert!(
            text.contains(&format!(".name: {symbol}\n")),
            "{symbol}: missing from module metadata"
        );
    }
    assert!(text.contains("amdgcn-amd-amdhsa--gfx1151"));
}

/// Emission is bit-deterministic, and every architecture but gfx1151 is refused (no implicit alias).
#[test]
fn emission_is_deterministic_and_refuses_other_arches() {
    for spec in arch_specs() {
        let (a, b) = (c1::emit(spec).unwrap(), c1::emit(spec).unwrap());
        assert_eq!(a.s_text, b.s_text, "{spec:?}");
        assert_eq!(a.proof.s_text_sha256, b.proof.s_text_sha256, "{spec:?}");
    }
    let (_, ta, pa) = c1::emit_module().unwrap();
    let (_, tb, pb) = c1::emit_module().unwrap();
    assert_eq!((ta, pa.s_text_sha256), (tb, pb.s_text_sha256));
    for arch in [Arch::Gfx1100, Arch::Gfx1201, Arch::Gfx942] {
        for kind in Kind::ALL {
            let spec = Spec { arch, kind };
            assert!(
                spec.validate().is_err() && c1::emit(spec).is_err(),
                "{arch:?} {kind:?}"
            );
        }
    }
}

/// Wave32, eight waves per workgroup, grid y = the slot, fixed LDS of one X row (10,240 B gate, 2,560 B down),
/// no scratch/spills/private, and the register budget the schedule targets (<= 96 VGPRs: 16 waves per SIMD).
#[test]
fn launch_resources_match_the_frozen_geometry() {
    assert_eq!(
        (
            c1::KERNARG_BYTES,
            c1::WORKGROUP_SIZE,
            c1::ROWS_PER_WORKGROUP,
            c1::SLOTS
        ),
        (80, 256, 8, 10)
    );
    for (spec, lds) in [
        (Spec::new(Kind::GateUp), 10240u32),
        (Spec::new(Kind::Down), 2560),
    ] {
        let e = c1::emit(spec).unwrap();
        let symbol = spec.symbol();
        assert_eq!(spec.kind.lds_bytes(), lds);
        assert!(
            e.s_text
                .contains(&format!(".amdhsa_group_segment_fixed_size {lds}\n")),
            "{symbol}: descriptor LDS"
        );
        assert!(
            e.s_text
                .contains(&format!(".group_segment_fixed_size: {lds}\n")),
            "{symbol}: metadata LDS"
        );
        assert_eq!(
            field(&e.s_text, ".max_flat_workgroup_size:"),
            [256],
            "{symbol}: block 256"
        );
        assert!(
            e.s_text.contains(".amdhsa_wavefront_size32 1"),
            "{symbol}: wave32"
        );
        assert!(
            e.s_text.contains(".amdhsa_system_sgpr_workgroup_id_y 1"),
            "{symbol}: grid y is the slot"
        );
        assert!(
            e.s_text.contains(".private_segment_fixed_size: 0\n"),
            "{symbol}: private segment"
        );
        assert!(
            e.s_text.contains(".sgpr_spill_count: 0\n")
                && e.s_text.contains(".vgpr_spill_count: 0\n"),
            "{symbol}: spills"
        );
        let vgprs = field(&e.s_text, ".amdhsa_next_free_vgpr");
        assert_eq!(vgprs.len(), 1, "{symbol}");
        assert!(vgprs[0] <= 96, "{symbol}: {} VGPRs", vgprs[0]);
        assert_eq!(u32::from(e.shape.next_free_vgpr), vgprs[0], "{symbol}");
        assert!(
            e.s_text.contains(&format!(".vgpr_count: {}\n", vgprs[0])),
            "{symbol}: metadata VGPR count"
        );
        assert!(
            !e.s_text
                .lines()
                .any(|l| mnemonic(l).starts_with("scratch_") || mnemonic(l).starts_with("flat_")),
            "{symbol}: scratch/flat"
        );
        assert!(e.shape.barriers >= 1, "{symbol}: the X staging barrier");
    }
}

/// The 80-byte kernarg block: per-argument offsets/sizes/names, access, the launch bound, the first loads.
#[test]
fn kernarg_abi_matches_the_frozen_80_byte_layout() {
    for spec in arch_specs() {
        let e = c1::emit(spec).unwrap();
        let symbol = spec.symbol();
        assert!(
            e.s_text.contains(".kernarg_segment_size: 80\n"),
            "{symbol}: kernarg size"
        );
        assert!(
            e.s_text.contains(".amdhsa_kernarg_size 80\n"),
            "{symbol}: descriptor kernarg size"
        );
        assert_eq!(
            field(&e.s_text, ".offset:"),
            ARGS.iter().map(|a| a.0).collect::<Vec<_>>(),
            "{symbol}: kernarg offsets"
        );
        assert_eq!(
            field(&e.s_text, ".size:"),
            ARGS.iter().map(|a| a.1).collect::<Vec<_>>(),
            "{symbol}: kernarg sizes"
        );
        let names: Vec<String> = e
            .s_text
            .lines()
            .filter_map(|l| l.trim().strip_prefix(".name: "))
            .take(ARGS.len())
            .map(str::to_owned)
            .collect();
        assert_eq!(
            names,
            ARGS.iter().map(|a| a.2.to_owned()).collect::<Vec<_>>(),
            "{symbol}: kernarg names"
        );
        let (last_off, last_size, _) = *ARGS.last().unwrap();
        assert_eq!(
            last_off + last_size,
            c1::KERNARG_BYTES,
            "{symbol}: arguments cover the kernarg block"
        );
        for (name, access) in [
            ("expert_weight_ptrs", "read_only"),
            ("topk", "read_only"),
            ("codec_table", "read_only"),
            ("X", "read_only"),
            ("Y0", "write_only"),
            ("Y1", "write_only"),
        ] {
            assert!(e.s_text.contains(&format!("      - .actual_access: {access}\n        .address_space: global\n        .name: {name}\n")), "{symbol}: {name} access {access}");
        }
        // The kernarg is loaded by four scalar loads covering offsets 0..76 (the reserved word is never loaded), no over-read;
        // `n_experts` lands in its own SGPR (s76, the declared ceiling is 77 SGPRs, `sgpr_max` of the contract).
        for load in [
            "s_load_b256 s[8:15], s[0:1], null",
            "s_load_b256 s[16:23], s[0:1], 0x20",
            "s_load_b64 s[24:25], s[0:1], 0x40",
            "s_load_b32 s76, s[0:1], 0x48",
        ] {
            assert!(e.s_text.contains(load), "{symbol}: kernarg load {load}");
        }
        assert!(
            e.s_text.contains(".amdhsa_next_free_sgpr 77\n"),
            "{symbol}: SGPR ceiling"
        );
        assert_eq!(e.shape.next_free_sgpr, 77, "{symbol}");
        assert!(
            e.proof
                .reg_plan
                .iter()
                .any(|r| r.name.as_ref() == "kernel_arguments_n_experts"),
            "{symbol}: typed n_experts ownership"
        );
        // The expert tensors and the codec table are never written, Y is only stored by lane 0 (checked in the census).
        assert!(
            e.proof
                .reg_plan
                .iter()
                .any(|r| r.name.as_ref() == "kernel_arguments_ptrs"),
            "{symbol}: typed kernarg ownership"
        );
    }
}

fn u16_at(b: &[u8], at: usize) -> usize {
    u16::from_le_bytes(b[at..at + 2].try_into().unwrap()) as usize
}
fn u32_at(b: &[u8], at: usize) -> usize {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap()) as usize
}
fn u64_at(b: &[u8], at: usize) -> usize {
    u64::from_le_bytes(b[at..at + 8].try_into().unwrap()) as usize
}
/// The bytes of the named section.
fn section(elf: &[u8], name: &str) -> Vec<u8> {
    let (shoff, entsize, count, strndx) = (
        u64_at(elf, 40),
        u16_at(elf, 58),
        u16_at(elf, 60),
        u16_at(elf, 62),
    );
    let header = |i: usize| shoff + i * entsize;
    let names = u64_at(elf, header(strndx) + 24);
    for i in 0..count {
        let at = names + u32_at(elf, header(i));
        let end = at + elf[at..].iter().position(|&c| c == 0).unwrap();
        if &elf[at..end] == name.as_bytes() {
            let (offset, size) = (u64_at(elf, header(i) + 24), u64_at(elf, header(i) + 32));
            return elf[offset..offset + size].to_vec();
        }
    }
    panic!("no section {name}")
}
fn function_sizes(elf: &[u8]) -> Vec<usize> {
    section(elf, ".symtab")
        .chunks(24)
        .filter(|s| s[4] & 0xf == 2)
        .map(|s| u64_at(s, 16))
        .collect()
}

/// The module (and each entry) carries the gfx11 `s_code_end` guard exactly once at the end, the native writer lays it
/// out after the last kernel (128-byte alignment fill then 96 `s_code_end` dwords) without moving any kernel, symbol
/// size or descriptor.
#[test]
fn code_end_tail_is_present_once_and_changes_nothing_else() {
    let (emitted, text, _) = c1::emit_module().unwrap();
    for t in emitted
        .iter()
        .map(|e| e.s_text.as_str())
        .chain([text.as_str()])
    {
        assert!(t.ends_with(CODE_END_TAIL));
        assert_eq!(t.matches(".p2alignl").count(), 1);
        assert_eq!(t.matches(".fill").count(), 1);
    }
    let bare = text
        .strip_suffix(CODE_END_TAIL)
        .expect("module ends with the guard");
    let guarded = native::assemble(&text, Arch::Gfx1151).unwrap();
    let plain = native::assemble(bare, Arch::Gfx1151).unwrap();
    let (guarded_text, bare_text) = (section(&guarded, ".text"), section(&plain, ".text"));
    assert_eq!(
        guarded_text[..bare_text.len()],
        bare_text[..],
        "the guard only appends"
    );
    assert!(
        guarded_text[bare_text.len()..]
            .chunks(4)
            .all(|w| w == S_CODE_END),
        "alignment fill and guard are s_code_end"
    );
    assert!(
        guarded_text.len() >= bare_text.len() + 96 * 4 && guarded_text.len() % 128 == 0,
        "128-byte aligned code end plus 96 guard dwords"
    );
    assert!(
        guarded_text
            .chunks(4)
            .rev()
            .take(96)
            .all(|w| w == S_CODE_END),
        "the last 96 dwords are s_code_end"
    );
    assert_eq!(function_sizes(&guarded), function_sizes(&plain));
    assert_eq!(section(&guarded, ".rodata"), section(&plain, ".rodata"));
    assert_eq!(function_sizes(&guarded).len(), SYMBOLS.len());
}

/// Per codec body: the X staging loads (`global_load_b128`) precede every weight load (b64 / b32 / u16), every weight load
/// of the body precedes its first `s_barrier` and none follows it, the single explicit `s_waitcnt vmcnt(<weight loads>)`
/// sits between the last weight load and the first `ds_store_b128`, and the post-barrier `vmcnt` ladder is the exact
/// per-group descending sequence (the loads of a group are the oldest not yet consumed).
#[test]
fn every_weight_load_precedes_the_barrier_and_the_vmcnt_ladder_is_exact() {
    for kind in Kind::ALL {
        let e = c1::emit(Spec::new(kind)).unwrap();
        let (w3, mq4) = bodies(&e.s_text, kind);
        let gate = kind == Kind::GateUp;
        // (body, weight loads, X loads, ds stores, ds loads, barriers, expected post-barrier vmcnt ladder)
        let expect: [(&Vec<&str>, usize, usize, usize, usize, usize, Vec<u32>); 2] = [
            (
                &w3,
                if gate { 20 } else { 10 },
                if gate { 3 } else { 1 },
                if gate { 3 } else { 1 },
                if gate { 20 } else { 5 },
                2,
                if gate {
                    (0..20).rev().collect()
                } else {
                    vec![8, 6, 4, 2, 0]
                },
            ),
            (
                &mq4,
                if gate { 20 } else { 10 },
                if gate { 3 } else { 1 },
                if gate { 3 } else { 1 },
                if gate { 20 } else { 5 },
                1,
                if gate {
                    (0..20).rev().collect()
                } else {
                    (0..10).rev().collect()
                },
            ),
        ];
        for (name, (body, weights, xs, stores, loads, barriers, ladder)) in
            ["w3", "mq4"].into_iter().zip(expect)
        {
            let ctx = format!("{} {name}", tag(kind));
            let idx = |pred: &dyn Fn(&str) -> bool| -> Vec<usize> {
                body.iter()
                    .enumerate()
                    .filter(|(_, l)| pred(l))
                    .map(|(i, _)| i)
                    .collect()
            };
            let weight_ix = idx(&|l| {
                matches!(
                    mnemonic(l),
                    "global_load_b64" | "global_load_b32" | "global_load_u16"
                )
            });
            let x_ix = idx(&|l| mnemonic(l) == "global_load_b128");
            let barrier_ix = idx(&|l| mnemonic(l) == "s_barrier");
            let store_ix = idx(&|l| mnemonic(l) == "ds_store_b128");
            assert_eq!(weight_ix.len(), weights, "{ctx}: weight loads");
            assert_eq!(x_ix.len(), xs, "{ctx}: X staging loads");
            assert_eq!(store_ix.len(), stores, "{ctx}: X stores");
            assert_eq!(
                idx(&|l| mnemonic(l) == "ds_load_b128").len(),
                loads,
                "{ctx}: LDS X reads"
            );
            assert_eq!(
                barrier_ix.len(),
                barriers,
                "{ctx}: barriers (the W3 body also retires the slot for the MQ4 body)"
            );
            assert!(
                x_ix.iter().all(|x| weight_ix.iter().all(|w| x < w)),
                "{ctx}: X loads precede every weight load"
            );
            assert!(
                weight_ix.iter().all(|w| *w < barrier_ix[0]),
                "{ctx}: every weight load precedes the first barrier"
            );
            assert!(
                barrier_ix
                    .iter()
                    .skip(1)
                    .all(|b| weight_ix.iter().all(|w| w < b))
                    && weight_ix.last() < barrier_ix.first(),
                "{ctx}: no weight load after the barrier"
            );
            assert!(
                store_ix
                    .iter()
                    .all(|s| *s < barrier_ix[0] && *s > *weight_ix.last().unwrap()),
                "{ctx}: X stores sit between the loads and the barrier"
            );
            // step 4: one explicit vmcnt(<weight loads>) after the last weight load, before the first store
            let wait_ix = idx(&|l| *l == format!("s_waitcnt vmcnt({weights})"));
            assert_eq!(wait_ix.len(), 1, "{ctx}: the step-4 wait");
            assert!(
                wait_ix[0] > *weight_ix.last().unwrap() && wait_ix[0] < store_ix[0],
                "{ctx}: step-4 wait placement"
            );
            // step 5: the vmcnt waits after the barrier, in order
            let after: Vec<u32> = body[barrier_ix[0]..]
                .iter()
                .filter_map(|l| l.strip_prefix("s_waitcnt vmcnt("))
                .filter_map(|r| r.split(')').next())
                .map(|n| n.parse().unwrap())
                .collect();
            assert_eq!(after, ladder, "{ctx}: vmcnt ladder");
            // stores: every Y store is lane 0 only (v_cmpx) and the only global store of the body
            assert_eq!(
                count(body, "global_store_b32"),
                1,
                "{ctx}: one Y store per wave"
            );
            assert_eq!(count(body, "v_cmpx_eq_u32_e32"), 1, "{ctx}");
            assert_eq!(count(body, "s_endpgm"), 1, "{ctx}");
            assert!(
                !body.iter().any(|l| mnemonic(l).starts_with("buffer_")
                    || mnemonic(l).starts_with("flat_")
                    || mnemonic(l).starts_with("scratch_")
                    || mnemonic(l).starts_with("global_atomic")),
                "{ctx}: no other memory ops"
            );
        }
    }
}

/// The addresses of the weight loads (group stride, packet / header / payload immediates) match the formats.
#[test]
fn weight_load_addresses_follow_the_group_formats() {
    for kind in Kind::ALL {
        let e = c1::emit(Spec::new(kind)).unwrap();
        let (w3, mq4) = bodies(&e.s_text, kind);
        let gate = kind == Kind::GateUp;
        // W3: one b64 window per 48-byte group (+ the down-only u16 scale at +2g from the scale pointer)
        let b64: Vec<u32> = w3
            .iter()
            .filter(|l| mnemonic(l) == "global_load_b64")
            .map(|l| imm_offset(l))
            .collect();
        let groups = kind.w3_groups();
        assert_eq!(
            b64,
            (0..groups).map(|g| 48 * g).collect::<Vec<_>>(),
            "{} W3 packet windows",
            tag(kind)
        );
        let scales: Vec<u32> = w3
            .iter()
            .filter(|l| mnemonic(l) == "global_load_u16")
            .map(|l| imm_offset(l))
            .collect();
        assert_eq!(
            scales,
            if gate {
                vec![]
            } else {
                (0..groups).map(|g| 2 * g).collect::<Vec<_>>()
            },
            "{} W3 scales",
            tag(kind)
        );
        if gate {
            // gate scales: the row's 40 bytes at +40*row, ten SGPRs
            assert!(
                w3.contains(&"s_load_b256 s[64:71], s[58:59], null")
                    && w3.contains(&"s_load_b64 s[72:73], s[58:59], 0x20")
            );
        }
        // MQ4: gate header at +136g, payload at +136g + 8; down header at +68g, payload u16 at +68g + 4
        let (hdr, pay): (Vec<u32>, Vec<u32>) = {
            let loads: Vec<&&str> = mq4
                .iter()
                .filter(|l| matches!(mnemonic(l), "global_load_b32" | "global_load_u16"))
                .collect();
            (
                loads.iter().step_by(2).map(|l| imm_offset(l)).collect(),
                loads
                    .iter()
                    .skip(1)
                    .step_by(2)
                    .map(|l| imm_offset(l))
                    .collect(),
            )
        };
        let (stride, pay_off, g4) = if gate { (136, 8, 10) } else { (68, 4, 5) };
        assert_eq!(
            hdr,
            (0..g4).map(|g| stride * g).collect::<Vec<_>>(),
            "{} MQ4 headers",
            tag(kind)
        );
        assert_eq!(
            pay,
            (0..g4).map(|g| stride * g + pay_off).collect::<Vec<_>>(),
            "{} MQ4 payloads",
            tag(kind)
        );
        let payload_mnemonic = if gate {
            "global_load_b32"
        } else {
            "global_load_u16"
        };
        assert_eq!(
            mq4.iter()
                .filter(|l| mnemonic(l) == payload_mnemonic)
                .count(),
            if gate { 2 * g4 as usize } else { g4 as usize },
            "{} payload-class loads",
            tag(kind)
        );
        // LDS: W3 and MQ4 down read 16 B per lane at 512g; MQ4 gate two 16 B at 1024g and +16
        let lds = |body: &[&str]| -> Vec<u32> {
            body.iter()
                .filter(|l| mnemonic(l) == "ds_load_b128")
                .map(|l| imm_offset(l))
                .collect()
        };
        assert_eq!(
            lds(&w3),
            (0..groups).map(|g| 512 * g).collect::<Vec<_>>(),
            "{} W3 LDS",
            tag(kind)
        );
        assert_eq!(
            lds(&mq4),
            if gate {
                (0..10)
                    .flat_map(|g| [1024 * g, 1024 * g + 16])
                    .collect::<Vec<_>>()
            } else {
                (0..5).map(|g| 512 * g).collect()
            },
            "{} MQ4 LDS",
            tag(kind)
        );
    }
}

/// Per-codec census of the frozen numerical tree: the operations, their counts, the chain assignment `acc[g & 3]`
/// (down: one chain), the gate fold `(a0+a1)+(a2+a3)`, the five snapshot shuffle steps with a mask select, the lane-0 store,
/// and nothing else on the value path (no packed math, dot, fma_mix, mad, VOPD, exp).
#[test]
fn instruction_census_matches_the_frozen_dag() {
    for kind in Kind::ALL {
        let e = c1::emit(Spec::new(kind)).unwrap();
        let (w3, mq4) = bodies(&e.s_text, kind);
        let gate = kind == Kind::GateUp;
        let fold = if gate { 3 } else { 0 };
        for (name, body, groups) in [
            ("w3", &w3, kind.w3_groups() as usize),
            ("mq4", &mq4, kind.mq4_groups() as usize),
        ] {
            let ctx = format!("{} {name}", tag(kind));
            let g = groups;
            // (mnemonic, expected count)
            let expect: Vec<(&str, usize)> = if name == "w3" {
                vec![
                    ("v_lshrrev_b64", g),
                    ("v_bfe_u32", 4 * g),
                    ("v_lshl_add_u32", 4 * g),
                    ("v_cvt_f32_i32_e32", 4 * g),
                    ("v_cvt_f32_f16_e64", g),
                    ("v_mul_f32_e32", 5 * g),
                    ("v_fmac_f32_e32", 3 * g),
                    ("v_add_f32_e32", g + fold + 5),
                    ("v_fma_f32", 0),
                    ("v_cvt_f32_u32_e32", 0),
                ]
            } else {
                let nw = if gate { 8 } else { 4 };
                vec![
                    ("v_lshrrev_b64", 0),
                    ("v_bfe_u32", nw * g),
                    ("v_lshl_add_u32", 0),
                    ("v_cvt_f32_i32_e32", 0),
                    ("v_cvt_f32_u32_e32", nw * g),
                    ("v_fma_f32", nw * g),
                    ("v_cvt_f32_f16_e64", 2 * g),
                    ("v_lshrrev_b32_e32", g),
                    ("v_mul_f32_e32", g),
                    ("v_fmac_f32_e32", (nw - 1) * g),
                    ("v_add_f32_e32", g + fold + 5),
                ]
            };
            for (m, n) in expect {
                assert_eq!(count(body, m), n, "{ctx}: {m}");
            }
            assert_eq!(
                count(body, "ds_bpermute_b32"),
                5,
                "{ctx}: five shuffle steps"
            );
            assert_eq!(
                count(body, "v_cndmask_b32_e64"),
                5,
                "{ctx}: five mask selects"
            );
            assert!(
                !body.iter().any(|l| {
                    let m = mnemonic(l);
                    m.starts_with("v_pk_")
                        || m.starts_with("v_dot")
                        || m.starts_with("v_wmma")
                        || m.starts_with("v_swmmac")
                        || m.starts_with("v_fma_mix")
                        || m.starts_with("v_mad")
                        || m.starts_with("v_fmamk")
                        || m.starts_with("v_fmaak")
                        || m.starts_with("v_dual")
                        || m.starts_with("v_exp")
                        || m.starts_with("v_rcp")
                }),
                "{ctx}: forbidden value-path instruction"
            );
            // chain adds: the first `groups` v_add_f32 are acc[c] = acc[c] + p (acc is src0), then the fold, then the shuffle adds
            let adds: Vec<&&str> = body
                .iter()
                .filter(|l| mnemonic(l) == "v_add_f32_e32")
                .collect();
            let chain = &adds[..g];
            let fold_adds = &adds[g..g + fold];
            let shuffle = &adds[g + fold..];
            assert_eq!(shuffle.len(), 5, "{ctx}");
            let acc: Vec<u32> = if gate {
                let (f1, f2, f3) = (fold_adds[0], fold_adds[1], fold_adds[2]);
                let (a0, a1, a2, a3) = (
                    vreg(&operand(f1, 0)),
                    vreg(&operand(f1, 2)),
                    vreg(&operand(f2, 0)),
                    vreg(&operand(f2, 2)),
                );
                assert_eq!(
                    (operand(f1, 1), operand(f2, 1)),
                    (format!("v{a0}"), format!("v{a2}")),
                    "{ctx}: fold keeps the accumulator as src0"
                );
                assert_eq!(
                    (vreg(&operand(f3, 0)), operand(f3, 1), vreg(&operand(f3, 2))),
                    (a0, format!("v{a0}"), a2),
                    "{ctx}: a0 = a0 + a2"
                );
                assert_eq!(
                    [a1, a2, a3],
                    [a0 + 1, a0 + 2, a0 + 3],
                    "{ctx}: four consecutive chains"
                );
                vec![a0, a1, a2, a3]
            } else {
                vec![vreg(&operand(shuffle[0], 1))]
            };
            for (gi, add) in chain.iter().enumerate() {
                let c = acc[gi % acc.len()];
                assert_eq!(
                    (vreg(&operand(add, 0)), operand(add, 1)),
                    (c, format!("v{c}")),
                    "{ctx}: group {gi} adds into chain {}",
                    gi % acc.len()
                );
            }
            // the shuffle: ds_bpermute exch, shuf_k, a0 ; exch = a0 + exch ; a0 = cndmask(a0, exch, mask_k)
            let a0 = acc[0];
            let perm: Vec<&&str> = body
                .iter()
                .filter(|l| mnemonic(l) == "ds_bpermute_b32")
                .collect();
            let masks: Vec<String> = body
                .iter()
                .filter(|l| mnemonic(l) == "v_cndmask_b32_e64")
                .map(|l| operand(l, 3))
                .collect();
            assert_eq!(masks.len(), 5);
            let exch = operand(perm[0], 0);
            for (k, p) in perm.iter().enumerate() {
                assert_eq!(
                    (operand(p, 0), operand(p, 2)),
                    (exch.clone(), format!("v{a0}")),
                    "{ctx}: bpermute {k} reads the accumulator"
                );
                assert_eq!(
                    (
                        operand(shuffle[k], 0),
                        operand(shuffle[k], 1),
                        operand(shuffle[k], 2)
                    ),
                    (exch.clone(), format!("v{a0}"), exch.clone()),
                    "{ctx}: exch = a0 + exch ({k})"
                );
            }
            let selects: Vec<&&str> = body
                .iter()
                .filter(|l| mnemonic(l) == "v_cndmask_b32_e64" && operand(l, 0) == format!("v{a0}"))
                .collect();
            assert_eq!(selects.len(), 5, "{ctx}: a0 = cndmask(a0, exch, mask)");
            for s in selects {
                assert_eq!(
                    (operand(s, 1), operand(s, 2)),
                    (format!("v{a0}"), exch.clone())
                );
            }
            // accumulators start at +0 and are only reduced into by the adds above
            assert_eq!(
                body.iter()
                    .filter(|l| l.starts_with("v_mov_b32_e32") && l.ends_with(", 0"))
                    .count(),
                acc.len(),
                "{ctx}: accumulators start at +0"
            );
        }
        // The unpack ties of one group of each codec (first group of the body), listed as exact operand patterns.
        let first = |body: &[&str], m: &str| body.iter().position(|l| mnemonic(l) == m).unwrap();
        let g0 = first(&w3, "v_lshrrev_b64");
        let a = &w3[g0];
        // `v_lshrrev_b64 v[pk:pk+1], shift, v[lo:hi]`: the packet is the low dword of the destination pair
        let pkt = format!(
            "v{}",
            operand(a, 0)
                .trim_start_matches("v[")
                .split(':')
                .next()
                .unwrap()
        );
        for (i, bfe) in w3[g0 + 1..]
            .iter()
            .filter(|l| mnemonic(l) == "v_bfe_u32")
            .take(4)
            .enumerate()
        {
            assert_eq!(
                (operand(bfe, 1), operand(bfe, 2), operand(bfe, 3)),
                (pkt.clone(), (3 * i).to_string(), "3".into()),
                "W3 code {i}: bfe(pkt, 3i, 3)"
            );
        }
        let lsa = w3.iter().find(|l| mnemonic(l) == "v_lshl_add_u32").unwrap();
        assert_eq!(
            (operand(lsa, 2), operand(lsa, 3)),
            ("1".into(), "-7".into()),
            "W3 grid: (c << 1) + -7"
        );
        let m0 = first(&mq4, "v_fma_f32");
        let fma = &mq4[m0];
        assert_eq!(
            operand(fma, 0),
            operand(fma, 2),
            "MQ4: w = fma(d, q, z) in place"
        );
    }
}

/// The shape contract derives deterministically from emission, covers both entries, forbids scratch/flat/WMMA/mix/dot/pk
/// and the gfx12 wait family (but not LDS, which this kernel uses), pins the register ceilings to the plan and the fixed LDS.
#[test]
fn shape_contract_is_derived_from_the_emission() {
    for spec in arch_specs() {
        let e = c1::emit(spec).unwrap();
        let c = c1::contract(spec, &e);
        assert_eq!(c, c1::contract(spec, &c1::emit(spec).unwrap()));
        assert_eq!(c["symbol"].as_str(), Some(spec.symbol().as_str()));
        assert_eq!(
            (c["vgpr_max"].as_u64(), c["sgpr_max"].as_u64()),
            (
                Some(u64::from(e.shape.next_free_vgpr)),
                Some(u64::from(e.shape.next_free_sgpr))
            )
        );
        assert_eq!(
            c["group_segment_fixed_bytes"].as_u64(),
            Some(u64::from(spec.kind.lds_bytes()))
        );
        assert_eq!(c["launch_dynamic_lds_bytes"].as_u64(), Some(0));
        for flag in [
            "require_wave32",
            "require_zero_spills",
            "require_zero_private",
        ] {
            assert_eq!(c[flag], true, "{flag}");
        }
        let forbidden: Vec<&str> = c["forbidden"]
            .as_array()
            .unwrap()
            .iter()
            .map(|f| f.as_str().unwrap())
            .collect();
        for f in [
            "scratch_*",
            "flat_*",
            "v_wmma*",
            "v_swmmac*",
            "v_fma_mix*",
            "v_dot*",
            "v_pk_*",
            "s_wait_loadcnt*",
            "s_wait_alu*",
        ] {
            assert!(forbidden.contains(&f), "{f}");
        }
        assert!(
            !forbidden
                .iter()
                .any(|f| f.starts_with("ds_") || f.starts_with("s_barrier")),
            "this kernel stages X in LDS"
        );
        // the counts equal the emitted text
        for (name, n) in c["counts"].as_object().unwrap() {
            let have = e.s_text.lines().filter(|l| mnemonic(l) == name).count() as u64;
            assert_eq!(n.as_u64(), Some(have), "{} counts {name}", spec.symbol());
        }
        assert_eq!(
            c["counts"]["s_barrier"].as_u64(),
            Some(3),
            "{}: two barriers in the W3 body (publish, retire), one in MQ4",
            spec.symbol()
        );
    }
}

/// The shipped `kernels/mq_i4_moe_gemv_c1_pm_gfx1151.hxaco` is exactly what the builder emits today (the whole offload bundle,
/// byte for byte, native writer, no ROCm), and the committed contract files are the derived contracts.
#[test]
fn committed_bundle_and_contracts_equal_fresh_emission() {
    let root = format!("{}/../..", env!("CARGO_MANIFEST_DIR"));
    let path = format!("{root}/kernels/{MODULE}.hxaco");
    let committed = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e} (emit it: peacemaker-isa emit --kernel mq_i4_moe_gemv_c1 --arch gfx1151 --epi all ... --bundle)"));
    let (emitted, text, _) = c1::emit_module().unwrap();
    let elf = native::assemble(&text, Arch::Gfx1151).unwrap();
    let fresh = native::bundle(&elf, Arch::Gfx1151, native::DEFAULT_HOST_TARGET);
    assert!(
        committed == fresh,
        "{MODULE}: fresh emission differs from the committed bundle {path}"
    );
    for (spec, e) in arch_specs().into_iter().zip(&emitted) {
        let file = format!(
            "{}/kernels/mq_i4_moe_gemv_c1.gfx1151.{}.contract.json",
            env!("CARGO_MANIFEST_DIR"),
            spec.tag()
        );
        let committed: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&file).unwrap_or_else(|e| panic!("{file}: {e}")))
                .unwrap();
        assert_eq!(
            committed,
            c1::contract(spec, e),
            "{file}: committed contract differs from the derived contract"
        );
    }
}

/// The module and every entry assemble with the native writer (no ROCm), and the bundle wraps the object.
#[test]
fn native_writer_assembles_every_entry_and_the_module() {
    let (emitted, text, _) = c1::emit_module().unwrap();
    for e in &emitted {
        let elf = native::assemble(&e.s_text, Arch::Gfx1151)
            .unwrap_or_else(|err| panic!("{}: {err}", e.proof.kernel_id));
        assert!(elf.starts_with(b"\x7fELF"));
    }
    let elf = native::assemble(&text, Arch::Gfx1151).unwrap();
    assert!(native::bundle(&elf, Arch::Gfx1151, native::DEFAULT_HOST_TARGET).len() > elf.len());
}

/// The expert-id range check is the FIRST use of the loaded id (after the `n_experts` kernarg load) and is an UNSIGNED
/// `s_cmp_ge_u32` immediately followed by the uniform branch to the exit label; before it the kernel has issued no
/// barrier, no LDS access, no global access and no scalar load but the kernarg loads and the `topk[u]` load itself; the id
/// shift, the weight-pointer table load and the codec-table read all follow it; the exit is shared with the K / `wg_y` guards.
#[test]
fn expert_id_guard_is_the_first_use_of_the_id_before_any_table_read_prefetch_or_barrier() {
    const EID: &str = "s26";
    const NEXP: &str = "s76";
    for spec in arch_specs() {
        let e = c1::emit(spec).unwrap();
        let (tag, symbol) = (tag(spec.kind), spec.symbol());
        let lines: Vec<&str> = e.s_text.lines().map(str::trim).collect();
        let exit_label = format!(".Lc1_{tag}_exit");
        let exit_branch = format!("s_cbranch_scc1 {exit_label}");
        let pos = |pred: &dyn Fn(&str) -> bool| {
            lines
                .iter()
                .position(|l| pred(*l))
                .unwrap_or_else(|| panic!("{symbol}: line not found"))
        };
        let uses_eid = |l: &str| {
            l.split_whitespace()
                .skip(1)
                .any(|t| t.trim_end_matches(',') == EID)
        };
        let nexp_load = pos(&|l: &str| l == "s_load_b32 s76, s[0:1], 0x48");
        let eid_load = pos(&|l: &str| l.starts_with("s_load_b32 s26, "));
        let cmp = (eid_load + 1..lines.len())
            .find(|&i| uses_eid(lines[i]))
            .unwrap();
        assert_eq!(
            lines[cmp],
            format!("s_cmp_ge_u32 {EID}, {NEXP}"),
            "{symbol}: the first use of the id is the unsigned n_experts compare"
        );
        assert!(
            nexp_load < eid_load && eid_load < cmp,
            "{symbol}: n_experts load, id load, compare order"
        );
        assert_eq!(
            lines[cmp + 1],
            exit_branch,
            "{symbol}: uniform exit right after the compare"
        );
        assert!(
            !lines[..cmp]
                .iter()
                .any(|l| uses_eid(l) && !l.starts_with("s_load_b32 s26, ")),
            "{symbol}: id used before the compare"
        );
        // Nothing before the compare touches memory but the kernarg loads and the topk[u] load.
        for l in &lines[..cmp] {
            let m = mnemonic(l);
            assert!(
                !(m == "s_barrier"
                    || m.starts_with("global_")
                    || m.starts_with("ds_load")
                    || m.starts_with("ds_store")
                    || m.starts_with("flat_")
                    || m.starts_with("scratch_")),
                "{symbol}: `{l}` before the id guard"
            );
            if m.starts_with("s_load_") {
                let base = l.split(", ").nth(1).unwrap_or_default();
                assert!(
                    base == "s[0:1]" || base == "s[32:33]",
                    "{symbol}: `{l}` is neither a kernarg load nor the topk load"
                );
            }
        }
        // The id shift, the weight-pointer table address and the codec table read follow the guard.
        let shift = pos(&|l: &str| l == "s_lshl_b32 s28, s26, 3");
        assert!(cmp < shift, "{symbol}: id shift after the guard");
        let table_loads: Vec<usize> = (0..lines.len())
            .filter(|&i| lines[i].starts_with("s_load_b64 s[38:39], "))
            .collect();
        assert_eq!(
            table_loads.len(),
            1,
            "{symbol}: one weight-pointer table load"
        );
        assert!(
            cmp < table_loads[0],
            "{symbol}: weight-pointer table load after the guard"
        );
        for i in (0..lines.len()).filter(|&i| lines[i].starts_with("s_load_b32 s29, ")) {
            assert!(cmp < i, "{symbol}: codec table read after the guard")
        }
        // Shared exit: wg_y, K and eid guards all branch to it before the first codec body.
        let w3 = pos(&|l: &str| l == format!(".Lc1_{tag}_w3:"));
        assert_eq!(
            lines[..w3].iter().filter(|l| **l == exit_branch).count(),
            3,
            "{symbol}: wg_y, K, expert-id exits"
        );
        assert!(
            lines[..w3]
                .iter()
                .filter(|l| mnemonic(l) == "s_cmp_ge_u32")
                .any(|l| *l == "s_cmp_ge_u32 s3, 10"),
            "{symbol}: wg_y guard unchanged"
        );
    }
}

// ------------------------------------------------------------------------------------------------ CPU interpreter
// The real emitted code object, lifted and run on the peacemaker CPU interpreter: invalid expert ids exit with no
// store and no dereference, valid ids are untouched by the guard. Tensor data are random full-mantissa as in K1's
// wide-emulator test; only experts 0 and `E_N - 1` are mapped, and the weight-pointer table is exactly `E_N` entries, so a
// table read at or past `E_N` (or an expert dereference through a null table slot) faults.

use peacemaker_ir::emu::{Emulator, Launch, Memory};
use peacemaker_ir::inst::Frontend;

const SENTINEL: u8 = 0xa5;
const GUARD: usize = 64;
const E_KERNARG: u64 = 0x1_0000;
const E_TABLE: u64 = 0x10_0000;
const E_CTAB: u64 = 0x18_0000;
const E_TOPK: u64 = 0x20_0000;
const E_EXPERTS: u64 = 0x1000_0000;
const E_STRIDE: u64 = 0x40_0000;
const E_X: u64 = 0x8000_0000;
const E_Y0: u64 = 0x9000_0000;
const E_Y1: u64 = 0xa000_0000;
/// The extent of the weight-pointer table (`n_experts`) of every case.
const E_N: u32 = 8;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// Normal f16 scale with exponent field 8..=20: `-8 * sc` is then exactly the symmetric MQ4 zero.
fn scale_bits(rng: &mut Rng) -> u16 {
    (((8 + rng.below(13)) as u16) << 10) | (rng.next() as u16 & 0x3ff)
}
fn sym_zero_bits(sc: u16) -> u16 {
    0x8000 | (sc + (3 << 10))
}

/// Planar W3 tensor: per row `K*3/8` code bytes (128 3-bit codes per 48-byte group), then the f16 scale plane.
fn w3_tensor(rng: &mut Rng, m: usize, k: usize) -> Vec<u8> {
    let (groups, row_bytes) = (k / 128, k * 3 / 8);
    let mut out = vec![0u8; m * row_bytes + 2 * m * groups];
    for r in 0..m {
        for g in 0..groups {
            let base = r * row_bytes + g * 48;
            for j in 0..128 {
                let code = rng.below(8) as u32;
                for b in 0..3 {
                    if (code >> b) & 1 == 1 {
                        out[base + (3 * j + b) / 8] |= 1 << ((3 * j + b) % 8);
                    }
                }
            }
            let off = m * row_bytes + 2 * (r * groups + g);
            out[off..off + 2].copy_from_slice(&scale_bits(rng).to_le_bytes());
        }
    }
    out
}

/// MQ4: gate G256 groups of 136 B (`[sc0 zp0 sc1 zp1]`, 128 payload bytes at +8), down G128 groups of 68 B (`[sc zp]`, 64 at +4).
fn mq4_tensor(rng: &mut Rng, kind: Kind, m: usize, k: usize) -> Vec<u8> {
    let (cols, stride, halves, payload_at, payload) = match kind {
        Kind::GateUp => (256, 136, 2, 8, 128),
        Kind::Down => (128, 68, 1, 4, 64),
    };
    let groups = k / cols;
    let mut out = vec![0u8; m * groups * stride];
    for r in 0..m {
        for g in 0..groups {
            let base = (r * groups + g) * stride;
            for h in 0..halves {
                let sc = scale_bits(rng);
                out[base + 4 * h..base + 4 * h + 2].copy_from_slice(&sc.to_le_bytes());
                out[base + 4 * h + 2..base + 4 * h + 4]
                    .copy_from_slice(&sym_zero_bits(sc).to_le_bytes());
            }
            for byte in &mut out[base + payload_at..base + payload_at + payload] {
                *byte = rng.next() as u8;
            }
        }
    }
    out
}

/// Full-mantissa activations over a wide exponent band, random signs.
fn x_rows(rng: &mut Rng, rows: usize, k: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(rows * k * 4);
    for _ in 0..rows * k {
        let (s, e, m) = (
            (rng.next() & 1) as u32,
            118 + rng.below(19) as u32,
            (rng.next() & 0x7f_ffff) as u32,
        );
        out.extend(((s << 31) | (e << 23) | m).to_le_bytes());
    }
    out
}

struct Rig {
    program: peacemaker_ir::Program,
}
impl Rig {
    fn new() -> Self {
        let (_, text, _) = c1::emit_module().unwrap();
        let elf = native::assemble(&text, Arch::Gfx1151).unwrap();
        let program = peacemaker_lift::lift_object(
            &elf,
            peacemaker_lift::Options {
                frontend: Frontend::Builder,
            },
        )
        .unwrap()
        .program;
        Rig { program }
    }
}

/// One projection kind's deterministic data: the X rows and one MQ4 and one W3 expert tensor.
struct Data {
    kind: Kind,
    m: usize,
    k: usize,
    xdiv: u32,
    ys: usize,
    x: Vec<u8>,
    mq4: Vec<u8>,
    w3: Vec<u8>,
}
fn data(kind: Kind) -> Data {
    let (m, xdiv, xrows, ys) = match kind {
        Kind::GateUp => (1280usize, 10u32, 1usize, 640usize),
        Kind::Down => (2560, 1, 10, 2560),
    };
    let k = kind.k() as usize;
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15 ^ k as u64);
    let mq4 = mq4_tensor(&mut rng, kind, m, k);
    let w3 = w3_tensor(&mut rng, m, k);
    let x = x_rows(&mut rng, xrows, k);
    Data {
        kind,
        m,
        k,
        xdiv,
        ys,
        x,
        mq4,
        w3,
    }
}

/// A codec path: `uniform_codec` (-1 = per-expert table) and the codec of experts `0` and `E_N - 1` (0 = MQ4, 1 = W3):
/// the expert tensors and the codec table both follow it.
struct Path {
    name: &'static str,
    uniform: i32,
    codecs: [u8; 2],
}
const PATHS: [Path; 4] = [
    Path {
        name: "uniform_mq4",
        uniform: 0,
        codecs: [0, 0],
    },
    Path {
        name: "uniform_w3",
        uniform: 1,
        codecs: [1, 1],
    },
    Path {
        name: "table_w3_then_mq4",
        uniform: -1,
        codecs: [1, 0],
    },
    Path {
        name: "table_mq4_then_w3",
        uniform: -1,
        codecs: [0, 1],
    },
];

/// The Y words and the number of Y bytes the launches wrote.
struct Out {
    y: Vec<u8>,
    written: usize,
}

/// Launch workgroups `(0, 0)` and `(M/8 - 1, 9)` (first and last row block, first and last slot; for the gate the second
/// lands in `Y1`) with every `topk` entry equal to `id` and `n_experts` as given.
fn run(rig: &Rig, d: &Data, p: &Path, id: i32, n_experts: u32) -> Result<Out, String> {
    let mut mem = Memory::default();
    let mut table = vec![0u8; E_N as usize * 8];
    for (slot, expert) in [(0usize, 0u32), (1, E_N - 1)] {
        let base = E_EXPERTS + u64::from(expert) * E_STRIDE;
        let bytes = if p.codecs[slot] == 0 {
            d.mq4.clone()
        } else {
            d.w3.clone()
        };
        assert!((bytes.len() as u64) < E_STRIDE);
        table[expert as usize * 8..expert as usize * 8 + 8].copy_from_slice(&base.to_le_bytes());
        mem.map(format!("e{expert}"), base, bytes)?;
    }
    let mut ctab = vec![2u8; E_N as usize];
    ctab[0] = p.codecs[0];
    ctab[E_N as usize - 1] = p.codecs[1];
    mem.map("table".into(), E_TABLE, table)?;
    mem.map("ctab".into(), E_CTAB, ctab)?;
    mem.map(
        "topk".into(),
        E_TOPK,
        (0..10).flat_map(|_| id.to_le_bytes()).collect(),
    )?;
    mem.map("x".into(), E_X, d.x.clone())?;
    let ybytes = 10 * d.ys * 4 + 2 * GUARD;
    mem.map("y0".into(), E_Y0, vec![SENTINEL; ybytes])?;
    let gate = d.kind == Kind::GateUp;
    if gate {
        mem.map("y1".into(), E_Y1, vec![SENTINEL; ybytes])?;
    }
    let mut karg = Vec::with_capacity(c1::KERNARG_BYTES as usize);
    for ptr in [
        E_TABLE,
        E_TOPK,
        E_CTAB,
        E_X,
        E_Y0 + GUARD as u64,
        if gate { E_Y1 + GUARD as u64 } else { 0 },
    ] {
        karg.extend(ptr.to_le_bytes());
    }
    // M, K, uniform_codec, x_row_div, x_row_stride, y_slot_stride, n_experts@72, reserved@76 (zero).
    for v in [
        d.m as u32,
        d.k as u32,
        p.uniform as u32,
        d.xdiv,
        d.k as u32,
        d.ys as u32,
        n_experts,
        0,
    ] {
        karg.extend(v.to_le_bytes());
    }
    assert_eq!(karg.len(), c1::KERNARG_BYTES as usize);
    mem.map("kernarg".into(), E_KERNARG, karg)?;
    let emu = Emulator::new(&rig.program, &Spec::new(d.kind).symbol())?;
    for wg in [[0u32, 0, 0], [d.m as u32 / 8 - 1, 9, 0]] {
        emu.run_workgroup(
            &Launch {
                kernarg_ptr: E_KERNARG,
                workgroup: wg,
                block: [256, 1, 1],
                dynamic_lds: 0,
            },
            &mut mem,
        )?;
    }
    let (mut y, mut written) = (Vec::new(), 0);
    for r in mem
        .regions
        .iter()
        .filter(|r| r.name == "y0" || r.name == "y1")
    {
        y.extend_from_slice(&r.bytes);
        written += r.written.count();
    }
    Ok(Out { y, written })
}

/// Invalid ids `-1`, `i32::MIN` (whose `<< 3` is 0: unguarded it would alias table entry 0 and store through expert 0), `n`
/// and `n + 1` (one past a table with exactly `n` entries, whose weight-pointer and codec-table reads would fault) exit
/// with no store; the valid id `n - 1` produces exactly the bytes the permissive `n_experts = u32::MAX` run produces and
/// writes the sixteen lane-0 words of the two workgroups. All four codec paths (uniform MQ4 / W3, table both orders), both kinds.
#[test]
fn out_of_range_expert_ids_exit_with_no_store_or_dereference_and_valid_ids_are_unchanged() {
    let rig = Rig::new();
    let n = E_N as i32;
    for kind in Kind::ALL {
        let d = data(kind);
        for p in &PATHS {
            let case = format!("{} {}", tag(kind), p.name);
            for id in [-1, i32::MIN, n, n + 1] {
                let out = run(&rig, &d, p, id, E_N).unwrap_or_else(|e| panic!("{case}: id {id}: interpreter fault (an expert or table read escaped the guard): {e}"));
                assert_eq!(out.written, 0, "{case}: id {id} stored");
                assert!(
                    out.y.iter().all(|&b| b == SENTINEL),
                    "{case}: id {id} changed Y"
                );
            }
            let valid = run(&rig, &d, p, n - 1, E_N)
                .unwrap_or_else(|e| panic!("{case}: valid id {}: {e}", n - 1));
            assert_eq!(
                valid.written,
                16 * 4,
                "{case}: valid id writes one lane-0 word per wave of both workgroups"
            );
            for w in valid
                .y
                .chunks(4)
                .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
                .filter(|&w| w != 0xa5a5_a5a5)
            {
                assert!(
                    f32::from_bits(w).is_finite(),
                    "{case}: non-finite output word {w:#x}"
                );
            }
            let open = run(&rig, &d, p, n - 1, u32::MAX).unwrap();
            assert!(
                valid.y == open.y && valid.written == open.written,
                "{case}: the guard changed a valid launch"
            );
        }
    }
}

#[cfg(feature = "toolchain")]
mod toolchain {
    use super::*;
    use peacemaker_isa::{ledger_replay, pm_check, toolchain::IsaShapeContract};

    /// Every symbol's waits replay independently of the builder's ledger (entry and module text), the contract
    /// parses as an `IsaShapeContract`, and M7 lifts the natively linked object byte-exactly with no obligations.
    #[test]
    fn every_export_passes_wait_replay_and_m7() {
        let (emitted, text, _) = c1::emit_module().unwrap();
        for (spec, e) in arch_specs().into_iter().zip(&emitted) {
            ledger_replay::replay_waits(&e.s_text, Arch::Gfx1151)
                .unwrap_or_else(|err| panic!("{}: {err}", e.proof.variant));
            assert!(e.s_text.contains(".kernarg_segment_size: 80\n"));
            serde_json::from_value::<IsaShapeContract>(c1::contract(spec, e)).unwrap();
        }
        ledger_replay::replay_waits(&text, Arch::Gfx1151).unwrap();
        let dir =
            std::env::temp_dir().join(format!("peacemaker-isa-mqi4c1-m7-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let co = dir.join("module.co");
        std::fs::write(&co, native::assemble(&text, Arch::Gfx1151).unwrap()).unwrap();
        for symbol in SYMBOLS {
            let m7 =
                pm_check::m7(&co, "gfx1151", symbol).unwrap_or_else(|e| panic!("{symbol}: {e}"));
            assert_eq!(
                (m7["lift"].as_str(), &m7["obligations"]),
                (Some("byte-exact"), &serde_json::json!({})),
                "{symbol}"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
