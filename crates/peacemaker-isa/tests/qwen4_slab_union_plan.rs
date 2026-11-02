// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Execute the emitted integer planner, not a Rust stand-in for its instructions.
//! The independent oracle groups routed slots directly, preserving the current
//! HIP planner's first-occurrence law and adapting it to fixed eight-row tiles.
use peacemaker_ir::{
    emu::{Emulator, Launch, Memory},
    inst::Frontend,
};
use peacemaker_isa::{kernels::qwen4_slab_union_plan as planner, Arch, Emitted};

const WORDS: usize = 482;
const CAPACITY: usize = 80;
const MAX_TILES: usize = 8;
const GUARD: usize = 64;
const POISON: u8 = 0xa5;
const TOPK: u64 = 0x10_0000;
const CODEC: u64 = 0x20_0000;
const OUTPUT: u64 = 0x30_0000;
const KERNARG: u64 = 0x40_0000;

/// This deliberately does not transcribe the emitter's rank/head/LDS scans.
/// Each bucket accumulates an ordered list of (expert, absolute slot list),
/// then serializes that list into a completely initialized descriptor.
fn oracle(topk: &[i32], codec: Option<&[u8]>, rows: i32, uniform: i32, experts: i32) -> Vec<u32> {
    assert!((1..=64).contains(&rows));
    assert!((-1..=1).contains(&uniform));
    assert!(experts >= 0);
    let mut descriptors = Vec::new();
    for tile in 0..(rows as usize).div_ceil(8) {
        let mut buckets: [Vec<(u32, Vec<u32>)>; 2] = [Vec::new(), Vec::new()];
        let start = tile * CAPACITY;
        let end = (start + CAPACITY).min(rows as usize * 10);
        for absolute in start..end {
            let id = topk[absolute];
            if id < 0 || id >= experts {
                continue;
            }
            let bucket = if uniform >= 0 {
                uniform as usize
            } else if codec.expect("valid mixed-bucket ID needs codec")[id as usize] == 1 {
                0
            } else {
                1
            };
            let groups = &mut buckets[bucket];
            if let Some((_, slots)) = groups.iter_mut().find(|(expert, _)| *expert == id as u32) {
                slots.push(absolute as u32);
            } else {
                groups.push((id as u32, vec![absolute as u32]));
            }
        }
        let mut descriptor = [0; WORDS];
        for (bucket, groups) in buckets.iter().enumerate() {
            let count: usize = groups.iter().map(|(_, slots)| slots.len()).sum();
            let offsets = 160 + bucket * 81;
            descriptor[offsets..offsets + 81].fill(count as u32);
            let mut position = 0;
            for (expert_index, (id, slots)) in groups.iter().enumerate() {
                descriptor[bucket * 80 + expert_index] = *id;
                descriptor[offsets + expert_index] = position as u32;
                let target = 322 + bucket * 80 + position;
                descriptor[target..target + slots.len()].copy_from_slice(slots);
                position += slots.len();
            }
        }
        descriptors.extend_from_slice(&descriptor);
    }
    descriptors
}

fn metadata(arch: Arch, emitted: &Emitted) {
    let text = &emitted.s_text;
    assert_eq!(
        planner::symbol(arch),
        format!("qwen4_slab_union_plan_t8_pm_{}", arch.name())
    );
    assert!(text.contains(&format!("{}:", planner::symbol(arch))));
    let yaml = &text[text.find("amdhsa.kernels:").expect("kernel metadata")..];
    let values = |key: &str| -> Vec<String> {
        yaml.lines()
            .filter_map(|line| {
                let line = line.trim();
                line.strip_prefix("- ")
                    .unwrap_or(line)
                    .strip_prefix(key)
                    .map(|s| s.trim().to_owned())
            })
            .collect()
    };
    assert_eq!(values(".offset:"), ["0", "8", "16", "24", "28", "32"]);
    assert_eq!(values(".size:"), ["8", "8", "8", "4", "4", "4"]);
    assert_eq!(
        values(".value_kind:"),
        [
            "global_buffer",
            "global_buffer",
            "global_buffer",
            "by_value",
            "by_value",
            "by_value"
        ]
    );
    assert_eq!(
        values(".actual_access:"),
        ["read_only", "read_only", "write_only"]
    );
    // The last logical field ends at36; the aligned segment is40, not36/48.
    for (name, offset, size) in [
        ("topk", 0, 8),
        ("codec", 8, 8),
        ("out", 16, 8),
        ("rows", 24, 4),
        ("uniform_bucket", 28, 4),
        ("n_experts", 32, 4),
    ] {
        assert!(
            yaml.contains(&format!(
                ".name: {name}\n        .offset: {offset}\n        .size: {size}\n"
            )),
            "{arch:?}: {name}"
        );
    }
    for required in [
        ".amdhsa_kernarg_size 40\n",
        ".kernarg_segment_size: 40\n",
        ".amdhsa_group_segment_fixed_size 1280\n",
        ".group_segment_fixed_size: 1280\n",
        ".max_flat_workgroup_size: 128\n",
        ".amdhsa_wavefront_size32 1\n",
        ".wavefront_size: 32\n",
        ".private_segment_fixed_size: 0\n",
        ".sgpr_spill_count: 0\n",
        ".vgpr_spill_count: 0\n",
    ] {
        assert!(text.contains(required), "{arch:?}: missing {required}");
    }
    assert_eq!(emitted.shape.next_free_vgpr, 24, "{arch:?}");
    // gfx1100: the 8-aligned kernarg block s[32:39] clears the workgroup id s15
    // (`FeatureUserSGPRInit16Bug`), so next_free_sgpr is 40.
    let next_free_sgpr = if arch == Arch::Gfx1100 { 40 } else { 27 };
    assert_eq!(emitted.shape.next_free_sgpr, next_free_sgpr, "{arch:?}");
    assert_eq!(emitted.proof.next_free_vgpr, 24);
    assert_eq!(emitted.proof.next_free_sgpr, next_free_sgpr);
    assert_eq!(emitted.shape.barriers, 2);
    assert_eq!(emitted.proof.barriers.len(), 2);
    let mnemonics: Vec<_> = text
        .lines()
        .filter_map(|line| line.split_whitespace().next())
        .collect();
    if arch == Arch::Gfx1201 {
        assert_eq!(
            mnemonics
                .iter()
                .filter(|m| **m == "s_barrier_signal")
                .count(),
            2
        );
        assert_eq!(
            mnemonics.iter().filter(|m| **m == "s_barrier_wait").count(),
            2
        );
        assert!(!mnemonics.contains(&"s_barrier"));
    } else {
        assert_eq!(mnemonics.iter().filter(|m| **m == "s_barrier").count(), 2);
        assert!(!mnemonics.contains(&"s_barrier_signal"));
        assert!(!mnemonics.contains(&"s_barrier_wait"));
    }
    for mnemonic in mnemonics {
        assert!(
            !mnemonic.starts_with("scratch_")
                && !mnemonic.starts_with("flat_")
                && !mnemonic.contains("atomic")
                && !mnemonic.contains("_f32")
                && !mnemonic.contains("_f16")
                && !mnemonic.contains("wmma"),
            "{arch:?}: {mnemonic}"
        );
    }
}

struct Case<'a> {
    name: &'a str,
    rows: i32,
    uniform: i32,
    experts: i32,
    // None means genuinely unmapped, not a readable dummy allocation.
    topk: Option<&'a [i32]>,
    codec: Option<&'a [u8]>,
}

fn bytes(memory: &Memory, name: &str) -> Vec<u8> {
    memory
        .regions
        .iter()
        .find(|r| r.name == name)
        .unwrap()
        .bytes
        .clone()
}

fn execute(emulator: &Emulator<'_>, arch: Arch, case: Case<'_>, workgroups: &[u32]) {
    let valid =
        (1..=64).contains(&case.rows) && (-1..=1).contains(&case.uniform) && case.experts >= 0;
    let mut expected = vec![POISON; GUARD + MAX_TILES * WORDS * 4 + GUARD];
    if valid {
        let words = oracle(
            case.topk.unwrap(),
            case.codec,
            case.rows,
            case.uniform,
            case.experts,
        );
        for (index, word) in words.iter().enumerate() {
            expected[GUARD + index * 4..GUARD + index * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
    }
    let mut memory = Memory::default();
    let topk_bytes = case.topk.map(|topk| {
        topk.iter()
            .flat_map(|id| id.to_le_bytes())
            .collect::<Vec<_>>()
    });
    if let Some(input) = &topk_bytes {
        memory.map("topk".into(), TOPK, input.clone()).unwrap();
    }
    if let Some(codec) = case.codec {
        memory.map("codec".into(), CODEC, codec.to_vec()).unwrap();
    }
    memory
        .map("output".into(), OUTPUT, vec![POISON; expected.len()])
        .unwrap();
    let mut kernarg = Vec::new();
    for pointer in [TOPK, CODEC, OUTPUT + GUARD as u64] {
        kernarg.extend_from_slice(&pointer.to_le_bytes());
    }
    for integer in [case.rows, case.uniform, case.experts] {
        kernarg.extend_from_slice(&integer.to_le_bytes());
    }
    assert_eq!(kernarg.len(), 36);
    kernarg.resize(40, 0xe7);
    memory
        .map("kernarg".into(), KERNARG, kernarg.clone())
        .unwrap();
    for &tile in workgroups {
        emulator
            .run_workgroup(
                &Launch {
                    kernarg_ptr: KERNARG,
                    workgroup: [tile, 0, 0],
                    block: [128, 1, 1],
                    dynamic_lds: 0,
                },
                &mut memory,
            )
            .unwrap_or_else(|error| {
                panic!(
                    "{arch:?} {} R={} uniform={} experts={} tile={tile}: {error}",
                    case.name, case.rows, case.uniform, case.experts
                )
            });
    }
    let actual = bytes(&memory, "output");
    if let Some(index) = actual.iter().zip(&expected).position(|(a, b)| a != b) {
        panic!("{arch:?} {} R={} uniform={} experts={}: output byte {index}, got {:02x}, expected {:02x} (descriptor word {:?})", case.name, case.rows, case.uniform, case.experts, actual[index], expected[index], index.checked_sub(GUARD).map(|offset| offset / 4));
    }
    if let Some(input) = topk_bytes {
        assert_eq!(bytes(&memory, "topk"), input, "{arch:?}: topk mutation");
    }
    if let Some(codec) = case.codec {
        assert_eq!(bytes(&memory, "codec"), codec, "{arch:?}: codec mutation");
    }
    assert_eq!(
        bytes(&memory, "kernarg"),
        kernarg,
        "{arch:?}: kernarg mutation"
    );
}

#[derive(Clone, Copy, Debug)]
enum Pattern {
    Identical,
    Disjoint,
    Skew,
    Invalid,
}

fn routed(rows: usize, pattern: Pattern) -> Vec<i32> {
    const IDENTICAL: [i32; 10] = [59, 2, 41, 0, 19, 7, 31, 11, 53, 23];
    (0..rows * 10)
        .map(|slot| {
            let token = slot / 10;
            let rank = slot % 10;
            match pattern {
                Pattern::Identical => IDENTICAL[rank],
                // Prime permutation: every routed expert differs across all64 tokens.
                Pattern::Disjoint => ((slot * 137 + 97) % 641) as i32,
                // Ten distinct IDs per token from a 13-expert pool. Its rotations
                // change first-occurrence order and create strongly unequal counts.
                Pattern::Skew => (((rank * 5 + token * 3) % 13) * 17 + 5) as i32,
                Pattern::Invalid => match slot % 9 {
                    0 => -1,
                    1 => 641,
                    2 => i32::MIN,
                    3 => i32::MAX,
                    4 => -2,
                    _ => ((rank * 7 + token * 3) % 31) as i32,
                },
            }
        })
        .collect()
}

fn exercise_arch(arch: Arch) {
    // Cache all expensive setup for this architecture across the entire matrix.
    // No per-case emitter, assembler, lift, or instruction semantic lookup.
    let emitted = planner::emit(arch).unwrap();
    metadata(arch, &emitted);
    let object = peacemaker_isa::native::assemble(&emitted.s_text, arch).unwrap();
    let program = peacemaker_lift::lift_object(
        &object,
        peacemaker_lift::Options {
            frontend: Frontend::Builder,
        },
    )
    .unwrap()
    .program;
    let emulator = Emulator::new(&program, &planner::symbol(arch)).unwrap();
    // Only byte1 means W3; other nonzero bytes must still mean bucket1.
    let codec: Vec<u8> = (0..641).map(|id| [1, 0, 2, 255][id % 4]).collect();
    for rows in 1usize..=64 {
        let tiles = rows.div_ceil(8);
        // Reverse order proves descriptors do not depend on preceding tiles.
        // The final dead workgroup must leave all existing output unchanged.
        let mut grid: Vec<u32> = (0..tiles as u32).rev().collect();
        grid.push(tiles as u32);
        for pattern in [
            Pattern::Identical,
            Pattern::Disjoint,
            Pattern::Skew,
            Pattern::Invalid,
        ] {
            let mut topk = routed(rows, pattern);
            if matches!(pattern, Pattern::Disjoint | Pattern::Invalid) {
                // Readable, valid expert IDs in every inactive tail slot. Any
                // accidental read/participation changes the descriptor bytes.
                topk.resize(tiles * 80 + 48, 640);
            }
            // Other patterns map exactly10R words: a tail load faults even if
            // the loaded value would subsequently be masked from the output.
            execute(
                &emulator,
                arch,
                Case {
                    name: &format!("mixed {pattern:?}"),
                    rows: rows as i32,
                    uniform: -1,
                    experts: 641,
                    topk: Some(&topk),
                    codec: Some(&codec),
                },
                &grid,
            );
        }
        for uniform in [0, 1] {
            let topk = routed(rows, Pattern::Disjoint);
            execute(
                &emulator,
                arch,
                Case {
                    name: "uniform with unmapped codec",
                    rows: rows as i32,
                    uniform,
                    experts: 641,
                    topk: Some(&topk),
                    codec: None,
                },
                &grid,
            );
        }
        for experts in [0, 641, i32::MAX] {
            let topk: Vec<i32> = (0..rows * 10)
                .map(|slot| [-1, experts, i32::MIN, i32::MAX][slot % 4])
                .collect();
            execute(
                &emulator,
                arch,
                Case {
                    name: "all invalid with unmapped codec",
                    rows: rows as i32,
                    uniform: -1,
                    experts,
                    topk: Some(&topk),
                    codec: None,
                },
                &grid,
            );
        }
    }
    // Signed upper boundary: MAX-1 is live when n_experts=MAX, while MAX is
    // invalid. Uniform routing permits this without allocating a huge table.
    for uniform in [0, 1] {
        let topk: Vec<i32> = (0..640)
            .map(|slot| [i32::MAX - 1, 0, i32::MAX, i32::MIN, -1][slot % 5])
            .collect();
        execute(
            &emulator,
            arch,
            Case {
                name: "signed expert boundary",
                rows: 64,
                uniform,
                experts: i32::MAX,
                topk: Some(&topk),
                codec: None,
            },
            &(0..8).collect::<Vec<_>>(),
        );
    }
    // All guards must precede input/codec access as well as output stores and
    // barriers. Both input pointers are deliberately unmapped in these cases.
    let mut invalid = Vec::new();
    for rows in [i32::MIN, -1, 0, 65, i32::MAX] {
        invalid.push((rows, -1, 641));
    }
    for uniform in [i32::MIN, -2, 2, i32::MAX] {
        invalid.push((64, uniform, 641));
    }
    for experts in [i32::MIN, -1] {
        invalid.push((64, -1, experts));
    }
    for (rows, uniform, experts) in invalid {
        execute(
            &emulator,
            arch,
            Case {
                name: "invalid domain writes nothing",
                rows,
                uniform,
                experts,
                topk: None,
                codec: None,
            },
            &[0, 1, 7, 8, u32::MAX],
        );
    }
    // Even huge dead tile indices must not wrap their row/output products.
    // No live launch here, so model this as an explicitly poisoned output.
    for tile in [8, 9, 1 << 29, 1 << 31, u32::MAX] {
        let mut memory = Memory::default();
        let output = vec![POISON; GUARD + MAX_TILES * WORDS * 4 + GUARD];
        memory.map("output".into(), OUTPUT, output.clone()).unwrap();
        let mut args = Vec::new();
        for pointer in [TOPK, CODEC, OUTPUT + GUARD as u64] {
            args.extend_from_slice(&pointer.to_le_bytes());
        }
        for integer in [64i32, -1, 641] {
            args.extend_from_slice(&integer.to_le_bytes());
        }
        args.resize(40, 0xe7);
        memory.map("kernarg".into(), KERNARG, args).unwrap();
        emulator
            .run_workgroup(
                &Launch {
                    kernarg_ptr: KERNARG,
                    workgroup: [tile, 0, 0],
                    block: [128, 1, 1],
                    dynamic_lds: 0,
                },
                &mut memory,
            )
            .unwrap_or_else(|error| panic!("{arch:?} dead tile {tile}: {error}"));
        assert_eq!(
            bytes(&memory, "output"),
            output,
            "{arch:?} dead tile {tile}"
        );
    }
}

#[test]
fn gfx1151_integer_emulator_matches_hip_union_semantics() {
    exercise_arch(Arch::Gfx1151);
}

#[test]
fn gfx1201_integer_emulator_matches_hip_union_semantics() {
    exercise_arch(Arch::Gfx1201);
}

#[test]
fn gfx1100_integer_emulator_matches_hip_union_semantics() {
    exercise_arch(Arch::Gfx1100);
}

/// Both eighty-wide scans must converge through the builder's hazard
/// fixed point: each recorded label needs at least two passes from the
/// entry/back-edge join.
#[test]
fn both_scans_reach_a_loop_fixpoint_on_all_arches() {
    for arch in [Arch::Gfx1151, Arch::Gfx1201, Arch::Gfx1100] {
        let emitted = planner::emit(arch).expect("slab union plan emits");
        let mut heads: Vec<u8> = emitted
            .proof
            .loop_fixpoints
            .iter()
            .filter(|fix| fix.head == ".Lunion_matches" || fix.head == ".Lunion_heads")
            .map(|fix| fix.iterations)
            .collect();
        heads.sort_unstable();
        assert_eq!(
            heads.len(),
            2,
            "{arch:?}: both scans must record a fixpoint"
        );
        assert!(
            heads.iter().all(|&passes| passes >= 2),
            "{arch:?}: {heads:?}"
        );
    }
}

/// Native assembly, lift, and the decoded M7 analyses must be
/// obligation-free on all three arches. Reads decoded obligations, not
/// source text, so a wait dropped by a reverted manual scan fails here.
#[test]
fn native_lift_has_no_m7_obligations_on_all_arches() {
    use peacemaker_ir::passes::{barriers, windows};
    for arch in [Arch::Gfx1151, Arch::Gfx1201, Arch::Gfx1100] {
        let emitted = planner::emit(arch).expect("slab union plan emits");
        let object = peacemaker_isa::native::assemble(&emitted.s_text, arch)
            .unwrap_or_else(|error| panic!("{arch:?}: native assemble: {error}"));
        let lifted = peacemaker_lift::lift_object(
            &object,
            peacemaker_lift::Options {
                frontend: Frontend::Builder,
            },
        )
        .unwrap_or_else(|error| panic!("{arch:?}: lift: {error}"));
        assert_eq!(
            peacemaker_lift::emit::module(&lifted.program).expect("re-emit"),
            object,
            "{arch:?}: lift must round-trip the linked object"
        );
        let kernel = lifted
            .program
            .kernels
            .iter()
            .find(|k| k.symbol.0 == planner::symbol(arch))
            .unwrap_or_else(|| panic!("{arch:?}: lifted kernel missing"));
        let analysis = peacemaker_ir::edit::analyze(lifted.program.clone(), &kernel.symbol)
            .unwrap_or_else(|error| panic!("{arch:?}: analyze: {error}"));
        let pairings = barriers::analyze(&kernel.body, lifted.program.target.arch)
            .unwrap_or_else(|error| panic!("{arch:?}: barriers: {error}"));
        let win = windows::check_windows(&kernel.body)
            .unwrap_or_else(|error| panic!("{arch:?}: windows: {error}"));
        let mut obligations: Vec<String> = analysis
            .obligations
            .iter()
            .chain(&pairings.obligations)
            .map(|o| format!("{:?}/{}: {}", o.kind, o.rule_id, o.text))
            .collect();
        if win.delays.iter().any(|d| d.is_ambiguous()) {
            obligations.push("ambiguous delay windows".into());
        }
        assert!(
            obligations.is_empty(),
            "{arch:?}: M7 obligations: {obligations:#?}"
        );
    }
}

#[test]
fn independent_oracle_has_first_occurrence_order_and_absolute_slots() {
    let mut topk = vec![-1; 90];
    for (slot, id) in [(79, 9), (80, 9), (81, 2), (82, 9), (83, 7), (84, 2)] {
        topk[slot] = id;
    }
    let mut codec = vec![0; 10];
    codec[9] = 1;
    codec[2] = 255;
    codec[7] = 2;
    let result = oracle(&topk, Some(&codec), 9, -1, 10);
    let mut expected = vec![0; 2 * WORDS];
    expected[0] = 9;
    expected[161..241].fill(1);
    expected[322] = 79;
    let second = &mut expected[WORDS..];
    second[0] = 9;
    second[80..82].copy_from_slice(&[2, 7]);
    second[161..241].fill(2);
    second[242] = 2;
    second[243..322].fill(3);
    second[322..324].copy_from_slice(&[80, 82]);
    second[402..405].copy_from_slice(&[81, 84, 83]);
    assert_eq!(result, expected);
    let routed = routed(64, Pattern::Disjoint);
    let full = oracle(&routed, None, 64, 1, 641);
    for tile in 0..8 {
        let descriptor = &full[tile * WORDS..(tile + 1) * WORDS];
        assert_eq!(
            &descriptor[402..482],
            &(tile as u32 * 80..(tile as u32 + 1) * 80).collect::<Vec<_>>()
        );
        assert_eq!(&descriptor[241..322], &(0u32..=80).collect::<Vec<_>>());
    }
    assert_eq!(full[8 * WORDS - 1], 639);
}
