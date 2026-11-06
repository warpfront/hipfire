// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx1201 M0 `EXPAND_LIVE` producer (`p0_expand_gfx1201::emit(block)`, o4 tid16 only; the dni tid18
//! expand stays static). The emitted module is assembled, lifted and run as one workgroup of `block`
//! threads on the PeaceMaker CPU interpreter; the whole live-list region (head canary, entries, unused
//! suffix, tail canary) and the count cell are compared byte for byte against an independent host
//! enumeration for blocks 32/64/128/256. Route arrays are mapped exactly sized (any out-of-range read
//! is an unmapped-access error); the real fixture is the j8284 seq19 route input (see
//! `tests/fixtures/p0_expand_live/SOURCE.json`). No GPU.
use peacemaker_ir::emu::{Emulator, Launch, Memory};
use peacemaker_isa::kernels::p0_expand_gfx1201 as ex;
use peacemaker_isa::Arch;
use sha2::{Digest, Sha256};
use std::sync::LazyLock;
#[path = "support/rocm.rs"]
mod rocm;

const BLOCKS: [u32; 4] = [32, 64, 128, 256];
const HEAD: usize = 64;
const TAIL: usize = 64;
const FIXTURE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/p0_expand_live");
const TILE_IDS_BIN: &[u8] = include_bytes!("fixtures/p0_expand_live/expert_tile_ids.i32le.bin");
const SLOTS_BIN: &[u8] = include_bytes!("fixtures/p0_expand_live/sorted_slot_index.i32le.bin");
const SOURCE_JSON: &str = include_str!("fixtures/p0_expand_live/SOURCE.json");

// ---- independent host enumeration ------------------------------------------------------------

#[derive(Clone)]
struct Case {
    name: String,
    tiles: Vec<i32>,
    slots: Vec<i32>,
    /// Extra live words appended past the logical end of `slots` (must never be read as pair members).
    slot_poison: usize,
    m: u32,
    rows: u32,
    n_experts: u32,
    /// Spare entries beyond the actual live-list length (0 = exact capacity).
    spare: u32,
    /// Entries by which capacity falls short of the actual count (kernel stores < capacity, publishes full count).
    under: u32,
}

struct Expected {
    /// `(vx, vy)` in y-ascending then x-ascending order.
    entries: Vec<(u32, u32)>,
    pairs: usize,
    mixed: usize,
}

/// Enumerates the live list straight from the contract text, without the grid extents.
fn enumerate(c: &Case) -> Expected {
    let (m, rows, e) = (c.m as u64, c.rows as u64, c.n_experts as i64);
    let mut entries = Vec::new();
    let (mut pairs, mut mixed) = (0, 0);
    let mut y = 0u64;
    while 2 * y < rows {
        let tile = c.tiles[((2 * y) >> 4) as usize];
        let a = c.slots[(2 * y) as usize] >= 0;
        let b = 2 * y + 1 < rows && c.slots[(2 * y + 1) as usize] >= 0;
        if tile >= 0 && (tile as i64) < e && (a || b) {
            pairs += 1;
            // "Mixed": a full pair with exactly one live row.
            if a != b && 2 * y + 1 < rows {
                mixed += 1;
            }
            let mut x = 0u64;
            while 16 * x < m {
                entries.push((x as u32, y as u32));
                x += 1;
            }
        }
        y += 1;
    }
    Expected {
        entries,
        pairs,
        mixed,
    }
}

fn words(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// Deterministic non-trivial background so an untouched byte can never equal a plausible store.
fn background(len: usize, salt: u8) -> Vec<u8> {
    (0..len)
        .map(|i| (i as u8).wrapping_mul(131).wrapping_add(7) ^ salt)
        .collect()
}

// ---- emission, assembly, lift -----------------------------------------------------------------

fn need_tools() {
    for t in ["llvm-mc", "ld.lld"] {
        if rocm::rocm_tool(t).is_none() {
            panic!("p0_expand_live needs the ROCm {t} CPU oracle tool; these tests never silently skip");
        }
    }
}

fn lift(block: u32) -> peacemaker_ir::Program {
    need_tools();
    let emitted = ex::emit(block).unwrap_or_else(|e| panic!("emit({block}): {e}"));
    let (text, _) = peacemaker_isa::kernels::iu4_gemm::module(
        &[emitted],
        &format!("p0_expand_gfx1201_b{block}"),
    )
    .unwrap_or_else(|e| panic!("module({block}): {e}"));
    let elf = peacemaker_isa::native::assemble(&text, Arch::Gfx1201)
        .unwrap_or_else(|e| panic!("assemble({block}): {e}"));
    peacemaker_lift::lift_object(
        &elf,
        peacemaker_lift::Options {
            frontend: peacemaker_ir::inst::Frontend::Builder,
        },
    )
    .unwrap_or_else(|e| panic!("lift({block}): {e}"))
    .program
}
static P32: LazyLock<peacemaker_ir::Program> = LazyLock::new(|| lift(32));
static P64: LazyLock<peacemaker_ir::Program> = LazyLock::new(|| lift(64));
static P128: LazyLock<peacemaker_ir::Program> = LazyLock::new(|| lift(128));
static P256: LazyLock<peacemaker_ir::Program> = LazyLock::new(|| lift(256));
fn program(block: u32) -> &'static peacemaker_ir::Program {
    match block {
        32 => &P32,
        64 => &P64,
        128 => &P128,
        256 => &P256,
        b => panic!("unsupported test block {b}"),
    }
}

// ---- one launch --------------------------------------------------------------------------------

fn first_diff(a: &[u8], b: &[u8]) -> Option<usize> {
    (0..a.len().max(b.len())).find(|&i| a.get(i) != b.get(i))
}

fn assert_bytes(what: &str, ctx: &str, got: &[u8], want: &[u8]) {
    if got != want {
        let i = first_diff(got, want).unwrap();
        let lo = i & !15;
        panic!(
            "{ctx}: {what} differs at byte {i} (len got {} want {}): got {:02x?} want {:02x?}",
            got.len(),
            want.len(),
            &got[lo.min(got.len())..(lo + 32).min(got.len())],
            &want[lo.min(want.len())..(lo + 32).min(want.len())]
        );
    }
}

/// Runs the case at `block`; returns the full list-region bytes and the count region bytes after
/// asserting both equal the independent enumeration (entries, unused suffix, canaries, count) and
/// that every input region is unchanged.
fn run(c: &Case, block: u32) -> (Vec<u8>, Vec<u8>) {
    let ctx = format!(
        "{} block {block} (M {} rows {} E {} spare {})",
        c.name, c.m, c.rows, c.n_experts, c.spare
    );
    let want = enumerate(c);
    let count = want.entries.len() as u32;
    let gx = c.m.div_ceil(16);
    let gy = c.rows.div_ceil(2);
    assert!(c.under <= count, "{ctx}: under exceeds count");
    let capacity = count + c.spare - c.under;
    assert!(
        capacity as u64 <= gx as u64 * gy as u64,
        "{ctx}: harness capacity {capacity} exceeds gx*gy {}",
        gx as u64 * gy as u64
    );
    assert_eq!(
        c.tiles.len(),
        (c.rows as usize).div_ceil(16),
        "{ctx}: tiles not exact"
    );
    assert_eq!(c.slots.len(), c.rows as usize, "{ctx}: slots not exact");

    let tiles_b = words(&c.tiles);
    let mut slots_vec = c.slots.clone();
    slots_vec.extend(std::iter::repeat_n(7, c.slot_poison));
    let slots_b = words(&slots_vec);
    let list_init = background(HEAD + capacity as usize * 16 + TAIL, 0x5A);
    let count_init: Vec<u8> = {
        let mut v = background(48, 0xC3);
        // Host-zeroed count cell: the release atomic add publishes onto 0 (guard bytes stay).
        v[16..20].copy_from_slice(&0u32.to_le_bytes());
        v
    };

    let mut mem = Memory::default();
    let guard = 0x2000;
    let kp = mem
        .map_guarded("kernarg".into(), vec![0; ex::KERNARG_BYTES as usize], guard)
        .unwrap();
    let tp = mem
        .map_guarded("tiles".into(), tiles_b.clone(), guard)
        .unwrap();
    let sp = mem
        .map_guarded("slots".into(), slots_b.clone(), guard)
        .unwrap();
    let lp = mem
        .map_guarded("list".into(), list_init.clone(), guard)
        .unwrap();
    let cp = mem
        .map_guarded("count".into(), count_init.clone(), guard)
        .unwrap();
    let mut kern = Vec::new();
    for p in [tp, sp, lp + HEAD as u64, cp + 16] {
        kern.extend(p.to_le_bytes());
    }
    for x in [c.m, c.rows, c.n_experts, gx, gy, capacity, 0, 0] {
        kern.extend(x.to_le_bytes());
    }
    assert_eq!(kern.len(), ex::KERNARG_BYTES as usize);
    let kernarg_image = kern.clone();
    mem.regions[0].bytes = kern;

    let emu = Emulator::new(program(block), ex::SYMBOL)
        .unwrap_or_else(|e| panic!("{ctx}: emulator: {e}"));
    emu.run_workgroup(
        &Launch {
            kernarg_ptr: kp,
            workgroup: [0, 0, 0],
            block: [block, 1, 1],
            dynamic_lds: 0,
        },
        &mut mem,
    )
    .unwrap_or_else(|e| panic!("{ctx}: {e}"));

    let region = |name: &str| {
        mem.regions
            .iter()
            .find(|r| r.name == name)
            .unwrap()
            .bytes
            .clone()
    };
    let mut want_list = list_init.clone();
    // Undercapacity: only indices < capacity are stored; the published count stays the full actual count.
    for (i, &(vx, vy)) in want.entries.iter().enumerate().take(capacity as usize) {
        let o = HEAD + i * 16;
        for (k, w) in [vx, vy, 0, 0].into_iter().enumerate() {
            want_list[o + 4 * k..o + 4 * k + 4].copy_from_slice(&w.to_le_bytes());
        }
    }
    let mut want_count = count_init.clone();
    want_count[16..20].copy_from_slice(&count.to_le_bytes());
    let (list, cnt) = (region("list"), region("count"));
    assert_bytes("count region", &ctx, &cnt, &want_count);
    assert_bytes("list region", &ctx, &list, &want_list);
    assert_bytes("kernarg", &ctx, &region("kernarg"), &kernarg_image);
    assert_bytes("expert_tile_ids", &ctx, &region("tiles"), &tiles_b);
    assert_bytes("sorted_slot_index", &ctx, &region("slots"), &slots_b);
    (list, cnt)
}

/// All four blocks must equal the enumeration, hence each other (checked explicitly too).
fn check_all_blocks(c: &Case) {
    let outs: Vec<_> = BLOCKS.iter().map(|&b| run(c, b)).collect();
    for (b, o) in BLOCKS.iter().zip(&outs).skip(1) {
        assert!(
            *o == outs[0],
            "{}: block {b} bytes differ from block 32",
            c.name
        );
    }
}

fn check_blocks(c: &Case, blocks: &[u32]) {
    for &b in blocks {
        run(c, b);
    }
}

// ---- case builders -----------------------------------------------------------------------------

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 24
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn case(name: &str, tiles: Vec<i32>, slots: Vec<i32>, m: u32, e: u32, spare: u32) -> Case {
    Case {
        name: name.into(),
        rows: slots.len() as u32,
        tiles,
        slots,
        slot_poison: 0,
        m,
        n_experts: e,
        spare,
        under: 0,
    }
}

fn tiles_for(rows: usize, f: impl Fn(usize) -> i32) -> Vec<i32> {
    (0..rows.div_ceil(16)).map(f).collect()
}

fn le_i32s(b: &[u8]) -> Vec<i32> {
    b.chunks(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn fixture_case(name: &str, e: u32, spare: u32) -> Case {
    case(
        name,
        le_i32s(TILE_IDS_BIN),
        le_i32s(SLOTS_BIN),
        1280,
        e,
        spare,
    )
}

// ---- tests ---------------------------------------------------------------------------------------

#[test]
fn constants_and_deterministic_emission() {
    assert_eq!(ex::TEMPLATE_ID, 1);
    assert_eq!(ex::SYMBOL, "pm_expand_live_gfx1201");
    assert_eq!(ex::KERNARG_BYTES, 64);
    assert_eq!(ex::SGPRS, 40);
    assert_eq!(ex::VGPRS, 8);
    assert_eq!(ex::LDS_BYTES, 0);
    for b in BLOCKS {
        let (x, y) = (ex::emit(b).unwrap(), ex::emit(b).unwrap());
        assert_eq!(x.s_text, y.s_text, "block {b} emission not deterministic");
        assert_eq!(x.proof.s_text_sha256, y.proof.s_text_sha256, "block {b}");
        assert!(x.s_text.contains(&format!("{}:", ex::SYMBOL)), "block {b}");
        // The count pointer's kernarg metadata is read-write (atomic RMW reads and writes).
        let spec = ex::kernel_spec(b);
        let access = |name: &str| {
            spec.kernargs
                .args
                .iter()
                .find(|a| a.name == name)
                .and_then(|a| a.actual_access.clone())
        };
        assert_eq!(
            access("count_cell").as_deref(),
            Some(peacemaker_isa::plan::Access::ReadWrite.name()),
            "block {b}: count_cell access"
        );
        assert_eq!(
            access("live_list").as_deref(),
            Some(peacemaker_isa::plan::Access::WriteOnly.name()),
            "block {b}: live_list access"
        );
        // Count publication is a release atomic add (EventCells), never a plain b32 store.
        let insns = &x.program.instructions;
        let atomics: Vec<usize> = (0..insns.len())
            .filter(|&i| insns[i].text.starts_with("global_atomic_add_u32"))
            .collect();
        assert_eq!(atomics.len(), 1, "block {b}: one count atomic add");
        assert!(
            !x.s_text.contains("global_store_b32"),
            "block {b}: plain count store"
        );
        assert!(
            x.s_text.contains("global_store_b128"),
            "block {b}: LiveList store"
        );
        assert!(
            x.proof.syncs.iter().any(|s| s.pc_index == atomics[0]
                && s.kind == peacemaker_isa::sync::SyncKind::ReleaseSignal
                && s.scope == peacemaker_isa::sync::Scope::Dev),
            "block {b}: count atomic is not a device-scope release signal"
        );
    }
    let texts: Vec<_> = BLOCKS
        .iter()
        .map(|&b| ex::emit(b).unwrap().s_text)
        .collect();
    for i in 0..texts.len() {
        for j in i + 1..texts.len() {
            assert_ne!(
                texts[i], texts[j],
                "blocks {} and {} share text",
                BLOCKS[i], BLOCKS[j]
            );
        }
    }
}

/// The committed arrays are exactly the cited byte ranges of the cited dump files.
#[test]
fn fixture_provenance_is_self_consistent() {
    let sha = |b: &[u8]| -> String {
        Sha256::digest(b)
            .iter()
            .map(|x| format!("{x:02x}"))
            .collect()
    };
    let v: serde_json::Value = serde_json::from_str(SOURCE_JSON).unwrap();
    assert_eq!(v["job"], "j8284");
    assert_eq!(v["seq19_kernarg"]["M"], 1280);
    assert_eq!(v["seq19_kernarg"]["grouped_rows"], 960);
    assert_eq!(v["seq19_kernarg"]["n_experts"], 512);
    assert_eq!(v["manifest"]["entries"], 155);
    let arrays = v["arrays"].as_array().unwrap();
    for (name, bytes, count) in [
        ("expert_tile_ids.i32le.bin", TILE_IDS_BIN, 60u64),
        ("sorted_slot_index.i32le.bin", SLOTS_BIN, 960),
    ] {
        let a = arrays
            .iter()
            .find(|a| a["fixture"] == name)
            .unwrap_or_else(|| panic!("{name} missing from SOURCE.json"));
        assert_eq!(a["fixture_sha256"], sha(bytes), "{name}");
        assert_eq!(a["i32_count"], count, "{name}");
        assert_eq!(bytes.len() as u64, count * 4, "{name}");
        assert_eq!(a["byte_range"][0], 0);
        assert_eq!(a["byte_range"][1], count * 4);
        assert_eq!(a["manifest_sha256"], a["source_sha256"], "{name}");
        // The dump may be absent on other hosts; when present it must hash and slice identically.
        let dump = format!(
            "{}/{}",
            v["dump_root"].as_str().unwrap().trim_end_matches('/'),
            a["source_file"].as_str().unwrap()
        );
        if let Ok(src) = std::fs::read(&dump) {
            assert_eq!(sha(&src), a["source_sha256"].as_str().unwrap(), "{dump}");
            assert_eq!(&src[..bytes.len()], bytes, "{name} != {dump} prefix");
        }
    }
    assert!(std::path::Path::new(FIXTURE).is_dir());
    // Auditable shape of the real route: 45 live pairs, 30 of them mixed live/pad.
    let e = enumerate(&fixture_case("real", 512, 0));
    assert_eq!((e.pairs, e.mixed, e.entries.len()), (45, 30, 45 * 80));
}

#[test]
fn real_j8284_route_all_blocks_exact_capacity() {
    check_all_blocks(&fixture_case("j8284 exact cap", 512, 0));
}

#[test]
fn real_j8284_route_spare_capacity_leaves_suffix_untouched() {
    check_blocks(&fixture_case("j8284 spare 17", 512, 17), &[32, 256]);
}

/// The real tile ids top out at expert 508: `E == 509` keeps it, `E == 508` (E-1 boundary) drops it.
#[test]
fn real_j8284_expert_bound_boundaries_change_the_list() {
    let base = enumerate(&fixture_case("e", 512, 0)).entries.len();
    let keep = enumerate(&fixture_case("e", 509, 0)).entries.len();
    let drop = enumerate(&fixture_case("e", 508, 0)).entries.len();
    assert_eq!(base, keep);
    assert!(drop < keep, "fixture lost its boundary expert");
    for e in [509, 508, 7, 1, 0] {
        check_blocks(&fixture_case(&format!("j8284 E={e}"), e, 0), &[32, 256]);
    }
}

#[test]
fn zero_live_leaves_whole_list_and_writes_zero_count() {
    let rows = 70;
    let dead_tiles = case(
        "zero live tiles",
        tiles_for(rows, |_| -1),
        vec![3; rows],
        48,
        8,
        5,
    );
    let dead_slots = case(
        "zero live slots",
        tiles_for(rows, |_| 0),
        vec![-1; rows],
        48,
        8,
        5,
    );
    let no_experts = case(
        "zero experts",
        tiles_for(rows, |_| 0),
        vec![0; rows],
        48,
        0,
        5,
    );
    for c in [&dead_tiles, &dead_slots, &no_experts] {
        assert_eq!(enumerate(c).entries.len(), 0);
        check_all_blocks(c);
    }
}

#[test]
fn all_pairs_live_fill_exact_capacity() {
    // 3 x 35 = 105 entries (list fills the whole grid).
    let c = case("all live", tiles_for(70, |_| 0), vec![0; 70], 48, 1, 0);
    assert_eq!(enumerate(&c).entries.len(), 3 * 35);
    check_all_blocks(&c);
    // The real j8284 geometry, every pair live: M 1280 x rows 960, E 512 -> 80 x 480 = 38400 entries.
    let full = case(
        "all live j8284 geometry",
        tiles_for(960, |_| 0),
        (0..960).collect(),
        1280,
        512,
        0,
    );
    assert_eq!(enumerate(&full).entries.len(), 80 * 480);
    check_all_blocks(&full);
    // Row width beyond every block: M 5000 -> gx 313 > 256 threads, one live pair.
    let wide = case(
        "all live gx313",
        tiles_for(2, |_| 0),
        vec![0; 2],
        5000,
        1,
        0,
    );
    assert_eq!(enumerate(&wide).entries.len(), 313);
    check_all_blocks(&wide);
    // Mid-size: 80 x 32 = 2560 entries (gx 80 exceeds blocks 32 only).
    let big = case(
        "all live 80x32",
        tiles_for(64, |_| 0),
        (0..64).collect(),
        1280,
        1,
        0,
    );
    assert_eq!(enumerate(&big).entries.len(), 80 * 32);
    check_blocks(&big, &[32, 128, 256]);
}

#[test]
fn expert_bound_and_negative_sentinels() {
    // One tile per 16 rows: E-1, E, E+1, -1, i32::MIN, i32::MAX, 0, a mid value, ...
    let e = 37;
    let ids = [
        e - 1,
        e,
        e + 1,
        -1,
        i32::MIN,
        i32::MAX,
        0,
        e / 2,
        -2,
        1 << 30,
        e - 1,
        e,
    ];
    let rows = ids.len() * 16;
    let c = case(
        "bounds",
        ids.to_vec(),
        (0..rows as i32).collect(),
        33,
        e as u32,
        3,
    );
    // Live tiles: E-1, 0, E/2, E-1 = 4 tiles x 8 pairs.
    assert_eq!(enumerate(&c).pairs, 4 * 8);
    check_all_blocks(&c);
    // Slot sentinels with a live tile: negative values dead, positive extremes live.
    let pattern = [-1, i32::MIN, -2, i32::MAX, 0, -1, 5, i32::MIN + 1];
    let mut full: Vec<i32> = pattern.iter().cycle().take(32).copied().collect();
    full[10] = -1;
    full[11] = -1;
    let c = case("slot sentinels", tiles_for(32, |_| 2), full, 17, 5, 0);
    check_all_blocks(&c);
}

#[test]
fn odd_grouped_rows_partial_pair() {
    for rows in [1usize, 3, 15, 17, 33, 47] {
        // The unpaired last row live, and dead (with live poison words one past `rows`).
        let mut slots = vec![-1; rows];
        slots[rows - 1] = 4;
        let live = case(
            "odd last live",
            tiles_for(rows, |_| 1),
            slots.clone(),
            20,
            4,
            0,
        );
        assert_eq!(enumerate(&live).pairs, 1);
        check_blocks(&live, &[32, 64, 256]);
        slots[rows - 1] = -1;
        let mut dead = case("odd last dead", tiles_for(rows, |_| 1), slots, 20, 4, 2);
        dead.slot_poison = 3;
        assert_eq!(enumerate(&dead).entries.len(), 0);
        check_blocks(&dead, &[32, 128]);
    }
    // Odd rows with every row live: gy = ceil(rows/2), the last pair partial but live.
    let all = case("odd all live", tiles_for(33, |_| 0), vec![0; 33], 16, 1, 0);
    assert_eq!(enumerate(&all).entries.len(), 17);
    check_all_blocks(&all);
}

#[test]
fn mixed_live_and_pad_pairs_are_emitted() {
    let rows = 64;
    // Pairs cycle (live,pad) / (pad,live) / (pad,pad) / (live,live).
    let slots: Vec<i32> = (0..rows)
        .map(|r| match (r / 2) % 4 {
            0 => {
                if r % 2 == 0 {
                    1
                } else {
                    -1
                }
            }
            1 => {
                if r % 2 == 0 {
                    -1
                } else {
                    1
                }
            }
            2 => -1,
            _ => 9,
        })
        .collect();
    let c = case("mixed", tiles_for(rows, |_| 3), slots, 40, 4, 0);
    let e = enumerate(&c);
    assert_eq!(e.pairs, 24);
    assert_eq!(e.mixed, 16);
    check_all_blocks(&c);
}

#[test]
fn random_routes_fixed_prng() {
    let mut rng = Rng(0x00C0_FFEE_5EED_1201);
    for n in 0..24usize {
        let rows = 1 + rng.below(200) as usize;
        let m = 1 + rng.below(100) as u32;
        let e = 1 + rng.below(40) as u32;
        let dead_pct = [0u64, 10, 50, 90, 100][rng.below(5) as usize];
        let tiles: Vec<i32> = (0..rows.div_ceil(16))
            .map(|_| match rng.below(8) {
                0 => -1,
                1 => i32::MIN,
                2 => i32::MAX,
                3 => e as i32,
                4 => e as i32 - 1,
                _ => rng.below(e as u64) as i32,
            })
            .collect();
        let slots: Vec<i32> = (0..rows)
            .map(|_| {
                if rng.below(100) < dead_pct {
                    -(1 + rng.below(1 << 20) as i32)
                } else {
                    rng.below(1 << 20) as i32
                }
            })
            .collect();
        let mut c = case(&format!("random #{n}"), tiles, slots, m, e, 0);
        let grid = m.div_ceil(16) as usize * rows.div_ceil(2);
        let room = (grid - enumerate(&c).entries.len()) as u32;
        c.spare = [0, 0, 1, 9][rng.below(4) as usize].min(room);
        check_blocks(&c, &[BLOCKS[n % 4], BLOCKS[(n + 1) % 4]]);
    }
}

/// The list region carries 64-byte canaries on both sides; with `spare = 0` the last entry abuts the
/// tail canary (an off-by-one entry write is caught there), and the count region carries canaries too.
#[test]
fn exact_cap_and_canary_writes() {
    // 60 rows: the first 40 live (20 pairs x gx 7 = 140 entries), the last 20 dead; grid 7 x 30 = 210
    // leaves 70 spare slots, so spare 1 and 31 are valid capacities.
    for spare in [0u32, 1, 31] {
        let c = case(
            "exact cap",
            tiles_for(60, |t| t as i32),
            (0..60).map(|r| if r < 40 { r } else { -1 }).collect(),
            100,
            4,
            spare,
        );
        assert_eq!(enumerate(&c).entries.len(), 20 * 7);
        check_blocks(&c, &[32, 64, 128, 256]);
    }
}

/// Zero-extent grids (the host never launches them; the contract formulas still give an empty list).
#[test]
fn zero_m_or_zero_rows_write_zero_count() {
    let zero_rows = case("zero rows", vec![], vec![], 32, 4, 0);
    let zero_m = case("zero M", tiles_for(32, |_| 0), vec![0; 32], 0, 4, 0);
    for c in [&zero_rows, &zero_m] {
        assert_eq!(enumerate(c).entries.len(), 0);
        check_blocks(c, &[32, 256]);
    }
}

/// Undercapacity: the kernel stores only entries with index < capacity, never writes past it (tail canary),
/// and still publishes the full actual count so the consumer can detect the overflow.
#[test]
fn undercapacity_stores_prefix_and_publishes_full_count() {
    for (under, blocks) in [(1u32, &BLOCKS[..]), (7, &[32, 256][..])] {
        let mut c = fixture_case("j8284 undercap", 512, 0);
        c.under = under;
        check_blocks(&c, blocks);
    }
    // Capacity zero with a live list: nothing stored, count still published.
    let mut c = case(
        "undercap zero cap",
        tiles_for(70, |_| 0),
        vec![0; 70],
        48,
        1,
        0,
    );
    c.under = 105;
    check_blocks(&c, &[32, 64, 128, 256]);
    // Capacity ending mid-wave and mid-block.
    let mut c = case(
        "undercap mid",
        tiles_for(64, |_| 0),
        (0..64).collect(),
        1280,
        1,
        0,
    );
    c.under = 2560 - 37;
    check_blocks(&c, &[32, 128, 256]);
}
