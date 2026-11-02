// SPDX-License-Identifier: Apache-2.0
//! Raw PM `ep_tile_mask_pm_gfx1100` G1 GPU runner; never participates in runtime dispatch.
//! Build: cargo build -p hipfire-runtime --release --features lab --example ep_tile_mask_gfx1100_g1
//! (Cargo.toml entry gated on `lab`; no registry or dispatch wiring: standalone qualification only).
//!   ep_tile_mask_gfx1100_g1 --codeobject /abs/ep_tile_mask_pm_gfx1100.hxaco [--expect-uuid GPU-..] [--expect-bdf 0000:..] [--seed N]
//! Loads the bundle with raw HIP, launches `ep_tile_mask_pm_gfx1100` (grid [ceil(tiles/256),1,1], block [256,1,1], 36-byte
//! kernarg: tile_ids, tile_ids_owned, owner_table pointers; tiles, n_experts, rank i32) over seeded random inputs, and compares
//! the device output word for word against the CPU law
//!   live(t)  = 0 <= tile_ids[t] < n_experts
//!   owned[t] = live(t) && owner_table[tile_ids[t]] == rank ? tile_ids[t] : -1
//! Inputs cover: tile counts that are not multiples of 256 (1, 2, 3, 255, 257, 511, 513, 1000, 4099) and exact multiples
//! (256, 512, 1024), grouped-run ids with a -1 padding tail, uniformly random ids spanning negatives and ids past the table
//! (including i32::MIN / i32::MAX), owner tables of random ranks in {-1, 0, 1, 2, 3} (-1 = the sentinel no rank holds), and
//! ranks 2 and 0 (the gfx1100 slab codec ranks) plus 1 and 3. Every launch also checks that the source ids are unmodified and
//! that the owned buffer's guard tail past `tiles` words is untouched (0x5A fill).
//! Negative controls (each MUST produce >= 1 mismatching word over the whole run, else the run fails): (1) CPU reference
//! with the wrong rank; (2) CPU reference with a perturbed owner table; (3) CPU reference with no liveness check (negative
//! and out-of-table ids clamped into the table); (4) the device output with one bit flipped in one word per launch.
//! No timing is measured. It prints one JSON receipt line with the device identity (uuid, bdf, arch).
use hip_bridge::HipRuntime;
use serde_json::{json, Value};

type Result<T, E = String> = std::result::Result<T, E>;
fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}
const SCHEMA: &str = "ep-tile-mask-gfx1100-g1-runner-v1";
const ARCH: &str = "gfx1100";
const SYMBOL: &str = "ep_tile_mask_pm_gfx1100";
const BLOCK: usize = 256;
const KERNARG: usize = 36;
const TAIL_WORDS: usize = 16;
const GUARD: u8 = 0x5A;
const USAGE: &str = "usage: ep_tile_mask_gfx1100_g1 --codeobject PATH [--expect-uuid UUID] [--expect-bdf BDF] [--seed N]";

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 24
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn reference(ids: &[i32], owner: &[i32], rank: i32) -> Vec<i32> {
    ids.iter()
        .map(|&id| {
            let live = id >= 0 && (id as usize) < owner.len();
            if live && owner[id as usize] == rank {
                id
            } else {
                -1
            }
        })
        .collect()
}
/// Negative control 3: no liveness check (an id outside the table is clamped into it).
fn reference_no_liveness(ids: &[i32], owner: &[i32], rank: i32) -> Vec<i32> {
    ids.iter()
        .map(|&id| {
            if owner[(id as u32 as usize) % owner.len()] == rank {
                id
            } else {
                -1
            }
        })
        .collect()
}

fn words(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|w| w.to_le_bytes()).collect()
}
fn from_words(b: &[u8]) -> Vec<i32> {
    b.chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}
fn diff(a: &[i32], b: &[i32]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count() + a.len().abs_diff(b.len())
}

fn make_ids(rng: &mut Rng, tiles: usize, experts: usize, kind: usize) -> Vec<i32> {
    match kind {
        // Ascending expert runs then the -1 tail, like the stable grouping.
        0 => {
            let live = tiles - rng.below(tiles / 3 + 1);
            let mut ids = Vec::with_capacity(tiles);
            let mut e = rng.below(experts.min(8).max(1)) as i32;
            while ids.len() < live {
                let run = 1 + rng.below(8);
                for _ in 0..run.min(live - ids.len()) {
                    ids.push(e);
                }
                e += 1 + rng.below(3) as i32;
                if e as usize >= experts {
                    e = rng.below(experts.min(8).max(1)) as i32;
                }
            }
            ids.resize(tiles, -1);
            ids
        }
        // Uniform over [-3, experts + 4], so negatives and ids past the table appear.
        1 => (0..tiles)
            .map(|_| rng.below(experts + 8) as i32 - 3)
            .collect(),
        // Mixed with extreme values.
        _ => {
            const ODD: [i32; 9] = [
                -1,
                -2,
                i32::MIN,
                i32::MAX,
                512,
                513,
                1 << 20,
                0x4000_0000,
                511,
            ];
            (0..tiles)
                .map(|_| {
                    if rng.below(3) == 0 {
                        ODD[rng.below(ODD.len())]
                    } else {
                        rng.below(experts) as i32
                    }
                })
                .collect()
        }
    }
}

fn run() -> Result<()> {
    let (mut codeobject, mut expect_uuid, mut expect_bdf, mut seed) =
        (None, None, None, 0x9E37_79B9_7F4A_7C15u64);
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let value = args.next().ok_or(USAGE)?;
        match flag.as_str() {
            "--codeobject" => codeobject = Some(value),
            "--expect-uuid" => expect_uuid = Some(value),
            "--expect-bdf" => expect_bdf = Some(value.to_ascii_lowercase()),
            "--seed" => {
                seed = value
                    .parse()
                    .map_err(|_| format!("--seed: not a u64: {value}"))?
            }
            _ => return Err(USAGE.into()),
        }
    }
    let codeobject = codeobject.ok_or(USAGE)?;
    let hip = HipRuntime::load().map_err(err)?;
    let id = hip.current_device().map_err(err)?;
    let uuid = hip.device_uuid(id).map_err(err)?;
    let bdf = hip.device_pci_bus_id(id).map_err(err)?.to_ascii_lowercase();
    let arch = hip.get_arch(id).map_err(err)?;
    let identity_match = arch == ARCH
        && expect_uuid.as_deref().is_none_or(|u| u == uuid)
        && expect_bdf.as_deref().is_none_or(|b| b == bdf);
    let device = json!({"uuid": uuid, "bdf": bdf, "arch": arch, "expected_arch": ARCH, "expected_uuid": expect_uuid, "expected_bdf": expect_bdf, "identity_match": identity_match});
    if !identity_match {
        println!(
            "{}",
            json!({"schema": SCHEMA, "device": device, "error": "ep_tile_mask gfx1100 device identity mismatch"})
        );
        return Err("ep_tile_mask gfx1100 device identity mismatch".into());
    }
    let image = std::fs::read(&codeobject).map_err(err)?;
    let module = hip.module_load_data(&image).map_err(err)?;
    let f = hip.module_get_function(&module, SYMBOL).map_err(err)?;

    let mut rng = Rng(seed);
    let (mut launches, mut mismatching_words, mut words_checked, mut guard_or_source_bad) =
        (0usize, 0usize, 0usize, 0usize);
    let (mut ctl_rank, mut ctl_table, mut ctl_live, mut ctl_flip) =
        (0usize, 0usize, 0usize, 0usize);
    let (mut live_taken, mut live_not_taken, mut dead) = (0usize, 0usize, 0usize);
    let mut cases = Vec::new();
    for &tiles in &[1usize, 2, 3, 255, 256, 257, 511, 512, 513, 1000, 1024, 4099] {
        for &experts in &[1usize, 7, 512] {
            for kind in 0..3 {
                let ids = make_ids(&mut rng, tiles, experts, kind);
                // Owner values in {-1, 0, 1, 2, 3}: -1 is the sentinel no rank holds.
                let owner: Vec<i32> = (0..experts).map(|_| rng.below(5) as i32 - 1).collect();
                for &rank in &[2i32, 0, 1, 3] {
                    let want = reference(&ids, &owner, rank);
                    let d_ids = hip.malloc(tiles * 4).map_err(err)?;
                    let d_owned = hip.malloc((tiles + TAIL_WORDS) * 4).map_err(err)?;
                    let d_table = hip.malloc(experts * 4).map_err(err)?;
                    hip.memcpy_htod(&d_ids, &words(&ids)).map_err(err)?;
                    hip.memcpy_htod(&d_table, &words(&owner)).map_err(err)?;
                    hip.memset(&d_owned, GUARD.into(), (tiles + TAIL_WORDS) * 4)
                        .map_err(err)?;
                    let mut kernarg = [0u8; KERNARG];
                    for (off, ptr) in [
                        (0, d_ids.as_ptr()),
                        (8, d_owned.as_ptr()),
                        (16, d_table.as_ptr()),
                    ] {
                        kernarg[off..off + 8].copy_from_slice(&(ptr as u64).to_le_bytes());
                    }
                    kernarg[24..28].copy_from_slice(&(tiles as i32).to_le_bytes());
                    kernarg[28..32].copy_from_slice(&(experts as i32).to_le_bytes());
                    kernarg[32..36].copy_from_slice(&rank.to_le_bytes());
                    let grid = [tiles.div_ceil(BLOCK) as u32, 1, 1];
                    // SAFETY: exact 36-byte ABI; every allocation stays live through the synchronize and read-backs below.
                    unsafe {
                        hip.launch_kernel_blob(
                            &f,
                            grid,
                            [BLOCK as u32, 1, 1],
                            0,
                            None,
                            &mut kernarg,
                        )
                    }
                    .map_err(err)?;
                    hip.device_synchronize().map_err(err)?;
                    let mut owned_b = vec![0u8; (tiles + TAIL_WORDS) * 4];
                    let mut ids_b = vec![0u8; tiles * 4];
                    hip.memcpy_dtoh(&mut owned_b, &d_owned).map_err(err)?;
                    hip.memcpy_dtoh(&mut ids_b, &d_ids).map_err(err)?;
                    let got = from_words(&owned_b[..tiles * 4]);
                    let bad = diff(&got, &want);
                    let tail_bad = owned_b[tiles * 4..].iter().filter(|&&b| b != GUARD).count();
                    let src_bad = usize::from(ids_b != words(&ids));
                    launches += 1;
                    words_checked += tiles;
                    mismatching_words += bad;
                    guard_or_source_bad += tail_bad + src_bad;
                    live_taken += want.iter().filter(|&&w| w >= 0).count();
                    live_not_taken += ids
                        .iter()
                        .zip(&want)
                        .filter(|(&i, &w)| i >= 0 && (i as usize) < experts && w < 0)
                        .count();
                    dead += ids
                        .iter()
                        .filter(|&&i| i < 0 || (i as usize) >= experts)
                        .count();
                    // Negative controls: each must differ from the (correct) device output somewhere over the run.
                    ctl_rank += diff(&got, &reference(&ids, &owner, rank + 1));
                    let mut perturbed = owner.clone();
                    for o in perturbed.iter_mut() {
                        *o = (*o + 1).min(3);
                    }
                    ctl_table += diff(&got, &reference(&ids, &perturbed, rank));
                    ctl_live += diff(&got, &reference_no_liveness(&ids, &owner, rank));
                    let mut flipped = got.clone();
                    flipped[rng.below(tiles)] ^= 1 << rng.below(31);
                    ctl_flip += diff(&flipped, &want);
                    if kind == 0 && experts == 512 {
                        cases.push(json!({"tiles": tiles, "experts": experts, "rank": rank, "mismatch": bad}));
                    }
                    for b in [d_ids, d_owned, d_table] {
                        hip.free(b).map_err(err)?;
                    }
                }
            }
        }
    }
    let controls = json!({"wrong_rank": ctl_rank, "perturbed_owner_table": ctl_table, "no_liveness_check": ctl_live, "device_bit_flip": ctl_flip});
    let controls_ok = ctl_rank > 0 && ctl_table > 0 && ctl_live > 0 && ctl_flip > 0;
    let pass = mismatching_words == 0 && guard_or_source_bad == 0 && controls_ok;
    println!(
        "{}",
        json!({
            "schema": SCHEMA, "symbol": SYMBOL, "codeobject": codeobject, "codeobject_bytes": image.len(), "device": device,
            "seed": seed, "kernarg_bytes": KERNARG, "ranks": [2, 0, 1, 3], "launches": launches, "words_checked": words_checked,
            "mismatching_words": mismatching_words, "guard_tail_or_source_violations": guard_or_source_bad,
            "coverage": {"owned_tiles": live_taken, "live_tiles_other_owner": live_not_taken, "dead_ids": dead},
            "negative_controls_mismatching_words": controls, "negative_controls_ok": controls_ok,
            "timing": Value::Null, "pass": pass,
        })
    );
    if pass {
        Ok(())
    } else {
        Err("ep_tile_mask gfx1100 G1 FAILED".into())
    }
}
fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
