// SPDX-License-Identifier: Apache-2.0
//! Scenario definitions: routes (tiles / slots), padding positions, expert recipes (synthetic or real files),
//! activation sets. Experts are built lazily per (codec, kind) so memory stays bounded.

use crate::model::*;
use hipfire_runtime::mq_i4_reference::A4Block;
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Class {
    Finite,
    Exceptional,
}

#[derive(Clone)]
pub struct RealRef {
    pub layer: u64,
    pub expert: u64,
    /// Codec of the tensor bytes (the sample's quant type decides it, not the run's `--codec`).
    pub codec: Codec,
    pub path: PathBuf,
    pub sha256: String,
    pub bytes: usize,
    pub artifact_offset: Option<u64>,
}

#[derive(Clone)]
pub enum Recipe {
    Synth(Gen, u32),
    Real(RealRef),
}

#[derive(Clone)]
pub struct Scenario {
    pub name: String,
    pub class: Class,
    pub div: usize,
    pub rows: usize,
    pub tiles: Vec<Tile>,
    pub recipes: Vec<Recipe>,
    pub x0: X0,
    pub x_seed: u32,
    /// Member of the negative-control subset.
    pub neg: bool,
    pub production: bool,
    pub real: bool,
    /// GateUp scenario whose CPU grouped BF16 H feeds a twin-chained Down scenario.
    pub feeds_chain: bool,
    /// Down scenario whose A4 input is the prepared chain (CPU GateUp H -> external K3 helper -> chain-a4.json); overrides `x_set`.
    pub chain_x: Option<XSet>,
    /// Experts of more than one codec in one tile set (per-codec tile masks, shared output).
    pub mixed: bool,
}

impl Scenario {
    pub fn build_experts(&self, codec: Codec, kind: Kind) -> Res<Vec<Expert>> {
        self.recipes
            .iter()
            .map(|r| match r {
                Recipe::Synth(g, seed) => gen_expert(codec, kind, *g, *seed),
                Recipe::Real(rr) => {
                    let bytes = std::fs::read(&rr.path).map_err(|e| format!("{}: {e}", rr.path.display()))?;
                    if bytes.len() != rr.bytes {
                        return Err(format!("{}: {} bytes, manifest says {}", rr.path.display(), bytes.len(), rr.bytes));
                    }
                    let sha = sha256_hex(&bytes);
                    if sha != rr.sha256 {
                        return Err(format!("{}: sha256 {sha} != manifest {}", rr.path.display(), rr.sha256));
                    }
                    let ex = Expert::from_bytes(rr.codec, kind, bytes, format!("real:L{}.E{}:{}:{}", rr.layer, rr.expert, rr.codec.name(), rr.path.display()))?;
                    if ex.guard_trips != 0 {
                        return Err(format!("{}: {} E8P scales trip the writer guard (a real E8P tensor must carry none: such groups fall back to MQ4)", rr.path.display(), ex.guard_trips));
                    }
                    Ok(ex)
                }
            })
            .collect()
    }

    pub fn x_set(&self, kind: Kind) -> XSet {
        match &self.chain_x {
            Some(x) => x.clone(),
            None => make_x(kind.groups(), self.rows, self.x_seed, &self.x0),
        }
    }

    pub fn referenced(&self) -> Vec<bool> {
        referenced_rows(&self.tiles, self.div, self.rows)
    }

    pub fn recipe_json(&self) -> serde_json::Value {
        serde_json::Value::Array(
            self.recipes
                .iter()
                .map(|r| match r {
                    Recipe::Synth(g, s) => serde_json::json!({"synthetic": format!("{g:?}"), "seed": s}),
                    Recipe::Real(rr) => serde_json::json!({"real": {"layer": rr.layer, "expert": rr.expert, "codec": rr.codec.name(), "file": rr.path.display().to_string(),
                        "sha256": rr.sha256, "bytes": rr.bytes, "artifact_offset": rr.artifact_offset}}),
                })
                .collect(),
        )
    }
}

fn sentinel_tile() -> Tile {
    (-1, [0; 16])
}

fn base(
    name: &str,
    div: usize,
    rows: usize,
    tiles: Vec<Tile>,
    recipes: Vec<Recipe>,
    seed: u32,
) -> Scenario {
    Scenario {
        name: name.to_string(),
        class: Class::Finite,
        div,
        rows,
        tiles,
        recipes,
        x0: X0::None,
        x_seed: seed ^ 0x51ed,
        neg: false,
        production: false,
        real: false,
        feeds_chain: false,
        chain_x: None,
        mixed: false,
    }
}

fn recipes(seed: u32, gens: &[Gen]) -> Vec<Recipe> {
    gens.iter()
        .enumerate()
        .map(|(i, g)| {
            Recipe::Synth(
                *g,
                seed.wrapping_mul(2654435761).wrapping_add(i as u32 * 7919),
            )
        })
        .collect()
}

pub const ROW_COUNTS: [usize; 8] = [1, 2, 7, 8, 15, 16, 17, 33];

pub fn synthetic(seed: u32) -> Vec<Scenario> {
    let mut out = Vec::new();
    let three = [Gen::Random, Gen::EveryCode, Gen::Extreme];
    // Rows 1/2/7/8/15/16/17/33 at x_row_div 1 and 10, one sentinel tile each.
    for (div, rows_per) in [(1usize, 20usize), (10, 4)] {
        for n in ROW_COUNTS {
            let slots: Vec<i32> = (0..n)
                .map(|i| ((i * 7 + 3) % (rows_per * div)) as i32)
                .collect();
            let e = (n % 3) as i32;
            let mut tiles = tiles_for(e, slots);
            tiles.push(sentinel_tile());
            let mut sc = base(
                &format!("rows{n}_div{div}"),
                div,
                rows_per,
                tiles,
                recipes(seed, &three),
                seed,
            );
            sc.neg = (n == 17 && div == 1) || (n == 33 && div == 10);
            out.push(sc);
        }
    }
    // Production boundary / tail: 1535, 1536, 1537 slots, experts interleaved by tile, x_row_div 1 and 10.
    for n in [1535usize, 1536, 1537] {
        for div in [1usize, 10] {
            let rows = n.div_ceil(div);
            let slots: Vec<i32> = (0..n as i32).collect();
            let mut tiles: Vec<Tile> = slots
                .chunks(16)
                .enumerate()
                .map(|(ci, c)| tile((ci % 3) as i32, c))
                .collect();
            tiles.push(sentinel_tile());
            let mut sc = base(
                &format!("production{n}_div{div}"),
                div,
                rows,
                tiles,
                recipes(seed, &[Gen::Random, Gen::Extreme, Gen::EveryCode]),
                seed,
            );
            sc.production = true;
            out.push(sc);
        }
    }
    // Mixed / repeated experts, first and last expert, non-monotone slot order, partial padding, mid-list sentinel.
    {
        let mut tiles = tiles_for(3, vec![5, 3, 19, 0, 1, 2, 11, 12]);
        tiles.extend(tiles_for(0, (0..17).map(|i| (i * 3 % 20) as i32).collect()));
        tiles.push(sentinel_tile());
        tiles.extend(tiles_for(1, vec![7]));
        tiles.push(tile(0, &[-1, 4, -1, 9]));
        tiles.extend(tiles_for(3, vec![2, 2, 6]));
        tiles.push(sentinel_tile());
        let mut sc = base(
            "mixed_experts_unsorted_pad",
            1,
            20,
            tiles,
            recipes(
                seed,
                &[Gen::Random, Gen::Extreme, Gen::EveryCode, Gen::Random],
            ),
            seed,
        );
        sc.neg = true;
        out.push(sc);
    }
    // Padding at every one of the 16 tile positions, plus a fully padded live tile.
    {
        let mut tiles = Vec::new();
        for k in 0..16usize {
            let slots: Vec<i32> = (0..16usize)
                .map(|j| {
                    if j == k {
                        -1
                    } else {
                        ((j * 3 + k) % 20) as i32
                    }
                })
                .collect();
            tiles.push(tile((k % 3) as i32, &slots));
        }
        tiles.push(tile(0, &[-1; 16]));
        let mut sc = base(
            "pad_every_position",
            1,
            20,
            tiles,
            recipes(seed, &three),
            seed,
        );
        sc.neg = true;
        out.push(sc);
    }
    // Exactly one live slot at every tile position (all other lanes padding).
    {
        let mut tiles = Vec::new();
        for k in 0..16usize {
            let mut slots = [-1i32; 16];
            slots[k] = ((k * 5 + 1) % 40) as i32;
            tiles.push(tile(((k + 1) % 3) as i32, &slots));
        }
        out.push(base(
            "single_live_each_position",
            10,
            4,
            tiles,
            recipes(seed, &three),
            seed,
        ));
    }
    // Scale-plane sentinels (bytes the last-row chunk's unaligned dword read can spill into).
    for (tag, first) in [("sentinel_ff", 0x3cffu16), ("sentinel_00", 0x3c00u16)] {
        let tiles = tiles_for(1, (0..16).collect());
        out.push(base(
            &format!("{tag}_last_row_chunk"),
            1,
            16,
            tiles,
            recipes(
                seed,
                &[Gen::Sentinel(first), Gen::Sentinel(first), Gen::Random],
            ),
            seed,
        ));
    }
    // Extreme / maximum-magnitude dots and f16 max / subnormal / zero scales.
    {
        let mut tiles = tiles_for(0, (0..16).collect());
        tiles.extend(tiles_for(1, (0..16).collect()));
        tiles.extend(tiles_for(2, (0..16).collect()));
        out.push(base(
            "maxdot_extreme_scales",
            1,
            20,
            tiles,
            recipes(seed, &[Gen::MaxDot, Gen::Extreme, Gen::Random]),
            seed,
        ));
    }
    // Targeted dot values: BF16 ties on both parities, signed-zero products, row-distinguishing dots.
    for (tag, var) in [
        ("ties", Var::Ties),
        ("signed_zero", Var::SignedZero),
        ("rowid", Var::RowId),
    ] {
        let mut tiles = tiles_for(0, (0..16).collect());
        tiles.extend(tiles_for(0, vec![3, 12, 25, 0, -1, 9, 31, 8]));
        tiles.extend(tiles_for(1, (0..10).collect()));
        tiles.push(sentinel_tile());
        let mut sc = base(
            &format!("targeted_{tag}"),
            10,
            4,
            tiles,
            recipes(seed, &[Gen::Targeted(var), Gen::Random, Gen::Extreme]),
            seed,
        );
        sc.x0 = X0::Pattern { d0: 1.0 };
        sc.neg = var == Var::Ties;
        out.push(sc);
    }
    // Exceptional class: finite max / overflow boundary and full-payload inf/NaN activation scales.
    {
        let mut tiles = tiles_for(0, (0..8).collect());
        tiles.push(tile(1, &[0, 1, 2, 3, 4, 5, 6]));
        tiles.push(tile(2, &[6, 5, 4, 3, 2, 1, 0, 7]));
        tiles.push(sentinel_tile());
        let mut sc = base(
            "exceptional_inf_nan_scales",
            1,
            8,
            tiles,
            recipes(
                seed,
                &[Gen::Targeted(Var::Overflow), Gen::Random, Gen::Extreme],
            ),
            seed,
        );
        sc.x0 = X0::Exceptional;
        sc.class = Class::Exceptional;
        out.push(sc);
        let mut tiles = tiles_for(0, (0..16).collect());
        tiles.push(sentinel_tile());
        let mut sc = base(
            "exceptional_flt_max_overflow",
            10,
            4,
            tiles,
            recipes(
                seed,
                &[Gen::Targeted(Var::Overflow), Gen::Random, Gen::Extreme],
            ),
            seed,
        );
        sc.x0 = X0::Pattern {
            d0: f32::from_bits(FLT_MAX_BITS),
        };
        sc.class = Class::Exceptional;
        out.push(sc);
    }
    out
}

/// Real-weight scenarios over the real expert files of one (codec, kind); activations and routes are synthetic.
pub fn real(refs: &[RealRef], seed: u32) -> Vec<Scenario> {
    let mut out = Vec::new();
    let rr = |i: usize| Recipe::Real(refs[i].clone());
    let mk = |name: String, div: usize, rows: usize, tiles: Vec<Tile>, recipes: Vec<Recipe>| {
        let mut s = base(&name, div, rows, tiles, recipes, seed);
        s.real = true;
        s
    };
    // groups of 4 experts: one full tile each, a partial padded tile, a repeated expert, a sentinel tile
    for (gi, chunk) in (0..refs.len()).collect::<Vec<_>>().chunks(4).enumerate() {
        let mut tiles = Vec::new();
        for (i, _) in chunk.iter().enumerate() {
            let slots: Vec<i32> = (0..16usize)
                .map(|j| ((i * 5 + j * 3 + gi) % 160) as i32)
                .collect();
            tiles.push(tile(i as i32, &slots));
        }
        tiles.push(tile((chunk.len() - 1) as i32, &[-1, 7, -1, 100, 33, -1]));
        tiles.push(sentinel_tile());
        tiles.push(tile(0, &[159, 0, 80, 81, 82]));
        let mut s = mk(
            format!(
                "real_group{gi}_L{}E{}..L{}E{}",
                refs[chunk[0]].layer,
                refs[chunk[0]].expert,
                refs[*chunk.last().unwrap()].layer,
                refs[*chunk.last().unwrap()].expert
            ),
            10,
            16,
            tiles,
            chunk.iter().map(|&i| rr(i)).collect(),
        );
        s.feeds_chain = true;
        // multi-expert, padded, repeated-expert routes: every mutation (incl. the expert pointer) is detectable on real weights
        s.neg = gi < 2;
        out.push(s);
    }
    // row counts on the first and the last real expert
    for (tag, idx) in [("first", 0usize), ("last", refs.len() - 1)] {
        for (div, rows_per) in [(1usize, 20usize), (10, 4)] {
            for n in ROW_COUNTS {
                let slots: Vec<i32> = (0..n)
                    .map(|i| ((i * 7 + 3) % (rows_per * div)) as i32)
                    .collect();
                let mut tiles = tiles_for(0, slots);
                tiles.push(sentinel_tile());
                let mut s = mk(
                    format!("real_rows{n}_div{div}_{tag}"),
                    div,
                    rows_per,
                    tiles,
                    vec![rr(idx)],
                );
                s.feeds_chain = matches!(n, 1 | 16 | 17 | 33);
                s.neg = tag == "first" && ((n == 17 && div == 1) || (n == 33 && div == 10));
                out.push(s);
            }
        }
    }
    // production-chunk boundary with first / middle / last real experts interleaved
    if !refs.is_empty() {
        let n = 1536usize;
        let slots: Vec<i32> = (0..n as i32).collect();
        let mut tiles: Vec<Tile> = slots
            .chunks(16)
            .enumerate()
            .map(|(ci, c)| tile((ci % 3) as i32, c))
            .collect();
        tiles.push(sentinel_tile());
        let mid = refs.len() / 2;
        let mut s = mk(
            "real_production1536_div10".to_string(),
            10,
            n.div_ceil(10),
            tiles,
            vec![rr(0), rr(mid), rr(refs.len() - 1)],
        );
        s.production = true;
        out.push(s);
    }
    out
}

/// Hot/cold codec-mixture scenarios: experts of two codecs (`cold`: W2 or W3, `hot`: MQ4) share one global expert-ID
/// space, one tile list, one slot array and one poisoned output; each codec's kernel runs with its own tile mask.
pub fn mixed(cold: &[RealRef], hot: &[RealRef], seed: u32) -> Vec<Scenario> {
    let n = 4usize.min(cold.len()).min(hot.len());
    let c = |i: usize| Recipe::Real(cold[i].clone());
    let h = |i: usize| Recipe::Real(hot[i].clone());
    let mk = |name: &str, div: usize, rows: usize, tiles: Vec<Tile>, recipes: Vec<Recipe>| {
        let mut s = base(name, div, rows, tiles, recipes, seed);
        s.real = true;
        s.mixed = true;
        s
    };
    let slots16 = |e: usize, gi: usize| -> Vec<i32> {
        (0..16usize)
            .map(|j| ((e * 5 + j * 3 + gi) % 160) as i32)
            .collect()
    };
    let mut out = Vec::new();
    // interleaved cold/hot ids: 0 cold, 1 hot, 2 cold, ...; every expert once, expert 1 partial/padded, first and last repeated
    {
        let ne = 2 * n;
        let recipes: Vec<Recipe> = (0..n).flat_map(|i| [c(i), h(i)]).collect();
        let mut tiles: Vec<Tile> = (0..ne).map(|e| tile(e as i32, &slots16(e, 0))).collect();
        tiles.push(tile(1, &[-1, 7, -1, 100, 33, -1]));
        tiles.push(sentinel_tile());
        tiles.push(tile(0, &[159, 0, 80, 81, 82]));
        tiles.push(tile((ne - 1) as i32, &[5, 3, 19, 0, 1, 2, 11, 12]));
        let mut s = mk("mixed_hotcold_interleaved", 10, 16, tiles, recipes);
        s.feeds_chain = true;
        out.push(s);
    }
    // cold ids first, hot ids last (first expert cold, last expert hot), tiles alternating codec
    {
        let ne = 2 * n;
        let mut recipes: Vec<Recipe> = (0..n).map(c).collect();
        recipes.extend((0..n).map(h));
        let mut tiles = Vec::new();
        for i in 0..n {
            tiles.push(tile(i as i32, &slots16(i, 1)));
            tiles.push(tile((n + i) as i32, &slots16(n + i, 1)));
        }
        tiles.push(sentinel_tile());
        tiles.push(tile(0, &[-1, 9, -1, 4]));
        tiles.push(tile((ne - 1) as i32, &[3, 12, 25, 0, -1, 9, 31, 8]));
        let mut s = mk("mixed_cold_first_hot_last", 10, 16, tiles, recipes);
        s.feeds_chain = true;
        out.push(s);
    }
    // hot ids first, cold ids last
    {
        let ne = 2 * n;
        let mut recipes: Vec<Recipe> = (0..n).map(h).collect();
        recipes.extend((0..n).map(c));
        let mut tiles = Vec::new();
        for i in 0..n {
            tiles.push(tile((n + i) as i32, &slots16(n + i, 2)));
            tiles.push(tile(i as i32, &slots16(i, 2)));
        }
        tiles.push(tile(0, &[-1, 9, -1, 4]));
        tiles.push(sentinel_tile());
        tiles.push(tile((ne - 1) as i32, &[3, 12, 25, 0, -1, 9, 31, 8]));
        let mut s = mk("mixed_hot_first_cold_last", 10, 16, tiles, recipes);
        s.feeds_chain = true;
        out.push(s);
    }
    // padding at every tile position over alternating codecs (cold, hot, cold)
    {
        let recipes = vec![c(0), h(0), c(1usize.min(cold.len() - 1))];
        let mut tiles = Vec::new();
        for k in 0..16usize {
            let slots: Vec<i32> = (0..16usize)
                .map(|j| {
                    if j == k {
                        -1
                    } else {
                        ((j * 3 + k) % 20) as i32
                    }
                })
                .collect();
            tiles.push(tile((k % 3) as i32, &slots));
        }
        tiles.push(tile(0, &[-1; 16]));
        let mut s = mk("mixed_pad_every_position", 1, 20, tiles, recipes);
        s.feeds_chain = true;
        out.push(s);
    }
    // row counts over alternating codec tiles
    for (div, rows_per) in [(1usize, 20usize), (10, 4)] {
        for rc in ROW_COUNTS {
            let slots: Vec<i32> = (0..rc)
                .map(|i| ((i * 7 + 3) % (rows_per * div)) as i32)
                .collect();
            let mut tiles: Vec<Tile> = slots
                .chunks(16)
                .enumerate()
                .map(|(ti, ch)| tile((ti % 2) as i32, ch))
                .collect();
            tiles.push(sentinel_tile());
            let mut s = mk(
                &format!("mixed_rows{rc}_div{div}"),
                div,
                rows_per,
                tiles,
                vec![c(0), h(0)],
            );
            s.feeds_chain = (rc == 17 && div == 1) || (rc == 33 && div == 10);
            out.push(s);
        }
    }
    // production boundary with all four codec-interleaved experts
    {
        let slots: Vec<i32> = (0..1536i32).collect();
        let ne = 2 * n;
        let recipes: Vec<Recipe> = (0..n).flat_map(|i| [c(i), h(i)]).collect();
        let mut tiles: Vec<Tile> = slots
            .chunks(16)
            .enumerate()
            .map(|(ti, ch)| tile((ti % ne) as i32, ch))
            .collect();
        tiles.push(sentinel_tile());
        let mut s = mk("mixed_production1536_div10", 10, 154, tiles, recipes);
        s.production = true;
        out.push(s);
    }
    out
}

/// Down scenario whose A4 input is the prepared chain: the GateUp scenario's grouped BF16 H (CPU twin, qualified hooks,
/// SAME real expert weights, never the candidate) -> external K3 producer (the Python helper's pilot_solve
/// `a4_emulate` over the R128 rotation, loaded from `chain-a4.json`) -> `tok`, token-major A4 rows indexed by grouped
/// row `p` (non-live rows are zero blocks and are poisoned by `referenced()`).
/// Down slots are `p` for live lanes and -1 for padding, `x_row_div` is 1, expert/tile order is preserved.
/// `Ok(None)` when an expert has no Down tensor in the sample set.
pub fn chain_down(
    gate: &Scenario,
    downs: &HashMap<(u64, u64, Codec), RealRef>,
    tok: Vec<A4Block>,
) -> Res<Option<Scenario>> {
    let mut recipes = Vec::new();
    for r in &gate.recipes {
        match r {
            Recipe::Real(rr) => match downs.get(&(rr.layer, rr.expert, rr.codec)) {
                Some(d) => recipes.push(Recipe::Real(d.clone())),
                None => return Ok(None),
            },
            Recipe::Synth(..) => return Err("chain_down needs real-weight recipes".into()),
        }
    }
    let m_total = gate.tiles.len() * 16;
    if tok.len() != m_total * 5 {
        return Err(format!(
            "chain_down: {} A4 blocks for {m_total} grouped rows x 5 groups",
            tok.len()
        ));
    }
    let tiles: Vec<Tile> = gate
        .tiles
        .iter()
        .enumerate()
        .map(|(ti, t)| {
            if t.0 < 0 {
                (-1, [0; 16])
            } else {
                let mut s = [-1i32; 16];
                for lane in 0..16 {
                    if t.1[lane] >= 0 {
                        s[lane] = (ti * 16 + lane) as i32;
                    }
                }
                (t.0, s)
            }
        })
        .collect();
    let mut s = base(
        &format!("chain_down__{}", gate.name),
        1,
        m_total,
        tiles,
        recipes,
        0,
    );
    s.real = true;
    s.mixed = gate.mixed;
    s.chain_x = Some(XSet {
        tok,
        rows: m_total,
        groups: 5,
    });
    Ok(Some(s))
}
