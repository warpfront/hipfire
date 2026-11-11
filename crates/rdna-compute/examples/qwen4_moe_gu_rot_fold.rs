// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Wide-verify gate/up A8 GEMM with the down producer folded into its epilogue: word ORACLE +
//! timing bench (exact gfx1201; developer example).
//!
//! `qwen4_moe_ep_a8_gate_up_rot_planar_nt{1,2}` (one launch, `HIPFIRE_QWEN4_MOE_GLUE_FOLD=c`,
//! `Gpu::gemm_qwen4_moe_gate_up_silu_a8_rot_sym` + `Gpu::gemm_qwen4_moe_down_a8_rot_sym`) against the
//! chain it replaces (`gemm_qwen4_moe_gate_up_silu_a8_sym` -> `qwen4_moe_rotate128_i8` ->
//! `gemm_qwen4_moe_down_a8_sym`), at the served C8 shapes: gate/up M 1280, K 2560, down M 2560, K 640,
//! planar experts (128 distinct seeded weight blobs behind a 512-entry pointer table, finite f16
//! scales incl. zeros), `x_row_div` 10, the real gate producer (`qwen4_moe_rotate256_i8`).
//!
//! `oracle [NT]`: tokens 1..=64 x routing {distinct, hot-expert piles, all tokens on the same ten
//! experts, holey (unrouted slots and dead rows inside tiles)} x tile ids {owner-masked mod2 rank 0,
//! mod2 rank 1, mod5 rank 1, unmasked, every expert foreign} (+ a served-bound variant with the
//! `slots * 16` grouped rows at 9/16/24/32/48/64 tokens: mostly sentinel tiles); `NT` (1 or 2) pins
//! the tile-run width (`HIPFIRE_QWEN4_MOE_EP_A8_NT`). Per cell: every 136-byte down-sidecar record
//! of the slots of owned live tiles must equal the standalone `qwen4_moe_rotate128_i8` record (word
//! for word); every other record of the fused arm (and the buffer tail) must still hold the dirty
//! canary fill; the shared gate sidecar must be byte-unchanged by the fused launch; and the final
//! down output (the whole allocation, canary tail included, NaN-filled before) of the SAME down
//! GEMM run on each chain must be equal. Controls (must be detected): one gate-sidecar byte of a
//! live owned token +1 in the fused arm only, and one `signs2` entry flipped in the fused arm only.
//!
//! `bench [NT]`: per layer at 16/24/32/48/64 tokens (served grouped bound), median of 200,
//! 48-layer back-to-back loops including launch gaps: [rotate256 + gate/up + rotate128] vs
//! [rotate256 + gate/up(rot)] and the full chain with the down GEMM.
//!
//! Usage: qwen4_moe_gu_rot_fold oracle|bench [1|2]

use rdna_compute::{DType, Gpu, GpuTensor, Int8MmqPrepared};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

const N_EXP: usize = 512;
const HIDDEN: usize = 2560;
const MI: usize = 640;
const M_GU: usize = 2 * MI;
const TOP: usize = 10;
const BLOBS: usize = 128;
const GUARD: usize = 512;
const REC: usize = 136;
const GROUPS: usize = MI / 128;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn normalish(&mut self) -> f32 {
        // sum of four uniforms, centred: bell-shaped, unit-ish variance
        (self.unit() + self.unit() + self.unit() + self.unit() - 2.0) * 1.7
    }
}

fn check_arch(gpu: &Gpu) -> Res<()> {
    if !gpu.arch_caps.is_gfx1201() {
        return Err("requires exact gfx1201".into());
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------------------
// Weights: planar MQ4G256V2P / MQ4G128V2P (nibble rows, then the f16 scale plane)
// ---------------------------------------------------------------------------------------------------------

fn rand_scale(r: &mut Rng) -> u16 {
    let s = (r.next() & 1) as u16;
    if r.below(16) == 0 {
        return s << 15; // +-0 scale
    }
    let e = 6 + r.below(7) as u16; // 2^-9 .. 2^-3
    let m = r.below(1024) as u16;
    (s << 15) | (e << 10) | m
}

fn planar_blob(r: &mut Rng, rows: usize, k: usize) -> Vec<u8> {
    let nib = rows * (k / 2);
    let mut out = Vec::with_capacity(nib + rows * (k / 128) * 2);
    while out.len() + 8 <= nib {
        out.extend_from_slice(&r.next().to_le_bytes());
    }
    while out.len() < nib {
        out.push(r.next() as u8);
    }
    for _ in 0..rows * (k / 128) {
        out.extend_from_slice(&rand_scale(r).to_le_bytes());
    }
    out
}

#[allow(dead_code)] // the data tensors own the memory the pointer tables point into
struct Weights {
    gu_data: GpuTensor,
    dn_data: GpuTensor,
    gu_ptrs: GpuTensor,
    dn_ptrs: GpuTensor,
}

fn build_weights(gpu: &mut Gpu, seed: u64) -> Res<Weights> {
    let mut r = Rng(seed);
    let mut gu = Vec::new();
    let mut dn = Vec::new();
    let (mut gu_per, mut dn_per) = (0usize, 0usize);
    for _ in 0..BLOBS {
        let g = planar_blob(&mut r, M_GU, HIDDEN);
        let d = planar_blob(&mut r, HIDDEN, MI);
        gu_per = g.len();
        dn_per = d.len();
        gu.extend(g);
        dn.extend(d);
    }
    let gu_data = gpu.upload_raw(&gu, &[gu.len()])?;
    let dn_data = gpu.upload_raw(&dn, &[dn.len()])?;
    let table = |base: u64, per: usize| -> Vec<u8> {
        (0..N_EXP)
            .flat_map(|e| (base + (((e * 37 + e / 64) % BLOBS) * per) as u64).to_le_bytes())
            .collect()
    };
    let gu_t = table(gu_data.buf.as_ptr() as usize as u64, gu_per);
    let dn_t = table(dn_data.buf.as_ptr() as usize as u64, dn_per);
    let gu_ptrs = gpu.upload_raw(&gu_t, &[gu_t.len()])?;
    let dn_ptrs = gpu.upload_raw(&dn_t, &[dn_t.len()])?;
    Ok(Weights {
        gu_data,
        dn_data,
        gu_ptrs,
        dn_ptrs,
    })
}

// ---------------------------------------------------------------------------------------------------------
// Routing / layout
// ---------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Debug)]
enum Routing {
    Distinct,
    Hot,
    Same,
    Holey,
}
const ROUTINGS: [Routing; 4] = [
    Routing::Distinct,
    Routing::Hot,
    Routing::Same,
    Routing::Holey,
];

fn route_ids(kind: Routing, tokens: usize, r: &mut Rng) -> Vec<i32> {
    let mut ids = vec![-1i32; tokens * TOP];
    const HOT: [i32; 6] = [3, 100, 101, 257, 300, 511];
    for t in 0..tokens {
        let mut picked: Vec<i32> = Vec::new();
        match kind {
            Routing::Same => {
                picked.extend(0..TOP as i32);
            }
            Routing::Hot => {
                while picked.len() < 4 {
                    let e = HOT[r.below(HOT.len())];
                    if !picked.contains(&e) {
                        picked.push(e);
                    }
                }
            }
            _ => {}
        }
        while picked.len() < TOP {
            let e = r.below(N_EXP) as i32;
            if !picked.contains(&e) {
                picked.push(e);
            }
        }
        for i in (1..TOP).rev() {
            picked.swap(i, r.below(i + 1));
        }
        for (k, e) in picked.iter().enumerate() {
            ids[t * TOP + k] = *e;
        }
    }
    if kind == Routing::Holey {
        for s in (4..ids.len()).step_by(9) {
            ids[s] = -1; // unrouted: never in `sorted`
        }
    }
    ids
}

#[derive(Clone)]
struct Layout {
    tiles: Vec<i32>,
    sorted: Vec<i32>,
    g: usize,
}

/// Scatter contract: per expert ascending, its flat slots ascending, padded to 16; then sentinel tiles
/// (id -1, sorted -1) up to `bound_rows` (at least `min_trailing` of them).
fn build_layout(ids: &[i32], bound_rows: Option<usize>, min_trailing: usize) -> Layout {
    let mut per: Vec<Vec<i32>> = vec![Vec::new(); N_EXP];
    for (s, &e) in ids.iter().enumerate() {
        if e >= 0 {
            per[e as usize].push(s as i32);
        }
    }
    let mut tiles = Vec::new();
    let mut sorted = Vec::new();
    for (e, v) in per.iter().enumerate() {
        for chunk in v.chunks(16) {
            tiles.push(e as i32);
            sorted.extend_from_slice(chunk);
            sorted.extend(std::iter::repeat(-1).take(16 - chunk.len()));
        }
    }
    let used = tiles.len();
    let want = bound_rows.map_or(used + min_trailing, |b| (b / 16).max(used + 1));
    while tiles.len() < want {
        tiles.push(-1);
        sorted.extend(std::iter::repeat(-1).take(16));
    }
    Layout {
        g: tiles.len() * 16,
        tiles,
        sorted,
    }
}

/// Dead rows inside run tiles (holey routing): every 11th live row.
fn punch_dead_rows(lay: &mut Layout) {
    let mut n = 0usize;
    for s in lay.sorted.iter_mut() {
        if *s >= 0 {
            n += 1;
            if n % 11 == 0 {
                *s = -1;
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Owner {
    Mod2R0,
    Mod2R1,
    Mod5R1,
    Unmasked,
    AllForeign,
}
const OWNERS: [Owner; 5] = [
    Owner::Mod2R0,
    Owner::Mod2R1,
    Owner::Mod5R1,
    Owner::Unmasked,
    Owner::AllForeign,
];

fn tile_ids_for(owner: Owner, tiles: &[i32]) -> Vec<i32> {
    tiles
        .iter()
        .map(|&e| {
            if e < 0 {
                return -1;
            }
            let keep = match owner {
                Owner::Mod2R0 => e % 2 == 0,
                Owner::Mod2R1 => e % 2 == 1,
                Owner::Mod5R1 => e % 5 == 1,
                Owner::Unmasked => true,
                Owner::AllForeign => false,
            };
            if keep {
                e
            } else {
                -1
            }
        })
        .collect()
}

fn bytes_i32(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn gen_x(r: &mut Rng, tokens: usize) -> Vec<f32> {
    (0..tokens * HIDDEN)
        .map(|i| {
            let mut v = r.normalish();
            if i % 197 == 11 {
                v *= 40.0;
            }
            if i % 4093 == 5 {
                v = 0.0;
            }
            v
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------------------
// Device plumbing
// ---------------------------------------------------------------------------------------------------------

fn alloc_filled(gpu: &mut Gpu, bytes: usize, fill: u8) -> Res<GpuTensor> {
    let n = (bytes + GUARD).div_ceil(4);
    let t = gpu.alloc_tensor(&[n], DType::F32)?;
    gpu.hip.memset(&t.buf, fill as i32, n * 4)?;
    Ok(t)
}

fn dl(gpu: &Gpu, t: &GpuTensor, n: usize) -> Res<Vec<u8>> {
    let mut v = vec![0u8; n];
    gpu.hip.memcpy_dtoh(&mut v, &t.buf)?;
    Ok(v)
}

fn dl_buf(gpu: &Gpu, buf: &hip_bridge::DeviceBuffer, n: usize) -> Res<Vec<u8>> {
    let mut v = vec![0u8; n];
    gpu.hip.memcpy_dtoh(&mut v, buf)?;
    Ok(v)
}

fn sync(gpu: &Gpu) -> Res<()> {
    gpu.hip.device_synchronize()?;
    Ok(())
}

struct Case {
    tokens: usize,
    lay: Layout,
    tiles: GpuTensor,
    sorted: GpuTensor,
    x: GpuTensor,
    /// per grouped row: the flat slot of a row of an owned (non-sentinel after masking) tile
    owned_slots: Vec<bool>,
    total_slots: usize,
}

fn make_case(
    gpu: &mut Gpu,
    r: &mut Rng,
    tokens: usize,
    routing: Routing,
    owner: Owner,
    bound: bool,
) -> Res<Case> {
    let ids = route_ids(routing, tokens, r);
    let mut lay = build_layout(&ids, bound.then_some(tokens * TOP * 16), 6);
    if routing == Routing::Holey {
        punch_dead_rows(&mut lay);
    }
    let masked = tile_ids_for(owner, &lay.tiles);
    let total_slots = tokens * TOP;
    let mut owned_slots = vec![false; total_slots];
    for (row, &s) in lay.sorted.iter().enumerate() {
        if s >= 0 && masked[row / 16] >= 0 {
            owned_slots[s as usize] = true;
        }
    }
    let tiles = gpu.upload_raw(&bytes_i32(&masked), &[masked.len() * 4])?;
    let sorted = gpu.upload_raw(&bytes_i32(&lay.sorted), &[lay.sorted.len() * 4])?;
    let xf = gen_x(r, tokens);
    let x = gpu.upload_f32(&xf, &[tokens, HIDDEN])?;
    Ok(Case {
        tokens,
        lay,
        tiles,
        sorted,
        x,
        owned_slots,
        total_slots,
    })
}

fn free_case(gpu: &mut Gpu, c: Case) -> Res<()> {
    gpu.free_tensor(c.tiles)?;
    gpu.free_tensor(c.sorted)?;
    gpu.free_tensor(c.x)?;
    Ok(())
}

/// Unfused chain: gate/up GEMM, standalone rotate128, down GEMM. Returns (down sidecar bytes of the
/// shared scratch, down output bytes + guard).
fn run_unfused(gpu: &mut Gpu, w: &Weights, c: &Case, fill: u8) -> Res<(Vec<u8>, Vec<u8>)> {
    let g = c.lay.g;
    let xq = gpu.qwen4_moe_rotate256_i8(&c.x, HIDDEN, c.tokens)?;
    let y = alloc_filled(gpu, g * MI * 2, fill)?;
    gpu.gemm_qwen4_moe_gate_up_silu_a8_sym(
        &w.gu_ptrs, &c.tiles, &c.sorted, &xq, &y, M_GU, HIDDEN, TOP, g, c.tokens,
    )?;
    let xd = gpu.qwen4_moe_rotate128_i8(&y, &c.sorted, MI, g, c.total_slots)?;
    let yd = alloc_filled(gpu, g * HIDDEN * 2, fill)?;
    gpu.gemm_qwen4_moe_down_a8_sym(
        &w.dn_ptrs,
        &c.tiles,
        &c.sorted,
        &xd,
        &yd,
        HIDDEN,
        MI,
        1,
        g,
        c.total_slots,
    )?;
    sync(gpu)?;
    let side = dl_buf(
        gpu,
        gpu.scratch.int8_mmq_x_scratch.as_ref().unwrap(),
        GROUPS * c.total_slots * REC,
    )?;
    let out = dl(gpu, &yd, g * HIDDEN * 2 + GUARD)?;
    gpu.free_tensor(y)?;
    gpu.free_tensor(yd)?;
    Ok((side, out))
}

struct FusedProbe {
    /// the fused arm's gate sidecar differed after the fused launch
    gate_clobbered: bool,
    side: Vec<u8>,
    side_tail_ok: bool,
    out: Vec<u8>,
}

#[derive(Clone, Copy)]
enum Perturb {
    None,
    GateByte(usize),
    Signs2,
}

fn run_fused(gpu: &mut Gpu, w: &Weights, c: &Case, fill: u8, perturb: Perturb) -> Res<FusedProbe> {
    let g = c.lay.g;
    let xq: Int8MmqPrepared = gpu.qwen4_moe_rotate256_i8(&c.x, HIDDEN, c.tokens)?;
    sync(gpu)?;
    let gate_bytes = HIDDEN / 128 * c.tokens * REC;
    if let Perturb::GateByte(token) = perturb {
        let buf = gpu.scratch.int8_mmq_x_scratch.as_ref().unwrap();
        let mut rec = dl_buf(gpu, buf, gate_bytes)?;
        let off = token * REC + 8;
        rec[off] = rec[off].wrapping_add(1);
        gpu.hip.memcpy_htod(buf, &rec)?;
    }
    let gate_before = dl_buf(
        gpu,
        gpu.scratch.int8_mmq_x_scratch.as_ref().unwrap(),
        gate_bytes,
    )?;
    // dirty the dedicated down scratch (what the launcher will reserve) with the canary
    gpu.reserve_int8_mmq_down(MI, c.total_slots)?;
    {
        let buf = gpu.scratch.int8_mmq_down_x_scratch.as_ref().unwrap();
        gpu.hip.memset(buf, fill as i32, buf.size())?;
    }
    let s2_orig = if let Perturb::Signs2 = perturb {
        let t = gpu.scratch.mq_signs2_128.as_ref().unwrap();
        let orig = dl(gpu, t, 128 * 4)?;
        let mut f = orig.clone();
        f[5 * 4 + 3] ^= 0x80; // flip the sign of entry 5
        gpu.hip.memcpy_htod(&t.buf, &f)?;
        Some(orig)
    } else {
        None
    };
    gpu.gemm_qwen4_moe_gate_up_silu_a8_rot_sym(
        &w.gu_ptrs, &c.tiles, &c.sorted, &xq, M_GU, HIDDEN, TOP, g, c.tokens,
    )?;
    sync(gpu)?;
    if let Some(orig) = s2_orig {
        let t = gpu.scratch.mq_signs2_128.as_ref().unwrap();
        gpu.hip.memcpy_htod(&t.buf, &orig)?;
    }
    let gate_after = dl_buf(
        gpu,
        gpu.scratch.int8_mmq_x_scratch.as_ref().unwrap(),
        gate_bytes,
    )?;
    let buf = gpu.scratch.int8_mmq_down_x_scratch.as_ref().unwrap();
    let all = dl_buf(gpu, buf, buf.size())?;
    let used = GROUPS * c.total_slots * REC;
    let side_tail_ok = all[used..].iter().all(|&b| b == fill);
    let side = all[..used].to_vec();
    let yd = alloc_filled(gpu, g * HIDDEN * 2, fill)?;
    gpu.gemm_qwen4_moe_down_a8_rot_sym(
        &w.dn_ptrs,
        &c.tiles,
        &c.sorted,
        &yd,
        HIDDEN,
        MI,
        g,
        c.total_slots,
    )?;
    sync(gpu)?;
    let out = dl(gpu, &yd, g * HIDDEN * 2 + GUARD)?;
    gpu.free_tensor(yd)?;
    Ok(FusedProbe {
        gate_clobbered: gate_before != gate_after,
        side,
        side_tail_ok,
        out,
    })
}

#[derive(Default)]
struct Totals {
    cells: usize,
    record_words: u64,
    record_mismatch: u64,
    canary_words: u64,
    canary_violations: u64,
    out_words: u64,
    out_mismatch: u64,
    gate_clobbers: u64,
    tail_violations: u64,
    live_records: u64,
}

fn compare_cell(
    label: &str,
    c: &Case,
    unfused: &(Vec<u8>, Vec<u8>),
    fused: &FusedProbe,
    fill: u8,
    t: &mut Totals,
    fails: &mut Vec<String>,
) {
    t.cells += 1;
    let (mut rm, mut cv) = (0u64, 0u64);
    for s in 0..c.total_slots {
        for grp in 0..GROUPS {
            let off = (grp * c.total_slots + s) * REC;
            let f = &fused.side[off..off + REC];
            if c.owned_slots[s] {
                let u = &unfused.0[off..off + REC];
                t.record_words += (REC / 4) as u64;
                t.live_records += 1;
                rm += f
                    .chunks_exact(4)
                    .zip(u.chunks_exact(4))
                    .filter(|(a, b)| a != b)
                    .count() as u64;
            } else {
                t.canary_words += (REC / 4) as u64;
                cv += f.iter().filter(|&&b| b != fill).count() as u64;
            }
        }
    }
    t.record_mismatch += rm;
    t.canary_violations += cv;
    let om = fused
        .out
        .chunks_exact(4)
        .zip(unfused.1.chunks_exact(4))
        .filter(|(a, b)| a != b)
        .count() as u64;
    t.out_words += (fused.out.len() / 4) as u64;
    t.out_mismatch += om;
    t.gate_clobbers += u64::from(fused.gate_clobbered);
    t.tail_violations += u64::from(!fused.side_tail_ok);
    if rm != 0 || cv != 0 || om != 0 || fused.gate_clobbered || !fused.side_tail_ok {
        fails.push(format!(
            "{label}: {rm} record words, {cv} canary bytes, {om} out words, gate_clobbered={}, tail_ok={}",
            fused.gate_clobbered, fused.side_tail_ok
        ));
    }
}

fn oracle(nt: Option<usize>) -> Res<()> {
    let mut gpu = Gpu::init()?;
    check_arch(&gpu)?;
    gpu.set_mq4_experts_planar(true);
    gpu.ensure_mq_signs_128()?;
    let w = build_weights(&mut gpu, 0xC0FFEE)?;
    let mut t = Totals::default();
    let mut fails: Vec<String> = Vec::new();
    let mut rng = Rng(0x5EED_0001);
    let fills = [0xA5u8, 0xFF, 0x7F, 0x3C];
    let mut cell_no = 0usize;
    let mut run = |gpu: &mut Gpu,
                   tokens: usize,
                   routing: Routing,
                   owner: Owner,
                   bound: bool,
                   rng: &mut Rng,
                   t: &mut Totals,
                   fails: &mut Vec<String>|
     -> Res<()> {
        cell_no += 1;
        let fill = fills[cell_no % fills.len()];
        let c = make_case(gpu, rng, tokens, routing, owner, bound)?;
        let unfused = run_unfused(gpu, &w, &c, fill)?;
        let fused = run_fused(gpu, &w, &c, fill, Perturb::None)?;
        let label = format!(
            "tokens={tokens} {routing:?} {owner:?}{} g={}",
            if bound { " bound" } else { "" },
            c.lay.g
        );
        compare_cell(&label, &c, &unfused, &fused, fill, t, fails);
        free_case(gpu, c)?;
        Ok(())
    };
    for tokens in 1..=64usize {
        for &routing in &ROUTINGS {
            for &owner in &OWNERS {
                run(
                    &mut gpu, tokens, routing, owner, false, &mut rng, &mut t, &mut fails,
                )?;
            }
        }
    }
    for &tokens in &[9usize, 16, 24, 32, 48, 64] {
        for &routing in &[Routing::Distinct, Routing::Hot] {
            for &owner in &[Owner::Mod2R0, Owner::Mod2R1, Owner::Unmasked] {
                run(
                    &mut gpu, tokens, routing, owner, true, &mut rng, &mut t, &mut fails,
                )?;
            }
        }
    }
    // Controls: the comparer must detect a 1-unit change of the fused path's input.
    let mut ctl_gate = 0u64;
    let mut ctl_sign = 0u64;
    let mut ctl_cells = 0usize;
    for &tokens in &[9usize, 24, 48, 64] {
        let c = make_case(
            &mut gpu,
            &mut rng,
            tokens,
            Routing::Distinct,
            Owner::Mod2R0,
            false,
        )?;
        let token = c
            .owned_slots
            .iter()
            .position(|&o| o)
            .map(|s| s / TOP)
            .ok_or("control: no owned live slot")?;
        let unfused = run_unfused(&mut gpu, &w, &c, 0xA5)?;
        for (kind, perturb) in [
            ("gate-sidecar byte +1", Perturb::GateByte(token)),
            ("signs2 sign flip", Perturb::Signs2),
        ] {
            let fused = run_fused(&mut gpu, &w, &c, 0xA5, perturb)?;
            let mut scratch = Totals::default();
            let mut f2 = Vec::new();
            compare_cell("control", &c, &unfused, &fused, 0xA5, &mut scratch, &mut f2);
            let det = scratch.record_mismatch + scratch.out_mismatch;
            println!("CONTROL tokens={tokens} {kind}: record+out mismatches {det}");
            match perturb {
                Perturb::GateByte(_) => ctl_gate += det,
                _ => ctl_sign += det,
            }
            ctl_cells += 1;
        }
        free_case(&mut gpu, c)?;
    }
    for f in fails.iter().take(40) {
        println!("FAIL {f}");
    }
    let nt_s = nt.map_or("auto".to_string(), |n| n.to_string());
    let ok = fails.is_empty() && ctl_gate > 0 && ctl_sign > 0;
    println!(
        "ORACLE nt={nt_s} cells={} live_records={} record_words={} record_mismatches={} canary_words={} canary_violations={} out_words={} out_mismatches={} gate_sidecar_clobbers={} tail_violations={} controls={} gate_ctl_mismatches={} sign_ctl_mismatches={} {}",
        t.cells,
        t.live_records,
        t.record_words,
        t.record_mismatch,
        t.canary_words,
        t.canary_violations,
        t.out_words,
        t.out_mismatch,
        t.gate_clobbers,
        t.tail_violations,
        ctl_cells,
        ctl_gate,
        ctl_sign,
        if ok { "PASS" } else { "FAIL" }
    );
    drop(w);
    if !ok {
        std::process::exit(1);
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------------------
// Bench
// ---------------------------------------------------------------------------------------------------------

fn pct(v: &mut [f64], p: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() - 1) as f64 * p).round() as usize]
}

fn bench(nt: Option<usize>) -> Res<()> {
    const SETS: usize = 8;
    const WARM: usize = 30;
    const ITERS: usize = 200;
    const LAYERS: usize = 48;
    let mut gpu = Gpu::init()?;
    check_arch(&gpu)?;
    gpu.set_mq4_experts_planar(true);
    gpu.ensure_mq_signs_128()?;
    let w = build_weights(&mut gpu, 0xC0FFEE)?;
    let e0 = gpu.hip.event_create()?;
    let e1 = gpu.hip.event_create()?;
    let mut rng = Rng(0xBE4C4);
    let nt_s = nt.map_or("auto".to_string(), |n| n.to_string());
    println!("nt={nt_s}  per MoE layer, us (median of {ITERS}, {LAYERS}-layer back-to-back loops incl. launch gaps)");
    println!("rows | r256 | r256+gu+rot128 | r256+gu_rot | saved | full unfused (+down) | full fused (+down) | saved");
    let mut json: Vec<String> = Vec::new();
    for &tokens in &[16usize, 24, 32, 48, 64] {
        let mut cases = Vec::new();
        for _ in 0..SETS {
            cases.push(make_case(
                &mut gpu,
                &mut rng,
                tokens,
                Routing::Distinct,
                Owner::Mod2R0,
                true,
            )?);
        }
        let g = cases[0].lay.g;
        let yd_bytes = cases.iter().map(|c| c.lay.g).max().unwrap() * HIDDEN * 2;
        let ys: Vec<GpuTensor> = (0..2)
            .map(|_| alloc_filled(&mut gpu, yd_bytes, 0))
            .collect::<Res<_>>()?;
        let y_gu = alloc_filled(
            &mut gpu,
            g.max(cases.iter().map(|c| c.lay.g).max().unwrap()) * MI * 2,
            0,
        )?;
        let ts = tokens * TOP;
        let mut r256 = Vec::new();
        let mut pair_u = Vec::new();
        let mut pair_f = Vec::new();
        let mut full_u = Vec::new();
        let mut full_f = Vec::new();
        for it in 0..WARM + ITERS {
            // r256 alone
            gpu.hip.event_record(&e0, gpu.active_stream.as_ref())?;
            for l in 0..LAYERS {
                let c = &cases[(it + l) % SETS];
                gpu.qwen4_moe_rotate256_i8(&c.x, HIDDEN, tokens)?;
            }
            gpu.hip.event_record(&e1, gpu.active_stream.as_ref())?;
            gpu.hip.event_synchronize(&e1)?;
            let a = gpu.hip.event_elapsed_ms(&e0, &e1)? as f64 * 1000.0 / LAYERS as f64;
            // r256 + gate/up + rotate128
            gpu.hip.event_record(&e0, gpu.active_stream.as_ref())?;
            for l in 0..LAYERS {
                let c = &cases[(it + l) % SETS];
                let xq = gpu.qwen4_moe_rotate256_i8(&c.x, HIDDEN, tokens)?;
                gpu.gemm_qwen4_moe_gate_up_silu_a8_sym(
                    &w.gu_ptrs, &c.tiles, &c.sorted, &xq, &y_gu, M_GU, HIDDEN, TOP, c.lay.g, tokens,
                )?;
                gpu.qwen4_moe_rotate128_i8(&y_gu, &c.sorted, MI, c.lay.g, ts)?;
            }
            gpu.hip.event_record(&e1, gpu.active_stream.as_ref())?;
            gpu.hip.event_synchronize(&e1)?;
            let b = gpu.hip.event_elapsed_ms(&e0, &e1)? as f64 * 1000.0 / LAYERS as f64;
            // r256 + fused gate/up (down left pending, replaced next layer)
            gpu.hip.event_record(&e0, gpu.active_stream.as_ref())?;
            for l in 0..LAYERS {
                let c = &cases[(it + l) % SETS];
                let xq = gpu.qwen4_moe_rotate256_i8(&c.x, HIDDEN, tokens)?;
                gpu.gemm_qwen4_moe_gate_up_silu_a8_rot_sym(
                    &w.gu_ptrs, &c.tiles, &c.sorted, &xq, M_GU, HIDDEN, TOP, c.lay.g, tokens,
                )?;
            }
            gpu.hip.event_record(&e1, gpu.active_stream.as_ref())?;
            gpu.hip.event_synchronize(&e1)?;
            let cc = gpu.hip.event_elapsed_ms(&e0, &e1)? as f64 * 1000.0 / LAYERS as f64;
            // full chains
            gpu.hip.event_record(&e0, gpu.active_stream.as_ref())?;
            for l in 0..LAYERS {
                let c = &cases[(it + l) % SETS];
                let xq = gpu.qwen4_moe_rotate256_i8(&c.x, HIDDEN, tokens)?;
                gpu.gemm_qwen4_moe_gate_up_silu_a8_sym(
                    &w.gu_ptrs, &c.tiles, &c.sorted, &xq, &y_gu, M_GU, HIDDEN, TOP, c.lay.g, tokens,
                )?;
                let xd = gpu.qwen4_moe_rotate128_i8(&y_gu, &c.sorted, MI, c.lay.g, ts)?;
                gpu.gemm_qwen4_moe_down_a8_sym(
                    &w.dn_ptrs,
                    &c.tiles,
                    &c.sorted,
                    &xd,
                    &ys[l % 2],
                    HIDDEN,
                    MI,
                    1,
                    c.lay.g,
                    ts,
                )?;
            }
            gpu.hip.event_record(&e1, gpu.active_stream.as_ref())?;
            gpu.hip.event_synchronize(&e1)?;
            let d = gpu.hip.event_elapsed_ms(&e0, &e1)? as f64 * 1000.0 / LAYERS as f64;
            gpu.hip.event_record(&e0, gpu.active_stream.as_ref())?;
            for l in 0..LAYERS {
                let c = &cases[(it + l) % SETS];
                let xq = gpu.qwen4_moe_rotate256_i8(&c.x, HIDDEN, tokens)?;
                gpu.gemm_qwen4_moe_gate_up_silu_a8_rot_sym(
                    &w.gu_ptrs, &c.tiles, &c.sorted, &xq, M_GU, HIDDEN, TOP, c.lay.g, tokens,
                )?;
                gpu.gemm_qwen4_moe_down_a8_rot_sym(
                    &w.dn_ptrs,
                    &c.tiles,
                    &c.sorted,
                    &ys[l % 2],
                    HIDDEN,
                    MI,
                    c.lay.g,
                    ts,
                )?;
            }
            gpu.hip.event_record(&e1, gpu.active_stream.as_ref())?;
            gpu.hip.event_synchronize(&e1)?;
            let e = gpu.hip.event_elapsed_ms(&e0, &e1)? as f64 * 1000.0 / LAYERS as f64;
            if it >= WARM {
                r256.push(a);
                pair_u.push(b);
                pair_f.push(cc);
                full_u.push(d);
                full_f.push(e);
            }
        }
        let (a, b, cc, d, e) = (
            pct(&mut r256, 0.5),
            pct(&mut pair_u, 0.5),
            pct(&mut pair_f, 0.5),
            pct(&mut full_u, 0.5),
            pct(&mut full_f, 0.5),
        );
        println!(
            "{tokens} | {a:.1} | {b:.1} | {cc:.1} | {:.1} | {d:.1} | {e:.1} | {:.1}",
            b - cc,
            d - e
        );
        json.push(format!(
            "{{\"rows\":{tokens},\"grouped_rows\":{g},\"r256\":{a:.2},\"unfused_gate_up_rot128\":{:.2},\"fused_gate_up_rot\":{:.2},\"full_unfused\":{d:.2},\"full_fused\":{e:.2}}}",
            b - a,
            cc - a
        ));
        for c in cases {
            free_case(&mut gpu, c)?;
        }
        gpu.free_tensor(y_gu)?;
        for y in ys {
            gpu.free_tensor(y)?;
        }
    }
    println!(
        "{{\"unit\":\"us\",\"stat\":\"median\",\"nt\":\"{nt_s}\",\"iters\":{ITERS},\"warmups\":{WARM},\"sets\":{SETS},\"layers\":{LAYERS},\"cells\":[{}]}}",
        json.join(",")
    );
    let _ = gpu.hip.event_destroy(e0);
    let _ = gpu.hip.event_destroy(e1);
    drop(w);
    Ok(())
}

fn main() -> Res<()> {
    let nt: Option<usize> = std::env::args().nth(2).and_then(|s| s.parse().ok());
    if let Some(n) = nt {
        std::env::set_var("HIPFIRE_QWEN4_MOE_EP_A8_NT", n.to_string());
    }
    match std::env::args().nth(1).as_deref() {
        Some("oracle") => oracle(nt),
        Some("bench") => bench(nt),
        _ => {
            eprintln!("usage: qwen4_moe_gu_rot_fold oracle|bench [1|2]");
            std::process::exit(2);
        }
    }
}
