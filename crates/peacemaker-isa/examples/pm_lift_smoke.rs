// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Lab, one-card gfx1201 (fabric): lifted hipcc bodies run as regions inside the
//! P0 pool dispatcher (`kernels::lifted_dispatch_gfx1201`) against the unmodified
//! standalone object on identical inputs. Per body:
//!
//! * conversion (`peacemaker_lift::rewrite::region::lift_region`) and its receipt;
//! * the pool object; `check_carried` re-lifts it and proves every body
//!   instruction is carried and the descriptor modes equal the body's;
//! * the standalone object at its original grid, then the pool with one task per
//!   `vcount` x-run of virtual workgroups; every buffer (inputs and outputs)
//!   compared byte for byte, `events[0] == tasks`, `poison == 0`, the trace
//!   records complete.
//!
//! `pm_lift_smoke STANDALONE_DIR OUT_DIR TEMPLATES.json`: every case runs in the shared
//! multi-template pool of its blockDim (every converted template of `TEMPLATES.json`).
//! A case names only its template id: its kernel symbol comes from the templates table and
//! its standalone object is the `STANDALONE_DIR/*/sa.co` that defines that symbol; the
//! standalone launch's kernarg image is laid out from that object's own metadata (the
//! explicit argument count must equal the case's). Writes `OUT_DIR/templates_r6_lift.json`
//! and `OUT_DIR/smoke.json`; exit 0 when every body is byte-identical.

#[path = "lift_common/mod.rs"]
mod lift_common;

use hip_bridge::{DeviceBuffer, Function, HipRuntime};
use peacemaker_isa::kernels::lifted_dispatch_gfx1201 as ld;
use peacemaker_lift::rewrite::region::{check_carried, lift_region, LiftedRegion, RegionInputs};
use serde_json::{json, Value};
use std::path::Path;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn f32(&mut self) -> f32 {
        (self.next() % 20001) as f32 / 10000.0 - 1.0
    }
    fn f16_small(&mut self) -> u16 {
        // sign, exponent 8..13 (2^-7..2^-2), mantissa
        ((self.next() & 0x8000) | ((8 + self.next() % 6) << 10) | (self.next() & 0x3ff)) as u16
    }
    fn bf16(&mut self) -> u16 {
        (self.f32().to_bits() >> 16) as u16
    }
}
fn f32s(v: impl IntoIterator<Item = f32>) -> Vec<u8> {
    v.into_iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn i32s(v: impl IntoIterator<Item = i32>) -> Vec<u8> {
    v.into_iter().flat_map(|x| x.to_le_bytes()).collect()
}

enum Arg {
    Buf(usize),
    I32(i32),
    U32(u32),
}
struct Case {
    tid: u32,
    name: &'static str,
    /// the template's kernel symbol (from the templates table)
    symbol: String,
    block: u32,
    grid: [u32; 3],
    /// (name, bytes); every buffer is compared after the run
    bufs: Vec<(String, Vec<u8>)>,
    /// buffer index whose content is a table of pointers: (target buffer, byte offset)
    ptr_table: Option<(usize, Vec<usize>)>,
    args: Vec<Arg>,
    vcount: u32,
}

fn o4_gate_up(rng: &mut Lcg) -> Case {
    let (m, k, n_experts, grouped_rows, tokens, x_row_div) =
        (256usize, 2560usize, 4usize, 48usize, 4usize, 10i32);
    let groups = k / 256;
    let expert_bytes = m * (k / 2) + m * groups * 4;
    let mut bufs = vec![("expert_ptrs".to_string(), vec![0u8; n_experts * 8])];
    for e in 0..n_experts {
        let mut w: Vec<u8> = (0..m * (k / 2)).map(|_| rng.next() as u8).collect();
        for _ in 0..m * groups * 2 {
            w.extend(rng.f16_small().to_le_bytes());
        }
        assert_eq!(w.len(), expert_bytes);
        bufs.push((format!("expert{e}"), w));
    }
    let tiles = grouped_rows / 16;
    bufs.push((
        "expert_tile_ids".into(),
        i32s((0..tiles).map(|t| [1, 3, 0][t % 3])),
    ));
    bufs.push((
        "sorted_slot_index".into(),
        i32s((0..grouped_rows).map(|i| {
            if i % 7 == 5 {
                -1
            } else {
                (rng.next() % (tokens as u32 * 10)) as i32
            }
        })),
    ));
    bufs.push(("X_src".into(), f32s((0..tokens * k).map(|_| rng.f32()))));
    bufs.push(("Y_grouped".into(), vec![0xa5; grouped_rows * m * 4]));
    let base = n_experts + 1;
    Case {
        tid: 16,
        name: "moe_gate_up_o4_r2",
        symbol: String::new(),
        block: 128,
        grid: [m.div_ceil(16) as u32, grouped_rows.div_ceil(2) as u32, 1],
        ptr_table: Some((0, (1..=n_experts).collect())),
        args: vec![
            Arg::Buf(0),
            Arg::Buf(base),
            Arg::Buf(base + 1),
            Arg::Buf(base + 2),
            Arg::Buf(base + 3),
            Arg::I32(m as i32),
            Arg::I32(k as i32),
            Arg::I32(x_row_div),
            Arg::I32(grouped_rows as i32),
            Arg::I32(n_experts as i32),
        ],
        bufs,
        vcount: 4,
    }
}

fn unscatter_silu(rng: &mut Lcg) -> Case {
    let (mi, m_total, tokens, shared_n, selectors) =
        (512usize, 40usize, 4usize, 300usize, 170usize);
    let extra = 3usize; // grid-x slots past m_total run the shared/selector tail
    let gy = mi / 128;
    let bufs = vec![
        (
            "Y_grouped".to_string(),
            f32s((0..m_total * 2 * mi).map(|_| rng.f32() * 4.0)),
        ),
        (
            "sorted_slot_index".into(),
            i32s((0..m_total).map(|i| {
                if i % 9 == 4 {
                    -1
                } else {
                    ((i * 7 + 3) % (tokens * 10)) as i32
                }
            })),
        ),
        (
            "signs1".into(),
            f32s((0..128).map(|_| if rng.next() & 1 == 0 { 1.0 } else { -1.0 })),
        ),
        (
            "signs2".into(),
            f32s((0..128).map(|_| if rng.next() & 1 == 0 { 1.0 } else { -1.0 })),
        ),
        ("rotated".into(), vec![0xa5; tokens * 10 * mi * 4]),
        (
            "shared_gate".into(),
            f32s((0..shared_n).map(|_| rng.f32() * 3.0)),
        ),
        (
            "shared_up".into(),
            f32s((0..shared_n).map(|_| rng.f32() * 3.0)),
        ),
        ("shared_out".into(), vec![0xa5; shared_n * 4]),
        (
            "selector".into(),
            f32s((0..selectors).map(|_| rng.f32() * 5.0)),
        ),
    ];
    Case {
        tid: 17,
        name: "moe_unscatter_silu_rot",
        symbol: String::new(),
        block: 32,
        grid: [(m_total + extra) as u32, gy as u32, 1],
        ptr_table: None,
        args: vec![
            Arg::Buf(0),
            Arg::Buf(1),
            Arg::Buf(2),
            Arg::Buf(3),
            Arg::Buf(4),
            Arg::I32(mi as i32),
            Arg::I32(m_total as i32),
            Arg::I32(1),
            Arg::Buf(5),
            Arg::Buf(6),
            Arg::Buf(7),
            Arg::Buf(8),
            Arg::I32(shared_n as i32),
            Arg::I32(selectors as i32),
        ],
        bufs,
        vcount: 3,
    }
}

fn gemv_x4_otile(rng: &mut Lcg) -> Case {
    let (rows, k) = (6usize, 1024usize);
    let ms = [37usize, 64, 5, 90];
    let mut bufs = Vec::new();
    for (i, m) in ms.iter().enumerate() {
        bufs.push((
            format!("W{i}"),
            (0..m * k).flat_map(|_| rng.bf16().to_le_bytes()).collect(),
        ));
    }
    bufs.push(("x".into(), f32s((0..rows * k).map(|_| rng.f32()))));
    for (i, m) in ms.iter().enumerate() {
        bufs.push((format!("y{i}"), vec![0xa5; rows * m * 4]));
    }
    let mut args: Vec<Arg> = (0..9).map(Arg::Buf).collect();
    args.extend(ms.iter().map(|&m| Arg::I32(m as i32)));
    args.push(Arg::I32(k as i32));
    Case {
        tid: 8,
        name: "gemv_x4_otile",
        symbol: String::new(),
        block: 32,
        grid: [ms.iter().sum::<usize>() as u32, 1, 1],
        ptr_table: None,
        args,
        bufs,
        vcount: 5,
    }
}

fn down_dni(rng: &mut Lcg) -> Case {
    let (m, k, n, n_exp, k_top) = (256usize, 640usize, 2usize, 4usize, 10usize);
    let mut bufs = vec![("expert_ptrs".to_string(), vec![0u8; n_exp * 8])];
    for e in 0..n_exp {
        // planar: codes M*320 bytes, then one f16 scale per (row, 128-group)
        let mut w: Vec<u8> = (0..m * 320).map(|_| rng.next() as u8).collect();
        for _ in 0..m * 5 {
            w.extend(rng.f16_small().to_le_bytes());
        }
        bufs.push((format!("expert{e}"), w));
    }
    bufs.push((
        "topk_indices".into(),
        i32s((0..n * k_top).map(|i| {
            if i % 6 == 4 {
                -1
            } else {
                (rng.next() % n_exp as u32) as i32
            }
        })),
    ));
    bufs.push((
        "rot_batch".into(),
        f32s((0..n * k_top * k).map(|_| rng.f32())),
    ));
    bufs.push(("expert_outputs".into(), vec![0xa5; n * k_top * m * 4]));
    let base = n_exp + 1;
    Case {
        tid: 18,
        name: "moe_down_dni",
        symbol: String::new(),
        block: 128,
        grid: [m.div_ceil(32) as u32, k_top as u32, n as u32],
        ptr_table: Some((0, (1..=n_exp).collect())),
        args: vec![
            Arg::Buf(0),
            Arg::Buf(base),
            Arg::Buf(base + 1),
            Arg::Buf(base + 2),
            Arg::I32(m as i32),
            Arg::I32(k as i32),
            Arg::I32(n as i32),
            Arg::I32(n_exp as i32),
            Arg::I32(k_top as i32),
        ],
        bufs,
        vcount: 4,
    }
}

struct Card {
    rt: HipRuntime,
}
impl Card {
    fn upload(&self, bytes: &[u8]) -> Result<DeviceBuffer, String> {
        let d = self
            .rt
            .malloc(bytes.len().max(16))
            .map_err(|e| format!("hipMalloc: {e}"))?;
        self.rt
            .memcpy_htod(&d, bytes)
            .map_err(|e| format!("upload: {e}"))?;
        Ok(d)
    }
    fn read(&self, d: &DeviceBuffer, n: usize) -> Result<Vec<u8>, String> {
        let mut v = vec![0u8; n];
        self.rt
            .memcpy_dtoh(&mut v, d)
            .map_err(|e| format!("readback: {e}"))?;
        Ok(v)
    }
    fn function(&self, elf: &[u8], symbol: &str) -> Result<Function, String> {
        let m = self
            .rt
            .module_load_data(elf)
            .map_err(|e| format!("hipModuleLoadData({symbol}): {e}"))?;
        let f = self
            .rt
            .module_get_function(&m, symbol)
            .map_err(|e| format!("hipModuleGetFunction({symbol}): {e}"))?;
        std::mem::forget(m);
        Ok(f)
    }
    fn run_blob_sync(
        &self,
        f: &Function,
        grid: [u32; 3],
        block: u32,
        blob: &mut [u8],
    ) -> Result<(), String> {
        // SAFETY: `blob` is the kernel's kernarg image; every pointer in it is a live allocation.
        unsafe {
            self.rt
                .launch_kernel_blob(f, grid, [block, 1, 1], 0, None, blob)
        }
        .map_err(|e| format!("launch: {e}"))?;
        self.rt
            .device_synchronize()
            .map_err(|e| format!("synchronize: {e}"))
    }
}

/// Upload a case's buffers (pointer table patched); returns the device buffers.
fn stage(card: &Card, case: &Case) -> Result<Vec<DeviceBuffer>, String> {
    let mut d = Vec::new();
    for (_, bytes) in &case.bufs {
        d.push(card.upload(bytes)?);
    }
    if let Some((table, targets)) = &case.ptr_table {
        let bytes: Vec<u8> = targets
            .iter()
            .flat_map(|&t| (d[t].as_ptr() as u64).to_le_bytes())
            .collect();
        card.rt
            .memcpy_htod(&d[*table], &bytes)
            .map_err(|e| format!("ptr table: {e}"))?;
    }
    Ok(d)
}

/// The explicit kernarg bytes at the metadata offsets.
fn explicit(
    case: &Case,
    d: &[DeviceBuffer],
    region: &LiftedRegion,
    offsets: &[(u32, u32)],
) -> Vec<u8> {
    let mut k = vec![0u8; region.kernarg_bytes as usize];
    for (arg, (off, size)) in case.args.iter().zip(offsets) {
        let bytes = match arg {
            Arg::Buf(i) => (d[*i].as_ptr() as u64).to_le_bytes().to_vec(),
            Arg::I32(v) => v.to_le_bytes().to_vec(),
            Arg::U32(v) => v.to_le_bytes().to_vec(),
        };
        k[*off as usize..(*off + *size) as usize].copy_from_slice(&bytes[..*size as usize]);
    }
    k
}

fn read_all(card: &Card, case: &Case, d: &[DeviceBuffer]) -> Result<Vec<Vec<u8>>, String> {
    case.bufs
        .iter()
        .zip(d)
        .map(|((_, b), dev)| card.read(dev, b.len()))
        .collect()
}

/// A shared pool to host the case instead of its own single-template pool.
type SharedPool = (String, Vec<u8>);

fn run_case(card: &Card, objects: &Path, case: Case, shared: &SharedPool) -> Result<Value, String> {
    let object = lift_common::find_object(objects, &case.symbol)
        .ok_or_else(|| format!("{}: no standalone object defines it", case.symbol))?
        .1;
    let region = lift_region(
        &object,
        &case.symbol,
        &RegionInputs {
            prefix: format!(".Lt{}b", case.tid),
            workgroup_x: ld::WG_X,
            workgroup_yz: ld::WG_YZ,
            block: case.block,
            dynamic_lds: 0,
        },
    )
    .map_err(|e| e.to_string())?;
    // explicit arg offsets from the metadata (in declaration order, hidden args excluded)
    let lifted = peacemaker_lift::lift_object(
        &object,
        peacemaker_lift::Options {
            frontend: peacemaker_ir::inst::Frontend::Hipcc,
        },
    )
    .map_err(|e| e.to_string())?;
    let kernel = lifted
        .program
        .kernels
        .iter()
        .find(|k| k.symbol.0 == case.symbol)
        .ok_or("kernel")?;
    let peacemaker_ir::inst::Abi::Hsa { metadata, .. } = &kernel.abi else {
        return Err("not HSA".into());
    };
    let offsets: Vec<(u32, u32)> = metadata
        .parsed
        .args
        .iter()
        .filter(|a| !a.value_kind.starts_with("hidden_"))
        .map(|a| (a.offset, a.size))
        .collect();
    if offsets.len() != case.args.len() {
        return Err(format!(
            "{}: {} explicit args in metadata, case has {}",
            &case.symbol,
            offsets.len(),
            case.args.len()
        ));
    }
    let (pool_symbol, pool_elf) = shared.clone();
    let carried = check_carried(&pool_elf, &pool_symbol, &region)?;

    // ---- standalone
    let solo = card.function(&object, &case.symbol)?;
    let d_solo = stage(card, &case)?;
    let mut blob = explicit(&case, &d_solo, &region, &offsets);
    let explicit_end = offsets.iter().map(|(o, s)| o + s).max().unwrap_or(0) as usize;
    blob.truncate(explicit_end.next_multiple_of(8));
    card.run_blob_sync(&solo, case.grid, case.block, &mut blob)?;
    let want = read_all(card, &case, &d_solo)?;

    // ---- persistent pool
    let d_pool = stage(card, &case)?;
    let mut ka = explicit(&case, &d_pool, &region, &offsets);
    let [gx, gy, gz] = case.grid;
    for h in &region.receipt.hidden_args {
        let value: u64 = match h.kind.as_str() {
            "hidden_block_count_x" => gx.into(),
            "hidden_block_count_y" => gy.into(),
            "hidden_block_count_z" => gz.into(),
            "hidden_group_size_x" => case.block.into(),
            "hidden_group_size_y" | "hidden_group_size_z" => 1,
            "hidden_grid_dims" => 1 + u64::from(gy > 1 || gz > 1) + u64::from(gz > 1),
            _ => 0,
        };
        ka[h.offset as usize..(h.offset + h.size) as usize]
            .copy_from_slice(&value.to_le_bytes()[..h.size as usize]);
    }
    let mut tasks = Vec::new();
    for z in 0..gz {
        for y in 0..gy {
            let mut x = 0;
            while x < gx {
                let vc = std::env::var("LIFT_VCOUNT")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(case.vcount);
                let n = vc.min(gx - x);
                let mut t = [0u32; 32];
                t[..12].copy_from_slice(&[case.tid, x, y, z, gx, gy, gz, 0, u32::MAX, 0, 0, n]);
                tasks.extend(t.iter().flat_map(|w| w.to_le_bytes()));
                x += n;
            }
        }
    }
    let n_tasks = (tasks.len() / 128) as u32;
    let dt = card.upload(&tasks)?;
    let ticket = card.upload(&[0u8; 4])?;
    let events = card.upload(&[0u8; 16])?;
    let kargs = card.upload(&ka)?;
    let trace = card.upload(&vec![0u8; n_tasks as usize * 48])?;
    let poison = card.upload(&[0u8; 4])?;
    let mut params = Vec::new();
    for p in [&dt, &ticket, &events, &kargs, &trace, &poison] {
        params.extend((p.as_ptr() as u64).to_le_bytes());
    }
    params.extend(1_000_000u64.to_le_bytes());
    params.extend(n_tasks.to_le_bytes());
    params.extend(u32::from(trace_on()).to_le_bytes());
    params.resize(ld::KERNARG_BYTES as usize, 0);
    let pool_fn = card.function(&pool_elf, &pool_symbol)?;
    let workers = n_tasks.min(
        std::env::var("LIFT_WORKERS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(48),
    );
    let t0 = std::time::Instant::now();
    card.run_blob_sync(&pool_fn, [workers, 1, 1], case.block, &mut params)?;
    let wall_us = t0.elapsed().as_secs_f64() * 1e6;
    let got = read_all(card, &case, &d_pool)?;
    let ev = u32::from_le_bytes(card.read(&events, 4)?[..4].try_into().unwrap());
    let tk = u32::from_le_bytes(card.read(&ticket, 4)?[..4].try_into().unwrap());
    let pz = u32::from_le_bytes(card.read(&poison, 4)?[..4].try_into().unwrap());
    let tr = card.read(&trace, n_tasks as usize * 48)?;
    let traced = tr
        .chunks(48)
        .enumerate()
        .filter(|(i, r)| {
            u32::from_le_bytes(r[0..4].try_into().unwrap()) == *i as u32 && r[40..48] != [0; 8]
        })
        .count();
    let mut mismatched = Vec::new();
    let table = case.ptr_table.as_ref().map(|(t, _)| *t);
    for (i, (((name, _), w), g)) in case.bufs.iter().zip(&want).zip(&got).enumerate() {
        if Some(i) == table {
            continue; // device addresses of the run's own allocations
        }
        let bad = w.iter().zip(g).filter(|(a, b)| a != b).count();
        if bad > 0 {
            let words: Vec<Value> = w
                .chunks(4)
                .zip(g.chunks(4))
                .enumerate()
                .filter(|(_, (a, b))| a != b)
                .take(12)
                .map(|(i, (a, b))| {
                    json!([
                        i,
                        format!("{:08x}", u32::from_le_bytes(a.try_into().unwrap())),
                        format!("{:08x}", u32::from_le_bytes(b.try_into().unwrap()))
                    ])
                })
                .collect();
            mismatched
                .push(json!({"buffer": name, "bytes": bad, "of": w.len(), "first_words": words}));
        }
    }
    let unwritten: usize = case
        .bufs
        .iter()
        .zip(&want)
        .filter(|((_, init), _)| init.first() == Some(&0xa5))
        .map(|(_, w)| w.chunks(4).filter(|c| *c == [0xa5; 4]).count())
        .sum();
    let pass = mismatched.is_empty()
        && ev == n_tasks
        && pz == 0
        && tk == n_tasks + workers
        && traced == if trace_on() { n_tasks as usize } else { 0 };
    for d in d_solo
        .into_iter()
        .chain(d_pool)
        .chain([dt, ticket, events, kargs, trace, poison])
    {
        card.rt.free(d).map_err(|e| e.to_string())?;
    }
    let r = &region.receipt;
    Ok(json!({
        "tid": case.tid, "name": case.name, "symbol": case.symbol, "pool_symbol": pool_symbol, "B": case.block,
        "grid": case.grid, "tasks": n_tasks, "vcount": case.vcount, "workers": workers, "pool_wall_us_incl_launch": wall_us,
        "pass": pass, "mismatched": mismatched, "untouched_output_words_in_standalone": unwritten,
        "events0": ev, "ticket": tk, "poison": pz, "traced": traced, "carried_instructions": carried,
        "pool_object_sha256": format!("{:x}", sha2_digest(&pool_elf)),
        "receipt": {
            "base_object_sha256": hex(&r.base_object_sha256), "waves": r.waves, "live_in": r.live_in,
            "undefined_live_in": r.undefined_live_in, "exits_converted": r.exits_converted,
            "dealloc_removed": r.dealloc_removed, "input_rewrites": r.input_rewrites, "barriers": r.barriers,
            "uniform_branches": r.uniform_branches, "divergent_branches": r.divergent_branches,
            "barrier_participation": r.barrier_participation,
            "hidden_args": r.hidden_args.iter().map(|h| json!({"kind": h.kind, "offset": h.offset, "size": h.size})).collect::<Vec<_>>(),
            "rsrc1_modes": format!("{:#x}", r.rsrc1_modes), "carried_verbatim": r.carried_verbatim,
            "carried_substituted": r.carried_substituted, "branches_retargeted": r.branches_retargeted,
            "sgprs": region.sgprs, "vgprs": region.vgprs, "lds_bytes": region.lds_bytes, "kernarg_bytes": region.kernarg_bytes
        }
    }))
}

/// COPY (tmpl 2) inside the shared 256-thread pool against the host's bytes.
fn copy_check(card: &Card, symbol: &str, elf: &[u8]) -> Result<Value, String> {
    let mut rng = Lcg(0xc0ffee);
    let bytes = 1_234_571usize; // a 3-byte tail
    let src: Vec<u8> = (0..bytes).map(|_| rng.next() as u8).collect();
    let (ds, dd) = (card.upload(&src)?, card.upload(&vec![0xa5; bytes + 64])?);
    let mut ka = Vec::new();
    for v in [dd.as_ptr() as u64, ds.as_ptr() as u64, bytes as u64] {
        ka.extend(v.to_le_bytes());
    }
    let gx = 64u32;
    let mut tasks = Vec::new();
    for x in (0..gx).step_by(4) {
        let mut t = [0u32; 32];
        t[..12].copy_from_slice(&[
            lift_common::COPY_TMPL,
            x,
            0,
            0,
            gx,
            1,
            1,
            0,
            u32::MAX,
            0,
            0,
            4,
        ]);
        tasks.extend(t.iter().flat_map(|w| w.to_le_bytes()));
    }
    let n_tasks = (tasks.len() / 128) as u32;
    let (dt, ticket, events, kargs, trace, poison) = (
        card.upload(&tasks)?,
        card.upload(&[0u8; 4])?,
        card.upload(&[0u8; 16])?,
        card.upload(&ka)?,
        card.upload(&vec![0u8; n_tasks as usize * 48])?,
        card.upload(&[0u8; 4])?,
    );
    let mut params = Vec::new();
    for p in [&dt, &ticket, &events, &kargs, &trace, &poison] {
        params.extend((p.as_ptr() as u64).to_le_bytes());
    }
    params.extend(1_000_000u64.to_le_bytes());
    params.extend(n_tasks.to_le_bytes());
    params.extend(u32::from(trace_on()).to_le_bytes());
    params.resize(ld::KERNARG_BYTES as usize, 0);
    let f = card.function(elf, symbol)?;
    card.run_blob_sync(&f, [8, 1, 1], 256, &mut params)?;
    let got = card.read(&dd, bytes + 64)?;
    let ev = u32::from_le_bytes(card.read(&events, 4)?[..4].try_into().unwrap());
    let pz = u32::from_le_bytes(card.read(&poison, 4)?[..4].try_into().unwrap());
    let bad = got[..bytes]
        .iter()
        .zip(&src)
        .filter(|(a, b)| a != b)
        .count();
    let overrun = got[bytes..].iter().filter(|&&b| b != 0xa5).count();
    for d in [ds, dd, dt, ticket, events, kargs, trace, poison] {
        card.rt.free(d).map_err(|e| e.to_string())?;
    }
    let pass = bad == 0 && overrun == 0 && ev == n_tasks && pz == 0;
    Ok(
        json!({"tid": lift_common::COPY_TMPL, "name": "copy", "pool_symbol": symbol, "bytes": bytes,
        "tasks": n_tasks, "mismatched_bytes": bad, "overrun_bytes": overrun, "events0": ev, "poison": pz, "pass": pass}),
    )
}

/// `LIFT_TRACE=0` runs the pools with the trace off (MegaM0's timing configuration).
fn trace_on() -> bool {
    std::env::var("LIFT_TRACE").map_or(true, |v| v != "0")
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}
fn sha2_digest(b: &[u8]) -> impl std::fmt::LowerHex {
    use sha2::Digest;
    sha2::Sha256::digest(b)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (Some(objects), Some(out), Some(templates)) = (args.get(1), args.get(2), args.get(3))
    else {
        eprintln!("usage: pm_lift_smoke STANDALONE_DIR OUT_DIR TEMPLATES.json");
        std::process::exit(2);
    };
    let (objects, out) = (Path::new(objects), Path::new(out));
    let table: Value =
        serde_json::from_str(&std::fs::read_to_string(templates).expect("templates"))
            .expect("json");
    // every case runs inside the shared pool of its blockDim (hosting every converted
    // rows-6 template of that blockDim)
    let shared: Vec<(u32, SharedPool)> =
        lift_common::build_pools(&lift_common::convert_all(&table, objects))
            .into_iter()
            .filter_map(|(b, symbol, elf, _)| elf.ok().map(|e| (b, (symbol, e))))
            .collect();
    std::fs::create_dir_all(out).expect("out dir");
    let rt = HipRuntime::load().expect("HipRuntime::load");
    rt.set_device(0).expect("hipSetDevice");
    let arch = rt.get_arch(0).expect("arch");
    assert!(arch.starts_with("gfx1201"), "device is {arch}");
    let card = Card { rt };
    let mut rng = Lcg(0x5eed_1234);
    let mut cases = vec![
        o4_gate_up(&mut rng),
        unscatter_silu(&mut rng),
        gemv_x4_otile(&mut rng),
        down_dni(&mut rng),
    ];
    if let Ok(only) = std::env::var("LIFT_ONLY") {
        cases.retain(|c| c.name == only);
    }
    for case in &mut cases {
        case.symbol = table
            .as_array()
            .and_then(|rows| rows.iter().find(|t| t["tid"] == case.tid))
            .and_then(|t| t["symbol"].as_str())
            .unwrap_or_else(|| panic!("template {} has no symbol", case.tid))
            .to_owned();
    }
    let mut results = Vec::new();
    let mut templates = Vec::new();
    let mut all = true;
    for case in cases {
        let (tid, name, symbol, block) = (case.tid, case.name, case.symbol.clone(), case.block);
        let Some(pool) = shared.iter().find(|(b, _)| *b == block).map(|(_, p)| p) else {
            println!(
                "{}",
                json!({"tid": tid, "name": name, "pass": false, "error": "no shared pool for this blockDim"})
            );
            all = false;
            continue;
        };
        let r = run_case(&card, objects, case, pool)
            .unwrap_or_else(|e| json!({"tid": tid, "name": name, "pass": false, "error": e}));
        println!("{r}");
        let pass = r["pass"] == true;
        all &= pass;
        templates.push(json!({
            "tid": tid, "name": name, "symbol": symbol, "B": block, "mode": if pass { "LIFT_NATIVE" } else { "ILLEGAL" },
            "pool": format!("b{block}"), "units_per_claim": 1, "vblock_step": 1,
            "kernarg_bytes": r["receipt"]["kernarg_bytes"], "lds_bytes": r["receipt"]["lds_bytes"],
            "hidden_args": r["receipt"]["hidden_args"], "legal": pass,
            "reason": r.get("error").cloned().unwrap_or(Value::String(String::new())),
            "kernel": if pass { Value::String(format!("p0_r6_pool_b{block}.co")) } else { Value::Null },
            "pool_symbol": r["pool_symbol"],
        }));
        results.push(r);
    }
    if let Some((_, (symbol, elf))) = shared.iter().find(|(b, _)| *b == 256) {
        let r = copy_check(&card, symbol, elf)
            .unwrap_or_else(|e| json!({"tid": 2, "name": "copy", "pass": false, "error": e}));
        println!("{r}");
        all &= r["pass"] == true;
        results.push(r);
    }
    std::fs::write(
        out.join("smoke.json"),
        serde_json::to_string_pretty(&results).unwrap(),
    )
    .unwrap();
    std::fs::write(
        out.join("templates_r6_lift.json"),
        serde_json::to_string_pretty(&templates).unwrap(),
    )
    .unwrap();
    std::process::exit(if all { 0 } else { 1 });
}
