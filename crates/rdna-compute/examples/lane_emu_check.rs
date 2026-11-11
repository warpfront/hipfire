// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! `lane_emu_check`: lift the two gfx1201 hipcc incumbent objects of the A3B exact batch route (lane_proj, wave64;
//! residual_lanes, wave32) and run them under `Emulator::hw_model` (in-flight loads, LDS happens-before races, all
//! schedules, strict LDS) on exact-size guarded buffers, comparing every output word with the bit-exact CPU references of
//! `hip_loop/lane_ref.rs` (NaN policy: non-NaN words bit-equal, NaN must meet NaN). No device.
//!
//! Usage: `lane_emu_check [--proj OBJ.elf] [--res OBJ.elf] [--threads N] [--seed S] [SUITE ...]`
//!   suites (default `proj res neg`): `proj` (n1,n2,n3,n5,n8 x {37,5,3,1 | 600} rows x random/wild/special data),
//!   `res` (M=20 and M=19, B=1..8, plus M=2048 sampled WGs, random/wild/special data), `neg` (negative controls),
//!   `time` (wall time of the FULL production grids: lane_proj n8 segs 8192,4096,32,32 grid 96; residual_lanes
//!   B=8 M=2048 grid 256 -- add `--sample K` to run first/last/extreme + K seeded random WGs instead).
//! Exit 0 = everything clean and bit-exact.

#[path = "hip_loop/lane_ref.rs"]
mod lane_ref;

use peacemaker_ir::emu::{Emulator, Launch, Memory};
use peacemaker_ir::inst::{Abi, Frontend};
use peacemaker_ir::Program;
use std::collections::BTreeMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::time::Instant;

const GUARD: u64 = 1 << 20;
const DEFAULT_PROJ: &str = "/home/kaden/qcal/release-0.4.3/pm-krsi-lane/tmp/inc/lane_proj.elf";
const DEFAULT_RES: &str = "/home/kaden/qcal/release-0.4.3/pm-krsi-lane/tmp/inc/res_lanes.elf";
type Res<T> = Result<T, String>;

// ---------------------------------------------------------------------------------------------------------------
// deterministic data
// ---------------------------------------------------------------------------------------------------------------
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Hdr {
    /// finite f16 scales/zero-points of moderate size
    Random,
    /// every header half is a uniformly random 16-bit pattern (inf/NaN/denormal included)
    WildAll,
    /// one header half in eight is random bits, the rest `Random`
    WildMixed,
}
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Act {
    Random,
    /// +-inf, NaN (several payloads/signs), denormals, +-0 sprinkled over random activations
    Special,
}
#[derive(Clone, Copy, Debug)]
struct Data {
    hdr: Hdr,
    act: Act,
}
const DATA_MODES: [(&str, Data); 4] = [
    (
        "random",
        Data {
            hdr: Hdr::Random,
            act: Act::Random,
        },
    ),
    (
        "wild-mixed",
        Data {
            hdr: Hdr::WildMixed,
            act: Act::Random,
        },
    ),
    (
        "wild-all",
        Data {
            hdr: Hdr::WildAll,
            act: Act::Random,
        },
    ),
    (
        "special-x",
        Data {
            hdr: Hdr::Random,
            act: Act::Special,
        },
    ),
];

fn half_bits(rng: &mut Rng, mode: Hdr, zp: bool) -> u16 {
    let finite = |rng: &mut Rng| -> u16 {
        // exponent field 6..=15 (2^-9..2^0) for scales, 10..=17 for zero points
        let e = if zp {
            10 + rng.below(8)
        } else {
            6 + rng.below(10)
        } as u16;
        (((rng.below(2) as u16) << 15) | (e << 10)) | rng.below(1024) as u16
    };
    match mode {
        Hdr::Random => finite(rng),
        Hdr::WildAll => rng.next() as u16,
        Hdr::WildMixed => {
            if rng.below(8) == 0 {
                rng.next() as u16
            } else {
                finite(rng)
            }
        }
    }
}
fn gen_weights(rng: &mut Rng, rows: usize, groups: usize, mode: Hdr) -> Vec<u8> {
    let mut w = vec![0u8; rows * groups * 136];
    for g in 0..rows * groups {
        let b = &mut w[g * 136..(g + 1) * 136];
        for h in 0..4 {
            b[2 * h..2 * h + 2].copy_from_slice(&half_bits(rng, mode, h & 1 == 1).to_le_bytes());
        }
        for byte in &mut b[8..] {
            *byte = rng.next() as u8;
        }
    }
    w
}
fn gen_x(rng: &mut Rng, n: usize, act: Act) -> Vec<f32> {
    const SPECIAL: [u32; 11] = [
        0x7f80_0000, // +inf
        0xff80_0000, // -inf
        0x7fc0_0000, // qNaN
        0xffc1_2345, // -qNaN payload
        0x7f80_0001, // sNaN
        0x0000_0001, // smallest denormal
        0x8000_0123, // -denormal
        0x8000_0000, // -0.0
        0x0000_0000, // +0.0
        0x7149_f2ca, // 1e30
        0xf149_f2ca, // -1e30
    ];
    (0..n)
        .map(|_| {
            if act == Act::Special && rng.below(48) == 0 {
                f32::from_bits(SPECIAL[rng.below(SPECIAL.len() as u64) as usize])
            } else {
                rng.unit() * 4.0 - 2.0
            }
        })
        .collect()
}
fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn words(b: &[u8]) -> Vec<u32> {
    b.chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

// ---------------------------------------------------------------------------------------------------------------
// objects, buffers, launches
// ---------------------------------------------------------------------------------------------------------------
fn lift(bytes: &[u8]) -> Res<Program> {
    peacemaker_lift::lift_object(
        bytes,
        peacemaker_lift::Options {
            frontend: Frontend::Hipcc,
        },
    )
    .map(|l| l.program)
    .map_err(|e| format!("lift refused the object: {e}"))
}
fn load_program(path: &str) -> Res<(Vec<u8>, Program)> {
    let image = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let p = lift(&image)?;
    Ok((image, p))
}

enum KArg {
    Ptr(u64),
    I32(i32),
}
struct BufSpec {
    name: String,
    bytes: Vec<u8>,
    output: bool,
    /// poison + strict-read (a load of a byte nobody stored is an error)
    undef: bool,
}
struct Job<'p> {
    program: &'p Program,
    symbol: String,
    bufs: Vec<BufSpec>,
    /// kernel explicit args in declaration order; `Buf(i)` = address of `bufs[i]`
    args: Vec<ArgSrc>,
    grid: u32,
    block: u32,
    dynamic_lds: u32,
}
enum ArgSrc {
    Buf(usize),
    Null,
    I32(i32),
}

/// A mapped job: the memory image, kernarg address and buffer bases.
struct Mapped {
    mem: Memory,
    kernarg: u64,
    bases: Vec<u64>,
}
fn map_job(job: &Job) -> Res<Mapped> {
    let kernel = job
        .program
        .kernels
        .iter()
        .find(|k| k.symbol.0 == job.symbol)
        .ok_or_else(|| format!("no kernel {}", job.symbol))?;
    let Abi::Hsa {
        descriptor,
        metadata,
    } = &kernel.abi
    else {
        return Err("not an HSA kernel".into());
    };
    let mut mem = Memory::default();
    let mut bases = Vec::new();
    for b in &job.bufs {
        let init = if b.undef {
            vec![0xA5u8; b.bytes.len()]
        } else {
            b.bytes.clone()
        };
        let base = mem
            .map_guarded(b.name.clone(), init, GUARD)
            .map_err(|e| format!("map {}: {e}", b.name))?;
        if b.undef {
            mem.set_strict_read(&b.name)?;
        }
        bases.push(base);
    }
    let vals: Vec<KArg> = job
        .args
        .iter()
        .map(|a| match a {
            ArgSrc::Buf(i) => KArg::Ptr(bases[*i]),
            ArgSrc::Null => KArg::Ptr(0),
            ArgSrc::I32(v) => KArg::I32(*v),
        })
        .collect();
    let explicit: Vec<_> = metadata
        .parsed
        .args
        .iter()
        .filter(|a| !a.value_kind.starts_with("hidden_"))
        .collect();
    if explicit.len() != vals.len() {
        return Err(format!(
            "object declares {} explicit args, job passes {}",
            explicit.len(),
            vals.len()
        ));
    }
    let end = metadata
        .parsed
        .args
        .iter()
        .map(|a| (a.offset + a.size) as usize)
        .max()
        .unwrap_or(0);
    let mut kern = vec![
        0u8;
        (metadata
            .parsed
            .kernarg_segment_size
            .max(descriptor.kernarg_size) as usize)
            .max(end)
    ];
    for (m, v) in explicit.iter().zip(&vals) {
        let bytes: Vec<u8> = match v {
            KArg::Ptr(p) => p.to_le_bytes().to_vec(),
            KArg::I32(x) => x.to_le_bytes().to_vec(),
        };
        if bytes.len() != m.size as usize {
            return Err(format!(
                "explicit arg is {} B in the object, {} B passed",
                m.size,
                bytes.len()
            ));
        }
        kern[m.offset as usize..m.offset as usize + bytes.len()].copy_from_slice(&bytes);
    }
    for a in metadata
        .parsed
        .args
        .iter()
        .filter(|a| a.value_kind.starts_with("hidden_"))
    {
        let v: u64 = match a.value_kind.as_str() {
            "hidden_block_count_x" => u64::from(job.grid),
            "hidden_block_count_y" | "hidden_block_count_z" => 1,
            "hidden_group_size_x" => u64::from(job.block),
            "hidden_group_size_y" | "hidden_group_size_z" => 1,
            "hidden_grid_dims" => 3,
            "hidden_dynamic_lds_size" => u64::from(job.dynamic_lds),
            _ => 0,
        };
        let (off, size) = (a.offset as usize, a.size as usize);
        kern[off..off + size].copy_from_slice(&v.to_le_bytes()[..size.min(8)]);
    }
    kern.resize(kern.len().next_multiple_of(16), 0);
    let kernarg = mem.map_guarded("kernarg".into(), kern, GUARD)?;
    Ok(Mapped {
        mem,
        kernarg,
        bases,
    })
}

fn emulator<'p>(program: &'p Program, symbol: &str) -> Res<Emulator<'p>> {
    Ok(Emulator::new(program, symbol)?
        .strict_lds()
        .hw_model()
        .nan_meets_nan()
        .gfx1100_as_gfx1151())
}

/// Findings of a run: (workgroup, message).
type Findings = Vec<(u32, String)>;

/// Run `wgs` of the job, `threads` at a time, each thread on its own copy of the memory; the stores of every workgroup
/// are merged into `m.mem` (workgroups write disjoint outputs; a byte stored by two workgroups is a finding).
fn run_wgs(job: &Job, m: &mut Mapped, wgs: &[u32], threads: usize) -> Res<(Findings, u64)> {
    let emu = emulator(job.program, &job.symbol)?;
    let mut findings: Findings = Vec::new();
    let mut instructions = 0u64;
    let nthreads = threads.min(wgs.len()).max(1);
    let chunks: Vec<Vec<u32>> = (0..nthreads)
        .map(|t| wgs.iter().copied().skip(t).step_by(nthreads).collect())
        .collect();
    let (kernarg, block, lds) = (m.kernarg, job.block, job.dynamic_lds);
    let results: Vec<(Memory, Findings, u64)> = std::thread::scope(|scope| {
        let handles: Vec<_> = chunks
            .iter()
            .map(|chunk| {
                let emu = &emu;
                let mut mem = m.mem.clone();
                scope.spawn(move || {
                    let (mut fs, mut n) = (Findings::new(), 0u64);
                    for &wg in chunk {
                        let launch = Launch {
                            kernarg_ptr: kernarg,
                            workgroup: [wg, 0, 0],
                            block: [block, 1, 1],
                            dynamic_lds: lds,
                        };
                        match catch_unwind(AssertUnwindSafe(|| {
                            emu.run_workgroup_report(&launch, &mut mem)
                        })) {
                            Ok(Ok(r)) => n += r.instructions,
                            Ok(Err(e)) => fs.push((wg, e)),
                            Err(p) => fs.push((
                                wg,
                                format!(
                                    "emulator panic: {}",
                                    p.downcast_ref::<String>()
                                        .cloned()
                                        .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                                        .unwrap_or_default()
                                ),
                            )),
                        }
                    }
                    (mem, fs, n)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("worker thread"))
            .collect()
    });
    for (mem, fs, n) in results {
        findings.extend(fs);
        instructions += n;
        for (i, r) in mem.regions.iter().enumerate() {
            if !r.name.starts_with('y') && r.name != "x_out" {
                if r.written.count() != 0 && r.name != "kernarg" {
                    findings.push((u32::MAX, format!("write to input buffer `{}`", r.name)));
                }
                if r.name == "kernarg" && r.written.count() != 0 {
                    findings.push((u32::MAX, "write to the kernarg segment".into()));
                }
                continue;
            }
            let (mut run_start, mut run_len) = (0usize, 0usize);
            let mut flush = |m: &mut Mapped, s: usize, l: usize| -> Res<()> {
                if l != 0 {
                    let base = m.mem.regions[i].base;
                    if (s..s + l).any(|b| m.mem.regions[i].written.contains(b)) {
                        findings.push((
                            u32::MAX,
                            format!(
                                "two workgroups stored to byte range {:#x}+{l} of `{}`",
                                s, r.name
                            ),
                        ));
                    }
                    m.mem.write(base + s as u64, &r.bytes[s..s + l])?;
                }
                Ok(())
            };
            for b in 0..r.bytes.len() {
                if r.written.contains(b) {
                    if run_len == 0 {
                        run_start = b;
                    }
                    run_len += 1;
                } else if run_len != 0 {
                    flush(m, run_start, run_len)?;
                    run_len = 0;
                }
            }
            flush(m, run_start, run_len)?;
        }
    }
    Ok((findings, instructions))
}

/// Workgroup list: all of `0..grid`, or the sampled set (first, last, grid middle, `k` seeded random).
fn sample_wgs(grid: u32, k: usize, seed: u64) -> Vec<u32> {
    let mut v = vec![0, grid - 1, grid / 2, grid / 4, 3 * grid / 4];
    let mut r = Rng(seed ^ 0x9E37_79B9_7F4A_7C15 | 1);
    while v.len() < 5 + k && v.len() < grid as usize {
        v.push(r.below(u64::from(grid)) as u32);
        v.sort_unstable();
        v.dedup();
    }
    v.sort_unstable();
    v.dedup();
    v
}

fn output_region<'a>(m: &'a Mapped, name: &str) -> &'a peacemaker_ir::emu::Region {
    m.mem
        .regions
        .iter()
        .find(|r| r.name == name)
        .expect("output region")
}

/// Compare an output buffer against the reference words. With `full` every word must have been stored.
fn compare_out(
    m: &Mapped,
    name: &str,
    want: &[u32],
    full: bool,
) -> Res<(lane_ref::WordCmp, usize)> {
    let r = output_region(m, name);
    let got = words(&r.bytes);
    if got.len() != want.len() {
        return Err(format!(
            "{name}: {} words, reference {}",
            got.len(),
            want.len()
        ));
    }
    let mut cmp = lane_ref::WordCmp::default();
    let (mut g, mut w) = (Vec::new(), Vec::new());
    let mut unwritten = 0usize;
    for (i, (&a, &b)) in got.iter().zip(want).enumerate() {
        if (0..4).all(|k| r.written.contains(i * 4 + k)) {
            g.push(a);
            w.push(b);
        } else {
            unwritten += 1;
        }
    }
    let c = lane_ref::compare_words(&g, &w);
    cmp.mismatches += c.mismatches;
    cmp.nan_payload_diffs += c.nan_payload_diffs;
    cmp.words += c.words;
    if full && unwritten != 0 {
        return Err(format!(
            "{name}: {unwritten} of {} words were never stored",
            got.len()
        ));
    }
    if let Some(i) = lane_ref::first_mismatch(&g, &w) {
        return Err(format!("{name}: word {i} of the stored set differs: got {:#010x} want {:#010x} ({} mismatches)", g[i], w[i], c.mismatches));
    }
    Ok((cmp, unwritten))
}

// ---------------------------------------------------------------------------------------------------------------
// lane_proj
// ---------------------------------------------------------------------------------------------------------------
struct ProjCase {
    n: usize,
    rows: [usize; 4],
    data: Data,
    seed: u64,
}
fn proj_job<'p>(
    program: &'p Program,
    c: &ProjCase,
    w: &[Vec<u8>],
    x: &[f32],
    rows_arg: [i32; 4],
    grid: u32,
) -> Job<'p> {
    let mut bufs = Vec::new();
    let mut args = Vec::new();
    let mut wi = [None; 4];
    for s in 0..4 {
        if c.rows[s] != 0 {
            wi[s] = Some(bufs.len());
            bufs.push(BufSpec {
                name: format!("A{s}"),
                bytes: w[s].clone(),
                output: false,
                undef: false,
            });
        }
    }
    for s in 0..4 {
        args.push(match wi[s] {
            Some(i) => ArgSrc::Buf(i),
            None => ArgSrc::Null,
        });
    }
    let mut yi = [None; 4];
    for s in 0..4 {
        if c.rows[s] != 0 {
            yi[s] = Some(bufs.len());
            bufs.push(BufSpec {
                name: format!("y{s}"),
                bytes: vec![0; c.n * c.rows[s] * 4],
                output: true,
                undef: true,
            });
        }
    }
    for s in 0..4 {
        args.push(match yi[s] {
            Some(i) => ArgSrc::Buf(i),
            None => ArgSrc::Null,
        });
    }
    let xi = bufs.len();
    bufs.push(BufSpec {
        name: "x".into(),
        bytes: f32_bytes(x),
        output: false,
        undef: false,
    });
    args.push(ArgSrc::Buf(xi));
    for v in rows_arg {
        args.push(ArgSrc::I32(v));
    }
    Job {
        program,
        symbol: format!("gemv_mq4g256v2_lane_proj_k2048_gfx1201_n{}", c.n),
        bufs,
        args,
        grid,
        block: 512,
        dynamic_lds: (c.n * 8192) as u32,
    }
}
fn proj_grid(total: usize) -> u32 {
    total.div_ceil(48).clamp(1, 96) as u32
}
fn proj_inputs(c: &ProjCase) -> (Vec<Vec<u8>>, Vec<f32>) {
    let mut rng = Rng(c.seed | 1);
    let w: Vec<Vec<u8>> = (0..4)
        .map(|s| gen_weights(&mut rng, c.rows[s], 8, c.data.hdr))
        .collect();
    let x = gen_x(&mut rng, c.n * 2048, c.data.act);
    (w, x)
}
fn proj_reference(c: &ProjCase, w: &[Vec<u8>], x: &[f32]) -> Vec<Vec<u32>> {
    let segs: Vec<usize> = (0..4).filter(|&s| c.rows[s] != 0).collect();
    let ws: Vec<&[u8]> = segs.iter().map(|&s| w[s].as_slice()).collect();
    let rows: Vec<usize> = segs.iter().map(|&s| c.rows[s]).collect();
    lane_ref::lane_proj_ref(&ws, &rows, x, c.n)
}
/// Returns (instructions, words compared, nan payload diffs) or an error naming the first failure.
fn check_proj(
    program: &Program,
    c: &ProjCase,
    wgs: Option<Vec<u32>>,
    threads: usize,
) -> Res<(u64, usize, usize)> {
    let (w, x) = proj_inputs(c);
    let total: usize = c.rows.iter().sum();
    let grid = proj_grid(total);
    let job = proj_job(
        program,
        c,
        &w,
        &x,
        [
            c.rows[0] as i32,
            c.rows[1] as i32,
            c.rows[2] as i32,
            c.rows[3] as i32,
        ],
        grid,
    );
    let mut m = map_job(&job)?;
    let full = wgs.is_none();
    let wgs = wgs.unwrap_or_else(|| (0..grid).collect());
    let (findings, n) = run_wgs(&job, &mut m, &wgs, threads)?;
    if let Some((wg, e)) = findings.first() {
        return Err(format!(
            "{} finding(s); first (wg {wg}): {e}",
            findings.len()
        ));
    }
    let want = proj_reference(c, &w, &x);
    let (mut words_cmp, mut nan_diffs) = (0, 0);
    let segs: Vec<usize> = (0..4).filter(|&s| c.rows[s] != 0).collect();
    for (k, &s) in segs.iter().enumerate() {
        let (cmp, _) = compare_out(&m, &format!("y{s}"), &want[k], full)?;
        words_cmp += cmp.words;
        nan_diffs += cmp.nan_payload_diffs;
    }
    Ok((n, words_cmp, nan_diffs))
}

fn suite_proj(program: &Program, threads: usize, seed: u64) -> (usize, usize) {
    let (mut ok, mut bad) = (0, 0);
    let shapes: [[usize; 4]; 2] = [[37, 5, 3, 1], [600, 0, 0, 0]];
    for &n in &[1usize, 2, 3, 5, 8] {
        for (si, rows) in shapes.iter().enumerate() {
            for (di, (dname, data)) in DATA_MODES.iter().enumerate() {
                let c = ProjCase {
                    n,
                    rows: *rows,
                    data: *data,
                    seed: seed
                        ^ ((n as u64) << 40)
                        ^ ((si as u64) << 32)
                        ^ ((di as u64) << 24)
                        ^ 0x51,
                };
                let t = Instant::now();
                let r = check_proj(program, &c, None, threads);
                match r {
                    Ok((instr, w, nan)) => {
                        ok += 1;
                        println!("  PASS lane_proj n{n} rows{rows:?} {dname}: {w} words bit-exact ({nan} NaN payload diffs), {instr} wave-insts [{} ms]", t.elapsed().as_millis());
                    }
                    Err(e) => {
                        bad += 1;
                        println!("  FAIL lane_proj n{n} rows{rows:?} {dname}: {e}");
                    }
                }
            }
        }
    }
    (ok, bad)
}

// ---------------------------------------------------------------------------------------------------------------
// residual_lanes
// ---------------------------------------------------------------------------------------------------------------
struct ResCase {
    m: usize,
    b: usize,
    data: Data,
    seed: u64,
}
const RES_SYMBOL: &str = "gemv_mq4g256v2_residual_lanes_gfx1201";
fn res_inputs(c: &ResCase) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let mut rng = Rng(c.seed | 1);
    let w = gen_weights(&mut rng, c.m, 16, c.data.hdr);
    let x = gen_x(&mut rng, c.b * 4096, c.data.act);
    let y: Vec<f32> = (0..c.b * c.m).map(|_| rng.unit() * 2.0 - 1.0).collect();
    (w, x, y)
}
fn res_job<'p>(
    program: &'p Program,
    c: &ResCase,
    w: &[u8],
    x: &[f32],
    y: &[f32],
    m_arg: i32,
) -> Job<'p> {
    Job {
        program,
        symbol: RES_SYMBOL.into(),
        bufs: vec![
            BufSpec {
                name: "A".into(),
                bytes: w.to_vec(),
                output: false,
                undef: false,
            },
            BufSpec {
                name: "x".into(),
                bytes: f32_bytes(x),
                output: false,
                undef: false,
            },
            BufSpec {
                name: "y".into(),
                bytes: f32_bytes(y),
                output: true,
                undef: false,
            },
        ],
        args: vec![
            ArgSrc::Buf(0),
            ArgSrc::Buf(1),
            ArgSrc::Buf(2),
            ArgSrc::I32(m_arg),
            ArgSrc::I32(c.b as i32),
        ],
        grid: c.m.div_ceil(8) as u32,
        block: 128,
        dynamic_lds: 0,
    }
}
fn check_res(
    program: &Program,
    c: &ResCase,
    wgs: Option<Vec<u32>>,
    threads: usize,
) -> Res<(u64, usize, usize)> {
    let (w, x, y0) = res_inputs(c);
    let job = res_job(program, c, &w, &x, &y0, c.m as i32);
    let mut m = map_job(&job)?;
    let full = wgs.is_none();
    let wgs = wgs.unwrap_or_else(|| (0..job.grid).collect());
    let (findings, n) = run_wgs(&job, &mut m, &wgs, threads)?;
    if let Some((wg, e)) = findings.first() {
        return Err(format!(
            "{} finding(s); first (wg {wg}): {e}",
            findings.len()
        ));
    }
    let mut y = y0.clone();
    lane_ref::residual_lanes_ref(&w, &x, &mut y, c.m, c.b);
    let want: Vec<u32> = y.iter().map(|v| v.to_bits()).collect();
    // an unsampled workgroup leaves y at its initial value: the compare only looks at stored words, and a full run
    // additionally requires every word to have been stored
    let (cmp, _) = compare_out(&m, "y", &want, full)?;
    Ok((n, cmp.words, cmp.nan_payload_diffs))
}
fn suite_res(program: &Program, threads: usize, seed: u64) -> (usize, usize) {
    let (mut ok, mut bad) = (0, 0);
    let mut cases: Vec<(ResCase, Option<Vec<u32>>, &str)> = Vec::new();
    for &m in &[20usize, 19] {
        for b in 1..=8usize {
            for (di, (dname, data)) in DATA_MODES.iter().enumerate() {
                cases.push((
                    ResCase {
                        m,
                        b,
                        data: *data,
                        seed: seed
                            ^ ((m as u64) << 36)
                            ^ ((b as u64) << 28)
                            ^ ((di as u64) << 20)
                            ^ 0x77,
                    },
                    None,
                    dname,
                ));
            }
        }
    }
    for (di, (dname, data)) in DATA_MODES.iter().enumerate() {
        cases.push((
            ResCase {
                m: 2048,
                b: 8,
                data: *data,
                seed: seed ^ ((di as u64) << 20) ^ 0x1234,
            },
            Some(sample_wgs(256, 8, seed ^ di as u64)),
            dname,
        ));
    }
    for (c, wgs, dname) in cases {
        let t = Instant::now();
        let sampled = wgs.as_ref().map(|w| w.len());
        match check_res(program, &c, wgs, threads) {
            Ok((instr, w, nan)) => {
                ok += 1;
                println!(
                    "  PASS residual_lanes M={} B={} {dname}{}: {w} words bit-exact ({nan} NaN payload diffs), {instr} wave-insts [{} ms]",
                    c.m,
                    c.b,
                    sampled.map(|n| format!(" ({n} sampled WGs)")).unwrap_or_default(),
                    t.elapsed().as_millis()
                );
            }
            Err(e) => {
                bad += 1;
                println!("  FAIL residual_lanes M={} B={} {dname}: {e}", c.m, c.b);
            }
        }
    }
    (ok, bad)
}

// ---------------------------------------------------------------------------------------------------------------
// negative controls
// ---------------------------------------------------------------------------------------------------------------
/// Replace every dword `pat` of the code object by `repl` (same size): the instruction is removed in place.
fn patch_dwords(image: &[u8], pat: u32, repl: u32, expect_at_least: usize) -> Res<Vec<u8>> {
    let mut out = image.to_vec();
    let mut n = 0;
    for off in (0..out.len().saturating_sub(3)).step_by(4) {
        if u32::from_le_bytes(out[off..off + 4].try_into().unwrap()) == pat {
            out[off..off + 4].copy_from_slice(&repl.to_le_bytes());
            n += 1;
        }
    }
    if n < expect_at_least {
        return Err(format!(
            "expected at least {expect_at_least} occurrences of {pat:#010x}, found {n}"
        ));
    }
    Ok(out)
}
/// As `patch_dwords` for every dword satisfying `pred`.
fn patch_where(
    image: &[u8],
    pred: impl Fn(u32) -> bool,
    repl: u32,
    expect_at_least: usize,
) -> Res<Vec<u8>> {
    let mut out = image.to_vec();
    let mut n = 0;
    for off in (0..out.len().saturating_sub(3)).step_by(4) {
        if pred(u32::from_le_bytes(out[off..off + 4].try_into().unwrap())) {
            out[off..off + 4].copy_from_slice(&repl.to_le_bytes());
            n += 1;
        }
    }
    if n < expect_at_least {
        return Err(format!(
            "expected at least {expect_at_least} matching dwords, found {n}"
        ));
    }
    Ok(out)
}
const S_NOP: u32 = 0xBF80_0000;
const S_BARRIER_WAIT_M1: u32 = 0xBF94_FFFF;
const S_BARRIER_SIGNAL_M1: u32 = 0xBE80_4EC1;
const S_WAIT_DSCNT_0: u32 = 0xBFC6_0000;
const S_WAIT_LOADCNT_0: u32 = 0xBFC0_0000;

fn expect_finding(what: &str, r: Res<(u64, usize, usize)>, needles: &[&str]) -> bool {
    match r {
        Ok(_) => {
            println!("  FAIL {what}: the mutant was NOT rejected");
            false
        }
        Err(e) => {
            let hit = needles.iter().any(|n| e.contains(n));
            let first: String = e.chars().take(230).collect();
            println!(
                "  {} {what}: rejected as expected = {hit} :: {first}",
                if hit { "PASS" } else { "FAIL" }
            );
            hit
        }
    }
}

fn suite_neg(proj_path: &str, res_path: &str, threads: usize, seed: u64) -> (usize, usize) {
    let (mut ok, mut bad) = (0, 0);
    let mut tally = |b: bool| {
        if b {
            ok += 1
        } else {
            bad += 1
        }
    };
    let proj_img = std::fs::read(proj_path).expect("lane_proj object");
    let res_img = std::fs::read(res_path).expect("residual_lanes object");
    let rnd = Data {
        hdr: Hdr::Random,
        act: Act::Random,
    };
    // sanity: the unmutated objects are clean on the same cases
    let pc = ProjCase {
        n: 2,
        rows: [37, 5, 3, 1],
        data: rnd,
        seed: seed ^ 0xABCD,
    };
    let rc = ResCase {
        m: 20,
        b: 8,
        data: rnd,
        seed: seed ^ 0xDCBA,
    };
    let base_p = lift(&proj_img).unwrap();
    let base_r = lift(&res_img).unwrap();
    tally(check_proj(&base_p, &pc, None, threads).is_ok());
    tally(check_res(&base_r, &rc, None, threads).is_ok());
    println!("  control: unmutated lane_proj n2 / residual_lanes are clean on the negative-control cases");
    // 1. barrier wait removed
    let mutants: [(&str, Res<Vec<u8>>, &[&str]); 6] = [
        ("lane_proj without any s_barrier_wait", patch_dwords(&proj_img, S_BARRIER_WAIT_M1, S_NOP, 8), &["[race]", "[in-flight]"]),
        ("lane_proj without any s_barrier_signal+wait (signal removed)", patch_dwords(&proj_img, S_BARRIER_SIGNAL_M1, S_NOP, 8), &["barrier wait without signal", "[race]"]),
        ("lane_proj without any s_wait_dscnt (ds reads/stores unretired)", patch_dwords(&proj_img, S_WAIT_DSCNT_0, S_NOP, 8), &["[in-flight]", "[race]"]),
        ("lane_proj without its partial s_wait_dscnt N>0 waits (ds_load_b128 results read before they land)", patch_where(&proj_img, |d| d & 0xFFFF_0000 == 0xBFC6_0000 && d & 0xFFFF != 0, S_NOP, 8), &["[in-flight]"]),
        ("residual_lanes without any s_wait_loadcnt 0 (loads read before they land)", patch_dwords(&res_img, S_WAIT_LOADCNT_0, S_NOP, 1), &["[in-flight]"]),
        ("residual_lanes without any s_wait_dscnt (ds_bpermute/swizzle result used early)", patch_dwords(&res_img, S_WAIT_DSCNT_0, S_NOP, 1), &["[in-flight]"]),
    ];
    for (what, bytes, needles) in mutants {
        let bytes = match bytes {
            Ok(b) => b,
            Err(e) => {
                println!("  FAIL {what}: cannot build the mutant: {e}");
                tally(false);
                continue;
            }
        };
        let prog = match lift(&bytes) {
            Ok(p) => p,
            Err(e) => {
                println!("  FAIL {what}: mutant does not lift: {e}");
                tally(false);
                continue;
            }
        };
        let r = if what.starts_with("lane_proj") {
            check_proj(&prog, &pc, None, threads)
        } else {
            check_res(&prog, &rc, None, threads)
        };
        tally(expect_finding(what, r, needles));
    }
    // 2. an extra-row overrun: the kernel is told one more row than the exact-size buffers hold
    {
        let (w, x) = proj_inputs(&pc);
        let total: usize = pc.rows.iter().sum();
        let job = proj_job(
            &base_p,
            &pc,
            &w,
            &x,
            [
                pc.rows[0] as i32 + 1,
                pc.rows[1] as i32,
                pc.rows[2] as i32,
                pc.rows[3] as i32,
            ],
            proj_grid(total + 1),
        );
        let r = map_job(&job).and_then(|mut m| {
            let wgs: Vec<u32> = (0..job.grid).collect();
            let (f, n) = run_wgs(&job, &mut m, &wgs, threads)?;
            match f.first() {
                Some((wg, e)) => Err(format!("{} finding(s); first (wg {wg}): {e}", f.len())),
                None => Ok((n, 0, 0)),
            }
        });
        tally(expect_finding(
            "lane_proj with m0 one row larger than the buffer (row overrun)",
            r,
            &["unmapped memory"],
        ));
        let (w, x, y) = res_inputs(&rc);
        let job = res_job(&base_r, &rc, &w, &x, &y, rc.m as i32 + 8);
        let r = map_job(&job).and_then(|mut m| {
            let wgs: Vec<u32> = (0..job.grid).collect();
            let (f, n) = run_wgs(&job, &mut m, &wgs, threads)?;
            match f.first() {
                Some((wg, e)) => Err(format!("{} finding(s); first (wg {wg}): {e}", f.len())),
                None => Ok((n, 0, 0)),
            }
        });
        tally(expect_finding(
            "residual_lanes with M 8 larger than the buffers (row overrun)",
            r,
            &["unmapped memory"],
        ));
    }
    (ok, bad)
}

// ---------------------------------------------------------------------------------------------------------------
// production-grid timing
// ---------------------------------------------------------------------------------------------------------------
fn suite_time(
    proj: &Program,
    res: &Program,
    threads: usize,
    seed: u64,
    sample: Option<usize>,
) -> (usize, usize) {
    let (mut ok, mut bad) = (0, 0);
    let c = ProjCase {
        n: 8,
        rows: [8192, 4096, 32, 32],
        data: Data {
            hdr: Hdr::Random,
            act: Act::Random,
        },
        seed: seed ^ 0xD00D,
    };
    let grid = proj_grid(c.rows.iter().sum());
    let wgs = sample.map(|k| sample_wgs(grid, k, seed));
    let t = Instant::now();
    let r = check_proj(proj, &c, wgs.clone(), threads);
    println!(
        "  lane_proj n8 group dn segs {:?}: grid {grid} WGs ({} run), {} thread(s): {} in {:.1} s{}",
        c.rows,
        wgs.as_ref().map_or(grid as usize, Vec::len),
        threads,
        match &r {
            Ok((i, w, nan)) => {
                ok += 1;
                format!("PASS {w} words bit-exact ({nan} NaN payload diffs), {i} wave-insts")
            }
            Err(e) => {
                bad += 1;
                format!("FAIL {e}")
            }
        },
        t.elapsed().as_secs_f64(),
        if wgs.is_some() { " [SAMPLED]" } else { "" }
    );
    let rc = ResCase {
        m: 2048,
        b: 8,
        data: Data {
            hdr: Hdr::Random,
            act: Act::Random,
        },
        seed: seed ^ 0xBEEF,
    };
    let wgs = sample.map(|k| sample_wgs(256, k, seed));
    let t = Instant::now();
    let r = check_res(res, &rc, wgs.clone(), threads);
    println!(
        "  residual_lanes B=8 M=2048: grid 256 WGs ({} run), {} thread(s): {} in {:.1} s{}",
        wgs.as_ref().map_or(256, Vec::len),
        threads,
        match &r {
            Ok((i, w, nan)) => {
                ok += 1;
                format!("PASS {w} words bit-exact ({nan} NaN payload diffs), {i} wave-insts")
            }
            Err(e) => {
                bad += 1;
                format!("FAIL {e}")
            }
        },
        t.elapsed().as_secs_f64(),
        if wgs.is_some() { " [SAMPLED]" } else { "" }
    );
    (ok, bad)
}

/// Write an `emu_race_gate` spec (spec_version 1) + buffer files for one case into `dir`.
/// `case`: `proj:N:r0,r1,r2,r3:MODE` or `res:M:B:MODE` (MODE = random | wild-mixed | wild-all | special-x).
fn emit_spec(dir: &str, case: &str, obj_path: &str, program: &Program, seed: u64) -> Res<()> {
    use sha2::{Digest, Sha256};
    let parts: Vec<&str> = case.split(':').collect();
    let mode = |m: &str| {
        DATA_MODES
            .iter()
            .find(|(n, _)| *n == m)
            .map(|(_, d)| *d)
            .ok_or_else(|| format!("unknown data mode {m}"))
    };
    let job_holder;
    let (job, name) = match parts.as_slice() {
        ["proj", n, rows, m] => {
            let r: Vec<usize> = rows
                .split(',')
                .map(|v| v.parse::<usize>().map_err(|e| e.to_string()))
                .collect::<Res<_>>()?;
            if r.len() != 4 {
                return Err("proj case needs 4 row counts".into());
            }
            let c = ProjCase {
                n: n.parse().map_err(|_| "N")?,
                rows: [r[0], r[1], r[2], r[3]],
                data: mode(m)?,
                seed,
            };
            let (w, x) = proj_inputs(&c);
            let g = proj_grid(c.rows.iter().sum());
            job_holder = proj_job(
                program,
                &c,
                &w,
                &x,
                [r[0] as i32, r[1] as i32, r[2] as i32, r[3] as i32],
                g,
            );
            (
                &job_holder,
                format!("lane_proj_n{}_{}_{}", c.n, rows.replace(',', "_"), m),
            )
        }
        ["res", mm, b, m] => {
            let c = ResCase {
                m: mm.parse().map_err(|_| "M")?,
                b: b.parse().map_err(|_| "B")?,
                data: mode(m)?,
                seed,
            };
            let (w, x, y) = res_inputs(&c);
            job_holder = res_job(program, &c, &w, &x, &y, c.m as i32);
            (
                &job_holder,
                format!("residual_lanes_M{}_B{}_{}", c.m, c.b, m),
            )
        }
        _ => return Err(format!("bad --emit-spec case `{case}`")),
    };
    std::fs::create_dir_all(dir).map_err(|e| format!("{dir}: {e}"))?;
    let obj = std::fs::read(obj_path).map_err(|e| format!("{obj_path}: {e}"))?;
    let obj_name = obj_path
        .rsplit('/')
        .next()
        .unwrap_or("object.elf")
        .to_owned();
    std::fs::write(format!("{dir}/{obj_name}"), &obj).map_err(|e| e.to_string())?;
    let mut buffers = Vec::new();
    for b in &job.bufs {
        if b.undef {
            buffers.push(serde_json::json!({"name": b.name, "bytes": b.bytes.len(), "init": "undef", "output": b.output}));
        } else {
            let f = format!("{}.bin", b.name);
            std::fs::write(format!("{dir}/{f}"), &b.bytes).map_err(|e| e.to_string())?;
            buffers.push(serde_json::json!({"name": b.name, "bytes": b.bytes.len(), "file": f, "init": "seeded", "output": b.output}));
        }
    }
    let args: Vec<serde_json::Value> = job
        .args
        .iter()
        .enumerate()
        .map(|(i, a)| match a {
            ArgSrc::Buf(k) => {
                serde_json::json!({"name": format!("arg{i}"), "buf": job.bufs[*k].name})
            }
            ArgSrc::Null => serde_json::json!({"name": format!("arg{i}"), "null": true}),
            ArgSrc::I32(v) => serde_json::json!({"name": format!("arg{i}"), "i32": v}),
        })
        .collect();
    let spec = serde_json::json!({
        "spec_version": 1, "name": name, "object": obj_name, "object_sha256": format!("{:x}", Sha256::digest(&obj)),
        "arch": "gfx1201", "symbol": job.symbol, "grid_wg": [job.grid, 1, 1], "block": [job.block, 1, 1],
        "dynamic_lds": job.dynamic_lds, "args": args, "buffers": buffers,
        "nan_policy": "meets_nan",
        "notes": format!("emitted by lane_emu_check --emit-spec; case {case}; seed {seed}"),
    });
    let p = format!("{dir}/spec.json");
    std::fs::write(
        &p,
        serde_json::to_vec_pretty(&spec).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    println!("wrote {p}");
    Ok(())
}

fn main() {
    let mut proj_path = DEFAULT_PROJ.to_owned();
    let mut res_path = DEFAULT_RES.to_owned();
    let (mut threads, mut seed, mut sample) = (8usize, 0xC0FFEE_u64, None);
    let mut emit: Option<(String, String)> = None;
    let mut suites: Vec<String> = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--proj" => proj_path = args.next().expect("--proj PATH"),
            "--res" => res_path = args.next().expect("--res PATH"),
            "--threads" => {
                threads = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .expect("--threads N")
            }
            "--seed" => seed = args.next().and_then(|v| v.parse().ok()).expect("--seed S"),
            "--sample" => {
                sample = Some(
                    args.next()
                        .and_then(|v| v.parse().ok())
                        .expect("--sample K"),
                )
            }
            "--emit-spec" => {
                let dir = args.next().expect("--emit-spec DIR CASE");
                let case = args.next().expect("--emit-spec DIR CASE");
                emit = Some((dir, case));
            }
            s => suites.push(s.to_owned()),
        }
    }
    if suites.is_empty() {
        suites = vec!["proj".into(), "res".into(), "neg".into()];
    }
    let (_pi, proj) = load_program(&proj_path).unwrap_or_else(|e| {
        eprintln!("lane_emu_check: {e}");
        std::process::exit(2)
    });
    let (_ri, res) = load_program(&res_path).unwrap_or_else(|e| {
        eprintln!("lane_emu_check: {e}");
        std::process::exit(2)
    });
    if let Some((dir, case)) = emit {
        let (path, which) = if case.starts_with("proj") {
            (&proj_path, &proj)
        } else {
            (&res_path, &res)
        };
        if let Err(e) = emit_spec(&dir, &case, path, which, seed) {
            eprintln!("lane_emu_check: {e}");
            std::process::exit(2);
        }
        return;
    }
    let mut totals: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for s in &suites {
        println!("== {s}");
        let t = Instant::now();
        let r = match s.as_str() {
            "proj" => suite_proj(&proj, threads, seed),
            "res" => suite_res(&res, threads, seed),
            "neg" => suite_neg(&proj_path, &res_path, threads, seed),
            "time" => suite_time(&proj, &res, threads, seed, sample),
            other => {
                eprintln!("unknown suite {other}");
                std::process::exit(2)
            }
        };
        println!(
            "== {s}: {} passed, {} failed [{:.1} s]",
            r.0,
            r.1,
            t.elapsed().as_secs_f64()
        );
        totals.insert(s.clone(), r);
    }
    let bad: usize = totals.values().map(|v| v.1).sum();
    std::process::exit(if bad == 0 { 0 } else { 1 });
}
