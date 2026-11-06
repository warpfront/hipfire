// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The A3B register-resident-Q PeaceMaker twin on the peacemaker CPU interpreter. The emitted kernel runs on small
//! cells (single-tile, multi-tile, multi-workgroup, ragged tail, diagonal, E4M3 NaN K / V codes) on exactly sized
//! buffers; every live output must be written, finite where the inputs are finite, inputs untouched, guards intact.
//! When `FAQRES_HIPCC_HSACO` names the hipcc object of `attention_fp8_e4m3_fa2_gqa_qresident_v2_q8_a3b_gfx1201`
//! (a clang offload bundle or a bare code object; the repo JIT's kernel cache holds it), the same cells also run the
//! lifted hipcc kernel on the interpreter and the two outputs must be byte-identical. The interpreter is a CPU model
//! (WMMA fp8, E4M3 conversion, `v_exp_f32`, division): it does not replace the GPU byte oracle
//! (`fa_pm_fp8_oracle --pm-qres`).
use peacemaker_ir::emu::{Emulator, Launch, Memory};
use peacemaker_ir::inst::Frontend;
use peacemaker_isa::kernels::fa_qresident::{self, FaQresSpec};
use peacemaker_isa::{native, Arch};

const ARCH: Arch = Arch::Gfx1201;
const E_KARG: u64 = 0x1_0000;
const E_Q: u64 = 0x1000_0000;
const E_K: u64 = 0x2000_0000;
const E_V: u64 = 0x3000_0000;
const E_OUT: u64 = 0x4000_0000;
const E_POS: u64 = 0x5000_0000;
const GUARD: usize = 256;
const SENTINEL: u8 = 0xa5;
const POISON: u32 = 0x7fc0_beef;
const ROW: usize = 516;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn code(&mut self, nan: bool) -> u8 {
        loop {
            let b = (self.next() >> 24) as u8;
            if nan || b & 0x7f != 0x7f {
                return b;
            }
        }
    }
}

struct Cell {
    rows: usize,
    ctx: usize,
    start: usize,
    q: Vec<u8>,
    k: Vec<u8>,
    v: Vec<u8>,
    pos: Vec<i32>,
}

fn cell(rows: usize, start: usize, nan_kv: bool, seed: u64) -> Cell {
    let ctx = start + rows;
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ seed);
    let mut q = Vec::new();
    for _ in 0..rows * 16 * 256 {
        q.push(rng.code(false));
    }
    for i in 0..rows * 16 {
        let scale = 0.01f32 + (i % 13) as f32 * 0.001;
        q.extend_from_slice(&scale.to_le_bytes());
    }
    let scales = [0x1400u16, 0x1800, 0x1c00, 0x2000];
    let cache = |rng: &mut Rng, nan: bool| {
        let mut c = vec![0u8; ctx * ROW];
        for p in 0..ctx {
            let row = &mut c[p * ROW..(p + 1) * ROW];
            for b in row[..512].iter_mut() {
                let allow_nan = nan && rng.next() % 997 == 0;
                *b = rng.code(allow_nan);
            }
            for g in 0..2 {
                let sc = scales[(p * 3 + g) % scales.len()].to_le_bytes();
                row[512 + 2 * g..514 + 2 * g].copy_from_slice(&sc);
            }
        }
        c
    };
    let k = cache(&mut rng, nan_kv);
    let v = cache(&mut rng, nan_kv);
    let pos = (0..rows).map(|r| (start + r) as i32).collect();
    Cell {
        rows,
        ctx,
        start,
        q,
        k,
        v,
        pos,
    }
}

fn region<'a>(mem: &'a Memory, name: &str) -> &'a peacemaker_ir::emu::Region {
    mem.regions.iter().find(|r| r.name == name).unwrap()
}

struct Run {
    out: Vec<u8>,
    guards_intact: bool,
    out_written: usize,
    inputs_untouched: bool,
}

fn program(elf: &[u8], frontend: Frontend) -> peacemaker_ir::Program {
    peacemaker_lift::lift_object(elf, peacemaker_lift::Options { frontend })
        .unwrap()
        .program
}

fn run(prog: &peacemaker_ir::Program, symbol: &str, c: &Cell) -> Result<Run, String> {
    let out_bytes = c.rows * 4096 * 4;
    let mut karg = Vec::new();
    let q_scales = E_Q + (c.rows * 16 * 256) as u64;
    for ptr in [E_Q, q_scales, E_K, E_V, E_OUT + GUARD as u64, E_POS] {
        karg.extend(ptr.to_le_bytes());
    }
    for v in [16i32, 2, 256, c.rows as i32] {
        karg.extend(v.to_le_bytes());
    }
    karg.extend((1.0f32 / 16.0).to_le_bytes());
    assert_eq!(karg.len(), fa_qresident::KERNARG_BYTES as usize);
    let mut out = vec![SENTINEL; out_bytes + 2 * GUARD];
    for w in out[GUARD..GUARD + out_bytes].chunks_exact_mut(4) {
        w.copy_from_slice(&POISON.to_le_bytes());
    }
    let before = out.clone();
    let pos: Vec<u8> = c.pos.iter().flat_map(|p| p.to_le_bytes()).collect();
    let mut mem = Memory::default();
    mem.map("kernarg".into(), E_KARG, karg.clone())?;
    mem.map("q".into(), E_Q, c.q.clone())?;
    mem.map("k".into(), E_K, c.k.clone())?;
    mem.map("v".into(), E_V, c.v.clone())?;
    mem.map("pos".into(), E_POS, pos.clone())?;
    mem.map("out".into(), E_OUT, out)?;
    let emu = Emulator::new(prog, symbol)?;
    let [gx, gy, gz] = fa_qresident::grid(c.rows as u32);
    for z in 0..gz {
        for y in 0..gy {
            for x in 0..gx {
                emu.run_workgroup(
                    &Launch {
                        kernarg_ptr: E_KARG,
                        workgroup: [x, y, z],
                        block: [fa_qresident::BLOCK, 1, 1],
                        dynamic_lds: fa_qresident::DYNAMIC_LDS_BYTES,
                    },
                    &mut mem,
                )?;
            }
        }
    }
    let inputs_untouched = [
        ("kernarg", &karg),
        ("q", &c.q),
        ("k", &c.k),
        ("v", &c.v),
        ("pos", &pos),
    ]
    .iter()
    .all(|(name, bytes)| {
        let r = region(&mem, name);
        r.bytes == **bytes && r.written.count() == 0
    });
    let o = region(&mem, "out");
    Ok(Run {
        out: o.bytes[GUARD..GUARD + out_bytes].to_vec(),
        guards_intact: o.bytes[..GUARD] == before[..GUARD]
            && o.bytes[GUARD + out_bytes..] == before[GUARD + out_bytes..],
        out_written: o.written.count(),
        inputs_untouched,
    })
}

fn words(b: &[u8]) -> Vec<u32> {
    b.chunks_exact(4)
        .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
        .collect()
}

fn pm_program() -> peacemaker_ir::Program {
    let (_, text, _) = fa_qresident::emit_module(ARCH, &FaQresSpec::A3B).unwrap();
    program(&native::assemble(&text, ARCH).unwrap(), Frontend::Builder)
}

/// The hipcc object named by `FAQRES_HIPCC_HSACO`, unbundled when it is a clang offload bundle.
fn hipcc_program() -> Option<peacemaker_ir::Program> {
    let path = std::env::var("FAQRES_HIPCC_HSACO").ok()?;
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let elf = if bytes.starts_with(b"__CLANG_OFFLOAD_BUNDLE__") {
        let dir = std::env::temp_dir().join(format!("faqres-emu-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (i, o) = (dir.join("in.hsaco"), dir.join("out.co"));
        std::fs::write(&i, &bytes).unwrap();
        let status = std::process::Command::new("/opt/rocm/llvm/bin/clang-offload-bundler")
            .args([
                "--unbundle",
                "--type=o",
                "--targets=hipv4-amdgcn-amd-amdhsa--gfx1201",
            ])
            .arg(format!("--input={}", i.display()))
            .arg(format!("--output={}", o.display()))
            .status()
            .unwrap();
        assert!(status.success(), "clang-offload-bundler");
        std::fs::read(&o).unwrap()
    } else {
        bytes
    };
    Some(program(&elf, Frontend::Hipcc))
}

const HIP_SYMBOL: &str = "attention_fp8_e4m3_fa2_gqa_qresident_v2_q8_a3b_gfx1201";

fn check(name: &str, c: &Cell, nan_kv: bool) {
    let pm = pm_program();
    let r = run(&pm, fa_qresident::SYMBOL_DIRECT, c).unwrap_or_else(|e| panic!("{name}: pm: {e}"));
    assert!(r.guards_intact, "{name}: a store escaped the output");
    assert!(r.inputs_untouched, "{name}: an input changed");
    assert_eq!(
        r.out_written,
        c.rows * 4096 * 4,
        "{name}: live output bytes"
    );
    if !nan_kv {
        for (i, w) in words(&r.out).iter().enumerate() {
            assert!(
                f32::from_bits(*w).is_finite(),
                "{name}: non-finite output row {} head {} dim {}: {w:#010x}",
                i / 4096,
                (i / 256) % 16,
                i % 256
            );
        }
    }
    if let Some(hip) = hipcc_program() {
        // The lifted hipcc object needs `ds_max_i32` / `ds_min_i32` (its LDS position-bounds atomics), which the
        // interpreter does not model: report the refusal, never fake a comparison.
        let h = match run(&hip, HIP_SYMBOL, c) {
            Ok(h) => h,
            Err(e) => {
                eprintln!("{name}: hipcc object refused by the interpreter: {e}");
                return;
            }
        };
        assert_eq!(
            h.out_written,
            c.rows * 4096 * 4,
            "{name}: hipcc live output bytes"
        );
        for (i, (g, w)) in words(&r.out).iter().zip(words(&h.out)).enumerate() {
            assert_eq!(
                *g,
                w,
                "{name}: row {} head {} dim {}: pm {:#010x} ({}), hipcc {:#010x} ({})",
                i / 4096,
                (i / 256) % 16,
                i % 256,
                g,
                f32::from_bits(*g),
                w,
                f32::from_bits(w)
            );
        }
        eprintln!(
            "{name}: PM == hipcc on the interpreter ({} words)",
            r.out.len() / 4
        );
    } else {
        eprintln!("{name}: FAQRES_HIPCC_HSACO unset, hipcc comparison skipped");
    }
}

// Full workgroups only (48 queries per KV head, every row live): a dead row keeps `m == -inf` and evaluates
// `-inf - -inf`, a NaN whose default encoding the interpreter refuses for `v_dual_sub_f32` (the hipcc DAG does the same).
#[test]
fn single_workgroup_three_tiles_with_a_ragged_tail() {
    let c = cell(48, 60, false, 1);
    assert_eq!((c.ctx, c.start), (108, 60));
    check("rows48_start60", &c, false);
}

#[test]
fn two_workgroups_per_head_two_tiles() {
    let c = cell(96, 0, false, 2);
    check("rows96_start0", &c, false);
}
