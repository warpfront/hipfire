// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! CPU-only pre-card gate for the MQ8G256 prefill kernels (`kernels/src/mq8_rotate_quantize_x_batched.hip`,
//! `gemm_mq8g256_mmq_i8.gfx12.hip`, `gemm_mq8g256_mmq_i8.gfx11.hip`). Every source is compiled with hipcc for its
//! arch(s) (`gfx1201`: producer + gfx12 GEMM; `gfx1100`: producer + gfx11 GEMM), lifted (`Frontend::Hipcc`) and EXECUTED on
//! the PeaceMaker emulator (gfx1201 natively, gfx1100 as `gfx1100_as_gfx1151`: exact for the integer WMMA chain), and the
//! output is compared BIT-EXACT to CPU references written from `m1/SPEC.md`:
//!
//! * Producer `mq8_rotate_quantize_x_batched`: int8 codes + F32 scale bits against `mq8_rotate_quantize_x` math
//!   (signs seeds 42/1042 from the `gen_fwht_signs` LCG, sign-FWHT256, amax/127, `rintf`, clamp [-128, 127]), on a plain
//!   shape, a token-tail shape (grid with exiting waves) and a shape with a zero group, a large-range group, a tiny-range
//!   group, an integer-valued group and an all-ones group.
//! * GEMMs: every tile of the arch, in BOTH layouts (`wa`: qt14 258 B groups in place, also at a weight base of +2 bytes;
//!   `wb`: aligned repack) and BOTH outputs (F32 and `_bf16`), on a one-tile shape, M/N tail shapes, K = 768 (rows
//!   alternate 4-byte alignment), a tile with M < tile and an extreme-code shape (|dot128| = 2^21). Reference: per output
//!   `acc = 0; for each K128 epoch ascending: dot = exact i32 sum; acc = fmaf(RN(sw * dx), dot as f32, acc)`, `sw` the exact
//!   f16 -> f32 of the 256-group scale, `dx = Xs[(e/2) * N + n]`; `_bf16`: RN-even to BF16 (non-finite passthrough). Layout
//!   A must equal layout B bitwise.
//! * Chains: emulated producer plane -> emulated GEMM against the full-recipe reference.
//!
//! Buffers are NaN-poisoned (outputs), 0xFF-padded at both ends; a write into a pad, a write into an input, an output byte
//! never written, and any READ of a pad byte (the pads are strict-read regions) are failures (`OVERRUN`).
//!
//! Output: one `EMU PASS|FAIL|UNRUN <name> ...` line per (kernel, shape) and a final tally. `FAIL` is a bit mismatch or a
//! buffer violation; `UNRUN` is a kernel the lifter/emulator cannot execute (first error printed). The test asserts there is
//! no `FAIL`; `MQ8_EMU_STRICT=1` also asserts there is no `UNRUN`. `MQ8_EMU_ONLY=<substr>[,<substr>..]` restricts to names
//! containing one of the substrings (a `gfx1100`/`gfx1201` substring also skips the other arch's compiles).
//!
//! Environment: `MQ8_EMU_OUT` (code-object / temp dir, default `$CARGO_TARGET_TMPDIR/mq8_prefill_emu`), `MQ8_EMU_HIPCC`
//! (default `/opt/rocm/core/bin/hipcc`), `MQ8_EMU_REBUILD=1` (recompile even when the `.co` is newer than its source),
//! `MQ8_EMU_BUILD_SLOT=1` (run hipcc under `/home/kaden/qcal/locks/build-slot.sh`; off by default because the whole cargo test
//! normally already runs inside a build slot and nesting two slot holders can deadlock), `MQ8_EMU_THREADS`.
#![allow(dead_code)]
use peacemaker_ir::emu::{Emulator, Launch, Memory};
use peacemaker_ir::inst::{Frontend, Program};
use std::collections::BTreeMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

const BUNDLE_MAGIC: &[u8] = b"__CLANG_OFFLOAD_BUNDLE__";
const DEFAULT_HIPCC: &str = "/opt/rocm/core/bin/hipcc";
const BUILD_SLOT: &str = "/home/kaden/qcal/locks/build-slot.sh";

const KARG: u64 = 0x1000_0000;
const BUF_A: u64 = 0x2000_0000;
const BUF_B: u64 = 0x3000_0000;
const BUF_C: u64 = 0x4000_0000;
const BUF_D: u64 = 0x5000_0000;
const BUF_Y: u64 = 0x6000_0000;
const BUF_E: u64 = 0x7000_0000;
/// Bytes mapped before and after each buffer; filled with 0xFF, required to stay unread and unwritten.
const PAD: usize = 64 * 1024;

// ------------------------------------------------------------------ rng / numeric helpers
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        let mut r = Rng(seed
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(0x123_4567));
        for _ in 0..8 {
            r.xs64();
        }
        r
    }
    fn xs64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit01(&mut self) -> f32 {
        ((self.xs64() >> 40) & 0xFF_FFFF) as f32 / (1u32 << 24) as f32
    }
    fn gauss(&mut self) -> f32 {
        let u1 = ((self.xs64() >> 11) + 1) as f64 / 9007199254740993.0;
        let u2 = (self.xs64() >> 11) as f64 / 9007199254740992.0;
        ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
    }
}

/// Exact f16 -> f32 (subnormals, inf, NaN).
fn f16_val(h: u16) -> f32 {
    let s = ((h as u32) & 0x8000) << 16;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    if e == 0 {
        let v = m as f32 * (1.0 / 16777216.0);
        return if s != 0 { -v } else { v };
    }
    if e == 31 {
        return f32::from_bits(s | 0x7f80_0000 | (m << 13));
    }
    f32::from_bits(s | ((e + 112) << 23) | (m << 13))
}

/// RN-even F32 -> BF16 -> F32 (non-finite passthrough): `mq8*_bf16_rn` of the kernels.
fn bf16_rn(v: f32) -> f32 {
    if !v.is_finite() {
        return v;
    }
    let u = v.to_bits();
    let lsb = (u >> 16) & 1;
    f32::from_bits(u.wrapping_add(0x7FFF).wrapping_add(lsb) & 0xFFFF_0000)
}

/// `gen_fwht_signs(seed, n)` (crates/hipfire-quantize/src/quant_fwht.rs).
fn gen_fwht_signs(seed: u32, n: usize) -> Vec<f32> {
    let mut st = seed;
    (0..n)
        .map(|_| {
            st = st.wrapping_mul(1103515245).wrapping_add(12345) & 0x7fff_ffff;
            if (st >> 16) & 1 == 1 {
                1.0
            } else {
                -1.0
            }
        })
        .collect()
}

// ------------------------------------------------------------------ producer: inputs and CPU reference
/// X [N][K] F32. `edges` (needs N >= 6, K >= 512) plants the producer's special groups.
fn gen_x(n: usize, k: usize, seed: u64, edges: bool) -> Vec<f32> {
    let mut r = Rng::new(seed);
    let mut x = vec![0f32; n * k];
    for nn in 0..n {
        let tokscale = 0.5f32 + 2.0 * r.unit01();
        for kk in 0..k {
            x[nn * k + kk] = r.gauss() * tokscale;
        }
    }
    if edges && n >= 6 && k >= 512 {
        let grp = |t: usize, g: usize| t * k + g * 256..t * k + g * 256 + 256;
        // zero group (scale 1.0, codes 0)
        x[grp(2, 0)].fill(0.0);
        // large range: x1e6 plus one spike (almost every other code rounds to 0)
        for v in &mut x[grp(3, 1)] {
            *v *= 1.0e6;
        }
        x[3 * k + 256 + 17] = 3.0e7;
        // tiny range
        for v in &mut x[grp(4, 0)] {
            *v *= 1.0e-20;
        }
        // small integers
        for v in &mut x[grp(0, 0)] {
            *v = ((r.xs64() % 7) as i32 - 3) as f32;
        }
        // all ones
        x[grp(1, 0)].fill(1.0);
        // huge uniform range
        for v in &mut x[grp(5, 1)] {
            *v *= 1.0e18;
        }
    }
    x
}

struct QRef {
    q: Vec<i8>,
    /// F32 scale bits, `[K/256][N]`.
    s: Vec<u32>,
}

/// CPU reference of `mq8_rotate_quantize_x_batched` (operation for operation, f32).
fn quant_ref(x: &[f32], n: usize, k: usize) -> QRef {
    let s1 = gen_fwht_signs(42, 256);
    let s2 = gen_fwht_signs(1042, 256);
    let mut r = QRef {
        q: vec![0; n * k],
        s: vec![0; (k / 256) * n],
    };
    for t in 0..n {
        for g in 0..k / 256 {
            let o = t * k + g * 256;
            let mut v = [0f32; 256];
            for i in 0..256 {
                v[i] = x[o + i] * s1[i];
            }
            let mut st = 1;
            while st < 256 {
                let mut i = 0;
                while i < 256 {
                    for j in 0..st {
                        let a = v[i + j];
                        let b = v[i + j + st];
                        v[i + j] = a + b;
                        v[i + j + st] = a - b;
                    }
                    i += 2 * st;
                }
                st <<= 1;
            }
            for i in 0..256 {
                v[i] *= 0.0625f32 * s2[i];
            }
            let mut amax = 0f32;
            for e in &v {
                amax = amax.max(e.abs());
            }
            let scale = if amax > 0.0 { amax / 127.0 } else { 1.0 };
            let inv = if amax > 0.0 { 127.0 / amax } else { 0.0 };
            r.s[g * n + t] = scale.to_bits();
            for i in 0..256 {
                let c = (v[i] * inv).round_ties_even().max(-128.0).min(127.0);
                r.q[o + i] = c as i32 as i8;
            }
        }
    }
    r
}

// ------------------------------------------------------------------ GEMM: inputs, layouts, CPU reference
struct GData {
    m: usize,
    k: usize,
    n: usize,
    /// int8 weight codes [M][K]
    wq: Vec<i8>,
    /// f16 bits of the 256-group scales [M][K/256]
    ws: Vec<u16>,
    /// int8 activation codes [N][K]
    xq: Vec<i8>,
    /// F32 activation scales [K/256][N]
    xs: Vec<f32>,
}

fn gen_gemm(m: usize, k: usize, n: usize, seed: u64, extreme: bool) -> GData {
    let mut r = Rng::new(seed);
    let mut wq = vec![0i8; m * k];
    for mm in 0..m {
        for kk in 0..k {
            let v = (r.xs64() % 256) as i32 - 128;
            wq[mm * k + kk] = if extreme {
                if (mm + kk / 128) & 1 == 0 {
                    -128
                } else {
                    127
                }
            } else {
                v as i8
            };
        }
    }
    let mut xq = vec![0i8; n * k];
    for nn in 0..n {
        for kk in 0..k {
            let v = (r.xs64() % 256) as i32 - 128;
            xq[nn * k + kk] = if extreme {
                if (nn + kk / 128) & 1 == 0 {
                    -128
                } else {
                    127
                }
            } else {
                v as i8
            };
        }
    }
    let ng = k / 256;
    let mut ws = vec![0u16; m * ng];
    for mm in 0..m {
        for g in 0..ng {
            let rv = r.xs64();
            let mut bits = (((6 + (rv >> 12) % 4) as u16) << 10) | (rv & 0x3ff) as u16;
            match (mm * 7 + g) % 23 {
                0 => bits = 0x0000,  // zero scale
                1 => bits = 0x0155,  // f16 subnormal
                2 => bits = 0x3555,  // ~0.333
                3 => bits |= 0x8000, // negative
                4 => bits = 0x7bff,  // f16 max
                _ => {}
            }
            ws[mm * ng + g] = bits;
        }
    }
    let mut xs = vec![0f32; ng * n];
    for v in xs.iter_mut() {
        *v = (1.0 + r.unit01()) * (1.0 / 64.0);
    }
    if ng * n > 5 {
        xs[1] = 1.0; // a zero-group scale
        xs[ng * n - 1] = 1.0e-3;
    }
    GData {
        m,
        k,
        n,
        wq,
        ws,
        xq,
        xs,
    }
}

/// Layout A: the artifact's qt14 rows, `K/256` groups of `{f16 scale, int8[256]}` (258 B) back to back.
fn layout_a(d: &GData) -> Vec<u8> {
    let ng = d.k / 256;
    let mut v = Vec::with_capacity(d.m * ng * 258);
    for mm in 0..d.m {
        for g in 0..ng {
            v.extend_from_slice(&d.ws[mm * ng + g].to_le_bytes());
            v.extend(
                d.wq[mm * d.k + g * 256..mm * d.k + (g + 1) * 256]
                    .iter()
                    .map(|&c| c as u8),
            );
        }
    }
    v
}

/// `mq8g256_b_code_offset(m, k)`.
fn b_code_offset(m: usize, k: usize) -> usize {
    (m * (k / 256) * 2).div_ceil(256) * 256
}

/// Layout B: f16 scale plane [M][K/256], padding to 256 B, then the int8 code plane [M][K].
fn layout_b(d: &GData) -> Vec<u8> {
    let off = b_code_offset(d.m, d.k);
    let mut v = vec![0xEEu8; off + d.m * d.k];
    for (i, s) in d.ws.iter().enumerate() {
        v[2 * i..2 * i + 2].copy_from_slice(&s.to_le_bytes());
    }
    for (i, c) in d.wq.iter().enumerate() {
        v[off + i] = *c as u8;
    }
    v
}

/// CPU reference (SPEC.md numerics): per K128 epoch ascending `acc = fmaf(RN(sw * dx), dot as f32, acc)`.
fn gemm_ref(d: &GData, xq: &[i8], xs: &[f32], bf16: bool) -> Vec<f32> {
    let ng = d.k / 256;
    let mut out = vec![0f32; d.n * d.m];
    for n in 0..d.n {
        for m in 0..d.m {
            let mut acc = 0f32;
            for e in 0..d.k / 128 {
                let mut dot = 0i32;
                for i in 0..128 {
                    let kk = 128 * e + i;
                    dot += d.wq[m * d.k + kk] as i32 * xq[n * d.k + kk] as i32;
                }
                let sw = f16_val(d.ws[m * ng + e / 2]);
                let dx = xs[(e / 2) * d.n + n];
                let scale = sw * dx;
                acc = scale.mul_add(dot as f32, acc);
            }
            out[n * d.m + m] = if bf16 { bf16_rn(acc) } else { acc };
        }
    }
    out
}

/// A deliberately wrong association (one fma per K256 group with the summed dot): the gate must be able to tell it apart.
fn gemm_ref_k256(d: &GData, bf16: bool) -> Vec<f32> {
    let ng = d.k / 256;
    let mut out = vec![0f32; d.n * d.m];
    for n in 0..d.n {
        for m in 0..d.m {
            let mut acc = 0f32;
            for g in 0..ng {
                let mut dot = 0i32;
                for i in 0..256 {
                    let kk = 256 * g + i;
                    dot += d.wq[m * d.k + kk] as i32 * d.xq[n * d.k + kk] as i32;
                }
                acc = (f16_val(d.ws[m * ng + g]) * d.xs[g * d.n + n]).mul_add(dot as f32, acc);
            }
            out[n * d.m + m] = if bf16 { bf16_rn(acc) } else { acc };
        }
    }
    out
}

// ------------------------------------------------------------------ hipcc
fn le64(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes(b[o..o + 8].try_into().unwrap())
}
fn rd16(b: &[u8], o: usize) -> usize {
    u16::from_le_bytes(b[o..o + 2].try_into().unwrap()) as usize
}
fn rd32(b: &[u8], o: usize) -> usize {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap()) as usize
}

/// The `arch` ELF of a clang offload bundle (or the file itself if it is already a bare ELF).
fn arch_elf(bytes: &[u8], arch: &str) -> Result<Vec<u8>, String> {
    if bytes.starts_with(b"\x7fELF") {
        return Ok(bytes.to_vec());
    }
    if !bytes.starts_with(BUNDLE_MAGIC) {
        return Err("neither ELF nor clang offload bundle".into());
    }
    let n = le64(bytes, 24) as usize;
    let mut p = 32;
    for _ in 0..n {
        let off = le64(bytes, p) as usize;
        let size = le64(bytes, p + 8) as usize;
        let idlen = le64(bytes, p + 16) as usize;
        let id = String::from_utf8_lossy(&bytes[p + 24..p + 24 + idlen]).to_string();
        p += 24 + idlen;
        if id.contains(arch) {
            return Ok(bytes[off..off + size].to_vec());
        }
    }
    Err(format!("bundle has no {arch} entry"))
}

fn kernels_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../kernels/src")
}

fn out_dir() -> PathBuf {
    std::env::var("MQ8_EMU_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| Path::new(env!("CARGO_TARGET_TMPDIR")).join("mq8_prefill_emu"))
}

/// Compiles `kernels/src/<src>` for `arch` into `<out>/<src>.<arch>.co` (reused when newer than the source).
fn compile(hipcc: &str, src: &str, arch: &str, out: &Path) -> Result<PathBuf, String> {
    let srcp = kernels_dir().join(src);
    let co = out.join(format!("{src}.{arch}.co"));
    let fresh = || -> bool {
        let (Ok(a), Ok(b)) = (std::fs::metadata(&co), std::fs::metadata(&srcp)) else {
            return false;
        };
        matches!((a.modified(), b.modified()), (Ok(x), Ok(y)) if x >= y)
    };
    if std::env::var("MQ8_EMU_REBUILD").as_deref() != Ok("1") && fresh() {
        return Ok(co);
    }
    let tmp = out.join(format!("{src}.{arch}.{}.tmp", std::process::id()));
    let slot = std::env::var("MQ8_EMU_BUILD_SLOT").as_deref() == Ok("1");
    let mut cmd = if slot {
        let mut c = std::process::Command::new(BUILD_SLOT);
        c.args(["nice", "-n", "19", hipcc]);
        c
    } else {
        let mut c = std::process::Command::new("nice");
        c.args(["-n", "19", hipcc]);
        c
    };
    cmd.args([
        "--genco",
        &format!("--offload-arch={arch}"),
        "-O3",
        "--no-offload-compress",
        "-fuse-cuid=none",
        "-mcode-object-version=6",
        "-o",
    ])
    .arg(&tmp)
    .arg(&srcp)
    .env("HIP_VISIBLE_DEVICES", "-1")
    .env("ROCR_VISIBLE_DEVICES", "-1")
    .env("TMPDIR", out);
    let o = cmd.output().map_err(|e| format!("spawn hipcc: {e}"))?;
    if !o.status.success() {
        let _ = std::fs::remove_file(&tmp);
        return Err(format!(
            "hipcc {src} --offload-arch={arch} failed: {}",
            String::from_utf8_lossy(&o.stderr)
        ));
    }
    std::fs::rename(&tmp, &co).map_err(|e| format!("rename {}: {e}", tmp.display()))?;
    Ok(co)
}

/// Overwrites the code of function symbol `name` in an ELF64 AMDGPU object with `s_endpgm` words.
fn blank_symbol(elf: &mut [u8], name: &str) -> bool {
    let shoff = le64(elf, 0x28) as usize;
    let (shentsize, shnum) = (rd16(elf, 0x3a), rd16(elf, 0x3c));
    let (phoff, phentsize, phnum) = (le64(elf, 0x20) as usize, rd16(elf, 0x36), rd16(elf, 0x38));
    for i in 0..shnum {
        let sh = shoff + i * shentsize;
        if rd32(elf, sh + 4) != 2 {
            continue; // SHT_SYMTAB
        }
        let (off, size, link) = (
            le64(elf, sh + 24) as usize,
            le64(elf, sh + 32) as usize,
            rd32(elf, sh + 40),
        );
        let strsh = shoff + link * shentsize;
        let stroff = le64(elf, strsh + 24) as usize;
        for s in 0..size / 24 {
            let e = off + s * 24;
            let n0 = stroff + rd32(elf, e);
            let end = n0 + elf[n0..].iter().position(|&c| c == 0).unwrap();
            if &elf[n0..end] != name.as_bytes() {
                continue;
            }
            let (value, len) = (le64(elf, e + 8), le64(elf, e + 16) as usize);
            for p in 0..phnum {
                let ph = phoff + p * phentsize;
                let (ty, poff, pva, pfs) = (
                    rd32(elf, ph),
                    le64(elf, ph + 8),
                    le64(elf, ph + 16),
                    le64(elf, ph + 32),
                );
                if ty == 1 && value >= pva && value < pva + pfs {
                    let at = (poff + value - pva) as usize;
                    for w in elf[at..at + len].chunks_exact_mut(4) {
                        w.copy_from_slice(&0xBFB0_0000u32.to_le_bytes());
                    }
                    return true;
                }
            }
        }
    }
    false
}

/// A lifted code object. `blanked`: kernels the lifter cannot decode (symbol -> lift error); their code is overwritten with
/// `s_endpgm` so the rest of the object lifts (the lifter refuses a whole object for one undecodable instruction), and the
/// kernel itself is reported `UNRUN` with the lift error.
struct Obj {
    prog: Program,
    blanked: BTreeMap<String, String>,
    as_1151: bool,
}

fn lift_co(path: &Path, arch: &str) -> Result<Obj, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut elf = arch_elf(&bytes, arch).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut blanked = BTreeMap::new();
    for _ in 0..64 {
        match peacemaker_lift::lift_object(
            &elf,
            peacemaker_lift::Options {
                frontend: Frontend::Hipcc,
            },
        ) {
            Ok(l) => {
                return Ok(Obj {
                    prog: l.program,
                    blanked,
                    as_1151: arch == "gfx1100",
                })
            }
            Err(e) => {
                let msg = e.to_string();
                // "Decode at 0x3338 (kernel): ..." names the kernel holding the bad instruction.
                let sym = msg
                    .split_once("Decode at ")
                    .and_then(|(_, r)| r.split_once('(').and_then(|(_, r)| r.split_once(')')))
                    .map(|(s, _)| s.to_string());
                match sym {
                    Some(sym) if blank_symbol(&mut elf, &sym) => {
                        println!("LIFT {arch} {}: cannot decode {sym}: {msg}", path.display());
                        blanked.insert(sym, msg);
                    }
                    _ => return Err(format!("lift_object(Hipcc) of {}: {msg}", path.display())),
                }
            }
        }
    }
    Err(format!(
        "lift_object(Hipcc) of {}: too many undecodable kernels",
        path.display()
    ))
}

// ------------------------------------------------------------------ launching
/// One buffer of a launch: a 0xFF pad of `PAD` bytes precedes and follows it.
struct Buf {
    name: &'static str,
    base: u64,
    bytes: Vec<u8>,
    /// read-only input: must come back byte-identical and never written
    ro: bool,
}

fn align8(n: usize) -> usize {
    (n + 7) & !7
}

/// hipcc kernarg segment: explicit args, then the 256-byte hidden-argument block (code object v5+ layout, offsets
/// relative to the 8-aligned end of the explicit arguments). `blockDim` / `gridDim` are read from it.
fn kernarg(explicit: &[u8], grid: [u32; 3], block: [u32; 3]) -> Vec<u8> {
    let e = align8(explicit.len());
    let mut k = vec![0u8; e + 256];
    k[..explicit.len()].copy_from_slice(explicit);
    for i in 0..3 {
        k[e + 4 * i..e + 4 * i + 4].copy_from_slice(&grid[i].to_le_bytes());
        k[e + 12 + 2 * i..e + 14 + 2 * i].copy_from_slice(&(block[i] as u16).to_le_bytes());
    }
    let dims: u16 = if grid[2] > 1 {
        3
    } else if grid[1] > 1 {
        2
    } else {
        1
    };
    k[e + 64..e + 66].copy_from_slice(&dims.to_le_bytes());
    k
}

fn p64(v: &mut Vec<u8>, p: u64) {
    v.extend_from_slice(&p.to_le_bytes());
}
fn p32(v: &mut Vec<u8>, p: i32) {
    v.extend_from_slice(&p.to_le_bytes());
}

struct Run {
    outs: Vec<Vec<u8>>,
    workgroups: usize,
    steps: u64,
}

/// Opcodes of `symbol` that have no interpreter semantics row (`Emulator::new` reports only the first).
fn unmodeled_ops(prog: &Program, symbol: &str) -> std::collections::BTreeSet<String> {
    let mut set = std::collections::BTreeSet::new();
    let Some(kernel) = prog.kernels.iter().find(|k| k.symbol.0 == symbol) else {
        return set;
    };
    for id in &kernel.body.layout {
        let Some(inst) = kernel.body.insts.get(*id) else {
            continue;
        };
        match peacemaker_ir::isa::lookup(prog.target.arch, inst.op, inst.form) {
            None => {
                set.insert("<unknown decoded opcode>".to_string());
            }
            Some(row) => {
                if peacemaker_ir::emu::semantic_class(row.name).is_none() {
                    set.insert(row.name.to_string());
                }
            }
        }
    }
    set
}

/// Runs every workgroup of `grid` (x fastest) of `symbol` on freshly mapped `bufs`; returns the bytes of the `outs`
/// buffers after checking that no pad region was read or written, every `ro` buffer is untouched and every `outs` byte was
/// written. Any emulator/lift error is returned.
fn run_kernel(
    obj: &Obj,
    symbol: &str,
    explicit: &[u8],
    grid: [u32; 3],
    block: [u32; 3],
    bufs: Vec<Buf>,
    outs: &[&str],
) -> Result<Run, String> {
    if let Some(why) = obj.blanked.get(symbol) {
        return Err(format!("lift_object(Hipcc): {why}"));
    }
    let mut emu = Emulator::new(&obj.prog, symbol).map_err(|e| {
        format!(
            "Emulator::new: {e}; all opcodes without semantics: {:?}",
            unmodeled_ops(&obj.prog, symbol)
        )
    })?;
    if obj.as_1151 {
        emu = emu.gfx1100_as_gfx1151();
    }
    let mut mem = Memory::default();
    mem.map("kernarg".into(), KARG, kernarg(explicit, grid, block))
        .map_err(|e| format!("map kernarg: {e}"))?;
    let mut ro_before = Vec::new();
    for b in &bufs {
        let len = b.bytes.len() as u64;
        if b.ro {
            ro_before.push((b.name, b.bytes.clone()));
        }
        mem.map(
            format!("{}_head", b.name),
            b.base - PAD as u64,
            vec![0xFF; PAD],
        )
        .map_err(|e| format!("map {} head: {e}", b.name))?;
        mem.map(b.name.into(), b.base, b.bytes.clone())
            .map_err(|e| format!("map {}: {e}", b.name))?;
        mem.map(format!("{}_tail", b.name), b.base + len, vec![0xFF; PAD])
            .map_err(|e| format!("map {} tail: {e}", b.name))?;
        mem.set_strict_read(&format!("{}_head", b.name)).unwrap();
        mem.set_strict_read(&format!("{}_tail", b.name)).unwrap();
    }
    let mut steps = 0u64;
    let mut workgroups = 0usize;
    for z in 0..grid[2] {
        for y in 0..grid[1] {
            for x in 0..grid[0] {
                let l = Launch {
                    kernarg_ptr: KARG,
                    workgroup: [x, y, z],
                    block,
                    dynamic_lds: 0,
                };
                let r = catch_unwind(AssertUnwindSafe(|| emu.run_workgroup_counted(&l, &mut mem)));
                match r {
                    Ok(Ok(n)) => steps += n,
                    Ok(Err(e)) if e.contains("[uninit]") => {
                        return Err(format!(
                            "OVERRUN: read outside a buffer, workgroup ({x},{y},{z}): {e}"
                        ))
                    }
                    Ok(Err(e)) => return Err(format!("workgroup ({x},{y},{z}): {e}")),
                    Err(p) => {
                        let msg = p
                            .downcast_ref::<String>()
                            .cloned()
                            .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                            .unwrap_or_default();
                        return Err(format!("workgroup ({x},{y},{z}) panicked: {msg}"));
                    }
                }
                workgroups += 1;
            }
        }
    }
    let region = |name: &str| mem.regions.iter().find(|r| r.name == name).unwrap();
    for b in &bufs {
        for side in ["head", "tail"] {
            let t = region(&format!("{}_{side}", b.name));
            if t.written.count() != 0 || t.bytes.iter().any(|&v| v != 0xFF) {
                return Err(format!("OVERRUN: wrote outside buffer {} ({side})", b.name));
            }
        }
    }
    for (name, before) in &ro_before {
        let r = region(name);
        if &r.bytes != before || r.written.count() != 0 {
            return Err(format!("OVERRUN: read-only buffer {name} was written"));
        }
    }
    let mut outv = Vec::new();
    for name in outs {
        let r = region(name);
        if let Some(i) = r.written.first_unwritten(0, r.bytes.len()) {
            return Err(format!(
                "OVERRUN: byte {i} of output buffer {name} was never written ({} of {} bytes written)",
                r.written.count(),
                r.bytes.len()
            ));
        }
        outv.push(r.bytes.clone());
    }
    Ok(Run {
        outs: outv,
        workgroups,
        steps,
    })
}

// ------------------------------------------------------------------ tally
struct Tally {
    pass: usize,
    fail: usize,
    unrun: usize,
    unrun_why: BTreeMap<String, String>,
    only: Vec<String>,
}
impl Tally {
    fn wanted(&self, name: &str) -> bool {
        self.only.is_empty() || self.only.iter().any(|s| name.contains(s.as_str()))
    }
    fn pass(&mut self, name: &str, detail: &str) {
        self.pass += 1;
        println!("EMU PASS {name} {detail}");
    }
    fn fail(&mut self, name: &str, detail: &str) {
        self.fail += 1;
        println!("EMU FAIL {name} {detail}");
    }
    fn unrun(&mut self, name: &str, why: &str) {
        self.unrun += 1;
        let first = why.lines().next().unwrap_or(why).to_string();
        println!("EMU UNRUN {name} {first}");
        self.unrun_why.insert(name.to_string(), first);
    }
    fn result(&mut self, name: &str, r: Result<String, String>) {
        match r {
            Ok(s) if s.starts_with("FAIL ") => self.fail(name, &s[5..]),
            Ok(s) => self.pass(name, &s),
            Err(e) if e.starts_with("OVERRUN") => self.fail(name, &e),
            Err(e) => self.unrun(name, &e),
        }
    }
}

fn as_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

type Line = (String, Result<String, String>);
type Job<'a> = Box<dyn FnOnce() -> Vec<Line> + Send + 'a>;

/// Runs `jobs` on a small thread pool; each result line is recorded and printed as it completes.
fn run_jobs(jobs: Vec<Job<'_>>, t: &Mutex<Tally>) {
    let threads = std::env::var("MQ8_EMU_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
                .min(8)
        });
    let queue = Mutex::new(jobs.into_iter().collect::<std::collections::VecDeque<_>>());
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| loop {
                let job = queue.lock().unwrap().pop_front();
                let Some(job) = job else { return };
                let lines = job();
                let mut t = t.lock().unwrap();
                for (name, r) in lines {
                    t.result(&name, r);
                }
            });
        }
    });
}

// ------------------------------------------------------------------ arch configuration
struct ArchCfg {
    arch: &'static str,
    gemm_src: &'static str,
    prefix: &'static str,
    /// (tile label, BM, BN)
    tiles: &'static [(&'static str, usize, usize)],
}
const PROD_SRC: &str = "mq8_rotate_quantize_x_batched.hip";
const PROD_SYM: &str = "mq8_rotate_quantize_x_batched";
const GFX1201: ArchCfg = ArchCfg {
    arch: "gfx1201",
    gemm_src: "gemm_mq8g256_mmq_i8.gfx12.hip",
    prefix: "gemm_mq8g256_mmq_i8_",
    tiles: &[
        ("t128x128", 128, 128),
        ("t128x64", 128, 64),
        ("t64x64", 64, 64),
    ],
};
const GFX1100: ArchCfg = ArchCfg {
    arch: "gfx1100",
    gemm_src: "gemm_mq8g256_mmq_i8.gfx11.hip",
    prefix: "gemm_mq8g256_mmq_i8_gfx11_",
    tiles: &[
        ("t128x128", 128, 128),
        ("t64x128", 64, 128),
        ("t64x64", 64, 64),
    ],
};
impl ArchCfg {
    fn sym(&self, tile: &str, layout: &str, bf16: bool) -> String {
        format!(
            "{}{tile}_{layout}{}",
            self.prefix,
            if bf16 { "_bf16" } else { "" }
        )
    }
}

struct ArchObjs {
    cfg: &'static ArchCfg,
    prod: Result<Obj, String>,
    gemm: Result<Obj, String>,
}

// ------------------------------------------------------------------ producer
/// Runs the producer; returns (Xq bytes, Xs bytes).
fn run_producer(
    obj: &Obj,
    x: &[f32],
    n: usize,
    k: usize,
) -> Result<(Run, Vec<u8>, Vec<u8>), String> {
    let xb: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    let sg = |seed: u32| -> Vec<u8> {
        gen_fwht_signs(seed, 256)
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect()
    };
    let mut ex = Vec::new();
    p64(&mut ex, BUF_A); // X
    p64(&mut ex, BUF_Y); // Xq
    p64(&mut ex, BUF_C); // Xs
    p64(&mut ex, BUF_B); // signs1
    p64(&mut ex, BUF_D); // signs2
    p32(&mut ex, k as i32);
    p32(&mut ex, n as i32);
    let bufs = vec![
        Buf {
            name: "x",
            base: BUF_A,
            bytes: xb,
            ro: true,
        },
        Buf {
            name: "xq",
            base: BUF_Y,
            bytes: vec![0xA5; n * k],
            ro: false,
        },
        Buf {
            name: "xs",
            base: BUF_C,
            bytes: vec![0xFF; (k / 256) * n * 4],
            ro: false,
        },
        Buf {
            name: "signs1",
            base: BUF_B,
            bytes: sg(42),
            ro: true,
        },
        Buf {
            name: "signs2",
            base: BUF_D,
            bytes: sg(1042),
            ro: true,
        },
    ];
    let blocks = (n * (k / 256)).div_ceil(8);
    let mut run = run_kernel(
        obj,
        PROD_SYM,
        &ex,
        [blocks as u32, 1, 1],
        [256, 1, 1],
        bufs,
        &["xq", "xs"],
    )?;
    let xs = run.outs.pop().unwrap();
    let xq = run.outs.pop().unwrap();
    Ok((run, xq, xs))
}

/// Codes and scale bits of a producer run against the CPU reference.
fn check_quant(xq: &[u8], xs: &[u8], r: &QRef, n: usize, k: usize) -> String {
    let (mut qbad, mut sbad) = (0usize, 0usize);
    let mut first = String::new();
    for t in 0..n {
        for g in 0..k / 256 {
            let o = t * k + g * 256;
            let got = &xq[o..o + 256];
            let want: Vec<u8> = r.q[o..o + 256].iter().map(|&v| v as u8).collect();
            if got != &want[..] {
                qbad += 1;
                if first.is_empty() {
                    let i = (0..256).find(|&i| got[i] != want[i]).unwrap();
                    first = format!(
                        "first code token={t} group={g} i={i} got {} want {}",
                        got[i] as i8, want[i] as i8
                    );
                }
            }
            let si = (g * n + t) * 4;
            let gs = u32::from_le_bytes(xs[si..si + 4].try_into().unwrap());
            if gs != r.s[g * n + t] {
                sbad += 1;
                if first.is_empty() {
                    first = format!(
                        "first scale token={t} group={g} got {gs:#010x} ({}) want {:#010x} ({})",
                        f32::from_bits(gs),
                        r.s[g * n + t],
                        f32::from_bits(r.s[g * n + t])
                    );
                }
            }
        }
    }
    let ngr = n * (k / 256);
    if qbad == 0 && sbad == 0 {
        format!("codes {ngr}/{ngr} groups exact, scales {ngr}/{ngr} exact")
    } else {
        format!("FAIL codes {qbad}/{ngr} groups differ, scales {sbad}/{ngr} differ; {first}")
    }
}

fn producer_job<'a>(
    a: &'a ArchObjs,
    n: usize,
    k: usize,
    seed: u64,
    edges: bool,
    label: &'static str,
) -> Job<'a> {
    Box::new(move || {
        let name = format!("{} {PROD_SYM} {label}", a.cfg.arch);
        let obj = match &a.prod {
            Ok(o) => o,
            Err(e) => return vec![(name, Err(e.clone()))],
        };
        let t0 = Instant::now();
        let x = gen_x(n, k, seed, edges);
        let r = run_producer(obj, &x, n, k).map(|(run, xq, xs)| {
            let s = check_quant(&xq, &xs, &quant_ref(&x, n, k), n, k);
            if s.starts_with("FAIL") {
                s
            } else {
                format!(
                    "{s} ({} workgroups, {} steps, {:.1}s)",
                    run.workgroups,
                    run.steps,
                    t0.elapsed().as_secs_f64()
                )
            }
        });
        vec![(name, r)]
    })
}

// ------------------------------------------------------------------ GEMMs
/// Runs one GEMM symbol; `w` is mapped at `w_base`. Returns F32 `[N][M]`.
fn run_gemm(
    obj: &Obj,
    symbol: &str,
    bm: usize,
    bn: usize,
    d: &GData,
    w: &[u8],
    w_base: u64,
    xq: &[u8],
    xs: &[u8],
) -> Result<(Vec<f32>, Run), String> {
    let mut ex = Vec::new();
    p64(&mut ex, w_base);
    p64(&mut ex, BUF_B);
    p64(&mut ex, BUF_C);
    p64(&mut ex, BUF_Y);
    p32(&mut ex, d.m as i32);
    p32(&mut ex, d.k as i32);
    p32(&mut ex, d.n as i32);
    let grid = [d.m.div_ceil(bm) as u32, d.n.div_ceil(bn) as u32, 1];
    let block = [(2 * bm) as u32, 1, 1];
    let bufs = vec![
        Buf {
            name: "w",
            base: w_base,
            bytes: w.to_vec(),
            ro: true,
        },
        Buf {
            name: "xq",
            base: BUF_B,
            bytes: xq.to_vec(),
            ro: true,
        },
        Buf {
            name: "xs",
            base: BUF_C,
            bytes: xs.to_vec(),
            ro: true,
        },
        Buf {
            name: "y",
            base: BUF_Y,
            bytes: vec![0xFF; d.n * d.m * 4],
            ro: false,
        },
    ];
    let mut run = run_kernel(obj, symbol, &ex, grid, block, bufs, &["y"])?;
    let y = as_f32(&run.outs.pop().unwrap());
    Ok((y, run))
}

/// Compares `got` (F32 `[N][M]`) bitwise with `want`; returns a one-line detail or a `FAIL` line.
fn compare_f32(got: &[f32], want: &[f32], m: usize) -> Result<String, String> {
    let bad: Vec<usize> = (0..want.len())
        .filter(|&i| got[i].to_bits() != want[i].to_bits())
        .collect();
    if bad.is_empty() {
        Ok(format!("exact {}/{}", want.len(), want.len()))
    } else {
        let i = bad[0];
        Ok(format!(
            "FAIL {} of {} mismatch; first n={} m={} got {:#010x} ({}) want {:#010x} ({})",
            bad.len(),
            want.len(),
            i / m,
            i % m,
            got[i].to_bits(),
            got[i],
            want[i].to_bits(),
            want[i]
        ))
    }
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn i8_bytes(v: &[i8]) -> Vec<u8> {
    v.iter().map(|&c| c as u8).collect()
}

struct Shape {
    m: usize,
    k: usize,
    n: usize,
    extreme: bool,
    /// also run layout A at a weight base of +2 bytes
    off2: bool,
    seed: u64,
    label: &'static str,
}

/// One (tile, output) on one shape: layout A, layout A at +2 bytes, layout B, and layout A == layout B.
fn gemm_job<'a>(
    a: &'a ArchObjs,
    tile: (&'static str, usize, usize),
    bf16: bool,
    sh: &'a Shape,
    d: &'a GData,
    want: &'a [f32],
    wa: &'a [u8],
    wb: &'a [u8],
) -> Job<'a> {
    Box::new(move || {
        let (tl, bm, bn) = tile;
        let dir = a.cfg;
        let mut lines: Vec<Line> = Vec::new();
        let obj = match &a.gemm {
            Ok(o) => o,
            Err(e) => {
                for lay in ["wa", "wb"] {
                    lines.push((
                        format!("{} {} {}", dir.arch, dir.sym(tl, lay, bf16), sh.label),
                        Err(e.clone()),
                    ));
                }
                return lines;
            }
        };
        let xq = i8_bytes(&d.xq);
        let xs = f32_bytes(&d.xs);
        let mut outs: Vec<(&str, Option<Vec<f32>>)> = Vec::new();
        let mut cases: Vec<(&str, &[u8], u64, String)> = vec![
            ("wa", wa, BUF_A, sh.label.to_string()),
            ("wb", wb, BUF_A, sh.label.to_string()),
        ];
        if sh.off2 {
            cases.push(("wa", wa, BUF_A + 2, format!("{} wbase+2", sh.label)));
        }
        for (lay, w, base, label) in cases {
            let sym = dir.sym(tl, lay, bf16);
            let name = format!("{} {sym} {label}", dir.arch);
            let t0 = Instant::now();
            let r = run_gemm(obj, &sym, bm, bn, d, w, base, &xq, &xs);
            let (res, got) = match r {
                Ok((y, run)) => {
                    let s = compare_f32(&y, want, d.m).unwrap();
                    let s = if s.starts_with("FAIL") {
                        s
                    } else {
                        format!(
                            "{s} ({} workgroups, {} steps, {:.1}s)",
                            run.workgroups,
                            run.steps,
                            t0.elapsed().as_secs_f64()
                        )
                    };
                    (Ok(s), Some(y))
                }
                Err(e) => (Err(e), None),
            };
            if base == BUF_A {
                outs.push((lay, got));
            }
            lines.push((name, res));
        }
        // layout A == layout B, bitwise (both outputs present)
        if let (Some((_, Some(ya))), Some((_, Some(yb)))) = (outs.first(), outs.get(1)) {
            let same = ya.iter().zip(yb).all(|(x, y)| x.to_bits() == y.to_bits());
            let name = format!(
                "{} {}{}_wa==wb {}",
                dir.arch,
                dir.prefix,
                format_args!("{tl}{}", if bf16 { "_bf16" } else { "" }),
                sh.label
            );
            lines.push((
                name,
                Ok(if same {
                    format!("layout A == layout B bitwise ({} words)", ya.len())
                } else {
                    "FAIL layout A and layout B outputs differ".to_string()
                }),
            ));
        }
        lines
    })
}

// ------------------------------------------------------------------ chains: emulated producer -> emulated GEMM
fn chain_job<'a>(
    a: &'a ArchObjs,
    tile: (&'static str, usize, usize),
    layout: &'static str,
    bf16: bool,
    sh: &'a Shape,
    d: &'a GData,
    x: &'a [f32],
    rq: &'a QRef,
    want: &'a [f32],
    wblob: &'a [u8],
) -> Job<'a> {
    Box::new(move || {
        let (tl, bm, bn) = tile;
        let sym = a.cfg.sym(tl, layout, bf16);
        let name = format!("{} chain:{PROD_SYM}->{sym} {}", a.cfg.arch, sh.label);
        let (po, go) = match (&a.prod, &a.gemm) {
            (Ok(p), Ok(g)) => (p, g),
            (Err(e), _) | (_, Err(e)) => return vec![(name, Err(e.clone()))],
        };
        let t0 = Instant::now();
        let r = (|| -> Result<String, String> {
            let (_, xq, xs) = run_producer(po, x, d.n, d.k)?;
            let s = check_quant(&xq, &xs, rq, d.n, d.k);
            if s.starts_with("FAIL") {
                return Ok(format!("FAIL producer stage: {}", &s[5..]));
            }
            let (y, _) = run_gemm(go, &sym, bm, bn, d, wblob, BUF_A, &xq, &xs)?;
            let s = compare_f32(&y, want, d.m)?;
            Ok(if s.starts_with("FAIL") {
                s
            } else {
                format!("{s} (producer exact, {:.1}s)", t0.elapsed().as_secs_f64())
            })
        })();
        vec![(name, r)]
    })
}

// ------------------------------------------------------------------ controls
/// Controls proving the gate can fail: the comparators flag a one-bit error, the references of different associations are
/// distinguishable, and a real emulated kernel reacts to a corrupted input exactly where the data flows. A control that does
/// not discriminate is a `FAIL` (the gate would be vacuous).
fn controls(t: &mut Tally, archs: &[ArchObjs]) {
    let d = gen_gemm(64, 256, 64, 77, false);
    let want = gemm_ref(&d, &d.xq, &d.xs, false);
    // 1. compare_f32 sees a single flipped mantissa bit.
    let mut got = want.clone();
    got[1000] = f32::from_bits(got[1000].to_bits() ^ 1);
    let c1 = compare_f32(&got, &want, d.m)
        .unwrap()
        .starts_with("FAIL 1 of");
    // 2. the K128 association and a K256 association differ (a kernel with the wrong fold granularity cannot pass).
    let other = gemm_ref_k256(&d, false);
    let differ = (0..want.len())
        .filter(|&i| want[i].to_bits() != other[i].to_bits())
        .count();
    // 3. check_quant sees a flipped code and a flipped scale bit.
    let x = gen_x(2, 256, 5, false);
    let r = quant_ref(&x, 2, 256);
    let mut xq: Vec<u8> = r.q.iter().map(|&v| v as u8).collect();
    let xs: Vec<u8> = r.s.iter().flat_map(|v| v.to_le_bytes()).collect();
    let clean = check_quant(&xq, &xs, &r, 2, 256);
    xq[300] ^= 0x40;
    let bad_q = check_quant(&xq, &xs, &r, 2, 256);
    xq[300] ^= 0x40;
    let mut xs_bad = xs.clone();
    xs_bad[4] ^= 1;
    let bad_s = check_quant(&xq, &xs_bad, &r, 2, 256);
    let c3 = !clean.starts_with("FAIL")
        && bad_q.starts_with("FAIL codes 1/")
        && bad_s.starts_with("FAIL codes 0/2 groups differ, scales 1/");
    // 4. bf16 rounding is exercised: the bf16 reference differs from the F32 one and is a fixed point of bf16_rn.
    let want16 = gemm_ref(&d, &d.xq, &d.xs, true);
    let c4 = want16
        .iter()
        .zip(&want)
        .any(|(a, b)| a.to_bits() != b.to_bits())
        && want16.iter().all(|v| bf16_rn(*v).to_bits() == v.to_bits());
    for (name, ok) in [
        ("compare_f32 flags a 1-bit error", c1),
        ("K128 and K256 fold references differ", differ > 0),
        ("check_quant flags a corrupted code / scale bit", c3),
        ("bf16 reference differs from F32 and is bf16-exact", c4),
    ] {
        if ok {
            t.pass(&format!("control: {name}"), "");
        } else {
            t.fail(&format!("control: {name}"), "did not discriminate");
        }
    }
    // 5. an emulated GEMM reacts to a corrupted input exactly where the data flows (per arch, smallest tile, layout B).
    for a in archs {
        let name = format!(
            "{} control: emulated GEMM reacts only where the corrupted data flows",
            a.cfg.arch
        );
        if !t.wanted(&name) {
            continue;
        }
        let Ok(obj) = &a.gemm else {
            t.unrun(&name, a.gemm.as_ref().err().unwrap());
            continue;
        };
        let (tl, bm, bn) = a.cfg.tiles[2];
        let sym = a.cfg.sym(tl, "wb", false);
        let go = |dd: &GData| {
            run_gemm(
                obj,
                &sym,
                bm,
                bn,
                dd,
                &layout_b(dd),
                BUF_A,
                &i8_bytes(&dd.xq),
                &f32_bytes(&dd.xs),
            )
        };
        let res = (|| -> Result<bool, String> {
            let (base, _) = go(&d)?;
            if compare_f32(&base, &want, d.m)?.starts_with("FAIL") {
                return Err("OVERRUN: control baseline is not exact".into());
            }
            let diff = |y: &[f32]| -> Vec<usize> {
                (0..y.len())
                    .filter(|&i| y[i].to_bits() != base[i].to_bits())
                    .collect()
            };
            // W code of row 9 at k = 130
            let mut dw = gen_gemm(64, 256, 64, 77, false);
            dw.wq[9 * 256 + 130] = dw.wq[9 * 256 + 130].wrapping_add(40);
            let dwd = diff(&go(&dw)?.0);
            // X code of token 5 at k = 200
            let mut dx = gen_gemm(64, 256, 64, 77, false);
            dx.xq[5 * 256 + 200] = dx.xq[5 * 256 + 200].wrapping_add(40);
            let dxd = diff(&go(&dx)?.0);
            // W scale of row 3
            let mut ds = gen_gemm(64, 256, 64, 77, false);
            ds.ws[3] ^= 0x0100;
            let dsd = diff(&go(&ds)?.0);
            // a weight buffer 16 bytes short makes the kernel read into the strict-read pad: flagged as OVERRUN
            let wfull = layout_b(&d);
            let over = run_gemm(
                obj,
                &sym,
                bm,
                bn,
                &d,
                &wfull[..wfull.len() - 16],
                BUF_A,
                &i8_bytes(&d.xq),
                &f32_bytes(&d.xs),
            );
            let over_ok = matches!(&over, Err(e) if e.starts_with("OVERRUN"));
            Ok(over_ok
                && !dwd.is_empty()
                && dwd.iter().all(|&i| i % d.m == 9)
                && !dxd.is_empty()
                && dxd.iter().all(|&i| i / d.m == 5)
                && !dsd.is_empty()
                && dsd.iter().all(|&i| i % d.m == 3))
        })();
        match res {
            Ok(true) => t.pass(&name, ""),
            Ok(false) => t.fail(&name, "did not discriminate"),
            Err(e) => t.result(&name, Err(e)),
        }
    }
}

// ------------------------------------------------------------------ the test
#[test]
fn mq8_prefill_kernels_equal_the_cpu_reference_on_the_emulator() {
    let only: Vec<String> = std::env::var("MQ8_EMU_ONLY")
        .ok()
        .map(|s| {
            s.split(',')
                .filter(|x| !x.is_empty())
                .map(String::from)
                .collect()
        })
        .unwrap_or_default();
    let hipcc = std::env::var("MQ8_EMU_HIPCC").unwrap_or_else(|_| DEFAULT_HIPCC.to_string());
    if !Path::new(&hipcc).exists() {
        println!("EMU SKIP hipcc not found at {hipcc}: cannot compile the MQ8 prefill kernels (set MQ8_EMU_HIPCC)");
        return;
    }
    let out = out_dir();
    std::fs::create_dir_all(&out).expect("create MQ8_EMU_OUT");
    // a `gfx1100` / `gfx1201` substring in the filter restricts the compiles to the named arch(es)
    let arch_wanted = |arch: &str| {
        let named = only.iter().filter(|s| s.contains("gfx")).count() > 0;
        !named || only.iter().any(|s| s.contains(arch))
    };
    let cfgs: Vec<&'static ArchCfg> = [&GFX1201, &GFX1100]
        .into_iter()
        .filter(|c| arch_wanted(c.arch))
        .collect();

    // 1. compile (parallel, nice 19) and lift
    let t_build = Instant::now();
    let archs: Vec<ArchObjs> = std::thread::scope(|s| {
        let hs: Vec<_> = cfgs
            .iter()
            .map(|&cfg| {
                let (hipcc, out) = (&hipcc, &out);
                let p = s.spawn(move || compile(hipcc, PROD_SRC, cfg.arch, out));
                let g = s.spawn(move || compile(hipcc, cfg.gemm_src, cfg.arch, out));
                (cfg, p, g)
            })
            .collect();
        hs.into_iter()
            .map(|(cfg, p, g)| {
                let lift = |r: Result<PathBuf, String>| r.and_then(|co| lift_co(&co, cfg.arch));
                ArchObjs {
                    cfg,
                    prod: lift(p.join().unwrap()),
                    gemm: lift(g.join().unwrap()),
                }
            })
            .collect()
    });
    for a in &archs {
        for (what, o) in [("producer", &a.prod), ("gemm", &a.gemm)] {
            match o {
                Ok(o) => println!(
                    "LIFT ok {} {what}: {} kernels",
                    a.cfg.arch,
                    o.prog.kernels.len()
                ),
                Err(e) => println!(
                    "LIFT FAILED {} {what}: {}",
                    a.cfg.arch,
                    e.lines().next().unwrap_or("")
                ),
            }
        }
    }
    println!("build+lift {:.1}s", t_build.elapsed().as_secs_f64());

    let tally = Mutex::new(Tally {
        pass: 0,
        fail: 0,
        unrun: 0,
        unrun_why: BTreeMap::new(),
        only: only.clone(),
    });
    let wanted = |name: &str| tally.lock().unwrap().wanted(name);

    // 2. data (shared by every arch)
    let shapes = [
        Shape {
            m: 128,
            k: 512,
            n: 128,
            extreme: false,
            off2: true,
            seed: 11,
            label: "M128 K512 N128",
        },
        Shape {
            m: 130,
            k: 512,
            n: 70,
            extreme: false,
            off2: true,
            seed: 12,
            label: "M130 K512 N70",
        },
        Shape {
            m: 70,
            k: 768,
            n: 130,
            extreme: false,
            off2: true,
            seed: 13,
            label: "M70 K768 N130",
        },
        Shape {
            m: 40,
            k: 256,
            n: 20,
            extreme: false,
            off2: false,
            seed: 14,
            label: "M40 K256 N20",
        },
        Shape {
            m: 128,
            k: 512,
            n: 128,
            extreme: true,
            off2: false,
            seed: 15,
            label: "M128 K512 N128 extreme",
        },
        Shape {
            m: 264,
            k: 1024,
            n: 200,
            extreme: false,
            off2: true,
            seed: 16,
            label: "M264 K1024 N200 multi-tile",
        },
        Shape {
            m: 130,
            k: 2560,
            n: 70,
            extreme: false,
            off2: false,
            seed: 17,
            label: "M130 K2560 N70 long-K",
        },
    ];
    let gdata: Vec<GData> = shapes
        .iter()
        .map(|s| gen_gemm(s.m, s.k, s.n, s.seed, s.extreme))
        .collect();
    let wa: Vec<Vec<u8>> = gdata.iter().map(layout_a).collect();
    let wb: Vec<Vec<u8>> = gdata.iter().map(layout_b).collect();
    let wants: Vec<[Vec<f32>; 2]> = gdata
        .iter()
        .map(|d| {
            [
                gemm_ref(d, &d.xq, &d.xs, false),
                gemm_ref(d, &d.xq, &d.xs, true),
            ]
        })
        .collect();

    // chains: producer -> GEMM (edge groups in X)
    let chain_shapes = [
        Shape {
            m: 130,
            k: 512,
            n: 70,
            extreme: false,
            off2: false,
            seed: 21,
            label: "M130 K512 N70 edge-groups",
        },
        Shape {
            m: 70,
            k: 768,
            n: 33,
            extreme: false,
            off2: false,
            seed: 22,
            label: "M70 K768 N33",
        },
    ];
    let cdata: Vec<GData> = chain_shapes
        .iter()
        .map(|s| gen_gemm(s.m, s.k, s.n, s.seed, false))
        .collect();
    let cx: Vec<Vec<f32>> = chain_shapes
        .iter()
        .map(|s| gen_x(s.n, s.k, s.seed + 1000, true))
        .collect();
    let cq: Vec<QRef> = chain_shapes
        .iter()
        .zip(&cx)
        .map(|(s, x)| quant_ref(x, s.n, s.k))
        .collect();
    // reference GEMM operands come from the CPU quantizer
    let cref: Vec<[Vec<f32>; 2]> = cdata
        .iter()
        .zip(&cq)
        .map(|(d, q)| {
            let xs: Vec<f32> = q.s.iter().map(|&b| f32::from_bits(b)).collect();
            [gemm_ref(d, &q.q, &xs, false), gemm_ref(d, &q.q, &xs, true)]
        })
        .collect();
    let cwa: Vec<Vec<u8>> = cdata.iter().map(layout_a).collect();
    let cwb: Vec<Vec<u8>> = cdata.iter().map(layout_b).collect();

    // 3. jobs
    let mut jobs: Vec<Job<'_>> = Vec::new();
    for a in &archs {
        for (n, k, seed, edges, label) in [
            (3usize, 256usize, 31u64, false, "N3 K256"),
            (9, 768, 32, false, "N9 K768"),
            (40, 2560, 34, false, "N40 K2560"),
            (6, 512, 33, true, "N6 K512 zero/large/tiny/int/ones groups"),
        ] {
            if wanted(&format!("{} {PROD_SYM} {label}", a.cfg.arch)) {
                jobs.push(producer_job(a, n, k, seed, edges, label));
            }
        }
        for (si, sh) in shapes.iter().enumerate() {
            for &tile in a.cfg.tiles {
                for bf16 in [false, true] {
                    let any = ["wa", "wb"].iter().any(|lay| {
                        wanted(&format!(
                            "{} {} {}",
                            a.cfg.arch,
                            a.cfg.sym(tile.0, lay, bf16),
                            sh.label
                        ))
                    }) || wanted(&format!(
                        "{} {}{}{} wa==wb {}",
                        a.cfg.arch,
                        a.cfg.prefix,
                        tile.0,
                        if bf16 { "_bf16" } else { "" },
                        sh.label
                    ));
                    if any {
                        jobs.push(gemm_job(
                            a,
                            tile,
                            bf16,
                            sh,
                            &gdata[si],
                            &wants[si][bf16 as usize],
                            &wa[si],
                            &wb[si],
                        ));
                    }
                }
            }
        }
        for (ci, sh) in chain_shapes.iter().enumerate() {
            for &tile in a.cfg.tiles {
                for (layout, bf16) in [("wa", false), ("wb", true)] {
                    let sym = a.cfg.sym(tile.0, layout, bf16);
                    if wanted(&format!(
                        "{} chain:{PROD_SYM}->{sym} {}",
                        a.cfg.arch, sh.label
                    )) {
                        let wblob = if layout == "wa" { &cwa[ci] } else { &cwb[ci] };
                        jobs.push(chain_job(
                            a,
                            tile,
                            layout,
                            bf16,
                            sh,
                            &cdata[ci],
                            &cx[ci],
                            &cq[ci],
                            &cref[ci][bf16 as usize],
                            wblob,
                        ));
                    }
                }
            }
        }
    }
    let njobs = jobs.len();
    let t_run = Instant::now();
    run_jobs(jobs, &tally);
    println!(
        "emulated {njobs} jobs in {:.1}s",
        t_run.elapsed().as_secs_f64()
    );

    // 4. controls and tally
    let mut t = tally.into_inner().unwrap();
    if only.is_empty() {
        controls(&mut t, &archs);
    }
    println!(
        "EMU TALLY pass={} fail={} unrun={}",
        t.pass, t.fail, t.unrun
    );
    let mut by_reason: BTreeMap<&String, Vec<&String>> = BTreeMap::new();
    for (k, v) in &t.unrun_why {
        by_reason.entry(v).or_default().push(k);
    }
    for (why, names) in &by_reason {
        println!("EMU UNRUN-SUMMARY {} entries: {why}", names.len());
    }
    assert_eq!(
        t.fail, 0,
        "{} kernel/shape pairs FAILED bit-exact comparison",
        t.fail
    );
    if std::env::var("MQ8_EMU_STRICT").as_deref() == Ok("1") {
        assert_eq!(
            t.unrun, 0,
            "{} kernels could not run on the emulator",
            t.unrun
        );
    }
    assert!(t.pass > 0, "no kernel ran at all");
}
