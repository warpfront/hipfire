// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx942 DS_OR_B32 / DS_OR_RTN_B32 hardware/interpreter oracle. Three wirings
//! (no-return, return with separate destination, return with destination == data),
//! seven DS byte offsets (including an LDS-allocation-end window), four address
//! patterns (distinct permutation, all-lanes-same, four-address, lane-pairs), four
//! data families, six EXEC masks (full, sparse, bit63, empty, lane0, high half) and two
//! poisons for LDS and global output. Every kernel's DS instruction is followed by an
//! explicit `s_waitcnt lgkmcnt(0)` in the same asm block; the generated code object is
//! disassembled and the instruction, offset, register wiring and wait are verified,
//! and the codec's encoding of that exact instruction must appear in the object.
//! Distinct-address lanes are compared exactly against both the simulator and an
//! independent `old | operand` model. Same-address lanes: the final LDS is compared
//! exactly; the returned values are only checked for being a legal atomic
//! linearization (hardware order is unspecified, the simulator's ascending-lane
//! serialization is not compared). Planted corruption controls must be detected.
//! Run only in a fabric job:
//! `cargo run -p peacemaker-isa --features lab --example cdna_ds_or_hw_oracle -- OUT_DIR`.
use hip_bridge::HipRuntime;
use peacemaker_cdna::enc::ds::{Ds, DsOp};
use peacemaker_cdna::enc::sop::{Sopp, SoppOp};
use peacemaker_cdna::sim::{exec_one, Config, Memory, Wave};
use peacemaker_cdna::{Arch, Inst, Src};
use peacemaker_ir::envelope::Bundle;
use peacemaker_lift::bundle::BundleCodec;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

const SEED: u64 = 0x5d0e_91b7_3c42_a6f9;
const GUARD: usize = 64;
const LDS_WORDS: usize = 8192;
/// Window words: 16 guard + 64 data + 16 guard; window starts at byte `off`.
const WIN: usize = 96;
const BIAS_BYTES: u32 = 64;
/// Input words per case: exec_lo, exec_hi, poison, pad, addr[64], op[64], ret0[64], window[96].
const IN_STRIDE: usize = 4 + 64 * 3 + WIN;
/// Output words per case: ret[64], window[96], outside-window mismatch count.
const OUT_STRIDE: usize = 64 + WIN + 1;
/// Last window ends exactly at the 32 KiB allocation end.
const OFFSETS: [u32; 7] = [0, 4, 252, 1024, 4096, 16380, 32384];
const POISONS: [u8; 2] = [0x5a, 0xa5];
const WAITCNT_LGKM0: u16 = 0xc07f;
const EXECS: [u64; 6] = [
    u64::MAX,
    0x5555_5555_5555_5555,
    1 << 63,
    0,
    1,
    0xffff_ffff_0000_0000,
];
const EDGES: [u32; 16] = [
    0,
    1,
    2,
    u32::MAX,
    u32::MAX - 1,
    1 << 31,
    (1 << 31) - 1,
    0xffff_0000,
    0x0000_ffff,
    0xaaaa_aaaa,
    0x5555_5555,
    0x8000_0001,
    0x0123_4567,
    0x89ab_cdef,
    1 << 16,
    0xdead_beef,
];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Form {
    NoRtn,
    Rtn,
    RtnAlias,
}
impl Form {
    const ALL: [Form; 3] = [Form::NoRtn, Form::Rtn, Form::RtnAlias];
    fn tag(self) -> &'static str {
        match self {
            Form::NoRtn => "nortn",
            Form::Rtn => "rtn",
            Form::RtnAlias => "rtn_alias",
        }
    }
    fn mnemonic(self) -> &'static str {
        match self {
            Form::NoRtn => "ds_or_b32",
            _ => "ds_or_rtn_b32",
        }
    }
    fn op(self) -> DsOp {
        match self {
            Form::NoRtn => DsOp::OrB32,
            _ => DsOp::OrRtnB32,
        }
    }
}
fn kname(f: Form, off: u32) -> String {
    format!("dsor_{}_o{off}", f.tag())
}

fn sha(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}
fn bytes(x: &[u32]) -> Vec<u8> {
    x.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn e<E: std::fmt::Display>(x: E) -> String {
    x.to_string()
}

fn source() -> String {
    let mut s = String::from(
        "#include <hip/hip_runtime.h>\ntypedef __attribute__((address_space(3))) unsigned lds_u32;\n",
    );
    for f in Form::ALL {
        for off in OFFSETS {
            let ds = match f {
                Form::NoRtn => format!("ds_or_b32 %2, %3 offset:{off}"),
                Form::Rtn => format!("ds_or_rtn_b32 %1, %3, %4 offset:{off}"),
                Form::RtnAlias => format!("ds_or_rtn_b32 %1, %3, %1 offset:{off}"),
            };
            // Rtn forms: %0 saved exec, %1 return/alias register, %2 exec mask, %3 address, %4 data.
            // No-return: %0 saved exec, %1 exec mask, %2 address, %3 data.
            let asm = match f {
                Form::NoRtn => format!(
                    "asm volatile(\"s_mov_b64 %0, exec\\n\\ts_mov_b64 exec, %1\\n\\t{ds}\\n\\ts_waitcnt lgkmcnt(0)\\n\\ts_mov_b64 exec, %0\" : \"=&s\"(sv) : \"s\"(mask), \"v\"(addr), \"v\"(opnd) : \"memory\");"
                ),
                Form::Rtn => format!(
                    "asm volatile(\"s_mov_b64 %0, exec\\n\\ts_mov_b64 exec, %2\\n\\t{ds}\\n\\ts_waitcnt lgkmcnt(0)\\n\\ts_mov_b64 exec, %0\" : \"=&s\"(sv), \"+&v\"(r) : \"s\"(mask), \"v\"(addr), \"v\"(opnd) : \"memory\");"
                ),
                Form::RtnAlias => format!(
                    "asm volatile(\"s_mov_b64 %0, exec\\n\\ts_mov_b64 exec, %2\\n\\t{ds}\\n\\ts_waitcnt lgkmcnt(0)\\n\\ts_mov_b64 exec, %0\" : \"=&s\"(sv), \"+&v\"(r) : \"s\"(mask), \"v\"(addr) : \"memory\");"
                ),
            };
            let init_r = match f {
                Form::NoRtn => "unsigned r = 0;",
                Form::Rtn => "unsigned r = p[132 + t];",
                Form::RtnAlias => "unsigned r = opnd;",
            };
            s.push_str(&format!(
                "extern \"C\" __global__ void {name}(const unsigned* P, unsigned* O) {{
__shared__ unsigned lds[{LDS_WORDS}];
const unsigned* p = P + blockIdx.x * {IN_STRIDE}u;
unsigned* o = O + blockIdx.x * {OUT_STRIDE}u;
const unsigned t = threadIdx.x;
const unsigned poison = p[2];
const unsigned w0 = {w0}u;
for (unsigned i = t; i < {LDS_WORDS}u; i += 64) lds[i] = poison;
__syncthreads();
for (unsigned i = t; i < {WIN}u; i += 64) lds[w0 + i] = p[196 + i];
__syncthreads();
unsigned long long mask = (unsigned long long)p[0] | ((unsigned long long)p[1] << 32);
unsigned long long sv;
unsigned opnd = p[68 + t];
unsigned addr = (unsigned)(size_t)(lds_u32*)lds + p[4 + t];
{init_r}
{asm}
__syncthreads();
o[t] = r;
for (unsigned i = t; i < {WIN}u; i += 64) o[64 + i] = lds[w0 + i];
unsigned bad = 0;
for (unsigned i = t; i < {LDS_WORDS}u; i += 64) if ((i < w0 || i >= w0 + {WIN}u) && lds[i] != poison) bad++;
for (int m = 32; m; m >>= 1) bad += __shfl_xor(bad, m);
if (t == 0) o[{OUT_STRIDE} - 1] = bad;
}}
",
                name = kname(f, off),
                w0 = off / 4
            ));
        }
    }
    s
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn u32(&mut self) -> u32 {
        (self.next() >> 16) as u32
    }
}

#[derive(Clone)]
struct Case {
    exec: u64,
    addr: [u32; 64],
    op: [u32; 64],
    ret0: [u32; 64],
    win: [u32; WIN],
    addr_pat: usize,
    data_kind: usize,
}
impl Case {
    fn active(&self, l: usize) -> bool {
        self.exec >> l & 1 != 0
    }
    /// Word index inside the window of lane `l`'s target.
    fn widx(&self, l: usize) -> usize {
        (self.addr[l] / 4) as usize
    }
}

fn cases_for(kidx: u64) -> Vec<Case> {
    let mut rng = Rng(SEED ^ kidx.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
    let mut v = Vec::new();
    for addr_pat in 0..4 {
        for data_kind in 0..4 {
            for &exec in &EXECS {
                let seed = rng.u32() as usize;
                let mut c = Case {
                    exec,
                    addr: [0; 64],
                    op: [0; 64],
                    ret0: [0; 64],
                    win: [0; WIN],
                    addr_pat,
                    data_kind,
                };
                for l in 0..64usize {
                    let wordi = match addr_pat {
                        0 => (l * 37 + 11) & 63,
                        1 => 13,
                        2 => (l & 3) * 5,
                        _ => l >> 1,
                    };
                    c.addr[l] = BIAS_BYTES + 4 * wordi as u32;
                    c.op[l] = match data_kind {
                        0 => EDGES[(l * 5 + seed) % 16],
                        1 => rng.u32(),
                        2 => 1u32 << (l % 32),
                        _ => match l % 4 {
                            0 => 0,
                            1 => u32::MAX,
                            _ => rng.u32(),
                        },
                    };
                    c.ret0[l] = rng.u32() ^ 0x13572468;
                }
                for i in 0..WIN {
                    c.win[i] = if i < 16 {
                        0xc0ff_ee00 + i as u32
                    } else if i >= 80 {
                        0xfeed_face ^ i as u32
                    } else {
                        match data_kind {
                            0 | 3 => EDGES[(i * 3 + seed + 1) % 16],
                            1 => rng.u32(),
                            _ => {
                                if i % 3 == 0 {
                                    0
                                } else {
                                    rng.u32()
                                }
                            }
                        }
                    };
                }
                v.push(c);
            }
        }
    }
    v
}

fn input_words(cases: &[Case], poison: u32) -> Vec<u32> {
    let mut w = Vec::with_capacity(cases.len() * IN_STRIDE);
    for c in cases {
        w.extend([c.exec as u32, (c.exec >> 32) as u32, poison, 0]);
        w.extend(c.addr);
        w.extend(c.op);
        w.extend(c.ret0);
        w.extend(c.win);
    }
    w
}

/// Independent expected results: final window and per-lane old values (active lanes,
/// ascending-lane serialization; only used as exact truth for singleton-address lanes).
fn independent(c: &Case) -> (Vec<u32>, [u32; 64]) {
    let mut win = c.win.to_vec();
    let mut old = [0; 64];
    for l in 0..64 {
        if c.active(l) {
            let i = c.widx(l);
            old[l] = win[i];
            win[i] |= c.op[l];
        }
    }
    (win, old)
}

struct SimOut {
    ret: [u32; 64],
    win: Vec<u32>,
    outside_bad: u32,
}
fn sim(form: Form, off: u32, c: &Case, poison: u32) -> Result<SimOut, String> {
    let (o0, o1) = Ds::split_offset(off).map_err(e)?;
    let (vdst, data) = match form {
        Form::NoRtn => (None, 2),
        Form::Rtn => (Some(Src::Vgpr(3)), 2),
        Form::RtnAlias => (Some(Src::Vgpr(2)), 2),
    };
    let ds = Inst::Ds(Ds {
        op: form.op(),
        vdst,
        addr: Some(Src::Vgpr(1)),
        data0: Some(Src::Vgpr(data)),
        data1: None,
        offset0: o0,
        offset1: o1,
        gds: false,
    });
    let wait = Inst::Sopp(Sopp {
        op: SoppOp::Waitcnt,
        simm16: WAITCNT_LGKM0,
    });
    let mut w = Wave::new(Arch::Gfx942, 0, 0xbaad_f00d);
    w.exec = c.exec;
    for l in 0..64 {
        w.v[1][l] = c.addr[l];
        w.v[2][l] = c.op[l];
        w.v[3][l] = c.ret0[l];
    }
    let mut lds = bytes(&vec![poison; LDS_WORDS]);
    let base = off as usize;
    lds[base..base + WIN * 4].copy_from_slice(&bytes(&c.win));
    let cfg = Config::default();
    let mut mem = Memory::default();
    exec_one(&mut w, &ds, &mut lds, &mut mem, &cfg).map_err(|x| format!("sim ds: {x}"))?;
    exec_one(&mut w, &wait, &mut lds, &mut mem, &cfg).map_err(|x| format!("sim wait: {x}"))?;
    let word = |i: usize| u32::from_le_bytes(lds[i * 4..i * 4 + 4].try_into().unwrap());
    let w0 = base / 4;
    let win: Vec<u32> = (0..WIN).map(|i| word(w0 + i)).collect();
    let outside_bad = (0..LDS_WORDS)
        .filter(|&i| (i < w0 || i >= w0 + WIN) && word(i) != poison)
        .count() as u32;
    let mut ret = [0; 64];
    for (l, r) in ret.iter_mut().enumerate() {
        *r = match form {
            Form::NoRtn => 0,
            _ => {
                w.v[usize::from(match vdst {
                    Some(Src::Vgpr(n)) => n,
                    _ => 0,
                })][l]
            }
        };
    }
    Ok(SimOut {
        ret,
        win,
        outside_bad,
    })
}

/// Is there an ordering of `lanes` (operand, returned) from initial value `init` such
/// that every returned value is `init | operands of the lanes ordered before`?
fn linearizable(init: u32, lanes: &[(u32, u32)]) -> bool {
    let mut cur = init;
    let mut rem = lanes.to_vec();
    while !rem.is_empty() {
        // An operand already contained in `cur` never changes the state: always safe.
        if let Some(k) = rem.iter().position(|&(o, r)| r == cur && o | cur == cur) {
            rem.swap_remove(k);
            continue;
        }
        let ch: Vec<usize> = (0..rem.len()).filter(|&i| rem[i].1 == cur).collect();
        if ch.len() != 1 {
            return false;
        }
        cur |= rem[ch[0]].0;
        rem.swap_remove(ch[0]);
    }
    true
}

#[derive(Default, Clone)]
struct Stats {
    cases: u64,
    exact_lanes: u64,
    inactive_lanes: u64,
    contended_groups: u64,
    contended_lanes: u64,
    empty_exec_cases: u64,
}

/// Verify one case's hardware output (`OUT_STRIDE` words) against expectations.
fn verify(
    form: Form,
    c: &Case,
    ind_win: &[u32],
    ind_old: &[u32; 64],
    got: &[u32],
    st: &mut Stats,
) -> Result<(), String> {
    if got[64..64 + WIN] != *ind_win {
        let i = (0..WIN).find(|&i| got[64 + i] != ind_win[i]).unwrap();
        return Err(format!(
            "final LDS word {i}: got {:#010x}, want {:#010x}",
            got[64 + i],
            ind_win[i]
        ));
    }
    if got[OUT_STRIDE - 1] != 0 {
        return Err(format!(
            "{} LDS words outside the window were modified",
            got[OUT_STRIDE - 1]
        ));
    }
    let mut groups: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for l in 0..64 {
        if c.active(l) {
            groups.entry(c.addr[l]).or_default().push(l);
        } else {
            let want = match form {
                Form::NoRtn => 0,
                Form::Rtn => c.ret0[l],
                Form::RtnAlias => c.op[l],
            };
            if got[l] != want {
                return Err(format!(
                    "inactive lane {l}: register changed to {:#010x}, want {want:#010x}",
                    got[l]
                ));
            }
            st.inactive_lanes += 1;
        }
    }
    if form == Form::NoRtn {
        for l in 0..64 {
            if got[l] != 0 {
                return Err(format!("no-return lane {l} register {:#010x}", got[l]));
            }
        }
    } else {
        for (addr, lanes) in &groups {
            if let [l] = lanes[..] {
                if got[l] != ind_old[l] {
                    return Err(format!(
                        "lane {l} returned {:#010x}, want {:#010x}",
                        got[l], ind_old[l]
                    ));
                }
                st.exact_lanes += 1;
            } else {
                let init = c.win[(addr / 4) as usize];
                let v: Vec<(u32, u32)> = lanes.iter().map(|&l| (c.op[l], got[l])).collect();
                if !linearizable(init, &v) {
                    return Err(format!(
                        "address {addr:#x}: returned values {:x?} are not a legal OR linearization of init {init:#010x}, operands {:x?}",
                        lanes.iter().map(|&l| got[l]).collect::<Vec<_>>(),
                        lanes.iter().map(|&l| c.op[l]).collect::<Vec<_>>()
                    ));
                }
                st.contended_groups += 1;
                st.contended_lanes += lanes.len() as u64;
            }
        }
    }
    if c.exec == 0 {
        st.empty_exec_cases += 1;
    }
    st.cases += 1;
    Ok(())
}

fn guards_ok(got: &[u8], poison: u8) -> bool {
    let n = got.len();
    got[..GUARD]
        .iter()
        .chain(&got[n - GUARD..])
        .all(|&v| v == poison)
}

struct Disasm {
    regs: Vec<String>,
    offset: u32,
    wait_ok: bool,
    words: Vec<u32>,
}
/// Disassemble the code object and bind each kernel to its requested DS instruction.
fn bind_object(co_path: &Path, code: &[u8]) -> Result<(String, BTreeMap<String, Disasm>), String> {
    let out = Command::new("/opt/rocm/llvm/bin/llvm-objdump")
        .args([
            "-d",
            "--mcpu=gfx942",
            co_path.to_str().ok_or("non-UTF8 path")?,
        ])
        .output()
        .map_err(e)?;
    if !out.status.success() {
        return Err(format!(
            "llvm-objdump: {}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    let mut per: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut cur: Option<String> = None;
    for line in text.lines() {
        let t = line.trim();
        if t.ends_with(">:") {
            if let Some(i) = t.rfind('<') {
                let n = &t[i + 1..t.len() - 2];
                if n.starts_with("dsor_") && !n.ends_with(".kd") {
                    cur = Some(n.to_string());
                }
            }
            continue;
        }
        let Some(k) = &cur else { continue };
        let ins = t
            .split("//")
            .next()
            .unwrap_or("")
            .split(';')
            .next()
            .unwrap_or("")
            .trim();
        if !ins.is_empty() {
            per.entry(k.clone()).or_default().push(ins.to_string());
        }
    }
    let code_words: Vec<u32> = code
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let mut res = BTreeMap::new();
    for f in Form::ALL {
        for off in OFFSETS {
            let name = kname(f, off);
            let lines = per
                .get(&name)
                .ok_or_else(|| format!("{name}: not found in disassembly"))?;
            let dsl: Vec<usize> = (0..lines.len())
                .filter(|&i| lines[i].starts_with("ds_or_"))
                .collect();
            if dsl.len() != 1 || !lines[dsl[0]].starts_with(&format!("{} ", f.mnemonic())) {
                return Err(format!(
                    "{name}: expected exactly one {}, found DS lines {:?}",
                    f.mnemonic(),
                    dsl.iter().map(|&i| &lines[i]).collect::<Vec<_>>()
                ));
            }
            let l = &lines[dsl[0]];
            let mut regs = Vec::new();
            let mut offset = 0u32;
            for tok in l.split(|ch: char| ch == ',' || ch.is_whitespace()).skip(1) {
                if let Some(n) = tok.strip_prefix("offset:") {
                    offset = n
                        .parse()
                        .map_err(|_| format!("{name}: bad offset in `{l}`"))?;
                } else if !tok.is_empty() {
                    regs.push(tok.to_string());
                }
            }
            if offset != off {
                return Err(format!("{name}: offset {offset} != {off} in `{l}`"));
            }
            let want_n = if f == Form::NoRtn { 2 } else { 3 };
            if regs.len() != want_n {
                return Err(format!("{name}: operands {regs:?} in `{l}`"));
            }
            if f == Form::RtnAlias && regs[0] != regs[2] {
                return Err(format!("{name}: destination != data in `{l}`"));
            }
            if f == Form::Rtn && (regs[0] == regs[1] || regs[0] == regs[2]) {
                return Err(format!("{name}: destination aliases an input in `{l}`"));
            }
            let wait_ok = lines
                .get(dsl[0] + 1)
                .is_some_and(|n| n.starts_with("s_waitcnt") && n.contains("lgkmcnt(0)"));
            if !wait_ok {
                return Err(format!(
                    "{name}: DS op not immediately followed by s_waitcnt lgkmcnt(0): {:?}",
                    lines.get(dsl[0] + 1)
                ));
            }
            let vn = |s: &str| -> Result<u8, String> {
                s.strip_prefix('v')
                    .and_then(|n| n.parse().ok())
                    .ok_or_else(|| format!("{name}: unparsable VGPR `{s}`"))
            };
            let (vdst, addr, data) = match f {
                Form::NoRtn => (None, vn(&regs[0])?, vn(&regs[1])?),
                _ => (Some(vn(&regs[0])?), vn(&regs[1])?, vn(&regs[2])?),
            };
            let (o0, o1) = Ds::split_offset(off).map_err(e)?;
            let inst = Inst::Ds(Ds {
                op: f.op(),
                vdst: vdst.map(Src::Vgpr),
                addr: Some(Src::Vgpr(addr)),
                data0: Some(Src::Vgpr(data)),
                data1: None,
                offset0: o0,
                offset1: o1,
                gds: false,
            });
            let words = inst
                .words(Arch::Gfx942)
                .map_err(|x| format!("{name}: codec encode `{l}`: {x}"))?;
            if !code_words.windows(words.len()).any(|w| w == &words[..]) {
                return Err(format!(
                    "{name}: codec words {words:?} for `{l}` absent from the code object"
                ));
            }
            res.insert(
                name,
                Disasm {
                    regs,
                    offset,
                    wait_ok,
                    words,
                },
            );
        }
    }
    Ok((sha(text.as_bytes()), res))
}

fn run(dir: &Path) -> Result<Value, String> {
    std::fs::create_dir_all(dir).map_err(e)?;
    let src = source();
    let src_path = dir.join("ds_or.hip");
    let co_path = dir.join("ds_or.hsaco");
    std::fs::write(&src_path, &src).map_err(e)?;
    let cc = Command::new("/opt/rocm/bin/hipcc")
        .args([
            "--genco",
            "--offload-arch=gfx942",
            "-O3",
            "--no-offload-compress",
            "-fuse-cuid=none",
            "-o",
            co_path.to_str().ok_or("non-UTF8 output path")?,
            src_path.to_str().ok_or("non-UTF8 source path")?,
        ])
        .output()
        .map_err(e)?;
    if !cc.status.success() {
        return Err(format!("hipcc: {}", String::from_utf8_lossy(&cc.stderr)));
    }
    let code = std::fs::read(&co_path).map_err(e)?;
    // hipcc --genco can return an offload bundle even with a .hsaco suffix.
    // Use the existing bundle codec and bind/disassemble/load the exact selected ELF.
    let native = if Bundle::is_bundle(&code) {
        let (bundle, payloads) = Bundle::read(&code).map_err(e)?;
        let triple = format!(
            "{}gfx942",
            peacemaker_lift::bundle::HIP_DEVICE_TRIPLE_PREFIX
        );
        let mut matches = bundle.entries.iter().enumerate().filter(|(_, entry)| {
            entry.triple == triple
                || entry
                    .triple
                    .strip_prefix(&triple)
                    .is_some_and(|suffix| suffix.starts_with(':'))
        });
        let (idx, _) = matches.next().ok_or("bundle has no gfx942 device entry")?;
        if matches.next().is_some() {
            return Err("bundle has multiple gfx942 device entries".into());
        }
        payloads[idx]
    } else {
        code.as_slice()
    };
    if !native.starts_with(b"\x7fELF") {
        return Err("selected gfx942 device entry is not an ELF".into());
    }
    let native_path = dir.join("ds_or.device.elf");
    std::fs::write(&native_path, native).map_err(e)?;
    let (disasm_sha, bound) = bind_object(&native_path, native)?;

    let rt = HipRuntime::load().map_err(e)?;
    rt.set_device(0).map_err(e)?;
    let arch = rt.get_arch(0).map_err(e)?;
    if arch != "gfx942" {
        return Err(format!("requires gfx942, got {arch}"));
    }
    let module = rt.module_load_data(native).map_err(e)?;

    let ncases = cases_for(0).len();
    let din = rt.malloc(ncases * IN_STRIDE * 4).map_err(e)?;
    let total = GUARD * 2 + ncases * OUT_STRIDE * 4;
    let dout = rt.malloc(total).map_err(e)?;

    let mut rows = Vec::new();
    let mut st = Stats::default();
    let mut controls = 0u64;
    let mut hashes = Sha256::new();
    let mut kidx = 0u64;
    for f in Form::ALL {
        for off in OFFSETS {
            let name = kname(f, off);
            let func = rt.module_get_function(&module, &name).map_err(e)?;
            let cases = cases_for(kidx);
            kidx += 1;
            for poison in POISONS {
                let pw = u32::from_ne_bytes([poison; 4]);
                // Simulator + independent model; they must agree before hardware is judged.
                let mut exp = Vec::with_capacity(cases.len());
                for (ci, c) in cases.iter().enumerate() {
                    let (iw, io) = independent(c);
                    let s = sim(f, off, c, pw)?;
                    let tag = format!("{name} poison {poison:#x} case {ci}");
                    if s.win != iw {
                        return Err(format!("{tag}: simulator final LDS != independent model"));
                    }
                    if s.outside_bad != 0 {
                        return Err(format!("{tag}: simulator wrote outside the window"));
                    }
                    let mut single = std::collections::HashMap::new();
                    for l in 0..64 {
                        if c.active(l) {
                            *single.entry(c.addr[l]).or_insert(0u32) += 1;
                        }
                    }
                    for l in 0..64 {
                        let want = if !c.active(l) {
                            match f {
                                Form::NoRtn => 0,
                                Form::Rtn => c.ret0[l],
                                Form::RtnAlias => c.op[l],
                            }
                        } else if f == Form::NoRtn {
                            0
                        } else if single[&c.addr[l]] == 1 {
                            io[l]
                        } else {
                            continue; // hardware order unspecified: sim serialization not compared
                        };
                        if s.ret[l] != want {
                            return Err(format!(
                                "{tag}: simulator lane {l} returned {:#010x}, independent model {want:#010x}",
                                s.ret[l]
                            ));
                        }
                    }
                    exp.push((iw, io));
                }

                let inb = bytes(&input_words(&cases, pw));
                rt.memcpy_htod(&din, &inb).map_err(e)?;
                rt.memset(&dout, i32::from(poison), total).map_err(e)?;
                let mut blob = Vec::new();
                for p in [
                    din.as_ptr() as usize as u64,
                    dout.as_ptr() as usize as u64 + GUARD as u64,
                ] {
                    blob.extend(p.to_le_bytes());
                }
                // Same hidden dispatch layout as cdna_nor_hw_oracle; two aligned pointers.
                for x in [ncases as u32, 1, 1] {
                    blob.extend(x.to_le_bytes());
                }
                for x in [64u16, 1, 1] {
                    blob.extend(x.to_le_bytes());
                }
                blob.resize(256, 0);
                // SAFETY: two live allocations sized for `ncases` waves, full hidden argument storage.
                unsafe {
                    rt.launch_kernel_blob(
                        &func,
                        [ncases as u32, 1, 1],
                        [64, 1, 1],
                        0,
                        None,
                        &mut blob,
                    )
                }
                .map_err(e)?;
                rt.device_synchronize().map_err(e)?;
                let mut raw = vec![0u8; total];
                rt.memcpy_dtoh(&mut raw, &dout).map_err(e)?;
                hashes.update(&raw);
                let guards = guards_ok(&raw, poison);
                let payload: Vec<u32> = raw[GUARD..total - GUARD]
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                let before = st.clone();
                let mut first_err = None;
                for (ci, c) in cases.iter().enumerate() {
                    let g = &payload[ci * OUT_STRIDE..(ci + 1) * OUT_STRIDE];
                    if let Err(x) = verify(f, c, &exp[ci].0, &exp[ci].1, g, &mut st) {
                        first_err = Some(format!(
                            "{name} poison {poison:#x} case {ci} (exec {:#018x}, addr pattern {}, data kind {}): {x}",
                            c.exec, c.addr_pat, c.data_kind
                        ));
                        break;
                    }
                }
                rows.push(json!({"kernel":name, "poison":poison, "cases":cases.len(),
                    "guards_ok":guards, "hardware_sha256":sha(&raw[GUARD..total - GUARD]),
                    "verified": first_err.is_none(), "cases_verified": st.cases - before.cases}));
                if !guards {
                    return Err(format!(
                        "{name} poison {poison:#x}: global guard overwritten"
                    ));
                }
                if let Some(x) = first_err {
                    return Err(x);
                }

                // Planted corruption controls: each must be rejected by the verifier.
                let pick = |pred: &dyn Fn(&Case) -> bool| cases.iter().position(|c| pred(c));
                let ci_any = pick(&|c| c.exec != 0).ok_or("no active case")?;
                let ci_same =
                    pick(&|c| c.exec == u64::MAX && c.addr_pat == 1).ok_or("no contention case")?;
                let mut scratch = Stats::default();
                let mut expect_err =
                    |c: &Case, ci: usize, m: &[u32], what: &str| -> Result<(), String> {
                        if verify(f, c, &exp[ci].0, &exp[ci].1, m, &mut scratch).is_ok() {
                            Err(format!("{name} poison {poison:#x}: missed {what} control"))
                        } else {
                            Ok(())
                        }
                    };
                for ci in [ci_any, ci_same] {
                    let c = &cases[ci];
                    let g = &payload[ci * OUT_STRIDE..(ci + 1) * OUT_STRIDE];
                    if f != Form::NoRtn {
                        // Returned-value corruption on every active lane of the case's first
                        // active lane; some bit flip must break exactness or linearization.
                        let l = (0..64).find(|&l| c.active(l)).unwrap();
                        let mut detected = false;
                        for bit in 0..32 {
                            let mut m = g.to_vec();
                            m[l] ^= 1 << bit;
                            if verify(f, c, &exp[ci].0, &exp[ci].1, &m, &mut Stats::default())
                                .is_err()
                            {
                                detected = true;
                                break;
                            }
                        }
                        if !detected {
                            return Err(format!("{name}: missed returned-bit control"));
                        }
                        controls += 1;
                    } else {
                        let mut m = g.to_vec();
                        m[0] ^= 1;
                        expect_err(c, ci, &m, "no-return register")?;
                        controls += 1;
                    }
                    // Final LDS data word, and a window guard word at each end.
                    for wi in [c.widx((0..64).find(|&l| c.active(l)).unwrap()), 0, WIN - 1] {
                        let mut m = g.to_vec();
                        m[64 + wi] ^= 0x8000_0000;
                        expect_err(c, ci, &m, "final LDS / guard word")?;
                        controls += 1;
                    }
                    let mut m = g.to_vec();
                    m[OUT_STRIDE - 1] = 1;
                    expect_err(c, ci, &m, "outside-window LDS")?;
                    controls += 1;
                }
                // Global allocation guards (both ends).
                for at in [0, total - 1] {
                    let mut m = raw.clone();
                    m[at] ^= 1;
                    if guards_ok(&m, poison) {
                        return Err(format!("{name}: missed global guard control"));
                    }
                    controls += 1;
                }
            }
        }
    }
    rt.free(din).map_err(e)?;
    rt.free(dout).map_err(e)?;

    let kernels: Vec<Value> = bound
        .iter()
        .map(|(k, d)| {
            json!({"kernel":k, "regs":d.regs, "offset":d.offset, "lgkm0_wait_follows":d.wait_ok,
                "codec_words":d.words})
        })
        .collect();
    Ok(
        json!({"status":"PASS", "oracle":"cdna_ds_or_hw_oracle", "arch":arch,
        "kernels":kernels.len(), "cases_per_kernel_poison":ncases,
        "cases": st.cases, "exact_return_lanes": st.exact_lanes,
        "inactive_lanes_checked": st.inactive_lanes,
        "contended_groups_linearized": st.contended_groups, "contended_lanes": st.contended_lanes,
        "empty_exec_cases": st.empty_exec_cases,
        "controls_detected": controls, "seed":SEED, "offsets":OFFSETS, "poisons":POISONS,
        "hip_source_sha256":sha(src.as_bytes()), "code_object_sha256":sha(&code),
        "device_elf_sha256":sha(native),
        "disassembly_sha256":disasm_sha, "hardware_output_sha256":format!("{:x}", hashes.finalize()),
        "oracle_source_sha256":sha(include_bytes!("cdna_ds_or_hw_oracle.rs")),
        "interpreter_source_sha256":sha(include_bytes!("../../peacemaker-cdna/src/sim/mem.rs")),
        "codec_source_sha256":sha(include_bytes!("../../peacemaker-cdna/src/enc/ds.rs")),
        "kernel_bindings":kernels, "rows":rows}),
    )
}

fn main() -> ExitCode {
    let mut args = std::env::args_os().skip(1);
    let Some(dir) = args.next().map(PathBuf::from) else {
        eprintln!("usage: cdna_ds_or_hw_oracle OUT_DIR");
        return ExitCode::FAILURE;
    };
    if args.next().is_some() {
        eprintln!("unexpected argument");
        return ExitCode::FAILURE;
    }
    let result = run(&dir);
    let v = match &result {
        Ok(v) => v.clone(),
        Err(x) => json!({"status":"FAIL", "error":x}),
    };
    let path = dir.join("receipt.json");
    let saved = serde_json::to_vec_pretty(&v)
        .map_err(e)
        .and_then(|b| std::fs::write(&path, b).map_err(e));
    println!("{v}");
    if result.is_ok() && saved.is_ok() {
        ExitCode::SUCCESS
    } else {
        eprintln!("oracle result: {result:?}; receipt write: {saved:?}");
        ExitCode::FAILURE
    }
}
