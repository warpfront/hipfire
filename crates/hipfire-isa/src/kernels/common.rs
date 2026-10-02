//! Shared builder substrate of the kernel families: register references,
//! the canonical literal spelling, raw instruction helpers, and the
//! arch-spelled scalar, SMEM and raw-buffer access helpers. Every helper
//! emits exactly the text its callers spelled before it was shared.
use crate::{Arch, Builder, insn::{Instruction, MemoryClass}, reg::{Kind, RegRef}};

pub fn v(n: u8) -> RegRef { RegRef { kind: Kind::V, base: n, len: 1 } }
pub fn vr(n: u8, len: u8) -> RegRef { RegRef { kind: Kind::V, base: n, len } }
pub fn s(n: u8) -> RegRef { RegRef { kind: Kind::S, base: n, len: 1 } }
pub fn sr(n: u8, len: u8) -> RegRef { RegRef { kind: Kind::S, base: n, len } }
/// Literal spelling as llvm-objdump prints it: inline integers in decimal,
/// everything else in hex, so parse-back compares canonical text.
pub fn lit(x: u32) -> String { if x <= 64 { x.to_string() } else if x as i32 >= -16 && (x as i32) < 0 { (x as i32).to_string() } else { format!("{x:#x}") } }
/// One instruction. An untyped kernel's branches and `s_endpgm` go
/// through `Builder::control` (a typed kernel's through its typed scopes).
pub fn op(b: &mut Builder, text: impl Into<String>, defs: &[RegRef], uses: &[RegRef]) -> Result<(), String> {
    let insn = Instruction::new(text, defs.to_vec(), uses.to_vec());
    if crate::insn::control_flow(insn.mnemonic()) { b.control(insn) } else { b.push(insn) }
}
pub fn mem(b: &mut Builder, text: impl Into<String>, defs: &[RegRef], uses: &[RegRef], class: MemoryClass) -> Result<(), String> {
    b.push(Instruction::new(text, defs.to_vec(), uses.to_vec()).memory(class))
}
/// Scalar instruction over single SGPRs.
pub fn sop(b: &mut Builder, text: impl Into<String>, defs: &[u8], uses: &[u8]) -> Result<(), String> {
    op(b, text, &defs.iter().map(|&n| s(n)).collect::<Vec<_>>(), &uses.iter().map(|&n| s(n)).collect::<Vec<_>>())
}

/// `s_add_u32` (gfx11) / `s_add_co_u32` (gfx12).
pub fn s_add_u32(a: Arch) -> &'static str { if a.gfx12() { "s_add_co_u32" } else { "s_add_u32" } }
/// `s_addc_u32` (gfx11) / `s_add_co_ci_u32` (gfx12).
pub fn s_addc_u32(a: Arch) -> &'static str { if a.gfx12() { "s_add_co_ci_u32" } else { "s_addc_u32" } }
/// `s_add_i32` (gfx11) / `s_add_co_i32` (gfx12).
pub fn s_add_i32(a: Arch) -> &'static str { if a.gfx12() { "s_add_co_i32" } else { "s_add_i32" } }
/// SMEM immediate offset as llvm-objdump prints it (gfx11 spells zero `null`).
pub fn smem_off(a: Arch, off: u32) -> String { if off != 0 { format!("{off:#x}") } else if a.gfx12() { "0x0".into() } else { "null".into() } }

/// `dst = src + x` over a 64-bit SGPR pair (or a buffer descriptor base), `x` an SGPR.
pub fn add64(b: &mut Builder, dst: u8, src: u8, x: u8) -> Result<(), String> {
    let a = b.spec.arch;
    sop(b, format!("{} s{dst}, s{src}, s{x}", s_add_u32(a)), &[dst], &[src, x])?;
    sop(b, format!("{} s{}, s{}, 0", s_addc_u32(a), dst + 1, src + 1), &[dst + 1], &[src + 1])
}
/// `reg += imm` over a 64-bit SGPR pair.
pub fn add64_imm(b: &mut Builder, reg: u8, imm: u32) -> Result<(), String> {
    let a = b.spec.arch;
    sop(b, format!("{} s{reg}, s{reg}, {}", s_add_u32(a), lit(imm)), &[reg], &[reg])?;
    sop(b, format!("{} s{1}, s{1}, 0", s_addc_u32(a), reg + 1), &[reg + 1], &[reg + 1])
}

/// `s_load_b{32,64,128,256}` of `len` dwords at `s[base:base+1] + off`.
pub fn smem(b: &mut Builder, dst: u8, len: u8, base: u8, off: u32) -> Result<(), String> {
    let name = match len { 1 => "s_load_b32", 2 => "s_load_b64", 4 => "s_load_b128", 8 => "s_load_b256", _ => return Err("SMEM width".into()) };
    let d = if len == 1 { s(dst) } else { sr(dst, len) };
    mem(b, format!("{name} {d}, s[{base}:{}], {}", base + 1, smem_off(b.spec.arch, off)), &[d], &[sr(base, 2)], MemoryClass::SmemLoad)
}

/// Raw buffer resource word 3 (32-bit untyped, OOB_SELECT raw): an offset at
/// or past `num_records` reads 0 without touching memory. GPU-proven on
/// gfx1100, gfx1151 and gfx1201.
pub const SRD_WORD3: u32 = 0x3100_4000;
/// Words 2 and 3 of a raw buffer descriptor whose base is already in words
/// 0/1: `num_records` from an SGPR, or unbounded (`-1`).
pub fn srd_tail(b: &mut Builder, srd: u8, records: Option<u8>) -> Result<(), String> {
    match records {
        Some(r) => sop(b, format!("s_mov_b32 s{}, s{r}", srd + 2), &[srd + 2], &[r])?,
        None => sop(b, format!("s_mov_b32 s{}, -1", srd + 2), &[srd + 2], &[])?,
    }
    sop(b, format!("s_mov_b32 s{}, {}", srd + 3, lit(SRD_WORD3)), &[srd + 3], &[])
}
/// Raw buffer load of `width` dwords at `v{voff} + offset` (soffset zero).
pub fn bload(b: &mut Builder, width: u8, dst: u8, voff: u8, srd: u8, offset: u32) -> Result<(), String> {
    let name = match width { 1 => "buffer_load_b32", 2 => "buffer_load_b64", 4 => "buffer_load_b128", _ => return Err("buffer load width".into()) };
    if offset > b.spec.arch.buffer_offset_max() { return Err(format!("buffer offset {offset} exceeds the field")) }
    let data = if width == 1 { v(dst) } else { vr(dst, width) };
    mem(b, format!("{name} {data}, v{voff}, s[{srd}:{}], {} offen{}", srd + 3, zero_soffset(b.spec.arch), imm_offset(offset)), &[data], &[v(voff), sr(srd, 4)], MemoryClass::VmemLoad)
}
/// Raw buffer store of four dwords at `v{voff} + offset` (soffset zero).
pub fn bstore_b128(b: &mut Builder, data: u8, voff: u8, srd: u8, offset: u32) -> Result<(), String> {
    if offset > b.spec.arch.buffer_offset_max() { return Err(format!("buffer offset {offset} exceeds the field")) }
    let d = vr(data, 4);
    mem(b, format!("buffer_store_b128 {d}, v{voff}, s[{srd}:{}], {} offen{}", srd + 3, zero_soffset(b.spec.arch), imm_offset(offset)), &[], &[v(voff), d, sr(srd, 4)], MemoryClass::VmemStore)
}
fn zero_soffset(a: Arch) -> &'static str { if a.gfx12() { "null" } else { "0" } }
fn imm_offset(o: u32) -> String { if o == 0 { String::new() } else { format!(" offset:{o}") } }

/// Buffer offset of a masked gather row: at or past any `num_records` a
/// gather descriptor declares, so every load of the row reads 0 and touches
/// no memory.
pub const GATHER_OOB: u32 = 0x7fff_ff00;
/// Largest index (exclusive) for which [`gather_offset`]'s `index / div`
/// is exact: the f32 reciprocal quotient is within one of the true quotient
/// below 2^22, and two integer corrections fix it.
pub const GATHER_MAX_INDEX: u32 = 1 << 22;

/// Registers of [`gather_offset`]: four dead VGPRs and a dead lane-mask
/// SGPR pair for the quotient corrections.
#[derive(Clone, Copy, Debug)]
pub struct GatherTemps { pub v: [u8; 4], pub mask: u8 }

/// Masked indexed-row gather through a raw buffer descriptor (QSA K3, the
/// grouped MoE activation rows): per lane, `index` is a signed row index,
/// negative for a padding lane. Writes `live` (lane-mask pair: `index >= 0`)
/// and `dst = live ? (index / div) * row_bytes : GATHER_OOB`, with `div` a
/// uniform SGPR divisor (`None`: 1). Loads at `dst + k` against a descriptor
/// of `rows * row_bytes` records then fetch row `index / div` for live lanes
/// and 0 for padding lanes, which never touch memory. Requires
/// `0 <= index < GATHER_MAX_INDEX` on live lanes and `row_bytes < 2^24`.
pub fn gather_offset(b: &mut Builder, dst: u8, index: u8, div: Option<u8>, row_bytes: u32, live: u8, t: GatherTemps) -> Result<(), String> {
    if row_bytes >= 1 << 24 { return Err("gather row_bytes must fit 24 bits".into()) }
    let [q, r, w, f] = t.v;
    let m = t.mask;
    match div {
        Some(d) => {
            op(b, format!("v_cvt_f32_u32_e32 v{f}, s{d}"), &[v(f)], &[s(d)])?;
            op(b, format!("v_rcp_iflag_f32_e32 v{f}, v{f}"), &[v(f)], &[v(f)])?;
            op(b, format!("v_cvt_f32_u32_e32 v{q}, v{index}"), &[v(q)], &[v(index)])?;
            op(b, format!("v_mul_f32_e32 v{q}, v{q}, v{f}"), &[v(q)], &[v(q), v(f)])?;
            op(b, format!("v_cvt_u32_f32_e32 v{q}, v{q}"), &[v(q)], &[v(q)])?;
            op(b, format!("v_mul_lo_u32 v{r}, v{q}, s{d}"), &[v(r)], &[v(q), s(d)])?;
            op(b, format!("v_sub_nc_u32_e32 v{r}, v{index}, v{r}"), &[v(r)], &[v(index), v(r)])?;
            // remainder < 0: q -= 1, r += d; then r >= d: q += 1.
            op(b, format!("v_cmp_gt_i32_e64 s{m}, 0, v{r}"), &[s(m)], &[v(r)])?;
            op(b, format!("v_add_nc_u32_e32 v{w}, -1, v{q}"), &[v(w)], &[v(q)])?;
            op(b, format!("v_cndmask_b32_e64 v{q}, v{q}, v{w}, s{m}"), &[v(q)], &[v(q), v(w), s(m)])?;
            op(b, format!("v_add_nc_u32_e32 v{w}, s{d}, v{r}"), &[v(w)], &[s(d), v(r)])?;
            op(b, format!("v_cndmask_b32_e64 v{r}, v{r}, v{w}, s{m}"), &[v(r)], &[v(r), v(w), s(m)])?;
            op(b, format!("v_cmp_le_u32_e64 s{m}, s{d}, v{r}"), &[s(m)], &[s(d), v(r)])?;
            op(b, format!("v_add_nc_u32_e32 v{w}, 1, v{q}"), &[v(w)], &[v(q)])?;
            op(b, format!("v_cndmask_b32_e64 v{q}, v{q}, v{w}, s{m}"), &[v(q)], &[v(q), v(w), s(m)])?;
        }
        None => op(b, format!("v_mov_b32_e32 v{q}, v{index}"), &[v(q)], &[v(index)])?,
    }
    op(b, format!("v_cmp_le_i32_e64 s{live}, 0, v{index}"), &[s(live)], &[v(index)])?;
    op(b, format!("v_mul_u32_u24_e32 v{q}, {}, v{q}", lit(row_bytes)), &[v(q)], &[v(q)])?;
    op(b, format!("v_cndmask_b32_e64 v{dst}, {}, v{q}, s{live}", lit(GATHER_OOB)), &[v(dst)], &[v(q), s(live)])
}
