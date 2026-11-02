# 3. Workgroup ids: grid_ids

In a 2D grid of 32-thread workgroups, thread 0 of workgroup `(x, y)` stores `out[y * gx + x] = x | (y << 16)` as a u32.
Launch a `gx` by `gy` grid, block `(32, 1, 1)`; no other thread stores, so every output word is written exactly once.

The 16-byte kernarg record is `out` (pointer, offset 0) and `gx` (u32, offset 8); bytes 12–15 are padding. `gx` and `gy` stay below 2^16 so both ids fit in the packed halves.

```rip variant=grid_ids
// SGPRs
const KARG: u8 = 0;    // s[0:1]   kernarg pointer
const WGIDS_IN: u8 = 2;// s[2:3]   gfx11 only: workgroup id x in s2, y in s3
const ARGS: u8 = 4;    // s[4:7]   out_lo, out_hi, gx, pad (one 16-byte load, one owner)
const GX: u8 = 6;
const WGX: u8 = 8;     // s8  workgroup id x
const WGY: u8 = 9;     // s9  workgroup id y
const IDX: u8 = 10;    // s10 y * gx + x, then the byte offset
const PACK: u8 = 11;   // s11 x | (y << 16)
const SRD: u8 = 12;    // s[12:15] descriptor over out
const MASK: u8 = 16;   // s[16:17] lane mask
// VGPRs
const TID: u8 = 0;     // v0 thread id
const OFF: u8 = 1;     // v1 byte offset
const VAL: u8 = 2;     // v2 packed ids

fn kernel_spec(arch: Arch) -> KernelSpec {
    KernelSpec {
        kernel_id: "grid_ids".into(), variant: "grid_ids".into(), arch,
        symbol: format!("grid_ids_{}", arch.name()),
        kernargs: KernargLayout::new(16).pointer("out", 0).hidden("gx", 8, 4, "by_value"),
        // Enables the y workgroup id: s3 on gfx11, the low half of ttmp7 on gfx12.
        user_sgpr_count: 2, system_sgpr_workgroup_id_y: true,
        workgroup_size: 32, group_segment_fixed_size: 0, wave32: true, cu_mode: false,
    }
}

fn plan(arch: Arch) -> Result<RegPlan, String> {
    let mut p = RegPlan::new(256, 104)?;
    p.s::<2>("kernarg_ptr", KARG, Live::Whole)?;
    // gfx11 launches both ids in system SGPRs, so the plan must own them;
    // gfx12 reads ttmp9 / ttmp7, which are not plan registers.
    if !arch.gfx12() { p.s::<2>("wg_ids_in", WGIDS_IN, Live::Whole)?; }
    p.s::<4>("args", ARGS, Live::Whole)?;
    p.s::<1>("wgx", WGX, Live::Whole)?;
    p.s::<1>("wgy", WGY, Live::Whole)?;
    p.s::<1>("idx", IDX, Live::Whole)?;
    p.s::<1>("pack", PACK, Live::Whole)?;
    p.s::<4>("srd_out", SRD, Live::Whole)?;
    p.s::<2>("mask", MASK, Live::Whole)?;
    for (name, r) in [("tid", TID), ("off", OFF), ("val", VAL)] {
        p.v::<1>(name, r, Live::Whole)?;
    }
    // A whole 8-VGPR granule: the descriptor's allocation equals the metadata count.
    p.v::<1>("granule_pad", 7, Live::Whole)?;
    Ok(p)
}

fn emit(spec: Spec) -> Result<Emitted, String> {
    if spec.variant != "grid_ids" { return Err(format!("grid_ids: unknown variant `{}` (grid_ids)", spec.variant)); }
    let arch = spec.arch;
    let mut builder = Builder::new(kernel_spec(arch), plan(arch)?);
    let mut wg = Workgroup::new(&mut builder, arch)?;
    author(&mut wg, arch)?;
    builder.finish()
}

fn author<T: Target>(wg: &mut Wg<T>, arch: Arch) -> Result<(), String> {
    let end = wg.exit(".Lgrid_ids_end")?;
    let b = wg.isa();
    smem(b, ARGS, 4, KARG, 0)?;
    // Axis 0 is x, axis 1 is y. The helper reads s2 / s3 on gfx11 and
    // ttmp9 / (ttmp7 & 0xffff) on gfx12; the destination is an owned SGPR.
    workgroup_id(b, 0, WGX)?;
    workgroup_id(b, 1, WGY)?;
    // pack = x | (y << 16): independent of the kernarg load, so no wait yet.
    sop(b, format!("s_lshl_b32 s{PACK}, s{WGY}, 16"), &[PACK], &[WGY])?;
    sop(b, format!("s_or_b32 s{PACK}, s{PACK}, s{WGX}"), &[PACK], &[PACK, WGX])?;
    // idx = y * gx + x; the first read of gx gets the automatic kernarg wait.
    sop(b, format!("s_mul_i32 s{IDX}, s{WGY}, s{GX}"), &[IDX], &[WGY, GX])?;
    sop(b, format!("{} s{IDX}, s{IDX}, s{WGX}", s_add_u32(arch)), &[IDX], &[IDX, WGX])?;
    sop(b, format!("s_lshl_b32 s{IDX}, s{IDX}, 2"), &[IDX], &[IDX])?;
    sop(b, format!("s_mov_b32 s{SRD}, s{ARGS}"), &[SRD], &[ARGS])?;
    sop(b, format!("s_mov_b32 s{}, s{}", SRD + 1, ARGS + 1), &[SRD + 1], &[ARGS + 1])?;
    srd_tail(b, SRD, None)?;
    // Scalars into VGPRs for the store, then keep lane 0 only (one wave).
    op(b, format!("v_mov_b32_e32 v{OFF}, s{IDX}"), &[v(OFF)], &[s(IDX)])?;
    op(b, format!("v_mov_b32_e32 v{VAL}, s{PACK}"), &[v(VAL)], &[s(PACK)])?;
    op(b, format!("v_cmp_eq_u32_e64 s{MASK}, 0, v{TID}"), &[sr(MASK, 2)], &[v(TID)])?;
    op(b, format!("s_and_b32 exec_lo, exec_lo, s{MASK}"), &[], &[sr(MASK, 2)])?;
    let soffset = if arch.gfx12() { "null" } else { "0" };
    mem(b, format!("buffer_store_b32 v{VAL}, v{OFF}, s[{SRD}:{}], {soffset} offen", SRD + 3),
        &[], &[v(OFF), v(VAL), sr(SRD, 4)], MemoryClass::VmemStore)?;
    wg.end(end)
}
```

## Why

**Workgroup ids differ by arch, and `workgroup_id` hides it.** `workgroup_id(b, axis: u8, dst: u8) -> Result<(), String>` takes axis 0 (x) or 1 (y) and an SGPR number to write; `dst` must be plan-owned. For example, `workgroup_id(b, 1, WGY)?` reads y. It emits:

| | gfx1151 (user SGPRs = 2) | gfx1201 |
|---|---|---|
| x | `s_mov_b32 dst, s2` | `s_mov_b32 dst, ttmp9` |
| y | `s_mov_b32 dst, s3` | `s_and_b32 dst, ttmp7, 0xffff` |
| z (axis 2) | refused: `KernelSpec` has no z flag | refused: `KernelSpec` has no z flag |

The hand-written equivalent, which this recipe deliberately does not use, is:

```text
gfx1151:  s_mov_b32 s8, s2            s_mov_b32 s9, s3
gfx1201:  s_mov_b32 s8, ttmp9         s_and_b32 s9, ttmp7, 0xffff
```

On gfx11 the launch writes the ids into the system SGPRs that follow the user SGPRs: x at `s{user_sgpr_count}`, y at the next one, only when the spec enables y. With `user_sgpr_count: 2` that is `s2` and `s3`. On gfx12 the ids arrive in architected trap temporaries: `ttmp9` holds x, and `ttmp7` packs y in bits 15:0 and z in bits 31:16. A plain `s_mov_b32 s9, ttmp7` would put z into the upper half of y. For a 2D grid z is 0, but the mask is still the correct extraction and costs one instruction.

**What the plan must own.** `ttmp` registers are special registers outside `RegPlan` and outside defs/uses declarations; the builtin writes only `dst`. On gfx11 the source is an ordinary SGPR, so the plan must own `s2` and `s3`: `p.s::<2>("wg_ids_in", 2, ...)` covers both (two single owners work too). If the plan does not own `s2` on gfx11, compilation stops with `s2 at ... has no unique live owner` (R0401). On gfx1201 the same plan entry is skipped with `if !arch.gfx12()`. Reading the gfx11 registers by hand on gfx1201 was a silent-wrong-answer trap in the baseline; an undefined SGPR read now reports R0604.

**Spec and refusals.** `system_sgpr_workgroup_id_y: true` is required for y: the builtin refuses axis 1 without it and names the flag. Axis 2 is refused on both architectures because `KernelSpec` has no z flag. The kernarg load is 16 bytes (`s::<4>`, a single owner); `gx` at s6 lies inside it.

**Owner groups and declared effects.** Every destination of `sop`/`op` here is a plan range: `wgx`, `wgy`, `idx`, `pack` are single SGPRs, `srd_out` is an `s::<4>`, `mask` an `s::<2>`. The declared lists include scalar operands of vector instructions: `v_mov_b32_e32 v1, s10` lists `s(IDX)` as a use. Omitting it is an R0603 error, and reading `s10` before it is written would be R0604. `v_cmp_eq_u32_e64 s16, 0, v0` reads no SGPR; its constant 0 is an inline constant.

**Automatic waits.** There is no wait in the source.
- The SMEM load is issued first. Everything up to `s_mul_i32` runs while it is in flight: the two id reads and the two `pack` instructions do not touch `args`.
- The builder places one wait immediately before `s_mul_i32 s10, s9, s6`, the first reader of `args`:

| | gfx1151 | gfx1201 |
|---|---|---|
| kernarg wait | `s_waitcnt lgkmcnt(0)` | `s_wait_kmcnt 0x0` |

- The final store needs no wait; `s_endpgm` retires it.

**Thread 0 only.** One 32-thread workgroup is exactly one wave32, so `tid == 0` is lane 0 and `v_cmp_eq_u32_e64` selects a single lane. `s_and_b32 exec_lo, exec_lo, s16` is the only mask change; nothing vector follows the store, so `exec` is never restored.

**Per-arch spellings in this script.** Scalar add: `s_add_u32` on gfx1151 and `s_add_co_u32` on gfx1201, from `s_add_u32(arch)`. Store `soffset` zero: `0` on gfx1151, `null` on gfx1201. `s_mul_i32`, `s_or_b32`, `s_lshl_b32`, `v_mov_b32_e32` and `v_cmp_eq_u32_e64` have one spelling on both.

## Variations

- Larger blocks: change `workgroup_size` to 64 (two waves) and keep the exec mask `tid == 0`; every wave then runs the compare, but only wave 0 has a lane whose `v0` is 0. This is the correct way to pick one thread of a multi-wave workgroup.
- 3D work: fold the third logical coordinate into x or y in the host grid and unpack it in the kernel. `workgroup_id(b, 2, dst)` is refused on both targets because `KernelSpec` cannot enable z.
- Store one word per thread: drop the compare and the `exec` change, use byte base `128 * (y * gx + x)`, and set `OFF = base + tid * 4` (`v_lshlrev_b32_e32 v1, 2, v0`, then `v_add_nc_u32_e32 v1, s10, v1`). The output then holds 32 words per workgroup.
