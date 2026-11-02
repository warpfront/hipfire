# 2. Elementwise bounds: subtract

Compute `c[i] = a[i] - b[i]` in f32 for `i < n`, with `i = workgroup_x * 64 + thread`.
Launch `ceil(n / 64)` workgroups of 64 threads and make the buffer descriptors, not `exec`, discard the lanes past `n`.

The 32-byte kernarg record is `a` (pointer, offset 0), `b` (pointer, offset 8), `c` (pointer, offset 16) and `n` (u32, offset 24); bytes 28–31 are padding. Each pointer becomes a raw buffer descriptor whose `num_records` is `n * 4` bytes (`n * 4` must fit u32). A load at or past `num_records` returns 0 without touching memory and a store there is dropped, so lanes with `i >= n` run the arithmetic harmlessly and write nothing.

```rip variant=subtract
// SGPRs
const KARG: u8 = 0;     // s[0:1]  kernarg pointer
const WGX_IN: u8 = 2;   // s2      gfx11 only: workgroup id x
const ARGS: u8 = 8;     // s[8:15] a, b, c, n, pad (one 32-byte load, one owner)
const A_PTR: u8 = 8;    // s[8:9]
const B_PTR: u8 = 10;   // s[10:11]
const C_PTR: u8 = 12;   // s[12:13]
const N: u8 = 14;       // s14
const SRD_A: u8 = 16;   // s[16:19] descriptor over a
const SRD_B: u8 = 20;   // s[20:23] descriptor over b
const SRD_C: u8 = 24;   // s[24:27] descriptor over c
const BYTES: u8 = 28;   // s28      n * 4 = num_records
const BASE: u8 = 29;    // s29      workgroup id * 64 * 4
// VGPRs
const TID: u8 = 0;      // v0 thread id
const OFF: u8 = 1;      // v1 byte offset 4 * i
const VA: u8 = 2;       // v2 a[i], later c[i]
const VB: u8 = 3;       // v3 b[i]

fn kernel_spec(arch: Arch) -> KernelSpec {
    KernelSpec {
        kernel_id: "subtract".into(), variant: "subtract".into(), arch,
        symbol: format!("subtract_{}", arch.name()),
        kernargs: KernargLayout::new(32).pointer("a", 0).pointer("b", 8).pointer("c", 16)
            .hidden("n", 24, 4, "by_value"),
        user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 64, group_segment_fixed_size: 0, wave32: true, cu_mode: false,
    }
}

fn plan(arch: Arch) -> Result<RegPlan, String> {
    let mut p = RegPlan::new(256, 104)?;
    p.s::<2>("kernarg_ptr", KARG, Live::Whole)?;
    if !arch.gfx12() { p.s::<1>("wg_x_in", WGX_IN, Live::Whole)?; }
    p.s::<8>("args", ARGS, Live::Whole)?;        // owns s8..s15: the whole b256 destination
    p.s::<4>("srd_a", SRD_A, Live::Whole)?;
    p.s::<4>("srd_b", SRD_B, Live::Whole)?;
    p.s::<4>("srd_c", SRD_C, Live::Whole)?;
    p.s::<1>("bytes", BYTES, Live::Whole)?;
    p.s::<1>("base", BASE, Live::Whole)?;
    for (name, r) in [("tid", TID), ("off", OFF), ("va", VA), ("vb", VB)] {
        p.v::<1>(name, r, Live::Whole)?;
    }
    // A whole 8-VGPR granule: the descriptor's allocation equals the metadata count.
    p.v::<1>("granule_pad", 7, Live::Whole)?;
    Ok(p)
}

fn emit(spec: Spec) -> Result<Emitted, String> {
    if spec.variant != "subtract" { return Err(format!("subtract: unknown variant `{}` (subtract)", spec.variant)); }
    let arch = spec.arch;
    let mut builder = Builder::new(kernel_spec(arch), plan(arch)?);
    let mut wg = Workgroup::new(&mut builder, arch)?;
    author(&mut wg, arch)?;
    builder.finish()
}

// Descriptor words 0/1 from a pointer pair already in SGPRs, words 2/3 from
// the byte count and the raw-buffer word 3 constant.
fn srd_over(b: &mut Builder, srd: u8, ptr: u8, records: u8) -> Result<(), String> {
    sop(b, format!("s_mov_b32 s{srd}, s{ptr}"), &[srd], &[ptr])?;
    sop(b, format!("s_mov_b32 s{}, s{}", srd + 1, ptr + 1), &[srd + 1], &[ptr + 1])?;
    srd_tail(b, srd, Some(records))
}

fn author<T: Target>(wg: &mut Wg<T>, arch: Arch) -> Result<(), String> {
    let end = wg.exit(".Lsubtract_end")?;
    let b = wg.isa();
    smem(b, ARGS, 8, KARG, 0)?;                    // all 32 kernarg bytes, one owner
    // Independent of the load: these run while the SMEM request is in flight.
    workgroup_id(b, 0, BASE)?;                     // s2 on gfx11, ttmp9 on gfx12
    sop(b, format!("s_lshl_b32 s{BASE}, s{BASE}, 8"), &[BASE], &[BASE])?;   // 64 lanes * 4 bytes
    op(b, format!("v_lshlrev_b32_e32 v{OFF}, 2, v{TID}"), &[v(OFF)], &[v(TID)])?;
    op(b, format!("v_add_nc_u32_e32 v{OFF}, s{BASE}, v{OFF}"), &[v(OFF)], &[s(BASE), v(OFF)])?;
    // First read of n: the automatic kernarg wait lands here.
    sop(b, format!("s_lshl_b32 s{BYTES}, s{N}, 2"), &[BYTES], &[N])?;
    srd_over(b, SRD_A, A_PTR, BYTES)?;
    srd_over(b, SRD_B, B_PTR, BYTES)?;
    srd_over(b, SRD_C, C_PTR, BYTES)?;
    // No exec mask: lanes with i >= n load 0 and their store is discarded.
    bload(b, 1, VA, OFF, SRD_A, 0)?;
    bload(b, 1, VB, OFF, SRD_B, 0)?;
    // The first read of v2 and v3 gets one automatic load wait with count 0:
    // v3 is the younger load and is needed too, so nothing may stay in flight.
    op(b, format!("v_sub_f32_e32 v{VA}, v{VA}, v{VB}"), &[v(VA)], &[v(VA), v(VB)])?;
    let soffset = if arch.gfx12() { "null" } else { "0" };
    mem(b, format!("buffer_store_b32 v{VA}, v{OFF}, s[{SRD_C}:{}], {soffset} offen", SRD_C + 3),
        &[], &[v(OFF), v(VA), sr(SRD_C, 4)], MemoryClass::VmemStore)?;
    wg.end(end)
}
```

```mnemonics
gfx1151: buffer_load_b32 v2, v1, s[16:19], 0 offen
gfx1201: buffer_load_b32 v2, v1, s[16:19], null offen
gfx1151: buffer_store_b32 v2, v1, s[24:27], 0 offen
gfx1201: buffer_store_b32 v2, v1, s[24:27], null offen
!gfx1151: buffer_load_dword v2, v1, s[16:19], 0 offen
!gfx1201: buffer_load_dword v2, v1, s[16:19], null offen
```

## Why

**Hardware bounds checking replaces the tail branch.** `srd_tail(b, srd, Some(r))` fills descriptor word 2 from SGPR `r` (here `n * 4`) and word 3 with `SRD_WORD3`, the raw-buffer constant (32-bit untyped, out-of-bounds select raw). With that word 3 a buffer access whose offset is at or past `num_records` reads 0 and a store is dropped; no memory is touched. Because the byte count is a multiple of four, no dword access straddles the bound. `exec` stays full, so there is no `v_cmp`, no `s_and_b32 exec_lo` and no mask register to own. Use `exec` masking (recipe 1) when the guard cannot be a byte range, for instance when the work is not one contiguous run.

**Owner groups.** `args` is one `s::<8>` at `s8` because `smem(b, ARGS, 8, KARG, 0)` writes `s[8:15]` with a single `s_load_b256`; the destination must lie wholly inside one owner and an `s::<8>` must start at a multiple of 8. Reading `s8`, `s9` (pointer a), `s10`, `s11` (b), `s12`, `s13` (c) and `s14` (n) individually is fine: each use lies inside `args`. Each descriptor is its own `s::<4>` (aligned to 4), and the buffer instructions name it as `sr(SRD, 4)`, all four dwords at once. `srd_over` receives the ISA handle returned by `wg.isa()` and emits instructions without reopening the Workgroup scope.

**Defs and uses are complete.** `sop(b, text, defs, uses)` takes SGPR numbers; `op`/`mem` take `RegRef`s built with `v(n)`, `s(n)` and `sr(n, len)`. The scalar operand `s{BASE}` of `v_add_nc_u32_e32` and `s{N}` of `s_lshl_b32` appear in the use lists. An omitted scalar is an R0603 error; reading a register that nothing defined, such as `s29` before `workgroup_id`, is an R0604 error.

**Automatic waits.** There is no `wg.wait` and no `b.wait` in the script.
- The SMEM load is pending from its issue. The three instructions that follow it do not read it, so no wait is emitted there. The wait appears immediately before `s_lshl_b32 s28, s14, 2`, the first read of `args`.
- Both buffer loads are pending on the VMEM load counter, which retires in issue order. The builder waits at the first instruction that reads `v2` or `v3`. `v_sub_f32_e32` reads both, and `v3` is the younger load, so the wait count is 0 and one wait drains everything. Reading only `v2` first would produce a wait that leaves the second load in flight (`vmcnt(1)` / `s_wait_loadcnt 0x1`).
- The store sits at the end of the wave; `s_endpgm` retires it and no store-counter wait is emitted.

| | gfx1151 | gfx1201 |
|---|---|---|
| kernarg wait | `s_waitcnt lgkmcnt(0)` | `s_wait_kmcnt 0x0` |
| load wait before `v_sub_f32_e32` | `s_waitcnt vmcnt(0)` | `s_wait_loadcnt 0x0` |

**Buffer mnemonics and `soffset`.** Both RDNA targets spell the 32-bit raw buffer access `buffer_load_b32` / `buffer_store_b32`; `bload` emits `buffer_load_b32` on gfx1151 and on gfx1201. The repository's instruction tables contain no `buffer_load_dword` for either target, and the `mnemonics` block above asserts that. The arch difference is the scalar offset operand that follows the descriptor: a zero `soffset` is the literal `0` on gfx11 and `null` on gfx12. `bload` writes the right one; `buffer_store_b32` has no helper, so the script picks it with `arch.gfx12()`. The baseline's wrong spelling (`0` on gfx1201) fails at assembly with `SOFFSET does not fit its encoding field`. The immediate `offset:` field also differs: at most 4095 on gfx11 and 8388607 on gfx12; `bload` checks it.

**Baseline pitfalls.** Instruction text valid on one arch is not valid on the other: `s_add_i32` and `v_max_f32_e32` do not exist on gfx1201 (`s_add_co_i32` and `v_max_num_f32_e32` do), so spell scalar adds through the arch helpers. `v_sub_f32_e32` exists on both targets, which is why this kernel needs no arch branch around the subtraction. The workgroup id is `workgroup_id(b, 0, dst)`, never a raw `s2`.

## Variations

- Only guard the output: give the loads an unbounded descriptor with `srd_tail(b, SRD_A, None)` and `SRD_B`, and keep the bounded `SRD_C`. Reads past `n` then touch memory, so only do it when the inputs are padded.
- One count per buffer: pass a different `records` SGPR to each `srd_over` when `a`, `b` and `c` have different lengths.
- Two elements per lane with `bload(b, 2, ...)` and `buffer_store_b64`: declare a `v::<2>` owner, double the per-thread and per-workgroup byte strides, and halve the grid. The bound is still `n * 4` bytes; require even `n` or handle the final single element separately.
- Use `exec` masking as well (recipe 1) when a lane must not execute side effects other than the store.
