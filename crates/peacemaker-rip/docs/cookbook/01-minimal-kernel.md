# 1. Minimal kernel: fill

Write `out[i] = i * k + c` (wrapping u32 arithmetic) for every global index `i = workgroup_x * 64 + thread` with `i < n`.
Launch `ceil(n / 64)` workgroups of 64 threads; lanes with `i >= n` must not store.

The 24-byte kernarg record is `out` (pointer, offset 0), `n` (u32, offset 8), `k` (u32, offset 12) and `c` (u32, offset 16); bytes 20–23 are padding and are never read. The script has three parts: `kernel_spec` (launch metadata), `plan` (register ownership) and `author` (the instructions); `emit` only wires them together. Later recipes reuse this skeleton.

```rip variant=fill
// SGPRs
const KARG: u8 = 0;    // s[0:1]   kernarg pointer (a user SGPR pair)
const WGX_IN: u8 = 2;  // s2       gfx11 only: the launch puts workgroup id x here
const ARGS: u8 = 4;    // s[4:7]   out_lo, out_hi, n, k (one 16-byte load, one owner)
const N: u8 = 6;
const K: u8 = 7;
const C: u8 = 8;       // s8       c, loaded separately
const BASE: u8 = 9;    // s9       workgroup id * 64
const SRD: u8 = 12;    // s[12:15] buffer descriptor over out
const MASK: u8 = 16;   // s[16:17] lane mask
// VGPRs
const TID: u8 = 0;     // v0 thread id inside the workgroup
const IDX: u8 = 1;     // v1 global index i
const OFF: u8 = 2;     // v2 byte offset 4 * i
const VAL: u8 = 3;     // v3 result

// Launch metadata: arch-specific symbol, kernarg layout, workgroup size.
fn kernel_spec(arch: Arch) -> KernelSpec {
    KernelSpec {
        kernel_id: "fill".into(), variant: "fill".into(), arch,
        symbol: format!("fill_{}", arch.name()),
        kernargs: KernargLayout::new(24).pointer("out", 0)
            .hidden("n", 8, 4, "by_value").hidden("k", 12, 4, "by_value")
            .hidden("c", 16, 4, "by_value"),
        user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 64, group_segment_fixed_size: 0, wave32: true, cu_mode: false,
    }
}

// Register ownership. Every register an instruction names must lie inside ONE
// range declared here; range widths are 1, 2, 4 or 8.
fn plan(arch: Arch) -> Result<RegPlan, String> {
    let mut p = RegPlan::new(256, 104)?;
    p.s::<2>("kernarg_ptr", KARG, Live::Whole)?;
    if !arch.gfx12() { p.s::<1>("wg_x_in", WGX_IN, Live::Whole)?; }
    p.s::<4>("args", ARGS, Live::Whole)?;      // owns s4..s7: the whole b128 destination
    p.s::<1>("c", C, Live::Whole)?;
    p.s::<1>("base", BASE, Live::Whole)?;
    p.s::<4>("srd_out", SRD, Live::Whole)?;
    p.s::<2>("mask", MASK, Live::Whole)?;
    for (name, r) in [("tid", TID), ("idx", IDX), ("off", OFF), ("val", VAL)] {
        p.v::<1>(name, r, Live::Whole)?;
    }
    // A whole 8-VGPR granule: the descriptor's allocation equals the metadata count.
    p.v::<1>("granule_pad", 7, Live::Whole)?;
    Ok(p)
}

fn emit(spec: Spec) -> Result<Emitted, String> {
    if spec.variant != "fill" { return Err(format!("fill: unknown variant `{}` (fill)", spec.variant)); }
    let arch = spec.arch;
    let mut builder = Builder::new(kernel_spec(arch), plan(arch)?);
    // The Workgroup scope lasts until the next statement that mentions `builder`.
    let mut wg = Workgroup::new(&mut builder, arch)?;
    author(&mut wg, arch)?;
    builder.finish()
}

// Only `wg` is visible here; `b` is the instruction sink that `wg.isa()` returns.
fn author<T: Target>(wg: &mut Wg<T>, arch: Arch) -> Result<(), String> {
    let end = wg.exit(".Lfill_end")?;
    let b = wg.isa();
    // Kernarg bytes 0..16 as ONE s_load_b128, then c (offset 16) as a second load.
    smem(b, ARGS, 4, KARG, 0)?;
    smem(b, C, 1, KARG, 16)?;
    // No SMEM result is read yet, so no wait is emitted yet.
    workgroup_id(b, 0, BASE)?;                                   // s2 on gfx11, ttmp9 on gfx12
    sop(b, format!("s_lshl_b32 s{BASE}, s{BASE}, 6"), &[BASE], &[BASE])?;
    // Descriptor over out: base from the loaded pointer, unbounded records.
    // The first read of s4 gets the automatic kernarg wait; it retires both loads.
    sop(b, format!("s_mov_b32 s{SRD}, s{ARGS}"), &[SRD], &[ARGS])?;
    sop(b, format!("s_mov_b32 s{}, s{}", SRD + 1, ARGS + 1), &[SRD + 1], &[ARGS + 1])?;
    srd_tail(b, SRD, None)?;
    // i = wg * 64 + tid, and only lanes with i < n stay active.
    op(b, format!("v_add_nc_u32_e32 v{IDX}, s{BASE}, v{TID}"), &[v(IDX)], &[s(BASE), v(TID)])?;
    op(b, format!("v_cmp_gt_u32_e64 s{MASK}, s{N}, v{IDX}"), &[sr(MASK, 2)], &[s(N), v(IDX)])?;
    op(b, format!("s_and_b32 exec_lo, exec_lo, s{MASK}"), &[], &[sr(MASK, 2)])?;
    // value = i * k + c (v_mul_lo_u32 and v_add_nc_u32 both wrap modulo 2^32).
    op(b, format!("v_mul_lo_u32 v{VAL}, v{IDX}, s{K}"), &[v(VAL)], &[v(IDX), s(K)])?;
    op(b, format!("v_add_nc_u32_e32 v{VAL}, s{C}, v{VAL}"), &[v(VAL)], &[s(C), v(VAL)])?;
    op(b, format!("v_lshlrev_b32_e32 v{OFF}, 2, v{IDX}"), &[v(OFF)], &[v(IDX)])?;
    let soffset = if arch.gfx12() { "null" } else { "0" };
    mem(b, format!("buffer_store_b32 v{VAL}, v{OFF}, s[{SRD}:{}], {soffset} offen", SRD + 3),
        &[], &[v(OFF), v(VAL), sr(SRD, 4)], MemoryClass::VmemStore)?;
    wg.end(end)
}
```

## Why

**Emit / spec / author.** `emit` builds a `Builder`, opens `Workgroup::new(&mut builder, arch)?` and calls `author(&mut wg, arch)?`. A Workgroup scope ends at the first later statement that mentions the builder variable, so `emit` must not touch `builder` between `Workgroup::new` and `builder.finish()`. In particular, if the outer builder is named `b`, writing `let b = wg.isa();` in that block mentions and shadows it, closing the scope too early. Keeping the instruction stream in `author`, which receives only `wg`, avoids that trap. `kernel_spec` and `plan` return values, so a variant can swap them without touching the instructions.

**The symbol is per arch.** `format!("fill_{}", arch.name())` gives `fill_gfx1151` and `fill_gfx1201`, so a module holding both never has two kernels with one name. The metadata must agree with the script: `KernargLayout::new(24)` is the kernarg segment size, `workgroup_size: 64` is the launch block, and `system_sgpr_workgroup_id_y: false` means gfx11 delivers only `s2`.

**Owner groups.** Each `p.s::<N>` / `p.v::<N>` call declares one owner, and N is 1, 2, 4 or 8 (an SGPR group wider than 1 must start at a multiple of its width). Every def and every use of an instruction must lie wholly inside one owner. `smem(b, ARGS, 4, KARG, 0)?` writes `s[4:7]`, so `args` is a single `s::<4>`; four `s::<1>` owners at s4..s7 would fail with `s[4:7] ... has no unique live owner`. Reading `s6` (n) or `s7` (k) alone is fine, because each lies inside `args`. `c` needs its own `s::<1>` because the second load is a separate destination. The lane mask is declared as a pair: declaring more than an instruction touches is allowed, declaring less is not.

**Declared defs and uses must be complete.** Every `op`/`sop`/`mem` call lists the registers it writes and reads, including scalar operands of vector instructions. Leaving `s(N)` out of the compare or `s(K)` out of the multiply is a compile error (R0603 for a missing declaration, R0604 for a read of a register no earlier instruction or launch value defined). The same lists drive the wait insertion below.

**Automatic waits.** The script contains no `wg.wait`. The builder records the two SMEM loads as pending and emits nothing until an instruction reads one of their destinations. SMEM loads can complete out of order, so the wait inserted before `s_mov_b32 s12, s4` drains the whole SMEM counter, which also retires the load of `c`. This kernel therefore has exactly one inserted wait:

| | gfx1151 | gfx1201 |
|---|---|---|
| wait before the first read of `s4` | `s_waitcnt lgkmcnt(0)` | `s_wait_kmcnt 0x0` |

The store at the end needs no wait: `s_endpgm` retires pending stores and the builder clears its ledger there, so no store-counter wait appears. On gfx1201 the builder also inserts `s_wait_alu` when an SGPR that a VALU instruction has read is rewritten by SALU and read again by VALU; its tracker inserts none in this script.

**Arch differences in this script.**

| | gfx1151 | gfx1201 |
|---|---|---|
| workgroup id x | `s2` (the plan must own it) | `ttmp9` (outside the plan) |
| buffer store `soffset` zero | `0` | `null` |
| SMEM offset zero (`smem` helper) | `null` | `0x0` |
| automatic SMEM wait | `s_waitcnt lgkmcnt(0)` | `s_wait_kmcnt 0x0` |

`workgroup_id(b, 0, BASE)?` reads the right source for the compile target and writes it to an owned SGPR; the raw spellings are in recipe 3. `smem` and `bload` choose the zero SMEM offset and `soffset` spelling themselves; `buffer_store_b32` has no helper, so the script spells `0` or `null`.

**Baseline pitfalls this avoids.**
- In the baseline, reading the workgroup id from `s2` on gfx1201 compiled and silently produced wrong output. An undefined read now reports R0604; use `workgroup_id`.
- Bare `v_add_nc_u32` is not a mnemonic; the VOP2 forms need `_e32`. `v_mul_lo_u32` is VOP3-only and has no suffix.
- `s_add_i32` does not exist on gfx1201. Spell scalar adds through `s_add_u32(arch)`, `s_addc_u32(arch)` or `s_add_i32(arch)` when you need the arch-correct name.
- With unbounded records (`srd_tail(b, SRD, None)`) the `exec` mask is the only tail guard. Without the compare and `s_and_b32`, lanes past `n` write past the buffer.

## Variations

- Bound the descriptor instead of masking: compute `n * 4` into an SGPR with `s_lshl_b32` and pass it as `srd_tail(b, SRD, Some(R))`; stores at or past it are discarded. Recipe 2 does exactly this for loads and stores.
- Split the record load: `smem(b, ARGS, 2, KARG, 0)` for the pointer and `smem(b, N, 2, KARG, 8)` for `n, k`, with `args` replaced by two `s::<2>` owners. Each load destination must still lie inside one owner.
- Four consecutive outputs per lane: declare a `v::<4>` value group, use `i = 4 * (workgroup_x * 64 + tid)`, fill four consecutive results, and store with `bstore_b128`. Launch `ceil(n / 256)` workgroups; require `n` to be a multiple of four or handle the last partial four-element record separately.
