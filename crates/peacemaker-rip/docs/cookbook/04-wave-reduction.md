# 4. Wave reduction

Reduce one 32-element u32 block per workgroup to its minimum using only lane-to-lane exchange, with no LDS and no barrier.
Launch one 32-thread workgroup (one wave32) per block: `output[wg.x] = min(input[32*wg.x .. 32*wg.x + 32])`.

The kernarg record is 16 bytes: `input` pointer at 0, `output` pointer at 8. There is no element count and no tail: the grid is exactly the number of blocks, and each block is full. The minimum may sit in either 16-lane half of the wave.

```rip variant=wave_min
// Entry: pick the variant, then hand the builder to the Workgroup driver.
fn emit(spec: Spec) -> Result<Emitted, String> {
    let arch = spec.arch;
    if spec.variant != "wave_min" { return Err("expected variant wave_min".into()); }
    let mut builder = Builder::new(kernel_spec(arch), reg_plan(arch)?);
    let mut wg = Workgroup::new(&mut builder, arch)?;
    // The scope of `wg` ends at the next statement that names `builder`.
    author(&mut wg, arch)?;
    builder.finish()
}

// Spec: the ABI. One symbol per architecture; 32 threads; no LDS.
fn kernel_spec(arch: Arch) -> KernelSpec {
    KernelSpec {
        kernel_id: "wave_min".into(), variant: "wave_min".into(), arch,
        symbol: format!("wave_min_{}", arch.name()),
        kernargs: KernargLayout::new(16).pointer("input", 0).pointer("output", 8),
        user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 0, wave32: true, cu_mode: false,
    }
}

// Spec: register ownership. Widths are 1, 2, 4 or 8; SGPR ranges are aligned to their width.
fn reg_plan(arch: Arch) -> Result<RegPlan, String> {
    let mut p = RegPlan::new(256, 104)?;
    p.s::<2>("kernarg_ptr", 0, Live::Whole)?;
    // gfx11 delivers the workgroup id in s2; gfx12 delivers it in ttmp9 and s2 holds nothing.
    if !arch.gfx12() { p.s::<1>("wg_x_in", 2, Live::Whole)?; }
    p.s::<4>("input_srd", 4, Live::Whole)?;
    p.s::<4>("output_srd", 8, Live::Whole)?;
    p.s::<4>("arg_pointers", 12, Live::Whole)?;   // input = s12:13, output = s14:15
    // s16 wg id, s17 input byte offset, s18 output byte offset,
    // s19 input records, s20 output records, s22/s23 permlane lane selects.
    p.s::<8>("scalars", 16, Live::Whole)?;
    p.v::<4>("lanes", 0, Live::Whole)?;           // v0 lane id (given), v1 byte offset, v2 value, v3 peer
    Ok(p)
}

// Author: the instruction stream. It sees only `wg` and `arch`, never the builder.
fn author<T: Target>(wg: &mut Wg<T>, arch: Arch) -> Result<(), String> {
    let end = wg.exit(".Lwave_min_end")?;
    let isa = wg.isa();
    smem(isa, 12, 4, 0, 0)?;                                 // both pointers in one 16-byte load
    if arch.gfx12() { sop(isa, "s_mov_b32 s16, ttmp9", &[16], &[])?; }
    else { sop(isa, "s_mov_b32 s16, s2", &[16], &[2])?; }
    sop(isa, "s_lshl_b32 s17, s16, 7", &[17], &[16])?;       // 32 u32 = 128 bytes per block
    sop(isa, "s_lshl_b32 s18, s16, 2", &[18], &[16])?;       // one u32 result per block
    sop(isa, format!("s_mov_b32 s19, {}", lit(128)), &[19], &[])?;
    sop(isa, "s_mov_b32 s20, 4", &[20], &[])?;
    // The first use of s12 inserts the SMEM wait. Both descriptors start at their block.
    add64(isa, 4, 12, 17)?;
    srd_tail(isa, 4, Some(19))?;
    add64(isa, 8, 14, 18)?;
    srd_tail(isa, 8, Some(20))?;                             // only offset 0 is inside 4 bytes
    sop(isa, "s_mov_b32 s22, 0x76543210", &[22], &[])?;      // identity lane selects 0..7
    sop(isa, "s_mov_b32 s23, 0xfedcba98", &[23], &[])?;      // identity lane selects 8..15
    op(isa, "v_lshlrev_b32_e32 v1, 2, v0", &[v(1)], &[v(0)])?;
    bload(isa, 1, 2, 1, 4, 0)?;                              // v2 = input[32*wg + lane]
    // Inside each 16-lane row: lane ^ 1, 2, 4, 8. The first DPP use of v2 inserts the load wait.
    for mask in [1u32, 2, 4, 8] {
        op(isa, format!("v_mov_b32_dpp v3, v2 row_xmask:{mask} row_mask:0xf bank_mask:0xf"),
           &[v(3)], &[v(2)])?;
        op(isa, "v_min_u32_e32 v2, v2, v3", &[v(2)], &[v(2), v(3)])?;
    }
    // Across the two rows: lane i reads lane i of the other row.
    op(isa, "v_permlanex16_b32 v3, v2, s22, s23", &[v(3)], &[v(2), s(22), s(23)])?;
    op(isa, "v_min_u32_e32 v2, v2, v3", &[v(2)], &[v(2), v(3)])?;
    // All lanes hold the minimum; lane l stores at byte 4*l and only lane 0 is inside 4 bytes.
    let zero = if arch.gfx12() { "null" } else { "0" };
    mem(isa, format!("buffer_store_b32 v2, v1, s[8:11], {zero} offen"),
        &[], &[v(2), v(1), sr(8, 4)], MemoryClass::VmemStore)?;
    wg.end(end)
}
```

## Why

- **Emit, spec and author are separate.** `emit` owns the variant check and the `Builder`/`Workgroup` pair; `kernel_spec` and `reg_plan` describe the ABI and register ownership; `author` is the instruction stream and receives only `wg` and `arch`. Because `author` never names `builder`, the Workgroup scope runs through it to `builder.finish()`. `let isa = wg.isa();` is a handle to the builder behind the driver, not the builder variable, so it does not close the scope.
- **Reduction shape.** `v_mov_b32_dpp ... row_xmask:m` copies the value of lane `l ^ m` inside a 16-lane row into a scratch VGPR; `v_min_u32_e32` folds it. After `m = 1, 2, 4, 8` every lane holds its row minimum. `v_permlanex16_b32` with identity selects (`s22 = 0x76543210`, `s23 = 0xfedcba98`) hands each lane the same-numbered lane of the other row, and one more fold makes all 32 lanes hold the block minimum. Minimum is exact in any order, so the exchange order is free.
- **One store, no tail.** Every lane executes the store with offset `4*lane`. The output descriptor has `num_records = 4` (`s20`), so only lane 0 is inside it; the others are discarded by the buffer unit and EXEC is never touched. The input descriptor is bounded to 128 bytes, so a stray lane could not read past its block.
- **Automatic waits.** The builder's ledger tracks the SMEM load of the kernarg pointers and the buffer load of the block and emits a wait only at the first instruction that reads the result. No `wait` appears in the script, and there is no wait before the final store: its sources are not redefined, and `s_endpgm` follows. A manual counter wait would also have to be spelled per architecture (see the pitfalls below).

Waits the builder emits for this kernel:

| Purpose | gfx1151 | gfx1201 |
|---|---|---|
| Kernarg ready before `add64` | `s_waitcnt lgkmcnt(0)` | `s_wait_kmcnt 0x0` |
| Block ready before the first DPP | `s_waitcnt vmcnt(0)` | `s_wait_loadcnt 0x0` |

Architecture differences used above:

| Item | gfx1151 | gfx1201 |
|---|---|---|
| Workgroup id x | `s2` (declared in the plan) | `ttmp9`; `s2` holds nothing |
| Scalar add pair inside `add64` | `s_add_u32` / `s_addc_u32` | `s_add_co_u32` / `s_add_co_ci_u32` |
| `soffset` of buffer ops | `0` | `null` (`bload` does this itself) |
| Zero SMEM offset | spelled `null` | spelled `0x0` (`smem` does this itself) |
| `v_min_u32_e32`, `v_mov_b32_dpp`, `v_permlanex16_b32` | same text | same text |

Baseline pitfalls this recipe avoids:

- Fused `v_max_f32_dpp`-style opcodes were rejected at assembly (no source position). Use `v_mov_b32_dpp` into a scratch register plus an ordinary `_e32` fold.
- `v_permlane16_b32` is not the cross-half exchange; `v_permlanex16_b32` is.
- A decimal literal outside the inline range (`128`) is rejected by the assembler. Format it with `lit(128)`, which prints `0x80`.
- Reading `s2` as the workgroup id on gfx12 compiled in the baseline and returned wrong answers; the checker now reports it as R0604. Copy from `ttmp9` on gfx12.
- Declaring only the VGPRs in an instruction's uses loses a dependency. `v_permlanex16_b32` reads `s22` and `s23`; the store reads the descriptor `s[8:11]`. Declare every register read, scalar ones included.
- `v_min_u32` is the unsigned minimum. gfx12 spells the float minimum `v_min_num_f32_e32`, so a float variant must branch on `arch.gfx12()`.

Source: `kernels/common.rs` (`smem`, `add64`, `srd_tail`, `bload`); `ledger.rs` (automatic dependency waits); `peacemaker-rip/docs/ERRORS.md` R0603 and R0604.

## Variations

- Maximum or signed minimum: change only the fold mnemonic and check every architecture; the exchange pattern does not change.
- Two values per lane (64-element blocks): issue two `bload`s at byte offsets 0 and 128, fold them with one `v_min_u32_e32`, then run the same exchange. Make the block constants 256 bytes.
- Row-only reduction: drop the `v_permlanex16_b32` fold to get one minimum per 16-lane row (two per wave); the two rows then need two output slots and distinct store offsets.
- Store with EXEC instead of a descriptor: write `exec_lo` to `1` around the store and restore it to `-1`; the descriptor form needs no EXEC edit and keeps the kernel branch-free.
- Several blocks per workgroup: use one wave per block and derive the block index from `wg.x` and the wave number; the exchange itself stays inside a wave.
