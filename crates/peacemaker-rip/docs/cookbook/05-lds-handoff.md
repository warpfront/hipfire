# 5. LDS handoff

Reverse one 64-element u32 block per workgroup through 256 bytes of LDS, so that data written by one wave is read by the other.
Launch one 64-thread workgroup (two wave32 waves) per block: `output[64*wg.x + t] = input[64*wg.x + 63 - t]`.

The kernarg record is 16 bytes: `input` pointer at 0, `output` pointer at 8. There is no element count and no tail: every block is full. Thread `t` of wave 0 reads what a thread of wave 1 stored, and the reverse, so the barrier is load-bearing.

```rip variant=reverse
// Entry: pick the variant, then hand the builder to the Workgroup driver.
fn emit(spec: Spec) -> Result<Emitted, String> {
    let arch = spec.arch;
    if spec.variant != "reverse" { return Err("expected variant reverse".into()); }
    let mut builder = Builder::new(kernel_spec(arch), reg_plan(arch)?);
    let mut wg = Workgroup::new(&mut builder, arch)?;
    // The scope of `wg` ends at the next statement that names `builder`.
    author(&mut wg, arch)?;
    builder.finish()
}

// Spec: the ABI. 64 threads and the 256 bytes of LDS the region will own.
fn kernel_spec(arch: Arch) -> KernelSpec {
    KernelSpec {
        kernel_id: "reverse".into(), variant: "reverse".into(), arch,
        symbol: format!("reverse_{}", arch.name()),
        kernargs: KernargLayout::new(16).pointer("input", 0).pointer("output", 8),
        user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 64, group_segment_fixed_size: 256, wave32: true, cu_mode: false,
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
    p.s::<4>("scalars", 16, Live::Whole)?;        // s16 wg id, s17 block byte offset, s18 records
    // v0 thread id (given), v1 own byte offset, v2 mirrored offset, v3 loaded value, v4 mirrored value.
    p.v::<8>("lanes", 0, Live::Whole)?;
    Ok(p)
}

region Reverse;
// Author: the instruction stream. It sees only `wg` and `arch`, never the builder.
fn author<T: Target>(wg: &mut Wg<T>, arch: Arch) -> Result<(), String> {
    let block = wg.lds::<Reverse>("block", 0, 256)?;
    let end = wg.exit(".Lreverse_end")?;
    let isa = wg.isa();
    smem(isa, 12, 4, 0, 0)?;                                 // both pointers in one 16-byte load
    if arch.gfx12() { sop(isa, "s_mov_b32 s16, ttmp9", &[16], &[])?; }
    else { sop(isa, "s_mov_b32 s16, s2", &[16], &[2])?; }
    sop(isa, "s_lshl_b32 s17, s16, 8", &[17], &[16])?;       // 64 u32 = 256 bytes per block
    sop(isa, format!("s_mov_b32 s18, {}", lit(256)), &[18], &[])?;
    // The first use of s12 inserts the SMEM wait. Both descriptors are bounded to the block.
    add64(isa, 4, 12, 17)?;
    srd_tail(isa, 4, Some(18))?;
    add64(isa, 8, 14, 17)?;
    srd_tail(isa, 8, Some(18))?;
    op(isa, "v_lshlrev_b32_e32 v1, 2, v0", &[v(1)], &[v(0)])?;                        // 4*t
    op(isa, format!("v_sub_nc_u32_e32 v2, {}, v1", lit(252)), &[v(2)], &[v(1)])?;     // 4*(63-t)
    bload(isa, 1, 3, 1, 4, 0)?;                              // v3 = input[64*wg + t]

    // Stage: acquire the region for writing; no `?`, begin_write returns (region, token).
    let staged = wg.begin_write(block);
    // The first use of v3 inserts the load wait. ds_store takes the tuple and returns it.
    let staged = wg.ds_store(staged, Instruction::new("ds_store_b32 v1, v3",
        vec![], vec![v(1), v(3)]).memory(MemoryClass::DsStore))?;
    let (block, pending) = staged;
    // Publish: drain the store explicitly, then the workgroup barrier.
    let drained = wg.wait(pending)?;
    let (block,) = wg.barrier((ready(block, drained),))?;

    // Consume: any wave may now read any byte. The result is waited for at its first use.
    wg.ds_load(&block, Instruction::new("ds_load_b32 v4, v2",
        vec![v(4)], vec![v(2)]).memory(MemoryClass::DsLoad))?;
    let isa = wg.isa();
    let zero = if arch.gfx12() { "null" } else { "0" };
    mem(isa, format!("buffer_store_b32 v4, v1, s[8:11], {zero} offen"),
        &[], &[v(4), v(1), sr(8, 4)], MemoryClass::VmemStore)?;
    wg.end(end)
}
```

## Why

- **Emit, spec and author are separate.** `emit` owns the variant check and the `Builder`/`Workgroup` pair; `kernel_spec` and `reg_plan` describe the ABI and register ownership; `author` is the instruction stream and receives only `wg` and `arch`. Because `author` never names `builder`, the Workgroup scope runs through it to `builder.finish()`.
- **Ownership protocol.** `wg.lds::<Reverse>("block", 0, 256)?` names the 256 bytes and returns the region Free. `wg.begin_write(block)` returns `(region, token)` and cannot fail. `wg.ds_store(staged, insn)?` takes that tuple and returns it again. `wg.wait(pending)?` returns the drained token, and `wg.barrier((ready(block, drained),))?` returns the region Published. `wg.ds_load(&block, insn)?` accepts only a Published region.
- **Which waits are automatic.** The builder inserts every wait whose purpose is a *register dependency*: the kernarg SMEM wait before `add64`, the buffer-load wait before `ds_store` reads `v3`, and the LDS-load wait before `buffer_store_b32` reads `v4`. It does not decide *publication*. The barrier does not wait for memory counters, so draining the stores is your explicit `wg.wait(..)`, and the barrier is your explicit `wg.barrier(..)`. Both are mandatory, and the driver checks that you wrote them.
- **Why 64 threads.** With 32 threads the LDS round trip could be misread as a wave-local exchange. With two waves, thread `t` (wave 0) loads the element thread `63 - t` (wave 1) stored, so without the drain and the barrier the data is not guaranteed to have landed.
- **Mutant 1, missing wait: R0301.** Replace `ready(block, drained)` by `ready(block, pending)` (or drop `wg.wait`). The driver refuses the barrier: `barrier would publish block with an undrained LDS store: wait for it first (s_barrier does not wait for memory counters)`. Code R0301.
- **Mutant 2, missing barrier: R0303.** Keep the wait but call `wg.ds_load(&block, ..)` on the region from the `ds_store` tuple. The region is still Writing: `block is not published: it is Writing`. Code R0303.
- **After the read.** A Published region may still be read by other waves. To store into it again, first `wg.barrier((retire(block),))?`; storing without it is R0302.

Waits and barrier spellings the builder emits for this kernel:

| Purpose | gfx1151 | gfx1201 |
|---|---|---|
| Kernarg ready before `add64` | `s_waitcnt lgkmcnt(0)` | `s_wait_kmcnt 0x0` |
| Block ready before `ds_store` | `s_waitcnt vmcnt(0)` | `s_wait_loadcnt 0x0` |
| Explicit store drain (`wg.wait`) | `s_waitcnt lgkmcnt(0)` | `s_wait_dscnt 0x0` |
| Barrier (`wg.barrier`) | `s_barrier` | `s_barrier_signal -1`, `s_barrier_wait 0xffff` |
| LDS load ready before the store | `s_waitcnt lgkmcnt(0)` | `s_wait_dscnt 0x0` |

Architecture differences used above:

| Item | gfx1151 | gfx1201 |
|---|---|---|
| Workgroup id x | `s2` (declared in the plan) | `ttmp9`; `s2` holds nothing |
| Scalar add pair inside `add64` | `s_add_u32` / `s_addc_u32` | `s_add_co_u32` / `s_add_co_ci_u32` |
| `soffset` of buffer ops | `0` | `null` (`bload` does this itself) |
| LDS counter | `Counter::Lgkm` (shared with SMEM) | `Counter::Ds` |
| `ds_store_b32`, `ds_load_b32`, `v_sub_nc_u32_e32` | same text | same text |

Baseline pitfalls this recipe avoids:

- `?` on `wg.begin_write(..)`: it returns a tuple, not a `Result` (the baseline error was `` `?` needs a Result or Option, found tuple ``).
- A barrier without a drained token was refused with a clear text but no position; the shape above makes the order explicit.
- A manual wait with the wrong counter name failed with `counter/architecture mismatch`. Here only the driver's `wait` names a counter, and it picks the right one per architecture.
- A decimal literal outside the inline range (`252`, `256`) is rejected by the assembler. Format it with `lit(..)`.
- Reading `s2` as the workgroup id on gfx12 is R0604; copy from `ttmp9` there.
- Declare every register an instruction reads, scalar ones included: the store reads the descriptor `s[8:11]`.

Source: `peacemaker-author/src/runtime.rs` (region phases and the barrier rules); `peacemaker-isa/src/lib.rs` (`Builder::emit`, barrier lowering); `ledger.rs`; `kernels/common.rs`; `peacemaker-rip/docs/ERRORS.md` R0301, R0302, R0303.

## Variations

- Bigger blocks: 128 or 256 threads need a bigger LDS region and `kernel_spec` sizes; scale the shifts (`s_lshl_b32 ..., 9` for 512 bytes), the mirrored offset constant and the descriptor records together.
- More elements per thread: chain several `wg.ds_store(staged, ..)` calls (for example `ds_store_b32 v1, v3 offset:256` into a larger region), then one `wg.wait` and one `wg.barrier`.
- Reuse the region: read, then `wg.barrier((retire(block),))?`, which returns it Free for the next `begin_write` (R0302 otherwise).
- Split the region: `wg.split(block, first)` makes two independently owned parts when different waves write different halves.
- Divergent producers: keep the barrier outside any wave-conditional body (R0304); use the driver's `skip_if`/`skip_unless` around the stores only.
