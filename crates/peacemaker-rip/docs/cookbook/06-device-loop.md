# 6. Device loop over a runtime length

One workgroup of 32 threads copies `n` u32 values, `n <= 100000`, in chunks of 32 elements; `n` is known only at launch, so the loop must run on the device.
A script `for` would unroll at emit time and cannot do this: use the Workgroup driver's loop operations and keep the loop state in scalar registers.

Kernel arguments are 24 bytes: `a` pointer at byte 0, `out` pointer at 8, `n` u32 at 16, padding at 20..24. Launch exactly **one** workgroup of **32** threads (`grid = [1,1,1]`, `block = [32,1,1]`, wave32, so one wave). `a` and `out` are distinct buffers of `n` u32; `n == 0` is legal and must touch no memory.

Variants (same script): `default` is a pre-test `loop_until`; `copy_carried` is a do-while `loop_carried` with an early exit for `n == 0`.

```rip variant=default
fn emit(spec: Spec) -> Result<Emitted, String> {
    let arch = spec.arch;
    if spec.variant != "default" && spec.variant != "copy_carried" {
        return Err(format!("unknown variant {}", spec.variant));
    }
    let carried = spec.variant == "copy_carried";
    let variant = if carried { "copy_carried" } else { "default" };
    let kspec = KernelSpec {
        kernel_id: "device_loop".into(), variant: variant.into(), arch,
        symbol: format!("device_loop_{variant}"),
        // 24 bytes: a@0, out@8, n@16, bytes 20..24 are padding and are never read.
        kernargs: KernargLayout::new(24).pointer("a", 0).pointer("out", 8)
            .hidden("n", 16, 4, "by_value"),
        user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 0, wave32: true, cu_mode: false,
    };
    let mut p = RegPlan::new(256, 104)?;
    p.s::<2>("kernarg_ptr", 0, Live::Whole)?;
    // gfx11 delivers the workgroup id in s2; reserve it even though one workgroup never reads it.
    if !arch.gfx12() { p.s::<1>("wg_x", 2, Live::Whole)?; }
    // Each descriptor is one owner. The 64-bit pointer is loaded straight into its
    // low two dwords; srd_tail then fills the other two.
    p.s::<4>("in_srd", 4, Live::Whole)?;
    p.s::<4>("out_srd", 8, Live::Whole)?;
    // Scalar temporaries: s12 = n, s13 = n * 4 (bytes), s14 = loop index, s15 spare.
    p.s::<4>("t", 12, Live::Whole)?;
    // v0 local lane (launch-provided), v1 byte offset, v2 data, v3 spare.
    p.v::<4>("v", 0, Live::Whole)?;
    // Pad to a whole 8-VGPR granule so `build --certify` sees descriptor == plan.
    p.v::<4>("granule_pad", 4, Live::Whole)?;
    let mut builder = Builder::new(kspec, p);
    let mut wg = Workgroup::new(&mut builder, arch)?;
    author(&mut wg, arch, carried)?;
    builder.finish()
}

// One 32-element chunk: load, store, then drain the store.
fn copy_chunk<T: Target>(w: &mut Wg<T>, arch: Arch) -> Result<(), String> {
    let b = w.isa();
    // Byte offset = (chunk base + lane) * 4. s14 is the scalar base, v0 the lane.
    op(b, "v_add_nc_u32_e32 v1, s14, v0", &[v(1)], &[s(14), v(0)])?;
    op(b, "v_lshlrev_b32_e32 v1, 2, v1", &[v(1)], &[v(1)])?;
    // bload(b, width, dst, voff, srd, imm_offset): buffer_load_b32 v2, v1, s[4:7].
    // Lanes at or past n*4 bytes read 0 without touching memory.
    bload(b, 1, 2, 1, 4, 0)?;
    // gfx12 wants `null` as soffset, gfx11 wants `0`. The store reads v2, so the
    // builder inserts the load wait before it; no manual load wait is needed.
    let zero = if arch.gfx12() { "null" } else { "0" };
    mem(b, format!("buffer_store_b32 v2, v1, s[8:11], {zero} offen"),
        &[], &[v(2), v(1), sr(8, 4)], MemoryClass::VmemStore)?;
    // Drain the store inside the body so the back edge carries no pending memory.
    b.wait(if arch.gfx12() { Counter::Store } else { Counter::Vs }, 0)?;
    Ok(())
}

// The Workgroup scope lives here, not in emit(): emit() mentions `builder` again at
// finish(), which would end the scope under any later `wg` use.
fn author<T: Target>(wg: &mut Wg<T>, arch: Arch, carried: bool) -> Result<(), String> {
    let end = wg.exit(".Lcopy_end")?;
    // n (u32 at byte 16) alone: the padding after it is not read.
    smem(wg.isa(), 12, 1, 0, 16)?;
    if carried {
        // A do-while runs its body once, so n == 0 must leave before the loop.
        // The compare consumes s12, so the builder waits for the SMEM load first and
        // nothing is pending on the exit path (the pointers are loaded afterwards).
        let nonzero = wg.scmp(Instruction::new("s_cmp_lg_u32 s12, 0", vec![], vec![s(12)]))?;
        wg.wave_exit_unless(nonzero, &end)?;
    }
    let b = wg.isa();
    smem(b, 4, 2, 0, 0)?;   // a   -> s[4:5], inside in_srd
    smem(b, 8, 2, 0, 8)?;   // out -> s[8:9], inside out_srd
    sop(b, "s_lshl_b32 s13, s12, 2", &[13], &[12])?;
    // num_records = n * 4 bytes for both descriptors: out-of-range lanes of the last
    // chunk read 0 and their stores are dropped.
    srd_tail(b, 4, Some(13))?;
    srd_tail(b, 8, Some(13))?;
    sop(b, "s_mov_b32 s14, 0", &[14], &[])?;
    // Retire the scalar loads before the loop so no iteration re-waits on them.
    b.wait(if arch.gfx12() { Counter::Km } else { Counter::Lgkm }, 0)?;
    if carried {
        // Script state is the emit-time chunk size, not a device variable.
        // The callback is emitted once; s14 changes on each device iteration.
        wg.loop_carried(".Lcopy_round", 32u32, |w, chunk| {
            copy_chunk(w, arch)?;
            sop(w.isa(), format!("{} s14, s14, {}", s_add_u32(arch), lit(chunk)), &[14], &[14])?;
            // The back edge is a workgroup branch: n and s14 derive only from
            // kernel arguments and constants, so every wave agrees.
            let more = w.scmp_wg_uniform(Instruction::new("s_cmp_lt_u32 s14, s12", vec![], vec![s(14), s(12)]))?;
            Ok((chunk, more))
        })?;
    } else {
        // Pre-test loop: break first, so n == 0 performs no load.
        wg.loop_until(".Lcopy_head", ".Lcopy_done", |w, brk| {
            let done = w.scmp(Instruction::new("s_cmp_ge_u32 s14, s12", vec![], vec![s(14), s(12)]))?;
            w.break_if(done, brk)?;
            copy_chunk(w, arch)?;
            sop(w.isa(), format!("{} s14, s14, 32", s_add_u32(arch)), &[14], &[14])?;
            Ok(())
        })?;
    }
    wg.end(end)
}
```

## Why

- **Pointers into descriptors.** `smem(b, dst: u8, len: u8, base: u8, byte_offset: u32) -> Result<(), String>`; `smem(b, 4, 2, 0, 0)?` loads `s[4:5]` from the kernarg pointer in `s[0:1]`. Each destination lies wholly inside one `RegPlan` owner (`in_srd` s4..7, `out_srd` s8..11, `t` s12..15); a load that spanned two owners would fail with "has no unique live owner". `srd_tail(b, srd: u8, records: Option<u8>) -> Result<(), String>` writes words 2 and 3; `srd_tail(b, 4, Some(13))?` bounds the buffer at `s13` bytes. `bload(b, width, dst, voff, srd, offset) -> Result<(), String>` is `bload(b, 1, 2, 1, 4, 0)?`. The store goes through `mem(b, text, defs, uses, MemoryClass::VmemStore) -> Result<(), String>`; `sop(b, text, defs, uses)` is its scalar twin (`sop(b, "s_mov_b32 s14, 0", &[14], &[])?`) and `op(b, text, &[RegRef], &[RegRef])` the vector one. `lit(x: u32) -> String` spells an immediate (`lit(32)` is `"32"`); `s_add_u32(arch) -> &str` is `s_add_u32` on gfx11 and `s_add_co_u32` on gfx12. List every register an instruction reads in `uses` and writes in `defs`: dependency waits and the R0603/R0604 checks depend on them.
- **gfx12 vs gfx11.** `soffset` is `null` on gfx12 and `0` on gfx11. Counters are `Counter::Store`/`Vs` (VMEM store) and `Counter::Km`/`Lgkm` (SMEM). `b.wait(counter, n) -> Result<(), String>` is explicit; example `b.wait(Counter::Vs, 0)?` on gfx1151. The builder already waits for a load before the first consumer of its destination, so only the store drain and the SMEM drain are written by hand.
- **`wg.isa()`** `-> Isa` is the builder behind the driver; example `smem(wg.isa(), 12, 1, 0, 16)?`.
- **`wg.scmp(insn) -> Result<Cond, String>`** emits a scalar compare. The `Cond` must be consumed by the very next driver branch call: any instruction in between redefines SCC and is refused. Example: `let c = wg.scmp(Instruction::new("s_cmp_ge_u32 s14, s12", vec![], vec![s(14), s(12)]))?;`. `wg.scmp_wg_uniform(insn) -> Result<WgCond, String>` is the same compare with your claim that all waves agree (operands from kernel arguments, workgroup ids, constants and counters derived from them); only the workgroup-scope forms accept it.
- **`wg.loop_until(head, exit, |w, brk| ..) -> Result<(), String>`** is the plain loop. The body is wave scope (no barrier), carries no typed state, and the back edge is `s_branch head`. The only exit is `w.break_if(cond: Cond, brk) -> Result<(), String>` to the `exit` label, which the loop places itself. Example: `w.break_if(done, brk)?;`. Code after the loop continues from the join of every `break_if` state; a loop with no `break_if` is refused ("has no break_if").
- **`wg.loop_carried(head, [state,] |w[, st]| ..) -> Result<state, String>`** is the workgroup do-while: the body always runs once, and it returns `Ok(more)` or `Ok((state, more))` with `more` from `scmp_wg_uniform`; the back edge is `s_cbranch_scc1 head` and the fall-through is the exit. Use it when the loop body holds a workgroup barrier or when typed LDS state (a region, a `Ring`, a pending-store token from `begin_write`/`ds_store`) must survive the back edge: the driver requires LDS ownership at the back edge, including whether the youngest store is drained, to equal the entry's. A store left undrained at the loop head is the Halo VerifyAttn race (see `tests/ui/loop_carried_undrained_store.rip`, rejected as R0404, and its drained twin `loop_carried_drained_store.rip`, which waits with `wg.isa().wait(Counter::Ds / Counter::Lgkm, 0)` before returning). This copy kernel has no LDS, so its state is a plain script value, the chunk size, returned unchanged; the same `Ok((st, more))` shape carries real LDS state. With no state, write `wg.loop_carried(head, |w| .. Ok(more))`. Choose `loop_until` for a wave-scope loop that can exit before its first iteration; choose `loop_carried` for a barrier/LDS loop that must run at least once.
- **Why the global store is drained at the back edge.** A buffer store is not LDS state, so the driver does not type it; the builder tracks it in its wait ledger. Draining it explicitly with the store counter leaves nothing pending at the back edge, so each iteration starts from the state the loop head already has. LDS stores have no such fallback: the driver refuses an undrained one at the back edge, so drain it yourself (see the `loop_carried` item).
- **n == 0 on the do-while.** `wg.wave_exit_unless(cond: Cond, &end) -> Result<(), String>` branches to the reserved exit when the condition fails: `wg.wave_exit_unless(nonzero, &end)?;`. It is wave scope and forbids any later workgroup barrier; a loop without barriers is fine. `wg.exit(label) -> Result<Exit, String>` reserves the exit up front (`let end = wg.exit(".Lcopy_end")?;`) and `wg.end(end) -> Result<(), String>` places it with `s_endpgm`; an exit that something branches to takes no tail.
- **Labels.** Every label (`.Lcopy_head`, `.Lcopy_done`, `.Lcopy_round`, `.Lcopy_end`) is unique and the driver places it. A script cannot add a back edge with a raw `s_branch`: the sealed builder refuses ("typed kernel: LDS, barriers, branches and loops go through peacemaker-author", R0403); backward edges exist only inside `loop_until` and `loop_carried`, and a reserved exit is placed only by `wg.end` (R0306).
- **Scope rule.** `Workgroup::new(&mut builder, arch)?` ends its scope at the next statement that mentions `builder`. All driver work is in `author` (which receives `wg`), and `builder.finish()` comes after it.
- **Plan details.** v0 is the launch-provided local thread index. n <= 100000 keeps `n * 4` far below 2^32. gfx11 reserves `s2` (workgroup id x) so the plan owns it; gfx12 passes it in `ttmp9` and this kernel does not read it.

Source: `kernels/common.rs` (`smem`, `srd_tail`, `bload`, `sop`, `mem`, `lit`, `s_add_u32`), `peacemaker-author/src/runtime.rs` (`scmp`, `scmp_wg_uniform`, `loop_until`, `break_if`, `loop_carried`, `wave_exit_unless`, `exit`, `end_with`), `peacemaker-rip/src/native.rs` (`wg_method`).

## Variations

- Compare the explicit store drain with the builder's automatic store-source waits in emitted assembly; automatic register readiness is not a general memory-ordering fence.
- Process wider chunks by raising `chunk` (carried variant) and the unrolled load/store pairs, keeping each descriptor bound at `n * 4`.
- Carry a real `LdsRegion` through `loop_carried` for a staged copy, draining its `ds_store` before returning the state.
- Use `loop_carried` with several workgroup waves and a barrier in the body; the per-iteration `n` stays workgroup-uniform.
- Check both variants on gfx1151 and gfx1201: `peacemaker-rip check kernel.rip --variant default` and `--variant copy_carried`.
