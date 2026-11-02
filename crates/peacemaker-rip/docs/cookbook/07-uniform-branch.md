# 7. Uniform branches: skip a load, skip an add

One workgroup of 32 threads computes `out[i] = a[i] + b[i]` when a runtime `flag` is non-zero and `out[i] = a[i]` when it is zero, for `i < n`, `n <= 32`.
The flag is a kernel argument, so the whole wave agrees and the branch is a scalar one: when `flag == 0` the `b` load is not even issued.

Kernel arguments are 32 bytes: `a` pointer at byte 0, `b` pointer at 8, `out` pointer at 16, `n` u32 at 24, `flag` u32 at 28. Launch exactly **one** workgroup of **32** threads (`grid = [1,1,1]`, `block = [32,1,1]`, wave32, one wave). `a`, `b`, `out` are separate buffers of `n` u32 each; `n == 0` exits before touching memory; any non-zero `flag` (1, 7, `u32::MAX`) means add.

```rip variant=default
fn emit(spec: Spec) -> Result<Emitted, String> {
    let arch = spec.arch;
    let kspec = KernelSpec {
        kernel_id: "uniform_branch".into(), variant: "default".into(), arch,
        symbol: "uniform_branch".into(),
        // 32 bytes: a@0, b@8, out@16, n@24, flag@28.
        kernargs: KernargLayout::new(32).pointer("a", 0).pointer("b", 8).pointer("out", 16)
            .hidden("n", 24, 4, "by_value").hidden("flag", 28, 4, "by_value"),
        user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 0, wave32: true, cu_mode: false,
    };
    let mut p = RegPlan::new(256, 104)?;
    p.s::<2>("kernarg_ptr", 0, Live::Whole)?;
    // gfx11 delivers the workgroup id in s2; reserve it even though it is unused here.
    if !arch.gfx12() { p.s::<1>("wg_x", 2, Live::Whole)?; }
    // One owner per descriptor; each 64-bit pointer is loaded into its low pair.
    p.s::<4>("a_srd", 4, Live::Whole)?;
    p.s::<4>("b_srd", 8, Live::Whole)?;
    p.s::<4>("out_srd", 12, Live::Whole)?;
    // s16 = n, s17 = flag (one 8-byte load), s18 = n * 4 bytes, s19 spare.
    p.s::<4>("args", 16, Live::Whole)?;
    // v0 local lane (launch-provided), v1 byte offset, v2 a then out, v3 b.
    p.v::<4>("v", 0, Live::Whole)?;
    // Pad to a whole 8-VGPR granule so `build --certify` sees descriptor == plan.
    p.v::<4>("granule_pad", 4, Live::Whole)?;
    let mut builder = Builder::new(kspec, p);
    let mut wg = Workgroup::new(&mut builder, arch)?;
    author(&mut wg, arch)?;
    builder.finish()
}

// The Workgroup scope lives here, not in emit(): emit() mentions `builder` again at
// finish(), which would end the scope under any later `wg` use.
fn author<T: Target>(wg: &mut Wg<T>, arch: Arch) -> Result<(), String> {
    let end = wg.exit(".Lbranch_end")?;
    // n and flag are adjacent: one s_load_b64 into s[16:17], inside the `args` owner.
    smem(wg.isa(), 16, 2, 0, 24)?;
    // n != 0, else leave. The compare reads s16, so the builder first waits for the
    // SMEM load: nothing is pending on the exit path (the pointers load afterwards).
    let nonzero = wg.scmp(Instruction::new("s_cmp_lg_u32 s16, 0", vec![], vec![s(16)]))?;
    wg.wave_exit_unless(nonzero, &end)?;
    let b = wg.isa();
    smem(b, 4, 2, 0, 0)?;    // a   -> s[4:5],   inside a_srd
    smem(b, 8, 2, 0, 8)?;    // b   -> s[8:9],   inside b_srd
    smem(b, 12, 2, 0, 16)?;  // out -> s[12:13], inside out_srd
    sop(b, "s_lshl_b32 s18, s16, 2", &[18], &[16])?;
    // Every descriptor is bounded at n * 4 bytes: lanes with i >= n read 0 from a/b
    // without touching memory, and their stores to out are dropped.
    srd_tail(b, 4, Some(18))?;
    srd_tail(b, 8, Some(18))?;
    srd_tail(b, 12, Some(18))?;
    op(b, "v_lshlrev_b32_e32 v1, 2, v0", &[v(1)], &[v(0)])?;
    // Define v3 on every path: when flag == 0 the load below is skipped.
    op(b, "v_mov_b32_e32 v3, 0", &[v(3)], &[])?;
    // bload(b, width, dst, voff, srd, imm_offset): out = a is the common part.
    bload(b, 1, 2, 1, 4, 0)?;

    // skip_if: branch over the body when the condition HOLDS. flag == 0 -> no B load.
    let no_b = wg.scmp(Instruction::new("s_cmp_eq_u32 s17, 0", vec![], vec![s(17)]))?;
    wg.skip_if(no_b, ".Lskip_b", |w| {
        bload(w.isa(), 1, 3, 1, 8, 0)?;
        Ok(())
    })?;

    // skip_unless: run the body only when the condition HOLDS. flag != 0 -> add.
    // The add reads v3, so the builder waits for the B load inside the body.
    let has_b = wg.scmp(Instruction::new("s_cmp_lg_u32 s17, 0", vec![], vec![s(17)]))?;
    wg.skip_unless(has_b, ".Lskip_add", |w| {
        op(w.isa(), "v_add_nc_u32_e32 v2, v2, v3", &[v(2)], &[v(2), v(3)])?;
        Ok(())
    })?;

    // gfx12 wants `null` as soffset, gfx11 wants `0`. The store reads v2: the builder
    // waits for a's load (and the add) before it.
    let zero = if arch.gfx12() { "null" } else { "0" };
    mem(wg.isa(), format!("buffer_store_b32 v2, v1, s[12:15], {zero} offen"),
        &[], &[v(2), v(1), sr(12, 4)], MemoryClass::VmemStore)?;
    // Drain the store before the kernel ends.
    wg.isa().wait(if arch.gfx12() { Counter::Store } else { Counter::Vs }, 0)?;
    wg.end(end)
}
```

## Why

- **Uniform condition.** `wg.scmp(insn) -> Result<Cond, String>` emits a scalar compare (`s_cmp_*`); SCC is wave-uniform, so a `Cond` may steer wave-level control. Example: `let nonzero = wg.scmp(Instruction::new("s_cmp_lg_u32 s16, 0", vec![], vec![s(16)]))?;`. A `Cond` must be consumed by the very next driver branch call. Any instruction between the compare and the branch redefines SCC and the driver refuses it, so write the compare directly before `skip_if`/`skip_unless`, and issue a second compare for the second branch.
- **`wg.skip_if(cond, label, [state,] |w[, st]| ..) -> Result<state, String>`** branches to `label` when `cond` holds and otherwise runs the body; with no state it returns unit. Example: `wg.skip_if(no_b, ".Lskip_b", |w| { bload(w.isa(), 1, 3, 1, 8, 0)?; Ok(()) })?;`. **`wg.skip_unless(cond, label, [state,] |w[, st]| ..) -> Result<state, String>`** is the mirror: the body runs only when `cond` holds. Example: `wg.skip_unless(has_b, ".Lskip_add", |w| { op(w.isa(), "v_add_nc_u32_e32 v2, v2, v3", &[v(2)], &[v(2), v(3)])?; Ok(()) })?;`. The driver places the label right after the body. The body is wave scope (no barrier) and must leave LDS ownership as it found it; the builder joins the skipped path's wait ledger into the body's at the label, so the add's wait for the B load is correct on both paths. A state value passed between `label` and the callback is returned from the callback and handed back by the call (`skip_if(c, l, st, |w, st| .. Ok(st))`).
- **`wg.wave_exit_unless(cond: Cond, &end) -> Result<(), String>`** leaves through the reserved exit when `cond` fails: `wg.wave_exit_unless(nonzero, &end)?;`. It is wave scope; no workgroup barrier may follow it. `wg.exit(label) -> Result<Exit, String>` reserves the exit before any branch (`let end = wg.exit(".Lbranch_end")?;`); `wg.end(end) -> Result<(), String>` places it and `s_endpgm`. An exit that something branches to takes no tail. The exit path must carry no outstanding memory operation, which is why the pointer loads come after the `n` compare.
- **Labels (R0306).** The driver, not the script, places each skip label right after the body, so a skip target must be a fresh name that nothing has declared or placed before (`.Lskip_b`, `.Lskip_add`, `.Lbranch_end` are distinct). Reusing a name is "duplicate label"; a branch can only go forward; a reserved exit is placed only by `wg.end` ("only Workgroup::end places it"). A script cannot invent control flow with a raw `s_branch`/`s_cbranch_*` through `op(..)`: the sealed builder refuses it ("typed kernel: LDS, barriers, branches and loops go through peacemaker-author", R0403). When a helper may run twice, build the label with a counter, for example `format!(".Lskip_b{k}")`.
- **Ownership and descriptors.** `smem(b, dst: u8, len: u8, base: u8, byte_offset: u32) -> Result<(), String>` loads `len` (1, 2, 4, 8) dwords; `smem(b, 4, 2, 0, 0)?` fills `s[4:5]`. Each destination sits wholly inside one `RegPlan` owner (`a_srd` s4..7, `b_srd` s8..11, `out_srd` s12..15, `args` s16..19). `n`/`flag` load into `args` as one 8-byte read. `srd_tail(b, srd: u8, records: Option<u8>) -> Result<(), String>` fills descriptor words 2 and 3: `srd_tail(b, 8, Some(18))?` bounds `b` at `s18` bytes. Input descriptors read zero past the bound without touching memory, and the output descriptor drops stores past it, which is how a 32-lane wave serves `n < 32` without EXEC masking. `bload(b, width, dst, voff, srd, offset) -> Result<(), String>`: `bload(b, 1, 2, 1, 4, 0)?`. `mem(b, text, defs, uses, MemoryClass::VmemStore) -> Result<(), String>`, `op(b, text, &[RegRef], &[RegRef])` and `sop(b, text, &[u8], &[u8])` are the raw helpers; list every register read in `uses` and written in `defs`, since waits and R0603/R0604 rely on them.
- **Arch spellings.** `soffset` is `null` on gfx12 and `0` on gfx11; the store counter is `Counter::Store` on gfx12 and `Counter::Vs` on gfx11. `wg.isa().wait(counter, n)` is explicit: `wg.isa().wait(Counter::Vs, 0)?` on gfx1151. Load readiness is automatic.
- **Scope rule.** `Workgroup::new(&mut builder, arch)?` ends its scope at the next statement that mentions `builder`; all driver work is in `author`, and `builder.finish()` follows it.
- **Definedness.** `v3` is zeroed first so every path defines it, because the `b` load is skipped when `flag == 0`.

Source: `peacemaker-author/src/runtime.rs` (`scmp`, `skip_if`, `skip_unless`, `wave_exit_unless`, `exit`, `end_with`, `fresh`), `peacemaker-isa/src/author.rs` (label and exit rules), `kernels/common.rs` (`smem`, `srd_tail`, `bload`, `sop`, `mem`), `peacemaker-rip/src/native.rs` (`wg_method`).

## Variations

- Use `wg.if_else(cond, else_label, join, |w| .., |w| ..)` for a two-arm branch (`out = a + b` versus `out = a - b`) instead of two one-armed skips; neither arm may hold a barrier or change LDS ownership.
- Replace the `flag` compare with `scmp_wg_uniform` and `wg.wg_skip_if(..)` when the body must contain a barrier (several waves); the claim that every wave agrees is then yours.
- Use `wg.wg_if_else(wgcond, else_label, join, |w| .., |w| ..)` for a workgroup-uniform two-arm branch whose arms hold barriers or change LDS ownership; the arms must end in the same ownership state (and the same ring `cur` index and live buffers), and the call places the join label.
- Use `exec_if(cond, |w| ..)` to predicate a body per whole wave while still issuing every instruction.
- Swap `skip_if`/`skip_unless` by inverting the compare (`s_cmp_lg_u32` for `skip_if`, `s_cmp_eq_u32` for `skip_unless`); the emitted branch flips between `s_cbranch_scc1` and `s_cbranch_scc0`.
- Load `flag` and `n` separately (`smem(b, 16, 1, 0, 24)?; smem(b, 17, 1, 0, 28)?;`) when the loads must be ordered differently; both stay inside the `args` owner.
