# 9. Kernarg loads and waits

Read a 40-byte launch record without splitting a multi-dword load across register owners.
Compute `out[i] = a[i] * alpha + i` for `first <= i < n`, launching `ceil((n-first)/32)` workgroups of 32 threads.

The record contains three pointers (`a`, `out`, and an unused reserved pointer), two u32 values (`n`, `first`), and one f32 (`alpha`). Bytes 36–39 are padding. The reserved pointer is part of this ABI but is not dereferenced. Inputs and output are distinct; `0 <= first <= n`, and `n * 4` must fit u32.

```rip variant=kernarg_affine
fn emit(spec: Spec) -> Result<Emitted, String> {
    let arch = spec.arch;
    let kspec = KernelSpec {
        kernel_id: "kernarg_affine".into(), variant: "kernarg_affine".into(), arch,
        symbol: "kernarg_affine".into(),
        kernargs: KernargLayout::new(40).pointer("a", 0).pointer("out", 8)
            .pointer("reserved", 16).hidden("n", 24, 4, "by_value")
            .hidden("first", 28, 4, "by_value").hidden("alpha", 32, 4, "by_value"),
        user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 0, wave32: true, cu_mode: false,
    };
    let mut p = RegPlan::new(256, 104)?;
    p.s::<2>("kernarg_ptr", 0, Live::Whole)?;
    if !arch.gfx12() { p.s::<1>("wg_x", 2, Live::Whole)?; }
    // One owner contains the entire eight-dword destination of smem below.
    p.s::<8>("record_head", 8, Live::Whole)?;
    p.s::<1>("alpha", 4, Live::Whole)?;
    p.s::<1>("nbytes", 5, Live::Whole)?;
    p.s::<1>("group_base", 6, Live::Whole)?;
    p.s::<4>("a_srd", 16, Live::Whole)?;
    p.s::<4>("out_srd", 20, Live::Whole)?;
    p.v::<8>("vectors", 0, Live::Whole)?;
    let mut builder = Builder::new(kspec, p);
    let mut wg = Workgroup::new(&mut builder, arch)?;
    author(&mut wg, arch)?;
    builder.finish()
}
region Affine;
fn author<T: Target>(wg: &mut Wg<T>, arch: Arch) -> Result<(), String> {
    let end = wg.exit(".Laffine_end")?;
    let b = wg.isa();
    smem(b, 8, 8, 0, 0)?;   // a=s8:9, out=s10:11, reserved=s12:13, n=s14, first=s15
    smem(b, 4, 1, 0, 32)?;  // alpha=s4; do not read the padding
    // Consuming s14 inserts an SMEM wait automatically.
    sop(b, "s_lshl_b32 s5, s14, 2", &[5], &[14])?;
    sop(b, "s_mov_b32 s16, s8", &[16], &[8])?;
    sop(b, "s_mov_b32 s17, s9", &[17], &[9])?;
    sop(b, "s_mov_b32 s20, s10", &[20], &[10])?;
    sop(b, "s_mov_b32 s21, s11", &[21], &[11])?;
    srd_tail(b, 16, Some(5))?;
    srd_tail(b, 20, Some(5))?;
    // Launch ABI differs: gfx11 supplies s2; gfx12 supplies ttmp9.
    if arch.gfx12() { sop(b, "s_mov_b32 s6, ttmp9", &[6], &[])?; }
    else { sop(b, "s_mov_b32 s6, s2", &[6], &[2])?; }
    sop(b, "s_lshl_b32 s6, s6, 5", &[6], &[6])?;
    sop(b, format!("{} s6, s6, s15", s_add_u32(arch)), &[6], &[6, 15])?;
    // v0 is the launch-provided local thread index (one wave per workgroup).
    op(b, "v_add_nc_u32_e32 v1, s6, v0", &[v(1)], &[s(6), v(0)])?;
    op(b, "v_lshlrev_b32_e32 v2, 2, v1", &[v(2)], &[v(1)])?;
    bload(b, 1, 3, 2, 16, 0)?;
    op(b, "v_cvt_f32_u32_e32 v4, v1", &[v(4)], &[v(1)])?;
    // The first use of v3 inserts a VMEM-load wait; no manual readiness wait.
    op(b, "v_mul_f32_e32 v3, s4, v3", &[v(3)], &[s(4), v(3)])?;
    op(b, "v_add_f32_e32 v3, v4, v3", &[v(3)], &[v(4), v(3)])?;
    let zero = if arch.gfx12() { "null" } else { "0" };
    mem(b, format!("buffer_store_b32 v3, v2, s[20:23], {zero} offen"),
        &[], &[v(3), v(2), sr(20, 4)], MemoryClass::VmemStore)?;
    // Explicit ordering only: the stores have completed before we end the wave.
    b.wait(if arch.gfx12() { Counter::Store } else { Counter::Vs }, 0)?;
    wg.end(end)
}
```

## Why

- `smem(b, dst: u8, len: u8, base: u8, byte_offset: u32) -> Result<(), String>` accepts widths 1, 2, 4, 8 dwords. Example: `smem(b, 8, 8, 0, 0)?`. The first load owns `s[8:15]` as **one** aligned `RegPlan` allocation. Eight separate one-register allocations are not equivalent. A smaller load wholly inside one owner is allowed; a load spanning adjacent owners is not.
- `srd_tail(b, base: u8, records: Option<u8>) -> Result<(), String>` fills descriptor words 2/3; example: `srd_tail(b, 16, Some(5))?`. For these raw byte-addressed buffers, the bound is `n * 4` bytes. Tail lanes read zero and their stores are suppressed, without changing EXEC.
- Definitions and uses passed to `op`/`mem` drive dependency tracking. Include every register actually read. Automatic waits protect SMEM and VMEM consumers, not arbitrary external ordering.
- `b.wait(counter: Counter, target: u8) -> Result<(), String>` emits an explicit counter wait; example: `b.wait(Counter::Store, 0)?` on gfx1201. Here it completes outstanding stores, rather than making a loaded value ready.

The emitted assembly has the following wait spellings (the executable cookbook test checks their presence and order):

| Purpose | gfx1151 | gfx1201 |
|---|---|---|
| Automatic kernarg readiness | `s_waitcnt lgkmcnt(0)` | `s_wait_kmcnt 0x0` |
| Automatic buffer-load readiness | `s_waitcnt vmcnt(0)` | `s_wait_loadcnt 0x0` |
| Explicit store completion | `s_waitcnt_vscnt null, 0x0` | `s_wait_storecnt 0x0` |

Counter names in `.rip` are enums, not assembly mnemonics:

| Operation | gfx1151 | gfx1201 |
|---|---|---|
| VMEM load | `Counter::Vm` | `Counter::Load` |
| VMEM store | `Counter::Vs` | `Counter::Store` |
| LDS | `Counter::Lgkm` | `Counter::Ds` |
| SMEM | `Counter::Lgkm` | `Counter::Km` |
| Export | `Counter::Exp` | `Counter::Exp` |

Source: `kernels/common.rs::smem`, `srd_tail`; `ledger.rs` counter accounting and wait lowering. `first` is an element index, not a byte offset.

## Variations

- Pack pointer pairs into separate owners and issue width-2 SMEM loads; keep every destination inside one owner.
- Use the reserved pointer in another ABI-compatible operation; this kernel intentionally leaves that memory untouched.
- Set `first = 0` for the full array. For a subrange, allocate the full `n` elements and compare only the suffix while retaining the prefix unchanged.
- Use `bload` for buffers covered by the CPU emulator; do not assume its coverage of global-memory instructions is identical.
