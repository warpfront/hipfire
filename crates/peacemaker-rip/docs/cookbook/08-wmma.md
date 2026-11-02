# 08. One WMMA tile: `D = A * B`, bf16 in, f32 out

**Problem.** One wave32 computes a single 16x16x16 product `D = A * B` with `v_wmma_f32_16x16x16_bf16`. `A` is bf16 **column-major** (`A[r][k]` at byte `2*(r + 16*k)`), `B` is bf16 **row-major** (`B[k][n]` at byte `2*(16*k + n)`), `D` is f32 row-major (`D[r][n]` at byte `4*(16*r + n)`).
**Launch.** One workgroup of 32 threads, kernargs `0: A`, `8: B`, `16: D` (24 bytes, pointers only). The same source must be correct on gfx1151 (RDNA3.5) and gfx1201 (RDNA4).

The kernel has two variants. `bf16_tile` zeroes the accumulator and issues the WMMA through a `Chain`; `bf16_raw` issues the same instruction through `op` with the accumulator input folded to the inline constant `0`. Both go through the same fragment gather, the same buffer descriptors and the same store.

```rip variant=bf16_tile
fn emit(spec: Spec) -> Result<Emitted, String> {
    let arch = spec.arch;
    let g12 = arch.gfx12();
    // `bf16_raw` takes the raw `op` path; any other variant takes the `Chain` path.
    let raw = spec.variant == "bf16_raw";
    let kspec = KernelSpec {
        kernel_id: "wmma_bf16_tile".into(), variant: spec.variant.clone(), arch,
        symbol: format!("wmma_bf16_tile_{}", arch.name()),
        // A at 0, B at 8, D at 16: three 64-bit pointers, no scalars.
        kernargs: KernargLayout::new(24).pointer("a", 0).pointer("b", 8).pointer("d", 16),
        user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 0, wave32: true, cu_mode: false,
    };

    // SGPRs: s0:1 kernarg pointer, s2 workgroup id x (gfx11 only; unused, there is one
    // workgroup), then one owner per buffer descriptor and one per descriptor bound.
    let mut p = RegPlan::new(256, 104)?;
    p.s::<2>("kernarg_ptr", 0, Live::Whole)?;
    if !g12 { p.s::<1>("wg_x", 2, Live::Whole)?; }
    p.s::<4>("srd_a", 4, Live::Whole)?;
    p.s::<4>("srd_b", 8, Live::Whole)?;
    p.s::<4>("srd_d", 12, Live::Whole)?;
    p.s::<1>("bound_ab", 16, Live::Whole)?;
    p.s::<1>("bound_d", 17, Live::Whole)?;
    // VGPRs: v0..7 lane temporaries (v0 holds the thread id at entry), then the A
    // fragment, the B fragment and the D accumulator, eight registers each. gfx12 uses
    // only the first four registers of the A and B owners.
    p.v::<8>("temps", 0, Live::Whole)?;
    p.v::<8>("a_frag", 8, Live::Whole)?;
    p.v::<8>("b_frag", 16, Live::Whole)?;
    p.v::<8>("d_frag", 24, Live::Whole)?;

    let mut b = Builder::new(kspec, p);
    let mut wg = Workgroup::new(&mut b, arch)?;
    let end = wg.exit(".Lwmma_end")?;

    // Pointer pairs, one `s_load_b64` each, straight from the kernarg segment.
    smem(wg.isa(), 4, 2, 0, 0)?;
    smem(wg.isa(), 8, 2, 0, 8)?;
    smem(wg.isa(), 12, 2, 0, 16)?;
    // Descriptor words 2 and 3: num_records is a byte bound (A 512, B 512, D 1024).
    sop(wg.isa(), format!("s_mov_b32 s16, {}", lit(512)), &[16], &[])?;
    sop(wg.isa(), format!("s_mov_b32 s17, {}", lit(1024)), &[17], &[])?;
    srd_tail(wg.isa(), 4, Some(16))?;
    srd_tail(wg.isa(), 8, Some(16))?;
    srd_tail(wg.isa(), 12, Some(17))?;

    // Lane geometry: v1 = idx = tid & 15, v2 = h = tid >> 4 (lane half).
    op(wg.isa(), "v_and_b32_e32 v1, 15, v0", &[v(1)], &[v(0)])?;
    op(wg.isa(), "v_lshrrev_b32_e32 v2, 4, v0", &[v(2)], &[v(0)])?;
    // v3: byte base of the 16-bit gathers. A[idx][k] and B[k][idx] both sit at
    // 2*idx + 32*k, so one base serves both fragments. gfx12 lane half h owns k bits
    // 4*h.., which adds 32 * 4 * h = 128 * h.
    op(wg.isa(), "v_lshlrev_b32_e32 v3, 1, v1", &[v(3)], &[v(1)])?;
    if g12 {
        op(wg.isa(), "v_lshlrev_b32_e32 v5, 7, v2", &[v(5)], &[v(2)])?;
        op(wg.isa(), "v_add_nc_u32_e32 v3, v3, v5", &[v(3)], &[v(3), v(5)])?;
    }
    // v4: byte base of the D store, 4*idx plus the lane half's row offset
    // (gfx11 row = 2*j + h: 64 * h; gfx12 row = j + 8*h: 512 * h).
    op(wg.isa(), "v_lshlrev_b32_e32 v4, 2, v1", &[v(4)], &[v(1)])?;
    op(wg.isa(), format!("v_lshlrev_b32_e32 v5, {}, v2", if g12 { 9 } else { 6 }), &[v(5)], &[v(2)])?;
    op(wg.isa(), "v_add_nc_u32_e32 v4, v4, v5", &[v(4)], &[v(4), v(5)])?;

    // Fragments. Register j packs two consecutive k values: the low half is k = lowK,
    // the high half k = lowK + 1. gfx11: lowK = 2*j. gfx12: lowK = 8*(j>>1) + 4*h +
    // 2*(j&1); the 4*h part is already in v3. Each half is one `buffer_load_u16`: the
    // low half zero-extends straight into the fragment register, the high half goes to
    // a temporary (v6 for A, v7 for B), is shifted up and or-ed in. There is no explicit
    // wait: the builder inserts the vmcnt/loadcnt wait before the first shift.
    let w: u8 = if g12 { 4 } else { 8 };
    let soff = if g12 { "null" } else { "0" };
    for j in 0..w {
        let jj = j as u32;
        let kimm: u32 = if g12 { (jj / 2) * 8 + (jj % 2) * 2 } else { 2 * jj };
        mem(wg.isa(), format!("buffer_load_u16 v{}, v3, s[4:7], {} offen offset:{}", 8 + j, soff, 32 * kimm),
            &[v(8 + j)], &[v(3), sr(4, 4)], MemoryClass::VmemLoad)?;
        mem(wg.isa(), format!("buffer_load_u16 v6, v3, s[4:7], {} offen offset:{}", soff, 32 * (kimm + 1)),
            &[v(6)], &[v(3), sr(4, 4)], MemoryClass::VmemLoad)?;
        mem(wg.isa(), format!("buffer_load_u16 v{}, v3, s[8:11], {} offen offset:{}", 16 + j, soff, 32 * kimm),
            &[v(16 + j)], &[v(3), sr(8, 4)], MemoryClass::VmemLoad)?;
        mem(wg.isa(), format!("buffer_load_u16 v7, v3, s[8:11], {} offen offset:{}", soff, 32 * (kimm + 1)),
            &[v(7)], &[v(3), sr(8, 4)], MemoryClass::VmemLoad)?;
        op(wg.isa(), "v_lshlrev_b32_e32 v6, 16, v6", &[v(6)], &[v(6)])?;
        op(wg.isa(), format!("v_or_b32_e32 v{}, v{}, v6", 8 + j, 8 + j), &[v(8 + j)], &[v(8 + j), v(6)])?;
        op(wg.isa(), "v_lshlrev_b32_e32 v7, 16, v7", &[v(7)], &[v(7)])?;
        op(wg.isa(), format!("v_or_b32_e32 v{}, v{}, v7", 16 + j, 16 + j), &[v(16 + j)], &[v(16 + j), v(7)])?;
    }

    if raw {
        // Raw path: the accumulator input is the inline constant 0, so D needs no
        // zeroing. Text, defs and uses are spelled by hand; A and B are `w` registers.
        let a_text = format!("v[8:{}]", 8 + w - 1);
        let b_text = format!("v[16:{}]", 16 + w - 1);
        op(wg.isa(), format!("v_wmma_f32_16x16x16_bf16 v[24:31], {}, {}, 0", a_text, b_text),
            &[vr(24, 8)], &[vr(8, w), vr(16, w)])?;
    } else {
        // Chain path: `step` accumulates into its destination (C = D), so zero D first.
        for j in 0..8u8 {
            op(wg.isa(), format!("v_mov_b32_e32 v{}, 0", 24 + j), &[v(24 + j)], &[])?;
        }
        let chain = Chain::new(24, 8, "bf16");
        chain.step(wg.isa(), 8, w, 16, w)?;
    }

    // D: register j of lane 16*h + idx is column idx of row 2*j + h (gfx11) or
    // j + 8*h (gfx12), so successive registers are 128 (gfx11) or 64 (gfx12) bytes apart.
    let dstride: u32 = if g12 { 64 } else { 128 };
    for j in 0..8u8 {
        mem(wg.isa(), format!("buffer_store_b32 v{}, v4, s[12:15], {} offen offset:{}", 24 + j, soff, dstride * (j as u32)),
            &[], &[v(24 + j), v(4), sr(12, 4)], MemoryClass::VmemStore)?;
    }
    wg.end(end)?;
    b.finish()
}
```

## Fragment tables

Source: `peacemaker-ir/src/emu/wmma.rs`, `f16_loc` (the bf16 shape uses the same locations as f16) and `acc_loc`. Lane `L = 16*h + idx`, `idx = L & 15`, `h = L >> 4`. For `A`, `idx` is the row `r` and the fragment holds `A[idx][k]`; for `B`, `idx` is the column `n` and the fragment holds `B[k][idx]`.

```text
gfx1151   f16_loc(idx, k) = (lane idx, reg k>>1, half k&1)   width 8 registers
          lanes idx and idx+16 hold identical A/B fragments (replicas); the emulator
          rejects a divergent half.
  reg j   low half   high half
  j0      k0         k1
  j1      k2         k3
  j2      k4         k5
  j3      k6         k7
  j4      k8         k9
  j5      k10        k11
  j6      k12        k13
  j7      k14        k15
  lowK = 2*j.  Gather: byte base 2*idx, immediate 64*j (low), 64*j + 32 (high).

gfx1201   f16_loc(idx, k) = (lane ((k>>2)&1)*16 + idx, reg (k>>3)*2 + ((k>>1)&1), half k&1)
          width 4 registers; the lane half h supplies k bit 2.
  reg j   h=0 low  h=0 high   h=1 low  h=1 high
  j0      k0       k1         k4       k5
  j1      k2       k3         k6       k7
  j2      k8       k9         k12      k13
  j3      k10      k11        k14      k15
  lowK = 8*(j>>1) + 4*h + 2*(j&1).
  Gather: byte base 2*idx + 128*h, immediate 32*(8*(j>>1) + 2*(j&1)) (low), +32 (high):
  j0 0/32, j1 64/96, j2 256/288, j3 320/352.
```

Both operands use the same byte base because `A` is column-major and `B` is row-major: `A[idx][k]` and `B[k][idx]` are both at `2*idx + 32*k`. Each 16-bit element is a strided gather, so each packed register costs two `buffer_load_u16`, a shift and an or.

```text
gfx1151   acc_loc(row, col) = (lane (row&1)*16 + col, reg row>>1)      row = 2*j + h
  reg j   h=0 row   h=1 row
  j0      0         1
  j1      2         3
  j2      4         5
  j3      6         7
  j4      8         9
  j5      10        11
  j6      12        13
  j7      14        15
  Store: byte base 4*idx + 64*h, immediate 128*j.

gfx1201   acc_loc(row, col) = (lane (row>>3)*16 + col, reg row&7)      row = j + 8*h
  reg j   h=0 row   h=1 row
  j0      0         8
  j1      1         9
  j2      2         10
  j3      3         11
  j4      4         12
  j5      5         13
  j6      6         14
  j7      7         15
  Store: byte base 4*idx + 512*h, immediate 64*j.
```

## The two builder paths

```text
Chain::new(dst: u8, width: u8, kind: &str) -> Chain
    dst   first VGPR of the accumulator; width must be 8
    kind  "f16" or "bf16"
chain.step(builder, a: u8, a_width: u8, b: u8, b_width: u8) -> Result<(), String>
    emits  v_wmma_f32_16x16x16_<kind> v[dst:dst+7], v[a:..], v[b:..], v[dst:dst+7]
    widths are 4 or 8 VGPRs (gfx12 4, gfx11 8); the destination is also the C input,
    declared as def and use.
    example: Chain::new(24, 8, "bf16").step(wg.isa(), 8, 4, 16, 4)?
        gfx1201: v_wmma_f32_16x16x16_bf16 v[24:31], v[8:11], v[16:19], v[24:31]

op(builder, text: String, defs: &[RegRef], uses: &[RegRef]) -> Result<(), String>
    emits the text as one instruction with the declared effects; the instruction is
    validated for the architecture; missing register definitions or uses are R0603 errors.
    example: op(wg.isa(), "v_wmma_f32_16x16x16_bf16 v[24:31], v[8:15], v[16:23], 0",
                &[vr(24, 8)], &[vr(8, 8), vr(16, 8)])?
```

## Why

- **`Chain` accumulates, `op` can start from zero.** `step` always reads the accumulator, so the `Chain` path writes `0` to `v24..v31` first. The raw path passes `0` as the C operand and needs no zeroing. A kernel that only ever does one product should prefer the raw form; the `Chain` form is the one that extends to a K loop.
- **Fragment width depends on the architecture.** gfx11 expects A and B replicated across the two lane halves, eight registers each. gfx12 splits K between the halves and needs four. The width passed to `step` must match (`Chain` accepts only 4 or 8).
- **Waits are the builder's job.** No `wait` call appears. The builder tracks every declared def and use and puts the right counter (`vmcnt`/`lgkmcnt` on gfx11, `loadcnt`/`kmcnt` on gfx12) before the first reader of a load. The declared effects have to be accurate for that to work: each `buffer_load_u16` lists its destination register, `v3` and the whole descriptor, and each store lists its data register. The kernel ends right after the stores; `s_endpgm` retires them.
- **One owner per descriptor.** `s[4:7]`, `s[8:11]` and `s[12:15]` each have one plan owner. The `s_load_b64` pointer loads write only the first two dwords of an owner, and `srd_tail` writes the last two (`num_records` from `s16` or `s17`, then the raw-buffer word 3). Declaring the plan owners at the width of the descriptor, rather than per dword, is what avoids "no unique live owner" errors on multi-dword SGPR ranges.
- **Byte bounds are real.** The descriptors cover exactly the data: 512 bytes for A and B (256 bf16), 1024 for D (256 f32). A stray offset reads zero or drops the store instead of touching memory.
- **soffset is `0` on gfx11 and `null` on gfx12.** The assembler refuses `0` on gfx12 ("SOFFSET does not fit its encoding field"); hand-written buffer text has to spell it per architecture, and `smem` and the buffer helpers do it for you.
- **gfx11 residues.** The emulator models the gfx11 silicon accumulation, which leaves small residues on exact inputs (`peacemaker-ir/src/emu/mma.rs`: C's sign frame, one's-complement opposing terms, a rounding at each internal DOT step). A gfx1151 run is compared within `1e-3 * max(1, |expected|)`; gfx1201 aligns product pairs with arithmetic floor and is compared bitwise.

## Variations

- **Many tiles.** Make the grid `T` workgroups. Use `s2` for the workgroup id on gfx11 and `ttmp9` on gfx12 (`s_mov_b32 s_tile, ttmp9`); reading `s2` on gfx12 compiles and gives wrong answers. Add `tile * 512` to A and B and `tile * 1024` to D with `add64` before `srd_tail`.
- **f16 inputs.** Use `Chain::new(24, 8, "f16")` or `v_wmma_f32_16x16x16_f16`; the fragment tables are the same.
- **C input.** Load a 16x16 f32 C tile into `v24..v31` with the D layout above and skip the zeroing; `step` then computes `D = C + A*B`.
- **Larger K.** Repeat the gather and the `step` for each 16-wide K slice; the chain keeps accumulating into the same registers.
- **Row-major A or column-major B.** Only the gather changes: when the fragment's two k values are adjacent in memory (A row-major), one `buffer_load_b32` per register replaces the two `buffer_load_u16`, a shift and an or. The byte base becomes `32*idx` (plus `8*h` on gfx12) and the immediate `2*lowK`.
