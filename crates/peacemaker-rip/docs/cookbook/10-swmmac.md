# 10. One sparse WMMA tile: `D = A_sparse * B` with SWMMAC (2:4), f16 or fp8 in, f32 out

**Problem.** One wave32 computes a single 16x16x32 product `D = A * B` where `A` (16 rows x 32 K) is **2:4 structured sparse**: in every group of four consecutive K values at most two are nonzero. `v_swmmac_*` takes `A` *compressed*, two kept values per K-group, plus a 2-bit selector per kept value that names its position inside the group. `B` is dense and read as `B^T` (**column-major B**: `bt[c][k]` at row `c`, 32 K values per row). `D` is f32 row-major (`D[r][c]` at byte `4*(16*r + c)`).
**Launch.** One workgroup of 32 threads, kernargs `0: a`, `8: idx`, `16: bt`, `24: d` (32 bytes, pointers only). SWMMAC exists on gfx12 only: this recipe is gfx1201 (RDNA4); on gfx1151 `Wmma::swmmac` refuses with `SWMMAC requires gfx12`.

The kernel has two variants selected by `spec.variant`. `sparse_f16` issues `v_swmmac_f32_16x16x32_f16` (`A` and `B` are f16). `sparse_fp8` issues `v_swmmac_f32_16x16x32_fp8_fp8` (`A` and `B` are raw E4M3 bytes). Same index word, same fragment geometry, same store; only the element width and the instruction differ.

Buffers (all indices are *lane order*, see the tables below):

- `a`: compressed `A`, row-major `[16 rows][16 slots]` (f16: 2 bytes per slot, fp8: 1 byte per slot). Slot `j` of row `r` is at element `16*r + j`.
- `idx`: `u32[32]`, one raw hardware index word per lane. Lane `L = 16*h + r` carries, in bits `2*i .. 2*i+1`, the selector of slot `8*h + i` of row `r` (`index_key` 0, so only the low 16 bits are used).
- `bt`: `B^T` row-major `[16 columns][32 K]` (f16 2 bytes, fp8 1 byte per element).
- `d`: f32 `[16][16]` row-major.

```rip variant=sparse_f16 arch=gfx1201
fn emit(spec: Spec) -> Result<Emitted, String> {
    let arch = spec.arch;
    // `sparse_fp8` takes the fp8 path; any other variant takes the f16 path.
    let fp8 = spec.variant == "sparse_fp8";
    let kspec = KernelSpec {
        kernel_id: "swmmac_sparse24".into(), variant: spec.variant.clone(), arch,
        symbol: format!("swmmac_sparse24_{}", arch.name()),
        // a at 0, idx at 8, bt at 16, d at 24: four 64-bit pointers, no scalars.
        kernargs: KernargLayout::new(32).pointer("a", 0).pointer("idx", 8).pointer("bt", 16).pointer("d", 24),
        user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 0, wave32: true, cu_mode: false,
    };
    // Per-variant element widths: bytes per element, and the descriptor byte bounds.
    let ebytes: u32 = if fp8 { 1 } else { 2 };
    let a_bytes = 256 * ebytes;
    let bt_bytes = 512 * ebytes;

    // SGPRs: s0:1 kernarg pointer, s2 workgroup id x (gfx11 only; unused), one owner per
    // buffer descriptor (s4 a, s8 idx, s12 bt, s16 d) and one per descriptor bound.
    let mut p = RegPlan::new(256, 104)?;
    p.s::<2>("kernarg_ptr", 0, Live::Whole)?;
    if !arch.gfx12() { p.s::<1>("wg_x", 2, Live::Whole)?; }
    p.s::<4>("srd_a", 4, Live::Whole)?;
    p.s::<4>("srd_idx", 8, Live::Whole)?;
    p.s::<4>("srd_bt", 12, Live::Whole)?;
    p.s::<4>("srd_d", 16, Live::Whole)?;
    p.s::<1>("bound_a", 20, Live::Whole)?;
    p.s::<1>("bound_idx", 21, Live::Whole)?;
    p.s::<1>("bound_bt", 22, Live::Whole)?;
    p.s::<1>("bound_d", 23, Live::Whole)?;
    // VGPRs: v0..7 lane temporaries (v0 holds the thread id at entry), the compressed A
    // fragment v8..11 (fp8 uses v8..9), the dense B fragment v12..19 (fp8 uses v12..15),
    // the sparse index v20, and the D accumulator v24..31.
    p.v::<8>("temps", 0, Live::Whole)?;
    p.v::<4>("a_frag", 8, Live::Whole)?;
    p.v::<8>("b_frag", 12, Live::Whole)?;
    p.v::<1>("index", 20, Live::Whole)?;
    p.v::<8>("d_frag", 24, Live::Whole)?;

    let mut b = Builder::new(kspec, p);
    let mut wg = Workgroup::new(&mut b, arch)?;
    let end = wg.exit(".Lswmmac_end")?;

    // Pointer pairs, one `s_load_b64` each, straight from the kernarg segment.
    smem(wg.isa(), 4, 2, 0, 0)?;
    smem(wg.isa(), 8, 2, 0, 8)?;
    smem(wg.isa(), 12, 2, 0, 16)?;
    smem(wg.isa(), 16, 2, 0, 24)?;
    // Descriptor words 2 and 3: num_records is the exact byte extent of each buffer.
    sop(wg.isa(), format!("s_mov_b32 s20, {}", lit(a_bytes)), &[20], &[])?;
    sop(wg.isa(), format!("s_mov_b32 s21, {}", lit(128)), &[21], &[])?;
    sop(wg.isa(), format!("s_mov_b32 s22, {}", lit(bt_bytes)), &[22], &[])?;
    sop(wg.isa(), format!("s_mov_b32 s23, {}", lit(1024)), &[23], &[])?;
    srd_tail(wg.isa(), 4, Some(20))?;
    srd_tail(wg.isa(), 8, Some(21))?;
    srd_tail(wg.isa(), 12, Some(22))?;
    srd_tail(wg.isa(), 16, Some(23))?;

    // Lane geometry: v1 = r = tid & 15 (A row, B column, D column), v2 = h = tid >> 4.
    op(wg.isa(), "v_and_b32_e32 v1, 15, v0", &[v(1)], &[v(0)])?;
    op(wg.isa(), "v_lshrrev_b32_e32 v2, 4, v0", &[v(2)], &[v(0)])?;
    // v3: byte offset of this lane's 8 compressed slots, a[r][8h..8h+7]:
    // (16*r + 8*h) elements = ebytes * (16*r + 8*h) bytes.
    let a_row_shift: u32 = if fp8 { 4 } else { 5 };
    op(wg.isa(), format!("v_lshlrev_b32_e32 v3, {}, v1", a_row_shift), &[v(3)], &[v(1)])?;
    op(wg.isa(), format!("v_lshlrev_b32_e32 v7, {}, v2", a_row_shift - 1), &[v(7)], &[v(2)])?;
    op(wg.isa(), "v_add_nc_u32_e32 v3, v3, v7", &[v(3)], &[v(3), v(7)])?;
    // v4: byte offset of this lane's 16 K values of B^T, bt[r][16h..16h+15]:
    // (32*r + 16*h) elements.
    let bt_row_shift: u32 = if fp8 { 5 } else { 6 };
    op(wg.isa(), format!("v_lshlrev_b32_e32 v4, {}, v1", bt_row_shift), &[v(4)], &[v(1)])?;
    op(wg.isa(), format!("v_lshlrev_b32_e32 v7, {}, v2", bt_row_shift - 1), &[v(7)], &[v(2)])?;
    op(wg.isa(), "v_add_nc_u32_e32 v4, v4, v7", &[v(4)], &[v(4), v(7)])?;
    // v5: byte offset of this lane's index word, 4*tid.
    op(wg.isa(), "v_lshlrev_b32_e32 v5, 2, v0", &[v(5)], &[v(0)])?;
    // v6: byte base of the D store: 4*column + 512*h (rows 8*h + j, 64 bytes apart).
    op(wg.isa(), "v_lshlrev_b32_e32 v6, 2, v1", &[v(6)], &[v(1)])?;
    op(wg.isa(), "v_lshlrev_b32_e32 v7, 9, v2", &[v(7)], &[v(2)])?;
    op(wg.isa(), "v_add_nc_u32_e32 v6, v6, v7", &[v(6)], &[v(6), v(7)])?;

    // Loads. A: the 8 compressed slots are contiguous (f16 16 bytes, fp8 8 bytes). B: the
    // 16 K values are contiguous (f16 32 bytes in two b128 loads, fp8 16 bytes in one).
    // The index word is one dword. No explicit wait: the builder inserts it before the
    // SWMMAC reads them.
    if fp8 {
        bload(wg.isa(), 2, 8, 3, 4, 0)?;
        bload(wg.isa(), 4, 12, 4, 12, 0)?;
    } else {
        bload(wg.isa(), 4, 8, 3, 4, 0)?;
        bload(wg.isa(), 4, 12, 4, 12, 0)?;
        bload(wg.isa(), 4, 16, 4, 12, 16)?;
    }
    bload(wg.isa(), 1, 20, 5, 8, 0)?;

    // SWMMAC has no C operand: D is read in place as the accumulator, so zero it first.
    for j in 0..8u8 {
        op(wg.isa(), format!("v_mov_b32_e32 v{}, 0", 24 + j), &[v(24 + j)], &[])?;
    }
    let mods = SwmmacMods { index_key: 0, neg_lo: 0, neg_hi: 0, clamp: false };
    if fp8 {
        wg.isa().push(Wmma::swmmac(arch, Swmmac::F32Fp8Fp8, vr(24, 8), vr(8, 2), vr(12, 4), v(20), mods)?)?;
    } else {
        wg.isa().push(Wmma::swmmac(arch, Swmmac::F32F16, vr(24, 8), vr(8, 4), vr(12, 8), v(20), mods)?)?;
    }

    // D: register j of lane 16*h + c is row 8*h + j, column c, so successive registers
    // are 64 bytes apart.
    let soff = if arch.gfx12() { "null" } else { "0" };
    for j in 0..8u8 {
        mem(wg.isa(), format!("buffer_store_b32 v{}, v6, s[16:19], {} offen offset:{}", 24 + j, soff, 64 * (j as u32)),
            &[], &[v(24 + j), v(6), sr(16, 4)], MemoryClass::VmemStore)?;
    }
    wg.end(end)?;
    b.finish()
}
```

## Layout tables

Source: `peacemaker-ir/src/emu/wmma.rs`, `swmmac` (the doc comment states the layout; every kind and the f16 / fp8 shapes use it). Lane `L = 16*h + r`, `r = L & 15`, `h = L >> 4`.

`A` holds `nA` compressed slots per lane: 8 for every x32 kind (f16: 4 VGPRs of two halves, fp8: 2 VGPRs of four bytes), 16 for the x64 iu4 kind. Slot `i` of the lane is VGPR `i / (32/bits)`, field `i % (32/bits)`, and compressed slot `j = 8*h + i` of row `r`. The index word of the same lane holds the selectors:

```text
sel(i)   = (idx >> (16*index_key + 2*i)) & 3         index_key = 0 here
slot j   meets dense K = 4*(j >> 1) + sel
         slots 2g and 2g+1 form K-group g (K 4g .. 4g+3) and must pick different positions
         (true 2:4); the other two positions of the group are the zeros the format elides.

  lane half   slots j       K-groups   dense K range
  h = 0       0..7          0..3       0..15
  h = 1       8..15         4..7       16..31
  per-lane    i  0  1 | 2  3 | 4  5 | 6  7
              g  4h+0   | 4h+1  | 4h+2  | 4h+3          (K = 4g + sel)
  idx bits    1:0 3:2 | 5:4 7:6 | 9:8 11:10 | 13:12 15:14
```

`B` is dense. Lane `L` is column `c = L & 15` and holds 16 contiguous K values from `K = 16*h`:

```text
element e (0..15) of lane L  ->  K = 16*h + e,  B[K][c] = bt[c][K]
  f16:  VGPR e/2, half e%2     -> bt row c, bytes 64*c + 32*h .. +31   (two buffer_load_b128)
  fp8:  VGPR e/4, byte e%4     -> bt row c, bytes 32*c + 16*h .. +15   (one buffer_load_b128)
A    :  a[r][8h .. 8h+7]       f16: bytes 32*r + 16*h .. +15   (one buffer_load_b128)
                               fp8: bytes 16*r +  8*h .. +7    (one buffer_load_b64)
idx  :  bytes 4*L .. +3                                          (one buffer_load_b32)
```

A lane's compressed slots only ever meet its own half's K range (`16*h .. 16*h+15`), so the 8 slots and the 16 K values of one lane line up: the dense reference is `D[r][c] = sum over slots j of a[r][j] * B[4*(j>>1) + sel(j)][c]`.

```text
D (f32)   acc layout, row = 8*h + j:  lane 16*h + c, VGPR j  ->  d[row][c]
  Store: byte base 4*c + 512*h, immediate 64*j, j = 0..7.
```

## The constructor

```text
Wmma::swmmac(arch, kind: Swmmac, dst, a, b, index, mods: SwmmacMods) -> Result<Instruction, String>
    dst, a, b  register ranges (vr(..)) of exactly the kind's width, see the table
    index      one VGPR, v(n)
    mods       SwmmacMods { index_key, neg_lo, neg_hi, clamp }  (all four fields required)
    emits      <mnemonic> dst, a, b, index [index_key:1] [neg_lo:[x,y,0]] [neg_hi:[x,y,0]] [clamp]
    effects    defs [dst], uses [a, b, index, dst]: D is read in place as the accumulator
    push it with wg.isa().push(..)?
```

## Why

- **The accumulator is `dst`.** SWMMAC has no separate C: the instruction reads `dst` as the accumulator and writes it. `Wmma::swmmac` therefore declares `dst` as a use as well as a def (hand-written `op` text must do the same; omitting it is refused with R0603 "also reads"). The kernel zeroes `v24..31` first; to accumulate across K slices, issue further `Wmma::swmmac` calls on the same `dst` with new `a`/`idx`/`b` fragments.
- **Compression is the host's job.** The kernel loads the compressed values and the selectors as given; it never checks that a pair of selectors differ. A pair with equal selectors is not a 2:4 matrix and is not a supported input. The test for this recipe rebuilds dense `A` from `a` and `idx`, asserts the 2:4 property and recomputes `D` independently.
- **Descriptor bounds are exact.** `a` is 512 bytes (f16) or 256 (fp8), `idx` 128, `bt` 1024 (f16) or 512 (fp8), `d` 1024. A stray offset reads zero or drops the store.
- **Waits are the builder's job.** No `wait` call appears: the four loads are declared as defs of `v8..`, `v12..` and `v20`, and the builder puts the loadcnt wait in front of the SWMMAC.
- **soffset is `null` on gfx12.** The helpers spell it for you; the explicit store text above uses `soff`.

## Kinds and modifiers

`Swmmac::{F32F16, F32Bf16, F16F16, Bf16Bf16, I32Iu8, I32Iu4, I32Iu4X64, F32Fp8Fp8, F32Fp8Bf8, F32Bf8Fp8, F32Bf8Bf8}`; the widths are in VGPRs, the index is always one VGPR. See `REFERENCE.md` section 10 for the table and the legal modifier sets.

## Hardware check

`hw_diff` runs the same ELF on the emulator and then on a gfx1201 card, and compares both against the case's CPU-reference `expect`. Extract the `rip` block above to `swmmac.rip`, then:

```text
cargo build --release -p peacemaker-rip --features hw-diff --example hw_diff
fabric-run.sh --class correct --card B --mem-gb 1 --max-secs 300 -- \
  target/release/examples/hw_diff swmmac.rip --case crates/peacemaker-rip/tests/cases/10_swmmac_fp8_sparse24.json \
    --variant sparse_fp8 --arch gfx1201 --device 0 --json
```

Both variants passed on 2026-10-22 (card B, `agree: true`, 0 mismatches over 256 words). The receipt is in `peacemaker-ir/isa/emu/float_edges.md`.

## Variations

- **bf16 or f16 accumulators.** `Swmmac::F32Bf16` takes bf16 bits in the same fragments; `F16F16` / `Bf16Bf16` accumulate into a 4-VGPR `dst` whose lane layout packs two rows per register (`row r`, VGPR `(r&7)>>1`, bits `16*(r&1)`).
- **The upper index half.** `index_key: 1` selects selectors from bits 16..31 of the same index word, so one dword per lane can steer two SWMMACs (two K slices).
- **Larger K.** Repeat the three loads and the `Wmma::swmmac` for each 32-wide K slice, accumulating into the same `dst`.
