# `.rip` language reference

This reference targets RDNA gfx1151 and gfx1201, wave32. gfx1100 uses the gfx11 instruction spellings below. `.rip` is an evaluated kernel-authoring language, not rustc and not device Rust.

## 1. Mental model

The host parses the file with `syn`, evaluates `emit`, and records device instructions in `peacemaker_isa::Builder`. A script `for` loop runs while emitting: it unrolls instructions. A device loop must use the Workgroup driver's loop operations. A script `if arch.gfx12()` chooses instructions while emitting; a device conditional uses a driver condition.

The runtime Driver checks LDS ownership, publication, draining and workgroup-uniform barriers and control flow. The builder checks register ownership, instruction policy and memory hazards, and inserts dependency waits. Instruction validation checks the mnemonic and operands for the selected architecture and rejects incorrect declared register effects (R0603). Native assembly/lifting checks for registers read but neither entry-initialized nor written anywhere in the kernel (R0604), including an incorrect s2 workgroup-ID read on gfx1201. This check is path-free: it does not prove that a write precedes each read, covers each branch, or initializes both halves. Those stronger checks belong to M7 certification.

`check` executes the selected entry/variant for each selected architecture and reports these errors and warnings plus register/LDS/instruction counts. It does not prove the numerical algorithm, bounds for arbitrary runtime inputs, launch correctness or emulator coverage. Unexecuted script paths are not checked. Assembly/link certification is a separate `build --certify` step.

## 2. Files, entries and variants

The preferred entry signature is `fn emit(spec: Spec) -> Result<Emitted, String>`. The injected `Spec` has `arch: Arch`, `variant: String`, and `kind`. Declare a unit `enum Kind { Default, Other }` to make `spec.kind` the matching enum variant (case-insensitive matching against `spec.variant`); without a matching variant, `kind` is the variant string. Match `spec.kind` to choose emission routines. The alternate entry is `fn emit(arch: Arch, variant: String) -> Result<Emitted, String>`; it receives the same architecture and variant string separately. Finish with `builder.finish()`, not `Ok(())`.

`Arch::{Gfx1100,Gfx1151,Gfx1201}` are available. `arch.gfx12() -> bool`, `arch.name() -> String`, and `arch.buffer_offset_max() -> u32` inspect it; for example `let name = spec.arch.name();`.

`check` defaults to gfx1151 and gfx1201; `--arch all` includes gfx1100. Pass `--variant default` explicitly when there is no `Kind`. When omitted, the CLI uses the script's declared `Kind` variants. Variant and architecture selection happen at emission time, not at kernel launch.

## 3. The evaluated Rust subset

Supported executed syntax includes functions, const/static values, structs, enums (unit and tuple variants and discriminants), impl methods and trait defaults, lexical closures, tuples, arrays/Vec, Option/Result, `?`, references to mutable places, field/index assignment, `let else`, `if let`, `while let`, match guards/ranges/or-patterns/bindings, and labelled `for`/`while`/`loop` with break/continue. Generics, lifetimes and most type annotations are accepted but are not a rustc type system; integer annotations do matter. `use` and unused items are not resolved as Rust imports. Native APIs are a fixed vocabulary, not arbitrary crate access.

Unsupported executed syntax is a located error, never a silently ignored operation. Do not assume async, unsafe blocks, arbitrary standard-library APIs or user macro expansion work. Iterator adaptors are eager/materialized, not lazy streams; range materialization is capped at 2^22 elements. Keep emission-time collections small.

Integer types are `u8/u16/u32/u64/usize/i8/i16/i32/i64/isize`. Unsuffixed literals adopt the other operand's or declared parameter/binding's integer type. Mixed typed operands are errors; arithmetic overflow, zero division and invalid shifts are errors. Integer `as` casts truncate to the destination width; widening can use `u32::from(x)`.

```rip eval
fn main() -> u32 {
    let n: u8 = 200;
    assert_eq!(n + 50, 250u8);
    let low = 0x1ffu32 as u8;
    assert_eq!(low, 255u8);
    u32::from(low) + 1
}
```

Macros include `format!`, `vec!`, `matches!`, `assert!`, `assert_eq!`, `assert_ne!`, `panic!`, `unreachable!`, `todo!`, `unimplemented!`, and print/println/eprint/eprintln. Panic-like macros abort evaluation; they do not emit GPU traps. `format!` supports positional, named and captured arguments, escaped braces, Display/Debug (`?`, `#?`), radix `x/X/b/o`, exponent `e`, fill/alignment/sign/alternate/zero padding, fixed width and precision. Dynamic `$` width/precision arguments are unsupported.

```rip eval
fn main() -> String {
    let n = 255u32;
    let values: Vec<u32> = [1u32, 2, 3].into_iter().map(|x| x + 1).collect();
    assert_eq!(values[2], 4u32);
    format!("{0:#x} {n:08X} {values:?}", n)
}
```

## 4. Item keywords

These are token-level item rewrites, not extra runtime objects:

| Syntax | Becomes | Example |
|---|---|---|
| `region R;` or `region R, S;` | empty enum tag(s) | `region Scratch;` |
| `tile T { fields }` | `struct T { fields }` | `tile Pair { x: u8, y: u8 }` |
| `chain f(args) -> Ret { body }` | `fn f(args) -> Ret { body }` | `chain inc(x: u32) -> u32 { x + 1 }` |

`region` and `tile` are top-level items; `chain` also works inside impl/trait/module items. Attributes and visibility remain attached. A variable named `tile`, a `.chain()` method or a string containing these words is unaffected. A tile does not allocate registers; a region tag does not allocate LDS. Use RegPlan and `wg.lds` respectively.

```rip eval
region Scratch;
tile Pair { x: u32, y: u32 }
chain total(p: Pair) -> u32 { p.x + p.y }
fn main() -> u32 { total(Pair { x: 4u32, y: 9u32 }) }
```

## 5. KernelSpec and kernargs

`KernelSpec { ... }` has no omitted/default fields:

| Field | Type | Meaning |
|---|---|---|
| `kernel_id` | String | logical kernel identity |
| `variant` | String | emitted variant identity |
| `arch` | Arch | target architecture |
| `symbol` | String | assembly entry symbol |
| `kernargs` | KernargLayout | argument metadata/layout |
| `user_sgpr_count` | u8 | user SGPR count; use 2 for the kernarg pointer |
| `system_sgpr_workgroup_id_y` | bool | enable workgroup y ID |
| `workgroup_size` | u16 | metadata maximum flat workgroup size |
| `group_segment_fixed_size` | u32 | statically allocated LDS bytes |
| `wave32` | bool | must be true (`builder requires wave32`) |
| `cu_mode` | bool | CU mode; metadata WGP mode is its inverse |

`KernargLayout::new(size: u32) -> KernargLayout` sets total bytes, including padding; example `KernargLayout::new(16)`. `.pointer(name: str, offset: u32) -> KernargLayout` adds an eight-byte global pointer; example `.pointer("input", 0)`. `.hidden(name: str, offset: u32, size: u32, value_kind: str) -> KernargLayout` adds explicit-sized metadata; example `.hidden("count", 8, 4, "by_value")`. `.validate() -> Result<(), String>` checks layout bounds/overlap; example `layout.validate()?`. These are metadata, not argument loads; offsets and total size must match the host's packed argument bytes. Metadata kernarg alignment is eight bytes.

For a shipped family, the host side of that agreement is not hand-packed: `crates/hipfire-kernel-abi` holds one `KernelAbiV1` per family layout (explicit fields and hidden words with offsets/widths/kinds, segment size/alignment, operand formats/access/alignment/extents, admitted domain, numeric contract, and per export: arch, pinned object SHA-256, block, dynamic and static LDS, grid). The `rdna-compute` launchers and `pm_loop` encode through `validate_invocation` → `encode_into`, which rejects a wrong arch/export/format/alignment/extent/domain/grid/block/LDS before enqueue; railgun builds its `LaunchRecord` from the same validated image and `railgun::contract::check_node` checks the authored node (and `KernelProgram::encode_kernarg_into` output) against the contract. `crates/rdna-compute/tests/kernel_abi_drift.rs` checks every contract against the generator's `KernelSpec`/`KernargLayout` and the committed object's metadata, with deliberately broken fixtures. Covered today: MQ6 int8 GEMM (PM and hipcc twins) and grouped symmetric-IU4 MoE (MQ4, E8P, W3). A contract is ABI agreement only, not numeric equivalence.

`smem(b, dst: u8, len: u8, base: u8, off: u32) -> Result<(), String>` loads `len` dwords from the 64-bit pointer in `s[base:base+1]` plus byte offset `off`. Allowed widths are 1/2/4/8, emitting `s_load_b32/b64/b128/b256`; other widths return `SMEM width`. The **entire destination range must fit one live plan allocation**. Four separately planned SGPRs do not own a four-dword load.

```rip body
smem(wg.isa(), 4, 4, 0, 0)?;
// s[4:7] is the single "args" allocation in the template.
op(wg.isa(), "v_mov_b32_e32 v1, s4", &[v(1)], &[s(4)])?;
```

## 6. Register plans

`RegPlan::new(vgpr_budget: u16, sgpr_budget: u16) -> Result<RegPlan, String>` accepts budgets up to 256 VGPRs and 104 SGPRs; example `let mut p = RegPlan::new(256, 104)?;`. A budget is a limit, not an allocation.

`p.s::<N>(name: str, base: u8, live: Live) -> Result<S<N>, String>` and `p.v::<N>(...) -> Result<V<N>, String>` allocate a named contiguous range; example `p.s::<4>("args", 4, Live::Whole)?;`. Only N = 1, 2, 4, 8 are supported (`register width must be 1, 2, 4 or 8`). Multi-dword scalar allocations require base alignment to their width. Register names in instruction strings are still physical registers, not these symbolic allocation names.

`Live::Whole` covers the whole kernel. `Live::Between(start: str, end: str)` covers a start-inclusive, end-exclusive label interval; example `Live::Between(".Lload", ".Ldone")`. Place those boundaries using `wg.label`. Do not reuse physical storage across overlapping lifetimes. `p.next_free_vgpr() -> u16` and `p.next_free_sgpr() -> u16` return the high-water marks; example `let top = p.next_free_vgpr();`. `p.scratch_pool(base: u8, len: u8) -> Result<(), String>` configures temporary VGPR storage; example `p.scratch_pool(128, 8)?;`.

Every declared access must have exactly one live owner containing the **whole access range**, not merely one owner per component. The exact diagnostic is, for example, `s[12:15] at entry has no unique live owner` (R0401). Fix a four-dword load by replacing four single-register allocations with `p.s::<4>("args", 12, Live::Whole)?;`, or split the actual load into owned widths. Ownership is checked for defs and uses, including incoming ABI registers you read.

Manual VGPR granule padding is **not required by RegPlan or the emitter**: the emitted count is the plan's high-water mark; the native wave32 descriptor encoder rounds its allocation encoding in units of eight. Some production scripts add an unused high-register owner to round metadata counts to eight; this is a convention, not a language requirement. Do not confuse descriptor encoding units with hardware occupancy/allocation granules.

## 7. Instruction helpers

`b` below is either a Builder or `wg.isa()`; register indices are u8, byte offsets u32. Helpers returning Result require `?`. Defs are registers written; uses are registers read, including read-modify-write destinations. These declarations drive automatic waits, register ownership and source-lock/hazard checks. Omitting an operand can drop a required wait; `check` rejects it with R0603. `MemoryClass::{VmemLoad,VmemStore,DsLoad,DsStore,SmemLoad,Export}` identifies the asynchronous operation family.

| Signature / return | Emission / example |
|---|---|
| `v(n), s(n) -> RegRef` | one VGPR/SGPR; `v(3)`, `s(4)` |
| `vr(base,len), sr(base,len) -> RegRef` | contiguous range (second argument is length, not last index); `sr(8,4)` is s[8:11] |
| `lit(x: u32) -> String` | canonical decimal for 0..64 and bit patterns -16..-1, hex otherwise; `lit(128)` is `"0x80"` |
| `op(b,text,defs:&[RegRef],uses:&[RegRef]) -> Result<(),String>` | given instruction text; `op(wg.isa(),"v_mov_b32_e32 v1, 0",&[v(1)],&[])?` |
| `mem(b,text,defs,uses,class) -> Result<(),String>` | given text with memory classification; `mem(wg.isa(),text,&[v(1)],&[v(0),sr(8,4)],MemoryClass::VmemLoad)?` |
| `sop(b,text,defs:&[u8],uses:&[u8]) -> Result<(),String>` | scalar single-register shorthand; `sop(wg.isa(),"s_mov_b32 s12, 0",&[12],&[])?` |
| `smem(b,dst,len,base,off) -> Result<(),String>` | scalar load, described above; `smem(wg.isa(),4,4,0,0)?` |
| `srd_tail(b,srd,records:Option<u8>) -> Result<(),String>` | writes descriptor words 2/3; records from SGPR or -1 for None; `srd_tail(wg.isa(),8,None)?` |
| `add64(b,dst,src,x) -> Result<(),String>` | SGPR-pair dst = src + scalar s[x], carry propagation; `add64(wg.isa(),12,4,6)?` |
| `add64_imm(b,reg,imm:u32) -> Result<(),String>` | increments an SGPR pair, using `lit`; `add64_imm(wg.isa(),12,128)?` |
| `bload(b,width,dst,voff,srd,offset) -> Result<(),String>` | buffer load width 1/2/4 dwords at v[voff]+offset; `bload(wg.isa(),1,1,0,8,0)?` |
| `bstore_b128(b,data,voff,srd,offset) -> Result<(),String>` | four-dword buffer store; `bstore_b128(wg.isa(),4,0,8,16)?` |
| `s_add_u32(arch), s_addc_u32(arch), s_add_i32(arch) -> str` | architecture-specific opcode strings, table below; `let add = s_add_u32(arch);` |
| `smem_off(arch,off:u32) -> String` | hex offset; zero is `null` on gfx11, `0x0` on gfx12; `smem_off(arch,0)` |
| `Instruction::new(text,defs,uses) -> Instruction` | instruction value for driver APIs; `Instruction::new("s_cmp_eq_u32 s12, 0", &[], &[s(12)])` |
| `insn.memory(class) -> Instruction` | classify instruction; `insn.memory(MemoryClass::DsStore)` |
| `workgroup_id(b,axis:u8,dst:u8) -> Result<(),String>` | read x/y (axis 0/1) into s[dst] using the architecture's ABI; `workgroup_id(wg.isa(),0,12)?;` |

`SRD_WORD3: u32 = 0x31004000` selects raw untyped 32-bit buffer accesses: offsets at/past num_records read zero without touching memory. `srd_tail` uses this constant. Buffer helpers use `offen`, zero soffset (`0` gfx11, `null` gfx12), and byte immediate offsets, limited to 4095 on gfx11 and 8388607 on gfx12. They do not construct descriptor base words 0/1. Buffer and global operations are different families of addressing instructions; do not replace one spelling with the other.

`gather_offset(b,dst:u8,index:u8,div:Option<u8>,row_bytes:u32,live:u8,t:GatherTemps) -> Result<(),String>` computes signed indexed-row byte offsets and live-lane mask. `GatherTemps { v:[u8;4], mask:u8 }` supplies four dead VGPRs and scalar mask scratch. Example: `gather_offset(wg.isa(),1,0,None,16,12,GatherTemps { v:[8,9,10,11], mask:13 })?;`. Live indices must be nonnegative and below `GATHER_MAX_INDEX = 1<<22`; negative padding indices become `GATHER_OOB = 0x7fffff00`. A supplied divisor is an SGPR index (positive uniform divisor); None means one. `row_bytes` must be below 2^24. Set the descriptor's record count to the actual byte extent; the helper emits reciprocal division plus integer corrections, offset multiplication and mask selection, not a load.

`Chain::new(dst:u8,width:u8,kind:str) -> Chain` describes a WMMA accumulator; width must be 8 and kind `"f16"` or `"bf16"`. Example `let c = Chain::new(0,8,"f16");`. `c.step(b,a:u8,aw:u8,b:u8,bw:u8) -> Result<(),String>` emits the kind's `v_wmma_f32_16x16x16_*` using and redefining its accumulator; example `c.step(wg.isa(),8,8,16,8)?;` on gfx11. Operand widths must match the architecture (eight on gfx11, four on gfx12); initialize the accumulator before its first step. The item keyword `chain` merely defines a function and is unrelated to this native Chain value.

`Wmma::swmmac(arch, kind: Swmmac, dst, a, b, index: RegRef, mods: SwmmacMods) -> Result<Instruction,String>` builds a gfx12 sparse (2:4) WMMA; push it with `wg.isa().push(..)?`. `kind` is `Swmmac::<Kind>`; `dst`, `a` and `b` are `vr(..)` ranges of exactly the kind's width (table below); `index` is one VGPR (`v(n)`); `mods` is the struct literal `SwmmacMods { index_key, neg_lo, neg_hi, clamp }` with all four fields written (`u8, u8, u8, bool`). SWMMAC has no C operand: `dst` is read in place as the accumulator, so the instruction declares defs `[dst]` and uses `[a, b, index, dst]`, and the accumulator must be initialized first. Example: `wg.isa().push(Wmma::swmmac(arch, Swmmac::F32F16, vr(24,8), vr(8,4), vr(12,8), v(20), SwmmacMods { index_key: 0, neg_lo: 0, neg_hi: 0, clamp: false })?)?;` emits `v_swmmac_f32_16x16x32_f16 v[24:31], v[8:11], v[12:19], v20`. Refusals carry the Rust error text: `SWMMAC requires gfx12` (gfx1100/gfx1151), `<mnemonic>: A requires 4 VGPRs, found v[8:10]` (width), and the modifier errors below. A hand-written `op` with the same text must declare `dst` in its uses as well. Worked example: `docs/cookbook/10-swmmac.md`.

| `Swmmac::` kind | mnemonic | D | A | B | modifiers |
|---|---|---|---|---|---|
| `F32F16` | v_swmmac_f32_16x16x32_f16 | 8 | 4 | 8 | float |
| `F32Bf16` | v_swmmac_f32_16x16x32_bf16 | 8 | 4 | 8 | float |
| `F16F16` | v_swmmac_f16_16x16x32_f16 | 4 | 4 | 8 | float |
| `Bf16Bf16` | v_swmmac_bf16_16x16x32_bf16 | 4 | 4 | 8 | float |
| `I32Iu8` | v_swmmac_i32_16x16x32_iu8 | 8 | 2 | 4 | int |
| `I32Iu4` | v_swmmac_i32_16x16x32_iu4 | 8 | 1 | 2 | int |
| `I32Iu4X64` | v_swmmac_i32_16x16x64_iu4 | 8 | 2 | 4 | int, no `index_key` |
| `F32Fp8Fp8` | v_swmmac_f32_16x16x32_fp8_fp8 | 8 | 2 | 4 | fp8 |
| `F32Fp8Bf8` | v_swmmac_f32_16x16x32_fp8_bf8 | 8 | 2 | 4 | fp8 |
| `F32Bf8Fp8` | v_swmmac_f32_16x16x32_bf8_fp8 | 8 | 2 | 4 | fp8 |
| `F32Bf8Bf8` | v_swmmac_f32_16x16x32_bf8_bf8 | 8 | 2 | 4 | fp8 |

Counts are VGPRs (dwords); the index is always one VGPR. Legal modifier sets (as accepted by `llvm-mc -mcpu=gfx1201`; anything else is refused as `<mnemonic> does not accept <modifier>`, `<mnemonic> has no index_key` or `<mnemonic>: neg_lo/neg_hi bit 2 (C) is not encodable`):

| class | `index_key` | `neg_lo` | `neg_hi` | `clamp` |
|---|---|---|---|---|
| float (`F32F16` `F32Bf16` `F16F16` `Bf16Bf16`) | 0 or 1 | bits 0-1 (A, B) | bits 0-1 | no |
| int (`I32Iu8` `I32Iu4`) | 0 or 1 | bits 0-1 | no | yes |
| `I32Iu4X64` | no (must be 0) | bits 0-1 | no | yes |
| fp8 (`F32Fp8Fp8` `F32Fp8Bf8` `F32Bf8Fp8` `F32Bf8Bf8`) | 0 or 1 | no | no | no |

`neg_lo`/`neg_hi` bit 0 negates A and bit 1 negates B; bit 2 (C) never encodes. Text order is `index_key:1 neg_lo:[a,b,0] neg_hi:[a,b,0] clamp`; `index_key:1` is OP_SEL bit 0 (`index_key:0` is the default and is not printed). In instruction text `index_key:N` is accepted only on `v_swmmac_*` rows other than `v_swmmac_i32_16x16x64_iu4`.

## 8. Workgroup driver

### Scope and complete skeleton

`let mut wg = Workgroup::new(&mut builder, arch)?;` must be a let initializer. Its scope ends at the first later statement mentioning the **builder variable's name**, even `let builder = wg.isa();`. Subsequent access fails with `wg.isa() outside the Workgroup scope` (R0305). Avoid borrowing an ISA alias in the owning block; call `wg.isa()` directly. Recommended: construct the Workgroup in `emit`, then call a separate author function. There, `let b = wg.isa();` does not mention the original `builder` identifier.

The following complete script is also the template for every `rip body` block on this page: splice statements at the comment. It supplies s0..s15 and v0..v15, but do not read uninitialized scratch values in a real kernel.

```rip
fn emit(spec: Spec) -> Result<Emitted, String> {
    let arch = spec.arch;
    let kspec = KernelSpec { kernel_id: "doc".into(), variant: "default".into(), arch, symbol: "doc_kernel".into(),
        kernargs: KernargLayout::new(16).pointer("a", 0).pointer("b", 8), user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: false, workgroup_size: 64, group_segment_fixed_size: 4096, wave32: true, cu_mode: false };
    let mut p = RegPlan::new(256, 104)?;
    p.s::<2>("kernarg_ptr", 0, Live::Whole)?;
    if !arch.gfx12() { p.s::<1>("wg_x", 2, Live::Whole)?; }
    p.s::<4>("args", 4, Live::Whole)?; p.s::<4>("srd", 8, Live::Whole)?; p.s::<4>("t", 12, Live::Whole)?;
    p.v::<8>("v", 0, Live::Whole)?; p.v::<8>("w", 8, Live::Whole)?;
    let mut builder = Builder::new(kspec, p);
    let mut wg = Workgroup::new(&mut builder, arch)?;
    author(&mut wg, arch)?;
    builder.finish()
}
region Doc;
fn author<T: Target>(wg: &mut Wg<T>, arch: Arch) -> Result<(), String> {
    let end = wg.exit(".Ldoc_end")?;
    /*BODY*/
    wg.end(end)
}
```

### LDS ownership and tokens

These signatures describe script values, which sometimes differ from the Rust runtime API. `Result<X>` below abbreviates `Result<X,String>`. A Place is a region, ring or its Pending store token. Region states are Free → Writing → Published → Free. A wait drains operations; a barrier publishes or retires ownership. A barrier is not a memory-counter wait.

| Script signature / return | Meaning and one-line example |
|---|---|
| `wg.isa() -> Isa` | checked instruction handle; `op(wg.isa(),text,defs,uses)?` |
| `Workgroup::new(builder,arch) -> Result<Workgroup>` | open scoped driver, must initialize let; `let mut wg = Workgroup::new(&mut builder,arch)?;` |
| `wg.position() -> usize` | instruction position; `let at = wg.position();` |
| `wg.lds::<R>(name:str,base:u32,len:u32) -> Result<Region>` | allocate LDS bytes with tag R; `let r = wg.lds::<Doc>("scratch",0,256)?;` |
| `wg.begin_write(p:Place) -> (Place,Pending)` | mark Writing without an instruction; **no `?`**; `let (r,p) = wg.begin_write(r);` |
| `wg.ds_store(p,insn) -> Result<(Place,Pending)>` | store to Free/Writing; `let (r,p) = wg.ds_store(r,insn)?;` |
| `wg.ds_store2(x,y,insn) -> Result<((Place,Pending),(Place,Pending))>` | one instruction stores to two distinct places; `let ((x,px),(y,py)) = wg.ds_store2(x,y,insn)?;` |
| `wg.ds_load(r,insn) -> Result<()>` | load Published region; `wg.ds_load(r,insn)?;` |
| `wg.ds_load_cur(g,insn) -> Result<()>` | load ring current Published buffer; `wg.ds_load_cur(g,insn)?;` |
| `wg.wait(p) -> Result<Drained>` | drain a Writing place's stores (not a Published place's loads); `let d = wg.wait(p)?;` |
| `wg.wait_all((p,q,...)) -> Result<(Drained,Drained,...)>` | same-order drained tuple; `let (dp,dq) = wg.wait_all((p,q))?;` |
| `ready(r[,drained]) -> Transition` | Writing→Published request; `ready(r,d)` |
| `retire(r[,drained]) -> Transition` | Published→Free request; no store wait needed; `retire(r)` |
| `wg.barrier((transition,...)) -> Result<(handle,...)>` | apply transitions in order, same-arity tuple; `let (r,) = wg.barrier((ready(r,d),))?;` |
| `wg.signal((transition,...)) -> Result<Arrived>` | gfx1201 split arrival, holds affected handles; `let a = wg.signal((ready(r,d),))?;` |
| `wg.wait_arrived(a:Arrived) -> Result<()>` | complete split barrier and apply transitions; `wg.wait_arrived(a)?;` |
| `wg.join((r,s,...)) -> Result<Region>` | consume ≥2 Free regions, up to four slots; `let rs = wg.join((r,s))?;` |
| `join((r,s,...)) -> Result<Region>` | free-function spelling; `let rs = join((r,s))?;` |
| `wg.split(r,first:usize) -> Result<(Region,Region)>` | split Free joined region at slot count, not byte offset; `let (r,s) = wg.split(rs,1)?;` |
| `wg.ring(name,x,y) -> Result<Ring>` | consume distinct Free regions; `let g = wg.ring("pingpong",x,y)?;` |
| `Ring::new(x,y[,z[,w]]) -> Ring` | ring construction over two to four Free regions (depth 2, 3 or 4; a bad region still errors, but the call returns the Ring itself: no `?`); the first store goes to the first region; `let g = Ring::new(x,y);` or `let g = Ring::new(x,y,z);` |
| `g.into_steady() -> Result<Ring>` | mark current buffer vacant for steady rotation; `let g = g.into_steady()?;` |
| `wg.ring_steady(g) -> Result<Ring>` | same steady-ring operation; `let g = wg.ring_steady(g)?;` |
| `g.cur_index(), g.next_index(), g.depth() -> usize` | current/next backing indices (`next = (cur + depth - 1) % depth`) and the number of buffers; `let i = g.cur_index();` |
| `g.starting_at(k) -> Ring` | on a fresh ring (nothing stored or published): buffer `k` takes the first store, so `next_index() == k`; refused (`before anything is published`) afterwards; `let g = g.starting_at(2);` |
| `g.phases() -> (Phase,Phase)` | backing phases; `let phases = g.phases();` |
| `r.phase() -> Phase` | Free/Writing/Published; `let phase = r.phase();` |
| `prime(g[,d]), rotate(g[,d]), retire_cur(g[,d]), advance(g) -> Transition` | ring barrier transitions; `let (g,) = wg.barrier((rotate(g,d),))?;`. A depth-D ring (`D >= 3`) fills over `D - 1` barriers: `prime`, then `rotate` over the vacant current buffer (nothing to retire); a read of a vacant current buffer is refused. `advance(g)` stores nothing: it retires the current buffer and makes the published buffer after it current, and is refused unless one is published (a depth-2 ring never has one) |
| `wg.relayout() -> Result<()>` | require all Free; invalidate old handles; `wg.relayout()?;` |

Optional Drained tokens document the operation you waited for, but the driver also checks its tracked youngest store: old evidence cannot cover a newer store. Use the tokens returned by the latest operations. `signal/wait_arrived` are gfx1201 only; gfx11 has one `s_barrier`, so use `barrier`.

```rip body
let r = wg.lds::<Doc>("scratch", 0, 256)?;
op(wg.isa(), "v_lshlrev_b32_e32 v1, 2, v0", &[v(1)], &[v(0)])?;
let store = Instruction::new("ds_store_b32 v1, v0", &[], &[v(1),v(0)]).memory(MemoryClass::DsStore);
let (r, pending) = wg.ds_store(r, store)?;
let drained = wg.wait(pending)?;
let (r,) = wg.barrier((ready(r, drained),))?;
let load = Instruction::new("ds_load_b32 v2, v1", &[v(2)], &[v(1)]).memory(MemoryClass::DsLoad);
wg.ds_load(r, load)?;
op(wg.isa(), "v_xor_b32_e32 v3, 1, v2", &[v(3)], &[v(2)])?;
let (r,) = wg.barrier((retire(r),))?;
```

### Conditions, closures, loops and exits

Conditions record SCC provenance. Consume a fresh condition before an SCC-writing instruction intervenes. `scmp_wg_uniform` is the author's assertion that operands are workgroup-uniform (kernargs, workgroup IDs, constants, and derived counters), not a proof of that assertion.

All closure methods below return Results. Closures return `Ok(...)` or propagate with `?`. `R` denotes the closure's script result value. A closure emits code; it is not called once per runtime lane. Loop closures can be replayed to reach a builder fixed point: keep emission deterministic.

| Script signature / return | Semantics and one-line example |
|---|---|
| `wg.scmp(insn) -> Result<Cond>` | wave-uniform scalar compare; `let c = wg.scmp(Instruction::new("s_cmp_eq_u32 s12, 0",&[],&[s(12)]))?;` |
| `wg.scmp_wg_uniform(insn) -> Result<WgCond>` | author claims WG uniformity; `let c = wg.scmp_wg_uniform(insn)?;` |
| `wg.skip_if(c,label,\|w\| -> Result<()> {...}) -> Result<()>` | skip body when true; `wg.skip_if(c,".Lskip",\|w\| Ok(()))?;` |
| `wg.skip_unless(c,label,\|w\| -> Result<()> {...}) -> Result<()>` | skip body when false; `wg.skip_unless(c,".Lskip",\|w\| Ok(()))?;` |
| `wg.wg_skip_if(c:WgCond,label,\|w\| -> Result<()> {...}) -> Result<()>` | WG-uniform skip, barriers allowed; `wg.wg_skip_if(c,".Lskip",\|w\| Ok(()))?;` |
| `skip_if/skip_unless/wg_skip_if(c,label,state,\|w,st\| -> Result<S> {...}) -> Result<S>` | explicit script state input/output; `let st = wg.skip_if(c,".Lskip",st,\|w,st\| Ok(st))?;` |
| `wg.exec_if(c,\|w\| -> Result<R> {...}) -> Result<R>` | EXEC all/none from SCC, then restored to all; wave scope; `wg.exec_if(c,\|w\| Ok(()))?;` |
| `wg.if_else(c,else_label,join_label,then,els) -> Result<()>` | **true selects els**, false selects then; both receive w, return Result<()>; `wg.if_else(c,".Lelse",".Ljoin",\|w\| Ok(()),\|w\| Ok(()))?; wg.label(".Ljoin")?;` |
| `wg.wg_if_else(c:WgCond,else_label,join_label,[state,]then,els) -> Result<S>` | workgroup-uniform two-way branch whose arms may hold barriers and change LDS ownership: **true selects els**, false selects then (which ends with `s_branch join`); each arm is `\|w[,st]\|`; the join label is placed by the call; both arms must end with the same LDS ownership shape and the same ring `cur` index and live buffers (else the call is refused, naming the difference); with `state` it returns the else arm's value; `let st = wg.wg_if_else(c,".Lelse",".Ljoin",st,\|w,st\| Ok(st),\|w,st\| Ok(st))?;` |
| `wg.forward(\|w,fw\| -> Result<R> {...}) -> Result<R>` | forward-only wave branches; `wg.forward(\|w,fw\| { fw.goto(".Lf")?; fw.place(".Lf")?; Ok(()) })?;` |
| `fw.branch_if(c,label), fw.branch_unless(c,label), fw.goto(label), fw.place(label) -> Result<()>` | every target placed once after branch; `fw.branch_if(c,".Lf")?;` |
| `wg.loop_until(head,exit,\|w,brk\| -> Result<()> {...}) -> Result<()>` | wave-scope device loop; `wg.loop_until(".Lhead",".Lbreak",\|w,brk\| { let c = w.scmp(insn)?; w.break_if(c,brk) })?;` |
| `wg.break_if(c,brk) -> Result<()>` | conditional exit from enclosing loop_until; `w.break_if(c,brk)?;` |
| `wg.loop_carried(head,\|w\| -> Result<WgCond> {...}) -> Result<()>` | WG loop, true back edge; `wg.loop_carried(".Lhead",\|w\| w.scmp_wg_uniform(insn))?;` |
| `wg.loop_carried(head,state,\|w,st\| -> Result<(S,WgCond)> {...}) -> Result<S>` | carry script state and WG condition; `let st = wg.loop_carried(".Lhead",st,\|w,st\| Ok((st,w.scmp_wg_uniform(insn)?)))?;` |
| `wg.exit(label) -> Result<Exit>` | reserve final kernel exit; `let end = wg.exit(".Lend")?;` |
| `wg.exit_if(c:WgCond,end) -> Result<()>` | whole WG exits when true; `wg.exit_if(c,&end)?;` |
| `wg.wave_exit_unless(c:Cond,end) -> Result<()>` | wave exits when false; **no later WG barrier**; `wg.wave_exit_unless(c,&end)?;` |
| `wg.wg_exit_unless(c:WgCond,target,end,\|w\| -> Result<()> {...}) -> Result<()>` | false runs body and exits; true continues at target; `wg.wg_exit_unless(c,".Lcontinue",&end,\|w\| Ok(()))?;` |
| `wg.end(end) -> Result<()>` | place exit and emit s_endpgm; `wg.end(end)?;` |
| `wg.end_with(end,\|w\| -> Result<()> {...}) -> Result<()>` | exit tail in wave scope; nonempty tail forbidden if any branch targets exit; `wg.end_with(end,\|w\| Ok(()))?;` |
| `wg.handoff(readers:Cond,label,end,r,write,read) -> Result<R>` | final wave-role phase: write receives `(w,r)` and returns Result<()>; read receives `(w,r,exit)` and returns Result<R>; `wg.handoff(c,".Lread",end,r,\|w,r\| { let _ = w.begin_write(r); Ok(()) },\|w,r,exit\| Ok(()))?;` |
| `wg.label(name:str) -> Result<()>` | straight-line lifetime boundary; `wg.label(".Lphase")?;` |

Wave-scope bodies (`skip_if`, `exec_if`, `forward`, `if_else`, `loop_until`) cannot contain workgroup barriers or change the LDS ownership shape across joining paths. Workgroup-uniform skips must likewise restore the skipped path's ownership shape; `wg_if_else` arms may instead change it, but must agree with each other at the join. Loops must have matching entry/back-edge ownership, including drain evidence; `loop_until` needs at least one `break_if`. `handoff` starts with a Free region, writers must leave it Writing, and the driver performs wait/publication, joins readers at the barrier and ends the kernel. Do not call `end` again after handoff.

```rip body
sop(wg.isa(), "s_mov_b32 s12, 0", &[12], &[])?;
let c = wg.scmp(Instruction::new("s_cmp_eq_u32 s12, 0", &[], &[s(12)]))?;
wg.skip_if(c, ".Lskip_doc", |w| {
    op(w.isa(), "v_mov_b32_e32 v1, 7", &[v(1)], &[])?;
    Ok(())
})?;
```

## 9. RDNA launch ABI

| Entry value | gfx1100 / gfx1151 | gfx1201 |
|---|---|---|
| kernarg pointer | s[0:1], with user_sgpr_count=2 | same |
| workgroup x | first system SGPR after user SGPRs: s2 with count=2; plan must own it | ttmp9, not in RegPlan |
| workgroup y (when enabled) | next SGPR: s3 with count=2; plan must own it | low 16 bits of ttmp7 |
| workgroup z ABI packing | next enabled system SGPR in ABI order | high 16 bits of ttmp7 |
| workitem x / flat thread ID | v0 within the workgroup | same |
| wave/EXEC | wave32; incoming EXEC identifies active workitems | same |

The current descriptor always enables x, optionally y, and disables z: z is **not available** through this KernelSpec on either architecture. It enables only workitem x (not separate y/z VGPRs). This reference assumes one-dimensional workgroups, where incoming v0 is the flat workgroup-local workitem index; for multidimensional launches v0 contains x only. It is not a global thread index or necessarily a lane index; for a 1D launch derive the lane with `v0 & 31`. Own every SGPR/VGPR range you access; ttmp registers are special registers outside the plan. A scalar group ID does not vary per wave or lane. Entry EXEC identifies active workitems, not necessarily every lane of a final partial wave.

Prefer `workgroup_id(wg.isa(), 0, 12)?;` over hand-written s2/ttmp9 selection. It defines s12; on gfx11 its incoming system SGPR source still needs a plan owner. gfx12 axis 0 emits `s_mov_b32` from ttmp9, axis 1 emits `s_and_b32` from ttmp7 with 0xffff. gfx11 uses `s_mov_b32` from the system SGPR after the user SGPRs (plus one for y). Axis 1 requires `system_sgpr_workgroup_id_y: true`; axis 2 is refused on both architectures.

```rip body
workgroup_id(wg.isa(), 0, 12)?;
op(wg.isa(), "v_mov_b32_e32 v1, s12", &[v(1)], &[s(12)])?;
```

## 10. Architecture-specific instruction vocabulary

The tables are the supported PeaceMaker vocabulary, not a complete silicon ISA catalogue. Use exact suffixes: `v_add_nc_u32_e32` exists; bare `v_add_nc_u32` is not its alias.

| Operation | gfx1100/gfx1151 | gfx1201 |
|---|---|---|
| unsigned add / carry-in | s_add_u32 / s_addc_u32 | s_add_co_u32 / s_add_co_ci_u32 |
| signed add | s_add_i32 | s_add_co_i32 |
| subtract / borrow-in | s_sub_u32 / s_subb_u32 | s_sub_co_u32 / s_sub_co_ci_u32 |
| signed subtract | s_sub_i32 | s_sub_co_i32 |
| float min/max | v_min_f32_e32 / v_max_f32_e32 | v_min_num_f32_e32 / v_max_num_f32_e32 |
| float DPP max | not in current tables | not in current tables |
| zero buffer soffset | 0 | null |
| zero SMEM offset | null | 0x0 |
| scalar dependency wait | s_waitcnt_depctr | s_wait_alu |
| barrier | s_barrier | s_barrier_signal -1; s_barrier_wait 0xffff |
| immediate buffer byte offset max | 4095 | 8388607 |
| v_permlane16_b32 | in gfx11 tables | absent |
| v_permlanex16_b32 | available | available |
| DPP row_xmask / row_shl | v_mov_b32_dpp | v_mov_b32_dpp |
| DPP quad_perm:[...] syntax | rejected by current modifier parser | rejected by current modifier parser |
| f16 WMMA 16x16x16 operands | dst/acc 8 VGPRs, A/B 8 each | dst/acc 8 VGPRs, A/B 4 each |

Inline integer constants are -16..64. A larger integer is a literal, spelled canonically with `lit`, not a decimal inline constant. There is one literal word per instruction (reusing the same literal is different from requesting two distinct literal words). VOP2 `_e32` has VGPR-only src1: use `_e64` when that source must be scalar or constant, or commute operands if valid. Not every opcode has both encodings in the tables. DPP controls are instruction modifiers, not standalone cross-lane instructions. `row_shr` is not a supported substitute for `row_shl`.

The following checks prove representative spellings and intentional rejections. F16 WMMA and SWMMAC widths are in dwords, not element counts.

```mnemonics
gfx1151: s_add_u32 s12, s12, s13
gfx1201: s_add_co_u32 s12, s12, s13
gfx1151: s_addc_u32 s12, s12, 0
gfx1201: s_add_co_ci_u32 s12, s12, 0
gfx1151: s_add_i32 s12, s12, s13
gfx1201: s_add_co_i32 s12, s12, s13
gfx1151: s_sub_u32 s12, s12, s13
gfx1201: s_sub_co_u32 s12, s12, s13
gfx1151: s_subb_u32 s12, s12, 0
gfx1201: s_sub_co_ci_u32 s12, s12, 0
gfx1151: s_sub_i32 s12, s12, s13
gfx1201: s_sub_co_i32 s12, s12, s13
gfx1151: v_min_f32_e32 v1, v2, v3
gfx1151: v_max_f32_e32 v1, v2, v3
gfx1201: v_min_num_f32_e32 v1, v2, v3
gfx1201: v_max_num_f32_e32 v1, v2, v3
!gfx1151: v_max_f32_dpp v1, v2, v3 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
!gfx1201: v_max_num_f32_dpp v1, v2, v3 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
gfx1151: buffer_load_b32 v1, v0, s[8:11], 0 offen offset:4095
gfx1201: buffer_load_b32 v1, v0, s[8:11], null offen offset:8388607
gfx1151: s_load_b32 s4, s[0:1], null
gfx1201: s_load_b32 s4, s[0:1], 0x0
gfx1151: s_waitcnt_depctr 0
gfx1201: s_wait_alu 0
gfx1151: s_barrier
gfx1201: s_barrier_signal -1
gfx1201: s_barrier_wait 0xffff
gfx1151: v_permlane16_b32 v1, v2, s12, s13
!gfx1201: v_permlane16_b32 v1, v2, s12, s13
gfx1151: v_permlanex16_b32 v1, v2, s12, 0xfedcba98
gfx1201: v_permlanex16_b32 v1, v2, s12, 0xfedcba98
gfx1151: v_mov_b32_dpp v1, v2 row_xmask:1 row_mask:0xf bank_mask:0xf
gfx1201: v_mov_b32_dpp v1, v2 row_xmask:1 row_mask:0xf bank_mask:0xf
!gfx1151: v_mov_b32_dpp v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
!gfx1201: v_mov_b32_dpp v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
gfx1151: v_mov_b32_dpp v1, v2 row_shl:1 row_mask:0xf bank_mask:0xf
gfx1201: v_mov_b32_dpp v1, v2 row_shl:1 row_mask:0xf bank_mask:0xf
gfx1151: v_add_nc_u32_e32 v1, 0x80, v2
gfx1201: v_add_nc_u32_e32 v1, 0x80, v2
!gfx1151: v_add_nc_u32_e32 v1, v2, 0x80
!gfx1201: v_add_nc_u32_e32 v1, v2, 0x80
gfx1151: v_add_nc_u32_e64 v1, v2, 0x80
gfx1201: v_add_nc_u32_e64 v1, v2, 0x80
!gfx1151: v_add_nc_u32_e64 v1, 0x80, 0x81
!gfx1201: v_add_nc_u32_e64 v1, 0x80, 0x81
!gfx1151: v_add_nc_u32 v1, v2, v3
!gfx1201: v_add_nc_u32 v1, v2, v3
gfx1151: v_wmma_f32_16x16x16_f16 v[0:7], v[8:15], v[16:23], v[0:7]
gfx1201: v_wmma_f32_16x16x16_f16 v[0:7], v[8:11], v[16:19], v[0:7]
gfx1201: v_swmmac_f32_16x16x32_f16 v[0:7], v[8:11], v[12:19], v20
gfx1201: v_swmmac_f32_16x16x32_f16 v[0:7], v[8:11], v[12:19], v20 index_key:1 neg_lo:[1,1,0] neg_hi:[1,1,0]
gfx1201: v_swmmac_i32_16x16x32_iu8 v[0:7], v[8:9], v[10:13], v14 index_key:1 neg_lo:[1,1,0] clamp
gfx1201: v_swmmac_f32_16x16x32_fp8_fp8 v[0:7], v[8:9], v[10:13], v14 index_key:1
!gfx1201: v_swmmac_i32_16x16x64_iu4 v[0:7], v[8:9], v[10:13], v14 index_key:1
!gfx1151: v_swmmac_f32_16x16x32_f16 v[0:7], v[8:11], v[12:19], v20
gfx1151: s_waitcnt vmcnt(0)
gfx1151: s_waitcnt lgkmcnt(0)
gfx1151: s_waitcnt_vscnt null, 0
gfx1151: s_waitcnt expcnt(0)
gfx1201: s_wait_loadcnt 0
gfx1201: s_wait_storecnt 0
gfx1201: s_wait_dscnt 0
gfx1201: s_wait_kmcnt 0
```

## 11. Waits and counters

Declared defs/uses let the builder insert waits for SMEM→use, buffer-load→use, DS-load→use and source reuse hazards. For example the `smem` body example above needs no handwritten wait before reading s4. Do not add blanket waits after every load. Conversely, omitting uses is not a performance optimization: it suppresses dependency knowledge.

`b.wait(counter: Counter, n: u8) -> Result<(),String>` emits an explicit ordering wait; example `wg.isa().wait(if arch.gfx12() { Counter::Store } else { Counter::Vs },0)?;`. The count is the maximum outstanding count allowed after the wait, not a cycle delay. Use driver waits for LDS ownership/publication, not raw waits alone.

| Counter | gfx1100/gfx1151 | gfx1201 |
|---|---|---|
| vector loads | Vm → s_waitcnt vmcnt(n) | Load → s_wait_loadcnt n |
| vector stores | Vs → s_waitcnt_vscnt null,n | Store → s_wait_storecnt n |
| LDS | Lgkm → s_waitcnt lgkmcnt(n) | Ds → s_wait_dscnt n |
| scalar memory | Lgkm (shared with LDS) | Km → s_wait_kmcnt n |
| export | Exp → s_waitcnt expcnt(n) | ledger accepts Exp but formats legacy s_waitcnt; avoid for gfx12 |

Passing a counter from the other architecture returns exactly `counter/architecture mismatch` (R0404). The gfx12 operational counter set is Load/Store/Ds/Km; gfx11 is Vm/Vs/Lgkm/Exp. `Exp` is a legacy ledger exception, not a portable gfx12 wait spelling. Architecture strings in the table are assembly, while Counter names are case-sensitive script enum variants.

```rip body
wg.isa().wait(if arch.gfx12() { Counter::Store } else { Counter::Vs }, 0)?;
```

## 12. Diagnostics and the agent loop

```sh
peacemaker-rip check kernel.rip --variant default
peacemaker-rip check kernel.rip --variant default --arch gfx1201 --json
peacemaker-rip fmt kernel.rip
peacemaker-rip fmt --check kernel.rip
peacemaker-rip explain R0301
peacemaker-rip explain --list
peacemaker-rip build kernel.rip both default kernel.s
peacemaker-rip build --certify kernel.rip gfx1151 default kernel.s
```

Read the code, severity, file/span and source caret, then the help and suggested replacement. JSON diagnostics include `code`, `severity`, `message`, `file`, `span`, `help`, `suggestion`, `notes`, and `explain`. Fix the earliest underlying error and rerun every architecture and variant you intend to ship. R0603 is an error: a missing effect can remove a necessary dependency wait. R0604 rejects a register read with no entry initialization or write anywhere in the lifted kernel; it is not an ordering/path proof. `check` permits any remaining warnings (exit 0); errors exit 1 and usage errors exit 2.

The [diagnostic reference](ERRORS.md) explains each stable code. R01xx is parsing/macros, R02xx evaluation, R03xx the driver, R04xx the builder, R05xx entry contract, R06xx instruction validation, R07xx certification and R09xx tools. `fmt` uses rustfmt with token-stream verification; formatting does not compile a kernel. `build` writes assembly; `both` produces architecture-suffixed output paths. `--certify` additionally assembles/links and runs wait-ledger replay and byte-exact lifting checks; it is not a numerical test.

### Source map

Entry/evaluation: `src/lib.rs`, `interp.rs`, `ops.rs`, `builtin.rs`, `expand.rs`, `fmt.rs`, `preprocess.rs`. Script-native signatures: `src/native.rs` (production generators: `src/native_gen.rs`; the fine `iu4_gemm` K-loop natives: `src/native_kloop.rs`). Kernel/layout and ownership: `peacemaker-isa/src/plan.rs`, `reg.rs`, `emit.rs`. Instruction helpers and waits: `peacemaker-isa/src/kernels/common.rs`, `ledger.rs`. Driver state/control flow: `peacemaker-author/src/runtime.rs` (the typed-ring bridge `Lend`: `take`, `take_writing`, `give_writing`, `view`). Instruction spellings/operand widths: `peacemaker-ir/isa/gfx1100.tbl`, `gfx12.tbl`. ABI entry initialization: `peacemaker-ir/src/emu/mod.rs`.

## 13. Production generators: `iu4_gemm` and `qwen4_moe_sym`

The fast kernels are Rust generators. A script reaches them in two layers, neither of which adds typestate surface: whole-kernel natives that take the generator's `Spec`, and (for `iu4_gemm`) its phase blocks, which a script composes under the same checks as its own code. Production output does not change: a composition that mirrors `iu4_gemm::emit` re-emits its bytes.

### Coarse natives

The `Spec` enums are enum values, like a script's own unit enums: `Fold::{K128, K256Shared, K256Pow2}`, `Tile::{T128x128x8, T256x128x16}`, `Cacc::{One, Two}`, `Epi::{Set, Add, AddSigmoid, GateUpSilu, GateUpSiluBf16, QkvzaGdn}`, `ALayout::{Token, Slab}`, `Mma::{Dense, S24X64, S24X32}`, `RingDepth::{Two, Three}`, `FoldSched::{Paired, Chain}` (`iu4_gemm`) and `MoeKind::{GateUp, Down}`, `WeightCodec::{Mq4Sym, Mq3I4, E8pI4}` (`qwen4_moe_sym`; `MoeKind` is the Rust `qwen4_moe_sym::Kind`, renamed so it cannot clash with the `enum Kind` a script declares for its own variants). A script's own `enum Fold` (or any of these names) wins over the native one. `Arch` is the existing `Arch::Gfx1201`. The struct literals map one to one onto the Rust `Spec`; every field is required, and an unknown field is an error.

| native | Rust | result |
|---|---|---|
| `Iu4Spec { fold, tile, cacc, epi, act, arch, mma [, ring] [, fold_sched] }` | `iu4_gemm::Spec` | a spec value (`mma: Mma::Dense` is the shipped WMMA K-loop; `S24X64` / `S24X32` the 2:4-sparse SWMMAC K-loops; the optional `ring: RingDepth::Two \| Three` is `Tune::ring`, the A/W payload rings' depth, default `Two`; the optional `fold_sched: FoldSched::Paired \| Chain` is `Tune::fold_sched`, default `Paired`: `Chain` makes `iu4::wmma` (fold) and `iu4::fold_scale` emit the chained fold and its `s_setprio 1`, for the dense k128 128x128x8 cacc1 K-loop at ring depth 2) |
| `Iu4Gemm::emit(spec)` | `iu4_gemm::emit` | `Result<Emitted, String>` |
| `Iu4Gemm::emit_module(fold, tile, cacc, act, arch)` | `iu4_gemm::emit_module` | `Result<(Vec<Emitted>, String, String), String>`: the symbols, the module text and its proof as JSON |
| `MoeSymSpec { arch, kind, nt, rr, codec }` | `qwen4_moe_sym::Spec` | a spec value |
| `MoeSym::emit(spec)` | `qwen4_moe_sym::emit` | `Result<Emitted, String>` |
| `MoeSym::emit_module(arch)` | `qwen4_moe_sym::emit_module` | as `Iu4Gemm::emit_module` |
| `MoeSym::emit_module_codec(arch, codec)` | `qwen4_moe_sym::emit_module_codec` | as `Iu4Gemm::emit_module` |

`Spec::validate` stays the gate: a combination without a generator returns the Rust refusal (`iu4_gemm targets gfx1201 only`, ...).

```rip arch=gfx1201
fn emit(spec: Spec) -> Result<Emitted, String> {
    Iu4Gemm::emit(Iu4Spec {
        fold: Fold::K128, tile: Tile::T128x128x8, cacc: Cacc::One,
        epi: Epi::GateUpSilu, act: ALayout::Slab, arch: spec.arch, mma: Mma::Dense,
    })
}
```

### `iu4_gemm` phase blocks

`Iu4Gen::new(Iu4Spec { .. })?` builds the generator of one valid spec (`Spec::validate`, gfx1201 only): the register assignment and geometry every block reads. It is an opaque value with read-only accessors.

| call | meaning |
|---|---|
| `gen.plan()?` | the kernel's `RegPlan` (a script may add ranges) |
| `gen.payload_base(role, slot)?` | `u32`, the LDS byte offset of payload slab `slot` of the ring `role` (`PayloadRole::A` or `W`) names; refuses a slot the ring depth lacks (`payload_base(A, 2): the payload rings are 2 deep (slots 0..2)`, R0000) |
| `gen.kernel_spec()` | the kernel's `KernelSpec` (symbol, kernargs, workgroup size) |
| `gen.spec()` | the `Iu4Spec` |
| `gen.reg("name")` | `Option<u8>`, the base of a physical register by name: `cacc acc magic d f0 f1 ds_nx sz_nx fr_a0 fr_a1 fr_w0 fr_w1 st_lds st_a meta_ds meta_sz ds_voff sz_voff sc_addr d_addr`, `a_pf<i> w_pf<i> st_w<i>`, SGPRs `karg srd_w srd_z srd_a0 srd_a1 srd_y step2 goff trips rs bs wave gpr136 mm1 hs tmp epi_s a_slab1`, kernel arguments `arg_a arg_u arg_gate arg_xq arg_y arg_m arg_k arg_n arg_bcx arg_bcy`; `None` for an unknown name or an absent optional register |

The blocks run inside a `Workgroup::new` scope. `lds` is the array of the four LDS rings (A, W, DS, SZ); `end` is a kernel exit from `wg.exit(..)`.

| call | Rust (`iu4_gemm::blocks`) | rings |
|---|---|---|
| `iu4::declare_lds(wg, gen)?` | `declare_lds` | declares them, every buffer `Free`; returns `lds` |
| `iu4::prologue(wg, gen, end, lds)?` | `prologue` | `Free`/`Free` in, `Published`/`Free` out (the block boundary); returns `lds` |
| `iu4::kloop(wg, gen, lds)?` | `kloop` | the whole rolled K-loop with its peeled group; returns `lds` |
| `iu4::block(wg, gen, tag, h, fetch_next, end_label, entry_label, lds)?` | `block` | one K128 block on plane `h` (0 or 1); `tag` names its labels, `end_label` closes its register lifetimes, `entry_label = false` leaves the block's own label to a `wg_skip_if` target; returns `lds` |
| `iu4::block_stepping(.., entry_label, step, lds)?` | `block_stepping` | `block` that also advances descriptor `step` (0 or 1) right after its slab-1 fetch: the production odd block does this for descriptor 0, inside one block, so it cannot sit between two `block` calls |
| `iu4::sparse_block(wg, gen, tag, h, fetch_next, end_label, entry_label, lds)?` | `sparse_block` | `block` for a sparse spec (`gen.spec().mma` is `S24X64` or `S24X32`): same arguments, ring transitions, publication and fold, SWMMAC in place of WMMA; refuses a dense spec, and `iu4::block` refuses a sparse one |
| `iu4::sparse_block_stepping(.., entry_label, step, lds)?` | `sparse_block_stepping` | `block_stepping` for a sparse spec |
| `iu4::step_descriptor(wg, gen, which)?` | `step_descriptor` | `s_add_nc_u64` of the group stride onto descriptor 0 or 1 |
| `iu4::trip_end(wg, gen)?` | `trip_end` | odd descriptor, group offset and trip counter advance |
| `iu4::trips_cmp(wg, gen, "s_cmp_lg_u32")?` | `trips_cmp` | a `WgCond` for `wg_skip_if` / `loop_carried` (`s_cmp_{eq,lg,lt,le,gt,ge}_{u32,i32}` only) |
| `iu4::epilogue(wg, gen, end)?` | `epilogue` | the store path; places `iu4::EPI` |
| `iu4::gdn_epilogue(wg, gen, lds)?` | `gdn_epilogue` | the fused projection's GDN preparation; consumes `lds` (`Epi::QkvzaGdn` only) |
| `iu4::label(tag)`, `iu4::tail_label(h)` | `label`, `tail_label` | `.Liu4_<tag>`, `.Liu4_t<h>` |
| `iu4::ENTRY K_BEGIN K_LOOP K_LOOP_END EPI END GDN` | the label constants | strings |

Each block is one checked `Driver::typed` call. The native takes the four rings from the driver, runs the Rust block on the typed `Workgroup`, and gives the rings back to the same driver slots, so a block cannot read a `Free` buffer, and a script loop (`loop_carried`) or branch (`wg_skip_if`) around blocks is checked by the driver's ordinary rule that a body leaves every ring in the state it found it. What the bridge refuses:

- a ring of the wrong kind (the W ring where the A ring is expected: the tag differs), a ring in another phase, a ring with an undrained store, a ring the script made with `Ring::new` (only rings a typed block gave can be lent);
- any typed call under wave-uniform control (`skip_if`, `exec_if`, `if_else`, `forward`, `loop_until`), after a wave-uniform kernel exit, or on a target other than gfx1201 (the blocks are gfx1201-typed);
- a typed body that leaves a taken ring out. After any `Err` from a typed call the lowering is dead, like any backend `Err`: stop at the first one.

A block's fold stage is two natives (`iu4::wmma(.., fold, ..)` interleaves stage 1 between the sub-block-1 WMMAs; `iu4::fold_scale` is stage 2): `iu4::block` is their composition with the fetches, publishes, waits and rendezvous in the generator's order, and the next section splits it.

The committed twin [`iu4_gemm.rip`](../../peacemaker-isa/src/kernels/iu4_gemm.rip) composes `declare_lds`, `prologue`, a K-loop spelled out in script (`wg_skip_if` + `loop_carried` + `block` + `block_stepping` + `trip_end`, then the peeled pair) and `epilogue` / `gdn_epilogue`. Its variants are the shipped variant names (`k128-128x128x8-cacc1-slab-silu-bf16`; the grammar is in the file header). The sparse variants carry a trailing `-s24x64` / `-s24x32` (`k128-128x128x8-cacc1-set-s24x64`) and take the sparse blocks. A test (`tests/iu4_gemm_twin.rs`) requires its output to be byte-identical to `iu4_gemm::emit` for every valid spec, sparse included; [cookbook 11](cookbook/11-production-blocks.md) shows how to vary it.

### `iu4_gemm` fine K-loop natives

`iu4::block` and `iu4::sparse_block` are themselves compositions of finer typed steps, and a script can write the block with them ([cookbook 17](cookbook/17-iu4-kloop-natives.md); the committed [`iu4_gemm_fine.rip`](../../peacemaker-isa/src/kernels/iu4_gemm_fine.rip) writes the dense block (`fine_block`) and the 2:4-sparse blocks (`fine_sparse_block`) this way, the dense fragment loads as the script body `rip_fragments` over `iu4::fragments_body`, and emits the default-tune bytes for every variant; `WAIT = "window"` writes the 128x128x8 tile's rendezvous-window block the shipped 27B gate/up `_b1s` export runs). So the order of fetches, publishes, waits and WMMA groups is a script choice. The natives are the `iu4_gemm::blocks` functions `kloop::block_with` / `kloop::sparse_block_with` are composed of (`iu4::fragments_body` is the exception: it replays a script-built body); each takes `wg` and the generator first. Ring states are `[current, next]` per ring: `Published`/`Free` at a block boundary.

| call | rings in -> out (A W DS SZ) | meaning |
|---|---|---|
| `iu4::block_begin(wg, gen, tag, h, end_label, entry_label, lds)?` | none changed | places the block's label (`entry_label`) and declares its register lifetimes (fragments, the fold's scale rows and products; `.Liu4_<tag>_s0` / `_s1` and `end_label` close them); refuses a plane the rings are not on (A current slot 0, DS current plane `h`) and a sparse spec; returns `lds` |
| `iu4::fetch_slab1(wg, gen, h)?` | none | the slab-1 loads of block `h` into the prefetch registers |
| `iu4::fetch_slab1_next_group(wg, gen)?` | none | the odd block's form: the next group's even block, before the group offset advances |
| `iu4::fetch_next(wg, gen, h)?` | none | the next block's slab 0 into the prefetch registers |
| `iu4::fetch_next_meta(wg, gen, h)?` | none | the next block's token scale and packed row header |
| `iu4::publish_slab(wg, gen, lds)?` | A, W: `Published`/`Free` -> `Published`/`Writing`, pending | the weight sign flip and the LDS stores of the staged slab into the next buffers |
| `iu4::publish_meta(wg, gen, lds)?` | DS, SZ: `Published`/`Free` -> `Published`/`Writing`, pending | the next block's token scale and converted row scale |
| `iu4::wait(wg, lds, set)?` | `Writing`, pending -> `Writing`, drained | the counter wait for `"slab"` (A, W), `"meta"` (DS, SZ) or `"all"`; returns the tuple of `Drained` tokens in ring order |
| `iu4::fragments_body(wg, gen, lds, loads)?` | A, W current `Published` (next both `Free` or both `Writing`, token preserved; depth 2 or 3), unchanged | replays the script's `loads` (a `Vec<NativeLoad>`) in order, each through the typed `ds_load_cur` of the ring its role names (dense spec; a sparse spec is refused) |
| `iu4::d_loads(wg, gen, lds)?` | DS current `Published`, unchanged | phase B's token-scale loads |
| `iu4::scale_loads(wg, gen, tag, lds)?` | SZ current `Published`, unchanged | places `.Liu4_<tag>_s0`, then the first three row-scale loads (dense spec) |
| `iu4::scale_tail(wg, gen, tag, lds)?` | SZ current `Published`, unchanged | places `.Liu4_<tag>_s1`, then the last row-scale load (dense spec) |
| `iu4::wmma(wg, gen, sb, first, fold, lds)?` | A, W current `Published`, unchanged | the eight WMMAs of 32-K sub-block `sb` (0 or 1); `first` seeds the accumulators, `fold` interleaves the fold's stage 1 (dense spec) |
| `iu4::fold_scale(wg, gen)?` | none | the fold's stage 2 |
| `iu4::publish_slab_set(wg, gen, set, lds)?` | as `publish_slab`; the current buffers may also be `Free` (a ring still filling) | `publish_slab` of prefetch set `set` (ring depth 3: set 0 holds the next block's slab 0, set 1 its slab 1) |
| `iu4::fetch_ahead(wg, gen, h, slab)?` | none | ring depth 3: slab `slab` of the block one group on, plane `h`, into prefetch set `slab` |
| `iu4::fetch_slab(wg, gen, set, slab, desc, woff)?` | none | slab `slab` of the block whose activation descriptor is `srd_a[desc]` and weight offset `woff` (from the group offset), into prefetch set `set` |
| `iu4::setup(wg, gen, end)?` | none | ring depth 3: the prologue without its ring part (arguments .. block 0's three fetches, accumulators cleared) |
| `iu4::split_trips(wg, gen)?`, `iu4::ring_cmp(wg, gen, v)?` | none | ring depth 3: `n = K/256 - 1` into `q = n / 3` loop trips and `r = n mod 3` leading groups; the `WgCond` `r != v` for `wg.wg_if_else` |
| `iu4::group_end(wg, gen, count)?` | none | ring depth 3: a group's trip end (the group offset advances; `count` also drops the trip counter) |
| `iu4::sparse_block_begin(wg, gen, tag, h, end_label, entry_label, lds)?` | none changed | `block_begin` of the 2:4-sparse K-loop: the label and the register lifetimes of the spec's SWMMAC form (x64: fragments end at `_s0`; x32: the W and index registers end at `_s1`); sparse spec only |
| `iu4::sparse_fragments(wg, gen, lds)?` | A, W current `Published`, unchanged | `s24x64`: the W loads and the eight `ds_load_b64` B operands of the slab; `s24x32`: the dense fragment loads |
| `iu4::sparse_wmma(wg, gen, sb, first, fold, lds)?` | A, W current `Published`, unchanged | the eight SWMMACs in bundle order: `s24x32` sub-block `sb` (0 or 1), `s24x64` the one K64 bundle (`sb = 0`; 1 is refused); `first` seeds each accumulator with `magic8`, `fold` interleaves the fold's stage 1 |
| `iu4::sparse_scale_loads(wg, gen, tag, lds)?` | SZ current `Published`, unchanged | places `.Liu4_<tag>_s0`, then the row-scale loads: `s24x64` all four (after its one bundle), `s24x32` the first two (between its sub-blocks) |
| `iu4::sparse_scale_tail(wg, gen, tag, lds)?` | SZ current `Published`, unchanged | `s24x32` only: places `.Liu4_<tag>_s1`, then the last two row-scale loads; `s24x64` has no `_s1` and is refused |

`iu4::step_descriptor(wg, gen, which)?` (above) is the descriptor step. The publishes, waits, fetches, `d_loads` and `fold_scale` do not depend on the matrix instruction and serve both K-loops; the `fragments_body` / `wmma` / `scale_*` / `block_begin` steps are paired with their `iu4::sparse_*` twins, and a dense step on a sparse spec (or the reverse) is refused before anything is emitted. The rendezvous is the generic driver's: `wg.barrier((rotate(lds[0], da), rotate(lds[1], dw)))?` rotates the drained slab (retires the current buffer, publishes the next one), `wg.signal((rotate(..), ..))?` / `wg.wait_arrived(arrived)?` split it. `fetch_*`, `fold_scale`, `wmma`, `sparse_wmma` and `step_descriptor` emit through the builder, so they may sit between `signal` and `wait_arrived`, where a `typed` call is refused.

`PayloadRole::{A, W}` and `NativeLoad { role, insn }` are built-in (no declaration). `fragments_body` lends the A and W rings in their state, replays each load through the ring its role names (the role maps to the sealed `PayA` / `PayW` tag; a script cannot make a tagged ring) and gives them back. Its capability is a read of the payload rings: every `insn` must be an LDS load (a `ds_load*` mnemonic with `MemoryClass::DsLoad`) whose declared defs and uses cover the decoded operands (R0603), and every destination register needs a unique live owner in the plan (R0401). All refusals come before anything is emitted. The committed twin's `rip_fragments` is the dense body (six loads per block, built from `gen.payload_base`, `gen.reg("f0")` etc.): edit it, no rebuild.

What the driver refuses, each a build error with a stable code ([ERRORS.md](ERRORS.md)):

| rule | refusal | code | test |
|---|---|---|---|
| publish into a buffer that is not `Free` | `iu4::publish_slab: ring .. next buffer is Writing ..` | R0399 | `publish_into_a_non_free_buffer_is_refused` |
| rotate before the wait | `barrier would publish .. with an undrained LDS store` | R0301 | `rotating_before_the_wait_is_refused` |
| wait with no store in flight | `wait on ring ..: no store of this wave is in flight` | R0399 | `waiting_without_a_publish_is_refused` |
| read or WMMA of an unpublished slot | `iu4::fragments_body: ring .. current buffer is Free, not published` (via `rip_fragments`) | R0303 | `reading_or_wmma_of_an_unpublished_slot_is_refused` |
| a ring of the wrong kind / one the script made | the typed lend's tag check | R0399 | `a_ring_of_the_wrong_kind_is_refused_by_the_fine_natives`, `a_ring_the_script_made_cannot_enter_the_fine_natives` |
| a fine step under wave-uniform control | `.. is workgroup scope` | R0304 | `a_fine_step_under_wave_uniform_control_is_refused` |
| a loop trip that leaves a publish in flight | the driver's loop rule | R0399 | `a_trip_that_leaves_a_publish_in_flight_is_refused` |
| a dense step on a sparse spec, a sparse step on a dense one | `iu4::fragments_body: the spec is s24x64: a dense step cannot run in the sparse K-loop ..` / `iu4::sparse_fragments: the spec is dense ..` | R0000 | `a_fine_step_of_the_other_mma_family_is_refused` |
| a second sub-block (or `_s1` scale tail) of an `s24x64` block | `sparse wmma group: sub-block 1: .. covers the whole K64 slab` | R0000 | `an_x64_block_has_no_second_sub_block` |
| a sparse read or SWMMAC group of an unpublished slot | `iu4::sparse_fragments: ring .. not published` | R0303 | `a_sparse_read_or_wmma_of_an_unpublished_slot_is_refused` |
| payload rings in different states | `iu4::fragments_body: .. different states ..` | R0000 | `the_script_fragments_body_keeps_the_typed_refusals` |
| a non-load in a body | `iu4::fragments_body: load i .. is not an LDS load: a fragments body only reads the payload rings` | R0399 | `a_fragments_body_outside_its_capability_is_refused` |
| a load whose defs/uses miss a decoded operand | `iu4::fragments_body: load i: op declares defs ..` | R0603 | `a_fragments_body_outside_its_capability_is_refused` |
| a destination register with no unique live owner | `v[240:243] at .Liu4_m0 has no unique live owner` | R0401 | `a_fragments_body_outside_its_capability_is_refused` |

(The tests are in `tests/iu4_gemm_fine_twin.rs`; `the_fine_sparse_composition_emits_the_shipped_bytes` requires `fine_sparse_block` to emit the bytes and proof of `iu4_gemm::emit` for every sparse variant, and `the_fine_composition_emits_the_shipped_bytes` the dense ones.) The opaque `iu4::sparse_block` / `iu4::sparse_block_stepping` remain and call the same composition. A sparse spec admits the shipped wait placement only (`WaitPlace::AfterMma`) and ring depth 2 only. The payload rings are two deep by default; `const RING = 3` in the twin builds the dense depth-3 K-loop from these natives plus the typed ring's `starting_at`, `advance` and `wg.wg_if_else` ([cookbook 17](cookbook/17-iu4-kloop-natives.md#ring-depth-3)).

### 2:4-sparse weights (`Mma::S24X64`, `Mma::S24X32`)

The weights are WMMA operand A, so they are the sparse side of SWMMAC. The staging (prologue, publish, the LDS layout, the `0x88888888` sign flip), the fold and the epilogue only move bytes or consume accumulators, so a sparse K128 row block lives in the dense 64 code bytes: `iu4_gemm::sparse24::pack_block(mma, codes, kept)` puts the compressed codes and the index words where the dense staging routes them into the registers the SWMMAC reads. The format saves no weight bytes; it measures SWMMAC's MMA-issue saving inside the production pipeline. A 2:4 group is four aligned consecutive natural K (`group_of(_, k) = k / 4`, 32 per K128 block); exactly two K per group are kept and every pruned code is 8 (zero). `S24X64` issues one `v_swmmac_i32_16x16x64_iu4` per accumulator per 64-K slab (the activation fragments of both sub-blocks are one four-VGPR B operand); `S24X32` one `v_swmmac_i32_16x16x32_iu4` per sub-block with `index_key` 0 / 1. SWMMAC has no C operand, so each accumulator is first seeded with `magic8` (`v_mov_b32`) where dense passes it as C. Valid only with `Fold::K128`, `Tile::T128x128x8`, `Cacc::One`, `ALayout::Token` and `Epi::Set | Epi::Add`; anything else is refused with the axis named. The correctness oracle is an emulator run of the whole kernel against the dense kernel on the pruned dense weights (`peacemaker-isa/tests/iu4_gemm_sparse_emu.rs`): bitwise equal.

## 14. Production generator with fine blocks: `mq6_i8_gemm`

The MQ6G256V2 x `block_i8_128` exact-INT8 GEMM (`peacemaker_isa::kernels::mq6_i8_gemm`, gfx1201) is composable from `.rip` at a finer grain than `iu4_gemm`: its natives own no LDS ring. The split:

- **LDS ring operations are script statements.** The script makes the eight LDS slots and the four two-buffer rings itself (`wg.lds`, `Ring::new`), and issues every `wg.begin_write`, `wg.ds_store`, `wg.ds_load_cur`, `wg.wait_all`, `wg.signal(rotate(..))` / `wg.wait_arrived`, `loop_carried` and skip. The checked driver refuses a wrong one (a read of a buffer that is not published, a rotate over an undrained store, a signalled barrier never waited, a loop body that leaves the rings in another state). No `Driver::typed`, no `Lend`: the typed core stays sealed.
- **Natives that touch LDS return `Instruction`s** (`mq6::store_w_insns`, `mq6::step_load_insns`, ...): the script hands them to `wg.ds_store` / `wg.ds_load_cur`.
- **Every other native emits straight into `wg.isa()`**: fetches, decode, WMMA steps, the fold, the seed fill, the prologue's scalar and lane set-up, the epilogue.

Geometry (128 rows x 128 tokens x 8 waves), the LDS layout, the register map and the two buffers per ring are a spec axis of the Rust generator and need a rebuild to change.

### Coarse natives and the generator

| call | meaning |
|---|---|
| `Mq6Gemm::emit(Mq6Spec { epi, raster, tile, arch[, ring] })?` | what `mq6_i8_gemm::emit` returns (`Emitted`); `Spec::validate` stays the gate (gfx1201 only) |
| `Mq6Gemm::emit_module(arch)?` | `(Vec<Emitted>, text, proof_json)` of the product module (both epilogue symbols, banded raster) |
| `Mq6Epi::{Set, SetBf16}`, `Raster::{Banded, Linear}`, `Mq6Tile::{T128x128, T256x128, T128x256}` | the spec enums (script enum values). `Linear` is the CPU-interpreter twin's raster. The tile is the CTA's weight rows x tokens: `T128x128` is the shipped 8-wave tile (256 threads, 36864 B LDS); the other two run 16 waves (512 threads, 55296 B LDS) of the same 32 x 64 per-wave tile, so they stay at the 192-VGPR ceiling |
| `Mq6Gen::new(Mq6Spec { .. })?` | the generator of one valid spec: an opaque value with read-only accessors |
| `gen.plan()?`, `gen.kernel_spec()`, `gen.spec()` | the `RegPlan`, the `KernelSpec`, the `Mq6Spec` |
| `gen.depth()`, `gen.pf_set(0 or 1)?` | the A/W ring depth (2 or 3); the generator with raw prefetch set 0 / 1 in the `a_pf` / `w_pf` slots (set 1 exists at depth 3) |
| `gen.a_rounds()`, `gen.w_rounds()` | staging rounds per 64-K slab of the tile's activations / weights: 2 / 2 on `T128x128`, 1 / 2 on `T256x128`, 2 / 1 on `T128x256`. A tile with one weight round has no second decode / store; `gen.reg("a_pf1")` / `reg("w_pf1")` are `None` for the round it lacks |
| `gen.reg("name")` | `Option<u8>`, a physical register base by name: VGPRs `cacc acc f0 f1 tok msn a_pf0 a_pf1 w_pf0 w_pf1 ds_nx sz_nx fr_a fr_w st_alds st_wlds st_a st_w meta_lds ds_voff sz_voff sc_addr d_addr scratch fold_sc fold_zp fold_t`, SGPRs `karg srd_w srd_a0 srd_a1 srd_y step2 goff goff1 trips rs bs wave gpr200 omask tmp epi_s`, kernel arguments `arg_a arg_xq arg_y arg_m arg_k arg_n arg_bcx arg_bcy` |
| `gen.slot("A0")` | `Option<(base, bytes)>` of an LDS slot: `A0 W0 A1 W1 DS0 DS1 SZ0 SZ1` (and `A2 W2` at depth 3) |
| `mq6::ring3_split`, `mq6::ring3_cmp_insn`, `mq6::group_end`, `mq6::trips_dec` | the depth-3 K split (`r` to `tmp`, `q` to the trip counter), its arm-selector compare, the group-offset move and the trip decrement |
| `mq6::ENTRY K_BEGIN K_LOOP K_LOOP_END EPI END ARM12 ARM12_END ARM2 K_ARMS_END`, `mq6::GROUP_BYTES` | the label constants (strings) and the MQ6G256V2 group size (200) |
| `mq6::label(tag)`, `mq6::tail_label(h)` | `.Lmq6i8_<tag>`, `.Lmq6i8_t<h>` |

### Blocks

`wg` is the workgroup scope of a `Workgroup::new` let initializer; `g` an `Mq6Gen`; `h` the K128 plane parity (0 or 1). Indices are checked: an out-of-range step, set, round, slot or plane is a refusal naming the native.

| call | what it does |
|---|---|
| `mq6::prologue_front(wg, g)?` | kernel arguments (`s_load_b256` x2) and the CTA raster: the tile origin `rs`, `bs` |
| `mq6::tile_outside_insn(g, which)?` | `Instruction`: `0` rows start at/past `M`, `1` tokens at/past `N`; consume with `wg.scmp_wg_uniform` and `wg.exit_if` |
| `mq6::prologue_scalars(wg, g)?`, `mq6::prologue_lanes(wg, g)?` | scalars and buffer descriptors; hoisted lane offsets and fold addresses |
| `mq6::prologue_fetch(wg, g)?`, `mq6::zero_acc(wg, g)?` | block 0's raw fetches; zero the float accumulators |
| `mq6::fetch_a(wg, g, slab, sb, srd)?` | raw activation chunks of K32 half `sb` of slab `slab` of the block whose descriptor SGPR is `srd` (`g.reg("srd_a0")` / `srd_a1`) |
| `mq6::fetch_w(wg, g, base)?` | raw weight dwords (both rounds) at group-relative byte offset `base` |
| `mq6::fetch_meta(wg, g, h, hdr)?` | the next block's token word and packed row header |
| `mq6::wait_all(wg)?` | the builder's ledger-wide wait (the staging registers are still sources of the previous stores until it) |
| `mq6::decode_insns(g, r)?` | `Vec<Instruction>`: weight round `r`'s (`0..gen.w_rounds()`) decode (VALU), for the script to push, spread between WMMAs |
| `mq6::store_w_insns(g, slot, r)?`, `mq6::store_a_insns(g, slot, sb)?` | `[Instruction; 2]` LDS stores of a decoded weight round / `Vec<Instruction>` (one per activation round) of a raw activation half into slot `slot`; issue with `wg.ds_store(ring, insn)` on the W / A ring |
| `mq6::meta_valu(wg, g)?`, `mq6::meta_store_insns(g, plane)?` | the metadata VALU words (before the DS / SZ rings begin their write), then `[Instruction; 2]` stores into plane `plane` (the DS ring's `next_index()`) |
| `mq6::mma(wg, g, step, set, first, i)?` | the `i`-th (0..8) `v_wmma_i32_16x16x16_iu8` of K16 step `step` (0..4) on fragment set `set`; `first` seeds from the decode scratch's `0x4B400000` octet |
| `mq6::step_load_insns(g, slot, step, set)?` | `[Instruction; 3]`: the weight pair (W ring) and two token pairs (A ring) `ds_load_b128`s of a K16 step; issue with `wg.ds_load_cur` |
| `mq6::seed_fill(wg, g)?` | refill the scratch with the next block's WMMA seed |
| `mq6::fold_addr(wg, g, h)?`, `mq6::fold_token_load_insn(g, nb)?`, `mq6::fold_sz_load_insn(g, h, dst, rel)?` | fold addresses and the DS-ring / SZ-ring loads (`Instruction`s) |
| `mq6::emit_msn(wg, g, nb)?` | `msn = -(12582912.0 + 32 s)` for token block `nb` |
| `mq6::fold_stage(wg, g, k, stage, sc, zp)?`, `mq6::fold_fragment(wg, g, k, sc, zp)?` | fragment `k = nb * 2 + rg`'s VOPD fold: stage `0..4` (8 `mul`/`add` packets, then 4 `fmac`, 4 `mul`, 4 `fmac` pairs), or all four; `sc` / `zp` the base registers of the `sc` / `z'` octets |
| `mq6::step_descriptor(wg, g, which)?`, `mq6::trip_end(wg, g)?` | the K-group descriptor step; the odd descriptor, group offsets and trip counter advance |
| `mq6::trips_cmp_insn(g, "s_cmp_lg_u32")?` | `Instruction` compare of the trip counter with zero (`s_cmp_{eq,lg,lt,le,gt,ge}_{u32,i32}` only); consume with `wg.scmp_wg_uniform` |
| `mq6::declare_block_regs(wg, g, entry, fold, end)?` | the register lifetimes of one block: fragment sets from label `entry` to `fold`, fold registers from `fold` to `end` |
| `mq6::epilogue(wg, g)?` | everything after the `mq6::EPI` label (the script places it with `wg.label`): output descriptor, lane masks, masked stores |

The committed twin [`mq6_i8_gemm.rip`](../../peacemaker-isa/src/kernels/mq6_i8_gemm.rip) writes the whole composition: ring declaration, the prologue's staging of block 0, the K-loop (`wg_skip_if` + `loop_carried` + `block` + trip end, then the peeled pair) and the block body (queued decode / store / fetch work interleaved between WMMAs, B1 and B2 split barriers, the fold), then the epilogue. Its variants are the shipped variant names (`128x128x8-set`, `256x128x16-set_bf16-linear`, ...: `TILE-EPI[-linear]`, twelve of them; the grammar is in the file header). `tests/mq6_i8_gemm_twin.rs` requires its `s_text` and proof to be identical to `mq6_i8_gemm::emit` for every valid spec, and `peacemaker-isa/tests/mq6_i8_gemm_digests.rs` pins `emit` itself to the digests recorded before the blocks were factored out. [Cookbook 12](cookbook/12-mq6-blocks.md) shows what to move.

## 15. Production generator on the raw `Builder`: `fp8_gemm`

The MQ4V2 x E4M3 ratio-fold prefill GEMM (`peacemaker_isa::kernels::fp8_gemm`, gfx1201, `v_wmma_f32_16x16x16_fp8_fp8`) is composable from `.rip`, but it is a different kind of kernel from §13 and §14: the shipped generator is an untyped-`Builder` kernel (raw `barrier_with_scope`, `loop_`, `control` branches, a jump-network epilogue), not a typed-core one. Its twin is therefore written on the raw `Builder` surface that `.rip` already exposes (`Builder::new`, `lds_slot`, `ds_store`, `ds_load`, `barrier_with_scope`, `barrier_signal`, `barrier_wait`, `loop_`, `control`, `wait`, `ready` / `retire`), not on `Workgroup`. That is a deliberate boundary: there is no typestate and no sealed ring here, and every checked rule is the Builder's (the LDS slot machine, the wait ledger, the hazard guards, the loop fixpoint). Nothing is an escape hatch: a store into a slot that is Reading, a read of a slot that is not Published, a barrier over an undrained store, a loop whose back edge differs from its entry are all refusals (`tests/fp8_gemm_twin.rs`).

- **LDS operations are script statements.** The script declares the six slots with `b.lds_slot` and issues every `b.ds_store(slot, insn)` / `b.ds_load(slot, insn)` and every barrier.
- **Natives that touch LDS return `Instruction`s** (`fp8::publish_a_insns`, `fp8::fragment_insns`, `fp8::ratio_store_insn`, `fp8::d_store_insn`, the `fp8::fold_*_load_insns`).
- **Every other native emits into the builder passed as its first argument** (`b`, from `Builder::new`): fetches, address set-up, the WMMAs, the fold arithmetic, the loop bookkeeping.
- **`fp8::epilogue` is one coarse native.** The epilogue is a jump network, and the GDN-fused QKVZA epilogue carries its own LDS ring and loops; both are composed in Rust and the script places them after the K-loop.

Geometry (256 rows x 128 tokens x 8 waves), the six LDS slots (`A0 A1 R0 R1 D0 D1`), and the register map are fixed by the Rust generator; changing them needs a rebuild.

### Coarse natives and the generator

| call | meaning |
|---|---|
| `Fp8Gemm::emit(Fp8Spec { arch, scale, epi })?` | what `fp8_gemm::emit` returns (`Emitted`); `Spec::validate` stays the gate (gfx1201 only; `silu-bf16` and `qkvzagdn` are row-scaled only) |
| `Fp8Gemm::emit_module(arch, scale, [epis])?` | `(Vec<Emitted>, text, proof_json)` of the module `emit --kernel fp8_gemm --scale S --epi ...` writes |
| `Fp8Scale::{Row, K128}`, `Fp8Epi::{Set, Add, GateUpSilu, GateUpSiluBf16, Qkv, Qkvza, QkvzaGdn}` | the spec enums (script enum values) |
| `Fp8Spec { arch, scale, epi }` | the struct literal (the Rust `act_scale` is `scale` in script) |
| `Fp8Gen::new(Fp8Spec { .. })?` | the generator of one valid spec: an opaque value |
| `g.plan()?`, `g.kernel_spec()`, `g.spec()` | the `RegPlan`, the `KernelSpec`, the `Fp8Spec` |
| `g.slot("A0")` | `Option<(base, bytes)>` of an LDS slot: `A0 A1 R0 R1 D0 D1` (`D*` are K128 only) |
| `fp8::BEGIN LOOP LOOP_END EPILOGUE END` | the label constants (strings) |
| `fp8::label(b, name)?` | place a label |

### Blocks

`b` is the `Builder` of `Builder::new(g.kernel_spec(), g.plan()?)`; `g` an `Fp8Gen`. Indices are checked: an out-of-range slot, unit, step, plane or WMMA index is a refusal naming the native, and the K128-only natives refuse a row spec (`only the K128-scaled kernel has a D ratio`).

| call | what it does |
|---|---|
| `fp8::prologue_front(b, g)?` | kernel arguments and the banded CTA raster; leaves the tile-outside compare for the script's `b.control` branch to `fp8::END` |
| `fp8::prologue_lanes(b, g)?` | buffer descriptors and the hoisted lane addresses |
| `fp8::fetch_a(b, g, kb_offset, slab)?` | the four raw `buffer_load_b64` of A slab `slab` of the block `kb_offset` K128 blocks ahead |
| `fp8::publish_a_insns(g, slot)?` | `Vec<Instruction>`: the LDS stores that publish the fetched slab into slot `slot`; issue with `b.ds_store(slot, insn)` |
| `fp8::weight_base(b, g)?`, `fp8::fetch_w(b, g, unit)?` | the weight base and the unit's four `buffer_load_b128`s (no W is staged in LDS) |
| `fp8::fetch_next_ratio(b, g, kb)?` | the next block's row-ratio loads |
| `fp8::ratio_store_insn(g, plane)?` | `Instruction`: the ratio store into plane `plane` (slot `2 + plane`) |
| `fp8::d_prepare(b, g, kb)?`, `fp8::d_store_insn(g, plane)?`, `fp8::d_restore(b, g)?` | K128 only: the next block's per-token D ratio, its store into slot `4 + plane`, the exec-mask restore |
| `fp8::fragment_insns(g, slot, step)?` | `Vec<Instruction>`: the four A fragment `ds_load_b128`s of K16 step `step` (0..4) from slot `slot`; issue with `b.ds_load(slot, insn)` |
| `fp8::mma(b, g, unit, step, first, i)?`, `fp8::mma_insn(...)` | the `i`-th (0..16) WMMA of K16 step `step` of unit `unit` (0..4); `first` starts the accumulators from the WMMA's inline zero. `mma_insn` returns the `Instruction` |
| `fp8::fold_row_group_load_insns(g, plane, i)?`, `fp8::fold_row_group_scale(b, g, i)?` | Row fold: the two `ds_load_b128` ratio loads of group `i`, then its 16 packets |
| `fp8::fold_k128_token_load_insns(g, plane)?`, `fp8::fold_k128_pair_load_insns(g, plane, i, e)?`, `fp8::fold_k128_pair_scale(b, g, i, e)?` | K128 fold: the four per-token D loads, a pair's two ratio loads, its four token chains |
| `fp8::advance_block(b, g)?`, `fp8::trips_setup(b, g)?`, `fp8::trip_end(b, g)?` | block offset advance; trip count; the loop's counter advance |
| `fp8::wait_all(b)?` | the builder's ledger-wide wait |
| `fp8::epilogue(b, g)?` | everything after the `fp8::EPILOGUE` label: the output jump network (and the GDN preparation for `qkvzagdn`) |

The committed twin [`fp8_gemm.rip`](../../peacemaker-isa/src/kernels/fp8_gemm.rip) writes the whole composition, down to the barrier lists: `declare_lds`, `prologue`, `kloop` (first block, loop of two blocks per trip, peeled last block), `block` (A1 fetched before slab 0, a uniform B1 barrier that retires A0 and publishes the next ratio planes, the next A0 payload, the fold, a split B2 barrier), `fold`, then `fp8::epilogue`. `fp8_gemm::emit` is written over the same blocks and is pinned to the digests recorded before they existed (`peacemaker-isa/tests/fp8_gemm_digests.rs`). It re-emits the bytes and the proof of every one of the twelve valid specs (`tests/fp8_gemm_twin.rs`). `pm_loop --kernel fp8_gemm --rip fp8_gemm.rip` times it ([cookbook 13](cookbook/13-fp8-blocks.md)).

## 16. Production generator on builder-level blocks: `qwen4_moe_sym` (`MoeNtGen`, `MoeNt1Gen`, `moent::*`, `moe1::*`)

The gfx1201 MQ4 entries of `qwen4_moe_sym`, composed in script like `mq6_i8_gemm` (section 14): `gate_up` (SwiGLU) and `down` at NT1, NT4 and NT8, and the grouped-position down entries `down_gp_nt4|nt8`. The committed twin is `peacemaker-isa/src/kernels/qwen4_moe_sym.rip` (variants `gate_up`, `gate_up_nt4`, `gate_up_nt8`, `down`, `down_nt4`, `down_nt8`, `down_gp_nt4`, `down_gp_nt8`: the pm_loop `--epi` names); `tests/qwen4_moe_sym_twin.rs` requires `s_text` and proof identity with `qwen4_moe_sym::emit` / `emit_down_gp` for all eight. `MoeSym::emit(..)` (section 13) is the coarse native. The Rust generators are written over the same blocks (`qwen4_moe_sym::blocks` for NT4/NT8, `qwen4_moe_sym::nt1_blocks` for NT1) and pinned to the digests recorded before the blocks existed (`peacemaker-isa/tests/qwen4_moe_sym_digests.rs`: every entry, the grouped-position entries, the module texts and the `R4Spec` experiments), so a change to a block moves both the generator and the twin.

Nothing here lends a typed region or opens a typed scope: **every block is a straight run of instructions** emitted into `wg.isa()`, every compare is a native returning an `Instruction` (turn it into a condition with `wg.scmp` / `wg.scmp_wg_uniform`), and the LDS accesses of the `gate_up` hand-off are natives returning `Instruction`s (`moent::publish_insns`, `moent::gate_load_insns`) that the script issues through `wg.ds_store` / `wg.ds_load` on the region it declared (`wg.lds::<Gate>("swiglu_gate", 0, g.lds_bytes())`). The script therefore owns: the exits (a tile past the end, a sentinel expert, a non-leader tile), the run walk's `loop_until`, the per-tile liveness branches of a fold and **where each wait sits** (`moent::wait_tile`; delete it and the build is refused: the next tile's loads are issued with the previous still pending), the order of loads and folds in a K-loop trip and across the tail, the descriptor steps, the trip counter, the odd/even tail, the per-tile output exits, and the hand-off (the gate waves' stores, the up waves' loads, `wg.handoff`). Tile geometry (`nt`, the register map, the fold math, the K-loop's prefetch schedule) is a spec axis of the Rust generator and needs a rebuild; the chains / no-prefetch / grouped axes of the `R4Spec` experiments stay Rust-only.

| call | meaning |
|---|---|
| `MoeNtGen::new(kind, nt, gp, arch)?` | one NT4/NT8 entry (`kind` is `"gate_up"` or `"down"`; `gp` only for `down`; the Rust `emit` / `emit_down_gp` refusals). `g.plan()?`, `g.kernel_spec()`, `g.nt()`, `g.gp()`, `g.gate_up()`, `g.lds_bytes()` (the hand-off region, 0 for down), `g.label(name)`, `g.exit_label()` |
| `MoeNt1Gen::new(kind, arch)?` | one NT1 entry (no LDS, no hand-off). `g.plan()?`, `g.kernel_spec()`, `g.gate_up()`, `g.label(name)`, `g.exit_label()` |
| `moent::pro_front(wg, g)?`, `moent::pro_expert(wg, g)?`, `moent::pro_run(wg, g)?`, `moent::pro_leader(wg, g)?`, `moent::pro_cnt(wg, g)?`, `moent::pro_desc(wg, g)?`, `moent::pro_init(wg, g)?` | prologue runs: kernel arguments and ids; the tile's expert; the expert's weights and the tile-id descriptor; this tile's position in its run; the live-tile count, row bytes, trips, row base and descriptors; descriptors, lane offsets and the first epoch's weights; the fold magic and zeroed sums |
| `moent::walk_head(wg, g)?`, `moent::walk_step(wg, g)?` | one iteration of the run walk up to its lane mask; its back-edge step (the body of `wg.loop_until(g.label("run_walk"), g.label("run_found"), ..)`) |
| `moent::pro_slot_load(wg, g, j)?`, `moent::pro_tile(wg, g, j)?` | tile `j`'s sorted-slot load; its activation row offset (dead tiles gather slot -1) |
| `moent::outside_insn(g)`, `moent::sentinel_insn(g)`, `moent::walk_hit_insn(g)`, `moent::follower_insn(g)` | `Instruction`: the tile is past the end; a sentinel expert; some lane found the run's start; not the run's leader |
| `moent::load_w(wg, g, p)?`, `moent::load_x(wg, g, j, p)?` | weight loads of epoch parity `p`; tile `j`'s activation loads through epoch `p`'s descriptor |
| `moent::compute_head(wg, g, p)?`, `moent::fold_tile(wg, g, p, j)?` | epoch `p`'s row scales and rebias; epoch `p` folded into tile `j`'s sums |
| `moent::wait_tile(wg, g, j)?`, `moent::tile_live_insn(g, j)?` | the waits tile `j`'s fold needs; `Instruction`: `cnt > j` (wave-uniform) |
| `moent::advance(wg)?`, `moent::trips_dec(wg)?` | step the weight and activation descriptors one trip on; one trip done |
| `moent::trips_zero_insn()`, `moent::trips_more_insn()`, `moent::odd_insn()`, `moent::up_insn()` | `Instruction`: no full trip; trips remain; K/128 is odd; the up waves (hand-off readers) |
| `moent::epilogue_label(wg, g)?`, `moent::y_descriptor(wg, g)?`, `moent::y_step(wg)?`, `moent::store_rows(wg, g, j, gate)?` | the epilogue label; the output descriptor; one tile on; tile `j`'s BF16 rows from the sums (`gate = false`) or the SwiGLU registers (`true`) |
| `moent::bf16_roundtrip(wg, g)?`, `moent::silu_tile(wg, g, j)?` | `gate_up` only: every wave's BF16 round trip before the hand-off; tile `j`'s SwiGLU after the gate rows are read back |
| `moent::publish_insns(g, j)?`, `moent::gate_load_insns(g, j)?` | `gate_up` only, `Vec<Instruction>` of two: tile `j`'s gate `ds_store_b128`s (issue with `wg.ds_store(region, insn)` in the `handoff` write closure); its read-back `ds_load_b128`s (`wg.ds_load(region, insn)` in the read closure) |
| `moe1::pro_front(wg, g)?`, `moe1::pro_expert(wg)?`, `moe1::pro_setup(wg, g)?`, `moe1::pro_init(wg, g)?` | NT1 prologue runs (setup includes the first set's loads) |
| `moe1::outside_insn()`, `moe1::sentinel_insn()`, `moe1::trips_zero_insn()`, `moe1::trips_more_insn()`, `moe1::odd_insn()` | `Instruction`, as the `moent::` forms |
| `moe1::load_set(wg, g, p)?`, `moe1::compute(wg, g, p)?`, `moe1::advance(wg, g)?`, `moe1::trips_dec(wg)?`, `moe1::tail_pair(wg, g)?` | set `p`'s loads; set `p` folded; descriptors one trip on; one trip done; the two epochs left after the trips |
| `moe1::epilogue_label(wg, g)?`, `moe1::swiglu(wg, g)?`, `moe1::store(wg, g)?` | the epilogue label; `gate_up` only: BF16 round trip and SwiGLU; the packed BF16 store at `Y` |

Every native refuses a tile index at or past `g.nt()` and a parity above 1 before emitting. `pm_loop --kernel qwen4_moe_sym --epi <variant> --rip crates/peacemaker-isa/src/kernels/qwen4_moe_sym.rip [--shape fn:<preset>]` times a script against the shipped entry through the card (cookbook 14); the twin's record says `identical_to_generator: true`.

Out of scope: the MQ3-I4 and E8P-I4 NT1 generators (`qwen4_moe_sym_w3`, `qwen4_moe_sym_e8p`), the gfx1151 entries (including the row-repeat NT4 `rr > 1` family and its `transition`) and the grouped-position rotate producer have no twin.

## 17. Production generator on the raw `Builder`: `fa_prefill`

The gfx1201 GQA-fused FA2 prefill attention for the A3B head geometry (H16 / KV2 / D256; `peacemaker_isa::kernels::fa_prefill`) is composable from `.rip`. Like `fp8_gemm` (§15) it is an untyped-`Builder` kernel (a tile loop with `loop_`, `control` branches, raw barriers), so its twins are written on the raw `Builder` surface, not on `Workgroup`: no typestate and no sealed ring apply, and the checks are the Builder's (LDS slot machine, wait ledger, hazard guards, loop fixpoint). Three exports are covered, in two twin files:

- [`fa_prefill.rip`](../../peacemaker-isa/src/kernels/fa_prefill.rip) (Q8_0 K/V): variants `direct` (`attention_q8_0_fa2_gqa_pm_gfx1201`) and `preconvert` (`attention_fa2_q_preconvert_pm_gfx1201`);
- [`fa_prefill_fp8.rip`](../../peacemaker-isa/src/kernels/fa_prefill_fp8.rip) (native fp8 E4M3 K/V): variant `direct` (`attention_fp8_e4m3_fa2_gqa_f16_pm_gfx1201`).

`fa_qprep` is not composed: it is a separate generator with its own module and no shared blocks.

The script declares the LDS slots (`b.lds_slot`, from `g.lds_slots()`: `k_plane` + `v_plane`, or the single reused `stage`), places every label (`fa::label`), issues every barrier (`b.barrier_with_scope(.., "lds_only")`) and the tile loop (`b.loop_(fa::LOOP, ..)`). The natives emit straight into the builder passed as their first argument. Geometry (four compute waves + one helper, KT64 tiles, the register map) is fixed by the Rust generator.

### Coarse natives and the generator

| call | meaning |
|---|---|
| `FaPrefill::emit(arch, spec)?`, `FaPrefill::emit_preconvert(arch, spec)?` | what `fa_prefill::emit` / `emit_preconvert` return (`Emitted`) |
| `FaPrefill::emit_module(arch, spec)?`, `FaPrefill::emit_fp8_module(arch, spec)?` | `(Vec<Emitted>, text, proof_json)` of the Q8_0 module (preconvert, direct) / the one-export fp8 module |
| `FaKv::{Q8_0, Fp8E4M3}`, `FaSpec { heads, kv_heads, head_dim, kv }` | the codec enum and the spec literal; the Rust gate (gfx1201, exactly 16 / 2 / 256) stays in force and every field is required |
| `FaGen::new(arch, spec)?` | the generator of one valid spec: an opaque value |
| `g.plan()?`, `g.kernel_spec()` | the direct export's `RegPlan` and `KernelSpec` |
| `g.preconvert_plan()?`, `g.preconvert_kernel_spec()` | the preconvert export's |
| `g.lds_slots()` | `Vec<(name, base, bytes)>`, in slot-id order |
| `g.spec()` | the `FaSpec` |
| `fa::LOOP FILL_DONE KFILL_DONE KREADY QK QK_DONE VFILL_DONE VREADY TURN EPI END PRECONVERT_END` | the label constants (strings) |

### Blocks

`b` is the `Builder` of `Builder::new(g.kernel_spec(), g.plan()?)`; `g` an `FaGen`.

| call | what it does |
|---|---|
| `fa::label(b, name)?` | place a label |
| `fa::prologue(b, g)?` | entry block: guards (branch to `fa::END`), scalar state, query row / head, group bounds, Q fragments, online state; branches to `fa::EPI` when no tile is live |
| `fa::tile_setup(b, g)?` | per-tile counts and the K/V tile bases (`ktile * 544` Q8_0, `ktile * 516` fp8) |
| `fa::fill(b, g)?` | Q8_0: the joint K and V fill; the helper wave branches to `fa::FILL_DONE`. Refuses an fp8 generator |
| `fa::fill_k(b, g)?`, `fa::fill_v(b, g)?` | fp8: the K fill, then (after the stage is retired) the V fill. Refuse a Q8_0 generator |
| `fa::skip_if_helper(b, label)?` | the helper wave (wave 4) skips to `label` |
| `fa::qk(b, g)?` | the QK product of the tile |
| `fa::softmax_sub(b, g, sub)?` | one key subtile's softmax and PV, `sub` in `0..4` |
| `fa::next_tile(b)?` | `ktile += 64` and the back edge to `fa::LOOP` while `ktile <= gmax` |
| `fa::epilogue(b, g)?` | label `fa::EPI` through `fa::END` and `s_endpgm`: `out = O * (l > 0 ? 1/l : 0)` and the stores |
| `fa::preconvert(b, g)?` | the whole preconvert body through `s_endpgm` |

The Q8_0 tile is `tile_setup`, `fill`, label `FILL_DONE`, `[ready(0), ready(1)]`, label `QK`, `skip_if_helper(TURN)`, `qk`, four `softmax_sub`, label `TURN`, `[retire(0), retire(1)]`, `next_tile`. The fp8 tile reuses one 32-KiB stage sequentially: `fill_k`, `[ready(0)]`, `qk` (helper skips to `QK_DONE`), `[retire(0)]`, `fill_v`, `[ready(0)]`, four `softmax_sub` (helper skips to `TURN`), `[retire(0)]`, `next_tile`. `fa_prefill::emit` is written over the same blocks and is pinned to the digests recorded before they existed for both modules, every export and the three shape contracts (`peacemaker-isa/tests/fa_prefill_digests.rs`); the twins re-emit the bytes and the proof of every export (`tests/fa_prefill_twin.rs`). `pm_loop --kernel fa_prefill --rip fa_prefill.rip` times them ([cookbook 15](cookbook/15-fa-prefill-blocks.md)).

## 18. The PM producers: `C8Gen`, `RotGen`, `GdnGen`

`iu4_c8_producer_pm` (four exports), `moe_rotate_i4_pm` (flat-slot and grouped-position) and `gdn_chunk_prep_pm` are untyped-`Builder` kernels, so their twins are too: `Builder::new(g.kernel_spec(), g.plan()?)`, `b.enable_delay_alu()`, no `Workgroup`. A generator is an opaque handle; the natives are the public `blocks` of `iu4_c8_producer`, `moe_rotate_i4_pm` and `gdn_chunk_prep`, each emitting straight into the builder `b` passed first. The committed twins are [`iu4_c8_producer.rip`](../../peacemaker-isa/src/kernels/iu4_c8_producer.rip), [`moe_rotate_i4_pm.rip`](../../peacemaker-isa/src/kernels/moe_rotate_i4_pm.rip) and [`gdn_chunk_prep.rip`](../../peacemaker-isa/src/kernels/gdn_chunk_prep.rip); `tests/producers_twin.rs` requires `s_text` and proof identity with the Rust emit of every export and the module text assembled from the twin's exports, and `peacemaker-isa/tests/producers_pinned.rs` pins the Rust side (every export and module) to the digests recorded before the blocks were split. `pm_loop --kernel iu4_c8_producer_pm|moe_rotate_i4_pm|gdn_chunk_prep_pm --rip FILE.rip` compiles the twin's variant, ABI-checks it against the Rust emit of the same entry and times it.

| call | meaning |
|---|---|
| `C8Gen::new(arch, tag)?` | one c8 export (`rotate`, `norm`, `sigmoid_rotate`, `norm_out`); `g.plan()?`, `g.kernel_spec()`, `g.tag()`, `g.lds_bytes()` |
| `RotGen::new(arch, gp)?` | `moe_rotate_i4_pm` (`gp`: the grouped-position twin, record `g * P + row`); `g.plan()?`, `g.kernel_spec()`, `g.gp()` |
| `GdnGen::new(arch)?` | `gdn_chunk_prep_pm` (the A3B geometry); `g.plan()?`, `g.kernel_spec()` |
| `prod::label(b, name)?`, `prod::wait_all(b)?` | place a label; drain every counter |
| `c8::prologue(b, g)?`, `c8::rotate_front(b)?`, `c8::sigmoid_rotate_front(b)?` | arguments, ids and exits, pointers, lane offsets; the rotate's `x * signs1` (AWQ divide first); the sigmoid gate in front of it |
| `c8::norm_row_loads(b)?`, `c8::norm_prefetch(b)?`, `c8::norm_sumsq(b)?` | the norm's row dwords; the next phase's loads; the per-thread `fma` chain |
| `c8::norm_reduce_pre(b, g)?`, `c8::norm_store_insn()`, `c8::norm_reduce_addr(b, g)?`, `c8::norm_load_insns()`, `c8::norm_reduce_post(b, g)?` | the reduction in five pieces: up to the LDS store; the store `Instruction` (for `b.ds_store(0, ..)`); the load address; the two `ds_load_b128` `Instruction`s (for `b.ds_load(0, ..)`); the eight-root tree (and for `norm_out` the in-wave levels). The barrier between is the script's `b.barrier_with_scope([ready(0)], "lds_only")` |
| `c8::norm_rms(b)?`, `c8::norm_apply(b, g)?` | `rsqrt(eps + mean)` with the denormal guard; `y = (x * w) * rms`, the norm-out store, the AWQ divide and signs1 |
| `c8::fwht_scale_store(b)?`, `c8::quantize(b)?`, `c8::store_record(b, g)?` | FWHT, scale and the rotated store; the C8 quantizer; the record store, the export's exit label and `s_endpgm` |
| `rot::prologue(b, g)?`, `rot::rotate(b)?`, `rot::quantize(b, e)?`, `rot::store(b)?` | the prologue; the FWHT128 (returns the four registers the quantizer reads); the `{5,7}` quantizer; the record store |
| `gdn::label(name)?` | `end`, `gate_loop`, `gate_done`, `k_start`, `v_start`, `epi`, `q_loop`, `k_loop`, `v_loop` |
| `gdn::prologue(b)?`, `gdn::gate_guard(b)?`, `gdn::gate_setup(b)?`, `gdn::gate_body(b)?` | arguments and ids; only `by == 0` blocks run the gates; the gate set-up; one trip of the gate loop (back branch included: use it as the body of `b.loop_(gdn::label("gate_loop")?, \|b\| ..)`) |
| `gdn::conv_setup(b)?`, `gdn::conv_to_v(b)?`, `gdn::conv_to_k(b)?`, `gdn::region_setup(b, "q" \| "k" \| "v")?`, `gdn::row_body(b, region, head)?`, `gdn::to_epi(b)?`, `gdn::epilogue(b)?` | the conv unit: set-up, the unit dispatch, a region's pointers, one trip of a region's row loop, `s_branch epi`, the chunk-0 conv_state advance |

Variants: `rotate`, `norm`, `sigmoid_rotate`, `norm_out` (c8), `rows`, `gp` (rotate), `chunk_prep` (gdn). gfx1201 only.
