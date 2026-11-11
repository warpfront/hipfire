# peacemaker-rip diagnostics

Every failure of `peacemaker-rip check`, `build` and `fmt` carries a stable code `Rxxxx`. This page is the reference for all of them; `peacemaker-rip explain Rxxxx` prints one section and `peacemaker-rip explain --list` lists the codes. The codes are grouped by stage:

- `R01xx` the source does not lex or parse, or uses a macro the evaluator does not know;
- `R02xx` the evaluator refused a statement while running the script (names, calls, types, unsupported syntax, assertions);
- `R03xx` the typed Workgroup driver refused an operation (LDS ownership, barriers, control flow, labels);
- `R04xx` the instruction builder refused the kernel (register plan, instruction policy, counters, kernel end);
- `R05xx` the entry contract of `emit`;
- `R06xx` instruction text validation;
- `R07xx` certification, `R09xx` the tool, `R0000` unclassified.

The diagnostics are rendered in the style of rustc:

```text
error[R0301]: barrier would publish t with an undrained LDS store: wait for it first (s_barrier does not wait for memory counters)
 --> kernel.rip:16:19
   |
16 |     let (t,) = wg.barrier((ready(t, st),))?;
   |                   ^^^^^^^^^^^^^^^^^^^^^^^^
  = help: Drain the LDS store for t: `let drained = wg.wait(store)?;`, then publish it with `wg.barrier((ready(t, drained),))?;`.
  = note: details: peacemaker-rip explain R0301
```

and as JSON (`--json`) with the fields `code`, `severity`, `message`, `file`, `span`, `help`, `suggestion`, `notes` and `explain`.

Each section below has a whole failing script (a fenced block marked `fails=Rxxxx`; the diagnostics shown are for `gfx1201`) and the whole corrected script. Every script is complete: save it to a file and run `peacemaker-rip check FILE --arch gfx1201`. The failing script produces exactly that code; the corrected script compiles without a diagnostic. R0701, R0901 and R0000 depend on the environment or are the classifier's fallback, so their sections have no script example.

## R0101: syntax error

The source is not valid `.rip`. `.rip` is Rust-shaped: the file is lexed, its item keywords (`region`, ...) are rewritten and the result is parsed with `syn`, so any token sequence that rustc would refuse at the parse stage is refused here, at the position of the first offending token. Nothing in the file is evaluated when this happens.

Typical messages: `expected an expression`, `` expected `;` ``, `unexpected end of input`, `unclosed delimiter`.

The usual causes are a missing operand after `=`, an unbalanced `(`, `{` or `[`, a missing `;`, and a literal the lexer cannot read. Fix the statement the caret points at. The failing script below has an initializer with no value.

Fails:

```rip fails=R0101
fn emit(spec: Spec) -> Result<Emitted, String> {
    let kspec = KernelSpec {
        kernel_id: "diag".into(), variant: "Default".into(), arch: spec.arch, symbol: "diag_kernel".into(),
        kernargs: KernargLayout::new(0), user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 1024, wave32: true, cu_mode: false,
    };
    let mut p = RegPlan::new(256, 104)?;
    for i in 0..4u8 { p.v::<8>("vectors", 8 * i, Live::Whole)?; }
    for i in 0..4u8 { p.s::<8>("scalars", 8 * i, Live::Whole)?; }
    let mut b = Builder::new(kspec, p);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    let lanes: u32 = ;
    let end = wg.exit(".Lend")?;
    wg.end(end)?;
    b.finish()
}
```

Fixed:

```rip
fn emit(spec: Spec) -> Result<Emitted, String> {
    let kspec = KernelSpec {
        kernel_id: "diag".into(), variant: "Default".into(), arch: spec.arch, symbol: "diag_kernel".into(),
        kernargs: KernargLayout::new(0), user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 1024, wave32: true, cu_mode: false,
    };
    let mut p = RegPlan::new(256, 104)?;
    for i in 0..4u8 { p.v::<8>("vectors", 8 * i, Live::Whole)?; }
    for i in 0..4u8 { p.s::<8>("scalars", 8 * i, Live::Whole)?; }
    let mut b = Builder::new(kspec, p);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    let lanes: u32 = 32;
    let end = wg.exit(".Lend")?;
    wg.end(end)?;
    b.finish()
}
```

## R0102: unsupported macro

The evaluator implements a fixed set of macros (`format!`, `assert!`, `assert_eq!`, `matches!`, `vec!`, `panic!` and the like). Any other macro is expanded into an error that fires when the statement is *executed*, so a macro inside a function nobody calls is not reported. Macros are never silently skipped.

Typical message: `` macro `wave_size!` is not supported by the .rip evaluator ``.

Replace the macro by plain expressions, a helper function or a `const`. Do not define your own `macro_rules!`: user macros are not expanded.

Fails:

```rip fails=R0102
fn emit(spec: Spec) -> Result<Emitted, String> {
    let kspec = KernelSpec {
        kernel_id: "diag".into(), variant: "Default".into(), arch: spec.arch, symbol: "diag_kernel".into(),
        kernargs: KernargLayout::new(0), user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 1024, wave32: true, cu_mode: false,
    };
    let mut p = RegPlan::new(256, 104)?;
    for i in 0..4u8 { p.v::<8>("vectors", 8 * i, Live::Whole)?; }
    for i in 0..4u8 { p.s::<8>("scalars", 8 * i, Live::Whole)?; }
    let mut b = Builder::new(kspec, p);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    wave_size!(32);
    let end = wg.exit(".Lend")?;
    wg.end(end)?;
    b.finish()
}
```

Fixed:

```rip
fn emit(spec: Spec) -> Result<Emitted, String> {
    let kspec = KernelSpec {
        kernel_id: "diag".into(), variant: "Default".into(), arch: spec.arch, symbol: "diag_kernel".into(),
        kernargs: KernargLayout::new(0), user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 1024, wave32: true, cu_mode: false,
    };
    let mut p = RegPlan::new(256, 104)?;
    for i in 0..4u8 { p.v::<8>("vectors", 8 * i, Live::Whole)?; }
    for i in 0..4u8 { p.s::<8>("scalars", 8 * i, Live::Whole)?; }
    let mut b = Builder::new(kspec, p);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    let lanes: u32 = 32;
    let end = wg.exit(".Lend")?;
    wg.end(end)?;
    b.finish()
}
```

## R0201: unknown name

A variable, constant or function name is not in scope where it is used. The evaluator resolves names lexically when the statement runs: locals, then the enclosing items of the script, then the native vocabulary (`v`, `s`, `op`, `mem`, `srd_tail`, `Workgroup::new`, ...).

Typical messages: `` cannot find variable `x` ``, `` cannot find `srd_tial` in this scope ``.

When the name is close to one in the native vocabulary the diagnostic carries a did-you-mean in `help` and the replacement source in `suggestion`. Otherwise check the spelling, that the `let` is in an enclosing block, and that the helper is defined in the script (the evaluator has no `use` of other files).

Fails:

```rip fails=R0201
fn emit(spec: Spec) -> Result<Emitted, String> {
    let kspec = KernelSpec {
        kernel_id: "diag".into(), variant: "Default".into(), arch: spec.arch, symbol: "diag_kernel".into(),
        kernargs: KernargLayout::new(0), user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 1024, wave32: true, cu_mode: false,
    };
    let mut p = RegPlan::new(256, 104)?;
    for i in 0..4u8 { p.v::<8>("vectors", 8 * i, Live::Whole)?; }
    for i in 0..4u8 { p.s::<8>("scalars", 8 * i, Live::Whole)?; }
    let mut b = Builder::new(kspec, p);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    srd_tial(wg.isa(), 12, None)?;
    let end = wg.exit(".Lend")?;
    wg.end(end)?;
    b.finish()
}
```

Fixed:

```rip
fn emit(spec: Spec) -> Result<Emitted, String> {
    let kspec = KernelSpec {
        kernel_id: "diag".into(), variant: "Default".into(), arch: spec.arch, symbol: "diag_kernel".into(),
        kernargs: KernargLayout::new(0), user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 1024, wave32: true, cu_mode: false,
    };
    let mut p = RegPlan::new(256, 104)?;
    for i in 0..4u8 { p.v::<8>("vectors", 8 * i, Live::Whole)?; }
    for i in 0..4u8 { p.s::<8>("scalars", 8 * i, Live::Whole)?; }
    let mut b = Builder::new(kspec, p);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    srd_tail(wg.isa(), 12, None)?;
    let end = wg.exit(".Lend")?;
    wg.end(end)?;
    b.finish()
}
```

## R0202: unknown method

A method is called on a value whose type has no such method. The message names both: `` no method `barier` on `Workgroup` ``. The receiver types of the native vocabulary (`Workgroup`, `Builder`, `RegPlan`, `Chain`, integers, `Vec`, `Option`, ...) each have a fixed method set.

A close spelling is offered as a did-you-mean (`help`) and as a replacement for the call (`suggestion`). If the method exists under another receiver, the receiver is usually the mistake: builder instructions go through `wg.isa()`, workgroup operations through `wg`, and `wg.isa()` is only valid inside the Workgroup scope (see R0305).

Fails:

```rip fails=R0202
fn emit(spec: Spec) -> Result<Emitted, String> {
    let kspec = KernelSpec {
        kernel_id: "diag".into(), variant: "Default".into(), arch: spec.arch, symbol: "diag_kernel".into(),
        kernargs: KernargLayout::new(0), user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 1024, wave32: true, cu_mode: false,
    };
    let mut p = RegPlan::new(256, 104)?;
    for i in 0..4u8 { p.v::<8>("vectors", 8 * i, Live::Whole)?; }
    for i in 0..4u8 { p.s::<8>("scalars", 8 * i, Live::Whole)?; }
    let mut b = Builder::new(kspec, p);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    wg.barier(())?;
    let end = wg.exit(".Lend")?;
    wg.end(end)?;
    b.finish()
}
```

Fixed:

```rip
fn emit(spec: Spec) -> Result<Emitted, String> {
    let kspec = KernelSpec {
        kernel_id: "diag".into(), variant: "Default".into(), arch: spec.arch, symbol: "diag_kernel".into(),
        kernargs: KernargLayout::new(0), user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 1024, wave32: true, cu_mode: false,
    };
    let mut p = RegPlan::new(256, 104)?;
    for i in 0..4u8 { p.v::<8>("vectors", 8 * i, Live::Whole)?; }
    for i in 0..4u8 { p.s::<8>("scalars", 8 * i, Live::Whole)?; }
    let mut b = Builder::new(kspec, p);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    wg.barrier(())?;
    let end = wg.exit(".Lend")?;
    wg.end(end)?;
    b.finish()
}
```

## R0203: wrong number of arguments

A function, method or closure was called with a different number of arguments than it takes. This covers the native vocabulary (`` `op` takes 4 arguments, got 3 ``), script functions and closures (`` closure takes 1 arguments, got 2 ``), and macros missing an operand (`assert_eq!` needs two).

Each native function documents its arity by its first use in the production kernels: `op(builder, text, defs, uses)`, `mem(builder, text, defs, uses, class)`, `sop(builder, text, defs, uses)`, `srd_tail(builder, srd, rec)`. Pass every argument, in order; `&[]` is the explicit empty register list.

Fails:

```rip fails=R0203
fn emit(spec: Spec) -> Result<Emitted, String> {
    let kspec = KernelSpec {
        kernel_id: "diag".into(), variant: "Default".into(), arch: spec.arch, symbol: "diag_kernel".into(),
        kernargs: KernargLayout::new(0), user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 1024, wave32: true, cu_mode: false,
    };
    let mut p = RegPlan::new(256, 104)?;
    for i in 0..4u8 { p.v::<8>("vectors", 8 * i, Live::Whole)?; }
    for i in 0..4u8 { p.s::<8>("scalars", 8 * i, Live::Whole)?; }
    let mut b = Builder::new(kspec, p);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    op(wg.isa(), "v_mov_b32_e32 v0, 0", &[v(0)])?;
    let end = wg.exit(".Lend")?;
    wg.end(end)?;
    b.finish()
}
```

Fixed:

```rip
fn emit(spec: Spec) -> Result<Emitted, String> {
    let kspec = KernelSpec {
        kernel_id: "diag".into(), variant: "Default".into(), arch: spec.arch, symbol: "diag_kernel".into(),
        kernargs: KernargLayout::new(0), user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 1024, wave32: true, cu_mode: false,
    };
    let mut p = RegPlan::new(256, 104)?;
    for i in 0..4u8 { p.v::<8>("vectors", 8 * i, Live::Whole)?; }
    for i in 0..4u8 { p.s::<8>("scalars", 8 * i, Live::Whole)?; }
    let mut b = Builder::new(kspec, p);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    op(wg.isa(), "v_mov_b32_e32 v0, 0", &[v(0)], &[])?;
    let end = wg.exit(".Lend")?;
    wg.end(end)?;
    b.finish()
}
```

## R0204: integer typing or overflow

Integer arithmetic in `.rip` is exact, as in a debug build of Rust. Each integer value has a width and signedness; two typed operands of different types are refused, an operation whose result does not fit its type is refused, and so are a zero divisor and an out-of-range shift or literal. Unsuffixed literals adopt the type of the other operand.

Typical messages: `attempt to add with overflow: 200 + 100`, `` mismatched integer types in `+`: u8 and u32 ``, `attempt to divide by zero`, `` literal 256 out of range for `u8` ``.

Declare the value with a type wide enough for every result (the example widens `u8` to `u32`), or convert one side explicitly with `as` (truncating) or `u32::from(x)` (widening).

Fails:

```rip fails=R0204
fn emit(spec: Spec) -> Result<Emitted, String> {
    let kspec = KernelSpec {
        kernel_id: "diag".into(), variant: "Default".into(), arch: spec.arch, symbol: "diag_kernel".into(),
        kernargs: KernargLayout::new(0), user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 1024, wave32: true, cu_mode: false,
    };
    let mut p = RegPlan::new(256, 104)?;
    for i in 0..4u8 { p.v::<8>("vectors", 8 * i, Live::Whole)?; }
    for i in 0..4u8 { p.s::<8>("scalars", 8 * i, Live::Whole)?; }
    let mut b = Builder::new(kspec, p);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    let lane: u8 = 200;
    let next = lane + 100;
    let end = wg.exit(".Lend")?;
    wg.end(end)?;
    b.finish()
}
```

Fixed:

```rip
fn emit(spec: Spec) -> Result<Emitted, String> {
    let kspec = KernelSpec {
        kernel_id: "diag".into(), variant: "Default".into(), arch: spec.arch, symbol: "diag_kernel".into(),
        kernargs: KernargLayout::new(0), user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 1024, wave32: true, cu_mode: false,
    };
    let mut p = RegPlan::new(256, 104)?;
    for i in 0..4u8 { p.v::<8>("vectors", 8 * i, Live::Whole)?; }
    for i in 0..4u8 { p.s::<8>("scalars", 8 * i, Live::Whole)?; }
    let mut b = Builder::new(kspec, p);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    let lane: u32 = 200;
    let next = lane + 100;
    let end = wg.exit(".Lend")?;
    wg.end(end)?;
    b.finish()
}
```

## R0205: type mismatch

A value of one type was used where the evaluator needs another. Conditions must be `bool` (there is no truthiness of integers), `?` applies only to `Result` and `Option`, operators need matching operand kinds, and the typed handle arguments of the driver (regions, rings, conditions, exits) must be handles of that kind.

Typical messages: `condition must be bool, found u32`, `` `?` needs a Result or Option, found u32 ``, `` mismatched types: expected `u8`, found `u32` ``.

Compare explicitly (`tiles != 0`), or pass the right kind of value.

Fails:

```rip fails=R0205
fn emit(spec: Spec) -> Result<Emitted, String> {
    let kspec = KernelSpec {
        kernel_id: "diag".into(), variant: "Default".into(), arch: spec.arch, symbol: "diag_kernel".into(),
        kernargs: KernargLayout::new(0), user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 1024, wave32: true, cu_mode: false,
    };
    let mut p = RegPlan::new(256, 104)?;
    for i in 0..4u8 { p.v::<8>("vectors", 8 * i, Live::Whole)?; }
    for i in 0..4u8 { p.s::<8>("scalars", 8 * i, Live::Whole)?; }
    let mut b = Builder::new(kspec, p);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    let tiles: u32 = 4;
    if tiles { wg.barrier(())?; }
    let end = wg.exit(".Lend")?;
    wg.end(end)?;
    b.finish()
}
```

Fixed:

```rip
fn emit(spec: Spec) -> Result<Emitted, String> {
    let kspec = KernelSpec {
        kernel_id: "diag".into(), variant: "Default".into(), arch: spec.arch, symbol: "diag_kernel".into(),
        kernargs: KernargLayout::new(0), user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 1024, wave32: true, cu_mode: false,
    };
    let mut p = RegPlan::new(256, 104)?;
    for i in 0..4u8 { p.v::<8>("vectors", 8 * i, Live::Whole)?; }
    for i in 0..4u8 { p.s::<8>("scalars", 8 * i, Live::Whole)?; }
    let mut b = Builder::new(kspec, p);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    let tiles: u32 = 4;
    if tiles != 0 { wg.barrier(())?; }
    let end = wg.exit(".Lend")?;
    wg.end(end)?;
    b.finish()
}
```

## R0206: unsupported language feature

The script *executed* syntax the evaluator does not support: `async` blocks, unsafe code, trait objects, user generics beyond what the helpers need, a refutable pattern in a `let` without `else`, and similar. Unsupported syntax in a function that is never called is not examined; once it runs it is a located error, never a silent no-op.

Typical messages: `unsupported expression (async block)`, `unsupported pattern (...)`, `unsupported literal`.

Kernel scripts need only plain Rust: `let`, `if`, `match`, `for`/`while`/`loop`, closures, structs, enums, impls, consts and the native vocabulary. Rewrite the construct with those.

Fails:

```rip fails=R0206
fn emit(spec: Spec) -> Result<Emitted, String> {
    let kspec = KernelSpec {
        kernel_id: "diag".into(), variant: "Default".into(), arch: spec.arch, symbol: "diag_kernel".into(),
        kernargs: KernargLayout::new(0), user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 1024, wave32: true, cu_mode: false,
    };
    let mut p = RegPlan::new(256, 104)?;
    for i in 0..4u8 { p.v::<8>("vectors", 8 * i, Live::Whole)?; }
    for i in 0..4u8 { p.s::<8>("scalars", 8 * i, Live::Whole)?; }
    let mut b = Builder::new(kspec, p);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    let probe = async { 1 };
    let end = wg.exit(".Lend")?;
    wg.end(end)?;
    b.finish()
}
```

Fixed:

```rip
fn emit(spec: Spec) -> Result<Emitted, String> {
    let kspec = KernelSpec {
        kernel_id: "diag".into(), variant: "Default".into(), arch: spec.arch, symbol: "diag_kernel".into(),
        kernargs: KernargLayout::new(0), user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 1024, wave32: true, cu_mode: false,
    };
    let mut p = RegPlan::new(256, 104)?;
    for i in 0..4u8 { p.v::<8>("vectors", 8 * i, Live::Whole)?; }
    for i in 0..4u8 { p.s::<8>("scalars", 8 * i, Live::Whole)?; }
    let mut b = Builder::new(kspec, p);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    let probe = 1;
    let end = wg.exit(".Lend")?;
    wg.end(end)?;
    b.finish()
}
```

## R0207: script panic or assertion

The script itself stopped: a `panic!`, an `unreachable!`, a failed `assert!` / `assert_eq!`, or an evaluator panic. The message is the script's own text plus the operand values for `assert_eq!`. This is how a kernel generator states its own preconditions (tile sizes, register counts), so the fix is in the value that violated the assertion, not in the assertion.

Typical messages: `assertion failed: tile must be a multiple of 16`, `` assertion failed: `24` vs `32` ``, `panic: unsupported tile 7`.

Fails:

```rip fails=R0207
const TILE: u32 = 24;
fn emit(spec: Spec) -> Result<Emitted, String> {
    let kspec = KernelSpec {
        kernel_id: "diag".into(), variant: "Default".into(), arch: spec.arch, symbol: "diag_kernel".into(),
        kernargs: KernargLayout::new(0), user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 1024, wave32: true, cu_mode: false,
    };
    let mut p = RegPlan::new(256, 104)?;
    for i in 0..4u8 { p.v::<8>("vectors", 8 * i, Live::Whole)?; }
    for i in 0..4u8 { p.s::<8>("scalars", 8 * i, Live::Whole)?; }
    let mut b = Builder::new(kspec, p);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    assert!(TILE % 16 == 0, "tile must be a multiple of 16");
    let end = wg.exit(".Lend")?;
    wg.end(end)?;
    b.finish()
}
```

Fixed:

```rip
const TILE: u32 = 32;
fn emit(spec: Spec) -> Result<Emitted, String> {
    let kspec = KernelSpec {
        kernel_id: "diag".into(), variant: "Default".into(), arch: spec.arch, symbol: "diag_kernel".into(),
        kernargs: KernargLayout::new(0), user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 1024, wave32: true, cu_mode: false,
    };
    let mut p = RegPlan::new(256, 104)?;
    for i in 0..4u8 { p.v::<8>("vectors", 8 * i, Live::Whole)?; }
    for i in 0..4u8 { p.s::<8>("scalars", 8 * i, Live::Whole)?; }
    let mut b = Builder::new(kspec, p);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    assert!(TILE % 16 == 0, "tile must be a multiple of 16");
    let end = wg.exit(".Lend")?;
    wg.end(end)?;
    b.finish()
}
```

## R0301: undrained LDS store at barrier

`s_barrier` does not wait for memory counters. A region that one wave wrote with `ds_store` may therefore be published by a barrier only after the store has been *drained* with `wg.wait(..)`; otherwise other waves can read it before the data has landed. The driver refuses the transition and names the region.

Typical message: `barrier would publish t with an undrained LDS store: wait for it first (s_barrier does not wait for memory counters)`.

`wg.ds_store(region, insn)?` returns the region (now Writing) and a pending-store token. `wg.wait(token)?` returns a drained token, and `ready(region, drained)` is what the barrier accepts. The wait picks the counter by architecture (`lgkmcnt` on gfx11, `dscnt` on gfx12).

Fails:

```rip fails=R0301
region T;
fn emit(spec: Spec) -> Result<Emitted, String> {
    let kspec = KernelSpec {
        kernel_id: "diag".into(), variant: "Default".into(), arch: spec.arch, symbol: "diag_kernel".into(),
        kernargs: KernargLayout::new(0), user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 1024, wave32: true, cu_mode: false,
    };
    let mut p = RegPlan::new(256, 104)?;
    for i in 0..4u8 { p.v::<8>("vectors", 8 * i, Live::Whole)?; }
    for i in 0..4u8 { p.s::<8>("scalars", 8 * i, Live::Whole)?; }
    let mut b = Builder::new(kspec, p);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    op(wg.isa(), "v_mov_b32_e32 v1, 0", &[v(1)], &[])?;
    let t = wg.lds::<T>("t", 0, 16)?;
    let (t, st) = wg.ds_store(t, Instruction::new("ds_store_b32 v0, v1", vec![], vec![v(0), v(1)]).memory(MemoryClass::DsStore))?;
    let (t,) = wg.barrier((ready(t, st),))?;
    wg.ds_load(&t, Instruction::new("ds_load_b32 v2, v0", vec![v(2)], vec![v(0)]).memory(MemoryClass::DsLoad))?;
    let end = wg.exit(".Lend")?;
    wg.end(end)?;
    b.finish()
}
```

Fixed:

```rip
region T;
fn emit(spec: Spec) -> Result<Emitted, String> {
    let kspec = KernelSpec {
        kernel_id: "diag".into(), variant: "Default".into(), arch: spec.arch, symbol: "diag_kernel".into(),
        kernargs: KernargLayout::new(0), user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 1024, wave32: true, cu_mode: false,
    };
    let mut p = RegPlan::new(256, 104)?;
    for i in 0..4u8 { p.v::<8>("vectors", 8 * i, Live::Whole)?; }
    for i in 0..4u8 { p.s::<8>("scalars", 8 * i, Live::Whole)?; }
    let mut b = Builder::new(kspec, p);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    op(wg.isa(), "v_mov_b32_e32 v1, 0", &[v(1)], &[])?;
    let t = wg.lds::<T>("t", 0, 16)?;
    let (t, st) = wg.ds_store(t, Instruction::new("ds_store_b32 v0, v1", vec![], vec![v(0), v(1)]).memory(MemoryClass::DsStore))?;
    let drained = wg.wait(st)?;
    let (t,) = wg.barrier((ready(t, drained),))?;
    wg.ds_load(&t, Instruction::new("ds_load_b32 v2, v0", vec![v(2)], vec![v(0)]).memory(MemoryClass::DsLoad))?;
    let end = wg.exit(".Lend")?;
    wg.end(end)?;
    b.finish()
}
```

### In a fine `iu4_gemm` K-loop schedule

A publish (`iu4::publish_slab` / `iu4::publish_meta`) leaves the next buffer `Writing` with a pending store. A barrier that rotates it before `iu4::wait` (or `wg.wait_all`) has drained those stores is refused, exactly as for a region: the stores may not have landed when other waves read.

The whole failing script and its fix are in [cookbook 17, "What the driver refuses"](cookbook/17-iu4-kloop-natives.md#what-the-driver-refuses); `tests/docs.rs` compiles it on gfx1201 and requires `R0301`.

Fix: drain first, and hand the drained tokens to the rotate:

```text
let (da, dw) = iu4::wait(wg, lds, "slab")?;
wg.barrier((rotate(lds[0], da), rotate(lds[1], dw)))?;
```


## R0302: region still published (not retired)

A region becomes Published when a barrier makes it readable by every wave. Until a later barrier *retires* it, other waves may still be reading, so anything that reuses or restructures it would race with those reads. The driver refuses a store into it, and refuses to join it with other regions, split it or re-lay it out, while it is still Published.

Typical messages: `cannot store into t: it is Published and other waves may still be reading it until a barrier retires it`, `cannot join mailbox: it is Published; other waves may still be reading it, only a barrier can retire their reads`, `cannot split a: it is Published`.

After the readers have run (`wg.ds_load(&t, ..)?`), retire the region with `wg.barrier((retire(t),))?` before reuse; it comes back Free and may be stored to, joined, split or re-laid out. A store wait does not retire readers. The failing script stores into a region that is still Published; the same fix applies to a join or split of one.

Fails:

```rip fails=R0302
region T;
fn emit(spec: Spec) -> Result<Emitted, String> {
    let kspec = KernelSpec {
        kernel_id: "diag".into(), variant: "Default".into(), arch: spec.arch, symbol: "diag_kernel".into(),
        kernargs: KernargLayout::new(0), user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 1024, wave32: true, cu_mode: false,
    };
    let mut p = RegPlan::new(256, 104)?;
    for i in 0..4u8 { p.v::<8>("vectors", 8 * i, Live::Whole)?; }
    for i in 0..4u8 { p.s::<8>("scalars", 8 * i, Live::Whole)?; }
    let mut b = Builder::new(kspec, p);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    op(wg.isa(), "v_mov_b32_e32 v1, 0", &[v(1)], &[])?;
    let t = wg.lds::<T>("t", 0, 16)?;
    let (t, st) = wg.ds_store(t, Instruction::new("ds_store_b32 v0, v1", vec![], vec![v(0), v(1)]).memory(MemoryClass::DsStore))?;
    let drained = wg.wait(st)?;
    let (t,) = wg.barrier((ready(t, drained),))?;
    wg.ds_store(t, Instruction::new("ds_store_b32 v0, v1", vec![], vec![v(0), v(1)]).memory(MemoryClass::DsStore))?;
    let end = wg.exit(".Lend")?;
    wg.end(end)?;
    b.finish()
}
```

Fixed:

```rip
region T;
fn emit(spec: Spec) -> Result<Emitted, String> {
    let kspec = KernelSpec {
        kernel_id: "diag".into(), variant: "Default".into(), arch: spec.arch, symbol: "diag_kernel".into(),
        kernargs: KernargLayout::new(0), user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 1024, wave32: true, cu_mode: false,
    };
    let mut p = RegPlan::new(256, 104)?;
    for i in 0..4u8 { p.v::<8>("vectors", 8 * i, Live::Whole)?; }
    for i in 0..4u8 { p.s::<8>("scalars", 8 * i, Live::Whole)?; }
    let mut b = Builder::new(kspec, p);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    op(wg.isa(), "v_mov_b32_e32 v1, 0", &[v(1)], &[])?;
    let t = wg.lds::<T>("t", 0, 16)?;
    let (t, st) = wg.ds_store(t, Instruction::new("ds_store_b32 v0, v1", vec![], vec![v(0), v(1)]).memory(MemoryClass::DsStore))?;
    let drained = wg.wait(st)?;
    let (t,) = wg.barrier((ready(t, drained),))?;
    wg.ds_load(&t, Instruction::new("ds_load_b32 v2, v0", vec![v(2)], vec![v(0)]).memory(MemoryClass::DsLoad))?;
    let (t,) = wg.barrier((retire(t),))?;
    let (t, st2) = wg.ds_store(t, Instruction::new("ds_store_b32 v0, v1", vec![], vec![v(0), v(1)]).memory(MemoryClass::DsStore))?;
    let drained2 = wg.wait(st2)?;
    let end = wg.exit(".Lend")?;
    wg.end(end)?;
    b.finish()
}
```

## R0303: read of an unpublished region

`ds_load` needs a Published region: one that a barrier has made visible to all waves. A region that has never been written, is still Writing (stored but not yet published), or was retired is refused, and the message names its phase.

Typical message: `t is not published: it is Writing`.

The order is: store, wait for the store, barrier with `ready(..)`, load. See R0301 for the first two steps.

Fails:

```rip fails=R0303
region T;
fn emit(spec: Spec) -> Result<Emitted, String> {
    let kspec = KernelSpec {
        kernel_id: "diag".into(), variant: "Default".into(), arch: spec.arch, symbol: "diag_kernel".into(),
        kernargs: KernargLayout::new(0), user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 1024, wave32: true, cu_mode: false,
    };
    let mut p = RegPlan::new(256, 104)?;
    for i in 0..4u8 { p.v::<8>("vectors", 8 * i, Live::Whole)?; }
    for i in 0..4u8 { p.s::<8>("scalars", 8 * i, Live::Whole)?; }
    let mut b = Builder::new(kspec, p);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    op(wg.isa(), "v_mov_b32_e32 v1, 0", &[v(1)], &[])?;
    let t = wg.lds::<T>("t", 0, 16)?;
    let (t, st) = wg.ds_store(t, Instruction::new("ds_store_b32 v0, v1", vec![], vec![v(0), v(1)]).memory(MemoryClass::DsStore))?;
    wg.ds_load(&t, Instruction::new("ds_load_b32 v2, v0", vec![v(2)], vec![v(0)]).memory(MemoryClass::DsLoad))?;
    let end = wg.exit(".Lend")?;
    wg.end(end)?;
    b.finish()
}
```

Fixed:

```rip
region T;
fn emit(spec: Spec) -> Result<Emitted, String> {
    let kspec = KernelSpec {
        kernel_id: "diag".into(), variant: "Default".into(), arch: spec.arch, symbol: "diag_kernel".into(),
        kernargs: KernargLayout::new(0), user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 1024, wave32: true, cu_mode: false,
    };
    let mut p = RegPlan::new(256, 104)?;
    for i in 0..4u8 { p.v::<8>("vectors", 8 * i, Live::Whole)?; }
    for i in 0..4u8 { p.s::<8>("scalars", 8 * i, Live::Whole)?; }
    let mut b = Builder::new(kspec, p);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    op(wg.isa(), "v_mov_b32_e32 v1, 0", &[v(1)], &[])?;
    let t = wg.lds::<T>("t", 0, 16)?;
    let (t, st) = wg.ds_store(t, Instruction::new("ds_store_b32 v0, v1", vec![], vec![v(0), v(1)]).memory(MemoryClass::DsStore))?;
    let drained = wg.wait(st)?;
    let (t,) = wg.barrier((ready(t, drained),))?;
    wg.ds_load(&t, Instruction::new("ds_load_b32 v2, v0", vec![v(2)], vec![v(0)]).memory(MemoryClass::DsLoad))?;
    let end = wg.exit(".Lend")?;
    wg.end(end)?;
    b.finish()
}
```

### In a fine `iu4_gemm` K-loop schedule

`iu4::fragments_body` (the dense fragment loads, called by the script body `rip_fragments`), `iu4::d_loads`, `iu4::scale_loads`, `iu4::scale_tail` (and the sparse `iu4::sparse_fragments`, `iu4::sparse_scale_loads`, `iu4::sparse_scale_tail`) read a ring's *current* buffer, and `iu4::wmma` (`iu4::sparse_wmma`) consumes fragments loaded from the payload rings' current slot. Each refuses a ring whose current buffer is not `Published` (`iu4::fragments_body: ring iu4.A current buffer is Free, not published: a read needs a buffer a barrier has published (publish, wait, then rotate it)`), so a WMMA group on a slab nobody published, and a read of a slot before its barrier, are build errors.

The whole failing script (the first WMMA group before the prologue published anything) and its fix are in [cookbook 17, "What the driver refuses"](cookbook/17-iu4-kloop-natives.md#what-the-driver-refuses); `tests/docs.rs` compiles it on gfx1201 and requires `R0303`.

Fix: run the prologue first (it stages and publishes block 0), then `rip_fragments(wg, g, lds)?` and the WMMA groups, as `fine_block` does.


## R0304: workgroup op under divergent control

A workgroup-scope operation (a barrier, an LDS ownership transition, a kernel exit that others wait for) must be reached by *every* wave of the workgroup. A condition made with `wg.scmp(..)` is a wave condition: waves can disagree about it, so the driver refuses a workgroup operation inside a body guarded by it.

Typical message: `barrier is workgroup scope: it cannot run under wave- or lane-dependent control`.

Branch on a workgroup-uniform fact instead. `wg.scmp_wg_uniform(..)` returns a `WgCond`, and `wg_skip_if` / `wg_exit_unless` / `exit_if` accept it. The condition is the programmer's claim that all waves agree, which is why it is a separate constructor. The same code is raised for a workgroup barrier after a wave-uniform kernel exit.

Fails:

```rip fails=R0304
fn emit(spec: Spec) -> Result<Emitted, String> {
    let kspec = KernelSpec {
        kernel_id: "diag".into(), variant: "Default".into(), arch: spec.arch, symbol: "diag_kernel".into(),
        kernargs: KernargLayout::new(0), user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 1024, wave32: true, cu_mode: false,
    };
    let mut p = RegPlan::new(256, 104)?;
    for i in 0..4u8 { p.v::<8>("vectors", 8 * i, Live::Whole)?; }
    for i in 0..4u8 { p.s::<8>("scalars", 8 * i, Live::Whole)?; }
    let mut b = Builder::new(kspec, p);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    let odd = wg.scmp(Instruction::new("s_cmp_lg_u32 s0, 0", vec![], vec![s(0)]))?;
    wg.skip_if(odd, ".Lskip", |wg| { wg.barrier(())?; Ok(()) })?;
    let end = wg.exit(".Lend")?;
    wg.end(end)?;
    b.finish()
}
```

Fixed:

```rip
fn emit(spec: Spec) -> Result<Emitted, String> {
    let kspec = KernelSpec {
        kernel_id: "diag".into(), variant: "Default".into(), arch: spec.arch, symbol: "diag_kernel".into(),
        kernargs: KernargLayout::new(0), user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 1024, wave32: true, cu_mode: false,
    };
    let mut p = RegPlan::new(256, 104)?;
    for i in 0..4u8 { p.v::<8>("vectors", 8 * i, Live::Whole)?; }
    for i in 0..4u8 { p.s::<8>("scalars", 8 * i, Live::Whole)?; }
    let mut b = Builder::new(kspec, p);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    let uniform = wg.scmp_wg_uniform(Instruction::new("s_cmp_lg_u32 s0, 0", vec![], vec![s(0)]))?;
    wg.wg_skip_if(uniform, ".Lskip", |wg| { wg.barrier(())?; Ok(()) })?;
    let end = wg.exit(".Lend")?;
    wg.end(end)?;
    b.finish()
}
```

## R0305: Workgroup scope

`let mut wg = Workgroup::new(&mut b, arch)?;` seals the builder under the typed driver. The scope of `wg` is the rest of that block, and it ends at the first later statement that mentions the builder variable again (`b.enable_delay_alu()`, `let x = b.wait(..)`, `b.finish()`...). Using `wg`, or `wg.isa()`, after that point has no active driver.

Typical messages: `` no active Workgroup here: `let mut wg = Workgroup::new(&mut b, arch)?;` opens the scope that the rest of its block runs in ``, `wg.isa() outside the Workgroup scope`, and `` `Workgroup::new(&mut b, arch)?` must be the initializer of a `let` statement ``.

Do every builder-level step (`enable_delay_alu`, `scratch_pool`, ...) before `Workgroup::new`, and write the body through `wg` only. Inside helper functions take the workgroup as a parameter (`fn author<T: Target>(wg: &mut Wg<T>, ..)`).

Fails:

```rip fails=R0305
fn emit(spec: Spec) -> Result<Emitted, String> {
    let kspec = KernelSpec {
        kernel_id: "diag".into(), variant: "Default".into(), arch: spec.arch, symbol: "diag_kernel".into(),
        kernargs: KernargLayout::new(0), user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 1024, wave32: true, cu_mode: false,
    };
    let mut p = RegPlan::new(256, 104)?;
    for i in 0..4u8 { p.v::<8>("vectors", 8 * i, Live::Whole)?; }
    for i in 0..4u8 { p.s::<8>("scalars", 8 * i, Live::Whole)?; }
    let mut b = Builder::new(kspec, p);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    b.enable_delay_alu();
    let end = wg.exit(".Lend")?;
    wg.end(end)?;
    b.finish()
}
```

Fixed:

```rip
fn emit(spec: Spec) -> Result<Emitted, String> {
    let kspec = KernelSpec {
        kernel_id: "diag".into(), variant: "Default".into(), arch: spec.arch, symbol: "diag_kernel".into(),
        kernargs: KernargLayout::new(0), user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 1024, wave32: true, cu_mode: false,
    };
    let mut p = RegPlan::new(256, 104)?;
    for i in 0..4u8 { p.v::<8>("vectors", 8 * i, Live::Whole)?; }
    for i in 0..4u8 { p.s::<8>("scalars", 8 * i, Live::Whole)?; }
    let mut b = Builder::new(kspec, p);
    b.enable_delay_alu();
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    let end = wg.exit(".Lend")?;
    wg.end(end)?;
    b.finish()
}
```

## R0306: labels, exits and branches

Labels, forward targets, reserved exits and loops have structural rules the driver checks as they are emitted: a label is placed once, a branch only goes forward (backward edges are the loop APIs), a forward target that has been branched to must be placed, and every reserved exit is placed by `wg.end(exit)?` and only there.

Typical messages: `duplicate label .Lloop`, `branch to .Lx after it was placed: branches only go forward`, `forward target .Lx is never placed`, `.Lend is a kernel exit: only Workgroup::end places it`, `loop .Lloop has no break_if: it never reaches .Lexit`.

Give every label in the kernel a distinct name (generate them with a counter when a helper is called more than once) and place each target exactly once.

Fails:

```rip fails=R0306
fn emit(spec: Spec) -> Result<Emitted, String> {
    let kspec = KernelSpec {
        kernel_id: "diag".into(), variant: "Default".into(), arch: spec.arch, symbol: "diag_kernel".into(),
        kernargs: KernargLayout::new(0), user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 1024, wave32: true, cu_mode: false,
    };
    let mut p = RegPlan::new(256, 104)?;
    for i in 0..4u8 { p.v::<8>("vectors", 8 * i, Live::Whole)?; }
    for i in 0..4u8 { p.s::<8>("scalars", 8 * i, Live::Whole)?; }
    let mut b = Builder::new(kspec, p);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    wg.label(".Lloop")?;
    wg.label(".Lloop")?;
    let end = wg.exit(".Lend")?;
    wg.end(end)?;
    b.finish()
}
```

Fixed:

```rip
fn emit(spec: Spec) -> Result<Emitted, String> {
    let kspec = KernelSpec {
        kernel_id: "diag".into(), variant: "Default".into(), arch: spec.arch, symbol: "diag_kernel".into(),
        kernargs: KernargLayout::new(0), user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 1024, wave32: true, cu_mode: false,
    };
    let mut p = RegPlan::new(256, 104)?;
    for i in 0..4u8 { p.v::<8>("vectors", 8 * i, Live::Whole)?; }
    for i in 0..4u8 { p.s::<8>("scalars", 8 * i, Live::Whole)?; }
    let mut b = Builder::new(kspec, p);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    wg.label(".Lloop")?;
    wg.label(".Lnext")?;
    let end = wg.exit(".Lend")?;
    wg.end(end)?;
    b.finish()
}
```

Another way to raise the same code:

```rip fails=R0306
fn emit(spec: Spec) -> Result<Emitted, String> {
    let kspec = KernelSpec {
        kernel_id: "diag".into(), variant: "Default".into(), arch: spec.arch, symbol: "diag_kernel".into(),
        kernargs: KernargLayout::new(0), user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 1024, wave32: true, cu_mode: false,
    };
    let mut p = RegPlan::new(256, 104)?;
    for i in 0..4u8 { p.v::<8>("vectors", 8 * i, Live::Whole)?; }
    for i in 0..4u8 { p.s::<8>("scalars", 8 * i, Live::Whole)?; }
    let mut b = Builder::new(kspec, p);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    let c = wg.scmp(Instruction::new("s_cmp_lg_u32 s0, 0", vec![], vec![s(0)]))?;
    wg.forward(|wg, f| { f.branch_if(wg, c, ".Lfar")?; Ok(()) })?;
    let end = wg.exit(".Lend")?;
    wg.end(end)?;
    b.finish()
}
```

Fixed:

```rip
fn emit(spec: Spec) -> Result<Emitted, String> {
    let kspec = KernelSpec {
        kernel_id: "diag".into(), variant: "Default".into(), arch: spec.arch, symbol: "diag_kernel".into(),
        kernargs: KernargLayout::new(0), user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 1024, wave32: true, cu_mode: false,
    };
    let mut p = RegPlan::new(256, 104)?;
    for i in 0..4u8 { p.v::<8>("vectors", 8 * i, Live::Whole)?; }
    for i in 0..4u8 { p.s::<8>("scalars", 8 * i, Live::Whole)?; }
    let mut b = Builder::new(kspec, p);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    let c = wg.scmp(Instruction::new("s_cmp_lg_u32 s0, 0", vec![], vec![s(0)]))?;
    wg.forward(|wg, f| { f.branch_if(wg, c, ".Lfar")?; f.place(wg, ".Lfar")?; Ok(()) })?;
    let end = wg.exit(".Lend")?;
    wg.end(end)?;
    b.finish()
}
```

## R0399: other driver refusal

The typed driver (`peacemaker_author::runtime`) refused an operation for a reason that has no code of its own: LDS regions out of bounds, joins and splits of regions in the wrong phase, rings, split-barrier pairing, ownership that differs between the two paths of a branch or loop, a handle from another driver, a region used after it was consumed.

Typical messages: `LDS region t [0, +70000) exceeds 65536 bytes on gfx1201`, `cannot join a: it is Published; ...`, `the paths disagree on LDS ownership: ...`, `a split barrier is already in flight`, `ring r: only a ring whose current buffer is Free and vacant can become steady`.

The message names the region and the state transition the driver wanted; `help` says which rule family applies. The example asks for a region larger than the architecture's LDS.

Fails:

```rip fails=R0399
region T;
fn emit(spec: Spec) -> Result<Emitted, String> {
    let kspec = KernelSpec {
        kernel_id: "diag".into(), variant: "Default".into(), arch: spec.arch, symbol: "diag_kernel".into(),
        kernargs: KernargLayout::new(0), user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 1024, wave32: true, cu_mode: false,
    };
    let mut p = RegPlan::new(256, 104)?;
    for i in 0..4u8 { p.v::<8>("vectors", 8 * i, Live::Whole)?; }
    for i in 0..4u8 { p.s::<8>("scalars", 8 * i, Live::Whole)?; }
    let mut b = Builder::new(kspec, p);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    let t = wg.lds::<T>("t", 0, 70000)?;
    let end = wg.exit(".Lend")?;
    wg.end(end)?;
    b.finish()
}
```

Fixed:

```rip
region T;
fn emit(spec: Spec) -> Result<Emitted, String> {
    let kspec = KernelSpec {
        kernel_id: "diag".into(), variant: "Default".into(), arch: spec.arch, symbol: "diag_kernel".into(),
        kernargs: KernargLayout::new(0), user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 1024, wave32: true, cu_mode: false,
    };
    let mut p = RegPlan::new(256, 104)?;
    for i in 0..4u8 { p.v::<8>("vectors", 8 * i, Live::Whole)?; }
    for i in 0..4u8 { p.s::<8>("scalars", 8 * i, Live::Whole)?; }
    let mut b = Builder::new(kspec, p);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    let t = wg.lds::<T>("t", 0, 1024)?;
    let end = wg.exit(".Lend")?;
    wg.end(end)?;
    b.finish()
}
```

### In a fine `iu4_gemm` K-loop schedule

The fine K-loop natives ([cookbook 17](cookbook/17-iu4-kloop-natives.md)) lend the four LDS rings to the typed block in the state its Rust step is typed for, and two of their refusals carry no code of their own:

- **A publish into a buffer that is not `Free`.** `iu4::publish_slab` / `iu4::publish_meta` write the ring's next buffer, which must be `Free`. A second publish before the first one was waited and rotated finds it `Writing` (`ring iu4.A next buffer is Writing (its stores are still pending): a publish writes a Free buffer, so wait for the slab in flight and rotate it first`).
- **A wait with nothing in flight.** `iu4::wait(wg, lds, set)` (and `wg.wait_all`) name rings whose next buffer has no store of this wave to drain (`wait on ring iu4.A: no store of this wave is in flight (Free)`).

The whole failing scripts (publish twice; wait with no publish) and their fixes are in [cookbook 17, "What the driver refuses"](cookbook/17-iu4-kloop-natives.md#what-the-driver-refuses): they are `iu4_gemm` kernels and so gfx1201-only, which the example convention of this page (every failing script also builds on gfx1151 unless it is about one architecture's spelling) does not cover; `tests/docs.rs` compiles them there and requires each to raise `R0399`.

Fix: finish the first publish (`iu4::wait(wg, lds, "slab")` then `wg.barrier((rotate(lds[0], da), rotate(lds[1], dw)))`) before the next one writes, as `fine_block` does; wait only after the publish whose stores it drains.


## R0401: register not in the plan

Every register an instruction reads or writes must be owned by exactly one range of the `RegPlan` that is live at that point. The plan is the kernel's complete register map; an instruction naming a register outside it, or one owned twice, is refused with the register and the label where it was seen.

Typical message: `v40 at entry has no unique live owner` (`entry` is the nearest label).

`help` contains the plan line that would declare the register. Either declare it (`p.v::<8>("name", base, Live::Whole)?;`, within the budget) or use a register that the plan already owns. The example touches `v40` while the plan covers `v0..v31`.

Fails:

```rip fails=R0401
fn emit(spec: Spec) -> Result<Emitted, String> {
    let kspec = KernelSpec {
        kernel_id: "diag".into(), variant: "Default".into(), arch: spec.arch, symbol: "diag_kernel".into(),
        kernargs: KernargLayout::new(0), user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 1024, wave32: true, cu_mode: false,
    };
    let mut p = RegPlan::new(256, 104)?;
    for i in 0..4u8 { p.v::<8>("vectors", 8 * i, Live::Whole)?; }
    for i in 0..4u8 { p.s::<8>("scalars", 8 * i, Live::Whole)?; }
    let mut b = Builder::new(kspec, p);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    op(wg.isa(), "v_mov_b32_e32 v40, 0", &[v(40)], &[])?;
    let end = wg.exit(".Lend")?;
    wg.end(end)?;
    b.finish()
}
```

Fixed:

```rip
fn emit(spec: Spec) -> Result<Emitted, String> {
    let kspec = KernelSpec {
        kernel_id: "diag".into(), variant: "Default".into(), arch: spec.arch, symbol: "diag_kernel".into(),
        kernargs: KernargLayout::new(0), user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 1024, wave32: true, cu_mode: false,
    };
    let mut p = RegPlan::new(256, 104)?;
    for i in 0..4u8 { p.v::<8>("vectors", 8 * i, Live::Whole)?; }
    for i in 0..4u8 { p.s::<8>("scalars", 8 * i, Live::Whole)?; }
    let mut b = Builder::new(kspec, p);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    op(wg.isa(), "v_mov_b32_e32 v1, 0", &[v(1)], &[])?;
    let end = wg.exit(".Lend")?;
    wg.end(end)?;
    b.finish()
}
```

## R0402: register plan

The `RegPlan` itself is inconsistent. Ranges are 1, 2, 4 or 8 registers wide, must lie inside the budget (at most 256 VGPRs and 104 SGPRs), must not overlap another range that is live at the same time, and wide SGPR tuples must be aligned to their width. `Live::Between(first, last)` endpoints must be labels that exist, in order.

Typical messages: `extra overlaps vectors with a shared lifetime`, `register width must be 1, 2, 4 or 8`, `big exceeds register budget`, `register budget exceeds architectural limit`, `invalid lifetime for x`.

Put each range at a base not covered by another range, or give both ranges disjoint `Live::Between` lifetimes when they should share registers. The example declares `v4..v11` inside the already declared `v0..v31`.

Fails:

```rip fails=R0402
fn emit(spec: Spec) -> Result<Emitted, String> {
    let kspec = KernelSpec {
        kernel_id: "diag".into(), variant: "Default".into(), arch: spec.arch, symbol: "diag_kernel".into(),
        kernargs: KernargLayout::new(0), user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 1024, wave32: true, cu_mode: false,
    };
    let mut p = RegPlan::new(256, 104)?;
    for i in 0..4u8 { p.v::<8>("vectors", 8 * i, Live::Whole)?; }
    for i in 0..4u8 { p.s::<8>("scalars", 8 * i, Live::Whole)?; }
    p.v::<8>("extra", 4, Live::Whole)?;
    let mut b = Builder::new(kspec, p);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    let end = wg.exit(".Lend")?;
    wg.end(end)?;
    b.finish()
}
```

Fixed:

```rip
fn emit(spec: Spec) -> Result<Emitted, String> {
    let kspec = KernelSpec {
        kernel_id: "diag".into(), variant: "Default".into(), arch: spec.arch, symbol: "diag_kernel".into(),
        kernargs: KernargLayout::new(0), user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 1024, wave32: true, cu_mode: false,
    };
    let mut p = RegPlan::new(256, 104)?;
    for i in 0..4u8 { p.v::<8>("vectors", 8 * i, Live::Whole)?; }
    for i in 0..4u8 { p.s::<8>("scalars", 8 * i, Live::Whole)?; }
    let mut b = Builder::new(kspec, p);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    let end = wg.exit(".Lend")?;
    wg.end(end)?;
    b.finish()
}
```

## R0403: instruction policy

The builder owns everything that must be tracked: barriers, branches, LDS access, clauses, scratch and special instructions. A script that passes such an instruction to the raw `op(..)` / `mem(..)` entry points is refused, because the builder could not keep its ledgers, hazards and LDS state true.

Typical messages: `DS access must carry an LDS slot through ds_load/ds_store`, `clauses and global visibility require their builder methods`, `typed kernel: LDS, barriers, branches and loops go through peacemaker-author`, `scratch instructions are forbidden`, and on gfx11 `s_barrier is a barrier: barriers go through the typed core` (on gfx12 the plain `s_barrier` does not exist, so it is R0601 first).

Use the typed operation: `wg.barrier(..)`, `wg.ds_store` / `wg.ds_load`, `wg.skip_if` and the loop APIs, `b.clause(..)`. The example loads from LDS with a raw `ds_load_b32`; the fixed script goes through the typed store, wait, barrier and `wg.ds_load`, which is the sequence the driver can prove.

Fails:

```rip fails=R0403
region T;
fn emit(spec: Spec) -> Result<Emitted, String> {
    let kspec = KernelSpec {
        kernel_id: "diag".into(), variant: "Default".into(), arch: spec.arch, symbol: "diag_kernel".into(),
        kernargs: KernargLayout::new(0), user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 1024, wave32: true, cu_mode: false,
    };
    let mut p = RegPlan::new(256, 104)?;
    for i in 0..4u8 { p.v::<8>("vectors", 8 * i, Live::Whole)?; }
    for i in 0..4u8 { p.s::<8>("scalars", 8 * i, Live::Whole)?; }
    let mut b = Builder::new(kspec, p);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    op(wg.isa(), "ds_load_b32 v2, v0", &[v(2)], &[v(0)])?;
    let end = wg.exit(".Lend")?;
    wg.end(end)?;
    b.finish()
}
```

Fixed:

```rip
region T;
fn emit(spec: Spec) -> Result<Emitted, String> {
    let kspec = KernelSpec {
        kernel_id: "diag".into(), variant: "Default".into(), arch: spec.arch, symbol: "diag_kernel".into(),
        kernargs: KernargLayout::new(0), user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 1024, wave32: true, cu_mode: false,
    };
    let mut p = RegPlan::new(256, 104)?;
    for i in 0..4u8 { p.v::<8>("vectors", 8 * i, Live::Whole)?; }
    for i in 0..4u8 { p.s::<8>("scalars", 8 * i, Live::Whole)?; }
    let mut b = Builder::new(kspec, p);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    op(wg.isa(), "v_mov_b32_e32 v1, 0", &[v(1)], &[])?;
    let t = wg.lds::<T>("t", 0, 16)?;
    let (t, st) = wg.ds_store(t, Instruction::new("ds_store_b32 v0, v1", vec![], vec![v(0), v(1)]).memory(MemoryClass::DsStore))?;
    let drained = wg.wait(st)?;
    let (t,) = wg.barrier((ready(t, drained),))?;
    wg.ds_load(&t, Instruction::new("ds_load_b32 v2, v0", vec![v(2)], vec![v(0)]).memory(MemoryClass::DsLoad))?;
    let end = wg.exit(".Lend")?;
    wg.end(end)?;
    b.finish()
}
```

## R0404: waits and counters

Wait counters are architecture specific. gfx12 splits the gfx11 counters (`Counter::Load`, `Store`, `Ds`, `Km` there; `Counter::Vm`, `Vs`, `Lgkm` on gfx11), and the builder refuses a wait that names the other generation's counter. It also refuses a loop whose back-edge wait ledger differs from the entry ledger, and a kernel that finishes with memory operations still outstanding.

Typical messages: `counter/architecture mismatch`, `loop .Lloop: back-edge ledger must equal the entry ledger`, `outstanding memory operations: wait or end the kernel before finish`.

For the loop back-edge ledger, wait for the body's stores with `wg.wait(..)` (or `wg.isa().wait(..)` on the right counter) before the back edge, so the ledger at the end of the body equals the one at its entry; `ui/loop_carried_drained_store.rip` is the worked fix. Pick the counter from `spec.arch` (`if spec.arch.gfx12() { Counter::Load } else { Counter::Vm }`), or use the typed `wg.wait(..)`, which chooses for you. The example uses the gfx11 `Vm` counter and fails when checked for gfx1201; checked for gfx1151 it is accepted.

Fails:

```rip fails=R0404 arch=gfx1201
fn emit(spec: Spec) -> Result<Emitted, String> {
    let kspec = KernelSpec {
        kernel_id: "diag".into(), variant: "Default".into(), arch: spec.arch, symbol: "diag_kernel".into(),
        kernargs: KernargLayout::new(0), user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 1024, wave32: true, cu_mode: false,
    };
    let mut p = RegPlan::new(256, 104)?;
    for i in 0..4u8 { p.v::<8>("vectors", 8 * i, Live::Whole)?; }
    for i in 0..4u8 { p.s::<8>("scalars", 8 * i, Live::Whole)?; }
    let mut b = Builder::new(kspec, p);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    wg.isa().wait(Counter::Vm, 0)?;
    let end = wg.exit(".Lend")?;
    wg.end(end)?;
    b.finish()
}
```

Fixed:

```rip
fn emit(spec: Spec) -> Result<Emitted, String> {
    let kspec = KernelSpec {
        kernel_id: "diag".into(), variant: "Default".into(), arch: spec.arch, symbol: "diag_kernel".into(),
        kernargs: KernargLayout::new(0), user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 1024, wave32: true, cu_mode: false,
    };
    let mut p = RegPlan::new(256, 104)?;
    for i in 0..4u8 { p.v::<8>("vectors", 8 * i, Live::Whole)?; }
    for i in 0..4u8 { p.s::<8>("scalars", 8 * i, Live::Whole)?; }
    let mut b = Builder::new(kspec, p);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    wg.isa().wait(if spec.arch.gfx12() { Counter::Load } else { Counter::Vm }, 0)?;
    let end = wg.exit(".Lend")?;
    wg.end(end)?;
    b.finish()
}
```

## R0405: kernel end

A kernel must end with exactly one `s_endpgm`, which the typed driver emits when `wg.end(exit)?` places the reserved kernel exit. `b.finish()` before that point, an instruction after it, or a kernel that never placed its exit is refused.

Typical messages: `kernel must end with s_endpgm`, `kernel exit .Lend is never placed`, `instruction after s_endpgm`, `kernel ends with a split barrier in flight`.

Open the workgroup, emit the body, reserve the exit with `wg.exit(label)?`, place it with `wg.end(exit)?`, then `b.finish()`. The example finishes a builder that never ended the kernel.

Fails:

```rip fails=R0405
fn emit(spec: Spec) -> Result<Emitted, String> {
    let kspec = KernelSpec {
        kernel_id: "diag".into(), variant: "Default".into(), arch: spec.arch, symbol: "diag_kernel".into(),
        kernargs: KernargLayout::new(0), user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 1024, wave32: true, cu_mode: false,
    };
    let mut p = RegPlan::new(256, 104)?;
    p.s::<8>("scalars", 0, Live::Whole)?;
    let mut b = Builder::new(kspec, p);
    b.finish()
}
```

Fixed:

```rip
fn emit(spec: Spec) -> Result<Emitted, String> {
    let kspec = KernelSpec {
        kernel_id: "diag".into(), variant: "Default".into(), arch: spec.arch, symbol: "diag_kernel".into(),
        kernargs: KernargLayout::new(0), user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 1024, wave32: true, cu_mode: false,
    };
    let mut p = RegPlan::new(256, 104)?;
    for i in 0..4u8 { p.v::<8>("vectors", 8 * i, Live::Whole)?; }
    for i in 0..4u8 { p.s::<8>("scalars", 8 * i, Live::Whole)?; }
    let mut b = Builder::new(kspec, p);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    let end = wg.exit(".Lend")?;
    wg.end(end)?;
    b.finish()
}
```

Another way to raise the same code:

```rip fails=R0405
fn emit(spec: Spec) -> Result<Emitted, String> {
    let kspec = KernelSpec {
        kernel_id: "diag".into(), variant: "Default".into(), arch: spec.arch, symbol: "diag_kernel".into(),
        kernargs: KernargLayout::new(0), user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 1024, wave32: true, cu_mode: false,
    };
    let mut p = RegPlan::new(256, 104)?;
    for i in 0..4u8 { p.v::<8>("vectors", 8 * i, Live::Whole)?; }
    for i in 0..4u8 { p.s::<8>("scalars", 8 * i, Live::Whole)?; }
    let mut b = Builder::new(kspec, p);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    let end = wg.exit(".Lend")?;
    b.finish()
}
```

Fixed:

```rip
fn emit(spec: Spec) -> Result<Emitted, String> {
    let kspec = KernelSpec {
        kernel_id: "diag".into(), variant: "Default".into(), arch: spec.arch, symbol: "diag_kernel".into(),
        kernargs: KernargLayout::new(0), user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 1024, wave32: true, cu_mode: false,
    };
    let mut p = RegPlan::new(256, 104)?;
    for i in 0..4u8 { p.v::<8>("vectors", 8 * i, Live::Whole)?; }
    for i in 0..4u8 { p.s::<8>("scalars", 8 * i, Live::Whole)?; }
    let mut b = Builder::new(kspec, p);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    let end = wg.exit(".Lend")?;
    wg.end(end)?;
    b.finish()
}
```

## R0501: entry contract

The compiler calls the script's `fn emit(spec: Spec) -> Result<Emitted, String>` (or `emit(arch, variant)`). The script must define it, with one or two parameters, and it must return the result of `b.finish()`.

Typical messages: `` no `fn emit(spec: Spec) -> Result<Emitted, String>` entry point ``, `` `emit` takes (spec) or (arch, variant), not 3 arguments ``, `` `emit` returned Ok(unit), not an Emitted (finish the builder: `b.finish()`) ``, `` `emit` must return Result<Emitted, String>, found u32 ``.

`spec.arch` is the target architecture and `spec.variant` the variant name that was requested; when the script declares `enum Kind`, `spec.kind` is the matching variant.

Fails:

```rip fails=R0501
fn build(spec: Spec) -> Result<Emitted, String> {
    let kspec = KernelSpec {
        kernel_id: "diag".into(), variant: "Default".into(), arch: spec.arch, symbol: "diag_kernel".into(),
        kernargs: KernargLayout::new(0), user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 1024, wave32: true, cu_mode: false,
    };
    let mut p = RegPlan::new(256, 104)?;
    for i in 0..4u8 { p.v::<8>("vectors", 8 * i, Live::Whole)?; }
    for i in 0..4u8 { p.s::<8>("scalars", 8 * i, Live::Whole)?; }
    let mut b = Builder::new(kspec, p);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    let end = wg.exit(".Lend")?;
    wg.end(end)?;
    b.finish()
}
```

Fixed:

```rip
fn emit(spec: Spec) -> Result<Emitted, String> {
    let kspec = KernelSpec {
        kernel_id: "diag".into(), variant: "Default".into(), arch: spec.arch, symbol: "diag_kernel".into(),
        kernargs: KernargLayout::new(0), user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 1024, wave32: true, cu_mode: false,
    };
    let mut p = RegPlan::new(256, 104)?;
    for i in 0..4u8 { p.v::<8>("vectors", 8 * i, Live::Whole)?; }
    for i in 0..4u8 { p.s::<8>("scalars", 8 * i, Live::Whole)?; }
    let mut b = Builder::new(kspec, p);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    let end = wg.exit(".Lend")?;
    wg.end(end)?;
    b.finish()
}
```

## R0502: script returned an error

The script's own `Err(..)` came out of `emit`: it decided, by its own logic, that this (architecture, variant) cannot be built. This is the generator's own refusal and its text is the script's, so the diagnostic is reported as the script's message and located at `fn emit`.

Typical message: `` script returned an error: unknown variant `default` ``.

Errors from the builder and the driver that reach `emit`'s result through `?` keep their own codes (R03xx and R04xx); only a bare `Err(..)` built by the script is R0502. Correct the condition in the script, or request a variant it supports. The example builds only the variant `select`: the failing block is checked with the default variant name and is refused, the fixed block is checked with `variant=select`.

Fails:

```rip fails=R0502
fn emit(spec: Spec) -> Result<Emitted, String> {
    let kspec = KernelSpec {
        kernel_id: "diag".into(), variant: "Default".into(), arch: spec.arch, symbol: "diag_kernel".into(),
        kernargs: KernargLayout::new(0), user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 1024, wave32: true, cu_mode: false,
    };
    let mut p = RegPlan::new(256, 104)?;
    for i in 0..4u8 { p.v::<8>("vectors", 8 * i, Live::Whole)?; }
    for i in 0..4u8 { p.s::<8>("scalars", 8 * i, Live::Whole)?; }
    let mut b = Builder::new(kspec, p);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    if spec.variant != "select" { return Err(format!("unknown variant `{}`", spec.variant)); }
    let end = wg.exit(".Lend")?;
    wg.end(end)?;
    b.finish()
}
```

Fixed:

```rip variant=select
fn emit(spec: Spec) -> Result<Emitted, String> {
    let kspec = KernelSpec {
        kernel_id: "diag".into(), variant: "Default".into(), arch: spec.arch, symbol: "diag_kernel".into(),
        kernargs: KernargLayout::new(0), user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 1024, wave32: true, cu_mode: false,
    };
    let mut p = RegPlan::new(256, 104)?;
    for i in 0..4u8 { p.v::<8>("vectors", 8 * i, Live::Whole)?; }
    for i in 0..4u8 { p.s::<8>("scalars", 8 * i, Live::Whole)?; }
    let mut b = Builder::new(kspec, p);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    if spec.variant != "select" { return Err(format!("unknown variant `{}`", spec.variant)); }
    let end = wg.exit(".Lend")?;
    wg.end(end)?;
    b.finish()
}
```

## R0601: unknown mnemonic

Every instruction text a script hands the builder (`op`, `mem`, `sop`, `Instruction::new`, ...) is parsed and encoded for the selected architecture before the builder sees it. The mnemonic must exist for that architecture; the message names it and, when one is close, offers the spelling.

Typical messages: `` unknown mnemonic `v_mov_b32_e33` on gfx1201; did you mean `v_mov_b32_e32`? ``, `` unknown mnemonic `s_add_i32` on gfx1201; this mnemonic is known on gfx11, not gfx12 ``.

Several gfx11 and gfx12 spellings differ (`s_add_i32` / `s_add_co_i32`); use the `s_add_i32(spec.arch)` family of helpers instead of hard-coding them.

Fails:

```rip fails=R0601
fn emit(spec: Spec) -> Result<Emitted, String> {
    let kspec = KernelSpec {
        kernel_id: "diag".into(), variant: "Default".into(), arch: spec.arch, symbol: "diag_kernel".into(),
        kernargs: KernargLayout::new(0), user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 1024, wave32: true, cu_mode: false,
    };
    let mut p = RegPlan::new(256, 104)?;
    for i in 0..4u8 { p.v::<8>("vectors", 8 * i, Live::Whole)?; }
    for i in 0..4u8 { p.s::<8>("scalars", 8 * i, Live::Whole)?; }
    let mut b = Builder::new(kspec, p);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    op(wg.isa(), "v_mov_b32_e33 v0, 0", &[v(0)], &[])?;
    let end = wg.exit(".Lend")?;
    wg.end(end)?;
    b.finish()
}
```

Fixed:

```rip
fn emit(spec: Spec) -> Result<Emitted, String> {
    let kspec = KernelSpec {
        kernel_id: "diag".into(), variant: "Default".into(), arch: spec.arch, symbol: "diag_kernel".into(),
        kernargs: KernargLayout::new(0), user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 1024, wave32: true, cu_mode: false,
    };
    let mut p = RegPlan::new(256, 104)?;
    for i in 0..4u8 { p.v::<8>("vectors", 8 * i, Live::Whole)?; }
    for i in 0..4u8 { p.s::<8>("scalars", 8 * i, Live::Whole)?; }
    let mut b = Builder::new(kspec, p);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    op(wg.isa(), "v_mov_b32_e32 v0, 0", &[v(0)], &[])?;
    let end = wg.exit(".Lend")?;
    wg.end(end)?;
    b.finish()
}
```

## R0602: malformed operands

The mnemonic exists but its operands do not parse or do not encode for the selected architecture: too few or too many operands, a register where an immediate is required, an immediate that does not fit the field, a register range of the wrong width or alignment.

Typical message: `` malformed operands for `v_mov_b32_e32` on gfx1201: ... `` followed by the reason.

Write the operands in the architecture's order and with legal widths and constants. The text the script hands the builder is exactly what is encoded, so the fix is in the string.

Two operand mistakes get a specific fix. A decimal that is not an inline constant (for example `bad operand "128" ... not an inline constant`) is reported with the actual number: inline integers are -16..64, so write it as a literal with `lit(128)`, which prints `0x..` (one literal per instruction). For a VOP2 `_e32` instruction the literal must be src0, or use `_e64`. On gfx1201 a `buffer_*` instruction whose SOFFSET does not fit gets: a zero buffer soffset is spelled `null`, not `0`.

Fails:

```rip fails=R0602
fn emit(spec: Spec) -> Result<Emitted, String> {
    let kspec = KernelSpec {
        kernel_id: "diag".into(), variant: "Default".into(), arch: spec.arch, symbol: "diag_kernel".into(),
        kernargs: KernargLayout::new(0), user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 1024, wave32: true, cu_mode: false,
    };
    let mut p = RegPlan::new(256, 104)?;
    for i in 0..4u8 { p.v::<8>("vectors", 8 * i, Live::Whole)?; }
    for i in 0..4u8 { p.s::<8>("scalars", 8 * i, Live::Whole)?; }
    let mut b = Builder::new(kspec, p);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    op(wg.isa(), "v_mov_b32_e32 v0", &[v(0)], &[])?;
    let end = wg.exit(".Lend")?;
    wg.end(end)?;
    b.finish()
}
```

Fixed:

```rip
fn emit(spec: Spec) -> Result<Emitted, String> {
    let kspec = KernelSpec {
        kernel_id: "diag".into(), variant: "Default".into(), arch: spec.arch, symbol: "diag_kernel".into(),
        kernargs: KernargLayout::new(0), user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 1024, wave32: true, cu_mode: false,
    };
    let mut p = RegPlan::new(256, 104)?;
    for i in 0..4u8 { p.v::<8>("vectors", 8 * i, Live::Whole)?; }
    for i in 0..4u8 { p.s::<8>("scalars", 8 * i, Live::Whole)?; }
    let mut b = Builder::new(kspec, p);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    op(wg.isa(), "v_mov_b32_e32 v0, 0", &[v(0)], &[])?;
    let end = wg.exit(".Lend")?;
    wg.end(end)?;
    b.finish()
}
```

## R0603: defs/uses disagree with the instruction

`check` reports this as an error and nothing is emitted. `op(b, text, defs, uses)` tells the builder which registers the instruction writes and reads; the builder uses that to place waits and hazard guards. If the lists omit a register the instruction actually reads or writes (as known from the opcode tables), the builder would have no way to protect it, so the call is refused.

Typical message: `` op declares defs [v2] and uses [v0] but `v_add_f32_e32 v2, v0, v1` also reads v1: write ... ``; the message ends with the corrected lists, and `help` repeats them.

Declare every destination in `defs` and every source in `uses`; extra declarations are allowed. Special registers (vcc, exec, m0, ttmp) are outside this check.

Fails:

```rip fails=R0603
fn emit(spec: Spec) -> Result<Emitted, String> {
    let kspec = KernelSpec {
        kernel_id: "diag".into(), variant: "Default".into(), arch: spec.arch, symbol: "diag_kernel".into(),
        kernargs: KernargLayout::new(0), user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 1024, wave32: true, cu_mode: false,
    };
    let mut p = RegPlan::new(256, 104)?;
    for i in 0..4u8 { p.v::<8>("vectors", 8 * i, Live::Whole)?; }
    for i in 0..4u8 { p.s::<8>("scalars", 8 * i, Live::Whole)?; }
    let mut b = Builder::new(kspec, p);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    op(wg.isa(), "v_mov_b32_e32 v1, 0", &[v(1)], &[])?;
    op(wg.isa(), "v_add_f32_e32 v2, v0, v1", &[v(2)], &[v(0)])?;
    let end = wg.exit(".Lend")?;
    wg.end(end)?;
    b.finish()
}
```

Fixed:

```rip
fn emit(spec: Spec) -> Result<Emitted, String> {
    let kspec = KernelSpec {
        kernel_id: "diag".into(), variant: "Default".into(), arch: spec.arch, symbol: "diag_kernel".into(),
        kernargs: KernargLayout::new(0), user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 1024, wave32: true, cu_mode: false,
    };
    let mut p = RegPlan::new(256, 104)?;
    for i in 0..4u8 { p.v::<8>("vectors", 8 * i, Live::Whole)?; }
    for i in 0..4u8 { p.s::<8>("scalars", 8 * i, Live::Whole)?; }
    let mut b = Builder::new(kspec, p);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    op(wg.isa(), "v_mov_b32_e32 v1, 0", &[v(1)], &[])?;
    op(wg.isa(), "v_add_f32_e32 v2, v0, v1", &[v(2)], &[v(0), v(1)])?;
    let end = wg.exit(".Lend")?;
    wg.end(end)?;
    b.finish()
}
```

## R0604: register read before it is defined

The builder collects every register the kernel writes, anywhere in it, and what the hardware initializes at launch. It refuses an instruction that reads a V or S register that is in neither set: nothing in the kernel writes it and the launch does not provide it. This is a whole-kernel check, not a per-path analysis: a register written on only one branch is accepted, and a write to part of a register counts for the whole register. The hardware state at entry on gfx12 is: `s[0:1]` the kernarg pointer, `ttmp9` the workgroup id (x), `ttmp7` the y id in the low half and the z id in the high half, and `v0` the thread id; every other register starts undefined.

Typical message: `` register read before it is defined: s2 in `s_mov_b32 s20, s2`; nothing in the kernel writes it and the launch does not initialize it (gfx1201 entry state: s[0:1] kernarg pointer, ttmp9 workgroup id x, ttmp7 = y | z<<16, v0 thread id) ``.

The usual mistake is carrying a gfx11 habit to gfx12: on gfx11 the workgroup id x arrives in `s2` (declared in the register plan), on gfx12 it is `ttmp9` and `s2` holds nothing. Read the register the architecture defines, or write the register (a `v_mov_b32` / `s_mov_b32`) somewhere in the kernel before it is read. The failing script reads `s2` and is shown for gfx1201 only; on gfx1151 it is accepted.

Fails:

```rip fails=R0604 arch=gfx1201
fn emit(spec: Spec) -> Result<Emitted, String> {
    let kspec = KernelSpec {
        kernel_id: "diag".into(), variant: "Default".into(), arch: spec.arch, symbol: "diag_kernel".into(),
        kernargs: KernargLayout::new(0), user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 1024, wave32: true, cu_mode: false,
    };
    let mut p = RegPlan::new(256, 104)?;
    for i in 0..4u8 { p.v::<8>("vectors", 8 * i, Live::Whole)?; }
    for i in 0..4u8 { p.s::<8>("scalars", 8 * i, Live::Whole)?; }
    let mut b = Builder::new(kspec, p);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    op(wg.isa(), "s_mov_b32 s20, s2", &[s(20)], &[s(2)])?;
    let end = wg.exit(".Lend")?;
    wg.end(end)?;
    b.finish()
}
```

Fixed:

```rip arch=gfx1201
fn emit(spec: Spec) -> Result<Emitted, String> {
    let kspec = KernelSpec {
        kernel_id: "diag".into(), variant: "Default".into(), arch: spec.arch, symbol: "diag_kernel".into(),
        kernargs: KernargLayout::new(0), user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 1024, wave32: true, cu_mode: false,
    };
    let mut p = RegPlan::new(256, 104)?;
    for i in 0..4u8 { p.v::<8>("vectors", 8 * i, Live::Whole)?; }
    for i in 0..4u8 { p.s::<8>("scalars", 8 * i, Live::Whole)?; }
    let mut b = Builder::new(kspec, p);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    op(wg.isa(), "s_mov_b32 s20, ttmp9", &[s(20)], &[])?;
    let end = wg.exit(".Lend")?;
    wg.end(end)?;
    b.finish()
}
```

## R0701: certification / assembly

Certification re-checks the emitted module at the object level: an independent wait-ledger replay, assembly and linking with the LLVM tools (`llvm-mc`, `ld.lld`), and the byte-exact lift of every kernel (M7). Any failure of these steps is reported as R0701. It can only come from `peacemaker-rip build --certify` or the library's `certify`; `check` never runs it.

Typical messages: `Default: wait replay: ...`, `module: assemble/link: ...`, `` kernel: M7 rejects the emitted object: ... ``.

A script cannot cause this on its own: everything a script can express is rejected earlier (R03xx, R04xx, R06xx), so R0701 signals a missing or broken toolchain, a certification mismatch in the builder, or a module-level problem (for example kernels of different architectures in one module). Check that the LLVM tools are installed and run `peacemaker-rip build --certify` again; if the tools work and the kernel was accepted by `check`, report it as a builder bug with the script.

Reproduce (needs the toolchain, not a script mistake):

```text
$ peacemaker-rip build --certify kernel.rip gfx1201 Default kernel.s
error[R0701]: module: assemble/link: <llvm-mc or ld.lld failure>
```

## R0901: tool error

The command-line tool could not do what was asked, independently of the script: a usage error, a file that cannot be read or written, a missing `rustfmt` for `peacemaker-rip fmt`, or the evaluator thread that could not start. No source position exists, so none is printed.

Typical messages: `usage: ...`, `` rustfmt not found on PATH: install it with `rustup component add rustfmt` ``, `x.rip: No such file or directory (os error 2)`.

I/O and formatter failures are reported as `error[R0901]` and make the tool exit with status 1; a command-line usage error is reported as plain `error: ...` plus the usage text and exits with status 2. No script can produce this code.

Reproduce (no script is involved):

```text
$ peacemaker-rip check missing.rip
error[R0901]: missing.rip: No such file or directory (os error 2)
$ peacemaker-rip fmt kernel.rip      # rustfmt not installed
error[R0901]: kernel.rip: rustfmt not found on PATH: install it with `rustup component add rustfmt`
```

## R0000: unclassified

A message arrived that none of the rules recognizes. It is reported verbatim with its position (when it has one) so nothing is lost; the code only says the classifier has no more specific explanation. The fallback exists so that every failure has a stable code in scripts and in `--json` output.

Native helpers of the kernel vocabulary refuse some arguments with a short message of their own. One that no rule claims is `SMEM width`, from `smem(builder, dst, len, base, off)` when `len` is not 1, 2, 4 or 8 dwords. Check the message and the surrounding source (here: use a supported width). If the failure deserves its own help, report the message so a rule can be added; the code does not change for messages that are already classified.

A native refusal that no rule claims, as it appears in the diagnostic (no script example: the code is the classifier's fallback, and this message is its only fixture):

```text
error[R0000]: SMEM width
```
