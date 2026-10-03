//! The standalone `.rip` front end.
//!
//! A `.rip` file is a Rust-shaped kernel script. This crate parses it at
//! run time (`syn`, no rustc), executes its `emit` function with a small
//! compile-time evaluator against the same generic builder the Rust DSL
//! uses (`hipfire_isa::Builder`, through `peacemaker_author`'s sealed
//! runtime driver), and returns the builder's `Emitted`. Editing a `.rip`
//! file and re-running needs no cargo rebuild.
//!
//! # Source contract
//!
//! - Entry: `fn emit(spec: Spec) -> Result<Emitted, String>`. [`compile`]
//!   injects `Spec { arch: Arch, variant: Str, kind }` where `kind` is the
//!   variant of the script's own `enum Kind` named like `variant`
//!   (case-insensitive), else the string itself. `fn emit(arch, variant)` is
//!   accepted too.
//! - Item keywords (token-level, see [`preprocess`]): `tile T { .. }` is a
//!   register-tile layout, `chain f(..) { .. }` an emission routine,
//!   `region R;` an LDS region tag.
//! - Builder vocabulary: [`native`]. Lowering of LDS regions, waits,
//!   barriers, branches and loops goes through
//!   `peacemaker_author::runtime::Driver`, which makes the same backend
//!   calls as the typed Rust core and checks dynamic ownership and scope;
//!   object certification ([`certify`]) is required before admission and
//!   runs the existing passes on the emitted code object.
//!
//! # The evaluated language
//!
//! Rust syntax, executed (not compiled). Supported: `const`/`static`,
//! `struct`/`enum` (unit and tuple variants, discriminants), `impl` and
//! `trait` default methods (looked up by the receiver's type, including the
//! native ones, so `impl PlanExt for RegPlan` works), closures with lexical
//! capture, `let`/`let else`/`if let`/`while let`, `match` with guards,
//! ranges, or-patterns and bindings, `for`/`while`/`loop` with labels,
//! `?`, `&mut` places and field/index assignment, `as` casts, exact integer
//! types (unsuffixed literals adopt their operand's type; overflow and
//! mixed types are errors), tuples, arrays / `Vec`, `Option`/`Result`, the
//! common iterator adaptors (materialized), `String`, and the macros
//! `format!`, `vec!`, `matches!`, `assert*!`, `panic!`, `println!`.
//! Generics, lifetimes and type annotations are accepted and ignored
//! (integers excepted); `use` and unused items are never examined. Anything
//! executed that is not supported is an error with `file:line:col`, never a
//! silent no-op.
//!
//! `let mut wg = Workgroup::new(&mut b, arch)?;` opens the sealed driver
//! over the builder `b` for the rest of its block, up to the first
//! statement that mentions `b` again (typically `b.finish()`).

pub mod builtin;
pub mod driver;
pub mod expand;
pub mod fmt;
pub mod interp;
pub mod native;
pub mod ops;
pub mod preprocess;
pub mod value;

use hipfire_isa::kernels::iu4_gemm::ModuleProof;
use hipfire_isa::{Arch, Emitted};
use interp::Interp;
use value::Value;

/// Parse a `.rip` source into a Rust syntax tree (keywords rewritten,
/// macros elaborated). Errors are `name:line:col: message`.
pub fn parse(source: &str, name: &str) -> Result<syn::File, String> {
    let ts = preprocess::rewrite(preprocess::lex(source).map_err(|e| format!("{name}:{e}"))?);
    let mut file = syn::parse2::<syn::File>(ts).map_err(|e| {
        let s = e.span().start();
        format!("{name}:{}:{}: {e}", s.line, s.column + 1)
    })?;
    expand::expand_file(&mut file);
    Ok(file)
}

/// Run `f` on a thread with a stack deep enough for the tree-walking
/// evaluator. A panic inside the evaluator is reported as an error.
fn big_stack<T: Send>(f: impl FnOnce() -> Result<T, String> + Send) -> Result<T, String> {
    std::thread::scope(|s| {
        std::thread::Builder::new()
            .name("rip-eval".into())
            .stack_size(256 << 20)
            .spawn_scoped(s, f)
            .map_err(|e| format!("cannot start the evaluator thread: {e}"))
            .and_then(|h| h.join().map_err(|p| format!("the evaluator panicked: {}", p.downcast_ref::<String>().map(String::as_str).or_else(|| p.downcast_ref::<&str>().copied()).unwrap_or("unknown"))))
    })
    .and_then(|r| r)
}

fn spec_value<'a>(it: &Interp<'a>, arch: Arch, variant: &str) -> Value<'a> {
    let kind = match it.enum_has_variant("Kind", variant) {
        Some(v) => Value::Enum("Kind".into(), v.as_str().into(), vec![]),
        None => Value::str(variant),
    };
    Value::Struct(
        "Spec".into(),
        vec![("arch".into(), Value::Nat(native::Nat::Arch(arch))), ("variant".into(), Value::str(variant)), ("kind".into(), kind)],
    )
}

fn emitted_of(v: Value) -> Result<Emitted, String> {
    match v {
        Value::Enum(t, var, mut p) if &*t == "Result" => {
            let payload = p.pop().unwrap_or(Value::Unit);
            if &*var == "Ok" {
                match payload {
                    Value::Nat(native::Nat::Emitted(e)) => Ok(std::rc::Rc::try_unwrap(e).unwrap_or_else(|rc| (*rc).clone())),
                    o => Err(format!("`emit` returned Ok({}), not an Emitted (finish the builder: `b.finish()`)", o.type_name())),
                }
            } else {
                Err(fmt::display(&payload).unwrap_or_else(|_| fmt::debug(&payload, false)))
            }
        }
        o => Err(format!("`emit` must return Result<Emitted, String>, found {}", o.type_name())),
    }
}

fn emit_one<'a>(it: &Interp<'a>, arch: Arch, variant: &str) -> Result<Emitted, String> {
    let args = match it.fn_arity("emit") {
        Some(1) => vec![spec_value(it, arch, variant)],
        Some(2) => vec![Value::Nat(native::Nat::Arch(arch)), Value::str(variant)],
        Some(n) => return Err(format!("`emit` takes (spec) or (arch, variant), not {n} arguments")),
        None => return Err("no `fn emit(spec: Spec) -> Result<Emitted, String>` entry point".into()),
    };
    emitted_of(it.call_entry("emit", args)?)
}

/// Compile one variant of a `.rip` source for `arch`.
pub fn compile(source: &str, arch: Arch, variant: &str) -> Result<Emitted, String> {
    compile_named(source, "rip", arch, variant)
}

/// [`compile`] with the source's name for error positions (a file path).
pub fn compile_named(source: &str, name: &str, arch: Arch, variant: &str) -> Result<Emitted, String> {
    big_stack(|| {
        let file = parse(source, name)?;
        let it = Interp::load(&file, name);
        emit_one(&it, arch, variant)
    })
}

/// Compile several variants and merge them into one assembly module.
pub fn module(source: &str, arch: Arch, variants: &[&str], module_name: &str) -> Result<(Vec<Emitted>, String, ModuleProof), String> {
    module_named(source, "rip", arch, variants, module_name)
}

pub fn module_named(source: &str, name: &str, arch: Arch, variants: &[&str], module_name: &str) -> Result<(Vec<Emitted>, String, ModuleProof), String> {
    let emitted = big_stack(|| -> Result<Vec<Emitted>, String> {
        let file = parse(source, name)?;
        let it = Interp::load(&file, name);
        variants.iter().map(|v| emit_one(&it, arch, v)).collect()
    })?;
    let (text, proof) = hipfire_isa::kernels::iu4_gemm::module(&emitted, module_name)?;
    Ok((emitted, text, proof))
}

/// The kernel symbol of an emitted program (`.amdhsa_kernel <name>`).
pub fn symbol_of(e: &Emitted) -> Result<String, String> {
    e.s_text
        .lines()
        .find_map(|l| l.trim().strip_prefix(".amdhsa_kernel "))
        .map(|s| s.trim().to_string())
        .ok_or_else(|| "emitted assembly has no `.amdhsa_kernel` symbol".to_string())
}

#[cfg(feature = "certify")]
pub use certify::{certify, Certificate};

#[cfg(feature = "certify")]
mod certify {
    use super::*;
    use hipfire_isa::toolchain::{build, Toolchain};
    use hipfire_isa::{ledger_replay, pm_check};
    use std::path::Path;

    /// What certification proved about one symbol of a module.
    #[derive(Debug)]
    pub struct Certificate {
        pub symbol: String,
        /// `pm_check::m7` report: lift identity and (empty) obligations.
        pub m7: serde_json::Value,
    }

    /// Certify an emitted module at the object level, with the existing
    /// passes: independent wait-ledger replay of every kernel's assembly,
    /// assemble + link (`llvm-mc`, `ld.lld`) into `dir`, then M7 on each
    /// symbol of the linked ELF (byte-exact lift, no wait / hazard /
    /// barrier / definedness obligations). `text` is the module assembly
    /// (`module(..).1`; for one kernel, its `s_text`).
    pub fn certify(emitted: &[Emitted], text: &str, arch: Arch, module_name: &str, dir: &Path) -> Result<Vec<Certificate>, String> {
        for e in emitted {
            ledger_replay::replay_waits(&e.s_text, arch).map_err(|err| format!("{}: wait replay: {err}", e.proof.variant))?;
        }
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let s = dir.join(format!("{module_name}.s"));
        std::fs::write(&s, text).map_err(|e| format!("{}: {e}", s.display()))?;
        let build = build(&Toolchain::default(), &s, &dir.join(format!("{module_name}.hsaco")), arch.name()).map_err(|e| format!("{module_name}: assemble/link: {e}"))?;
        let mut out = Vec::new();
        for e in emitted {
            let symbol = symbol_of(e)?;
            let m7 = pm_check::m7(&build.elf, arch.name(), &symbol).map_err(|err| format!("{symbol}: {err}"))?;
            if m7["lift"].as_str() != Some("byte-exact") || m7["obligations"] != serde_json::json!({}) {
                return Err(format!("{symbol}: M7 rejects the emitted object: {m7}"));
            }
            out.push(Certificate { symbol, m7 });
        }
        Ok(out)
    }
}

/// Evaluate the zero-argument function `entry` of a script and render its
/// value (`Debug`). A tool for tests and for checking helpers of a `.rip`
/// without building a kernel; faults come back as `name:line:col: message`.
pub fn eval(source: &str, entry: &str) -> Result<String, String> {
    big_stack(|| {
        let file = parse(source, "rip")?;
        let it = Interp::load(&file, "rip");
        it.call_entry(entry, vec![]).map(|v| fmt::debug(&v, false))
    })
}
