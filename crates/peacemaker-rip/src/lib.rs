//! The standalone `.rip` front end.
//!
//! A `.rip` file is a Rust-shaped kernel script. This crate parses it at
//! run time (`syn`, no rustc), executes its `emit` function with a small
//! compile-time evaluator against the same generic builder the Rust DSL
//! uses (`peacemaker_isa::Builder`, through `peacemaker_author`'s sealed
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
//! - Discovery: `const VARIANTS: &[&str]` names the channel's variants
//!   (else the script's `enum Kind`), `const ARCHES: &[&str]` its arches
//!   (else the default arch list). Explicit `--variant` / `--arch` flags
//!   beat both consts; `--arch all` means the declared `ARCHES` when
//!   present, else every arch.
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
//!
//! # Offline features
//!
//! [`features::collect`] and `peacemaker-rip features FILE [--json]` report
//! shared [`check`] resources, linked-object schedule costs, and instruction-by-
//! instruction emulator coverage without a GPU. Source and assembly SHA256
//! receipts identify the inputs. Optional `--routes FILE` supplies the original
//! cost-lint routes JSON; unavailable route-dependent facts remain null, and
//! compile, link, lift, or routes errors are structured per-target failures.
//!
//! # Offline execution
//!
//! [`run`] implements `peacemaker-rip run FILE --case FILE.json`: compile,
//! assemble natively, lift, execute the declared grid on the CPU emulator and
//! check every buffer, including input immutability and output guards. See its
//! module documentation for the strict case schema, reports and behavior dumps.

pub mod builtin;
pub mod diag;
pub mod driver;
pub mod expand;
pub mod features;
pub mod fmt;
pub mod interp;
pub mod native;
pub mod ops;
pub mod preprocess;
pub mod run;
pub mod srcfmt;
pub mod validate;
pub mod value;

use interp::Interp;
use peacemaker_isa::kernels::iu4_gemm::ModuleProof;
use peacemaker_isa::{Arch, Emitted};
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
            .and_then(|h| {
                h.join().map_err(|p| {
                    format!(
                        "the evaluator panicked: {}",
                        p.downcast_ref::<String>()
                            .map(String::as_str)
                            .or_else(|| p.downcast_ref::<&str>().copied())
                            .unwrap_or("unknown")
                    )
                })
            })
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
        vec![
            ("arch".into(), Value::Nat(native::Nat::Arch(arch))),
            ("variant".into(), Value::str(variant)),
            ("kind".into(), kind),
        ],
    )
}

/// An `emit` failure: the located message, and whether it is the script's own
/// `Err(..)` (bare when returned, so located here at the name of `fn emit`)
/// rather than a native error or an entry-contract violation.
type EmitError = (String, bool);

/// The value `emit` returned as an `Emitted`. A native error that reached the
/// result through `?` is already located and passes through; the script's own
/// bare `Err(..)` and every contract violation are located at the name of `fn emit`.
fn emitted_of(it: &Interp, v: Value) -> Result<Emitted, EmitError> {
    let at = it.fn_loc("emit");
    match v {
        Value::Enum(t, var, mut p) if &*t == "Result" => {
            let payload = p.pop().unwrap_or(Value::Unit);
            if &*var == "Ok" {
                match payload {
                    Value::Nat(native::Nat::Emitted(e)) => Ok(std::rc::Rc::try_unwrap(e).unwrap_or_else(|rc| (*rc).clone())),
                    o => Err((format!("{at}: `emit` returned Ok({}), not an Emitted (finish the builder: `b.finish()`)", o.type_name()), false)),
                }
            } else {
                let e = fmt::display(&payload).unwrap_or_else(|_| fmt::debug(&payload, false));
                if e.starts_with(&format!("{}:", it.name)) {
                    Err((e, false))
                } else {
                    Err((format!("{at}: {e}"), true))
                }
            }
        }
        o => Err((
            format!(
                "{at}: `emit` must return Result<Emitted, String>, found {}",
                o.type_name()
            ),
            false,
        )),
    }
}

fn emit_one_raw<'a>(it: &Interp<'a>, arch: Arch, variant: &str) -> Result<Emitted, EmitError> {
    let at = it.fn_loc("emit");
    let args = match it.fn_arity("emit") {
        Some(1) => vec![spec_value(it, arch, variant)],
        Some(2) => vec![Value::Nat(native::Nat::Arch(arch)), Value::str(variant)],
        Some(n) => {
            return Err((
                format!("{at}: `emit` takes (spec) or (arch, variant), not {n} arguments"),
                false,
            ))
        }
        None => {
            return Err((
                format!("{at}: no `fn emit(spec: Spec) -> Result<Emitted, String>` entry point"),
                false,
            ))
        }
    };
    emitted_of(it, it.call_entry("emit", args).map_err(|e| (e, false))?)
}

fn emit_one<'a>(it: &Interp<'a>, arch: Arch, variant: &str) -> Result<Emitted, String> {
    let emitted = emit_one_raw(it, arch, variant).map_err(|(m, _)| m)?;
    validate::emitted(arch, &emitted.s_text, it.take_instruction_sites())
        .map_err(|errors| errors.join("\n"))?;
    Ok(emitted)
}

/// Compile one variant of a `.rip` source for `arch`.
pub fn compile(source: &str, arch: Arch, variant: &str) -> Result<Emitted, String> {
    compile_named(source, "rip", arch, variant)
}

/// [`compile`] with the source's name for error positions (a file path).
pub fn compile_named(
    source: &str,
    name: &str,
    arch: Arch,
    variant: &str,
) -> Result<Emitted, String> {
    big_stack(|| {
        let file = parse(source, name)?;
        let it = Interp::load(&file, name);
        emit_one(&it, arch, variant)
    })
}

/// Resources of one emitted kernel, read from its descriptor and body.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Resources {
    pub vgprs: u32,
    pub sgprs: u32,
    pub lds_bytes: u32,
    /// Instruction lines of the kernel body (waits and hazard guards included).
    pub instructions: usize,
}

/// The outcome of [`check`]: the compiled variant and every diagnostic.
#[derive(Clone, Debug)]
pub struct Checked {
    pub arch: Arch,
    pub variant: String,
    /// `Some` iff there is no error.
    pub emitted: Option<Emitted>,
    /// From the emitted descriptor (`.amdhsa_next_free_vgpr/sgpr`, group segment size).
    pub resources: Option<Resources>,
    /// Errors and warnings, in source order.
    pub diagnostics: Vec<diag::Diagnostic>,
}

fn descriptor_int(text: &str, key: &str) -> Option<u32> {
    text.lines()
        .find_map(|l| l.trim().strip_prefix(key)?.trim().parse().ok())
}

/// Wave size declared by the emitted kernel, never inferred from its architecture.
pub(crate) fn wave_size_of(e: &Emitted) -> Option<u32> {
    match descriptor_int(&e.s_text, ".amdhsa_wavefront_size32 ")? {
        0 => Some(64),
        1 => Some(32),
        _ => None,
    }
}

fn resources_of(e: &Emitted) -> Option<Resources> {
    let symbol = symbol_of(e).ok()?;
    let label = format!("{symbol}:");
    let end = format!(".L{symbol}_end:");
    let mut body = e
        .s_text
        .lines()
        .skip_while(|l| l.trim() != label)
        .skip(1)
        .take_while(|l| l.trim() != end);
    let instructions = body
        .by_ref()
        .map(str::trim)
        .filter(|l| {
            !l.is_empty()
                && !l.starts_with('.')
                && !l.starts_with(';')
                && !l.starts_with("//")
                && !l.ends_with(':')
        })
        .count();
    Some(Resources {
        vgprs: descriptor_int(&e.s_text, ".amdhsa_next_free_vgpr ")?,
        sgprs: descriptor_int(&e.s_text, ".amdhsa_next_free_sgpr ")?,
        lds_bytes: descriptor_int(&e.s_text, ".amdhsa_group_segment_fixed_size ")?,
        instructions,
    })
}

/// Compile one variant and validate its final bytes; all errors come back as diagnostics.
/// [`compile_named`] enforces the same validation and returns its errors as strings.
pub fn check(source: &str, name: &str, arch: Arch, variant: &str) -> Checked {
    let run = big_stack(
        || -> Result<(Result<Emitted, EmitError>, Vec<(String, String)>), String> {
            let file = parse(source, name)?;
            let it = Interp::load(&file, name);
            let r = emit_one_raw(&it, arch, variant);
            Ok((r, it.take_instruction_sites()))
        },
    );
    let (result, sites) = match run {
        Ok(pair) => pair,
        Err(e) => (Err((e, false)), Vec::new()),
    };
    let mut diagnostics = Vec::new();
    let emitted = match result {
        Ok(e) => match validate::emitted(arch, &e.s_text, sites) {
            Ok(()) => Some(e),
            Err(errors) => {
                diagnostics.extend(errors.iter().map(|error| {
                    diag::Diagnostic::from_raw(error, name, source, diag::Severity::Error)
                }));
                None
            }
        },
        Err((e, script)) => {
            let mut d = diag::Diagnostic::from_raw(&e, name, source, diag::Severity::Error);
            if script {
                // `from_raw` cannot tell the script's own `Err(..)` from any other bare
                // message; only the entry plumbing knows it was returned unlocated.
                d.code = "R0502";
                d.help = Some("the script's `emit` returned this `Err` itself: fix the condition it reports in the script".into());
                d.suggestion = None;
                // `from_raw` widens a bare position to the call there; the position is the
                // name of `fn emit`, so the span is exactly that identifier.
                if let Some(s) = d.span {
                    d.span = Some(diag::Span {
                        line: s.line,
                        col: s.col,
                        end_line: s.line,
                        end_col: s.col + "emit".len(),
                    });
                }
            }
            diagnostics.push(d);
            None
        }
    };
    diagnostics.sort_by_key(|d| d.span.map_or((usize::MAX, 0), |s| (s.line, s.col)));
    let resources = emitted.as_ref().and_then(resources_of);
    Checked {
        arch,
        variant: variant.to_string(),
        emitted,
        resources,
        diagnostics,
    }
}

/// Check every (arch, variant) of a `.rip` source without cargo: the default
/// arches and variants below, unless the caller passes explicit lists.
///
/// - `const VARIANTS: &[&str]`: the channel's variant names. When absent,
///   the variants of the script's own `enum Kind` (case-insensitive at the
///   call site), else an empty list. An empty list is a usage error naming
///   `VARIANTS`: pass `--variant`.
/// - `const ARCHES: &[&str]`: the channel's architectures (any of
///   `gfx1100`, `gfx1151`, `gfx1201`). When absent, the default arch list.
///   An unknown name is a usage error; `--arch all` means the declared
///   `ARCHES` when present, else all three arches.
/// - Explicit `--arch` / `--variant` flags beat both consts. A refusal the
///   script's own `emit` returns (an arch or variant it does not build) is
///   an error, even when the arch came from an explicit flag.
pub const DEFAULT_ARCHES: [Arch; 2] = [Arch::Gfx1151, Arch::Gfx1201];

/// Every arch a `--arch` list or `--arch all` may name.
pub const ALL_ARCHES: [Arch; 3] = [Arch::Gfx1100, Arch::Gfx1151, Arch::Gfx1201];

/// Parse one arch name of a `--arch` list or a script's `const ARCHES`.
pub fn parse_arch(name: &str) -> Result<Arch, String> {
    name.parse::<Arch>()
        .map_err(|_| format!("unsupported architecture {name}"))
}

/// The variants declared by the script's `const VARIANTS`, else by its `enum
/// Kind` with the declared spelling ([`compile`] matches them
/// case-insensitively); empty only when the script declares neither.
pub fn variants(source: &str, name: &str) -> Result<Vec<String>, String> {
    big_stack(|| {
        let file = parse(source, name)?;
        let it = Interp::load(&file, name);
        match it.const_str_list("VARIANTS")? {
            Some(v) => Ok(v),
            None => Ok(it.enum_variants("Kind").unwrap_or_default()),
        }
    })
}

/// The arches declared by the script's `const ARCHES` (`None` when it
/// declares none); a declared-but-malformed const is an error.
pub fn arches(source: &str, name: &str) -> Result<Option<Vec<String>>, String> {
    big_stack(|| {
        let file = parse(source, name)?;
        let it = Interp::load(&file, name);
        it.const_str_list("ARCHES")
    })
}

/// Compile several variants and merge them into one assembly module.
pub fn module(
    source: &str,
    arch: Arch,
    variants: &[&str],
    module_name: &str,
) -> Result<(Vec<Emitted>, String, ModuleProof), String> {
    module_named(source, "rip", arch, variants, module_name)
}

pub fn module_named(
    source: &str,
    name: &str,
    arch: Arch,
    variants: &[&str],
    module_name: &str,
) -> Result<(Vec<Emitted>, String, ModuleProof), String> {
    let emitted = big_stack(|| -> Result<Vec<Emitted>, String> {
        let file = parse(source, name)?;
        let it = Interp::load(&file, name);
        variants.iter().map(|v| emit_one(&it, arch, v)).collect()
    })?;
    let (text, proof) = peacemaker_isa::kernels::iu4_gemm::module(&emitted, module_name)?;
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
    use peacemaker_isa::toolchain::{build, Toolchain};
    use peacemaker_isa::{ledger_replay, pm_check};
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
    pub fn certify(
        emitted: &[Emitted],
        text: &str,
        arch: Arch,
        module_name: &str,
        dir: &Path,
    ) -> Result<Vec<Certificate>, String> {
        for e in emitted {
            ledger_replay::replay_waits(&e.s_text, arch)
                .map_err(|err| format!("{}: wait replay: {err}", e.proof.variant))?;
        }
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let s = dir.join(format!("{module_name}.s"));
        std::fs::write(&s, text).map_err(|e| format!("{}: {e}", s.display()))?;
        let build = build(
            &Toolchain::default(),
            &s,
            &dir.join(format!("{module_name}.hsaco")),
            arch.name(),
        )
        .map_err(|e| format!("{module_name}: assemble/link: {e}"))?;
        let mut out = Vec::new();
        for e in emitted {
            let symbol = symbol_of(e)?;
            let m7 = pm_check::m7(&build.elf, arch.name(), &symbol)
                .map_err(|err| format!("{symbol}: {err}"))?;
            if m7["lift"].as_str() != Some("byte-exact")
                || m7["obligations"] != serde_json::json!({})
            {
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
