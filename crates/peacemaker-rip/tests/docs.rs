//! The `.rip` documentation is executable. Every fenced block of
//! `docs/REFERENCE.md` and `docs/cookbook/*.md` that carries one of the info
//! strings below is checked against the real front end, so the prose cannot
//! drift from the parser.
//!
//! | info string                         | check |
//! |-------------------------------------|-------|
//! | `rip`                               | `fn emit(spec: Spec)`; `peacemaker_rip::check` has no error on gfx1151 and gfx1201 (variant `default`) |
//! | `rip variant=NAME`                  | same, variant `NAME` |
//! | `rip arch=gfx1151`                  | same, that one arch (`gfx1100`, `gfx1151`, `gfx1201` accepted) |
//! | `rip eval`                          | `peacemaker_rip::eval(src, "main")` succeeds |
//! | `rip fails=R0xxx`                   | `check` reports that code as its first error, else (no error) its first warning; both arches unless `arch=` |
//! | `rip body`                          | statements spliced into [`BODY_TEMPLATE`] at `/*BODY*/`, compiled for both arches |
//! | `mnemonics`                         | `gfxNNNN: text` lines must parse and encode (`validate::effects`, then `validate::instruction` declaring those effects); `!gfxNNNN: text` must be rejected |
//! | anything else (`text`, `sh`, ...)   | not checked |
//!
//! `variant=`, `arch=` and `fails=` combine with `rip` and `rip body`; `eval`
//! stands alone. Any other word after `rip` is an error, so a typo cannot
//! silently disable a block. Failures are reported as `file:line` of the
//! opening fence (and of the offending line inside the block where the
//! diagnostic carries a span).

use peacemaker_isa::Arch;
use peacemaker_rip::diag::{self, Severity};
use std::path::{Path, PathBuf};

/// The wrapper every `rip body` block is spliced into (shown once in
/// `docs/REFERENCE.md`). It declares: kernargs `a` (offset 0) and `b` (8);
/// SGPR plan `kernarg_ptr` s0:1, `wg_x` s2 (gfx11 only), `args` s4:7, `srd`
/// s8:11, `t` s12:15; VGPR plan `v` v0:7, `w` v8:15; workgroup size 64 and
/// 4096 bytes of LDS.
const BODY_TEMPLATE: &str = r#"fn emit(spec: Spec) -> Result<Emitted, String> {
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
"#;

const BODY_MARK: &str = "/*BODY*/";

// ---------------------------------------------------------------- parsing

#[derive(Debug, PartialEq)]
enum Kind {
    /// `rip` (and `rip body`): compiled by `check`.
    Rip {
        body: bool,
        fails: Option<String>,
    },
    Eval,
    Mnemonics,
    Other,
}

#[derive(Debug)]
struct Block {
    /// 1-based line of the opening fence.
    line: usize,
    info: String,
    kind: Kind,
    variant: String,
    arch: Option<Arch>,
    /// The block's text, lines joined by `\n`, with a trailing newline.
    text: String,
    /// A malformed info string.
    bad_info: Option<String>,
}

fn parse_arch(name: &str) -> Option<Arch> {
    [Arch::Gfx1100, Arch::Gfx1151, Arch::Gfx1201]
        .into_iter()
        .find(|a| a.name() == name)
}

fn is_code(s: &str) -> bool {
    s.len() == 5 && s.starts_with('R') && s[1..].bytes().all(|b| b.is_ascii_digit())
}

fn classify(info: &str) -> (Kind, String, Option<Arch>, Option<String>) {
    let mut words = info.split_whitespace();
    let lang = words.next().unwrap_or("");
    let rest: Vec<&str> = words.collect();
    let mut variant = "default".to_string();
    let mut arch = None;
    let mut bad = None;
    let kind = match lang {
        "mnemonics" => {
            if !rest.is_empty() {
                bad = Some(format!(
                    "`mnemonics` takes no attributes, found `{}`",
                    rest.join(" ")
                ));
            }
            Kind::Mnemonics
        }
        "rip" => {
            let (mut body, mut eval, mut fails) = (false, false, None);
            for w in &rest {
                if *w == "body" {
                    body = true;
                } else if *w == "eval" {
                    eval = true;
                } else if let Some(c) = w.strip_prefix("fails=") {
                    if is_code(c) {
                        fails = Some(c.to_string());
                    } else {
                        bad = Some(format!(
                            "`fails={c}` is not an error code (R followed by four digits)"
                        ));
                    }
                } else if let Some(v) = w.strip_prefix("variant=") {
                    if v.is_empty() {
                        bad = Some("`variant=` needs a name".into());
                    }
                    variant = v.to_string();
                } else if let Some(a) = w.strip_prefix("arch=") {
                    match parse_arch(a) {
                        Some(a) => arch = Some(a),
                        None => {
                            bad = Some(format!("`arch={a}`: expected gfx1100, gfx1151 or gfx1201"))
                        }
                    }
                } else {
                    bad = Some(format!("unknown attribute `{w}` on a `rip` block (known: eval, body, fails=R0xxx, variant=NAME, arch=gfxNNNN)"));
                }
            }
            if eval && (body || fails.is_some() || arch.is_some() || variant != "default") {
                bad = Some("`eval` cannot be combined with other attributes".into());
            }
            if eval {
                Kind::Eval
            } else {
                Kind::Rip { body, fails }
            }
        }
        _ => Kind::Other,
    };
    (kind, variant, arch, bad)
}

/// Fenced blocks of a markdown document. A fence is a line whose first
/// non-blank text is three or more backticks; it closes at the first line
/// that is only backticks, at least as many as the opening run.
fn blocks(md: &str) -> Vec<Block> {
    let mut out = Vec::new();
    let mut open: Option<(usize, usize, String, Vec<&str>)> = None; // (line, ticks, info, lines)
    for (i, raw) in md.lines().enumerate() {
        let t = raw.trim();
        let ticks = t.bytes().take_while(|&b| b == b'`').count();
        match &mut open {
            None => {
                if ticks >= 3 {
                    open = Some((i + 1, ticks, t[ticks..].trim().to_string(), Vec::new()));
                }
            }
            Some((line, n, info, lines)) => {
                if ticks >= *n && t.len() == ticks {
                    let (kind, variant, arch, bad_info) = classify(info);
                    let mut text = lines.join("\n");
                    text.push('\n');
                    out.push(Block {
                        line: *line,
                        info: info.clone(),
                        kind,
                        variant,
                        arch,
                        text,
                        bad_info,
                    });
                    open = None;
                } else {
                    lines.push(raw);
                }
            }
        }
    }
    if let Some((line, ..)) = open {
        out.push(Block {
            line,
            info: String::new(),
            kind: Kind::Other,
            variant: "default".into(),
            arch: None,
            text: String::new(),
            bad_info: Some("unterminated code fence".into()),
        });
    }
    out
}

// ---------------------------------------------------------------- checking

fn rip_arches(b: &Block) -> Vec<Arch> {
    match b.arch {
        Some(a) => vec![a],
        None => vec![Arch::Gfx1151, Arch::Gfx1201],
    }
}

/// Splice a `rip body` block into the template; returns the source and the
/// 1-based template line of the `/*BODY*/` marker.
fn splice(body: &str) -> (String, usize) {
    let at = BODY_TEMPLATE
        .find(BODY_MARK)
        .expect("template has a body marker");
    let marker_line = BODY_TEMPLATE[..at].matches('\n').count() + 1;
    (
        BODY_TEMPLATE.replacen(BODY_MARK, body.trim_end_matches('\n'), 1),
        marker_line,
    )
}

/// `name` is also what diagnostics print; it carries no colon so positions
/// stay parseable.
fn check_rip(doc: &str, b: &Block, errors: &mut Vec<String>) {
    let (body, fails) = match &b.kind {
        Kind::Rip { body, fails } => (*body, fails.as_deref()),
        _ => unreachable!(),
    };
    let (src, marker_line) = if body {
        splice(&b.text)
    } else {
        (b.text.clone(), 0)
    };
    let name = format!("block-L{}", b.line);
    for arch in rip_arches(b) {
        let checked = peacemaker_rip::check(&src, &name, arch, &b.variant);
        // The first error; for a `fails=` block with no error, the first warning (`R0603`-style codes).
        let first = checked
            .diagnostics
            .iter()
            .find(|d| d.severity == Severity::Error)
            .or_else(|| {
                if fails.is_some() {
                    checked.diagnostics.first()
                } else {
                    None
                }
            });
        // Position of a diagnostic in the document, when it points into the block's own text.
        let at = |d: &diag::Diagnostic| -> String {
            let Some(span) = d.span else {
                return format!("{doc}:{}", b.line);
            };
            let inner = if body {
                (span.line + 1)
                    .checked_sub(marker_line)
                    .filter(|&l| l >= 1 && l <= b.text.lines().count().max(1))
            } else {
                Some(span.line)
            };
            match inner {
                Some(l) => format!("{doc}:{}", b.line + l),
                None => format!("{doc}:{} (template line {})", b.line, span.line),
            }
        };
        match (fails, first) {
            (None, None) => {}
            (None, Some(d)) => errors.push(format!(
                "{} [{}, variant {}]: block `{}` does not compile\n{}",
                at(d),
                arch.name(),
                b.variant,
                b.info,
                d.render(&src)
            )),
            (Some(code), Some(d)) if d.code == code => {}
            (Some(code), Some(d)) => errors.push(format!(
                "{}:{} [{}]: expected first diagnostic {code} (first error, else first warning), got {}\n{}",
                doc,
                b.line,
                arch.name(),
                d.code,
                d.render(&src)
            )),
            (Some(code), None) => errors.push(format!("{doc}:{} [{}]: expected {code}, but the block compiles without errors or warnings", b.line, arch.name())),
        }
    }
}

fn check_eval(doc: &str, b: &Block, errors: &mut Vec<String>) {
    if let Err(e) = peacemaker_rip::eval(&b.text, "main") {
        errors.push(format!("{doc}:{}: `rip eval` block failed: {e}", b.line));
    }
}

/// Does the instruction text parse and encode for `arch`? `validate::instruction` also rejects
/// declared defs/uses that omit a register effect, which is not what a mnemonic line proves, so the
/// table's own effects are declared (`validate::effects`) and only parse/encode faults remain.
fn mnemonic_verdict(arch: Arch, text: &str) -> Result<(), String> {
    let (defs, uses) = peacemaker_rip::validate::effects(arch, text)?;
    peacemaker_rip::validate::instruction(arch, text, &defs, &uses)
}

fn check_mnemonics(doc: &str, b: &Block, errors: &mut Vec<String>) {
    for (i, raw) in b.text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("//") {
            continue;
        }
        let at = format!("{doc}:{}", b.line + 1 + i);
        let (negated, rest) = match line.strip_prefix('!') {
            Some(r) => (true, r.trim_start()),
            None => (false, line),
        };
        let Some((arch_name, text)) = rest.split_once(':') else {
            errors.push(format!(
                "{at}: mnemonics line must be `gfxNNNN: <instruction>`: `{line}`"
            ));
            continue;
        };
        let Some(arch) = parse_arch(arch_name.trim()) else {
            errors.push(format!(
                "{at}: unknown architecture `{}` (gfx1100, gfx1151 or gfx1201)",
                arch_name.trim()
            ));
            continue;
        };
        let text = text.trim();
        match (mnemonic_verdict(arch, text), negated) {
            (Ok(()), false) | (Err(_), true) => {}
            (Err(e), false) => errors.push(format!(
                "{at}: `{text}` is rejected for {}: {e}",
                arch.name()
            )),
            (Ok(()), true) => errors.push(format!(
                "{at}: `{text}` is accepted for {}, but the line is marked `!`",
                arch.name()
            )),
        }
    }
}

/// Check every block of one document; returns the failures.
fn check_doc(path: &Path, label: &str) -> Vec<String> {
    let md = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut errors = Vec::new();
    let mut checked = 0usize;
    for b in blocks(&md) {
        if let Some(why) = &b.bad_info {
            errors.push(format!("{label}:{}: {why}", b.line));
            continue;
        }
        match &b.kind {
            Kind::Rip { .. } => check_rip(label, &b, &mut errors),
            Kind::Eval => check_eval(label, &b, &mut errors),
            Kind::Mnemonics => check_mnemonics(label, &b, &mut errors),
            Kind::Other => continue,
        }
        checked += 1;
    }
    eprintln!("{label}: {checked} checked block(s)");
    errors
}

fn docs_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("docs")
}

fn report(errors: Vec<String>) {
    assert!(
        errors.is_empty(),
        "{} documentation block(s) failed:\n\n{}",
        errors.len(),
        errors.join("\n\n")
    );
}

// ---------------------------------------------------------------- tests

#[test]
fn reference_blocks_compile() {
    let path = docs_dir().join("REFERENCE.md");
    if !path.exists() {
        eprintln!("note: {} does not exist; skipped", path.display());
        return;
    }
    report(check_doc(&path, "docs/REFERENCE.md"));
}

/// The worked examples of `docs/ERRORS.md` are blocks too.
#[test]
fn errors_md_blocks_compile() {
    let path = docs_dir().join("ERRORS.md");
    if !path.exists() {
        eprintln!("note: {} does not exist; skipped", path.display());
        return;
    }
    report(check_doc(&path, "docs/ERRORS.md"));
}

/// Every `docs/cookbook/*.md`, walked at run time (so new recipes are covered
/// without editing this file). Each failure names its `file:line`.
#[test]
fn cookbook_blocks_compile_and_run() {
    let dir = docs_dir().join("cookbook");
    let Ok(rd) = std::fs::read_dir(&dir) else {
        eprintln!("note: {} does not exist; skipped", dir.display());
        return;
    };
    let mut files: Vec<PathBuf> = rd
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "md"))
        .collect();
    files.sort();
    if files.is_empty() {
        eprintln!("note: {} has no .md files; skipped", dir.display());
    }
    let mut errors = Vec::new();
    for f in files {
        let label = format!("docs/cookbook/{}", f.file_name().unwrap().to_string_lossy());
        errors.extend(check_doc(&f, &label));
    }
    report(errors);
}

/// The `## Rxxxx` sections of `docs/ERRORS.md` and the registered codes agree.
#[test]
fn errors_md_matches_registered_codes() {
    let path = docs_dir().join("ERRORS.md");
    let md = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut documented: Vec<String> = md
        .lines()
        .filter_map(|l| l.strip_prefix("## "))
        .filter_map(|h| h.split(|c: char| c == ':' || c.is_whitespace()).next())
        .filter(|c| is_code(c))
        .map(str::to_string)
        .collect();
    let mut registered: Vec<String> = diag::codes()
        .into_iter()
        .map(|c| c.code.to_string())
        .collect();
    documented.sort();
    registered.sort();
    let dup: Vec<_> = documented
        .windows(2)
        .filter(|w| w[0] == w[1])
        .map(|w| w[0].clone())
        .collect();
    assert!(
        dup.is_empty(),
        "docs/ERRORS.md documents {dup:?} more than once"
    );
    let missing_in_code: Vec<_> = documented
        .iter()
        .filter(|c| !registered.contains(c))
        .collect();
    let missing_in_docs: Vec<_> = registered
        .iter()
        .filter(|c| !documented.contains(c))
        .collect();
    assert!(
        missing_in_code.is_empty() && missing_in_docs.is_empty(),
        "docs/ERRORS.md has sections for {missing_in_code:?} that diag::codes() lacks; diag::codes() has {missing_in_docs:?} without a section"
    );
}

// ---------------------------------------------------------------- harness self-tests

#[test]
fn body_template_compiles_empty_on_both_arches() {
    let (src, _) = splice("");
    for arch in [Arch::Gfx1151, Arch::Gfx1201] {
        let c = peacemaker_rip::check(&src, "template", arch, "default");
        let errs: Vec<String> = c
            .diagnostics
            .iter()
            .filter(|d| d.severity == Severity::Error)
            .map(|d| d.render(&src))
            .collect();
        assert!(
            errs.is_empty(),
            "{}: the `rip body` template does not compile with an empty body:\n{}",
            arch.name(),
            errs.join("\n")
        );
    }
}

#[test]
fn info_strings_are_classified() {
    let md = "\
```rip
a
```
```rip eval
b
```
```rip fails=R0201
c
```
```rip body variant=x arch=gfx1151
d
```
```mnemonics
gfx1151: s_endpgm
```
```text
plain
```
```rip bogus
e
```
```rip eval body
f
```
~~~
not a fence
~~~
";
    let bs = blocks(md);
    assert_eq!(bs.len(), 8);
    assert_eq!(
        bs[0].kind,
        Kind::Rip {
            body: false,
            fails: None
        }
    );
    assert_eq!(bs[0].line, 1);
    assert_eq!(bs[1].kind, Kind::Eval);
    assert_eq!(
        bs[2].kind,
        Kind::Rip {
            body: false,
            fails: Some("R0201".into())
        }
    );
    assert_eq!(
        bs[3].kind,
        Kind::Rip {
            body: true,
            fails: None
        }
    );
    assert_eq!(bs[3].variant, "x");
    assert_eq!(bs[3].arch, Some(Arch::Gfx1151));
    assert_eq!(bs[4].kind, Kind::Mnemonics);
    assert_eq!(bs[5].kind, Kind::Other);
    assert!(bs[6].bad_info.is_some() && bs[7].bad_info.is_some());
    assert!(
        blocks("```rip\nfn emit() {}\n")[0].bad_info.as_deref() == Some("unterminated code fence")
    );
    // `eval` cannot be combined; an unknown attribute is an error.
    assert!(classify("rip eval body").3.is_some());
    assert!(classify("rip bogus").3.is_some());
    assert!(classify("rip fails=R12").3.is_some());
    assert!(classify("rip arch=gfx900").3.is_some());
}

#[test]
fn mnemonics_blocks_enforce_both_polarities() {
    let ok = Block {
        line: 1,
        info: "mnemonics".into(),
        kind: Kind::Mnemonics,
        variant: "default".into(),
        arch: None,
        text: "gfx1151: s_endpgm\ngfx1201: s_endpgm\n!gfx1151: s_no_such_instruction\n".into(),
        bad_info: None,
    };
    let mut errors = Vec::new();
    check_mnemonics("t.md", &ok, &mut errors);
    assert!(errors.is_empty(), "{errors:?}");
    let bad = Block {
        text: "gfx1151: s_no_such_instruction\n!gfx1151: s_endpgm\ngfx1151 s_endpgm\n".into(),
        ..ok
    };
    check_mnemonics("t.md", &bad, &mut errors);
    assert_eq!(errors.len(), 3, "{errors:?}");
}

#[test]
fn fails_blocks_require_the_named_code() {
    let src =
        "```rip fails=R0501\nfn not_emit() {}\n```\n\n```rip fails=R0201\nfn not_emit() {}\n```\n";
    let mut errors = Vec::new();
    for b in blocks(src) {
        check_rip("t.md", &b, &mut errors);
    }
    assert_eq!(
        errors.len(),
        2,
        "one right code (R0501 passes on both arches), one wrong: {errors:?}"
    );
    assert!(
        errors
            .iter()
            .all(|e| e.contains("expected first diagnostic R0201")),
        "{errors:?}"
    );
}
