//! Diagnostics: the code reference (`docs/ERRORS.md`) is executable. Every documented
//! failing script must raise exactly its code through `check`, every corrected script must
//! compile clean on both RDNA targets, and the rendering, the JSON form and the span
//! widening are pinned by golden strings.

use peacemaker_isa::Arch;
use peacemaker_rip::diag::{self, Diagnostic, Severity, Span};
use peacemaker_rip::{check, Checked};
use serde_json::json;

const DOC: &str = include_str!("../docs/ERRORS.md");
const NAME: &str = "example.rip";

/// Every code in reference order, with its stable title.
const CODES: &[(&str, &str)] = &[
    ("R0101", "syntax error"),
    ("R0102", "unsupported macro"),
    ("R0201", "unknown name"),
    ("R0202", "unknown method"),
    ("R0203", "wrong number of arguments"),
    ("R0204", "integer typing or overflow"),
    ("R0205", "type mismatch"),
    ("R0206", "unsupported language feature"),
    ("R0207", "script panic or assertion"),
    ("R0301", "undrained LDS store at barrier"),
    ("R0302", "region still published (not retired)"),
    ("R0303", "read of an unpublished region"),
    ("R0304", "workgroup op under divergent control"),
    ("R0305", "Workgroup scope"),
    ("R0306", "labels, exits and branches"),
    ("R0399", "other driver refusal"),
    ("R0401", "register not in the plan"),
    ("R0402", "register plan"),
    ("R0403", "instruction policy"),
    ("R0404", "waits and counters"),
    ("R0405", "kernel end"),
    ("R0501", "entry contract"),
    ("R0502", "script returned an error"),
    ("R0601", "unknown mnemonic"),
    ("R0602", "malformed operands"),
    ("R0603", "defs/uses disagree with the instruction"),
    ("R0604", "register read before it is defined"),
    ("R0701", "certification / assembly"),
    ("R0901", "tool error"),
    ("R0000", "unclassified"),
];

/// Codes only the environment or the classifier's fallback can produce: their sections carry no script.
const NO_SCRIPT: &[&str] = &["R0701", "R0901", "R0000"];

// ---- the reference document --------------------------------------------------------------

struct Block {
    /// The code a ```rip block is marked with (`fails=Rxxxx`); `None` for a corrected script.
    fails: Option<String>,
    /// `arch=gfxNNNN`: the one architecture the block is about; both RDNA targets otherwise.
    arch: Option<Arch>,
    /// `variant=NAME`; the default variant name otherwise.
    variant: String,
    source: String,
}

impl Block {
    fn arches(&self) -> Vec<Arch> {
        self.arch
            .map_or_else(|| vec![Arch::Gfx1151, Arch::Gfx1201], |a| vec![a])
    }
}
struct Section {
    code: String,
    title: String,
    blocks: Vec<Block>,
}

fn sections() -> Vec<Section> {
    let mut out: Vec<Section> = Vec::new();
    let mut lines = DOC.lines();
    while let Some(line) = lines.next() {
        if let Some(heading) = line.strip_prefix("## ") {
            let (code, title) = heading
                .split_once(": ")
                .unwrap_or_else(|| panic!("heading `{heading}` is not `Rxxxx: title`"));
            out.push(Section {
                code: code.into(),
                title: title.into(),
                blocks: Vec::new(),
            });
        } else if let Some(info) = line.strip_prefix("```rip") {
            let (mut fails, mut arch, mut variant) = (None, None, "Default".to_string());
            for word in info.split_whitespace() {
                if let Some(c) = word.strip_prefix("fails=") {
                    fails = Some(c.to_string());
                } else if let Some(a) = word.strip_prefix("arch=") {
                    arch = Some(
                        a.parse::<Arch>()
                            .unwrap_or_else(|e| panic!("`{word}`: {e}")),
                    );
                } else if let Some(v) = word.strip_prefix("variant=") {
                    variant = v.to_string();
                } else {
                    panic!("unknown fence attribute `{word}`");
                }
            }
            let mut source = String::new();
            for l in lines.by_ref() {
                if l == "```" {
                    break;
                }
                source.push_str(l);
                source.push('\n');
            }
            out.last_mut()
                .expect("a ```rip block outside any section")
                .blocks
                .push(Block {
                    fails,
                    arch,
                    variant,
                    source,
                });
        }
    }
    out
}

fn run_block(b: &Block, arch: Arch) -> Checked {
    check(&b.source, NAME, arch, &b.variant)
}

fn errors(c: &Checked) -> Vec<&Diagnostic> {
    c.diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect()
}

fn word(severity: Severity) -> &'static str {
    match severity {
        Severity::Error => "error",
        Severity::Warning => "warning",
    }
}

#[test]
fn every_code_is_documented_in_order_with_its_exact_title() {
    let found: Vec<(String, String)> = sections().into_iter().map(|s| (s.code, s.title)).collect();
    let want: Vec<(String, String)> = CODES
        .iter()
        .map(|(c, t)| (c.to_string(), t.to_string()))
        .collect();
    assert_eq!(found, want, "docs/ERRORS.md headings");
    // The library serves the same sections.
    let docs = diag::codes();
    assert_eq!(
        docs.iter().map(|d| (d.code, d.title)).collect::<Vec<_>>(),
        CODES.to_vec()
    );
    for (code, title) in CODES {
        let doc = diag::explain(code).unwrap_or_else(|| panic!("explain({code})"));
        assert_eq!((doc.code, doc.title), (*code, *title));
        assert!(
            doc.body.len() > 200,
            "{code}: the explanation is a paragraph, not a stub"
        );
        assert!(
            !doc.body.starts_with("##"),
            "{code}: the body does not repeat the heading"
        );
    }
    assert!(diag::explain("R9999").is_none());
    assert!(diag::explain("").is_none());
    assert!(diag::explain("r0301").is_none(), "codes are case sensitive");
}

#[test]
fn every_section_carries_whole_failing_and_corrected_scripts_where_a_script_can_raise_the_code() {
    for s in sections() {
        let fails: Vec<&Block> = s.blocks.iter().filter(|b| b.fails.is_some()).collect();
        let fixed: Vec<&Block> = s.blocks.iter().filter(|b| b.fails.is_none()).collect();
        if NO_SCRIPT.contains(&s.code.as_str()) {
            assert!(
                s.blocks.is_empty(),
                "{}: environment-only codes carry no script example",
                s.code
            );
            continue;
        }
        assert!(!fails.is_empty(), "{}: no failing script", s.code);
        assert_eq!(
            fails.len(),
            fixed.len(),
            "{}: every failing script is followed by its corrected script",
            s.code
        );
        for b in &fails {
            assert_eq!(
                b.fails.as_deref(),
                Some(s.code.as_str()),
                "{}: a failing block is marked with its own section's code",
                s.code
            );
        }
        for b in &s.blocks {
            assert!(
                b.source.contains("fn emit(") || b.source.contains("fn build("),
                "{}: a whole kernel script, not a fragment",
                s.code
            );
            assert!(
                b.source.contains("b.finish()") || s.code == "R0101",
                "{}: scripts finish the builder",
                s.code
            );
        }
    }
}

#[test]
fn every_failing_example_raises_exactly_its_code() {
    for s in sections() {
        for b in s.blocks.iter().filter(|b| b.fails.is_some()) {
            let code = b.fails.as_deref().unwrap();
            let c = run_block(b, b.arch.unwrap_or(Arch::Gfx1201));
            assert!(
                c.emitted.is_none() && c.resources.is_none(),
                "{code}: an error leaves nothing emitted"
            );
            let e = errors(&c);
            assert_eq!(e.len(), 1, "{code}: {:?}", c.diagnostics);
            let d: &Diagnostic = e[0];
            assert_eq!(d.code, code, "{}: {}", code, d.message);
            assert_eq!(d.severity, Severity::Error);
            assert_eq!(d.file, NAME, "{code}");
            assert!(
                !d.message.starts_with(NAME) && !d.message.is_empty(),
                "{code}: the message carries no position prefix: {}",
                d.message
            );
            assert!(
                d.help.as_deref().is_some_and(|h| !h.is_empty()),
                "{code}: every diagnostic has a fix-it"
            );
            if code != "R0501" && code != "R0502" {
                assert!(
                    d.span.is_some(),
                    "{code}: located at the construct that failed: {d:?}"
                );
            }

            // Rendering.
            let text = d.render(&b.source);
            assert!(
                text.starts_with(&format!("{}[{code}]: {}\n", word(d.severity), d.message)),
                "{code}:\n{text}"
            );
            assert!(
                text.ends_with(&format!(
                    "  = note: details: peacemaker-rip explain {code}\n"
                )),
                "{code}:\n{text}"
            );
            if let Some(sp) = d.span {
                assert!(
                    text.contains(&format!(" --> {NAME}:{}:{}\n", sp.line, sp.col)),
                    "{code}:\n{text}"
                );
                let line = b.source.lines().nth(sp.line - 1).unwrap();
                assert!(
                    text.contains(&format!(" | {line}\n")),
                    "{code}: the failing line is shown\n{text}"
                );
                assert!(text.contains('^'), "{code}:\n{text}");
            }

            // JSON.
            let j = d.to_json();
            assert_eq!(j["code"], code);
            assert_eq!(j["severity"], word(d.severity));
            assert_eq!(j["message"], d.message.as_str());
            assert_eq!(j["file"], NAME);
            assert_eq!(j["explain"], format!("peacemaker-rip explain {code}"));
            assert_eq!(j["help"], d.help.as_deref().unwrap());
            match d.span {
                Some(sp) => assert_eq!(
                    j["span"],
                    json!({"line": sp.line, "col": sp.col, "end_line": sp.end_line, "end_col": sp.end_col})
                ),
                None => assert!(j["span"].is_null()),
            }
            assert!(
                diag::explain(code)
                    .is_some_and(|doc| doc.body.contains(&format!("```rip fails={code}"))),
                "{code}: the explanation embeds its failing script"
            );
        }
    }
}

#[test]
fn failing_examples_raise_the_same_code_on_gfx1151_unless_they_are_written_for_gfx1201() {
    for s in sections() {
        for b in s.blocks.iter().filter(|b| b.fails.is_some()) {
            let code = b.fails.as_deref().unwrap();
            if b.arch.is_some() {
                // Written for one architecture (`Counter::Vm` is the gfx11 spelling, only wrong for gfx12;
                // `s2` is the workgroup id only on gfx11): the other target accepts it silently.
                assert_eq!(b.arch, Some(Arch::Gfx1201), "{code}");
                let c = run_block(b, Arch::Gfx1151);
                assert!(
                    c.diagnostics.is_empty() && c.emitted.is_some(),
                    "{code} on gfx1151: {:?}",
                    c.diagnostics
                );
                continue;
            }
            let c = run_block(b, Arch::Gfx1151);
            let d = c
                .diagnostics
                .first()
                .unwrap_or_else(|| panic!("{code}: no diagnostic on gfx1151"));
            assert_eq!(d.code, code, "gfx1151 {}", d.message);
        }
    }
}

#[test]
fn corrected_examples_compile_clean_on_both_targets() {
    for s in sections() {
        for b in s.blocks.iter().filter(|b| b.fails.is_none()) {
            for arch in b.arches() {
                let c = run_block(b, arch);
                assert!(
                    c.diagnostics.is_empty(),
                    "{} on {arch:?}: {:?}",
                    s.code,
                    c.diagnostics
                );
                let e = c
                    .emitted
                    .as_ref()
                    .unwrap_or_else(|| panic!("{} on {arch:?}: nothing emitted", s.code));
                assert!(e.s_text.contains("s_endpgm"), "{}", s.code);
                assert!(
                    c.resources.is_some(),
                    "{}: resources come from the emitted descriptor",
                    s.code
                );
            }
        }
    }
}

#[test]
fn the_overview_render_is_the_real_output_for_the_r0301_example() {
    let intro = DOC.split("\n## ").next().unwrap();
    let start = intro
        .find("```text\nerror[R0301]")
        .expect("the overview shows an R0301 rendering")
        + "```text\n".len();
    let shown = &intro[start..start + intro[start..].find("```").unwrap()];
    let source = sections()
        .into_iter()
        .find(|s| s.code == "R0301")
        .unwrap()
        .blocks
        .remove(0)
        .source;
    let c = check(&source, "kernel.rip", Arch::Gfx1201, "Default");
    assert_eq!(c.diagnostics.len(), 1, "{:?}", c.diagnostics);
    assert_eq!(c.diagnostics[0].render(&source), shown);
}

// ---- main-only scripts through eval ------------------------------------------------------

fn eval_code(source: &str) -> Diagnostic {
    let raw = peacemaker_rip::eval(source, "main")
        .err()
        .unwrap_or_else(|| panic!("the script must fail: {source}"));
    Diagnostic::from_raw(&raw, "rip", source, Severity::Error)
}

#[test]
fn evaluator_faults_of_main_only_scripts_are_classified_and_located() {
    let cases: &[(&str, &str)] = &[
        ("R0102", "fn main() -> u32 { weird!(1); 2 }"),
        ("R0201", "fn main() -> u32 { missing + 1 }"),
        (
            "R0202",
            "fn main() -> u32 {\n    let a: u8 = 1;\n    a.nope()\n}",
        ),
        (
            "R0203",
            "fn f(x: u32) -> u32 { x } fn main() -> u32 { f(1, 2) }",
        ),
        ("R0204", "fn main() -> u8 { let a: u8 = 200; a + 56 }"),
        ("R0204", "fn main() -> u8 { let a: u8 = 0; a - 1 }"),
        (
            "R0204",
            "fn main() -> u32 { let a: u8 = 1; let b: u32 = 2; a + b }",
        ),
        (
            "R0204",
            "fn f(x: u8) -> u8 { x } fn main() -> u8 { f(256) }",
        ),
        (
            "R0205",
            "fn main() -> u32 { let n: u32 = 1; if n { 1 } else { 2 } }",
        ),
        ("R0205", "fn main() -> u32 { let n: u32 = 1; n? }"),
        ("R0206", "fn main() -> u32 { let _ = async { 1 }; 2 }"),
        ("R0207", "fn main() -> u32 { assert_eq!(1 + 1, 3); 0 }"),
        ("R0207", "fn main() -> u32 { panic!(\"boom {}\", 1) }"),
    ];
    for (code, src) in cases {
        let d = eval_code(src);
        assert_eq!(d.code, *code, "{src}: {}", d.message);
        assert_eq!(d.file, "rip");
        assert!(
            d.span.is_some_and(|s| s.line >= 1 && s.col >= 1),
            "{src}: located: {d:?}"
        );
        assert!(!d.message.starts_with("rip:"), "{src}: {}", d.message);
    }
    // A syntax error stops before anything runs, and is positioned too.
    let src = "fn main() -> u32 {\n    let x: u32 = ;\n    1\n}\n";
    let raw = peacemaker_rip::parse(src, "t.rip")
        .err()
        .expect("syntax error");
    let d = Diagnostic::from_raw(&raw, "t.rip", src, Severity::Error);
    assert_eq!((d.code, d.file.as_str()), ("R0101", "t.rip"), "{raw}");
    assert_eq!(d.span.map(|s| s.line), Some(2));
}

// ---- classification of messages that no script reaches without a tool or environment fault -----

/// One representative raw pipeline message per code. Messages are the exact formats of the
/// sources that produce them (evaluator, author driver, builder, validation, certify, tool).
const MESSAGES: &[(&str, &str)] = &[
    ("R0101", "t.rip:12:22: expected an expression"),
    ("R0102", "t.rip:1:1: macro `wave_size!` is not supported by the .rip evaluator"),
    ("R0201", "t.rip:12:43: cannot find `vgpr` in this scope"),
    ("R0201", "t.rip:3:9: cannot find variable `lanes`"),
    ("R0202", "t.rip:12:5: no method `barier` on `Workgroup`"),
    ("R0203", "t.rip:12:5: `op` takes 4 arguments, got 3"),
    ("R0203", "t.rip:5:9: closure takes 1 arguments, got 2"),
    ("R0204", "t.rip:13:16: attempt to add with overflow: 200 + 100"),
    ("R0204", "t.rip:2:5: mismatched integer types in `+`: u8 and u32"),
    ("R0204", "t.rip:2:5: attempt to divide by zero"),
    ("R0204", "t.rip:2:5: literal 256 out of range for `u8`"),
    ("R0205", "t.rip:13:5: condition must be bool, found u32"),
    ("R0205", "t.rip:4:5: `?` needs a Result or Option, found u32"),
    ("R0205", "t.rip:4:5: mismatched types: expected `u8`, found `u32`"),
    ("R0206", "t.rip:12:17: unsupported expression (async block)"),
    ("R0206", "t.rip:2:5: struct-like enum variant `E::A` is not supported"),
    ("R0207", "t.rip:1:1: assertion failed: tile must be a multiple of 16"),
    ("R0207", "t.rip:1:1: assertion failed: `24` vs `32`"),
    ("R0207", "t.rip:1:1: panic: unsupported tile 7"),
    ("R0301", "barrier would publish t with an undrained LDS store: wait for it first (s_barrier does not wait for memory counters)"),
    ("R0302", "cannot store into t: it is Published and other waves may still be reading it until a barrier retires it"),
    ("R0303", "t is not published: it is Writing"),
    ("R0304", "barrier is workgroup scope: it cannot run under wave- or lane-dependent control"),
    ("R0305", "t.rip:13:15: no active Workgroup here: `let mut wg = Workgroup::new(&mut b, arch)?;` opens the scope that the rest of its block runs in"),
    ("R0305", "t.rip:3:5: wg.isa() outside the Workgroup scope"),
    ("R0306", "duplicate label .Lloop"),
    ("R0306", "forward target .Lfar is never placed"),
    ("R0306", ".Lfar is already placed"),
    ("R0306", "branch to .Lx after it was placed: branches only go forward"),
    ("R0399", "LDS region t [0, +70000) exceeds 65536 bytes on gfx1201"),
    ("R0302", "cannot join mailbox: it is Published; other waves may still be reading it, only a barrier can retire their reads"),
    ("R0399", "the paths disagree on LDS ownership: t is Free on one path and Writing on the other"),
    ("R0401", "v40 at entry has no unique live owner"),
    ("R0402", "extra overlaps vectors with a shared lifetime"),
    ("R0402", "register width must be 1, 2, 4 or 8"),
    ("R0402", "register budget exceeds architectural limit"),
    ("R0403", "s_barrier is a barrier: barriers go through the typed core"),
    ("R0403", "typed kernel: LDS, barriers, branches and loops go through peacemaker-author"),
    ("R0404", "counter/architecture mismatch"),
    ("R0404", "loop .Lloop: back-edge ledger must equal the entry ledger"),
    // Only reachable when the ledger is non-empty at `finish`; `s_endpgm` drains it, so no script gets here.
    ("R0404", "outstanding memory operations: wait or end the kernel before finish"),
    ("R0405", "kernel must end with s_endpgm"),
    ("R0405", "kernel exit .Lend is never placed"),
    ("R0405", "instruction after s_endpgm"),
    ("R0405", "kernel ends with a split barrier in flight"),
    ("R0501", "t.rip:1:1: no `fn emit(spec: Spec) -> Result<Emitted, String>` entry point"),
    ("R0501", "t.rip:1:1: `emit` takes (spec) or (arch, variant), not 3 arguments"),
    ("R0601", "t.rip:12:5: unknown mnemonic `v_mov_b32_e33` on gfx1201; did you mean `v_mov_b32_e32`?"),
    ("R0602", "t.rip:12:5: malformed operands for `v_mov_b32_e32` on gfx1201: expected 2 operands"),
    ("R0603", "t.rip:12:5: op declares defs [v2] and uses [v0] but `v_add_f32_e32 v2, v0, v1` also reads v1: write &[v(2)], &[v(0), v(1)]"),
    ("R0604", "t.rip:12:5: register read before it is defined: s2 in `s_mov_b32 s20, s2`; nothing in the kernel writes it and the launch does not initialize it (gfx1201 entry state: s[0:1] kernarg pointer, ttmp9 workgroup id x, ttmp7 = y | z<<16, v0 thread id)"),
    ("R0701", "Default: wait replay: s_waitcnt is missing before the first use"),
    ("R0701", "kernel: assemble/link: llvm-mc exited with status 1"),
    ("R0901", "usage: peacemaker-rip check FILE [--arch LIST|all] [--variant LIST] [--json]"),
    ("R0901", "rustfmt not found on PATH: install it with `rustup component add rustfmt`"),
    ("R0901", "cannot start the evaluator thread: Resource temporarily unavailable"),
    ("R0000", "SMEM width"),
];

#[test]
fn raw_messages_classify_to_their_codes() {
    let source = "fn main() {}\n";
    for (code, raw) in MESSAGES {
        let d = Diagnostic::from_raw(raw, "t.rip", source, Severity::Error);
        assert_eq!(d.code, *code, "{raw}");
        assert_eq!(d.severity, Severity::Error);
        assert!(
            d.help.is_some(),
            "{code}: {raw}: every classification has a fix-it"
        );
        assert!(diag::explain(d.code).is_some(), "{code}");
        assert_eq!(d.file, "t.rip");
        assert!(!d.message.starts_with("t.rip:"), "{raw} -> {}", d.message);
    }
    // R0502 is not a message class: `check` assigns it to the script's own `Err(..)` (see the
    // R0502 example above). Every other code is covered here.
    let covered: std::collections::BTreeSet<&str> = MESSAGES.iter().map(|(c, _)| *c).collect();
    for (code, _) in CODES {
        assert!(
            covered.contains(code) || *code == "R0502",
            "{code} has no raw-message classification test"
        );
    }
}

#[test]
fn warnings_keep_their_severity_and_carry_the_corrected_lists() {
    let raw = "t.rip:12:5: op declares defs [v2] and uses [v0] but `v_add_f32_e32 v2, v0, v1` also reads v1: write &[v(2)], &[v(0), v(1)]";
    let d = Diagnostic::from_raw(raw, "t.rip", "x\n", Severity::Warning);
    assert_eq!((d.code, d.severity), ("R0603", Severity::Warning));
    assert!(
        d.help.as_deref().unwrap().contains("&[v(0), v(1)]"),
        "{:?}",
        d.help
    );
    assert!(d.render("x\n").starts_with("warning[R0603]: "));
    assert_eq!(d.to_json()["severity"], "warning");
    let raw = "t.rip:3:5: register read before it is defined: s2 in `s_mov_b32 s20, s2`; nothing in the kernel writes it and the launch does not initialize it (gfx1201 entry state: s[0:1] kernarg pointer, ttmp9 workgroup id x, ttmp7 = y | z<<16, v0 thread id)";
    let d = Diagnostic::from_raw(raw, "t.rip", "x\n", Severity::Warning);
    assert_eq!((d.code, d.severity), ("R0604", Severity::Warning));
    assert!(d.help.as_deref().unwrap().contains("ttmp9"), "{:?}", d.help);
}

#[test]
fn unpositioned_messages_have_no_span_and_a_positioned_prefix_is_split_off() {
    let d = Diagnostic::from_raw(
        "v40 at entry has no unique live owner",
        "k.rip",
        "fn emit() {}\n",
        Severity::Error,
    );
    assert_eq!(
        (d.file.as_str(), d.span, d.message.as_str()),
        ("k.rip", None, "v40 at entry has no unique live owner")
    );
    let rendered = d.render("fn emit() {}\n");
    assert!(!rendered.contains("-->"), "{rendered}");
    assert!(rendered.contains("  = help: "), "{rendered}");
    // The prefix names the file; a path with directories is kept whole.
    let d = Diagnostic::from_raw(
        "kernels/a/k.rip:1:4: no method `x` on `u8`",
        "ignored.rip",
        "fn emit() {}\n",
        Severity::Error,
    );
    assert_eq!(
        (d.file.as_str(), d.message.as_str()),
        ("kernels/a/k.rip", "no method `x` on `u8`")
    );
    assert_eq!(d.span.map(|s| (s.line, s.col)), Some((1, 4)));
    // Messages that merely contain colons and digits are not positions.
    let d = Diagnostic::from_raw(
        "module: assemble/link: exit 1: failed",
        "k.rip",
        "",
        Severity::Error,
    );
    assert_eq!((d.code, d.span, d.file.as_str()), ("R0701", None, "k.rip"));
}

#[test]
fn name_errors_suggest_the_nearest_vocabulary_entry() {
    let src = "fn emit() {\n    srd_tial(wg.isa(), 12, None)?;\n    wg.barier(())?;\n}\n";
    let d = Diagnostic::from_raw(
        "t.rip:2:5: cannot find `srd_tial` in this scope",
        "t.rip",
        src,
        Severity::Error,
    );
    assert_eq!(d.code, "R0201");
    assert!(
        d.help.as_deref().unwrap().contains("`srd_tail`"),
        "{:?}",
        d.help
    );
    assert_eq!(
        d.suggestion.as_deref(),
        Some("srd_tail(wg.isa(), 12, None)")
    );
    let d = Diagnostic::from_raw(
        "t.rip:3:8: no method `barier` on `Workgroup`",
        "t.rip",
        src,
        Severity::Error,
    );
    assert_eq!(d.code, "R0202");
    assert!(
        d.help.as_deref().unwrap().contains("`barrier`"),
        "{:?}",
        d.help
    );
    assert_eq!(d.suggestion.as_deref(), Some("barrier(())"));
    assert_eq!(d.to_json()["suggestion"], "barrier(())");
}

#[test]
fn fix_its_use_the_message_region_register_and_listed_vocabulary() {
    let source = "wg.barrier((ready(t, store),))?;\n";
    let d = Diagnostic::from_raw(
        "kernel.rip:1:4: barrier would publish t with an undrained LDS store: wait for it first (s_barrier does not wait for memory counters)",
        "kernel.rip", source, Severity::Error,
    );
    assert_eq!(d.code, "R0301");
    let help = d.help.as_deref().unwrap();
    assert!(
        help.contains("for t:")
            && help.contains("let drained = wg.wait(store)?;")
            && help.contains("ready(t, drained)")
    );
    println!("{}", d.render(source));

    let source = "op(wg.isa(), \"v_mov_b32 v99, 0\", &[v(99)], &[])?;\n";
    let d = Diagnostic::from_raw(
        "kernel.rip:1:1: v99 at entry has no unique live owner",
        "kernel.rip",
        source,
        Severity::Error,
    );
    assert_eq!(d.code, "R0401");
    assert!(d
        .help
        .as_deref()
        .unwrap()
        .contains("p.v::<1>(\"v99\", 99, Live::Whole)?;"));
    println!("{}", d.render(source));
    let d = Diagnostic::from_raw(
        "s[12:15] at entry has no unique live owner",
        "kernel.rip",
        "",
        Severity::Error,
    );
    assert!(d
        .help
        .as_deref()
        .unwrap()
        .contains("p.s::<4>(\"s12\", 12, Live::Whole)?;"));

    let d = Diagnostic::from_raw(
        "no method `publsh` on `Mailbox` (methods: publish, release)",
        "kernel.rip",
        "",
        Severity::Error,
    );
    assert!(d.help.as_deref().unwrap().contains("`publish`"));
    let d = Diagnostic::from_raw(
        "cannot find `rgion` (available: region, lane)",
        "kernel.rip",
        "",
        Severity::Error,
    );
    assert!(d.help.as_deref().unwrap().contains("`region`"));
    let d = Diagnostic::from_raw(
        "wg.isa() outside the Workgroup scope",
        "kernel.rip",
        "",
        Severity::Error,
    );
    let help = d.help.as_deref().unwrap();
    assert!(
        help.contains("first later statement")
            && help.contains("let b = wg.isa();")
            && help.contains("fn author<T: Target>")
    );
}

#[test]
fn published_reuse_and_loop_ledger_refusals_have_specific_fix_its() {
    for raw in [
        "cannot store into mailbox: it is Published and other waves may still be reading it",
        "cannot join mailbox: it is Published; only a barrier can retire their reads",
        "cannot split mailbox: it is Published",
        "mailbox is Published at relayout: a barrier must retire every reader first",
    ] {
        let d = Diagnostic::from_raw(raw, "kernel.rip", "", Severity::Error);
        assert_eq!(d.code, "R0302", "{raw}");
        assert!(
            d.help.as_deref().unwrap().contains("retire(mailbox)"),
            "{:?}",
            d.help
        );
    }
    let d = Diagnostic::from_raw(
        "cannot split mailbox: it is Writing",
        "kernel.rip",
        "",
        Severity::Error,
    );
    assert_eq!(d.code, "R0399");
    let d = Diagnostic::from_raw(
        "loop .Lloop: back-edge ledger must equal the entry ledger",
        "kernel.rip",
        "",
        Severity::Error,
    );
    assert_eq!(d.code, "R0404");
    let help = d.help.as_deref().unwrap();
    assert!(
        help.contains(".Lloop")
            && help.contains("wg.wait(store)?")
            && help.contains("before the back edge"),
        "{help}"
    );
}

#[test]
fn non_inline_operand_help_uses_the_number_from_the_message() {
    for (number, mnemonic) in [("128", "v_add_nc_u32_e32"), ("512", "v_sub_nc_u32_e32")] {
        let raw = format!("t.rip:12:5: malformed operands for `{mnemonic}` on gfx1201: bad operand \"{number}\" for {mnemonic}: not an inline constant");
        let d = Diagnostic::from_raw(&raw, "t.rip", "x\n", Severity::Error);
        assert_eq!(d.code, "R0602", "{raw}");
        let help = d.help.as_deref().unwrap();
        assert!(
            help.contains(&format!("`{number}` is not an inline constant")),
            "{help}"
        );
        assert!(help.contains("-16..64"), "{help}");
        assert!(help.contains(&format!("lit({number})")), "{help}");
        assert!(help.contains("`0x..`"), "{help}");
        assert!(help.contains("one literal per instruction"), "{help}");
        assert!(
            help.contains("src0") && help.contains("_e32") && help.contains("_e64"),
            "{help}"
        );
        assert!(!help.contains("operand order, register widths"), "{help}");
    }
}

#[test]
fn gfx1201_buffer_soffset_help_says_null_and_other_operand_errors_stay_generic() {
    let raw = "t.rip:12:5: malformed operands for `buffer_load_b32` on gfx1201: unrecognized machine instruction at byte offset 0: SOFFSET does not fit its encoding field";
    let d = Diagnostic::from_raw(raw, "t.rip", "x\n", Severity::Error);
    assert_eq!(d.code, "R0602");
    let help = d.help.as_deref().unwrap();
    assert!(
        help.contains("On gfx12 a zero buffer soffset is spelled `null`, not `0`."),
        "{help}"
    );
    assert!(!help.contains("lit("), "{help}");

    let generic = "Use the selected architecture's operand order, register widths and legal inline constants for this instruction.";
    for raw in [
        "t.rip:12:5: malformed operands for `buffer_load_b32` on gfx1151: unrecognized machine instruction at byte offset 0: SOFFSET does not fit its encoding field",
        "t.rip:12:5: malformed operands for `v_mov_b32_e32` on gfx1201: SOFFSET does not fit its encoding field",
        "t.rip:12:5: malformed operands for `v_mov_b32_e32` on gfx1201: expected 2 operands",
    ] {
        let d = Diagnostic::from_raw(raw, "t.rip", "x\n", Severity::Error);
        assert_eq!(d.code, "R0602", "{raw}");
        assert_eq!(d.help.as_deref(), Some(generic), "{raw}");
    }
}

// ---- golden render and JSON ----------------------------------------------------------------

fn diag(
    code: &'static str,
    severity: Severity,
    message: &str,
    span: Option<Span>,
    help: Option<&str>,
    suggestion: Option<&str>,
    notes: &[&str],
) -> Diagnostic {
    Diagnostic {
        code,
        severity,
        message: message.into(),
        file: "t.rip".into(),
        span,
        help: help.map(Into::into),
        suggestion: suggestion.map(Into::into),
        notes: notes.iter().map(|n| n.to_string()).collect(),
    }
}

#[test]
fn render_single_line_span_golden() {
    let source = "fn main() {\n    let x = missing_fn(1, (2, 3));\n}\n";
    let d = diag(
        "R0201",
        Severity::Error,
        "cannot find `missing_fn` in this scope",
        Some(Span {
            line: 2,
            col: 13,
            end_line: 2,
            end_col: 34,
        }),
        Some("Declare `missing_fn` before use."),
        None,
        &["the call is in `main`"],
    );
    let want = "\
error[R0201]: cannot find `missing_fn` in this scope
 --> t.rip:2:13
  |
2 |     let x = missing_fn(1, (2, 3));
  |             ^^^^^^^^^^^^^^^^^^^^^
  = help: Declare `missing_fn` before use.
  = note: the call is in `main`
  = note: details: peacemaker-rip explain R0201
";
    assert_eq!(d.render(source), want);
}

#[test]
fn render_multi_line_span_golden_uses_a_two_digit_gutter() {
    let mut source = String::new();
    for i in 1..=8 {
        source.push_str(&format!("// line {i}\n"));
    }
    source.push_str("    op(wg.isa(),\n       \"x\", &[], &[])?;\n");
    let d = diag(
        "R0203",
        Severity::Error,
        "`op` takes 4 arguments, got 3",
        Some(Span {
            line: 9,
            col: 5,
            end_line: 10,
            end_col: 22,
        }),
        None,
        None,
        &[],
    );
    let want = "\
error[R0203]: `op` takes 4 arguments, got 3
 --> t.rip:9:5
   |
 9 |     op(wg.isa(),
   |     ^^^^^^^^^^^^
10 |        \"x\", &[], &[])?;
   | ^^^^^^^^^^^^^^^^^^^^^
  = note: details: peacemaker-rip explain R0203
";
    assert_eq!(d.render(&source), want);
}

#[test]
fn render_point_span_warning_and_unlocated_goldens() {
    let source = "x = ;\n";
    let point = diag(
        "R0101",
        Severity::Error,
        "expected an expression",
        Some(Span {
            line: 1,
            col: 5,
            end_line: 1,
            end_col: 6,
        }),
        Some("Repair the syntax."),
        None,
        &[],
    );
    assert_eq!(
        point.render(source),
        "\
error[R0101]: expected an expression
 --> t.rip:1:5
  |
1 | x = ;
  |     ^
  = help: Repair the syntax.
  = note: details: peacemaker-rip explain R0101
"
    );
    let warn = diag(
        "R0603",
        Severity::Warning,
        "op declares defs [v2]",
        Some(Span {
            line: 1,
            col: 1,
            end_line: 1,
            end_col: 2,
        }),
        None,
        Some("op(..)"),
        &[],
    );
    assert_eq!(
        warn.render(source),
        "\
warning[R0603]: op declares defs [v2]
 --> t.rip:1:1
  |
1 | x = ;
  | ^
  = note: details: peacemaker-rip explain R0603
"
    );
    let bare = diag(
        "R0501",
        Severity::Error,
        "no `fn emit` entry point",
        None,
        Some("Define `fn emit`."),
        None,
        &[],
    );
    assert_eq!(
        bare.render(source),
        "\
error[R0501]: no `fn emit` entry point
  = help: Define `fn emit`.
  = note: details: peacemaker-rip explain R0501
"
    );
}

#[test]
fn json_golden() {
    let d = diag(
        "R0201",
        Severity::Error,
        "cannot find `srd_tial` in this scope",
        Some(Span {
            line: 2,
            col: 5,
            end_line: 2,
            end_col: 33,
        }),
        Some("Did you mean `srd_tail` instead of `srd_tial`?"),
        Some("srd_tail(wg.isa(), 12, None)"),
        &["first note"],
    );
    assert_eq!(
        d.to_json(),
        json!({
            "code": "R0201",
            "severity": "error",
            "message": "cannot find `srd_tial` in this scope",
            "file": "t.rip",
            "span": {"line": 2, "col": 5, "end_line": 2, "end_col": 33},
            "help": "Did you mean `srd_tail` instead of `srd_tial`?",
            "suggestion": "srd_tail(wg.isa(), 12, None)",
            "notes": ["first note"],
            "explain": "peacemaker-rip explain R0201",
        })
    );
    let bare = diag("R0901", Severity::Warning, "x", None, None, None, &[]);
    assert_eq!(
        bare.to_json(),
        json!({
            "code": "R0901",
            "severity": "warning",
            "message": "x",
            "file": "t.rip",
            "span": null,
            "help": null,
            "suggestion": null,
            "notes": [],
            "explain": "peacemaker-rip explain R0901",
        })
    );
}

// ---- span widening -------------------------------------------------------------------------

/// 1-based (line, column in chars) of `offset` chars into the first occurrence of `needle`.
fn pos(source: &str, needle: &str, offset: usize) -> (usize, usize) {
    let at = source
        .find(needle)
        .unwrap_or_else(|| panic!("`{needle}` not in the source"));
    let before = &source[..at];
    let line = before.matches('\n').count() + 1;
    let col = before.rsplit('\n').next().unwrap().chars().count() + 1;
    (line, col + offset)
}

/// The text a span covers (columns count chars; `end_col` is exclusive).
fn slice(source: &str, s: Span) -> String {
    let lines: Vec<&str> = source.lines().collect();
    let mut out = String::new();
    for n in s.line..=s.end_line {
        let chars: Vec<char> = lines[n - 1].chars().collect();
        let from = if n == s.line { s.col - 1 } else { 0 };
        let to = if n == s.end_line {
            s.end_col - 1
        } else {
            chars.len()
        };
        out.extend(&chars[from..to]);
        if n != s.end_line {
            out.push('\n');
        }
    }
    out
}

/// The span the classifier gives a diagnostic reported at (line, col) of `source`.
fn widened(source: &str, (line, col): (usize, usize)) -> Span {
    Diagnostic::from_raw(
        &format!("t.rip:{line}:{col}: cannot find `x` in this scope"),
        "t.rip",
        source,
        Severity::Error,
    )
    .span
    .expect("a positioned message has a span")
}

fn covers(source: &str, needle: &str, offset: usize, want: &str) {
    let s = widened(source, pos(source, needle, offset));
    assert_eq!(
        slice(source, s),
        want,
        "{source:?} at `{needle}`+{offset}: {s:?}"
    );
}

#[test]
fn span_covers_an_identifier_and_extends_over_its_call() {
    covers(
        "fn emit() {\n    let n = lane_count + 1;\n}\n",
        "lane_count",
        0,
        "lane_count",
    );
    covers(
        "fn emit() {\n    let m = lane_count;\n}\n",
        "lane_count",
        0,
        "lane_count",
    );
    covers(
        "    srd_tial(wg.isa(), 12, None)?;\n",
        "srd_tial",
        0,
        "srd_tial(wg.isa(), 12, None)",
    );
    // A method: the position is the method name; the receiver is not part of the span.
    covers("    wg.barier(())?;\n", "barier", 0, "barier(())");
    // Nested parentheses balance; the `?` after the call is not part of it.
    covers(
        "    foo(bar(1), (2, 3))?;\n",
        "foo",
        0,
        "foo(bar(1), (2, 3))",
    );
    // The caller may point into the middle of the identifier: the span still starts at its beginning.
    let src = "    let n = lane_count + 1;\n";
    covers(src, "lane_count", 4, "lane_count");
    assert_eq!(
        widened(src, pos(src, "lane_count", 4)).col,
        pos(src, "lane_count", 0).1
    );
    // Whitespace between the name and its parenthesis does not break the call.
    covers(
        "    foo\n        (1, 2)?;\n",
        "foo",
        0,
        "foo\n        (1, 2)",
    );
    // An unbalanced call keeps just the name.
    covers("    foo(1, 2\n", "foo", 0, "foo");
}

#[test]
fn span_skips_a_turbofish_between_the_name_and_the_call() {
    covers(
        "    p.v::<8>(\"vectors\", 8 * i, Live::Whole)?;\n",
        "v::<8>",
        0,
        "v::<8>(\"vectors\", 8 * i, Live::Whole)",
    );
    covers(
        "    p.v :: < 8 > /* width */ (\"x\", 0, Live::Whole)?;\n",
        "v :: <",
        0,
        "v :: < 8 > /* width */ (\"x\", 0, Live::Whole)",
    );
    // Nested generics, a grouped const expression and a function type arrow inside the turbofish.
    covers(
        "    wg.method::<Vec<u32>>(\"(\")?;\n",
        "method",
        0,
        "method::<Vec<u32>>(\"(\")",
    );
    covers(
        "    wg.method::<{ 1 > 0 }, fn() -> u32>(\"(\")?;\n",
        "method",
        0,
        "method::<{ 1 > 0 }, fn() -> u32>(\"(\")",
    );
    // A path that is not a turbofish stops at the identifier.
    covers("    let live = Live::Whole;\n", "Live", 0, "Live");
    covers(
        "    let mut wg = Workgroup::new(&mut b, arch)?;\n",
        "Workgroup",
        0,
        "Workgroup",
    );
}

#[test]
fn span_matching_call_skips_parentheses_in_literals_and_comments() {
    // A parenthesis and an escaped quote inside a string.
    covers(
        "    op(wg.isa(), \"mov (\\\")\", &[v(0)], &[])?;\n",
        "op(",
        0,
        "op(wg.isa(), \"mov (\\\")\", &[v(0)], &[])",
    );
    // Character literals, including the escaped quote.
    covers(
        "    f(')', '(', '\\'', 1)?;\n",
        "f(",
        0,
        "f(')', '(', '\\'', 1)",
    );
    // A raw string with a quote and a parenthesis inside.
    covers("    f(r#\"a)\"b\"#, 1)?;\n", "f(", 0, "f(r#\"a)\"b\"#, 1)");
    covers("    f(br\"x)\", 1)?;\n", "f(", 0, "f(br\"x)\", 1)");
    // A line comment and a block comment hide their parentheses; the call spans the lines.
    covers(
        "    op(wg.isa(), // )))\n       /* ( */ \"x\", &[], &[])?;\n",
        "op(",
        0,
        "op(wg.isa(), // )))\n       /* ( */ \"x\", &[], &[])",
    );
    // A nested block comment.
    covers(
        "    f(/* a /* ) */ ) */ 1)?;\n",
        "f(",
        0,
        "f(/* a /* ) */ ) */ 1)",
    );
    // The span reports end line/col of a multi-line call.
    let src = "    op(wg.isa(),\n       \"x\", &[], &[])?;\n";
    let s = widened(src, pos(src, "op(", 0));
    assert_eq!((s.line, s.col, s.end_line, s.end_col), (1, 5, 2, 22));
}

#[test]
fn span_columns_count_chars_not_bytes() {
    let src = "    let s = \"é→\"; foo(1);\n";
    let at = pos(src, "foo", 0);
    assert_eq!(at, (1, 19), "é and → are one column each");
    let s = widened(src, at);
    assert_eq!((s.line, s.col, s.end_line, s.end_col), (1, 19, 1, 25));
    assert_eq!(slice(src, s), "foo(1)");
    // Unicode identifiers widen like ASCII ones, and a parenthesis inside a Unicode string is skipped.
    covers("    let π = 3; naïve(π);\n", "naïve", 0, "naïve(π)");
    covers("    naïve(\"é)\", 1)?;\n", "naïve", 0, "naïve(\"é)\", 1)");
    covers("    naïve(1)?;\n", "naïve", 2, "naïve(1)");
}

#[test]
fn span_at_a_non_identifier_or_outside_the_source_is_a_point() {
    let src = "    let n = 1;\n";
    let (line, col) = pos(src, "=", 0);
    let s = widened(src, (line, col));
    assert_eq!((s.line, s.col, s.end_line, s.end_col), (1, col, 1, col + 1));
    assert_eq!(slice(src, s), "=");
    // Beyond the last line and beyond the end of a line.
    let s = widened(src, (99, 1));
    assert_eq!((s.line, s.col, s.end_line, s.end_col), (99, 1, 99, 2));
    let s = widened(src, (1, 400));
    assert_eq!((s.line, s.col, s.end_line, s.end_col), (1, 400, 1, 401));
}
