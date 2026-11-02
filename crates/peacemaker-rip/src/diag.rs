// SPDX-License-Identifier: Apache-2.0
//! Stable diagnostics for the evaluator, author driver and instruction builder.

use serde_json::{json, Value};
use std::fmt::Write;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
}

/// One-based line/column, with an exclusive end column.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    pub line: usize,
    pub col: usize,
    pub end_line: usize,
    pub end_col: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Diagnostic {
    pub code: &'static str,
    pub severity: Severity,
    pub message: String,
    pub file: String,
    pub span: Option<Span>,
    pub help: Option<String>,
    pub suggestion: Option<String>,
    pub notes: Vec<String>,
}

struct Rule {
    code: &'static str,
    matches: fn(&str) -> bool,
    help: fn(&str) -> Option<String>,
}

macro_rules! rule {
    ($code:literal, [$($needle:literal),+], $help:expr) => {
        Rule { code: $code, matches: |m| false $(|| m.contains($needle))+, help: $help }
    };
}

// Specific refusals precede broad evaluator and driver families. In particular,
// assembly errors may mention unknown mnemonics, and kernel exits have their own
// finish diagnostic rather than the general label diagnostic.
const RULES: &[Rule] = &[
    rule!(
        "R0701",
        [
            "assemble/link:",
            "assemble:",
            "assembly failed",
            "certification",
            "certify",
            "wait replay:",
            "built without the `certify` feature"
        ],
        |_| {
            Some("Use an instruction supported by the selected architecture and satisfy the launch and wait-replay contract before certification.".into())
        }
    ),
    rule!(
        "R0901",
        [
            "usage:",
            "bad arguments",
            "rustfmt",
            "cannot read",
            "cannot write",
            "No such file or directory",
            "Permission denied",
            "cannot start the evaluator thread",
            "tool error:"
        ],
        |_| {
            Some("Check the command arguments and file paths; install rustfmt when using `peacemaker-rip fmt`.".into())
        }
    ),
    rule!(
        "R0501",
        [
            "no `fn emit",
            "no `fn main",
            "`emit` takes (spec)",
            "`emit` returned Ok(",
            "`emit` must return"
        ],
        |_| {
            Some("Define `fn emit(spec: Spec) -> Result<Emitted, String>` and return `b.finish()` after ending the kernel.".into())
        }
    ),
    rule!(
        "R0207",
        ["panic:", "assertion failed", "the evaluator panicked"],
        |_| {
            Some("Fix the failed assertion or panic condition in the script; use the reported operand values to correct the computation.".into())
        }
    ),
    rule!(
        "R0601",
        ["unknown mnemonic", "unsupported mnemonic"],
        mnemonic_help
    ),
    rule!(
        "R0602",
        [
            "malformed operands",
            "bad operand",
            "missing operand",
            "too many operands"
        ],
        operand_help
    ),
    rule!(
        "R0603",
        [
            "op declares defs",
            "defs/uses disagree",
            "missing declared defs",
            "missing declared uses"
        ],
        effects_help
    ),
    rule!("R0604", ["read before it is defined"], definedness_help),
    rule!(
        "R0301",
        [
            "with an undrained LDS store",
            "publish with undrained LDS stores"
        ],
        undrained_help
    ),
    Rule {
        code: "R0302",
        matches: published_message,
        help: published_help,
    },
    rule!(
        "R0303",
        [
            "is not published:",
            "not published",
            "read of a ring buffer no barrier has published"
        ],
        unpublished_help
    ),
    rule!(
        "R0304",
        [
            "is workgroup scope: it cannot run under",
            "workgroup barrier after a wave-uniform kernel exit"
        ],
        |_| {
            Some("Move the operation outside wave- or lane-dependent control; use a WgCond from `wg.scmp_wg_uniform(..)` for workgroup-uniform branches.".into())
        }
    ),
    rule!(
        "R0305",
        [
            "no active Workgroup",
            "outside the Workgroup scope",
            "Workgroup scope is already open",
            "builder already belongs to a Workgroup",
            "must be the initializer of a `let`"
        ],
        |_| {
            Some("Create `let mut wg = Workgroup::new(&mut builder, arch)?;`. Its scope ends at the first later statement mentioning the builder variable, including `let b = wg.isa();` when the builder is named b; rename that local or move the body into `fn author<T: Target>(wg: &mut Wg<T>, ..)`.".into())
        }
    ),
    rule!("R0401", ["has no unique live owner"], register_help),
    rule!(
        "R0402",
        [
            "register budget",
            "register width",
            "register plan",
            "scratch pool",
            "misaligned SGPR",
            "shared lifetime",
            "lifetime label",
            "invalid lifetime",
            "lifetimes overlap",
            "live ranges",
            "VGPR bank mismatch",
            "VGPR parity mismatch",
            "invalid V8 pair"
        ],
        |_| {
            Some("Declare disjoint `p.v::<N>(name, base, live)?` / `p.s::<N>(name, base, live)?` ranges with width 1, 2, 4 or 8 inside the register budget; align SGPR tuples to their width and place lifetime endpoints.".into())
        }
    ),
    rule!(
        "R0403",
        [
            "scratch instructions are forbidden",
            "immediate",
            "single assembler statement",
            "changes control flow",
            "is a barrier: barriers go through",
            "clauses and global visibility require",
            "DS access must carry",
            "typed kernel:",
            "is not a branch or s_endpgm"
        ],
        |_| {
            Some("Use the typed Workgroup operation for barriers, branches and LDS access; pass one non-scratch instruction with an encodable immediate to `op(..)`.".into())
        }
    ),
    rule!(
        "R0404",
        [
            "counter/architecture mismatch",
            "outstanding memory operations",
            "back-edge ledger",
            "tag_lds_load:",
            "depctr_vm_vsrc requires"
        ],
        wait_help
    ),
    rule!(
        "R0405",
        [
            "kernel must end with s_endpgm",
            "instruction after s_endpgm",
            "kernel ends with a split barrier",
            "s_endpgm without a placed kernel exit"
        ],
        |_| {
            Some("Call `wg.end(exit)?` once, after waiting for split barriers; emit no instructions after the end and then return `b.finish()`.".into())
        }
    ),
    Rule {
        code: "R0405",
        matches: |m| m.contains("kernel exit") && m.contains("is never placed"),
        help: |_| {
            Some(
                "Place every reserved kernel exit with `wg.end(exit)?` before `b.finish()`.".into(),
            )
        },
    },
    rule!(
        "R0306",
        [
            "duplicate label",
            "branches only go forward",
            "forward target",
            "is already placed",
            "not the target of a branch",
            "not a reserved kernel exit",
            "only Workgroup::end places",
            "is a branch target: it takes no tail",
            "has no break_if",
            "emitted no body"
        ],
        |_| {
            Some("Use unique labels, place every forward target once after its branches, and terminate reserved exits with `wg.end(exit)?`; use loop APIs for back edges.".into())
        }
    ),
    Rule {
        code: "R0399",
        matches: driver_message,
        help: driver_help,
    },
    rule!("R0102", ["is not supported by the .rip evaluator"], |_| {
        Some("Replace this macro with supported evaluator expressions or a helper function; supported macros include format!, assert!, assert_eq! and matches!.".into())
    }),
    rule!("R0202", ["no method `"], name_help),
    rule!(
        "R0201",
        ["cannot find variable", "cannot find `", "unknown variant"],
        name_help
    ),
    rule!(
        "R0203",
        [
            "arguments, got",
            "argument, got",
            "needs two operands",
            "needs a condition",
            "needs a format string"
        ],
        |_| {
            Some("Pass exactly the number of arguments shown in the message, in the function's declared order.".into())
        }
    ),
    rule!(
        "R0204",
        [
            "mismatched integer types",
            "overflow",
            "divide by zero",
            "divisor of zero",
            "cannot negate an unsigned",
            "shift amount",
            "integer out of range",
            "value does not fit",
            "out of range for `"
        ],
        |_| {
            Some("Give both operands the same integer type with an explicit `as` cast (for example `x as u32`); keep the result in range and the divisor nonzero.".into())
        }
    ),
    rule!(
        "R0206",
        [
            "unsupported",
            "not implemented",
            "not supported",
            "recursion limit",
            "unbounded range",
            "refutable pattern",
            "did not diverge",
            "not an assignable place"
        ],
        |_| {
            Some("Rewrite the executed construct using supported Rust-shaped expressions, concrete values and helper functions.".into())
        }
    ),
    Rule {
        code: "R0101",
        matches: |m| m.starts_with("expected `"),
        help: |_| {
            Some("Repair the syntax at the highlighted token: check delimiters, punctuation, literals and the surrounding statement.".into())
        },
    },
    rule!(
        "R0101",
        [
            "cannot parse",
            "unexpected token",
            "unexpected end",
            "unclosed delimiter",
            "unterminated",
            "invalid digit",
            "invalid suffix",
            "not a valid",
            "lex error",
            "cannot convert",
            "invalid char",
            "unknown start of token",
            "expected an expression",
            "expected identifier",
            "expected one of",
            "expected curly braces",
            "expected parentheses",
            "expected square brackets"
        ],
        |_| {
            Some("Repair the syntax at the highlighted token: check delimiters, punctuation, literals and the surrounding statement.".into())
        }
    ),
    rule!(
        "R0205",
        [
            "expected",
            "found",
            "condition must be bool",
            "`?` needs",
            "cannot apply operator",
            "cannot order",
            "does not implement Display",
            "non-exhaustive match",
            "takes a region",
            "takes a ring",
            "takes a wave-uniform",
            "takes a WgCond",
            "takes an Exit",
            "returns Ok(more)",
            "out of bounds"
        ],
        |_| {
            Some("Use a value of the required type; apply `?` only to Result or Option and use a bool as a condition.".into())
        }
    ),
    Rule {
        code: "R0000",
        matches: |_| true,
        help: |_| {
            Some("Inspect the reported message and surrounding source; use `peacemaker-rip explain R0000` for unclassified failures.".into())
        },
    },
];

fn between<'a>(text: &'a str, before: &str, after: &str) -> Option<&'a str> {
    text.split_once(before)?
        .1
        .split_once(after)
        .map(|(value, _)| value)
}

fn undrained_help(m: &str) -> Option<String> {
    let region = between(m, "barrier would publish ", " with an undrained").unwrap_or("region");
    Some(format!("Drain the LDS store for {region}: `let drained = wg.wait(store)?;`, then publish it with `wg.barrier((ready({region}, drained),))?;`."))
}
fn published_message(m: &str) -> bool {
    m.contains("store to published")
        || (m.contains("Published")
            && [
                "cannot store into",
                "cannot join",
                "cannot split",
                "at relayout:",
            ]
            .iter()
            .any(|needle| m.contains(needle)))
}
fn published_help(m: &str) -> Option<String> {
    let region = ["cannot store into ", "cannot join ", "cannot split "]
        .iter()
        .find_map(|prefix| between(m, prefix, ":"))
        .or_else(|| m.split_once(" is ").map(|(region, _)| region))
        .unwrap_or("region");
    Some(format!("Retire {region} with `wg.barrier((retire({region}),))?;` before reusing it; a store wait alone does not retire other waves' readers."))
}
fn unpublished_help(m: &str) -> Option<String> {
    let region = m
        .split_once(" is not published:")
        .map(|(r, _)| r)
        .unwrap_or("region");
    Some(format!("Write {region}, drain its pending store with `wg.wait(store)?`, and publish it with `wg.barrier((ready({region}, drained),))?;` before reading."))
}
fn wait_help(m: &str) -> Option<String> {
    if m.contains("back-edge ledger must equal the entry ledger") {
        let head = between(m, "loop ", ":").unwrap_or("the loop");
        Some(format!("Drain {head}'s body stores before the back edge: call `wg.wait(store)?;` or emit an explicit counter wait at the end of the body so the back-edge ledger equals the entry ledger."))
    } else {
        Some("Use Counter::Load/Store/Ds/Km on gfx12 and Counter::Vm/Vs/Lgkm on gfx11; drain outstanding operations with the matching wait before finishing.".into())
    }
}
fn register_help(m: &str) -> Option<String> {
    let register = m.split_once(" at ")?.0;
    let kind = register.chars().next()?;
    if kind != 'v' && kind != 's' {
        return None;
    }
    let (base, width) = if let Some(range) = register[1..].strip_prefix('[') {
        let (base, last) = range.strip_suffix(']')?.split_once(':')?;
        let base: u16 = base.parse().ok()?;
        let last: u16 = last.parse().ok()?;
        (base, last.checked_sub(base)? + 1)
    } else {
        (register[1..].parse().ok()?, 1)
    };
    Some(format!("Declare {register} as one live range before constructing the builder: `p.{kind}::<{width}>(\"{kind}{base}\", {base}, Live::Whole)?;` (increase the plan's budget if needed)."))
}
fn effects_help(m: &str) -> Option<String> {
    if let Some((_, fix)) = m.rsplit_once(": write ") {
        Some(format!(
            "Declare every written/read register in the corresponding defs/uses list: use {fix}."
        ))
    } else {
        Some("Put every instruction destination in defs and every source in uses; missing declarations prevent the builder from inserting required waits.".into())
    }
}
fn definedness_help(m: &str) -> Option<String> {
    let registers = between(m, "register read before it is defined: ", " in `")
        .unwrap_or("each source register");
    Some(format!("Define {registers} on every incoming path before reading it; the workgroup X id is `ttmp9` on gfx12, but `s2` (declared in the register plan) on gfx11."))
}
fn operand_help(m: &str) -> Option<String> {
    if m.contains("not an inline constant") {
        let number = between(m, "bad operand \"", "\"")
            .or_else(|| between(m, "bad operand `", "`"))
            .or_else(|| {
                m.split_once("not an inline constant")
                    .and_then(|(before, _)| {
                        before
                            .split_whitespace()
                            .rev()
                            .map(|word| {
                                word.trim_matches(|c: char| {
                                    matches!(c, '`' | '\'' | '"' | ':' | ',' | '(' | ')')
                                })
                            })
                            .find(|word| word.parse::<i128>().is_ok())
                    })
            });
        if let Some(number) = number {
            return Some(format!("`{number}` is not an inline constant (inline integers are -16..64). Write it as a literal with `lit({number})`, which prints `0x..` (one literal per instruction). For VOP2 `_e32` the literal must be src0, or use `_e64`."));
        }
    }
    if m.contains("buffer_") && m.contains("gfx1201") && m.contains("SOFFSET does not fit") {
        return Some("On gfx12 a zero buffer soffset is spelled `null`, not `0`.".into());
    }
    Some("Use the selected architecture's operand order, register widths and legal inline constants for this instruction.".into())
}
fn mnemonic_help(m: &str) -> Option<String> {
    let suggestion =
        between(m, "did you mean `", "`").or_else(|| between(m, "did-you-mean `", "`"));
    Some(match suggestion {
        Some(name) => format!("Use `{name}` for the selected architecture."),
        None => "Use a mnemonic supported by the selected architecture; gfx12 uses `s_add_co_i32` rather than `s_add_i32`.".into(),
    })
}
fn driver_message(m: &str) -> bool {
    [
        "LDS",
        "another Driver",
        "backend targets",
        "split barrier",
        "barrier wait without signal",
        "unmatched barrier signal",
        "wait_arrived",
        "was consumed",
        "was re-laid out",
        "at relayout:",
        "cannot join",
        "cannot split",
        "a join needs",
        "a region cannot be joined",
        "a ring needs",
        "ring ",
        "begin_write of",
        "wait on ",
        "one barrier carries",
        "ready of ",
        "retire of ",
        "rotate of ",
        "prime of ",
        "retire_cur of ",
        "advance of ",
        "the arms end in different",
        "handoff",
        "SCC was redefined",
        "does not define SCC",
        "the paths cannot join",
        "the arms cannot join",
        "no split barrier to wait",
        "has one s_barrier",
        "ds_store requires",
        "ds_load requires",
        "ds_crosslane requires",
        "condition from another",
        "belongs to another Driver",
        "hazard and LDS state",
        "a store cannot cover the same place twice",
    ]
    .iter()
    .any(|needle| m.contains(needle))
}
fn driver_help(m: &str) -> Option<String> {
    Some(
        if m.contains("wait_arrived") || m.contains("split barrier") || m.contains("barrier signal")
        {
            "Pair each `wg.signal(..)?` with `wg.wait_arrived(arrived)?` before accessing its held regions or ending the kernel.".into()
        } else if m.contains("wait on ") {
            "Wait on the pending store token returned by `wg.ds_store(..)?`, not a Free or Published region.".into()
        } else if m.contains("SCC was redefined") {
            "Consume the Cond immediately after `wg.scmp(..)?`, before another instruction can redefine SCC.".into()
        } else if m.contains("LDS ownership") {
            "Make all control paths agree on region phases and ownership at the join or loop back edge.".into()
        } else {
            "Keep region, ring, condition and exit handles in the Workgroup that created them and satisfy the state transition named in the refusal.".into()
        },
    )
}

fn distance(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, x) in a.chars().enumerate() {
        let mut diagonal = row[0];
        row[0] = i + 1;
        for (j, &y) in b.iter().enumerate() {
            let old = row[j + 1];
            row[j + 1] = (row[j] + 1)
                .min(old + 1)
                .min(diagonal + usize::from(x != y));
            diagonal = old;
        }
    }
    row[b.len()]
}
fn closest_name(m: &str) -> Option<String> {
    let name = between(m, "`", "`")?;
    let mut candidates: Vec<&str> = Vec::new();
    // Messages may carry receiver-specific vocabulary, e.g. `(supported: f16, bf16)`.
    for marker in ["supported:", "available:", "known names:", "methods:"] {
        if let Some((_, names)) = m.split_once(marker) {
            candidates.extend(
                names
                    .split(|c: char| !(c.is_alphanumeric() || c == '_' || c == ':'))
                    .filter(|n| !n.is_empty()),
            );
        }
    }
    if candidates.is_empty() {
        candidates.extend(crate::native::COMMON_FNS);
        candidates.extend(crate::native::ASSOC_FNS);
        candidates.extend(crate::native::WORKGROUP_METHODS);
        candidates.extend([
            "s",
            "v",
            "scratch_pool",
            "next_free_vgpr",
            "next_free_sgpr",
            "finish",
            "wait",
            "position",
            "enable_delay_alu",
            "clause",
            "name",
            "gfx12",
            "buffer_offset_max",
            "pointer",
            "hidden",
            "validate",
            "memory",
            "step",
            "into_steady",
            "cur_index",
            "next_index",
            "depth",
            "starting_at",
            "phases",
            "phase",
            "branch_if",
            "branch_unless",
            "goto",
            "place",
        ]);
    }
    let (score, candidate) = candidates
        .into_iter()
        .filter(|candidate| *candidate != name)
        .map(|candidate| (distance(name, candidate), candidate))
        .min()?;
    (score <= name.chars().count().div_ceil(2).max(2)).then(|| candidate.to_owned())
}
fn name_help(m: &str) -> Option<String> {
    let name = between(m, "`", "`").unwrap_or("this name");
    Some(match closest_name(m) {
        Some(candidate) => format!("Did you mean `{candidate}` instead of `{name}`?"),
        None => format!("Declare `{name}` before use or use a name from the native vocabulary; check its spelling and receiver type."),
    })
}

impl Diagnostic {
    pub fn from_raw(raw: &str, file: &str, source: &str, severity: Severity) -> Diagnostic {
        let raw = raw
            .strip_prefix("error: ")
            .or_else(|| raw.strip_prefix("warning: "))
            .unwrap_or(raw);
        let mut message = raw;
        let mut filename = file;
        let mut span = None;
        for (i, ch) in raw.char_indices() {
            if ch != ':' {
                continue;
            }
            let Some((line, rest)) = raw[i + 1..].split_once(':') else {
                continue;
            };
            let Some((col, rest)) = rest.split_once(": ") else {
                continue;
            };
            let (Ok(line), Ok(col)) = (line.parse::<usize>(), col.parse::<usize>()) else {
                continue;
            };
            if line == 0 || col == 0 {
                continue;
            }
            filename = &raw[..i];
            message = rest;
            span = Some(widen(source, line, col));
            break;
        }
        let rule = RULES
            .iter()
            .find(|rule| (rule.matches)(message))
            .expect("fallback diagnostic rule");
        let suggestion = if matches!(rule.code, "R0201" | "R0202") {
            // The span includes a call, so only replace its name, preserving arguments.
            closest_name(message)
                .filter(|name| !name.contains("::"))
                .and_then(|name| {
                    span.and_then(|s| {
                        span_text(source, s).map(|text| {
                            let end = text
                                .char_indices()
                                .find(|(_, c)| !identifier(*c))
                                .map(|(i, _)| i)
                                .unwrap_or(text.len());
                            format!("{name}{}", &text[end..])
                        })
                    })
                })
        } else {
            None
        };
        Diagnostic {
            code: rule.code,
            severity,
            message: message.into(),
            file: filename.into(),
            span,
            help: (rule.help)(message),
            suggestion,
            notes: Vec::new(),
        }
    }

    pub fn render(&self, source: &str) -> String {
        let severity = match self.severity {
            Severity::Error => "error",
            Severity::Warning => "warning",
        };
        let mut text = format!("{severity}[{}]: {}\n", self.code, self.message);
        if let Some(span) = self.span {
            let _ = writeln!(text, " --> {}:{}:{}", self.file, span.line, span.col);
            let width = span.end_line.max(span.line).to_string().len();
            let _ = writeln!(text, "{}|", " ".repeat(width + 1));
            for (i, line) in source.lines().enumerate().skip(span.line.saturating_sub(1)) {
                let n = i + 1;
                if n > span.end_line {
                    break;
                }
                let start = if n == span.line {
                    span.col.saturating_sub(1)
                } else {
                    0
                };
                let end = if n == span.end_line {
                    span.end_col.saturating_sub(1)
                } else {
                    line.chars().count()
                };
                let count = end.min(line.chars().count()).saturating_sub(start).max(1);
                let padding: String = line
                    .chars()
                    .take(start)
                    .map(|c| if c == '\t' { '\t' } else { ' ' })
                    .collect();
                let _ = writeln!(text, "{n:>width$} | {line}");
                let _ = writeln!(
                    text,
                    "{} | {padding}{}",
                    " ".repeat(width),
                    "^".repeat(count)
                );
            }
        }
        if let Some(help) = &self.help {
            let _ = writeln!(text, "  = help: {help}");
        }
        for note in &self.notes {
            let _ = writeln!(text, "  = note: {note}");
        }
        let _ = writeln!(
            text,
            "  = note: details: peacemaker-rip explain {}",
            self.code
        );
        text
    }

    pub fn to_json(&self) -> Value {
        json!({
            "code": self.code,
            "severity": match self.severity { Severity::Error => "error", Severity::Warning => "warning" },
            "message": self.message, "file": self.file,
            "span": self.span.map(|s| json!({"line": s.line, "col": s.col, "end_line": s.end_line, "end_col": s.end_col})),
            "help": self.help, "suggestion": self.suggestion, "notes": self.notes,
            "explain": format!("peacemaker-rip explain {}", self.code),
        })
    }
}

fn identifier(c: char) -> bool {
    c == '_' || c.is_alphanumeric()
}
fn offset(source: &str, line: usize, col: usize) -> Option<usize> {
    let mut base = 0;
    for (i, text) in source.split_inclusive('\n').enumerate() {
        if i + 1 == line {
            return text
                .char_indices()
                .map(|(j, _)| j)
                .chain(std::iter::once(text.len()))
                .nth(col.checked_sub(1)?)
                .map(|j| base + j);
        }
        base += text.len();
    }
    None
}
fn position(source: &str, byte: usize) -> (usize, usize) {
    let prefix = &source[..byte];
    let line = prefix.bytes().filter(|b| *b == b'\n').count() + 1;
    let col = prefix.rsplit('\n').next().unwrap_or("").chars().count() + 1;
    (line, col)
}
fn span_text(source: &str, span: Span) -> Option<&str> {
    source.get(offset(source, span.line, span.col)?..offset(source, span.end_line, span.end_col)?)
}

// Skip literals and comments without treating parentheses inside them as syntax.
// A lone apostrophe is a lifetime, not a char literal.
fn skip_noncode(source: &str, i: usize) -> Option<usize> {
    let bytes = source.as_bytes();
    let rest = &source[i..];
    if rest.starts_with("//") {
        return Some(rest.find('\n').map(|n| i + n).unwrap_or(bytes.len()));
    }
    if rest.starts_with("/*") {
        let (mut j, mut depth) = (i + 2, 1);
        while j < bytes.len() {
            if bytes[j..].starts_with(b"/*") {
                depth += 1;
                j += 2;
            } else if bytes[j..].starts_with(b"*/") {
                depth -= 1;
                j += 2;
                if depth == 0 {
                    return Some(j);
                }
            } else {
                j += 1;
            }
        }
        return Some(bytes.len());
    }
    let raw_start = if rest.starts_with("br") || rest.starts_with("cr") {
        i + 2
    } else if rest.starts_with('r') {
        i + 1
    } else {
        i
    };
    if raw_start > i {
        let mut j = raw_start;
        while bytes.get(j) == Some(&b'#') {
            j += 1;
        }
        if bytes.get(j) == Some(&b'"') {
            let hashes = j - raw_start;
            j += 1;
            while j < bytes.len() {
                if bytes[j] == b'"'
                    && bytes
                        .get(j + 1..j + 1 + hashes)
                        .is_some_and(|s| s.iter().all(|c| *c == b'#'))
                {
                    return Some(j + 1 + hashes);
                }
                j += 1;
            }
            return Some(bytes.len());
        }
    }
    let quote = if rest.starts_with("b\"") || rest.starts_with("c\"") || rest.starts_with("b'") {
        i + 1
    } else {
        i
    };
    if bytes.get(quote) == Some(&b'"') {
        let mut j = quote + 1;
        while j < bytes.len() {
            if bytes[j] == b'\\' {
                j = (j + 2).min(bytes.len());
            } else if bytes[j] == b'"' {
                return Some(j + 1);
            } else {
                j += 1;
            }
        }
        return Some(bytes.len());
    }
    if bytes.get(quote) == Some(&b'\'') {
        let start = quote + 1;
        let mut j = start;
        if bytes.get(j) == Some(&b'\\') {
            j += 2;
            if bytes.get(start + 1) == Some(&b'u') && bytes.get(j) == Some(&b'{') {
                j = source[j..]
                    .find('}')
                    .map(|n| j + n + 1)
                    .unwrap_or(bytes.len());
            } else if bytes.get(start + 1) == Some(&b'x') {
                j += 2;
            }
        } else if let Some(ch) = source.get(j..).and_then(|s| s.chars().next()) {
            j += ch.len_utf8();
        }
        if bytes.get(j) == Some(&b'\'') {
            return Some(j + 1);
        }
    }
    None
}
fn trivia(source: &str, mut i: usize) -> usize {
    while i < source.len() {
        let c = source[i..].chars().next().unwrap();
        if c.is_whitespace() {
            i += c.len_utf8();
        } else if source[i..].starts_with("//") || source[i..].starts_with("/*") {
            i = skip_noncode(source, i).unwrap_or(i);
        } else {
            break;
        }
    }
    i
}
fn matching(source: &str, start: usize, open: u8, close: u8) -> Option<usize> {
    let (mut i, mut depth) = (start, 0usize);
    while i < source.len() {
        if let Some(end) = skip_noncode(source, i) {
            i = end;
            continue;
        }
        let byte = source.as_bytes()[i];
        if open == b'<' {
            let nested_close = match byte {
                b'(' => Some(b')'),
                b'[' => Some(b']'),
                b'{' => Some(b'}'),
                _ => None,
            };
            if let Some(nested_close) = nested_close {
                i = matching(source, i, byte, nested_close)?;
                continue;
            }
            // A function type's return arrow is not a generic delimiter.
            if byte == b'>' && i > 0 && source.as_bytes()[i - 1] == b'-' {
                i += 1;
                continue;
            }
        }
        if byte == open {
            depth += 1;
        } else if byte == close {
            depth = depth.checked_sub(1)?;
            if depth == 0 {
                return Some(i + 1);
            }
        }
        i += source[i..].chars().next()?.len_utf8();
    }
    None
}
fn widen(source: &str, line: usize, col: usize) -> Span {
    let point = Span {
        line,
        col,
        end_line: line,
        end_col: col + 1,
    };
    let Some(mut start) = offset(source, line, col) else {
        return point;
    };
    if start >= source.len() {
        return point;
    }
    let Some(first) = source[start..].chars().next() else {
        return point;
    };
    if !identifier(first) {
        return point;
    }
    // A caller may give the middle of an identifier, so include its beginning.
    while let Some((i, c)) = source[..start].char_indices().next_back() {
        if !identifier(c) {
            break;
        }
        start = i;
    }
    let mut end = start;
    for c in source[start..].chars() {
        if !identifier(c) {
            break;
        }
        end += c.len_utf8();
    }
    let mut next = trivia(source, end);
    if source[next..].starts_with("::") {
        let generic = trivia(source, next + 2);
        if source[generic..].starts_with('<') {
            if let Some(after) = matching(source, generic, b'<', b'>') {
                next = trivia(source, after);
            }
        }
    }
    if source[next..].starts_with('(') {
        if let Some(after) = matching(source, next, b'(', b')') {
            end = after;
        }
    }
    let (line, col) = position(source, start);
    let (end_line, end_col) = position(source, end);
    Span {
        line,
        col,
        end_line,
        end_col,
    }
}

pub struct CodeDoc {
    pub code: &'static str,
    pub title: &'static str,
    pub body: &'static str,
}
const DOCS: &str = include_str!("../docs/ERRORS.md");

pub fn explain(code: &str) -> Option<CodeDoc> {
    codes().into_iter().find(|doc| doc.code == code)
}

pub fn codes() -> Vec<CodeDoc> {
    DOCS.split("\n## ")
        .skip(1)
        .filter_map(|section| {
            let (header, body) = section.split_once('\n')?;
            let (code, title) = header.split_once(": ")?;
            Some(CodeDoc {
                code,
                title,
                body: body.trim(),
            })
        })
        .collect()
}
