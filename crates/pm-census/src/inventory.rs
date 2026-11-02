// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Inventory of the code objects `rdna-compute` embeds with `include_bytes!` (the PM objects:
//! `kernels/**/*.hxaco` / `*.co`) and of the hipcc kernel names declared in `kernels/src/**/*.hip`,
//! plus the per-kernel origin join against a rocprof [`Trace`].
//!
//! ## What a loaded code object's `memory_size` is
//!
//! rocprofv3 reports `code_objects[].memory_size` as the byte length of the **device ELF** the
//! HSA loader was handed, not the length of the offload bundle (`.hxaco`) that wraps it. Verified
//! on the r9700 prefill capture (binary built from `5fde8959db`): the MQ6 PM code object reports
//! `memory_size = 46792` = the `hipv4-amdgcn-amd-amdhsa--gfx1201` payload of
//! `kernels/gemm_mq6g256v2_mmq_i8_pm_gfx1201.hxaco`, while that file is 50888 bytes; likewise
//! 37688 (file 41784) for `qwen4_moe_down_gp_pm_gfx1201.hxaco` and 110048 (file 114144) for
//! `qwen4_moe_iu4_sym_pm_gfx1201.hxaco`. `load_size` (57344 for the MQ6 object) is the
//! page-rounded loaded image and does not correspond to any inventory length, so it is never used
//! as a fallback.
//!
//! ELF and offload-bundle structure is read here directly (the bundle layout and its constants
//! come from [`peacemaker_lift::bundle`]): the lifter's `Envelope`/`Bundle` codecs are typed in
//! `peacemaker-ir` and re-serialise-verify every kernel, which this census does not need and
//! cannot name without a direct `peacemaker-ir` dependency.

use std::collections::{BTreeMap, BTreeSet};
use std::io::{BufRead, BufReader, Read, Write};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use peacemaker_lift::bundle::{COMPRESSED_MAGIC, HIP_DEVICE_TRIPLE_PREFIX, MAGIC};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::rocprof::{KernelSymbol, Trace};

pub use hipfire_atlas::workload::OriginV1 as Origin;

/// Directory whose Rust sources declare the embedded objects.
const RUST_SRC_DIR: &str = "crates/rdna-compute/src";
/// Directory of the hipcc kernel sources.
const HIP_SRC_DIR: &str = "kernels/src";
/// Embedded objects must live under this directory.
const OBJECT_DIR: &str = "kernels/";

/// One device (AMDGPU) ELF of a PM object.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceElf {
    /// gfx target, e.g. `gfx1201` (bundle target id without features, or from `e_flags`).
    pub arch: String,
    /// Byte length of the device ELF: the value HIP reports as the code object `memory_size`.
    pub bytes: u64,
    pub sha256: String,
    /// Kernel symbols (`<name>.kd` descriptors) without the suffix, sorted.
    pub symbols: Vec<String>,
}

/// An object file embedded by `include_bytes!` under `kernels/`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PmObject {
    /// Repo-relative path.
    pub path: String,
    /// SHA-256 of the whole file.
    pub sha256: String,
    pub bytes: u64,
    /// `file:line` of every `include_bytes!` (or macro invocation expanding to one).
    pub declared_at: Vec<String>,
    pub device_elfs: Vec<DeviceElf>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Inventory {
    pub repo: String,
    /// Resolved commit the files were read from; `None` = working tree.
    pub commit: Option<String>,
    pub pm_objects: Vec<PmObject>,
    /// Names of the non-template `__global__` kernels in `kernels/src/**/*.hip`.
    pub hip_kernel_names: BTreeSet<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Attribution {
    pub origin: Origin,
    /// The single PM object the kernel was attributed to, when unambiguous.
    pub object_path: Option<String>,
    pub object_sha256: Option<String>,
    /// How the origin was decided, e.g. `code_object_size+symbol: ...`, `name_only: ...`,
    /// `hip_source_name, no PM export`, `unresolved: ...`.
    pub join: String,
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

// ---------------------------------------------------------------------------------------
// Source access: working tree or one commit (one `git cat-file --batch` process)
// ---------------------------------------------------------------------------------------

struct Git {
    repo: PathBuf,
    commit: String,
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl Drop for Git {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

enum Source {
    Tree(PathBuf),
    Git(Box<Git>),
}

fn git_output(repo: &Path, args: &[&str]) -> Result<Vec<u8>, String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(args)
        .output()
        .map_err(|e| format!("git {}: {e}", args.join(" ")))?;
    if !out.status.success() {
        return Err(format!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(out.stdout)
}

impl Source {
    fn open(repo: &Path, commit: Option<&str>) -> Result<Source, String> {
        let Some(commit) = commit else {
            if !repo.is_dir() {
                return Err(format!("{}: not a directory", repo.display()));
            }
            return Ok(Source::Tree(repo.to_path_buf()));
        };
        let rev = format!("{commit}^{{commit}}");
        let resolved = git_output(repo, &["rev-parse", "--verify", &rev])?;
        let resolved = String::from_utf8_lossy(&resolved).trim().to_string();
        let mut child = Command::new("git")
            .arg("-C")
            .arg(repo)
            .args(["cat-file", "--batch"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("git cat-file --batch: {e}"))?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| "git cat-file --batch: no stdin".to_string())?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "git cat-file --batch: no stdout".to_string())?;
        Ok(Source::Git(Box::new(Git {
            repo: repo.to_path_buf(),
            commit: resolved,
            child,
            stdin,
            stdout: BufReader::new(stdout),
        })))
    }

    fn commit(&self) -> Option<String> {
        match self {
            Source::Tree(_) => None,
            Source::Git(g) => Some(g.commit.clone()),
        }
    }

    /// Repo-relative paths of every file under `dir`, sorted, `/`-separated.
    fn list(&self, dir: &str) -> Result<Vec<String>, String> {
        match self {
            Source::Tree(root) => {
                let mut out = Vec::new();
                walk(root, dir, &mut out)?;
                out.sort();
                Ok(out)
            }
            Source::Git(g) => {
                let raw = git_output(
                    &g.repo,
                    &["ls-tree", "-r", "-z", "--name-only", &g.commit, "--", dir],
                )?;
                let mut out = Vec::new();
                for name in raw.split(|&b| b == 0).filter(|n| !n.is_empty()) {
                    out.push(
                        String::from_utf8(name.to_vec())
                            .map_err(|_| format!("non-UTF-8 path under {dir} at {}", g.commit))?,
                    );
                }
                if out.is_empty() {
                    return Err(format!("no files under {dir} at {}", g.commit));
                }
                out.sort();
                Ok(out)
            }
        }
    }

    fn read(&mut self, path: &str) -> Result<Vec<u8>, String> {
        match self {
            Source::Tree(root) => {
                let full = root.join(path);
                std::fs::read(&full).map_err(|e| format!("{}: {e}", full.display()))
            }
            Source::Git(g) => {
                let spec = format!("{}:{path}", g.commit);
                writeln!(g.stdin, "{spec}")
                    .and_then(|()| g.stdin.flush())
                    .map_err(|e| format!("git cat-file: {e}"))?;
                let mut header = String::new();
                let n = g
                    .stdout
                    .read_line(&mut header)
                    .map_err(|e| format!("git cat-file: {e}"))?;
                if n == 0 {
                    return Err(format!("git cat-file closed while reading {spec}"));
                }
                let header = header.trim_end();
                if header.ends_with(" missing") {
                    return Err(format!("{path}: not present at {}", g.commit));
                }
                let fields: Vec<&str> = header.rsplitn(3, ' ').collect();
                // rsplitn yields [size, type, oid-or-spec]
                match fields.as_slice() {
                    [size, "blob", _] => {
                        let size: usize = size
                            .parse()
                            .map_err(|_| format!("git cat-file: bad size in {header:?}"))?;
                        let mut buf = vec![0u8; size];
                        g.stdout
                            .read_exact(&mut buf)
                            .map_err(|e| format!("git cat-file: {e}"))?;
                        let mut newline = [0u8; 1];
                        g.stdout
                            .read_exact(&mut newline)
                            .map_err(|e| format!("git cat-file: {e}"))?;
                        Ok(buf)
                    }
                    _ => Err(format!(
                        "{path}: not a blob at {} (git replied {header:?})",
                        g.commit
                    )),
                }
            }
        }
    }
}

fn walk(root: &Path, rel: &str, out: &mut Vec<String>) -> Result<(), String> {
    let dir = root.join(rel);
    let entries = std::fs::read_dir(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    for entry in entries {
        let entry = entry.map_err(|e| format!("{}: {e}", dir.display()))?;
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| format!("{}: non-UTF-8 file name", dir.display()))?;
        let child = format!("{rel}/{name}");
        let Ok(meta) = std::fs::metadata(entry.path()) else {
            continue; // dangling symlink
        };
        if meta.is_dir() {
            walk(root, &child, out)?;
        } else if meta.is_file() {
            out.push(child);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------
// Rust lexer: just enough to find `include_bytes!` / `macro_rules!` without being fooled by
// comments, strings, raw strings, char literals or lifetimes.
// ---------------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq)]
enum Tk {
    Ident(String),
    Str(String),
    Punct(char),
}

#[derive(Clone, Debug)]
struct Token {
    tk: Tk,
    line: usize,
}

fn is_punct(t: &Token, c: char) -> bool {
    t.tk == Tk::Punct(c)
}

fn is_ident(t: &Token, s: &str) -> bool {
    matches!(&t.tk, Tk::Ident(i) if i == s)
}

fn is_open(t: &Token) -> bool {
    matches!(t.tk, Tk::Punct('(' | '[' | '{'))
}

fn is_close(t: &Token) -> bool {
    matches!(t.tk, Tk::Punct(')' | ']' | '}'))
}

/// Parses a non-raw string literal whose opening quote is `c[quote]`; returns its value and the
/// index after the closing quote.
fn lex_string(c: &[char], quote: usize, line: &mut usize) -> (String, usize) {
    let n = c.len();
    let mut i = quote + 1;
    let mut s = String::new();
    while i < n && c[i] != '"' {
        match c[i] {
            '\\' if i + 1 < n => {
                i += 1;
                match c[i] {
                    '\n' => {
                        // line continuation: skip the newline and the next line's indentation
                        *line += 1;
                        i += 1;
                        while i < n && c[i].is_whitespace() {
                            if c[i] == '\n' {
                                *line += 1;
                            }
                            i += 1;
                        }
                        continue;
                    }
                    'n' => s.push('\n'),
                    't' => s.push('\t'),
                    'r' => s.push('\r'),
                    '0' => s.push('\0'),
                    other => s.push(other),
                }
                i += 1;
            }
            '\n' => {
                *line += 1;
                s.push('\n');
                i += 1;
            }
            ch => {
                s.push(ch);
                i += 1;
            }
        }
    }
    (s, (i + 1).min(n))
}

fn lex_rust(text: &str) -> Vec<Token> {
    let c: Vec<char> = text.chars().collect();
    let n = c.len();
    let mut i = 0;
    let mut line = 1;
    let mut out = Vec::new();
    while i < n {
        let ch = c[i];
        if ch == '\n' {
            line += 1;
            i += 1;
            continue;
        }
        if ch.is_whitespace() {
            i += 1;
            continue;
        }
        if ch == '/' && i + 1 < n && c[i + 1] == '/' {
            while i < n && c[i] != '\n' {
                i += 1;
            }
            continue;
        }
        if ch == '/' && i + 1 < n && c[i + 1] == '*' {
            let mut depth = 1;
            i += 2;
            while i < n && depth > 0 {
                if c[i] == '\n' {
                    line += 1;
                    i += 1;
                } else if c[i] == '/' && i + 1 < n && c[i + 1] == '*' {
                    depth += 1;
                    i += 2;
                } else if c[i] == '*' && i + 1 < n && c[i + 1] == '/' {
                    depth -= 1;
                    i += 2;
                } else {
                    i += 1;
                }
            }
            continue;
        }
        if ch == 'r' || ch == 'b' {
            // `b'x'` byte char: drop the prefix, the char-literal arm below takes over.
            if ch == 'b' && i + 1 < n && c[i + 1] == '\'' {
                i += 1;
                continue;
            }
            let mut j = i;
            if c[j] == 'b' {
                j += 1;
            }
            let raw = j < n && c[j] == 'r';
            if raw {
                j += 1;
            }
            let mut hashes = 0;
            if raw {
                while j < n && c[j] == '#' {
                    hashes += 1;
                    j += 1;
                }
            }
            if j < n && c[j] == '"' && (raw || j == i + 1) {
                let start_line = line;
                if raw {
                    let start = j + 1;
                    let mut k = start;
                    let end = loop {
                        if k >= n {
                            break n;
                        }
                        if c[k] == '"'
                            && c[k + 1..]
                                .iter()
                                .take(hashes)
                                .filter(|&&h| h == '#')
                                .count()
                                == hashes
                        {
                            break k;
                        }
                        if c[k] == '\n' {
                            line += 1;
                        }
                        k += 1;
                    };
                    out.push(Token {
                        tk: Tk::Str(c[start..end].iter().collect()),
                        line: start_line,
                    });
                    i = (end + 1 + hashes).min(n);
                } else {
                    let (s, next) = lex_string(&c, j, &mut line);
                    out.push(Token {
                        tk: Tk::Str(s),
                        line: start_line,
                    });
                    i = next;
                }
                continue;
            }
        }
        if ch == '"' {
            let start_line = line;
            let (s, next) = lex_string(&c, i, &mut line);
            out.push(Token {
                tk: Tk::Str(s),
                line: start_line,
            });
            i = next;
            continue;
        }
        if ch == '\'' {
            if i + 1 < n && c[i + 1] == '\\' {
                // escaped char literal: skip to the closing quote
                let mut j = i + 3;
                while j < n && c[j] != '\'' {
                    j += 1;
                }
                i = j + 1;
            } else if i + 2 < n && c[i + 2] == '\'' {
                i += 3;
            } else {
                i += 1; // lifetime / label
            }
            continue;
        }
        if ch.is_alphabetic() || ch == '_' {
            let mut j = i;
            while j < n && (c[j].is_alphanumeric() || c[j] == '_') {
                j += 1;
            }
            out.push(Token {
                tk: Tk::Ident(c[i..j].iter().collect()),
                line,
            });
            i = j;
            continue;
        }
        if ch.is_ascii_digit() {
            while i < n && (c[i].is_alphanumeric() || c[i] == '_') {
                i += 1;
            }
            continue;
        }
        out.push(Token {
            tk: Tk::Punct(ch),
            line,
        });
        i += 1;
    }
    out
}

/// Index of the bracket closing the one at `open`.
fn group_end(toks: &[Token], open: usize) -> Result<usize, String> {
    let mut depth = 0usize;
    for (i, t) in toks.iter().enumerate().skip(open) {
        if is_open(t) {
            depth += 1;
        } else if is_close(t) {
            depth = depth
                .checked_sub(1)
                .ok_or_else(|| format!("line {}: unbalanced bracket", t.line))?;
            if depth == 0 {
                return Ok(i);
            }
        }
    }
    Err(format!(
        "line {}: bracket is never closed",
        toks.get(open).map_or(0, |t| t.line)
    ))
}

/// Splits at commas that are not inside any bracket.
fn split_top(toks: &[Token]) -> Vec<&[Token]> {
    let mut pieces = Vec::new();
    let mut depth = 0i32;
    let mut start = 0;
    for (i, t) in toks.iter().enumerate() {
        if is_open(t) {
            depth += 1;
        } else if is_close(t) {
            depth -= 1;
        } else if depth == 0 && is_punct(t, ',') {
            pieces.push(&toks[start..i]);
            start = i + 1;
        }
    }
    pieces.push(&toks[start..]);
    pieces
}

enum Part {
    Lit(String),
    Var(String),
}

/// The path expression inside `include_bytes!(...)`: a string literal, a `$placeholder`, or a
/// `concat!` of those. Anything else is an error: an unresolved expression would silently drop
/// an embedded object from the inventory.
fn parse_path_expr(file: &str, line: usize, arg: &[Token]) -> Result<Vec<Part>, String> {
    let unsupported = || {
        format!(
            "{file}:{line}: include_bytes! argument must be a string literal, a concat! of \
             literals/macro placeholders, or a macro placeholder"
        )
    };
    let atom = |toks: &[Token]| -> Option<Part> {
        match toks {
            [Token { tk: Tk::Str(s), .. }] => Some(Part::Lit(s.clone())),
            [Token {
                tk: Tk::Punct('$'), ..
            }, Token {
                tk: Tk::Ident(v), ..
            }] => Some(Part::Var(v.clone())),
            _ => None,
        }
    };
    let mut a = arg;
    while let Some((last, rest)) = a.split_last() {
        if is_punct(last, ',') {
            a = rest;
        } else {
            break;
        }
    }
    if let Some(part) = atom(a) {
        return Ok(vec![part]);
    }
    if a.len() >= 4
        && is_ident(&a[0], "concat")
        && is_punct(&a[1], '!')
        && is_open(&a[2])
        && group_end(a, 2)? == a.len() - 1
    {
        let inner = &a[3..a.len() - 1];
        let mut parts = Vec::new();
        for piece in split_top(inner) {
            if piece.is_empty() {
                continue;
            }
            parts.push(atom(piece).ok_or_else(unsupported)?);
        }
        return Ok(parts);
    }
    Err(unsupported())
}

struct Arm {
    pattern: Range<usize>,
}

struct MacroDef {
    name: String,
    /// Token range from `macro_rules` to the closing brace of the body.
    span: Range<usize>,
    arms: Vec<Arm>,
}

fn find_macros(toks: &[Token]) -> Result<Vec<MacroDef>, String> {
    let mut out = Vec::new();
    for i in 0..toks.len() {
        if !(is_ident(&toks[i], "macro_rules")
            && toks.get(i + 1).is_some_and(|t| is_punct(t, '!'))
            && toks.get(i + 3).is_some_and(is_open))
        {
            continue;
        }
        let Some(Tk::Ident(name)) = toks.get(i + 2).map(|t| t.tk.clone()) else {
            continue;
        };
        let body_open = i + 3;
        let body_close = group_end(toks, body_open)?;
        let mut arms = Vec::new();
        let mut at = body_open + 1;
        while at < body_close {
            let parsed = (|| {
                if !is_open(&toks[at]) {
                    return None;
                }
                let pat_end = group_end(toks, at).ok()?;
                let (a, b) = toks.get(pat_end + 1).zip(toks.get(pat_end + 2))?;
                if !(is_punct(a, '=') && is_punct(b, '>'))
                    || !toks.get(pat_end + 3).is_some_and(is_open)
                {
                    return None;
                }
                let arm_end = group_end(toks, pat_end + 3).ok()?;
                Some((pat_end, arm_end))
            })();
            // A macro whose arms this scanner cannot read only matters if it contains an
            // `include_bytes!`; `scan_includes` reports that case.
            let Some((pat_end, arm_end)) = parsed else {
                arms.clear();
                break;
            };
            arms.push(Arm {
                pattern: at + 1..pat_end,
            });
            at = arm_end + 1;
            if at < body_close && is_punct(&toks[at], ';') {
                at += 1;
            }
        }
        out.push(MacroDef {
            name,
            span: i..body_close + 1,
            arms,
        });
    }
    Ok(out)
}

/// For each top-level comma-separated piece of a macro pattern, the name of its
/// `$name:literal` placeholder (if it has one).
fn pattern_slots(pattern: &[Token]) -> Vec<Option<String>> {
    split_top(pattern)
        .into_iter()
        .map(|piece| {
            piece
                .windows(4)
                .find_map(|w| match (&w[0].tk, &w[1].tk, &w[2].tk, &w[3].tk) {
                    (Tk::Punct('$'), Tk::Ident(name), Tk::Punct(':'), Tk::Ident(kind))
                        if kind == "literal" =>
                    {
                        Some(name.clone())
                    }
                    _ => None,
                })
        })
        .collect()
}

/// `file`-relative path -> repo-relative path, lexically normalised.
fn resolve_path(file: &str, rel: &str) -> Result<String, String> {
    if rel.starts_with('/') {
        return Err(format!("{file}: absolute include_bytes! path {rel:?}"));
    }
    let dir = file.rsplit_once('/').map_or("", |(d, _)| d);
    let mut stack: Vec<&str> = Vec::new();
    for part in dir.split('/').chain(rel.split('/')) {
        match part {
            "" | "." => {}
            ".." => {
                if stack.pop().is_none() {
                    return Err(format!(
                        "{file}: include_bytes! path {rel:?} leaves the repo"
                    ));
                }
            }
            p => stack.push(p),
        }
    }
    Ok(stack.join("/"))
}

fn is_embedded_object(path: &str) -> bool {
    path.starts_with(OBJECT_DIR) && (path.ends_with(".hxaco") || path.ends_with(".co"))
}

/// `(repo-relative object path, line)` of every `include_bytes!` in `text` (the contents of the
/// repo-relative `file`) that embeds a `kernels/**/*.hxaco|.co` object. Macro-expanded includes
/// are resolved through the macro's invocations and reported at the invocation line.
fn scan_includes(file: &str, text: &str) -> Result<Vec<(String, usize)>, String> {
    let toks = lex_rust(text);
    let macros = find_macros(&toks)?;
    let mut found = Vec::new();
    let mut push = |path: String, line: usize| {
        if is_embedded_object(&path) {
            found.push((path, line));
        }
    };
    for i in 0..toks.len() {
        if !(is_ident(&toks[i], "include_bytes")
            && toks.get(i + 1).is_some_and(|t| is_punct(t, '!'))
            && toks.get(i + 2).is_some_and(is_open))
        {
            continue;
        }
        let line = toks[i].line;
        let close = group_end(&toks, i + 2)?;
        let parts = parse_path_expr(file, line, &toks[i + 3..close])?;
        if parts.iter().all(|p| matches!(p, Part::Lit(_))) {
            let rel: String = parts
                .iter()
                .map(|p| match p {
                    Part::Lit(s) => s.as_str(),
                    Part::Var(_) => "",
                })
                .collect();
            push(resolve_path(file, &rel)?, line);
            continue;
        }
        let mac = macros.iter().find(|m| m.span.contains(&i)).ok_or_else(|| {
            format!("{file}:{line}: macro placeholder outside a macro_rules body")
        })?;
        if mac.arms.len() != 1 {
            return Err(format!(
                "{file}:{line}: include_bytes! inside macro `{}`, which does not have exactly one \
                 readable macro_rules arm",
                mac.name
            ));
        }
        let slots = pattern_slots(&toks[mac.arms[0].pattern.clone()]);
        for j in 0..toks.len() {
            if mac.span.contains(&j)
                || !is_ident(&toks[j], &mac.name)
                || !toks.get(j + 1).is_some_and(|t| is_punct(t, '!'))
                || !toks.get(j + 2).is_some_and(is_open)
            {
                continue;
            }
            let end = group_end(&toks, j + 2)?;
            let args = split_top(&toks[j + 3..end]);
            let mut rel = String::new();
            for part in &parts {
                match part {
                    Part::Lit(s) => rel.push_str(s),
                    Part::Var(var) => {
                        let slot = slots
                            .iter()
                            .position(|s| s.as_deref() == Some(var.as_str()))
                            .ok_or_else(|| {
                                format!(
                                    "{file}:{line}: `${var}` is not a `$name:literal` placeholder \
                                     of macro `{}`",
                                    mac.name
                                )
                            })?;
                        let lit = args
                            .get(slot)
                            .and_then(|piece| {
                                piece.iter().rev().find_map(|t| match &t.tk {
                                    Tk::Str(s) => Some(s.as_str()),
                                    _ => None,
                                })
                            })
                            .ok_or_else(|| {
                                format!(
                                    "{file}:{}: invocation of `{}` has no string literal for `${var}`",
                                    toks[j].line, mac.name
                                )
                            })?;
                        rel.push_str(lit);
                    }
                }
            }
            push(resolve_path(file, &rel)?, toks[j].line);
        }
    }
    Ok(found)
}

// ---------------------------------------------------------------------------------------
// Code objects: offload bundle + AMDGPU ELF
// ---------------------------------------------------------------------------------------

fn rd<const N: usize>(b: &[u8], off: usize, what: &str) -> Result<[u8; N], String> {
    off.checked_add(N)
        .and_then(|end| b.get(off..end))
        .and_then(|s| s.try_into().ok())
        .ok_or_else(|| format!("truncated {what} at byte {off}"))
}

fn rd_u16(b: &[u8], off: usize, what: &str) -> Result<u16, String> {
    rd::<2>(b, off, what).map(u16::from_le_bytes)
}

fn rd_u32(b: &[u8], off: usize, what: &str) -> Result<u32, String> {
    rd::<4>(b, off, what).map(u32::from_le_bytes)
}

fn rd_u64(b: &[u8], off: usize, what: &str) -> Result<u64, String> {
    rd::<8>(b, off, what).map(u64::from_le_bytes)
}

fn slice_at<'a>(b: &'a [u8], off: u64, len: u64, what: &str) -> Result<&'a [u8], String> {
    off.checked_add(len)
        .filter(|&end| end <= b.len() as u64)
        .map(|end| &b[off as usize..end as usize])
        .ok_or_else(|| {
            format!(
                "{what} [{off:#x}, +{len:#x}) lies outside the {}-byte file",
                b.len()
            )
        })
}

/// `EF_AMDGPU_MACH` -> target name, for the targets hipfire builds or inspects.
fn mach_arch(mach: u32) -> Option<&'static str> {
    Some(match mach {
        0x2c => "gfx900",
        0x2f => "gfx906",
        0x30 => "gfx908",
        0x3f => "gfx90a",
        0x4c => "gfx942",
        0x4f => "gfx950",
        0x33 => "gfx1010",
        0x36 => "gfx1030",
        0x41 => "gfx1100",
        0x46 => "gfx1101",
        0x47 => "gfx1102",
        0x44 => "gfx1103",
        0x43 => "gfx1150",
        0x4a => "gfx1151",
        0x48 => "gfx1200",
        0x4e => "gfx1201",
        _ => return None,
    })
}

const EM_AMDGPU: u16 = 224;
const SHT_SYMTAB: u32 = 2;
const SHT_DYNSYM: u32 = 11;
const ELF64_SHDR: usize = 64;
const ELF64_SYM: u64 = 24;

#[derive(Debug)]
struct ElfInfo {
    /// Target from `e_flags`; `None` for machines outside [`mach_arch`].
    arch: Option<&'static str>,
    mach: u32,
    /// `.kd` descriptor symbols without the suffix, sorted, unique.
    symbols: Vec<String>,
}

fn elf_info(elf: &[u8]) -> Result<ElfInfo, String> {
    if elf.get(..4) != Some(b"\x7fELF".as_slice()) {
        return Err("not an ELF file".into());
    }
    if elf.get(4) != Some(&2) || elf.get(5) != Some(&1) {
        return Err("not ELFCLASS64 / little-endian".into());
    }
    let machine = rd_u16(elf, 18, "e_machine")?;
    if machine != EM_AMDGPU {
        return Err(format!(
            "e_machine {machine} is not EM_AMDGPU ({EM_AMDGPU})"
        ));
    }
    let mach = rd_u32(elf, 48, "e_flags")? & 0xff;
    let shoff = rd_u64(elf, 40, "e_shoff")?;
    let shentsize = rd_u16(elf, 58, "e_shentsize")?;
    let shnum = rd_u16(elf, 60, "e_shnum")? as usize;
    let mut symbols = BTreeSet::new();
    if shnum != 0 {
        if usize::from(shentsize) != ELF64_SHDR {
            return Err(format!("e_shentsize {shentsize} is not {ELF64_SHDR}"));
        }
        // (type, offset, size, link, entsize) of every section header
        let mut sections = Vec::with_capacity(shnum);
        for i in 0..shnum {
            let base = (shoff as usize)
                .checked_add(i * ELF64_SHDR)
                .ok_or_else(|| "section header offset overflows".to_string())?;
            sections.push((
                rd_u32(elf, base + 4, "sh_type")?,
                rd_u64(elf, base + 24, "sh_offset")?,
                rd_u64(elf, base + 32, "sh_size")?,
                rd_u32(elf, base + 40, "sh_link")? as usize,
                rd_u64(elf, base + 56, "sh_entsize")?,
            ));
        }
        for &(ty, off, size, link, entsize) in &sections {
            if ty != SHT_SYMTAB && ty != SHT_DYNSYM {
                continue;
            }
            if entsize != ELF64_SYM {
                return Err(format!("symbol entry size {entsize} is not {ELF64_SYM}"));
            }
            let &(_, str_off, str_size, _, _) = sections
                .get(link)
                .ok_or_else(|| format!("symbol table links to missing section {link}"))?;
            let strtab = slice_at(elf, str_off, str_size, "string table")?;
            let table = slice_at(elf, off, size, "symbol table")?;
            for entry in table.chunks_exact(ELF64_SYM as usize) {
                let name_off = rd_u32(entry, 0, "st_name")? as usize;
                let tail = strtab
                    .get(name_off..)
                    .ok_or_else(|| format!("st_name {name_off} lies outside the string table"))?;
                let end = tail
                    .iter()
                    .position(|&b| b == 0)
                    .ok_or_else(|| "unterminated symbol name".to_string())?;
                let name = std::str::from_utf8(&tail[..end])
                    .map_err(|_| "symbol name is not UTF-8".to_string())?;
                if let Some(kernel) = name.strip_suffix(".kd") {
                    symbols.insert(kernel.to_string());
                }
            }
        }
    }
    Ok(ElfInfo {
        arch: mach_arch(mach),
        mach,
        symbols: symbols.into_iter().collect(),
    })
}

/// The device ELFs of an embedded object: every `hipv4-amdgcn-amd-amdhsa--<arch>` payload of an
/// uncompressed offload bundle, or the file itself when it is a bare AMDGPU ELF.
fn device_elfs(bytes: &[u8]) -> Result<Vec<DeviceElf>, String> {
    if bytes.starts_with(COMPRESSED_MAGIC) {
        return Err("compressed offload bundle (CCOB) is not supported".into());
    }
    if !bytes.starts_with(MAGIC) {
        let info = elf_info(bytes)?;
        let arch = info
            .arch
            .map(str::to_string)
            .unwrap_or_else(|| format!("mach_{:#04x}", info.mach));
        return Ok(vec![DeviceElf {
            arch,
            bytes: bytes.len() as u64,
            sha256: sha256_hex(bytes),
            symbols: info.symbols,
        }]);
    }
    let mut pos = MAGIC.len();
    let count = rd_u64(bytes, pos, "bundle entry count")?;
    pos += 8;
    if count > (bytes.len().saturating_sub(pos) / 24) as u64 {
        return Err(format!(
            "bundle claims {count} entries but is only {} bytes",
            bytes.len()
        ));
    }
    let mut out = Vec::new();
    for index in 0..count {
        let offset = rd_u64(bytes, pos, "bundle entry offset")?;
        let size = rd_u64(bytes, pos + 8, "bundle entry size")?;
        let triple_len = rd_u64(bytes, pos + 16, "bundle triple length")?;
        pos += 24;
        let triple = slice_at(bytes, pos as u64, triple_len, "bundle triple")?;
        pos += triple.len();
        let triple = std::str::from_utf8(triple)
            .map_err(|_| format!("bundle entry {index}: triple is not UTF-8"))?;
        let Some(target) = triple.strip_prefix(HIP_DEVICE_TRIPLE_PREFIX) else {
            continue;
        };
        if size == 0 {
            continue;
        }
        let arch = target.split(':').next().unwrap_or(target).to_string();
        let elf = slice_at(bytes, offset, size, &format!("bundle entry {triple}"))?;
        let info = elf_info(elf).map_err(|e| format!("bundle entry {triple}: {e}"))?;
        if let Some(flags_arch) = info.arch {
            if flags_arch != arch {
                return Err(format!(
                    "bundle entry {triple}: ELF e_flags say {flags_arch}, bundle target says {arch}"
                ));
            }
        }
        out.push(DeviceElf {
            arch,
            bytes: elf.len() as u64,
            sha256: sha256_hex(elf),
            symbols: info.symbols,
        });
    }
    if out.is_empty() {
        return Err("bundle has no non-empty hipv4-amdgcn-amd-amdhsa entry".into());
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------
// hipcc kernel names: `__global__` declarations in kernels/src/**/*.hip
// ---------------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq)]
enum Ct {
    Ident(String),
    Punct(char),
    Other,
}

/// C++ tokens, dropping comments and preprocessor lines; strings/char/number literals become
/// [`Ct::Other`].
fn lex_cpp(text: &str) -> Vec<Ct> {
    let c: Vec<char> = text.chars().collect();
    let n = c.len();
    let mut i = 0;
    let mut line_start = true;
    let mut out = Vec::new();
    while i < n {
        let ch = c[i];
        if ch == '\n' {
            line_start = true;
            i += 1;
            continue;
        }
        if ch.is_whitespace() {
            i += 1;
            continue;
        }
        if ch == '#' && line_start {
            // preprocessor directive: to end of line, honouring `\` continuations
            while i < n && c[i] != '\n' {
                if c[i] == '\\' && i + 1 < n && c[i + 1] == '\n' {
                    i += 2;
                } else if c[i] == '/' && i + 1 < n && c[i + 1] == '*' {
                    i += 2;
                    while i + 1 < n && !(c[i] == '*' && c[i + 1] == '/') {
                        i += 1;
                    }
                    i = (i + 2).min(n);
                } else {
                    i += 1;
                }
            }
            continue;
        }
        line_start = false;
        if ch == '/' && i + 1 < n && c[i + 1] == '/' {
            while i < n && c[i] != '\n' {
                i += 1;
            }
            continue;
        }
        if ch == '/' && i + 1 < n && c[i + 1] == '*' {
            i += 2;
            while i + 1 < n && !(c[i] == '*' && c[i + 1] == '/') {
                i += 1;
            }
            i = (i + 2).min(n);
            continue;
        }
        if ch == '"' || ch == '\'' {
            let quote = ch;
            i += 1;
            while i < n && c[i] != quote {
                i += if c[i] == '\\' { 2 } else { 1 };
            }
            i += 1;
            out.push(Ct::Other);
            continue;
        }
        if ch.is_alphabetic() || ch == '_' {
            let mut j = i;
            while j < n && (c[j].is_alphanumeric() || c[j] == '_') {
                j += 1;
            }
            // raw string literal R"delim( ... )delim"
            if c[j - 1] == 'R' && j < n && c[j] == '"' {
                let mut k = j + 1;
                let delim_start = k;
                while k < n && c[k] != '(' {
                    k += 1;
                }
                let close: Vec<char> = std::iter::once(')')
                    .chain(c[delim_start..k.min(n)].iter().copied())
                    .chain(std::iter::once('"'))
                    .collect();
                let mut m = k;
                while m + close.len() <= n && c[m..m + close.len()] != close[..] {
                    m += 1;
                }
                i = (m + close.len()).min(n);
                out.push(Ct::Other);
                continue;
            }
            out.push(Ct::Ident(c[i..j].iter().collect()));
            i = j;
            continue;
        }
        if ch.is_ascii_digit() {
            while i < n
                && (c[i].is_alphanumeric()
                    || c[i] == '_'
                    || c[i] == '.'
                    || (c[i] == '\'' && i + 1 < n && c[i + 1].is_alphanumeric()))
            {
                i += 1;
            }
            out.push(Ct::Other);
            continue;
        }
        out.push(Ct::Punct(ch));
        i += 1;
    }
    out
}

/// Attribute-like specifiers that may sit between `__global__` and the kernel name and carry a
/// parenthesised argument list.
const KERNEL_SPECIFIERS: &[&str] = &[
    "__launch_bounds__",
    "__attribute__",
    "__maxnreg__",
    "__declspec",
    "__cluster_dims__",
];

/// Names of the non-template `__global__` kernels declared in a HIP source. Template kernels
/// (their instantiated names are mangled) and macro-token names (`#define`d placeholders, all
/// upper case) are skipped.
fn scan_hip_kernels(text: &str) -> Vec<String> {
    let t = lex_cpp(text);
    let mut out = Vec::new();
    for k in 0..t.len() {
        if !matches!(&t[k], Ct::Ident(s) if s == "__global__") {
            continue;
        }
        let mut stmt_start = k;
        while stmt_start > 0 && !matches!(t[stmt_start - 1], Ct::Punct(';' | '{' | '}')) {
            stmt_start -= 1;
        }
        if t[stmt_start..k]
            .iter()
            .any(|x| matches!(x, Ct::Ident(s) if s == "template"))
        {
            continue;
        }
        let mut m = k + 1;
        let mut prev: Option<&str> = None;
        let mut name: Option<&str> = None;
        while m < t.len() {
            match &t[m] {
                Ct::Ident(s) if KERNEL_SPECIFIERS.contains(&s.as_str()) => {
                    m += 1;
                    if matches!(t.get(m), Some(Ct::Punct('('))) {
                        let mut depth = 0i32;
                        while m < t.len() {
                            match t[m] {
                                Ct::Punct('(') => depth += 1,
                                Ct::Punct(')') => {
                                    depth -= 1;
                                    if depth == 0 {
                                        m += 1;
                                        break;
                                    }
                                }
                                _ => {}
                            }
                            m += 1;
                        }
                    }
                    continue;
                }
                Ct::Punct('(') => {
                    name = prev;
                    break;
                }
                Ct::Punct(';' | '{' | '}') => break,
                Ct::Ident(s) => prev = Some(s.as_str()),
                _ => {}
            }
            m += 1;
        }
        if let Some(name) = name {
            if name.chars().any(|c| c.is_ascii_lowercase()) {
                out.push(name.to_string());
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------------------
// Inventory
// ---------------------------------------------------------------------------------------

/// Builds the inventory of `repo` at `commit` (read via `git cat-file`), or of the working tree
/// when `commit` is `None`.
pub fn build(repo: &Path, commit: Option<&str>) -> Result<Inventory, String> {
    let mut src = Source::open(repo, commit)?;

    let rust_files: Vec<String> = src
        .list(RUST_SRC_DIR)?
        .into_iter()
        .filter(|f| f.ends_with(".rs"))
        .collect();
    if rust_files.is_empty() {
        return Err(format!("no .rs files under {RUST_SRC_DIR}"));
    }
    let mut declared: BTreeMap<String, BTreeSet<(String, usize)>> = BTreeMap::new();
    for file in &rust_files {
        let bytes = src.read(file)?;
        let text = String::from_utf8(bytes).map_err(|_| format!("{file}: not UTF-8"))?;
        for (object, line) in scan_includes(file, &text)? {
            declared
                .entry(object)
                .or_default()
                .insert((file.clone(), line));
        }
    }

    let mut pm_objects = Vec::with_capacity(declared.len());
    for (path, at) in declared {
        let bytes = src.read(&path).map_err(|e| {
            let sites: Vec<String> = at.iter().map(|(f, l)| format!("{f}:{l}")).collect();
            format!("{e} (declared at {})", sites.join(", "))
        })?;
        let elfs = device_elfs(&bytes).map_err(|e| format!("{path}: {e}"))?;
        pm_objects.push(PmObject {
            sha256: sha256_hex(&bytes),
            bytes: bytes.len() as u64,
            declared_at: at.iter().map(|(f, l)| format!("{f}:{l}")).collect(),
            device_elfs: elfs,
            path,
        });
    }

    let mut hip_kernel_names = BTreeSet::new();
    for file in src
        .list(HIP_SRC_DIR)?
        .into_iter()
        .filter(|f| f.ends_with(".hip"))
    {
        let bytes = src.read(&file)?;
        hip_kernel_names.extend(scan_hip_kernels(&String::from_utf8_lossy(&bytes)));
    }

    Ok(Inventory {
        repo: repo.display().to_string(),
        commit: src.commit(),
        pm_objects,
        hip_kernel_names,
    })
}

fn base_arch(arch: &str) -> &str {
    arch.split(':').next().unwrap_or(arch)
}

impl Inventory {
    /// Decides who produced the code that ran `kernel`:
    ///
    /// 1. JSON trace, the kernel's code object `memory_size` equals the byte length of a PM
    ///    device ELF (same arch as the code object's agent, when known) that exports the symbol
    ///    -> `Pm`, join `code_object_size+symbol`.
    /// 2. Some PM object exports the symbol but its size is known and matches no PM device ELF
    ///    -> `Unknown`, `unresolved: PM export ...` (the loaded object is not one this inventory
    ///    describes: a different build, or a hipcc kernel sharing a PM export's name).
    /// 3. No PM object exports it and it is a `__global__` name in `kernels/src` -> `Hipcc`.
    /// 4. JSON trace, no PM object exports it, and its code object's `memory_size` matches no
    ///    PM device ELF at all: the object is not one of the embedded PM images, and hipfire's
    ///    only other code generator is hipcc (JIT modules and kernel packs, including kernels
    ///    whose names are spelled by macros or `-D` substitution and so never appear as a
    ///    `__global__` literal) -> `Hipcc`, join `non_pm_object`. A code object whose URI is a
    ///    ROCm library (`/opt/rocm`, rocBLAS / hipBLASLt / Tensile) stays `Unknown`.
    /// 5. Otherwise `Unknown`.
    ///
    /// Without a code-object size (CSV traces, or a JSON code object lacking `memory_size`)
    /// rule 1/2 are unavailable and the same name-only decision is made with a `name_only` join.
    pub fn attribute(&self, trace: &Trace, kernel: &KernelSymbol) -> Attribution {
        let name = kernel.name.as_str();
        let at = self.commit.as_deref().unwrap_or("the working tree");
        let code_object = kernel.code_object_id.and_then(|id| trace.code_object(id));
        let size = code_object.and_then(|c| c.memory_size);
        let name_only = size.is_none();
        let prefix = if name_only { "name_only: " } else { "" };

        let exporters: Vec<(&PmObject, &DeviceElf)> = self
            .pm_objects
            .iter()
            .flat_map(|o| o.device_elfs.iter().map(move |e| (o, e)))
            .filter(|(_, e)| e.symbols.iter().any(|s| s == name))
            .collect();
        let hip_source = self.hip_kernel_names.contains(name);

        if !exporters.is_empty() {
            let Some(size) = size else {
                let paths = distinct_paths(&exporters);
                let join = format!("name_only: PM export of {}", paths.join(", "));
                return single(Origin::Pm, &exporters, join);
            };
            let agent_arch = code_object
                .and_then(|c| trace.agent(c.agent))
                .map(|a| base_arch(&a.name));
            let matched: Vec<(&PmObject, &DeviceElf)> = exporters
                .iter()
                .copied()
                .filter(|(_, e)| {
                    e.bytes == size && agent_arch.map_or(true, |a| a == base_arch(&e.arch))
                })
                .collect();
            if matched.is_empty() {
                let also = if hip_source {
                    "; also a __global__ name in kernels/src"
                } else {
                    ""
                };
                return Attribution {
                    origin: Origin::Unknown,
                    object_path: None,
                    object_sha256: None,
                    join: format!(
                        "unresolved: PM export {name}, object size {size} matches no inventory \
                         object at {at}{also}"
                    ),
                };
            }
            let paths = distinct_paths(&matched);
            let join = format!(
                "code_object_size+symbol: memory_size {size} == device ELF of {} exporting {name}",
                paths.join(", ")
            );
            return single(Origin::Pm, &matched, join);
        }
        if hip_source {
            return Attribution {
                origin: Origin::Hipcc,
                object_path: None,
                object_sha256: None,
                join: format!("{prefix}hip_source_name, no PM export"),
            };
        }
        if let (Some(size), Some(object)) = (size, code_object) {
            let pm_sized = self
                .pm_objects
                .iter()
                .flat_map(|o| o.device_elfs.iter())
                .any(|e| e.bytes == size);
            let library = name.starts_with("__amd_rocclr_")
                || ["/opt/rocm", "rocblas", "hipblaslt", "tensile"]
                    .iter()
                    .any(|needle| object.uri.to_ascii_lowercase().contains(needle));
            if library {
                return Attribution {
                    origin: Origin::Unknown,
                    object_path: None,
                    object_sha256: None,
                    join: format!(
                        "unresolved: {name} is a HIP runtime / ROCm library kernel ({})",
                        object.uri
                    ),
                };
            }
            if !pm_sized {
                return Attribution {
                    origin: Origin::Hipcc,
                    object_path: None,
                    object_sha256: None,
                    join: format!(
                        "non_pm_object: memory_size {size} matches no PM device ELF at {at} and \
                         no PM object exports {name}"
                    ),
                };
            }
        }
        Attribution {
            origin: Origin::Unknown,
            object_path: None,
            object_sha256: None,
            join: format!(
                "unresolved: {prefix}{name} is not exported by any PM object at {at} and is not a \
                 __global__ name in {HIP_SRC_DIR}"
            ),
        }
    }
}

fn distinct_paths(objs: &[(&PmObject, &DeviceElf)]) -> Vec<String> {
    let set: BTreeSet<&str> = objs.iter().map(|(o, _)| o.path.as_str()).collect();
    set.into_iter().map(str::to_string).collect()
}

/// A `Pm` attribution naming its object only when exactly one object path is involved.
fn single(origin: Origin, objs: &[(&PmObject, &DeviceElf)], join: String) -> Attribution {
    let paths = distinct_paths(objs);
    let (object_path, object_sha256) = match paths.as_slice() {
        [only] => {
            let sha = objs
                .iter()
                .find(|(o, _)| &o.path == only)
                .map(|(o, _)| o.sha256.clone());
            (Some(only.clone()), sha)
        }
        _ => (None, None),
    };
    Attribution {
        origin,
        object_path,
        object_sha256,
        join,
    }
}

const USAGE: &str = "usage: pm-census inventory [--repo DIR] [--commit SHA]\n  \
    Prints the embedded PM object inventory (include_bytes! of kernels/**/*.hxaco|.co under \
crates/rdna-compute/src) and the hipcc __global__ kernel names as JSON.\n  \
    --repo DIR     repository root (default: .)\n  \
    --commit SHA   read files from this commit instead of the working tree";

pub fn cli(args: &[String]) -> Result<(), String> {
    let mut repo = PathBuf::from(".");
    let mut commit: Option<String> = None;
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--repo" => repo = PathBuf::from(it.next().ok_or("--repo needs a directory")?),
            "--commit" => commit = Some(it.next().ok_or("--commit needs a SHA")?.clone()),
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(());
            }
            other => return Err(format!("unknown argument {other:?}\n{USAGE}")),
        }
    }
    let inventory = build(&repo, commit.as_deref())?;
    let json = serde_json::to_string_pretty(&inventory).map_err(|e| e.to_string())?;
    println!("{json}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rocprof::{Agent, CodeObject, TraceSource};

    const ELF64_EHDR: usize = 64;
    const EMPTY: &[u8] = &[];

    // ---- synthetic code objects -------------------------------------------------------

    /// A minimal AMDGPU ELF64: header, string table, symbol table with one `.kd` symbol per
    /// kernel plus a non-descriptor symbol, and three section headers (null, symtab, strtab).
    fn synthetic_elf(mach: u32, kernels: &[&str]) -> Vec<u8> {
        let mut strtab = vec![0u8];
        let mut name_offs = Vec::new();
        for k in kernels {
            name_offs.push(strtab.len() as u32);
            strtab.extend(format!("{k}.kd").bytes());
            strtab.push(0);
        }
        let helper_off = strtab.len() as u32;
        strtab.extend(b"helper\0");
        let mut symtab = vec![0u8; 24];
        for off in name_offs.iter().copied().chain(std::iter::once(helper_off)) {
            let mut sym = [0u8; 24];
            sym[0..4].copy_from_slice(&off.to_le_bytes());
            symtab.extend(sym);
        }
        let strtab_off = ELF64_EHDR as u64;
        let symtab_off = strtab_off + strtab.len() as u64;
        let shoff = symtab_off + symtab.len() as u64;
        let mut e = vec![0u8; ELF64_EHDR];
        e[0..4].copy_from_slice(b"\x7fELF");
        e[4] = 2;
        e[5] = 1;
        e[6] = 1;
        e[18..20].copy_from_slice(&EM_AMDGPU.to_le_bytes());
        e[40..48].copy_from_slice(&shoff.to_le_bytes());
        e[48..52].copy_from_slice(&mach.to_le_bytes());
        e[52..54].copy_from_slice(&(ELF64_EHDR as u16).to_le_bytes());
        e[58..60].copy_from_slice(&(ELF64_SHDR as u16).to_le_bytes());
        e[60..62].copy_from_slice(&3u16.to_le_bytes());
        e.extend(&strtab);
        e.extend(&symtab);
        let header = |ty: u32, off: u64, size: u64, link: u32, entsize: u64| {
            let mut h = [0u8; ELF64_SHDR];
            h[4..8].copy_from_slice(&ty.to_le_bytes());
            h[24..32].copy_from_slice(&off.to_le_bytes());
            h[32..40].copy_from_slice(&size.to_le_bytes());
            h[40..44].copy_from_slice(&link.to_le_bytes());
            h[56..64].copy_from_slice(&entsize.to_le_bytes());
            h
        };
        e.extend([0u8; ELF64_SHDR]);
        e.extend(header(
            SHT_SYMTAB,
            symtab_off,
            symtab.len() as u64,
            2,
            ELF64_SYM,
        ));
        e.extend(header(3, strtab_off, strtab.len() as u64, 0, 0));
        e
    }

    fn synthetic_bundle(entries: &[(&str, &[u8])]) -> Vec<u8> {
        let mut out = MAGIC.to_vec();
        out.extend((entries.len() as u64).to_le_bytes());
        let header_len: usize = 32 + entries.iter().map(|(t, _)| 24 + t.len()).sum::<usize>();
        let mut offset = header_len as u64;
        let mut payloads = Vec::new();
        for (triple, payload) in entries {
            out.extend(offset.to_le_bytes());
            out.extend((payload.len() as u64).to_le_bytes());
            out.extend((triple.len() as u64).to_le_bytes());
            out.extend(triple.bytes());
            payloads.extend_from_slice(payload);
            offset += payload.len() as u64;
        }
        out.extend(payloads);
        out
    }

    #[test]
    fn elf_symbols_arch_and_bundle_are_read() {
        let elf = synthetic_elf(0x4e, &["zeta_pm_gfx1201", "alpha_pm_gfx1201"]);
        let info = elf_info(&elf).unwrap();
        assert_eq!(info.arch, Some("gfx1201"));
        assert_eq!(info.symbols, ["alpha_pm_gfx1201", "zeta_pm_gfx1201"]);

        // bare ELF
        let devs = device_elfs(&elf).unwrap();
        assert_eq!(devs.len(), 1);
        assert_eq!(devs[0].arch, "gfx1201");
        assert_eq!(devs[0].bytes, elf.len() as u64);
        assert_eq!(devs[0].sha256, sha256_hex(&elf));

        // bundle: host entry is empty, device entry is the ELF
        let bundle = synthetic_bundle(&[
            ("host-x86_64-unknown-linux-gnu-", EMPTY),
            ("hipv4-amdgcn-amd-amdhsa--gfx1201", elf.as_slice()),
        ]);
        let devs = device_elfs(&bundle).unwrap();
        assert_eq!(devs.len(), 1);
        assert_eq!(devs[0].arch, "gfx1201");
        assert_eq!(devs[0].bytes, elf.len() as u64);
        assert!(devs[0].bytes < bundle.len() as u64);
        assert_eq!(devs[0].symbols, ["alpha_pm_gfx1201", "zeta_pm_gfx1201"]);

        // target features after ':' are not part of the arch
        let featured =
            synthetic_bundle(&[("hipv4-amdgcn-amd-amdhsa--gfx1201:xnack-", elf.as_slice())]);
        assert_eq!(device_elfs(&featured).unwrap()[0].arch, "gfx1201");
    }

    #[test]
    fn bundle_arch_must_agree_with_elf_flags() {
        let elf = synthetic_elf(0x4e, &["k"]);
        let bundle = synthetic_bundle(&[("hipv4-amdgcn-amd-amdhsa--gfx1100", elf.as_slice())]);
        let err = device_elfs(&bundle).unwrap_err();
        assert!(err.contains("gfx1201") && err.contains("gfx1100"), "{err}");
    }

    #[test]
    fn malformed_objects_are_errors() {
        assert!(device_elfs(b"not an object").is_err());
        let mut compressed = COMPRESSED_MAGIC.to_vec();
        compressed.extend([0u8; 16]);
        assert!(device_elfs(&compressed).unwrap_err().contains("CCOB"));
        let host_only = synthetic_bundle(&[("host-x86_64-unknown-linux-gnu-", EMPTY)]);
        assert!(device_elfs(&host_only)
            .unwrap_err()
            .contains("no non-empty"));
        let mut truncated = synthetic_elf(0x4e, &["k"]);
        truncated.truncate(ELF64_EHDR + 4);
        assert!(elf_info(&truncated).is_err());
        let mut wrong_machine = synthetic_elf(0x4e, &["k"]);
        wrong_machine[18..20].copy_from_slice(&62u16.to_le_bytes());
        assert!(elf_info(&wrong_machine).unwrap_err().contains("EM_AMDGPU"));
    }

    // ---- include_bytes! scanning ------------------------------------------------------

    const RS: &str = r####"
// include_bytes!("../../../kernels/commented_pm_gfx1201.hxaco")
/* include_bytes!("../../../kernels/block_commented.hxaco") */
const NOTE: &str = "include_bytes!(\"../../../kernels/in_string.hxaco\")";
const RAW: &str = r#"include_bytes!("../../../kernels/in_raw.hxaco")"#;
const QUOTE: char = '"';
fn lifetimes<'a>(x: &'a str) -> &'a str { x }

macro_rules! twin {
    ($module:literal, $file:literal, $symbols:expr, source = $src:literal) => {
        Twin {
            image: include_bytes!(concat!(
                "../../../kernels/pm-decode/gfx1201/",
                $file,
                ".hxaco"
            )),
        }
    };
}
static TWINS: &[Twin] = &[
    twin!("m1", "first_mq4v2", &["a", "b"], source = "x"),
    twin!(
        "m2",
        "second_mq4v2",
        &["c"],
        source = "y"
    ),
];
const A: &[u8] = include_bytes!("../../../kernels/a_pm_gfx1201.hxaco");
const B: &[u8] = include_bytes!(
    "../../../kernels/pm-decode/gfx1201/b_xbatch.co"
);
const SAME: &[u8] = include_bytes!("../../../kernels/a_pm_gfx1201.hxaco");
const HSACO: &[u8] = include_bytes!("../../../kernels/not_pm.hsaco");
const FIXTURE: &[u8] = include_bytes!("../tests/fixtures/pack/tensor_ops.co");
const BIN: &[u8] = include_bytes!("../../../kernels/other.bin");
"####;

    #[test]
    fn scan_includes_resolves_literals_multiline_and_macro_expansions() {
        let found = scan_includes("crates/rdna-compute/src/kernels.rs", RS).unwrap();
        let mut paths: Vec<&str> = found.iter().map(|(p, _)| p.as_str()).collect();
        paths.sort();
        assert_eq!(
            paths,
            [
                "kernels/a_pm_gfx1201.hxaco",
                "kernels/a_pm_gfx1201.hxaco",
                "kernels/pm-decode/gfx1201/b_xbatch.co",
                "kernels/pm-decode/gfx1201/first_mq4v2.hxaco",
                "kernels/pm-decode/gfx1201/second_mq4v2.hxaco",
            ]
        );
        // macro-expanded includes are reported at the invocation line
        let line_of = |p: &str| found.iter().find(|(q, _)| q == p).unwrap().1;
        let first = RS.lines().position(|l| l.contains("twin!(\"m1\"")).unwrap() + 1;
        assert_eq!(
            line_of("kernels/pm-decode/gfx1201/first_mq4v2.hxaco"),
            first
        );
        assert_eq!(
            line_of("kernels/pm-decode/gfx1201/second_mq4v2.hxaco"),
            first + 1
        );
    }

    #[test]
    fn scan_includes_rejects_unresolvable_expressions() {
        let file = "crates/rdna-compute/src/x.rs";
        let env = r#"const X: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/k.hxaco"));"#;
        assert!(scan_includes(file, env)
            .unwrap_err()
            .contains("must be a string literal"));
        let ident = "const X: &[u8] = include_bytes!(PATH);";
        assert!(scan_includes(file, ident).is_err());
        let orphan = r#"const X: &[u8] = include_bytes!(concat!("../kernels/", $f, ".hxaco"));"#;
        assert!(scan_includes(file, orphan)
            .unwrap_err()
            .contains("outside a macro_rules"));
        let escapes = r#"const X: &[u8] = include_bytes!("../../../../../kernels/k.hxaco");"#;
        assert!(scan_includes(file, escapes)
            .unwrap_err()
            .contains("leaves the repo"));
    }

    #[test]
    fn resolve_path_normalises_relative_includes() {
        assert_eq!(
            resolve_path(
                "crates/rdna-compute/src/kernels.rs",
                "../../../kernels/k.hxaco"
            )
            .unwrap(),
            "kernels/k.hxaco"
        );
        assert_eq!(
            resolve_path(
                "crates/rdna-compute/src/rdna/a.rs",
                "../../../../kernels/./k.co"
            )
            .unwrap(),
            "kernels/k.co"
        );
    }

    // ---- hip kernel scanning ----------------------------------------------------------

    #[test]
    fn scan_hip_kernels_handles_forms_and_skips_templates_comments_macros() {
        let src = r#"
#define K_MACRO __global__ void from_define(
// __global__ void commented_out(int x)
/* __global__ void block_commented(int x) */
const char* s = "__global__ void in_string(int x)";
extern "C" __global__ void plain(const float* x, float* y) { }
extern "C" __global__ void __launch_bounds__(256) bounded(int n) { }
extern "C" __global__ void __launch_bounds__(256, 2) __attribute__((amdgpu_waves_per_eu(1, 2))) both(int n) { }
__global__ __launch_bounds__(128) void leading_bounds(int n) { }
template <typename T>
__global__ void templated(T* x) { }
template <>
__global__ void specialised<float>(float* x) { }
extern "C" __global__ void GDN_CHUNK_SCAN_KERNEL(int n) { }
extern "C"
__global__ void
split_across_lines(
    int n) { }
__device__ void helper(int x) { }
extern "C" __global__ void last_one(int x);
"#;
        assert_eq!(
            scan_hip_kernels(src),
            [
                "plain",
                "bounded",
                "both",
                "leading_bounds",
                "split_across_lines",
                "last_one"
            ]
        );
    }

    // ---- attribution ------------------------------------------------------------------

    const GOOD: &str = "kernels/a_pm_gfx1201.hxaco";

    fn inventory() -> Inventory {
        let elf = DeviceElf {
            arch: "gfx1201".into(),
            bytes: 4000,
            sha256: "e".repeat(64),
            symbols: vec!["k_pm_gfx1201".into(), "shared_name".into()],
        };
        let other = DeviceElf {
            arch: "gfx1100".into(),
            bytes: 3000,
            sha256: "f".repeat(64),
            symbols: vec!["k_pm_gfx1100".into(), "shared_name".into()],
        };
        Inventory {
            repo: ".".into(),
            commit: Some("c0ffee".into()),
            pm_objects: vec![
                PmObject {
                    path: GOOD.into(),
                    sha256: "a".repeat(64),
                    bytes: 5000,
                    declared_at: vec!["crates/rdna-compute/src/kernels.rs:1".into()],
                    device_elfs: vec![elf],
                },
                PmObject {
                    path: "kernels/b_pm_gfx1100.hxaco".into(),
                    sha256: "b".repeat(64),
                    bytes: 3500,
                    declared_at: vec!["crates/rdna-compute/src/kernels.rs:2".into()],
                    device_elfs: vec![other],
                },
            ],
            hip_kernel_names: ["hip_only".to_string(), "shared_name".to_string()]
                .into_iter()
                .collect(),
        }
    }

    fn symbol(id: u64, name: &str, code_object: Option<u64>) -> KernelSymbol {
        KernelSymbol {
            kernel_id: id,
            code_object_id: code_object,
            name: name.into(),
            kernarg_segment_size: None,
            group_segment_size: None,
            private_segment_size: None,
            sgpr_count: None,
            vgpr_count: None,
        }
    }

    fn trace(format: &str, kernels: Vec<KernelSymbol>) -> Trace {
        let co = |id: u64, agent: u64, memory_size: Option<u64>| CodeObject {
            id,
            agent,
            uri: String::new(),
            memory_size,
            load_size: 65536,
        };
        Trace {
            source: TraceSource {
                path: "t".into(),
                sha256: "0".repeat(64),
                format: format.into(),
            },
            agents: vec![
                Agent {
                    id: 200,
                    kind: "gpu".into(),
                    name: "gfx1201".into(),
                    ordinal: Some(0),
                },
                Agent {
                    id: 201,
                    kind: "gpu".into(),
                    name: "gfx1100".into(),
                    ordinal: Some(1),
                },
            ],
            code_objects: vec![
                co(10, 200, Some(4000)), // == device ELF size
                co(11, 200, Some(5000)), // == bundle file size, not the ELF
                co(12, 201, Some(4000)), // right size, wrong agent arch
                co(13, 200, None),       // no memory_size
                co(14, 200, Some(7)),    // unrelated object
            ],
            kernels,
            dispatches: Vec::new(),
            copies: Vec::new(),
            markers: Vec::new(),
        }
    }

    #[test]
    fn rule1_pm_by_device_elf_size_and_symbol() {
        let inv = inventory();
        let k = symbol(1, "k_pm_gfx1201", Some(10));
        let t = trace("rocprofv3-json", vec![k.clone()]);
        let a = inv.attribute(&t, &k);
        assert_eq!(a.origin, Origin::Pm);
        assert_eq!(a.object_path.as_deref(), Some(GOOD));
        assert_eq!(a.object_sha256, Some("a".repeat(64)));
        assert!(a.join.starts_with("code_object_size+symbol"), "{}", a.join);
        assert!(
            a.join.contains("4000") && a.join.contains(GOOD),
            "{}",
            a.join
        );
    }

    #[test]
    fn rule2_pm_export_with_unmatched_size_is_unresolved() {
        let inv = inventory();
        // memory_size equals the *bundle file* length: HIP reports the device ELF, so no match
        let bundle_sized = symbol(1, "k_pm_gfx1201", Some(11));
        // same size as the ELF but loaded on a gfx1100 agent
        let wrong_arch = symbol(2, "k_pm_gfx1201", Some(12));
        let unrelated = symbol(3, "k_pm_gfx1201", Some(14));
        let t = trace(
            "rocprofv3-json",
            vec![bundle_sized.clone(), wrong_arch.clone(), unrelated.clone()],
        );
        for k in [&bundle_sized, &wrong_arch, &unrelated] {
            let a = inv.attribute(&t, k);
            assert_eq!(a.origin, Origin::Unknown, "{}", k.kernel_id);
            assert_eq!(a.object_path, None);
            assert!(
                a.join.starts_with("unresolved: PM export k_pm_gfx1201"),
                "{}",
                a.join
            );
            assert!(
                a.join.contains("matches no inventory object at c0ffee"),
                "{}",
                a.join
            );
        }
        assert!(inv
            .attribute(&t, &unrelated)
            .join
            .contains("object size 7 "));
    }

    #[test]
    fn rule2_notes_a_colliding_hip_source_name() {
        let inv = inventory();
        let k = symbol(1, "shared_name", Some(14));
        let t = trace("rocprofv3-json", vec![k.clone()]);
        let a = inv.attribute(&t, &k);
        assert_eq!(a.origin, Origin::Unknown);
        assert!(a.join.contains("also a __global__ name"), "{}", a.join);
    }

    #[test]
    fn rule3_hipcc_and_rule4_unknown() {
        let inv = inventory();
        let hip = symbol(1, "hip_only", Some(14));
        let mystery = symbol(2, "__amd_rocclr_copyBuffer", Some(14));
        let t = trace("rocprofv3-json", vec![hip.clone(), mystery.clone()]);
        let a = inv.attribute(&t, &hip);
        assert_eq!(a.origin, Origin::Hipcc);
        assert_eq!(a.join, "hip_source_name, no PM export");
        assert_eq!((a.object_path, a.object_sha256), (None, None));
        let u = inv.attribute(&t, &mystery);
        assert_eq!(u.origin, Origin::Unknown);
        assert!(u.join.starts_with("unresolved:"), "{}", u.join);
        assert!(u.join.contains("__amd_rocclr_copyBuffer"), "{}", u.join);
    }

    #[test]
    fn rule4_non_pm_object_is_hipcc_and_libraries_stay_unknown() {
        let inv = inventory();
        // A macro-spelled hipcc kernel: no PM export, no `__global__` literal, object size 7
        // matches no PM device ELF.
        let macro_named = symbol(1, "qwen4_wmma_lds_128_128_32_64_k64", Some(14));
        // Same name in an object whose size equals a PM device ELF: not provably non-PM.
        let pm_sized = symbol(2, "qwen4_wmma_lds_128_128_32_64_k64", Some(10));
        let mut t = trace(
            "rocprofv3-json",
            vec![macro_named.clone(), pm_sized.clone()],
        );
        let a = inv.attribute(&t, &macro_named);
        assert_eq!(a.origin, Origin::Hipcc);
        assert!(
            a.join.starts_with("non_pm_object: memory_size 7 "),
            "{}",
            a.join
        );
        let b = inv.attribute(&t, &pm_sized);
        assert_eq!(b.origin, Origin::Unknown, "{}", b.join);
        // ROCm library objects never count as hipcc.
        t.code_objects[4].uri = "file:///opt/rocm/lib/rocblas/library/TensileLibrary.co".into();
        let lib = inv.attribute(&t, &macro_named);
        assert_eq!(lib.origin, Origin::Unknown);
        assert!(lib.join.contains("ROCm library"), "{}", lib.join);
    }

    #[test]
    fn csv_and_sizeless_traces_use_name_only_rules() {
        let inv = inventory();
        let pm = symbol(1, "k_pm_gfx1201", None);
        let hip = symbol(2, "hip_only", None);
        let unknown = symbol(3, "mystery", None);
        let no_size = symbol(4, "k_pm_gfx1201", Some(13));
        let t = trace(
            "rocprofv3-csv",
            vec![pm.clone(), hip.clone(), unknown.clone(), no_size.clone()],
        );
        let a = inv.attribute(&t, &pm);
        assert_eq!(a.origin, Origin::Pm);
        assert_eq!(a.object_path.as_deref(), Some(GOOD));
        assert!(a.join.starts_with("name_only"), "{}", a.join);
        let h = inv.attribute(&t, &hip);
        assert_eq!(h.origin, Origin::Hipcc);
        assert!(h.join.starts_with("name_only"), "{}", h.join);
        let u = inv.attribute(&t, &unknown);
        assert_eq!(u.origin, Origin::Unknown);
        assert!(u.join.contains("name_only"), "{}", u.join);
        assert_eq!(inv.attribute(&t, &no_size).origin, Origin::Pm);

        // a name exported by two objects is Pm without naming one of them
        let both = symbol(5, "shared_name", None);
        let a = inv.attribute(&t, &both);
        assert_eq!(a.origin, Origin::Pm);
        assert_eq!(a.object_path, None);
        assert!(
            a.join.contains(GOOD) && a.join.contains("kernels/b_pm_gfx1100.hxaco"),
            "{}",
            a.join
        );
    }

    // ---- the real repo ----------------------------------------------------------------

    fn repo_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    fn object<'a>(inv: &'a Inventory, path: &str) -> &'a PmObject {
        inv.pm_objects
            .iter()
            .find(|o| o.path == path)
            .unwrap_or_else(|| panic!("{path} is not in the inventory"))
    }

    fn exports(obj: &PmObject, arch: &str, symbol: &str) -> bool {
        obj.device_elfs
            .iter()
            .any(|e| e.arch == arch && e.symbols.iter().any(|s| s == symbol))
    }

    #[test]
    fn working_tree_inventory_finds_mq6_and_qwen4_moe_sym_bundles() {
        let inv = build(&repo_root(), None).unwrap();
        assert_eq!(inv.commit, None);

        let mq6 = object(&inv, "kernels/gemm_mq6g256v2_mmq_i8_pm_gfx1201.hxaco");
        assert!(exports(
            mq6,
            "gfx1201",
            "gemm_mq6g256v2_mmq_i8_pm_set_gfx1201"
        ));
        assert!(!mq6.declared_at.is_empty());
        assert!(mq6
            .declared_at
            .iter()
            .all(|d| d.starts_with("crates/rdna-compute/src/") && d.contains(".rs:")));
        let elf = &mq6.device_elfs[0];
        assert!(
            elf.bytes < mq6.bytes,
            "device ELF is smaller than its bundle"
        );
        assert_eq!(mq6.sha256.len(), 64);

        let moe = object(&inv, "kernels/qwen4_moe_iu4_sym_pm_gfx1201.hxaco");
        assert!(exports(
            moe,
            "gfx1201",
            "qwen4_moe_down_iu4_sym_pm_gfx1201_nt8"
        ));
        assert!(exports(
            moe,
            "gfx1201",
            "qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1201_nt8"
        ));
        let gp = object(&inv, "kernels/qwen4_moe_down_gp_pm_gfx1201.hxaco");
        assert!(exports(
            gp,
            "gfx1201",
            "qwen4_moe_down_iu4_sym_gp_pm_gfx1201_nt8"
        ));
        assert!(exports(gp, "gfx1201", "moe_rotate_i4_gp_pm_gfx1201"));

        // the pm-decode twins are declared through a macro and must be found
        assert!(inv
            .pm_objects
            .iter()
            .any(|o| o.path.starts_with("kernels/pm-decode/gfx1201/")));
        // every object parsed down to at least one device ELF with symbols
        assert!(inv.pm_objects.iter().all(|o| !o.device_elfs.is_empty()));
        assert!(!inv.hip_kernel_names.is_empty());
        assert!(inv.hip_kernel_names.contains("argmax_f32"));
        // PM exports and hip source names are different populations
        assert!(!inv
            .hip_kernel_names
            .contains("gemm_mq6g256v2_mmq_i8_pm_set_gfx1201"));
    }

    /// The r9700 prefill capture was taken with a binary built from `5fde8959db`: the join must
    /// resolve its PM code objects by device-ELF size, and nothing PM may come out unresolved.
    #[test]
    fn real_r9700_trace_attributes_pm_objects_by_device_elf_size() {
        const COMMIT: &str = "5fde8959db";
        let trace_path = Path::new(
            "/home/kaden/qcal/release-0.4.3/fn-prefill-profile-002/r9700/prof/trace_results.json",
        );
        let repo = repo_root();
        let have_commit = Command::new("git")
            .arg("-C")
            .arg(&repo)
            .args(["cat-file", "-e", &format!("{COMMIT}^{{commit}}")])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        if !trace_path.exists() || !have_commit {
            return;
        }
        let trace = crate::rocprof::load(trace_path).unwrap();
        let inv = build(&repo, Some(COMMIT)).unwrap();
        assert!(inv.commit.as_deref().is_some_and(|c| c.starts_with(COMMIT)));

        let by_name = |name: &str| trace.kernels.iter().find(|k| k.name == name).unwrap();
        for (kernel, object_path) in [
            (
                "gemm_mq6g256v2_mmq_i8_pm_set_gfx1201",
                "kernels/gemm_mq6g256v2_mmq_i8_pm_gfx1201.hxaco",
            ),
            (
                "qwen4_moe_down_iu4_sym_pm_gfx1201_nt8",
                "kernels/qwen4_moe_iu4_sym_pm_gfx1201.hxaco",
            ),
            (
                "qwen4_moe_down_iu4_sym_gp_pm_gfx1201_nt8",
                "kernels/qwen4_moe_down_gp_pm_gfx1201.hxaco",
            ),
        ] {
            let a = inv.attribute(&trace, by_name(kernel));
            assert_eq!(a.origin, Origin::Pm, "{kernel}: {}", a.join);
            assert_eq!(a.object_path.as_deref(), Some(object_path), "{kernel}");
            assert!(a.join.starts_with("code_object_size+symbol"), "{}", a.join);
        }

        // Every kernel some PM object exports is matched by size (none unresolved).
        for k in &trace.kernels {
            let exported = inv
                .pm_objects
                .iter()
                .flat_map(|o| &o.device_elfs)
                .any(|e| e.symbols.iter().any(|s| *s == k.name));
            if exported {
                let a = inv.attribute(&trace, k);
                assert_eq!(a.origin, Origin::Pm, "{}: {}", k.name, a.join);
            }
        }
    }
}
