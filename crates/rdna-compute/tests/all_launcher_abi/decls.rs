//! Declared ABI per kernel symbol: HIP source declarations (as the JIT sees
//! them, including Rust-side `#define` prefixes) and PeaceMaker image metadata.

use crate::hip::{decls_of, Arg, Decl};
use crate::lex::*;
use crate::scan::*;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use peacemaker_ir::{envelope::Source, inst::Arch};
use peacemaker_lift::bundle::SourceCodec;

pub type DeclList = Vec<(String, Decl)>;

#[derive(Default)]
pub struct Symbols {
    pub hip: BTreeMap<String, DeclList>,
    pub pm: BTreeMap<String, DeclList>,
    /// Constants whose text is (derived from) a changed HIP source.
    pub changed_consts: BTreeSet<(String, String)>,
    /// Changed constants whose text could not be evaluated (must stay empty).
    pub unevaluable: Vec<(String, String)>,
    /// Origin string -> symbols it declares (`const:NAME@file`, `file:path`, `image:path`).
    pub origin_symbols: BTreeMap<String, Vec<String>>,
    /// PM image -> (kernarg_segment_size per symbol).
    pub pm_segment: BTreeMap<(String, String), u32>,
}

impl Symbols {
    pub fn all_names(&self) -> BTreeSet<String> {
        self.hip.keys().chain(self.pm.keys()).cloned().collect()
    }

    pub fn decls(&self, name: &str) -> DeclList {
        let mut v = self.hip.get(name).cloned().unwrap_or_default();
        v.extend(self.pm.get(name).cloned().unwrap_or_default());
        v
    }

    pub fn is_changed_const(&self, name: &str, file: &str) -> bool {
        self.changed_consts
            .contains(&(name.to_string(), file.to_string()))
    }
}

pub fn const_origin(name: &str, file: &str) -> String {
    format!("const:{name}@{file}")
}

/// Index every changed HIP source: constants derived from them (through
/// `include_str!`/`concat!`/const references) and the raw files.
pub fn build_hip(
    world: &World,
    root: &Path,
    changed_files: &[String],
    old_consts: &BTreeMap<String, String>,
    tables: &BTreeMap<String, String>,
    syms: &mut Symbols,
) -> Result<(), String> {
    let base: BTreeSet<String> = changed_files
        .iter()
        .map(|f| {
            Path::new(f)
                .file_name()
                .unwrap()
                .to_string_lossy()
                .to_string()
        })
        .collect();
    // Direct: include_str! of a changed file.
    for f in &world.files {
        for (name, defs) in &f.consts {
            for d in defs {
                let mut lits = Vec::new();
                strings_in(&d.tokens, &mut lits);
                if lits.iter().any(|s| {
                    s.ends_with(".hip") && base.contains(s.rsplit('/').next().unwrap_or(""))
                }) {
                    syms.changed_consts.insert((name.clone(), f.path.clone()));
                }
            }
        }
    }
    // Historical OLD_* source file names are not current source: classify per const.
    for key in old_consts.keys() {
        let (file, name) = split_key(key)?;
        if !syms.is_changed_const(name, file) {
            return Err(format!(
                "old_consts {key}: not a const naming a changed .hip file (stale entry)"
            ));
        }
        verify_old_const(world, name, file)?;
        syms.changed_consts
            .remove(&(name.to_string(), file.to_string()));
    }
    // Transitive: const expressions over changed consts.
    loop {
        let mut added = false;
        for f in &world.files {
            for (name, defs) in &f.consts {
                if defs.len() != 1 || syms.is_changed_const(name, &f.path) {
                    continue;
                }
                let mut ids = Vec::new();
                idents_in(&defs[0].init(), &mut ids);
                if ids.iter().any(|i| {
                    i != name
                        && (syms.is_changed_const(i, &f.path)
                            || syms.is_changed_const(i, KERNELS_RS))
                }) {
                    syms.changed_consts.insert((name.clone(), f.path.clone()));
                    added = true;
                }
            }
        }
        if !added {
            break;
        }
    }
    let consts: Vec<(String, String)> = syms.changed_consts.iter().cloned().collect();
    let mut used_tables: BTreeSet<String> = BTreeSet::new();
    for (name, file) in consts {
        match world.const_text(&name, &file) {
            None => {
                let key = format!("{file}::{name}");
                if tables.contains_key(&key) {
                    verify_table_const(world, syms, &name, &file)?;
                    used_tables.insert(key);
                } else {
                    syms.unevaluable.push((name, file));
                }
            }
            Some(text) => {
                if !text.contains("__global__") {
                    continue;
                }
                let found = decls_of(&text).map_err(|e| format!("const {name} ({file}): {e}"))?;
                let origin = const_origin(&name, &file);
                for (sym, d) in found {
                    syms.origin_symbols
                        .entry(origin.clone())
                        .or_default()
                        .push(sym.clone());
                    syms.hip.entry(sym).or_default().push((origin.clone(), d));
                }
            }
        }
    }
    for key in tables.keys() {
        if !used_tables.contains(key) {
            return Err(format!(
                "source_tables {key}: not an unevaluable changed const (stale entry)"
            ));
        }
    }
    for f in changed_files {
        let text = std::fs::read_to_string(root.join(f)).map_err(|e| format!("{f}: {e}"))?;
        let found = decls_of(&text).map_err(|e| format!("{f}: {e}"))?;
        let origin = format!("file:{f}");
        for (sym, d) in found {
            syms.origin_symbols
                .entry(origin.clone())
                .or_default()
                .push(sym.clone());
            syms.hip.entry(sym).or_default().push((origin.clone(), d));
        }
    }
    Ok(())
}

fn split_key(key: &str) -> Result<(&str, &str), String> {
    key.rsplit_once("::")
        .ok_or_else(|| format!("inventory key `{key}` is not `file::NAME`"))
}

/// An `OLD_*` constant names a historical kernel source FILE for the baseline
/// loader; it is not current JIT source. Verified, not assumed: a plain
/// `OldFile { file: "<name>.hip", syms: .. }` literal (no include_str!/concat!),
/// the loader renames every symbol with the `__beta` suffix before compiling,
/// and every use in the file is either `load_old(&NAME)`, `&NAME` in a list, or
/// the `NAME.syms` field.
fn verify_old_const(world: &World, name: &str, file: &str) -> Result<(), String> {
    let key = format!("{file}::{name}");
    let f = world
        .file(file)
        .ok_or_else(|| format!("old_consts {key}: file not scanned"))?;
    if !name.starts_with("OLD_") {
        return Err(format!("old_consts {key}: not an OLD_* constant"));
    }
    let defs = f
        .consts
        .get(name)
        .filter(|d| d.len() == 1)
        .ok_or_else(|| format!("old_consts {key}: not a unique const"))?;
    let init = defs[0].init();
    let mut ids = Vec::new();
    idents_in(&init, &mut ids);
    if ids.iter().any(|i| {
        matches!(
            i.as_str(),
            "include_str" | "concat" | "format" | "env" | "stringify"
        )
    }) {
        return Err(format!("old_consts {key}: builds source text (include_str!/concat!/format!); it must be a plain file-name literal"));
    }
    let mut lits = Vec::new();
    strings_in(&init, &mut lits);
    let hips: Vec<&String> = lits.iter().filter(|s| s.ends_with(".hip")).collect();
    if hips.len() != 1 || hips[0].contains('/') {
        return Err(format!(
            "old_consts {key}: expected exactly one bare `.hip` file-name literal, found {hips:?}"
        ));
    }
    // Loader contract.
    let suffix = f
        .consts
        .get("OLD_SUFFIX")
        .filter(|d| d.len() == 1)
        .ok_or_else(|| format!("old_consts {key}: no OLD_SUFFIX const"))?;
    let mut sl = Vec::new();
    strings_in(&suffix[0].init(), &mut sl);
    if sl != ["__beta".to_string()] {
        return Err(format!(
            "old_consts {key}: OLD_SUFFIX is {sl:?}, expected \"__beta\""
        ));
    }
    let fn_ids = |fname: &str| -> Option<(Vec<String>, Vec<String>)> {
        let func = f.fns.iter().find(|x| x.name == fname)?;
        let mut i = Vec::new();
        idents_in(&func.body, &mut i);
        let mut s = Vec::new();
        strings_in(&func.body, &mut s);
        Some((i, s))
    };
    let (_, prelude_strs) =
        fn_ids("old_prelude").ok_or_else(|| format!("old_consts {key}: no old_prelude fn"))?;
    if !prelude_strs
        .iter()
        .any(|s| s.starts_with("#define {s} {s}{OLD_SUFFIX}"))
    {
        return Err(format!(
            "old_consts {key}: old_prelude no longer renames every symbol with OLD_SUFFIX"
        ));
    }
    let load = f
        .fns
        .iter()
        .find(|x| x.name == "load_old")
        .ok_or_else(|| format!("old_consts {key}: no load_old fn"))?;
    let mut load_ids = Vec::new();
    idents_in(&load.body, &mut load_ids);
    for need in ["old_prelude", "old_source"] {
        if !load_ids.iter().any(|i| i == need) {
            return Err(format!("old_consts {key}: load_old no longer uses {need}"));
        }
    }
    // The kernels it compiles must be NAMED with the suffix, from the prelude'd source.
    let idx = BindIndex::build(&load.body);
    let load_params = fn_params(load);
    let mut compiles: Vec<Vec<Vec<Tt>>> = Vec::new();
    calls_of(&load.body, "ensure_kernel_public", &mut compiles);
    let flows = !compiles.is_empty()
        && compiles.iter().all(|a| {
            a.len() == 3
                && name_ends_with_const(&a[2], "OLD_SUFFIX", &idx, &load_params, 0)
                && expr_uses_fn(&a[1], "old_prelude", &idx, 0)
        });
    if !flows {
        return Err(format!(
            "old_consts {key}: load_old no longer compiles its kernels under `<sym>{{OLD_SUFFIX}}` from old_prelude'd source"
        ));
    }
    let mut uses = 0usize;
    for func in &f.fns {
        check_old_uses(&func.body, name, None, '{', &mut uses)
            .map_err(|e| format!("old_consts {key}: {} in `{}`", e, func.name))?;
    }
    if uses == 0 {
        return Err(format!("old_consts {key}: never used (stale entry)"));
    }
    Ok(())
}

fn check_old_uses(
    ts: &[Tt],
    name: &str,
    call: Option<&str>,
    open: char,
    n: &mut usize,
) -> Result<(), String> {
    for k in 0..ts.len() {
        match &ts[k] {
            Tt::Id(s, _) if s == name => {
                *n += 1;
                let by_ref = k > 0 && ts[k - 1].is_p('&');
                let syms_field = ts.get(k + 1).map_or(false, |t| t.is_p('.'))
                    && ts.get(k + 2).map_or(false, |t| t.is_id("syms"));
                if !((by_ref && (call == Some("load_old") || open == '[')) || syms_field) {
                    return Err(format!("use of {name} that is neither load_old(&{name}), a `&{name}` list entry nor {name}.syms"));
                }
            }
            Tt::G(d, c, _) => {
                let (callee, o) = if *d == '(' {
                    (if k > 0 { ts[k - 1].id() } else { None }, '(')
                } else if *d == '{' {
                    (call, open)
                } else {
                    (None, *d)
                };
                check_old_uses(c, name, callee, o, n)?;
            }
            _ => {}
        }
    }
    Ok(())
}

/// A const table (e.g. `VARIANTS`) is not source text: it carries kernel names
/// and references the source consts. Verified: no source-building macro, and
/// every changed source const it references (directly or through same-file
/// const fns it calls) is itself text-evaluable and therefore indexed.
fn verify_table_const(world: &World, syms: &Symbols, name: &str, file: &str) -> Result<(), String> {
    let key = format!("{file}::{name}");
    let f = world
        .file(file)
        .ok_or_else(|| format!("source_tables {key}: file not scanned"))?;
    let defs = f
        .consts
        .get(name)
        .filter(|d| d.len() == 1)
        .ok_or_else(|| format!("source_tables {key}: not a unique const"))?;
    let mut ids = Vec::new();
    idents_in(&defs[0].init(), &mut ids);
    let mut all: Vec<String> = ids.clone();
    for id in &ids {
        for func in f.fns.iter().filter(|x| &x.name == id) {
            idents_in(&func.body, &mut all);
        }
    }
    if all
        .iter()
        .any(|i| matches!(i.as_str(), "include_str" | "concat" | "format" | "env"))
    {
        return Err(format!(
            "source_tables {key}: builds source text itself; it must only reference source consts"
        ));
    }
    let mut referenced = 0usize;
    for id in &all {
        if id != name && syms.is_changed_const(id, file) {
            if world.const_text(id, file).is_none() {
                return Err(format!("source_tables {key}: references changed const `{id}` whose text cannot be evaluated"));
            }
            referenced += 1;
        }
    }
    if referenced == 0 {
        return Err(format!(
            "source_tables {key}: references no changed source const (stale entry)"
        ));
    }
    Ok(())
}

fn arch_of(path: &str) -> Result<Arch, String> {
    for (suffix, arch) in [
        ("gfx1151", Arch::Gfx1151),
        ("gfx1201", Arch::Gfx1201),
        ("gfx1100", Arch::Gfx1100),
        ("gfx1030", Arch::Gfx1030),
    ] {
        if path.contains(suffix) {
            return Ok(arch);
        }
    }
    Err(format!(
        "{path}: cannot infer the image architecture from its name"
    ))
}

/// Index PeaceMaker images by their embedded HSA metadata.
pub fn build_pm(root: &Path, images: &[String], syms: &mut Symbols) -> Result<(), String> {
    for path in images {
        let bytes = std::fs::read(root.join(path)).map_err(|e| format!("{path}: {e}"))?;
        let arch = arch_of(path)?;
        let (_, kernels) = Source::read(&bytes, arch).map_err(|e| format!("{path}: {e:?}"))?;
        let origin = format!("image:{path}");
        for k in kernels {
            let m = &k.metadata.parsed;
            let mut args: Vec<Arg> = Vec::new();
            for a in m
                .args
                .iter()
                .filter(|a| !a.value_kind.starts_with("hidden_"))
            {
                args.push(Arg {
                    offset: a.offset as u32,
                    size: a.size as u32,
                    kind: if a.value_kind == "global_buffer" {
                        'p'
                    } else {
                        'v'
                    },
                    text: format!("{}:{}", a.name, a.value_kind),
                });
            }
            let end = args.iter().map(|a| a.offset + a.size).max().unwrap_or(0);
            syms.pm_segment.insert(
                (path.clone(), k.name.clone()),
                m.kernarg_segment_size as u32,
            );
            syms.origin_symbols
                .entry(origin.clone())
                .or_default()
                .push(k.name.clone());
            syms.pm
                .entry(k.name.clone())
                .or_default()
                .push((origin.clone(), Decl { args, end }));
        }
    }
    Ok(())
}

/// Expected 17-argument layout of one external tiled `_t8` candidate symbol:
/// seven 8-byte `global_buffer` pointers at 0..56, ten 4-byte `by_value`
/// scalars at 56..96. `Arg.text` carries `name:value_kind`; both halves are
/// checked exactly.
fn expect_t8_layout(symbol: &str, args: &[Arg], end: u32, segment: u32) -> Result<(), String> {
    const NAMES: [&str; 17] = [
        "expert_weight_ptrs",
        "expert_ids",
        "expert_offsets",
        "slot_ids",
        "Xrot",
        "Y0",
        "Y1",
        "M",
        "K",
        "U",
        "C",
        "x_row_div",
        "x_src_rows",
        "x_row_stride",
        "y_slot_stride",
        "n_experts",
        "reserved",
    ];
    if args.len() != 17 {
        return Err(format!(
            "{symbol}: external declaration has {} arguments, want 17",
            args.len()
        ));
    }
    for (i, (a, want)) in args.iter().zip(NAMES.iter()).enumerate() {
        let mut parts = a.text.splitn(2, ':');
        let name = parts.next().unwrap_or("");
        let vk = parts.next().unwrap_or("");
        if name != *want {
            return Err(format!(
                "{symbol}: arg {i} is named `{name}`, want `{want}`"
            ));
        }
        let (want_off, want_size, want_kind, want_vk) = if i < 7 {
            (i as u32 * 8, 8u32, 'p', "global_buffer")
        } else {
            (56 + (i as u32 - 7) * 4, 4u32, 'v', "by_value")
        };
        if a.offset != want_off || a.size != want_size {
            return Err(format!(
                "{symbol}: arg {i} `{want}` at offset {}, {} bytes; want offset {want_off}, {want_size} bytes",
                a.offset, a.size
            ));
        }
        if a.kind != want_kind {
            return Err(format!(
                "{symbol}: arg {i} `{want}` has class {}, want {want_kind}",
                a.kind
            ));
        }
        if vk != want_vk {
            return Err(format!(
                "{symbol}: arg {i} `{want}` has value_kind `{vk}`, want `{want_vk}`"
            ));
        }
    }
    if end != 96 || segment != 96 {
        return Err(format!(
            "{symbol}: declared end {end} / segment {segment}; want 96 / 96"
        ));
    }
    Ok(())
}

/// Checked JSON-integer to `u32`: `as u32` would silently wrap `2^32 + x`.
fn u32_field(fixture: &str, ctx: &str, v: &serde_json::Value, field: &str) -> Result<u32, String> {
    let n = v
        .get(field)
        .and_then(|x| x.as_u64())
        .ok_or_else(|| format!("{fixture}:{ctx}: missing `{field}`"))?;
    u32::try_from(n).map_err(|_| format!("{fixture}:{ctx}: `{field}` {n} does not fit in u32"))
}

/// A 64-character lowercase hex SHA-256 digest; anything else is refused.
fn sha256_field(
    fixture: &str,
    ctx: &str,
    v: &serde_json::Value,
    field: &str,
) -> Result<String, String> {
    let s = v
        .get(field)
        .and_then(|x| x.as_str())
        .ok_or_else(|| format!("{fixture}:{ctx}: missing `{field}`"))?;
    if s.len() != 64
        || !s
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !(b as char).is_ascii_uppercase())
    {
        return Err(format!(
            "{fixture}:{ctx}: `{field}` is not a 64-char lowercase hex digest"
        ));
    }
    Ok(s.to_string())
}

/// The exact tiled-candidate family: `mq_i4_moe_gemv_[cn_]<codec>_r<kind>_t8`.
/// A bare `_t8` suffix is not enough: any other stem is refused.
fn t8_family_parts(symbol: &str) -> Result<(bool, &str, &str), String> {
    let rest = symbol
        .strip_prefix("mq_i4_moe_gemv_")
        .ok_or_else(|| format!("{symbol}: not a mq_i4_moe_gemv candidate"))?;
    let (cn, rest) = match rest.strip_prefix("cn_") {
        Some(r) => (true, r),
        None => (false, rest),
    };
    let rest = rest
        .strip_suffix("_t8")
        .ok_or_else(|| format!("{symbol}: not a `_t8` candidate"))?;
    let (codec, kind) = rest
        .split_once("_r")
        .ok_or_else(|| format!("{symbol}: no codec/kind separator"))?;
    if codec != "mq3i4" && codec != "mq4sym" {
        return Err(format!("{symbol}: unknown codec `{codec}`"));
    }
    if kind != "256_gate_up" && kind != "128_down" {
        return Err(format!("{symbol}: unknown kind `r{kind}`"));
    }
    Ok((cn, codec, kind))
}

/// The exact external tiled-candidate export family: every `{cn_,}<codec>_r<kind>_t8`
/// combination over both codecs (`mq3i4`, `mq4sym`) and both kinds (`256_gate_up`,
/// `128_down`) on both arches. The fixture must declare all eight and nothing else:
/// a missing export would leave a launch with no declaration, so a subset is a hard
/// error, never a smaller valid family.
const EXPECTED_T8_FAMILY: [&str; 8] = [
    "mq_i4_moe_gemv_cn_mq3i4_r128_down_t8",
    "mq_i4_moe_gemv_cn_mq3i4_r256_gate_up_t8",
    "mq_i4_moe_gemv_cn_mq4sym_r128_down_t8",
    "mq_i4_moe_gemv_cn_mq4sym_r256_gate_up_t8",
    "mq_i4_moe_gemv_mq3i4_r128_down_t8",
    "mq_i4_moe_gemv_mq3i4_r256_gate_up_t8",
    "mq_i4_moe_gemv_mq4sym_r128_down_t8",
    "mq_i4_moe_gemv_mq4sym_r256_gate_up_t8",
];

/// Index the externally supplied tiled `_t8` candidate family from a truthful
/// JSON fixture (see `fixtures/all-launcher-abi/external_t8_abi.json`).
///
/// This pins the DECLARED 96-byte/17-argument layout shared by the eight `_t8`
/// exports; it never opens the out-of-tree `--co` files and claims nothing
/// about any supplied `--co` file at runtime. Every field is validated
/// fail-closed: the exact eight-symbol membership, wrong offsets/sizes/names/
/// kinds/value_kinds/segment, unknown fields, non-`u32` numerics, malformed
/// provenance digests, missing contract paths, wrong arch/module/symbol
/// association, or symbols whose layouts differ are hard errors, never silent
/// defaults.
pub fn build_external(root: &Path, fixture: &str, syms: &mut Symbols) -> Result<(), String> {
    let text =
        std::fs::read_to_string(root.join(fixture)).map_err(|e| format!("{fixture}: {e}"))?;
    let v: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| format!("{fixture}: {e}"))?;
    let obj = v
        .as_object()
        .ok_or_else(|| format!("{fixture}: top level must be an object"))?;
    for k in obj.keys() {
        if !["scope", "provenance", "kernarg_segment_bytes", "symbols"].contains(&k.as_str()) {
            return Err(format!("{fixture}: unknown top-level field `{k}`"));
        }
    }
    if obj.get("kernarg_segment_bytes").and_then(|x| x.as_u64()) != Some(96) {
        return Err(format!("{fixture}: kernarg_segment_bytes must be 96"));
    }
    // Provenance: both raw artifact dirs and both raw module.co SHA-256
    // digests are required, well-formed, and non-empty. The digests pin which
    // raw exports the declared layout was read from; they are format- and
    // presence-checked here, not re-hashed (the loader never opens the .co).
    let prov = obj
        .get("provenance")
        .and_then(|x| x.as_object())
        .ok_or_else(|| format!("{fixture}: missing `provenance` object"))?;
    for k in prov.keys() {
        if ![
            "note",
            "gfx1201_artifact_dir",
            "cn_artifact_dir",
            "gfx1201_module_co_sha256",
            "cn_module_co_sha256",
        ]
        .contains(&k.as_str())
        {
            return Err(format!("{fixture}: unknown provenance field `{k}`"));
        }
    }
    for dir_field in ["gfx1201_artifact_dir", "cn_artifact_dir"] {
        let d = prov
            .get(dir_field)
            .and_then(|x| x.as_str())
            .ok_or_else(|| format!("{fixture}: missing provenance `{dir_field}`"))?;
        if d.is_empty() {
            return Err(format!("{fixture}: empty provenance `{dir_field}`"));
        }
    }
    let prov_gfx1201_sha = sha256_field(
        fixture,
        "provenance",
        &serde_json::Value::Object(prov.clone()),
        "gfx1201_module_co_sha256",
    )?;
    let prov_cn_sha = sha256_field(
        fixture,
        "provenance",
        &serde_json::Value::Object(prov.clone()),
        "cn_module_co_sha256",
    )?;
    let _ = (prov_gfx1201_sha, prov_cn_sha);
    let syms_obj = obj
        .get("symbols")
        .and_then(|x| x.as_object())
        .ok_or_else(|| format!("{fixture}: `symbols` must be an object"))?;
    if syms_obj.is_empty() {
        return Err(format!("{fixture}: no symbols declared"));
    }
    // Exact family membership: every expected export present, nothing else. (A
    // 7-of-8 subset used to load fine and left the dropped launch with no
    // declaration downstream; it must fail here instead.)
    {
        let mut missing: Vec<&str> = Vec::new();
        for want in EXPECTED_T8_FAMILY {
            if !syms_obj.contains_key(want) {
                missing.push(want);
            }
        }
        let mut extra: Vec<&String> = syms_obj
            .keys()
            .filter(|k| !EXPECTED_T8_FAMILY.contains(&k.as_str()))
            .collect();
        extra.sort();
        if !missing.is_empty() || !extra.is_empty() {
            return Err(format!(
                "{fixture}: external family must declare exactly the eight _t8 exports; missing {missing:?}, unexpected {extra:?}"
            ));
        }
    }
    let origin = format!("external:{fixture}");
    let mut layouts: Vec<Decl> = Vec::new();
    for (symbol, s) in syms_obj {
        let s = s
            .as_object()
            .ok_or_else(|| format!("{fixture}:{symbol} must be an object"))?;
        for k in s.keys() {
            if ![
                "arch", "module", "contract", "export", "end", "segment", "args",
            ]
            .contains(&k.as_str())
            {
                return Err(format!("{fixture}:{symbol}: unknown field `{k}`"));
            }
        }
        let (cn, _codec, _kind) = t8_family_parts(symbol).map_err(|e| format!("{fixture}:{e}"))?;
        let arch = s
            .get("arch")
            .and_then(|x| x.as_str())
            .ok_or_else(|| format!("{fixture}:{symbol}: missing `arch`"))?;
        if arch != "gfx1151" && arch != "gfx1201" {
            return Err(format!("{fixture}:{symbol}: unknown arch `{arch}`"));
        }
        // Exact arch/module/symbol association: gfx1151 is the Cn family in
        // the Cn module; gfx1201 is non-Cn in the K1 module. A swapped or
        // foreign module is refused, not remapped.
        let want_module = if arch == "gfx1151" {
            "mq_i4_moe_gemv_cn_pm_gfx1151"
        } else {
            "mq_i4_moe_gemv_pm_gfx1201"
        };
        let module = s
            .get("module")
            .and_then(|x| x.as_str())
            .ok_or_else(|| format!("{fixture}:{symbol}: missing `module`"))?;
        if module != want_module {
            return Err(format!(
                "{fixture}:{symbol}: module `{module}` does not match arch {arch} (want `{want_module}`)"
            ));
        }
        if (arch == "gfx1151") != cn {
            return Err(format!(
                "{fixture}:{symbol}: arch/module mismatch (arch {arch}, cn {cn})"
            ));
        }
        // Contract provenance is required: a non-empty `.contract.json` path
        // naming this symbol's arch, so a declaration without raw-export
        // provenance cannot slip in.
        let contract = s
            .get("contract")
            .and_then(|x| x.as_str())
            .ok_or_else(|| format!("{fixture}:{symbol}: missing `contract`"))?;
        if contract.is_empty() || !contract.ends_with(".contract.json") || !contract.contains(arch)
        {
            return Err(format!("{fixture}:{symbol}: bad `contract` `{contract}`"));
        }
        if s.get("export").and_then(|x| x.as_str()) != Some("FUNC") {
            return Err(format!("{fixture}:{symbol}: export must be FUNC"));
        }
        let sv = serde_json::Value::Object(s.clone());
        let end = u32_field(fixture, symbol, &sv, "end")?;
        let segment = u32_field(fixture, symbol, &sv, "segment")?;
        let arr = s
            .get("args")
            .and_then(|x| x.as_array())
            .ok_or_else(|| format!("{fixture}:{symbol}: `args` must be an array"))?;
        let mut args: Vec<Arg> = Vec::new();
        for a in arr {
            let a = a
                .as_object()
                .ok_or_else(|| format!("{fixture}:{symbol}: arg must be an object"))?;
            for k in a.keys() {
                if !["name", "offset", "size", "kind", "value_kind"].contains(&k.as_str()) {
                    return Err(format!("{fixture}:{symbol}: unknown arg field `{k}`"));
                }
            }
            let aname = a
                .get("name")
                .and_then(|x| x.as_str())
                .ok_or_else(|| format!("{fixture}:{symbol}: arg missing `name`"))?;
            let av = serde_json::Value::Object(a.clone());
            let offset = u32_field(fixture, &format!("{symbol} arg `{aname}`"), &av, "offset")?;
            let size = u32_field(fixture, &format!("{symbol} arg `{aname}`"), &av, "size")?;
            let kind = a
                .get("kind")
                .and_then(|x| x.as_str())
                .ok_or_else(|| format!("{fixture}:{symbol}: arg `{aname}` missing `kind`"))?;
            let kind = match kind {
                "p" => 'p',
                "v" => 'v',
                _ => {
                    return Err(format!(
                        "{fixture}:{symbol}: arg `{aname}` has unknown kind `{kind}`"
                    ))
                }
            };
            let vk = a
                .get("value_kind")
                .and_then(|x| x.as_str())
                .ok_or_else(|| format!("{fixture}:{symbol}: arg `{aname}` missing `value_kind`"))?;
            if vk != "global_buffer" && vk != "by_value" {
                return Err(format!(
                    "{fixture}:{symbol}: arg `{aname}` has unknown value_kind `{vk}`"
                ));
            }
            args.push(Arg {
                offset,
                size,
                kind,
                text: format!("{aname}:{vk}"),
            });
        }
        expect_t8_layout(symbol, &args, end, segment)?;
        let decl = Decl { args, end };
        layouts.push(decl.clone());
        syms.origin_symbols
            .entry(origin.clone())
            .or_default()
            .push(symbol.clone());
        syms.pm
            .entry(symbol.clone())
            .or_default()
            .push((origin.clone(), decl));
    }
    for d in layouts.iter().skip(1) {
        if !crate::hip::same_layout(&layouts[0], d) {
            return Err(format!(
                "{fixture}: external symbols declare different ABIs"
            ));
        }
    }
    Ok(())
}

/// Argument lists of every call `name(..)` in `ts` (nested groups included).
pub fn calls_of(ts: &[Tt], name: &str, out: &mut Vec<Vec<Vec<Tt>>>) {
    for k in 0..ts.len() {
        if let Tt::G(d, c, _) = &ts[k] {
            if *d == '(' && k > 0 && ts[k - 1].is_id(name) {
                out.push(split_top(c, ','));
            }
            calls_of(c, name, out);
        }
    }
}

fn strip_refs(ts: &[Tt]) -> Vec<Tt> {
    let mut t: Vec<Tt> = ts.to_vec();
    while t.first().map_or(false, |x| x.is_p('&')) {
        t.remove(0);
    }
    t
}

/// Does this expression build a name that ENDS with the value of the const `konst`?
/// Only `format!("..{KONST}")` / `format!("{}{}", .., KONST)` count (a plain mention of
/// the const elsewhere in the function does not), following a single local binding.
/// Anything that could change which value the placeholder prints is refused: a named
/// format argument `KONST = ..`, or a local/parameter that shadows the const.
pub fn name_ends_with_const(
    ts: &[Tt],
    konst: &str,
    idx: &BindIndex,
    params: &std::collections::HashSet<String>,
    depth: u32,
) -> bool {
    if params.contains(konst) || idx.map.contains_key(konst) {
        return false;
    }
    let t = strip_refs(ts);
    if t.len() == 1 {
        if let Some(id) = t[0].id() {
            if depth < 3 {
                if let Some(bs) = idx.map.get(id) {
                    return bs.len() == 1
                        && name_ends_with_const(&bs[0].init, konst, idx, params, depth + 1);
                }
            }
            return false;
        }
    }
    if t.len() == 3 && t[0].is_id("format") && t[1].is_p('!') {
        if let Some(g) = t[2].group() {
            let parts = split_top(g, ',');
            let overridden = parts.iter().skip(1).any(|p| {
                p.len() >= 2
                    && p[0].is_id(konst)
                    && p[1].is_p('=')
                    && !p.get(2).map_or(false, |x| x.is_p('='))
            });
            if overridden {
                return false;
            }
            if let Some([Tt::S(tpl, _)]) = parts.first().map(|p| p.as_slice()) {
                if tpl.ends_with(&format!("{{{konst}}}")) {
                    return true;
                }
                if tpl.ends_with("{}") && parts.len() >= 2 {
                    let last = parts.last().unwrap();
                    return last.len() == 1 && last[0].is_id(konst);
                }
            }
        }
    }
    false
}

/// Does this expression (or the single local it names) call `func`?
pub fn expr_uses_fn(ts: &[Tt], func: &str, idx: &BindIndex, depth: u32) -> bool {
    let t = strip_refs(ts);
    let mut ids = Vec::new();
    idents_in(&t, &mut ids);
    if ids.iter().any(|i| i == func) {
        return true;
    }
    if t.len() == 1 && depth < 3 {
        if let Some(id) = t[0].id() {
            if let Some(bs) = idx.map.get(id) {
                return bs.len() == 1 && expr_uses_fn(&bs[0].init, func, idx, depth + 1);
            }
        }
    }
    false
}

/// The value the OLD helpers' `OLD_SUFFIX` placeholder really prints: one const in
/// the file whose only string is `__beta` (an empty or changed suffix would make the
/// "OLD" launch name the current kernel).
pub fn old_suffix_is_beta(world: &World, file: &str) -> Result<(), String> {
    let f = world
        .file(file)
        .ok_or_else(|| format!("{file}: file not scanned"))?;
    let defs = f
        .consts
        .get("OLD_SUFFIX")
        .filter(|d| d.len() == 1)
        .ok_or("no unique OLD_SUFFIX const")?;
    let mut lits = Vec::new();
    strings_in(&defs[0].init(), &mut lits);
    if lits != ["__beta".to_string()] {
        return Err(format!("OLD_SUFFIX is {lits:?}, expected \"__beta\""));
    }
    Ok(())
}
