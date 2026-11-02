// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Product reader of the PM tune table (`hipfire.tune_table.v1`).
//!
//! `HIPFIRE_PM_TUNE_TABLE=<path>` (developer var) names a table exported by
//! `pm-census`; it is the only input. With the variable unset no table exists:
//! [`lookup`] is one `LazyLock` load that returns `None`, nothing is read or
//! hashed, and the planner and selector use the shipped image exactly as
//! before.
//!
//! The reader mirrors only what it consumes (`schema`, `rows[].product`,
//! `rows[].key.numeric_contract` for logging); `rdna-compute` must not depend
//! on `pm-census`, and unknown fields are ignored.
//!
//! Selection is by exact `(arch, symbol, M, K)`. `N` is not keyed: the
//! function cache is symbol-keyed, so one image binds a symbol per process.
//! A row is refused (the incumbent runs, one `pm_tune:` stderr note) when its
//! `incumbent_image_sha256` is not the sha256 of the image this build would
//! otherwise load (stale key), when the code object is unreadable or its
//! sha256 differs from `code_object_sha256`, or when two rows share an exact
//! key (all of them refused). Rows that are malformed or carry no `product`
//! block are ignored with a note, and a wrong `schema` refuses the whole
//! table. A table that does not parse at all is a hard error at the first
//! lookup: the explicit opt-in was given, so it never degrades silently.
//!
//! A hit loads the variant under the distinct module name
//! `pm_tune:<sha16>` with the same kernel symbol, geometry and kernargs. The
//! route planner (`kernel_registry::route_entries_sealed`) consults the same
//! [`lookup`] so the variant, not the incumbent, is preloaded; a dispatch-time
//! hit then finds the symbol already bound to that image. If the symbol is
//! bound to a different image in this process the launch fails loudly: there
//! is no silent fallback.

use crate::Gpu;
use hip_bridge::{HipError, HipResult};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex, OnceLock};

/// Developer variable naming the table file.
pub const TABLE_VAR: &str = "HIPFIRE_PM_TUNE_TABLE";
/// The only schema the reader accepts.
pub const SCHEMA: &str = "hipfire.tune_table.v1";

/// The one table-eligible symbol of this slice (non-bf16 slab gate_up SiLU).
pub(crate) const GATE_UP_B1S_SYMBOL: &str = "gemm_mq4g256v2_gate_up_silu_mmq_iu4_b1s";

/// Local serde mirror of `pm-census` `ProductSelectV1`.
#[derive(Debug, Deserialize)]
struct ProductIn {
    symbol: String,
    arch: String,
    m: u64,
    k: u64,
    incumbent_image_sha256: String,
    code_object: String,
    code_object_sha256: String,
}

/// Local serde mirror of the key fields the reader logs.
#[derive(Debug, Default, Deserialize)]
struct KeyIn {
    #[serde(default)]
    numeric_contract: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RowIn {
    #[serde(default)]
    key: Option<KeyIn>,
    #[serde(default)]
    product: Option<ProductIn>,
}

/// One product-selectable row, paths already resolved.
#[derive(Debug)]
struct Row {
    /// Position in `rows[]`, for diagnostics.
    index: usize,
    symbol: String,
    arch: String,
    m: u64,
    k: u64,
    incumbent_image_sha256: String,
    code_object: PathBuf,
    code_object_sha256: String,
    numeric_contract: Option<String>,
}

impl Row {
    fn row_id(&self) -> String {
        format!(
            "rows[{}]:{}:{}:{}x{}:{}",
            self.index,
            self.arch,
            self.symbol,
            self.m,
            self.k,
            self.sha16()
        )
    }

    fn sha16(&self) -> &str {
        &self.code_object_sha256[..self.code_object_sha256.len().min(16)]
    }

    fn key_matches(&self, arch: &str, symbol: &str, m: u64, k: u64) -> bool {
        self.arch == arch && self.symbol == symbol && self.m == m && self.k == k
    }
}

/// A parsed table. Row-level problems are recorded in `notes` (the row is not
/// selectable); a table-level refusal (wrong schema) makes every lookup refuse.
#[derive(Debug)]
pub(crate) struct Table {
    refusal: Option<String>,
    rows: Vec<Row>,
    notes: Vec<String>,
}

/// Outcome of one lookup.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Selection {
    Incumbent,
    Variant {
        module_name: String,
        bytes: Vec<u8>,
        row_id: String,
    },
    Refused {
        reason: String,
    },
}

/// Lowercase hex sha256.
fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for b in digest {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

/// Parse table bytes. `table_dir` anchors relative `code_object` paths.
///
/// `Err` only when the bytes are not a table document at all (not JSON, no
/// string `schema`, or a matching schema without a `rows` array). A wrong
/// schema or a bad row parses into a [`Table`] that refuses.
pub(crate) fn parse_table(bytes: &[u8], table_dir: &Path) -> Result<Table, String> {
    let doc: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|e| format!("not valid JSON: {e}"))?;
    let obj = doc
        .as_object()
        .ok_or_else(|| "top level is not a JSON object".to_owned())?;
    let schema = obj
        .get("schema")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "missing string `schema`".to_owned())?;
    if schema != SCHEMA {
        return Ok(Table {
            refusal: Some(format!("schema {schema:?} is not {SCHEMA:?}")),
            rows: Vec::new(),
            notes: Vec::new(),
        });
    }
    let rows_in = obj
        .get("rows")
        .and_then(|v| v.as_array())
        .ok_or_else(|| "missing `rows` array".to_owned())?;
    let mut rows = Vec::new();
    let mut notes = Vec::new();
    for (index, value) in rows_in.iter().enumerate() {
        let row: RowIn = match serde_json::from_value(value.clone()) {
            Ok(r) => r,
            Err(e) => {
                notes.push(format!("rows[{index}] malformed, ignored: {e}"));
                continue;
            }
        };
        let Some(p) = row.product else {
            notes.push(format!("rows[{index}] has no product block, ignored"));
            continue;
        };
        let co = Path::new(&p.code_object);
        let code_object = if co.is_absolute() {
            co.to_path_buf()
        } else {
            table_dir.join(co)
        };
        rows.push(Row {
            index,
            symbol: p.symbol,
            arch: p.arch,
            m: p.m,
            k: p.k,
            incumbent_image_sha256: p.incumbent_image_sha256.to_ascii_lowercase(),
            code_object,
            code_object_sha256: p.code_object_sha256.to_ascii_lowercase(),
            numeric_contract: row.key.and_then(|k| k.numeric_contract),
        });
    }
    Ok(Table {
        refusal: None,
        rows,
        notes,
    })
}

/// Pure selection. `incumbent_image_sha256` is the sha256 of the image this
/// build would otherwise load for `symbol`; `read_file` reads a code object.
pub(crate) fn select(
    table: &Table,
    arch: &str,
    symbol: &str,
    m: usize,
    k: usize,
    incumbent_image_sha256: &str,
    read_file: impl Fn(&Path) -> io::Result<Vec<u8>>,
) -> Selection {
    if let Some(reason) = &table.refusal {
        return Selection::Refused {
            reason: reason.clone(),
        };
    }
    let (m, k) = (m as u64, k as u64);
    let mut hits = table
        .rows
        .iter()
        .filter(|r| r.key_matches(arch, symbol, m, k));
    let Some(row) = hits.next() else {
        return Selection::Incumbent;
    };
    if let Some(other) = hits.next() {
        return Selection::Refused {
            reason: format!(
                "duplicate exact key ({arch}, {symbol}, {m}, {k}): {} and {} (all rows of the key refused)",
                row.row_id(),
                other.row_id()
            ),
        };
    }
    let id = row.row_id();
    if row.incumbent_image_sha256 != incumbent_image_sha256.to_ascii_lowercase() {
        return Selection::Refused {
            reason: format!(
                "{id}: stale key: row incumbent_image_sha256 {} != this build's image {}",
                row.incumbent_image_sha256, incumbent_image_sha256
            ),
        };
    }
    let bytes = match read_file(&row.code_object) {
        Ok(b) => b,
        Err(e) => {
            return Selection::Refused {
                reason: format!(
                    "{id}: code object {} unreadable: {e}",
                    row.code_object.display()
                ),
            };
        }
    };
    let got = sha256_hex(&bytes);
    if got != row.code_object_sha256 {
        return Selection::Refused {
            reason: format!(
                "{id}: code object {} sha256 {got} != row code_object_sha256 {}",
                row.code_object.display(),
                row.code_object_sha256
            ),
        };
    }
    Selection::Variant {
        module_name: format!("pm_tune:{}", row.sha16()),
        bytes,
        row_id: id,
    }
}

/// The variant selected for this process. Owned data held by a process-wide
/// `OnceLock`, so `&'static PmVariant` needs no leaking. One image per symbol
/// per process and one eligible symbol make one slot sufficient.
#[derive(Debug)]
pub(crate) struct PmVariant {
    pub(crate) module: String,
    pub(crate) bytes: Vec<u8>,
    pub(crate) row_id: String,
}

static SELECTED: OnceLock<PmVariant> = OnceLock::new();

/// `None`: variable unset (the product default). `Some(Err)`: set but the
/// table file could not be read or parsed — a hard error at first lookup.
static TABLE: LazyLock<Option<Result<Table, String>>> = LazyLock::new(|| {
    let path = hipfire_config::developer_var(TABLE_VAR).ok()?;
    if path.is_empty() {
        return None;
    }
    Some(load_table(Path::new(&path)).inspect(|t| {
        for note in &t.notes {
            eprintln!("pm_tune: {path}: {note}");
        }
        eprintln!("pm_tune: table {path}: {} product row(s)", t.rows.len());
    }))
});

fn load_table(path: &Path) -> Result<Table, String> {
    let bytes = std::fs::read(path)
        .map_err(|e| format!("{TABLE_VAR}={}: unreadable: {e}", path.display()))?;
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    parse_table(&bytes, dir).map_err(|e| format!("{TABLE_VAR}={}: {e}", path.display()))
}

type DecisionKey = (String, usize, usize);
static DECISIONS: LazyLock<Mutex<HashMap<DecisionKey, Option<&'static PmVariant>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
/// sha256 of the incumbent image; computed only once a table is configured.
static INCUMBENT_SHA: OnceLock<String> = OnceLock::new();

/// True when `HIPFIRE_PM_TUNE_TABLE` is set (the table is then loaded here).
/// Unset is the product default and costs one `LazyLock` load.
#[inline]
pub(crate) fn enabled() -> bool {
    TABLE.is_some()
}

/// Decide the image for `(arch, symbol, m, k)`; `incumbent` is the image this
/// build would otherwise load. Only [`GATE_UP_B1S_SYMBOL`] is eligible. With
/// no table configured this is a single `LazyLock` check: no I/O, no hashing,
/// no allocation. Decisions are cached per `(arch, m, k)`.
pub(crate) fn lookup(
    arch: &str,
    symbol: &str,
    m: usize,
    k: usize,
    incumbent: &[u8],
) -> HipResult<Option<&'static PmVariant>> {
    let Some(loaded) = &*TABLE else {
        return Ok(None);
    };
    if symbol != GATE_UP_B1S_SYMBOL {
        return Ok(None);
    }
    let table = loaded
        .as_ref()
        .map_err(|e| HipError::new(0, &format!("pm_tune: {e}")))?;
    let key = (arch.to_owned(), m, k);
    let mut decisions = DECISIONS.lock().unwrap_or_else(|p| p.into_inner());
    if let Some(d) = decisions.get(&key) {
        return Ok(*d);
    }
    let incumbent_sha = INCUMBENT_SHA.get_or_init(|| sha256_hex(incumbent));
    let decision = match select(table, arch, symbol, m, k, incumbent_sha, |p| {
        std::fs::read(p)
    }) {
        Selection::Incumbent => None,
        Selection::Refused { reason } => {
            eprintln!("pm_tune: refused, using incumbent for {symbol} m={m} k={k}: {reason}");
            None
        }
        Selection::Variant {
            module_name,
            bytes,
            row_id,
        } => {
            let contract = table
                .rows
                .iter()
                .find(|r| r.key_matches(arch, symbol, m as u64, k as u64))
                .and_then(|r| r.numeric_contract.as_deref())
                .unwrap_or("unspecified");
            eprintln!(
                "pm_tune: selected {row_id} for {symbol} m={m} k={k} as module {module_name} (numeric_contract {contract})"
            );
            let variant = SELECTED.get_or_init(|| PmVariant {
                module: module_name.clone(),
                bytes: bytes.clone(),
                row_id: row_id.clone(),
            });
            if variant.bytes != bytes {
                return Err(HipError::new(
                    0,
                    &format!(
                        "pm_tune: {row_id} selects a different image for {symbol} than {} \
                         already selected in this process (one image per symbol per process)",
                        variant.row_id
                    ),
                ));
            }
            Some(variant)
        }
    };
    decisions.insert(key, decision);
    Ok(decision)
}

impl Gpu {
    /// Bind a selected variant for `kernel`. Fails loudly (never falls back)
    /// when the symbol is already bound to another image in this process: a
    /// planned route's preload binds the image it planned (the variant, when
    /// the planner saw the same table row), and any earlier launch of
    /// another shape binds its own.
    pub(crate) fn ensure_pm_tune_variant(
        &mut self,
        variant: &'static PmVariant,
        kernel: &str,
    ) -> HipResult<()> {
        self.ensure_route_modules_preloaded()?;
        if self.functions.contains_key(kernel) {
            return self
                .compiler
                .code_objects()
                .check_function_image(kernel, &variant.bytes)
                .map_err(|reason| {
                    HipError::new(
                        0,
                        &format!(
                            "pm_tune: {} selected a variant for {kernel} but the symbol is already \
                             bound to a different image in this process (a route preload or an \
                             earlier launch bound the incumbent); refusing to fall back: {reason}",
                            variant.row_id
                        ),
                    )
                });
        }
        self.ensure_embedded_kernel(variant.module.as_str(), variant.bytes.as_slice(), kernel)
            .map_err(|e| {
                HipError::new(
                    0,
                    &format!(
                        "pm_tune: {} variant load for {kernel} failed: {e}",
                        variant.row_id
                    ),
                )
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};
    use std::cell::RefCell;

    const SYM: &str = GATE_UP_B1S_SYMBOL;
    const ARCH: &str = "gfx1201";
    const M: usize = 17408;
    const K: usize = 5120;
    const INCUMBENT: &[u8] = b"incumbent image bytes";
    const VARIANT: &[u8] = b"variant code object bytes";

    fn row(code_object: &str) -> Value {
        json!({
            "key": { "numeric_contract": "exact", "unrelated": 1 },
            "speedup": 1.1,
            "product": {
                "symbol": SYM,
                "arch": ARCH,
                "m": M,
                "k": K,
                "incumbent_image_sha256": sha256_hex(INCUMBENT),
                "code_object": code_object,
                "code_object_sha256": sha256_hex(VARIANT),
            }
        })
    }

    fn table_json(schema: &str, rows: Vec<Value>) -> Vec<u8> {
        serde_json::to_vec(&json!({ "schema": schema, "rows": rows, "extra": [1, 2] })).unwrap()
    }

    fn parse(rows: Vec<Value>) -> Table {
        parse_table(&table_json(SCHEMA, rows), Path::new("/tbl")).unwrap()
    }

    fn serve(path: &Path) -> io::Result<Vec<u8>> {
        if path == Path::new("/tbl/objects/v.co") {
            Ok(VARIANT.to_vec())
        } else {
            Err(io::Error::new(io::ErrorKind::NotFound, "nope"))
        }
    }

    fn run(t: &Table, arch: &str, sym: &str, m: usize, k: usize) -> Selection {
        select(t, arch, sym, m, k, &sha256_hex(INCUMBENT), serve)
    }

    fn assert_refused(sel: Selection, needle: &str) {
        match sel {
            Selection::Refused { reason } => {
                assert!(reason.contains(needle), "{needle:?} not in {reason:?}")
            }
            other => panic!("expected Refused({needle}), got {other:?}"),
        }
    }

    #[test]
    fn exact_key_hit() {
        let t = parse(vec![row("objects/v.co")]);
        match run(&t, ARCH, SYM, M, K) {
            Selection::Variant {
                module_name,
                bytes,
                row_id,
            } => {
                assert_eq!(bytes, VARIANT);
                assert_eq!(
                    module_name,
                    format!("pm_tune:{}", &sha256_hex(VARIANT)[..16])
                );
                assert!(row_id.starts_with("rows[0]:gfx1201:"), "{row_id}");
            }
            other => panic!("expected hit, got {other:?}"),
        }
    }

    #[test]
    fn miss_on_each_key_field() {
        let t = parse(vec![row("objects/v.co")]);
        assert_eq!(run(&t, "gfx1100", SYM, M, K), Selection::Incumbent);
        assert_eq!(
            run(
                &t,
                ARCH,
                "gemm_mq4g256v2_gate_up_silu_mmq_iu4_b1s_bf16",
                M,
                K
            ),
            Selection::Incumbent
        );
        assert_eq!(run(&t, ARCH, SYM, M + 128, K), Selection::Incumbent);
        assert_eq!(run(&t, ARCH, SYM, M, K + 256), Selection::Incumbent);
    }

    #[test]
    fn stale_incumbent_key_refused() {
        let t = parse(vec![row("objects/v.co")]);
        let sel = select(&t, ARCH, SYM, M, K, &sha256_hex(b"other build"), serve);
        assert_refused(sel, "stale key");
    }

    #[test]
    fn code_object_sha_mismatch_refused() {
        let t = parse(vec![row("objects/v.co")]);
        let sel = select(&t, ARCH, SYM, M, K, &sha256_hex(INCUMBENT), |_| {
            Ok(b"tampered".to_vec())
        });
        assert_refused(sel, "code_object_sha256");
    }

    #[test]
    fn unreadable_code_object_refused() {
        let t = parse(vec![row("objects/missing.co")]);
        assert_refused(run(&t, ARCH, SYM, M, K), "unreadable");
    }

    #[test]
    fn duplicate_exact_keys_refused() {
        let t = parse(vec![row("objects/v.co"), row("objects/v.co")]);
        assert_refused(run(&t, ARCH, SYM, M, K), "duplicate exact key");
        // A different key in the same table is unaffected.
        assert_eq!(run(&t, ARCH, SYM, M + 128, K), Selection::Incumbent);
    }

    #[test]
    fn wrong_schema_refused() {
        let t = parse_table(
            &table_json("hipfire.tune_table.v2", vec![row("objects/v.co")]),
            Path::new("/tbl"),
        )
        .unwrap();
        assert_refused(run(&t, ARCH, SYM, M, K), "schema");
        assert_refused(run(&t, ARCH, SYM, M + 1, K), "schema");
    }

    #[test]
    fn unparseable_table_is_an_error() {
        let dir = Path::new("/tbl");
        assert!(parse_table(b"not json", dir).is_err());
        assert!(parse_table(b"[1,2]", dir).is_err());
        assert!(parse_table(br#"{"rows":[]}"#, dir).is_err());
        assert!(parse_table(format!(r#"{{"schema":"{SCHEMA}"}}"#).as_bytes(), dir).is_err());
    }

    #[test]
    fn empty_table_is_incumbent() {
        let t = parse(vec![]);
        assert_eq!(run(&t, ARCH, SYM, M, K), Selection::Incumbent);
    }

    #[test]
    fn malformed_and_productless_rows_are_ignored() {
        let mut bad = row("objects/v.co");
        bad["product"]["m"] = json!("not a number");
        let t = parse(vec![
            bad,
            json!({ "key": {}, "speedup": 1.0 }),
            row("objects/v.co"),
        ]);
        assert_eq!(t.notes.len(), 2, "{:?}", t.notes);
        assert!(matches!(
            run(&t, ARCH, SYM, M, K),
            Selection::Variant { .. }
        ));
    }

    #[test]
    fn relative_and_absolute_code_object_paths() {
        let seen = RefCell::new(Vec::<PathBuf>::new());
        let record = |p: &Path| {
            seen.borrow_mut().push(p.to_path_buf());
            Ok(VARIANT.to_vec())
        };
        let inc = sha256_hex(INCUMBENT);

        let rel = parse(vec![row("objects/v.co")]);
        assert!(matches!(
            select(&rel, ARCH, SYM, M, K, &inc, record),
            Selection::Variant { .. }
        ));
        let abs = parse(vec![row("/elsewhere/abs.co")]);
        assert!(matches!(
            select(&abs, ARCH, SYM, M, K, &inc, record),
            Selection::Variant { .. }
        ));
        assert_eq!(
            *seen.borrow(),
            vec![
                PathBuf::from("/tbl/objects/v.co"),
                PathBuf::from("/elsewhere/abs.co")
            ]
        );
    }
}
