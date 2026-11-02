// SPDX-License-Identifier: Apache-2.0
//! Qualified gfx1100 epilogue hooks.
//!
//! `QualifiedHooks` answers `EpilogueIntrinsics::{exp2, ldexp, divide}` ONLY from observations recorded by the
//! independent gfx1100 intrinsic probe (`consumer24.gfx1100.epilogue-probe.v1`), keyed by exact operand bits.
//! There is no portable fallback: a missing key records an error, returns a quiet NaN (the trait returns `f32`)
//! and the caller MUST fail the case after the CPU call (`missing_count` / `missing_samples`).
//! `CpuOnlySelfCheckHooks` is the one explicitly CPU-only hook set; it is never constructed in device mode.

use hipfire_runtime::mq_i4_reference::{EpilogueIntrinsics, PortableEpilogue};
use serde_json::{json, Value};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;

pub const SCHEMA: &str = "consumer24.gfx1100.epilogue-probe.v1";

/// One probe record. `from_exception` marks entries of the separate `exceptions` array.
#[derive(Clone, Copy, Debug)]
pub struct Entry {
    pub gate_bits: u32,
    pub exp2_arg: u32,
    pub exp2_result: u32,
    pub ldexp_arg: u32,
    pub ldexp_exponent: i32,
    pub ldexp_result: u32,
    pub divide_num: u32,
    pub divide_den: u32,
    pub divide_result: u32,
    pub from_exception: bool,
}

fn num_u32(v: &Value, key: &str) -> Result<u32, String> {
    let x = v
        .get(key)
        .ok_or_else(|| format!("probe entry missing field `{key}`"))?;
    if let Some(n) = x.as_u64() {
        return u32::try_from(n).map_err(|_| format!("field `{key}` = {n} does not fit u32"));
    }
    if let Some(s) = x.as_str() {
        let t = s.trim();
        let parsed = if let Some(h) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
            u32::from_str_radix(h, 16)
        } else {
            t.parse::<u32>()
        };
        return parsed.map_err(|e| format!("field `{key}` = {s:?}: {e}"));
    }
    Err(format!(
        "field `{key}` is neither a u32 number nor a numeric string"
    ))
}

fn num_i32(v: &Value, key: &str) -> Result<i32, String> {
    let x = v
        .get(key)
        .ok_or_else(|| format!("probe entry missing field `{key}`"))?;
    if let Some(n) = x.as_i64() {
        return i32::try_from(n).map_err(|_| format!("field `{key}` = {n} does not fit i32"));
    }
    if let Some(s) = x.as_str() {
        return s
            .trim()
            .parse::<i32>()
            .map_err(|e| format!("field `{key}` = {s:?}: {e}"));
    }
    Err(format!(
        "field `{key}` is neither an i32 number nor a numeric string"
    ))
}

fn parse_entry(v: &Value, from_exception: bool) -> Result<Entry, String> {
    Ok(Entry {
        gate_bits: num_u32(v, "gate_bits")?,
        exp2_arg: num_u32(v, "exp2_arg")?,
        exp2_result: num_u32(v, "exp2_result")?,
        ldexp_arg: num_u32(v, "ldexp_arg")?,
        ldexp_exponent: num_i32(v, "ldexp_exponent")?,
        ldexp_result: num_u32(v, "ldexp_result")?,
        divide_num: num_u32(v, "divide_num")?,
        divide_den: num_u32(v, "divide_den")?,
        divide_result: num_u32(v, "divide_result")?,
        from_exception,
    })
}

fn insert<
    K: std::hash::Hash + Eq + std::fmt::Debug + Copy,
    V: PartialEq + std::fmt::Debug + Copy,
>(
    map: &mut HashMap<K, V>,
    key: K,
    value: V,
    what: &str,
    identical_dups: &mut usize,
) -> Result<(), String> {
    match map.get(&key) {
        None => {
            map.insert(key, value);
            Ok(())
        }
        Some(prev) if *prev == value => {
            *identical_dups += 1;
            Ok(())
        }
        Some(prev) => Err(format!(
            "conflicting duplicate {what} observation for key {key:?}: {prev:?} vs {value:?}"
        )),
    }
}

pub struct QualifiedHooks {
    exp2: HashMap<u32, u32>,
    ldexp: HashMap<(u32, i32), u32>,
    divide: HashMap<(u32, u32), u32>,
    pub entries: Vec<Entry>,
    pub identical_duplicates: usize,
    pub top_level: Value,
    hits: Cell<u64>,
    missing_count: Cell<u64>,
    missing_samples: RefCell<Vec<String>>,
}

impl QualifiedHooks {
    /// Parse a probe observation document. Conflicting duplicate keys are rejected; identical duplicates counted.
    pub fn from_json(doc: &Value) -> Result<Self, String> {
        let schema = doc
            .get("schema")
            .and_then(Value::as_str)
            .ok_or("probe table has no `schema` string")?;
        if schema != SCHEMA {
            return Err(format!("probe table schema {schema:?} != {SCHEMA:?}"));
        }
        let mut entries = Vec::new();
        for (field, exc) in [("observations", false), ("exceptions", true)] {
            match doc.get(field) {
                Some(Value::Array(a)) => {
                    for (i, e) in a.iter().enumerate() {
                        entries
                            .push(parse_entry(e, exc).map_err(|m| format!("{field}[{i}]: {m}"))?);
                    }
                }
                Some(_) => return Err(format!("probe table `{field}` is not an array")),
                None if !exc => return Err("probe table has no `observations` array".into()),
                None => {}
            }
        }
        let (mut exp2, mut ldexp, mut divide) = (HashMap::new(), HashMap::new(), HashMap::new());
        let mut dups = 0usize;
        for e in &entries {
            insert(&mut exp2, e.exp2_arg, e.exp2_result, "exp2", &mut dups)?;
            insert(
                &mut ldexp,
                (e.ldexp_arg, e.ldexp_exponent),
                e.ldexp_result,
                "ldexp",
                &mut dups,
            )?;
            insert(
                &mut divide,
                (e.divide_num, e.divide_den),
                e.divide_result,
                "divide",
                &mut dups,
            )?;
        }
        let mut top = doc.clone();
        if let Some(o) = top.as_object_mut() {
            o.remove("observations");
            o.remove("exceptions");
        }
        Ok(Self {
            exp2,
            ldexp,
            divide,
            entries,
            identical_duplicates: dups,
            top_level: top,
            hits: Cell::new(0),
            missing_count: Cell::new(0),
            missing_samples: RefCell::new(Vec::new()),
        })
    }

    pub fn missing_count(&self) -> u64 {
        self.missing_count.get()
    }
    pub fn hits(&self) -> u64 {
        self.hits.get()
    }
    pub fn missing_samples(&self) -> Vec<String> {
        self.missing_samples.borrow().clone()
    }
    pub fn clear_missing(&self) {
        self.missing_count.set(0);
        self.missing_samples.borrow_mut().clear();
    }
    pub fn key_counts(&self) -> Value {
        json!({"exp2": self.exp2.len(), "ldexp": self.ldexp.len(), "divide": self.divide.len(),
               "observations": self.entries.iter().filter(|e| !e.from_exception).count(),
               "exceptions": self.entries.iter().filter(|e| e.from_exception).count(),
               "identical_duplicates": self.identical_duplicates})
    }

    fn miss(&self, what: String) {
        self.missing_count.set(self.missing_count.get() + 1);
        let mut s = self.missing_samples.borrow_mut();
        if s.len() < 32 && !s.contains(&what) {
            s.push(what);
        }
    }

    /// Re-derive, from `gate_bits` alone, every operand the CPU twin's `silu_mul` DAG builds before each
    /// intrinsic, and compare with what the probe recorded (this validates the probe table against the DAG; it
    /// does not model any device result). Entries whose operands are non-finite are skipped and counted:
    /// their NaN payload/sign is hardware behaviour, resolved only by exact-key hook lookup.
    pub fn dag_consistency(&self) -> Value {
        let neg_log2e = f32::from_bits(0xbfb8_aa3b);
        let neg_log2e_lo = f32::from_bits(0xb2a5_705f);
        let (mut checked, mut bad, mut skipped) = (0usize, 0usize, 0usize);
        let mut first: Option<Value> = None;
        for e in &self.entries {
            let g = f32::from_bits(e.divide_num);
            let ph = neg_log2e * g;
            let mut pl = neg_log2e.mul_add(g, -ph);
            pl = neg_log2e_lo.mul_add(g, pl);
            let ee = ph.round_ties_even();
            let a = (ph - ee) + pl;
            if !g.is_finite() || !a.is_finite() || !ph.is_finite() {
                skipped += 1;
                continue;
            }
            checked += 1;
            let mut r = f32::from_bits(e.ldexp_result);
            if f32::from_bits(0x42ce_8ed0) < g {
                r = 0.0;
            }
            if g < f32::from_bits(0xc2b1_7218) {
                r = f32::INFINITY;
            }
            let den = 1.0 + r;
            let mut why: Vec<&str> = Vec::new();
            if e.gate_bits != e.divide_num && (e.gate_bits << 16) != e.divide_num {
                why.push("gate_bits!=divide_num");
            }
            if e.exp2_arg != a.to_bits() {
                why.push("exp2_arg");
            }
            if e.ldexp_exponent != ee as i32 {
                why.push("ldexp_exponent");
            }
            if e.ldexp_arg != e.exp2_result {
                why.push("ldexp_arg!=exp2_result");
            }
            if e.divide_den != den.to_bits() {
                why.push("divide_den");
            }
            if !why.is_empty() {
                bad += 1;
                if first.is_none() {
                    first = Some(
                        json!({"gate_bits": format!("{:#010x}", e.gate_bits), "fields": why,
                        "recorded_exp2_arg": format!("{:#010x}", e.exp2_arg), "derived_exp2_arg": format!("{:#010x}", a.to_bits()),
                        "recorded_exponent": e.ldexp_exponent, "derived_exponent": ee as i32,
                        "recorded_den": format!("{:#010x}", e.divide_den), "derived_den": format!("{:#010x}", den.to_bits())}),
                    );
                }
            }
        }
        json!({"checked": checked, "mismatches": bad, "skipped_nonfinite": skipped, "first": first})
    }
}

impl EpilogueIntrinsics for QualifiedHooks {
    fn exp2(&self, x: f32) -> f32 {
        match self.exp2.get(&x.to_bits()) {
            Some(r) => {
                self.hits.set(self.hits.get() + 1);
                f32::from_bits(*r)
            }
            None => {
                self.miss(format!("exp2 arg={:#010x}", x.to_bits()));
                f32::NAN
            }
        }
    }
    fn ldexp(&self, x: f32, e: i32) -> f32 {
        match self.ldexp.get(&(x.to_bits(), e)) {
            Some(r) => {
                self.hits.set(self.hits.get() + 1);
                f32::from_bits(*r)
            }
            None => {
                self.miss(format!("ldexp arg={:#010x} exponent={e}", x.to_bits()));
                f32::NAN
            }
        }
    }
    fn divide(&self, numerator: f32, denominator: f32) -> f32 {
        match self
            .divide
            .get(&(numerator.to_bits(), denominator.to_bits()))
        {
            Some(r) => {
                self.hits.set(self.hits.get() + 1);
                f32::from_bits(*r)
            }
            None => {
                self.miss(format!(
                    "divide num={:#010x} den={:#010x}",
                    numerator.to_bits(),
                    denominator.to_bits()
                ));
                f32::NAN
            }
        }
    }
}

/// CPU-only hooks for `--self-check` (harness validation without a device). NEVER a device expectation.
#[derive(Default)]
pub struct CpuOnlySelfCheckHooks(PortableEpilogue);

impl EpilogueIntrinsics for CpuOnlySelfCheckHooks {
    fn exp2(&self, x: f32) -> f32 {
        self.0.exp2(x)
    }
    fn ldexp(&self, x: f32, e: i32) -> f32 {
        self.0.ldexp(x, e)
    }
    fn divide(&self, n: f32, d: f32) -> f32 {
        self.0.divide(n, d)
    }
}

/// Hook-set protocol checks that need no device: conflicting duplicates rejected, identical duplicates counted,
/// missing key => recorded error + NaN, exact-key hit returns the recorded bits.
pub fn protocol_selftest() -> Value {
    let good = |gate: u32, exp2_arg: u32| {
        json!({"gate_bits": gate, "exp2_arg": exp2_arg, "exp2_result": 0x3f80_0000u32, "ldexp_arg": 0x3f80_0000u32,
        "ldexp_exponent": 0, "ldexp_result": 0x3f80_0000u32, "divide_num": gate, "divide_den": 0x4000_0000u32, "divide_result": 0x3f00_0000u32})
    };
    let conflicting = {
        let mut a = good(0x3f80_0000, 0x1111_1111);
        let mut b = good(0x4000_0000, 0x1111_1111);
        b["exp2_result"] = json!(0x3f00_0000u32);
        a["divide_num"] = json!(0x3f80_0000u32);
        json!({"schema": SCHEMA, "observations": [a, b]})
    };
    let rejected_conflict = QualifiedHooks::from_json(&conflicting).is_err();
    let identical = json!({"schema": SCHEMA, "observations": [good(0x3f80_0000, 0x1111_1111), good(0x3f80_0000, 0x1111_1111)]});
    let h = QualifiedHooks::from_json(&identical);
    let accepted_identical = h
        .as_ref()
        .map(|h| h.identical_duplicates > 0)
        .unwrap_or(false);
    let (mut hit_exact, mut miss_recorded) = (false, false);
    if let Ok(h) = h {
        hit_exact =
            h.exp2(f32::from_bits(0x1111_1111)).to_bits() == 0x3f80_0000 && h.missing_count() == 0;
        let v = h.exp2(f32::from_bits(0x2222_2222));
        miss_recorded = v.is_nan() && h.missing_count() == 1 && h.missing_samples().len() == 1;
    }
    let wrong_schema_rejected =
        QualifiedHooks::from_json(&json!({"schema": "x", "observations": []})).is_err();
    json!({"conflicting_duplicate_rejected": rejected_conflict, "identical_duplicate_accepted": accepted_identical,
           "exact_key_hit": hit_exact, "missing_key_recorded_and_nan": miss_recorded, "wrong_schema_rejected": wrong_schema_rejected,
           "pass": rejected_conflict && accepted_identical && hit_exact && miss_recorded && wrong_schema_rejected})
}

pub const DIVIDE_SCHEMA: &str = "consumer24.gfx1100.divide-cases.v1";

impl QualifiedHooks {
    /// Merge exact arbitrary-operand divide observations (`divide_extras`) from the probe's supplementary
    /// document. Conflicts with existing keys are rejected; returns the number of new keys.
    pub fn merge_divide_extras(&mut self, doc: &Value) -> Result<usize, String> {
        let schema = doc
            .get("schema")
            .and_then(Value::as_str)
            .ok_or("divide table has no `schema` string")?;
        if schema != DIVIDE_SCHEMA {
            return Err(format!(
                "divide table schema {schema:?} != {DIVIDE_SCHEMA:?}"
            ));
        }
        let extras = doc
            .get("divide_extras")
            .and_then(Value::as_array)
            .ok_or("divide table has no `divide_extras` array")?;
        let before = self.divide.len();
        for (i, e) in extras.iter().enumerate() {
            let n = num_u32(e, "divide_num").map_err(|m| format!("divide_extras[{i}]: {m}"))?;
            let d = num_u32(e, "divide_den").map_err(|m| format!("divide_extras[{i}]: {m}"))?;
            let r = num_u32(e, "divide_result").map_err(|m| format!("divide_extras[{i}]: {m}"))?;
            insert(
                &mut self.divide,
                (n, d),
                r,
                "divide(extras)",
                &mut self.identical_duplicates,
            )?;
        }
        Ok(self.divide.len() - before)
    }
}

pub const INTRINSIC_OBS_SCHEMA: &str = "consumer24.gfx1100.intrinsic-observations.v1";
pub const INTRINSIC_KEYS_SCHEMA: &str = "consumer24.gfx1100.intrinsic-keys.v1";

impl QualifiedHooks {
    /// Exact-key lookups that neither count hits nor record misses (used by the CPU hook-key census, which records
    /// every query itself).
    pub fn get_exp2(&self, arg: u32) -> Option<u32> {
        self.exp2.get(&arg).copied()
    }
    pub fn get_ldexp(&self, arg: u32, exponent: i32) -> Option<u32> {
        self.ldexp.get(&(arg, exponent)).copied()
    }
    pub fn get_divide(&self, numerator: u32, denominator: u32) -> Option<u32> {
        self.divide.get(&(numerator, denominator)).copied()
    }

    /// Merge one generic exact-key intrinsic observation document (`consumer24.gfx1100.intrinsic-observations.v1`):
    /// `exp2:[{arg,result}]`, `ldexp:[{arg,exponent,result}]`, `divide:[{numerator,denominator,result}]`, each an
    /// independent operation observation. A key already known with a different result is an error; identical
    /// duplicates are counted. Device qualification is checked by the caller (`intrinsic_qualification`).
    /// Returns `{new_keys:{exp2,ldexp,divide}, identical_duplicates_in_this_merge, records:{...}}`.
    pub fn merge_intrinsic_observations(&mut self, doc: &Value) -> Result<Value, String> {
        let schema = doc
            .get("schema")
            .and_then(Value::as_str)
            .ok_or("intrinsic table has no `schema` string")?;
        if schema != INTRINSIC_OBS_SCHEMA {
            return Err(format!(
                "intrinsic table schema {schema:?} != {INTRINSIC_OBS_SCHEMA:?}"
            ));
        }
        let arr = |k: &str| {
            doc.get(k)
                .and_then(Value::as_array)
                .ok_or_else(|| format!("intrinsic table has no `{k}` array"))
        };
        let (e2, ld, dv) = (arr("exp2")?, arr("ldexp")?, arr("divide")?);
        let (b2, bl, bd, bdup) = (
            self.exp2.len(),
            self.ldexp.len(),
            self.divide.len(),
            self.identical_duplicates,
        );
        for (i, e) in e2.iter().enumerate() {
            let k = num_u32(e, "arg").map_err(|m| format!("exp2[{i}]: {m}"))?;
            let r = num_u32(e, "result").map_err(|m| format!("exp2[{i}]: {m}"))?;
            insert(
                &mut self.exp2,
                k,
                r,
                "exp2(intrinsic table)",
                &mut self.identical_duplicates,
            )?;
        }
        for (i, e) in ld.iter().enumerate() {
            let a = num_u32(e, "arg").map_err(|m| format!("ldexp[{i}]: {m}"))?;
            let x = num_i32(e, "exponent").map_err(|m| format!("ldexp[{i}]: {m}"))?;
            let r = num_u32(e, "result").map_err(|m| format!("ldexp[{i}]: {m}"))?;
            insert(
                &mut self.ldexp,
                (a, x),
                r,
                "ldexp(intrinsic table)",
                &mut self.identical_duplicates,
            )?;
        }
        for (i, e) in dv.iter().enumerate() {
            let n = num_u32(e, "numerator").map_err(|m| format!("divide[{i}]: {m}"))?;
            let d = num_u32(e, "denominator").map_err(|m| format!("divide[{i}]: {m}"))?;
            let r = num_u32(e, "result").map_err(|m| format!("divide[{i}]: {m}"))?;
            insert(
                &mut self.divide,
                (n, d),
                r,
                "divide(intrinsic table)",
                &mut self.identical_duplicates,
            )?;
        }
        Ok(
            json!({"new_keys": {"exp2": self.exp2.len() - b2, "ldexp": self.ldexp.len() - bl, "divide": self.divide.len() - bd},
                  "identical_duplicates_in_this_merge": self.identical_duplicates - bdup,
                  "records": {"exp2": e2.len(), "ldexp": ld.len(), "divide": dv.len()}}),
        )
    }
}

/// Qualification problems of an intrinsic observation document: schema, passing selfcheck, the expected device
/// identity (arch, uuid / bdf when pinned, `identity_match`) and `FLOAT_DENORM_MODE_32 = FLOAT_DENORM_MODE_16_64 = 3`
/// (`probe.denorm`, else top-level `denorm`). gfx1100 pins the xtx uuid and bdf; gfx1151 has no ROCr UUID, so its uuid is `None`.
pub fn intrinsic_qualification(
    doc: &Value,
    arch: &str,
    uuid: Option<&str>,
    bdf: Option<&str>,
) -> Vec<String> {
    let mut p = Vec::new();
    if doc["schema"] != json!(INTRINSIC_OBS_SCHEMA) {
        p.push(format!("schema is not {INTRINSIC_OBS_SCHEMA}"));
    }
    if doc["selfcheck"]["ok"] != json!(true) {
        p.push("selfcheck.ok is not true".into());
    }
    let d = &doc["device"];
    if d["arch"] != json!(arch)
        || uuid.is_some_and(|u| d["uuid"] != json!(u))
        || bdf.is_some_and(|b| d["bdf"] != json!(b))
        || d["identity_match"] != json!(true)
    {
        p.push(format!("device is not the expected {arch} (uuid {uuid:?}, bdf {bdf:?}) with identity_match true: {d}"));
    }
    let den = if doc["probe"]["denorm"].is_object() {
        &doc["probe"]["denorm"]
    } else {
        &doc["denorm"]
    };
    if den["float_denorm_mode_32"] != json!(3) || den["float_denorm_mode_16_64"] != json!(3) {
        p.push(format!(
            "denorm modes are not FLOAT_DENORM_MODE_32 = FLOAT_DENORM_MODE_16_64 = 3: {den}"
        ));
    }
    p
}
