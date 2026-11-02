// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Runs the arbitrary-operand gfx1100 intrinsic-key probe (`gfx1100_intrinsic_keys_probe`,
//! peacemaker-isa example) on caller-supplied exact operation keys and writes the device result for each
//! (B4-C key closure). It measures `exp2` (`v_exp_f32`), `ldexp` (`v_ldexp_f32`), the complete copied
//! gemm_uk IEEE divide DAG and the final `Finish::MulUp` product (`v_mul_f32_e32 dst, q, up`) independently of each other and of any candidate K2 code; it is independent of
//! `gfx1100_epilogue_intrinsic_observe` and `gfx1100_divide_cases_observe`, whose code objects, manifests
//! and schemas it leaves alone.
//!
//! Generate: cargo run -p peacemaker-isa --release --features lab --example gfx1100_intrinsic_keys_probe -- --out-dir DIR
//! Build:    cargo build -p hipfire-runtime --release --features lab --example gfx1100_intrinsic_keys_observe
//! Run on the locked xtx (gfx1100) through the fabric scheduler:
//!   gfx1100_intrinsic_keys_observe --keys KEYS.json --codeobject DIR/gfx1100_intrinsic_keys_probe.co \
//!     --manifest DIR/gfx1100_intrinsic_keys_probe.manifest.json --card xtx \
//!     --expect-uuid GPU-43390a851e296ee5 [--expect-bdf 0000:67:00.0] --out OUT.json [--repeats 3]
//!
//! `KEYS.json` (schema `consumer24.gfx1100.intrinsic-keys.v1`) is
//! `{"schema": ..., "exp2": [u32], "ldexp": [{"value_bits": u32, "exponent": i32}],
//!   "divide": [{"numerator_bits": u32, "denominator_bits": u32}],
//!   "mul": [{"lhs_bits": u32, "rhs_bits": u32}]}`; each u32 is a JSON number or a string
//! (`"0x7fc12345"`, decimal); an exponent is a JSON number in the `i32` range or a string (signed decimal, or
//! `0x` hex of the 32-bit pattern). A `mul` key is an ordered pair (`lhs` = `q` in src0, `rhs` = `up` in vsrc1); the
//! swapped pair is a different key. `mul` is OPTIONAL: an absent `mul` means no mul keys and the output `mul` is empty.
//! The legacy arrays `exp2`/`ldexp`/`divide` MUST be present (a missing array is an invalid-size error, never an
//! empty default; an array may itself be empty) unless `mul` is present, in which case each missing legacy array is
//! empty (a mul-only batch). At least one key overall is required.
//! Every 32-bit pattern is allowed (NaN payloads, inf, zero, subnormal) and nothing is normalised.
//!
//! Output (schema `consumer24.gfx1100.intrinsic-observations.v1`): `exp2: [{arg, result}]`,
//! `ldexp: [{arg, exponent, result}]` (`arg` = value bits), `divide: [{numerator, denominator, result}]`,
//! `mul: [{lhs, rhs, result}]` (always present, empty when no mul keys were requested),
//! each unique key once in first-seen order, merged by KeyCensus; `device`, `probe`, `counts` (`listed`/`unique` per op,
//! mul included), `selfcheck`
//! (marker / echo / index / zero-word / repeat-run / duplicate-key consistency; failure exits 1 after the
//! file is written), `divide_trace` and `divide_host_ieee` (informational Rust `f32` division comparison; no host
//! comparison is made for mul, so no host NaN payload ever enters the record).
//! Refuses (exit 1, nothing launched) on `HSA_CU_MASK` / `ROC_GLOBAL_CU_MASK`, any manifest hash/ABI/denorm
//! disagreement, a non-gfx1100 device, a UUID/BDF mismatch, or `--card xtx` not resolving to the known xtx.

use hip_bridge::HipRuntime;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

type Result<T, E = String> = std::result::Result<T, E>;
fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}
fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

const KEYS_SCHEMA: &str = "consumer24.gfx1100.intrinsic-keys.v1";
const SCHEMA: &str = "consumer24.gfx1100.intrinsic-observations.v1";
const BUILD_SCHEMA: &str = "consumer24.gfx1100.intrinsic-keys-build.v1";
const ARCH: &str = "gfx1100";
const SYMBOL: &str = "c24_intrinsic_keys_probe_gfx1100";
const KERNARG: usize = 24;
const KEY_WORDS: usize = 8;
const KEY_BYTES: usize = KEY_WORDS * 4;
const WORDS: usize = 32;
const RECORD_BYTES: usize = WORDS * 4;
const MARKER: u32 = 0x4b45_5950;
const POISON: u8 = 0xa5;
const MAX_KEYS: usize = 1 << 22;
const XTX: (&str, &str, &str) = ("xtx", "GPU-43390a851e296ee5", "0000:67:00.0");
const LINK_CAVEAT: &str = "hipx xtx sits behind a PCIe 4.0 x4 root port (00:03.1); this probe measures arithmetic bits only, no speed or bandwidth";
const USAGE: &str = "usage: gfx1100_intrinsic_keys_observe --keys PATH --codeobject PATH --manifest PATH --card LABEL --expect-uuid UUID [--expect-bdf BDF] --out PATH [--repeats N]";
const DIVIDE_TRACE_FIELDS: [(&str, usize); 13] = [
    ("numerator_bits", 5),
    ("denominator_bits", 6),
    ("div_scale_den", 7),
    ("div_scale_num", 8),
    ("rcp", 9),
    ("fma_one_minus_d_r", 10),
    ("fmac_r_refine", 11),
    ("mul_q0", 12),
    ("fma_residual0", 13),
    ("fmac_q1", 14),
    ("fma_residual1", 15),
    ("div_fmas", 16),
    ("divide_result", 17),
];
static EMPTY_KEYS: Vec<Value> = Vec::new();
// Benign operands for padding lanes (1.0 / (1.0, 0) / (1.0, 1.0)); their records are checked then discarded.
const PAD_ONE: u32 = 0x3f80_0000;

type Rec = [u32; WORDS];

fn bits(v: &Value, what: &str) -> Result<u32> {
    let n = match v {
        Value::Number(n) => n
            .as_u64()
            .ok_or(format!("{what}: not an unsigned integer"))?,
        Value::String(s) => {
            let s = s.trim();
            let parsed = match s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
                Some(h) => u64::from_str_radix(h, 16),
                None => s.parse::<u64>(),
            };
            parsed.map_err(|e| format!("{what}: {e}"))?
        }
        _ => return Err(format!("{what}: expected number or string")),
    };
    u32::try_from(n).map_err(|_| format!("{what}: {n:#x} exceeds 32 bits"))
}

fn exponent(v: &Value, what: &str) -> Result<i32> {
    match v {
        Value::Number(n) => i32::try_from(n.as_i64().ok_or(format!("{what}: not an integer"))?)
            .map_err(|_| format!("{what}: outside the i32 range")),
        Value::String(s) => {
            let s = s.trim();
            match s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
                Some(h) => u32::from_str_radix(h, 16)
                    .map(|u| u as i32)
                    .map_err(|e| format!("{what}: {e}")),
                None => s.parse::<i32>().map_err(|e| format!("{what}: {e}")),
            }
        }
        _ => Err(format!("{what}: expected number or string")),
    }
}

fn array<'a>(doc: &'a Value, key: &str) -> Result<&'a Vec<Value>> {
    doc.get(key)
        .and_then(Value::as_array)
        .ok_or(format!("keys: invalid size: missing or non-array `{key}`"))
}

/// A legacy op array: required, unless the document carries `mul` (a mul-only batch), in which case a missing one is empty.
fn legacy_array<'a>(doc: &'a Value, has_mul: bool, key: &str) -> Result<&'a Vec<Value>> {
    if has_mul && doc.get(key).is_none() {
        Ok(&EMPTY_KEYS)
    } else {
        array(doc, key)
    }
}

/// Order-preserving unique keys plus, per listed key, the unique-slot it maps to.
fn uniq<K: std::hash::Hash + Eq + Copy>(listed: &[K]) -> (Vec<K>, Vec<usize>) {
    let mut slot: HashMap<K, usize> = HashMap::new();
    let (mut unique, mut map) = (Vec::new(), Vec::with_capacity(listed.len()));
    for &k in listed {
        let s = *slot.entry(k).or_insert_with(|| {
            unique.push(k);
            unique.len() - 1
        });
        map.push(s);
    }
    (unique, map)
}

fn run() -> Result<()> {
    let (
        mut keys_path,
        mut codeobject,
        mut manifest_path,
        mut card,
        mut expect_uuid,
        mut expect_bdf,
        mut out_path,
    ) = (None, None, None, None, None, None, None);
    let mut repeats = 3usize;
    let argv: Vec<String> = std::env::args().collect();
    let mut args = argv.iter().skip(1);
    while let Some(flag) = args.next() {
        let value = args.next().ok_or(USAGE)?.clone();
        match flag.as_str() {
            "--keys" => keys_path = Some(value),
            "--codeobject" => codeobject = Some(value),
            "--manifest" => manifest_path = Some(value),
            "--card" => card = Some(value),
            "--expect-uuid" => expect_uuid = Some(value),
            "--expect-bdf" => expect_bdf = Some(value.to_ascii_lowercase()),
            "--out" => out_path = Some(value),
            "--repeats" => repeats = value.parse().map_err(err)?,
            _ => return Err(USAGE.into()),
        }
    }
    let (keys_path, codeobject, manifest_path, card, expect_uuid, out_path) = (
        keys_path.ok_or(USAGE)?,
        codeobject.ok_or(USAGE)?,
        manifest_path.ok_or(USAGE)?,
        card.ok_or(USAGE)?,
        expect_uuid.ok_or(USAGE)?,
        out_path.ok_or(USAGE)?,
    );
    if repeats == 0 {
        return Err("--repeats must be at least 1".into());
    }
    for var in ["HSA_CU_MASK", "ROC_GLOBAL_CU_MASK"] {
        if std::env::var_os(var).is_some() {
            return Err(format!(
                "{var} is set: CU-mask variables are forbidden on RDNA devices"
            ));
        }
    }

    // Inputs and bindings.
    let image = std::fs::read(&codeobject).map_err(|e| format!("{codeobject}: {e}"))?;
    let manifest_bytes =
        std::fs::read(&manifest_path).map_err(|e| format!("{manifest_path}: {e}"))?;
    let manifest: Value = serde_json::from_slice(&manifest_bytes).map_err(err)?;
    let m_str = |k: &str| manifest[k].as_str().ok_or(format!("manifest: missing {k}"));
    let m_u64 = |k: &str| manifest[k].as_u64().ok_or(format!("manifest: missing {k}"));
    if m_str("schema")? != BUILD_SCHEMA
        || m_str("probe_schema")? != SCHEMA
        || m_str("keys_schema")? != KEYS_SCHEMA
    {
        return Err("manifest: schema mismatch".into());
    }
    if m_str("symbol")? != SYMBOL || m_str("arch")? != ARCH {
        return Err("manifest: symbol/arch mismatch".into());
    }
    if m_u64("kernarg_bytes")? as usize != KERNARG
        || m_u64("key_words")? as usize != KEY_WORDS
        || m_u64("record_words")? as usize != WORDS
        || m_u64("marker")? as u32 != MARKER
    {
        return Err("manifest: kernarg/key/record ABI mismatch".into());
    }
    if manifest["record_fields"][22].as_str() != Some("mul_result")
        || manifest["record_fields"][20].as_str() != Some("mul_lhs")
        || manifest["record_fields"][21].as_str() != Some("mul_rhs")
    {
        return Err(
            "manifest: record layout lacks the mul words 20..22 (regenerate the probe)".into(),
        );
    }
    if manifest["denorm"]["float_denorm_mode_32"].as_u64() != Some(3)
        || manifest["denorm"]["float_denorm_mode_16_64"].as_u64() != Some(3)
    {
        return Err("manifest: denorm modes are not 3/3".into());
    }
    let co_sha = sha(&image);
    if m_str("co_sha256")? != co_sha {
        return Err(format!(
            "code object sha256 {co_sha} differs from the manifest's {}",
            m_str("co_sha256")?
        ));
    }

    let keys_bytes = std::fs::read(&keys_path).map_err(|e| format!("{keys_path}: {e}"))?;
    let keys_doc: Value = serde_json::from_slice(&keys_bytes).map_err(err)?;
    if keys_doc["schema"].as_str() != Some(KEYS_SCHEMA) {
        return Err(format!("keys: schema must be {KEYS_SCHEMA}"));
    }
    let has_mul = keys_doc.get("mul").is_some();
    let (l_exp2, l_ldexp, l_divide) = (
        legacy_array(&keys_doc, has_mul, "exp2")?,
        legacy_array(&keys_doc, has_mul, "ldexp")?,
        legacy_array(&keys_doc, has_mul, "divide")?,
    );
    let l_mul: &Vec<Value> = if has_mul {
        array(&keys_doc, "mul")?
    } else {
        &EMPTY_KEYS
    };
    let mul_keys: Vec<(u32, u32)> = l_mul
        .iter()
        .enumerate()
        .map(|(i, k)| -> Result<(u32, u32)> {
            Ok((
                bits(&k["lhs_bits"], &format!("mul[{i}].lhs_bits"))?,
                bits(&k["rhs_bits"], &format!("mul[{i}].rhs_bits"))?,
            ))
        })
        .collect::<Result<_>>()?;
    let exp2_keys: Vec<u32> = l_exp2
        .iter()
        .enumerate()
        .map(|(i, k)| bits(k, &format!("exp2[{i}]")))
        .collect::<Result<_>>()?;
    let ldexp_keys: Vec<(u32, i32)> = l_ldexp
        .iter()
        .enumerate()
        .map(|(i, k)| -> Result<(u32, i32)> {
            Ok((
                bits(&k["value_bits"], &format!("ldexp[{i}].value_bits"))?,
                exponent(&k["exponent"], &format!("ldexp[{i}].exponent"))?,
            ))
        })
        .collect::<Result<_>>()?;
    let divide_keys: Vec<(u32, u32)> = l_divide
        .iter()
        .enumerate()
        .map(|(i, k)| -> Result<(u32, u32)> {
            Ok((
                bits(&k["numerator_bits"], &format!("divide[{i}].numerator_bits"))?,
                bits(
                    &k["denominator_bits"],
                    &format!("divide[{i}].denominator_bits"),
                )?,
            ))
        })
        .collect::<Result<_>>()?;
    let longest = exp2_keys
        .len()
        .max(ldexp_keys.len())
        .max(divide_keys.len())
        .max(mul_keys.len());
    if longest == 0 {
        return Err("keys: invalid size: no exp2, ldexp, divide or mul keys".into());
    }
    if longest > MAX_KEYS {
        return Err(format!(
            "keys: invalid size: {longest} keys in one operation exceed {MAX_KEYS}"
        ));
    }

    // Device identity.
    let hip = HipRuntime::load().map_err(err)?;
    let id = hip.current_device().map_err(err)?;
    let uuid = hip.device_uuid(id).map_err(err)?;
    let bdf = hip.device_pci_bus_id(id).map_err(err)?.to_ascii_lowercase();
    let arch = hip.get_arch(id).map_err(err)?;
    let card_known = card != XTX.0 || (uuid == XTX.1 && bdf == XTX.2);
    let identity_match = arch == ARCH
        && uuid == expect_uuid
        && expect_bdf.as_deref().is_none_or(|b| b == bdf)
        && card_known;
    let hip_runtime = hip
        .runtime_version()
        .map(|(a, b)| format!("{a}.{b}"))
        .map_err(err)?;
    let amdgpu_module = std::fs::read_to_string("/sys/module/amdgpu/version")
        .ok()
        .map(|s| s.trim().to_string());
    let device = json!({"card": card, "uuid": uuid, "bdf": bdf, "arch": arch, "expected_arch": ARCH, "expected_uuid": expect_uuid,
                        "expected_bdf": expect_bdf, "card_known_identity_checked": card == XTX.0, "identity_match": identity_match,
                        "hip_runtime": hip_runtime, "amdgpu_module_version": amdgpu_module});
    if !identity_match {
        println!(
            "{}",
            json!({"schema": SCHEMA, "device": device, "error": "gfx1100 intrinsic-keys probe device identity mismatch"})
        );
        return Err("gfx1100 intrinsic-keys probe device identity mismatch".into());
    }

    // Launch: each list padded to a multiple of 32 indices with benign operands.
    let n = longest.div_ceil(32) * 32;
    let mut padded = vec![[0u32; KEY_WORDS]; n];
    for (i, p) in padded.iter_mut().enumerate() {
        let (lv, le) = ldexp_keys
            .get(i)
            .map_or((PAD_ONE, 0), |&(v, e)| (v, e as u32));
        let (dn, dd) = divide_keys.get(i).copied().unwrap_or((PAD_ONE, PAD_ONE));
        let (ml, mr) = mul_keys.get(i).copied().unwrap_or((PAD_ONE, PAD_ONE));
        *p = [
            exp2_keys.get(i).copied().unwrap_or(PAD_ONE),
            lv,
            le,
            dn,
            dd,
            ml,
            mr,
            0,
        ];
    }
    let key_bytes: Vec<u8> = padded
        .iter()
        .flatten()
        .flat_map(|w| w.to_le_bytes())
        .collect();
    let module = hip.module_load_data(&image).map_err(err)?;
    let f = hip.module_get_function(&module, SYMBOL).map_err(err)?;
    let (dkeys, dout) = (
        hip.malloc(n * KEY_BYTES).map_err(err)?,
        hip.malloc(n * RECORD_BYTES).map_err(err)?,
    );
    hip.memcpy_htod(&dkeys, &key_bytes).map_err(err)?;
    let mut runs: Vec<Vec<u8>> = Vec::new();
    for _ in 0..repeats {
        hip.memset(&dout, i32::from(POISON), n * RECORD_BYTES)
            .map_err(err)?;
        let mut kernarg = [0u8; KERNARG];
        kernarg[0..8].copy_from_slice(&(dkeys.as_ptr() as u64).to_le_bytes());
        kernarg[8..16].copy_from_slice(&(dout.as_ptr() as u64).to_le_bytes());
        kernarg[16..20].copy_from_slice(&(n as u32).to_le_bytes());
        // SAFETY: exact 24-byte ABI (`keys, out, n`); both allocations outlive the synchronize.
        unsafe {
            hip.launch_kernel_blob(
                &f,
                [(n / 32) as u32, 1, 1],
                [32, 1, 1],
                0,
                None,
                &mut kernarg,
            )
        }
        .map_err(err)?;
        hip.device_synchronize().map_err(err)?;
        let mut bytes = vec![0u8; n * RECORD_BYTES];
        hip.memcpy_dtoh(&mut bytes, &dout).map_err(err)?;
        runs.push(bytes);
    }
    let recs: Vec<Rec> = runs[0]
        .chunks_exact(RECORD_BYTES)
        .map(|c| {
            let mut r = [0u32; WORDS];
            for (i, w) in c.chunks_exact(4).enumerate() {
                r[i] = u32::from_le_bytes(w.try_into().unwrap());
            }
            r
        })
        .collect();

    // Self-checks over every lane, padding included.
    let repeat_identical = runs.iter().all(|r| *r == runs[0]);
    let (mut marker_missing, mut echo_mismatch, mut index_mismatch, mut zero_words_nonzero) =
        (0u64, 0u64, 0u64, 0u64);
    let mut first_bad: Vec<Value> = Vec::new();
    let note = |first_bad: &mut Vec<Value>, i: usize, problem: &str| {
        if first_bad.len() < 16 {
            first_bad.push(json!({"index": i, "problem": problem}));
        }
    };
    for (i, r) in recs.iter().enumerate() {
        if r[31] != MARKER {
            marker_missing += 1;
            note(&mut first_bad, i, "marker");
        }
        if r[19] != i as u32 {
            index_mismatch += 1;
            note(&mut first_bad, i, "index");
        }
        if r[23..31].iter().any(|&w| w != 0) {
            zero_words_nonzero += 1;
            note(&mut first_bad, i, "zero_words");
        }
        let k = &padded[i];
        if (r[0], r[2], r[3], r[5], r[6], r[20], r[21])
            != (k[0], k[1], k[2], k[3], k[4], k[5], k[6])
        {
            echo_mismatch += 1;
            note(&mut first_bad, i, "echo");
        }
    }

    // Results per listed key (input order, duplicates included), then unique output with conflict check.
    let (ux, mx) = uniq(&exp2_keys);
    let (ul, ml) = uniq(&ldexp_keys);
    let (ud, md) = uniq(&divide_keys);
    let (um, mm) = uniq(&mul_keys);
    let (mut exp2_slot, mut ldexp_slot, mut divide_slot, mut mul_slot) = (
        vec![None::<usize>; ux.len()],
        vec![None::<usize>; ul.len()],
        vec![None::<usize>; ud.len()],
        vec![None::<usize>; um.len()],
    );
    let (mut dup_exp2, mut dup_ldexp, mut dup_divide, mut dup_mul) = (0u64, 0u64, 0u64, 0u64);
    let (mut conflict_exp2, mut conflict_ldexp, mut conflict_divide, mut conflict_mul) =
        (0u64, 0u64, 0u64, 0u64);
    for (i, &s) in mx.iter().enumerate() {
        match exp2_slot[s] {
            Some(j) => {
                dup_exp2 += 1;
                if recs[i][1] != recs[j][1] {
                    conflict_exp2 += 1;
                }
            }
            None => exp2_slot[s] = Some(i),
        }
    }
    for (i, &s) in ml.iter().enumerate() {
        match ldexp_slot[s] {
            Some(j) => {
                dup_ldexp += 1;
                if recs[i][4] != recs[j][4] {
                    conflict_ldexp += 1;
                }
            }
            None => ldexp_slot[s] = Some(i),
        }
    }
    for (i, &s) in md.iter().enumerate() {
        match divide_slot[s] {
            Some(j) => {
                dup_divide += 1;
                if recs[i][17] != recs[j][17] {
                    conflict_divide += 1;
                }
            }
            None => divide_slot[s] = Some(i),
        }
    }
    for (i, &s) in mm.iter().enumerate() {
        match mul_slot[s] {
            Some(j) => {
                dup_mul += 1;
                if recs[i][22] != recs[j][22] {
                    conflict_mul += 1;
                }
            }
            None => mul_slot[s] = Some(i),
        }
    }
    let exp2_out: Vec<Value> = exp2_slot
        .iter()
        .map(|s| {
            let r = &recs[s.unwrap()];
            json!({"arg": r[0], "result": r[1]})
        })
        .collect();
    let ldexp_out: Vec<Value> = ldexp_slot
        .iter()
        .map(|s| {
            let r = &recs[s.unwrap()];
            json!({"arg": r[2], "exponent": r[3] as i32, "result": r[4]})
        })
        .collect();
    let mul_out: Vec<Value> = mul_slot
        .iter()
        .map(|s| {
            let r = &recs[s.unwrap()];
            json!({"lhs": r[20], "rhs": r[21], "result": r[22]})
        })
        .collect();
    let mut divide_out = Vec::with_capacity(ud.len());
    let mut divide_trace = Vec::with_capacity(ud.len());
    let mut host_differing = Vec::new();
    for s in &divide_slot {
        let r = &recs[s.unwrap()];
        divide_out.push(json!({"numerator": r[5], "denominator": r[6], "result": r[17]}));
        let mut m = Map::new();
        for (name, w) in DIVIDE_TRACE_FIELDS {
            m.insert(name.into(), json!(r[w]));
        }
        m.insert("numerator_scale_carry".into(), json!(r[18]));
        divide_trace.push(Value::Object(m));
        let host = (f32::from_bits(r[5]) / f32::from_bits(r[6])).to_bits();
        if host != r[17] {
            host_differing.push(json!({"numerator": r[5], "denominator": r[6], "device_result": r[17], "host_ieee_result": host}));
        }
    }
    let records_sha = sha(&runs[0]);
    let dups_consistent =
        conflict_exp2 == 0 && conflict_ldexp == 0 && conflict_divide == 0 && conflict_mul == 0;
    let checks_ok = marker_missing == 0
        && echo_mismatch == 0
        && index_mismatch == 0
        && zero_words_nonzero == 0
        && repeat_identical
        && dups_consistent;
    let selfcheck = json!({
        "ok": checks_ok, "launches": repeats, "repeat_identical": repeat_identical, "marker_missing": marker_missing,
        "echo_mismatch": echo_mismatch, "index_mismatch": index_mismatch, "zero_words_nonzero": zero_words_nonzero,
        "duplicates": {"exp2": dup_exp2, "ldexp": dup_ldexp, "divide": dup_divide, "mul": dup_mul},
        "duplicates_inconsistent": {"exp2": conflict_exp2, "ldexp": conflict_ldexp, "divide": conflict_divide, "mul": conflict_mul},
        "first_problems": first_bad,
    });
    let counts = json!({
        "listed": {"exp2": exp2_keys.len(), "ldexp": ldexp_keys.len(), "divide": divide_keys.len(), "mul": mul_keys.len()},
        "unique": {"exp2": ux.len(), "ldexp": ul.len(), "divide": ud.len(), "mul": um.len()}, "padded_indices": n,
    });
    let doc = json!({
        "schema": SCHEMA,
        "exp2": exp2_out, "ldexp": ldexp_out, "divide": divide_out, "mul": mul_out,
        "counts": counts,
        "divide_trace": divide_trace,
        "divide_host_ieee": {"unique_keys": ud.len(), "differs": host_differing.len(), "differing_keys": host_differing,
                             "host_semantics": "Rust f32 division (correctly rounded, round-to-nearest-even); NaN payloads follow the host; informational only"},
        "device": device,
        "probe": {"symbol": SYMBOL, "codeobject": codeobject, "co_sha256": co_sha, "manifest": manifest_path, "manifest_sha256": sha(&manifest_bytes),
                  "keys": keys_path, "keys_file_sha256": sha(&keys_bytes), "kernarg_bytes": KERNARG, "key_words": KEY_WORDS, "record_words": WORDS,
                  "grid_x": n / 32, "block_x": 32, "padded_indices": n, "denorm": manifest["denorm"],
                  "dag_source": manifest["dag_source"], "dag_source_sha256": manifest["dag_source_sha256"], "listing_instructions": manifest["instructions"],
                  "hip_runtime": hip_runtime, "amdgpu_module_version": amdgpu_module},
        "records_sha256": records_sha,
        "selfcheck": selfcheck,
        "scope": "gfx1100 exp2 / ldexp / divide-DAG / MulUp-product bits for the supplied exact operation keys only; no candidate K2 code participated",
        "link_caveat": LINK_CAVEAT,
        "command": argv.join(" "),
    });
    let out_bytes = serde_json::to_vec(&doc).map_err(err)?;
    std::fs::write(&out_path, &out_bytes).map_err(|e| format!("{out_path}: {e}"))?;
    println!(
        "{}",
        json!({
            "schema": SCHEMA, "out": out_path, "out_sha256": sha(&out_bytes), "co_sha256": co_sha, "manifest_sha256": sha(&manifest_bytes),
            "keys_file_sha256": sha(&keys_bytes), "records_sha256": records_sha, "device": device, "counts": counts,
            "selfcheck_ok": checks_ok, "divide_host_ieee_differs": host_differing.len(), "link_caveat": LINK_CAVEAT,
        })
    );
    if !checks_ok {
        return Err(
            "intrinsic-keys probe self-checks failed (see selfcheck in the output file)".into(),
        );
    }
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("gfx1100_intrinsic_keys_observe: {e}");
        std::process::exit(1);
    }
}
