// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! X1 lift-and-repass prototype (PM-KRSI-1100 LiftProbe): lift a hipcc code object,
//! apply ONE parameterised transform to ONE kernel through the checked edit API
//! (`Analyzed::edit`), re-emit, validate, write the new object.
//!
//! ```text
//! x1_repass IN.hsaco --kernel SYMBOL --transform T --out OUT.co [--report R.json]
//! T = identity                 no edit; the output MUST equal the input bytes
//!   | strip-delay              remove every s_delay_alu of the kernel
//!   | builder-delay[:W]        strip, then re-place hints with peacemaker-isa's
//!                              `DelayAlu` policy (hazard.rs; window W VALUs, default 4;
//!                              one instid0 hint per dependent VALU; reset at block leaders)
//!   | wait-probe               report (no write) which s_waitcnt counters the replay lets
//!                              us loosen by +1 (`Edit::SetWait`, checked)
//! ```
//! Validation (all hard failures): `emit::checked` (the emitted bytes lift again to the same
//! kernels, instruction streams, metadata and layout-derived descriptors), every OTHER kernel
//! keeps its exact `.text`, descriptor resource fields (everything but the entry offset
//! and the layout-derived INST_PREF_SIZE) and metadata equal the input's, M7's analysis
//! obligations (`edit::analyze`) are no worse than the input's, no ambiguous delay hint.

use peacemaker_ir::cfg::InstId;
use peacemaker_ir::edit::{analyze, Edit, InstRange};
use peacemaker_ir::effects::Control;
use peacemaker_ir::inst::{Abi, Arch, Form, Frontend, Kernel, KernelOrigin, Program, SymbolId};
use peacemaker_ir::passes::windows::check_windows;
use peacemaker_ir::reg::{Kind, RegRef};
use peacemaker_ir::state::Analyzed;
use peacemaker_lift::rewrite::{apply, EquivalenceKind, Rewrite};
use peacemaker_lift::text::parse_line;
use peacemaker_lift::{emit, lift_object, Options};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

type R<T> = Result<T, Box<dyn std::error::Error>>;
const OPTS: Options = Options {
    frontend: Frontend::Hipcc,
};

fn hex(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}

fn rawhex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn kernel<'a>(p: &'a Program, s: &SymbolId) -> &'a Kernel {
    p.kernels.iter().find(|k| &k.symbol == s).expect("kernel")
}

fn is_valu(form: Form) -> bool {
    matches!(
        form,
        Form::Vop1
            | Form::Vop2
            | Form::Vop3
            | Form::Vop3p
            | Form::Vopd
            | Form::Vopc
            | Form::Vinterp
    )
}

fn delay_ids(k: &Kernel) -> Vec<InstId> {
    k.body
        .layout
        .iter()
        .copied()
        .filter(|id| {
            k.body
                .insts
                .get(*id)
                .is_some_and(|i| i.effects.control == Control::Delay)
        })
        .collect()
}

fn census(k: &Kernel, arch: Arch) -> BTreeMap<&'static str, usize> {
    let mut m = BTreeMap::new();
    for id in &k.body.layout {
        let i = k.body.insts.get(*id).unwrap();
        let n = i.op.name(arch).unwrap_or("?");
        let key = match n {
            "s_delay_alu" => "s_delay_alu",
            "s_clause" => "s_clause",
            "s_barrier" => "s_barrier",
            "s_waitcnt" => "s_waitcnt",
            _ if n.starts_with("v_dual_") => "vopd",
            _ if n.starts_with("v_wmma") => "wmma",
            _ => "other",
        };
        *m.entry(key).or_default() += 1;
    }
    m.remove("other");
    m
}

/// `peacemaker-isa::hazard::DelayAlu::step`, on lifted instructions (same policy and text).
struct BuilderDelay {
    window: usize,
    recent: Vec<(Vec<RegRef>, bool)>,
}
impl BuilderDelay {
    const TRANS: [&'static str; 7] = [
        "v_rcp_", "v_sqrt_", "v_rsq_", "v_sin_", "v_cos_", "v_exp_", "v_log_",
    ];
    fn step(&mut self, mnemonic: &str, uses: &[RegRef], defs: &[RegRef]) -> Option<String> {
        let (mut valu, mut trans, mut trans_seen) = (None, None, 0usize);
        for (distance, (written, is_trans)) in self
            .recent
            .iter()
            .rev()
            .enumerate()
            .map(|(i, r)| (i + 1, r))
        {
            if *is_trans {
                trans_seen += 1
            }
            if !written
                .iter()
                .any(|w| uses.iter().any(|u| u.kind == Kind::V && w.overlaps(*u)))
            {
                continue;
            }
            if *is_trans && trans_seen <= 3 {
                trans.get_or_insert(trans_seen);
            } else if !*is_trans {
                valu.get_or_insert(distance);
            }
        }
        let trans_insn = Self::TRANS.iter().any(|p| mnemonic.starts_with(p));
        self.recent.push((
            defs.iter().copied().filter(|r| r.kind == Kind::V).collect(),
            trans_insn,
        ));
        if self.recent.len() > self.window {
            self.recent.remove(0);
        }
        match (valu, trans) {
            (None, None) => None,
            (Some(v), None) => Some(format!("s_delay_alu instid0(VALU_DEP_{v})")),
            (None, Some(t)) => Some(format!("s_delay_alu instid0(TRANS32_DEP_{t})")),
            (Some(v), Some(t)) => Some(format!(
                "s_delay_alu instid0(VALU_DEP_{v}) | instid1(TRANS32_DEP_{t})"
            )),
        }
    }
}

/// One batch removing every `s_delay_alu` of the kernel (`None` when it has none).
fn strip_edit(k: &Kernel) -> Option<Edit> {
    let ids = delay_ids(k);
    (!ids.is_empty()).then(|| {
        Edit::Batch(
            ids.iter()
                .map(|id| Edit::Remove {
                    range: InstRange::single(*id),
                })
                .collect(),
        )
    })
}

fn builder_delay(a: &Analyzed<Program>, sym: &SymbolId, window: usize) -> R<(Edit, usize, usize)> {
    let arch = a.program.target.arch;
    let k = kernel(&a.program, sym);
    let mut inserts: Vec<(InstId, String)> = Vec::new();
    for block in &k.body.blocks {
        let mut policy = BuilderDelay {
            window,
            recent: Vec::new(),
        };
        for pos in block.range.0..block.range.1 {
            let id = k.body.layout[pos];
            let inst = k.body.insts.get(id).unwrap();
            if !is_valu(inst.form) {
                continue;
            }
            let name = inst.op.name(arch).unwrap_or("");
            if let Some(text) = policy.step(name, &inst.effects.uses, &inst.effects.defs) {
                inserts.push((id, text));
            }
        }
    }
    // An inserted `s_delay_alu` window must end inside the inserted sequence, so each
    // consumer is replaced by `[hint, consumer]` (the same instruction, a new arena id).
    // `Replace` re-proves definedness of the re-inserted consumer, so a consumer the analysis
    // already flags (a `definedness-*` obligation: hipcc reads ABI state the entry seed does
    // not model) cannot be re-inserted; it keeps no hint and is counted as skipped.
    use peacemaker_ir::state::ObligationKind;
    let mut flagged: std::collections::HashSet<InstId> = a
        .obligations
        .iter()
        .filter(|o| o.kind == ObligationKind::Definedness)
        .flat_map(|o| o.insts.iter().copied())
        .collect();
    // Check 11 treats a re-inserted consumer as new code: it may not read anything the
    // base already reads undefined (e.g. the preserved half of a true16 destination).
    flagged.extend(
        peacemaker_lift::rewrite::definedness::undefined_reads(k)
            .iter()
            .map(|u| u.inst),
    );
    let mut edits = Vec::new();
    let mut skipped = 0usize;
    for (consumer, text) in &inserts {
        if flagged.contains(consumer) {
            skipped += 1;
            continue;
        }
        let original = k.body.insts.get(*consumer).unwrap().clone();
        edits.push(Edit::Replace {
            range: InstRange::single(*consumer),
            insts: vec![parse_line(text, arch)?, original],
        });
    }
    let n = edits.len();
    Ok((Edit::Batch(edits), n, skipped))
}

/// Which waits the whole-CFG replay lets us loosen by one unit on one counter.
fn wait_probe(a: &Analyzed<Program>, sym: &SymbolId) -> R<Value> {
    use peacemaker_ir::wait::WaitImm;
    let k = kernel(&a.program, sym);
    let (mut tried, mut ok) = (0usize, 0usize);
    let mut why: BTreeMap<String, usize> = BTreeMap::new();
    let mut accepted = Vec::new();
    for id in &k.body.layout {
        let inst = k.body.insts.get(*id).unwrap();
        if inst.effects.control != Control::Wait {
            continue;
        }
        let Some(old) = inst.mods.wait.clone() else {
            continue;
        };
        for c in 0..old.per_counter.len() {
            let Some(n) = old.per_counter[c] else {
                continue;
            };
            let mut imm: WaitImm = old.clone();
            imm.per_counter[c] = Some(n + 1);
            tried += 1;
            match a.edit(sym, Edit::SetWait { inst: *id, imm }) {
                Ok(_) => {
                    ok += 1;
                    accepted
                        .push(json!({"pc": inst.prov.pc, "counter": c, "from": n, "to": n + 1}));
                }
                Err(e) => {
                    let e = e.to_string();
                    let key: String = e.chars().take(48).collect();
                    *why.entry(key).or_default() += 1;
                }
            }
        }
    }
    Ok(json!({"tried": tried, "accepted": ok, "accepted_list": accepted, "rejected_by": why}))
}

fn obligations(a: &Analyzed<Program>) -> BTreeMap<String, usize> {
    let mut m = BTreeMap::new();
    for o in &a.obligations {
        *m.entry(format!("{:?}/{}", o.kind, o.rule_id)).or_default() += 1;
    }
    m
}

fn main() -> R<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (mut input, mut sym, mut tf, mut out, mut report) = (None, None, None, None, None);
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--kernel" => {
                sym = Some(args[i + 1].clone());
                i += 2
            }
            "--transform" => {
                tf = Some(args[i + 1].clone());
                i += 2
            }
            "--out" => {
                out = Some(args[i + 1].clone());
                i += 2
            }
            "--report" => {
                report = Some(args[i + 1].clone());
                i += 2
            }
            _ => {
                input = Some(args[i].clone());
                i += 1
            }
        }
    }
    let usage = "usage: x1_repass IN.hsaco --kernel SYMBOL --transform identity|strip-delay|builder-delay[:W]|wait-probe [--out OUT.co] [--report R.json]";
    let (input, sym, tf) = (
        input.ok_or(usage)?,
        SymbolId(sym.ok_or(usage)?),
        tf.ok_or(usage)?,
    );
    let bytes = std::fs::read(&input)?;
    let lifted = lift_object(&bytes, OPTS)?;
    let arch = lifted.program.target.arch;
    let before = analyze(lifted.program.clone(), &sym)?;
    let k0 = kernel(&before.program, &sym);
    let text0 = emit::bytes(k0, arch)?;
    let win0 = check_windows(&k0.body)?;
    let mut rep = json!({
        "input": input, "input_sha256": hex(&bytes), "kernel": sym.0, "transform": tf,
        "before": {"text_sha256": hex(&text0), "text_bytes": text0.len(), "census": census(k0, arch),
                   "obligations": obligations(&before), "ambiguous_delays": win0.delays.iter().filter(|d| d.is_ambiguous()).count()},
    });

    if tf == "wait-probe" {
        rep["wait_probe"] = wait_probe(&before, &sym)?;
        println!("{}", serde_json::to_string_pretty(&rep)?);
        return Ok(());
    }

    // The script (core edits on this kernel) and the claim it makes.
    let (script, claim, note): (Vec<Edit>, EquivalenceKind, Value) = match tf.as_str() {
        "identity" => (Vec::new(), EquivalenceKind::ByteIdentical, json!({})),
        "strip-delay" => (
            strip_edit(k0).into_iter().collect(),
            EquivalenceKind::Unproved,
            json!({}),
        ),
        t if t == "builder-delay" || t.starts_with("builder-delay:") => {
            let w: usize = t.split_once(':').map_or(Ok(4), |(_, w)| w.parse())?;
            let mut script: Vec<Edit> = strip_edit(k0).into_iter().collect();
            let mut stripped = before.clone();
            for e in &script {
                stripped = stripped.edit(&sym, e.clone())?.0;
            }
            let (e, n, skipped) = builder_delay(&stripped, &sym, w)?;
            script.push(e);
            (
                script,
                EquivalenceKind::Unproved,
                json!({"window": w, "hints_inserted": n, "consumers_skipped_definedness": skipped}),
            )
        }
        other => return Err(format!("unknown transform {other}\n{usage}").into()),
    };
    // `rewrite::apply`: every step a checked transaction, re-layout, `emit::checked`, check 11.
    let cert = apply(
        &bytes,
        OPTS,
        Rewrite {
            kernel: sym.clone(),
            script,
            claim,
        },
    )?;
    let (after, emitted) = (cert.program.clone(), cert.bytes.clone());
    let back = lift_object(&emitted, OPTS)?;
    let k1 = kernel(&after.program, &sym);
    let text1 = emit::bytes(k1, arch)?;
    let win1 = check_windows(&k1.body)?;
    let ambiguous = win1.delays.iter().filter(|d| d.is_ambiguous()).count();

    // Other kernels: exact .text, entry VA may move only if the layout moved.
    let mut other_changed = Vec::new();
    for k in &lifted.program.kernels {
        if k.symbol == sym {
            continue;
        }
        let a = emit::bytes(k, arch)?;
        let b = emit::bytes(kernel(&back.program, &k.symbol), arch)?;
        if a != b {
            other_changed.push(k.symbol.0.clone());
        }
    }
    // Descriptor/metadata: all fields but the layout-derived entry offset / INST_PREF_SIZE.
    let (d0, m0) = match &k0.abi {
        Abi::Hsa {
            descriptor,
            metadata,
        } => (descriptor, metadata),
        _ => return Err("raw ABI".into()),
    };
    let kb = kernel(&back.program, &sym);
    let (d1, m1) = match &kb.abi {
        Abi::Hsa {
            descriptor,
            metadata,
        } => (descriptor, metadata),
        _ => return Err("raw ABI".into()),
    };
    let mut diffs = Vec::new();
    macro_rules! cmp {
        ($f:ident) => {
            if d0.$f != d1.$f {
                diffs.push(stringify!($f))
            }
        };
    }
    cmp!(group_segment_fixed_size);
    cmp!(private_segment_fixed_size);
    cmp!(kernarg_size);
    cmp!(compute_pgm_rsrc1);
    cmp!(compute_pgm_rsrc2);
    cmp!(kernel_code_properties);
    cmp!(kernarg_preload);
    if d0.compute_pgm_rsrc3 != d1.compute_pgm_rsrc3 {
        diffs.push("compute_pgm_rsrc3(INST_PREF_SIZE tracks code size)")
    }
    let meta_same = m0.parsed == m1.parsed;
    let entry = |p: &Program, s: &SymbolId| match kernel(p, s).origin {
        KernelOrigin::Frontend { entry_va, size, .. } => (entry_va, size),
        _ => (0, 0),
    };
    let after_obl = obligations(&after);
    let worse: Vec<_> = after_obl
        .iter()
        .filter(|(k, v)| **v > before_obl_get(&before, k))
        .collect();
    rep["after"] = json!({
        "text_sha256": hex(&text1), "text_bytes": text1.len(), "census": census(k1, arch),
        "obligations": after_obl, "ambiguous_delays": ambiguous,
        "entry_va_size_in": entry(&lifted.program, &sym), "entry_va_size_out": entry(&back.program, &sym),
    });
    rep["note"] = note;
    let r = &cert.receipt;
    rep["receipt"] = json!({
        "claim": format!("{:?}", r.equivalence.kind),
        "base_object_sha256": rawhex(&r.base_object_sha256), "object_sha256": rawhex(&r.object_sha256),
        "steps": r.script.len(), "claims": r.claims.len(), "delay_rewrites": r.delay_rewrites.len(),
        "reencoded_branches": r.reencoded_branches,
        "new_obligations": r.new_obligations.len(),
        "check11_definedness": match &r.definedness { Ok(()) => "ok".to_owned(), Err(e) => e.chars().take(300).collect() },
    });
    rep["validation"] = json!({
        "emit_checked": "ok (rewrite::apply ran emit::checked: relift to the same kernels, streams, metadata, descriptors)",
        "output_module_sha256": hex(&emitted), "output_equals_input": emitted == bytes,
        "other_kernels_text_changed": other_changed,
        "descriptor_fields_changed_vs_input": diffs,
        "metadata_equal": meta_same,
        "resources": {"max_vgpr": [before.facts.resources[0].max_vgpr, after.facts.resources[0].max_vgpr],
                      "max_sgpr": [before.facts.resources[0].max_sgpr, after.facts.resources[0].max_sgpr]},
        "obligations_worse": worse.iter().map(|(k, v)| format!("{k}: {v}")).collect::<Vec<_>>(),
    });
    let mut failures = Vec::new();
    if tf == "identity" && emitted != bytes {
        failures.push("identity transform changed the bytes".to_owned())
    }
    if !other_changed.is_empty() {
        failures.push(format!("other kernels changed: {other_changed:?}"))
    }
    if !meta_same {
        failures.push("metadata changed".into())
    }
    if let Err(e) = &r.definedness {
        failures.push(format!("check 11: {e}"))
    }
    if diffs.iter().any(|d| !d.starts_with("compute_pgm_rsrc3")) {
        failures.push(format!("descriptor fields changed: {diffs:?}"))
    }
    if !worse.is_empty() {
        failures.push("analysis obligations got worse".into())
    }
    let ambiguous_before = win0.delays.iter().filter(|d| d.is_ambiguous()).count();
    if ambiguous > ambiguous_before {
        failures.push(format!(
            "{ambiguous} ambiguous delay hints (input had {ambiguous_before})"
        ))
    }
    rep["failures"] = json!(failures);
    if let Some(p) = out {
        std::fs::write(&p, &emitted)?;
        rep["output"] = json!(p);
    }
    let text = serde_json::to_string_pretty(&rep)?;
    if let Some(p) = report {
        std::fs::write(p, &text)?;
    }
    println!("{text}");
    if failures.is_empty() {
        Ok(())
    } else {
        Err("validation failed".into())
    }
}

fn before_obl_get(a: &Analyzed<Program>, key: &str) -> usize {
    obligations(a).get(key).copied().unwrap_or(0)
}
