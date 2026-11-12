// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! CFG wait verification (`ledger_cfg`) of every committed pinned image and of the lifted
//! pool dispatchers, on the CPU.
//!
//! * `every_committed_image_verifies_or_is_an_explicit_allowlist_entry`: every
//!   `kernels/*.hxaco` and `kernels/*.co` is lifted (`peacemaker_lift::lift_object`), each
//!   kernel is printed back as label + canonical instruction lines from the IR body and
//!   verified with every line decoded through the lift's instruction table, so the check
//!   does not depend on the builder's metadata.
//! * `lifted_pools_...`: every gfx1201 kernel that `lift_region` converts is hosted by the
//!   existing dispatcher (`lifted_dispatch_gfx1201::pool`), alone and with every other
//!   converted kernel of its blockDim. Every pool must pass the dispatcher-only verifier
//!   (`verify_dispatcher_emitted`: any hazard outside the exact body spans is fatal, no
//!   allowance) and then the strict one (`verify_emitted`): a strict report is accepted only
//!   inside a body span and only through an exact `POOL_BODY` entry (module/symbol,
//!   body pc, category, instruction and need), never a scan-position template id.
//!
//! Every outcome lands in exactly one bucket: verified, finding (a real missing wait on a
//! feasible path) or unsupported (an input the verifier refuses rather than skips). A finding
//! or an unsupported input passes only through an exact per-target allowlist entry
//! (`allowlist.rs`: scope, target, error substring, and what is and is not established); an unlisted one fails the
//! test, and so does a listed one that no longer occurs. Nothing is counted verified unless
//! `ledger_cfg::verify` returned `Ok`.

use peacemaker_ir::inst::{Abi, Arch as IrArch, Frontend, Kernel, Program as IrProgram};
use peacemaker_isa::insn::{Instruction, Program};
use peacemaker_isa::kernels::lifted_dispatch_gfx1201 as ld;
use peacemaker_isa::ledger_cfg::{self, Stats};
use peacemaker_isa::{Arch, Emitted};
use peacemaker_lift::bundle::BundleCodec;
use peacemaker_lift::rewrite::region::{HipccObject, LiftedRegion, RegionError, RegionInputs};
use peacemaker_lift::{lift_object, Options};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[path = "ledger_cfg_pins/allowlist.rs"]
mod allowlist;
use allowlist::{
    Allow, Category, PoolStateCap, FINDINGS, POOL_BODY, POOL_STATE_CAP, STATE_CAP_ERROR,
    UNSUPPORTED,
};

/// gfx942 images: `lift_object` has no opcode table for them (they are emitted by the typed
/// `cdna` builder and encoded by `peacemaker_cdna`), so there is no instruction text to
/// verify here. They are counted, never verified.
const GFX942: &str = "gfx942";

/// The dispatcher store-source WAR once shared by every pool (`.Lnext`, `v_mov_b32_e32 v1,
/// 0`, `needs Store wait 0`) is FIXED by `1fc3fc059d` (the dispatcher backedge drains all
/// counters before `s_branch .Lnext`) and kept only as history in
/// `local://ledger-join-finding.md`; no pool allowlist entry for it remains.
/// Marker `ld::pool` puts in the error of an emission its fatal verifier rejected.
const POOL_VERIFY_FAILED: &str = "CFG wait verification failed";

struct Image {
    file: String,
    bytes: Vec<u8>,
    lifted: Result<IrProgram, String>,
}

/// Every committed image, lifted once; the builder frontend for the PM-emitted images.
fn scan() -> Vec<Image> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../kernels");
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("kernels dir")
        .flatten()
        .map(|e| e.path())
        .filter(|p| matches!(p.extension().and_then(|e| e.to_str()), Some("hxaco" | "co")))
        .collect();
    files.sort();
    files
        .into_iter()
        .map(|path| {
            let file = path.file_name().unwrap().to_string_lossy().into_owned();
            let bytes = std::fs::read(&path).expect("image");
            let frontend = if file.contains("_pm_") {
                Frontend::Builder
            } else {
                Frontend::Hipcc
            };
            let lifted = lift_object(&bytes, Options { frontend })
                .map(|l| l.program)
                .map_err(|e| e.to_string());
            Image {
                file,
                bytes,
                lifted,
            }
        })
        .collect()
}

fn isa_arch(arch: IrArch) -> Result<Arch, String> {
    match arch {
        IrArch::Gfx1100 => Ok(Arch::Gfx1100),
        IrArch::Gfx1151 => Ok(Arch::Gfx1151),
        IrArch::Gfx1200 => Ok(Arch::Gfx1200),
        IrArch::Gfx1201 => Ok(Arch::Gfx1201),
        other => Err(format!("{other:?} is not a supported builder architecture")),
    }
}

/// One kernel as label + canonical instruction lines in layout order: a block's label is
/// `.LBB<id>:`, the spelling the printer gives a branch's label operand.
fn kernel_program(kernel: &Kernel, arch: IrArch) -> Result<Program, String> {
    let body = &kernel.body;
    let starts: BTreeMap<usize, usize> = body.blocks.iter().map(|b| (b.range.0, b.id.0)).collect();
    let mut instructions = Vec::new();
    for (index, &id) in body.layout.iter().enumerate() {
        if let Some(block) = starts.get(&index) {
            instructions.push(Instruction::new(format!(".LBB{block}:"), vec![], vec![]));
        }
        let inst = body
            .insts
            .get(id)
            .ok_or_else(|| format!("layout index {index} names a missing instruction"))?;
        let text = peacemaker_lift::text::canonical(inst, arch)
            .map_err(|e| format!("not printable: {e}"))?;
        instructions.push(Instruction::new(text, vec![], vec![]));
    }
    Ok(Program {
        arch: isa_arch(arch)?,
        instructions,
    })
}

/// A verifier outcome that is a real missing wait (as opposed to an unsupported input).
fn is_finding(error: &str) -> bool {
    error.contains("): needs ")
}

#[derive(Default)]
struct Report {
    verified: usize,
    allowed: Vec<String>,
    unmatched: Vec<String>,
    used: Vec<(bool, usize)>,
}
impl Report {
    fn record(&mut self, scope: &str, target: &str, result: Result<Stats, String>) {
        let error = match result {
            Ok(_) => {
                self.verified += 1;
                return;
            }
            Err(error) => error,
        };
        let finding = is_finding(&error);
        let table = if finding { FINDINGS } else { UNSUPPORTED };
        match table
            .iter()
            .position(|a| a.scope == scope && a.target == target && error.contains(a.needle))
        {
            Some(i) => {
                self.used.push((finding, i));
                self.allowed
                    .push(format!("{target}: {} ({})", table[i].needle, table[i].why));
            }
            None => self.unmatched.push(format!(
                "[{}] {target}: {error}",
                if finding { "FINDING" } else { "UNSUPPORTED" }
            )),
        }
    }
    fn finish(self, scope: &str, extra: &[String]) {
        println!(
            "{scope}: {} verified, {} allowlisted, {} other",
            self.verified,
            self.allowed.len(),
            extra.len()
        );
        for line in self.allowed.iter().chain(extra) {
            println!("  {line}");
        }
        let mut stale = Vec::new();
        for (finding, table) in [(true, FINDINGS), (false, UNSUPPORTED)] {
            for (i, a) in table.iter().enumerate() {
                if a.scope == scope && !self.used.contains(&(finding, i)) {
                    stale.push(format!("{}: {} (no longer occurs)", a.target, a.needle));
                }
            }
        }
        assert!(
            self.unmatched.is_empty() && stale.is_empty(),
            "{scope}: {} outcome(s) without an allowlist entry:\n{}\n{} stale allowlist entr(ies):\n{}",
            self.unmatched.len(),
            self.unmatched.join("\n"),
            stale.len(),
            stale.join("\n")
        );
    }
}

/// The gfx942 anchors (hipcc) and twins (PM) with typed coverage: the rows16 and P1b wide
/// BF16 MFMA objects and the FP8 grouped MoE object. Each is lifted and every kernel is
/// certified by `peacemaker_isa::cdna::lifted::certify` (CDNA hazards + typed CFG wait ledger).
/// Their absence fails the gate. Every other gfx942 image stays an explicit not-verified entry.
/// Each entry is the committed file and the exact kernel symbols it must contain, and nothing
/// else.
const CDNA_CERTIFIED: [(&str, &[&str]); 12] = [
    (
        "gemm_bf16_mfma_gfx942_rows16_hipcc_gfx942.co",
        &[
            "gemm_bf16_mfma_gfx942",
            "gemm_bf16_mfma_gfx942_rows16",
            "gemm_bf16_mfma_gfx942_rows16_x2",
        ],
    ),
    (
        "gemm_bf16_mfma_gfx942_rows16_pm_gfx942.hxaco",
        &[
            "gemm_bf16_mfma_gfx942",
            "gemm_bf16_mfma_gfx942_rows16",
            "gemm_bf16_mfma_gfx942_rows16_x2",
        ],
    ),
    (
        "gemm_bf16_mfma_wide_gfx942_hipcc_gfx942.co",
        &[
            "gemm_bf16_mfma_gfx942_wide64",
            "gemm_bf16_mfma_gfx942_wide64_m32",
            "gemm_bf16_mfma_gfx942_wide64_m64",
        ],
    ),
    (
        "gemm_bf16_mfma_wide_gfx942_pm_gfx942.hxaco",
        &[
            "gemm_bf16_mfma_gfx942_wide64",
            "gemm_bf16_mfma_gfx942_wide64_m32",
            "gemm_bf16_mfma_gfx942_wide64_m64",
        ],
    ),
    (
        "gemm_fp8b128_moe_grouped_bf16mfma_hipcc_gfx942.co",
        &[
            "gemm_fp8b128_moe_grouped_bf16mfma",
            "gemm_fp8b128_moe_grouped_bf16mfma_xbf16",
            "gemm_fp8b128_moe_tile_desc_pad16",
        ],
    ),
    (
        "gemm_fp8b128_moe_grouped_bf16mfma_pm_gfx942.hxaco",
        &[
            "gemm_fp8b128_moe_grouped_bf16mfma",
            "gemm_fp8b128_moe_grouped_bf16mfma_xbf16",
            "gemm_fp8b128_moe_tile_desc_pad16",
        ],
    ),
    (
        "qwen4_g942_glue_elem_hipcc_gfx942.co",
        &[
            "qwen4_g942_gemm_bf16_rows16_hc_act",
            "qwen4_g942_gemm_bf16_rows16_hc_act_x2",
        ],
    ),
    (
        "qwen4_g942_glue_elem_pm_gfx942.hxaco",
        &[
            "qwen4_g942_gemm_bf16_rows16_hc_act",
            "qwen4_g942_gemm_bf16_rows16_hc_act_x2",
        ],
    ),
    (
        "qwen4_g942_glue_moe_hipcc_gfx942.co",
        &[
            "qwen4_g942_moe_group_scatter_desc",
            "qwen4_g942_moe_unscatter_act",
            "qwen4_g942_moe_combine_shared_add",
        ],
    ),
    (
        "qwen4_g942_glue_moe_pm_gfx942.hxaco",
        &[
            "qwen4_g942_moe_group_scatter_desc",
            "qwen4_g942_moe_unscatter_act",
            "qwen4_g942_moe_combine_shared_add",
        ],
    ),
    (
        "qwen4_g942_hc_gates_hipcc_gfx942.co",
        &[
            "gemv_bf16_xf32_k4_rows_tiled_r2",
            "gemv_bf16_xf32_k4_rows_tiled_r3",
            "gemv_bf16_xf32_k4_rows_tiled_r4",
            "gemv_bf16_xf32_k4_rows_tiled_r5",
            "gemv_bf16_xf32_k4_rows_tiled_r6",
            "gemv_bf16_xf32_k4_rows_tiled_r7",
            "gemv_bf16_xf32_k4_rows_tiled_r8",
        ],
    ),
    (
        "qwen4_g942_hc_gates_pm_gfx942.hxaco",
        &[
            "gemv_bf16_xf32_k4_rows_tiled_r2",
            "gemv_bf16_xf32_k4_rows_tiled_r3",
            "gemv_bf16_xf32_k4_rows_tiled_r4",
            "gemv_bf16_xf32_k4_rows_tiled_r5",
            "gemv_bf16_xf32_k4_rows_tiled_r6",
            "gemv_bf16_xf32_k4_rows_tiled_r7",
            "gemv_bf16_xf32_k4_rows_tiled_r8",
        ],
    ),
];

/// The gfx942 device ELF of `bytes`: itself when raw, else the single native
/// `hipv4-amdgcn-amd-amdhsa--gfx942` bundle entry (an optional `:feature` suffix allowed).
fn gfx942_elf(bytes: &[u8]) -> Result<&[u8], String> {
    use peacemaker_ir::envelope::Bundle;
    if !Bundle::is_bundle(bytes) {
        return Ok(bytes);
    }
    let (bundle, payloads) = Bundle::read(bytes).map_err(|e| format!("bundle: {e}"))?;
    let triple = format!(
        "{}gfx942",
        peacemaker_lift::bundle::HIP_DEVICE_TRIPLE_PREFIX
    );
    let mut hits = bundle.entries.iter().enumerate().filter(|(_, e)| {
        e.triple == triple
            || e.triple
                .strip_prefix(&triple)
                .is_some_and(|s| s.starts_with(':'))
    });
    let idx = hits
        .next()
        .map(|(i, _)| i)
        .ok_or("bundle has no gfx942 device entry")?;
    if hits.next().is_some() {
        return Err("bundle has multiple gfx942 device entries".into());
    }
    Ok(payloads[idx])
}

/// Typed lift + certify of every kernel; returns the kernel names on success.
fn certify_gfx942(bytes: &[u8]) -> Result<Vec<String>, String> {
    let elf = gfx942_elf(bytes)?;
    let lifted = peacemaker_lift::cdna::lift_object(elf).map_err(|e| format!("cdna lift: {e}"))?;
    peacemaker_lift::cdna::checked(&lifted).map_err(|e| format!("cdna checked: {e}"))?;
    let mut names = Vec::new();
    for kernel in &lifted.kernels {
        peacemaker_isa::cdna::lifted::certify(kernel)
            .map_err(|e| format!("cdna certify {}: {e}", kernel.image.name))?;
        names.push(kernel.image.name.clone());
    }
    Ok(names)
}

#[test]
fn every_committed_image_verifies_or_is_an_explicit_allowlist_entry() {
    let images = scan();
    assert!(!images.is_empty(), "no committed images found");
    for (required, _) in CDNA_CERTIFIED {
        assert!(
            images.iter().any(|i| i.file == required),
            "required gfx942 artifact {required} is absent from kernels/"
        );
    }
    let mut report = Report::default();
    let mut gfx942 = Vec::new();
    for image in &images {
        if let Some((_, expected)) = CDNA_CERTIFIED.iter().find(|(f, _)| *f == image.file) {
            match certify_gfx942(&image.bytes) {
                Ok(names) => {
                    let mut got = names.clone();
                    got.sort();
                    let mut want: Vec<String> = expected.iter().map(|s| s.to_string()).collect();
                    want.sort();
                    assert_eq!(got, want, "{}: kernel symbol set", image.file);
                    // One verified outcome per certified kernel.
                    for name in &names {
                        report.verified += 1;
                        println!("  certified (typed CDNA) {}#{name}", image.file);
                    }
                }
                Err(e) => report.record("pins", &image.file, Err(e)),
            }
            continue;
        }
        let program = match &image.lifted {
            Ok(program) => program,
            Err(_) if image.file.contains(GFX942) => {
                gfx942.push(format!(
                    "{}: gfx942, no typed CDNA certification wired (not verified)",
                    image.file
                ));
                continue;
            }
            Err(e) => {
                report.record("pins", &image.file, Err(format!("lift rejected: {e}")));
                continue;
            }
        };
        for kernel in &program.kernels {
            let target = format!("{}#{}", image.file, kernel.symbol.0);
            let result = kernel_program(kernel, program.target.arch)
                .and_then(|p| ledger_cfg::verify(&p, &[0..p.instructions.len()]));
            report.record("pins", &target, result);
        }
    }
    report.finish("pins", &gfx942);
}

/// A converted kernel awaiting its pool.
struct Region {
    target: String,
    tid: u32,
    region: LiftedRegion,
}

/// The pc a verifier report names (`pc <n> ...`).
fn error_pc(error: &str) -> Option<usize> {
    error
        .strip_prefix("pc ")?
        .split_whitespace()
        .next()?
        .parse()
        .ok()
}

/// Match a finding by stable module/symbol, body pc, instruction and need. The category
/// is checked against the corresponding standalone finding by the exhaustive pool test.
fn body_entry(target: &str, body_pc: usize, error: &str) -> Result<usize, String> {
    let i = POOL_BODY
        .iter()
        .position(|a| a.target == target && a.body_pc == body_pc)
        .ok_or_else(|| format!("unlisted body finding: {target}, body pc {body_pc}"))?;
    let a = &POOL_BODY[i];
    if !error.contains(a.insn) || !error.contains(a.need) {
        return Err(format!(
            "{target} body pc {body_pc} is listed for {} {} but the report differs",
            a.insn, a.need
        ));
    }
    Ok(i)
}

/// Resolve the emission-local template id before looking up a stable body key.
/// A dispatcher pc or a region's entry/exit label has no entry by construction.
fn body_entry_for_template(
    tid_targets: &[(u32, String)],
    tid: u32,
    body_pc: usize,
    error: &str,
) -> Result<usize, String> {
    let target = tid_targets
        .iter()
        .find(|(id, _)| *id == tid)
        .map(|(_, target)| target.as_str())
        .ok_or_else(|| format!("template {tid} is not a converted region"))?;
    body_entry(target, body_pc, error)
}

fn classify_body(
    emitted: &Emitted,
    tid_targets: &[(u32, String)],
    error: &str,
) -> Result<usize, String> {
    let pc = error_pc(error).ok_or("not a hazard report")?;
    let region = emitted
        .proof
        .regions
        .iter()
        .find(|r| (r.entry_pc + 1..r.exit_pc).contains(&pc))
        .ok_or_else(|| {
            format!("pc {pc} is outside every body span: a dispatcher finding has no allowance")
        })?;
    let tid = region
        .template_id
        .ok_or_else(|| format!("region {} has no template id", region.name))?;
    // Region body lines follow the entry label and its `s_wait_alu 0x0`.
    let body_pc = pc
        .checked_sub(region.entry_pc + 2)
        .ok_or_else(|| format!("pc {pc} is the region entry of template {tid}"))?;
    body_entry_for_template(tid_targets, tid, body_pc, error)
}

#[derive(Default)]
struct PoolCounts {
    built: usize,
    clean: usize,
    body_findings: usize,
    suppressed: usize,
    summarized: usize,
    state_caps: usize,
}

const STATE_CAP_PREFIX: &str =
    "wait verification: more than 512 distinct in-flight memory states reach label";

/// The one error class the state-cap category takes; anything else is never routed to it.
fn is_state_cap(error: &str) -> bool {
    error.starts_with(STATE_CAP_PREFIX)
}

/// The entry a hosted state-cap error is, by pool, template identity, canonical label and the
/// exact whole error. `in_body(tid, label)` says whether the pool label lies inside that
/// template's emitted body span. OPEN: verifier state cap, not proven.
fn classify_state_cap_with(
    table: &[PoolStateCap],
    tid_targets: &[(u32, String)],
    pool_target: &str,
    error: &str,
    in_body: impl Fn(u32, &str) -> bool,
) -> Result<usize, String> {
    if !is_state_cap(error) {
        return Err("not a state-cap error".into());
    }
    let label = error
        .split('`')
        .nth(1)
        .ok_or("state-cap error names no label")?;
    let rest = label
        .strip_prefix(".Lt")
        .ok_or_else(|| format!("label {label} is not a hosted-body label"))?;
    let (tid, index) = rest
        .split_once("b_")
        .ok_or_else(|| format!("label {label} is not `.Lt<tid>b_<n>`"))?;
    let tid: u32 = tid
        .parse()
        .map_err(|_| format!("label {label}: bad template id"))?;
    let canonical = format!(".LBB{index}");
    let template = tid_targets
        .iter()
        .find(|(t, _)| *t == tid)
        .map(|(_, s)| s.as_str())
        .ok_or_else(|| format!("label {label}: template {tid} is not a converted region"))?;
    if !in_body(tid, label) {
        return Err(format!(
            "label {label} is not inside the emitted body span of template {tid}"
        ));
    }
    let i = table
        .iter()
        .position(|c| {
            c.pool_target == pool_target
                && c.template_target == template
                && c.body_label == canonical
        })
        .ok_or_else(|| {
            format!(
                "unlisted state cap: pool {pool_target}, template {template}, label {canonical}"
            )
        })?;
    let expected = table[i]
        .error
        .replace("{tid}", &tid.to_string())
        .replace("{n}", index);
    if error != expected {
        return Err(format!(
            "state-cap error differs from the pinned text `{expected}`"
        ));
    }
    Ok(i)
}

/// `classify_state_cap_with` against a real emission: the label must be an instruction
/// strictly inside the body span (entry..exit) of the region hosting template `tid`.
fn classify_state_cap(
    emitted: &Emitted,
    table: &[PoolStateCap],
    tid_targets: &[(u32, String)],
    pool_target: &str,
    error: &str,
) -> Result<usize, String> {
    classify_state_cap_with(table, tid_targets, pool_target, error, |tid, label| {
        let Some(region) = emitted
            .proof
            .regions
            .iter()
            .find(|r| r.template_id == Some(tid))
        else {
            return false;
        };
        let want = format!("{label}:");
        emitted
            .program
            .instructions
            .iter()
            .position(|i| i.text.trim() == want)
            .is_some_and(|pc| (region.entry_pc + 1..region.exit_pc).contains(&pc))
    })
}

/// Table invariants: every entry's standalone `UNSUPPORTED` row exists exactly once with the
/// same canonical label; pool targets are distinct; labels are canonical `.LBB<n>`.
fn check_state_cap_table(table: &[PoolStateCap], unsupported: &[Allow]) -> Result<(), String> {
    for (i, c) in table.iter().enumerate() {
        if !c.body_label.starts_with(".LBB") {
            return Err(format!(
                "entry {i}: label {} is not `.LBB<n>`",
                c.body_label
            ));
        }
        let rows = unsupported
            .iter()
            .filter(|u| {
                u.scope == "pins"
                    && u.target == c.template_target
                    && u.needle.contains(&format!("`{}`", c.body_label))
                    && u.needle
                        .contains("more than 512 distinct in-flight memory states")
            })
            .count();
        if rows != 1 {
            return Err(format!(
                "entry {i}: {rows} standalone UNSUPPORTED `{}` row(s) for {}",
                c.body_label, c.template_target
            ));
        }
        if table
            .iter()
            .filter(|o| o.pool_target == c.pool_target)
            .count()
            != 1
        {
            return Err(format!(
                "entry {i}: pool {} listed more than once",
                c.pool_target
            ));
        }
    }
    Ok(())
}

/// One built pool: dispatcher-only verification must pass; strict verification must be clean
/// or report exactly one listed body finding. `hosted` are the template ids it hosts; `used`
/// collects the entries a single-template pool reproduced.
#[allow(clippy::too_many_arguments)]
fn check_pool(
    report: &mut Report,
    counts: &mut PoolCounts,
    used: &mut Vec<usize>,
    caps_used: &mut Vec<usize>,
    tid_targets: &[(u32, String)],
    target: &str,
    emitted: &Emitted,
    hosted: &[u32],
) {
    counts.built += 1;
    let dispatcher = match ledger_cfg::verify_dispatcher_emitted(emitted) {
        Ok(stats) => stats,
        Err(e) => {
            report.unmatched.push(format!("[DISPATCHER] {target}: {e}"));
            return;
        }
    };
    counts.suppressed += dispatcher.suppressed;
    counts.summarized += dispatcher.summarized;
    match ledger_cfg::verify_emitted(emitted) {
        Ok(_) => {
            let listed: Vec<u32> = hosted
                .iter()
                .copied()
                .filter(|t| {
                    tid_targets.iter().any(|(id, target)| {
                        *id == *t && POOL_BODY.iter().any(|a| a.target == target)
                    })
                })
                .collect();
            if dispatcher.suppressed != 0 || !listed.is_empty() {
                report.unmatched.push(format!(
                    "[POOL BODY] {target}: strict verification is clean but {} body hazard(s) \
                     were suppressed and templates {listed:?} are listed (stale entries)",
                    dispatcher.suppressed
                ));
            } else {
                report.verified += 1;
                counts.clean += 1;
            }
        }
        Err(e) if is_state_cap(&e) => {
            // OPEN: verifier state cap, not proven. Counted separately from verified.
            match classify_state_cap(emitted, POOL_STATE_CAP, tid_targets, target, &e) {
                Ok(i) => {
                    counts.state_caps += 1;
                    caps_used.push(i);
                    let c = &POOL_STATE_CAP[i];
                    println!(
                        "  OPEN: verifier state cap, not proven: pool {} template {} label {}: {}",
                        c.pool_target, c.template_target, c.body_label, c.why
                    );
                }
                Err(why) => report
                    .unmatched
                    .push(format!("[POOL STATE CAP] {target}: {why}: {e}")),
            }
        }
        Err(e) => match classify_body(emitted, tid_targets, &e) {
            // A summarized region is not walked by the dispatcher-only verifier, so whether
            // it counted this body finding as suppressed depends on the summary; the strict
            // walk is what finds it and the exact entry is what accepts it.
            Ok(i) => {
                counts.body_findings += 1;
                if hosted.len() == 1 {
                    used.push(i);
                }
            }
            Err(why) => report
                .unmatched
                .push(format!("[POOL BODY] {target}: {why}: {e}")),
        },
    }
}

#[test]
fn lifted_pools_dispatcher_is_fatal_and_bodies_match_exact_entries() {
    let images = scan();
    let mut report = Report::default();
    let mut by_block: BTreeMap<u32, Vec<Region>> = BTreeMap::new();
    let mut tid = 16u32;
    for image in &images {
        let Ok(program) = &image.lifted else { continue };
        if program.target.arch != IrArch::Gfx1201 {
            continue;
        }
        // The Hipcc lift of the object, once for all its kernels (`lift_region` would lift
        // and re-emit the whole object per kernel).
        let mut hipcc: Option<Result<HipccObject, String>> = None;
        for kernel in &program.kernels {
            let Abi::Hsa { metadata, .. } = &kernel.abi else {
                continue;
            };
            let block = metadata.parsed.max_flat_workgroup_size;
            let target = format!("{}#{}", image.file, kernel.symbol.0);
            let inputs = RegionInputs {
                prefix: format!(".Lt{tid}b"),
                workgroup_x: ld::WG_X,
                workgroup_yz: ld::WG_YZ,
                block,
                dynamic_lds: 0,
            };
            let object = hipcc.get_or_insert_with(|| HipccObject::lift(&image.bytes));
            let lifted = match object {
                Ok(object) => object.region(&kernel.symbol.0, &inputs),
                Err(reason) => Err(RegionError {
                    symbol: kernel.symbol.0.clone(),
                    reason: reason.clone(),
                }),
            };
            match lifted {
                Ok(region) => {
                    by_block.entry(block).or_default().push(Region {
                        target,
                        tid,
                        region,
                    });
                    tid += 1;
                }
                Err(e) => report.record(
                    "pools",
                    &format!("region:{target}"),
                    Err(format!("not convertible: {}", e.reason)),
                ),
            }
        }
    }
    assert!(!by_block.is_empty(), "no kernel converts to a region");
    // Every stable key names exactly one converted module/symbol and the `pins`
    // finding it is the hosted copy of (same instruction, need and category).
    for a in POOL_BODY {
        if by_block
            .values()
            .flatten()
            .filter(|r| r.target == a.target)
            .count()
            != 1
        {
            report.unmatched.push(format!(
                "[POOL BODY ENTRY] {} is not exactly one region in this scan",
                a.target
            ));
        }
        let prefix = match a.category {
            Category::ModelLimit => "MODEL LIMITATION",
            Category::Candidate => "UNPROVEN",
        };
        if !FINDINGS.iter().any(|f| {
            f.scope == "pins"
                && f.target == a.target
                && f.needle.contains(a.insn)
                && f.needle.contains(a.need)
                && f.why.starts_with(prefix)
        }) {
            report.unmatched.push(format!(
                "[POOL BODY ENTRY] {}: no `pins` finding with {} {} and category {:?}",
                a.target, a.insn, a.need, a.category
            ));
        }
        if POOL_BODY.iter().filter(|b| b.target == a.target).count() != 1 {
            report.unmatched.push(format!(
                "[POOL BODY ENTRY] {} has more than one entry",
                a.target
            ));
        }
    }
    // The OPEN state-cap entries: each template target is exactly one region of this scan
    // and has its standalone `UNSUPPORTED` `.LBB28` row.
    for c in POOL_STATE_CAP {
        let n = by_block
            .values()
            .flatten()
            .filter(|r| r.target == c.template_target)
            .count();
        if n != 1 {
            report.unmatched.push(format!(
                "[POOL STATE CAP ENTRY] {} is {n} region(s) in this scan, not exactly one",
                c.template_target
            ));
        }
    }
    if let Err(why) = check_state_cap_table(POOL_STATE_CAP, UNSUPPORTED) {
        report
            .unmatched
            .push(format!("[POOL STATE CAP ENTRY] {why}"));
    }
    // `ld::pool` already verifies its own emission with the dispatcher-only verifier and
    // refuses it on any dispatcher hazard (`POOL_VERIFY_FAILED`); that is never a refusal
    // with an allowlist entry. Preconstruction refusals (blockDim, SGPR limit, ...) are not
    // verifier findings and keep their exact `UNSUPPORTED` entries.
    let mut counts = PoolCounts::default();
    let mut used: Vec<usize> = Vec::new();
    let mut caps_used: Vec<usize> = Vec::new();
    // The current template id -> `file#symbol` identity of every converted region.
    let tid_targets: Vec<(u32, String)> = by_block
        .values()
        .flatten()
        .map(|r| (r.tid, r.target.clone()))
        .collect();
    let mut first_single: Option<Emitted> = None;
    for (&block, regions) in &by_block {
        let mut built = Vec::new();
        for r in regions {
            let single = ld::pool(
                &format!("pm_pool_{}", r.tid),
                block,
                &[ld::Hosted {
                    tmpl: r.tid,
                    region: &r.region,
                }],
                &[],
            );
            let target = format!("pool:b{block}:{}", r.target);
            match single {
                Ok(emitted) => {
                    built.push(r);
                    check_pool(
                        &mut report,
                        &mut counts,
                        &mut used,
                        &mut caps_used,
                        &tid_targets,
                        &target,
                        &emitted,
                        &[r.tid],
                    );
                    first_single.get_or_insert(emitted);
                }
                Err(e) if e.contains(POOL_VERIFY_FAILED) => report
                    .unmatched
                    .push(format!("[DISPATCHER FATAL] {target}: {e}")),
                Err(e) => report.record("pools", &target, Err(format!("pool refused: {e}"))),
            }
        }
        if !built.is_empty() {
            let hosted: Vec<ld::Hosted> = built
                .iter()
                .map(|r| ld::Hosted {
                    tmpl: r.tid,
                    region: &r.region,
                })
                .collect();
            let tids: Vec<u32> = built.iter().map(|r| r.tid).collect();
            let target = format!("pool:b{block}:multi");
            match ld::pool(&format!("pm_pool_b{block}"), block, &hosted, &[]) {
                Ok(emitted) => check_pool(
                    &mut report,
                    &mut counts,
                    &mut used,
                    &mut caps_used,
                    &tid_targets,
                    &target,
                    &emitted,
                    &tids,
                ),
                Err(e) if e.contains(POOL_VERIFY_FAILED) => report
                    .unmatched
                    .push(format!("[DISPATCHER FATAL] {target}: {e}")),
                Err(e) => report.record("pools", &target, Err(format!("pool refused: {e}"))),
            }
        }
    }
    assert!(counts.built > 0, "no pool was built");
    // Fail closed: an entry no single-template pool reproduced is stale.
    for (i, a) in POOL_BODY.iter().enumerate() {
        if !used.contains(&i) {
            report.unmatched.push(format!(
                "[STALE POOL BODY] {} body pc {} no longer occurs",
                a.target, a.body_pc
            ));
        }
    }
    // Same for the OPEN state-cap table: every entry must occur in its own pool.
    for (i, c) in POOL_STATE_CAP.iter().enumerate() {
        if !caps_used.contains(&i) {
            report.unmatched.push(format!(
                "[STALE POOL STATE CAP] OPEN: verifier state cap, not proven: {} ({} {}) no \
                 longer occurs",
                c.pool_target, c.template_target, c.body_label
            ));
        }
    }
    // The dispatcher is fatal outside the body spans, with no allowance: removing the one
    // KMcnt wait that retires `t_ready` before the trace record reads it (the dispatcher
    // finding fixed in this tree: unmatched template id -> `.Lpublish` -> `.Lsignaled`) must be
    // rejected by the dispatcher-only verifier, with the body spans' findings still unreported.
    let mut mutated = first_single.expect("a single-template pool was built");
    let signaled = mutated
        .program
        .instructions
        .iter()
        .position(|i| i.text.trim() == ".Lsignaled:")
        .expect("the dispatcher has a `.Lsignaled` label");
    let traced = mutated
        .program
        .instructions
        .iter()
        .position(|i| i.text.trim() == ".Ltraced:")
        .expect("the dispatcher has a `.Ltraced` label");
    let waits: Vec<usize> = (signaled..traced)
        .filter(|&pc| {
            mutated.program.instructions[pc]
                .text
                .trim()
                .starts_with("s_wait_kmcnt")
        })
        .collect();
    assert!(!waits.is_empty(), "the trace path waits KMcnt");
    for pc in waits.into_iter().rev() {
        mutated.program.instructions.remove(pc);
    }
    let e = ledger_cfg::verify_dispatcher_emitted(&mutated)
        .expect_err("a dispatcher without the t_ready KMcnt wait must be rejected");
    assert!(
        e.contains("needs Km wait") && e.contains("s[86:87]"),
        "unexpected dispatcher rejection: {e}"
    );
    report.finish(
        "pools",
        &[format!(
            "{} pools built, dispatcher verified fatally in all ({} region exits summarized as \
             drained, {} body hazards suppressed in the regions walked); {} strict-clean, {} with \
             an exact body finding; {} hosted bodies OPEN: verifier state cap, not proven (full \
             strict wait certification not claimed)",
            counts.built,
            counts.summarized,
            counts.suppressed,
            counts.clean,
            counts.body_findings,
            counts.state_caps
        )],
    );
}

#[test]
fn state_cap_classifier_rejects_wrong_label_template_pool_and_other_errors() {
    // OPEN: verifier state cap, not proven. Synthetic context, real classifier logic.
    let t_a = "a.hxaco#ka".to_string();
    let t_b = "b.hxaco#kb".to_string();
    let tids = vec![(7u32, t_a.clone()), (8u32, t_b.clone())];
    let entry = |pool: &'static str, tmpl: &'static str, label: &'static str| PoolStateCap {
        pool_target: pool,
        template_target: tmpl,
        body_label: label,
        error: STATE_CAP_ERROR,
        why: "test",
    };
    let table = [entry("pool:b64:a.hxaco#ka", "a.hxaco#ka", ".LBB28")];
    let err = |tid: u32, n: u32| {
        format!(
            "wait verification: more than 512 distinct in-flight memory states reach label \
             `.Lt{tid}b_{n}` (path-sensitive state explosion); wait the counters before the \
             branches that reach it"
        )
    };
    let inside = |_: u32, _: &str| true;
    let pool = "pool:b64:a.hxaco#ka";
    // Correct pinned entry succeeds.
    assert_eq!(
        classify_state_cap_with(&table, &tids, pool, &err(7, 28), inside),
        Ok(0)
    );
    // Wrong label index.
    assert!(classify_state_cap_with(&table, &tids, pool, &err(7, 29), inside).is_err());
    // Wrong template (tid maps to another target).
    assert!(classify_state_cap_with(&table, &tids, pool, &err(8, 28), inside).is_err());
    // Right template, wrong pool.
    assert!(classify_state_cap_with(&table, &tids, "pool:b64:multi", &err(7, 28), inside).is_err());
    // Label outside the template's body span (dispatcher/entry/exit).
    assert!(classify_state_cap_with(&table, &tids, pool, &err(7, 28), |_, _| false).is_err());
    // Fabricated entries with the wrong label or wrong template never match the real error.
    let wrong_label = [entry(pool, "a.hxaco#ka", ".LBB29")];
    assert!(classify_state_cap_with(&wrong_label, &tids, pool, &err(7, 28), inside).is_err());
    let wrong_tmpl = [entry(pool, "b.hxaco#kb", ".LBB28")];
    assert!(classify_state_cap_with(&wrong_tmpl, &tids, pool, &err(7, 28), inside).is_err());
    // Altered text and non-state-cap errors.
    let altered = err(7, 28).replace("512", "513");
    assert!(classify_state_cap_with(&table, &tids, pool, &altered, inside).is_err());
    let extended = format!("{} extra", err(7, 28));
    assert!(classify_state_cap_with(&table, &tids, pool, &extended, inside).is_err());
    let hazard = "pc 12 `v_add_co_u32 v1, s0, v1, s2`: needs Store wait 0";
    assert!(!is_state_cap(hazard));
    assert!(classify_state_cap_with(&table, &tids, pool, hazard, inside).is_err());
}

#[test]
fn state_cap_table_is_pinned_to_standalone_unsupported_rows() {
    assert_eq!(POOL_STATE_CAP.len(), 5);
    check_state_cap_table(POOL_STATE_CAP, UNSUPPORTED).expect("table invariants");
    let bogus = [PoolStateCap {
        pool_target: "pool:b64:x#y",
        template_target: "x#y",
        body_label: ".LBB28",
        error: STATE_CAP_ERROR,
        why: "test",
    }];
    assert!(check_state_cap_table(&bogus, UNSUPPORTED).is_err());
}

/// Frozen mechanically from af76fde0b1's complete positional table, before removing tids.
/// Equality of all predicate fields proves equality of the accepted finding sets, not
/// merely equality of the findings the current verifier happens to emit.
#[test]
fn stable_body_keys_accept_exactly_the_legacy_finding_set() {
    let legacy: Vec<serde_json::Value> =
        serde_json::from_str(include_str!("ledger_cfg_pins/legacy_pool_body.json")).unwrap();
    assert_eq!(
        POOL_BODY
            .iter()
            .filter(|row| legacy.iter().any(|old| old["target"] == row.target))
            .count(),
        legacy.len()
    );
    let tids: Vec<(u32, String)> = legacy
        .iter()
        .map(|r| {
            (
                r["tid"].as_u64().unwrap() as u32,
                r["target"].as_str().unwrap().into(),
            )
        })
        .collect();
    let mut shifted: Vec<(u32, String)> = tids
        .iter()
        .map(|(tid, target)| (tid + 1, target.clone()))
        .collect();
    // A new image sorting before every existing template. It has no finding entry.
    shifted.push((16, "a_new_image.hxaco#new_kernel".into()));
    for old in &legacy {
        let tid = old["tid"].as_u64().unwrap() as u32;
        let target = old["target"].as_str().unwrap();
        let pc = old["body_pc"].as_u64().unwrap() as usize;
        let insn = old["insn"].as_str().unwrap();
        let need = old["need"].as_str().unwrap();
        let category = match old["category"].as_str().unwrap() {
            "ModelLimit" => Category::ModelLimit,
            "Candidate" => Category::Candidate,
            other => panic!("unknown legacy category {other}"),
        };
        let i = POOL_BODY
            .iter()
            .position(|row| row.target == target)
            .unwrap();
        let row = &POOL_BODY[i];
        // Every field of the old acceptance predicate survives unchanged, and there is
        // exactly one old tid and one new target for each predicate.
        assert_eq!(
            (row.target, row.body_pc, row.category, row.insn, row.need),
            (target, pc, category, insn, need)
        );
        assert_eq!(tids.iter().filter(|(t, _)| *t == tid).count(), 1);
        assert_eq!(tids.iter().filter(|(_, t)| t == target).count(), 1);
        let error = format!("pc {pc} {insn} (after label `.Lt{tid}b_0`): {need}");
        assert_eq!(body_entry_for_template(&tids, tid, pc, &error), Ok(i));
        let shifted_error = error.replace(&format!(".Lt{tid}b_"), &format!(".Lt{}b_", tid + 1));
        assert_eq!(
            body_entry_for_template(&shifted, tid + 1, pc, &shifted_error),
            Ok(i)
        );
        // Counterexamples remain rejected; renumbering is the only allowed difference.
        assert!(body_entry_for_template(&tids, tid, pc + 1, &error).is_err());
        assert!(
            body_entry_for_template(&tids, tid, pc, &error.replace(insn, "`changed`")).is_err()
        );
        assert!(
            body_entry_for_template(&tids, tid, pc, &error.replace(need, "needs changed")).is_err()
        );
        assert!(body_entry_for_template(&shifted, 16, pc, &error).is_err());
    }
}
