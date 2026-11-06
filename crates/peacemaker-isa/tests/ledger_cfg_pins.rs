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
//!   converted kernel of its blockDim, and each pool is verified (`verify_emitted`).
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
use peacemaker_isa::Arch;
use peacemaker_lift::rewrite::region::{lift_region, LiftedRegion, RegionInputs};
use peacemaker_lift::{lift_object, Options};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[path = "ledger_cfg_pins/allowlist.rs"]
mod allowlist;
use allowlist::{FINDINGS, UNSUPPORTED};

/// gfx942 images: `lift_object` has no opcode table for them (they are emitted by the typed
/// `cdna` builder and encoded by `peacemaker_cdna`), so there is no instruction text to
/// verify here. They are counted, never verified.
const GFX942: &str = "gfx942";

/// The proof of the dispatcher finding every pool shares (`.Lnext`, `v_mov_b32_e32 v1, 0`,
/// `needs Store wait 0`): the publish path of the dispatcher with a done event (a
/// non-returning `global_atomic_add_u32 v1, v2, s[PTR_B]` that counts on STOREcnt and locks
/// v1) takes `.Ltraced` with `TRACE_ON = 0`, which skips the trace block's `release(Dev)`
/// (`storecnt 0`); the builder's linear `wait_all` at `.Ltraced` then emits no store wait
/// (its linear path drained the atomic in the trace block), and the back edge `s_branch
/// .Lnext` reaches `v_mov_b32_e32 v1, 0` while the atomic may still read v1. Both inputs are
/// independent runtime values (task descriptor done_event, kernarg trace_on). The
/// timeout-path witness the verifier first prints is scalar-infeasible (timeout forces
/// `.Ldone`); this one is feasible. Source: `kernels/lifted_dispatch_gfx1201.rs` publish and
/// trace blocks, `global_sync.rs` release/atomic accounting. No product fix is applied.
const DISPATCHER_WAR: &str = "(after label `.Lnext`): needs Store wait 0";
const DISPATCHER_WAR_INSN: &str = "`v_mov_b32_e32 v1, 0`";

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

#[test]
fn every_committed_image_verifies_or_is_an_explicit_allowlist_entry() {
    let images = scan();
    assert!(!images.is_empty(), "no committed images found");
    let mut report = Report::default();
    let mut gfx942 = Vec::new();
    for image in &images {
        let program = match &image.lifted {
            Ok(program) => program,
            Err(_) if image.file.contains(GFX942) => {
                gfx942.push(format!(
                    "{}: gfx942, no opcode table (not verified)",
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

#[test]
fn lifted_pools_are_verified_and_share_exactly_the_dispatcher_finding() {
    let images = scan();
    let mut report = Report::default();
    let mut by_block: BTreeMap<u32, Vec<Region>> = BTreeMap::new();
    let mut tid = 16u32;
    for image in &images {
        let Ok(program) = &image.lifted else { continue };
        if program.target.arch != IrArch::Gfx1201 {
            continue;
        }
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
            match lift_region(&image.bytes, &kernel.symbol.0, &inputs) {
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
    // The dispatcher finding is expected in every pool that builds; any other outcome needs
    // its own allowlist entry.
    let (mut pools, mut dispatcher) = (0usize, 0usize);
    let mut outcome = |report: &mut Report, target: String, built: Result<_, String>| {
        let result = built.and_then(|e| ledger_cfg::verify_emitted(&e));
        match &result {
            Err(e) if e.contains(DISPATCHER_WAR) && e.contains(DISPATCHER_WAR_INSN) => {
                pools += 1;
                dispatcher += 1;
            }
            Ok(_) => pools += 1,
            // The pool itself was refused: no emission to verify.
            Err(_) => {}
        }
        if !matches!(&result, Err(e) if e.contains(DISPATCHER_WAR) && e.contains(DISPATCHER_WAR_INSN))
        {
            report.record("pools", &target, result);
        }
    };
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
                None,
            );
            if single.is_ok() {
                built.push(r);
            }
            outcome(
                &mut report,
                format!("pool:b{block}:{}", r.target),
                single.map_err(|e| format!("pool refused: {e}")),
            );
        }
        if !built.is_empty() {
            let hosted: Vec<ld::Hosted> = built
                .iter()
                .map(|r| ld::Hosted {
                    tmpl: r.tid,
                    region: &r.region,
                })
                .collect();
            outcome(
                &mut report,
                format!("pool:b{block}:multi"),
                ld::pool(&format!("pm_pool_b{block}"), block, &hosted, None)
                    .map_err(|e| format!("pool refused: {e}")),
            );
        }
    }
    assert!(pools > 0, "no pool was built");
    // Fail closed both ways: a pool without the dispatcher finding means it was fixed (drop
    // `DISPATCHER_WAR`); a finding in only some pools means they are not one family.
    let summary = vec![format!(
        "{dispatcher} of {pools} pools carry the dispatcher WAR finding ({DISPATCHER_WAR_INSN} {DISPATCHER_WAR})"
    )];
    report.finish("pools", &summary);
    assert_eq!(
        dispatcher, pools,
        "the dispatcher finding must be exactly the pool family"
    );
}
