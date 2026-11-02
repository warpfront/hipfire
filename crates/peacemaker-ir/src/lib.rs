// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Typed RDNA machine-code representation and gfx12 instruction tables.
pub mod cfg;
pub mod codec;
pub mod descriptor;
pub mod edit;
pub mod effects;
pub mod emu;
pub mod envelope;
pub mod inst;
pub mod isa;
pub mod kernarg;
pub mod lds;
pub mod metadata;
pub mod operand;
pub mod passes;
pub mod provenance;
pub mod reg;
pub mod state;
pub mod wait;

pub use cfg::{Block, BlockId, Body, InstId};
pub use inst::{Arch, Form, FormFields, Inst, Kernel, Opcode, Program, Target};

#[cfg(test)]
#[path = "../../rocm.rs"]
mod rocm;

/// Test-only assembler from the shared selected ROCm root.
#[cfg(test)]
pub(crate) fn rocm_llvm_mc() -> std::path::PathBuf {
    rocm::llvm("llvm-mc")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        collections::{HashMap, HashSet},
        process::Command,
    };
    #[test]
    fn builder_gfx1201_golden_spellings_print_from_typed_operands() {
        use operand::{CacheScope, Depctr, ImmField, InlineConst, Msg, Operand as O, VmemToken};
        use reg::{Kind, RegRef};
        fn v(base: u16, len: u8) -> O {
            O::Reg(RegRef {
                kind: Kind::V,
                base,
                len,
            })
        }
        fn s(base: u16, len: u8) -> O {
            O::Reg(RegRef {
                kind: Kind::S,
                base,
                len,
            })
        }
        fn u(value: u32) -> O {
            O::Imm(ImmField::Unsigned(value))
        }
        fn render(
            name: &str,
            operands: Vec<O>,
            mods: operand::Modifiers,
            fields: FormFields,
        ) -> String {
            let row = isa::gfx12().iter().find(|r| r.name == name).unwrap();
            Inst::from_parts(
                Arch::Gfx1201,
                row.op,
                row.form,
                fields,
                operands.into_iter().collect(),
                mods,
                None,
                Default::default(),
            )
            .unwrap()
            .text(Arch::Gfx1201)
            .unwrap()
        }
        let plain = operand::Modifiers::default();
        let wmma = operand::Modifiers {
            neg_lo: 3,
            ..plain.clone()
        };
        let mul = isa::gfx12()
            .iter()
            .find(|r| r.name == "v_dual_mul_f32")
            .unwrap()
            .op;
        let fmac = isa::gfx12()
            .iter()
            .find(|r| r.name == "v_dual_fmac_f32")
            .unwrap()
            .op;
        let expected = [
            (
                render(
                    "s_wait_loadcnt",
                    vec![u(3)],
                    plain.clone(),
                    FormFields::None,
                ),
                "s_wait_loadcnt 0x3",
            ),
            (
                render("s_wait_dscnt", vec![u(1)], plain.clone(), FormFields::None),
                "s_wait_dscnt 0x1",
            ),
            (
                render(
                    "s_wait_loadcnt_dscnt",
                    vec![u(0x205)],
                    plain.clone(),
                    FormFields::None,
                ),
                "s_wait_loadcnt_dscnt 0x205",
            ),
            (
                render("s_wait_kmcnt", vec![u(0)], plain.clone(), FormFields::None),
                "s_wait_kmcnt 0x0",
            ),
            (
                render(
                    "s_wait_alu",
                    vec![O::Depctr(Depctr::SaSdst(0))],
                    plain.clone(),
                    FormFields::None,
                ),
                "s_wait_alu depctr_sa_sdst(0)",
            ),
            (
                render(
                    "s_wait_alu",
                    vec![O::Depctr(Depctr::VaSdst(0))],
                    plain.clone(),
                    FormFields::None,
                ),
                "s_wait_alu depctr_va_sdst(0)",
            ),
            (
                render(
                    "s_barrier_signal",
                    vec![O::Imm(ImmField::Sopp(-1))],
                    plain.clone(),
                    FormFields::None,
                ),
                "s_barrier_signal -1",
            ),
            (
                render(
                    "s_barrier_wait",
                    vec![u(0xffff)],
                    plain.clone(),
                    FormFields::None,
                ),
                "s_barrier_wait 0xffff",
            ),
            (
                render(
                    "global_inv",
                    vec![O::Scope(CacheScope::Se)],
                    plain.clone(),
                    FormFields::None,
                ),
                "global_inv scope:SCOPE_SE",
            ),
            (
                render(
                    "v_wmma_i32_16x16x32_iu4",
                    vec![
                        v(57, 8),
                        v(71, 2),
                        v(1, 2),
                        O::Inline(InlineConst::Integer(0)),
                    ],
                    wmma,
                    FormFields::None,
                ),
                "v_wmma_i32_16x16x32_iu4 v[57:64], v[71:72], v[1:2], 0 neg_lo:[1,1,0]",
            ),
            (
                render(
                    "v_swmmac_i32_16x16x64_iu4",
                    vec![v(0, 8), v(8, 2), v(10, 4), v(14, 1)],
                    plain.clone(),
                    FormFields::None,
                ),
                "v_swmmac_i32_16x16x64_iu4 v[0:7], v[8:9], v[10:13], v14",
            ),
            (
                render(
                    "v_dual_mul_f32",
                    vec![v(83, 1), v(97, 1), v(81, 1), v(84, 1), v(98, 1), v(81, 1)],
                    plain.clone(),
                    FormFields::Vopd {
                        y_op: mul,
                        x_operands: 3,
                    },
                ),
                "v_dual_mul_f32 v83, v97, v81 :: v_dual_mul_f32 v84, v98, v81",
            ),
            (
                render(
                    "v_dual_fmac_f32",
                    vec![v(163, 1), v(83, 1), v(57, 1), v(166, 1), v(84, 1), v(58, 1)],
                    plain.clone(),
                    FormFields::Vopd {
                        y_op: fmac,
                        x_operands: 3,
                    },
                ),
                "v_dual_fmac_f32 v163, v83, v57 :: v_dual_fmac_f32 v166, v84, v58",
            ),
            (
                render(
                    "buffer_load_b64",
                    vec![
                        v(0, 2),
                        v(2, 1),
                        s(4, 4),
                        s(8, 1),
                        O::Vmem(VmemToken::Offen),
                    ],
                    plain.clone(),
                    FormFields::None,
                ),
                "buffer_load_b64 v[0:1], v2, s[4:7], s8 offen",
            ),
            (
                render(
                    "ds_load_2addr_stride64_b64",
                    vec![v(71, 4), v(81, 1), O::Imm(ImmField::DsOffset1(1))],
                    plain.clone(),
                    FormFields::None,
                ),
                "ds_load_2addr_stride64_b64 v[71:74], v81 offset1:1",
            ),
            (
                render("s_clause", vec![u(1)], plain.clone(), FormFields::None),
                "s_clause 0x1",
            ),
            (
                render(
                    "v_cvt_f32_i32_e32",
                    vec![v(57, 1), v(57, 1)],
                    plain.clone(),
                    FormFields::None,
                ),
                "v_cvt_f32_i32_e32 v57, v57",
            ),
            (
                render(
                    "s_sendmsg",
                    vec![O::SendMsg(Msg { id: 3, op: 0 })],
                    plain.clone(),
                    FormFields::None,
                ),
                "s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)",
            ),
            (
                render("s_endpgm", vec![], plain, FormFields::None),
                "s_endpgm",
            ),
        ];
        assert_eq!(expected.len(), 19);
        for (actual, canonical) in expected {
            assert_eq!(actual, canonical);
        }
    }

    #[test]
    fn program_rejects_reachable_path_without_endpgm_and_wave_mismatch() {
        use cfg::{Block, BlockId, Terminator};
        use inst::{Abi, KernelOrigin, Setting, SymbolId, Wave};
        let row = isa::gfx12()
            .iter()
            .find(|row| row.name == "s_endpgm")
            .unwrap();
        let end = Inst::from_parts(
            Arch::Gfx1201,
            row.op,
            row.form,
            FormFields::None,
            Default::default(),
            Default::default(),
            None,
            Default::default(),
        )
        .unwrap();
        let mut body = Body::default();
        let id = body.insts.insert(end);
        body.layout.push(id);
        body.blocks.push(Block {
            id: BlockId(0),
            range: (0, 1),
            term: Terminator::EndPgm,
            preds: Default::default(),
            succs: Default::default(),
        });
        let mut program = Program {
            target: Target {
                arch: Arch::Gfx1201,
                xnack: Setting::Any,
                sramecc: Setting::Any,
                abi_version: 4,
            },
            kernels: vec![Kernel {
                symbol: SymbolId("test".into()),
                wave: Wave::Wave32,
                abi: Abi::Raw(inst::RawAbi {
                    wave: Wave::Wave32,
                    resources: Default::default(),
                    fp: Default::default(),
                    relocations: vec![],
                    entry: vec![],
                    sidecar_sha256: [0; 32],
                }),
                body,
                origin: KernelOrigin::Authored {
                    builder_crate: "test".into(),
                    version: "1".into(),
                    git: "0".into(),
                },
            }],
            source: None,
        };
        program.validate().unwrap();
        program.kernels[0].body.blocks[0].term = Terminator::FallThrough;
        assert!(
            matches!(&program.validate(), Err(inst::ValidateError::Layout(reason)) if reason.contains("terminator disagrees"))
        );
        program.kernels[0].body.blocks[0].term = Terminator::EndPgm;
        program.kernels[0].wave = Wave::Wave64;
        assert!(
            matches!(&program.validate(), Err(inst::ValidateError::Layout(reason)) if reason.contains("raw ABI wave"))
        );
    }

    #[test]
    fn raw_abi_rejects_malformed_entry_facts_and_resources() {
        use inst::{
            Axis, EntryFact, EntryLoc, EntryValue, FpState, LocalIdField, RawAbi, RawResources,
            Relocation, Wave,
        };
        use operand::Special;
        use reg::{Kind, RegRef};
        let reg = |kind, base, len| EntryLoc::Reg(RegRef { kind, base, len });
        let fact = |loc, value| EntryFact { loc, value };
        let good = RawAbi {
            wave: Wave::Wave32,
            resources: RawResources {
                vgprs_actual: 4,
                vgprs_allocated: 8,
                sgprs_actual: 4,
                sgprs_allocated: 8,
                ..Default::default()
            },
            fp: FpState::default(),
            relocations: vec![],
            sidecar_sha256: [0; 32],
            entry: vec![
                fact(reg(Kind::S, 0, 2), EntryValue::Defined),
                fact(reg(Kind::Ttmp, 9, 1), EntryValue::WorkgroupId(Axis::X)),
                fact(
                    reg(Kind::V, 0, 1),
                    EntryValue::LocalId(vec![LocalIdField {
                        axis: Axis::X,
                        shift: 0,
                        bits: 10,
                        max: 31,
                    }]),
                ),
                fact(EntryLoc::Special(Special::Exec), EntryValue::Defined),
                fact(reg(Kind::S, 2, 1), EntryValue::Exact(7)),
                fact(reg(Kind::S, 3, 1), EntryValue::Bounded { min: 0, max: 255 }),
            ],
        };
        good.validate().unwrap();
        let cells = good.entry_cells().unwrap();
        assert!(cells.iter().any(|(c, _)| *c == inst::EntryCell::ExecLo));
        assert!(
            !cells.iter().any(|(c, _)| *c == inst::EntryCell::ExecHi),
            "wave32 EXEC is one dword"
        );
        assert!(
            !cells.iter().any(|(c, _)| *c == inst::EntryCell::VccLo),
            "VCC is undefined unless declared"
        );
        let reject = |mutate: &dyn Fn(&mut RawAbi)| {
            let mut abi = good.clone();
            mutate(&mut abi);
            assert!(
                matches!(abi.validate(), Err(inst::ValidateError::RawAbi(_))),
                "{abi:?}"
            );
        };
        reject(&|a| a.entry.push(fact(reg(Kind::S, 1, 1), EntryValue::Defined)));
        reject(&|a| a.entry.push(fact(reg(Kind::S, 6, 2), EntryValue::Exact(0))));
        reject(&|a| {
            a.entry.push(fact(
                reg(Kind::S, 7, 1),
                EntryValue::Bounded { min: 2, max: 1 },
            ))
        });
        reject(&|a| {
            a.entry.push(fact(
                EntryLoc::Special(Special::ExecHi),
                EntryValue::Defined,
            ))
        });
        reject(&|a| {
            a.entry
                .push(fact(EntryLoc::Special(Special::Null), EntryValue::Defined))
        });
        reject(&|a| a.entry.push(fact(reg(Kind::V, 8, 1), EntryValue::Defined)));
        reject(&|a| a.entry.push(fact(reg(Kind::S, 8, 1), EntryValue::Defined)));
        reject(&|a| {
            a.entry
                .push(fact(reg(Kind::V, 1, 1), EntryValue::WorkgroupId(Axis::Y)))
        });
        reject(&|a| {
            a.entry.push(fact(
                reg(Kind::V, 2, 2),
                EntryValue::LocalId(vec![LocalIdField {
                    axis: Axis::X,
                    shift: 0,
                    bits: 4,
                    max: 3,
                }]),
            ))
        });
        reject(&|a| {
            a.entry.push(fact(
                reg(Kind::V, 2, 1),
                EntryValue::LocalId(vec![
                    LocalIdField {
                        axis: Axis::X,
                        shift: 0,
                        bits: 10,
                        max: 31,
                    },
                    LocalIdField {
                        axis: Axis::Y,
                        shift: 9,
                        bits: 10,
                        max: 31,
                    },
                ]),
            ))
        });
        reject(&|a| {
            a.entry.push(fact(
                reg(Kind::V, 2, 1),
                EntryValue::LocalId(vec![LocalIdField {
                    axis: Axis::X,
                    shift: 0,
                    bits: 4,
                    max: 16,
                }]),
            ))
        });
        reject(&|a| a.fp.round32 = 4);
        reject(&|a| a.fp.exceptions = 0x200);
        reject(&|a| a.resources.vgprs_actual = 9);
        reject(&|a| {
            a.relocations.push(Relocation {
                offset: 2,
                r_type: 1,
                symbol: "s".into(),
                addend: 0,
            })
        });
    }

    #[test]
    fn vopd_packet_has_two_defs_and_fmac_reads_each_destination() {
        use operand::Operand;
        use reg::{Kind, RegRef};
        fn v(n: u16) -> Operand {
            Operand::Reg(RegRef {
                kind: Kind::V,
                base: n,
                len: 1,
            })
        }
        let row = isa::gfx12()
            .iter()
            .find(|row| row.name == "v_dual_fmac_f32")
            .unwrap();
        let inst = Inst::from_parts(
            Arch::Gfx1201,
            row.op,
            row.form,
            FormFields::Vopd {
                y_op: row.op,
                x_operands: 3,
            },
            [v(163), v(83), v(57), v(166), v(84), v(58)]
                .into_iter()
                .collect(),
            Default::default(),
            None,
            Default::default(),
        )
        .unwrap();
        assert_eq!(
            inst.effects.defs.as_slice(),
            &[
                RegRef {
                    kind: Kind::V,
                    base: 163,
                    len: 1
                },
                RegRef {
                    kind: Kind::V,
                    base: 166,
                    len: 1
                }
            ]
        );
        assert!(inst.effects.uses.contains(&RegRef {
            kind: Kind::V,
            base: 163,
            len: 1
        }));
        assert!(inst.effects.uses.contains(&RegRef {
            kind: Kind::V,
            base: 166,
            len: 1
        }));
    }

    #[test]
    fn partial_writes_and_hidden_compare_destination_have_real_dependencies() {
        use effects::{Effects, ImplicitSet};
        use operand::{Half, Operand};
        use reg::{Kind, RegRef};
        fn v(base: u16, len: u8) -> RegRef {
            RegRef {
                kind: Kind::V,
                base,
                len,
            }
        }
        fn s(base: u16) -> RegRef {
            RegRef {
                kind: Kind::S,
                base,
                len: 1,
            }
        }
        fn effect(name: &str, operands: &[Operand]) -> Effects {
            let row = isa::gfx12().iter().find(|r| r.name == name).unwrap();
            Effects::from_table(
                Arch::Gfx1201,
                row.op,
                row.form,
                operands,
                &Default::default(),
            )
            .unwrap()
        }
        let d16 = effect(
            "global_load_d16_hi_u8",
            &[
                Operand::Reg(v(139, 1)),
                Operand::Reg(v(2, 2)),
                Operand::Vmem(operand::VmemToken::Off),
            ],
        );
        assert_eq!(d16.defs.as_slice(), &[v(139, 1)]);
        assert!(
            d16.uses.contains(&v(139, 1)),
            "partial load preserves the other destination half"
        );

        let mov = effect(
            "v_mov_b16_e32",
            &[
                Operand::Half(v(5, 1), Half::Hi),
                Operand::Inline(operand::InlineConst::Integer(0)),
            ],
        );
        assert_eq!(mov.defs.as_slice(), &[v(5, 1)]);
        assert!(
            mov.uses.contains(&v(5, 1)),
            "high-half write preserves the low half"
        );

        let fmac = effect(
            "s_fmac_f32",
            &[Operand::Reg(s(6)), Operand::Reg(s(0)), Operand::Reg(s(4))],
        );
        assert!(fmac.defs.contains(&s(6)));
        assert!(
            fmac.uses.contains(&s(6)),
            "accumulator reads its previous destination"
        );

        let cmpx = effect(
            "v_cmpx_gt_i32_e64",
            &[Operand::Reg(s(7)), Operand::Reg(v(4, 1))],
        );
        assert!(
            cmpx.defs.is_empty(),
            "the encoded EXEC destination is not a printed operand"
        );
        assert_eq!(cmpx.uses.as_slice(), &[s(7), v(4, 1)]);
        assert_ne!(cmpx.implicit.writes & ImplicitSet::EXEC, 0);
    }

    /// Pinned-encoding gate: every declared gfx12 form's sample instruction is
    /// assembled by the ROCm `llvm-mc` and compared byte-for-byte with the
    /// table's committed encoding. Resolves via `ROCM_PATH` like
    /// `peacemaker-isa::toolchain`; a host without the toolchain (the no-GPU CI
    /// runner) skips the comparison — the gate still runs wherever the
    /// assembler exists.
    #[test]
    fn opcode_examples_match_pinned_llvm_mc_for_every_declared_form() {
        let mc = rocm_llvm_mc();
        if rocm::tests_disabled() {
            eprintln!("skipping table gate: explicitly disabled ROCm");
            return;
        }
        let tables = [
            ("gfx1201", isa::gfx12()),
            ("gfx1100", isa::gfx1100()),
            ("gfx1151", isa::gfx1151()),
        ];
        for (cpu, row) in tables
            .into_iter()
            .flat_map(|(cpu, table)| table.iter().map(move |row| (cpu, row)))
        {
            let mut output = Command::new(&mc)
                .args([
                    "-triple=amdgcn-amd-amdhsa",
                    &format!("-mcpu={cpu}"),
                    "-show-encoding",
                ])
                .stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::piped())
                .spawn()
                .expect("launch pinned llvm-mc");
            use std::io::Write;
            output
                .stdin
                .take()
                .expect("stdin")
                .write_all(format!("{}\n", row.sample).as_bytes())
                .expect("write instruction");
            let result = output.wait_with_output().expect("llvm-mc completion");
            assert!(
                result.status.success(),
                "{cpu} {} {:?}: {}",
                row.name,
                row.form,
                String::from_utf8_lossy(&result.stderr)
            );
            let stdout = String::from_utf8(result.stdout).expect("llvm-mc UTF-8 output");
            let encoded = stdout
                .split("encoding: [")
                .nth(1)
                .unwrap_or_else(|| panic!("no encoding for {}", row.sample))
                .split(']')
                .next()
                .expect("closing bracket");
            let bytes: Vec<_> = encoded
                .split(',')
                .map(|word| u8::from_str_radix(word.trim().trim_start_matches("0x"), 16).unwrap())
                .collect();
            let expected: Vec<_> = row
                .encoding
                .split_whitespace()
                .flat_map(|word| u32::from_str_radix(word, 16).unwrap().to_le_bytes())
                .collect();
            assert_eq!(bytes, expected, "{cpu} {} {:?}", row.sample, row.form);
        }
    }

    /// GPU compiler crates that must never be reachable from a runtime root
    /// through normal (non-dev, non-build) dependency edges. `peacemaker-npu` is
    /// deliberately absent: it is the sanctioned runtime hardware builder, and
    /// its own normal children are still traversed and checked.
    const FORBIDDEN_GPU_COMPILER_CRATES: [&str; 5] = [
        "peacemaker-ir",
        "peacemaker-lift",
        "peacemaker-author",
        "peacemaker-isa",
        "peacemaker-rip",
    ];

    /// Walks only normal (`dep_kinds.kind == null`) edges of a `cargo metadata`
    /// resolve graph from `root` and returns the first forbidden GPU compiler
    /// package reached, if any.
    fn runtime_reached_gpu_compiler(meta: &serde_json::Value, root: &str) -> Option<String> {
        let packages: HashMap<&str, &str> = meta["packages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| (p["id"].as_str().unwrap(), p["name"].as_str().unwrap()))
            .collect();
        let graph: HashMap<&str, Vec<&str>> = meta["resolve"]["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|node| {
                (
                    node["id"].as_str().unwrap(),
                    node["deps"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .filter(|dep| {
                            dep["dep_kinds"]
                                .as_array()
                                .expect("dep_kinds in cargo metadata")
                                .iter()
                                .any(|kind| kind["kind"].is_null())
                        })
                        .map(|dep| dep["pkg"].as_str().unwrap())
                        .collect(),
                )
            })
            .collect();
        let id = *packages
            .iter()
            .find(|(_, &name)| name == root)
            .expect("runtime crate present")
            .0;
        let mut stack = vec![id];
        let mut visited = HashSet::new();
        while let Some(pkg) = stack.pop() {
            if !visited.insert(pkg) {
                continue;
            }
            if FORBIDDEN_GPU_COMPILER_CRATES.contains(&packages[pkg]) {
                return Some(packages[pkg].to_string());
            }
            stack.extend(graph[pkg].iter().copied());
        }
        None
    }

    #[test]
    fn runtime_dependency_graph_does_not_include_peacemaker() {
        let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../Cargo.toml");
        let output = Command::new("cargo")
            .args(["metadata", "--format-version", "1", "--manifest-path"])
            .arg(manifest)
            .output()
            .expect("cargo metadata");
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let meta: serde_json::Value =
            serde_json::from_slice(&output.stdout).expect("cargo metadata JSON");
        for root in [
            "hipfire-daemon",
            "rdna-compute",
            "railgun-corpus",
            "railgun",
        ] {
            if let Some(reached) = runtime_reached_gpu_compiler(&meta, root) {
                panic!("{root} reaches GPU compiler crate {reached} through normal dependencies");
            }
        }
    }

    #[test]
    fn runtime_graph_walk_ignores_dev_and_build_edges_but_follows_normal_transitives() {
        let edge = |pkg: &str, kind: serde_json::Value| {
            serde_json::json!({
            "pkg": pkg, "dep_kinds": [{ "kind": kind, "target": null }] })
        };
        let meta = serde_json::json!({
            "packages": [
                { "id": "root", "name": "railgun" },
                { "id": "npu", "name": "peacemaker-npu" },
                { "id": "mid", "name": "middle" },
                { "id": "ir", "name": "peacemaker-ir" },
                { "id": "isa", "name": "peacemaker-isa" },
            ],
            "resolve": { "nodes": [
                { "id": "root", "deps": [edge("npu", serde_json::Value::Null), edge("ir", "dev".into()), edge("isa", "build".into())] },
                { "id": "npu", "deps": [edge("mid", serde_json::Value::Null)] },
                { "id": "mid", "deps": [] },
                { "id": "ir", "deps": [] },
                { "id": "isa", "deps": [] },
            ] },
        });
        // dev/build edges to forbidden crates and the sanctioned peacemaker-npu are accepted.
        assert_eq!(runtime_reached_gpu_compiler(&meta, "railgun"), None);

        // A normal transitive edge below peacemaker-npu to a GPU compiler crate is still detected.
        let mut leaking = meta.clone();
        leaking["resolve"]["nodes"][2]["deps"] =
            serde_json::json!([edge("isa", serde_json::Value::Null)]);
        assert_eq!(
            runtime_reached_gpu_compiler(&leaking, "railgun").as_deref(),
            Some("peacemaker-isa")
        );
    }

    #[test]
    fn dangerous_literal_selector_and_unaligned_smem_are_rejected() {
        use operand::Operand;
        use reg::{Kind, RegRef};
        let row = isa::gfx12()
            .iter()
            .find(|r| r.name == "v_add_co_u32")
            .unwrap();
        let mut inst = Inst {
            op: row.op,
            form: row.form,
            fields: FormFields::Vop3b { src2_unused: 0xff },
            operands: Default::default(),
            mods: Default::default(),
            literal: None,
            effects: Default::default(),
            prov: Default::default(),
        };
        assert!(matches!(
            inst.validate(Arch::Gfx1201),
            Err(inst::ValidateError::DangerousFill {
                field: "src2_unused",
                value: 0xff
            })
        ));
        inst.fields = FormFields::Vop3b { src2_unused: 0x80 };
        inst.validate(Arch::Gfx1201).unwrap();
        let row = isa::gfx12()
            .iter()
            .find(|r| r.name == "s_load_b128")
            .unwrap();
        inst.op = row.op;
        inst.form = row.form;
        inst.fields = FormFields::None;
        inst.operands.push(Operand::Reg(RegRef {
            kind: Kind::S,
            base: 5,
            len: 4,
        }));
        assert!(matches!(
            inst.validate(Arch::Gfx1201),
            Err(inst::ValidateError::MisalignedSmemSdata { .. })
        ));
    }
}
