// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>

//! CPU-only tests for the union identity schema. Every digest here is a
//! synthetic constant: nothing is loaded, compiled, proven or measured.

use radiowave::union::*;
use radiowave::{CompileManifest, Wavefront};
use serde_json::{Value, json};

/// A valid, distinct-looking identity for any domain.
fn h<T: HexIdentity>(seed: u8) -> T {
    T::parse_hex(&format!("{seed:02x}").repeat(T::HEX_LEN / 2)).unwrap()
}

/// The same hex value reinterpreted in another domain (for deliberate bindings).
fn same<T: HexIdentity>(value: &str) -> T {
    T::parse_hex(value).unwrap()
}

fn target() -> UnionTarget {
    UnionTarget {
        architecture: "gfx1201".to_owned(),
        wave: Wavefront::Wave32,
    }
}

fn semantic() -> SemanticRecipe {
    SemanticRecipe {
        recipe_id: h(20),
        name: "mq4_gemv_residual".to_owned(),
        domain: h(21),
        layout: h(22),
        fp: h(23),
        access: h(24),
    }
}

fn radv_producer() -> ProducerRecord {
    ProducerRecord {
        kind: ProducerKind::RadvAco,
        source_sha256: h(10),
        spirv_sha256: Some(h(11)),
        compiler: BuildPin {
            version: "mesa-dev".to_owned(),
            build_sha256: h(12),
        },
        driver: Some(BuildPin {
            version: "radv-dev".to_owned(),
            build_sha256: h(13),
        }),
        options_sha256: h(14),
        mesa: Some(MesaPin {
            commit: h(15),
            patched: true,
            patch_sha256: Some(h(16)),
        }),
    }
}

fn hipcc_producer() -> ProducerRecord {
    ProducerRecord {
        kind: ProducerKind::HipccLlvm,
        source_sha256: h(10),
        spirv_sha256: None,
        compiler: BuildPin {
            version: "hipcc-7.14".to_owned(),
            build_sha256: h(12),
        },
        driver: None,
        options_sha256: h(14),
        mesa: None,
    }
}

fn windows_producer() -> ProducerRecord {
    ProducerRecord {
        kind: ProducerKind::WindowsDriver,
        mesa: None,
        ..radv_producer()
    }
}

fn raw() -> RawArtifact {
    RawArtifact {
        total_length: 4096,
        code: Extent {
            offset: 256,
            length: 1024,
            sha256: h(1),
        },
        data: Some(Extent {
            offset: 2048,
            length: 128,
            sha256: h(2),
        }),
        config_sha256: h(3),
        entry_abi_sha256: h(4),
        sidecar_sha256: h(5),
        entry_name: "gemv_mq4".to_owned(),
        entry_offset: 0,
    }
}

fn certificate(raw: &RawArtifact) -> StaticCertificate {
    StaticCertificate {
        certificate_sha256: h(6),
        certified_code_sha256: same(raw.code.sha256.as_str()),
        certified_entry_abi_sha256: same(raw.entry_abi_sha256.as_str()),
    }
}

fn adapter() -> AdapterRecord {
    let body: BodyDigest = h(9);
    AdapterRecord {
        adapter_sha256: h(7),
        raw_body: Extent {
            offset: 0,
            length: 1024,
            sha256: body.clone(),
        },
        adapted_code_length: 1088,
        adapted_code_sha256: h(8),
        prologue: Extent {
            offset: 0,
            length: 64,
            sha256: h(17),
        },
        adapted_body: Extent {
            offset: 64,
            length: 1024,
            sha256: body,
        },
        entry_offset: 0,
    }
}

fn record_with(
    producer: ProducerRecord,
    adapter: Option<AdapterRecord>,
    envelope: EnvelopeRecord,
) -> CertifiedExecutableRecord {
    let raw = raw();
    CertifiedExecutableRecord {
        schema_version: UNION_SCHEMA_VERSION,
        semantic: semantic(),
        producer,
        target: target(),
        certificate: certificate(&raw),
        raw,
        adapter,
        envelope,
    }
}

fn hsa_envelope(code: Extent<ExecutableCodeDigest>) -> EnvelopeRecord {
    EnvelopeRecord::Hsa(HsaEnvelope {
        image_kind: HsaImageKind::Elf,
        image_sha256: h(30),
        image_length: 8192,
        code,
        symbol: "gemv_mq4".to_owned(),
    })
}

/// ACO-origin code adapted to HSA (the Hpm arm).
fn hsa_adapted() -> CertifiedExecutableRecord {
    let adapter = adapter();
    let code = Extent {
        offset: 512,
        length: adapter.adapted_code_length,
        sha256: same(adapter.adapted_code_sha256.as_str()),
    };
    record_with(radv_producer(), Some(adapter), hsa_envelope(code))
}

/// hipcc code object: unchanged code, no adapter.
fn hsa_hipcc() -> CertifiedExecutableRecord {
    let code = Extent {
        offset: 512,
        length: 1024,
        sha256: same(raw().code.sha256.as_str()),
    };
    record_with(hipcc_producer(), None, hsa_envelope(code))
}

/// Original RADV pipeline binary (the Vpm arm).
fn radv_original() -> CertifiedExecutableRecord {
    record_with(
        radv_producer(),
        None,
        EnvelopeRecord::RadvOriginalBinary(RadvOriginalEnvelope {
            binary_key_sha256: h(31),
            blob_sha256: h(32),
            blob_length: 8192,
            global_key_sha256: h(33),
        }),
    )
}

fn radv_private() -> CertifiedExecutableRecord {
    let certificate = certificate(&raw());
    let envelope: PrivateEnvelopeDigest = h(34);
    let cache_key = derive_private_cache_key(&envelope, &certificate.certificate_sha256);
    record_with(
        radv_producer(),
        Some(adapter()),
        EnvelopeRecord::RadvPrivatePm(RadvPrivateEnvelope {
            private_envelope_sha256: envelope,
            private_envelope_length: 8192,
            private_icd_build_sha256: h(35),
            cache_key_sha256: cache_key,
        }),
    )
}

fn windows_original() -> CertifiedExecutableRecord {
    record_with(
        windows_producer(),
        None,
        EnvelopeRecord::WindowsDriverOriginalBinary(WindowsOriginalEnvelope {
            binary_key_sha256: h(36),
            blob_sha256: h(37),
            blob_length: 8192,
            global_key_sha256: h(38),
        }),
    )
}

fn all_records() -> Vec<CertifiedExecutableRecord> {
    vec![
        hsa_adapted(),
        hsa_hipcc(),
        radv_original(),
        radv_private(),
        windows_original(),
    ]
}

fn mutate_json(json: &str, change: impl FnOnce(&mut Value)) -> String {
    let mut value: Value = serde_json::from_str(json).unwrap();
    change(&mut value);
    value.to_string()
}

fn loaded_for(record: &CertifiedExecutableRecord) -> LoadedExecutableReceipt {
    let (code_sha, code_length) = record.executable_code();
    LoadedExecutableReceipt {
        schema_version: UNION_SCHEMA_VERSION,
        record_sha256: record.record_digest().unwrap(),
        target: record.target.clone(),
        executable_sha256: same(record.envelope.identity_hex()),
        upload: UploadRecord {
            path: UploadPath::VerbatimCopy,
            uploaded_length: code_length,
            uploaded_sha256: same(code_sha),
        },
        relocation: RelocationRecord {
            applied_count: 0,
            applied_sha256: None,
        },
        entry: LoadedEntry {
            name: record.raw.entry_name.clone(),
            offset: record.executable_entry_offset(),
            state_sha256: h(40),
        },
    }
}

fn surfaces() -> Vec<StateSurface> {
    vec![
        StateSurface {
            name: "out".to_owned(),
            kind: SurfaceKind::Output,
        },
        StateSurface {
            name: "guard".to_owned(),
            kind: SurfaceKind::Guard,
        },
        StateSurface {
            name: "weights".to_owned(),
            kind: SurfaceKind::Input,
        },
    ]
}

fn arm(arm: OracleArm, backend: Backend, identity: ArmIdentity, plan: u8) -> ArmBinding {
    ArmBinding {
        arm,
        backend,
        identity,
        runtime_plan_sha256: h(plan),
        pre_image_sha256: h(50),
        post_state_sha256: h(52),
    }
}

fn pair_receipt(
    hpm: &CertifiedExecutableRecord,
    vpm: &CertifiedExecutableRecord,
) -> PairOracleReceipt {
    let surfaces = surfaces();
    let claim = |lhs, rhs| PairClaim {
        lhs,
        rhs,
        compared_bytes: 4096,
        differing_bytes: 0,
    };
    PairOracleReceipt {
        schema_version: UNION_SCHEMA_VERSION,
        semantic_recipe_id: same(hpm.semantic.recipe_id.as_str()),
        target: target(),
        device_uuid: "GPU-0000000000000001".to_owned(),
        mode: ExecutionMode::Eager,
        fixture_sha256: h(50),
        state_surface_sha256: state_surface_digest(&surfaces).unwrap(),
        surfaces,
        arms: FourArms {
            v0: arm(
                OracleArm::V0,
                Backend::Vulkan,
                ArmIdentity::OriginalSpirv(OriginalSpirvArm {
                    spirv_sha256: h(11),
                    driver_build_sha256: h(13),
                }),
                51,
            ),
            hpm: arm(
                OracleArm::Hpm,
                Backend::Hip,
                ArmIdentity::Certified(CertifiedArm {
                    record_sha256: hpm.record_digest().unwrap(),
                    loaded_executable_sha256: same(hpm.envelope.identity_hex()),
                }),
                53,
            ),
            vpm: arm(
                OracleArm::Vpm,
                Backend::Vulkan,
                ArmIdentity::Certified(CertifiedArm {
                    record_sha256: vpm.record_digest().unwrap(),
                    loaded_executable_sha256: same(vpm.envelope.identity_hex()),
                }),
                54,
            ),
            hllvm: arm(
                OracleArm::Hllvm,
                Backend::Hip,
                ArmIdentity::HipccTwin(HipccTwinArm {
                    source_sha256: h(10),
                    code_object_sha256: h(55),
                }),
                56,
            ),
        },
        claims: vec![
            claim(OracleArm::V0, OracleArm::Hpm),
            claim(OracleArm::V0, OracleArm::Vpm),
            claim(OracleArm::Hpm, OracleArm::Hllvm),
            claim(OracleArm::Vpm, OracleArm::Hllvm),
        ],
        negative_control: NegativeControl {
            mutated_surface: "out".to_owned(),
            detected: true,
        },
    }
}

fn good_pair() -> (
    CertifiedExecutableRecord,
    CertifiedExecutableRecord,
    PairOracleReceipt,
) {
    let hpm = hsa_adapted();
    let vpm = radv_original();
    let receipt = pair_receipt(&hpm, &vpm);
    (hpm, vpm, receipt)
}

fn vulkan_record() -> VulkanExecutableRecord {
    let certified = radv_original();
    let raw = raw();
    let code = Extent {
        offset: 512,
        length: raw.code.length,
        sha256: same(raw.code.sha256.as_str()),
    };
    let data = raw.data.as_ref().map(|data| Extent {
        offset: 2048,
        length: data.length,
        sha256: same::<RawDataDigest>(data.sha256.as_str()),
    });
    let storage = |binding| DescriptorBinding {
        binding,
        descriptor_type: DescriptorType::StorageBuffer,
        count: 1,
    };
    let requirement = |binding, access, name: &str| BufferBindingRequirement {
        set: 0,
        binding,
        access,
        min_range_bytes: 4096,
        offset_alignment: 16,
        robustness: Robustness::RobustBufferAccess2,
        logical_resource: name.to_owned(),
    };
    VulkanExecutableRecord {
        schema_version: UNION_SCHEMA_VERSION,
        certified,
        pipeline_layout: PipelineLayoutRecord {
            layout_sha256: h(70),
            push_constant_bytes: 16,
            sets: vec![DescriptorSetLayout {
                set: 0,
                bindings: vec![storage(0), storage(1)],
            }],
        },
        specialization: SpecializationRecord {
            data_sha256: h(71),
            data_length: 4,
            entries: vec![SpecializationEntry {
                constant_id: 0,
                offset: 0,
                size: 4,
            }],
        },
        workgroup: WorkgroupContract {
            size: [64, 1, 1],
            shared_memory_bytes: 0,
        },
        subgroup: SubgroupContract {
            size: 32,
            full_subgroups_required: true,
        },
        extraction: ExtractionMapping {
            mapping_sha256: h(72),
            code,
            data,
        },
        buffer_bindings: vec![
            requirement(0, BufferAccess::Read, "weights"),
            requirement(1, BufferAccess::Write, "out"),
        ],
    }
}

fn view(generation: u64) -> SharedBufferViewRecord {
    SharedBufferViewRecord {
        schema_version: UNION_SCHEMA_VERSION,
        resource_id: 1,
        physical_range: PhysicalRange {
            allocation_id: 7,
            offset: 0,
            length: 4096,
        },
        generation,
        placement: Placement::DeviceLocal,
        hip_view_sha256: h(80),
        vulkan_view_sha256: h(81),
        layout_id: h(22),
    }
}

fn host_boundary() -> BoundaryContractRecord {
    BoundaryContractRecord {
        schema_version: UNION_SCHEMA_VERSION,
        boundary_id: 1,
        epoch: 1,
        producer: Endpoint {
            api: Api::Hip,
            queue_family: 0,
        },
        consumer: Endpoint {
            api: Api::Vulkan,
            queue_family: 0,
        },
        accesses: vec![RangeAccess {
            resource_id: 1,
            generation: 1,
            range: PhysicalRange {
                allocation_id: 7,
                offset: 0,
                length: 1024,
            },
            producer: AccessKind::Write,
            consumer: AccessKind::Read,
        }],
        visibility: vec![
            VisibilityOp::ProducerRelease,
            VisibilityOp::HostFenceWait,
            VisibilityOp::ConsumerAcquire,
        ],
        sync_mode: SyncMode::HostFenced,
        completion: CompletionTokenRecord {
            api: Api::Hip,
            epoch: 1,
            kind: CompletionKind::HostFence,
        },
        capability_receipt_sha256: None,
    }
}

fn semaphore_boundary() -> BoundaryContractRecord {
    BoundaryContractRecord {
        visibility: vec![VisibilityOp::SemaphoreSignal, VisibilityOp::SemaphoreWait],
        sync_mode: SyncMode::ExternalSemaphore,
        completion: CompletionTokenRecord {
            api: Api::Hip,
            epoch: 1,
            kind: CompletionKind::TimelineSemaphore,
        },
        capability_receipt_sha256: Some(h(62)),
        ..host_boundary()
    }
}

fn island(id: u32, backend: IslandBackend) -> IslandRecord {
    IslandRecord {
        island_id: id,
        backend,
        queue_family: 0,
        kernels: vec![IslandKernel {
            record_sha256: h(60),
            loaded_receipt_sha256: h(61),
        }],
    }
}

fn plan() -> UnionPlanRecord {
    UnionPlanRecord {
        schema_version: UNION_SCHEMA_VERSION,
        mode: UnionMode::HostFenced,
        cell: CellKey {
            semantic_recipe_id: h(20),
            target: target(),
            driver_build_sha256: Some(h(13)),
            shape_sha256: h(90),
            dtype_qtype: "mq4g256_f16".to_owned(),
            geometry: Geometry {
                grid: [128, 1, 1],
                workgroup: [64, 1, 1],
            },
            placement: Placement::DeviceLocal,
            execution_mode: ExecutionMode::Eager,
        },
        islands: vec![
            island(1, IslandBackend::Hip),
            island(2, IslandBackend::Vulkan),
        ],
        edges: vec![IslandEdge {
            from: 1,
            to: 2,
            boundary_id: Some(1),
        }],
        boundaries: vec![host_boundary()],
        resources: vec![view(1)],
    }
}

// ---------------------------------------------------------------------------
// Serialization round trips
// ---------------------------------------------------------------------------

#[test]
fn certified_records_round_trip_for_every_envelope_kind() {
    let kinds: Vec<EnvelopeKind> = all_records().iter().map(|r| r.envelope.kind()).collect();
    assert_eq!(
        kinds,
        [
            EnvelopeKind::Hsa,
            EnvelopeKind::Hsa,
            EnvelopeKind::RadvOriginalBinary,
            EnvelopeKind::RadvPrivatePm,
            EnvelopeKind::WindowsDriverOriginalBinary
        ]
    );
    for record in all_records() {
        let json = record.to_json().unwrap();
        let parsed = CertifiedExecutableRecord::from_json(&json).unwrap();
        assert_eq!(parsed, record);
        assert_eq!(
            parsed.record_digest().unwrap(),
            record.record_digest().unwrap()
        );
    }
}

#[test]
fn every_other_record_round_trips() {
    let (_, _, receipt) = good_pair();
    let json = receipt.to_json().unwrap();
    assert_eq!(PairOracleReceipt::from_json(&json).unwrap(), receipt);

    let loaded = loaded_for(&hsa_adapted());
    assert_eq!(
        LoadedExecutableReceipt::from_json(&loaded.to_json().unwrap()).unwrap(),
        loaded
    );

    let vulkan = vulkan_record();
    assert_eq!(
        VulkanExecutableRecord::from_json(&vulkan.to_json().unwrap()).unwrap(),
        vulkan
    );

    let shared = view(3);
    assert_eq!(
        SharedBufferViewRecord::from_json(&shared.to_json().unwrap()).unwrap(),
        shared
    );

    for boundary in [host_boundary(), semaphore_boundary()] {
        assert_eq!(
            BoundaryContractRecord::from_json(&boundary.to_json().unwrap()).unwrap(),
            boundary
        );
    }

    let plan = plan();
    assert_eq!(
        UnionPlanRecord::from_json(&plan.to_json().unwrap()).unwrap(),
        plan
    );
    let mut off = plan.clone();
    off.mode = UnionMode::Off;
    off.islands.clear();
    off.edges.clear();
    off.boundaries.clear();
    off.resources.clear();
    assert_eq!(
        UnionPlanRecord::from_json(&off.to_json().unwrap()).unwrap(),
        off
    );
}

#[test]
fn record_digest_is_stable_and_domain_separated() {
    let record = hsa_adapted();
    assert_eq!(
        record.record_digest().unwrap(),
        record.record_digest().unwrap()
    );
    let mut other = record.clone();
    other.raw.config_sha256 = h(99);
    assert_ne!(
        other.record_digest().unwrap(),
        record.record_digest().unwrap()
    );
    // The private cache key is derived, never equal to its inputs.
    let envelope: PrivateEnvelopeDigest = h(34);
    let cert: StaticCertificateDigest = h(6);
    let key = derive_private_cache_key(&envelope, &cert);
    assert_ne!(key.as_str(), envelope.as_str());
    assert_ne!(key.as_str(), cert.as_str());
}

// ---------------------------------------------------------------------------
// Strict schema: unknown fields, versions, malformed hashes
// ---------------------------------------------------------------------------

#[test]
fn unknown_fields_are_rejected_at_every_level() {
    let json = hsa_adapted().to_json().unwrap();
    for path in [
        "/",
        "/producer",
        "/raw",
        "/raw/code",
        "/envelope",
        "/adapter",
        "/certificate",
        "/target",
    ] {
        let mutated = mutate_json(&json, |value| {
            let node = if path == "/" {
                value
            } else {
                value.pointer_mut(path).unwrap()
            };
            node.as_object_mut()
                .unwrap()
                .insert("surprise".to_owned(), json!(1));
        });
        assert!(
            CertifiedExecutableRecord::from_json(&mutated).is_err(),
            "unknown field accepted at {path}"
        );
    }
    let plan_json = plan().to_json().unwrap();
    let mutated = mutate_json(&plan_json, |value| {
        value["boundaries"][0]["accesses"][0]
            .as_object_mut()
            .unwrap()
            .insert("surprise".to_owned(), json!(true));
    });
    assert!(UnionPlanRecord::from_json(&mutated).is_err());
    let (_, _, receipt) = good_pair();
    let mutated = mutate_json(&receipt.to_json().unwrap(), |value| {
        value["arms"]["hpm"]["identity"]
            .as_object_mut()
            .unwrap()
            .insert("surprise".to_owned(), json!(true));
    });
    assert!(PairOracleReceipt::from_json(&mutated).is_err());
}

#[test]
fn other_schema_versions_are_rejected() {
    let json = radv_original().to_json().unwrap();
    for version in [
        json!(0),
        json!(2),
        json!(UNION_SCHEMA_VERSION + 1),
        json!("1"),
    ] {
        let mutated = mutate_json(&json, |value| value["schema_version"] = version.clone());
        assert!(
            matches!(
                CertifiedExecutableRecord::from_json(&mutated),
                Err(UnionError::UnsupportedSchemaVersion { .. })
            ),
            "version {version} accepted"
        );
    }
    let missing = mutate_json(&json, |value| {
        value.as_object_mut().unwrap().remove("schema_version");
    });
    assert!(matches!(
        CertifiedExecutableRecord::from_json(&missing),
        Err(UnionError::MissingSchemaVersion { .. })
    ));
    // A programmatically built record with the wrong version fails validation too.
    let mut record = radv_original();
    record.schema_version = 9;
    assert!(matches!(
        record.validate(),
        Err(UnionError::UnsupportedSchemaVersion { .. })
    ));
    let mut shared = view(1);
    shared.schema_version = 2;
    assert!(shared.validate().is_err());
}

#[test]
fn malformed_hashes_are_rejected() {
    let json = hsa_adapted().to_json().unwrap();
    let cases = [
        ("uppercase", "AB".repeat(32)),
        ("short", "ab".repeat(31)),
        ("long", "ab".repeat(33)),
        ("non-hex", "zz".repeat(32)),
        ("all zero", "00".repeat(32)),
        ("empty", String::new()),
    ];
    for (name, bad) in cases {
        for path in [
            "/semantic/recipe_id",
            "/raw/sidecar_sha256",
            "/raw/config_sha256",
            "/raw/entry_abi_sha256",
            "/raw/code/sha256",
            "/certificate/certificate_sha256",
            "/adapter/prologue/sha256",
            "/envelope/image_sha256",
        ] {
            let mutated = mutate_json(&json, |value| {
                *value.pointer_mut(path).unwrap() = json!(bad.clone());
            });
            assert!(
                CertifiedExecutableRecord::from_json(&mutated).is_err(),
                "{name} hash accepted at {path}"
            );
        }
    }
    // Mesa pins are 40-hex git commits, not SHA-256.
    let radv = radv_original().to_json().unwrap();
    let mutated = mutate_json(&radv, |value| {
        *value.pointer_mut("/producer/mesa/commit").unwrap() = json!("ab".repeat(32));
    });
    assert!(CertifiedExecutableRecord::from_json(&mutated).is_err());
    assert!(GitCommit::parse(&"ab".repeat(20)).is_ok());
    assert!(GitCommit::parse(&"ab".repeat(32)).is_err());
}

// ---------------------------------------------------------------------------
// Sidecar, config, entry ABI: distinct, bound and byte-verifiable
// ---------------------------------------------------------------------------

#[test]
fn sidecar_config_and_entry_abi_are_distinct_bound_identities() {
    let base = hsa_adapted();
    let digest = base.record_digest().unwrap();
    let mut changed = base.clone();
    changed.raw.sidecar_sha256 = h(98);
    assert!(
        changed.validate().is_ok(),
        "sidecar is independent of the certificate"
    );
    assert_ne!(changed.record_digest().unwrap(), digest);

    let mut changed = base.clone();
    changed.raw.config_sha256 = h(98);
    assert!(
        changed.validate().is_ok(),
        "config is independent of the sidecar"
    );
    assert_ne!(changed.record_digest().unwrap(), digest);

    // The entry ABI is bound to the certificate: changing it alone must reject.
    let mut changed = base.clone();
    changed.raw.entry_abi_sha256 = h(98);
    assert!(matches!(
        changed.validate(),
        Err(UnionError::Mismatch { .. })
    ));

    // And so is the certified code.
    let mut changed = base;
    changed.raw.code.sha256 = h(98);
    assert!(matches!(
        changed.validate(),
        Err(UnionError::Mismatch { .. })
    ));
}

#[test]
fn raw_bytes_are_verified_independently_of_stored_hashes() {
    let mut stream = vec![0u8; 4096];
    for (index, byte) in stream.iter_mut().enumerate() {
        *byte = (index % 251) as u8;
    }
    let sidecar = b"{\"sidecar\":true}".to_vec();
    let mut artifact = raw();
    artifact.code.sha256 = RawCodeDigest::of_bytes(&stream[256..1280]);
    artifact.data.as_mut().unwrap().sha256 = RawDataDigest::of_bytes(&stream[2048..2176]);
    artifact.sidecar_sha256 = SidecarDigest::of_bytes(&sidecar);
    artifact.verify_bytes(&stream, &sidecar).unwrap();

    let mut flipped = stream.clone();
    flipped[300] ^= 1;
    assert!(artifact.verify_bytes(&flipped, &sidecar).is_err());
    let mut flipped = stream.clone();
    flipped[2100] ^= 1;
    assert!(artifact.verify_bytes(&flipped, &sidecar).is_err());
    assert!(
        artifact
            .verify_bytes(&stream, b"{\"sidecar\":false}")
            .is_err()
    );
    assert!(artifact.verify_bytes(&stream[..4000], &sidecar).is_err());
}

#[test]
fn raw_ranges_must_be_well_formed() {
    let mutations: Vec<(&str, Box<dyn Fn(&mut RawArtifact)>)> = vec![
        (
            "zero code length",
            Box::new(|r: &mut RawArtifact| r.code.length = 0),
        ),
        (
            "code past stream",
            Box::new(|r: &mut RawArtifact| r.code.offset = 4000),
        ),
        (
            "code overflow",
            Box::new(|r: &mut RawArtifact| r.code.offset = u64::MAX),
        ),
        (
            "data past stream",
            Box::new(|r: &mut RawArtifact| r.data.as_mut().unwrap().offset = 4090),
        ),
        (
            "data overlaps code",
            Box::new(|r: &mut RawArtifact| r.data.as_mut().unwrap().offset = 512),
        ),
        (
            "entry outside code",
            Box::new(|r: &mut RawArtifact| r.entry_offset = 1024),
        ),
        (
            "empty entry name",
            Box::new(|r: &mut RawArtifact| r.entry_name = " ".to_owned()),
        ),
        (
            "zero stream",
            Box::new(|r: &mut RawArtifact| r.total_length = 0),
        ),
    ];
    for (name, mutate) in mutations {
        let mut record = hsa_adapted();
        mutate(&mut record.raw);
        assert!(record.validate().is_err(), "{name} accepted");
    }
}

// ---------------------------------------------------------------------------
// Producer pins and target
// ---------------------------------------------------------------------------

#[test]
fn producer_pins_are_complete_for_the_producer_kind() {
    let mutations: Vec<(&str, Box<dyn Fn(&mut CertifiedExecutableRecord)>)> = vec![
        (
            "RADV without Mesa pin",
            Box::new(|r: &mut CertifiedExecutableRecord| r.producer.mesa = None),
        ),
        (
            "RADV without SPIR-V",
            Box::new(|r: &mut CertifiedExecutableRecord| r.producer.spirv_sha256 = None),
        ),
        (
            "RADV without driver",
            Box::new(|r: &mut CertifiedExecutableRecord| r.producer.driver = None),
        ),
        (
            "patched without digest",
            Box::new(|r: &mut CertifiedExecutableRecord| {
                r.producer.mesa.as_mut().unwrap().patch_sha256 = None
            }),
        ),
        (
            "unpatched with digest",
            Box::new(|r: &mut CertifiedExecutableRecord| {
                r.producer.mesa.as_mut().unwrap().patched = false
            }),
        ),
        (
            "empty compiler version",
            Box::new(|r: &mut CertifiedExecutableRecord| {
                r.producer.compiler.version = String::new()
            }),
        ),
    ];
    for (name, mutate) in mutations {
        let mut record = radv_original();
        mutate(&mut record);
        assert!(record.validate().is_err(), "{name} accepted");
    }
    let mut unpatched = radv_original();
    {
        let mesa = unpatched.producer.mesa.as_mut().unwrap();
        mesa.patched = false;
        mesa.patch_sha256 = None;
    }
    unpatched.validate().unwrap();

    // hipcc must not carry Mesa/SPIR-V/driver facts.
    let mut hipcc = hsa_hipcc();
    hipcc.validate().unwrap();
    hipcc.producer.mesa = radv_producer().mesa;
    assert!(hipcc.validate().is_err());
    let mut hipcc = hsa_hipcc();
    hipcc.producer.spirv_sha256 = Some(h(11));
    assert!(hipcc.validate().is_err());

    // A Windows driver is never a Mesa tree.
    let mut windows = windows_original();
    windows.validate().unwrap();
    windows.producer.mesa = radv_producer().mesa;
    assert!(windows.validate().is_err());
}

#[test]
fn target_must_be_exact_and_match_the_device() {
    for bad in [
        "gfx11",
        "gfx11-generic",
        "gfx1151:xnack+",
        "GFX1201",
        "",
        "navi48",
    ] {
        let mut record = radv_original();
        record.target.architecture = bad.to_owned();
        assert!(record.validate().is_err(), "`{bad}` accepted");
    }
    for good in [
        "gfx1010", "gfx1100", "gfx1151", "gfx1201", "gfx90a", "gfx942",
    ] {
        let mut record = radv_original();
        record.target.architecture = good.to_owned();
        record.validate().unwrap();
    }
    let record = radv_original();
    record.check_target(&target()).unwrap();
    let wrong_arch = UnionTarget {
        architecture: "gfx1151".to_owned(),
        wave: Wavefront::Wave32,
    };
    assert!(matches!(
        record.check_target(&wrong_arch),
        Err(UnionError::Mismatch { .. })
    ));
    let wrong_wave = UnionTarget {
        architecture: "gfx1201".to_owned(),
        wave: Wavefront::Wave64,
    };
    assert!(matches!(
        record.check_target(&wrong_wave),
        Err(UnionError::Mismatch { .. })
    ));
}

// ---------------------------------------------------------------------------
// Adapter body / prologue
// ---------------------------------------------------------------------------

#[test]
fn adapter_must_preserve_the_original_body() {
    let mutations: Vec<(&str, Box<dyn Fn(&mut AdapterRecord)>)> = vec![
        (
            "adapted body digest differs from raw body",
            Box::new(|a: &mut AdapterRecord| a.adapted_body.sha256 = h(99)),
        ),
        (
            "adapted body shorter",
            Box::new(|a: &mut AdapterRecord| a.adapted_body.length = 1023),
        ),
        (
            "raw body outside raw code",
            Box::new(|a: &mut AdapterRecord| a.raw_body.offset = 8),
        ),
        (
            "raw body empty",
            Box::new(|a: &mut AdapterRecord| a.raw_body.length = 0),
        ),
        (
            "prologue overlaps body",
            Box::new(|a: &mut AdapterRecord| a.prologue.length = 65),
        ),
        (
            "body past adapted code",
            Box::new(|a: &mut AdapterRecord| a.adapted_code_length = 1000),
        ),
        (
            "entry outside prologue",
            Box::new(|a: &mut AdapterRecord| a.entry_offset = 64),
        ),
        (
            "ranges overflow",
            Box::new(|a: &mut AdapterRecord| a.adapted_body.offset = u64::MAX),
        ),
        (
            "empty adapted code",
            Box::new(|a: &mut AdapterRecord| a.adapted_code_length = 0),
        ),
    ];
    for (name, mutate) in mutations {
        let mut record = hsa_adapted();
        mutate(record.adapter.as_mut().unwrap());
        assert!(record.validate().is_err(), "{name} accepted");
    }
}

#[test]
fn adapter_presence_follows_the_envelope_and_producer() {
    // Foreign (ACO) code in an HSA envelope needs an adapter.
    let mut record = hsa_adapted();
    record.adapter = None;
    assert!(matches!(
        record.validate(),
        Err(UnionError::Incomplete { .. })
    ));
    // Original binaries are never rewritten.
    let mut original = radv_original();
    original.adapter = Some(adapter());
    assert!(original.validate().is_err());
    let mut windows = windows_original();
    windows.adapter = Some(adapter());
    assert!(windows.validate().is_err());
    // Native/hipcc code is already HSA: nothing to adapt.
    let mut hipcc = hsa_hipcc();
    hipcc.adapter = Some(adapter());
    assert!(hipcc.validate().is_err());
    // Private import always rewrites the entry.
    let mut private = radv_private();
    private.adapter = None;
    assert!(private.validate().is_err());
}

#[test]
fn hsa_envelope_code_must_be_the_adapted_or_raw_code() {
    let mut record = hsa_adapted();
    if let EnvelopeRecord::Hsa(hsa) = &mut record.envelope {
        hsa.code.sha256 = h(99);
    }
    assert!(record.validate().is_err());
    let mut record = hsa_hipcc();
    if let EnvelopeRecord::Hsa(hsa) = &mut record.envelope {
        hsa.code.length = 512;
    }
    assert!(record.validate().is_err());
    let mut record = hsa_hipcc();
    if let EnvelopeRecord::Hsa(hsa) = &mut record.envelope {
        hsa.code.offset = 8192;
    }
    assert!(record.validate().is_err());
}

// ---------------------------------------------------------------------------
// Envelope compatibility
// ---------------------------------------------------------------------------

#[test]
fn envelope_compatibility_rejects_wrong_kind_and_global_key() {
    let radv = radv_original();
    let ok = EnvelopeExpectation::RadvOriginalBinary {
        global_key_sha256: h(33),
    };
    radv.check_envelope_compat(&ok).unwrap();
    let wrong_key = EnvelopeExpectation::RadvOriginalBinary {
        global_key_sha256: h(99),
    };
    assert!(matches!(
        radv.check_envelope_compat(&wrong_key),
        Err(UnionError::EnvelopeIncompatible(_))
    ));
    // A RADV blob is not an HSA image, a private PM envelope or a Windows blob.
    for expected in [
        EnvelopeExpectation::Hsa,
        EnvelopeExpectation::RadvPrivatePm {
            private_icd_build_sha256: h(35),
        },
        EnvelopeExpectation::WindowsDriverOriginalBinary {
            global_key_sha256: h(38),
        },
    ] {
        assert!(matches!(
            radv.check_envelope_compat(&expected),
            Err(UnionError::EnvelopeIncompatible(_))
        ));
    }
    hsa_adapted()
        .check_envelope_compat(&EnvelopeExpectation::Hsa)
        .unwrap();
    assert!(
        hsa_adapted().check_envelope_compat(&ok).is_err(),
        "HSA must not satisfy a RADV host"
    );

    let private = radv_private();
    private
        .check_envelope_compat(&EnvelopeExpectation::RadvPrivatePm {
            private_icd_build_sha256: h(35),
        })
        .unwrap();
    assert!(
        private
            .check_envelope_compat(&EnvelopeExpectation::RadvPrivatePm {
                private_icd_build_sha256: h(99),
            })
            .is_err()
    );

    let windows = windows_original();
    windows
        .check_envelope_compat(&EnvelopeExpectation::WindowsDriverOriginalBinary {
            global_key_sha256: h(38),
        })
        .unwrap();
    assert!(
        windows
            .check_envelope_compat(&EnvelopeExpectation::WindowsDriverOriginalBinary {
                global_key_sha256: h(99),
            })
            .is_err()
    );
    // Windows keys are a different namespace from RADV even when equal as text.
    assert!(
        windows
            .check_envelope_compat(&EnvelopeExpectation::RadvOriginalBinary {
                global_key_sha256: same(windows_global_key_hex().as_str()),
            })
            .is_err()
    );
}

fn windows_global_key_hex() -> String {
    match windows_original().envelope {
        EnvelopeRecord::WindowsDriverOriginalBinary(w) => w.global_key_sha256.as_str().to_owned(),
        _ => unreachable!(),
    }
}

#[test]
fn envelope_kind_must_match_the_producer() {
    // hipcc code cannot be an "original RADV binary".
    let mut record = radv_original();
    record.producer = hipcc_producer();
    assert!(matches!(
        record.validate(),
        Err(UnionError::EnvelopeIncompatible(_))
    ));
    // RADV code cannot be an "original Windows binary".
    let mut record = windows_original();
    record.producer = radv_producer();
    assert!(matches!(
        record.validate(),
        Err(UnionError::EnvelopeIncompatible(_))
    ));
}

#[test]
fn private_cache_key_must_derive_from_envelope_and_certificate() {
    let mut record = radv_private();
    if let EnvelopeRecord::RadvPrivatePm(private) = &mut record.envelope {
        private.cache_key_sha256 = h(99);
    }
    assert!(matches!(
        record.validate(),
        Err(UnionError::Mismatch { .. })
    ));

    // Re-certifying with a different certificate invalidates the old key.
    let mut record = radv_private();
    record.certificate.certificate_sha256 = h(97);
    assert!(record.validate().is_err());

    // The key never reuses a stock-looking value of the original envelope.
    let mut record = radv_private();
    if let EnvelopeRecord::RadvPrivatePm(private) = &mut record.envelope {
        private.private_envelope_sha256 = h(96);
    }
    assert!(record.validate().is_err());

    let mut record = radv_private();
    if let EnvelopeRecord::RadvPrivatePm(private) = &mut record.envelope {
        private.private_envelope_length = 0;
    }
    assert!(record.validate().is_err());
}

// ---------------------------------------------------------------------------
// Loaded executable linkage
// ---------------------------------------------------------------------------

#[test]
fn loaded_receipt_binds_to_exactly_the_certified_record() {
    for record in all_records() {
        let loaded = loaded_for(&record);
        loaded.bind_to(&record).unwrap();
        loaded.receipt_digest().unwrap();
    }
    let record = hsa_adapted();
    let good = loaded_for(&record);

    let mutations: Vec<(&str, Box<dyn Fn(&mut LoadedExecutableReceipt)>)> = vec![
        (
            "different record",
            Box::new(|l: &mut LoadedExecutableReceipt| l.record_sha256 = h(99)),
        ),
        (
            "baseline fallback executable",
            Box::new(|l: &mut LoadedExecutableReceipt| l.executable_sha256 = h(99)),
        ),
        (
            "uploaded code digest",
            Box::new(|l: &mut LoadedExecutableReceipt| l.upload.uploaded_sha256 = h(99)),
        ),
        (
            "uploaded length",
            Box::new(|l: &mut LoadedExecutableReceipt| l.upload.uploaded_length += 4),
        ),
        (
            "entry symbol",
            Box::new(|l: &mut LoadedExecutableReceipt| l.entry.name = "other".to_owned()),
        ),
        (
            "entry offset",
            Box::new(|l: &mut LoadedExecutableReceipt| l.entry.offset = 4),
        ),
        (
            "device arch",
            Box::new(|l: &mut LoadedExecutableReceipt| {
                l.target.architecture = "gfx1151".to_owned()
            }),
        ),
        (
            "device wave",
            Box::new(|l: &mut LoadedExecutableReceipt| l.target.wave = Wavefront::Wave64),
        ),
        (
            "relocated load",
            Box::new(|l: &mut LoadedExecutableReceipt| {
                l.upload.path = UploadPath::RelocatedLoad;
                l.relocation = RelocationRecord {
                    applied_count: 2,
                    applied_sha256: Some(h(41)),
                };
            }),
        ),
        (
            "unrecorded relocation",
            Box::new(|l: &mut LoadedExecutableReceipt| l.relocation.applied_count = 1),
        ),
        (
            "relocation digest without count",
            Box::new(|l: &mut LoadedExecutableReceipt| l.relocation.applied_sha256 = Some(h(41))),
        ),
        (
            "relocated path without relocations",
            Box::new(|l: &mut LoadedExecutableReceipt| l.upload.path = UploadPath::RelocatedLoad),
        ),
        (
            "empty upload",
            Box::new(|l: &mut LoadedExecutableReceipt| l.upload.uploaded_length = 0),
        ),
    ];
    for (name, mutate) in mutations {
        let mut loaded = good.clone();
        mutate(&mut loaded);
        assert!(loaded.bind_to(&record).is_err(), "{name} accepted");
    }

    // A receipt for the HSA envelope cannot satisfy the Vulkan record.
    assert!(good.bind_to(&radv_original()).is_err());

    // Editing the record after the receipt was made breaks the link.
    let mut edited = record.clone();
    edited.raw.config_sha256 = h(99);
    assert!(good.bind_to(&edited).is_err());
}

#[test]
fn loaded_entry_follows_the_adapter_entry_when_present() {
    let record = hsa_adapted();
    let mut loaded = loaded_for(&record);
    // The original (unadapted) entry offset is wrong for adapted code
    // whenever the prologue entry differs.
    let mut shifted = record.clone();
    shifted.adapter.as_mut().unwrap().entry_offset = 8;
    shifted.validate().unwrap();
    loaded.record_sha256 = shifted.record_digest().unwrap();
    assert!(loaded.bind_to(&shifted).is_err());
    loaded.entry.offset = 8;
    loaded.bind_to(&shifted).unwrap();
}

// ---------------------------------------------------------------------------
// Four-arm oracle receipt
// ---------------------------------------------------------------------------

#[test]
fn pair_receipt_validates_and_binds_certified_records() {
    let (hpm, vpm, receipt) = good_pair();
    receipt.validate().unwrap();
    receipt.validate_exact().unwrap();
    receipt.bind_certified(&hpm, &vpm).unwrap();
    assert!(receipt.all_pairs_exact());
}

#[test]
fn pair_receipt_binding_failures() {
    let (hpm, vpm, receipt) = good_pair();

    // Swapped records: Hpm must be HSA, Vpm must be Vulkan.
    assert!(receipt.bind_certified(&vpm, &hpm).is_err());
    // A different (edited) record fails the digest binding.
    let mut edited = hpm.clone();
    edited.raw.sidecar_sha256 = h(99);
    assert!(receipt.bind_certified(&edited, &vpm).is_err());
    let mut edited = vpm.clone();
    edited.producer.options_sha256 = h(99);
    assert!(receipt.bind_certified(&hpm, &edited).is_err());
    // Wrong target / recipe on the receipt.
    let mut wrong = receipt.clone();
    wrong.target.architecture = "gfx1151".to_owned();
    assert!(wrong.bind_certified(&hpm, &vpm).is_err());
    let mut wrong = receipt.clone();
    wrong.semantic_recipe_id = h(99);
    assert!(wrong.bind_certified(&hpm, &vpm).is_err());
    // Loaded executable named by the arm is not the record's envelope.
    let mut wrong = receipt.clone();
    wrong.arms.hpm.identity = ArmIdentity::Certified(CertifiedArm {
        record_sha256: hpm.record_digest().unwrap(),
        loaded_executable_sha256: h(99),
    });
    assert!(wrong.bind_certified(&hpm, &vpm).is_err());
    // Hpm as a Vulkan record is a different pair entirely.
    let vpm_private = radv_private();
    let private_receipt = pair_receipt(&hpm, &vpm_private);
    private_receipt.bind_certified(&hpm, &vpm_private).unwrap();
    let windows = windows_original();
    pair_receipt(&hpm, &windows)
        .bind_certified(&hpm, &windows)
        .unwrap();
}

#[test]
fn pair_receipt_structural_mutations_are_rejected() {
    let (_, _, receipt) = good_pair();
    let mutations: Vec<(&str, Box<dyn Fn(&mut PairOracleReceipt)>)> = vec![
        (
            "fixture",
            Box::new(|r: &mut PairOracleReceipt| r.fixture_sha256 = h(99)),
        ),
        (
            "arm pre-image",
            Box::new(|r: &mut PairOracleReceipt| r.arms.vpm.pre_image_sha256 = h(99)),
        ),
        (
            "arm slot",
            Box::new(|r: &mut PairOracleReceipt| r.arms.hpm.arm = OracleArm::Vpm),
        ),
        (
            "backend",
            Box::new(|r: &mut PairOracleReceipt| r.arms.hllvm.backend = Backend::Vulkan),
        ),
        (
            "identity kind",
            Box::new(|r: &mut PairOracleReceipt| {
                r.arms.v0.identity = ArmIdentity::HipccTwin(HipccTwinArm {
                    source_sha256: h(10),
                    code_object_sha256: h(55),
                });
            }),
        ),
        (
            "state surface digest",
            Box::new(|r: &mut PairOracleReceipt| r.state_surface_sha256 = h(99)),
        ),
        (
            "surface list changed",
            Box::new(|r: &mut PairOracleReceipt| {
                r.surfaces.push(StateSurface {
                    name: "extra".to_owned(),
                    kind: SurfaceKind::State,
                });
            }),
        ),
        (
            "duplicate surface",
            Box::new(|r: &mut PairOracleReceipt| {
                r.surfaces.push(r.surfaces[0].clone());
                r.state_surface_sha256 = state_surface_digest(&r.surfaces).unwrap();
            }),
        ),
        (
            "no output surface",
            Box::new(|r: &mut PairOracleReceipt| {
                r.surfaces.retain(|s| s.kind != SurfaceKind::Output);
                r.state_surface_sha256 = state_surface_digest(&r.surfaces).unwrap();
            }),
        ),
        (
            "missing claim",
            Box::new(|r: &mut PairOracleReceipt| {
                r.claims.pop();
            }),
        ),
        (
            "duplicate claim",
            Box::new(|r: &mut PairOracleReceipt| {
                let first = r.claims[0].clone();
                r.claims[3] = first;
            }),
        ),
        (
            "extra claim",
            Box::new(|r: &mut PairOracleReceipt| {
                r.claims.push(PairClaim {
                    lhs: OracleArm::Hpm,
                    rhs: OracleArm::Vpm,
                    compared_bytes: 1,
                    differing_bytes: 0,
                });
            }),
        ),
        (
            "claim says differing but states equal",
            Box::new(|r: &mut PairOracleReceipt| r.claims[1].differing_bytes = 3),
        ),
        (
            "claim says equal but states differ",
            Box::new(|r: &mut PairOracleReceipt| r.arms.vpm.post_state_sha256 = h(99)),
        ),
        (
            "zero compared bytes",
            Box::new(|r: &mut PairOracleReceipt| r.claims[0].compared_bytes = 0),
        ),
        (
            "differing exceeds compared",
            Box::new(|r: &mut PairOracleReceipt| {
                r.claims[0].differing_bytes = 9999;
                r.arms.hpm.post_state_sha256 = h(99);
            }),
        ),
        (
            "negative control on unbound surface",
            Box::new(|r: &mut PairOracleReceipt| {
                r.negative_control.mutated_surface = "nowhere".to_owned()
            }),
        ),
        (
            "empty device",
            Box::new(|r: &mut PairOracleReceipt| r.device_uuid.clear()),
        ),
        (
            "bad target",
            Box::new(|r: &mut PairOracleReceipt| r.target.architecture = "gfx11".to_owned()),
        ),
    ];
    for (name, mutate) in mutations {
        let mut mutated = receipt.clone();
        mutate(&mut mutated);
        assert!(mutated.validate().is_err(), "{name} accepted");
    }
}

#[test]
fn pair_receipt_requires_four_distinct_arms_in_json() {
    let (_, _, receipt) = good_pair();
    let json = receipt.to_json().unwrap();
    for arm in ["v0", "hpm", "vpm", "hllvm"] {
        let mutated = mutate_json(&json, |value| {
            value["arms"].as_object_mut().unwrap().remove(arm);
        });
        assert!(
            PairOracleReceipt::from_json(&mutated).is_err(),
            "missing {arm}"
        );
    }
    let mutated = mutate_json(&json, |value| {
        value["arms"]
            .as_object_mut()
            .unwrap()
            .insert("fifth".to_owned(), json!({}));
    });
    assert!(PairOracleReceipt::from_json(&mutated).is_err());
}

#[test]
fn exactness_requires_zero_differing_bytes_and_a_caught_control() {
    let (_, _, mut receipt) = good_pair();
    receipt.negative_control.detected = false;
    receipt.validate().unwrap();
    assert!(receipt.validate_exact().is_err());

    let (_, _, mut receipt) = good_pair();
    // A real, recorded mismatch is a valid receipt but not an exact one.
    receipt.arms.vpm.post_state_sha256 = h(99);
    receipt.claims[1].differing_bytes = 17;
    receipt.claims[3].differing_bytes = 17;
    receipt.validate().unwrap();
    assert!(!receipt.all_pairs_exact());
    assert!(receipt.validate_exact().is_err());
}

#[test]
fn execution_mode_and_plan_identity_change_the_receipt() {
    let (_, _, receipt) = good_pair();
    let mut other = receipt.clone();
    other.mode = ExecutionMode::HipGraph;
    assert_ne!(other, receipt);
    assert_ne!(
        other.to_json().unwrap(),
        receipt.to_json().unwrap(),
        "mode is part of the serialized identity"
    );
    let mut plan_changed = receipt.clone();
    plan_changed.arms.hpm.runtime_plan_sha256 = h(99);
    plan_changed.validate().unwrap();
    assert_ne!(plan_changed.to_json().unwrap(), receipt.to_json().unwrap());
    let plan_digest = plan().runtime_plan_digest().unwrap();
    let mut changed = plan();
    changed.cell.dtype_qtype = "other".to_owned();
    assert_ne!(changed.runtime_plan_digest().unwrap(), plan_digest);
}

// ---------------------------------------------------------------------------
// U-F2 Vulkan record
// ---------------------------------------------------------------------------

#[test]
fn vulkan_record_validates_and_rejects_mismatched_contracts() {
    vulkan_record().validate().unwrap();

    let mutations: Vec<(&str, Box<dyn Fn(&mut VulkanExecutableRecord)>)> = vec![
        (
            "extraction code digest",
            Box::new(|v: &mut VulkanExecutableRecord| v.extraction.code.sha256 = h(99)),
        ),
        (
            "extraction code length",
            Box::new(|v: &mut VulkanExecutableRecord| v.extraction.code.length = 1000),
        ),
        (
            "extraction past blob",
            Box::new(|v: &mut VulkanExecutableRecord| v.extraction.code.offset = 8000),
        ),
        (
            "extraction data missing",
            Box::new(|v: &mut VulkanExecutableRecord| v.extraction.data = None),
        ),
        (
            "extraction data overlaps code",
            Box::new(|v: &mut VulkanExecutableRecord| {
                v.extraction.data.as_mut().unwrap().offset = 600
            }),
        ),
        (
            "extraction data digest",
            Box::new(|v: &mut VulkanExecutableRecord| {
                v.extraction.data.as_mut().unwrap().sha256 = h(99)
            }),
        ),
        (
            "subgroup vs wave",
            Box::new(|v: &mut VulkanExecutableRecord| v.subgroup.size = 64),
        ),
        (
            "workgroup not full subgroups",
            Box::new(|v: &mut VulkanExecutableRecord| v.workgroup.size = [48, 1, 1]),
        ),
        (
            "workgroup too large",
            Box::new(|v: &mut VulkanExecutableRecord| v.workgroup.size = [1024, 2, 1]),
        ),
        (
            "workgroup zero",
            Box::new(|v: &mut VulkanExecutableRecord| v.workgroup.size = [64, 0, 1]),
        ),
        (
            "specialization duplicate id",
            Box::new(|v: &mut VulkanExecutableRecord| {
                v.specialization.entries.push(SpecializationEntry {
                    constant_id: 0,
                    offset: 0,
                    size: 4,
                });
            }),
        ),
        (
            "specialization outside data",
            Box::new(|v: &mut VulkanExecutableRecord| v.specialization.entries[0].size = 8),
        ),
        (
            "binding missing from layout",
            Box::new(|v: &mut VulkanExecutableRecord| v.buffer_bindings[0].binding = 5),
        ),
        (
            "layout binding without requirement",
            Box::new(|v: &mut VulkanExecutableRecord| {
                v.buffer_bindings.pop();
            }),
        ),
        (
            "duplicate requirement",
            Box::new(|v: &mut VulkanExecutableRecord| v.buffer_bindings[1].binding = 0),
        ),
        (
            "zero range",
            Box::new(|v: &mut VulkanExecutableRecord| v.buffer_bindings[0].min_range_bytes = 0),
        ),
        (
            "alignment not power of two",
            Box::new(|v: &mut VulkanExecutableRecord| v.buffer_bindings[0].offset_alignment = 24),
        ),
        (
            "duplicate set",
            Box::new(|v: &mut VulkanExecutableRecord| {
                let set = v.pipeline_layout.sets[0].clone();
                v.pipeline_layout.sets.push(set);
            }),
        ),
        (
            "zero descriptor count",
            Box::new(|v: &mut VulkanExecutableRecord| {
                v.pipeline_layout.sets[0].bindings[0].count = 0
            }),
        ),
        (
            "HSA envelope",
            Box::new(|v: &mut VulkanExecutableRecord| v.certified = hsa_adapted()),
        ),
    ];
    for (name, mutate) in mutations {
        let mut record = vulkan_record();
        mutate(&mut record);
        assert!(record.validate().is_err(), "{name} accepted");
    }
}

#[test]
fn vulkan_private_and_windows_modes_keep_their_own_envelopes() {
    let mut private = vulkan_record();
    private.certified = radv_private();
    // Private mode maps the adapted code, not the raw code.
    assert!(private.validate().is_err());
    let adapted = private.certified.executable_code().0.to_owned();
    private.extraction.code.sha256 = same(&adapted);
    private.extraction.code.length = 1088;
    private.validate().unwrap();

    let mut windows = vulkan_record();
    windows.certified = windows_original();
    windows.validate().unwrap();
}

// ---------------------------------------------------------------------------
// U-F3 / U-F4
// ---------------------------------------------------------------------------

#[test]
fn boundary_contracts_require_consistent_synchronization() {
    host_boundary().validate().unwrap();
    semaphore_boundary().validate().unwrap();

    let mutations: Vec<(&str, Box<dyn Fn(&mut BoundaryContractRecord)>)> = vec![
        (
            "zero epoch",
            Box::new(|b: &mut BoundaryContractRecord| b.epoch = 0),
        ),
        (
            "same API both sides",
            Box::new(|b: &mut BoundaryContractRecord| b.consumer.api = Api::Hip),
        ),
        (
            "no accesses",
            Box::new(|b: &mut BoundaryContractRecord| b.accesses.clear()),
        ),
        (
            "empty range",
            Box::new(|b: &mut BoundaryContractRecord| b.accesses[0].range.length = 0),
        ),
        (
            "host fence missing",
            Box::new(|b: &mut BoundaryContractRecord| {
                b.visibility = vec![VisibilityOp::ProducerRelease]
            }),
        ),
        (
            "semaphore in host mode",
            Box::new(|b: &mut BoundaryContractRecord| {
                b.visibility.push(VisibilityOp::SemaphoreSignal)
            }),
        ),
        (
            "duplicate op",
            Box::new(|b: &mut BoundaryContractRecord| {
                b.visibility.push(VisibilityOp::ProducerRelease)
            }),
        ),
        (
            "host mode semaphore completion",
            Box::new(|b: &mut BoundaryContractRecord| {
                b.completion.kind = CompletionKind::BinarySemaphore
            }),
        ),
        (
            "completion epoch",
            Box::new(|b: &mut BoundaryContractRecord| b.completion.epoch = 2),
        ),
        (
            "completion of the other API",
            Box::new(|b: &mut BoundaryContractRecord| b.completion.api = Api::Vulkan),
        ),
    ];
    for (name, mutate) in mutations {
        let mut boundary = host_boundary();
        mutate(&mut boundary);
        assert!(boundary.validate().is_err(), "{name} accepted");
    }

    let mut async_boundary = semaphore_boundary();
    async_boundary.capability_receipt_sha256 = None;
    assert!(matches!(
        async_boundary.validate(),
        Err(UnionError::Incomplete { .. })
    ));
    let mut async_boundary = semaphore_boundary();
    async_boundary.visibility.push(VisibilityOp::HostFenceWait);
    assert!(async_boundary.validate().is_err());
    let mut async_boundary = semaphore_boundary();
    async_boundary.completion.kind = CompletionKind::HostFence;
    assert!(async_boundary.validate().is_err());

    let mut view = view(1);
    view.generation = 0;
    assert!(view.validate().is_err());
    let mut view = self::view(1);
    view.physical_range.length = 0;
    assert!(view.validate().is_err());
}

#[test]
fn union_plan_validates_modes_edges_and_generations() {
    plan().validate().unwrap();
    plan().runtime_plan_digest().unwrap();

    let mut async_plan = plan();
    async_plan.mode = UnionMode::Async;
    assert!(
        async_plan.validate().is_err(),
        "async plan with a host-fenced boundary"
    );
    async_plan.boundaries = vec![semaphore_boundary()];
    async_plan.validate().unwrap();
    let mut host_plan = plan();
    host_plan.boundaries = vec![semaphore_boundary()];
    assert!(host_plan.validate().is_err());

    let mutations: Vec<(&str, Box<dyn Fn(&mut UnionPlanRecord)>)> = vec![
        (
            "off with islands",
            Box::new(|p: &mut UnionPlanRecord| p.mode = UnionMode::Off),
        ),
        (
            "no islands",
            Box::new(|p: &mut UnionPlanRecord| p.islands.clear()),
        ),
        (
            "duplicate island",
            Box::new(|p: &mut UnionPlanRecord| p.islands[1].island_id = 1),
        ),
        (
            "island without kernel",
            Box::new(|p: &mut UnionPlanRecord| p.islands[0].kernels.clear()),
        ),
        (
            "cross-API edge without boundary",
            Box::new(|p: &mut UnionPlanRecord| p.edges[0].boundary_id = None),
        ),
        (
            "unknown boundary",
            Box::new(|p: &mut UnionPlanRecord| p.edges[0].boundary_id = Some(9)),
        ),
        (
            "same-API edge with boundary",
            Box::new(|p: &mut UnionPlanRecord| p.islands[1].backend = IslandBackend::Railgun),
        ),
        (
            "boundary direction",
            Box::new(|p: &mut UnionPlanRecord| {
                p.edges = vec![IslandEdge {
                    from: 2,
                    to: 1,
                    boundary_id: Some(1),
                }]
            }),
        ),
        (
            "unknown island",
            Box::new(|p: &mut UnionPlanRecord| p.edges[0].to = 9),
        ),
        (
            "self edge",
            Box::new(|p: &mut UnionPlanRecord| p.edges[0].to = 1),
        ),
        (
            "unused boundary",
            Box::new(|p: &mut UnionPlanRecord| {
                let mut extra = host_boundary();
                extra.boundary_id = 2;
                p.boundaries.push(extra);
            }),
        ),
        (
            "duplicate boundary",
            Box::new(|p: &mut UnionPlanRecord| p.boundaries.push(host_boundary())),
        ),
        (
            "stale generation",
            Box::new(|p: &mut UnionPlanRecord| p.resources[0].generation = 2),
        ),
        (
            "duplicate resource generation",
            Box::new(|p: &mut UnionPlanRecord| p.resources.push(view(1))),
        ),
        (
            "range outside resource",
            Box::new(|p: &mut UnionPlanRecord| p.boundaries[0].accesses[0].range.length = 8192),
        ),
        (
            "range in another allocation",
            Box::new(|p: &mut UnionPlanRecord| p.boundaries[0].accesses[0].range.allocation_id = 8),
        ),
        (
            "zero geometry",
            Box::new(|p: &mut UnionPlanRecord| p.cell.geometry.grid[0] = 0),
        ),
        (
            "bad cell target",
            Box::new(|p: &mut UnionPlanRecord| p.cell.target.architecture = "gfx11".to_owned()),
        ),
        (
            "empty qtype",
            Box::new(|p: &mut UnionPlanRecord| p.cell.dtype_qtype.clear()),
        ),
    ];
    for (name, mutate) in mutations {
        let mut mutated = plan();
        mutate(&mut mutated);
        assert!(mutated.validate().is_err(), "{name} accepted");
    }
}

#[test]
fn union_plan_rejects_cycles() {
    let mut cyclic = plan();
    cyclic.islands.push(island(3, IslandBackend::Hip));
    cyclic.edges = vec![
        IslandEdge {
            from: 1,
            to: 3,
            boundary_id: None,
        },
        IslandEdge {
            from: 3,
            to: 1,
            boundary_id: None,
        },
        IslandEdge {
            from: 1,
            to: 2,
            boundary_id: Some(1),
        },
    ];
    assert!(matches!(
        cyclic.validate(),
        Err(UnionError::Malformed { .. })
    ));
    let mut dag = plan();
    dag.islands.push(island(3, IslandBackend::Hip));
    dag.edges.push(IslandEdge {
        from: 1,
        to: 3,
        boundary_id: None,
    });
    dag.validate().unwrap();
}

// ---------------------------------------------------------------------------
// Existing manifests are unchanged
// ---------------------------------------------------------------------------

fn old_manifest() -> Value {
    json!({
        "schema_version": 3,
        "compiler": "radiowave",
        "generated_unix_seconds": 0,
        "source": "kernel.hip",
        "output": "kernel.hsaco",
        "arch": "gfx1201",
        "wavefront": "wave64",
        "scheduler_profile": "default",
        "hipcc": "hipcc",
        "hipcc_version": "test",
        "command": [],
        "source_sha256": "source",
        "support_header_sha256": "header",
        "output_sha256": "output",
        "inspection": null
    })
}

#[test]
fn existing_manifests_deserialize_and_serialize_unchanged() {
    let original = old_manifest();
    let manifest: CompileManifest = serde_json::from_value(original.clone()).unwrap();
    assert_eq!(serde_json::to_value(&manifest).unwrap(), original);
    // No union field appears, and the schema version is untouched.
    assert!(!original.to_string().contains("union"));
    assert_eq!(manifest.schema_version, 3);
}

#[test]
fn old_and_union_schemas_cannot_masquerade_as_each_other() {
    // An old manifest is not a union record: it has no union fields and its
    // schema_version of 3 is not a union version.
    let manifest = old_manifest().to_string();
    assert!(CertifiedExecutableRecord::from_json(&manifest).is_err());
    assert!(PairOracleReceipt::from_json(&manifest).is_err());
    // A union record is not a CompileManifest.
    let union_json = radv_original().to_json().unwrap();
    assert!(serde_json::from_str::<CompileManifest>(&union_json).is_err());
}
