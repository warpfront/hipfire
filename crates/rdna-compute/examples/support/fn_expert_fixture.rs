//! Shared no-model Flash-Next expert-indirection fixtures and replay checks.
//! Fixture layout follows Kaden Schutt's existing moe_few.rs tests; this
//! original lab oracle uses unchanged production launchers.

use rdna_compute::replay::{PreparedReplayIdentity, RecordedHipLaunch, ReplayQuiescence};
use rdna_compute::{Gpu, GpuTensor};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

pub(crate) type R<T> = Result<T, String>;
pub(crate) const E: usize = 512;
pub(crate) const TOP: usize = 10;
pub(crate) const DISTINCT: usize = 3;
pub(crate) const HIDDEN: usize = 2560;
pub(crate) const MI: usize = 640;
pub(crate) const GU_M: usize = 2 * MI;
pub(crate) const GU_STRIDE: usize = GU_M * (HIDDEN / 256) * 136;
pub(crate) const DN_STRIDE: usize = HIDDEN * (MI / 128) * 68;
pub(crate) const GU_ELEMS: usize = TOP * MI;
pub(crate) const DN_ELEMS: usize = TOP * HIDDEN;
pub(crate) const GUARD: usize = 64;
pub(crate) const POISON: u32 = 0x7fc0_1234;
pub(crate) const GU_SYMBOL: &str = "gemv_mq4g256v2_moe_gate_up_k2560_rows8_indexed_batched";
pub(crate) const DN_SYMBOL: &str = "gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded";

pub(crate) fn check(ok: bool, message: impl Into<String>) -> R<()> {
    if ok {
        Ok(())
    } else {
        Err(message.into())
    }
}
pub(crate) fn hip<T, X: std::fmt::Display>(result: Result<T, X>) -> R<T> {
    result.map_err(|e| e.to_string())
}
pub(crate) fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
pub(crate) fn launch_count() -> u64 {
    hip_bridge::launch_counters::launch_kernel::count()
}
pub(crate) fn graph_count() -> u64 {
    hip_bridge::launch_counters::graph_launch::count()
}

#[derive(Default)]
pub(crate) struct Arena {
    pub(crate) roots: Vec<GpuTensor>,
}
impl Arena {
    pub(crate) fn own(&mut self, tensor: GpuTensor) -> usize {
        self.roots.push(tensor);
        self.roots.len() - 1
    }
    pub(crate) fn get(&self, id: usize) -> &GpuTensor {
        &self.roots[id]
    }
    pub(crate) fn raw(&mut self, gpu: &mut Gpu, bytes: &[u8]) -> R<usize> {
        let t = hip(gpu.upload_raw(bytes, &[bytes.len()]))?;
        Ok(self.own(t))
    }
    pub(crate) fn f32(&mut self, gpu: &mut Gpu, values: &[f32]) -> R<usize> {
        let t = hip(gpu.upload_f32(values, &[values.len()]))?;
        Ok(self.own(t))
    }
    pub(crate) fn release(&mut self, gpu: &mut Gpu) -> R<()> {
        let mut errors = Vec::new();
        while let Some(t) = self.roots.pop() {
            if let Err(e) = gpu.free_tensor(t) {
                errors.push(e.to_string());
            }
        }
        check(errors.is_empty(), format!("allocation cleanup: {errors:?}"))
    }
}

pub(crate) struct Guarded {
    pub(crate) root: usize,
    pub(crate) view: GpuTensor,
    pub(crate) elems: usize,
}
impl Guarded {
    pub(crate) fn new(gpu: &mut Gpu, arena: &mut Arena, elems: usize) -> R<Self> {
        let values = vec![f32::from_bits(POISON); elems + 2 * GUARD];
        let root = arena.f32(gpu, &values)?;
        let view = arena.get(root).sub_offset(GUARD, elems);
        Ok(Self { root, view, elems })
    }
    pub(crate) fn reset(&self, gpu: &Gpu, arena: &Arena) -> R<()> {
        let bytes = POISON.to_le_bytes().repeat(self.elems + 2 * GUARD);
        hip(gpu.hip.memcpy_htod(&arena.get(self.root).buf, &bytes))
    }
    pub(crate) fn read(&self, gpu: &Gpu, arena: &Arena, label: &str) -> R<Vec<u8>> {
        let bytes = hip(gpu.download_raw_bytes(arena.get(self.root)))?;
        check(
            bytes.len() == (self.elems + 2 * GUARD) * 4,
            format!("{label}: readback extent"),
        )?;
        for (i, word) in bytes.chunks_exact(4).enumerate() {
            let bits = u32::from_le_bytes(word.try_into().unwrap());
            if i < GUARD || i >= GUARD + self.elems {
                check(bits == POISON, format!("{label}: guard changed at {i}"))?;
            } else {
                check(
                    f32::from_bits(bits).is_finite(),
                    format!("{label}: unwritten/nonfinite payload at {}", i - GUARD),
                )?;
            }
        }
        check(
            bytes[GUARD * 4..(GUARD + self.elems) * 4]
                .chunks_exact(4)
                .any(|w| f32::from_le_bytes(w.try_into().unwrap()) != 0.0),
            format!("{label}: all-zero payload is vacuous"),
        )?;
        Ok(bytes)
    }
}

// Finite asymmetric f16 headers and independent random nibble payloads.
// QT44: two 4-byte headers followed by 128 payload bytes, stride 136.
// QT53: one 4-byte header followed by 64 payload bytes, stride 68.
pub(crate) fn packed_weights(stride: usize, group_bytes: usize) -> Vec<u8> {
    let mut bytes = vec![0u8; DISTINCT * stride];
    let mut rng = 0x4d59_5df4_d0f3_3173u64 ^ group_bytes as u64;
    for byte in &mut bytes {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        *byte = rng as u8;
    }
    let headers = [
        [0x00, 0x24, 0x00, 0xb0],
        [0x00, 0xa0, 0x00, 0x2c],
        [0x01, 0x28, 0x01, 0xb0],
        [0x01, 0xa4, 0x01, 0x30],
    ];
    let halves = if group_bytes == 136 { 2 } else { 1 };
    for (expert, payload) in bytes.chunks_exact_mut(stride).enumerate() {
        for (g, group) in payload.chunks_exact_mut(group_bytes).enumerate() {
            for h in 0..halves {
                group[h * 4..h * 4 + 4].copy_from_slice(&headers[(expert + g + h) % headers.len()]);
            }
        }
    }
    bytes
}

pub(crate) struct Fixture {
    pub(crate) gu_device: usize,
    pub(crate) gu_host: usize,
    pub(crate) dn_device: usize,
    pub(crate) dn_host: usize,
    pub(crate) gu_table: GpuTensor,
    pub(crate) dn_table: GpuTensor,
    pub(crate) ids: GpuTensor,
    pub(crate) gu_x: GpuTensor,
    pub(crate) dn_x: GpuTensor,
    pub(crate) gate: Guarded,
    pub(crate) up: Guarded,
    pub(crate) down: Guarded,
}
impl Fixture {
    pub(crate) fn build(gpu: &mut Gpu, arena: &mut Arena, report: &mut Value) -> R<Self> {
        let gu = packed_weights(GU_STRIDE, 136);
        let dn = packed_weights(DN_STRIDE, 68);
        let gu_device = arena.raw(gpu, &gu)?;
        let dn_device = arena.raw(gpu, &dn)?;
        let t = hip(gpu.upload_raw_host_mapped(&gu, &[gu.len()]))?;
        let gu_host = arena.own(t);
        let t = hip(gpu.upload_raw_host_mapped(&dn, &[dn.len()]))?;
        let dn_host = arena.own(t);
        for (name, id, source, mapped) in [
            ("gu_device", gu_device, &gu, false),
            ("gu_host", gu_host, &gu, true),
            ("dn_device", dn_device, &dn, false),
            ("dn_host", dn_host, &dn, true),
        ] {
            check(
                arena.get(id).buf.is_host_mapped() == mapped,
                format!("{name}: unexpected owner placement"),
            )?;
            let actual = hip(gpu.download_raw_bytes(arena.get(id)))?;
            check(
                actual == *source,
                format!("{name}: uploaded weight bytes differ"),
            )?;
        }
        let gu_table_root = arena.raw(gpu, &vec![0; E * 8])?;
        let dn_table_root = arena.raw(gpu, &vec![0; E * 8])?;
        // Bounded table views give the production launcher exactly 512 entries,
        // independent of any allocation-pool capacity rounding.
        let gu_table = arena.get(gu_table_root).sub_offset(0, E * 8);
        let dn_table = arena.get(dn_table_root).sub_offset(0, E * 8);
        check(
            gu_table.buf.size() == E * 8 && dn_table.buf.size() == E * 8,
            "bounded pointer table size",
        )?;
        let ids_bytes: Vec<u8> = (0..TOP as i32).flat_map(i32::to_le_bytes).collect();
        let ids_root = arena.raw(gpu, &ids_bytes)?;
        let ids = arena.get(ids_root).sub_offset(0, ids_bytes.len());
        let gu_values: Vec<f32> = (0..HIDDEN)
            .map(|i| ((i * 37 % 1021) as f32 - 510.0) / 997.0)
            .collect();
        let dn_values: Vec<f32> = (0..TOP * MI)
            .map(|i| ((i * 53 % 1031) as f32 - 515.0) / 991.0)
            .collect();
        let gx = arena.f32(gpu, &gu_values)?;
        let dx = arena.f32(gpu, &dn_values)?;
        let gu_x = arena.get(gx).sub_offset(0, HIDDEN);
        let dn_x = arena.get(dx).sub_offset(0, TOP * MI);
        report["fixture"] = json!({
            "gu_m": GU_M, "gu_k": HIDDEN, "dn_m": HIDDEN, "dn_k": MI,
            "rows": 1, "n_experts": E, "topk": TOP, "distinct_payloads": DISTINCT,
            "gu_stride": GU_STRIDE, "dn_stride": DN_STRIDE,
            "weight_bytes_per_placement": gu.len() + dn.len(),
            "gu_weight_sha256": hash(&gu), "dn_weight_sha256": hash(&dn),
            "topk_le_i32_sha256": hash(&ids_bytes),
            "weight_copy_full_readback_exact": true,
            "inputs": "fixed independent projections, not a gate-to-down numerical chain",
            "guard_f32_elements_each_side": GUARD, "poison_bits": POISON,
            "host_allocation_api": "Gpu::upload_raw_host_mapped",
            "table_gu_address": gu_table.buf.as_ptr() as usize,
            "table_dn_address": dn_table.buf.as_ptr() as usize,
            "host_payloads": [
                {"device_alias": arena.get(gu_host).buf.as_ptr() as usize,
                 "allocation_bytes": arena.get(gu_host).buf.size()},
                {"device_alias": arena.get(dn_host).buf.as_ptr() as usize,
                 "allocation_bytes": arena.get(dn_host).buf.size()}
            ]
        });
        Ok(Self {
            gu_device,
            gu_host,
            dn_device,
            dn_host,
            gu_table,
            dn_table,
            ids,
            gu_x,
            dn_x,
            gate: Guarded::new(gpu, arena, GU_ELEMS)?,
            up: Guarded::new(gpu, arena, GU_ELEMS)?,
            down: Guarded::new(gpu, arena, DN_ELEMS)?,
        })
    }
    pub(crate) fn publish(&self, gpu: &Gpu, arena: &Arena, case: Case) -> R<Value> {
        let gu_base = arena
            .get(if case.gu_host {
                self.gu_host
            } else {
                self.gu_device
            })
            .buf
            .as_ptr() as usize;
        let dn_base = arena
            .get(if case.dn_host {
                self.dn_host
            } else {
                self.dn_device
            })
            .buf
            .as_ptr() as usize;
        let table = |base: usize, stride: usize, wrong: bool| -> Vec<u8> {
            (0..E)
                .flat_map(|e| {
                    let payload = if wrong && e == 0 { 1 } else { e % DISTINCT };
                    ((base + payload * stride) as u64).to_le_bytes()
                })
                .collect()
        };
        let gu = table(gu_base, GU_STRIDE, case.wrong_gu);
        let dn = table(dn_base, DN_STRIDE, case.wrong_dn);
        // Every previous arm is retired. These synchronous writes and the
        // following device sync complete before any HIP or HSA submission.
        hip(gpu.hip.memcpy_htod(&self.gu_table.buf, &gu))?;
        hip(gpu.hip.memcpy_htod(&self.dn_table.buf, &dn))?;
        hip(gpu.hip.device_synchronize())?;
        check(
            hip(gpu.download_raw_bytes(&self.gu_table))? == gu,
            "gate/up table publication readback",
        )?;
        check(
            hip(gpu.download_raw_bytes(&self.dn_table))? == dn,
            "down table publication readback",
        )?;
        Ok(json!({"gu_table_sha256": hash(&gu), "dn_table_sha256": hash(&dn)}))
    }
    pub(crate) fn reset(&self, gpu: &Gpu, arena: &Arena) -> R<()> {
        self.gate.reset(gpu, arena)?;
        self.up.reset(gpu, arena)?;
        self.down.reset(gpu, arena)?;
        hip(gpu.hip.device_synchronize())
    }
    pub(crate) fn launch(&self, gpu: &mut Gpu) -> R<()> {
        hip(gpu.gemv_mq4g256v2_moe_gate_up_top10_indexed_batched(
            &self.gu_table,
            &self.ids,
            &self.gu_x,
            &self.gate.view,
            &self.up.view,
            GU_M,
            HIDDEN,
            1,
        ))?;
        check(
            gpu.last_launched_kernel() == Some(GU_SYMBOL),
            format!(
                "unexpected gate/up kernel: {:?}",
                gpu.last_launched_kernel()
            ),
        )?;
        hip(gpu.gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded(
            &self.dn_table,
            &self.ids,
            &self.dn_x,
            &self.down.view,
            HIDDEN,
            MI,
            1,
            E,
            TOP,
        ))?;
        check(
            gpu.last_launched_kernel() == Some(DN_SYMBOL),
            format!("unexpected down kernel: {:?}", gpu.last_launched_kernel()),
        )
    }
    pub(crate) fn read(&self, gpu: &Gpu, arena: &Arena) -> R<Snapshot> {
        Ok(Snapshot {
            gate: self.gate.read(gpu, arena, "gate")?,
            up: self.up.read(gpu, arena, "up")?,
            down: self.down.read(gpu, arena, "down")?,
        })
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Case {
    pub(crate) name: &'static str,
    pub(crate) gu_host: bool,
    pub(crate) dn_host: bool,
    pub(crate) wrong_gu: bool,
    pub(crate) wrong_dn: bool,
}
pub(crate) fn cases() -> Vec<Case> {
    let v = |name| Case {
        name,
        gu_host: false,
        dn_host: false,
        wrong_gu: false,
        wrong_dn: false,
    };
    vec![
        v("vram_initial"),
        Case {
            gu_host: true,
            ..v("host_gate_up")
        },
        v("vram_after_gate_up"),
        Case {
            dn_host: true,
            ..v("host_down")
        },
        v("vram_after_down"),
        Case {
            gu_host: true,
            dn_host: true,
            ..v("host_both")
        },
        v("vram_after_both"),
        Case {
            wrong_gu: true,
            ..v("wrong_gate_up_entry0")
        },
        v("vram_after_wrong_gate_up"),
        Case {
            wrong_dn: true,
            ..v("wrong_down_entry0")
        },
        v("vram_after_wrong_down"),
    ]
}
pub(crate) struct Snapshot {
    pub(crate) gate: Vec<u8>,
    pub(crate) up: Vec<u8>,
    pub(crate) down: Vec<u8>,
}
impl Snapshot {
    pub(crate) fn differences(&self, reference: &Self) -> [usize; 3] {
        let mismatch = |a: &[u8], b: &[u8]| {
            if a.len() != b.len() {
                return usize::MAX;
            }
            a.chunks_exact(4)
                .zip(b.chunks_exact(4))
                .filter(|(x, y)| x != y)
                .count()
        };
        [
            mismatch(&self.gate, &reference.gate),
            mismatch(&self.up, &reference.up),
            mismatch(&self.down, &reference.down),
        ]
    }
    pub(crate) fn digests(&self) -> Value {
        json!({"gate": hash(&self.gate), "up": hash(&self.up), "down": hash(&self.down)})
    }
}
pub(crate) struct Tape {
    pub(crate) launches: Vec<RecordedHipLaunch>,
    pub(crate) prepared: PreparedReplayIdentity,
    pub(crate) gu_table: usize,
    pub(crate) dn_table: usize,
}
impl Tape {
    pub(crate) fn verify(&self, gpu: &Gpu, f: &Fixture) -> R<()> {
        check(
            gpu.replay.recorded_launches() == self.launches.as_slice(),
            "recorded tape changed or was recaptured",
        )?;
        check(
            gpu.replay.prepared_route_identity() == Some(self.prepared),
            "prepared PM4 identity changed",
        )?;
        check(
            gpu.replay.prepared_pm4_plan_ready(),
            "prepared PM4 plan absent",
        )?;
        check(
            f.gu_table.buf.as_ptr() as usize == self.gu_table
                && f.dn_table.buf.as_ptr() as usize == self.dn_table,
            "pointer table allocation changed",
        )
    }
}
#[derive(Clone, Copy)]
pub(crate) enum Arm {
    Eager,
    RecordedHip,
    Pm4,
    HipGraph,
}
impl Arm {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Eager => "hip_eager",
            Self::RecordedHip => "recorded_hip",
            Self::Pm4 => "pm4",
            Self::HipGraph => "hip_graph",
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn arm(
    gpu: &mut Gpu,
    arena: &Arena,
    f: &Fixture,
    tape: &Tape,
    exec: Option<&hip_bridge::GraphExec>,
    which: Arm,
    idle: &mut bool,
) -> R<(Snapshot, Value)> {
    tape.verify(gpu, f)?;
    f.reset(gpu, arena)?;
    let before = launch_count();
    let graphs_before = graph_count();
    gpu.replay.begin_replay_observation_window();
    *idle = false;
    match which {
        Arm::Eager => f.launch(gpu)?,
        Arm::RecordedHip => hip(gpu.replay_recorded_hip_prefix_at(2, 0))?,
        Arm::Pm4 => {
            // Only table contents change; all captured allocations, input/output
            // addresses, layouts and packed payload owners remain live.
            match unsafe { gpu.replay.replay_pm4_checked(0) } {
                Ok(_) => {}
                Err(failure) => {
                    *idle = failure.quiescence == ReplayQuiescence::Proven;
                    return Err(format!(
                        "PM4: {}; quiescence={:?}",
                        failure.error, failure.quiescence
                    ));
                }
            }
        }
        Arm::HipGraph => hip(gpu.launch_graph(exec.ok_or("hipGraph exec absent")?))?,
    }
    hip(gpu.hip.device_synchronize())?;
    *idle = true;
    let launches = launch_count().saturating_sub(before);
    let graph_launches = graph_count().saturating_sub(graphs_before);
    let observation = gpu.replay.replay_observation();
    match which {
        Arm::Eager | Arm::RecordedHip => check(
            launches == 2 && graph_launches == 0,
            format!(
                "{}: actual HIP launch count {launches}/{graph_launches}",
                which.name()
            ),
        )?,
        Arm::Pm4 => {
            check(
                launches == 0 && graph_launches == 0,
                "PM4 silently ran a HIP body",
            )?;
            check(
                observation.count == 1
                    && observation.first_position == Some(0)
                    && observation.last_position == Some(0)
                    && !observation.failed,
                format!("PM4 replay observation: {observation:?}"),
            )?;
        }
        Arm::HipGraph => check(
            launches == 0 && graph_launches == 1,
            "hipGraph launch count",
        )?,
    }
    if !matches!(which, Arm::Pm4) {
        check(
            observation.count == 0 && !observation.failed,
            "unexpected retained replay",
        )?;
    }
    tape.verify(gpu, f)?;
    let output = f.read(gpu, arena)?;
    let proof = json!({
        "arm": which.name(), "hip_kernel_launches": launches,
        "hip_graph_launches": graph_launches,
        "pm4_replays": observation.count, "pm4_failed": observation.failed,
        "outputs_reinitialized_to_nan_before_submit": true,
        "finite_nonzero_full_payload_and_guards_checked": true,
        "same_table_allocation": true, "same_recorded_tape": true,
        "same_prepared_pm4_identity": true, "output_sha256": output.digests()
    });
    Ok((output, proof))
}
