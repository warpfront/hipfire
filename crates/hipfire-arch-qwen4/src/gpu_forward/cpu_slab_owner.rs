// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! CPU views of a host-mapped expert-slab layer: the slab span kind of the D2
//! host-span owner.
//!
//! A slab layer keeps its routed experts in ONE arena (wholly device-resident
//! or wholly host-mapped), each expert at its own `(gate_up, down)` offsets
//! with its own codec (MQ4, W3 or E8P). When the arena is host-mapped every
//! present expert is host-placed, so the D2 executor's placement is all host
//! for that layer and the CPU multiplies straight out of the arena's pinned
//! host bytes. [`CpuSlabOwners`] caches the OWNING arena's host address once
//! (`Gpu::host_bytes` of the owner, never a borrowed device view, the same
//! rule as `slab_stage`'s host sources) and hands out checked per-expert
//! slices. Nothing is copied.
//!
//! Each expert's CPU format is resolved at construction from its own dtype
//! pair; a host expert whose pair has no CPU decoder refuses the whole layer
//! (it then keeps the PCIe mapped-expert read), so a payload request can only
//! fail for an absent or out-of-range expert, or a dtype that disagrees with
//! the arena table.

use super::*;

/// One expert of a slab layer as the arena plan declares it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct SlabSpanEntry {
    /// A pruned (never routed) expert: no payload in the arena.
    pub absent: bool,
    pub gate_up_offset: usize,
    pub gate_up_len: usize,
    pub down_offset: usize,
    pub down_len: usize,
    pub gate_up_dtype: DType,
    pub down_dtype: DType,
}

/// One present expert's validated arena ranges and dtypes (each proven to have
/// a CPU decoder).
#[derive(Clone, Debug, PartialEq, Eq)]
struct SlabExpertSpan {
    gate_up: std::ops::Range<usize>,
    down: std::ops::Range<usize>,
    gate_up_dtype: DType,
    down_dtype: DType,
}

/// `[offset, offset + len)` inside `[0, limit)`, `None` for an empty or
/// overflowing or out-of-bounds span.
fn arena_span(offset: usize, len: usize, limit: usize) -> Option<std::ops::Range<usize>> {
    if len == 0 {
        return None;
    }
    let end = offset.checked_add(len)?;
    (end <= limit).then_some(offset..end)
}

/// Validate every entry against the arena extent `limit` (the plan's payload
/// bytes, already checked to fit the host allocation) and resolve each present
/// expert's CPU formats. Pure: the first unresolvable expert is reported by
/// index.
fn slab_span_table(
    entries: impl Iterator<Item = SlabSpanEntry>,
    limit: usize,
) -> Result<Vec<Option<SlabExpertSpan>>, String> {
    entries
        .enumerate()
        .map(|(expert, entry)| {
            if entry.absent {
                return Ok(None);
            }
            let gate_up = arena_span(entry.gate_up_offset, entry.gate_up_len, limit)
                .ok_or_else(|| format!("expert {expert} gate/up span leaves the arena"))?;
            let down = arena_span(entry.down_offset, entry.down_len, limit)
                .ok_or_else(|| format!("expert {expert} down span leaves the arena"))?;
            if gate_up.start < down.end && down.start < gate_up.end {
                return Err(format!("expert {expert} gate/up and down spans overlap"));
            }
            for dtype in [entry.gate_up_dtype, entry.down_dtype] {
                if hipfire_dispatch::cpu_exec::cpu_quant_for(dtype).is_none() {
                    return Err(format!(
                        "expert {expert} format {dtype:?} has no CPU decoder"
                    ));
                }
            }
            Ok(Some(SlabExpertSpan {
                gate_up,
                down,
                gate_up_dtype: entry.gate_up_dtype,
                down_dtype: entry.down_dtype,
            }))
        })
        .collect()
}

/// Cached host views of one host-mapped slab layer's present experts.
///
/// The address is the OWNING arena allocation's pinned host bytes; the arena
/// lives in the bundle's weights, which outlive the private runtime holding
/// this cache, and is never replaced after the forward attaches.
pub(super) struct CpuSlabOwners {
    host_base: usize,
    spans: Vec<Option<SlabExpertSpan>>,
}

impl CpuSlabOwners {
    /// Resolve every present expert of a host-mapped arena. `arena` must be
    /// the owning tensor (a borrowed view has no `host_bytes` registry entry and
    /// is reported as an error, never retagged); `arena_bytes` is the plan's
    /// payload extent. `Err` names the first span that cannot be served; the
    /// caller keeps that layer on the PCIe mapped-expert read.
    pub(super) fn new(
        gpu: &Gpu,
        arena: &GpuTensor,
        arena_bytes: usize,
        entries: impl Iterator<Item = SlabSpanEntry>,
    ) -> Result<Self, String> {
        let host = gpu
            .host_bytes(arena)
            .ok_or_else(|| "the slab arena has no pinned host bytes".to_string())?;
        Self::from_host(host, arena_bytes, entries)
    }

    fn from_host(
        host: &[u8],
        arena_bytes: usize,
        entries: impl Iterator<Item = SlabSpanEntry>,
    ) -> Result<Self, String> {
        if arena_bytes > host.len() {
            return Err(format!(
                "the slab arena holds {} host bytes, its plan needs {arena_bytes}",
                host.len()
            ));
        }
        Ok(Self {
            host_base: host.as_ptr() as usize,
            spans: slab_span_table(entries, arena_bytes)?,
        })
    }

    /// Borrowed CPU payload of one present expert; `None` for an absent or
    /// out-of-range expert, or when `gate_up_dtype` / `down_dtype` (the caller's
    /// per-expert view dtypes) disagree with the arena table.
    pub(super) fn payload(
        &self,
        expert_idx: usize,
        gate_up_dtype: DType,
        down_dtype: DType,
    ) -> Option<hipfire_dispatch::cpu_exec::CpuMoeExpert<'_>> {
        let span = self.spans.get(expert_idx)?.as_ref()?;
        if span.gate_up_dtype != gate_up_dtype || span.down_dtype != down_dtype {
            return None;
        }
        // Proven `Some` at construction (`slab_span_table`).
        let gate_up_quant = hipfire_dispatch::cpu_exec::cpu_quant_for(gate_up_dtype)?;
        let down_quant = hipfire_dispatch::cpu_exec::cpu_quant_for(down_dtype)?;
        // SAFETY: `host_base` is the CPU address of the owning host-mapped
        // arena's pinned bytes (`Gpu::host_bytes`), and `slab_span_table`
        // proved both ranges lie inside the plan extent, which `from_host`
        // proved fits that allocation. The arena outlives this cache (the
        // runtime is torn down before the weights free their host
        // allocations) and is never written after load. The slices are shared
        // reads that borrow `self`.
        let (gate_up, down) = unsafe {
            (
                std::slice::from_raw_parts(
                    (self.host_base + span.gate_up.start) as *const u8,
                    span.gate_up.len(),
                ),
                std::slice::from_raw_parts(
                    (self.host_base + span.down.start) as *const u8,
                    span.down.len(),
                ),
            )
        };
        Some(hipfire_dispatch::cpu_exec::CpuMoeExpert {
            gate_up_quant,
            gate_up,
            down_quant,
            down,
        })
    }

    /// The D2 executor's ownership mask: `1` (host, CPU) for every present
    /// expert of the host-mapped arena, `0` for an absent one.
    pub(super) fn placement(&self) -> Vec<u8> {
        self.spans
            .iter()
            .map(|span| u8::from(span.is_some()))
            .collect()
    }

    /// Number of present (CPU-served) experts.
    pub(super) fn cold_experts(&self) -> usize {
        self.spans.iter().flatten().count()
    }

    /// Every dtype a present expert's two tensors hold (the formats the CPU
    /// arm decodes on this layer), with repeats.
    pub(super) fn dtypes(&self) -> impl Iterator<Item = DType> + '_ {
        self.spans
            .iter()
            .flatten()
            .flat_map(|span| [span.gate_up_dtype, span.down_dtype])
    }
}

impl CpuSlabOwners {
    /// `[(gate_up dtype, gate_up bytes), (down dtype, down bytes)]` of one
    /// present expert: its own arena table entry, never another slot's or the
    /// layer's dominant format. `None` for an absent or out-of-range expert.
    pub(super) fn expert_formats(&self, expert_idx: usize) -> Option<[(DType, usize); 2]> {
        let span = self.spans.get(expert_idx)?.as_ref()?;
        Some([
            (span.gate_up_dtype, span.gate_up.len()),
            (span.down_dtype, span.down.len()),
        ])
    }

    /// One present expert as a D2 default probe slot: the production CPU
    /// payload in the expert's own codecs and, as an independent input,
    /// `gpu_read_bytes` (the GPU route's read of that same slot,
    /// [`gpu_read_bytes`]). `None` for an absent or out-of-range expert.
    pub(super) fn probe_expert(
        &self,
        expert_idx: usize,
        gpu_read_bytes: u64,
    ) -> Option<crate::d2_default::ProbeExpert<'_>> {
        let formats = self.expert_formats(expert_idx)?;
        let cpu = self.payload(expert_idx, formats[0].0, formats[1].0)?;
        Some(crate::d2_default::ProbeExpert {
            cpu,
            gpu_read_bytes,
        })
    }
}

/// Per-expert span entries of one layer's arena plan, resolved exactly as the
/// layer build resolves them: each expert's own format (an absent, pruned
/// expert carries the MQ4 pair and a zero-length span), its payload extents and
/// its offsets; `kept` is the layer's `qwen4.expert_keep` row (a pruned expert
/// with zero-byte spans is absent). The one place the plan turns into [`SlabSpanEntry`]s, shared by
/// the layer build and the D2 default's real-arm probe. MQ2-I4 has no
/// expert-slab route and is refused.
pub(super) fn plan_entries(
    plan: &crate::expert_arena::ExpertArenaLayer,
    kept: Option<&[u16]>,
) -> Result<Vec<SlabSpanEntry>, String> {
    plan.experts
        .iter()
        .enumerate()
        .map(|(expert, entry)| {
            let format = entry.effective_format();
            if format == crate::expert_slabs::ExpertFormat::Mq2I4 {
                return Err(format!("expert {expert}: MQ2-I4 has no expert-slab route"));
            }
            let (gate_up_dtype, down_dtype) = format.dtypes();
            let absent = expert_pruned(kept, expert) && entry.gate_up_offset == entry.down_offset;
            Ok(SlabSpanEntry {
                absent,
                gate_up_offset: entry.gate_up_offset,
                gate_up_len: format.gate_up_bytes(),
                down_offset: entry.down_offset,
                down_len: format.down_bytes(),
                gate_up_dtype,
                down_dtype,
            })
        })
        .collect()
}

/// Bytes the GPU route reads over the link for each slot of one slab layer
/// (gate/up plus down), `None` for an absent expert. A host-mapped slab arena's
/// decode and `<=4`-row verify read every routed slot in place in the slot's
/// OWN codec (the sealed slab kernels bind the per-slot codec tag; see
/// `slab_stage`: "decode and MTP verify keep reading in place"), so a slot's
/// traffic is its own format's gate/up and down payload: an E8P slot moves the
/// E8P bytes, an MQ4 slot the MQ4 bytes. The figure comes from the slot's plan
/// format (the table the GPU views are built from), never from the CPU span
/// ranges, the CPU quant, the arena allocation or one constant for the layer.
pub(super) fn gpu_read_bytes(plan: &crate::expert_arena::ExpertArenaLayer) -> Vec<Option<u64>> {
    plan.experts
        .iter()
        .map(|entry| {
            (!entry.is_absent())
                .then(|| u64::try_from(entry.effective_format().slab_bytes()).ok())
                .flatten()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expert_slabs::ExpertFormat;
    use hipfire_dispatch::cpu_exec::cpu_quant_for;

    /// Real E8P / MQ4 per-expert extents of the mq2 artifact.
    const E8P: (usize, usize) = (870_400, 435_200);
    const MQ4: (usize, usize) = (1_740_800, 870_400);

    fn entry(offset: usize, (gu, dn): (usize, usize), e8p: bool) -> SlabSpanEntry {
        let (gate_up_dtype, down_dtype) = if e8p {
            (DType::E8PI4G128R256, DType::E8PI4G128R128)
        } else {
            (DType::MQ4G256V2, DType::MQ4G128V2)
        };
        SlabSpanEntry {
            absent: false,
            gate_up_offset: offset,
            gate_up_len: gu,
            down_offset: offset + gu,
            down_len: dn,
            gate_up_dtype,
            down_dtype,
        }
    }

    /// A packed mixed layer: E8P, MQ4, absent, E8P back to back.
    fn mixed() -> (Vec<SlabSpanEntry>, usize) {
        let mut entries = Vec::new();
        let mut at = 0;
        for (shape, e8p) in [(E8P, true), (MQ4, false)] {
            entries.push(entry(at, shape, e8p));
            at += shape.0 + shape.1;
        }
        entries.push(SlabSpanEntry {
            absent: true,
            gate_up_offset: at,
            gate_up_len: 0,
            down_offset: at,
            down_len: 0,
            gate_up_dtype: DType::MQ4G256V2,
            down_dtype: DType::MQ4G128V2,
        });
        entries.push(entry(at, E8P, true));
        at += E8P.0 + E8P.1;
        (entries, at)
    }

    #[test]
    fn mixed_layer_resolves_per_expert_formats_and_spans() {
        for (format, (gu, dn)) in [(ExpertFormat::E8pI4, E8P), (ExpertFormat::Mq4Xts, MQ4)] {
            assert_eq!(
                (format.gate_up_bytes(), format.down_bytes()),
                (gu, dn),
                "{format:?}"
            );
        }
        let (entries, bytes) = mixed();
        let host = vec![0u8; bytes + 4096];
        let owners = CpuSlabOwners::from_host(&host, bytes, entries.iter().copied()).unwrap();
        assert_eq!(owners.placement(), vec![1, 1, 0, 1]);
        assert_eq!(owners.cold_experts(), 3);
        let e0 = owners
            .payload(0, DType::E8PI4G128R256, DType::E8PI4G128R128)
            .expect("E8P expert");
        assert_eq!(Some(e0.gate_up_quant), cpu_quant_for(DType::E8PI4G128R256));
        assert_eq!(Some(e0.down_quant), cpu_quant_for(DType::E8PI4G128R128));
        assert_eq!((e0.gate_up.len(), e0.down.len()), E8P);
        assert_eq!(e0.gate_up.as_ptr(), host.as_ptr());
        let e1 = owners
            .payload(1, DType::MQ4G256V2, DType::MQ4G128V2)
            .expect("MQ4 expert");
        assert_eq!(Some(e1.gate_up_quant), cpu_quant_for(DType::MQ4G256V2));
        assert_eq!(
            e1.gate_up.as_ptr() as usize - host.as_ptr() as usize,
            E8P.0 + E8P.1
        );
        assert_eq!(
            e1.down.as_ptr() as usize - host.as_ptr() as usize,
            E8P.0 + E8P.1 + MQ4.0
        );
        // Absent, out of range and a dtype disagreeing with the table never resolve.
        assert!(owners
            .payload(2, DType::MQ4G256V2, DType::MQ4G128V2)
            .is_none());
        assert!(owners
            .payload(4, DType::E8PI4G128R256, DType::E8PI4G128R128)
            .is_none());
        assert!(owners
            .payload(0, DType::MQ4G256V2, DType::MQ4G128V2)
            .is_none());
        // Every resolved payload passes the D2 CPU engine's own span contract
        // for the real expert shape (hidden 2560, moe intermediate 640).
        for (idx, (gd, dd)) in [
            (0, (DType::E8PI4G128R256, DType::E8PI4G128R128)),
            (1, (DType::MQ4G256V2, DType::MQ4G128V2)),
            (3, (DType::E8PI4G128R256, DType::E8PI4G128R128)),
        ] {
            let e = owners.payload(idx, gd, dd).unwrap();
            hipfire_dispatch::cpu_exec::validate_moe_cpu_host_slot_outputs(
                2560,
                640,
                1,
                1,
                4,
                &[idx as i32],
                None,
                &[0],
                &[e],
                2560,
                false,
                2560,
            )
            .unwrap_or_else(|error| panic!("expert {idx}: {error:?}"));
        }
    }

    #[test]
    fn spans_outside_the_plan_or_short_hosts_refuse_the_layer() {
        let (entries, bytes) = mixed();
        let host = vec![0u8; bytes];
        // The plan extent must fit the host allocation.
        let short = CpuSlabOwners::from_host(&host[..bytes - 1], bytes, entries.iter().copied());
        assert!(short.is_err());
        // A span past the plan extent names its expert.
        let mut bad = entries.clone();
        bad[3].down_len += 1;
        let err = CpuSlabOwners::from_host(&host, bytes, bad.into_iter())
            .err()
            .unwrap();
        assert!(err.contains("expert 3 down"), "{err}");
        // Overlapping projections refuse.
        let mut overlap = entries.clone();
        overlap[0].down_offset = overlap[0].gate_up_offset + 16;
        let err = CpuSlabOwners::from_host(&host, bytes, overlap.into_iter())
            .err()
            .unwrap();
        assert!(err.contains("overlap"), "{err}");
        // A format without a CPU decoder refuses the layer instead of a slot.
        let mut w2 = entries;
        w2[1].gate_up_dtype = DType::MQ2I4G128R256;
        let err = CpuSlabOwners::from_host(&host, bytes, w2.into_iter())
            .err()
            .unwrap();
        assert!(
            err.contains("expert 1") && err.contains("no CPU decoder"),
            "{err}"
        );
    }

    /// A real E8P/MQ4 mixed slab layer, from the arena plan to the production
    /// CPU arm: every slot's GPU read bytes are its OWN plan format's gate/up +
    /// down payload (E8P 1,305,600 B, MQ4 2,611,200 B at the mq2 shape), an
    /// independent input from the CPU span lengths and quant, and a mixed
    /// batch's byte count is the sum of its slots'. CPU only.
    #[test]
    fn mixed_real_e8p_mq4_slots_carry_their_own_gpu_read_bytes() {
        use crate::d2_default::{sample_cpu_arm, PROBE_SAMPLES, PROBE_WARM_BATCHES};
        use crate::expert_arena::{ArenaExpert, ExpertArenaLayer};

        const E8P_SLOT: u64 = 1_305_600;
        const MQ4_SLOT: u64 = 2_611_200;
        assert_eq!((E8P.0 + E8P.1) as u64, E8P_SLOT);
        assert_eq!((MQ4.0 + MQ4.1) as u64, MQ4_SLOT);
        let (dim, mi, top_k) = (2560, 640, 2);
        let timed = (PROBE_WARM_BATCHES + PROBE_SAMPLES) * top_k;
        // Experts 0..12: MQ4 where `i % 3 == 2`, E8P elsewhere; expert 12 is
        // a pruned (absent) expert at the layer's tail.
        let format_of = |expert: usize| {
            if expert % 3 == 2 {
                ExpertFormat::Mq4Xts
            } else {
                ExpertFormat::E8pI4
            }
        };
        let mut experts = Vec::new();
        let mut at = 0;
        for expert in 0..timed {
            let format = format_of(expert);
            experts.push(ArenaExpert {
                format,
                gate_up_offset: at,
                down_offset: at + format.gate_up_bytes(),
            });
            at += format.slab_bytes();
        }
        experts.push(ArenaExpert {
            format: ExpertFormat::Mq4Xts,
            gate_up_offset: at,
            down_offset: at,
        });
        let plan = ExpertArenaLayer {
            file_offset: 0,
            bytes: at,
            experts,
        };
        let kept: Vec<u16> = (0..timed as u16).collect();
        let entries = plan_entries(&plan, Some(&kept)).unwrap();
        assert_eq!(entries.len(), timed + 1);
        assert!(entries[timed].absent && !entries[..timed].iter().any(|entry| entry.absent));
        for (expert, entry) in entries[..timed].iter().enumerate() {
            let want = if expert % 3 == 2 { MQ4 } else { E8P };
            assert_eq!((entry.gate_up_len, entry.down_len), want, "expert {expert}");
        }

        // Real E8P tensors: a codes plane of arbitrary 16-bit words (every word
        // is a legal E8P word) then a plane of finite f16 scales. MQ4 spans are
        // arbitrary bytes, as in the other slot tests.
        let e8p_tensor = |len: usize, seed: usize| {
            let mut bytes = vec![0u8; len];
            let mut state = (seed as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
            for byte in bytes.iter_mut() {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                *byte = (state >> 24) as u8;
            }
            bytes
        };
        let mut host = vec![0u8; plan.bytes + 4096];
        for (expert, entry) in entries[..timed].iter().enumerate() {
            let gate_up = e8p_tensor(entry.gate_up_len, expert);
            let down = e8p_tensor(entry.down_len, expert + 1000);
            host[entry.gate_up_offset..][..gate_up.len()].copy_from_slice(&gate_up);
            host[entry.down_offset..][..down.len()].copy_from_slice(&down);
        }
        // Finite f16 scales over the E8P scale planes (the last
        // `rows * (k / 128) * 2` bytes of each E8P tensor).
        for (expert, entry) in entries[..timed].iter().enumerate() {
            if expert % 3 == 2 {
                continue;
            }
            for (offset, len, rows, k) in [
                (entry.gate_up_offset, entry.gate_up_len, 1280, 2560),
                (entry.down_offset, entry.down_len, 2560, 640),
            ] {
                let scales = rows * (k / 128) * 2;
                assert_eq!(len, rows * k / 4 + scales);
                for (i, pair) in host[offset + len - scales..offset + len]
                    .chunks_exact_mut(2)
                    .enumerate()
                {
                    // f16 0x2800 + (0..64) = 2^-7 .. 2^-4 scale, finite.
                    pair.copy_from_slice(&(0x2800u16 + (i % 64) as u16).to_le_bytes());
                }
            }
        }
        let owners = CpuSlabOwners::from_host(&host, plan.bytes, entries.iter().copied()).unwrap();
        let gpu_bytes = gpu_read_bytes(&plan);
        assert_eq!(gpu_bytes.len(), timed + 1);
        assert_eq!(gpu_bytes[timed], None, "an absent expert moves no bytes");
        assert!(
            owners.probe_expert(timed, 1).is_none(),
            "an absent expert is no probe slot"
        );

        let mut probes = Vec::new();
        for expert in 0..timed {
            let probe = owners
                .probe_expert(expert, gpu_bytes[expert].unwrap())
                .unwrap_or_else(|| panic!("expert {expert} resolves"));
            let (quant_gu, quant_dn, want) = if expert % 3 == 2 {
                (DType::MQ4G256V2, DType::MQ4G128V2, MQ4_SLOT)
            } else {
                (DType::E8PI4G128R256, DType::E8PI4G128R128, E8P_SLOT)
            };
            assert_eq!(probe.gpu_read_bytes, want, "expert {expert}");
            assert_eq!(Some(probe.cpu.gate_up_quant), cpu_quant_for(quant_gu));
            assert_eq!(Some(probe.cpu.down_quant), cpu_quant_for(quant_dn));
            // The CPU span is the same slot's payload (here the GPU reads that
            // codec in place), but it is not where the GPU figure comes from.
            assert_eq!(
                (probe.cpu.gate_up.len() + probe.cpu.down.len()) as u64,
                want
            );
            probes.push(probe);
        }
        assert_eq!(
            owners.expert_formats(0).unwrap(),
            [(DType::E8PI4G128R256, E8P.0), (DType::E8PI4G128R128, E8P.1)]
        );
        assert_eq!(
            owners.expert_formats(2).unwrap(),
            [(DType::MQ4G256V2, MQ4.0), (DType::MQ4G128V2, MQ4.1)]
        );

        // The production arm over the mixed slots; timed batches are experts
        // (2,3) (4,5) (6,7) (8,9) (10,11): MQ4+E8P, E8P+MQ4, E8P+E8P, MQ4+E8P,
        // E8P+MQ4.
        let (samples, batch_bytes) = sample_cpu_arm(dim, mi, top_k, 13, &probes).unwrap();
        assert!(samples.iter().all(|&ns| ns > 0), "{samples:?}");
        assert_eq!(
            batch_bytes,
            [
                MQ4_SLOT + E8P_SLOT,
                E8P_SLOT + MQ4_SLOT,
                2 * E8P_SLOT,
                MQ4_SLOT + E8P_SLOT,
                E8P_SLOT + MQ4_SLOT
            ]
        );
    }
}
