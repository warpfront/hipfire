//! GDN recurrent-state precision accounting (F32 vs Q8) for Qwen3.8-Flash-Next:
//! every figure is read from the owning helpers (`GdnStateFormat::state_bytes`,
//! `state_tensor_plan`, `Qwen4State::device_bytes*`, `row_capture_bytes*`, the
//! native-MTP request query, `session_snapshot_bytes`, the lane store census) and
//! compared with the storage arithmetic of the format; the PLAN §2 byte counts
//! appear as literals so a drift in any owner fails here. These are storage
//! sizes, not measured costs.

use super::*;
use crate::bundle::session_snapshot_bytes;
use crate::config::{compact_test_config, QWEN4_DEFAULT_CONTEXT};
use crate::mtp_spec::{native_mtp_request_bytes, native_mtp_window_rows};

const MIB: usize = 1 << 20;
/// One layer's recurrent state: 48 value heads x 128 rows x (128 i8 codes + one
/// F32 scale) for Q8, 48 x 128 x 128 F32 for F32.
const Q8_SLOT: usize = 811_008;
const F32_SLOT: usize = 3_145_728;
const GDN_LAYERS: usize = 36;
const Q8_ALL: usize = 29_196_288;
const F32_ALL: usize = 113_246_208;
/// `F32_ALL - Q8_ALL` = 80.15625 MiB: one live recurrent set, or one mirror.
const SET_DELTA: usize = 84_049_920;
/// `F32_SLOT - Q8_SLOT`: one layer's recurrent slot delta.
const SLOT_DELTA: usize = F32_SLOT - Q8_SLOT;
/// Capture-ring rows R = 4, 8, 32.
const RINGS: [usize; 3] = [4, 8, 32];
/// Ring state slots a state's capture ring allocates across `layers` GDN layers for `r`-row
/// blocks: the shared pool `2 * (layers - 1) + r + 1` by default
/// (`HIPFIRE_QWEN4_LANE_RESERVE_TRIM`), `r + 1` per layer with the trim kill switch, `2r` per
/// layer with the compact-ring kill switch too.
fn ring_slots(r: usize, layers: usize) -> usize {
    current_capture_ring_total_slots(r, layers).unwrap()
}
/// Q8 / F32 capture ring bytes at `RINGS[index]`: the recurrent slot x the whole model's ring slots.
fn ring_q8(index: usize) -> usize {
    Q8_SLOT * ring_slots(RINGS[index], GDN_LAYERS)
}
fn ring_f32(index: usize) -> usize {
    F32_SLOT * ring_slots(RINGS[index], GDN_LAYERS)
}

const FORMATS: [GdnStateFormat; 2] = [GdnStateFormat::Q8, GdnStateFormat::F32];

fn format_of(qsa: QsaKvFormat, gdn: GdnStateFormat) -> Qwen4StateFormat {
    Qwen4StateFormat {
        qsa,
        gdn,
        qsa_f16_arena: false,
        qsa_pooled_bf16: false,
        gdn_stream: crate::GdnStreamKind::None,
    }
}

/// Recurrent slot bytes per layer. Q8Ef stores plain Q8 codes+scales in the
/// slot (its live f16 residual is a separate owner, charged once per live
/// state and never in the ring/mirror/snapshot; see `gdn_q8ef_reset_lifecycle`).
fn slot_of(gdn: GdnStateFormat) -> usize {
    match gdn {
        GdnStateFormat::Q8 | GdnStateFormat::Q8Ef => Q8_SLOT,
        GdnStateFormat::F32 => F32_SLOT,
    }
}

fn ring_of(gdn: GdnStateFormat, index: usize) -> usize {
    match gdn {
        GdnStateFormat::Q8 | GdnStateFormat::Q8Ef => ring_q8(index),
        GdnStateFormat::F32 => ring_f32(index),
    }
}

/// Bytes of every GDN recurrent owner in `state_tensor_plan`, in layer order.
fn planned_recurrent_bytes(config: &Qwen4Config, format: Qwen4StateFormat) -> Vec<usize> {
    let scope = Qwen4BandScope::single(config);
    let map = Qwen4StateMap::new(config, &scope).expect("state map");
    let geometry = StateGeometry::new(config, &format, 1024).expect("geometry");
    state_tensor_plan(config, &format, 1024, &geometry, &map)
        .expect("tensor plan")
        .iter()
        .filter(|spec| spec.key.ends_with(".recurrent"))
        .map(|spec| spec.elements * spec.dtype.size())
        .collect()
}

/// Bytes of the whole-model live recurrent set (plan owners).
fn live_set(config: &Qwen4Config, gdn: GdnStateFormat) -> usize {
    planned_recurrent_bytes(config, format_of(QsaKvFormat::Q8, gdn))
        .iter()
        .sum()
}

fn pp_scopes(counts: &[usize]) -> Vec<Qwen4BandScope> {
    let mut start = 0;
    counts
        .iter()
        .enumerate()
        .map(|(stage, &count)| {
            let band = start..start + count;
            start += count;
            Qwen4BandScope {
                band,
                input: stage == 0,
                output: stage + 1 == counts.len(),
            }
        })
        .collect()
}

/// PLAN §2's exact all-model session entry formula, with every input read
/// from the real helpers.
fn plan_entry_bytes(
    config: &Qwen4Config,
    format: Qwen4StateFormat,
    native_mtp: bool,
    p: usize,
) -> u64 {
    let a = |x: usize| x.next_multiple_of(256);
    let gdn = a(format.gdn.state_bytes(
        config.linear_num_value_heads,
        config.linear_key_head_dim,
        config.linear_value_head_dim,
    ));
    let conv = a(StateGeometry::new(config, &format, 1024)
        .expect("geometry")
        .conv_len
        * std::mem::size_of::<f32>());
    let k = format
        .qsa
        .kv_row_bytes(config.num_key_value_heads, config.head_dim);
    let raw_width = config.indexer_kv_heads * config.indexer_head_dim;
    let i = raw_width * format.qsa.index_dtype().size();
    let s = config.qsa_selected_capacity();
    let h = config.ple_conv_history_rows();
    let feedback = config.hc_count * config.hidden_size * 4;
    let pooled = p.div_ceil(config.indexer_compress_ratio);
    let mut total = config.n_linear_layers() * (gdn + conv)
        + config.n_full_layers() * (2 * a(p * k) + a(p * i) + a(pooled * i) + a(4 * s))
        + a(h * feedback)
        + a(feedback);
    if native_mtp {
        // Head QSA K/V, raw and pooled index, selection, selected length,
        // wide hidden and the target boundary hidden.
        let full_width = config.num_key_value_heads * config.head_dim;
        total += 2 * a(p * full_width * 4)
            + a(p * raw_width * 4)
            + a(pooled * raw_width * 4)
            + a(4 * s)
            + a(4)
            + 2 * a(feedback);
    }
    total as u64
}

#[test]
fn gdn_precision_owner_accounting() {
    let config = compact_test_config();
    assert_eq!(
        (config.n_linear_layers(), config.n_full_layers()),
        (GDN_LAYERS, 12)
    );
    assert_eq!(
        (
            config.linear_num_value_heads,
            config.linear_key_head_dim,
            config.linear_value_head_dim
        ),
        (48, 128, 128)
    );

    // Per-layer and all-36 owner bytes, from the format and from the plan.
    for (gdn, slot, all) in [
        (GdnStateFormat::Q8, Q8_SLOT, Q8_ALL),
        (GdnStateFormat::F32, F32_SLOT, F32_ALL),
    ] {
        assert_eq!(gdn.state_bytes(48, 128, 128), slot, "{gdn:?}");
        let plan = planned_recurrent_bytes(&config, format_of(QsaKvFormat::Q8, gdn));
        assert_eq!(plan.len(), GDN_LAYERS);
        assert!(plan.iter().all(|&bytes| bytes == slot), "{gdn:?}");
        assert_eq!(plan.iter().sum::<usize>(), all, "{gdn:?}");
    }
    assert_eq!(F32_ALL - Q8_ALL, SET_DELTA);
    assert_eq!(Q8_ALL, 27 * MIB + 27 * MIB / 32);
    assert_eq!(F32_ALL, 108 * MIB);
    assert_eq!(SET_DELTA, 80 * MIB + 5 * MIB / 32);
    // One recurrent set: 28,311,552 F32 elements, 221,184 Q8 rows.
    assert_eq!(
        GDN_LAYERS * GdnStateFormat::F32.state_units(48, 128, 128),
        28_311_552
    );
    assert_eq!(GDN_LAYERS * 48 * 128, 221_184);

    // Live + the snapshot arena's mirror: the census doubles the live set.
    let context = Qwen4ContextCommit::legacy(1024);
    let device =
        |gdn| Qwen4State::device_bytes(&config, format_of(QsaKvFormat::Q8, gdn), &context).unwrap();
    assert_eq!(
        device(GdnStateFormat::F32) - device(GdnStateFormat::Q8),
        2 * SET_DELTA
    );
    // AR live + transaction mirror, recurrent only: 55.6875 -> 216 MiB.
    assert_eq!(2 * Q8_ALL, 55 * MIB + 11 * MIB / 16);
    assert_eq!(2 * F32_ALL, 216 * MIB);

    // Capture ring R+1 slots (2R with the kill switch); the non-ring terms are format independent.
    for (index, &r) in RINGS.iter().enumerate() {
        let q8 = Qwen4State::row_capture_bytes(&config, GdnStateFormat::Q8, r).unwrap();
        let f32 = Qwen4State::row_capture_bytes(&config, GdnStateFormat::F32, r).unwrap();
        assert_eq!(
            q8 - ring_q8(index),
            f32 - ring_f32(index),
            "R={r} non-ring terms"
        );
        assert_eq!(
            f32 - q8,
            ring_slots(r, GDN_LAYERS) * SLOT_DELTA,
            "R={r} ring delta"
        );
        for gdn in FORMATS {
            assert_eq!(
                ring_of(gdn, index),
                slot_of(gdn) * ring_slots(r, GDN_LAYERS),
                "{gdn:?} R={r} ring is {} recurrent slots",
                ring_slots(r, GDN_LAYERS)
            );
        }
        // Ring + one mirror = the ring's slot deltas + one recurrent-set delta more than Q8.
        assert_eq!(
            (ring_f32(index) + F32_ALL) - (ring_q8(index) + Q8_ALL),
            ring_slots(r, GDN_LAYERS) * SLOT_DELTA + SET_DELTA,
            "R={r}"
        );
        // The kill-switch layout stays 2R slots.
        assert_eq!(capture_ring_slots(false, r), Some(2 * r));
        assert_eq!(capture_ring_slots(true, r), Some(r + 1));
    }
    assert_eq!(
        ring_q8(0) + Q8_ALL,
        Q8_SLOT * (ring_slots(4, GDN_LAYERS) + GDN_LAYERS)
    );
    assert_eq!(
        ring_f32(0) + F32_ALL,
        F32_SLOT * (ring_slots(4, GDN_LAYERS) + GDN_LAYERS)
    );
    assert_eq!(ring_q8(0), Q8_SLOT * ring_slots(4, GDN_LAYERS));
    assert_eq!(F32_ALL * 2 * 32, 6912 * MIB);
    assert_eq!(ring_f32(2), F32_SLOT * ring_slots(32, GDN_LAYERS));

    // The full packed session entry: the plan formula with the real helper
    // values, for every QSA K/V format, with and without native MTP.
    for qsa in [QsaKvFormat::Q8, QsaKvFormat::F32, QsaKvFormat::Fp8] {
        for native in [false, true] {
            for p in [0usize, 1, 5, 1536, 4096] {
                let mut entry = [0u64; 2];
                for (slot, gdn) in FORMATS.into_iter().enumerate() {
                    let format = format_of(qsa, gdn);
                    entry[slot] = session_snapshot_bytes(&config, format, native, p).unwrap();
                    assert_eq!(
                        entry[slot],
                        plan_entry_bytes(&config, format, native, p),
                        "{qsa:?} {gdn:?} native={native} p={p}"
                    );
                }
                // The precision-only growth per entry is exactly the delta of
                // the 36 layers' recurrent fixed parts, whatever `p` is.
                assert_eq!(
                    entry[1] - entry[0],
                    (GDN_LAYERS * (F32_SLOT - Q8_SLOT)) as u64,
                    "{qsa:?} native={native} p={p}"
                );
            }
        }
    }
    // PLAN's Q8 QSA constants: K = 544 B, I = 256 B.
    let q8_qsa = format_of(QsaKvFormat::Q8, GdnStateFormat::Q8);
    assert_eq!(
        q8_qsa
            .qsa
            .kv_row_bytes(config.num_key_value_heads, config.head_dim),
        544
    );
    assert_eq!(
        config.indexer_kv_heads * config.indexer_head_dim * q8_qsa.qsa.index_dtype().size(),
        256
    );
}

#[test]
fn gdn_precision_owner_accounting_qsa_unchanged_at_max_context() {
    let config = compact_test_config();
    let backends = [
        crate::Qwen4ContextCommit::legacy(QWEN4_DEFAULT_CONTEXT),
        crate::Qwen4ContextCommit::new(
            crate::Qwen4KvBackend::Vmm,
            QWEN4_DEFAULT_CONTEXT,
            QWEN4_DEFAULT_CONTEXT,
            2 << 20,
        ),
        crate::Qwen4ContextCommit::legacy(262_144),
    ];
    for qsa in [QsaKvFormat::Q8, QsaKvFormat::F32, QsaKvFormat::Fp8] {
        for context in &backends {
            let q8 = format_of(qsa, GdnStateFormat::Q8);
            let f32 = format_of(qsa, GdnStateFormat::F32);
            // The QSA context arenas are not a function of the GDN format.
            assert_eq!(
                context.committed_layer_bytes(&config, q8.layout()),
                context.committed_layer_bytes(&config, f32.layout()),
                "{qsa:?} {context:?}"
            );
            // The whole state differs by exactly live + mirror recurrent sets.
            let delta = Qwen4State::device_bytes(&config, f32, context).unwrap()
                - Qwen4State::device_bytes(&config, q8, context).unwrap();
            assert_eq!(delta, 2 * SET_DELTA, "{qsa:?} {context:?}");
        }
        // A max-context session entry grows by the same delta, nothing else.
        for p in [QWEN4_DEFAULT_CONTEXT, 262_144] {
            let q8 = session_snapshot_bytes(&config, format_of(qsa, GdnStateFormat::Q8), true, p)
                .unwrap();
            let f32 = session_snapshot_bytes(&config, format_of(qsa, GdnStateFormat::F32), true, p)
                .unwrap();
            assert_eq!(f32 - q8, SET_DELTA as u64, "{qsa:?} p={p}");
        }
    }
}

#[test]
fn gdn_precision_owner_accounting_native_mtp_reserve_covers_the_owners() {
    let config = compact_test_config();
    let (max_seq, chunk, max_k) = (4096usize, 1536usize, 3usize);
    assert_eq!(native_mtp_window_rows(max_k), 4);
    let context = Qwen4ContextCommit::legacy(max_seq);
    for gdn in FORMATS {
        let format = format_of(QsaKvFormat::Q8, gdn);
        let state = Qwen4State::device_bytes(&config, format, &context).unwrap();
        let live = live_set(&config, gdn);
        for (index, &r) in RINGS.iter().enumerate() {
            let capture = Qwen4State::row_capture_bytes(&config, gdn, r).unwrap();
            let request =
                native_mtp_request_bytes(&config, max_seq, chunk, max_k, Some((gdn, r)), false)
                    .unwrap() as usize;
            // The ring is the request's capture term: nothing is dropped.
            assert!(request >= capture, "{gdn:?} R={r}");
            // The reserve charges live + mirror (state census) AND the full
            // ring; the steady owners are ring + mirror, the ring replacing
            // the live slot. The surplus is at least the live set.
            let reserve = state + request;
            let owners = state - live + capture;
            assert!(reserve >= owners, "{gdn:?} R={r}");
            assert!(reserve - owners >= live, "{gdn:?} R={r}");
            // Recurrent owners alone: reserve = live + mirror + 2R slots.
            let reserve_recurrent = 2 * live + ring_of(gdn, index);
            let owners_recurrent = ring_of(gdn, index) + live;
            assert_eq!(reserve_recurrent - owners_recurrent, live);
        }
    }
    // F32 raises the reserve by exactly (2 + 2R) recurrent-set deltas.
    for (index, &r) in RINGS.iter().enumerate() {
        let reserve = |gdn| {
            let format = format_of(QsaKvFormat::Q8, gdn);
            Qwen4State::device_bytes(&config, format, &context).unwrap() as u64
                + native_mtp_request_bytes(&config, max_seq, chunk, max_k, Some((gdn, r)), false)
                    .unwrap()
        };
        assert_eq!(
            reserve(GdnStateFormat::F32) - reserve(GdnStateFormat::Q8),
            (2 * SET_DELTA + ring_slots(r, GDN_LAYERS) * SLOT_DELTA) as u64,
            "R={r}"
        );
        assert_eq!(
            ring_f32(index) - ring_q8(index),
            ring_slots(r, GDN_LAYERS) * SLOT_DELTA
        );
    }
    // A ring narrower than the window never shrinks the charge.
    for gdn in FORMATS {
        assert_eq!(
            native_mtp_request_bytes(&config, max_seq, chunk, max_k, Some((gdn, 2)), false),
            native_mtp_request_bytes(&config, max_seq, chunk, max_k, Some((gdn, 4)), false)
        );
    }
}

#[test]
fn gdn_precision_owner_accounting_ring_growth_holds_old_and_new_rings() {
    let config = compact_test_config();
    let context = Qwen4ContextCommit::legacy(1024);
    for gdn in FORMATS {
        let state =
            Qwen4State::device_bytes(&config, format_of(QsaKvFormat::Q8, gdn), &context).unwrap();
        let live = live_set(&config, gdn);
        for (old, new) in [(4usize, 8usize), (8, 32), (4, 32)] {
            let ring = |r: usize| Qwen4State::row_capture_bytes(&config, gdn, r).unwrap();
            // The state census holds live + mirror; the first ring is built
            // beside the live slot (which stays charged as fixed state), so
            // creation's peak is exactly what the native-MTP reserve charges:
            // live + mirror + the full new ring.
            let creation = state + ring(new);
            // `ensure_row_capture` builds the new ring before it frees the
            // old one: growing an existing ring holds the old ring (its first
            // slot is the live state) and the new one beside the mirror.
            let growth = (state - live) + ring(old) + ring(new);
            assert_eq!(
                growth - creation,
                ring(old) - live,
                "{gdn:?} {old}->{new}: the old ring minus the live slot it contains"
            );
        }
    }
    // Growth peak delta, F32 minus Q8: old + new rings and one mirror.
    for (old, new) in [(4usize, 8usize), (8, 32)] {
        let ring = |gdn, r| Qwen4State::row_capture_bytes(&config, gdn, r).unwrap();
        let peak = |gdn| ring(gdn, old) + ring(gdn, new) + live_set(&config, gdn);
        assert_eq!(
            peak(GdnStateFormat::F32) - peak(GdnStateFormat::Q8),
            SET_DELTA + (ring_slots(old, GDN_LAYERS) + ring_slots(new, GDN_LAYERS)) * SLOT_DELTA
        );
    }
}

#[test]
fn gdn_precision_owner_accounting_pp_and_lane_counts() {
    let config = compact_test_config();
    let context = Qwen4ContextCommit::legacy(1024);
    let q8 = format_of(QsaKvFormat::Q8, GdnStateFormat::Q8);
    let f32 = format_of(QsaKvFormat::Q8, GdnStateFormat::F32);
    let slot_delta = F32_SLOT - Q8_SLOT;
    for counts in [&[9usize, 8, 8, 8, 8, 7][..], &[24, 24][..]] {
        let mut gdn_total = 0;
        for scope in pp_scopes(counts) {
            // PP bands charge their own GDN layers, nothing divided.
            let (gdn_layers, _) = scope_layer_counts(&config, &scope);
            gdn_total += gdn_layers;
            let band = Qwen4State::device_bytes_scoped(&config, f32, &context, &scope).unwrap()
                - Qwen4State::device_bytes_scoped(&config, q8, &context, &scope).unwrap();
            assert_eq!(
                band,
                2 * gdn_layers * slot_delta,
                "{counts:?} {:?}",
                scope.band
            );
            for r in RINGS {
                let ring = |gdn| {
                    Qwen4State::row_capture_bytes_scoped(&config, gdn, r, gdn_layers).unwrap()
                };
                assert_eq!(
                    ring(GdnStateFormat::F32) - ring(GdnStateFormat::Q8),
                    ring_slots(r, gdn_layers) * slot_delta,
                    "{counts:?} {:?} R={r}",
                    scope.band
                );
            }
        }
        assert_eq!(gdn_total, GDN_LAYERS, "{counts:?}");
    }
    // EP replicates the recurrent state: a rank's census is the whole-model
    // single-scope one whatever the rank count (no rank parameter exists).
    let single = Qwen4BandScope::single(&config);
    assert_eq!(
        Qwen4State::device_bytes_scoped(&config, f32, &context, &single),
        Qwen4State::device_bytes(&config, f32, &context)
    );
    // Lanes multiply the owners: per AR lane a live state (no snapshot
    // mirror), per MTP lane a live state + mirror and a (k + 1)-row ring.
    for k in [None, Some(3usize)] {
        let store = |format, lanes| {
            crate::lane::Qwen4LaneStore::device_bytes(&config, format, &context, lanes, 0, k)
                .unwrap()
        };
        let one = store(f32, 1) - store(q8, 1);
        let recurrent_sets = if k.is_some() { 2 } else { 1 };
        let per_lane = recurrent_sets * GDN_LAYERS * slot_delta
            + k.map_or(0, |k| ring_slots(k + 1, GDN_LAYERS) * slot_delta);
        assert_eq!(one, per_lane as u64, "{k:?}");
        for lanes in [2usize, 4, 8] {
            assert_eq!(
                store(f32, lanes) - store(q8, lanes),
                one * lanes as u64,
                "{k:?} lanes={lanes}"
            );
        }
    }
}

/// An AR lane has no snapshot mirror, so a lane of the store costs exactly the
/// mirror less than a snapshot-capable state (the resident state, an MTP
/// lane). Per state, Flash-Next (36 GDN + 12 QSA layers, QSA fp8/q8 index
/// dtype BF16): every GDN layer's recurrent slot and conv history, per QSA
/// layer the partial K (2048 B), partial V (8192 B), the 2051-entry selection
/// and the 4 x 128 BF16 raw-index circular block, and the PLE conv history
/// (9 x 4 x 2560 F32) plus hyper feedback. Context independent; the figures
/// are storage arithmetic from the owners, not measured.
#[test]
fn gdn_precision_owner_accounting_ar_lane_has_no_snapshot_mirror() {
    let config = compact_test_config();
    let single = Qwen4BandScope::single(&config);
    let conv = StateGeometry::new(
        &config,
        &format_of(QsaKvFormat::Q8, GdnStateFormat::Q8),
        1024,
    )
    .expect("geometry")
    .conv_len
        * std::mem::size_of::<f32>();
    assert_eq!(conv, 122_880);
    let qsa = 2_048 + 8_192 + 2_051 * 4 + 4 * 128 * 2;
    assert_eq!(qsa, 19_468);
    let per_request = 9 * 2560 * 4 * 4 + 4 * 2560 * 4;
    assert_eq!(per_request, 409_600);
    let contexts = [
        Qwen4ContextCommit::legacy(1024),
        Qwen4ContextCommit::new(Qwen4KvBackend::Vmm, 16384, 4096, 2 << 20),
    ];
    for (gdn, mirror, eight_lanes) in [
        (GdnStateFormat::Q8, 34_263_184usize, 274_105_472usize),
        (GdnStateFormat::F32, 118_313_104, 946_504_832),
    ] {
        assert_eq!(
            GDN_LAYERS * (slot_of(gdn) + conv) + config.n_full_layers() * qsa + per_request,
            mirror,
            "{gdn:?}"
        );
        assert_eq!(8 * mirror, eight_lanes, "{gdn:?}");
        let format = format_of(QsaKvFormat::Q8, gdn);
        for context in &contexts {
            let state = |mirror| {
                Qwen4State::device_bytes_with_mirror(&config, format, context, &single, mirror)
                    .unwrap()
            };
            assert_eq!(
                state(StateMirror::Snapshot) - state(StateMirror::Absent),
                mirror,
                "{gdn:?} {context:?}"
            );
            // The store of AR lanes charges the live state per lane, the MTP
            // store the mirror too; the step buffers are the same.
            for lanes in [1usize, 2, 8] {
                let step = (2 * lanes * (config.vocab_size * 4 + 4)) as u64;
                let ar = crate::lane::Qwen4LaneStore::device_bytes(
                    &config, format, context, lanes, 0, None,
                )
                .unwrap();
                assert_eq!(
                    ar,
                    lanes as u64 * state(StateMirror::Absent) as u64 + step,
                    "{gdn:?} {context:?} lanes={lanes}"
                );
            }
        }
    }
}

#[test]
fn gdn_precision_owner_accounting_checked_overflow() {
    let config = compact_test_config();
    for gdn in FORMATS {
        let format = format_of(QsaKvFormat::Q8, gdn);
        for native in [false, true] {
            assert_eq!(
                session_snapshot_bytes(&config, format, native, usize::MAX),
                None
            );
            assert_eq!(
                session_snapshot_bytes(&config, format, native, usize::MAX / 2),
                None
            );
        }
        for rows in [usize::MAX, usize::MAX / 2, 1usize << 62] {
            assert_eq!(Qwen4State::row_capture_bytes(&config, gdn, rows), None);
            assert_eq!(
                Qwen4State::row_capture_bytes_scoped(&config, gdn, rows, 36),
                None
            );
            assert_eq!(
                native_mtp_request_bytes(&config, 4096, 1536, 3, Some((gdn, rows)), false),
                None
            );
        }
        // Rings below two rows hold nothing.
        assert_eq!(Qwen4State::row_capture_bytes(&config, gdn, 1), Some(0));
        let huge = Qwen4ContextCommit::legacy(usize::MAX);
        assert_eq!(Qwen4State::device_bytes(&config, format, &huge), None);
        assert!(crate::lane::Qwen4LaneStore::device_bytes(
            &config,
            format,
            &Qwen4ContextCommit::legacy(1024),
            usize::MAX,
            0,
            None,
        )
        .is_none());
        assert!(matches!(
            StateGeometry::new(&config, &format, usize::MAX),
            Err(StateError::DimensionOverflow)
        ));
        let geometry = StateGeometry::new(&config, &format, usize::MAX / 2).unwrap();
        let map = Qwen4StateMap::new(&config, &Qwen4BandScope::single(&config)).unwrap();
        assert!(matches!(
            state_tensor_plan(&config, &format, usize::MAX / 2, &geometry, &map),
            Err(StateError::DimensionOverflow)
        ));
    }
}

/// The session fixed parts in `session_layout` order (all GDN recurrent and
/// conv, then QSA selections, PLE conv, feedback), bytes per part.
fn planned_session_fixed_parts(config: &Qwen4Config, format: Qwen4StateFormat) -> Vec<usize> {
    let scope = Qwen4BandScope::single(config);
    let map = Qwen4StateMap::new(config, &scope).expect("state map");
    let geometry = StateGeometry::new(config, &format, 1024).expect("geometry");
    let plan = state_tensor_plan(config, &format, 1024, &geometry, &map).expect("tensor plan");
    let bytes = |spec: &StateTensorSpec| spec.elements * spec.dtype.size();
    let mut parts = Vec::new();
    for spec in &plan {
        if spec.key.starts_with("gdn[") {
            parts.push(bytes(spec));
        }
    }
    for spec in &plan {
        if spec.key.ends_with(".selected_indices") {
            parts.push(bytes(spec));
        }
    }
    for spec in &plan {
        if spec.key == "ple_conv" || spec.key == "hyper_feedback" {
            parts.push(bytes(spec));
        }
    }
    parts
}

#[test]
fn gdn_precision_owner_accounting_session_fixed_parts_are_format_exact() {
    let config = compact_test_config();
    let q8 = planned_session_fixed_parts(&config, format_of(QsaKvFormat::Q8, GdnStateFormat::Q8));
    let f32 = planned_session_fixed_parts(&config, format_of(QsaKvFormat::Q8, GdnStateFormat::F32));
    assert_eq!(q8.len(), f32.len());
    // Only the 36 recurrent parts (positions 0, 2, 4, ...) differ in length:
    // the restore's per-part length comparison (hipfire-runtime
    // `session_cache`, "snapshot layout mismatch") cannot accept a snapshot of
    // the other format, and nothing converts it.
    for (index, (&a, &b)) in q8.iter().zip(&f32).enumerate() {
        let recurrent = index < 2 * GDN_LAYERS && index % 2 == 0;
        assert_eq!(a != b, recurrent, "part {index}");
        if recurrent {
            assert_eq!((a, b), (Q8_SLOT, F32_SLOT), "part {index}");
        }
    }
    assert!(q8.iter().zip(&f32).any(|(a, b)| a != b));
    // The aligned fixed parts are exactly the p = 0 session entry.
    for gdn in FORMATS {
        let parts = planned_session_fixed_parts(&config, format_of(QsaKvFormat::Q8, gdn));
        let aligned: usize = parts.iter().map(|bytes| bytes.next_multiple_of(256)).sum();
        assert_eq!(
            session_snapshot_bytes(&config, format_of(QsaKvFormat::Q8, gdn), false, 0),
            Some(aligned as u64),
            "{gdn:?}"
        );
    }
}

#[test]
fn gdn_precision_owner_accounting_live_session_parts_differ_by_format() {
    let Some(mut gpu) = Gpu::init().ok() else {
        eprintln!("skip: live session parts need a GPU");
        return;
    };
    let config = compact_test_config();
    let mut lens = Vec::new();
    for gdn in FORMATS {
        let state = Qwen4State::new(
            &mut gpu,
            &config,
            8,
            Qwen4StateFormat {
                gdn,
                ..Qwen4StateFormat::F32
            },
        )
        .expect("state");
        let (_, layout) = state.session_parts().expect("session parts");
        let parts: Vec<usize> = layout.fixed.iter().map(|part| part.bytes).collect();
        assert_eq!(parts[0], slot_of(gdn), "{gdn:?}");
        assert_eq!(
            parts,
            planned_session_fixed_parts(
                &config,
                Qwen4StateFormat {
                    gdn,
                    ..Qwen4StateFormat::F32
                }
            ),
            "{gdn:?}"
        );
        lens.push(parts);
        state.free_gpu(&mut gpu).expect("free state");
    }
    assert_ne!(lens[0], lens[1]);
}
