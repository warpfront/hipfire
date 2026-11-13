// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Mid-decode client cancel keeps the committed turn prefix (no GPU).
//!
//! Pins the host-side contract of the cancel epilogue and of the checkpoint
//! prefix-hash attestation it relies on:
//!
//! - a cancel at a committed boundary whose cursors attest keeps the stream
//!   (`prompt` + the fed decode tokens, the pending seed excluded); anything the
//!   cursors cannot attest, or a route outside the validated configuration, takes
//!   the full reset;
//! - a checkpoint resumes only when its stored prefix fingerprint equals the
//!   fingerprint of the conversation prefix it would be restored onto; a
//!   mismatching or unattested checkpoint degrades to an earlier attested one or
//!   a cold prefill;
//! - composed: after a kept cancel, a re-render that diverges inside the
//!   cancelled turn resumes from the latest attested checkpoint instead of
//!   cold-prefilling (the state the old full reset wiped).

#![allow(clippy::all)]

use hipfire_arch_qwen35::speculative::{
    attested_checkpoint_positions, checkpoint_prefix_attests, checkpoint_prefix_hash,
    snapshot_alloc_failure_action, SnapshotAllocFailure,
};
use hipfire_generate::common::{
    cancel_disposition, spec_cancel_committed_stream, CancelDisposition,
};
use hipfire_generate::qwen::plan_from_rendered;
use hipfire_runtime::checkpoint_pool::{prefix_fingerprint, prefix_fingerprint_chain};

fn toks(n: u32) -> Vec<u32> {
    (0..n).map(|i| 1000 + i * 7 % 997).collect()
}

#[test]
fn cancel_at_attested_boundary_keeps_committed_prefix() {
    assert_eq!(
        cancel_disposition(true, Some(120), 120),
        CancelDisposition::KeepCommitted { len: 120 }
    );
}

#[test]
fn cancel_falls_back_to_full_reset_when_not_attestable() {
    // Host record and device cursor disagree (either direction).
    assert_eq!(
        cancel_disposition(true, Some(119), 120),
        CancelDisposition::FullReset
    );
    assert_eq!(
        cancel_disposition(true, Some(121), 120),
        CancelDisposition::FullReset
    );
    // The record could not be reconstructed.
    assert_eq!(
        cancel_disposition(true, None, 120),
        CancelDisposition::FullReset
    );
    // Nothing was fed: there is no committed prefix worth keeping.
    assert_eq!(
        cancel_disposition(true, Some(0), 0),
        CancelDisposition::FullReset
    );
    // Route/configuration outside the validated one (mesh, eviction, other target).
    assert_eq!(
        cancel_disposition(false, Some(120), 120),
        CancelDisposition::FullReset
    );
}

#[test]
fn spec_boundary_stream_is_prompt_plus_fed_decode_without_pending_seed() {
    let prompt = [1, 2, 3];
    // emitted: first token and two accepted, the last (12) is the pending seed.
    let emitted = [10, 11, 12];
    // Fed: prompt (3) + 2 decode tokens (10, 11); the seed 12 is not fed.
    assert_eq!(
        spec_cancel_committed_stream(&prompt, &emitted, 5, Some(12)),
        Some(vec![1, 2, 3, 10, 11])
    );
    // First loop top: only the first token emitted, nothing fed beyond the prompt.
    assert_eq!(
        spec_cancel_committed_stream(&prompt, &[10], 3, Some(10)),
        Some(vec![1, 2, 3])
    );
    // A non-committable terminal seed: every emitted token is fed.
    assert_eq!(
        spec_cancel_committed_stream(&prompt, &[10, 11], 5, None),
        Some(vec![1, 2, 3, 10, 11])
    );
}

#[test]
fn spec_boundary_refuses_inconsistent_cursors() {
    let prompt = [1, 2, 3];
    // position behind the prompt.
    assert_eq!(
        spec_cancel_committed_stream(&prompt, &[10], 2, Some(10)),
        None
    );
    // emitted longer than fed + pending seed (an unrecorded token was fed).
    assert_eq!(
        spec_cancel_committed_stream(&prompt, &[10, 11, 12], 4, Some(12)),
        None
    );
    // emitted shorter than fed (a fed token missing from the record).
    assert_eq!(
        spec_cancel_committed_stream(&prompt, &[10, 12], 5, Some(12)),
        None
    );
    // the pending seed is not the last emitted token.
    assert_eq!(
        spec_cancel_committed_stream(&prompt, &[10, 11, 12], 5, Some(11)),
        None
    );
}

#[test]
fn chain_fingerprint_equals_concatenated_prefix_fingerprint() {
    let t = toks(200);
    assert_eq!(
        prefix_fingerprint_chain(&t[..70], &t[70..160]),
        prefix_fingerprint(&t[..160])
    );
    assert_eq!(prefix_fingerprint_chain(&t, &[]), prefix_fingerprint(&t));
}

#[test]
fn checkpoint_hash_attests_exactly_its_prefix() {
    let conv = toks(300);
    let h = checkpoint_prefix_hash(128, (&conv[..100], &conv[100..128]));
    assert_ne!(h, 0);
    assert!(checkpoint_prefix_attests(h, 128, &conv));
    // One token of the conversation prefix differs: the checkpoint no longer attests.
    let mut edited = conv.clone();
    edited[40] ^= 1;
    assert!(!checkpoint_prefix_attests(h, 128, &edited));
    // A difference past the checkpoint position is irrelevant to it.
    let mut tail_edit = conv.clone();
    tail_edit[200] ^= 1;
    assert!(checkpoint_prefix_attests(h, 128, &tail_edit));
    // The same stored hash at another position never attests.
    assert!(!checkpoint_prefix_attests(h, 127, &conv));
    // A position beyond the conversation never attests.
    assert!(!checkpoint_prefix_attests(h, 128, &conv[..100]));
}

#[test]
fn unprovable_checkpoint_is_unattested() {
    let conv = toks(300);
    // Physical position != committed token count (KV eviction): hash 0.
    assert_eq!(checkpoint_prefix_hash(128, (&conv[..100], &[])), 0);
    // A stored 0 never attests, even against an empty-prefix coincidence.
    assert!(!checkpoint_prefix_attests(0, 128, &conv));
}

#[test]
fn hash_mismatch_degrades_resume_to_an_earlier_checkpoint_then_cold() {
    let prior = toks(400);
    let attested = |pos: usize| checkpoint_prefix_hash(pos, (&prior[..pos], &[]));
    let poisoned = |pos: usize| {
        // A checkpoint taken over a different stream (the drifted-state shape).
        let mut other = prior.clone();
        other[pos - 5] ^= 1;
        checkpoint_prefix_hash(pos, (&other[..pos], &[]))
    };
    // The render diverges from the conversation at token 350.
    let mut rendered = prior[..350].to_vec();
    rendered.extend_from_slice(&[7, 8, 9, 10]);

    // Ring: 128 good, 256 good, 320 poisoned, 360 beyond the divergence.
    let ring = [
        (128, attested(128)),
        (256, attested(256)),
        (320, poisoned(320)),
        (360, attested(360)),
    ];
    let positions = attested_checkpoint_positions(&ring, &prior);
    assert_eq!(positions, vec![128, 256, 360]);
    let plan = plan_from_rendered(&prior, rendered.clone(), true, &positions, true, "test");
    assert!(plan.cache_hit);
    assert_eq!(
        plan.resume_from,
        Some(256),
        "320 is poisoned and 360 is past the divergence"
    );
    assert_eq!(plan.start_pos, 256);
    assert_eq!(plan.new_tokens, rendered[256..].to_vec());

    // Only the poisoned checkpoint is in range: cold prefill, not a restore.
    let only_bad = [(320, poisoned(320))];
    let positions = attested_checkpoint_positions(&only_bad, &prior);
    assert!(positions.is_empty());
    let plan = plan_from_rendered(&prior, rendered.clone(), true, &positions, true, "test");
    assert!(!plan.cache_hit);
    assert_eq!(plan.resume_from, None);
    assert_eq!(plan.start_pos, 0);
    assert_eq!(plan.new_tokens, rendered);

    // An unattested (hash 0) checkpoint behaves like a mismatch.
    let unattested = [(256, 0)];
    assert!(attested_checkpoint_positions(&unattested, &prior).is_empty());
}

#[test]
fn kept_cancel_prefix_resumes_where_the_old_wipe_cold_prefilled() {
    // Prompt of 300 tokens; the cancelled decode fed 40 more before the cancel.
    let prompt = toks(300);
    let decoded: Vec<u32> = (0..41).map(|i| 5000 + i).collect();
    let position = prompt.len() + 40;
    let kept =
        spec_cancel_committed_stream(&prompt, &decoded, position, Some(*decoded.last().unwrap()))
            .expect("cursors attest the boundary");
    assert_eq!(kept.len(), position);
    assert_eq!(&kept[..prompt.len()], &prompt[..]);

    // The prompt-prefill ring (the decode takes no checkpoints on the spec route).
    let ring = [
        (128, checkpoint_prefix_hash(128, (&kept[..128], &[]))),
        (256, checkpoint_prefix_hash(256, (&kept[..256], &[]))),
    ];
    let positions = attested_checkpoint_positions(&ring, &kept);
    assert_eq!(positions, vec![128, 256]);

    // The client echoes the cancelled turn re-rendered: it matches the prompt and
    // diverges inside the partial turn (reasoning dropped by the template).
    let mut rendered = prompt.clone();
    rendered.extend_from_slice(&[9001, 9002]);
    rendered.extend_from_slice(&[6000, 6001, 6002]);
    let plan = plan_from_rendered(&kept, rendered.clone(), true, &positions, true, "test");
    assert!(
        plan.cache_hit,
        "a kept conversation resumes instead of prefilling cold"
    );
    assert_eq!(plan.resume_from, Some(256));
    assert_eq!(plan.cached_tokens, 256);

    // The old full reset left an empty conversation: nothing to hit or resume.
    let wiped = plan_from_rendered(&[], rendered.clone(), false, &[], true, "test");
    assert!(!wiped.cache_hit);
    assert_eq!(wiped.cached_tokens, 0);
    assert_eq!(wiped.new_tokens, rendered);
}

#[test]
fn snapshot_alloc_failure_recycles_the_oldest_instead_of_freezing_the_ring() {
    // Memory pressure after the fit gate said yes: a non-empty ring slides forward
    // on its oldest buffers (the cap/fit recycle path) ...
    assert_eq!(
        snapshot_alloc_failure_action(1),
        SnapshotAllocFailure::RecycleOldest
    );
    assert_eq!(
        snapshot_alloc_failure_action(6),
        SnapshotAllocFailure::RecycleOldest
    );
    // ... an empty ring has nothing to reuse: skip this position.
    assert_eq!(snapshot_alloc_failure_action(0), SnapshotAllocFailure::Skip);
}
