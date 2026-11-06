// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! A1 release/acquire classification on gfx12 (`atomic_release`,
//! `load_acquire`), against PeaceMaker's event-sync kernels.
//!
//! Fixtures (regenerate with `cargo run -p peacemaker-isa --example
//! railgun_cert_event_sync_fixtures`; `railgun-cert` itself has no
//! `peacemaker-isa` dependency, so the objects are committed):
//!
//! * `pm_event_handoff.gfx1201.hsaco`: `pm_event_handoff` as emitted. Its
//!   `event` argument (offset 8) is signalled with a fenced (`s_wait_loadcnt
//!   0`, `s_wait_storecnt 0`, `global_atomic_add_u32 ... scope:SCOPE_DEV`)
//!   release and spun on by a device-scope poll whose exit is `s_wait_loadcnt
//!   0; global_inv scope:SCOPE_DEV`;
//! * `pm_event_handoff_no_release…`: the same stream without the two release
//!   waits (a relaxed signal);
//! * `pm_event_handoff_no_acquire…`: the same stream without `global_inv`
//!   (a plain poll);
//! * `pm_event_handoff_sys…`: the handoff at system scope: `global_wb
//!   scope:SCOPE_SYS` before the release waits, `SCOPE_SYS` signal and poll,
//!   `global_inv scope:SCOPE_SYS`;
//! * `pm_event_handoff_sys_no_wb…`: the same without the `global_wb` and with
//!   a `SCOPE_DEV` `global_inv` under the `SCOPE_SYS` poll;
//! * `pm_event_handoff_load_before_inv…`: a data load between the spin's exit
//!   wait and its `global_inv`;
//! * `pm_persistent_dispatch…`: the toy persistent dispatcher. Its queue
//!   (offset 40) takes a relaxed returning claim increment and a fenced
//!   completion increment.
use railgun_cert::{certify, KernelRecord};

const HANDOFF: &[u8] = include_bytes!("fixtures/pm_event_handoff.gfx1201.hsaco");
const NO_RELEASE: &[u8] = include_bytes!("fixtures/pm_event_handoff_no_release.gfx1201.hsaco");
const NO_ACQUIRE: &[u8] = include_bytes!("fixtures/pm_event_handoff_no_acquire.gfx1201.hsaco");
const SYS: &[u8] = include_bytes!("fixtures/pm_event_handoff_sys.gfx1201.hsaco");
const SYS_NO_WB: &[u8] = include_bytes!("fixtures/pm_event_handoff_sys_no_wb.gfx1201.hsaco");
const LOAD_BEFORE_INV: &[u8] =
    include_bytes!("fixtures/pm_event_handoff_load_before_inv.gfx1201.hsaco");
const DISPATCH: &[u8] = include_bytes!("fixtures/pm_persistent_dispatch.gfx1201.hsaco");

fn arg_classes(object: &[u8], symbol: &str, offset: u32) -> Vec<String> {
    let receipt = certify(object, None).expect("certifies");
    let kernel: &KernelRecord = receipt
        .railgun
        .kernels
        .iter()
        .find(|k| k.symbol == symbol)
        .unwrap_or_else(|| panic!("no {symbol}"));
    assert_eq!(kernel.status, "proven", "{symbol}: {:?}", kernel.unresolved);
    let mut classes = kernel
        .args
        .iter()
        .find(|a| a.offset == offset)
        .unwrap_or_else(|| panic!("no arg at {offset}"))
        .classes
        .clone();
    classes.sort();
    classes
}

const EVENT: u32 = 8;

#[test]
fn fenced_handoff_event_is_release_and_acquire() {
    assert_eq!(
        arg_classes(HANDOFF, "pm_event_handoff", EVENT),
        ["atomic_release", "load_acquire"]
    );
}

#[test]
fn handoff_without_release_waits_stays_relaxed() {
    let classes = arg_classes(NO_RELEASE, "pm_event_handoff", EVENT);
    assert_eq!(classes, ["atomic_commutative", "load_acquire"]);
    assert!(!classes.iter().any(|c| c == "atomic_release"));
}

#[test]
fn handoff_without_global_inv_is_a_plain_load() {
    let classes = arg_classes(NO_ACQUIRE, "pm_event_handoff", EVENT);
    assert_eq!(classes, ["atomic_release", "vmem_load"]);
    assert!(!classes.iter().any(|c| c == "load_acquire"));
}

#[test]
fn dispatcher_queue_is_a_relaxed_claim_and_a_release_completion() {
    assert_eq!(
        arg_classes(DISPATCH, "pm_persistent_mq2_xbatch_dispatch", 40),
        ["atomic_commutative", "atomic_release"]
    );
}

#[test]
fn system_scope_release_needs_the_writeback_and_acquire_needs_its_scope() {
    assert_eq!(
        arg_classes(SYS, "pm_event_handoff", EVENT),
        ["atomic_release", "load_acquire"],
        "global_wb + waits + SCOPE_SYS atomic; SCOPE_SYS poll then global_inv SCOPE_SYS"
    );
    assert_eq!(
        arg_classes(SYS_NO_WB, "pm_event_handoff", EVENT),
        ["atomic_commutative", "vmem_load"],
        "no global_wb at system scope; global_inv DEV below a SYS poll"
    );
}

#[test]
fn a_load_between_the_wait_and_global_inv_voids_the_acquire() {
    assert_eq!(
        arg_classes(LOAD_BEFORE_INV, "pm_event_handoff", EVENT),
        ["atomic_release", "vmem_load"]
    );
}
