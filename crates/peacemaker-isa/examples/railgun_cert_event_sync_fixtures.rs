// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Regenerates the `railgun-cert` event-sync fixtures
//! (`crates/railgun-cert/tests/fixtures/pm_*.gfx1201.hsaco`, read by
//! `crates/railgun-cert/tests/event_sync.rs`) from PeaceMaker's own kernels.
//! `railgun-cert` does not depend on `peacemaker-isa`, so the objects are
//! committed:
//!
//! ```text
//! cargo run -p peacemaker-isa --example railgun_cert_event_sync_fixtures
//! ```
//!
//! * `pm_event_handoff.gfx1201.hsaco`: `pd::handoff()` as emitted (a fenced
//!   release signal and an acquiring spin);
//! * `pm_event_handoff_no_release.gfx1201.hsaco`: the same stream with the two
//!   waits before the producer's atomic removed;
//! * `pm_event_handoff_no_acquire.gfx1201.hsaco`: the same stream with the
//!   `global_inv` removed;
//! * `pm_persistent_dispatch.gfx1201.hsaco`: the toy persistent dispatcher
//!   (claim queue plus a release completion counter);
//! * `pm_event_handoff_sys.gfx1201.hsaco`: the handoff at system scope (a
//!   `global_wb scope:SCOPE_SYS` before the release waits, `SCOPE_SYS` on the
//!   signal, the poll and the `global_inv`);
//! * `pm_event_handoff_sys_no_wb…`: the same without the `global_wb`, and with
//!   the `global_inv` narrowed to `SCOPE_DEV` below the poll's `SCOPE_SYS`;
//! * `pm_event_handoff_load_before_inv…`: a data load between the spin's exit
//!   wait and its `global_inv`.
//!
//! The first two mutations are `mutate` of `pm_persistent_smoke.rs`.
use peacemaker_isa::kernels::persistent_dispatch_gfx1201 as pd;
use peacemaker_isa::{native, sync, Arch};

/// Remove the release waits before the event signal (the two lines before the
/// producer's `global_atomic_add_u32`) or the acquire's `global_inv`.
fn mutate(text: &str, drop_release: bool, drop_acquire: bool) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let signal = lines
        .iter()
        .position(|l| l.trim().starts_with("global_atomic_add_u32"))
        .expect("producer signal");
    lines
        .iter()
        .enumerate()
        .filter(|(i, l)| {
            let t = l.trim();
            !(drop_release && (*i + 2 == signal || *i + 1 == signal) && t.starts_with("s_wait_"))
                && !(drop_acquire && t.starts_with("global_inv"))
        })
        .map(|(_, l)| format!("{l}\n"))
        .collect()
}

/// The handoff at system scope: the producer's signal, the consumer's poll
/// and the `global_inv` take `SCOPE_SYS` (the `global_inv` takes `inv_scope`);
/// `wb` puts a `global_wb scope:SCOPE_SYS` before the release waits;
/// `extra_load` inserts a data load between the spin's exit wait and the
/// `global_inv`.
fn sys_variant(text: &str, wb: bool, inv_scope: &str, extra_load: bool) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let signal = lines
        .iter()
        .position(|l| l.trim().starts_with("global_atomic_add_u32"))
        .expect("producer signal");
    let first_wait = signal - 2;
    let mut out = String::new();
    for (i, l) in lines.iter().enumerate() {
        let t = l.trim();
        let mut emit = |s: &str| {
            out.push_str(s);
            out.push('\n');
        };
        if t.starts_with("global_atomic_add_u32")
            || (t.starts_with("global_load_b32") && t.contains("scope:SCOPE_DEV"))
        {
            emit(&t.replace("SCOPE_DEV", "SCOPE_SYS"));
        } else if t.starts_with("global_inv") {
            if extra_load {
                emit("global_load_b32 v8, v1, s[4:5]");
            }
            emit(&format!("global_inv scope:{inv_scope}"));
        } else {
            if wb && i == first_wait {
                emit("global_wb scope:SCOPE_SYS");
            }
            emit(l);
        }
    }
    out
}

fn main() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../railgun-cert/tests/fixtures");
    let write = |name: &str, text: &str| {
        let elf = native::assemble(text, Arch::Gfx1201).expect("assemble");
        std::fs::write(dir.join(name), elf).expect("write fixture");
    };
    let h = pd::handoff().expect("handoff");
    sync::verify(&h.program, &h.proof.syncs).expect("handoff proves");
    write("pm_event_handoff.gfx1201.hsaco", &h.s_text);
    write(
        "pm_event_handoff_no_release.gfx1201.hsaco",
        &mutate(&h.s_text, true, false),
    );
    write(
        "pm_event_handoff_no_acquire.gfx1201.hsaco",
        &mutate(&h.s_text, false, true),
    );
    write(
        "pm_event_handoff_sys.gfx1201.hsaco",
        &sys_variant(&h.s_text, true, "SCOPE_SYS", false),
    );
    write(
        "pm_event_handoff_sys_no_wb.gfx1201.hsaco",
        &sys_variant(&h.s_text, false, "SCOPE_DEV", false),
    );
    write(
        "pm_event_handoff_load_before_inv.gfx1201.hsaco",
        &sys_variant(&h.s_text, true, "SCOPE_SYS", true),
    );
    let d = pd::dispatcher().expect("dispatcher");
    write("pm_persistent_dispatch.gfx1201.hsaco", &d.s_text);
}
