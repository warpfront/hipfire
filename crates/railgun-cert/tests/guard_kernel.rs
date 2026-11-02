// SPDX-License-Identifier: Apache-2.0
//! The #867 guard kernels (`kernels/railgun_guard_pm_gfx1151.co`, emitted by
//! `peacemaker-isa emit --kernel railgun_guard`) through the certifier: both
//! guarded kernels carry the guard certificate on their ControlWord at +0,
//! the unguarded flag producer and the hipcc copy kernels carry none, and
//! every kernel is A1-Proven with exact modes.
use railgun_cert::{certify, KernelRecord};

fn arg(kernel: &KernelRecord, offset: u32) -> (Option<&str>, Vec<&str>) {
    let a = kernel.args.iter().find(|a| a.offset == offset).unwrap();
    (
        a.mode.as_deref(),
        a.classes.iter().map(String::as_str).collect(),
    )
}

#[test]
fn guarded_kernels_certify_their_no_op_path() {
    let receipt = certify(
        include_bytes!("../../../kernels/railgun_guard_pm_gfx1151.co"),
        None,
    )
    .expect("certifies");
    assert_eq!(receipt.arch, "gfx1151");
    let k = |symbol: &str| {
        receipt
            .railgun
            .kernels
            .iter()
            .find(|k| k.symbol == symbol)
            .unwrap_or_else(|| panic!("no {symbol}"))
    };
    let (add, step, set) = (
        k("railgun_guard_add_pm"),
        k("railgun_guard_step_pm"),
        k("railgun_flag_set_pm"),
    );
    for kernel in [add, step, set] {
        assert_eq!(
            kernel.status, "proven",
            "{}: {:?}",
            kernel.symbol, kernel.unresolved
        );
        assert_eq!(kernel.differential.status, "pass", "{}", kernel.symbol);
    }
    assert_eq!(
        (add.guard_flag, step.guard_flag, set.guard_flag),
        (Some(0), Some(0), None)
    );
    assert_eq!(arg(add, 0), (Some("read"), vec!["smem_load"]));
    assert_eq!(arg(add, 8).0, Some("read"));
    assert_eq!(arg(add, 16).0, Some("read_write"));
    let (mode, mut classes) = arg(step, 0);
    classes.sort();
    assert_eq!(
        (mode, classes),
        (Some("read_write"), vec!["smem_load", "vmem_store"])
    );
    assert_eq!(arg(step, 8).0, Some("read"));
    assert_eq!(arg(set, 0), (Some("write"), vec!["vmem_store"]));
}

#[test]
fn unguarded_objects_carry_no_guard() {
    let receipt = certify(include_bytes!("fixtures/railgun_copy.gfx1151.hsaco"), None).unwrap();
    assert!(receipt
        .railgun
        .kernels
        .iter()
        .all(|k| k.guard_flag.is_none()));
}
