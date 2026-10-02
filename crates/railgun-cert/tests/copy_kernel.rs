//! The railgun copy kernels (`kernels/src/railgun_copy.hip`, milestone MC)
//! through the certifier, on every arch railgun certifies them for. Both must
//! be A1-Proven with exact modes: `railgun_copy` reads `src` and writes
//! `dst`; `railgun_copy_batch` reads its slot table (scalar loads only),
//! reads the source slots `s0..s7` and writes the destination slots
//! `d0..d7`, and its scalars (`bytes`, `n`) are never pointers.
//!
//! Fixtures: the objects `railgun-jit-corpus build` compiles with the
//! runtime's JIT recipe from that source (source sha256
//! `a6de564c96f301fd0d88b14d22214f44c8f431694e0aa4bb4c498baacbc1a4fb`).
//!
//! `railgun_copy_ptrtable.gfx1201.hsaco` is the MC pointer-table kernel that
//! the slot table replaced (source sha256
//! `fbcc18a5d88639868e9b2cba6c5df94a06356e323d8f275329549dec612e7d53`), kept
//! as a G8 adversarial case: accesses through pointers loaded from memory
//! must stay `Unknown`.
use railgun_cert::{certify, KernelRecord};

fn mode(kernel: &KernelRecord, offset: u32) -> Option<&str> {
    kernel.args.iter().find(|a| a.offset == offset).and_then(|a| a.mode.as_deref())
}

fn kernel<'r>(receipt: &'r railgun_cert::Receipt, symbol: &str) -> &'r KernelRecord {
    receipt.railgun.kernels.iter().find(|k| k.symbol == symbol).unwrap_or_else(|| panic!("no {symbol}"))
}

#[test]
fn both_copy_kernels_are_proven_with_exact_modes_on_every_arch() {
    for (arch, object) in [
        ("gfx1201", &include_bytes!("fixtures/railgun_copy.gfx1201.hsaco")[..]),
        ("gfx1100", &include_bytes!("fixtures/railgun_copy.gfx1100.hsaco")[..]),
        ("gfx1151", &include_bytes!("fixtures/railgun_copy.gfx1151.hsaco")[..]),
    ] {
        let receipt = certify(object, None).unwrap_or_else(|e| panic!("{arch}: {e}"));
        assert_eq!(receipt.arch, arch);

        let copy = kernel(&receipt, "railgun_copy");
        assert_eq!(copy.status, "proven", "{arch}: {:?}", copy.unresolved);
        assert_eq!((mode(copy, 0), mode(copy, 8), mode(copy, 16)), (Some("read"), Some("write"), None), "{arch}");

        let batch = kernel(&receipt, "railgun_copy_batch");
        assert_eq!(batch.status, "proven", "{arch}: {:?}", batch.unresolved);
        assert_eq!((mode(batch, 0), mode(batch, 8)), (Some("read"), None), "{arch}: table read, n a scalar");
        let table = batch.args.iter().find(|a| a.offset == 0).unwrap();
        assert_eq!(table.classes, ["smem_load"], "{arch}: the table is only read by scalar loads");
        for slot in 0..8 {
            assert_eq!(mode(batch, 16 + 8 * slot), Some("read"), "{arch}: source slot s{slot}");
            assert_eq!(mode(batch, 80 + 8 * slot), Some("write"), "{arch}: destination slot d{slot}");
        }

        for k in [copy, batch] {
            assert_eq!(k.differential.status, "pass", "{arch} {}: {:?}", k.symbol, k.differential.args);
        }
    }
}

#[test]
fn copies_through_table_loaded_pointers_stay_unknown() {
    let receipt = certify(include_bytes!("fixtures/railgun_copy_ptrtable.gfx1201.hsaco"), None).expect("certifies");
    let batch = kernel(&receipt, "railgun_copy_batch");
    assert_eq!(mode(batch, 0), Some("read"), "the pointer table is read-only");
    assert_eq!(batch.status, "unknown");
    assert!(
        !batch.unresolved.is_empty() && batch.unresolved.iter().all(|u| u.rule == "base-memory-derived"),
        "{:?}",
        batch.unresolved
    );
    assert_eq!(batch.differential.status, "pass", "{:?}", batch.differential.args);
}
