use peacemaker_isa::kernels::gemv_mq2_xbatch_pm_gfx1201::{
    emit_bundle, emit_elf, emit_text, emit_text_prefixed, SHIPPED,
};
use sha2::{Digest, Sha256};

const SHIPPED_HXACO: &[u8] = include_bytes!("../../../kernels/gemv_mq2_xbatch_pm_gfx1201.hxaco");

/// The device ELF payload of an uncompressed offload bundle (second entry).
fn bundle_elf(bundle: &[u8]) -> &[u8] {
    let u64_at = |at: usize| u64::from_le_bytes(bundle[at..at + 8].try_into().unwrap()) as usize;
    assert_eq!(&bundle[..24], b"__CLANG_OFFLOAD_BUNDLE__");
    let count = u64_at(24);
    let mut at = 32;
    let mut entries = Vec::new();
    for _ in 0..count {
        let (off, size, tl) = (u64_at(at), u64_at(at + 8), u64_at(at + 16));
        entries.push((off, size));
        at += 24 + tl;
    }
    let (off, size) = entries[1];
    &bundle[off..off + size]
}

#[test]
fn shipped_elf_matches_embedded_elf() {
    let elf = emit_elf(&SHIPPED).unwrap();
    let embedded = bundle_elf(SHIPPED_HXACO);
    assert_eq!(elf, embedded);
    let sha = format!("{:x}", Sha256::digest(&elf));
    assert_eq!(
        sha,
        "0e9b63c7e3ac07c9f73e3c9e1ad86451c81c24a2ec58e24741a38d876b221bd4"
    );
}

#[test]
fn shipped_bundle_is_byte_exact() {
    assert!(emit_bundle(&SHIPPED).unwrap() == SHIPPED_HXACO);
    let sha = include_str!("../../../kernels/gemv_mq2_xbatch_pm_gfx1201.hxaco.sha256");
    assert_eq!(format!("{:x}", Sha256::digest(SHIPPED_HXACO)), sha);
}

#[test]
fn prefix_changes_only_labels() {
    let base = emit_text(&SHIPPED).unwrap();
    let q = emit_text_prefixed(&SHIPPED, ".Lq").unwrap();
    assert_ne!(base, q);
    assert_eq!(base.replace(".Lpx_", ".Lq_"), q);
}
