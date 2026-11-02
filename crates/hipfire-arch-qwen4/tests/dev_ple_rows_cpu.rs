// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! CPU proof that a developer (synthetic) profile's embedded PLE rows load
//! through the bounded row store with the profile's own geometry (shard
//! count, rows per shard, row width, head count), not the Flash-Next
//! constants: the rows a [`RowFetch`] returns for hashed token ids are byte
//! equal to the rows stored in the artifact file.
//!
//! `HIPFIRE_QWEN4_DEV_PLE_ARTIFACT` names a developer artifact that declares
//! PLE with BF16 (source-exact) embedded rows, e.g. the S2 tiny dense
//! artifact.  Without it the test reports a skip.

#![cfg(feature = "qwen4-dev-profiles")]

use hipfire_arch_qwen4::{admit_hfqm_artifact, PleHistory, Qwen4PleRows};
use hipfire_runtime::external_rows::{RowFetch, RowStore};
use hipfire_runtime::hfq::HfqFile;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

const ARTIFACT_ENV: &str = "HIPFIRE_QWEN4_DEV_PLE_ARTIFACT";

#[test]
fn embedded_bf16_rows_of_a_synthetic_artifact_load_with_the_profile_geometry() {
    let Ok(path) = std::env::var(ARTIFACT_ENV) else {
        eprintln!("skip: set {ARTIFACT_ENV} to a developer artifact with embedded BF16 PLE rows");
        return;
    };
    let hfq = HfqFile::open(Path::new(&path)).expect("open developer artifact");
    let receipt = admit_hfqm_artifact(&hfq).expect("developer artifact admission");
    let ple = receipt
        .config
        .ple
        .clone()
        .expect("the artifact declares PLE");
    assert!(!receipt.config.profile.is_flash_next());
    let metadata = receipt.ple.clone().expect("PLE hash metadata");
    assert!(matches!(receipt.ple_rows, Qwen4PleRows::Embedded));

    let descriptors = receipt
        .ple_row_descriptors(&hfq)
        .expect("shard descriptors");
    assert_eq!(descriptors.len(), ple.shard_count);
    for descriptor in &descriptors {
        assert_eq!(descriptor.logical_shape, [ple.shard_rows, ple.row_width]);
        assert_eq!(
            descriptor.dtype, "BF16",
            "this proof reads BF16 (source-exact) rows byte for byte"
        );
    }
    let row_bytes = ple.row_width * 2;

    // One request of 4 tokens prefetches `4 * head_count` rows: the staging is
    // sized from the profile's head count.
    let tokens = [1u32, 2, 3, 1];
    let staging_rows = tokens.len() * ple.head_count;
    let store = RowStore::with_staging_rows(
        "qwen4-dev-ple-test",
        descriptors.clone(),
        metadata.valid_rows(),
        staging_rows,
        true,
    )
    .expect("row store over the profile's shards");
    assert!(store.max_rows_per_prefetch() / ple.head_count >= tokens.len());

    let ids = PleHistory::new(receipt.config.eos_token_id).row_ids(&metadata, &tokens);
    assert_eq!(ids.len(), tokens.len() * ple.head_count);
    let mut fetch = RowFetch::begin(&store).expect("begin epoch");
    fetch.prefetch(ids.clone()).expect("prefetch hashed ids");
    let rows = fetch
        .wait()
        .expect("rows complete")
        .as_bytes()
        .expect("lease bytes")
        .to_vec();
    assert_eq!(rows.len(), ids.len() * row_bytes);

    let mut file = std::fs::File::open(&path).expect("open artifact bytes");
    for (slot, &id) in ids.iter().enumerate() {
        let shard = (id / ple.shard_rows as u64) as usize;
        let row = (id % ple.shard_rows as u64) as usize;
        let mut expected = vec![0u8; row_bytes];
        file.seek(SeekFrom::Start(
            descriptors[shard].offset + (row * row_bytes) as u64,
        ))
        .expect("seek row");
        file.read_exact(&mut expected).expect("read row");
        assert_eq!(
            &rows[slot * row_bytes..(slot + 1) * row_bytes],
            expected.as_slice(),
            "row {slot} (id {id}, shard {shard}, row {row})"
        );
    }
    fetch.complete();
    drop(fetch);
    store.unload().expect("unload row store");
}
