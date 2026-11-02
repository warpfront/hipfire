// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#![no_main]

use libfuzzer_sys::fuzz_target;
use peacemaker_ir::codec::gfx12::{decode, encode};

fuzz_target!(|bytes: &[u8]| {
    let mut input = arbitrary::Unstructured::new(bytes);
    let mut words = Vec::with_capacity(bytes.len() / 4);
    while let Ok(word) = input.bytes(4) {
        words.push(u32::from_le_bytes([word[0], word[1], word[2], word[3]]));
    }

    if let Ok((inst, consumed)) = decode(&words) {
        assert!(consumed > 0 && consumed <= words.len());
        let encoded = encode(&inst).expect("decoded instructions must encode");
        assert_eq!(encoded.as_slice(), &words[..consumed]);
    }
});
