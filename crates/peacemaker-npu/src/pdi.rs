// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! PDI (programmable device image) wrapping one CDO, as loaded by the NPU firmware at CONFIG_CU.
//!
//! Layout (bootgen `versal/include/imageheadertable-versal.h`, partition header in
//! `versal/include/partitionheader-versal.h`; constants match what bootgen emits for an
//! `aie_image` with `id_code = 0x14ca8093`, `extended_id_code = 0x01`, one `type=cdo` partition):
//!   0x000  4 SMAP width-detection words
//!   0x010  image header table   (32 words, checksum last)
//!   0x090  image header         (16 words, checksum last)
//!   0x0D0  partition header     (32 words, checksum last)
//!   0x150  CDO, zero-padded to 16 bytes
//! Checksums are `!sum(words)` over the structure minus its checksum word.

const IHT_OFF: usize = 0x10;
const IH_OFF: usize = 0x90;
const PH_OFF: usize = 0xD0;
const DATA_OFF: usize = 0x150;

fn put(w: &mut [u32], byte_off: usize, v: u32) {
    w[byte_off / 4] = v;
}

fn checksum(w: &mut [u32], start: usize, words: usize) {
    let s = start / 4;
    let ck = !w[s..s + words - 1]
        .iter()
        .fold(0u32, |a, &x| a.wrapping_add(x));
    w[s + words - 1] = ck;
}

/// Build a PDI around `cdo` (full CDO words including its 5-word header).
pub fn build(cdo: &[u32]) -> Vec<u8> {
    let cdo_words = cdo.len();
    let padded = cdo_words.div_ceil(4) * 4;
    let total = DATA_OFF / 4 + padded;
    let mut w = vec![0u32; total];
    // SMAP bus-width detection pattern
    w[0..4].copy_from_slice(&[0x0000_00DD, 0x1122_3344, 0x5566_7788, 0x99AA_BBCC]);
    // ---- image header table ----
    put(&mut w, IHT_OFF, 0x0004_0000); // version
    put(&mut w, IHT_OFF + 0x04, 1); // imageTotalCount
    put(&mut w, IHT_OFF + 0x08, (IH_OFF / 4) as u32); // firstImageHeaderWordOffset
    put(&mut w, IHT_OFF + 0x0C, 1); // partitionTotalCount
    put(&mut w, IHT_OFF + 0x10, (PH_OFF / 4) as u32); // firstPartitionHeaderWordOffset
    put(&mut w, IHT_OFF + 0x18, 0x14CA_8093); // idCode
    put(&mut w, IHT_OFF + 0x28, 0x5050_4449); // identificationString "IDPP"
    put(&mut w, IHT_OFF + 0x2C, 0x0020_1020); // headerSizes: PH 0x20, IH 0x10, IHT 0x20 words
    put(&mut w, IHT_OFF + 0x30, ((DATA_OFF - IH_OFF) / 4) as u32); // totalMetaHdrLength (words)
    put(&mut w, IHT_OFF + 0x44, 1); // extendedIdCode
    checksum(&mut w, IHT_OFF, 32);
    // ---- image header ----
    put(&mut w, IH_OFF, (PH_OFF / 4) as u32); // partitionHeaderWordOffset
    put(&mut w, IH_OFF + 0x04, 1); // dataSectionCount
    let name = b"aie_image\0\0\0\0\0\0\0";
    for (i, c) in name.chunks(4).enumerate() {
        put(
            &mut w,
            IH_OFF + 0x10 + 4 * i,
            u32::from_le_bytes([c[0], c[1], c[2], c[3]]),
        );
    }
    put(&mut w, IH_OFF + 0x20, 0x1C00_0000); // imageId (AIE)
    checksum(&mut w, IH_OFF, 16);
    // ---- partition header ----
    put(&mut w, PH_OFF, padded as u32); // encrypted data word length
    put(&mut w, PH_OFF + 0x04, cdo_words as u32); // unencrypted data word length
    put(&mut w, PH_OFF + 0x08, padded as u32); // total partition word length
    put(&mut w, PH_OFF + 0x18, 0xFFFF_FFFF); // destination load address lo
    put(&mut w, PH_OFF + 0x1C, 0xFFFF_FFFF); // destination load address hi
    put(&mut w, PH_OFF + 0x20, (DATA_OFF / 4) as u32); // data word offset
    put(&mut w, PH_OFF + 0x24, 0x0200_0006); // partition attributes (as bootgen emits for type=cdo)
    put(&mut w, PH_OFF + 0x28, 1); // section count
    checksum(&mut w, PH_OFF, 32);
    // ---- data ----
    w[DATA_OFF / 4..DATA_OFF / 4 + cdo_words].copy_from_slice(cdo);
    w.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Extract the CDO words from a PDI built in this layout (single image / partition).
pub fn extract_cdo(pdi: &[u8]) -> Option<Vec<u32>> {
    let w: Vec<u32> = pdi
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let ph = *w.get(IHT_OFF / 4 + 4)? as usize;
    let unenc = *w.get(ph + 1)? as usize;
    let data = *w.get(ph + 8)? as usize;
    w.get(data..data + unenc).map(|s| s.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Re-wrapping the CDO lifted from a bootgen-built PDI must reproduce it byte for byte
    /// (checks every header field and checksum). Skipped if the reference file is absent.
    #[test]
    fn rewrap_reproduces_bootgen_pdi() {
        let Some(corpus) = crate::vendor_corpus() else {
            return;
        };
        let p = corpus.join("8309a7e39620419663f2aff6/main.pdi");
        let Ok(orig) = std::fs::read(&p) else {
            eprintln!("skip: {} missing", p.display());
            return;
        };
        let cdo = extract_cdo(&orig).unwrap();
        assert_eq!(build(&cdo), orig);
    }
}
