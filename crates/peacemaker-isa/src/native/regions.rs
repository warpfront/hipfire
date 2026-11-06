//! `.note.pm_regions`: binary region metadata for tagged plan regions.
//!
//! Raw concatenated kernel records, no padding, little endian. Per kernel that has rows:
//! `"PMRG"`, `u32` version (1), `u32` symbol length + bytes, `u32` row count, eight `u16`
//! dispatcher register numbers, then rows sorted by template id ascending:
//! `u32 tid, u64 entry, u64 exit` (byte offsets from the kernel symbol, exit exclusive),
//! `u32` prefix length + bytes, `u16 wg_x, u16 wg_yz, u32 block, u32 dynamic_lds`,
//! 32 digest bytes, `u32` source symbol length + bytes.
//! A native row (no lift/provenance) has an empty prefix, `u16::MAX` workgroups, zero block,
//! zero dynamic LDS, a zero digest and an empty source symbol.

use std::collections::HashMap;

use super::Kernel;
use crate::plan::RegionProof;

const MAGIC: &[u8; 4] = b"PMRG";
const VERSION: u32 = 1;
/// Dispatcher lo/hi, tmpl, ka_off, idx, kernargs_pair_low, wg_x, wg_yz.
const DISPATCH_REGS: [u16; 8] = [72, 103, 95, 98, 94, 0, 102, 103];

struct Row<'a> {
    tid: u32,
    entry: u64,
    exit: u64,
    proof: &'a RegionProof,
}

fn put_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u32).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn resolve(
    name: &str,
    label: String,
    labels: &HashMap<&str, (usize, usize)>,
) -> Result<(usize, usize), String> {
    labels
        .get(label.as_str())
        .copied()
        .ok_or_else(|| format!("region `{name}`: label `{label}` not found"))
}

/// Encode the `.note.pm_regions` payload; empty (unallocated) when no region carries a template id.
pub(super) fn encode(
    regions: &[RegionProof],
    kernels: &[Kernel<'_>],
    labels: &HashMap<&str, (usize, usize)>,
) -> Result<Vec<u8>, String> {
    if regions.iter().all(|r| r.template_id.is_none()) {
        return Ok(Vec::new());
    }
    let mut per_kernel: Vec<Vec<Row<'_>>> = (0..kernels.len()).map(|_| Vec::new()).collect();
    for r in regions {
        let Some(tid) = r.template_id else { continue };
        let name = &r.name;
        let (ek, ew) = resolve(name, crate::plan::region_entry_label(name), labels)?;
        let (xk, xw) = resolve(name, crate::plan::region_exit_label(name), labels)?;
        if ek != xk {
            return Err(format!(
                "region `{name}`: entry and exit labels are in different kernels"
            ));
        }
        let kernel = kernels
            .get(ek)
            .ok_or_else(|| format!("region `{name}`: kernel index {ek} out of range"))?;
        if xw <= ew {
            return Err(format!(
                "region `{name}`: empty or inverted range (words {ew}..{xw})"
            ));
        }
        if xw > kernel.words.len() {
            return Err(format!(
                "region `{name}`: exit label beyond kernel `{}`",
                kernel.symbol
            ));
        }
        let present = [
            r.lift_inputs.is_some(),
            r.source_object_sha256.is_some(),
            r.source_symbol.is_some(),
        ];
        if present.iter().any(|p| *p) && !present.iter().all(|p| *p) {
            return Err(format!(
                "region `{name}`: partial lift provenance (lift_inputs, source_object_sha256, source_symbol must be all set or all absent)"
            ));
        }
        let rows = &mut per_kernel[ek];
        if rows.iter().any(|x| x.tid == tid) {
            return Err(format!(
                "duplicate template id {tid} in kernel `{}`",
                kernel.symbol
            ));
        }
        rows.push(Row {
            tid,
            entry: ew as u64 * 4,
            exit: xw as u64 * 4,
            proof: r,
        });
    }

    let mut out = Vec::new();
    for (kernel, rows) in kernels.iter().zip(per_kernel.iter_mut()) {
        if rows.is_empty() {
            continue;
        }
        rows.sort_by_key(|r| r.tid);
        let mut by_entry: Vec<&Row<'_>> = rows.iter().collect();
        by_entry.sort_by_key(|r| (r.entry, r.exit));
        for w in by_entry.windows(2) {
            if w[1].entry < w[0].exit {
                return Err(format!(
                    "regions `{}` and `{}` overlap in kernel `{}`",
                    w[0].proof.name, w[1].proof.name, kernel.symbol
                ));
            }
        }
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        put_str(&mut out, kernel.symbol);
        out.extend_from_slice(&(rows.len() as u32).to_le_bytes());
        for reg in DISPATCH_REGS {
            out.extend_from_slice(&reg.to_le_bytes());
        }
        for row in rows.iter() {
            out.extend_from_slice(&row.tid.to_le_bytes());
            out.extend_from_slice(&row.entry.to_le_bytes());
            out.extend_from_slice(&row.exit.to_le_bytes());
            match (&row.proof.lift_inputs, &row.proof.source_symbol) {
                (Some(l), Some(sym)) => {
                    put_str(&mut out, &l.prefix);
                    out.extend_from_slice(&l.workgroup_x.to_le_bytes());
                    out.extend_from_slice(&l.workgroup_yz.to_le_bytes());
                    out.extend_from_slice(&l.block.to_le_bytes());
                    out.extend_from_slice(&l.dynamic_lds.to_le_bytes());
                    out.extend_from_slice(&row.proof.source_object_sha256.unwrap_or_default());
                    put_str(&mut out, sym);
                }
                _ => {
                    put_str(&mut out, "");
                    out.extend_from_slice(&u16::MAX.to_le_bytes());
                    out.extend_from_slice(&u16::MAX.to_le_bytes());
                    out.extend_from_slice(&0u32.to_le_bytes());
                    out.extend_from_slice(&0u32.to_le_bytes());
                    out.extend_from_slice(&[0u8; 32]);
                    put_str(&mut out, "");
                }
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::RegionLiftInputs;

    fn kernel(symbol: &str, n: usize) -> Kernel<'_> {
        Kernel {
            symbol,
            words: vec![0; n],
            branches: Vec::new(),
            size: None,
            descriptor: None,
        }
    }

    fn proof(name: &str, tid: Option<u32>) -> RegionProof {
        RegionProof {
            name: name.into(),
            template_id: tid,
            ..Default::default()
        }
    }

    fn lifted(name: &str, tid: u32) -> RegionProof {
        RegionProof {
            lift_inputs: Some(RegionLiftInputs {
                prefix: "pfx".into(),
                workgroup_x: 64,
                workgroup_yz: 1,
                block: 7,
                dynamic_lds: 256,
            }),
            source_object_sha256: Some([0xab; 32]),
            source_symbol: Some("src".into()),
            ..proof(name, Some(tid))
        }
    }

    fn label_map<'a>(
        names: &'a [(String, String)],
        spec: &[(usize, usize, usize)],
    ) -> HashMap<&'a str, (usize, usize)> {
        let mut m = HashMap::new();
        for ((e, x), (k, ew, xw)) in names.iter().zip(spec) {
            m.insert(e.as_str(), (*k, *ew));
            m.insert(x.as_str(), (*k, *xw));
        }
        m
    }

    fn names(regions: &[&str]) -> Vec<(String, String)> {
        regions
            .iter()
            .map(|n| {
                (
                    crate::plan::region_entry_label(n),
                    crate::plan::region_exit_label(n),
                )
            })
            .collect()
    }

    fn u32at(b: &[u8], o: usize) -> u32 {
        u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
    }
    fn u64at(b: &[u8], o: usize) -> u64 {
        u64::from_le_bytes(b[o..o + 8].try_into().unwrap())
    }

    #[test]
    fn no_tagged_is_empty_without_allocation() {
        let ks = [kernel("k", 10)];
        let out = encode(&[proof("g", None)], &ks, &HashMap::new()).unwrap();
        assert!(out.is_empty());
        assert_eq!(out.capacity(), 0);
    }

    #[test]
    fn schema_lifted_and_native_sorted() {
        let ks = [kernel("k0", 20), kernel("k1", 20)];
        let rs = [lifted("b", 9), proof("a", Some(3)), proof("g", None)];
        let n = names(&["b", "a"]);
        let m = label_map(&n, &[(0, 10, 12), (0, 2, 5)]);
        let out = encode(&rs, &ks, &m).unwrap();
        assert_eq!(&out[0..4], b"PMRG");
        assert_eq!(u32at(&out, 4), 1);
        assert_eq!(u32at(&out, 8), 2);
        assert_eq!(&out[12..14], b"k0");
        assert_eq!(u32at(&out, 14), 2);
        let regs: Vec<u16> = (0..8)
            .map(|i| u16::from_le_bytes([out[18 + 2 * i], out[19 + 2 * i]]))
            .collect();
        assert_eq!(regs, [72, 103, 95, 98, 94, 0, 102, 103]);
        let mut o = 34;
        // native tid 3 first
        assert_eq!(u32at(&out, o), 3);
        assert_eq!(u64at(&out, o + 4), 8);
        assert_eq!(u64at(&out, o + 12), 20);
        o += 20;
        assert_eq!(u32at(&out, o), 0);
        o += 4;
        assert_eq!(&out[o..o + 4], &[0xff; 4]);
        o += 4;
        assert_eq!(u32at(&out, o), 0);
        assert_eq!(u32at(&out, o + 4), 0);
        o += 8;
        assert_eq!(&out[o..o + 32], &[0u8; 32]);
        o += 32;
        assert_eq!(u32at(&out, o), 0);
        o += 4;
        // lifted tid 9
        assert_eq!(u32at(&out, o), 9);
        assert_eq!(u64at(&out, o + 4), 40);
        assert_eq!(u64at(&out, o + 12), 48);
        o += 20;
        assert_eq!(u32at(&out, o), 3);
        assert_eq!(&out[o + 4..o + 7], b"pfx");
        o += 7;
        assert_eq!(u16::from_le_bytes([out[o], out[o + 1]]), 64);
        assert_eq!(u16::from_le_bytes([out[o + 2], out[o + 3]]), 1);
        assert_eq!(u32at(&out, o + 4), 7);
        assert_eq!(u32at(&out, o + 8), 256);
        o += 12;
        assert_eq!(&out[o..o + 32], &[0xab; 32]);
        o += 32;
        assert_eq!(u32at(&out, o), 3);
        assert_eq!(&out[o + 4..o + 7], b"src");
        assert_eq!(out.len(), o + 7);
    }

    #[test]
    fn deterministic_regardless_of_input_order() {
        let ks = [kernel("k", 30)];
        let n = names(&["a", "b"]);
        let m = label_map(&n, &[(0, 1, 3), (0, 5, 8)]);
        let x = encode(&[proof("a", Some(2)), proof("b", Some(1))], &ks, &m).unwrap();
        let n2 = names(&["b", "a"]);
        let m2 = label_map(&n2, &[(0, 5, 8), (0, 1, 3)]);
        let y = encode(&[proof("b", Some(1)), proof("a", Some(2))], &ks, &m2).unwrap();
        assert_eq!(x, y);
        // first row tid is 1 (region b)
        assert_eq!(u32at(&x, 12 + ks[0].symbol.len() + 4 + 16), 1);
    }

    #[test]
    fn failures() {
        let ks = [kernel("k0", 20), kernel("k1", 20)];
        let n = names(&["a", "b"]);
        // duplicate tid
        let m = label_map(&n, &[(0, 1, 3), (0, 5, 8)]);
        let e = encode(&[proof("a", Some(1)), proof("b", Some(1))], &ks, &m).unwrap_err();
        assert!(e.contains("duplicate"), "{e}");
        // cross-kernel
        let mut m = label_map(&n, &[(0, 1, 3), (0, 5, 8)]);
        m.insert(".La_exit", (1, 3));
        let e = encode(&[proof("a", Some(1))], &ks, &m).unwrap_err();
        assert!(e.contains("different kernels"), "{e}");
        // missing
        let e = encode(&[proof("a", Some(1))], &ks, &HashMap::new()).unwrap_err();
        assert!(e.contains("not found"), "{e}");
        // overlap
        let m = label_map(&n, &[(0, 1, 6), (0, 5, 8)]);
        let e = encode(&[proof("a", Some(1)), proof("b", Some(2))], &ks, &m).unwrap_err();
        assert!(e.contains("overlap"), "{e}");
        // empty
        let m = label_map(&n, &[(0, 4, 4), (0, 5, 8)]);
        let e = encode(&[proof("a", Some(1))], &ks, &m).unwrap_err();
        assert!(e.contains("empty"), "{e}");
        // partial provenance: missing digest
        let m = label_map(&n, &[(0, 1, 3), (0, 5, 8)]);
        let mut nodigest = lifted("a", 1);
        nodigest.source_object_sha256 = None;
        assert!(encode(&[nodigest], &ks, &m)
            .unwrap_err()
            .contains("partial"));
        // partial provenance
        let mut part = lifted("a", 1);
        part.source_symbol = None;
        assert!(encode(&[part], &ks, &m).unwrap_err().contains("partial"));
    }
}
