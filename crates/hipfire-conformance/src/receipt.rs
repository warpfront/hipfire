//! Executed-recipe receipts, parsed from the module-load audit lines
//! (`HIPFIRE_MODULE_LOAD_AUDIT=1`, `hip-bridge/src/ffi.rs`):
//!
//! ```text
//! HIPFIRE_MODULE_LOAD_AUDIT\tload\t<handle>\t<path>
//! HIPFIRE_MODULE_LOAD_AUDIT\tload_data\t<handle>\t<len>\t<fnv64 hex>
//! HIPFIRE_MODULE_LOAD_AUDIT\tget_function\t<handle>\t<name>
//! HIPFIRE_MODULE_LOAD_AUDIT\thip\t<json>        (rdna-compute/src/scratch.rs)
//! HIPFIRE_MODULE_LOAD_AUDIT\tembedded\t<json>   (rdna-compute/src/scratch.rs, dispatch.rs)
//! ```
//!
//! The two JSON kinds name where a module's code came from (compiled HIP
//! source or an image embedded in the binary); the card runner emits both.

use serde::Serialize;
use std::collections::BTreeSet;

const TAG: &str = "HIPFIRE_MODULE_LOAD_AUDIT";

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct LoadedImage {
    pub handle: String,
    /// File path, or `data:<len>:<fnv64>` for an embedded image.
    pub image: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Receipt {
    pub images: Vec<LoadedImage>,
    /// Resolved kernel symbols with the image they came from, in load order.
    pub functions: Vec<(String, String)>,
    /// `(origin, module, symbol)` from the `hip` and `embedded` lines.
    pub provenance: Vec<(String, String, String)>,
}

impl Receipt {
    /// Parse audit lines out of arbitrary stderr text; other lines are ignored.
    /// A malformed audit line is an error: a receipt must not silently shrink.
    pub fn parse(log: &str) -> Result<Receipt, String> {
        let mut r = Receipt::default();
        for (n, line) in log.lines().enumerate() {
            let mut f = line.split('\t');
            if f.next() != Some(TAG) {
                continue;
            }
            let bad = || format!("line {}: malformed audit line: {line}", n + 1);
            let kind = f.next().ok_or_else(bad)?;
            if kind == "hip" || kind == "embedded" {
                let body = f.next().ok_or_else(bad)?;
                let v: serde_json::Value = serde_json::from_str(body).map_err(|_| bad())?;
                let field = |k: &str| v.get(k).and_then(|x| x.as_str()).map(str::to_string);
                let (Some(module), Some(symbol)) = (field("module"), field("symbol")) else {
                    return Err(bad());
                };
                r.provenance.push((kind.to_string(), module, symbol));
                continue;
            }
            let handle = f.next().ok_or_else(bad)?.to_string();
            match kind {
                "load" => {
                    let path = f.next().ok_or_else(bad)?;
                    r.images.push(LoadedImage {
                        handle,
                        image: path.to_string(),
                    });
                }
                "load_data" => {
                    let len = f.next().ok_or_else(bad)?;
                    let fnv = f.next().ok_or_else(bad)?;
                    r.images.push(LoadedImage {
                        handle,
                        image: format!("data:{len}:{fnv}"),
                    });
                }
                "get_function" => {
                    let name = f.next().ok_or_else(bad)?;
                    r.functions.push((handle, name.to_string()));
                }
                _ => return Err(bad()),
            }
        }
        Ok(r)
    }

    pub fn function_names(&self) -> BTreeSet<&str> {
        self.functions.iter().map(|(_, n)| n.as_str()).collect()
    }

    /// Required kernels this arm never resolved. Empty means the intended
    /// recipe ran.
    pub fn missing(&self, required: &[String]) -> Vec<String> {
        let have = self.function_names();
        required
            .iter()
            .filter(|k| !have.contains(k.as_str()))
            .cloned()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_all_three_line_kinds() {
        let log = "noise\nHIPFIRE_MODULE_LOAD_AUDIT\tload\t0x1\t/a.hsaco\n\
HIPFIRE_MODULE_LOAD_AUDIT\tload_data\t0x2\t128\t00000000deadbeef\n\
HIPFIRE_MODULE_LOAD_AUDIT\tget_function\t0x2\tgemv_q8\n";
        let r = Receipt::parse(log).unwrap();
        assert_eq!(r.images.len(), 2);
        assert_eq!(r.images[1].image, "data:128:00000000deadbeef");
        assert_eq!(
            r.functions,
            vec![("0x2".to_string(), "gemv_q8".to_string())]
        );
        assert_eq!(
            r.missing(&["gemv_q8".into(), "gemm".into()]),
            vec!["gemm".to_string()]
        );
    }

    #[test]
    fn malformed_audit_line_is_an_error() {
        assert!(Receipt::parse("HIPFIRE_MODULE_LOAD_AUDIT\tload\t0x1\n").is_err());
        assert!(Receipt::parse("HIPFIRE_MODULE_LOAD_AUDIT\tbogus\t0x1\n").is_err());
        assert!(Receipt::parse("HIPFIRE_MODULE_LOAD_AUDIT\thip\tnot json\n").is_err());
        assert!(
            Receipt::parse("HIPFIRE_MODULE_LOAD_AUDIT\tembedded\t{\"arch\":\"gfx1201\"}\n")
                .is_err()
        );
    }

    #[test]
    fn provenance_lines_are_recorded() {
        let log = "HIPFIRE_MODULE_LOAD_AUDIT\thip\t{\"arch\":\"gfx1201\",\"module\":\"rope\",\"symbol\":\"rope_f32\"}\n\
HIPFIRE_MODULE_LOAD_AUDIT\tembedded\t{\"module\":\"silu_mul\",\"symbol\":\"silu_mul_f32\"}\n";
        let r = Receipt::parse(log).unwrap();
        assert_eq!(
            r.provenance,
            vec![
                ("hip".into(), "rope".into(), "rope_f32".into()),
                ("embedded".into(), "silu_mul".into(), "silu_mul_f32".into()),
            ]
        );
    }
}
