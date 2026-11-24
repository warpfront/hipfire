//! Executed-recipe receipts, parsed from the module-load audit lines
//! (`HIPFIRE_MODULE_LOAD_AUDIT=1`, `hip-bridge/src/ffi.rs`):
//!
//! ```text
//! HIPFIRE_MODULE_LOAD_AUDIT\tload\t<handle>\t<path>
//! HIPFIRE_MODULE_LOAD_AUDIT\tload_data\t<handle>\t<len>\t<fnv64 hex>
//! HIPFIRE_MODULE_LOAD_AUDIT\tget_function\t<handle>\t<name>
//! ```

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
            let handle = f.next().ok_or_else(bad)?.to_string();
            match kind {
                "load" => {
                    let path = f.next().ok_or_else(bad)?;
                    r.images.push(LoadedImage { handle, image: path.to_string() });
                }
                "load_data" => {
                    let len = f.next().ok_or_else(bad)?;
                    let fnv = f.next().ok_or_else(bad)?;
                    r.images.push(LoadedImage { handle, image: format!("data:{len}:{fnv}") });
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
        required.iter().filter(|k| !have.contains(k.as_str())).cloned().collect()
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
        assert_eq!(r.functions, vec![("0x2".to_string(), "gemv_q8".to_string())]);
        assert_eq!(r.missing(&["gemv_q8".into(), "gemm".into()]), vec!["gemm".to_string()]);
    }

    #[test]
    fn malformed_audit_line_is_an_error() {
        assert!(Receipt::parse("HIPFIRE_MODULE_LOAD_AUDIT\tload\t0x1\n").is_err());
        assert!(Receipt::parse("HIPFIRE_MODULE_LOAD_AUDIT\tbogus\t0x1\n").is_err());
    }
}
