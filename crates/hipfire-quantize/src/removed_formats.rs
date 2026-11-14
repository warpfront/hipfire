//! `--format` values whose producers were removed in 0.5.0.
//!
//! No published artifact (registry/v1.json and every hipfire-models Hugging
//! Face file, read by header), shipped recipe, or validation fixture carries
//! the quant types these spellings produce, so the quantizer refuses them instead
//! of writing an artifact nothing ships. Loader and kernel support is removed
//! separately. User-facing list: docs/QUANTIZE.md ("Removed in 0.5.0").

/// (format string, qt it produced, nearest supported `--format`).
pub(crate) const REMOVED_FORMATS: &[(&str, &str, &str)] = &[
    ("q4f16", "0", "hfq4"),
    ("q8hfq", "5", "q8"),
    ("hfq3g256", "9", "mq3"),
    ("hf3", "10", "mq3"),
    ("hfq3", "10", "mq3"),
    ("hfq3g128", "10", "mq3"),
    ("hfq2g256", "11", "mq2-lloyd"),
    ("hf2", "12", "mq2-lloyd"),
    ("hfq2", "12", "mq2-lloyd"),
    ("hfq2g128", "12", "mq2-lloyd"),
    ("mq2", "18", "mq2-lloyd"),
    ("mq2g256", "18", "mq2-lloyd"),
    ("mf4p", "24", "mfp4e8"),
    ("mfp4", "24", "mfp4e8"),
    ("mfp4g32", "24", "mfp4e8"),
    ("deepseek4-mq4lloyd", "30", "deepseek4-mq3lloyd"),
    ("mq4-lloyd", "30", "mq4"),
    ("mq4g256-lloyd", "30", "mq4"),
    ("mq4lloyd", "30", "mq4"),
    ("mfp4-lloyd", "32", "mfp4e8"),
    ("mfp4g32-lloyd", "32", "mfp4e8"),
    ("mfp4l", "32", "mfp4e8"),
    ("mfp4lloyd", "32", "mfp4e8"),
    ("mfp4+p", "33", "mfp4e8"),
    ("mfp4-p", "33", "mfp4e8"),
    ("mfp4p", "33", "mfp4e8"),
    ("mfp3-e8", "36", "mfp4e8"),
    ("mfp3-e8-gptq", "36", "mfp4e8-gptq"),
    ("mfp3e8", "36", "mfp4e8"),
    ("mfp3e8-gptq", "36", "mfp4e8-gptq"),
    ("mfp2-e8", "37", "mfp4e8"),
    ("mfp2-e8-gptq", "37", "mfp4e8-gptq"),
    ("mfp2e8", "37", "mfp4e8"),
    ("mfp2e8-gptq", "37", "mfp4e8-gptq"),
    ("mq4c", "45", "mq4"),
    ("mq4cg256", "45", "mq4"),
    ("mq4g256c", "45", "mq4"),
    ("mq2g256v2", "50", "mq3v2"),
    ("mq2v2", "50", "mq3v2"),
];

/// Err naming the removal and the nearest supported format when `format` is
/// one of the [`REMOVED_FORMATS`].
pub(crate) fn refuse_removed_format(format: &str) -> Result<(), String> {
    match REMOVED_FORMATS.iter().find(|(name, _, _)| *name == format) {
        Some((name, qt, nearest)) => Err(format!(
            "--format {name} (qt {qt}) was removed in 0.5.0: no published artifact uses it; use --format {nearest}"
        )),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every `--format` spelling the quantizer still produces.
    const KEPT_FORMATS: &[&str] = &[
        "all-mq2-gptq",
        "antirez-mq",
        "antirez-mq-gptq",
        "bf16",
        "bf16-passthrough",
        "deepseek4-dense-mfp4e8soa-overlay",
        "deepseek4-dspark-e8soa",
        "deepseek4-dspark-mq2r",
        "deepseek4-mq2rxt-overlay",
        "deepseek4-mq3lloyd",
        "deepseek4-mtp-precise",
        "deepseek4-q8",
        "deepseek4-q8-mtp",
        "deepseek4-source",
        "deepseek4-source-precision",
        "ds4-dense-e8soa-overlay",
        "ds4-dspark-e8soa",
        "f16",
        "f16-passthrough",
        "f32",
        "f32-passthrough",
        "fast",
        "fp4",
        "hf4",
        "hf4p",
        "hf6",
        "hfp4",
        "hfp4g32",
        "hfq-mixed",
        "hfq4",
        "hfq4g256",
        "hfq6",
        "hfq6g256",
        "magnum",
        "maple",
        "maple-preview",
        "maple-ternary",
        "mfp4-e8",
        "mfp4-e8-gptq",
        "mfp4-e8-soa",
        "mfp4e8",
        "mfp4e8-gptq",
        "mfp4e8-soa",
        "mfp4e8soa",
        "mfp4l8",
        "mixed",
        "mq2-lloyd",
        "mq2-lloyd-anchored",
        "mq2-lloyd_anchored",
        "mq2g256-lloyd",
        "mq2g256-lloyd-anchored",
        "mq2g256-lloyd_anchored",
        "mq2lloyd",
        "mq2lloyd-anchored",
        "mq2lloyd_anchored",
        "mq3",
        "mq3-lloyd",
        "mq3g256",
        "mq3g256-lloyd",
        "mq3g256v2",
        "mq3lloyd",
        "mq3v2",
        "mq4",
        "mq4-mq2gl-exp",
        "mq4-mq2glexp",
        "mq4-mq2glexperts",
        "mq4-mq2lloyd-exp",
        "mq4-mq2lloyd-gptq",
        "mq4-mq2lloyd-gptq-all",
        "mq4-mq2lloyd-imatrix",
        "mq4-mq2lloyd-imatrix-kmap",
        "mq4-mq2lloyd-kmap",
        "mq4-mq2lloyd-kmap-imatrix",
        "mq4-mq2lloyd-kmap-promote",
        "mq4-mq2lloyd-native",
        "mq4-mq2lloyd-respectkmap",
        "mq4-mq2lloyd-routed",
        "mq4-mq2lloydexp",
        "mq4-mq2lloydexp-native",
        "mq4-mq2lloydexperts",
        "mq4-mq3lloyd-exp",
        "mq4-mq3lloyd-kmap",
        "mq4-mq3lloyd-routed",
        "mq4-mq6exp",
        "mq4-mq6experts",
        "mq4-mqlloyd-antirez",
        "mq4-mqlloyd-antirez-gptq",
        "mq4-mqlloyd-asym",
        "mq4-mqlloyd-asym-gptq",
        "mq4-mqlloyd-tiered",
        "mq4-mqlloyd-tiered-imatrix",
        "mq4g256",
        "mq4g256v2",
        "mq4g256v2-lloyd",
        "mq4g256v2l",
        "mq4l",
        "mq4v1",
        "mq4v2",
        "mq4v2-lloyd",
        "mq4v2l",
        "mq5",
        "mq5g256",
        "mq5g256v2",
        "mq5v2",
        "mq6",
        "mq6g256",
        "mq6g256v2",
        "mq6v2",
        "mq8",
        "mq8g256",
        "mqlloyd-tiered",
        "oracle",
        "q4k",
        "q4k-q8embed",
        "q8",
        "q8-fast",
        "q8-mixed",
        "q8f16",
        "qwen3-dspark-q8",
        "qwen35-dspark-q8",
    ];

    #[test]
    fn removed_formats_are_refused_with_name_and_nearest() {
        for (name, qt, nearest) in REMOVED_FORMATS {
            let err = refuse_removed_format(name).unwrap_err();
            assert!(err.contains(&format!("--format {name} (qt {qt})")), "{err}");
            assert!(err.contains("removed in 0.5.0"), "{err}");
            assert!(err.ends_with(&format!("use --format {nearest}")), "{err}");
            assert!(
                KEPT_FORMATS.contains(nearest),
                "{name} -> {nearest} is not a kept format"
            );
        }
    }

    #[test]
    fn kept_formats_are_accepted() {
        for name in KEPT_FORMATS {
            assert_eq!(refuse_removed_format(name), Ok(()), "{name}");
        }
    }

    #[test]
    fn kept_formats_are_known_to_the_pipeline() {
        let src = include_str!("pipeline.rs");
        for name in KEPT_FORMATS {
            assert!(
                src.contains(&format!("format == \"{name}\""))
                    || crate::pipeline::is_early_special_format(name),
                "{name} is no longer a pipeline format"
            );
        }
        for (name, _, _) in REMOVED_FORMATS {
            assert!(
                !src.contains(&format!("format == \"{name}\"")),
                "{name} still has a pipeline arm"
            );
        }
    }
}
