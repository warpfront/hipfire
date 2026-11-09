//! `peacemaker_rip::search`: `SEARCH_AXES` parsing and refusals, the byte-exact
//! `with_axes` splice, canonical point order, and a real kernel that compiles at
//! every point of a declared axis.

use peacemaker_isa::Arch;
use peacemaker_rip::search::{point_count, points, reference_point, search_axes, with_axes, Axis};
use std::path::Path;

const SRC: &str = r#"// header
const SEARCH_AXES: &[(&str, &str, &[&str])] = &[
    ("ROWS", "schedule", &["1", "2", "4"]),
    ("SPLIT", "schedule", &["false", "true"]),
    ("TAG", "schedule", &["a", "b\"q"]),
];
const ROWS: u8 = 4; // keep
pub const SPLIT: bool = false;
const TAG: &str = "a";
fn emit(spec: Spec) -> Result<Emitted, String> { Err(String::new()) }
"#;

fn pt(items: &[(&str, &str)]) -> Vec<(String, String)> {
    items
        .iter()
        .map(|(a, v)| (a.to_string(), v.to_string()))
        .collect()
}

fn differing_lines(a: &str, b: &str) -> Vec<usize> {
    let (a, b): (Vec<&str>, Vec<&str>) = (a.split('\n').collect(), b.split('\n').collect());
    assert_eq!(a.len(), b.len(), "line count changed");
    (0..a.len())
        .filter(|&i| a[i] != b[i])
        .map(|i| i + 1)
        .collect()
}

#[test]
fn parses_axes_in_declared_order() {
    let axes = search_axes(SRC, "t.rip").unwrap();
    assert_eq!(
        axes,
        vec![
            Axis {
                name: "ROWS".into(),
                class: "schedule".into(),
                values: vec!["1".into(), "2".into(), "4".into()],
                reference: "4".into(),
                line: 7,
                ty: "u8".into(),
            },
            Axis {
                name: "SPLIT".into(),
                class: "schedule".into(),
                values: vec!["false".into(), "true".into()],
                reference: "false".into(),
                line: 8,
                ty: "bool".into(),
            },
            Axis {
                name: "TAG".into(),
                class: "schedule".into(),
                values: vec!["a".into(), "b\"q".into()],
                reference: "a".into(),
                line: 9,
                ty: "&str".into(),
            },
        ]
    );
    assert_eq!(point_count(&axes), Some(12));
    assert_eq!(
        reference_point(&axes),
        pt(&[("ROWS", "4"), ("SPLIT", "false"), ("TAG", "a")])
    );
}

#[test]
fn no_search_axes_is_empty() {
    let src = "const ROWS: u8 = 4;\nfn emit(spec: Spec) -> Result<Emitted, String> { Err(String::new()) }\n";
    assert_eq!(search_axes(src, "t.rip").unwrap(), vec![]);
    assert_eq!(with_axes(src, "t.rip", &[]).unwrap(), src);
    assert_eq!(points(&[]), vec![Vec::<(String, String)>::new()]);
}

#[test]
fn splice_changes_only_the_literals() {
    let out = with_axes(
        SRC,
        "t.rip",
        &pt(&[("TAG", "b\"q"), ("ROWS", "2"), ("SPLIT", "true")]),
    )
    .unwrap();
    assert_eq!(differing_lines(SRC, &out), vec![7, 8, 9]);
    let lines: Vec<&str> = out.split('\n').collect();
    assert_eq!(lines[6], "const ROWS: u8 = 2; // keep");
    assert_eq!(lines[7], "pub const SPLIT: bool = true;");
    assert_eq!(lines[8], r#"const TAG: &str = "b\"q";"#);
    // Partial point: unnamed axes keep their declared literal.
    let one = with_axes(SRC, "t.rip", &pt(&[("ROWS", "1")])).unwrap();
    assert_eq!(differing_lines(SRC, &one), vec![7]);
    assert_eq!(
        one,
        SRC.replace("const ROWS: u8 = 4;", "const ROWS: u8 = 1;")
    );
    // The reference point splices to the same bytes (the literals are canonical here).
    let axes = search_axes(SRC, "t.rip").unwrap();
    assert_eq!(
        with_axes(SRC, "t.rip", &reference_point(&axes)).unwrap(),
        SRC
    );
    // The result is itself a valid declaration whose reference is the spliced point.
    let re = search_axes(&out, "t.rip").unwrap();
    assert_eq!(
        reference_point(&re),
        pt(&[("ROWS", "2"), ("SPLIT", "true"), ("TAG", "b\"q")])
    );
}

#[test]
fn splice_accepts_suffixed_hex_and_negative_literals() {
    let src = "const SEARCH_AXES: &[(&str, &str, &[&str])] = &[(\"A\", \"schedule\", &[\"16\", \"3\"]), (\"B\", \"schedule\", &[\"-1\", \"2\"])];\nconst A: u16 = 0x10u16;\nconst B: i8 = -1;\n";
    let axes = search_axes(src, "t.rip").unwrap();
    assert_eq!(axes[0].reference, "16");
    assert_eq!(axes[1].reference, "-1");
    let out = with_axes(src, "t.rip", &pt(&[("A", "3"), ("B", "2")])).unwrap();
    assert_eq!(out, src.replace("0x10u16", "3").replace("= -1;", "= 2;"));
}

#[test]
fn points_enumerate_last_axis_fastest() {
    let axes = search_axes(SRC, "t.rip").unwrap();
    let all = points(&axes);
    assert_eq!(all.len(), 12);
    assert_eq!(
        all[0],
        pt(&[("ROWS", "1"), ("SPLIT", "false"), ("TAG", "a")])
    );
    assert_eq!(
        all[1],
        pt(&[("ROWS", "1"), ("SPLIT", "false"), ("TAG", "b\"q")])
    );
    assert_eq!(
        all[2],
        pt(&[("ROWS", "1"), ("SPLIT", "true"), ("TAG", "a")])
    );
    assert_eq!(
        all[4],
        pt(&[("ROWS", "2"), ("SPLIT", "false"), ("TAG", "a")])
    );
    assert_eq!(
        all[11],
        pt(&[("ROWS", "4"), ("SPLIT", "true"), ("TAG", "b\"q")])
    );
    // Every point is distinct and the reference point is among them.
    for (i, p) in all.iter().enumerate() {
        assert!(!all[..i].contains(p));
    }
    assert!(all.contains(&reference_point(&axes)));
}

fn refused(src: &str, needle: &str) {
    let err = search_axes(src, "t.rip").expect_err(needle);
    assert!(err.starts_with("t.rip:"), "unlocated: {err}");
    assert!(err.contains(needle), "wanted `{needle}` in: {err}");
}

#[test]
fn refuses_malformed_declarations() {
    // Missing axis const.
    refused(
        &SRC.replace("const TAG: &str = \"a\";\n", ""),
        "axis const `TAG` is not declared",
    );
    // Multi-line axis const.
    refused(
        &SRC.replace("const ROWS: u8 = 4;", "const ROWS: u8 =\n    4;"),
        "must be one line",
    );
    // Duplicate axis const.
    refused(
        &format!("{SRC}const ROWS: u8 = 2;\n"),
        "declared more than once",
    );
    // Reference not in the values.
    refused(
        &SRC.replace("const ROWS: u8 = 4;", "const ROWS: u8 = 3;"),
        "not in its search values",
    );
    // Unknown type.
    refused(
        &SRC.replace("const ROWS: u8 = 4;", "const ROWS: f32 = 4.0;"),
        "unsupported type `f32`",
    );
    // Value out of range / wrong shape for the type.
    refused(
        &SRC.replace("&[\"1\", \"2\", \"4\"]", "&[\"1\", \"300\", \"4\"]"),
        "not a valid u8",
    );
    refused(
        &SRC.replace("&[\"false\", \"true\"]", "&[\"false\", \"yes\"]"),
        "not a valid bool",
    );
    // Duplicate axis, duplicate value, empty values.
    refused(
        &SRC.replace(
            "    (\"TAG\", \"schedule\", &[\"a\", \"b\\\"q\"]),\n",
            "    (\"ROWS\", \"schedule\", &[\"4\"]),\n",
        ),
        "declared twice",
    );
    refused(
        &SRC.replace("&[\"1\", \"2\", \"4\"]", "&[\"1\", \"4\", \"4\"]"),
        "twice",
    );
    refused(
        &SRC.replace("&[\"false\", \"true\"]", "&[]"),
        "has no values",
    );
    // Not a tuple of three.
    refused(
        &SRC.replace(
            "(\"SPLIT\", \"schedule\", &[\"false\", \"true\"])",
            "(\"SPLIT\", &[\"false\", \"true\"])",
        ),
        "(name, class, values) tuple",
    );
    // Declared twice.
    refused(
        &format!("{SRC}const SEARCH_AXES: &[(&str, &str, &[&str])] = &[];\n"),
        "`SEARCH_AXES` is declared more than once",
    );
    // Trailing junk after the literal.
    refused(
        &SRC.replace(
            "const ROWS: u8 = 4; // keep",
            "const ROWS: u8 = 4; const X: u8 = 1;",
        ),
        "must be one line",
    );
}

#[test]
fn refusal_positions_name_the_axis_const_line() {
    let err = search_axes(
        &SRC.replace("const ROWS: u8 = 4;", "const ROWS: u8 = 3;"),
        "t.rip",
    )
    .unwrap_err();
    assert!(err.starts_with("t.rip:7:1: "), "{err}");
}

#[test]
fn with_axes_refuses_bad_points() {
    for (point, needle) in [
        (pt(&[("NOPE", "1")]), "not a declared search axis"),
        (pt(&[("ROWS", "256")]), "not a valid u8"),
        (pt(&[("SPLIT", "maybe")]), "not a valid bool"),
        (pt(&[("ROWS", "1"), ("ROWS", "2")]), "set twice"),
    ] {
        let err = with_axes(SRC, "t.rip", &point).unwrap_err();
        assert!(err.contains(needle), "wanted `{needle}` in: {err}");
    }
}

/// The single complete kernel of a cookbook recipe (same extraction as `tests/run.rs`).
fn recipe(file: &str) -> (String, String) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("docs/cookbook")
        .join(file);
    let markdown =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut lines = markdown.lines();
    while let Some(line) = lines.next() {
        if let Some(info) = line.trim().strip_prefix("```rip") {
            let variant = info
                .split_whitespace()
                .find_map(|w| w.strip_prefix("variant="))
                .expect("recipe must declare variant=NAME")
                .to_string();
            let mut source = String::new();
            for body in lines.by_ref() {
                if body.trim() == "```" {
                    return (source, variant);
                }
                source.push_str(body);
                source.push('\n');
            }
        }
    }
    panic!("{}: no rip block", path.display());
}

#[test]
fn spliced_kernel_compiles_at_every_point() {
    let (kernel, variant) = recipe("01-minimal-kernel.md");
    // `C` is the SGPR that holds the loaded constant; any free SGPR works.
    let src = format!(
        "const SEARCH_AXES: &[(&str, &str, &[&str])] = &[(\"C\", \"schedule\", &[\"8\", \"10\"])];\n{kernel}"
    );
    let axes = search_axes(&src, "fill.rip").unwrap();
    assert_eq!(axes.len(), 1);
    assert_eq!(axes[0].reference, "8");
    let original = peacemaker_rip::compile_named(&kernel, "fill.rip", Arch::Gfx1201, &variant)
        .expect("the recipe compiles");
    let mut texts = Vec::new();
    for point in points(&axes) {
        let candidate = with_axes(&src, "fill.rip", &point).unwrap();
        // Exactly the axis const's line changes, and only away from the reference point.
        assert_eq!(
            differing_lines(&src, &candidate).len(),
            usize::from(point[0].1 != "8")
        );
        let checked = peacemaker_rip::check(&candidate, "fill.rip", Arch::Gfx1201, &variant);
        let emitted = checked
            .emitted
            .unwrap_or_else(|| panic!("C={} failed: {:?}", point[0].1, checked.diagnostics));
        texts.push(emitted.s_text);
    }
    // The reference point is the shipped kernel byte for byte; the other point moved a register.
    assert_eq!(texts[0], original.s_text);
    assert_ne!(texts[0], texts[1]);
}
