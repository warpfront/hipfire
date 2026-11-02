use peacemaker_rip::preprocess::lex;
use peacemaker_rip::srcfmt::{format_source, mask, unmask};
use std::path::PathBuf;

const KERNELS: [(&str, &str); 4] = [
    (
        "qsa_gather",
        include_str!("../../peacemaker-isa/src/kernels/qsa_gather.rip"),
    ),
    (
        "qsa_score",
        include_str!("../../peacemaker-isa/src/kernels/qsa_score.rip"),
    ),
    (
        "qsa_score_bf16",
        include_str!("../../peacemaker-isa/src/kernels/qsa_score_bf16.rip"),
    ),
    (
        "qsa_topk",
        include_str!("../../peacemaker-isa/src/kernels/qsa_topk.rip"),
    ),
];

fn rustfmt_present() -> bool {
    std::process::Command::new("rustfmt")
        .arg("--version")
        .output()
        .is_ok()
}

fn ui_sources() -> Vec<(String, String)> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/ui");
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "rip"))
        .collect();
    files.sort();
    assert_eq!(files.len(), 8, "tests/ui holds eight .rip files");
    files
        .into_iter()
        .map(|p| {
            (
                p.file_name().unwrap().to_string_lossy().into_owned(),
                std::fs::read_to_string(&p).unwrap(),
            )
        })
        .collect()
}

/// Identifiers and literals in order. Layout may add, drop or move punctuation and delimiters
/// (trailing commas, closure and match-arm braces); it must not touch these.
fn words(source: &str) -> Vec<String> {
    use proc_macro2::{TokenStream, TokenTree};
    fn go(ts: TokenStream, out: &mut Vec<String>) {
        for t in ts {
            match t {
                TokenTree::Group(g) => go(g.stream(), out),
                TokenTree::Punct(_) => {}
                other => out.push(other.to_string()),
            }
        }
    }
    let mut out = Vec::new();
    go(lex(source).unwrap(), &mut out);
    out
}

fn check_formats(name: &str, source: &str) {
    let once = format_source(source).unwrap_or_else(|e| panic!("{name}: {e}"));
    let twice = format_source(&once).unwrap_or_else(|e| panic!("{name} (second pass): {e}"));
    assert_eq!(once, twice, "{name}: formatting is not idempotent");
    // `format_source` already compares the full token streams modulo layout-only tokens.
    assert_eq!(
        words(source),
        words(&once),
        "{name}: identifiers or literals changed"
    );
}

#[test]
fn masking_round_trips() {
    let src = "\
//! crate doc
/// A tile.
pub tile Regs { a: u8, b: [u8; 2] }
tile Plain { x: u8 }
/// Chain doc.
#[inline]
pub chain f(x: u32) -> u32 { let tile = x; tile }
pub(crate) chain g() {}
region A, B;
region C;
pub region D;
impl Ext { chain h(&self) {} }
fn body() { let chain = 1; let tile = chain; x.chain(tile); }
";
    let masked = mask(src).unwrap();
    assert!(masked.contains("#[rip_tile] pub struct Regs {"), "{masked}");
    assert!(masked.contains("#[rip_tile] struct Plain {"), "{masked}");
    assert!(masked.contains("#[rip_chain] pub fn f("), "{masked}");
    assert!(masked.contains("#[rip_chain] pub(crate) fn g("), "{masked}");
    assert!(masked.contains("__rip_region!(A, B);"), "{masked}");
    assert!(masked.contains("__rip_pub_region!(D);"), "{masked}");
    assert!(masked.contains("impl Ext { #[rip_chain] fn h("), "{masked}");
    assert!(
        masked.contains("let chain = 1; let tile = chain; x.chain(tile);"),
        "bodies are untouched: {masked}"
    );
    assert_eq!(unmask(&masked).unwrap(), src);
}

#[test]
fn unmask_tolerates_attribute_on_its_own_line() {
    let rustfmt_shaped = "\
/// doc
#[rip_tile]
pub struct Regs {
    a: u8,
}

impl X {
    #[rip_chain]
    pub(crate) fn f() {}
}

__rip_region!(
    A,
    B,
);
";
    let got = unmask(rustfmt_shaped).unwrap();
    assert_eq!(got, "/// doc\npub tile Regs {\n    a: u8,\n}\n\nimpl X {\n    pub(crate) chain f() {}\n}\n\nregion A, B;\n");
}

#[test]
fn qsa_kernels_format() {
    if !rustfmt_present() {
        println!("note: rustfmt not on PATH, skipping qsa_kernels_format");
        return;
    }
    for (name, source) in KERNELS {
        check_formats(name, source);
    }
}

#[test]
fn ui_fixtures_format() {
    if !rustfmt_present() {
        println!("note: rustfmt not on PATH, skipping ui_fixtures_format");
        return;
    }
    for (name, source) in ui_sources() {
        check_formats(&name, &source);
    }
}

#[test]
fn messy_snippet_gets_golden_formatting() {
    if !rustfmt_present() {
        println!("note: rustfmt not on PATH, skipping messy_snippet_gets_golden_formatting");
        return;
    }
    let messy = "region   A ,B ;\n/// Regs\npub   tile Regs{a:u8,b:[u8;2]}\nchain  add(x:u32,y:u32)->u32{let z=x+y;\nz}\nfn emit(spec:Spec)->Result<Emitted,String>{let tile=1;Ok(tile)}\n";
    let golden = "\
region A, B;
/// Regs
pub tile Regs {
    a: u8,
    b: [u8; 2],
}
chain add(x: u32, y: u32) -> u32 {
    let z = x + y;
    z
}
fn emit(spec: Spec) -> Result<Emitted, String> {
    let tile = 1;
    Ok(tile)
}
";
    assert_eq!(format_source(messy).unwrap(), golden);
}

#[test]
fn syntax_errors_are_reported() {
    if !rustfmt_present() {
        println!("note: rustfmt not on PATH, skipping syntax_errors_are_reported");
        return;
    }
    assert!(format_source("fn emit( {").is_err());
}
