// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! GPU-free tests of the first-load corpus run, against fake `from-cache`
//! scripts (no analyzer, no device).

use std::os::unix::fs::PermissionsExt;
use std::sync::Mutex;

use super::*;

/// Spawning right after writing an executable can race another test's fork
/// (`ETXTBSY`): the tests that spawn run one at a time.
static SPAWN: Mutex<()> = Mutex::new(());

struct Rig {
    root: PathBuf,
    spec: ProvisionSpec,
}

impl Rig {
    fn new(tag: &str) -> Self {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/railgun-provision-tests")
            .join(format!("{}-{tag}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let hot = root.join("cache/gfx1201");
        fs::create_dir_all(&hot).expect("hot dir");
        fs::write(hot.join("mod_a.hsaco"), b"obj").expect("object");
        let spec = ProvisionSpec {
            arch: "gfx1201".into(),
            program: "raon-ar-decode-gfx1201".into(),
            toolchain_id: "AMD clang version test".into(),
            hot_dir: hot,
            pack_dir: None,
            out_dir: root.join("cache/railgun-corpus/gfx1201"),
        };
        Self { root, spec }
    }

    fn calls(&self) -> usize {
        fs::read_to_string(self.root.join("calls"))
            .map(|s| s.lines().count())
            .unwrap_or(0)
    }

    fn args(&self) -> String {
        fs::read_to_string(self.root.join("calls")).unwrap_or_default()
    }

    /// An executable `from-cache` fake. `body` runs after the argument line is
    /// appended to `calls`; `$OUT` is the `--out` value.
    fn script(&self, name: &str, body: &str) -> PathBuf {
        let path = self.root.join(name);
        let text = format!(
            "#!/bin/sh\necho \"$@\" >> '{calls}'\n\
             OUT=\"\"\nwhile [ $# -gt 0 ]; do [ \"$1\" = --out ] && OUT=\"$2\"; shift; done\n{body}\n",
            calls = self.root.join("calls").display()
        );
        fs::write(&path, text).expect("script");
        fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).expect("chmod");
        path
    }

    fn run(&self, binary: Option<&Path>) -> Result<Provisioned, String> {
        let _g = SPAWN.lock().unwrap_or_else(|e| e.into_inner());
        provision(&self.spec, binary, Duration::from_secs(20))
    }
}

impl Drop for Rig {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

const PUBLISH: &str = "mkdir -p \"$OUT.tmp\" && echo '{}' > \"$OUT.tmp/index.json\" \
    && rm -rf \"$OUT\" && mv \"$OUT.tmp\" \"$OUT\"\n\
    echo '{\"status\":\"ok\",\"nodes\":3,\"objects\":2}'";

#[test]
fn a_successful_run_publishes_the_corpus_and_clears_any_refusal() {
    let rig = Rig::new("ok");
    fs::write(
        rig.spec.hot_dir.join(MARKER),
        r#"{"fingerprint":"stale","reason":"old"}"#,
    )
    .unwrap();
    let bin = rig.script("fc-ok", PUBLISH);
    let done = rig.run(Some(&bin)).expect("provisioned");
    assert_eq!(done.out_dir, rig.spec.out_dir);
    assert!(done.out_dir.join("index.json").is_file());
    assert!(done.report.contains("\"nodes\":3"));
    assert!(!rig.spec.hot_dir.join(MARKER).exists());
    let args = rig.args();
    let hot = rig.spec.hot_dir.display().to_string();
    let out = rig.spec.out_dir.display().to_string();
    assert_eq!(
        args.trim(),
        format!(
            "from-cache --arch gfx1201 --program raon-ar-decode-gfx1201 --cache {hot} --out {out}"
        )
    );
}

#[test]
fn the_pack_is_passed_only_when_one_is_installed() {
    let mut rig = Rig::new("pack");
    rig.spec.pack_dir = Some(rig.root.join("kernels/compiled/gfx1201"));
    let bin = rig.script("fc-pack", PUBLISH);
    rig.run(Some(&bin)).expect("provisioned");
    assert!(rig.args().contains(&format!(
        "--pack {}",
        rig.root.join("kernels/compiled/gfx1201").display()
    )));
}

#[test]
fn a_missing_binary_is_refused_with_the_exact_reason_and_recorded() {
    let rig = Rig::new("absent");
    let e = rig.run(None).expect_err("no binary");
    assert!(e.contains(BINARY) && e.contains("not found"), "{e}");
    assert!(rig.spec.hot_dir.join(MARKER).is_file());
    // The same cache is not retried: the recorded reason comes back.
    let again = rig.run(None).expect_err("still refused");
    assert!(again.contains("recorded for this cache"), "{again}");
    assert!(again.contains("not found"), "{again}");
}

#[test]
fn a_refusal_is_quoted_persisted_and_not_re_run_for_the_same_cache() {
    let rig = Rig::new("refused");
    let bin = rig.script(
        "fc-refuse",
        "echo '{\"status\":\"refused\",\"reason\":\"node 7 gemv: no object in the cache\"}'\nexit 2",
    );
    let e = rig.run(Some(&bin)).expect_err("refused");
    assert!(e.contains("exit 2"), "{e}");
    assert!(e.contains("node 7 gemv: no object in the cache"), "{e}");
    assert_eq!(rig.calls(), 1);
    assert!(!rig.spec.out_dir.exists(), "nothing published on refusal");
    // A fresh process with the same cache and binary: no spawn.
    let again = rig.run(Some(&bin)).expect_err("recorded");
    assert!(again.contains("node 7 gemv"), "{again}");
    assert_eq!(rig.calls(), 1, "the recorded refusal must not spawn again");
}

#[test]
fn a_changed_cache_or_binary_retries() {
    let rig = Rig::new("retry");
    let bin = rig.script(
        "fc-refuse2",
        "echo '{\"status\":\"refused\",\"reason\":\"r\"}'\nexit 2",
    );
    rig.run(Some(&bin)).expect_err("refused");
    assert_eq!(rig.calls(), 1);
    // A new object in the hot cache changes the fingerprint.
    fs::write(rig.spec.hot_dir.join("mod_b.hsaco"), b"more").unwrap();
    rig.run(Some(&bin)).expect_err("refused again");
    assert_eq!(rig.calls(), 2);
    // So does a different binary.
    let other = rig.script("fc-ok2", PUBLISH);
    rig.run(Some(&other)).expect("now ok");
    assert_eq!(rig.calls(), 3);
}

#[test]
fn the_log_and_marker_do_not_change_the_fingerprint() {
    let rig = Rig::new("fp");
    let before = fingerprint(&rig.spec, None);
    fs::write(rig.spec.hot_dir.join(LOG), b"x").unwrap();
    fs::write(rig.spec.hot_dir.join(MARKER), b"{}").unwrap();
    assert_eq!(fingerprint(&rig.spec, None), before);
    let mut other = rig.spec.clone();
    other.toolchain_id.push('x');
    assert_ne!(fingerprint(&other, None), before);
}

#[test]
fn an_unwritable_output_parent_is_refused_without_spawning() {
    let mut rig = Rig::new("ro");
    // The output's parent is a regular file: creating it fails whoever runs.
    let blocker = rig.root.join("blocker");
    fs::write(&blocker, b"file").unwrap();
    rig.spec.out_dir = blocker.join("railgun-corpus/gfx1201");
    let bin = rig.script("fc-unused", PUBLISH);
    let e = rig.run(Some(&bin)).expect_err("cannot create the parent");
    assert!(e.contains("cannot create"), "{e}");
    assert_eq!(rig.calls(), 0);
}

#[test]
fn a_zero_exit_without_a_published_index_is_refused() {
    let rig = Rig::new("noindex");
    let bin = rig.script("fc-liar", "echo '{\"status\":\"ok\"}'");
    let e = rig.run(Some(&bin)).expect_err("no index.json");
    assert!(e.contains("holds no index.json"), "{e}");

    let rig = Rig::new("noreport");
    let bin = rig.script("fc-silent", "exit 0");
    let e = rig.run(Some(&bin)).expect_err("no report");
    assert!(e.contains("without an ok report"), "{e}");
}

#[test]
fn a_failure_without_json_quotes_the_log_tail() {
    let rig = Rig::new("crash");
    let bin = rig.script("fc-crash", "echo 'panicked at the lifter' >&2\nexit 101");
    let e = rig.run(Some(&bin)).expect_err("crash");
    assert!(
        e.contains("exit 101") && e.contains("panicked at the lifter"),
        "{e}"
    );
}

#[test]
fn a_hung_run_is_killed_at_the_timeout() {
    let rig = Rig::new("hang");
    let bin = rig.script("fc-hang", "sleep 30");
    let started = Instant::now();
    let e = {
        let _g = SPAWN.lock().unwrap_or_else(|e| e.into_inner());
        provision(&rig.spec, Some(&bin), Duration::from_millis(150)).expect_err("timeout")
    };
    assert!(e.contains("was killed"), "{e}");
    assert!(started.elapsed() < Duration::from_secs(10));
}

#[test]
fn the_binary_is_found_beside_the_executable_then_the_home_then_path() {
    let rig = Rig::new("locate");
    let make = |d: &str, mode: u32| {
        let dir = rig.root.join(d);
        fs::create_dir_all(&dir).unwrap();
        let f = dir.join(BINARY);
        fs::write(&f, b"#!/bin/sh\n").unwrap();
        fs::set_permissions(&f, fs::Permissions::from_mode(mode)).unwrap();
        dir
    };
    let exe = make("exe", 0o755);
    let home = make("home/bin", 0o755);
    let path_dir = make("pathdir", 0o755);
    let path = std::env::join_paths([&path_dir]).unwrap();
    let all = BinaryLookup {
        exe_dir: Some(&exe),
        home_bin: Some(&home),
        path: Some(&path),
    };
    assert_eq!(locate_binary(&all, BINARY), Some(exe.join(BINARY)));
    let no_exe = BinaryLookup {
        exe_dir: None,
        ..all.clone()
    };
    assert_eq!(locate_binary(&no_exe, BINARY), Some(home.join(BINARY)));
    let only_path = BinaryLookup {
        exe_dir: None,
        home_bin: None,
        path: Some(&path),
    };
    assert_eq!(
        locate_binary(&only_path, BINARY),
        Some(path_dir.join(BINARY))
    );
    // A file without the execute bit does not count.
    let plain = make("plain", 0o644);
    let skipped = BinaryLookup {
        exe_dir: Some(&plain),
        home_bin: Some(&home),
        path: None,
    };
    assert_eq!(locate_binary(&skipped, BINARY), Some(home.join(BINARY)));
    assert_eq!(locate_binary(&BinaryLookup::default(), BINARY), None);
}
