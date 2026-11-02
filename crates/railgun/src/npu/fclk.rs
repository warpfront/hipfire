// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Fabric-clock (fclk) guard for concurrent GPU+NPU work on Strix Halo — the one implementation of pin / verify /
//! restore (`tools/npu/npu-window.sh` calls it through `npu-tools fclk`).
//!
//! The Halo iGPU (PCI `0000:bf:00.0`) shares the SoC data fabric with the NPU. Concurrent GPU+NPU work must run with
//! the fabric clock held at its top level, never while it changes speed. Interface: the standard amdgpu sysfs
//! attributes (kernel docs, amdgpu "GPU Power/Thermal Controls and Monitoring"):
//! * `power_dpm_force_performance_level`: `auto` (driver policy), `manual`, `low`, `high`, `profile_*`;
//! * `pp_dpm_fclk`: one `N: <freq>Mhz` line per level; writing an index (perf level `manual` only) restricts fclk to
//!   it, and the current level carries a `*`. On hipx the levels are 400…2000 MHz and `auto` shows no `*`.
//!
//! Pinned = perf level `manual` AND exactly one `*`, on the last (top) level line. Pin = write `manual`, write the
//! top index, poll the read-back until pinned (≤ 30 s). Writes go to the file directly; when that is not permitted
//! (sysfs is root-only `0644`) through `sudo -n tee FILE`, as `npu-window.sh` always did (hipx grants kaden
//! passwordless sudo). No permission either way → pin fails with that error.
//!
//! [`FabricClockGuard`] (RAII) is taken where concurrent GPU+NPU work starts; mode from `NPU_FCLK_GUARD`:
//! * `require` (default, also when unset): pinned, or refuse to start;
//! * `pin`: already pinned → hold it as found (no restore); else pin and restore the prior perf level on drop —
//!   normal return or panic unwind (not a fatal signal: `npu-window.sh` restores those). A prior `manual` that is not
//!   the pin is refused: its fclk mask cannot be read back, so it could not be restored;
//! * `off`: no check, logged loudly on stderr — experiments only.
use std::fmt;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread::sleep;
use std::time::{Duration, Instant};

/// Strix Halo iGPU PCI bus id: `NPU_IGPU_PCI`, default `0000:bf:00.0` (hipx). Halo boxes enumerate it differently.
pub fn igpu_pci() -> String {
    std::env::var("NPU_IGPU_PCI").unwrap_or_else(|_| "0000:bf:00.0".into())
}
/// Guard mode variable: `require` (default) | `pin` | `off`.
pub const ENV: &str = "NPU_FCLK_GUARD";
pub const PERF: &str = "power_dpm_force_performance_level";
pub const FCLK: &str = "pp_dpm_fclk";
/// Read-back wait after the pin writes (the same 30 s `npu-window.sh` allowed).
pub const PIN_SETTLE: Duration = Duration::from_secs(30);
const POLL: Duration = Duration::from_millis(100);

/// Writes `value` to the sysfs attribute `path`.
pub type WriteFn = fn(&Path, &str) -> Result<(), String>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Require,
    Pin,
    Off,
}

impl Mode {
    pub fn parse(s: &str) -> Result<Mode, String> {
        match s {
            "" | "require" => Ok(Mode::Require),
            "pin" => Ok(Mode::Pin),
            "off" => Ok(Mode::Off),
            o => Err(format!("{ENV}={o:?}: expected require|pin|off")),
        }
    }

    /// `NPU_FCLK_GUARD`, unset → `require`.
    pub fn from_env() -> Result<Mode, String> {
        match std::env::var(ENV) {
            Ok(v) => Mode::parse(&v),
            Err(std::env::VarError::NotPresent) => Ok(Mode::Require),
            Err(e) => Err(format!("{ENV}: {e}")),
        }
    }
}

/// One `pp_dpm_fclk` line.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Level {
    pub index: String,
    pub freq: String,
    pub current: bool,
}

/// Read-back of the two attributes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct State {
    pub perf: String,
    pub levels: Vec<Level>,
}

impl State {
    pub fn parse(perf: &str, fclk: &str) -> Result<State, String> {
        let levels = fclk
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| {
                let (index, rest) = l
                    .split_once(':')
                    .ok_or_else(|| format!("{FCLK}: malformed line {l:?}"))?;
                let rest = rest.trim();
                let current = rest.ends_with('*');
                Ok(Level {
                    index: index.trim().to_string(),
                    freq: rest.trim_end_matches('*').trim().to_string(),
                    current,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        if levels.is_empty() {
            return Err(format!("{FCLK}: no levels"));
        }
        Ok(State {
            perf: perf.trim().to_string(),
            levels,
        })
    }

    /// `manual` AND exactly one `*`, on the top (last) level.
    pub fn pinned(&self) -> bool {
        self.perf == "manual"
            && self.levels.iter().filter(|l| l.current).count() == 1
            && self.top().current
    }

    pub fn top(&self) -> &Level {
        self.levels.last().expect("State::parse guarantees a level")
    }
}

impl fmt::Display for State {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let cur: Vec<String> = self
            .levels
            .iter()
            .filter(|l| l.current)
            .map(|l| format!("{}:{}", l.index, l.freq))
            .collect();
        write!(
            f,
            "perf={} fclk_current=[{}] top={}:{}",
            self.perf,
            cur.join(","),
            self.top().index,
            self.top().freq
        )
    }
}

/// Direct write, else `sudo -n tee` (non-interactive: fails instead of prompting).
pub fn write_attr(path: &Path, value: &str) -> Result<(), String> {
    let direct = match fs::OpenOptions::new().write(true).open(path) {
        Ok(mut f) => match f.write_all(value.as_bytes()) {
            Ok(()) => return Ok(()),
            Err(e) => e,
        },
        Err(e) => e,
    };
    if direct.kind() != io::ErrorKind::PermissionDenied {
        return Err(format!("write {value:?} to {}: {direct}", path.display()));
    }
    let mut child = Command::new("sudo")
        .args(["-n", "tee"])
        .arg(path)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("write {value:?} to {}: {direct}; sudo: {e}", path.display()))?;
    child
        .stdin
        .take()
        .expect("piped stdin")
        .write_all(value.as_bytes())
        .map_err(|e| format!("write {value:?} to {}: sudo tee stdin: {e}", path.display()))?;
    let out = child
        .wait_with_output()
        .map_err(|e| format!("sudo tee {}: {e}", path.display()))?;
    if out.status.success() {
        Ok(())
    } else {
        Err(format!(
            "no permission to write {value:?} to {} (direct: {direct}; sudo -n tee: {} {})",
            path.display(),
            out.status,
            String::from_utf8_lossy(&out.stderr).trim()
        ))
    }
}

/// The two attributes of one amdgpu device.
pub struct Sysfs {
    dir: PathBuf,
    write: WriteFn,
    settle: Duration,
}

impl Sysfs {
    /// The Halo iGPU, real writes, 30 s settle.
    pub fn igpu() -> Sysfs {
        Sysfs::at(
            format!("/sys/bus/pci/devices/{}", igpu_pci()),
            write_attr,
            PIN_SETTLE,
        )
    }

    pub fn at(dir: impl Into<PathBuf>, write: WriteFn, settle: Duration) -> Sysfs {
        Sysfs {
            dir: dir.into(),
            write,
            settle,
        }
    }

    fn read(&self, attr: &str) -> Result<String, String> {
        let p = self.dir.join(attr);
        fs::read_to_string(&p).map_err(|e| format!("read {}: {e}", p.display()))
    }

    pub fn state(&self) -> Result<State, String> {
        State::parse(&self.read(PERF)?, &self.read(FCLK)?)
    }

    /// `manual`, top index, then wait for the pinned read-back. Leaves the writes in place on failure (caller restores).
    pub fn pin(&self) -> Result<State, String> {
        let top = self.state()?.top().index.clone();
        (self.write)(&self.dir.join(PERF), "manual")?;
        (self.write)(&self.dir.join(FCLK), &top)?;
        let t0 = Instant::now();
        loop {
            let s = self.state()?;
            if s.pinned() {
                return Ok(s);
            }
            if t0.elapsed() >= self.settle {
                return Err(format!(
                    "fclk pin not read back after {:.1} s: {s}",
                    t0.elapsed().as_secs_f64()
                ));
            }
            sleep(POLL);
        }
    }

    /// Write a perf level and verify the read-back (`auto` releases the fclk mask).
    pub fn set_perf(&self, level: &str) -> Result<State, String> {
        (self.write)(&self.dir.join(PERF), level)?;
        let s = self.state()?;
        if s.perf == level {
            Ok(s)
        } else {
            Err(format!("perf level {level:?} not read back: {s}"))
        }
    }
}

/// Holds the precondition for concurrent GPU+NPU work for its lifetime; see the module doc.
pub struct FabricClockGuard {
    sys: Sysfs,
    restore: Option<String>,
}

impl FabricClockGuard {
    /// The Halo iGPU, mode from `NPU_FCLK_GUARD`. `what` names the concurrent path in messages.
    pub fn acquire(what: &str) -> Result<FabricClockGuard, String> {
        FabricClockGuard::with(Sysfs::igpu(), Mode::from_env()?, what)
    }

    pub fn with(sys: Sysfs, mode: Mode, what: &str) -> Result<FabricClockGuard, String> {
        let dev = sys.dir.display().to_string();
        if mode == Mode::Off {
            let s = sys.state().map_or_else(|e| e, |s| s.to_string());
            eprintln!("fclk: ################################################################################");
            eprintln!("fclk: WARNING {ENV}=off: {what} runs concurrent GPU+NPU work WITHOUT the fabric-clock check");
            eprintln!("fclk: WARNING {dev}: {s}");
            eprintln!("fclk: WARNING an unpinned fclk can hang the host or corrupt NPU results; experiments only");
            eprintln!("fclk: ################################################################################");
            return Ok(FabricClockGuard { sys, restore: None });
        }
        let s = sys
            .state()
            .map_err(|e| format!("fclk guard ({what}): cannot read the fabric clock state: {e}"))?;
        if s.pinned() {
            eprintln!("fclk: {what}: {dev} pinned ({s}); mode {mode:?}, held as found");
            return Ok(FabricClockGuard { sys, restore: None });
        }
        match mode {
            Mode::Require => Err(format!(
                "fclk guard: refusing concurrent GPU+NPU work ({what}): {dev} fabric clock is not pinned ({s}); required \
                 perf level `manual` with only the top {FCLK} level `{}` current. Run under tools/npu/npu-window.sh, or set \
                 {ENV}=pin to pin it for this process", s.top().index)),
            Mode::Pin if s.perf == "manual" => Err(format!(
                "fclk guard ({what}): {dev} is `manual` but not pinned ({s}); its fclk mask cannot be restored, refusing to change it")),
            Mode::Pin => {
                let prior = s.perf.clone();
                let guard = FabricClockGuard { sys, restore: Some(prior.clone()) };
                // On failure `guard` drops here and restores `prior`.
                let p = guard.sys.pin().map_err(|e| format!("fclk guard ({what}): pin failed: {e}"))?;
                eprintln!("fclk: {what}: pinned {dev} ({p}); restores perf level `{prior}` on drop");
                Ok(guard)
            }
            Mode::Off => unreachable!(),
        }
    }
}

impl Drop for FabricClockGuard {
    fn drop(&mut self) {
        if let Some(prior) = self.restore.take() {
            match self.sys.set_perf(&prior) {
                Ok(s) => eprintln!("fclk: restored {} ({s})", self.sys.dir.display()),
                Err(e) => eprintln!("fclk: RESTORE FAILED for {}: {e}", self.sys.dir.display()),
            }
        }
    }
}

const CLI_USAGE: &str = "usage: npu-tools fclk status|pin|restore
  status   print the iGPU fabric clock state; exit 0 pinned, 1 not
  pin      npu-window.sh contract: refuse (exit 6) unless the perf level is `auto`; pin and verify (exit 5 on
           failure, after writing `auto` back)
  restore  write `auto` and verify (exit 4 on failure)";

/// `npu-tools fclk ...` (the `npu-window.sh` steps). Ok = exit status.
pub fn cli(args: &[String]) -> Result<i32, String> {
    cli_on(&Sysfs::igpu(), args)
}

fn cli_on(sys: &Sysfs, args: &[String]) -> Result<i32, String> {
    match args
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        ["status"] => {
            let s = sys.state()?;
            println!("fclk: {} {s} pinned={}", sys.dir.display(), s.pinned());
            Ok(if s.pinned() { 0 } else { 1 })
        }
        ["pin"] => {
            let s = sys.state()?;
            if s.perf != "auto" {
                println!(
                    "fclk: REFUSE: incoming perf level is {:?}, not auto ({s})",
                    s.perf
                );
                return Ok(6);
            }
            let t0 = Instant::now();
            match sys.pin() {
                Ok(p) => {
                    println!(
                        "fclk: pinned {} after {} ms ({p})",
                        sys.dir.display(),
                        t0.elapsed().as_millis()
                    );
                    Ok(0)
                }
                Err(e) => {
                    println!("fclk: PIN FAILED: {e}");
                    match sys.set_perf("auto") {
                        Ok(r) => println!("fclk: restored ({r})"),
                        Err(r) => println!("fclk: RESTORE FAILED: {r}"),
                    }
                    Ok(5)
                }
            }
        }
        ["restore"] => match sys.set_perf("auto") {
            Ok(s) => {
                println!("fclk: restored {} ({s})", sys.dir.display());
                Ok(0)
            }
            Err(e) => {
                println!("fclk: RESTORE FAILED: {e}");
                Ok(4)
            }
        },
        ["-h"] | ["--help"] => {
            println!("{CLI_USAGE}");
            Ok(0)
        }
        _ => Err(CLI_USAGE.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    const LEVELS: [&str; 8] = [
        "400Mhz", "1000Mhz", "1200Mhz", "1400Mhz", "1600Mhz", "1700Mhz", "1850Mhz", "2000Mhz",
    ];

    /// A fake device directory with hipx's level table; removed on drop.
    struct Fake(PathBuf);
    impl Fake {
        fn new(perf: &str, current: Option<usize>) -> Fake {
            static N: AtomicUsize = AtomicUsize::new(0);
            let dir = std::env::temp_dir().join(format!(
                "fclk-test-{}-{}",
                std::process::id(),
                N.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir_all(&dir).unwrap();
            fs::write(dir.join(PERF), format!("{perf}\n")).unwrap();
            fs::write(dir.join(FCLK), table(current)).unwrap();
            Fake(dir)
        }
        fn sys(&self, write: WriteFn) -> Sysfs {
            Sysfs::at(&self.0, write, Duration::from_millis(300))
        }
        fn perf(&self) -> String {
            fs::read_to_string(self.0.join(PERF))
                .unwrap()
                .trim()
                .to_string()
        }
    }
    impl Drop for Fake {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn table(current: Option<usize>) -> String {
        LEVELS
            .iter()
            .enumerate()
            .map(|(i, f)| format!("{i}: {f} {}\n", if current == Some(i) { "*" } else { "" }))
            .collect()
    }

    /// Emulates the driver: perf writes replace the level (`auto` clears the mask, no `*` shown); an fclk index write
    /// needs `manual` and moves the `*`.
    fn emulate(path: &Path, value: &str) -> Result<(), String> {
        let dir = path.parent().unwrap();
        match path.file_name().unwrap().to_str().unwrap() {
            PERF => {
                fs::write(path, format!("{value}\n")).unwrap();
                if value == "auto" {
                    fs::write(dir.join(FCLK), table(None)).unwrap();
                }
                Ok(())
            }
            FCLK => {
                if fs::read_to_string(dir.join(PERF)).unwrap().trim() != "manual" {
                    return Err("EINVAL: not manual".into());
                }
                let i: usize = value.trim().parse().map_err(|_| "EINVAL".to_string())?;
                fs::write(path, table(Some(i))).unwrap();
                Ok(())
            }
            o => panic!("unexpected attribute {o}"),
        }
    }

    /// A driver that takes `manual` but ignores the fclk index (the read-back never shows the pin).
    fn stuck(path: &Path, value: &str) -> Result<(), String> {
        if path.file_name().unwrap() == FCLK {
            Ok(())
        } else {
            emulate(path, value)
        }
    }

    fn forbidden(_: &Path, _: &str) -> Result<(), String> {
        panic!("require mode must not write")
    }

    #[test]
    fn require_accepts_pinned() {
        let f = Fake::new("manual", Some(7));
        let g = FabricClockGuard::with(f.sys(forbidden), Mode::Require, "t").unwrap();
        drop(g);
        assert_eq!(f.perf(), "manual");
    }

    #[test]
    fn require_refuses_unpinned() {
        let f = Fake::new("auto", None);
        let e = FabricClockGuard::with(f.sys(forbidden), Mode::Require, "t")
            .err()
            .unwrap();
        assert!(e.contains("not pinned"), "{e}");
        // auto that happens to sit on the top level is still not pinned
        let f = Fake::new("auto", Some(7));
        assert!(FabricClockGuard::with(f.sys(forbidden), Mode::Require, "t").is_err());
    }

    #[test]
    fn require_refuses_wrong_level() {
        let f = Fake::new("manual", Some(6));
        assert!(FabricClockGuard::with(f.sys(forbidden), Mode::Require, "t").is_err());
        let two = fs::read_to_string(f.0.join(FCLK))
            .unwrap()
            .replace("2000Mhz ", "2000Mhz *");
        fs::write(f.0.join(FCLK), two).unwrap();
        assert!(
            !f.sys(forbidden).state().unwrap().pinned(),
            "two current levels"
        );
        assert!(FabricClockGuard::with(f.sys(forbidden), Mode::Require, "t").is_err());
    }

    #[test]
    fn require_errors_without_sysfs() {
        let sys = Sysfs::at("/nonexistent/fclk", forbidden, Duration::ZERO);
        assert!(FabricClockGuard::with(sys, Mode::Require, "t")
            .err()
            .unwrap()
            .contains("cannot read"));
    }

    #[test]
    fn pin_restores_on_drop() {
        let f = Fake::new("auto", None);
        let g = FabricClockGuard::with(f.sys(emulate), Mode::Pin, "t").unwrap();
        assert!(f.sys(forbidden).state().unwrap().pinned());
        drop(g);
        let s = f.sys(forbidden).state().unwrap();
        assert_eq!(s.perf, "auto");
        assert!(!s.pinned());
    }

    #[test]
    fn pin_restores_prior_non_auto_level() {
        let f = Fake::new("high", None);
        drop(FabricClockGuard::with(f.sys(emulate), Mode::Pin, "t").unwrap());
        assert_eq!(f.perf(), "high");
    }

    #[test]
    fn pin_restores_on_panic() {
        let f = Fake::new("auto", None);
        let sys = f.sys(emulate);
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _g = FabricClockGuard::with(sys, Mode::Pin, "t").unwrap();
            assert_eq!(fs::read_to_string(f.0.join(PERF)).unwrap().trim(), "manual");
            panic!("concurrent work failed");
        }));
        assert!(r.is_err());
        assert_eq!(f.perf(), "auto");
    }

    #[test]
    fn pin_leaves_existing_pin() {
        let f = Fake::new("manual", Some(7));
        drop(FabricClockGuard::with(f.sys(forbidden), Mode::Pin, "t").unwrap());
        assert!(f.sys(forbidden).state().unwrap().pinned());
    }

    #[test]
    fn pin_refuses_foreign_manual_mask() {
        let f = Fake::new("manual", Some(3));
        assert!(FabricClockGuard::with(f.sys(forbidden), Mode::Pin, "t").is_err());
        assert!(
            f.sys(forbidden).state().unwrap().levels[3].current,
            "foreign mask left untouched"
        );
    }

    #[test]
    fn pin_failure_restores_prior() {
        let f = Fake::new("auto", None);
        let e = FabricClockGuard::with(f.sys(stuck), Mode::Pin, "t")
            .err()
            .unwrap();
        assert!(e.contains("not read back"), "{e}");
        assert_eq!(f.perf(), "auto");
    }

    #[test]
    fn off_checks_nothing() {
        let f = Fake::new("auto", None);
        drop(FabricClockGuard::with(f.sys(forbidden), Mode::Off, "t").unwrap());
        assert_eq!(f.perf(), "auto");
    }

    #[test]
    fn mode_parse() {
        assert_eq!(Mode::parse("").unwrap(), Mode::Require);
        assert_eq!(Mode::parse("pin").unwrap(), Mode::Pin);
        assert_eq!(Mode::parse("off").unwrap(), Mode::Off);
        assert!(Mode::parse("0").is_err());
    }

    #[test]
    fn cli_window_steps() {
        let a = |s: &str| vec![s.to_string()];
        let f = Fake::new("auto", None);
        assert_eq!(cli_on(&f.sys(emulate), &a("status")).unwrap(), 1);
        assert_eq!(cli_on(&f.sys(emulate), &a("pin")).unwrap(), 0);
        assert_eq!(cli_on(&f.sys(emulate), &a("status")).unwrap(), 0);
        assert_eq!(
            cli_on(&f.sys(emulate), &a("pin")).unwrap(),
            6,
            "incoming manual is refused"
        );
        assert_eq!(cli_on(&f.sys(emulate), &a("restore")).unwrap(), 0);
        assert_eq!(f.perf(), "auto");
        assert_eq!(cli_on(&f.sys(stuck), &a("pin")).unwrap(), 5);
        assert_eq!(f.perf(), "auto", "failed pin writes auto back");
    }
}
