// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! `npu-tools m4 OUTDIR` — Phase-2 GPU+NPU concurrency run, inside ONE npu-window.sh
//! hold (fclk pinned).
//!
//! GPU side reproduces `~/pm-wave/halo-moe-bytes/run.sh plain` (same daemon, model, env, config)
//! but drives the daemon over a pipe so one model load serves every phase:
//!   load; 1 warm pp8192; 3 pp8192 GPU-alone; NPU loop started (2nd process) + 3 pp8192
//!   concurrent; NPU loop alone; unload.
//! NPU side: `$NPU_M4_CMD` (concurrent loop, must cover the 3 concurrent prefills, <= 60 s) and
//! `$NPU_M4_ALONE_CMD` (the same loop, GPU idle). The `soc-metrics` subcommand of this executable
//! samples DRAM / NPU bandwidth every 20 ms into `OUTDIR/metrics.txt`; `OUTDIR/marks.txt` has
//! realtime_ns phase marks for slicing it.
//!
//! Exit status (the `Ok` value): 0 success, 2 an NPU loop failed, 3 refused for low memory.
//! `Err` corresponds to an uncaught Python exception (exit 1, after the same cleanup).

use serde_json::Value;
use std::env;
use std::fs::{self, File};
use std::io::Write;
use std::os::unix::process::ExitStatusExt;
use std::path::{Component, Path, PathBuf};
use std::process::{Child, ChildStdin, Command, ExitStatus, Stdio};
use std::thread::sleep;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const MODEL: &str = "/home/kaden/fold-flash-next/models/qwen3.8-flash-next.mq4";
const POLL: Duration = Duration::from_millis(50);
const CLEANUP_POLL: Duration = Duration::from_millis(20);

fn expanduser(rest: &str) -> Result<PathBuf, String> {
    let home = env::var_os("HOME").ok_or("HOME is not set")?;
    Ok(PathBuf::from(home).join(rest))
}

/// `os.path.abspath`: lexical, no symlink resolution.
fn abspath(p: &str) -> Result<PathBuf, String> {
    let full = if Path::new(p).is_absolute() {
        PathBuf::from(p)
    } else {
        env::current_dir()
            .map_err(|e| format!("current_dir: {e}"))?
            .join(p)
    };
    let mut out = PathBuf::from("/");
    for c in full.components() {
        match c {
            Component::Normal(n) => out.push(n),
            Component::ParentDir => {
                out.pop();
            }
            _ => {}
        }
    }
    Ok(out)
}

/// Python `str()` of a decoded JSON scalar.
fn py_str(v: &Value) -> String {
    match v {
        Value::Null => "None".to_string(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Python `Popen.returncode`: exit code, or negative signal number.
fn returncode(s: &ExitStatus) -> i32 {
    s.code().or_else(|| s.signal().map(|n| -n)).unwrap_or(-1)
}

fn io<T>(r: std::io::Result<T>, what: &str) -> Result<T, String> {
    r.map_err(|e| format!("{what}: {e}"))
}

/// All transitive children of `pid`, found by scanning /proc.
fn descendants(pid: u32) -> Vec<u32> {
    let mut parent_of: Vec<(u32, u32)> = Vec::new();
    if let Ok(rd) = fs::read_dir("/proc") {
        for ent in rd.flatten() {
            let Some(p) = ent.file_name().to_str().and_then(|n| n.parse::<u32>().ok()) else {
                continue;
            };
            let Ok(stat) = fs::read_to_string(ent.path().join("stat")) else {
                continue;
            };
            // "pid (comm) S ppid ..." — comm may contain spaces/parens, so split after the last ')'.
            let Some(tail) = stat.rfind(')').map(|i| &stat[i + 1..]) else {
                continue;
            };
            if let Some(pp) = tail
                .split_whitespace()
                .nth(1)
                .and_then(|s| s.parse::<u32>().ok())
            {
                parent_of.push((p, pp));
            }
        }
    }
    let mut found = vec![pid];
    let mut i = 0;
    while i < found.len() {
        let cur = found[i];
        for &(p, pp) in &parent_of {
            if pp == cur && !found.contains(&p) {
                found.push(p);
            }
        }
        i += 1;
    }
    found.remove(0);
    found
}

fn send_signal(sig: &str, pids: &[u32]) -> bool {
    if pids.is_empty() {
        return true;
    }
    Command::new("kill")
        .arg("-s")
        .arg(sig)
        .arg("--")
        .args(pids.iter().map(u32::to_string))
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// SIGKILL `child` and everything below it, then reap `child`.
fn kill_tree(child: &mut Child) {
    let kids = descendants(child.id());
    let _ = child.kill();
    send_signal("KILL", &kids);
    let _ = child.wait();
}

/// `Popen.wait(timeout)`: `Ok(None)` on timeout.
fn wait_timeout(child: &mut Child, secs: u64) -> Result<Option<ExitStatus>, String> {
    let end = Instant::now() + Duration::from_secs(secs);
    loop {
        if let Some(st) = io(child.try_wait(), "wait")? {
            return Ok(Some(st));
        }
        if Instant::now() >= end {
            return Ok(None);
        }
        sleep(CLEANUP_POLL);
    }
}

struct M4 {
    out: PathBuf,
    repo: PathBuf,
    marks: File,
    resp_path: PathBuf,
    daemon: Child,
    daemon_stdin: Option<ChildStdin>,
    sampler: Child,
    seen: usize,
    npu_loop: Option<Child>,
    npu_alone: Option<Child>,
    alone: Vec<Value>,
    conc: Vec<Value>,
}

fn write_mark(marks: &mut File, what: &str) -> Result<(), String> {
    let ns = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| format!("clock: {e}"))?
        .as_nanos();
    let line = format!("{ns} {what}\n");
    io(marks.write_all(line.as_bytes()), "marks.txt")?;
    let mut so = std::io::stdout().lock();
    io(so.write_all(line.as_bytes()), "stdout")?;
    io(so.flush(), "stdout")
}

/// Spawn `cmd` via the shell in `repo`, stdout+stderr to `out/log`.
fn spawn_npu(
    marks: &mut File,
    out: &Path,
    repo: &Path,
    cmd: &str,
    log: &str,
) -> Result<Child, String> {
    write_mark(marks, &format!("npu-start {log}"))?;
    let f = io(File::create(out.join(log)), log)?;
    let f2 = io(f.try_clone(), log)?;
    io(
        Command::new("/bin/sh")
            .arg("-c")
            .arg(cmd)
            .current_dir(repo)
            .stdout(f)
            .stderr(f2)
            .spawn(),
        "spawn npu command",
    )
}

impl M4 {
    fn mark(&mut self, what: &str) -> Result<(), String> {
        write_mark(&mut self.marks, what)
    }

    /// Send one request and wait for the next response line of type `want`.
    fn request(&mut self, line: &str, want: &str, timeout: u64) -> Result<Value, String> {
        {
            let stdin = self.daemon_stdin.as_mut().ok_or("daemon stdin closed")?;
            io(stdin.write_all(line.as_bytes()), "daemon stdin")?;
            io(stdin.write_all(b"\n"), "daemon stdin")?;
            io(stdin.flush(), "daemon stdin")?;
        }
        let end = Instant::now() + Duration::from_secs(timeout);
        while Instant::now() < end {
            // Poll before reading so a final line written just before exit is not missed.
            let exited = io(self.daemon.try_wait(), "daemon poll")?;
            let bytes = io(fs::read(&self.resp_path), "resp.jsonl")?;
            let text = String::from_utf8_lossy(&bytes);
            let lines: Vec<&str> = text
                .split_inclusive('\n')
                .filter(|l| exited.is_some() || l.ends_with('\n'))
                .map(|l| l.trim_end_matches('\n'))
                .collect();
            for l in lines.iter().skip(self.seen) {
                self.seen += 1;
                let Ok(msg) = serde_json::from_str::<Value>(l) else {
                    continue;
                };
                let Some(obj) = msg.as_object() else {
                    return Err(format!("response is not a JSON object: {l}"));
                };
                match obj.get("type").and_then(Value::as_str) {
                    Some(t) if t == want => return Ok(msg),
                    Some("error") => return Err((*l).to_string()),
                    _ => {}
                }
            }
            if let Some(st) = exited {
                return Err(format!("daemon exited rc={}", returncode(&st)));
            }
            sleep(POLL);
        }
        Err(format!("no {want} within {timeout}s"))
    }

    fn prefill(&mut self, tag: &str) -> Result<Value, String> {
        self.mark(&format!("prefill-start {tag}"))?;
        let r = self.request(
            r#"{"type": "bench_prefill", "tokens": 8192}"#,
            "prefill_result",
            60,
        )?;
        let ms = r.get("ms").ok_or("KeyError: 'ms'")?.clone();
        let tok_s = r.get("tok_s").ok_or("KeyError: 'tok_s'")?.clone();
        self.mark(&format!(
            "prefill-end {tag} ms={} tok_s={}",
            py_str(&ms),
            py_str(&tok_s)
        ))?;
        if ms.as_f64().is_none() {
            return Err(format!("prefill ms is not a number: {}", py_str(&ms)));
        }
        Ok(ms)
    }

    fn prefills(&mut self, prefix: &str) -> Result<Vec<Value>, String> {
        (0..3)
            .map(|i| self.prefill(&format!("{prefix}-{i}")))
            .collect()
    }

    fn phases(&mut self, npu_cmd: &str, npu_alone_cmd: &str) -> Result<i32, String> {
        self.mark("load-start")?;
        let load = format!(
            r#"{{"type": "load", "model": {}, "params": {{"max_seq": 16384}}}}"#,
            Value::String(MODEL.to_string())
        );
        self.request(&load, "loaded", 170)?;
        self.mark("load-end")?;
        self.prefill("warm")?;
        self.alone = self.prefills("gpu-alone")?;
        let child = spawn_npu(
            &mut self.marks,
            &self.out,
            &self.repo,
            npu_cmd,
            "npu-concurrent.log",
        )?;
        self.npu_loop = Some(child);
        // npu-gemm builds the design and CPU reference first; start the GPU only once the first
        // submit has completed and been verified, so the loop overlaps all concurrent prefills.
        let end = Instant::now() + Duration::from_secs(60);
        let log = self.out.join("npu-concurrent.log");
        loop {
            let text = io(fs::read(&log), "npu-concurrent.log")?;
            if String::from_utf8_lossy(&text).contains("submit 0:") {
                break;
            }
            let lp = self.npu_loop.as_mut().ok_or("npu loop missing")?;
            if io(lp.try_wait(), "npu loop poll")?.is_some() || Instant::now() > end {
                return Err("NPU loop did not reach its first submit".to_string());
            }
            sleep(POLL);
        }
        self.mark("npu-first-submit-verified")?;
        self.conc = self.prefills("concurrent")?;
        self.mark("concurrent-prefills-done")?;
        let lp = self.npu_loop.as_mut().ok_or("npu loop missing")?;
        let Some(st) = wait_timeout(lp, 70)? else {
            return Err("npu loop timed out after 70 seconds".to_string());
        };
        let loop_rc = returncode(&st);
        self.mark(&format!("npu-end concurrent rc={loop_rc}"))?;
        let child = spawn_npu(
            &mut self.marks,
            &self.out,
            &self.repo,
            npu_alone_cmd,
            "npu-alone.log",
        )?;
        self.npu_alone = Some(child);
        let al = self.npu_alone.as_mut().ok_or("npu alone missing")?;
        let Some(st) = wait_timeout(al, 70)? else {
            return Err("npu alone timed out after 70 seconds".to_string());
        };
        let alone_rc = returncode(&st);
        self.mark(&format!("npu-end alone rc={alone_rc}"))?;
        self.request(r#"{"type": "unload"}"#, "unloaded", 60)?;
        self.mark("unloaded")?;
        Ok(if loop_rc == 0 && alone_rc == 0 { 0 } else { 2 })
    }

    /// The `finally:` block, plus killing NPU children the original leaked on failure.
    fn cleanup(&mut self, rc: i32) -> Result<(), String> {
        for slot in [&mut self.npu_loop, &mut self.npu_alone] {
            if let Some(c) = slot {
                if matches!(c.try_wait(), Ok(None)) {
                    kill_tree(c);
                }
            }
        }
        drop(self.daemon_stdin.take());
        match wait_timeout(&mut self.daemon, 30) {
            Ok(Some(_)) => {}
            _ => kill_tree(&mut self.daemon),
        }
        terminate(&mut self.sampler);
        if !self.alone.is_empty() && !self.conc.is_empty() {
            let mean =
                |v: &[Value]| v.iter().filter_map(Value::as_f64).sum::<f64>() / v.len() as f64;
            let (a, c) = (mean(&self.alone), mean(&self.conc));
            if a == 0.0 {
                self.mark(&format!("end rc={rc}")).ok();
                return Err("ZeroDivisionError: float division by zero".to_string());
            }
            let list =
                |v: &[Value]| format!("[{}]", v.iter().map(py_str).collect::<Vec<_>>().join(", "));
            let line = format!(
                "summary gpu_alone_ms={} concurrent_ms={} mean_alone={a:.1} mean_concurrent={c:.1} slowdown={:.2}%",
                list(&self.alone),
                list(&self.conc),
                (c / a - 1.0) * 100.0
            );
            self.mark(&line)?;
        }
        self.mark(&format!("end rc={rc}"))
    }
}

/// `Popen.terminate()` (SIGTERM), then reap with a bounded wait.
fn terminate(child: &mut Child) {
    if !send_signal("TERM", &[child.id()]) {
        let _ = child.kill();
    }
    if !matches!(wait_timeout(child, 5), Ok(Some(_))) {
        kill_tree(child);
    }
}

pub fn run(args: &[String]) -> Result<i32, String> {
    let d = expanduser("pm-wave/halo-moe-bytes")?;
    let b = expanduser("pm-wave/hyper-halo/bin")?;
    let repo = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .ok_or("CARGO_MANIFEST_DIR has no parent")?
        .to_path_buf();

    let arg = args.first().ok_or("usage: npu-m4 OUTDIR")?;
    let out = abspath(arg)?;
    io(
        fs::create_dir_all(out.join("home/.hipfire")),
        "create OUTDIR",
    )?;
    let npu_cmd = env::var("NPU_M4_CMD").map_err(|_| "KeyError: 'NPU_M4_CMD'".to_string())?;
    let npu_alone_cmd = env::var("NPU_M4_ALONE_CMD").unwrap_or_else(|_| npu_cmd.clone());
    let mut marks = io(File::create(out.join("marks.txt")), "marks.txt")?;

    let meminfo = io(fs::read_to_string("/proc/meminfo"), "/proc/meminfo")?;
    let avail_kib: u64 = meminfo
        .lines()
        .filter_map(|l| l.split_once(':'))
        .find(|(k, _)| *k == "MemAvailable")
        .and_then(|(_, v)| v.split_whitespace().next())
        .and_then(|v| v.parse().ok())
        .ok_or("KeyError: 'MemAvailable'")?;
    write_mark(&mut marks, &format!("meminfo MemAvailable={avail_kib} kB"))?;
    if avail_kib < 16 * 1024 * 1024 {
        // hipx rule (Main): MemAvailable >= 16 GiB, logged
        write_mark(&mut marks, "REFUSE: MemAvailable < 16 GiB")?;
        return Ok(3);
    }
    // GPU prefill + NPU loop run concurrently: fabric clock pinned for the whole run (`railgun::npu::fclk`).
    let _fclk = railgun::npu::fclk::FabricClockGuard::acquire("npu-tools m4")?;

    let daemon_bin = b.join("daemon");
    {
        let mut f = io(File::create(out.join("identity.txt")), "identity.txt")?;
        let md5 = io(Command::new("md5sum").arg(&daemon_bin).output(), "md5sum")?;
        io(f.write_all(&md5.stdout), "identity.txt")?;
        if let Ok(sha) = fs::read(b.join("SHA")) {
            io(f.write_all(&sha), "identity.txt")?;
        }
        io(
            f.write_all(format!("npu_cmd={npu_cmd}\nnpu_alone_cmd={npu_alone_cmd}\n").as_bytes()),
            "identity.txt",
        )?;
    }
    io(
        fs::write(
            out.join("home/.hipfire/config.toml"),
            "[reasoning]\neffort = \"none\"\n[memory]\nmax_seq = 16384\n",
        ),
        "config.toml",
    )?;

    let exe = io(env::current_exe(), "current_exe")?;
    let mut sampler = io(
        Command::new(exe)
            .arg("soc-metrics")
            .arg(out.join("metrics.txt"))
            .arg("0.02")
            .spawn(),
        "spawn soc-metrics sampler",
    )?;

    let started = (|| -> Result<(Child, ChildStdin), String> {
        let mut env: Vec<(std::ffi::OsString, std::ffi::OsString)> = env::vars_os()
            .filter(|(k, _)| {
                let k = k.to_string_lossy();
                !(k.starts_with("HIPFIRE_") || k.starts_with("HIP_VISIBLE"))
            })
            .collect();
        let extra = [
            ("HOME", out.join("home").into_os_string()),
            ("HIPFIRE_DEVICES", railgun::npu::fclk::igpu_pci().into()),
            ("ROCR_VISIBLE_DEVICES", "1".into()),
            ("HIPFIRE_LOCK_DIR", d.join("lockdir").into_os_string()),
            ("HIPFIRE_GRAPH", "1".into()),
            ("HIPFIRE_DPM_WARMUP_SECS", "10".into()),
            (
                "HIPFIRE_KERNEL_CACHE",
                expanduser("pm-wave/hyper-halo/kcache")?.into_os_string(),
            ),
        ];
        for (k, v) in extra {
            env.retain(|(ek, _)| ek != k);
            env.push((k.into(), v));
        }

        let mut sudo = io(
            Command::new("sudo")
                .args(["-n", "tee", "/proc/sys/vm/drop_caches"])
                .stdin(Stdio::piped())
                .stdout(Stdio::null())
                .spawn(),
            "spawn sudo",
        )?;
        if let Some(mut si) = sudo.stdin.take() {
            let _ = si.write_all(b"2\n"); // like communicate(): a broken pipe is ignored
        }
        io(sudo.wait(), "sudo")?;

        let resp = io(File::create(out.join("resp.jsonl")), "resp.jsonl")?;
        let log = io(File::create(out.join("daemon.log")), "daemon.log")?;
        let mut daemon = io(
            Command::new(&daemon_bin)
                .stdin(Stdio::piped())
                .env_clear()
                .envs(env)
                .stdout(resp)
                .stderr(log)
                .spawn(),
            "spawn daemon",
        )?;
        match daemon.stdin.take() {
            Some(si) => Ok((daemon, si)),
            None => {
                kill_tree(&mut daemon);
                Err("daemon stdin unavailable".to_string())
            }
        }
    })();
    let (daemon, daemon_stdin) = match started {
        Ok(v) => v,
        Err(e) => {
            terminate(&mut sampler);
            return Err(e);
        }
    };

    let mut m = M4 {
        resp_path: out.join("resp.jsonl"),
        out,
        repo,
        marks,
        daemon,
        daemon_stdin: Some(daemon_stdin),
        sampler,
        seen: 0,
        npu_loop: None,
        npu_alone: None,
        alone: Vec::new(),
        conc: Vec::new(),
    };
    let result = m.phases(&npu_cmd, &npu_alone_cmd);
    let cleaned = m.cleanup(*result.as_ref().unwrap_or(&1));
    match (result, cleaned) {
        (Err(e), _) => Err(e),
        (Ok(_), Err(e)) => Err(e),
        (Ok(rc), Ok(())) => Ok(rc),
    }
}
