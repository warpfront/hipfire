//! Card half: arms that drive the real `daemon` binary over its JSON-lines
//! protocol, on whatever card the caller's device selection gives it.
//!
//! The daemon runs with `HIPFIRE_EMIT_TOKEN_IDS=1` (one `committed` event per
//! committed token id) and `HIPFIRE_MODULE_LOAD_AUDIT=1` (the receipt). A
//! request's committed ids become one row stream of the arm's state, so the
//! runner's first difference names the request and the first differing token
//! position.
//!
//! Two row kinds:
//! * `c1_cn` (I5i at a fixed lane width): the same requests one at a time
//!   (reference) and as one burst (candidate) against daemons loaded with the
//!   same `load_params`. The burst must really overlap, or the arm errors.
//! * `restore` (I2): conversation turn 1, then turn 2 whose history holds
//!   turn 1's reply, reached per [`RestoreMode`] (default: `cold` reference,
//!   `live` candidate, the #666 "restored == fresh"; `evicted` forces the
//!   session snapshot round trip).
//!
//! Planted faults change one arm's wire input, so each row shows it can fail.

use crate::digest::{HostPart, HostRows, HostState};
use crate::verdict::{
    run_row, Arm, ArmRun, ClassTable, ComparisonClass, InputSchedule, Recipe, RecipePair, Refusal,
    RowReport, RowSpec,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{channel, Receiver, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

fn default_timeout() -> u64 {
    600
}

/// One daemon configuration: binary, model, load params, extra environment.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DaemonSpec {
    pub daemon: PathBuf,
    pub model: PathBuf,
    #[serde(default)]
    pub load_params: Value,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    /// Longest wait for any single daemon event, seconds.
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
}

/// Everything one request produced on the wire.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Outcome {
    pub id: String,
    pub token_ids: Vec<u32>,
    pub text: String,
    /// `logprob` / `top_logprobs` of each `token` event, when requested.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub logprobs: Vec<Value>,
    pub commit_ready: Option<Value>,
    pub done: Option<Value>,
}

impl Outcome {
    fn field(&self, k: &str) -> Option<&Value> {
        self.commit_ready
            .as_ref()
            .and_then(|c| c.get(k))
            .or_else(|| self.done.as_ref().and_then(|d| d.get(k)))
    }
    pub fn cached_tokens(&self) -> u64 {
        self.field("cached_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0)
    }
    pub fn finish_reason(&self) -> String {
        self.field("finish_reason")
            .and_then(Value::as_str)
            .unwrap_or("none")
            .to_string()
    }
    pub fn route(&self) -> String {
        let loop_ = self.field("loop").and_then(Value::as_str).unwrap_or("?");
        let mode = self
            .field("execution_mode")
            .and_then(Value::as_str)
            .unwrap_or("-");
        format!("loop={loop_} mode={mode}")
    }
}

struct Daemon {
    child: Child,
    stdin: Option<ChildStdin>,
    rx: Receiver<Value>,
    stderr: Arc<Mutex<String>>,
    timeout: Duration,
    attempts: u64,
}

impl Daemon {
    fn spawn(spec: &DaemonSpec) -> Result<Daemon, String> {
        let mut child = Command::new(&spec.daemon)
            .envs(&spec.env)
            .env("HIPFIRE_EMIT_TOKEN_IDS", "1")
            .env("HIPFIRE_MODULE_LOAD_AUDIT", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("spawn {}: {e}", spec.daemon.display()))?;
        let stdout = child.stdout.take().ok_or("daemon stdout")?;
        let mut err = child.stderr.take().ok_or("daemon stderr")?;
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                // Non-JSON stdout is a protocol defect; keep it visible.
                let v = serde_json::from_str(&line).unwrap_or(Value::String(line));
                if tx.send(v).is_err() {
                    break;
                }
            }
        });
        let stderr = Arc::new(Mutex::new(String::new()));
        let sink = stderr.clone();
        std::thread::spawn(move || {
            let mut buf = [0u8; 65536];
            while let Ok(n) = err.read(&mut buf) {
                if n == 0 {
                    break;
                }
                sink.lock()
                    .unwrap()
                    .push_str(&String::from_utf8_lossy(&buf[..n]));
            }
        });
        Ok(Daemon {
            stdin: child.stdin.take(),
            child,
            rx,
            stderr,
            timeout: Duration::from_secs(spec.timeout_secs),
            attempts: 0,
        })
    }

    fn send(&mut self, v: &Value) -> Result<(), String> {
        let stdin = self.stdin.as_mut().ok_or("daemon stdin closed")?;
        writeln!(stdin, "{v}")
            .and_then(|_| stdin.flush())
            .map_err(|e| format!("daemon stdin: {e}{}", self.tail()))
    }

    fn tail(&self) -> String {
        let s = self.stderr.lock().unwrap();
        let lines: Vec<&str> = s
            .lines()
            .filter(|l| !l.starts_with("HIPFIRE_MODULE_LOAD_AUDIT"))
            .collect();
        let from = lines.len().saturating_sub(12);
        format!("; stderr tail:\n{}", lines[from..].join("\n"))
    }

    fn next(&mut self) -> Result<Value, String> {
        match self.rx.recv_timeout(self.timeout) {
            Ok(v) => Ok(v),
            Err(RecvTimeoutError::Timeout) => Err(format!(
                "no daemon event in {} s{}",
                self.timeout.as_secs(),
                self.tail()
            )),
            Err(RecvTimeoutError::Disconnected) => Err(format!("daemon exited{}", self.tail())),
        }
    }

    fn load(&mut self, spec: &DaemonSpec) -> Result<Value, String> {
        let mut params = spec.load_params.clone();
        if params.is_null() {
            params = json!({});
        }
        self.send(&json!({"type": "load", "model": spec.model, "params": params}))?;
        loop {
            let v = self.next()?;
            match v.get("type").and_then(Value::as_str) {
                Some("loaded") => return Ok(v),
                Some("error") => return Err(format!("load refused: {v}{}", self.tail())),
                _ => {}
            }
        }
    }

    /// Send `reqs` (all at once when `burst`, else each after the previous
    /// `done`) and collect every outcome. Returns the outcomes in request order
    /// and the peak number of requests live at once (`gen_start` .. `done`).
    fn run(&mut self, reqs: &[Value], burst: bool) -> Result<(Vec<Outcome>, usize), String> {
        let mut out: BTreeMap<String, Outcome> = BTreeMap::new();
        let ids: Vec<String> = reqs
            .iter()
            .map(|r| {
                r.get("id")
                    .and_then(Value::as_str)
                    .map(str::to_string)
                    .ok_or_else(|| format!("request without id: {r}"))
            })
            .collect::<Result<_, _>>()?;
        for id in &ids {
            out.insert(
                id.clone(),
                Outcome {
                    id: id.clone(),
                    ..Outcome::default()
                },
            );
        }
        // The daemon requires an `attempt_id` on every generate; stamp one per
        // send unless the row set its own.
        let reqs: Vec<Value> = reqs
            .iter()
            .map(|r| {
                let mut r = r.clone();
                if r.get("attempt_id").is_none() {
                    self.attempts += 1;
                    r["attempt_id"] = json!(self.attempts);
                }
                r
            })
            .collect();
        let (mut live, mut peak, mut sent, mut done) = (0usize, 0usize, 0usize, 0usize);
        if burst {
            for r in &reqs {
                self.send(r)?;
            }
            sent = reqs.len();
        }
        while done < reqs.len() {
            if !burst && sent == done {
                self.send(&reqs[sent])?;
                sent += 1;
            }
            let v = self.next()?;
            let ty = v.get("type").and_then(Value::as_str).unwrap_or("");
            let id = v.get("id").and_then(Value::as_str).unwrap_or("");
            if ty == "error" && (id.is_empty() || !out.contains_key(id)) {
                return Err(format!("daemon error: {v}{}", self.tail()));
            }
            let Some(o) = out.get_mut(id) else { continue };
            match ty {
                "gen_start" => {
                    live += 1;
                    peak = peak.max(live);
                }
                "committed" => {
                    let tok = v.get("tok_id").and_then(Value::as_u64).ok_or("tok_id")?;
                    o.token_ids.push(tok as u32);
                }
                "token" => {
                    o.text
                        .push_str(v.get("text").and_then(Value::as_str).unwrap_or(""));
                    if v.get("logprob").is_some() {
                        o.logprobs.push(json!({
                            "logprob": v.get("logprob"),
                            "top": v.get("top_logprobs"),
                        }));
                    }
                }
                "commit_ready" => {
                    // The terminal handshake: without a `commit` the daemon
                    // ends the request `aborted client_cancelled`.
                    let ack = json!({
                        "type": "commit",
                        "id": id,
                        "attempt_id": v.get("attempt_id").cloned().unwrap_or(Value::Null),
                    });
                    o.commit_ready = Some(v.clone());
                    self.send(&ack)?;
                }
                "done" => {
                    o.done = Some(v);
                    live = live.saturating_sub(1);
                    done += 1;
                }
                "error" => return Err(format!("request {id} failed: {v}{}", self.tail())),
                _ => {}
            }
        }
        Ok((ids.iter().map(|i| out.remove(i).unwrap()).collect(), peak))
    }

    /// Unload, close stdin, reap; returns the whole stderr text.
    fn finish(mut self) -> String {
        let _ = self.send(&json!({"type": "unload"}));
        self.stdin = None;
        let t0 = Instant::now();
        loop {
            match self.child.try_wait() {
                Ok(Some(_)) => break,
                Ok(None) if t0.elapsed() < Duration::from_secs(60) => {
                    std::thread::sleep(Duration::from_millis(100))
                }
                _ => {
                    // Our own child only.
                    let _ = self.child.kill();
                    let _ = self.child.wait();
                    break;
                }
            }
        }
        // Let the stderr reader drain.
        std::thread::sleep(Duration::from_millis(200));
        let s = self.stderr.lock().unwrap().clone();
        s
    }
}

impl Drop for Daemon {
    /// An arm that errors before `finish` must not leave its daemon (and the
    /// card's memory) behind. Our own child only.
    fn drop(&mut self) {
        self.stdin = None;
        if matches!(self.child.try_wait(), Ok(None)) {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

/// A planted fault, applied to the candidate arm's wire input.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Plant {
    #[default]
    None,
    /// `c1_cn`: the burst sends request `index` with `seed + 1`.
    WrongSeed { index: usize },
    /// `c1_cn`: the burst trades the prompts of requests `a` and `b`
    /// (a lane reading its peer's input).
    SwappedLaneInput { a: usize, b: usize },
    /// `restore`: turn 2's new user message differs by one word in the
    /// candidate. The restored turn-1 prefix still hits (live reuse needs the
    /// committed history byte-for-byte), so only the state after it differs.
    StaleRestore,
}

fn state_of(outcomes: &[Outcome]) -> Vec<HostState> {
    // Step i holds request i: its committed ids as rows (4 bytes each), its
    // finish reason as a fixed part.
    outcomes
        .iter()
        .map(|o| HostState {
            meta: Vec::new(),
            fixed: vec![HostPart {
                name: format!("{}.finish", o.id),
                bytes: o.finish_reason().into_bytes(),
            }],
            rows: vec![HostRows {
                name: format!("{}.token_ids", o.id),
                owner: format!("request {} ({})", o.id, o.route()),
                row_bytes: 4,
                bytes: o.token_ids.iter().flat_map(|t| t.to_le_bytes()).collect(),
            }],
        })
        .collect()
}

/// What an arm observed beyond its state: per-request outcomes and checks.
#[derive(Clone, Debug, Default, Serialize)]
pub struct ArmLog {
    pub recipe: String,
    pub outcomes: Vec<Outcome>,
    pub peak_live: usize,
    pub loaded: Vec<Value>,
    /// The daemon's `hipfire load summary` / `hipfire load phases` lines.
    pub load_lines: Vec<String>,
}

fn load_lines(stderr: &str) -> Vec<String> {
    stderr
        .lines()
        .filter(|l| l.starts_with("hipfire load summary") || l.starts_with("hipfire load phases"))
        .map(str::to_string)
        .collect()
}

/// `c1_cn` arm. `burst == false` is C1, `true` is CN.
pub struct C1CnArm {
    pub spec: DaemonSpec,
    pub requests: Vec<Value>,
    pub burst: bool,
    pub plant: Plant,
    pub log: ArmLog,
}

impl C1CnArm {
    fn wire(&self) -> Vec<Value> {
        let mut reqs = self.requests.clone();
        match self.plant {
            Plant::WrongSeed { index } => {
                let s = reqs[index].get("seed").and_then(Value::as_u64).unwrap_or(0);
                reqs[index]["seed"] = json!(s + 1);
            }
            Plant::SwappedLaneInput { a, b } => {
                let pa = reqs[a]["prompt"].take();
                let pb = std::mem::replace(&mut reqs[b]["prompt"], pa);
                reqs[a]["prompt"] = pb;
            }
            _ => {}
        }
        reqs
    }
}

impl Arm for C1CnArm {
    fn run(&mut self, recipe: &Recipe, _s: &InputSchedule) -> Result<ArmRun, String> {
        let mut d = Daemon::spawn(&self.spec)?;
        let loaded = d.load(&self.spec)?;
        let res = d.run(&self.wire(), self.burst);
        let audit_log = d.finish();
        let (outcomes, peak) = res?;
        self.log = ArmLog {
            recipe: recipe.name.clone(),
            outcomes: outcomes.clone(),
            peak_live: peak,
            loaded: vec![loaded],
            load_lines: load_lines(&audit_log),
        };
        if self.burst && peak < self.requests.len() {
            return Err(format!(
                "burst serialized: peak {peak} live of {} requests",
                self.requests.len()
            ));
        }
        if !self.burst && peak > 1 {
            return Err(format!("C1 arm overlapped: peak {peak}"));
        }
        require_ids(&outcomes)?;
        Ok(ArmRun {
            audit_log,
            steps: state_of(&outcomes),
        })
    }
}

/// How a `restore` arm reaches turn 2.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RestoreMode {
    /// Turn 2 on a new daemon: whole history prefilled cold (`cached_tokens == 0`).
    /// Its prefill partition differs from the other modes (an I1 question).
    Cold,
    /// Turn 2 right after turn 1 on the same daemon: the live state continues.
    Live,
    /// Turn 1, then the `evict` request, then turn 2 on the same daemon: turn 2
    /// must come back from the session snapshot (`cached_tokens > 0`).
    Evicted,
}

/// `restore` arm.
pub struct RestoreArm {
    pub spec: DaemonSpec,
    /// Turn-1 request (`messages` with one user turn).
    pub turn1: Value,
    /// The second user message.
    pub user2: String,
    /// Unrelated request between the turns ([`RestoreMode::Evicted`]); not compared.
    pub evict: Option<Value>,
    pub mode: RestoreMode,
    pub plant: Plant,
    pub log: ArmLog,
}

impl RestoreArm {
    fn turn2(&self, reply: &str) -> Result<Value, String> {
        let mut t2 = self.turn1.clone();
        let msgs = t2
            .get_mut("messages")
            .and_then(Value::as_array_mut)
            .ok_or("turn1 needs `messages`")?;
        let mut user2 = self.user2.clone();
        if self.plant == Plant::StaleRestore {
            user2.push_str(" Briefly.");
        }
        msgs.push(json!({"role": "assistant", "content": reply}));
        msgs.push(json!({"role": "user", "content": user2}));
        let id = t2["id"].as_str().unwrap_or("t").to_string();
        t2["id"] = json!(format!("{id}.turn2"));
        Ok(t2)
    }
}

impl Arm for RestoreArm {
    fn run(&mut self, recipe: &Recipe, _s: &InputSchedule) -> Result<ArmRun, String> {
        let mut d = Daemon::spawn(&self.spec)?;
        let mut loaded = vec![d.load(&self.spec)?];
        let mut a = d.run(std::slice::from_ref(&self.turn1), false)?.0;
        let t2 = self.turn2(&a[0].text)?;
        let (b, audit) = match self.mode {
            RestoreMode::Live | RestoreMode::Evicted => {
                if self.mode == RestoreMode::Evicted {
                    let evict = self.evict.as_ref().ok_or("evicted mode needs `evict`")?;
                    d.run(std::slice::from_ref(evict), false)?;
                }
                let r = d.run(std::slice::from_ref(&t2), false);
                let audit = d.finish();
                (r?.0, audit)
            }
            RestoreMode::Cold => {
                let mut audit = d.finish();
                let mut f = Daemon::spawn(&self.spec)?;
                loaded.push(f.load(&self.spec)?);
                let r = f.run(std::slice::from_ref(&t2), false);
                audit.push_str(&f.finish());
                (r?.0, audit)
            }
        };
        let cached = b[0].cached_tokens();
        a.extend(b);
        self.log = ArmLog {
            recipe: recipe.name.clone(),
            outcomes: a.clone(),
            peak_live: 1,
            loaded,
            load_lines: load_lines(&audit),
        };
        match self.mode {
            RestoreMode::Cold if cached != 0 => {
                return Err(format!(
                    "cold arm: turn 2 hit a cache on a new daemon ({cached} tokens)"
                ))
            }
            RestoreMode::Live | RestoreMode::Evicted if cached == 0 => {
                return Err(format!(
                    "{:?} arm: turn 2 reused no cached tokens; nothing was restored",
                    self.mode
                ))
            }
            _ => {}
        }
        require_ids(&a)?;
        Ok(ArmRun {
            audit_log: audit,
            steps: state_of(&a),
        })
    }
}

/// A request that produced text but no `committed` ids ran on a route that
/// does not emit them; comparing empty id streams would pass anything.
fn require_ids(outcomes: &[Outcome]) -> Result<(), String> {
    match outcomes
        .iter()
        .find(|o| o.token_ids.is_empty() && !o.text.is_empty())
    {
        Some(o) => Err(format!(
            "request {} ({}) streamed text but no committed token ids: this route does not emit them under HIPFIRE_EMIT_TOKEN_IDS",
            o.id,
            o.route()
        )),
        None => Ok(()),
    }
}

/// One card row of a rows file.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CardRow {
    pub name: String,
    /// Class-table id: `I5i` for `c1_cn`, `I2` for `restore`.
    pub invariant: String,
    #[serde(flatten)]
    pub kind: CardKind,
    #[serde(default)]
    pub plant: Plant,
    /// A planted row passes only if the runner reports `fail`.
    #[serde(default)]
    pub expect_fail: bool,
    pub daemon: DaemonSpec,
    #[serde(default)]
    pub required_kernels: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CardKind {
    C1Cn {
        requests: Vec<Value>,
    },
    Restore {
        turn1: Value,
        user2: String,
        #[serde(default)]
        evict: Option<Value>,
        #[serde(default = "default_reference_mode")]
        reference: RestoreMode,
        #[serde(default = "default_candidate_mode")]
        candidate: RestoreMode,
    },
}

fn default_reference_mode() -> RestoreMode {
    RestoreMode::Cold
}

fn default_candidate_mode() -> RestoreMode {
    RestoreMode::Live
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CardFile {
    pub fixture: String,
    pub row: Vec<CardRow>,
}

/// Replace every `${NAME}` in a rows file with the environment value, so one
/// file serves every host (`${CONFORMANCE_DAEMON}`, `${QWEN4_MODEL}`, ...).
/// An unset name is an error, never an empty path.
pub fn expand_env(text: &str) -> Result<String, String> {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(i) = rest.find("${") {
        out.push_str(&rest[..i]);
        let tail = &rest[i + 2..];
        let end = tail.find('}').ok_or("unterminated ${ in rows file")?;
        let name = &tail[..end];
        let v = std::env::var(name).map_err(|_| format!("rows file needs ${{{name}}}: unset"))?;
        out.push_str(&v);
        rest = &tail[end + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

/// Result of one card row: the runner's report plus both arms' logs, and
/// whether the row behaved as required (`pass`, or `fail` for a planted row).
#[derive(Debug, Serialize)]
pub struct CardResult {
    pub name: String,
    pub plant: Plant,
    pub expect_fail: bool,
    pub ok: bool,
    pub report: Option<RowReport>,
    pub refusal: Option<String>,
    pub reference: ArmLog,
    pub candidate: ArmLog,
}

/// Records a successful run so a later row with the same reference can reuse it.
struct Recorded<'a, T: Arm> {
    inner: &'a mut T,
    out: Option<ArmRun>,
}

impl<T: Arm> Arm for Recorded<'_, T> {
    fn run(&mut self, recipe: &Recipe, s: &InputSchedule) -> Result<ArmRun, String> {
        let r = self.inner.run(recipe, s)?;
        self.out = Some(r.clone());
        Ok(r)
    }
}

struct Replay(ArmRun);

impl Arm for Replay {
    fn run(&mut self, _: &Recipe, _: &InputSchedule) -> Result<ArmRun, String> {
        Ok(self.0.clone())
    }
}

type Memo = BTreeMap<String, (ArmRun, ArmLog)>;

/// Run one row; a reference arm already run by an earlier row with the same
/// `key` (daemon spec and reference input) is reused, not rerun: a planted
/// row's reference is the unplanted row's reference.
fn run_pair<T: Arm>(
    table: &ClassTable,
    spec: &RowSpec,
    memo: &mut Memo,
    key: String,
    mut a: T,
    mut b: T,
    log: fn(&T) -> &ArmLog,
) -> (Result<RowReport, Refusal>, ArmLog, ArmLog) {
    if let Some((run, mut alog)) = memo.get(&key).cloned() {
        alog.recipe.push_str(" (reused from an earlier row)");
        let r = run_row(table, spec, &mut Replay(run), &mut b);
        return (r, alog, log(&b).clone());
    }
    let mut rec = Recorded {
        inner: &mut a,
        out: None,
    };
    let r = run_row(table, spec, &mut rec, &mut b);
    if let Some(out) = rec.out.take() {
        memo.insert(key, (out, log(&a).clone()));
    }
    (r, log(&a).clone(), log(&b).clone())
}

/// Run every row of `file` in order, one daemon at a time.
pub fn run_card_file(table: &ClassTable, file: &CardFile) -> Vec<CardResult> {
    let mut results = Vec::new();
    let mut memo = Memo::new();
    for row in &file.row {
        let recipe = |name: &str| Recipe {
            name: name.into(),
            required_kernels: row.required_kernels.clone(),
        };
        let spec = RowSpec {
            fixture: file.fixture.clone(),
            invariant: row.invariant.clone(),
            recipe_pair: match &row.kind {
                CardKind::C1Cn { .. } => RecipePair {
                    reference: recipe("c1"),
                    candidate: recipe("cn"),
                },
                CardKind::Restore {
                    reference,
                    candidate,
                    ..
                } => RecipePair {
                    reference: recipe(&format!("{reference:?}").to_lowercase()),
                    candidate: recipe(&format!("{candidate:?}").to_lowercase()),
                },
            },
            input_schedule: InputSchedule { seed: 0, steps: 0 },
            comparison_class: ComparisonClass::Exact,
            bounded_difference: None,
            budgets: Default::default(),
        };
        let (res, reference, candidate) = match &row.kind {
            CardKind::C1Cn { requests } => {
                let arm = |burst, plant| C1CnArm {
                    spec: row.daemon.clone(),
                    requests: requests.clone(),
                    burst,
                    plant,
                    log: ArmLog::default(),
                };
                let key = json!({"daemon": row.daemon, "c1": requests}).to_string();
                run_pair(
                    table,
                    &spec,
                    &mut memo,
                    key,
                    arm(false, Plant::None),
                    arm(true, row.plant.clone()),
                    |a| &a.log,
                )
            }
            CardKind::Restore {
                turn1,
                user2,
                evict,
                reference,
                candidate,
            } => {
                let arm = |mode, plant| RestoreArm {
                    spec: row.daemon.clone(),
                    turn1: turn1.clone(),
                    user2: user2.clone(),
                    evict: evict.clone(),
                    mode,
                    plant,
                    log: ArmLog::default(),
                };
                let key = json!({
                    "daemon": row.daemon, "turn1": turn1, "user2": user2,
                    "evict": evict, "mode": reference,
                })
                .to_string();
                run_pair(
                    table,
                    &spec,
                    &mut memo,
                    key,
                    arm(*reference, Plant::None),
                    arm(*candidate, row.plant.clone()),
                    |a| &a.log,
                )
            }
        };
        let (report, refusal) = match res {
            Ok(r) => (Some(r), None),
            Err(e) => (None, Some(e.to_string())),
        };
        let want = if row.expect_fail { "fail" } else { "pass" };
        let ok = report.as_ref().is_some_and(|r| r.verdict.name() == want);
        results.push(CardResult {
            name: row.name.clone(),
            plant: row.plant.clone(),
            expect_fail: row.expect_fail,
            ok,
            report,
            refusal,
            reference,
            candidate,
        });
    }
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    fn outcome(id: &str, ids: &[u32]) -> Outcome {
        Outcome {
            id: id.into(),
            token_ids: ids.to_vec(),
            done: Some(json!({"finish_reason": "length", "loop": "engine"})),
            ..Outcome::default()
        }
    }

    #[test]
    fn first_differing_token_is_reported_with_its_request() {
        use crate::digest::first_difference;
        let a = state_of(&[outcome("r0", &[1, 2, 3]), outcome("r1", &[4, 5, 6])]);
        let b = state_of(&[outcome("r0", &[1, 2, 3]), outcome("r1", &[4, 9, 6])]);
        assert!(first_difference(&a[0].digest(), &b[0].digest()).is_none());
        let d = first_difference(&a[1].digest(), &b[1].digest()).unwrap();
        assert_eq!(d.component, "r1.token_ids");
        assert_eq!(d.row, Some(1));
        assert!(d.owner.starts_with("request r1"));
    }

    #[test]
    fn plants_change_only_the_named_requests() {
        let arm = |plant| C1CnArm {
            spec: DaemonSpec {
                daemon: "daemon".into(),
                model: "m".into(),
                load_params: Value::Null,
                env: BTreeMap::new(),
                timeout_secs: 1,
            },
            requests: vec![
                json!({"id": "a", "prompt": "p0", "seed": 7}),
                json!({"id": "b", "prompt": "p1", "seed": 8}),
            ],
            burst: true,
            plant,
            log: ArmLog::default(),
        };
        let w = arm(Plant::WrongSeed { index: 1 }).wire();
        assert_eq!(
            (w[0]["seed"].as_u64(), w[1]["seed"].as_u64()),
            (Some(7), Some(9))
        );
        let w = arm(Plant::SwappedLaneInput { a: 0, b: 1 }).wire();
        assert_eq!(
            (w[0]["prompt"].as_str(), w[1]["prompt"].as_str()),
            (Some("p1"), Some("p0"))
        );
        assert_eq!(arm(Plant::None).wire(), arm(Plant::None).requests);
    }

    #[test]
    fn rows_file_parses() {
        let f: CardFile = serde_json::from_value(json!({
            "fixture": "qwen3.8-27b.mq4-xts",
            "row": [{
                "name": "x", "invariant": "I5i", "kind": "c1_cn",
                "requests": [{"id": "a", "prompt": "hi"}],
                "plant": {"wrong_seed": {"index": 0}}, "expect_fail": true,
                "daemon": {"daemon": "/d", "model": "/m"}
            }, {
                "name": "y", "invariant": "I2", "kind": "restore",
                "turn1": {"id": "t", "messages": [{"role": "user", "content": "hi"}]},
                "user2": "more", "plant": "stale_restore",
                "daemon": {"daemon": "/d", "model": "/m"}
            }]
        }))
        .unwrap();
        assert_eq!(f.row[0].plant, Plant::WrongSeed { index: 0 });
        assert_eq!(f.row[1].plant, Plant::StaleRestore);
    }
}
