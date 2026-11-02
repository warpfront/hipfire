# Decide Calibration (v1.2) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Per-(model, question type) temperature scaling for
`POST /v1/systemone` answers. The temperatures are fitted offline on
held-out labelled rows, configured per model in `models.toml`, forwarded by
serve and applied by the engine. The Jev wire format is unchanged.

**Architecture:**

- `hipfire-engine::decide` (pure):
  - `Calibration { choice, score, noul }` and `parse_calibration`, which
    reads the daemon message's optional `calibration`;
  - `softmax_t`, and `assemble_answer(q, label_logits, t)`;
  - `assemble_answers(questions, per_q_logits, &Calibration)`, which both
    decide modes use.
- `hipfire-generate::decide`: plain and session modes read the calibration
  from the parsed request and echo a non-identity one in `timing`.
- `hipfire-config`: three request-scoped keys,
  `decide.calibration.{choice,score,noul}`.
- serve: resolves them for the model it uses and adds `"calibration"` to the
  daemon message when any T ≠ 1.
- `hipfire-daemon/src/main.rs` is not touched.
- Offline (`scripts/jev_eval/`, stdlib Python):
  - `calibrate.py` freezes the reported rows, fits T and applies it;
  - `heldout.py` builds the held-out set with no overlap, and collects its
    answers;
  - `report.py` shows raw vs calibrated;
  - `calib_live_check.py` is the GPU equivalence check.

**Tech Stack:** Rust (the crates above), serde_json, toml; Python 3 stdlib.

**Spec:** `docs/specs/2026-09-28-jev-decide-design.md` §13 (and §3, §7, §12).
Read §13 first.

## Global Constraints

- **Preserve every existing decide invariant:**
  - §7's answer shapes and confidence formulas;
  - unrounded probabilities;
  - additive `usage`;
  - status codes 422 / 400 / 409 / 503 / 500;
  - the model left reset after a plain decide;
  - session commit rules;
  - every snapshot reaching `free_gpu`.
- **With no calibration configured, every answer is bit-identical to
  v1.1** (spec §13.3 invariant 1).
- **The Jev wire format is unchanged:**
  - request `{model, state, questions}` (or session `messages`);
  - response `{model, answers, usage}`, and no new body fields;
  - a client's `calibration` body field is never forwarded;
  - calibration is visible only in `x-hipfire-timing`.
- **Layering** (`docs/ARCHITECTURE.md`): the math lives in
  `hipfire-engine`, and the runner only threads the value through. **No
  change to `crates/hipfire-daemon/src/main.rs`**: `daemon_lines` sits at its
  ceiling of 4882 in `scripts/leanup-thresholds.txt`.
- **CI ratchets and crate maps pass.**
  - `bash scripts/leanup-ratchets.sh`;
  - `python3 scripts/check-crate-maps.py --check`;
  - `bash scripts/ci-rustfmt-changed.sh`;
  - `./scripts/no-gpu-ci.sh`.

  After any change under `crates/`, regenerate the maps with
  `python3 scripts/check-crate-maps.py <crate>...`.
- **The external clones are read-only:** `~/repos/jev-evals/jev-bench` and
  `~/repos/jev-evals/jev-ood-calibration`.
  - Never write into them.
  - jevbench's `ROOT` is redirected to the work dir.
  - Every script that imports from them sets `sys.dont_write_bytecode = True`
    first.
  - The eval worktree `~/repos/hipfire-jev-eval` is read-only too: read its
    saved answers and run its built binaries, and never edit or rebuild it.
- **Never fit on reported rows** (spec §13.4, §13.5).
  - `calibrate.py fit` reads only a `heldout/` directory.
  - Held-out states are disjoint from R by construction, and
    `heldout.py verify` proves it.
  - Reported rows are only evaluated, after T is fixed.
- **No dataset text is committed.**
  - Held-out examples live in `$JEV_CALIB_WORK` (default
    `~/.cache/hipfire-jev-calib`).
  - The repo gets text-free rows only. Stage files by explicit path, never
    `git add -A`.
- **GPU etiquette (Task 5 only; every other task is CPU):**
  - one daemon at a time;
  - before starting anything, `pgrep -a -x daemon; pgrep -a -x hipfire`
    must print nothing. If they print anything (for example the running
    serve battery), wait and ask; **never kill a daemon or serve you did not
    start**;
  - never launch the installed `~/.hipfire/bin` binaries;
  - kill everything you start, by the PIDs you recorded, and confirm with
    `pgrep` afterwards;
  - check `grep MemAvailable /proc/meminfo` first: ≥ 12 GiB for the 4B,
    ≥ 30 GiB for the 27B;
  - `export PATH=/opt/rocm-7.2.2/bin:$PATH`;
  - Bash calls are capped at 10 minutes, so run anything longer in the
    background (`run_in_background`) and poll its log;
  - no foreground `sleep` loops. `heldout.py run` waits for serve's
    `/health` itself.
- **Never write `~/.hipfire/models.toml`.** Older hipfire binaries refuse
  unknown override keys (spec §13.9). Applying the fitted snippet is the
  human's step.
- **Environment:**
  - worktree `/home/nick/repos/hipfire-jev-calib`, branch
    `feat/jev-decide-calibration` (stacked on `feat/jev-decide`);
  - do not touch other checkouts;
  - never push.
- **Commits:** end every message with a blank line, then
  `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `crates/hipfire-engine/src/decide.rs` | modify | `Calibration`, `parse_calibration`, `softmax_t`, `assemble_answer(…, t)`, `assemble_answers`; `calibration` field on `DecideRequest` / `SessionRequest`. |
| `crates/hipfire-generate/src/decide.rs` | modify | Plain mode assembles with the request's calibration; `note_calibration` timing echo. |
| `crates/hipfire-generate/src/decide/session.rs` | modify | `SessionPlan.calibration`; session and debug paths use it. |
| `crates/hipfire-generate/examples/split_prefill_probe.rs` | modify | `assemble_answer(…, 1.0)` (raw; the gates compare raw). |
| `crates/hipfire-config/src/lib.rs` | modify | Fields `decide.calibration.{choice,score,noul}` + tests. |
| `crates/hipfire-cli/src/serve/decide.rs` | modify | `calibration_message`; forward it in the daemon message. |
| `crates/hipfire-cli/src/main.rs` | modify | Serve route tests (Task 11 harness). |
| `docs/CONFIG.md` | modify | "Decide calibration" section. |
| `crates/{hipfire-engine,hipfire-generate,hipfire-config,hipfire-cli}/map.md` | regenerate | Crate maps. |
| `scripts/jev_eval/calsets.py` | create | Shared jev-ood-calibration record builders (from `run_calibration.py`), plus a `split` / `n` parameter. |
| `scripts/jev_eval/run_calibration.py` | modify | Imports `calsets`; behaviour unchanged. |
| `scripts/jev_eval/calibrate.py` | create | freeze / fit / apply / metrics. |
| `scripts/jev_eval/report.py` | rewrite | Raw vs calibrated tables from the frozen rows. |
| `scripts/jev_eval/heldout.py` | create | build / verify / run for the held-out set. |
| `scripts/jev_eval/calib_live_check.py` | create | GPU check L1. |
| `scripts/jev_eval/test_calibrate.py`, `test_heldout.py` | create | CPU unit tests. |
| `scripts/no-gpu-ci.sh` | modify | Run the jev_eval unit tests. |
| `scripts/jev_eval/README.md`, `bench/jev/README.md` | modify | Calibration workflow and results pointer. |
| `bench/jev/<model>/probs/` | create | Frozen reported rows (text-free) + `SOURCES.json`. |
| `bench/jev/heldout/{manifest.json,reported_state_sha256.txt}` | create | Held-out manifest and R. |
| `bench/jev/<model>/heldout/*.jsonl` | create | Held-out answers (text-free). |
| `bench/jev/<model>/{calibration.json,report.md}` | create / regenerate | Fit and raw-vs-calibrated report. |

Task order: 1 → 2 (Rust), 3 → 4 (Python, CPU; independent of 1–2), 5 (GPU;
needs 1 and 4), 6 (needs 3 and 5).

---

### Task 1: Engine calibration and runner plumbing

**Files:**
- Modify: `crates/hipfire-engine/src/decide.rs`
- Modify: `crates/hipfire-generate/src/decide.rs`
- Modify: `crates/hipfire-generate/src/decide/session.rs`
- Modify: `crates/hipfire-generate/examples/split_prefill_probe.rs`

**Interfaces:**
- Produces (in `hipfire_engine::decide`):
  - `#[derive(Debug, Clone, Copy, PartialEq)] pub struct Calibration { pub choice: f64, pub score: f64, pub noul: f64 }`
    with `Calibration::IDENTITY`, `Default`, `fn temperature(&self, &QuestionKind) -> f64`,
    `fn is_identity(&self) -> bool`, `fn to_json(&self) -> Value`;
  - `pub const MIN_CALIBRATION_T: f64 = 0.05; pub const MAX_CALIBRATION_T: f64 = 20.0;`
  - `pub fn parse_calibration(v: &Value) -> Result<Calibration, String>`. It
    reads `v["calibration"]`, and every `Err` is a 422 message;
  - `pub fn softmax_t(logits: &[f32], t: f64) -> Vec<f64>`. `softmax(l)` is
    now `softmax_t(l, 1.0)`;
  - `pub fn assemble_answer(q: &Question, label_logits: &[f32], t: f64) -> Value`.
    This is a **signature change**: every caller passes a temperature;
  - `pub fn assemble_answers(questions: &[Question], per_question_logits: &[Vec<f32>], calibration: &Calibration) -> Value`;
  - `DecideRequest { state_text, questions, calibration: Calibration }`;
  - `SessionRequest { messages, tools, questions, calibration: Calibration }`.
- Produces (in `hipfire_generate::decide`, private):
  `fn note_calibration(timing: &mut Value, cal: &d::Calibration)`.

- [ ] **Step 1: Update the existing test calls to the new signature, then
  add the failing tests.** In `crates/hipfire-engine/src/decide.rs`, the six
  existing test calls of `assemble_answer(a, b)` become
  `assemble_answer(a, b, 1.0)`:

```bash
cd /home/nick/repos/hipfire-jev-calib
sed -i -E 's/assemble_answer\((&[^,]+), (&\[[^]]*\]|&logits)\)/assemble_answer(\1, \2, 1.0)/' crates/hipfire-engine/src/decide.rs
grep -c 'assemble_answer(.*, 1.0)' crates/hipfire-engine/src/decide.rs   # expect 6
```

Then append these tests inside `mod tests` (after
`assemble_rejects_single_label`):

```rust
    #[test]
    fn calibration_absent_or_null_is_identity() {
        assert_eq!(parse_calibration(&json!({})).unwrap(), Calibration::IDENTITY);
        assert_eq!(
            parse_calibration(&json!({"calibration": null})).unwrap(),
            Calibration::IDENTITY
        );
        assert!(Calibration::default().is_identity());
    }

    #[test]
    fn calibration_parses_partial_objects() {
        let c = parse_calibration(&json!({"calibration": {"choice": 1.3, "noul": 0.7}})).unwrap();
        assert_eq!(
            c,
            Calibration {
                choice: 1.3,
                score: 1.0,
                noul: 0.7
            }
        );
        assert!(!c.is_identity());
        assert_eq!(c.to_json(), json!({"choice": 1.3, "score": 1.0, "noul": 0.7}));
        let b = parse_calibration(&json!({"calibration": {"choice": 0.05, "score": 20}})).unwrap();
        assert_eq!((b.choice, b.score), (0.05, 20.0));
    }

    #[test]
    fn calibration_rejects_bad_shapes() {
        let e = |v: Value| parse_calibration(&json!({ "calibration": v })).unwrap_err();
        assert!(e(json!(1.3)).contains("calibration must be an object"));
        assert!(e(json!({"bogus": 1.0})).contains("choice, score or noul"));
        for bad in [json!(0.0), json!(0.049), json!(20.5), json!(-1.0), json!("1.3"), json!(null)] {
            let m = e(json!({ "choice": bad.clone() }));
            assert!(
                m.contains("calibration.choice must be a number in [0.05, 20]"),
                "{bad}: {m}"
            );
        }
    }

    #[test]
    fn temperature_one_is_bit_identical_to_the_v1_readout() {
        let z = [3.25f32, -1.5, 0.125, 7.0, -40.0];
        // v1's softmax, verbatim.
        let max = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
        let e: Vec<f64> = z.iter().map(|&x| (x as f64 - max).exp()).collect();
        let s: f64 = e.iter().sum();
        let v1: Vec<f64> = e.into_iter().map(|x| x / s).collect();
        let t1 = softmax_t(&z, 1.0);
        assert!(v1.iter().zip(&t1).all(|(a, b)| a.to_bits() == b.to_bits()));
        let q = choice_q(&["a", "b", "c", "d", "e"]);
        let all = assemble_answers(std::slice::from_ref(&q), &[z.to_vec()], &Calibration::IDENTITY);
        assert_eq!(all["q"], assemble_answer(&q, &z, 1.0));
    }

    #[test]
    fn temperature_scales_label_logits_before_the_softmax() {
        let z = [2.0f32, 0.0, -1.0];
        for t in [0.5, 1.7, 4.0] {
            let p = softmax_t(&z, t);
            let e: Vec<f64> = z.iter().map(|&x| (x as f64 / t).exp()).collect();
            let s: f64 = e.iter().sum();
            for (a, b) in p.iter().zip(e.iter().map(|x| x / s)) {
                assert!((a - b).abs() < 1e-12, "t={t}");
            }
        }
        // Log-probabilities give back the logits up to a constant, so
        // softmax(log p / T) == softmax(z / T) (spec §13.3 invariant 4).
        let lp: Vec<f32> = softmax(&z).iter().map(|p| p.ln() as f32).collect();
        for (x, y) in softmax_t(&z, 2.5).iter().zip(&softmax_t(&lp, 2.5)) {
            assert!((x - y).abs() < 1e-6);
        }
    }

    #[test]
    fn calibrated_choice_keeps_the_argmax_and_moves_the_confidence() {
        let q = choice_q(&["a", "b", "c", "d"]);
        let z = [0.5f32, 2.0, 1.0, -1.0];
        let (raw, flat, sharp) = (
            assemble_answer(&q, &z, 1.0),
            assemble_answer(&q, &z, 2.0),
            assemble_answer(&q, &z, 0.5),
        );
        for a in [&raw, &flat, &sharp] {
            assert_eq!(a["choice"], "b");
        }
        let conf = |a: &Value| a["confidence"].as_f64().unwrap();
        assert!(conf(&flat) < conf(&raw) && conf(&raw) < conf(&sharp));
        let p = softmax_t(&z, 2.0);
        assert!((conf(&flat) - (p[1] - 0.25) / 0.75).abs() < 1e-12);
        assert_eq!(flat["probabilities"]["b"].as_f64().unwrap(), p[1]);
    }

    #[test]
    fn calibrated_noul_is_a_sigmoid_of_the_scaled_log_odds() {
        let n = Question {
            name: "n".into(),
            instructions: "i".into(),
            kind: QuestionKind::Noul {
                true_desc: None,
                false_desc: None,
            },
        };
        // log-odds ln 3 at T = 2: p = sqrt(3) / (sqrt(3) + 1).
        let a = assemble_answer(&n, &[(3.0f32).ln(), 0.0], 2.0);
        let want = 3f64.sqrt() / (3f64.sqrt() + 1.0);
        assert!((a["noul"].as_f64().unwrap() - want).abs() < 1e-6);
    }

    #[test]
    fn calibrated_score_mean_follows_the_calibrated_distribution() {
        let q = Question {
            name: "s".into(),
            instructions: "i".into(),
            kind: QuestionKind::Score {
                levels: vec!["a".into(), "b".into(), "c".into()],
            },
        };
        let z = [0.0f32, 1.0, 3.0];
        let a = assemble_answer(&q, &z, 3.0);
        let p = softmax_t(&z, 3.0);
        let score = a["score"].as_f64().unwrap();
        assert!((score - (p[1] + 2.0 * p[2])).abs() < 1e-12);
        assert!((a["confidence"].as_f64().unwrap() - p[2]).abs() < 1e-12);
        // Flatter than raw: the mean moves toward the middle level.
        assert!(score < assemble_answer(&q, &z, 1.0)["score"].as_f64().unwrap());
    }

    #[test]
    fn assemble_answers_uses_each_question_type_temperature() {
        let cal = Calibration {
            choice: 2.0,
            score: 0.5,
            noul: 4.0,
        };
        let s = Question {
            name: "s".into(),
            instructions: "i".into(),
            kind: QuestionKind::Score {
                levels: vec!["x".into(), "y".into()],
            },
        };
        let n = Question {
            name: "n".into(),
            instructions: "i".into(),
            kind: QuestionKind::Noul {
                true_desc: None,
                false_desc: None,
            },
        };
        let qs = vec![choice_q(&["a", "b"]), s, n];
        let z = vec![vec![1.0f32, 0.0]; 3];
        let out = assemble_answers(&qs, &z, &cal);
        let keys: Vec<&String> = out.as_object().unwrap().keys().collect();
        assert_eq!(keys, vec!["q", "s", "n"]);
        assert_eq!(out["q"], assemble_answer(&qs[0], &z[0], 2.0));
        assert_eq!(out["s"], assemble_answer(&qs[1], &z[1], 0.5));
        assert_eq!(out["n"], assemble_answer(&qs[2], &z[2], 4.0));
    }

    #[test]
    fn requests_carry_their_calibration_in_both_modes() {
        let qs = json!({"q": {"type": "noul", "instructions": "i"}});
        let cal = json!({"choice": 1.5});
        let plain = parse_request(&json!({"state": "s", "questions": qs, "calibration": cal})).unwrap();
        assert_eq!(plain.calibration.choice, 1.5);
        assert!(parse_request(&json!({"state": "s", "questions": qs}))
            .unwrap()
            .calibration
            .is_identity());
        let sess = parse_session_request(&json!({
            "messages": [{"role": "user", "content": "u"}], "questions": qs, "calibration": cal
        }))
        .unwrap();
        assert_eq!(sess.calibration.choice, 1.5);
        let e = parse_request(&json!({"state": "s", "questions": qs, "calibration": {"noul": 99}}))
            .unwrap_err();
        assert!(e.contains("calibration.noul"), "{e}");
    }
```

- [ ] **Step 2: Run the tests and watch them fail to compile**

```bash
cargo test -p hipfire-engine --lib decide 2>&1 | grep -E "^error" | head
```

Expected: `cannot find type Calibration`, `parse_calibration`,
`softmax_t` and `assemble_answers`, and the `assemble_answer` arity error.

- [ ] **Step 3: Implement in `hipfire-engine`.** Replace the whole
  `/// Numerically stable softmax in f64.` function with:

```rust
/// Readout temperature per question type (spec §13). `T = 1` is the raw
/// readout; an answer's probabilities are `softmax(label_scores / T)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Calibration {
    pub choice: f64,
    pub score: f64,
    pub noul: f64,
}

/// Accepted temperature range (spec §13.2); the config schema enforces the
/// same bounds for `decide.calibration.*`.
pub const MIN_CALIBRATION_T: f64 = 0.05;
pub const MAX_CALIBRATION_T: f64 = 20.0;

impl Calibration {
    pub const IDENTITY: Calibration = Calibration {
        choice: 1.0,
        score: 1.0,
        noul: 1.0,
    };

    pub fn temperature(&self, kind: &QuestionKind) -> f64 {
        match kind {
            QuestionKind::Choice { .. } => self.choice,
            QuestionKind::Score { .. } => self.score,
            QuestionKind::Noul { .. } => self.noul,
        }
    }

    pub fn is_identity(&self) -> bool {
        *self == Self::IDENTITY
    }

    pub fn to_json(&self) -> Value {
        json!({"choice": self.choice, "score": self.score, "noul": self.noul})
    }
}

impl Default for Calibration {
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// The decide message's optional `calibration` (spec §13.2): absent or
/// `null` is the identity; `choice` / `score` / `noul` are each optional
/// (default 1) and must be numbers in [0.05, 20]. `Err` is a 422 message.
pub fn parse_calibration(v: &Value) -> Result<Calibration, String> {
    let obj = match v.get("calibration") {
        None | Some(Value::Null) => return Ok(Calibration::IDENTITY),
        Some(Value::Object(o)) => o,
        Some(_) => return Err("calibration must be an object {choice, score, noul}".into()),
    };
    let mut c = Calibration::IDENTITY;
    for (key, val) in obj {
        let slot = match key.as_str() {
            "choice" => &mut c.choice,
            "score" => &mut c.score,
            "noul" => &mut c.noul,
            other => {
                return Err(format!(
                    "calibration keys must be choice, score or noul, got {other:?}"
                ))
            }
        };
        *slot = val
            .as_f64()
            .filter(|t| (MIN_CALIBRATION_T..=MAX_CALIBRATION_T).contains(t))
            .ok_or_else(|| {
                format!(
                    "calibration.{key} must be a number in \
                     [{MIN_CALIBRATION_T}, {MAX_CALIBRATION_T}]"
                )
            })?;
    }
    Ok(c)
}

/// Numerically stable softmax of `logits / t` in f64 (spec §13.3). At
/// `t = 1` the division is exact, so the result is v1's bit for bit.
pub fn softmax_t(logits: &[f32], t: f64) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let exps: Vec<f64> = logits
        .iter()
        .map(|&x| ((x as f64 - max) / t).exp())
        .collect();
    let sum: f64 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

/// Numerically stable softmax in f64: the raw readout (T = 1).
pub fn softmax(logits: &[f32]) -> Vec<f64> {
    softmax_t(logits, 1.0)
}
```

Replace `assemble_answer`'s doc comment, signature and first lines:

```rust
/// Jev answer JSON for one question. `label_logits` follows `labels_for`
/// order (logits or log-probabilities: only differences matter); `t` is the
/// question type's calibration temperature (spec §13.3; 1 = raw).
/// Probabilities are not rounded.
pub fn assemble_answer(q: &Question, label_logits: &[f32], t: f64) -> Value {
    debug_assert!(
        label_logits.len() >= 2,
        "assemble_answer: need at least 2 labels"
    );
    debug_assert!(
        t.is_finite() && t > 0.0,
        "assemble_answer: temperature must be positive"
    );
    let p = softmax_t(label_logits, t);
```

The rest of the body (`let k = …` through the `match`) is unchanged. Add
directly after `assemble_answer`:

```rust
/// Every question's answer, in request order, each at its type's
/// temperature (spec §13.3). The one assembly both decide modes use.
pub fn assemble_answers(
    questions: &[Question],
    per_question_logits: &[Vec<f32>],
    calibration: &Calibration,
) -> Value {
    let mut answers = serde_json::Map::new();
    for (q, ll) in questions.iter().zip(per_question_logits) {
        answers.insert(
            q.name.clone(),
            assemble_answer(q, ll, calibration.temperature(&q.kind)),
        );
    }
    Value::Object(answers)
}
```

Add the field to both request types and fill it in their parsers:

```rust
#[derive(Debug, Clone, PartialEq)]
pub struct DecideRequest {
    pub state_text: String,
    pub questions: Vec<Question>,
    /// Readout temperatures from the daemon message (spec §13.2).
    pub calibration: Calibration,
}
```

In `parse_request`, the final `Ok(DecideRequest { … })` becomes:

```rust
    Ok(DecideRequest {
        state_text: render_state(state),
        questions,
        calibration: parse_calibration(v)?,
    })
```

In `SessionRequest`, add `pub calibration: Calibration,` after
`pub questions: Vec<Question>,`. In `parse_session_request`, the final
`Ok(SessionRequest { … })` becomes:

```rust
    Ok(SessionRequest {
        messages,
        tools,
        questions: parse_questions(v)?,
        calibration: parse_calibration(v)?,
    })
```

- [ ] **Step 4: Run the engine tests**

```bash
cargo test -p hipfire-engine --lib decide
```

Expected: all pass, including the 10 new tests.

- [ ] **Step 5: Thread it through the runner.** In
  `crates/hipfire-generate/src/decide.rs`, add after `fn ms(…)`:

```rust
/// Echo a non-identity calibration into the timing (spec §13.2). It is
/// diagnostic only and never reaches the Jev body.
fn note_calibration(timing: &mut Value, cal: &d::Calibration) {
    if !cal.is_identity() {
        timing["calibration"] = cal.to_json();
    }
}
```

In `execute_decide`, replace

```rust
    let mut answers = serde_json::Map::new();
    for (q, ll) in parsed.questions.iter().zip(per_q_logits.iter()) {
        answers.insert(q.name.clone(), d::assemble_answer(q, ll));
    }
    let lens: Vec<usize> = seqs.iter().map(Vec::len).collect();
    timing["total_ms"] = json!(ms(t_total));
    Ok(DecideOutcome {
        answers: Value::Object(answers),
```

with

```rust
    let answers = d::assemble_answers(&parsed.questions, &per_q_logits, &parsed.calibration);
    note_calibration(&mut timing, &parsed.calibration);
    let lens: Vec<usize> = seqs.iter().map(Vec::len).collect();
    timing["total_ms"] = json!(ms(t_total));
    Ok(DecideOutcome {
        answers,
```

Add this test to its `mod tests`:

```rust
    #[test]
    fn timing_echoes_only_a_non_identity_calibration() {
        let mut t = serde_json::json!({"prefix_tokens": 0});
        note_calibration(&mut t, &d::Calibration::IDENTITY);
        assert!(t.get("calibration").is_none());
        let c = d::Calibration {
            choice: 1.3,
            score: 1.0,
            noul: 0.7,
        };
        note_calibration(&mut t, &c);
        assert_eq!(
            t["calibration"],
            serde_json::json!({"choice": 1.3, "score": 1.0, "noul": 0.7})
        );
    }
```

In `crates/hipfire-generate/src/decide/session.rs`:

- add `note_calibration` to the `use super::{…}` list;
- add `calibration: d::Calibration,` to `SessionPlan` after
  `questions: Vec<d::Question>,`;
- in `plan_session`'s `Ok(SessionPlan { … })`, add
  `calibration: parsed.calibration,` after `questions: parsed.questions,`.

In `execute_session`'s `plan.no_snapshot` branch, the destructure and the
v1 plan become:

```rust
        let SessionPlan {
            questions,
            calibration,
            carrier,
            label_ids,
            seqs,
            ..
        } = plan;
        let v1 = DecidePlan {
            parsed: d::DecideRequest {
                state_text: String::new(),
                questions,
                calibration,
            },
```

At the end of `execute_session`, replace

```rust
    let mut answers = serde_json::Map::new();
    for (q, ll) in plan.questions.iter().zip(&per_q) {
        answers.insert(q.name.clone(), d::assemble_answer(q, ll));
    }
    timing["total_ms"] = json!(ms(t_total));
    (
        Ok(DecideOutcome {
            answers: Value::Object(answers),
```

with

```rust
    let answers = d::assemble_answers(&plan.questions, &per_q, &plan.calibration);
    note_calibration(&mut timing, &plan.calibration);
    timing["total_ms"] = json!(ms(t_total));
    (
        Ok(DecideOutcome {
            answers,
```

In `crates/hipfire-generate/examples/split_prefill_probe.rs`, the probe
compares raw readouts:

```bash
sed -i 's/d::assemble_answer(q, &la)/d::assemble_answer(q, \&la, 1.0)/; s/d::assemble_answer(q, &lc)/d::assemble_answer(q, \&lc, 1.0)/' crates/hipfire-generate/examples/split_prefill_probe.rs
grep -n 'assemble_answer' crates/hipfire-generate/examples/split_prefill_probe.rs   # both carry ", 1.0"
```

- [ ] **Step 6: Build and test the runner, including the lab-gated probe**

```bash
cargo test -p hipfire-engine --lib decide
cargo test -p hipfire-generate --lib decide
cargo check -p hipfire-generate --features lab --examples
cargo check --workspace --examples
```

Expected: all pass, and nothing else in the workspace calls
`assemble_answer` (`grep -rn "assemble_answer(" crates` shows only the
engine, runner, session and probe sites).

- [ ] **Step 7: Maps, format, commit**

```bash
rustfmt --edition 2021 crates/hipfire-engine/src/decide.rs crates/hipfire-generate/src/decide.rs \
  crates/hipfire-generate/examples/split_prefill_probe.rs   # decide.rs pulls in decide/session.rs
bash scripts/ci-rustfmt-changed.sh
cargo test -p hipfire-engine --lib decide && cargo test -p hipfire-generate --lib decide
python3 scripts/check-crate-maps.py hipfire-engine hipfire-generate
git add crates/hipfire-engine/src/decide.rs crates/hipfire-generate/src/decide.rs \
  crates/hipfire-generate/src/decide/session.rs crates/hipfire-generate/examples/split_prefill_probe.rs \
  crates/hipfire-engine/map.md crates/hipfire-generate/map.md
git commit -m "feat(decide): per-type temperature calibration in answer assembly

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Config keys and serve forwarding

**Files:**
- Modify: `crates/hipfire-config/src/lib.rs` (3 fields + 2 tests)
- Modify: `crates/hipfire-cli/src/serve/decide.rs` (`calibration_message`,
  forwarding, 1 unit test)
- Modify: `crates/hipfire-cli/src/main.rs` (3 route tests in the Task 11
  harness module)
- Modify: `docs/CONFIG.md`

**Interfaces:**
- Consumes:
  - `crate::config_f64(&ResolvedConfig, &str) -> anyhow::Result<f64>`
    (main.rs);
  - the `ServeRuntime::ensure_model` result (`ResolvedConfig`, per model,
    re-read every request);
  - the Task 11 harness helpers `post_systemone`, `read_requests_log`,
    `ops_of_type`.
- Produces:
  - config keys `decide.calibration.{choice,score,noul}`: `Request` scope,
    `Float` in [0.05, 20], default 1.0, not registry-backed, no env alias;
  - `fn calibration_message(resolved: &hipfire_config::ResolvedConfig) -> anyhow::Result<Option<serde_json::Value>>`
    in serve/decide.rs;
  - daemon message field `"calibration": {"choice","score","noul"}`, only
    when one differs from 1.

- [ ] **Step 1: Write the failing config tests.** Inside
  `crates/hipfire-config/src/lib.rs`'s `mod tests`, after
  `schema_has_unique_keys_and_legacy_keys`:

```rust
    #[test]
    fn decide_calibration_fields_are_per_model_request_temperatures() {
        for kind in ["choice", "score", "noul"] {
            let key = format!("decide.calibration.{kind}");
            let f = field(&key).unwrap_or_else(|| panic!("{key} missing"));
            assert!(matches!(f.scope, ConfigScope::Request), "{key}");
            assert!(f.env_compat.is_none() && !f.registry_allowed, "{key}");
            assert_eq!(f.default.to_value(), ConfigValue::Float(1.0));
            for ok in [ConfigValue::Float(0.05), ConfigValue::Integer(2), ConfigValue::Float(20.0)] {
                assert!(f.validate(&ok).is_ok(), "{key} {ok:?}");
            }
            for bad in [ConfigValue::Float(0.0), ConfigValue::Float(0.04), ConfigValue::Float(20.5)] {
                assert!(f.validate(&bad).is_err(), "{key} {bad:?}");
            }
        }
    }

    #[test]
    fn models_toml_accepts_and_round_trips_decide_calibration() {
        let root = temp_root("decide-calibration");
        fs::create_dir_all(&root).unwrap();
        let paths = ConfigPaths::under(&root);
        fs::write(
            &paths.models_toml,
            r#"schema_version = 1

[models."qwen3.5-4b.mq4"]
path = "/models/qwen3.5-4b.mq4"
overrides = { decide = { calibration = { choice = 1.3, score = 2, noul = 0.7 } } }
"#,
        )
        .unwrap();
        let loaded = load_catalog(&paths).unwrap();
        let (_, model) = loaded.catalog.model("qwen3.5-4b.mq4").unwrap();
        assert_eq!(
            model.overrides.get("decide.calibration.choice"),
            Some(&ConfigValue::Float(1.3))
        );
        assert_eq!(
            model.overrides.get("decide.calibration.score"),
            Some(&ConfigValue::Integer(2))
        );
        write_catalog_toml(&paths, &loaded.catalog).unwrap();
        let written = fs::read_to_string(&paths.models_toml).unwrap();
        assert!(written.contains("decide.calibration"), "{written}");
        assert_eq!(load_catalog_toml(&paths.models_toml).unwrap(), loaded.catalog);
        fs::write(
            &paths.models_toml,
            "schema_version = 1\n[models.m]\npath = \"/m\"\n\
             [models.m.overrides.decide.calibration]\nchoice = 0\n",
        )
        .unwrap();
        assert!(load_catalog(&paths).is_err(), "T = 0 must be refused");
    }
```

Note: `write_catalog_toml` renders nested tables, e.g.
`[models."qwen3.5-4b.mq4".overrides.decide.calibration]`; that table header
contains `decide.calibration`.

```bash
cargo test -p hipfire-config decide_calibration 2>&1 | tail -5
cargo test -p hipfire-config models_toml_accepts_and_round_trips 2>&1 | tail -5
```

Expected: both FAIL (the `{key} missing` panic, then `UnknownKey`).

- [ ] **Step 2: Add the fields.** In `FIELDS`, immediately before
  `    field!(\n        "generation.top_p",`, insert:

```rust
    field!(
        "decide.calibration.choice",
        "decide_calibration_choice",
        Generation,
        Request,
        DefaultValue::Float(1.0),
        ValueRule::Float {
            min: 0.05,
            max: 20.0,
            min_inclusive: true
        },
        false,
        false,
        None,
        "Decide (/v1/systemone) temperature T for choice answers: probabilities are softmax(label logits / T); 1 = raw. Set per model from a fit (docs/specs/2026-09-28-jev-decide-design.md section 13)."
    ),
    field!(
        "decide.calibration.score",
        "decide_calibration_score",
        Generation,
        Request,
        DefaultValue::Float(1.0),
        ValueRule::Float {
            min: 0.05,
            max: 20.0,
            min_inclusive: true
        },
        false,
        false,
        None,
        "Decide (/v1/systemone) temperature T for score answers: probabilities are softmax(label logits / T); 1 = raw. Set per model from a fit (docs/specs/2026-09-28-jev-decide-design.md section 13)."
    ),
    field!(
        "decide.calibration.noul",
        "decide_calibration_noul",
        Generation,
        Request,
        DefaultValue::Float(1.0),
        ValueRule::Float {
            min: 0.05,
            max: 20.0,
            min_inclusive: true
        },
        false,
        false,
        None,
        "Decide (/v1/systemone) temperature T for noul (yes/no) answers: p(yes) = sigmoid(log-odds / T); 1 = raw. Set per model from a fit (docs/specs/2026-09-28-jev-decide-design.md section 13)."
    ),
```

```bash
cargo test -p hipfire-config
```

Expected: all pass, including `schema_has_unique_keys_and_legacy_keys`
(the legacy keys are unique and the defaults validate).

- [ ] **Step 3: Write the failing serve tests.** In
  `crates/hipfire-cli/src/serve/decide.rs`'s `mod tests`:

```rust
    #[test]
    fn calibration_message_only_when_a_temperature_differs_from_one() {
        let none = hipfire_config::resolve(Vec::<hipfire_config::NamedLayer>::new()).unwrap();
        assert_eq!(calibration_message(&none).unwrap(), None);
        let mut layer = hipfire_config::ConfigLayer::default();
        layer
            .set("decide.calibration.choice", hipfire_config::ConfigValue::Float(1.5))
            .unwrap();
        let resolved = hipfire_config::resolve(vec![hipfire_config::NamedLayer {
            source: hipfire_config::ConfigSource::ModelUser {
                model: "m".into(),
                path: "/x/models.toml".into(),
            },
            layer,
        }])
        .unwrap();
        assert_eq!(
            calibration_message(&resolved).unwrap(),
            Some(json!({"choice": 1.5, "score": 1.0, "noul": 1.0}))
        );
    }
```

In `crates/hipfire-cli/src/main.rs`, after
`systemone_does_not_forward_debug_no_snapshot`:

```rust
    /// Spec §13.2: write a per-model `decide.calibration` override for the
    /// harness fixture into the harness's own models.toml.
    #[cfg(unix)]
    fn write_decide_calibration(harness: &Task11HttpHarness, table: &str) {
        let catalog = format!(
            "schema_version = 1\n\n[models.\"fixture\"]\npath = {:?}\n\n\
             [models.\"fixture\".overrides.decide.calibration]\n{table}\n",
            harness.model()
        );
        fs::write(&harness.paths.config.models_toml, catalog).unwrap();
    }

    /// POST a decide (must succeed) and return the message serve forwarded.
    #[cfg(unix)]
    fn forwarded_decide(harness: &Task11HttpHarness, body: &serde_json::Value) -> serde_json::Value {
        let (status, _, text) = post_systemone(harness.port(), body);
        assert_eq!(status, 200, "{text}");
        let log = harness.read_requests_log();
        let decides = Task11HttpHarness::ops_of_type(&log, "decide");
        (*decides.last().expect("a forwarded decide")).clone()
    }

    #[cfg(unix)]
    #[test]
    fn systemone_forwards_the_per_model_calibration() {
        let harness = Task11HttpHarness::spawn("systemone-cal");
        write_decide_calibration(&harness, "choice = 1.5\nnoul = 0.8");
        let body = serde_json::json!({"model": harness.model(), "state": "s",
            "questions": {"q": {"type": "noul", "instructions": "i"}}});
        let d = forwarded_decide(&harness, &body);
        assert_eq!(
            d["calibration"],
            serde_json::json!({"choice": 1.5, "score": 1.0, "noul": 0.8})
        );
        // Session mode forwards it too.
        let d = forwarded_decide(&harness, &session_body(harness.model(), None));
        assert_eq!(d["calibration"]["choice"], 1.5);
    }

    #[cfg(unix)]
    #[test]
    fn systemone_sends_no_calibration_without_an_override() {
        let harness = Task11HttpHarness::spawn("systemone-cal-none");
        let body = serde_json::json!({"model": harness.model(), "state": "s",
            "questions": {"q": {"type": "noul", "instructions": "i"}}});
        let d = forwarded_decide(&harness, &body);
        assert!(d.get("calibration").is_none(), "{d}");
    }

    #[cfg(unix)]
    #[test]
    fn systemone_client_cannot_set_calibration() {
        let harness = Task11HttpHarness::spawn("systemone-cal-client");
        let body = serde_json::json!({"model": harness.model(), "state": "s",
            "calibration": {"choice": 9.0},
            "questions": {"q": {"type": "noul", "instructions": "i"}}});
        let d = forwarded_decide(&harness, &body);
        assert!(d.get("calibration").is_none(), "client calibration forwarded: {d}");
        write_decide_calibration(&harness, "noul = 0.8");
        let d = forwarded_decide(&harness, &body);
        assert_eq!(
            d["calibration"],
            serde_json::json!({"choice": 1.0, "score": 1.0, "noul": 0.8})
        );
    }
```

`session_body(model, tool_choice)` is the existing session-test helper.

```bash
cargo test -p hipfire-cli systemone_ 2>&1 | tail -15
cargo test -p hipfire-cli calibration_message 2>&1 | tail -5
```

Expected: the build fails with `cannot find function
calibration_message`. Once that exists, the three route tests would still
fail on a missing `calibration` field until Step 4 forwards it.

- [ ] **Step 4: Implement the forwarding** in
  `crates/hipfire-cli/src/serve/decide.rs`. Add above `run_decide`:

```rust
/// Per-model decide calibration (spec §13.2): the resolved
/// `decide.calibration.{choice,score,noul}` temperatures, sent to the
/// daemon only when one differs from 1, so an uncalibrated model's decide
/// message is unchanged. Never taken from the request body.
fn calibration_message(
    resolved: &hipfire_config::ResolvedConfig,
) -> anyhow::Result<Option<serde_json::Value>> {
    let mut out = serde_json::Map::new();
    let mut identity = true;
    for kind in ["choice", "score", "noul"] {
        let t = crate::config_f64(resolved, &format!("decide.calibration.{kind}"))?;
        identity &= t == 1.0;
        out.insert(kind.to_string(), serde_json::json!(t));
    }
    Ok((!identity).then_some(serde_json::Value::Object(out)))
}
```

In `run_decide`:

- the block's tuple becomes `let (engine, model_echo, session, calibration) = {`;
- right after the `let session = match body.get("messages") { … };`
  statement, insert:

```rust
            let calibration = match calibration_message(&resolved) {
                Ok(c) => c,
                Err(e) => {
                    return DecideOutcome::Err {
                        status: 500,
                        message: format!("decide calibration config: {e:#}"),
                        required_max_seq: None,
                    }
                }
            };
```

- the block's last expression becomes
  `(runtime.engine.clone(), echo, session, calibration)`;
- after the `if let Some((messages, tools)) = &session { … }` block, insert:

```rust
        if let Some(c) = &calibration {
            msg["calibration"] = c.clone();
        }
```

The `for key in ["state", "questions"]` copy loop stays as it is. It is
what keeps a client's `calibration` out.

```bash
cargo test -p hipfire-cli systemone_
cargo test -p hipfire-cli calibration_message
cargo test -p hipfire-config -p hipfire-cli
```

Expected: all pass (the three new route tests and every existing
`systemone_` test).

- [ ] **Step 5: Document the keys.** In `docs/CONFIG.md`, insert this
  section immediately before `## Per-model overlay`:

~~~markdown
## Decide calibration

Per-model temperatures for `POST /v1/systemone` answers (spec
`docs/specs/2026-09-28-jev-decide-design.md` §13). An answer's
probabilities are `softmax(label logits / T)`, with one `T` per question
type.

| Key | Default | Range | Scope |
|---|---|---|---|
| `decide.calibration.choice` | `1.0` | 0.05–20 | request (per-model overlay) |
| `decide.calibration.score` | `1.0` | 0.05–20 | request (per-model overlay) |
| `decide.calibration.noul` | `1.0` | 0.05–20 | request (per-model overlay) |

`1.0` is the raw readout. T > 1 makes answers less confident and T < 1 more
confident; the chosen option never changes. Set them per model, from a fit
(`scripts/jev_eval/calibrate.py fit`):

```toml
[models."qwen3.5-4b.mq4".overrides.decide.calibration]
choice = 1.3
noul = 0.7
```

or `hipfire config qwen3.5-4b.mq4 set decide.calibration.choice 1.3`. Serve
resolves them on every decide, so a change applies without a reload. They
are not in the Jev response body; a non-identity calibration shows in the
`x-hipfire-timing` header. A hipfire binary built before these keys existed
refuses a `models.toml` that contains them.
~~~

- [ ] **Step 6: Gates, maps, commit**

```bash
rustfmt --edition 2021 crates/hipfire-config/src/lib.rs crates/hipfire-cli/src/main.rs   # main.rs pulls in serve/decide.rs
bash scripts/ci-rustfmt-changed.sh
cargo test -p hipfire-config -p hipfire-cli
python3 scripts/check-crate-maps.py hipfire-config hipfire-cli
bash scripts/leanup-ratchets.sh
python3 scripts/check-env-docs.py
git diff --stat -- crates/hipfire-daemon   # must print nothing
git add crates/hipfire-config/src/lib.rs crates/hipfire-cli/src/serve/decide.rs crates/hipfire-cli/src/main.rs \
  crates/hipfire-config/map.md crates/hipfire-cli/map.md docs/CONFIG.md
git commit -m "feat(decide): per-model decide.calibration config forwarded by serve

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

Expected: the ratchets pass with `daemon_lines` unchanged at 4882.

---

### Task 3: Offline calibration library, report, and frozen reported rows

**Files:**
- Create: `scripts/jev_eval/calsets.py`
- Modify: `scripts/jev_eval/run_calibration.py` (import from `calsets`)
- Create: `scripts/jev_eval/calibrate.py`
- Rewrite: `scripts/jev_eval/report.py`
- Create: `scripts/jev_eval/test_calibrate.py`
- Modify: `scripts/no-gpu-ci.sh`
- Create: `bench/jev/qwen3.5-4b/probs/`, `bench/jev/qwen3.8-27b-mq4-xts/probs/`

**Interfaces:**
- Produces:
  - `calsets.hf_rows(ds, cfg, n, split="validation")`,
    `calsets.public_records(name, split="validation", n=None)`,
    `calsets.load_synth(path)`, `calsets.to_question(r)`,
    `calsets.PUBLIC_N`;
  - `calibrate.apply_t(probs, t)`, `ece(pairs)`, `nll(rows, t)`,
    `fit_temperature(rows)`, `fit_types(rows)`, `summarize(rows, temps)`,
    `read_jsonl`, `write_rows`, `sha256_file`, `toml_snippet`,
    `main(argv)` with the subcommands `freeze` and `fit`;
  - row schema `{"source", "type", "probs", "gold"}`.

- [ ] **Step 1: Write the failing tests** (`scripts/jev_eval/test_calibrate.py`):

```python
"""CPU tests for calibrate.py (spec §13.7).
Run: python3 -m unittest discover -s scripts/jev_eval -p 'test_*.py'"""
import json
import math
import random
import tempfile
import unittest
from pathlib import Path

import calibrate as cb


def softmax(z, t=1.0):
    m = max(z)
    e = [math.exp((x - m) / t) for x in z]
    s = sum(e)
    return [x / s for x in e]


def synthetic_rows(t_true, n=4000, k=4, seed=0, source="syn"):
    """Served probabilities softmax(z); gold labels drawn from softmax(z / t_true)."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        z = [rng.gauss(0, 2) for _ in range(k)]
        y = rng.choices(range(k), weights=softmax(z, t_true))[0]
        rows.append({"source": source, "type": "choice", "probs": softmax(z), "gold": y})
    return rows


class ApplyT(unittest.TestCase):
    def test_identity(self):
        p = [0.7, 0.2, 0.1]
        for a, b in zip(cb.apply_t(p, 1.0), p):
            self.assertAlmostEqual(a, b, places=15)

    def test_equals_softmax_of_scaled_logits(self):
        z = [2.0, 0.0, -1.0, 4.5]
        for t in (0.5, 1.7, 4.0):
            for a, b in zip(cb.apply_t(softmax(z), t), softmax(z, t)):
                self.assertAlmostEqual(a, b, places=12)


class Fit(unittest.TestCase):
    def test_recovers_a_known_temperature(self):
        for t_true in (2.0, 0.5):
            t = cb.fit_temperature(synthetic_rows(t_true))
            self.assertLess(abs(t / t_true - 1), 0.08, (t_true, t))

    def test_nll_is_minimal_at_the_fit(self):
        rows = synthetic_rows(1.6, n=1500, seed=3)
        t = cb.fit_temperature(rows)
        best = cb.nll(rows, t)
        self.assertLessEqual(best, cb.nll(rows, t * 1.05))
        self.assertLessEqual(best, cb.nll(rows, t / 1.05))
        self.assertLess(best, cb.nll(rows, 1.0))

    def test_fit_at_a_bound_raises(self):
        # Always right at p = 0.6: sharpening always helps, the optimum is T -> 0.
        row = {"source": "s", "type": "noul", "probs": [0.6, 0.4], "gold": 0}
        with self.assertRaises(ValueError):
            cb.fit_temperature([row] * 50)

    def test_ship_rule(self):
        hot = cb.fit_types(synthetic_rows(2.0, n=1500))["choice"]
        self.assertEqual(hot["n"], cb.PER_SOURCE_CAP)
        self.assertTrue(hot["ship"], hot)
        calm = cb.fit_types(synthetic_rows(1.0, n=1500))["choice"]
        self.assertFalse(calm["ship"], calm)

    def test_pool_caps_each_source(self):
        rows = synthetic_rows(1.0, n=400, source="a") + synthetic_rows(1.0, n=50, source="b")
        self.assertEqual({s: len(v) for s, v in cb.pool(rows, "choice").items()}, {"a": 300, "b": 50})
        self.assertEqual(cb.pool(rows, "noul"), {})


class Metrics(unittest.TestCase):
    def test_ece_matches_jevbench_binning(self):
        pairs = [(0.95, 1), (0.95, 0), (0.55, 1), (0.0, 0)]
        self.assertAlmostEqual(cb.ece(pairs), 0.45 * 0.5 + 0.45 * 0.25, places=12)

    def test_summarize_keeps_accuracy_and_moves_confidence(self):
        rows = [{"source": "s", "type": "score", "probs": [0.1, 0.2, 0.7], "gold": 2},
                {"source": "s", "type": "score", "probs": [0.6, 0.3, 0.1], "gold": 1}]
        raw = cb.summarize(rows, {})
        cal = cb.summarize(rows, {"score": 2.0})
        self.assertEqual((raw["acc"], cal["acc"]), (0.5, 0.5))
        self.assertEqual(raw["ece_raw"], raw["ece_cal"])
        self.assertNotEqual(cal["ece_raw"], cal["ece_cal"])
        self.assertAlmostEqual(raw["mae_raw"], (0.4 + 0.5) / 2, places=12)
        self.assertIn("mae_cal", cal)


class Freeze(unittest.TestCase):
    def test_bench_row_shapes(self):
        n = cb.slim_bench_row("sms-spam", {"label": "false", "prob": {"true": 0.25, "false": 0.75}})
        self.assertEqual((n["type"], n["probs"], n["gold"]), ("noul", [0.25, 0.75], 1))
        self.assertNotIn("keys", n)
        s = cb.slim_bench_row("yelp-stars", {"label": "3", "prob": {str(i): 0.2 for i in range(5)}})
        self.assertEqual((s["type"], s["gold"]), ("score", 3))
        c = cb.slim_bench_row("banking77", {"label": "b", "prob": {"a": 0.5, "b": 0.5}})
        self.assertEqual((c["type"], c["gold"]), ("choice", 1))
        self.assertEqual(cb.r12(0.1234567890123456), 0.123456789012)

    def test_cal_row_shape(self):
        r = cb.slim_cal_row("synth", {"type": "noul", "option_keys": ["yes", "no"], "probs": [0.9, 0.1],
                                      "gold": "no"})
        self.assertEqual((r["source"], r["gold"], r["probs"]), ("synth", 1, [0.9, 0.1]))

    def test_freeze_checks_rows_against_saved_predictions(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "m"
            (src / "jevbench/raw").mkdir(parents=True)
            (src / "jevbench/predictions").mkdir()
            (src / "calibration").mkdir()
            raw = {"state": "secret text", "label": "true", "prob": {"true": 0.8, "false": 0.2}}
            (src / "jevbench/raw/sms-spam.jsonl").write_text(json.dumps(raw) + "\n")
            pred = src / "jevbench/predictions/sms-spam.jsonl"
            pred.write_text(json.dumps({"i": 0, "p_top": 0.8, "correct": 1}) + "\n")
            out = Path(d) / "probs"
            cb.main(["freeze", "--src", str(src), "--out", str(out)])
            frozen = (out / "jevbench_sms-spam.jsonl").read_text()
            self.assertNotIn("secret", frozen)
            self.assertEqual(json.loads(frozen)["gold"], 0)
            pred.write_text(json.dumps({"i": 0, "p_top": 0.7, "correct": 1}) + "\n")
            with self.assertRaises(SystemExit):
                cb.main(["freeze", "--src", str(src), "--out", str(out)])


class FitCommand(unittest.TestCase):
    def test_fit_refuses_anything_but_heldout(self):
        with tempfile.TemporaryDirectory() as d:
            probs = Path(d) / "probs"
            probs.mkdir()
            with self.assertRaises(SystemExit):
                cb.main(["fit", "--heldout", str(probs), "--model-id", "m", "--build", "b",
                         "--out", str(Path(d) / "c.json")])

    def test_fit_writes_calibration(self):
        with tempfile.TemporaryDirectory() as d:
            h = Path(d) / "heldout"
            h.mkdir()
            cb.write_rows(h / "syn.jsonl", synthetic_rows(2.0, n=1200))
            cb.main(["fit", "--heldout", str(h), "--model-id", "m", "--build", "b",
                     "--out", str(Path(d) / "c.json")])
            doc = json.loads((Path(d) / "c.json").read_text())
            self.assertEqual(doc["types"]["choice"]["sources"], {"syn": 300})
            self.assertEqual(list(doc["decide_calibration"]), ["choice"])
            self.assertIn('[models."m".overrides.decide.calibration]',
                          cb.toml_snippet("m", doc["decide_calibration"]))


if __name__ == "__main__":
    unittest.main()
```

```bash
python3 -m unittest discover -s scripts/jev_eval -p 'test_*.py' 2>&1 | tail -3
```

Expected: `ModuleNotFoundError: No module named 'calibrate'`.

- [ ] **Step 2: Create `scripts/jev_eval/calsets.py`.** It holds
  `run_calibration.py`'s record builders, moved verbatim except that
  `split` and `n` are now parameters:

```python
"""Record builders shared by run_calibration.py and heldout.py: the
scienthoon/jev-ood-calibration sets (synthetic tickets, public QA) and their
decide questions. The validation rebuild mirrors that repo's convert.py
(verified row-for-row against Jev's committed results on 2026-09-28)."""
import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request

NOUL_TRUE, NOUL_FALSE = "Yes, the statement is true.", "No, the statement is false."
# Rows read per public set for the reported (validation) rebuild.
PUBLIC_N = {"openbookqa": 500, "commonsense_qa": 1221, "hellaswag": 2000}


def _get_json(url):
    """GET with patience: datasets-server rate-limits bursts and 5xxs at times."""
    for i in range(8):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except (urllib.error.URLError, TimeoutError):
            if i == 7:
                raise
            time.sleep(min(60, 3 * 2 ** i))


def hf_rows(ds, cfg, n, split="validation"):
    base = (f"https://datasets-server.huggingface.co/rows?dataset={urllib.parse.quote(ds, safe='')}"
            f"&config={cfg}&split={split}")
    rows = []
    for off in range(0, n, 100):
        rows += [x["row"] for x in _get_json(base + f"&offset={off}&length={min(100, n - off)}")["rows"]]
    return rows


def public_records(name, split="validation", n=None):
    """The first n rows of `split`, filtered, then random.Random(1).shuffle.
    With the defaults: exactly the reported validation rows."""
    n = PUBLIC_N[name] if n is None else n
    if name == "openbookqa":
        src = hf_rows("allenai/openbookqa", "main", n, split)
        recs = [{"state": s["question_stem"], "type": "choice", "question": "Which option is the correct answer?",
                 "options": dict(zip(s["choices"]["label"], s["choices"]["text"])), "label": str(s["answerKey"])}
                for s in src]
    elif name == "commonsense_qa":
        src = hf_rows("tau/commonsense_qa", "default", n, split)
        recs = [{"state": s["question"], "type": "choice", "question": "Which option is the correct answer?",
                 "options": dict(zip(s["choices"]["label"], s["choices"]["text"])), "label": str(s["answerKey"])}
                for s in src]
    else:
        src = hf_rows("Rowan/hellaswag", "default", n, split)
        recs = [{"state": s["ctx"], "type": "choice", "question": "Which ending most plausibly continues the text?",
                 "options": {str(i): e for i, e in enumerate(s["endings"])}, "label": str(s["label"]).strip()}
                for s in src]
    recs = [r for r in recs if r["label"] in r["options"]]
    random.Random(1).shuffle(recs)
    for r in recs:
        r["source"] = name
    return recs


def load_synth(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def to_question(r):
    """(decide question, option keys, one-hot target) for one record."""
    if r["type"] == "choice":
        keys = list(r["options"])
        return ({"type": "choice", "instructions": r["question"], "criteria": r["options"]},
                keys, [1.0 if k == r["label"] else 0.0 for k in keys])
    if r["type"] == "score":
        keys = [str(i) for i in range(len(r["levels"]))]
        return ({"type": "score", "instructions": r["question"], "criteria": [str(x) for x in r["levels"]]},
                keys, [1.0 if i == int(r["label"]) else 0.0 for i in range(len(keys))])
    return ({"type": "noul", "instructions": r["question"], "criteria": {"true": NOUL_TRUE, "false": NOUL_FALSE}},
            ["yes", "no"], [1.0, 0.0] if r["label"] is True else [0.0, 1.0])
```

- [ ] **Step 3: Point `run_calibration.py` at it.** Replace everything
  from `import random` through the end of `def to_question(r): …`
  (imports, `NOUL_TRUE/NOUL_FALSE`, `hf_rows`, `public_records`,
  `to_question`) so that the file reads:

```python
"""Run the scienthoon/jev-ood-calibration sets against local hipfire serve,
writing rows in that repo's result schema (type, option_keys, probs, target,
pred, gold, confidence, source, usage)."""
import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

from calsets import load_synth, public_records, to_question

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=11435)
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("sets", nargs="+", choices=["synth", "openbookqa", "commonsense_qa", "hellaswag"])
a = ap.parse_args()
cal = Path(os.environ.get("JEVCAL_DIR", Path.home() / "repos/jev-evals/jev-ood-calibration"))
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
URL = f"http://127.0.0.1:{a.port}/v1/systemone"


def ask(r):
    q, keys, target = to_question(r)
    body = json.dumps({"model": a.model, "state": r["state"], "questions": {"q": q}}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        v = json.load(resp)
    ans = v["answers"]["q"]
    probs = [ans["noul"], 1 - ans["noul"]] if r["type"] == "noul" else [ans["probabilities"][k] for k in keys]
    pred = keys[max(range(len(probs)), key=probs.__getitem__)]
    gold = keys[max(range(len(target)), key=target.__getitem__)]
    return {"type": r["type"], "option_keys": keys, "probs": probs, "target": target, "pred": pred,
            "gold": gold, "confidence": ans.get("confidence"), "source": r.get("source"),
            "usage": {"inputTokens": v["usage"]["input_tokens"], "outputTokens": 0}}


for s in a.sets:
    recs = load_synth(cal / "data/val.jsonl") if s == "synth" else public_records(s)
    path = out / f"hipfire_{s}.jsonl"
    with open(path, "w") as f:
        for i, r in enumerate(recs):
            f.write(json.dumps(ask(r)) + "\n")
            if i % 100 == 0:
                print(f"{s}: {i}/{len(recs)}", file=sys.stderr)
    print(f"wrote {path}")
```

Check that the refactor did not change the reported rows' construction
(CPU, network: it rebuilds OpenBookQA validation both ways):

```bash
cd /home/nick/repos/hipfire-jev-calib
python3 - <<'EOF'
import json, random, subprocess, sys
sys.dont_write_bytecode = True
sys.path.insert(0, "scripts/jev_eval")
import calsets
old = subprocess.run(["git", "show", "HEAD:scripts/jev_eval/run_calibration.py"], capture_output=True, text=True).stdout
ns = {}
exec(old.split("for s in a.sets:")[0].split("ap = argparse.ArgumentParser()")[0] + "\n" +
     old[old.index("def hf_rows"):old.index("def to_question")], ns)
assert ns["public_records"]("openbookqa") == calsets.public_records("openbookqa")
print("openbookqa validation rebuild unchanged")
EOF
```

Expected: `openbookqa validation rebuild unchanged`.

- [ ] **Step 4: Create `scripts/jev_eval/calibrate.py`**

```python
"""Decide calibration (spec §13): freeze the reported predictions, fit one
temperature per question type on held-out rows, apply it.

Rows everywhere are {"source", "type", "probs", "gold"}: the probabilities
as served (raw, T = 1) in answer-key order, gold an index into them. No text,
no keys (git stores these files zlib-compressed; the keys would dominate).

  freeze  --src ~/repos/hipfire-jev-eval/bench/jev/<model> --out bench/jev/<model>/probs
  fit     --heldout bench/jev/<model>/heldout --model-id <models.toml id>
          --build <sha> --out bench/jev/<model>/calibration.json
"""
import argparse
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

TYPES = ("choice", "score", "noul")
T_MIN, T_MAX = 0.05, 20.0
PER_SOURCE_CAP = 300
BOOTSTRAP = 100
MIN_NLL_GAIN = 0.01
SCORE_TASKS = {"yelp-stars", "sentiment-it"}
NOUL_TASKS = {"sms-spam", "duplicates", "doc-yesno", "offensive"}


def r12(x):
    """12 significant digits: log p stays exact to ~1e-12, files stay small."""
    return float(f"{x:.12g}")


def argmax(xs):
    return max(range(len(xs)), key=xs.__getitem__)


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_rows(paths):
    rows = []
    for p in paths:
        for r in read_jsonl(p):
            if min(r["probs"]) <= 0:
                raise ValueError(f"{p}: a probability <= 0 has no label logit (spec §13.3)")
            rows.append(r)
    return rows


def apply_t(probs, t):
    """softmax(log p / t) == softmax(z / t), since log p = z - logsumexp(z)."""
    s = [math.log(p) / t for p in probs]
    m = max(s)
    e = [math.exp(x - m) for x in s]
    tot = sum(e)
    return [x / tot for x in e]


def ece(pairs, bins=10):
    """Top-probability ECE, binned exactly as jevbench.py's ece()."""
    e = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        ins = [(p, c) for p, c in pairs if lo < p <= hi or (b == 0 and p == 0)]
        if ins:
            e += len(ins) / len(pairs) * abs(sum(p for p, _ in ins) / len(ins) - sum(c for _, c in ins) / len(ins))
    return e


def _prep(rows):
    return [([math.log(p) for p in r["probs"]], r["gold"]) for r in rows]


def _nll(data, t):
    tot = 0.0
    for lp, y in data:
        s = [x / t for x in lp]
        m = max(s)
        tot += m + math.log(sum(math.exp(x - m) for x in s)) - s[y]
    return tot / len(data)


def nll(rows, t):
    """Mean negative log-likelihood of the gold labels at temperature t."""
    return _nll(_prep(rows), t)


def _grad_hess(data, b):
    """dNLL/db and d2NLL/db2 at b = 1/T: mean(E_q[l] - l_y), mean(Var_q[l])."""
    g = h = 0.0
    for lp, y in data:
        s = [b * x for x in lp]
        m = max(s)
        w = [math.exp(x - m) for x in s]
        z = sum(w)
        mean = sum(wi * x for wi, x in zip(w, lp)) / z
        g += mean - lp[y]
        h += sum(wi * (x - mean) ** 2 for wi, x in zip(w, lp)) / z
    return g / len(data), h / len(data)


def _fit(data, strict=True):
    """argmin over T in [T_MIN, T_MAX] of NLL. NLL is convex in b = 1/T, so
    safeguarded Newton on its monotone derivative. strict: a bound is an error."""
    lo, hi = 1.0 / T_MAX, 1.0 / T_MIN
    b = 1.0
    for _ in range(200):
        g, h = _grad_hess(data, b)
        if g > 0:
            hi = b
        else:
            lo = b
        nb = b - g / h if h > 0 else (lo + hi) / 2
        if not lo < nb < hi:
            nb = (lo + hi) / 2
        if abs(nb - b) <= 1e-12 * b:
            b = nb
            break
        b = nb
    t = 1.0 / b
    if strict and (t <= T_MIN * (1 + 1e-6) or t >= T_MAX * (1 - 1e-6)):
        raise ValueError(f"fitted T={t:.6g} is at the search bound [{T_MIN}, {T_MAX}]")
    return t


def fit_temperature(rows):
    return _fit(_prep(rows))


def bootstrap_ci(data, n=BOOTSTRAP, seed=0):
    """95% interval of T over n resamples (a resample at a bound counts at the bound)."""
    rng = random.Random(seed)
    ts = sorted(_fit(rng.choices(data, k=len(data)), strict=False) for _ in range(n))
    return ts[int(0.025 * n)], ts[int(0.975 * n) - 1]


def pool(rows, qtype):
    """Rows of one type, at most PER_SOURCE_CAP per source (first ones)."""
    by = defaultdict(list)
    for r in rows:
        if r["type"] == qtype:
            by[r["source"]].append(r)
    return {s: rs[:PER_SOURCE_CAP] for s, rs in sorted(by.items())}


def fit_types(rows):
    """Per type: T, its 95% interval, held-out NLL raw / calibrated, ship rule (spec §13.4)."""
    out = {}
    for qtype in TYPES:
        srcs = pool(rows, qtype)
        data = _prep([r for rs in srcs.values() for r in rs])
        if not data:
            continue
        entry = {"n": len(data), "sources": {s: len(v) for s, v in srcs.items()}}
        try:
            t = round(_fit(data), 3)
        except ValueError as e:
            out[qtype] = {**entry, "error": str(e), "ship": False}
            continue
        lo, hi = bootstrap_ci(data)
        raw, cal = _nll(data, 1.0), _nll(data, t)
        out[qtype] = {**entry, "t": t, "t_ci95": [round(lo, 3), round(hi, 3)], "nll_raw": raw,
                      "nll_cal": cal, "ship": (raw - cal) / raw >= MIN_NLL_GAIN and not lo <= 1.0 <= hi}
    return out


def summarize(rows, temps):
    """Accuracy, ECE and NLL raw vs calibrated at temps ({type: T}, missing = 1);
    for score rows also mean |E[score] - gold|. Asserts no argmax moved."""
    raw, cal = [], []
    nll_raw = nll_cal = mae_raw = mae_cal = 0.0
    n_score = 0
    for r in rows:
        p = r["probs"]
        q = apply_t(p, temps.get(r["type"], 1.0))
        top = argmax(p)
        if argmax(q) != top:
            raise AssertionError(f"{r['source']}: calibration moved an argmax")
        c = int(top == r["gold"])
        raw.append((p[top], c))
        cal.append((q[top], c))
        nll_raw -= math.log(p[r["gold"]])
        nll_cal -= math.log(q[r["gold"]])
        if r["type"] == "score":
            n_score += 1
            mae_raw += abs(sum(i * x for i, x in enumerate(p)) - r["gold"])
            mae_cal += abs(sum(i * x for i, x in enumerate(q)) - r["gold"])
    n = len(rows)
    out = {"n": n, "acc": sum(c for _, c in raw) / n, "ece_raw": ece(raw), "ece_cal": ece(cal),
           "nll_raw": nll_raw / n, "nll_cal": nll_cal / n}
    if n_score:
        out["mae_raw"], out["mae_cal"] = mae_raw / n_score, mae_cal / n_score
    return out


def slim_bench_row(task, r):
    """A jevbench raw/ row (full `prob` distribution) without its text."""
    keys = list(r["prob"])
    qtype = "noul" if task in NOUL_TASKS else "score" if task in SCORE_TASKS else "choice"
    return {"source": task, "type": qtype, "probs": [r12(r["prob"][k]) for k in keys],
            "gold": keys.index(r["label"])}


def slim_cal_row(name, r):
    """A run_calibration.py row without its usage/confidence fields."""
    keys = r["option_keys"]
    return {"source": name, "type": r["type"], "probs": [r12(p) for p in r["probs"]],
            "gold": keys.index(r["gold"])}


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_rows(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def cmd_freeze(a):
    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sources = {}
    for p in sorted((src / "jevbench/raw").glob("*.jsonl")):
        if p.stem.startswith("x-"):
            continue  # the experiments are not in the report tables
        rows = [slim_bench_row(p.stem, r) for r in read_jsonl(p)]
        pred_path = src / "jevbench/predictions" / f"{p.stem}.jsonl"
        preds = read_jsonl(pred_path)
        if len(preds) != len(rows):
            sys.exit(f"{p.stem}: {len(rows)} raw rows vs {len(preds)} saved predictions")
        for r, pr in zip(rows, preds):
            top = argmax(r["probs"])
            if abs(r["probs"][top] - pr["p_top"]) > 5e-5 or int(top == r["gold"]) != pr["correct"]:
                sys.exit(f"{p.stem} row {pr['i']}: raw row does not match the saved prediction")
        write_rows(out / f"jevbench_{p.stem}.jsonl", rows)
        sources[str(p)] = sha256_file(p)
        sources[str(pred_path)] = sha256_file(pred_path)
    for p in sorted((src / "calibration").glob("hipfire_*.jsonl")):
        name = p.stem.removeprefix("hipfire_")
        write_rows(out / f"cal_{name}.jsonl", [slim_cal_row(name, r) for r in read_jsonl(p)])
        sources[str(p)] = sha256_file(p)
    (out / "SOURCES.json").write_text(json.dumps(sources, indent=1, sort_keys=True) + "\n")
    print(f"froze {len(sources)} source files -> {out}")


def toml_snippet(model_id, temps):
    if not temps:
        return f"# {model_id}: no question type passed the ship rule; leave it uncalibrated"
    body = "\n".join(f"{k} = {v}" for k, v in temps.items())
    return f'[models."{model_id}".overrides.decide.calibration]\n{body}'


def cmd_fit(a):
    src = Path(a.heldout)
    if src.name != "heldout":
        sys.exit("fit reads only a heldout/ directory: reported rows never feed the fit (spec §13.4)")
    rows = load_rows(sorted(src.glob("*.jsonl")))
    types = fit_types(rows)
    ship = {t: v["t"] for t, v in types.items() if v["ship"]}
    doc = {"model_id": a.model_id, "build": a.build, "types": types, "decide_calibration": ship}
    Path(a.out).write_text(json.dumps(doc, indent=1) + "\n")
    print(toml_snippet(a.model_id, ship))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--src", required=True)
    f.add_argument("--out", required=True)
    g = sub.add_parser("fit")
    g.add_argument("--heldout", required=True)
    g.add_argument("--model-id", required=True)
    g.add_argument("--build", required=True)
    g.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    {"freeze": cmd_freeze, "fit": cmd_fit}[a.cmd](a)


if __name__ == "__main__":
    main()
```

```bash
python3 -m unittest discover -s scripts/jev_eval -p 'test_calibrate.py' -v 2>&1 | tail -4
```

Expected: `Ran 14 tests`, `OK`.

- [ ] **Step 5: Rewrite `scripts/jev_eval/report.py`**

```python
"""hipfire decide vs Jev: accuracy, top-probability ECE and log loss, raw vs
calibrated (spec §13.6). Reads the frozen reported rows (<out>/probs/) and the
fitted temperatures (<out>/calibration.json; without it every T is 1). No GPU,
no dataset text."""
import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import calibrate as cb

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True, help="bench/jev/<model>")
a = ap.parse_args()
out = Path(a.out)
bench = Path(os.environ.get("JEVBENCH_DIR", Path.home() / "repos/jev-evals/jev-bench"))
cal = Path(os.environ.get("JEVCAL_DIR", Path.home() / "repos/jev-evals/jev-ood-calibration"))
cfile = out / "calibration.json"
fit = json.loads(cfile.read_text()) if cfile.exists() else None
temps = fit["decide_calibration"] if fit else {}


def jev_bench(name):
    rows = cb.read_jsonl(bench / "predictions" / f"{name}.jsonl")
    return sum(r["correct"] for r in rows) / len(rows), cb.ece([(r["p_top"], r["correct"]) for r in rows])


def jev_cal(name):
    g = defaultdict(list)
    for r in cb.read_jsonl(cal / "results" / f"jev_{name}.jsonl"):
        if "probs" in r:
            g[r["type"]].append((max(r["probs"]), int(r["pred"] == r["gold"])))
    return {t: (sum(c for _, c in v) / len(v), cb.ece(v)) for t, v in g.items()}


def f3(x):
    return f"{x:.3f}"


def cells(s):
    return f"{f3(s['acc'])} | {{je}} | {f3(s['ece_raw'])} | {f3(s['ece_cal'])} | {f3(s['nll_raw'])} | {f3(s['nll_cal'])} |"


applied = ", ".join(f"{t} T={temps.get(t, 1.0):g}" for t in cb.TYPES)
L = ["# hipfire decide vs Jev", "",
     f"Calibration applied: {applied} "
     f"({'from calibration.json' if fit else 'no calibration.json, so calibrated = raw'}). "
     "Calibrated = softmax(log p / T) per question type (spec §13.3); accuracy cannot change.", ""]
if fit:
    L += [f"## Fit (held-out rows only, build {fit['build']}, spec §13.4)", "",
          "| type | n | T | 95% interval | held-out NLL raw | held-out NLL cal | shipped |",
          "|---|---:|---:|---|---:|---:|---|"]
    for t, v in fit["types"].items():
        if "error" in v:
            L.append(f"| {t} | {v['n']} | - | {v['error']} | - | - | no |")
            continue
        lo, hi = v["t_ci95"]
        L.append(f"| {t} | {v['n']} | {v['t']:.3f} | {lo:.3f}-{hi:.3f} | {f3(v['nll_raw'])} | "
                 f"{f3(v['nll_cal'])} | {'yes' if v['ship'] else 'no'} |")
    L.append("")
L += ["## jev-bench (500 fixed-seed rows per task)", "",
     "| task | n | Jev acc | hipfire acc | Jev ECE | ECE raw | ECE cal | NLL raw | NLL cal |",
     "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
score = []
for p in sorted(glob.glob(str(out / "probs/jevbench_*.jsonl"))):
    name = Path(p).stem.removeprefix("jevbench_")
    s = cb.summarize(cb.read_jsonl(p), temps)
    jacc, je = jev_bench(name)
    L.append(f"| {name} | {s['n']} | {f3(jacc)} | " + cells(s).format(je=f3(je)))
    if "mae_raw" in s:
        score.append(f"| {name} | {s['n']} | {f3(s['mae_raw'])} | {f3(s['mae_cal'])} |")
L += ["", "## jev-ood-calibration", "",
      "| set | type | n | Jev acc | hipfire acc | Jev ECE | ECE raw | ECE cal | NLL raw | NLL cal |",
      "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
for p in sorted(glob.glob(str(out / "probs/cal_*.jsonl"))):
    name = Path(p).stem.removeprefix("cal_")
    by = defaultdict(list)
    for r in cb.read_jsonl(p):
        by[r["type"]].append(r)
    jev = jev_cal(name)
    for t in cb.TYPES:
        if t not in by:
            continue
        s = cb.summarize(by[t], temps)
        jacc, je = jev.get(t, (float("nan"), float("nan")))
        L.append(f"| {name} | {t} | {s['n']} | {f3(jacc)} | " + cells(s).format(je=f3(je)))
        if "mae_raw" in s:
            score.append(f"| {name} ({t}) | {s['n']} | {f3(s['mae_raw'])} | {f3(s['mae_cal'])} |")
if score:
    L += ["", "## score answers: mean |E[score] - gold|", "",
          "| rows | n | raw | calibrated |", "|---|---:|---:|---:|"] + score
(out / "report.md").write_text("\n".join(L) + "\n")
print((out / "report.md").read_text())
```

- [ ] **Step 6: Run the unit tests in the no-GPU CI.** In
  `scripts/no-gpu-ci.sh`, after the `python3 -m unittest tools.redline…`
  line, add:

```bash
python3 -m unittest discover -s scripts/jev_eval -p 'test_*.py'
```

- [ ] **Step 7: Freeze the reported rows** (CPU; reads the eval worktree,
  writes only under this repo's `bench/jev/`)

```bash
cd /home/nick/repos/hipfire-jev-calib
for m in qwen3.5-4b qwen3.8-27b-mq4-xts; do
  python3 scripts/jev_eval/calibrate.py freeze --src /home/nick/repos/hipfire-jev-eval/bench/jev/$m --out bench/jev/$m/probs
done
grep -l '"state"' bench/jev/*/probs/*.jsonl   # must print nothing
python3 scripts/jev_eval/report.py --out bench/jev/qwen3.5-4b | head -12
python3 scripts/jev_eval/report.py --out bench/jev/qwen3.8-27b-mq4-xts | grep -E "ledgar|banking77"
git diff --stat bench/jev/qwen3.5-4b/report.md bench/jev/qwen3.8-27b-mq4-xts/report.md
```

Expected:

- `froze 7 source files` (4B: 3 tasks + predictions + synth) and
  `froze 28 source files` (27B: 12 tasks + predictions + 4 sets);
- every raw row matched its saved prediction, or `freeze` would have
  exited;
- the regenerated reports' `hipfire acc` and `ECE raw` columns equal the
  committed v1 reports (4B banking77: 0.546 / 0.188; 27B ledgar: 0.752 /
  0.121), and `ECE cal` equals `ECE raw`, since there is no
  `calibration.json` yet.

About 6 MB of JSONL (git stores it zlib-compressed, about 2 MB).

- [ ] **Step 8: Commit**

```bash
./scripts/no-gpu-ci.sh 2>&1 | tail -5
git add scripts/jev_eval/calsets.py scripts/jev_eval/run_calibration.py scripts/jev_eval/calibrate.py \
  scripts/jev_eval/report.py scripts/jev_eval/test_calibrate.py scripts/no-gpu-ci.sh \
  bench/jev/qwen3.5-4b/probs bench/jev/qwen3.8-27b-mq4-xts/probs \
  bench/jev/qwen3.5-4b/report.md bench/jev/qwen3.8-27b-mq4-xts/report.md
git commit -m "feat(jev_eval): calibration fit/apply library, raw-vs-calibrated report, frozen reported rows

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Held-out set builder (no overlap with reported rows)

**Files:**
- Create: `scripts/jev_eval/heldout.py`
- Create: `scripts/jev_eval/test_heldout.py`
- Create: `bench/jev/heldout/manifest.json`,
  `bench/jev/heldout/reported_state_sha256.txt`
- Work dir (not committed): `$JEV_CALIB_WORK/{jevbench/data,examples}/`

**Interfaces:**
- Consumes:
  - `calsets.*`;
  - jevbench's `TASKS`, `hf_rows`, `_get`, `ROOT` (module attributes,
    patched and then restored);
  - `generate.build_synthetic(n, label_noise, seed)`.
- Produces:
  - `heldout.state_hash(state)`, `keys_for(question)`,
    `select(cands, reported, cap)`, `jevbench_candidates(jb, task, split)`,
    `verify_dirs(out, work)`, `answer_row(...)`, `read_jsonl`, `WORK`,
    `OUT`;
  - the CLI `build | verify | run`;
  - work-dir examples `{"state", "question", "keys", "gold"}`.

- [ ] **Step 1: Write the failing tests** (`scripts/jev_eval/test_heldout.py`)

```python
"""CPU tests for heldout.py (spec §13.5, §13.7). No network.
Run: python3 -m unittest discover -s scripts/jev_eval -p 'test_*.py'"""
import json
import tempfile
import unittest
from pathlib import Path

import heldout as h


class StateHash(unittest.TestCase):
    def test_normalisation(self):
        self.assertEqual(h.state_hash(" hi \n"), h.state_hash("hi"))
        self.assertEqual(h.state_hash({"b": 1, "a": "é"}), h.state_hash({"a": "é", "b": 1}))
        self.assertNotEqual(h.state_hash({"a": 1}), h.state_hash('{"a": 1}'))
        self.assertNotEqual(h.state_hash("Hi"), h.state_hash("hi"))


class Keys(unittest.TestCase):
    def test_keys_follow_the_decide_answer_order(self):
        self.assertEqual(h.keys_for({"type": "choice", "criteria": {"b": "x", "a": "y"}}), ["b", "a"])
        self.assertEqual(h.keys_for({"type": "score", "criteria": ["lo", "mid", "hi"]}), ["0", "1", "2"])
        self.assertEqual(h.keys_for({"type": "noul", "instructions": "i"}), ["true", "false"])


class Select(unittest.TestCase):
    def test_drops_reported_and_repeats_and_caps(self):
        cands = [{"state": s} for s in ["a", "b", "a", "c", "d", "e"]]
        kept, excluded, repeats = h.select(cands, {h.state_hash("c")}, cap=3)
        self.assertEqual([c["state"] for c in kept], ["a", "b", "d"])
        self.assertEqual((excluded, repeats), (1, 1))

    def test_object_states_match_regardless_of_key_order(self):
        kept, excluded, _ = h.select([{"state": {"x": 1, "y": 2}}], {h.state_hash({"y": 2, "x": 1})})
        self.assertEqual((kept, excluded), ([], 1))


class Redirect(unittest.TestCase):
    def test_task_loader_is_redirected_then_restored(self):
        calls = []

        class JB:
            pass

        jb = JB()

        def hf_rows(dataset, config, split, n, seed=0, page=100):
            calls.append((dataset, split, n, seed))
            return [{"x": 1}], []

        def get(url):
            calls.append(url)
            return b""

        def task(n):  # shaped like jevbench's task constructors
            jb.hf_rows("ds", "cfg", "test", 3000)
            jb._get("https://x/banking_data/test.csv")
            return [{"state": "s", "label": "l"}], {"type": "noul"}

        jb.hf_rows, jb._get, jb.TASKS = hf_rows, get, {"t": (task, "", "")}
        ex = h.jevbench_candidates(jb, "t", "validation")
        self.assertEqual(ex, [{"state": "s", "label": "l"}])
        self.assertEqual(calls, [("ds", "validation", h.DRAW, h.HELDOUT_SEED),
                                 "https://x/banking_data/train.csv"])
        self.assertIs(jb.hf_rows, hf_rows)
        self.assertIs(jb._get, get)


class Verify(unittest.TestCase):
    def test_catches_an_overlap_and_a_changed_examples_file(self):
        with tempfile.TemporaryDirectory() as d:
            out, work = Path(d) / "out", Path(d) / "work"
            (work / "examples").mkdir(parents=True)
            out.mkdir()
            ex = work / "examples/src.jsonl"
            ex.write_text(json.dumps({"state": "fresh"}) + "\n")
            (out / "manifest.json").write_text(json.dumps({"src": {"kept": 1, "sha256": h.sha256_file(ex)}}))
            hashes = out / "reported_state_sha256.txt"
            hashes.write_text(h.state_hash("old") + "\n")
            h.verify_dirs(out, work)
            hashes.write_text(h.state_hash("fresh") + "\n")
            with self.assertRaises(AssertionError):
                h.verify_dirs(out, work)
            hashes.write_text(h.state_hash("old") + "\n")
            ex.write_text(json.dumps({"state": "changed"}) + "\n")
            with self.assertRaises(AssertionError):
                h.verify_dirs(out, work)


class AnswerRow(unittest.TestCase):
    def test_noul_and_choice_rows(self):
        n = h.answer_row("sms-spam", "noul", ["true", "false"], 1, {"type": "noul", "noul": 0.25})
        self.assertEqual((n["probs"], n["gold"]), ([0.25, 0.75], 1))
        c = h.answer_row("ag-news", "choice", ["b", "a"], 0,
                         {"type": "choice", "probabilities": {"a": 0.4, "b": 0.6}, "confidence": 0.2})
        self.assertEqual(c["probs"], [0.6, 0.4])
        self.assertNotIn("state", c)


if __name__ == "__main__":
    unittest.main()
```

```bash
python3 -m unittest discover -s scripts/jev_eval -p 'test_heldout.py' 2>&1 | tail -3
```

Expected: `ModuleNotFoundError: No module named 'heldout'`.

- [ ] **Step 2: Create `scripts/jev_eval/heldout.py`**

```python
"""Held-out labelled rows for fitting decide calibration (spec §13.5).

  build   draw candidates, drop every one whose state is a reported row's,
          keep PER_SOURCE per source. Examples (with text) go to the work dir;
          the text-free manifest and the reported-state hash list go to
          bench/jev/heldout/.
  verify  re-check the committed hash list and manifest against the
          work-dir examples: no held-out state is a reported state.
  run     ask a running serve every held-out example, one question per
          request as the reported runs did; write text-free answer rows.

The external clones are read-only: jevbench's cache ROOT is redirected to the
work dir, and no bytecode is written next to modules imported from them."""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path

sys.dont_write_bytecode = True
import calsets  # noqa: E402  (scripts/jev_eval is sys.path[0])

REPO = Path(__file__).resolve().parents[2]
BENCH = Path(os.environ.get("JEVBENCH_DIR", Path.home() / "repos/jev-evals/jev-bench"))
CAL = Path(os.environ.get("JEVCAL_DIR", Path.home() / "repos/jev-evals/jev-ood-calibration"))
EVAL = Path(os.environ.get("JEV_EVAL_DIR", Path.home() / "repos/hipfire-jev-eval/bench/jev"))
WORK = Path(os.environ.get("JEV_CALIB_WORK", Path.home() / ".cache/hipfire-jev-calib"))
OUT = REPO / "bench/jev/heldout"
HELDOUT_SEED = 1  # jevbench.SEED is 0
SYNTH_SEED = 7  # generate.py's documented fresh seed; data/val.jsonl is seed 0
SYNTH_TICKETS = 400
DRAW = 400
PER_SOURCE = 300
REPORTED_MODELS = ("qwen3.5-4b", "qwen3.8-27b-mq4-xts")
# jev-bench task -> held-out split (spec §13.5 table).
JEVBENCH_SPLIT = {
    "banking77": "train", "massive-en": "validation", "massive-it": "validation",
    "clinc150": "validation", "ledgar": "validation", "ag-news": "train",
    "sms-spam": "train", "duplicates": "train", "doc-yesno": "train",
    "offensive": "validation", "yelp-stars": "train", "sentiment-it": "validation",
}
PUBLIC_TRAIN = ("openbookqa", "commonsense_qa")


def state_hash(state):
    """SHA-256 of the state text: strings as-is, objects as sorted compact JSON."""
    text = state if isinstance(state, str) else json.dumps(
        state, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def keys_for(question):
    """Answer keys in the order decide returns them (noul: true, false)."""
    if question["type"] == "choice":
        return list(question["criteria"])
    if question["type"] == "score":
        return [str(i) for i in range(len(question["criteria"]))]
    return ["true", "false"]


def select(cands, reported, cap=PER_SOURCE):
    """Selection rules 1-3: drop reported states, drop repeats, keep the first cap.
    Returns (kept, n_excluded_reported, n_repeats)."""
    kept, seen, excluded, repeats = [], set(), 0, 0
    for c in cands:
        h = state_hash(c["state"])
        if h in reported:
            excluded += 1
        elif h in seen:
            repeats += 1
        else:
            seen.add(h)
            kept.append(c)
    return kept[:cap], excluded, repeats


def jevbench():
    """jevbench with its dataset cache redirected to WORK; the clone is never written."""
    root = WORK / "jevbench"
    (root / "data").mkdir(parents=True, exist_ok=True)
    for f in (BENCH / "data").glob("*.json"):
        if not (root / "data" / f.name).exists():
            shutil.copy2(f, root / "data" / f.name)
    sys.path.insert(0, str(BENCH))
    import jevbench as jb
    jb.ROOT = root
    return jb


def jevbench_candidates(jb, task, split):
    """The task's own constructor with its loader redirected to the held-out split."""
    orig_rows, orig_get = jb.hf_rows, jb._get

    def rows(dataset, config, _split, _n, seed=0, page=100):
        return orig_rows(dataset, config, split, DRAW, seed=HELDOUT_SEED, page=page)

    def get(url):
        return orig_get(url.replace("/banking_data/test.csv", "/banking_data/train.csv"))

    jb.hf_rows, jb._get = rows, get
    try:
        return jb.TASKS[task][0](DRAW)[0]
    finally:
        jb.hf_rows, jb._get = orig_rows, orig_get


def reported_hashes(jb):
    """R (spec §13.5): every state of a reported row."""
    hashes = set()
    for task in jb.TASKS:
        hashes |= {state_hash(e["state"]) for e in jb.TASKS[task][0](500)[0]}
    for model in REPORTED_MODELS:
        files = sorted((EVAL / model / "jevbench/raw").glob("*.jsonl"))
        if not files:
            sys.exit(f"no saved raw rows under {EVAL / model}: R would be incomplete")
        for f in files:
            hashes |= {state_hash(r["state"]) for r in read_jsonl(f)}
    hashes |= {state_hash(r["state"]) for r in calsets.load_synth(CAL / "data/val.jsonl")}
    for name in calsets.PUBLIC_N:
        hashes |= {state_hash(r["state"]) for r in calsets.public_records(name)}
    return hashes


def record_examples(recs):
    out = []
    for r in recs:
        q, keys, target = calsets.to_question(r)
        out.append({"state": r["state"], "question": q, "keys": keys, "gold": target.index(1.0)})
    return out


def candidates(jb):
    """[(source, type, split, seed, n drawn, candidate examples)] in build order."""
    out = []
    for task, split in JEVBENCH_SPLIT.items():
        q = jb.TASKS[task][0](500)[1]  # the reported question, criteria in reported order
        keys = keys_for(q)
        drawn = jevbench_candidates(jb, task, split)
        ex = [{"state": e["state"], "question": q, "keys": keys, "gold": keys.index(e["label"])}
              for e in drawn if e["label"] in keys]
        out.append((task, q["type"], split, HELDOUT_SEED, len(drawn), ex))
    sys.path.insert(0, str(CAL / "data"))
    import generate
    synth = generate.build_synthetic(SYNTH_TICKETS, 0.05, SYNTH_SEED)
    for qtype in ("choice", "score", "noul"):
        recs = [r for r in synth if r["type"] == qtype]
        out.append((f"synth-{qtype}", qtype, f"generate.py seed {SYNTH_SEED}", SYNTH_SEED, len(recs),
                    record_examples(recs)))
    for name in PUBLIC_TRAIN:
        recs = calsets.public_records(name, split="train", n=DRAW)
        out.append((name, "choice", "train", 1, DRAW, record_examples(recs)))
    return out


def cmd_build(_a):
    jb = jevbench()
    rep = reported_hashes(jb)
    OUT.mkdir(parents=True, exist_ok=True)
    (WORK / "examples").mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, qtype, split, seed, drawn, ex in candidates(jb):
        kept, excluded, repeats = select(ex, rep)
        assert not {state_hash(c["state"]) for c in kept} & rep, name
        path = WORK / "examples" / f"{name}.jsonl"
        with open(path, "w") as f:
            for c in kept:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        manifest[name] = {"type": qtype, "split": split, "seed": seed, "drawn": drawn,
                          "label_filtered": drawn - len(ex), "excluded_reported": excluded,
                          "repeats": repeats, "kept": len(kept), "sha256": sha256_file(path)}
        print(f"{name:<16} {qtype:<6} drawn {drawn:>4}  excluded {excluded:>3}  kept {len(kept)}")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    (OUT / "reported_state_sha256.txt").write_text("\n".join(sorted(rep)) + "\n")
    verify_dirs(OUT, WORK)


def verify_dirs(out, work):
    """No held-out example is a reported state, and every examples file is the one built."""
    rep = set((out / "reported_state_sha256.txt").read_text().split())
    manifest = json.loads((out / "manifest.json").read_text())
    total = 0
    for name, m in manifest.items():
        path = work / "examples" / f"{name}.jsonl"
        assert sha256_file(path) == m["sha256"], f"{name}: examples changed since build"
        rows = read_jsonl(path)
        assert len(rows) == m["kept"], f"{name}: {len(rows)} rows, manifest says {m['kept']}"
        hit = sum(state_hash(r["state"]) in rep for r in rows)
        assert hit == 0, f"{name}: {hit} held-out rows are reported rows"
        total += len(rows)
    print(f"verify: {total} held-out rows in {len(manifest)} sources, none among {len(rep)} reported states")


def cmd_verify(_a):
    verify_dirs(OUT, WORK)


def wait_health(port, timeout=900):
    deadline = time.monotonic() + timeout
    while True:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if r.status == 200:
                    return
        except OSError:
            pass
        if time.monotonic() > deadline:
            sys.exit(f"serve on :{port} not healthy after {timeout}s")
        time.sleep(2)


def answer_row(name, qtype, keys, gold, ans):
    probs = [ans["noul"], 1 - ans["noul"]] if qtype == "noul" else [ans["probabilities"][k] for k in keys]
    return {"source": name, "type": qtype, "probs": [float(f"{p:.12g}") for p in probs], "gold": gold}


def cmd_run(a):
    manifest = json.loads((OUT / "manifest.json").read_text())
    out = Path(a.out)
    if out.name != "heldout":
        sys.exit("--out must be bench/jev/<model>/heldout")
    out.mkdir(parents=True, exist_ok=True)
    verify_dirs(OUT, WORK)
    wait_health(a.port)
    url = f"http://127.0.0.1:{a.port}/v1/systemone"
    for name in a.sources or list(manifest):
        rows = read_jsonl(WORK / "examples" / f"{name}.jsonl")
        path = out / f"{name}.jsonl"
        done = len(read_jsonl(path)) if path.exists() else 0  # resumable
        with open(path, "a") as f:
            for i in range(done, len(rows)):
                r = rows[i]
                body = json.dumps({"model": a.model, "state": r["state"], "questions": {"q": r["question"]}})
                req = urllib.request.Request(url, data=body.encode(), headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=900) as resp:
                    ans = json.load(resp)["answers"]["q"]
                f.write(json.dumps(answer_row(name, manifest[name]["type"], r["keys"], r["gold"], ans)) + "\n")
                f.flush()
                if i % 50 == 0:
                    print(f"{name}: {i}/{len(rows)}", file=sys.stderr, flush=True)
        print(f"{name}: {len(rows)} rows -> {path}", flush=True)
    print("done", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    sub.add_parser("verify")
    r = sub.add_parser("run")
    r.add_argument("--port", type=int, default=11435)
    r.add_argument("--model", required=True, help="model as the reported runs named it (the file path)")
    r.add_argument("--out", required=True, help="bench/jev/<model>/heldout")
    r.add_argument("sources", nargs="*", help="default: every source in the manifest")
    a = ap.parse_args(argv)
    {"build": cmd_build, "verify": cmd_verify, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()
```

```bash
python3 -m unittest discover -s scripts/jev_eval -p 'test_*.py' 2>&1 | tail -3
```

Expected: `Ran 21 tests`, `OK`.

- [ ] **Step 3: Build the held-out set.** This is CPU and network
  (Hugging Face datasets-server, GitHub raw), a few minutes. It reads the
  eval worktree's saved raw rows for R.

```bash
cd /home/nick/repos/hipfire-jev-calib
git -C ~/repos/jev-evals/jev-bench status --short; git -C ~/repos/jev-evals/jev-ood-calibration status --short   # note the output
python3 scripts/jev_eval/heldout.py build 2>&1 | tee $HOME/.cache/hipfire-jev-calib/build.log
git -C ~/repos/jev-evals/jev-bench status --short; git -C ~/repos/jev-evals/jev-ood-calibration status --short   # must equal the output before
python3 scripts/jev_eval/heldout.py verify
```

Expected, from a CPU dry run on 2026-09-29:

- one line per source, 17 sources;
- `kept 300` everywhere. sentiment-it's validation split has 324 rows. A
  source that keeps fewer than 300 is recorded in the manifest; that is not
  an error;
- `excluded` about 66 for sms-spam, 15 for each synth-* type, 2 for
  massive-it, and 0 elsewhere;
- verify ends `5100 held-out rows in 17 sources, none among N reported
  states`, with N ≈ 10,250.

A transient Hugging Face 5xx is retried, by calsets and by jevbench.

If a jev-bench source reports `label_filtered` > 10% of `drawn`, the
held-out split's label set differs from the reported question's. Report it;
do not change the question.

- [ ] **Step 4: Commit**

```bash
git add scripts/jev_eval/heldout.py scripts/jev_eval/test_heldout.py bench/jev/heldout/manifest.json \
  bench/jev/heldout/reported_state_sha256.txt
git commit -m "feat(jev_eval): held-out calibration set with provably no reported rows

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: GPU runs: held-out answers (eval build) and live check L1 + gates (this branch's build)

**Files:**
- Create: `scripts/jev_eval/calib_live_check.py`
- Create: `bench/jev/qwen3.5-4b/heldout/*.jsonl`,
  `bench/jev/qwen3.8-27b-mq4-xts/heldout/*.jsonl`

**Interfaces:**
- Consumes:
  - `heldout.py run`, which needs a serve on :11435;
  - `daemon_client.Daemon` (this worktree's `target/release/daemon`);
  - `calibrate.apply_t`.
- Produces: held-out answer rows, the live-check verdict, and the gate
  verdicts.

**Why two builds.**

- The held-out answers must come from the build that produced the reported
  rows. That is the PR #768 eval build, `~/repos/hipfire-jev-eval` at
  237bb7bba, whose binaries date from 2026-09-28 17:11. T belongs to that
  build's numerics (spec §13.4).
- The live check and the gates test this branch's code, so they use this
  worktree's build.
- The eval build has no calibration code and is only ever asked for raw
  answers.
- Do not rebuild or edit the eval worktree. If its binaries are missing,
  stop and ask.

- [ ] **Step 1: Create `scripts/jev_eval/calib_live_check.py`**

```python
"""GPU check L1 (spec §13.7): with the repo daemon driven directly, a
calibrated decide equals the offline apply_t of the raw decide, an explicit
T = 1 is bit-identical to no calibration, the timing echoes the temperatures,
and a malformed calibration is a 422. One model, held-out examples only.

Exit 0 PASS, 1 FAIL, 2 INCONCLUSIVE (the raw readout does not repeat bit for
bit, so two runs cannot be compared)."""
import argparse
import json
import sys

sys.dont_write_bytecode = True
import calibrate as cb  # noqa: E402
import heldout  # noqa: E402
from daemon_client import Daemon  # noqa: E402

CAL = {"choice": 1.7, "score": 0.6, "noul": 2.3}
ONE = {"choice": 1.0, "score": 1.0, "noul": 1.0}
TOL = 1e-9


def probs_of(ans, keys):
    return [ans["noul"], 1 - ans["noul"]] if ans["type"] == "noul" else [ans["probabilities"][k] for k in keys]


def check_pair(raw, cal, keys, t):
    """(problems, max |Δp|) for one question's raw and calibrated answers."""
    bad = []
    rp, cp = probs_of(raw, keys), probs_of(cal, keys)
    d = max(abs(x - y) for x, y in zip(cb.apply_t(rp, t), cp))
    if d > TOL:
        bad.append(f"probabilities differ from apply_t(raw) by {d:.3g}")
    if raw["type"] == "choice":
        k = len(cp)
        if cal["choice"] != raw["choice"]:
            bad.append(f"choice moved {raw['choice']} -> {cal['choice']}")
        if abs(cal["confidence"] - (max(cp) - 1 / k) / (1 - 1 / k)) > 1e-12:
            bad.append("choice confidence is not (p_max - 1/K)/(1 - 1/K) of the calibrated p")
    if raw["type"] == "score":
        if abs(cal["score"] - sum(i * x for i, x in enumerate(cp))) > 1e-9:
            bad.append("score is not the mean of the calibrated p")
        if abs(cal["confidence"] - max(cp)) > 1e-12:
            bad.append("score confidence is not the calibrated p_max")
    return bad, d


def run(d, picked):
    fails, worst = [], 0.0
    name, r = picked[0]
    qs = {"q": r["question"]}
    if d.decide(r["state"], qs).get("answers") != d.decide(r["state"], qs).get("answers"):
        print(f"INCONCLUSIVE: a repeated raw decide ({name}) is not bit-identical")
        return None, worst
    for name, r in picked:
        qs = {"q": r["question"]}
        raw = d.decide(r["state"], qs)
        cal = d.decide(r["state"], qs, calibration=CAL)
        one = d.decide(r["state"], qs, calibration=ONE)
        if any("error" in v for v in (raw, cal, one)):
            fails.append(f"{name}: daemon error {[v.get('error') for v in (raw, cal, one)]}")
            continue
        bad, delta = check_pair(raw["answers"]["q"], cal["answers"]["q"], r["keys"], CAL[r["question"]["type"]])
        worst = max(worst, delta)
        fails += [f"{name}: {b}" for b in bad]
        if one["answers"] != raw["answers"]:
            fails.append(f"{name}: explicit T = 1 is not bit-identical to no calibration")
        if cal["timing"].get("calibration") != CAL:
            fails.append(f"{name}: timing.calibration = {cal['timing'].get('calibration')}")
        if "calibration" in raw["timing"] or "calibration" in one["timing"]:
            fails.append(f"{name}: an identity calibration was echoed in the timing")

    # Session mode: cold, extend, calibrated extend (the last two share the split point).
    name, r = next((n, x) for n, x in picked if x["question"]["type"] == "choice")
    text = r["state"] if isinstance(r["state"], str) else json.dumps(r["state"], ensure_ascii=False)
    msgs = [{"role": "user", "content": text}, {"role": "assistant", "content": "Noted."}]
    qs = {"q": r["question"]}
    replies = [d.session_decide(msgs, qs), d.session_decide(msgs, qs), d.session_decide(msgs, qs, calibration=CAL)]
    if any("error" in v for v in replies):
        fails.append(f"session: daemon error {[v.get('error') for v in replies]}")
    else:
        starts = [v["timing"].get("start") for v in replies]
        if starts[1:] != ["extend", "extend"]:
            print(f"INCONCLUSIVE (session part): starts {starts}, expected [cold, extend, extend]")
        else:
            bad, delta = check_pair(replies[1]["answers"]["q"], replies[2]["answers"]["q"], r["keys"], CAL["choice"])
            worst = max(worst, delta)
            fails += [f"session: {b}" for b in bad]

    for bad_cal in ({"choice": 0}, {"bogus": 1.0}, "hot"):
        v = d.decide(r["state"], qs, calibration=bad_cal)
        err = v.get("error") or {}
        if err.get("status") != 422 or "calibration" not in err.get("message", ""):
            fails.append(f"malformed calibration {bad_cal!r}: {v}")
    if "error" in d.decide(r["state"], qs):
        fails.append("a normal decide after the 422s failed")
    return fails, worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--per-source", type=int, default=5)
    ap.add_argument("--max-seq", type=int, default=16384)
    a = ap.parse_args()
    manifest = json.loads((heldout.OUT / "manifest.json").read_text())
    picked = []
    for name in manifest:
        rows = heldout.read_jsonl(heldout.WORK / "examples" / f"{name}.jsonl")
        picked += [(name, r) for r in rows[: a.per_source]]
    d = Daemon()
    try:
        d.load(a.model, max_seq=a.max_seq)
        fails, worst = run(d, picked)
    finally:
        d.close()
    if fails is None:
        return 2
    print(f"L1 live equivalence: {len(picked)} examples, worst |dp| {worst:.3g} (tolerance {TOL})")
    for f in fails:
        print("FAIL", f)
    print("PASS" if not fails else f"FAIL ({len(fails)})")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
```

```bash
python3 -m py_compile scripts/jev_eval/calib_live_check.py && echo ok
```

- [ ] **Step 2: Pre-flight** (every check must hold; otherwise stop and
  ask)

```bash
pgrep -a -x daemon; pgrep -a -x hipfire          # must print nothing (the serve battery must be finished)
grep MemAvailable /proc/meminfo                   # >= 12 GiB now; >= 30 GiB before the 27B (Step 4)
ss -ltn '( sport = :11435 )' | tail -n +2         # must print nothing
git -C /home/nick/repos/hipfire-jev-eval rev-parse --short HEAD    # 237bb7bba
ls -l /home/nick/repos/hipfire-jev-eval/target/release/{hipfire,daemon}
/home/nick/repos/hipfire-jev-eval/target/release/hipfire serve --help | grep -c -- --no-prewarm   # 1
python3 scripts/jev_eval/heldout.py verify
```

- [ ] **Step 3: Held-out answers, Qwen3.5-4B (eval build, serve).** Start
  serve, recording its PID. `--no-prewarm` keeps it from loading the
  configured default model, and the model loads on the first request, as in
  the reported runs.

```bash
export PATH=/opt/rocm-7.2.2/bin:$PATH
EVAL=/home/nick/repos/hipfire-jev-eval; WORK=$HOME/.cache/hipfire-jev-calib; mkdir -p $WORK/logs
cd /home/nick/repos/hipfire-jev-calib
HIPFIRE_DAEMON_BIN=$EVAL/target/release/daemon nohup $EVAL/target/release/hipfire serve --no-prewarm \
  > $WORK/logs/serve-4b.log 2>&1 &
echo $! > $WORK/logs/serve.pid; cat $WORK/logs/serve.pid
```

Then start the runner as a **background** Bash call (`run_in_background:
true`). It waits for `/health`, and it is resumable: rerunning continues
each source file where it stopped.

```bash
cd /home/nick/repos/hipfire-jev-calib && python3 scripts/jev_eval/heldout.py run \
  --model /home/nick/.hipfire/models/qwen3.5-4b.mq4 --out bench/jev/qwen3.5-4b/heldout \
  > $HOME/.cache/hipfire-jev-calib/logs/run-4b.log 2>&1
```

Once rows appear, record the daemon serve spawned, then poll every few
minutes:

```bash
WORK=$HOME/.cache/hipfire-jev-calib
pgrep -P $(cat $WORK/logs/serve.pid) -x daemon | tee $WORK/logs/daemon.pid
tail -n 2 $WORK/logs/run-4b.log; cat bench/jev/qwen3.5-4b/heldout/*.jsonl 2>/dev/null | wc -l
```

Expected: the log ends with `done`, and the row count equals the sum of
`kept` in `bench/jev/heldout/manifest.json` (about 5,100). That takes about
25–35 min.

Stop what you started, and only that:

```bash
WORK=$HOME/.cache/hipfire-jev-calib; S=$(cat $WORK/logs/serve.pid); D=$(cat $WORK/logs/daemon.pid)
kill $S; timeout 120 tail --pid=$S -f /dev/null
if kill -0 $D 2>/dev/null; then kill $D; timeout 60 tail --pid=$D -f /dev/null; fi
pgrep -a -x daemon; pgrep -a -x hipfire   # both print nothing
```

- [ ] **Step 4: Held-out answers, Qwen3.8-27B mq4-xts (eval build,
  serve).** Check `grep MemAvailable /proc/meminfo` first (≥ 30 GiB), then
  run the same sequence as Step 3 with `serve-27b.log`, `run-27b.log`,
  `--model /home/nick/.hipfire/models/qwen3.8-27b.mq4-xts` and
  `--out bench/jev/qwen3.8-27b-mq4-xts/heldout`.

  Expected: `done` and the same row count, in about 70–100 min. The first
  request includes the 27B load. Stop serve and its daemon exactly as in
  Step 3, and confirm with `pgrep`.

- [ ] **Step 5: Build this branch** (CPU; background, since it can exceed
  10 min)

```bash
cd /home/nick/repos/hipfire-jev-calib && cargo build --release -p hipfire-daemon -p hipfire-cli \
  && cargo build --release -p hipfire-generate --features lab --example split_prefill_probe
```

- [ ] **Step 6: Live check L1** (this branch's daemon, 4B, background;
  pre-flight as in Step 2)

```bash
export PATH=/opt/rocm-7.2.2/bin:$PATH
cd /home/nick/repos/hipfire-jev-calib && python3 scripts/jev_eval/calib_live_check.py \
  --model /home/nick/.hipfire/models/qwen3.5-4b.mq4 > $HOME/.cache/hipfire-jev-calib/logs/live.log 2>&1; echo "exit $?" >> $HOME/.cache/hipfire-jev-calib/logs/live.log
```

Expected:

- the log ends with `PASS` and `exit 0`;
- `worst |dp|` is ≤ 1e-9;
- `daemon_client` terminates its daemon, and `pgrep -a -x daemon` prints
  nothing afterwards.

An `INCONCLUSIVE` (exit 2) means the raw readout did not repeat bit for
bit. That is a finding: report it; do not loosen the tolerance.

- [ ] **Step 7: v1/v1.1 gates, raw** (this branch's build, 4B, background)

```bash
export PATH=/opt/rocm-7.2.2/bin:$PATH
cd /home/nick/repos/hipfire-jev-calib && python3 scripts/jev_eval/gates.py \
  --model /home/nick/.hipfire/models/qwen3.5-4b.mq4 > $HOME/.cache/hipfire-jev-calib/logs/gates.log 2>&1; echo "exit $?" >> $HOME/.cache/hipfire-jev-calib/logs/gates.log
```

Expected: `summary: … 0 FAIL`, exit 0, then `pgrep -a -x daemon` prints
nothing.

- [ ] **Step 8: Commit** (text-free rows only)

```bash
grep -l '"state"' bench/jev/*/heldout/*.jsonl   # must print nothing
git add scripts/jev_eval/calib_live_check.py bench/jev/qwen3.5-4b/heldout bench/jev/qwen3.8-27b-mq4-xts/heldout
git commit -m "bench(decide): held-out calibration answers (eval build 237bb7bba) and live check

L1 live equivalence and the v1/v1.1 gates pass on this branch's build
(Qwen3.5-4B); logs in ~/.cache/hipfire-jev-calib/logs/{live,gates}.log.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Fit, calibrated reports, docs

**Files:**
- Create: `bench/jev/qwen3.5-4b/calibration.json`,
  `bench/jev/qwen3.8-27b-mq4-xts/calibration.json`
- Regenerate: `bench/jev/qwen3.5-4b/report.md`,
  `bench/jev/qwen3.8-27b-mq4-xts/report.md`
- Modify: `scripts/jev_eval/README.md`, `bench/jev/README.md`,
  `docs/specs/2026-09-28-jev-decide-design.md` (status line)

- [ ] **Step 1: Fit** (CPU; reads `heldout/` only)

```bash
cd /home/nick/repos/hipfire-jev-calib
python3 scripts/jev_eval/calibrate.py fit --heldout bench/jev/qwen3.5-4b/heldout \
  --model-id qwen3.5-4b.mq4 --build 237bb7bba --out bench/jev/qwen3.5-4b/calibration.json
python3 scripts/jev_eval/calibrate.py fit --heldout bench/jev/qwen3.8-27b-mq4-xts/heldout \
  --model-id qwen3.8-27b.mq4-xts --build 237bb7bba --out bench/jev/qwen3.8-27b-mq4-xts/calibration.json
```

Expected:

- each prints a `models.toml` snippet, or the "no question type passed"
  line;
- `calibration.json` has `choice`, `score` and `noul` entries with `n`
  (≈ 2,700 / 900 / 1,500), `t`, `t_ci95` and `ship`;
- an entry with `error` means the fit hit a bound. Report it as found.

- [ ] **Step 2: Regenerate the reports** (CPU; frozen rows, no GPU)

```bash
python3 scripts/jev_eval/report.py --out bench/jev/qwen3.5-4b
python3 scripts/jev_eval/report.py --out bench/jev/qwen3.8-27b-mq4-xts
python3 scripts/jev_eval/heldout.py verify
```

Expected:

- each report has a "Fit" table and the raw and calibrated columns;
- `hipfire acc` is unchanged from Task 3's reports, because `summarize`
  asserts that no argmax moved;
- verify passes.

- [ ] **Step 3: Docs.** In `scripts/jev_eval/README.md`, append:

~~~markdown
6. Calibration (spec §13; CPU unless noted). Held-out rows never overlap
   the reported rows, and the fit never reads reported rows.
   ```bash
   # once: freeze the reported rows (text-free) from the eval run's raw answers
   python3 scripts/jev_eval/calibrate.py freeze --src ~/repos/hipfire-jev-eval/bench/jev/qwen3.5-4b --out bench/jev/qwen3.5-4b/probs
   # once: build the held-out set (network); text stays in ~/.cache/hipfire-jev-calib
   python3 scripts/jev_eval/heldout.py build && python3 scripts/jev_eval/heldout.py verify
   # GPU: answer it on the build that served the reported rows (serve on :11435, raw)
   python3 scripts/jev_eval/heldout.py run --model ~/.hipfire/models/qwen3.5-4b.mq4 --out bench/jev/qwen3.5-4b/heldout
   # fit per question type, then report raw vs calibrated on the reported rows
   python3 scripts/jev_eval/calibrate.py fit --heldout bench/jev/qwen3.5-4b/heldout --model-id qwen3.5-4b.mq4 --build <sha> --out bench/jev/qwen3.5-4b/calibration.json
   python3 scripts/jev_eval/report.py --out bench/jev/qwen3.5-4b
   ```
   The fit prints the `models.toml` snippet (`decide.calibration.*`; see
   `docs/CONFIG.md`). GPU check that the daemon applies T exactly as the
   fitter does: `python3 scripts/jev_eval/calib_live_check.py --model ~/.hipfire/models/qwen3.5-4b.mq4`.
~~~

In `bench/jev/README.md`, replace the last caveat bullet (the two lines
starting `- hipfire's probabilities are raw model probabilities.`) with:

~~~markdown
- Calibration (spec §13): each report's "Fit" table gives the per-type
  temperatures fitted on held-out rows only (`heldout/`, answered by the
  same eval build; `../heldout/manifest.json` lists the sources and proves
  no overlap). The "ECE cal" / "NLL cal" columns apply them to the saved
  reported answers (`probs/`); nothing was re-run. The raw columns are the
  served probabilities.
~~~

In the spec's status line, change `**DESIGN — v1.2 calibration (§13)**` to
`**IMPLEMENTED — v1.2 calibration (§13)**`:

```bash
sed -i 's/\*\*DESIGN — v1.2 calibration (§13)\*\*/**IMPLEMENTED — v1.2 calibration (§13)**/' docs/specs/2026-09-28-jev-decide-design.md
```

- [ ] **Step 4: Final checks and commit**

```bash
./scripts/no-gpu-ci.sh 2>&1 | tail -3
bash scripts/leanup-ratchets.sh
python3 scripts/check-crate-maps.py --check
git add bench/jev/qwen3.5-4b/calibration.json bench/jev/qwen3.8-27b-mq4-xts/calibration.json \
  bench/jev/qwen3.5-4b/report.md bench/jev/qwen3.8-27b-mq4-xts/report.md \
  scripts/jev_eval/README.md bench/jev/README.md docs/specs/2026-09-28-jev-decide-design.md
git commit -m "docs(decide): fitted calibration and raw-vs-calibrated reports

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 5: Hand back.** Report:

- each model's shipped temperatures (from `calibration.json`) and the
  `models.toml` snippets;
- the raw → calibrated ECE per task;
- the L1 and gate verdicts.

Do **not** write `~/.hipfire/models.toml`. Applying the snippet needs every
hipfire that reads that catalog, including the installed
`~/.hipfire/bin`, to be at or past this branch (spec §13.9). That is the
human's call. Don't push.
