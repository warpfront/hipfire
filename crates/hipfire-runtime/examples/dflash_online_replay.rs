//! Offline replay of `HIPFIRE_DFLASH_ONLINE_DUMP` records through the online
//! draft tuner (developer-only). Block starts and emitted text are the
//! recorded ones, so the reported `picked τ` is the tuner's acceptance at those
//! starts on that text. Replaying a dump of the *same* policy reproduces its
//! online τ exactly; scoring a *different* policy on an argmax session
//! overstates it when it accepts more, because online its later block starts
//! land on harder positions (see the 2026-10-06 perf-checkpoint amendment).
//! Use it to rank ideas, then confirm online.
//! Each dump is one session; the summary covers its last request.
//!
//! `--first N`: score only the first N verify cycles of each session (short
//! requests). `--carry-loo`: each session runs after all the others in one
//! tuner with `carry` on (weights kept across requests, session statistics
//! reset) — what a long-running daemon sees. `--no-prompt`: ignore the
//! recorded prompt (session history starts at the first generated token).
//!
//! Usage: dflash_online_replay [--hp key=val,...] [--first N] [--carry-loo] [--no-prompt] DUMP...

use hipfire_runtime::dflash_online::{Hyper, Mode, OnlineDraftTuner, K};

fn u64_at(b: &[u8], o: &mut usize) -> u64 {
    let v = u64::from_le_bytes(b[*o..*o + 8].try_into().unwrap());
    *o += 8;
    v
}

fn u32s(b: &[u8], o: &mut usize, n: usize) -> Vec<u32> {
    let v = b[*o..*o + 4 * n]
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    *o += 4 * n;
    v
}

/// Feed the records of one dump into `t` (a request boundary calls
/// `t.reset()`), stopping after `max_obs` verify cycles of a request.
fn feed(path: &str, b: &[u8], t: &mut OnlineDraftTuner, max_obs: usize, prompt: bool) {
    let (mut o, mut obs) = (0usize, 0usize);
    let mut prop: Option<(usize, u32, usize, Vec<u32>)> = None;
    while o < b.len() {
        let tag = b[o];
        o += 1;
        match tag {
            b'P' => {
                let start = u64_at(b, &mut o) as usize;
                let seed = u64_at(b, &mut o) as u32;
                let rows = u64_at(b, &mut o) as usize;
                prop = Some((start, seed, rows, u32s(b, &mut o, rows * K)));
            }
            b'L' => {
                let (start, seed, rows, ids) = prop.take().expect("L record without P");
                let vals: Vec<f32> = u32s(b, &mut o, ids.len())
                    .into_iter()
                    .map(f32::from_bits)
                    .collect();
                t.propose(start, seed, &ids, &vals, rows);
            }
            b'O' => {
                let start = u64_at(b, &mut o) as usize;
                let accepted = u64_at(b, &mut o) as usize;
                let n = u64_at(b, &mut o) as usize;
                t.observe(start, &u32s(b, &mut o, n), accepted);
                obs += 1;
                if obs >= max_obs {
                    return;
                }
            }
            b'R' => {
                t.reset();
                obs = 0;
            }
            b'S' => {
                let n = u64_at(b, &mut o) as usize;
                let toks = u32s(b, &mut o, n);
                if prompt {
                    t.seed_prompt(&toks);
                }
            }
            _ => panic!("{path}: bad record tag {tag} at {}", o - 1),
        }
    }
}

/// One request of a `HIPFIRE_DFLASH_ONLINE_TUNE=sweep` dump: the prompt, the
/// draft top-K (and seed) at every block start, and the committed text.
struct Sweep {
    prompt: Vec<u32>,
    recs: std::collections::BTreeMap<usize, (u32, usize, Vec<u32>, Vec<f32>)>,
    text: std::collections::HashMap<usize, u32>,
}

/// Parse the last request of a sweep dump.
fn parse_sweep(path: &str, b: &[u8]) -> Sweep {
    let mut sw = Sweep {
        prompt: Vec::new(),
        recs: Default::default(),
        text: Default::default(),
    };
    let mut o = 0usize;
    let mut prop: Option<(usize, u32, usize, Vec<u32>)> = None;
    while o < b.len() {
        let tag = b[o];
        o += 1;
        match tag {
            b'P' => {
                let start = u64_at(b, &mut o) as usize;
                let seed = u64_at(b, &mut o) as u32;
                let rows = u64_at(b, &mut o) as usize;
                prop = Some((start, seed, rows, u32s(b, &mut o, rows * K)));
            }
            b'L' => {
                let (start, seed, rows, ids) = prop.take().expect("L record without P");
                let vals = u32s(b, &mut o, ids.len())
                    .into_iter()
                    .map(f32::from_bits)
                    .collect();
                sw.recs.insert(start, (seed, rows, ids, vals));
                sw.text.insert(start, seed);
            }
            b'O' => {
                let start = u64_at(b, &mut o) as usize;
                let _ = u64_at(b, &mut o);
                let n = u64_at(b, &mut o) as usize;
                for (i, t) in u32s(b, &mut o, n).into_iter().enumerate() {
                    sw.text.insert(start + 1 + i, t);
                }
            }
            b'R' => {
                sw = Sweep {
                    prompt: Vec::new(),
                    recs: Default::default(),
                    text: Default::default(),
                }
            }
            b'S' => {
                let n = u64_at(b, &mut o) as usize;
                sw.prompt = u32s(b, &mut o, n);
            }
            _ => panic!("{path}: bad record tag {tag} at {}", o - 1),
        }
    }
    sw
}

/// Run a policy over a sweep session with its *own* block starts: accept the
/// longest prefix of its picks matching the text, commit the bonus, start the
/// next block after it. Returns (tokens, windows).
fn simulate(sw: &Sweep, t: &mut OnlineDraftTuner) -> (usize, usize) {
    t.seed_prompt(&sw.prompt);
    let (mut tokens, mut windows) = (0usize, 0usize);
    let Some(mut s) = sw.recs.keys().next().copied() else {
        return (0, 0);
    };
    while let Some((seed, rows, ids, vals)) = sw.recs.get(&s) {
        let picks = t.propose(s, *seed, ids, vals, *rows);
        let acc = picks
            .iter()
            .enumerate()
            .take_while(|(i, p)| sw.text.get(&(s + 1 + i)) == Some(p))
            .count();
        let Some(&bonus) = sw.text.get(&(s + acc + 1)) else {
            break;
        };
        let committed: Vec<u32> = picks[..acc].iter().copied().chain([bonus]).collect();
        t.observe(s, &committed, acc);
        tokens += acc + 1;
        windows += 1;
        s += acc + 1;
    }
    (tokens, windows)
}

fn arg_value(args: &mut Vec<String>, flag: &str) -> Option<String> {
    let i = args.iter().position(|a| a == flag)?;
    let v = args.remove(i + 1);
    args.remove(i);
    Some(v)
}

fn take_flag(args: &mut Vec<String>, flag: &str) -> bool {
    args.iter()
        .position(|a| a == flag)
        .map(|i| args.remove(i))
        .is_some()
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let hyper = Hyper::parse(&arg_value(&mut args, "--hp").unwrap_or_default())
        .unwrap_or_else(|e| panic!("{e}"));
    let first = arg_value(&mut args, "--first").map_or(usize::MAX, |v| v.parse().unwrap());
    let carry_loo = take_flag(&mut args, "--carry-loo");
    if take_flag(&mut args, "--simulate") {
        // Sweep dumps: tokens per window of the tuned policy vs argmax, each
        // with its own block starts, one cold request per session.
        // With --carry-loo, the tuned run first serves every other session in
        // the same tuner (weights carried, as a long-running daemon would).
        let sweeps: Vec<Sweep> = args
            .iter()
            .map(|p| parse_sweep(p, &std::fs::read(p).unwrap_or_else(|e| panic!("{p}: {e}"))))
            .collect();
        let mut ratios = Vec::new();
        for (i, (p, sw)) in args.iter().zip(&sweeps).enumerate() {
            let (at, aw) = simulate(sw, &mut OnlineDraftTuner::new(Mode::Stats, hyper.clone()));
            let mut t = OnlineDraftTuner::new(
                Mode::On,
                Hyper {
                    carry: carry_loo,
                    ..hyper.clone()
                },
            );
            if carry_loo {
                for (_, other) in sweeps.iter().enumerate().filter(|(j, _)| *j != i) {
                    simulate(other, &mut t);
                    t.reset();
                }
            }
            let (bt, bw) = simulate(sw, &mut t);
            let (a, b) = (at as f64 / aw.max(1) as f64, bt as f64 / bw.max(1) as f64);
            println!("{p}: argmax tok/win={a:.3} ({at}/{aw}) tuned tok/win={b:.3} ({bt}/{bw})");
            ratios.push(b / a);
        }
        let geo = (ratios.iter().map(|r| r.ln()).sum::<f64>() / ratios.len().max(1) as f64).exp();
        println!(
            "SIM GEOMEAN tuned/argmax tok/win over {} sessions: {geo:.4} (hp: {hyper:?})",
            ratios.len()
        );
        return;
    }
    let prompt = !take_flag(&mut args, "--no-prompt");
    let data: Vec<Vec<u8>> = args
        .iter()
        .map(|p| std::fs::read(p).unwrap_or_else(|e| panic!("{p}: {e}")))
        .collect();
    let mut ratios = Vec::new();
    for (i, path) in args.iter().enumerate() {
        let mut t = if carry_loo {
            let mut t = OnlineDraftTuner::new(
                Mode::On,
                Hyper {
                    carry: true,
                    ..hyper.clone()
                },
            );
            for (j, p) in args.iter().enumerate().filter(|(j, _)| *j != i) {
                feed(p, &data[j], &mut t, usize::MAX, prompt);
            }
            t.reset();
            t
        } else {
            OnlineDraftTuner::new(Mode::On, hyper.clone())
        };
        feed(path, &data[i], &mut t, first, prompt);
        let (fin, base, pick) = t.summary();
        println!("{path}: cycles={fin} argmax_tau={base:.3} picked_tau={pick:.3}");
        ratios.push(pick / base);
    }
    let geo = (ratios.iter().map(|r| r.ln()).sum::<f64>() / ratios.len().max(1) as f64).exp();
    println!(
        "GEOMEAN picked/argmax tau over {} sessions: {geo:.4} ({}hp: {hyper:?})",
        ratios.len(),
        if carry_loo { "carry-loo, " } else { "" }
    );
}
