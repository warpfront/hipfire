# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Promotion-lane performance rows (imported by lane.py; stdlib only).

A perf row compares the candidate archive with the last promoted beta on the
same card: each fresh process contributes the median of its measured runs,
an arm is the median of its process medians, the row is the ratio of the arms
(new/old for tok/s, old/new for TTFT, so a ratio below 1 is always a loss)
with a bootstrap CI over processes, and a per-class, per-row refusal threshold
1 - max(floor, aa_multiplier x A/A half-width) taken from policy.json
`perf.classes`. A row regresses when the CI upper bound is below it. The
process, not the run, is the unit: runs inside one process are not
independent (gfx1100 27B 5.9k TTFT is ~19% slower on every process's third
request), so resampling runs would turn a fixed pattern into noise.

Every measurement is one fresh `hipfire bench` process with its own HOME,
HIPFIRE_HOME and empty kernel cache, `--warmups 0`: sample 0 is the cold
first request (the cold row), the first `discard` samples are dropped for
the warm rows. Each process records host pressure (PSI `some` stall time over
its lifetime, /proc/pressure cpu/memory/io) and load average; a process over
`measure.max_pressure` is rejected and re-run (bounded), so a thrashing host
can neither fail a candidate nor set a threshold. Baseline numbers come from
the previous promotion record of the gate's own beta when card, host, HIP
runtime and measurement spec match
(`reuse`); otherwise, or to confirm a regression seen against reused numbers,
the baseline archive runs interleaved with the candidate (`pair`).

A relation row (a row with a `reference` proc) judges the candidate alone,
against itself: the median over `proc`'s processes of the metric, over the
median over `reference`'s processes measured in the same job, may exceed 1
by at most the row's `max_excess`. `warm_ttft_over_reference` takes each
process's median warm TTFT, so a speculative load whose later requests wait
for work the previous request left on the device fails against the plain-AR
load on the same prompt. Cold requests stay out of it: a load that calibrates
(runs prefill and speculative windows) before `loaded` has no cold first
request to compare against.
"""
import hashlib
import json
import os
import random
import shutil
import socket
import statistics
import subprocess
import time
from datetime import datetime, timezone

PERF_SCHEMA = "promotion-perf-v1"
# metric -> (bench samples key, better direction, sample window)
METRICS = {
    "decode_tok_s": ("decode", "higher", "warm"),
    "ttft_ms": ("ttft_ms", "lower", "warm"),
    "warm_ttft_over_reference": ("ttft_ms", "lower", "relation"),
    "cold_wall_tok_s": ("wall", "higher", "cold"),
}
PASS = ("pass", "waived")
VERDICTS = ("pass", "regressed", "failed", "skipped", "waived")
SPECS = ("default", "off", "dflash", "mtp", "ngram", "dspark", "auto")
# host pressure (PSI `some`, fraction of a process's wall time): recorded for every resource, gated on PSI_GATED;
# fabric's timing class locks the GPU, not the host CPU or memory, and cold rows are CPU-bound
PSI_DIR = "/proc/pressure"
PSI_RESOURCES = ("cpu", "memory", "io")
PSI_GATED = ("cpu", "memory")
# fabric card selectors survive the scrub; every other HIPFIRE_*/HIP_*/GPU_* knob could move a number
KEEP_ENV = ("HIPFIRE_DEVICES", "HIPFIRE_LOCK_DIR", "HIPFIRE_FABRIC_ADMISSION", "HIP_VISIBLE_DEVICES",
            "ROCR_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL")
SCRUB_PREFIXES = ("HIPFIRE_", "HIP_", "GPU_", "HSA_", "MIOPEN", "ROCR_", "AMD_")
MEASURE_KEYS = ("bench_args", "proc_timeout_s")
JUDGE_KEYS = ("procs_per_arm", "bootstrap_draws", "bootstrap_seed", "ci", "floor", "aa_multiplier",
              "baseline_max_age_days")


class PerfError(Exception):
    """Infrastructure problem (fixture, prompt, policy): the caller maps it to exit 2."""


def _is_num(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _utc_now():
    return datetime.now(timezone.utc)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_cached(path, cache):
    """Same cache layout as device_mesh_matrix.py's .digest-cache.json."""
    st = os.stat(path)
    key = "%s|%d|%d" % (path, st.st_size, int(st.st_mtime))
    hit = cache.get(path)
    if isinstance(hit, dict) and hit.get("key") == key:
        return hit["sha256"]
    cache[path] = {"key": key, "sha256": _sha256(path)}
    return cache[path]["sha256"]


# --------------------------------------------------------------------------- policy

def validate(perf, classes):
    """Problems in policy.json `perf` against the policy's class names."""
    if not isinstance(perf, dict):
        return ["policy: perf must be an object"]
    problems = []
    measure, judge = perf.get("measure"), perf.get("judge")
    if not isinstance(measure, dict) or not isinstance(measure.get("bench_args"), list) \
            or not all(isinstance(a, str) for a in measure["bench_args"]) \
            or not _is_num(measure.get("proc_timeout_s")) or measure["proc_timeout_s"] <= 0:
        problems.append("policy: perf.measure needs bench_args (list of strings) and proc_timeout_s > 0")
    elif not (isinstance(measure.get("max_pressure"), dict) and set(measure["max_pressure"]) == set(PSI_GATED)
              and all(_is_num(v) and 0 < v < 1 for v in measure["max_pressure"].values())
              and isinstance(measure.get("pressure_retries"), int) and measure["pressure_retries"] >= 0
              and _is_num(measure.get("pressure_wait_s")) and measure["pressure_wait_s"] >= 0):
        problems.append("policy: perf.measure needs max_pressure {%s: fraction in (0,1)}, pressure_retries >= 0 "
                        "and pressure_wait_s >= 0" % ", ".join(PSI_GATED))
    if not isinstance(judge, dict) or any(not _is_num(judge.get(k)) for k in JUDGE_KEYS):
        problems.append("policy: perf.judge needs numeric %s" % ", ".join(JUDGE_KEYS))
    elif not (judge["procs_per_arm"] >= 2 and judge["bootstrap_draws"] >= 500 and 0.5 < judge["ci"] < 1
              and 0 < judge["floor"] < 0.5 and judge["aa_multiplier"] >= 1 and judge["baseline_max_age_days"] > 0):
        problems.append("policy: perf.judge out of range (procs_per_arm>=2, bootstrap_draws>=500, 0.5<ci<1, "
                        "0<floor<0.5, aa_multiplier>=1, baseline_max_age_days>0)")
    fixtures, prompts = perf.get("fixtures"), perf.get("prompts")
    if not isinstance(fixtures, dict) or not all(
            isinstance(v, dict) and isinstance(v.get("sha256"), str) and len(v["sha256"]) == 64
            and isinstance(v.get("size_bytes"), int) for v in fixtures.values()):
        problems.append("policy: perf.fixtures must map file name -> {size_bytes, sha256}")
        fixtures = {}
    if not isinstance(prompts, dict) or not all(
            isinstance(v, dict) and isinstance(v.get("file"), str) and isinstance(v.get("sha256"), str)
            for v in prompts.values()):
        problems.append("policy: perf.prompts must map id -> {file, sha256}")
        prompts = {}
    procs = perf.get("procs") if isinstance(perf.get("procs"), dict) else {}
    if not procs:
        problems.append("policy: perf.procs must be a non-empty object")
    for pid, proc in procs.items():
        if not isinstance(proc, dict):
            problems.append("policy: perf proc %s must be an object" % pid)
            continue
        if bool(proc.get("model")) == bool(proc.get("model_file")):
            problems.append("policy: perf proc %s needs exactly one of model (registry tag) or model_file" % pid)
        if proc.get("spec") not in SPECS:
            problems.append("policy: perf proc %s spec must be one of %s" % (pid, "/".join(SPECS)))
        if proc.get("prompt") not in prompts:
            problems.append("policy: perf proc %s prompt %r is not in perf.prompts" % (pid, proc.get("prompt")))
        files = proc.get("files")
        if not isinstance(files, list) or not files or any(f not in fixtures for f in files):
            problems.append("policy: perf proc %s files must be a non-empty list of perf.fixtures names" % pid)
        if not (isinstance(proc.get("runs"), int) and isinstance(proc.get("discard"), int)
                and isinstance(proc.get("max_tokens"), int) and proc.get("max_tokens", 0) > 0
                and 1 <= proc.get("discard", 0) < proc.get("runs", 0) - 1):
            problems.append("policy: perf proc %s needs ints max_tokens>0, 1<=discard<runs-1" % pid)
        own = proc.get("max_pressure", {})
        if not (isinstance(own, dict) and set(own) <= set(PSI_GATED)
                and all(_is_num(v) and 0 < v < 1 for v in own.values())):
            problems.append("policy: perf proc %s max_pressure must map %s to a fraction in (0,1)"
                            % (pid, "/".join(PSI_GATED)))
    rows = perf.get("rows") if isinstance(perf.get("rows"), dict) else {}
    if not rows:
        problems.append("policy: perf.rows must be a non-empty object")
    for rid, row in rows.items():
        if not isinstance(row, dict) or row.get("proc") not in procs or row.get("metric") not in METRICS:
            problems.append("policy: perf row %s needs proc (a perf.procs id) and metric (%s)"
                            % (rid, "/".join(METRICS)))
        elif is_relation(row) != (METRICS[row["metric"]][2] == "relation"):
            problems.append("policy: perf row %s: metric %s %s a reference proc" % (
                rid, row["metric"], "needs" if METRICS[row["metric"]][2] == "relation" else "takes no"))
        elif is_relation(row) and (row["reference"] not in procs or row["reference"] == row["proc"]):
            problems.append("policy: perf relation row %s needs reference (another perf.procs id)" % rid)
    pclasses = perf.get("classes") if isinstance(perf.get("classes"), dict) else None
    if pclasses is None:
        problems.append("policy: perf.classes must be an object")
        pclasses = {}
    for name, spec in pclasses.items():
        if name not in classes:
            problems.append("policy: perf class %s is not a policy class" % name)
        crow = spec.get("rows") if isinstance(spec, dict) else None
        if not isinstance(crow, dict) or not crow:
            problems.append("policy: perf class %s rows must be a non-empty object" % name)
            continue
        for rid, cal in crow.items():
            if rid not in rows:
                problems.append("policy: perf class %s row %s is not in perf.rows" % (name, rid))
            elif is_relation(rows[rid]):
                if not isinstance(cal, dict) or not _is_num(cal.get("max_excess")) or not 0 < cal["max_excess"] < 1:
                    problems.append("policy: perf class %s relation row %s needs max_excess in (0, 1)"
                                    % (name, rid))
                continue
            if not isinstance(cal, dict) or not _is_num(cal.get("aa_half_width")) or not 0 <= cal["aa_half_width"] < 0.5:
                problems.append("policy: perf class %s row %s needs a calibrated aa_half_width in [0, 0.5)"
                                % (name, rid))
    return problems


def check_files(perf, root):
    """Committed prompt files exist under `root` with their pinned sha256."""
    problems = []
    for pid, prompt in (perf.get("prompts") or {}).items():
        path = os.path.join(root, prompt["file"])
        if not os.path.isfile(path):
            problems.append("perf prompt %s: %s missing" % (pid, prompt["file"]))
        elif _sha256(path) != prompt["sha256"]:
            problems.append("perf prompt %s: %s sha256 differs from policy" % (pid, prompt["file"]))
    return problems


def spec_sha256(perf):
    """Identity of everything that shapes one process's samples. Judging
    parameters (thresholds, draws, process count) are not part of it, so a
    recalibration does not invalidate reusable baseline numbers."""
    measured = {"measure": {k: perf["measure"][k] for k in MEASURE_KEYS}, "procs": perf["procs"],
                "prompts": perf["prompts"], "fixtures": perf["fixtures"]}
    return hashlib.sha256(json.dumps(measured, sort_keys=True).encode()).hexdigest()


def class_rows(perf, cls):
    """[(row id, row def, calibration)] the class requires; [] when it has none."""
    spec = (perf or {}).get("classes", {}).get(cls)
    if not spec:
        return []
    return [(rid, perf["rows"][rid], cal) for rid, cal in spec["rows"].items()]


def is_relation(row):
    """A relation row judges the candidate against its own `reference` proc, not against a baseline."""
    return isinstance(row, dict) and "reference" in row


def row_procs(row):
    """The perf.procs ids a row needs measured."""
    return [row["proc"], row["reference"]] if is_relation(row) else [row["proc"]]


def threshold(perf, cal):
    judge = perf["judge"]
    return round(1.0 - max(judge["floor"], judge["aa_multiplier"] * cal["aa_half_width"]), 6)


# --------------------------------------------------------------------------- statistics

def median(values):
    return statistics.median(values)


def ratio_of(old, new, better):
    return new / old if better == "higher" else old / new


def bootstrap(old, new, better, draws, seed, ci):
    """(point, lo, hi): ratio of the arms' medians of per-process medians and
    its bootstrap CI. old/new are lists of per-process sample lists; each draw
    resamples the processes of both arms."""
    rng = random.Random(seed)
    old_m, new_m = [median(p) for p in old], [median(p) for p in new]

    def resampled(arm):
        return median([arm[rng.randrange(len(arm))] for _ in range(len(arm))])

    point = ratio_of(median(old_m), median(new_m), better)
    stats = sorted(ratio_of(resampled(old_m), resampled(new_m), better) for _ in range(int(draws)))
    tail = (1.0 - ci) / 2.0
    lo = stats[max(0, int(tail * len(stats)))]
    hi = stats[min(len(stats) - 1, int((1.0 - tail) * len(stats)))]
    return round(point, 5), round(lo, 5), round(hi, 5)


def values_of(entry, metric, proc):
    """The metric's samples from one process entry (None when unusable)."""
    key, _, window = METRICS[metric]
    samples = (entry.get("samples") or {}).get(key) if entry.get("rc") == 0 and not entry.get("rejected") else None
    if not isinstance(samples, list) or len(samples) < proc["runs"] or not all(_is_num(v) and v > 0 for v in samples):
        return None
    if window == "relation":
        return [median(samples[proc["discard"]:])]
    return [samples[0]] if window == "cold" else samples[proc["discard"]:]


def judge_relation(perf, rid, row, cal, new_entries, ref_entries):
    """A relation row over the candidate's own processes: the median over `proc`'s processes of the
    metric over the median over `reference`'s may exceed 1 by at most the class's max_excess."""
    proc, ref = perf["procs"][row["proc"]], perf["procs"][row["reference"]]
    out = {"id": rid, "required": True, "metric": row["metric"], "better": "lower", "window": "relation",
           "proc": row["proc"], "reference": row["reference"], "mode": "candidate",
           "max_excess": cal["max_excess"], "threshold": None, "ratio": None, "ci": None,
           "reference_value": None, "n_new": len(new_entries), "n_reference": len(ref_entries)}
    pressured = [e.get("label", "?") for e in new_entries + ref_entries if e.get("rejected")]
    if pressured:
        return dict(out, verdict="skipped", note="host pressure: no clean sample after retries (%s)"
                    % ", ".join(pressured))
    new = [values_of(e, row["metric"], proc) for e in new_entries]
    refs = [values_of(e, row["metric"], ref) for e in ref_entries]
    bad = [e.get("label", "?") for e, v in zip(new_entries + ref_entries, new + refs) if v is None]
    if not new_entries or not ref_entries or bad:
        return dict(out, verdict="failed", note="candidate process failed or short: %s"
                    % (", ".join(bad) or "no %s process" % (row["proc"] if not new_entries else row["reference"])))
    value, ref_value = median([v[0] for v in new]), median([v[0] for v in refs])
    ratio, limit = round(value / ref_value, 5), round(1.0 + cal["max_excess"], 5)
    note = "%s %s %.2f / %s %.2f = %.4f, limit 1 + %.4f = %.4f (per process %s vs %s)" % (
        row["metric"], row["proc"], value, row["reference"], ref_value, ratio, cal["max_excess"], limit,
        ", ".join("%.2f" % v[0] for v in new), ", ".join("%.2f" % v[0] for v in refs))
    return dict(out, ratio=ratio, value=round(value, 3), reference_value=round(ref_value, 3), threshold=limit,
                verdict="pass" if ratio <= limit else "regressed", note=note)


def judge_row(perf, rid, row, cal, old_entries, new_entries, mode):
    if is_relation(row):
        return judge_relation(perf, rid, row, cal, new_entries, [])
    proc = perf["procs"][row["proc"]]
    _, better, window = METRICS[row["metric"]]
    thr = threshold(perf, cal)
    out = {"id": rid, "required": True, "metric": row["metric"], "better": better, "window": window,
           "proc": row["proc"], "mode": mode, "threshold": thr, "aa_half_width": cal["aa_half_width"],
           "ratio": None, "ci": None, "old_median": None, "new_median": None,
           "n_old": len(old_entries), "n_new": len(new_entries)}
    new = [values_of(e, row["metric"], proc) for e in new_entries]
    old = [values_of(e, row["metric"], proc) for e in old_entries]
    bad_new = [e.get("label", "?") for e, v in zip(new_entries, new) if v is None]
    pressured = [e.get("label", "?") for e in new_entries + old_entries if e.get("rejected")]
    if pressured:
        return dict(out, verdict="skipped", note="host pressure: no clean sample after retries (%s)"
                    % ", ".join(pressured))
    if not new_entries or bad_new:
        return dict(out, verdict="failed", note="candidate process failed or short: %s"
                    % (", ".join(bad_new) or "no candidate process"))
    bad_old = [e.get("label", "?") for e, v in zip(old_entries, old) if v is None]
    if len(old_entries) < 2 or bad_old:
        return dict(out, verdict="skipped", note="no usable baseline (%s)"
                    % (", ".join(bad_old) or "%d baseline processes" % len(old_entries)))
    seed = "%s:%s" % (perf["judge"]["bootstrap_seed"], rid)
    point, lo, hi = bootstrap(old, new, better, perf["judge"]["bootstrap_draws"], seed, perf["judge"]["ci"])
    taus = {arm: [t for e in entries for t in (e.get("tau") or []) if _is_num(t)]
            for arm, entries in (("old", old_entries), ("new", new_entries))}
    note = "%s %s/%s %.4f CI [%.4f, %.4f] vs threshold %.4f" % (
        row["metric"], "new" if better == "higher" else "old", "old" if better == "higher" else "new",
        point, lo, hi, thr)
    if taus["old"] and taus["new"]:
        note += "; tau old %.2f new %.2f" % (median(taus["old"]), median(taus["new"]))
    toks = {arm: sorted({e.get("prompt_tokens") for e in entries}) for arm, entries
            in (("old", old_entries), ("new", new_entries))}
    if toks["old"] != toks["new"]:
        note += "; prompt_tokens old %s new %s" % (toks["old"], toks["new"])
    return dict(out, ratio=point, ci=[lo, hi], verdict="regressed" if hi < thr else "pass",
                old_median=round(median([median(p) for p in old]), 3),
                new_median=round(median([median(p) for p in new]), 3), note=note)


def row_problem(perf, row, cal):
    """A record perf row whose verdict its own numbers contradict (or that
    another threshold judged) -> the problem text; None when sound."""
    verdict = row.get("verdict")
    if verdict not in VERDICTS:
        return "verdict %r" % (verdict,)
    if is_relation((perf.get("rows") or {}).get(row.get("id"))):
        ref, ratio, thr = row.get("reference_value"), row.get("ratio"), row.get("threshold")
        if row.get("max_excess") != cal.get("max_excess"):
            return "max_excess %r but policy says %r" % (row.get("max_excess"), cal.get("max_excess"))
        if verdict == "pass" and not (_is_num(ref) and _is_num(ratio) and _is_num(thr)
                                      and thr == round(1.0 + cal["max_excess"], 5) and ratio <= thr):
            return "verdict pass but ratio %r is over 1 + max_excess" % (ratio,)
        return None
    if row.get("threshold") != threshold(perf, cal):
        return "threshold %r but policy says %r" % (row.get("threshold"), threshold(perf, cal))
    ci = row.get("ci")
    if verdict == "pass" and not (isinstance(ci, list) and len(ci) == 2 and all(_is_num(c) for c in ci)
                                  and ci[1] >= row["threshold"]):
        return "verdict pass but CI %r is below threshold %r" % (ci, row.get("threshold"))
    return None


# --------------------------------------------------------------------------- processes

def proc_env(base, install, workdir, models_dir):
    env = {k: v for k, v in base.items() if not k.startswith(SCRUB_PREFIXES) or k in KEEP_ENV}
    home, cache, tmp = (os.path.join(workdir, d) for d in ("home", "kernel-cache", "tmp"))
    for path in (home, cache, tmp):
        os.makedirs(path)
    env.update({"HOME": home, "HIPFIRE_HOME": os.path.join(home, ".hipfire"), "HIPFIRE_KERNEL_CACHE": cache,
                "TMPDIR": tmp, "HIPFIRE_NO_DEVICE_COMPILER": "1", "HIPFIRE_NO_REGISTRY_FETCH": "1",
                "HIPFIRE_CLI_BIN": os.path.join(install, "bin", "hipfire"),
                "HIPFIRE_DAEMON_BIN": os.path.join(install, "bin", "daemon"), "HIPFIRE_MODELS_DIR": models_dir})
    return env


def bench_argv(perf, proc, cli, prompts_root, models_dir):
    model = proc.get("model") or os.path.join(models_dir, proc["model_file"])
    spec = [] if proc["spec"] == "default" else ["--spec", proc["spec"]]
    return ([cli, "bench", model] + spec +
            ["--prompt-file", os.path.join(prompts_root, perf["prompts"][proc["prompt"]]["file"]),
             "--runs", str(proc["runs"]), "--warmups", "0", "--max-tokens", str(proc["max_tokens"]),
             "--min-tokens", str(proc["max_tokens"])] + list(perf["measure"]["bench_args"]))


def run_proc(perf, pid, arm, rep, install, base_env, prompts_root, models_dir, workroot, attempt=0):
    """One fresh bench process -> an entry with its samples (rc != 0 on failure)."""
    proc = perf["procs"][pid]
    label = "%s-%s-r%d" % (arm, pid, rep) + ("-a%d" % attempt if attempt else "")
    work = os.path.join(workroot, label)
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    env = proc_env(base_env, install, work, models_dir)
    argv = bench_argv(perf, proc, env["HIPFIRE_CLI_BIN"], prompts_root, models_dir)
    entry = {"label": label, "arm": arm, "proc": pid, "rep": rep, "argv": argv[1:],
             "started_utc": _utc_now().isoformat(), "loadavg_before": _loadavg()}
    psi_before = psi_totals()
    start = time.time()
    with open(os.path.join(workroot, label + ".json"), "w") as out, \
            open(os.path.join(workroot, label + ".err"), "w") as err:
        try:
            proc_h = subprocess.Popen(argv, env=env, stdout=out, stderr=err, start_new_session=True)
            rc = proc_h.wait(timeout=perf["measure"]["proc_timeout_s"])
        except subprocess.TimeoutExpired:
            os.killpg(proc_h.pid, 9)
            proc_h.wait()
            rc = 124
        except OSError as exc:
            err.write("spawn: %s\n" % exc)
            rc = 127
    elapsed = time.time() - start
    entry.update({"rc": rc, "elapsed_s": round(elapsed, 1), "loadavg_after": _loadavg(),
                  "pressure": psi_fractions(psi_before, psi_totals(), elapsed)})
    limits = proc_limits(perf, pid)
    over = {k: v for k, v in (entry["pressure"] or {}).items() if k in PSI_GATED and v > limits[k]}
    if over:
        entry["rejected"] = "host pressure %s (max %s)" % (
            ", ".join("%s %.3f" % kv for kv in sorted(over.items())),
            ", ".join("%s %.3f" % (k, limits[k]) for k in sorted(over)))
    try:
        doc = json.load(open(os.path.join(workroot, label + ".json")))
    except (OSError, ValueError):
        doc = None
        if rc == 0:
            entry["rc"] = 65
    if isinstance(doc, dict):
        device = doc.get("power_device") if isinstance(doc.get("power_device"), dict) else {}
        gpu = doc.get("gpu") if isinstance(doc.get("gpu"), dict) else {}
        entry.update({"samples": {k: (doc.get("samples") or {}).get(k) for k in ("decode", "ttft_ms", "wall")},
                      "tau": doc.get("spec_tau"), "spec_route": doc.get("spec_route"),
                      "prompt_tokens": doc.get("prompt_tokens"), "prompt_md5": doc.get("prompt_md5"),
                      "pci": device.get("pci"), "gpu_arch": gpu.get("arch") or device.get("arch"),
                      "hip_version": gpu.get("hip_version")})
    shutil.rmtree(work, ignore_errors=True)
    return entry


def _loadavg():
    try:
        return open("/proc/loadavg").read().split()[0]
    except OSError:
        return None


def psi_totals():
    """{resource: cumulative `some` stall microseconds} from /proc/pressure; None without PSI."""
    out = {}
    for res in PSI_RESOURCES:
        try:
            line = open(os.path.join(PSI_DIR, res)).readline()
        except OSError:
            return None
        fields = dict(f.split("=", 1) for f in line.split()[1:] if "=" in f)
        try:
            out[res] = int(fields["total"])
        except (KeyError, ValueError):
            return None
    return out


def psi_avg10():
    """{resource: `some` avg10 as a fraction}; None without PSI."""
    out = {}
    for res in PSI_RESOURCES:
        try:
            fields = dict(f.split("=", 1) for f in open(os.path.join(PSI_DIR, res)).readline().split()[1:])
            out[res] = float(fields["avg10"]) / 100.0
        except (OSError, KeyError, ValueError):
            return None
    return out


def psi_fractions(before, after, elapsed):
    """Fraction of the process's wall time during which some task stalled on each resource."""
    if before is None or after is None or elapsed <= 0:
        return None
    return {res: round(max(0, after[res] - before[res]) / (elapsed * 1e6), 4) for res in PSI_RESOURCES}


def proc_limits(perf, pid):
    """Pressure limits of one process kind: measure.max_pressure, overridden by the proc's own (a model
    whose host-mapped weights reclaim page cache at load, e.g. Flash-Next, stalls on memory by itself)."""
    return dict(perf["measure"]["max_pressure"], **perf["procs"][pid].get("max_pressure", {}))


def wait_quiet(perf, log):
    """Before a process: wait (bounded) until the host's 10 s pressure averages are under the limits."""
    limits, deadline = perf["measure"]["max_pressure"], time.time() + perf["measure"]["pressure_wait_s"]
    while True:
        now = psi_avg10()
        loud = {k: now[k] for k in PSI_GATED if now and now[k] > limits[k]}
        if not loud or time.time() >= deadline:
            return
        log("  perf: host pressure %s; waiting" % ", ".join("%s %.3f" % kv for kv in sorted(loud.items())))
        time.sleep(5)


def verify_fixtures(perf, procs, models_dir, digests):
    """Every fixture file the selected processes read matches its lock -> {file: sha256}."""
    seen = {}
    for pid in procs:
        for name in perf["procs"][pid]["files"]:
            if name in seen:
                continue
            path = os.path.join(models_dir, name)
            lock = perf["fixtures"][name]
            if not os.path.isfile(path):
                raise PerfError("perf fixture %s missing under %s" % (name, models_dir))
            if os.path.getsize(path) != lock["size_bytes"]:
                raise PerfError("perf fixture %s is %d bytes, lock says %d" % (name, os.path.getsize(path),
                                                                                lock["size_bytes"]))
            digest = sha256_cached(path, digests)
            if digest != lock["sha256"]:
                raise PerfError("perf fixture %s sha256 %s, lock says %s" % (name, digest, lock["sha256"]))
            seen[name] = digest
    return seen


# --------------------------------------------------------------------------- baseline reuse

def card_identity(env):
    """The fabric's card selector (GPU UUID); None outside a fabric job."""
    for key in ("HIPFIRE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        if env.get(key):
            return env[key]
    return None


def baseline_section(doc):
    """A promotion record's perf section, or a standalone perf document."""
    if isinstance(doc, dict) and isinstance(doc.get("perf"), dict):
        doc = doc["perf"]
    return doc if isinstance(doc, dict) and doc.get("schema") == PERF_SCHEMA else None


def reuse_problem(perf, base, cls, host, card, now, expect_sha, needed):
    """Why the baseline section cannot stand in for a fresh baseline arm; None when it can."""
    if base is None:
        return "no %s section" % PERF_SCHEMA
    fp = base.get("fingerprint") or {}
    if expect_sha and base.get("candidate") != expect_sha:
        return "baseline numbers are for %s, the gate is %s" % (str(base.get("candidate"))[:12], expect_sha[:12])
    for key, want in (("class", cls), ("host", host), ("card", card), ("spec_sha256", spec_sha256(perf))):
        if want is None or fp.get(key) != want:
            return "%s differs (%r, now %r)" % (key, fp.get(key), want)
    try:
        measured = datetime.fromisoformat(base["finished_utc"])
    except (KeyError, TypeError, ValueError):
        return "baseline finish time unreadable"
    if measured.tzinfo is None:
        measured = measured.replace(tzinfo=timezone.utc)
    age = (now - measured).total_seconds() / 86400.0
    if age > perf["judge"]["baseline_max_age_days"]:
        return "baseline is %.1f days old (max %s)" % (age, perf["judge"]["baseline_max_age_days"])
    procs = base.get("procs") or {}
    for pid in needed:
        good = [e for e in procs.get(pid, {}).get("new", []) if e.get("rc") == 0 and not e.get("rejected")]
        if len(good) < 2:
            return "baseline has %d usable %s processes" % (len(good), pid)
    return None


def runtime_mismatch(base, new_entries):
    """HIP runtime / PCI slot seen by the candidate processes vs the reused ones."""
    def seen(entries, key):
        return {e.get(key) for e in entries if e.get("rc") == 0 and e.get(key)}
    old = [e for arm in (base.get("procs") or {}).values() for e in arm.get("new", [])]
    for key in ("hip_version", "pci", "gpu_arch"):
        a, b = seen(old, key), seen(new_entries, key)
        if a and b and a != b:
            return "%s %s vs %s" % (key, sorted(a), sorted(b))
    return None


# --------------------------------------------------------------------------- driver

def run(perf, cls, install, models_dir, prompts_root, out, base_env, baseline_install=None,
        baseline_doc=None, expect_sha=None, digests=None, procs_per_arm=None, row_filter=None,
        candidate=None, host=None, log=print):
    """Measure and judge the class's perf rows -> the record's perf section."""
    rows = [r for r in class_rows(perf, cls) if row_filter is None or r[0] in row_filter]
    if not rows:
        return None
    started = _utc_now()
    host = host or socket.gethostname()
    card = card_identity(base_env)
    n = int(procs_per_arm or perf["judge"]["procs_per_arm"])
    needed = sorted({pid for _, row, _ in rows for pid in row_procs(row)})
    workroot = os.path.join(out, "perf")
    shutil.rmtree(workroot, ignore_errors=True)
    os.makedirs(workroot)
    digests = {} if digests is None else digests
    fixtures = verify_fixtures(perf, needed, models_dir, digests)
    prompt_problems = check_files(perf, prompts_root)
    if prompt_problems:
        raise PerfError("; ".join(prompt_problems))

    base = baseline_section(baseline_doc) if baseline_doc is not None else None
    why_not = reuse_problem(perf, base, cls, host, card, started, expect_sha, needed) \
        if baseline_doc is not None else "no baseline record given"
    procs = {pid: {"new": [], "old": []} for pid in needed}

    def measure(pids, arms_per_rep, tag):
        # grouped by process kind (each model loads into a page cache the previous kind did not evict),
        # arms interleaved and alternated per rep inside the group
        for pid in pids:
            for rep in range(n):
                order = arms_per_rep if rep % 2 == 0 else list(reversed(arms_per_rep))
                for arm in order:
                    inst = install if arm == "new" else baseline_install
                    retries = perf["measure"]["pressure_retries"]
                    for attempt in range(retries + 1):
                        wait_quiet(perf, log)
                        entry = run_proc(perf, pid, arm, rep, inst, base_env, prompts_root, models_dir,
                                         os.path.join(workroot, tag), attempt)
                        log("  perf %s %s rc=%s %.0fs%s" % (tag, entry["label"], entry["rc"], entry["elapsed_s"],
                                                           " REJECTED " + entry["rejected"]
                                                           if entry.get("rejected") else ""))
                        if not entry.get("rejected") or attempt == retries:
                            break
                        procs[pid].setdefault("rejected", []).append(entry)
                    procs[pid][arm].append(entry)

    for tag in ("reuse", "pair"):
        os.makedirs(os.path.join(workroot, tag))
    mode = None
    if why_not is None:
        measure(needed, ["new"], "reuse")
        why_not = runtime_mismatch(base, [e for p in procs.values() for e in p["new"]])
        if why_not is None:
            mode = "reuse"
            for pid in needed:
                procs[pid]["old"] = [dict(e, arm="old") for e in base["procs"][pid]["new"]
                                     if e.get("rc") == 0 and not e.get("rejected")]
        else:
            for pid in needed:
                procs[pid]["new"] = []
    if mode is None:
        if baseline_install is None:
            mode = "none"
        else:
            mode = "pair"
            measure(needed, ["old", "new"], "pair")

    results = []
    for rid, row, cal in rows:
        p = procs[row["proc"]]
        if is_relation(row):
            # judged on the candidate's own processes: no baseline needed
            results.append(judge_relation(perf, rid, row, cal, p["new"], procs[row["reference"]]["new"]))
            continue
        if mode == "none":
            results.append(judge_row(perf, rid, row, cal, [], p["new"], mode))
            results[-1].update(verdict="skipped", note="no reusable baseline (%s) and no baseline archive" % why_not)
            continue
        results.append(judge_row(perf, rid, row, cal, p["old"], p["new"], mode))

    confirm = sorted({r["proc"] for r in results if r["verdict"] == "regressed" and r["mode"] == "reuse"})
    if confirm and baseline_install is not None:
        log("  perf: %s regressed against reused baseline numbers; confirming with the baseline archive"
            % ", ".join(r["id"] for r in results if r["verdict"] == "regressed"))
        reused = {pid: procs[pid] for pid in confirm}
        for pid in confirm:
            procs[pid] = {"new": [], "old": [], "reused": reused[pid]}
        measure(confirm, ["old", "new"], "pair")
        for i, (rid, row, cal) in enumerate(rows):
            if row["proc"] in confirm and not is_relation(row):
                p = procs[row["proc"]]
                results[i] = judge_row(perf, rid, row, cal, p["old"], p["new"], "pair-confirm")
        mode = "reuse+confirm"

    finished = _utc_now()
    new_ok = [e for p in procs.values() for e in p["new"] if e.get("rc") == 0]
    return {"schema": PERF_SCHEMA, "candidate": candidate, "class": cls, "mode": mode,
            "reuse_refused": why_not if mode != "reuse" else None,
            "fingerprint": {"class": cls, "host": host, "card": card, "spec_sha256": spec_sha256(perf),
                            "hip_version": sorted({e.get("hip_version") for e in new_ok if e.get("hip_version")}),
                            "pci": sorted({e.get("pci") for e in new_ok if e.get("pci")})},
            "baseline": {"record_candidate": (base or {}).get("candidate"),
                         "record_finished_utc": (base or {}).get("finished_utc"),
                         "archive": bool(baseline_install)},
            "fixtures": fixtures, "procs_per_arm": n, "procs": procs, "rows": results,
            "started_utc": started.isoformat(), "finished_utc": finished.isoformat(),
            "wall_s": round((finished - started).total_seconds(), 1)}


def print_rows(section, out=print):
    for row in section.get("rows", []):
        if row.get("ratio") is None or row.get("ci") is None:
            out("  %s: %s (%s) %s" % (row["id"], row["verdict"], row["mode"], row.get("note", "")))
        else:
            out("  %s: %s ratio %.4f CI [%.4f, %.4f] threshold %.4f (%s)%s"
                % (row["id"], row["verdict"], row["ratio"], row["ci"][0], row["ci"][1], row["threshold"],
                   row["mode"], "; " + row["note"].split("; ", 1)[1] if "; " in row.get("note", "") else ""))


def calibrate(perf, docs, cls):
    """A/A analysis of perf documents that ran one archive against itself.
    Per row: every document's interleaved pair (old vs new arm), and every
    later document's candidate arm against an earlier one's (the reuse
    protocol: baseline numbers from another job). The half-width of a trial
    is max(1 - lo, hi - 1) of its CI; the row's aa_half_width is the largest."""
    out = {}
    for rid, row in perf["rows"].items():
        if is_relation(row):
            continue  # judged against the candidate itself: nothing to calibrate
        pid = row["proc"]
        arms = [(d.get("label", "doc%d" % i), (d.get("procs") or {}).get(pid)) for i, d in enumerate(docs)]
        arms = [(label, p) for label, p in arms if p and p.get("new")]
        if not arms:
            continue
        cal = {"aa_half_width": 0.0}
        trials = [("pair " + label, p.get("old") or [], p["new"]) for label, p in arms]
        trials += [("cross %s->%s" % (arms[i][0], arms[j][0]), arms[i][1]["new"], arms[j][1]["new"])
                   for i in range(len(arms)) for j in range(i + 1, len(arms))]
        results = []
        for label, old, new in trials:
            judged = judge_row(perf, rid, row, cal, old, new, label)
            if judged["ci"] is None:
                results.append({"trial": label, "verdict": judged["verdict"], "note": judged["note"]})
                continue
            lo, hi = judged["ci"]
            results.append({"trial": label, "ratio": judged["ratio"], "ci": judged["ci"],
                            "half_width": round(max(1.0 - lo, hi - 1.0), 5),
                            "old_median": judged["old_median"], "new_median": judged["new_median"]})
        widths = [r["half_width"] for r in results if "half_width" in r]
        aa = max(widths) if widths else None
        out[rid] = {"class": cls, "trials": results, "aa_half_width": aa,
                    "threshold": threshold(perf, {"aa_half_width": aa}) if aa is not None else None}
    return out
