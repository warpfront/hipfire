#!/usr/bin/env python3
"""S1 combined-lever quality campaign harness (OFF + PLACE_IU4 qualified fresh; PLACE / IU4 / QSA / STACK
definitions retained historical-only; all flags state-q8, no carry).

One lab executable (`qwen4_slowlink_quality`, built from this worktree) serves every arm. Each arm is
compared, through the native CPU `score` phase, against the SAME immutable beta (old Q1a `baseline`
phase directory) reference; nothing is ever scored against an `s1a` control. Per arm: one `s1a` GPU
phase (det7 + primary16k/4k teacher+greedy + resume) and one `s1b` GPU phase (det7 repeat, same route).
Qualified campaign: four GPU processes total (OFF + PLACE_IU4, 2 arms x 2 phases), each its own <=1800 s
fabric job. Old OFF / PLACE / IU4 / QSA / STACK samples are historical-only and are never reused as the
qualified baseline or candidate (no logical math/route/fixture equivalence proven).

Frozen interface (all paths absolute; every later command pins the sealed campaign file by sha256):

  prepare     fresh ROOT: validates the beta reference + staging identity, writes the per-arm plans,
              per-arm/phase native manifests, campaign.json. Prints the campaign sha256.
  check-plans native `--phase check-plan` over every arm plan (CPU).
  launch      detached external wrapper for ONE phase (nohup+setsid, fds>2 closed, own group, one
              fabric job). Refuses --max-secs > 1800.
  run-phase   runs inside the admitted held child: ONE fresh native GPU phase, private HOME/config/
              cache/locks, sanitized env, own-group drain, device-loss => INCOMPLETE, never retried.
  wait        read-only waiter for a launched phase.
  score       native CPU score of an arm vs the beta reference (+ --s1b-dir determinism).
  serial      detached, strictly serial driver: launch -> wait -> verify one phase at a time in the order
              off-a off-b place_iu4-a place_iu4-b (arms present in the campaign; historical definitions
              place / iu4 / qsa / stack run only if explicitly selected); never pre-queues; stops (no
              launch of the next) on any non-OK record, XTX drop/INCOMPLETE, leftover process or wait timeout.
  report      pure-Python quality report: reference identity, same-route determinism, decoded text, actual
              route evidence, and the absolute KL(beta||arm) GATE (point means vs the beta own-pair CI95
              upper bounds, 16K 0.03948458079446915 / 4K 0.027284316982763658, declared rounded
              0.0395/0.0273 also shown) with descriptive S1Integrate bootstrap intervals; optional
              paired delta diagnostic vs OFF (no verdict).

Exact invocations (W = the worktree holding this file; X = staged executable dir, e.g.
$ROOT/../stage; model/receipts as pinned by the Q1a stage):

  python3 -B $W/scripts/slowlink-gap/quality_campaign.py prepare --root $ROOT [--arms off,place_iu4] \\
      --exe $X/bin/qwen4_slowlink_quality --expect-exe-sha256 <sha> --config $CFG \\
      --source-pin $PIN.json [--expect-pack-digest <sha>] [--plan ... --beta-dir ... --bounds-file ...]
  ... check-plans --root $ROOT --expect-campaign-sha256 $C
  ... serial --root $ROOT --expect-campaign-sha256 $C --priority P1 --detach        (whole campaign)
  ... launch --root $ROOT --expect-campaign-sha256 $C --arm place_iu4 --phase s1a --max-secs 1800 --priority P1
  ... wait --root $ROOT --arm place_iu4 --phase s1a                                 (one phase per job)
  ... score --root $ROOT --expect-campaign-sha256 $C --arm off      (CPU, outside any hold; per arm)
  ... report --root $ROOT --expect-campaign-sha256 $C [--arms ...]

Lever env (the three historical keys set explicitly on every arm after sanitizing, plus the actual current
QSA gather kill-switch pinned 0 on every arm):
  HIPFIRE_QWEN4_SLOWLINK_RESIDENT_LAYERS  off=front   on=0,3,7,15,19,23,27,39,43
  HIPFIRE_QWEN4_MOE_SYM_IU4               off=0       on=1
  HIPFIRE_FN_QSA_GATHER_GFX1100           off=0       on=1 (historical key, inert in current runtime)
  HIPFIRE_QWEN4_QSA_WMMA_GATHER            0 on every arm (actual current gather control, default-on; =0 kills
                                          reservation and dispatch, so the failed QSA route cannot engage)
No carry: the q8 state cadence is unchanged and HIPFIRE_STATE_QUANT stays q8.
HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS stays unset. No native subset, count, tile, row or plan check is relaxed.
"""
import argparse
import copy
import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
import quality_scorer as qs  # noqa: E402

CAMPAIGN_SCHEMA = "hipfire.qwen4.slowlink_quality_campaign.v1"
RUN_RECORD_SCHEMA = "hipfire.qwen4.slowlink_quality_campaign.run_record.v1"
MANIFEST_SCHEMA = "hipfire.qwen4.slowlink_quality.manifest.v1"
PLAN_SCHEMA = "hipfire.qwen4.slowlink_quality.plan.v1"
PHASE_SUMMARY_SCHEMA = "hipfire.qwen4.slowlink_quality.phase_summary.v1"

RECIPE_SCHEMA = "hipfire.qwen4.slowlink_quality.execution_recipe.v1"
RECIPE_ARCHES = ("gfx1100", "gfx1201")
DEFAULT_ARCH = "gfx1100"
NEW_DEVICE_E = {"arch": "gfx1201", "card": "e", "bdf": "0000:c3:00.0", "vendor_id": "0x1002",
                "device_id": "0x744c", "uuid": "GPU-05f92432f2312a0e", "unique_id": "05f92432f2312a0e",
                "hip_visible": "0"}

R = "/home/kaden/qcal/release-0.4.3/slowlink-prefill"
DEFAULT_PLAN = R + "/holds/stage-q1-native-40a4e3bee/inputs/plan.json"
DEFAULT_BETA_DIR = (R + "/holds/collected/quality-q1a-40a4e3bee-native6c516939b-execution/run/"
                    "quality-q1a-40a4e3bee-native6c516939b.OUT/phases/baseline")
DEFAULT_MODEL_RECEIPT = R + "/holds/stage-q1-native-40a4e3bee/inputs/model.json"
DEFAULT_PLE_RECEIPT = R + "/holds/stage-q1-native-40a4e3bee/inputs/ple-q4.json"
DEFAULT_WRAPPER = R + "/wt-hold1-r7-harness/scripts/slowlink-holds/run-with-watcher.sh"
DEFAULT_FABRIC = "/home/kaden/qcal/locks/fabric/fabric-run.sh"

EXPECTED_BETA_PLAN_SHA256 = "74ce7b8291b15bc25254b589844ccd0c2f6b342dfb103e725736c8232b33bcf6"

# ---- the native class (hard constants; mirrors qwen4_slowlink_quality.rs PINNED_ENV and Q1a) ----
SELECTOR_ENV = "HIPFIRE_QWEN4_SLOWLINK_PREFILL"
EXPLICIT_LAYERS_ENV = "HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS"
PINNED_ENV = {
    "HIPFIRE_STATE_QUANT": "q8",
    "HIPFIRE_KV_MODE": "q8",
    "HIPFIRE_GRAPH": "0",
    "HIPFIRE_NORMALIZE_PROMPT": "0",
    "HIPFIRE_CHAT_CURRENT_DATE": "2026-10-17",
    "HIPFIRE_OFFLOAD_EXEC": "pcie",
    "HIPFIRE_PLE_FORMAT": "q4",
    "HIPFIRE_QWEN4_D2_DEFAULT": "0",
}
PLACE_SET = "0,3,7,15,19,23,27,39,43"
PLACE_LAYERS = [int(x) for x in PLACE_SET.split(",")]
LEVER_ENV = {  # lever -> (env name, off value, on value)
    "place": ("HIPFIRE_QWEN4_SLOWLINK_RESIDENT_LAYERS", "front", PLACE_SET),
    "iu4": ("HIPFIRE_QWEN4_MOE_SYM_IU4", "0", "1"),
    "qsa": ("HIPFIRE_FN_QSA_GATHER_GFX1100", "0", "1"),
}
# Actual current QSA gather kill-switch (default-on in runtime): pinned 0 on EVERY arm, qualified and
# historical alike, so the failed gather route cannot silently engage. Carried in expect.env (native
# check_env hard-asserts it) but outside LEVER_ENV/ARMS so historical lever combinations are unchanged.
QSA_OFF_ENV = "HIPFIRE_QWEN4_QSA_WMMA_GATHER"
QSA_OFF_VALUE = "0"
LEVERS = tuple(LEVER_ENV)
ARMS = {"off": (), "place": ("place",), "iu4": ("iu4",), "qsa": ("qsa",),
        "stack": ("place", "iu4", "qsa"), "place_iu4": ("place", "iu4")}
DEFAULT_BOUNDS_FILE = R + "/holds/Q1-PAIR-EXECUTION-RESULT.json"
PHASES = ("s1a", "s1b")
RESIDENT_LAYERS = 9
S1_TILE_ROWS = 768
DEVICE = {"arch": "gfx1100", "card": "xtx", "bdf": "0000:67:00.0", "vendor_id": "0x1002", "device_id": "0x744c",
          "uuid": "GPU-43390a851e296ee5", "unique_id": "43390a851e296ee5", "hip_visible": "0"}
PACK_REL_BY_ARCH = {"gfx1100": os.path.join("kernels", "compiled", "gfx1100"),
                    "gfx1201": os.path.join("kernels", "compiled", "gfx1201")}
PACK_REL = PACK_REL_BY_ARCH["gfx1100"]
ENV_STRIP_PREFIXES = ("HIPFIRE_", "HIP_", "ROCR_", "ROC_", "HSA_", "GPU_", "AMD_")
ENV_KEEP_EXACT = ("HIPFIRE_FABRIC_ADMISSION",)
MAX_JOB_SECS = 1800
CHILD_CAP_SECS = 1700      # 100 s reserved for termination / drain / finalization
KILL_GRACE_SECS = 20
DRAIN_VRAM_BYTES = 2 * 1024 ** 3
MIN_MEM_AVAILABLE_KIB = 80 * 1024 * 1024
NATIVE_CPU_CAP_SECS = 1200
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
EXAMPLE_REL = "crates/hipfire-arch-qwen4/examples/qwen4_slowlink_quality.rs"


class Refusal(Exception):
    """Refused before any mutation (no process, no GPU)."""


def camp_device(camp):
    d = (camp or {}).get("device")
    return dict(d) if isinstance(d, dict) else dict(DEVICE)


def camp_arch(camp):
    return camp_device(camp).get("arch", DEFAULT_ARCH)


def pack_rel_for(arch):
    if arch not in PACK_REL_BY_ARCH:
        raise Refusal("unknown arch %r (one of %s)" % (arch, list(PACK_REL_BY_ARCH)))
    return PACK_REL_BY_ARCH[arch]


def load_execution_recipe(path):
    need_abs(path, "--execution-recipe")
    raw = open(path, "rb").read()
    try:
        r = json.loads(raw)
    except ValueError as e:
        raise Refusal("execution recipe is not JSON: %s" % e)
    if r.get("schema") != RECIPE_SCHEMA:
        raise Refusal("execution recipe schema %r is not %s" % (r.get("schema"), RECIPE_SCHEMA))
    dev = r.get("device") or {}
    for k in ("arch", "card", "bdf", "vendor_id", "device_id", "uuid", "unique_id", "hip_visible"):
        if not isinstance(dev.get(k), str) or not dev[k]:
            raise Refusal("execution recipe device.%s must be a nonempty string" % k)
    if dev["arch"] not in RECIPE_ARCHES:
        raise Refusal("execution recipe device.arch %r not in %s" % (dev["arch"], list(RECIPE_ARCHES)))
    src = r.get("source") or {}
    if not isinstance(src.get("origin"), str) or not src["origin"]:
        raise Refusal("execution recipe source.origin must be a nonempty string")
    for k in ("pack_dir", "pack_digest", "pack_count"):
        if src.get(k) is None:
            raise Refusal("execution recipe source.%s is required" % k)
    if not os.path.isabs(src["pack_dir"]):
        raise Refusal("execution recipe source.pack_dir must be absolute")
    need_hex64(src["pack_digest"], "execution recipe source.pack_digest")
    elf = src.get("elf") or {}
    if not os.path.isabs(elf.get("path") or ""):
        raise Refusal("execution recipe source.elf.path must be absolute")
    need_hex64(elf.get("sha256", ""), "execution recipe source.elf.sha256")
    if not isinstance(elf.get("size"), int) or elf["size"] <= 0:
        raise Refusal("execution recipe source.elf.size must be a positive int")
    need_hex64(src.get("reference_plan_sha256", ""), "execution recipe source.reference_plan_sha256")
    pin = src.get("reference_source_pin") or {}
    if not HEX40.match(str(pin.get("commit", ""))) or not HEX64.match(str(pin.get("example_sha256", ""))):
        raise Refusal("execution recipe source.reference_source_pin needs 40-hex commit + 64-hex example_sha256")
    vc = r.get("verified_cache") or {}
    if not os.path.isabs(vc.get("home") or "") or not os.path.isabs(vc.get("path") or ""):
        raise Refusal("execution recipe verified_cache.home/path must be absolute")
    need_hex64(vc.get("sha256", ""), "execution recipe verified_cache.sha256")
    for e in vc.get("entries") or []:
        if not os.path.isabs(e.get("path") or ""):
            raise Refusal("execution recipe verified_cache.entries[].path must be absolute")
        need_hex64(e.get("sha256", ""), "execution recipe verified_cache entry sha256")
        for k in ("dev", "ino", "size", "mtime_ns"):
            if not isinstance(e.get(k), int):
                raise Refusal("execution recipe verified_cache entry %s must be an int" % k)
    exp = r.get("expect") or {}
    if not isinstance(exp.get("resident_layers"), int) or exp["resident_layers"] <= 0:
        raise Refusal("execution recipe expect.resident_layers must be a positive int")
    if not isinstance(exp.get("placement"), str) or not exp["placement"]:
        raise Refusal("execution recipe expect.placement must be a nonempty string")
    if exp.get("resident_layer_set") is not None:
        if (not isinstance(exp["resident_layer_set"], list)
                or sorted(exp["resident_layer_set"]) != sorted(set(exp["resident_layer_set"]))):
            raise Refusal("execution recipe expect.resident_layer_set must be a distinct list")
        if len(exp["resident_layer_set"]) != exp["resident_layers"]:
            raise Refusal("execution recipe expect.resident_layer_set length != resident_layers")
    res = r.get("resources") or {}
    for k in ("max_job_secs", "child_cap_secs", "kill_grace_secs", "drain_vram_bytes",
              "min_mem_available_kib", "native_cpu_cap_secs"):
        if not isinstance(res.get(k), int) or res[k] <= 0:
            raise Refusal("execution recipe resources.%s must be a positive int" % k)
    return r, sha256_bytes(raw)


def verify_verified_cache(recipe):
    vc = recipe.get("verified_cache") or {}
    problems = []
    try:
        raw = open(vc["path"], "rb").read()
    except OSError as ex:
        return ["verified cache %s unreadable: %s" % (vc.get("path"), ex)]
    if sha256_bytes(raw) != vc["sha256"]:
        return ["verified cache %s sha256 differs from the recipe seal" % vc["path"]]
    for e in vc.get("entries") or []:
        try:
            t = stat_tuple(e["path"])
        except OSError as ex:
            problems.append("verified cache entry %s unreadable: %s" % (e["path"], ex))
            continue
        for k in ("size", "dev", "ino", "mtime_ns"):
            if t[k] != e[k]:
                problems.append("verified cache entry %s: sealed %s != live %s" % (e["path"], k, t[k]))
    return problems


def plan_structural_equal_except_artifact_path(a, b, recipe=None):
    """Execution-binding whitelist: candidate plan must equal the frozen original except
    artifact.path (staged relocation, same verified sha+stat) and, ONLY when an explicit recipe
    binds it, expect.resident_layers. The expected set lives ONLY in recipe.expect.resident_layer_set
    (the frozen strict Expect has no set field; no new plan field is introduced, so native
    deny_unknown_fields is untouched). baseline_env keeps the original N9 truth; all else frozen."""
    x, y = copy.deepcopy(a), copy.deepcopy(b)
    for p in (x, y):
        if isinstance(p.get("artifact"), dict):
            p["artifact"].pop("path", None)
    if recipe is not None:
        rexp = recipe.get("expect") or {}
        for p in (x, y):
            exp = p.get("expect") if isinstance(p.get("expect"), dict) else {}
            exp.pop("resident_layers", None)
        yexp_actual = (b.get("expect") or {})
        if yexp_actual.get("resident_layers") != rexp.get("resident_layers"):
            return False
    return canonical(x) == canonical(y)


# ------------------------------------------------------------------ utilities

def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def pretty(obj):
    return (json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def write_new(path, data):
    """create-new + fsync (never overwrite)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    return sha256_bytes(data)


def read_json(path):
    try:
        with open(path, "rb") as f:
            return json.loads(f.read())
    except (OSError, ValueError) as e:
        raise Refusal("cannot read JSON %s: %s" % (path, e))


def stat_tuple(path):
    st = os.stat(path)
    return {"size": st.st_size, "dev": st.st_dev, "ino": st.st_ino, "mtime_ns": st.st_mtime_ns}


def need_abs(path, what):
    if not isinstance(path, str) or not os.path.isabs(path):
        raise Refusal("%s must be an absolute path: %r" % (what, path))
    return path


def need_hex64(v, what):
    if not isinstance(v, str) or not HEX64.match(v):
        raise Refusal("%s is not 64 lowercase hex digits: %r" % (what, v))
    return v


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def run_id(arm, phase):
    return "%s-%s" % (arm, phase)


def check_arm_phase(arm, phase=None):
    if arm not in ARMS:
        raise Refusal("unknown arm %r (one of %s)" % (arm, sorted(ARMS)))
    if phase is not None and phase not in PHASES:
        raise Refusal("unknown phase %r (one of %s)" % (phase, list(PHASES)))


class Host:
    """Clock / sleep / sysfs / env abstraction (CPU tests inject a fake)."""

    def __init__(self, environ=None):
        self._environ = dict(os.environ if environ is None else environ)

    def environ(self):
        return dict(self._environ)

    def monotonic(self):
        return time.monotonic()

    def sleep(self, seconds):
        time.sleep(seconds)

    def read_text(self, path):
        with open(path, "r") as f:
            return f.read()


# --------------------------------------------------------- env / plan / manifest

def lever_env(arm):
    """The three lever keys with their explicit value for `arm`, plus the actual current QSA gather
    kill-switch pinned 0 on every arm."""
    check_arm_phase(arm)
    on = ARMS[arm]
    env = {LEVER_ENV[l][0]: (LEVER_ENV[l][2] if l in on else LEVER_ENV[l][1]) for l in LEVERS}
    env[QSA_OFF_ENV] = QSA_OFF_VALUE
    return env


def sanitized_env(parent_env):
    """(env, unset names). FABRIC_* and the fabric admission token survive; every other HIPFIRE_/HIP_/ROCR_/
    ROC_/HSA_/GPU_/AMD_ variable is dropped. Lever keys are applied AFTER this. HIPFIRE_HOME is always
    dropped: the sealed recipe home (or the private run tree) is applied afterwards, never inherited."""
    env, unset = {}, []
    for k, v in parent_env.items():
        if k in ENV_KEEP_EXACT or k.startswith("FABRIC_"):
            env[k] = v
        elif k.startswith(ENV_STRIP_PREFIXES):
            unset.append(k)
        else:
            env[k] = v
    return env, sorted(unset)


def applied_env(arm, run_dir, device=None, hipfire_home=None):
    """Exactly what the harness sets for one GPU phase (deterministic; recorded in the manifest).
    Without a recipe this is the original xtx/gfx1100 private tree; with a recipe the device binding
    and the real HIPFIRE_HOME cache root come from the sealed recipe while HOME stays private."""
    device = dict(device) if device is not None else dict(DEVICE)
    home = os.path.join(run_dir, "home")
    a = {
        "HOME": home, "HIPFIRE_HOME": hipfire_home or os.path.join(home, ".hipfire"),
        "HIPFIRE_LOCK_DIR": os.path.join(run_dir, "lockdir"), "HIPFIRE_KERNEL_CACHE": os.path.join(run_dir, "kcache"),
        "HIPFIRE_DEVICES": device["uuid"], "ROCR_VISIBLE_DEVICES": device["uuid"],
        "HIP_VISIBLE_DEVICES": device["hip_visible"], "HIPFIRE_NO_REGISTRY_FETCH": "1",
        SELECTOR_ENV: "1",
    }
    a.update(PINNED_ENV)
    a.update(lever_env(arm))
    return dict(sorted(a.items()))


def process_env(parent_env, arm, run_dir, device=None, hipfire_home=None):
    """(env, unset, applied): sanitize first, then apply. EXPLICIT_LAYERS_ENV must be unset afterwards."""
    env, unset = sanitized_env(parent_env)
    applied = applied_env(arm, run_dir, device, hipfire_home)
    env.update(applied)
    if EXPLICIT_LAYERS_ENV in env:
        raise Refusal("%s survived sanitizing on an S1 route" % EXPLICIT_LAYERS_ENV)
    return env, unset, applied


def arm_plan(base_plan, arm, recipe=None):
    """The Q1a plan with ONLY expect.env extended by the explicit lever values (three historical keys plus
    the actual current QSA gather kill-switch pinned 0). The eight beta pins are asserted, never altered.
    With an explicit recipe the new resident count comes ONLY from recipe.expect (which must agree with
    the sealed plan); without a recipe the original N9/768/4096 checks are unchanged."""
    plan = copy.deepcopy(base_plan)
    if plan.get("schema") != PLAN_SCHEMA:
        raise Refusal("plan schema %r is not %s" % (plan.get("schema"), PLAN_SCHEMA))
    expect = plan.get("expect")
    if not isinstance(expect, dict) or not isinstance(expect.get("env"), dict):
        raise Refusal("plan.expect.env missing")
    for k, v in PINNED_ENV.items():
        if expect["env"].get(k) != v:
            raise Refusal("base plan expect.env[%s]=%r, expected the pinned %r" % (k, expect["env"].get(k), v))
    if recipe is None:
        if expect.get("resident_layers") != RESIDENT_LAYERS or expect.get("s1_tile_rows") != S1_TILE_ROWS \
                or expect.get("baseline_tile_rows") != 4096:
            raise Refusal("base plan expect counts/tiles differ from N9/4096/768 (native checks are never relaxed)")
    else:
        rexp = recipe.get("expect") or {}
        if expect.get("resident_layers") != rexp.get("resident_layers"):
            raise Refusal("recipe expect.resident_layers %r != plan expect %r" % (rexp.get("resident_layers"),
                                                                                  expect.get("resident_layers")))
        if expect.get("s1_tile_rows") != S1_TILE_ROWS or expect.get("baseline_tile_rows") != 4096:
            raise Refusal("recipe plan tiles differ from 4096/768 (native checks are never relaxed)")
    if EXPLICIT_LAYERS_ENV not in expect.get("env_absent", []):
        raise Refusal("base plan env_absent must name %s" % EXPLICIT_LAYERS_ENV)
    levers = lever_env(arm)
    for name in levers:
        if name in expect["env"] or name in expect.get("env_absent", []) or name in expect.get("baseline_env", {}):
            raise Refusal("base plan already mentions lever key %s" % name)
    expect["env"].update(levers)
    return plan


def plan_diff_only_expect_env(base, derived):
    """True iff the two plans differ only inside expect.env (corpus/history identical)."""
    a, b = copy.deepcopy(base), copy.deepcopy(derived)
    a["expect"].pop("env", None)
    b["expect"].pop("env", None)
    return canonical(a) == canonical(b)


def native_argv(exe, phase, plan_path, manifest_path, model_path, out_dir):
    return [exe, "--phase", phase, "--plan", plan_path, "--manifest", manifest_path, "--model", model_path,
            "--out-dir", out_dir]


def native_manifest(camp, arm, phase):
    """The native manifest of one arm/phase (writer: harness only). `camp` is the campaign dict (or its
    pre-seal form)."""
    a = camp["arms"][arm]["runs"][phase]
    argv = a["argv"]
    device = camp_device(camp)
    recipe_home = (camp.get("execution_recipe") or {}).get("verified_cache", {}).get("home")
    applied = applied_env(arm, a["run_dir"], device, recipe_home)
    m = camp["model"]
    source = {
        "kind": "s1-quality-lever-campaign", "campaign_label": camp["label"], "arm": arm, "phase": phase,
        "levers_on": list(ARMS[arm]), "plan_sha256": camp["arms"][arm]["plan"]["sha256"],
        "base_plan_sha256": camp["plan"]["sha256"], "harness_sha256": camp["harness"],
        "source_pin": camp["source_pin"], "pack_digest": camp["pack"]["digest"],
        "config_sha256": camp["config"]["sha256"],
    }
    if camp.get("execution_recipe") is not None:
        source["execution_recipe"] = camp["execution_recipe"]
    return {
        "schema": MANIFEST_SCHEMA,
        "binary": {"sha256": camp["exe"]["sha256"], "path": camp["exe"]["path"], "size_bytes": camp["exe"]["size"]},
        "source": source,
        "launch": {
            "controller": "quality_campaign.v1", "arm": arm, "phase": phase, "route": "s1",
            "command": argv, "command_text": " ".join(shlex.quote(x) for x in argv),
            "env": {"set": applied,
                    "unset_rule": "every HIPFIRE_/HIP_/ROCR_/ROC_/HSA_/GPU_/AMD_ variable except FABRIC_* and "
                                  "HIPFIRE_FABRIC_ADMISSION is removed before `set` is applied; "
                                  "%s is never set" % EXPLICIT_LAYERS_ENV},
            "lever_env": lever_env(arm), "device": device,
            "locks": "external fabric lock; private HIPFIRE_LOCK_DIR", "run_dir": a["run_dir"],
        },
        "model": {"path": m["path"], "dev": m["dev"], "ino": m["ino"], "size": m["size"],
                  "mtime_sec": m["mtime_ns"] // 10 ** 9, "mtime_nsec": m["mtime_ns"] % 10 ** 9,
                  "sha256": m["sha256"]},
    }


# ------------------------------------------------------------------- staging

def walk_pack(pack_dir):
    """Every regular file of the pack: rel, size, ino, mtime_ns, sha256. Symlinks / specials refuse."""
    files = []
    for dirpath, dirnames, filenames in os.walk(pack_dir):
        dirnames.sort()
        for name in sorted(filenames):
            p = os.path.join(dirpath, name)
            st = os.lstat(p)
            if not os.path.isfile(p) or os.path.islink(p):
                raise Refusal("pack member %s is not a regular file" % p)
            files.append({"rel": os.path.relpath(p, pack_dir), "size": st.st_size, "ino": st.st_ino,
                          "mtime_ns": st.st_mtime_ns, "sha256": sha256_file(p)})
        for d in dirnames:
            if os.path.islink(os.path.join(dirpath, d)):
                raise Refusal("pack member %s is a symlinked directory" % os.path.join(dirpath, d))
    if not files:
        raise Refusal("pack %s is empty" % pack_dir)
    return files


def pack_digest(files):
    return sha256_bytes(canonical([[f["rel"], f["size"], f["sha256"]] for f in files]))


def exe_identity(exe):
    real = os.path.realpath(exe)
    if not os.path.isfile(real):
        raise Refusal("executable %s is not a regular file" % exe)
    t = stat_tuple(real)
    t.update({"path": exe, "realpath": real, "sha256": sha256_file(real)})
    return t


def pack_dir_of(exe, arch=None):
    return os.path.join(os.path.dirname(exe), pack_rel_for(arch or DEFAULT_ARCH))


def verify_staging(camp, rehash=True):
    """Re-verify at run time: executable and pack (stat tuples, then sha256), config, plan, manifest files."""
    problems = []
    e = camp["exe"]
    try:
        real = os.path.realpath(e["path"])
        t = stat_tuple(real)
        if real != e["realpath"] or any(t[k] != e[k] for k in ("size", "dev", "ino", "mtime_ns")):
            problems.append("executable stat/realpath differs from the sealed identity")
        elif rehash and sha256_file(real) != e["sha256"]:
            problems.append("executable sha256 differs from the sealed identity")
    except OSError as ex:
        problems.append("executable unreadable: %s" % ex)
    pk = camp["pack"]
    try:
        for f in pk["files"]:
            p = os.path.join(pk["dir"], f["rel"])
            st = os.lstat(p)
            if os.path.islink(p) or st.st_size != f["size"] or st.st_ino != f["ino"] or st.st_mtime_ns != f["mtime_ns"]:
                problems.append("pack member %s differs (stat)" % f["rel"])
            elif rehash and sha256_file(p) != f["sha256"]:
                problems.append("pack member %s differs (sha256)" % f["rel"])
        live = sorted(os.path.relpath(os.path.join(d, n), pk["dir"]) for d, _s, ns in os.walk(pk["dir"]) for n in ns)
        if live != sorted(f["rel"] for f in pk["files"]):
            problems.append("pack member set differs from the sealed manifest")
    except OSError as ex:
        problems.append("pack unreadable: %s" % ex)
    for label, ent in (("config", camp["config"]),):
        try:
            if sha256_file(ent["path"]) != ent["sha256"]:
                problems.append("%s file differs from the sealed sha256" % label)
        except OSError as ex:
            problems.append("%s unreadable: %s" % (label, ex))
    for arm, a in camp["arms"].items():
        for what, ent in [("plan", a["plan"])] + [("manifest " + ph, r["manifest"]) for ph, r in a["runs"].items()]:
            try:
                if sha256_file(ent["path"]) != ent["sha256"]:
                    problems.append("%s of %s differs from the sealed sha256" % (what, arm))
            except OSError as ex:
                problems.append("%s of %s unreadable: %s" % (what, arm, ex))
    return problems


def verify_model_stat(camp):
    """Cached-stat only (never a payload read): model and ple-q4 against the sealed tuples."""
    problems = []
    for label in ("model", "ple"):
        m = camp[label]
        try:
            t = stat_tuple(m["path"])
        except OSError as ex:
            problems.append("%s %s unreadable: %s" % (label, m["path"], ex))
            continue
        for k in ("size", "dev", "ino", "mtime_ns"):
            if t[k] != m[k]:
                problems.append("%s %s: sealed %s != live %s" % (label, m["path"], m[k], t[k]))
    return problems


def cached_model_entry(receipt_path, path, plan_artifact=None):
    r = read_json(receipt_path)
    if r.get("path") != path:
        raise Refusal("receipt %s names %r, expected %r" % (receipt_path, r.get("path"), path))
    need_hex64(r.get("sha256"), "receipt sha256")
    t = stat_tuple(path)   # live stat; the payload is never read or hashed
    for src, k in (("size_bytes", "size"), ("device", "dev"), ("inode", "ino"), ("mtime_ns", "mtime_ns")):
        if r.get(src) != t[k]:
            raise Refusal("%s: cached %s %r != live %r" % (path, src, r.get(src), t[k]))
    if plan_artifact is not None:
        if plan_artifact["path"] != path or plan_artifact["size"] != t["size"] or plan_artifact["sha256"] != r["sha256"]:
            raise Refusal("plan.artifact differs from the cached model receipt / live size")
    return {"path": path, "size": t["size"], "dev": t["dev"], "ino": t["ino"], "mtime_ns": t["mtime_ns"],
            "sha256": r["sha256"], "receipt": receipt_path, "receipt_sha256": sha256_file(receipt_path),
            "payload_hashed_here": False}


# ----------------------------------------------------------- beta reference

BETA_CELLS = {"primary-ctx16k-teacher": qs.EXPECTED_ROWS, "primary-ctx4k-teacher": qs.EXPECTED_ROWS,
              "resume-cold": 32, "resume-split": 32}   # name -> rows (resume raw refs for the immediate s1a score)


def validate_beta(beta_dir, plan_sha):
    """Seal the immutable beta (Q1a baseline) reference by summary/rows-file hashes + file stat tuples.
    The multi-hundred-MiB logits files are stat-checked here; the native score re-verifies every hash."""
    need_abs(beta_dir, "--beta-dir")
    sp = os.path.join(beta_dir, "phase-summary.json")
    s = read_json(sp)
    if s.get("schema") != PHASE_SUMMARY_SCHEMA or s.get("complete") is not True:
        raise Refusal("beta phase-summary is not a complete %s" % PHASE_SUMMARY_SCHEMA)
    if s.get("phase") != "baseline" or s.get("arm") != "baseline" or s.get("state_f32") is not False:
        raise Refusal("beta summary phase/arm/state_f32 is not baseline/baseline/false")
    if (s.get("plan") or {}).get("sha256") != plan_sha:
        raise Refusal("beta summary plan sha256 %r != the supplied plan %s" % ((s.get("plan") or {}).get("sha256"), plan_sha))
    ld = s.get("load") or {}
    want = {"placement": "explicit-env", "resident_layers": RESIDENT_LAYERS, "scratch_rows": 4096,
            "gpu_arch": "gfx1100", "ple_format": "q4", "slowlink_profile": False}
    for k, v in want.items():
        if ld.get(k) != v:
            raise Refusal("beta load.%s=%r, expected %r" % (k, ld.get(k), v))
    if ld.get("state_format") != {"gdn": "Q8", "qsa_kv": "Q8"}:
        raise Refusal("beta state_format is not Q8/Q8: %r" % (ld.get("state_format"),))
    env = s.get("environment") or {}
    for k, v in dict(PINNED_ENV, HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS="9", HIPFIRE_PREFILL_CHUNK_ROWS="4096").items():
        if env.get(k) != v:
            raise Refusal("beta environment[%s]=%r, expected %r" % (k, env.get(k), v))
    vocab = ld.get("vocab")
    cells = {c["cell"]: c for c in s.get("cells", []) if isinstance(c, dict) and "cell" in c}
    cellfile = os.path.join(beta_dir, "cells.jsonl")
    full = {}
    with open(cellfile, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                full[rec["cell"]] = rec
    out = {}
    for name in BETA_CELLS:
        c, fc = cells.get(name), full.get(name)
        if c is None or fc is None:
            raise Refusal("beta lacks cell %s" % name)
        need_hex64(c.get("stream_sha256"), name + " stream sha256")
        exp_rows = BETA_CELLS[name]
        if c["stream_sha256"] != fc.get("stream_sha256") or c.get("rows") != exp_rows or fc.get("rows") != exp_rows:
            raise Refusal("beta cell %s summary/cells.jsonl disagree or rows != %d" % (name, exp_rows))
        lp = os.path.join(beta_dir, fc["bin"])
        rp = os.path.join(beta_dir, fc["rows_file"])
        lt = stat_tuple(lp)
        if lt["size"] != exp_rows * vocab * 4 or lt["size"] != fc.get("bin_bytes"):
            raise Refusal("beta %s logits size %d != rows*vocab*4" % (name, lt["size"]))
        if sha256_file(rp) != fc.get("rows_file_sha256"):
            raise Refusal("beta %s rows file hash differs from cells.jsonl" % name)
        out[name] = {"stream_sha256": c["stream_sha256"], "rows": c["rows"], "prompt_sha256": fc.get("prompt_sha256"),
                     "logits": dict(lt, path=lp), "rows_file": dict(stat_tuple(rp), path=rp, sha256=fc["rows_file_sha256"])}
    return {"dir": beta_dir, "vocab": vocab, "phase_summary": dict(stat_tuple(sp), path=sp, sha256=sha256_file(sp)),
            "cells_jsonl": dict(stat_tuple(cellfile), path=cellfile, sha256=sha256_file(cellfile)),
            "binary_sha256": (s.get("binary") or {}).get("sha256"), "cells": out,
            "manifest_sha256": (s.get("manifest") or {}).get("sha256"),
            "manifest_source": ((s.get("manifest") or {}).get("content") or {}).get("source"),
            "plan_sha256": (s.get("plan") or {}).get("sha256")}


def verify_beta(camp):
    """Immutable reference unchanged: summary hash + every sealed stat tuple."""
    b = camp["beta"]
    problems = []
    ents = [b["phase_summary"], b["cells_jsonl"]]
    for c in b["cells"].values():
        ents += [c["logits"], c["rows_file"]]
    for e in ents:
        try:
            t = stat_tuple(e["path"])
        except OSError as ex:
            problems.append("beta file %s unreadable: %s" % (e["path"], ex))
            continue
        if any(t[k] != e[k] for k in ("size", "dev", "ino", "mtime_ns")):
            problems.append("beta file %s changed (stat)" % e["path"])
    try:
        if sha256_file(b["phase_summary"]["path"]) != b["phase_summary"]["sha256"]:
            problems.append("beta phase-summary hash changed")
    except OSError as ex:
        problems.append(str(ex))
    return problems


def parse_arms(text):
    """Canonical-order arm list from a comma list (default: the qualified OFF + PLACE_IU4)."""
    names = text.split(",") if text else ["off", "place_iu4"]
    if not names or len(set(names)) != len(names):
        raise Refusal("--arms must be a nonempty list of distinct arms")
    for a in names:
        check_arm_phase(a)
    return [a for a in ARMS if a in names]


def verify_bounds_file(path):
    """The sealed exact beta own-pair CI95 upper bounds must be the ones the scorer gates on."""
    need_abs(path, "--bounds-file")
    sha = sha256_file(path)
    if sha != qs.BOUNDS_FILE_SHA256:
        raise Refusal("bounds file sha256 %s != the approved %s" % (sha, qs.BOUNDS_FILE_SHA256))
    d = read_json(path)
    try:
        ctxs = d["parts"]["scale"]["summary"]["scale"]["contexts"]
        got = {c: ctxs[c]["kl_mean_ci95_upper"] for c in qs.GATE_BOUNDS_EXACT}
    except (KeyError, TypeError) as e:
        raise Refusal("bounds file lacks parts.scale.summary.scale.contexts.<ctx>.kl_mean_ci95_upper: %s" % e)
    if got != qs.GATE_BOUNDS_EXACT:
        raise Refusal("bounds file values %r != scorer bounds %r" % (got, qs.GATE_BOUNDS_EXACT))
    return {"path": path, "sha256": sha, "values": got}


# -------------------------------------------------------------------- prepare

def prepare(args):
    root = need_abs(args.root, "--root")
    if os.path.exists(root):
        raise Refusal("refusing existing --root %s" % root)
    recipe, recipe_sha = (load_execution_recipe(args.execution_recipe) if args.execution_recipe else (None, None))
    if recipe is not None and args.expect_pack_digest and args.expect_pack_digest != recipe["source"]["pack_digest"]:
        raise Refusal("--expect-pack-digest differs from the execution recipe source.pack_digest")
    exe = need_abs(args.exe, "--exe")
    need_hex64(args.expect_exe_sha256, "--expect-exe-sha256")
    ident = exe_identity(exe)
    if ident["sha256"] != args.expect_exe_sha256:
        raise Refusal("executable sha256 %s != --expect-exe-sha256" % ident["sha256"])
    arch = (recipe["device"]["arch"] if recipe else DEFAULT_ARCH)
    device = dict(recipe["device"]) if recipe else dict(DEVICE)
    pdir = pack_dir_of(exe, arch)
    if not os.path.isdir(pdir):
        raise Refusal("pack %s missing (the kernel pack must sit at <exe dir>/%s)" % (pdir, pack_rel_for(arch)))
    files = walk_pack(pdir)
    digest = pack_digest(files)
    if recipe is not None:
        if recipe["source"]["pack_dir"] != pdir:
            raise Refusal("recipe source.pack_dir %r != staged pack dir %r" % (recipe["source"]["pack_dir"], pdir))
        if recipe["source"]["pack_digest"] != digest:
            raise Refusal("recipe source.pack_digest differs from the staged pack digest")
        if recipe["source"]["pack_count"] != len(files):
            raise Refusal("recipe source.pack_count != staged file count")
        if recipe["source"]["elf"]["sha256"] != ident["sha256"] or recipe["source"]["elf"]["size"] != ident["size"]:
            raise Refusal("recipe source.elf identity differs from the staged executable")
    if args.expect_pack_digest and args.expect_pack_digest != digest:
        raise Refusal("pack digest %s != --expect-pack-digest" % digest)
    pin = read_json(need_abs(args.source_pin, "--source-pin"))
    if not isinstance(pin, dict) or not HEX40.match(str(pin.get("commit", ""))) \
            or not HEX64.match(str(pin.get("example_sha256", ""))):
        raise Refusal("--source-pin needs a 40-hex `commit` and a 64-hex `example_sha256`")
    w_root = os.path.dirname(os.path.dirname(HERE))
    example = os.path.join(w_root, EXAMPLE_REL)
    if sha256_file(example) != pin["example_sha256"]:
        raise Refusal("source pin example_sha256 differs from %s" % example)
    if recipe is not None and recipe["source"]["reference_source_pin"] != pin:
        raise Refusal("recipe source.reference_source_pin differs from --source-pin")
    cfg = need_abs(args.config, "--config")
    if not os.path.isfile(cfg):
        raise Refusal("--config %s is not a file" % cfg)
    plan_path = need_abs(args.plan, "--plan")
    plan_bytes = open(plan_path, "rb").read()
    plan_sha = sha256_bytes(plan_bytes)
    if recipe is None:
        if plan_sha != EXPECTED_BETA_PLAN_SHA256 and not args.allow_other_plan:
            raise Refusal("plan sha256 %s is not the Q1a plan %s" % (plan_sha, EXPECTED_BETA_PLAN_SHA256))
    else:
        ref_sha = recipe["source"]["reference_plan_sha256"]
        if ref_sha != EXPECTED_BETA_PLAN_SHA256:
            raise Refusal("recipe source.reference_plan_sha256 is not the frozen Q1a original")
        ref_dir = os.path.dirname(plan_path)
        ref_candidates = [os.path.join(ref_dir, "reference-plan.json"), DEFAULT_PLAN]
        ref_bytes = None
        for cand in ref_candidates:
            try:
                ref_bytes = open(cand, "rb").read()
            except OSError:
                continue
            if sha256_bytes(ref_bytes) == ref_sha:
                break
            ref_bytes = None
        if ref_bytes is None and not args.allow_other_plan:
            raise Refusal("frozen Q1a original plan %s unavailable next to --plan (or --allow-other-plan)" % ref_sha)
        if ref_bytes is not None:
            ref = json.loads(ref_bytes)
            cand_plan = json.loads(plan_bytes)
            if not plan_structural_equal_except_artifact_path(ref, cand_plan, recipe):
                raise Refusal("candidate plan differs from the frozen original outside artifact.path and "
                              "recipe-bound expect.resident_layers/resident_layer_set")
            rart, cart = ref.get("artifact") or {}, cand_plan.get("artifact") or {}
            if cart.get("sha256") != rart.get("sha256") or cart.get("size") != rart.get("size"):
                raise Refusal("candidate plan artifact sha/size differ from the frozen original")
            rb, cb = ref.get("expect", {}).get("baseline_env"), cand_plan.get("expect", {}).get("baseline_env")
            if rb != cb:
                raise Refusal("candidate plan baseline_env differs from the frozen original N9 truth")
    base = json.loads(plan_bytes)
    art = base.get("artifact") or {}
    model = cached_model_entry(args.model_receipt, art.get("path"), plan_artifact=art)
    ple = cached_model_entry(args.ple_receipt, art["path"] + ".ple-q4")
    if recipe is not None:
        entries = {e["path"]: e for e in recipe["verified_cache"].get("entries") or []}
        for m in (model, ple):
            e = entries.get(m["path"])
            if e is None or e["sha256"] != m["sha256"] or any(e[k] != m[k] for k in ("size", "dev", "ino", "mtime_ns")):
                raise Refusal("recipe verified_cache entry for %s differs from the sealed model stat+sha" % m["path"])
        if sha256_file(recipe["verified_cache"]["path"]) != recipe["verified_cache"]["sha256"]:
            raise Refusal("verified cache file differs from the recipe seal")
    beta = validate_beta(args.beta_dir, plan_sha if recipe is None else recipe["source"]["reference_plan_sha256"])
    selected = parse_arms(args.arms)
    bounds = verify_bounds_file(args.bounds_file)
    label = args.label or "s1-quality-" + time.strftime("%Y%m%d", time.gmtime())
    harness = harness_hashes(HERE)
    os.makedirs(root)
    camp = {
        "schema": CAMPAIGN_SCHEMA, "label": label, "created_utc": utc_now(), "root": root, "harness": harness,
        "source_pin": pin,
        "exe": {k: ident[k] for k in ("path", "realpath", "sha256", "size", "dev", "ino", "mtime_ns")},
        "pack": {"dir": pdir, "digest": digest, "count": len(files), "files": files, "arch": arch},
        "config": {"path": cfg, "sha256": sha256_file(cfg)},
        "model": model, "ple": ple,
        "plan": {"path": plan_path, "sha256": plan_sha},
        "beta": beta, "device": device,
        "reference_declaration": {
            "kind": "frozen-old-reference-cross-binary",
            "reference_binary_sha256": beta["binary_sha256"], "reference_manifest_sha256": beta["manifest_sha256"],
            "reference_manifest_source": beta["manifest_source"], "reference_plan_sha256": beta["plan_sha256"],
            "candidate_binary_sha256": ident["sha256"], "candidate_source_pin": pin,
            "common_identity_asserted": False,
            "note": ("the reference is the old immutable Q1a baseline produced by a DIFFERENT binary/source; no common "
                     "binary/source identity is claimed. Native `score` does not require equal binary/plan hashes: it "
                     "re-verifies model, corpus, prompt, teacher, vocab, row and stream hashes. Every arm plan differs "
                     "from the reference plan only in expect.env (the three explicit lever keys); `bridge-check` "
                     "proves the new executable reproduces the archived Q1a score bit-for-bit.")},
        "policy": {"gate": "mean KL(beta||arm) <= beta own-pair CI95 upper, per context, inclusive",
                   "bounds_exact": qs.GATE_BOUNDS_EXACT, "bounds_declared_rounded": qs.GATE_BOUNDS_DECLARED,
                   "bounds_file": bounds, "pooled_gated": False,
                   "bootstrap": {"method": "stratified moving-block (descriptive only)", "block": qs.BOOT_BLOCK,
                                 "reps": qs.BOOT_REPS, "seed_label": qs.BOOT_LABEL},
                   "selected_pairs": list(qs.PRIMARY_PAIRS), "arms": selected},
        "arms": {},
    }
    for arm in selected:
        ap = arm_plan(base, arm)
        if not plan_diff_only_expect_env(base, ap):
            raise Refusal("internal: arm plan differs from the base outside expect.env")
        pbytes = json.dumps(ap, separators=(",", ":")).encode("utf-8")
        ppath = os.path.join(root, "plans", arm + ".plan.json")
        psha = write_new(ppath, pbytes)
        runs = {}
        for ph in PHASES:
            rid = run_id(arm, ph)
            run_dir = os.path.join(root, "runs", rid)
            out_dir = os.path.join(root, "phases", rid)
            mpath = os.path.join(root, "manifests", rid + ".manifest.json")
            runs[ph] = {"run_dir": run_dir, "out_dir": out_dir,
                        "argv": native_argv(exe, ph, ppath, mpath, model["path"], out_dir),
                        "manifest": {"path": mpath, "sha256": None}}
        camp["arms"][arm] = {"levers_on": list(ARMS[arm]), "lever_env": lever_env(arm),
                             "plan": {"path": ppath, "sha256": psha}, "runs": runs}
    for arm in selected:
        for ph in PHASES:
            r = camp["arms"][arm]["runs"][ph]
            r["manifest"]["sha256"] = write_new(r["manifest"]["path"], pretty(native_manifest(camp, arm, ph)))
    csha = write_new(os.path.join(root, "campaign.json"), pretty(camp))
    print(json.dumps({"campaign": os.path.join(root, "campaign.json"), "campaign_sha256": csha,
                      "pack_digest": digest, "plans": {a: camp["arms"][a]["plan"]["sha256"] for a in selected}},
                     indent=2, sort_keys=True))
    return 0


def harness_hashes(directory):
    """Identity of the Python harness and its native source, never the runtime binary."""
    need_abs(directory, "harness directory")
    return {
        "quality_campaign_sha256": sha256_file(os.path.join(directory, "quality_campaign.py")),
        "quality_scorer_sha256": sha256_file(os.path.join(directory, "quality_scorer.py")),
        "example_sha256": sha256_file(os.path.join(os.path.dirname(os.path.dirname(directory)), EXAMPLE_REL)),
    }


def load_campaign(root, expect_sha, *, recorded_harness_dir=None, report_only=False):
    need_abs(root, "--root")
    need_hex64(expect_sha, "--expect-campaign-sha256")
    path = os.path.join(root, "campaign.json")
    try:
        data = open(path, "rb").read()
    except OSError as e:
        raise Refusal("cannot read %s: %s" % (path, e))
    if sha256_bytes(data) != expect_sha:
        raise Refusal("campaign.json sha256 %s != --expect-campaign-sha256" % sha256_bytes(data))
    camp = json.loads(data)
    if camp.get("schema") != CAMPAIGN_SCHEMA or camp.get("root") != root:
        raise Refusal("campaign schema/root mismatch")
    if recorded_harness_dir is not None and not report_only:
        raise Refusal("--recorded-harness-dir is restricted to CPU report")
    directory = recorded_harness_dir if recorded_harness_dir is not None else HERE
    if harness_hashes(directory) != camp.get("harness"):
        raise Refusal("harness source hashes differ from the sealed campaign (%s)" % directory)
    return camp


def clean_cpu_env(parent_env, home):
    env, _unset = sanitized_env(parent_env)
    env["HOME"] = home
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def check_plans(args):
    camp = load_campaign(args.root, args.expect_campaign_sha256)
    bad = verify_staging(camp, rehash=True)
    if bad:
        raise Refusal("; ".join(bad))
    home = os.path.join(camp["root"], "cpu-home")
    os.makedirs(home, exist_ok=True)
    env = clean_cpu_env(os.environ, home)
    rc = 0
    cpu_cap = ((camp.get("execution_recipe") or {}).get("resources") or {}).get("native_cpu_cap_secs",
                                                                                  NATIVE_CPU_CAP_SECS)
    for arm, a in camp["arms"].items():
        argv = [camp["exe"]["path"], "--phase", "check-plan", "--plan", a["plan"]["path"]]
        p = subprocess.run(argv, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=cpu_cap)
        ok = p.returncode == 0
        try:
            rep = json.loads(p.stdout)
        except ValueError:
            rep = {}
        if ok and rep.get("plan_sha256") != a["plan"]["sha256"]:
            ok = False
        print("%s: %s" % (arm, "valid" if ok else "INVALID rc=%d %s" % (p.returncode, p.stderr.strip()[:400])))
        rc = rc or (0 if ok else 1)
    return rc


# --------------------------------------------------------------------- launch

def build_launch_command(camp, arm, phase, max_secs, priority, wrapper=DEFAULT_WRAPPER, fabric=DEFAULT_FABRIC,
                         owner="S1QualG", python="python3"):
    """argv of the external detached launcher body (pure; tested)."""
    check_arm_phase(arm, phase)
    recipe_max = ((camp.get("execution_recipe") or {}).get("resources") or {}).get("max_job_secs", MAX_JOB_SECS)
    if not isinstance(max_secs, int) or max_secs <= 0 or max_secs > recipe_max:
        raise Refusal("--max-secs must be 1..%d" % recipe_max)
    if not re.match(r"^P[0-9]$", str(priority)):
        raise Refusal("--priority must look like P1")
    me = os.path.realpath(__file__)
    csha = sha256_file(os.path.join(camp["root"], "campaign.json"))
    inner = [python, "-B", me, "run-phase", "--root", camp["root"], "--expect-campaign-sha256", csha,
             "--arm", arm, "--phase", phase]
    return ["bash", wrapper, fabric, "--class", "correct", "--card", camp_device(camp)["card"], "--mem-gb", "22.4",
            "--hostmem-gb", "67", "--hostmem-lock", "--max-secs", str(max_secs), "--priority", priority,
            "--owner", owner, "--"] + inner


DETACH_SCRIPT = (
    'for f in /proc/$$/fd/*; do n=${f##*/}; [ "$n" -gt 2 ] 2>/dev/null && eval "exec $n>&-"; done; '
    'exec env OUT_PREFIX="$1" PYTHONDONTWRITEBYTECODE=1 "${@:2}"'
)


def launch(args):
    camp = load_campaign(args.root, args.expect_campaign_sha256)
    check_arm_phase(args.arm, args.phase)
    if args.arm not in camp["arms"]:
        raise Refusal("arm %s is not in this campaign (%s)" % (args.arm, sorted(camp["arms"])))
    rid = run_id(args.arm, args.phase)
    ldir = os.path.join(camp["root"], "launch", rid)
    if os.path.exists(ldir) or os.path.exists(camp["arms"][args.arm]["runs"][args.phase]["run_dir"]):
        raise Refusal("refusing: %s was already launched/run (no automatic retry)" % rid)
    inner = build_launch_command(camp, args.arm, args.phase, args.max_secs, args.priority,
                                 wrapper=args.wrapper, fabric=args.fabric)
    os.makedirs(ldir)
    prefix = os.path.join(ldir, "watch")
    log = open(os.path.join(ldir, "launcher.log"), "ab")
    p = subprocess.Popen(["nohup", "setsid", "bash", "-c", DETACH_SCRIPT, "launcher", prefix] + inner,
                         stdin=subprocess.DEVNULL, stdout=log, stderr=log, close_fds=True, start_new_session=False)
    time.sleep(1)
    rec = {"id": rid, "pid": p.pid, "boot_id": open("/proc/sys/kernel/random/boot_id").read().strip(),
           "launch_unix": time.time(), "argv": inner, "out_prefix": prefix}
    write_new(os.path.join(ldir, "launch.json"), pretty(rec))
    print(json.dumps(rec, indent=2, sort_keys=True))
    return 0


def wait_cmd(args):
    """Read-only: blocks until <launch>/watch.exit exists, then prints the exit + run record summary."""
    camp_root = need_abs(args.root, "--root")
    rid = run_id(args.arm, args.phase)
    prefix = os.path.join(camp_root, "launch", rid, "watch")
    end = time.monotonic() + args.max_wait
    while not os.path.exists(prefix + ".exit"):
        if time.monotonic() >= end:
            print("WAIT_TIMEOUT " + rid)
            return 75
        time.sleep(15)
    print("exit=" + open(prefix + ".exit").read().strip())
    rr = os.path.join(camp_root, "runs", rid, "run-record.json")
    if os.path.exists(rr):
        r = json.load(open(rr))
        print(json.dumps({k: r.get(k) for k in ("status", "exit_code", "device_lost", "leftover", "failures")},
                         sort_keys=True))
    else:
        print("no run-record.json (run did not reach its record)")
    return 0


# ------------------------------------------------------------------ lifecycle

def read_sysfs(host, name, bdf=None):
    try:
        return host.read_text("/sys/bus/pci/devices/%s/%s" % (bdf or DEVICE["bdf"], name)).strip()
    except OSError:
        return None


def device_state(host, device=None):
    """(ok, problems). Any disappearance is INCOMPLETE, never retried. Arch/uuid/bdf come from the
    sealed recipe device when a campaign binds one, else the original xtx default."""
    device = dict(device) if device is not None else dict(DEVICE)
    problems = []
    vendor, dev = read_sysfs(host, "vendor", device["bdf"]), read_sysfs(host, "device", device["bdf"])
    if vendor != device["vendor_id"]:
        problems.append("vendor %r != %s" % (vendor, device["vendor_id"]))
    if dev != device["device_id"]:
        problems.append("device %r != %s" % (dev, device["device_id"]))
    uid = read_sysfs(host, "unique_id", device["bdf"])
    if uid is not None and uid.lower() != device["unique_id"]:
        problems.append("unique_id %r != %s" % (uid, device["unique_id"]))
    return (not problems), problems


def vram_used(host, device=None):
    v = read_sysfs(host, "mem_info_vram_used", (device or DEVICE)["bdf"])
    try:
        return int(v)
    except (TypeError, ValueError):
        return -1


def mem_available_kib(host):
    for line in host.read_text("/proc/meminfo").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])
    return -1


def group_members(pgid):
    """(pid, comm) of every process whose process-group id is pgid (own-group enumeration; never a pattern kill)."""
    out = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open("/proc/%s/stat" % name, "r") as f:
                stat = f.read()
            tail = stat[stat.rindex(")") + 2:].split()
            if int(tail[2]) == pgid and tail[0] not in ("Z", "X"):   # an unreaped zombie holds no resources
                out.append((int(name), stat[stat.index("(") + 1:stat.rindex(")")]))
        except (OSError, ValueError, IndexError):
            continue
    return out


def terminate_group(pgid, grace, sleep=time.sleep, poll=0.1):
    """TERM the OWN group, wait `grace`, then KILL it; returns the members still live afterwards."""
    for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 10)):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return []
        waited = 0.0
        while waited < wait:
            if not group_members(pgid):
                return []
            sleep(poll)
            waited += poll
    return group_members(pgid)


class Stop:
    flag = None


def _on_signal(signum, _frame):
    Stop.flag = signal.Signals(signum).name


def phase_expectations(camp, arm, phase):
    """Plan-derived cell order of the phase."""
    plan = read_json(camp["arms"][arm]["plan"]["path"])
    det = ["det-" + i for i in plan["determinism"]["case_ids"]]
    if phase == "s1b":
        return det
    cells = []
    for ctx in plan["primary"]["contexts"]:
        cells += ["primary-%s-teacher" % ctx["id"], "primary-%s-greedy" % ctx["id"]]
    return det + cells + ["resume-split", "resume-cold"]


def verify_phase_dir(camp, arm, phase):
    """Native outputs of a completed phase against the sealed identities; returns problems."""
    a = camp["arms"][arm]
    out = a["runs"][phase]["out_dir"]
    probs = []
    try:
        s = json.load(open(os.path.join(out, "phase-summary.json")))
    except (OSError, ValueError) as e:
        return ["phase-summary.json unreadable (incomplete phase): %s" % e]
    exp = dict(PINNED_ENV, **a["lever_env"])
    checks = [
        (s.get("schema") == PHASE_SUMMARY_SCHEMA, "summary schema"),
        (s.get("complete") is True, "complete"),
        (s.get("phase") == phase and s.get("arm") == "s1", "phase/arm"),
        (s.get("state_f32") is False, "state_f32 must be false (q8 state pin)"),
        ((s.get("binary") or {}).get("sha256") == camp["exe"]["sha256"], "binary sha256"),
        ((s.get("plan") or {}).get("sha256") == a["plan"]["sha256"], "plan sha256"),
        ((s.get("manifest") or {}).get("sha256") == a["runs"][phase]["manifest"]["sha256"], "manifest sha256"),
        ((s.get("model") or {}).get("pinned_sha256") == camp["model"]["sha256"], "model pinned sha256"),
    ]
    probs += ["phase-summary: %s mismatch" % w for ok, w in checks if not ok]
    env = s.get("environment") or {}
    for k, v in exp.items():
        if env.get(k) != v:
            probs.append("observed environment[%s]=%r, expected %r" % (k, env.get(k), v))
    if env.get(EXPLICIT_LAYERS_ENV, None) is not None:
        probs.append("observed %s is set" % EXPLICIT_LAYERS_ENV)
    ld = s.get("load") or {}
    arch = camp_arch(camp)
    recipe_exp = ((camp.get("execution_recipe") or {}).get("expect")) if camp.get("execution_recipe") else None
    want_n = recipe_exp["resident_layers"] if recipe_exp else RESIDENT_LAYERS
    for k, v in {"placement": "slowlink-profile", "slowlink_profile": True, "resident_layers": want_n,
                 "scratch_rows": S1_TILE_ROWS, "gpu_arch": arch, "ple_format": "q4"}.items():
        if ld.get(k) != v:
            probs.append("load.%s=%r, expected %r" % (k, ld.get(k), v))
    if ld.get("state_format") != {"gdn": "Q8", "qsa_kv": "Q8"}:
        probs.append("load.state_format is not Q8/Q8: %r" % (ld.get("state_format"),))
    prof = ld.get("profile") or {}
    if prof.get("tile_rows") != S1_TILE_ROWS or prof.get("purpose") != "product" or prof.get("resident_layers") != want_n:
        probs.append("load.profile is not the product 768-tile N%d profile: %r" % (want_n, prof,))
    if not isinstance(ld.get("host_mapped_tensors"), int) or ld["host_mapped_tensors"] <= 0:
        probs.append("load.host_mapped_tensors must be > 0")
    want = phase_expectations(camp, arm, phase)
    if s.get("cell_order") != want:
        probs.append("cell_order %r != %r" % (s.get("cell_order"), want))
    return probs


def run_phase(args, host=None):
    host = host or Host()
    camp = load_campaign(args.root, args.expect_campaign_sha256)
    check_arm_phase(args.arm, args.phase)
    if args.arm not in camp["arms"]:
        raise Refusal("arm %s is not in this campaign (%s)" % (args.arm, sorted(camp["arms"])))
    arm, phase = args.arm, args.phase
    ent = camp["arms"][arm]["runs"][phase]
    rid = run_id(arm, phase)
    device = camp_device(camp)
    arch = device.get("arch", DEFAULT_ARCH)
    recipe = camp.get("execution_recipe")
    recipe_home = (recipe or {}).get("verified_cache", {}).get("home") if recipe else None
    resources = (recipe or {}).get("resources") or {}
    cap_limit = resources.get("child_cap_secs", CHILD_CAP_SECS)
    drain_limit = resources.get("drain_vram_bytes", DRAIN_VRAM_BYTES)
    mem_limit = resources.get("min_mem_available_kib", MIN_MEM_AVAILABLE_KIB)
    env_h = host.environ()
    if not env_h.get("FABRIC_JOB") or device["card"] not in str(env_h.get("FABRIC_CARDS", "")):
        raise Refusal("run-phase must run inside an admitted %s fabric hold (FABRIC_JOB/FABRIC_CARDS missing)"
                      % device["card"])
    cap = min(args.cap_secs, cap_limit)
    if cap <= 0:
        raise Refusal("--cap-secs must be > 0")
    bad = verify_staging(camp, rehash=True) + verify_model_stat(camp) + verify_beta(camp)
    if recipe is not None:
        bad += verify_verified_cache(recipe)
    if bad:
        raise Refusal("identity preconditions failed: " + "; ".join(bad))
    if canonical(json.load(open(ent["manifest"]["path"]))) != canonical(native_manifest(camp, arm, phase)):
        raise Refusal("on-disk manifest differs from the one derived from the sealed campaign")
    for p in (ent["run_dir"], ent["out_dir"]):
        if os.path.exists(p):
            raise Refusal("refusing existing %s (fresh phase/process only; no retry)" % p)
    ok, probs = device_state(host, device)
    if not ok:
        raise Refusal("DEVICE: " + "; ".join(probs))
    used = vram_used(host, device)
    if used < 0 or used >= drain_limit:
        raise Refusal("card not idle: vram used %d" % used)
    avail = mem_available_kib(host)
    if avail < mem_limit:
        raise Refusal("MemAvailable %d KiB < %d KiB" % (avail, mem_limit))

    run_dir = ent["run_dir"]
    os.makedirs(os.path.join(run_dir, "home", ".hipfire"))
    os.mkdir(os.path.join(run_dir, "lockdir"))
    os.mkdir(os.path.join(run_dir, "kcache"))
    hh = os.path.join(run_dir, "home", ".hipfire")
    with open(camp["config"]["path"], "rb") as f:
        write_new(os.path.join(hh, "config.toml"), f.read())
    os.symlink(os.path.dirname(os.path.realpath(camp["model"]["path"])), os.path.join(hh, "models"))
    pack_kernels_root = os.path.dirname(os.path.dirname(camp["pack"]["dir"]))
    pack_link = os.path.join(hh, "bin", "kernels")
    os.mkdir(os.path.join(hh, "bin"))
    os.symlink(pack_kernels_root, pack_link)
    if os.path.realpath(pack_link) != os.path.realpath(pack_kernels_root):
        raise Refusal("private HOME cold-lookup symlink %s does not resolve to the sealed kernels root" % pack_link)
    if os.path.realpath(os.path.join(pack_link, "compiled", arch)) != os.path.realpath(camp["pack"]["dir"]):
        raise Refusal("private HOME cold-lookup symlink does not expose the sealed pack dir")
    env, unset, applied = process_env(env_h, arm, run_dir, device, recipe_home)
    if applied != json.load(open(ent["manifest"]["path"]))["launch"]["env"]["set"]:
        raise Refusal("applied env differs from the manifest launch.env.set")
    argv = ent["argv"]
    pre = {"label": rid, "host": os.uname().nodename, "vram_used_before": used, "mem_available_kib_before": avail,
           "fabric_job": env_h.get("FABRIC_JOB"), "fabric_cards": env_h.get("FABRIC_CARDS"),
           "boot_id": open("/proc/sys/kernel/random/boot_id").read().strip(), "start_unix": time.time(),
           "device_ok": True}
    write_new(os.path.join(run_dir, "preflight.json"), pretty(pre))
    write_new(os.path.join(run_dir, "command.txt"), (" ".join(shlex.quote(x) for x in argv) + "\n").encode())
    write_new(os.path.join(run_dir, "env-applied.json"), pretty({"set": applied, "unset": unset}))

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, _on_signal)
    Stop.flag = None
    failures, device_lost, leftover, rc, reason = [], False, [], None, None
    t0 = host.monotonic()
    so = open(os.path.join(run_dir, "stdout.log"), "wb")
    se = open(os.path.join(run_dir, "stderr.log"), "wb")
    child = subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL, stdout=so, stderr=se, close_fds=True,
                             start_new_session=True)
    pgid = child.pid
    write_new(os.path.join(run_dir, "child.pgid"), ("%d\n" % pgid).encode())
    try:
        last_dev = t0
        while True:
            rc = child.poll()
            if rc is not None:
                break
            now = host.monotonic()
            if Stop.flag:
                reason = "SIGNAL:" + Stop.flag
            elif now - t0 > cap:
                reason = "TIMEOUT: exceeded %ds" % cap
            elif now - last_dev >= 5:
                last_dev = now
                dok, dprobs = device_state(host)
                if not dok:
                    device_lost, reason = True, "DEVICE_LOST: " + "; ".join(dprobs)
            if reason:
                failures.append(reason)
                break
            host.sleep(1)
    finally:
        if child.poll() is None or group_members(pgid):
            left = terminate_group(pgid, KILL_GRACE_SECS)
            if left:
                leftover = ["%d %s" % m for m in left]
        try:
            rc = child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            rc = None
        so.close()
        se.close()
    # own-group drain, then the explicit post-exit group scan
    members = group_members(pgid)
    if members:
        left = terminate_group(pgid, KILL_GRACE_SECS)
        leftover = ["%d %s" % m for m in (left or members)]
        failures.append("LIFECYCLE_LEFTOVER: own group %d still live after the CLI exit" % pgid)
    if rc != 0 and not reason:
        failures.append("child exit %r" % (rc,))
    dok, dprobs = device_state(host)
    if not dok:
        device_lost = True
        failures.append("DEVICE_LOST after phase: " + "; ".join(dprobs))
    drained = None
    if not device_lost:
        for _ in range(90):
            drained = vram_used(host)
            if 0 <= drained < DRAIN_VRAM_BYTES:
                break
            host.sleep(1)
        else:
            failures.append("VRAM did not drain below 2 GiB (last %r)" % drained)
    kfiles = []
    for d, _s, ns in os.walk(os.path.join(run_dir, "kcache")):
        kfiles += [os.path.relpath(os.path.join(d, n), run_dir) for n in ns]
    problems = verify_phase_dir(camp, arm, phase) if rc == 0 and not device_lost else ["phase not verified (child failed)"]
    pack_ev = pack_binding_evidence(open(os.path.join(run_dir, "stderr.log"), "r", errors="replace").read(),
                                    kfiles, camp["pack"]["dir"], os.path.join(run_dir, "kcache"),
                                    camp["pack"].get("files"))
    if rc == 0 and not device_lost:
        problems += ["pack binding: " + p for p in pack_ev["problems"]]
    if rc == 0 and not device_lost and problems:
        failures += ["phase output: " + p for p in problems]
    status = "INCOMPLETE" if device_lost else ("OK" if not failures else "FAILED")
    rec = {"schema": RUN_RECORD_SCHEMA, "id": rid, "arm": arm, "phase": phase, "status": status, "exit_code": rc,
           "failures": failures, "device_lost": device_lost, "leftover": leftover, "argv": argv,
           "env": {"set": applied, "unset": unset}, "fabric_job": env_h.get("FABRIC_JOB"),
           "wall_seconds": host.monotonic() - t0, "vram_after": drained, "end_unix": time.time(),
           "stderr_sha256": sha256_file(os.path.join(run_dir, "stderr.log")),
           "kcache_files": sorted(kfiles), "pack_install": {"link": pack_link, "target": pack_kernels_root},
           "pack_binding": {k: pack_ev[k] for k in ("cold_lookup_dirs", "pack_hits", "pack_verified", "jit_objects",
                                                    "fallback_markers", "ok")},
           "retry": "never; a failed or INCOMPLETE phase is not rerun automatically"}
    write_new(os.path.join(run_dir, "run-record.json"), pretty(rec))
    if device_lost:
        write_new(os.path.join(run_dir, "INCOMPLETE"), b"device disappeared; stop, no retry\n")
    print(json.dumps({k: rec[k] for k in ("id", "status", "exit_code", "failures")}, sort_keys=True))
    return 0 if status == "OK" else (3 if device_lost else 1)

# -------------------------------------------------------------- score / report

def score(args):
    camp = load_campaign(args.root, args.expect_campaign_sha256)
    check_arm_phase(args.arm)
    if args.arm not in camp["arms"]:
        raise Refusal("arm %s is not in this campaign (%s)" % (args.arm, sorted(camp["arms"])))
    bad = verify_staging(camp, rehash=True) + verify_beta(camp)
    if bad:
        raise Refusal("identity preconditions failed: " + "; ".join(bad))
    a = camp["arms"][args.arm]
    s1a_only = bool(getattr(args, "s1a_only", False))
    tag = args.arm + (".s1a-only" if s1a_only else "")
    for ph in (("s1a",) if s1a_only else PHASES):
        p = verify_phase_dir(camp, args.arm, ph)
        if p:
            raise Refusal("%s %s: %s" % (args.arm, ph, "; ".join(p)))
    out = os.path.join(camp["root"], "score", tag)
    if os.path.exists(out):
        raise Refusal("refusing existing %s" % out)
    argv = [camp["exe"]["path"], "--phase", "score", "--plan", a["plan"]["path"], "--baseline-dir", camp["beta"]["dir"],
            "--s1a-dir", a["runs"]["s1a"]["out_dir"]]
    if not s1a_only:   # the s1b repeat feeds native determinism.json
        argv += ["--s1b-dir", a["runs"]["s1b"]["out_dir"]]
    argv += ["--out-dir", out]
    home = os.path.join(camp["root"], "cpu-home")
    os.makedirs(home, exist_ok=True)
    p = subprocess.run(argv, env=clean_cpu_env(os.environ, home), stdin=subprocess.DEVNULL, capture_output=True,
                       text=True, timeout=NATIVE_CPU_CAP_SECS)
    logdir = os.path.join(camp["root"], "score-logs")
    os.makedirs(logdir, exist_ok=True)
    write_new(os.path.join(logdir, tag + ".stdout"), p.stdout.encode())
    write_new(os.path.join(logdir, tag + ".stderr"), p.stderr.encode())
    write_new(os.path.join(logdir, tag + ".argv.json"), pretty(argv))
    print("%s: native score exit %d" % (tag, p.returncode))
    return p.returncode


def bridge_check(args):
    """CPU proof of the declared cross-binary bridge. Runs the NEW native executable's `score` over the OLD Q1a
    candidate (`s1a` next to the frozen beta) with the ARM plan (differs from the Q1a plan only in expect.env), and
    requires every pair summary to be bit-identical to the archived Q1a score. A guard bypass or a plan the native
    rejects shows up as a refusal/difference. Output: ROOT/bridge/<arm>/ and bridge-<arm>.json."""
    camp = load_campaign(args.root, args.expect_campaign_sha256)
    check_arm_phase(args.arm)
    if args.arm not in camp["arms"]:
        raise Refusal("arm %s is not in this campaign (%s)" % (args.arm, sorted(camp["arms"])))
    bad = verify_staging(camp, rehash=True) + verify_beta(camp)
    if bad:
        raise Refusal("identity preconditions failed: " + "; ".join(bad))
    base = os.path.dirname(camp["beta"]["dir"])
    old_s1a = os.path.join(base, "s1a")
    old_score = os.path.normpath(os.path.join(base, "..", "score", "kl-summary.json"))
    for p_ in (os.path.join(old_s1a, "phase-summary.json"), old_score):
        if not os.path.exists(p_):
            raise Refusal("bridge needs the archived Q1a %s" % p_)
    out = os.path.join(camp["root"], "bridge", args.arm)
    if os.path.exists(out):
        raise Refusal("refusing existing %s" % out)
    argv = [camp["exe"]["path"], "--phase", "score", "--plan", camp["arms"][args.arm]["plan"]["path"],
            "--baseline-dir", camp["beta"]["dir"], "--s1a-dir", old_s1a, "--out-dir", out]
    home = os.path.join(camp["root"], "cpu-home")
    os.makedirs(home, exist_ok=True)
    p = subprocess.run(argv, env=clean_cpu_env(os.environ, home), stdin=subprocess.DEVNULL, capture_output=True,
                       text=True, timeout=NATIVE_CPU_CAP_SECS)
    rec = {"argv": argv, "exit": p.returncode, "stderr_tail": p.stderr[-400:], "archived_score": old_score,
           "candidate_binary_new": camp["exe"]["sha256"], "reference_binary_old": camp["beta"]["binary_sha256"],
           "pairs_equal": None, "ok": False}
    if p.returncode == 0:
        new = {x["pair"]: x for x in read_json(os.path.join(out, "kl-summary.json"))["pairs"]}
        old = {x["pair"]: x for x in read_json(old_score)["pairs"]}
        diffs = [k for k in sorted(old) if new.get(k) != old[k]]
        rec["pairs_equal"] = not diffs and sorted(new) == sorted(old)
        rec["differing_pairs"] = diffs
        rec["ok"] = rec["pairs_equal"]
    write_new(os.path.join(camp["root"], "bridge-%s.json" % args.arm), pretty(rec))
    print(json.dumps({k: rec[k] for k in ("exit", "ok", "pairs_equal")}, sort_keys=True))
    return 0 if rec["ok"] else 1


# ---- evidence ------------------------------------------------------------

RE_LOAD_SET = re.compile(r"qwen4 slowlink load: .*?resident_layer_set=\[([0-9, ]*)\]")
RE_LOAD = re.compile(r"qwen4 slowlink load: placement=(\S+) resident layers=(\d+)")
RE_QSA_RESERVED = re.compile(r"qwen4 QSA gather workspace: (\d+) MiB reserved for (\d+) context tokens")
RE_QSA_MAPPED = re.compile(r"qwen4 QSA context mapped for (\d+) tokens: .*?, (\d+) MiB gather workspace")
RE_QSA_PROFILE = re.compile(r"QSA workspace (\d+) KiB")
QSA_MODULE = "indexed_attention_gathered_wmma"
RE_QSA_MODULE_LINE = re.compile(r"^\s*indexed_attention_gathered_wmma: indexed package verified: (\S+)", re.M)
RE_PACK_HIT = re.compile(r"^\s*([A-Za-z0-9_]+): indexed package verified: (\S+)", re.M)
RE_PRECOMPILED_DIR = re.compile(r"^\s*pre-compiled kernels: (\S+)", re.M)
RE_PACK_FALLBACK = re.compile(r"packaged object rejected|falling back to JIT|falling back to install dir reads|hot-path seed failed|no cached build matches; recompiling|index missing for installed object|no usable cached kernel image|unverified kernel image|JIT worker panicked|the cached kernel image is invalid")
IU4_ROUTE = "aggregate-sym-iu4"


def layer_major_receipts(cells):
    out = []
    for c in cells:
        for r in (c.get("prefill") or {}).get("receipts") or []:
            if isinstance(r, dict) and r.get("route") == "layer-major":
                out.append((c.get("cell"), r))
    return out


def pack_binding_evidence(stderr_text, kcache_files, sealed_pack_dir, kcache_dir, pack_files=None):
    """Fail-closed runtime kernel-pack binding of one phase: every kernel object the phase actually
    resolved must be an indexed pack hit under the sealed pack dir, bound by the sealed index/object
    SHA, with a nonempty hit set; any runtime JIT, compiler fallback or unbound identity fails.
    Sealed stage-manifest equality alone never passes this. `pack_files` is the sealed campaign
    `pack.files` list (rel/size/sha256); `kcache_files` should be a live rescan of the private
    kernel cache, not trusted recorded names alone."""
    sealed = os.path.realpath(sealed_pack_dir)
    hot = os.path.realpath(kcache_dir)
    sealed_by_rel = {}
    if pack_files:
        for f in pack_files:
            sealed_by_rel[f["rel"]] = f
    problems = []
    lookups = RE_PRECOMPILED_DIR.findall(stderr_text)
    if not lookups:
        problems.append("no `pre-compiled kernels:` lookup line in stderr: the cold lookup path is unproven")
    for d in lookups:
        if os.path.realpath(d) not in (sealed, hot):
            problems.append("cold lookup dir %r is neither the sealed pack dir nor the phase-private kernel cache" % d)
    hits = RE_PACK_HIT.findall(stderr_text)
    if not hits:
        problems.append("no `indexed package verified:` hits in stderr: no used-code pack identity was proven")
    verified = 0
    for mod, p in hits:
        real = os.path.realpath(p)
        if os.path.dirname(real) != sealed:
            problems.append("indexed pack hit outside the sealed pack dir: %s" % p)
            continue
        want = (mod + ".hsaco", mod + ".index.json")
        if os.path.basename(real) != want[0]:
            problems.append("indexed pack hit %s is not the module object %s" % (p, want[0]))
            continue
        idx_path = os.path.join(sealed, want[1])
        try:
            with open(idx_path, "r", encoding="utf-8") as f:
                idx = json.load(f)
        except (OSError, ValueError) as e:
            problems.append("%s unreadable: %s" % (want[1], e))
            continue
        if idx.get("module") != mod:
            problems.append("%s names module %r, expected %r" % (want[1], idx.get("module"), mod))
            continue
        try:
            digest = sha256_file(real)
        except OSError as e:
            problems.append("pack object %s unreadable: %s" % (want[0], e))
            continue
        if digest != idx.get("object_sha256"):
            problems.append("pack object %s sha256 %s != index object_sha256 %s" % (want[0], digest, idx.get("object_sha256")))
            continue
        if pack_files is not None:
            for name, got, field in ((want[0], digest, "object"), (want[1], sha256_file(idx_path), "index")):
                ent = sealed_by_rel.get(name)
                if ent is None:
                    problems.append("sealed pack manifest has no %s entry for reported module %s" % (field, mod))
                elif ent["sha256"] != got:
                    problems.append("sealed %s %s sha256 %s != live %s" % (field, name, ent["sha256"], got))
        verified += 1
    if hits and not verified:
        problems.append("no reported indexed hit verified against its sealed index/object SHA")
    live_jits = []
    try:
        for d, _s, ns in os.walk(kcache_dir):
            live_jits += [os.path.relpath(os.path.join(d, n), kcache_dir) for n in ns if n.endswith(".hsaco")]
    except OSError as e:
        problems.append("private kernel cache unreadable: %s" % e)
    rec_jits = sorted({f for f in (kcache_files or []) if f.endswith(".hsaco")})
    if sorted(live_jits) != rec_jits:
        problems.append("recorded kcache names %r != live rescan %r" % (rec_jits[:8], sorted(live_jits)[:8]))
    if live_jits:
        problems.append("phase-private kernel cache holds %d compiled object(s) (runtime JIT/pack miss): %s"
                        % (len(live_jits), ", ".join(sorted(live_jits)[:8])))
    fb = sorted(set(RE_PACK_FALLBACK.findall(stderr_text)))
    if fb:
        problems.append("runtime compiler fallback markers in stderr: %s" % "; ".join(fb[:8]))
    return {"cold_lookup_dirs": lookups[:4], "pack_hits": len(hits), "pack_verified": verified,
            "jit_objects": sorted(live_jits)[:8], "fallback_markers": fb[:8], "ok": not problems, "problems": problems}

def route_evidence(arm, cells, stderr_text, kcache_files, *, phase="s1a", expected_cases=(), camp=None,
                   arch=None, want_resident=None):
    """ACTUAL route evidence of one phase from raw receipts, the loader log and the private kernel cache.
    Never inferred from the requested env. Counts/arch parameterize the checks: the recipe expect gives
    the genuine expected resident placement (both arms of the iu4-only/off pair use the SAME placement),
    and the recipe device gives the arch. Raw actual-CO checks are retained."""
    arch = arch or (camp_arch(camp) if camp else DEFAULT_ARCH)
    recipe_exp = ((camp.get("execution_recipe") or {}).get("expect")) if camp else None
    want_n = want_resident if want_resident is not None else (recipe_exp["resident_layers"] if recipe_exp
                                                              else RESIDENT_LAYERS)
    on = ARMS[arm]
    if phase not in PHASES:
        raise Refusal("unknown route-evidence phase %r" % phase)
    recs = layer_major_receipts(cells)
    ev = {"layer_major_receipts": len(recs)}
    common = []
    if not recs:
        common.append("no layer-major receipts")
    for cell, r in recs:
        if r.get("status") != "ok":
            common.append("%s: receipt status %r" % (cell, r.get("status")))
        if r.get("resident_layers") != want_n or r.get("tile_rows") != S1_TILE_ROWS \
                or r.get("retained_scratch_rows") != S1_TILE_ROWS:
            common.append("%s: resident/tile %r/%r/%r" % (cell, r.get("resident_layers"), r.get("tile_rows"),
                                                          r.get("retained_scratch_rows")))
    present = {c for c, _r in recs}
    required = ("primary-ctx16k-teacher", "primary-ctx4k-teacher") if phase == "s1a" else ()
    for need in required:
        if need not in present:
            common.append("%s has no layer-major receipt" % need)
    if phase == "s1b" and not expected_cases:
        common.append("s1b requires the plan's determinism case ids")
    for cid in expected_cases:
        name = "det-" + cid
        matches = [c for c in cells if c.get("cell") == name]
        if len(matches) != 1:
            common.append("%s: expected exactly one determinism cell" % name)
            continue
        prefill = matches[0].get("prefill") or {}
        calls, receipts = prefill.get("calls"), prefill.get("receipts")
        if not isinstance(calls, list) or not calls or not isinstance(receipts, list) or len(calls) != len(receipts):
            common.append("%s: missing or mismatched calls/receipts" % name)
            continue
        for call, receipt in zip(calls, receipts):
            rows = call.get("rows")
            if type(rows) is not int or rows <= 0:
                common.append("%s: invalid call rows" % name)
                continue
            want_route = "layer-major" if rows >= 1537 else "incumbent"
            if receipt.get("route") != want_route:
                common.append("%s: %d-row call route %r != %s" % (name, rows, receipt.get("route"), want_route))
    ev["phase"] = phase
    ev["common"] = {"ok": not common, "problems": common}

    # PLACE: the actual set (receipts derive it from the per-layer mappings) and the loader line.
    # Recipe expect gives the genuine expected set; off/iu4 share the same front placement.
    recipe_set = recipe_exp.get("resident_layer_set") if recipe_exp else None
    want = recipe_set if ("place" in on and recipe_set is not None) else (
        PLACE_LAYERS if "place" in on else list(range(want_n)))
    sets = sorted({tuple(r["resident_layer_set"]) if isinstance(r.get("resident_layer_set"), list) else None
                   for _c, r in recs}, key=str)
    m = RE_LOAD_SET.search(stderr_text)
    logged = [int(x) for x in m.group(1).split(",") if x.strip()] if m else None
    lm = RE_LOAD.search(stderr_text)
    place_problems = []
    if sets != [tuple(want)]:
        place_problems.append("receipt resident_layer_set values %r != [%r]" % (sets, tuple(want)))
    if "place" in on and logged != want:
        place_problems.append("loader log resident_layer_set %r != %r" % (logged, want))
    if "place" not in on and m is not None:
        place_problems.append("loader log reports an explicit resident_layer_set on a front-placement arm")
    if lm is None or lm.group(1) != "slowlink-profile" or int(lm.group(2)) != want_n:
        place_problems.append("loader log lacks placement=slowlink-profile resident layers=%d" % want_n)
    ev["place"] = {"on_expected": "place" in on, "expected_set": want, "receipt_sets": [list(s) if s else None for s in sets],
                   "loader_set": logged, "ok": not place_problems, "problems": place_problems}

    # IU4: raw receipt moe_iu4 object.
    iu_problems, routes, agg = [], set(), {"aggregate_iu4": 0, "aggregate_wmma": 0, "aggregate_f32_tail": 0,
                                           "slab_k2": 0, "k1_or_decode": 0}
    for cell, r in recs:
        mo = r.get("moe_iu4")
        if not isinstance(mo, dict):
            iu_problems.append("%s: receipt lacks the moe_iu4 object" % cell)
            continue
        routes.add(mo.get("window_route"))
        for k in agg:
            v = (mo.get("tiles") or {}).get(k)
            if isinstance(v, int):
                agg[k] += v
        if "iu4" in on:
            hv = mo.get("final_headers_verified")
            desc = mo.get("descriptor")
            if mo.get("window_route") != IU4_ROUTE:
                iu_problems.append("%s: window_route %r != %s" % (cell, mo.get("window_route"), IU4_ROUTE))
            if mo.get("requested") is not True:
                iu_problems.append("%s: requested is not true" % cell)
            if not isinstance(desc, dict) or desc.get("arch") != arch:
                iu_problems.append("%s: descriptor missing / not %s" % (cell, arch))
            if mo.get("source_headers_symmetric") is not True:
                iu_problems.append("%s: source headers not symmetric" % cell)
            if not (isinstance(hv, list) and len(hv) == 2 and hv[0] == hv[1] and hv[0] > 0):
                iu_problems.append("%s: final_headers_verified %r" % (cell, hv))
        else:
            if mo.get("window_route") == IU4_ROUTE or mo.get("requested") is True:
                iu_problems.append("%s: IU4 engaged on a non-IU4 arm" % cell)
    if "iu4" in on:
        if agg["aggregate_iu4"] <= 0:
            iu_problems.append("no aggregate_iu4 tiles executed")
        if agg["aggregate_wmma"] != 0:
            iu_problems.append("aggregate_wmma tiles %d on the IU4 route" % agg["aggregate_wmma"])
    elif agg["aggregate_iu4"] != 0:
        iu_problems.append("aggregate_iu4 tiles %d on a non-IU4 arm" % agg["aggregate_iu4"])
    ev["iu4"] = {"on_expected": "iu4" in on, "window_routes": sorted(str(x) for x in routes), "tile_totals": agg,
                 "ok": not iu_problems, "problems": iu_problems}

    # QSA: no per-call dispatch counter exists. Source-grounded evidence (gfx1100 Q8, QSA=1):
    #  * workspace line `qwen4 QSA gather workspace: N MiB reserved ...` is LOAD-TIME config (gathered format enabled),
    #    not a dispatch;
    #  * `<module>: indexed package verified: <pack>/indexed_attention_gathered_wmma.hsaco` is printed by
    #    compile_for_symbol the first time `qsa_gathered_wmma_launch_with` calls ensure_kernel (lazy, first gathered
    #    launch; no walked route plan exists for gfx1100 so there is no generic prewarm). A pack hit writes NOTHING to
    #    the kernel cache, so a kcache file appears only on a pack miss (hipcc fallback) and is a secondary witness.
    # Neither counts launches nor proves every tile took the route; the strength label says so.
    reserved = [(int(a), int(b)) for a, b in RE_QSA_RESERVED.findall(stderr_text)]
    mapped = [(int(a), int(b)) for a, b in RE_QSA_MAPPED.findall(stderr_text)]
    profile_kib = [int(x) for x in RE_QSA_PROFILE.findall(stderr_text)]
    reserved_pos = [x for x in reserved if x[0] > 0] + [x for x in mapped if x[1] > 0]
    module_lines = RE_QSA_MODULE_LINE.findall(stderr_text)
    module_files = [f for f in kcache_files if QSA_MODULE in f]
    resolved = bool(module_lines or module_files)
    qsa_problems = []
    if "qsa" in on:
        if not reserved_pos and not any(k > 0 for k in profile_kib):
            qsa_problems.append("no positive QSA gather workspace line in the log")
        if not resolved:
            qsa_problems.append("gathered-WMMA module was never resolved (no `indexed package verified` line, no "
                                "kcache object): the gather launch path was not entered")
    else:
        if reserved_pos:
            qsa_problems.append("QSA gather workspace reserved on a non-QSA arm: %r" % reserved_pos[:3])
        if resolved:
            qsa_problems.append("gathered-WMMA module resolved on a non-QSA arm")
    ev["qsa"] = {"on_expected": "qsa" in on, "workspace_reserved_lines": reserved[:8], "context_mapped_lines": mapped[:8],
                 "profile_qsa_workspace_kib": profile_kib[:4], "gathered_module_pack_lines": module_lines[:4],
                 "gathered_module_in_private_kcache": module_files[:8],
                 "strength": ("first-gather-launch module resolution + workspace (no per-call counter exists)"
                              if (reserved_pos and resolved) else
                              "workspace-only: load-time config, NOT proof of dispatch" if reserved_pos else "none"),
                 "ok": not qsa_problems, "problems": qsa_problems}
    ev["ok"] = all(ev[k]["ok"] for k in ("common", "place", "iu4", "qsa"))
    return ev


def receipt_signature(cells):
    """Route signature of the determinism cells' receipts, for the same-route a/b comparison."""
    sig = {}
    for c in cells:
        if not str(c.get("cell", "")).startswith("det-"):
            continue
        rows = []
        for r in (c.get("prefill") or {}).get("receipts") or []:
            mo = r.get("moe_iu4") if isinstance(r.get("moe_iu4"), dict) else {}
            rows.append([r.get("route"), r.get("resident_layers"), r.get("resident_layer_set"), r.get("tile_rows"),
                         mo.get("window_route"), mo.get("tiles")])
        sig[c["cell"]] = rows
    return sig


def determinism_report(det, expected_cases, cells_a, cells_b):
    """det = native determinism.json (s1a vs s1b). Token equality and bit identity reported separately."""
    probs = []
    if det.get("schema") != "hipfire.qwen4.slowlink_quality.determinism.v1":
        probs.append("invalid native determinism schema")
    if det.get("all_identical") is not True:
        probs.append("native all_identical is not true")
    native_cases = det.get("cases")
    if not isinstance(native_cases, list) or any(not isinstance(c, dict) or not isinstance(c.get("case"), str)
                                                 for c in native_cases):
        native_cases = []
        probs.append("invalid native cases array")
    cases = {c["case"]: c for c in native_cases}
    if len(cases) != len(native_cases):
        probs.append("duplicate native determinism cases")
    if sorted(cases) != sorted(expected_cases):
        probs.append("determinism cases %r != plan cases %r" % (sorted(cases), sorted(expected_cases)))
    out = []
    by_a = {c["cell"]: c for c in cells_a}
    by_b = {c["cell"]: c for c in cells_b}
    for cid in expected_cases:
        c = cases.get(cid, {})
        finish = c.get("finish_reason")
        finish_equal = (isinstance(finish, list) and len(finish) == 2
                        and all(isinstance(x, str) for x in finish) and finish[0] == finish[1])
        row = {"case": cid, "token_ids_equal": c.get("token_ids_equal"), "finish_reason": finish,
               "finish_equal": finish_equal, "bit_identical": c.get("identical"),
               "first_differing_row": c.get("first_differing_row")}
        if c.get("token_ids_equal") is not True or not finish_equal:
            probs.append("%s: token ids/finish differ or native finish_reason is not a two-string pair" % cid)
        if c.get("identical") is not True:
            probs.append("%s: logits not bit identical (reported; stricter than token equality)" % cid)
        cell = "det-" + cid
        hs = c.get("stream_sha256")
        if not isinstance(hs, list) or len(hs) != 2 or not all(isinstance(x, str) for x in hs):
            probs.append("%s: invalid native stream_sha256 pair" % cid)
        elif by_a.get(cell, {}).get("stream_sha256") != hs[0] or by_b.get(cell, {}).get("stream_sha256") != hs[1]:
            probs.append("%s: determinism.json stream hashes differ from the phase summaries" % cid)
        if by_a.get(cell, {}).get("text") != by_b.get(cell, {}).get("text"):
            probs.append("%s: decoded text differs between s1a and s1b" % cid)
        out.append(row)
    sa, sb = receipt_signature(cells_a), receipt_signature(cells_b)
    if sa != sb:
        probs.append("route signature (receipts) differs between s1a and s1b")
    return {"all_identical_native": det.get("all_identical"), "cases": out, "route_signature_equal": sa == sb,
            "ok": not probs, "problems": probs}


def text_evidence(cells_a, cells_b):
    """Decoded text of every greedy/det cell for human review + cheap degeneration flags (NOT a coherence proof)."""
    items = []
    for phase, cells in (("s1a", cells_a), ("s1b", cells_b)):
        for c in cells:
            if "text" not in c:
                continue
            ids = c.get("token_ids") if isinstance(c.get("token_ids"), list) else None
            text = c.get("text") or ""
            flags = []
            if not text.strip():
                flags.append("empty_text")
            if "\ufffd" in text:
                flags.append("replacement_char")
            if ids:
                if len(set(ids)) / len(ids) < 0.25:
                    flags.append("low_unique_token_ratio")
                run = best = 1
                for x, y in zip(ids, ids[1:]):
                    run = run + 1 if x == y else 1
                    best = max(best, run)
                if best >= 8:
                    flags.append("repeat_run_%d" % best)
            items.append({"phase": phase, "cell": c["cell"], "finish_reason": c.get("finish_reason"),
                          "token_count": len(ids) if ids else None, "token_ids": ids, "text": text,
                          "auto_flags": flags})
    return {"manual_review_required": True,
            "note": "automatic flags are cheap degeneration screens; coherence must be read by a human",
            "auto_flag_cells": [i["cell"] + "@" + i["phase"] for i in items if i["auto_flags"]], "cells": items}


def load_cells(out_dir):
    cells = []
    with open(os.path.join(out_dir, "cells.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cells.append(json.loads(line))
    return cells


def report(args):
    recorded = args.recorded_harness_dir
    if recorded is not None and args.out_dir is None:
        raise Refusal("historical CPU report requires a fresh explicit --out-dir")
    camp = load_campaign(args.root, args.expect_campaign_sha256, recorded_harness_dir=recorded, report_only=True)
    root = camp["root"]
    outdir = need_abs(args.out_dir, "--out-dir") if args.out_dir else os.path.join(root, "report")
    if os.path.exists(outdir):
        raise Refusal("refusing existing %s" % outdir)
    arms = [a for a in args.arms.split(",")] if args.arms else list(camp["arms"])
    for a in arms:
        if a not in camp["arms"]:
            raise Refusal("arm %s is not in this campaign (%s)" % (a, sorted(camp["arms"])))
    bad = verify_beta(camp)
    if bad:
        raise Refusal("; ".join(bad))
    beta_cells = {n: c["stream_sha256"] for n, c in camp["beta"]["cells"].items()}
    summaries, klrecs, cand_cells, per_arm = {}, {}, {}, {}
    plan = read_json(camp["arms"][arms[0]]["plan"]["path"])
    expected_cases = plan["determinism"]["case_ids"]
    for arm in arms:
        a = camp["arms"][arm]
        sdir = os.path.join(root, "score", arm)
        summaries[arm] = read_json(os.path.join(sdir, "kl-summary.json"))
        klrecs[arm] = qs.load_jsonl(os.path.join(sdir, "kl.jsonl"))
        cells_a = load_cells(a["runs"]["s1a"]["out_dir"])
        cand_cells[arm] = {c["cell"]: c.get("stream_sha256") for c in cells_a}
    try:
        identity = qs.check_reference_identity(beta_cells, summaries, cand_cells)
        for arm in arms:
            qs.check_summary_matches_rows(summaries[arm], klrecs[arm], arm)
        identity_ok, identity_err = True, None
    except qs.Invalid as e:
        identity, identity_ok, identity_err = None, False, str(e)
    for arm in arms:
        a = camp["arms"][arm]
        probs = verify_phase_dir(camp, arm, "s1a") + verify_phase_dir(camp, arm, "s1b")
        cells_a = load_cells(a["runs"]["s1a"]["out_dir"])
        cells_b = load_cells(a["runs"]["s1b"]["out_dir"])
        rr = {ph: read_json(os.path.join(a["runs"][ph]["run_dir"], "run-record.json")) for ph in PHASES}
        ev = {}
        for ph, cells in (("s1a", cells_a), ("s1b", cells_b)):
            stderr = open(os.path.join(a["runs"][ph]["run_dir"], "stderr.log"), "r", errors="replace").read()
            ev[ph] = route_evidence(arm, cells, stderr, rr[ph].get("kcache_files", []),
                                    phase=ph, expected_cases=expected_cases, camp=camp)
            ev[ph]["run_status"] = rr[ph]["status"]
            pack_ev = pack_binding_evidence(stderr, rr[ph].get("kcache_files", []), camp["pack"]["dir"],
                                            os.path.join(a["runs"][ph]["run_dir"], "kcache"), camp["pack"].get("files"))
            ev[ph]["pack_binding"] = pack_ev
            probs += ["%s pack binding: %s" % (ph, p) for p in pack_ev["problems"]]
            if rr[ph]["status"] != "OK":
                probs.append("%s run-record status %s" % (ph, rr[ph]["status"]))
        det = read_json(os.path.join(root, "score", arm, "determinism.json"))
        dr = determinism_report(det, expected_cases, cells_a, cells_b)
        te = text_evidence(cells_a, cells_b)
        write_new(os.path.join(outdir, arm, "text-evidence.json"), pretty(te))
        write_new(os.path.join(outdir, arm, "route-evidence.json"), pretty(ev))
        write_new(os.path.join(outdir, arm, "determinism-report.json"), pretty(dr))
        try:
            absolute = qs.analyze_absolute(klrecs[arm])
            gate = qs.evaluate_gate(absolute)
        except qs.Invalid as e:
            absolute, gate = {"status": "invalid", "error": str(e)}, None
        entry = {"arm": arm, "levers_on": list(ARMS[arm]), "phase_problems": probs, "absolute_kl_vs_beta": absolute,
                 "kl_gate": gate, "route_ok": ev["s1a"]["ok"] and ev["s1b"]["ok"], "determinism_ok": dr["ok"],
                 "pack_ok": ev["s1a"]["pack_binding"]["ok"] and ev["s1b"]["pack_binding"]["ok"],
                 "text_auto_flag_cells": te["auto_flag_cells"], "text_manual_review_required": True,
                 "native_kl_summary_raw": summaries[arm],
                 "run_wall_seconds": {ph: rr[ph]["wall_seconds"] for ph in PHASES}}
        if arm != "off" and "off" in klrecs:
            try:
                entry["delta_diagnostic_vs_off"] = qs.analyze_delta_diagnostic(klrecs["off"], klrecs[arm])
            except qs.Invalid as e:
                entry["delta_diagnostic_vs_off"] = {"status": "invalid", "error": str(e)}
        per_arm[arm] = entry
    summary = {"schema": "hipfire.qwen4.slowlink_quality_campaign.report.v1", "campaign_sha256": args.expect_campaign_sha256,
               "reporter": {"harness_dir": HERE, "hashes": harness_hashes(HERE),
                            "source_commit": subprocess.check_output(["git", "-C", HERE, "rev-parse", "HEAD"], text=True).strip(),
                            "source_tree": subprocess.check_output(["git", "-C", HERE, "rev-parse", "HEAD^{tree}"], text=True).strip()},
               "recorded_harness": {"directory": recorded or HERE, "hashes": camp["harness"],
                                    "source_pin": camp["source_pin"], "historical_cpu_report_only": recorded is not None},
               "reference": {"identity_ok": identity_ok, "error": identity_err, "evidence": identity,
                             "beta_dir": camp["beta"]["dir"], "beta_binary_sha256": camp["beta"]["binary_sha256"],
                             "declaration": camp["reference_declaration"],
                             "candidate_binary_sha256": camp["exe"]["sha256"],
                             "arm_plan_sha256": {a: camp["arms"][a]["plan"]["sha256"] for a in arms}},
               "method": {"gate": "absolute KL(beta||arm) point mean per context vs the beta own-pair CI95 upper "
                                  "(exact; declared rounded also shown); pooled reported, not gated",
                          "intervals": "descriptive stratified moving-block bootstrap (block 32, 10000 reps, "
                                       "label %s); never gates" % qs.BOOT_LABEL,
                          "delta": "optional paired diagnostic, no verdict"},
               "arms": {}}
    for arm, e in per_arm.items():
        g, ab = e["kl_gate"], e["absolute_kl_vs_beta"]
        summary["arms"][arm] = {
            "levers_on": e["levers_on"], "route_ok": e["route_ok"], "determinism_ok": e["determinism_ok"],
            "pack_ok": e["pack_ok"],
            "phase_problems": e["phase_problems"], "text_auto_flag_cells": e["text_auto_flag_cells"],
            "absolute_raw_means": ({"ctx16k": ab["contexts"]["ctx16k"]["raw_mean_kl_nats"],
                                    "ctx4k": ab["contexts"]["ctx4k"]["raw_mean_kl_nats"],
                                    "pooled": ab["pooled"]["raw_mean_kl_nats"]} if ab.get("status") == "ok" else None),
            "kl_gate": g,
            "mechanical_pass": bool(g and g["pass"] and identity_ok and e["route_ok"] and e["determinism_ok"]
                                    and e["pack_ok"] and not e["phase_problems"]),
            "text_review": "required (human) before any qualification claim",
        }
    write_new(os.path.join(outdir, "summary.json"), pretty(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if identity_ok else 2


SERIAL_ORDER = [(a, p) for a in ("off", "place", "iu4", "qsa", "stack", "place_iu4") for p in PHASES]


def serial(args, host=None):
    """Strictly serial campaign driver (one GPU phase per fabric job; the next is submitted only after the
    previous finished with an OK run record and the device is still healthy). Restartable: phases with an OK
    run record are skipped; a STOPPED marker must be reviewed and removed by hand before a restart."""
    host = host or Host()
    camp = load_campaign(args.root, args.expect_campaign_sha256)
    sdir = os.path.join(camp["root"], "serial")
    if args.detach:
        os.makedirs(sdir, exist_ok=True)
        if os.path.exists(os.path.join(sdir, "serial.json")):
            raise Refusal("refusing: a serial driver was already started (%s)" % sdir)
        cmd = ["python3", "-B", os.path.realpath(__file__), "serial", "--root", camp["root"],
               "--expect-campaign-sha256", args.expect_campaign_sha256, "--priority", args.priority,
               "--max-secs", str(args.max_secs), "--wrapper", args.wrapper, "--fabric", args.fabric,
               "--max-wait-per-phase", str(args.max_wait_per_phase)]
        log = open(os.path.join(sdir, "serial.log"), "ab")
        p = subprocess.Popen(["nohup", "setsid", "bash", "-c", DETACH_SCRIPT, "launcher", os.path.join(sdir, "driver")] + cmd,
                             stdin=subprocess.DEVNULL, stdout=log, stderr=log, close_fds=True)
        write_new(os.path.join(sdir, "serial.json"), pretty({"pid": p.pid, "argv": cmd, "start_unix": time.time()}))
        print(json.dumps({"serial_pid": p.pid, "dir": sdir}))
        return 0
    os.makedirs(sdir, exist_ok=True)
    stopped = os.path.join(sdir, "STOPPED")
    if os.path.exists(stopped):
        raise Refusal("%s exists: review the stop reason and remove it before restarting" % stopped)

    def note(msg):
        with open(os.path.join(sdir, "serial.log"), "a") as f:
            f.write("%s %s\n" % (utc_now(), msg))

    def stop(reason):
        note("STOP: " + reason)
        write_new(stopped, (reason + "\n").encode())
        return 1

    order = [(a, p) for a, p in SERIAL_ORDER if a in camp["arms"]]
    for arm, ph in order:
        ent = camp["arms"][arm]["runs"][ph]
        rid = run_id(arm, ph)
        rr_path = os.path.join(ent["run_dir"], "run-record.json")
        ldir = os.path.join(camp["root"], "launch", rid)
        if os.path.exists(rr_path):
            status = read_json(rr_path).get("status")
            if status == "OK":
                note("skip %s (OK run record)" % rid)
                continue
            return stop("%s run record status %s (no retry)" % (rid, status))
        if os.path.exists(ent["run_dir"]) or os.path.exists(ldir):
            return stop("%s was launched earlier and has no OK run record (no retry)" % rid)
        marks = [d for d in os.listdir(os.path.join(camp["root"], "runs"))
                 if os.path.exists(os.path.join(camp["root"], "runs", d, "INCOMPLETE"))] \
            if os.path.isdir(os.path.join(camp["root"], "runs")) else []
        if marks:
            return stop("INCOMPLETE marker(s) from %s" % marks)
        dok, dprobs = device_state(host, camp_device(camp))
        if not dok:
            return stop("%s not healthy before submitting %s: %s" % (camp_device(camp)["card"], rid, "; ".join(dprobs)))
        note("launch %s" % rid)
        launch(argparse.Namespace(root=camp["root"], expect_campaign_sha256=args.expect_campaign_sha256, arm=arm,
                                  phase=ph, max_secs=args.max_secs, priority=args.priority,
                                  wrapper=args.wrapper, fabric=args.fabric))
        exit_file = os.path.join(ldir, "watch.exit")
        end = host.monotonic() + args.max_wait_per_phase
        while not os.path.exists(exit_file):
            if host.monotonic() > end:
                return stop("%s: wait timeout; the job is left running (never killed here)" % rid)
            host.sleep(15)
        wexit = open(exit_file).read().strip()
        if not os.path.exists(rr_path):
            return stop("%s: wrapper exit %s but no run record" % (rid, wexit))
        rr = read_json(rr_path)
        if rr.get("status") != "OK" or wexit != "0":
            return stop("%s: run status %s, wrapper exit %s, failures %s" % (rid, rr.get("status"), wexit, rr.get("failures")))
        pg = os.path.join(ent["run_dir"], "child.pgid")
        if os.path.exists(pg) and group_members(int(open(pg).read().strip())):
            return stop("%s: own child group still has live members" % rid)
        note("done %s OK" % rid)
    note("campaign complete")
    return 0


# ------------------------------------------------------------------------ CLI

def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, arm=False, phase=False):
        p.add_argument("--root", required=True)
        p.add_argument("--expect-campaign-sha256", required=True)
        if arm:
            p.add_argument("--arm", required=True, choices=sorted(ARMS))
        if phase:
            p.add_argument("--phase", required=True, choices=PHASES)

    p = sub.add_parser("prepare")
    p.add_argument("--root", required=True)
    p.add_argument("--exe", required=True)
    p.add_argument("--expect-exe-sha256", required=True)
    p.add_argument("--expect-pack-digest")
    p.add_argument("--execution-recipe", help="explicit execution-recipe JSON (device/pack/source/cache/expect/resources)")
    p.add_argument("--config", required=True)
    p.add_argument("--source-pin", required=True)
    p.add_argument("--plan", default=DEFAULT_PLAN)
    p.add_argument("--allow-other-plan", action="store_true")
    p.add_argument("--beta-dir", default=DEFAULT_BETA_DIR)
    p.add_argument("--model-receipt", default=DEFAULT_MODEL_RECEIPT)
    p.add_argument("--ple-receipt", default=DEFAULT_PLE_RECEIPT)
    p.add_argument("--label")
    p.add_argument("--arms", help="comma list of arms to prepare (default off,place_iu4)")
    p.add_argument("--bounds-file", default=DEFAULT_BOUNDS_FILE)
    p = sub.add_parser("check-plans")
    common(p)
    p = sub.add_parser("launch")
    common(p, True, True)
    p.add_argument("--max-secs", type=int, required=True)
    p.add_argument("--priority", required=True)
    p.add_argument("--wrapper", default=DEFAULT_WRAPPER)
    p.add_argument("--fabric", default=DEFAULT_FABRIC)
    p = sub.add_parser("run-phase")
    common(p, True, True)
    p.add_argument("--cap-secs", type=int, default=CHILD_CAP_SECS)
    p = sub.add_parser("wait")
    p.add_argument("--root", required=True)
    p.add_argument("--arm", required=True, choices=sorted(ARMS))
    p.add_argument("--phase", required=True, choices=PHASES)
    p.add_argument("--max-wait", type=int, default=3500)
    p = sub.add_parser("score")
    common(p, True)
    p.add_argument("--s1a-only", action="store_true",
                   help="score right after s1a (no --s1b-dir, no determinism.json); out dir score/<arm>.s1a-only")
    p = sub.add_parser("bridge-check")
    common(p, True)
    p = sub.add_parser("report")
    common(p)
    p.add_argument("--arms", help="comma list (default: every arm of the campaign)")
    p.add_argument("--recorded-harness-dir",
                   help="CPU report only: directory containing the original sealed harness sources")
    p.add_argument("--out-dir", help="fresh absolute report output directory; required for historical reports")
    p = sub.add_parser("serial")
    common(p)
    p.add_argument("--priority", required=True)
    p.add_argument("--max-secs", type=int, default=MAX_JOB_SECS)
    p.add_argument("--wrapper", default=DEFAULT_WRAPPER)
    p.add_argument("--fabric", default=DEFAULT_FABRIC)
    p.add_argument("--max-wait-per-phase", type=int, default=6 * 3600)
    p.add_argument("--detach", action="store_true")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    fn = {"prepare": prepare, "check-plans": check_plans, "launch": launch, "run-phase": run_phase,
          "wait": wait_cmd, "score": score, "report": report, "serial": serial, "bridge-check": bridge_check}[args.cmd]
    try:
        return fn(args)
    except Refusal as e:
        print("REFUSED: %s" % e, file=sys.stderr)
        return 2
    except qs.Invalid as e:
        print("INVALID: %s" % e, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
