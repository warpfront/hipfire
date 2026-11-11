#!/usr/bin/env python3
"""Flash-Next expert-parallel exactness comparator (stdlib only).

Compares two `fn_ep_oracle` output directories (crates/hipfire-loader/examples/
fn_ep_oracle.rs): the REFERENCE arm (one card, topology.ep = 1, or EP-R with
--ref-ranks R) and an EP arm
(topology.ep = N). Each directory holds meta.json, logits.jsonl, state.jsonl
(and optionally raw logits-p<prompt>-s<step>.f32 dumps).

    fn_ep_compare.py REF_DIR EP_DIR [--expect-ranks N] [--ref-ranks R] [--json OUT.json] [--max-report K]
    fn_ep_compare.py --self-test

The EP arm must equal the reference byte for byte (hashes of bytes):

  * identity: both runs completed; same max_seq / tokens / vocab / model
    sha256 (when both recorded it) / prompt files, token counts, token-id md5;
    the reference has --ref-ranks ranks (default 1) and the EP arm has --expect-ranks
    (default: the EP arm's own meta.json `ranks`, which must be >= 2 and, when
    --ref-ranks > 1, greater than --ref-ranks);
  * logits: for every (prompt, step) the same record set, the same greedy
    token and the same rank-0 logits sha256;
  * mirrors: for every EP record every peer rank's logits mirror sha256 equals
    rank 0's (one entry per rank);
  * state: for every (prompt, step) the EP arm has a digest for EVERY rank and
    every digest row (names and values) equals the reference rank 0's;
  * with --ref-ranks R > 1 the reference (itself EP, R ranks) is also checked
    against itself (families mirrors / state_ranks / state, messages prefixed
    "reference"): R mirror hashes equal to its own logits sha, and state
    digests for ranks 0..R-1 all equal to its rank 0.

Prints one line per failing check family (first mismatch in detail) and a
final one-line summary beginning `PASS` or `FAIL`; `--json` writes the full
verdict (counts, first mismatch, up to --max-report mismatches, per-family
results). Exit status: 0 only when every check passed; 1 on any mismatch,
missing or malformed data; 2 on usage errors.

EP reserve-decoupling gates (HIPFIRE_EP_RESERVE_DECOUPLE, mgfn/ep-reserve/DECISION.md section 8) live under
the `reserve` subcommand and leave the invocation above untouched:

    fn_ep_compare.py reserve arms OFF_DIR ON_DIR [--min-resident 33] [--min-tokens 128] [--expect-mode ar|mtp]
                                  [--require-lengths 8,9,...] [--baseline-repeat] [--json OUT.json]
    fn_ep_compare.py reserve perf PERF.json [--procs 3] [--requests 3] [--json OUT.json]
    fn_ep_compare.py reserve c1 MANIFEST.json [--min-pairs 3] [--json OUT.json]
    fn_ep_compare.py reserve harvest LOG OUT_DIR

`arms` takes the two oracle directories of ONE binary (switch =0 vs =1, same EP2 ranks) and requires the
usual byte-exact logits/mirror/state equality plus a `reserve-receipt.json` in each (schema ep-reserve-receipt/1):
per-rank plan receipts, per-prompt chunk/bank/restore records, and committed id arrays. `perf` evaluates the
pp32K kill rule (throughput ratio < 0.95 or restore-inclusive latency ratio > 1.05) and `c1` the weighted
sampled C1 rule (gain < +8%); either kills at < 33 resident layers on a rank. `harvest` assembles the receipt
from `EP_RESERVE_RECEIPT <json>` lines of an arm's log. Exit status: 0 only on PASS (perf: also REPORTED for
non-pp32K cells); 1 on any other verdict or malformed data; 2 on usage errors. No threshold is a performance claim.
"""
import argparse
import contextlib
import io
import json
import math
import os
import struct
import sys
import tempfile
from fractions import Fraction


# ---------------------------------------------------------------- helpers
def read_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError as error:
                raise ValueError("%s:%d: %s" % (path, number, error))
    return out


def read_meta(arm_dir):
    path = os.path.join(arm_dir, "meta.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def first_diff_byte(a, b):
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return i
    return min(len(a), len(b)) if len(a) != len(b) else None


def raw_logits_report(ref_dir, ep_dir, prompt, step):
    """Max-abs / first differing byte of the raw logits at (prompt, step), if saved."""
    name = "logits-p%d-s%d.f32" % (prompt, step)
    path_a, path_b = os.path.join(ref_dir, name), os.path.join(ep_dir, name)
    if not (os.path.exists(path_a) and os.path.exists(path_b)):
        return "raw logits not saved at prompt %d step %d in both arms" % (prompt, step)
    with open(path_a, "rb") as handle:
        a = handle.read()
    with open(path_b, "rb") as handle:
        b = handle.read()
    offset = first_diff_byte(a, b)
    if offset is None:
        return "raw logits identical (a sha mismatch is then a recording error)"
    count = min(len(a), len(b)) // 4
    xs = struct.unpack("<%df" % count, a[:count * 4])
    ys = struct.unpack("<%df" % count, b[:count * 4])
    max_abs, max_index = 0.0, -1
    for i, (x, y) in enumerate(zip(xs, ys)):
        d = abs(x - y)
        if math.isnan(d):
            d = math.inf
        if d > max_abs:
            max_abs, max_index = d, i
    word = offset - (offset % 4)
    return "raw logits: max-abs=%g at element %d; first differing byte %d (word@%d %s vs %s); len %d vs %d" % (
        max_abs, max_index, offset, word, a[word:word + 4].hex(), b[word:word + 4].hex(), len(a), len(b))


class Verdict:
    """Collects mismatches per check family; the first one overall is `first`."""

    def __init__(self, max_report):
        self.max_report = max_report
        self.families = {}
        self.first = None
        self.mismatches = []

    def family(self, name):
        return self.families.setdefault(name, {"checked": 0, "failed": 0})

    def checked(self, name, n=1):
        self.family(name)["checked"] += n

    def fail(self, name, text, **detail):
        fam = self.family(name)
        fam["failed"] += 1
        entry = {"check": name, "message": text}
        entry.update(detail)
        if self.first is None:
            self.first = entry
        if len(self.mismatches) < self.max_report:
            self.mismatches.append(entry)

    @property
    def ok(self):
        return self.first is None


# --------------------------------------------------------------- identity
def check_identity(v, ref_meta, ep_meta, expect_ranks, ref_ranks=1, paired=False):
    for label, meta in (("reference", ref_meta), ("EP", ep_meta)):
        v.checked("identity")
        if meta is None:
            v.fail("identity", "%s arm: meta.json missing" % label)
        elif not meta.get("completed"):
            v.fail("identity", "%s arm: meta.json says the run did not complete" % label)
    if ref_meta is None or ep_meta is None:
        return
    for key in ("max_seq", "tokens", "vocab"):
        v.checked("identity")
        if ref_meta.get(key) != ep_meta.get(key):
            v.fail("identity", "%s differs: reference %r vs EP %r" % (key, ref_meta.get(key), ep_meta.get(key)))
    sha_ref, sha_ep = ref_meta.get("model_sha256"), ep_meta.get("model_sha256")
    v.checked("identity")
    if sha_ref and sha_ep and sha_ref != sha_ep:
        v.fail("identity", "model sha256 differs: %s vs %s" % (sha_ref, sha_ep))

    def prompts(meta):
        return [(p.get("token_count"), p.get("file_md5"), p.get("ids_md5")) for p in meta.get("prompts", [])]

    v.checked("identity")
    if prompts(ref_meta) != prompts(ep_meta):
        v.fail("identity", "prompt identity (token_count, file_md5, ids_md5) differs: %r vs %r"
               % (prompts(ref_meta), prompts(ep_meta)))
    v.checked("identity")
    if ref_meta.get("ranks") != ref_ranks:
        v.fail("identity", "reference arm ran %r ranks, expected %d" % (ref_meta.get("ranks"), ref_ranks))
    ranks = ep_meta.get("ranks")
    v.checked("identity")
    if not isinstance(ranks, int) or ranks < 2:
        v.fail("identity", "EP arm ran %r ranks, expected >= 2" % (ranks,))
    elif ref_ranks > 1 and ranks <= ref_ranks and not paired:
        v.fail("identity", "EP arm ran %d ranks, expected more than the %d-rank reference" % (ranks, ref_ranks))
    elif expect_ranks is not None and ranks != expect_ranks:
        v.fail("identity", "EP arm ran %d ranks, expected %d" % (ranks, expect_ranks))


# ----------------------------------------------------------------- logits
def load_logits(arm_dir):
    out = {}
    for rec in read_jsonl(os.path.join(arm_dir, "logits.jsonl")):
        if rec.get("kind") != "logits":
            continue
        out[(rec["prompt"], rec["step"])] = rec
    return out


def check_logits(v, ref_dir, ep_dir, ref, ep, ranks):
    v.checked("record_set")
    if set(ref) != set(ep):
        only_ref = sorted(set(ref) - set(ep))[:3]
        only_ep = sorted(set(ep) - set(ref))[:3]
        v.fail("record_set", "logits record sets differ (only in reference %s, only in EP %s)" % (only_ref, only_ep))
    if not ref:
        v.fail("record_set", "reference logits.jsonl has no logits records")
    for key in sorted(set(ref) & set(ep)):
        prompt, step = key
        ra, rb = ref[key], ep[key]
        v.checked("logits")
        if ra["logits_sha256"] != rb["logits_sha256"] or ra.get("token") != rb.get("token"):
            report = "prompt=%d step=%d: reference token %s sha %s vs EP token %s sha %s" % (
                prompt, step, ra.get("token"), ra["logits_sha256"][:16], rb.get("token"), rb["logits_sha256"][:16])
            v.fail("logits", report, prompt=prompt, step=step, raw=raw_logits_report(ref_dir, ep_dir, prompt, step))
        v.checked("mirrors")
        mirrors = rb.get("rank_logits_sha256")
        if not mirrors or (ranks and len(mirrors) != ranks):
            v.fail("mirrors", "prompt=%d step=%d: EP record has %s logits mirror hashes, expected %s"
                   % (prompt, step, len(mirrors) if mirrors else 0, ranks), prompt=prompt, step=step)
        else:
            bad = [r for r, sha in enumerate(mirrors) if sha != rb["logits_sha256"]]
            if bad:
                v.fail("mirrors", "prompt=%d step=%d: rank(s) %s logits mirror != rank 0" % (prompt, step, bad),
                       prompt=prompt, step=step, ranks=bad)


# ------------------------------------------------------------------ state
def load_state(arm_dir):
    out = {}
    path = os.path.join(arm_dir, "state.jsonl")
    if not os.path.exists(path):
        return out
    for rec in read_jsonl(path):
        if rec.get("kind") != "state":
            continue
        out[(rec["prompt"], rec["step"], rec["rank"])] = [tuple(row) for row in rec["digest"]]
    return out


def diff_digest(ref_rows, ep_rows):
    notes = []
    if [n for n, _ in ref_rows] != [n for n, _ in ep_rows]:
        notes.append("row names differ (%d vs %d rows)" % (len(ref_rows), len(ep_rows)))
    ep_map = dict(ep_rows)
    for name, value in ref_rows:
        if name in ep_map and ep_map[name] != value:
            notes.append("%s: %s vs %s" % (name, value, ep_map[name]))
    return notes


def check_state(v, ref, ep, ranks):
    ref_keys = sorted((p, s) for (p, s, r) in ref if r == 0)
    v.checked("state_set")
    if not ref_keys:
        v.fail("state_set", "reference state.jsonl has no rank-0 state records")
        return
    ep_keys = sorted(set((p, s) for (p, s, r) in ep))
    if ep_keys != ref_keys:
        v.fail("state_set", "state (prompt, step) sets differ: reference %s vs EP %s" % (ref_keys[:6], ep_keys[:6]))
    for prompt, step in ref_keys:
        ref_rows = ref[(prompt, step, 0)]
        have = sorted(r for (p, s, r) in ep if p == prompt and s == step)
        v.checked("state_ranks")
        if ranks and have != list(range(ranks)):
            v.fail("state_ranks", "prompt=%d step=%d: EP state records for ranks %s, expected 0..%d"
                   % (prompt, step, have, ranks - 1), prompt=prompt, step=step)
        for rank in have:
            v.checked("state")
            notes = diff_digest(ref_rows, ep[(prompt, step, rank)])
            if notes:
                v.fail("state", "prompt=%d step=%d rank %d: %d digest row(s) differ from reference rank 0: %s"
                       % (prompt, step, rank, len(notes), "; ".join(notes[:4])),
                       prompt=prompt, step=step, rank=rank)


def check_reference(v, ref_logits, ref_state, ref_ranks):
    """An EP-R reference must itself be internally exact: R mirrors and R state digests equal to rank 0."""
    for (prompt, step), rec in sorted(ref_logits.items()):
        v.checked("mirrors")
        mirrors = rec.get("rank_logits_sha256")
        if not mirrors or len(mirrors) != ref_ranks:
            v.fail("mirrors", "reference prompt=%d step=%d: record has %s logits mirror hashes, expected %d"
                   % (prompt, step, len(mirrors) if mirrors else 0, ref_ranks), prompt=prompt, step=step)
        else:
            bad = [r for r, sha in enumerate(mirrors) if sha != rec["logits_sha256"]]
            if bad:
                v.fail("mirrors", "reference prompt=%d step=%d: rank(s) %s logits mirror != rank 0" % (prompt, step, bad),
                       prompt=prompt, step=step, ranks=bad)
    for prompt, step in sorted(set((p, s) for (p, s, _) in ref_state)):
        have = sorted(r for (p, s, r) in ref_state if p == prompt and s == step)
        v.checked("state_ranks")
        if have != list(range(ref_ranks)):
            v.fail("state_ranks", "reference prompt=%d step=%d: state records for ranks %s, expected 0..%d"
                   % (prompt, step, have, ref_ranks - 1), prompt=prompt, step=step)
        if 0 not in have:
            continue
        base = ref_state[(prompt, step, 0)]
        for rank in have:
            if rank == 0:
                continue
            v.checked("state")
            notes = diff_digest(base, ref_state[(prompt, step, rank)])
            if notes:
                v.fail("state", "reference prompt=%d step=%d rank %d: %d digest row(s) differ from reference rank 0: %s"
                       % (prompt, step, rank, len(notes), "; ".join(notes[:4])),
                       prompt=prompt, step=step, rank=rank)


# ------------------------------------------------------------------- main
def compare(ref_dir, ep_dir, expect_ranks, max_report, ref_ranks=1):
    v = Verdict(max_report)
    try:
        ref_meta, ep_meta = read_meta(ref_dir), read_meta(ep_dir)
    except ValueError as error:
        v.fail("identity", "meta.json unreadable: %s" % error)
        return v, {}
    check_identity(v, ref_meta, ep_meta, expect_ranks, ref_ranks)
    ranks = expect_ranks or (ep_meta or {}).get("ranks")
    try:
        ref_logits, ep_logits = load_logits(ref_dir), load_logits(ep_dir)
        ref_state, ep_state = load_state(ref_dir), load_state(ep_dir)
    except (OSError, ValueError, KeyError) as error:
        v.fail("record_set", "cannot read oracle output: %s" % error)
        return v, {}
    check_logits(v, ref_dir, ep_dir, ref_logits, ep_logits, ranks if isinstance(ranks, int) else None)
    check_state(v, ref_state, ep_state, ranks if isinstance(ranks, int) else None)
    if ref_ranks > 1:
        check_reference(v, ref_logits, ref_state, ref_ranks)
    prompts = sorted(set(p for (p, _) in ref_logits))
    steps = sorted(set(s for (_, s) in ref_logits))
    shape = {
        "ranks": ranks,
        "ref_ranks": ref_ranks,
        "prompts": len(prompts),
        "logits_records": len(ref_logits),
        "max_step": steps[-1] if steps else None,
        "state_records": len(ep_state),
    }
    return v, shape


def summary_line(v, shape, ref_dir, ep_dir):
    counts = {k: v.families[k]["checked"] for k in v.families}
    if v.ok:
        ref_ranks = shape.get("ref_ranks") or 1
        ref_name = "single" if ref_ranks == 1 else "EP%d reference" % ref_ranks
        return ("PASS fn_ep_compare: EP%s == %s, %s prompt(s) x steps 0..%s (%s logits records token+sha, "
                "%s mirror hashes, %s state digests) byte-identical"
                % (shape.get("ranks"), ref_name, shape.get("prompts"), shape.get("max_step"), counts.get("logits", 0),
                   counts.get("mirrors", 0), counts.get("state", 0)))
    failed = ", ".join("%s=%d" % (k, f["failed"]) for k, f in sorted(v.families.items()) if f["failed"])
    return "FAIL fn_ep_compare: %d failing check(s) [%s]; first: %s" % (
        sum(f["failed"] for f in v.families.values()), failed, v.first["message"])


def run(argv):
    if argv and argv[0] == "reserve":
        return reserve_main(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], add_help=True)
    parser.add_argument("ref_dir", help="reference arm directory (single card, ep = 1; or ep = --ref-ranks)")
    parser.add_argument("ep_dir", help="EP arm directory (ep = N)")
    parser.add_argument("--expect-ranks", type=int, default=None, help="required EP rank count")
    parser.add_argument("--ref-ranks", type=int, default=1, help="reference arm rank count (1 = single card)")
    parser.add_argument("--json", default=None, help="write the verdict JSON here")
    parser.add_argument("--max-report", type=int, default=50, help="mismatches kept in the JSON")
    args = parser.parse_args(argv)
    if args.ref_ranks < 1:
        print("--ref-ranks must be >= 1", file=sys.stderr)
        return 2
    for d in (args.ref_dir, args.ep_dir):
        if not os.path.isdir(d):
            print("not a directory: %s" % d, file=sys.stderr)
            return 2
    v, shape = compare(args.ref_dir, args.ep_dir, args.expect_ranks, args.max_report, args.ref_ranks)
    for name, fam in sorted(v.families.items()):
        print("%s %-10s checked=%d failed=%d" % ("PASS" if not fam["failed"] else "FAIL", name, fam["checked"], fam["failed"]))
    if v.first is not None:
        print("first mismatch [%s]: %s" % (v.first["check"], v.first["message"]))
        if v.first.get("raw"):
            print("     " + v.first["raw"])
    line = summary_line(v, shape, args.ref_dir, args.ep_dir)
    print(line)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump({
                "verdict": "PASS" if v.ok else "FAIL",
                "summary": line,
                "reference": os.path.abspath(args.ref_dir),
                "ep": os.path.abspath(args.ep_dir),
                "shape": shape,
                "families": v.families,
                "first_mismatch": v.first,
                "mismatches": v.mismatches,
            }, handle, indent=2)
            handle.write("\n")
    return 0 if v.ok else 1


# ------------------------------------------------- EP reserve decoupling gate
# Frozen by mgfn/ep-reserve/DECISION.md sections 4, 5, 7.2 (row P7) and 8. Stdlib only, no GPU. Every
# function below takes parsed JSON so tests/test_fn_ep_reserve_gate.py can drive it with CPU fixtures.
# Nothing here measures anything: it validates receipts the cards produce and applies the kill rules.
RESERVE_RECEIPT_SCHEMA = "ep-reserve-receipt/1"
RESERVE_PERF_SCHEMA = "ep-reserve-perf/1"
RESERVE_C1_SCHEMA = "ep-reserve-c1/1"
RESERVE_RANKS = 2
RESERVE_PROMPT_CHUNK_ROWS = 4096          # never lowered (DECISION section 1)
RESERVE_RESIDENT_BANK_ROWS = 512          # calls of <= 512 rows use the resident bank
RESERVE_MIN_RESIDENT_LAYERS = 33          # Gate 1/2/3: < 33 resident layers on either rank kills
RESERVE_MIN_TOKENS = 128                  # Gate 4: at least 128 committed tokens
RESERVE_BOUNDARY_LENGTHS = (8, 9, 511, 512, 513, 2047, 2048, 4095, 4096, 4097, 4396)
RESERVE_PP32K_MIN_PROMPT_TOKENS = 32768
RESERVE_PP32K_THROUGHPUT_FLOOR = Fraction(19, 20)   # on/off throughput ratio < 0.95 kills
RESERVE_PP32K_LATENCY_CEILING = Fraction(21, 20)    # on/off restore-inclusive latency ratio > 1.05 kills
RESERVE_C1_RATIO_FLOOR = Fraction(27, 25)           # weighted sampled C1 on/off < 1.08 (gain < +8%) kills
# Reviewer-facing guards, NOT thresholds frozen by DECISION.md: a category below this on/off ratio, or a
# window-weighted tau shift above this relative size, is reported (category cliff -> REVIEW, tau -> note).
RESERVE_C1_CLIFF_RATIO = Fraction(19, 20)
RESERVE_TAU_NOTE_REL = Fraction(3, 100)
RESERVE_SCORED_CATEGORIES = ("code", "file_edit", "json", "prose", "reasoning", "summarization")
RESERVE_PERF_GATES = ("pp8k", "pp32k", "pp64k", "medium")  # only pp32k carries a kill threshold
RESERVE_RECEIPT_LINE = "EP_RESERVE_RECEIPT "


def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _frac(x):
    """Exact decimal value of a JSON number (str() of a float is its shortest round-trip decimal)."""
    return Fraction(str(x))


def _median(values):
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def reserve_chunk_plan(rows, chunk=RESERVE_PROMPT_CHUNK_ROWS):
    """Row counts of the successive forwards of an outer prompt: full 4096-row tiles, then the tail."""
    full, tail = divmod(rows, chunk)
    return [chunk] * full + ([tail] if tail else [])


def reserve_expected_banks(chunks):
    """513..4096 rows borrow; <= 512 rows (including every tail of 1..512 rows) use the resident bank."""
    return ["borrowed" if rows > RESERVE_RESIDENT_BANK_ROWS else "resident" for rows in chunks]


def reserve_expected_restore_points(banks):
    """Chunk indices a global restore happens before; len(banks) means the outer prompt completion.

    A borrowed -> resident-bank transition (tiny tail, 512-row S1 call) restores before that call; a prompt
    that ends borrowed restores once at completion; full tiles in between never restore.
    """
    points = []
    for index, bank in enumerate(banks):
        if bank == "resident" and index > 0 and banks[index - 1] == "borrowed":
            points.append(index)
    if banks and banks[-1] == "borrowed":
        points.append(len(banks))
    return points


def reserve_expected_epochs(banks):
    """Number of maximal borrowed runs = number of acquire/restore pairs for the prompt."""
    return sum(1 for i, bank in enumerate(banks) if bank == "borrowed" and (i == 0 or banks[i - 1] != "borrowed"))


def reserve_pp32k_kill_reasons(throughput_ratio, latency_ratio):
    """Kill reasons for a pp32K on/off pair; ratios are Fractions. Exactly 0.95 / 1.05 pass."""
    reasons = []
    if throughput_ratio < RESERVE_PP32K_THROUGHPUT_FLOOR:
        reasons.append("pp32K throughput ratio %.4f < 0.95" % float(throughput_ratio))
    if latency_ratio > RESERVE_PP32K_LATENCY_CEILING:
        reasons.append("pp32K restore-inclusive latency ratio %.4f > 1.05" % float(latency_ratio))
    return reasons


def reserve_c1_kills(ratio):
    """True when the weighted sampled C1 on/off ratio is below +8%; exactly 1.08 passes."""
    return ratio < RESERVE_C1_RATIO_FLOOR


def reserve_resident_kills(layers, minimum=RESERVE_MIN_RESIDENT_LAYERS):
    return layers < minimum


# ---- receipts: harvest from the arm's log lines
_PROMPT_RANK_SAME = ("rows", "chunk_rows", "banks", "borrowed_epochs")
_PROMPT_RANK_LISTS = ("exchange_address_before", "exchange_address_after", "donor_hash_before", "donor_hash_after")
RESERVE_PHASES = ("target", "mtp_fill")


def _entry_tag(index, phase, ordinal):
    return "prompt %d entry (%s,%s)" % (index, phase, ordinal)


def _merge_prompt_ranks(index, by_rank, want_ranks, errors, phase="target", ordinal=0):
    """Merge the per-rank `prompt_rank` records of one outermost prompt entry into one `prompt` record.

    `(index, phase, ordinal)` is the entry's key: the oracle prompt, the driver ("target" / "mtp_fill") and the entry's
    per-rank count within them. Shared facts (rows, chunk_rows, banks, borrowed_epochs, restore positions and bytes) must be
    identical on every rank; per-rank scalars become lists ordered by rank; restore_ms and elapsed take the slowest rank.
    """
    tag = _entry_tag(index, phase, ordinal)
    ranks = sorted(by_rank)
    for r in sorted(set(want_ranks) - set(ranks)):
        errors.append("%s: no prompt_rank record from rank %d" % (tag, r))
    recs = [by_rank[r] for r in ranks]
    out = {"prompt": index, "phase": phase, "ordinal": ordinal}
    for key in _PROMPT_RANK_SAME:
        values = [rec.get(key) for rec in recs]
        if any(v != values[0] for v in values):
            errors.append("%s: ranks disagree on %s: %r" % (tag, key, values))
        out[key] = values[0]
    event_sets = [rec.get("restores") if isinstance(rec.get("restores"), list) else [] for rec in recs]
    if any(len(e) != len(event_sets[0]) or [x.get("before_chunk") for x in e] != [x.get("before_chunk") for x in event_sets[0]]
           or [x.get("bytes") for x in e] != [x.get("bytes") for x in event_sets[0]] for e in event_sets):
        errors.append("%s: ranks disagree on restore positions or bytes" % tag)
    events = []
    for i, first in enumerate(event_sets[0]):
        per = [e[i] for e in event_sets if i < len(e)]
        merged = {"before_chunk": first.get("before_chunk"), "bytes": first.get("bytes")}
        times = [x.get("restore_ms") for x in per]
        if all(_is_num(t) for t in times):
            merged["restore_ms"] = max(times)
        for key in ("barrier_wait_ms", "s2_dma_tail_ms"):
            merged[key] = [x.get(key) for x in per]
        events.append(merged)
    out["restores"] = events
    elapsed = [rec.get("elapsed_ms_incl_restore") for rec in recs]
    if all(_is_num(t) for t in elapsed):
        out["elapsed_ms_incl_restore"] = max(elapsed)
    for key in _PROMPT_RANK_LISTS:
        if all(key in rec for rec in recs):  # a hash that failed to read back is absent: the checker rejects the prompt
            out[key] = [rec[key] for rec in recs]
    return out


def reserve_assemble_receipt(lines):
    """Build a reserve-receipt.json document from `EP_RESERVE_RECEIPT <json>` log lines.

    Record kinds: header (top-level keys; the forward emits one per rank, repeats must agree), rank (one per rank),
    prompt (one per prompt, oracle side), prompt_rank (one per rank per outermost prompt entry, forward side), ids
    ({"prompt": i, "ids": [...]}) and end ({"complete": true}). A run with no end record is truncated.

    A `prompt_rank` record is keyed by (prompt, phase, ordinal): the oracle prompt the forward ran under, the driver that
    opened the outermost entry ("target" or "mtp_fill"), and the rank's count of entries within that (prompt, phase). A
    process runs several outermost entries per oracle prompt (target prefill, the native-MTP fill, decode / verify
    advances), so records are merged per key across ranks (`_merge_prompt_ranks`): (i, "target", 0) becomes `prompts[i]`,
    every other key a merged record in `prompts[i]["entries"]`. A missing phase/ordinal or a repeated
    (prompt, phase, ordinal, rank) is a harvest error.
    """
    doc = {"schema": RESERVE_RECEIPT_SCHEMA, "ranks": [], "prompts": [], "token_ids": [], "complete": False,
           "harvest_errors": []}
    ranks, prompts, ids, prompt_ranks = {}, {}, {}, {}
    seen = 0
    for line in lines:
        at = line.find(RESERVE_RECEIPT_LINE)
        if at < 0:
            continue
        seen += 1
        try:
            rec = json.loads(line[at + len(RESERVE_RECEIPT_LINE):])
        except ValueError as error:
            doc["harvest_errors"].append("unparsable receipt line: %s" % error)
            continue
        kind = rec.pop("kind", None) if isinstance(rec, dict) else None
        if kind == "header":
            for k, val in rec.items():
                if k in ("ranks", "prompts", "token_ids", "complete"):
                    continue
                if k in doc and doc[k] != val:
                    doc["harvest_errors"].append("header key %s disagrees: %r vs %r" % (k, doc[k], val))
                doc[k] = val
        elif kind == "rank" and _is_int(rec.get("rank")):
            if rec["rank"] in ranks:
                doc["harvest_errors"].append("duplicate rank %d receipt" % rec["rank"])
            ranks[rec["rank"]] = rec
        elif kind == "prompt" and _is_int(rec.get("prompt")):
            if rec["prompt"] in prompts:
                doc["harvest_errors"].append("duplicate prompt %d receipt" % rec["prompt"])
            prompts[rec["prompt"]] = rec
        elif kind == "prompt_rank" and _is_int(rec.get("prompt")) and _is_int(rec.get("rank")):
            phase, ordinal = rec.get("phase"), rec.get("ordinal")
            if phase not in RESERVE_PHASES or not _is_int(ordinal) or ordinal < 0:
                doc["harvest_errors"].append("prompt_rank record for prompt %d rank %d has phase %r / ordinal %r (phase must be one of %s, "
                                             "ordinal a non-negative integer)" % (rec["prompt"], rec["rank"], phase, ordinal, "/".join(RESERVE_PHASES)))
                continue
            slot = prompt_ranks.setdefault((rec["prompt"], phase, ordinal), {})
            if rec["rank"] in slot:
                doc["harvest_errors"].append("duplicate prompt_rank record for prompt %d phase %s ordinal %d rank %d"
                                             % (rec["prompt"], phase, ordinal, rec["rank"]))
            slot[rec["rank"]] = rec
        elif kind == "ids" and _is_int(rec.get("prompt")):
            if rec["prompt"] in ids:
                doc["harvest_errors"].append("duplicate id array for prompt %d" % rec["prompt"])
            ids[rec["prompt"]] = rec.get("ids")
        elif kind == "end":
            doc["complete"] = rec.get("complete") is True
        else:
            doc["harvest_errors"].append("unknown receipt record kind %r" % (kind,))
    doc["ranks"] = [ranks[r] for r in sorted(ranks)]
    merged_entries = {}
    for (index, phase, ordinal), by_rank in sorted(prompt_ranks.items()):
        if index in prompts:
            doc["harvest_errors"].append("prompt %d has both a prompt and prompt_rank record" % index)
            continue
        merged_entries.setdefault(index, {})[(phase, ordinal)] = _merge_prompt_ranks(index, by_rank, sorted(ranks), doc["harvest_errors"], phase, ordinal)
    mode = doc.get("mode")
    base_key = ("mtp_fill", 0) if mode == "mtp" else ("target", 0)  # mtp mode has no target prefill entry
    for index, entries in merged_entries.items():
        base = entries.pop(base_key, None)
        if base is None:
            doc["harvest_errors"].append("prompt %d has prompt_rank records %s but no (%s,%d) entry (mode %s)"
                                         % (index, sorted(entries), base_key[0], base_key[1], mode or "ar"))
            continue
        base["entries"] = [entries[key] for key in sorted(entries, key=lambda k: (RESERVE_PHASES.index(k[0]), k[1]))]
        prompts[index] = base
    doc["prompts"] = [prompts[p] for p in sorted(prompts)]
    doc["token_ids"] = [ids.get(p) for p in range(max(ids) + 1)] if ids else []
    return doc, seen


# ---- receipts: validation
def reserve_load_receipt(arm_dir):
    path = os.path.join(arm_dir, "reserve-receipt.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _event_list(fail, ev, key, ranks):
    value = ev.get(key)
    if not isinstance(value, list) or len(value) != ranks or not all(_is_num(x) and x >= 0 for x in value):
        fail("restore", "restore event %s must be %d non-negative numbers (one per rank), got %r" % (key, ranks, value))


def _check_entry_rules(fail, v, tag, p, on, ranks, chunks):
    """Bank, restore, epoch, timing, donor-hash and exchange-address rules of one merged outermost entry `p` whose forward
    row sequence is `chunks` (the off arm: every bank resident, no restore, no epoch, no donor hash)."""
    banks = reserve_expected_banks(chunks) if on else ["resident"] * len(chunks)
    if p.get("banks") != banks:
        fail("boundary", "%s: bank sequence %r, expected %r" % (tag, p.get("banks"), banks))
    points = reserve_expected_restore_points(banks) if on else []
    events = p.get("restores")
    if not isinstance(events, list):
        fail("restore", "%s: restore events missing" % tag)
        events = []
    got = [e.get("before_chunk") if isinstance(e, dict) else None for e in events]
    v.checked("restore")
    if got != points:
        fail("restore", "%s: restores before chunks %r, expected %r (one per borrowed epoch, none per full tile)" % (tag, got, points))
    if p.get("borrowed_epochs") != (reserve_expected_epochs(banks) if on else 0):
        fail("restore", "%s: borrowed_epochs=%r, expected %d" % (tag, p.get("borrowed_epochs"), reserve_expected_epochs(banks) if on else 0))
    restore_total = 0
    for e in events:
        if not isinstance(e, dict):
            continue
        if not _is_num(e.get("restore_ms")) or e["restore_ms"] < 0:
            fail("restore", "%s: restore timing omitted (restore_ms=%r)" % (tag, e.get("restore_ms")))
        else:
            restore_total += e["restore_ms"]
        if not _is_int(e.get("bytes")) or e["bytes"] <= 0:
            fail("restore", "%s: restore bytes missing (bytes=%r)" % (tag, e.get("bytes")))
        for key in ("barrier_wait_ms", "s2_dma_tail_ms"):
            _event_list(lambda fam, text, tag=tag: fail(fam, "%s: %s" % (tag, text)), e, key, RESERVE_RANKS)
    elapsed = p.get("elapsed_ms_incl_restore")
    if not _is_num(elapsed) or elapsed <= 0 or elapsed < restore_total:
        fail("restore", "%s: elapsed_ms_incl_restore=%r must be a positive time that includes the %.3f ms restore" % (tag, elapsed, restore_total))
    if on:
        # The forward hashes the donors only around an entry that opens on the borrowed bank (a resident-bank entry
        # never dirties them; a read-back per decode window would dominate the run); such an entry must carry both.
        borrows = "borrowed" in (p.get("banks") or [])
        hashed = borrows or "donor_hash_before" in p or "donor_hash_after" in p
        keys = ("exchange_address_before", "exchange_address_after") + (("donor_hash_before", "donor_hash_after") if hashed else ())
        for key in keys:
            value = p.get(key)
            if not isinstance(value, list) or len(value) != RESERVE_RANKS or not all(isinstance(x, str) and x for x in value):
                fail("restore", "%s: %s must be %d non-empty strings (one per rank)" % (tag, key, RESERVE_RANKS))
        if p.get("donor_hash_before") != p.get("donor_hash_after"):
            fail("restore", "%s: donor bytes differ after restore (%r vs %r)" % (tag, p.get("donor_hash_before"), p.get("donor_hash_after")))
        fixed = [r.get("exchange_address") for r in ranks]
        if len(fixed) == RESERVE_RANKS and (p.get("exchange_address_before") != fixed or p.get("exchange_address_after") != fixed):
            fail("restore", "%s: exchange address moved across the prompt (%r -> %r, registered %r)"
                 % (tag, p.get("exchange_address_before"), p.get("exchange_address_after"), fixed))


def _check_prompt_entries(fail, v, index, p, on, ranks, mode):
    """The outermost entries of oracle prompt `index` other than the whole-prompt entry `p` (`p["entries"]`).

    `p` is (target,0) in ar mode and the native-MTP prefill (mtp_fill,0) in mtp mode (there is no target prefill entry
    there). Every entry here (decode / verify / advance windows, mtp mode's (target,0) first window included) carries its
    own forward row sequence (each call 1..4096 rows, summing to its rows) and meets the bank / restore / epoch / timing /
    donor-hash / exchange-address rules. An mtp_fill entry here is an error (mode mtp: a second fill; mode ar: none).
    """
    entries = p.get("entries")
    if entries is None:
        entries = []
    if not isinstance(entries, list):
        fail("boundary", "prompt %d: entries is not a list" % index)
        return
    base = ("mtp_fill", 0) if mode == "mtp" else ("target", 0)
    seen = set()
    for ent in entries:
        if not isinstance(ent, dict):
            fail("boundary", "prompt %d: entry record is not an object" % index)
            continue
        phase, ordinal = ent.get("phase"), ent.get("ordinal")
        tag = _entry_tag(index, phase, ordinal)
        if ent.get("prompt") != index or phase not in RESERVE_PHASES or not _is_int(ordinal) or ordinal < 0:
            fail("boundary", "%s: entry key (prompt %r, phase %r, ordinal %r) is malformed" % (tag, ent.get("prompt"), phase, ordinal))
            continue
        if (phase, ordinal) == base or (phase, ordinal) in seen:
            fail("boundary", "%s: duplicate entry key (prompt, phase, ordinal)" % tag)
            continue
        seen.add((phase, ordinal))
        if phase == "mtp_fill":
            fail("boundary", "prompt %d: %s" % (index, "mode mtp needs exactly one (mtp_fill,0) entry, got a further %s" % tag
                                                 if mode == "mtp" else "%s in an ar-mode run" % tag))
            continue
        v.checked("boundary")
        rows = ent.get("rows")
        if not _is_int(rows) or rows < 1:
            fail("boundary", "%s: rows %r" % (tag, rows))
            continue
        chunks = ent.get("chunk_rows")
        if (not isinstance(chunks, list) or not chunks or not all(_is_int(c) and 1 <= c <= RESERVE_PROMPT_CHUNK_ROWS for c in chunks)
                or sum(chunks) != rows):
            fail("chunk", "%s: forward row sequence %r must be 1..%d-row calls summing to rows %r"
                 % (tag, chunks, RESERVE_PROMPT_CHUNK_ROWS, rows))
            continue
        _check_entry_rules(fail, v, tag, ent, on, ranks, chunks)


def check_reserve_receipt(v, label, rec, meta, on, min_resident, min_tokens, expect_mode, logits, chains=None):
    """Validate one arm's receipt against its own oracle meta. `on` selects the decoupled-arm rules."""

    def fail(family, text):
        v.fail(family, "%s: %s" % (label, text))

    v.checked("receipt")
    if not isinstance(rec, dict):
        fail("receipt", "reserve-receipt.json absent or not an object (per-rank plan/restore receipts are required)")
        return
    if rec.get("schema") != RESERVE_RECEIPT_SCHEMA:
        fail("receipt", "schema %r, expected %r" % (rec.get("schema"), RESERVE_RECEIPT_SCHEMA))
    for error in rec.get("harvest_errors") or []:
        fail("receipt", "log harvest: %s" % error)
    if rec.get("complete") is not True:
        fail("receipt", "run is truncated (complete is not true)")
    if rec.get("decouple") is not on:
        fail("receipt", "decouple=%r, expected %r for this arm" % (rec.get("decouple"), on))
    for key in ("binary_id", "model_sha256"):
        if not isinstance(rec.get(key), str) or not rec.get(key):
            fail("receipt", "%s missing" % key)
    if not _is_int(rec.get("seed")):
        fail("receipt", "seed missing (must be an integer)")
    mode = rec.get("mode")
    if mode not in ("ar", "mtp") or (expect_mode and mode != expect_mode):
        fail("receipt", "mode %r, expected %s" % (mode, expect_mode or "ar or mtp"))
    if rec.get("prompt_chunk_rows") != RESERVE_PROMPT_CHUNK_ROWS:
        fail("chunk", "prompt_chunk_rows=%r, the chunk stays %d (never lowered)" % (rec.get("prompt_chunk_rows"), RESERVE_PROMPT_CHUNK_ROWS))
    meta = meta or {}
    if meta.get("max_seq") is not None and rec.get("max_seq") != meta.get("max_seq"):
        fail("receipt", "max_seq %r != oracle meta %r" % (rec.get("max_seq"), meta.get("max_seq")))

    # ---- per-rank plan receipts
    v.checked("plan")
    ranks = rec.get("ranks")
    got_ranks = [r.get("rank") if isinstance(r, dict) else None for r in ranks] if isinstance(ranks, list) else None
    if got_ranks is None or sorted(map(str, got_ranks)) != [str(i) for i in range(RESERVE_RANKS)]:
        fail("plan", "rank receipts must be exactly ranks 0..%d, got %r" % (RESERVE_RANKS - 1, got_ranks))
        ranks = []
    ranks = sorted((r for r in ranks if isinstance(r, dict)), key=lambda r: r["rank"])
    for r in ranks:
        tag = "rank %s" % r["rank"]
        layers = r.get("resident_layers")
        if not _is_int(layers) or layers < 1:
            fail("plan", "%s: resident_layers missing" % tag)
            continue
        if not on:
            continue
        if reserve_resident_kills(layers, min_resident):
            fail("plan", "%s: %d resident layers < %d (kill)" % (tag, layers, min_resident))
        for key in ("source_generation", "plan_digest"):
            value = r.get(key)  # EpReserveMeshReceipt carries both as u64; strings are accepted too
            if not ((isinstance(value, str) and value) or _is_int(value)):
                fail("plan", "%s: %s missing" % (tag, key))
        if not isinstance(r.get("exchange_address"), str) or not r.get("exchange_address"):
            fail("plan", "%s: exchange_address missing" % tag)
        first, count, s2 = r.get("donor_first_layer"), r.get("donor_count"), r.get("s2_donor_layers")
        if not _is_int(first) or not _is_int(count) or count < 1 or first + count != layers:
            fail("plan", "%s: scratch donors [%r,+%r) must end at resident_layers %d" % (tag, first, count, layers))
        elif not isinstance(s2, list) or sorted(s2) != [first - 2, first - 1]:
            fail("plan", "%s: S2 donors %r must be the two layers just below the scratch donors (%d, %d)" % (tag, s2, first - 2, first - 1))
        for key in ("prompt_rows", "exchange_rows"):
            if r.get(key) != RESERVE_PROMPT_CHUNK_ROWS:
                fail("chunk", "%s: %s=%r, expected the 4096-row plan" % (tag, key, r.get(key)))
        for key in ("s1_admitted", "s2_admitted"):
            if r.get(key) is not True:
                fail("plan", "%s: %s is not true" % (tag, key))
    if on and len(ranks) == RESERVE_RANKS:
        for key in ("plan_digest", "source_generation", "resident_layers", "donor_first_layer", "donor_count", "prompt_rows", "exchange_rows"):
            if ranks[0].get(key) != ranks[1].get(key):
                fail("plan", "ranks disagree on %s: %r vs %r" % (key, ranks[0].get(key), ranks[1].get(key)))
        if sorted(ranks[0].get("s2_donor_layers") or []) != sorted(ranks[1].get("s2_donor_layers") or []):
            fail("plan", "ranks disagree on s2_donor_layers")

    # ---- per-prompt boundary / restore receipts
    meta_prompts = meta.get("prompts") or []
    prompts = rec.get("prompts")
    v.checked("boundary")
    if not isinstance(prompts, list) or len(prompts) != len(meta_prompts):
        fail("boundary", "%s prompt receipts, oracle meta has %d prompts (truncated run?)"
             % (len(prompts) if isinstance(prompts, list) else "no", len(meta_prompts)))
        prompts = [p for p in prompts if isinstance(p, dict)] if isinstance(prompts, list) else []
    for index, p in enumerate(prompts):
        if not isinstance(p, dict):
            continue
        tag = "prompt %d" % index
        want_rows = meta_prompts[index].get("token_count") if index < len(meta_prompts) else None
        rows = p.get("rows")
        if p.get("prompt") != index or not _is_int(rows) or rows < 1 or (want_rows is not None and rows != want_rows):
            fail("boundary", "%s: rows %r / index %r do not match the oracle prompt (%r tokens)" % (tag, rows, p.get("prompt"), want_rows))
            continue
        if rec.get("mode") == "mtp" and (p.get("phase"), p.get("ordinal")) != ("mtp_fill", 0):
            # mtp mode: the native-MTP prefill is the one whole-prompt entry of the oracle prompt
            fail("boundary", "%s: mode mtp needs the whole-prompt entry to be (mtp_fill,0), got (%s,%s)"
                 % (tag, p.get("phase"), p.get("ordinal")))
            continue
        chunks = reserve_chunk_plan(rows)
        if p.get("chunk_rows") != chunks:
            fail("chunk", "%s: forward row sequence %r, expected %r (4096-row tiles, never lowered)" % (tag, p.get("chunk_rows"), chunks))
        _check_entry_rules(fail, v, tag, p, on, ranks, chunks)
        _check_prompt_entries(fail, v, index, p, on, ranks, rec.get("mode"))

    # ---- committed id arrays
    v.checked("ids")
    ids = rec.get("token_ids")
    tokens = meta.get("tokens")
    if _is_int(tokens) and tokens < min_tokens:
        fail("ids", "oracle committed %d tokens, at least %d required" % (tokens, min_tokens))
    if not isinstance(ids, list) or len(ids) != len(meta_prompts):
        fail("ids", "id arrays missing or %s of them for %d prompts" % (len(ids) if isinstance(ids, list) else "no", len(meta_prompts)))
        return
    for index, arr in enumerate(ids):
        if not isinstance(arr, list) or not arr or not all(_is_int(x) for x in arr):
            fail("ids", "prompt %d: id array missing or empty" % index)
            continue
        if _is_int(tokens) and len(arr) != tokens:
            fail("ids", "prompt %d: %d ids, oracle meta says %d tokens (truncated)" % (index, len(arr), tokens))
        if chains is not None:
            chain = chains.get(index)  # MTP: the oracle's `tokens` record (no per-step logits records)
            source = "the logits.jsonl tokens record"
        else:
            chain = [logits[k].get("token") for k in sorted(k for k in logits if k[0] == index and k[1] >= 1)] if logits else []
            source = "logits.jsonl tokens"
        if chain and chain != arr:
            fail("ids", "prompt %d: id array disagrees with %s" % (index, source))


def check_reserve_pair(v, off, on, off_meta, on_meta, min_resident, baseline_repeat):
    """Cross-arm identity: same binary/model/seed/source/prompts, residency gain, identical ids."""
    if not isinstance(off, dict) or not isinstance(on, dict):
        return
    v.checked("cross")
    for key in ("binary_id", "model_sha256", "ple_sha256", "seed", "mode", "max_seq", "prompt_chunk_rows"):
        if off.get(key) != on.get(key):
            v.fail("cross", "arms differ on %s: off %r vs on %r (same-binary off/on pair required)" % (key, off.get(key), on.get(key)))
    if (off_meta or {}).get("prompts") and (on_meta or {}).get("prompts"):
        if [p.get("token_count") for p in off_meta["prompts"]] != [p.get("token_count") for p in on_meta["prompts"]]:
            v.fail("cross", "arms ran different prompt lengths")
    off_ranks = {r.get("rank"): r for r in off.get("ranks") or [] if isinstance(r, dict)}
    on_ranks = {r.get("rank"): r for r in on.get("ranks") or [] if isinstance(r, dict)}
    for rank in sorted(set(off_ranks) & set(on_ranks)):
        a, b = off_ranks[rank], on_ranks[rank]
        if a.get("source_generation") != b.get("source_generation") and a.get("source_generation") is not None:
            v.fail("cross", "rank %s source_generation differs between arms: %r vs %r" % (rank, a.get("source_generation"), b.get("source_generation")))
        if not baseline_repeat and _is_int(a.get("resident_layers")) and _is_int(b.get("resident_layers")) and b["resident_layers"] <= a["resident_layers"]:
            v.fail("cross", "rank %s: %d resident layers with the switch on vs %d off: no residency gain" % (rank, b["resident_layers"], a["resident_layers"]))
    a_ids, b_ids = off.get("token_ids"), on.get("token_ids")
    if isinstance(a_ids, list) and isinstance(b_ids, list):
        v.checked("ids")
        for index, (x, y) in enumerate(zip(a_ids, b_ids)):
            if isinstance(x, list) and isinstance(y, list) and x != y:
                at = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
                v.fail("ids", "prompt %d: committed ids differ from the off arm at token %d (%s vs %s)"
                       % (index, at, x[at] if at < len(x) else None, y[at] if at < len(y) else None), prompt=index)


def load_all_records(arm_dir):
    """Every record of an arm's logits.jsonl, in file order (the MTP oracle writes window / tokens records, no logits)."""
    return list(read_jsonl(os.path.join(arm_dir, "logits.jsonl")))


def mtp_token_chains(records):
    """prompt -> committed ids of the oracle's per-prompt `tokens` record (prefill seed included)."""
    return {rec.get("prompt"): rec.get("ids") for rec in records if rec.get("kind") == "tokens"}


def check_mtp_records(v, off, on):
    """MTP arms: every logits.jsonl record (window / tokens) must be identical between the arms, in order."""
    v.checked("record_set")
    if not off:
        v.fail("record_set", "reference logits.jsonl has no records")
    if len(off) != len(on):
        v.fail("record_set", "logits.jsonl record counts differ: off %d vs on %d" % (len(off), len(on)))
    kinds = {rec.get("kind") for rec in off}
    if off and not ({"window", "tokens"} & kinds):
        v.fail("record_set", "MTP logits.jsonl has no window / tokens records (kinds %s)" % sorted(map(str, kinds)))
    for pos, (a, b) in enumerate(zip(off, on)):
        v.checked("logits")
        if a != b:
            v.fail("logits", "logits.jsonl record %d (%s prompt %s window %s) differs: off %s vs on %s"
                   % (pos, a.get("kind"), a.get("prompt"), a.get("window"), json.dumps(a, sort_keys=True)[:300],
                      json.dumps(b, sort_keys=True)[:300]), prompt=a.get("prompt"))
            break  # the first differing record is the report


def compare_reserve_arms(off_dir, on_dir, min_resident=RESERVE_MIN_RESIDENT_LAYERS, min_tokens=RESERVE_MIN_TOKENS,
                         expect_mode=None, require_lengths=(), baseline_repeat=False, max_report=50):
    """Gate 1/4 pair: one binary, switch off vs on (or off vs off with baseline_repeat)."""
    v = Verdict(max_report)
    try:
        off_meta, on_meta = read_meta(off_dir), read_meta(on_dir)
    except ValueError as error:
        v.fail("identity", "meta.json unreadable: %s" % error)
        return v, {}
    check_identity(v, off_meta, on_meta, RESERVE_RANKS, RESERVE_RANKS, paired=True)
    spec_kinds = {((m or {}).get("spec") or {}).get("kind") for m in (off_meta, on_meta)}
    mtp = expect_mode == "mtp" or (expect_mode is None and "mtp" in spec_kinds)
    try:
        off_logits, on_logits = load_logits(off_dir), load_logits(on_dir)
        off_state, on_state = load_state(off_dir), load_state(on_dir)
        off_records, on_records = (load_all_records(off_dir), load_all_records(on_dir)) if mtp else ([], [])
    except (OSError, ValueError, KeyError) as error:
        v.fail("record_set", "cannot read oracle output: %s" % error)
        return v, {}
    if mtp:
        check_mtp_records(v, off_records, on_records)
    else:
        check_logits(v, off_dir, on_dir, off_logits, on_logits, RESERVE_RANKS)
    check_state(v, off_state, on_state, RESERVE_RANKS)
    check_reference(v, off_logits, off_state, RESERVE_RANKS)
    receipts = []
    for label, directory in (("off arm", off_dir), ("on arm", on_dir)):
        try:
            receipts.append(reserve_load_receipt(directory))
        except (OSError, ValueError) as error:
            receipts.append(None)
            v.fail("receipt", "%s: reserve-receipt.json unreadable: %s" % (label, error))
            continue
        if receipts[-1] is None:
            v.checked("receipt")
            v.fail("receipt", "%s: reserve-receipt.json absent (per-rank plan/restore receipts are required)" % label)
    off_rec, on_rec = receipts
    off_chains = mtp_token_chains(off_records) if mtp else None
    on_chains = mtp_token_chains(on_records) if mtp else None
    if off_rec is not None:
        check_reserve_receipt(v, "off arm", off_rec, off_meta, False, min_resident, min_tokens, expect_mode, off_logits, off_chains)
    if on_rec is not None:
        check_reserve_receipt(v, "repeat arm" if baseline_repeat else "on arm", on_rec, on_meta, not baseline_repeat,
                              min_resident, min_tokens, expect_mode, on_logits, on_chains)
    check_reserve_pair(v, off_rec, on_rec, off_meta, on_meta, min_resident, baseline_repeat)
    lengths = sorted(p.get("token_count") for p in (on_meta or {}).get("prompts", []) if isinstance(p, dict))
    v.checked("coverage")
    missing = sorted(set(require_lengths) - set(lengths))
    if missing:
        v.fail("coverage", "boundary lengths not exercised: %s (prompt token counts: %s)" % (missing, lengths))
    shape = {"ranks": RESERVE_RANKS, "prompt_lengths": lengths, "baseline_repeat": baseline_repeat,
             "logits_records": len(off_records) if mtp else len(off_logits), "state_records": len(on_state)}
    return v, shape


# ---- performance gates
def _reserve_invalid(errors, extra=None):
    out = {"verdict": "INVALID", "reasons": errors}
    out.update(extra or {})
    return out


def _forbidden_reference_keys(obj, where, errors):
    for key in obj:
        if isinstance(key, str) and key.lower().startswith(("reference", "baseline", "historical")):
            errors.append("%s carries %r: historical or hand-entered references are not accepted; the off arm must be "
                          "measured in the same schedule" % (where, key))


def evaluate_reserve_perf(doc, min_resident=RESERVE_MIN_RESIDENT_LAYERS, procs=3, requests=3):
    """Gate 2: paired off/on prompt-fill latency, process medians, restore included.

    Returns a verdict dict: PASS / KILL / INCONCLUSIVE / REPORTED (non-pp32K cells: no threshold claimed) /
    INVALID (schema or schedule violation). Every sample is retained in the result.
    """
    errors = []
    if not isinstance(doc, dict):
        return _reserve_invalid(["perf document is not an object"])
    if doc.get("schema") != RESERVE_PERF_SCHEMA:
        errors.append("schema %r, expected %r" % (doc.get("schema"), RESERVE_PERF_SCHEMA))
    _forbidden_reference_keys(doc, "document", errors)
    gate, mode = doc.get("gate"), doc.get("mode")
    if gate not in RESERVE_PERF_GATES:
        errors.append("gate %r not one of %s" % (gate, list(RESERVE_PERF_GATES)))
    if mode not in ("ar", "mtp"):
        errors.append("mode %r must be ar or mtp (measured separately, never inferred)" % (mode,))
    if doc.get("chunk_rows") != RESERVE_PROMPT_CHUNK_ROWS:
        errors.append("chunk_rows=%r, the chunk stays %d" % (doc.get("chunk_rows"), RESERVE_PROMPT_CHUNK_ROWS))
    for key in ("binary_id", "prompt_sha256"):
        if not isinstance(doc.get(key), str) or not doc.get(key):
            errors.append("%s missing" % key)
    prompt_tokens, generated = doc.get("prompt_tokens"), doc.get("generated_tokens")
    if not _is_int(prompt_tokens) or prompt_tokens < 1:
        errors.append("prompt_tokens missing")
    elif gate == "pp32k" and prompt_tokens < RESERVE_PP32K_MIN_PROMPT_TOKENS:
        errors.append("pp32k prompt has %d tokens < %d" % (prompt_tokens, RESERVE_PP32K_MIN_PROMPT_TOKENS))
    if not _is_int(generated) or generated < 1:
        errors.append("generated_tokens missing")
    for key in ("elapsed_includes_restore", "elapsed_spans_all_ranks"):
        if doc.get(key) is not True:
            errors.append("%s must be true: the interval includes the restore and the slowest rank" % key)
    samples = doc.get("samples")
    if not isinstance(samples, list) or not samples:
        errors.append("samples missing")
        return _reserve_invalid(errors)
    order = doc.get("launch_order")
    want_order_len = 2 * procs
    if (not isinstance(order, list) or len(order) != want_order_len or sorted(map(str, order)) != ["off"] * procs + ["on"] * procs
            or any(order[i] == order[i + 1] for i in range(len(order) - 1))):
        errors.append("launch_order %r must alternate off/on over %d fresh processes (%d per arm)" % (order, want_order_len, procs))
    timed = {"off": {}, "on": {}}
    resident = []
    for n, s in enumerate(samples):
        if not isinstance(s, dict):
            errors.append("sample %d is not an object" % n)
            continue
        _forbidden_reference_keys(s, "sample %d" % n, errors)
        arm = s.get("arm")
        if arm not in timed:
            errors.append("sample %d: arm %r must be off or on" % (n, arm))
            continue
        if not _is_int(s.get("process")) or not _is_int(s.get("request")) or not isinstance(s.get("warmup"), bool):
            errors.append("sample %d: process/request/warmup missing" % n)
            continue
        for key in ("binary_id", "prompt_sha256", "mode", "chunk_rows", "generated_tokens"):
            if s.get(key) != doc.get(key):
                errors.append("sample %d (%s p%d r%d): %s %r differs from the document %r" % (n, arm, s["process"], s["request"], key, s.get(key), doc.get(key)))
        if not _is_num(s.get("elapsed_ms")) or s["elapsed_ms"] <= 0:
            errors.append("sample %d: elapsed_ms missing" % n)
            continue
        if arm == "on":
            if not _is_num(s.get("restore_ms")) or s["restore_ms"] < 0:
                errors.append("sample %d (on p%d r%d): restore timing omitted" % (n, s["process"], s["request"]))
            elif s["restore_ms"] > s["elapsed_ms"]:
                errors.append("sample %d: restore_ms exceeds elapsed_ms" % n)
            rl = s.get("resident_layers")
            if not isinstance(rl, list) or len(rl) != RESERVE_RANKS or not all(_is_int(x) for x in rl):
                errors.append("sample %d (on): resident_layers must list %d ranks" % (n, RESERVE_RANKS))
            else:
                resident.append(min(rl))
        if not s["warmup"]:
            timed[arm].setdefault(s["process"], []).append(_frac(s["elapsed_ms"]))
    for arm in timed:
        if sorted(timed[arm]) != list(range(procs)):
            errors.append("%s arm has processes %s, expected 0..%d (truncated run?)" % (arm, sorted(timed[arm]), procs - 1))
        for proc, vals in sorted(timed[arm].items()):
            if len(vals) != requests:
                errors.append("%s process %d has %d timed requests, expected %d" % (arm, proc, len(vals), requests))
    if errors:
        return _reserve_invalid(errors)

    medians = {arm: [_median(timed[arm][p]) for p in range(procs)] for arm in timed}
    arm_ms = {arm: _median(medians[arm]) for arm in timed}
    tokens = Fraction(prompt_tokens)
    throughput = {arm: tokens * 1000 / arm_ms[arm] for arm in timed}
    tp_ratio = throughput["on"] / throughput["off"]
    lat_ratio = arm_ms["on"] / arm_ms["off"]
    pair_lat = [medians["on"][p] / medians["off"][p] for p in range(procs)]

    def spread(vals):
        mid = _median(vals)
        return float((max(vals) - min(vals)) / mid)

    out = {
        "schema": "ep-reserve-perf-verdict/1", "gate": gate, "mode": mode, "prompt_tokens": prompt_tokens,
        "binary_id": doc["binary_id"],
        "process_median_ms": {arm: [float(x) for x in medians[arm]] for arm in timed},
        "arm_median_ms": {arm: float(arm_ms[arm]) for arm in timed},
        "throughput_tok_s": {arm: float(throughput[arm]) for arm in timed},
        "throughput_ratio": float(tp_ratio), "latency_ratio": float(lat_ratio),
        "pair_latency_ratios": [float(x) for x in pair_lat],
        "noise_spread": {arm: spread(medians[arm]) for arm in timed},
        "min_on_resident_layers": min(resident) if resident else None,
        "samples": samples, "reasons": [],
    }
    kills = []
    if resident and reserve_resident_kills(min(resident), min_resident):
        kills.append("%d resident layers < %d" % (min(resident), min_resident))
    if gate == "pp32k":
        kills += reserve_pp32k_kill_reasons(tp_ratio, lat_ratio)
    if kills:
        out["verdict"], out["reasons"] = "KILL", kills
    elif gate != "pp32k":
        out["verdict"] = "REPORTED"
        out["reasons"] = ["%s carries no kill threshold in DECISION.md; ratios are reported, not judged" % gate]
    elif any(r > RESERVE_PP32K_LATENCY_CEILING for r in pair_lat):
        out["verdict"] = "INCONCLUSIVE"
        out["reasons"] = ["median passes but a paired process crosses the 1.05 latency ceiling (noise): no pass, no tuning queue"]
    else:
        out["verdict"] = "PASS"
    return out


def evaluate_reserve_c1(doc, min_resident=RESERVE_MIN_RESIDENT_LAYERS, min_pairs=3, timed=20, max_seq=16384):
    """Gate 3: fresh-server off/on pairs scored with the six weighted bbc1 categories (sampled, native MTP)."""
    errors = []
    if not isinstance(doc, dict):
        return _reserve_invalid(["c1 manifest is not an object"])
    if doc.get("schema") != RESERVE_C1_SCHEMA:
        errors.append("schema %r, expected %r" % (doc.get("schema"), RESERVE_C1_SCHEMA))
    _forbidden_reference_keys(doc, "manifest", errors)
    if not isinstance(doc.get("binary_id"), str) or not doc.get("binary_id"):
        errors.append("binary_id missing")
    if doc.get("max_seq") != max_seq:
        errors.append("max_seq=%r, the protocol is %d" % (doc.get("max_seq"), max_seq))
    if doc.get("spec") != "mtp":
        errors.append("spec=%r, the C1 protocol runs native MTP" % (doc.get("spec"),))
    sampling = doc.get("sampling") or {}
    if (sampling.get("temperature"), sampling.get("top_p"), sampling.get("top_k")) != (0.7, 0.95, 20):
        errors.append("sampling must be T=0.7 top_p=0.95 top_k=20, got %r" % (sampling,))
    servers = doc.get("servers")
    if not isinstance(servers, list) or not servers:
        return _reserve_invalid(errors + ["servers missing"])
    weights = None
    per = {"off": [], "on": []}
    orders = []
    for n, s in enumerate(servers):
        if not isinstance(s, dict) or s.get("arm") not in per or not _is_int(s.get("order")):
            errors.append("server %d: arm/order missing" % n)
            continue
        tag = "server %d (%s #%d)" % (n, s["arm"], s["order"])
        _forbidden_reference_keys(s, tag, errors)
        if s.get("binary_id") != doc.get("binary_id"):
            errors.append("%s: binary_id %r differs from the manifest %r (same binary required)" % (tag, s.get("binary_id"), doc.get("binary_id")))
        rl = s.get("resident_layers")
        if not isinstance(rl, list) or len(rl) != RESERVE_RANKS or not all(_is_int(x) for x in rl):
            errors.append("%s: resident_layers must list %d ranks" % (tag, RESERVE_RANKS))
            rl = None
        summary = s.get("summary")
        if not isinstance(summary, dict):
            errors.append("%s: summary missing" % tag)
            continue
        if summary.get("complete") is not True:
            errors.append("%s: summary is not complete (failed or truncated requests)" % tag)
        cats = summary.get("categories")
        if not isinstance(cats, dict) or sorted(cats) != sorted(RESERVE_SCORED_CATEGORIES):
            errors.append("%s: categories %r, expected exactly %s" % (tag, sorted(cats) if isinstance(cats, dict) else cats, list(RESERVE_SCORED_CATEGORIES)))
            continue
        w, decode, tau = {}, {}, {}
        for cat in RESERVE_SCORED_CATEGORIES:
            c = cats[cat]
            if not isinstance(c, dict) or c.get("n_timed") != timed or c.get("n_ok") != timed:
                errors.append("%s: %s has n_timed=%r n_ok=%r, expected %d successful timed requests" % (tag, cat, (c or {}).get("n_timed"), (c or {}).get("n_ok"), timed))
                break
            if not _is_num(c.get("decode_med")) or c["decode_med"] <= 0 or not _is_num(c.get("weight")) or c["weight"] <= 0:
                errors.append("%s: %s decode_med/weight missing" % (tag, cat))
                break
            t = (c.get("mtp") or {}).get("tau_window_weighted")
            if not _is_num(t) or t <= 0:
                errors.append("%s: %s tau_window_weighted not recorded" % (tag, cat))
                break
            w[cat], decode[cat], tau[cat] = _frac(c["weight"]), _frac(c["decode_med"]), _frac(t)
        else:
            if sum(w.values()) != 1:
                errors.append("%s: category weights sum to %s, not 1" % (tag, float(sum(w.values()))))
            if weights is None:
                weights = w
            elif w != weights:
                errors.append("%s: category weights differ between servers" % tag)
            score = sum(w[c] * decode[c] for c in w)
            claimed = summary.get("weighted_decode")
            if _is_num(claimed) and abs(float(score) - claimed) > 1e-6 * max(1.0, abs(claimed)):
                errors.append("%s: weighted_decode %r disagrees with the recomputed %.6f" % (tag, claimed, float(score)))
            orders.append((s["order"], s["arm"]))
            per[s["arm"]].append({"order": s["order"], "score": score, "decode": decode, "tau": tau, "resident": rl})
    if len(per["off"]) < min_pairs or len(per["on"]) < min_pairs or len(per["off"]) != len(per["on"]):
        errors.append("need >= %d fresh servers per arm and equal counts, got off=%d on=%d" % (min_pairs, len(per["off"]), len(per["on"])))
    arms_in_order = [a for _, a in sorted(orders)]
    if len({o for o, _ in orders}) != len(orders) or any(arms_in_order[i] == arms_in_order[i + 1] for i in range(len(arms_in_order) - 1)):
        errors.append("servers must be interleaved off/on in launch order, got %r" % arms_in_order)
    if errors:
        return _reserve_invalid(errors)

    off_score = _median([x["score"] for x in per["off"]])
    on_score = _median([x["score"] for x in per["on"]])
    ratio = on_score / off_score
    cat_ratio = {}
    tau_notes = []
    tau_out = {}
    for cat in RESERVE_SCORED_CATEGORIES:
        a = _median([x["decode"][cat] for x in per["off"]])
        b = _median([x["decode"][cat] for x in per["on"]])
        cat_ratio[cat] = b / a
        ta = _median([x["tau"][cat] for x in per["off"]])
        tb = _median([x["tau"][cat] for x in per["on"]])
        tau_out[cat] = {"off": float(ta), "on": float(tb)}
        if abs(tb - ta) / ta > RESERVE_TAU_NOTE_REL:
            tau_notes.append("%s tau %.3f -> %.3f" % (cat, float(ta), float(tb)))
    on_resident = [r for x in per["on"] for r in (x["resident"] or [])]
    out = {
        "schema": "ep-reserve-c1-verdict/1", "binary_id": doc["binary_id"],
        "weighted_off": float(off_score), "weighted_on": float(on_score),
        "weighted_ratio": float(ratio), "weighted_gain_pct": float((ratio - 1) * 100),
        "server_scores": {arm: [float(x["score"]) for x in sorted(per[arm], key=lambda x: x["order"])] for arm in per},
        "category_ratio": {c: float(r) for c, r in cat_ratio.items()}, "tau": tau_out,
        "tau_notes": tau_notes, "min_on_resident_layers": min(on_resident) if on_resident else None,
        "reasons": [],
    }
    kills = []
    if on_resident and reserve_resident_kills(min(on_resident), min_resident):
        kills.append("%d resident layers < %d" % (min(on_resident), min_resident))
    if reserve_c1_kills(ratio):
        kills.append("weighted sampled C1 gain %+.2f%% < +8%%" % float((ratio - 1) * 100))
    cliffs = [c for c, r in cat_ratio.items() if r < RESERVE_C1_CLIFF_RATIO]
    if kills:
        out["verdict"], out["reasons"] = "KILL", kills
    elif cliffs:
        out["verdict"] = "REVIEW"
        out["reasons"] = ["category cliff below %.2f on/off: %s (reviewer decision; guard is not a DECISION.md threshold)" % (float(RESERVE_C1_CLIFF_RATIO), ", ".join(cliffs))]
    else:
        out["verdict"] = "PASS"
    return out


# ---- CLI
def _reserve_write_json(path, payload):
    if path:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)
            handle.write("\n")


def _reserve_report(result, label, json_path):
    print("%s fn_ep_compare reserve %s: %s%s" % ("PASS" if result["verdict"] in ("PASS", "REPORTED") else "FAIL", label,
                                                 result["verdict"], "".join("\n    - " + r for r in result.get("reasons", []))))
    _reserve_write_json(json_path, result)
    return 0 if result["verdict"] in ("PASS", "REPORTED") else 1


def reserve_main(argv):
    parser = argparse.ArgumentParser(prog="fn_ep_compare.py reserve", description="EP reserve-decoupling gates (DECISION.md section 8)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    arms = sub.add_parser("arms", help="off/on oracle pair: exactness + plan/restore receipts + boundary fixtures")
    arms.add_argument("off_dir")
    arms.add_argument("on_dir")
    arms.add_argument("--min-resident", type=int, default=RESERVE_MIN_RESIDENT_LAYERS)
    arms.add_argument("--min-tokens", type=int, default=RESERVE_MIN_TOKENS)
    arms.add_argument("--expect-mode", choices=("ar", "mtp"), default=None)
    arms.add_argument("--require-lengths", default="", help="comma list of prompt token counts that must be present")
    arms.add_argument("--baseline-repeat", action="store_true", help="both dirs are switch-off runs: repeatability of the baseline")
    arms.add_argument("--json", default=None)
    arms.add_argument("--max-report", type=int, default=50)
    perf = sub.add_parser("perf", help="Gate 2 prompt-fill pair (ep-reserve-perf/1 JSON)")
    perf.add_argument("perf_json")
    perf.add_argument("--min-resident", type=int, default=RESERVE_MIN_RESIDENT_LAYERS)
    perf.add_argument("--procs", type=int, default=3)
    perf.add_argument("--requests", type=int, default=3)
    perf.add_argument("--json", default=None)
    c1 = sub.add_parser("c1", help="Gate 3 weighted sampled C1 (ep-reserve-c1/1 manifest)")
    c1.add_argument("manifest")
    c1.add_argument("--min-resident", type=int, default=RESERVE_MIN_RESIDENT_LAYERS)
    c1.add_argument("--min-pairs", type=int, default=3)
    c1.add_argument("--json", default=None)
    harvest = sub.add_parser("harvest", help="assemble reserve-receipt.json from `EP_RESERVE_RECEIPT` log lines")
    harvest.add_argument("log")
    harvest.add_argument("out_dir")
    harvest.add_argument("--model-sha256", default=None,
                         help="the gate's model hash (fn_ep_oracle runs with --skip-model-hash); fills a receipt without one")
    args = parser.parse_args(argv)

    if args.cmd == "arms":
        for d in (args.off_dir, args.on_dir):
            if not os.path.isdir(d):
                print("not a directory: %s" % d, file=sys.stderr)
                return 2
        try:
            lengths = [int(x) for x in args.require_lengths.split(",") if x.strip()]
        except ValueError:
            print("--require-lengths must be a comma list of integers", file=sys.stderr)
            return 2
        v, shape = compare_reserve_arms(args.off_dir, args.on_dir, args.min_resident, args.min_tokens, args.expect_mode,
                                        lengths, args.baseline_repeat, args.max_report)
        for name, fam in sorted(v.families.items()):
            print("%s %-10s checked=%d failed=%d" % ("PASS" if not fam["failed"] else "FAIL", name, fam["checked"], fam["failed"]))
        if v.first is not None:
            print("first mismatch [%s]: %s" % (v.first["check"], v.first["message"]))
        what = "off == off (baseline repeatability)" if args.baseline_repeat else "on == off"
        line = ("PASS fn_ep_compare reserve arms: %s, receipts, boundaries and ids verified" % what) if v.ok else \
            "FAIL fn_ep_compare reserve arms: %d failing check(s); first: %s" % (sum(f["failed"] for f in v.families.values()), v.first["message"])
        print(line)
        _reserve_write_json(args.json, {"verdict": "PASS" if v.ok else "FAIL", "summary": line, "shape": shape,
                                        "families": v.families, "first_mismatch": v.first, "mismatches": v.mismatches})
        return 0 if v.ok else 1
    if args.cmd == "harvest":
        try:
            with open(args.log, "r", encoding="utf-8", errors="replace") as handle:
                doc, seen = reserve_assemble_receipt(handle)
        except OSError as error:
            print("cannot read %s: %s" % (args.log, error), file=sys.stderr)
            return 2
        if args.model_sha256 and not doc.get("model_sha256"):
            doc["model_sha256"] = args.model_sha256
        target = os.path.join(args.out_dir, "reserve-receipt.json")
        if os.path.exists(target):
            print("keeping existing %s" % target)
            return 0
        if not seen:
            print("no %r lines in %s" % (RESERVE_RECEIPT_LINE.strip(), args.log))
            return 1
        _reserve_write_json(target, doc)
        print("wrote %s (%d record line(s))" % (target, seen))
        return 0
    path = args.perf_json if args.cmd == "perf" else args.manifest
    try:
        with open(path, "r", encoding="utf-8") as handle:
            doc = json.load(handle)
    except (OSError, ValueError) as error:
        print("cannot read %s: %s" % (path, error), file=sys.stderr)
        return 2
    if args.cmd == "perf":
        result = evaluate_reserve_perf(doc, args.min_resident, args.procs, args.requests)
        return _reserve_report(result, "perf", args.json)
    result = evaluate_reserve_c1(doc, args.min_resident, args.min_pairs)
    return _reserve_report(result, "c1", args.json)


# -------------------------------------------------------------- self-test
def _write(path, data, mode="w"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode) as handle:
        handle.write(data)


def _arm(root, name, shas, tokens, ranks=1, mirrors=None, state=None, flip_state_rank=None, raw=None, completed=True):
    d = os.path.join(root, name)
    meta = {"completed": completed, "ranks": ranks, "max_seq": 512, "tokens": len(shas) - 1, "vocab": 4,
            "model_sha256": "m", "prompts": [{"prompt": 0, "token_count": 3, "file_md5": "f", "ids_md5": "i"}]}
    _write(os.path.join(d, "meta.json"), json.dumps(meta))
    lines = []
    for step, sha in enumerate(shas):
        rec = {"kind": "logits", "prompt": 0, "step": step, "token": None if step == 0 else tokens[step - 1],
               "position": step + 3, "logits_sha256": sha}
        if ranks > 1:
            rec["rank_logits_sha256"] = [sha] + [(mirrors or shas)[step]] * (ranks - 1)
        lines.append(json.dumps(rec))
    _write(os.path.join(d, "logits.jsonl"), "\n".join(lines) + "\n")
    if raw:
        for step, values in raw.items():
            _write(os.path.join(d, "logits-p0-s%d.f32" % step), struct.pack("<%df" % len(values), *values), "wb")
    lines = []
    for rank in range(ranks):
        digest = [["position", "9"], ["gdn[0].recurrent", state or "s0"]]
        if rank == flip_state_rank:
            digest[1][1] = "deadbeef"
        lines.append(json.dumps({"kind": "state", "prompt": 0, "step": 1, "rank": rank, "digest": digest}))
    _write(os.path.join(d, "state.jsonl"), "\n".join(lines) + "\n")
    return d


def _quiet(*argv):
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        code = run(list(argv))
    return code, sink.getvalue()


def self_test():
    failures = []

    def expect(label, got, want):
        if got != want:
            failures.append("%s: got %r want %r" % (label, got, want))

    good = ["a", "b", "c"]
    with tempfile.TemporaryDirectory() as tmp:
        single = _arm(tmp, "single", good, [7, 8], raw={2: [1.0, 2.0]})
        ok = _arm(tmp, "ep_ok", good, [7, 8], ranks=3)
        out_json = os.path.join(tmp, "v.json")
        code, out = _quiet(single, ok, "--expect-ranks", "3", "--json", out_json)
        expect("equal", code, 0)
        if not out.strip().splitlines()[-1].startswith("PASS"):
            failures.append("equal run did not end with PASS: %s" % out)
        with open(out_json) as handle:
            expect("json verdict", json.load(handle)["verdict"], "PASS")

        expect("rank count", _quiet(single, ok, "--expect-ranks", "6")[0], 1)

        bad = _arm(tmp, "ep_bad", ["a", "b", "X"], [7, 8], ranks=2, raw={2: [1.0, 2.5]})
        code, out = _quiet(single, bad)
        expect("sha differs", code, 1)
        if "step=2" not in out or "max-abs=0.5" not in out or not out.strip().splitlines()[-1].startswith("FAIL"):
            failures.append("logits diff report wrong: %s" % out)

        expect("token differs", _quiet(single, _arm(tmp, "ep_tok", good, [7, 9], ranks=2))[0], 1)

        code, out = _quiet(single, _arm(tmp, "ep_mirror", good, [7, 8], ranks=3, mirrors=["a", "Y", "c"]))
        expect("mirror differs", code, 1)
        if "mirrors" not in out:
            failures.append("mirror disagreement not reported: %s" % out)

        code, out = _quiet(single, _arm(tmp, "ep_state", good, [7, 8], ranks=2, flip_state_rank=1))
        expect("state differs", code, 1)
        if "gdn[0].recurrent" not in out:
            failures.append("state diff row not named: %s" % out)

        expect("incomplete", _quiet(single, _arm(tmp, "ep_inc", good, [7, 8], ranks=2, completed=False))[0], 1)
        expect("missing dir", _quiet(single, os.path.join(tmp, "nope"))[0], 2)
        empty = os.path.join(tmp, "empty")
        os.makedirs(empty)
        expect("empty arm", _quiet(single, empty)[0], 1)

        # EP-R reference (--ref-ranks R > 1)
        ref2 = _arm(tmp, "ref_ep2", good, [7, 8], ranks=2)
        ep4 = _arm(tmp, "ep4", good, [7, 8], ranks=4)
        code, out = _quiet(ref2, ep4, "--ref-ranks", "2", "--expect-ranks", "4")
        expect("ref-ranks 2 vs EP4", code, 0)
        if "EP4 == EP2 reference" not in out.strip().splitlines()[-1]:
            failures.append("EP-R summary wording wrong: %s" % out)
        expect("ref-ranks default vs EP-R reference", _quiet(ref2, ep4)[0], 1)
        code, out = _quiet(ref2, ep4, "--ref-ranks", "3")
        expect("reference rank-count mismatch", code, 1)
        if "reference arm ran 2 ranks, expected 3" not in out:
            failures.append("reference rank-count mismatch not reported: %s" % out)
        code, out = _quiet(ref2, _arm(tmp, "ep2", good, [7, 8], ranks=2), "--ref-ranks", "2")
        expect("EP not larger than reference", code, 1)
        if "more than the 2-rank reference" not in out:
            failures.append("EP-not-larger not reported: %s" % out)
        code, out = _quiet(_arm(tmp, "ref_mirror", good, [7, 8], ranks=2, mirrors=["a", "Y", "c"]), ep4, "--ref-ranks", "2")
        expect("reference mirror differs", code, 1)
        if "reference prompt=0 step=1" not in out or "mirrors" not in out:
            failures.append("reference mirror disagreement not reported: %s" % out)
        code, out = _quiet(_arm(tmp, "ref_state", good, [7, 8], ranks=2, flip_state_rank=1), ep4, "--ref-ranks", "2")
        expect("reference rank-1 state differs", code, 1)
        if "reference prompt=0 step=1 rank 1" not in out or "gdn[0].recurrent" not in out:
            failures.append("reference rank-1 state diff not reported: %s" % out)

    for line in failures:
        print("SELF-TEST FAIL:", line)
    if failures:
        return 1
    print("self-test PASS")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--self-test":
        sys.exit(self_test())
    sys.exit(run(sys.argv[1:]))
