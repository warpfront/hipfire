# slowlink-gap: S1 pp16384 >= 2000 tok/s ABBAAB qualification harness (performance only)

Adapted from the untracked `gap2000/scripts/gap-{cell,job,launch,wait}.sh`; deployed via git (push, then a worktree on
hipx), never rsynced as source. CPU scripts: `gap_lib.py gap_stage.py gap_plan.py gap_score.py gap_selftest.py`.
GPU scripts (run only inside a fabric hold on hipx/xtx): `gap-launch.sh -> gap-job.sh -> gap-cell.sh`, `gap-wait.sh`.

Contract encoded here: six fresh `hipfire bench` processes ABBAAB = A1 B1 B2 A2 A3 B3, each `--matrix --pp 16384 --tg 1
--ctx 1 --runs 3 --warmups 1 --max-seq 16416 --json` (3 daemon-measured prefill rates/cell, flat 9/arm). Arm A = all S1
levers off; arm B = placement spread-list + IU4 panels (candidate combination `place_iu4`, QSA off, inert Q8w omitted).
Both arms: same stage, same `candidate.toml` (S1 enabled by config), same
env `HIPFIRE_STATE_QUANT=q8 HIPFIRE_KV_MODE=q8 HIPFIRE_QWEN4_D2_DEFAULT=0 HIPFIRE_QWEN4_QSA_WMMA_GATHER=0`,
no f32 carry/global f32/q8w/trace.
Only `HIPFIRE_QWEN4_SLOWLINK_RESIDENT_LAYERS` and `HIPFIRE_QWEN4_MOE_SYM_IU4` vary.
Fabric: class timing, card xtx, 22.4 GiB, host-mem 67 GiB lock, max 1800 s, P1. Own HOME/config/cache/locks per cell,
UUID `GPU-43390a851e296ee5`, BDF `0000:67:00.0`, MemAvailable >= 80 GiB per cell.

## Flow (hipx; the harness checkout is the pushed worktree)

```
H=<hipx worktree>/scripts/slowlink-gap
export GAP_RUNS=/home/kaden/qcal/release-0.4.3/s1-qualified-levers-001/runs  # fresh owned outputs, not archived runs
A=/home/kaden/qcal/release-0.4.3/slowlink-prefill/gap2000/artifacts/<NAME>   # rsync -a ARTIFACTS hipx:$A  (no model)
# 0. CPU self-test of the harness python (fabricated artifacts, no GPU, no model)
python3 $H/gap_selftest.py
# 1. symlink-only stage + read-only identity manifest (same script local or remote; no payload copied)
python3 $H/gap_stage.py --out $S/stage-<tag> --cli $A/hipfire --daemon $A/daemon \
    --kernels $A/kernels/compiled/gfx1100 --pack-manifest $A/<pack>.manifest.json \
    [--pack-tar-sha256 <hex>] [--probe qwen4_slowlink_quality=$A/qwen4_slowlink_quality]
# 2. plan + pins (stage identity, config sha, model cached stat, env, synthetic digest) in a FRESH read-only dir
python3 $H/gap_plan.py make --stage-root $S/stage-<tag> --out-dir $P/plan-<tag> [--place 0,3,7,15,19,23,27,39,43]
# 3. detached launch (nohup setsid, fds>2 closed) + detached waiter; run dir must not exist
bash $H/gap-launch.sh <runname> $P/plan-<tag>           # -> $GAP_RUNS/<runname> (default gap2000/runs)
bash $H/gap-wait.sh --detach <runname>                  # -> <run>/waiter.log, waiter.exit
# 4. CPU scorer on the (rsynced) run dir
python3 $H/gap_score.py $GAP_RUNS/<runname> --out $GAP_RUNS/<runname>.score.json
```

`gap_score.py` exit: 0 PASS (candidate flat median over all nine >= 2000), 1 FAIL_GATE, 2 INVALID (contract assertion),
3 INCOMPLETE (XTX drop or partial run: stop, tell the parent, no retry). The score json keeps every rate sample,
per-process medians, applied env per cell, route lines, stage identity, and the synthetic stream digest
(sha256 of u32le ids `10 + (i % 1000)`, i < 16384, the daemon's `bench_prefill` stream).

Receipts: each run lives in a fresh directory (`runs/<name>`), the job writes `cells/MANIFEST.sha256` and makes `cells/`
read-only. Cell failure: no automatic rerun. Original gap2000 artifacts are untouched.

The 2000 tok/s verdict remains a measured follow-up target, not an extra base-S1 release hold. The historical
2054.2 tok/s full-stack diagnostic included the failed QSA route and is not qualification of this candidate.

## Native quality reports

Fresh combined quality must pass **before** ABBAAB. The archived IU4 pass used front placement, not the PLACE
spread-list; separate PLACE and IU4 passes do not establish their composition on the current binary.
Use one current lab executable and matching candidate pack, retaining the immutable beta reference:
`STAGE` is the captured stage root; `QUALITY`, `CFG` and `PIN` are absolute paths. Use the actual executable hash
and the campaign hash printed by `prepare`. The foreground serial controller must finish cleanly before reporting.

```
python3 -B $H/quality_campaign.py prepare --root $QUALITY --arms off,place_iu4 \
  --exe $STAGE/bin/qwen4_slowlink_quality --expect-exe-sha256 $EXE_SHA --config $CFG --source-pin $PIN
python3 -B $H/quality_campaign.py check-plans --root $QUALITY --expect-campaign-sha256 $CAMPAIGN_SHA
python3 -B $H/quality_campaign.py serial --root $QUALITY --expect-campaign-sha256 $CAMPAIGN_SHA --priority P1
python3 -B $H/quality_campaign.py score --root $QUALITY --expect-campaign-sha256 $CAMPAIGN_SHA --arm off
python3 -B $H/quality_campaign.py score --root $QUALITY --expect-campaign-sha256 $CAMPAIGN_SHA --arm place_iu4
python3 -B $H/quality_campaign.py report --root $QUALITY --expect-campaign-sha256 $CAMPAIGN_SHA
```

These are full fresh `s1a+s1b` phases for OFF and PLACE+IU4, not a new battery or a rerun of accepted base-S1
qualification. Require actual N9/768-row S1, the explicit PLACE set, `aggregate-sym-iu4` proof, same-route
determinism and decoded-text review. The fixed context-mean KL maxima are 0.03948458079446915 (16K) and
0.027284316982763658 (4K). QSA failed 4K (0.029107700567698017); it is excluded, not waived.
The current gather key `HIPFIRE_QWEN4_QSA_WMMA_GATHER` is pinned `0`; the historical FN key is inert.

`quality_campaign.py` seals its Python sources and the native example source in each campaign. All campaign-loading
commands verify those hashes; launch, run-phase and scoring cannot substitute another harness. Determinism consumes
the native `hipfire.qwen4.slowlink_quality.determinism.v1` schema: `finish_reason` must be a two-string equal pair
(there is no native `finish_equal` field). Token equality and logit bit identity remain separately reported.
Kernel-pack binding is fail-closed on actual runtime evidence, never the sealed stage manifest alone: each
`run-phase` installs the sealed pack at the resolver's `~/.hipfire/bin/kernels` lookup (the existing gap-cell
private-`HOME/bin/kernels` pattern; no global compiler change) and records the install link plus a per-phase
`pack_binding` verdict in `run-record.json`. The verdict requires the `pre-compiled kernels:` cold-lookup line to
name the sealed pack dir (or the phase-private kernel cache), a nonempty set of `indexed package verified:` hits
each resolving under the sealed pack dir and verifying against its sealed `<module>.index.json` module/object SHA
(plus the sealed campaign manifest when present), a live-rescanned empty phase-private `kcache` (recorded names
must match the rescan; any `.hsaco` there is a runtime JIT/pack miss), and no
compiler-fallback markers (`packaged object rejected`, `falling back to JIT`, stale/unverified image lines) in
stderr. `report` re-checks the same evidence per phase, carries `pack_ok` per arm, and folds it into
`mechanical_pass`; any runtime JIT, fallback or unbound used-code identity fails the phase and the report.

For an explicitly versioned, CPU-only correction of historical reports, retain the original campaign and harness
checkout unchanged and use the new reporter with:

```
python3 -B $NEW_H/quality_campaign.py report --root $OLD_CAMPAIGN \
  --expect-campaign-sha256 $OLD_SHA --arms off --recorded-harness-dir $OLD_H \
  --out-dir $OLD_CAMPAIGN/report-corrected-$FIX_COMMIT
```

`OLD_H` is the directory containing the original `quality_campaign.py` and `quality_scorer.py`, in its original
repository layout (the native example is also hashed). Its hashes must match the old sealed campaign. The summary
records old source pins and hashes separately from the new reporter's commit, tree and hashes. Historical override
is accepted only by `report`, requires an explicit fresh absolute output directory and does not change any original
campaign, phase, score or report. `--out-dir` also permits fresh per-arm reports for new campaigns. Manual decoded-text
review remains required; a mechanical report is not final qualification. Prepare remaining GPU arms as a new campaign
