# KILL — FA sigmoid + AWQ rotate/int4 producer fusion

## Candidate

The exact gfx1201 iu4 FA-wo producer folded `sigmoid_mul_f32(attn, gate)` into the existing `rotate_x_mq_awq_i4_gfx12` lane and emitted the unchanged f32 rotated output plus block-i4 sidecar in one kernel/dispatch. It was implemented only in worktree `wt-prod` on branch `gfx1201-prod-iu4`; it is not committed because it misses the ship gate.

Pinned binaries (`binaries.md5`):

- CLI: `a3f7b94aab213dadcf8f0d5a78c0aefa`
- baseline daemon: `56462d2e844d33241fa103d7db8dadb7`
- candidate daemon: `8e078f0bdd0fa729e3877e94de0374a2`

## Exactness

`oracle-full.log`: PASS on real gfx1201 against the device-resident standalone chain. All **786,432 f32 rotated values** and **442,368 block-i4 bytes** were bit-identical at the first FA layer (`K=6144`, `N=128`), and the candidate preserved its attention input.

## Pinned-daemon A/B/A/B

Command for every timed row:

```text
hipfire bench qwen3.8:27b-mq4-xt --matrix --pp 512,2048,8192,32768 --ctx 128 --tg 64 --spec off --runs 3 --warmups 1 --kv-mode fp8 --json
```

Every row used explicit `HIPFIRE_DAEMON_BIN`; the candidate was separately warmed in `warm-candidate.json`. Both pairs meet the decode gate (>=36.4 tok/s and baseline/candidate within 1%).

| pair | arm | pp512 | pp2048 | pp8192 | pp32768 | decode |
|---|---|---:|---:|---:|---:|---:|
| 1 | baseline (`baseline-1.json`) | 2343.3 | 2416.2 | 2318.8 | 1941.5 | 36.5206 |
| 1 | candidate (`candidate-1.json`) | 2354.7 | 2436.9 | 2340.6 | 1949.8 | 36.5456 |
| 1 | delta | +0.486% | +0.857% | **+0.940%** | +0.428% | +0.068% |
| 2 | baseline (`baseline-2.json`) | 2349.4 | 2428.8 | 2333.1 | 1944.0 | 36.5165 |
| 2 | candidate (`candidate-2.json`) | 2362.3 | 2436.2 | 2339.9 | 1949.6 | 36.5210 |
| 2 | delta | +0.549% | +0.305% | **+0.291%** | +0.288% | +0.012% |
| mean | baseline | 2346.35 | 2422.50 | 2325.95 | 1942.75 | 36.5185 |
| mean | candidate | 2358.50 | 2436.55 | 2340.25 | 1949.70 | 36.5333 |
| mean | delta | +0.518% | +0.580% | **+0.615%** | +0.358% | +0.040% |

## Verdict

**KILL / do not merge.** The candidate is exact and no matrix point regresses, but both independent pp8192 pairs are below the mandatory **+1.5%** ship threshold (+0.940%, +0.291%; mean +0.615%). The trace already bounded the standalone launch at only 1.932566 true one-pass us/token, so this result closes the remaining FA sigmoid producer dispatch rather than justifying permanent code.
