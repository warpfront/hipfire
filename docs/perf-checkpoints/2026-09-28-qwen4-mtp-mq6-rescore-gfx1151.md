# Qwen4 MTP draft re-score on an MQ6G256V2 LM head (gfx1151) — 2026-09-28

**Lifecycle:** `historical`

**Disposition:** measured local A/B on `gfx1151`. Not a G5 admission, not a
retained-replay certification, not a cross-architecture result, not a product
speed-floor update.

**Change under test:** `86461c4a1` feat(qwen4): exact top-8 MTP draft re-score
for an MQ6G256V2 LM head. The MTP drafter ranks the vocabulary with an MQ2 copy
of the LM head and re-scores that copy's top 8 against the real head. Before
this change the re-score required a Q8_0 head. The canonical artifact since
2026-09-28, `qwen3.8-flash-next.mq4` (recipe r2), ships its head as MQ6G256V2,
so its drafts were the MQ2 copy's plain argmax.

## Fixture

- Host `halo`: gfx1151 (Strix Halo, 128 GB UMA), HIP 7.2. The GPU was shared
  with an external `llama-server`.
- Model: `qwen3.8-flash-next.mq4`, 125288540696 bytes, sha256
  `cd7cbb911d3d016e034b1d22be1be37b42873a21699c94f09337528f1eee9db6`.
- Q8-head arm: the same file with only `lm_head.weight` replaced by Q8F16
  (qt 3), quantized from the BF16 source checkpoint. Its head bytes (md5
  `9bb952a25442dc760efdc5a343a4edb6`) are identical to the head of
  `qwen3.8-flash-next.mq6q8-pleq8`. It was a local, temporary artifact and has
  been deleted.
- Daemon md5 `d0fc9051a4244f131871e601773ac536` (source of `86461c4a1`).
- Native daemon protocol, driven directly (`hipfire bench` cannot load this
  model; see `AGENTS.md`):
  - Load: `max_seq` 2048, KV q8, `mtp_mode` off/on, `mtp_k` 3.
  - Generation: greedy, `max_think_tokens` 1, `max_tokens` 200.
  - Environment: `HIPFIRE_GRAPH=0 HIPFIRE_AR_GRAPH=0 HIPFIRE_CASK_OFF=1`.
- Every session was a fresh daemon under the GPU lock. It ran one warmup
  prompt (`prose_river_short`, md5 `07a7880965142971dbb3cc7493f8fb94`) and then
  the nine measured prompts:

  | prompt | md5 |
  |---|---|
  | `fiction_lighthouse` | `d3768bd6479ce9aae08c9e0152b39e06` |
  | `merge_sort_thinking_off` | `253c7ac50857fe6d0e10fb0d2c5e35c0` |
  | `humaneval_3_below_zero` | `37c5aad9f9efe93b5c47f27256bdf149` |
  | `mixed_code_then_prose` | `7b0278b4cb82802847d7ce338a85dcb3` |
  | `trains-meet` | `db92b572702ab947c5fabd9c342eb616` |
  | `code_edit_rewrite_copy` | `80b910784c456bbb5532eb0493a497b9` |
  | `bare_factual` | `1d32df5f12c414d3e34c7b35b6611e6c` |
  | `glimmer_prefill_1024` | `0ee8f86ada3683eda452bc294ec824a9` |
  | `lru_cache_pep8_strict` | `df5dedc8040ce70ba55080c4548e6024` |

- Three arms × (AR, MTP) × 3 rounds, with the arm order rotated every round.
  The decode rate is the streamed rate from the first to the last token of
  each prompt.

## Result

Geomean over the nine prompts of each prompt's best-of-3 decode tok/s:

| arm | AR | MTP | mean tau | MTP / AR |
|---|---:|---:|---:|---:|
| MQ6 head, no re-score (`HIPFIRE_MTP_DRAFT_HEAD=mq2`) | 33.69¹ | 55.40 | 1.89 | 1.64 |
| **MQ6 head + MQ6 re-score (default `mq2r`)** | 33.69¹ | **59.47** | **2.10** | **1.77** |
| Q8 head + Q8 re-score | 33.14 | 59.18 | 2.09 | 1.79 |

¹ AR does not depend on the draft head. The two MQ6 arms are one AR population
(best of six sessions).

- **MTP per round (geomean):**
  - MQ6 with re-score: 55.81, 58.89, 59.30. The first round lost
    `bare_factual` to contention (38.9).
  - MQ6 without re-score: 55.20, 55.15, 55.08.
  - Q8 head: 58.27, 59.12, 58.93.
- **Where the change acts:** the gain is on prose and long prompts; code
  prompts are within about 3%.

  | prompt | MTP tok/s before → after | tau before → after |
  |---|---:|---:|
  | `fiction_lighthouse` | 38.9 → 47.6 | 1.01 → 1.21 |
  | `trains-meet` | 55.3 → 64.4 | 1.65 → 2.26 |
  | `lru_cache_pep8_strict` | 55.6 → 64.5 | 1.70 → 2.65 |
  | `humaneval_3_below_zero` | 53.3 → 51.9 | 1.82 → 1.87 |
  | `bare_factual` | 50.5 → 50.1 | 1.46 → 1.38 |

- **The head's own AR cost:** an earlier two-arm run of the same method
  (daemon at `daba98fe9`) measured AR 33.84 with the MQ6 head against 33.33
  with the Q8 head, which is −1.5% for the Q8 head. It reads about 0.18 GB more
  per token.
- **Correctness:** in every session of every arm, the greedy MTP token ids
  equal the AR ids. `qwen4_rows` at 1, 4 and 8 rows is bitwise equal to
  single-token decode on `mq4`.

## Caveats

- Several AR samples show external GPU contention (28–30 tok/s against a
  steady ~33.7). Best-of-3 is used for that reason. Per-session raw reports
  are local: `.codeinsight+research/qwen4/mtp-20260927/head_ab_mq6r.json` and
  `head_ab.json` (gitignored).
- These are fixture-bound results. The margin thresholds of the draft stop
  were calibrated on Q8-head margins and were not re-tuned for the MQ6 head.

## Validation at `bd0570508` (daemon md5 `8b23b3658514a8b94524c644af82492b`)

This commit follows `86461c4a1`. It moves the MTP head's HC read onto the
trunk's shared hyper-read op, and on gfx1151 it launches the same kernels.

- **Draft identity:** the 9-prompt exact sweep gives MTP ids equal to AR's,
  and every prompt's tau is identical to the measured `86461c4a1` run.
- **`serve_harness.py` battery** on `mq4` (MTP on, greedy, thinking off,
  kv q8, `max_seq` 2048): 5/5 turns `finish=stop`, with no runaway, empty or
  attractor turns. The answers were read and are coherent.
- **`redline_daemon_harness.py --qwen4 --pm4 --skip-prefill
  --shadow-iterations 5`** on `mq4`:
  - Capture: stable, 1069 launches, 32 kernels, hash `2cca77287bdc22bd`.
  - AQL contract probe: 32 kernels.
  - Qwen4 shadow parity: HIP, retained PM4 and blob are bit-exact at
    positions 129–133 over 126,623,888 state bytes per position; harness
    failures `[]`.
  - Two setup notes:
    - The harness's recipe preflight had to admit recipe r2's MQ6G256V2 head.
    - The prefill bench arm is not implemented for Qwen4, hence
      `--skip-prefill`.
  - Raw report: `.codeinsight+research/qwen4/pr774-final/redline-qwen4-pm4-5pos.json`
    (sha256 `4383bb22f002a7bfdb190afda6a1621d6c078edbea1ce48337be33002980efb5`,
    gitignored).
