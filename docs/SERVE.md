# Serve (OpenAI-compatible HTTP)

`hipfire serve` starts the native Rust HTTP control plane, owns one GPU daemon
process, and exposes an OpenAI-shaped chat API. Defaults come from typed TOML
configuration ([CONFIG.md](CONFIG.md)); the HTTP surface is implemented by
`crates/hipfire-cli` and the shared transport lives in `crates/hipfire-client`.

| Field | Default (source) |
|---|---|
| Bind host | `serve.host = "127.0.0.1"` (loopback only) |
| Port | `serve.port = 11435` |
| Pre-warm model | `serve.default_model = "qwen3.5:9b"` or a positional model arg |
| Idle unload | `serve.idle_timeout_seconds = 300` (`0` = never) |
| Max request body | `serve.max_request_bytes = 67108864` (64 MiB) |
| Admission queue | `serve.max_queue = 64`, `serve.queue_timeout_ms = 600000` (10 min) |
| Prefix cache | `serve.prefix_cache = false` (opt-in; paged slots, text only) |
| Pid / log | `~/.hipfire/serve.pid`, `~/.hipfire/serve.log` |

Truth state: **shipped / ref-pinned** for the HTTP contract and lifecycle
below. Branch-only model routes (for example LFM framing details) are not
implied by this page — see [MODELS.md](MODELS.md) and [VALIDATION.md](VALIDATION.md).

## Security (bearer authentication and HTTPS proxies)

The API key is **optional**. Without one, every route is unauthenticated
(including on a non-loopback bind, where serve starts and prints one line
recommending a key); anyone who can reach the bind address can call every
endpoint, including chat completions. A key is **recommended** whenever you
bind a non-loopback address. Configure `serve.api_key_file` (or
`HIPFIRE_SERVE_API_KEY_FILE`), or pass `--api-key-file <absolute-path>`. Only
the path is stored in configuration. Local CLI/TUI clients read the configured
file automatically. Native Windows does not generate a key for you: create the
file yourself.

The file must contain at least 32 bytes of printable ASCII after trimming trailing
whitespace; a space, line break, control character or non-ASCII character inside the
key is refused at startup with an error naming the file and the character position
(the key is sent as an HTTP header value).
On Unix, group/other permissions are refused: use `chmod 600`. Windows ACLs are
not checked yet; restrict access to the account running serve yourself.
For example, create a private random token and persist its **path**, not its contents:

```bash
umask 077
openssl rand -hex 32 > "$HOME/.hipfire/api-key"
hipfire config set serve.api_key_file "$HOME/.hipfire/api-key"
hipfire serve 127.0.0.1:11435        # key from serve.api_key_file
```

With a key configured, every route requires `Authorization: Bearer <key>`,
except `/health`: without valid credentials it returns only minimal liveness
and version, not process identity or model details. Authenticated `/health`
retains detailed diagnostics. Missing/wrong credentials receive the same 401.
The existing instance token identifies a process; it is not the API credential.

Authentication runs before routing or request-body reads, including UI assets
and `OPTIONS` requests. A 401 has an empty body and `WWW-Authenticate: Bearer`.
The embedded UI cannot supply credentials and therefore receives 401 when a
key is configured; browser pairing is not implemented. Without a key it works
unchanged. Existing successful responses retain permissive CORS
(`Access-Control-Allow-Origin: *`), but an unauthenticated preflight is
refused when a key is configured. Native clients do not need CORS.

Serve itself has **no TLS**. A bearer on plaintext HTTP can be intercepted.
Use loopback behind verified HTTPS for remote access; never publish the raw
port to the internet. Clients support `http://` and verified `https://` URLs
(including proxy base paths), with no insecure-certificate fallback.

### Tailscale Serve (tailnet-only HTTPS)

Keep the backend bound to loopback, with bearer authentication enabled (recommended):

```bash
hipfire serve 127.0.0.1:11435 --api-key-file "$HOME/.hipfire/api-key"
tailscale serve --bg http://127.0.0.1:11435
```

Use the HTTPS URL printed by Tailscale Serve and supply the same bearer to
your client. Retain tailnet ACLs/grants as well as app authentication.
Do **not** enable public Funnel by default. For a manual LAN proxy, use
HTTPS with a trusted certificate and forward the Authorization header.
The embedded browser UI does not provide credential pairing; use a
bearer-capable API client for protected access.

## Start and stop

```bash
hipfire serve                         # foreground; Ctrl-C stops (default bind 127.0.0.1)
hipfire serve 127.0.0.1:11435         # loopback-only (preferred local bind)
hipfire serve -d                      # background (setsid/nohup); polls /health up to 60s
hipfire serve qwen3.5:9b -d           # pre-warm a specific tag this run
hipfire serve 0.0.0.0:11435 --api-key-file /absolute/private/key -d # key recommended on a non-loopback bind; see Security
hipfire serve --no-prewarm -d         # bind first; load on first request
hipfire serve --kv-mode q8 --idle-timeout 0 -d
hipfire serve --tp 2 -d               # expert-parallel load (multi-GPU arches)
hipfire serve --spec off -d           # plain AR (no MTP/DFlash) for every load, e.g. an A/B
hipfire serve --max-seq 32768 -d      # KV context for every load (replaces memory.max_seq)

hipfire ps                            # tracked process liveness + port state
hipfire stop                          # tracked serve only: SIGTERM (Unix), graceful HTTP (Windows)
# Forceful — inspect first (hipfire ps, curl /health):
hipfire stop --force                  # also kill this install's untracked daemons (TERM, KILL after 5s)
hipfire stop --all                    # --force plus pkill orphan quantize jobs
hipfire restart -d                    # stop --force semantics, then serve again
```

`serve -d` first checks `serve.pid` and `/health`. If a tracked server already
answers on the same port it prints `hipfire serve already running (pid N, port
P, log …)` and exits 0; a tracked server on another port, or a different
process answering on the port, exits 1 with that reason. Readiness requires
the `/health` `pid` to be the child it just spawned, and if the child exits
early the error includes the last `FATAL`/`hipfire:` line from `serve.log`.

**Forceful lifecycle flags (Unix).** Plain `hipfire stop` only SIGTERMs the
tracked pid after ownership checks — use that by default. `--force`, `--all`,
and `restart` also stop this install's other daemons: PIDs from the pidfile and
the GPU lock, each verified by command line (this install's daemon binary)
and matching `HIPFIRE_HOME`, sent SIGTERM and then SIGKILL after 5 s. They no
longer kill processes by name or by port; if something else still holds the
port, the CLI prints `port N is still in use by a process that is not this
install's hipfire; not killing it`. `--all` also runs `pkill -f` against
release quantize binaries.

**Local management (Windows).** Windows has no signal `hipfire stop` can
send, so each Windows serve process binds a separate `127.0.0.1` listener on
an OS-assigned port. Only that listener accepts `POST /api/v1/shutdown`,
authenticated with its own random 32-byte bearer credential. The inference
listener never serves this route, including requests forwarded by a
localhost reverse proxy. The management credential is separate from the
public instance-identity token, which is not authorization. A Windows
`serve.pid` stores `manage_port` and `manage_token` and inherits ACLs from the
user's hipfire home. Keep it private: never share its contents or forward the
management listener through a proxy. Unix serves bind no management listener
and their `serve.pid` is unchanged.

**Windows stop.** After checking image, creation time, and port ownership,
`hipfire stop` sends the local management request and waits up to 35 seconds
for exit. Shutdown cancels the existing token and drains inference requests
for at most 30 seconds. If exit times out, the command reports an error rather
than force-terminating the process. Old pid records without management fields
require Ctrl-C and a restart with the updated CLI. Windows `--force` / `--all`
do not reap daemons when ownership cannot be proven from process environment;
the CLI explains this conservative skip and never kills by name or port.

Flags (also accepted by `restart`). In detached mode (`-d`) the parent forwards
these to the foreground child as CLI arguments; none of them is turned into an
environment variable.

| Flag | Effect |
|---|---|
| `-d` / `--detach` / `--background` | Fork detached child; log → `~/.hipfire/serve.log` |
| `--model <m>` | Model tag/path to pre-warm for this process (same as the positional model) |
| `--api-key-file <absolute-path>` | Protected printable-ASCII bearer file; overrides `serve.api_key_file` for this process |
| `--no-prewarm` | Bind first and load on the first request instead of pre-warming |
| `--kv-mode <m>` | KV preset for models this service loads (sent as the `kv_mode` load param; same contract as `run --kv-mode`) |
| `--kv-k <f>` / `--kv-v <f>` | Qwen-only K / V format override for models this service loads |
| `--kv-backend legacy\|vmm` | KV storage backend for models this service loads |
| `--max-seq N` | KV context for every load, within the `memory.max_seq` bounds (512..1048576); overrides the configured/automatic context |
| `--spec <mode>` (alias `--speculation`) | Speculation mode for every load: `off`, `auto`, `ngram`, `dflash`, `mtp`, `dspark`; `off` disables MTP/DFlash |
| `--idle-timeout <s>` | Idle model-unload timeout, 0..86400 s (`0` = never); overrides `serve.idle_timeout_seconds` |
| `--tp N` | Tensor/expert-parallel degree (1..64) |
| `--continuous-batch-size N` | Number of eligible concurrent batched lanes, 1..256 (`1` keeps sequential behavior), subject to memory admission; same range as the `serve.continuous_batch_size` config key. The runtime batch width is dynamic, and the 128/63 chunk sizes are not lane limits |
| `--vision <path>` | Vision-tower sidecar wired into every model load; overrides the registry `vision` slot and `HIPFIRE_VISION_SIDECAR`. Honored by foreground serve only: `-d` does not forward it to the child |
| `--ui` | Serve the embedded chat UI (see below) |
| `--open` | Open the chat UI in the default browser once serving; implies `--ui` |

Positional forms: `[model] [host] [port]`, or `host:port` / `[ipv6]:port`.
Without a model arg, pre-warm uses `HIPFIRE_MODEL` if set, else
`cfg.default_model`.

Config and env owners for bind, idle, queue, and body limits:
[CONFIG.md](CONFIG.md), [env-vars.md](env-vars.md).

## Lifecycle

1. **Bind + pid record.** Serve writes its numeric pid to
   `~/.hipfire/serve.pid`. `stop` verifies that `/proc/<pid>/cmdline` still
   identifies a native `hipfire serve` process before signaling it.
2. **Daemon.** Spawns the Rust `daemon` example over stdio JSON.
3. **Pre-warm (default).** Loads the chosen model asynchronously. Failures log
   and leave the process serving; the model loads on the first real request.
   A failure the config itself decides — a load-time refusal such as
   `dflash_mode=on` with no DFlash draft and no MTP head — is not transient:
   every later request reloads and is refused the same way, while `/health`
   still answers `ok`. Fix the config (or pass `developer.dflash_draft`)
   rather than retrying. The same holds for the partial-offload refusals, each
   of which names its knob on screen: a pinned `memory.gpu_layer_budget` /
   `memory.moe_expert_budget` placement that does not fit device weights, a
   spill that exceeds pinned host RAM (+ 4 GiB headroom) or the GTT
   `pages_limit`, an unpackable host-placed expert layer (neither the uniform
   nor the graded packer applies), host-placed packed AWQ experts under
   `memory.offload_exec=cpu`, a configured spill on a non-spillable load
   (`tp > 1`, `pp > 1`, EP, VL tower, paged experts, REAP keep-map, or a
   non-Qwen3.5 arch), and `memory.offload_exec=cpu` together with a
   retained-replay (Redline) backend.
   Multi-slot serve (`serve.multi_slot = true`) fails closed instead: the slot
   engine is its only backend, so a failed pre-warm (for example, slot-engine
   allocations that do not fit beside another process on the card) logs the
   load error, answers `/health` with 503 `unhealthy`, and exits 1.
4. **HTTP.** The native server accepts traffic. Serve is single-stream by
   default: one generation holds the daemon at a time and later requests wait
   in a bounded queue (`serve.max_queue`). Continuous batching covers only
   thinking-off, single-turn Qwen requests (`serve.continuous_batch_size`).
   The VMM continuous-batching route (`serve.vmm_batch`,
   `HIPFIRE_SERVE_VMM_BATCH`, **default off, experimental**) instead batches
   eligible TP1 Qwen3.5-family requests over the one resident weight set with
   per-request VMM KV/DeltaNet owners (no fixed lanes, no second model). The
   loaded ack then reports `continuous_batch_route: "vmm"` and the actual
   owner receipt (`vmm_batch_kv_backend`, `vmm_batch_kv_mode`,
   `vmm_batch_max_seq_bound`). By default it runs the **exact** route
   (per request byte-identical to the singleton route; prefill uses the
   singleton's own chunking, so a long prompt prefills in singleton-sized
   chunks), staged only where the executor admits it (gfx1201, uniform
   MQ4G256V2 dense layers); the ack reports `continuous_batch_exact: true`.
   `serve.batch_nonexact` / `HIPFIRE_SERVE_BATCH_NONEXACT=1` selects the
   non-exact shared slots body instead (`continuous_batch_nonexact: true`).
   `serve.batch_sampled` / `HIPFIRE_SERVE_BATCH_SAMPLED` (default on; `0` is
   the kill switch) additionally admits sampled (temperature>0) requests to the
   exact route, each in the lane mode its singleton route selects (AR, native
   MTP, FAST_SAMPLE selector-chain DFlash) and byte-identical to that
   singleton for the same wire seed. A sampled request the batch cannot serve
   in its own mode (stop strings or a think budget on a speculative lane, or
   a prompt + `max_tokens` beyond the lane engine's cap, 32,768 for the MTP
   head) stays on the singleton route, never an AR substitute;
   sampled speculative singletons are not promoted mid-generation. The load
   ack reports `continuous_batch_sampling: "sampled"` and
   `continuous_batch_sampled_modes`; each batched request's
   `continuous_batch` receipt carries `spec_mode`, `sampling` and
   `equivalence: "singleton_exact"`. With the flag off sampled requests stay
   on the serialized singleton route. Flash-Next (Qwen4) fn-lanes admit
   sampled requests with neutral penalties the same way: AR lanes draw on the
   host with the AR singleton's sampler and request seed; MTP lanes verify
   sampled windows naively (drafts stay the head's argmax, each verify row is
   one AR-sampler draw), so seeded output equals the AR singleton's wherever
   the logits match. Sampled Qwen4 requests are never promoted from a running
   singleton, and `false` keeps them on the singleton route.
   Greedy requests (and sampled ones under `serve.batch_sampled`) batch with
   penalties applied as on the singleton route; `serve.batch_spec`
   (`HIPFIRE_SERVE_BATCH_SPEC=0` disables it) lets lanes decode as
   cross-request MTP/DFlash speculative verify rows. A request that is alone at
   dispatch keeps the unchanged singleton
   route (including native MTP/DFlash); requests arriving while a singleton
   generation runs wait for it to finish, so their TTFT includes that
   generation (known gap). The fixed-slot `serve.multi_slot` engine below is
   the legacy route.

With `serve.multi_slot = true`, up to `serve.multi_slot_slots` requests
   decode at once on the experimental multi-slot engine. Concurrent requests
   can produce different greedy text than serial requests at ≥4 slots; output
   is deterministic for a fixed batch composition.

   The **embedding** route (arch 17) uses the same switch with a different
   engine behind it: `POST /v1/embeddings` requests arriving together are
   served as **one batched forward** rather than one at a time, and because the
   encoder is deterministic (no KV, GDN state reset per sequence) that batch is
   **bit-identical** to encoding each input alone — batching is a throughput
   choice there, never an output change. `serve.multi_slot_slots` is the number
   of requests one forward may hold and `serve.max_batch_tokens` its row
   budget; an isolated request pays only the collector's window
   (`HIPFIRE_EMBED_BATCH_WINDOW_US`, default 2 ms). Measured on gfx1101: eight
   concurrent 9-token queries go from 100 ms serial (80 req/s) to 19 ms
   (**416 req/s, 5.2x**) in a single 72-row forward. 2 slots matched serial in
   earlier testing; on 35B-A3B MoE checkpoints they diverged from serial on
   1–4 of 4 prompts.

Parked (swapped or cold) sessions are bounded even when clients abandon them:
a session idle for `serve.multi_slot_session_ttl_secs` (default one hour; `0`
disables the idle sweep) is closed by the next request or by the engine's idle
timer (it wakes at most every 30 s), and the non-resident table is capped at
`serve.multi_slot_max_sessions` (default auto `max(64, 16 × slots)`), oldest
first. Closing one costs a cold re-prefill from the next prompt; resident
sessions are never swept.

   The **embedding** route (arch 17) uses the same switch with a different
   engine behind it: `POST /v1/embeddings` requests arriving together are
   served as **one batched forward** rather than one at a time, and because the
   encoder is deterministic (no KV, GDN state reset per sequence) that batch is
   **bit-identical** to encoding each input alone — batching is a throughput
   choice there, never an output change (`scripts/check_embed_multislot.py`
   checks it bit-for-bit). `serve.multi_slot_slots` is the number of requests
   one forward may hold and `serve.max_batch_tokens` its row budget; an
   isolated request pays only the collector's window
   (`HIPFIRE_EMBED_BATCH_WINDOW_US`, default 2 ms). Measured on gfx1101, eight
   concurrent 9-token queries: 96 ms serial (83 req/s) against a 16 ms
   concurrent batch (501 req/s, **6.0x**) — all eight in one forward. A waiter that is still queued after `serve.queue_timeout_ms`
   (default 10 minutes, long enough for one full generation; `0` waits
   forever) gets
   **503** "server busy" with `Retry-After`. A failed `accept` (for example,
   out of file descriptors) is logged and retried, never fatal. At most 512
   connections are served at once; further connects wait in the listen
   backlog. A client must send a complete request head within 30 s of
   connecting or of its previous response, which also closes idle keep-alive
   connections.
5. **Idle eviction.** When `idle_timeout > 0`, an interval unloads the model
   after idle seconds **and** only when no generation is in flight and the
   serve lock is free. Next request reloads.
6. **Stop.** `hipfire stop` validates pid ownership before SIGTERM. Stale reused
   pids are **not** killed; the pidfile is removed instead. On SIGTERM or
   Ctrl-C, serve stops accepting and frees the port at once, lets requests in
   flight finish for up to 30 s, then exits; a second signal exits at once.

Detached readiness: parent polls `GET /health` for up to **60 seconds**.
`/health` means the native process is answering; inspect `model` and
`loading_model` to distinguish ready, loading, and unloaded state.

## Endpoints

Implemented paths (anything else → `404`):

| Method | Path | Role |
|---|---|---|
| `GET` | `/health` | Readiness JSON: `status`, `model`, `loading_model`, `pid`, `token`, `native`, `capabilities`, `n_ctx`, `reasoning_contract`, `reasoning_effort_native`, `reasoning_efforts`, `reasoning_effort_probe_error`, `swap`; `503` unless `status` is `ok` |
| `GET` | `/v1/models` | `{ object: "list", data: [{ id, object: "model", owned_by, created?, meta?, architecture?, load_summary? }, ...] }` from local model files; the resident model is listed first and alone carries `meta` / `architecture` / `load_summary`. `architecture.input_modalities` is the post-gate truth from the load ack: `text` always, `image` on a resident vision tower, `audio` on a resident audio tower, `video` on the embed route's video leg (arch 18) — an arch that refuses a modality by name never advertises it. `load_summary` (additive; absent when the daemon ack predates it) is the daemon's one-line load explanation, `{ line, spec, kv, context, placement, prefill }`: what speculation is on or off and why, the KV mode/backend, the admitted context and its bound, weight placement, and load-time prefill decisions (see [CLI.md](CLI.md#load-summary)). The same `line` is printed once on the daemon's stderr per load |
| `GET` | `/stats` | Serve telemetry: `model`, `uptime_sec`, `queue_depth`, `requests_served`, `retries_attempted`, `retries_succeeded`, `recent_tok_s`, `loaded_config`, `swap`. `loaded_config` is `{ model, kv_mode, kv_backend, max_seq }` — the values the resident model was actually loaded with, from the daemon's load ack (a field is `null` when the ack did not carry it); the whole object is `null` while no model is resident |
| `GET` | `/metrics` | Prometheus text exposition (v0.0.4); see below |
| `POST` | `/v1/chat/completions` | Chat completions (stream or non-stream) |
| `POST` | `/v1/messages` | Anthropic Messages API (stream or non-stream), adapted onto chat completions |
| `POST` | `/v1/responses` | OpenAI Responses API (stream or non-stream), adapted onto chat completions |
| `POST` | `/v1/images/generations` | OpenAI-shaped image generation on the resident FLUX checkpoint; see below |
| `POST` | `/v1/images/edits` | Reference-image edit (multipart, FLUX.2 Klein); see below |
| `GET` | `/`, `/ui`, `/ui/`, `/ui/index.html`, `/ui/app.js`, `/ui/db.js`, `/ui/render.js`, `/ui/style.css` | Embedded chat UI (opt-in; see below). Exactly these paths: `/` redirects (302) to `/ui`; other `/ui/...` paths are `404` |

There is **no** `/v1/completions` route in the current CLI serve.

### Embedded chat UI (`--ui` / `serve.ui` / `HIPFIRE_SERVE_UI`)

Off by default. `hipfire serve --ui` (or `--open`, which implies `--ui`
and launches the browser) serves a dependency-free chat frontend at
`/ui` on the same listener; `serve.ui = true` or `HIPFIRE_SERVE_UI=1`
enable it for any serve start, including detached ones. `GET /`
redirects to `/ui` only while the UI is enabled.

The UI talks to the same `/v1/chat/completions` gateway as any other
client and keeps chat history in the browser's IndexedDB — serve stores
nothing on its behalf. It cannot send a bearer token: without
`serve.api_key_file` it works as before (on loopback or any other bind), and on
a keyed serve its assets and requests receive 401
(see [Security](#security-bearer-authentication-and-https-proxies)).

For the sealed MQ4R reproduction, default-model handoff, and copyable Hermes
Agent / Pi custom-provider configuration, see
[`GOLDEN-REDLINE.md`](GOLDEN-REDLINE.md).

### `GET /health`

`200` with `status: "ok"` while the daemon is up. `503` with
`status: "restarting"` while serve respawns a daemon that exited (crash, panic,
or a sticky GPU fault, after which the daemon exits 75) and reloads the
resident model; `503` with `status: "unhealthy"` if respawning failed, or if
multi-slot serve could not load its model (pre-warm, or the reload after a
respawn), after which serve exits 1 so a service manager can restart it.
`model` is the loaded tag/path or `null` when idle/unloaded or after a failed
model switch; `loading_model` names an asynchronous pre-warm or post-restart
reload. `token` is a per-process instance token. `n_ctx` is the `max_seq` the
resident model was loaded with (`null` while none is resident).
`reasoning_contract`, `reasoning_effort_native` and `reasoning_efforts`
publish the resident model's reasoning capability (see the reasoning contract
below); `reasoning_effort_probe_error` is the full render error when the
daemon's baseline effort probe failed, `null` otherwise. `swap` publishes the
multi-slot swap tier: `tier` (`host+disk`/`host`/`disk`/`none`), `evictions`,
`restores`, `failures`, `host_budget`, `host_used`, `disk_budget`, `disk_used`.
It is configured from the resolved process policy at startup, replaced with the
daemon's effective budgets on the first `done` (a read-only swap directory can
disable the disk tier), and refreshed from each subsequent `done`; it is
`null` on the standard route. An idle sweep between requests updates the
daemon's counters immediately but reaches `/health`/`/metrics` on the next
`done`. The same snapshot is exposed as Prometheus gauges on `/metrics`
(`hipfire_swap_info{tier=...}`, `hipfire_swap_evictions`,
`hipfire_swap_restores`, `hipfire_swap_failures`, and the four `*_bytes`
gauges). All of these come from serve's own metadata, so `/health` never waits
on a model load.

Model switch lifecycle: a load request is first admitted read-only; a refused
admission leaves the prior model (and its drafter) loaded. Once admitted, a
single-device or pipeline-parallel load (Flash-Next/Qwen4 included) unloads
the prior model and requires clean VMM state before constructing the new
one, so a failure after admission leaves no model loaded and the error says
`no model loaded`. Only tp>1 expert-parallel loads stage the new model before
retiring the prior one. Flash-Next with VMM QSA state (gfx1151/gfx1201 automatic) owns
VMM arenas of its own: the target and MTP context arenas and the gathered
attention workspace. All of them are released when the model unloads, so the
next load's clean-VMM check starts from zero owners in either swap direction.

```bash
# Loopback example (safe default for local smoke):
curl -s http://127.0.0.1:11435/health
```

### `GET /metrics`

Prometheus text format, read from gateway counters and `ServeMeta`, so a
scrape never waits on a model load. Token counts, throughput and tau come
from the daemon's `done` event and are recorded only when that event carries
the field: a route that does not report one adds nothing, never a zero.
TTFT, latency and TPOT are measured at the gateway.

| Series | Type | Meaning |
|---|---|---|
| `hipfire_requests_total` | counter | Completed chat and image requests |
| `hipfire_requests_failed_total` | counter | Requests answered with an error, or whose stream ended in one |
| `hipfire_admission_rejected_total` | counter | Requests refused by admission (queue full or wait timed out) |
| `hipfire_prompt_tokens_total` | counter | `done.prompt_tokens` (`prefill_tokens + cached_tokens` if absent, as in `usage`) |
| `hipfire_completion_tokens_total` | counter | `done.tokens` (`done.completion_tokens` if absent) |
| `hipfire_cached_prompt_tokens_total` | counter | `done.cached_tokens` |
| `hipfire_retries_total`, `hipfire_retries_succeeded_total` | counter | Transient-failure retries attempted / that then completed |
| `hipfire_queue_depth`, `hipfire_queue_capacity` | gauge | Requests in flight or queued; admission capacity |
| `hipfire_uptime_seconds` | gauge | Seconds since serve started |
| `hipfire_engine_up` | gauge | 1 while the daemon is up, 0 while restarting or unhealthy |
| `hipfire_model_loading` | gauge | 1 while a model load is in progress |
| `hipfire_context_capacity_tokens` | gauge | Context the resident model was loaded with (0 when none) |
| `hipfire_model_info{model="…"}` | gauge | 1 for the resident model; no sample when none is resident |
| `hipfire_ttft_milliseconds` | histogram | Time to first generated token (reasoning included), chat requests |
| `hipfire_request_latency_milliseconds` | histogram | Wall time, chat and image requests |
| `hipfire_time_per_output_token_milliseconds` | histogram | `(latency − ttft) / (tokens − 1)`; requests with at least 2 tokens |
| `hipfire_decode_tokens_per_second`, `hipfire_prefill_tokens_per_second` | histogram | `done.decode_tok_s` (`done.tok_s` if absent), `done.prefill_tok_s` |
| `hipfire_spec_tau` | summary (`_sum`/`_count`) | `done.tau`, speculative routes only |

Buckets: TTFT and latency `10 … 60000` ms; TPOT `2 … 500` ms; throughput
`1 … 5000` tok/s.

Multi-slot serve (`serve.multi_slot`) attaches the engine's counters to every
`done` event as `engine_stats`, and serve exports the latest snapshot. The
values change when a request finishes, not during a long generation. Free
pages are sampled at the start of that request's last engine step; active
slots exclude the finishing request, so an idle engine reads 0. The standard
route sends no snapshot, so these series are absent there (never 0); a model
switch clears them until the new model finishes a request.

| Series | Type | Meaning |
|---|---|---|
| `hipfire_kv_pages_free`, `hipfire_kv_pages_total` | gauge | Free and total pages in the KV page pool (absent when the engine is not paged) |
| `hipfire_slots_active`, `hipfire_slots_total` | gauge | Slots holding a request; engine slots |
| `hipfire_evictions_total`, `hipfire_restores_total` | counter | Idle sessions parked to host swap; parked sessions restored to a slot |
| `hipfire_prefix_hits_total` | counter | Continuations served from a session's resident KV |
| `hipfire_prefix_reused_tokens_total` | counter | Prompt tokens reused from the cross-session prefix cache |

The engine counters start at 0 when the model loads.

### `POST /v1/chat/completions`

```bash
curl -N http://127.0.0.1:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5:9b",
    "messages": [{"role": "user", "content": "hi"}],
    "max_tokens": 512,
    "chat_template_kwargs": {"enable_thinking": false},
    "stream": true
  }'
```

> **`max_tokens` includes reasoning.** Qwen tags think by default
> (`thinking = "on"`), and the `<think>` span counts against `max_tokens`. With
> a modest cap the whole budget can go to reasoning, so the response carries
> only `reasoning_content`, `content` is `""`, and `finish_reason` is `length`.
> Most OpenAI clients show only `content`, so this looks like a blank answer.
> Send `"chat_template_kwargs": {"enable_thinking": false}` (as above) for
> direct answers, or raise `max_tokens` / omit it to keep thinking.

- **`stream: true`** (typical clients): SSE `data: {chat.completion.chunk}`
  lines until `data: [DONE]`. The `200` and the role chunk go out when
  generation sends its first frame, or after 15 s of silence (cold load, long
  prefill), so a request that fails before then gets the same JSON error and
  status as a non-stream request. A stream that fails after that ends with
  `data: {"error": {"message", "type"}}` and then `data: [DONE]`. Until
  `[DONE]`, 15 s without a frame sends a `: keepalive` SSE comment, which
  SSE parsers ignore. While the engine prefills a long prompt, the committed
  stream also carries a `: prefill <done>/<total>` SSE comment (prompt tokens
  processed so far; for example `: prefill 512/2048`) after each prefill chunk,
  before the first token. It is sent only after the `200` is on the wire (it
  never commits a stream early, so a pre-commit failure is still a JSON error;
  progress reported before the commit is dropped), counts as activity for the
  keepalive timer, never follows the final frame, and is ignored by SSE
  parsers. `hipfire_client::read_openai_sse` surfaces it as
  `OpenAiSseEvent::PrefillProgress`, and `hipfire run` and the TUI show it.
  `/v1/messages` and `/v1/responses` streams carry the same comment;
  non-streaming requests have none.
- **`stream: false` / omitted falsey:** single `chat.completion` JSON body.
- Oversized body → **413** before the daemon lock (Content-Length or streamed
  cap at `max_request_bytes`).
- Saturated admission queue → **503** with `Retry-After`.
- Invalid JSON body → **400**.
- Daemon errors map on their typed class: `validation`, `context_length`,
  `unsupported` → **400**; `transient` → **503** with `Retry-After: 1`; any
  other class → **500**. An unknown model → **404**.

Request fields honored by the serve layer (non-exhaustive; sampling falls
through to per-model / registry / daemon defaults when omitted):

| Field | Notes |
|---|---|
| `model` | Tag or path of an installed model; reloads if different from the resident model. A registry model that is not installed is 404 ("run `hipfire pull`") unless `serve.allow_request_pull = true`; a file outside the models directory, the catalog and the pre-warm model is 404 unless `serve.allow_request_paths = true`. |
| `max_tokens` / `max_completion_tokens` | Omitted: the configured default (Qwen tags 81920, DeepSeek V4 Flash 393216, else `generation.max_tokens`) is fitted to the context left after the prompt, minus 64. Explicit and at or above the loaded context: 400. Never triggers a reload. |
| `messages` | OpenAI chat messages (required for useful chat) |
| `messages[].content[].image_url` | One base64 PNG/JPEG data URI for VL models; remote URLs and multiple images are rejected |
| `stream`, `stream_options.include_usage` | Streaming + optional usage on stream end |
| `temperature`, `top_p`, `top_k`, `min_p`, `repeat_penalty` | Sampling; explicit request values win, otherwise per-model TOML / registry-card values are applied. On the Qwen AR samplers (GPU kernel and the CPU sampler used by Flash-Next AR, grammar-constrained Qwen turns and the multi-slot VL first token) `top_k` keeps at most k candidates from a 64-wide gather (absent = 20; `0` or > 64 = 64) and `min_p` drops candidates below `min_p` × the top candidate's probability, both before `top_p`. At `temperature: 0` all three are no-ops. Other routes do not apply every field yet: Gemma4 eager, LFM2, Cohere2, MiniMax, DeepSeek V4 AR, LLaMA AR and single-slot Qwen3.5-VL image turns ignore the request `top_k`; those plus Muse Glimmer, DeepSeek V4 speculative and n-gram speculative decoding ignore `min_p`; Qwen2, Maple and dots.ocr text decode greedily. |
| `seed` | OpenAI-compatible sampling seed: non-negative integer (≤ u64::MAX). Same seed is reproducible only when prompt, sampling parameters, and execution shape match. Cold vs prefix-resumed and solo vs co-batched shapes may differ until `serve.batch_invariant` is implemented; `null`/omitted = fresh entropy. Negative/fractional/non-integer values are rejected. |
| `presence_penalty`, `frequency_penalty` | Forwarded natively to the daemon (≥ 0); `presence_penalty` also inherits per-model / registry defaults |
| `max_tokens` | Generation cap, shared by the `<think>` span and the answer (see the note under the example above) |
| `stop` | A string or an array of up to 4 strings, each ≤ 64 characters; any other value is a 400. Matched on the answer only (never inside reasoning), across token boundaries; streaming holds back a possible partial match, the stop text and everything after it are not returned, and `finish_reason` is `stop` (or `tool_calls` when a call completed before the match). Honoured on the Qwen3.5-family routes (`qwen_ar`, `qwen_dflash`, which covers MTP), Flash-Next (`qwen4_ar`, `qwen4_spec` = native MTP) and Raon-1-mini (`raon_ar`, `raon_spec` = DFlash2); every other route answers 400 instead of ignoring it |
| `tools` | Tool definitions (with structured `messages` when Jinja chat is on). Flash-Next accepts them with native MTP attached: greedy tool turns run on MTP and match AR byte for byte |
| `response_format` | **Not enforced on the standard route.** Any value (`json_object`, `json_schema`, `text`, even an unknown type) is accepted unvalidated with HTTP 200 and is not forwarded to the daemon; the model output is unconstrained (a model asked for JSON may still wrap it in a code fence). Only `serve.multi_slot = true` enforces `json_schema` (pre-sampling grammar mask) and refuses `json_object` / unknown types with 400 |
| `chat_template_kwargs.enable_thinking` | Mode axis. `false` forces a no-think turn (Qwen native empty closed think). Independent of effort and cap. |
| `chat_template_kwargs.preserve_thinking` | Keep `<think>` in final non-stream content |
| `thinking` / `thinking.type` | Mode axis (`on`/`off`, or DeepSeek-style `enabled`/`disabled`). Thinking disabled wins and drops effort/cap with a warning. |
| `reasoning.effort` / `reasoning_effort` | **Semantic effort only** — prompt strength. Never mapped to a think-token budget. Family ladders differ; unsupported/cross-family values are dropped with a warning (see below). |
| `reasoning.max_tokens` / `max_think_tokens` | Explicit **integer** think-span cap on Qwen Jinja contracts. `0` or omitted = uncapped. DeepSeek, Gemma, Glimmer, and unsupported contracts drop it with a warning. Independent of effort. |
| `thinking_budget` / `reasoning.budget` | Legacy **named** cap preset only on non-effort-native Qwen templates that still accept it. Dropped+warned elsewhere. |

### `POST /v1/messages` (Anthropic Messages API)

An adapter onto the chat-completions route, not a second generation path.
The request becomes a chat-completions body, which then goes through the
same admission (batching eligibility and model lease), tool gate, worker,
tool-call parser, think routing and stop handling as `/v1/chat/completions`.
Serve sends the daemon the same `generate` request as the matching chat
request, so batching and the session cache behave the same. The response is
rendered back as an Anthropic `message` object or SSE events.

```bash
curl -N http://127.0.0.1:11435/v1/messages \
  -H "Content-Type: application/json" -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "qwen3.5:4b",
    "max_tokens": 512,
    "system": "Be brief.",
    "messages": [{"role": "user", "content": "hi"}],
    "stream": true
  }'
```

| Messages field | Chat-completions equivalent |
|---|---|
| `model`, `max_tokens` (required) | `model`, `max_tokens` |
| `system` (string or text blocks) | A leading `system` message. Text blocks are concatenated with no separator, as chat content parts are |
| `messages[]` `text` blocks | Plain-string `content`. Text-only turns keep the plain-string shape the batch routes require |
| `image` block (base64; `url` source) | One `image_url` part (only base64 PNG/JPEG is accepted, as on chat) |
| assistant `thinking` blocks | `reasoning_content` (replayed as for chat; `signature` ignored). `redacted_thinking` is dropped |
| assistant `tool_use` blocks | `tool_calls` (input serialized as the arguments string) |
| user `tool_result` blocks | `tool` messages with `tool_call_id = tool_use_id`, placed before the turn's text. `is_error: true` prefixes the result with `Error: ` |
| `tools` (`name`, `description`, `input_schema`) | Function tools. Server tools (`type` other than `custom`) → 400 |
| `tool_choice` `auto` / `any` / `tool` / `none` | `auto` / `required` / named function / `none`. `disable_parallel_tool_use` is not enforced, like chat's `parallel_tool_calls` |
| `thinking` `enabled` (+ `budget_tokens`) / `adaptive` / `disabled` | `thinking.type` enabled/disabled; `budget_tokens` → `max_think_tokens` |
| `thinking` absent | Thinking **off** (the Messages API default). Send `thinking` to enable it |
| `stop_sequences` | `stop` (same limits: up to 4, each ≤ 64 characters; an empty list is not forwarded) |
| `temperature`, `top_p`, `top_k` | Same fields |
| `stream` | SSE (below) |
| `metadata`, `service_tier`, `cache_control` | Ignored |

Refused with 400 rather than silently changed: a final assistant message
(response prefill), `document` and other unsupported block types, images
inside `tool_result`, server tools and `mcp_servers`.

Response: `content` holds a `thinking` block (reasoning, `signature: ""`),
then a `text` block, then the released `tool_use` blocks. Each appears only
when it has content. Tool ids are the daemon's or `toolu_<request>_<n>`.
`stop_reason`: `end_turn` (stop), `max_tokens` (length), `tool_use` (released
tool calls). The daemon's terminal does not say which stop sequence matched,
so a stop-sequence stop also reports `end_turn` with `stop_sequence: null`.
`usage`: `input_tokens` is the prompt minus session-cache hits,
`cache_read_input_tokens` is the cache hits, `cache_creation_input_tokens` is
always 0, and `output_tokens` is the chat `completion_tokens`. `timings` and
`hipfire` carry the same evidence as on chat.

Streaming: `message_start`, then one `content_block_start` /
`content_block_delta` (`thinking_delta` or `text_delta`) / `content_block_stop`
run per thinking or text span, as generated. The released `tool_use` blocks
follow at the end, each with one `input_json_delta` holding the full arguments.
Then `message_delta` (`stop_reason`, `usage`) and `message_stop`. The
`message_start` usage is zero, because token counts are known only at the
terminal, and `message_delta` carries the real counts. Commit timing,
keepalives, `: prefill` progress comments and the pre-commit JSON errors work
as on chat. A failure after commit ends the stream with `event: error`.

Errors use the Anthropic shape `{"type": "error", "error": {"type", "message"}}`
with chat's status mapping: 400 `invalid_request_error`, 404
`not_found_error`, 413 `request_too_large`, 503 `overloaded_error` (with
`Retry-After`), 500 `api_error`. `POST /v1/messages/count_tokens` is not
implemented (404).

### `POST /v1/responses` (OpenAI Responses API)

An adapter onto the chat-completions route, like `/v1/messages`: same
admission, batching eligibility, session cache, tool parsing, think routing
and stop handling. Serve stores no responses. The client resends the
conversation in `input` (`store` is always reported `false`).

```bash
curl -N http://127.0.0.1:11435/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5:4b",
    "instructions": "Be brief.",
    "input": "hi",
    "max_output_tokens": 512,
    "stream": true
  }'
```

| Responses field | Chat-completions equivalent |
|---|---|
| `instructions` | A leading `system` message |
| `input` string | One `user` message |
| `input` message items (`user` / `system` / `developer` / `assistant`; `input_text` / `output_text` parts; `input_image` with `image_url`) | Chat messages with plain-string content (an image keeps the array form, as on chat) |
| `reasoning` items | `reasoning_content` of the following assistant turn (`content` text, else the `summary` text) |
| `function_call` items | `tool_calls` of the current assistant turn (`call_id` → id) |
| `function_call_output` items | `tool` messages (`call_id` → `tool_call_id`) |
| `tools` (`type: "function"`) | Function tools. Hosted tools → 400 |
| `tool_choice` `auto` / `none` / `required` / `{type: function, name}` | Same |
| `max_output_tokens` | `max_tokens` (omitted: fitted to the context as on chat) |
| `temperature`, `top_p`; extensions `top_k`, `stop` | Same fields |
| `reasoning.effort` | `reasoning_effort` (absent: the server's reasoning default, as on chat) |
| `text.format` `json_schema` / `json_object` | `response_format` (enforced only where chat enforces it) |
| `include`, `store`, `metadata`, `user`, `parallel_tool_calls`, `truncation` | Accepted; not enforced (`parallel_tool_calls` is ignored, as on chat) |

Refused with 400: `previous_response_id`, `conversation`, `prompt`,
`item_reference` items, `background: true`, hosted tools and `input_file`.

Response: `output` holds a `reasoning` item (`content[0].reasoning_text`,
empty `summary`), then a `message` item (`output_text`), then the released
`function_call` items. Each appears only when it has content. Call ids are the
daemon's or `call_<request>_<n>`. `status` is `completed`, or `incomplete`
with `incomplete_details.reason: "max_output_tokens"` on length. `usage` has
`input_tokens` (session-cache hits included and reported in
`input_tokens_details.cached_tokens`), `output_tokens` and `total_tokens`.
Reasoning tokens are not counted separately, so `output_tokens_details` is
omitted.

Streaming: `response.created`, `response.in_progress`, then per reasoning
span `response.output_item.added` / `response.reasoning_text.delta` /
`response.reasoning_text.done` / `response.output_item.done`, and per text
span `output_item.added` / `content_part.added` / `output_text.delta` /
`output_text.done` / `content_part.done` / `output_item.done`. Released
function calls follow (`output_item.added`,
`function_call_arguments.delta` with the full arguments,
`function_call_arguments.done`, `output_item.done`), then
`response.completed` (or `response.incomplete`) with the full response. Every
event carries `sequence_number`. A failure after commit sends `error`, then
`response.failed`. Errors before commit are OpenAI-shaped `{"error": {...}}`
with chat's status mapping.

### `POST /v1/images/generations`

OpenAI-shaped image generation on the **resident** diffusion checkpoint (arch 40
FLUX.1, arch 45 FLUX.2 Klein). Serve does not switch models for this route:
load the checkpoint first, and `model` is only echoed back in the response
(it is not a selector). Body fields: `prompt` (required), optional `width` /
`height` (or OpenAI `size: "WxH"`, which wins), `steps` (architecture default
when omitted: 4 for `flux.schnell`, 28 for `flux.dev`), `seed`, `n` (must be
1), `response_format` (must be `b64_json`), and the daemon-validated
`sampler` (euler flow-match only), `negative_prompt` and `backend`
(`cpu` or `gpu`). Same denoise path as the `img_generate` daemon message
(`hipfire img`). Image requests share the admission queue with chat.

```json
{"model": "flux.schnell:1", "prompt": "a tiny lighthouse on a rock, sunset", "size": "512x512", "response_format": "b64_json"}
```

Response: `{ "created": 1790000000, "model": "<loaded model path>", "data": [ { "b64_json": "…", "size": "512x512" } ], "hipfire": { "seed": 0, "steps": 4, "width": 512, "height": 512, "ms": 1234 } }`.

This body takes no image input. A request with an `images` field is refused
with a 400 that points at `/v1/images/edits`.

### `POST /v1/images/edits`

OpenAI-shaped reference edit (FLUX.2 Klein, arch 45 only): `multipart/form-data`
with one to four `image` file parts (repeat the part; `image[]` is also
accepted; PNG or JPEG, at most 32 MiB each) plus the text fields of
`/v1/images/generations` (`prompt` required; `size`, `width`, `height`,
`steps`, `seed`, `n`, `response_format`). Numeric fields must be non-negative
integers. The image bytes travel in the request; the server never reads a file
the client names. Each reference is area-capped at 1 MP and floored to a
multiple of 16, and conditions every denoise step without being denoised.
Without `size` the output takes the first reference's size. The whole body
counts against `serve.max_request_bytes`.

```bash
curl -s -X POST http://127.0.0.1:11435/v1/images/edits \
  -F image=@bike.png -F prompt="make the bicycle blue" -F steps=4 -F seed=0 \
  | jq -r '.data[0].b64_json' | base64 -d > bike-blue.png
```

Response: the same shape as `/v1/images/generations`.

### Reasoning request contract

Mode, effort, and cap are three independent axes — full key table and budget
map in [`CONFIG.md`](CONFIG.md). HTTP accepts the same meanings as CLI/config.

**Normalization (warn + drop, not silent reinterpretation):**

- Recognizable unsupported or cross-family values are dropped or aliased with
  `[WARN: INVALID CONFIG]`. Warnings appear in the server log and, on OpenAI
  responses, under `hipfire.reasoning` / `hipfire.config_warnings` metadata.
- Malformed types and out-of-range integers remain hard request errors.
- Effort is **never** converted into a token cap (including on Qwen3.6 and other
  non-effort-native templates).
- Qwen3.8 and Ornith 1.5 default to an uncapped think span unless the request sets a
  positive integer cap (semantic effort stays independent of that cap). DeepSeek V4 and
  Muse Glimmer expose semantic effort but no independent parent-defined cap; cap fields
  are dropped+warned.
- A request that omits effort (or sends `auto`) on Qwen3.8 or Ornith 1.5 defers to
  configured effort, and if none is set resolves to hipfire's `medium` default (the
  official template default is `xhigh`). An explicit non-`auto` `reasoning_effort` /
  `reasoning.effort` in the request wins, so `"xhigh"` restores the old behavior.
  Thinking off drops the effort. This default does not imply a cap.

**Examples:**

```bash
# Qwen3.8 — disable thinking natively (empty closed think block)
curl -s http://127.0.0.1:11435/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.8:27b",
  "messages": [{"role": "user", "content": "hi"}],
  "chat_template_kwargs": {"enable_thinking": false}
}'

# Qwen3.8 — semantic effort only; omitted/auto resolves to medium, xhigh is explicit (still uncapped unless max_think_tokens is set)
curl -s http://127.0.0.1:11435/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.8:27b",
  "messages": [{"role": "user", "content": "plan a refactor"}],
  "reasoning_effort": "xhigh"
}'

# Ornith 1.5 — Qwen3.8-compatible semantic effort (omitted/auto resolves to medium; still uncapped unless max_think_tokens)
curl -s http://127.0.0.1:11435/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "ornith-1.5:35b-a3b",
  "messages": [{"role": "user", "content": "plan a refactor"}],
  "reasoning_effort": "low"
}'

# Qwen3.6 — no native effort: effort is dropped+warned; set an explicit cap if wanted
curl -s http://127.0.0.1:11435/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.6:27b",
  "messages": [{"role": "user", "content": "hi"}],
  "reasoning_effort": "low",
  "max_think_tokens": 2048
}'

# DeepSeek V4 — mode + effort; medium/xhigh alias to high; no cap from effort
curl -s http://127.0.0.1:11435/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "deepseek-v4-flash",
  "messages": [{"role": "user", "content": "hi"}],
  "thinking": {"type": "enabled"},
  "reasoning_effort": "high"
}'

# Gemma4 — boolean thinking (official default off); unsupported effort/cap dropped
curl -s http://127.0.0.1:11435/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "gemma4:31b",
  "messages": [{"role": "user", "content": "hi"}],
  "chat_template_kwargs": {"enable_thinking": true}
}'

# Muse Glimmer — always reasons via Onyx; strength dial; off is dropped+warned
curl -s http://127.0.0.1:11435/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "muse-glimmer",
  "messages": [{"role": "user", "content": "hi"}],
  "reasoning_effort": "high"
}'
```

| Family | Mode | Effort | Cap |
|---|---|---|---|
| Qwen3.8 | `enable_thinking=false` → empty closed think | `low` \| `medium` \| `xhigh` (default `xhigh`; semantic prompt only) | Integer only; named budget dropped |
| Ornith 1.5 | same Qwen on/off | Qwen3.8-compatible `low` \| `medium` \| `xhigh` (default `xhigh`; not a budget) | Integer only; named budget dropped |
| Qwen3.6 | on/off | Dropped+warned (never → cap) | Named preset or integer if set |
| DeepSeek V4 | `thinking.type` enabled/disabled (default on) | `low` \| `high` \| `max` (`medium`/`xhigh`→`high`) | Integer only; default uncapped |
| Gemma4 | Boolean; default **off** | Dropped+warned | Dropped+warned unless explicit engine integer |
| Muse Glimmer | Always on (off dropped) | `low` \| `medium` \| `high` \| `xhigh` | Integer only; default uncapped |

When `messages` contains no `system` or `developer` role, the serve layer inserts `prompt.system` from a per-model TOML override or the registry card's `recommended_settings.system_prompt`. A client-supplied system/developer message always wins.

`finish_reason` values emitted to clients: `stop`, `length`, `tool_calls`. A
turn that runs out of `max_tokens` while still reasoning ends with `length`:
the partial `reasoning_content` is returned and `content` is empty. A turn
the model ends (end-of-turn token) while still reasoning ends the same way
with `stop`. Neither turn releases a tool call or is cached.

Streaming error contract: a failure that happens BEFORE the first byte is a
plain HTTP error status (no SSE body). A failure after the stream has started
emits an OpenAI-shaped SSE `data: {"error": {...}}` frame followed by
`data: [DONE]`, so a streaming client always sees a terminal event instead of
a silently truncated `200`. Non-streaming failures keep the typed
status mapping (`400` request/config, `503` + `Retry-After` overload or
transient, `500` internal).

Loads whose ack reports `cache_capable: true`, or whose route is continuous-batch
capable, skip the per-request `reset` so multi-turn longest-common-prefix reuse
can hit. Other loads reset every request (stateless OpenAI shape).

### Client notes (Zed / OpenAI-compatible agents)

Live-GPU endpoint validation on gfx1100 and gfx1201 exercised Zed-shaped
streamed requests against the canonical Qwen3.6 35B-A3B MQ4R artifact. The
matrix covered automatic and required tool calls, multiple calls in one turn,
multi-turn `reasoning_content` replay, `tool_choice: "none"`, usage chunks, and
strict UTF-8 SSE/JSON decoding. This validates the OpenAI-compatible endpoint,
not the Zed application UI itself.

Inbound assistant `tool_calls[].function.arguments` strings are parsed into
JSON objects for Qwen template replay; outbound OpenAI arguments remain JSON
strings. Tool-call deltas are emitted in bulk at the terminal chunk,
`finish_reason=tool_calls` is produced, and disconnect cancels in-flight
generation.

Current caveats for agent clients:

- **Multi-turn reasoning replay:** Qwen3.5/3.6-family arches replay assistant
  `reasoning_content` into the next turn (alongside `muse_glimmer`).
- **Tool-argument streaming:** progressive per-token argument deltas are not
  emitted; arguments arrive bulk-at-terminal with the tool-call chunk.
- **`parallel_tool_calls`:** the explicit request field is ignored.
- **Zed temperature:** Zed defaults sampling temperature to **1.0** unless the
  agent profile overrides it.

For Qwen3.5/3.6 agent use, prefer temperature **0.2–0.6** in the profile, and
enable interleaved reasoning on Zed’s OpenAI-compatible capability when the
client exposes that toggle.

## Auto-routing from `hipfire run`

```bash
hipfire serve -d
hipfire run qwen3.5:9b "..."                 # uses HTTP when /health is up
HIPFIRE_LOCAL=1 hipfire run qwen3.5:9b "..." # force one-shot local daemon
```

`run` probes `http://<probe-host>:<port>/health` (500 ms). Probe host maps
`0.0.0.0` / `::` → `127.0.0.1`. If serve is up, `run` POSTs
`/v1/chat/completions` and does **not** spawn a second daemon. If serve is up
but the request fails, `run` exits rather than colliding on the GPU lock.
`--json` / `--no-stream` also force local control.

Model mismatch: serve reloads to the requested model on the chat path (cold
start cost on that first switched request).

## Experimental prefix cache (multi-slot)

`serve.prefix_cache` is **off by default**. On the multi-slot engine
(`serve.multi_slot=true`) it enables cross-session radix reuse of sealed
128-token KV pages plus Qwen hybrid checkpoints. `.mtp` sidecars are
probed as both `foo.mq4v2.mtp` and `foo.mtp` on the slot engine and the
single-slot loader (`hipfire run --spec mtp`). Verified on gfx1101 /
ROCm 10 (`test_serve_prefix_cache --mtp-k 4`): greedy MTP, sampled AR,
and JSON-Schema AR all reuse ≥256 tokens; `reset` forces a cold miss.
Vision+prefix reuse stays off. Non-empty stop sequences, logprobs, and the images+tools combination stay refused on slots; tools alone are supported.
This is not a registry admission.

### Capability advertisement

`/health` carries a `capabilities` object built once at startup from the
SAME resolved config the daemon's slot engine reads, so what is advertised
is what the engine was built with. OpenAI-compatible clients ignore the
extra field; hipfire clients (and deployment tooling) use it for discovery:

```json
{
  "status": "ok",
  "capabilities": {
    "openai_compatible": true,
    "mode": "multi-slot",
    "multi_slot": true,
    "multi_slot_slots": 2,
    "prefix_cache": true,
    "structured_output": true,
    "structured_output_subset": "json-schema-strict-v1",
    "refused_request_fields": ["stop", "logprobs", "response_format:json_object"]
  }
}
```

The standard route advertises the honest absence (`"multi_slot": false`,
`"structured_output": false`) rather than aspirational capabilities.

```bash
HIPFIRE_SERVE_MULTI_SLOT=true HIPFIRE_SERVE_PREFIX_CACHE=true \
  HIPFIRE_SERVE_PREFIX_CACHE_MAX_BYTES=268435456 \
  hipfire serve 127.0.0.1:11435 <model.hfq>
```

## Production smoke (GPU)

User-facing serve behavior is a **manual GPU** route — not proven by
container builds or no-GPU CI.

| Claim | Route | Owner |
|---|---|---|
| Generic serve semantics (finish reasons, empty/runaway, cache, timing hooks) | `scripts/serve_harness.py --model <path>` | [VALIDATION.md](VALIDATION.md) |
| Optional wrapper (serve + optional Redline/perf arms) | `scripts/gates.sh --model <path>` | same |

### Branch-only: LFM framing harness

**Branch-implemented** at audited `lfm-redline@692a726dde53508cb53de1a74c720e75a7c9f33e`; **absent** from comparison base `origin/beta@202282de8759dfa6963ea5184ad2bf2b9259cef6`. Not a shipped/ref-pinned HTTP contract on this page.

| Claim | Route | Owner |
|---|---|---|
| LFM2.5 thinking / framing smoke | `scripts/serve_harness.py` with an exact `lfm2.5:*` tag | [VALIDATION.md](VALIDATION.md) |

Example (harness spawns its own serve by default on port `11520`):

```bash
python3 scripts/serve_harness.py --model ~/.hipfire/models/<file> --tag qwen3.5:9b --mode battery
```

No-GPU CI (`.github/workflows/no-gpu-ci.yml` → `scripts/no-gpu-ci.sh`) never
certifies GPU serve, model coherence, or throughput.

## Flash-Next exact lane batching (qwen4, arch 16)

Concurrent Qwen3.8 Flash-Next requests can share one trunk step on the exact
**fn-lanes** route: each admitted request owns a lane (its own request state:
QSA K/V, GDN recurrent state, PLE history) over the one resident weight set,
and every committed token and state byte per request is identical to that
request run alone on the singleton route. **Default off, experimental.**

- **Enable:** `serve.vmm_batch = true` (`HIPFIRE_SERVE_VMM_BATCH=1`) **and**
  `serve.continuous_batch_size >= 2` (`HIPFIRE_CONTINUOUS_BATCH_SIZE`); the
  row budget per step is `serve.max_batch_tokens`. The QSA context backend must
  be VMM and the lane store must fit the free device memory; otherwise the
  load logs `[daemon] qwen4 lanes refused: <reason> — fallback to sequential`
  and serves the singleton route exactly as before. With
  `continuous_batch_size >= 2` but `serve.vmm_batch` off the load logs
  `[daemon] qwen4 lanes need serve.vmm_batch — fallback to sequential` and
  advertises `continuous_batch_capable: false`.
- **Kill switch:** `HIPFIRE_SERVE_VMM_BATCH=0` (the compatibility spelling of
  `serve.vmm_batch = false`) stages no lane store and restores today's serial
  behavior.
- **Eligible requests:** greedy decoding, plain text chat only — no tools,
  images, logprobs or penalties. Ineligible requests keep the singleton route.
- **Singleton when alone:** a request that is alone at dispatch runs the
  unchanged singleton route (including native MTP). When a batched peer
  queues behind it, a greedy singleton moves into a lane at its next
  committed window boundary (promotion): into an MTP lane with its trunk,
  head and draft states, or, with `serve.batch_spec = false`, into an AR lane
  with its trunk state, continuing the AR singleton's text byte for byte.
- **MTP lanes (default when the MTP head is loaded):** each lane runs its own
  native MTP window per step: a step-synchronous batched draft for every
  drafting lane, then one `(d+1)`-row verify per lane inside the shared lane
  step (the step's row budget counts lanes, not rows). A lane that
  drafts nothing, or that the floor retires, decodes one AR row per step.
  `serve.batch_spec = false` (`HIPFIRE_SERVE_BATCH_SPEC=0`) keeps the AR lanes
  of before (one token per lane per step). There is no per-lane session
  cache or parking.
- **Per-lane AR floor:** each lane keeps the singleton's MTP floor (probe
  windows, sticky retirement, decayed per-depth costs), priced in AR-row
  units from a measured model of the shared step (an AR row costs its
  proportional share of an all-AR step; a window adds its verify rows'
  marginal cost), so a lane whose drafts do not pay falls back to AR rows.
- **Exact per lane:** every MTP lane window is byte-identical to the one
  singleton window it takes (a batched window at the lane's draft depth, or
  the singleton's AR step), on the lane's own states. Stateful stages (PLE
  conv, GDN core, QSA core) run per lane on the lane's own state; a row-local
  stage runs once over all lanes' rows only where the G0 byte-identity oracle
  admitted it, otherwise per lane.
  On gfx1151 G0 admitted every row-local stage (embedding, HC read/write, the
  GDN/QSA projections, the MoE, final HC read, head, argmax); other cards run
  every stage per lane until their own G0 run.
- **Under `topology.pp` / `topology.ep` (2..=8 cards):** the same exact
  **AR** fn-lanes route is staged across the cards, with these differences from
  the single-card route above. The lane store is budgeted per device: every
  band (PP) or rank (EP) must satisfy `need <= free` on its own card, else
  nothing is staged and the singleton route is kept. There are **no native-MTP
  lanes** (`continuous_batch_spec: false`, spec mode `"off"`) and a running
  singleton is **never promoted** into a lane
  (`continuous_batch_promotion: false`). EP replicates the lane store on every
  rank; PP keeps a band-scoped store per band, the HC streams crossing each
  band edge through pinned host memory; a decode step is split into `G`
  contiguous whole-lane micro-batches that wavefront through the bands
  (`G` = the maximal overlap `min(lanes, 2 * stages)` under the default
  per-lane policy; adaptive over `1..=min(lanes, 2 * stages)` under a shared
  policy; the developer variables `HIPFIRE_PP_LANE_GROUPS=<G>` and
  `HIPFIRE_FN_LANE_POLICY=<spec>` override group count and stage policy, see
  [env-vars.md](env-vars.md)). Row-local stages run once over all lanes'
  rows only where per-stage proof rows cover the bound routes (gfx1201: PP2
  mq2 and EP2 mq4, byte-exact against the topology singleton); every other
  stage runs per lane. Measured on 2x R9700 (16 prompts x 256 tokens, 3 fresh
  processes per arm), aggregate output at C8 is 1.97x (PP2) and 1.83x (EP2)
  the batching-off C1, and batching on costs C1 0.3% (PP2) and 2.2% (EP2).
  Like the single-card route it stays default-off. Details:
  [multi-gpu.md](multi-gpu.md#flash-next-fn-lanes-continuous-batching-under-pp-and-ep).

The loaded ack reports `continuous_batch_route: "fn-lanes"`,
`continuous_batch_history_capable: true` (the typed capability the serve
gateway uses to admit appended multi-turn histories; it never matches route
names),
`continuous_batch_slots`, `continuous_batch_exact: true`,
`continuous_batch_sampling: "greedy_only"`, the speculation state
(`continuous_batch_spec`, `continuous_batch_spec_mode` = `"mtp"` or `"off"`,
`continuous_batch_spec_k` with MTP lanes, `continuous_batch_spec_requested`)
and the lane-store receipt:
`fn_lanes_kv_backend`, `fn_lanes_qsa_format`, `fn_lanes_gdn_format`,
`fn_lanes_ring_rows`, `fn_lanes_max_lanes`, `fn_lanes_row_budget`,
`fn_lanes_mapped_bytes`, `fn_lanes_stage_policy` (the per-stage sharing
policy) and `fn_lanes_stage_evidence` (the G0 receipt id for the production
policy). Each lane's `done` event carries the singleton's MTP counters
(`mtp`, `mtp_windows`, `ar_windows`, `mtp_retired`) plus `mtp_drafts` and
`mtp_accepted`. Under PP/EP the ack also reports `fn_lanes_topology`
(`"single"`, `"ep"` or `"pp"`) and `fn_lanes_device_receipts`, one entry per
card (EP rank order; PP input band to output band); the scalar `fn_lanes_*`
receipt is the root's (rank 0, or the output band), and
`continuous_batch_promotion` is `true` only on a single card.

## Multi-process notes

- One listener per bind; second serve on the same host/port exits.
- Prefer plain `hipfire stop` when the pidfile is healthy. `hipfire stop
  --force` / `restart` also reap this install's untracked daemons, but never
  another install's or a non-hipfire port holder. Avoid ad-hoc `pkill` unless
  you know the exact process tree.
- `hipfire chat` starts a tracked detached native service if none is healthy;
  stop it explicitly with `hipfire stop` when the session is done.

## Logs and state files

| Path | Role |
|---|---|
| `~/.hipfire/serve.log` | Detached stdout/stderr |
| `~/.hipfire/serve.pid` | Ownership record for stop/ps/restart |
| `~/.hipfire/config.toml` | Sparse typed global configuration |
| `~/.hipfire/models.toml` | Local catalog, aliases, registry identities, and per-model overrides |
| `~/.hipfire/{config,models}.json`, `per_model_config.json` | Legacy migration inputs only |

```bash
tail -f ~/.hipfire/serve.log
```

## Related

- CLI inventory: [CLI.md](CLI.md)
- Chat TUI attach behavior: [CHAT.md](CHAT.md)
- Containers: [CONTAINER.md](CONTAINER.md)
- NixOS service: [NIXOS.md](NIXOS.md)
