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

## Security (no auth / no TLS)

The native handler implements **no authentication and no TLS**.
Anyone who can reach the bind address can call every endpoint, including
chat completions. Default bind is `127.0.0.1` (loopback only); set
`serve.host = "0.0.0.0"` (or pass `0.0.0.0:11435`) to listen on all interfaces.

- Prefer loopback for local use: `hipfire serve 127.0.0.1:11435`
- Expose beyond localhost only behind a trusted network or an authenticated
  TLS reverse proxy you control. Do not publish the raw port to the internet.

## Start and stop

```bash
hipfire serve                         # foreground; Ctrl-C stops (default bind 127.0.0.1)
hipfire serve 127.0.0.1:11435         # loopback-only (preferred local bind)
hipfire serve -d                      # background (setsid/nohup); polls /health up to 300s
hipfire serve qwen3.5:9b -d           # pre-warm a specific tag this run
hipfire serve 0.0.0.0:11435 -d        # all-interfaces bind — unauthenticated; see Security
hipfire serve --no-prewarm -d         # bind first; load on first request
hipfire serve --kv-mode q8 --idle-timeout 0 -d
hipfire serve --tp 2 -d               # expert-parallel load (multi-GPU arches)

hipfire ps                            # daemons / quantize / uploads + port state (Linux)
hipfire stop                          # SIGTERM tracked pid in serve.pid (safe default)
# Destructive — inspect first (hipfire ps, curl /health, port owner):
hipfire stop --force                  # also pkill -x daemon + fuser -k <port>/tcp
hipfire stop --all                    # --force plus pkill orphan quantize jobs
hipfire restart -d                    # stop --force semantics, then serve again
```

**Destructive lifecycle flags.** Plain `hipfire stop` only SIGTERMs the tracked
pid after ownership checks — use that by default. `--force`, `--all`, and
`restart` call `reapOrphans`: system-wide `pkill -x daemon` (exact name), and
`fuser -k <port>/tcp` which **SIGKILLs whichever process owns the port**.
`--all` also runs `pkill -f` against release quantize binaries. Before using
them, run `hipfire ps`, `curl -s http://127.0.0.1:<port>/health`, and confirm
the port owner so you do not kill an unrelated daemon or listener.

Flags (also accepted by `restart`):

| Flag | Effect |
|---|---|
| `-d` / `--detach` / `--background` | Fork detached child; log → `~/.hipfire/serve.log` |
| `--kv-mode <m>` | KV preset for models this service loads (sent as the `kv_mode` load param; same contract as `run --kv-mode`) |
| `--idle-timeout <s>` | Sets `HIPFIRE_IDLE_TIMEOUT` (0..86400) |
| `--no-prewarm` | Sets `HIPFIRE_NO_PREWARM=1` |
| `--tp N` | Sets `HIPFIRE_TP` (1..64) for expert-parallel load |

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
   Multi-slot serve (`serve.multi_slot = true`) fails closed instead: the slot
   engine is its only backend, so a failed pre-warm (for example, slot-engine
   allocations that do not fit beside another process on the card) logs the
   load error, answers `/health` with 503 `unhealthy`, and exits 1.
4. **HTTP.** The native server accepts traffic. Serve is single-stream by
   default: one generation holds the daemon at a time and later requests wait
   in a bounded queue (`serve.max_queue`). Continuous batching covers only
   thinking-off, single-turn Qwen requests (`serve.continuous_batch_size`).
   With `serve.multi_slot = true`, up to `serve.multi_slot_slots` requests
   decode at once on the experimental multi-slot engine. Concurrent requests
   can produce different greedy text than serial requests at ≥4 slots; output
   is deterministic for a fixed batch composition. 2 slots matched serial in
   earlier testing; on 35B-A3B MoE checkpoints they diverged from serial on
   1–4 of 4 prompts. A waiter that is still queued after `serve.queue_timeout_ms`
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
| `GET` | `/health` | Liveness JSON: `status`, `model`, `loading_model`, `pid`, `native`, `capabilities` |
| `GET` | `/v1/models` | `{ data: [{ id }, ...] }` from local model files |
| `GET` | `/stats` | Serve telemetry: uptime, queue depth, requests served, recent decode tok/s |
| `POST` | `/v1/chat/completions` | Chat completions (stream or non-stream) |
| `GET` | `/ui`, `/ui/*` | Embedded chat UI (opt-in; see below) |

There is **no** `/v1/completions` route in the current CLI serve.

### Embedded chat UI (`--ui` / `serve.ui` / `HIPFIRE_SERVE_UI`)

Off by default. `hipfire serve --ui` (or `--open`, which implies `--ui`
and launches the browser) serves a dependency-free chat frontend at
`/ui` on the same listener; `serve.ui = true` or `HIPFIRE_SERVE_UI=1`
enable it for any serve start, including detached ones. `GET /`
redirects to `/ui` only while the UI is enabled.

The UI talks to the same `/v1/chat/completions` gateway as any other
client and keeps chat history in the browser's IndexedDB — serve stores
nothing on its behalf. Like every serve endpoint it is **unauthenticated**:
binding a non-loopback host with the UI enabled prints a warning, since
that makes the gateway reachable from any browser on the network.

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
reload.

Model switch lifecycle: a load request is first admitted read-only; a refused
admission leaves the prior model (and its drafter) loaded. Once admitted, a
single-device or pipeline-parallel load (Flash-Next/Qwen4 included) unloads
the prior model and requires clean VMM state before constructing the new
one, so a failure after admission leaves no model loaded and the error says
`no model loaded`. Only tp>1 expert-parallel loads stage the new model before
retiring the prior one.

```bash
# Loopback example (safe default for local smoke):
curl -s http://127.0.0.1:11435/health
```

### `POST /v1/chat/completions`

```bash
curl -N http://127.0.0.1:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5:9b",
    "messages": [{"role": "user", "content": "hi"}],
    "stream": true
  }'
```

- **`stream: true`** (typical clients): SSE `data: {chat.completion.chunk}`
  lines until `data: [DONE]`. The `200` and the role chunk go out when
  generation sends its first frame, or after 15 s of silence (cold load, long
  prefill), so a request that fails before then gets the same JSON error and
  status as a non-stream request. A stream that fails after that ends with
  `data: {"error": {"message", "type"}}` and then `data: [DONE]`. Until
  `[DONE]`, 15 s without a frame sends a `: keepalive` SSE comment, which
  SSE parsers ignore.
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
| `temperature`, `top_p`, `top_k`, `min_p`, `repeat_penalty` | Sampling; explicit request values win, otherwise per-model TOML / registry-card values are applied |
| `seed` | OpenAI-compatible sampling seed: non-negative integer (≤ u64::MAX). Same seed is reproducible only when prompt, sampling parameters, and execution shape match. Cold vs prefix-resumed and solo vs co-batched shapes may differ until `serve.batch_invariant` is implemented; `null`/omitted = fresh entropy. Negative/fractional/non-integer values are rejected. |
| `presence_penalty`, `frequency_penalty` | Forwarded natively to the daemon (≥ 0); `presence_penalty` also inherits per-model / registry defaults |
| `max_tokens` | Generation cap |
| `stop` | A string or an array of up to 4 strings, each ≤ 64 characters; any other value is a 400. Matched on the answer only (never inside reasoning), and the stop text is not returned. Honoured on the Qwen3.5-family routes (`qwen_ar`, `qwen_dflash`, which covers MTP); every other route answers 400 instead of ignoring it |
| `tools` | Tool definitions (with structured `messages` when Jinja chat is on) |
| `chat_template_kwargs.enable_thinking` | Mode axis. `false` forces a no-think turn (Qwen native empty closed think). Independent of effort and cap. |
| `chat_template_kwargs.preserve_thinking` | Keep `<think>` in final non-stream content |
| `thinking` / `thinking.type` | Mode axis (`on`/`off`, or DeepSeek-style `enabled`/`disabled`). Thinking disabled wins and drops effort/cap with a warning. |
| `reasoning.effort` / `reasoning_effort` | **Semantic effort only** — prompt strength. Never mapped to a think-token budget. Family ladders differ; unsupported/cross-family values are dropped with a warning (see below). |
| `reasoning.max_tokens` / `max_think_tokens` | Explicit **integer** think-span cap on Qwen Jinja contracts. `0` or omitted = uncapped. DeepSeek, Gemma, Glimmer, and unsupported contracts drop it with a warning. Independent of effort. |
| `thinking_budget` / `reasoning.budget` | Legacy **named** cap preset only on non-effort-native Qwen templates that still accept it. Dropped+warned elsewhere. |

### `POST /v1/images/generations`

OpenAI-shaped image generation on a loaded diffusion checkpoint (arch 40
FLUX.1, arch 45 FLUX.2 Klein). Body fields: `model` (must already be loaded on
this server), `prompt`, optional `width` /
`height` (or OpenAI `size: "WxH"`), `steps` (defaults to the architecture
default: 4 for `flux.schnell`, 28 for `flux.dev`), `seed`, `n` (must be 1),
`response_format` (must be `b64_json`). References the same denoise path as the
`img_generate` daemon message (`hipfire img`).

```json
{"model": "flux.schnell:1", "prompt": "a tiny lighthouse on a rock, sunset", "size": "512x512", "response_format": "b64_json"}
```

Response: `{ "data": [ { "b64_json": "…", "width": 512, "height": 512, "seed": 0, "steps": 4 } ], "model": "…", "hipfire": { "ms": 1234 } }`.

This body takes no image input. A request with an `images` field is refused
with a 400 that points at `/v1/images/edits`.

### `POST /v1/images/edits`

OpenAI-shaped reference edit (FLUX.2 Klein, arch 45 only): `multipart/form-data`
with one to four `image` file parts (PNG or JPEG, at most 32 MB each after
upload) plus the text fields of `/v1/images/generations` (`prompt` required;
`size`, `steps`, `seed`, `n`, `response_format`). The image bytes travel in
the request; the server never reads a file the client names. Each reference
is area-capped at 1 MP and floored to a multiple of 16, and conditions every
denoise step without being denoised. Without `size` the output takes the
first reference's size. The whole body counts against `serve.max_request_bytes`.

```bash
curl -s -X POST http://127.0.0.1:11580/v1/images/edits \
  -F image=.png -F prompt="make the bicycle blue" -F steps=4 -F seed=0 \
  | jq -r .data[0].b64_json | base64 -d > bike-blue.png
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

**Examples:**

```bash
# Qwen3.8 — disable thinking natively (empty closed think block)
curl -s http://127.0.0.1:11435/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.8:27b",
  "messages": [{"role": "user", "content": "hi"}],
  "chat_template_kwargs": {"enable_thinking": false}
}'

# Qwen3.8 — semantic effort only (still uncapped unless max_think_tokens is set)
curl -s http://127.0.0.1:11435/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.8:27b",
  "messages": [{"role": "user", "content": "plan a refactor"}],
  "reasoning_effort": "medium"
}'

# Ornith 1.5 — Qwen3.8-compatible semantic effort (default xhigh; still uncapped unless max_think_tokens)
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
the partial `reasoning_content` is returned and `content` is empty.

Streaming error contract: a failure that happens BEFORE the first byte is a
plain HTTP error status (no SSE body). A failure after the stream has started
emits an OpenAI-shaped SSE `data: {"error": {...}}` frame followed by
`data: [DONE]`, so a streaming client always sees a terminal event instead of
a silently truncated `200`. Non-streaming failures keep the typed
status mapping (`400` request/config, `503` + `Retry-After` overload or
transient, `500` internal).

Prefix-cache capable arches (daemon `cache_capable`, or arch allowlist
`deepseek4` / `qwen3_5` / `qwen3_5_moe`) skip per-request `reset` so multi-turn
LCP can hit. Other arches reset every request (stateless OpenAI shape).

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

## Multi-process notes

- One listener per bind; second serve on the same host/port exits.
- Prefer plain `hipfire stop` when the pidfile is healthy. Reach for
  `hipfire stop --force` / `restart` only after `hipfire ps` and `/health`
  show a stuck orphan or port holder — those paths are destructive
  (system-wide `pkill -x daemon` + `fuser -k` on the port). Avoid ad-hoc
  `pkill` unless you know the exact process tree.
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
