# Chat

hipfire provides two native Rust chat surfaces:

- `hipfire tui` (or bare `hipfire`) opens the full ratatui application and its
  Chat tab. This requires the `hipfire-tui` binary; when it is not installed
  the command exits with an error that names the build command.
- `hipfire chat [model]` is a lightweight line-oriented multi-turn client for
  terminals and scripts. It is part of the `hipfire` binary and does not need
  `hipfire-tui`.

Both use the native OpenAI-compatible service documented in
[`SERVE.md`](SERVE.md).

## Full terminal UI

```bash
hipfire
# or
hipfire tui
```

Open the Chat tab, select a downloaded model in Models, or use `/model <tag>`.
The TUI supports streaming, multiline input (`Ctrl+O`), cancellation, saved
sessions, model/system/sampling commands, regeneration/edit/copy actions, and
live serve status.

Chat-tab slash commands run locally and never send a message (a message whose
text must begin with `/` is sent as `//text`, which transmits `/text`):

| Command | Effect |
|---|---|
| `/model [tag]` (`/m`) | Show the active model, or switch to `tag`. The tag must be a known registry/local entry or alias, otherwise the command reports `unknown model`. |
| `/system [text]` (`/sys`) | Set the system prompt; with no text, clear it. |
| `/temp [value\|off]` | Per-session temperature override, `0..=2`; no value or `off` returns to the serve default. |
| `/top_p [value\|off]` | Per-session nucleus override, `0..=1`; no value or `off` returns to the serve default. |
| `/clear` | Clear the whole conversation. |
| `/save <name>`, `/load <name>` | Save the current conversation as a named session, or replace the conversation with a saved one. |
| `/sessions` (`/ls`), `/delete <name>` (`/del`) | List saved session names, or delete one. |
| `/regen` (`/retry`) | Drop the last reply and regenerate from the last user turn. |
| `/edit` | Pull the last user turn back into the input (dropping its reply) for editing and resending. |
| `/copy` | Copy the last non-empty assistant reply to the clipboard (OSC 52 escape; replies over 100 KB are refused). |
| `/help` | Show the command and key list as a note in the transcript (never sent). |

The transcript follows new output while you are at the bottom: Up/Down and
PgUp/PgDn scroll (also while a reply streams), Home jumps to the top and End
resumes following. Typing keeps working during a reply; only Enter waits for it
to finish, and Esc stops it (the reply is marked `[stopped]`). Reasoning text is
shown dimmed under a `thinking` label. The token count under a reply is the
serve's `usage.completion_tokens` when the serve reports it, otherwise the
number of streamed chunks, labelled as chunks.

The TUI checks the serve's `/health` every 2 s on every tab. When it is not
online, Enter queues your prompt (Esc cancels it and puts the text back), starts
`hipfire serve -d` only when the first check says the serve is down and no
serve process exists, and sends the prompt by itself once the serve is online. Config, registry, model lifecycle, and HTTP calls all use the shared
Rust crates; there is no separate script runtime.

## Lightweight chat command

```bash
hipfire chat qwen3.6:35b-a3b-mq4r
```

```text
you> Explain wave32 WMMA briefly.
assistant> ...
you> /clear
you> /exit
```

Supported local commands are `/clear` (drops every message except the system
message), `/exit`, and `/quit`. Flags:

| Flag | Purpose |
|---|---|
| `-t, --temp <f>` | Explicit temperature for this session, `0..=2` |
| `--top-p <f>` | Explicit nucleus probability, `(0, 1]` |
| `-n, --max-tokens <n>` | Per-turn generation cap, `1..=393216`; default `generation.max_tokens` |
| `--system <text>` | Initial system message |
| `--no-color` | Compatibility flag; the line client emits no ANSI color |

`[model]` is optional and defaults to the configured `serve.default_model`.

If no healthy service exists, the command starts a tracked detached native
service on the configured host/port with lazy model loading. That service
remains available after the line client exits; use `hipfire stop` when you want
to release it.

## Resolution and request behavior

- Model names use the same registry/alias/path resolver as `run` and `serve`.
- The service resolves load-time KV/speculation/model overrides from
  `config.toml`, `models.toml`, registry recommendations, and compatible env.
- Explicit chat sampling flags are sent on every turn. Omitted sampling fields
  follow the service's per-model/registry/daemon fallback contract.
- The full `messages` array is sent each turn, so conversation context is
  preserved by request content rather than hidden client state.
- Streaming answer and reasoning deltas share the same Rust HTTP client used by
  the TUI.

## Thinking and model framing

Reasoning models may emit a thinking stream before the answer. The HTTP API
exposes it as `reasoning_content`; visible answer text remains `content`.

Mode, semantic effort, and hard cap are **independent** — see
[`CONFIG.md`](CONFIG.md) for keys/defaults and [`SERVE.md`](SERVE.md) for
per-request HTTP fields (`reasoning_effort`, `enable_thinking`,
`max_think_tokens`, legacy `thinking_budget`). Effort never maps to a token
budget. Unsupported or cross-family values are dropped with a warning (server
log + OpenAI `hipfire` response metadata), not silently reinterpreted.

Chat framing and stop behavior are model-specific. Use the exact registry tag
and do not assume Qwen `<think>` conventions apply to LFM, Gemma, Glimmer, or
DeepSeek. The LFM framing smoke route is `scripts/serve_harness.py` with the
exact `lfm2.5:*` tag; see [`VALIDATION.md`](VALIDATION.md).

## Troubleshooting

| Symptom | Action |
|---|---|
| Model not found | `hipfire list -r`, then `hipfire pull <tag>` |
| Service will not start | `hipfire diag`; inspect `~/.hipfire/serve.log` |
| Port conflict | `hipfire ps`; stop the owned service with `hipfire stop` |
| Truncated answer | Raise `--max-tokens`; on Qwen, if the think span was explicitly force-closed, raise `max_think_tokens` or leave it uncapped |
| Need richer session controls | Use `hipfire tui` and its Chat tab |

The service has no authentication or TLS. Local chat should use a loopback bind;
see [`SERVE.md`](SERVE.md) before exposing it to a network.
