### Fixes: concurrency bench system message

- **`hipfire bench --concurrency --system <text>` opens every conversation with a system message; `{conv}` becomes the conversation's index (`w<i>` for warmups).** A `prompt.system` set with `hipfire config set` (global user layer) never reached the bench: like `hipfire serve`, the bench reads that layer only where it overrides a registry model value. `scripts/engine_loop_ab.py --conc` now passes the route's `chat_system` message, named per conversation, so a `chat_system` route's follow-ups can reuse a prefix that is their own.
