### Fixes: engine_loop_ab concurrency cache gate

- **`scripts/engine_loop_ab.py` `conc.require_reuse` fails a follow-up that reuses nothing.** Without it, zero prefix hits only mark the run `pass-partial`. The concurrency tap also keeps the requests the bench sent (`<wire>.in`), so a prompt is checked on the wire, not in the configuration.
