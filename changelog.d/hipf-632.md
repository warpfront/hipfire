### Fixes: engine_loop_ab concurrency cache gate

- **`scripts/engine_loop_ab.py` `conc.require_reuse` is a strict cache gate.** Every follow-up on the engine and `ref` arms must reuse a prefix, every first turn must start cold, both arms must hold the same requests, and each k>1 request must equal its k=1 request; refusals and earlier-turn reuse exempt nothing. Without it, zero prefix hits only mark the run `pass-partial`.
