### Fixes: speculative decode stats

- **Legacy `generate_spec` counts each speculative window once again.** `0b6ba5c04` added a second `spec_cycles` counter that shadowed the τ counter and was also incremented before each step, so every legacy spec route reported `cycles`/`rounds` at 2× and `tau` halved (Gemma 4 EAGLE: real τ/2 + 0.5). The null-stream ledger guard now gets the 0-based window index from the single counter, so it leaves warm-up after the same number of windows rather than half as many.
