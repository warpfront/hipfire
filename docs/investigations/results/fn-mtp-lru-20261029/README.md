# Native MTP with the exact Flash-Next expert LRU

This follow-up builds on frozen PR #1022 at `2c1c46365e9f5da9fd361dfccdeef0e8840c4846`.
The existing native MTP executor and expert-cache transport are extended with
one stable reader per bounded verify batch. No throughput result is claimed here.

Current status: 207 scoped product/regression CPU test executions and nine
Rust oracle controls pass. The no-model GPU controls at `7cd1213a6` pass collector rows 1–8 and
paired publication; their hashes are in `hardware-controls-7cd1213a6.json`.
The first cache-off model attempt was rejected before model materialization:
its fixed 20,000 MiB hot budget exceeded 19,158 MiB after the matched K7
reserves. A matching 18,000 MiB budget is planned in both arms. Full model
qualification remains pending; CPU and no-model controls do not qualify native MTP.

`fn_mtp_lru_oracle` requires `lab,reference-parity`. Its default is plan-only;
`--cpu-controls` opens no GPU. Explicit collector execution uses tiny fixtures
and the production collector. The separate existing
`fn_expert_cache_publication --execute` exercises paired async transport.

The control (cache off) runs first, followed by cache on, in separate fresh
processes. The model arm uses the product loader, the same K7 resource reserve in both
cache modes, and real forced native K3 windows. It compares target R1/2/4/8
logits/state, native token IDs, completion/publication tickets, terminal repair
and prefix restore. Different head schedules are diagnostic; cache-on/off head
hashes are comparable only after their actual native schedules match.
Terminal repair replays the recorded native windows from the original prompt.
It compares the consumed target history with sequential AR, then checks live
session planning preserves repaired state. Cold-prefilling generated history
changes the GDN/head schedule, so cross-schedule next-seed equality is excluded.
The independent verifier also compares repaired head/target hashes across cache
modes only under identical replayed windows.
See `EXACTNESS-POLICY.json` for the precise domain and exclusions.

The developer runner and independent receipt verifier are under `tools/`.
Run no-model controls before model arms, retain every failed attempt, and hold
the shared project GPU lock. The runner never retries or overwrites proof.
Its timings are instrumented and are not performance evidence.
