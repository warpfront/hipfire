### Fixes: engine_loop_ab G-spec

- **`scripts/engine_loop_ab.py` G-spec no longer fails on MTP noise or on a singleton reference.** On a `spec_noise` route the A/B run repeats the legacy arm (`--spec-reps`, default 2) and bounds the engine tau by the lowest legacy run − 1 %. Concurrency cells at k>1 take their tau reference from the `legacy-batch` arm (lanes vs lanes) instead of the serialized `ref` arm.
