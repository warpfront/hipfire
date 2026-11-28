### Fixes: G12 engine loop, engine-loop A/B harness

- **Engine-loop A/B harness: the batch scenario checks that the D2 and D3 aborts land on the resident state (HIPF-616).** The engine logs `[engine] request <tag> cancelled on the resident state` for each resident cancel; a lane cancel logs nothing. `scripts/engine_loop_ab.py --batch` maps the tags to requests (`resident_cancels` in the receipt) and fails the engine arm when D2 or D3 is not in that list. Before, a changed timing that cancelled D2 in a lane passed every reuse gate, so the resident cancel path of HIPF-602 had no detector.
