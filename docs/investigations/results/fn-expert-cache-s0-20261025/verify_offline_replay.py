#!/usr/bin/env python3
"""Check the cache replay against an independent full-history timestamp oracle."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    source = args.repo / "scripts/fn-expert-lru-replay.py"
    spec = importlib.util.spec_from_file_location("fn_lru_replay", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cases = []
    for capacity in [0, 10, 11, 17, 64]:
        rng = random.Random(4100 + capacity)
        core = [set(range(3)) for _ in range(module.LAYERS)]
        seeds = [list(range(3, 3 + capacity)) for _ in range(module.LAYERS)]
        policy = module.Policy("oracle", core, seeds, [capacity] * module.LAYERS)
        policy.begin_request()
        # Unlike the bounded OrderedDict implementation, this reference keeps
        # every historical timestamp and sorts the complete history each step.
        history = [{expert: i - len(seed) for i, expert in enumerate(seed)} for seed in seeds]
        resident = [set(seed) for seed in seeds]
        hits = misses = fills = 0
        for step in range(120):
            routes = []
            for layer in range(module.LAYERS):
                ids = rng.sample(range(512), 10)
                routes.extend(ids)
                absent = sum(expert not in core[layer] and expert not in resident[layer]
                             for expert in ids)
                hits += 10 - absent
                misses += absent
                fills += absent if capacity else 0
                for order, expert in enumerate(ids):
                    if expert not in core[layer]:
                        history[layer][expert] = step * 10 + order
                resident[layer] = set(sorted(history[layer], key=history[layer].get,
                                             reverse=True)[:capacity]) if capacity else set()
            policy.token(routes)
            for layer in range(module.LAYERS):
                if set(policy.slots[layer]) != resident[layer]:
                    raise AssertionError(("slot mismatch", capacity, step, layer))
        if (policy.hits, policy.misses, policy.fills) != (hits, misses, fills):
            raise AssertionError(("hit/miss/fill mismatch", capacity))
        cases.append({"capacity": capacity, "tokens": 120, "layers": module.LAYERS,
                      "slots_equal_slow_timestamp_reference": True,
                      "hit_miss_fill_counts_equal": True})
    report = {
        "scope": "CPU-only replay checker vs independent full-history timestamp reference",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "checker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases": cases,
        "passed": True,
    }
    encoded = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
