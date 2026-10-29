#!/usr/bin/env python3
"""Replay exact Flash-Next routes with a fixed core and per-layer LRU slots.

CPU-only planning evidence. This does not model DMA latency, concurrent
transfers, kernel execution, allocator padding, or production admission.
The generation receipt fixes request boundaries; position resets are not guessed.
"""

import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path


LAYERS, EXPERTS, TOPK = 48, 512, 10


def identity(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


class Policy:
    def __init__(self, name, core, seeds, capacities, static=False):
        if any(0 < capacity < TOPK for capacity in capacities):
            raise ValueError("this bounded replay requires zero or at least top-k slots per layer")
        self.name = name
        self.core = core
        self.seeds = seeds
        self.capacities = capacities
        self.static = static
        self.hits = self.misses = self.fills = self.evictions = 0
        self.requests = []

    def begin_request(self):
        self.slots = [OrderedDict.fromkeys(seed) for seed in self.seeds]
        self.request_start = (self.hits, self.misses, self.fills)

    def token(self, routes):
        for layer in range(LAYERS):
            ids = routes[layer * TOPK : (layer + 1) * TOPK]
            cache = self.slots[layer]
            core = self.core[layer]
            # A token runs against its starting table. No within-token fill
            # turns a miss into a hit or removes a still-needed hit.
            missing = []
            for expert in ids:
                if expert in core or expert in cache:
                    self.hits += 1
                else:
                    self.misses += 1
                    missing.append(expert)
            if self.static:
                continue
            # Protect this token's existing hits before choosing victims.
            # Otherwise a miss early in the ranked list could evict a hit
            # later in that same list and manufacture an unnecessary refill.
            for expert in ids:
                if expert in cache:
                    cache.move_to_end(expert)
            for expert in missing:
                if self.capacities[layer]:
                    if len(cache) == self.capacities[layer]:
                        cache.popitem(last=False)
                        self.evictions += 1
                    cache[expert] = None
            # Final recency follows the recorded rank order. All fills are
            # assumed complete before the next token, without DMA modelling.
            for expert in ids:
                if expert in cache:
                    cache.move_to_end(expert)
            self.fills += len(missing) if self.capacities[layer] else 0

    def end_request(self):
        h, m, f = self.request_start
        self.requests.append(
            {"hits": self.hits - h, "misses": self.misses - m, "fills": self.fills - f}
        )

    def report(self, tokens, expert_bytes):
        total = self.hits + self.misses
        core_count = sum(map(len, self.core))
        slots = sum(self.capacities)
        return {
            "name": self.name,
            "core_experts": core_count,
            "device_slots": slots,
            "slots_per_layer": self.capacities,
            "resident_expert_capacity": core_count + slots,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total,
            "misses_per_token": self.misses / tokens,
            "idealized_fills_per_token": self.fills / tokens,
            "evictions": self.evictions,
            "host_payload_bytes_estimate": (LAYERS * EXPERTS - core_count) * expert_bytes,
            "device_expert_payload_bytes_estimate": (core_count + slots) * expert_bytes,
            "zero_copy_miss_MiB_per_token_estimate": self.misses / tokens * expert_bytes / 2**20,
            "fill_MiB_per_token_estimate": self.fills / tokens * expert_bytes / 2**20,
            "requests": self.requests,
        }


def per_layer(rank):
    result = [[] for _ in range(LAYERS)]
    for global_id in rank:
        layer, expert = divmod(global_id, EXPERTS)
        result[layer].append(expert)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routes", required=True, type=Path)
    parser.add_argument("--generations", required=True, type=Path)
    parser.add_argument("--hotset", required=True, type=Path)
    parser.add_argument("--resident-experts", type=int, default=8737)
    parser.add_argument("--expert-bytes", type=int, default=2611200,
                        help="payload only: QT44 GU1280x2560 + QT53 down2560x640")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if not 1 <= args.resident_experts < LAYERS * EXPERTS or args.expert_bytes <= 0:
        parser.error("invalid capacity or expert payload bytes")
    hotset = json.loads(args.hotset.read_text())
    if (hotset.get("layers"), hotset.get("experts_per_layer"), hotset.get("top_k")) != (
        LAYERS, EXPERTS, TOPK
    ):
        raise ValueError("hotset geometry is not Flash-Next 48x512/top10")
    rank = hotset["rank"]
    if len(rank) != LAYERS * EXPERTS or set(rank) != set(range(LAYERS * EXPERTS)):
        raise ValueError("hotset rank must be a full permutation")
    generations = [json.loads(line) for line in args.generations.read_text().splitlines() if line]
    if not generations:
        raise ValueError("empty generation receipts")
    resident = rank[: args.resident_experts]
    original = per_layer(resident)
    empty = [[] for _ in range(LAYERS)]
    policies = [Policy("static_hotset", [set(x) for x in original], empty, [0] * LAYERS, True),
                Policy("cold_lru170", [set() for _ in range(LAYERS)], empty, [170] * LAYERS)]
    for percent in (25, 50, 75):
        count = args.resident_experts * percent // 100
        core = per_layer(resident[:count])
        dynamic = per_layer(resident[count:])
        # Reversed seed order makes the hottest seed MRU, not the coldest.
        policies.append(Policy(f"core{percent}_warm_lru", [set(x) for x in core],
                               [list(reversed(x)) for x in dynamic], list(map(len, dynamic))))
    tokens = 0
    with args.routes.open() as stream:
        for request_index, generation in enumerate(generations):
            if generation["i"] != request_index:
                raise ValueError("generation receipts are not in request order")
            prompt = generation["usage"]["prompt_tokens"]
            count = generation["usage"]["completion_tokens"]
            if count <= 0:
                raise ValueError("a request has no routed decode tokens")
            for policy in policies:
                policy.begin_request()
            for step in range(count):
                line = stream.readline()
                if not line:
                    raise ValueError("route trace ends before generation receipts")
                values = [int(x) for x in line.split()]
                if len(values) != 1 + LAYERS * TOPK or values[0] != prompt + step:
                    raise ValueError(f"route geometry/position mismatch at request {request_index}/{step}")
                routes = values[1:]
                for layer in range(LAYERS):
                    ids = routes[layer * TOPK : (layer + 1) * TOPK]
                    if len(set(ids)) != TOPK or any(not 0 <= x < EXPERTS for x in ids):
                        raise ValueError("invalid or repeated expert id")
                for policy in policies:
                    policy.token(routes)
                tokens += 1
            for policy in policies:
                policy.end_request()
        if stream.read().strip():
            raise ValueError("route trace has rows absent from generation receipts")
    report = {
        "scope": "offline_capacity_and_traffic_estimate_only",
        "base": "9f71309bfc7542a78e748e478bc751d4111c2845",
        "inputs": {name: identity(getattr(args, name)) for name in ("routes", "generations", "hotset")},
        "tokens": tokens,
        "requests": len(generations),
        "model_sha256_from_hotset": hotset["model_sha256"],
        "payload_bytes_per_expert_assumption": args.expert_bytes,
        "limitations": [
            "No inference, DMA latency, asynchronous completion, or speed measurement.",
            "Hits use the start-of-token table; fills become available before the next token.",
            "Each request resets to its configured empty or static-hotset warm start.",
            "Host/device bytes exclude alignment, padding, tables, runtime, prefill staging and other weights.",
            "Static hotset keeps canonical VRAM entries without host copies; its host estimate is core-only.",
            "The raw trace is used directly, without recomputing or changing routing.",
        ],
        "policies": [p.report(tokens, args.expert_bytes) for p in policies],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"tokens": tokens, "requests": len(generations), "out": str(args.out),
                      "hit_rates": {x["name"]: x["hit_rate"] for x in report["policies"]}}))


if __name__ == "__main__":
    main()
