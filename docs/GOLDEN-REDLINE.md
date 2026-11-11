# Golden Redline reproduction

One-command, fail-closed reproduction of the measured Qwen 3.6 35B-A3B MQ4R
retained-PM4 routes on gfx1100, gfx1151, and gfx1201.

| Field | Value |
|---|---|
| Page state | **branch-implemented** |
| Fixture registry | [`registry/redline-golden-v1.json`](../registry/redline-golden-v1.json) |
| Runner | [`tools/redline/golden.py`](../tools/redline/golden.py) (`python3 -m tools.redline golden`) |
| Product harness | [`tools/redline/product_bench.py`](../tools/redline/product_bench.py) (`python3 -m tools.redline bench`) |
| Certification policy | [`REDLINE.md`](REDLINE.md) |

This runner is developer orchestration. Persistent product configuration still
goes through the native Rust `hipfire` CLI. Passing a golden reproduction does
not create a model/route admission; admissions remain owned exclusively by
[`admissions.yml`](admissions.yml).

## Run it

From a source checkout:

```bash
python3 -m tools.redline golden
```

The runner:

1. selects the fixture for the physical GPU selected by `--device`;
2. filters that physical device through ROCr and exposes it to HIP as logical
   device zero;
3. verifies the checked-in MQ4R registry card and sampling-profile hashes;
4. verifies the exact 18.7 GB model size and SHA-256 before GPU load;
5. builds the release daemon when it is absent;
6. runs the product TG128 HIP-versus-retained-PM4 benchmark with the sealed
   Q8, context, warmup, stationarity, transport, and PM4-policy parameters;
7. requires positive timed-arm route proof and the exact architecture-specific
   tape/prepared identity;
8. checks the architecture-specific throughput and speedup floors; and
9. writes both the raw product report and a hashed golden attestation under
   `.redline-work/golden/`.

Useful forms:

```bash
# Show fixtures without touching a GPU.
python3 -m tools.redline golden --list

# Pull the model when absent and pin it as the default after a pass.
python3 -m tools.redline golden --pull --set-default --yes

# Select a physical device on a multi-GPU host.
python3 -m tools.redline golden --device 3

# Print the exact command without building, hashing, loading, or running.
python3 -m tools.redline golden --arch gfx1201 --dry-run

# Audit an existing report; exact source and daemon hashes are mandatory here.
python3 -m tools.redline golden \
  --arch gfx1100 \
  --report /path/to/report.json \
  --strict-binary
```

The stationarity ceiling is 120 TG128 rows. This preserves the existing slope,
spread, confirmation, and median-drift criteria while allowing slow cold-start
clock convergence. It does not relax the acceptance gates.

## Reproducing on a box that disagrees

`python3 -m tools.redline golden` pins the model, sampling profile, benchmark contract, PM4
policy and route identity — but not the compiled code objects or the host
toolchain. When a contributor reports the same route identity (same dispatch
count, kernel count and sequence hash) with different throughput, the
divergence is below the tape and none of the above will catch it.

`scripts/redline-repro-package.sh` closes that gap.

```bash
# On a box where the fixture passes, right after a successful golden run:
scripts/redline-repro-package.sh capture --arch gfx1201

# On the box that cannot reproduce:
scripts/redline-repro-package.sh verify --package repro-gfx1201-*.tar.gz
scripts/redline-repro-package.sh run    --package repro-gfx1201-*.tar.gz
```

The package size and code-object count depend on the arch and kernel cache it
was captured from; the manifest records the actual `count` and `total_bytes`.
It carries the compiled code objects (`.hsaco` and `.hash` files from the
kernel cache) with a manifest hash, the ROCm and code-generator versions, GPU
state, the PM4 policy, the acceptance floors, the required tape, and the
reference attestation.

`verify` classifies every difference as BLOCKING (arch, model SHA-256, PM4
policy) or ADVISORY (source commit, daemon hash, ROCm version, code-generator
version, code-object manifest) and exits 2 on a blocking mismatch. `--force`
proceeds anyway.

`--pin-kernels` copies the packaged `.hsaco`/`.hash` files into
`kernels/compiled/<arch>/`, evicts the local kernel cache for that arch, and
runs the golden fixture with `HIPFIRE_NO_DEVICE_COMPILER=1` so the engine
treats the device compiler as absent instead of recompiling with the local
toolchain.

**Known compatibility limitation (current source, not exercised here).** The
script's own header comments and echo lines still describe an older engine
branch in which an unverified pre-compiled blob is used with an
`Output may be incorrect` warning. The current compiler
(`crates/rdna-compute/src/compiler.rs`) does not have that branch. With no
compiler it accepts only an installed object whose `<module>.index.json`
verifies (source SHA-256, flags, cache ABI, packaging key, toolchain identity)
and whose `.hsaco`/`.hash` pair and object SHA-256 match the index. A
mismatched index, a missing index for an installed object, or an unverified
legacy object is rejected with an error ("packaged object rejected" /
"unverified kernel image"); nothing falls back to using it. The script
packages and installs only `.hsaco` and `.hash` files, not `.index.json`, so a
pinned run is expected to fail closed rather than execute those objects. Treat
`--pin-kernels` as unsupported against the current compiler until the script
is updated; `verify` (the BLOCKING/ADVISORY comparison above) is unaffected.

## What counts as a pass

Identity is exact. Performance is an evidence-bound floor rather than a demand
that every board produce the same final decimal.

| Architecture | Route-proof reference | Acceptance floor | Required tape (registry `route`) |
|---|---:|---:|---|
| gfx1100 | 251.798 tok/s | 245.000 tok/s and 1.08x | 603 dispatches / 22 kernels / `9a15e1013970b7d7` |
| gfx1151 | 115.290 tok/s | 115.021 tok/s and 1.10x | 604 dispatches / 23 kernels / `42f566b752920679` |
| gfx1201 | 202.460 tok/s | 197.000 tok/s and 1.10x | 693 dispatches / 22 kernels / `c3e02aad4cf40828` |

The README headline measurements—253.31, 115.10, and 203.93 tok/s—remain
recorded in the fixture registry as historical measured context. The table
above uses the newer positive route-proof reports where available. The
gfx1151 record also pins the sealed certification payload, HIP-source
aggregate, dispatch-source aggregate, 23-HSACO manifest, model, and daemon
hashes.

### gfx1151 Silver baseline

**Silver** is the coherent stopgap on the path back to, and then beyond, the
gfx1151 Golden result. It does not lower or replace the Golden acceptance
floor above.

The preserved Silver source snapshot is
`8445fca2acd462d8e9d7547de2fe3823874c24bb`. Its daemon SHA-256 is
`73a89a1ea57f8938ea02e264ebdd55b8d0b11f950a95f08def4553f580dc795f`;
both reports use the same 604-launch / 23-kernel tape with sequence hash
`42f566b752920679`.

| Evidence | PM4 median | What it proves | Report SHA-256 |
|---|---:|---|---|
| Silver high-water | **114.209 tok/s** | Stationary 8-run measurement and complete retained-route proof | `27f04979fc98fe8315936916c67e9b4560fafde0c2408a909c3d069bb81f8234` |
| Silver coherence certification | **113.652 tok/s** | Same daemon and tape; HIP and retained-PM4 CLI/serve one-turn Flagstaff checks (the product harness's embedded serve smoke) passed | `c728740e9f0446124d75ea850cb363c79cf0dfa4c37d52b1386d7fcc8902f681` |

The high-water report predates the embedded coherence step (an arm of
`tools.redline bench`, not the retired `coherence-gate-*` scripts). Coherence
is therefore certified by the later same-binary report, not inferred from the
114.209 tok/s number alone. The retained route is unchanged. Against the
115.021 tok/s Golden floor, the Silver high-water is 0.812 tok/s slower
(0.71%, conventionally reported as the small approximately 1 tok/s loss).
ROCm 7.14 is only a hypothesis for that gap; neither causality nor absence of
a Redline route regression is proven. Silver remains a stopgap and never a
Golden reproduction.

Results are classified explicitly:

- `exact-reference-binary`: source commit, daemon bytes, model, benchmark
  contract, route identity, and performance gates all match.
- `route-compatible-reproduction`: source or daemon is newer, but the exact
  model, benchmark contract, route identity, and performance gates pass.
- `failed`: any required identity, route-proof, stationarity, throughput, or
  speedup gate fails.

Use `--strict-binary` when only the exact pinned source and daemon bytes count.

## Make the validated model the default

After a successful interactive run, the script offers:

```text
Set this model as the hipfire default with its pinned registry sampling profile and Q8 KV? [y/N]
```

Accepting it, or passing `--set-default`, uses the native CLI to set
`serve.default_model` to `qwen3.6:35b-a3b-mq4r` and copy the validated
registry profile into that model's TOML overlay. A global generation override
would otherwise take precedence over registry defaults; the model-specific
layer makes this handoff deterministic without deleting unrelated global
preferences.

| Setting | Value |
|---|---:|
| temperature | 1.0 |
| top-p | 0.95 |
| top-k | 20 |
| min-p | 0.0 |
| presence penalty | 0.0 |
| repeat penalty | 1.0 |
| KV mode | q8 |

Explicit OpenAI request fields still win. A third-party client that sends its
own sampling values overrides these registry values; omit those fields or
configure the client to match the table when exact sampling behavior matters.

## Serve it

```bash
hipfire serve qwen3.6:35b-a3b-mq4r 127.0.0.1:11435 -d
curl http://127.0.0.1:11435/v1/models
```

Client settings:

| Field | Value |
|---|---|
| OpenAI base URL | `http://127.0.0.1:11435/v1` |
| API key | Any non-empty placeholder if the client requires one |
| Model | `qwen3.6:35b-a3b-mq4r` |
| API shape | OpenAI Chat Completions |

The server has no built-in authentication or TLS. Keep it on loopback, or put
an authenticated TLS reverse proxy in front of it before remote access.

### Generic OpenAI-compatible client

Any client that can target an OpenAI Chat Completions base URL can use the
endpoint and model above. A minimal request:

```bash
curl http://127.0.0.1:11435/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer placeholder-key' \
  -d '{"model":"qwen3.6:35b-a3b-mq4r","messages":[{"role":"user","content":"Say hello."}],"max_tokens":64}'
```

If the client asks for a context window or maximum output length, set them to
values no larger than the effective `max_seq` Hipfire loaded; do not advertise
a larger context merely to satisfy a client default. Hipfire supports
streaming, usage on stream end, system and developer roles, tools,
`reasoning_effort`, `max_tokens`, and the registry sampling fallback described
in [`SERVE.md`](SERVE.md).

## What this says about board variance

A few percent of throughput spread alone is not evidence of silicon lottery.
Automatic clocks, thermal and power state, ROCm/compiler builds, host load,
container policy, and cold-start convergence can move results. A faster HIP
control paired with a slower retained-PM4 arm especially points away from a
uniformly faster or slower chip.

The golden runner makes that distinction inspectable:

- a different tape/prepared identity is a software or route mismatch;
- a matching identity below the stationary floor is an environment, runtime,
  or hardware investigation;
- a matching identity and passing floor is a reproduction even when the final
  decimal differs from the reference board.
