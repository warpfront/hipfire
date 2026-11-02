# Containerized hipfire

Docker and podman compose stack for `hipfire serve` on AMD GPUs: an init
service pulls the model into a named volume, the engine serves the OpenAI-style
API on port 11435, and optional profiles add monitoring and Strix Halo host
tuning. Needs `/dev/kfd` and `/dev/dri` on the host and a recent kernel
amdgpu driver; no ROCm install on the host.

## Quickstart

From the repo root:

```bash
docker compose up -d          # or: podman compose up -d
docker compose logs -f hipfire-init hipfire
```

The first start builds the image (until a published image exists), then
`hipfire-init` pulls the model into the `hipfire-models` volume (default
`qwen3.8:27b-mq4-xts`: 15.0 GB of weights plus its MTP, DFlash draft and
vision sidecars, about 17.4 GB in all). The engine starts when the pull
finishes and loads the model before it reports healthy.

```bash
curl http://127.0.0.1:11435/health
curl http://127.0.0.1:11435/v1/models
curl http://127.0.0.1:11435/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8:27b-mq4-xts","messages":[{"role":"user","content":"Hello"}]}'
```

Settings live in `.env` (every variable also has a compose default, so the
file is optional):

```bash
cp .env.example .env
```

| Variable | Default | Role |
|---|---|---|
| `HIPFIRE_MODEL` | `qwen3.8:27b-mq4-xts` | Model tag pulled by `hipfire-init` and pre-warmed by serve |
| `HIPFIRE_PORT` | `11435` | Host port |
| `HIPFIRE_BIND` | `127.0.0.1` | Host address the port is published on |
| `HIPFIRE_UI` | `0` | `1` serves the web UI at `/ui` |
| `HIPFIRE_SERVE_ARGS` | empty | Extra `hipfire serve` arguments (for example `--api-key-file …`) |
| `HIPFIRE_MODELS_PATH` | `hipfire-models` | Named volume, or a host directory to reuse existing models |
| `HIPFIRE_IMAGE` / `HIPFIRE_IMAGE_TARGET` | `ghcr.io/warpfront/hipfire:latest` / `full` | Image tag and build target |
| `HF_TOKEN` | unset | Optional Hugging Face token for the pull |
| `COMPOSE_FILE` | unset | Set to `compose.yml:compose.halo.yml` for the Halo override |

The UI is a `.env` switch rather than a compose profile because `--profile X`
replaces `COMPOSE_PROFILES`, so a profile-gated engine would break
`--profile monitoring`.

Models are never baked into the image. Config and the kernel cache persist in
the `hipfire-config` and `hipfire-kernels` volumes.

## Per-GPU notes

| GPU | Arch | Notes |
|---|---|---|
| Strix Halo | gfx1151 | Enable `compose.halo.yml` for the NPU (`/dev/accel` plus memlock). Flagship `qwen3.8:flash-next` is about 125 GB. |
| Radeon AI PRO R9700 | gfx1201 | 32 GB; the default tag fits. |
| RX 7900 XTX | gfx1100 | 24 GB; the default tag fits. |

Prebuilt kernels for gfx1100, gfx1151 and gfx1201 are baked into the image, so
the first run loads packaged kernels instead of compiling them. A route whose
kernel is not in the packs yet compiles once into the `hipfire-kernels` cache
(the default `full` image carries hipcc). Measured on Strix Halo with the slim
`runtime` image before this fix, the default tag failed its first chat request
because `tape_copy_batched` was not in the packs. That module and the
lane-batched verify conv `conv1d_silu_split_f32_lanes` are now registered in
`kernel_registry`, so packs rebuilt from this source carry them; the slim image
has not been re-measured, and `full` stays the default.
Other arches need a build with the `HIPFIRE_GPU_ARCHES` build arg.

Multi-GPU host: export `HIPFIRE_DEVICES` or `ROCR_VISIBLE_DEVICES` in the shell
before `docker compose up`; both are passed through to the containers.

Strix Halo NPU:

```bash
echo 'COMPOSE_FILE=compose.yml:compose.halo.yml' >> .env
docker compose up -d
```

`doctor --npu` fails its `gpu_free` check while a serve holds the GPU, so stop
the engine first:

```bash
docker compose stop hipfire
docker compose run --rm --no-deps hipfire doctor --npu
```

## Images

| Target | Contents | Size on disk / download |
|---|---|---|
| `full` (default) | ROCm 10.1 runtime plus hipcc, LLVM and HIP headers, so a route without a prebuilt kernel compiles it once | 4.29 GB / 713 MB |
| `runtime` | ubuntu 26.04, the HIP runtime library closure only (no compiler), daemon, CLI, baked kernels | 781 MB / 192 MB |

Measured with docker 29 on x86-64 (download = compressed layers). The previous
single image (ROCm 7.2.4 dev base) was 5.44 GB / 1.27 GB.

Both use ROCm 10.1, the same toolchain that builds the release kernel packs.
`HIPFIRE_GPU_ARCHES` (default `gfx1100 gfx1151 gfx1201`) selects which arches
are baked.

```bash
docker build -f Containerfile --target full    -t hipfire:full .
docker build -f Containerfile --target runtime -t hipfire .
HIPFIRE_IMAGE_TARGET=runtime docker compose up -d --build
docker build -f Containerfile --build-arg HIPFIRE_GPU_ARCHES="gfx1030" --target full -t hipfire:full .
```

Without hipcc, a kernel module missing from the baked pack is a hard error, not
a fallback, so `runtime` only suits models whose routes the packs fully cover
(measured before the registry fix: `qwen3.5:0.8b` yes; `qwen3.8:27b-mq4-xts` no,
on gfx1151 its draft route needed `tape_copy_batched`, now registered and
awaiting a pack rebuild and re-measurement).

Non-root: the entrypoint starts as root only to add the groups that own
`/dev/kfd`, `/dev/dri` and `/dev/accel`, then drops to user `hipfire`
(uid 1000) with `setpriv`. Rootless podman keeps the invoking user's groups
through the `run.oci.keep_original_groups` annotation set in `compose.yml`.

Host model directory (`HIPFIRE_MODELS_PATH=/path/to/models`): rootless podman
maps container root to your host user and uid 1000 to a subordinate id that owns
none of your files, so `hipfire` cannot write a directory you own (the pull of a
missing sidecar fails with `Permission denied`). When the entrypoint sees a
rootless user namespace and the models directory is not writable by `hipfire`,
it keeps the namespace's root instead of dropping (it logs one line saying so);
that root is exactly your host user, and files the pull writes are owned by you.
No `chown`, `chmod` or `podman unshare` is needed. Rootful docker still drops,
so a bind-mounted directory there must be writable by uid 1000
(`chown 1000:1000 /path/to/models`) or be a named volume.

## Profiles

### `monitoring`

```bash
docker compose --profile monitoring up -d
```

| Service | Address | Role |
|---|---|---|
| `prometheus` | `127.0.0.1:9090` | Scrapes hipfire `/metrics` every 5 s |
| `node-exporter` | internal | GPU memory (VRAM and GTT) and host memory from sysfs |
| `grafana` | `127.0.0.1:3000` | Provisioned hipfire dashboard |

The dashboard shows decode and prefill tok/s, TTFT, queue depth, GPU memory
(VRAM on discrete cards, GTT on Strix Halo) and speculative acceptance
(`hipfire_spec_tau`). Ports are `PROMETHEUS_PORT` and `GRAFANA_PORT`.

### `halo-tune`

An explicit opt-in, privileged one-shot that applies host power settings for
Strix Halo. It is never started by `docker compose up`.

It puts the host in the power state hipfire's published Strix Halo numbers are
measured in, changing only values that differ and logging each as
`<path>: <before> -> <after>`:

| Setting | Change |
|---|---|
| ACPI platform profile (`/sys/firmware/acpi/platform_profile`) | to `performance`, when the firmware lists it. System-wide: raises package power limits, fan noise and power draw |
| GPU `power_dpm_force_performance_level` | to `auto`; a `manual` level (another tool's clock pin) is left alone. `HIPFIRE_HALO_PERF_LEVEL=high` pins the top GPU DPM level instead (optional; not what published numbers use) |
| GPU `pp_power_profile_mode` | from `POWER_SAVING` to `BOOTUP_DEFAULT` |
| GPU hwmon `power1_cap` | up to `power1_cap_default` when below it |

Attributes the machine does not expose are reported and skipped; some boxes
expose only the performance level. It never touches BIOS or SMU limits or FCLK.
The original values are saved first (`halo-tune-state` volume) for `restore`,
and every change is lost on reboot.

```bash
docker compose run --rm halo-tune apply --dry-run   # show what would change
docker compose run --rm halo-tune apply
docker compose run --rm halo-tune status
docker compose run --rm halo-tune restore           # put the saved values back
```

Rootless podman cannot write the host sysfs from a user-namespaced container.
Run the same script on the host instead:

```bash
sudo docker/hipfire-halo-tune.sh apply
```

## Security

- There is no TLS. The default publish is loopback only (`HIPFIRE_BIND=127.0.0.1`).
- To serve the LAN set `HIPFIRE_BIND=0.0.0.0` **and** pass an API key file, for
  example `HIPFIRE_SERVE_ARGS=--api-key-file /home/hipfire/.hipfire/api-key`,
  or put a TLS reverse proxy in front. See
  [SERVE.md](SERVE.md#security-bearer-authentication-and-https-proxies).
- Grafana ships with the default admin password from `GRAFANA_ADMIN_PASSWORD`
  in `.env.example`. Change it before exposing port 3000.

## Troubleshooting

- **`permission denied` on `/dev/kfd`:** the user needs the `render` group on the
  host. Rootless podman relies on the keep-groups annotation; check that your
  podman honors it. `HIPFIRE_RUN_AS_ROOT=1` skips the privilege drop and is a last resort.
- **`Permission denied` creating a model file under a host `HIPFIRE_MODELS_PATH`:**
  rootful docker: `chown 1000:1000` the directory (or use the default named
  volume). Rootless podman handles this itself, see Non-root above; if it
  persists, the image predates that entrypoint, rebuild it.
- **`docker compose` not found:** install the compose plugin. `podman compose`
  needs a compose provider (`docker-compose` or `podman-compose`).
- **Model pull fails:** check network access and, for gated repos, `HF_TOKEN`.
  Read `docker compose logs hipfire-init`; the pull is idempotent, so the next
  `up` retries it and skips files that already verify.
- **Unhealthy while loading:** the healthcheck allows a 15 minute `start_period`.
  `/health` reports `loading_model` with the tag while a load is in progress.
- **Packaged kernel missing on the `runtime` image** (`no usable cached kernel
  image exists, but hipcc is unavailable`): use the default `full` target.
- **NPU not used:** enable `compose.halo.yml`, confirm the memlock limit, and run
  `doctor --npu` as above.
- **Port in use:** change `HIPFIRE_PORT`.

## Gate runner (`scripts/container-gate.sh`)

The script builds the `gate-runner` target (full builder tree plus source) and
runs one command in it with GPU passthrough and the host models directory
mounted. It has no default gate: with no argument it prints guidance and exits
`2`; pick a route from [VALIDATION.md](VALIDATION.md). A successful image build
does not certify GPU correctness or perf.

```bash
scripts/container-gate.sh -lc 'exec python3 scripts/serve_harness.py --mode chain --model /root/.hipfire/models/<file> --tag qwen3.5:9b'
```

Environment: `HIPFIRE_CONTAINER` (`podman` or `docker`), `HIPFIRE_MODELS_DIR`,
`HIPFIRE_IMAGE`, `HIPFIRE_SKIP_BUILD=1`, `HIPFIRE_GPU_LOCKFILE`.

Embedding encoders (arch 17 / arch 18 routes in [VALIDATION.md](VALIDATION.md))
take a checkpoint *directory*, so mount it through the same models dir. The
helper's only mount is `<models dir>:/root/.hipfire/models`, so everything a
harness reads must live under it — including the arch-18 `.vl` tower pack
(`ensure_vision_sidecar` finds the pack beside the checkpoint and then needs no
`--quantizer`) and the reference `.npz`. A farm of hard links into the harness's
HF snapshot works and keeps the cache out of the container:

```bash
SNAP=…   # HF snapshot dir
mkdir -p ~/.hipfire/models/farm/$NAME
for e in "$SNAP"*; do ln "$(readlink -f "$e")" ~/.hipfire/models/farm/$NAME/; done
cp <ref>.npz ~/.hipfire/models/
# the tower pack must be a sibling of the DIRECTORY: ~/.hipfire/models/farm/$NAME.vl

HIPFIRE_MODELS_DIR=~/.hipfire/models HIPFIRE_SKIP_BUILD=1 \
  scripts/container-gate.sh bash -lc 'set -e; python3 -m pip install --quiet --break-system-packages pillow soundfile;
    exec python3 scripts/check_embeddinggemma2_parity.py \
      --model /root/.hipfire/models/farm/$NAME --ref /root/.hipfire/models/<ref>.npz \
      --daemon target/release/daemon'
```

`pillow` (image/video frames) and `soundfile` (WAV) are not baked into
`gate-runner`; install them in the same shell. `hipfire-gate`'s `ENTRYPOINT` is
empty (its `CMD` is the no-gate guidance, run in `/hipfire`), so name the shell
explicitly: `bash -lc '<cmd>'`. A bare `-lc '<cmd>'` is taken as the executable
and fails with `exec: "-lc": executable file not found in $PATH` (verified with
docker 29.9; the `-lc` example above predates this one and fails the same
way). `HIPFIRE_CONTAINER=docker` picks docker over the default podman (docker
treats podman's `keep-groups` as a literal group name, so the wrapper only adds
it for podman). Use `HIPFIRE_SKIP_BUILD=1` once the `gate-runner` tag is
current.

## Publishing status

A draft workflow, `.github/workflows/container.yml`, exists. It runs on manual
dispatch only and has image push disabled pending approval. No image is
published yet, so `docker compose up` builds locally.

Related: [SERVE.md](SERVE.md) (HTTP API), [NIXOS.md](NIXOS.md) (NixOS install).
