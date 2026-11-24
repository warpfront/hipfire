# hipfire-daemon — map

> **Status:** `production` (vocabulary owned by [`docs/GLOSSARY.md`](../../docs/GLOSSARY.md)).
> **Layer:** Product binary — top of the DAG, above `hipfire-generate`; `[[bin]] name = "daemon"`; see the layering table in [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md).

## Purpose

The product entry point (`crates/hipfire-daemon/src/main.rs`). Owns message dispatch and process lifetime — `hardware.devices` resolution with machine-wide per-card lock files under the resolved lock directory (`src/gpu_lock.rs`: `flock(2)` on Unix, `LockFileEx` on Windows), JSONL stdin/stdout protocol (`load`/`generate`/`unload`, `img_generate`, and the embedding verb `embed` — which carries image bytes too, arch 17/18), and the `generate(…)` ladder that delegates to `hipfire-loader::load_model` and the `hipfire-generate` bodies. Build is `cargo build --release` with no `--features`.

## Gotchas

- Daemons may share a HOME only when their physical GPU sets do not overlap. Lock files are keyed by resolved identity: `gpu-GPU-<uuid>.lock`, or `gpu-pci-<dddd:bb:dd.f>.lock` for cards without a UUID. With `hardware.devices`, `hipfire_config::devices` resolves the list from the KFD topology and the daemon takes each card's lock (non-blocking) inside that step, so a `gfxNNNN` entry skips busy cards; after HIP loads, every logical device's arch and PCI bus ID must match or startup aborts with both tables. Without it, the HIP-visible cards are mapped to KFD by PCI address and locked sorted, all-or-nothing. Hosts without KFD (WSL2/ROCDXG: `/dev/dxg` and no `/dev/kfd`; native Windows) identify cards from HIP's UUID and PCI bus ID instead and refuse `hardware.devices`. Contention reports the identity, PCI BDF and holder PID. Never unlink a GPU lock file: the OS lock releases on death, and unlinking creates a second inode.
- Lock directory: `HIPFIRE_LOCK_DIR` if set (must be absolute and writable); otherwise `/run/lock/hipfire` when writable, then `/tmp/hipfire-locks` (Windows: `%ProgramData%\hipfire\locks`, then `%TEMP%\hipfire-locks`). The daemon logs its resolved directory at startup. Daemons or scripts pointed at **different** lock directories do not contend; all users/processes sharing a card must use the same directory. `gpu-lock.sh` follows the identical selection policy.
- WSL2/ROCDXG and native Windows are uncertified for Redline PM4: the retained default falls back to the HIP graph with one log line and an explicit `replay.backend=redline|shadow` exits at startup (WSL override: `HIPFIRE_UNSAFE_WSL_REDLINE=1`, unsafe until certified).
- `~/.hipfire/daemon-<identity>[_<identity>...].pid` (identity = `GPU-<uuid>` or `pci-<bdf>`) is advisory PID discovery for uninstall, not a mutex. Native CLI/client processes communicate with their own stdio daemon; serve discovery uses `serve.pid`/its HTTP port. A private HOME does **not** bypass GPU contention.
- The binary name is `daemon` (`[[bin]]` in `Cargo.toml`), not `hipfire-daemon`; the `target/release/hipfire` wrapper is separate.
- The serve vs one-shot decision lives in `hipfire-cli` (forced local when `HIPFIRE_LOCAL=1`, `--kv-mode`, `--json`, or `--no-stream`) — daemon always assumes it was given the correct path.
- The `embed` arm carries media exactly like the generate routes: the daemon refuses a media-carrying embed when `vision_mode=off` suppressed the tower sidecar (`refuse_media_without_tower`, naming the tower the caller asked for — vision or audio — plus the knob and the sidecar), while *whether a tower is resident* is decided in `hipfire_generate::embed` — `main.rs` must keep naming no arch type (`bundle.vision` / `bundle.audio` are behind typed accessors, and the tower-less refusal texts are the arch crate's `NO_VISION_TOWER` / `NO_AUDIO_TOWER`).

## Crate map

<!-- crate-map:generated:begin -->

Modules, exported symbols and dependencies are generated, not committed: see
`map.generated.md` in this directory (`cargo xtask regen` writes it).

<!-- crate-map:generated:end -->
