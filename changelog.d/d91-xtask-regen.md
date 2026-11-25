### 0.5.0 consolidation

- **Changed (contributors): `scripts/regen-generated.sh` is replaced by `cargo xtask regen [--all|--staged]`.** Same generators, same outputs, same modes; the pre-commit hook calls it. `crates/xtask` is repository tooling (no workspace or third-party dependency, not in the release archive). It takes the host build slot once for the whole run and never asks for a second one from inside a held slot (the 2026-11-24 re-entry deadlock is its regression test).
