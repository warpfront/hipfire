<!-- SPDX-License-Identifier: Apache-2.0 -->
# Canonical PeaceMaker mirror

[hipfire](https://github.com/warpfront/hipfire) remains canonical. Publish its
committed crate state locally to
[PeaceMaker](https://github.com/warpfront/peacemaker) `main` using
`scripts/mirror-push.sh peacemaker`. There is no Actions publishing workflow,
no deploy-key secret, and no repository creation step: the public repository
already exists. Submit canonical crate changes to hipfire, not to the mirror.

## Split contents and attribution

The mirror contains `peacemaker-ir`, `peacemaker-lift`, `peacemaker-author`,
`peacemaker-npu`, `peacemaker-isa`, and `peacemaker-rip`, plus their minimal
standalone dependency closure. `radiowave` is self-contained and included for
the ISA `toolchain` feature. The runtime-dependent `dflash_accept_emu` and
`qwen4_ep_owner_assign_pm` integration tests live in `rdna-compute/tests`, not
ISA. ISA has no dev dependency on the runtime, so the mirror does not pull it
in. Shared ROCm support and emitted kernel fixtures, including the EP assets,
are retained. Source SPDX headers are unchanged.

A generated PeaceMaker-specific LICENSE points to LICENSE-APACHE,
LICENSE-MIT and LICENSE-LLVM-EXCEPTION and to hipfire's canonical governance
documents. NOTICE is preserved. hipfire lacks the full LLVM exception text;
the mirror generates the standard LLVM Project exception.

The split preserves retained revisions, dates, and ancestry. Historical crate
paths map to their new names. Mixed-purpose monorepo messages become neutral
canonical source-commit references. `scripts/mirror.mailmap` normalizes author
and committer metadata on every run; generated commits use
Kaden Schutt <kaden@hipfire.dev>. A deterministic final commit adds the
workspace, pruned lockfile, README and licenses. Unrelated hipfire root
Cargo.toml/README history is not copied. Existing source and artifact
provenance is copied faithfully; generated content has no private workstation
provenance.

## Prerequisites

Use a committed checkout with full history, Python 3.11+, Cargo and
`git-filter-repo==2.47.0` available to the split script. If the filter is not on
PATH, set `MIRROR_FILTER_REPO` to its executable. `MIRROR_SOURCE_REV` optionally
selects a source commit; otherwise HEAD is used. Source files are never edited.

The push requires an external publication guard executable. Put
`oppo-guard.sh` on PATH or set `OPPO_GUARD` to its executable. Missing guards
are fatal, not bypassed. Auth uses the caller's Git credentials:

- Default remote: `git@github.com:warpfront/peacemaker.git` (SSH).
- Optional `MIRROR_PUSH_REMOTE=https://github.com/warpfront/peacemaker.git`;
  configure the caller's GitHub authentication with `gh auth setup-git`.

No other repository URL or embedded credentials are accepted.

## Preview without publishing

```bash
scripts/mirror-split.sh peacemaker "$CUT_ROOT/mirror-cut/peacemaker"
cargo check --workspace --manifest-path "$CUT_ROOT/mirror-cut/peacemaker/Cargo.toml"
cargo test --workspace --no-run --manifest-path "$CUT_ROOT/mirror-cut/peacemaker/Cargo.toml"
```

The optional destination defaults to `mirror-preview/peacemaker`. Existing
destinations are refused. The split script never pushes. The operator supplies
`CUT_ROOT` as an external scratch directory.

## Release cut and local publication

After the NPU and EP merges and before the release tag, inside the dedicated
`chore/peacemaker-split` worktree:

```bash
git fetch origin
git rebase origin/beta
# Preserve merged implementations while migrating new crate references.
cargo check --workspace --locked
cargo test -p peacemaker-isa -p peacemaker-rip --no-run --locked
```

Released CHANGELOG sections are immutable: keep their historical crate names
and exact tagged text. New naming belongs only in Unreleased.

CPU-only merge gate specs are
`-p peacemaker-npu --lib;-p peacemaker-isa --lib;-p peacemaker-rip --lib;-p railgun-cert every_recording_predicate_in_the_tree_has_an_inventory_row;-p npu-tools;-p hipfire-arch-qwen4;-p rdna-compute;-p peacemaker-ir;-p peacemaker-lift;-p peacemaker-author;-p radiowave`.
The operator's merge gate rebases, regenerates maps, runs documentation
ratchets and these tests, and pushes beta. It does not publish the mirror.

After the branch has merged, use the committed beta checkout:

```bash
scripts/mirror-push.sh peacemaker
```

The push script makes a temporary split, runs `cargo check --workspace` and
`cargo test --workspace --no-run`, verifies author/committer normalization and
the generated commit identity, and runs the publication guard over every
filtered-history parent/commit edge. Any failure stops publication. Only then
it force-pushes `HEAD:main` to PeaceMaker. No other branch or release tag is
pushed. Temporary files are removed on exit. The force update is intentional:
hipfire remains canonical, including its history and mailmap changes.
