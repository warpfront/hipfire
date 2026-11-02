#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Publish the PeaceMaker sub-project to its standalone public repository.
#
#   scripts/mirror-push.sh peacemaker
#
# Builds the history-preserving split with scripts/mirror-split.sh into a fresh
# temporary directory, runs every gate against it, and only then force-pushes
# the split's HEAD to branch `main` of the public repository. Any failing gate
# aborts before the push. The source checkout is only read, never modified.
#
# Gates (all must pass):
#   1. cargo check --workspace and cargo test --workspace --no-run, run
#      standalone inside the split.
#   2. Mailmap: every author and committer identity in the split history is
#      already normalized by scripts/mirror.mailmap, and the tip commit's
#      author and committer are both "Kaden Schutt <kaden@hipfire.dev>".
#   3. External guard run over every commit delta of the split history. The
#      root commit is checked against an unrelated, empty synthetic commit.
#
# Guard interface: the executable named by OPPO_GUARD (or `oppo-guard.sh` found
# on PATH when OPPO_GUARD is unset) is run inside the split as
#
#   <guard> <base-commit> <commit>
#
# once per (parent, commit) edge, and once per root commit with the empty
# synthetic commit as <base-commit>. A non-zero exit refuses the push. A guard
# that cannot be found or executed also refuses the push.
#
# Environment:
#   OPPO_GUARD            path of the guard executable (default: oppo-guard.sh on PATH)
#   MIRROR_PUSH_REMOTE    optional; exactly git@github.com:warpfront/peacemaker.git
#                         (default) or https://github.com/warpfront/peacemaker.git;
#                         HTTPS uses the caller's own git/gh credential setup
#   MIRROR_SOURCE_REV, MIRROR_FILTER_REPO  as for scripts/mirror-split.sh
# Requires: everything scripts/mirror-split.sh requires, plus cargo.
set -euo pipefail

ssh_remote="git@github.com:warpfront/peacemaker.git"
https_remote="https://github.com/warpfront/peacemaker.git"
want_identity="Kaden Schutt <kaden@hipfire.dev>"

die() {
    echo "mirror-push: $*" >&2
    exit 1
}

[[ $# -eq 1 && $1 == peacemaker ]] || die "usage: scripts/mirror-push.sh peacemaker"

remote="$ssh_remote"
if [[ -v MIRROR_PUSH_REMOTE ]]; then
    remote="$MIRROR_PUSH_REMOTE"
fi
[[ $remote == "$ssh_remote" || $remote == "$https_remote" ]] ||
    die "MIRROR_PUSH_REMOTE must be exactly $ssh_remote or $https_remote"

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mailmap="$here/mirror.mailmap"
[[ -f $mailmap ]] || die "missing $mailmap"

if [[ -v OPPO_GUARD ]]; then
    guard="$OPPO_GUARD"
else
    guard=$(command -v oppo-guard.sh) || die "oppo-guard.sh not found on PATH and OPPO_GUARD is unset"
fi
[[ -n $guard ]] || die "OPPO_GUARD is empty"
if [[ $guard != */* ]]; then
    guard=$(command -v -- "$guard") || die "guard '$guard' not found"
fi
[[ -f $guard && -x $guard ]] || die "guard '$guard' is not an executable file"
guard=$(cd -- "$(dirname -- "$guard")" && pwd)/$(basename -- "$guard")

tmp=$(mktemp -d)
cleanup() {
    rm -rf -- "$tmp"
}
trap cleanup EXIT
split="$tmp/peacemaker"

echo "mirror-push: splitting into temporary directory"
"$here/mirror-split.sh" peacemaker "$split"

[[ -d $split/.git ]] || die "split produced no repository"
[[ $(git -C "$split" symbolic-ref --short HEAD) == main ]] || die "split HEAD is not branch main"

echo "mirror-push: gate: cargo check --workspace"
(cd -- "$split" && cargo check --workspace) || die "cargo check --workspace failed; not pushing"

echo "mirror-push: gate: cargo test --workspace --no-run"
(cd -- "$split" && cargo test --workspace --no-run) || die "cargo test --workspace --no-run failed; not pushing"

echo "mirror-push: gate: mailmap normalization"
identities="$tmp/identities"
normalized="$tmp/normalized"
git -C "$split" log --format='%an <%ae>%n%cn <%ce>' HEAD >"$identities"
[[ -s $identities ]] || die "split history has no commits; not pushing"
git -C "$split" -c "mailmap.file=$mailmap" check-mailmap --stdin <"$identities" >"$normalized" ||
    die "git check-mailmap failed; not pushing"
if ! cmp -s -- "$identities" "$normalized"; then
    echo "mirror-push: identities still requiring normalization:" >&2
    diff -- "$identities" "$normalized" >&2 || true
    die "mailmap check failed; not pushing"
fi
tip_identity=$(git -C "$split" log -1 --format='%an <%ae>%n%cn <%ce>' HEAD)
[[ $tip_identity == "$want_identity"$'\n'"$want_identity" ]] ||
    die "tip author/committer is not $want_identity; not pushing"

echo "mirror-push: gate: guard over every commit delta"
empty_tree=$(git -C "$split" hash-object -t tree -w --stdin </dev/null)
empty_base=$(
    GIT_AUTHOR_NAME=mirror-push GIT_AUTHOR_EMAIL=mirror-push@invalid \
    GIT_COMMITTER_NAME=mirror-push GIT_COMMITTER_EMAIL=mirror-push@invalid \
    GIT_AUTHOR_DATE="1970-01-01T00:00:00Z" GIT_COMMITTER_DATE="1970-01-01T00:00:00Z" \
        git -C "$split" commit-tree -m "synthetic empty baseline" "$empty_tree"
)
checked=0
commits="$tmp/commits"
git -C "$split" rev-list --parents --reverse HEAD >"$commits"
while read -r commit parents; do
    if [[ -z $parents ]]; then
        parents="$empty_base"
    fi
    for parent in $parents; do
        (cd -- "$split" && "$guard" "$parent" "$commit") ||
            die "guard rejected ${parent:0:12}..${commit:0:12}; not pushing"
        checked=$((checked + 1))
    done
done <"$commits"
[[ $checked -gt 0 ]] || die "guard checked no commit deltas; not pushing"
echo "mirror-push: guard passed $checked commit delta(s)"

echo "mirror-push: all gates passed; pushing HEAD to main of $remote"
git -C "$split" push --force "$remote" HEAD:main
echo "mirror-push: pushed peacemaker to $remote (branch main)"
