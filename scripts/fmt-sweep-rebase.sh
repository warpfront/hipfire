#!/usr/bin/env bash
# Replay a branch cut before the one-time 0.4.3 workspace rustfmt sweep onto the
# sweep commit S. A plain rebase across S conflicts on every hunk the branch
# shares with it; this instead replays each commit as sweep(tree(commit)), so
# the result is S plus the branch's own changes, formatted exactly as the sweep.
#
# Usage:
#   scripts/fmt-sweep-rebase.sh SWEEP_SHA [UPSTREAM]   # inside the owner's worktree
#   scripts/fmt-sweep-rebase.sh --sweep-tree DIR       # the transform itself
#
# TWO-STAGE: this tool produces fmt-swept/<branch>, which sits on the sweep
# commit S ONLY. It is NOT a final or landable state: it lacks every commit after
# S on UPSTREAM (the later land-gate/helper commit and newer product commits).
# The owner must then run an ordinary `git rebase UPSTREAM` onto the current tip,
# resolve the real conflicts, and rerun its canonical CPU acceptance before
# declaring the branch READY.
#
# SWEEP_SHA mode (UPSTREAM defaults to origin/beta). Fetching UPSTREAM (only when it
# names a configured remote) updates remote-tracking refs BEFORE validation runs; it is
# the one thing this tool changes outside scratch. The owner's branch, worktree and
# index are NEVER touched (no reset, no backup refs). Every check below runs before
# any ref is created:
#   1. Preconditions: on a branch; no staged/unstaged/untracked changes; no
#      rebase/merge/cherry-pick/revert/sequencer in progress; SWEEP_SHA is a
#      commit reachable from UPSTREAM, subject exactly the sweep subject, diff vs
#      its parent touches only .rs files; no merge commits in
#      merge-base(HEAD, S)..HEAD. If HEAD already contains S: exit 0.
#   2. Toolchain binding: in a scratch worktree, sweep(P = S^) must equal S's tree
#      byte for byte, else the local rustfmt/toolchain does not reproduce the sweep.
#   3. If P is not an ancestor of HEAD, `git rebase P` runs in the SCRATCH worktree
#      on a temporary branch (conflict: abort, report paths, exit 1).
#   4. Each commit C in P..tip (oldest first) becomes sweep(tree(C)) on top of S,
#      with C's message and author name/email/date; the committer is the local git
#      config. A commit that is empty over the sweep is dropped.
#   5. tip tree == sweep(tree(tip)) and `cargo-fmt --all -- --check` is clean.
#   6. Creates refs/heads/fmt-swept/<branch> (create-only). Rerun with an identical
#      replay (same ancestry on S, trees, full messages, author name/email/date)
#      prints "already prepared"; any other existing ref is an error and is left in
#      place: the owner deletes it explicitly.
#   7. The scratch worktree and temporary branch are removed on every exit path.
#
# --sweep-tree DIR: apply the transform to the git work tree DIR in place. The ONLY
#   writer is CI's per-file command (scripts/ci-rustfmt-changed.sh: rustfmt with the
#   owning crate's edition and skip_children=true, so it formats exactly the named file
#   and never follows `mod` declarations or #[path] attributes), run on every
#   non-excluded tracked .rs file that fails the per-file check, repeated per file until
#   its check is clean (max 4 passes: rustfmt is not idempotent on a few files).
#   `cargo-fmt --all -- --check` then runs read-only and must be clean: it is the
#   independent proof that the per-file result is the workspace fixpoint. cargo-fmt
#   never writes. Any rustfmt error, enumeration/worker failure, non-convergence or
#   unclean final check is exit 1. Fails closed, before any formatter runs, on a
#   tracked symlink that is a .rs file, points at a directory or escapes the tree.
#   Left as is: crates/peacemaker-npu/src/isa/{gen,sched}.rs (generated), trybuild
#   compile_fail fixtures (foo.rs with a sibling foo.stderr) and docs/**.
set -euo pipefail

SWEEP_SUBJECT='style: rustfmt the whole workspace (one-time 0.4.3 sweep)'
self=$(readlink -f "${BASH_SOURCE[0]}")
export ROCR_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 HIPFIRE_DEVICES=-1
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-16}"

die() { echo "fmt-sweep-rebase: $*" >&2; exit 1; }
warn() { echo "fmt-sweep-rebase: $*" >&2; }
short() { git rev-parse --short "$1"; }

usage() {
  cat >&2 <<'EOF'
usage: fmt-sweep-rebase.sh SWEEP_SHA [UPSTREAM]   (UPSTREAM default: origin/beta)
       fmt-sweep-rebase.sh --sweep-tree DIR

Replays the current branch onto the one-time rustfmt sweep commit SWEEP_SHA and
creates fmt-swept/<branch> at the result. The owner's branch, worktree and index are
never touched (a fetch of UPSTREAM may update remote-tracking refs first).
fmt-swept/<branch> sits on the sweep commit ONLY and is not a landable state: next
run `git rebase UPSTREAM`, resolve real conflicts, rerun the canonical CPU acceptance.
EOF
}

excluded() {
  case "$1" in
    crates/peacemaker-npu/src/isa/gen.rs|crates/peacemaker-npu/src/isa/sched.rs|docs/*) return 0 ;;
  esac
  [[ -f "${1%.rs}.stderr" ]]
}

tmp=""
# ---------------------------------------------------------------- transform
sweep_tree() {
  local dir=$1 top rec f path target resolved
  tmp=""
  [[ -d "$dir" ]] || die "--sweep-tree: no such directory: $dir"
  cd "$dir"
  top=$(git rev-parse --show-toplevel 2>/dev/null) || die "--sweep-tree: $dir is not a git work tree"
  [[ "$(cd "$top" && pwd -P)" == "$(pwd -P)" ]] || die "--sweep-tree: $dir is not the top of a git work tree"
  top=$(pwd -P)
  tmp=$(mktemp -d "${TMPDIR:-/tmp}/fmt-sweep-tree.XXXXXX") || die "cannot create a temporary directory"
  trap 'rm -rf "$tmp"' EXIT

  # ---- containment: the only writer is per-file rustfmt on tracked paths (skip_children:
  # no module traversal), so the remaining way out of the tree is a symlink. Fail closed on
  # tracked symlinks that are .rs files, directories or escape the tree.
  [[ -f Cargo.toml ]] || die "--sweep-tree: no Cargo.toml at the work tree root (cargo-fmt would search upward, outside it)"
  git ls-files -z -s > "$tmp/stage" || die "git ls-files failed"
  while IFS= read -r -d '' rec; do
    [[ "${rec%% *}" == 120000 ]] || continue
    path=${rec#*$'\t'}
    [[ "$path" != *.rs ]] || die "tracked .rs symlink refused: $path"
    target=$(readlink -- "./$path") || die "cannot read symlink $path"
    if [[ "$target" == /* ]]; then resolved=$(realpath -m -- "$target")
    else resolved=$(realpath -m -- "./$(dirname -- "$path")/$target"); fi || die "cannot resolve symlink $path"
    [[ ! -d "$resolved" ]] || die "tracked directory symlink refused: $path -> $target"
    [[ "$resolved" == "$top" || "$resolved" == "$top"/* ]] || die "tracked symlink escapes the work tree: $path -> $target"
  done < "$tmp/stage"

  sed -n '/^file_edition()/,/^}/p' scripts/ci-rustfmt-changed.sh > "$tmp/fe.sh" 2>/dev/null || true
  [[ -s "$tmp/fe.sh" ]] || die "--sweep-tree: cannot read file_edition from scripts/ci-rustfmt-changed.sh"
  . "$tmp/fe.sh"
  export -f file_edition

  # Sole writer. Operands are ./-prefixed tracked paths (never parsed as options). Each
  # worker checks its files and formats a failing one until its check is clean; a rustfmt
  # error or non-convergence makes the worker exit 1, which xargs reports as nonzero.
  local -a files=()
  git ls-files -z -- '*.rs' > "$tmp/rs" || die "git ls-files failed"
  while IFS= read -r -d '' f; do
    excluded "$f" || files+=("./$f")
  done < "$tmp/rs"
  : > "$tmp/done"
  if [[ "${#files[@]}" -gt 0 ]]; then
    printf '%s\0' "${files[@]}" > "$tmp/files" || die "cannot write the file list"
    nice -n 10 xargs -0 -r -P 16 -n 16 bash -c '
      for f; do
        ed=$(file_edition "$f") || exit 1
        rustfmt --edition "$ed" --check --config skip_children=true "$f" >/dev/null 2>&1 && continue
        for pass in 1 2 3 4; do
          rustfmt --edition "$ed" --config skip_children=true "$f" >/dev/null 2>&1 \
            || { echo "fmt-sweep-rebase: rustfmt failed on $f" >&2; exit 1; }
          rustfmt --edition "$ed" --check --config skip_children=true "$f" >/dev/null 2>&1 && break
          [[ "$pass" -lt 4 ]] || { echo "fmt-sweep-rebase: rustfmt did not converge on $f" >&2; exit 1; }
        done
        printf "%s\n" "$f" || exit 1
      done' _ < "$tmp/files" > "$tmp/done" || die "per-file formatting failed (xargs exit $?; see the rustfmt message above)"
  fi
  sort "$tmp/done" | sed 's|^\./|per-file: |' || die "cannot list formatted files"
  echo "per-file formatted: $(wc -l < "$tmp/done")"

  # Read-only, independent fixpoint proof. cargo-fmt never writes here; failure stops.
  cargo-fmt --all -- --check >/dev/null 2>&1 || die "cargo-fmt --all -- --check is not clean after the per-file pass (no fallback to a cargo-fmt write)"
  echo "sweep: done"
}

# Digest of the replayed commits S..TIP: ancestry on S, and per commit the tree,
# the full message and the author line (name, email, date). Used to decide whether an
# existing fmt-swept/<branch> is exactly this replay. Any failure is an error.
seq_digest() {
  local s=$1 tip=$2 c prev=$1 a b
  git merge-base --is-ancestor "$s" "$tip" || { echo "not-on-sweep"; return 0; }
  git rev-list --reverse "$s..$tip" > "$scratch/digest-list" || return 1
  while IFS= read -r c; do
    [[ "$(git rev-list --parents -n 1 "$c" | cut -d' ' -f2-)" == "$prev" ]] || { echo "nonlinear"; return 0; }
    a=$(git cat-file commit "$c" | sed -n '/^author /{p;q}') || return 1
    b=$(git cat-file commit "$c" | sed '1,/^$/d' | git hash-object --stdin) || return 1
    printf 'tree=%s %s msg=%s\n' "$(git rev-parse "$c^{tree}")" "$a" "$b"
    prev=$c
  done < "$scratch/digest-list"
  echo "end"
}

# ------------------------------------------------------------ scratch state
scratch=""
wt=""
tmp_branch=""
cleanup_done=0

cleanup_scratch() {
  local rc=0 gd
  [[ -n "$scratch" ]] || return 0
  cleanup_done=1
  if [[ -d "$wt" ]]; then
    gd=$(git -C "$wt" rev-parse --absolute-git-dir 2>/dev/null) || gd=""
    if [[ -n "$gd" && -d "$gd/rebase-merge" || -n "$gd" && -d "$gd/rebase-apply" ]]; then
      git -C "$wt" rebase --abort >/dev/null 2>&1 || { warn "cleanup: rebase --abort failed in $wt"; rc=1; }
    fi
    # Index must hold the worktree (formatting left unstaged edits) so that
    # read-tree -u can restore everything and `worktree remove` needs no --force.
    git -C "$wt" add -A >/dev/null 2>&1 || { warn "cleanup: git add -A failed in $wt"; rc=1; }
    git -C "$wt" read-tree -u --reset HEAD >/dev/null 2>&1 || { warn "cleanup: read-tree --reset failed in $wt"; rc=1; }
    git worktree remove "$wt" >/dev/null 2>&1 || { warn "cleanup: could not remove scratch worktree $wt"; rc=1; }
  fi
  if [[ -n "$tmp_branch" ]] && git show-ref --verify --quiet "refs/heads/$tmp_branch"; then
    git branch -D -q "$tmp_branch" >/dev/null 2>&1 || { warn "cleanup: could not delete temporary branch $tmp_branch"; rc=1; }
  fi
  if [[ "$rc" -eq 0 ]]; then
    rm -rf "$scratch" || { warn "cleanup: could not remove $scratch"; rc=1; }
  else
    warn "cleanup incomplete; leftovers under $scratch (git worktree list / git branch)"
  fi
  return "$rc"
}

on_exit() {
  local rc=$?
  trap - EXIT
  set +e
  if [[ "$cleanup_done" -eq 0 ]]; then
    cleanup_scratch || { [[ "$rc" -ne 0 ]] || rc=1; }
  fi
  exit "$rc"
}

# --------------------------------------------------------------- main mode
rebase_main() {
  local sweep_arg=$1 upstream=${2:-origin/beta}
  local top gitdir branch m r S P up_sha p mb head bind_got bind_want tmp_name
  local c tree new_parent last_tree rebased_tip i
  local -a commits=() dropped=() made=()

  scratch=$(mktemp -d "${TMPDIR:-/tmp}/fmt-sweep-rebase.XXXXXX") || die "cannot create a scratch directory"
  trap on_exit EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM

  top=$(git rev-parse --show-toplevel 2>/dev/null) || die "not inside a git work tree"
  cd "$top"
  gitdir=$(git rev-parse --absolute-git-dir)

  # ---- 1. preconditions (no mutation yet)
  branch=$(git symbolic-ref --quiet --short HEAD) || die "detached HEAD; check out the branch first"
  [[ -z "$(git status --porcelain --untracked-files=all)" ]] \
    || die "work tree is not clean (staged, unstaged or untracked changes); commit or stash first"
  for m in rebase-merge rebase-apply MERGE_HEAD CHERRY_PICK_HEAD REVERT_HEAD sequencer; do
    [[ ! -e "$gitdir/$m" ]] || die "a rebase/merge/cherry-pick/revert is in progress ($m); finish or abort it first"
  done
  git var GIT_COMMITTER_IDENT >/dev/null 2>&1 || die "no committer identity (set user.name and user.email)"

  for r in $(git remote); do
    if [[ "$upstream" == "$r"/* ]]; then
      git fetch -q --no-tags "$r" "${upstream#"$r"/}" || die "cannot fetch $upstream"
      break
    fi
  done
  up_sha=$(git rev-parse --verify --quiet "$upstream^{commit}") || die "UPSTREAM does not resolve to a commit: $upstream"
  S=$(git rev-parse --verify --quiet "$sweep_arg^{commit}") || die "SWEEP_SHA does not resolve to a commit: $sweep_arg"
  git merge-base --is-ancestor "$S" "$up_sha" || die "$S is not reachable from $upstream"
  [[ "$(git log -1 --format=%s "$S")" == "$SWEEP_SUBJECT" ]] \
    || die "$S is not the sweep commit (subject must be exactly: $SWEEP_SUBJECT)"
  [[ "$(git rev-list --parents -n 1 "$S" | wc -w)" -eq 2 ]] || die "$S must have exactly one parent"
  P=$(git rev-parse "$S^")
  git diff-tree -r --no-renames --name-only -z "$P" "$S" > "$scratch/sweep-paths" || die "git diff-tree failed for $(short "$S")"
  while IFS= read -r -d '' p; do
    [[ "$p" == *.rs ]] || die "$S touches a non-.rs file ($p); not a pure rustfmt sweep"
  done < "$scratch/sweep-paths"

  head=$(git rev-parse HEAD)
  if git merge-base --is-ancestor "$S" HEAD; then
    echo "fmt-sweep-rebase: $branch already contains the sweep ($(short "$S")); already on the sweep; rebase normally."
    exit 0
  fi
  mb=$(git merge-base HEAD "$S") || die "HEAD and $S share no history"
  [[ -z "$(git rev-list --merges "$mb..HEAD")" ]] \
    || die "merge commits in $(short "$mb")..HEAD; linearize the branch first (this tool replays linear history only)"

  # ---- 2. toolchain binding in a scratch worktree
  wt="$scratch/wt"
  git -c core.hooksPath=/dev/null worktree add -q --detach "$wt" "$P" || die "cannot create scratch worktree"

  bash "$self" --sweep-tree "$wt" >/dev/null || die "sweep transform failed on $(short "$P")"
  git -C "$wt" add -A
  bind_got=$(git -C "$wt" write-tree)
  bind_want=$(git rev-parse "$S^{tree}")
  if [[ "$bind_got" != "$bind_want" ]]; then
    echo "fmt-sweep-rebase: local rustfmt/toolchain does not reproduce the sweep" >&2
    echo "  sweep($(short "$P")) = tree $bind_got, but $(short "$S") has tree $bind_want" >&2
    echo "  local: $(rustfmt --version)" >&2
    exit 1
  fi
  local binding
  binding="OK: sweep($(short "$P")) == tree $(short "$bind_want") of $(short "$S") ($(rustfmt --version))"
  echo "fmt-sweep-rebase: toolchain binding $binding"

  # ---- 3. move onto P in scratch only
  rebased_tip=$head
  if ! git merge-base --is-ancestor "$P" "$head"; then
    echo "fmt-sweep-rebase: $(short "$P") is not an ancestor of $branch; rebasing a scratch copy onto it"
    git -C "$wt" read-tree -u --reset "$P"
    tmp_name="fmt-sweep-tmp-${scratch##*.}"
    # Ownership is recorded only after the branch is actually created, so cleanup
    # can never delete a pre-existing ref of the same name.
    git -c core.hooksPath=/dev/null -C "$wt" checkout -q -b "$tmp_name" "$head" \
      || die "cannot create temporary branch $tmp_name (a branch of that name may already exist; left untouched)"
    tmp_branch=$tmp_name
    if ! git -c core.hooksPath=/dev/null -c commit.gpgsign=false -C "$wt" rebase -q "$P" >/dev/null 2>&1; then
      echo "fmt-sweep-rebase: rebasing $branch onto the pre-sweep commit $(short "$P") conflicts in:" >&2
      git -C "$wt" diff --name-only --diff-filter=U -z > "$scratch/conflicts" || true
      while IFS= read -r -d '' p; do echo "  $p" >&2; done < "$scratch/conflicts"
      git -C "$wt" rebase --abort >/dev/null 2>&1 || true
      echo "fmt-sweep-rebase: rebase $branch onto $P manually (git rebase $P), resolve, and rerun; nothing was created." >&2
      exit 1
    fi
    rebased_tip=$(git -C "$wt" rev-parse HEAD)
  fi

  # ---- 4. replay
  git rev-list --reverse "$P..$rebased_tip" > "$scratch/commits" || die "git rev-list failed"
  mapfile -t commits < "$scratch/commits"
  new_parent=$S
  last_tree=$bind_want
  for c in "${commits[@]}"; do
    git -C "$wt" read-tree -u --reset "$c"
    bash "$self" --sweep-tree "$wt" >/dev/null || die "sweep transform failed on commit $(short "$c") ($(git log -1 --format=%s "$c")); nothing installed"
    git -C "$wt" add -A
    tree=$(git -C "$wt" write-tree)
    last_tree=$tree
    if [[ "$tree" == "$(git rev-parse "$new_parent^{tree}")" ]]; then
      echo "  $(short "$c") dropped (empty over the sweep)  $(git log -1 --format=%s "$c")"
      dropped+=("$(short "$c") $(git log -1 --format=%s "$c")")
      continue
    fi
    git cat-file commit "$c" | sed '1,/^$/d' > "$scratch/msg"
    new_parent=$(
      GIT_AUTHOR_NAME="$(git log -1 --format=%an "$c")" \
      GIT_AUTHOR_EMAIL="$(git log -1 --format=%ae "$c")" \
      GIT_AUTHOR_DATE="$(git log -1 --format=%ad --date=raw "$c")" \
      git commit-tree "$tree" -p "$new_parent" -F "$scratch/msg"
    ) || die "git commit-tree failed for $(short "$c")"
    made+=("$(git log -1 --format=%s "$c")")
    echo "  $(short "$c") -> $(short "$new_parent")  $(git log -1 --format=%s "$c")"
  done

  # ---- 5. final verification
  [[ "$last_tree" == "$(git rev-parse "$new_parent^{tree}")" ]] \
    || die "internal: replayed tip tree != sweep(tree($(short "$rebased_tip")))"
  git -C "$wt" read-tree -u --reset "$new_parent"
  (cd "$wt" && cargo-fmt --all -- --check >/dev/null 2>&1) || die "cargo-fmt --all -- --check is not clean on the replayed tip; nothing installed"

  # ---- 6. install (create-only; the owner's branch is never touched)
  local ref="refs/heads/fmt-swept/$branch" old status_line d_old d_new
  if old=$(git rev-parse --verify --quiet "$ref^{commit}"); then
    d_old=$(seq_digest "$S" "$old") && d_new=$(seq_digest "$S" "$new_parent") \
      || die "cannot compare the existing fmt-swept/$branch with the replay; left in place"
    if [[ "$d_old" == "$d_new" ]]; then
      status_line="already prepared"
      new_parent=$old
    else
      die "fmt-swept/$branch already exists and differs from the replay (ancestry, trees, messages or authors); left in place: delete it explicitly (git branch -D fmt-swept/$branch) and rerun"
    fi
  else
    git update-ref -m "fmt-sweep-rebase: replay $branch onto $(short "$S")" "$ref" "$new_parent" "" \
      || die "could not create $ref"
    status_line="prepared"
  fi

  # ---- 7. cleanup
  local crc=0
  cleanup_scratch || crc=1

  # ---- 8. report
  echo "fmt-sweep-rebase: fmt-swept/$branch $status_line"
  echo "  source branch : $branch @ $head (unchanged)"
  echo "  sweep S       : $S"
  echo "  pre-sweep P   : $P"
  echo "  new branch    : fmt-swept/$branch @ $new_parent"
  echo "  commits       : ${#made[@]} replayed, ${#dropped[@]} dropped"
  for i in "${dropped[@]}"; do echo "  dropped       : $i"; done
  echo "  toolchain     : $binding"
  cat <<EOF

fmt-swept/$branch sits on the sweep commit $(short "$S") ONLY. It is NOT a final or landable
state: it lacks every commit after the sweep on $upstream (the later land-gate/helper commit
and newer product commits). Two stages remain:
  1. git switch fmt-swept/$branch
       (or, once verified and with $branch not checked out: git branch -f $branch fmt-swept/$branch)
  2. git rebase $upstream
       onto the current tip; resolve the real conflicts, then rerun your canonical CPU
       acceptance BEFORE declaring the branch READY.
EOF
  exit "$crc"
}

case "${1:-}" in
  --sweep-tree)
    [[ $# -eq 2 ]] || { usage; exit 2; }
    sweep_tree "$2"
    ;;
  -h|--help) usage; exit 0 ;;
  "") usage; exit 2 ;;
  -*) usage; exit 2 ;;
  *)
    [[ $# -le 2 ]] || { usage; exit 2; }
    rebase_main "$@"
    ;;
esac
