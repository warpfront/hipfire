#!/usr/bin/env bash

# test-fmt-sweep-rebase.sh — fixture tests for scripts/fmt-sweep-rebase.sh (CPU only,
# no GPU, no network). Builds a throwaway two-crate cargo workspace repo under
# mktemp, makes a "sweep" commit with the real script's --sweep-tree, adds a later
# commit C (stand-in for the land-gate commit), then replays branches onto the sweep.
#
# Cases:
#   a basic            b skip-attributed parent    c gate + ordinary rebase onto C
#   d edition change   e dirty / untracked refuse  f P-rebase conflict
#   g rerun / differs  h syntax error              i merge commit
#   j wrong SWEEP_SHA  k inherited-hunk drop       l already on the sweep
#   m --sweep-tree == frozen oracle                n containment: symlinks refused; no write outside
#   f2 temp-branch name collision   g2 existing ref differing in author/date/body/tree
#   o failing xargs producer is an error
# Expected trees (oracles) come from scripts/fixtures/fmt-sweep-oracle.sh, a frozen,
# tracked, byte-pinned copy of the original one-time sweep script (it writes with
# cargo-fmt --all, then per-file rustfmt): an implementation independent of the script
# under test, available in any fresh checkout. There is no override and no skip mode.
# Exits nonzero if any assertion fails.

set -uo pipefail
cd "$(dirname "$0")/.."
REAL=$PWD
SCRIPT=$REAL/scripts/fmt-sweep-rebase.sh
ORACLE=$REAL/scripts/fixtures/fmt-sweep-oracle.sh
ORACLE_SHA256=117b250c482840eae76c303681a73943a123db0a6f990c0328edff586fbcab62
git -C "$REAL" ls-files --error-unmatch scripts/fixtures/fmt-sweep-oracle.sh >/dev/null 2>&1 \
  || { echo "  FAIL: scripts/fixtures/fmt-sweep-oracle.sh is not a tracked file"; exit 1; }
[[ "$(sha256sum < "$ORACLE" | cut -c1-64)" == "$ORACLE_SHA256" ]] \
  || { echo "  FAIL: scripts/fixtures/fmt-sweep-oracle.sh differs from the frozen reference (sha256 $ORACLE_SHA256)"; exit 1; }
SUBJECT='style: rustfmt the whole workspace (one-time 0.4.3 sweep)'

export ROCR_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 HIPFIRE_DEVICES=-1
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-16}"

T=$(mktemp -d "${TMPDIR:-/tmp}/test-fmt-sweep-rebase.XXXXXX")
trap 'rm -rf "$T"' EXIT
# Isolate from the user's git config and identity env (an exported identity overrides the fixture's
# local config, which the committer cases assert), and keep scratch dirs inspectable.
export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_SYSTEM=/dev/null
unset GIT_AUTHOR_NAME GIT_AUTHOR_EMAIL GIT_COMMITTER_NAME GIT_COMMITTER_EMAIL
mkdir -p "$T/tmp"
export TMPDIR="$T/tmp"

pass=0 fail=0
ok()  { echo "  PASS: $1"; pass=$((pass+1)); }
bad() { echo "  FAIL: $1"; fail=$((fail+1)); }
expect_eq() { if [[ "$2" == "$3" ]]; then ok "$1"; else bad "$1 (got '$2', want '$3')"; fi; }
expect_ne() { if [[ "$2" != "$3" ]]; then ok "$1"; else bad "$1 (both '$2')"; fi; }
expect_has() { if grep -qF -- "$3" <<<"$2"; then ok "$1"; else bad "$1 (missing '$3' in: $(head -c 400 <<<"$2"))"; fi; }
expect_rc() { expect_eq "$1" "$RC" "$2"; }

R=$T/repo
OUT="" RC=0
G() { git -C "$R" "$@"; }
W() { echo "$T/w-$1"; }
run_in() { local d=$1; shift; OUT=$(cd "$d" && bash "$SCRIPT" "$@" 2>&1); RC=$?; }
mkbranch() { G worktree add -q -b "$1" "$(W "$1")" "$2" >/dev/null 2>&1; }
ref_exists() { G rev-parse --verify --quiet "refs/heads/$1" >/dev/null; }
wt_count() { G worktree list | wc -l; }
tmp_leftovers() { { ls -d "$T"/tmp/fmt-sweep-rebase.* 2>/dev/null; G branch --list 'fmt-sweep-tmp*'; } | wc -l; }
files_hash() { # byte hash of every tracked and untracked file (and symlink) in a worktree, .git excluded
  (cd "$1" && find . -path ./.git -prune -o \( -type f -o -type l \) -print0 | LC_ALL=C sort -z \
     | xargs -0 -r sha256sum 2>&1 | sha256sum)
}
snap() { # owner state: HEAD, branch ref, porcelain status, index, and every file's bytes
  local d=$1
  echo "$(git -C "$d" rev-parse HEAD) $(git -C "$d" symbolic-ref -q HEAD) $(git -C "$d" status --porcelain --untracked-files=all | sha1sum) $(git -C "$d" ls-files -s | sha1sum) $(files_hash "$d")"
}

tick=0
FIXDATE=""
gcommit() { # dir message  (author: Alice, distinct date per commit unless FIXDATE; committer: local config)
  tick=$((tick+1))
  git -C "$1" add -A
  GIT_AUTHOR_NAME="Alice Author" GIT_AUTHOR_EMAIL="alice@example.org" \
  GIT_AUTHOR_DATE="${FIXDATE:-2026-02-0${tick}T03:04:05+0200}" git -C "$1" commit -q -m "$2"
}

oracle_tree() { # tree of sweep(<commit>) from the frozen oracle (independent of the script under test)
  local d="$T/oracle-$$-$RANDOM" t rc
  G worktree add -q --detach "$d" "$1" >/dev/null 2>&1 || { echo ORACLE-WORKTREE-FAILED; return; }
  bash "$ORACLE" "$d" >/dev/null 2>&1; rc=$?
  if [[ "$rc" -ne 0 ]]; then G worktree remove --force "$d" >/dev/null 2>&1; echo "ORACLE-FAILED-rc$rc"; return; fi
  git -C "$d" add -A
  t=$(git -C "$d" write-tree)
  G worktree remove --force "$d" >/dev/null 2>&1
  echo "$t"
}

# ------------------------------------------------------------------ fixture
build_fixture() {
  mkdir -p "$R"
  git init -q -b beta "$R"
  G config user.name "Local Committer"
  G config user.email "local@example.org"
  G config commit.gpgsign false
  (
    cd "$R"
    mkdir -p scripts crates/peacemaker-npu/src/isa crates/b/src crates/b/tests/ui docs tools/t/src
    cp "$REAL/scripts/ci-rustfmt-changed.sh" scripts/ci-rustfmt-changed.sh
    cat > Cargo.toml <<'EOF'
[workspace]
resolver = "2"
members = ["crates/peacemaker-npu", "crates/b"]

[workspace.package]
edition = "2021"
EOF
    cat > crates/peacemaker-npu/Cargo.toml <<'EOF'
[package]
name = "peacemaker-npu"
version = "0.0.0"
edition = "2021"
EOF
    cat > crates/b/Cargo.toml <<'EOF'
[package]
name = "b"
version = "0.0.0"
edition = "2021"
EOF
    printf 'pub mod isa;\npub fn f(x:i32)->i32{x+1}\n' > crates/peacemaker-npu/src/lib.rs
    cat > crates/peacemaker-npu/src/isa/mod.rs <<'EOF'
// isa parent
#[rustfmt::skip]
mod gen;
#[rustfmt::skip]
mod sched;
pub fn g(a:u8)->u8{a}
EOF
    printf 'pub const T:[u8;3]=[1,2,3];\n' > crates/peacemaker-npu/src/isa/gen.rs
    printf 'pub const U:[u8;2]=[4,5];\n'   > crates/peacemaker-npu/src/isa/sched.rs
    printf 'use a::{b9,b10};\npub fn h( ){ }\n' > crates/b/src/lib.rs
    printf 'fn main(){let x:i32="a";}\n' > crates/b/tests/ui/bad.rs
    printf 'error[E0308]: mismatched types\n --> bad.rs:1:23\n' > crates/b/tests/ui/bad.stderr
    printf 'fn   docs( ){ }\n' > docs/x.rs
    cat > tools/t/Cargo.toml <<'EOF'
[package]
name = "t"
version = "0.0.0"
edition = "2021"

[workspace]
EOF
    printf 'fn main(){println!("hi")}\n' > tools/t/src/main.rs
    cat > scripts/gate.sh <<'EOF'
#!/usr/bin/env bash
set -e
echo land-gate-v0
# ----
# ----
# ----
echo tail
EOF
  )
  gcommit "$R" "base"
  C0=$(G rev-parse HEAD)
  sed -i 's/x+1/x+2/' "$R/crates/peacemaker-npu/src/lib.rs"
  gcommit "$R" "pre-sweep tweak"
  P=$(G rev-parse HEAD)
  # S: the sweep commit, made by the real transform.
  bash "$SCRIPT" --sweep-tree "$R" >"$T/fixture-sweep.log" 2>&1 || { echo "fixture: sweep failed"; cat "$T/fixture-sweep.log"; exit 2; }
  git -C "$R" add -A
  GIT_AUTHOR_NAME="Sweeper" GIT_AUTHOR_EMAIL="sweep@example.org" git -C "$R" commit -q -m "$SUBJECT"
  S=$(G rev-parse HEAD)
  # C: the later "gate/helper" commit on the sweep.
  sed -i 's/land-gate-v0/land-gate-v1/' "$R/scripts/gate.sh"
  sed -i '1s|.*|// isa parent (C)|' "$R/crates/peacemaker-npu/src/isa/mod.rs"
  gcommit "$R" "gate: land-gate-v1"
  C=$(G rev-parse HEAD)
}

echo "[fixture] building throwaway workspace"
build_fixture
echo "  P=$(git -C "$R" rev-parse --short "$P") S=$(git -C "$R" rev-parse --short "$S") C=$(git -C "$R" rev-parse --short "$C")"
G show "$S:crates/b/src/lib.rs" | grep -q 'b10, b9' || echo "  (note: fixture sweep did not produce '{b10, b9}')"

# ---------------------------------------------------------------------- a
echo "[a] basic: unformatted .rs + non-Rust file in two commits"
mkbranch br-a "$P"
printf 'pub fn extra(a:u8)->u8{a*2}\n' >> "$(W br-a)/crates/peacemaker-npu/src/lib.rs"
gcommit "$(W br-a)" "a1: add extra"
sed -i 's/^echo tail/echo a-tail/' "$(W br-a)/scripts/gate.sh"
gcommit "$(W br-a)" "a2: gate tail"
before=$(snap "$(W br-a)"); tipA=$(G rev-parse br-a)
run_in "$(W br-a)" "$S" beta
after=$(snap "$(W br-a)")
expect_rc "a: exit 0" 0
ref_exists fmt-swept/br-a && ok "a: fmt-swept/br-a exists" || bad "a: fmt-swept/br-a missing ($OUT)"
expect_eq "a: tip tree == sweep(tree(branch tip))" "$(G rev-parse fmt-swept/br-a^{tree})" "$(oracle_tree "$tipA")"
expect_eq "a: owner branch/worktree/index unchanged" "$after" "$before"
expect_eq "a: owner branch tip unchanged" "$(G rev-parse br-a)" "$tipA"
expect_eq "a: replayed commits are S + 2" "$(G rev-list --count "$S..fmt-swept/br-a")" 2
expect_eq "a: subjects preserved" "$(G log --reverse --format=%s "$S..fmt-swept/br-a" | tr '\n' '|')" "a1: add extra|a2: gate tail|"
expect_eq "a: author name/email/date preserved" \
  "$(G log --reverse --format='%an|%ae|%ad' --date=raw "$S..fmt-swept/br-a")" \
  "$(G log --reverse --format='%an|%ae|%ad' --date=raw "$P..br-a")"
expect_eq "a: committer = local git config" \
  "$(G log --format='%cn|%ce' "$S..fmt-swept/br-a" | sort -u)" "Local Committer|local@example.org"
expect_has "a: report names source tip" "$OUT" "br-a @ $tipA (unchanged)"
expect_has "a: report names S" "$OUT" "$S"
expect_has "a: report names P" "$OUT" "$P"
expect_has "a: report names new branch tip" "$OUT" "fmt-swept/br-a @ $(G rev-parse fmt-swept/br-a)"
expect_has "a: report has dropped count" "$OUT" "0 dropped"
expect_has "a: report has toolchain binding" "$OUT" "toolchain     : OK"
expect_has "a: report states two-stage identity" "$OUT" "NOT a final or landable"
expect_has "a: next step rebases onto UPSTREAM" "$OUT" "git rebase beta"
expect_eq "a: no scratch leftovers" "$(tmp_leftovers)" 0

# ---------------------------------------------------------------------- b
echo "[b] branch modifies the skip-attributed parent file"
mkbranch br-b "$P"
sed -i 's/{a}$/{a+1}/' "$(W br-b)/crates/peacemaker-npu/src/isa/mod.rs"
gcommit "$(W br-b)" "b1: touch isa parent"
run_in "$(W br-b)" "$S" beta
expect_rc "b: exit 0" 0
mod=$(G show fmt-swept/br-b:crates/peacemaker-npu/src/isa/mod.rs)
expect_eq "b: #[rustfmt::skip] kept on both module decls" "$(grep -c '^#\[rustfmt::skip\]$' <<<"$mod")" 2
expect_has "b: parent file formatted" "$mod" "a + 1"
expect_eq "b: gen.rs byte-identical to P" "$(G rev-parse fmt-swept/br-b:crates/peacemaker-npu/src/isa/gen.rs)" "$(G rev-parse "$P:crates/peacemaker-npu/src/isa/gen.rs")"
expect_eq "b: sched.rs byte-identical to P" "$(G rev-parse fmt-swept/br-b:crates/peacemaker-npu/src/isa/sched.rs)" "$(G rev-parse "$P:crates/peacemaker-npu/src/isa/sched.rs")"

# ---------------------------------------------------------------------- c
echo "[c] gate.sh + formatted file; then the ordinary rebase onto C"
mkbranch br-c "$P"
sed -i 's/^echo tail/echo branch-tail/' "$(W br-c)/scripts/gate.sh"
sed -i 's/{a}$/{a+1}/' "$(W br-c)/crates/peacemaker-npu/src/isa/mod.rs"
gcommit "$(W br-c)" "c1: gate tail + isa parent"
run_in "$(W br-c)" "$S" beta
expect_rc "c: exit 0" 0
expect_has "c: branch's gate.sh change present" "$(G show fmt-swept/br-c:scripts/gate.sh)" "echo branch-tail"
expect_has "c: sweep formatting present in the branch's file" "$(G show fmt-swept/br-c:crates/peacemaker-npu/src/isa/mod.rs)" "a + 1"
expect_eq "c: fmt-swept/br-c is based on S, not C" "$(G merge-base fmt-swept/br-c beta)" "$S"
# stage two: ordinary rebase onto the current tip C
G worktree add -q -b rb-c "$T/rb-c" fmt-swept/br-c >/dev/null 2>&1
rb=$(cd "$T/rb-c" && git rebase beta 2>&1); rbrc=$?
expect_eq "c: ordinary git rebase beta succeeds" "$rbrc" 0
rbtree_gate=$(git -C "$T/rb-c" show HEAD:scripts/gate.sh)
rbtree_mod=$(git -C "$T/rb-c" show HEAD:crates/peacemaker-npu/src/isa/mod.rs)
expect_has "c: C's gate line present after rebase" "$rbtree_gate" "echo land-gate-v1"
expect_has "c: branch's gate line present after rebase" "$rbtree_gate" "echo branch-tail"
expect_eq "c: #[rustfmt::skip] attrs present after rebase" "$(grep -c '^#\[rustfmt::skip\]$' <<<"$rbtree_mod")" 2
expect_has "c: C's parent-file comment present after rebase" "$rbtree_mod" "// isa parent (C)"
# conflicting variant: branch edits the very line C edits
mkbranch br-c2 "$P"
sed -i 's/land-gate-v0/branch-gate/' "$(W br-c2)/scripts/gate.sh"
gcommit "$(W br-c2)" "c2: edit the gate line"
run_in "$(W br-c2)" "$S" beta
expect_rc "c2: replay itself succeeds (stage one)" 0
G worktree add -q -b rb-c2 "$T/rb-c2" fmt-swept/br-c2 >/dev/null 2>&1
rb=$(cd "$T/rb-c2" && git rebase beta 2>&1); rbrc=$?
expect_ne "c2: ordinary rebase onto C stops on the genuine conflict" "$rbrc" 0
expect_has "c2: conflict is reported clearly" "$rb" "CONFLICT"
expect_has "c2: conflict names scripts/gate.sh" "$rb" "scripts/gate.sh"
(cd "$T/rb-c2" && git rebase --abort >/dev/null 2>&1)

# ---------------------------------------------------------------------- d
echo "[d] edition change 2021 -> 2024 on the second crate"
mkbranch br-d "$P"
sed -i 's/edition = "2021"/edition = "2024"/' "$(W br-d)/crates/b/Cargo.toml"
gcommit "$(W br-d)" "d1: crate b to edition 2024"
tipD=$(G rev-parse br-d)
run_in "$(W br-d)" "$S" beta
expect_rc "d: exit 0" 0
expect_eq "d: replayed tree == oracle sweep(tree(tip))" "$(G rev-parse fmt-swept/br-d^{tree})" "$(oracle_tree "$tipD")"
expect_has "d: S formatted crate b under 2021 version order" "$(G show "$S:crates/b/src/lib.rs")" "use a::{b10, b9};"
expect_has "d: replay re-formatted crate b under 2024 (differs from a per-file-only result)" \
  "$(G show fmt-swept/br-d:crates/b/src/lib.rs)" "use a::{b9, b10};"

# ---------------------------------------------------------------------- e
echo "[e] dirty tracked file / untracked file are refused"
mkbranch br-e "$P"
printf 'pub fn e(){}\n' >> "$(W br-e)/crates/b/src/lib.rs"
gcommit "$(W br-e)" "e1: commit"
wts=$(wt_count)
echo "// dirty" >> "$(W br-e)/crates/b/src/lib.rs"
before=$(snap "$(W br-e)")
run_in "$(W br-e)" "$S" beta
expect_rc "e: dirty tracked file refused" 1
expect_has "e: dirty message" "$OUT" "not clean"
ref_exists fmt-swept/br-e && bad "e: fmt-swept/br-e must not exist" || ok "e: nothing created (dirty)"
expect_eq "e: dirty owner state unchanged" "$(snap "$(W br-e)")" "$before"
git -C "$(W br-e)" checkout -q -- crates/b/src/lib.rs
touch "$(W br-e)/untracked.txt"
run_in "$(W br-e)" "$S" beta
expect_rc "e: untracked file refused" 1
expect_has "e: untracked message" "$OUT" "not clean"
ref_exists fmt-swept/br-e && bad "e: fmt-swept/br-e must not exist" || ok "e: nothing created (untracked)"
expect_eq "e: no extra worktrees" "$(wt_count)" "$wts"
rm -f "$(W br-e)/untracked.txt"
# in-progress state marker
mkdir -p "$(git -C "$(W br-e)" rev-parse --absolute-git-dir)/rebase-merge"
run_in "$(W br-e)" "$S" beta
expect_rc "e: rebase-in-progress refused" 1
expect_has "e: in-progress message" "$OUT" "in progress"
rmdir "$(git -C "$(W br-e)" rev-parse --absolute-git-dir)/rebase-merge"
git -C "$(W br-e)" checkout -q --detach
run_in "$(W br-e)" "$S" beta
expect_rc "e: detached HEAD refused" 1

# ---------------------------------------------------------------------- f
echo "[f] P-rebase conflict (branch cut before a conflicting pre-sweep commit)"
mkbranch br-f "$C0"
sed -i 's/x+1/x+9/' "$(W br-f)/crates/peacemaker-npu/src/lib.rs"
gcommit "$(W br-f)" "f1: conflicts with the pre-sweep tweak"
wts=$(wt_count); before=$(snap "$(W br-f)")
run_in "$(W br-f)" "$S" beta
expect_rc "f: exit 1" 1
expect_has "f: names the conflicting path" "$OUT" "crates/peacemaker-npu/src/lib.rs"
expect_has "f: tells the owner to rebase onto P manually" "$OUT" "rebase br-f onto $P manually"
ref_exists fmt-swept/br-f && bad "f: fmt-swept/br-f must not exist" || ok "f: no fmt-swept branch"
expect_eq "f: no leftover worktree" "$(wt_count)" "$wts"
expect_eq "f: no temp branch / scratch dir" "$(tmp_leftovers)" 0
expect_eq "f: owner untouched" "$(snap "$(W br-f)")" "$before"

# ---------------------------------------------------------------------- g
echo "[g] rerun after success / differing existing fmt-swept branch"
gbefore=$(G rev-parse fmt-swept/br-a)
run_in "$(W br-a)" "$S" beta
expect_rc "g: rerun exit 0" 0
expect_has "g: rerun says already prepared" "$OUT" "already prepared"
expect_eq "g: rerun leaves the branch alone" "$(G rev-parse fmt-swept/br-a)" "$gbefore"
mkbranch br-g "$P"
printf 'pub fn g2(){}\n' >> "$(W br-g)/crates/b/src/lib.rs"
gcommit "$(W br-g)" "g1: commit"
G branch fmt-swept/br-g "$S"
run_in "$(W br-g)" "$S" beta
expect_rc "g: differing existing branch refused" 1
expect_has "g: tells the owner to delete it explicitly" "$OUT" "delete it explicitly"
expect_eq "g: existing branch not moved" "$(G rev-parse fmt-swept/br-g)" "$S"
expect_eq "g: no scratch leftovers" "$(tmp_leftovers)" 0

# --------------------------------------------------------------------- g2
echo "[g2] existing fmt-swept ref differing only in author / date / body / intermediate tree: refused, kept"
FIXDATE="2026-03-01T01:02:03+0000"   # every commit of these fixtures shares one author date
g2_build() { # BRANCH: two commits (v1 / v2 with a body), run the tool
  local b=$1 w; w=$(W "$b")
  mkbranch "$b" "$P"
  printf 'pub fn one(){}\n' >> "$w/crates/b/src/lib.rs"
  gcommit "$w" "v1: one"
  printf 'pub fn two(){}\n' >> "$w/crates/b/src/lib.rs"
  gcommit "$w" "$(printf 'v2: two\n\nbody line')"
  run_in "$w" "$S" beta
}
g2_case() { # NAME MUTATE-FN
  local name=$1 b="br-g2-$1" w kept src
  w=$(W "$b")
  g2_build "$b"
  expect_rc "g2/$name: first run exit 0" 0
  kept=$(G rev-parse "fmt-swept/$b"); src=$(G rev-parse "$b")
  "$2" "$w"
  expect_ne "g2/$name: owner branch really was rewritten" "$(G rev-parse "$b")" "$src"
  run_in "$w" "$S" beta
  expect_rc "g2/$name: rerun refused" 1
  expect_has "g2/$name: says it differs" "$OUT" "already exists and differs"
  expect_eq "g2/$name: existing ref left in place" "$(G rev-parse "fmt-swept/$b")" "$kept"
  expect_eq "g2/$name: no scratch leftovers" "$(tmp_leftovers)" 0
}
g2_author() { GIT_AUTHOR_DATE="$FIXDATE" git -C "$1" commit -q --amend --no-edit --author="Mallory <m@example.org>"; }
g2_date()   { GIT_AUTHOR_NAME="Alice Author" GIT_AUTHOR_EMAIL="alice@example.org" GIT_AUTHOR_DATE="2026-03-02T01:02:03+0000" \
                git -C "$1" commit -q --amend --no-edit --reset-author; }
g2_body()   { GIT_AUTHOR_NAME="Alice Author" GIT_AUTHOR_EMAIL="alice@example.org" GIT_AUTHOR_DATE="$FIXDATE" \
                git -C "$1" commit -q --amend -m "v2: two" -m "a different body"; }
g2_tree()   { # same subjects, same final tree, different tree after v1
  git -C "$1" reset -q --hard "$P"
  printf 'pub fn zero(){}\npub fn one(){}\n' >> "$1/crates/b/src/lib.rs"
  gcommit "$1" "v1: one"
  sed -i '/zero/d' "$1/crates/b/src/lib.rs"
  printf 'pub fn two(){}\n' >> "$1/crates/b/src/lib.rs"
  gcommit "$1" "$(printf 'v2: two\n\nbody line')"
}
g2_case author g2_author
g2_case date   g2_date
g2_case body   g2_body
g2_case tree   g2_tree
FIXDATE=""

# --------------------------------------------------------------------- f2
echo "[f2] temporary-branch name collision: the unrelated ref and the owner are preserved"
mkdir -p "$T/shim-mktemp"
REAL_MKTEMP=$(command -v mktemp)
cat > "$T/shim-mktemp/mktemp" <<EOF
#!/bin/sh
case "\$*" in
  */fmt-sweep-rebase.*) d="\$TMPDIR/fmt-sweep-rebase.COLLIDE"; rm -rf "\$d"; mkdir "\$d" && echo "\$d" ;;
  *) exec "$REAL_MKTEMP" "\$@" ;;
esac
EOF
chmod +x "$T/shim-mktemp/mktemp"
mkbranch br-coll "$C0"
printf 'pub fn coll(){}\n' >> "$(W br-coll)/crates/b/src/lib.rs"
gcommit "$(W br-coll)" "coll1: commit"
G branch fmt-sweep-tmp-COLLIDE "$C0"
wts=$(wt_count); before=$(snap "$(W br-coll)")
OLDPATH=$PATH; PATH="$T/shim-mktemp:$PATH"
run_in "$(W br-coll)" "$S" beta
PATH=$OLDPATH
expect_rc "f2: exit 1" 1
expect_has "f2: names the collision" "$OUT" "cannot create temporary branch fmt-sweep-tmp-COLLIDE"
expect_eq "f2: unrelated ref preserved" "$(G rev-parse fmt-sweep-tmp-COLLIDE)" "$C0"
ref_exists fmt-swept/br-coll && bad "f2: fmt-swept/br-coll must not exist" || ok "f2: nothing created"
expect_eq "f2: no leftover worktree" "$(wt_count)" "$wts"
expect_eq "f2: scratch dir removed" "$([[ -e "$T/tmp/fmt-sweep-rebase.COLLIDE" ]] && echo present || echo gone)" gone
expect_eq "f2: owner untouched" "$(snap "$(W br-coll)")" "$before"
G branch -D fmt-sweep-tmp-COLLIDE >/dev/null
# ---------------------------------------------------------------------- h
echo "[h] syntax error in a branch commit"
mkbranch br-h "$P"
printf 'pub fn broken( {\n' >> "$(W br-h)/crates/b/src/lib.rs"
gcommit "$(W br-h)" "h1: syntax error"
wts=$(wt_count); before=$(snap "$(W br-h)")
run_in "$(W br-h)" "$S" beta
expect_rc "h: exit 1" 1
expect_has "h: failure names the transform on the bad commit" "$OUT" "sweep transform failed on commit"
ref_exists fmt-swept/br-h && bad "h: fmt-swept/br-h must not exist" || ok "h: nothing created"
expect_eq "h: no leftover worktree" "$(wt_count)" "$wts"
expect_eq "h: no scratch leftovers" "$(tmp_leftovers)" 0
expect_eq "h: owner untouched" "$(snap "$(W br-h)")" "$before"

# ---------------------------------------------------------------------- i
echo "[i] merge commit in range"
mkbranch br-i "$P"
printf 'pub fn i1(){}\n' >> "$(W br-i)/crates/b/src/lib.rs"
gcommit "$(W br-i)" "i1: main side"
G branch side-i "$P"
G worktree add -q "$T/w-side-i" side-i >/dev/null 2>&1
printf 'pub fn i2(){}\n' >> "$T/w-side-i/crates/peacemaker-npu/src/lib.rs"
gcommit "$T/w-side-i" "i2: side"
git -C "$(W br-i)" merge -q --no-ff side-i -m "i3: merge" >/dev/null 2>&1
wts=$(wt_count); before=$(snap "$(W br-i)")
run_in "$(W br-i)" "$S" beta
expect_rc "i: exit 1" 1
expect_has "i: merge commits named" "$OUT" "merge commits"
ref_exists fmt-swept/br-i && bad "i: fmt-swept/br-i must not exist" || ok "i: nothing created"
expect_eq "i: no worktree was ever created" "$(wt_count)" "$wts"
expect_eq "i: owner untouched" "$(snap "$(W br-i)")" "$before"

# ---------------------------------------------------------------------- j
echo "[j] wrong SWEEP_SHA"
mkbranch br-j "$P"
printf 'pub fn j1(){}\n' >> "$(W br-j)/crates/b/src/lib.rs"
gcommit "$(W br-j)" "j1: commit"
run_in "$(W br-j)" "$P" beta
expect_rc "j: SWEEP_SHA with a different subject refused" 1
expect_has "j: subject message" "$OUT" "not the sweep commit"
# fake sweep: right subject, tree != sweep(parent)
G branch fake-up "$P"
G worktree add -q "$T/w-fake" fake-up >/dev/null 2>&1
printf '// fake\n' >> "$T/w-fake/crates/b/src/lib.rs"
git -C "$T/w-fake" add -A
git -C "$T/w-fake" commit -q -m "$SUBJECT"
FAKE=$(git -C "$T/w-fake" rev-parse HEAD)
run_in "$(W br-j)" "$FAKE" fake-up
expect_rc "j: fake sweep (tree != sweep(parent)) refused" 1
expect_has "j: toolchain-binding message" "$OUT" "does not reproduce the sweep"
# fake sweep touching a non-.rs file
G branch fake2-up "$P"
G worktree add -q "$T/w-fake2" fake2-up >/dev/null 2>&1
echo "# x" >> "$T/w-fake2/scripts/gate.sh"
git -C "$T/w-fake2" add -A
git -C "$T/w-fake2" commit -q -m "$SUBJECT"
FAKE2=$(git -C "$T/w-fake2" rev-parse HEAD)
run_in "$(W br-j)" "$FAKE2" fake2-up
expect_rc "j: fake sweep touching a non-.rs file refused" 1
expect_has "j: non-.rs message" "$OUT" "non-.rs"
# real sweep, not reachable from UPSTREAM
run_in "$(W br-j)" "$S" "$P"
expect_rc "j: SWEEP_SHA not reachable from UPSTREAM refused" 1
expect_has "j: unreachable message" "$OUT" "not reachable"
for b in br-j; do ref_exists "fmt-swept/$b" && bad "j: nothing may be created" || ok "j: nothing created"; done
expect_eq "j: no scratch leftovers" "$(tmp_leftovers)" 0

# ---------------------------------------------------------------------- k
echo "[k] branch whose only commit formats an inherited hunk"
mkbranch br-k "$P"
rustfmt --edition 2021 --config skip_children=true "$(W br-k)/crates/peacemaker-npu/src/lib.rs"
gcommit "$(W br-k)" "k1: rustfmt inherited hunks"
run_in "$(W br-k)" "$S" beta
expect_rc "k: exit 0" 0
expect_has "k: reported as dropped" "$OUT" "dropped (empty over the sweep)"
expect_has "k: dropped commit listed in the report" "$OUT" "k1: rustfmt inherited hunks"
expect_eq "k: fmt-swept tip == SWEEP" "$(G rev-parse fmt-swept/br-k)" "$S"

# ---------------------------------------------------------------------- l
echo "[l] branch already containing the sweep"
mkbranch br-l "$S"
printf 'pub fn l1() {}\n' >> "$(W br-l)/crates/b/src/lib.rs"
gcommit "$(W br-l)" "l1: after the sweep"
before=$(snap "$(W br-l)")
run_in "$(W br-l)" "$S" beta
expect_rc "l: exit 0" 0
expect_has "l: says already on the sweep" "$OUT" "already on the sweep; rebase normally"
ref_exists fmt-swept/br-l && bad "l: fmt-swept/br-l must not exist" || ok "l: nothing created"
expect_eq "l: owner untouched" "$(snap "$(W br-l)")" "$before"

# ---------------------------------------------------------------------- m
echo "[m] --sweep-tree == the frozen oracle (the original sweep script)"
G worktree add -q --detach "$T/m-old" "$P" >/dev/null 2>&1
G worktree add -q --detach "$T/m-new" "$P" >/dev/null 2>&1
bash "$ORACLE" "$T/m-old" >/dev/null 2>&1; oldrc=$?
bash "$SCRIPT" --sweep-tree "$T/m-new" >/dev/null 2>&1; newrc=$?
git -C "$T/m-old" add -A; git -C "$T/m-new" add -A
expect_eq "m: oracle exit 0" "$oldrc" 0
expect_eq "m: --sweep-tree exit 0" "$newrc" 0
expect_eq "m: trees byte-identical" "$(git -C "$T/m-new" write-tree)" "$(git -C "$T/m-old" write-tree)"
expect_eq "m: ... and equal to S's tree" "$(git -C "$T/m-new" write-tree)" "$(G rev-parse "$S^{tree}")"
expect_ne "m: ... and the sweep changed something" "$(git -C "$T/m-new" write-tree)" "$(G rev-parse "$P^{tree}")"

# ---------------------------------------------------------------------- n
echo "[n] containment: symlinks fail closed; nothing outside the tree is ever written"
mkdir -p "$T/outside/xcrate/src"
printf 'fn   evil( ){ }\n' > "$T/outside/evil.rs"
printf '[package]\nname = "xcrate"\nversion = "0.0.0"\nedition = "2021"\n' > "$T/outside/xcrate/Cargo.toml"
printf 'pub fn   x( ){ }\n' > "$T/outside/xcrate/src/lib.rs"
outside_hash() { (cd "$T/outside" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum); }
canary=$(outside_hash)
n_wt() { local d="$T/n-$1"; shift; G worktree add -q --detach "$d" "$P" >/dev/null 2>&1; (cd "$d" && "$@"); git -C "$d" add -A; echo "$d"; }
# (1) Tracked symlinks: refused before any formatter runs.
n_refuse() { # NAME MESSAGE-FRAGMENT SETUP-COMMAND...
  local name=$1 frag=$2 d; shift 2
  d=$(n_wt "$name" "$@")
  OUT=$(bash "$SCRIPT" --sweep-tree "$d" 2>&1); RC=$?
  expect_rc "n/$name: exit 1" 1
  expect_has "n/$name: message" "$OUT" "$frag"
  expect_eq "n/$name: outside canary unchanged" "$(outside_hash)" "$canary"
  expect_eq "n/$name: nothing was formatted in the tree" "$(git -C "$d" diff --stat | wc -l)" 0
}
n_rs_symlink()  { ln -s crates/b/src/lib.rs link.rs; }
n_dir_symlink() { ln -s crates/b linkdir; }
n_esc_dir()     { ln -s "$T/outside" escdir; }
n_esc_file()    { ln -s "$T/outside/evil.rs" esc.txt; }
n_refuse rs-symlink  "link.rs"                   n_rs_symlink
n_refuse dir-symlink "tracked directory symlink" n_dir_symlink
n_refuse esc-dir     "tracked directory symlink" n_esc_dir
n_refuse esc-file    "escapes the work tree"     n_esc_file
# (2) Out-of-tree modules / path deps / members, in any valid spelling. The per-file writer
#     (skip_children) never follows them and cargo-fmt only checks, so the outside canary is
#     unchanged; where the read-only workspace check reaches the unformatted outside file (or
#     cargo cannot load the manifest) the run stops with exit 1. No spelling is parsed.
n_reach() { # NAME RC-WANT SETUP-COMMAND...
  local name=$1 want=$2 d; shift 2
  d=$(n_wt "$name" "$@")
  OUT=$(bash "$SCRIPT" --sweep-tree "$d" 2>&1); RC=$?
  expect_rc "n/$name: exit $want" "$want"
  [[ "$want" == 0 ]] || expect_has "n/$name: read-only check stops it" "$OUT" "is not clean after the per-file pass"
  expect_eq "n/$name: outside canary unchanged" "$(outside_hash)" "$canary"
  expect_eq "n/$name: no file created in the tree" "$(git -C "$d" status --porcelain --untracked-files=all | grep -c '^??')" 0
}
lib=crates/b/src/lib.rs
n_mod()          { printf '%s\nmod evil;\n' "$1" >> $lib; }
n_path_canon()   { n_mod '#[path = "../../../../outside/evil.rs"]'; }
n_path_abs()     { n_mod "#[path = \"$T/outside/evil.rs\"]"; }
n_path_comment() { n_mod '#[path/**/="../../../../outside/evil.rs"]'; }
n_path_spaced()  { n_mod '# [ path = "../../../../outside/evil.rs" ]'; }
n_path_raw()     { n_mod '#[path = r"../../../../outside/evil.rs"]'; }
n_path_rawid()   { n_mod '#[r#path = "../../../../outside/evil.rs"]'; }
n_path_multi()   { printf '#[path =\n    "../../../../outside/evil.rs"]\nmod evil;\n' >> $lib; }
n_cfgattr_multi(){ printf '#[cfg_attr(\n    all(),\n    path = "../../../../outside/evil.rs"\n)]\nmod evil;\n' >> $lib; }
n_toml_dep()     { printf '\n[dependencies]\nxcrate = { path = "../../../../outside/xcrate" }\n' >> crates/b/Cargo.toml; }
n_toml_dep_sq()  { printf "\n[dependencies]\nxcrate = { 'path' = '../../../../outside/xcrate' }\n" >> crates/b/Cargo.toml; }
n_toml_member()  { sed -i 's|^members = \[|members = ["../../outside/xcrate", |' Cargo.toml; }
n_reach path-canon    1 n_path_canon
n_reach path-abs      1 n_path_abs
n_reach path-comment  1 n_path_comment
n_reach path-spaced   1 n_path_spaced
n_reach path-raw      1 n_path_raw
n_reach path-multi    1 n_path_multi
n_reach toml-dep      1 n_toml_dep
n_reach toml-dep-sq   1 n_toml_dep_sq
n_reach toml-member   1 n_toml_member
# rustfmt may not resolve these two spellings; whatever it does, it only reads.
for c in path-rawid:n_path_rawid cfgattr-multi:n_cfgattr_multi; do
  d=$(n_wt "${c%%:*}" "${c#*:}")
  OUT=$(bash "$SCRIPT" --sweep-tree "$d" 2>&1); RC=$?
  expect_eq "n/${c%%:*}: outside canary unchanged (exit $RC)" "$(outside_hash)" "$canary"
  expect_eq "n/${c%%:*}: no file created in the tree" "$(git -C "$d" status --porcelain --untracked-files=all | grep -c '^??')" 0
done
# Negative control: the fixtures above can see an outside write. The frozen oracle formats
# with a cargo-fmt --all WRITE, which follows the #[path] module outside the tree.
for c in ctl-path:n_path_canon; do
  d=$(n_wt "${c%%:*}" "${c#*:}")
  bash "$ORACLE" "$d" >/dev/null 2>&1
  expect_ne "n/${c%%:*}: a cargo-fmt writer DOES change the outside canary" "$(outside_hash)" "$canary"
  printf 'fn   evil( ){ }\n' > "$T/outside/evil.rs"; printf 'pub fn   x( ){ }\n' > "$T/outside/xcrate/src/lib.rs"
  expect_eq "n/${c%%:*}: canary restored" "$(outside_hash)" "$canary"
done

# ---------------------------------------------------------------------- o
echo "[o] a failing file-enumeration/worker producer (xargs) is an error, not an empty list"
mkdir -p "$T/shim-xargs"
printf '#!/bin/sh\nexit 127\n' > "$T/shim-xargs/xargs"; chmod +x "$T/shim-xargs/xargs"
G worktree add -q --detach "$T/o-wt" "$P" >/dev/null 2>&1
OLDPATH=$PATH; PATH="$T/shim-xargs:$PATH"
OUT=$(bash "$SCRIPT" --sweep-tree "$T/o-wt" 2>&1); RC=$?
PATH=$OLDPATH
expect_rc "o: exit 1" 1
expect_has "o: worker failure reported" "$OUT" "per-file formatting failed (xargs exit"

echo
echo "fmt-sweep-rebase tests: $pass passed, $fail failed"
[[ "$fail" -eq 0 ]]
