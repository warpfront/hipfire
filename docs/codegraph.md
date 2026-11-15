# Code intelligence — CodeGraph

**Lifecycle:** shipped / reference. The full usage contract, moved out of
`AGENTS.md` on 2026-11-15; the rules summary stays in
[`AGENTS.md`](../AGENTS.md) § "Code intelligence — CodeGraph".

GitNexus is **removed** from this project. The code-intelligence tool for
hipfire is **CodeGraph**: a per-project SQLite graph of symbols, edges,
and call paths, queried through the `codegraph_explore` MCP tool (or the
`codegraph` CLI, which prints the same output).

## Always

- **`codegraph_explore` FIRST, before a grep/read loop or an edit.** Ask
  it how an area works, or hand it a bag of symbol/file names (`"MoE tape
  replay dflash_spec_demo verify_block"`). It returns the relevant
  symbols' verbatim, line-numbered source grouped by file, plus the call
  paths between them, in one capped call. Source it shows is *already
  read* — do not re-open those files.
- **Point it at an index that exists.** Not every checkout carries a
  `.codegraph/` — fresh worktrees start without one. In an unindexed
  checkout either run `codegraph init .` (the index is per-checkout,
  machine-local, and gitignored) or pass `projectPath` at a sibling
  checkout that is indexed.
- **Treat a foreign `projectPath` as orientation only.** Its source and
  line numbers belong to that checkout's commit, not yours. Re-`read` the
  file in *this* worktree before editing it.
- **Keep the index honest.** `codegraph status` for freshness,
  `codegraph sync` after a large rebase or a big kernel/dispatch change.
- **Respect the explore budget.** Every call reports how many explores
  remain for that project and how many files each covers. Spend the
  remaining calls on areas an earlier call did not reach, then
  synthesize — dropping back to a grep/read loop is the more expensive
  option, not the safer one.

## Never

- NEVER cite a symbol, caller, or line number from CodeGraph that you
  have not seen in its output — the index can be stale, and a stale
  callsite claim is the same failure class as an invented one.
- NEVER use CodeGraph as the authority for "did I get every callsite" on
  an exported symbol. Use `lsp references` for that; CodeGraph shows the
  call *paths*, the language server shows the *set*.
- NEVER rename across files with find-and-replace. Use `lsp rename`.

## CLI (same index, when MCP is unavailable)

| Task | Command |
|------|---------|
| Explore an area | `codegraph explore "<symbols or question>"` |
| One symbol's source + caller/callee trail | `codegraph node <name>` |
| Who calls / what it calls | `codegraph callers <sym>` · `codegraph callees <sym>` |
| Blast radius of a change | `codegraph impact <sym>` |
| Tests touched by changed files | `codegraph affected <files...>` |
| Index state / rebuild / incremental | `codegraph status` · `index` · `sync` |
