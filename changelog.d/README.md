# changelog.d: one file per change

`CHANGELOG.md` is not edited by changes. Add one fragment per change here:

- Name: `<issue-or-slug>.md` (several sections: `<label>-<n>.md`).
- First line: the `### ` section heading the entry belongs under, then a blank
  line, then the bullet(s), verbatim as they should read (multi-line bullets
  keep their indented continuation lines).

```
### Fixes: GDN prefill independent of outer chunking

- **Fixed: ...** What changed, why, and the beta commit.
```

`scripts/changelog-assemble.py` folds fragments into the `## Unreleased`
section at release prep (bullets join an existing identical `### ` heading in
Unreleased, otherwise a new heading is appended) and deletes them.
`scripts/changelog-assemble.py --check` prints the result without touching
files. `scripts/check-changelog.py` rejects a fragment with no `### ` heading,
no bullet, or a bullet that repeats one already in Unreleased or another
fragment. This file (`README.md`) is not a fragment.
