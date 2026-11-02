# hipfire-tui

The hipfire terminal UI (`hipfire tui`, or bare `hipfire`).

Run from the repository root:

```bash
cargo run -p hipfire-tui
```

`hipfire-tui --settings` opens on the Settings tab (what `hipfire config` runs),
`--tuning` opens it with the Tuning preset chooser (`hipfire config tuning`),
`--config-profile-wizard` opens the profile wizard (`hipfire config profile`),
and `--check` loads config, registry and local models without a terminal.

Tabs:

- **Home**: serve state, endpoint, active model, local/registry counts.
- **Dashboard**: live serve stats, GPU/VRAM (`amd-smi`, else `rocm-smi`, found
  through the ROCm-root lookup), live `/metrics`. `s` starts the serve; `x`
  (stop) and `R` (restart, which force-reaps this install's serve daemons) ask
  y/n first.
- **Chat**: streams through the serve's `/v1/chat/completions`. The transcript
  follows new output; Up/Down/PgUp/PgDn/Home/End scroll, also while a reply
  streams. When the serve is offline, Enter queues the prompt, starts
  `hipfire serve -d` if no serve process exists, and sends it once `/health` is
  online. `/help` lists the slash commands.
- **Models**: registry tags grouped by family plus local model files (sidecars
  such as PLE rows, drafts and hot sets are not listed). `Enter` on a local
  model makes it the default model; `p` pulls with `hipfire pull <tag> --yes`
  (a chooser offers PLE rows and draft sidecars when the entry has them), `c`
  cancels the pull, `d` deletes after a y/n.
- **Settings**: the typed global config (`~/.hipfire/config.toml`) grouped by
  intent (Quality, Speed, Memory & context, Speculation, Hardware & devices,
  Advanced), with *Using now* (the value predicted for the active model on
  this GPU, or serve's live KV mode / backend / context when its resident
  model is the active one), *From* (default, registry, you, `env HIPFIRE_*`,
  ...) and, at 160 columns, *Applies* plus a "why this value" pane. Keys:
  `1`-`6` / `[` `]` switch group, `/` searches every key (all printable keys
  go into the buffer; `Esc` restores the group and row), `c` changed only,
  `h` hide unused, `←/→/Space` preview an enum (`Enter` saves) or toggle a
  boolean, `Enter` edits a number or text, `Del` goes back to inherited, `R`
  resets all, `p` opens the Tuning presets (chooser, preview with measured
  trade-offs, `y` save / `Y` save and restart serve, `u` undo last tuning;
  state in `src/hipfire/tuning_view.rs`, keys in `src/app/tuning.rs`, screens
  in `src/ui_tuning.rs`). The header leads with `Tuning: <preset>`. Values an environment variable owns are locked. Every
  non-deprecated key is editable (editor kind from the schema's value rule);
  deprecated keys are read-only. Rows come from `SettingsModel`
  (`src/hipfire/settings_model.rs`), drawn by `src/ui_settings.rs`. Golden
  layouts live in `tests/snapshots/`; regenerate them with
  `UPDATE_SNAPSHOTS=1 cargo test -p hipfire-tui golden_`.
  **Per-model scope** (`src/hipfire/scope_view.rs`, `src/app/scope.rs`,
  `src/ui_scope.rs`): the header reads `Editing: all models  m`; `m` opens the
  scope picker (All models, the active model, every other installed model,
  each with its own-value count from `models.toml`). In a model's scope the
  group box reads `<Group> · values for <model> only`, rows resolve for that
  model, and writes go through `hipfire_config::set_model_override` /
  `remove_model_override`, the code behind `hipfire config <model> set` /
  `reset` (so `models.toml` is byte-identical either way). `Del` removes that
  model's value, `R` removes all of its values (y/n). Process- and
  Diagnostic-scoped keys, `developer.*` keys and the Prompt processing row read
  `all models only (locked)`; `←/→/Del/Enter` on them say why and write
  nothing. In the all-models scope a row the active model has its own value
  for reads `this model` and the explainer warns the change won't affect it.
  `o` opens **Per-model overrides** (also from the Models tab, focused on the
  selected model): every model × setting × value next to the all-models value;
  `Enter` opens Settings in that model's scope on the row, `Del` removes the
  value, `Esc` goes back. At 160×48 the same screen is a models list (with
  own-value counts and loaded / on disk state) beside that model's values, with
  `←→` choosing a value and `D` removing all of the model's values. The Models
  tab shows `N own values` before each model's notes.
- **System**: GPU, HIP and path diagnostics; `d` runs `hipfire diag`.
- **Logs**: the tail of `serve.log`.

The serve's `/health` is checked every 2 s on every tab. `?` shows the keys of
the current tab, Esc only closes or cancels, and `q` quits (asking first while a
pull is running).
