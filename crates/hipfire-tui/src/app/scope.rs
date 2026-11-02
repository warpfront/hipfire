// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Per-model scope on the Settings tab (spec §3.5): `m` picks what Settings
//! edits, `o` lists every model's own values, and edits in a model's scope
//! write `models.toml` through the same library functions as
//! `hipfire config <model> set` / `reset`.

use std::path::PathBuf;

use crossterm::event::{KeyCode, KeyEvent};
use hipfire_config::{
    effective::{Applies, ArchCtx, Origin},
    intent::intent_of,
    is_developer_key, ConfigPaths, RemoveModelOverride,
};
use hipfire_registry::RegistryV1;

use super::{App, Tab};
use crate::hipfire::{
    knobs,
    scope_view::{self, ScopeEntry, ScopeScreen, ScreenKind},
    settings_model::{registry_v1, value_string, RowId, SettingsRow},
};

/// Rows PgUp / PgDn move by.
const PAGE_ROWS: isize = 8;

/// The lines a scope changes in a row's explainer (spec §4 S5).
pub struct ScopeText {
    /// Replaces line 2: this model's value next to everyone else's.
    pub values: Option<String>,
    /// Replaces line 3.
    pub hint: String,
    /// Replaces line 4.
    pub keys: Option<String>,
}

fn applies_phrase(applies: Applies) -> &'static str {
    match applies {
        Applies::NextMessage => "applies to the next message",
        Applies::NewChat => "applies to the new chat",
        Applies::NextModelLoad => "applies at the next model load",
        Applies::ServeRestart => "applies after a serve restart",
    }
}

impl App {
    /// A scope screen (picker or overrides list) is open over the tabs.
    pub fn scope_screen_open(&self) -> bool {
        self.scope.screen.is_some()
    }

    /// The model Settings edits, `None` for all models.
    pub fn scope_model(&self) -> Option<&str> {
        self.scope.model.as_deref()
    }

    /// The model the Settings rows are predicted for and the context for it:
    /// the scope's model, else the active one.
    pub(crate) fn settings_subject(&self, registry: &RegistryV1) -> (String, ArchCtx) {
        match &self.scope.model {
            Some(model) => {
                let mut arch = self.arch.clone();
                let card = registry.model(model);
                arch.model_arch_id = card.and_then(|(_, entry)| entry.arch_id);
                arch.registry_tag = card.map(|(tag, _)| tag.to_owned());
                (model.clone(), arch)
            }
            None => (self.active_model.clone(), self.arch.clone()),
        }
    }

    /// The model serve has resident, when serve is online.
    fn resident_model(&self) -> Option<String> {
        let dashboard = self.dashboard.as_ref().filter(|dash| dash.serve_up)?;
        dashboard
            .stats
            .as_ref()?
            .loaded_config
            .as_ref()
            .map(|loaded| loaded.model.clone())
    }

    fn scope_entries(&self) -> Vec<ScopeEntry> {
        scope_view::entries(
            &self.scope.catalog,
            &self.registry,
            &self.active_model,
            self.resident_model().as_deref(),
        )
    }

    /// `models.toml` changed (or was re-read): bring an open scope screen up to
    /// date.
    pub(crate) fn refresh_scope_screen(&mut self) {
        if self.scope.screen.is_none() {
            return;
        }
        let entries = self.scope_entries();
        if let Some(screen) = self.scope.screen.as_mut() {
            screen.refresh(entries);
        }
    }

    /// `m`: choose what Settings edits.
    pub(crate) fn open_scope_picker(&mut self) {
        self.discard_settings_preview();
        let mut screen = ScopeScreen::new(ScreenKind::Picker, self.scope_entries());
        if let Some(model) = self.scope.model.clone() {
            screen.focus(&model);
        }
        self.scope.screen = Some(screen);
    }

    /// `o`: every model's own values, the cursor on `focus` (default: the
    /// model Settings is editing).
    pub(crate) fn open_overrides(&mut self, focus: Option<&str>) {
        self.discard_settings_preview();
        let mut screen = ScopeScreen::new(ScreenKind::Overrides, self.scope_entries());
        let model = focus
            .map(str::to_owned)
            .or_else(|| self.scope.model.clone());
        if let Some(model) = model {
            if !screen.focus(&model) {
                self.toast_info(format!("{model} is not in the model list"));
            } else if self.scope.own_count(&model) == 0 {
                self.toast_info(format!("{model} has no own values; it follows all models"));
            }
        }
        self.scope.screen = Some(screen);
    }

    /// Esc, a tab click: drop the open scope screen.
    pub(crate) fn close_scope_screen(&mut self) {
        self.scope.screen = None;
    }

    /// Settings edits `model` (`None`: all models) from now on.
    pub(crate) fn set_scope(&mut self, model: Option<String>) {
        self.discard_settings_preview();
        self.settings_search = None;
        self.scope.model = model;
        self.rebuild_settings();
        self.last_reload = match &self.scope.model {
            Some(model) => format!("Editing {model}: values for this model only"),
            None => "Editing all models".to_owned(),
        };
    }

    /// Put the Settings cursor on the row of `key`, showing its group.
    fn select_settings_key(&mut self, key: &str) {
        self.settings_group = intent_of(key);
        self.settings_changed_only = false;
        self.settings_hide_unused = false;
        self.settings_search = None;
        let at = self
            .settings_visible()
            .iter()
            .position(|row| row.id.key() == Some(key));
        self.settings_selected = at.unwrap_or(0);
    }

    /// Enter: leave for Settings in `model`'s scope, on `key` when given.
    fn jump_to_settings(&mut self, model: Option<String>, key: Option<String>) {
        self.scope.screen = None;
        self.tab = Tab::Settings;
        self.set_scope(model);
        match key {
            Some(key) => self.select_settings_key(&key),
            None => self.settings_selected = 0,
        }
    }

    pub(crate) fn handle_scope_key(&mut self, key: KeyEvent) {
        let Some(screen) = self.scope.screen.as_mut() else {
            return;
        };
        if screen.confirm_remove_all {
            match key.code {
                KeyCode::Char('y') | KeyCode::Char('Y') => self.remove_all_of_selected(),
                KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => {
                    screen.confirm_remove_all = false;
                    self.last_reload = "remove cancelled".into();
                }
                _ => {}
            }
            return;
        }
        // The picker popup at 80 columns lists models only: no value is on
        // screen to remove or step through.
        let values_shown = screen.kind == ScreenKind::Overrides || screen.wide.get();
        match key.code {
            KeyCode::Esc => self.close_scope_screen(),
            KeyCode::Down | KeyCode::Char('j') => screen.move_selection(1),
            KeyCode::Up | KeyCode::Char('k') => screen.move_selection(-1),
            KeyCode::PageDown => screen.move_selection(PAGE_ROWS),
            KeyCode::PageUp => screen.move_selection(-PAGE_ROWS),
            KeyCode::Home => screen.jump_to_end(false),
            KeyCode::End => screen.jump_to_end(true),
            KeyCode::Right if values_shown && !screen.is_flat() => screen.move_value(1),
            KeyCode::Left if values_shown && !screen.is_flat() => screen.move_value(-1),
            KeyCode::Enter => self.open_selected_in_settings(),
            KeyCode::Delete | KeyCode::Backspace if values_shown => self.remove_selected_value(),
            KeyCode::Char('D') if values_shown => self.arm_remove_all(),
            KeyCode::Delete | KeyCode::Backspace | KeyCode::Char('D') => {
                self.toast_info("o lists a model's own values; Del removes one there");
            }
            _ => {}
        }
    }

    fn open_selected_in_settings(&mut self) {
        let Some(screen) = self.scope.screen.as_ref() else {
            return;
        };
        let Some(entry) = screen.selected_entry() else {
            return;
        };
        let model = entry.model.clone();
        // Only the overrides list points at a value; the picker just changes
        // what Settings edits.
        let key = (screen.kind == ScreenKind::Overrides)
            .then(|| screen.selected_value().map(|own| own.key.clone()))
            .flatten();
        self.jump_to_settings(model, key);
    }

    fn arm_remove_all(&mut self) {
        let Some(screen) = self.scope.screen.as_mut() else {
            return;
        };
        let Some(entry) = screen.selected_entry() else {
            return;
        };
        let (name, count) = (entry.name().to_owned(), entry.own.len());
        if entry.model.is_none() || count == 0 {
            self.toast_info(format!("{name} has no own values to remove"));
            return;
        }
        screen.confirm_remove_all = true;
        self.last_reload =
            format!("remove all {count} own values of {name}? y to confirm, n / Esc to cancel");
    }

    /// Del on a scope screen: remove the selected own value.
    fn remove_selected_value(&mut self) {
        let Some(screen) = self.scope.screen.as_ref() else {
            return;
        };
        let Some((Some(model), Some(own))) = screen
            .selected_entry()
            .map(|entry| (entry.model.clone(), screen.selected_value().cloned()))
        else {
            self.toast_info("nothing to remove here");
            return;
        };
        match hipfire_config::remove_model_override(&self.config_paths(), &model, Some(&own.key)) {
            Ok(_) => {
                self.after_config_write();
                self.last_reload = format!("{model} {} removed; it follows all models", own.key);
                self.toast_info(format!("{model}: {} removed", own.key));
            }
            Err(err) => {
                self.last_reload = format!("remove failed: {err}");
                self.toast_error(format!("remove failed: {err}"));
            }
        }
    }

    /// `y` after `D`: remove every own value of the selected model.
    fn remove_all_of_selected(&mut self) {
        let Some(screen) = self.scope.screen.as_mut() else {
            return;
        };
        screen.confirm_remove_all = false;
        let Some(model) = screen
            .selected_entry()
            .and_then(|entry| entry.model.clone())
        else {
            return;
        };
        self.remove_all_for_model(&model);
    }

    /// Remove every own value of `model` (the `R` confirm in Settings and `D`
    /// on the overrides list).
    pub(crate) fn remove_all_for_model(&mut self, model: &str) {
        match hipfire_config::remove_model_override(&self.config_paths(), model, None) {
            Ok(outcome) => {
                self.after_config_write();
                self.last_reload = match outcome {
                    RemoveModelOverride::NoOverrides => format!("{model} had no own values"),
                    _ => format!("all {model} own values removed; it follows all models"),
                };
                self.toast_info(format!("{model}: own values removed"));
            }
            Err(err) => {
                self.last_reload = format!("remove failed: {err}");
                self.toast_error(format!("remove failed: {err}"));
            }
        }
    }

    fn config_paths(&self) -> ConfigPaths {
        ConfigPaths::under(&self.paths.root)
    }

    /// The file of `model` for the `models.toml` record: its catalog path, else
    /// the registry file when it is downloaded.
    fn model_local_path(&self, registry: &RegistryV1, model: &str) -> Option<PathBuf> {
        if let Some((_, record)) = self.scope.catalog.model(model) {
            if let Some(path) = record.path.as_ref().filter(|path| path.is_file()) {
                return Some(path.clone());
            }
        }
        let (_, entry) = registry.model(model)?;
        let path = self.paths.models.join(&entry.file);
        path.is_file().then_some(path)
    }

    /// Save a value for `model` in `models.toml`. Returns `(ok, status)`, like
    /// the global write.
    pub(crate) fn persist_model_setting(
        &mut self,
        model: &str,
        key: &str,
        raw: &str,
    ) -> (bool, String) {
        let registry = registry_v1(&self.registry);
        let tag = registry.model(model).map(|(tag, _)| tag.to_owned());
        let local = self.model_local_path(&registry, model);
        match hipfire_config::set_model_override(
            &self.config_paths(),
            tag.as_deref(),
            local,
            model,
            key,
            raw,
        ) {
            Ok(saved) => {
                self.after_config_write();
                self.toast_info(format!("{key} saved for {model}"));
                (
                    true,
                    format!(
                        "{key} = {} saved for {model} in ~/.hipfire/models.toml",
                        value_string(&saved.value)
                    ),
                )
            }
            Err(err) => {
                self.toast_error(format!("save failed: {err}"));
                (false, format!("{err}"))
            }
        }
    }

    /// Why `row` has no per-model value.
    fn lock_why(row: &SettingsRow) -> &'static str {
        if row.keys.iter().any(|key| is_developer_key(key)) {
            "developer settings are global"
        } else {
            "one value per serve process"
        }
    }

    /// Why `row` has no per-model value, in model scope; `None` when it can be
    /// edited (or Settings edits all models).
    pub(crate) fn scope_lock_reason(&self, row: &SettingsRow) -> Option<String> {
        self.scope.model.as_ref()?;
        row.model_scope_lock()?;
        Some(format!(
            "{}: {}, so it is all models only",
            row.title,
            Self::lock_why(row)
        ))
    }

    /// Del in model scope: the selected row follows "all models" again.
    pub(crate) fn reset_selected_for_model(&mut self, model: &str) {
        let Some(row) = self.selected_row().cloned() else {
            self.last_reload = "this row has no resettable setting".into();
            self.toast_info("nothing to reset on this row");
            return;
        };
        if let Some(reason) = self.scope_lock_reason(&row) {
            self.last_reload = reason.clone();
            self.toast_info(reason);
            return;
        }
        if let Some(lock) = row.locked.clone() {
            self.locked_notice(&lock);
            return;
        }
        let RowId::Key(key) = &row.id else {
            return;
        };
        if !self.scope.holds(model, key) {
            self.last_reload = format!("{key} already follows all models for {model}");
            self.toast_info(format!("{key}: {model} has no value of its own"));
            return;
        }
        match hipfire_config::remove_model_override(&self.config_paths(), model, Some(key)) {
            Ok(_) => {
                let id = Some(row.id.clone());
                self.after_config_write();
                self.restore_selection(id);
                let now = self
                    .settings
                    .row(&row.id)
                    .map_or_else(String::new, |now| now.display.clone());
                self.last_reload = format!("{key} follows all models again ({now}) for {model}");
                self.toast_info(format!("{key} follows all models"));
            }
            Err(err) => {
                self.last_reload = format!("{key} reset failed: {err}");
                self.toast_error(format!("reset failed: {err}"));
            }
        }
    }

    /// What the scope changes in `row`'s explainer, if anything.
    pub fn scope_text(&self, row: &SettingsRow) -> Option<ScopeText> {
        let key = row.id.key();
        match self.scope.model.as_deref() {
            Some(model) => {
                if self.scope_lock_reason(row).is_some() {
                    return Some(ScopeText {
                        values: None,
                        hint: format!(
                            "All models only: {}. m edits all models.",
                            Self::lock_why(row)
                        ),
                        keys: None,
                    });
                }
                if row.locked.is_some() {
                    return None;
                }
                let key = key?;
                let mine = crate::hipfire::settings_model::value_text(key, &row.resolved);
                let all = self.scope.all_models_value(key);
                let others = |text: &str| {
                    all.as_ref()
                        .map(|(value, source)| format!("{text}{value} ({source})."))
                };
                let own = self.scope.holds(model, key);
                let values = if own {
                    let head = format!("{model}: {mine} (this model).");
                    match others("All other models: ") {
                        Some(tail) => format!("{head} {tail}"),
                        None => head,
                    }
                } else {
                    others(&format!("{model} follows all models: "))
                        .unwrap_or_else(|| format!("{model}: {mine} (follows all models)."))
                };
                let hint = if own {
                    "Del removes this model's value so it follows \"all models\" again.".to_owned()
                } else {
                    format!("Change it to give {model} its own value.")
                };
                Some(ScopeText {
                    values: Some(values),
                    hint,
                    keys: Some(format!(
                        "{key} · {} · m edit all models",
                        applies_phrase(row.effective.applies)
                    )),
                })
            }
            None => {
                if row.locked.is_none() && matches!(row.origin, Origin::ThisModel) {
                    return Some(ScopeText {
                        values: None,
                        hint: format!(
                            "{} has its own value, so this change won't affect it. m to edit it.",
                            self.active_model
                        ),
                        keys: None,
                    });
                }
                None
            }
        }
    }

    /// The four-line explainer of `row` with the scope's lines applied.
    pub fn row_explainer(&self, row: &SettingsRow) -> [String; 4] {
        let mut lines = self.settings.explainer(
            row,
            self.live_value(row).as_deref(),
            self.row_needs_restart(row),
        );
        if let Some(text) = self.scope_text(row) {
            if let Some(values) = text.values {
                lines[1] = knobs::fit(values);
            }
            lines[2] = knobs::fit(text.hint);
            if let Some(keys) = text.keys {
                lines[3] = knobs::fit(keys);
            }
        }
        lines
    }
}

#[cfg(test)]
mod tests {
    use crossterm::event::KeyCode;
    use hipfire_config::effective::A8Prefill;

    use crate::{
        app::Tab,
        hipfire::scope_view::ScreenKind,
        test_support::{select_key, Fixture, FixtureBuilder, ACTIVE_MODEL},
    };

    use super::*;

    /// A key as the running UI routes it: an open scope screen first.
    fn press(app: &mut App, code: KeyCode) {
        app.handle_tab_key(crossterm::event::KeyEvent::new(
            code,
            crossterm::event::KeyModifiers::NONE,
        ));
    }

    const LOW_EFFORT: &str = "[reasoning]\neffort = \"low\"\n";

    fn strix() -> FixtureBuilder {
        FixtureBuilder::new("gfx1151")
            .a8(A8Prefill::OptIn)
            .config(LOW_EFFORT)
    }

    /// `m`, the active model, Enter: Settings now edits `ACTIVE_MODEL` only.
    fn enter_model_scope(fixture: &mut Fixture) {
        press(&mut fixture.app, KeyCode::Char('m'));
        press(&mut fixture.app, KeyCode::Down);
        press(&mut fixture.app, KeyCode::Enter);
        assert_eq!(fixture.app.scope_model(), Some(ACTIVE_MODEL));
    }

    fn toast(app: &App) -> String {
        app.toast
            .as_ref()
            .map(|toast| toast.text.clone())
            .unwrap_or_default()
    }

    fn own_keys(fixture: &Fixture, model: &str) -> Vec<String> {
        let loaded = hipfire_config::load_catalog(&ConfigPaths::under(fixture.root())).unwrap();
        loaded
            .catalog
            .model(model)
            .map(|(_, record)| record.overrides.values.keys().cloned().collect())
            .unwrap_or_default()
    }

    #[test]
    fn m_opens_the_picker_and_enter_scopes_settings_to_a_model() {
        let mut fixture = strix()
            .model_value(ACTIVE_MODEL, "reasoning.effort", "high")
            .model("qwen3.5:9b", true)
            .build();
        press(&mut fixture.app, KeyCode::Char('m'));
        let screen = fixture.app.scope.screen.as_ref().expect("picker open");
        assert_eq!(screen.kind, ScreenKind::Picker);
        let names: Vec<&str> = screen.entries.iter().map(|entry| entry.name()).collect();
        assert_eq!(names, ["All models", ACTIVE_MODEL, "qwen3.5:9b"]);
        assert_eq!(screen.entries[1].own.len(), 1);
        // Esc leaves the scope alone.
        press(&mut fixture.app, KeyCode::Esc);
        assert!(fixture.app.scope.screen.is_none());
        assert_eq!(fixture.app.scope_model(), None);
        enter_model_scope(&mut fixture);
        assert!(fixture.app.scope.screen.is_none());
        let effort = fixture
            .app
            .settings
            .row(&RowId::Key("reasoning.effort".into()))
            .unwrap();
        assert_eq!(effort.origin, Origin::ThisModel);
        assert_eq!(effort.display, "high");
    }

    #[test]
    fn model_scope_locks_process_and_developer_rows_and_writes_nothing() {
        let mut fixture = strix()
            .model_value(ACTIVE_MODEL, "reasoning.effort", "high")
            .build();
        enter_model_scope(&mut fixture);
        let before = fixture.models_toml_bytes();
        assert!(!before.is_empty());
        let config_before = fixture.config_bytes();

        let locked: Vec<RowId> = fixture
            .app
            .settings
            .rows
            .iter()
            .filter(|row| row.model_scope_lock().is_some())
            .map(|row| row.id.clone())
            .collect();
        assert!(locked.contains(&RowId::Virtual(
            crate::hipfire::settings_model::VirtualRow::PromptProcessing
        )));
        assert!(locked.contains(&RowId::Key("developer.qwen4_moe_sym_iu4".into())));
        assert!(
            locked
                .iter()
                .any(|id| matches!(id, RowId::Key(key) if !key.starts_with("developer."))),
            "a Process-scoped key row is locked too"
        );
        for id in locked {
            crate::test_support::select_row(&mut fixture.app, &id);
            let row = fixture.app.selected_row().unwrap().clone();
            assert_eq!(
                fixture.app.scope_lock_reason(&row).is_some(),
                true,
                "{:?}",
                row.id
            );
            for code in [
                KeyCode::Right,
                KeyCode::Left,
                KeyCode::Delete,
                KeyCode::Enter,
            ] {
                fixture.app.toast = None;
                press(&mut fixture.app, code);
                let text = toast(&fixture.app);
                assert!(
                    text.contains("one value per serve process")
                        || text.contains("developer settings are global"),
                    "{:?} {code:?}: {text}",
                    row.id
                );
                assert!(
                    fixture.app.settings_pending.is_none(),
                    "{:?} {code:?}",
                    row.id
                );
            }
            assert_eq!(fixture.models_toml_bytes(), before, "{:?}", row.id);
            assert_eq!(fixture.config_bytes(), config_before, "{:?}", row.id);
        }
    }

    /// The bytes `hipfire config qwen3.8:27b set reasoning.effort high` leaves:
    /// the CLI calls `set_model_override(paths, registry tag, local path,
    /// model, key, raw)` with exactly these arguments for a registry model
    /// that is not downloaded.
    fn cli_models_toml() -> Vec<u8> {
        let fixture = strix().build();
        hipfire_config::set_model_override(
            &ConfigPaths::under(fixture.root()),
            Some(ACTIVE_MODEL),
            None,
            ACTIVE_MODEL,
            "reasoning.effort",
            "high",
        )
        .unwrap();
        fixture.models_toml_bytes()
    }

    #[test]
    fn model_scope_write_matches_the_cli_models_toml_bytes() {
        let cli = cli_models_toml();
        assert!(String::from_utf8_lossy(&cli).contains("effort = \"high\""));

        let mut fixture = strix().build();
        let config_before = fixture.config_bytes();
        assert!(fixture.models_toml_bytes().is_empty());
        enter_model_scope(&mut fixture);
        select_key(&mut fixture.app, "reasoning.effort");
        for _ in 0..8 {
            if fixture
                .app
                .settings_pending
                .as_ref()
                .is_some_and(|pending| pending.value == "high")
            {
                break;
            }
            press(&mut fixture.app, KeyCode::Right);
        }
        press(&mut fixture.app, KeyCode::Enter);
        assert_eq!(fixture.models_toml_bytes(), cli);
        assert_eq!(
            fixture.config_bytes(),
            config_before,
            "config.toml untouched"
        );
        let effort = fixture
            .app
            .settings
            .row(&RowId::Key("reasoning.effort".into()))
            .unwrap();
        assert_eq!(
            (effort.display.as_str(), effort.origin.clone()),
            ("high", Origin::ThisModel)
        );
    }

    #[test]
    fn del_removes_only_the_selected_models_value() {
        let mut fixture = strix()
            .model_value(ACTIVE_MODEL, "reasoning.effort", "high")
            .model_value(ACTIVE_MODEL, "memory.max_seq", "131072")
            .model_value("qwen3.5:9b", "reasoning.effort", "medium")
            .model("qwen3.5:9b", true)
            .build();
        let config_before = fixture.config_bytes();
        enter_model_scope(&mut fixture);
        select_key(&mut fixture.app, "reasoning.effort");
        press(&mut fixture.app, KeyCode::Delete);
        assert_eq!(own_keys(&fixture, ACTIVE_MODEL), ["memory.max_seq"]);
        assert_eq!(own_keys(&fixture, "qwen3.5:9b"), ["reasoning.effort"]);
        assert_eq!(fixture.config_bytes(), config_before);
        let effort = fixture
            .app
            .settings
            .row(&RowId::Key("reasoning.effort".into()))
            .unwrap();
        assert_eq!(
            (effort.display.as_str(), effort.origin.clone()),
            ("low", Origin::You)
        );
        // Nothing left to remove on that row.
        fixture.app.toast = None;
        press(&mut fixture.app, KeyCode::Delete);
        assert!(
            toast(&fixture.app).contains("no value of its own"),
            "{}",
            toast(&fixture.app)
        );
    }

    #[test]
    fn r_in_model_scope_removes_all_of_that_models_values_after_a_confirm() {
        let mut fixture = strix()
            .model_value(ACTIVE_MODEL, "reasoning.effort", "high")
            .model_value(ACTIVE_MODEL, "memory.max_seq", "131072")
            .model_value("qwen3.5:9b", "reasoning.effort", "medium")
            .model("qwen3.5:9b", true)
            .build();
        enter_model_scope(&mut fixture);
        press(&mut fixture.app, KeyCode::Char('R'));
        assert!(fixture.app.confirm_reset_all);
        assert_eq!(
            own_keys(&fixture, ACTIVE_MODEL).len(),
            2,
            "nothing removed yet"
        );
        press(&mut fixture.app, KeyCode::Char('n'));
        assert_eq!(own_keys(&fixture, ACTIVE_MODEL).len(), 2);
        press(&mut fixture.app, KeyCode::Char('R'));
        press(&mut fixture.app, KeyCode::Char('y'));
        assert!(own_keys(&fixture, ACTIVE_MODEL).is_empty());
        assert_eq!(own_keys(&fixture, "qwen3.5:9b"), ["reasoning.effort"]);
    }

    #[test]
    fn all_models_scope_marks_this_model_and_warns_that_the_change_will_not_affect_it() {
        let mut fixture = strix()
            .model_value(ACTIVE_MODEL, "reasoning.effort", "high")
            .build();
        select_key(&mut fixture.app, "reasoning.effort");
        let row = fixture.app.selected_row().unwrap().clone();
        assert_eq!(row.origin.label(), "this model");
        let lines = fixture.app.row_explainer(&row);
        assert_eq!(
            lines[2],
            format!(
                "{ACTIVE_MODEL} has its own value, so this change won't affect it. m to edit it."
            )
        );
        // A row the model holds nothing for keeps its own explainer.
        select_key(&mut fixture.app, "reasoning.budget");
        let other = fixture.app.selected_row().unwrap().clone();
        assert!(!fixture.app.row_explainer(&other)[2].contains("its own value"));
    }

    #[test]
    fn model_scope_explainer_shows_both_values() {
        let mut fixture = strix()
            .model_value(ACTIVE_MODEL, "reasoning.effort", "high")
            .build();
        enter_model_scope(&mut fixture);
        select_key(&mut fixture.app, "reasoning.effort");
        let row = fixture.app.selected_row().unwrap().clone();
        let lines = fixture.app.row_explainer(&row);
        assert_eq!(
            lines[1],
            "qwen3.8:27b: high (this model). All other models: low (you, config.toml)."
        );
        assert_eq!(
            lines[2],
            "Del removes this model's value so it follows \"all models\" again."
        );
        assert_eq!(
            lines[3],
            "reasoning.effort · applies to the next message · m edit all models"
        );
    }

    #[test]
    fn o_lists_every_models_values_and_enter_opens_that_model_in_settings() {
        let mut fixture = strix()
            .model_value(ACTIVE_MODEL, "reasoning.effort", "high")
            .model_value("gemma4:12b", "speculation.mode", "off")
            .build();
        press(&mut fixture.app, KeyCode::Char('o'));
        let screen = fixture.app.scope.screen.as_ref().expect("overrides open");
        assert_eq!(screen.kind, ScreenKind::Overrides);
        assert_eq!(screen.totals(), (2, 2));
        // Down to the gemma4 row, Enter jumps to Settings on its key.
        press(&mut fixture.app, KeyCode::Down);
        press(&mut fixture.app, KeyCode::Enter);
        assert!(fixture.app.scope.screen.is_none());
        assert_eq!(fixture.app.tab, Tab::Settings);
        assert_eq!(fixture.app.scope_model(), Some("gemma4:12b"));
        assert_eq!(
            fixture.app.selected_row().and_then(|row| row.id.key()),
            Some("speculation.mode")
        );
    }

    #[test]
    fn del_on_the_overrides_list_removes_that_value_only() {
        let mut fixture = strix()
            .model_value(ACTIVE_MODEL, "reasoning.effort", "high")
            .model_value("gemma4:12b", "speculation.mode", "off")
            .build();
        press(&mut fixture.app, KeyCode::Char('o'));
        press(&mut fixture.app, KeyCode::Delete);
        assert!(
            own_keys(&fixture, ACTIVE_MODEL).is_empty()
                != own_keys(&fixture, "gemma4:12b").is_empty()
        );
        let screen = fixture.app.scope.screen.as_ref().expect("still open");
        assert_eq!(screen.totals(), (1, 1));
    }

    #[test]
    fn models_tab_o_opens_the_overrides_list_on_the_selected_model() {
        let mut fixture = strix()
            .model_value(ACTIVE_MODEL, "reasoning.effort", "high")
            .model_value("qwen3.5:9b", "reasoning.effort", "medium")
            .model("qwen3.5:9b", true)
            .build();
        fixture.app.tab = Tab::Models;
        fixture
            .app
            .registry
            .expanded_groups
            .insert("qwen3.5".into());
        let items = fixture.app.registry.visible_items();
        let at = items
            .iter()
            .position(|item| matches!(item, crate::hipfire::registry::ModelListItem::Model { model_index } if fixture.app.registry.models[*model_index].tag == "qwen3.5:9b"))
            .expect("model row visible");
        fixture.app.registry.selected = at;
        press(&mut fixture.app, KeyCode::Char('o'));
        let screen = fixture.app.scope.screen.as_ref().expect("overrides open");
        assert_eq!(screen.kind, ScreenKind::Overrides);
        assert_eq!(
            screen
                .selected_entry()
                .and_then(|entry| entry.model.as_deref()),
            Some("qwen3.5:9b")
        );
        assert_eq!(
            fixture.app.tab,
            Tab::Models,
            "Esc returns to the Models tab"
        );
    }

    #[test]
    fn the_picker_popup_removes_nothing() {
        let mut fixture = strix()
            .model_value(ACTIVE_MODEL, "reasoning.effort", "high")
            .build();
        let before = fixture.models_toml_bytes();
        press(&mut fixture.app, KeyCode::Char('m'));
        press(&mut fixture.app, KeyCode::Down);
        for code in [KeyCode::Delete, KeyCode::Char('D'), KeyCode::Right] {
            press(&mut fixture.app, code);
        }
        assert!(fixture.app.scope_screen_open());
        assert!(
            !fixture
                .app
                .scope
                .screen
                .as_ref()
                .unwrap()
                .confirm_remove_all
        );
        assert_eq!(fixture.models_toml_bytes(), before);
    }
}
