// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! The Settings tab's state and keys (spec §3): the grouped list, `/` search,
//! the `c` / `h` filters, enum previews, locks, resets and the live-from-serve
//! and restart-to-apply helpers the renderer reads.

use std::time::{Duration, SystemTime};

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use hipfire_config::{
    effective::{Applies, Origin},
    intent::Intent,
    ConfigValue,
};

use super::{deprecated_status, not_editable_status, App, EditState, PendingEnum};
use crate::hipfire::{
    dashboard::LoadedConfig,
    knobs::PROMPT_PROCESSING_ID,
    settings_model::{live_text, value_string, RowFilter, RowId, SettingsRow, VirtualRow},
    writer::{self, FieldKind},
};

/// Rows PgUp / PgDn move by.
const PAGE_ROWS: usize = 8;

/// The `/` search: its buffer, whether the buffer has the keyboard, and where
/// Esc returns to.
#[derive(Clone, Debug)]
pub struct SettingsSearch {
    pub query: String,
    /// The buffer has focus: every printable key (q, r, ? included) goes in.
    pub typing: bool,
    prev_group: Intent,
    prev_selected: usize,
}

impl App {
    /// Search and filters currently applied to the Settings rows.
    pub fn settings_filter(&self) -> RowFilter {
        RowFilter {
            search: self
                .settings_search
                .as_ref()
                .map(|search| search.query.clone())
                .unwrap_or_default(),
            changed_only: self.settings_changed_only,
            hide_unused: self.settings_hide_unused,
        }
    }

    /// The rows on screen: every group's matches while a search is open, else
    /// the current group's rows.
    pub fn settings_visible(&self) -> Vec<&SettingsRow> {
        let filter = self.settings_filter();
        if self.settings_search.is_some() {
            self.settings.search(&filter)
        } else {
            self.settings.visible(self.settings_group, &filter)
        }
    }

    pub fn selected_row(&self) -> Option<&SettingsRow> {
        self.settings_visible().get(self.settings_selected).copied()
    }

    /// The search buffer owns the keyboard.
    pub fn settings_search_typing(&self) -> bool {
        self.settings_search
            .as_ref()
            .is_some_and(|search| search.typing)
    }

    /// Switch the model the Settings rows predict for: its registry card
    /// refreshes the GPU/model context, then the rows rebuild.
    pub(crate) fn set_active_model(&mut self, tag: &str) {
        self.active_model = tag.to_owned();
        let registry = crate::hipfire::settings_model::registry_v1(&self.registry);
        let card = registry.model(tag);
        self.arch.model_arch_id = card.and_then(|(_, entry)| entry.arch_id);
        self.arch.registry_tag = card.map(|(tag, _)| tag.to_owned());
        self.rebuild_settings();
    }

    /// serve's `/stats` `loaded_config`, when serve is online and its resident
    /// model is the active model.
    pub fn serve_loaded_config(&self) -> Option<&LoadedConfig> {
        // Settings edits another model: serve's resident one is not its subject.
        if self
            .scope
            .model
            .as_deref()
            .is_some_and(|model| model != self.active_model)
        {
            return None;
        }
        let dashboard = self.dashboard.as_ref().filter(|dash| dash.serve_up)?;
        let loaded = dashboard.stats.as_ref()?.loaded_config.as_ref()?;
        let active = loaded.model == self.active_model
            || self.arch.registry_tag.as_deref() == Some(loaded.model.as_str());
        active.then_some(loaded)
    }

    /// What the resident model reports for `row` (KV mode, KV backend,
    /// context), for the rows that have a live value.
    pub fn live_value(&self, row: &SettingsRow) -> Option<String> {
        live_text(row.id.key()?, self.serve_loaded_config()?)
    }

    /// `config.toml` was written after the running serve started.
    pub fn config_restart_pending(&self) -> bool {
        let Some(stats) = self
            .dashboard
            .as_ref()
            .filter(|dash| dash.serve_up)
            .and_then(|dash| dash.stats.as_ref())
        else {
            return false;
        };
        let Some(started) = SystemTime::now().checked_sub(Duration::from_secs(stats.uptime_s))
        else {
            return false;
        };
        // `models.toml` counts too: a model's own values load with the model.
        [&self.paths.config, &self.paths.models_catalog]
            .into_iter()
            .filter_map(|path| {
                std::fs::metadata(path)
                    .and_then(|meta| meta.modified())
                    .ok()
            })
            .any(|modified| modified > started)
    }

    /// `row` is read at load or start and `config.toml` is newer than serve.
    pub fn row_needs_restart(&self, row: &SettingsRow) -> bool {
        matches!(
            row.effective.applies,
            Applies::ServeRestart | Applies::NextModelLoad
        ) && self.config_restart_pending()
    }

    pub(crate) fn clamp_settings_selection(&mut self) {
        let len = self.settings_visible().len();
        self.settings_selected = self.settings_selected.min(len.saturating_sub(1));
    }

    /// Put the cursor back on `id` (it may have moved or left the list).
    pub(super) fn restore_selection(&mut self, id: Option<RowId>) {
        let position =
            id.and_then(|id| self.settings_visible().iter().position(|row| row.id == id));
        match position {
            Some(position) => self.settings_selected = position,
            None => self.clamp_settings_selection(),
        }
    }

    /// A config write landed: rebuild the rows and the dashboard's view of
    /// host/port, and keep the cursor on its row.
    pub(super) fn after_config_write(&mut self) {
        let id = self.selected_row().map(|row| row.id.clone());
        self.rebuild_settings();
        self.config = self.load_config();
        self.restore_selection(id);
        self.dashboard_worker.update_config(self.config.clone());
        self.dashboard_worker.force_refresh();
    }

    /// Persist a validated value for `key`. Returns `(ok, status)`; `ok` is
    /// false on a validation or write error so the editor keeps its buffer.
    fn persist_setting(&mut self, key: &str, raw: &str) -> (bool, String) {
        if let Some(model) = self.scope.model.clone() {
            return self.persist_model_setting(&model, key, raw);
        }
        match writer::write_value(&self.paths.config, key, raw) {
            Ok(value) => {
                let shown = match &value {
                    serde_json::Value::String(text) => text.clone(),
                    serde_json::Value::Null => "(unset)".to_owned(),
                    other => other.to_string(),
                };
                self.after_config_write();
                self.toast_info(format!("{key} saved"));
                (
                    true,
                    format!("{key} = {shown} saved to ~/.hipfire/config.toml"),
                )
            }
            Err(err) => {
                self.toast_error(format!("save failed: {err}"));
                (false, format!("{err}"))
            }
        }
    }

    /// An environment variable owns the row: say so, write nothing.
    pub(super) fn locked_notice(&mut self, text: &str) {
        self.last_reload = text.to_owned();
        self.toast_info(text.to_owned());
    }

    pub(super) fn settings_goto_group(&mut self, group: Intent) {
        self.discard_settings_preview();
        self.settings_search = None;
        self.settings_group = group;
        self.settings_selected = 0;
    }

    fn settings_step_group(&mut self, forward: bool) {
        let all = Intent::ALL;
        let at = all
            .iter()
            .position(|group| *group == self.settings_group)
            .unwrap_or(0);
        let next = if forward {
            (at + 1) % all.len()
        } else {
            (at + all.len() - 1) % all.len()
        };
        self.settings_goto_group(all[next]);
    }

    fn open_settings_search(&mut self) {
        self.discard_settings_preview();
        match self.settings_search.as_mut() {
            Some(search) => search.typing = true,
            None => {
                self.settings_search = Some(SettingsSearch {
                    query: String::new(),
                    typing: true,
                    prev_group: self.settings_group,
                    prev_selected: self.settings_selected,
                });
                self.settings_selected = 0;
            }
        }
    }

    /// Drop the search; `restore` returns to the group and row it started from.
    fn close_settings_search(&mut self, restore: bool) {
        if let Some(search) = self.settings_search.take() {
            if restore {
                self.settings_group = search.prev_group;
                self.settings_selected = search.prev_selected;
            }
        }
        self.clamp_settings_selection();
    }

    fn move_settings_selection(&mut self, to: usize) {
        self.discard_settings_preview();
        let len = self.settings_visible().len();
        self.settings_selected = to.min(len.saturating_sub(1));
    }

    /// Reset the selected setting to its inherited value by removing it from
    /// config.toml. Single-key reset is reversible, so it needs no confirm.
    fn reset_selected_setting(&mut self) {
        if let Some(model) = self.scope.model.clone() {
            self.reset_selected_for_model(&model);
            return;
        }
        // A corrupt config.toml loads as all-defaults, so a per-key reset would
        // wrongly say "already default" (and the writer refuses to clobber it).
        // R (reset-all) is the recovery hatch.
        if let Some(warn) = self.config.warning.clone() {
            self.last_reload = format!("config.toml unreadable ({warn}); press R to reset all");
            self.toast_error("config.toml unreadable — use R (reset-all) to recover");
            return;
        }
        let Some(row) = self.selected_row().cloned() else {
            self.last_reload = "this row has no resettable setting".into();
            self.toast_info("nothing to reset on this row");
            return;
        };
        if let Some(lock) = row.locked.clone() {
            self.locked_notice(&lock);
            return;
        }
        match &row.id {
            RowId::Virtual(_) => self.reset_virtual_row(&row),
            RowId::Key(key) => self.reset_key_row(&row, key),
        }
    }

    fn reset_virtual_row(&mut self, row: &SettingsRow) {
        let title = row.title.clone();
        let Some(default) = row.choices.iter().find(|choice| choice.builtin_default) else {
            self.last_reload = format!("{title} is not used on this GPU");
            self.toast_info("nothing to reset on this row");
            return;
        };
        if !row.origin.marked() {
            self.last_reload = format!("{title} is already at its default ({})", row.display);
            self.toast_info(format!("{title} already default"));
            return;
        }
        match writer::write_choice(&self.paths.config, default) {
            Ok(()) => {
                self.after_config_write();
                self.last_reload = format!("{title} reset to default ({})", default.label);
                self.toast_info(format!("{title} reset to default"));
            }
            Err(err) => {
                self.last_reload = format!("{title} reset failed: {err}");
                self.toast_error(format!("reset failed: {err}"));
            }
        }
    }

    fn reset_key_row(&mut self, row: &SettingsRow, key: &str) {
        match &row.origin {
            Origin::Registry => {
                self.last_reload = format!("{key} comes from the model card");
                self.toast_info("nothing to reset (comes from the model card)");
                return;
            }
            Origin::Default | Origin::DefaultArch(_) => {
                self.last_reload = format!("{key} is already at its default ({})", row.display);
                self.toast_info(format!("{key} already default"));
                return;
            }
            _ => {}
        }
        match writer::delete_key(&self.paths.config, key) {
            Ok(true) => {
                let was = row.display.clone();
                self.after_config_write();
                let now = self
                    .settings
                    .row(&row.id)
                    .map_or_else(String::new, |row| row.display.clone());
                self.last_reload = format!("{key} reset to default ({now}); was {was}");
                self.toast_info(format!("{key} reset to default"));
            }
            Ok(false) => {
                self.last_reload = format!("{key} is already at its default ({})", row.display);
                self.toast_info(format!("{key} already default"));
            }
            Err(err) => {
                self.last_reload = format!("{key} reset failed: {err}");
                self.toast_error(format!("reset failed: {err}"));
            }
        }
    }

    /// Arm the "reset ALL config to defaults" confirmation. Wiping the whole
    /// file is destructive, so it waits for an explicit y.
    fn arm_reset_all(&mut self) {
        self.confirm_reset_all = true;
        self.last_reload = match self.scope.model.as_deref() {
            Some(model) => {
                format!("remove ALL of {model}'s own values? y to confirm, n / Esc to cancel")
            }
            None => "reset ALL settings to defaults? y to confirm, n / Esc to cancel".into(),
        };
    }

    fn cancel_reset_all(&mut self) {
        if self.confirm_reset_all {
            self.confirm_reset_all = false;
            self.last_reload = "reset-all cancelled".into();
            self.toast_info("reset-all cancelled");
        }
    }

    /// Commit the confirmed reset-all: overwrite config.toml with an empty
    /// layer, reload, and keep the cursor in range.
    fn confirm_reset_all_yes(&mut self) {
        self.confirm_reset_all = false;
        if let Some(model) = self.scope.model.clone() {
            self.remove_all_for_model(&model);
            return;
        }
        match writer::reset_all(&self.paths.config) {
            Ok(()) => {
                self.after_config_write();
                self.last_reload = "all settings reset to defaults (config.toml cleared)".into();
                self.toast_info("all settings reset to defaults");
            }
            Err(err) => {
                self.last_reload = format!("reset-all failed: {err}");
                self.toast_error(format!("reset-all failed: {err}"));
            }
        }
    }

    pub(crate) fn handle_settings_key(&mut self, key: KeyEvent) {
        // A Tuning screen (chooser / preview / undo confirm) owns the keyboard.
        if self.tuning_open() {
            self.handle_tuning_key(key);
            return;
        }
        // A reset-all confirmation is modal: only y / n / Esc act while armed.
        if self.confirm_reset_all {
            match key.code {
                KeyCode::Char('y') | KeyCode::Char('Y') => self.confirm_reset_all_yes(),
                KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => self.cancel_reset_all(),
                _ => {}
            }
            return;
        }
        // A numeric/string edit in progress owns the keyboard.
        if self.settings_edit.is_some() {
            self.handle_settings_edit_key(key);
            return;
        }
        if self.settings_search_typing() {
            self.handle_settings_search_key(key);
            return;
        }

        let selected = self.settings_selected;
        match key.code {
            KeyCode::Down | KeyCode::Char('j') => self.move_settings_selection(selected + 1),
            KeyCode::Up | KeyCode::Char('k') => {
                self.move_settings_selection(selected.saturating_sub(1))
            }
            KeyCode::PageDown => self.move_settings_selection(selected + PAGE_ROWS),
            KeyCode::PageUp => self.move_settings_selection(selected.saturating_sub(PAGE_ROWS)),
            KeyCode::Home => self.move_settings_selection(0),
            KeyCode::End => self.move_settings_selection(usize::MAX),
            KeyCode::Char(digit @ '1'..='6') => {
                let index = digit as usize - '1' as usize;
                self.settings_goto_group(Intent::ALL[index]);
            }
            KeyCode::Char('[') => self.settings_step_group(false),
            KeyCode::Char(']') => self.settings_step_group(true),
            KeyCode::Char('/') => self.open_settings_search(),
            KeyCode::Char('p') => {
                self.discard_settings_preview();
                self.open_tuning();
            }
            KeyCode::Char('u') => {
                self.discard_settings_preview();
                self.open_tuning_undo();
            }
            KeyCode::Char('m') => self.open_scope_picker(),
            KeyCode::Char('o') => self.open_overrides(None),
            KeyCode::Char('c') => {
                self.discard_settings_preview();
                self.settings_changed_only = !self.settings_changed_only;
                self.settings_selected = 0;
                self.last_reload = format!(
                    "changed only: {}",
                    if self.settings_changed_only {
                        "on"
                    } else {
                        "off"
                    }
                );
            }
            KeyCode::Char('h') => {
                self.discard_settings_preview();
                self.settings_hide_unused = !self.settings_hide_unused;
                self.settings_selected = 0;
                self.last_reload = format!(
                    "hide unused: {}",
                    if self.settings_hide_unused {
                        "on"
                    } else {
                        "off"
                    }
                );
            }
            // Esc cancels an enum preview, else drops a kept search filter.
            KeyCode::Esc => {
                if self.settings_pending.is_some() {
                    self.discard_settings_preview();
                    self.last_reload = "preview cancelled".into();
                } else if self.settings_search.is_some() {
                    self.close_settings_search(true);
                }
            }
            // Del and Backspace both read as "back to inherited".
            KeyCode::Delete | KeyCode::Backspace => {
                self.discard_settings_preview();
                self.reset_selected_setting();
            }
            // Shift+R arms reset-ALL; lowercase `r` is the global refresh.
            KeyCode::Char('R') => {
                self.discard_settings_preview();
                self.arm_reset_all();
            }
            KeyCode::Right | KeyCode::Char(' ') | KeyCode::Left => {
                self.settings_change(!matches!(key.code, KeyCode::Left));
            }
            KeyCode::Enter => self.settings_commit(),
            _ => {}
        }
    }

    fn handle_settings_search_key(&mut self, key: KeyEvent) {
        let selected = self.settings_selected;
        match key.code {
            KeyCode::Esc => self.close_settings_search(true),
            KeyCode::Enter => {
                if let Some(search) = self.settings_search.as_mut() {
                    search.typing = false;
                }
                self.clamp_settings_selection();
            }
            KeyCode::Down => self.move_settings_selection(selected + 1),
            KeyCode::Up => self.move_settings_selection(selected.saturating_sub(1)),
            KeyCode::Backspace => {
                if let Some(search) = self.settings_search.as_mut() {
                    search.query.pop();
                }
                self.settings_selected = 0;
            }
            KeyCode::Char(c)
                if !key
                    .modifiers
                    .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) =>
            {
                if let Some(search) = self.settings_search.as_mut() {
                    search.query.push(c);
                }
                self.settings_selected = 0;
            }
            _ => {}
        }
    }

    /// ←/→/Space on the selected row: an enum previews, a bool toggles and
    /// saves, the Prompt processing row previews its next choice.
    fn settings_change(&mut self, forward: bool) {
        let Some(row) = self.selected_row().cloned() else {
            return;
        };
        if let Some(reason) = self.scope_lock_reason(&row) {
            self.discard_settings_preview();
            self.locked_notice(&reason);
            return;
        }
        if let Some(lock) = row.locked.clone() {
            self.discard_settings_preview();
            self.locked_notice(&lock);
            return;
        }
        match &row.id {
            RowId::Virtual(VirtualRow::PromptProcessing) => {
                self.preview_next_choice(&row, forward);
            }
            RowId::Key(key) => self.change_key_row(&row, key, forward),
        }
    }

    fn preview_next_choice(&mut self, row: &SettingsRow, forward: bool) {
        let count = row.choices.len();
        if count == 0 {
            // No route on this GPU: ←/→ do nothing.
            self.discard_settings_preview();
            return;
        }
        let from = match &self.settings_pending {
            Some(pending) if pending.key == PROMPT_PROCESSING_ID => row
                .choices
                .iter()
                .position(|choice| choice.id == pending.value),
            _ => row.current_choice,
        };
        let next = match (from, forward) {
            (Some(at), true) => (at + 1) % count,
            (Some(at), false) => (at + count - 1) % count,
            (None, true) => 0,
            (None, false) => count - 1,
        };
        let choice = &row.choices[next];
        self.settings_pending = Some(PendingEnum {
            key: PROMPT_PROCESSING_ID.to_owned(),
            value: choice.id.to_owned(),
        });
        self.last_reload = format!(
            "{} → {} (preview) · Enter commit · Esc cancel",
            row.title, choice.label
        );
    }

    fn change_key_row(&mut self, row: &SettingsRow, key: &str, forward: bool) {
        if row.deprecated {
            self.discard_settings_preview();
            self.last_reload = deprecated_status(key);
            return;
        }
        match writer::field_spec(key).map(|spec| spec.kind) {
            Some(FieldKind::Enum(_)) => {
                // Cycle from the current PREVIEW when mid-preview on this row,
                // else from the value in effect. Stage only.
                let from = match &self.settings_pending {
                    Some(pending) if pending.key == key => pending.value.clone(),
                    _ => value_string(&row.resolved),
                };
                if let Some(next) = writer::cycle_enum(key, &from, forward) {
                    self.settings_pending = Some(PendingEnum {
                        key: key.to_owned(),
                        value: next.clone(),
                    });
                    self.last_reload =
                        format!("{key} → {next} (preview) · Enter commit · Esc cancel");
                }
            }
            Some(FieldKind::Bool) => {
                // A binary flip is a decision, not browsing: toggle and save.
                self.discard_settings_preview();
                let now = match (&row.resolved, &row.effective.value) {
                    (ConfigValue::Bool(flag), _) | (_, ConfigValue::Bool(flag)) => *flag,
                    _ => false,
                };
                let (_, status) = self.persist_setting(key, if now { "false" } else { "true" });
                self.last_reload = status;
            }
            Some(_) => {
                self.last_reload = format!("{key} is numeric/text; press Enter to edit");
            }
            None => self.last_reload = not_editable_status(key),
        }
    }

    /// Enter: commit a staged preview, else open the editor for a number or
    /// text row.
    fn settings_commit(&mut self) {
        if let Some(pending) = self.settings_pending.take() {
            match self.selected_row().cloned() {
                Some(row) if pending_belongs_to(&row, &pending) => {
                    self.apply_pending(&row, &pending)
                }
                // The selection moved under the preview: discard rather than
                // write a stale value to the wrong key.
                _ => self.last_reload = "preview discarded".into(),
            }
            return;
        }
        let Some(row) = self.selected_row().cloned() else {
            return;
        };
        if let Some(reason) = self.scope_lock_reason(&row) {
            self.locked_notice(&reason);
            return;
        }
        if let Some(lock) = row.locked.clone() {
            self.locked_notice(&lock);
            return;
        }
        let key = match &row.id {
            RowId::Virtual(_) => {
                self.last_reload = if row.choices.is_empty() {
                    format!("{} is not used on this GPU", row.title)
                } else {
                    format!("{}: ←/→ choose, Enter saves", row.title)
                };
                return;
            }
            RowId::Key(key) => key.clone(),
        };
        if row.deprecated {
            self.last_reload = deprecated_status(&key);
            return;
        }
        match writer::field_kind(&key) {
            Some(FieldKind::Int) | Some(FieldKind::Float) | Some(FieldKind::FreeStr { .. }) => {
                let buffer = value_string(&row.resolved);
                self.settings_edit = Some(EditState::new(key.clone(), buffer));
                let range = writer::schema_range(&key)
                    .map(|range| format!(" ({range})"))
                    .unwrap_or_default();
                self.last_reload =
                    format!("editing {key}{range}: type a value, Enter to save, Esc to cancel");
            }
            Some(FieldKind::Enum(_)) => {
                self.last_reload =
                    format!("{key}: Left/Right/Space to preview values, Enter to commit");
            }
            Some(FieldKind::Bool) => {
                self.last_reload = format!("{key} is on/off; Left/Right/Space toggles");
            }
            None => self.last_reload = not_editable_status(&key),
        }
    }

    fn apply_pending(&mut self, row: &SettingsRow, pending: &PendingEnum) {
        match &row.id {
            RowId::Virtual(_) => {
                let Some(choice) = row
                    .choices
                    .iter()
                    .find(|choice| choice.id == pending.value)
                    .cloned()
                else {
                    self.last_reload = "preview discarded".into();
                    return;
                };
                match writer::write_choice(&self.paths.config, &choice) {
                    Ok(()) => {
                        self.after_config_write();
                        self.last_reload = format!(
                            "{} → {} saved to ~/.hipfire/config.toml",
                            row.title, choice.label
                        );
                        self.toast_info(format!("{}: {} saved", row.title, choice.label));
                    }
                    Err(err) => {
                        self.toast_error(format!("save failed: {err}"));
                        self.last_reload = format!("{err}");
                    }
                }
            }
            RowId::Key(key) => {
                let (_, status) = self.persist_setting(key, &pending.value);
                self.last_reload = status;
            }
        }
    }

    /// Drop any staged-but-uncommitted preview. Called when the cursor leaves
    /// the row, the tab changes, or another action supersedes it.
    pub(crate) fn discard_settings_preview(&mut self) {
        self.settings_pending = None;
    }

    fn handle_settings_edit_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Esc => {
                self.settings_edit = None;
                self.last_reload = "edit cancelled".into();
            }
            KeyCode::Enter => {
                // Peek the buffer; only clear the editor on a successful save so
                // a rejected value stays on screen for correction.
                if let Some(edit) = self.settings_edit.as_ref() {
                    let key = edit.key.clone();
                    let raw = edit.buffer.trim().to_string();
                    let (ok, status) = self.persist_setting(&key, &raw);
                    self.last_reload = status;
                    if ok {
                        self.settings_edit = None;
                    }
                }
            }
            KeyCode::Backspace => {
                if let Some(edit) = self.settings_edit.as_mut() {
                    edit.backspace();
                }
            }
            KeyCode::Delete => {
                if let Some(edit) = self.settings_edit.as_mut() {
                    edit.delete();
                }
            }
            KeyCode::Left => {
                if let Some(edit) = self.settings_edit.as_mut() {
                    edit.left();
                }
            }
            KeyCode::Right => {
                if let Some(edit) = self.settings_edit.as_mut() {
                    edit.right();
                }
            }
            KeyCode::Home => {
                if let Some(edit) = self.settings_edit.as_mut() {
                    edit.home();
                }
            }
            KeyCode::End => {
                if let Some(edit) = self.settings_edit.as_mut() {
                    edit.end();
                }
            }
            KeyCode::Char(c) => {
                if let Some(edit) = self.settings_edit.as_mut() {
                    edit.insert(c);
                }
            }
            _ => {}
        }
    }
}

/// A staged preview belongs to the selected row.
fn pending_belongs_to(row: &SettingsRow, pending: &PendingEnum) -> bool {
    match &row.id {
        RowId::Virtual(VirtualRow::PromptProcessing) => pending.key == PROMPT_PROCESSING_ID,
        RowId::Key(key) => pending.key == *key,
    }
}

#[cfg(test)]
mod tests {
    use crossterm::event::KeyCode;
    use hipfire_config::{
        effective::A8Prefill, field, load_global, ConfigLayer, ConfigPaths, ConfigValue,
    };

    use super::*;
    use crate::{
        app::{Tab, ToastLevel},
        hipfire::dashboard::LoadedConfig,
        test_support::{
            online_dashboard, press, select_key, select_row, type_text, Fixture, FixtureBuilder,
            ACTIVE_MODEL,
        },
    };

    fn toast(app: &App) -> String {
        app.toast
            .as_ref()
            .map(|toast| toast.text.clone())
            .unwrap_or_default()
    }

    fn global(fixture: &Fixture) -> ConfigLayer {
        load_global(&ConfigPaths::under(fixture.root()))
            .expect("config.toml parses")
            .layer
    }

    fn config_text(fixture: &Fixture) -> String {
        String::from_utf8(fixture.config_bytes()).expect("utf-8 config")
    }

    fn prompt_row() -> RowId {
        RowId::Virtual(VirtualRow::PromptProcessing)
    }

    fn row_display(app: &App, id: &RowId) -> String {
        app.settings.row(id).expect("row").display.clone()
    }

    #[test]
    fn number_keys_and_brackets_switch_groups() {
        let mut fixture = FixtureBuilder::new("gfx1201").build();
        let app = &mut fixture.app;
        assert_eq!(app.settings_group, Intent::Quality);
        for (digit, group) in Intent::ALL.iter().enumerate() {
            press(app, KeyCode::Char(char::from(b'1' + digit as u8)));
            assert_eq!(app.settings_group, *group, "key {}", digit + 1);
            assert_eq!(app.settings_selected, 0);
        }
        // `]` walks forward and wraps, `[` walks back and wraps.
        press(app, KeyCode::Char('1'));
        press(app, KeyCode::Char(']'));
        assert_eq!(app.settings_group, Intent::Speed);
        press(app, KeyCode::Char('['));
        press(app, KeyCode::Char('['));
        assert_eq!(app.settings_group, Intent::Advanced, "wraps to the end");
        press(app, KeyCode::Char(']'));
        assert_eq!(app.settings_group, Intent::Quality, "wraps to the start");
        // 7-9 are not groups.
        press(app, KeyCode::Char('7'));
        assert_eq!(app.settings_group, Intent::Quality);
    }

    #[test]
    fn movement_keys_move_within_the_group() {
        let mut fixture = FixtureBuilder::new("gfx1201").build();
        let app = &mut fixture.app;
        press(app, KeyCode::Char('6'));
        let len = app.settings_visible().len();
        assert!(len > 20, "Advanced is the long list");
        press(app, KeyCode::Down);
        press(app, KeyCode::Char('j'));
        assert_eq!(app.settings_selected, 2);
        press(app, KeyCode::Up);
        press(app, KeyCode::Char('k'));
        assert_eq!(app.settings_selected, 0);
        press(app, KeyCode::PageDown);
        assert_eq!(app.settings_selected, PAGE_ROWS);
        press(app, KeyCode::PageUp);
        assert_eq!(app.settings_selected, 0);
        press(app, KeyCode::End);
        assert_eq!(app.settings_selected, len - 1);
        press(app, KeyCode::Down);
        assert_eq!(app.settings_selected, len - 1, "clamped at the end");
        press(app, KeyCode::Home);
        assert_eq!(app.settings_selected, 0);
    }

    #[test]
    fn e_and_a_no_longer_do_anything() {
        let mut fixture = strix_with_effort();
        let before = fixture.config_bytes();
        let app = &mut fixture.app;
        press(app, KeyCode::Char('3'));
        press(app, KeyCode::Down);
        let (group, selected) = (app.settings_group, app.settings_selected);
        app.last_reload.clear();
        press(app, KeyCode::Char('e'));
        press(app, KeyCode::Char('a'));
        assert_eq!(
            (app.settings_group, app.settings_selected),
            (group, selected)
        );
        assert!(app.settings_search.is_none() && app.settings_pending.is_none());
        assert!(app.settings_edit.is_none() && app.last_reload.is_empty());
        assert_eq!(fixture.config_bytes(), before);
    }

    fn strix_with_effort() -> Fixture {
        FixtureBuilder::new("gfx1151")
            .a8(A8Prefill::OptIn)
            .config("[reasoning]\neffort = \"low\"\n")
            .build()
    }

    #[test]
    fn search_takes_every_printable_key_and_esc_restores_group_and_selection() {
        let mut fixture = FixtureBuilder::new("gfx1201").build();
        let app = &mut fixture.app;
        press(app, KeyCode::Char('3'));
        press(app, KeyCode::Down);
        press(app, KeyCode::Down);
        let (group, selected) = (app.settings_group, app.settings_selected);
        assert_eq!(selected, 2);

        press(app, KeyCode::Char('/'));
        assert!(app.settings_search_typing());
        // q, r, ? and the old e/a/h/c keys are text while the buffer has focus.
        type_text(app, "qr?eahc");
        assert_eq!(app.settings_search.as_ref().unwrap().query, "qr?eahc");
        assert!(!app.settings_changed_only && !app.settings_hide_unused);
        press(app, KeyCode::Backspace);
        assert_eq!(app.settings_search.as_ref().unwrap().query, "qr?eah");
        for _ in 0..6 {
            press(app, KeyCode::Backspace);
        }
        type_text(app, "kv");
        press(app, KeyCode::Down);
        assert!(app.settings_selected > 0);

        press(app, KeyCode::Esc);
        assert!(app.settings_search.is_none());
        assert_eq!(
            (app.settings_group, app.settings_selected),
            (group, selected)
        );
    }

    #[test]
    fn search_kv_lists_memory_and_advanced_rows_and_enter_moves_focus_to_results() {
        let mut fixture = FixtureBuilder::new("gfx1201").build();
        let app = &mut fixture.app;
        press(app, KeyCode::Char('/'));
        type_text(app, "kv");
        let hits: Vec<(Intent, String)> = app
            .settings_visible()
            .iter()
            .map(|row| (row.group, row.title.clone()))
            .collect();
        assert!(
            hits.iter()
                .any(|(group, title)| *group == Intent::MemoryContext
                    && title == "KV cache precision")
        );
        assert!(hits.iter().any(|(group, _)| *group == Intent::Advanced));

        press(app, KeyCode::Enter);
        assert!(
            !app.settings_search_typing(),
            "Enter moves focus to results"
        );
        assert!(app.settings_search.is_some(), "and keeps the filter");
        let kept = app.settings_visible().len();
        // `j` now moves instead of typing.
        press(app, KeyCode::Char('j'));
        assert_eq!(app.settings_selected, 1);
        assert_eq!(app.settings_visible().len(), kept);
        // Esc clears the kept filter and returns to the group.
        press(app, KeyCode::Esc);
        assert!(app.settings_search.is_none());
        assert_eq!(app.settings_group, Intent::Quality);
        assert_eq!(app.settings_selected, 0);
    }

    #[test]
    fn changed_only_and_hide_unused_filters_apply_to_the_group() {
        let mut fixture = strix_with_effort();
        let app = &mut fixture.app;
        let all = app.settings_visible().len();
        press(app, KeyCode::Char('c'));
        assert!(app.settings_changed_only);
        let titles: Vec<String> = app
            .settings_visible()
            .iter()
            .map(|row| row.title.clone())
            .collect();
        assert_eq!(titles, ["Thinking effort"]);
        press(app, KeyCode::Char('c'));
        assert_eq!(app.settings_visible().len(), all);

        press(app, KeyCode::Char('h'));
        let used: Vec<String> = app
            .settings_visible()
            .iter()
            .map(|row| row.title.clone())
            .collect();
        assert!(used.contains(&"Prompt processing".to_owned()));
        assert!(!used.contains(&"Flash-Next experts".to_owned()), "{used:?}");
    }

    #[test]
    fn env_owned_row_is_locked_and_writes_nothing() {
        let mut fixture = FixtureBuilder::new("gfx1201")
            .env("memory.kv_cache", "q8")
            .build();
        let before = fixture.config_bytes();
        select_key(&mut fixture.app, "memory.kv_cache");
        let app = &mut fixture.app;
        assert_eq!(
            app.selected_row().unwrap().origin,
            Origin::Env("HIPFIRE_KV_MODE".into())
        );
        for code in [
            KeyCode::Right,
            KeyCode::Left,
            KeyCode::Char(' '),
            KeyCode::Delete,
            KeyCode::Backspace,
            KeyCode::Enter,
        ] {
            app.toast = None;
            press(app, code);
            assert_eq!(
                toast(app),
                "set by HIPFIRE_KV_MODE; unset it to change",
                "{code:?}"
            );
            assert!(app.settings_pending.is_none() && app.settings_edit.is_none());
        }
        assert_eq!(fixture.config_bytes(), before, "config.toml is untouched");
    }

    #[test]
    fn registry_value_reads_registry_and_del_explains() {
        let mut fixture = FixtureBuilder::new("gfx1201").card_kv_mode("fwht3").build();
        let before = fixture.config_bytes();
        select_key(&mut fixture.app, "memory.kv_cache");
        let app = &mut fixture.app;
        let row = app.selected_row().unwrap();
        assert_eq!(row.origin, Origin::Registry);
        assert_eq!(row.origin.label(), "registry");
        press(app, KeyCode::Delete);
        assert_eq!(toast(app), "nothing to reset (comes from the model card)");
        assert_eq!(fixture.config_bytes(), before);
    }

    #[test]
    fn prompt_processing_cycles_choices_and_writes_the_arch_key() {
        let mut fixture = FixtureBuilder::new("gfx1201").build();
        select_row(&mut fixture.app, &prompt_row());
        assert_eq!(
            row_display(&fixture.app, &prompt_row()),
            "Fast (4-bit math)"
        );
        let before = fixture.config_bytes();

        // → stages Accurate; nothing is written until Enter.
        press(&mut fixture.app, KeyCode::Right);
        let pending = fixture.app.settings_pending.as_ref().expect("preview");
        assert_eq!(pending.value, "accurate");
        assert_eq!(fixture.config_bytes(), before);
        press(&mut fixture.app, KeyCode::Enter);
        assert!(fixture.app.settings_pending.is_none());
        assert_eq!(row_display(&fixture.app, &prompt_row()), "Accurate (fp8)");
        assert_eq!(
            global(&fixture).get("kernel.iu4_prefill"),
            Some(&ConfigValue::Bool(false))
        );

        // → again wraps to Fast; Enter removes the key.
        press(&mut fixture.app, KeyCode::Right);
        assert_eq!(fixture.app.settings_pending.as_ref().unwrap().value, "fast");
        press(&mut fixture.app, KeyCode::Enter);
        assert_eq!(global(&fixture).get("kernel.iu4_prefill"), None);
        assert_eq!(
            row_display(&fixture.app, &prompt_row()),
            "Fast (4-bit math)"
        );

        // ← cycles backwards; Del returns to the built-in route.
        press(&mut fixture.app, KeyCode::Left);
        press(&mut fixture.app, KeyCode::Enter);
        assert_eq!(
            global(&fixture).get("kernel.iu4_prefill"),
            Some(&ConfigValue::Bool(false))
        );
        press(&mut fixture.app, KeyCode::Delete);
        assert_eq!(global(&fixture).get("kernel.iu4_prefill"), None);
        assert_eq!(toast(&fixture.app), "Prompt processing reset to default");
    }

    #[test]
    fn prompt_processing_without_a_route_is_inert() {
        let mut fixture = FixtureBuilder::new("gfx1030").build();
        select_row(&mut fixture.app, &prompt_row());
        assert_eq!(
            row_display(&fixture.app, &prompt_row()),
            "— not used on this GPU"
        );
        let before = fixture.config_bytes();
        for code in [
            KeyCode::Right,
            KeyCode::Left,
            KeyCode::Char(' '),
            KeyCode::Enter,
        ] {
            press(&mut fixture.app, code);
            assert!(fixture.app.settings_pending.is_none(), "{code:?}");
            assert!(fixture.app.settings_edit.is_none());
        }
        assert_eq!(fixture.config_bytes(), before);
    }

    #[test]
    fn prompt_processing_on_strix_halo_writes_the_key_the_a8_state_selects() {
        // Without PR #902 in the schema, "Accurate" is iu4 off.
        let mut fixture = FixtureBuilder::new("gfx1151").build();
        select_row(&mut fixture.app, &prompt_row());
        press(&mut fixture.app, KeyCode::Right);
        press(&mut fixture.app, KeyCode::Enter);
        assert_eq!(
            global(&fixture).get("kernel.iu4_prefill"),
            Some(&ConfigValue::Bool(false))
        );

        // With the A8 key in the schema, "Accurate" turns A8 on instead. Only
        // runs where the schema has the field (it does not on today's beta).
        if field("kernel.gfx11_a8_prefill").is_some() {
            let mut fixture = FixtureBuilder::new("gfx1151").a8(A8Prefill::OptIn).build();
            select_row(&mut fixture.app, &prompt_row());
            press(&mut fixture.app, KeyCode::Right);
            press(&mut fixture.app, KeyCode::Enter);
            assert_eq!(
                global(&fixture).get("kernel.gfx11_a8_prefill"),
                Some(&ConfigValue::Bool(true))
            );
        }
    }

    #[test]
    fn iu4_prefill_and_kv_bf16_are_settable_from_advanced() {
        let mut fixture = FixtureBuilder::new("gfx1201").build();
        select_key(&mut fixture.app, "kernel.iu4_prefill");
        assert_eq!(fixture.app.selected_row().unwrap().group, Intent::Advanced);
        press(&mut fixture.app, KeyCode::Right); // bool: toggles and saves
        assert!(
            fixture.app.settings_pending.is_none(),
            "bool does not preview"
        );
        assert_eq!(
            global(&fixture).get("kernel.iu4_prefill"),
            Some(&ConfigValue::Bool(false))
        );

        // memory.kv_cache lives in Memory; the Advanced search reaches the same
        // row. auto → q8 → fwht4 → fwht3 → fwht2 → fp8 → bf16.
        select_key(&mut fixture.app, "memory.kv_cache");
        for _ in 0..6 {
            press(&mut fixture.app, KeyCode::Right);
        }
        assert_eq!(fixture.app.settings_pending.as_ref().unwrap().value, "bf16");
        press(&mut fixture.app, KeyCode::Enter);
        assert_eq!(
            global(&fixture).get("memory.kv_cache"),
            Some(&ConfigValue::String("bf16".into()))
        );
    }

    #[test]
    fn sym_iu4_route_is_written_as_a_toml_boolean() {
        let mut fixture = FixtureBuilder::new("gfx1151").build();
        select_key(&mut fixture.app, "developer.qwen4_moe_sym_iu4");
        press(&mut fixture.app, KeyCode::Right);
        let text = config_text(&fixture);
        assert!(
            text.contains("qwen4_moe_sym_iu4 = true") || text.contains("qwen4_moe_sym_iu4 = false"),
            "{text}"
        );
        assert!(
            !text.contains("qwen4_moe_sym_iu4 = \""),
            "never a string: {text}"
        );
        assert!(matches!(
            global(&fixture).get("developer.qwen4_moe_sym_iu4"),
            Some(ConfigValue::Bool(_))
        ));
    }

    #[test]
    fn enum_preview_stages_commits_and_cancels() {
        let mut fixture = FixtureBuilder::new("gfx1201").build();
        select_key(&mut fixture.app, "attention.flash");
        let before = fixture.config_bytes();
        press(&mut fixture.app, KeyCode::Right);
        let first = fixture.app.settings_pending.as_ref().unwrap().value.clone();
        assert_eq!(fixture.config_bytes(), before, "browsing writes nothing");
        // A second → advances from the preview, not the committed value.
        press(&mut fixture.app, KeyCode::Right);
        let second = fixture.app.settings_pending.as_ref().unwrap().value.clone();
        assert_ne!(first, second);
        // Esc and moving away both discard.
        press(&mut fixture.app, KeyCode::Esc);
        assert!(fixture.app.settings_pending.is_none());
        press(&mut fixture.app, KeyCode::Right);
        // The attention kernel is the group's last row, so move up and back.
        press(&mut fixture.app, KeyCode::Up);
        assert!(fixture.app.settings_pending.is_none());
        assert_eq!(fixture.config_bytes(), before);
        press(&mut fixture.app, KeyCode::Down);
        // Leaving the tab discards too.
        press(&mut fixture.app, KeyCode::Right);
        fixture.app.next_tab();
        assert!(fixture.app.settings_pending.is_none());
        fixture.app.tab = Tab::Settings;
        // Enter commits a staged value.
        press(&mut fixture.app, KeyCode::Right);
        let staged = fixture.app.settings_pending.as_ref().unwrap().value.clone();
        press(&mut fixture.app, KeyCode::Enter);
        assert_eq!(
            global(&fixture).get("attention.flash"),
            Some(&ConfigValue::String(staged))
        );
        assert!(fixture.app.settings_pending.is_none());
    }

    #[test]
    fn del_resets_a_changed_row_and_is_a_noop_at_the_default() {
        let mut fixture = strix_with_effort();
        select_key(&mut fixture.app, "reasoning.effort");
        press(&mut fixture.app, KeyCode::Delete);
        assert_eq!(global(&fixture).get("reasoning.effort"), None);
        assert!(fixture.app.last_reload.contains("reset to default"));
        assert_eq!(fixture.app.selected_row().unwrap().title, "Thinking effort");

        let before = fixture.config_bytes();
        press(&mut fixture.app, KeyCode::Delete);
        assert_eq!(fixture.config_bytes(), before, "no rewrite at the default");
        assert!(fixture.app.last_reload.contains("already at its default"));
    }

    #[test]
    fn reset_all_needs_confirmation_and_is_modal() {
        let mut fixture = FixtureBuilder::new("gfx1201")
            .config("[memory]\nkv_cache = \"q8\"\n[speculation]\ndflash = \"auto\"\n")
            .build();
        let app = &mut fixture.app;
        press(app, KeyCode::Char('R'));
        assert!(app.confirm_reset_all);
        press(app, KeyCode::Char('j')); // swallowed
        assert!(app.confirm_reset_all);
        press(app, KeyCode::Char('n'));
        assert!(!app.confirm_reset_all);
        assert!(
            global(&fixture).values.len() >= 3,
            "cancel keeps everything"
        );

        let app = &mut fixture.app;
        press(app, KeyCode::Char('R'));
        press(app, KeyCode::Char('y'));
        assert!(!app.confirm_reset_all);
        assert!(
            global(&fixture).values.is_empty(),
            "confirm clears the file"
        );
    }

    #[test]
    fn corrupt_config_points_reset_at_reset_all() {
        let mut fixture = FixtureBuilder::new("gfx1201").build();
        std::fs::write(fixture.config_path(), "{ broken not toml").unwrap();
        fixture.app.config = fixture.app.load_config();
        assert!(fixture.app.config.warning.is_some());
        press(&mut fixture.app, KeyCode::Delete);
        assert!(
            fixture.app.last_reload.contains("unreadable"),
            "{}",
            fixture.app.last_reload
        );
        assert_eq!(config_text(&fixture), "{ broken not toml");
    }

    #[test]
    fn typed_values_commit_clear_and_keep_the_buffer_on_error() {
        let mut fixture = FixtureBuilder::new("gfx1201").build();
        select_key(&mut fixture.app, "memory.gpu_layer_budget");
        press(&mut fixture.app, KeyCode::Enter);
        assert_eq!(fixture.app.settings_edit.as_ref().unwrap().buffer, "");
        type_text(&mut fixture.app, "32");
        press(&mut fixture.app, KeyCode::Enter);
        assert!(fixture.app.settings_edit.is_none());
        assert_eq!(
            global(&fixture).get("memory.gpu_layer_budget"),
            Some(&ConfigValue::Integer(32))
        );
        assert_eq!(
            row_display(&fixture.app, &RowId::Key("memory.gpu_layer_budget".into())),
            "32 on GPU"
        );

        // The editor seeds from the value and an emptied buffer clears the key.
        press(&mut fixture.app, KeyCode::Enter);
        assert_eq!(fixture.app.settings_edit.as_ref().unwrap().buffer, "32");
        press(&mut fixture.app, KeyCode::Backspace);
        press(&mut fixture.app, KeyCode::Backspace);
        press(&mut fixture.app, KeyCode::Enter);
        assert!(fixture.app.settings_edit.is_none());
        assert_eq!(global(&fixture).get("memory.gpu_layer_budget"), None);

        // A rejected value keeps the editor open with the buffer intact.
        select_key(&mut fixture.app, "memory.max_seq");
        press(&mut fixture.app, KeyCode::Enter);
        for _ in 0..5 {
            press(&mut fixture.app, KeyCode::Backspace);
        }
        type_text(&mut fixture.app, "nonsense");
        press(&mut fixture.app, KeyCode::Enter);
        assert_eq!(
            fixture.app.settings_edit.as_ref().unwrap().buffer,
            "nonsense"
        );
        assert!(matches!(
            fixture.app.toast.as_ref().map(|toast| toast.level),
            Some(ToastLevel::Error)
        ));
        // Esc leaves the editor without writing.
        press(&mut fixture.app, KeyCode::Esc);
        assert!(fixture.app.settings_edit.is_none());
    }

    #[test]
    fn edit_cursor_inserts_and_deletes_at_the_cursor() {
        let mut fixture = FixtureBuilder::new("gfx1201").build();
        select_key(&mut fixture.app, "memory.max_seq");
        press(&mut fixture.app, KeyCode::Enter);
        let edit = fixture.app.settings_edit.as_ref().expect("editor opened");
        assert_eq!((edit.buffer.as_str(), edit.cursor), ("32768", 5));
        // Home, then step right twice: 32|768 -> insert 9 -> 329|768.
        press(&mut fixture.app, KeyCode::Home);
        press(&mut fixture.app, KeyCode::Right);
        press(&mut fixture.app, KeyCode::Right);
        press(&mut fixture.app, KeyCode::Char('9'));
        let edit = fixture.app.settings_edit.as_ref().unwrap();
        assert_eq!((edit.buffer.as_str(), edit.cursor), ("329768", 3));
        press(&mut fixture.app, KeyCode::Backspace);
        assert_eq!(fixture.app.settings_edit.as_ref().unwrap().buffer, "32768");
        press(&mut fixture.app, KeyCode::Delete);
        let edit = fixture.app.settings_edit.as_ref().unwrap();
        assert_eq!((edit.buffer.as_str(), edit.cursor), ("3268", 2));
    }

    #[test]
    fn deprecated_rows_are_read_only_but_resettable() {
        let mut fixture = FixtureBuilder::new("gfx1201").build();
        let before = fixture.config_bytes();
        for legacy in [
            "cask",
            "prefill_compression",
            "prefill_threshold",
            "cask_budget",
        ] {
            let key = field(legacy).expect("deprecated key").key;
            select_key(&mut fixture.app, key);
            for code in [
                KeyCode::Left,
                KeyCode::Right,
                KeyCode::Char(' '),
                KeyCode::Enter,
            ] {
                fixture.app.last_reload.clear();
                press(&mut fixture.app, code);
                assert!(
                    fixture.app.last_reload.contains("deprecated"),
                    "{key} {code:?}: {}",
                    fixture.app.last_reload
                );
                assert!(fixture.app.settings_pending.is_none());
                assert!(fixture.app.settings_edit.is_none());
            }
        }
        assert_eq!(fixture.config_bytes(), before);

        // Read-only does not mean stuck: Del removes an existing value.
        std::fs::write(
            fixture.config_path(),
            "schema_version = 1\n[memory.cask]\nenabled = true\n",
        )
        .unwrap();
        fixture.app.rebuild_settings();
        let key = field("cask").unwrap().key;
        select_key(&mut fixture.app, key);
        assert!(fixture.app.selected_row().unwrap().origin.marked());
        press(&mut fixture.app, KeyCode::Delete);
        assert_eq!(global(&fixture).get(key), None);
    }

    #[test]
    fn live_value_needs_an_online_serve_with_the_active_model_resident() {
        let loaded = |model: &str| LoadedConfig {
            model: model.into(),
            kv_mode: Some("q8".into()),
            kv_backend: Some("vmm".into()),
            max_seq: Some(65536),
        };
        let mut fixture = FixtureBuilder::new("gfx1201").build();
        let app = &mut fixture.app;
        let kv = RowId::Key("memory.kv_cache".into());
        let live = |app: &App| app.live_value(app.settings.row(&kv).unwrap());

        assert_eq!(live(app), None, "no dashboard yet");
        app.dashboard = Some(online_dashboard(Some(loaded(ACTIVE_MODEL)), 60));
        assert_eq!(live(app).as_deref(), Some("q8"));
        let max_seq = app
            .settings
            .row(&RowId::Key("memory.max_seq".into()))
            .unwrap();
        assert_eq!(app.live_value(max_seq).as_deref(), Some("65,536 tokens"));
        let other = app
            .settings
            .row(&RowId::Key("reasoning.effort".into()))
            .unwrap();
        assert_eq!(app.live_value(other), None, "only KV and context are live");

        app.dashboard = Some(online_dashboard(Some(loaded("other:7b")), 60));
        assert_eq!(live(app), None, "another model is resident");
        let mut offline = online_dashboard(Some(loaded(ACTIVE_MODEL)), 60);
        offline.serve_up = false;
        app.dashboard = Some(offline);
        assert_eq!(live(app), None, "serve is down");
        app.dashboard = Some(online_dashboard(None, 60));
        assert_eq!(live(app), None, "an older serve reports nothing");
    }

    #[test]
    fn restart_badge_needs_a_newer_config_and_a_read_at_load_row() {
        let mut fixture = strix_with_effort();
        let app = &mut fixture.app;
        let flash = RowId::Key("developer.qwen4_moe_sym_iu4".into());
        let effort = RowId::Key("reasoning.effort".into());
        let needs = |app: &App, id: &RowId| app.row_needs_restart(app.settings.row(id).unwrap());

        assert!(!needs(app, &flash), "serve is not running");
        // config.toml was written just now; a serve up for an hour is older.
        app.dashboard = Some(online_dashboard(None, 3600));
        assert!(app.config_restart_pending());
        assert!(needs(app, &flash), "process-scoped rows need a restart");
        assert!(!needs(app, &effort), "request-scoped rows do not");

        // A serve that started after the last write has the config.
        let old = SystemTime::now() - Duration::from_secs(7200);
        std::fs::File::options()
            .write(true)
            .open(&app.paths.config)
            .unwrap()
            .set_modified(old)
            .unwrap();
        assert!(!app.config_restart_pending());
        assert!(!needs(app, &flash));
    }

    #[test]
    fn switching_the_active_model_refreshes_the_model_context() {
        let mut fixture = FixtureBuilder::new("gfx1151").build();
        let app = &mut fixture.app;
        assert_eq!(app.arch.registry_tag.as_deref(), Some(ACTIVE_MODEL));
        app.set_active_model("local-only.mq4");
        assert_eq!(app.active_model, "local-only.mq4");
        assert_eq!(app.arch.registry_tag, None);
        assert_eq!(app.arch.model_arch_id, None);
        app.set_active_model(ACTIVE_MODEL);
        assert_eq!(app.arch.registry_tag.as_deref(), Some(ACTIVE_MODEL));
        assert_eq!(app.arch.model_arch_id, Some(5));
    }
}
