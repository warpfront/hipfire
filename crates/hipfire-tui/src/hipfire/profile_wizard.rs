// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

use std::io;

use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEvent};
use hipfire_config::{
    apply_config_profile, create_config_profile, detect_config_profile, diff_config_layers, fields,
    list_config_profiles, load_global, write_global_toml, ConfigField, ConfigLayer,
    ConfigLayerDiff, ConfigPaths, ConfigProfileEntry, ConfigProfileKind, DefaultValue,
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph, Wrap},
    Frame, Terminal,
};

use super::search::field_matches;
use super::HipfirePaths;

const BG: Color = Color::Rgb(7, 7, 9);
const PANEL: Color = Color::Rgb(18, 16, 18);
const PANEL_2: Color = Color::Rgb(40, 24, 27);
const TEXT: Color = Color::Rgb(222, 226, 232);
const MUTED: Color = Color::Rgb(142, 150, 163);
const ACCENT: Color = Color::Rgb(237, 45, 57);
const GREEN: Color = Color::Rgb(102, 217, 139);
const RED: Color = Color::Rgb(255, 95, 104);
const YELLOW: Color = Color::Rgb(240, 200, 90);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Focus {
    Profiles,
    Variables,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InputMode {
    Browse,
    Search,
    Create,
    Confirm,
}

pub struct ProfileWizard {
    paths: ConfigPaths,
    layer: ConfigLayer,
    profiles: Vec<ConfigProfileEntry>,
    active_profile: Option<String>,
    profile_index: usize,
    field_index: usize,
    filtered_fields: Vec<&'static ConfigField>,
    focus: Focus,
    mode: InputMode,
    search: String,
    create_name: String,
    status: String,
    status_error: bool,
    pending: Option<PendingProfile>,
}

/// A profile merge awaiting the user's y/n.
struct PendingProfile {
    name: String,
    next: ConfigLayer,
    diff: ConfigLayerDiff,
}

impl ProfileWizard {
    fn load() -> Result<Self> {
        let root = HipfirePaths::discover().root;
        Self::load_from_paths(ConfigPaths::under(&root))
    }

    fn load_from_paths(paths: ConfigPaths) -> Result<Self> {
        let loaded = load_global(&paths)?;
        let profiles = list_config_profiles(&paths)?;
        let active_profile = detect_config_profile(&paths, &loaded.layer);
        let profile_index = active_profile
            .as_deref()
            .and_then(|active| profiles.iter().position(|profile| profile.name == active))
            .unwrap_or(0);
        Ok(Self {
            paths,
            layer: loaded.layer,
            profiles,
            active_profile,
            profile_index,
            field_index: 0,
            filtered_fields: fields().iter().collect(),
            focus: Focus::Profiles,
            mode: InputMode::Browse,
            search: String::new(),
            create_name: String::new(),
            status: "Select a profile or browse the complete configuration schema.".to_owned(),
            status_error: false,
            pending: None,
        })
    }

    fn refresh_profiles(&mut self) -> Result<()> {
        self.profiles = list_config_profiles(&self.paths)?;
        self.active_profile = detect_config_profile(&self.paths, &self.layer);
        self.profile_index = self
            .profile_index
            .min(self.profiles.len().saturating_sub(1));
        Ok(())
    }

    fn apply_selected_profile(&mut self) {
        let Some(name) = self
            .profiles
            .get(self.profile_index)
            .map(|profile| profile.name.clone())
        else {
            return;
        };
        self.begin_apply(&name);
    }

    /// Compute the merged layer and its diff; open the confirm modal, or
    /// report that nothing would change. Never writes.
    fn begin_apply(&mut self, name: &str) {
        let mut next = self.layer.clone();
        if let Err(error) = apply_config_profile(&mut next, &self.paths, name) {
            self.set_status(format!("Could not apply '{name}': {error}"), true);
            return;
        }
        let diff = diff_config_layers(&self.layer, &next);
        if diff.is_empty() {
            self.set_status(
                format!("Profile '{name}' is already applied; nothing to change."),
                false,
            );
            return;
        }
        self.pending = Some(PendingProfile {
            name: name.to_owned(),
            next,
            diff,
        });
        self.mode = InputMode::Confirm;
    }

    fn confirm_apply(&mut self) {
        self.mode = InputMode::Browse;
        let Some(pending) = self.pending.take() else {
            return;
        };
        let name = pending.name;
        if let Err(error) = write_global_toml(&self.paths, &pending.next) {
            self.set_status(format!("Could not apply '{name}': {error}"), true);
            return;
        }
        self.layer = pending.next;
        match self.refresh_profiles() {
            Ok(()) => self.set_status(
                format!("Applied profile '{name}' ({} changes).", pending.diff.len()),
                false,
            ),
            Err(error) => self.set_status(
                format!("Applied '{name}', but the list could not refresh: {error}"),
                true,
            ),
        }
    }

    fn cancel_apply(&mut self) {
        self.mode = InputMode::Browse;
        if let Some(pending) = self.pending.take() {
            self.set_status(
                format!("Cancelled; profile '{}' was not applied.", pending.name),
                false,
            );
        }
    }

    fn create_profile(&mut self) {
        let name = self.create_name.trim().to_owned();
        if name.is_empty() {
            self.set_status("Profile name must not be empty.".to_owned(), true);
            return;
        }
        match create_config_profile(&self.paths, &name, &self.layer) {
            Ok(path) => {
                if let Err(error) = self.refresh_profiles() {
                    self.set_status(
                        format!("Profile was saved, but the list could not refresh: {error}"),
                        true,
                    );
                    return;
                }
                if let Some(index) = self
                    .profiles
                    .iter()
                    .position(|profile| profile.name == name)
                {
                    self.profile_index = index;
                }
                self.mode = InputMode::Browse;
                self.create_name.clear();
                self.set_status(
                    format!("Created profile '{name}' at {}.", path.display()),
                    false,
                );
            }
            Err(error) => {
                self.set_status(format!("Could not create '{name}': {error}"), true);
            }
        }
    }

    fn set_status(&mut self, status: String, error: bool) {
        self.status = status;
        self.status_error = error;
    }

    fn update_filter(&mut self) {
        self.filtered_fields = fields()
            .iter()
            .filter(|field| field_matches(field, &self.search))
            .collect();
        self.field_index = self
            .field_index
            .min(self.filtered_fields.len().saturating_sub(1));
    }

    fn move_selection(&mut self, delta: isize) {
        let (index, len) = match self.focus {
            Focus::Profiles => (&mut self.profile_index, self.profiles.len()),
            Focus::Variables => (&mut self.field_index, self.filtered_fields.len()),
        };
        if len == 0 {
            *index = 0;
            return;
        }
        if delta < 0 {
            *index = index.saturating_sub(delta.unsigned_abs());
        } else {
            *index = (*index + delta as usize).min(len - 1);
        }
    }

    fn handle_key(&mut self, key: KeyEvent) -> bool {
        match self.mode {
            InputMode::Search => match key.code {
                KeyCode::Esc => {
                    self.search.clear();
                    self.update_filter();
                    self.mode = InputMode::Browse;
                }
                KeyCode::Enter => self.mode = InputMode::Browse,
                KeyCode::Backspace => {
                    self.search.pop();
                    self.update_filter();
                }
                KeyCode::Char(character) => {
                    self.search.push(character);
                    self.update_filter();
                }
                _ => {}
            },
            InputMode::Create => match key.code {
                KeyCode::Esc => {
                    self.create_name.clear();
                    self.mode = InputMode::Browse;
                }
                KeyCode::Enter => self.create_profile(),
                KeyCode::Backspace => {
                    self.create_name.pop();
                }
                KeyCode::Char(character) => self.create_name.push(character),
                _ => {}
            },
            InputMode::Confirm => match key.code {
                KeyCode::Char('y' | 'Y') => self.confirm_apply(),
                KeyCode::Char('n' | 'N') | KeyCode::Esc => self.cancel_apply(),
                _ => {}
            },
            InputMode::Browse => match key.code {
                KeyCode::Char('q') => return true,
                KeyCode::Tab | KeyCode::BackTab => {
                    self.focus = match self.focus {
                        Focus::Profiles => Focus::Variables,
                        Focus::Variables => Focus::Profiles,
                    };
                }
                KeyCode::Up => self.move_selection(-1),
                KeyCode::Down => self.move_selection(1),
                KeyCode::Enter if self.focus == Focus::Profiles => self.apply_selected_profile(),
                KeyCode::Char('n') => {
                    self.focus = Focus::Profiles;
                    self.create_name.clear();
                    self.mode = InputMode::Create;
                }
                KeyCode::Char('/') => {
                    self.focus = Focus::Variables;
                    self.mode = InputMode::Search;
                }
                KeyCode::Esc if !self.search.is_empty() => {
                    self.search.clear();
                    self.update_filter();
                }
                _ => {}
            },
        }
        false
    }
}

pub fn run(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) -> Result<()> {
    let mut wizard = ProfileWizard::load()?;
    loop {
        terminal.draw(|frame| draw(frame, &wizard))?;
        if event::poll(std::time::Duration::from_millis(80))? {
            match event::read()? {
                Event::Key(key) if wizard.handle_key(key) => break,
                Event::Key(_) | Event::Mouse(_) | Event::Resize(_, _) => {}
                _ => {}
            }
        }
    }
    Ok(())
}

fn draw(frame: &mut Frame, wizard: &ProfileWizard) {
    let area = frame.area();
    frame.render_widget(Clear, area);
    let root = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(4),
            Constraint::Min(10),
            Constraint::Length(2),
        ])
        .split(area);

    let status_color = if wizard.status_error { RED } else { MUTED };
    let header = Text::from(vec![
        Line::from(vec![
            Span::styled(
                "HIPFIRE  ",
                Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                "Configuration profiles",
                Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(Span::styled(
            wizard.status.clone(),
            Style::default().fg(status_color),
        )),
    ]);
    frame.render_widget(
        Paragraph::new(header)
            .block(
                Block::default()
                    .borders(Borders::BOTTOM)
                    .border_style(Style::default().fg(PANEL_2)),
            )
            .style(Style::default().fg(TEXT).bg(BG)),
        root[0],
    );

    let body = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(30), Constraint::Percentage(70)])
        .split(root[1]);
    draw_profiles(frame, wizard, body[0]);
    draw_variables(frame, wizard, body[1]);

    let footer = match wizard.mode {
        InputMode::Browse => {
            "↑/↓ move · Enter review & apply · n new · / search variables · Tab/Shift+Tab switch · q quit"
                .to_owned()
        }
        InputMode::Search => format!(
            "Search: {}_  · Enter keep filter · Esc clear",
            wizard.search
        ),
        InputMode::Create => format!(
            "New profile: {}_  · Enter create snapshot · Esc cancel",
            wizard.create_name
        ),
        InputMode::Confirm => "y apply · n/Esc cancel".to_owned(),
    };
    frame.render_widget(
        Paragraph::new(footer).style(Style::default().fg(MUTED).bg(PANEL)),
        root[2],
    );
    if wizard.mode == InputMode::Confirm {
        if let Some(pending) = &wizard.pending {
            draw_confirm(frame, pending, area);
        }
    }
}

/// Display lines for the confirm modal body: one per change, trimmed to
/// `capacity` with a trailing "… N more" when they do not all fit.
fn confirm_lines(diff: &ConfigLayerDiff, capacity: usize) -> Vec<Line<'static>> {
    let mut entries: Vec<Line<'static>> = Vec::with_capacity(diff.len());
    for (key, value) in &diff.added {
        entries.push(Line::from(vec![
            Span::styled("+ ", Style::default().fg(GREEN)),
            Span::styled(key.clone(), Style::default().fg(TEXT)),
            Span::styled(
                format!(" = {}", value.to_toml_string()),
                Style::default().fg(MUTED),
            ),
        ]));
    }
    for (key, old, new) in &diff.changed {
        entries.push(Line::from(vec![
            Span::styled("~ ", Style::default().fg(YELLOW)),
            Span::styled(key.clone(), Style::default().fg(TEXT)),
            Span::styled(
                format!(": {} → {}", old.to_toml_string(), new.to_toml_string()),
                Style::default().fg(MUTED),
            ),
        ]));
    }
    for (key, old) in &diff.removed {
        entries.push(Line::from(vec![
            Span::styled("- ", Style::default().fg(RED)),
            Span::styled(key.clone(), Style::default().fg(TEXT)),
            Span::styled(
                format!(" (was {})", old.to_toml_string()),
                Style::default().fg(MUTED),
            ),
        ]));
    }
    if entries.len() > capacity {
        if capacity == 0 {
            return Vec::new();
        }
        let hidden = entries.len() - (capacity - 1);
        entries.truncate(capacity - 1);
        entries.push(Line::from(Span::styled(
            format!("… {hidden} more"),
            Style::default().fg(MUTED),
        )));
    }
    entries
}

fn draw_confirm(frame: &mut Frame, pending: &PendingProfile, area: Rect) {
    let width = area.width.saturating_sub(4).min(78);
    let height = area.height.saturating_sub(2).min(22);
    let rect = Rect {
        x: area.x + (area.width.saturating_sub(width)) / 2,
        y: area.y + (area.height.saturating_sub(height)) / 2,
        width,
        height,
    };
    frame.render_widget(Clear, rect);
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(ACCENT))
        .title(format!(" Apply profile '{}'? ", pending.name))
        .style(Style::default().fg(TEXT).bg(PANEL));
    let inner = block.inner(rect);
    frame.render_widget(block, rect);
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2),
            Constraint::Min(0),
            Constraint::Length(1),
        ])
        .split(inner);
    let diff = &pending.diff;
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(format!(
                "{} added · {} overwritten · {} removed",
                diff.added.len(),
                diff.changed.len(),
                diff.removed.len()
            )),
            Line::from(""),
        ])
        .style(Style::default().fg(TEXT).bg(PANEL)),
        rows[0],
    );
    frame.render_widget(
        Paragraph::new(confirm_lines(diff, rows[1].height as usize))
            .style(Style::default().fg(TEXT).bg(PANEL)),
        rows[1],
    );
    frame.render_widget(
        Paragraph::new("y apply · n/Esc cancel").style(Style::default().fg(MUTED).bg(PANEL)),
        rows[2],
    );
}

fn draw_profiles(frame: &mut Frame, wizard: &ProfileWizard, area: Rect) {
    let focused = wizard.focus == Focus::Profiles;
    let items: Vec<ListItem> = wizard
        .profiles
        .iter()
        .map(|profile| {
            let active = wizard.active_profile.as_deref() == Some(profile.name.as_str());
            let kind = match profile.kind {
                ConfigProfileKind::Builtin => "built-in",
                ConfigProfileKind::Custom => "custom",
            };
            ListItem::new(Line::from(vec![
                Span::styled(if active { "● " } else { "  " }, Style::default().fg(GREEN)),
                Span::styled(profile.name.clone(), Style::default().fg(TEXT)),
                Span::styled(format!("  {kind}"), Style::default().fg(MUTED)),
            ]))
        })
        .collect();
    let mut state = ListState::default();
    if !items.is_empty() {
        state.select(Some(wizard.profile_index));
    }
    let border = if focused { ACCENT } else { PANEL_2 };
    let title = if wizard.mode == InputMode::Create {
        format!(" Profiles · new: {}_ ", wizard.create_name)
    } else {
        " Profiles ".to_owned()
    };
    let list = List::new(items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(border))
                .title(title),
        )
        .highlight_style(
            Style::default()
                .bg(PANEL_2)
                .fg(TEXT)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("› ")
        .style(Style::default().fg(TEXT).bg(PANEL));
    frame.render_stateful_widget(list, area, &mut state);
}

fn draw_variables(frame: &mut Frame, wizard: &ProfileWizard, area: Rect) {
    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(42), Constraint::Percentage(58)])
        .split(area);
    let focused = wizard.focus == Focus::Variables;
    let border = if focused { ACCENT } else { PANEL_2 };
    let title = if wizard.search.is_empty() {
        format!(" Variables · all {} ", fields().len())
    } else {
        format!(
            " Variables · /{} · {} matches ",
            wizard.search,
            wizard.filtered_fields.len()
        )
    };
    let items: Vec<ListItem> = wizard
        .filtered_fields
        .iter()
        .map(|field| {
            ListItem::new(Line::from(vec![
                Span::styled(field.key, Style::default().fg(TEXT)),
                Span::styled(
                    format!("  ({})", field.legacy_key),
                    Style::default().fg(MUTED),
                ),
            ]))
        })
        .collect();
    let mut state = ListState::default();
    if !items.is_empty() {
        state.select(Some(wizard.field_index));
    }
    let list = List::new(items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(border))
                .title(title),
        )
        .highlight_style(
            Style::default()
                .bg(PANEL_2)
                .fg(TEXT)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("› ")
        .style(Style::default().fg(TEXT).bg(PANEL));
    frame.render_stateful_widget(list, columns[0], &mut state);

    let detail = wizard
        .filtered_fields
        .get(wizard.field_index)
        .map(|field| field_detail(wizard, field))
        .unwrap_or_else(|| Text::from("No configuration variables match this search."));
    frame.render_widget(
        Paragraph::new(detail)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(PANEL_2))
                    .title(" Variable reference "),
            )
            .wrap(Wrap { trim: false })
            .style(Style::default().fg(TEXT).bg(PANEL)),
        columns[1],
    );
}

fn field_detail(wizard: &ProfileWizard, field: &ConfigField) -> Text<'static> {
    let current = wizard
        .layer
        .get(field.key)
        .map(|value| format!("{value:?} (profile override)"))
        .unwrap_or_else(|| format!("{} (inherited default)", default_text(field.default)));
    let env = field.env_compat.unwrap_or("none");
    Text::from(vec![
        Line::from(Span::styled(
            field.key.to_owned(),
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        detail_line("Legacy key", field.legacy_key),
        detail_line("Category", &format!("{:?}", field.category)),
        detail_line("Scope", &format!("{:?}", field.scope)),
        detail_line("Type / rule", &format!("{:?}", field.rule)),
        detail_line("Default", &default_text(field.default)),
        detail_line("Current", &current),
        detail_line("Environment", env),
        detail_line(
            "Registry",
            if field.registry_allowed {
                "allowed"
            } else {
                "not allowed"
            },
        ),
        detail_line(
            "Experimental",
            if field.experimental { "yes" } else { "no" },
        ),
        Line::from(""),
        Line::from(Span::styled(
            "What it controls",
            Style::default().fg(MUTED).add_modifier(Modifier::BOLD),
        )),
        Line::from(field.help.to_owned()),
    ])
}

fn detail_line(label: &str, value: &str) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("{label:<13}"), Style::default().fg(MUTED)),
        Span::styled(value.to_owned(), Style::default().fg(TEXT)),
    ])
}

fn default_text(value: DefaultValue) -> String {
    match value {
        DefaultValue::Bool(value) => value.to_string(),
        DefaultValue::Integer(value) => value.to_string(),
        DefaultValue::Float(value) => value.to_string(),
        DefaultValue::String(value) => format!("\"{value}\""),
        DefaultValue::Null => "null".to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        sync::atomic::{AtomicU64, Ordering},
    };

    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    use hipfire_config::{load_global, ConfigValue};

    use super::*;

    static NEXT_ID: AtomicU64 = AtomicU64::new(0);

    fn test_paths(label: &str) -> ConfigPaths {
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "hipfire-profile-wizard-{label}-{}-{id}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        ConfigPaths::under(&root)
    }

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    #[test]
    fn search_matches_help_legacy_and_environment_metadata() {
        let help_field = fields()
            .iter()
            .find(|field| !field.help.is_empty())
            .unwrap();
        assert!(field_matches(
            help_field,
            help_field.help.split_whitespace().next().unwrap()
        ));
        assert!(field_matches(help_field, help_field.legacy_key));
        let env_field = fields()
            .iter()
            .find(|field| field.env_compat.is_some())
            .unwrap();
        assert!(field_matches(env_field, env_field.env_compat.unwrap()));
    }

    #[test]
    fn search_matches_registry_and_experimental_metadata() {
        let allowed = fields()
            .iter()
            .find(|field| {
                field.registry_allowed
                    && !field_text_contains(field, "not allowed")
                    && !field_text_contains(field, "stable")
                    && !field_text_contains(field, "experimental")
            })
            .expect("registry-allowed field");
        let not_allowed = fields()
            .iter()
            .find(|field| {
                !field.registry_allowed
                    && !field_text_contains(field, "allowed")
                    && !field_text_contains(field, "stable")
                    && !field_text_contains(field, "experimental")
            })
            .expect("registry-not-allowed field");
        assert!(field_matches(allowed, "allowed"));
        assert!(!field_matches(allowed, "not allowed"));
        assert!(field_matches(not_allowed, "not allowed"));
        assert!(!field_matches(not_allowed, "allowed"));

        let experimental = fields()
            .iter()
            .find(|field| {
                field.experimental
                    && !field_text_contains(field, "stable")
                    && !field_text_contains(field, "allowed")
                    && !field_text_contains(field, "not allowed")
            })
            .expect("experimental field");
        let stable = fields()
            .iter()
            .find(|field| {
                !field.experimental
                    && !field_text_contains(field, "experimental")
                    && !field_text_contains(field, "allowed")
                    && !field_text_contains(field, "not allowed")
            })
            .expect("stable field");
        assert!(field_matches(experimental, "experimental"));
        assert!(!field_matches(experimental, "stable"));
        assert!(field_matches(stable, "stable"));
        assert!(!field_matches(stable, "experimental"));
    }

    fn field_text_contains(field: &ConfigField, needle: &str) -> bool {
        let needle = needle.to_ascii_lowercase();
        [
            field.key.to_owned(),
            field.legacy_key.to_owned(),
            field.help.to_owned(),
            field.env_compat.unwrap_or_default().to_owned(),
            format!("{:?}", field.category),
            format!("{:?}", field.scope),
            format!("{:?}", field.rule),
        ]
        .iter()
        .any(|candidate| candidate.to_ascii_lowercase().contains(&needle))
    }

    /// Seed `config.toml` with keys unrelated to every built-in profile except
    /// `serve.port`, which the "default" profile overwrites.
    fn seeded_wizard(label: &str) -> (ProfileWizard, ConfigPaths) {
        let paths = test_paths(label);
        let mut layer = ConfigLayer::default();
        layer.set_cli("hardware.devices", "1,2").unwrap();
        layer.set_cli("serve.port", "11527").unwrap();
        layer.set_cli("serve.default_model", "qwen3.6:27b").unwrap();
        layer.set_cli("kernel.prefill_batched", "false").unwrap();
        layer.set_cli("memory.max_seq", "262144").unwrap();
        write_global_toml(&paths, &layer).unwrap();
        let wizard = ProfileWizard::load_from_paths(paths.clone()).unwrap();
        (wizard, paths)
    }

    fn select_profile(wizard: &mut ProfileWizard, name: &str) {
        wizard.profile_index = wizard
            .profiles
            .iter()
            .position(|profile| profile.name == name)
            .unwrap();
    }

    fn config_bytes(paths: &ConfigPaths) -> Vec<u8> {
        fs::read(&paths.config_toml).unwrap()
    }

    const UNRELATED_KEYS: [&str; 4] = [
        "hardware.devices",
        "serve.default_model",
        "kernel.prefill_batched",
        "memory.max_seq",
    ];

    #[test]
    fn enter_on_a_profile_arms_the_confirm_without_writing() {
        let (mut wizard, paths) = seeded_wizard("enter-arms");
        let before = config_bytes(&paths);
        select_profile(&mut wizard, "default");
        wizard.handle_key(key(KeyCode::Enter));
        assert_eq!(wizard.mode, InputMode::Confirm);
        let pending = wizard.pending.as_ref().expect("pending merge");
        assert_eq!(pending.name, "default");
        assert!(pending.diff.removed.is_empty(), "merge never removes keys");
        assert!(pending
            .diff
            .changed
            .iter()
            .any(|(key, old, new)| key == "serve.port"
                && *old == ConfigValue::Integer(11527)
                && *new == ConfigValue::Integer(11435)));
        assert!(pending
            .diff
            .added
            .iter()
            .any(|(key, _)| key == "serve.host"));
        assert_eq!(config_bytes(&paths), before);
        let _ = fs::remove_dir_all(paths.root);
    }

    #[test]
    fn n_and_esc_cancel_the_confirm_without_writing() {
        for cancel in [KeyCode::Char('n'), KeyCode::Esc] {
            let (mut wizard, paths) = seeded_wizard("cancel");
            let before = config_bytes(&paths);
            let layer_before = wizard.layer.clone();
            select_profile(&mut wizard, "default");
            wizard.handle_key(key(KeyCode::Enter));
            assert_eq!(wizard.mode, InputMode::Confirm);
            wizard.handle_key(key(cancel));
            assert_eq!(wizard.mode, InputMode::Browse, "{cancel:?}");
            assert!(wizard.pending.is_none());
            assert_eq!(wizard.layer, layer_before);
            assert!(wizard.status.contains("not applied"), "{}", wizard.status);
            assert_eq!(config_bytes(&paths), before, "{cancel:?}");
            let _ = fs::remove_dir_all(paths.root);
        }
    }

    #[test]
    fn y_applies_the_merge_and_keeps_unrelated_keys() {
        let (mut wizard, paths) = seeded_wizard("confirm-y");
        select_profile(&mut wizard, "default");
        wizard.handle_key(key(KeyCode::Enter));
        wizard.handle_key(key(KeyCode::Char('y')));
        assert_eq!(wizard.mode, InputMode::Browse);
        assert!(wizard.pending.is_none());
        assert!(!wizard.status_error, "{}", wizard.status);
        let loaded = load_global(&paths).unwrap();
        assert_eq!(loaded.layer, wizard.layer);
        for key in UNRELATED_KEYS {
            assert!(loaded.layer.get(key).is_some(), "{key} was dropped");
        }
        assert_eq!(
            loaded.layer.get("memory.max_seq"),
            Some(&ConfigValue::Integer(262144))
        );
        assert_eq!(
            loaded.layer.get("serve.port"),
            Some(&ConfigValue::Integer(11435))
        );
        assert!(loaded.layer.get("memory.kv_cache").is_none());
        assert!(loaded.layer.get("speculation.dflash").is_none());
        assert_eq!(wizard.active_profile.as_deref(), Some("default"));
        let _ = fs::remove_dir_all(paths.root);
    }

    #[test]
    fn applying_an_already_applied_profile_reports_and_writes_nothing() {
        let (mut wizard, paths) = seeded_wizard("already");
        select_profile(&mut wizard, "hip");
        wizard.handle_key(key(KeyCode::Enter));
        wizard.handle_key(key(KeyCode::Char('y')));
        let before = config_bytes(&paths);
        wizard.handle_key(key(KeyCode::Enter));
        assert_eq!(wizard.mode, InputMode::Browse);
        assert!(wizard.pending.is_none());
        assert_eq!(
            wizard.status,
            "Profile 'hip' is already applied; nothing to change."
        );
        assert_eq!(config_bytes(&paths), before);
        let _ = fs::remove_dir_all(paths.root);
    }

    fn rendered_text(wizard: &ProfileWizard, width: u16, height: u16) -> String {
        let backend = ratatui::backend::TestBackend::new(width, height);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| draw(frame, wizard)).unwrap();
        let buffer = terminal.backend().buffer();
        let mut out = String::new();
        for row in 0..buffer.area.height {
            for column in 0..buffer.area.width {
                out.push_str(buffer[(column, row)].symbol());
            }
            out.push('\n');
        }
        out
    }

    #[test]
    fn confirm_modal_lists_changes_in_toml_form_at_80x24() {
        let (mut wizard, paths) = seeded_wizard("render");
        select_profile(&mut wizard, "default");
        wizard.handle_key(key(KeyCode::Enter));
        let text = rendered_text(&wizard, 80, 24);
        assert!(text.contains("Apply profile 'default'?"), "{text}");
        assert!(text.contains("serve.host"), "{text}");
        assert!(text.contains("generation.loop_guard_threshold"), "{text}");
        assert!(text.contains("serve.port: 11527 → 11435"), "{text}");
        assert!(text.contains("serve.host = \"127.0.0.1\""), "{text}");
        assert!(text.contains("y apply · n/Esc cancel"), "{text}");
        for debug in ["Enum(", "{ min:", "Integer(", "String("] {
            assert!(
                !text.contains(debug),
                "Debug text {debug:?} in modal:\n{text}"
            );
        }
        let _ = fs::remove_dir_all(paths.root);
    }

    #[test]
    fn confirm_lines_truncate_with_a_more_marker() {
        let mut diff = ConfigLayerDiff::default();
        for index in 0..5 {
            diff.added
                .push((format!("key.{index}"), ConfigValue::Integer(index)));
        }
        let lines = confirm_lines(&diff, 3);
        assert_eq!(lines.len(), 3);
        let last: String = lines[2]
            .spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect();
        assert_eq!(last, "… 3 more");
        assert_eq!(confirm_lines(&diff, 5).len(), 5);
        assert!(confirm_lines(&diff, 0).is_empty());
    }

    #[test]
    fn selecting_dev_profile_merges_instead_of_replacing() {
        let (mut wizard, paths) = seeded_wizard("select");
        wizard
            .layer
            .set("generation.temperature", ConfigValue::Float(0.42))
            .unwrap();
        write_global_toml(&paths, &wizard.layer).unwrap();
        select_profile(&mut wizard, "dev");
        wizard.handle_key(key(KeyCode::Enter));
        wizard.handle_key(key(KeyCode::Char('y')));
        let loaded = load_global(&paths).unwrap();
        assert_eq!(loaded.layer, wizard.layer);
        assert_eq!(
            detect_config_profile(&paths, &loaded.layer).as_deref(),
            Some("dev")
        );
        assert_eq!(
            loaded.layer.get("generation.temperature"),
            Some(&ConfigValue::Float(0.42))
        );
        assert_eq!(
            loaded.layer.get("serve.port"),
            Some(&ConfigValue::Integer(11527))
        );
        let _ = fs::remove_dir_all(paths.root);
    }

    #[test]
    fn selecting_hip_profile_persists_redline_opt_out() {
        let paths = test_paths("hip-opt-out");
        let mut wizard = ProfileWizard::load_from_paths(paths.clone()).unwrap();
        wizard.begin_apply("hip");
        wizard.confirm_apply();
        let loaded = load_global(&paths).unwrap();
        assert_eq!(
            loaded.layer.get("replay.backend"),
            Some(&ConfigValue::String("hip".to_owned()))
        );
        let _ = fs::remove_dir_all(paths.root);
    }

    #[test]
    fn creating_custom_profile_snapshots_current_layer() {
        let paths = test_paths("create");
        let mut wizard = ProfileWizard::load_from_paths(paths.clone()).unwrap();
        wizard
            .layer
            .set("generation.temperature", ConfigValue::Float(0.42))
            .unwrap();
        wizard.mode = InputMode::Create;
        wizard.create_name = "lab".to_owned();
        wizard.create_profile();
        assert_eq!(wizard.mode, InputMode::Browse);
        let saved = hipfire_config::load_config_profile(&paths, "lab").unwrap();
        assert_eq!(saved, wizard.layer);
        let _ = fs::remove_dir_all(paths.root);
    }

    #[test]
    fn invalid_name_keeps_create_mode_for_correction() {
        let paths = test_paths("invalid");
        let mut wizard = ProfileWizard::load_from_paths(paths.clone()).unwrap();
        wizard.mode = InputMode::Create;
        wizard.create_name = "../escape".to_owned();
        wizard.handle_key(key(KeyCode::Enter));
        assert_eq!(wizard.mode, InputMode::Create);
        assert!(wizard.status_error);
        assert!(wizard.status.contains("invalid profile name"));
        let _ = fs::remove_dir_all(paths.root);
    }

    #[test]
    fn key_modes_switch_and_search_filters_live() {
        let paths = test_paths("keys");
        let mut wizard = ProfileWizard::load_from_paths(paths.clone()).unwrap();
        wizard.handle_key(key(KeyCode::Char('/')));
        assert_eq!(wizard.mode, InputMode::Search);
        assert_eq!(wizard.focus, Focus::Variables);
        for character in "replay".chars() {
            wizard.handle_key(key(KeyCode::Char(character)));
        }
        assert!(!wizard.filtered_fields.is_empty());
        assert!(wizard.filtered_fields.len() < fields().len());
        wizard.handle_key(key(KeyCode::Esc));
        assert_eq!(wizard.mode, InputMode::Browse);
        assert_eq!(wizard.filtered_fields.len(), fields().len());
        wizard.handle_key(key(KeyCode::Char('n')));
        assert_eq!(wizard.mode, InputMode::Create);
        wizard.handle_key(key(KeyCode::Esc));
        assert_eq!(wizard.mode, InputMode::Browse);
        let _ = fs::remove_dir_all(paths.root);
    }
}
