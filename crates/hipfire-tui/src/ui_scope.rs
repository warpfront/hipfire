// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! The per-model scope screens (spec §3.5, mockups S5, S6 and L3): the
//! scope picker, the per-model overrides list at 80 columns and the models
//! list with a values pane at 160, plus the header text Settings shows for the
//! scope.

use hipfire_config::{effective::Applies, field};
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph},
    Frame,
};

use crate::{
    app::App,
    hipfire::{
        scope_view::{own_values_text, ScopeScreen, ScreenKind},
        settings_model::row_title,
    },
    ui::{block, pad, ACCENT, PANEL, PANEL_2, TEXT, YELLOW},
    ui_settings::{cell, gpu_text, heading, muted, plain, wrap, WIDE_COLS, WIDE_ROWS},
};

/// Width of the models list of the 160-column layout.
const MODELS_W: u16 = 52;

/// What the Settings header says is being edited (`m` changes it).
pub fn editing_text(app: &App, wide: bool) -> String {
    let name = app.scope_model().unwrap_or("all models");
    if wide {
        format!("Editing: {name} (m)")
    } else {
        format!("Editing: {name}  m")
    }
}

fn is_wide(area: Rect) -> bool {
    area.width >= WIDE_COLS && area.height >= WIDE_ROWS
}

/// A scope screen replaces the tab body: the overrides list always, the picker
/// at 160 columns (where it is the models list). Records the layout so the
/// keys know whether ↑↓ walk models or value rows.
pub fn draws_screen(app: &App, area: Rect) -> bool {
    let Some(screen) = app.scope.screen.as_ref() else {
        return false;
    };
    let wide = is_wide(area);
    screen.wide.set(wide);
    screen.kind == ScreenKind::Overrides || wide
}

pub fn draw_screen(frame: &mut Frame, app: &App, area: Rect) {
    let Some(screen) = app.scope.screen.as_ref() else {
        return;
    };
    if is_wide(area) {
        draw_models_and_values(frame, app, screen, area);
    } else {
        draw_overrides(frame, app, screen, area);
    }
}

/// Footer hints of the open screen.
pub fn footer_segments(app: &App) -> Option<Vec<String>> {
    let screen = app.scope.screen.as_ref()?;
    let segs = |list: &[&str]| list.iter().map(|text| (*text).to_owned()).collect();
    if screen.confirm_remove_all {
        let name = screen.selected_entry().map_or("this model", |e| e.name());
        return Some(vec![
            format!("Remove all of {name}'s own values?"),
            "y confirm".to_owned(),
            "n / Esc cancel".to_owned(),
        ]);
    }
    Some(if screen.wide.get() {
        segs(&[
            "↑↓ model",
            "←→ value",
            "Enter edit this model in Settings",
            "Del remove selected value",
            "D remove all values for this model",
            "Esc back",
        ])
    } else if screen.kind == ScreenKind::Overrides {
        segs(&[
            "↑↓ move",
            "Enter open in Settings",
            "Del remove",
            "Esc back",
        ])
    } else {
        segs(&["↑↓ move", "Enter edit this scope", "Esc cancel"])
    })
}

fn selected_style(selected: bool, base: Style) -> Style {
    if selected {
        Style::default().fg(ACCENT).bg(PANEL_2)
    } else {
        base
    }
}

fn marker(selected: bool) -> &'static str {
    if selected {
        "> "
    } else {
        "  "
    }
}

fn all_models_text(app: &App, key: &str) -> Option<String> {
    app.scope.all_models_value(key).map(|(value, _)| value)
}

fn count_line(screen: &ScopeScreen) -> String {
    let (values, models) = screen.totals();
    format!(
        "{values} override{} on {models} model{}. Models without a row follow \"all models\".",
        if values == 1 { "" } else { "s" },
        if models == 1 { "" } else { "s" },
    )
}

// ---------------------------------------------------------------------------
// S6: the overrides list at 80 columns
// ---------------------------------------------------------------------------

fn draw_overrides(frame: &mut Frame, app: &App, screen: &ScopeScreen, area: Rect) {
    let rect = pad(area, 1, 0);
    let inner_h = rect.height.saturating_sub(2) as usize;
    let mut lines = vec![
        plain("Settings that apply to one model only (~/.hipfire/models.toml)."),
        Line::from(""),
    ];
    let rows = screen.flat_rows();
    if rows.is_empty() {
        lines.push(plain(
            "No per-model overrides yet. Every model follows \"all models\".",
        ));
        lines.push(Line::from(""));
        lines.push(muted(
            "Give one model its own value: m picks the model in Settings,",
        ));
        lines.push(muted("or hipfire config <model> set <key> <value>."));
    } else {
        let value_w = rows
            .iter()
            .map(|&(entry, value)| screen.entries[entry].own[value].text().chars().count())
            .max()
            .unwrap_or(5)
            .clamp(5, 16)
            + 1;
        let (model_w, setting_w) = (23usize, 22usize);
        lines.push(muted(format!(
            "  {}{}{}(all models)",
            cell("Model", model_w),
            cell("Setting", setting_w),
            cell("Value", value_w),
        )));
        // Header rows above (2 + column header) and below (blank + count).
        let capacity = inner_h.saturating_sub(5).max(1);
        let at = rows
            .iter()
            .position(|row| *row == (screen.entry, screen.value))
            .unwrap_or(0);
        let start = if at < capacity { 0 } else { at + 1 - capacity };
        for (index, &(entry, value)) in rows.iter().enumerate().skip(start).take(capacity) {
            let item = &screen.entries[entry];
            let own = &item.own[value];
            let all =
                all_models_text(app, &own.key).map_or_else(String::new, |text| format!("({text})"));
            let selected = index == at;
            let text = format!(
                "{}{}{}{}{}",
                marker(selected),
                cell(item.name(), model_w),
                cell(&row_title(&own.key), setting_w),
                cell(&own.text(), value_w),
                all,
            );
            lines.push(Line::from(Span::styled(
                text,
                selected_style(selected, Style::default().fg(TEXT)),
            )));
        }
        lines.push(Line::from(""));
        let note = if screen.confirm_remove_all {
            let name = screen.selected_entry().map_or("this model", |e| e.name());
            Line::from(Span::styled(
                format!("Remove all of {name}'s own values? y / n"),
                Style::default().fg(YELLOW),
            ))
        } else {
            plain(count_line(screen))
        };
        lines.push(note);
    }
    frame.render_widget(
        Paragraph::new(lines)
            .block(block("Per-model overrides"))
            .style(Style::default().fg(TEXT).bg(PANEL)),
        rect,
    );
}

// ---------------------------------------------------------------------------
// The picker at 80 columns
// ---------------------------------------------------------------------------

/// The `m` picker over the Settings tab at 80 columns.
pub fn draw_picker_popup(frame: &mut Frame, app: &App, area: Rect) {
    let Some(screen) = app.scope.screen.as_ref() else {
        return;
    };
    if screen.kind != ScreenKind::Picker {
        return;
    }
    let width = 64u16.min(area.width.saturating_sub(2));
    let inner_w = width.saturating_sub(2) as usize;
    let total = screen.entries.len();
    let capacity = (area.height.saturating_sub(6) as usize).clamp(1, total.max(1));
    let start = if screen.entry < capacity {
        0
    } else {
        screen.entry + 1 - capacity
    };
    let mut lines = vec![
        muted("Settings edit one model's own values, or all models."),
        Line::from(""),
    ];
    for (index, entry) in screen.entries.iter().enumerate().skip(start).take(capacity) {
        let selected = index == screen.entry;
        let (count, state) = match (&entry.model, entry.state) {
            (Some(_), Some(state)) => (own_values_text(entry.own.len()), state.label()),
            _ => (String::new(), ""),
        };
        let name_w = inner_w.saturating_sub(2 + 14 + 12).max(12);
        let text = format!(
            "{}{}{}{}",
            marker(selected),
            cell(entry.name(), name_w),
            cell(&count, 14),
            state
        );
        lines.push(Line::from(Span::styled(
            text,
            selected_style(selected, Style::default().fg(TEXT)),
        )));
    }
    lines.push(Line::from(""));
    lines.push(muted("Own values live in ~/.hipfire/models.toml."));
    let height = (lines.len() as u16 + 2).min(area.height);
    let rect = Rect {
        x: area.x + area.width.saturating_sub(width) / 2,
        y: area.y + area.height.saturating_sub(height) / 2,
        width,
        height,
    };
    frame.render_widget(Clear, rect);
    frame.render_widget(
        Paragraph::new(lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(ACCENT))
                    .title(" Editing: choose a model "),
            )
            .style(Style::default().fg(TEXT).bg(PANEL)),
        rect,
    );
}

// ---------------------------------------------------------------------------
// L3: models list and values pane at 160 columns
// ---------------------------------------------------------------------------

fn draw_models_and_values(frame: &mut Frame, app: &App, screen: &ScopeScreen, area: Rect) {
    let rect = pad(area, 1, 0);
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Min(8)])
        .split(rect);
    frame.render_widget(
        Paragraph::new(format!(
            "GPU: {}    Editing: choose a model (m)",
            gpu_text(app, true)
        ))
        .style(Style::default().fg(TEXT)),
        rows[0],
    );
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(MODELS_W), Constraint::Min(40)])
        .split(rows[1]);
    draw_models_list(frame, screen, cols[0]);
    draw_values_pane(frame, app, screen, cols[1]);
}

fn draw_models_list(frame: &mut Frame, screen: &ScopeScreen, area: Rect) {
    let inner_w = area.width.saturating_sub(2) as usize;
    let inner_h = area.height.saturating_sub(2) as usize;
    let note = wrap(
        "Own values live in ~/.hipfire/models.toml and win over \"All models\" for that model only.",
        inner_w,
    );
    // Intro (2) + column header, and the note under the list.
    let capacity = inner_h.saturating_sub(3 + 1 + note.len()).max(1);
    let start = if screen.entry < capacity {
        0
    } else {
        screen.entry + 1 - capacity
    };
    let mut lines = vec![
        plain("Pick a model to see or change its own values."),
        Line::from(""),
        muted(format!(
            "  {}{}State",
            cell("Model", 25),
            cell("Own values", 12)
        )),
    ];
    for (index, entry) in screen.entries.iter().enumerate().skip(start).take(capacity) {
        let selected = index == screen.entry;
        let (count, state) = match entry.model {
            Some(_) => (
                entry.own.len().to_string(),
                entry.state.map_or("", |state| state.label()),
            ),
            None => ("—".to_owned(), "default"),
        };
        let text = format!(
            "{}{}{}{}",
            marker(selected),
            cell(entry.name(), 25),
            cell(&count, 12),
            state
        );
        lines.push(Line::from(Span::styled(
            text,
            selected_style(selected, Style::default().fg(TEXT)),
        )));
    }
    lines.push(Line::from(""));
    lines.extend(note.into_iter().map(muted));
    frame.render_widget(
        Paragraph::new(lines)
            .block(block("Models"))
            .style(Style::default().fg(TEXT).bg(PANEL)),
        area,
    );
}

fn applies_of(key: &str) -> &'static str {
    field(key)
        .map(|schema| Applies::from_scope(schema.scope).label())
        .unwrap_or("")
}

fn draw_values_pane(frame: &mut Frame, app: &App, screen: &ScopeScreen, area: Rect) {
    let inner_w = area.width.saturating_sub(2) as usize;
    let Some(entry) = screen.selected_entry() else {
        frame.render_widget(block("Own values"), area);
        return;
    };
    let mut lines: Vec<Line<'static>> = Vec::new();
    let sample = entry
        .own
        .first()
        .map_or(("reasoning.effort".to_owned(), "high".to_owned()), |own| {
            (own.key.clone(), own.text())
        });
    let title = match &entry.model {
        Some(model) => {
            lines.push(plain("Values that differ from \"All models\""));
            lines.push(Line::from(""));
            if entry.own.is_empty() {
                lines.push(plain(format!(
                    "  {model} has no values of its own: it follows \"All models\" for every setting."
                )));
            } else {
                lines.push(muted(format!(
                    "  {}{}{}{}Applies",
                    cell("Setting", 25),
                    cell("This model", 18),
                    cell("All models", 18),
                    cell("From", 22),
                )));
                for (index, own) in entry.own.iter().enumerate() {
                    let selected = index == screen.value;
                    let all = all_models_text(app, &own.key).unwrap_or_else(|| "—".to_owned());
                    let text = format!(
                        "{}{}{}{}{}{}",
                        marker(selected),
                        cell(&row_title(&own.key), 25),
                        cell(&format!("● {}", own.text()), 18),
                        cell(&all, 18),
                        cell("this model", 22),
                        applies_of(&own.key),
                    );
                    lines.push(Line::from(Span::styled(
                        text,
                        selected_style(selected, Style::default().fg(TEXT)),
                    )));
                }
            }
            lines.push(Line::from(""));
            for text in wrap(
                &format!(
                    "Everything else follows \"All models\". Press Enter to edit all of this model's settings (Settings opens with Editing: {model})."
                ),
                inner_w,
            ) {
                lines.push(plain(text));
            }
            format!("{model} — own values")
        }
        None => {
            lines.push(plain("Values every model without its own value follows."));
            lines.push(Line::from(""));
            for text in wrap(
                "Press Enter to edit all models in Settings. Pick a model on the left to see or change what it holds of its own.",
                inner_w,
            ) {
                lines.push(plain(text));
            }
            "All models".to_owned()
        }
    };
    lines.push(Line::from(""));
    lines.push(heading("Only all-models (one value per serve process)"));
    for text in wrap(
        "Prompt processing, Flash-Next experts, Flash-Next word table, GPU devices, graph capture and other kernel switches. hipfire refuses them in models.toml (memory and kernel routes are chosen once when serve starts).",
        inner_w.saturating_sub(2),
    ) {
        lines.push(plain(format!("  {text}")));
    }
    lines.push(Line::from(""));
    match &entry.model {
        Some(model) => {
            lines.push(plain(format!(
                "CLI: hipfire config {model} list   ·   hipfire config {model} set {} {}",
                sample.0, sample.1
            )));
            lines.push(plain(format!(
                "     hipfire config {model} reset {}",
                sample.0
            )));
        }
        None => {
            lines.push(plain(
                "CLI: hipfire config list   ·   hipfire config set <key> <value>",
            ));
            lines.push(plain("     hipfire config overrides"));
        }
    }
    if screen.confirm_remove_all {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            format!("Remove all of {}'s own values? y / n", entry.name()),
            Style::default().fg(YELLOW).add_modifier(Modifier::BOLD),
        )));
    }
    frame.render_widget(
        Paragraph::new(lines)
            .block(block(&title))
            .style(Style::default().fg(TEXT).bg(PANEL)),
        area,
    );
}

#[cfg(test)]
mod tests {
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    use hipfire_config::effective::A8Prefill;
    use ratatui::{backend::TestBackend, Terminal};

    use crate::{
        app::Tab,
        hipfire::dashboard::LoadedConfig,
        test_support::{online_dashboard, select_key, Fixture, FixtureBuilder, ACTIVE_MODEL},
        ui::draw,
    };

    use super::*;

    /// A key as the running UI routes it: an open scope screen first.
    fn press(app: &mut App, code: KeyCode) {
        app.handle_tab_key(KeyEvent::new(code, KeyModifiers::NONE));
    }

    fn render_rows(app: &mut App, width: u16, height: u16) -> Vec<String> {
        let mut terminal = Terminal::new(TestBackend::new(width, height)).expect("terminal");
        terminal
            .draw(|frame| draw(frame, app))
            .expect("draw must not panic");
        let buf = terminal.backend().buffer().clone();
        (0..buf.area.height)
            .map(|y| {
                let row: String = (0..buf.area.width).map(|x| buf[(x, y)].symbol()).collect();
                row.trim_end().to_owned()
            })
            .collect()
    }

    fn render_text(app: &mut App, width: u16, height: u16) -> String {
        render_rows(app, width, height).join("\n") + "\n"
    }

    /// Compare `text` to `tests/snapshots/<name>.txt`; `UPDATE_SNAPSHOTS=1`
    /// rewrites it.
    fn assert_snapshot(name: &str, text: &str) {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/snapshots")
            .join(format!("{name}.txt"));
        if std::env::var_os("UPDATE_SNAPSHOTS").is_some() {
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, text).unwrap();
            return;
        }
        let want = std::fs::read_to_string(&path).unwrap_or_else(|_| {
            panic!(
                "missing snapshot {}; run with UPDATE_SNAPSHOTS=1",
                path.display()
            )
        });
        assert_eq!(
            text, want,
            "snapshot {name} differs; rerun with UPDATE_SNAPSHOTS=1 to accept"
        );
    }

    fn strix() -> FixtureBuilder {
        FixtureBuilder::new("gfx1151")
            .a8(A8Prefill::OptIn)
            .config("[reasoning]\neffort = \"low\"\n")
    }

    /// Four overrides on three models, over `reasoning.effort = low`.
    fn four_overrides() -> FixtureBuilder {
        strix()
            .model_value(ACTIVE_MODEL, "reasoning.effort", "high")
            .model_value(ACTIVE_MODEL, "memory.max_seq", "131072")
            .model_value("qwen3.6:35b-a3b", "memory.kv_cache", "q8")
            .model_value("gemma4:12b", "speculation.mode", "off")
    }

    fn scope_to_active_model(fixture: &mut Fixture) {
        press(&mut fixture.app, KeyCode::Char('m'));
        press(&mut fixture.app, KeyCode::Down);
        press(&mut fixture.app, KeyCode::Enter);
    }

    fn assert_rows_fit(rows: &[String], width: u16, context: &str) {
        for row in rows {
            assert!(
                Line::from(row.as_str()).width() <= width as usize,
                "{context}: row wider than {width}: {row:?}"
            );
        }
    }

    #[test]
    fn golden_scope_s5_80x24() {
        let mut fixture = strix()
            .model_value(ACTIVE_MODEL, "reasoning.effort", "high")
            .build();
        scope_to_active_model(&mut fixture);
        select_key(&mut fixture.app, "reasoning.effort");
        assert_snapshot("scope_s5_80x24", &render_text(&mut fixture.app, 80, 24));
    }

    #[test]
    fn golden_overrides_s6_80x24() {
        let mut fixture = four_overrides().build();
        press(&mut fixture.app, KeyCode::Char('o'));
        assert_snapshot("overrides_s6_80x24", &render_text(&mut fixture.app, 80, 24));
    }

    #[test]
    fn golden_overrides_l3_160x48() {
        let mut fixture = four_overrides()
            .model("qwen3.8:flash-next", true)
            .model("qwen3.6:35b-a3b", true)
            .model("gemma4:12b", true)
            .model("qwen3.5:9b", true)
            .build();
        fixture.app.dashboard = Some(online_dashboard(
            Some(LoadedConfig {
                model: ACTIVE_MODEL.into(),
                kv_mode: None,
                kv_backend: None,
                max_seq: None,
            }),
            3600,
        ));
        press(&mut fixture.app, KeyCode::Char('o'));
        assert_snapshot(
            "overrides_l3_160x48",
            &render_text(&mut fixture.app, 160, 48),
        );
    }

    #[test]
    fn header_names_the_scope_and_its_key_in_both_layouts() {
        let mut fixture = strix().build();
        let rows = render_rows(&mut fixture.app, 80, 24);
        assert!(rows[5].ends_with("Editing: all models  m"), "{:?}", rows[5]);
        let rows = render_rows(&mut fixture.app, 160, 48);
        assert!(rows
            .iter()
            .any(|row| row.contains("Editing: all models (m)")));
        scope_to_active_model(&mut fixture);
        let rows = render_rows(&mut fixture.app, 80, 24);
        assert!(
            rows[5].ends_with(&format!("Editing: {ACTIVE_MODEL}  m")),
            "{:?}",
            rows[5]
        );
        assert!(rows[7].contains(&format!("Quality · values for {ACTIVE_MODEL} only")));
        let rows = render_rows(&mut fixture.app, 160, 48);
        assert!(rows
            .iter()
            .any(|row| row.contains(&format!("Editing: {ACTIVE_MODEL} (m)"))));
        assert!(rows
            .iter()
            .any(|row| row.contains(&format!("Quality · values for {ACTIVE_MODEL} only"))));
    }

    #[test]
    fn locked_rows_read_all_models_only_in_both_layouts() {
        for (width, height) in [(80, 24), (160, 48)] {
            let mut fixture = strix().build();
            scope_to_active_model(&mut fixture);
            let rows = render_rows(&mut fixture.app, width, height);
            let line = rows
                .iter()
                .find(|row| row.contains("Prompt processing") && row.contains("Fast"))
                .expect("Prompt processing row");
            assert!(
                line.contains("all models only (locked)"),
                "{width}: {line:?}"
            );
            assert_rows_fit(&rows, width, "model scope");
        }
    }

    #[test]
    fn all_models_scope_shows_this_model_and_the_warning_line() {
        let mut fixture = strix()
            .model_value(ACTIVE_MODEL, "reasoning.effort", "high")
            .build();
        select_key(&mut fixture.app, "reasoning.effort");
        let rows = render_rows(&mut fixture.app, 80, 24);
        assert!(
            rows.iter()
                .any(|row| row.contains("Thinking effort") && row.contains("this model")),
            "{rows:#?}"
        );
        assert!(rows.iter().any(|row| row.contains(
            "qwen3.8:27b has its own value, so this change won't affect it. m to edit it."
        )));
    }

    #[test]
    fn models_tab_shows_the_own_values_count_within_80_columns() {
        let mut fixture = strix()
            .model_value(ACTIVE_MODEL, "reasoning.effort", "high")
            .model_value(ACTIVE_MODEL, "memory.max_seq", "131072")
            .build();
        fixture.app.tab = Tab::Models;
        fixture
            .app
            .registry
            .expanded_groups
            .insert("qwen3.8".into());
        let rows = render_rows(&mut fixture.app, 80, 24);
        let line = rows
            .iter()
            .find(|row| row.contains("remote") && row.contains(ACTIVE_MODEL))
            .expect("model row");
        assert!(line.contains("2 own values"), "{line:?}");
        assert_rows_fit(&rows, 80, "models tab");
        assert!(rows[22].contains("o overrides"), "{:?}", rows[22]);
    }

    #[test]
    fn picker_popup_lists_the_scopes_at_80_columns() {
        let mut fixture = strix()
            .model_value(ACTIVE_MODEL, "reasoning.effort", "high")
            .model("qwen3.5:9b", true)
            .build();
        press(&mut fixture.app, KeyCode::Char('m'));
        let rows = render_rows(&mut fixture.app, 80, 24);
        let text = rows.join("\n");
        assert!(text.contains("Editing: choose a model"), "{text}");
        assert!(text.contains("All models"));
        assert!(text.contains("1 own value"));
        assert!(text.contains("on disk"));
        assert_rows_fit(&rows, 80, "picker");
        assert!(rows[22].contains("Enter edit this scope"), "{:?}", rows[22]);
    }

    #[test]
    fn overrides_screen_swallows_every_key_but_esc_closes_it() {
        let mut fixture = four_overrides().build();
        press(&mut fixture.app, KeyCode::Char('o'));
        for code in [
            KeyCode::Char('q'),
            KeyCode::Tab,
            KeyCode::Char('?'),
            KeyCode::Char('r'),
        ] {
            assert!(
                crate::handle_key(&mut fixture.app, KeyEvent::new(code, KeyModifiers::NONE))
                    == false
            );
            assert!(fixture.app.scope_screen_open(), "{code:?}");
            assert_eq!(fixture.app.tab, Tab::Settings);
        }
        crate::handle_key(
            &mut fixture.app,
            KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE),
        );
        assert!(!fixture.app.scope_screen_open());
    }

    #[test]
    fn overrides_empty_state_and_long_lists_fit() {
        let mut fixture = strix().build();
        press(&mut fixture.app, KeyCode::Char('o'));
        let rows = render_rows(&mut fixture.app, 80, 24);
        assert!(rows
            .iter()
            .any(|row| row.contains("No per-model overrides yet")));
        let mut builder = strix();
        for index in 0..30 {
            builder = builder.model_value(&format!("m{index:02}"), "reasoning.effort", "high");
        }
        let mut fixture = builder.build();
        press(&mut fixture.app, KeyCode::Char('o'));
        for _ in 0..25 {
            press(&mut fixture.app, KeyCode::Down);
        }
        for (width, height) in [(80, 24), (160, 48)] {
            let rows = render_rows(&mut fixture.app, width, height);
            assert_rows_fit(&rows, width, "long overrides");
        }
    }
}
