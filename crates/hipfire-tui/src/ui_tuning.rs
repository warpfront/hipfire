// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! The Tuning screens (spec §4): the preset chooser (S2), the preview (S3),
//! the 160×48 cards + preview table (L2) and the undo confirmation. All state
//! lives in [`TuningView`]; every number comes from `hipfire_config`.

use hipfire_config::{
    evidence::{format_closer, format_slower, EvidenceRow},
    tuning::{Conflict, PlanLine, Preset, TuningStatus},
};
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame,
};

use crate::{
    app::App,
    hipfire::tuning_view::{changes_label, rows, takes_effect_text, Preview, Stage, TuningView},
    ui::{block, pad, ACCENT, GREEN, MUTED, PANEL, PANEL_2, TEXT, YELLOW},
    ui_settings::{cell, gpu_text, wrap},
};

/// Terminal size from which the cards + preview table layout shows.
const WIDE_COLS: u16 = 160;
const WIDE_ROWS: u16 = 48;
/// Height of a preset card at the wide layout (borders + 5 text rows).
const CARD_H: u16 = 7;

pub fn is_wide(area: Rect) -> bool {
    area.width >= WIDE_COLS && area.height >= WIDE_ROWS
}

/// Footer hint segments of the Tuning screens, highest priority first.
pub fn footer_segments(app: &App, view: &TuningView) -> Vec<String> {
    let segs = |list: &[&str]| list.iter().map(|text| (*text).to_owned()).collect();
    match view.stage {
        Stage::Chooser => segs(&[
            "Enter preview",
            "↑↓ choose",
            "u undo last tuning",
            "Esc close",
        ]),
        Stage::UndoConfirm => segs(&["y undo", "Esc cancel"]),
        Stage::Preview => {
            let preview = view.preview(&app.env_layer);
            let save =
                (preview.count > 0).then(|| format!("y save {}", changes_label(preview.count)));
            let mut out = Vec::new();
            if app.tuning_wide {
                out.push("←→ preset".to_owned());
            }
            out.extend(save);
            if app.tuning_wide && preview.count > 0 {
                out.push("Y save and restart serve".to_owned());
            }
            out.push("Space tick/untick".to_owned());
            out.push("↑↓ move".to_owned());
            if app.tuning_wide {
                out.push("u undo last tuning".to_owned());
                out.push("Esc back to Settings".to_owned());
            } else {
                out.push("Esc back".to_owned());
            }
            out
        }
    }
}

pub fn draw_tuning(frame: &mut Frame, app: &App, view: &TuningView, area: Rect) {
    let area = pad(area, 1, 0);
    match view.stage {
        Stage::UndoConfirm => draw_undo(frame, app, view, area),
        _ if app.tuning_wide => draw_wide(frame, app, view, area),
        Stage::Chooser => draw_chooser(frame, app, view, area),
        Stage::Preview => draw_preview(frame, app, view, area),
    }
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

fn text_line(text: impl Into<String>) -> Line<'static> {
    Line::from(Span::styled(text.into(), Style::default().fg(TEXT)))
}

fn muted_line(text: impl Into<String>) -> Line<'static> {
    Line::from(Span::styled(text.into(), Style::default().fg(MUTED)))
}

fn heading_line(text: impl Into<String>) -> Line<'static> {
    Line::from(Span::styled(
        text.into(),
        Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
    ))
}

fn panel(frame: &mut Frame, area: Rect, title: &str, lines: Vec<Line<'static>>) {
    frame.render_widget(
        Paragraph::new(lines)
            .block(block(title))
            .style(Style::default().fg(TEXT).bg(PANEL)),
        area,
    );
}

/// `lines` wrapped to `width`, each continuation indented by `indent`.
fn wrapped(text: &str, width: usize, indent: usize) -> Vec<Line<'static>> {
    wrap(text, width.saturating_sub(indent).max(8))
        .into_iter()
        .map(|line| text_line(format!("{}{line}", " ".repeat(indent))))
        .collect()
}

fn gpu_title(app: &App) -> String {
    gpu_text(app, false)
}

/// `Current: …` under the chooser (spec §4 S2).
fn current_text(app: &App, view: &TuningView) -> String {
    match view.status(&app.arch) {
        TuningStatus::Balanced => {
            "Current: Balanced (no preset-owned settings in config.toml).".to_owned()
        }
        TuningStatus::Preset(preset) => format!(
            "Current: {} (its settings are in config.toml; u undoes the last change).",
            preset.title()
        ),
        TuningStatus::PresetEdited(preset) => format!(
            "Current: {} (edited): a setting it wrote was changed by hand.",
            preset.title()
        ),
        TuningStatus::Custom => {
            "Current: Custom (no preset applied, but tuning keys in config.toml are set by hand)."
                .to_owned()
        }
    }
}

// ---------------------------------------------------------------------------
// S2: the chooser
// ---------------------------------------------------------------------------

fn draw_chooser(frame: &mut Frame, app: &App, view: &TuningView, area: Rect) {
    let width = area.width.saturating_sub(2) as usize;
    let mut lines = vec![
        text_line("Pick how hipfire should trade speed for answer quality on this machine."),
        text_line("You will see every change before anything is saved."),
        text_line(""),
    ];
    for (index, preset) in Preset::ALL.iter().enumerate() {
        let selected = index == view.cursor;
        let dot = if *preset == view.current() {
            '●'
        } else {
            '○'
        };
        let name = cell(preset.title(), 15);
        let body = wrap(
            &format!("{} {}", preset.one_liner(), preset.second_line()),
            width.saturating_sub(19),
        );
        let style = if selected {
            Style::default()
                .fg(TEXT)
                .bg(PANEL_2)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(TEXT)
        };
        for (row, text) in body.iter().enumerate() {
            let lead = if row == 0 {
                format!("{} {dot} {name}", if selected { '>' } else { ' ' })
            } else {
                " ".repeat(19)
            };
            lines.push(Line::from(Span::styled(format!("{lead}{text}"), style)));
        }
    }
    lines.push(text_line(""));
    lines.push(text_line(current_text(app, view)));
    panel(
        frame,
        area,
        &format!("Tuning for {} · all models", gpu_title(app)),
        lines,
    );
}

// ---------------------------------------------------------------------------
// Preview content shared by S3 and L2
// ---------------------------------------------------------------------------

/// The checks of the preview: `(ok, text)`. `!!` rows always show; `ok` rows
/// only at the wide layout, where there is room.
fn checks(app: &App, view: &TuningView, preview: &Preview, wide: bool) -> Vec<(bool, String)> {
    let mut out = Vec::new();
    let env_conflicts: Vec<&Conflict> = preview
        .conflicts
        .iter()
        .filter(|conflict| matches!(conflict, Conflict::Env { .. }))
        .collect();
    for conflict in &preview.conflicts {
        match conflict {
            Conflict::Env { key, var } => out.push((
                false,
                format!("{var} is set in this shell and overrides {key}: unset it, or the saved value is ignored."),
            )),
            Conflict::PerModel { model, key } => out.push((
                false,
                format!("{model} sets {key} itself: this preset does not change that model."),
            )),
        }
    }
    if app.dashboard.as_ref().is_some_and(|dash| dash.serve_up) && preview.count > 0 {
        let model = app
            .serve_loaded_config()
            .map(|loaded| loaded.model.clone())
            .or_else(|| app.dashboard.as_ref().and_then(|dash| dash.model.clone()))
            .unwrap_or_else(|| app.active_model.clone());
        out.push((
            false,
            format!("serve is running {model} with the old values: restart serve to use the new ones (Y saves and restarts)."),
        ));
    }
    if wide {
        let keys: Vec<&str> = view.plan().keys().into_iter().collect();
        if env_conflicts.is_empty() && !keys.is_empty() {
            out.push((
                true,
                format!(
                    "No environment variable overrides {} in this shell.",
                    keys.join(", ")
                ),
            ));
        }
        if !preview
            .conflicts
            .iter()
            .any(|conflict| matches!(conflict, Conflict::PerModel { .. }))
            && !keys.is_empty()
        {
            out.push((
                true,
                "No model has its own value for these keys (models.toml).".to_owned(),
            ));
        }
    }
    out
}

fn check_lines(checks: &[(bool, String)], width: usize) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    for (ok, text) in checks {
        let (tag, color) = if *ok { ("ok", GREEN) } else { ("!!", YELLOW) };
        for (row, part) in wrap(text, width.saturating_sub(6)).into_iter().enumerate() {
            let lead = if row == 0 {
                format!("  {tag}  ")
            } else {
                "      ".to_owned()
            };
            lines.push(Line::from(vec![
                Span::styled(
                    lead,
                    Style::default().fg(color).add_modifier(Modifier::BOLD),
                ),
                Span::styled(part, Style::default().fg(TEXT)),
            ]));
        }
    }
    lines
}

/// `Answers 43% closer to the full model · prompts 1.6× slower · 27B-class
/// dense models`, with the gaps said plainly.
fn evidence_text(line: &PlanLine, width: usize) -> String {
    let Some(row) = line.evidence else {
        return "not measured on this GPU".to_owned();
    };
    let closer = format_closer(row).map(|closer| {
        if closer.ends_with("further") {
            format!("Answers {closer} from the full model")
        } else {
            format!("Answers {closer} to the full model")
        }
    });
    let slower = format_slower(row).map(|slower| format!("prompts {slower}"));
    let mut parts = Vec::new();
    match (closer, slower) {
        (None, None) => {
            parts.push(match row.caveat {
                Some(caveat) => format!("not measured on this GPU ({caveat})"),
                None => "not measured on this GPU".to_owned(),
            });
        }
        (closer, slower) => {
            parts.push(
                closer.unwrap_or_else(|| "quality gain: not measured on this GPU".to_owned()),
            );
            parts.push(
                slower.unwrap_or_else(|| "prompt speed: not measured on this GPU".to_owned()),
            );
        }
    }
    let mut text = parts.join(" · ");
    // What the line applies to, when it still fits on one row.
    if let Some(scope) = line.scope {
        if text.chars().count() + 3 + scope.chars().count() <= width {
            text.push_str(" · ");
            text.push_str(scope);
        }
    }
    text
}

/// The preview's list: the intro and one block per plan line (two or more
/// rows each). Returns the lines and the row where the cursor line starts.
fn plan_lines(view: &TuningView, preview: &Preview, width: usize) -> (Vec<Line<'static>>, usize) {
    let plan = view.plan();
    let mut lines = vec![
        text_line("Only the ticked settings change; everything else you set stays as it is."),
        text_line(""),
    ];
    let mut anchor = 0;
    if plan.lines.is_empty() {
        let text = match plan.preset {
            Preset::Balanced => {
                "Balanced writes nothing. It gives back the settings the applied preset still owns."
            }
            _ => "This preset has no lines for this GPU.",
        };
        lines.extend(wrapped(text, width, 0));
    }
    for (index, line) in plan.lines.iter().enumerate() {
        if index == view.selected {
            anchor = lines.len();
        }
        let ticked = view.is_ticked(line);
        let mark = if ticked { "[x]" } else { "[ ]" };
        let style = if !line.requirement.met {
            Style::default().fg(MUTED)
        } else if index == view.selected {
            Style::default()
                .fg(TEXT)
                .bg(PANEL_2)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(TEXT)
        };
        lines.push(Line::from(Span::styled(
            format!("{mark} {}{}", cell(line.title, 25), line.summary),
            style,
        )));
        for row in wrap(
            &evidence_text(line, width.saturating_sub(4)),
            width.saturating_sub(4),
        ) {
            lines.push(muted_line(format!("    {row}")));
        }
        if !line.requirement.met {
            let reason = line
                .requirement
                .reason
                .as_deref()
                .unwrap_or("requirement not met");
            let text = match line.requirement.fix.as_deref() {
                Some(fix) => format!("Not ticked: {reason}. Fix: {fix}"),
                None => format!("Not ticked: {reason}."),
            };
            for row in wrap(&text, width.saturating_sub(4)) {
                lines.push(Line::from(Span::styled(
                    format!("    {row}"),
                    Style::default().fg(YELLOW),
                )));
            }
        }
    }
    for note in &plan.notes {
        lines.push(muted_line(note.clone()));
    }
    if !preview.given_back.is_empty() {
        lines.push(text_line(""));
        lines.push(heading_line(format!(
            "Gives back what {} set:",
            view.current().title()
        )));
        for (key, from, to) in &preview.given_back {
            lines.push(text_line(format!("  {} {from} → {to}", cell(key, 32))));
        }
    }
    if !preview.kept.is_empty() {
        lines.push(text_line(""));
        lines.push(muted_line(format!(
            "Kept as you set them: {}",
            preview.kept.join(", ")
        )));
    }
    (lines, anchor)
}

/// Rows pinned under the list: the typing note, where it saves, the checks.
fn pinned_lines(
    app: &App,
    view: &TuningView,
    preview: &Preview,
    width: usize,
) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    let ticked: Vec<&PlanLine> = view
        .plan()
        .lines
        .iter()
        .filter(|line| view.is_ticked(line))
        .collect();
    let typing_unchanged = !ticked.is_empty()
        && ticked.iter().all(|line| {
            line.evidence
                .and_then(|row| row.typing)
                .is_some_and(|typing| typing.starts_with("unchanged"))
        });
    if typing_unchanged {
        lines.push(text_line(
            "Typing speed (decode) is unchanged for the ticked lines.",
        ));
    }
    if preview.count == 0 {
        lines.push(text_line(format!(
            "Nothing to change: {} is already in effect.",
            view.preset().title()
        )));
    } else {
        lines.push(text_line(format!(
            "Saved to config.toml for all models · {}.",
            takes_effect_text(preview.applies)
        )));
    }
    let checks = checks(app, view, preview, false);
    if !checks.is_empty() {
        lines.push(text_line(""));
        lines.push(heading_line("Checks"));
        lines.extend(check_lines(&checks, width));
    }
    lines
}

// ---------------------------------------------------------------------------
// S3: the 80-column preview
// ---------------------------------------------------------------------------

fn draw_preview(frame: &mut Frame, app: &App, view: &TuningView, area: Rect) {
    let title = format!("Preview · {} · {}", view.preset().title(), gpu_title(app));
    let frame_block = block(&title);
    let inner = frame_block.inner(area);
    frame.render_widget(frame_block, area);
    let width = inner.width as usize;
    let preview = view.preview(&app.env_layer);
    let (list, anchor) = plan_lines(view, &preview, width);
    let pinned = pinned_lines(app, view, &preview, width);
    let pinned_h = (pinned.len() as u16 + 1).min(inner.height.saturating_sub(3));
    let parts = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(2), Constraint::Length(pinned_h)])
        .split(inner);
    // Scroll the list so the cursor line stays visible.
    let capacity = parts[0].height as usize;
    let start = if anchor + 2 <= capacity {
        0
    } else {
        (anchor + 2 - capacity).min(list.len().saturating_sub(capacity))
    };
    let shown: Vec<Line<'static>> = list.into_iter().skip(start).take(capacity).collect();
    frame.render_widget(
        Paragraph::new(shown).style(Style::default().fg(TEXT).bg(PANEL)),
        parts[0],
    );
    let mut bottom = vec![text_line("")];
    bottom.extend(pinned);
    frame.render_widget(
        Paragraph::new(bottom).style(Style::default().fg(TEXT).bg(PANEL)),
        parts[1],
    );
}

// ---------------------------------------------------------------------------
// L2: cards + preview table
// ---------------------------------------------------------------------------

fn draw_wide(frame: &mut Frame, app: &App, view: &TuningView, area: Rect) {
    let parts = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(CARD_H), Constraint::Min(10)])
        .split(area);
    let cards = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Ratio(1, 3),
            Constraint::Ratio(1, 3),
            Constraint::Ratio(1, 3),
        ])
        .split(parts[0]);
    for (index, preset) in Preset::ALL.iter().enumerate() {
        draw_card(
            frame,
            app,
            view,
            *preset,
            index == view.cursor,
            cards[index],
        );
    }
    draw_wide_preview(frame, app, view, parts[1]);
}

fn draw_card(
    frame: &mut Frame,
    _app: &App,
    view: &TuningView,
    preset: Preset,
    selected: bool,
    area: Rect,
) {
    let current = preset == view.current();
    let title = format!(
        " {} {}{} ",
        if selected { '>' } else { ' ' },
        preset.title(),
        if current { "  (current)" } else { "" }
    );
    let width = area.width.saturating_sub(2) as usize;
    let mut lines = Vec::new();
    for sentence in [preset.one_liner(), preset.second_line()] {
        lines.extend(wrapped(sentence, width, 0));
    }
    let count = view.default_change_count(preset);
    lines.push(muted_line(if current && count == 0 {
        "in effect now".to_owned()
    } else if selected {
        format!("{} · preview below", changes_label(count))
    } else {
        changes_label(count)
    }));
    let border = if selected {
        Style::default().fg(ACCENT)
    } else {
        Style::default().fg(PANEL_2)
    };
    frame.render_widget(
        Paragraph::new(lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(border)
                    .title(Span::styled(
                        title,
                        Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
                    ))
                    .style(Style::default().bg(PANEL)),
            )
            .style(Style::default().fg(TEXT).bg(PANEL)),
        area,
    );
}

/// Column widths of the preview table (the rest goes to `Typing speed`).
const COL_TICK: usize = 4;
const COL_SETTING: usize = 24;
const COL_CHANGE: usize = 42;
const COL_CLOSER: usize = 30;
const COL_PROMPT: usize = 30;

fn table_row(cells: [String; 6], style: Style, width: usize) -> Line<'static> {
    let fixed = [COL_TICK, COL_SETTING, COL_CHANGE, COL_CLOSER, COL_PROMPT];
    let used: usize = fixed.iter().sum();
    let mut text = String::new();
    for (index, content) in cells.iter().enumerate() {
        let column = fixed
            .get(index)
            .copied()
            .unwrap_or_else(|| width.saturating_sub(used));
        text.push_str(&cell(content, column));
    }
    Line::from(Span::styled(text.trim_end().to_owned(), style))
}

fn thousands(value: f64) -> String {
    let digits = format!("{:.0}", value.round());
    let mut out = String::new();
    for (index, ch) in digits.chars().rev().enumerate() {
        if index > 0 && index % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out.chars().rev().collect()
}

/// The numbers of one wide table row: `(closer, kld, speed, speed metric)`.
fn row_numbers(row: &EvidenceRow) -> (String, String, String, String) {
    let closer = format_closer(row).unwrap_or_else(|| "not measured here".to_owned());
    let kld = match (row.kld_from, row.kld_to) {
        (Some(from), Some(to)) => {
            format!("{} {from:.4} → {to:.4}", row.kld_metric.unwrap_or("KLD"))
        }
        _ => String::new(),
    };
    let speed = match (row.speed_from, row.speed_to) {
        (Some(from), Some(to)) => format!("{} → {} tok/s", thousands(from), thousands(to)),
        _ => "not measured here".to_owned(),
    };
    let metric = match (row.speed_metric, format_slower(row)) {
        (Some(metric), Some(slower)) => format!(
            "{} · {slower}",
            metric.trim_end_matches(" (≈)").trim_end_matches(" tok/s")
        ),
        (None, Some(slower)) => slower,
        _ => String::new(),
    };
    (closer, kld, speed, metric)
}

fn draw_wide_preview(frame: &mut Frame, app: &App, view: &TuningView, area: Rect) {
    let title = format!("Preview · {} · {}", view.preset().title(), gpu_title(app));
    let frame_block = block(&title);
    let inner = frame_block.inner(area);
    frame.render_widget(frame_block, area);
    let width = inner.width as usize;
    let preview = view.preview(&app.env_layer);
    let plan = view.plan();
    let mut lines = vec![
        text_line(
            "Only the ticked rows are written to ~/.hipfire/config.toml (all models). Nothing else you set is touched; u undoes it later.",
        ),
        text_line(""),
    ];
    if plan.lines.is_empty() {
        let text = match plan.preset {
            Preset::Balanced => {
                "Balanced writes nothing. It gives back the settings the applied preset still owns."
            }
            _ => "This preset has no lines for this GPU.",
        };
        lines.push(text_line(text));
    } else {
        lines.push(table_row(
            [
                String::new(),
                "Setting".to_owned(),
                "Change".to_owned(),
                "Closer to full model".to_owned(),
                "Prompt speed".to_owned(),
                "Typing speed".to_owned(),
            ],
            Style::default().fg(MUTED),
            width,
        ));
    }
    for (index, line) in plan.lines.iter().enumerate() {
        let ticked = view.is_ticked(line);
        let style = if !line.requirement.met {
            Style::default().fg(MUTED)
        } else if index == view.selected {
            Style::default()
                .fg(TEXT)
                .bg(PANEL_2)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(TEXT)
        };
        let writes = line
            .writes
            .iter()
            .map(|(key, value)| format!("{key} = {}", value.to_toml_string()))
            .collect::<Vec<_>>()
            .join("; ");
        let mark = (if ticked { "[x]" } else { "[ ]" }).to_owned();
        let unknown = "not measured here".to_owned();
        let (first, second, measured) = match line.evidence {
            Some(row) => {
                let (closer, kld, speed, metric) = row_numbers(row);
                let typing = row
                    .typing
                    .map(|typing| typing.split(" (").next().unwrap_or(typing).to_owned())
                    .unwrap_or_else(|| "not measured".to_owned());
                let measured: Vec<&str> = [row.source, row.caveat].into_iter().flatten().collect();
                (
                    [
                        mark,
                        line.title.to_owned(),
                        line.summary.clone(),
                        closer,
                        speed,
                        typing,
                    ],
                    [
                        String::new(),
                        "  writes".to_owned(),
                        writes,
                        kld,
                        metric,
                        String::new(),
                    ],
                    measured.join(" · "),
                )
            }
            None => (
                [
                    mark,
                    line.title.to_owned(),
                    line.summary.clone(),
                    unknown.clone(),
                    unknown,
                    "not measured".to_owned(),
                ],
                [
                    String::new(),
                    "  writes".to_owned(),
                    writes,
                    String::new(),
                    String::new(),
                    String::new(),
                ],
                String::new(),
            ),
        };
        lines.push(table_row(first, style, width));
        lines.push(table_row(second, style, width));
        if !measured.is_empty() {
            lines.push(muted_line(format!("      measured: {measured}")));
        }
        if !line.requirement.met {
            let reason = line
                .requirement
                .reason
                .as_deref()
                .unwrap_or("requirement not met");
            let text = match line.requirement.fix.as_deref() {
                Some(fix) => format!("Not ticked: {reason}. Fix: {fix}"),
                None => format!("Not ticked: {reason}."),
            };
            lines.push(Line::from(Span::styled(
                format!("    {text}"),
                Style::default().fg(YELLOW),
            )));
        }
    }
    for note in &plan.notes {
        lines.push(muted_line(note.clone()));
    }
    if !preview.given_back.is_empty() {
        lines.push(text_line(""));
        lines.push(heading_line(format!(
            "Gives back what {} set",
            view.current().title()
        )));
        for (key, from, to) in &preview.given_back {
            lines.push(text_line(format!("    {} {from} → {to}", cell(key, 36))));
        }
    }
    if !preview.kept.is_empty() {
        lines.push(muted_line(format!(
            "    Kept as you set them: {}",
            preview.kept.join(", ")
        )));
    }
    lines.push(text_line(""));
    lines.push(heading_line("Not changed, and why"));
    for item in &plan.not_changed {
        lines.push(text_line(format!(
            "    {} {}",
            cell(&item.what, 36),
            item.why
        )));
    }
    let checks = checks(app, view, &preview, true);
    if !checks.is_empty() {
        lines.push(text_line(""));
        lines.push(heading_line("Checks"));
        lines.extend(check_lines(&checks, width));
    }
    lines.push(text_line(""));
    lines.push(text_line(format!(
        "After saving, these rows show ● and From \"preset\". Changing one by hand makes the tuning read \"{} (edited)\".",
        view.preset().title()
    )));
    frame.render_widget(
        Paragraph::new(lines).style(Style::default().fg(TEXT).bg(PANEL)),
        inner,
    );
}

// ---------------------------------------------------------------------------
// Undo confirmation
// ---------------------------------------------------------------------------

fn draw_undo(frame: &mut Frame, _app: &App, view: &TuningView, area: Rect) {
    let width = area.width.saturating_sub(2) as usize;
    let mut lines = Vec::new();
    match &view.undo {
        None => lines.push(text_line("Nothing to undo.")),
        Some(outcome) => {
            lines.extend(wrapped(
                "Undo the last tuning action. These settings in ~/.hipfire/config.toml are restored:",
                width,
                0,
            ));
            lines.push(text_line(""));
            let changed = rows(&outcome.net);
            if changed.is_empty() {
                lines.extend(wrapped(
                    "No change to config.toml; only the tuning record goes back.",
                    width,
                    2,
                ));
            }
            for (key, from, to) in changed {
                lines.push(text_line(format!("  {} {from} → {to}", cell(&key, 32))));
            }
            if !outcome.kept.is_empty() {
                lines.push(text_line(""));
                lines.extend(wrapped(
                    &format!(
                        "Kept as you set them (changed since): {}",
                        outcome.kept.join(", ")
                    ),
                    width,
                    0,
                ));
            }
            lines.push(text_line(""));
            lines.push(text_line(format!(
                "Tuning goes back to {}. Press y to undo, Esc to cancel.",
                outcome.restored_preset.title()
            )));
        }
    }
    panel(frame, area, "Undo last tuning", lines);
}

#[cfg(test)]
mod tests {
    use crossterm::event::KeyCode;
    use hipfire_config::effective::A8Prefill;
    use ratatui::{backend::TestBackend, Terminal};

    use crate::{
        app::App,
        test_support::{press, FixtureBuilder},
        ui::draw,
    };

    fn render_text(app: &mut App, width: u16, height: u16) -> String {
        let mut terminal = Terminal::new(TestBackend::new(width, height)).expect("terminal");
        terminal
            .draw(|frame| draw(frame, app))
            .expect("draw must not panic");
        let buf = terminal.backend().buffer().clone();
        let rows: Vec<String> = (0..buf.area.height)
            .map(|y| {
                let row: String = (0..buf.area.width).map(|x| buf[(x, y)].symbol()).collect();
                row.trim_end().to_owned()
            })
            .collect();
        rows.join("\n") + "\n"
    }

    /// Compare `text` to `tests/snapshots/<name>.txt`. `UPDATE_SNAPSHOTS=1`
    /// rewrites the file instead.
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

    #[test]
    fn golden_tuning_s2_chooser_80x24() {
        let mut fixture = FixtureBuilder::new("gfx1151").a8(A8Prefill::OptIn).build();
        press(&mut fixture.app, KeyCode::Char('p'));
        assert_snapshot("tuning_s2_80x24", &render_text(&mut fixture.app, 80, 24));
    }

    #[test]
    fn golden_tuning_s3_preview_80x24() {
        let mut fixture = FixtureBuilder::new("gfx1151")
            .a8(A8Prefill::OptIn)
            .flash_next_split_base()
            .build();
        press(&mut fixture.app, KeyCode::Char('p'));
        press(&mut fixture.app, KeyCode::Enter);
        assert_snapshot("tuning_s3_80x24", &render_text(&mut fixture.app, 80, 24));
    }

    #[test]
    fn golden_tuning_l2_160x48() {
        let mut fixture = FixtureBuilder::new("gfx1201").build();
        // The first frame tells the app the terminal is wide.
        render_text(&mut fixture.app, 160, 48);
        press(&mut fixture.app, KeyCode::Char('p'));
        assert_snapshot("tuning_l2_160x48", &render_text(&mut fixture.app, 160, 48));
    }
}
