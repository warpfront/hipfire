// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! The Settings tab renderer (spec §3, mockups S1/S4/L1): the grouped list
//! with Using now / From columns and the four-line explainer at 80 columns,
//! and at 160 columns and up the group rail, the Applies column and the "why
//! this value" pane.

use hipfire_config::{
    effective::Applies,
    evidence::{evidence_for, format_closer, format_slower},
    intent::Intent,
};
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
    Frame,
};

use crate::{
    app::{App, EditState, PendingEnum},
    hipfire::{
        knobs::{self, Tradeoff, PREDICTION_NOTE},
        settings_model::{value_string, RowId, SettingsRow, VirtualRow, MODEL_SCOPE_LOCK},
        writer::{self, FieldKind, SYM_IU4_KEY},
    },
    ui::{block, pad, ACCENT, MUTED, PANEL, PANEL_2, RED, TEXT, YELLOW},
};

/// Terminal width from which the rail, Applies column and why pane show.
pub(crate) const WIDE_COLS: u16 = 160;
/// ... and the height they need.
pub(crate) const WIDE_ROWS: u16 = 24;
/// Rail and pane widths at the wide layout.
const RAIL_W: u16 = 24;
const PANE_W: u16 = 42;

/// Group labels on the tab row.
fn tab_label(group: Intent) -> &'static str {
    match group {
        Intent::Quality => "Quality",
        Intent::Speed => "Speed",
        Intent::MemoryContext => "Memory",
        Intent::Speculation => "Speculation",
        Intent::HardwareDevices => "Hardware",
        Intent::Advanced => "Advanced",
    }
}

/// Group labels in the wide rail.
fn rail_label(group: Intent) -> &'static str {
    match group {
        Intent::HardwareDevices => "Hardware/devices",
        other => other.title(),
    }
}

/// The paragraph under the wide list that says what a group means.
fn group_blurb(group: Intent) -> &'static str {
    match group {
        Intent::Quality => {
            "Quality settings change the answers themselves. \"Better\" here means closer to the full-precision model, and it usually costs speed. Speed and speculation settings are designed not to change answers."
        }
        Intent::Speed => {
            "Speed settings change how fast answers arrive, not what they say. Most follow the model's own policy; change one only to compare."
        }
        Intent::MemoryContext => {
            "Memory settings decide how much fits: longer chats need more conversation memory (KV cache), and bigger models need layers kept on the GPU or spilled to system memory."
        }
        Intent::Speculation => {
            "Speculation drafts tokens ahead and keeps only the ones the model agrees with, so answers are designed not to change; it only changes typing speed."
        }
        Intent::HardwareDevices => {
            "Which GPUs serve uses and how they work together. A one-GPU machine can leave all of these alone."
        }
        Intent::Advanced => {
            "Every other setting, by canonical key. Press / to find any setting by name, key, environment variable or description."
        }
    }
}

/// `Changes the answers; better is usually slower.` → the box-title form.
fn title_copy(group: Intent) -> String {
    let header = group.header().trim_end_matches('.');
    let mut chars = header.chars();
    match chars.next() {
        Some(first) => first.to_lowercase().chain(chars).collect(),
        None => String::new(),
    }
}

/// `text` in a `width`-column cell that always ends in a space, so adjacent
/// columns never touch: padded when it fits, else cut with an ellipsis.
pub(crate) fn cell(text: &str, width: usize) -> String {
    if width == 0 {
        return String::new();
    }
    let count = text.chars().count();
    if count < width {
        return format!("{text}{}", " ".repeat(width - count));
    }
    let mut cut: String = text.chars().take(width.saturating_sub(2)).collect();
    cut.push('…');
    cut.push(' ');
    cut
}

/// Greedy word wrap at `cols` columns.
pub(crate) fn wrap(text: &str, cols: usize) -> Vec<String> {
    let cols = cols.max(8);
    let mut lines = Vec::new();
    let mut line = String::new();
    for word in text.split_whitespace() {
        let extra = usize::from(!line.is_empty()) + word.chars().count();
        if !line.is_empty() && line.chars().count() + extra > cols {
            lines.push(std::mem::take(&mut line));
        }
        if !line.is_empty() {
            line.push(' ');
        }
        line.push_str(word);
    }
    if !line.is_empty() {
        lines.push(line);
    }
    lines
}

pub fn draw_settings(frame: &mut Frame, app: &App, area: Rect) {
    if let Some(view) = &app.tuning_view {
        crate::ui_tuning::draw_tuning(frame, app, view, area);
        return;
    }
    if area.width >= WIDE_COLS && area.height >= WIDE_ROWS {
        draw_wide(frame, app, area);
    } else {
        draw_compact(frame, app, area);
    }
}

// ---------------------------------------------------------------------------
// Row cells
// ---------------------------------------------------------------------------

/// What one row shows in its Using now and From cells.
struct Cells {
    using: String,
    /// A staged, uncommitted choice.
    preview: bool,
    marked: bool,
    from: String,
    /// An environment variable owns the row.
    locked: bool,
    /// The active model or GPU never reads the row.
    dim: bool,
}

fn pending_for<'a>(app: &'a App, row: &SettingsRow) -> Option<&'a PendingEnum> {
    let pending = app.settings_pending.as_ref()?;
    let belongs = match &row.id {
        RowId::Virtual(VirtualRow::PromptProcessing) => pending.key == knobs::PROMPT_PROCESSING_ID,
        RowId::Key(key) => pending.key == *key,
    };
    belongs.then_some(pending)
}

/// The label of a staged value (`accurate` → `Accurate (8-bit math)`).
fn pending_label(row: &SettingsRow, pending: &PendingEnum) -> String {
    match &row.id {
        RowId::Virtual(_) => row
            .choices
            .iter()
            .find(|choice| choice.id == pending.value)
            .map_or_else(|| pending.value.clone(), |choice| choice.label.to_owned()),
        RowId::Key(key) => option_label(key, &pending.value),
    }
}

/// The plain label of enum option `value` of `key`.
fn option_label(key: &str, value: &str) -> String {
    knobs::curated_knob(key)
        .and_then(|knob| knob.options.iter().find(|(option, _)| *option == value))
        .map_or_else(|| value.to_owned(), |(_, label)| (*label).to_owned())
}

fn from_label(app: &App, row: &SettingsRow) -> String {
    if app.scope_lock_reason(row).is_some() {
        return MODEL_SCOPE_LOCK.to_owned();
    }
    let flash_next_only = matches!(row.id.key(), Some(SYM_IU4_KEY | "memory.ple_format"))
        && row.effective.used_by_model == Some(false);
    if flash_next_only {
        "Flash-Next only".to_owned()
    } else if !row.effective.used_on_gpu {
        "other GPUs only".to_owned()
    } else {
        row.origin.label()
    }
}

fn cells(app: &App, row: &SettingsRow) -> Cells {
    let marked = row.origin.marked();
    // A value on a row nobody reads is not in effect, so it carries no ● mark.
    let marker = if marked && !row.is_unused() {
        "● "
    } else {
        ""
    };
    let (using, from, preview) = match pending_for(app, row) {
        Some(pending) => (
            format!("→ {}", pending_label(row, pending)),
            from_label(app, row),
            true,
        ),
        None => match app.live_value(row) {
            Some(live) => (
                format!("{marker}{live}"),
                "live from serve".to_owned(),
                false,
            ),
            None => (
                format!("{marker}{}", row.display),
                from_label(app, row),
                false,
            ),
        },
    };
    Cells {
        using,
        preview,
        marked,
        from,
        locked: row.locked.is_some() || app.scope_lock_reason(row).is_some(),
        dim: row.is_unused(),
    }
}

/// Column widths of the list.
struct Columns {
    setting: usize,
    using: usize,
    from: usize,
    tail: usize,
    /// `Applies` or `Group`.
    tail_title: &'static str,
    wide: bool,
}

impl Columns {
    /// `inner` is the list's inner width.
    fn new(inner: usize, wide: bool, searching: bool) -> Self {
        let body = inner.saturating_sub(2);
        let (setting, using, from) = match (wide, searching) {
            (false, false) => (30, 24, body.saturating_sub(54).max(10)),
            (false, true) => (26, 18, 19),
            (true, false) => (30, 24, 20),
            (true, true) => (26, 22, 20),
        };
        let used = setting + using + from;
        let tail = body.saturating_sub(used);
        let (from, tail) = if wide || searching {
            (from, tail)
        } else {
            (from + tail, 0)
        };
        Self {
            setting,
            using,
            from,
            tail,
            tail_title: if searching { "Group" } else { "Applies" },
            wide,
        }
    }

    /// In a model's scope the From column must hold `all models only (locked)`.
    fn for_scope(mut self, app: &App) -> Self {
        if app.scope_model().is_none() || self.tail_title == "Group" {
            return self;
        }
        let body = self.setting + self.using + self.from + self.tail;
        (self.setting, self.using, self.from) = if self.wide {
            (26, 24, 25)
        } else {
            (25, 24, body.saturating_sub(49).max(25))
        };
        self.tail = if self.wide {
            body.saturating_sub(75)
        } else {
            0
        };
        self
    }

    fn header(&self) -> Line<'static> {
        let mut text = format!(
            "  {}{}{}",
            cell("Setting", self.setting),
            cell("Using now", self.using),
            cell("From", self.from)
        );
        if self.tail > 0 {
            text.push_str(&cell(self.tail_title, self.tail));
        }
        Line::from(Span::styled(text, Style::default().fg(MUTED)))
    }
}

fn applies_short(app: &App, row: &SettingsRow) -> String {
    if app.row_needs_restart(row) {
        return "↻ restart".to_owned();
    }
    match row.effective.applies {
        Applies::NextMessage => "next message",
        Applies::NewChat => "new chat",
        Applies::NextModelLoad => "next load",
        Applies::ServeRestart => "restart",
    }
    .to_owned()
}

fn row_line(
    app: &App,
    row: &SettingsRow,
    selected: bool,
    columns: &Columns,
    searching: bool,
) -> Line<'static> {
    let cells = cells(app, row);
    let base = if selected {
        Style::default().fg(ACCENT).bg(PANEL_2)
    } else if cells.dim {
        Style::default().fg(MUTED)
    } else {
        Style::default().fg(TEXT)
    };
    let using_style = if cells.preview {
        let style = Style::default().fg(YELLOW).add_modifier(Modifier::ITALIC);
        if selected {
            style.bg(PANEL_2)
        } else {
            style
        }
    } else if cells.marked && !selected {
        base.add_modifier(Modifier::BOLD)
    } else {
        base
    };
    let from_style = if selected {
        base
    } else if cells.locked {
        Style::default().fg(YELLOW)
    } else {
        Style::default().fg(MUTED)
    };
    let tail = if columns.tail == 0 {
        String::new()
    } else if searching {
        cell(
            if columns.wide {
                row.group.title()
            } else {
                tab_label(row.group)
            },
            columns.tail,
        )
    } else {
        cell(&applies_short(app, row), columns.tail)
    };
    let mut spans = vec![
        Span::styled(if selected { "> " } else { "  " }, base),
        Span::styled(cell(&row.title, columns.setting), base),
        Span::styled(cell(&cells.using, columns.using), using_style),
        Span::styled(cell(&cells.from, columns.from), from_style),
    ];
    if columns.tail > 0 {
        spans.push(Span::styled(tail, from_style));
    }
    Line::from(spans)
}

/// Rows the list shows and where its window starts, so the cursor is visible.
fn window(selected: usize, capacity: usize) -> usize {
    if selected < capacity {
        0
    } else {
        selected + 1 - capacity
    }
}

fn scroll_note(start: usize, capacity: usize, total: usize) -> Option<Line<'static>> {
    if total <= capacity {
        return None;
    }
    let end = (start + capacity).min(total);
    let up = if start > 0 { "↑" } else { "" };
    let down = if end < total { "↓" } else { "" };
    let text = format!(" rows {}–{} of {} {}{} ", start + 1, end, total, up, down);
    Some(
        Line::from(Span::styled(
            text.trim_end().to_owned() + " ",
            Style::default().fg(MUTED),
        ))
        .right_aligned(),
    )
}

/// The list box content: header row, the visible rows, or the empty note.
fn list_lines(
    app: &App,
    rows: &[&SettingsRow],
    columns: &Columns,
    start: usize,
    capacity: usize,
    searching: bool,
) -> Vec<Line<'static>> {
    let mut lines = vec![columns.header()];
    if rows.is_empty() {
        lines.push(Line::from(Span::styled(
            "  No settings match. Esc clears the search; c / h toggle the filters.",
            Style::default().fg(MUTED),
        )));
        return lines;
    }
    for (index, row) in rows.iter().enumerate().skip(start).take(capacity) {
        lines.push(row_line(
            app,
            row,
            index == app.settings_selected,
            columns,
            searching,
        ));
    }
    lines
}

// ---------------------------------------------------------------------------
// Header, tabs, search line
// ---------------------------------------------------------------------------

/// The GPU text of the header: the probe's name when there is one, the arch
/// from `install.json`, and a note when the live probe disagrees.
pub(crate) fn gpu_text(app: &App, wide: bool) -> String {
    use crate::hipfire::dashboard::Probe;
    let system = app.dashboard.as_ref().and_then(|dash| dash.system.as_ref());
    let name = system.and_then(|info| match &info.gpu_name {
        Probe::Value(name) => Some(name.as_str()),
        Probe::Unavailable(_) => None,
    });
    let probe_arch = system.and_then(|info| match &info.gpu_arch {
        Probe::Value(arch) => Some(arch.as_str()),
        Probe::Unavailable(_) => None,
    });
    let install = app.arch.gpu_arch.as_deref();
    match (install, probe_arch) {
        (Some(arch), Some(probe)) if arch != probe => {
            format!("{arch} (install.json) · live probe says {probe}")
        }
        (Some(arch), probe) => {
            let shown = match name {
                Some(name) if wide => format!("{name} · {arch}"),
                Some(name) => format!("{name} ({arch})"),
                None => arch.to_owned(),
            };
            match (wide, probe) {
                (true, Some(_)) => format!("{shown} (install.json; live probe agrees)"),
                (true, None) => format!("{shown} (install.json)"),
                (false, _) => shown,
            }
        }
        (None, Some(probe)) => match name {
            Some(name) => format!("{name} ({probe}) (live probe)"),
            None => format!("{probe} (live probe)"),
        },
        (None, None) => "not detected (run hipfire setup)".to_owned(),
    }
}

/// The header while a value is being edited: the buffer with the char under
/// the cursor drawn reversed (a reversed space at the end of the buffer).
fn edit_line(edit: &EditState) -> Line<'static> {
    let chars: Vec<char> = edit.buffer.chars().collect();
    let cursor = edit.cursor.min(chars.len());
    let before: String = chars[..cursor].iter().collect();
    let under = chars.get(cursor).copied().unwrap_or(' ');
    let after: String = chars.iter().skip(cursor + 1).collect();
    Line::from(vec![
        Span::raw(format!("Editing {} = {before}", edit.key)),
        Span::styled(
            under.to_string(),
            Style::default().add_modifier(Modifier::REVERSED),
        ),
        Span::raw(format!("{after}  (Enter save · Esc cancel)")),
    ])
}

fn header_line(app: &App, wide: bool) -> Line<'static> {
    if let Some(edit) = &app.settings_edit {
        return edit_line(edit);
    }
    let model_note = app.serve_loaded_config().map_or("default", |_| "loaded");
    let mut text = if wide {
        format!(
            "GPU: {}    Model: {} ({model_note})    {}",
            gpu_text(app, true),
            app.active_model,
            crate::ui_scope::editing_text(app, true)
        )
    } else {
        format!(
            "GPU: {}    {}",
            gpu_text(app, false),
            crate::ui_scope::editing_text(app, false)
        )
    };
    if let Some(profile) = app.settings.profile.as_deref().filter(|_| wide) {
        text.push_str(&format!("    Profile: {profile}"));
    }
    // The Tuning preset in effect leads the header (spec §4 S1 / L1).
    let tuning_key = if wide { " (p)" } else { "  p" };
    text.insert_str(
        0,
        &format!("Tuning: {}{tuning_key}    ", app.settings.tuning.label()),
    );
    let warning = app.settings.warnings.first();
    let mut spans = vec![Span::styled(text, Style::default().fg(TEXT))];
    if let Some(warning) = warning {
        spans.push(Span::styled(
            format!("    ! {warning}"),
            Style::default().fg(YELLOW),
        ));
    }
    Line::from(spans)
}

fn tabs_line(app: &App) -> Line<'static> {
    let searching = app.settings_search.is_some();
    let mut spans = Vec::new();
    for group in Intent::ALL {
        let label = tab_label(group);
        if group == app.settings_group && !searching {
            spans.push(Span::styled(
                format!("[{label}]  "),
                Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
            ));
        } else {
            spans.push(Span::styled(
                format!("{label}  "),
                Style::default().fg(MUTED),
            ));
        }
    }
    Line::from(spans)
}

/// The row under the header: the group tabs, or the search line.
fn draw_tabs_or_search(frame: &mut Frame, app: &App, area: Rect) {
    match &app.settings_search {
        None => {
            frame.render_widget(Paragraph::new(tabs_line(app)), area);
            frame.render_widget(
                Paragraph::new(Line::from(Span::styled(
                    "/ search",
                    Style::default().fg(MUTED),
                )))
                .right_aligned(),
                area,
            );
        }
        Some(search) => {
            let cursor = if search.typing { "█" } else { "" };
            frame.render_widget(
                Paragraph::new(Line::from(vec![
                    Span::styled("Search: ", Style::default().fg(MUTED)),
                    Span::styled(
                        format!("{}{cursor}", search.query),
                        Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
                    ),
                ])),
                area,
            );
            let matches = app.settings_visible().len();
            let noun = if matches == 1 { "match" } else { "matches" };
            frame.render_widget(
                Paragraph::new(Line::from(Span::styled(
                    format!("{matches} {noun} · Esc clear"),
                    Style::default().fg(MUTED),
                )))
                .right_aligned(),
                area,
            );
        }
    }
}

fn confirm_box(frame: &mut Frame, area: Rect, one_line: bool, model: Option<&str>) {
    let lines = if let Some(model) = model {
        let ask = format!("Remove ALL of {model}'s own values?");
        let keys = "They follow \"all models\" again.  y = confirm · n / Esc = cancel";
        if one_line {
            vec![Line::from(format!("{ask}  {keys}"))]
        } else {
            vec![Line::from(ask), Line::from(keys)]
        }
    } else if one_line {
        vec![Line::from(
            "Reset ALL settings to defaults? This clears ~/.hipfire/config.toml.  y = confirm · n / Esc = cancel",
        )]
    } else {
        vec![
            Line::from("Reset ALL settings to defaults? This clears ~/.hipfire/config.toml."),
            Line::from("y = confirm · n / Esc = cancel"),
        ]
    };
    frame.render_widget(
        Paragraph::new(lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(RED))
                    .title(if model.is_some() {
                        " Confirm remove all "
                    } else {
                        " Confirm reset-all "
                    }),
            )
            .style(Style::default().fg(TEXT).bg(PANEL)),
        area,
    );
}

// ---------------------------------------------------------------------------
// 80-column layout (mockups S1, S4)
// ---------------------------------------------------------------------------

fn draw_compact(frame: &mut Frame, app: &App, area: Rect) {
    let area = pad(area, 1, 0);
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Min(3),
            Constraint::Length(6),
        ])
        .split(area);

    frame.render_widget(
        Paragraph::new(header_line(app, false)).style(Style::default().fg(TEXT)),
        chunks[0],
    );
    draw_tabs_or_search(frame, app, chunks[1]);

    let rows = app.settings_visible();
    let searching = app.settings_search.is_some();
    let title = if searching {
        "Search results — name, key, env var or description".to_owned()
    } else if let Some(model) = app.scope_model() {
        format!("{} · values for {model} only", app.settings_group.title())
    } else {
        format!(
            "{} — {}",
            app.settings_group.title(),
            title_copy(app.settings_group)
        )
    };
    let inner_w = chunks[2].width.saturating_sub(2) as usize;
    let capacity = (chunks[2].height.saturating_sub(2) as usize)
        .saturating_sub(1)
        .max(1);
    let start = window(app.settings_selected, capacity);
    let columns = Columns::new(inner_w, false, searching).for_scope(app);
    let mut list_block = block(&title);
    if let Some(note) = scroll_note(start, capacity, rows.len()) {
        list_block = list_block.title_bottom(note);
    }
    frame.render_widget(
        Paragraph::new(list_lines(app, &rows, &columns, start, capacity, searching))
            .block(list_block)
            .style(Style::default().fg(TEXT).bg(PANEL)),
        chunks[2],
    );

    if app.confirm_reset_all {
        confirm_box(frame, chunks[3], false, app.scope_model());
        return;
    }
    let (title, lines) = match app.selected_row() {
        Some(row) => {
            let lines = app.row_explainer(row);
            (row.title.clone(), lines.to_vec())
        }
        None => ("Nothing selected".to_owned(), Vec::new()),
    };
    frame.render_widget(
        Paragraph::new(lines.into_iter().map(Line::from).collect::<Vec<_>>())
            .block(block(&title))
            .style(Style::default().fg(TEXT).bg(PANEL)),
        chunks[3],
    );
}

// ---------------------------------------------------------------------------
// 160-column layout (mockup L1)
// ---------------------------------------------------------------------------

fn draw_wide(frame: &mut Frame, app: &App, area: Rect) {
    let area = pad(area, 1, 0);
    let rows_area = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(10)])
        .split(area);
    if app.confirm_reset_all {
        confirm_box(frame, rows_area[0], true, app.scope_model());
    } else {
        frame.render_widget(
            Paragraph::new(header_line(app, true))
                .block(block("Settings"))
                .style(Style::default().fg(TEXT).bg(PANEL)),
            rows_area[0],
        );
    }
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Length(RAIL_W),
            Constraint::Min(40),
            Constraint::Length(PANE_W),
        ])
        .split(rows_area[1]);
    draw_rail(frame, app, cols[0]);
    draw_wide_list(frame, app, cols[1]);
    draw_why_pane(frame, app, cols[2]);
}

fn draw_rail(frame: &mut Frame, app: &App, area: Rect) {
    let inner = area.width.saturating_sub(2) as usize;
    let searching = app.settings_search.is_some();
    let mut lines = vec![Line::from(Span::styled(
        "Groups",
        Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
    ))];
    for (group, count) in app.settings.group_counts(&app.settings_filter()) {
        let current = group == app.settings_group && !searching;
        let label = cell(rail_label(group), inner.saturating_sub(5));
        let text = format!("{}{label}{count:>3}", if current { "> " } else { "  " });
        let style = if current {
            Style::default().fg(ACCENT).bg(PANEL_2)
        } else {
            Style::default().fg(TEXT)
        };
        lines.push(Line::from(Span::styled(text, style)));
    }
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "Filters",
        Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
    )));
    let onoff = |on: bool| if on { "on" } else { "off" };
    for (label, on) in [
        ("Changed only (c)", app.settings_changed_only),
        ("Hide unused (h)", app.settings_hide_unused),
    ] {
        let value = onoff(on);
        let gap = inner.saturating_sub(2 + label.chars().count() + value.len());
        lines.push(Line::from(Span::styled(
            format!("  {label}{}{value}", " ".repeat(gap)),
            Style::default().fg(if on { YELLOW } else { TEXT }),
        )));
    }
    frame.render_widget(
        Paragraph::new(lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(PANEL_2)),
            )
            .style(Style::default().fg(TEXT).bg(PANEL)),
        area,
    );
}

fn draw_wide_list(frame: &mut Frame, app: &App, area: Rect) {
    let rows = app.settings_visible();
    let searching = app.settings_search.is_some();
    let title = if searching {
        "Search results".to_owned()
    } else if let Some(model) = app.scope_model() {
        format!("{} · values for {model} only", app.settings_group.title())
    } else {
        app.settings_group.title().to_owned()
    };
    let inner_w = area.width.saturating_sub(2) as usize;
    let inner_h = area.height.saturating_sub(2) as usize;
    let columns = Columns::new(inner_w, true, searching).for_scope(app);

    // Under the rows: the "also affects quality" link and what the group means.
    let blurb = if searching {
        Vec::new()
    } else {
        wrap(group_blurb(app.settings_group), inner_w.saturating_sub(2))
    };
    let link = app.settings_group == Intent::Quality
        && !searching
        && app
            .settings
            .row(&RowId::Key("memory.kv_cache".into()))
            .is_some();
    let reserved = if blurb.is_empty() { 0 } else { blurb.len() + 1 } + if link { 2 } else { 0 };
    let capacity = inner_h.saturating_sub(1 + reserved).max(1);
    let start = window(app.settings_selected, capacity);

    let mut lines = list_lines(app, &rows, &columns, start, capacity, searching);
    if link {
        if let Some(kv) = app.settings.row(&RowId::Key("memory.kv_cache".into())) {
            lines.push(Line::from(Span::styled(
                "  — also affects quality —",
                Style::default().fg(MUTED),
            )));
            let shown = cells(app, kv);
            lines.push(Line::from(Span::styled(
                format!(
                    "  {}{}{}{}",
                    cell(&kv.title, columns.setting),
                    cell(&shown.using, columns.using),
                    cell(&shown.from, columns.from),
                    cell("see Memory", columns.tail)
                ),
                Style::default().fg(MUTED),
            )));
        }
    }
    if !blurb.is_empty() {
        // Pin the blurb to the bottom of the box.
        let used = lines.len();
        let room = inner_h.saturating_sub(used + blurb.len());
        lines.extend((0..room).map(|_| Line::from("")));
        lines.extend(
            blurb.into_iter().map(|text| {
                Line::from(Span::styled(format!(" {text}"), Style::default().fg(MUTED)))
            }),
        );
    }

    let mut list_block = block(&title);
    if let Some(note) = scroll_note(start, capacity, rows.len()) {
        list_block = list_block.title_bottom(note);
    }
    frame.render_widget(
        Paragraph::new(lines)
            .block(list_block)
            .style(Style::default().fg(TEXT).bg(PANEL)),
        area,
    );
}

// ---------------------------------------------------------------------------
// The "why this value" pane
// ---------------------------------------------------------------------------

pub(crate) fn heading(text: &str) -> Line<'static> {
    Line::from(Span::styled(
        text.to_owned(),
        Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
    ))
}

pub(crate) fn plain(text: impl Into<String>) -> Line<'static> {
    Line::from(Span::styled(text.into(), Style::default().fg(TEXT)))
}

pub(crate) fn muted(text: impl Into<String>) -> Line<'static> {
    Line::from(Span::styled(text.into(), Style::default().fg(MUTED)))
}

/// `label  value`, the value wrapped under itself when the pane is narrow.
fn detail(label: &str, value: &str, width: usize) -> Vec<Line<'static>> {
    const LABEL_W: usize = 14;
    let mut lines = Vec::new();
    for (index, text) in wrap(value, width.saturating_sub(LABEL_W))
        .into_iter()
        .enumerate()
    {
        let head = if index == 0 {
            format!("{label:<LABEL_W$}")
        } else {
            " ".repeat(LABEL_W)
        };
        lines.push(Line::from(vec![
            Span::styled(head, Style::default().fg(MUTED)),
            Span::styled(text, Style::default().fg(TEXT)),
        ]));
    }
    lines
}

/// One option line: `● label  detail` when it fits, else two lines.
fn option_lines(active: bool, label: &str, note: &str, width: usize) -> Vec<Line<'static>> {
    let dot = if active { "●" } else { "○" };
    let style = if active {
        Style::default().fg(ACCENT)
    } else {
        Style::default().fg(TEXT)
    };
    let one = format!("  {dot} {label}");
    if note.is_empty() {
        return vec![Line::from(Span::styled(one, style))];
    }
    if one.chars().count() + 2 + note.chars().count() < width {
        return vec![Line::from(vec![
            Span::styled(one, style),
            Span::styled(format!("  {note}"), Style::default().fg(MUTED)),
        ])];
    }
    vec![
        Line::from(Span::styled(one, style)),
        muted(format!("      {note}")),
    ]
}

/// The evidence row the knob quotes, for this GPU and A8 state.
fn evidence_row(
    app: &App,
    row: &SettingsRow,
) -> Option<(&'static hipfire_config::evidence::EvidenceRow, &'static str)> {
    let knob = knobs::curated_knob(row.id.knob_id())?;
    let Tradeoff::Evidence { line, side, .. } = knob.tradeoff else {
        return None;
    };
    let gpu = app.arch.gpu_arch.as_deref().unwrap_or("");
    evidence_for(line, gpu, app.arch.a8_prefill).map(|found| (found, side))
}

fn options_section(app: &App, row: &SettingsRow, width: usize) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    match &row.id {
        RowId::Virtual(_) if !row.choices.is_empty() => {
            lines.push(heading("Options on this GPU"));
            let evidence = evidence_row(app, row).map(|(found, _)| found);
            for choice in &row.choices {
                let active =
                    row.resolved == hipfire_config::ConfigValue::String(choice.id.to_owned());
                let note = if choice.builtin_default {
                    "built-in default".to_owned()
                } else {
                    evidence
                        .map(|found| {
                            [format_closer(found), format_slower(found)]
                                .into_iter()
                                .flatten()
                                .collect::<Vec<_>>()
                                .join(" · ")
                        })
                        .unwrap_or_default()
                };
                lines.extend(option_lines(active, choice.label, &note, width));
            }
        }
        RowId::Key(key) => {
            let options: Vec<(String, String)> = match writer::field_spec(key).map(|s| s.kind) {
                Some(FieldKind::Bool) => ["true", "false"]
                    .iter()
                    .map(|value| ((*value).to_owned(), bool_option_label(key, value)))
                    .collect(),
                Some(FieldKind::Enum(values)) => values
                    .iter()
                    .map(|value| ((*value).to_owned(), option_label(key, value)))
                    .collect(),
                _ => Vec::new(),
            };
            if options.is_empty() {
                return lines;
            }
            lines.push(heading("Options on this GPU"));
            let current = value_string(&row.resolved);
            const MAX: usize = 12;
            let info = legacy_info(row);
            for (value, label) in options.iter().take(MAX) {
                let note = info
                    .and_then(|info| info.options.iter().find(|(option, _)| option == value))
                    .map_or("", |(_, desc)| *desc);
                lines.extend(option_lines(*value == current, label, note, width));
            }
            if options.len() > MAX {
                lines.push(muted(format!("    … {} more (←/→)", options.len() - MAX)));
            }
        }
        RowId::Virtual(_) => {}
    }
    lines
}

/// The legacy explainer entry behind a key row (`knobs::KNOBS`), if any.
fn legacy_info(row: &SettingsRow) -> Option<&'static knobs::KnobInfo> {
    let schema = hipfire_config::field(row.id.key()?)?;
    knobs::knob_info(schema.legacy_key)
}

fn bool_option_label(key: &str, value: &str) -> String {
    knobs::curated_knob(key)
        .and_then(|knob| knob.options.iter().find(|(option, _)| *option == value))
        .map_or_else(
            || if value == "true" { "on" } else { "off" }.to_owned(),
            |(_, label)| (*label).to_owned(),
        )
}

fn chain_section(row: &SettingsRow, shown: &str, width: usize) -> Vec<Line<'static>> {
    let mut lines = vec![heading(&format!("Why \"{shown}\""))];
    if row.chain.is_empty() {
        lines.push(muted("  no layer sets this value"));
        return lines;
    }
    for (index, entry) in row.chain.iter().enumerate() {
        let used = if entry.used { "  ← used" } else { "" };
        let head = format!("  {}. {}", index + 1, entry.label);
        let style = if entry.used {
            Style::default().fg(TEXT)
        } else {
            Style::default().fg(MUTED)
        };
        let one = format!("{head}  {}{used}", entry.value);
        if one.chars().count() <= width {
            lines.push(Line::from(Span::styled(one, style)));
        } else {
            lines.push(Line::from(Span::styled(head, style)));
            lines.push(Line::from(Span::styled(
                format!("     {}{used}", entry.value),
                style,
            )));
        }
    }
    if row.chain.len() == 1 && !row.origin.marked() {
        for text in wrap(
            "no registry, profile, preset, user, model or environment value is set",
            width.saturating_sub(6),
        ) {
            lines.push(muted(format!("     {text}")));
        }
    }
    if let Some(reason) = &row.effective.reason {
        for text in wrap(reason, width.saturating_sub(6)) {
            lines.push(muted(format!("     {text}")));
        }
    }
    lines
}

fn measured_section(app: &App, row: &SettingsRow, width: usize) -> Vec<Line<'static>> {
    let Some(knob) = knobs::curated_knob(row.id.knob_id()) else {
        return Vec::new();
    };
    let Tradeoff::Evidence { .. } = knob.tradeoff else {
        return Vec::new();
    };
    let gpu = app.arch.gpu_arch.as_deref().unwrap_or("this GPU");
    let Some((found, side)) = evidence_row(app, row) else {
        return vec![
            heading(&format!("Measured · {gpu}")),
            muted("  not measured on this GPU"),
        ];
    };
    let from = match side {
        "Accurate" => "Fast",
        "q8" => "q4",
        _ => "before",
    };
    let mut lines = vec![heading(&format!("Measured · {gpu}"))];
    let kld = |metric: &str| metric.trim_end_matches(" KLD").to_owned();
    if let (Some(before), Some(after)) = (found.kld_from, found.kld_to) {
        lines.push(plain("  KLD vs full model (lower = closer)"));
        let label = found.kld_metric.map_or_else(String::new, kld);
        lines.push(plain(format!(
            "    {label}  {from} {before:.4}   {side} {after:.4}"
        )));
        if let Some(extra) = found.kld_extra {
            lines.push(plain(format!(
                "    {}  {from} {:.4}   {side} {:.4}",
                kld(extra.metric),
                extra.from,
                extra.to
            )));
        }
    }
    if let (Some(before), Some(after)) = (found.speed_from, found.speed_to) {
        let metric = found.speed_metric.unwrap_or("tok/s");
        lines.push(plain(format!("  prompt speed {metric}")));
        lines.push(plain(format!("    {from} {before:.1}   {side} {after:.1}")));
        if let Some(extra) = found.speed_extra {
            lines.push(plain(format!("  {}", extra.metric)));
            lines.push(plain(format!(
                "    {from} {:.1}   {side} {:.1}",
                extra.from, extra.to
            )));
        }
    }
    if let Some(typing) = found.typing {
        lines.push(plain(format!("  typing speed: {typing}")));
    }
    if let Some(caveat) = found.caveat {
        for text in wrap(&format!("caveat: {caveat}"), width.saturating_sub(2)) {
            lines.push(muted(format!("  {text}")));
        }
    }
    if let Some(source) = found.source {
        for text in wrap(&format!("source: {source}"), width.saturating_sub(2)) {
            lines.push(muted(format!("  {text}")));
        }
    }
    lines
}

fn takes_effect(app: &App, row: &SettingsRow) -> String {
    if app.row_needs_restart(row) {
        return "↻ restart to apply".to_owned();
    }
    match row.effective.applies {
        Applies::NextMessage => "from the next message",
        Applies::NewChat => "in a new chat",
        Applies::NextModelLoad => "at the next model load",
        Applies::ServeRestart => "after serve restart",
    }
    .to_owned()
}

/// Every line of the "why this value" pane for `row`, `width` columns wide.
fn why_lines(app: &App, row: &SettingsRow, width: usize) -> Vec<Line<'static>> {
    let live = app.live_value(row);
    let explainer = app
        .settings
        .explainer(row, live.as_deref(), app.row_needs_restart(row));
    let curated = knobs::curated_knob(row.id.knob_id()).is_some();
    let what = if curated {
        explainer[0].clone()
    } else {
        format!("{} {}", explainer[0], explainer[1])
            .trim()
            .to_owned()
    };

    let mut lines: Vec<Line<'static>> = wrap(&what, width).into_iter().map(plain).collect();
    if let Some(text) = app.scope_text(row) {
        lines.push(Line::from(""));
        for note in text.values.iter().chain(std::iter::once(&text.hint)) {
            lines.extend(wrap(note, width).into_iter().map(plain));
        }
    }
    let options = options_section(app, row, width);
    if !options.is_empty() {
        lines.push(Line::from(""));
        lines.extend(options);
    }
    if let Some(info) = legacy_info(row).filter(|_| !curated) {
        lines.push(Line::from(""));
        for text in wrap(&format!("When: {}", info.when), width) {
            lines.push(muted(text));
        }
        if let Some(note) = info.note {
            for text in wrap(&format!("Note: {note}"), width) {
                lines.push(muted(text));
            }
        }
    }
    lines.push(Line::from(""));
    let shown = live.clone().unwrap_or_else(|| row.display.clone());
    lines.extend(chain_section(row, &shown, width));
    let measured = measured_section(app, row, width);
    if !measured.is_empty() {
        lines.push(Line::from(""));
        lines.extend(measured);
    }
    lines.push(Line::from(""));
    lines.extend(detail("Takes effect", &takes_effect(app, row), width));
    let writes = if row.keys.is_empty() {
        "—".to_owned()
    } else {
        row.keys.join(", ")
    };
    lines.extend(detail("Writes", &writes, width));
    let env = row
        .keys
        .first()
        .and_then(|key| knobs::env_name(key))
        .unwrap_or_else(|| "—".to_owned());
    lines.extend(detail("Environment", &env, width));
    lines.extend(detail(
        "Per model",
        if row.model_scope_lock().is_some() {
            "no — one value per serve process"
        } else {
            "yes — models.toml can hold its own"
        },
        width,
    ));
    if let Some(info) = legacy_info(row).filter(|_| !curated) {
        let default = if info.default.is_empty() {
            "(unset)"
        } else {
            info.default
        };
        lines.extend(detail("Default", default, width));
    }
    lines.push(Line::from(""));
    let note = if live.is_some() {
        "Live from serve: the values the resident model was loaded with."
    } else {
        PREDICTION_NOTE
    };
    lines.extend(wrap(note, width).into_iter().map(muted));
    lines
}

fn draw_why_pane(frame: &mut Frame, app: &App, area: Rect) {
    let width = area.width.saturating_sub(2) as usize;
    let (title, lines) = match app.selected_row() {
        Some(row) => (row.title.clone(), why_lines(app, row, width)),
        None => ("Nothing selected".to_owned(), Vec::new()),
    };
    frame.render_widget(
        Paragraph::new(lines)
            // Every pane line wraps; none is cut at the border.
            .wrap(Wrap { trim: false })
            .block(block(&title))
            .style(Style::default().fg(TEXT).bg(PANEL)),
        area,
    );
}

#[cfg(test)]
mod tests {
    use hipfire_config::effective::A8Prefill;
    use ratatui::{backend::TestBackend, Terminal};

    use super::*;
    use crate::{
        test_support::{online_dashboard, press, select_key, type_text, Fixture, FixtureBuilder},
        ui::draw,
    };
    use crossterm::event::KeyCode;

    /// Render the whole UI and return one trimmed string per terminal row.
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

    /// The S1/L1 fixture: Strix Halo, PR #902 opt-in, `reasoning.effort = low`.
    fn strix_halo() -> Fixture {
        FixtureBuilder::new("gfx1151")
            .a8(A8Prefill::OptIn)
            .config("[reasoning]\neffort = \"low\"\n")
            .build()
    }

    #[test]
    fn golden_s1_quality_80x24() {
        let mut fixture = strix_halo();
        assert_snapshot("settings_s1_80x24", &render_text(&mut fixture.app, 80, 24));
    }

    #[test]
    fn golden_l1_quality_160x48() {
        let mut fixture = strix_halo();
        assert_snapshot(
            "settings_l1_160x48",
            &render_text(&mut fixture.app, 160, 48),
        );
    }

    #[test]
    fn golden_s4_search_kv_80x24() {
        let mut fixture = FixtureBuilder::new("gfx1201").build();
        press(&mut fixture.app, KeyCode::Char('/'));
        type_text(&mut fixture.app, "kv");
        assert_snapshot("settings_s4_80x24", &render_text(&mut fixture.app, 80, 24));
    }

    fn assert_rows_fit(rows: &[String], width: u16, context: &str) {
        for row in rows {
            assert!(
                Line::from(row.as_str()).width() <= width as usize,
                "{context}: row wider than {width}: {row:?}"
            );
            // A box row keeps its right border in the last-but-one column; an
            // overflowing cell would push it out.
            if row.starts_with(" │") || row.starts_with(" ┌") || row.starts_with(" └") {
                assert_eq!(
                    Line::from(row.as_str()).width(),
                    width as usize - 1,
                    "{context}: box row lost its border: {row:?}"
                );
            }
        }
    }

    #[test]
    fn no_rendered_line_exceeds_the_width_in_any_group() {
        for (width, height) in [(80, 24), (160, 48)] {
            for (gpu, a8) in [
                ("gfx1151", A8Prefill::OptIn),
                ("gfx1201", A8Prefill::NotLanded),
                ("gfx1030", A8Prefill::NotLanded),
            ] {
                let mut fixture = FixtureBuilder::new(gpu)
                    .a8(a8)
                    .config("[reasoning]\neffort = \"low\"\n[memory]\nkv_cache = \"q8\"\n")
                    .build();
                for digit in '1'..='6' {
                    press(&mut fixture.app, KeyCode::Char(digit));
                    for at in [0usize, 4, usize::MAX] {
                        press(&mut fixture.app, KeyCode::Home);
                        for _ in 0..at.min(4) {
                            press(&mut fixture.app, KeyCode::Down);
                        }
                        if at == usize::MAX {
                            press(&mut fixture.app, KeyCode::End);
                        }
                        let rows = render_rows(&mut fixture.app, width, height);
                        assert_eq!(rows.len(), height as usize);
                        assert_rows_fit(
                            &rows,
                            width,
                            &format!("{gpu} {width}x{height} group {digit}"),
                        );
                    }
                }
                // Search results and the editor fit too.
                press(&mut fixture.app, KeyCode::Char('/'));
                type_text(&mut fixture.app, "kv");
                let rows = render_rows(&mut fixture.app, width, height);
                assert_rows_fit(&rows, width, &format!("{gpu} {width}x{height} search"));
            }
        }
    }

    #[test]
    fn no_curated_using_now_is_truncated_at_80_columns() {
        // A marked gfx1030 row is the longest cell ("● — not used on this GPU").
        for (gpu, a8, config) in [
            (
                "gfx1151",
                A8Prefill::OptIn,
                "[reasoning]\neffort = \"low\"\n",
            ),
            (
                "gfx1201",
                A8Prefill::NotLanded,
                "[memory]\nkv_cache = \"q8\"\ngpu_layer_budget = -1\noffload_exec = \"passback\"\n",
            ),
            (
                "gfx1030",
                A8Prefill::NotLanded,
                "[kernel]\niu4_prefill = false\n",
            ),
            ("gfx1100", A8Prefill::NotLanded, ""),
        ] {
            let mut fixture = FixtureBuilder::new(gpu).a8(a8).config(config).build();
            let columns = Columns::new(76, false, false);
            for digit in '1'..='5' {
                press(&mut fixture.app, KeyCode::Char(digit));
                let text = render_text(&mut fixture.app, 80, 80);
                let rows: Vec<SettingsRow> = fixture
                    .app
                    .settings_visible()
                    .into_iter()
                    .filter(|row| match &row.id {
                        RowId::Virtual(_) => true,
                        RowId::Key(key) => knobs::curated_knob(key).is_some(),
                    })
                    .cloned()
                    .collect();
                assert!(!rows.is_empty());
                for row in rows {
                    let using = cells(&fixture.app, &row).using;
                    assert!(
                        using.chars().count() < columns.using,
                        "{gpu} {}: {using:?} is wider than the {}-column cell",
                        row.title,
                        columns.using
                    );
                    let line = text
                        .lines()
                        .find(|line| line.starts_with(" │") && line.contains(&row.title))
                        .unwrap_or_else(|| panic!("{gpu}: no rendered row for {}", row.title));
                    assert!(
                        line.contains(&using),
                        "{gpu} {}: {line:?} cut {using:?}",
                        row.title
                    );
                    assert!(!line.contains('…'), "{gpu} {}: {line:?}", row.title);
                }
            }
        }
    }

    #[test]
    fn env_row_reads_the_variable_and_registry_row_reads_registry() {
        let mut fixture = FixtureBuilder::new("gfx1201")
            .env("memory.kv_cache", "q8")
            .build();
        select_key(&mut fixture.app, "memory.kv_cache");
        let text = render_text(&mut fixture.app, 80, 24);
        assert!(text.contains("env HIPFIRE_KV_MODE"), "{text}");
        assert!(text.contains("● q8"), "{text}");
        assert!(
            text.contains("Set by HIPFIRE_KV_MODE; unset it to change"),
            "{text}"
        );

        let mut fixture = FixtureBuilder::new("gfx1201").card_kv_mode("fwht3").build();
        select_key(&mut fixture.app, "memory.kv_cache");
        let text = render_text(&mut fixture.app, 80, 24);
        let line = text
            .lines()
            .find(|l| l.contains("KV cache precision"))
            .unwrap();
        assert!(
            line.contains("fwht3") && line.contains("registry"),
            "{line}"
        );
        assert!(
            !line.contains('●'),
            "a model-card value is not a ● override: {line}"
        );
    }

    #[test]
    fn search_line_group_column_and_cursor() {
        let mut fixture = FixtureBuilder::new("gfx1201").build();
        press(&mut fixture.app, KeyCode::Char('/'));
        type_text(&mut fixture.app, "kv");
        let text = render_text(&mut fixture.app, 80, 24);
        assert!(text.contains("Search: kv█"), "{text}");
        assert!(text.contains("matches · Esc clear"), "{text}");
        assert!(text.contains("Group"), "{text}");
        assert!(text.contains("Search results"), "{text}");
        press(&mut fixture.app, KeyCode::Enter);
        let text = render_text(&mut fixture.app, 80, 24);
        assert!(
            text.contains("Search: kv ") && !text.contains('█'),
            "{text}"
        );
        assert!(text.contains("Esc clear search"), "footer: {text}");
        // No matches says so.
        press(&mut fixture.app, KeyCode::Char('/'));
        type_text(&mut fixture.app, "zzzzzz");
        let text = render_text(&mut fixture.app, 80, 24);
        assert!(
            text.contains("0 matches") && text.contains("No settings match"),
            "{text}"
        );
    }

    #[test]
    fn preview_is_shown_with_an_arrow_and_commit_footer() {
        let mut fixture = FixtureBuilder::new("gfx1201").build();
        select_key(&mut fixture.app, "attention.flash");
        press(&mut fixture.app, KeyCode::Right);
        let text = render_text(&mut fixture.app, 80, 24);
        assert!(text.contains("→ always"), "{text}");
        assert!(text.contains("Enter commit"), "{text}");
    }

    #[test]
    fn e_and_a_are_gone_from_footer_and_help() {
        let mut fixture = FixtureBuilder::new("gfx1201").build();
        let rows = render_rows(&mut fixture.app, 80, 24);
        let footer = rows[22..].join(" ");
        assert!(
            footer.contains("Del default") && footer.contains("/ search"),
            "{footer}"
        );
        for gone in ["e easy", "a advanced", "easy", "advanced"] {
            assert!(
                !footer.contains(gone),
                "footer still says {gone:?}: {footer}"
            );
        }
        fixture.app.show_help = true;
        let help = render_text(&mut fixture.app, 80, 24);
        assert!(
            help.contains("Keybindings") && help.contains("Del default"),
            "{help}"
        );
        for gone in ["e easy", "a advanced", "easy"] {
            assert!(!help.contains(gone), "help still says {gone:?}: {help}");
        }
    }

    #[test]
    fn header_names_the_probed_gpu_and_flags_a_disagreeing_probe() {
        use crate::hipfire::dashboard::{Probe, SystemInfo};
        let system = |name: &str, arch: &str| SystemInfo {
            gpu_name: Probe::Value(name.into()),
            gpu_arch: Probe::Value(arch.into()),
            hip_version: Probe::Unavailable("n/a".into()),
            kernel_cache: Probe::Unavailable("n/a".into()),
            loaded_model: Probe::Unavailable("n/a".into()),
            model_checksum: Probe::Unavailable("n/a".into()),
        };
        let mut fixture = FixtureBuilder::new("gfx1201").build();
        let mut dash = online_dashboard(None, 10);
        dash.system = Some(system("Radeon AI PRO R9700", "gfx1201"));
        fixture.app.dashboard = Some(dash);
        let rows = render_rows(&mut fixture.app, 80, 24);
        assert!(
            rows.iter().any(
                |row| row.contains("Tuning: Balanced  p    GPU: Radeon AI PRO R9700 (gfx1201)")
            ),
            "{rows:#?}"
        );

        let mut dash = online_dashboard(None, 10);
        dash.system = Some(system("Radeon RX 7900 XTX", "gfx1100"));
        fixture.app.dashboard = Some(dash);
        let rows = render_rows(&mut fixture.app, 80, 24);
        assert!(
            rows.iter()
                .any(|row| row.contains("GPU: gfx1201 (install.json) · live probe says gfx1100")),
            "{rows:#?}"
        );

        // No install.json arch and no probe: say so rather than guess.
        let mut fixture = FixtureBuilder::new("gfx1201").build();
        fixture.app.arch.gpu_arch = None;
        fixture.app.rebuild_settings();
        let rows = render_rows(&mut fixture.app, 80, 24);
        assert!(
            rows.iter().any(|row| row.contains("GPU: not detected")),
            "{rows:#?}"
        );
    }

    #[test]
    fn live_serve_values_and_the_restart_badge_render() {
        use crate::hipfire::dashboard::LoadedConfig;
        let mut fixture = FixtureBuilder::new("gfx1201").build();
        select_key(&mut fixture.app, "memory.kv_cache");
        fixture.app.dashboard = Some(online_dashboard(
            Some(LoadedConfig {
                model: crate::test_support::ACTIVE_MODEL.into(),
                kv_mode: Some("q8".into()),
                kv_backend: Some("vmm".into()),
                max_seq: Some(65536),
            }),
            3600,
        ));
        let text = render_text(&mut fixture.app, 80, 24);
        let line = text
            .lines()
            .find(|l| l.contains("KV cache precision"))
            .unwrap();
        assert!(
            line.contains("q8") && line.contains("live from serve"),
            "{line}"
        );
        assert!(
            text.contains("Live from serve: q8; config predicts fp8 (auto)."),
            "{text}"
        );
        // The context row shows the resident model's context, not the config's.
        press(&mut fixture.app, KeyCode::Up);
        let text = render_text(&mut fixture.app, 80, 24);
        let line = text.lines().find(|l| l.contains("Context length")).unwrap();
        assert!(
            line.contains("65,536 tokens") && line.contains("live from serve"),
            "{line}"
        );

        // At 160 columns the pane says so instead of the prediction note.
        let text = render_text(&mut fixture.app, 160, 48);
        assert!(text.contains("Live from serve: the values the"), "{text}");
        assert!(!text.contains("Predicted from your config"), "{text}");
        // The config was written after serve started: process rows say restart.
        select_key(&mut fixture.app, "kernel.iu4_prefill");
        let text = render_text(&mut fixture.app, 160, 48);
        assert!(text.contains("↻ restart"), "{text}");
        assert!(text.contains("↻ restart to apply"), "{text}");
        let text = render_text(&mut fixture.app, 80, 24);
        assert!(text.contains("↻ restart to apply"), "{text}");
    }

    #[test]
    fn wide_pane_has_options_chain_measurements_and_the_honesty_line() {
        let mut fixture = strix_halo();
        let row = fixture.app.selected_row().unwrap().clone();
        let lines: Vec<String> = why_lines(&fixture.app, &row, 44)
            .iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|span| span.content.as_ref())
                    .collect()
            })
            .collect();
        let joined = lines.join("\n");
        for want in [
            "Options on this GPU",
            "● Fast (4-bit math)",
            "○ Accurate (8-bit math)",
            "43% closer · 1.6× slower",
            "Why \"Fast (4-bit math)\"",
            "1. built-in default · gfx1151",
            "← used",
            "Measured · gfx1151",
            "KLD vs full model (lower = closer)",
            "Fast 0.0695",
            "Accurate 0.0394",
            "Fast 974.2",
            "Accurate 603.8",
            "typing speed: unchanged",
            "source: PR #902",
            "Takes effect",
            "after serve restart",
            "Writes",
            "kernel.gfx11_a8_prefill",
            "Environment",
            "Per model",
            "no — one value per serve",
        ] {
            assert!(joined.contains(want), "missing {want:?} in:\n{joined}");
        }
        let note_at = lines
            .iter()
            .position(|line| line.starts_with("Predicted"))
            .expect("the honesty line is in the pane");
        assert_eq!(lines[note_at..].join(" "), PREDICTION_NOTE);

        // A changed row's chain ends with the layer that set it.
        select_key(&mut fixture.app, "reasoning.effort");
        let row = fixture.app.selected_row().unwrap().clone();
        let lines: Vec<String> = why_lines(&fixture.app, &row, 44)
            .iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|span| span.content.as_ref())
                    .collect()
            })
            .collect();
        let joined = lines.join("\n");
        assert!(joined.contains("1. built-in default"), "{joined}");
        assert!(joined.contains("2. config.toml"), "{joined}");
        let used: Vec<&String> = lines
            .iter()
            .filter(|line| line.contains("← used"))
            .collect();
        assert_eq!(used.len(), 1, "{joined}");
        assert!(joined.find("2. config.toml").unwrap() < joined.find("← used").unwrap());
    }

    #[test]
    fn wide_layout_shows_rail_counts_filters_and_the_applies_column() {
        let mut fixture = strix_halo();
        let text = render_text(&mut fixture.app, 160, 48);
        for want in [
            "Groups",
            "> Quality",
            "Filters",
            "Changed only (c)",
            "Hide unused (h)",
            "Applies",
            "next message",
            "Editing: all models (m)",
            "Tuning: Balanced (p)",
        ] {
            assert!(text.contains(want), "missing {want:?}:\n{text}");
        }
        for gone in ["(u)", "Profiles", "Per-model (o)"] {
            assert!(
                !text.contains(gone),
                "deferred item {gone:?} advertised:\n{text}"
            );
        }
        press(&mut fixture.app, KeyCode::Char('c'));
        let text = render_text(&mut fixture.app, 160, 48);
        assert!(text.contains("Changed only (c)  on"), "{text}");
    }

    #[test]
    fn reset_all_confirm_renders_in_both_layouts() {
        for (width, height) in [(80, 24), (160, 48)] {
            let mut fixture = FixtureBuilder::new("gfx1201").build();
            press(&mut fixture.app, KeyCode::Char('R'));
            let text = render_text(&mut fixture.app, width, height);
            assert!(
                text.contains("Confirm reset-all") && text.contains("y = confirm"),
                "{text}"
            );
        }
    }

    #[test]
    fn edit_draws_a_visible_cursor_at_its_position() {
        let mut fixture = FixtureBuilder::new("gfx1201").build();
        select_key(&mut fixture.app, "memory.max_seq");
        press(&mut fixture.app, KeyCode::Enter);
        press(&mut fixture.app, KeyCode::Home);
        press(&mut fixture.app, KeyCode::Right);
        press(&mut fixture.app, KeyCode::Right); // cursor on the '7' of 32|768
        let (width, height) = (120, 32);
        let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
        terminal
            .draw(|frame| draw(frame, &mut fixture.app))
            .unwrap();
        let buf = terminal.backend().buffer().clone();
        let reversed: Vec<String> = (0..buf.area.height)
            .flat_map(|y| (0..buf.area.width).map(move |x| (x, y)))
            .filter(|&(x, y)| buf[(x, y)].modifier.contains(Modifier::REVERSED))
            .map(|(x, y)| buf[(x, y)].symbol().to_owned())
            .collect();
        assert_eq!(
            reversed,
            ["7"],
            "exactly the char under the cursor is reversed"
        );
    }

    /// Words the pane draws for the selected row, and words it was asked to
    /// draw, rendering the pane tall enough that nothing is clipped below.
    fn pane_words(app: &App) -> (Vec<String>, Vec<String>) {
        let area = Rect::new(0, 0, PANE_W, 200);
        let mut terminal = Terminal::new(TestBackend::new(area.width, area.height)).unwrap();
        terminal
            .draw(|frame| draw_why_pane(frame, app, area))
            .unwrap();
        let buf = terminal.backend().buffer().clone();
        let mut drawn = Vec::new();
        for y in 1..buf.area.height - 1 {
            let row: String = (1..buf.area.width - 1)
                .map(|x| buf[(x, y)].symbol())
                .collect();
            drawn.extend(row.split_whitespace().map(str::to_owned));
        }
        let row = app.selected_row().unwrap();
        let asked = why_lines(app, row, PANE_W as usize - 2)
            .iter()
            .flat_map(|line| {
                line.spans
                    .iter()
                    .map(|span| span.content.as_ref())
                    .collect::<String>()
                    .split_whitespace()
                    .map(str::to_owned)
                    .collect::<Vec<_>>()
            })
            .collect();
        (drawn, asked)
    }

    #[test]
    fn why_pane_wraps_every_line_and_cuts_none_at_160x48() {
        for (gpu, a8) in [
            ("gfx1151", A8Prefill::OptIn),
            ("gfx1201", A8Prefill::NotLanded),
            ("gfx1100", A8Prefill::NotLanded),
        ] {
            let mut fixture = FixtureBuilder::new(gpu).a8(a8).build();
            for digit in '1'..='6' {
                press(&mut fixture.app, KeyCode::Char(digit));
                press(&mut fixture.app, KeyCode::Home);
                for _ in 0..8 {
                    let (drawn, asked) = pane_words(&fixture.app);
                    assert_eq!(
                        drawn, asked,
                        "{gpu} group {digit} row {}",
                        fixture.app.settings_selected
                    );
                    press(&mut fixture.app, KeyCode::Down);
                }
            }
        }
        // The reviewed case: the typing-speed line wraps whole at 160x48.
        let mut fixture = strix_halo();
        let text = render_text(&mut fixture.app, 160, 48);
        assert!(!text.contains("batch ≥ │"), "{text}");
        assert!(
            text.contains("64)"),
            "the wrapped tail is on screen: {text}"
        );
    }

    #[test]
    fn adjacent_columns_never_touch() {
        assert_eq!(cell("ab", 5), "ab   ");
        assert_eq!(cell("abcd", 5), "abcd ");
        assert_eq!(cell("abcde", 5), "abc… ");
        assert_eq!(cell("abcdefgh", 3), "a… ");
        for (width, height, search) in [
            (80, 24, true),
            (80, 24, false),
            (120, 32, true),
            (160, 48, true),
            (160, 48, false),
        ] {
            let mut fixture = FixtureBuilder::new("gfx1201").build();
            if search {
                press(&mut fixture.app, KeyCode::Char('/'));
                type_text(&mut fixture.app, "kv");
            }
            for digit in if search { "1" } else { "123456" }.chars() {
                if !search {
                    press(&mut fixture.app, KeyCode::Char(digit));
                }
                let text = render_text(&mut fixture.app, width, height);
                let chars: Vec<char> = text.chars().collect();
                for pair in chars.windows(2) {
                    assert!(
                        !(pair[0] == '…' && (pair[1].is_alphanumeric() || pair[1] == '│')),
                        "a cut cell touches the next column at {width}x{height}:\n{text}"
                    );
                }
            }
        }
    }

    #[test]
    fn group_paragraph_is_drawn_once_at_the_bottom_of_the_wide_list() {
        for digit in '1'..='6' {
            let mut fixture = strix_halo();
            press(&mut fixture.app, KeyCode::Char(digit));
            for end in [false, true] {
                if end {
                    press(&mut fixture.app, KeyCode::End);
                }
                let rows = render_rows(&mut fixture.app, 160, 48);
                let blurb = wrap(group_blurb(fixture.app.settings_group), 80);
                let first_word = group_blurb(fixture.app.settings_group)
                    .split_whitespace()
                    .next()
                    .unwrap();
                // The paragraph's first line appears once, and its last line
                // sits directly above the box's bottom border.
                let first_hits = rows
                    .iter()
                    .filter(|r| r.contains(&format!("││ {first_word} ")))
                    .count();
                assert_eq!(first_hits, 1, "group {digit}: {rows:#?}");
                let last_word = blurb
                    .last()
                    .unwrap()
                    .split_whitespace()
                    .last()
                    .unwrap()
                    .to_owned();
                let hits: Vec<usize> = rows
                    .iter()
                    .enumerate()
                    .filter(|(_, r)| r.contains(&last_word) && r.contains("││"))
                    .map(|(i, _)| i)
                    .collect();
                let border = rows.iter().rposition(|r| r.starts_with(" └")).unwrap();
                assert!(hits.contains(&(border - 1)), "group {digit}: {rows:#?}");
            }
        }
    }

    #[test]
    fn search_ranks_title_and_key_hits_before_env_and_help_text_hits() {
        let mut fixture = FixtureBuilder::new("gfx1201").build();
        press(&mut fixture.app, KeyCode::Char('/'));
        type_text(&mut fixture.app, "kv");
        let titles: Vec<String> = fixture
            .app
            .settings_visible()
            .iter()
            .map(|row| row.title.clone())
            .collect();
        let at = |title: &str| {
            titles
                .iter()
                .position(|t| t == title)
                .unwrap_or_else(|| panic!("{title} missing in {titles:?}"))
        };
        let context = at("Context length");
        for title in [
            "KV cache precision",
            "KV fit-to-memory",
            "KV cache layout",
            "KV key format",
            "KV value format",
        ] {
            assert!(
                at(title) < context,
                "{title} ranks after Context length: {titles:?}"
            );
        }
        // The count is unchanged by ranking.
        let unranked = fixture
            .app
            .settings
            .rows
            .iter()
            .filter(|row| fixture.app.settings_filter().matches(row))
            .count();
        assert_eq!(titles.len(), unranked);

        // Tiers hold: a key hit never follows a help-text-only hit.
        let tiers: Vec<u8> = fixture
            .app
            .settings_visible()
            .iter()
            .map(|row| {
                let keys: Vec<&str> = row.keys.iter().map(String::as_str).collect();
                crate::hipfire::search::match_tier(&row.title, &keys, "kv")
            })
            .collect();
        assert!(tiers.windows(2).all(|pair| pair[0] <= pair[1]), "{tiers:?}");
        assert_eq!(tiers[0], 0);
        assert_eq!(*tiers.last().unwrap(), 2, "help-text-only hits come last");
    }

    #[test]
    fn kv_layout_and_axis_rows_have_plain_titles() {
        let fixture = FixtureBuilder::new("gfx1201").build();
        for (key, title) in [
            ("memory.kv_backend", "KV cache layout"),
            ("memory.kv_k", "KV key format"),
            ("memory.kv_v", "KV value format"),
        ] {
            let row = fixture
                .app
                .settings
                .row(&RowId::Key(key.into()))
                .unwrap_or_else(|| panic!("no row {key}"));
            assert_eq!(row.title, title);
        }
    }
}
