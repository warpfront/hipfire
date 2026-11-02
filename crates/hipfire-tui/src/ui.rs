// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{
        Block, Borders, Clear, Gauge, List, ListItem, Paragraph, Row, Sparkline, Table, Tabs, Wrap,
    },
    Frame,
};

use crate::{
    app::{App, Tab},
    hipfire::dashboard::{LoadState, VramState},
    hipfire::metrics::MetricsView,
    hipfire::pull_choice::PullChoice,
    hipfire::registry::ModelListItem,
    hipfire::serve_ctrl::ServeAction,
};

pub(crate) const BG: Color = Color::Rgb(7, 7, 9);
pub(crate) const PANEL: Color = Color::Rgb(18, 16, 18);
pub(crate) const PANEL_2: Color = Color::Rgb(40, 24, 27);
pub(crate) const TEXT: Color = Color::Rgb(222, 226, 232);
pub(crate) const MUTED: Color = Color::Rgb(142, 150, 163);
pub(crate) const ACCENT: Color = Color::Rgb(237, 45, 57);
pub(crate) const GREEN: Color = Color::Rgb(102, 217, 139);
pub(crate) const YELLOW: Color = Color::Rgb(238, 190, 95);
pub(crate) const RED: Color = Color::Rgb(255, 95, 104);

/// Tab-bar divider, shared between the renderer (`Tabs::divider`) and the mouse
/// hit-test math in `draw_header` so the two cannot drift (the hit-test advances
/// `x` by exactly this width between tab cells).
const TAB_DIVIDER: &str = " | ";

pub fn draw(frame: &mut Frame, app: &mut App) {
    let area = frame.area();
    app.tuning_wide = crate::ui_tuning::is_wide(area);
    frame.render_widget(Clear, area);
    let root = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(5),
            Constraint::Min(10),
            Constraint::Length(2),
        ])
        .split(area);

    draw_header(frame, app, root[0]);
    if crate::ui_scope::draws_screen(app, root[1]) {
        crate::ui_scope::draw_screen(frame, app, root[1]);
    } else {
        match app.tab {
            Tab::Home => draw_home(frame, app, root[1]),
            Tab::Dashboard => draw_dashboard(frame, app, root[1]),
            Tab::Chat => draw_chat(frame, app, root[1]),
            Tab::Models => draw_models(frame, app, root[1]),
            Tab::Settings => crate::ui_settings::draw_settings(frame, app, root[1]),
            Tab::System => draw_system(frame, app, root[1]),
            Tab::Logs => draw_logs(frame, app, root[1]),
        }
        crate::ui_scope::draw_picker_popup(frame, app, root[1]);
    }
    draw_footer(frame, app, root[2]);

    // The `?` help overlay draws last, centered on top of everything.
    if app.show_help {
        draw_help_overlay(frame, app, area);
    }
}

/// Centered modal listing the global + active-tab keybindings. Dismissed by any
/// key (handled in main::handle_key). Reuses `footer_segments` for the per-tab
/// lines so there is a single source of truth for tab keys.
fn draw_help_overlay(frame: &mut Frame, app: &App, area: Rect) {
    let mut lines: Vec<Line> = vec![
        Line::from(Span::styled(
            "Global",
            Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
        )),
        Line::from("  Tab / Shift+Tab  switch tabs"),
        Line::from("  ?                toggle this help (in Chat: Esc first)"),
        Line::from("  Esc              close / cancel (never quits)"),
        Line::from("  r                refresh live data"),
        Line::from("  q                quit"),
        Line::from("  mouse            click a tab · scroll to move in a list"),
        Line::from(""),
        Line::from(Span::styled(
            format!("This tab — {}", app.tab.title()),
            Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
        )),
    ];
    // The active-tab keys, one per line — keeps each short so nothing wraps and
    // clips inside the modal, and makes the line count (hence height) exact. The
    // globally-listed keys (Tab/Shift+Tab, ?, r, q) are skipped to avoid
    // duplication.
    for part in footer_segments(app) {
        let part = part.trim();
        if part.is_empty()
            || part.starts_with("Tab/Shift+Tab")
            || part == "q quit"
            || part == "r refresh"
            || part.contains("? help")
        {
            continue;
        }
        lines.push(Line::from(format!("  {part}")));
    }
    if app.tab == Tab::Dashboard {
        lines.push(Line::from("  x / R ask y / n first; R force-reaps serve"));
    }
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "press any key to close",
        Style::default().fg(MUTED),
    )));

    let w = 60u16.min(area.width.saturating_sub(2));
    let h = (lines.len() as u16 + 2).min(area.height.saturating_sub(2));
    let x = area.x + area.width.saturating_sub(w) / 2;
    let y = area.y + area.height.saturating_sub(h) / 2;
    let rect = Rect {
        x,
        y,
        width: w,
        height: h,
    };

    frame.render_widget(Clear, rect);
    frame.render_widget(
        Paragraph::new(lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(ACCENT))
                    .title(" Keybindings "),
            )
            .wrap(Wrap { trim: false })
            .style(Style::default().fg(TEXT).bg(PANEL)),
        rect,
    );
}

fn draw_header(frame: &mut Frame, app: &mut App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(2), Constraint::Length(3)])
        .split(area);

    let title = Line::from(vec![
        Span::styled(
            "hipfire",
            Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
        ),
        Span::styled(" · control panel", Style::default().fg(ACCENT)),
        Span::styled(
            format!(
                "    serve: {}    model: {}",
                app.status.serve_label(),
                app.active_model
            ),
            Style::default().fg(MUTED),
        ),
    ]);
    frame.render_widget(
        Paragraph::new(title)
            .style(Style::default().bg(BG))
            .alignment(Alignment::Center),
        chunks[0],
    );

    let titles = Tab::ALL
        .iter()
        .map(|tab| Line::from(Span::raw(tab.title())))
        .collect::<Vec<_>>();
    let selected = Tab::ALL.iter().position(|t| *t == app.tab).unwrap_or(0);
    let tabs = Tabs::new(titles)
        .select(selected)
        .block(
            Block::default()
                .borders(Borders::BOTTOM)
                .border_style(Style::default().fg(PANEL_2)),
        )
        .style(Style::default().fg(MUTED).bg(BG))
        .highlight_style(Style::default().fg(ACCENT).add_modifier(Modifier::BOLD))
        .divider(Span::styled(TAB_DIVIDER, Style::default().fg(PANEL_2)));
    frame.render_widget(tabs, chunks[1]);

    // Record per-tab hit regions for mouse clicks. ratatui 0.30's Tabs layout:
    // from the inner left, each tab cell is 1 (left pad) + title + 1 (right pad),
    // with a 3-col " | " divider between cells. The block has only a BOTTOM
    // border, so inner.x == area.x and titles sit on the area's top row.
    app.tab_row_y = chunks[1].y;
    app.tab_hitboxes.clear();
    let right = chunks[1].x.saturating_add(chunks[1].width);
    let mut x = chunks[1].x;
    for tab in Tab::ALL {
        if x >= right {
            break;
        }
        let cell_w = 2 + tab.title().len() as u16; // left pad + title + right pad
        let end = x.saturating_add(cell_w).min(right);
        app.tab_hitboxes.push((x, end, tab));
        x = end.saturating_add(TAB_DIVIDER.len() as u16); // divider between cells
    }
}

/// Separator between footer segments.
const FOOTER_SEP: &str = " · ";

/// The quit-while-pulling y/n prompt (shown while `confirm_quit` is armed).
fn quit_confirm_prompt(app: &App) -> String {
    let tag = app.pull.as_ref().map_or("", |job| job.tag.as_str());
    format!("A pull of {tag} is running. Quit and cancel it?  y / n")
}

/// What the footer shows for the current tab/state: an ordered list of whole
/// segments, highest priority first. [`fit_segments`] drops the ones that do not
/// fit the footer width (never cutting one mid-word), so list order IS priority
/// order: put the keys that matter most (and `? help`) near the front. Other
/// slices extend a tab by inserting one segment into its arm.
///
/// Segments must not contain the ` · ` separator themselves.
fn footer_segments(app: &App) -> Vec<String> {
    fn segs(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| (*s).to_string()).collect()
    }
    if app.confirm_quit {
        return vec![quit_confirm_prompt(app)];
    }
    if let Some(segments) = crate::ui_scope::footer_segments(app) {
        return segments;
    }
    match app.tab {
        Tab::Home => segs(&["? help", "r refresh", "Tab/Shift+Tab switch", "q quit"]),
        Tab::Dashboard => {
            if let Some(action) = app.confirm_serve {
                vec![serve_confirm_prompt(action).to_string()]
            } else if app.serve_cmd_running() {
                vec![
                    format!("serve {}\u{2026}", app.serve_cmd_label),
                    "? help".to_string(),
                    "Tab/Shift+Tab switch".to_string(),
                    "q quit".to_string(),
                ]
            } else {
                segs(&[
                    "s start",
                    "x stop",
                    "R restart",
                    "? help",
                    "r refresh",
                    "Tab/Shift+Tab switch",
                    "q quit",
                ])
            }
        }
        Tab::Chat => {
            // Chat owns q/r/? as text input while the input is focused, so help
            // is reached via Esc (blur) first.
            if app.chat.sending {
                segs(&["Esc stop reply", "Up/Down/PgUp/PgDn scroll", "End follow"])
            } else if app.chat.is_input_focused() {
                segs(&[
                    "Enter send",
                    "Esc then ? help",
                    "Ctrl+O newline",
                    "Up/Down/PgUp/PgDn scroll",
                    "End follow",
                    "/help commands",
                    "Esc stop / blur",
                ])
            } else {
                segs(&[
                    "Enter send",
                    "? help",
                    "Up/Down/PgUp/PgDn scroll",
                    "End follow",
                    "Tab/Shift+Tab switch",
                    "q quit",
                ])
            }
        }
        Tab::Models => {
            if app.pull_choice.is_some() {
                segs(&[
                    "Up/Down row",
                    "Left/Right change",
                    "Enter pull",
                    "Esc cancel",
                ])
            } else if app.confirm_delete.is_some() {
                segs(&["Delete this model?", "y confirm", "n / Esc cancel"])
            } else if app.pull.is_some() {
                segs(&[
                    "c cancel pull",
                    "Up/Down select",
                    "Enter open/expand",
                    "d delete",
                    "o overrides",
                    "? help",
                    "r refresh",
                    "Tab/Shift+Tab switch",
                    "q quit",
                ])
            } else {
                segs(&[
                    "Up/Down select",
                    "Enter open/expand",
                    "p pull",
                    "d delete",
                    "o overrides",
                    "? help",
                    "r refresh",
                    "Tab/Shift+Tab switch",
                    "q quit",
                ])
            }
        }
        Tab::Settings => {
            if let Some(view) = &app.tuning_view {
                crate::ui_tuning::footer_segments(app, view)
            } else if app.confirm_reset_all {
                match app.scope_model() {
                    Some(model) => vec![
                        format!("Remove all of {model}'s own values?"),
                        "y confirm".to_owned(),
                        "n / Esc cancel".to_owned(),
                    ],
                    None => segs(&[
                        "Reset ALL settings to defaults?",
                        "y confirm",
                        "n / Esc cancel",
                    ]),
                }
            } else if app.settings_pending.is_some() {
                segs(&[
                    "Preview: Left/Right/Space change",
                    "Enter commit",
                    "Esc cancel",
                ])
            } else if app.settings_edit.is_some() {
                segs(&[
                    "Editing: type value",
                    "Enter save",
                    "Backspace delete",
                    "Esc cancel",
                ])
            } else if app.settings_search_typing() {
                segs(&[
                    "type to filter",
                    "↑↓ move",
                    "Enter results",
                    "Esc clear search",
                ])
            } else if app.settings_search.is_some() {
                segs(&[
                    "↑↓ move",
                    "←→ change",
                    "Enter save",
                    "Esc clear search",
                    "/ edit search",
                    "? help",
                    "r refresh",
                    "Tab/Shift+Tab switch",
                    "q quit",
                ])
            } else if app.scope_model().is_some() {
                segs(&[
                    "↑↓ move",
                    "←→ change",
                    "Del follow all-models",
                    "m scope",
                    "o overrides",
                    "? help",
                    "Enter save",
                    "1-6 group",
                    "/ search",
                    "c changed",
                    "h hide unused",
                    "R remove all values",
                    "r refresh",
                    "Tab/Shift+Tab switch",
                    "q quit",
                ])
            } else {
                segs(&[
                    "↑↓ move",
                    "←→ change",
                    "Del default",
                    "/ search",
                    "? help",
                    "p tuning",
                    "m scope",
                    "o overrides",
                    "R reset-all",
                    "1-6 group",
                    "Enter save",
                    "c changed",
                    "h hide unused",
                    "r refresh",
                    "Tab/Shift+Tab switch",
                    "q quit",
                ])
            }
        }
        Tab::System => {
            if app.doctor_running() {
                segs(&[
                    "doctor running\u{2026}",
                    "? help",
                    "live diagnostics",
                    "r refresh",
                    "Tab/Shift+Tab switch",
                    "q quit",
                ])
            } else {
                segs(&[
                    "d doctor",
                    "? help",
                    "live diagnostics (auto ~1.5s)",
                    "r refresh",
                    "Tab/Shift+Tab switch",
                    "q quit",
                ])
            }
        }
        Tab::Logs => segs(&[
            "? help",
            "serve.log tail (auto ~1s)",
            "r refresh",
            "Tab/Shift+Tab switch",
            "q quit",
        ]),
    }
}

/// Prompt shown while a Dashboard stop/restart confirm is armed.
fn serve_confirm_prompt(action: ServeAction) -> &'static str {
    match action {
        ServeAction::Restart => {
            "Restart runs `hipfire stop --force` (force-reaps this install's serve daemons), then starts serve. y / n"
        }
        _ => "Stop hipfire serve? y / n",
    }
}

/// Display width of `s` in terminal columns (all footer text is single-width).
fn text_cols(s: &str) -> usize {
    s.chars().count()
}

/// Join the leading-priority segments that fit in `width` columns, keeping list
/// order and skipping (not cutting) any segment that does not fit.
fn fit_segments(segs: &[String], width: usize) -> String {
    let mut out = String::new();
    for seg in segs {
        let extra = if out.is_empty() {
            text_cols(seg)
        } else {
            text_cols(FOOTER_SEP) + text_cols(seg)
        };
        if text_cols(&out) + extra <= width {
            if !out.is_empty() {
                out.push_str(FOOTER_SEP);
            }
            out.push_str(seg);
        }
    }
    out
}

/// The footer's hint text for `width` columns plus the status text, which is
/// `Some` only when it fits entirely in the room left after the hints.
fn footer_text(app: &App, width: u16) -> (String, Option<String>) {
    let width = width as usize;
    let hints = fit_segments(&footer_segments(app), width);
    let gap = "    ";
    let status = (!app.last_reload.is_empty()
        && text_cols(&hints) + text_cols(gap) + text_cols(&app.last_reload) <= width)
        .then(|| format!("{gap}{}", app.last_reload));
    (hints, status)
}

fn draw_footer(frame: &mut Frame, app: &App, area: Rect) {
    // An armed quit confirm outranks everything (it is modal on every tab and
    // must stay visible even while a toast is live).
    if app.confirm_quit {
        frame.render_widget(
            Paragraph::new(quit_confirm_prompt(app))
                .wrap(Wrap { trim: true })
                .style(
                    Style::default()
                        .fg(YELLOW)
                        .bg(BG)
                        .add_modifier(Modifier::BOLD),
                ),
            area,
        );
        return;
    }
    // An armed serve stop/restart confirm owns the whole footer (wrapped over
    // both rows) and outranks any toast, so the prompt cannot be hidden.
    if let (Tab::Dashboard, Some(action)) = (app.tab, app.confirm_serve) {
        frame.render_widget(
            Paragraph::new(serve_confirm_prompt(action))
                .wrap(Wrap { trim: true })
                .style(Style::default().fg(YELLOW).bg(BG)),
            area,
        );
        return;
    }
    // A live toast takes over the footer line entirely (colored by severity);
    // otherwise show the per-tab hints + the last persistent status line.
    if let Some(toast) = &app.toast {
        let (fg, tag) = match toast.level {
            crate::app::ToastLevel::Error => (RED, "!"),
            crate::app::ToastLevel::Info => (GREEN, "·"),
        };
        frame.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled(
                    format!(" {tag} "),
                    Style::default().fg(BG).bg(fg).add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!(" {}", toast.text),
                    Style::default().fg(fg).add_modifier(Modifier::BOLD),
                ),
            ]))
            .wrap(Wrap { trim: false })
            .style(Style::default().bg(BG)),
            area,
        );
        return;
    }

    let (hints, status) = footer_text(app, area.width);
    let mut spans = vec![Span::styled(hints, Style::default().fg(MUTED))];
    if let Some(status) = status {
        spans.push(Span::styled(status, Style::default().fg(Color::DarkGray)));
    }
    frame.render_widget(
        Paragraph::new(Line::from(spans)).style(Style::default().bg(BG)),
        area,
    );
}

fn draw_home(frame: &mut Frame, app: &App, area: Rect) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(52), Constraint::Percentage(48)])
        .split(pad(area, 1, 0));
    let right = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(8), Constraint::Length(7)])
        .split(cols[1]);

    let serve_color = if app.status.serve_http_ok {
        GREEN
    } else if !app.status.health_known
        || app.status.serve_pid_alive
        || app.status.serve_pid.is_some()
    {
        YELLOW
    } else {
        RED
    };
    // Lines are laid out here to fit the card's inner width (long values keep
    // their tail, long warnings wrap under their label), so the card height is
    // exactly the line count and nothing is cut off at the bottom.
    let inner_w = (cols[0].width as usize).saturating_sub(2);
    let plain = Style::default();
    let mut status = wrap_labeled(
        "Serve      ",
        &app.status.serve_label(),
        inner_w,
        Style::default().fg(serve_color),
    );
    status.extend([
        fit_line(
            "Endpoint   ",
            &format!("{}:{}", app.config.probe_host(), app.config.port),
            inner_w,
            plain,
        ),
        fit_line(
            "PID        ",
            &app.status
                .serve_pid
                .map(|p| p.to_string())
                .unwrap_or_else(|| "-".into()),
            inner_w,
            plain,
        ),
        fit_line("Active     ", &app.active_model, inner_w, plain),
        fit_line(
            "Config     ",
            &format!(
                "{} · {} overlays",
                if app.config.loaded_from_disk {
                    "custom"
                } else {
                    "defaults"
                },
                app.config.per_model_count
            ),
            inner_w,
            plain,
        ),
        fit_line(
            "Models     ",
            &format!(
                "{} local / {} registry",
                app.registry.local_files.len(),
                app.registry.models.len()
            ),
            inner_w,
            plain,
        ),
    ]);
    // The registry line shows the same short label as the Models header; a
    // registry warning (e.g. bundled copy newer than the cache) only tints it,
    // the full text lives on the Models tab. Config warnings wrap but are
    // capped at two rows so they cannot push the Tabs list off a short terminal.
    let warn = Style::default().fg(YELLOW);
    if let Some(warning) = &app.config.warning {
        let mut rows = wrap_labeled("Config     ", warning, inner_w, warn);
        rows.truncate(2);
        status.extend(rows);
    }
    status.push(fit_line(
        "Registry   ",
        &registry_label(app),
        inner_w,
        if app.registry.warning.is_some() {
            warn
        } else {
            plain
        },
    ));
    // Size the Runtime card from its line count (+2 borders); the Tabs list
    // takes whatever height is left.
    let runtime_h = (status.len() as u16).saturating_add(2);
    let left = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(runtime_h), Constraint::Min(0)])
        .split(cols[0]);
    frame.render_widget(card("Runtime", status), left[0]);

    let actions = vec![
        ListItem::new("Dashboard — serve status, tok/s, VRAM"),
        ListItem::new("Chat — talk to your local serve"),
        ListItem::new("Models — registry, downloads, pulls"),
        ListItem::new("Settings — edit ~/.hipfire config"),
        ListItem::new("System / Logs — GPU, paths, serve.log"),
    ];
    frame.render_widget(
        List::new(actions)
            .block(block("Tabs  (Tab / Shift+Tab, ? help)"))
            .style(Style::default().fg(TEXT).bg(PANEL)),
        left[1],
    );

    let quickstart = Text::from(vec![
        Line::from(Span::styled(
            "Quick start",
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        )),
        Line::from("1. Pull a model   ·  hipfire pull qwen3.5:4b"),
        Line::from("2. Start serving  ·  hipfire serve -d"),
        Line::from("3. Chat           ·  open the Chat tab and type"),
        Line::from(""),
        Line::from("Press ? on any tab for its keys."),
        Line::from(Span::styled(
            "(in Chat, press Esc first)",
            Style::default().fg(MUTED),
        )),
    ]);
    frame.render_widget(
        Paragraph::new(quickstart)
            .block(block("Getting started"))
            .wrap(Wrap { trim: false })
            .style(Style::default().fg(TEXT).bg(PANEL)),
        right[0],
    );

    let keys = vec![
        Row::new(["Tab / Shift+Tab", "switch tabs"]),
        Row::new(["\u{2191}  \u{2193} · Enter", "move · select"]),
        Row::new(["?", "help for this tab"]),
        Row::new(["r · q", "refresh · quit"]),
    ];
    frame.render_widget(
        Table::new(keys, [Constraint::Length(15), Constraint::Min(17)])
            .header(Row::new(["Key", "Action"]).style(Style::default().fg(MUTED)))
            .block(block("Keys  (full list in each tab's footer)"))
            .style(Style::default().fg(TEXT).bg(PANEL))
            .row_highlight_style(Style::default().bg(PANEL_2)),
        right[1],
    );
}

fn draw_dashboard(frame: &mut Frame, app: &App, area: Rect) {
    // One extra line for the multi-slot engine row, only when serve exports it.
    let card_height = match &app.live_metrics {
        MetricsView::Live(m) if m.engine.is_some() => 9,
        _ => 8,
    };
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(8), Constraint::Length(card_height)])
        .split(pad(area, 1, 0));
    draw_live_metrics(frame, app, rows[1]);
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(rows[0]);

    // ── Serve panel ─────────────────────────────────────────────────────
    let mut serve_lines: Vec<Line> = Vec::new();
    let dash = app.dashboard.as_ref();
    let serve_up = dash.map(|d| d.serve_up).unwrap_or(false);

    if dash.is_none() {
        serve_lines.push(Line::from(Span::styled(
            "Probing serve…",
            Style::default().fg(MUTED),
        )));
    } else if !serve_up {
        let d = dash.unwrap();
        serve_lines.push(Line::from(vec![
            Span::raw("Serve      "),
            Span::styled(
                "offline",
                Style::default().fg(RED).add_modifier(Modifier::BOLD),
            ),
        ]));
        serve_lines.push(Line::from(format!("Endpoint   {}", d.endpoint)));
        serve_lines.push(Line::from(""));
        serve_lines.push(Line::from(Span::styled(
            d.offline_hint
                .clone()
                .unwrap_or_else(|| "serve offline — start it with `hipfire serve -d`".into()),
            Style::default().fg(YELLOW),
        )));
        serve_lines.push(Line::from(Span::styled(
            "(Chat tab Enter also starts a background serve)",
            Style::default().fg(MUTED),
        )));
    } else {
        let d = dash.unwrap();
        serve_lines.push(Line::from(vec![
            Span::raw("Serve      "),
            Span::styled(
                "online",
                Style::default().fg(GREEN).add_modifier(Modifier::BOLD),
            ),
        ]));
        serve_lines.push(Line::from(format!("Endpoint   {}", d.endpoint)));
        serve_lines.push(Line::from(format!(
            "Model      {}",
            d.model.clone().unwrap_or_else(|| "(none loaded)".into())
        )));
        if let Some(s) = &d.stats {
            serve_lines.push(Line::from(format!("Uptime     {}", fmt_uptime(s.uptime_s))));
            serve_lines.push(Line::from(vec![
                Span::raw("Queue      "),
                Span::styled(
                    s.queue_depth.to_string(),
                    Style::default().fg(if s.queue_depth > 0 { YELLOW } else { TEXT }),
                ),
                Span::raw("  in-flight"),
            ]));
            serve_lines.push(Line::from(format!(
                "Requests   {} served",
                s.requests_served
            )));
            serve_lines.push(Line::from(match s.recent_tok_s {
                Some(t) => format!("Recent     {t:.1} tok/s"),
                None => "Recent     — (no completed generation yet)".into(),
            }));
        } else {
            serve_lines.push(Line::from(Span::styled(
                "stats unavailable (older serve build, /stats not present)",
                Style::default().fg(YELLOW),
            )));
        }
        if !d.model_ids.is_empty() {
            serve_lines.push(Line::from(format!(
                "Available  {} model(s)",
                d.model_ids.len()
            )));
        }
    }
    frame.render_widget(card("Live serve", serve_lines), cols[0]);

    // ── VRAM panel ──────────────────────────────────────────────────────
    let mut vram_lines: Vec<Line> = Vec::new();
    match dash.map(|d| &d.vram) {
        Some(VramState::Available(gpus)) => {
            for g in gpus {
                let pct = if g.total_bytes > 0 {
                    (g.used_bytes as f64 / g.total_bytes as f64) * 100.0
                } else {
                    0.0
                };
                let color = if pct > 90.0 {
                    RED
                } else if pct > 70.0 {
                    YELLOW
                } else {
                    GREEN
                };
                vram_lines.push(Line::from(vec![
                    Span::styled(
                        format!("GPU {}  ", g.index),
                        Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        format!(
                            "{:.1} / {:.1} GB used ({pct:.0}%)",
                            g.used_gb(),
                            g.total_gb()
                        ),
                        Style::default().fg(color),
                    ),
                ]));
                vram_lines.push(Line::from(Span::styled(
                    format!("        {:.1} GB free", g.free_gb()),
                    Style::default().fg(MUTED),
                )));
            }
        }
        Some(VramState::Unavailable(reason)) => {
            vram_lines.push(Line::from(Span::styled(
                reason.clone(),
                Style::default().fg(YELLOW),
            )));
        }
        None => {
            vram_lines.push(Line::from(Span::styled(
                "Probing GPU stats…",
                Style::default().fg(MUTED),
            )));
        }
    }

    // ── Live load (util / temp / power) ─────────────────────────────────
    match dash.map(|d| &d.gpu_load) {
        Some(LoadState::Available(loads)) => {
            vram_lines.push(Line::from(""));
            for l in loads {
                let mut parts = Vec::new();
                if let Some(u) = l.util_pct {
                    parts.push(format!("{u:.0}% util"));
                }
                if let Some(t) = l.temp_c {
                    parts.push(format!("{t:.0}°C"));
                }
                if let Some(p) = l.power_w {
                    parts.push(format!("{p:.0} W"));
                }
                let color = match l.util_pct {
                    Some(u) if u > 90.0 => RED,
                    Some(u) if u > 50.0 => YELLOW,
                    _ => GREEN,
                };
                let body = if parts.is_empty() {
                    "no load fields".to_string()
                } else {
                    parts.join("  ·  ")
                };
                vram_lines.push(Line::from(vec![
                    Span::styled(
                        format!("GPU {}  ", l.index),
                        Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(body, Style::default().fg(color)),
                ]));
            }
        }
        Some(LoadState::Unavailable(reason)) => {
            vram_lines.push(Line::from(""));
            vram_lines.push(Line::from(Span::styled(
                reason.clone(),
                Style::default().fg(MUTED),
            )));
        }
        None => {}
    }
    frame.render_widget(card("GPU", vram_lines), cols[1]);
}

/// "Live metrics" card: windowed rates and quantiles from the worker's
/// background `/metrics` ring. Every row is one short line so narrow terminals
/// truncate a row instead of reflowing the card.
fn draw_live_metrics(frame: &mut Frame, app: &App, area: Rect) {
    let outer = block("Live metrics");
    let inner = outer.inner(area);
    frame.render_widget(outer, area);
    let message = |text: &str, color: Color| {
        Paragraph::new(Span::styled(text.to_string(), Style::default().fg(color)))
    };
    let m = match &app.live_metrics {
        MetricsView::Live(m) => m,
        MetricsView::Collecting => {
            frame.render_widget(message("collecting…", MUTED), inner);
            return;
        }
        MetricsView::Unavailable => {
            frame.render_widget(message("metrics unavailable (older serve)", YELLOW), inner);
            return;
        }
        MetricsView::Offline => {
            frame.render_widget(message("serve offline", MUTED), inner);
            return;
        }
    };
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Min(0),
        ])
        .split(inner);
    for (row, label, series, color) in [
        (
            rows[0],
            format!("tok/s {:>8.1}", m.tok_s),
            &m.tok_s_series,
            GREEN,
        ),
        (
            rows[1],
            format!("req/s {:>8.2}", m.req_s),
            &m.req_s_series,
            ACCENT,
        ),
    ] {
        let cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Length(16), Constraint::Min(0)])
            .split(row);
        frame.render_widget(Paragraph::new(label), cols[0]);
        // Sparkline draws from the first point; keep the newest that fit.
        let tail = &series[series.len().saturating_sub(cols[1].width as usize)..];
        frame.render_widget(
            Sparkline::default()
                .data(tail)
                .style(Style::default().fg(color)),
            cols[1],
        );
    }
    let quantiles = |label: &str, q: Option<(f64, f64)>| match q {
        Some((p50, p95)) => format!("{label}  p50 {p50:.0} ms · p95 {p95:.0} ms"),
        None => format!("{label}  —"),
    };
    let mut extra = vec![format!("{} window", fmt_uptime(m.window_s as u64))];
    if let Some(pct) = m.cache_hit_pct {
        extra.insert(0, format!("cache hit {pct:.0}%"));
    }
    if let Some(tau) = m.tau {
        extra.push(format!("tau {tau:.2}"));
    }
    let mut lines = vec![
        Line::from(quantiles("TTFT", m.ttft_ms)),
        Line::from(quantiles("TPOT", m.tpot_ms)),
        Line::from(Span::styled(
            format!(
                "errors {:.1}/min · rejects {:.1}/min",
                m.errors_per_min, m.rejects_per_min
            ),
            Style::default().fg(if m.errors_per_min + m.rejects_per_min > 0.0 {
                YELLOW
            } else {
                TEXT
            }),
        )),
        Line::from(Span::styled(extra.join(" · "), Style::default().fg(MUTED))),
    ];
    if let Some(e) = &m.engine {
        let mut row = vec![format!("slots {:.0}/{:.0}", e.slots_active, e.slots_total)];
        if let Some((free, total)) = e.kv_pages {
            row.push(format!("KV free {free:.0}/{total:.0} pages"));
        }
        row.push(format!(
            "evict {:.0} · restore {:.0} · prefix hits {:.0}",
            e.evictions, e.restores, e.prefix_hits
        ));
        lines.push(Line::from(row.join(" · ")));
    }
    frame.render_widget(Paragraph::new(lines), rows[2]);
}

fn fmt_uptime(secs: u64) -> String {
    let h = secs / 3600;
    let m = (secs % 3600) / 60;
    let s = secs % 60;
    if h > 0 {
        format!("{h}h {m}m {s}s")
    } else if m > 0 {
        format!("{m}m {s}s")
    } else {
        format!("{s}s")
    }
}

/// Hard-wrap chat input to `width` display columns. Returns the rows to draw;
/// the cursor sits at the end of the last row (a trailing newline, or a last
/// row that exactly fills the width, yields a fresh empty row for it). Wrapping
/// is by character (not word) so the cursor position is exact.
fn wrap_input_rows(input: &str, width: usize) -> Vec<String> {
    let width = width.max(1);
    let mut rows = vec![String::new()];
    let mut col = 0usize;
    for c in input.chars() {
        if c == '\n' {
            rows.push(String::new());
            col = 0;
            continue;
        }
        let w = Span::raw(c.to_string()).width();
        if col + w > width && col > 0 {
            rows.push(String::new());
            col = 0;
        }
        if let Some(row) = rows.last_mut() {
            row.push(c);
        }
        col += w;
    }
    if col >= width {
        rows.push(String::new());
    }
    rows
}

fn draw_chat(frame: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(10), Constraint::Length(5)])
        .split(pad(area, 1, 0));

    let mut lines = Vec::new();
    if app.chat.messages.is_empty() {
        lines.push(Line::from(Span::styled(
            "No messages yet. Type below and press Enter.",
            Style::default().fg(MUTED),
        )));
        lines.push(Line::from(""));
        lines.push(Line::from(
            "Chat streams from your local hipfire serve (OpenAI-compatible).",
        ));
    } else {
        // Fenced ``` code blocks get a distinct background; comments + inline
        // `code` are tinted. Rendering is isolated in `ui_chat`.
        let code_theme = crate::ui_chat::CodeTheme {
            text: TEXT,
            code_fg: ACCENT,
            code_bg: PANEL_2,
            comment: MUTED,
            fence: YELLOW,
        };
        let thinking = Style::default().fg(MUTED).add_modifier(Modifier::ITALIC);
        for (idx, msg) in app.chat.messages.iter().enumerate() {
            let color = if msg.role == "user" { ACCENT } else { GREEN };
            lines.push(Line::from(Span::styled(
                format!("{}:", msg.role),
                Style::default().fg(color).add_modifier(Modifier::BOLD),
            )));
            // Reasoning is rendered before content in the same bubble, in a
            // muted italic style under a `thinking` label so it reads apart from
            // the answer.
            if let Some(rc) = msg.reasoning_content.as_deref().filter(|s| !s.is_empty()) {
                lines.push(Line::from(Span::styled(
                    "thinking",
                    thinking.add_modifier(Modifier::DIM),
                )));
                lines.extend(
                    rc.lines()
                        .map(|l| Line::from(Span::styled(l.to_string(), thinking))),
                );
                // Thin separator when both channels present.
                if !msg.content.is_empty() {
                    lines.push(Line::from(""));
                }
            }
            lines.extend(crate::ui_chat::render_body(&msg.content, &code_theme));
            if app.chat.stopped.contains(&idx) {
                lines.push(Line::from(Span::styled(
                    "[stopped]",
                    Style::default().fg(YELLOW).add_modifier(Modifier::DIM),
                )));
            }
            lines.push(Line::from(""));
        }
    }
    if let Some(notice) = &app.chat.notice {
        let dim = Style::default().fg(MUTED).add_modifier(Modifier::DIM);
        lines.extend(
            notice
                .lines()
                .map(|l| Line::from(Span::styled(l.to_string(), dim))),
        );
    }

    // Scroll is measured from the bottom (0 = follow the newest output): measure
    // the wrapped transcript at the current inner width, then offset from the end.
    let transcript_block = block("Chat shell");
    let inner = transcript_block.inner(chunks[0]);
    let para = Paragraph::new(lines).wrap(Wrap { trim: false });
    let total = if inner.width == 0 {
        0
    } else {
        para.line_count(inner.width)
    };
    let max_scroll = total
        .saturating_sub(usize::from(inner.height))
        .min(usize::from(u16::MAX)) as u16;
    app.chat.view.set(Some(crate::app::ChatView {
        viewport: inner.height,
        max_scroll,
    }));
    let scroll = app.chat.scroll.min(max_scroll);
    let transcript_block = if scroll > 0 {
        transcript_block.title_bottom(
            Line::from(Span::styled(
                format!(" \u{2193} {scroll} more (End to follow) "),
                Style::default().fg(YELLOW),
            ))
            .right_aligned(),
        )
    } else {
        transcript_block
    };
    frame.render_widget(
        para.block(transcript_block)
            .scroll((max_scroll - scroll, 0))
            .style(Style::default().fg(TEXT).bg(PANEL)),
        chunks[0],
    );

    let input_title = if app.chat.sending {
        format!("Input - {} - model {}", app.chat.status, app.active_model)
    } else {
        let mut extras = vec![format!("model {}", app.active_model)];
        if !app.chat.system_prompt.is_empty() {
            extras.push("sys".into());
        }
        if let Some(t) = app.chat.temp {
            extras.push(format!("temp {t}"));
        }
        if let Some(p) = app.chat.top_p {
            extras.push(format!("top_p {p}"));
        }
        if let Some(s) = app.chat.last_stats {
            let unit = s.unit();
            extras.push(format!("{} {unit} @ {:.0} {unit}/s", s.tokens, s.tps));
        }
        format!("Input - {} - {}", app.chat.status, extras.join(" · "))
    };
    let input_block = block(&input_title);
    let input_inner = input_block.inner(chunks[1]);
    // Pre-wrapped rows: show the tail so the cursor row is always visible.
    let rows = wrap_input_rows(&app.chat.input, usize::from(input_inner.width));
    let visible = usize::from(input_inner.height).max(1);
    let first = rows.len().saturating_sub(visible);
    let input = Paragraph::new(
        rows[first..]
            .iter()
            .map(|r| Line::from(r.as_str()))
            .collect::<Vec<_>>(),
    )
    .block(input_block)
    .style(Style::default().fg(TEXT).bg(PANEL_2));
    frame.render_widget(input, chunks[1]);
    if app.chat.is_input_focused()
        && !app.show_help
        && input_inner.width > 0
        && input_inner.height > 0
    {
        let last = rows.len() - 1;
        let col = Span::raw(rows[last].as_str()).width();
        let x = input_inner.x + (col as u16).min(input_inner.width - 1);
        let y = input_inner.y + (last - first) as u16;
        frame.set_cursor_position((x, y));
    }
}

/// Short registry source label shared by the Models header and Home: the
/// fetched cache path, `bundled (<date>)` for the compiled-in copy, or
/// `missing` when there really is none.
fn registry_label(app: &App) -> String {
    match (&app.registry.loaded_path, app.registry.bundled) {
        (Some(path), _) => path.display().to_string(),
        (None, true) => match app.registry.generated_at.get(..10) {
            Some(date) => format!("bundled ({date})"),
            None => "bundled".into(),
        },
        (None, false) => "missing".into(),
    }
}

fn draw_models(frame: &mut Frame, app: &App, area: Rect) {
    // An in-flight pull or an armed delete-confirm gets a 3-row status strip at
    // the bottom; otherwise the list takes the full height.
    let show_status = app.pull.is_some() || app.confirm_delete.is_some();
    let constraints: Vec<Constraint> = if show_status {
        vec![
            Constraint::Length(3),
            Constraint::Min(6),
            Constraint::Length(3),
        ]
    } else {
        vec![Constraint::Length(3), Constraint::Min(8)]
    };
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .split(pad(area, 1, 0));
    let registry_label = registry_label(app);
    // Whole parts only, most useful first: at 80 columns `aliases` drops out
    // instead of the registry label being cut mid-date.
    let summary = fit_segments(
        &[
            format!("active: {}", app.active_model),
            format!(
                "{} of {} downloaded",
                app.registry.downloaded_count(),
                app.registry.models.len()
            ),
            format!("registry: {registry_label}"),
            format!("aliases: {}", app.registry.aliases.len()),
        ],
        chunks[0].width.saturating_sub(2) as usize,
    );
    frame.render_widget(
        Paragraph::new(summary)
            .block(block("Model hub"))
            .style(Style::default().fg(TEXT).bg(PANEL)),
        chunks[0],
    );

    let visible_items = app.registry.visible_items();
    if visible_items.is_empty() {
        // Empty-state guidance: no registry entries and no local downloads.
        let lines = vec![
            Line::from(Span::styled(
                "No models found.",
                Style::default().fg(YELLOW).add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
            Line::from("Pull a model with:"),
            Line::from(Span::styled(
                "    hipfire pull <id>",
                Style::default().fg(GREEN),
            )),
            Line::from(""),
            Line::from(Span::styled(
                "The bundled registry lists available ids; downloads land in the configured models directory.",
                Style::default().fg(MUTED),
            )),
        ];
        frame.render_widget(
            Paragraph::new(Text::from(lines))
                .block(block("Registry browser"))
                .wrap(Wrap { trim: false })
                .style(Style::default().fg(TEXT).bg(PANEL)),
            chunks[1],
        );
        return;
    }
    // Fixed-width columns; the Tag column grows to the longest visible tag (up to
    // what leaves Notes a usable remainder) so wide terminals show full tags.
    const TAG_MIN: usize = 24;
    const HAVE_W: usize = 11; // fits "NN/NN local" group counts
    const SIZE_W: usize = 9;
    const VRAM_W: usize = 7;
    const COL_GAPS: usize = 4;
    // Notes is the most expendable column: at 80 columns the full tag
    // (`qwen3.8:flash-next-mq3-xts`) matters more than the description.
    const NOTES_MIN: usize = 14;
    let inner_w = chunks[1].width.saturating_sub(2) as usize;
    let longest_tag = visible_items
        .iter()
        .map(|item| match item {
            ModelListItem::Group { name, .. } => 2 + name.chars().count(),
            ModelListItem::Model { model_index } => {
                4 + app.registry.models[*model_index].tag.chars().count()
            }
        })
        .max()
        .unwrap_or(0);
    let tag_cap = inner_w
        .saturating_sub(HAVE_W + SIZE_W + VRAM_W + COL_GAPS + NOTES_MIN)
        .max(TAG_MIN);
    let tag_w = longest_tag.clamp(TAG_MIN, tag_cap);
    let rows = visible_items
        .iter()
        .enumerate()
        .skip(scroll_start(app.registry.selected, chunks[1].height, 3))
        .take(visible_rows(chunks[1].height, 3))
        .map(|(idx, item)| {
            let selected = idx == app.registry.selected;
            let row = match item {
                ModelListItem::Group {
                    name,
                    count,
                    downloaded,
                    expanded,
                } => {
                    let marker = if *expanded { "v" } else { ">" };
                    // The action hint belongs to the selected group only, and
                    // stays short enough for the 80-column Notes cell (Enter
                    // toggles too; the footer says so).
                    let hint = match (selected, *expanded) {
                        (false, _) => "",
                        (true, false) => "\u{2192} expand",
                        (true, true) => "\u{2190} collapse",
                    };
                    Row::new([
                        format!("{marker} {name}"),
                        format!("{downloaded}/{count} local"),
                        String::new(),
                        String::new(),
                        hint.into(),
                    ])
                }
                ModelListItem::Model { model_index } => {
                    let row = &app.registry.models[*model_index];
                    let active = if row.tag == app.active_model {
                        "*"
                    } else {
                        " "
                    };
                    let have = if row.downloaded { "local" } else { "remote" };
                    let mtp = if row.has_mtp { "[mtp] " } else { "" };
                    let own = crate::hipfire::scope_view::own_count(
                        &app.scope.catalog,
                        &row.tag,
                        &row.entry.file,
                    );
                    let own = if own > 0 {
                        format!("{} · ", crate::hipfire::scope_view::own_values_text(own))
                    } else {
                        String::new()
                    };
                    let notes = if row.entry.repo.is_empty() {
                        format!("{own}{mtp}{} (no remote repo)", row.entry.desc)
                    } else {
                        format!("{own}{mtp}{}", row.entry.desc)
                    };
                    Row::new([
                        format!("  {active} {}", row.tag),
                        have.into(),
                        format!("{:.1} GB", row.entry.size_gb),
                        // Local files outside the registry carry no VRAM
                        // estimate; say so instead of a fake "0 GB".
                        if row.entry.min_vram_gb > 0.0 {
                            format!("{:.0} GB", row.entry.min_vram_gb)
                        } else {
                            "\u{2014}".into()
                        },
                        notes,
                    ])
                }
            };
            row.style(if selected {
                Style::default().fg(ACCENT).bg(PANEL_2)
            } else {
                match item {
                    ModelListItem::Group { .. } => Style::default()
                        .fg(YELLOW)
                        .bg(PANEL)
                        .add_modifier(Modifier::BOLD),
                    ModelListItem::Model { model_index } => {
                        if app.registry.models[*model_index].tag == app.active_model {
                            Style::default().fg(GREEN).bg(PANEL)
                        } else {
                            Style::default().fg(TEXT).bg(PANEL)
                        }
                    }
                }
            })
        })
        .collect::<Vec<_>>();
    let table = Table::new(
        rows,
        [
            Constraint::Length(tag_w as u16),
            Constraint::Length(HAVE_W as u16),
            Constraint::Length(SIZE_W as u16),
            Constraint::Length(VRAM_W as u16),
            Constraint::Min(10),
        ],
    )
    .header(Row::new(["Tag", "Have", "Size", "VRAM", "Notes"]).style(Style::default().fg(MUTED)))
    .block(block("Registry browser"))
    .style(Style::default().bg(PANEL));
    frame.render_widget(table, chunks[1]);

    if show_status {
        draw_models_status(frame, app, chunks[2]);
    }
    if let Some(choice) = &app.pull_choice {
        draw_pull_choice(frame, choice, area);
    }
}

/// The `p` pull chooser popup, centered over the Models tab: a `PLE rows` line
/// (auto / q4 / q8 with sizes) and a `Draft sidecars` line (yes / no with the
/// kinds and sizes), the focused row marked with `>`.
fn draw_pull_choice(frame: &mut Frame, choice: &PullChoice, area: Rect) {
    use crate::hipfire::pull_choice::{format_size, ChoiceRow};

    let selected = Style::default().fg(GREEN).add_modifier(Modifier::BOLD);
    let plain = Style::default().fg(MUTED);
    let label_style = |focused: bool| {
        if focused {
            Style::default().fg(TEXT).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(TEXT)
        }
    };
    let marker = |row: ChoiceRow| if choice.row == row { "> " } else { "  " };
    let option = |text: String, on: bool| {
        if on {
            Span::styled(format!("[{text}]"), selected)
        } else {
            Span::styled(format!(" {text} "), plain)
        }
    };

    let mut lines: Vec<Line> = vec![Line::from(""), {
        Line::from(Span::styled(
            format!("  {}", choice.tag),
            Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
        ))
    }];
    lines.push(Line::from(""));
    if let Some(ple) = &choice.ple {
        let mut spans = vec![Span::styled(
            format!("{}PLE rows        ", marker(ChoiceRow::Ple)),
            label_style(choice.row == ChoiceRow::Ple),
        )];
        for (i, opt) in ple.options.iter().enumerate() {
            let size = format_size(opt.size_bytes);
            let text = if size.is_empty() {
                opt.label.to_string()
            } else {
                format!("{} {size}", opt.label)
            };
            spans.push(option(text, i == ple.selected));
        }
        lines.push(Line::from(spans));
    }
    if let Some(draft) = &choice.draft {
        let kinds = draft
            .kinds
            .iter()
            .map(|k| {
                let size = format_size(k.size_bytes);
                if size.is_empty() {
                    k.label.to_string()
                } else {
                    format!("{} {size}", k.label)
                }
            })
            .collect::<Vec<_>>()
            .join(", ");
        lines.push(Line::from(vec![
            Span::styled(
                format!("{}Draft sidecars  ", marker(ChoiceRow::Draft)),
                label_style(choice.row == ChoiceRow::Draft),
            ),
            option("yes".into(), draft.yes),
            option("no".into(), !draft.yes),
            Span::styled(format!("  {kinds}"), plain),
        ]));
    }
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "  Up/Down row \u{b7} Left/Right/Space change \u{b7} Enter pull \u{b7} Esc cancel",
        plain,
    )));

    let w = 78u16.min(area.width);
    let h = (lines.len() as u16 + 2).min(area.height);
    let rect = Rect {
        x: area.x + area.width.saturating_sub(w) / 2,
        y: area.y + area.height.saturating_sub(h) / 2,
        width: w,
        height: h,
    };
    frame.render_widget(Clear, rect);
    frame.render_widget(
        Paragraph::new(lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(ACCENT))
                    .title(" Pull options "),
            )
            .style(Style::default().fg(TEXT).bg(PANEL)),
        rect,
    );
}

/// The Models-tab bottom strip: a delete-confirm prompt (takes priority) or the
/// in-flight pull's progress gauge (parsed percent + the raw CLI line label).
fn draw_models_status(frame: &mut Frame, app: &App, area: Rect) {
    if let Some(tag) = &app.confirm_delete {
        frame.render_widget(
            Paragraph::new(format!(
                "Delete {tag}?  press y to confirm, n / Esc to cancel"
            ))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(RED))
                    .title(" Confirm delete "),
            )
            .style(Style::default().fg(TEXT).bg(PANEL)),
            area,
        );
    } else if let Some(job) = &app.pull {
        let ratio = job.percent.unwrap_or(0.0).clamp(0.0, 100.0) / 100.0;
        let label = if job.handle.cancel_requested() {
            "cancelling\u{2026}".to_string()
        } else if job.line.is_empty() {
            format!("pulling {}\u{2026}", job.tag)
        } else {
            job.line.clone()
        };
        frame.render_widget(
            Gauge::default()
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title(format!(" Pulling {} · c cancel ", job.tag)),
                )
                .gauge_style(Style::default().fg(GREEN).bg(PANEL_2))
                .ratio(ratio)
                .label(label),
            area,
        );
    }
}

fn draw_system(frame: &mut Frame, app: &App, area: Rect) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(52), Constraint::Percentage(48)])
        .split(pad(area, 1, 0));
    let left = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(10), Constraint::Min(8)])
        .split(cols[0]);

    let gpu_lines = app
        .status
        .gpu_lines
        .iter()
        .map(|line| Line::from(line.clone()))
        .collect::<Vec<_>>();
    frame.render_widget(
        Paragraph::new(gpu_lines)
            .block(block("Hardware glimpse"))
            .wrap(Wrap { trim: false })
            .style(Style::default().fg(TEXT).bg(PANEL)),
        left[0],
    );

    let paths = app
        .status
        .paths_ok
        .iter()
        .map(|(label, ok)| {
            Row::new([
                label.clone(),
                if *ok {
                    "present".into()
                } else {
                    "missing".into()
                },
            ])
            .style(if *ok {
                Style::default().fg(GREEN)
            } else {
                Style::default().fg(YELLOW)
            })
        })
        .collect::<Vec<_>>();
    frame.render_widget(
        Table::new(paths, [Constraint::Length(24), Constraint::Length(12)])
            .header(Row::new(["Path", "Status"]).style(Style::default().fg(MUTED)))
            .block(block("Files"))
            .style(Style::default().bg(PANEL)),
        left[1],
    );

    let mut diagnostic_lines = vec![Line::from(Span::styled(
        "Live diagnostics",
        Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
    ))];
    match app.dashboard.as_ref().and_then(|d| d.system.as_ref()) {
        Some(sys) => {
            diagnostic_lines.push(probe_line("GPU", &sys.gpu_name));
            diagnostic_lines.push(probe_line("Arch", &sys.gpu_arch));
            diagnostic_lines.push(probe_line("HIP/ROCm", &sys.hip_version));
            diagnostic_lines.push(probe_line("Kernel cache", &sys.kernel_cache));
            diagnostic_lines.push(probe_line("Loaded model", &sys.loaded_model));
            diagnostic_lines.push(probe_line("Checksum", &sys.model_checksum));
        }
        None => {
            // Worker has not produced a snapshot yet (first frame on this tab).
            diagnostic_lines.push(Line::from(Span::styled(
                "Probing GPU / HIP / kernel cache…",
                Style::default().fg(MUTED),
            )));
        }
    }
    diagnostic_lines.extend([
        Line::from(""),
        Line::from(Span::styled(
            "Local model files:",
            Style::default().fg(MUTED),
        )),
    ]);
    if app.registry.local_files.is_empty() {
        // Empty-state guidance: tell the user how to get a model.
        diagnostic_lines.push(Line::from(Span::styled(
            "No local models under ~/.hipfire/models.",
            Style::default().fg(YELLOW),
        )));
        diagnostic_lines.push(Line::from(Span::styled(
            "Pull one with `hipfire pull <id>` (see the Models tab for ids).",
            Style::default().fg(YELLOW),
        )));
    } else {
        diagnostic_lines.extend(
            app.registry
                .local_files
                .iter()
                .take(6)
                .map(|m| Line::from(format!("{}  {}", m.size, m.file))),
        );
    }
    // Request inspector — the TUI's own recent chat generations (honest, local).
    diagnostic_lines.push(Line::from(""));
    diagnostic_lines.push(Line::from(Span::styled(
        "Recent requests (this session):",
        Style::default().fg(MUTED),
    )));
    if app.request_log.is_empty() {
        diagnostic_lines.push(Line::from(Span::styled(
            "none yet — send a Chat message",
            Style::default().fg(MUTED),
        )));
    } else {
        for r in app.request_log.iter().take(8) {
            diagnostic_lines.push(Line::from(format!(
                "{}  {} {} · {:.0} {}/s · {:.1}s",
                r.model,
                r.tokens,
                r.unit(),
                r.tps,
                r.unit(),
                r.secs
            )));
        }
    }

    // In-TUI doctor — `hipfire diag` pass/fail (on-demand via `d`).
    diagnostic_lines.push(Line::from(""));
    if app.doctor_running() {
        diagnostic_lines.push(Line::from(Span::styled(
            "Doctor: running hipfire diag…",
            Style::default().fg(YELLOW),
        )));
    } else {
        match app.doctor.as_ref() {
            None => diagnostic_lines.push(Line::from(Span::styled(
                "Doctor: press d to run hipfire diag",
                Style::default().fg(MUTED),
            ))),
            Some(report) if report.error.is_some() => {
                diagnostic_lines.push(Line::from(Span::styled(
                    format!("Doctor error: {}", report.error.as_deref().unwrap_or("")),
                    Style::default().fg(RED),
                )));
            }
            Some(report) => {
                diagnostic_lines.push(Line::from(Span::styled(
                    "Doctor (hipfire diag):",
                    Style::default().fg(MUTED),
                )));
                for c in &report.checks {
                    let (mark, color) = if c.ok { ("ok ", GREEN) } else { ("XX ", RED) };
                    diagnostic_lines.push(Line::from(vec![
                        Span::styled(format!("  {mark}"), Style::default().fg(color)),
                        Span::styled(format!("{}: ", c.name), Style::default().fg(TEXT)),
                        Span::styled(c.detail.clone(), Style::default().fg(MUTED)),
                    ]));
                }
            }
        }
    }

    diagnostic_lines.extend([
        Line::from(""),
        Line::from(Span::styled(
            "Serve /health response:",
            Style::default().fg(MUTED),
        )),
        Line::from(if app.status.health_text.is_empty() {
            "serve offline — start with `hipfire serve -d`".to_string()
        } else {
            app.status.health_text.chars().take(300).collect::<String>()
        }),
    ]);
    frame.render_widget(
        Paragraph::new(Text::from(diagnostic_lines))
            .block(block("System"))
            .wrap(Wrap { trim: false })
            .style(Style::default().fg(TEXT).bg(PANEL)),
        cols[1],
    );
}

/// Render one `label: value` diagnostic line, coloring an honest "unavailable"
/// probe yellow and a real value green-tinted text.
fn probe_line(label: &str, probe: &crate::hipfire::dashboard::Probe) -> Line<'static> {
    let value_style = if probe.is_available() {
        Style::default().fg(TEXT)
    } else {
        Style::default().fg(YELLOW)
    };
    Line::from(vec![
        Span::styled(format!("{label:<13}"), Style::default().fg(MUTED)),
        Span::styled(probe.display().to_string(), value_style),
    ])
}

/// `label` + `value` on one row `width` columns wide. An over-long value keeps
/// its tail (`…` prefix), which is the informative end of a path or tag.
fn fit_line(label: &str, value: &str, width: usize, style: Style) -> Line<'static> {
    let room = width.saturating_sub(label.chars().count()).max(1);
    let n = value.chars().count();
    let shown: String = if n <= room {
        value.to_string()
    } else {
        let tail: String = value.chars().skip(n - (room - 1)).collect();
        format!("\u{2026}{tail}")
    };
    Line::from(vec![
        Span::raw(label.to_string()),
        Span::styled(shown, style),
    ])
}

/// `label` + `value` wrapped to `width` columns, continuation rows indented
/// under the value, so the returned line count is the exact rendered height.
fn wrap_labeled(label: &str, value: &str, width: usize, style: Style) -> Vec<Line<'static>> {
    let indent = label.chars().count();
    let room = width.saturating_sub(indent).max(1);
    let chars: Vec<char> = value.chars().collect();
    let mut out = Vec::new();
    for (i, chunk) in chars.chunks(room).enumerate() {
        let lead = if i == 0 {
            label.to_string()
        } else {
            " ".repeat(indent)
        };
        out.push(Line::from(vec![
            Span::styled(lead, style),
            Span::styled(chunk.iter().collect::<String>(), style),
        ]));
    }
    if out.is_empty() {
        out.push(Line::from(Span::styled(label.to_string(), style)));
    }
    out
}

fn card(title: &str, lines: Vec<Line<'static>>) -> Paragraph<'static> {
    Paragraph::new(lines)
        .block(block(title))
        .style(Style::default().fg(TEXT).bg(PANEL))
        .wrap(Wrap { trim: false })
}

fn draw_logs(frame: &mut Frame, app: &App, area: Rect) {
    use crate::hipfire::log_tail::LogStatus;
    let inner = pad(area, 1, 0);
    let (title, body): (String, Text) =
        match &app.logs.status {
            LogStatus::Pending => (
                "serve.log".into(),
                Text::from(Line::from(Span::styled(
                    "reading serve.log\u{2026}",
                    Style::default().fg(MUTED),
                ))),
            ),
            LogStatus::Missing => (
                "serve.log (not found)".into(),
                Text::from(vec![
                Line::from(Span::styled("No serve.log yet.", Style::default().fg(YELLOW))),
                Line::from(""),
                Line::from(Span::styled(
                    "Start serve (Dashboard tab: s) — its output lands in ~/.hipfire/serve.log.",
                    Style::default().fg(MUTED),
                )),
            ]),
            ),
            LogStatus::Empty => (
                "serve.log (empty)".into(),
                Text::from(Line::from(Span::styled(
                    "serve.log is empty.",
                    Style::default().fg(MUTED),
                ))),
            ),
            LogStatus::Error(e) => (
                "serve.log (error)".into(),
                Text::from(Line::from(Span::styled(
                    format!("read error: {e}"),
                    Style::default().fg(RED),
                ))),
            ),
            LogStatus::Ok => {
                let title = format!("serve.log (last {} lines)", app.logs.lines.len());
                // Show the tail that fits, newest at the bottom.
                let height = inner.height.saturating_sub(2) as usize; // borders
                let start = app.logs.lines.len().saturating_sub(height.max(1));
                let lines: Vec<Line> = app.logs.lines[start..]
                    .iter()
                    .map(|l| Line::from(l.as_str()))
                    .collect();
                (title, Text::from(lines))
            }
        };
    frame.render_widget(
        Paragraph::new(body)
            .block(block(&title))
            .style(Style::default().fg(TEXT).bg(PANEL)),
        inner,
    );
}

pub(crate) fn block(title: &str) -> Block<'static> {
    Block::default()
        .title(Span::styled(
            format!(" {title} "),
            Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
        ))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(PANEL_2))
        .style(Style::default().bg(PANEL))
}

pub(crate) fn pad(area: Rect, x: u16, y: u16) -> Rect {
    Rect {
        x: area.x.saturating_add(x),
        y: area.y.saturating_add(y),
        width: area.width.saturating_sub(x * 2),
        height: area.height.saturating_sub(y * 2),
    }
}

fn visible_rows(height: u16, chrome: u16) -> usize {
    height.saturating_sub(chrome).max(1) as usize
}

fn scroll_start(selected: usize, height: u16, chrome: u16) -> usize {
    let visible = visible_rows(height, chrome);
    selected.saturating_sub(visible.saturating_sub(1))
}

#[cfg(test)]
mod render_tests {
    use super::*;
    use crate::app::App;
    use crate::hipfire::dashboard::{Dashboard, GpuLoad, GpuMem, LoadState, ServeStats, VramState};
    use ratatui::{backend::TestBackend, Terminal};

    /// Render the whole UI for the Dashboard tab with the given snapshot and
    /// return the flattened buffer text. The app is hermetic (an empty temp
    /// home); the Dashboard panel itself draws purely from `app.dashboard`,
    /// which we inject here.
    fn render_dashboard(dash: Option<Dashboard>) -> String {
        let (mut app, _home) = crate::test_support::hermetic_app();
        app.tab = Tab::Dashboard;
        app.dashboard = dash;
        let backend = TestBackend::new(110, 30);
        let mut terminal = Terminal::new(backend).expect("terminal");
        terminal
            .draw(|frame| draw(frame, &mut app))
            .expect("draw must not panic");
        let buf = terminal.backend().buffer().clone();
        buf.content().iter().map(|c| c.symbol()).collect()
    }

    fn render_metrics(view: crate::hipfire::metrics::MetricsView, width: u16) -> String {
        let (mut app, _home) = crate::test_support::hermetic_app();
        app.tab = Tab::Dashboard;
        app.live_metrics = view;
        let mut terminal = Terminal::new(TestBackend::new(width, 30)).expect("terminal");
        terminal
            .draw(|frame| draw(frame, &mut app))
            .expect("draw must not panic");
        let buf = terminal.backend().buffer().clone();
        buf.content().iter().map(|c| c.symbol()).collect()
    }

    #[test]
    fn live_metrics_card_states() {
        use crate::hipfire::metrics::{EngineRow, LiveMetrics, MetricsView};
        let live = MetricsView::Live(LiveMetrics {
            window_s: 180.0,
            req_s: 0.33,
            tok_s: 47.2,
            errors_per_min: 0.0,
            rejects_per_min: 0.0,
            cache_hit_pct: Some(50.0),
            ttft_ms: Some((62.5, 100.0)),
            tpot_ms: None,
            tau: Some(3.0),
            tok_s_series: (0..119).collect(),
            req_s_series: vec![0; 119],
            engine: None,
        });
        let text = render_metrics(live.clone(), 110);
        assert!(text.contains("Live metrics"));
        assert!(text.contains("tok/s     47.2"));
        assert!(text.contains("TTFT  p50 62 ms · p95 100 ms"));
        assert!(text.contains("TPOT  —"));
        assert!(text.contains("cache hit 50% · 3m 0s window · tau 3.00"));
        assert!(
            !text.contains("slots "),
            "standard route shows no engine row"
        );
        // Narrow terminal: rows truncate instead of panicking or reflowing.
        assert!(render_metrics(live.clone(), 40).contains("tok/s     47.2"));
        let MetricsView::Live(mut multi) = live else {
            unreachable!()
        };
        multi.engine = Some(EngineRow {
            slots_active: 1.0,
            slots_total: 4.0,
            kv_pages: Some((90.0, 128.0)),
            evictions: 2.0,
            restores: 1.0,
            prefix_hits: 3.0,
        });
        let text = render_metrics(MetricsView::Live(multi), 110);
        assert!(
            text.contains("slots 1/4 · KV free 90/128 pages · evict 2 · restore 1 · prefix hits 3")
        );
        // The engine row is extra: the existing rows stay visible.
        assert!(text.contains("cache hit 50% · 3m 0s window · tau 3.00"));
        assert!(render_metrics(MetricsView::Collecting, 110).contains("collecting…"));
        assert!(render_metrics(MetricsView::Unavailable, 110)
            .contains("metrics unavailable (older serve)"));
    }

    #[test]
    fn renders_online_with_stats_and_vram() {
        let dash = Dashboard {
            serve_up: true,
            endpoint: "127.0.0.1:11435".into(),
            health_text: r#"{"status":"ok","model":"qwen3.5:9b"}"#.into(),
            model: Some("qwen3.5:9b".into()),
            stats: Some(ServeStats {
                model: Some("qwen3.5:9b".into()),
                uptime_s: 3725, // 1h 2m 5s
                queue_depth: 1,
                requests_served: 7,
                recent_tok_s: Some(151.2),
                loaded_config: None,
            }),
            model_ids: vec!["qwen3.5:9b".into(), "qwen3.5:27b".into()],
            vram: VramState::Available(vec![GpuMem {
                index: 0,
                used_bytes: 8_000_000_000,
                total_bytes: 24_000_000_000,
            }]),
            gpu_load: LoadState::Available(vec![GpuLoad {
                index: 0,
                util_pct: Some(37.0),
                temp_c: Some(45.0),
                power_w: Some(120.0),
            }]),
            offline_hint: None,
            system: None,
        };
        let text = render_dashboard(Some(dash));
        assert!(text.contains("online"), "expected online marker");
        assert!(text.contains("qwen3.5:9b"), "expected model name");
        assert!(text.contains("127.0.0.1:11435"), "expected endpoint");
        assert!(text.contains("1h 2m 5s"), "expected uptime");
        assert!(text.contains("served"), "expected requests served");
        assert!(text.contains("tok/s"), "expected recent tok/s");
        assert!(text.contains("GPU 0"), "expected per-GPU VRAM");
        assert!(text.contains("37% util"), "expected live GPU utilization");
        assert!(text.contains("120 W"), "expected live GPU power");
    }

    #[test]
    fn renders_offline_state_without_panic_or_stale_numbers() {
        let dash = Dashboard::offline(
            "127.0.0.1:11435".into(),
            VramState::Unavailable(
                "GPU stats unavailable: neither amd-smi nor rocm-smi found (ROCm root: none)"
                    .into(),
            ),
            LoadState::Unavailable("GPU load unavailable: no GPU stats tool".into()),
        );
        let text = render_dashboard(Some(dash));
        assert!(text.contains("offline"), "expected offline marker");
        assert!(text.contains("hipfire serve"), "expected start hint");
        assert!(
            text.contains("GPU stats unavailable"),
            "expected honest GPU-stats-unavailable line"
        );
        assert!(
            !text.contains("No GPU detected") && !text.contains("not installed"),
            "a missing stats tool must not read as a missing GPU"
        );
        // No fabricated telemetry: offline must not invent a tok/s or queue.
        assert!(!text.contains("tok/s"), "offline must not show tok/s");
    }

    #[test]
    fn renders_before_first_probe() {
        // dashboard == None (initial frame) must render honest "probing" text,
        // not a crash and not zeros-as-data.
        let text = render_dashboard(None);
        assert!(
            text.contains("Probing serve"),
            "expected probing placeholder"
        );
    }

    use crate::hipfire::dashboard::{Probe, SystemInfo};

    /// Render any tab and return flattened buffer text. Mutates `app` via the
    /// passed closure first (set tab / inject dashboard / toast). CPU-only, no
    /// TTY, no GPU, no serve.
    fn render_with(setup: impl FnOnce(&mut App)) -> String {
        let (mut app, _home) = crate::test_support::hermetic_app();
        setup(&mut app);
        let backend = TestBackend::new(120, 32);
        let mut terminal = Terminal::new(backend).expect("terminal");
        terminal
            .draw(|frame| draw(frame, &mut app))
            .expect("draw must not panic");
        let buf = terminal.backend().buffer().clone();
        buf.content().iter().map(|c| c.symbol()).collect()
    }

    fn dash_with_system(system: SystemInfo) -> Dashboard {
        let mut d = Dashboard::offline(
            "127.0.0.1:11435".into(),
            VramState::Unavailable(
                "GPU stats unavailable: neither amd-smi nor rocm-smi found (ROCm root: none)"
                    .into(),
            ),
            LoadState::Unavailable("GPU load unavailable: no GPU stats tool".into()),
        );
        d.system = Some(system);
        d
    }

    #[test]
    fn system_tab_renders_live_diagnostics() {
        let sys = SystemInfo {
            gpu_name: Probe::Value("Radeon RX 7900 XTX".into()),
            gpu_arch: Probe::Value("gfx1100".into()),
            hip_version: Probe::Value("HIP 6.2.41134".into()),
            kernel_cache: Probe::Value("/home/u/.hipfire_kernels (present, 12 entries)".into()),
            loaded_model: Probe::Value("/home/u/.hipfire/models/q.mq4".into()),
            model_checksum: Probe::Value("deadbeefdeadbeef (4.20 GB)".into()),
        };
        let text = render_with(|app| {
            app.tab = Tab::System;
            app.dashboard = Some(dash_with_system(sys.clone()));
        });
        assert!(text.contains("Live diagnostics"), "expected live header");
        assert!(text.contains("Radeon RX 7900 XTX"), "expected gpu name");
        assert!(text.contains("gfx1100"), "expected gpu arch");
        assert!(text.contains("HIP 6.2"), "expected HIP version");
        assert!(
            text.contains(".hipfire_kernels"),
            "expected kernel cache path"
        );
        assert!(text.contains("deadbeef"), "expected model checksum");
        // Must NOT show the old static placeholder.
        assert!(
            !text.contains("Diagnostics roadmap"),
            "System tab must be live, not the static placeholder"
        );
    }

    #[test]
    fn system_tab_renders_unavailable_states_honestly() {
        let sys = SystemInfo {
            gpu_name: Probe::Unavailable(
                "GPU stats unavailable: neither amd-smi nor rocm-smi found (ROCm root: none)"
                    .into(),
            ),
            gpu_arch: Probe::Unavailable("unavailable: rocminfo not found".into()),
            hip_version: Probe::Unavailable("unavailable: hipconfig not found".into()),
            kernel_cache: Probe::Unavailable(
                "/home/u/.hipfire_kernels (absent — populated on first run)".into(),
            ),
            loaded_model: Probe::Unavailable("no model loaded".into()),
            model_checksum: Probe::Unavailable("no model loaded".into()),
        };
        let text = render_with(|app| {
            app.tab = Tab::System;
            app.dashboard = Some(dash_with_system(sys));
        });
        assert!(text.contains("unavailable"), "expected honest unavailable");
        assert!(
            text.contains("no model loaded"),
            "expected idle model state"
        );
        // No fabricated gfx string when the probe failed.
        assert!(!text.contains("gfx1"), "must not invent a gfx target");
    }

    #[test]
    fn system_tab_before_first_probe_shows_probing() {
        let text = render_with(|app| {
            app.tab = Tab::System;
            app.dashboard = None;
        });
        assert!(
            text.contains("Probing GPU"),
            "expected probing placeholder before first snapshot"
        );
    }

    #[test]
    fn footer_hints_are_per_tab() {
        let settings = render_with(|app| app.tab = Tab::Settings);
        assert!(
            settings.contains("change"),
            "settings footer mentions change"
        );
        assert!(
            settings.contains("search"),
            "settings footer mentions search"
        );

        let models = render_with(|app| app.tab = Tab::Models);
        assert!(models.contains("select"), "models footer mentions select");
        assert!(models.contains("expand"), "models footer mentions expand");

        let dash = render_with(|app| app.tab = Tab::Dashboard);
        assert!(
            dash.contains("refresh"),
            "dashboard footer mentions refresh"
        );

        let chat = render_with(|app| app.tab = Tab::Chat);
        assert!(chat.contains("send"), "chat footer mentions send");

        let system = render_with(|app| app.tab = Tab::System);
        assert!(
            system.contains("diagnostics"),
            "system footer mentions diagnostics"
        );
        // Global hints present on a non-chat tab.
        assert!(dash.contains("quit"), "global quit hint present");
    }

    #[test]
    fn error_toast_overrides_footer() {
        let text = render_with(|app| {
            app.tab = Tab::Settings;
            app.toast_error("save failed: permission denied");
        });
        assert!(
            text.contains("save failed: permission denied"),
            "expected error toast text in footer"
        );
    }

    #[test]
    fn models_empty_state_guides_to_pull() {
        // A hermetic app with no registry / no local models yields empty visible
        // items; the Models tab must render pull guidance, not an empty table.
        let text = render_with(|app| {
            app.tab = Tab::Models;
            app.registry.models.clear();
            app.registry.local_files.clear();
        });
        assert!(
            text.contains("No models found"),
            "expected empty-state header"
        );
        assert!(text.contains("hipfire pull"), "expected pull guidance");
    }

    #[test]
    fn help_overlay_renders_keybindings() {
        let text = render_with(|app| app.show_help = true);
        assert!(text.contains("Keybindings"), "help overlay title");
        assert!(text.contains("switch tabs"), "global keys listed");
        assert!(text.contains("toggle this help"), "the ? key is documented");
    }

    #[test]
    fn header_records_ordered_tab_hitboxes() {
        let (mut app, _home) = crate::test_support::hermetic_app();
        let backend = TestBackend::new(110, 30);
        let mut terminal = Terminal::new(backend).expect("terminal");
        terminal
            .draw(|frame| draw(frame, &mut app))
            .expect("draw must not panic");
        assert_eq!(app.tab_hitboxes.len(), Tab::ALL.len());
        for (i, (start, end, tab)) in app.tab_hitboxes.iter().enumerate() {
            assert!(end > start, "tab {tab:?} has an empty hit region");
            assert_eq!(*tab, Tab::ALL[i], "hit regions ordered like Tab::ALL");
        }
        // A click inside the first tab's region resolves to Home.
        let (start, _, _) = app.tab_hitboxes[0];
        assert_eq!(app.tab_at(start, app.tab_row_y), Some(Tab::Home));
    }

    #[test]
    fn tab_hitboxes_align_with_rendered_columns() {
        // Cross-check each recorded hit region against where ratatui ACTUALLY
        // drew the tab title in the buffer — the only check that catches a real
        // off-by-one in draw_header's cell-width / divider math (the fabricated-
        // hitbox click tests cannot).
        let (mut app, _home) = crate::test_support::hermetic_app();
        let backend = TestBackend::new(110, 30);
        let mut terminal = Terminal::new(backend).expect("terminal");
        terminal
            .draw(|frame| draw(frame, &mut app))
            .expect("draw must not panic");
        let buf = terminal.backend().buffer().clone();
        let width = buf.area().width as usize;
        let row = app.tab_row_y as usize;
        let content = buf.content();
        let row_text: String = (0..width)
            .map(|c| content[row * width + c].symbol())
            .collect();
        for (start, _end, tab) in &app.tab_hitboxes {
            let title = tab.title();
            let title_col = (*start as usize) + 1; // title is drawn after the 1-col left pad
            let drawn: String = row_text
                .chars()
                .skip(title_col)
                .take(title.chars().count())
                .collect();
            assert_eq!(
                drawn, title,
                "tab {tab:?}: title must render at hitbox.start+1 (col {title_col})"
            );
        }
    }

    #[test]
    fn narrow_terminal_drops_overflowing_tabs() {
        // At a width where not all tabs fit, the hit regions must not register a
        // tab ratatui didn't draw, every region is clamped to the area, and a
        // click well past the visible tabs resolves to nothing.
        let (mut app, _home) = crate::test_support::hermetic_app();
        let backend = TestBackend::new(20, 12);
        let mut terminal = Terminal::new(backend).expect("terminal");
        terminal
            .draw(|frame| draw(frame, &mut app))
            .expect("draw must not panic");
        assert!(
            app.tab_hitboxes.len() < Tab::ALL.len(),
            "not all tabs fit at width 20"
        );
        for (_, end, _) in &app.tab_hitboxes {
            assert!(*end <= 20, "hit region clamped to the area width");
        }
        assert_eq!(
            app.tab_at(100, app.tab_row_y),
            None,
            "click far past the tabs"
        );
    }

    #[test]
    fn help_overlay_does_not_panic_at_tiny_geometry() {
        // Locks in the saturating-sub underflow guards in draw_help_overlay.
        for (w, h) in [(4u16, 3u16), (2, 2), (1, 1)] {
            let (mut app, _home) = crate::test_support::hermetic_app();
            app.show_help = true;
            let backend = TestBackend::new(w, h);
            let mut terminal = Terminal::new(backend).expect("terminal");
            terminal
                .draw(|frame| draw(frame, &mut app))
                .expect("draw must not panic at tiny geometry");
        }
    }

    #[test]
    fn dashboard_footer_shows_serve_controls() {
        let text = render_with(|app| app.tab = Tab::Dashboard);
        assert!(text.contains("s start"), "serve start control in footer");
        assert!(text.contains("x stop"), "serve stop control in footer");
        assert!(
            text.contains("R restart"),
            "serve restart control in footer"
        );
    }

    #[test]
    fn models_confirm_delete_prompt_renders() {
        let text = render_with(|app| {
            app.tab = Tab::Models;
            app.confirm_delete = Some("qwen3.5:9b".into());
        });
        assert!(text.contains("Confirm delete"), "confirm prompt title");
        assert!(text.contains("y to confirm"), "y/n guidance shown");
    }

    #[test]
    fn settings_reset_all_confirm_prompt_renders() {
        // 5a: the armed reset-all confirm shows the destructive prompt + y/n
        // guidance and names the file it clears.
        let text = render_with(|app| {
            app.tab = Tab::Settings;
            app.confirm_reset_all = true;
        });
        assert!(text.contains("Confirm reset-all"), "confirm prompt title");
        assert!(text.contains("config.toml"), "names the file it clears");
        assert!(text.contains("y = confirm"), "y/n guidance shown");
    }

    #[test]
    fn settings_footer_shows_reset_controls() {
        // 5a: reset + reset-all are discoverable in the Settings footer.
        let text = render_with(|app| app.tab = Tab::Settings);
        assert!(text.contains("Del default"), "single-key reset hint");
        assert!(text.contains("R reset-all"), "reset-all hint");
    }

    #[test]
    fn models_pull_progress_gauge_renders() {
        let text = render_with(|app| {
            app.tab = Tab::Models;
            let (_tx, rx) = std::sync::mpsc::channel();
            app.inject_pull("qwen3.5:9b".into(), rx);
            let job = app.pull.as_mut().unwrap();
            job.percent = Some(37.0);
            job.line = "[bar] 37.0% 8 MB/s".into();
        });
        assert!(text.contains("Pulling"), "gauge title shows the model");
        assert!(text.contains("37"), "percent in the gauge label");
    }

    #[test]
    fn models_footer_offers_cancel_while_pulling() {
        let text = render_with(|app| {
            app.tab = Tab::Models;
            let (_tx, rx) = std::sync::mpsc::channel();
            app.inject_pull("qwen3.5:9b".into(), rx);
        });
        assert!(text.contains("c cancel"), "footer/gauge advertise c cancel");
    }

    #[test]
    fn quit_confirm_prompt_is_visible_on_every_tab() {
        for tab in [Tab::Home, Tab::Chat, Tab::Models, Tab::Logs] {
            let text = render_with(|app| {
                app.tab = tab;
                let (_tx, rx) = std::sync::mpsc::channel();
                app.inject_pull("qwen3.5:9b".into(), rx);
                app.toast_info("a toast that must not hide the prompt");
                assert!(!app.request_quit());
            });
            assert!(
                text.contains("A pull of qwen3.5:9b is running. Quit and cancel it?"),
                "prompt missing on {tab:?}"
            );
        }
    }

    #[test]
    fn logs_tab_missing_state_is_honest() {
        use crate::hipfire::log_tail::{LogSnapshot, LogStatus};
        let text = render_with(|app| {
            app.tab = Tab::Logs;
            app.logs = LogSnapshot {
                lines: vec![],
                status: LogStatus::Missing,
            };
        });
        assert!(
            text.contains("No serve.log yet"),
            "honest missing-state guidance"
        );
    }

    #[test]
    fn system_tab_shows_request_inspector() {
        let text = render_with(|app| {
            app.tab = Tab::System;
            app.request_log = vec![crate::app::RequestRecord {
                model: "qwen3.5:9b".into(),
                tokens: 128,
                tps: 151.0,
                secs: 0.85,
                exact: true,
            }];
        });
        assert!(
            text.contains("Recent requests"),
            "inspector section present"
        );
        assert!(text.contains("128 tok"), "token count shown");
    }

    #[test]
    fn system_tab_renders_doctor_report() {
        use crate::hipfire::doctor::{DoctorCheck, DoctorReport};
        let text = render_with(|app| {
            app.tab = Tab::System;
            app.doctor = Some(DoctorReport {
                checks: vec![
                    DoctorCheck {
                        name: "amdgpu module".into(),
                        ok: true,
                        detail: "loaded".into(),
                    },
                    DoctorCheck {
                        name: "/dev/kfd".into(),
                        ok: false,
                        detail: "missing".into(),
                    },
                ],
                error: None,
            });
        });
        assert!(text.contains("Doctor (hipfire diag)"), "doctor section");
        assert!(text.contains("amdgpu module"), "check name shown");
        assert!(text.contains("missing"), "failed-check detail shown");
    }

    #[test]
    fn chat_renders_fenced_code_block() {
        let text = render_with(|app| {
            app.tab = Tab::Chat;
            app.chat.messages = vec![crate::hipfire::chat::ChatMessage {
                role: "assistant".into(),
                content: "here:\n```rust\nlet x = 1;\n```".into(),
                reasoning_content: None,
            }];
        });
        assert!(text.contains("let x = 1;"), "code line is rendered");
        assert!(text.contains("rust"), "fence language label is shown");
    }

    #[test]
    fn logs_tab_renders_tail_lines() {
        use crate::hipfire::log_tail::{LogSnapshot, LogStatus};
        let text = render_with(|app| {
            app.tab = Tab::Logs;
            app.logs = LogSnapshot {
                lines: vec!["serve started on :11435".into(), "loaded qwen3.5:9b".into()],
                status: LogStatus::Ok,
            };
        });
        assert!(text.contains("loaded qwen3.5:9b"), "tail line is shown");
        assert!(
            text.contains("last 2 lines"),
            "title reflects the line count"
        );
    }
    /// Render at `w`x`h` and return each buffer row as a string.
    fn render_rows(w: u16, h: u16, setup: impl FnOnce(&mut App)) -> Vec<String> {
        let (mut app, _home) = crate::test_support::hermetic_app();
        setup(&mut app);
        let mut terminal = Terminal::new(TestBackend::new(w, h)).expect("terminal");
        terminal
            .draw(|frame| draw(frame, &mut app))
            .expect("draw must not panic");
        let buf = terminal.backend().buffer().clone();
        (0..h)
            .map(|y| {
                (0..w)
                    .map(|x| buf[(x, y)].symbol().to_string())
                    .collect::<String>()
            })
            .collect()
    }

    /// The footer's first row (the hints row) of a render.
    fn footer_row(rows: &[String]) -> String {
        rows[rows.len() - 2].trim_end().to_string()
    }

    #[test]
    fn footers_fit_whole_segments_and_advertise_help_at_80x24() {
        let mut checked = 0;
        for tab in Tab::ALL {
            let mut segs = Vec::new();
            let rows = render_rows(80, 24, |app| {
                app.tab = tab;
                app.last_reload.clear();
                segs = footer_segments(app);
            });
            let row = footer_row(&rows);
            assert!(
                row.contains("? help"),
                "{tab:?} footer advertises help: {row:?}"
            );
            assert!(
                row.chars().count() <= 80,
                "{tab:?} footer fits the width: {row:?}"
            );
            for part in row.split(FOOTER_SEP) {
                assert!(
                    segs.iter().any(|s| s == part),
                    "{tab:?} footer piece {part:?} is not a whole segment of {segs:?}"
                );
            }
            checked += 1;
        }
        assert_eq!(checked, Tab::ALL.len());
    }

    #[test]
    fn chat_footer_variant_advertises_help_via_esc() {
        let rows = render_rows(80, 24, |app| {
            app.tab = Tab::Chat;
            app.chat.focus_input();
            app.last_reload.clear();
        });
        assert!(footer_row(&rows).contains("Esc then ? help"));
        let rows = render_rows(80, 24, |app| {
            app.tab = Tab::Chat;
            app.chat.blur_input();
            app.last_reload.clear();
        });
        assert!(footer_row(&rows).contains("? help"));
    }

    #[test]
    fn footer_status_text_only_appears_when_it_fits() {
        let status = "STATUS-MARKER reloaded config";
        let wide = render_rows(160, 48, |app| {
            app.tab = Tab::Dashboard;
            app.last_reload = status.into();
        });
        let row = footer_row(&wide);
        let help = row.find("? help").expect("hints present");
        let at = row.find(status).expect("status renders when room remains");
        assert!(at > help, "status comes after the hints: {row:?}");

        let narrow = render_rows(80, 24, |app| {
            app.tab = Tab::Dashboard;
            app.last_reload = status.into();
        });
        assert!(
            !footer_row(&narrow).contains("STATUS-MARKER"),
            "status is dropped, never cut, when the hints use the width"
        );
    }

    #[test]
    fn segments_are_dropped_not_cut() {
        let segs: Vec<String> = ["alpha one", "beta", "gamma three"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(fit_segments(&segs, 80), "alpha one · beta · gamma three");
        assert_eq!(fit_segments(&segs, 14), "alpha one");
        assert_eq!(fit_segments(&segs, 18), "alpha one · beta");
        // A later, shorter segment still fits after a long one is skipped.
        let mixed: Vec<String> = ["alpha one", "a long segment here", "b"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(fit_segments(&mixed, 16), "alpha one · b");
        assert_eq!(fit_segments(&segs, 0), "");
    }

    #[test]
    fn serve_confirm_prompts_render_in_footer() {
        use crate::hipfire::serve_ctrl::ServeAction;
        let rows = render_rows(80, 24, |app| {
            app.tab = Tab::Dashboard;
            app.confirm_serve = Some(ServeAction::Stop);
        });
        assert!(footer_row(&rows).contains("Stop hipfire serve? y / n"));
        let rows = render_rows(80, 24, |app| {
            app.tab = Tab::Dashboard;
            app.confirm_serve = Some(ServeAction::Restart);
        });
        let both = format!("{} {}", rows[22], rows[23]);
        assert!(both.contains("hipfire stop --force"), "{both:?}");
        assert!(both.contains("force-reaps"), "{both:?}");
        assert!(both.contains("y / n"), "{both:?}");
    }

    #[test]
    fn home_lists_logs_and_shows_runtime_models_and_registry_lines() {
        for (w, h) in [(80u16, 24u16), (160, 48)] {
            let rows = render_rows(w, h, |app| {
                app.tab = Tab::Home;
                app.registry.warning = Some("registry note".into());
            });
            let text = rows.join("\n");
            assert!(text.contains("Logs"), "{w}x{h}: Home tab list names Logs");
            assert!(
                text.contains("Models     "),
                "{w}x{h}: Runtime Models line visible"
            );
            assert!(
                text.contains("Registry   "),
                "{w}x{h}: Runtime Registry line visible"
            );
            assert!(
                text.contains("? help") || text.contains("? help for"),
                "{w}x{h}: Home mentions ? help"
            );
        }
    }

    #[test]
    fn help_overlay_documents_esc_confirms_and_help_key() {
        let text = render_rows(80, 30, |app| {
            app.tab = Tab::Dashboard;
            app.show_help = true;
        })
        .join("\n");
        assert!(text.contains("close / cancel"), "Esc is close / cancel");
        assert!(!text.contains("Esc              quit"));
        assert!(text.contains("ask y / n"), "x / R confirms are documented");
        assert!(text.contains("this help"), "? is documented");
    }

    fn chat_key(app: &mut App, code: crossterm::event::KeyCode) {
        app.handle_tab_key(crossterm::event::KeyEvent::new(
            code,
            crossterm::event::KeyModifiers::NONE,
        ));
    }

    /// Draw `app` at `w`×`h`; return the flattened buffer text and the terminal
    /// cursor position set by the frame.
    fn draw_at(app: &mut App, w: u16, h: u16) -> (String, ratatui::layout::Position) {
        let mut terminal = Terminal::new(TestBackend::new(w, h)).expect("terminal");
        terminal
            .draw(|frame| draw(frame, app))
            .expect("draw must not panic");
        let cursor = terminal.get_cursor_position().expect("cursor");
        let buf = terminal.backend().buffer().clone();
        (buf.content().iter().map(|c| c.symbol()).collect(), cursor)
    }

    fn msg(role: &str, content: &str) -> crate::hipfire::chat::ChatMessage {
        crate::hipfire::chat::ChatMessage {
            role: role.into(),
            content: content.into(),
            reasoning_content: None,
        }
    }

    fn long_reply_app() -> App {
        let (mut app, _home) = crate::test_support::hermetic_app();
        app.tab = Tab::Chat;
        let mut body: Vec<String> = (1..=40).map(|i| format!("line {i:02} xxxx")).collect();
        body.push("END-OF-REPLY".into());
        app.chat.messages = vec![msg("user", "go"), msg("assistant", &body.join("\n"))];
        app
    }

    fn visible_lines(text: &str) -> Vec<usize> {
        (1..=40)
            .filter(|i| text.contains(&format!("line {i:02} x")))
            .collect()
    }

    fn chat_follows_and_scrolls_from_bottom(w: u16, h: u16) {
        let mut app = long_reply_app();
        let (text, _) = draw_at(&mut app, w, h);
        assert!(
            text.contains("END-OF-REPLY"),
            "{w}x{h}: newest output visible with no key presses"
        );
        assert!(
            !text.contains("more (End to follow)"),
            "no hint while following"
        );
        let before = visible_lines(&text);

        for _ in 0..5 {
            chat_key(&mut app, crossterm::event::KeyCode::Up);
        }
        assert_eq!(app.chat.scroll, 5, "Up scrolls toward older text");
        let (scrolled, _) = draw_at(&mut app, w, h);
        let after = visible_lines(&scrolled);
        assert!(
            after.first() < before.first(),
            "{w}x{h}: an older line is visible after Up (before {before:?}, after {after:?})"
        );
        assert!(
            !scrolled.contains("END-OF-REPLY"),
            "{w}x{h}: end of reply scrolled out of view"
        );
        assert!(scrolled.contains("5 more (End to follow)"), "hint shown");

        chat_key(&mut app, crossterm::event::KeyCode::Down);
        assert_eq!(app.chat.scroll, 4, "Down scrolls toward newer text");

        chat_key(&mut app, crossterm::event::KeyCode::End);
        assert_eq!(app.chat.scroll, 0);
        let (followed, _) = draw_at(&mut app, w, h);
        assert!(followed.contains("END-OF-REPLY"), "End resumes following");
    }

    #[test]
    fn chat_scrolls_from_bottom_80x24() {
        chat_follows_and_scrolls_from_bottom(80, 24);
    }

    #[test]
    fn chat_scrolls_from_bottom_160x48() {
        chat_follows_and_scrolls_from_bottom(160, 48);
    }

    #[test]
    fn chat_scroll_clamps_pages_and_jumps_home() {
        let mut app = long_reply_app();
        let (_, _) = draw_at(&mut app, 80, 24);
        let view = app.chat.view.get().expect("view recorded by draw");
        assert!(view.max_scroll > 0 && view.viewport > 0);

        for _ in 0..500 {
            chat_key(&mut app, crossterm::event::KeyCode::Up);
        }
        assert_eq!(app.chat.scroll, view.max_scroll, "Up clamps at the top");
        let (top, _) = draw_at(&mut app, 80, 24);
        assert!(top.contains("line 01 x"), "oldest text visible at the top");

        chat_key(&mut app, crossterm::event::KeyCode::PageDown);
        assert_eq!(
            app.chat.scroll,
            view.max_scroll - view.viewport.min(view.max_scroll)
        );
        chat_key(&mut app, crossterm::event::KeyCode::End);
        chat_key(&mut app, crossterm::event::KeyCode::PageUp);
        assert_eq!(app.chat.scroll, view.viewport.min(view.max_scroll));
        chat_key(&mut app, crossterm::event::KeyCode::Home);
        assert_eq!(app.chat.scroll, view.max_scroll);
    }

    #[test]
    fn chat_long_input_tail_and_cursor_visible() {
        let (mut app, _home) = crate::test_support::hermetic_app();
        app.tab = Tab::Chat;
        app.chat.input = format!("{}END", "x".repeat(400));
        let (text, cursor) = draw_at(&mut app, 80, 24);
        assert!(text.contains("END"), "tail of a long input is visible");
        // Input box: rows h-7..h-3 (footer takes the last 2), inner rows h-6..=h-4.
        assert!(
            (18..=20).contains(&cursor.y),
            "cursor row {} inside the input box",
            cursor.y
        );
        assert!(cursor.x >= 2 && cursor.x < 79, "cursor col {}", cursor.x);
    }

    #[test]
    fn wrap_input_rows_places_cursor_row() {
        assert_eq!(wrap_input_rows("abcdefgh", 4), vec!["abcd", "efgh", ""]);
        assert_eq!(wrap_input_rows("ab\n", 4), vec!["ab", ""]);
        assert_eq!(wrap_input_rows("", 4), vec![""]);
    }

    #[test]
    fn chat_help_goes_to_transcript_not_input_title() {
        let (mut app, _home) = crate::test_support::hermetic_app();
        app.tab = Tab::Chat;
        app.chat.messages = vec![msg("user", "hi"), msg("assistant", "hello")];
        app.chat.input = "/help".into();
        chat_key(&mut app, crossterm::event::KeyCode::Enter);
        assert_eq!(app.chat.messages.len(), 2, "help is not a message");
        assert!(app.chat.input.is_empty());
        assert!(
            !app.chat.status.contains("/model"),
            "status/input title carries no command list"
        );
        let (text, _) = draw_at(&mut app, 80, 24);
        assert!(
            text.contains("/regen"),
            "help text rendered in the transcript"
        );
        assert!(text.contains("Commands"));
        assert!(
            !text.contains("/model · /system"),
            "old one-line title is gone"
        );
    }

    #[test]
    fn chat_reasoning_is_labelled_and_styled_apart() {
        let (mut app, _home) = crate::test_support::hermetic_app();
        app.tab = Tab::Chat;
        app.chat.messages = vec![
            msg("user", "hi"),
            crate::hipfire::chat::ChatMessage {
                role: "assistant".into(),
                content: "the answer".into(),
                reasoning_content: Some("pondering".into()),
            },
        ];
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).expect("terminal");
        terminal.draw(|frame| draw(frame, &mut app)).expect("draw");
        let buf = terminal.backend().buffer().clone();
        let text: String = buf.content().iter().map(|c| c.symbol()).collect();
        assert!(text.contains("thinking"), "reasoning label shown");
        let find = |needle: &str| {
            buf.content()
                .iter()
                .enumerate()
                .find(|(i, _)| {
                    let start = *i;
                    let w = needle.chars().count();
                    start + w <= buf.content().len()
                        && buf.content()[start..start + w]
                            .iter()
                            .map(|c| c.symbol())
                            .collect::<String>()
                            == needle
                })
                .map(|(_, c)| c.style())
                .expect("needle on screen")
        };
        let think = find("pondering");
        let answer = find("the answer");
        assert!(think.add_modifier.contains(Modifier::ITALIC));
        assert!(!answer.add_modifier.contains(Modifier::ITALIC));
        assert_ne!(think.fg, answer.fg, "reasoning uses the muted colour");
    }

    #[test]
    fn chat_input_title_shows_prefill_progress_while_sending() {
        let (mut app, _home) = crate::test_support::hermetic_app();
        app.tab = Tab::Chat;
        app.chat.sending = true;
        app.chat.status = "prefilling 512/2048 (25%) · ETA 6s".into();
        let (text, _) = draw_at(&mut app, 100, 24);
        assert!(
            text.contains("prefilling 512/2048 (25%) · ETA 6s"),
            "prefill readout in the chat view: {text}"
        );
    }

    #[test]
    fn chat_abort_marks_stopped_in_transcript_only() {
        let (mut app, _home) = crate::test_support::hermetic_app();
        app.tab = Tab::Chat;
        app.chat.messages = vec![msg("user", "hi"), msg("assistant", "partial reply")];
        app.chat.sending = true;
        app.chat.request_abort();
        assert!(!app.chat.sending);
        assert_eq!(app.chat.messages[1].content, "partial reply");
        assert!(!app.chat.messages[1].content.contains("[stopped]"));
        let (text, _) = draw_at(&mut app, 80, 24);
        assert!(text.contains("[stopped]"), "stopped marker rendered");
    }

    #[test]
    fn chat_stats_label_is_honest_about_chunks() {
        let (mut app, _home) = crate::test_support::hermetic_app();
        app.tab = Tab::Chat;
        app.chat.last_stats = Some(crate::app::GenStats {
            tokens: 7,
            tps: 3.0,
            exact: false,
        });
        let (text, _) = draw_at(&mut app, 160, 24);
        assert!(text.contains("7 chunks @ 3 chunks/s"));
        assert!(!text.contains("7 tok"));
        app.chat.last_stats = Some(crate::app::GenStats {
            tokens: 5,
            tps: 2.0,
            exact: true,
        });
        let (text, _) = draw_at(&mut app, 160, 24);
        assert!(text.contains("5 tok @ 2 tok/s"));
    }

    /// Render the Models tab at `width`x`height` over a fixed registry: two
    /// all-remote groups (`0/2 local`), and an expanded `qwen3.8` group holding
    /// a long tag that is the active (local) model.
    fn render_models(width: u16, height: u16) -> String {
        use crate::hipfire::registry::{LocalModel, ModelEntry, ModelRow};
        let (mut app, _home) = crate::test_support::hermetic_app();
        app.tab = Tab::Models;
        let row = |tag: &str, downloaded: bool, has_mtp: bool| ModelRow {
            tag: tag.into(),
            entry: ModelEntry {
                repo: "org/repo".into(),
                size_gb: 1.5,
                min_vram_gb: 8.0,
                desc: "test row".into(),
                ..ModelEntry::default()
            },
            downloaded,
            has_mtp,
        };
        app.active_model = "qwen3.8:local-fixture-long-tag".into();
        app.registry.models = vec![
            row("alpha:a", false, false),
            row("alpha:b", false, false),
            row("beta:a", false, false),
            row("beta:b", false, false),
            row("qwen3.8:local-fixture-long-tag", true, true),
            row("qwen3.8:flash-next-mq3-xts", false, false),
        ];
        // The `.ple-q4` row file is on disk but is a sidecar, never a Models row.
        app.registry.local_files = vec![LocalModel {
            file: "qwen3.8.mq4.ple-q4".into(),
            size: "29.1 GiB".into(),
            bytes: 31_243_000_000,
        }];
        app.registry.expanded_groups = ["qwen3.8".to_string()].into_iter().collect();
        app.registry.selected = 0;
        app.registry.loaded_path = None;
        app.registry.bundled = true;
        app.registry.generated_at = "2026-10-01T00:00:00Z".into();
        let mut terminal = Terminal::new(TestBackend::new(width, height)).expect("terminal");
        terminal
            .draw(|frame| draw(frame, &mut app))
            .expect("draw must not panic");
        let buf = terminal.backend().buffer().clone();
        buf.content().iter().map(|c| c.symbol()).collect()
    }

    #[test]
    fn models_tab_groups_and_header_are_fully_visible_at_80_and_160() {
        for (w, h) in [(80, 24), (160, 48)] {
            let text = render_models(w, h);
            assert_eq!(
                text.matches("0/2 local").count(),
                2,
                "{w}x{h}: both all-remote groups show their full `0/2 local` count"
            );
            assert!(text.contains("1/2 local"), "{w}x{h}: expanded group count");
            assert!(!text.contains(".ple-q4"), "{w}x{h}: no .ple-q4 rows");
            assert!(
                !text.contains("triattn"),
                "{w}x{h}: no triattn sidecar display"
            );
            assert!(
                !text.contains("local-only"),
                "{w}x{h}: Have is local/remote only"
            );
            assert_eq!(
                text.matches("\u{2192} expand").count(),
                1,
                "{w}x{h}: only the selected group carries the action hint"
            );
        }
    }

    #[test]
    fn models_tab_wide_screen_shows_long_tag_and_separate_active_marker() {
        let text = render_models(160, 48);
        assert!(
            text.contains("* qwen3.8:local-fixture-long-tag"),
            "the long tag is fully visible with the active marker beside it"
        );
        assert!(
            text.contains("qwen3.8:flash-next-mq3-xts"),
            "the second long tag is visible"
        );
        assert!(
            text.contains("registry: bundled (2026-10-01)"),
            "a bundled registry is named with its date, not `missing`"
        );
        assert!(text.contains("remote") && text.contains("local"));
    }

    #[test]
    fn pull_chooser_renders_ple_and_draft_rows_at_80x24() {
        use crate::hipfire::pull_choice::PullChoice;
        use crate::hipfire::registry::ModelEntry;
        use hipfire_registry::Sidecar;
        let sc = |size| Sidecar {
            file: "x".into(),
            sha256: None,
            size_bytes: Some(size),
        };
        let mut entry = ModelEntry::default();
        entry.ple.insert("q4".into(), sc(1_500_000_000));
        entry.ple.insert("q8".into(), sc(3_000_000_000));
        entry.mtp = Some(sc(400_000_000));
        entry.dflash = Some(sc(1_200_000_000));
        let (mut app, _home) = crate::test_support::hermetic_app();
        app.tab = Tab::Models;
        app.pull_choice = PullChoice::for_entry("qwen3.8:flash-next-mq4-xts", &entry);
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).expect("terminal");
        terminal
            .draw(|frame| draw(frame, &mut app))
            .expect("draw must not panic");
        let buf = terminal.backend().buffer().clone();
        let text: String = buf.content().iter().map(|c| c.symbol()).collect();
        for needle in [
            "PLE rows",
            "auto",
            "q4 1.5 GB",
            "q8 3.0 GB",
            "Draft sidecars",
            "yes",
            "MTP 400 MB",
            "Enter pull",
            "Esc cancel",
        ] {
            assert!(
                text.contains(needle),
                "missing {needle:?} in:\n{}",
                text.as_bytes()
                    .chunks(80)
                    .map(|c| String::from_utf8_lossy(c).into_owned())
                    .collect::<Vec<_>>()
                    .join("\n")
            );
        }
        let segs = footer_segments(&app);
        assert_eq!(
            segs,
            [
                "Up/Down row",
                "Left/Right change",
                "Enter pull",
                "Esc cancel"
            ]
        );
    }
}
