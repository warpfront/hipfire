// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

mod app;
mod hipfire;
mod ui;
mod ui_chat;
mod ui_scope;
mod ui_settings;
mod ui_tuning;

#[cfg(test)]
mod test_support;

use std::{io, panic};

use anyhow::Result;
use app::App;
use base64::Engine as _;
use crossterm::{
    event::{
        self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEvent, KeyModifiers,
        MouseButton, MouseEvent, MouseEventKind,
    },
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};

const VERSION: &str = env!("CARGO_PKG_VERSION");

fn main() -> Result<()> {
    // Non-interactive argument handling (headless-safe, no TTY required).
    // Only the first argument is inspected — every branch returns/exits, so
    // there is never a second iteration.
    let mut start_settings = false;
    let mut start_tuning = false;
    let profile_wizard = if let Some(arg) = std::env::args().nth(1) {
        match arg.as_str() {
            "--version" | "-V" => {
                println!("hipfire-tui {VERSION}");
                return Ok(());
            }
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            "--check" => {
                return run_check();
            }
            "--config-profile-wizard" => true,
            // `hipfire config` launches with this: open on Settings for this
            // launch only (see `tab_to_persist`).
            "--settings" => {
                start_settings = true;
                false
            }
            // `hipfire config tuning` launches with this: Settings with the
            // Tuning preset chooser open.
            "--tuning" => {
                start_settings = true;
                start_tuning = true;
                false
            }
            other => {
                eprintln!("hipfire-tui: unknown argument '{other}'");
                print_help();
                std::process::exit(2);
            }
        }
    } else {
        false
    };

    let mut terminal = setup_terminal()?;
    let result = if profile_wizard {
        hipfire::profile_wizard::run(&mut terminal)
    } else {
        run(&mut terminal, start_settings, start_tuning)
    };
    restore_terminal(&mut terminal)?;
    result
}
fn print_help() {
    println!("{}", help_text());
}

fn help_text() -> String {
    format!(
        "hipfire-tui {VERSION} - terminal UI for hipfire\n\
         \n\
         USAGE:\n    \
             hipfire-tui [FLAGS]\n\
         \n\
         FLAGS:\n    \
             -h, --help               Print this help and exit\n    \
             -V, --version            Print version and exit\n    \
                 --check              Load config/registry/models without entering the\n                         \
                                      render loop, then exit 0 on success (headless smoke)\n    \
                 --config-profile-wizard\n                         \
                                      Select/create profiles and browse config variables\n    \
                 --settings           Open on the Settings tab for this launch (what\n                         \
                                      `hipfire config` runs); the saved tab is kept\n    \
                 --tuning             Open on Settings with the Tuning preset chooser\n                         \
                                      (what `hipfire config tuning` runs)\n\
         \n\
         With no flags, hipfire-tui launches the interactive ratatui UI (requires a TTY).\n\
         Tabs: Home, Dashboard, Chat, Models, Settings, System, Logs. Tab/Shift+Tab to switch, ? for help, q to quit.\n\
         Esc never quits: it only closes/cancels (help, prompts, edits, chat)."
    )
}

/// Construct the App state (config + registry + local models) WITHOUT entering
/// the ratatui render/event loop. Exits 0 on success, non-zero on init failure.
/// This is the headless smoke path for CI and TTY-less environments.
fn run_check() -> Result<()> {
    let app = App::load()?;
    println!("hipfire-tui --check OK");
    println!("  config: default_model = {:?}", app.config.default_model);
    println!(
        "  registry: {} models, {} aliases",
        app.registry.models.len(),
        app.registry.aliases.len()
    );
    println!("  local models: {}", app.registry.local_files.len());
    if let Some(warn) = &app.registry.warning {
        println!("  registry warning: {warn}");
    }
    Ok(())
}

fn setup_terminal() -> Result<Terminal<CrosstermBackend<io::Stdout>>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;

    let hook = panic::take_hook();
    panic::set_hook(Box::new(move |info| {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen, DisableMouseCapture);
        hook(info);
    }));

    let backend = CrosstermBackend::new(stdout);
    match Terminal::new(backend) {
        Ok(t) => Ok(t),
        Err(e) => {
            // We enabled raw mode + alt screen + mouse capture above; undo them
            // (best-effort) before surfacing the failure so the terminal is left
            // usable.
            let _ = execute!(io::stdout(), LeaveAlternateScreen, DisableMouseCapture);
            let _ = disable_raw_mode();
            Err(e.into())
        }
    }
}

fn restore_terminal(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) -> Result<()> {
    // Attempt every step independently so an early failure can't leave mouse
    // capture, the alternate screen, or raw mode enabled. Return the first
    // error after all steps have been tried.
    let leave = execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    );
    let cursor = terminal.show_cursor();
    let raw = disable_raw_mode();
    leave?;
    cursor?;
    raw?;
    Ok(())
}

/// The tab to write to ui_state.json on exit. A `--settings` launch overrides
/// the restored tab for that launch only: if the user is still on the forced
/// Settings tab, the tab ui_state restored (`launch_saved`) stays persisted;
/// once they navigate elsewhere, wherever they ended up is saved as usual.
fn tab_to_persist(current: app::Tab, launch_saved: Option<app::Tab>) -> app::Tab {
    match launch_saved {
        Some(saved) if current == app::Tab::Settings => saved,
        _ => current,
    }
}

/// Apply `--settings`: switch to the Settings tab and return the tab ui_state
/// restored, so [`tab_to_persist`] can keep it. `None` when not overridden.
fn apply_launch_tab(app: &mut App, start_settings: bool) -> Option<app::Tab> {
    start_settings.then(|| std::mem::replace(&mut app.tab, app::Tab::Settings))
}

fn run(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    start_settings: bool,
    start_tuning: bool,
) -> Result<()> {
    let mut app = App::load()?;
    let launch_saved = apply_launch_tab(&mut app, start_settings);
    if start_tuning {
        app.open_tuning();
    }
    let result = event_loop(terminal, &mut app);
    app.tab = tab_to_persist(app.tab, launch_saved);
    // Persist session UI state (active tab + expanded Models groups) on ANY exit
    // — clean quit OR a propagated error — so the next launch reopens where the
    // user left off. Best-effort (the save swallows its own errors); a panic
    // still skips this, which is acceptable for non-load-bearing UI state.
    app.save_ui_state();
    result
}

/// Copy `text` to the terminal's clipboard via OSC 52. Supported by many
/// terminals (kitty, iTerm2, wezterm, tmux with `set-clipboard on`); a harmless
/// no-op where unsupported — the escape is consumed without altering the screen.
fn emit_clipboard(text: &str) {
    use std::io::Write;
    let seq = osc52_sequence(text);
    let mut out = io::stdout();
    let _ = out.write_all(seq.as_bytes());
    let _ = out.flush();
}

/// The OSC 52 set-clipboard escape frame for `text` (extracted for testing).
fn osc52_sequence(text: &str) -> String {
    use base64::engine::general_purpose::STANDARD;
    format!("\x1b]52;c;{}\x07", STANDARD.encode(text.as_bytes()))
}

fn event_loop(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>, app: &mut App) -> Result<()> {
    loop {
        // Live-serve Dashboard: the fetch (HTTP + rocm-smi) runs on a dedicated
        // background thread. Here on the UI thread we ONLY mirror its latest
        // snapshot (a cheap lock+clone) and tell it whether the Dashboard tab
        // is focused — never any synchronous network / rocm-smi call, so a hung
        // probe cannot block render or input.
        app.sync_dashboard();
        app.sync_logs();
        app.expire_toast();

        terminal.draw(|frame| ui::draw(frame, app))?;
        app.drain_chat_events();
        app.drain_serve_command();
        app.drain_pull();
        app.drain_rm();
        app.drain_doctor();
        if let Some(text) = app.pending_clipboard.take() {
            emit_clipboard(&text);
        }

        if event::poll(std::time::Duration::from_millis(80))? {
            match event::read()? {
                Event::Key(key) => {
                    if handle_key(app, key) {
                        break;
                    }
                }
                Event::Mouse(mouse) => handle_mouse(app, mouse),
                Event::Resize(_, _) => {}
                _ => {}
            }
        }
    }
    Ok(())
}

fn handle_key(app: &mut App, key: KeyEvent) -> bool {
    // The quit-while-pulling confirmation is modal on every tab (and over the
    // help overlay): y quits after cancelling the pull, n / Esc disarms, and
    // every other key (including a second Ctrl+C or a global `q`) is swallowed.
    if app.confirm_quit {
        return app.handle_quit_confirm_key(key);
    }

    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
        if app.chat.sending {
            app.chat.status = "generation in progress — press Esc to stop it".into();
            return false;
        }
        return app.request_quit();
    }

    // The `?` help overlay is modal: while it is open, any key (other than the
    // Ctrl+C quit handled above) simply closes it.
    if app.show_help {
        app.show_help = false;
        return false;
    }

    // The scope picker and the per-model overrides list are modal: every key
    // goes to them (Esc closes), so q / r / ? / Tab cannot act underneath.
    if app.scope_screen_open() {
        app.handle_tab_key(key);
        return false;
    }

    // A Models delete-confirmation is modal: route every key to the tab handler
    // (which acts only on y / n / Esc) so a global `q` or `r` cannot quit or
    // reload while the confirm is armed.
    if app.tab == app::Tab::Models && app.confirm_delete.is_some() {
        app.handle_tab_key(key);
        return false;
    }

    // The Models pull chooser (PLE / draft) is modal as well: every key goes to
    // the tab handler so q / r / ? / Tab cannot act while it is open.
    if app.tab == app::Tab::Models && app.pull_choice.is_some() {
        app.handle_tab_key(key);
        return false;
    }

    // A Settings Tuning screen (chooser, preview, undo confirm) owns every key:
    // the preview must not be left by a global `q`, `r` or Tab by accident.
    if app.tab == app::Tab::Settings && app.tuning_open() {
        app.handle_tab_key(key);
        return false;
    }

    // A Dashboard stop/restart confirmation is modal too: every key goes to the
    // tab handler (y / n / Esc only) so q / r / ? / Tab cannot act while armed.
    if app.tab == app::Tab::Dashboard && app.confirm_serve.is_some() {
        app.handle_tab_key(key);
        return false;
    }

    // The Settings reset-all confirmation is likewise modal: route every key to
    // the tab handler (which acts only on y / n / Esc) so a global `q` or `r`
    // can't quit or reload while the destructive prompt is armed.
    if app.tab == app::Tab::Settings && app.confirm_reset_all {
        app.handle_tab_key(key);
        return false;
    }

    // While a settings value is being edited or the `/` search buffer has
    // focus, keystrokes (including q/r/?) feed it rather than triggering
    // global shortcuts.
    let editing_setting = app.tab == app::Tab::Settings
        && (app.settings_edit.is_some() || app.settings_search_typing());
    if editing_setting && !matches!(key.code, KeyCode::Tab | KeyCode::BackTab) {
        app.handle_tab_key(key);
        return false;
    }

    // The chat input only captures global keys (q/r/Esc) while the Chat tab is
    // focused AND its input is focused. On every other tab the input is not on
    // screen, so q/r/Esc act as global shortcuts immediately from startup
    // (chat input defaults focused, but that must not gate other tabs).
    let chat_capturing = app.tab == app::Tab::Chat && app.chat.is_input_focused();

    match key.code {
        KeyCode::Char('q') if !chat_capturing => return app.request_quit(),
        KeyCode::Esc => {
            // Esc only closes or cancels (a settings edit / enum preview, a chat
            // generation, the chat input focus); it never quits. With nothing to
            // cancel it does nothing. Help and the y/n modals are handled above.
            if app.tab == app::Tab::Settings
                && (app.settings_edit.is_some()
                    || app.settings_pending.is_some()
                    || app.settings_search.is_some())
            {
                app.handle_tab_key(key);
            } else if app.tab == app::Tab::Chat && app.has_queued_send() {
                // A queued send (serve not online yet) is cancelled before
                // Esc can blur the input or stop a generation.
                app.cancel_queued_send();
            } else if app.chat.sending {
                app.chat.request_abort();
            } else if chat_capturing {
                app.chat.blur_input();
            }
        }
        KeyCode::Tab => app.next_tab(),
        KeyCode::BackTab => app.prev_tab(),
        KeyCode::Char('r') if !chat_capturing => app.reload(),
        // `?` opens the keybinding help overlay (unless the chat input is
        // capturing text, where `?` is a literal character).
        KeyCode::Char('?') if !chat_capturing => app.show_help = true,
        _ => app.handle_tab_key(key),
    }

    false
}

/// Mouse handling: click a tab in the header to switch to it; scroll the wheel
/// to move within the active tab's list. All other mouse events are ignored.
/// A click or scroll also dismisses an open help overlay.
fn handle_mouse(app: &mut App, mouse: MouseEvent) {
    match mouse.kind {
        MouseEventKind::Down(MouseButton::Left) => {
            if app.show_help {
                app.show_help = false;
                return;
            }
            // Tab-bar click: map the clicked column to a tab via the hit regions
            // the renderer recorded this frame. Leaving the Settings tab discards
            // an armed reset-all confirm so it can't linger (and re-arm itself
            // when the user returns) after the modal key-guard no longer applies.
            if let Some(tab) = app.tab_at(mouse.column, mouse.row) {
                if tab != app::Tab::Settings {
                    // Leaving Settings discards an armed reset-all confirm and an
                    // uncommitted enum preview (neither should linger off-tab).
                    app.confirm_reset_all = false;
                    app.settings_pending = None;
                }
                if tab != app::Tab::Dashboard {
                    // Likewise a pending serve stop/restart confirm.
                    app.confirm_serve = None;
                }
                if tab != app::Tab::Models {
                    // And an open pull chooser (modal; must not reopen on return).
                    app.pull_choice = None;
                }
                app.close_scope_screen();
                app.tab = tab;
            }
        }
        MouseEventKind::ScrollDown => {
            if app.show_help {
                app.show_help = false;
                return;
            }
            app.handle_tab_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
        }
        MouseEventKind::ScrollUp => {
            if app.show_help {
                app.show_help = false;
                return;
            }
            app.handle_tab_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use app::Tab;

    fn test_app() -> (App, crate::test_support::TempHome) {
        crate::test_support::hermetic_app()
    }
    fn k(c: KeyCode) -> KeyEvent {
        KeyEvent::new(c, KeyModifiers::NONE)
    }
    fn mouse(kind: MouseEventKind, column: u16, row: u16) -> MouseEvent {
        MouseEvent {
            kind,
            column,
            row,
            modifiers: KeyModifiers::NONE,
        }
    }

    #[test]
    fn settings_flag_opens_on_settings_for_this_launch_only() {
        let (mut app, _home) = test_app();
        app.tab = Tab::Models; // what ui_state.json restored
        assert_eq!(apply_launch_tab(&mut app, false), None);
        assert_eq!(app.tab, Tab::Models, "no flag leaves the restored tab");

        let saved = apply_launch_tab(&mut app, true);
        assert_eq!(saved, Some(Tab::Models));
        assert_eq!(app.tab, Tab::Settings, "--settings starts on Settings");
        // Quitting while still on the forced tab keeps the previously saved one.
        assert_eq!(tab_to_persist(app.tab, saved), Tab::Models);
        // Navigating elsewhere persists as usual.
        assert_eq!(tab_to_persist(Tab::Logs, saved), Tab::Logs);
        // Without the flag nothing is rewritten.
        assert_eq!(tab_to_persist(Tab::Settings, None), Tab::Settings);
    }

    #[test]
    fn help_documents_the_settings_flag() {
        let help = help_text();
        assert!(help.contains("--settings"), "{help}");
        assert!(help.contains("Settings tab"), "{help}");
    }

    #[test]
    fn base64_matches_rfc4648_vectors() {
        use base64::engine::general_purpose::STANDARD;
        assert_eq!(STANDARD.encode(b""), "");
        assert_eq!(STANDARD.encode(b"f"), "Zg==");
        assert_eq!(STANDARD.encode(b"fo"), "Zm8=");
        assert_eq!(STANDARD.encode(b"foo"), "Zm9v");
        assert_eq!(STANDARD.encode(b"foob"), "Zm9vYg==");
        assert_eq!(STANDARD.encode(b"foobar"), "Zm9vYmFy");
        // High bytes exercise the full 6-bit table range.
        assert_eq!(STANDARD.encode([0xFF, 0x00, 0x80]), "/wCA");
        // Multibyte UTF-8 round-trips through the byte encoder.
        assert_eq!(STANDARD.encode("é".as_bytes()), "w6k=");
        // 32-byte blob (OSC 52 payloads are rarely tiny): bytes 0..32.
        let blob: Vec<u8> = (0..32).collect();
        assert_eq!(
            STANDARD.encode(&blob),
            "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8="
        );
    }

    #[test]
    fn osc52_frame_shape() {
        assert_eq!(osc52_sequence("foo"), "\x1b]52;c;Zm9v\x07");
    }

    #[test]
    fn question_mark_opens_help_then_any_key_closes_it() {
        let (mut app, _home) = test_app();
        app.tab = Tab::Home;
        assert!(!app.show_help);
        let quit = handle_key(&mut app, k(KeyCode::Char('?')));
        assert!(app.show_help, "? opens the help overlay");
        assert!(!quit);
        // Any subsequent key closes it, and must not quit or do anything else.
        let quit = handle_key(&mut app, k(KeyCode::Char('j')));
        assert!(!app.show_help, "any key closes the help overlay");
        assert!(!quit, "dismissing help must not quit");
    }

    #[test]
    fn left_click_on_tab_region_switches_tab() {
        let (mut app, _home) = test_app();
        app.tab = Tab::Home;
        app.tab_row_y = 3;
        app.tab_hitboxes = vec![(0, 6, Tab::Home), (9, 20, Tab::Dashboard)];
        handle_mouse(
            &mut app,
            mouse(MouseEventKind::Down(MouseButton::Left), 12, 3),
        );
        assert_eq!(app.tab, Tab::Dashboard);
    }

    #[test]
    fn click_off_the_tab_bar_does_not_switch() {
        let (mut app, _home) = test_app();
        app.tab = Tab::Home;
        app.tab_row_y = 3;
        app.tab_hitboxes = vec![(0, 6, Tab::Home), (9, 20, Tab::Dashboard)];
        handle_mouse(
            &mut app,
            mouse(MouseEventKind::Down(MouseButton::Left), 12, 10),
        );
        assert_eq!(
            app.tab,
            Tab::Home,
            "a click off the tab row changes nothing"
        );
    }

    #[test]
    fn scroll_dismisses_help_overlay() {
        let (mut app, _home) = test_app();
        app.show_help = true;
        handle_mouse(&mut app, mouse(MouseEventKind::ScrollDown, 0, 0));
        assert!(!app.show_help);
    }

    #[test]
    fn scroll_moves_list_selection() {
        // The wheel must actually move the active list (its real purpose), not
        // just dismiss the overlay. The Quality group always has several rows,
        // so the selection movement is deterministic.
        let (mut app, _home) = test_app();
        app.tab = Tab::Settings;
        app.settings_group = hipfire_config::intent::Intent::Quality;
        app.settings_search = None;
        app.settings_changed_only = false;
        app.settings_hide_unused = false;
        app.settings_selected = 0;
        handle_mouse(&mut app, mouse(MouseEventKind::ScrollDown, 0, 0));
        assert_eq!(
            app.settings_selected, 1,
            "scroll down advances the selection"
        );
        handle_mouse(&mut app, mouse(MouseEventKind::ScrollUp, 0, 0));
        assert_eq!(app.settings_selected, 0, "scroll up retreats it");
    }

    #[test]
    fn settings_search_keeps_q_r_and_help_keys_as_text() {
        let mut fixture = crate::test_support::FixtureBuilder::new("gfx1201").build();
        let app = &mut fixture.app;
        assert!(!handle_key(app, k(KeyCode::Char('/'))));
        for c in ['q', 'r', '?', 'e'] {
            assert!(!handle_key(app, k(KeyCode::Char(c))), "{c} quit");
        }
        assert_eq!(app.settings_search.as_ref().unwrap().query, "qr?e");
        assert!(!app.show_help);
        // Esc clears the search; with the buffer unfocused, q quits again.
        assert!(!handle_key(app, k(KeyCode::Esc)));
        assert!(app.settings_search.is_none());
        assert!(!handle_key(app, k(KeyCode::Char('/'))));
        assert!(!handle_key(app, k(KeyCode::Enter)));
        assert!(handle_key(app, k(KeyCode::Char('q'))));
    }

    #[test]
    fn esc_never_quits_on_any_non_chat_tab() {
        for tab in [
            Tab::Home,
            Tab::Dashboard,
            Tab::Models,
            Tab::Settings,
            Tab::System,
            Tab::Logs,
        ] {
            let (mut app, _home) = test_app();
            app.tab = tab;
            app.settings_edit = None;
            app.settings_pending = None;
            assert!(
                !handle_key(&mut app, k(KeyCode::Esc)),
                "Esc quit on {tab:?}"
            );
            assert_eq!(app.tab, tab);
        }
    }

    #[test]
    fn dashboard_restart_then_n_starts_no_serve_command() {
        let (mut app, _home) = test_app();
        app.tab = Tab::Dashboard;
        assert!(!handle_key(&mut app, k(KeyCode::Char('R'))));
        assert!(app.confirm_serve.is_some(), "R arms the confirm");
        assert!(!app.serve_cmd_running(), "arming starts nothing");
        assert!(!handle_key(&mut app, k(KeyCode::Char('n'))));
        assert!(app.confirm_serve.is_none());
        assert!(!app.serve_cmd_running(), "n starts no serve command");
        assert!(app.serve_cmd_label.is_empty());
        assert_eq!(
            app.toast.as_ref().map(|t| t.text.as_str()),
            Some("cancelled")
        );
    }

    #[test]
    fn dashboard_stop_then_y_starts_the_stop_command() {
        let (mut app, _home) = test_app();
        app.tab = Tab::Dashboard;
        handle_key(&mut app, k(KeyCode::Char('x')));
        assert!(app.confirm_serve.is_some());
        assert!(!app.serve_cmd_running(), "x alone runs nothing");
        assert!(!handle_key(&mut app, k(KeyCode::Char('y'))));
        assert!(app.confirm_serve.is_none());
        assert!(app.serve_cmd_running(), "y runs the stop command");
        assert_eq!(app.serve_cmd_label, "stop");
    }

    #[test]
    fn armed_serve_confirm_is_modal_and_disarms_on_tab_leave() {
        let (mut app, _home) = test_app();
        app.tab = Tab::Dashboard;
        handle_key(&mut app, k(KeyCode::Char('R')));
        assert!(
            !handle_key(&mut app, k(KeyCode::Char('q'))),
            "q must not quit"
        );
        assert!(
            !handle_key(&mut app, k(KeyCode::Char('r'))),
            "r must not reload"
        );
        assert!(!handle_key(&mut app, k(KeyCode::Char('?'))));
        assert!(!app.show_help, "? must not open help while armed");
        handle_key(&mut app, k(KeyCode::Tab));
        assert_eq!(app.tab, Tab::Dashboard, "Tab is swallowed while armed");
        assert!(app.confirm_serve.is_some(), "still armed after stray keys");
        assert!(
            !handle_key(&mut app, k(KeyCode::Esc)),
            "Esc cancels, never quits"
        );
        assert!(app.confirm_serve.is_none());

        // Leaving by Tab or mouse disarms.
        handle_key(&mut app, k(KeyCode::Char('x')));
        app.confirm_serve = Some(crate::hipfire::serve_ctrl::ServeAction::Stop);
        app.next_tab();
        assert!(app.confirm_serve.is_none(), "Tab disarms");
        app.tab = Tab::Dashboard;
        app.confirm_serve = Some(crate::hipfire::serve_ctrl::ServeAction::Restart);
        app.tab_row_y = 3;
        app.tab_hitboxes = vec![(0, 6, Tab::Home), (9, 20, Tab::Dashboard)];
        handle_mouse(
            &mut app,
            mouse(MouseEventKind::Down(MouseButton::Left), 2, 3),
        );
        assert_eq!(app.tab, Tab::Home);
        assert!(app.confirm_serve.is_none(), "mouse tab switch disarms");
        app.tab = Tab::Dashboard;
        app.confirm_serve = Some(crate::hipfire::serve_ctrl::ServeAction::Restart);
        app.prev_tab();
        assert!(app.confirm_serve.is_none(), "BackTab disarms");
    }

    fn ctrl_c() -> KeyEvent {
        KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)
    }

    fn cancel_set(flag: &std::sync::atomic::AtomicBool) -> bool {
        flag.load(std::sync::atomic::Ordering::SeqCst)
    }

    #[test]
    fn q_with_running_pull_asks_before_quitting() {
        let (mut app, _home) = test_app();
        app.tab = Tab::Home;
        let (_tx, rx) = std::sync::mpsc::channel();
        let cancel = app.inject_pull("qwen3.5:4b".into(), rx);

        assert!(
            !handle_key(&mut app, k(KeyCode::Char('q'))),
            "q arms, not quits"
        );
        assert!(app.confirm_quit);
        assert!(!cancel_set(&cancel));

        // Modal: tab switching, a second q and Ctrl+C are all swallowed.
        assert!(!handle_key(&mut app, k(KeyCode::Tab)));
        assert_eq!(app.tab, Tab::Home, "Tab is swallowed while armed");
        assert!(!handle_key(&mut app, k(KeyCode::Char('q'))));
        assert!(!handle_key(&mut app, ctrl_c()));
        assert!(app.confirm_quit);

        assert!(!handle_key(&mut app, k(KeyCode::Char('n'))), "n disarms");
        assert!(!app.confirm_quit);
        assert!(!cancel_set(&cancel), "n must not cancel the pull");
        assert!(app.pull.is_some());

        assert!(!handle_key(&mut app, k(KeyCode::Char('q'))));
        assert!(handle_key(&mut app, k(KeyCode::Char('y'))), "y quits");
        assert!(cancel_set(&cancel), "y cancels the pull before quitting");
    }

    #[test]
    fn ctrl_c_with_running_pull_asks_before_quitting() {
        let (mut app, _home) = test_app();
        app.tab = Tab::Models;
        let (_tx, rx) = std::sync::mpsc::channel();
        let cancel = app.inject_pull("qwen3.5:4b".into(), rx);

        assert!(!handle_key(&mut app, ctrl_c()));
        assert!(app.confirm_quit);
        assert!(
            !handle_key(&mut app, k(KeyCode::Esc)),
            "Esc disarms, not quits"
        );
        assert!(!app.confirm_quit);
        assert!(!cancel_set(&cancel));

        assert!(!handle_key(&mut app, ctrl_c()));
        assert!(handle_key(&mut app, k(KeyCode::Char('Y'))));
        assert!(cancel_set(&cancel));
    }

    #[test]
    fn quit_is_immediate_without_a_pull() {
        let (mut app, _home) = test_app();
        app.tab = Tab::Home;
        assert!(handle_key(&mut app, k(KeyCode::Char('q'))));
        assert!(!app.confirm_quit);
        assert!(handle_key(&mut app, ctrl_c()));
    }

    #[test]
    fn esc_on_chat_cancels_a_queued_send_instead_of_blurring_or_quitting() {
        let (mut app, _home) = test_app();
        app.tab = Tab::Chat;
        // Health is unknown at this point, so Enter queues (and starts nothing).
        app.chat.input = "hello".into();
        assert!(!handle_key(&mut app, k(KeyCode::Enter)));
        assert!(app.has_queued_send());
        assert!(app.chat.input.is_empty());

        assert!(!handle_key(&mut app, k(KeyCode::Esc)), "Esc must not quit");
        assert!(!app.has_queued_send());
        assert_eq!(app.chat.input, "hello", "text is back in the input box");
        assert_eq!(app.chat.status, "queued message cancelled");
        assert!(
            app.chat.is_input_focused(),
            "cancelling does not also blur the input"
        );
    }

    #[test]
    fn pull_chooser_is_modal_q_does_not_quit_and_esc_closes_it() {
        use crate::hipfire::registry::{ModelEntry, ModelRow};
        let (mut app, _home) = test_app();
        app.tab = Tab::Models;
        let mut entry = ModelEntry::default();
        entry.ple.insert(
            "q4".into(),
            hipfire_registry::Sidecar {
                file: "x.ple-q4".into(),
                sha256: None,
                size_bytes: Some(1_000_000_000),
            },
        );
        app.registry.models = vec![ModelRow {
            tag: "zz:ple".into(),
            entry,
            downloaded: false,
            has_mtp: false,
        }];
        app.registry.expanded_groups = ["zz".to_string()].into_iter().collect();
        app.registry.selected = 1;

        assert!(!handle_key(&mut app, k(KeyCode::Char('p'))));
        assert!(app.pull_choice.is_some());

        assert!(
            !handle_key(&mut app, k(KeyCode::Char('q'))),
            "q must not quit"
        );
        assert!(!app.confirm_quit);
        assert!(
            !handle_key(&mut app, k(KeyCode::Char('r'))),
            "r must not reload"
        );
        assert!(!handle_key(&mut app, k(KeyCode::Char('?'))));
        assert!(!app.show_help, "? must not open help");
        assert!(!handle_key(&mut app, k(KeyCode::Tab)));
        assert_eq!(app.tab, Tab::Models, "Tab must not switch");
        assert!(app.pull_choice.is_some());

        assert!(!handle_key(&mut app, k(KeyCode::Esc)));
        assert!(app.pull_choice.is_none(), "Esc closes the chooser");
        assert!(app.pull.is_none(), "Esc starts no pull");
        assert!(handle_key(&mut app, k(KeyCode::Char('q'))), "q quits again");
    }
}
