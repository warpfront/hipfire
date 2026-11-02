// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Keys of the Tuning screens (spec §2, §4 S2/S3/L2): `p` opens the chooser,
//! Enter previews, Space ticks, `y` saves the ticked net diff, `Y` saves and
//! restarts serve, `u` confirms an undo. The preview never writes; only `y`,
//! `Y` and the undo's `y` touch `config.toml`.

use crossterm::event::{KeyCode, KeyEvent};
use hipfire_config::tuning::Preset;

use super::App;
use crate::hipfire::{
    serve_ctrl::ServeAction,
    settings_model,
    tuning_view::{changes_label, Stage, TuningView},
};

impl App {
    /// A Tuning screen is on top of Settings.
    pub fn tuning_open(&self) -> bool {
        self.tuning_view.is_some()
    }

    /// `p`: the chooser (or, at 160×48 and up, the cards with the preview).
    pub(crate) fn open_tuning(&mut self) {
        let registry = settings_model::registry_v1(&self.registry);
        let mut view = TuningView::open(&self.paths, &self.arch, &registry);
        if self.tuning_wide {
            view.stage = Stage::Preview;
        }
        self.tuning_view = Some(view);
    }

    /// `u` on Settings: confirm what undo restores, or say there is nothing.
    pub(crate) fn open_tuning_undo(&mut self) {
        let registry = settings_model::registry_v1(&self.registry);
        let mut view = TuningView::open(&self.paths, &self.arch, &registry);
        if view.open_undo(false) {
            self.tuning_view = Some(view);
        } else {
            self.toast_info("nothing to undo");
        }
    }

    pub(crate) fn handle_tuning_key(&mut self, key: KeyEvent) {
        let wide = self.tuning_wide;
        let Some(view) = self.tuning_view.as_mut() else {
            return;
        };
        // At 160×48 the cards and the preview share one screen.
        if wide && view.stage == Stage::Chooser {
            view.stage = Stage::Preview;
        }
        match view.stage {
            Stage::Chooser => match key.code {
                KeyCode::Down | KeyCode::Char('j') => view.move_cursor(1),
                KeyCode::Up | KeyCode::Char('k') => view.move_cursor(-1),
                KeyCode::Enter => view.stage = Stage::Preview,
                KeyCode::Char('u') => {
                    if !view.open_undo(true) {
                        self.toast_info("nothing to undo");
                    }
                }
                KeyCode::Esc => self.tuning_view = None,
                _ => {}
            },
            Stage::Preview => match key.code {
                KeyCode::Down | KeyCode::Char('j') => view.move_selected(1),
                KeyCode::Up | KeyCode::Char('k') => view.move_selected(-1),
                KeyCode::Left => view.move_cursor(-1),
                KeyCode::Right => view.move_cursor(1),
                KeyCode::Char(' ') => {
                    if let Err(reason) = view.toggle_selected() {
                        self.toast_info(reason);
                    }
                }
                KeyCode::Char('y') => self.tuning_save(false),
                KeyCode::Char('Y') => self.tuning_save(true),
                KeyCode::Char('u') => {
                    if !view.open_undo(true) {
                        self.toast_info("nothing to undo");
                    }
                }
                KeyCode::Esc => {
                    if wide {
                        self.tuning_view = None;
                    } else {
                        view.stage = Stage::Chooser;
                    }
                }
                _ => {}
            },
            Stage::UndoConfirm => match key.code {
                KeyCode::Char('y') | KeyCode::Char('Y') => self.tuning_undo_save(),
                KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => {
                    if !view.cancel_undo() {
                        self.tuning_view = None;
                    }
                }
                _ => {}
            },
        }
    }

    /// `y` / `Y` in the preview: write the ticked net diff, close, rebuild
    /// Settings. `restart` then runs the Dashboard's serve restart (the user
    /// pressed `Y`, so there is no second confirm).
    fn tuning_save(&mut self, restart: bool) {
        let Some(view) = self.tuning_view.as_ref() else {
            return;
        };
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |elapsed| elapsed.as_secs());
        match view.save(&self.paths.root, now) {
            Ok(saved) => {
                self.tuning_view = None;
                self.after_config_write();
                if saved.changes == 0 {
                    self.toast_info(format!(
                        "nothing to change: {} is already in effect",
                        saved.preset.title()
                    ));
                    return;
                }
                let message = saved_message(saved.preset, saved.changes, saved.applies);
                self.last_reload = message.clone();
                self.toast_info(message);
                if restart {
                    self.start_serve_command(ServeAction::Restart);
                }
            }
            Err(error) => self.toast_error(format!("save failed: {error}")),
        }
    }

    /// `y` in the undo confirmation.
    fn tuning_undo_save(&mut self) {
        let Some(view) = self.tuning_view.as_ref() else {
            return;
        };
        match view.undo_save(&self.paths.root) {
            Ok(undone) => {
                self.tuning_view = None;
                self.after_config_write();
                let message = format!(
                    "Undone ({}) · now {}",
                    changes_label(undone.changes),
                    undone.restored.title()
                );
                self.last_reload = message.clone();
                self.toast_info(message);
            }
            Err(error) => self.toast_error(format!("undo failed: {error}")),
        }
    }
}

/// The after-save toast (spec §2.6).
fn saved_message(
    preset: Preset,
    changes: usize,
    applies: Option<hipfire_config::effective::Applies>,
) -> String {
    use hipfire_config::effective::Applies;
    match preset {
        Preset::Balanced => format!(
            "Back to Balanced ({changes} setting{} removed)",
            if changes == 1 { "" } else { "s" }
        ),
        _ => {
            let restart = if applies == Some(Applies::ServeRestart) {
                " · restart serve to use them"
            } else {
                ""
            };
            format!(
                "{} saved ({}){restart} · u undoes",
                preset.title(),
                changes_label(changes)
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use crossterm::event::KeyCode;
    use hipfire_config::{
        effective::{A8Prefill, Origin},
        load_global, write_global_toml, ConfigPaths, ConfigValue,
    };
    use ratatui::{backend::TestBackend, Terminal};

    use crate::{
        app::App,
        hipfire::{serve_ctrl::ServeAction, settings_model::RowId, tuning_view::Stage, writer},
        test_support::{online_dashboard, press, Fixture, FixtureBuilder},
        ui::draw,
    };

    fn render(app: &mut App, width: u16, height: u16) -> String {
        let mut terminal = Terminal::new(TestBackend::new(width, height)).expect("terminal");
        terminal.draw(|frame| draw(frame, app)).expect("draw");
        let buf = terminal.backend().buffer().clone();
        (0..buf.area.height)
            .map(|y| {
                let row: String = (0..buf.area.width).map(|x| buf[(x, y)].symbol()).collect();
                row.trim_end().to_owned()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// gfx1201 (no A8, no PLE line): Quality-first = two changes.
    fn r9700() -> Fixture {
        let mut fixture = FixtureBuilder::new("gfx1201").build();
        normalise_config(&fixture);
        fixture.app.rebuild_settings();
        fixture
    }

    /// Re-serialise `config.toml` through the same writer the app uses, so a
    /// later byte comparison is not about formatting.
    fn normalise_config(fixture: &Fixture) {
        let paths = ConfigPaths::under(fixture.root());
        let layer = load_global(&paths).unwrap().layer;
        write_global_toml(&paths, &layer).unwrap();
    }

    fn global(fixture: &Fixture) -> hipfire_config::ConfigLayer {
        load_global(&ConfigPaths::under(fixture.root()))
            .unwrap()
            .layer
    }

    fn open_preview(fixture: &mut Fixture) {
        press(&mut fixture.app, KeyCode::Char('p'));
        press(&mut fixture.app, KeyCode::Enter);
    }

    #[test]
    fn p_opens_the_chooser_enter_previews_and_config_stays_byte_identical() {
        let mut fixture = r9700();
        let before = fixture.config_bytes();
        press(&mut fixture.app, KeyCode::Char('p'));
        assert_eq!(
            fixture.app.tuning_view.as_ref().unwrap().stage,
            Stage::Chooser
        );
        let chooser = render(&mut fixture.app, 80, 24);
        assert!(
            chooser.contains("Tuning for gfx1201 · all models"),
            "{chooser}"
        );
        assert!(chooser.contains("Quality-first"), "{chooser}");
        press(&mut fixture.app, KeyCode::Enter);
        assert_eq!(
            fixture.app.tuning_view.as_ref().unwrap().stage,
            Stage::Preview
        );
        let preview = render(&mut fixture.app, 80, 24);
        assert!(preview.contains("Preview · Quality-first"), "{preview}");
        assert!(preview.contains("[x] Prompt processing"), "{preview}");
        assert!(preview.contains("y save 2 changes"), "{preview}");
        assert_eq!(fixture.config_bytes(), before, "preview must not write");
        assert!(!fixture.root().join("tuning.toml").exists());
        // Esc backs out one screen at a time; nothing was ever written.
        press(&mut fixture.app, KeyCode::Esc);
        assert_eq!(
            fixture.app.tuning_view.as_ref().unwrap().stage,
            Stage::Chooser
        );
        press(&mut fixture.app, KeyCode::Esc);
        assert!(fixture.app.tuning_view.is_none());
        assert_eq!(fixture.config_bytes(), before);
    }

    #[test]
    fn space_unticks_and_the_footer_count_drops() {
        let mut fixture = r9700();
        open_preview(&mut fixture);
        press(&mut fixture.app, KeyCode::Char(' '));
        let text = render(&mut fixture.app, 80, 24);
        assert!(text.contains("[ ] Prompt processing"), "{text}");
        assert!(text.contains("y save 1 change"), "{text}");
        assert!(!text.contains("y save 1 changes"), "{text}");
        // Ticking it again restores the count.
        press(&mut fixture.app, KeyCode::Char(' '));
        assert!(render(&mut fixture.app, 80, 24).contains("y save 2 changes"));
    }

    #[test]
    fn an_unmet_line_cannot_be_ticked_and_space_says_why() {
        let mut fixture = FixtureBuilder::new("gfx1151")
            .a8(A8Prefill::OptIn)
            .flash_next_split_base()
            .build();
        open_preview(&mut fixture);
        let text = render(&mut fixture.app, 80, 24);
        assert!(text.contains("[ ] Flash-Next word table"), "{text}");
        assert!(
            text.contains("hipfire pull qwen3.8:flash-next --ple q8"),
            "{text}"
        );
        assert!(text.contains("y save 2 changes"), "{text}");
        press(&mut fixture.app, KeyCode::Down);
        press(&mut fixture.app, KeyCode::Down);
        press(&mut fixture.app, KeyCode::Char(' '));
        let toast = fixture
            .app
            .toast
            .as_ref()
            .expect("reason toast")
            .text
            .clone();
        assert!(toast.contains("no .ple-q8"), "{toast}");
        assert!(
            toast.contains("hipfire pull qwen3.8:flash-next --ple q8"),
            "{toast}"
        );
        fixture.app.toast = None;
        let after = render(&mut fixture.app, 80, 24);
        assert!(after.contains("[ ] Flash-Next word table"), "{after}");
        assert!(after.contains("y save 2 changes"), "{after}");
    }

    #[test]
    fn y_writes_exactly_the_net_diff_and_rows_read_preset() {
        let mut fixture = r9700();
        let before = global(&fixture);
        open_preview(&mut fixture);
        press(&mut fixture.app, KeyCode::Char('y'));
        assert!(fixture.app.tuning_view.is_none(), "y closes the screen");
        let toast = fixture
            .app
            .toast
            .as_ref()
            .expect("saved toast")
            .text
            .clone();
        assert_eq!(
            toast,
            "Quality-first saved (2 changes) · restart serve to use them · u undoes"
        );
        let mut want = before;
        want.set("kernel.iu4_prefill", ConfigValue::Bool(false))
            .unwrap();
        want.set("developer.qwen4_moe_sym_iu4", ConfigValue::Bool(false))
            .unwrap();
        assert_eq!(global(&fixture), want);
        assert!(fixture.root().join("tuning.toml").exists());
        assert_eq!(fixture.app.settings.tuning.label(), "Quality-first");
        let header = render(&mut fixture.app, 80, 24);
        assert!(header.contains("Tuning: Quality-first  p"), "{header}");
        for key in ["kernel.iu4_prefill", "developer.qwen4_moe_sym_iu4"] {
            let row = fixture
                .app
                .settings
                .row(&RowId::Key(key.to_owned()))
                .expect(key);
            assert_eq!(row.origin, Origin::Preset, "{key}");
            assert!(row.origin.marked(), "{key} shows ●");
        }
        let prompt = fixture
            .app
            .settings
            .row(&RowId::Virtual(
                crate::hipfire::settings_model::VirtualRow::PromptProcessing,
            ))
            .unwrap();
        assert_eq!(prompt.origin, Origin::Preset);
    }

    #[test]
    fn hand_editing_an_owned_key_reads_edited() {
        let mut fixture = r9700();
        open_preview(&mut fixture);
        press(&mut fixture.app, KeyCode::Char('y'));
        writer::write_value(&fixture.app.paths.config, "kernel.iu4_prefill", "true").unwrap();
        fixture.app.rebuild_settings();
        assert_eq!(
            fixture.app.settings.tuning.label(),
            "Quality-first (edited)"
        );
        let header = render(&mut fixture.app, 80, 24);
        assert!(
            header.contains("Tuning: Quality-first (edited)"),
            "{header}"
        );
    }

    #[test]
    fn u_lists_the_restores_and_y_restores_the_bytes() {
        let mut fixture = r9700();
        let before = fixture.config_bytes();
        // Nothing applied yet: nothing to undo.
        press(&mut fixture.app, KeyCode::Char('u'));
        assert!(fixture.app.tuning_view.is_none());
        assert_eq!(
            fixture.app.toast.as_ref().map(|toast| toast.text.as_str()),
            Some("nothing to undo")
        );
        open_preview(&mut fixture);
        press(&mut fixture.app, KeyCode::Char('y'));
        assert_ne!(fixture.config_bytes(), before);

        press(&mut fixture.app, KeyCode::Char('u'));
        assert_eq!(
            fixture.app.tuning_view.as_ref().unwrap().stage,
            Stage::UndoConfirm
        );
        let confirm = render(&mut fixture.app, 80, 24);
        assert!(confirm.contains("kernel.iu4_prefill"), "{confirm}");
        assert!(confirm.contains("developer.qwen4_moe_sym_iu4"), "{confirm}");
        assert!(confirm.contains("false → unset"), "{confirm}");
        assert!(
            confirm.contains("Tuning goes back to Balanced"),
            "{confirm}"
        );
        // Esc cancels without writing.
        let applied = fixture.config_bytes();
        press(&mut fixture.app, KeyCode::Esc);
        assert!(fixture.app.tuning_view.is_none());
        assert_eq!(fixture.config_bytes(), applied);

        press(&mut fixture.app, KeyCode::Char('u'));
        press(&mut fixture.app, KeyCode::Char('y'));
        assert!(fixture.app.tuning_view.is_none());
        assert_eq!(fixture.config_bytes(), before, "undo restores the bytes");
        assert_eq!(fixture.app.settings.tuning.label(), "Balanced");
        // Single level: a second u has nothing left.
        press(&mut fixture.app, KeyCode::Char('u'));
        assert!(fixture.app.tuning_view.is_none());
    }

    #[test]
    fn capital_y_saves_then_restarts_serve_once() {
        let mut fixture = r9700();
        fixture.app.dashboard = Some(online_dashboard(None, 120));
        open_preview(&mut fixture);
        let preview = render(&mut fixture.app, 80, 24);
        assert!(
            preview.contains("!!  serve is running qwen3.8:27b with the old values"),
            "{preview}"
        );
        press(&mut fixture.app, KeyCode::Char('Y'));
        assert!(fixture.app.tuning_view.is_none());
        assert_eq!(fixture.app.settings.tuning.label(), "Quality-first");
        assert_eq!(fixture.app.serve_commands, [ServeAction::Restart]);
        assert_eq!(fixture.app.serve_cmd_label, "restart");
    }

    #[test]
    fn lowercase_y_never_restarts_serve() {
        let mut fixture = r9700();
        open_preview(&mut fixture);
        press(&mut fixture.app, KeyCode::Char('y'));
        assert!(fixture.app.serve_commands.is_empty());
    }

    #[test]
    fn an_env_override_shows_the_check_and_still_saves() {
        let mut fixture = FixtureBuilder::new("gfx1201")
            .env_cli("kernel.iu4_prefill", "1")
            .build();
        open_preview(&mut fixture);
        let text = render(&mut fixture.app, 80, 24);
        assert!(
            text.contains("!!  HIPFIRE_IU4_PREFILL is set in this shell and overrides"),
            "{text}"
        );
        press(&mut fixture.app, KeyCode::Char('y'));
        assert!(fixture.app.tuning_view.is_none());
        assert_eq!(
            global(&fixture).get("kernel.iu4_prefill"),
            Some(&ConfigValue::Bool(false))
        );
    }

    #[test]
    fn balanced_lists_what_it_gives_back() {
        let mut fixture = r9700();
        open_preview(&mut fixture);
        press(&mut fixture.app, KeyCode::Char('y'));
        fixture.app.toast = None;
        press(&mut fixture.app, KeyCode::Char('p'));
        // Cursor starts on the applied preset; step to Balanced.
        press(&mut fixture.app, KeyCode::Up);
        press(&mut fixture.app, KeyCode::Enter);
        let text = render(&mut fixture.app, 80, 24);
        assert!(text.contains("Preview · Balanced"), "{text}");
        assert!(
            text.contains("Gives back what Quality-first set:"),
            "{text}"
        );
        assert!(text.contains("kernel.iu4_prefill"), "{text}");
        assert!(text.contains("false → unset"), "{text}");
        assert!(text.contains("y save 2 changes"), "{text}");
        press(&mut fixture.app, KeyCode::Char('y'));
        assert_eq!(
            fixture.app.toast.as_ref().map(|toast| toast.text.as_str()),
            Some("Back to Balanced (2 settings removed)")
        );
        assert_eq!(fixture.app.settings.tuning.label(), "Balanced");
    }

    #[test]
    fn wide_layout_goes_straight_to_the_preview_and_arrows_switch_presets() {
        let mut fixture = FixtureBuilder::new("gfx1201").build();
        render(&mut fixture.app, 160, 48);
        assert!(fixture.app.tuning_wide);
        press(&mut fixture.app, KeyCode::Char('p'));
        assert_eq!(
            fixture.app.tuning_view.as_ref().unwrap().stage,
            Stage::Preview
        );
        press(&mut fixture.app, KeyCode::Right);
        let text = render(&mut fixture.app, 160, 48);
        assert!(text.contains("Preview · Low-power"), "{text}");
        assert!(text.contains("y save 1 change"), "{text}");
        press(&mut fixture.app, KeyCode::Esc);
        assert!(fixture.app.tuning_view.is_none(), "Esc leaves for Settings");
    }
}
