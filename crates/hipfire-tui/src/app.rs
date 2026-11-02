// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

use std::{
    cell::Cell,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver},
        Arc,
    },
    thread,
    time::{Duration, Instant},
};

use anyhow::Result;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use hipfire_config::{effective::ArchCtx, intent::Intent};

use crate::hipfire::{
    chat::{stream_chat, ChatEvent, ChatMessage},
    chat_history::ChatHistory,
    config::ConfigState,
    dashboard::{Dashboard, DashboardWorker, HealthSnapshot},
    doctor::{self, DoctorReport},
    log_tail::{LogSnapshot, LogTailer},
    metrics::MetricsView,
    model_actions::{self, PullEvent, PullHandle, RmOutcome},
    pull_choice::PullChoice,
    registry::{RegistryAction, RegistryState},
    scope_view::ScopeState,
    serve_ctrl::{self, ServeAction, ServeOutcome},
    settings_model::{self, SettingsModel},
    status::{start_background_serve, StatusState},
    tuning_view::TuningView,
    ui_state::UiState,
    writer, HipfirePaths,
};

mod scope;
mod settings;
mod tuning;

pub use settings::SettingsSearch;

/// In-progress numeric/string edit of a settings field (Enter to commit,
/// Esc to cancel). Enum fields use [`PendingEnum`] instead of this state.
pub struct EditState {
    pub key: String,
    pub buffer: String,
    /// Insertion point as a CHAR index into `buffer` (`0..=chars`). Typing
    /// inserts here; Backspace/Delete remove before/after it.
    pub cursor: usize,
}

impl EditState {
    /// A fresh edit seeded with the current value, cursor at the end.
    pub fn new(key: String, buffer: String) -> Self {
        let cursor = buffer.chars().count();
        Self {
            key,
            buffer,
            cursor,
        }
    }

    fn byte_at(&self, char_index: usize) -> usize {
        self.buffer
            .char_indices()
            .nth(char_index)
            .map_or(self.buffer.len(), |(byte, _)| byte)
    }

    pub fn insert(&mut self, c: char) {
        let at = self.byte_at(self.cursor);
        self.buffer.insert(at, c);
        self.cursor += 1;
    }

    /// Remove the char before the cursor.
    pub fn backspace(&mut self) {
        if self.cursor > 0 {
            self.cursor -= 1;
            let at = self.byte_at(self.cursor);
            self.buffer.remove(at);
        }
    }

    /// Remove the char under the cursor.
    pub fn delete(&mut self) {
        if self.cursor < self.buffer.chars().count() {
            let at = self.byte_at(self.cursor);
            self.buffer.remove(at);
        }
    }

    pub fn left(&mut self) {
        self.cursor = self.cursor.saturating_sub(1);
    }

    pub fn right(&mut self) {
        self.cursor = (self.cursor + 1).min(self.buffer.chars().count());
    }

    pub fn home(&mut self) {
        self.cursor = 0;
    }

    pub fn end(&mut self) {
        self.cursor = self.buffer.chars().count();
    }
}

/// Status line for a key the inline editor has no control for.
pub fn not_editable_status(key: &str) -> String {
    format!("{key} is not editable inline — edit with `hipfire config set {key}`")
}

/// Status line for a deprecated CASK/PFlash key: shown, never edited inline.
pub fn deprecated_status(key: &str) -> String {
    format!(
        "{key} is deprecated (removal 0.5.0) and read-only here — edit with `hipfire config set {key}`"
    )
}

/// A previewed-but-uncommitted enum value (5b). Cycling Left/Right/Space stages
/// the next value here (dimmed in the UI) WITHOUT writing config.toml, so merely
/// browsing the options can't mutate config. Enter commits it; Esc or navigating
/// away discards it. Booleans don't use this — a binary flip is a decision, not
/// browsing, so they still toggle-on-key.
pub struct PendingEnum {
    pub key: String,
    pub value: String,
}

/// Severity of a transient toast, drives its color in the footer overlay.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ToastLevel {
    Info,
    Error,
}

/// What a queued Chat send will do once the serve is online.
#[derive(Clone, Debug, PartialEq, Eq)]
enum QueuedKind {
    /// Send this prompt as a new user turn.
    Prompt(String),
    /// Re-stream the last user turn (`/regen`).
    Regen,
}

/// A Chat send held back because the serve is not (yet) online. Sent
/// automatically when `/health` turns online; Esc on Chat cancels it.
#[derive(Clone, Debug, PartialEq, Eq)]
struct QueuedSend {
    kind: QueuedKind,
    /// `hipfire serve -d` was already requested (or found running) for this
    /// queue entry; never re-spawn it every frame.
    serve_start_handled: bool,
}

/// A transient status line shown over the footer for a few seconds after an
/// action (a write failure, a serve-unreachable action, a save confirmation).
/// Expires on its own — the UI checks [`Toast::is_live`] each frame.
#[derive(Clone, Debug)]
pub struct Toast {
    pub text: String,
    pub level: ToastLevel,
    born: Instant,
    ttl: Duration,
}

impl Toast {
    pub fn is_live(&self) -> bool {
        self.born.elapsed() < self.ttl
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Tab {
    Home,
    Dashboard,
    Chat,
    Models,
    Settings,
    System,
    Logs,
}

impl Tab {
    pub const ALL: [Tab; 7] = [
        Tab::Home,
        Tab::Dashboard,
        Tab::Chat,
        Tab::Models,
        Tab::Settings,
        Tab::System,
        Tab::Logs,
    ];

    pub fn title(self) -> &'static str {
        match self {
            Tab::Home => "Home",
            Tab::Dashboard => "Dashboard",
            Tab::Chat => "Chat",
            Tab::Models => "Models",
            Tab::Settings => "Settings",
            Tab::System => "System",
            Tab::Logs => "Logs",
        }
    }

    /// Resolve a tab from its title (for restoring the persisted active tab).
    pub fn from_title(s: &str) -> Option<Tab> {
        Tab::ALL.into_iter().find(|t| t.title() == s)
    }
}

/// In-flight model download. `percent`/`line` are refreshed each frame from the
/// background pull worker's progress events. Dropping the job cancels the pull
/// and reaps the child (see [`PullHandle`]).
pub struct PullJob {
    pub tag: String,
    pub handle: PullHandle,
    pub percent: Option<f64>,
    pub line: String,
}

/// How long a quit waits for the pull worker to kill and reap its child.
const PULL_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(3);

pub struct App {
    pub paths: HipfirePaths,
    pub config: ConfigState,
    /// GPU and active-model context the effective values are predicted for.
    pub arch: ArchCtx,
    /// The `HIPFIRE_*` environment layer, captured at load (injected in tests).
    pub env_layer: hipfire_config::ConfigLayer,
    /// Config and registry loads read nothing but `paths` (no process env, no
    /// registry fetch): set by [`App::load_from_paths`], never by [`App::load`].
    hermetic: bool,
    /// Settings rows (intent groups, effective values, origins, locks).
    pub settings: SettingsModel,
    /// Per-model scope: what Settings edits, `models.toml`, the `m` / `o` screens.
    pub scope: ScopeState,
    pub registry: RegistryState,
    pub status: StatusState,
    pub active_model: String,
    pub tab: Tab,
    /// The Settings group on screen (ignored while a search is active).
    pub settings_group: Intent,
    /// Selected row within [`App::settings_visible`].
    pub settings_selected: usize,
    /// The `/` search, while one is open or its filter is kept.
    pub settings_search: Option<SettingsSearch>,
    /// `c`: show only changed (●) rows.
    pub settings_changed_only: bool,
    /// `h`: hide rows the active model does not use.
    pub settings_hide_unused: bool,
    pub settings_edit: Option<EditState>,
    /// A staged-but-uncommitted enum value preview (5b). `Some` only while the
    /// user is cycling an enum and hasn't pressed Enter (commit) / Esc (cancel).
    pub settings_pending: Option<PendingEnum>,
    /// Settings tab: the Tuning chooser / preview / undo screen (`p`, `u`).
    pub tuning_view: Option<TuningView>,
    /// The last frame was 160×48 or larger (the Tuning cards + preview layout).
    pub tuning_wide: bool,
    /// Test seam: every serve lifecycle command [`App::start_serve_command`]
    /// started, in order.
    #[cfg(test)]
    pub serve_commands: Vec<ServeAction>,
    pub chat: ChatState,
    pub last_reload: String,
    /// Transient over-footer status toast (action failures + confirmations).
    /// `None` when nothing recent; the UI clears it once it expires.
    pub toast: Option<Toast>,
    /// Latest live-serve Dashboard snapshot mirrored from the background fetch
    /// thread each frame (None until the worker's first fetch completes). The
    /// UI thread ONLY reads this — it never calls fetch_dashboard / rocm-smi /
    /// HTTP synchronously, so a hung probe cannot block render or input.
    pub dashboard: Option<Dashboard>,
    /// Latest Live metrics view from the worker's background `/metrics` scrape,
    /// mirrored each frame while the Dashboard tab is shown.
    pub live_metrics: MetricsView,
    /// Whether the `?` keybinding help overlay is currently open.
    pub show_help: bool,
    /// Per-frame tab-bar hit regions `[x_start, x_end)` -> tab, recomputed by
    /// the renderer each frame so a mouse-click column maps to a tab.
    pub tab_hitboxes: Vec<(u16, u16, Tab)>,
    /// Terminal row the tab labels render on (for click hit-testing).
    pub tab_row_y: u16,
    /// In-flight serve lifecycle command (start/stop/restart). Its single
    /// outcome arrives on this receiver; `drain_serve_command` consumes it each
    /// frame and turns it into a toast. `None` when no command is running.
    serve_cmd: Option<Receiver<ServeOutcome>>,
    /// Label ("start"/"stop"/"restart") of the in-flight serve command, for the
    /// status line; empty when idle.
    pub serve_cmd_label: String,
    /// In-flight model pull (Models tab). `None` when idle.
    pub pull: Option<PullJob>,
    /// In-flight `rm` command receiver (Models tab).
    rm_cmd: Option<Receiver<RmOutcome>>,
    /// Model tag awaiting a y/n delete confirmation; the Models tab shows a
    /// prompt while this is `Some`.
    pub confirm_delete: Option<String>,
    /// Settings tab: a "reset ALL config to defaults" y/n confirmation is armed.
    /// Single-key reset is reversible so it commits immediately; wiping the whole
    /// file is gated behind this prompt (mirrors the Models delete-confirm modal).
    pub confirm_reset_all: bool,
    /// Dashboard tab: a stop/restart y/n confirmation is armed for this action.
    /// Stop and restart are destructive (`R` force-reaps this install's serve
    /// daemons), so they wait for an explicit y; `s` start stays immediate.
    pub confirm_serve: Option<ServeAction>,
    /// A y/n "quit and cancel the running pull?" confirmation is armed. Modal on
    /// every tab; only ever armed while `pull` is `Some`.
    pub confirm_quit: bool,
    /// Models tab: the `p` pull chooser (PLE format + draft sidecars) is open for
    /// this model. Modal; `None` when closed.
    pub pull_choice: Option<PullChoice>,
    /// Text staged by Chat `/copy` for the event loop to emit to the terminal
    /// clipboard (OSC52) after the next render; cleared once emitted.
    pub pending_clipboard: Option<String>,
    /// Recent finished chat requests (newest first), for the System-tab request
    /// inspector. Bounded to [`REQUEST_LOG_CAP`].
    pub request_log: Vec<RequestRecord>,
    /// In-flight `hipfire diag --json` doctor run (System tab `d`).
    doctor_cmd: Option<Receiver<DoctorReport>>,
    /// Latest doctor report (System tab); `None` until first run.
    pub doctor: Option<DoctorReport>,
    /// Latest serve.log tail (Logs tab), mirrored from the LogTailer worker.
    pub logs: LogSnapshot,
    /// Background serve.log tailer. Reads the file off the UI thread; dropped
    /// (joined) when the App is dropped.
    log_tailer: LogTailer,
    /// Background fetch thread. Owns the network + rocm-smi I/O off the UI
    /// thread. Dropped (joined) when the App is dropped.
    dashboard_worker: DashboardWorker,
    /// Chat send waiting for the serve to come online (see [`QueuedSend`]).
    queued_send: Option<QueuedSend>,
}

impl App {
    /// Load from the real `~/.hipfire` (or `HIPFIRE_HOME`), the GPU recorded in
    /// its `install.json`, and the process environment.
    pub fn load() -> Result<Self> {
        let paths = HipfirePaths::discover();
        let (env, env_warning) = match hipfire_config::load_env_layer() {
            Ok(layer) => (layer, None),
            Err(error) => (
                hipfire_config::ConfigLayer::default(),
                Some(format!("environment: {error}")),
            ),
        };
        let mut app = Self::assemble(paths, None, env, false)?;
        if let Some(message) = env_warning {
            app.config.warning = Some(match app.config.warning.take() {
                Some(previous) => format!("{previous}; {message}"),
                None => message,
            });
        }
        Ok(app)
    }

    /// Load from explicit inputs: the state root, the GPU/model context and the
    /// environment layer. Reads neither `~/.hipfire` nor the process
    /// environment, so tests are hermetic.
    #[allow(dead_code)] // consumed by the Settings pane rewrite and the tests
    pub fn load_from_paths(
        paths: HipfirePaths,
        arch: ArchCtx,
        env: hipfire_config::ConfigLayer,
    ) -> Result<Self> {
        Self::assemble(paths, Some(arch), env, true)
    }

    fn load_config(&self) -> ConfigState {
        if self.hermetic {
            ConfigState::load_hermetic(&self.paths)
        } else {
            ConfigState::load(&self.paths)
        }
    }

    /// `arch: None` derives the context from `install.json` and the active
    /// model's registry card.
    fn assemble(
        paths: HipfirePaths,
        arch: Option<ArchCtx>,
        env_layer: hipfire_config::ConfigLayer,
        hermetic: bool,
    ) -> Result<Self> {
        let (config, mut registry) = if hermetic {
            (
                ConfigState::load_hermetic(&paths),
                RegistryState::load_bundled(&paths),
            )
        } else {
            (ConfigState::load(&paths), RegistryState::load(&paths))
        };
        let status = StatusState::load_local(&paths);
        let active_model = config.default_model.clone();
        let registry_v1 = settings_model::registry_v1(&registry);
        let arch = arch
            .unwrap_or_else(|| settings_model::arch_ctx(&paths.root, &registry_v1, &active_model));
        let settings = SettingsModel::build(&paths, &active_model, &registry_v1, &arch, &env_layer);
        let mut scope = ScopeState::default();
        scope.reload(&hipfire_config::ConfigPaths::under(&paths.root), &env_layer);
        let dashboard_worker = DashboardWorker::spawn(config.clone());
        let log_tailer = LogTailer::spawn(paths.serve_log.clone());

        // Restore session UI state (active tab + expanded Models groups) so the
        // TUI reopens where the user left off. Missing/corrupt -> defaults. The
        // active *model* is restored separately via config.default_model, which
        // the Models tab already persists.
        let ui = UiState::load(&paths.ui_state);
        let tab = Tab::from_title(&ui.active_tab).unwrap_or(Tab::Home);
        registry.expanded_groups = ui.expanded_groups.into_iter().collect();

        Ok(Self {
            paths,
            config,
            arch,
            env_layer,
            hermetic,
            settings,
            scope,
            registry,
            status,
            active_model,
            tab,
            settings_group: Intent::Quality,
            settings_selected: 0,
            settings_search: None,
            settings_changed_only: false,
            settings_hide_unused: false,
            settings_edit: None,
            settings_pending: None,
            tuning_view: None,
            tuning_wide: false,
            #[cfg(test)]
            serve_commands: Vec::new(),
            chat: ChatState::default(),
            last_reload: "loaded hipfire state".into(),
            toast: None,
            dashboard: None,
            live_metrics: MetricsView::Collecting,
            show_help: false,
            tab_hitboxes: Vec::new(),
            tab_row_y: 0,
            serve_cmd: None,
            serve_cmd_label: String::new(),
            pull: None,
            rm_cmd: None,
            confirm_delete: None,
            confirm_reset_all: false,
            confirm_serve: None,
            confirm_quit: false,
            pull_choice: None,
            pending_clipboard: None,
            request_log: Vec::new(),
            doctor_cmd: None,
            doctor: None,
            logs: LogSnapshot::default(),
            log_tailer,
            dashboard_worker,
            queued_send: None,
        })
    }

    /// Rebuild the Settings rows from disk, the loaded registry and the
    /// captured environment layer (after a config write or an active-model
    /// change).
    pub fn rebuild_settings(&mut self) {
        let registry_v1 = settings_model::registry_v1(&self.registry);
        self.scope.reload(
            &hipfire_config::ConfigPaths::under(&self.paths.root),
            &self.env_layer,
        );
        let (subject, arch) = self.settings_subject(&registry_v1);
        self.settings =
            SettingsModel::build(&self.paths, &subject, &registry_v1, &arch, &self.env_layer);
        self.refresh_scope_screen();
        self.clamp_settings_selection();
    }

    /// Mirror the latest serve.log tail from the background tailer, and tell it
    /// whether the Logs tab is focused (so it only reads while visible). Called
    /// each frame — a cheap lock+clone, never a synchronous file read on the UI
    /// thread.
    pub fn sync_logs(&mut self) {
        self.log_tailer.set_active(self.tab == Tab::Logs);
        if self.tab == Tab::Logs {
            self.logs = self.log_tailer.snapshot();
        }
    }

    /// Start downloading `tag` on a background worker (Models tab `p`). No-op
    /// (with a toast) if a pull is already running or the model is already local.
    fn start_pull(&mut self, tag: String, extra_args: Vec<String>) {
        if self.pull.is_some() {
            self.toast_info("a pull is already running");
            return;
        }
        let handle = model_actions::pull(tag.clone(), extra_args);
        self.toast_info(format!("pulling {tag}\u{2026}"));
        self.pull = Some(PullJob {
            tag,
            handle,
            percent: None,
            line: String::new(),
        });
    }

    /// `p` on a remote model: open the PLE / draft chooser when the registry
    /// entry has PLE files or draft sidecars, otherwise pull immediately.
    fn request_pull(&mut self, tag: String) {
        if self.pull.is_some() {
            self.toast_info("a pull is already running");
            return;
        }
        let choice = self
            .registry
            .entry(&tag)
            .and_then(|entry| PullChoice::for_entry(&tag, entry));
        match choice {
            Some(choice) => self.pull_choice = Some(choice),
            None => self.start_pull(tag, Vec::new()),
        }
    }

    /// Keys while the pull chooser is open (modal): Up/Down pick the row,
    /// Left/Right/Space change its value, Enter starts the pull, Esc closes.
    fn handle_pull_choice_key(&mut self, key: KeyEvent) {
        let Some(choice) = self.pull_choice.as_mut() else {
            return;
        };
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => choice.move_row(false),
            KeyCode::Down | KeyCode::Char('j') => choice.move_row(true),
            KeyCode::Left => choice.change(false),
            KeyCode::Right | KeyCode::Char(' ') => choice.change(true),
            KeyCode::Enter => {
                if let Some(choice) = self.pull_choice.take() {
                    let args = choice.extra_args();
                    self.start_pull(choice.tag, args);
                }
            }
            KeyCode::Esc => {
                self.pull_choice = None;
                self.toast_info("pull not started");
            }
            _ => {}
        }
    }

    /// Ask the running pull to stop (Models tab `c`). The worker kills and reaps
    /// the child and answers with `PullEvent::Cancelled`, which `drain_pull`
    /// turns into the "pull cancelled" toast.
    fn cancel_pull(&mut self) {
        match &self.pull {
            Some(job) => job.handle.cancel(),
            None => self.toast_info("no pull running"),
        }
    }

    /// Cancel the running pull (if any) and wait, bounded, for its worker to
    /// kill and reap the child.
    fn shutdown_pull(&mut self) {
        if let Some(mut job) = self.pull.take() {
            job.handle.shutdown(PULL_SHUTDOWN_TIMEOUT);
        }
    }

    /// A global quit was requested (`q` / Ctrl-C). Returns true when the app may
    /// quit right now; while a pull is running it instead arms the y/n quit
    /// confirm and returns false.
    pub fn request_quit(&mut self) -> bool {
        if self.pull.is_some() {
            self.confirm_quit = true;
            false
        } else {
            true
        }
    }

    /// Route a key to the armed quit confirm: `y`/`Y` cancels the pull, waits for
    /// the child to be reaped and returns true (quit); `n`/`N`/Esc disarms; every
    /// other key is swallowed (the prompt is modal on every tab).
    pub fn handle_quit_confirm_key(&mut self, key: KeyEvent) -> bool {
        match key.code {
            KeyCode::Char('y') | KeyCode::Char('Y') => {
                self.confirm_quit = false;
                self.shutdown_pull();
                true
            }
            KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => {
                self.confirm_quit = false;
                false
            }
            _ => false,
        }
    }

    /// Consume pull progress/result events (called each frame). On completion it
    /// reloads the registry so the new local model appears.
    pub fn drain_pull(&mut self) {
        enum Finished {
            Done(String),
            Failed(String),
            Cancelled,
        }
        let mut finished: Option<Finished> = None;
        if let Some(job) = &mut self.pull {
            loop {
                match job.handle.rx.try_recv() {
                    Ok(PullEvent::Progress { percent, line }) => {
                        job.percent = percent;
                        job.line = line;
                    }
                    Ok(PullEvent::Done) => {
                        finished = Some(Finished::Done(job.tag.clone()));
                        break;
                    }
                    Ok(PullEvent::Failed(err)) => {
                        finished = Some(Finished::Failed(format!("pull {}: {err}", job.tag)));
                        break;
                    }
                    Ok(PullEvent::Cancelled) => {
                        finished = Some(Finished::Cancelled);
                        break;
                    }
                    Err(mpsc::TryRecvError::Empty) => break,
                    Err(mpsc::TryRecvError::Disconnected) => {
                        finished =
                            Some(Finished::Failed(format!("pull {}: worker exited", job.tag)));
                        break;
                    }
                }
            }
        }
        if let Some(result) = finished {
            self.pull = None;
            match result {
                Finished::Done(tag) => {
                    self.toast_info(format!("pulled {tag}"));
                    self.reload(); // surface the new local model
                }
                Finished::Failed(msg) => self.toast_error(msg),
                Finished::Cancelled => self.toast_info("pull cancelled"),
            }
        }
        // The quit confirm is only meaningful while a pull is running.
        if self.pull.is_none() {
            self.confirm_quit = false;
        }
    }

    /// Request deletion of the selected local model — arms the y/n confirm.
    fn request_delete(&mut self) {
        match self.registry.selected_model() {
            Some((tag, true)) => self.confirm_delete = Some(tag),
            Some((_, false)) => self.toast_info("that model isn't downloaded"),
            None => self.toast_info("select a model row to delete"),
        }
    }

    /// Confirm the armed delete: spawn `rm <tag> --yes` and disarm.
    fn confirm_delete_yes(&mut self) {
        if let Some(tag) = self.confirm_delete.take() {
            if self.rm_cmd.is_some() {
                self.toast_info("a delete is already running");
                return;
            }
            self.rm_cmd = Some(model_actions::remove(tag.clone()));
            self.toast_info(format!("deleting {tag}\u{2026}"));
        }
    }

    /// Cancel the armed delete.
    fn cancel_delete(&mut self) {
        if self.confirm_delete.take().is_some() {
            self.toast_info("delete cancelled");
        }
    }

    /// Consume the rm outcome if it has arrived (called each frame). Reloads the
    /// registry on success so the removed model drops out of the list.
    pub fn drain_rm(&mut self) {
        let outcome = match &self.rm_cmd {
            Some(rx) => match rx.try_recv() {
                Ok(o) => o,
                Err(mpsc::TryRecvError::Empty) => return,
                Err(mpsc::TryRecvError::Disconnected) => {
                    RmOutcome::Failed("rm worker exited".into())
                }
            },
            None => return,
        };
        self.rm_cmd = None;
        match outcome {
            RmOutcome::Ok(msg) => {
                self.toast_info(msg);
                self.reload();
            }
            RmOutcome::Failed(msg) => self.toast_error(msg),
        }
    }

    /// True while a serve lifecycle command (start/stop/restart) is running.
    pub fn serve_cmd_running(&self) -> bool {
        self.serve_cmd.is_some()
    }

    /// Kick off a serve lifecycle command on a background thread. No-op (with a
    /// toast) if one is already in flight. The live serve up/down state updates
    /// separately via the DashboardWorker once the daemon responds.
    fn start_serve_command(&mut self, action: ServeAction) {
        if self.serve_cmd.is_some() {
            self.toast_info("a serve command is already running");
            return;
        }
        #[cfg(test)]
        self.serve_commands.push(action);
        self.serve_cmd = Some(serve_ctrl::run(action));
        self.serve_cmd_label = action.label().to_string();
        self.toast_info(format!("serve {}\u{2026}", action.label()));
    }

    /// Consume the serve command outcome if it has arrived (called each frame).
    /// Surfaces a toast and kicks an immediate dashboard re-probe so the new
    /// serve state shows up promptly.
    pub fn drain_serve_command(&mut self) {
        let outcome = match &self.serve_cmd {
            Some(rx) => match rx.try_recv() {
                Ok(o) => Some(o),
                Err(mpsc::TryRecvError::Empty) => return,
                Err(mpsc::TryRecvError::Disconnected) => Some(ServeOutcome::Failed(
                    "serve command thread exited unexpectedly".into(),
                )),
            },
            None => return,
        };
        self.serve_cmd = None;
        self.serve_cmd_label.clear();
        match outcome {
            Some(ServeOutcome::Ok(msg)) => self.toast_info(msg),
            Some(ServeOutcome::Failed(msg)) => self.toast_error(msg),
            None => {}
        }
        self.dashboard_worker.force_refresh();
    }

    /// Dashboard-tab keys: serve lifecycle controls.
    fn handle_dashboard_key(&mut self, key: KeyEvent) {
        // An armed stop/restart confirm is modal: only y / n / Esc act.
        if let Some(action) = self.confirm_serve {
            match key.code {
                KeyCode::Char('y') | KeyCode::Char('Y') => {
                    self.confirm_serve = None;
                    self.start_serve_command(action);
                }
                KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => {
                    self.confirm_serve = None;
                    self.toast_info("cancelled");
                }
                _ => {}
            }
            return;
        }
        match key.code {
            KeyCode::Char('s') => self.start_serve_command(ServeAction::Start),
            KeyCode::Char('x') => self.arm_serve_confirm(ServeAction::Stop),
            KeyCode::Char('R') => self.arm_serve_confirm(ServeAction::Restart),
            _ => {}
        }
    }

    /// Arm the y/n confirm for a stop/restart (unless one is already running).
    fn arm_serve_confirm(&mut self, action: ServeAction) {
        if self.serve_cmd.is_some() {
            self.toast_info("a serve command is already running");
            return;
        }
        self.toast = None;
        self.confirm_serve = Some(action);
    }

    /// Test seam: inject an in-flight serve-command receiver without spawning a
    /// real `hipfire` process, so `drain_serve_command` can be exercised in unit tests.
    #[cfg(test)]
    pub fn inject_serve_cmd(&mut self, rx: Receiver<ServeOutcome>) {
        self.serve_cmd = Some(rx);
        self.serve_cmd_label = "test".into();
    }

    /// Test seam: inject an in-flight pull without spawning a real process.
    /// Returns the pull's cancel flag so tests can observe a cancel request.
    #[cfg(test)]
    pub fn inject_pull(&mut self, tag: String, rx: Receiver<PullEvent>) -> Arc<AtomicBool> {
        let (handle, cancel) = PullHandle::detached(rx);
        self.pull = Some(PullJob {
            tag,
            handle,
            percent: None,
            line: String::new(),
        });
        cancel
    }

    /// Persist the current session UI state (active tab + expanded Models
    /// groups) for the next launch. Best-effort — a failed write is ignored
    /// (UI state is a convenience, not load-bearing).
    pub fn save_ui_state(&self) {
        let ui = UiState {
            active_tab: self.tab.title().to_string(),
            expanded_groups: self.registry.expanded_groups.iter().cloned().collect(),
        };
        let _ = ui.save(&self.paths.ui_state);
    }

    /// Raise a transient error toast (3.5s) over the footer.
    pub fn toast_error(&mut self, text: impl Into<String>) {
        self.toast = Some(Toast {
            text: text.into(),
            level: ToastLevel::Error,
            born: Instant::now(),
            ttl: Duration::from_millis(3500),
        });
    }

    /// Raise a transient info/confirmation toast (2.5s) over the footer.
    pub fn toast_info(&mut self, text: impl Into<String>) {
        self.toast = Some(Toast {
            text: text.into(),
            level: ToastLevel::Info,
            born: Instant::now(),
            ttl: Duration::from_millis(2500),
        });
    }

    /// Drop the toast once it has expired. Called each frame before render.
    pub fn expire_toast(&mut self) {
        if let Some(t) = &self.toast {
            if !t.is_live() {
                self.toast = None;
            }
        }
    }

    /// Mirror the latest snapshot produced by the background fetch thread into
    /// `self.dashboard`. Called every render frame from the UI thread — this is
    /// a cheap lock + clone and performs NO network / rocm-smi I/O. It also
    /// tells the worker whether the Dashboard tab is currently focused so the
    /// worker only fetches while the tab is visible.
    pub fn sync_dashboard(&mut self) {
        // The full fetch (stats, models, VRAM/GPU, SystemInfo) feeds the
        // Dashboard and System tabs only, so it is gated to them.
        self.dashboard_worker
            .set_active(self.tab == Tab::Dashboard || self.tab == Tab::System);
        if let Some(snap) = self.dashboard_worker.snapshot() {
            // Fold the worker's off-thread rocm-smi results into the live
            // status fields (gpu_lines) so the Home + System tabs render live
            // data without any synchronous probe.
            self.status.overlay_live(&snap);
            self.dashboard = Some(snap);
        }
        // Serve health is polled on every tab by its own cheap thread and
        // wins over the (possibly stale) full snapshot.
        let health = self.dashboard_worker.health();
        self.apply_health(health);
        if self.tab == Tab::Dashboard {
            self.live_metrics = self.dashboard_worker.metrics();
        }
    }

    /// Fold a `/health` result into the status, keep the stored full snapshot
    /// from contradicting it, and advance any queued Chat send.
    fn apply_health(&mut self, health: HealthSnapshot) {
        let was_up = self.status.serve_http_ok;
        self.status.overlay_health(&health);
        if health.checked {
            if let Some(dash) = self.dashboard.as_mut() {
                if dash.serve_up && !health.serve_up {
                    dash.mark_offline(&health);
                }
            }
            // On a flip, don't leave a visible full-snapshot tab waiting for
            // its next cadence tick.
            if was_up != health.serve_up && (self.tab == Tab::Dashboard || self.tab == Tab::System)
            {
                self.dashboard_worker.force_refresh();
            }
        }
        self.drive_queued_send();
    }

    /// Test seam: feed a health result without the background thread.
    #[cfg(test)]
    fn inject_health(&mut self, health: HealthSnapshot) {
        self.apply_health(health);
    }

    pub fn reload(&mut self) {
        // Reload ONLY the fast local data synchronously (config / registry /
        // local-models / serve.pid — all file I/O). The live serve health, VRAM
        // and system diagnostics come from the background DashboardWorker, so we
        // never run /health or lspci/rocm-smi on the UI thread here.
        self.config = self.load_config();
        // A reload re-reads config from disk, so drop any uncommitted enum preview
        // (5b) rather than leave it pointing at a now-stale value.
        self.settings_pending = None;
        // Preserve the user's expanded Models groups across an `r` refresh — a
        // fresh RegistryState::load() defaults them to empty, which would
        // collapse everything (and then a clean quit would persist that empty
        // set back to ui_state.json).
        let expanded = self.registry.expanded_groups.clone();
        self.registry = if self.hermetic {
            RegistryState::load_bundled(&self.paths)
        } else {
            RegistryState::load(&self.paths)
        };
        self.registry.expanded_groups = expanded;
        self.rebuild_settings();
        // Preserve the live overlay (serve health / gpu_lines) already mirrored
        // from the workers; only the local fields are refreshed.
        let mut status = StatusState::load_local(&self.paths);
        if let Some(snap) = self.dashboard.as_ref() {
            status.overlay_live(snap);
        }
        status.overlay_health(&self.dashboard_worker.health());
        self.status = status;
        // Keep the worker pointed at the (possibly changed) host/port and kick
        // an immediate non-blocking re-fetch; the snapshot updates on a later frame.
        self.dashboard_worker.update_config(self.config.clone());
        self.dashboard_worker.force_refresh();
        self.last_reload = "reloaded config, registry, models; refreshing serve status".into();
    }

    pub fn next_tab(&mut self) {
        // Leaving Settings discards an uncommitted enum preview (5b); leaving
        // the Dashboard disarms a pending stop/restart confirm.
        self.settings_pending = None;
        self.confirm_serve = None;
        let idx = Tab::ALL.iter().position(|t| *t == self.tab).unwrap_or(0);
        self.tab = Tab::ALL[(idx + 1) % Tab::ALL.len()];
    }

    pub fn prev_tab(&mut self) {
        self.settings_pending = None;
        self.confirm_serve = None;
        let idx = Tab::ALL.iter().position(|t| *t == self.tab).unwrap_or(0);
        self.tab = Tab::ALL[(idx + Tab::ALL.len() - 1) % Tab::ALL.len()];
    }

    /// Resolve the tab whose recorded header hit region contains `(col, row)`,
    /// or None when the position is not on the tab bar. The hit regions are
    /// refreshed every frame by the renderer (`ui::draw_header`).
    pub fn tab_at(&self, col: u16, row: u16) -> Option<Tab> {
        if row != self.tab_row_y {
            return None;
        }
        self.tab_hitboxes
            .iter()
            .find(|(start, end, _)| col >= *start && col < *end)
            .map(|(_, _, tab)| *tab)
    }

    pub fn handle_tab_key(&mut self, key: KeyEvent) {
        if self.scope_screen_open() {
            self.handle_scope_key(key);
            return;
        }
        match self.tab {
            Tab::Dashboard => self.handle_dashboard_key(key),
            Tab::Chat => self.handle_chat_key(key),
            Tab::Models => self.handle_models_key(key),
            Tab::Settings => self.handle_settings_key(key),
            Tab::System => self.handle_system_key(key),
            _ => {}
        }
    }

    fn handle_models_key(&mut self, key: KeyEvent) {
        // The pull chooser is modal: it owns every key while open.
        if self.pull_choice.is_some() {
            self.handle_pull_choice_key(key);
            return;
        }
        // A delete confirmation is modal: intercept y / n / Esc.
        if self.confirm_delete.is_some() {
            match key.code {
                KeyCode::Char('y') | KeyCode::Char('Y') => self.confirm_delete_yes(),
                KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => self.cancel_delete(),
                _ => {}
            }
            return;
        }
        let len = self.registry.visible_len().max(1);
        match key.code {
            KeyCode::Down | KeyCode::Char('j') => {
                self.registry.selected = (self.registry.selected + 1).min(len - 1);
            }
            KeyCode::Up | KeyCode::Char('k') => {
                self.registry.selected = self.registry.selected.saturating_sub(1);
            }
            KeyCode::Enter | KeyCode::Char(' ') => {
                // A remote (not-downloaded) row can't be the default model: saving
                // it would point serve at a file that doesn't exist. Local rows and
                // group headers keep their behavior.
                if let Some((tag, false)) = self.registry.selected_model() {
                    self.toast_info(format!("{tag} is not downloaded — p to pull"));
                    return;
                }
                if let Some(action) = self.registry.activate_selected() {
                    match action {
                        RegistryAction::ToggledGroup { name, expanded } => {
                            self.last_reload = format!(
                                "{} {name}",
                                if expanded { "expanded" } else { "collapsed" }
                            );
                        }
                        RegistryAction::SelectedModel { tag } => {
                            self.set_active_model(&tag);
                            self.chat.status = format!("model selected: {tag}");
                            // Persist default_model to ~/.hipfire/config.toml
                            // (read-modify-write, atomic). Was session-only.
                            match writer::write_raw_value(
                                &self.paths.config,
                                "default_model",
                                serde_json::Value::String(tag.clone()),
                            ) {
                                Ok(()) => {
                                    self.config.default_model = tag.clone();
                                    self.config
                                        .values
                                        .insert("default_model".into(), tag.clone());
                                    self.config.loaded_from_disk = true;
                                    self.last_reload = format!(
                                        "default_model = {tag} saved to ~/.hipfire/config.toml"
                                    );
                                    self.toast_info(format!("default model → {tag}"));
                                }
                                Err(err) => {
                                    self.last_reload = format!("default_model save failed: {err}");
                                    self.toast_error(format!("default_model save failed: {err}"));
                                }
                            }
                        }
                    }
                }
            }
            KeyCode::Right => {
                if let Some(name) = self.registry.expand_selected_group() {
                    self.last_reload = format!("expanded {name}");
                }
            }
            KeyCode::Left => {
                if let Some(name) = self.registry.collapse_selected_group() {
                    self.last_reload = format!("collapsed {name}");
                }
            }
            KeyCode::Char('p') => match self.registry.selected_model() {
                Some((tag, false)) => self.request_pull(tag),
                Some((_, true)) => self.toast_info("already downloaded"),
                None => self.toast_info("select a model row to pull"),
            },
            KeyCode::Char('c') => self.cancel_pull(),
            KeyCode::Char('d') => self.request_delete(),
            KeyCode::Char('o') => {
                let model = self.registry.selected_model().map(|(tag, _)| tag);
                self.open_overrides(model.as_deref());
            }
            _ => {}
        }
    }

    fn handle_chat_key(&mut self, key: KeyEvent) {
        // Typing, Backspace, Ctrl+O and scrolling stay live while a reply
        // streams; only Enter (send / command) is blocked until it finishes.
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('o') {
            self.chat.input.push('\n');
            self.chat.focus_input();
            return;
        }

        match key.code {
            KeyCode::Enter if self.chat.sending => {
                self.chat.status = "generation in progress — Esc stops it".into();
            }
            KeyCode::Enter => {
                let input = self.chat.input.trim().to_string();
                if input.is_empty() {
                    self.chat.focus_input();
                    return;
                }
                // A leading '/' is a command — except `//…`, which escapes to a
                // literal message whose text starts with a single '/'.
                if let Some(rest) = input.strip_prefix('/') {
                    if let Some(literal) = rest.strip_prefix('/') {
                        self.submit_chat_prompt(format!("/{literal}"));
                        self.chat.focus_input();
                        return;
                    }
                    // Commands act locally (no serve needed) and never send.
                    self.chat.input.clear();
                    self.handle_chat_command(rest);
                    self.chat.focus_input();
                    return;
                }
                self.submit_chat_prompt(input);
                self.chat.focus_input();
            }
            KeyCode::Backspace => {
                self.chat.input.pop();
                self.chat.focus_input();
            }
            KeyCode::Char(c) => {
                self.chat.input.push(c);
                self.chat.focus_input();
            }
            // Scroll is measured from the bottom: 0 follows the newest output.
            KeyCode::Up => self.chat.scroll_by(1),
            KeyCode::Down => self.chat.scroll_by(-1),
            KeyCode::PageUp => self.chat.scroll_by(self.chat.page_rows()),
            KeyCode::PageDown => self.chat.scroll_by(-self.chat.page_rows()),
            KeyCode::Home => self.chat.scroll_to_top(),
            KeyCode::End => self.chat.scroll = 0,
            _ => {}
        }
    }

    /// Enter on a prompt: send now when the serve is online, otherwise queue it
    /// (it sends automatically once `/health` turns online; Esc cancels).
    fn submit_chat_prompt(&mut self, prompt: String) {
        if self.queued_send.is_some() {
            self.toast_info("a message is already queued — Esc cancels it");
            return;
        }
        if self.status.serve_http_ok {
            self.send_chat(prompt);
            return;
        }
        // The prompt leaves the input box while queued; Esc puts it back.
        self.chat.input.clear();
        self.queue_send(QueuedKind::Prompt(prompt));
    }

    /// Hold `kind` until the serve is online and let `drive_queued_send` start
    /// the serve if that is safe.
    fn queue_send(&mut self, kind: QueuedKind) {
        self.chat.status = if self.status.health_known {
            "serve offline — starting hipfire serve -d; your message sends when it is online (Esc cancels)"
        } else {
            "checking serve health — your message sends when it is online (Esc cancels)"
        }
        .into();
        self.queued_send = Some(QueuedSend {
            kind,
            serve_start_handled: false,
        });
        self.drive_queued_send();
    }

    /// Whether a Chat send is waiting for the serve to come online.
    pub fn has_queued_send(&self) -> bool {
        self.queued_send.is_some()
    }

    /// Esc on Chat: drop the queued send. A queued prompt goes back into the
    /// input box. Returns whether anything was queued.
    pub fn cancel_queued_send(&mut self) -> bool {
        let Some(queued) = self.queued_send.take() else {
            return false;
        };
        self.restore_queued_input(queued);
        self.chat.status = "queued message cancelled".into();
        true
    }

    fn restore_queued_input(&mut self, queued: QueuedSend) {
        if let QueuedKind::Prompt(text) = queued.kind {
            self.chat.input = if self.chat.input.is_empty() {
                text
            } else {
                format!("{text}\n{}", self.chat.input)
            };
        }
        self.chat.focus_input();
    }

    /// Advance the queued send: send it once the serve is online; otherwise,
    /// only after the first `/health` result is in, start `hipfire serve -d`
    /// exactly once — unless a serve.pid is alive or the serve rejects us (a
    /// serve is answering or on its way), in which case just wait / report.
    fn drive_queued_send(&mut self) {
        if self.queued_send.is_none() || self.chat.sending {
            return;
        }
        if self.status.serve_http_ok {
            if let Some(queued) = self.queued_send.take() {
                match queued.kind {
                    QueuedKind::Prompt(text) => self.send_chat(text),
                    QueuedKind::Regen => self.regenerate_now(),
                }
            }
            return;
        }
        if !self.status.health_known {
            return;
        }
        if let Some(reason) = self.status.health_failure.clone() {
            if let Some(queued) = self.queued_send.take() {
                self.restore_queued_input(queued);
            }
            self.chat.status = reason.clone();
            self.toast_error(reason);
            return;
        }
        if self
            .queued_send
            .as_ref()
            .is_some_and(|q| q.serve_start_handled)
        {
            return;
        }
        if let Some(queued) = self.queued_send.as_mut() {
            queued.serve_start_handled = true;
        }
        // A serve started since the last reload (e.g. from another terminal)
        // must be seen before we decide to spawn one.
        self.status.refresh_pid(&self.paths);
        if self.status.serve_pid_alive {
            self.chat.status =
                "serve process is starting — your message sends when it is online (Esc cancels)"
                    .into();
            return;
        }
        match start_background_serve() {
            Ok(()) => {
                self.chat.status =
                    "serve offline — starting hipfire serve -d; your message sends when it is online (Esc cancels)"
                        .into();
                self.last_reload = "requested background serve start".into();
                self.status.refresh_pid(&self.paths);
                self.dashboard_worker.force_refresh();
            }
            Err(err) => {
                if let Some(queued) = self.queued_send.take() {
                    self.restore_queued_input(queued);
                }
                self.chat.status = format!("{err}");
                self.toast_error(format!("serve start failed: {err}"));
            }
        }
    }

    /// Push the user prompt, then spawn the stream worker (system prompt +
    /// sampling applied in spawn_stream).
    fn send_chat(&mut self, prompt: String) {
        self.chat.input.clear();
        self.chat.messages.push(ChatMessage {
            role: "user".into(),
            content: prompt,
            reasoning_content: None,
        });
        self.spawn_stream();
    }

    /// Push an empty assistant slot and spawn the stream worker for the current
    /// `messages` tail. Shared by send (after pushing the user turn) and
    /// regenerate (the user turn is already last). Applies the session system
    /// prompt + sampling overrides and resets per-generation stats.
    fn spawn_stream(&mut self) {
        // Drop stale [stopped] marks (a /regen popped the reply they pointed at),
        // dismiss any help note, and follow the new reply.
        self.chat.prune_stopped();
        self.chat.notice = None;
        self.chat.scroll = 0;
        self.chat.messages.push(ChatMessage {
            role: "assistant".into(),
            content: String::new(),
            reasoning_content: None,
        });
        self.chat.sending = true;
        self.chat.status = CHAT_STREAMING_STATUS.into();
        self.chat.gen_start = Some(Instant::now());
        self.chat.gen_model = self.active_model.clone();
        self.chat.gen_tokens = 0;
        self.chat.gen_usage = None;
        self.chat.last_stats = None;

        let (tx, rx) = mpsc::channel();
        self.chat.rx = Some(rx);
        // Fresh cancel flag per generation; Esc flips it (request_abort).
        self.chat.abort = Arc::new(AtomicBool::new(false));
        let abort = self.chat.abort.clone();
        let config = self.config.clone();
        let model = self.active_model.clone();
        let temp = self.chat.temp;
        let top_p = self.chat.top_p;
        let mut messages = self.chat.messages.clone();
        if let Some(last) = messages.last_mut() {
            let reasoning_empty = last
                .reasoning_content
                .as_deref()
                .map_or(true, |s| s.is_empty());
            if last.role == "assistant" && last.content.is_empty() && reasoning_empty {
                messages.pop();
            }
        }
        if !self.chat.system_prompt.is_empty() {
            messages.insert(
                0,
                ChatMessage {
                    role: "system".into(),
                    content: self.chat.system_prompt.clone(),
                    reasoning_content: None,
                },
            );
        }
        // Endpoint (and bearer key-file read) resolves on the worker thread so
        // file I/O stays off the UI thread; failures surface as a chat error.
        thread::spawn(move || match config.service_endpoint() {
            Ok(endpoint) => {
                let _ = stream_chat(&endpoint, &model, &messages, temp, top_p, tx, abort);
            }
            Err(error) => {
                let _ = tx.send(ChatEvent::Error(error));
            }
        });
    }

    /// Whether the conversation tail is `[.., user]` or `[.., user, assistant]`
    /// — the only shapes /regen and /edit can act on. Checked BEFORE any mutation
    /// so a malformed or freshly-loaded history is never corrupted.
    fn tail_is_redoable(&self) -> bool {
        let msgs = &self.chat.messages;
        let n = msgs.len();
        match msgs.last().map(|m| m.role.as_str()) {
            Some("assistant") => n >= 2 && msgs[n - 2].role == "user",
            Some("user") => true,
            _ => false,
        }
    }

    /// `/regen` — drop the last assistant reply and re-stream from the last user
    /// turn (no new user message).
    fn regenerate(&mut self) {
        if self.chat.sending {
            self.toast_info("generation in progress");
            return;
        }
        if !self.tail_is_redoable() {
            self.toast_error("nothing to regenerate");
            return;
        }
        // Check serve readiness BEFORE the destructive pop, so an offline serve
        // doesn't lose the existing reply.
        if self.queued_send.is_some() {
            self.toast_info("a message is already queued — Esc cancels it");
            return;
        }
        if !self.status.serve_http_ok {
            self.queue_send(QueuedKind::Regen);
            return;
        }
        self.regenerate_now();
    }

    /// Drop the last assistant reply (if any) and re-stream the last user turn.
    /// The serve is known online; the tail is re-validated because a queued
    /// regen may have waited while the conversation changed.
    fn regenerate_now(&mut self) {
        if !self.tail_is_redoable() {
            self.toast_error("nothing to regenerate");
            return;
        }
        if self.chat.messages.last().map(|m| m.role.as_str()) == Some("assistant") {
            self.chat.messages.pop();
        }
        self.spawn_stream();
    }

    /// `/edit` — pull the last user turn back into the input (dropping its reply)
    /// so it can be edited and re-sent.
    fn edit_last(&mut self) {
        if self.chat.sending {
            self.toast_info("generation in progress");
            return;
        }
        if !self.tail_is_redoable() {
            self.toast_error("no message to edit");
            return;
        }
        if self.chat.messages.last().map(|m| m.role.as_str()) == Some("assistant") {
            self.chat.messages.pop();
        }
        let last = self
            .chat
            .messages
            .pop()
            .expect("tail validated as user turn");
        self.chat.prune_stopped();
        self.chat.input = last.content;
        self.chat.reset_stats();
        self.chat.focus_input();
        self.chat.status = "editing last message — Enter to resend".into();
    }

    /// `/copy` — stage the last non-empty assistant reply for the OSC52 clipboard
    /// emit (performed by the event loop after render).
    fn copy_last(&mut self) {
        match self
            .chat
            .messages
            .iter()
            .rev()
            .find(|m| m.role == "assistant" && !m.content.is_empty())
        {
            Some(m) => {
                // Cap the OSC52 payload — a very large escape can stall or be
                // rejected by the terminal.
                const MAX_COPY: usize = 100 * 1024;
                if m.content.len() > MAX_COPY {
                    self.toast_error(format!(
                        "reply too large to copy ({} KB > 100 KB)",
                        m.content.len() / 1024
                    ));
                } else {
                    self.pending_clipboard = Some(m.content.clone());
                    self.chat.status = "copied last reply to clipboard".into();
                }
            }
            None => self.toast_error("no reply to copy"),
        }
    }

    /// Record a finished request in the inspector ring (newest first, capped).
    fn log_request(&mut self, stats: GenStats, secs: f64) {
        self.request_log.insert(
            0,
            RequestRecord {
                model: self.chat.gen_model.clone(),
                tokens: stats.tokens,
                tps: stats.tps,
                secs,
                exact: stats.exact,
            },
        );
        self.request_log.truncate(REQUEST_LOG_CAP);
    }

    /// Whether a doctor run is in flight (for the footer + re-trigger guard).
    pub fn doctor_running(&self) -> bool {
        self.doctor_cmd.is_some()
    }

    /// Start a `hipfire diag --json` doctor run on a background thread (System
    /// tab `d`). No-op if one is already running.
    fn run_doctor(&mut self) {
        if self.doctor_cmd.is_some() {
            self.toast_info("doctor already running");
            return;
        }
        self.doctor_cmd = Some(doctor::run());
        self.toast_info("running hipfire diag\u{2026}");
    }

    /// Consume the doctor report when it arrives (called each frame).
    pub fn drain_doctor(&mut self) {
        let report = match &self.doctor_cmd {
            Some(rx) => match rx.try_recv() {
                Ok(r) => r,
                Err(mpsc::TryRecvError::Empty) => return,
                Err(mpsc::TryRecvError::Disconnected) => DoctorReport {
                    checks: Vec::new(),
                    error: Some("doctor worker exited".into()),
                },
            },
            None => return,
        };
        self.doctor_cmd = None;
        match &report.error {
            Some(err) => self.toast_error(format!("doctor: {err}")),
            None => {
                let fails = report.checks.iter().filter(|c| !c.ok).count();
                if fails == 0 {
                    self.toast_info("doctor: all checks passed");
                } else {
                    self.toast_error(format!("doctor: {fails} check(s) failed"));
                }
            }
        }
        self.doctor = Some(report);
    }

    /// System-tab keys: `d` runs the doctor.
    fn handle_system_key(&mut self, key: KeyEvent) {
        if let KeyCode::Char('d') = key.code {
            self.run_doctor();
        }
    }

    /// Dispatch a Chat slash command (the input after the leading '/').
    fn handle_chat_command(&mut self, cmd: &str) {
        let mut parts = cmd.splitn(2, char::is_whitespace);
        let name = parts.next().unwrap_or("");
        let arg = parts.next().unwrap_or("").trim();
        match name {
            "model" | "m" => {
                if arg.is_empty() {
                    self.chat.status = format!("model: {} (use /model <tag>)", self.active_model);
                } else if self.registry.has_model(arg) {
                    self.set_active_model(arg);
                    self.chat.status = format!("model -> {arg}");
                } else {
                    self.toast_error(format!("unknown model: {arg}"));
                }
            }
            "system" | "sys" => {
                self.chat.system_prompt = arg.to_string();
                self.chat.status = if arg.is_empty() {
                    "system prompt cleared".into()
                } else {
                    format!("system prompt set ({} chars)", arg.chars().count())
                };
            }
            "temp" => self.set_sampling(true, arg),
            "top_p" => self.set_sampling(false, arg),
            "clear" => {
                self.chat.messages.clear();
                self.chat.stopped.clear();
                self.chat.notice = None;
                self.chat.scroll = 0;
                self.chat.reset_stats();
                self.chat.status = "conversation cleared".into();
            }
            "save" => self.chat_save(arg),
            "load" => self.chat_load(arg),
            "sessions" | "ls" => self.chat_sessions(),
            "delete" | "del" => self.chat_delete(arg),
            "regen" | "retry" => self.regenerate(),
            "edit" => self.edit_last(),
            "copy" => self.copy_last(),
            "help" | "" => {
                self.chat.notice = Some(CHAT_HELP.to_string());
                self.chat.scroll = 0;
            }
            other => self.toast_error(format!("unknown command: /{other} (try /help)")),
        }
    }

    /// `/save <name>` — persist the current conversation as a named session.
    fn chat_save(&mut self, name: &str) {
        if name.is_empty() {
            self.toast_error("usage: /save <name>");
            return;
        }
        if self.chat.messages.is_empty() {
            self.toast_info("nothing to save (empty conversation)");
            return;
        }
        let mut h = ChatHistory::load(&self.paths.chat_history);
        h.upsert(name, self.chat.messages.clone());
        match h.save(&self.paths.chat_history) {
            Ok(()) => self.chat.status = format!("saved session '{name}'"),
            Err(e) => self.toast_error(format!("save failed: {e}")),
        }
    }

    /// `/load <name>` — replace the conversation with a saved session.
    fn chat_load(&mut self, name: &str) {
        if name.is_empty() {
            self.toast_error("usage: /load <name> (see /sessions)");
            return;
        }
        let h = ChatHistory::load(&self.paths.chat_history);
        match h.get(name) {
            Some(msgs) => {
                self.chat.messages = msgs.to_vec();
                self.chat.stopped.clear();
                self.chat.notice = None;
                self.chat.scroll = 0;
                self.chat.reset_stats();
                self.chat.status =
                    format!("loaded '{name}' ({} messages)", self.chat.messages.len());
            }
            None => self.toast_error(format!("no session '{name}' (see /sessions)")),
        }
    }

    /// `/sessions` — list saved session names.
    fn chat_sessions(&mut self) {
        let h = ChatHistory::load(&self.paths.chat_history);
        let names = h.names();
        self.chat.status = if names.is_empty() {
            "no saved sessions (use /save <name>)".into()
        } else {
            format!("sessions: {}", names.join(", "))
        };
    }

    /// `/delete <name>` — remove a saved session.
    fn chat_delete(&mut self, name: &str) {
        if name.is_empty() {
            self.toast_error("usage: /delete <name>");
            return;
        }
        let mut h = ChatHistory::load(&self.paths.chat_history);
        if h.remove(name) {
            match h.save(&self.paths.chat_history) {
                Ok(()) => self.chat.status = format!("deleted session '{name}'"),
                Err(e) => self.toast_error(format!("delete failed: {e}")),
            }
        } else {
            self.toast_error(format!("no session '{name}'"));
        }
    }

    /// Set or clear a per-session sampling override. `is_temp` selects
    /// temperature (range 0..=2) vs top_p (range 0..=1). An empty/"off" arg
    /// clears the override (serve default).
    fn set_sampling(&mut self, is_temp: bool, arg: &str) {
        let (label, lo, hi) = if is_temp {
            ("temperature", 0.0, 2.0)
        } else {
            ("top_p", 0.0, 1.0)
        };
        if arg.is_empty() || arg == "off" {
            if is_temp {
                self.chat.temp = None;
            } else {
                self.chat.top_p = None;
            }
            self.chat.status = format!("{label}: serve default");
            return;
        }
        match arg.parse::<f64>() {
            Ok(v) if (lo..=hi).contains(&v) => {
                if is_temp {
                    self.chat.temp = Some(v);
                } else {
                    self.chat.top_p = Some(v);
                }
                self.chat.status = format!("{label}: {v}");
            }
            _ => self.toast_error(format!("{label} must be a number in {lo}..={hi}")),
        }
    }

    pub fn drain_chat_events(&mut self) {
        let mut finished = false;
        if let Some(rx) = self.chat.rx.take() {
            while let Ok(event) = rx.try_recv() {
                match event {
                    ChatEvent::Reasoning(text) => {
                        self.chat.end_prefill();
                        if let Some(last) = self.chat.messages.last_mut() {
                            let rc = last.reasoning_content.get_or_insert_with(String::new);
                            rc.push_str(&text);
                        }
                        self.chat.gen_tokens += 1;
                    }
                    ChatEvent::Content(text) => {
                        self.chat.end_prefill();
                        if let Some(last) = self.chat.messages.last_mut() {
                            last.content.push_str(&text);
                        }
                        self.chat.gen_tokens += 1;
                    }
                    ChatEvent::PrefillProgress { done, total } => {
                        self.chat.prefill = Some((done, total));
                        let elapsed = self
                            .chat
                            .gen_start
                            .map_or(std::time::Duration::ZERO, |start| start.elapsed());
                        self.chat.status = prefill_status_text(done, total, elapsed);
                    }
                    ChatEvent::Usage { completion_tokens } => {
                        self.chat.gen_usage = Some(completion_tokens);
                    }
                    ChatEvent::Done => {
                        // Normalize empty reasoning to None so serialization omits it.
                        if let Some(last) = self.chat.messages.last_mut() {
                            if last
                                .reasoning_content
                                .as_deref()
                                .is_some_and(|s| s.is_empty())
                            {
                                last.reasoning_content = None;
                            }
                        }
                        // Record throughput + drop a zero-delta empty assistant.
                        if let Some((stats, secs)) = self.chat.finalize_generation() {
                            self.log_request(stats, secs);
                        }
                        self.chat.status = "ready".into();
                        self.chat.sending = false;
                        finished = true;
                    }
                    ChatEvent::Error(err) => {
                        // Normalize empty reasoning as with Done.
                        if let Some(last) = self.chat.messages.last_mut() {
                            if last
                                .reasoning_content
                                .as_deref()
                                .is_some_and(|s| s.is_empty())
                            {
                                last.reasoning_content = None;
                            }
                        }
                        // finalize drops the trailing empty assistant so a failed
                        // request doesn't read as a blank reply; surface the cause.
                        // An errored partial is NOT logged to the inspector
                        // (incomplete + misleading tok/s, same rule as abort).
                        let _ = self.chat.finalize_generation();
                        self.toast_error(format!("chat error: {err}"));
                        self.chat.status = format!("error: {err}");
                        self.chat.sending = false;
                        finished = true;
                    }
                }
            }

            if !finished {
                self.chat.rx = Some(rx);
            }
        }
    }
}

/// Chat status while a reply streams (also what the prefill readout yields to).
const CHAT_STREAMING_STATUS: &str = "streaming from hipfire serve";

/// Chat status while the server prefills the prompt, e.g.
/// `prefilling 512/2048 (25%) · ETA 6s`. `elapsed` is the time since the
/// request was sent; the ETA extrapolates the rate so far (`?` until one exists).
fn prefill_status_text(done: u64, total: u64, elapsed: std::time::Duration) -> String {
    let eta = match hipfire_client::prefill_eta_secs(done, total, elapsed) {
        Some(secs) => format!("{secs:.0}"),
        None => "?".to_owned(),
    };
    format!(
        "prefilling {done}/{total} ({}%) · ETA {eta}s",
        hipfire_client::prefill_percent(done, total)
    )
}

/// Help note shown in the Chat transcript by `/help` (never a message).
const CHAT_HELP: &str = concat!(
    "Commands: /model [tag] · /system [text] · /temp [x] · /top_p [x] · /clear\n",
    "          /save [name] · /load [name] · /sessions · /delete [name] · /regen · /edit · /copy\n",
    "          //text sends a message that starts with a literal '/'\n",
    "Keys:     Enter send · Ctrl+O newline · Up/Down/PgUp/PgDn scroll · End follow · Esc stop",
);

/// Throughput of one completed generation over the wall-clock generation time.
/// `exact` is true when `tokens` is the server-reported completion token count
/// (stream usage chunk); otherwise it counts streamed SSE chunks, which can be
/// fewer or more than tokens, and is labelled as such.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GenStats {
    pub tokens: usize,
    pub tps: f64,
    pub exact: bool,
}

impl GenStats {
    /// Unit for `tokens` / `tps`: `tok` when server-reported, else `chunks`.
    pub fn unit(&self) -> &'static str {
        if self.exact {
            "tok"
        } else {
            "chunks"
        }
    }
}

/// One finished chat request, for the System-tab request inspector. All fields
/// are measured locally from the TUI's own generation — never fabricated.
#[derive(Clone, Debug, PartialEq)]
pub struct RequestRecord {
    pub model: String,
    pub tokens: usize,
    pub tps: f64,
    pub secs: f64,
    /// `tokens` is the server-reported count (else streamed chunks).
    pub exact: bool,
}

impl RequestRecord {
    pub fn unit(&self) -> &'static str {
        if self.exact {
            "tok"
        } else {
            "chunks"
        }
    }
}

/// Cap on retained request records (newest kept).
const REQUEST_LOG_CAP: usize = 20;

/// Geometry of the last drawn chat transcript, recorded by the renderer (which
/// only has `&App`) so the key handler can page and clamp scrolling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChatView {
    /// Transcript viewport height in rows.
    pub viewport: u16,
    /// Largest useful `scroll` (wrapped lines above the viewport when at top).
    pub max_scroll: u16,
}

pub struct ChatState {
    pub input: String,
    pub messages: Vec<ChatMessage>,
    pub status: String,
    pub sending: bool,
    /// Transcript scroll in rows measured from the bottom: 0 follows the newest
    /// output; larger values show older text.
    pub scroll: u16,
    /// Geometry of the last rendered transcript (None before the first draw).
    pub view: Cell<Option<ChatView>>,
    /// Local note shown after the messages (e.g. `/help`). Never a message: not
    /// sent upstream, saved, or considered by /regen or /edit.
    pub notice: Option<String>,
    /// Indices into `messages` of replies the user aborted with Esc (render-only
    /// `[stopped]` marker; the message content is untouched).
    pub stopped: Vec<usize>,
    /// Session system prompt, prepended to the request when non-empty (/system).
    pub system_prompt: String,
    /// Per-session sampling overrides (/temp, /top_p); None = serve default.
    pub temp: Option<f64>,
    pub top_p: Option<f64>,
    /// Wall-clock start of the in-flight generation (for tok/s).
    gen_start: Option<Instant>,
    /// Model the in-flight generation was sent to (captured at send, since
    /// active_model can change mid-stream via /model).
    gen_model: String,
    /// Streamed deltas (SSE chunks) counted so far this generation.
    gen_tokens: usize,
    /// Server-reported completion tokens for this generation, once the final
    /// usage chunk has arrived.
    gen_usage: Option<usize>,
    /// Latest `(done, total)` prompt-token prefill progress of the in-flight
    /// generation; cleared when the first content/reasoning arrives.
    pub prefill: Option<(u64, u64)>,
    /// Stats of the most recent completed generation (shown under the reply).
    pub last_stats: Option<GenStats>,
    rx: Option<Receiver<ChatEvent>>,
    input_focused: bool,
    // Set true to ask the in-flight stream thread to stop (checked per line).
    abort: Arc<AtomicBool>,
}

impl Default for ChatState {
    fn default() -> Self {
        Self {
            input: String::new(),
            messages: Vec::new(),
            status: "ready".into(),
            sending: false,
            scroll: 0,
            view: Cell::new(None),
            notice: None,
            stopped: Vec::new(),
            system_prompt: String::new(),
            temp: None,
            top_p: None,
            gen_start: None,
            gen_model: String::new(),
            gen_tokens: 0,
            gen_usage: None,
            prefill: None,
            last_stats: None,
            rx: None,
            input_focused: true,
            abort: Arc::new(AtomicBool::new(false)),
        }
    }
}

impl ChatState {
    /// Ask an in-flight generation to stop, and optimistically free the tab. A
    /// hung read may never deliver Done, so we don't wait on the worker: clear
    /// `sending`, drop the receiver, and finalize. The detached thread exits when
    /// its socket errors (bounded by the read timeout); its late events land on a
    /// dropped receiver. The partial reply already streamed is kept and marked
    /// `[stopped]` in the transcript.
    pub fn request_abort(&mut self) {
        if self.sending {
            self.abort.store(true, Ordering::Relaxed);
            self.sending = false;
            self.rx = None;
            let _ = self.finalize_generation(); // aborted = incomplete, not logged
            if self.messages.last().is_some_and(|m| m.role == "assistant") {
                self.stopped.push(self.messages.len() - 1);
            }
            self.status = "stopped".into();
        }
    }

    /// End-of-generation cleanup, shared by Done / Error / abort: record tok/s
    /// (only if any tokens actually streamed) and drop a trailing empty assistant
    /// bubble left by a zero-delta finish (immediate [DONE], empty deltas, or
    /// abort-before-first-token) so it isn't rendered or sent in the next turn.
    fn finalize_generation(&mut self) -> Option<(GenStats, f64)> {
        let mut recorded = None;
        self.prefill = None;
        if let Some(start) = self.gen_start.take() {
            let (count, exact) = match self.gen_usage {
                Some(n) => (n, true),
                None => (self.gen_tokens, false),
            };
            if count > 0 {
                let secs = start.elapsed().as_secs_f64();
                let tps = if secs > 0.0 { count as f64 / secs } else { 0.0 };
                let stats = GenStats {
                    tokens: count,
                    tps,
                    exact,
                };
                self.last_stats = Some(stats);
                recorded = Some((stats, secs));
            }
        }
        if let Some(last) = self.messages.last() {
            let reasoning_empty = last
                .reasoning_content
                .as_deref()
                .map_or(true, |s| s.is_empty());
            if last.role == "assistant" && last.content.is_empty() && reasoning_empty {
                self.messages.pop();
            }
        }
        self.prune_stopped();
        recorded
    }

    /// The first content/reasoning arrived: the prefill readout is over.
    fn end_prefill(&mut self) {
        if self.prefill.take().is_some() {
            self.status = CHAT_STREAMING_STATUS.into();
        }
    }

    /// Forget `[stopped]` marks whose message no longer exists.
    fn prune_stopped(&mut self) {
        let n = self.messages.len();
        self.stopped.retain(|&i| i < n);
    }

    /// Move the transcript view by `delta` rows (positive = older), clamped to
    /// the last measured extent so scrolling past the top doesn't accumulate.
    fn scroll_by(&mut self, delta: i32) {
        let max = self.view.get().map_or(u16::MAX, |v| v.max_scroll);
        let next = (i32::from(self.scroll) + delta).clamp(0, i32::from(max));
        self.scroll = next as u16;
    }

    /// Rows moved by PageUp/PageDown: one transcript viewport (at least 1).
    fn page_rows(&self) -> i32 {
        i32::from(self.view.get().map_or(1, |v| v.viewport).max(1))
    }

    /// Home: jump to the oldest text (no-op until the view has been measured).
    fn scroll_to_top(&mut self) {
        if let Some(v) = self.view.get() {
            self.scroll = v.max_scroll;
        }
    }

    /// Clear any displayed throughput + in-flight counters. Called when the
    /// conversation the stats referred to is replaced (/clear, /load, /edit).
    fn reset_stats(&mut self) {
        self.last_stats = None;
        self.gen_start = None;
        self.prefill = None;
        self.gen_tokens = 0;
        self.gen_usage = None;
    }

    pub fn focus_input(&mut self) {
        self.input_focused = true;
    }

    pub fn blur_input(&mut self) {
        self.input_focused = false;
    }

    pub fn is_input_focused(&self) -> bool {
        self.input_focused
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{hermetic_app_with_config, TempHome};

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    /// Build a hermetic App (isolated temp home with an empty `config.toml`),
    /// so the settings tests never touch the user's real ~/.hipfire. The
    /// returned [`TempHome`] derefs to the home path and removes it on drop.
    fn test_app() -> (App, TempHome) {
        let (mut app, home) = hermetic_app_with_config("schema_version = 1\n");
        app.tab = Tab::Settings;
        (app, home)
    }

    #[test]
    fn tab_at_maps_column_to_tab() {
        let (mut app, dir) = test_app();
        app.tab_row_y = 3;
        app.tab_hitboxes = vec![
            (0, 6, Tab::Home),
            (9, 20, Tab::Dashboard),
            (23, 29, Tab::Chat),
        ];
        assert_eq!(app.tab_at(3, 3), Some(Tab::Home));
        assert_eq!(app.tab_at(5, 3), Some(Tab::Home)); // end-exclusive: 5 < 6
        assert_eq!(app.tab_at(6, 3), None); // divider gap between cells
        assert_eq!(app.tab_at(12, 3), Some(Tab::Dashboard));
        assert_eq!(app.tab_at(3, 4), None); // wrong row
        assert_eq!(app.tab_at(200, 3), None); // past the last tab
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn ui_state_persists_tab_and_expanded_groups() {
        let (mut app, dir) = test_app();
        app.paths.ui_state = dir.join("ui_state.json");
        app.tab = Tab::Models;
        app.registry.expanded_groups.insert("qwen".into());
        app.save_ui_state();
        let restored = crate::hipfire::ui_state::UiState::load(&app.paths.ui_state);
        assert_eq!(restored.active_tab, "Models");
        assert!(restored.expanded_groups.contains(&"qwen".to_string()));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn drain_serve_command_idle_is_noop() {
        let (mut app, dir) = test_app();
        assert!(!app.serve_cmd_running());
        app.drain_serve_command();
        assert!(app.toast.is_none(), "no in-flight command -> no toast");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn drain_serve_command_consumes_outcome_and_toasts() {
        let (mut app, dir) = test_app();
        let (tx, rx) = std::sync::mpsc::channel();
        tx.send(ServeOutcome::Ok("serve start: done".into()))
            .unwrap();
        app.inject_serve_cmd(rx);
        assert!(app.serve_cmd_running());
        app.drain_serve_command();
        assert!(
            !app.serve_cmd_running(),
            "outcome clears the in-flight command"
        );
        assert!(app.toast.is_some(), "outcome raises a toast");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn delete_confirm_can_be_cancelled() {
        let (mut app, dir) = test_app();
        app.confirm_delete = Some("qwen3.5:9b".into());
        app.cancel_delete();
        assert!(
            app.confirm_delete.is_none(),
            "Esc/n clears the armed delete"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn drain_pull_and_rm_idle_are_noops() {
        let (mut app, dir) = test_app();
        app.drain_pull();
        app.drain_rm();
        assert!(app.toast.is_none(), "no in-flight pull/rm -> no toast");
        assert!(app.pull.is_none());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn drain_pull_updates_progress() {
        let (mut app, dir) = test_app();
        let (tx, rx) = std::sync::mpsc::channel();
        tx.send(PullEvent::Progress {
            percent: Some(42.0),
            line: "[bar] 42.0% 8 MB/s".into(),
        })
        .unwrap();
        app.inject_pull("qwen3.5:9b".into(), rx);
        app.drain_pull();
        let job = app
            .pull
            .as_ref()
            .expect("pull stays active during progress");
        assert_eq!(job.percent, Some(42.0));
        assert_eq!(job.line, "[bar] 42.0% 8 MB/s");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn c_key_cancels_running_pull_and_cancelled_event_clears_it() {
        let (mut app, dir) = test_app();
        app.tab = Tab::Models;
        let (tx, rx) = std::sync::mpsc::channel();
        let cancel = app.inject_pull("qwen3.5:9b".into(), rx);
        app.handle_tab_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::NONE));
        assert!(cancel.load(Ordering::SeqCst), "c sets the cancel flag");
        assert!(app.pull.is_some(), "the job stays until the worker answers");
        tx.send(PullEvent::Cancelled).unwrap();
        app.drain_pull();
        assert!(app.pull.is_none());
        let toast = app.toast.as_ref().expect("cancel toast");
        assert_eq!(toast.text, "pull cancelled");
        assert_eq!(toast.level, ToastLevel::Info, "cancel is not an error");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn c_key_without_pull_says_so() {
        let (mut app, dir) = test_app();
        app.tab = Tab::Models;
        app.handle_tab_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::NONE));
        assert_eq!(app.toast.as_ref().unwrap().text, "no pull running");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn quit_confirm_disarms_when_the_pull_finishes() {
        let (mut app, dir) = test_app();
        let (tx, rx) = std::sync::mpsc::channel();
        app.inject_pull("qwen3.5:9b".into(), rx);
        assert!(!app.request_quit());
        assert!(app.confirm_quit);
        tx.send(PullEvent::Failed("boom".into())).unwrap();
        app.drain_pull();
        assert!(!app.confirm_quit, "no pull left to confirm about");
        assert!(app.request_quit(), "idle quit is immediate");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn chat_slash_commands_set_session_state() {
        let (mut app, dir) = test_app();
        app.tab = Tab::Chat;

        // /system sets then clears the prompt.
        app.handle_chat_command("system You are a helpful pirate.");
        assert_eq!(app.chat.system_prompt, "You are a helpful pirate.");
        app.handle_chat_command("system");
        assert!(app.chat.system_prompt.is_empty());

        // /temp + /top_p validate range and accept off.
        app.handle_chat_command("temp 0.7");
        assert_eq!(app.chat.temp, Some(0.7));
        app.handle_chat_command("temp 9"); // out of 0..=2 -> rejected, unchanged
        assert_eq!(app.chat.temp, Some(0.7));
        app.handle_chat_command("temp off");
        assert_eq!(app.chat.temp, None);
        app.handle_chat_command("top_p 0.9");
        assert_eq!(app.chat.top_p, Some(0.9));
        app.handle_chat_command("top_p 5"); // out of 0..=1 -> rejected
        assert_eq!(app.chat.top_p, Some(0.9));

        // /clear empties the conversation.
        app.chat.messages.push(ChatMessage {
            role: "user".into(),
            content: "hi".into(),
            reasoning_content: None,
        });
        app.handle_chat_command("clear");
        assert!(app.chat.messages.is_empty());

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn chat_unknown_command_toasts_does_not_panic() {
        let (mut app, dir) = test_app();
        app.handle_chat_command("definitely-not-a-command");
        assert!(
            app.toast.is_some(),
            "unknown command surfaces an error toast"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn chat_save_load_sessions_round_trip() {
        let (mut app, dir) = test_app();
        app.paths.chat_history = dir.join("chat_history.json");
        app.tab = Tab::Chat;
        app.chat.messages = vec![
            ChatMessage {
                role: "user".into(),
                content: "hi".into(),
                reasoning_content: None,
            },
            ChatMessage {
                role: "assistant".into(),
                content: "hello".into(),
                reasoning_content: None,
            },
        ];
        app.handle_chat_command("save debug");
        app.handle_chat_command("clear");
        assert!(app.chat.messages.is_empty());
        app.handle_chat_command("load debug");
        assert_eq!(app.chat.messages.len(), 2);
        assert_eq!(app.chat.messages[1].content, "hello");
        app.handle_chat_command("sessions");
        assert!(app.chat.status.contains("debug"));
        app.handle_chat_command("delete debug");
        app.handle_chat_command("load debug");
        assert!(
            app.toast.is_some(),
            "loading a deleted session surfaces an error"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn chat_edit_last_pulls_user_message_into_input() {
        let (mut app, dir) = test_app();
        app.tab = Tab::Chat;
        app.chat.messages = vec![
            ChatMessage {
                role: "user".into(),
                content: "first".into(),
                reasoning_content: None,
            },
            ChatMessage {
                role: "assistant".into(),
                content: "reply".into(),
                reasoning_content: None,
            },
        ];
        app.handle_chat_command("edit");
        assert_eq!(app.chat.input, "first");
        assert!(
            app.chat.messages.is_empty(),
            "the edited turn and its reply are removed"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn chat_copy_stages_last_reply_for_clipboard() {
        let (mut app, dir) = test_app();
        app.tab = Tab::Chat;
        app.chat.messages = vec![
            ChatMessage {
                role: "user".into(),
                content: "q".into(),
                reasoning_content: None,
            },
            ChatMessage {
                role: "assistant".into(),
                content: "the answer".into(),
                reasoning_content: None,
            },
        ];
        app.handle_chat_command("copy");
        assert_eq!(app.pending_clipboard.as_deref(), Some("the answer"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn chat_regenerate_without_history_toasts() {
        let (mut app, dir) = test_app();
        app.tab = Tab::Chat;
        app.handle_chat_command("regen");
        assert!(
            app.toast.is_some(),
            "regenerate with no turn to redo toasts"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn regenerate_offline_preserves_the_reply() {
        let (mut app, dir) = test_app();
        app.tab = Tab::Chat;
        app.status.serve_http_ok = false; // serve down
        app.chat.messages = vec![
            ChatMessage {
                role: "user".into(),
                content: "q".into(),
                reasoning_content: None,
            },
            ChatMessage {
                role: "assistant".into(),
                content: "a".into(),
                reasoning_content: None,
            },
        ];
        app.regenerate();
        // Serve-readiness is checked BEFORE the destructive pop, so the reply
        // survives an offline regen attempt.
        assert_eq!(
            app.chat.messages.len(),
            2,
            "reply preserved when serve offline"
        );
        assert_eq!(app.chat.messages[1].content, "a");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn zero_delta_done_drops_empty_assistant_bubble() {
        let (mut app, dir) = test_app();
        app.tab = Tab::Chat;
        app.chat.messages = vec![
            ChatMessage {
                role: "user".into(),
                content: "hi".into(),
                reasoning_content: None,
            },
            ChatMessage {
                role: "assistant".into(),
                content: String::new(),
                reasoning_content: None, // empty slot, no deltas streamed
            },
        ];
        let (tx, rx) = std::sync::mpsc::channel();
        app.chat.rx = Some(rx);
        app.chat.sending = true;
        app.chat.gen_start = Some(Instant::now());
        tx.send(ChatEvent::Done).unwrap();
        app.drain_chat_events();
        assert_eq!(app.chat.messages.len(), 1, "empty assistant bubble dropped");
        assert!(
            app.chat.last_stats.is_none(),
            "no stats when zero tokens streamed"
        );
        assert!(!app.chat.sending);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn prefill_text_reports_tokens_percent_and_eta() {
        assert_eq!(
            prefill_status_text(512, 2048, std::time::Duration::from_secs(2)),
            "prefilling 512/2048 (25%) · ETA 6s"
        );
        assert_eq!(
            prefill_status_text(0, 2048, std::time::Duration::from_secs(1)),
            "prefilling 0/2048 (0%) · ETA ?s"
        );
    }

    #[test]
    fn prefill_progress_shows_until_first_content_then_streams() {
        let (mut app, dir) = test_app();
        app.tab = Tab::Chat;
        app.chat.messages = vec![ChatMessage {
            role: "assistant".into(),
            content: String::new(),
            reasoning_content: None,
        }];
        let (tx, rx) = std::sync::mpsc::channel();
        app.chat.rx = Some(rx);
        app.chat.sending = true;
        app.chat.status = CHAT_STREAMING_STATUS.into();
        app.chat.gen_start = Some(Instant::now());
        tx.send(ChatEvent::PrefillProgress {
            done: 512,
            total: 2048,
        })
        .unwrap();
        app.drain_chat_events();
        assert_eq!(app.chat.prefill, Some((512, 2048)));
        assert!(
            app.chat
                .status
                .starts_with("prefilling 512/2048 (25%) · ETA "),
            "{}",
            app.chat.status
        );
        tx.send(ChatEvent::Content("hi".into())).unwrap();
        app.drain_chat_events();
        assert_eq!(app.chat.prefill, None);
        assert_eq!(app.chat.status, CHAT_STREAMING_STATUS);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn clear_resets_stale_throughput_stats() {
        let (mut app, dir) = test_app();
        app.tab = Tab::Chat;
        app.chat.last_stats = Some(GenStats {
            tokens: 42,
            tps: 99.0,
            exact: true,
        });
        app.handle_chat_command("clear");
        assert!(
            app.chat.last_stats.is_none(),
            "tok/s cleared with the conversation"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn finished_request_is_logged_for_the_inspector() {
        let (mut app, dir) = test_app();
        app.tab = Tab::Chat;
        app.chat.messages = vec![
            ChatMessage {
                role: "user".into(),
                content: "hi".into(),
                reasoning_content: None,
            },
            ChatMessage {
                role: "assistant".into(),
                content: String::new(),
                reasoning_content: None,
            },
        ];
        let (tx, rx) = std::sync::mpsc::channel();
        app.chat.rx = Some(rx);
        app.chat.sending = true;
        app.chat.gen_start = Some(Instant::now());
        app.chat.gen_model = "qwen3.5:9b".into();
        tx.send(ChatEvent::Content("hello".into())).unwrap();
        tx.send(ChatEvent::Content(" world".into())).unwrap();
        tx.send(ChatEvent::Done).unwrap();
        app.drain_chat_events();
        assert_eq!(app.request_log.len(), 1, "completed request is logged");
        assert_eq!(app.request_log[0].model, "qwen3.5:9b");
        assert_eq!(app.request_log[0].tokens, 2);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn errored_request_is_not_logged() {
        let (mut app, dir) = test_app();
        app.tab = Tab::Chat;
        app.chat.messages = vec![
            ChatMessage {
                role: "user".into(),
                content: "hi".into(),
                reasoning_content: None,
            },
            ChatMessage {
                role: "assistant".into(),
                content: String::new(),
                reasoning_content: None,
            },
        ];
        let (tx, rx) = std::sync::mpsc::channel();
        app.chat.rx = Some(rx);
        app.chat.sending = true;
        app.chat.gen_start = Some(Instant::now());
        app.chat.gen_model = "qwen3.5:9b".into();
        tx.send(ChatEvent::Content("partial".into())).unwrap();
        tx.send(ChatEvent::Error("boom".into())).unwrap();
        app.drain_chat_events();
        assert!(
            app.request_log.is_empty(),
            "an errored partial is not logged as a request"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn reload_preserves_expanded_groups() {
        // An `r` refresh reloads the registry; it must NOT collapse the user's
        // expanded Models groups (a fresh RegistryState defaults them empty).
        let (mut app, dir) = test_app();
        app.registry.expanded_groups.insert("qwen".into());
        app.reload();
        assert!(
            app.registry.expanded_groups.contains("qwen"),
            "reload must carry expanded groups across the registry reload"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Enter on a not-downloaded (remote) Models row must not persist it as
    /// `default_model` (serve would point at a file that isn't there) — it
    /// toasts the pull hint instead. A local row still saves.
    #[test]
    fn models_enter_on_remote_row_does_not_save_default_model() {
        use crate::hipfire::registry::{ModelEntry, ModelRow};
        let (mut app, dir) = test_app();
        app.tab = Tab::Models;
        let row = |tag: &str, downloaded: bool| ModelRow {
            tag: tag.into(),
            entry: ModelEntry::default(),
            downloaded,
            has_mtp: false,
        };
        app.registry.models = vec![row("zz:remote", false), row("zz:local", true)];
        app.registry.expanded_groups.insert("zz".into());
        let before_cfg = std::fs::read_to_string(&app.paths.config).unwrap();
        let before_default = app.config.default_model.clone();
        let before_value = app.config.values.get("default_model").cloned();
        let before_active = app.active_model.clone();

        // visible: [group zz, zz:remote, zz:local]
        app.registry.selected = 1;
        app.handle_models_key(key(KeyCode::Enter));
        assert_eq!(
            std::fs::read_to_string(&app.paths.config).unwrap(),
            before_cfg,
            "a remote row must leave config.toml unchanged"
        );
        assert!(!before_cfg.contains("default_model"));
        assert_eq!(app.config.default_model, before_default);
        assert_eq!(app.active_model, before_active);
        assert_eq!(
            app.config.values.get("default_model").cloned(),
            before_value
        );
        assert_eq!(
            app.toast.as_ref().map(|t| t.text.as_str()),
            Some("zz:remote is not downloaded — p to pull")
        );

        // A local row keeps the old behavior.
        app.registry.selected = 2;
        app.handle_models_key(key(KeyCode::Enter));
        assert_eq!(app.config.default_model, "zz:local");
        assert_eq!(app.active_model, "zz:local");
        let on_disk = std::fs::read_to_string(&app.paths.config).unwrap();
        assert!(on_disk.contains("zz:local"), "got: {on_disk}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn drain_chat_events_separates_reasoning_and_content() {
        // SSE fold now routes Reasoning and Content to separate accumulators;
        // rendering keeps visual order but history must keep them distinct.
        let (mut app, dir) = test_app();
        app.tab = Tab::Chat;
        app.chat.messages = vec![
            ChatMessage {
                role: "user".into(),
                content: "hi".into(),
                reasoning_content: None,
            },
            ChatMessage {
                role: "assistant".into(),
                content: String::new(),
                reasoning_content: None,
            },
        ];
        let (tx, rx) = std::sync::mpsc::channel();
        app.chat.rx = Some(rx);
        app.chat.sending = true;
        app.chat.gen_start = Some(Instant::now());
        app.chat.gen_model = "test".into();
        tx.send(ChatEvent::Reasoning("think ".into())).unwrap();
        tx.send(ChatEvent::Reasoning("step".into())).unwrap();
        tx.send(ChatEvent::Content("answer".into())).unwrap();
        tx.send(ChatEvent::Done).unwrap();
        app.drain_chat_events();
        let last = app.chat.messages.last().unwrap();
        assert_eq!(
            last.reasoning_content.as_deref(),
            Some("think step"),
            "reasoning deltas accumulated separately"
        );
        assert_eq!(
            last.content, "answer",
            "content deltas accumulated separately"
        );
        // Serialized request keeps them as distinct keys for prefix-cache.
        let v = serde_json::to_value(&*app.chat.messages).unwrap();
        let asst = &v[1];
        assert_eq!(asst["content"], "answer");
        assert_eq!(asst["reasoning_content"], "think step");
        assert_eq!(app.chat.gen_tokens, 3);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn chat_message_legacy_json_without_reasoning_deserializes() {
        // Old chat_history.json files lack the field; they must load as None.
        let msg: ChatMessage = serde_json::from_value(serde_json::json!({
            "role": "assistant",
            "content": "old"
        }))
        .unwrap();
        assert!(msg.reasoning_content.is_none());
        let v = serde_json::to_value(&msg).unwrap();
        assert!(v.get("reasoning_content").is_none());
    }

    #[test]
    fn reasoning_empty_normalizes_to_none_on_done() {
        let (mut app, dir) = test_app();
        app.tab = Tab::Chat;
        app.chat.messages = vec![
            ChatMessage {
                role: "user".into(),
                content: "hi".into(),
                reasoning_content: None,
            },
            ChatMessage {
                role: "assistant".into(),
                content: String::new(),
                reasoning_content: None,
            },
        ];
        let (tx, rx) = std::sync::mpsc::channel();
        app.chat.rx = Some(rx);
        app.chat.sending = true;
        app.chat.gen_start = Some(Instant::now());
        // No reasoning deltas, but an empty Some("") was somehow set.
        app.chat.messages.last_mut().unwrap().reasoning_content = Some(String::new());
        tx.send(ChatEvent::Content("hi".into())).unwrap();
        tx.send(ChatEvent::Done).unwrap();
        app.drain_chat_events();
        assert!(app
            .chat
            .messages
            .last()
            .unwrap()
            .reasoning_content
            .is_none());
        let _ = std::fs::remove_dir_all(dir);
    }

    fn chat_msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: role.into(),
            content: content.into(),
            reasoning_content: None,
        }
    }

    #[test]
    fn chat_typing_and_scrolling_work_while_streaming_but_enter_does_not_send() {
        let (mut app, dir) = test_app();
        app.tab = Tab::Chat;
        app.chat.messages = vec![chat_msg("user", "hi"), chat_msg("assistant", "")];
        app.chat.sending = true;
        app.chat.view.set(Some(ChatView {
            viewport: 10,
            max_scroll: 30,
        }));
        for c in ['a', 'b', 'c'] {
            app.handle_chat_key(key(KeyCode::Char(c)));
        }
        assert_eq!(app.chat.input, "abc", "typing is kept mid-stream");
        app.handle_chat_key(key(KeyCode::Backspace));
        app.handle_chat_key(key(KeyCode::Char('c')));
        assert_eq!(app.chat.input, "abc");
        app.handle_chat_key(KeyEvent::new(KeyCode::Char('o'), KeyModifiers::CONTROL));
        assert_eq!(app.chat.input, "abc\n", "Ctrl+O newline works mid-stream");
        app.chat.input = "abc".into();

        app.handle_chat_key(key(KeyCode::Up));
        assert_eq!(app.chat.scroll, 1, "Up scrolls mid-stream");
        app.handle_chat_key(key(KeyCode::PageUp));
        assert_eq!(app.chat.scroll, 11);

        app.handle_chat_key(key(KeyCode::Enter));
        assert_eq!(app.chat.input, "abc", "Enter does not consume the input");
        assert_eq!(app.chat.messages.len(), 2, "Enter does not send");
        assert!(app.chat.sending);
        assert!(app.chat.status.contains("Esc stops it"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn chat_help_note_is_not_a_message() {
        let (mut app, dir) = test_app();
        app.tab = Tab::Chat;
        app.chat.messages = vec![chat_msg("user", "hi"), chat_msg("assistant", "yo")];
        app.handle_chat_command("help");
        assert!(app
            .chat
            .notice
            .as_deref()
            .is_some_and(|n| n.contains("/regen")));
        assert_eq!(app.chat.messages.len(), 2);
        assert!(app.tail_is_redoable(), "regen/edit still see the real tail");
        app.handle_chat_command("clear");
        assert!(app.chat.notice.is_none(), "/clear dismisses the note");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn abort_marks_message_index_and_edit_prunes_it() {
        let (mut app, dir) = test_app();
        app.tab = Tab::Chat;
        app.chat.messages = vec![chat_msg("user", "hi"), chat_msg("assistant", "part")];
        app.chat.sending = true;
        app.chat.request_abort();
        assert_eq!(app.chat.stopped, vec![1]);
        app.edit_last();
        assert!(app.chat.stopped.is_empty(), "mark dropped with its reply");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn usage_chunk_gives_exact_tokens_else_chunk_count() {
        let (mut app, dir) = test_app();
        app.tab = Tab::Chat;
        for usage in [Some(5usize), None] {
            app.chat.messages = vec![chat_msg("user", "hi"), chat_msg("assistant", "")];
            let (tx, rx) = std::sync::mpsc::channel();
            app.chat.rx = Some(rx);
            app.chat.sending = true;
            app.chat.gen_start = Some(Instant::now());
            app.chat.gen_tokens = 0;
            app.chat.gen_usage = None;
            tx.send(ChatEvent::Content("a".into())).unwrap();
            tx.send(ChatEvent::Content("b".into())).unwrap();
            if let Some(n) = usage {
                tx.send(ChatEvent::Usage {
                    completion_tokens: n,
                })
                .unwrap();
            }
            tx.send(ChatEvent::Done).unwrap();
            app.drain_chat_events();
            let stats = app.chat.last_stats.expect("stats recorded");
            match usage {
                Some(n) => {
                    assert_eq!((stats.tokens, stats.exact, stats.unit()), (n, true, "tok"));
                }
                None => {
                    assert_eq!((stats.tokens, stats.exact), (2, false));
                    assert_eq!(stats.unit(), "chunks");
                    assert_eq!(app.request_log[0].unit(), "chunks");
                }
            }
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    // ── Chat serve-readiness queue ─────────────────────────────────────────

    fn health(up: bool) -> HealthSnapshot {
        HealthSnapshot {
            checked: true,
            serve_up: up,
            model: up.then(|| "m".to_string()),
            text: if up {
                r#"{"status":"ok","model":"m"}"#.into()
            } else {
                String::new()
            },
            failure: None,
        }
    }

    /// A Chat-tab app isolated from the real `~/.hipfire` (serve.pid) and from
    /// any real serve (config port points at a free, closed port).
    fn chat_app() -> (App, TempHome) {
        let (mut app, dir) = test_app();
        app.tab = Tab::Chat;
        app.paths.serve_pid = dir.join("serve.pid");
        app.config.port = std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        (app, dir)
    }

    fn type_and_enter(app: &mut App, text: &str) {
        app.chat.input = text.into();
        app.handle_tab_key(key(KeyCode::Enter));
    }

    /// Fake `hipfire` that appends its arguments to `<dir>/serve-calls.log`.
    fn fake_cli(dir: &std::path::Path) -> (std::path::PathBuf, std::path::PathBuf) {
        use std::os::unix::fs::PermissionsExt;
        let log = dir.join("serve-calls.log");
        let bin = dir.join("fake-hipfire");
        std::fs::write(
            &bin,
            format!("#!/bin/sh\necho \"$@\" >> '{}'\n", log.display()),
        )
        .unwrap();
        std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).unwrap();
        (bin, log)
    }

    /// `serve …` invocations the fake CLI logged. Other tests running in
    /// parallel may route a `stop`/`diag` through the same HIPFIRE_CLI_BIN
    /// while it is set, so only lines that start a serve are counted.
    fn serve_calls(log: &std::path::Path) -> usize {
        std::fs::read_to_string(log).map_or(0, |s| {
            s.lines()
                .filter(|line| line.split_whitespace().next() == Some("serve"))
                .count()
        })
    }

    #[test]
    fn chat_enter_with_unknown_health_queues_and_sends_when_online() {
        let (mut app, dir) = chat_app();
        assert!(!app.status.health_known);
        type_and_enter(&mut app, "hello");

        assert!(app.has_queued_send());
        assert!(app.chat.input.is_empty(), "prompt leaves the input box");
        assert!(app.chat.messages.is_empty(), "not pushed into messages yet");
        assert!(!app.chat.sending);
        assert!(
            app.toast.is_none(),
            "no 'serve start failed' / 'starting' toast while health is unknown: {:?}",
            app.toast
        );
        assert!(app.chat.status.contains("sends when it is online"));
        assert!(!app.chat.status.contains("press r"));

        // Still unknown: nothing happens.
        app.inject_health(HealthSnapshot::default());
        assert!(app.has_queued_send() && !app.chat.sending);

        app.inject_health(health(true));
        assert!(!app.has_queued_send());
        assert!(app.chat.sending, "queued prompt sent once online");
        let last_user = app
            .chat
            .messages
            .iter()
            .rfind(|m| m.role == "user")
            .unwrap();
        assert_eq!(last_user.content, "hello");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn chat_literal_slash_prompt_queues_and_sends_when_online() {
        let (mut app, dir) = chat_app();
        type_and_enter(&mut app, "//not-a-command");
        assert!(app.has_queued_send());
        assert!(app.chat.messages.is_empty());
        app.inject_health(health(true));
        assert!(app.chat.sending);
        assert_eq!(app.chat.messages[0].content, "/not-a-command");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn chat_enter_while_online_sends_immediately() {
        let (mut app, dir) = chat_app();
        app.inject_health(health(true));
        type_and_enter(&mut app, "now");
        assert!(!app.has_queued_send());
        assert!(app.chat.sending);
        assert_eq!(app.chat.messages[0].content, "now");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn cancelling_a_queued_send_restores_the_input() {
        let (mut app, dir) = chat_app();
        type_and_enter(&mut app, "keep me");
        assert!(app.has_queued_send());
        assert!(app.cancel_queued_send());
        assert!(!app.has_queued_send());
        assert_eq!(app.chat.input, "keep me");
        assert_eq!(app.chat.status, "queued message cancelled");
        assert!(!app.cancel_queued_send(), "nothing left to cancel");
        // Going online afterwards must not resurrect it.
        app.inject_health(health(true));
        assert!(!app.chat.sending && app.chat.messages.is_empty());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn health_known_down_without_pid_starts_serve_exactly_once() {
        let _guard = crate::hipfire::ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let (mut app, dir) = chat_app();
        let (bin, log) = fake_cli(&dir);
        std::env::set_var("HIPFIRE_CLI_BIN", &bin);

        type_and_enter(&mut app, "hi");
        assert_eq!(serve_calls(&log), 0, "health unknown: never spawns");
        app.inject_health(health(false));
        for _ in 0..5 {
            app.inject_health(health(false));
        }
        let deadline = Instant::now() + Duration::from_secs(3);
        while serve_calls(&log) == 0 && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(20));
        }
        thread::sleep(Duration::from_millis(150));
        std::env::remove_var("HIPFIRE_CLI_BIN");

        let log_text = std::fs::read_to_string(&log).unwrap();
        let serve_lines: Vec<&str> = log_text
            .lines()
            .filter(|line| line.split_whitespace().next() == Some("serve"))
            .collect();
        assert_eq!(
            serve_lines,
            ["serve -d"],
            "exactly one background serve start"
        );
        assert!(app.has_queued_send(), "prompt stays queued while it boots");
        assert!(app.chat.status.contains("starting hipfire serve -d"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn health_known_down_with_live_pid_waits_instead_of_spawning() {
        let _guard = crate::hipfire::ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let (mut app, dir) = chat_app();
        let (bin, log) = fake_cli(&dir);
        std::env::set_var("HIPFIRE_CLI_BIN", &bin);
        // Our own pid is guaranteed alive under /proc.
        std::fs::write(&app.paths.serve_pid, std::process::id().to_string()).unwrap();

        app.inject_health(health(false));
        type_and_enter(&mut app, "hi");
        app.inject_health(health(false));
        thread::sleep(Duration::from_millis(200));
        std::env::remove_var("HIPFIRE_CLI_BIN");

        assert_eq!(serve_calls(&log), 0, "live serve.pid: no second serve");
        assert!(app.has_queued_send());
        assert!(app.chat.status.contains("sends when it is online"));

        app.inject_health(health(true));
        assert!(app.chat.sending);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn spawn_failure_gives_the_prompt_back() {
        let _guard = crate::hipfire::ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("HIPFIRE_CLI_BIN");
        let (mut app, dir) = chat_app();
        app.inject_health(health(false));
        type_and_enter(&mut app, "hi");
        assert!(!app.has_queued_send(), "nothing will ever send it");
        assert_eq!(app.chat.input, "hi");
        assert!(app
            .toast
            .as_ref()
            .is_some_and(|t| t.text.contains("serve start failed")));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn rejected_bearer_does_not_start_a_serve_and_gives_the_prompt_back() {
        let (mut app, dir) = chat_app();
        type_and_enter(&mut app, "hi");
        let mut h = health(false);
        h.failure = Some("serve rejected the bearer token (HTTP 401)".into());
        h.text = h.failure.clone().unwrap();
        app.inject_health(h);
        assert!(!app.has_queued_send());
        assert_eq!(app.chat.input, "hi");
        assert!(app.chat.status.contains("rejected the bearer"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn regen_queues_without_popping_the_reply_then_runs_when_online() {
        let (mut app, dir) = chat_app();
        app.chat.messages = vec![
            ChatMessage {
                role: "user".into(),
                content: "q".into(),
                reasoning_content: None,
            },
            ChatMessage {
                role: "assistant".into(),
                content: "a".into(),
                reasoning_content: None,
            },
        ];
        app.handle_chat_command("regen");
        assert!(app.has_queued_send());
        assert_eq!(app.chat.messages.len(), 2);
        assert_eq!(app.chat.messages[1].content, "a", "reply kept while queued");
        assert!(app.toast.is_none());

        app.inject_health(health(true));
        assert!(app.chat.sending);
        assert_eq!(app.chat.messages.len(), 2);
        assert_eq!(app.chat.messages[0].content, "q");
        assert!(
            app.chat.messages[1].content.is_empty(),
            "old reply replaced"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn health_wins_over_a_stale_full_snapshot() {
        use crate::hipfire::dashboard::{LoadState, VramState};
        let (mut app, dir) = chat_app();
        let mut snap = Dashboard::offline(
            "127.0.0.1:1".into(),
            VramState::Unavailable("x".into()),
            LoadState::Unavailable("x".into()),
        );
        snap.serve_up = true;
        snap.model = Some("old".into());
        app.dashboard = Some(snap.clone());
        app.status.overlay_live(&snap);
        app.inject_health(health(true));
        assert!(app.status.serve_http_ok);

        app.inject_health(health(false));
        assert!(!app.status.serve_http_ok);
        let dash = app.dashboard.as_ref().unwrap();
        assert!(!dash.serve_up && dash.model.is_none());
        // overlay_live (full snapshot) never rewrites serve_http_ok.
        app.status.overlay_live(&snap);
        assert!(!app.status.serve_http_ok);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn edit_state_is_char_safe() {
        let mut edit = EditState::new("chat_template".into(), "é/é".into());
        edit.home();
        edit.right();
        edit.insert('ß');
        assert_eq!(edit.buffer, "éß/é");
        edit.end();
        edit.backspace();
        assert_eq!(edit.buffer, "éß/");
    }

    /// A Models-tab app over two remote rows: `rich:full` (q4+q8 PLE and an MTP
    /// sidecar) and `plain:base` (nothing). `rich:full` is selected.
    fn pull_app() -> (App, TempHome) {
        use crate::hipfire::registry::{ModelEntry, ModelRow};
        use hipfire_registry::Sidecar;
        let (mut app, dir) = test_app();
        app.tab = Tab::Models;
        let sc = |size| Sidecar {
            file: "x".into(),
            sha256: None,
            size_bytes: Some(size),
        };
        let mut rich = ModelEntry::default();
        rich.ple.insert("q4".into(), sc(1_500_000_000));
        rich.ple.insert("q8".into(), sc(3_000_000_000));
        rich.mtp = Some(sc(400_000_000));
        let row = |tag: &str, entry: ModelEntry| ModelRow {
            tag: tag.into(),
            entry,
            downloaded: false,
            has_mtp: false,
        };
        app.registry.models = vec![
            row("plain:base", ModelEntry::default()),
            row("rich:full", rich),
        ];
        app.registry.expanded_groups = ["plain".to_string(), "rich".to_string()]
            .into_iter()
            .collect();
        (app, dir)
    }

    /// Select the model row `tag` among the visible items.
    fn select_tag(app: &mut App, tag: &str) {
        let items = app.registry.visible_items();
        let at = items
            .iter()
            .position(|item| match item {
                crate::hipfire::registry::ModelListItem::Model { model_index } => {
                    app.registry.models[*model_index].tag == tag
                }
                _ => false,
            })
            .expect("row visible");
        app.registry.selected = at;
    }

    /// Install a fake CLI that records `"$@"` (one line per call) and exits 1
    /// unless `$1` is `pull`; returns the record path.
    fn pull_recorder(dir: &std::path::Path) -> std::path::PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let record = dir.join("pull-argv.txt");
        let bin = dir.join("hipfire-fake");
        std::fs::write(
            &bin,
            format!(
                "#!/bin/sh\n[ \"$1\" = pull ] || exit 1\necho \"$@\" >> '{}'\nexit 0\n",
                record.display()
            ),
        )
        .unwrap();
        std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).unwrap();
        std::env::set_var("HIPFIRE_CLI_BIN", &bin);
        record
    }

    fn wait_for_record(record: &std::path::Path) -> String {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if let Ok(text) = std::fs::read_to_string(record) {
                if !text.trim().is_empty() {
                    return text;
                }
            }
            assert!(Instant::now() < deadline, "fake CLI never recorded argv");
            thread::sleep(Duration::from_millis(20));
        }
    }

    #[test]
    fn p_on_ple_model_opens_chooser_and_enter_passes_ple_and_no_draft() {
        let _guard = crate::hipfire::ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let (mut app, dir) = pull_app();
        let record = pull_recorder(&dir);
        select_tag(&mut app, "rich:full");

        app.handle_tab_key(key(KeyCode::Char('p')));
        assert!(app.pull.is_none(), "chooser opens before any pull starts");
        assert!(app.pull_choice.is_some());

        // PLE row: auto -> q4.
        app.handle_tab_key(key(KeyCode::Right));
        // Draft row: yes -> no.
        app.handle_tab_key(key(KeyCode::Down));
        app.handle_tab_key(key(KeyCode::Char(' ')));
        app.handle_tab_key(key(KeyCode::Enter));
        assert!(app.pull_choice.is_none(), "Enter closes the chooser");
        assert!(app.pull.is_some(), "Enter starts the pull");

        let argv = wait_for_record(&record);
        std::env::remove_var("HIPFIRE_CLI_BIN");
        assert_eq!(argv.trim(), "pull rich:full --yes --ple q4 --no-draft");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn chooser_defaults_pass_no_extra_flags() {
        let _guard = crate::hipfire::ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let (mut app, dir) = pull_app();
        let record = pull_recorder(&dir);
        select_tag(&mut app, "rich:full");
        app.handle_tab_key(key(KeyCode::Char('p')));
        app.handle_tab_key(key(KeyCode::Enter));
        let argv = wait_for_record(&record);
        std::env::remove_var("HIPFIRE_CLI_BIN");
        assert_eq!(argv.trim(), "pull rich:full --yes");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn p_on_model_without_ple_or_draft_pulls_immediately() {
        let _guard = crate::hipfire::ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let (mut app, dir) = pull_app();
        let record = pull_recorder(&dir);
        select_tag(&mut app, "plain:base");
        app.handle_tab_key(key(KeyCode::Char('p')));
        assert!(app.pull_choice.is_none(), "no chooser state");
        assert!(app.pull.is_some(), "pull started immediately");
        let argv = wait_for_record(&record);
        std::env::remove_var("HIPFIRE_CLI_BIN");
        assert_eq!(argv.trim(), "pull plain:base --yes");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn esc_closes_the_chooser_without_pulling() {
        let (mut app, dir) = pull_app();
        select_tag(&mut app, "rich:full");
        app.handle_tab_key(key(KeyCode::Char('p')));
        assert!(app.pull_choice.is_some());
        app.handle_tab_key(key(KeyCode::Esc));
        assert!(app.pull_choice.is_none());
        assert!(app.pull.is_none(), "Esc must not start a pull");
        assert_eq!(
            app.toast.as_ref().map(|t| t.text.as_str()),
            Some("pull not started")
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    /// `load_from_paths` reads only what it is handed: a fixture root, an
    /// injected GPU/model context and an injected environment layer.
    #[test]
    fn load_from_paths_uses_the_injected_arch_env_and_root() {
        use hipfire_config::{
            effective::{A8Prefill, Origin},
            ConfigValue,
        };
        let home = TempHome::new("load-from-paths");
        let root = home.to_path_buf();
        std::fs::write(root.join("config.toml"), "[reasoning]\neffort = \"low\"\n").unwrap();
        let arch = ArchCtx {
            gpu_arch: Some("gfx1151".into()),
            gpu_count: 1,
            model_arch_id: Some(5),
            registry_tag: Some("qwen3.8:27b".into()),
            a8_prefill: A8Prefill::NotLanded,
        };
        let mut env = hipfire_config::ConfigLayer::default();
        env.set("memory.kv_cache", ConfigValue::String("q8".into()))
            .unwrap();

        let app =
            App::load_from_paths(HipfirePaths::under(root.clone()), arch.clone(), env.clone())
                .unwrap();
        assert_eq!(app.arch, arch);
        assert_eq!(app.env_layer, env);
        assert_eq!(app.paths.root, root);
        let row = |key: &str| {
            app.settings
                .rows
                .iter()
                .find(|row| row.id.key() == Some(key))
                .unwrap_or_else(|| panic!("no row {key}"))
        };
        assert_eq!(row("reasoning.effort").display, "low");
        assert_eq!(row("reasoning.effort").origin, Origin::You);
        assert_eq!(
            row("memory.kv_cache").origin,
            Origin::Env("HIPFIRE_KV_MODE".into())
        );
        assert!(row("memory.kv_cache").locked.is_some());
        let _ = std::fs::remove_dir_all(root);
    }
}
