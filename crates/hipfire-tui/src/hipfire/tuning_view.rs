// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! State of the Tuning screens (spec §2, mockups S2/S3/L2): the preset
//! chooser, the preview with its ticks, and the undo confirmation.
//!
//! Everything numeric comes from `hipfire_config::tuning` (plan, preview,
//! apply, undo) and `hipfire_config::evidence`; this module only holds the
//! cursor and tick state and reads/writes through that engine. Nothing is
//! written until [`TuningView::save`] / [`TuningView::undo_save`].

use std::path::Path;

use hipfire_config::{
    effective::{Applies, ArchCtx},
    load_catalog, load_global,
    tuning::{
        self, current_value, find_conflicts, Conflict, PlanEnv, PlanLine, Preset, TuningPlan,
        TuningState, TuningStatus, UndoOutcome,
    },
    ConfigLayer, ConfigLayerDiff, ConfigPaths, ConfigValue, ModelCatalog,
};
use hipfire_registry::RegistryV1;

use super::HipfirePaths;

/// Which Tuning screen is on top.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// S2: the three presets.
    Chooser,
    /// S3 / L2: what the chosen preset would change.
    Preview,
    /// The `u` confirmation listing what undo restores.
    UndoConfirm,
}

/// What a preview would write, computed from the live tick state.
#[derive(Clone, Debug)]
pub struct Preview {
    /// Keys that would change (the footer's `y save N changes`).
    pub count: usize,
    /// Rows `key  from → to` of keys the old preset gives back, which the
    /// new plan does not write itself.
    pub given_back: Vec<(String, String, String)>,
    /// Old-preset keys left alone because they were edited since.
    pub kept: Vec<String>,
    pub conflicts: Vec<Conflict>,
    /// The slowest scope among the keys that change.
    pub applies: Option<Applies>,
}

/// The outcome of a successful save, for the toast.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Saved {
    pub preset: Preset,
    /// Keys changed in `config.toml`; 0 means nothing was written.
    pub changes: usize,
    pub applies: Option<Applies>,
}

/// The outcome of a successful undo.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Undone {
    pub restored: Preset,
    pub changes: usize,
}

pub struct TuningView {
    pub stage: Stage,
    /// Preset under the cursor (index into [`Preset::ALL`]).
    pub cursor: usize,
    /// Plan line under the cursor in the preview.
    pub selected: usize,
    ticked: Vec<&'static str>,
    plans: Vec<TuningPlan>,
    global: ConfigLayer,
    state: TuningState,
    catalog: ModelCatalog,
    /// What `u` would restore (the confirmation's content).
    pub undo: Option<UndoOutcome>,
    /// The screen the undo confirmation was opened over; Esc goes back there.
    /// `None` when it was opened straight from Settings (Esc closes).
    undo_return: Option<Stage>,
}

impl TuningView {
    /// Load `config.toml`, `tuning.toml`, `models.toml` and the local files
    /// the plan needs, and compute the three plans for `arch`. An unreadable
    /// file reads as empty, like the Settings header does.
    pub fn open(paths: &HipfirePaths, arch: &ArchCtx, registry: &RegistryV1) -> Self {
        let config_paths = ConfigPaths::under(&paths.root);
        let global = load_global(&config_paths)
            .map(|loaded| loaded.layer)
            .unwrap_or_default();
        let state = TuningState::load(&config_paths).unwrap_or_default();
        let catalog = load_catalog(&config_paths)
            .map(|loaded| loaded.catalog)
            .unwrap_or_default();
        let plan_env = hipfire_registry::tuning_plan_env(&paths.models, registry);
        Self::from_parts(global, state, catalog, arch, &plan_env)
    }

    fn from_parts(
        global: ConfigLayer,
        state: TuningState,
        catalog: ModelCatalog,
        arch: &ArchCtx,
        plan_env: &PlanEnv,
    ) -> Self {
        let plans: Vec<TuningPlan> = Preset::ALL
            .iter()
            .map(|preset| tuning::plan(*preset, arch, plan_env))
            .collect();
        // Start on Quality-first when nothing is applied (the useful pick),
        // else on what is applied.
        let cursor = match state.preset() {
            Preset::Balanced => 1,
            other => Preset::ALL.iter().position(|p| *p == other).unwrap_or(0),
        };
        let mut view = Self {
            stage: Stage::Chooser,
            cursor,
            selected: 0,
            ticked: Vec::new(),
            plans,
            global,
            state,
            catalog,
            undo: None,
            undo_return: None,
        };
        view.reset_ticks();
        view
    }

    fn reset_ticks(&mut self) {
        self.ticked = self.plans[self.cursor].default_ticked();
        self.selected = 0;
    }

    /// The preset that is applied now.
    pub fn current(&self) -> Preset {
        self.state.preset()
    }

    pub fn status(&self, arch: &ArchCtx) -> TuningStatus {
        tuning::status(&self.global, &self.state, arch)
    }

    /// The preset under the cursor.
    pub fn preset(&self) -> Preset {
        Preset::ALL[self.cursor]
    }

    pub fn plan(&self) -> &TuningPlan {
        &self.plans[self.cursor]
    }

    /// Move the preset cursor (chooser and preview); the ticks reset to the
    /// new plan's defaults.
    pub fn move_cursor(&mut self, delta: isize) {
        let count = Preset::ALL.len() as isize;
        let next = (self.cursor as isize + delta).clamp(0, count - 1) as usize;
        if next != self.cursor {
            self.cursor = next;
            self.reset_ticks();
        }
    }

    pub fn move_selected(&mut self, delta: isize) {
        let len = self.plan().lines.len() as isize;
        if len == 0 {
            self.selected = 0;
            return;
        }
        self.selected = (self.selected as isize + delta).clamp(0, len - 1) as usize;
    }

    pub fn is_ticked(&self, line: &PlanLine) -> bool {
        line.requirement.met && self.ticked.contains(&line.id)
    }

    pub fn ticked_ids(&self) -> Vec<&'static str> {
        self.plan()
            .lines
            .iter()
            .filter(|line| self.is_ticked(line))
            .map(|line| line.id)
            .collect()
    }

    /// Tick or untick the line under the cursor. An unmet line cannot be
    /// ticked: `Err` carries its reason and the command that fixes it.
    pub fn toggle_selected(&mut self) -> Result<(), String> {
        let Some(line) = self.plan().lines.get(self.selected) else {
            return Ok(());
        };
        if !line.requirement.met {
            let reason = line
                .requirement
                .reason
                .as_deref()
                .unwrap_or("requirement not met");
            return Err(match line.requirement.fix.as_deref() {
                Some(fix) => format!("{} can't be ticked: {reason}: {fix}", line.title),
                None => format!("{} can't be ticked: {reason}", line.title),
            });
        }
        let id = line.id;
        if let Some(position) = self.ticked.iter().position(|ticked| *ticked == id) {
            self.ticked.remove(position);
        } else {
            self.ticked.push(id);
        }
        Ok(())
    }

    /// Open the undo confirmation. `over_screen` keeps the screen it was
    /// opened over for Esc to return to. `false` (nothing to undo) leaves the
    /// view where it is.
    pub fn open_undo(&mut self, over_screen: bool) -> bool {
        let Some(outcome) = tuning::undo(&self.global, &self.state) else {
            return false;
        };
        self.undo_return = over_screen.then_some(self.stage);
        self.undo = Some(outcome);
        self.stage = Stage::UndoConfirm;
        true
    }

    /// Leave the undo confirmation; `false` when the view should close.
    pub fn cancel_undo(&mut self) -> bool {
        self.undo = None;
        match self.undo_return.take() {
            Some(stage) => {
                self.stage = stage;
                true
            }
            None => false,
        }
    }

    /// What the preview would write with the current ticks. `env` is the
    /// `HIPFIRE_*` layer (it wins over `config.toml`).
    pub fn preview(&self, env: &ConfigLayer) -> Preview {
        self.preview_of(self.cursor, &self.ticked_ids(), env)
    }

    /// Changes `preset` would make with its default ticks (the chooser's
    /// per-card count).
    pub fn default_change_count(&self, preset: Preset) -> usize {
        let index = Preset::ALL.iter().position(|p| *p == preset).unwrap_or(0);
        self.preview_of(
            index,
            &self.plans[index].default_ticked(),
            &ConfigLayer::default(),
        )
        .count
    }

    fn preview_of(&self, index: usize, ticked: &[&'static str], env: &ConfigLayer) -> Preview {
        let plan = &self.plans[index];
        let conflicts = find_conflicts(plan, ticked, env, &self.catalog);
        match tuning::preview(&self.global, &self.state, plan, ticked) {
            Ok(preview) => {
                let own: Vec<&str> = plan.keys().into_iter().collect();
                let given_back = rows(&preview.net)
                    .into_iter()
                    .filter(|(key, _, _)| !own.contains(&key.as_str()))
                    .collect();
                let keys: Vec<&str> = preview
                    .net
                    .added
                    .iter()
                    .map(|(key, _)| key.as_str())
                    .chain(preview.net.changed.iter().map(|(key, _, _)| key.as_str()))
                    .chain(preview.net.removed.iter().map(|(key, _)| key.as_str()))
                    .collect();
                Preview {
                    count: preview.net.len(),
                    applies: tuning::takes_effect(keys),
                    given_back,
                    kept: preview.kept,
                    conflicts,
                }
            }
            // The engine refuses a write the schema rejects (a preset naming
            // a key this build lacks; only reachable when a test injects the
            // #902 state). Count what the ticked lines would change so the
            // screen still reads; `save` reports the engine's error.
            Err(_) => {
                let writes = plan.writes_for(ticked);
                let keys: Vec<&str> = writes
                    .iter()
                    .filter(|(key, value)| {
                        current_value(&self.global, key).as_ref() != Some(*value)
                    })
                    .map(|(key, _)| key.as_str())
                    .collect();
                Preview {
                    count: keys.len(),
                    applies: tuning::takes_effect(keys),
                    given_back: Vec::new(),
                    kept: Vec::new(),
                    conflicts,
                }
            }
        }
    }

    /// Apply the ticked net diff to `config.toml` and record ownership in
    /// `tuning.toml`. Reads both files fresh, so a hand edit since the
    /// preview is merged, not overwritten.
    pub fn save(&self, root: &Path, now: u64) -> Result<Saved, String> {
        let config_paths = ConfigPaths::under(root);
        let global = load_global(&config_paths)
            .map_err(|error| error.to_string())?
            .layer;
        let state = TuningState::load(&config_paths).map_err(|error| error.to_string())?;
        let outcome = tuning::apply(&global, &state, self.plan(), &self.ticked_ids(), now)
            .map_err(|error| error.to_string())?;
        let changes = outcome.net.len();
        if !outcome.changed {
            return Ok(Saved {
                preset: self.preset(),
                changes: 0,
                applies: None,
            });
        }
        tuning::commit(&config_paths, &outcome.layer, &outcome.state)
            .map_err(|error| error.to_string())?;
        let touched: Vec<&str> = outcome
            .written
            .iter()
            .chain(&outcome.reverted)
            .map(String::as_str)
            .collect();
        Ok(Saved {
            preset: self.preset(),
            changes,
            applies: tuning::takes_effect(touched),
        })
    }

    /// Apply the undo the confirmation lists (fresh files, as in `save`).
    pub fn undo_save(&self, root: &Path) -> Result<Undone, String> {
        let config_paths = ConfigPaths::under(root);
        let global = load_global(&config_paths)
            .map_err(|error| error.to_string())?
            .layer;
        let state = TuningState::load(&config_paths).map_err(|error| error.to_string())?;
        let outcome = tuning::undo(&global, &state).ok_or("nothing to undo")?;
        tuning::commit(&config_paths, &outcome.layer, &outcome.state)
            .map_err(|error| error.to_string())?;
        Ok(Undone {
            restored: outcome.restored_preset,
            changes: outcome.net.len(),
        })
    }
}

/// `(key, from, to)` rows of a diff: `unset` for an absent side.
pub fn rows(net: &ConfigLayerDiff) -> Vec<(String, String, String)> {
    let show = |value: &ConfigValue| value.to_toml_string();
    let mut rows = Vec::new();
    for (key, value) in &net.added {
        rows.push((key.clone(), "unset".to_owned(), show(value)));
    }
    for (key, from, to) in &net.changed {
        rows.push((key.clone(), show(from), show(to)));
    }
    for (key, value) in &net.removed {
        rows.push((key.clone(), show(value), "unset".to_owned()));
    }
    rows.sort();
    rows
}

/// `1 change` / `2 changes`.
pub fn changes_label(count: usize) -> String {
    format!("{count} {}", if count == 1 { "change" } else { "changes" })
}

/// The slowest-scope phrase of the preview's save line.
pub fn takes_effect_text(applies: Option<Applies>) -> &'static str {
    match applies {
        Some(Applies::ServeRestart) => "takes effect after serve restart",
        Some(Applies::NextModelLoad) => "takes effect on the next model load",
        Some(Applies::NewChat) => "takes effect in a new chat",
        Some(Applies::NextMessage) => "takes effect on the next message",
        None => "nothing to write",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_config::effective::A8Prefill;

    fn ctx(arch: &str) -> ArchCtx {
        ArchCtx {
            gpu_arch: Some(arch.to_owned()),
            gpu_count: 1,
            model_arch_id: None,
            registry_tag: None,
            a8_prefill: A8Prefill::NotLanded,
        }
    }

    fn view(arch: &str, ple_q8: bool) -> TuningView {
        let env = PlanEnv {
            ple_q8_present: ple_q8,
            flash_next_installed: true,
            ple_q8_missing: if ple_q8 {
                Vec::new()
            } else {
                vec!["qwen3.8-flash-next-base.mq4-xts".to_owned()]
            },
            pull_tag: Some("qwen3.8:flash-next".to_owned()),
        };
        TuningView::from_parts(
            ConfigLayer::default(),
            TuningState::default(),
            ModelCatalog::default(),
            &ctx(arch),
            &env,
        )
    }

    #[test]
    fn chooser_starts_on_quality_first_and_ticks_follow_the_plan() {
        let mut view = view("gfx1151", false);
        assert_eq!(view.preset(), Preset::QualityFirst);
        assert_eq!(
            view.ticked_ids(),
            ["prompt-processing", "flash-next-experts"]
        );
        let preview = view.preview(&ConfigLayer::default());
        assert_eq!(preview.count, 2, "{preview:?}");
        view.move_cursor(1);
        assert_eq!(view.preset(), Preset::LowPower);
        assert_eq!(view.ticked_ids(), ["shorter-thinking"]);
    }

    #[test]
    fn unmet_line_cannot_be_ticked_and_says_why() {
        let mut view = view("gfx1151", false);
        view.selected = 2;
        let reason = view.toggle_selected().unwrap_err();
        assert!(reason.contains("no .ple-q8"), "{reason}");
        assert!(reason.contains("hipfire pull qwen3.8:flash-next --ple q8"));
        assert_eq!(view.ticked_ids().len(), 2);
        view.selected = 0;
        view.toggle_selected().unwrap();
        assert_eq!(view.preview(&ConfigLayer::default()).count, 1);
    }

    #[test]
    fn labels_are_singular_and_plural() {
        assert_eq!(changes_label(1), "1 change");
        assert_eq!(changes_label(2), "2 changes");
        assert_eq!(changes_label(0), "0 changes");
    }
}
