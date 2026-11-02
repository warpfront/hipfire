// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! The Models-tab pull chooser: which PLE rows and draft sidecars a `p` pull
//! should fetch. Built from a registry entry; maps the choices to the
//! `hipfire pull` flags (`--ple q4|q8`, `--no-draft`).

use hipfire_registry::{ModelEntry, Sidecar};

/// One selectable PLE format. `auto` has no flag (the CLI default); `q4`/`q8`
/// are offered only when the entry publishes them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PleOption {
    pub label: &'static str,
    pub size_bytes: Option<u64>,
}

/// One draft sidecar kind the entry carries (MTP / DSpark / DFlash).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DraftKind {
    pub label: &'static str,
    pub size_bytes: Option<u64>,
}

/// The PLE row: options and the highlighted one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PleChoice {
    pub options: Vec<PleOption>,
    pub selected: usize,
}

/// The draft-sidecar row: which kinds exist and whether to fetch them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DraftChoice {
    pub kinds: Vec<DraftKind>,
    pub yes: bool,
}

/// Which chooser row has focus.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChoiceRow {
    Ple,
    Draft,
}

/// Modal pull chooser state (`App::pull_choice`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PullChoice {
    pub tag: String,
    pub ple: Option<PleChoice>,
    pub draft: Option<DraftChoice>,
    pub row: ChoiceRow,
}

impl PullChoice {
    /// Build the chooser for `entry`; `None` when it has neither PLE files nor
    /// draft sidecars (the pull then starts immediately with no extra args).
    pub fn for_entry(tag: &str, entry: &ModelEntry) -> Option<Self> {
        let ple = (!entry.ple.is_empty()).then(|| {
            let mut options = vec![PleOption {
                label: "auto",
                size_bytes: None,
            }];
            for label in ["q4", "q8"] {
                if let Some(sidecar) = entry.ple.get(label) {
                    options.push(PleOption {
                        label,
                        size_bytes: sidecar.size_bytes,
                    });
                }
            }
            PleChoice {
                options,
                selected: 0,
            }
        });
        let kinds: Vec<DraftKind> = [
            ("MTP", &entry.mtp),
            ("DSpark", &entry.dspark),
            ("DFlash", &entry.dflash),
        ]
        .into_iter()
        .filter_map(|(label, sidecar): (&'static str, &Option<Sidecar>)| {
            sidecar.as_ref().map(|s| DraftKind {
                label,
                size_bytes: s.size_bytes,
            })
        })
        .collect();
        let draft = (!kinds.is_empty()).then_some(DraftChoice { kinds, yes: true });
        let row = if ple.is_some() {
            ChoiceRow::Ple
        } else {
            ChoiceRow::Draft
        };
        (ple.is_some() || draft.is_some()).then(|| Self {
            tag: tag.to_string(),
            ple,
            draft,
            row,
        })
    }

    /// Rows present, top to bottom.
    fn rows(&self) -> Vec<ChoiceRow> {
        let mut rows = Vec::with_capacity(2);
        if self.ple.is_some() {
            rows.push(ChoiceRow::Ple);
        }
        if self.draft.is_some() {
            rows.push(ChoiceRow::Draft);
        }
        rows
    }

    /// Move focus to the row below / above (clamped).
    pub fn move_row(&mut self, down: bool) {
        let rows = self.rows();
        let Some(at) = rows.iter().position(|r| *r == self.row) else {
            return;
        };
        let next = if down {
            (at + 1).min(rows.len() - 1)
        } else {
            at.saturating_sub(1)
        };
        self.row = rows[next];
    }

    /// Change the focused row's value: PLE steps (wrapping) forward/back; the
    /// draft row toggles yes/no in either direction.
    pub fn change(&mut self, forward: bool) {
        match self.row {
            ChoiceRow::Ple => {
                if let Some(ple) = &mut self.ple {
                    let n = ple.options.len();
                    ple.selected = if forward {
                        (ple.selected + 1) % n
                    } else {
                        (ple.selected + n - 1) % n
                    };
                }
            }
            ChoiceRow::Draft => {
                if let Some(draft) = &mut self.draft {
                    draft.yes = !draft.yes;
                }
            }
        }
    }

    /// The extra `hipfire pull` args for the current choices. PLE auto passes no
    /// flag (the CLI default is auto); draft yes passes none (the default).
    pub fn extra_args(&self) -> Vec<String> {
        let mut args = Vec::new();
        if let Some(ple) = &self.ple {
            if let Some(option) = ple.options.get(ple.selected) {
                if option.label != "auto" {
                    args.push("--ple".to_string());
                    args.push(option.label.to_string());
                }
            }
        }
        if self.draft.as_ref().is_some_and(|d| !d.yes) {
            args.push("--no-draft".to_string());
        }
        args
    }
}

/// `1.2 GB` (decimal, like the registry's `size_gb`), or empty when unknown.
pub fn format_size(bytes: Option<u64>) -> String {
    match bytes {
        Some(b) if b >= 1_000_000_000 => format!("{:.1} GB", b as f64 / 1e9),
        Some(b) => format!("{:.0} MB", b as f64 / 1e6),
        None => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sc(size: Option<u64>) -> Sidecar {
        Sidecar {
            file: "x".into(),
            sha256: None,
            size_bytes: size,
        }
    }

    #[test]
    fn plain_entry_has_no_chooser() {
        assert!(PullChoice::for_entry("a:b", &ModelEntry::default()).is_none());
    }

    #[test]
    fn ple_offers_only_published_formats_and_maps_to_flags() {
        let mut entry = ModelEntry::default();
        entry.ple.insert("q4".into(), sc(Some(1_500_000_000)));
        let mut c = PullChoice::for_entry("a:b", &entry).unwrap();
        let labels: Vec<_> = c
            .ple
            .as_ref()
            .unwrap()
            .options
            .iter()
            .map(|o| o.label)
            .collect();
        assert_eq!(labels, ["auto", "q4"]);
        assert!(c.draft.is_none());
        assert!(c.extra_args().is_empty());
        c.change(true);
        assert_eq!(c.extra_args(), ["--ple", "q4"]);
        c.change(true); // wraps back to auto
        assert!(c.extra_args().is_empty());
    }

    #[test]
    fn draft_only_entry_toggles_no_draft() {
        let entry = ModelEntry {
            dflash: Some(sc(None)),
            ..ModelEntry::default()
        };
        let mut c = PullChoice::for_entry("a:b", &entry).unwrap();
        assert_eq!(c.row, ChoiceRow::Draft);
        assert!(c.extra_args().is_empty());
        c.change(true);
        assert_eq!(c.extra_args(), ["--no-draft"]);
    }

    #[test]
    fn row_movement_is_clamped() {
        let mut entry = ModelEntry {
            mtp: Some(sc(None)),
            ..ModelEntry::default()
        };
        entry.ple.insert("q8".into(), sc(None));
        let mut c = PullChoice::for_entry("a:b", &entry).unwrap();
        assert_eq!(c.row, ChoiceRow::Ple);
        c.move_row(false);
        assert_eq!(c.row, ChoiceRow::Ple);
        c.move_row(true);
        assert_eq!(c.row, ChoiceRow::Draft);
        c.move_row(true);
        assert_eq!(c.row, ChoiceRow::Draft);
    }

    #[test]
    fn sizes_format() {
        assert_eq!(format_size(Some(1_234_000_000)), "1.2 GB");
        assert_eq!(format_size(Some(420_000_000)), "420 MB");
        assert_eq!(format_size(None), "");
    }
}
