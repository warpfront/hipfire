//! Host-side speculative spine proposal plumbing.
//!
//! This module does not implement an XDNA kernel. It provides the small,
//! testable contract used by DFlash experiments to ask an external sidecar for
//! a short token spine, gate the response, and feed accepted proposals into the
//! existing target-verify path.

use serde::{Deserialize, Serialize};
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpecSpineMode {
    Off,
    Shadow,
    Active,
}

impl SpecSpineMode {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "off" | "0" | "false" => Some(Self::Off),
            "shadow" | "observe" => Some(Self::Shadow),
            "active" | "on" | "1" | "true" => Some(Self::Active),
            _ => None,
        }
    }

    pub fn is_enabled(self) -> bool {
        !matches!(self, Self::Off)
    }

    pub fn is_active(self) -> bool {
        matches!(self, Self::Active)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SpecSpineGate {
    pub min_len: usize,
    pub max_len: usize,
    pub min_confidence: f32,
}

impl Default for SpecSpineGate {
    fn default() -> Self {
        Self {
            min_len: 3,
            max_len: 8,
            min_confidence: 0.60,
        }
    }
}

impl SpecSpineGate {
    pub fn accept(&self, proposal: &SpecSpineProposal) -> Option<Vec<u32>> {
        if proposal.confidence < self.min_confidence {
            return None;
        }
        let n = proposal.tokens.len().min(self.max_len);
        if n < self.min_len {
            return None;
        }
        Some(proposal.tokens[..n].to_vec())
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SpecSpineRequest {
    pub position: usize,
    pub seed_token: u32,
    pub max_tokens: usize,
    pub prompt_len: usize,
    pub tail_tokens: Vec<u32>,
    pub pld_tokens: Option<Vec<u32>>,
    pub pld_consensus: Option<usize>,
}

impl SpecSpineRequest {
    pub fn new(
        position: usize,
        seed_token: u32,
        max_tokens: usize,
        prompt_len: usize,
        tail_tokens: Vec<u32>,
    ) -> Self {
        Self {
            position,
            seed_token,
            max_tokens,
            prompt_len,
            tail_tokens,
            pld_tokens: None,
            pld_consensus: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpecSpineProposal {
    pub tokens: Vec<u32>,
    pub confidence: f32,
    pub source: String,
}

impl SpecSpineProposal {
    pub fn prefix_match_len(&self, committed_tail: &[u32]) -> usize {
        self.tokens
            .iter()
            .zip(committed_tail.iter())
            .take_while(|(a, b)| a == b)
            .count()
    }
}

#[derive(Debug, Deserialize)]
struct RawSpecSpineResponse {
    tokens: Vec<u32>,
    confidence: Option<f32>,
    source: Option<String>,
}

pub fn parse_spec_spine_response(output: &str) -> Result<SpecSpineProposal, serde_json::Error> {
    let raw: RawSpecSpineResponse = serde_json::from_str(output)?;
    Ok(SpecSpineProposal {
        tokens: raw.tokens,
        confidence: raw.confidence.unwrap_or(1.0),
        source: raw.source.unwrap_or_else(|| "external".to_string()),
    })
}

#[derive(Debug, Clone)]
pub struct CommandSpecSpineProvider {
    pub program: PathBuf,
    pub args: Vec<String>,
}

impl CommandSpecSpineProvider {
    pub fn new(program: impl Into<PathBuf>) -> Self {
        Self {
            program: program.into(),
            args: Vec::new(),
        }
    }

    pub fn propose(&self, req: &SpecSpineRequest) -> io::Result<SpecSpineProposal> {
        let mut child = Command::new(&self.program)
            .args(&self.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;

        {
            let stdin = child.stdin.as_mut().ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "spec spine command stdin unavailable",
                )
            })?;
            serde_json::to_writer(&mut *stdin, req)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
            stdin.write_all(b"\n")?;
        }

        let out = child.wait_with_output()?;
        if !out.status.success() {
            let stderr = String::from_utf8_lossy(&out.stderr);
            return Err(io::Error::new(
                io::ErrorKind::Other,
                format!(
                    "spec spine command exited with {}: {}",
                    out.status,
                    stderr.trim()
                ),
            ));
        }

        let stdout = String::from_utf8_lossy(&out.stdout);
        parse_spec_spine_response(stdout.trim())
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }
}

#[derive(Debug, Default, Clone)]
pub struct SpecSpineStats {
    pub proposed_cycles: usize,
    pub gated_cycles: usize,
    pub shadow_cycles: usize,
    pub active_cycles: usize,
    pub provider_errors: usize,
    pub proposed_tokens: usize,
    pub active_accepted_tokens: usize,
    pub shadow_prefix_tokens: usize,
    pub shadow_first_token_hits: usize,
}

impl SpecSpineStats {
    pub fn record_provider_error(&mut self) {
        self.provider_errors += 1;
    }

    pub fn record_proposal(&mut self, proposal: &SpecSpineProposal) {
        self.proposed_cycles += 1;
        self.proposed_tokens += proposal.tokens.len();
    }

    pub fn record_gated(&mut self) {
        self.gated_cycles += 1;
    }

    pub fn record_shadow(&mut self, proposal: &SpecSpineProposal, committed_tail: &[u32]) {
        self.shadow_cycles += 1;
        let matched = proposal.prefix_match_len(committed_tail);
        self.shadow_prefix_tokens += matched;
        if matched > 0 {
            self.shadow_first_token_hits += 1;
        }
    }

    pub fn record_active(&mut self, accepted_tokens: usize) {
        self.active_cycles += 1;
        self.active_accepted_tokens += accepted_tokens;
    }

    pub fn mean_proposed_len(&self) -> f32 {
        if self.proposed_cycles == 0 {
            0.0
        } else {
            self.proposed_tokens as f32 / self.proposed_cycles as f32
        }
    }

    pub fn active_tau(&self) -> f32 {
        if self.active_cycles == 0 {
            0.0
        } else {
            self.active_accepted_tokens as f32 / self.active_cycles as f32
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_shadow_and_active_modes() {
        assert!(matches!(
            SpecSpineMode::parse("shadow"),
            Some(SpecSpineMode::Shadow)
        ));
        assert!(matches!(
            SpecSpineMode::parse("active"),
            Some(SpecSpineMode::Active)
        ));
        assert!(matches!(
            SpecSpineMode::parse("off"),
            Some(SpecSpineMode::Off)
        ));
        assert!(SpecSpineMode::parse("invalid").is_none());
    }

    #[test]
    fn gate_requires_minimum_length_and_confidence() {
        let cfg = SpecSpineGate {
            min_len: 3,
            max_len: 8,
            min_confidence: 0.60,
        };

        let short = SpecSpineProposal {
            tokens: vec![1, 2],
            confidence: 0.99,
            source: "test".to_string(),
        };
        let weak = SpecSpineProposal {
            tokens: vec![1, 2, 3],
            confidence: 0.59,
            source: "test".to_string(),
        };
        let accepted = SpecSpineProposal {
            tokens: vec![1, 2, 3, 4],
            confidence: 0.80,
            source: "test".to_string(),
        };

        assert!(cfg.accept(&short).is_none());
        assert!(cfg.accept(&weak).is_none());
        assert_eq!(cfg.accept(&accepted), Some(vec![1, 2, 3, 4]));
    }

    #[test]
    fn gate_truncates_to_max_len() {
        let cfg = SpecSpineGate {
            min_len: 2,
            max_len: 4,
            min_confidence: 0.0,
        };
        let proposal = SpecSpineProposal {
            tokens: vec![10, 11, 12, 13, 14, 15],
            confidence: 1.0,
            source: "test".to_string(),
        };

        assert_eq!(cfg.accept(&proposal), Some(vec![10, 11, 12, 13]));
    }

    #[test]
    fn parses_json_response() {
        let proposal = parse_spec_spine_response(
            r#"{"tokens":[42,43,44],"confidence":0.75,"source":"npu-smoke"}"#,
        )
        .expect("parse proposal");

        assert_eq!(proposal.tokens, vec![42, 43, 44]);
        assert_eq!(proposal.confidence, 0.75);
        assert_eq!(proposal.source, "npu-smoke");
    }

    #[test]
    fn counts_prefix_match_against_committed_tail() {
        let proposal = SpecSpineProposal {
            tokens: vec![7, 8, 9],
            confidence: 0.9,
            source: "test".to_string(),
        };

        assert_eq!(proposal.prefix_match_len(&[7, 8, 10]), 2);
        assert_eq!(proposal.prefix_match_len(&[1, 2, 3]), 0);
    }

    #[test]
    fn command_provider_reads_json_proposal_from_stdout() {
        let mut provider = CommandSpecSpineProvider::new("/bin/sh");
        provider.args = vec![
            "-c".to_string(),
            "cat >/dev/null; printf '%s\n' '{\"tokens\":[5,6,7],\"confidence\":0.91,\"source\":\"sh\"}'".to_string(),
        ];
        let req = SpecSpineRequest::new(12, 99, 8, 4, vec![1, 2, 3, 99]);

        let proposal = provider.propose(&req).expect("proposal");

        assert_eq!(proposal.tokens, vec![5, 6, 7]);
        assert_eq!(proposal.confidence, 0.91);
        assert_eq!(proposal.source, "sh");
    }
}
