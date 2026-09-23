//! AfterOpen-only TokTrie mask accelerator.
//!
//! Builds a reusable vocabulary byte trie once, then on each AfterOpen `fill`
//! rebuilds a small header-continuation automaton from
//! `matcher.allowed_continuations()`, seeds it with `matcher.partial()` bytes
//! (not via `add_bias` start), and runs `TokTrie::add_bias` with `start = []`.
//! Free states write all-true for `vocab.len()` slots; every non-AfterOpen
//! constrained state falls back to the Matcher oracle `token_mask`.

use saddle_core::grammar::json::{Matcher, State};
use toktrie::{Recognizer, SimpleVob, TokRxInfo, TokTrie};

/// Sentinel: no child edge in the header automaton.
const NO_CHILD: u32 = u32::MAX;
/// Absorbing accept-all state once a full header template has been matched.
const ABSORBING: u32 = u32::MAX - 1;

/// Compact header-prefix automaton (separate from the token TokTrie).
///
/// Terminal nodes are absorbing: any further byte stays allowed so merged
/// header+body tokens match the oracle
/// `cont.starts_with(s) || s.starts_with(cont)`.
struct HeaderAuto {
    /// `child_table[node * 256 + byte] -> child node` or `NO_CHILD`.
    child_table: Vec<u32>,
    /// `terminal[node] == true` once a full header ends on that node.
    terminal: Vec<bool>,
}

impl HeaderAuto {
    fn new() -> Self {
        Self {
            child_table: Vec::new(),
            terminal: Vec::new(),
        }
    }

    fn clear(&mut self) {
        self.child_table.clear();
        self.terminal.clear();
    }

    fn node_count(&self) -> usize {
        self.terminal.len()
    }

    fn add_node(&mut self) -> u32 {
        let id = self.node_count() as u32;
        self.terminal.push(false);
        self.child_table
            .extend(std::iter::repeat(NO_CHILD).take(256));
        id
    }

    /// Rebuild from full header templates (once per fill).
    fn rebuild(&mut self, headers: &[String]) {
        self.clear();
        let _root = self.add_node();
        for h in headers {
            let mut cur = 0u32;
            for &b in h.as_bytes() {
                let slot = cur as usize * 256 + b as usize;
                if self.child_table[slot] == NO_CHILD {
                    let nxt = self.add_node();
                    self.child_table[slot] = nxt;
                }
                cur = self.child_table[cur as usize * 256 + b as usize];
            }
            self.terminal[cur as usize] = true;
        }
    }

    #[inline(always)]
    fn next(&self, state: u32, byte: u8) -> Option<u32> {
        if state == ABSORBING {
            return Some(ABSORBING);
        }
        let n = state as usize;
        // Completed header → every suffix byte is legal (absorbing).
        if self.terminal.get(n).copied().unwrap_or(false) {
            return Some(ABSORBING);
        }
        let slot = n * 256 + byte as usize;
        let child = self.child_table.get(slot).copied().unwrap_or(NO_CHILD);
        if child == NO_CHILD {
            None
        } else {
            Some(child)
        }
    }

    /// Walk `partial` from root. `None` = invalid seed (reject all non-empty).
    fn seed(&self, partial: &str) -> Option<u32> {
        let mut state = 0u32;
        // Empty automaton (no tools / no headers): only empty partial is
        // "valid" as a seed position; any byte would fail at fill time.
        if self.node_count() == 0 {
            return if partial.is_empty() {
                Some(0)
            } else {
                None
            };
        }
        for &b in partial.as_bytes() {
            match self.next(state, b) {
                Some(s) => state = s,
                None => return None,
            }
        }
        Some(state)
    }
}

/// Dynamic-depth recognizer: reusable `Vec` stack, capacity `max_token_len+1`,
/// O(1) pop via `truncate`. Avoids `StackRecognizer`'s fixed 300 limit.
struct DynRec<'a> {
    auto: &'a HeaderAuto,
    stack: &'a mut Vec<u32>,
    seed: u32,
}

impl Recognizer for DynRec<'_> {
    #[inline(always)]
    fn pop_bytes(&mut self, num: usize) {
        debug_assert!(num <= self.stack.len());
        let keep = self.stack.len() - num;
        self.stack.truncate(keep);
    }

    fn collapse(&mut self) {
        let top = self.stack.last().copied().unwrap_or(self.seed);
        self.stack.clear();
        self.stack.push(top);
    }

    fn trie_finished(&mut self) {
        self.stack.clear();
        self.stack.push(self.seed);
    }

    #[inline(always)]
    fn try_push_byte(&mut self, byte: u8) -> bool {
        let cur = *self.stack.last().expect("recognizer stack seeded");
        match self.auto.next(cur, byte) {
            Some(nxt) => {
                self.stack.push(nxt);
                true
            }
            None => false,
        }
    }
}

/// Reusable AfterOpen TokTrie mask accelerator.
pub struct AfterOpenTrie {
    trie: TokTrie,
    bitset: SimpleVob,
    /// Recognizer stack scratch (length ≤ max_token_bytes + 1).
    stack: Vec<u32>,
    /// Header automaton rebuilt each AfterOpen fill (schema-safe).
    header: HeaderAuto,
    max_token_bytes: usize,
}

impl AfterOpenTrie {
    /// Build a reusable TokTrie over exact lossy decoded-vocab `String` bytes.
    pub fn new(vocab: &[String]) -> Self {
        let words: Vec<Vec<u8>> = vocab.iter().map(|s| s.as_bytes().to_vec()).collect();
        let n = words.len() as u32;
        // EOS id is metadata only; add_bias does not auto-unmask it.
        let info = TokRxInfo::new(n, 0);
        let trie = TokTrie::from(&info, &words);
        let bitset = trie.alloc_token_set();
        let max_token_bytes = trie.max_token_len();
        let mut stack = Vec::with_capacity(max_token_bytes.saturating_add(1).max(1));
        stack.push(0);
        Self {
            trie,
            bitset,
            stack,
            header: HeaderAuto::new(),
            max_token_bytes,
        }
    }

    /// Longest vocabulary token length in bytes (TokTrie).
    pub fn max_token_bytes(&self) -> usize {
        self.max_token_bytes
    }

    /// Fill `out[..vocab.len()]` with the allowed mask for `matcher`.
    ///
    /// - Free states: all `true` (suffix of `out` untouched).
    /// - `AfterOpen`: header automaton + `add_bias` (+ empty-token restore).
    /// - All other constrained states: oracle `matcher.token_mask`.
    pub fn fill(&mut self, matcher: &Matcher, vocab: &[String], out: &mut [bool]) {
        debug_assert!(out.len() >= vocab.len());

        if matcher.is_free() {
            for slot in out.iter_mut().take(vocab.len()) {
                *slot = true;
            }
            return;
        }

        if matcher.state() != &State::AfterOpen {
            matcher.token_mask(vocab, out);
            return;
        }

        self.fill_after_open(matcher, vocab, out);
    }

    fn fill_after_open(&mut self, matcher: &Matcher, vocab: &[String], out: &mut [bool]) {
        let n = vocab.len();
        debug_assert!(out.len() >= n);

        // Fresh headers every fill — tool schemas may change between calls.
        let conts = matcher.allowed_continuations();
        self.header.rebuild(&conts);

        let partial = matcher.partial();
        let seed = match self.header.seed(partial) {
            Some(s) => s,
            None => {
                // Invalid partial: oracle rejects every non-empty token.
                for slot in out.iter_mut().take(n) {
                    *slot = false;
                }
                force_allow_empty(vocab, out);
                return;
            }
        };

        self.bitset.set_all(false);
        self.stack.clear();
        self.stack.push(seed);

        // Disjoint-field borrows: header/stack vs trie/bitset.
        let AfterOpenTrie {
            trie,
            bitset,
            stack,
            header,
            ..
        } = self;
        {
            let mut rec = DynRec {
                auto: header,
                stack,
                seed,
            };
            // start MUST be [] — partial is already in recognizer seed state.
            trie.add_bias(&mut rec, bitset, &[]);
        }

        // SimpleVob → bool mask (do NOT use SimpleVob::apply_to).
        for slot in out.iter_mut().take(n) {
            *slot = false;
        }
        bitset.iter_set_entries(|id| {
            if id < n {
                out[id] = true;
            }
        });

        // Empty decoded ids never enter TokTrie::from; restore via string oracle.
        force_allow_empty(vocab, out);
    }
}

#[inline]
fn force_allow_empty(vocab: &[String], out: &mut [bool]) {
    for (i, s) in vocab.iter().enumerate() {
        if s.is_empty() {
            out[i] = true;
        }
    }
}
