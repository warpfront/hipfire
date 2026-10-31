//! Surya-2.5 compact block output → HTML.
//!
//! Surya-ocr checkpoints (Datalab's `surya-ocr-2.5`, and anything else trained
//! on that contract) do not emit `<div data-label data-bbox>` markup. A
//! top-level block is a *label token* plus four *coordinate tokens* on a
//! 0–1000 grid, followed by the block's HTML with no `</div>`; `</td>`,
//! `</th>` and `</tr>` are omitted, a `data-bbox` on any other tag is written
//! as `<|box|>` plus four coordinates before that tag, and merged table cells
//! are `data-kind="parent"` / `"ghost"` cells with `data-cs` / `data-rs`.
//!
//! This is a port of `surya.inference.compact.decode` plus the
//! `surya.inference.merged_cells.convert_merged_cells` pass the surya client
//! runs afterwards, entered the way `surya.inference.parsers` enters them
//! (`_strip_fences` + `strip`, then decode, then merged cells). It exists so
//! `serve` can answer a request in the shape every downstream consumer of
//! this model expects, without the caller having to install the Python
//! package.
//!
//! Differential-tested against the Python reference over the captured pages
//! and generated documents; what is deliberately *not* mirrored is
//! BeautifulSoup's HTML error recovery, which re-balances malformed markup the
//! linear decoder passes through verbatim. Two consequences: a tag left open
//! at the end of a block stays open here (as in `compact.decode`), and the
//! `truncated` flag is computed on that output, so a page cut off mid-table is
//! flagged where surya's re-serialized check no longer sees it.
//!
//! Raw output stays the default: the model card requires a server to return
//! the special tokens (`llama-server --special`), because surya's own client
//! parses exactly that string. Clients opt into the decoded form per request
//! (`ocr_format`), and a client that wants raw output — the surya package
//! itself — simply omits the field.

use serde_json::json;

/// Every label token that can head a top-level block (`LAYOUT_LABELS`).
pub(crate) const BLOCK_LABELS: &[&str] = &[
    "Caption",
    "Footnote",
    "Equation-Block",
    "List-Group",
    "Page-Header",
    "Page-Footer",
    "Image",
    "Section-Header",
    "Table",
    "Text",
    "Complex-Block",
    "Code-Block",
    "Form",
    "Table-Of-Contents",
    "Figure",
    "Chemical-Block",
    "Diagram",
    "Bibliography",
    "Blank-Page",
    "Picture",
];

/// Attribute ordering the Python decoder canonicalizes to when it has to
/// rebuild a tag (the `<|box|>` path). Anything unlisted sorts last by name.
const ATTR_ORDER: &[&str] = &[
    "type",
    "checked",
    "value",
    "colspan",
    "rowspan",
    "data-cs",
    "data-rs",
    "data-kind",
    "display",
    "class",
    "align",
    "style",
    "border",
    "href",
    "alt",
    "text",
];

const BOOLEAN_ATTRS: &[&str] = &["checked"];
const VOID_TAGS: &[&str] = &["br", "hr", "img", "input"];
/// End tags the compact format leaves out.
const OMIT_END: &[&str] = &["td", "th", "tr"];

/// Requested rendering of a compact-block assistant turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OcrFormat {
    /// `<div data-bbox="…" data-label="…">…</div>` per block, newline-joined.
    Html,
    /// JSON array of `{"label", "bbox", "html"}` (stringified into `content`).
    Blocks,
}

impl OcrFormat {
    /// Parse the request field's value. `None` is a client error.
    pub(crate) fn parse(value: &str) -> Option<Self> {
        match value {
            "html" => Some(Self::Html),
            "blocks" => Some(Self::Blocks),
            _ => None,
        }
    }

    /// The values accepted on the wire (for error text).
    pub(crate) fn wire_values() -> &'static str {
        "\"html\", \"blocks\" or \"raw\""
    }

    /// Map the serve-level default (`serve.ocr_format` /
    /// `HIPFIRE_SERVE_OCR_FORMAT`) to a rendering. `raw` is the identity, so
    /// it maps to `None`; an unknown value is `None` at the outer level.
    pub(crate) fn from_config(value: &str) -> Option<Option<Self>> {
        match value {
            "raw" | "" => Some(None),
            "html" => Some(Some(Self::Html)),
            "blocks" => Some(Some(Self::Blocks)),
            _ => None,
        }
    }

    /// The config spelling, for `/health` and logs.
    pub(crate) fn config_value(self) -> &'static str {
        match self {
            Self::Html => "html",
            Self::Blocks => "blocks",
        }
    }
}

/// One decoded block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Block {
    pub(crate) label: String,
    pub(crate) bbox: [u16; 4],
    pub(crate) html: String,
}

impl Block {
    fn bbox_string(&self) -> String {
        self.bbox
            .iter()
            .map(u16::to_string)
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn div(&self) -> String {
        format!(
            "<div data-bbox=\"{}\" data-label=\"{}\">{}</div>",
            self.bbox_string(),
            self.label,
            self.html
        )
    }

    /// One entry of the `blocks` rendering: the maintainers'
    /// `ParsedFullPageBlock` (`label`, `bbox`, inner `html`, `truncated`).
    ///
    /// The inner HTML has every nested `data-bbox` / `data-label` removed, as
    /// `parse_full_page_html` does ("so downstream consumers don't see model
    /// debug info on every child element"). This is the structured form; the
    /// `html` rendering keeps them, since it is the raw decoder output a
    /// client's own parser reads.
    fn parsed_json(&self, truncated: bool) -> serde_json::Value {
        json!({
            "label": self.label,
            "bbox": self.bbox,
            "html": strip_nested_data_attrs(&self.html),
            "truncated": truncated,
        })
    }
}

/// Decode compact block output into the `<div data-label data-bbox>` HTML that
/// surya's parsers (`clean_block_html` / `parse_full_page_html`) read.
///
/// The whole entry point mirrors `parse_full_page_html`: code fences around
/// the answer are stripped first, then `compact.decode` runs, then the
/// merged-cell pass. Blocks and any loose text between them keep their order,
/// so text after an explicit `</div>` stays where the model wrote it.
///
/// Output with no block headers is returned unchanged: a text-only answer is
/// not an OCR page, and rewriting it would hide that.
pub(crate) fn decode_html(text: &str) -> String {
    let decoded = Decoder::run(strip_fences(text));
    if decoded.block_count() == 0 {
        return text.to_owned();
    }
    decoded.render()
}

/// Same decode, rendered as a JSON array of the maintainers' parsed blocks
/// (stringified, because OpenAI's `message.content` is a string). Output with
/// no block headers is passed through unchanged, matching [`decode_html`].
pub(crate) fn blocks_json(text: &str) -> String {
    let cleaned = strip_fences(text);
    let decoded = Decoder::run(cleaned);
    let count = decoded.block_count();
    if count == 0 {
        return text.to_owned();
    }
    // `parse_full_page_html` marks the last block truncated when the answer
    // ends mid-token or leaves structural tags open (the token cap cut it).
    let truncated = partial_compact_header(cleaned) || looks_truncated(&decoded.render());
    let values: Vec<serde_json::Value> = decoded
        .blocks()
        .enumerate()
        .map(|(index, block)| block.parsed_json(truncated && index + 1 == count))
        .collect();
    serde_json::Value::Array(values).to_string()
}

/// The decoded blocks, for callers that want them structured. Fences are
/// stripped, as in [`decode_html`]; merged cells are already converted.
pub(crate) fn decode_blocks(text: &str) -> Vec<Block> {
    Decoder::run(strip_fences(text)).blocks().cloned().collect()
}

/// `surya.inference.parsers._strip_fences`: strip a ```` ```lang ```` wrapper
/// (and surrounding whitespace) from an answer.
fn strip_fences(text: &str) -> &str {
    let mut cleaned = text.trim();
    // `^```[a-zA-Z]*\n` — a fence line, not a literal ``` inside the answer.
    if let Some(rest) = cleaned.strip_prefix("```") {
        if let Some((fence, body)) = rest.split_once('\n') {
            if fence.bytes().all(|b| b.is_ascii_alphabetic()) {
                cleaned = body;
            }
        }
    }
    // `\n```\s*$` — the closer only counts on its own line at the very end.
    let core = cleaned.trim_end();
    match core.strip_suffix("```") {
        Some(head) => head.strip_suffix('\n').unwrap_or(head),
        None => core,
    }
}

/// `_strip_fences` for a stream's lead-in text: the wrapper is at the start of
/// the answer, so only the prefix is removed here.
fn strip_fence_prefix(text: &str) -> &str {
    let trimmed = text.trim_start();
    let Some(rest) = trimmed.strip_prefix("```") else {
        return trimmed;
    };
    match rest.split_once('\n') {
        Some((fence, body)) if fence.bytes().all(|b| b.is_ascii_alphabetic()) => body,
        _ => rest,
    }
}

/// `_strip_fences` for a stream's tail: trailing whitespace, plus a closing
/// fence on its own line.
fn strip_tail(text: &str) -> &str {
    let core = text.trim_end();
    match core.strip_suffix("```") {
        Some(head) => head.strip_suffix('\n').unwrap_or(head),
        None => core,
    }
}

/// Structural tags `html_looks_truncated` balances (`_BALANCED_TAGS`).
const BALANCED_TAGS: &[&str] = &[
    "div", "table", "math", "pre", "code", "p", "h1", "h2", "h3", "h4", "h5", "ul", "ol", "li",
];

/// `_PARTIAL_COMPACT_HEADER_RE`: a label token with 0–3 coordinates, or a
/// dangling `<|`, at the very end of the answer.
fn partial_compact_header(text: &str) -> bool {
    let bytes = text.as_bytes();
    for start in 0..bytes.len() {
        if bytes[start] != b'<' || bytes.get(start + 1) != Some(&b'|') {
            continue;
        }
        let tail = &text[start..];
        let after_open = &tail[2..];
        if let Some(close) = after_open.find("|>") {
            let label = &after_open[..close];
            if !label.is_empty()
                && label.bytes().all(|b| b.is_ascii_alphabetic() || b == b'-')
                && is_partial_header_tail(&after_open[close + 2..])
            {
                return true;
            }
        }
        // `<|[^<>|]*$`.
        if after_open.chars().all(|c| !matches!(c, '<' | '>' | '|')) {
            return true;
        }
    }
    false
}

/// `(?:<\|\d{1,4}\|>){0,3}$` — zero to three coordinate tokens, then the end.
fn is_partial_header_tail(tail: &str) -> bool {
    let mut rest = tail;
    let mut coords = 0;
    while let Some(after) = rest.strip_prefix("<|") {
        let Some(close) = after.find("|>") else {
            return false;
        };
        let digits = &after[..close];
        if digits.is_empty() || digits.len() > 4 || !digits.bytes().all(|b| b.is_ascii_digit()) {
            return false;
        }
        rest = &after[close + 2..];
        coords += 1;
        if coords > 3 {
            return false;
        }
    }
    rest.is_empty()
}

/// `surya.inference.parsers.html_looks_truncated`: an open structural tag with
/// no closer, a half-written tag, or a half-written block header, at the end.
fn looks_truncated(html: &str) -> bool {
    for tag in BALANCED_TAGS {
        if count_tag(html, tag, false) > count_tag(html, tag, true) {
            return true;
        }
    }
    // `<[^<>]*$`.
    if let Some(at) = html.rfind('<') {
        if !html[at..].contains('>') {
            return true;
        }
    }
    partial_compact_header(html)
}

/// `<(tag)(?=[\s>/])` opens, `</(tag)\s*>` closes, case-insensitive.
fn count_tag(html: &str, name: &str, close: bool) -> usize {
    let needle = if close { "</" } else { "<" };
    let mut count = 0;
    for (start, _) in html.match_indices(needle) {
        let name_at = start + needle.len();
        if !html
            .get(name_at..name_at + name.len())
            .is_some_and(|slice| slice.eq_ignore_ascii_case(name))
        {
            continue;
        }
        let mut rest = html[name_at + name.len()..].chars();
        let matched = if close {
            let mut next = rest.next();
            while next.is_some_and(char::is_whitespace) {
                next = rest.next();
            }
            next == Some('>')
        } else {
            match rest.next() {
                Some(c) => c.is_whitespace() || c == '>' || c == '/',
                None => false,
            }
        };
        count += usize::from(matched);
    }
    count
}

// ---------------------------------------------------------------------------
// Token scanning
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct TagRef<'a> {
    start: usize,
    end: usize,
    /// The tag text exactly as written.
    raw: &'a str,
    close: bool,
    /// Name exactly as written, e.g. `TD`.
    name_raw: &'a str,
    /// Attribute text between the name and the closing `>` (may lead with
    /// whitespace; excludes a trailing `/`).
    attrs: &'a str,
    selfclose: bool,
}

impl<'a> TagRef<'a> {
    fn name_lower(&self) -> String {
        self.name_raw.to_ascii_lowercase()
    }
}

/// One unit of model output.
#[derive(Debug)]
enum Token<'a> {
    Header { label: &'a str, bbox: [u16; 4] },
    Box([u16; 4]),
    Tag(TagRef<'a>),
    Text(&'a str),
}

/// A header token: `<|Label|>` + four coordinate tokens, all adjacent.
fn header_at(s: &str) -> Option<(usize, &str, [u16; 4])> {
    let rest = s.strip_prefix("<|")?;
    let close = rest.find("|>")?;
    let label = &rest[..close];
    if !BLOCK_LABELS.contains(&label) {
        return None;
    }
    let mut consumed = 2 + close + 2;
    let mut bbox = [0u16; 4];
    for slot in bbox.iter_mut() {
        let (len, value) = coord_at(&s[consumed..])?;
        *slot = value;
        consumed += len;
    }
    Some((consumed, label, bbox))
}

/// One coordinate token: `<|123|>` (1–4 digits).
fn coord_at(s: &str) -> Option<(usize, u16)> {
    let rest = s.strip_prefix("<|")?;
    let close = rest.find("|>")?;
    let digits = &rest[..close];
    if digits.is_empty() || digits.len() > 4 || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    Some((2 + close + 2, digits.parse().ok()?))
}

/// `<|box|>` + four coordinates: a `data-bbox` for the tag that follows.
fn box_at(s: &str) -> Option<(usize, [u16; 4])> {
    if !s.starts_with("<|box|>") {
        return None;
    }
    let mut consumed = "<|box|>".len();
    let mut bbox = [0u16; 4];
    for slot in bbox.iter_mut() {
        let (len, value) = coord_at(&s[consumed..])?;
        *slot = value;
        consumed += len;
    }
    Some((consumed, bbox))
}

/// A plain HTML tag: `<` [`/`] name attrs [`/`] `>`.
///
/// Mirrors `compact._TAG_RE`: the attribute run cannot contain `<` or `>`, and
/// a non-empty attribute run must start with whitespace. Malformed tags stay
/// literal text, which is what the Python decoder does too.
fn tag_at(s: &str) -> Option<TagRef<'_>> {
    let bytes = s.as_bytes();
    if bytes.first() != Some(&b'<') {
        return None;
    }
    let mut i = 1;
    let close = bytes.get(i) == Some(&b'/');
    if close {
        i += 1;
    }
    let name_start = i;
    if !matches!(bytes.get(i), Some(b) if b.is_ascii_alphabetic()) {
        return None;
    }
    i += 1;
    while matches!(bytes.get(i), Some(b) if b.is_ascii_alphanumeric() || *b == b'-') {
        i += 1;
    }
    let name_raw = &s[name_start..i];
    let gt = i + s[i..].find('>')?;
    let raw_attrs = &s[i..gt];
    if raw_attrs.contains('<') {
        return None;
    }
    // `\s` in `compact._TAG_RE` is Unicode whitespace, not just ASCII.
    let trimmed = raw_attrs.trim_end_matches(char::is_whitespace);
    let (attrs, selfclose) = match trimmed.strip_suffix('/') {
        Some(prefix) => (prefix, true),
        None => (trimmed, false),
    };
    if !attrs.is_empty() && !attrs.starts_with(char::is_whitespace) {
        return None;
    }
    Some(TagRef {
        start: 0,
        end: gt + 1,
        raw: &s[..gt + 1],
        close,
        name_raw,
        attrs,
        selfclose,
    })
}

// ---------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------

/// One unit of decoded output, in the order the model wrote it.
enum Piece {
    /// Text outside any block: an answer's lead-in, or text a model wrote
    /// after it closed a block with an explicit `</div>`.
    Text(String),
    Block(Block),
}

struct Decoded {
    pieces: Vec<Piece>,
}

impl Decoded {
    fn block_count(&self) -> usize {
        self.pieces
            .iter()
            .filter(|piece| matches!(piece, Piece::Block(_)))
            .count()
    }

    fn blocks(&self) -> impl Iterator<Item = &Block> {
        self.pieces.iter().filter_map(|piece| match piece {
            Piece::Block(block) => Some(block),
            Piece::Text(_) => None,
        })
    }

    /// The decoded document: pieces in order, one newline between a preceding
    /// piece and a block (`compact.decode`'s `if out and not out[-1].endswith
    /// ("\n")` rule).
    fn render(&self) -> String {
        let mut html = String::new();
        for piece in &self.pieces {
            match piece {
                Piece::Text(text) => html.push_str(text),
                Piece::Block(block) => {
                    if !html.is_empty() && !html.ends_with('\n') {
                        html.push('\n');
                    }
                    html.push_str(&block.div());
                }
            }
        }
        html
    }
}

struct Decoder {
    pieces: Vec<Piece>,
    /// Content of the block being decoded.
    out: Vec<String>,
    stack: Vec<String>,
    in_block: bool,
    label: String,
    bbox: [u16; 4],
    pending_box: Option<String>,
}

impl Decoder {
    fn run(text: &str) -> Decoded {
        let mut decoder = Self {
            pieces: Vec::new(),
            out: Vec::new(),
            stack: Vec::new(),
            in_block: false,
            label: String::new(),
            bbox: [0; 4],
            pending_box: None,
        };
        let mut pos = 0usize;
        while pos < text.len() {
            let rest = &text[pos..];
            if let Some((consumed, label, bbox)) = header_at(rest) {
                decoder.close_block();
                decoder.stack.clear();
                decoder.stack.push("div".to_owned());
                decoder.in_block = true;
                decoder.label = label.to_owned();
                decoder.bbox = bbox;
                decoder.pending_box = None;
                pos += consumed;
                continue;
            }
            if let Some((consumed, bbox)) = box_at(rest) {
                decoder.pending_box = Some(
                    bbox.iter()
                        .map(u16::to_string)
                        .collect::<Vec<_>>()
                        .join(" "),
                );
                pos += consumed;
                continue;
            }
            if let Some(tag) = tag_at(rest) {
                let len = tag.end;
                let mut tag = tag;
                tag.start = pos;
                tag.end = pos + len;
                decoder.on_tag(tag);
                pos += len;
                continue;
            }
            // Unmatched `<` is text; consume up to the next candidate. Scan
            // bytes rather than chars: `<` is ASCII, so a byte scan never
            // lands inside a multi-byte character (`<भारत` used to panic
            // here, failing the whole request).
            let end = rest.as_bytes()[1..]
                .iter()
                .position(|byte| *byte == b'<')
                .map_or(rest.len(), |offset| offset + 1);
            decoder.emit(&rest[..end]);
            pos += end;
        }
        decoder.close_block();
        Decoded {
            pieces: decoder.pieces,
        }
    }

    fn emit(&mut self, s: &str) {
        if self.in_block {
            self.out.push(s.to_owned());
            return;
        }
        match self.pieces.last_mut() {
            Some(Piece::Text(text)) => text.push_str(s),
            _ => self.pieces.push(Piece::Text(s.to_owned())),
        }
    }

    fn close_cells(&mut self, upto: &[&str]) {
        loop {
            let Some(top) = self.stack.last() else {
                break;
            };
            if !OMIT_END.contains(&top.as_str()) || upto.contains(&top.as_str()) {
                break;
            }
            let top = self.stack.pop().expect("top checked above");
            self.emit(&format!("</{top}>"));
        }
    }

    fn close_block(&mut self) {
        if !self.in_block {
            return;
        }
        self.close_cells(&[]);
        while let Some(top) = self.stack.last() {
            if !OMIT_END.contains(&top.as_str()) {
                break;
            }
            let top = self.stack.pop().expect("top checked above");
            self.emit(&format!("</{top}>"));
        }
        if self.stack.last().map(String::as_str) == Some("div") {
            self.stack.pop();
        }
        let inner = self.out.concat();
        self.pieces.push(Piece::Block(Block {
            label: std::mem::take(&mut self.label),
            bbox: self.bbox,
            html: convert_merged_cells(&inner),
        }));
        self.out.clear();
        self.in_block = false;
    }

    fn on_tag(&mut self, tag: TagRef<'_>) {
        let name = tag.name_lower();
        if tag.close {
            // A box token never applies across a closing tag.
            self.pending_box = None;
            if name == "table" || name == "thead" || name == "tbody" {
                let upto: &[&str] = if name == "table" {
                    &["table"]
                } else {
                    &["table", "thead", "tbody"]
                };
                self.close_cells(upto);
            }
            if name == "div" && self.in_block && self.stack.len() == 1 && self.stack[0] == "div" {
                self.close_block();
                return;
            }
            if let Some(match_at) = self.stack.iter().rposition(|open| *open == name) {
                while self.stack.len() > match_at + 1 {
                    let top = self.stack.pop().expect("above match");
                    if OMIT_END.contains(&top.as_str()) {
                        self.emit(&format!("</{top}>"));
                    }
                }
                self.stack.pop();
            }
            self.emit(tag.raw);
            return;
        }
        if name == "td" || name == "th" {
            if matches!(self.stack.last().map(String::as_str), Some("td" | "th")) {
                let top = self.stack.pop().expect("checked above");
                self.emit(&format!("</{top}>"));
            }
        } else if name == "tr" {
            self.close_cells(&["table", "thead", "tbody"]);
        } else if name == "thead" || name == "tbody" {
            self.close_cells(&["table"]);
        }
        let rendered = match self.pending_box.take() {
            Some(bbox) => {
                let attrs = if tag.attrs.trim().is_empty() {
                    format!(" data-bbox=\"{bbox}\"")
                } else {
                    format!("{} data-bbox=\"{bbox}\"", tag.attrs)
                };
                canonical_tag(false, tag.name_raw, &attrs, tag.selfclose)
            }
            None => tag.raw.to_owned(),
        };
        self.emit(&rendered);
        if !VOID_TAGS.contains(&name.as_str()) && !tag.selfclose {
            self.stack.push(name);
        }
    }
}

// ---------------------------------------------------------------------------
// Tag canonicalization (the `<|box|>` path)
// ---------------------------------------------------------------------------

fn attr_key(name: &str) -> (usize, &str) {
    match ATTR_ORDER.iter().position(|key| *key == name) {
        Some(index) => (index, ""),
        None => (ATTR_ORDER.len(), name),
    }
}

/// One spelling per tag, as `compact.canonical_tag` writes it: lowercase name,
/// fixed attribute order, double quotes, booleans valueless, void tags
/// self-closed.
fn canonical_tag(close: bool, name: &str, attrs: &str, selfclose: bool) -> String {
    let name = name.to_ascii_lowercase();
    if close {
        return format!("</{name}>");
    }
    let mut pairs = parse_attrs(attrs);
    pairs.sort_by(|left, right| attr_key(&left.name).cmp(&attr_key(&right.name)));
    let mut parts = Vec::with_capacity(pairs.len());
    for attr in pairs {
        if BOOLEAN_ATTRS.contains(&attr.name.as_str()) {
            parts.push(attr.name);
        } else if let Some(value) = attr.value {
            parts.push(format!(
                "{}=\"{}\"",
                attr.name,
                value.replace('"', "&quot;")
            ));
        } else {
            parts.push(format!("{}=\"\"", attr.name));
        }
    }
    let inner = if parts.is_empty() {
        name.clone()
    } else {
        format!("{name} {}", parts.join(" "))
    };
    if VOID_TAGS.contains(&name.as_str()) || selfclose {
        format!("<{inner}/>")
    } else {
        format!("<{inner}>")
    }
}

/// One attribute of an attribute run, mirroring `compact._ATTR_RE`.
struct Attr {
    name: String,
    value: Option<String>,
    /// Byte range of the `name[=value]` run inside the attribute text.
    span: (usize, usize),
    /// Byte index of the whitespace run that precedes the attribute.
    ws_start: usize,
}

/// Attribute names/values in an attribute run, mirroring `compact._ATTR_RE`.
///
/// Whitespace is Unicode whitespace, as Python's `\s` is: a stray no-break
/// space between attributes must not truncate the tag.
fn parse_attrs(raw: &str) -> Vec<Attr> {
    let chars: Vec<(usize, char)> = raw.char_indices().collect();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < chars.len() {
        let ws_start = i;
        while i < chars.len() && chars[i].1.is_whitespace() {
            i += 1;
        }
        if i >= chars.len() {
            break;
        }
        let is_name_start = |c: char| c.is_ascii_alphabetic() || c == '_' || c == ':';
        if !is_name_start(chars[i].1) {
            i += 1;
            continue;
        }
        let name_start = chars[i].0;
        i += 1;
        while i < chars.len()
            && (chars[i].1.is_ascii_alphanumeric() || matches!(chars[i].1, '-' | '_' | ':' | '.'))
        {
            i += 1;
        }
        // Name characters are ASCII, so the name ends one byte past the last.
        let name_end = chars[i - 1].0 + 1;
        let name = raw[name_start..name_end].to_ascii_lowercase();
        let mut end = name_end;
        let mut cursor = i;
        while cursor < chars.len() && chars[cursor].1.is_whitespace() {
            cursor += 1;
        }
        let mut value = None;
        if cursor < chars.len() && chars[cursor].1 == '=' {
            let mut at = cursor + 1;
            while at < chars.len() && chars[at].1.is_whitespace() {
                at += 1;
            }
            if at < chars.len() && matches!(chars[at].1, '"' | '\'') {
                let quote = chars[at].1;
                let value_start = chars[at].0 + 1;
                let mut close = at + 1;
                while close < chars.len() && chars[close].1 != quote {
                    close += 1;
                }
                if close < chars.len() {
                    value = Some(raw[value_start..chars[close].0].to_owned());
                    end = chars[close].0 + 1;
                    i = close + 1;
                } else {
                    value = Some(raw[value_start..].to_owned());
                    end = raw.len();
                    i = chars.len();
                }
            } else if at < chars.len() {
                let value_start = chars[at].0;
                let mut close = at;
                while close < chars.len()
                    && !chars[close].1.is_whitespace()
                    && !matches!(chars[close].1, '"' | '\'' | '=' | '<' | '>' | '`')
                {
                    close += 1;
                }
                // `[^\s"'=<>`]+` needs at least one character.
                if close > at {
                    value = Some(raw[value_start..chars[close].0].to_owned());
                    end = chars[close].0;
                    i = close;
                }
            }
        }
        out.push(Attr {
            name,
            value,
            span: (name_start, end),
            ws_start: chars[ws_start].0,
        });
    }
    out
}

/// Drop `data-bbox` / `data-label` from every tag in a block's inner HTML, as
/// `parse_full_page_html` does before handing a block to a client.
fn strip_nested_data_attrs(html: &str) -> String {
    if !html.contains("data-bbox") && !html.contains("data-label") {
        return html.to_owned();
    }
    let tags = scan_tags(html);
    let mut edits: Vec<(usize, usize, String)> = Vec::new();
    for tag in &tags {
        if tag.close || !tag.raw.contains("data-") {
            continue;
        }
        let attrs = parse_attrs(tag.attrs);
        let drop: Vec<&Attr> = attrs
            .iter()
            .filter(|attr| matches!(attr.name.as_str(), "data-bbox" | "data-label"))
            .collect();
        if drop.is_empty() {
            continue;
        }
        // Rebuild the run, keeping every other attribute byte-for-byte.
        let mut kept = String::with_capacity(tag.attrs.len());
        let mut cursor = 0usize;
        for attr in &drop {
            let from = cursor.max(attr.ws_start);
            kept.push_str(&tag.attrs[cursor..from]);
            cursor = attr.span.1;
        }
        kept.push_str(&tag.attrs[cursor..]);
        let attrs_start = tag.start + 1 + tag.name_raw.len();
        edits.push((attrs_start, attrs_start + tag.attrs.len(), kept));
    }
    if edits.is_empty() {
        return html.to_owned();
    }
    let mut out = String::with_capacity(html.len());
    let mut cursor = 0usize;
    for (start, end, replacement) in edits {
        if start < cursor {
            continue;
        }
        out.push_str(&html[cursor..start]);
        out.push_str(&replacement);
        cursor = end;
    }
    out.push_str(&html[cursor..]);
    out
}

// ---------------------------------------------------------------------------
// Incremental decode for streams
// ---------------------------------------------------------------------------

/// Decode compact blocks as they arrive, for SSE.
///
/// A block ends where the next header starts, so each completed block can be
/// emitted the moment its successor's header token appears — the API streams
/// decoded HTML at block granularity instead of buffering the whole page. The
/// scanner holds back at any `<` that could still grow into a header (a token
/// can be split across deltas), so a partial `<|Te` never leaks as content.
pub(crate) struct BlockStream {
    text: String,
    /// Next index a header scan resumes from.
    scan: usize,
    /// Start of the current block's content (the end of its header).
    content_start: usize,
    current: Option<(String, [u16; 4])>,
    /// A header has been seen, so the lead-in text has been handed out.
    started: bool,
    /// Last character handed to the client, so the concatenated deltas equal
    /// the non-streaming HTML (one newline between blocks).
    last_char: Option<char>,
}

impl BlockStream {
    pub(crate) fn new() -> Self {
        Self {
            text: String::new(),
            scan: 0,
            content_start: 0,
            current: None,
            started: false,
            last_char: None,
        }
    }

    /// Append a content delta; returns the decoded HTML of every block the
    /// delta completed (empty when nothing closed yet).
    pub(crate) fn push(&mut self, delta: &str) -> String {
        self.text.push_str(delta);
        let mut out = String::new();
        loop {
            let Some(offset) = self.text[self.scan..].find('<') else {
                self.scan = self.text.len();
                break;
            };
            let at = self.scan + offset;
            match probe_header(&self.text[at..]) {
                HeaderProbe::Found { len, label, bbox } => {
                    // Text before the first block is not OCR content; it is
                    // passed through the way [`decode_html`] passes it.
                    if !self.started {
                        let leading = strip_fence_prefix(&self.text[..at]).to_owned();
                        self.append_text(&mut out, &leading);
                        self.started = true;
                    }
                    if let Some((label, bbox)) = self.current.take() {
                        let decoded =
                            decode_block(&label, bbox, &self.text[self.content_start..at]);
                        self.append_decoded(&mut out, decoded);
                    }
                    self.content_start = at + len;
                    self.current = Some((label, bbox));
                    self.scan = self.content_start;
                }
                HeaderProbe::Incomplete => {
                    self.scan = at;
                    break;
                }
                HeaderProbe::NotHeader => self.scan = at + 1,
            }
        }
        out
    }

    /// Flush the block still open at end of generation.
    pub(crate) fn finish(&mut self) -> String {
        let mut out = String::new();
        match self.current.take() {
            Some((label, bbox)) => {
                // The answer's tail: trailing whitespace and a closing code
                // fence are the wrapper's, exactly as [`decode_html`] strips
                // them before decoding.
                let tail = strip_tail(&self.text[self.content_start..]).to_owned();
                let decoded = decode_block(&label, bbox, &tail);
                self.append_decoded(&mut out, decoded);
            }
            // No header ever arrived: a plain answer, returned as it is.
            None => {
                let text = std::mem::take(&mut self.text);
                self.append_text(&mut out, &text);
            }
        }
        out
    }

    /// Hand one decoded block (and any text the model wrote after its
    /// explicit `</div>`) to the client.
    fn append_decoded(&mut self, out: &mut String, decoded: Decoded) {
        for piece in decoded.pieces {
            match piece {
                Piece::Block(block) => self.append_block(out, block.div()),
                Piece::Text(text) => self.append_text(out, &text),
            }
        }
    }

    /// Hand `text` to the client, remembering where the stream ended.
    fn append_text(&mut self, out: &mut String, text: &str) {
        if text.is_empty() {
            return;
        }
        out.push_str(text);
        self.last_char = text.chars().last();
    }

    /// A block div, separated from the previous one exactly as the
    /// non-streaming decoder separates them.
    fn append_block(&mut self, out: &mut String, div: String) {
        if matches!(self.last_char, Some(last) if last != '\n') {
            out.push('\n');
        }
        self.append_text(out, &div);
    }
}

enum HeaderProbe {
    Found {
        len: usize,
        label: String,
        bbox: [u16; 4],
    },
    /// The tail could still become a header once more text arrives.
    Incomplete,
    NotHeader,
}

/// Classify a suffix as a block header, a partial header, or ordinary text.
fn probe_header(s: &str) -> HeaderProbe {
    if !s.starts_with('<') {
        return HeaderProbe::NotHeader;
    }
    let Some(rest) = s.strip_prefix("<|") else {
        // A lone `<` may still become the start of a header token.
        return if s.len() == 1 {
            HeaderProbe::Incomplete
        } else {
            HeaderProbe::NotHeader
        };
    };
    let Some(close) = rest.find("|>") else {
        // No closing `|>` yet: a prefix of a known label (optionally already
        // followed by the `|` of `|>`) is still in flight.
        let label = rest.strip_suffix('|').unwrap_or(rest);
        return if label.chars().all(|c| c.is_ascii_alphanumeric() || c == '-')
            && BLOCK_LABELS.iter().any(|known| known.starts_with(label))
        {
            HeaderProbe::Incomplete
        } else {
            HeaderProbe::NotHeader
        };
    };
    let label = &rest[..close];
    if !BLOCK_LABELS.contains(&label) {
        return HeaderProbe::NotHeader;
    }
    let mut pos = 2 + close + 2;
    let mut bbox = [0u16; 4];
    for slot in bbox.iter_mut() {
        let tail = &s[pos..];
        let Some(rest) = tail.strip_prefix("<|") else {
            return if tail.is_empty() || tail == "<" {
                HeaderProbe::Incomplete
            } else {
                HeaderProbe::NotHeader
            };
        };
        let Some(coord_close) = rest.find("|>") else {
            let digits = rest.strip_suffix('|').unwrap_or(rest);
            return if digits.len() <= 4 && digits.bytes().all(|b| b.is_ascii_digit()) {
                HeaderProbe::Incomplete
            } else {
                HeaderProbe::NotHeader
            };
        };
        let digits = &rest[..coord_close];
        if digits.is_empty() || digits.len() > 4 || !digits.bytes().all(|b| b.is_ascii_digit()) {
            return HeaderProbe::NotHeader;
        }
        *slot = match digits.parse() {
            Ok(value) => value,
            Err(_) => return HeaderProbe::NotHeader,
        };
        pos += 2 + coord_close + 2;
    }
    HeaderProbe::Found {
        len: pos,
        label: label.to_owned(),
        bbox,
    }
}

/// One block's pieces, decoded through the same path as a whole answer: the
/// block itself, plus anything the model wrote after it closed the block early
/// with an explicit `</div>`.
fn decode_block(label: &str, bbox: [u16; 4], content: &str) -> Decoded {
    let mut raw = String::with_capacity(content.len() + 32);
    raw.push_str("<|");
    raw.push_str(label);
    raw.push_str("|>");
    for value in bbox {
        raw.push_str(&format!("<|{value}|>"));
    }
    raw.push_str(content);
    Decoder::run(&raw)
}

// ---------------------------------------------------------------------------
// Merged cells (data-kind parent/ghost → colspan/rowspan)
// ---------------------------------------------------------------------------

/// Turn `data-kind="parent"` / `"ghost"` cells into real `colspan` / `rowspan`
/// and drop the ghost cells, as `merged_cells.convert_merged_cells` does.
///
/// Every table is converted, nested ones included, matching the reference's
/// `soup.find_all("table")`. A nested table is read from its own rows and
/// cells only: surya's pass walks `table.find_all("tr")`, which also counts a
/// nested table's rows as the outer table's (rows with no direct cells), so
/// the two disagree only where a nested table's rows would shift the outer
/// grid — which the model does not emit.
fn convert_merged_cells(html: &str) -> String {
    if !html.contains("data-kind") {
        return html.to_owned();
    }
    let tags = scan_tags(html);
    let mut edits: Vec<(usize, usize, Option<String>)> = Vec::new();
    let mut index = 0usize;
    while index < tags.len() {
        let tag = &tags[index];
        if tag.close || tag.selfclose || !tag.name_raw.eq_ignore_ascii_case("table") {
            index += 1;
            continue;
        }
        // A table the token cap cut off has no `</table>`; the answer's end
        // stands in for it (surya's BeautifulSoup pass auto-closes it, so the
        // reference still converts such a table's merged cells).
        let close = matching_close(&tags, index, "table");
        let body_end = close.unwrap_or(tags.len());
        let end_byte = close.map_or(html.len(), |at| tags[at].start);
        convert_table(&tags, index + 1, body_end, end_byte, &mut edits);
        // `index += 1`, not `close + 1`: a table nested in a cell is converted
        // from its own opening tag on the next iterations.
        index += 1;
    }
    if edits.is_empty() {
        return html.to_owned();
    }
    edits.sort_by_key(|(start, _, _)| *start);
    let mut out = String::with_capacity(html.len());
    let mut cursor = 0usize;
    for (start, end, replacement) in edits {
        if start < cursor {
            continue;
        }
        out.push_str(&html[cursor..start]);
        if let Some(text) = replacement {
            out.push_str(&text);
        }
        cursor = end;
    }
    out.push_str(&html[cursor..]);
    out
}

struct Cell {
    open: usize,
    /// Byte end of the cell's element (its `</td>`/`</th>`, or the end of the
    /// row when the token cap cut it off) — what a ghost cell removes.
    end_byte: usize,
    attrs: Vec<Attr>,
}

impl Cell {
    fn data_kind(&self) -> Option<&str> {
        self.attrs
            .iter()
            .find(|attr| attr.name == "data-kind")
            .and_then(|attr| attr.value.as_deref())
    }
}

fn convert_table(
    tags: &[TagRef<'_>],
    body_start: usize,
    body_end: usize,
    end_byte: usize,
    edits: &mut Vec<(usize, usize, Option<String>)>,
) {
    let mut rows: Vec<(usize, Vec<Cell>)> = Vec::new();
    let mut index = body_start;
    while index < body_end {
        let tag = &tags[index];
        if tag.close || !tag.name_raw.eq_ignore_ascii_case("tr") {
            index += 1;
            continue;
        }
        // A row the cap cut off has no `</tr>`: it ends where the table does.
        let row_close = matching_close(tags, index, "tr").unwrap_or(body_end);
        let row_end_byte = tags.get(row_close).map_or(end_byte, |tag| tag.start);
        let mut cells = Vec::new();
        let mut cursor = index + 1;
        while cursor < row_close {
            let cell = &tags[cursor];
            let name = if cell.name_raw.eq_ignore_ascii_case("td") {
                "td"
            } else if cell.name_raw.eq_ignore_ascii_case("th") {
                "th"
            } else {
                cursor += 1;
                continue;
            };
            if cell.close {
                cursor += 1;
                continue;
            }
            let (cell_close, cell_end_byte) = match matching_close(tags, cursor, name) {
                Some(at) => (at, tags[at].end),
                None => (row_close, row_end_byte),
            };
            cells.push(Cell {
                open: cursor,
                end_byte: cell_end_byte,
                attrs: parse_attrs(cell.attrs),
            });
            cursor = cell_close + 1;
        }
        rows.push((index, cells));
        index = row_close + 1;
    }
    if rows.is_empty() {
        return;
    }
    let column_count = rows.iter().map(|(_, cells)| cells.len()).max().unwrap_or(0);
    if column_count == 0 {
        return;
    }
    let row_count = rows.len();
    let mut processed = vec![vec![false; column_count]; row_count];
    for row in 0..row_count {
        for column in 0..column_count {
            let Some(cell) = rows[row].1.get(column) else {
                processed[row][column] = true;
                continue;
            };
            if processed[row][column] {
                continue;
            }
            if cell.data_kind() != Some("parent") {
                processed[row][column] = true;
                continue;
            }
            let mut colspan = 1usize;
            while column + colspan < column_count {
                let neighbour = &rows[row].1.get(column + colspan);
                match neighbour {
                    Some(next) if next.data_kind() == Some("ghost") => colspan += 1,
                    _ => break,
                }
            }
            let mut rowspan = 1usize;
            while row + rowspan < row_count {
                let mut all_ghosts = true;
                for offset in 0..colspan {
                    match rows[row + rowspan].1.get(column + offset) {
                        Some(next) if next.data_kind() == Some("ghost") => {}
                        _ => {
                            all_ghosts = false;
                            break;
                        }
                    }
                }
                if !all_ghosts {
                    break;
                }
                rowspan += 1;
            }
            for span in 0..rowspan {
                for offset in 0..colspan {
                    if row + span < row_count && column + offset < column_count {
                        processed[row + span][column + offset] = true;
                    }
                }
            }
            let open = &tags[cell.open];
            let rebuilt = rebuild_cell_tag(open.name_raw, &cell.attrs, colspan, rowspan);
            edits.push((open.start, open.end, Some(rebuilt)));
        }
    }
    for (_, cells) in &rows {
        for cell in cells {
            if cell.data_kind() == Some("ghost") {
                let open = &tags[cell.open];
                edits.push((open.start, cell.end_byte, None));
            }
        }
    }
}

/// A parent cell's opening tag: `data-cs` / `data-rs` / `data-kind` dropped,
/// `colspan` / `rowspan` appended when they are real.
fn rebuild_cell_tag(name: &str, attrs: &[Attr], colspan: usize, rowspan: usize) -> String {
    let mut parts: Vec<String> = Vec::with_capacity(attrs.len() + 2);
    for attr in attrs {
        if matches!(attr.name.as_str(), "data-cs" | "data-rs" | "data-kind") {
            continue;
        }
        // A span this pass derives replaces any the model wrote, so the tag
        // never carries two `colspan`s (first-one-wins in a browser).
        if (attr.name == "colspan" && colspan > 1) || (attr.name == "rowspan" && rowspan > 1) {
            continue;
        }
        match &attr.value {
            Some(value) => parts.push(format!(
                "{}=\"{}\"",
                attr.name,
                value.replace('"', "&quot;")
            )),
            None => parts.push(attr.name.clone()),
        }
    }
    if colspan > 1 {
        parts.push(format!("colspan=\"{colspan}\""));
    }
    if rowspan > 1 {
        parts.push(format!("rowspan=\"{rowspan}\""));
    }
    if parts.is_empty() {
        format!("<{name}>")
    } else {
        format!("<{name} {}>", parts.join(" "))
    }
}

/// Index of the `</name>` that closes the opening tag at `open`, counting
/// nested same-name elements.
fn matching_close(tags: &[TagRef<'_>], open: usize, name: &str) -> Option<usize> {
    let mut depth = 0usize;
    for (offset, tag) in tags.iter().enumerate().skip(open) {
        if !tag.name_raw.eq_ignore_ascii_case(name) || tag.selfclose {
            continue;
        }
        if tag.close {
            if depth == 0 {
                return Some(offset);
            }
            depth -= 1;
        } else if offset != open {
            depth += 1;
        }
    }
    None
}

fn scan_tags(html: &str) -> Vec<TagRef<'_>> {
    let mut tags = Vec::new();
    let mut pos = 0usize;
    while let Some(offset) = html[pos..].find('<') {
        let start = pos + offset;
        match tag_at(&html[start..]) {
            Some(mut tag) => {
                tag.start = start;
                tag.end = start + tag.end;
                pos = tag.end;
                tags.push(tag);
            }
            None => pos = start + 1,
        }
    }
    tags
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Expectations generated with `surya.inference.compact.decode` (and
    /// `merged_cells.convert_merged_cells` where a table is involved).
    #[test]
    fn decodes_blocks_and_bboxes() {
        let input = "<|Text|><|41|><|25|><|128|><|54|><p>hello</p>\
                     <|Text|><|0|><|0|><|10|><|10|><p>world</p>";
        assert_eq!(
            decode_html(input),
            "<div data-bbox=\"41 25 128 54\" data-label=\"Text\"><p>hello</p></div>\n\
             <div data-bbox=\"0 0 10 10\" data-label=\"Text\"><p>world</p></div>"
        );
    }

    #[test]
    fn restores_box_attribute_on_the_following_tag() {
        let input = "<|Text|><|0|><|0|><|10|><|10|>\
                     <div><|box|><|1|><|2|><|3|><|4|><img src=\"x\"></div>";
        assert_eq!(
            decode_html(input),
            "<div data-bbox=\"0 0 10 10\" data-label=\"Text\">\
             <div><img data-bbox=\"1 2 3 4\" src=\"x\"/></div></div>"
        );
    }

    #[test]
    fn restores_omitted_cell_and_row_closers() {
        let input = "<|Table|><|1|><|2|><|3|><|4|>\
                     <table><tr><th>H1<th>H2<tr><td>a<td>b</table>";
        assert_eq!(
            decode_html(input),
            "<div data-bbox=\"1 2 3 4\" data-label=\"Table\"><table><tr><th>H1</th><th>H2</th></tr>\
             <tr><td>a</td><td>b</td></tr></table></div>"
        );
    }

    #[test]
    fn converts_merged_cells_to_colspan_and_drops_ghosts() {
        let input = "<|Table|><|10|><|10|><|90|><|90|>\
                     <table><tr><td data-cs=\"2\" data-kind=\"parent\">A<td data-kind=\"ghost\">B\
                     <tr><td data-kind=\"ghost\">C<td>D</table>";
        assert_eq!(
            decode_html(input),
            "<div data-bbox=\"10 10 90 90\" data-label=\"Table\"><table><tr>\
             <td colspan=\"2\">A</td></tr><tr><td>D</td></tr></table></div>"
        );
    }

    #[test]
    fn canonicalizes_a_boxed_cell_tag() {
        let input = "<|Table|><|1|><|1|><|9|><|9|><table><tr>\
                     <|box|><|2|><|2|><|8|><|3|><td data-kind=\"parent\" data-cs=\"2\">A\
                     <td data-kind=\"ghost\">B</table>";
        // The merged-cell pass runs after the box pass, so the cell carries the
        // restored `data-bbox` plus the derived span. Attribute order is not
        // part of the contract (surya reads this with BeautifulSoup).
        let html = decode_html(input);
        assert_eq!(
            html,
            "<div data-bbox=\"1 1 9 9\" data-label=\"Table\"><table><tr>\
             <td data-bbox=\"2 2 8 3\" colspan=\"2\">A</td></tr></table></div>"
        );
    }

    #[test]
    fn keeps_unrecognized_token_shaped_text() {
        let input = "<|Foo|><|1|><|1|><|2|><|2|><p>lit</p>";
        assert_eq!(decode_html(input), input);
        assert!(decode_blocks(input).is_empty());
    }

    #[test]
    fn blocks_render_as_json() {
        let input = "<|Text|><|0|><|0|><|10|><|10|><p>x</p>";
        assert_eq!(
            blocks_json(input),
            "[{\"label\":\"Text\",\"bbox\":[0,0,10,10],\"html\":\"<p>x</p>\",\
              \"truncated\":false}]"
        );
    }

    /// `parse_full_page_html` removes a block's inner `data-bbox` /
    /// `data-label` before handing it on; the block's own div keeps its own.
    #[test]
    fn blocks_drop_nested_data_attrs_but_keep_the_block_box() {
        let input = "<|Table|><|1|><|1|><|9|><|9|><table><tr>\
                     <|box|><|2|><|2|><|8|><|3|><td data-kind=\"parent\" data-cs=\"2\">A\
                     <td data-kind=\"ghost\">B</table>";
        assert_eq!(
            blocks_json(input),
            "[{\"label\":\"Table\",\"bbox\":[1,1,9,9],\
              \"html\":\"<table><tr><td colspan=\\\"2\\\">A</td></tr></table>\",\
              \"truncated\":false}]"
        );
        // The `html` rendering keeps the restored box: it is the raw decode a
        // client's own parser reads.
        assert!(decode_html(input).contains("<td data-bbox=\"2 2 8 3\" colspan=\"2\">A</td>"));
    }

    /// `_strip_fences`: both parser entry points strip a ```` ```html ````
    /// wrapper before decoding.
    #[test]
    fn strips_a_code_fence_around_the_answer() {
        let fenced = "```html\n<|Text|><|1|><|1|><|2|><|2|><p>x</p>\n```";
        assert_eq!(
            decode_html(fenced),
            "<div data-bbox=\"1 1 2 2\" data-label=\"Text\"><p>x</p></div>"
        );
        assert!(blocks_json(fenced).contains("\"html\":\"<p>x</p>\""));
        // A fence line without a newline is not a wrapper.
        assert_eq!(
            decode_html("```nope<|Text|><|1|><|1|><|2|><|2|><p>x</p>"),
            "```nope\n<div data-bbox=\"1 1 2 2\" data-label=\"Text\"><p>x</p></div>"
        );
    }

    /// A `</div>` closes a block early; anything the model wrote after it stays
    /// after the block (it used to be hoisted in front of every block).
    #[test]
    fn keeps_text_after_an_explicit_block_close_in_order() {
        let input = "<|Text|><|0|><|0|><|5|><|5|><p>a</p></div>tail";
        assert_eq!(
            decode_html(input),
            "<div data-bbox=\"0 0 5 5\" data-label=\"Text\"><p>a</p></div>tail"
        );
        let two = "<|Text|><|0|><|0|><|5|><|5|><p>a</p></div>mid\
                   <|Text|><|6|><|6|><|9|><|9|><p>b</p>";
        assert_eq!(
            decode_html(two),
            "<div data-bbox=\"0 0 5 5\" data-label=\"Text\"><p>a</p></div>mid\n\
             <div data-bbox=\"6 6 9 9\" data-label=\"Text\"><p>b</p></div>"
        );
    }

    /// A table nested in a cell is converted too (`find_all("table")`).
    #[test]
    fn converts_merged_cells_in_a_nested_table() {
        let input = "<|Table|><|1|><|1|><|9|><|9|><table><tr><td>\
                     <table><tr><td data-kind=\"parent\" data-cs=\"2\">n\
                     <td data-kind=\"ghost\">n</table><td>x</table>";
        assert_eq!(
            decode_html(input),
            "<div data-bbox=\"1 1 9 9\" data-label=\"Table\"><table><tr><td>\
             <table><tr><td colspan=\"2\">n</td></tr></table></td><td>x</td></tr></table></div>"
        );
    }

    /// A table the token cap cut off still has its merged cells converted:
    /// surya's BeautifulSoup pass auto-closes it, so the reference converts a
    /// truncated table too (only the decoder's own `</table>` stays missing).
    #[test]
    fn converts_merged_cells_in_a_truncated_table() {
        let cut_row = "<|Table|><|1|><|1|><|9|><|9|><table><tr>\
                       <td data-kind=\"parent\" data-cs=\"2\">A<td data-kind=\"ghost\">B<tr><td>X";
        assert_eq!(
            decode_html(cut_row),
            "<div data-bbox=\"1 1 9 9\" data-label=\"Table\"><table><tr>\
             <td colspan=\"2\">A</td></tr><tr><td>X</td></tr></div>"
        );
        let cut_cell = "<|Table|><|1|><|1|><|9|><|9|><table><tr>\
                        <td data-kind=\"parent\" data-rs=\"2\">A<td data-kind=\"ghost\">B";
        assert_eq!(
            decode_html(cut_cell),
            "<div data-bbox=\"1 1 9 9\" data-label=\"Table\"><table><tr>\
             <td colspan=\"2\">A</td></tr></div>"
        );
    }

    /// `ParsedFullPageBlock.truncated`: the token cap cut the answer, so the
    /// last block is flagged (and only the last one).
    #[test]
    fn blocks_report_a_truncated_answer() {
        let unclosed = "<|Text|><|0|><|0|><|1|><|1|><p>open";
        assert!(blocks_json(unclosed).contains("\"truncated\":true"));
        let partial_token = "<|Text|><|0|><|0|><|1|><|1|><p>x</p><|Se";
        assert!(blocks_json(partial_token).contains("\"truncated\":true"));
        let two = "<|Text|><|0|><|0|><|1|><|1|><p>a</p>\
                   <|Text|><|2|><|2|><|3|><|3|><p>b</p><table><tr><td>c";
        let json = blocks_json(two);
        assert!(json.contains("\"truncated\":false"), "{json}");
        assert!(json.contains("\"truncated\":true"), "{json}");
        assert!(
            json.rfind("\"truncated\":true").unwrap() > json.find("\"truncated\":false").unwrap()
        );
        let complete = "<|Text|><|0|><|0|><|1|><|1|><p>a</p>";
        assert!(!blocks_json(complete).contains("\"truncated\":true"));
    }

    /// `_TAG_RE`/`_ATTR_RE` whitespace is Python `\s` (Unicode), so a no-break
    /// space between a tag name and its attributes neither turns the tag into
    /// text nor hides it from the merged-cell pass; the tag stays byte-faithful
    /// (raw, as `compact.decode` writes it), and a stale `colspan` a parent
    /// cell carries is replaced rather than doubled.
    ///
    /// HTML tokenization is the one place the reference itself disagrees:
    /// BeautifulSoup reads `<table\u{a0}class="x">` as a tag *named*
    /// `table\u{a0}class="x"` (a no-break space is not HTML whitespace), so its
    /// merged-cell pass leaves such a table alone. The decoder and this pass
    /// agree here instead, which is the coherent reading of `compact.decode`'s
    /// own `\s`.
    #[test]
    fn unicode_whitespace_and_stale_spans() {
        let nbsp = "<|Text|><|0|><|0|><|1|><|1|><p\u{a0}class=\"x\">t</p>";
        assert_eq!(
            decode_html(nbsp),
            "<div data-bbox=\"0 0 1 1\" data-label=\"Text\"><p\u{a0}class=\"x\">t</p></div>"
        );
        let spaced_table = "<|Table|><|1|><|1|><|9|><|9|><table\u{a0}class=\"x\"><tr>\
                            <td data-kind=\"parent\" data-cs=\"2\">A<td data-kind=\"ghost\">B</table>";
        assert_eq!(
            decode_html(spaced_table),
            "<div data-bbox=\"1 1 9 9\" data-label=\"Table\"><table\u{a0}class=\"x\"><tr>\
             <td colspan=\"2\">A</td></tr></table></div>"
        );
        let stale = "<|Table|><|1|><|1|><|9|><|9|><table><tr>\
                     <td data-kind=\"parent\" data-cs=\"2\" colspan=\"3\">A\
                     <td data-kind=\"ghost\">B</table>";
        assert_eq!(
            decode_html(stale),
            "<div data-bbox=\"1 1 9 9\" data-label=\"Table\"><table><tr>\
             <td colspan=\"2\">A</td></tr></table></div>"
        );
    }

    /// A literal `<` followed by a multi-byte character is text, not a slice
    /// boundary (`<भारत` used to panic and fail the whole request).
    #[test]
    fn literal_lt_before_a_multibyte_char_is_text() {
        let input = "<|Text|><|0|><|0|><|9|><|9|><p>1 <भारत</p> < 2";
        assert_eq!(
            decode_html(input),
            "<div data-bbox=\"0 0 9 9\" data-label=\"Text\"><p>1 <भारत</p> < 2</div>"
        );
        assert_eq!(decode_html("<भारत"), "<भारत");
    }

    #[test]
    fn nested_divs_inside_a_block_stay_balanced() {
        let input = "<|Text|><|0|><|0|><|5|><|5|><div>outer<div>inner</div>tail</div>";
        assert_eq!(
            decode_html(input),
            "<div data-bbox=\"0 0 5 5\" data-label=\"Text\">\
             <div>outer<div>inner</div>tail</div></div>"
        );
    }

    #[test]
    fn formats_parse_and_reject() {
        assert_eq!(OcrFormat::parse("html"), Some(OcrFormat::Html));
        assert_eq!(OcrFormat::parse("blocks"), Some(OcrFormat::Blocks));
        assert_eq!(OcrFormat::parse("HTML"), None);
        assert_eq!(OcrFormat::parse("json"), None);
    }

    #[test]
    fn stream_decodes_blocks_across_arbitrary_chunk_splits() {
        let raw = "<|Text|><|41|><|25|><|128|><|54|><p>hello</p>\
                   <|Table|><|1|><|2|><|3|><|4|>\
                   <table><tr><td data-kind=\"parent\" data-cs=\"2\">A<td data-kind=\"ghost\">B</table>";
        // One delta per byte is the worst case: headers, coordinates and tags
        // all arrive split.
        let mut stream = BlockStream::new();
        let mut decoded = String::new();
        for piece in raw.as_bytes().chunks(1) {
            decoded.push_str(&stream.push(std::str::from_utf8(piece).unwrap()));
        }
        decoded.push_str(&stream.finish());
        assert_eq!(decoded, decode_html(raw));
    }

    #[test]
    fn stream_emits_each_block_as_soon_as_the_next_header_arrives() {
        let mut stream = BlockStream::new();
        let raw = "<|Text|><|0|><|0|><|1|><|1|><p>one</p>\
                   <|Text|><|2|><|2|><|3|><|3|><p>two</p>";
        let first = stream.push("<|Text|><|0|><|0|><|1|><|1|><p>one</p>");
        assert_eq!(first, "");
        let second = stream.push("<|Text|><|2|><|2|><|3|><|3|>");
        assert_eq!(
            second,
            "<div data-bbox=\"0 0 1 1\" data-label=\"Text\"><p>one</p></div>"
        );
        let rest = stream.push("<p>two</p>");
        assert_eq!(rest, "");
        // The concatenation of every delta is the non-streaming HTML; the
        // separator newline therefore lands at the head of the next block.
        let last = stream.finish();
        assert_eq!(
            format!("{second}{last}"),
            decode_html(raw),
            "streamed deltas must concatenate to the static decode"
        );
    }

    /// The stream stays byte-identical to the static decode across a fenced
    /// answer, an explicit `</div>` with text after it, and multi-byte text.
    #[test]
    fn stream_matches_the_static_decode_for_wrapped_and_closed_answers() {
        for raw in [
            "```html\n<|Text|><|1|><|1|><|2|><|2|><p>a</p>\n```",
            "<|Text|><|1|><|1|><|2|><|2|><p>a</p></div>tail<|Text|><|3|><|3|><|4|><|4|><p>b</p>",
            "lead in <|Text|><|1|><|1|><|2|><|2|><p>भारत</p>",
            "<|Text|><|1|><|1|><|2|><|2|><p>a</p><|box|><|0|><|0|><|1|><|1|><img src=\"x\">",
        ] {
            for chunk in [1usize, 2, 3, 7] {
                let mut stream = BlockStream::new();
                let mut decoded = String::new();
                // Split on byte offsets, so a delta can end mid-character: the
                // pipe delivers whatever bytes the tokenizer emitted.
                let mut pending: Vec<u8> = Vec::new();
                for piece in raw.as_bytes().chunks(chunk) {
                    pending.extend_from_slice(piece);
                    let valid = match std::str::from_utf8(&pending) {
                        Ok(text) => text.len(),
                        Err(error) => error.valid_up_to(),
                    };
                    if valid == 0 {
                        continue;
                    }
                    let text = std::str::from_utf8(&pending[..valid]).expect("validated prefix");
                    decoded.push_str(&stream.push(text));
                    pending.drain(..valid);
                }
                assert!(pending.is_empty(), "raw text must be valid UTF-8");
                decoded.push_str(&stream.finish());
                assert_eq!(decoded, decode_html(raw), "chunk {chunk}: {raw:?}");
            }
        }
    }

    #[test]
    fn stream_passes_through_a_plain_answer() {
        let mut stream = BlockStream::new();
        assert_eq!(stream.push("no blocks"), "");
        assert_eq!(stream.push(" here"), "");
        assert_eq!(stream.finish(), "no blocks here");
    }

    #[test]
    fn stream_holds_back_partial_tokens() {
        let mut stream = BlockStream::new();
        assert_eq!(stream.push("<|Te"), "");
        assert_eq!(stream.push("xt|><|1|><|2|><|3|><|4|>"), "");
        assert_eq!(stream.push("<p>a"), "");
        assert_eq!(stream.push("</p>"), "");
        assert_eq!(
            stream.finish(),
            "<div data-bbox=\"1 2 3 4\" data-label=\"Text\"><p>a</p></div>"
        );
    }

    #[test]
    fn config_values_map_and_reject() {
        assert_eq!(OcrFormat::from_config("raw"), Some(None));
        assert_eq!(OcrFormat::from_config(""), Some(None));
        assert_eq!(OcrFormat::from_config("html"), Some(Some(OcrFormat::Html)));
        assert_eq!(
            OcrFormat::from_config("blocks"),
            Some(Some(OcrFormat::Blocks))
        );
        assert_eq!(OcrFormat::from_config("markdown"), None);
        assert_eq!(OcrFormat::Html.config_value(), "html");
        assert_eq!(OcrFormat::Blocks.config_value(), "blocks");
    }
}
