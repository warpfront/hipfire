//! `.amdgpu_metadata` YAML → the `NT_AMDGPU_METADATA` MessagePack blob.
//!
//! The assembler reads the document into an `llvm::msgpack::Document`
//! (`DocNode::fromString`: an unsigned integer, else a signed one, else a
//! boolean, else a string) and writes it with `Document::writeToBlob`: maps
//! sorted by key bytes, every integer and length in its smallest form,
//! `str8` allowed. Only the YAML the builder writes is read: block maps,
//! block sequences (an item may open on the `- ` line or the next one), flow
//! sequences of scalars and plain scalars. Anything else — and any scalar the
//! assembler would read as a float, null or a non-`true`/`false` boolean —
//! is an error, never a guess.

enum Node { Map(Vec<(String, Node)>), Seq(Vec<Node>), Scalar(String), Empty }

struct Line<'a> { indent: usize, text: &'a str }

fn parse_block(lines: &mut Vec<Line<'_>>, at: &mut usize, indent: usize) -> Result<Node, String> {
    let first = lines.get(*at).ok_or("metadata: expected a block")?;
    if first.indent != indent { return Err(format!("metadata: bad indentation at {:?}", first.text)) }
    if first.text == "-" || first.text.starts_with("- ") {
        let mut items = Vec::new();
        while let Some(line) = lines.get(*at) {
            if line.indent < indent { break }
            if line.indent > indent || !(line.text == "-" || line.text.starts_with("- ")) {
                return Err(format!("metadata: bad sequence item {:?}", line.text));
            }
            let rest = line.text[1..].trim_start();
            if rest.is_empty() {
                *at += 1;
                let child = lines.get(*at).map(|l| l.indent).filter(|&i| i > indent).ok_or("metadata: empty sequence item")?;
                items.push(parse_block(lines, at, child)?);
            } else {
                // The item's first key sits on the `- ` line: its column is the item's indentation.
                let column = indent + (line.text.len() - rest.len());
                lines[*at] = Line { indent: column, text: rest };
                items.push(parse_block(lines, at, column)?);
            }
        }
        return Ok(Node::Seq(items));
    }
    let mut entries: Vec<(String, Node)> = Vec::new();
    while let Some(line) = lines.get(*at) {
        if line.indent < indent { break }
        if line.indent > indent { return Err(format!("metadata: bad indentation at {:?}", line.text)) }
        let (key, value) = match line.text.split_once(": ") {
            Some((k, v)) => (k, v.trim()),
            None => (line.text.strip_suffix(':').ok_or_else(|| format!("metadata: expected `key: value`, got {:?}", line.text))?, ""),
        };
        if entries.iter().any(|(k, _)| k == key) { return Err(format!("metadata: duplicate key {key}")) }
        *at += 1;
        let node = if value.is_empty() {
            // `key:` with nothing nested is an empty value (a kernel without arguments: `.args:`).
            match lines.get(*at).map(|l| l.indent).filter(|&i| i > indent) {
                Some(child) => parse_block(lines, at, child)?,
                None => Node::Empty,
            }
        } else if let Some(flow) = value.strip_prefix('[') {
            let flow = flow.strip_suffix(']').ok_or_else(|| format!("metadata: unterminated flow sequence {value}"))?;
            Node::Seq(flow.split(',').map(|s| Node::Scalar(s.trim().to_owned())).collect())
        } else {
            Node::Scalar(value.to_owned())
        };
        entries.push((key.to_owned(), node));
    }
    Ok(Node::Map(entries))
}

/// The YAML between `---` and `...` (exclusive) of an `.amdgpu_metadata` block.
fn parse(yaml: &str) -> Result<Node, String> {
    let mut lines: Vec<Line<'_>> = yaml.lines().filter(|l| !l.trim().is_empty()).map(|l| {
        let text = l.trim_start_matches(' ');
        Line { indent: l.len() - text.len(), text: text.trim_end() }
    }).collect();
    if lines.iter().any(|l| l.text.starts_with('#') || l.text.contains('\t') || l.text.contains(['"', '\'', '{', '&', '*', '!', '|', '>'])) {
        return Err("metadata: unsupported YAML syntax".into());
    }
    let mut at = 0;
    let node = parse_block(&mut lines, &mut at, 0)?;
    if at != lines.len() { return Err(format!("metadata: trailing YAML at {:?}", lines[at].text)) }
    Ok(node)
}

fn uint(out: &mut Vec<u8>, v: u64) {
    match v {
        0..=0x7f => out.push(v as u8),
        0x80..=0xff => out.extend([0xcc, v as u8]),
        0x100..=0xffff => { out.push(0xcd); out.extend((v as u16).to_be_bytes()) }
        0x1_0000..=0xffff_ffff => { out.push(0xce); out.extend((v as u32).to_be_bytes()) }
        _ => { out.push(0xcf); out.extend(v.to_be_bytes()) }
    }
}
fn int(out: &mut Vec<u8>, v: i64) {
    if v >= 0 { return uint(out, v as u64) }
    if v >= -32 { out.push(v as i8 as u8) }
    else if v >= i64::from(i8::MIN) { out.extend([0xd0, v as i8 as u8]) }
    else if v >= i64::from(i16::MIN) { out.push(0xd1); out.extend((v as i16).to_be_bytes()) }
    else if v >= i64::from(i32::MIN) { out.push(0xd2); out.extend((v as i32).to_be_bytes()) }
    else { out.push(0xd3); out.extend(v.to_be_bytes()) }
}
fn string(out: &mut Vec<u8>, s: &str) {
    let n = s.len();
    match n {
        0..=31 => out.push(0xa0 | n as u8),
        32..=0xff => out.extend([0xd9, n as u8]),
        0x100..=0xffff => { out.push(0xda); out.extend((n as u16).to_be_bytes()) }
        _ => { out.push(0xdb); out.extend((n as u32).to_be_bytes()) }
    }
    out.extend(s.as_bytes());
}
fn length(out: &mut Vec<u8>, n: usize, fix: u8, wide16: u8) {
    if n < 16 { out.push(fix | n as u8) } else if n <= 0xffff { out.push(wide16); out.extend((n as u16).to_be_bytes()) }
    else { out.push(wide16 + 1); out.extend((n as u32).to_be_bytes()) }
}

/// One plain scalar the way `DocNode::fromString` types it.
fn scalar(out: &mut Vec<u8>, s: &str) -> Result<(), String> {
    let digits = |t: &str| !t.is_empty() && t.bytes().all(|b| b.is_ascii_digit());
    if digits(s) && (s == "0" || !s.starts_with('0')) {
        return s.parse::<u64>().map(|v| uint(out, v)).map_err(|e| format!("metadata: {s}: {e}"));
    }
    if s.strip_prefix('-').is_some_and(|m| digits(m) && (m == "0" || !m.starts_with('0'))) {
        return s.parse::<i64>().map(|v| int(out, v)).map_err(|e| format!("metadata: {s}: {e}"));
    }
    match s {
        "true" => { out.push(0xc3); return Ok(()) }
        "false" => { out.push(0xc2); return Ok(()) }
        _ => {}
    }
    // Plain identifiers only: nothing YAML or the msgpack reader could type
    // otherwise (numbers in other radices, floats, null, yes/no/on/off...).
    // The pinned llvm-mc keeps one-letter `Y`/`N` (argument names) strings.
    let reserved = ["~", "null", "Null", "NULL", "yes", "Yes", "YES", "no", "No", "NO",
        "on", "On", "ON", "off", "Off", "OFF", "True", "TRUE", "False", "FALSE", ".inf", ".Inf", ".INF", ".nan", ".NaN", ".NAN"];
    let first = s.bytes().next().ok_or("metadata: empty scalar")?;
    if reserved.contains(&s) || first.is_ascii_digit() || first == b'-' || first == b'+'
        || !s.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'-')) {
        return Err(format!("metadata: unsupported scalar {s:?}"));
    }
    string(out, s);
    Ok(())
}

fn write(out: &mut Vec<u8>, node: &Node) -> Result<(), String> {
    match node {
        Node::Scalar(s) => scalar(out, s)?,
        Node::Empty => return Err("metadata: empty value".into()),
        Node::Seq(items) => {
            length(out, items.len(), 0x90, 0xdc);
            for item in items { write(out, item)? }
        }
        Node::Map(entries) => {
            let mut sorted: Vec<&(String, Node)> = entries.iter().collect();
            sorted.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
            length(out, sorted.len(), 0x80, 0xde);
            for (key, value) in sorted {
                // Keys are typed like any scalar; the builder's are all strings.
                let mut k = Vec::new();
                scalar(&mut k, key)?;
                if !matches!(k.first(), Some(0xa0..=0xbf | 0xd9..=0xdb)) { return Err(format!("metadata: non-string key {key}")) }
                out.extend(k);
                // An argument-less kernel's empty `.args:` is the empty array.
                if matches!(value, Node::Empty) && key == ".args" { out.push(0x90) } else { write(out, value)? }
            }
        }
    }
    Ok(())
}

/// The MessagePack blob of an `.amdgpu_metadata` YAML document.
pub(crate) fn msgpack(yaml: &str) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    write(&mut out, &parse(yaml)?)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::msgpack;
    #[test]
    fn items_open_inline_or_on_the_next_line_and_maps_sort_by_key() {
        let yaml = "b:\n  - .x: 1\n    .a: global\n  - \n    .k: false\na: [1, 2]\n";
        assert_eq!(msgpack(yaml).unwrap(), [0x82, 0xa1, b'a', 0x92, 1, 2, 0xa1, b'b', 0x92,
            0x82, 0xa2, b'.', b'a', 0xa6, b'g', b'l', b'o', b'b', b'a', b'l', 0xa2, b'.', b'x', 1,
            0x81, 0xa2, b'.', b'k', 0xc2]);
    }
    #[test]
    fn integers_take_their_smallest_form_and_ambiguous_scalars_are_refused() {
        assert_eq!(msgpack("a: 200\nb: 70000\n").unwrap(), [0x82, 0xa1, b'a', 0xcc, 200, 0xa1, b'b', 0xce, 0, 1, 0x11, 0x70]);
        for bad in ["a: 1.5\n", "a: yes\n", "a: 0x10\n", "a: \"s\"\n", "a: 012\n"] { assert!(msgpack(bad).is_err(), "{bad}") }
    }
}
