// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Bearer-token secret handling shared by `hipfire serve` and its clients.
//!
//! The token lives only in the file named by `serve.api_key_file`; configuration,
//! logs, and error messages carry the file path and never its contents.

use crate::{ConfigError, Result};
use std::{fmt, fs, io::Read, path::Path};
use subtle::{Choice, ConstantTimeEq};

/// Configuration key whose file this module loads; used to label errors.
const KEY: &str = "serve.api_key_file";

/// Shortest accepted token, in bytes (after a key file's trailing whitespace
/// is trimmed).
pub const MIN_BEARER_TOKEN_BYTES: usize = 32;

/// A bearer token, loaded from a key file ([`BearerToken::load`]) or built from
/// an in-memory value ([`BearerToken::new`]).
///
/// `Debug` and `Display` are redacted so the secret cannot reach logs or error
/// chains through formatting. The only ways to read the value are
/// [`BearerToken::as_str`] (for attaching the `Authorization` header) and the
/// constant-time [`BearerToken::matches`]. The type deliberately has no
/// `PartialEq`, `Serialize`, or `Deref`.
#[derive(Clone)]
pub struct BearerToken(String);

impl BearerToken {
    /// Loads and validates the token stored at `path`.
    ///
    /// The file must be UTF-8. Trailing whitespace (the usual final newline) is
    /// trimmed; the remainder must be at least [`MIN_BEARER_TOKEN_BYTES`] bytes
    /// and contain only printable ASCII characters (no whitespace, control or
    /// non-ASCII characters), so it is always a valid single-line header value.
    ///
    /// On Unix the file must be a regular file that grants nothing to group or
    /// other (`mode & 0o077 == 0`); the mode is checked on the same opened
    /// handle that is then read, not on a second lookup of the path. (Its owner
    /// can still change the mode of an open file afterwards.) A looser mode is
    /// refused with a `chmod 600` instruction rather than tightened silently.
    ///
    /// Windows gap: no ACL check is made. Windows file permissions are ACLs, not
    /// a mode, so a token file readable by other local users is accepted there.
    /// Operators must restrict the file to the service account themselves
    /// (for example `icacls <file> /inheritance:r /grant:r "%USERNAME%:R"`).
    ///
    /// Errors name the path and the violated rule, never the file contents.
    pub fn load(path: &Path) -> Result<Self> {
        let mut file = fs::File::open(path).map_err(|source| ConfigError::Read {
            path: path.to_path_buf(),
            source,
        })?;
        let meta = file.metadata().map_err(|source| ConfigError::Read {
            path: path.to_path_buf(),
            source,
        })?;
        if !meta.is_file() {
            return Err(invalid(format!("{} is not a regular file", path.display())));
        }
        check_permissions(path, &meta)?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)
            .map_err(|source| ConfigError::Read {
                path: path.to_path_buf(),
                source,
            })?;
        Self::from_bytes(path, bytes)
    }

    /// Builds a token from an in-memory value, for callers that receive it
    /// other than from a key file (a remote session, a command-line prompt, a
    /// test). It applies the same rules as [`BearerToken::load`] — at least
    /// [`MIN_BEARER_TOKEN_BYTES`] bytes, printable ASCII only (no whitespace,
    /// control or non-ASCII characters) —
    /// but never trims: a value with surrounding whitespace is rejected, not
    /// silently altered. Errors never echo the value.
    pub fn new(value: impl Into<String>) -> Result<Self> {
        let value = value.into();
        match problem(&value) {
            Some(problem) => Err(ConfigError::InvalidValue {
                key: "bearer_token".to_owned(),
                message: format!("the token {problem}"),
            }),
            None => Ok(Self(value)),
        }
    }

    fn from_bytes(path: &Path, bytes: Vec<u8>) -> Result<Self> {
        let Ok(mut text) = String::from_utf8(bytes) else {
            return Err(invalid(format!(
                "{} is not valid UTF-8; a bearer token must be printable ASCII",
                path.display()
            )));
        };
        let len = text.trim_end().len();
        text.truncate(len);
        if let Some(problem) = problem(&text) {
            return Err(invalid(format!(
                "{} holds a token that {problem}",
                path.display()
            )));
        }
        Ok(Self(text))
    }

    /// The token, for attaching to an outgoing `Authorization: Bearer` header.
    /// Do not log or format the returned value.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Compares `candidate` with the token in constant time using
    /// [`subtle::ConstantTimeEq`]: every byte of the token is compared with no
    /// early exit, and the length check is accumulated rather than branched on.
    /// Work is proportional to the token length only, so timing reveals neither
    /// the position of a mismatch nor the candidate's length.
    pub fn matches(&self, candidate: &str) -> bool {
        let expected = self.0.as_bytes();
        let candidate = candidate.as_bytes();
        let mut equal = Choice::from(u8::from(expected.len() == candidate.len()));
        for (i, byte) in expected.iter().enumerate() {
            equal &= byte.ct_eq(&candidate.get(i).copied().unwrap_or(0));
        }
        bool::from(equal)
    }
}

impl fmt::Debug for BearerToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("BearerToken(<redacted>)")
    }
}

impl fmt::Display for BearerToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("<redacted>")
    }
}

fn invalid(message: String) -> ConfigError {
    ConfigError::InvalidValue {
        key: KEY.to_owned(),
        message,
    }
}

/// Why `token` is unacceptable, phrased to follow "the token" / "a token that";
/// `None` when it is a valid single-line printable-ASCII value of at least
/// [`MIN_BEARER_TOKEN_BYTES`] bytes (ASCII, so bytes equal characters). Never
/// includes the token: an offending character is named by kind or code point.
fn problem(token: &str) -> Option<String> {
    if token.is_empty() {
        return Some("is empty".to_owned());
    }
    if let Some((index, ch)) = token
        .chars()
        .enumerate()
        .find(|(_, c)| !c.is_ascii() || c.is_whitespace() || c.is_control())
    {
        return Some(format!(
            "contains {} at character {}; it must be a single line of printable ASCII \
             characters because it is sent as an HTTP header value",
            crate::http_auth::describe_char(ch),
            index + 1
        ));
    }
    (token.len() < MIN_BEARER_TOKEN_BYTES).then(|| {
        format!(
            "is {} bytes; at least {MIN_BEARER_TOKEN_BYTES} are required",
            token.len()
        )
    })
}

#[cfg(unix)]
fn check_permissions(path: &Path, meta: &fs::Metadata) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let mode = meta.permissions().mode() & 0o777;
    if mode & 0o077 != 0 {
        return Err(invalid(format!(
            "{} has mode {mode:04o}, which grants access to group or other users; \
             run `chmod 600 {}`",
            path.display(),
            path.display()
        )));
    }
    Ok(())
}

#[cfg(not(unix))]
fn check_permissions(_path: &Path, _meta: &fs::Metadata) -> Result<()> {
    // ACLs are not inspected; see the Windows gap note on `BearerToken::load`.
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        env,
        path::PathBuf,
        sync::atomic::{AtomicUsize, Ordering},
    };

    const TOKEN: &str = "k3y-0123456789abcdef0123456789ABCDEF";

    struct Dir(PathBuf);

    impl Dir {
        fn new() -> Self {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let dir = env::temp_dir().join(format!(
                "hipfire-secret-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }

        fn file(&self, bytes: &[u8]) -> PathBuf {
            let path = self.0.join("key");
            fs::write(&path, bytes).unwrap();
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
            }
            path
        }
    }

    impl Drop for Dir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn load_err(bytes: &[u8]) -> String {
        let dir = Dir::new();
        BearerToken::load(&dir.file(bytes)).unwrap_err().to_string()
    }

    #[test]
    fn loads_token_and_trims_trailing_whitespace() {
        let dir = Dir::new();
        for suffix in ["", "\n", "\r\n", " \t\n\n"] {
            let path = dir.file(format!("{TOKEN}{suffix}").as_bytes());
            let token = BearerToken::load(&path).unwrap();
            assert_eq!(token.as_str(), TOKEN, "suffix {suffix:?}");
        }
    }

    #[test]
    fn accepts_exactly_the_minimum_length() {
        let min = "a".repeat(MIN_BEARER_TOKEN_BYTES);
        let dir = Dir::new();
        assert_eq!(
            BearerToken::load(&dir.file(min.as_bytes()))
                .unwrap()
                .as_str(),
            min
        );
        // Printable ASCII punctuation is fine; only whitespace, control and
        // non-ASCII characters are refused.
        let punct = "!#$%&'()*+,-./:;<=>?@[]^_`{|}~0123456789".to_owned();
        assert_eq!(
            BearerToken::load(&dir.file(punct.as_bytes()))
                .unwrap()
                .as_str(),
            punct
        );
    }

    #[test]
    fn rejects_empty_and_whitespace_only_files() {
        assert!(load_err(b"").contains("is empty"));
        assert!(load_err(b" \n\t\r\n").contains("is empty"));
    }

    #[test]
    fn rejects_short_tokens_without_echoing_them() {
        let short = "s3cr3t-short-token-31-bytes-xxx";
        assert_eq!(short.len(), MIN_BEARER_TOKEN_BYTES - 1);
        let message = load_err(format!("{short}\n").as_bytes());
        assert!(message.contains("31 bytes"), "{message}");
        assert!(message.contains("at least 32"), "{message}");
        assert!(!message.contains(short), "{message}");
        // Trailing whitespace does not count toward the length.
        let padded = format!("{short}{}", " ".repeat(8));
        assert!(load_err(padded.as_bytes()).contains("31 bytes"));
    }

    #[test]
    fn rejects_invalid_utf8_without_echoing_bytes() {
        let mut bytes = TOKEN.as_bytes().to_vec();
        bytes.push(0xff);
        let message = load_err(&bytes);
        assert!(message.contains("not valid UTF-8"), "{message}");
        assert!(!message.contains(TOKEN), "{message}");
    }

    #[test]
    fn rejects_interior_whitespace_and_control_characters() {
        for (body, expect) in [
            (format!("{TOKEN}\n{TOKEN}"), "a newline at character 37"),
            (format!("{TOKEN} {TOKEN}"), "a space at character 37"),
            (format!("{TOKEN}\0{TOKEN}"), "a control character (U+0000)"),
            (format!("\n{TOKEN}"), "a newline at character 1"),
        ] {
            let message = load_err(body.as_bytes());
            assert!(message.contains(expect), "{message}");
            assert!(message.contains("serve.api_key_file"), "{message}");
            assert!(
                message.contains("single line of printable ASCII"),
                "{message}"
            );
            assert!(!message.contains(TOKEN), "{message}");
        }
    }

    #[test]
    fn rejects_non_ascii_naming_the_file_without_echoing_it() {
        let dir = Dir::new();
        for (body, expect) in [
            // 16 two-byte characters = 32 bytes: long enough, still refused.
            (
                "\u{e9}".repeat(16),
                "a non-ASCII character (U+00E9) at character 1",
            ),
            (
                format!("{TOKEN}\u{1F511}"),
                "a non-ASCII character (U+1F511) at character 37",
            ),
            (
                format!("{}\u{e9}{}", &TOKEN[..10], &TOKEN[10..]),
                "at character 11",
            ),
        ] {
            let path = dir.file(body.as_bytes());
            let message = BearerToken::load(&path).unwrap_err().to_string();
            assert!(message.contains(expect), "{message}");
            assert!(message.contains(&path.display().to_string()), "{message}");
            assert!(message.contains("serve.api_key_file"), "{message}");
            assert!(!message.contains(TOKEN), "{message}");
            assert!(!message.contains('\u{e9}'), "{message}");
        }
    }

    #[test]
    fn missing_file_and_directory_are_errors() {
        let dir = Dir::new();
        let missing = dir.0.join("absent");
        assert!(matches!(
            BearerToken::load(&missing),
            Err(ConfigError::Read { .. })
        ));
        assert!(BearerToken::load(&dir.0).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn refuses_group_or_other_access_with_chmod_hint() {
        use std::os::unix::fs::PermissionsExt;
        let dir = Dir::new();
        let path = dir.file(TOKEN.as_bytes());
        for mode in [0o640, 0o604, 0o660, 0o644, 0o666, 0o601, 0o710] {
            fs::set_permissions(&path, fs::Permissions::from_mode(mode)).unwrap();
            let message = BearerToken::load(&path).unwrap_err().to_string();
            assert!(message.contains(&format!("{mode:04o}")), "{message}");
            assert!(
                message.contains(&format!("chmod 600 {}", path.display())),
                "{message}"
            );
            assert!(message.contains("serve.api_key_file"), "{message}");
            assert!(!message.contains(TOKEN), "{message}");
        }
        for mode in [0o600, 0o400] {
            fs::set_permissions(&path, fs::Permissions::from_mode(mode)).unwrap();
            assert_eq!(BearerToken::load(&path).unwrap().as_str(), TOKEN);
        }
    }

    #[test]
    fn new_applies_file_rules_without_trimming_or_echoing() {
        let token = BearerToken::new(TOKEN).unwrap();
        assert_eq!(token.as_str(), TOKEN);
        assert!(token.matches(TOKEN));
        assert_eq!(BearerToken::new(TOKEN.to_owned()).unwrap().as_str(), TOKEN);
        let short = &TOKEN[..MIN_BEARER_TOKEN_BYTES - 1];
        for (value, expect) in [
            (String::new(), "is empty"),
            (short.to_owned(), "31 bytes"),
            (format!("{TOKEN}\n"), "a newline at character 37"),
            (format!(" {TOKEN}"), "a space at character 1"),
            (format!("{TOKEN} {TOKEN}"), "a space at character 37"),
            (
                format!("{TOKEN}\0"),
                "a control character (U+0000) at character 37",
            ),
            (
                format!("{TOKEN}\u{e9}"),
                "a non-ASCII character (U+00E9) at character 37",
            ),
            (
                "\u{e9}".repeat(16),
                "a non-ASCII character (U+00E9) at character 1",
            ),
        ] {
            let message = BearerToken::new(value).unwrap_err().to_string();
            assert!(message.contains(expect), "{message}");
            assert!(!message.contains(TOKEN), "{message}");
            assert!(!message.contains(short), "{message}");
        }
        // The in-memory rule is the file rule: non-ASCII is refused whatever the length.
        assert!(BearerToken::new("\u{e9}".repeat(32)).is_err());
    }

    #[test]
    fn matches_is_exact_for_equal_prefix_suffix_and_length() {
        let dir = Dir::new();
        let token = BearerToken::load(&dir.file(TOKEN.as_bytes())).unwrap();
        assert!(token.matches(TOKEN));
        assert!(!token.matches(""));
        assert!(!token.matches(&TOKEN[..TOKEN.len() - 1]));
        assert!(!token.matches(&format!("{TOKEN}x")));
        assert!(!token.matches(&format!("{TOKEN}\0")));
        let mut flipped = TOKEN.to_owned().into_bytes();
        for i in 0..flipped.len() {
            flipped[i] ^= 1;
            assert!(
                !token.matches(std::str::from_utf8(&flipped).unwrap()),
                "{i}"
            );
            flipped[i] ^= 1;
        }
    }

    #[test]
    fn formatting_never_reveals_the_token() {
        let dir = Dir::new();
        let token = BearerToken::load(&dir.file(TOKEN.as_bytes())).unwrap();
        #[derive(Debug)]
        struct Holder {
            _token: BearerToken,
        }
        let holder = Holder {
            _token: token.clone(),
        };
        for rendered in [
            format!("{token}"),
            format!("{token:?}"),
            format!("{token:#?}"),
            format!("{holder:?}"),
            format!("{holder:#?}"),
        ] {
            assert!(rendered.contains("<redacted>"), "{rendered}");
            assert!(!rendered.contains(TOKEN), "{rendered}");
            assert!(!rendered.contains(&TOKEN[..8]), "{rendered}");
        }
    }
}
