// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Embedded chat frontend.
//!
//! Concern: static asset serving only. Compile-time-embedded files, one
//! route matcher, security headers. The UI itself is plain HTML/CSS/JS and
//! talks to the same `/v1/chat/completions` gateway every other client
//! uses — no second protocol, no extra dependencies. Chat history is kept
//! in the browser's IndexedDB, so serve stores nothing on its behalf.
//!
//! CSP is deliberately strict: model output is untrusted text rendered
//! into the page, so inline script/style are refused outright and the UI
//! only ever writes `textContent`.

use crate::serve::http::{boxed_full, BoxBody};
use hyper::{header, Response};

const INDEX_HTML: &str = include_str!("../../assets/ui/index.html");
const APP_JS: &str = include_str!("../../assets/ui/app.js");
const DB_JS: &str = include_str!("../../assets/ui/db.js");
const RENDER_JS: &str = include_str!("../../assets/ui/render.js");
const STYLE_CSS: &str = include_str!("../../assets/ui/style.css");

/// `script-src 'self'` + `style-src 'self'` is why the assets are separate
/// files: no `unsafe-inline`. `connect-src 'self'` permits the fetch/XHR
/// the app makes to /v1/* and /stats; `img-src data:` permits the image
/// preview data URI the vision attach produces.
///
/// `frame-ancestors 'none'` is what actually stops another origin from
/// embedding this page in an iframe (framing is not governed by CORS —
/// the API-wide `Access-Control-Allow-Origin: *` is irrelevant here, and
/// `default-src` does not fall back to `frame-ancestors`); without it an
/// attacker page could clickjack this unauthenticated control surface.
/// `X-Frame-Options: DENY` covers the legacy header check. `form-action`
/// and `base-uri` close navigation-based exfiltration paths.
const CSP: &str = "default-src 'none'; script-src 'self'; style-src 'self'; \
                   connect-src 'self'; img-src 'self' data:; base-uri 'none'; \
                   frame-ancestors 'none'; form-action 'none'";

fn asset_response(bytes: &'static str, content_type: &str) -> Response<BoxBody> {
    Response::builder()
        .status(200)
        .header(header::CONTENT_TYPE, content_type)
        .header(header::CONTENT_SECURITY_POLICY, CSP)
        .header("X-Frame-Options", "DENY")
        .header(header::X_CONTENT_TYPE_OPTIONS, "nosniff")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(boxed_full(bytes.as_bytes().to_vec()))
        .expect("static asset response builds")
}

/// Redirect `/` (and `/ui/` variants) at the UI index.
fn redirect_to_ui() -> Response<BoxBody> {
    Response::builder()
        .status(302)
        .header(header::LOCATION, "/ui")
        .header(header::CONTENT_SECURITY_POLICY, CSP)
        .header("X-Frame-Options", "DENY")
        .body(boxed_full(Vec::new()))
        .expect("static redirect builds")
}

/// Serve the embedded UI for `path` (already query-stripped). Returns
/// `None` for paths that are not UI routes so the caller falls through to
/// the 404 arm. Caller gates on `serve.ui` — this function does not
/// re-check the flag.
pub(crate) fn ui_asset(path: &str) -> Option<Response<BoxBody>> {
    match path {
        "/" => Some(redirect_to_ui()),
        "/ui" | "/ui/" | "/ui/index.html" => {
            Some(asset_response(INDEX_HTML, "text/html; charset=utf-8"))
        }
        "/ui/app.js" => Some(asset_response(APP_JS, "text/javascript; charset=utf-8")),
        "/ui/db.js" => Some(asset_response(DB_JS, "text/javascript; charset=utf-8")),
        "/ui/render.js" => Some(asset_response(RENDER_JS, "text/javascript; charset=utf-8")),
        "/ui/style.css" => Some(asset_response(STYLE_CSS, "text/css; charset=utf-8")),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ui_routes_cover_assets_and_redirect() {
        assert_eq!(ui_asset("/").unwrap().status(), 302);
        assert_eq!(
            ui_asset("/ui").unwrap().headers()[header::CONTENT_TYPE],
            "text/html; charset=utf-8"
        );
        for path in ["/ui/app.js", "/ui/db.js", "/ui/render.js", "/ui/style.css"] {
            let resp = ui_asset(path).expect(path);
            assert_eq!(resp.status(), 200, "{path}");
            assert!(resp.headers().contains_key(header::CONTENT_SECURITY_POLICY));
            assert_eq!(resp.headers()[header::X_CONTENT_TYPE_OPTIONS], "nosniff");
        }
        // Everything else stays out of the UI namespace — the router
        // falls through to 404.
        for path in ["/v1/models", "/ui/../etc", "/ui/foo.js", "/health"] {
            assert!(ui_asset(path).is_none(), "{path}");
        }
    }

    #[test]
    fn ui_headers_block_framing() {
        // `frame-ancestors` does not fall back to `default-src`, so it must
        // be spelled out; X-Frame-Options covers legacy checks. Assert the
        // value, not just presence — a later edit that drops either must
        // fail this test, not silently re-open clickjacking.
        for path in ["/", "/ui", "/ui/app.js", "/ui/style.css"] {
            let resp = ui_asset(path).unwrap();
            let csp = resp.headers()[header::CONTENT_SECURITY_POLICY]
                .to_str()
                .unwrap();
            assert!(csp.contains("frame-ancestors 'none'"), "{path}: {csp}");
            assert!(csp.contains("form-action 'none'"), "{path}: {csp}");
            assert!(csp.contains("default-src 'none'"), "{path}: {csp}");
            assert_eq!(resp.headers()["X-Frame-Options"], "DENY", "{path}");
        }
    }
}
