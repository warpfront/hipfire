### Tests: conformance runner card rows

- **Added: `hipfire-conformance --card ROWS.json [REPORT.json]`, card rows through the real daemon.**
  Two row kinds drive the `daemon` binary over its JSON-lines protocol with
  `HIPFIRE_EMIT_TOKEN_IDS=1` and compare committed token ids bitwise: `c1_cn`
  (I5i: the same requests one at a time and as one burst at a fixed configured
  lane width; a burst that never overlaps is an error, not a pass) and
  `restore` (I2: turn 2 of a conversation on a cold daemon against the daemon
  that served turn 1, or after an evicting request). A route that streams text
  without committed ids is refused rather than compared. Row files for the G12
  M1 fixtures are in `crates/hipfire-conformance/rows/`, with planted controls
  (wrong seed, swapped lane input, stale restored history) that must fail. The
  receipt parser now accepts the `hip` and `embedded` module-load audit lines.
