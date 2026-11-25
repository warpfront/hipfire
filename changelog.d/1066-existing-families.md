### Changed: existing families touched by the LFM3 port

- **Changed: an image request with `tools` is refused on Qwen3.5-VL, dots.ocr
  and LFM2-VL (#1066).** These vision routes do not render tool schemas, and
  they used to serve the request with the tools silently dropped; the daemon now
  refuses it as `unsupported`, naming the route. LFM3-VL is the only vision route
  that renders `tools`. Requests without `tools` are unchanged.
- **Changed: LFM2.5 (arch 11) accepts `stop` and `tools` (#1066).** Both are
  served on the `lfm_ar` route it shares with LFM3: `stop` through the runtime
  stop matcher (the stop text is never emitted), `tools` through the Liquid
  Pythonic tool-call parser with a `tool_calls` terminal. Before, the route
  refused both. Plain LFM2.5 output is unchanged.
- **Changed: the Qwen3.5-VL image preprocessor takes its pixel budget as
  parameters (#1066)** so LFM3-VL can pass its own; Qwen3.5-VL keeps its existing
  budget and its outputs are unchanged.
