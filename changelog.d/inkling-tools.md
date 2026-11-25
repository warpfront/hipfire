### New model: Inkling-2-Mini (arch 24)

- **Added: Inkling-2-Mini structured tool calls.** `InklingAr` now accepts `tools`: they
  render through the model's own template (declarations, assistant tool calls, tool
  results), and the model's `<|content_invoke_tool_json|>` messages are parsed by token id,
  validated against the declared tools when the message closes, and released only on a
  `finish_reason=tool_calls` terminal. A malformed call, or a turn that ends inside one,
  fails the request closed; a length cut returns no calls. The chat-template `tojson`
  filter now honours `separators=` and `sort_keys=` as transformers does, which this
  template needs to render tool declarations and calls byte-for-byte.
