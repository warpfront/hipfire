### New model: Inkling-2-Mini (arch 24)

- **Added: Inkling-2-Mini reasoning is separated from the answer.** The model writes its
  thinking and its answer as separate messages framed by protocol tokens
  (`<|message_model|><|content_thinking|>…<|end_message|>`, then `<|content_text|>`).
  `InklingAr` now splits them by token id into `reasoning_content` and `content` (the
  same wire shape as Qwen) instead of streaming the frames as literal text. The router
  (`MessageFrameRouter` in `hipfire-runtime`) and the new `message_frames` reasoning
  contract are shared, not Inkling-specific. `reasoning_effort` maps onto the template's
  own effort ladder (`low`…`max`, probed at load), and thinking off renders the
  template's `none` rung; the model may still write a short thinking message. Think-token
  caps are dropped with a warning. Sampled token ids are unchanged.
