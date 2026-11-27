### Models: Inkling-2-Mini in the registry

- **New: `inkling2:mini` pulls Inkling-2-Mini (arch 24).** The default tier is
  `inkling2:mini-mq4-xts` (`hipfire-models/inkling2-mini`, file
  `inkling2-mini.mq4-xts`, 14.44 GB), a symmetric MQ4V2 XT quant (qt44; H2
  per-expert Hessian GPTQ plus AWQ, F16 routers). It is also reachable as
  `inkling2`. Against Raufaser's uniform MQ4 RTN, paired KLD to the source
  NVFP4/MXFP8 teacher over 24×2048 chunks on gfx1201 is 0.0202 vs 0.0328 on
  WikiText-2 (−38%) and 0.0144 vs 0.0229 on code24 (−37%), lower on every
  chunk. On an R9700 it decodes 6.6% faster at 128 tokens of context and 2.8%
  faster at 2K, and prefills 3.1% faster. The RTN stays pullable as
  `inkling2:mini-mq4` from `Raufaser/Inkling-2-Mini-hipfire-MQ4`. Text-only,
  single-stream, speculation off.
