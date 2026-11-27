### Fixes: G12 engine loop failure handling

- **G12 engine loop (`generation.engine_loop = engine`): failure handling (HIPF-469, Loki r3).** A load with an attached speculator (DFlash2 or native MTP) or the exact expert cache now declines the engine and keeps its legacy loop. A failed device synchronization on the engine's one-owner mesh returns the daemon's device instead of aborting, and refuses every later engine request on that load. A submission that fails part-way, or a quarantine, keeps the request's output buffers allocated rather than freeing them under possibly in-flight work.
