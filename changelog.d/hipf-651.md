### Fixes: hipfire bench

- **`hipfire bench --concurrency` fails the sweep when the daemon stream closes without `done` (HIPF-651).** Before, `closed_loop` recorded the closed stream as one failed request and then submitted the next queued request. A process in front of the daemon (for example the `engine_loop_ab.py` wire tap) can still accept that write after the daemon stdout is gone. No reader stays to answer it, so the sweep waited for the 300 s stall deadline. Now the sweep cancels the requests in flight and fails at once.
