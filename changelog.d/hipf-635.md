### Fixes: continuous-batch occupancy on the Flash-Next lane driver

- **`continuous_batch.max_active_lanes` on the legacy Qwen4 lane driver (`drive_qwen4_lane_batch`) counts the Running lanes at each step (HIPF-635).** Before, the driver never recorded occupancy, so every terminal reported 1, also for overlapping requests. The value is the peak number of Running lanes while the request was in a lane. It is not the number of rows in one forward.
