#!/usr/bin/env python3
import csv
import signal
import sys
import time

sys.path.insert(0, "/opt/rocm/core-10.0/share/amd_smi")
import amdsmi

TARGET_UUID = "9e007551-0000-1000-80b7-aeda51c88ffd"
running = True

def stop(_signum, _frame):
    global running
    running = False

signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGTERM, stop)
amdsmi.amdsmi_init()
try:
    devices = [
        device
        for device in amdsmi.amdsmi_get_processor_handles()
        if amdsmi.amdsmi_get_gpu_device_uuid(device) == TARGET_UUID
    ]
    if len(devices) != 1:
        raise RuntimeError(f"Card-A UUID matches: {len(devices)}")
    device = devices[0]
    with open(sys.argv[1], "w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow([
            "unix_ns", "elapsed_ms", "current_gfxclk_mhz", "average_gfxclk_mhz",
            "current_uclk_mhz", "average_uclk_mhz", "average_socket_power_w",
        ])
        start = time.monotonic_ns()
        deadline = start
        while running:
            metrics = amdsmi.amdsmi_get_gpu_metrics_info(device)
            now = time.monotonic_ns()
            writer.writerow([
                time.time_ns(), (now - start) / 1e6,
                metrics.get("current_gfxclk", "N/A"),
                metrics.get("average_gfxclk_frequency", "N/A"),
                metrics.get("current_uclk", "N/A"),
                metrics.get("average_uclk_frequency", "N/A"),
                metrics.get("average_socket_power", "N/A"),
            ])
            output.flush()
            deadline += 100_000_000
            time.sleep(max(0.0, (deadline - time.monotonic_ns()) / 1e9))
finally:
    amdsmi.amdsmi_shut_down()
