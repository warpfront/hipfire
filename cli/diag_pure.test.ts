import { describe, expect, test } from "bun:test";
import { classifyNpuDiag } from "./diag_pure.ts";

describe("classifyNpuDiag", () => {
  test("reports ready when the accel node and xrt-smi are usable", () => {
    const result = classifyNpuDiag({
      accelExists: true,
      accelIsChar: true,
      accelMode: "0660",
      accelOwner: "root:render",
      memlockSoftBytes: 134217728,
      minMemlockBytes: 134217728,
      xrtSmiFound: true,
      xrtSmiStatus: 0,
      xrtSmiOutput: "Device(s) Present\n|[0000:c0:00.1]  |RyzenAI-npu5  |",
    });

    expect(result.readiness).toBe("Ready");
    expect(result.lines).toContain("XDNA NPU:     Ready (RyzenAI-npu5)");
    expect(result.advice).toEqual([]);
  });

  test("reports low memlock when xrt-smi hits mmap EAGAIN", () => {
    const result = classifyNpuDiag({
      accelExists: true,
      accelIsChar: true,
      accelMode: "0660",
      accelOwner: "root:render",
      memlockSoftBytes: 8388608,
      minMemlockBytes: 134217728,
      xrtSmiFound: true,
      xrtSmiStatus: 1,
      xrtSmiOutput: "mmap failed (err=-11): Resource temporarily unavailable",
    });

    expect(result.readiness).toBe("MemlockTooLow");
    expect(result.lines).toContain("XDNA NPU:     MemlockTooLow");
    expect(result.advice.some((line) => line.includes("RLIMIT_MEMLOCK"))).toBe(true);
  });
});
