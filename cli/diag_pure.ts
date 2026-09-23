// Pure helpers for `hipfire diag`. Kept side-effect-free so host-specific
// probes can be tested without loading cli/index.ts.

export type NpuReadiness =
  | "Ready"
  | "MissingDeviceNode"
  | "DeviceNodeNotChar"
  | "XrtSmiMissing"
  | "MemlockTooLow"
  | "XrtSmiFailed";

export interface NpuDiagInput {
  accelExists: boolean;
  accelIsChar: boolean;
  accelMode: string;
  accelOwner: string;
  memlockSoftBytes: number | null;
  minMemlockBytes: number;
  xrtSmiFound: boolean;
  xrtSmiStatus: number | null;
  xrtSmiOutput: string;
}

export interface NpuDiagResult {
  readiness: NpuReadiness;
  lines: string[];
  advice: string[];
}

export function classifyNpuDiag(input: NpuDiagInput): NpuDiagResult {
  const advice: string[] = [];
  let readiness: NpuReadiness;

  if (!input.accelExists) {
    readiness = "MissingDeviceNode";
    advice.push("expected XDNA device node at /dev/accel/accel0");
  } else if (!input.accelIsChar) {
    readiness = "DeviceNodeNotChar";
    advice.push("/dev/accel/accel0 exists but is not a character device");
  } else if (!input.xrtSmiFound) {
    readiness = "XrtSmiMissing";
    advice.push("install XRT utilities so xrt-smi is available");
  } else if (
    input.xrtSmiStatus !== 0 &&
    input.xrtSmiOutput.includes("Resource temporarily unavailable") &&
    !memlockAtLeast(input.memlockSoftBytes, input.minMemlockBytes)
  ) {
    readiness = "MemlockTooLow";
    advice.push(`raise RLIMIT_MEMLOCK to at least ${input.minMemlockBytes} bytes for the hipfire daemon/user`);
  } else if (input.xrtSmiStatus !== 0) {
    readiness = "XrtSmiFailed";
    advice.push("xrt-smi examine failed; inspect output for XDNA driver/runtime state");
  } else {
    readiness = "Ready";
  }

  const name = readiness === "Ready" ? extractXrtDeviceName(input.xrtSmiOutput) : "";
  const lines = [
    `XDNA NPU:     ${readiness}${name ? ` (${name})` : ""}`,
    `/dev/accel:   ${input.accelExists ? (input.accelIsChar ? `char ${input.accelMode} ${input.accelOwner}` : "not a char device") : "NOT FOUND"}`,
    `memlock:      ${formatMemlock(input.memlockSoftBytes)}`,
    `xrt-smi:      ${input.xrtSmiFound ? `status ${input.xrtSmiStatus ?? "unknown"}` : "NOT FOUND"}`,
  ];

  return { readiness, lines, advice };
}

function memlockAtLeast(value: number | null, minBytes: number): boolean {
  return value === null || value >= minBytes;
}

function formatMemlock(value: number | null): string {
  return value === null ? "unlimited" : `${value} bytes`;
}

function extractXrtDeviceName(output: string): string {
  const match = output.match(/\|\[[^\]]+\]\s*\|([^|\n]+)\|/);
  return match ? match[1].trim() : "";
}
