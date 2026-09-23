#!/usr/bin/env node

import fs from "node:fs";

const trace = process.argv[2];
if (!trace) {
  throw new Error("usage: analyze_decode_trace.mjs <trace_kernel_trace.csv>");
}

function parseCsvLine(line) {
  const fields = [];
  let field = "";
  let quoted = false;
  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];
    if (quoted) {
      if (char === '"' && line[i + 1] === '"') {
        field += '"';
        i += 1;
      } else if (char === '"') {
        quoted = false;
      } else {
        field += char;
      }
    } else if (char === '"') {
      quoted = true;
    } else if (char === ",") {
      fields.push(field);
      field = "";
    } else {
      field += char;
    }
  }
  fields.push(field);
  return fields;
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[middle - 1] + sorted[middle]) / 2
    : sorted[middle];
}

function fnv1a64(text) {
  let value = 0xcbf29ce484222325n;
  for (const byte of Buffer.from(text)) {
    value ^= BigInt(byte);
    value = BigInt.asUintN(64, value * 0x100000001b3n);
  }
  return value.toString(16).padStart(16, "0");
}

const lines = fs.readFileSync(trace, "utf8").trim().split(/\r?\n/);
const header = parseCsvLine(lines.shift());
const rows = lines.map((line) => {
  const fields = parseCsvLine(line);
  return Object.fromEntries(header.map((name, index) => [name, fields[index]]));
});
rows.sort((a, b) => Number(a.Start_Timestamp) - Number(b.Start_Timestamp));

const starts = [];
for (let index = 0; index < rows.length; index += 1) {
  if (rows[index].Kernel_Name === "embedding_q8") {
    starts.push(index);
  }
}
if (starts.length < 3) {
  throw new Error(`expected several embedding_q8 decode boundaries, found ${starts.length}`);
}

const segments = starts.map((start, index) => {
  const end = starts[index + 1] ?? rows.length;
  return rows.slice(start, end);
});

const tokenTotalsNs = segments.map((segment) =>
  segment.reduce(
    (total, row) => total + Number(row.End_Timestamp) - Number(row.Start_Timestamp),
    0,
  ),
);
const tokenMedianNs = median(tokenTotalsNs);
const names = [...new Set(segments.flatMap((segment) => segment.map((row) => row.Kernel_Name)))];
const ranking = [];

for (const name of names) {
  const callsByToken = [];
  const nsByToken = [];
  const invocationNs = [];
  for (const segment of segments) {
    const calls = segment.filter((row) => row.Kernel_Name === name);
    callsByToken.push(calls.length);
    const durations = calls.map(
      (row) => Number(row.End_Timestamp) - Number(row.Start_Timestamp),
    );
    invocationNs.push(...durations);
    nsByToken.push(durations.reduce((sum, value) => sum + value, 0));
  }
  const medianNsPerToken = median(nsByToken);
  ranking.push({
    kernel: name,
    calls_per_token: {
      min: Math.min(...callsByToken),
      median: median(callsByToken),
      max: Math.max(...callsByToken),
    },
    total_ms_per_token: {
      min: Math.min(...nsByToken) / 1e6,
      median: medianNsPerToken / 1e6,
      max: Math.max(...nsByToken) / 1e6,
    },
    percent_of_summed_kernel_time: (medianNsPerToken / tokenMedianNs) * 100,
    median_us_per_call: median(invocationNs) / 1e3,
  });
}
ranking.sort(
  (left, right) => right.total_ms_per_token.median - left.total_ms_per_token.median,
);

const sequences = segments.map((segment) => segment.map((row) => row.Kernel_Name));
const result = {
  trace,
  boundary: "embedding_q8",
  tokens: segments.length,
  kv_depth_range: [2048, 2048 + segments.length],
  launches_per_token: {
    raw: segments.map((segment) => segment.length),
    min: Math.min(...segments.map((segment) => segment.length)),
    median: median(segments.map((segment) => segment.length)),
    max: Math.max(...segments.map((segment) => segment.length)),
  },
  summed_kernel_ms_per_token: {
    raw: tokenTotalsNs.map((value) => value / 1e6),
    min: Math.min(...tokenTotalsNs) / 1e6,
    median: tokenMedianNs / 1e6,
    max: Math.max(...tokenTotalsNs) / 1e6,
  },
  ordered_kernel_name_hashes: sequences.map((sequence) => fnv1a64(sequence.join("\0"))),
  sequence_stable: new Set(sequences.map((sequence) => sequence.join("\0"))).size === 1,
  ranked_kernels: ranking,
};
process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
