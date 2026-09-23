#!/usr/bin/env node

import fs from "node:fs";

const path = process.argv[2];
if (!path) throw new Error("usage: analyze_e8_shapes.mjs <model.hfq>");

const fd = fs.openSync(path, "r");
const header = Buffer.alloc(32);
fs.readSync(fd, header, 0, header.length, 0);
if (header.subarray(0, 4).toString() !== "HFQM") {
  throw new Error("not an HFQM container");
}
const tensorCount = header.readUInt32LE(12);
const metadataOffset = Number(header.readBigUInt64LE(16));
const dataOffset = Number(header.readBigUInt64LE(24));
const indexRegion = Buffer.alloc(dataOffset - metadataOffset);
fs.readSync(fd, indexRegion, 0, indexRegion.length, metadataOffset);
fs.closeSync(fd);

let depth = 0;
let inString = false;
let escaped = false;
let jsonEnd = 0;
for (let index = 0; index < indexRegion.length; index += 1) {
  const byte = indexRegion[index];
  if (escaped) {
    escaped = false;
  } else if (byte === 0x5c && inString) {
    escaped = true;
  } else if (byte === 0x22) {
    inString = !inString;
  } else if (!inString && byte === 0x7b) {
    depth += 1;
  } else if (!inString && byte === 0x7d) {
    depth -= 1;
    if (depth === 0) {
      jsonEnd = index + 1;
      break;
    }
  }
}
if (jsonEnd === 0) throw new Error("metadata JSON is not brace-terminated");

let position = jsonEnd;
const indexedCount = indexRegion.readUInt32LE(position);
position += 4;
if (indexedCount !== tensorCount) {
  throw new Error(`header tensor count ${tensorCount} != index count ${indexedCount}`);
}

const tensors = [];
for (let index = 0; index < tensorCount; index += 1) {
  const nameLength = indexRegion.readUInt16LE(position);
  position += 2;
  const name = indexRegion.subarray(position, position + nameLength).toString();
  position += nameLength;
  const quantType = indexRegion[position];
  position += 1;
  const dimensionCount = indexRegion[position];
  position += 1;
  const shape = [];
  for (let dimension = 0; dimension < dimensionCount; dimension += 1) {
    shape.push(indexRegion.readUInt32LE(position));
    position += 4;
  }
  const groupSize = indexRegion.readUInt32LE(position);
  position += 4;
  const dataSize = Number(indexRegion.readBigUInt64LE(position));
  position += 8;
  tensors.push({ name, quantType, shape, groupSize, dataSize });
}

const e8 = tensors.filter((tensor) => tensor.quantType === 35);
const u4 = e8.filter((tensor) => !tensor.name.endsWith(".attn.wo_a.weight"));
const groups = new Map();
for (const tensor of u4) {
  const key = tensor.shape.join("x");
  const group = groups.get(key) ?? {
    shape: tensor.shape,
    tensor_count: 0,
    bytes_per_token: 0,
    examples: [],
  };
  group.tensor_count += 1;
  group.bytes_per_token += tensor.dataSize;
  if (group.examples.length < 4) group.examples.push(tensor.name);
  groups.set(key, group);
}

const result = {
  path,
  e8_tensor_count: e8.length,
  u4_tensor_count: u4.length,
  u4_bytes_per_token: u4.reduce((sum, tensor) => sum + tensor.dataSize, 0),
  shapes: [...groups.values()].sort(
    (lhs, rhs) => rhs.bytes_per_token - lhs.bytes_per_token,
  ),
};
process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
