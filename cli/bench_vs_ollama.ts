// Side-by-side bench of hipfire (asym3 MQ4) vs ollama (default Q4_K_M)
// with MATCHED prompt length. Reports prefill tok/s at pp128 and pp512
// to compare apples-to-apples — ollama prefill tok/s on short (22-tok)
// prompts is dominated by per-token launch overhead and massively
// understates its steady-state throughput.
//
// Builds the long prompt by repeating a short seed sentence (tokens
// roughly scale ~1.3 tokens/word for Qwen tokenizers, so 100 and 400
// words give ~128 and ~512 tokens respectively — we send and read the
// actual `prompt_eval_count` back from ollama for exact accounting).
//
// Usage:
//   bun cli/bench_vs_ollama.ts                   # all common models
//   bun cli/bench_vs_ollama.ts qwen3.5:9b        # single model
import { spawn } from "bun";

type OllamaBench = {
  prefill_tok: number; prefill_tok_s: number;
  decode_tok: number;  decode_tok_s: number;
};

// Seed is a 10-token sentence (Qwen tokenizer); repeating it ~13× ≈ 130
// tokens ≈ pp128, ~51× ≈ 510 tokens ≈ pp512.
const SEED = "The quick brown fox jumps over the lazy dog. ";

async function benchOllama(model: string, prompt: string, num_predict = 128): Promise<OllamaBench | null> {
  try {
    await fetch("http://localhost:11434/api/generate", {
      method: "POST",
      body: JSON.stringify({ model, prompt: "hi", stream: false,
        options: { num_predict: 4, temperature: 0 } }),
    });
    const res = await fetch("http://localhost:11434/api/generate", {
      method: "POST",
      body: JSON.stringify({ model, prompt, stream: false,
        options: { num_predict, temperature: 0 } }),
    });
    const d: any = await res.json();
    if (!d.eval_count) return null;
    return {
      prefill_tok: d.prompt_eval_count,
      prefill_tok_s: d.prompt_eval_count / (d.prompt_eval_duration / 1e9),
      decode_tok: d.eval_count,
      decode_tok_s: d.eval_count / (d.eval_duration / 1e9),
    };
  } catch (e) {
    console.error(`ollama bench failed for ${model}:`, e);
    return null;
  }
}

type HipBench = {
  pp128: number; pp512: number;
  decode_tok_s: number; ttft_ms: number;
};

async function benchHipfire(tag: string): Promise<HipBench | null> {
  const proc = spawn({
    cmd: ["bun", `${import.meta.dir}/index.ts`, "bench", tag],
    stdout: "pipe", stderr: "inherit",
  });
  const out = await new Response(proc.stdout).text();
  await proc.exited;
  const pp = out.match(/pp128\s+([\d.]+)/);
  const pp2 = out.match(/pp512\s+([\d.]+)/);
  const dec = out.match(/Decode\s+tok\/s\s+([\d.]+)/);
  const ttft = out.match(/TTFT\s+ms\s+([\d.]+)/);
  if (!pp || !dec) { console.error("failed to parse hipfire bench"); return null; }
  return {
    pp128: parseFloat(pp[1]),
    pp512: pp2 ? parseFloat(pp2[1]) : NaN,
    decode_tok_s: parseFloat(dec[1]),
    ttft_ms: ttft ? parseFloat(ttft[1]) : NaN,
  };
}

const PAIRS: { name: string; hipfire: string; ollama: string }[] = [
  { name: "0.8b", hipfire: "qwen3.5:0.8b", ollama: "qwen3.5:0.8b" },
  { name: "4b",   hipfire: "qwen3.5:4b",   ollama: "qwen3.5:4b" },
  { name: "9b",   hipfire: "qwen3.5:9b",   ollama: "qwen3.5:9b" },
];

const args = process.argv.slice(2);
const filter = args.length ? args[0] : null;

// Build prompts that tokenize to ~128 and ~512 tokens.
const prompt128 = SEED.repeat(13);   // ~128 tok
const prompt512 = SEED.repeat(52);   // ~512 tok

const rows: any[] = [];
for (const p of PAIRS) {
  if (filter && p.hipfire !== filter) continue;
  console.log(`\n=== ${p.name} ===`);

  console.log(`→ hipfire asym3 ...`);
  const h = await benchHipfire(p.hipfire);
  if (h) console.log(`  pp128=${h.pp128.toFixed(0)} pp512=${h.pp512.toFixed(0)} decode=${h.decode_tok_s.toFixed(1)} ttft=${h.ttft_ms.toFixed(1)}ms`);

  // num_predict=128 matches hipfire's default decode-run length. Short
  // generations under-report decode tok/s because launch-overhead
  // dominates; 128 is long enough to hit steady state.
  console.log(`→ ollama Q4_K_M (pp128 + tg128)...`);
  const o128 = await benchOllama(p.ollama, prompt128, 128);
  if (o128) console.log(`  prefill=${o128.prefill_tok_s.toFixed(0)} tok/s (${o128.prefill_tok} tok) decode=${o128.decode_tok_s.toFixed(1)} tok/s (${o128.decode_tok} tok)`);

  console.log(`→ ollama Q4_K_M (pp512 + tg128)...`);
  const o512 = await benchOllama(p.ollama, prompt512, 128);
  if (o512) console.log(`  prefill=${o512.prefill_tok_s.toFixed(0)} tok/s (${o512.prefill_tok} tok) decode=${o512.decode_tok_s.toFixed(1)} tok/s (${o512.decode_tok} tok)`);

  // Decode baseline from the larger prefill measurement (steady state).
  rows.push({ model: p.name, hipfire: h, ollama128: o128, ollama512: o512 });
}

console.log(`\n\n╔═════════ hipfire asym3 vs ollama Q4_K_M (7900 XTX) ═════════╗\n`);
const fmt = (n?: number) => n === undefined || Number.isNaN(n) ? "   —  " : n.toFixed(0).padStart(7);
const fmtF = (n?: number) => n === undefined || Number.isNaN(n) ? "  —   " : n.toFixed(1).padStart(7);
console.log(`Model    |   hf pp128 |  oll pp128 |   hf pp512 |  oll pp512 |   hf dec |  oll dec | decode×`);
console.log(`---------|------------|------------|------------|------------|----------|----------|--------`);
for (const r of rows) {
  const h = r.hipfire || {};
  const o128 = r.ollama128 || {};
  const o512 = r.ollama512 || {};
  // Use the LONGER ollama prefill number for the ollama decode baseline
  // (identical results, just more prefill tokens). Decode speedup shown
  // against the pp128 decode number since that matches hipfire's default.
  const odec = o128.decode_tok_s ?? o512.decode_tok_s;
  const speedup = h.decode_tok_s && odec
    ? `${(h.decode_tok_s / odec).toFixed(2)}×`
    : " —";
  console.log(`${r.model.padEnd(8)} |    ${fmt(h.pp128)} |    ${fmt(o128.prefill_tok_s)} |    ${fmt(h.pp512)} |    ${fmt(o512.prefill_tok_s)} |  ${fmtF(h.decode_tok_s)} |  ${fmtF(odec)} |  ${speedup}`);
}
