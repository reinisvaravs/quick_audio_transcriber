#!/usr/bin/env node
// Long-recording transcriber — same engine as transcribe.js, but built for
// files measured in hours rather than minutes.
//
// transcribe.js runs whisper.cpp once over the whole file: fine for a reel,
// risky for a 7-hour lecture, where an hour of GPU time is lost to any single
// failure and nothing is printed until the very end. This splits the audio into
// chunks, transcribes them one at a time, and writes each chunk's text to disk
// as it lands. Re-running skips every chunk already done, so an interrupted run
// picks up where it stopped.
//
// Chunk boundaries are placed on silence (found near each target mark) so no
// word is cut in half by the split.
//
// Usage:  node transcribe-long.js <file> [-en | -lv]

import "dotenv/config";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { spawn, spawnSync } from "node:child_process";

// --- config -----------------------------------------------------------------

const HERE = path.dirname(fileURLToPath(import.meta.url));
const MODEL =
  process.env.TRANSCRIBE_MODEL ||
  path.join(HERE, "models", "ggml-large-v3-turbo.bin");
const OUTPUT_DIR = path.join(HERE, "output");
// Metal does the work; 4 threads matches transcribe.js.
const THREADS = process.env.TRANSCRIBE_THREADS || "4";
// Chunk length. 20 min keeps the resume granularity tight (~2.5 min of GPU
// time lost to a crash) without paying model-startup cost too often.
const CHUNK_SECONDS = Number(process.env.TRANSCRIBE_CHUNK_SECONDS || 1200);
// How far from a target mark we'll look for a silence to cut on.
const SILENCE_SEARCH_WINDOW = 90;
// Resumable state lives here, outside output/ so it isn't mistaken for a result.
const CACHE_ROOT = path.join(HERE, ".transcribe-cache");

// --- tiny logger (all progress goes to stderr so stdout stays clean) --------

const log = (...a) => process.stderr.write(a.join(" ") + "\n");
const die = (msg) => {
  process.stderr.write(`\nError: ${msg}\n`);
  process.exit(1);
};

const fmtDuration = (s) => {
  const t = Math.round(s);
  const h = Math.floor(t / 3600);
  const m = Math.floor((t % 3600) / 60);
  const sec = t % 60;
  return h ? `${h}h${String(m).padStart(2, "0")}m` : `${m}m${String(sec).padStart(2, "0")}s`;
};

// --- arg parsing ------------------------------------------------------------

function parseArgs(argv) {
  const args = { input: null, language: "auto", outDir: null };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "-en") args.language = "en";
    else if (a === "-lv") args.language = "lv";
    else if (a === "--help" || a === "-h") args.help = true;
    else if (a === "--clean") args.clean = true;
    else if (a === "-o" || a === "--out") {
      const next = argv[++i];
      if (!next) die(`${a} needs a directory path`);
      args.outDir = path.resolve(next);
    } else if (!a.startsWith("-") && !args.input) args.input = a;
    else die(`unknown argument: ${a}`);
  }
  return args;
}

function printHelp() {
  log(`transcribe-long — resumable offline transcription for multi-hour files

Usage:
  node transcribe-long.js <file> [language]

Language:
  -en          Force English
  -lv          Force Latvian
  (none)       Auto-detect (default)

Output:
  -o, --out <dir>   Save the transcript to <dir> instead of output/

Other:
  --clean      Discard cached chunks for this file and start over
  -h, --help   Show this help

The file is split on silence into ~${Math.round(CHUNK_SECONDS / 60)}-minute chunks (override with
TRANSCRIBE_CHUNK_SECONDS). Each chunk's text is written to
.transcribe-cache/ as soon as it finishes, so re-running after an interruption
resumes instead of restarting. The joined transcript is printed, copied to the
clipboard, and saved to output/<name>.txt.

For anything under an hour, use transcribe.js instead — it is simpler and the
chunking buys you nothing.`);
}

// --- external binaries ------------------------------------------------------

function requireBinary(name, hint) {
  const r = spawnSync(name, ["--help"], { stdio: "ignore" });
  // whisper-cli returns non-zero on --help; treat "spawned at all" as success.
  if (r.error && r.error.code === "ENOENT") die(`${name} not found on PATH. ${hint || ""}`);
}

// Run a command, resolving with its stdout. stderr is collected for the error
// message only — these are long jobs and a bare exit code says nothing.
function run(cmd, cmdArgs) {
  return new Promise((resolve, reject) => {
    const p = spawn(cmd, cmdArgs);
    let out = "";
    let err = "";
    p.stdout.on("data", (d) => (out += d));
    p.stderr.on("data", (d) => (err += d));
    p.on("error", reject);
    p.on("close", (code) =>
      code === 0 ? resolve(out) : reject(new Error(err.trim() || `${cmd} exited ${code}`))
    );
  });
}

// --- planning ---------------------------------------------------------------

async function probeDuration(input) {
  const out = await run("ffprobe", [
    "-v", "error",
    "-show_entries", "format=duration",
    "-of", "csv=p=0",
    input,
  ]);
  const seconds = Number(out.trim());
  if (!Number.isFinite(seconds) || seconds <= 0)
    throw new Error(`could not read the duration of ${path.basename(input)}`);
  return seconds;
}

// One decode pass over the file, collecting every silent stretch. ffmpeg writes
// silencedetect results to stderr, so this one can't use run().
function detectSilences(input) {
  return new Promise((resolve, reject) => {
    const p = spawn("ffmpeg", [
      "-hide_banner",
      "-nostats",
      "-i", input,
      "-af", "silencedetect=noise=-30dB:d=0.35",
      "-f", "null",
      "-",
    ]);
    let err = "";
    p.stderr.on("data", (d) => (err += d));
    p.on("error", reject);
    p.on("close", (code) => {
      if (code !== 0) return reject(new Error(err.trim() || `ffmpeg exited ${code}`));
      // Pair each "silence_start: X" with the "silence_end: Y" that follows it.
      const silences = [];
      let start = null;
      for (const line of err.split("\n")) {
        const s = line.match(/silence_start:\s*(-?[\d.]+)/);
        if (s) start = Number(s[1]);
        const e = line.match(/silence_end:\s*(-?[\d.]+)/);
        if (e && start !== null) {
          silences.push({ start, end: Number(e[1]) });
          start = null;
        }
      }
      resolve(silences);
    });
  });
}

// Cut points at roughly CHUNK_SECONDS apart, nudged onto the middle of whatever
// silence sits closest to each mark. Falls back to the exact mark when the
// speaker talks straight through it.
function planChunks(duration, silences) {
  const cuts = [0];
  for (let mark = CHUNK_SECONDS; mark < duration - 60; mark += CHUNK_SECONDS) {
    let best = null;
    for (const s of silences) {
      const mid = (s.start + s.end) / 2;
      const drift = Math.abs(mid - mark);
      if (drift > SILENCE_SEARCH_WINDOW) continue;
      // Prefer the closest silence to the mark; break ties toward longer gaps.
      if (!best || drift < best.drift) best = { mid, drift };
    }
    const cut = best ? best.mid : mark;
    if (cut > cuts[cuts.length - 1] + 60) cuts.push(cut);
  }
  cuts.push(duration);

  const chunks = [];
  for (let i = 0; i < cuts.length - 1; i++)
    chunks.push({ index: i, start: cuts[i], end: cuts[i + 1] });
  return chunks;
}

// --- per-chunk work ---------------------------------------------------------

// Decode just this slice to the 16kHz mono WAV whisper.cpp expects. -ss before
// -i seeks without decoding the skipped audio, which matters at hour 6.
async function extractChunk(input, chunk, wav) {
  await run("ffmpeg", [
    "-hide_banner", "-loglevel", "error", "-y",
    "-ss", String(chunk.start),
    "-to", String(chunk.end),
    "-i", input,
    "-vn",
    "-ac", "1",
    "-ar", "16000",
    "-c:a", "pcm_s16le",
    wav,
  ]);
}

async function transcribeChunk(wav, language) {
  const raw = await run("whisper-cli", [
    "-m", MODEL,
    "-f", wav,
    "-nt",              // no timestamps — clean prose
    "-l", language,
    "-t", THREADS,
  ]);
  return raw.replace(/\n{3,}/g, "\n\n").trim();
}

// --- main -------------------------------------------------------------------

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help || !args.input) {
    printHelp();
    process.exit(args.input ? 0 : 1);
  }

  if (!fs.existsSync(MODEL))
    die(`model not found: ${MODEL}\nDownload it (see README) or set TRANSCRIBE_MODEL in .env.`);
  requireBinary("ffmpeg", "Install it: brew install ffmpeg");
  requireBinary("ffprobe", "Install it: brew install ffmpeg");
  requireBinary("whisper-cli", "Install it: brew install whisper-cpp");

  const input = path.resolve(args.input);
  if (!fs.existsSync(input)) die(`path not found: ${input}`);
  if (fs.statSync(input).isDirectory())
    die(`this is for a single long file; use transcribe.js for folders.`);

  const base = path.basename(input, path.extname(input));
  // Size in the key so an edited or replaced file doesn't reuse stale chunks.
  const cacheDir = path.join(
    CACHE_ROOT,
    `${base.replace(/[/\\]/g, "_")}-${fs.statSync(input).size}`
  );
  if (args.clean) fs.rmSync(cacheDir, { recursive: true, force: true });
  fs.mkdirSync(cacheDir, { recursive: true });

  // The plan is cached too: resuming has to reuse the exact boundaries the
  // finished chunks were cut on, or the transcript would gain or lose audio.
  const planFile = path.join(cacheDir, "plan.json");
  let chunks;
  if (fs.existsSync(planFile)) {
    chunks = JSON.parse(fs.readFileSync(planFile, "utf8"));
    log(`Resuming ${base} — ${chunks.length} chunks planned.`);
  } else {
    const duration = await probeDuration(input);
    log(`${base} — ${fmtDuration(duration)}. Scanning for silence to cut on ...`);
    const silences = await detectSilences(input);
    chunks = planChunks(duration, silences);
    fs.writeFileSync(planFile, JSON.stringify(chunks, null, 2));
    log(`Split into ${chunks.length} chunks (${silences.length} silences found).`);
  }

  const done = chunks.filter((c) => fs.existsSync(path.join(cacheDir, `${c.index}.txt`)));
  if (done.length) log(`${done.length}/${chunks.length} chunks already transcribed — skipping those.`);

  const wav = path.join(cacheDir, "current.wav");
  const startedAt = Date.now();
  let secondsDone = 0;

  for (const chunk of chunks) {
    const txtFile = path.join(cacheDir, `${chunk.index}.txt`);
    const length = chunk.end - chunk.start;
    if (fs.existsSync(txtFile)) continue;

    const n = chunk.index + 1;
    process.stderr.write(
      `[${n}/${chunks.length}] ${fmtDuration(chunk.start)}–${fmtDuration(chunk.end)} ... `
    );
    const t0 = Date.now();
    try {
      await extractChunk(input, chunk, wav);
      const text = await transcribeChunk(wav, args.language);
      // Written only after a clean exit, so a half-transcribed chunk is redone.
      fs.writeFileSync(txtFile, text + "\n");
    } catch (err) {
      log("failed.");
      die(
        `${err?.message || String(err)}\n\n` +
          `Finished chunks are cached — re-run the same command to resume.`
      );
    }
    const took = (Date.now() - t0) / 1000;
    secondsDone += length;

    // ETA from this run's own throughput, not the benchmark.
    const elapsed = (Date.now() - startedAt) / 1000;
    const remaining = chunks
      .filter((c) => c.index > chunk.index && !fs.existsSync(path.join(cacheDir, `${c.index}.txt`)))
      .reduce((sum, c) => sum + (c.end - c.start), 0);
    const eta = remaining * (elapsed / secondsDone);
    log(`done in ${fmtDuration(took)} (${(length / took).toFixed(1)}x) — ETA ${fmtDuration(eta)}`);
  }
  fs.rmSync(wav, { force: true });

  const transcript = chunks
    .map((c) => fs.readFileSync(path.join(cacheDir, `${c.index}.txt`), "utf8").trim())
    .filter(Boolean)
    .join("\n");
  if (!transcript) die("no speech detected in the file.");

  // stdout = clean transcript (pipe-friendly).
  process.stdout.write(transcript + "\n");

  const destDir = args.outDir || OUTPUT_DIR;
  fs.mkdirSync(destDir, { recursive: true });
  const outFile = path.join(destDir, `${base}.txt`);
  fs.writeFileSync(outFile, transcript + "\n");
  log(`\nSaved to ${path.relative(HERE, outFile)}`);

  const pb = spawnSync("pbcopy", { input: transcript });
  if (pb.status === 0) log("Copied to clipboard.");

  // Cache is kept on purpose: a re-run is then instant, and --clean drops it.
  log(`Chunk cache: ${path.relative(HERE, cacheDir)} (delete it, or pass --clean, to redo)`);
}

main().catch((err) => die(err?.message || String(err)));
