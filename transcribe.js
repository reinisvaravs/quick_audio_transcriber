#!/usr/bin/env node
// Local terminal transcriber — runs entirely offline on your Mac.
// Takes any audio OR video file, extracts the audio with ffmpeg, then
// transcribes it with a local whisper.cpp model (Metal-accelerated on Apple
// Silicon). Prints the transcript and copies it to the clipboard.
//
// Nothing leaves your machine and no API key is needed.
//
// Usage:  node transcribe.js <file> [-en | -lv]

import "dotenv/config";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { spawn, spawnSync } from "node:child_process";

// --- config -----------------------------------------------------------------

const HERE = path.dirname(fileURLToPath(import.meta.url));
// large-v3-turbo: best transcription accuracy that fits comfortably on 8GB.
const MODEL =
  process.env.TRANSCRIBE_MODEL ||
  path.join(HERE, "models", "ggml-large-v3-turbo.bin");
// Transcripts are always saved here.
const OUTPUT_DIR = path.join(HERE, "output");
// M3 = 4 performance + 4 efficiency cores; 4 threads keeps it snappy.
const THREADS = process.env.TRANSCRIBE_THREADS || "4";

// --- tiny logger (all progress goes to stderr so stdout stays clean) --------

const log = (...a) => process.stderr.write(a.join(" ") + "\n");
const die = (msg) => {
  process.stderr.write(`\nError: ${msg}\n`);
  process.exit(1);
};

// --- arg parsing ------------------------------------------------------------

function parseArgs(argv) {
  const args = { file: null, language: "auto" };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "-en") args.language = "en";
    else if (a === "-lv") args.language = "lv";
    else if (a === "--help" || a === "-h") args.help = true;
    else if (!a.startsWith("-") && !args.file) args.file = a;
    else die(`unknown argument: ${a}`);
  }
  return args;
}

function printHelp() {
  log(`transcribe — local, offline audio/video transcription (whisper.cpp)

Usage:
  node transcribe.js <file> [language]

Language:
  -en          Force English
  -lv          Force Latvian
  (none)       Auto-detect (default)

Other:
  -h, --help   Show this help

The transcript is printed, copied to your clipboard, and always saved to output/.
Works with any audio or video file ffmpeg can read (mp3, m4a, wav, mp4, mov, ...).
For video, only the audio track is used. Runs fully offline — no API key, no cost.`);
}

// --- external binaries ------------------------------------------------------

function requireBinary(name, hint) {
  const r = spawnSync(name, ["--help"], { stdio: "ignore" });
  // whisper-cli returns non-zero on --help; treat "spawned at all" as success.
  if (r.error && r.error.code === "ENOENT") die(`${name} not found on PATH. ${hint || ""}`);
}

// Extract + downmix the audio to the 16kHz mono 16-bit WAV whisper.cpp expects.
// No compression/size limit needed since transcription is local.
function runFfmpeg(fnArgs) {
  return new Promise((resolve, reject) => {
    const p = spawn("ffmpeg", ["-hide_banner", "-loglevel", "error", "-y", ...fnArgs]);
    let err = "";
    p.stderr.on("data", (d) => (err += d));
    p.on("error", reject);
    p.on("close", (code) =>
      code === 0 ? resolve() : reject(new Error(err.trim() || `ffmpeg exited ${code}`))
    );
  });
}

async function extractAudio(input, outWav) {
  await runFfmpeg([
    "-i", input,
    "-vn",              // drop any video track
    "-ac", "1",         // mono
    "-ar", "16000",     // 16kHz — what whisper expects
    "-c:a", "pcm_s16le", // 16-bit PCM WAV
    outWav,
  ]);
}

// Run whisper-cli and capture the transcript from stdout.
function transcribe(wav, language) {
  return new Promise((resolve, reject) => {
    const wArgs = [
      "-m", MODEL,
      "-f", wav,
      "-nt",              // no timestamps — clean prose
      "-l", language,
      "-t", THREADS,
    ];
    const p = spawn("whisper-cli", wArgs);
    let out = "";
    let err = "";
    p.stdout.on("data", (d) => (out += d));
    p.stderr.on("data", (d) => (err += d));
    p.on("error", reject);
    p.on("close", (code) =>
      code === 0
        ? resolve(out)
        : reject(new Error(err.trim() || `whisper-cli exited ${code}`))
    );
  });
}

// --- main -------------------------------------------------------------------

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help || !args.file) {
    printHelp();
    process.exit(args.file ? 0 : 1);
  }

  const input = path.resolve(args.file);
  if (!fs.existsSync(input)) die(`file not found: ${input}`);
  if (!fs.existsSync(MODEL))
    die(`model not found: ${MODEL}\nDownload it (see README) or set TRANSCRIBE_MODEL in .env.`);

  requireBinary("ffmpeg", "Install it: brew install ffmpeg");
  requireBinary("whisper-cli", "Install it: brew install whisper-cpp");

  const workDir = fs.mkdtempSync(path.join(os.tmpdir(), "transcriber-"));

  try {
    log(`Extracting audio from ${path.basename(input)} ...`);
    const wav = path.join(workDir, "audio.wav");
    await extractAudio(input, wav);

    log(`Transcribing with ${path.basename(MODEL)} (Metal GPU) ...`);
    const raw = await transcribe(wav, args.language);
    const transcript = raw.replace(/\n{3,}/g, "\n\n").trim();

    if (!transcript) die("no speech detected in the file.");

    // stdout = clean transcript (pipe-friendly).
    process.stdout.write(transcript + "\n");

    // Always save to output/<input-name>.txt.
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
    const base = path.basename(input, path.extname(input));
    const outFile = path.join(OUTPUT_DIR, `${base}.txt`);
    fs.writeFileSync(outFile, transcript + "\n");
    log(`\nSaved to ${path.relative(HERE, outFile)}`);

    // Always copy to clipboard.
    const pb = spawnSync("pbcopy", { input: transcript });
    if (pb.status === 0) log("Copied to clipboard.");
  } catch (err) {
    die(err?.message || String(err));
  } finally {
    fs.rmSync(workDir, { recursive: true, force: true });
  }
}

main();
