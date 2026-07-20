#!/usr/bin/env node
// Local terminal transcriber — runs entirely offline on your Mac.
// Takes any audio OR video file, extracts the audio with ffmpeg, then
// transcribes it with a local whisper.cpp model (Metal-accelerated on Apple
// Silicon). Prints the transcript and copies it to the clipboard.
//
// Nothing leaves your machine and no API key is needed.
//
// Usage:  node transcribe.js <file-or-folder> [-en | -lv]
//
// Pass a folder and every audio/video file inside is transcribed; the .txt
// transcripts are written to a new sibling folder named "<folder>-transcripts".

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
// Extensions we hand to ffmpeg when scanning a folder. ffmpeg reads far more,
// but this keeps us from trying to "transcribe" random files (.txt, .jpg, ...).
const MEDIA_EXTS = new Set([
  ".mp3", ".m4a", ".wav", ".flac", ".ogg", ".opus", ".aac", ".wma", ".aiff",
  ".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".flv", ".wmv", ".3gp",
]);

// --- tiny logger (all progress goes to stderr so stdout stays clean) --------

const log = (...a) => process.stderr.write(a.join(" ") + "\n");
const die = (msg) => {
  process.stderr.write(`\nError: ${msg}\n`);
  process.exit(1);
};

// --- arg parsing ------------------------------------------------------------

function parseArgs(argv) {
  const args = { input: null, language: "auto" };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "-en") args.language = "en";
    else if (a === "-lv") args.language = "lv";
    else if (a === "--help" || a === "-h") args.help = true;
    else if (!a.startsWith("-") && !args.input) args.input = a;
    else die(`unknown argument: ${a}`);
  }
  return args;
}

function printHelp() {
  log(`transcribe — local, offline audio/video transcription (whisper.cpp)

Usage:
  node transcribe.js <file-or-folder> [language]

Language:
  -en          Force English
  -lv          Force Latvian
  (none)       Auto-detect (default)

Other:
  -h, --help   Show this help

Pass a single file: the transcript is printed, copied to your clipboard, and
saved to output/<name>.txt.

Pass a folder: every audio/video file inside is transcribed and each transcript
is saved to a new sibling folder named "<folder>-transcripts/<name>.txt".

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

// Extract + transcribe one media file, returning the cleaned transcript text.
// `workDir` is a caller-owned scratch dir; the temp WAV is removed afterwards.
async function transcribeOne(input, language, workDir) {
  const wav = path.join(workDir, "audio.wav");
  try {
    await extractAudio(input, wav);
    const raw = await transcribe(wav, language);
    return raw.replace(/\n{3,}/g, "\n\n").trim();
  } finally {
    fs.rmSync(wav, { force: true });
  }
}

// --- single-file mode -------------------------------------------------------

async function runFile(input, language, workDir) {
  log(`Extracting audio from ${path.basename(input)} ...`);
  log(`Transcribing with ${path.basename(MODEL)} (Metal GPU) ...`);
  const transcript = await transcribeOne(input, language, workDir);
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
}

// --- folder mode ------------------------------------------------------------

async function runFolder(dir, language, workDir) {
  const entries = fs
    .readdirSync(dir, { withFileTypes: true })
    .filter((e) => e.isFile() && MEDIA_EXTS.has(path.extname(e.name).toLowerCase()))
    .map((e) => e.name)
    .sort((a, b) => a.localeCompare(b));

  if (entries.length === 0)
    die(`no audio or video files found in folder: ${dir}`);

  // New sibling folder: "<folder>-transcripts" (next to the input folder).
  const outDir = path.join(
    path.dirname(dir),
    `${path.basename(dir)}-transcripts`
  );
  fs.mkdirSync(outDir, { recursive: true });

  log(`Found ${entries.length} file(s). Writing transcripts to ${outDir}\n`);

  let ok = 0;
  let failed = 0;
  for (let i = 0; i < entries.length; i++) {
    const name = entries[i];
    const input = path.join(dir, name);
    const label = `[${i + 1}/${entries.length}] ${name}`;
    try {
      log(`${label} — transcribing ...`);
      const transcript = await transcribeOne(input, language, workDir);
      const base = path.basename(name, path.extname(name));
      const outFile = path.join(outDir, `${base}.txt`);
      if (!transcript) {
        log(`${label} — no speech detected, skipped.`);
        failed++;
        continue;
      }
      fs.writeFileSync(outFile, transcript + "\n");
      log(`${label} — saved ${path.basename(outFile)}`);
      ok++;
    } catch (err) {
      log(`${label} — failed: ${err?.message || String(err)}`);
      failed++;
    }
  }

  log(`\nDone. ${ok} transcribed, ${failed} skipped/failed.`);
  log(`Transcripts are in ${outDir}`);
}

// --- main -------------------------------------------------------------------

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help || !args.input) {
    printHelp();
    process.exit(args.input ? 0 : 1);
  }

  const input = path.resolve(args.input);
  if (!fs.existsSync(input)) die(`path not found: ${input}`);
  if (!fs.existsSync(MODEL))
    die(`model not found: ${MODEL}\nDownload it (see README) or set TRANSCRIBE_MODEL in .env.`);

  requireBinary("ffmpeg", "Install it: brew install ffmpeg");
  requireBinary("whisper-cli", "Install it: brew install whisper-cpp");

  const isDir = fs.statSync(input).isDirectory();
  const workDir = fs.mkdtempSync(path.join(os.tmpdir(), "transcriber-"));

  try {
    if (isDir) await runFolder(input, args.language, workDir);
    else await runFile(input, args.language, workDir);
  } catch (err) {
    die(err?.message || String(err));
  } finally {
    fs.rmSync(workDir, { recursive: true, force: true });
  }
}

main();
