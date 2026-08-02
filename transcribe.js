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
import { MEDIA_EXTS } from "./formats.js";
import {
  isInstagramUrl,
  instagramShortcode,
  fetchInstagramVideo,
} from "./instagram.js";

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
// Instagram support lives in instagram.js, shared with the bot: it resolves a
// public post/reel to a direct MP4 and downloads it into memory, so the video
// is fed straight to ffmpeg and never saved to disk.
//
// Extensions we hand to ffmpeg when scanning a folder (see formats.js). ffmpeg
// reads far more, but this keeps us from trying to "transcribe" random files
// (.txt, .jpg, ...).

// --- tiny logger (all progress goes to stderr so stdout stays clean) --------

const log = (...a) => process.stderr.write(a.join(" ") + "\n");
const die = (msg) => {
  process.stderr.write(`\nError: ${msg}\n`);
  process.exit(1);
};

// --- arg parsing ------------------------------------------------------------

function parseArgs(argv) {
  const args = { input: null, language: "auto", outDir: null };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "-en") args.language = "en";
    else if (a === "-lv") args.language = "lv";
    else if (a === "--help" || a === "-h") args.help = true;
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
  log(`transcribe — local, offline audio/video transcription (whisper.cpp)

Usage:
  node transcribe.js <file-or-folder-or-instagram-url> [language]

Language:
  -en          Force English
  -lv          Force Latvian
  (none)       Auto-detect (default)

Output:
  -o, --out <dir>   Save transcripts to <dir> instead of the default location.
                    Created if it doesn't exist. Works in every mode.

Other:
  -h, --help   Show this help

Pass a single file: the transcript is printed, copied to your clipboard, and
saved to output/<name>.txt (or <dir>/<name>.txt with -o).

Pass a folder: every audio/video file inside is transcribed and each transcript
is saved to a new sibling folder named "<folder>-transcripts/<name>.txt"
(or to <dir>/<name>.txt with -o).

Pass an Instagram URL: the public post/reel is downloaded into memory via
downreels.com and transcribed. The transcript is printed, copied to your
clipboard, and saved to output/<shortcode>.txt.

  node transcribe.js https://www.instagram.com/reel/XXXXXXXXXXX/

Works with any audio or video file ffmpeg can read (mp3, m4a, wav, mp4, mov, ...).
For video, only the audio track is used. Transcription runs fully offline — no API
key, no cost. (Instagram mode fetches the video over the network, then transcribes
it locally.)`);
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

// --- Instagram mode ---------------------------------------------------------

// Same as extractAudio, but the source is an in-memory MP4 fed via ffmpeg stdin.
function extractAudioFromBuffer(buf, outWav) {
  return new Promise((resolve, reject) => {
    const p = spawn("ffmpeg", [
      "-hide_banner", "-loglevel", "error", "-y",
      "-i", "pipe:0",     // read the MP4 from stdin
      "-vn",              // drop the video track
      "-ac", "1",         // mono
      "-ar", "16000",     // 16kHz — what whisper expects
      "-c:a", "pcm_s16le", // 16-bit PCM WAV
      outWav,
    ]);
    let err = "";
    p.stderr.on("data", (d) => (err += d));
    p.on("error", reject);
    p.on("close", (code) =>
      code === 0 ? resolve() : reject(new Error(err.trim() || `ffmpeg exited ${code}`))
    );
    // ffmpeg may close stdin early once it has the moov atom; swallow EPIPE.
    p.stdin.on("error", () => {});
    p.stdin.end(buf);
  });
}

// Extract audio from an in-memory MP4 and transcribe it. The temp WAV whisper
// needs is written to workDir and removed afterwards; the MP4 never touches disk.
async function transcribeBuffer(buf, language, workDir) {
  const wav = path.join(workDir, "audio.wav");
  try {
    await extractAudioFromBuffer(buf, wav);
    const raw = await transcribe(wav, language);
    return raw.replace(/\n{3,}/g, "\n\n").trim();
  } finally {
    fs.rmSync(wav, { force: true });
  }
}

async function runInstagram(pageUrl, language, workDir, outDir) {
  log(`Resolving Instagram video via downreels.com ...`);
  const buf = await fetchInstagramVideo(pageUrl);

  log(
    `Downloaded ${(buf.length / 1024 / 1024).toFixed(1)} MB. ` +
      `Transcribing with ${path.basename(MODEL)} (Metal GPU) ...`
  );
  const transcript = await transcribeBuffer(buf, language, workDir);
  if (!transcript) die("no speech detected in the video.");

  // stdout = clean transcript (pipe-friendly).
  process.stdout.write(transcript + "\n");

  // Save to <outDir>/<shortcode>.txt (default: output/).
  const destDir = outDir || OUTPUT_DIR;
  fs.mkdirSync(destDir, { recursive: true });
  const outFile = path.join(destDir, `${instagramShortcode(pageUrl)}.txt`);
  fs.writeFileSync(outFile, transcript + "\n");
  log(`\nSaved to ${path.relative(HERE, outFile)}`);

  // Copy to clipboard.
  const pb = spawnSync("pbcopy", { input: transcript });
  if (pb.status === 0) log("Copied to clipboard.");
}

// --- single-file mode -------------------------------------------------------

async function runFile(input, language, workDir, outDir) {
  log(`Extracting audio from ${path.basename(input)} ...`);
  log(`Transcribing with ${path.basename(MODEL)} (Metal GPU) ...`);
  const transcript = await transcribeOne(input, language, workDir);
  if (!transcript) die("no speech detected in the file.");

  // stdout = clean transcript (pipe-friendly).
  process.stdout.write(transcript + "\n");

  // Always save to <outDir>/<input-name>.txt (default: output/).
  const destDir = outDir || OUTPUT_DIR;
  fs.mkdirSync(destDir, { recursive: true });
  const base = path.basename(input, path.extname(input));
  const outFile = path.join(destDir, `${base}.txt`);
  fs.writeFileSync(outFile, transcript + "\n");
  log(`\nSaved to ${path.relative(HERE, outFile)}`);

  // Always copy to clipboard.
  const pb = spawnSync("pbcopy", { input: transcript });
  if (pb.status === 0) log("Copied to clipboard.");
}

// --- folder mode ------------------------------------------------------------

// <destDir>/<base>.txt, or <base>-2.txt, -3.txt … if that name is taken. A
// folder holding the same recording as talk.m4a and talk.wav would otherwise
// have the second transcript quietly overwrite the first.
function uniquePath(destDir, base) {
  let candidate = path.join(destDir, `${base}.txt`);
  for (let n = 2; fs.existsSync(candidate); n++)
    candidate = path.join(destDir, `${base}-${n}.txt`);
  return candidate;
}

async function runFolder(dir, language, workDir, outDir) {
  const entries = fs
    .readdirSync(dir, { withFileTypes: true })
    .filter((e) => e.isFile() && MEDIA_EXTS.has(path.extname(e.name).toLowerCase()))
    .map((e) => e.name)
    .sort((a, b) => a.localeCompare(b));

  if (entries.length === 0)
    die(`no audio or video files found in folder: ${dir}`);

  // Default: a new sibling folder "<folder>-transcripts" next to the input
  // folder. With -o, use the given directory instead.
  const destDir =
    outDir || path.join(path.dirname(dir), `${path.basename(dir)}-transcripts`);
  fs.mkdirSync(destDir, { recursive: true });

  log(`Found ${entries.length} file(s). Writing transcripts to ${destDir}\n`);

  let ok = 0;
  let failed = 0;
  for (let i = 0; i < entries.length; i++) {
    const name = entries[i];
    const input = path.join(dir, name);
    const label = `[${i + 1}/${entries.length}] ${name}`;
    try {
      log(`${label} — transcribing ...`);
      const transcript = await transcribeOne(input, language, workDir);
      const outFile = uniquePath(destDir, path.basename(name, path.extname(name)));
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
  log(`Transcripts are in ${destDir}`);
}

// --- main -------------------------------------------------------------------

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help || !args.input) {
    printHelp();
    process.exit(args.input ? 0 : 1);
  }

  const rawInput = args.input;
  const igMode = isInstagramUrl(rawInput);

  // A URL we can't resolve to a video isn't a local path either. Instagram
  // links to a profile or a story land here too — only posts, reels and IGTV
  // carry a video we can fetch.
  if (!igMode && /^https?:\/\//i.test(rawInput))
    die(
      `only Instagram post/reel URLs are supported for download ` +
        `(instagram.com/p/... , /reel/... , /tv/...); got: ${rawInput}`
    );

  if (!fs.existsSync(MODEL))
    die(`model not found: ${MODEL}\nDownload it (see README) or set TRANSCRIBE_MODEL in .env.`);

  requireBinary("ffmpeg", "Install it: brew install ffmpeg");
  requireBinary("whisper-cli", "Install it: brew install whisper-cpp");

  const workDir = fs.mkdtempSync(path.join(os.tmpdir(), "transcriber-"));

  try {
    if (igMode) {
      await runInstagram(rawInput, args.language, workDir, args.outDir);
    } else {
      const input = path.resolve(rawInput);
      if (!fs.existsSync(input)) die(`path not found: ${input}`);
      const isDir = fs.statSync(input).isDirectory();
      if (isDir) await runFolder(input, args.language, workDir, args.outDir);
      else await runFile(input, args.language, workDir, args.outDir);
    }
  } catch (err) {
    die(err?.message || String(err));
  } finally {
    fs.rmSync(workDir, { recursive: true, force: true });
  }
}

main();
