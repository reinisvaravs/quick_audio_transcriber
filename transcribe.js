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
// Instagram support: downreels.com's backend resolves a public IG post/reel to a
// direct MP4. We call the same endpoint its web UI calls, download the video into
// memory, and feed the bytes straight to ffmpeg — nothing is saved to disk.
const IG_API =
  process.env.INSTAGRAM_API || "https://api.zoraahub.com/fetch.php";
// A browser-like User-Agent; the media host rejects some default clients.
const HTTP_UA =
  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 " +
  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36";
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
  node transcribe.js <file-or-folder-or-instagram-url> [language]

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

function isInstagramUrl(s) {
  return /^https?:\/\/(www\.)?instagram\.com\//i.test(s);
}

// Best-effort shortcode for the output filename, e.g. .../reel/ABC123/ -> ABC123.
function instagramShortcode(url) {
  const m = url.match(
    /instagram\.com\/(?:[^/]+\/)?(?:reel|reels|p|tv)\/([A-Za-z0-9._-]+)/i
  );
  return m ? m[1] : "instagram-video";
}

// Ask downreels.com's backend to resolve a public IG URL to a direct MP4 link.
async function resolveInstagram(pageUrl) {
  let res;
  try {
    res = await fetch(IG_API, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Origin: "https://downreels.com",
        Referer: "https://downreels.com/",
        "User-Agent": HTTP_UA,
      },
      body: JSON.stringify({ url: pageUrl }),
    });
  } catch (e) {
    throw new Error(`could not reach the downloader service: ${e.message}`);
  }

  let data;
  try {
    data = await res.json();
  } catch {
    throw new Error(`downloader returned a non-JSON response (HTTP ${res.status}).`);
  }

  if (data.status !== "ok" || !Array.isArray(data.videos) || data.videos.length === 0)
    throw new Error(
      data.message ||
        "no downloadable video found — the post may be private, deleted, or image-only."
    );

  const video = data.videos.find((v) => v && v.url && v.isVideo !== false) || data.videos[0];
  if (!video || !video.url)
    throw new Error("no video track in the post (it may be image-only).");
  return video.url;
}

// Download a URL fully into memory. Videos are small enough (reels are seconds
// to a couple minutes) that buffering is fine and keeps the MP4 off disk.
async function downloadToBuffer(url) {
  let res;
  try {
    res = await fetch(url, { headers: { "User-Agent": HTTP_UA } });
  } catch (e) {
    throw new Error(`failed to download the video: ${e.message}`);
  }
  if (!res.ok) throw new Error(`failed to download the video (HTTP ${res.status}).`);
  const buf = Buffer.from(await res.arrayBuffer());
  if (buf.length === 0) throw new Error("the downloaded video was empty.");
  return buf;
}

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

async function runInstagram(pageUrl, language, workDir) {
  log(`Resolving Instagram video via downreels.com ...`);
  const videoUrl = await resolveInstagram(pageUrl);

  log(`Downloading video into memory ...`);
  const buf = await downloadToBuffer(videoUrl);

  log(
    `Downloaded ${(buf.length / 1024 / 1024).toFixed(1)} MB. ` +
      `Transcribing with ${path.basename(MODEL)} (Metal GPU) ...`
  );
  const transcript = await transcribeBuffer(buf, language, workDir);
  if (!transcript) die("no speech detected in the video.");

  // stdout = clean transcript (pipe-friendly).
  process.stdout.write(transcript + "\n");

  // Save to output/<shortcode>.txt.
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  const outFile = path.join(OUTPUT_DIR, `${instagramShortcode(pageUrl)}.txt`);
  fs.writeFileSync(outFile, transcript + "\n");
  log(`\nSaved to ${path.relative(HERE, outFile)}`);

  // Copy to clipboard.
  const pb = spawnSync("pbcopy", { input: transcript });
  if (pb.status === 0) log("Copied to clipboard.");
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

  const rawInput = args.input;
  const igMode = isInstagramUrl(rawInput);

  // A non-Instagram http(s) URL isn't a local path and isn't supported.
  if (!igMode && /^https?:\/\//i.test(rawInput))
    die(`only Instagram URLs are supported for download; got: ${rawInput}`);

  if (!fs.existsSync(MODEL))
    die(`model not found: ${MODEL}\nDownload it (see README) or set TRANSCRIBE_MODEL in .env.`);

  requireBinary("ffmpeg", "Install it: brew install ffmpeg");
  requireBinary("whisper-cli", "Install it: brew install whisper-cpp");

  const workDir = fs.mkdtempSync(path.join(os.tmpdir(), "transcriber-"));

  try {
    if (igMode) {
      await runInstagram(rawInput, args.language, workDir);
    } else {
      const input = path.resolve(rawInput);
      if (!fs.existsSync(input)) die(`path not found: ${input}`);
      const isDir = fs.statSync(input).isDirectory();
      if (isDir) await runFolder(input, args.language, workDir);
      else await runFile(input, args.language, workDir);
    }
  } catch (err) {
    die(err?.message || String(err));
  } finally {
    fs.rmSync(workDir, { recursive: true, force: true });
  }
}

main();
