#!/usr/bin/env node
// Batch-transcribe every reel in an Apify instagram-scraper dataset export.
//
// Reads the scraper's JSON array, keeps the video posts, and transcribes each
// one with the same local whisper.cpp pipeline transcribe.js uses. Every reel
// gets its own .txt named for its position in the dataset: 1.txt ... 100.txt.
// The folder they land in is named for the profile and the scrape it came from
// (patrickwithprospectflo_2026-08-02_12-39-47-067), so scrapes of different
// accounts — or the same account on different days — never collide.
//
// Each item carries a direct CDN `videoUrl` (fast, but the signed link expires
// a few days after the scrape) and a permalink `url`. We try the CDN link first
// and fall back to resolving the permalink through downreels.com, so an aged
// dataset still works.
//
// Usage:
//   node batch-reels.js <dataset.json> [-o <dir>] [-en|-lv] [--limit N] [--force]

import "dotenv/config";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { spawn, spawnSync } from "node:child_process";
import {
  instagramShortcode,
  downloadToBuffer,
  fetchInstagramVideo,
} from "./instagram.js";

// --- config -----------------------------------------------------------------

const HERE = path.dirname(fileURLToPath(import.meta.url));
const MODEL =
  process.env.TRANSCRIBE_MODEL ||
  path.join(HERE, "models", "ggml-large-v3-turbo.bin");
const THREADS = process.env.TRANSCRIBE_THREADS || "4";

const log = (...a) => process.stderr.write(a.join(" ") + "\n");
const mb = (buf) => (buf.length / 1024 / 1024).toFixed(1);
const die = (msg) => {
  process.stderr.write(`\nError: ${msg}\n`);
  process.exit(1);
};

// --- arg parsing ------------------------------------------------------------

function parseArgs(argv) {
  const args = {
    input: null,
    language: "auto",
    outDir: null,
    limit: 0,
    force: false,
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "-en") args.language = "en";
    else if (a === "-lv") args.language = "lv";
    else if (a === "--force") args.force = true;
    else if (a === "--help" || a === "-h") args.help = true;
    else if (a === "-o" || a === "--out") {
      const next = argv[++i];
      if (!next) die(`${a} needs a directory path`);
      args.outDir = path.resolve(next);
    } else if (a === "--limit") {
      const next = Number(argv[++i]);
      if (!Number.isInteger(next) || next < 1) die("--limit needs a positive integer");
      args.limit = next;
    } else if (!a.startsWith("-") && !args.input) args.input = a;
    else die(`unknown argument: ${a}`);
  }
  return args;
}

function printHelp() {
  log(`batch-reels — transcribe every reel in an Instagram scraper export

Usage:
  node batch-reels.js <dataset.json> [options]

Options:
  -o, --out <dir>   Output folder (default: output/<profile>_<scrape-stamp>)
  -en | -lv         Force English / Latvian (default: auto-detect)
  --limit N         Only process the first N reels
  --force           Re-transcribe reels whose .txt already exists
  -h, --help        Show this help

One .txt per reel, named for its position in the dataset: 1.txt ... 100.txt.
index.tsv in the same folder maps each number back to its post URL. Reruns
skip files that already exist, so an interrupted run just resumes.`);
}

// --- pipeline (same as transcribe.js: ffmpeg -> 16kHz mono WAV -> whisper) ---

function requireBinary(name, hint) {
  const r = spawnSync(name, ["--help"], { stdio: "ignore" });
  if (r.error && r.error.code === "ENOENT")
    die(`${name} not found on PATH. ${hint || ""}`);
}

// Feed the in-memory MP4 to ffmpeg over stdin; the video never touches disk.
function extractAudioFromBuffer(buf, outWav) {
  return new Promise((resolve, reject) => {
    const p = spawn("ffmpeg", [
      "-hide_banner", "-loglevel", "error", "-y",
      "-i", "pipe:0",
      "-vn",
      "-ac", "1",
      "-ar", "16000",
      "-c:a", "pcm_s16le",
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

function whisper(wav, language) {
  return new Promise((resolve, reject) => {
    const p = spawn("whisper-cli", [
      "-m", MODEL,
      "-f", wav,
      "-nt",
      "-l", language,
      "-t", THREADS,
    ]);
    let out = "";
    let err = "";
    p.stdout.on("data", (d) => (out += d));
    p.stderr.on("data", (d) => (err += d));
    p.on("error", reject);
    p.on("close", (code) =>
      code === 0 ? resolve(out) : reject(new Error(err.trim() || `whisper-cli exited ${code}`))
    );
  });
}

async function transcribeBuffer(buf, language, workDir) {
  const wav = path.join(workDir, "audio.wav");
  try {
    await extractAudioFromBuffer(buf, wav);
    const raw = await whisper(wav, language);
    return raw.replace(/\n{3,}/g, "\n\n").trim();
  } finally {
    fs.rmSync(wav, { force: true });
  }
}

// --- dataset ----------------------------------------------------------------

// Pull the reels out of the scraper export, keeping dataset order — that order
// is what the numbering encodes, so it has to stay stable across reruns.
function readReels(file) {
  let data;
  try {
    data = JSON.parse(fs.readFileSync(file, "utf8"));
  } catch (e) {
    die(`could not read ${file}: ${e.message}`);
  }
  // Apify exports a bare array; some tools wrap it in { items: [...] }.
  const items = Array.isArray(data) ? data : data?.items;
  if (!Array.isArray(items)) die(`${file} is not a scraper dataset (expected a JSON array).`);

  const reels = [];
  for (const it of items) {
    if (!it || typeof it !== "object") continue;
    const pageUrl = it.url;
    // Videos only — image posts and sidecars have nothing to transcribe.
    const isVideo = it.type === "Video" || it.productType === "clips" || !!it.videoUrl;
    if (!isVideo || !pageUrl) continue;
    reels.push({
      pageUrl,
      videoUrl: it.videoUrl || null,
      shortCode: it.shortCode || instagramShortcode(pageUrl),
      owner: it.ownerUsername || "",
      caption: it.caption || "",
      timestamp: it.timestamp || "",
      duration: it.videoDuration || 0,
    });
  }
  return reels;
}

// Name the output folder after the profile and the scrape it came from, e.g.
// "patrickwithprospectflo_2026-08-02_12-39-47-067". Derived rather than random
// so a rerun lands in the same folder and can skip what's already transcribed;
// a fresh scrape carries a new timestamp and so gets a fresh folder.
function outputFolderName(datasetFile, reels) {
  const base = path.basename(datasetFile, path.extname(datasetFile));
  // Apify names exports "dataset_<actor>_<date>_<time>"; the trailing stamp is
  // the part that actually distinguishes one scrape from the next.
  const stamp = base.match(/\d{4}-\d{2}-\d{2}[_-][\d-]+/)?.[0] || base;

  // A profile scrape is single-owner; a hashtag/search scrape isn't.
  const owners = new Set(reels.map((r) => r.owner).filter(Boolean));
  const who =
    owners.size === 1 ? [...owners][0] : owners.size > 1 ? "mixed" : "instagram";

  return `${who}_${stamp}`.replace(/[^A-Za-z0-9._-]/g, "-");
}

// The signed CDN link is fastest but expires; the permalink always resolves.
// Reports which route produced the buffer, because the CDN one has a second
// failure mode that only shows up later (see NO_AUDIO).
async function fetchVideo(reel) {
  if (reel.videoUrl) {
    try {
      return { buf: await downloadToBuffer(reel.videoUrl), viaCdn: true };
    } catch (e) {
      log(`      CDN link stale (${e.message}); resolving permalink ...`);
    }
  }
  return { buf: await fetchInstagramVideo(reel.pageUrl), viaCdn: false };
}

// Some dataset `videoUrl`s are a DASH video-only rendition — VP9, no audio
// track at all — so ffmpeg has nothing to write and bails with this. The
// permalink resolver hands back the muxed H.264+AAC file instead.
const NO_AUDIO = /does not contain any stream/i;

// --- main -------------------------------------------------------------------

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help || !args.input) {
    printHelp();
    process.exit(args.input ? 0 : 1);
  }

  const datasetFile = path.resolve(args.input);
  if (!fs.existsSync(datasetFile)) die(`file not found: ${datasetFile}`);
  if (!fs.existsSync(MODEL))
    die(`model not found: ${MODEL}\nDownload it (see README) or set TRANSCRIBE_MODEL in .env.`);

  requireBinary("ffmpeg", "Install it: brew install ffmpeg");
  requireBinary("whisper-cli", "Install it: brew install whisper-cpp");

  let reels = readReels(datasetFile);
  if (reels.length === 0) die("no video posts found in the dataset.");
  if (args.limit) reels = reels.slice(0, args.limit);

  const destDir =
    args.outDir || path.join(HERE, "output", outputFolderName(datasetFile, reels));
  fs.mkdirSync(destDir, { recursive: true });

  const workDir = fs.mkdtempSync(path.join(os.tmpdir(), "batch-reels-"));

  log(`${reels.length} reel(s) from ${path.basename(datasetFile)}`);
  log(`Writing transcripts to ${destDir}\n`);

  // An index.tsv maps each numbered file back to its post — otherwise the
  // number is the only link between a transcript and the reel it came from.
  const index = [];
  let ok = 0;
  let skipped = 0;
  let failed = 0;

  try {
    for (let i = 0; i < reels.length; i++) {
      const reel = reels[i];
      const num = String(i + 1);
      const name = `${num}.txt`;
      const outFile = path.join(destDir, name);
      const label = `[${i + 1}/${reels.length}] ${name}`;

      index.push([num, reel.shortCode, reel.pageUrl, reel.timestamp].join("\t"));

      if (!args.force && fs.existsSync(outFile)) {
        log(`${label} — already done, skipping.`);
        skipped++;
        continue;
      }

      try {
        log(`${label} — downloading ...`);
        const { buf, viaCdn } = await fetchVideo(reel);
        log(`${label} — transcribing ${mb(buf)} MB ...`);
        let transcript;
        try {
          transcript = await transcribeBuffer(buf, args.language, workDir);
        } catch (err) {
          if (!viaCdn || !NO_AUDIO.test(err.message || "")) throw err;
          log(`${label} — CDN copy has no audio track; resolving permalink ...`);
          const muxed = await fetchInstagramVideo(reel.pageUrl);
          log(`${label} — transcribing ${mb(muxed)} MB ...`);
          transcript = await transcribeBuffer(muxed, args.language, workDir);
        }
        if (!transcript) {
          log(`${label} — no speech detected, skipped.`);
          failed++;
          continue;
        }
        // A short header keeps each file self-describing once it's out of
        // the folder; the transcript itself starts after the blank line.
        const header = `# ${reel.pageUrl}\n# ${reel.timestamp}\n\n`;
        fs.writeFileSync(outFile, header + transcript + "\n");
        log(`${label} — saved (${transcript.length} chars)`);
        ok++;
      } catch (err) {
        log(`${label} — failed: ${err?.message || String(err)}`);
        failed++;
      }
    }
  } finally {
    fs.rmSync(workDir, { recursive: true, force: true });
  }

  fs.writeFileSync(
    path.join(destDir, "index.tsv"),
    "num\tshortcode\turl\ttimestamp\n" + index.join("\n") + "\n"
  );

  log(`\nDone. ${ok} transcribed, ${skipped} already existed, ${failed} skipped/failed.`);
  log(`Transcripts are in ${destDir}`);
}

main();
