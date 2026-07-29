#!/usr/bin/env node
// Telegram transcription bot — the hosted counterpart to transcribe.js.
//
// transcribe.js  = local CLI, offline, whisper.cpp, free.
// bot.js         = deployed on Render, transcribes via the OpenAI API.
//
// Send the bot an audio or video file (voice note, audio, video, video note, or
// a document with an audio/video mime type) and it replies with the transcript.
// Anything else — text, photos, stickers, strangers — is ignored in silence.
//
// Local run:  node bot.js            (long polling; no public URL needed)
// On Render:  node bot.js            (webhook; RENDER_EXTERNAL_URL is set for us)

import "dotenv/config";
import http from "node:http";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawn } from "node:child_process";
import OpenAI, { toFile } from "openai";
import ffmpegPath from "ffmpeg-static";
import { MEDIA_EXTS, extOf, needsTranscode } from "./formats.js";

// --- config -----------------------------------------------------------------

const TOKEN = requireEnv("TELEGRAM_BOT_TOKEN");
const OPENAI_API_KEY = requireEnv("OPENAI_API_KEY");

// Only these numeric Telegram user IDs may talk to the bot. Everyone else is
// ignored without a reply, so the bot never confirms it exists to a stranger.
const ALLOWED_IDS = new Set(
  (process.env.TELEGRAM_ALLOWED_IDS || "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean)
);
if (ALLOWED_IDS.size === 0)
  die("TELEGRAM_ALLOWED_IDS is empty — set it to your numeric Telegram user ID.");

// gpt-4o-transcribe is the current best-accuracy model; whisper-1 is the
// cheaper classic. Either works — both take the same multipart upload.
const MODEL = process.env.OPENAI_TRANSCRIBE_MODEL || "gpt-4o-transcribe";
// "auto" lets the model detect the language; or force one with an ISO-639-1 code.
const LANGUAGE = (process.env.TRANSCRIBE_LANGUAGE || "auto").toLowerCase();

const PORT = Number(process.env.PORT) || 3000;
// Render injects RENDER_EXTERNAL_URL (e.g. https://my-bot.onrender.com).
const PUBLIC_URL = (process.env.PUBLIC_URL || process.env.RENDER_EXTERNAL_URL || "")
  .trim()
  .replace(/\/+$/, "");
// Telegram echoes this back in a header on every webhook call, so we can reject
// anyone who guesses the URL. Falls back to a slice of the bot token.
const WEBHOOK_SECRET = process.env.WEBHOOK_SECRET || TOKEN.split(":")[1] || "secret";
const WEBHOOK_PATH = "/telegram";

const API = `https://api.telegram.org/bot${TOKEN}`;
const FILE_API = `https://api.telegram.org/file/bot${TOKEN}`;

// Telegram's Bot API refuses to serve downloads larger than 20 MB, so we can
// never hit OpenAI's own 25 MB upload ceiling.
const TELEGRAM_MAX_BYTES = 20 * 1024 * 1024;
// Telegram messages cap at 4096 characters; leave room for the chunk counter.
const MSG_LIMIT = 3900;
// Past this, a wall of chunked messages is worse than a file attachment.
const DOC_THRESHOLD = 3900 * 4;

// Which extensions count as media when a file arrives as a plain document lives
// in formats.js, shared with the CLI.

// ffmpeg-static ships a prebuilt binary, so this works on Render too — no
// system package needed. FFMPEG_PATH overrides it if you'd rather use your own.
const FFMPEG = process.env.FFMPEG_PATH || ffmpegPath;

const openai = new OpenAI({ apiKey: OPENAI_API_KEY });

// --- tiny helpers -----------------------------------------------------------

const log = (...a) => console.log(new Date().toISOString(), ...a);

function requireEnv(name) {
  const v = process.env[name];
  if (!v) die(`${name} is not set. Copy .env.example to .env and fill it in.`);
  return v;
}

function die(msg) {
  console.error(`\nError: ${msg}\n`);
  process.exit(1);
}

// --- Telegram API -----------------------------------------------------------

async function tg(method, params) {
  const res = await fetch(`${API}/${method}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  const data = await res.json().catch(() => ({}));
  if (!data.ok)
    throw new Error(`Telegram ${method} failed: ${data.description || res.status}`);
  return data.result;
}

// Same as tg(), but never throws — for cosmetic calls we don't want to fail on.
async function tgQuiet(method, params) {
  try {
    return await tg(method, params);
  } catch (err) {
    log(`(ignored) ${err.message}`);
    return null;
  }
}

async function sendText(chatId, text, replyTo) {
  return tg("sendMessage", {
    chat_id: chatId,
    text,
    reply_to_message_id: replyTo,
    // The transcript is user content — don't let it be parsed as markup.
    disable_notification: false,
  });
}

// Upload the transcript as a .txt attachment (used when it's too long to chat).
async function sendDocument(chatId, filename, text, replyTo) {
  const form = new FormData();
  form.append("chat_id", String(chatId));
  if (replyTo) form.append("reply_to_message_id", String(replyTo));
  form.append("document", new Blob([text], { type: "text/plain" }), filename);
  const res = await fetch(`${API}/sendDocument`, { method: "POST", body: form });
  const data = await res.json().catch(() => ({}));
  if (!data.ok)
    throw new Error(`Telegram sendDocument failed: ${data.description || res.status}`);
  return data.result;
}

// Download a Telegram file into memory. Files are <= 20 MB by Bot API rule.
async function downloadTelegramFile(fileId) {
  const info = await tg("getFile", { file_id: fileId });
  const res = await fetch(`${FILE_API}/${info.file_path}`);
  if (!res.ok) throw new Error(`file download failed (HTTP ${res.status})`);
  const buf = Buffer.from(await res.arrayBuffer());
  if (buf.length === 0) throw new Error("the downloaded file was empty");
  return { buf, path: info.file_path };
}

// --- message classification -------------------------------------------------

// Pull the media out of a message, or return null if there is none. Returning
// null is the signal to stay silent — "anything random" gets no reply.
function extractMedia(msg) {
  if (msg.voice)
    return { fileId: msg.voice.file_id, size: msg.voice.file_size, name: "voice.ogg" };
  if (msg.audio)
    return {
      fileId: msg.audio.file_id,
      size: msg.audio.file_size,
      name: msg.audio.file_name || "audio.mp3",
    };
  if (msg.video)
    return {
      fileId: msg.video.file_id,
      size: msg.video.file_size,
      name: msg.video.file_name || "video.mp4",
    };
  if (msg.video_note)
    return {
      fileId: msg.video_note.file_id,
      size: msg.video_note.file_size,
      name: "video_note.mp4",
    };

  // A file sent "as a file" rather than as media arrives as a document.
  if (msg.document) {
    const d = msg.document;
    const mime = (d.mime_type || "").toLowerCase();
    const name = d.file_name || "file";
    const looksLikeMedia =
      mime.startsWith("audio/") ||
      mime.startsWith("video/") ||
      MEDIA_EXTS.has(extOf(name));
    if (looksLikeMedia)
      return { fileId: d.file_id, size: d.file_size, name };
  }

  return null;
}

// A per-message language override: send the file with caption "-en" or "-lv".
function languageFor(msg) {
  const caption = (msg.caption || "").trim().toLowerCase();
  const m = caption.match(/(?:^|\s)-([a-z]{2})(?:\s|$)/);
  if (m) return m[1];
  return LANGUAGE;
}

// --- format normalisation ---------------------------------------------------

// The OpenAI endpoint accepts only ten containers (see OPENAI_EXTS), and picks
// its decoder from the filename extension. Plenty of everyday files fall
// outside that set — .opus and .amr voice notes, .aiff and .caf from Apple
// gear, .wma from Windows, .mkv/.mov/.avi screen and camera recordings — so we
// run those through ffmpeg first.
//
// Output is 16 kHz mono MP3: speech loses nothing at that rate, and it shrinks
// a 20 MB video down to a couple of MB, which keeps us clear of the API's own
// 25 MB upload ceiling.
function transcodeToMp3(buf, ext) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "tg-transcode-"));
  // Keep the original extension on the temp file — some demuxers (and ffmpeg's
  // probe) do better with the hint. A temp *file* rather than a pipe matters
  // for .mov/.mp4 whose index sits at the end and needs seeking.
  const src = path.join(dir, `input${ext || ""}`);
  const dst = path.join(dir, "audio.mp3");

  return new Promise((resolve, reject) => {
    fs.writeFileSync(src, buf);
    const p = spawn(FFMPEG, [
      "-hide_banner", "-loglevel", "error", "-y",
      "-i", src,
      "-vn",            // drop any video track
      "-ac", "1",       // mono
      "-ar", "16000",   // 16 kHz — speech is band-limited anyway
      "-b:a", "64k",
      dst,
    ]);
    let err = "";
    p.stderr.on("data", (d) => (err += d));
    p.on("error", reject);
    p.on("close", (code) => {
      if (code !== 0)
        return reject(new Error(err.trim().split("\n").pop() || `ffmpeg exited ${code}`));
      resolve(fs.readFileSync(dst));
    });
  }).finally(() => fs.rmSync(dir, { recursive: true, force: true }));
}

// Return { buf, filename } ready to upload, converting when we have to.
async function prepareForUpload(buf, filename) {
  if (!needsTranscode(filename)) return { buf, filename };
  const ext = extOf(filename);
  log(`converting ${ext || "unknown format"} -> mp3 (${buf.length} bytes)`);
  const out = await transcodeToMp3(buf, ext);
  if (out.length === 0) throw new Error("the converted audio was empty");
  return { buf: out, filename: "audio.mp3" };
}

// --- transcription ----------------------------------------------------------

async function transcribe(buf, filename, language) {
  const file = await toFile(buf, filename);
  const params = { file, model: MODEL, response_format: "text" };
  // Omitting `language` entirely is what triggers auto-detection.
  if (language && language !== "auto") params.language = language;

  const result = await openai.audio.transcriptions.create(params);
  // With response_format "text" the SDK hands back a plain string.
  const text = typeof result === "string" ? result : result?.text || "";
  return text.replace(/\n{3,}/g, "\n\n").trim();
}

// Split a long transcript on paragraph/sentence boundaries where possible.
function chunk(text, limit = MSG_LIMIT) {
  const parts = [];
  let rest = text;
  while (rest.length > limit) {
    let cut = rest.lastIndexOf("\n\n", limit);
    if (cut < limit * 0.5) cut = rest.lastIndexOf(". ", limit);
    if (cut < limit * 0.5) cut = rest.lastIndexOf(" ", limit);
    if (cut < limit * 0.5) cut = limit;
    parts.push(rest.slice(0, cut).trim());
    rest = rest.slice(cut).trim();
  }
  if (rest) parts.push(rest);
  return parts;
}

// --- update handling --------------------------------------------------------

// Telegram redelivers an update if we don't ack fast enough. We ack immediately
// and process in the background, so guard against handling one twice.
const seenUpdates = new Set();
function alreadySeen(updateId) {
  if (seenUpdates.has(updateId)) return true;
  seenUpdates.add(updateId);
  if (seenUpdates.size > 1000) seenUpdates.delete(seenUpdates.values().next().value);
  return false;
}

async function handleUpdate(update) {
  const msg = update.message || update.channel_post;
  if (!msg) return;

  // Whitelist check first: unknown senders get nothing at all.
  const from = msg.from?.id;
  if (!from || !ALLOWED_IDS.has(String(from))) {
    log(`ignored message from unauthorized id=${from ?? "unknown"}`);
    return;
  }

  const media = extractMedia(msg);
  if (!media) return; // random text/photo/sticker — stay silent

  const chatId = msg.chat.id;

  if (media.size && media.size > TELEGRAM_MAX_BYTES) {
    await sendText(
      chatId,
      `That file is ${(media.size / 1024 / 1024).toFixed(1)} MB. Telegram only lets ` +
        `bots download files up to 20 MB — please trim or compress it first.`,
      msg.message_id
    );
    return;
  }

  const status = await tgQuiet("sendMessage", {
    chat_id: chatId,
    text: "🎧 Transcribing…",
    reply_to_message_id: msg.message_id,
  });

  try {
    log(`transcribing ${media.name} (${media.size ?? "?"} bytes) for ${from}`);
    const { buf, path: tgPath } = await downloadTelegramFile(media.fileId);
    // Trust Telegram's stored extension over the client-supplied filename —
    // it reflects what Telegram actually holds, and the decoder is chosen from
    // the extension (both by ffmpeg's probe and by the OpenAI endpoint).
    const ext = extOf(tgPath);
    const named = ext && extOf(media.name) !== ext ? `audio${ext}` : media.name;

    const ready = await prepareForUpload(buf, named);
    const transcript = await transcribe(ready.buf, ready.filename, languageFor(msg));

    if (!transcript) {
      await editOrSend(chatId, status, "No speech detected in that file.", msg.message_id);
      return;
    }

    if (transcript.length > DOC_THRESHOLD) {
      const base = media.name.replace(/\.[^.]+$/, "") || "transcript";
      await sendDocument(chatId, `${base}.txt`, transcript, msg.message_id);
      await editOrSend(
        chatId,
        status,
        `✅ Transcript is ${transcript.length} characters — sent as a file.`,
        msg.message_id
      );
      return;
    }

    const parts = chunk(transcript);
    await editOrSend(chatId, status, parts[0], msg.message_id);
    for (const part of parts.slice(1)) await sendText(chatId, part);
    log(`done: ${transcript.length} chars in ${parts.length} message(s)`);
  } catch (err) {
    log(`failed: ${err?.message || err}`);
    await editOrSend(
      chatId,
      status,
      `Transcription failed: ${err?.message || "unknown error"}`,
      msg.message_id
    );
  }
}

// Replace the "Transcribing…" placeholder with the result, or send fresh if the
// placeholder never made it out.
async function editOrSend(chatId, status, text, replyTo) {
  if (status?.message_id) {
    const edited = await tgQuiet("editMessageText", {
      chat_id: chatId,
      message_id: status.message_id,
      text,
    });
    if (edited) return;
  }
  await sendText(chatId, text, replyTo);
}

// --- webhook server (Render) ------------------------------------------------

// This host only exists to receive Telegram webhooks, so "/" is just a note
// explaining that to whoever opens the URL directly.
const LANDING_HTML = `<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Transcriber bot</title>
<body style="max-width:34rem;margin:4rem auto;padding:0 1.25rem;font:16px/1.6 system-ui,sans-serif">
<h1 style="font-size:1.5rem">Transcriber bot</h1>
<p>This is the webhook server for a Telegram bot that turns speech into text.
Send it a voice note, audio file, video, video note, or any audio/video file
sent as a document, and it replies with a transcript. Files that the
transcription API can't read directly are converted with ffmpeg first, so
formats like Opus, AMR, CAF, WMA, MOV and MKV all work. Telegram caps bot
downloads at 20 MB. Add a caption like <code>-en</code> or <code>-lv</code> to
force a language instead of letting it auto-detect; long transcripts come back
as a <code>.txt</code> file rather than a message.</p>
<p>There is nothing to use on this page — the bot lives in Telegram.</p>
</body>`;

function startServer() {
  const server = http.createServer((req, res) => {
    if (req.method === "GET") {
      // "/" explains what this thing is, for anyone who lands on the Render URL.
      // Every other GET stays a bare "ok" — that's the health check's job.
      if (req.url.split("?")[0] === "/") {
        res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
        res.end(LANDING_HTML);
        return;
      }
      res.writeHead(200, { "Content-Type": "text/plain" });
      res.end("ok");
      return;
    }

    if (req.method !== "POST" || !req.url.startsWith(WEBHOOK_PATH)) {
      res.writeHead(404).end();
      return;
    }

    if (req.headers["x-telegram-bot-api-secret-token"] !== WEBHOOK_SECRET) {
      log("rejected webhook call with a bad secret token");
      res.writeHead(401).end();
      return;
    }

    let body = "";
    req.on("data", (d) => {
      body += d;
      if (body.length > 2_000_000) req.destroy(); // no legitimate update is this big
    });
    req.on("end", () => {
      // Ack immediately — transcription takes far longer than Telegram will wait.
      res.writeHead(200).end();
      let update;
      try {
        update = JSON.parse(body);
      } catch {
        return;
      }
      if (alreadySeen(update.update_id)) return;
      handleUpdate(update).catch((err) => log(`unhandled: ${err?.message || err}`));
    });
  });

  server.listen(PORT, () => log(`listening on :${PORT}`));
}

async function useWebhook() {
  const url = `${PUBLIC_URL}${WEBHOOK_PATH}`;
  await tg("setWebhook", {
    url,
    secret_token: WEBHOOK_SECRET,
    allowed_updates: ["message"],
    drop_pending_updates: true,
    max_connections: 10,
  });
  log(`webhook set to ${url}`);
}

// --- long polling (local dev) -----------------------------------------------

async function usePolling() {
  await tgQuiet("deleteWebhook", { drop_pending_updates: true });
  log("no PUBLIC_URL — running in long-polling mode");
  let offset = 0;
  for (;;) {
    try {
      const updates = await tg("getUpdates", {
        offset,
        timeout: 50,
        allowed_updates: ["message"],
      });
      for (const update of updates) {
        offset = update.update_id + 1;
        if (alreadySeen(update.update_id)) continue;
        handleUpdate(update).catch((err) => log(`unhandled: ${err?.message || err}`));
      }
    } catch (err) {
      log(`poll error: ${err?.message || err}`);
      await new Promise((r) => setTimeout(r, 3000));
    }
  }
}

// --- main -------------------------------------------------------------------

async function main() {
  const me = await tg("getMe", {});
  log(`starting @${me.username} — model ${MODEL}, ${ALLOWED_IDS.size} allowed user(s)`);

  // Not fatal: the ten formats OpenAI takes directly still work without it.
  if (!FFMPEG || !fs.existsSync(FFMPEG))
    log("warning: no ffmpeg binary found — formats needing conversion will fail");

  if (PUBLIC_URL) {
    startServer();
    await useWebhook();
  } else {
    // Still bind a port if one was handed to us, so a PaaS health check passes.
    if (process.env.PORT) startServer();
    await usePolling();
  }
}

main().catch((err) => die(err?.message || String(err)));
