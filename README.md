# transcriber

Audio and video → text. Two independent front-ends in one repo:

| | `transcribe.js` — **local CLI** | `bot.js` — **Telegram bot** |
|---|---|---|
| Engine | whisper.cpp on your GPU | OpenAI transcription API |
| Runs on | your Mac, fully offline | Render (or anywhere with Node) |
| Cost | free | per-minute OpenAI billing |
| Needs | `ffmpeg` + `whisper-cpp` + a 1.5GB model | an API key, nothing installed |
| Input | file, folder, or Instagram URL | a file sent in a Telegram chat |

They share nothing but this repo and `.env` — pick whichever fits. Everything
down to [Telegram bot](#telegram-bot) covers the local CLI.

For video files, only the audio track is used (the video is ignored).

---

# Local CLI

Transcribes any audio **or video** file — or a public **Instagram** post/reel —
using a local whisper.cpp model, Metal-accelerated on Apple Silicon.
Transcription runs offline with no API key or cost; the transcript is printed,
copied to your clipboard, and saved to `output/`.

## How it works

1. `ffmpeg` extracts the audio to 16kHz mono WAV.
2. `whisper-cli` (whisper.cpp) transcribes it locally on your GPU.
3. The transcript is printed, copied to your clipboard, and saved to `output/<name>.txt`.

## Setup

Requires Node 20+, `ffmpeg`, and `whisper-cpp` (all installed):

```bash
brew install ffmpeg whisper-cpp
npm install
```

Download the model (once, ~1.5GB) into `models/`:

```bash
mkdir -p models
curl -L -o models/ggml-large-v3-turbo.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin
```

> **Model choice:** `large-v3-turbo` gives large-v3-level transcription accuracy
> while fitting comfortably in 8GB RAM. To use a different model, drop its `.bin`
> in `models/` and set `TRANSCRIBE_MODEL` in `.env`.

## Usage

```bash
node transcribe.js path/to/file.mp4
```

Language flags (default is auto-detect):

```
-en          Force English
-lv          Force Latvian
-h, --help   Show help
```

Examples:

```bash
node transcribe.js interview.m4a          # auto-detect language
node transcribe.js lecture.mov -en        # force English
node transcribe.js saruna.mp3 -lv         # force Latvian
```

Every run prints the transcript, copies it to your clipboard, and saves it to
`output/<input-name>.txt` (overwriting a previous run of the same file). The
transcript goes to stdout and progress goes to stderr, so piping stays clean:

```bash
node transcribe.js meeting.mp4 > meeting.txt
```

### Transcribe an Instagram post or reel

Pass an Instagram URL instead of a file and the public video is downloaded **into
memory** via [downreels.com](https://downreels.com/en1/), then transcribed locally:

```bash
node transcribe.js https://www.instagram.com/reel/XXXXXXXXXXX/ -en
```

The video is streamed straight into `ffmpeg` — the MP4 is never written to disk.
The transcript is printed, copied to your clipboard, and saved to
`output/<shortcode>.txt` (e.g. `output/Cvuh19eNWv_.txt`). Only **public** posts
work; private, deleted, or image-only posts return an error.

> Unlike file/folder mode, this step fetches the video over the network (the
> downloader resolves Instagram's CDN link for you). The transcription itself
> still runs fully offline. The resolver endpoint can be overridden with
> `INSTAGRAM_API` in `.env`.

### Transcribe a whole folder

Point it at a **folder** and every audio/video file inside is transcribed. The
transcripts are written to a **new sibling folder** named
`<folder>-transcripts`, one `.txt` per input file:

```bash
node transcribe.js ~/recordings -en
```

```
~/recordings/                 ~/recordings-transcripts/
  ep01.mp3          ───▶        ep01.txt
  ep02.m4a          ───▶        ep02.txt
  clip.mp4          ───▶        clip.txt
```

Files are processed one at a time (progress is printed to stderr). A file that
fails or has no speech is skipped without stopping the rest, and a summary of how
many succeeded/failed is printed at the end. In folder mode nothing is copied to
the clipboard and `output/` is not used — everything goes to the new folder.
Only common audio/video extensions are picked up (`.mp3 .m4a .wav .flac .ogg
.opus .aac .mp4 .mov .mkv .webm .avi …`); other files in the folder are ignored.

## Config (all optional, via `.env`)

- `TRANSCRIBE_MODEL` — path to a whisper.cpp `.bin` (default: `models/ggml-large-v3-turbo.bin`)
- `TRANSCRIBE_THREADS` — default `4`
- `INSTAGRAM_API` — endpoint that resolves an Instagram URL to a direct MP4
  (default: downreels.com's backend). Only needed if that service changes.

## Performance

On an M3, `large-v3-turbo` runs several times faster than realtime — expect a
1-hour recording to take a few minutes. The whole machine won't lag; whisper uses
the GPU and a few CPU threads.

---

# Telegram bot

Send the bot an audio or video file; it replies with the transcript. That's the
whole interface.

- **Whitelisted.** Only the numeric Telegram user IDs in `TELEGRAM_ALLOWED_IDS`
  get a response. Anyone else is ignored in complete silence — the bot never
  replies, so it never confirms to a stranger that it's alive.
- **Silent on anything else.** Text, photos, stickers, links, commands: no reply.
  Only voice notes, audio, video, video notes, and documents with an audio/video
  type are acted on.
- **No local dependencies.** No ffmpeg, no whisper.cpp, no model download — the
  file goes straight from Telegram to the OpenAI API.

Long transcripts are split across messages; very long ones arrive as a `.txt`
attachment. Caption a file with `-en` or `-lv` to force that language for it
(default is auto-detect).

> **Size limit:** Telegram's Bot API refuses to hand a bot any file over
> **20 MB**, so that's the ceiling. Larger files get a short explanatory reply.
> An hour of voice note is well under it; an hour of 1080p video is not — run
> those through the local CLI instead.

## Setup

1. **Create the bot.** Message [@BotFather](https://t.me/BotFather), send
   `/newbot`, and copy the token.
2. **Get your user ID.** Message [@userinfobot](https://t.me/userinfobot); it
   replies with your numeric ID.
3. **Get an OpenAI key** at <https://platform.openai.com/api-keys>.
4. Copy `.env.example` to `.env` and fill in:

```bash
TELEGRAM_BOT_TOKEN=123456:ABC...
OPENAI_API_KEY=sk-...
TELEGRAM_ALLOWED_IDS=123456789
```

Run it locally (no public URL needed — it falls back to long polling):

```bash
npm install
npm run bot
```

Then send it a voice note.

## Deploy to Render

`render.yaml` in this repo is a ready blueprint.

1. Push the repo to GitHub.
2. In Render: **New → Blueprint**, point it at the repo.
3. Fill in the three secrets it prompts for: `TELEGRAM_BOT_TOKEN`,
   `OPENAI_API_KEY`, `TELEGRAM_ALLOWED_IDS`. `WEBHOOK_SECRET` is generated for
   you.
4. Deploy. On boot the bot registers its own Telegram webhook using Render's
   `RENDER_EXTERNAL_URL` — there is no manual `setWebhook` step.

Prefer clicking through the dashboard? **New → Web Service**, runtime Node, build
`npm ci --omit=dev`, start `node bot.js`, then add the same env vars.

> **Free tier caveat:** Render spins a free service down after ~15 minutes idle.
> The first message after that wakes it, and Telegram may need a retry or two
> before the reply lands. A paid instance (or an external uptime pinger hitting
> the service's `/` health URL) avoids the cold start.

## Config (via `.env` / Render env vars)

| Var | Required | Notes |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | yes | from @BotFather |
| `OPENAI_API_KEY` | yes | |
| `TELEGRAM_ALLOWED_IDS` | yes | comma-separated numeric IDs; the bot refuses to start if empty |
| `OPENAI_TRANSCRIBE_MODEL` | no | default `gpt-4o-transcribe`; `whisper-1` is cheaper |
| `TRANSCRIBE_LANGUAGE` | no | ISO-639-1 code, default auto-detect |
| `WEBHOOK_SECRET` | no | shared secret Telegram echoes back; set it in production |
| `PUBLIC_URL` | no | webhook base URL; Render supplies `RENDER_EXTERNAL_URL` automatically. Blank ⇒ long polling |
| `PORT` | no | default `3000`; Render sets it |

## License

[MIT](LICENSE)
