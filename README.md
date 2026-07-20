# transcriber

Local, **fully offline** terminal tool that transcribes any audio **or video** file
using a local whisper.cpp model — Metal-accelerated on Apple Silicon. No API key,
no cost, nothing leaves your machine. The transcript is printed, copied to your
clipboard, and saved to `output/`.

For video files, only the audio track is used (the video is ignored).

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

## Performance

On an M3, `large-v3-turbo` runs several times faster than realtime — expect a
1-hour recording to take a few minutes. The whole machine won't lag; whisper uses
the GPU and a few CPU threads.

## License

[MIT](LICENSE)
