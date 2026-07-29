// Format knowledge shared by the CLI and the Telegram bot.
//
// The two transcribers have very different appetites:
//   transcribe.js  feeds files to ffmpeg, which reads essentially everything.
//   bot.js         uploads to the OpenAI API, which accepts only ten containers.
//
// So we keep one broad "is this even media?" list, plus the narrow list OpenAI
// will actually take, and let the bot transcode the gap.

// Every audio/video extension we plausibly run into day to day. ffmpeg handles
// all of these; the list exists so we don't try to "transcribe" a .txt or .jpg
// when scanning a folder or receiving a document.
export const AUDIO_EXTS = new Set([
  // mainstream
  ".mp3", ".m4a", ".wav", ".wave", ".flac", ".aac", ".ogg", ".oga", ".opus",
  // Apple / Windows
  ".m4b", ".m4r", ".m4p", ".aiff", ".aif", ".aifc", ".caf", ".alac",
  ".wma", ".asf", ".snd", ".au",
  // messaging & telephony (WhatsApp, Telegram, voicemail, call recorders)
  ".amr", ".awb", ".3ga", ".gsm", ".spx", ".sln", ".vox",
  // broadcast / surround
  ".ac3", ".eac3", ".dts", ".mp2", ".mpga", ".mpa", ".m2a",
  // lossless & tracker-ish oddities that still show up
  ".ape", ".wv", ".tta", ".shn", ".dsf", ".dff", ".w64", ".voc", ".ra", ".rm",
  // Matroska / WebM audio-only
  ".mka", ".weba",
]);

// Video containers — we only ever use the audio track, but people send these
// constantly (screen recordings, reels, camera clips).
export const VIDEO_EXTS = new Set([
  ".mp4", ".m4v", ".mov", ".qt", ".mkv", ".webm", ".avi", ".wmv", ".flv",
  ".f4v", ".3gp", ".3g2", ".mpg", ".mpeg", ".mpe", ".m1v", ".m2v", ".ts",
  ".mts", ".m2ts", ".vob", ".ogv", ".rmvb", ".divx", ".mxf", ".dv",
]);

// Everything worth handing to ffmpeg.
export const MEDIA_EXTS = new Set([...AUDIO_EXTS, ...VIDEO_EXTS]);

// The only extensions the OpenAI transcription endpoint accepts. Anything else
// has to be transcoded first, no matter how ordinary the codec inside is.
//
// .oga is documented as supported but the endpoint rejects it in practice
// ("400 Unsupported file format oga"), and Telegram stores every Opus/Ogg
// upload — voice notes, WhatsApp .opus exports — under that extension. Leaving
// it out sends those through ffmpeg, which is where they belonged anyway.
export const OPENAI_EXTS = new Set([
  ".flac", ".m4a", ".mp3", ".mp4", ".mpeg", ".mpga", ".ogg", ".wav",
  ".webm",
]);

// Lowercased extension including the dot, or "" when there isn't one.
export function extOf(name = "") {
  const i = name.lastIndexOf(".");
  const slash = Math.max(name.lastIndexOf("/"), name.lastIndexOf("\\"));
  if (i <= slash + 1) return "";
  return name.slice(i).toLowerCase();
}

export function isMediaName(name) {
  return MEDIA_EXTS.has(extOf(name));
}

// True when we must run the bytes through ffmpeg before uploading to OpenAI.
// An unknown/absent extension counts: we'd rather transcode than have the API
// guess a decoder from a filename we don't trust.
export function needsTranscode(name) {
  return !OPENAI_EXTS.has(extOf(name));
}
