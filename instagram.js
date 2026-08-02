// Instagram link resolution, shared by the CLI and the Telegram bot.
//
// downreels.com's backend resolves a public IG post/reel to a direct MP4. We
// call the same endpoint its web UI calls and hand back the bytes, so neither
// front-end has to keep its own copy of this.

// The endpoint downreels.com's web UI posts to.
const IG_API =
  process.env.INSTAGRAM_API || "https://api.zoraahub.com/fetch.php";

// A browser-like User-Agent; the media host rejects some default clients.
const HTTP_UA =
  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 " +
  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36";

// Posts (/p/), reels (/reel/, /reels/) and IGTV (/tv/) all carry video, and
// they appear both bare and under a profile (/username/reel/XXXX/). Share
// links pick up ?igsh=... query junk, which the resolver tolerates.
const IG_URL_RE =
  /^https?:\/\/(?:www\.)?instagram\.com\/(?:[^/]+\/)?(?:reel|reels|p|tv)\/[A-Za-z0-9._-]+/i;

export function isInstagramUrl(s = "") {
  return IG_URL_RE.test(s.trim());
}

// Pull the first Instagram URL out of a block of text, or null. Pasting a link
// into a chat often drags along a caption or a trailing flag.
export function findInstagramUrl(text = "") {
  for (const word of text.split(/\s+/)) if (isInstagramUrl(word)) return word;
  return null;
}

// Best-effort shortcode for the output filename, e.g. .../reel/ABC123/ -> ABC123.
export function instagramShortcode(url = "") {
  const m = url.match(
    /instagram\.com\/(?:[^/]+\/)?(?:reel|reels|p|tv)\/([A-Za-z0-9._-]+)/i
  );
  return m ? m[1] : "instagram-video";
}

// Ask downreels.com's backend to resolve a public IG URL to a direct MP4 link.
export async function resolveInstagram(pageUrl) {
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

// Download a URL fully into memory. Reels are seconds to a couple of minutes,
// so buffering is fine and keeps the MP4 off disk. `maxBytes` guards against a
// surprise feature-length upload on a memory-limited host.
export async function downloadToBuffer(url, maxBytes = 0) {
  let res;
  try {
    res = await fetch(url, { headers: { "User-Agent": HTTP_UA } });
  } catch (e) {
    throw new Error(`failed to download the video: ${e.message}`);
  }
  if (!res.ok) throw new Error(`failed to download the video (HTTP ${res.status}).`);

  const declared = Number(res.headers.get("content-length")) || 0;
  if (maxBytes && declared > maxBytes)
    throw new Error(
      `the video is ${(declared / 1024 / 1024).toFixed(1)} MB, over the ` +
        `${(maxBytes / 1024 / 1024).toFixed(0)} MB limit.`
    );

  const buf = Buffer.from(await res.arrayBuffer());
  if (buf.length === 0) throw new Error("the downloaded video was empty.");
  // Not every host sends content-length, so re-check once we have the bytes.
  if (maxBytes && buf.length > maxBytes)
    throw new Error(
      `the video is ${(buf.length / 1024 / 1024).toFixed(1)} MB, over the ` +
        `${(maxBytes / 1024 / 1024).toFixed(0)} MB limit.`
    );
  return buf;
}

// Resolve + download in one call — what both front-ends actually want.
export async function fetchInstagramVideo(pageUrl, maxBytes = 0) {
  const videoUrl = await resolveInstagram(pageUrl);
  return downloadToBuffer(videoUrl, maxBytes);
}
