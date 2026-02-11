# Download Panel (formerly YouTube Panel)

MASterSelects includes a built-in Download panel for searching, downloading, and editing online videos from multiple platforms directly in your project.

## Supported Platforms

The Download panel supports video URLs from the following platforms via yt-dlp:

| Platform | URL Detection | Project Subfolder |
|----------|---------------|-------------------|
| YouTube | `youtube.com`, `youtu.be` | `Downloads/YT/` |
| TikTok | `tiktok.com` | `Downloads/TikTok/` |
| Instagram | `instagram.com` | `Downloads/Instagram/` |
| Twitter / X | `twitter.com`, `x.com` | `Downloads/Twitter/` |
| Facebook | `facebook.com`, `fb.watch` | `Downloads/Facebook/` |
| Reddit | `reddit.com` | `Downloads/Reddit/` |
| Vimeo | `vimeo.com` | `Downloads/Vimeo/` |
| Twitch | `twitch.tv` | `Downloads/Twitch/` |
| Dailymotion | `dailymotion.com` | `Downloads/Other/` |
| Other | Any HTTP(S) URL | `Downloads/Other/` |

Any URL supported by yt-dlp can be downloaded via the Native Helper, even if the platform is not explicitly listed above. Unrecognized platforms are stored under `Downloads/Other/`.

## Download Panel

Access the Download panel from the dock system. It provides:

- **URL Paste**: Paste any supported video URL to fetch metadata and add to the panel
- **YouTube Search**: Search YouTube videos via the Data API (requires API key) or paste YouTube URLs directly (no key required)
- **Thumbnails**: Display video thumbnails, titles, channels, and duration
- **Quality Selection**: Choose video quality/format before downloading (via Native Helper)
- **Download**: Download videos via Native Helper (yt-dlp)
- **Add to Timeline**: Download and add directly to the timeline in one step
- **Auto Download**: Toggle to automatically start downloading when a URL is pasted

## Download Methods

### Native Helper (Required)

The Native Helper provides downloads for all supported platforms:

1. Install the Native Helper from the toolbar indicator
2. The helper includes yt-dlp which supports hundreds of video sites
3. Downloads are saved to the project's `Downloads/<Platform>/` folder
4. H.264 codec is preferred for maximum compatibility

Non-YouTube URLs **require** the Native Helper. YouTube URLs also work best with the helper for quality selection and reliable downloads.

### YouTube oEmbed Metadata

For YouTube URLs specifically, video metadata (title, channel, thumbnail) is fetched via the YouTube oEmbed API without requiring the Native Helper or an API key. The actual download still requires the Native Helper.

## Adding Videos to Timeline

### Quick Add

1. Paste a video URL in the Download panel
2. The panel auto-detects the platform and fetches metadata
3. Click the "+" button to add to timeline
4. Select video quality (format dialog appears)
5. Video downloads and appears on the timeline at the playhead position

### Drag & Drop

1. Drag a video card from the Download panel
2. Drop it onto the timeline
3. The video downloads and is placed at the drop position

### Download Only

Click the download arrow button on a video card to download the file without adding it to the timeline. The file is saved to the project's platform-specific subfolder and also offered as a browser download.

## Project Storage

Downloaded videos are organized into platform-specific subfolders:

```
{ProjectFolder}/
  Downloads/
    YT/            # YouTube videos
    TikTok/        # TikTok videos
    Instagram/     # Instagram videos
    Twitter/       # Twitter/X videos
    Facebook/      # Facebook videos
    Reddit/        # Reddit videos
    Vimeo/         # Vimeo videos
    Twitch/        # Twitch clips
    Other/         # Dailymotion, unrecognized platforms
```

- Files are automatically saved when a project is open
- Downloaded files are added to the Media Panel
- Files persist with project saves
- When no project is open, files are kept in memory only

## Format Selection

When downloading via Native Helper, a format dialog shows available qualities. The system prefers:

| Priority | Codec | Container | Notes |
|----------|-------|-----------|-------|
| 1 | H.264 | MP4 | Best compatibility |
| 2 | VP9 | WebM | Good quality, larger files |
| 3 | AV1 | WebM | Best compression, may need fallback |

The system prefers H.264 for maximum WebCodecs compatibility during export. If no format recommendations are available, the default format is used automatically.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Downloads fail | Check Native Helper is running |
| No quality options | Install Native Helper for quality selection |
| Non-YouTube URL fails | Native Helper is required for non-YouTube platforms |
| Video won't play | Check codec support, prefer H.264 |
| Audio missing | Ensure audio track was included in download |
| "URL may not be supported" | yt-dlp may not support this particular site |

## API Keys (Optional)

For YouTube search functionality, configure the YouTube Data API key:

1. Open Settings from the menu
2. Enter YouTube Data API key
3. API provides search results with metadata (title, duration, views, channel)

Without an API key, you can still paste YouTube URLs directly (metadata is fetched via oEmbed). Non-YouTube platforms do not require any API key.

---

*See also: [Media Panel](./Media-Panel.md) | [Native Helper](./Native-Helper.md)*
