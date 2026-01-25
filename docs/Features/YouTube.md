# YouTube Integration

MASterSelects includes a built-in YouTube panel for searching, downloading, and editing YouTube videos directly in your project.

## YouTube Panel

Access the YouTube panel from the dock system. It provides:

- **Search**: Search YouTube videos via Invidious (no API key required) or YouTube Data API
- **Thumbnails**: Display video thumbnails, titles, channels, and duration
- **Quality Selection**: Choose video quality before downloading
- **Download**: Download videos via Native Helper (yt-dlp) or Cobalt API fallback

## Download Methods

### Native Helper (Recommended)

The Native Helper provides the fastest and most reliable downloads:

1. Install the Native Helper from the toolbar indicator
2. The helper includes yt-dlp for YouTube downloads
3. Downloads are saved to the project's `YT/` folder
4. H.264 codec is preferred for maximum compatibility

### Cobalt API Fallback

When Native Helper is unavailable:

- Multiple Cobalt instances are tried with CORS proxy
- Automatic fallback between instances
- Downloads may be slower than Native Helper

## Adding Videos to Timeline

### Quick Add

1. Search for a video in the YouTube panel
2. Click "Add to Timeline" button
3. Select video quality (if Native Helper available)
4. Video downloads and appears on timeline

### Paste URL

1. Paste a YouTube URL in the panel
2. Video is automatically fetched and displayed
3. Click "Add to Timeline" to download

## Project Storage

Downloaded YouTube videos are:

- Saved to `{ProjectFolder}/YT/` directory
- Automatically added to Media Panel
- Persisted with project saves
- Linked to timeline clips

## Format Selection

When downloading via Native Helper:

| Priority | Codec | Container | Notes |
|----------|-------|-----------|-------|
| 1 | H.264 | MP4 | Best compatibility |
| 2 | VP9 | WebM | Good quality, larger files |
| 3 | AV1 | WebM | Best compression, may need fallback |

The system prefers H.264 for maximum WebCodecs compatibility during export.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Downloads fail | Check Native Helper is running |
| No quality options | Install Native Helper for quality selection |
| Slow downloads | Use Native Helper instead of Cobalt |
| Video won't play | Check codec support, prefer H.264 |
| Audio missing | Ensure audio track was included in download |

## API Keys (Optional)

For higher rate limits, configure YouTube Data API:

1. Open Settings from the menu
2. Enter YouTube Data API key
3. API provides better metadata and search results

Without an API key, Invidious instances are used (rate-limited but free).

---

*See also: [Media Panel](./Media-Panel.md) | [Native Helper](./Native-Helper.md)*
