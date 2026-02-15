# MasterSelects AI Tool Executor

Execute MasterSelects video editor AI tools via browser automation on localhost:5173.

## allowed-tools
- mcp__claude-in-chrome__javascript_tool
- mcp__claude-in-chrome__tabs_context_mcp
- mcp__claude-in-chrome__navigate
- mcp__claude-in-chrome__tabs_create_mcp
- Bash

## Workflow

1. Call `tabs_context_mcp` to get available tabs
2. Find a tab already on `localhost:5173`, or create a new tab and navigate to `http://localhost:5173`
3. Execute the user's request using `javascript_tool` with:
   ```javascript
   (async () => {
     const result = await window.aiTools.execute('toolName', { ...args });
     return JSON.stringify(result);
   })()
   ```
   **Important:** Always wrap in `(async () => { ... })()` — bare `await` is not supported in browser context.
4. Return the result to the user. If `success: false`, report the error.

## User Request

Interpret `$ARGUMENTS` as a natural language request. Determine which tool(s) to call and with what parameters. If the request involves multiple operations, use `executeBatch` to group them into a single undo point.

If no arguments are provided, call `getTimelineState` and report the current project state.

---

## YouTube Search & Download (via Bash)

**IMPORTANT:** YouTube search must be done via Bash with `python -m yt_dlp`, NOT via the browser `searchVideos` tool (which gets blocked by browser data filters).

### Search for videos
```bash
python -m yt_dlp "ytsearch10:QUERY" --flat-playlist -j --no-warnings 2>/dev/null | python -c "
import sys, json
for line in sys.stdin:
    v = json.loads(line)
    dur = v.get('duration') or 0
    title = v.get('title','?')
    vid = v.get('id','?')
    uploader = v.get('uploader','?')
    views = v.get('view_count', 0)
    print(f'{vid}\t{dur}\t{views}\t{uploader}\t{title}')
"
```
- Change `ytsearch10` number to control result count (e.g. `ytsearch5` for 5 results)
- Filter by duration in the python script: add `if dur <= MAX_SECONDS:` before print
- Video URL is `https://www.youtube.com/watch?v={vid}`

### List available formats
```bash
python -m yt_dlp "URL" -F --no-warnings 2>/dev/null
```

### Download workflow
After finding a video via Bash search, use `downloadAndImportVideo` via `javascript_tool` to download and import into the timeline. The download still goes through the Native Helper.

### Complete YouTube download flow (example)
1. **Search** (Bash): `python -m yt_dlp "ytsearch5:nature trees" --flat-playlist -j --no-warnings`
2. **Filter** results by duration, pick a video
3. **Create composition** (javascript_tool): `createComposition` with name (auto-opens it)
4. **Download & import** (javascript_tool): `downloadAndImportVideo` with url, title, compositionId

---

## Complete Tool Reference (42 tools)

### Timeline (3)
| Tool | Parameters | Description |
|------|-----------|-------------|
| `getTimelineState` | _(none)_ | Get tracks, clips, playhead, duration. **Call this first!** |
| `setPlayhead` | `time` (seconds) | Move playhead |
| `setInOutPoints` | `inPoint?`, `outPoint?` (seconds) | Set playback/export range |

### Clip Editing (11)
| Tool | Parameters | Description |
|------|-----------|-------------|
| `getClipDetails` | `clipId` | Full clip info (analysis, transcript, effects, transform) |
| `getClipsInTimeRange` | `startTime`, `endTime`, `trackType?` | Find clips in a time range |
| `splitClip` | `clipId`, `splitTime`, `withLinked?` | Split clip at time |
| `splitClipEvenly` | `clipId`, `parts`, `withLinked?` | Split into N equal parts |
| `splitClipAtTimes` | `clipId`, `times[]`, `withLinked?` | Split at multiple times |
| `deleteClip` | `clipId`, `withLinked?` | Delete a clip |
| `deleteClips` | `clipIds[]`, `withLinked?` | Delete multiple clips |
| `moveClip` | `clipId`, `newStartTime`, `newTrackId?`, `withLinked?` | Move clip |
| `trimClip` | `clipId`, `inPoint`, `outPoint` | Trim clip source in/out |
| `cutRangesFromClip` | `clipId`, `ranges[]` | Cut out time ranges from clip |
| `reorderClips` | `clipIds[]`, `withLinked?` | Reorder clips sequentially |

### Selection (2)
| Tool | Parameters | Description |
|------|-----------|-------------|
| `selectClips` | `clipIds[]` | Select clips |
| `clearSelection` | _(none)_ | Deselect all |

### Track Management (4)
| Tool | Parameters | Description |
|------|-----------|-------------|
| `createTrack` | `type` ("video" or "audio") | Create a new track |
| `deleteTrack` | `trackId` | Delete track and its clips |
| `setTrackVisibility` | `trackId`, `visible` | Show/hide video track |
| `setTrackMuted` | `trackId`, `muted` | Mute/unmute audio track |

### Analysis & Transcript (6)
| Tool | Parameters | Description |
|------|-----------|-------------|
| `getClipAnalysis` | `clipId` | Get motion/faces/colors analysis |
| `getClipTranscript` | `clipId` | Get word-level transcript |
| `findSilentSections` | `clipId`, `minDuration?` | Find speech pauses |
| `findLowQualitySections` | `clipId`, `metric?`, `threshold?`, `minDuration?` | Find blurry/dark sections |
| `startClipAnalysis` | `clipId` | Start background analysis |
| `startClipTranscription` | `clipId` | Start background transcription |

### Preview & Frame Capture (3)
| Tool | Parameters | Description |
|------|-----------|-------------|
| `captureFrame` | `time?` | Screenshot as base64 PNG |
| `getCutPreviewQuad` | `cutTime`, `frameSpacing?` | 8-frame grid around cut point |
| `getFramesAtTimes` | `times[]`, `columns?` | Capture frames at specific times |

### Media Panel (8)
| Tool | Parameters | Description |
|------|-----------|-------------|
| `getMediaItems` | `folderId?` | List files, folders, compositions |
| `createMediaFolder` | `name`, `parentFolderId?` | Create folder |
| `renameMediaItem` | `itemId`, `newName` | Rename item |
| `deleteMediaItem` | `itemId` | Delete item |
| `moveMediaItems` | `itemIds[]`, `targetFolderId?` | Move items |
| `createComposition` | `name`, `width?`, `height?`, `frameRate?`, `duration?`, `openAfterCreate?` | New composition (auto-opens by default) |
| `openComposition` | `compositionId` | Open composition in timeline |
| `selectMediaItems` | `itemIds[]` | Select media items in panel |

### YouTube / Downloads (4)
| Tool | Parameters | Description |
|------|-----------|-------------|
| `searchVideos` | `query`, `maxResults?`, `maxDuration?`, `minDuration?` | **DO NOT USE** — gets blocked by browser filters. Use Bash yt-dlp instead (see above). |
| `listVideoFormats` | `url` | List available download formats (needs Native Helper) |
| `downloadAndImportVideo` | `url`, `title`, `formatId?`, `thumbnail?`, `compositionId?`, `startTime?` | Download & import to timeline. Clips go to position 0 on empty timelines. |
| `getYouTubeVideos` | _(none)_ | Get current download list |

### Batch Execution (1)
| Tool | Parameters | Description |
|------|-----------|-------------|
| `executeBatch` | `actions[]` | Run multiple tools as single undo point. Each action: `{ tool: "toolName", args: { ... } }` |

---

## Linked Clips
Video imports create paired video+audio clips linked via `linkedClipId`. All editing tools support `withLinked` (default: `true`) to edit both together. Set `withLinked: false` to edit only the targeted clip independently.

## Notes
- All times are in seconds (float)
- All modifying tools create undo/redo history points
- `executeBatch` groups all actions into a single undo point
- Download/import requires the Native Helper to be running (WebSocket port 9876)
- YouTube **search** uses Bash (`python -m yt_dlp`), **download** uses Native Helper
- `createComposition` now auto-opens — no need for separate `openComposition` call
- `downloadAndImportVideo` places clips at 0 on empty timelines (not at 60s)
