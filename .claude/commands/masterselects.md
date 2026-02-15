# MasterSelects AI Tool Executor

Execute MasterSelects video editor AI tools via browser automation on localhost:5173.

## allowed-tools
- mcp__claude-in-chrome__javascript_tool
- mcp__claude-in-chrome__tabs_context_mcp
- mcp__claude-in-chrome__navigate
- mcp__claude-in-chrome__tabs_create_mcp

## Workflow

1. Call `tabs_context_mcp` to get available tabs
2. Find a tab already on `localhost:5173`, or create a new tab and navigate to `http://localhost:5173`
3. Execute the user's request using `javascript_tool` with:
   ```javascript
   const result = await window.aiTools.execute('toolName', { ...args });
   JSON.stringify(result);
   ```
4. Return the result to the user. If `success: false`, report the error.

## User Request

Interpret `$ARGUMENTS` as a natural language request. Determine which tool(s) to call and with what parameters. If the request involves multiple operations, use `executeBatch` to group them into a single undo point.

If no arguments are provided, call `getTimelineState` and report the current project state.

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
| `createComposition` | `name`, `width?`, `height?`, `frameRate?`, `duration?` | New composition |
| `openComposition` | `compositionId` | Open composition in timeline |
| `selectMediaItems` | `itemIds[]` | Select media items in panel |

### YouTube / Downloads (4)
| Tool | Parameters | Description |
|------|-----------|-------------|
| `searchVideos` | `query`, `maxResults?`, `maxDuration?`, `minDuration?` | Search videos via yt-dlp (needs Native Helper) |
| `listVideoFormats` | `url` | List available download formats (needs Native Helper) |
| `downloadAndImportVideo` | `url`, `title`, `formatId?`, `thumbnail?` | Download & import to timeline |
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
- YouTube/download tools require the Native Helper to be running (WebSocket port 9876)
