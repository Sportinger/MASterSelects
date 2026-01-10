# AI Integration

[← Back to Index](./README.md)

GPT-powered editing with 50+ tools, transcription, and multicam EDL generation.

---

## Table of Contents

- [AI Chat Panel](#ai-chat-panel)
- [AI Video Panel](#ai-video-panel)
- [AI Editor Tools](#ai-editor-tools)
- [Transcription](#transcription)
- [Multicam EDL](#multicam-edl)
- [Configuration](#configuration)

---

## AI Chat Panel

### Location
- Default tab in dock panels
- View menu → AI Chat

### Features
- Interactive chat interface
- Model selection dropdown
- Conversation history
- Clear chat button
- Auto-scrolling
- Tool execution indicators

### Available Models
```
GPT-5.2 series (Dec 2025)
GPT-5.1, GPT-5
GPT-4.1, GPT-4o variants
o3, o4-mini, o3-pro (reasoning)
```

### Editor Mode
When enabled (default):
- Includes timeline context in prompts
- 50+ editing tools available
- AI can manipulate timeline directly

---

## AI Video Panel

### Location
- Tab next to AI Chat in dock panels
- View menu → AI Video

### Supported Services
Currently supports **Kling AI** for video generation.

### Generation Modes

#### Text-to-Video
Generate video from text prompts:
- Describe the scene, subjects, and actions
- Select aspect ratio (16:9, 9:16, 1:1)
- Control camera movement

#### Image-to-Video
Animate images:
- Upload start frame (required)
- Optional end frame for guided animation
- Video morphs between frames

### Parameters

| Parameter | Options | Description |
|-----------|---------|-------------|
| **Model** | v1.0, v1.5, v1.6, v2.0, v2.1 | Newer models have better quality |
| **Duration** | 5s, 10s | Video length |
| **Aspect Ratio** | 16:9, 9:16, 1:1 | Output dimensions |
| **Quality** | Standard, Professional | Pro is slower but higher quality |
| **CFG Scale** | 0.0-1.0 | Prompt adherence strength |
| **Camera** | None, Down&Back, Forward&Up, etc. | Camera movement presets |
| **Negative Prompt** | Text | What to avoid in generation |

### Generation Queue
- Jobs appear in queue with status
- Status: Queued → Processing → Done/Failed
- Download generated videos directly
- Remove jobs from queue

### API Authentication
Kling uses JWT authentication. Get your API key from:
[Kling AI Developer Portal](https://app.klingai.com/global/dev/document-api)

### Task-Based Workflow
```
1. Submit generation request
2. Receive task ID
3. Poll for status (every 5 seconds)
4. Download video when complete
```

---

## AI Editor Tools

### 50+ Tools Implemented

#### Timeline State (3 tools)
| Tool | Description |
|------|-------------|
| `getTimelineState` | Full timeline state (tracks, clips, playhead) |
| `getClipDetails` | Detailed clip info + analysis + transcript |
| `getClipsInTimeRange` | Find clips in time range |

#### Playback (2 tools)
| Tool | Description |
|------|-------------|
| `setPlayhead` | Move playhead to time |
| `setInOutPoints` | Set in/out markers |

#### Clip Editing (6 tools)
| Tool | Description |
|------|-------------|
| `splitClip` | Split at specific time |
| `deleteClip` | Delete single clip |
| `deleteClips` | Delete multiple clips |
| `moveClip` | Move to new position/track |
| `trimClip` | Adjust in/out points |
| `cutRangesFromClip` | Remove multiple sections |

#### Track Tools (4 tools)
| Tool | Description |
|------|-------------|
| `createTrack` | Create video/audio track |
| `deleteTrack` | Delete track and clips |
| `setTrackVisibility` | Show/hide track |
| `setTrackMuted` | Mute/unmute track |

#### Visual Capture (2 tools)
| Tool | Description |
|------|-------------|
| `captureFrame` | Export PNG at time |
| `getFramesAtTimes` | Grid image at multiple times |

#### Selection (2 tools)
| Tool | Description |
|------|-------------|
| `selectClips` | Select clips by ID |
| `clearSelection` | Clear selection |

#### Analysis & Transcript (6 tools)
| Tool | Description |
|------|-------------|
| `getClipAnalysis` | Motion/focus/brightness data |
| `getClipTranscript` | Word-level transcript |
| `findSilentSections` | Find silence gaps |
| `findLowQualitySections` | Find blurry sections |
| `startClipAnalysis` | Trigger background analysis |
| `startClipTranscription` | Trigger transcription |

#### Media Panel (7 tools)
| Tool | Description |
|------|-------------|
| `getMediaItems` | Files, compositions, folders |
| `createMediaFolder` | Create folder |
| `renameMediaItem` | Rename item |
| `deleteMediaItem` | Delete item |
| `moveMediaItems` | Move to folder |
| `createComposition` | Create new composition |
| `selectMediaItems` | Select in panel |

### Tool Execution Loop
```
1. User sends message
2. System builds prompt with timeline context
3. OpenAI API call with function calling
4. If tool_calls returned → execute sequentially
5. Collect results → send back to OpenAI
6. Loop until no tool_calls (max 10 iterations)
7. Display final response
```

### Undo Support
All AI edits are undoable with `Ctrl+Z`:
```typescript
// History tracking for batch operations
startHistoryBatch()
// ... execute tools ...
endHistoryBatch()
```

---

## Transcription

### 4 Providers

#### Local Whisper (Browser)
- Uses `@xenova/transformers`
- `whisper-tiny` model
- No API key needed
- Runs in Web Worker

#### OpenAI Whisper API
```
Endpoint: /v1/audio/transcriptions
Model: whisper-1
Format: verbose_json
Granularity: word
```

#### AssemblyAI
```
Upload: /v2/upload
Transcribe: /v2/transcript
Features: Speaker diarization
Polling: 2-minute timeout
```

#### Deepgram
```
Endpoint: /v1/listen
Model: nova-2
Features: Punctuation, speaker diarization
```

### Transcript Format
```typescript
interface TranscriptEntry {
  id: string;
  start: number;   // ms
  end: number;     // ms
  text: string;
  speaker?: string; // For diarization
}
```

### Time Offset Handling
For trimmed clips:
```
Clip inPoint = 5000ms
Word timestamp = 3000ms (within trimmed audio)
Final timestamp = 3000 + 5000 = 8000ms (timeline time)
```

---

## Multicam EDL

### Claude API Integration
```typescript
// Endpoint
https://api.anthropic.com/v1/messages

// Model
claude-sonnet-4-20250514

// Max tokens
4096
```

### Edit Style Presets
| Style | Description |
|-------|-------------|
| `podcast` | Cut to speaker, reaction shots, 3s min |
| `interview` | Show speaker, cut for questions, 2s min |
| `music` | Beat-driven, fast pacing, 1-2s min |
| `documentary` | Long cuts (5+s), B-roll, wide establishing |
| `custom` | User-provided instructions |

### EDL Format
```typescript
interface EditDecision {
  id: string;
  start: number;        // ms
  end: number;          // ms
  cameraId: string;
  reason?: string;
  confidence?: number;  // 0-1
}
```

### Input Data
Claude receives:
- Camera info (names, roles)
- Analysis data (motion, sharpness, faces)
- Transcript with speaker identification
- Audio levels

---

## Configuration

### API Keys
Settings dialog → API Keys:
- OpenAI API key (for chat + transcription)
- Claude API key (for multicam EDL)
- AssemblyAI key
- Deepgram key

Settings dialog → AI Video Generation:
- Kling AI API key (for text-to-video, image-to-video)

### Storage
Keys stored in browser localStorage.

---

## Usage Examples

### Effective Prompts
```
"Move the selected clip to track 2"
"Trim the clip to just the talking parts"
"Remove all segments where motion > 0.7"
"Create a rough cut keeping only focused shots"
"Split at all the 'um' and 'uh' moments"
```

### Iterative Editing
1. Make AI edit
2. Preview result
3. Undo if needed (`Ctrl+Z`)
4. Refine prompt
5. Repeat

---

## Related Features

- [Timeline](./Timeline.md) - Editing interface
- [Audio](./Audio.md) - Multicam sync
- [Media Panel](./Media-Panel.md) - Organization
- [Keyboard Shortcuts](./Keyboard-Shortcuts.md)

---

*Source: `src/components/panels/AIChatPanel.tsx`, `src/components/panels/AIVideoPanel.tsx`, `src/services/aiTools.ts`, `src/services/claudeService.ts`, `src/services/klingService.ts`*
