# AI Integration

[← Back to Index](./README.md)

GPT-powered editing with 50+ tools, transcription, multicam EDL generation, and AI object segmentation.

---

## Table of Contents

- [AI Chat Panel](#ai-chat-panel)
- [AI Video Panel](#ai-video-panel)
- [AI Segmentation (SAM 2)](#ai-segmentation-sam-2)
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

### Panel Tabs
- **AI Video**: Generation interface
- **History**: List of all generated videos

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
- Drag & drop or click to upload start/end frames
- **Use Current Frame** button captures timeline preview
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

### Timeline Integration
- **Add to Timeline** checkbox (enabled by default)
- Videos auto-import to "KlingAI" folder in Media Panel
- Clips placed on empty or new video track at playhead
- Videos with audio get linked audio clips

### Generation Queue
- Jobs appear in queue with status
- Status: Queued → Processing → Done/Failed
- Download generated videos directly
- Remove jobs from queue

### History Tab
- Persistent list of all generated videos
- Video thumbnails with play/pause
- Draggable to timeline
- "In Timeline" badge for added clips
- "+ Timeline" button to add manually

### API Authentication
Kling uses Access Key + Secret Key for JWT authentication:
1. Get credentials from [Kling AI Developer Portal](https://app.klingai.com/global/dev/document-api)
2. Enter Access Key (AK) in Settings
3. Enter Secret Key (SK) in Settings
4. JWT tokens are generated automatically with 30-minute caching

### Task-Based Workflow
```
1. Submit generation request
2. Receive task ID
3. Poll for status (every 5 seconds)
4. On completion:
   - Import video to KlingAI folder
   - Optionally add clip to timeline
   - Add to history for later access
```

---

## AI Segmentation (SAM 2)

> **Status:** Work in progress (WIP badge in the panel tab)

### Overview
AI-powered object segmentation using Meta's **Segment Anything Model 2 (SAM 2)**. Click on objects in the preview to create precise masks, then optionally propagate those masks across video frames. All inference runs locally in the browser using ONNX Runtime with WebGPU acceleration — no API keys or cloud services required.

### Location
- Tab in dock panels alongside AI Chat and AI Video
- View menu → AI Segment

### One-Time Model Download
On first use, the panel prompts for a one-time model download:
- **Model:** SAM 2 Hiera Small (ORT-optimized encoder + ONNX decoder)
- **Total size:** ~184 MB
- **Storage:** Cached in the browser's Origin Private File System (OPFS) for persistent local storage
- **Progress:** Download progress bar shown in the panel
- **Fallback URLs:** If the primary CDN is unavailable, models are fetched from fallback hosts automatically
- After download, the model auto-loads into ONNX sessions for immediate use
- On subsequent visits, cached models are detected and loaded automatically

### Model Lifecycle

| Status | Description |
|--------|-------------|
| **Not Downloaded** | Panel shows download prompt |
| **Downloading** | Progress bar with percentage |
| **Downloaded** | Cached in OPFS, auto-loading |
| **Loading** | Creating ONNX inference sessions |
| **Ready** | Green status dot, ready for segmentation |
| **Error** | Red status dot with error message and retry button |

### Point-Based Segmentation
Once the model is ready and a clip is selected in the timeline:

1. **Activate** segmentation mode using the Activate button
2. **Left-click** on the preview canvas to place a **foreground point** (green) — marks regions to include in the mask
3. **Right-click** to place a **background point** (red) — marks regions to exclude from the mask
4. Each point triggers an immediate decode pass, updating the mask in real time
5. Points are listed in the panel with coordinates and can be individually removed

The **Auto-Detect** button places a center-point and runs a full encode + decode cycle, useful for quick initial segmentation of a prominent subject.

### Preview Overlay
When segmentation is active, a transparent overlay appears on top of the preview canvas:
- **Mask visualization:** Selected regions shown as a blue semi-transparent overlay
- **Point markers:** Foreground points shown as green dots, background points as red dots, each with a white border and center dot
- **Processing indicator:** Text overlay appears while the model is computing
- **Crosshair cursor:** Indicates the overlay is ready for point placement

### Display Settings

| Setting | Range | Description |
|---------|-------|-------------|
| **Opacity** | 0–100% | Transparency of the mask overlay |
| **Feather** | 0–50px | Edge softness of the mask |
| **Invert Mask** | On/Off | Swap foreground and background regions |

### Video Propagation
After creating a mask on the current frame, propagate it forward across subsequent frames:
- **Forward** button propagates the mask up to 150 frames (~5 seconds at 30fps)
- Progress bar and percentage shown during propagation
- **Stop** button to cancel propagation at any time
- Each propagated frame's mask is RLE-compressed and stored efficiently in memory
- Propagation uses the SAM 2 memory bank mechanism to track objects across frames

### Architecture
All heavy computation runs off the main thread to keep the UI responsive:
- **Web Worker** (`sam2Worker`) handles ONNX encoder and decoder inference
- **Encoder** runs with WebGPU acceleration (WASM fallback if WebGPU is unavailable in the worker)
- **Decoder** runs on WASM (small model, fast enough without GPU)
- **Message protocol:** Main thread sends encode/decode/propagate requests; worker responds with embeddings, masks, and progress updates
- **RLE compression** for storing per-frame masks efficiently (run-length encoding of binary mask data)

### Workflow
```
1. Open AI Segment panel
2. Download model (first time only, ~184 MB)
3. Select a video clip in the timeline
4. Click "Activate" to enable segmentation mode
5. Left-click objects to include, right-click to exclude
6. Adjust opacity/feather/invert as needed
7. Optionally propagate mask forward through video
8. Clear All to reset and start over
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

### No API Key Required
- **SAM 2 AI Segmentation** — runs entirely in the browser, no external service

### Storage
Keys stored in browser localStorage. SAM 2 model files stored in OPFS.

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

*Source: `src/components/panels/AIChatPanel.tsx`, `src/components/panels/AIVideoPanel.tsx`, `src/components/panels/SAM2Panel.tsx`, `src/components/preview/SAM2Overlay.tsx`, `src/services/sam2/SAM2Service.ts`, `src/services/sam2/SAM2ModelManager.ts`, `src/services/sam2/sam2Worker.ts`, `src/stores/sam2Store.ts`, `src/services/aiTools.ts`, `src/services/claudeService.ts`, `src/services/klingService.ts`*
