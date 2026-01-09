# AI Integration

[← Back to Index](./README.md)

GPT-powered intelligent editing with timeline manipulation tools.

---

## Table of Contents

- [AI Chat Panel](#ai-chat-panel)
- [AI Editor Tools](#ai-editor-tools)
- [Analysis Tools](#analysis-tools)
- [Transcription](#transcription)
- [Context Awareness](#context-awareness)
- [Configuration](#configuration)

---

## AI Chat Panel

Interactive chat interface for AI-assisted editing.

### Opening AI Chat
- Default tab in dock
- Or View menu → AI Chat
- Middle mouse scroll to switch tabs

### Chat Features
- **Text selection** in messages enabled
- **Dark gray** user messages
- **Streaming** responses

### Available Models
- GPT-5.2, GPT-5.1
- GPT-4 variants
- o3, o4 reasoning models
- Uses `max_completion_tokens` for newer models

---

## AI Editor Tools

AI can manipulate timeline through function calling.

### Available Tools

#### Clip Operations
| Tool | Function |
|------|----------|
| `moveClip` | Move clip to new position/track |
| `trimClip` | Adjust clip in/out points |
| `splitClip` | Split clip at time |
| `deleteClip` | Remove clip from timeline |
| `cutRangesFromClip` | Batch remove sections |

#### Track Operations
| Tool | Function |
|------|----------|
| `addTrack` | Create new track |
| `renameTrack` | Change track name |
| `toggleTrackVisibility` | Show/hide track |

#### Playback
| Tool | Function |
|------|----------|
| `setPlayhead` | Move playhead to time |
| `cutPreview` | Preview edit result |

### Batch Operations
`cutRangesFromClip` enables removing multiple sections:
```json
{
  "clipId": "abc123",
  "ranges": [
    {"start": 10, "end": 15},
    {"start": 30, "end": 35}
  ]
}
```

### Undo Support
- All AI edits are undoable
- `Ctrl+Z` reverts AI changes
- Full integration with history system

---

## Analysis Tools

AI can access clip analysis data.

### Available Analysis Data
- **Focus** - Sharpness/blur detection
- **Motion** - Movement analysis
- **Brightness** - Exposure levels
- **Face count** - Face detection per frame

### Using Analysis
AI sees analysis data for context:
```
"At timestamp 5.2s: focus=0.8, motion=0.3, faceCount=2"
```

Enables queries like:
- "Find the sharpest moments"
- "Cut sections with no faces"
- "Find high-motion segments"

---

## Transcription

AI can work with speech transcripts.

### Transcript-Based Editing
- Word-level timestamps available
- AI sees spoken content
- Enables text-based editing

### Example Commands
- "Remove all 'um' and 'uh' moments"
- "Cut to this speaker's parts"
- "Find where they mention [topic]"

### Transcription Tools
| Tool | Function |
|------|----------|
| `getTranscript` | Retrieve clip transcript |
| `transcribeClip` | Generate transcript |

---

## Context Awareness

AI has full awareness of timeline state.

### What AI Knows

#### Selection State
- Currently selected clips
- Selected clip properties
- Active composition

#### Timeline Structure
- Track names and IDs
- Clip names per track
- Clip positions and durations

#### Visible Range
- Current view bounds
- Playhead position
- In/Out points

### Smart Defaults
- Operations target **selected clip** by default
- Commands respect **visible range**
- No need to specify obvious context

---

## Media Panel Tools

AI can manage media organization.

### Available Tools
| Tool | Function |
|------|----------|
| `createFolder` | Create media folder |
| `moveToFolder` | Organize media |
| `deleteMedia` | Remove from project |
| `renameMedia` | Change media name |

---

## Configuration

### API Key Setup
1. Open Settings dialog
2. Enter OpenAI API key
3. Key stored locally

### Model Selection
- Choose model in chat interface
- Different models for different needs
- Newer models support extended tokens

---

## Usage Tips

### Effective Prompts
```
"Move the selected clip to track 2"
"Trim the clip to just the talking parts"
"Remove all segments where motion > 0.7"
"Create a rough cut keeping only focused shots"
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

*Commits: 86789b3 through d63e381*
