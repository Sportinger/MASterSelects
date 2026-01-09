# Audio

[← Back to Index](./README.md)

Audio track management, waveform visualization, and multicam synchronization.

---

## Table of Contents

- [Audio Tracks](#audio-tracks)
- [Waveforms](#waveforms)
- [Multicam Sync](#multicam-sync)
- [Audio Playback](#audio-playback)
- [Linked Audio](#linked-audio)

---

## Audio Tracks

### Track Types
- **Audio tracks** at bottom of timeline
- Separate from video tracks
- Auto-created when needed

### Track Restrictions
- Audio files only drop on audio tracks
- Video files cannot go on audio tracks
- Visual indicators during drag

### Creating Audio Tracks
- Auto-created when dropping audio
- Or manually add via track controls
- Position always below video tracks

---

## Waveforms

### Automatic Generation
- Generated when audio added
- Runs in background
- Progress indicator shown

### Waveform Display
- Canvas-based rendering (optimized)
- 50 samples per second
- Peak detection for accuracy

### Waveform Features
- Scales with zoom level
- Downsampled to match pixel width
- Handles long audio clips

### Manual Generation
For clips without waveform:
1. Right-click audio clip
2. Select "Generate Waveform"
3. Progress shown in background

### Toggle Generation
In settings:
- Enable/disable auto waveform generation
- Useful for slow systems
- Large files (>500MB) skip by default

### Technical Details
- Canvas limited for very long clips
- Prevents memory issues
- Maintains visual quality

---

## Multicam Sync

Automatically synchronize audio between cameras.

### How It Works
1. Analyzes audio from each clip
2. Cross-correlates waveforms
3. Calculates time offset
4. Aligns clips automatically

### Using Multicam Sync
1. Select clips to sync
2. Right-click → "Sync Audio"
3. Choose sync method
4. Clips align automatically

### Sync Methods

#### Audio-Based
- Compares audio waveforms
- Best for clips with same sound
- Typical multicam scenario

#### Transcript-Based
- Uses speech content
- Good for interviews
- Requires transcription first

### Performance Optimization
- Only first 30 seconds analyzed
- Cross-correlation optimized
- Background processing

### Respecting Trim Bounds
- Sync respects clip in/out points
- Only trimmed portion analyzed
- Accurate for pre-trimmed clips

---

## Audio Playback

### Sync with Video
- Audio playback synced to timeline
- Follows playhead position
- Matches video frame rate

### Mute/Solo Controls
| Control | Function |
|---------|----------|
| **Mute** | Silence track audio |
| **Solo** | Only play this track |

### Solo Behavior
- Dims non-solo tracks visually
- Multiple tracks can be solo'd
- Quick way to isolate audio

---

## Linked Audio

Video clips can have linked audio.

### Linked Behavior
- Audio moves with video clip
- Maintains sync during drag
- Split together with `C` key

### Overlap Resistance
When dragging linked clips:
- 2.0 second resistance buffer
- Prevents accidental overlap
- Magnetic resistance effect

### Breaking Links
- Not yet implemented
- Linked audio always follows video

---

## Transcription

Generate text transcripts from audio.

### Local Transcription
Uses Whisper model locally:
1. Select clip
2. Right-click → "Transcribe"
3. Runs in Web Worker

### Transcription Features
- **Language selection** (German default)
- **Streaming results** during processing
- **Word-level timestamps**
- **Persistent storage** (survives refresh)

### Transcript Panel
When clip selected:
- Shows word-by-word transcript
- Real-time highlighting during playback
- Words wrap properly

### Transcript Settings
- Language selection
- Model quality options
- Anti-hallucination filters

### Delete Transcript
Right-click clip → "Delete Transcript"

---

## Clip Analysis

Analyze audio/video clips for editing.

### Analysis Types
| Type | Description |
|------|-------------|
| **Focus** | Sharpness/blur |
| **Motion** | Movement detection |
| **Brightness** | Exposure levels |
| **Face Count** | Faces per frame |

### Analysis Panel
- Real-time values at playhead
- Graph visualization
- Gradient coloring by threshold

### Analysis Graph
- Shows values over time
- Color-coded by quality
- Amplified line display

### Cached Results
- Analysis stored in IndexedDB
- Persists across refresh
- Cancel button during analysis

---

## Related Features

- [Timeline](./Timeline.md) - Track management
- [AI Integration](./AI-Integration.md) - Transcript editing
- [Media Panel](./Media-Panel.md) - Audio import
- [Keyboard Shortcuts](./Keyboard-Shortcuts.md)

---

*Commits: 71e2895 through d63e381*
