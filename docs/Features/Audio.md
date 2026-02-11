# Audio

[← Back to Index](./README.md)

Audio processing with 10-band EQ, audio master clock, varispeed scrubbing, and multicam synchronization.

---

## Table of Contents

- [Audio Master Clock](#audio-master-clock)
- [Varispeed Scrubbing](#varispeed-scrubbing)
- [Audio Tracks](#audio-tracks)
- [10-Band EQ](#10-band-eq)
- [Waveforms](#waveforms)
- [Multicam Sync](#multicam-sync)
- [Transcription](#transcription)
- [Audio Manager](#audio-manager)

---

## Audio Master Clock

The playhead follows audio timing for perfect sync, like Premiere Pro and DaVinci Resolve.

### How It Works
- Audio playback drives the timeline position
- Playhead syncs to `audioContext.currentTime`
- Gradual drift correction using playback rate adjustment
- Prevents audio glitches from hard seeks

### Benefits
- Perfect audio-video sync during playback
- No audio pops or clicks from constant re-syncing
- Professional-grade timing accuracy

### Implementation
```typescript
// Playhead follows audio instead of system time
currentTime = audioStartTime + (audioContext.currentTime - playbackStartTime)

// Gentle drift correction via playback rate
if (drift > threshold) {
  playbackRate = 1.0 + (drift * correctionFactor)
}
```

---

## Varispeed Scrubbing

Continuous audio playback with speed adjustment while scrubbing the timeline.

### Features
- **Continuous playback**: Audio plays at varying speeds during scrub
- **Direction-aware**: Plays forward or backward based on scrub direction
- **Speed-scaled**: Playback rate matches scrub speed
- **All clips**: Works with proxy and non-proxy video clips

### Experience
- Scrub slowly: Hear audio at reduced speed
- Scrub fast: Audio speeds up proportionally
- Stop scrubbing: Audio fades out smoothly

### Technical Details
```typescript
// Speed calculated from scrub velocity
const velocity = (newTime - prevTime) / deltaTime
const playbackRate = clamp(Math.abs(velocity), 0.25, 4.0)

// Time-based triggers for smooth audio
triggerInterval = Math.max(50, 200 / Math.abs(velocity))
```

---

## Audio Tracks

### Track Configuration
```typescript
interface TimelineTrack {
  type: 'video' | 'audio';
  muted: boolean;
  visible: boolean;
  solo: boolean;
}
```

### Default Setup
- 2 video tracks
- 1 audio track (at bottom)
- New audio tracks created automatically when needed

### Track Controls
| Control | Function |
|---------|----------|
| **M** (Mute) | Silence track audio |
| **S** (Solo) | Only play this track |
| **Eye** | Toggle visibility |

---

## 10-Band EQ

### Frequencies
```
31Hz, 62Hz, 125Hz, 250Hz, 500Hz, 1kHz, 2kHz, 4kHz, 8kHz, 16kHz
```

### Gain Range
- **-12dB to +12dB** per band
- Q factor: 1.4 (standard 10-band)

### Implementation
```typescript
// Audio Manager chain
Input → EQ Band 0-9 → Master Gain → Output

// Methods
setEQBand(bandIndex, gainDB)  - Adjust single band
getEQBands()                   - Get all band values
setAllEQBands(gains)           - Set multiple at once
resetEQ()                      - Flatten to 0dB
```

### Keyframe Support
Each EQ band can be animated:
```typescript
// Effect parameters stored per-clip
{
  band31: 0, band62: 0, band125: 0, band250: 0, band500: 0,
  band1k: 0, band2k: 0, band4k: 0, band8k: 0, band16k: 0
}
```

### Live EQ via Web Audio
- Real-time equalization using Web Audio API
- Hear EQ changes instantly during playback
- Changes apply in real-time without needing to re-render

### Properties Panel - Volume Tab
When an audio clip is selected, the Properties panel shows the Volume tab:
- **Volume slider**: 0-200% (with keyframe toggle)
- **Keep Pitch checkbox**: Maintain original pitch when speed changes (default on)
- **10 vertical EQ sliders**: One per frequency band
- **Reset button**: Flattens all bands to 0dB
- **Keyframe toggles**: Per parameter for animation (single toggle for all bands)
- EQ effect is automatically added on first use

### Audio Tab for Video Clips
Video clips now have a dedicated **Audio** tab in the Properties panel:
- Volume controls and keyframes for linked audio
- Same EQ and volume controls as audio-only clips
- Accessible without selecting the linked audio clip separately

See [UI Panels - Properties Panel](./UI-Panels.md#properties-panel) for details.

---

## Speed and Pitch

### Speed Effect on Audio
Audio clips respect the clip's speed property (set on linked video or directly):
- **Range**: 0.25x to 4x (browser limitation)
- **Playback rate**: Audio plays at the same speed as video
- **Sync**: Uses speed integration for keyframe-accurate timing

### Pitch Preservation
The `preservesPitch` property controls pitch behavior during speed changes:

| Setting | Behavior | Use Case |
|---------|----------|----------|
| **Keep Pitch ON** (default) | Maintains original pitch | Speech, music |
| **Keep Pitch OFF** | Pitch shifts with speed | Chipmunk effect, slowed audio |

```typescript
interface TimelineClip {
  speed?: number;           // Playback speed (default 1.0)
  preservesPitch?: boolean; // Keep pitch when speed changes (default true)
}
```

### Implementation
Uses Web Audio API's `playbackRate` and `preservesPitch` properties:
```typescript
audio.playbackRate = currentSpeed;  // 0.25 to 4.0
audio.preservesPitch = clip.preservesPitch !== false;
```

See [UI Panels - Properties Panel](./UI-Panels.md#properties-panel) for details.

---

## Waveforms

### Generation
```typescript
generateWaveform(file, samplesPerSecond = 50, onProgress?)
```

- Uses Web Audio API
- ~50 samples per second
- Peak-based values for visual clarity
- Normalized amplitude (0-1)
- Dynamic sample count (200-10000)

### Clip Integration
```typescript
interface TimelineClip {
  waveform?: number[];        // Amplitude values
  waveformGenerating?: boolean;
  waveformProgress?: number;  // 0-100
}
```

### Display
- Canvas-based rendering (optimized)
- Scales with zoom level
- Toggle: "Wave On/Off" in timeline controls
- Skipped for files >4GB

---

## Multicam Sync

### Algorithm
**Normalized cross-correlation** (Pearson coefficient):
- Search range: ±10 seconds
- Uses downsampled audio fingerprints (2000Hz)
- Returns offset in milliseconds with confidence

### Methods

#### Two-File Sync
```typescript
findOffset(masterMediaFileId, targetMediaFileId, maxOffsetSeconds = 30)
// Returns: offset in ms (positive = target delayed)
```

#### Multicam Batch Sync
```typescript
syncMultipleClips(masterClip, targetClips, onProgress)
// Accounts for clip in-points and durations
// Returns: Map<clipId, offsetMs>
```

### Fingerprint Caching
- Cache key: `${mediaFileId}-${startTime}-${duration}`
- Prevents redundant processing
- `clearCache()` for manual cleanup

### Audio Analysis
```typescript
// RMS level analysis
analyzeLevels(mediaFileId, windowSizeMs = 100)

// Audio fingerprinting
generateFingerprint(mediaFileId, targetSampleRate = 2000, startTimeSeconds, maxDurationSeconds = 30)

// Peak at timestamp
getLevelAtTime(mediaFileId, timestampMs)
```

---

## Transcription

### Supported Providers

| Provider | Type | Features |
|----------|------|----------|
| **Local Whisper** | Browser | No API key needed |
| **OpenAI Whisper** | Cloud | Fast, accurate |
| **AssemblyAI** | Cloud | Speaker diarization |
| **Deepgram** | Cloud | Real-time capable |

### Languages Supported
`de, en, es, fr, it, pt, nl, pl, ru, ja, zh, ko`

### Clip Transcription
```typescript
// Transcribes trimmed portion only (inPoint to outPoint)
transcribeClip(clipId, options)

// Returns word-level transcript
interface TranscriptEntry {
  id: string;
  start: number;   // ms
  end: number;     // ms
  text: string;
}
```

### Web Worker Support
- Transcription runs in background
- Doesn't block UI
- Progress callbacks (0-100%)

### Transcript Storage
```typescript
interface TimelineClip {
  transcript?: TranscriptEntry[];
  transcriptStatus: 'none' | 'transcribing' | 'ready';
}
```

---

## Audio Manager

### Web Audio API Integration
```typescript
class AudioManager {
  audioContext: AudioContext;
  masterGain: GainNode;
  eqFilters: BiquadFilterNode[]; // 10 bands
}
```

### Methods
```typescript
init()                           // Initialize AudioContext
connectMediaElement(element)     // Connect video/audio
disconnectMediaElement(element)  // Cleanup
setMasterVolume(volume)          // 0-1 range
getMasterVolume()                // Get current
destroy()                        // Cleanup resources
```

### Autoplay Policy
- Automatically resumes AudioContext
- Handles browser autoplay restrictions

---

## Multicam Panel

### Features
- Add/remove camera sources
- **Master camera** designation for audio reference
- Camera **role assignment** (wide, close-up, detail, custom)
- Sync offset display per camera
- Camera reordering

### State Structure
```typescript
interface MultiCamSource {
  id: string;
  mediaFileId: string;
  name: string;
  role: 'wide' | 'closeup' | 'detail' | 'custom';
  syncOffset: number;  // ms, relative to master
  duration: number;    // ms
}
```

### EDL Generation
AI-powered edit decisions via Claude API:
```typescript
interface EditDecision {
  start: number;        // ms
  end: number;          // ms
  cameraId: string;
  reason?: string;
  confidence?: number;  // 0-1
}
```

---

## Composition Audio Mixdown

When nesting compositions, the inner composition's audio tracks are mixed down to a single buffer for use in the parent timeline.

### How It Works
- `CompositionAudioMixer` extracts and mixes all audio from a nested composition
- Processes all audio tracks including volume, EQ, and speed/pitch changes
- Produces a single mixed buffer at the composition's sample rate
- Used automatically when a composition clip plays in a parent timeline

### Source
- `src/services/compositionAudioMixer.ts`

---

## Audio Export

Audio is exported alongside video with full effect processing.

### Export Pipeline
```
1. Extract audio from clips (AudioExtractor)
2. Apply speed/pitch changes (TimeStretchProcessor)
3. Render EQ and volume with keyframes (AudioEffectRenderer)
4. Mix all tracks (AudioMixer)
5. Encode to AAC (AudioEncoder)
6. Mux with video
```

### Export Settings
| Setting | Options |
|---------|---------|
| **Sample Rate** | 48kHz (video standard), 44.1kHz (CD) |
| **Bitrate** | 128-320 kbps |
| **Normalize** | Peak normalize (prevent clipping) |

### Time Stretch Processing
- Uses **SoundTouchJS** for pitch-preserved tempo changes
- Handles keyframed speed values
- Accurate to keyframe timing via integration

### Effect Rendering
- Offline rendering via `OfflineAudioContext`
- Sample-accurate keyframe automation
- 10-band EQ with animated gains
- Volume automation with bezier curves

### Track Handling
- Respects track mute/solo settings
- Overlapping clips sum correctly
- Peak normalization optional

See [Export](./Export.md) for full export documentation.

---

## Not Implemented

- Audio compression/dynamics
- Reverb/delay effects
- Audio meters/spectrum display
- Loudness normalization
- Noise reduction

---

## Related Features

- [Timeline](./Timeline.md) - Track management
- [AI Integration](./AI-Integration.md) - Transcript editing
- [Media Panel](./Media-Panel.md) - Audio import
- [Keyboard Shortcuts](./Keyboard-Shortcuts.md)

---

*Source: `src/services/audioManager.ts`, `src/services/audioSync.ts`, `src/engine/audio/`, `src/components/panels/PropertiesPanel.tsx`*
