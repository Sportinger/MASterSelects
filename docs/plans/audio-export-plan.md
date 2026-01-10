# Audio Export Implementation Plan

## Overview

Two-tier audio system for MASterSelects:
1. **Preview Tier**: Fast, real-time playback (already implemented)
2. **Export Tier**: High-quality offline rendering (to implement)

---

## Current State

### What Works (Preview Tier)
- [x] Audio playback with `HTMLAudioElement`
- [x] Speed via `playbackRate` (0.25x - 4x)
- [x] Pitch preservation via `preservesPitch`
- [x] Real-time EQ (10-band via Web Audio API)
- [x] Real-time volume control
- [x] Track mute/solo
- [x] Audio sync with video playback
- [x] Speed keyframe integration for timing

### What's Missing (Export Tier)
- [ ] Audio track in exported MP4
- [ ] Offline rendering of speed changes
- [ ] Sample-accurate effect automation
- [ ] Multi-track mixing
- [ ] AAC encoding

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      AUDIO EXPORT PIPELINE                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Timeline State                                                      │
│  ├── clips[]                                                         │
│  ├── tracks[]                                                        │
│  ├── clipKeyframes (Map)                                             │
│  └── duration                                                        │
│           │                                                          │
│           ▼                                                          │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    AudioExportPipeline                       │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │    │
│  │  │   Extract   │→ │  Process    │→ │  Render Effects     │  │    │
│  │  │   Audio     │  │  Speed/Pitch│  │  (EQ, Volume)       │  │    │
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘  │    │
│  │         │                │                    │              │    │
│  │         ▼                ▼                    ▼              │    │
│  │  ┌─────────────────────────────────────────────────────┐    │    │
│  │  │                   Track Mixer                        │    │    │
│  │  │  - Position clips on timeline                        │    │    │
│  │  │  - Apply track mute/solo                             │    │    │
│  │  │  - Sum overlapping audio                             │    │    │
│  │  └─────────────────────────────────────────────────────┘    │    │
│  │                          │                                   │    │
│  │                          ▼                                   │    │
│  │  ┌─────────────────────────────────────────────────────┐    │    │
│  │  │                  Audio Encoder                       │    │    │
│  │  │  - WebCodecs AudioEncoder                            │    │    │
│  │  │  - AAC-LC @ 256kbps                                  │    │    │
│  │  │  - Chunked encoding                                  │    │    │
│  │  └─────────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                          │                                          │
│                          ▼                                          │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                      MP4 Muxer                               │    │
│  │  - Video track (existing)                                    │    │
│  │  - Audio track (new)                                         │    │
│  │  - Synchronized timestamps                                   │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                          │                                          │
│                          ▼                                          │
│                    Final MP4 File                                   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Components

### 1. AudioExtractor

**Purpose**: Decode audio from video/audio files into AudioBuffer

**Location**: `src/engine/audio/AudioExtractor.ts`

```typescript
interface AudioExtractor {
  /**
   * Extract audio from a media file
   * @param file - Video or audio file
   * @returns Decoded AudioBuffer
   */
  extractAudio(file: File): Promise<AudioBuffer>;

  /**
   * Trim AudioBuffer to specific time range
   * @param buffer - Source AudioBuffer
   * @param startTime - Start time in seconds
   * @param endTime - End time in seconds
   * @returns Trimmed AudioBuffer
   */
  trimBuffer(buffer: AudioBuffer, startTime: number, endTime: number): AudioBuffer;

  /**
   * Cache management for decoded audio
   */
  getCached(fileId: string): AudioBuffer | null;
  setCached(fileId: string, buffer: AudioBuffer): void;
  clearCache(): void;
}
```

**Implementation Details**:
- Use `AudioContext.decodeAudioData()` for decoding
- Cache decoded buffers by mediaFileId to avoid re-decoding
- Handle both video files (extract audio track) and audio files
- Memory management: clear cache when export completes

**Edge Cases**:
- Video without audio track → return silent buffer
- Corrupted audio → throw with meaningful error
- Very long files → consider streaming decode (future)

---

### 2. TimeStretchProcessor

**Purpose**: Handle speed changes and pitch preservation

**Location**: `src/engine/audio/TimeStretchProcessor.ts`

**Dependency**: `soundtouch-ts` (~15KB)

```typescript
interface TimeStretchProcessor {
  /**
   * Process audio with constant speed
   * @param buffer - Source AudioBuffer
   * @param speed - Playback speed (0.1 to 10.0)
   * @param preservePitch - Whether to maintain original pitch
   * @returns Processed AudioBuffer
   */
  processConstantSpeed(
    buffer: AudioBuffer,
    speed: number,
    preservePitch: boolean
  ): Promise<AudioBuffer>;

  /**
   * Process audio with speed keyframes
   * @param buffer - Source AudioBuffer
   * @param keyframes - Speed keyframes array
   * @param defaultSpeed - Default speed if no keyframes
   * @param preservePitch - Whether to maintain original pitch
   * @returns Processed AudioBuffer with variable speed applied
   */
  processWithKeyframes(
    buffer: AudioBuffer,
    keyframes: Keyframe[],
    defaultSpeed: number,
    preservePitch: boolean
  ): Promise<AudioBuffer>;
}
```

**Algorithm for Keyframed Speed**:

```
Input: AudioBuffer (original), SpeedKeyframes[], preservePitch

1. Calculate output duration:
   - Integrate speed curve to find total source time consumed
   - outputDuration = timelineDuration (clip.duration)

2. Create output buffer with outputDuration

3. For each output sample position (t_out):
   a. Calculate corresponding source position (t_src):
      - t_src = integral of speed from 0 to t_out
   b. If preservePitch:
      - Use SoundTouch time-stretch at current speed
   c. Else:
      - Simple resampling (pitch shifts with speed)

4. Return processed buffer
```

**SoundTouch Integration**:
```typescript
import { SoundTouch } from 'soundtouch-ts';

const soundtouch = new SoundTouch();
soundtouch.tempo = speed;      // Time stretch
soundtouch.pitch = 1.0;        // Keep pitch (if preservePitch)
// or
soundtouch.rate = speed;       // Change both tempo and pitch
```

**Chunk Processing** (for memory efficiency):
```
- Process in 10-second chunks
- Overlap-add for seamless joins
- Progress callback for UI updates
```

---

### 3. AudioEffectRenderer

**Purpose**: Apply EQ and volume effects with keyframe automation

**Location**: `src/engine/audio/AudioEffectRenderer.ts`

```typescript
interface AudioEffectRenderer {
  /**
   * Render all effects for a clip
   * @param buffer - Source AudioBuffer (already speed-processed)
   * @param effects - Array of effects (audio-eq, audio-volume)
   * @param keyframes - All keyframes for this clip
   * @param clipDuration - Duration for automation
   * @returns Processed AudioBuffer
   */
  renderEffects(
    buffer: AudioBuffer,
    effects: Effect[],
    keyframes: Keyframe[],
    clipDuration: number
  ): Promise<AudioBuffer>;
}
```

**Implementation using OfflineAudioContext**:

```typescript
async function renderEffects(buffer, effects, keyframes, duration) {
  const offline = new OfflineAudioContext(
    buffer.numberOfChannels,
    buffer.sampleRate * duration,
    buffer.sampleRate
  );

  // Source
  const source = offline.createBufferSource();
  source.buffer = buffer;

  // EQ Chain (10 bands)
  const eqFilters = EQ_FREQUENCIES.map(freq => {
    const filter = offline.createBiquadFilter();
    filter.type = 'peaking';
    filter.frequency.value = freq;
    filter.Q.value = 1.4;
    return filter;
  });

  // Automate EQ gains from keyframes
  eqFilters.forEach((filter, i) => {
    const paramKeyframes = keyframes.filter(
      k => k.property === `effect.${eqEffectId}.band${EQ_FREQUENCIES[i]}`
    );
    automateParam(filter.gain, paramKeyframes, duration);
  });

  // Volume/Gain
  const gainNode = offline.createGain();
  const volumeKeyframes = keyframes.filter(
    k => k.property === `effect.${volumeEffectId}.volume`
  );
  automateParam(gainNode.gain, volumeKeyframes, duration);

  // Connect chain
  source.connect(eqFilters[0]);
  for (let i = 0; i < eqFilters.length - 1; i++) {
    eqFilters[i].connect(eqFilters[i + 1]);
  }
  eqFilters[eqFilters.length - 1].connect(gainNode);
  gainNode.connect(offline.destination);

  // Render
  source.start(0);
  return await offline.startRendering();
}
```

**Keyframe Automation**:
```typescript
function automateParam(param: AudioParam, keyframes: Keyframe[], duration: number) {
  if (keyframes.length === 0) return;

  // Sort by time
  const sorted = [...keyframes].sort((a, b) => a.time - b.time);

  // Set initial value
  param.setValueAtTime(sorted[0].value, 0);

  // Automate between keyframes
  for (let i = 1; i < sorted.length; i++) {
    const kf = sorted[i];
    const prevKf = sorted[i - 1];

    switch (kf.easing) {
      case 'linear':
        param.linearRampToValueAtTime(kf.value, kf.time);
        break;
      case 'ease-in':
      case 'ease-out':
      case 'ease-in-out':
        // Approximate with exponential
        param.exponentialRampToValueAtTime(Math.max(0.0001, kf.value), kf.time);
        break;
      case 'bezier':
        // Sample bezier curve and set multiple points
        automateBezier(param, prevKf, kf);
        break;
      default:
        param.setValueAtTime(kf.value, kf.time);
    }
  }
}
```

---

### 4. AudioMixer

**Purpose**: Mix multiple audio tracks into single stereo output

**Location**: `src/engine/audio/AudioMixer.ts`

```typescript
interface AudioTrackData {
  clipId: string;
  buffer: AudioBuffer;        // Already processed (speed, effects)
  startTime: number;          // Position on timeline
  trackId: string;
  trackMuted: boolean;
  trackSolo: boolean;
}

interface AudioMixer {
  /**
   * Mix all tracks into single output
   * @param tracks - Array of processed audio tracks
   * @param duration - Total timeline duration
   * @param sampleRate - Output sample rate (48000)
   * @returns Mixed stereo AudioBuffer
   */
  mixTracks(
    tracks: AudioTrackData[],
    duration: number,
    sampleRate: number
  ): Promise<AudioBuffer>;
}
```

**Implementation**:

```typescript
async function mixTracks(tracks, duration, sampleRate) {
  const numSamples = Math.ceil(duration * sampleRate);
  const offline = new OfflineAudioContext(2, numSamples, sampleRate);

  // Determine which tracks to include (handle solo)
  const hasSolo = tracks.some(t => t.trackSolo);
  const activeTracks = tracks.filter(t => {
    if (t.trackMuted) return false;
    if (hasSolo && !t.trackSolo) return false;
    return true;
  });

  // Create source for each track at correct position
  for (const track of activeTracks) {
    const source = offline.createBufferSource();
    source.buffer = track.buffer;
    source.connect(offline.destination);
    source.start(track.startTime);
  }

  return await offline.startRendering();
}
```

**Overlap Handling**:
- OfflineAudioContext automatically sums overlapping audio
- No special handling needed for crossfades (automatic summing)

**Stereo Positioning** (future enhancement):
- Pan control per track
- Stereo width adjustment

---

### 5. AudioEncoder

**Purpose**: Encode AudioBuffer to AAC using WebCodecs

**Location**: `src/engine/audio/AudioEncoder.ts`

```typescript
interface AudioEncoderWrapper {
  /**
   * Initialize encoder
   * @param sampleRate - Input sample rate
   * @param channels - Number of channels (1 or 2)
   * @param bitrate - Target bitrate (128000-320000)
   */
  init(sampleRate: number, channels: number, bitrate: number): Promise<boolean>;

  /**
   * Encode AudioBuffer to AAC chunks
   * @param buffer - Mixed AudioBuffer
   * @param onChunk - Callback for encoded chunks
   */
  encode(
    buffer: AudioBuffer,
    onChunk: (chunk: EncodedAudioChunk, meta: EncodedAudioChunkMetadata) => void
  ): Promise<void>;

  /**
   * Flush remaining data and close
   */
  finalize(): Promise<void>;
}
```

**Implementation**:

```typescript
class AudioEncoderWrapper {
  private encoder: AudioEncoder | null = null;

  async init(sampleRate: number, channels: number, bitrate: number) {
    // Check AAC support
    const config = {
      codec: 'mp4a.40.2',  // AAC-LC
      sampleRate,
      numberOfChannels: channels,
      bitrate,
    };

    const support = await AudioEncoder.isConfigSupported(config);
    if (!support.supported) {
      throw new Error('AAC encoding not supported');
    }

    this.encoder = new AudioEncoder({
      output: this.handleChunk.bind(this),
      error: this.handleError.bind(this),
    });

    await this.encoder.configure(config);
  }

  async encode(buffer: AudioBuffer, onChunk: ChunkCallback) {
    this.onChunk = onChunk;

    // Convert AudioBuffer to AudioData chunks
    const chunkSize = 1024;  // AAC frame size
    const totalSamples = buffer.length;

    for (let offset = 0; offset < totalSamples; offset += chunkSize) {
      const samples = Math.min(chunkSize, totalSamples - offset);

      // Interleave channels for AudioData
      const interleaved = this.interleaveChannels(buffer, offset, samples);

      const audioData = new AudioData({
        format: 'f32-planar',
        sampleRate: buffer.sampleRate,
        numberOfFrames: samples,
        numberOfChannels: buffer.numberOfChannels,
        timestamp: (offset / buffer.sampleRate) * 1_000_000,  // microseconds
        data: interleaved,
      });

      this.encoder.encode(audioData);
      audioData.close();
    }
  }

  async finalize() {
    await this.encoder.flush();
    this.encoder.close();
  }
}
```

**Fallback for unsupported browsers**:
- Check `AudioEncoder` availability
- Fall back to uncompressed PCM in WebM container
- Or show warning that audio export not supported

---

### 6. MP4 Muxer Integration

**Purpose**: Add audio track to existing video export

**Location**: Modify `src/engine/FrameExporter.ts`

```typescript
// Updated Muxer configuration
this.muxer = new Muxer({
  target: new ArrayBufferTarget(),
  video: {
    codec: settings.codec === 'h264' ? 'avc' : 'vp9',
    width: settings.width,
    height: settings.height,
  },
  audio: {
    codec: 'aac',
    sampleRate: 48000,
    numberOfChannels: 2,
  },
  fastStart: 'in-memory',
});

// Add audio chunks during export
muxer.addAudioChunk(chunk, meta);
```

---

### 7. AudioExportPipeline (Orchestrator)

**Purpose**: Coordinate all components for export

**Location**: `src/engine/audio/AudioExportPipeline.ts`

```typescript
interface ExportProgress {
  phase: 'extracting' | 'processing' | 'effects' | 'mixing' | 'encoding';
  percent: number;
  currentClip?: string;
}

class AudioExportPipeline {
  private extractor: AudioExtractor;
  private timeStretch: TimeStretchProcessor;
  private effectRenderer: AudioEffectRenderer;
  private mixer: AudioMixer;
  private encoder: AudioEncoderWrapper;

  /**
   * Export all audio from timeline
   * @param onProgress - Progress callback
   * @returns Encoded audio chunks for muxing
   */
  async exportAudio(
    onProgress: (progress: ExportProgress) => void
  ): Promise<EncodedAudioChunk[]> {
    const { clips, tracks, clipKeyframes, duration } = useTimelineStore.getState();

    // 1. Find all clips with audio
    const audioClips = this.getClipsWithAudio(clips, tracks);

    // 2. Extract audio from each clip
    onProgress({ phase: 'extracting', percent: 0 });
    const extractedBuffers = await this.extractAll(audioClips, onProgress);

    // 3. Process speed/pitch for each clip
    onProgress({ phase: 'processing', percent: 0 });
    const processedBuffers = await this.processSpeed(
      audioClips,
      extractedBuffers,
      clipKeyframes,
      onProgress
    );

    // 4. Render effects for each clip
    onProgress({ phase: 'effects', percent: 0 });
    const effectBuffers = await this.renderAllEffects(
      audioClips,
      processedBuffers,
      clipKeyframes,
      onProgress
    );

    // 5. Mix all tracks
    onProgress({ phase: 'mixing', percent: 0 });
    const mixedBuffer = await this.mixer.mixTracks(
      this.prepareTrackData(audioClips, effectBuffers, tracks),
      duration,
      48000
    );

    // 6. Encode to AAC
    onProgress({ phase: 'encoding', percent: 0 });
    const chunks: EncodedAudioChunk[] = [];
    await this.encoder.encode(mixedBuffer, (chunk, meta) => {
      chunks.push(chunk);
    });
    await this.encoder.finalize();

    // 7. Cleanup
    this.extractor.clearCache();

    return chunks;
  }
}
```

---

## File Structure

```
src/
├── engine/
│   ├── audio/
│   │   ├── index.ts                 # Exports AudioExportPipeline
│   │   ├── AudioExtractor.ts        # Decode audio from files
│   │   ├── TimeStretchProcessor.ts  # Speed/pitch with SoundTouch
│   │   ├── AudioEffectRenderer.ts   # EQ, volume with keyframes
│   │   ├── AudioMixer.ts            # Multi-track mixing
│   │   └── AudioEncoder.ts          # WebCodecs AAC encoding
│   │
│   └── FrameExporter.ts             # Modified: add audio track
│
├── components/
│   └── export/
│       └── ExportDialog.tsx         # Modified: audio options UI
│
└── types/
    └── index.ts                     # Add audio export types
```

---

## Export Settings UI

Add to ExportDialog.tsx:

```typescript
// Audio export options
interface AudioExportSettings {
  includeAudio: boolean;      // Default: true
  audioCodec: 'aac';          // Only AAC for now
  audioBitrate: number;       // 128, 192, 256, 320 kbps
  sampleRate: number;         // 44100, 48000
  normalizeAudio: boolean;    // Peak normalize to -1dB (future)
}
```

**UI Mockup**:
```
┌─────────────────────────────────────────┐
│ Export Settings                         │
├─────────────────────────────────────────┤
│                                         │
│ Video                                   │
│ ├── Resolution: [1920x1080 ▼]          │
│ ├── Frame Rate: [30 ▼]                 │
│ └── Bitrate: [15 Mbps]                 │
│                                         │
│ Audio                                   │
│ ├── [✓] Include Audio                  │
│ ├── Bitrate: [256 kbps ▼]              │
│ └── Sample Rate: [48000 Hz ▼]          │
│                                         │
│ [Cancel]                    [Export]    │
└─────────────────────────────────────────┘
```

---

## Quality Settings

### Preview (Real-time)
| Parameter | Value | Notes |
|-----------|-------|-------|
| Speed Range | 0.25x - 4x | Browser limitation |
| Pitch | Browser native | Good quality |
| EQ | Real-time | Some latency |
| Sample Rate | Native | From source |

### Export (Offline)
| Parameter | Value | Notes |
|-----------|-------|-------|
| Speed Range | 0.1x - 10x | SoundTouch handles wider range |
| Pitch | SoundTouchJS | High quality WSOLA |
| EQ | Offline render | Sample-accurate |
| Sample Rate | 48000 Hz | Standard for video |
| Bit Depth | 32-bit float | Internal processing |
| Output Codec | AAC-LC | Universal compatibility |
| Output Bitrate | 256 kbps | High quality |

---

## Dependencies

| Package | Version | Size | Purpose |
|---------|---------|------|---------|
| `soundtouch-ts` | ^1.0.0 | ~15KB | Time-stretch/pitch |
| `mp4-muxer` | existing | - | Already in project |
| Web Audio API | native | - | Effects, mixing |
| WebCodecs | native | - | AAC encoding |

**Install**:
```bash
npm install soundtouch-ts
```

---

## Progress Reporting

Export progress shown in UI:

```
Phase 1: Extracting Audio     [████████░░░░░░░░] 50%
         Processing: clip_001.mp4

Phase 2: Processing Speed     [████░░░░░░░░░░░░] 25%
         Clip: interview_main

Phase 3: Rendering Effects    [██████████░░░░░░] 65%
         Applying EQ...

Phase 4: Mixing Tracks        [████████████░░░░] 80%
         4 tracks

Phase 5: Encoding Audio       [██████████████░░] 90%
         AAC @ 256kbps

Phase 6: Muxing               [████████████████] 100%
         Finalizing MP4...
```

---

## Error Handling

| Error | Cause | Resolution |
|-------|-------|------------|
| `AudioDecodingError` | Corrupt file | Skip clip, warn user |
| `EncoderNotSupported` | Old browser | Fall back to no audio, warn |
| `OutOfMemory` | Very long export | Process in chunks |
| `TimeStretchError` | Invalid speed | Clamp to valid range |

```typescript
class AudioExportError extends Error {
  constructor(
    message: string,
    public phase: string,
    public clipId?: string,
    public recoverable: boolean = false
  ) {
    super(message);
  }
}
```

---

## Memory Management

**Concerns**:
- AudioBuffer for 10-minute stereo @ 48kHz = ~230MB
- Multiple clips = potential GB of memory

**Strategies**:
1. **Process sequentially**: One clip at a time
2. **Clear buffers**: Release memory after each phase
3. **Chunk encoding**: Don't hold entire encoded output
4. **Cache limits**: Max 5 decoded files cached

```typescript
// Memory cleanup after each clip
async processClip(clip) {
  const buffer = await this.extractor.extractAudio(clip.file);
  const processed = await this.timeStretch.process(buffer);
  buffer = null;  // Allow GC

  const withEffects = await this.effectRenderer.render(processed);
  processed = null;  // Allow GC

  return withEffects;
}
```

---

## Testing Plan

### Unit Tests
- [ ] AudioExtractor: decode various formats
- [ ] TimeStretchProcessor: constant speed, keyframes
- [ ] AudioEffectRenderer: EQ bands, volume automation
- [ ] AudioMixer: overlapping clips, mute/solo
- [ ] AudioEncoder: AAC output validity

### Integration Tests
- [ ] Full export with single audio clip
- [ ] Export with multiple overlapping clips
- [ ] Export with speed keyframes
- [ ] Export with EQ keyframes
- [ ] Very long export (30+ minutes)

### Manual Tests
- [ ] Listen to exported audio quality
- [ ] Compare speed-changed audio with original
- [ ] Verify sync with video
- [ ] Test in various players (VLC, QuickTime, browser)

---

## Implementation Order

| Phase | Component | Dependencies | Est. Effort |
|-------|-----------|--------------|-------------|
| 1 | AudioExtractor | None | 2 hours |
| 2 | AudioEncoder | mp4-muxer types | 3 hours |
| 3 | AudioMixer | AudioExtractor | 2 hours |
| 4 | TimeStretchProcessor | soundtouch-ts | 4 hours |
| 5 | AudioEffectRenderer | None | 3 hours |
| 6 | AudioExportPipeline | All above | 4 hours |
| 7 | FrameExporter integration | Pipeline | 3 hours |
| 8 | ExportDialog UI | Pipeline | 2 hours |
| 9 | Testing & fixes | All | 4 hours |
| **Total** | | | **~27 hours** |

---

## Future Enhancements

### Phase 2 (Later)
- [ ] Audio compression/limiter effect
- [ ] Reverb/delay effects
- [ ] Loudness normalization (LUFS)
- [ ] Audio meters during export
- [ ] Waveform preview of export

### Phase 3 (Future)
- [ ] Multi-format export (MP3, WAV, FLAC)
- [ ] Surround sound (5.1)
- [ ] Audio-only export
- [ ] Batch export presets

---

## References

- [WebCodecs AudioEncoder](https://developer.mozilla.org/en-US/docs/Web/API/AudioEncoder)
- [Web Audio API OfflineAudioContext](https://developer.mozilla.org/en-US/docs/Web/API/OfflineAudioContext)
- [SoundTouch Audio Processing](https://www.surina.net/soundtouch/)
- [soundtouch-ts](https://github.com/AnthumChris/soundtouch-ts)
- [mp4-muxer](https://github.com/nicksam112/mp4-muxer)
- [AAC Audio Codec](https://wiki.multimedia.cx/index.php/AAC)

---

*Last Updated: 2026-01-10*
*Status: Planning*
