# AI Multicam Editor

Automatischer Multicam-Schnitt via LLM + Computer Vision Analyse.

---

## Konzept

Das LLM bekommt keine Bilder, sondern nur extrahierte Metadaten als Kurven/Graphen plus Transcript. Basierend darauf erstellt es einen Schnittplan (EDL).

```
┌─────────────────────────────────────────────────────────────┐
│                    Video Input (4 Kameras)                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Proxy Generation + CV Analysis                  │
│              (Ein Durchlauf, alles parallel)                 │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐   │
│  │ Bewegung  │ │ Schärfe   │ │ Audio     │ │ Gesichter │   │
│  │ (WebGPU)  │ │ (WebGPU)  │ │ (WebAudio)│ │ (WebGPU)  │   │
│  └───────────┘ └───────────┘ └───────────┘ └───────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Metadaten pro Zeitpunkt                    │
│  {                                                          │
│    timestamp: 00:00:05,                                     │
│    cameras: [                                               │
│      { id: 1, motion: 0.1, sharpness: 0.8, faces: [...] }, │
│      { id: 2, motion: 0.0, sharpness: 0.9, faces: [...] }, │
│      ...                                                    │
│    ],                                                       │
│    audio: { speaker: "Anna", level: 0.7 },                 │
│    transcript: "Also ich denke dass..."                    │
│  }                                                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        LLM (Claude)                          │
│  Input:  Metadaten-Stream + Transcript + Schnitt-Regeln     │
│  Output: EDL (Edit Decision List)                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Timeline / Export                       │
│  • Premiere XML                                             │
│  • DaVinci Resolve EDL                                      │
│  • Direkte Preview im WebVJ Mixer                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Proxy + Analysis Workflow

**Kernidee:** Während das Video für Proxy-Erstellung durchläuft, werden ALLE CV-Analysen parallel auf der GPU gemacht. Ein Durchlauf = Proxy + alle Metadaten.

```
Video Frame
     │
     ├──► WebCodecs Decode
     │         │
     │         ├──► Proxy Encode (kleinere Auflösung)
     │         │
     │         └──► GPUExternalTexture
     │                    │
     │    ┌───────────────┼───────────────┐
     │    │               │               │
     │    ▼               ▼               ▼
     │  Motion         Sharpness       Face Det.
     │  Compute        Compute         (TF.js)
     │  Shader         Shader          
     │    │               │               │
     │    └───────────────┴───────────────┘
     │                    │
     │                    ▼
     │              Metadata Buffer
     │                    │
     └──────────────────► Speichern als .json neben Proxy
```

**Vorteile:**
- Nur 1x durch das Video gehen
- GPU macht alles parallel
- Zero-Copy: VideoFrame → GPUTexture direkt
- Proxy + Metadaten sind synchron

---

## CV Analysis Module (Alle GPU-basiert)

### Motion Detection (WebGPU Compute Shader)

Optical Flow ist overkill. Einfache Frame-Differenz reicht.

```wgsl
// motion.compute.wgsl
@group(0) @binding(0) var prevFrame: texture_2d<f32>;
@group(0) @binding(1) var currFrame: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> result: f32;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let prev = textureLoad(prevFrame, id.xy, 0);
    let curr = textureLoad(currFrame, id.xy, 0);
    
    // Grayscale Luminance
    let prevLum = dot(prev.rgb, vec3(0.299, 0.587, 0.114));
    let currLum = dot(curr.rgb, vec3(0.299, 0.587, 0.114));
    
    // Absolute Differenz, atomicAdd für Summe
    let diff = abs(currLum - prevLum);
    atomicAdd(&result, diff);
}
```

**Output:** Single float 0-1, normalisiert über Pixelanzahl.

### Sharpness Detection (WebGPU Compute Shader)

Laplacian Kernel → Varianz = Schärfe.

```wgsl
// sharpness.compute.wgsl
@group(0) @binding(0) var frame: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> variance: f32;
@group(0) @binding(2) var<storage, read_write> count: u32;

// Laplacian Kernel: [[0,-1,0],[-1,4,-1],[0,-1,0]]
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let c = textureLoad(frame, id.xy, 0).rgb;
    let t = textureLoad(frame, id.xy + vec2(0, -1), 0).rgb;
    let b = textureLoad(frame, id.xy + vec2(0, 1), 0).rgb;
    let l = textureLoad(frame, id.xy + vec2(-1, 0), 0).rgb;
    let r = textureLoad(frame, id.xy + vec2(1, 0), 0).rgb;
    
    let lap = 4.0 * c - t - b - l - r;
    let lum = dot(lap, vec3(0.299, 0.587, 0.114));
    
    atomicAdd(&variance, lum * lum);
    atomicAdd(&count, 1u);
}
```

**Output:** Varianz-Wert, höher = schärfer.

### Audio Analysis (Web Audio API)

Kein GPU nötig, läuft parallel im AudioContext.

```typescript
class AudioAnalyzer {
  private context: AudioContext;
  private analyser: AnalyserNode;
  
  async analyze(audioBuffer: AudioBuffer): Promise<AudioCurve> {
    const data = audioBuffer.getChannelData(0);
    const sampleRate = audioBuffer.sampleRate;
    const windowSize = sampleRate / 10; // 100ms windows
    
    const curve: AudioCurve = [];
    for (let i = 0; i < data.length; i += windowSize) {
      const window = data.slice(i, i + windowSize);
      const rms = Math.sqrt(window.reduce((a, b) => a + b * b, 0) / window.length);
      curve.push({
        timestamp: (i / sampleRate) * 1000,
        level: rms
      });
    }
    return curve;
  }
}
```

### Face Detection (TensorFlow.js + WebGPU Backend)

**Beste Option:** MediaPipe Face Detection via TensorFlow.js mit WebGPU Backend.

```typescript
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgpu';
import * as faceDetection from '@tensorflow-models/face-detection';

class FaceAnalyzer {
  private detector: faceDetection.FaceDetector | null = null;
  
  async init() {
    // WebGPU Backend aktivieren
    await tf.setBackend('webgpu');
    await tf.ready();
    
    // BlazeFace ist am schnellsten
    this.detector = await faceDetection.createDetector(
      faceDetection.SupportedModels.MediaPipeFaceDetector,
      {
        runtime: 'tfjs',
        maxFaces: 10
      }
    );
  }
  
  async detect(frame: VideoFrame): Promise<Face[]> {
    // VideoFrame → ImageBitmap → TF Input
    const bitmap = await createImageBitmap(frame);
    const faces = await this.detector!.estimateFaces(bitmap);
    bitmap.close();
    
    return faces.map(f => ({
      bbox: f.box,
      keypoints: f.keypoints,
      confidence: f.box.score ?? 1
    }));
  }
}
```

**Alternativen:**
- **BlazeFace** (schnellste, ~190KB Model)
- **MediaPipe Face Mesh** (478 Landmarks, für detaillierte Analyse)
- **face-api.js** (älter, aber bewährt, nutzt WebGL)

**Performance-Vergleich:**
| Model | Size | Speed (M1) | Accuracy |
|-------|------|------------|----------|
| BlazeFace | 190KB | ~4ms | Gut |
| SSD MobileNet | 5.4MB | ~20ms | Sehr gut |
| MediaPipe Face Mesh | 3MB | ~10ms | Exzellent |

**Empfehlung:** BlazeFace für Multicam (schnell genug für 4 Kameras @ 30fps)

---

## Datenstruktur für LLM

```typescript
interface MultiCamAnalysis {
  // Projekt-Info
  project: {
    duration: number;           // ms
    cameras: CameraInfo[];
    speakers: SpeakerInfo[];
  };
  
  // Zeitbasierte Daten (alle X ms gesamplet)
  timeline: TimelineEntry[];
  
  // Vollständiges Transcript
  transcript: TranscriptEntry[];
}

interface CameraInfo {
  id: number;
  name: string;           // "Totale", "Person Links", etc.
  defaultUse: string;     // "wide", "closeup", "detail"
}

interface SpeakerInfo {
  id: string;
  name: string;
  preferredCamera?: number;
}

interface TimelineEntry {
  timestamp: number;
  cameras: CameraState[];
  audio: {
    speaker: string | null;
    level: number;
  };
}

interface CameraState {
  id: number;
  motion: number;         // 0-1
  sharpness: number;      // 0-1
  faces: {
    id: string;
    size: number;         // Wie groß im Bild
    position: string;     // "center", "left", "right"
  }[];
}

interface TranscriptEntry {
  start: number;          // ms
  end: number;
  speaker: string;
  text: string;
}
```

---

## LLM Prompt Struktur

```
Du bist ein erfahrener Video-Editor. 

PROJEKT:
- 4 Kameras: K1 (Totale), K2 (Anna Closeup), K3 (Tom Closeup), K4 (Detail/Hände)
- 2 Sprecher: Anna, Tom
- Dauer: 5:32

REGELN:
- Schneide auf den Sprecher wenn möglich
- Nutze Reaktionen sparsam aber effektiv
- Totale bei Themenwechsel oder längeren Pausen
- Vermeide unscharfe Kameras (sharpness < 0.5)
- Mindestens 2 Sekunden pro Schnitt

METADATEN:
[Timeline-Daten als JSON oder kompaktes Format]

TRANSCRIPT:
[Vollständiger Text mit Timestamps und Sprechern]

Erstelle einen Schnittplan im Format:
START - END | KAMERA | BEGRÜNDUNG
```

---

## Output Format (EDL)

```typescript
interface EditDecision {
  start: number;          // ms
  end: number;
  camera: number;
  reason?: string;        // Optional: Warum diese Entscheidung
}

type EDL = EditDecision[];
```

**Export-Formate:**

```
// Premiere Pro XML
<sequence>
  <clipitem>
    <start>0</start>
    <end>150</end>
    <file id="camera_1"/>
  </clipitem>
  ...
</sequence>

// DaVinci Resolve EDL
001  CAM1  V  C  00:00:00:00 00:00:05:00 00:00:00:00 00:00:05:00
002  CAM2  V  C  00:00:05:00 00:00:12:00 00:00:05:00 00:00:12:00
```

---

## Integration in WebVJ Mixer

### Neuer Store: `multicamStore.ts`

```typescript
interface MultiCamStore {
  // Projekt
  cameras: MultiCamSource[];
  analysis: MultiCamAnalysis | null;
  edl: EDL | null;
  
  // Status
  analyzing: boolean;
  generating: boolean;
  
  // Actions
  addCamera(file: File, name: string): void;
  analyzeAll(): Promise<void>;
  generateEDL(prompt?: string): Promise<void>;
  exportEDL(format: 'premiere' | 'resolve'): void;
  
  // Preview
  previewEDL(): void;  // Spielt EDL im Mixer ab
}
```

### Neues Panel: `MultiCamPanel.tsx`

```
┌─────────────────────────────────────────────────────────┐
│ MULTICAM EDITOR                                          │
├─────────────────────────────────────────────────────────┤
│ Cameras:                                                │
│ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                       │
│ │ K1  │ │ K2  │ │ K3  │ │ K4  │  [+ Add Camera]       │
│ │Totale│ │Anna │ │Tom  │ │Detail│                      │
│ └─────┘ └─────┘ └─────┘ └─────┘                       │
├─────────────────────────────────────────────────────────┤
│ Analysis:                        [Analyze All]          │
│ ┌─────────────────────────────────────────────────────┐│
│ │ Motion  ▁▂▃▅▂▁▁▃▅▇▅▃▂▁▁▂▃▂▁                        ││
│ │ Sharp   ▇▇▇▆▇▇▇▅▆▇▇▇▇▇▇▆▇▇▇                        ││
│ │ Audio   ▁▃▅▇▅▃▁▁▃▅▇▇▅▃▁▃▅▇▅                        ││
│ └─────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────┤
│ Transcript:                      [Import SRT/VTT]       │
│ 00:00 Anna: Also ich denke dass...                     │
│ 00:15 Tom: Ja genau, und wenn man...                   │
├─────────────────────────────────────────────────────────┤
│ Style: [Podcast ▼]  [Generate Edit]                    │
├─────────────────────────────────────────────────────────┤
│ EDL Preview:                                            │
│ ┌──K1──┬──K2──────┬─K3─┬──K2────┬──K1──┐              │
│ 0:00   0:05      0:15 0:18     0:30   0:35             │
│                                                         │
│ [Export Premiere] [Export Resolve] [Preview]           │
└─────────────────────────────────────────────────────────┘
```

---

## Implementation Steps

### Phase 1: CV Pipeline

1. `MotionAnalyzer.ts` - Frame-Differenz basierte Bewegungserkennung
2. `SharpnessAnalyzer.ts` - Laplacian Variance
3. `AudioAnalyzer.ts` - Web Audio API für Pegel
4. `FaceAnalyzer.ts` - face-api.js Integration
5. `AnalysisCombiner.ts` - Alles zusammenführen

### Phase 2: LLM Integration

1. `TranscriptParser.ts` - SRT/VTT Import
2. `PromptBuilder.ts` - Metadaten → LLM Prompt
3. `EDLParser.ts` - LLM Output → EDL
4. `ClaudeAPI.ts` - API Calls (oder lokales LLM)

### Phase 3: UI + Export

1. `MultiCamPanel.tsx` - Haupt-UI
2. `AnalysisCurves.tsx` - Visualisierung der Kurven
3. `EDLTimeline.tsx` - Edit Preview
4. `ExportFormats.ts` - Premiere/Resolve Export

---

## Offene Fragen

- [ ] Sample-Rate für Analyse? (alle 500ms? 1s? 2s?)
- [ ] Lokales LLM vs Claude API?
- [ ] Face Recognition Training für bekannte Sprecher?
- [ ] Wie mit sehr langen Videos umgehen? (Chunking?)

---

## Dependencies

```json
{
  "@tensorflow/tfjs": "^4.x",
  "@tensorflow/tfjs-backend-webgpu": "^4.x",
  "@tensorflow-models/face-detection": "^1.x",
  "srt-parser-2": "^1.2.3"
}
```

Minimal gehalten. TensorFlow.js mit WebGPU Backend für Face Detection, Rest ist pure WebGPU/WebCodecs.

---

## Verbesserungen gegenüber V1

### 1. Single-Pass Processing
- **Alt:** Separates Durchlaufen für jede Analyse
- **Neu:** Ein Durchlauf = Proxy + alle CV-Daten
- **Vorteil:** 4x schneller bei 4 Analyse-Typen

### 2. GPU-First Design
- **Alt:** CPU-basierte CV mit JS Libraries
- **Neu:** WebGPU Compute Shaders für Motion/Sharpness
- **Vorteil:** 10-50x schneller, läuft parallel zum Decode

### 3. Zero-Copy Pipeline
- **Alt:** VideoFrame → Canvas → ImageData → Analysis
- **Neu:** VideoFrame → GPUExternalTexture → Compute → Result
- **Vorteil:** Keine CPU/GPU Transfers

### 4. Chunked LLM Processing
Für lange Videos (>30min):

```typescript
interface AnalysisChunk {
  startTime: number;
  endTime: number;
  summary: {
    dominantSpeaker: string;
    avgMotion: Record<number, number>;  // per camera
    topicKeywords: string[];
  };
  rawData: TimelineEntry[];  // Detailed data für diesen Chunk
}

// LLM bekommt:
// 1. Summaries aller Chunks
// 2. Detailed data nur für aktuellen Edit-Bereich
```

### 5. Face Clustering
Automatisches Gruppieren von Gesichtern ohne manuelle Labels:

```typescript
class FaceClusterer {
  private embeddings: Map<string, Float32Array> = new Map();
  
  async cluster(faces: DetectedFace[]): Promise<FaceCluster[]> {
    // Face Embeddings extrahieren (FaceNet-style)
    // DBSCAN oder K-Means Clustering
    // Resultat: Gruppen von "gleichen" Gesichtern
  }
}
```

**Vorteil:** Keine manuelle "Das ist Anna" Zuweisung nötig. System erkennt "Person A spricht in Kamera 2".

### 6. Confidence Scoring für Schnittentscheidungen

```typescript
interface CutDecision {
  timestamp: number;
  fromCamera: number;
  toCamera: number;
  confidence: number;  // 0-1
  reasons: string[];
  alternatives: Array<{
    camera: number;
    confidence: number;
  }>;
}
```

LLM gibt Confidence mit. UI kann Low-Confidence Cuts markieren für manuelle Review.

---

## Potential Future Features

- [ ] **Whisper Integration:** Lokale Transcription wenn kein SRT vorhanden
- [ ] **Beat Detection:** Schnitte auf Musik-Beats für dynamischere Edits
- [ ] **Style Transfer:** "Schneide wie ein Podcast" vs "Schneide wie ein Interview"
- [ ] **Live Multicam:** Echtzeit-Switching während Aufnahme
- [ ] **LLM Fine-tuning:** Auf eigenen Schnitt-Stil trainieren

---

## Tests

| Test File | Tests | Coverage |
|-----------|-------|----------|
| [`crossCorrelation.test.ts`](../../tests/unit/crossCorrelation.test.ts) | 6 | Audio sync cross-correlation |

Run tests: `npx vitest run`
