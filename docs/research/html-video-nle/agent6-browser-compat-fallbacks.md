# Agent 6: Browser-Kompatibilität, Codec-Support & Fallback-Strategien

## 1. Die Decoder-Fallback-Kette in MasterSelects

MasterSelects implementiert eine dreistufige Fallback-Kette für Video-Decoding.
Die Priorität wird im `LayerCollector` (`src/engine/render/LayerCollector.ts`) bestimmt:

```
1. NativeHelper (FFmpeg/Rust)  → ImageBitmap → texture_2d<f32>
2. WebCodecs (VideoDecoder)    → VideoFrame  → texture_external
3. HTMLVideoElement            → video elem  → texture_external
```

Konkret im Code (`collectLayerData`, ab Zeile 198):

```typescript
// 1. Try Native Helper decoder (turbo mode) - most efficient
if (source.nativeDecoder) {
  const bitmap = source.nativeDecoder.getCurrentFrame();
  // ... creates ImageBitmap texture
  this.currentDecoder = 'NativeHelper';
}

// 2. Try direct VideoFrame (parallel decoder)
if (source.videoFrame) {
  const extTex = deps.textureManager.importVideoTexture(frame);
  this.currentDecoder = 'ParallelDecode';
}

// 3. Try full WebCodecs VideoFrame
// ... frameProvider.getCurrentFrame()
this.currentDecoder = 'WebCodecs';

// 4. HTMLVideoElement (fallback)
const extTex = deps.textureManager.importVideoTexture(video);
this.currentDecoder = 'HTMLVideo';
```

Der `currentDecoder`-Wert wird in den Stats angezeigt und kann via AI Debug Tools abgefragt werden.

---

## 2. Feature Detection im Codebase

### 2.1 WebCodecs API Check

In `src/stores/timeline/helpers/webCodecsHelpers.ts`:

```typescript
export function hasWebCodecsSupport(): boolean {
  return 'VideoDecoder' in window && 'VideoFrame' in window;
}
```

Wird vor jeder WebCodecs-Initialisierung aufgerufen. Wenn `false`, fällt der gesamte Clip auf HTMLVideoElement zurück.

### 2.2 VideoDecoder.isConfigSupported()

Wird an mehreren Stellen genutzt:

**Export-Encoder** (`src/engine/export/codecHelpers.ts`):
```typescript
export async function checkCodecSupport(
  codec: VideoCodec, width: number, height: number
): Promise<boolean> {
  if (!('VideoEncoder' in window)) return false;
  const support = await VideoEncoder.isConfigSupported({
    codec: getCodecString(codec), width, height,
    bitrate: 10_000_000, framerate: 30,
  });
  return support.supported ?? false;
}
```

**Proxy-Generator** (`src/services/proxyGenerator.ts`) mit Fallback-Logik:
```typescript
private async findSupportedCodec(
  baseCodec: string, width: number, height: number, description?: Uint8Array
): Promise<VideoDecoderConfig | null> {
  const h264Fallbacks = [
    baseCodec,
    'avc1.42001e', 'avc1.4d001e', 'avc1.64001e',
    'avc1.640028', 'avc1.4d0028', 'avc1.42E01E',
    'avc1.4D401E', 'avc1.640029',
  ];
  // Tries each codec string until one is supported
}
```

Dies ist besonders wichtig: Verschiedene Browser akzeptieren unterschiedliche H.264 Profile-Strings. Der ProxyGenerator probiert bis zu 8 verschiedene `avc1.*`-Varianten durch.

### 2.3 WebGPU Check

In `src/engine/core/WebGPUContext.ts`:
```typescript
if (!navigator.gpu) {
  log.error('WebGPU not supported');
  return false;
}
```

### 2.4 Feature Flags

`src/engine/featureFlags.ts` steuert den WebCodecs-Modus:
```typescript
export const flags = {
  useFullWebCodecsPlayback: false,  // Preview runs HTML-only for now
  useDecoderPool: false,
  useRenderGraph: false,
};
```

`useFullWebCodecsPlayback` ist derzeit `false` -- Preview nutzt im Default HTMLVideoElement, nur Export und Proxy-Generation verwenden WebCodecs.

---

## 3. Codec-Support-Matrix

### 3.1 Decoding (was der Browser abspielen kann)

| Codec | Chrome | Firefox | Safari | MasterSelects Handling |
|-------|--------|---------|--------|----------------------|
| H.264 (AVC) | Ja (HW) | Ja (HW) | Ja (HW) | HTMLVideoElement + WebCodecs |
| H.265 (HEVC) | Ab 107 (HW) | Nein | Ja (HW) | WebCodecs mit `hvc1`/`hev1` prefix |
| VP8 | Ja | Ja | Ab 12.1 | HTMLVideoElement |
| VP9 | Ja (HW) | Ja | Ab 14.1 (teilweise) | HTMLVideoElement + WebCodecs |
| AV1 | Ab 108 (HW) | Ab 98 (SW) | Ab 17 (HW) | WebCodecs mit `av01` prefix |
| ProRes | Nein | Nein | Safari HW | Nur via NativeHelper (FFmpeg) |
| DNxHD/HR | Nein | Nein | Nein | Nur via NativeHelper (FFmpeg) |

MasterSelects erkennt in `src/stores/timeline/helpers/mediaTypeHelpers.ts` professionelle Codecs:
```typescript
export function isProfessionalCodecFile(file: File): boolean {
  const ext = file.name.toLowerCase().split('.').pop();
  return ext === 'mov' || ext === 'mxf';
}
```

### 3.2 Encoding (Export)

Codec-Strings in `src/engine/export/codecHelpers.ts`:
```typescript
case 'h264': return 'avc1.4d0028';  // Main Profile, Level 4.0
case 'h265': return 'hvc1.1.6.L93.B0';  // Main Profile, Level 3.1
case 'vp9':  return 'vp09.00.10.08';  // Profile 0, 8-bit
case 'av1':  return 'av01.0.04M.08';  // Main Profile, 8-bit
```

Container-Codec-Kompatibilität:
```typescript
export function isCodecSupportedInContainer(
  codec: VideoCodec, container: ContainerFormat
): boolean {
  if (container === 'webm') {
    return codec === 'vp9' || codec === 'av1';
  }
  return true; // MP4 supports all
}

export function getFallbackCodec(container: ContainerFormat): VideoCodec {
  return container === 'webm' ? 'vp9' : 'h264';
}
```

### 3.3 Audio-Codec-Fallback

In `VideoEncoderWrapper.ts` (Zeile 97-125):
```
MP4-Container: AAC → Opus → Audio deaktiviert
WebM-Container: Opus → Audio deaktiviert
```

Linux-Chromium hat häufig keinen AAC-Encoder, daher der Opus-Fallback.

---

## 4. Container-Format-Probleme

### 4.1 MP4Box.js als zentrale Demuxing-Lösung

MasterSelects nutzt MP4Box.js (`mp4box` npm-Paket) als einzigen Demuxer. Das hat klare Limitierungen:

**Unterstützte Container:** MP4, MOV, M4V, 3GP (alles ISO BMFF basiert)
**Nicht unterstützt:** WebM, MKV, AVI, FLV, MXF

Für WebM wird kein Demuxing durchgeführt -- diese Dateien laufen ausschließlich über HTMLVideoElement.

### 4.2 Moov-Atom-Problem

Kamera-MOV-Dateien haben den `moov`-Atom oft am Ende der Datei (nicht "fast-start"). Der `mp4MetadataHelper.ts` löst das durch paralleles Lesen:

```typescript
// Strategy: Read from start AND end of file in parallel
// Camera MOV files often have moov atom at the end
const chunkSize = 1024 * 1024;       // 1MB chunks
const maxFromStart = 5 * 1024 * 1024; // 5MB from start
const maxFromEnd = 5 * 1024 * 1024;   // 5MB from end

// Run both in parallel
readStart();
readEnd();
```

Wenn das Parsing trotzdem fehlschlägt, greift die Duration-Schätzung:
```typescript
export function estimateDurationFromFileSize(file: File): number {
  // MOV: ~150 Mbps, MP4: ~50 Mbps
}
```

### 4.3 WebM Audio-Detection

Für WebM/MKV-Dateien wird in `audioDetection.ts` ein eigener EBML-Parser verwendet, der nach Audio-Track-Markern sucht:
```typescript
const audioMarkers = [
  [0x83, 0x02],                                       // TrackType = 2
  [0x86, 0x41, 0x5f, 0x4f, 0x50, 0x55, 0x53],       // "A_OPUS"
  [0x86, 0x41, 0x5f, 0x56, 0x4f, 0x52, 0x42, 0x49, 0x53], // "A_VORBIS"
];
```

---

## 5. Variable Frame Rate (VFR)

VFR-Videos sind ein echtes Problem. HTMLVideoElement abstrahiert die Zeitachse, aber bei WebCodecs muss man sich selbst darum kümmern.

### 5.1 VFR-aware Seek Tolerance

Der `WebCodecsPlayer` hat eine VFR-bewusste Seek-Toleranz (Zeile 1276):

```typescript
private computeSeekToleranceUs(targetIndex: number): number {
  const nominalFrameUs = 1_000_000 / Math.max(this.frameRate, 1);
  let neighborDeltaUs = Infinity;

  // Check actual spacing to previous and next samples
  if (targetIndex > 0) {
    const prev = this.samples[targetIndex - 1];
    const prevDelta = Math.abs(target.cts - prev.cts) * 1_000_000 / target.timescale;
    if (prevDelta > 0) neighborDeltaUs = Math.min(neighborDeltaUs, prevDelta);
  }
  // ... same for next sample

  const vfrAwareUs = Number.isFinite(neighborDeltaUs)
    ? neighborDeltaUs * 0.75
    : nominalFrameUs * 1.5;

  return Math.max(2_000, Math.min(200_000, Math.max(vfrAwareUs, nominalFrameUs)));
}
```

Das heisst: statt einer festen Toleranz basierend auf nomineller FPS wird der tatsächliche Abstand zwischen benachbarten Samples gemessen. Bei VFR-Videos (Screen-Recordings, Handy-Videos) kann der Abstand zwischen Frames stark variieren.

### 5.2 FPS-Erkennung

Die FPS wird aus dem Container gelesen, nicht aus den tatsächlichen Sample-Timings:
```typescript
this.frameRate = videoTrack.nb_samples / (videoTrack.duration / videoTrack.timescale);
```

Bei VFR-Videos ist das ein Durchschnitt, nicht die tatsächliche Rate. MasterSelects kompensiert dies über die VFR-aware Toleranz.

---

## 6. CORS und Cross-Origin Video

### 6.1 Lokale Dateien (Hauptanwendungsfall)

MasterSelects arbeitet primär mit lokalen Dateien via `URL.createObjectURL(file)`. Diese sind same-origin und haben keine CORS-Probleme. Video-Elemente werden mit `crossOrigin = 'anonymous'` erstellt (in `webCodecsHelpers.ts`):

```typescript
export function createVideoElement(file: File): HTMLVideoElement {
  const video = document.createElement('video');
  video.src = URL.createObjectURL(file);
  video.crossOrigin = 'anonymous';
  // ...
}
```

### 6.2 Remote-Videos (YouTube-Downloads, AI-generierte Clips)

Für heruntergeladene Videos (YouTube, PiAPI) werden die Dateien erst lokal als Blob gespeichert, dann über Blob-URLs verwendet -- kein CORS-Problem.

### 6.3 Tainted Canvas

WebGPU's `importExternalTexture()` und `copyExternalImageToTexture()` werden durch CORS-Einschränkungen blockiert. Ein Video ohne korrekte CORS-Header kann nicht als GPU-Textur importiert werden. Da MasterSelects ausschließlich mit lokalen Dateien (Blob-URLs) arbeitet, tritt dieses Problem in der Praxis nicht auf.

---

## 7. Firefox-spezifische Behandlung

### 7.1 importExternalTexture Black Frames

Firefox hat ein spezifisches Problem mit `importExternalTexture()`: Während der HTML-Video-Playback werden importierte Textures intermittierend schwarz. MasterSelects hat dafür eine explizite Workaround-Datei:

`src/engine/render/htmlVideoPreviewFallback.ts`:
```typescript
function isFirefoxBrowser(): boolean {
  return typeof navigator !== 'undefined' && /Firefox\//.test(navigator.userAgent);
}

export function getCopiedHtmlVideoPreviewFrame(video, scrubbingCache, ...): ... {
  if (!isFirefoxBrowser() || !scrubbingCache) {
    return null;
  }
  // Firefox: copy frame into persistent texture instead of using
  // importExternalTexture, which can sample as black intermittently
  const captured = scrubbingCache.captureVideoFrame(video, captureOwnerId);
  // ...
}
```

Statt des zero-copy `importExternalTexture` Pfads wird auf Firefox das Videobild in eine persistente GPU-Textur kopiert (`copyExternalImageToTexture`). Das ist langsamer, aber stabil.

### 7.2 WebCodecs auf Firefox

Firefox unterstützt WebCodecs seit Version 130 (September 2024), aber die Implementierung ist weniger ausgereift als in Chromium. MasterSelects' Feature-Detection (`hasWebCodecsSupport()`) behandelt Firefox korrekt -- wenn die API vorhanden ist, wird sie genutzt.

---

## 8. Linux-spezifische Probleme

### 8.1 Vulkan/VAAPI Performance

Linux mit Chrome ohne Vulkan erreicht nur ~15 FPS. MasterSelects zeigt eine explizite Warnung (`src/components/common/LinuxVulkanWarning.tsx`):

```
Linux detected: For best performance (60fps), enable Vulkan in Chrome.
Go to chrome://flags/#enable-vulkan and set it to Enabled
```

### 8.2 AAC-Encoding

Linux-Chromium hat oft keinen AAC-Encoder (Patent-Probleme). Der `AudioEncoderWrapper` erkennt das und fällt auf Opus zurück:
```typescript
const aacSupported = await AudioEncoderWrapper.isAACSupported();
if (!aacSupported) {
  const opusSupported = await AudioEncoderWrapper.isOpusSupported();
  if (opusSupported) this.audioCodec = 'opus';
}
```

---

## 9. Safari-spezifische Betrachtungen

### 9.1 WebGPU in Safari

Safari unterstützt WebGPU seit Version 17 (September 2023). Die `WebGPUContext` macht keinen browser-spezifischen Check, sondern prüft nur `navigator.gpu`.

### 9.2 Autoplay-Restriktionen

Alle Video-Elemente werden mit `muted = true` und `playsInline = true` erstellt, was die Autoplay-Policies aller Browser erfüllt:
```typescript
video.muted = true;
video.playsInline = true;
```

### 9.3 Codec-Support

Safari dekodiert ProRes nativ (Hardware), während Chrome/Firefox das nicht können. MasterSelects behandelt `.mov`-Dateien als "professional codec" und empfiehlt den NativeHelper:
```typescript
export function isProfessionalCodecFile(file: File): boolean {
  const ext = file.name.toLowerCase().split('.').pop();
  return ext === 'mov' || ext === 'mxf';
}
```

---

## 10. GPU-Surface-Warmup nach Page Reload

Ein subtiles, aber kritisches Problem: Nach einem Page-Reload sind die GPU-Surfaces von `<video>`-Elementen leer. Alle Rendering-APIs (`importExternalTexture`, `canvas.drawImage`, `copyExternalImageToTexture`) liefern schwarze Frames -- obwohl `readyState >= 2` ist.

Der `VideoSyncManager` löst das durch einen "Warmup"-Mechanismus:

```typescript
// After page reload, video GPU surfaces are empty — all sync rendering APIs
// return black. The ONLY way to populate the GPU surface is video.play().
private warmingUpVideos = new WeakSet<HTMLVideoElement>();
```

Der `LayerCollector` tracked den Zustand:
```typescript
private videoGpuReady = new WeakSet<HTMLVideoElement>();

if (!this.videoGpuReady.has(video) && !deps.isExporting) {
  if (!video.paused && !video.seeking) {
    this.videoGpuReady.add(video);  // GPU surface is now active
  }
}
```

---

## 11. Texture-Import-Pfade

Verschiedene Quellen erzeugen verschiedene GPU-Texture-Typen:

| Quelle | API | GPU-Typ | Kosten |
|--------|-----|---------|--------|
| HTMLVideoElement | `importExternalTexture` | `texture_external` | Zero-copy |
| VideoFrame (WebCodecs) | `importExternalTexture` | `texture_external` | Zero-copy |
| ImageBitmap (NativeHelper) | `copyExternalImageToTexture` | `texture_2d<f32>` | GPU-Upload |
| HTMLImageElement | `copyExternalImageToTexture` | `texture_2d<f32>` | Einmal, gecacht |
| HTMLCanvasElement (Text) | `copyExternalImageToTexture` | `texture_2d<f32>` | Gecacht |

Der `TextureManager` (`src/engine/texture/TextureManager.ts`) behandelt alle Fälle:

```typescript
importVideoTexture(source: HTMLVideoElement | VideoFrame): GPUExternalTexture | null {
  if (source instanceof HTMLVideoElement) {
    if (source.readyState < 2 || source.videoWidth === 0) return null;
  } else if (source instanceof VideoFrame) {
    if ((source as any).closed || source.codedWidth === 0) return null;
  }
  return this.device.importExternalTexture({ source });
}
```

Wichtig: Ein geschlossenes `VideoFrame` an `importExternalTexture` zu übergeben crasht den GPU-Prozess (`STATUS_BREAKPOINT`). Der Guard dagegen ist essenziell.

---

## 12. Codec-Description-Extraktion (avcC/hvcC)

Für korrekte WebCodecs-Dekodierung muss die Codec-Description (avcC-Box für H.264, hvcC für HEVC) aus dem MP4-Container extrahiert werden. `WebCodecsPlayer.ts` macht das im `onReady`-Callback:

```typescript
const entry = trak.mdia.minf.stbl.stsd.entries[0];
const configBox = entry.avcC || entry.hvcC || entry.vpcC || entry.av1C;
if (configBox) {
  const stream = new MP4Box.DataStream(undefined, 0, MP4Box.DataStream.BIG_ENDIAN);
  configBox.write(stream);
  description = stream.buffer.slice(8); // Skip box header
}
```

Ohne die Description akzeptieren viele Decoder den Stream nicht oder dekodieren fehlerhaft. Der ProxyGenerator hat sogar eine explizite Retry-Logik ohne Description:

```typescript
// If first attempt fails, retry without description
if (this.codecConfig?.description) {
  const configWithoutDesc = { ...config };
  delete configWithoutDesc.description;
  const support = await VideoDecoder.isConfigSupported(configWithoutDesc);
  if (support.supported) { /* retry */ }
}
```

---

## 13. canPlayType() -- Warum es in MasterSelects NICHT verwendet wird

Interessanterweise nutzt MasterSelects `canPlayType()` **nicht**. Stattdessen wird Media-Support durch eine Kombination aus:

1. **File-Extension-Erkennung** (Haupt-Mechanismus)
2. **MP4Box.js Container-Parsing** (für Codec-Identifikation)
3. **VideoDecoder.isConfigSupported()** (für WebCodecs)
4. **HTMLVideoElement loadedmetadata/error Events** (als Fallback)

Das ist eine bewusste Design-Entscheidung: `canPlayType()` gibt nur `"maybe"` oder `"probably"` zurück, nie `"yes"`. Für eine NLE ist diese Unsicherheit nicht akzeptabel. Stattdessen wird einfach versucht, die Datei zu öffnen, und Fehler werden abgefangen.

---

## 14. Die Rolle des NativeHelper

Der NativeHelper (`tools/native-helper/`) ist ein Rust-Programm mit FFmpeg-Bindungen, das als lokaler Server läuft (WebSocket 9876, HTTP 9877). Er löst das Problem der Browser-Codec-Limitierung komplett:

```typescript
export async function getNativeCodecs(): Promise<string[]> {
  return ['prores', 'dnxhd', 'dnxhr', 'ffv1', 'utvideo', 'mjpeg', 'h264', 'h265', 'vp9'];
}
```

Der Upgrade-Pfad ist in `upgradeToNativeDecoder.ts`:
- Bei Verbindung: Alle bestehenden Video-Clips werden auf NativeDecoder upgegradet
- Bei Trennung: Automatischer Downgrade zurück auf WebCodecs/HTMLVideoElement
- Clip-Watcher: Neue Clips werden automatisch upgegradet

```typescript
export function downgradeAllClipsFromNativeDecoder(): void {
  // Close all native decoders, remove from clips
  // WebCodecs/HTMLVideo takes over automatically
}
```

---

## 15. Zusammenfassung der Fallback-Strategien

| Szenario | Primär | Fallback 1 | Fallback 2 |
|----------|--------|------------|------------|
| **Preview (Playback)** | HTMLVideoElement | WebCodecs (wenn Flag) | - |
| **Preview (NativeHelper)** | NativeDecoder | HTMLVideoElement | - |
| **Export** | WebCodecs VideoEncoder | FFmpeg WASM | - |
| **Proxy-Generation** | WebCodecs VideoDecoder | Retry ohne Description | Error |
| **Scrub/Seek** | Live-Import | Scrubbing-Cache | Emergency-Hold Frame |
| **Metadata** | HTMLVideoElement | MP4Box.js | File-Size-Schätzung |
| **Audio-Detection** | MP4Box.js | HTMLVideoElement audioTracks | EBML-Parsing (WebM) |
| **Audio-Encoding** | AAC (mp4a.40.2) | Opus | Deaktiviert |
| **Firefox Playback** | Copied texture | Scrubbing-Cache | - |
| **GPU Surface Cold** | Warmup (play+pause) | Cache-Fallback | Drop Frame |

Die Architektur ist bemerkenswert robust: Jeder Pfad hat mindestens einen Fallback, und das System degradiert graceful statt zu crashen. Der ScrubbingCache dient als universeller "letzter Ausweg" für alle Video-Preview-Situationen.
