# MASterSelects - Feature Handbuch

Vollständige Dokumentation aller Features der Video-Editing-Anwendung.

---

## 1. Timeline

### Basis-Funktionen
- **Multi-Track Timeline**: Video- und Audio-Tracks
- **Clip-Management**: Hinzufügen, Entfernen, Verschieben von Clips
- **Clip-Trimming**: Links/Rechts-Kanten ziehen für In/Out-Points
- **Linked Clips**: Audio/Video bleiben beim Verschieben synchron
- **Overlap Resistance**: Magnetischer Widerstand beim Überlappen (Video + Audio)
- **Clip-Splitting**: Clips teilen mit C-Taste (splittet ALLE Clips am Playhead, oder nur ausgewählte)
- **Multi-Select**: Shift+Klick für Mehrfachauswahl, gemeinsam verschieben
- **Linked Clip Selection**: Klick wählt Video+Audio zusammen aus
- **Solid Color Clips**: Farbige Solid-Layer erstellen
- **Playhead**: Klicken zum Springen, Ziehen zum Scrubben
- **JKL Playback**: Industry-Standard J/K/L Shortcuts
- **Zoom**: Exponentieller Zoom mit Alt+Scroll (8% pro Schritt)
- **Fit Button**: Zoom anpassen um ganze Komposition zu sehen
- **Track-Höhe**: Individuelle Track-Höhen anpassbar
- **Track Solo/Mute**: Audio-Tracks stumm schalten oder solo hören

### Playback-Controls
- **Play/Pause/Stop**: Standard-Wiedergabe
- **Loop-Playback**: Schleifenwiedergabe innerhalb In/Out-Punkten
- **In/Out-Marker**: Arbeitsbereich mit I/O-Punkten setzen
- **Editierbare Duration**: Klick auf Gesamtdauer zum Ändern der Kompositionslänge

### Erweiterte Features
- **RAM Preview**: Frames cachen für flüssige Wiedergabe
- **Proxy-System**: Niedrigere Auflösung für bessere Performance
- **Waveform-Anzeige**: Audio-Wellenformen auf Clips (Rechtsklick zum Generieren)
- **Thumbnail-Strips**: Filmstreifen-Vorschau auf Video-Clips
- **Compositions**: Verschachtelte Timelines als Clips
- **Undo/Redo**: Ctrl+Z / Ctrl+Shift+Z (oder Ctrl+Y)

---

## 2. Keyframe-Animation

### Keyframe-Steuerung
- **Keyframes erstellen**: Auf Clip-Properties
- **Recording-Modus**: Keyframes live während Wiedergabe aufnehmen
- **Keyframes bearbeiten**: Zeit, Wert und Easing ändern
- **Multi-Selection**: Mehrere Keyframes mit Shift+Klick auswählen
- **Interpolation**: Automatische Wertberechnung zwischen Keyframes

### Easing-Funktionen
- Linear
- Ease-In
- Ease-Out
- Ease-In-Out
- Custom Bezier (frei einstellbare Bezier-Kurve)

### Erweiterte Keyframe-Features
- **Copy/Paste**: Ctrl+C / Ctrl+V für Keyframes
- **Keyframe-Ticks**: Visuelle Tick-Markierungen auf Clip-Balken in der Timeline
- **Curve Editor Auto-Scale**: Automatische Skalierung der Kurvenansicht
- **Custom Bezier Easing**: 5. Easing-Modus mit frei definierbarer Bezier-Kurve

### Animierbare Properties
- **Opacity**: 0-1
- **Position**: X, Y, Z (Pixel)
- **Scale**: X, Y (Prozent)
- **Rotation**: X, Y, Z (Grad)
- **Effekt-Parameter**: Alle Effekt-Parameter animierbar

---

## 3. Effekte & Transforms

### Verfügbare Effekte (30 Effekte)

#### Color (9)
| Effekt | Beschreibung |
|--------|-------------|
| Brightness | Helligkeit anpassen |
| Contrast | Kontrast und Mitten |
| Saturation | Farbsättigung |
| Vibrance | Intelligente Sättigung (schont Hauttöne) |
| Hue Shift | Farben im HSV-Raum rotieren |
| Temperature | Farbtemperatur (warm/kalt) |
| Exposure | Belichtungskorrektur |
| Levels | Histogram mit Input/Output Black/White, Gamma |
| Invert | Farbumkehrung |

#### Blur (5)
| Effekt | Beschreibung |
|--------|-------------|
| Box Blur | Einfacher Weichzeichner |
| Gaussian Blur | Gausssche Unschärfe |
| Motion Blur | Bewegungsunschärfe mit Richtung |
| Radial Blur | Kreisförmige Unschärfe |
| Zoom Blur | Zentraler Zoom-Effekt |

#### Distort (7)
| Effekt | Beschreibung |
|--------|-------------|
| Pixelate | Mosaik-Effekt |
| Kaleidoscope | Segment-Spiegelung mit Rotation |
| Mirror | Horizontal/Vertikal spiegeln |
| RGB Split | Chromatische Aberration |
| Twirl | Spiralförmige Verzerrung |
| Wave | Wellenförmige Verzerrung |
| Bulge | Kugelförmige Verzerrung (Fischauge) |

#### Stylize (8)
| Effekt | Beschreibung |
|--------|-------------|
| Vignette | Randabdunklung |
| Grain | Film-Korn Simulation |
| Glow | Leuchtender Weichzeichner |
| Posterize | Farbreduktion / Poster-Effekt |
| Edge Detect | Kantenerkennung |
| Scanlines | CRT-Monitor Scanlines |
| Threshold | Schwellwert-Binarisierung |
| Sharpen | Nachschärfen |

#### Keying (1)
| Effekt | Beschreibung |
|--------|-------------|
| Chroma Key | Greenscreen/Bluescreen-Entfernung |

### Blend-Modes (After Effects-Style)
**Normal**: normal, dissolve, dancing-dissolve
**Darken**: darken, multiply, color-burn, linear-burn, darker-color
**Lighten**: add, lighten, screen, color-dodge, linear-dodge, lighter-color
**Contrast**: overlay, soft-light, hard-light, linear-light, vivid-light, pin-light, hard-mix
**Inversion**: difference, exclusion, subtract, divide
**Component**: hue, saturation, color, luminosity
**Stencil**: stencil-alpha, stencil-luma, silhouette-alpha, silhouette-luma, alpha-add

### Transform-Properties
- **Position**: X, Y, Z Koordinaten
- **Scale**: Unabhängige X/Y Skalierung
- **Rotation**: 3D-Rotation auf X, Y, Z Achsen
- **Opacity**: 0-100% Transparenz
- **Präzisions-Slider**: Shift=langsam, Ctrl=ultra-langsam

---

## 4. Media-Management

### Import & Organisation
- **Multi-Format**: Video, Audio, Bilder
- **Drag-and-Drop**: Dateien direkt ins Media Panel ziehen
- **File System Access API**: Native Dateiauswahl
- **Ordner-Struktur**: Ordner erstellen, umbenennen, löschen
- **Thumbnails**: Auto-generierte Vorschaubilder

### Composition-System
- **Compositions erstellen**: Neue verschachtelte Kompositionen
- **Composition-Settings**: Auflösung, Framerate
- **Verschachtelte Timelines**: Compositions als Clips verwenden
- **Tab-Wechsel**: Zwischen mehreren Compositions wechseln

### Proxy-System
- **Proxy-Generierung**: Niedrigere Auflösung für Performance
- **Proxy-Ordner**: Eigener Ausgabeort wählbar
- **Status-Anzeige**: "P" Badge auf Clips mit Proxy
- **Progress-Tracking**: Echtzeit-Fortschritt
- **Proxy-Cache**: Frame-Cache für flüssiges Scrubbing

---

## 5. Export

### WebCodecs Export (Standard)
- **Frame-by-Frame Rendering**: Präzise Frame-Ausgabe
- **Format**: MP4 mit H.264/VP9 Codec
- **Qualitätseinstellungen**: Auflösung und Bitrate
- **Preset-Auflösungen**: 1080p, 4K, Custom
- **Framerate**: 24p, 30p, 60p, etc.
- **Zeitbereich**: In/Out-Range oder komplette Timeline
- **Bitrate-Schätzung**: Automatische Empfehlung
- **Dateigröße-Vorschau**: Geschätzte Ausgabegröße
- **WebCodecs**: Hardware-beschleunigtes Encoding
- **Progress**: Echtzeit-Fortschrittsanzeige

### FFmpeg Export (Professionell)
- **Professional Codecs**: ProRes, DNxHR, HAP
- **Lossless**: FFV1, Ut Video
- **Delivery**: H.264 (x264), H.265 (x265), VP9, AV1
- **Container**: MOV, MP4, MKV, WebM, MXF
- **Platform Presets**: YouTube, Vimeo, Instagram, TikTok
- **NLE Presets**: Premiere, Final Cut, DaVinci, Avid
- **VJ Presets**: HAP Q für Media Server
- **On-Demand Loading**: WASM lädt bei Bedarf (~20MB)

---

## 6. Masken

### Zeichenwerkzeuge
- **Pen Tool**: Freiform-Masken mit Bezier-Kurven
- **Rechteck-Maske**: Schnelle rechteckige Maske
- **Ellipsen-Maske**: Kreisförmige/ovale Maske
- **Bezier-Handles**: Kubische Kontrollpunkte für glatte Kurven

### Masken-Properties
- **Opacity**: 0-100%
- **Feather**: Unschärfe-Radius (0-50px)
- **Feather Quality**: Low (9), Medium (17), High (25 Samples)
- **Invertiert**: Maske umkehren
- **Mask Modes**: Add, Subtract, Intersect
- **Mehrere Masken**: Stapeln auf einem Clip
- **Reihenfolge**: Drag zum Umsortieren

---

## 7. Transkription (Speech-to-Text)

### Provider
- **Local Whisper**: Browser-basiert, kein API-Key nötig
- **OpenAI Whisper API**: Cloud-basiert
- **AssemblyAI**: Alternative Cloud
- **Deepgram**: Alternative Cloud

### Features
- **11+ Sprachen**: Englisch, Deutsch, Französisch, Spanisch, etc.
- **Wort-Level Timing**: Präzise Zeitstempel pro Wort
- **Suche**: Transkript durchsuchen, Treffer hervorheben
- **Timeline-Marker**: Visuelle Marker auf Timeline
- **Echtzeit-Highlighting**: Aktuelles Wort während Wiedergabe
- **Status-Tracking**: Fortschritt und Fehlermeldungen

---

## 8. Video-Analyse

### Analysierte Metriken
- **Focus/Sharpness**: Laplacian Variance-basierte Schärfeerkennung
- **Motion**:
  - Global Motion (Kamerabewegung, Szenenwechsel)
  - Local Motion (Objektbewegung)
  - Schnitterkennung
- **Face Detection**: Gesichter zählen und lokalisieren

### Visualisierung
- **Farbcodierte Schwellwerte**:
  - Focus: Grün (>70% = gut), Rot (<40% = unscharf)
  - Motion: Blau (stabil), Rot (hoch/wackelig)
  - Gesichter: Gelbe Punkte oben
- **Analyse-Overlay**: Dual-Line Graph (Focus + Motion)
- **Frame-Sampling**: 500ms Intervall
- **GPU-Beschleunigung**: Optical Flow Analyzer

---

## 9. Download Panel (ehemals YouTube)

### Features
- **Multi-Plattform Downloads**: Videos von verschiedenen Plattformen herunterladen
- **Unterstützte Plattformen**: YouTube, TikTok, Instagram, Twitter/X, Facebook, Reddit, Vimeo, Twitch
- **Native Helper Integration**: Downloads via yt-dlp über den Native Helper Service
- **Qualitätsauswahl**: Verschiedene Auflösungen und Formate wählbar

---

## 10. Multicam

### Features
- **Mehrere Kameras**: Clips zur Multicam-Gruppe hinzufügen
- **Master-Kamera**: Audio-Referenz wählen
- **Kamera-Rollen**: Wide, Close-up, Detail, Custom
- **Sync-Methoden**:
  - Audio-Synchronisation (Cross-Correlation)
  - Transkript-Synchronisation (Speech-Matching)
- **Sync-Offset**: Automatische Offset-Berechnung
- **Linked Groups**: Sync-Clips zusammen verschieben

---

## 11. AI Chat Panel

### Integration
- **OpenAI**: Direkte Chat-Verbindung
- **Modelle**: GPT-5.2, GPT-5.1, o3, o4-mini, GPT-4.1, etc.
- **Message History**: Kontext bleibt erhalten
- **API Key Management**: Sichere Schlüsselspeicherung

### SAM2 AI Segmentation
- **AI-basierte Objektsegmentierung**: Meta SAM2 Modell zur automatischen Maskenerstellung
- **Interaktive Auswahl**: Objekte im Video per Klick auswählen und verfolgen

---

## 12. Video Scopes

### GPU-beschleunigte Scopes
- **Histogram**: RGB-Verteilungsgraph mit R/G/B/Luma View-Modes
- **Vectorscope**: Farb-Vektoranalyse mit Phosphor-Glow
- **Waveform**: DaVinci-style Waveform mit Sub-Pixel-Verteilung
- **Unabhängige Panels**: Jeder Scope als eigenständiges dockbares Panel
- **IRE Legende**: Broadcast-Referenz
- **Zero-Copy**: Vollständig GPU-gerendert, kein readPixels Overhead

---

## 13. Transitions

### Features
- **Crossfade-Transitions**: GPU-beschleunigte Überblendungen
- **Transitions Panel**: Modular mit Drag-and-Drop
- **Timeline-Integration**: Visuell auf Clips dargestellt

---

## 14. UI Features

### Dock-System
- **Anpassbares Layout**: Panels per Drag-and-Drop
- **Floating Panels**: Als separate Fenster
- **Panel-Tabs**: Mehrere Panels in einem Container
- **Split Panes**: Vertikal/Horizontal teilen
- **Layout-Persistenz**: Layouts speichern/laden

### Panels (16 Typen)
- Timeline, Preview, Media Panel, Properties Panel
- Export, Multicam, AI Chat, AI Video
- Download, AI Segment, Transitions, Histogram, Vectorscope, Waveform, Slots

### Toolbar
- **Projekt-Management**: New, Save, Load, Delete
- **Resolution Settings**: Ausgabeauflösung
- **Output Windows**: Zusätzliche Displays
- **MIDI Control**: MIDI-Input aktivieren

---

## 15. Performance

### Optimierungen
- **WebGPU**: GPU-beschleunigtes Rendering
- **Zero-Copy Textures**: Direkt VideoFrame zu GPU
- **Hardware Decoding**: WebCodecs für effizientes Decoding
- **Ping-Pong Buffers**: Effizientes Compositing
- **Lazy Loading**: On-Demand Ressourcen-Laden

### Monitoring
- **Echtzeit-FPS**: Aktuelle Frames pro Sekunde
- **Render-Timing**: Import/Render/Submit Aufschlüsselung
- **Frame-Drop Tracking**: Zählung und Gründe
- **Bottleneck-Erkennung**: Automatische Problem-Identifikation

---

## 16. Projekt-Management

### Funktionen
- **Neues Projekt**: Leere Projekte erstellen
- **Lokale Ordner**: Projekte in lokalen Ordnern speichern (File System Access API)
- **Raw-Ordner**: Importierte Medien automatisch nach Raw/ kopiert
- **Auto-Relink**: Fehlende Dateien automatisch aus Raw-Ordner wiederhergestellt
- **Auto-Save**: Konfigurierbares Intervall (1-10 min)
- **Backup-System**: Letzte 20 Backups automatisch
- **Save As**: Projekt an neuen Ort exportieren

---

## 17. Output Manager

### Multi-Output Rendering
- **RenderTarget-System**: Mehrere unabhängige Ausgabeziele gleichzeitig rendern
- **Source Routing**: Quellzuweisung pro Output (aktive Komposition, spezifische Compositions, Slots)
- **Slice-System**: Corner-Pin Warping für Projection Mapping
- **Mask Layers**: Sichtbarkeitskontrolle pro Output über Masken-Ebenen
- **Auto-Save**: Konfigurationen werden automatisch pro Projekt gespeichert
- **Window Management**: Fensterposition und -größe werden wiederhergestellt beim Öffnen

---

## 18. Slot Grid

### Resolume-Style Performance Grid
- **4x12 Grid**: 4 Layer (A-D) mit je 12 Slots
- **Click-to-Play**: Klick startet Clip, erneuter Klick startet neu
- **Multi-Layer Composition**: Layer A-D werden live übereinandergelegt
- **Unabhängige Wall-Clock Time**: Jeder Layer läuft mit eigener Echtzeit-Uhr
- **Ansichtswechsel**: Ctrl+Shift+Scroll zum Umschalten zwischen Timeline und Slot Grid

---

## 19. Tutorial System

### Interaktives Onboarding
- **Welcome Screen**: Programmauswahl beim ersten Start
- **Spotlight-Einführung**: Panel-basierte Einführung mit Spotlight-Hervorhebung
- **Timeline Deep-Dive**: Teil 2 mit detaillierter Timeline-Erklärung
- **Clippy Maskottchen**: Animierter Begleiter während des Tutorials

---

## 20. Technische Spezifikationen

### Unterstützte Formate
- **Video**: H.264, HEVC, VP9, AV1 (via WebCodecs)
- **Audio**: WAV, MP3, OGG, FLAC, AAC, M4A, AIFF, Opus, WMA
- **Bilder**: JPG, PNG, WebP, GIF

### Anforderungen
- **WebGPU**: Chrome 113+, Safari 18+
- **WebCodecs**: Chrome 94+, Safari 16.4+
- **Display**: Empfohlen 1920x1080 minimum

### Performance-Ziele
- **Target FPS**: 60 FPS
- **Render-Zeit**: <16.67ms pro Frame
- **Multi-Layer**: 10+ Layer bei voller Auflösung
