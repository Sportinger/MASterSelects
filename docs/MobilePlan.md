# Mobile UI Plan

## Konzept

Mobile-first Video Editor mit Touch-optimierter Bedienung. Volle Editing-Funktionalität, angepasstes UI.

## Layout

```
┌─────────────────────────────────┐
│                                 │
│           PREVIEW               │  ← Fix, immer sichtbar
│                                 │
├─────────────────────────────────┤
│  [🔍]  ────────●────────  [✂️]  │  ← Precision + Cut Buttons
├─────────────────────────────────┤
│ ▶ ├──Clip──┤  ├──Clip──┤       │  ← Timeline
│            ▼ Playhead           │
│   [Keyframes wenn selected]     │
└─────────────────────────────────┘
```

### Panel-System

| Richtung | Panel | Breite |
|----------|-------|--------|
| Pull ↓ von oben | Properties, Effects, Transform, Audio, Masks | 100% |
| Swipe ← von rechts | Media Browser, Compositions | 50%, halbtransparent |
| Swipe → von links | File Menu, Export, Settings, Undo/Redo | 50%, halbtransparent |

### Aktiver Slider

Wenn im Properties Panel ein Slider angetippt wird:
1. Properties Panel schließt
2. Einzelner Slider erscheint zwischen Preview und Timeline
3. User sieht Änderung live in Preview
4. Tap woanders → Slider verschwindet

## Gesten

| Geste | Aktion |
|-------|--------|
| Tap | Select Clip |
| Drag | Timeline scrollen |
| Double-Tap + Drag | Clip verschieben |
| Tap Clip-Anfang/Ende + Drag | Trimmen |
| Pinch | Timeline Zoom |
| 2-Finger Swipe ← | Undo |
| 2-Finger Swipe → | Redo |
| Pull von oben | Properties Panel öffnen |
| Swipe von links | Media Panel (50%) |
| Swipe von rechts | Options Menu (50%) |
| Hold 🔍 + Drag Playhead | Precision Mode (langsamer) |
| Tap ✂️ | Cut/Split am Playhead |

## Architektur

### Wiederverwenden (100%)

Alles was nicht UI ist:

```
src/
├── engine/           ← WebGPU Engine komplett
├── stores/           ← Zustand Stores komplett
│   ├── timeline/     ← Timeline State
│   ├── mediaStore    ← Media State
│   └── ...
├── services/         ← Alle Services
│   ├── proxyGenerator
│   ├── audioManager
│   ├── projectSync
│   ├── aiTools
│   └── ...
└── shaders/          ← WGSL Shaders
```

### Neu bauen (Mobile UI)

```
src/
├── components/
│   └── mobile/                    ← Neuer Ordner
│       ├── MobileApp.tsx          ← Root Component
│       ├── MobilePreview.tsx      ← Preview (touch gestures)
│       ├── MobileTimeline.tsx     ← Timeline (gestures, precision)
│       ├── MobileProperties.tsx   ← Pull-down Panel
│       ├── MobileMediaPanel.tsx   ← Swipe-in Panel
│       ├── MobileOptionsMenu.tsx  ← Swipe-in Menu
│       ├── MobileSlider.tsx       ← Touch-optimierter Slider
│       ├── MobileToolbar.tsx      ← Cut, Precision buttons
│       └── hooks/
│           ├── useSwipePanel.ts   ← Panel swipe logic
│           ├── usePinchZoom.ts    ← Pinch gesture
│           ├── usePrecisionDrag.ts← Slow-drag mode
│           └── useTwoFingerSwipe.ts← Undo/Redo
└── styles/
    └── mobile.css                 ← Mobile-specific styles
```

### Ansatz: Responsive mit separaten Komponenten

```tsx
// App.tsx
function App() {
  const isMobile = useMediaQuery('(max-width: 768px)');

  return isMobile ? <MobileApp /> : <DesktopApp />;
}
```

**Vorteile:**
- Stores/Services 1x pflegen
- Klare Trennung Desktop vs Mobile UI
- Keine Kompromisse in beiden UIs
- Schrittweise entwickelbar

## Implementation Phasen

### Phase 1: Grundgerüst
- [ ] MobileApp.tsx mit Layout
- [ ] MobilePreview.tsx (nur Anzeige)
- [ ] MobileTimeline.tsx (nur Anzeige, Scroll)
- [ ] Erkennung Mobile vs Desktop

### Phase 2: Basic Editing
- [ ] Clip Selection (Tap)
- [ ] Playhead Drag
- [ ] Precision Mode (🔍 Button)
- [ ] Cut Button (✂️)
- [ ] Timeline Pinch Zoom

### Phase 3: Clip Manipulation
- [ ] Double-Tap + Drag (Clip move)
- [ ] Trim (Tap Ende + Drag)
- [ ] 2-Finger Undo/Redo

### Phase 4: Panels
- [ ] Pull-down Properties Panel
- [ ] Aktiver Slider Modus
- [ ] Swipe Media Panel
- [ ] Swipe Options Menu

### Phase 5: Polish
- [ ] Alle Effects im Properties Panel
- [ ] Keyframe Editing
- [ ] Export
- [ ] AI Chat Integration

## Touch Considerations

### Finger-freundliche Targets
- Minimum 44x44px für Touch Targets
- Slider-Tracks breiter als Desktop
- Großzügige Hit-Areas für Clip-Enden

### Feedback
- Haptic Feedback bei Actions (wenn verfügbar)
- Visual Feedback bei Touch
- Undo-Toast nach Aktionen

### Performance
- Throttle Touch-Events
- GPU-beschleunigtes Scrolling
- Lazy Loading für Media Thumbnails

## Offene Fragen

- [ ] Landscape vs Portrait? (Vermutlich nur Landscape)
- [ ] Tablet-spezifisches Layout? (Mehr Platz)
- [ ] PWA mit Offline-Support?
- [ ] Keyboard-Support wenn angeschlossen?
