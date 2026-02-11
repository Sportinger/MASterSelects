# Keyboard Shortcuts

[← Back to Index](./README.md)

Complete reference of all keyboard shortcuts (verified from codebase).

---

## Playback

| Shortcut | Action |
|----------|--------|
| `Space` | Play/Pause toggle |
| `J` | Reverse playback (press multiple times for faster) |
| `K` | Pause playback |
| `L` | Forward playback (press multiple times for faster) |
| `I` | Set In point at playhead |
| `O` | Set Out point at playhead |
| `X` | Clear In/Out points |
| `Home` | Go to start of timeline |
| `End` | Go to end of timeline |

---

## Timeline Navigation

| Shortcut | Action |
|----------|--------|
| `Scroll` | Vertical scroll |
| `Shift + Scroll` | Horizontal scroll |
| `Alt + Scroll` | Zoom (centered on playhead) |
| `Alt + Scroll` | Zoom (exponential 8% per step, centered on playhead) |
| `←` / `→` | Frame-by-frame |

---

## Editing

| Shortcut | Action |
|----------|--------|
| `C` | Cut tool - split clips at playhead (with snapping) |
| `Alt + C` | Cut without snapping |
| `Ctrl + C` | Copy selected clips or keyframes |
| `Ctrl + V` | Paste clips or keyframes at playhead |
| `Delete` / `Backspace` | Delete selected (keyframes first, then clips) |
| `Tab` | Toggle edit mode in preview |
| `Escape` | Deselect all |

---

## Selection

| Action | Method |
|--------|--------|
| Single select | Click clip |
| Multi-select | `Shift + Click` |
| Add/remove from selection | `Ctrl + Click` |
| Linked clip select | Click (selects both video + audio) |
| Independent select | `Shift + Click` linked clip |
| Move multi-selection | Drag any selected clip |
| Deselect | Click empty or `Escape` |

---

## Keyframes

| Action | Method |
|--------|--------|
| Select keyframe | Click diamond |
| Multi-select | `Shift + Click` |
| Fine drag | `Shift + Drag` (10x slower) |
| Copy keyframes | `Ctrl + C` (with keyframes selected) |
| Paste keyframes | `Ctrl + V` (at playhead on selected clip) |
| Move multi-select | Drag any selected keyframe |
| Easing menu | Right-click keyframe |

---

## Blend Modes

| Shortcut | Action |
|----------|--------|
| `Shift + +` | Next blend mode |
| `Shift + -` | Previous blend mode |
| `Numpad +` | Next blend mode (alternative) |
| `Numpad -` | Previous blend mode (alternative) |

---

## Project

| Shortcut | Action |
|----------|--------|
| `Ctrl + N` | New Project |
| `Ctrl + S` | Save Project |
| `Ctrl + O` | Open (shows file menu) |
| `Ctrl + Z` | Undo |
| `Ctrl + Shift + Z` | Redo |

---

## Modifiers

### Shift Key
| Context | Effect |
|---------|--------|
| + Scroll | Horizontal scroll |
| + Drag playhead | Snap to keyframes |
| + Drag keyframe | Fine control (10x slower) |
| + `+`/`-` | Cycle blend modes |
| + Marquee | Extend selection |
| + Drag curve handle | Constrain horizontal |

### Alt Key
| Context | Effect |
|---------|--------|
| + Scroll | Zoom timeline |
| + Drag clip | Skip linked clip movement |
| + Drag in group | Skip group movement |

### Ctrl/Cmd Key
| Context | Effect |
|---------|--------|
| + Click | Add/remove from selection |
| + `Z` | Undo |
| + `Shift + Z` | Redo |
| + `S` | Save |
| + `N` | New Project |

---

## Context-Specific

### Property Values
| Action | Effect |
|--------|--------|
| Left-click drag | Scrub value |
| Right-click | Reset to default |

### Track Headers
| Action | Effect |
|--------|--------|
| Double-click name | Edit track name |
| Click Eye | Toggle visibility |
| Click M | Toggle mute |
| Click S | Toggle solo |
| Click expand arrow | Show keyframe lanes |

### Clip Clips
| Action | Effect |
|--------|--------|
| Drag center | Move clip |
| Drag edges | Trim clip |
| Right-click | Context menu |

### Preview Edit Mode
| Action | Effect |
|--------|--------|
| `Tab` | Toggle edit mode on/off |
| Drag center | Move layer |
| Drag corner handle | Scale layer |
| Drag edge handle | Scale from edge |
| `Shift + Drag` | Lock aspect ratio during scale |

### Curve Editor
| Action | Effect |
|--------|--------|
| Drag keyframe | Move time + value |
| `Shift + Drag` | Constrain axis |
| Drag handle | Adjust bezier curve |
| Click empty | Deselect |

---

## Quick Reference Card

```
┌─────────────────────────────────────────┐
│           PLAYBACK                      │
│  Space = Play    J/K/L = Shuttle       │
│  I/O = In/Out    X = Clear I/O         │
│  Home/End = Start/End of timeline      │
├─────────────────────────────────────────┤
│           EDITING                       │
│  C = Cut/Split   Del = Delete          │
│  Ctrl+C = Copy   Ctrl+V = Paste        │
│  Ctrl+Z = Undo   Ctrl+Shift+Z = Redo   │
│  Tab = Edit Mode                        │
├─────────────────────────────────────────┤
│           SELECTION                     │
│  Shift+Click = Multi-select            │
│  Ctrl+Click = Add/Remove               │
├─────────────────────────────────────────┤
│           PROJECT                       │
│  Ctrl+N = New    Ctrl+S = Save         │
├─────────────────────────────────────────┤
│           NAVIGATION                    │
│  Alt+Scroll = Zoom (exponential)       │
│  Shift+Scroll = H-Scroll               │
├─────────────────────────────────────────┤
│           BLEND MODES                   │
│  Shift++ = Next  Shift+- = Previous    │
│  Numpad+/- = Cycle blend modes         │
└─────────────────────────────────────────┘
```

---

## Related Features

- [Timeline](./Timeline.md) - Main editing
- [Keyframes](./Keyframes.md) - Animation
- [Preview](./Preview.md) - Playback
- [Effects](./Effects.md) - Blend modes

---

## Tests

| Test File | Tests | Coverage |
|-----------|-------|----------|
| [`playbackSlice.test.ts`](../../tests/stores/timeline/playbackSlice.test.ts) | 16 | Playback shortcuts (space, JKL, in/out) |

Run tests: `npx vitest run`

---

*Compiled from codebase analysis*
