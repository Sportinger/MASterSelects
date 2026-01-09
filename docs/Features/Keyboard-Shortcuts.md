# Keyboard Shortcuts

[← Back to Index](./README.md)

Complete reference of all keyboard shortcuts.

---

## Table of Contents

- [Playback](#playback)
- [Timeline Navigation](#timeline-navigation)
- [Selection](#selection)
- [Editing](#editing)
- [View](#view)
- [Modifiers](#modifiers)

---

## Playback

| Shortcut | Action |
|----------|--------|
| `Space` | Play/Pause |
| `←` | Previous frame |
| `→` | Next frame |
| `Home` | Go to start |
| `End` | Go to end |
| `I` | Set In point |
| `O` | Set Out point |
| `L` | Loop toggle |

---

## Timeline Navigation

| Shortcut | Action |
|----------|--------|
| `Alt + Scroll` | Zoom in/out (centered on playhead) |
| `Shift + Scroll` | Horizontal scroll |
| `Scroll` | Vertical scroll |
| `Shift + Drag Playhead` | Snap to keyframes |

---

## Selection

| Shortcut | Action |
|----------|--------|
| `Click` | Select clip |
| `Click Empty` | Deselect all |
| `Drag Empty` | Marquee selection |
| `Delete` | Delete selected |

### Selection Priority
- When keyframes selected: `Delete` removes keyframes
- When clips selected: `Delete` removes clips

---

## Editing

| Shortcut | Action |
|----------|--------|
| `C` | Split clips at playhead |
| `Delete` | Delete selected clips/keyframes |
| `Ctrl + Z` | Undo |
| `Ctrl + Shift + Z` | Redo |
| `Ctrl + Y` | Redo (alternative) |

---

## View

| Shortcut | Action |
|----------|--------|
| `Shift + +` | Next blend mode |
| `Shift + -` | Previous blend mode |
| `F` | Fit timeline to view |

---

## Modifiers

### Shift Key
| Context | Effect |
|---------|--------|
| Drag playhead | Snap to keyframes/clip edges |
| Scroll | Horizontal scroll |
| +/- keys | Cycle blend modes |
| Drag vertex | Scale bezier handles |

### Alt Key
| Context | Effect |
|---------|--------|
| Scroll | Zoom timeline |

### Ctrl/Cmd Key
| Context | Effect |
|---------|--------|
| Z | Undo |
| Shift + Z | Redo |
| Y | Redo |

---

## Context-Specific

### In Timeline
| Shortcut | Action |
|----------|--------|
| `C` | Split all clips at playhead |
| Drag clip | Move clip |
| Drag clip edge | Trim clip |

### In Property Values
| Action | Effect |
|--------|--------|
| Left-click drag | Scrub value |
| Right-click | Reset to default |

### In Keyframe Area
| Action | Effect |
|--------|--------|
| Click diamond | Select keyframe |
| Drag diamond | Move keyframe |
| Right-click | Easing menu |
| Drag empty | Marquee select |

### In Preview Edit Mode
| Action | Effect |
|--------|--------|
| Drag center | Move layer |
| Drag corner | Scale layer |
| Drag outside | Rotate layer |

---

## Track Controls

| Action | Effect |
|--------|--------|
| Click eye | Toggle visibility |
| Click M | Toggle mute |
| Click S | Toggle solo |
| Double-click name | Edit track name |
| Click expand | Show keyframes |

---

## Panel Navigation

| Shortcut | Action |
|----------|--------|
| Middle mouse scroll | Cycle tabs in panel |
| Hold tab 500ms | Enable drag |

---

## Quick Reference Card

```
┌─────────────────────────────────────────┐
│           PLAYBACK                      │
│  Space = Play    ←→ = Frame step       │
│  I/O = In/Out    L = Loop               │
├─────────────────────────────────────────┤
│           EDITING                       │
│  C = Split       Del = Delete           │
│  Ctrl+Z = Undo   Ctrl+Shift+Z = Redo   │
├─────────────────────────────────────────┤
│           NAVIGATION                    │
│  Alt+Scroll = Zoom                      │
│  Shift+Scroll = H-Scroll                │
│  Shift+Drag = Snap                      │
├─────────────────────────────────────────┤
│           BLEND MODES                   │
│  Shift++ = Next   Shift+- = Previous   │
└─────────────────────────────────────────┘
```

---

## Related Features

- [Timeline](./Timeline.md) - Main editing
- [Keyframes](./Keyframes.md) - Animation
- [Preview](./Preview.md) - Playback
- [Effects](./Effects.md) - Blend modes

---

*Compiled from all commits*
