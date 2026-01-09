# UI & Panels

[← Back to Index](./README.md)

Dockable panel system, menu bar, and workspace layouts.

---

## Table of Contents

- [Panel System](#panel-system)
- [Menu Bar](#menu-bar)
- [Dock Layouts](#dock-layouts)
- [Panel Types](#panel-types)
- [Workspace](#workspace)

---

## Panel System

### Dockable Panels
All panels can be:
- Dragged to rearrange
- Grouped in tabs
- Resized
- Closed/opened

### Tab Behavior
- Click tab to activate
- Middle mouse scroll to cycle tabs
- Hold-to-drag (500ms) for reordering

### Hold-to-Drag
1. Click and hold tab for 500ms
2. Glow animation indicates ready
3. Drag to new position
4. Drop to place

### Tab Slot Indicators
Resolume-style visual feedback:
- Shows valid drop locations
- Highlights target slot
- Guides panel placement

---

## Menu Bar

After Effects-style menu bar at top.

### Menu Structure
| Menu | Contents |
|------|----------|
| **File** | New, Save, Export |
| **Edit** | Undo, Redo, Cut, Copy, Paste |
| **View** | Panel visibility, layouts |
| **Composition** | Settings, duration |

### Project Name
- Displayed at left of menu bar
- Logo removed for cleaner look
- Indicates current project

### Panel Visibility Toggles
View menu contains checkboxes for each panel:
- Toggle panels on/off
- Quick access to any panel
- Persisted in layout

---

## Dock Layouts

### Default Layout
3-column configuration:
```
┌─────────────────────────────────────────┐
│              Menu Bar                    │
├───────────┬─────────────────┬───────────┤
│  Media    │                 │  Effects  │
│  Panel    │    Preview      │  Panel    │
│           │                 │           │
│  Layers   │                 │  Props    │
│  Panel    │                 │  Panel    │
├───────────┴─────────────────┴───────────┤
│              Timeline                    │
└─────────────────────────────────────────┘
```

### Saving Layouts
- Layout auto-saved to localStorage
- Survives page refresh
- Multiple preview panels preserved

### Minimum Panel Sizes
- Panels have minimum dimensions
- Prevents collapsing too small
- Maintains usability

---

## Panel Types

### Preview Panel
- Canvas for composition output
- Composition selector dropdown
- Close button
- Edit mode toggle

### Timeline Panel
- Multi-track editor
- Composition tabs
- Playback controls
- Ruler and tracks

### Media Panel
- Media browser
- Folder organization
- Composition list
- Add button/dropdown

### Effects Panel
- Effect list
- Parameter controls
- Add effect buttons

### Clip Properties Panel
- Transform controls
- Position, Scale, Rotation
- Opacity slider
- Blend mode selector

### AI Chat Panel
- Chat interface
- Model selector
- Message history
- Default tab position

### Analysis Panel
- Real-time values
- Analysis graphs
- Per-clip data

### Export Panel
- Export settings
- Resolution options
- Format selection
- Export button

---

## Workspace

### Context Menus
- Right-click for context menus
- Stay within viewport bounds
- Solid backgrounds for readability

### Dropdown Menus
- Stay visible properly
- Account for screen edges
- Close on outside click

### Accent Colors
- Main accent: Grey
- Timeline selection: Blue
- Consistent throughout UI

### Toolbar
Combined toolbar row:
- Composition tabs
- Playback controls
- Zoom controls
- Fit button

---

## Scrolling Behavior

### Timeline Scrolling
| Input | Action |
|-------|--------|
| Scroll wheel | Vertical scroll |
| Shift + Scroll | Horizontal scroll |
| Alt + Scroll | Zoom (centered on playhead) |

### Panel Scrolling
- Standard scroll in panels
- Overflow handled gracefully
- Smooth scrolling

---

## Sticky Elements

### Timeline Header
- Sticky when scrolling vertically
- Track names always visible
- Controls accessible

### Playhead
- Extends into header row
- Always visible position
- Clear time reference

---

## Related Features

- [Timeline](./Timeline.md) - Timeline panel details
- [Preview](./Preview.md) - Preview panel details
- [Media Panel](./Media-Panel.md) - Media browser
- [Keyboard Shortcuts](./Keyboard-Shortcuts.md)

---

*Commits: 5d141c4 through d63e381*
