# Timeline

[← Back to Index](./README.md)

The Timeline is the core editing interface, providing multi-track video and audio editing with composition support.

---

## Table of Contents

- [Track Types](#track-types)
- [Clips](#clips)
- [Snapping & Resistance](#snapping--resistance)
- [Selection](#selection)
- [Compositions](#compositions)
- [Multicam Support](#multicam-support)
- [Track Controls](#track-controls)

---

## Track Types

### Video Tracks
- Support video files, images, and nested compositions
- Stack from top to bottom (top track = front layer)
- Expandable to show keyframe properties
- Default: 2 video tracks

### Audio Tracks
- Audio-only tracks at bottom of timeline
- Waveform visualization (50 samples/second)
- Linked audio follows video clip movement
- Default: 1 audio track

### Track Management
```
addTrack()           - Create video/audio track
removeTrack()        - Delete track
renameTrack()        - Change track name (double-click)
setTrackHeight()     - Resize track
scaleTracksOfType()  - Batch height adjustment
```

---

## Clips

### Adding Clips
1. Drag from Media Panel to timeline
2. Shows dashed preview with actual duration during drag
3. Thumbnails generated in background (skipped for >500MB)

### Clip Operations

| Action | Method | Notes |
|--------|--------|-------|
| **Move** | Drag clip | Supports snapping + resistance |
| **Trim** | Drag edges | Left/right trim handles |
| **Split** | `C` key | Cut tool - splits all clips at playhead |
| **Copy** | `Ctrl+C` | Copy selected clips |
| **Paste** | `Ctrl+V` | Paste clips at playhead |
| **Delete** | `Delete` key | Removes selected clips |
| **Reverse** | Context menu | Shows ↻ badge |

### Cut Tool
- **Shortcut:** `C`
- **Snapping:** Automatically snaps to clip edges (hold `Alt` to disable)
- **Linked clips:** Splits both video and audio together
- **Visual indicator:** Cut line extends across all linked clips

### Copy/Paste
- **Copy:** `Ctrl+C` copies selected clips with all properties
- **Paste:** `Ctrl+V` pastes at playhead position
- **Preserved:** Effects, keyframes, masks, thumbnails, waveforms
- **Undo support:** Full undo/redo for paste operations

### Clip Properties (Keyframeable)
- Position (X, Y, Z depth)
- Scale (X, Y)
- Rotation (X, Y, Z) - full 3D with perspective
- Opacity (0-100%)
- **Speed** (-400% to 400%)

### Speed Control
The Speed property controls playback rate with full keyframe support:

| Speed Value | Effect |
|-------------|--------|
| 100% | Normal playback |
| 50% | Slow motion (2x longer) |
| 200% | Fast forward (2x faster) |
| 0% | Freeze frame |
| -100% | Reverse playback |

**Features:**
- Keyframeable with bezier curves for smooth ramps
- Negative values play backwards
- Works with RAM Preview
- Speed changes affect source time through integration

**Implementation:**
- Source time = integral of speed curve over clip duration
- Supports smooth transitions between speeds
- Handles direction changes (forward to reverse)

### Linked Clips
- Video clips can have linked audio
- Alt+drag to move independently
- Split together with `C` key
- Visual indicator: linked clips move together
- **Linked selection:** Click a linked video/audio clip to select both
- **Independent selection:** Shift+click for selecting only one side

### Multi-Select Movement
- **Shift+Click** to select multiple clips
- Drag any selected clip to move all together
- Group boundary collision prevents clips from overlapping
- Visual preview shown for all selected clips during drag
- Audio/video stay in sync during multi-drag

---

## Snapping & Resistance

### Snap Toggle
Toolbar button to enable/disable magnetic snapping:
- Click magnet icon to toggle
- Active state shows highlighted button
- Tooltip shows current status

### Magnetic Snapping
When enabled:
- **Snap distance**: 0.1 seconds
- **Snap points**: Clip edges, timeline start (0s)
- Automatic edge-to-edge alignment

### Overlap Resistance
When dragging clips over others:
- **100px horizontal resistance** must be pushed through
- **100px vertical resistance** prevents accidental cross-track moves
- Visual `.forcing-overlap` feedback
- Auto-trims overlapped clips when forced
- Smart overlap prevention on track changes: find free track or create new one

### Implementation
```typescript
getSnappedPosition()      - Calculate snap-adjusted position
getPositionWithResistance() - Snap + resistance calculation
trimOverlappingClips()    - Auto-trim when placing
```

---

## Selection

### Clip Selection
| Action | Effect |
|--------|--------|
| Click | Select single clip |
| Ctrl+Click | Add/remove from selection |
| Click empty | Deselect all |
| Escape | Deselect all |

### Marquee Selection
- Click and drag on empty timeline area
- Rectangle selects all clips it touches
- Shift+marquee extends/subtracts selection
- Live visual feedback during drag

### Keyframe Selection
- Click keyframe diamond to select
- Shift+click for multi-select
- `Delete` removes selected keyframes (priority over clips)
- See [Keyframes](./Keyframes.md) for details

---

## Compositions

### Creating Compositions
1. Media Panel → Add → Composition
2. Set name, resolution, frame rate
3. Composition appears in Media Panel

### Composition Settings
- **Resolution**: Up to 7680×4320 (8K)
- **Frame rates**: 23.976, 24, 25, 29.97, 30, 50, 59.94, 60 fps
- **Duration**: Editable in timeline controls

### Nested Compositions
- Drag composition from Media Panel to Timeline
- Double-click to enter and edit contents
- Changes reflect in parent composition
- Recursive rendering for deep nesting
- **Orange outline** for easy identification
- **Boundary markers** show clip start/end positions
- **Content-aware thumbnails** sample at clip boundaries

### Fade Curves (Bezier)
Visual opacity fade curves displayed directly on timeline clips:
- **Creating fades:** Add opacity keyframes at clip start/end
- **Bezier visualization:** Shows smooth fade curve on clip
- **Real-time updates:** Curves update instantly during adjustment
- **Fade handles:** Drag to adjust fade duration while preserving easing

### Composition Tabs
- Open compositions appear as tabs
- Click to switch between compositions
- Drag tabs to reorder
- Each composition has independent timeline data

---

## Multicam Support

### Linked Groups
Multiple clips can be grouped for synchronized movement:

```typescript
createLinkedGroup()  - Group clips with offsets
unlinkGroup()        - Remove group relationship
```

### Group Behavior
- All clips in group move together
- Alt+drag to skip group movement
- Visual indicator: ⊝ badge
- Stored offsets maintain sync timing

### Audio Sync
See [Audio](./Audio.md#multicam-sync) for cross-correlation sync.

---

## Track Controls

Each track header contains:

| Control | Function |
|---------|----------|
| **Eye** (👁) | Toggle track visibility |
| **M** | Mute track audio |
| **S** | Solo this track |
| **Name** | Double-click to edit |
| **Expand** (▶) | Show keyframe lanes |

### Solo Behavior
- Dims non-solo tracks visually
- Multiple tracks can be solo'd
- Quick way to isolate content

### Track Height
- Drag track dividers to resize with continuous scrolling (no fixed steps)
- Minimum 20px for ultra-compact view with single line of text
- Expanded tracks show property rows
- Height auto-adjusts for curve editors

---

## Playback Controls

Located in timeline toolbar:

| Control | Shortcut | Function |
|---------|----------|----------|
| Stop | - | Return to time 0 |
| Play/Pause | `Space` | Toggle playback |
| JKL Shuttle | `J`/`K`/`L` | Reverse / Pause / Forward playback |
| Loop | `L` (no focus) | Toggle loop mode |
| In Point | `I` | Set at playhead |
| Out Point | `O` | Set at playhead |
| Clear I/O | `X` | Clear markers |
| Go to Start | `Home` | Jump to beginning |
| Go to End | `End` | Jump to end |

### Duration Editing
- Click duration display to edit
- Enter new duration, press Enter
- Locks duration (won't auto-extend)

---

## Performance Features

### Thumbnails
- Auto-generated for video clips
- Toggle: "Thumb On/Off" button
- Skipped for files >500MB

### Waveforms
- Generated for audio clips
- Toggle: "Wave On/Off" button
- 50 samples per second resolution

### Keyframe Tick Marks
- Small amber diamond markers at the bottom of clips
- Show keyframe positions without expanding tracks
- Visible at all zoom levels

### Timeline Zoom
- **Alt+Scroll**: Exponential zoom (8% per step) centered on playhead
- Consistent zoom feel at all zoom levels

### Vertical Scroll Snapping
- Vertical scrolling snaps to track boundaries
- Each scroll step moves exactly one layer

### Video/Audio Separator
- Green divider line between video and audio tracks
- Clearer visual structure for track organization

### Clip Entrance Animations
- When switching compositions, clips animate in with entrance transitions
- Animation phases: `exiting` (old clips fade out) then `entering` (new clips animate in)
- Controlled by `clipEntranceAnimationKey` which increments on each composition switch
- Only clips present at the time of the switch receive the animation class

### RAM Preview
- Toggle: "RAM ON/OFF" button
- Caches 30fps frames
- Green indicator shows cached ranges
- See [Preview](./Preview.md#ram-preview)

---

## Related Features

- [Keyframes](./Keyframes.md) - Animate clip properties
- [Preview](./Preview.md) - Playback and RAM Preview
- [Audio](./Audio.md) - Audio tracks and multicam sync
- [Keyboard Shortcuts](./Keyboard-Shortcuts.md)

---

*Source: `src/components/timeline/`, `src/stores/timeline/`*
