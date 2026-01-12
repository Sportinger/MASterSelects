# Media Panel

[← Back to Index](./README.md)

Import, organize, and manage media assets with folder structure and proxy generation.

---

## Table of Contents

- [Importing Media](#importing-media)
- [Folder Organization](#folder-organization)
- [Compositions](#compositions)
- [Proxy Generation](#proxy-generation)
- [Context Menu](#context-menu)

---

## Importing Media

### Supported Formats

| Type | Formats |
|------|---------|
| **Video** | MP4, WebM, MOV |
| **Audio** | WAV, MP3, AAC, OGG (up to 4GB) |
| **Image** | PNG, JPG, GIF, WebP |

### Import Methods

#### Add Dropdown
1. Click "Add" button
2. Select: Import Media, New Composition, New Folder
3. Choose files from picker

#### Drag and Drop
- Drag files directly into Media Panel
- Multiple files supported
- Auto-organizes by type

### File System Access API
When supported (Chrome/Edge):
- Native file picker
- Persistent file handles
- Path information available

### Large File Handling
| Size | Behavior |
|------|----------|
| < 500MB | Full thumbnails/waveforms |
| > 500MB | Skip auto-generation |
| > 4GB | Audio waveform skipped |

---

## Folder Organization

### Creating Folders
1. Add dropdown → New Folder
2. Or right-click → New Folder
3. Name the folder

### Folder Features
- **Nested folders** supported
- **Drag-and-drop** items into folders
- **Expand/collapse** tree view
- **Cycle detection** prevents invalid nesting

### Operations
```typescript
createFolder(name, parentId?)  // Create folder
removeFolder(id)               // Delete (moves children to parent)
renameFolder(id, name)         // Rename
toggleFolderExpanded(id)       // Toggle expand
moveToFolder(itemIds[], folderId) // Move items
```

---

## Compositions

### Creating Compositions
1. Add dropdown → Composition
2. Configure settings:
   - Name
   - Width (1-7680)
   - Height (1-4320)
   - Frame rate

### Frame Rate Options
```
23.976, 24, 25, 29.97, 30, 50, 59.94, 60 fps
```

### Composition Operations
```typescript
createComposition(name, settings?)
duplicateComposition(id)        // Creates "Name Copy"
removeComposition(id)
updateComposition(id, updates)
openCompositionTab(id)          // Edit in timeline
closeCompositionTab(id)
```

### Nested Compositions
- Drag composition to timeline
- Double-click to edit contents
- Changes reflect in parent

---

## Proxy Generation

### GPU-Accelerated Proxies
For large video files:
1. Right-click video
2. Select "Generate Proxy"
3. Choose storage folder (first time)

### Proxy Settings
```typescript
FPS: 30
Quality: 0.92 (WebP)
Max width: 1920px (maintains aspect)
```

### Storage Options
- **IndexedDB** - Browser storage
- **File System** - External folder (via File System Access API)

### Progress Tracking
```typescript
interface MediaFile {
  proxyStatus: 'none' | 'generating' | 'ready' | 'error';
  proxyProgress: number; // 0-100
}
```

### Visual Indicators
| Badge | Meaning |
|-------|---------|
| **P** | Proxy ready |
| **⏳ X%** | Generating |

---

## Context Menu

### Media Files
- Rename
- Delete
- Generate Proxy / Stop Proxy / Proxy Ready
- Show in Explorer → Raw / Proxy

### Compositions
- Rename
- Composition Settings
- Duplicate
- Delete

### Folders
- Rename
- Delete

---

## Media Properties

### Column Display
The media list displays items in a table with the following columns:

| Column | Description | Example |
|--------|-------------|---------|
| **Name** | File name with icon and thumbnail | 🎥 Video.mp4 |
| **Duration** | Clip length (mm:ss) | 4:02 |
| **Resolution** | Width × Height | 1920×1080 |
| **FPS** | Frame rate | 25 |
| **Container** | File container format | MP4, MKV, WebM |
| **Codec** | Video/audio codec | H.264, VP9 |
| **Size** | File size | 125.4 MB |

### Column Customization

**Reorder Columns:**
- Drag column headers to rearrange
- Order is saved in localStorage

**Resize Name Column:**
- Drag the vertical line between Name and other columns
- Width is saved in localStorage (120px - 500px)

### Horizontal Scrolling
When columns don't fit in the panel width, scroll horizontally with mouse.

### Metadata
```typescript
interface MediaFile {
  id: string;
  name: string;
  file: File;
  type: 'video' | 'audio' | 'image';
  duration?: number;
  width?: number;
  height?: number;
  fps?: number;           // Frame rate (video)
  codec?: string;         // H.264, VP9, AAC, etc.
  container?: string;     // MP4, MKV, WebM, etc.
  fileSize?: number;      // File size in bytes
  thumbnailUrl?: string;
}
```

---

## Drag to Timeline

### Process
1. Select media in panel
2. Drag to timeline
3. Drop on track or empty area

### Drop Behavior
- Creates clip from media
- Uses actual video duration
- Shows loading preview

### Track Type Enforcement
| Media Type | Allowed Tracks |
|------------|----------------|
| Video/Image | Video tracks only |
| Audio | Audio tracks only |

---

## Project Integration

### Auto-Save
Media references saved with project to IndexedDB.

### Restoration
On project load:
- Media files reconstructed from IndexedDB
- Blob URLs regenerated
- Folder structure restored

### Media File IDs
- Each media has unique ID
- Clips reference media by ID
- Survives project reload

---

## Media Relinking

### Relink Dialog
When media files are missing or moved:
1. **Automatic detection** - App detects missing files on project load
2. **Relink dialog** - Shows list of missing files
3. **Auto-scan** - Scans folders to find matching files
4. **Manual browse** - Locate individual files manually

### Reload All
Toolbar button to restore file permissions:
- Click when files show "needs reload" indicator
- Re-requests access to all file handles
- Useful after browser restart

### Visual Indicators
| Badge | Meaning |
|-------|---------|
| ⚠️ Yellow | Permission lost, click to reload |
| ❌ Red | File missing, needs relink |
| P (blue) | Proxy available |

---

## Not Implemented

- Cloud storage integration
- Asset library across projects
- Batch import settings

---

## Related Features

- [Timeline](./Timeline.md) - Using media in edits
- [Audio](./Audio.md) - Audio media handling
- [Project Persistence](./Project-Persistence.md) - Saving
- [Export](./Export.md) - Rendering output

---

*Source: `src/components/panels/MediaPanel.tsx`, `src/stores/mediaStore.ts`*
