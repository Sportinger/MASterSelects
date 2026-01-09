# Media Panel

[← Back to Index](./README.md)

Import, organize, and manage media assets and compositions.

---

## Table of Contents

- [Importing Media](#importing-media)
- [Folder Organization](#folder-organization)
- [Compositions](#compositions)
- [Media Properties](#media-properties)
- [Drag and Drop](#drag-and-drop)

---

## Importing Media

### Supported Formats

#### Video
- MP4, WebM, MOV
- Most browser-supported codecs
- Proxy generation for large files

#### Audio
- WAV, MP3, AAC, OGG
- Up to 4GB file size
- Waveform generation

#### Images
- PNG, JPG, GIF, WebP
- Static images become clips
- Duration configurable

### Import Methods

#### Add Dropdown
1. Click "Add" button in Media Panel
2. Select file type
3. Choose files from picker

#### Drag and Drop
- Drag files directly into Media Panel
- Multiple files supported
- Auto-organizes by type

### Large File Handling
- Files >500MB skip thumbnail/waveform generation
- Preserves performance
- Can manually trigger generation later

---

## Folder Organization

### Creating Folders
1. Click "Add" → "New Folder"
2. Or right-click → "New Folder"
3. Name the folder

### Organizing Media
- **Drag and drop** media into folders
- Nest folders for hierarchy
- Collapse/expand folder tree

### Folder Features
- Visual folder icons
- Expandable tree view
- Context menu operations

---

## Compositions

### Creating Compositions
1. Click `+` button in Media Panel
2. Set composition name
3. Set duration
4. Composition appears in panel

### Composition Properties
- **Name** - Display name
- **Duration** - Length in seconds (editable)
- **Resolution** - Output dimensions

### Composition Settings Dialog
Access via right-click → "Composition Settings":
- Edit duration
- Change resolution
- Rename composition

### Nested Compositions
- Drag composition to timeline
- Creates composition clip
- Double-click to edit contents
- Changes reflect in parent

---

## Media Properties

### Viewing Properties
Select media item to see:
- File name
- Duration
- Resolution
- File size
- Format

### Thumbnails
- Generated automatically for video
- Shows first frame
- Skipped for files >500MB

### Waveforms
- Generated for audio files
- Toggle in settings
- Stored locally

---

## Drag and Drop

### To Timeline
1. Select media in panel
2. Drag to timeline
3. Drop on track or empty area

### Drop Behavior
- Creates clip from media
- Uses actual video duration
- Shows loading preview during drop

### Track Type Enforcement
- Video/images → Video tracks only
- Audio → Audio tracks only
- Prevents incorrect track placement

### From File System
- Drag files from OS file manager
- Drops into Media Panel
- Also works directly to Timeline

---

## Context Menu

Right-click media for options:

| Option | Function |
|--------|----------|
| Rename | Change display name |
| Delete | Remove from project |
| Move to Folder | Organize media |
| Generate Waveform | For audio files |
| Composition Settings | For compositions |

---

## Proxy System

### Proxy Generation
For large video files:
1. Right-click video
2. Select "Generate Proxy"
3. Choose storage folder

### Proxy Folder
- First-time prompts for folder selection
- Uses File System Access API
- Stores proxy files externally

### Proxy Usage
- Editor uses proxy for preview
- Full-res used for export
- Seamless switching

---

## Project Persistence

### Auto-Save
- Media references saved to IndexedDB
- Survives page refresh
- Includes folder structure

### Media File IDs
- Each media has unique ID
- Clips reference media by ID
- Enables restore on reload

---

## Related Features

- [Timeline](./Timeline.md) - Using media in edits
- [Audio](./Audio.md) - Audio media handling
- [Export](./Export.md) - Rendering output
- [UI Panels](./UI-Panels.md) - Panel system

---

*Commits: d6e130c through d63e381*
