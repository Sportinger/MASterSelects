# Project Persistence

[← Back to Index](./README.md)

Local project folder storage with auto-save, backups, and smart media relinking.

---

## Table of Contents

- [Welcome Overlay](#welcome-overlay)
- [Project Folder Structure](#project-folder-structure)
- [Auto-Save](#auto-save)
- [Backup System](#backup-system)
- [Media Relinking](#media-relinking)
- [What Gets Saved](#what-gets-saved)
- [Project Management](#project-management)

---

## Welcome Overlay

### First Launch
On first launch or when no project is open, the Welcome Overlay appears:
- Animated entrance with blur backdrop
- "Local. Private. Free." tagline
- Two options: **New Project** or **Open Existing**

### Browser Warning
If using a non-Chromium browser (Safari, Firefox), a red warning banner appears:
- **"Unsupported Browser"** label with detected browser name displayed prominently
- Explains that WebGPU requires Google Chrome (with download link)
- Notes that users can continue without saving, but many features won't work

### Select Project Folder
1. Click "Select Project Folder"
2. Choose or create a folder for your project
3. App creates `project.json` and required subfolders
4. Folder remembered for future sessions

### Continue Without Saving
- Work without persistence
- Project lost on refresh
- Useful for quick experiments

---

## Project Folder Structure

Projects are stored in a local folder you choose:

```
MyProject/
├── project.json           # Main project file
├── Raw/                   # Auto-copied media files (portable)
│   ├── Interview_01.mp4
│   └── Music.wav
├── YT/                    # Downloaded YouTube videos
│   └── video_title.mp4
├── Backups/               # Auto-backup folder
│   ├── project_2026-01-11_14-00-00.json
│   └── ... (last 20 backups)
├── Proxy/                 # Generated proxy frames
│   └── {mediaHash}/       # Per-file proxy data
│       └── frames/        # WebP proxy frames
├── Thumbnails/            # Media thumbnails
│   └── {mediaHash}.webp
└── Analysis/              # Clip analysis data
    └── {mediaHash}.json
```

### Auto-Copy to Raw Folder
When importing media files, they are **automatically copied** to the project's `Raw/` folder:
- Original files remain untouched at their source location
- Project becomes portable — copy the folder to another machine
- No broken links when moving projects between computers

### Auto-Relink from Raw Folder
When opening a project with missing media files:
- App automatically checks the `Raw/` folder for matching files
- Files are restored from Raw without user intervention
- Eliminates manual relinking for projects with Raw copies

### Benefits of Local Storage
- **No browser storage limits** - Use as much disk space as needed
- **Portable projects** - Copy folder (including Raw/) to move between machines
- **External backup** - Use any backup tool on the folder
- **Version control** - Can use Git for project history

---

## Auto-Save

### Autosave Configuration
Access via **File → Autosave** submenu:

| Setting | Options | Default |
|---------|---------|---------|
| Enable Autosave | On/Off | Off |
| Interval | 1, 2, 5, 10 minutes | 5 min |

### Automatic Triggers
- Configurable interval (1-10 minutes)
- On page unload (beforeunload)
- When switching compositions
- After transcription completes
- After analysis completes

### Manual Save
- `Ctrl+S` shortcut
- File menu → Save
- Shows yellow "Saved" toast in center of screen

### No Data Loss
- Changes saved immediately
- Survives page refresh
- Survives browser restart

---

## Backup System

### How It Works
Before each **autosave**, the current project file is automatically backed up:
1. Copy current `project.json` to `Backups/` folder
2. Name format: `project_2026-01-11_14-30-00.json`
3. Then save to main `project.json`

### Backup Storage
```
ProjectFolder/
├── project.json          # Current project
├── Backups/
│   ├── project_2026-01-11_14-00-00.json
│   ├── project_2026-01-11_14-05-00.json
│   └── ... (last 20 backups)
```

### Automatic Cleanup
- Keeps only the **last 20 backups**
- Oldest backups automatically deleted
- Sorted by file timestamp

### Restoring from Backup
1. Navigate to `ProjectFolder/Backups/`
2. Find backup by timestamp
3. Rename to `project.json`
4. Replace main `project.json`
5. Reopen project

---

## Media Relinking

### Auto-Reconnect
When opening a project, the app automatically:
1. Checks if media source folder is accessible
2. Requests permission if needed
3. Scans for files matching project media

### Relink Dialog
When media files are missing, the **Relink Dialog** appears:

| Feature | Description |
|---------|-------------|
| Missing files list | Shows all files that need relinking |
| Auto-scan | Scans selected folder for matching files |
| Manual relink | Browse for individual files |
| Skip | Continue with missing media (clips show warning) |

### Smart Relink
The app attempts to auto-match files by:
1. **Exact filename match** - Same name in new location
2. **Hash match** - Same content (file hash)
3. **Similar name** - Fuzzy matching for renamed files

### Reload All Button
In Media Panel toolbar:
- Click "Reload All" to restore file permissions
- Useful after browser restart
- Re-requests access to stored file handles

### Visual Indicators
| Indicator | Meaning |
|-----------|---------|
| ⚠️ Yellow badge | File needs reload (permission lost) |
| ❌ Red badge | File missing (needs relink) |
| ✓ Normal | File accessible |

---

## What Gets Saved

### Project Data (project.json)
```typescript
interface StoredProject {
  id: string;
  name: string;
  createdAt: number;
  updatedAt: number;
  compositions: Composition[];
  folders: Folder[];
  mediaFiles: MediaFileMetadata[];
  openTabs: string[];
  expandedFolders: string[];
  mediaSourceFolder?: string;
}
```

### Per Composition
- All tracks and clips
- Clip positions and durations
- Trim points (inPoint/outPoint)
- Transform properties
- Keyframe animations
- Effect parameters
- Mask shapes

### Media Metadata
- File paths (relative to source folder)
- Duration, dimensions, FPS
- Codec and container info
- File hash for deduplication

### Stored in Project Folder
| Location | Contents |
|----------|----------|
| `project.json` | Main project data |
| `Backups/` | Auto-backup files |
| `Proxy/` | Proxy frames (WebP) |
| `Thumbnails/` | Media thumbnails |
| `Analysis/` | Clip analysis cache |

---

## Project Management

### New Project
- File menu → New Project
- Or `Ctrl+N`
- Prompts to save current project first

### Save Project
- `Ctrl+S` saves to project folder
- Shows yellow "Saved" toast
- Updates `project.json`

### Save As
- File menu → Save As
- Choose new folder location
- Copies project to new location

### Open Existing Project
- From Welcome Overlay: "Open Existing Project"
- Or File menu → Open Project
- Select folder containing `project.json`

---

## Layout Persistence

### Dock Layout
Separate from project data:
- Panel positions
- Tab arrangements
- Panel sizes
- Stored in localStorage

### View Toggle Persistence
View toggle states are saved in the project file:
- Thumbnail visibility (on/off)
- Waveform visibility (on/off)
- Scopes visibility
- Restored when opening a project

### Output Manager Persistence
The Output Manager window state is tracked via `localStorage`:
- `masterselects-om-open` key stores whether the Output Manager was open
- On page refresh, the app detects the existing popup and reconnects via `reconnectOutputManager()`
- Uses `sessionStorage` guard to prevent false reconnection on fresh tabs
- Window position and size preserved by the browser's named window (`output_manager`)

### Composition Resolution Persistence
Each composition stores its own resolution (width/height) in the project file:
- Resolution is saved per composition, not globally
- Changing resolution adjusts clip transforms proportionally (auto-reposition)
- Restored when opening a project or switching compositions

### Actions
```typescript
saveLayoutAsDefault()  // View menu
resetLayout()          // View menu
```

---

## Troubleshooting

### IndexedDB Error Dialog
If IndexedDB storage becomes corrupted, an error dialog appears automatically:
- Explains the issue and provides instructions for clearing site data
- Offers a "Refresh" button to reload the app after clearing
- Dismissable via Escape key or backdrop click
- Source: `src/components/common/IndexedDBErrorDialog.tsx`

### Project Not Loading
1. Check if `project.json` exists in folder
2. Verify folder permissions
3. Check browser console for errors

### Missing Media After Reload
1. Click "Reload All" in Media Panel
2. Or use Relink Dialog to locate files
3. Check if source folder is accessible

### Restore from Backup
1. Navigate to `ProjectFolder/Backups/`
2. Find backup by timestamp
3. Copy to `project.json` (rename existing first)
4. Reopen project

---

## Storage Comparison

| Storage | Used For | Limits |
|---------|----------|--------|
| **Project Folder** | Project, proxies, analysis | Disk space |
| **IndexedDB** | File handles, preferences | ~50MB |
| **localStorage** | Dock layout, settings | ~5MB |

---

## Related Features

- [Media Panel](./Media-Panel.md) - Media management
- [Timeline](./Timeline.md) - Timeline data
- [Audio](./Audio.md) - Transcript persistence
- [UI Panels](./UI-Panels.md) - Layout saving

---

*Source: `src/services/projectDB.ts`, `src/services/fileSystemService.ts`, `src/stores/mediaStore.ts`*
