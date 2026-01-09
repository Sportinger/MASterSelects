# Project Persistence

[← Back to Index](./README.md)

Save and restore projects across browser sessions using IndexedDB.

---

## Table of Contents

- [Auto-Save](#auto-save)
- [What Gets Saved](#what-gets-saved)
- [Database Structure](#database-structure)
- [New Project](#new-project)

---

## Auto-Save

### Automatic Saving
Project state saved automatically:
- After timeline changes
- After transcription completes
- After analysis data generated
- On significant user actions

### Storage Location
- Uses browser's IndexedDB
- Persists across page refresh
- Survives browser restart

### No Manual Save Required
- Changes saved immediately
- No "Save" button needed
- No risk of data loss

---

## What Gets Saved

### Timeline Data
- All tracks and clips
- Clip positions and durations
- Trim points (in/out)
- Transform properties

### Compositions
- All compositions
- Nested composition structure
- Composition settings
- Tab order

### Media References
- Media file IDs
- File associations
- Folder organization
- Original file paths

### Clip Data
- Keyframe animations
- Effect parameters
- Mask shapes
- Blend modes

### Analysis Data
- Focus analysis
- Motion analysis
- Brightness data
- Face detection results

### Transcripts
- Word-level timestamps
- Language settings
- Full transcript text

---

## Database Structure

### IndexedDB Schema
```
MASterSelects DB
├── projects
│   └── Project data
├── compositions
│   └── Composition definitions
├── clips
│   └── Clip data with keyframes
├── media
│   └── Media file references
├── analysis
│   └── Clip analysis results
└── transcripts
    └── Transcript data
```

### Restoration Process
1. Page loads
2. IndexedDB queried
3. State reconstructed
4. Media files re-linked
5. UI populated

---

## New Project

### Creating New Project
1. File menu → New Project
2. Or "New Project" button
3. Clears current state
4. Starts fresh

### Warning
- New project clears all data
- No confirmation currently
- Consider exporting first

---

## Media File Handling

### File References
- Media stored by reference
- Original files stay on disk
- File System Access API for paths

### Re-linking Media
If files moved:
- Media shows as missing
- Re-import required
- New file ID assigned

### Media File IDs
- Each media has unique ID
- Clips reference by ID
- Survives project reload

---

## Layout Persistence

### Dock Layout
- Panel positions saved
- Tab arrangements saved
- Panel sizes saved
- Multiple previews preserved

### Storage
- Uses localStorage
- Separate from project data
- Global across projects

---

## Troubleshooting

### Data Not Restoring
1. Check browser IndexedDB
2. Clear cache if corrupted
3. Check console for errors

### Missing Media
- Re-import files
- Check file paths
- Verify file access permissions

### Reset Project
- Clear IndexedDB manually
- Or create new project
- DevTools → Application → IndexedDB

---

## Related Features

- [Media Panel](./Media-Panel.md) - Media management
- [Timeline](./Timeline.md) - Timeline data
- [Audio](./Audio.md) - Transcript persistence
- [UI Panels](./UI-Panels.md) - Layout saving

---

*Commits: d504a98 through d63e381*
