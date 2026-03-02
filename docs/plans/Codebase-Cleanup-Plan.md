# Codebase Cleanup Plan

> **Goal:** Clean folder structure, easy to maintain, slick root, best practices

## Status: Phase 1 COMPLETE ✓

**Completed:**
- [x] Updated .gitignore (terminal files, test videos, Rust targets)
- [x] Created `tools/` folder with all build tools
- [x] Organized docs/ (architecture, COMPLETED folder)
- [x] Updated CLAUDE.md with new paths
- [x] Removed duplicate folders from root

---

---

## Overview

### Current Issues
- Root folder has 15+ items (cluttered)
- Build tools scattered in root
- Docs files not organized
- Terminal state files not gitignored
- Test videos (222MB) in repo

### Target State
- Root: 10 essential items only
- All tools under `tools/`
- All docs organized by purpose
- Proper .gitignore

---

## Phase 1: Immediate Cleanup (No Code Changes)

### 1.1 Update .gitignore

Add these entries:
```gitignore
# Terminal state (Claude Code)
.terminal_*

# Test videos (large files)
public/test-videos/

# Build artifacts
dist/test_prores.mov
```

### 1.2 Create tools/ folder structure

```bash
mkdir -p tools
mv helpers tools/
mv native-helper tools/
mv ffmpeg-build tools/
mv ffmpeg-wasm-build tools/
```

**Result:**
```
tools/
├── helpers/              # Rust native helpers (per-OS)
│   ├── win/             # Windows: YouTube download only
│   ├── linux/           # Linux: Full FFmpeg decoder
│   └── mac/             # macOS: Full FFmpeg decoder
├── native-helper/        # Electron tray app
├── ffmpeg-build/         # FFmpeg Docker build
└── ffmpeg-wasm-build/    # FFmpeg WASM build
```

### 1.3 Organize docs/

```bash
# Create architecture folder
mkdir -p docs/architecture

# Move root doc
mv codeplan.md docs/architecture/

# Move scattered docs to proper locations
mv docs/MobilePlan.md docs/plans/
mv docs/FEATURE_MULTICAM_AI.md docs/Features/Multicam-AI.md
mv docs/webvj-mixer-plan.md docs/plans/

# Archive completed refactor plans
mkdir -p docs/refactor/COMPLETED
mv docs/refactor/ClipSlice-Refactor-Plan.md docs/refactor/COMPLETED/
mv docs/refactor/MediaStore-Refactor-Plan.md docs/refactor/COMPLETED/
```

**Result:**
```
docs/
├── Features/             # User-facing feature docs
│   ├── README.md
│   ├── Multicam-AI.md   # ← moved from root docs/
│   └── *.md
├── plans/                # Implementation plans
│   ├── MobilePlan.md    # ← moved
│   ├── webvj-mixer-plan.md  # ← moved
│   └── *.md
├── refactor/             # Refactoring guides
│   ├── COMPLETED/       # Archive
│   │   ├── ClipSlice-Refactor-Plan.md
│   │   └── MediaStore-Refactor-Plan.md
│   ├── FrameExporter-Refactor-Plan.md
│   ├── Timeline-Refactor-Plan.md
│   └── WebGPUEngine-Refactor-Plan.md
└── architecture/         # High-level architecture
    └── codeplan.md      # ← moved from root
```

### 1.4 Update CLAUDE.md

Update the Native Helper section to reflect new path:
```markdown
# Native Helper

```bash
# Standard (if FFmpeg < 8.0):
cd tools/helpers/linux && cargo run --release

# Windows (YouTube download only):
cd tools/helpers/win && cargo run --release
```
```

---

## Phase 2: After Refactors Complete

### 2.1 After WebGPUEngine Refactor

Move video-related files into `src/engine/video/`:
```bash
mv src/engine/WebCodecsPlayer.ts src/engine/video/
mv src/engine/ParallelDecodeManager.ts src/engine/video/
```

Update imports in affected files.

### 2.2 After FrameExporter Refactor

The old `src/engine/FrameExporter.ts` will be replaced by `src/engine/export/` folder.
Verify all imports updated.

### 2.3 After Timeline Refactor

New structure under `src/components/timeline/`:
```
src/components/timeline/
├── Timeline.tsx              # Main (~900 LOC, was 2109)
├── hooks/
│   ├── useExternalDrop.ts   # NEW
│   ├── usePlaybackLoop.ts   # NEW
│   ├── useVideoPreload.ts   # NEW
│   └── useAutoFeatures.ts   # NEW
├── components/
│   ├── TimelineOverlays.tsx # NEW
│   └── NewTrackDropZone.tsx # NEW
└── utils/
    └── fileTypeHelpers.ts   # NEW
```

---

## Final Root Structure

```
masterselects/
├── .git/
├── .gitignore
├── CLAUDE.md
├── README.md
├── index.html
├── package.json
├── package-lock.json
├── tsconfig.json
├── tsconfig.app.json
├── tsconfig.node.json
├── vite.config.ts
├── eslint.config.js
├── masterselects.ico
│
├── docs/                 # All documentation
├── public/               # Static assets
├── src/                  # Source code
├── tools/                # Build tools & native helpers
├── node_modules/         # Dependencies (gitignored)
└── dist/                 # Build output (gitignored)
```

**Root items: 13 files + 5 folders = 18 items** (down from 25+)

---

## Validation Checklist

After Phase 1:
- [ ] `npm run dev` works
- [ ] `npm run build` works
- [ ] `npm run lint` works
- [ ] Native helper runs from `tools/helpers/`
- [ ] All doc links work

After Phase 2:
- [ ] All imports resolve
- [ ] TypeScript compiles: `npx tsc --noEmit`
- [ ] Export functionality works
- [ ] Timeline functionality works

---

## Rollback

If issues occur:
```bash
git checkout .
git clean -fd
```

---

## Summary Table

| Change | Before | After |
|--------|--------|-------|
| Root items | 25+ | 18 |
| Tools location | Scattered | `tools/` |
| Docs location | Mixed | Organized |
| Test videos | In repo | Gitignored |
| Terminal files | In repo | Gitignored |
| Completed refactors | In active folder | Archived |
