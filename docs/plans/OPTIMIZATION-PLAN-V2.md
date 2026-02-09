# React Optimization Plan v2

**Created:** 2026-02-04
**Updated:** 2026-02-04
**Based on:** Full codebase audit + REACT-BEST-PRACTICES.md

---

## Completion Status

| Phase | Task | Status |
|-------|------|--------|
| 1.1 | PropertiesPanel store subscriptions | ✅ DONE (a739556) |
| 1.2 | MediaPanel store subscriptions | ✅ DONE (93d5ab9) |
| 1.3 | Toolbar polling fix | ✅ DONE (c09635d) |
| 2.1 | PropertiesPanel component split | ✅ DONE (dbd9a0f) |
| 2.2 | useLayerSync split | ⏭️ SKIPPED (high complexity, medium impact) |
| 3.1 | Inline arrow functions | ⏭️ N/A (components now split, marginal benefit) |
| 4.x | Low priority items | ⏸️ DEFERRED |

**Key Results:**
- PropertiesPanel split into 8 lazy-loaded modules (61K → ~3-7K each)
- Eliminated infinite re-render loop from store action selectors
- Fixed 12 nested components subscribing to entire store
- Reduced MediaPanel store subscriptions from 30+ to 8 reactive values
- Reduced Toolbar polling from 500ms to 2000ms

---

## Executive Summary

After reviewing 56 component files, 23 hooks, and 52 store files, I identified **15 high-impact optimization opportunities**. The main issues are:

1. **Store subscription anti-patterns** in PropertiesPanel, MediaPanel, Toolbar
2. **Polling loops** that could use event-driven updates
3. **Inline arrow functions** causing unnecessary re-renders
4. **Large components** that should be split

---

## Phase 1: CRITICAL - Store Subscription Fixes

### 1.1 PropertiesPanel.tsx - Nested Component Store Subscriptions (61K)

**Problem:** Multiple nested components subscribe to the ENTIRE timeline store:

```typescript
// ❌ BAD - KeyframeToggle.tsx (line 42)
function KeyframeToggle({ clipId, property, value }: KeyframeToggleProps) {
  const { isRecording, toggleKeyframeRecording, hasKeyframes, addKeyframe } = useTimelineStore();
  // Subscribes to ALL state changes, re-renders on every playhead move!
}

// ❌ BAD - TransformTab (line 271)
function TransformTab({ clipId, transform, speed = 1 }: TransformTabProps) {
  const { setPropertyValue, updateClipTransform } = useTimelineStore();
  // Creates new object reference every render
}

// ❌ BAD - VolumeTab (line 395)
const { setPropertyValue, getInterpolatedEffects, playheadPosition, clips, ... } = useTimelineStore();

// ❌ BAD - MaskItem (line 826)
const { updateMask, removeMask, setActiveMask, setMaskEditMode } = useTimelineStore();

// ❌ BAD - TranscriptTab (line 997)
const { setPlayheadPosition, playheadPosition } = useTimelineStore();
```

**Solution:** Use `getState()` for actions, selectors for reactive data:

```typescript
// ✅ GOOD - Actions from getState() (stable, no subscription)
function KeyframeToggle({ clipId, property, value }: KeyframeToggleProps) {
  const store = useTimelineStore.getState();
  const { isRecording, toggleKeyframeRecording, hasKeyframes, addKeyframe } = store;
}

// ✅ GOOD - Only subscribe to needed reactive data
function TranscriptTab({ ... }) {
  const playheadPosition = useTimelineStore(state => state.playheadPosition);
  const setPlayheadPosition = useTimelineStore.getState().setPlayheadPosition;
}
```

**Files to modify:**
- `src/components/panels/PropertiesPanel.tsx`

**Impact:** HIGH - PropertiesPanel renders on every frame during playback due to these subscriptions

---

### 1.2 MediaPanel.tsx - Mass Destructuring (48K)

**Problem:** Destructures ~30 properties from useMediaStore in one call:

```typescript
// ❌ BAD - Line 57-89
export function MediaPanel() {
  const {
    files, compositions, folders, selectedIds, expandedFolderIds,
    importFiles, importFilesWithPicker, createComposition, createFolder,
    removeFile, removeComposition, removeFolder, renameFile, renameFolder,
    reloadFile, toggleFolderExpanded, setSelection, addToSelection,
    getItemsByFolder, openCompositionTab, updateComposition, generateProxy,
    cancelProxyGeneration, fileSystemSupported, proxyFolderName, pickProxyFolder,
    showInExplorer, activeCompositionId, moveToFolder, createTextItem, getOrCreateTextFolder,
  } = useMediaStore();
```

**Solution:** Split into data selectors and action getters:

```typescript
// ✅ GOOD - Reactive data with selectors
const files = useMediaStore(state => state.files);
const compositions = useMediaStore(state => state.compositions);
const selectedIds = useMediaStore(state => state.selectedIds);

// ✅ GOOD - Actions from getState() (stable)
const store = useMediaStore.getState();
const { importFiles, createComposition, removeFile, ... } = store;
```

**Files to modify:**
- `src/components/panels/MediaPanel.tsx`

**Impact:** MEDIUM-HIGH - MediaPanel re-renders on any store change

---

### 1.3 Toolbar.tsx - Polling Anti-Pattern (27K)

**Problem:** Polls project state every 500ms with setInterval:

```typescript
// ❌ BAD - Line 56-74
useEffect(() => {
  const updateProjectState = () => {
    const data = projectFileService.getProjectData();
    // ...
  };
  updateProjectState();
  const interval = setInterval(updateProjectState, 500); // Unnecessary polling!
  return () => clearInterval(interval);
}, []);
```

**Solution:** Use event-based updates instead of polling:

```typescript
// ✅ GOOD - Subscribe to project changes
useEffect(() => {
  const unsubscribe = projectFileService.subscribe((data) => {
    if (data) {
      setProjectName(data.name);
      setIsProjectOpen(true);
    } else {
      setProjectName('No Project Open');
      setIsProjectOpen(false);
    }
  });
  return unsubscribe;
}, []);
```

**Files to modify:**
- `src/components/common/Toolbar.tsx`
- `src/services/projectFileService.ts` (add subscribe method)

**Impact:** MEDIUM - Reduces CPU usage, eliminates 2 renders/second

---

## Phase 2: HIGH - Component Splitting

### 2.1 PropertiesPanel.tsx - Split Into Separate Files (61K → ~10K each)

**Problem:** Single 61K file with 7+ tab components that could be lazy-loaded.

**Current structure:**
```
PropertiesPanel.tsx (61K)
├── KeyframeToggle
├── ScaleKeyframeToggle
├── PrecisionSlider
├── DraggableNumber
├── TransformTab
├── VolumeTab
├── EffectsTab (huge!)
├── MasksTab
├── MaskItem
├── TranscriptTab
└── AnalysisTab
```

**Solution:** Extract into separate files:

```
src/components/panels/properties/
├── index.tsx (main panel, lazy loads tabs)
├── KeyframeToggle.tsx
├── PrecisionSlider.tsx
├── DraggableNumber.tsx
├── TransformTab.tsx
├── VolumeTab.tsx
├── EffectsTab.tsx
├── MasksTab.tsx
├── TranscriptTab.tsx
└── AnalysisTab.tsx
```

With lazy loading:
```typescript
const TransformTab = lazy(() => import('./properties/TransformTab'));
const EffectsTab = lazy(() => import('./properties/EffectsTab'));
// etc.
```

**Impact:** HIGH - Faster initial load, better code splitting

---

### 2.2 useLayerSync.ts - Extract Sub-Hooks (47K)

**Problem:** Single 47K hook with multiple responsibilities.

**Current responsibilities:**
1. Video element seeking
2. Proxy frame management
3. Layer building for video tracks
4. Layer building for audio tracks
5. Engine layer synchronization

**Solution:** Extract into focused hooks:

```
src/components/timeline/hooks/
├── useLayerSync.ts (orchestrator, ~10K)
├── useVideoSeek.ts (video seeking logic)
├── useProxyFrames.ts (proxy frame caching)
├── useLayerBuilder.ts (layer construction)
└── useAudioSync.ts (audio track sync)
```

**Impact:** MEDIUM - Better maintainability, testability

---

## Phase 3: MEDIUM - Render Optimization

### 3.1 Inline Arrow Functions in PropertiesPanel

**Problem:** Many inline arrow functions in render:

```typescript
// ❌ BAD - Creates new function every render
<button onClick={() => updateMask(clipId, mask.id, { expanded: !mask.expanded })}>

// ❌ BAD - Object spread creates new reference
onChange={(v) => updateMask(clipId, mask.id, { position: { ...mask.position, x: v } })}
```

**Solution:** Use useCallback or stable handlers:

```typescript
// ✅ GOOD - Stable callback
const handleExpandToggle = useCallback(() => {
  updateMask(clipId, mask.id, { expanded: !mask.expanded });
}, [clipId, mask.id, mask.expanded, updateMask]);

// ✅ GOOD - Or pass primitive props
<MaskPositionInput
  clipId={clipId}
  maskId={mask.id}
  axis="x"
  value={mask.position.x}
/>
```

**Files to modify:**
- `src/components/panels/PropertiesPanel.tsx` (MaskItem, EffectItem)

**Impact:** MEDIUM - Reduces child re-renders

---

### 3.2 ExportPanel.tsx - getState() Usage in Helpers (58K)

**Problem:** Helper functions call `useTimelineStore.getState()` correctly, but they're defined at module level and could benefit from being memoized.

```typescript
// Line 38-43 - Good pattern but called frequently
async function seekAllClipsToTime(time: number): Promise<void> {
  const { clips, tracks, getSourceTimeForClip, getInterpolatedSpeed } = useTimelineStore.getState();
  // ...
}
```

**No change needed** - This is the correct pattern ✅

---

### 3.3 Preview.tsx - StatsOverlay Optimization

**Problem:** `StatsOverlay` component uses `useMemo` correctly but receives stats object that changes every frame.

```typescript
// Line 37-47 - Good useMemo usage
const bottleneck = useMemo(() => {
  const { timing } = stats;
  // ...
}, [stats.timing]);
```

**Solution:** Consider throttling stats updates to 10fps instead of 60fps for display.

**Impact:** LOW - Already reasonably optimized

---

## Phase 4: LOW - Code Quality

### 4.1 Create Shared UI Components

**Problem:** `PrecisionSlider` and `DraggableNumber` are defined in PropertiesPanel but used elsewhere.

**Solution:** Extract to shared components:

```
src/components/ui/
├── PrecisionSlider.tsx
├── DraggableNumber.tsx
├── KeyframeToggle.tsx
└── index.ts
```

**Impact:** LOW - Code organization improvement

---

### 4.2 Consolidate useState in MediaPanel

**Problem:** MediaPanel has 10+ useState calls for related state:

```typescript
const [renamingId, setRenamingId] = useState<string | null>(null);
const [renameValue, setRenameValue] = useState('');
const [contextMenu, setContextMenu] = useState<...>(null);
const [settingsDialog, setSettingsDialog] = useState<...>(null);
const [addDropdownOpen, setAddDropdownOpen] = useState(false);
const [dragOverFolderId, setDragOverFolderId] = useState<string | null>(null);
const [internalDragId, setInternalDragId] = useState<string | null>(null);
const [isExternalDragOver, setIsExternalDragOver] = useState(false);
const [columnOrder, setColumnOrder] = useState<ColumnId[]>(...);
const [draggingColumn, setDraggingColumn] = useState<ColumnId | null>(null);
const [dragOverColumn, setDragOverColumn] = useState<ColumnId | null>(null);
```

**Solution:** Group related state:

```typescript
// Rename state
const [renameState, setRenameState] = useState({ id: null, value: '' });

// Drag state (extract to useMediaPanelDrag hook)
const dragState = useMediaPanelDrag();

// Column state (extract to useColumnOrder hook)
const columnState = useColumnOrder();
```

**Impact:** LOW - Code organization improvement

---

## Implementation Priority

| Order | Task | Priority | Impact | Complexity | Status |
|-------|------|----------|--------|------------|--------|
| 1 | PropertiesPanel store subscriptions (1.1) | CRITICAL | HIGH | Medium | ✅ DONE |
| 2 | MediaPanel store subscriptions (1.2) | CRITICAL | MEDIUM-HIGH | Low | ✅ DONE |
| 3 | Toolbar polling fix (1.3) | HIGH | MEDIUM | Low | ✅ DONE |
| 4 | PropertiesPanel component split (2.1) | HIGH | HIGH | High | ✅ DONE |
| 5 | Inline arrow functions (3.1) | MEDIUM | MEDIUM | Medium | ⏭️ N/A |
| 6 | useLayerSync split (2.2) | MEDIUM | MEDIUM | High | ⏭️ SKIPPED |
| 7 | Shared UI components (4.1) | LOW | LOW | Low | ⏸️ DEFERRED |
| 8 | MediaPanel useState consolidation (4.2) | LOW | LOW | Low | ⏸️ DEFERRED |

---

## Quick Wins (Can Do Now)

1. **PropertiesPanel KeyframeToggle** - Change `useTimelineStore()` to `useTimelineStore.getState()` for actions
2. **MediaPanel actions** - Use `getState()` for non-reactive actions
3. **Toolbar polling** - Increase interval from 500ms to 2000ms (quick fix before event-based)

---

## Verification

After each change:
1. `npm run build` - No errors
2. Check React DevTools Profiler for reduced re-renders
3. Test playback smoothness
4. Test property editing responsiveness

---

## Files Summary

### Must Review (High Impact):
- `src/components/panels/PropertiesPanel.tsx` (61K) - CRITICAL
- `src/components/panels/MediaPanel.tsx` (48K) - HIGH
- `src/components/common/Toolbar.tsx` (27K) - HIGH
- `src/components/timeline/hooks/useLayerSync.ts` (47K) - MEDIUM

### Already Optimized:
- `src/components/timeline/Timeline.tsx` - ✅ Uses selectors correctly
- `src/components/timeline/TimelineClip.tsx` - ✅ Uses memo + selectors
- `src/components/timeline/TimelineTrack.tsx` - ✅ Uses memo
- `src/components/timeline/TimelineHeader.tsx` - ✅ Uses memo

### Low Priority:
- `src/components/export/ExportPanel.tsx` - Uses getState() correctly
- `src/components/preview/Preview.tsx` - Reasonably optimized
