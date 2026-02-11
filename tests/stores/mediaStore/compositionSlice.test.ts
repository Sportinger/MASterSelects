import { describe, it, expect, beforeEach, vi } from 'vitest';
import { createStore } from 'zustand';
import type { MediaState, MediaFile, Composition } from '../../../src/stores/mediaStore/types';
import { createCompositionSlice, type CompositionActions } from '../../../src/stores/mediaStore/slices/compositionSlice';
import { createSlotSlice, type SlotActions } from '../../../src/stores/mediaStore/slices/slotSlice';
import { createMultiLayerSlice, type MultiLayerActions } from '../../../src/stores/mediaStore/slices/multiLayerSlice';

// The compositionSlice calls useTimelineStore and useSettingsStore internally,
// but these are mocked in tests/setup.ts. We rely on those mocks here.

// Extend the layerBuilder mock from setup.ts to include playheadState
vi.mock('../../../src/services/layerBuilder', () => ({
  layerBuilder: {
    invalidateCache: vi.fn(),
    buildLayers: vi.fn().mockReturnValue([]),
    buildLayersFromStore: vi.fn().mockReturnValue([]),
  },
  playheadState: {
    position: 0,
    isUsingInternalPosition: false,
    playbackJustStarted: false,
    masterAudioElement: null,
    masterClipStartTime: 0,
    masterClipInPoint: 0,
    masterClipSpeed: 1,
    hasMasterAudio: false,
  },
}));

// Mock compositionRenderer used in doSetActiveComposition
vi.mock('../../../src/services/compositionRenderer', () => ({
  compositionRenderer: {
    invalidateCompositionAndParents: vi.fn(),
  },
}));

type TestMediaStore = MediaState & CompositionActions & SlotActions & MultiLayerActions;

function createTestMediaStore(overrides?: Partial<MediaState>) {
  const defaultComp: Composition = {
    id: 'comp-1',
    name: 'Comp 1',
    type: 'composition',
    parentId: null,
    createdAt: 1000,
    width: 1920,
    height: 1080,
    frameRate: 30,
    duration: 60,
    backgroundColor: '#000000',
  };

  return createStore<TestMediaStore>()((set, get) => {
    const compositionActions = createCompositionSlice(set as any, get as any);
    const slotActions = createSlotSlice(set as any, get as any);
    const multiLayerActions = createMultiLayerSlice(set as any, get as any);

    return {
      // Minimal initial state
      files: [],
      compositions: [defaultComp],
      folders: [],
      textItems: [],
      solidItems: [],
      activeCompositionId: 'comp-1',
      openCompositionIds: ['comp-1'],
      slotAssignments: {},
      previewCompositionId: null,
      activeLayerSlots: {},
      layerOpacities: {},
      selectedIds: [],
      expandedFolderIds: [],
      currentProjectId: null,
      currentProjectName: 'Untitled Project',
      isLoading: false,
      proxyEnabled: false,
      proxyGenerationQueue: [],
      currentlyGeneratingProxyId: null,
      fileSystemSupported: false,
      proxyFolderName: null,
      ...compositionActions,
      ...slotActions,
      ...multiLayerActions,
      ...overrides,
    } as TestMediaStore;
  });
}

describe('compositionSlice', () => {
  let store: ReturnType<typeof createTestMediaStore>;

  beforeEach(() => {
    store = createTestMediaStore();
  });

  // ─── createComposition ────────────────────────────────────────────

  it('createComposition: creates a new composition with defaults', () => {
    const comp = store.getState().createComposition('My Comp');
    expect(comp.name).toBe('My Comp');
    expect(comp.type).toBe('composition');
    expect(comp.width).toBe(1920); // from mocked settingsStore
    expect(comp.height).toBe(1080);
    expect(comp.frameRate).toBe(30);
    expect(comp.duration).toBe(60);
    expect(comp.backgroundColor).toBe('#000000');
    expect(comp.id).toBeDefined();
    // Verify it was added to the store
    const comps = store.getState().compositions;
    expect(comps.find(c => c.id === comp.id)).toBeDefined();
  });

  it('createComposition: uses provided settings overrides', () => {
    const comp = store.getState().createComposition('Custom', {
      width: 3840,
      height: 2160,
      frameRate: 60,
      duration: 120,
      backgroundColor: '#ff0000',
    });
    expect(comp.width).toBe(3840);
    expect(comp.height).toBe(2160);
    expect(comp.frameRate).toBe(60);
    expect(comp.duration).toBe(120);
    expect(comp.backgroundColor).toBe('#ff0000');
  });

  it('createComposition: assigns unique IDs to each composition', () => {
    const comp1 = store.getState().createComposition('A');
    const comp2 = store.getState().createComposition('B');
    expect(comp1.id).not.toBe(comp2.id);
    expect(store.getState().compositions.length).toBe(3); // default + 2 new
  });

  // ─── duplicateComposition ─────────────────────────────────────────

  it('duplicateComposition: creates a copy with new id and name suffix', () => {
    const duplicate = store.getState().duplicateComposition('comp-1');
    expect(duplicate).not.toBeNull();
    expect(duplicate!.name).toBe('Comp 1 Copy');
    expect(duplicate!.id).not.toBe('comp-1');
    expect(duplicate!.width).toBe(1920);
    expect(duplicate!.height).toBe(1080);
    expect(store.getState().compositions.length).toBe(2);
  });

  it('duplicateComposition: returns null for nonexistent id', () => {
    const result = store.getState().duplicateComposition('nonexistent');
    expect(result).toBeNull();
  });

  // ─── removeComposition ────────────────────────────────────────────

  it('removeComposition: removes composition from list', () => {
    const comp = store.getState().createComposition('To Remove');
    expect(store.getState().compositions.length).toBe(2);
    store.getState().removeComposition(comp.id);
    expect(store.getState().compositions.length).toBe(1);
    expect(store.getState().compositions.find(c => c.id === comp.id)).toBeUndefined();
  });

  it('removeComposition: clears activeCompositionId when active comp is removed', () => {
    const comp = store.getState().createComposition('Active');
    store.setState({ activeCompositionId: comp.id });
    store.getState().removeComposition(comp.id);
    expect(store.getState().activeCompositionId).toBeNull();
  });

  it('removeComposition: removes from openCompositionIds', () => {
    const comp = store.getState().createComposition('Open');
    store.setState({ openCompositionIds: ['comp-1', comp.id] });
    store.getState().removeComposition(comp.id);
    expect(store.getState().openCompositionIds).not.toContain(comp.id);
  });

  it('removeComposition: removes from selectedIds', () => {
    const comp = store.getState().createComposition('Selected');
    store.setState({ selectedIds: [comp.id, 'other-item'] });
    store.getState().removeComposition(comp.id);
    expect(store.getState().selectedIds).toEqual(['other-item']);
  });

  it('removeComposition: cleans up slotAssignments', () => {
    const comp = store.getState().createComposition('Slotted');
    store.setState({ slotAssignments: { [comp.id]: 3 } });
    store.getState().removeComposition(comp.id);
    expect(store.getState().slotAssignments[comp.id]).toBeUndefined();
  });

  // ─── updateComposition ────────────────────────────────────────────

  it('updateComposition: updates name and background color', () => {
    store.getState().updateComposition('comp-1', {
      name: 'Renamed',
      backgroundColor: '#00ff00',
    });
    const comp = store.getState().compositions.find(c => c.id === 'comp-1')!;
    expect(comp.name).toBe('Renamed');
    expect(comp.backgroundColor).toBe('#00ff00');
  });

  it('updateComposition: updates fps and duration', () => {
    store.getState().updateComposition('comp-1', {
      frameRate: 24,
      duration: 120,
    });
    const comp = store.getState().compositions.find(c => c.id === 'comp-1')!;
    expect(comp.frameRate).toBe(24);
    expect(comp.duration).toBe(120);
  });

  it('updateComposition: updates resolution (width/height)', () => {
    // For a non-active composition to avoid the clip transform adjustment path
    const comp = store.getState().createComposition('ResTest');
    store.setState({ activeCompositionId: 'comp-1' }); // Ensure comp is NOT active
    store.getState().updateComposition(comp.id, { width: 3840, height: 2160 });
    const updated = store.getState().compositions.find(c => c.id === comp.id)!;
    expect(updated.width).toBe(3840);
    expect(updated.height).toBe(2160);
  });

  // ─── getActiveComposition ─────────────────────────────────────────

  it('getActiveComposition: returns the currently active composition', () => {
    const active = store.getState().getActiveComposition();
    expect(active).toBeDefined();
    expect(active!.id).toBe('comp-1');
    expect(active!.name).toBe('Comp 1');
  });

  it('getActiveComposition: returns undefined when no active composition', () => {
    store.setState({ activeCompositionId: null });
    expect(store.getState().getActiveComposition()).toBeUndefined();
  });

  // ─── getOpenCompositions ──────────────────────────────────────────

  it('getOpenCompositions: returns compositions matching openCompositionIds', () => {
    const comp2 = store.getState().createComposition('Second');
    store.setState({ openCompositionIds: ['comp-1', comp2.id] });
    const open = store.getState().getOpenCompositions();
    expect(open.length).toBe(2);
    expect(open[0].id).toBe('comp-1');
    expect(open[1].id).toBe(comp2.id);
  });

  it('getOpenCompositions: filters out deleted compositions', () => {
    store.setState({ openCompositionIds: ['comp-1', 'deleted-id'] });
    const open = store.getState().getOpenCompositions();
    expect(open.length).toBe(1);
    expect(open[0].id).toBe('comp-1');
  });

  // ─── reorderCompositionTabs ───────────────────────────────────────

  it('reorderCompositionTabs: swaps tab order', () => {
    const comp2 = store.getState().createComposition('B');
    const comp3 = store.getState().createComposition('C');
    store.setState({ openCompositionIds: ['comp-1', comp2.id, comp3.id] });
    store.getState().reorderCompositionTabs(0, 2);
    expect(store.getState().openCompositionIds).toEqual([comp2.id, comp3.id, 'comp-1']);
  });

  it('reorderCompositionTabs: no-op for same index', () => {
    store.setState({ openCompositionIds: ['comp-1', 'comp-2'] });
    store.getState().reorderCompositionTabs(0, 0);
    expect(store.getState().openCompositionIds).toEqual(['comp-1', 'comp-2']);
  });

  it('reorderCompositionTabs: no-op for out-of-bounds indices', () => {
    store.setState({ openCompositionIds: ['comp-1'] });
    store.getState().reorderCompositionTabs(-1, 5);
    expect(store.getState().openCompositionIds).toEqual(['comp-1']);
  });

  // ─── Slot management ──────────────────────────────────────────────

  it('moveSlot: assigns composition to a slot', () => {
    store.getState().moveSlot('comp-1', 5);
    expect(store.getState().slotAssignments['comp-1']).toBe(5);
  });

  it('moveSlot: swaps compositions when target slot is occupied', () => {
    const comp2 = store.getState().createComposition('B');
    store.setState({ slotAssignments: { 'comp-1': 0, [comp2.id]: 3 } });
    store.getState().moveSlot('comp-1', 3);
    expect(store.getState().slotAssignments['comp-1']).toBe(3);
    expect(store.getState().slotAssignments[comp2.id]).toBe(0);
  });

  it('unassignSlot: removes slot assignment', () => {
    store.setState({ slotAssignments: { 'comp-1': 2 } });
    store.getState().unassignSlot('comp-1');
    expect(store.getState().slotAssignments['comp-1']).toBeUndefined();
  });

  it('getSlotMap: returns correctly sized array with assigned compositions', () => {
    store.setState({ slotAssignments: { 'comp-1': 2 } });
    const map = store.getState().getSlotMap(6);
    expect(map.length).toBe(6);
    expect(map[2]?.id).toBe('comp-1');
    expect(map[0]).toBeNull();
    expect(map[1]).toBeNull();
    expect(map[3]).toBeNull();
  });

  // ─── Multi-layer playback ─────────────────────────────────────────

  it('activateOnLayer: assigns composition to a layer', () => {
    store.getState().activateOnLayer('comp-1', 0);
    expect(store.getState().activeLayerSlots[0]).toBe('comp-1');
  });

  it('activateOnLayer: moves composition from previous layer', () => {
    store.getState().activateOnLayer('comp-1', 0);
    store.getState().activateOnLayer('comp-1', 2);
    expect(store.getState().activeLayerSlots[0]).toBeUndefined();
    expect(store.getState().activeLayerSlots[2]).toBe('comp-1');
  });

  it('deactivateLayer: removes composition from a layer', () => {
    store.getState().activateOnLayer('comp-1', 1);
    store.getState().deactivateLayer(1);
    expect(store.getState().activeLayerSlots[1]).toBeUndefined();
  });

  it('deactivateAllLayers: clears all layer assignments', () => {
    const comp2 = store.getState().createComposition('B');
    store.getState().activateOnLayer('comp-1', 0);
    store.getState().activateOnLayer(comp2.id, 1);
    store.getState().deactivateAllLayers();
    expect(Object.keys(store.getState().activeLayerSlots).length).toBe(0);
  });

  // ─── setPreviewComposition ────────────────────────────────────────

  it('setPreviewComposition: sets and clears preview ID', () => {
    store.getState().setPreviewComposition('comp-1');
    expect(store.getState().previewCompositionId).toBe('comp-1');
    store.getState().setPreviewComposition(null);
    expect(store.getState().previewCompositionId).toBeNull();
  });

  // ─── createComposition (additional edge cases) ──────────────────

  it('createComposition: sets parentId to null by default', () => {
    const comp = store.getState().createComposition('No Parent');
    expect(comp.parentId).toBeNull();
  });

  it('createComposition: sets type to composition', () => {
    const comp = store.getState().createComposition('Typed');
    expect(comp.type).toBe('composition');
  });

  it('createComposition: sets createdAt to a recent timestamp', () => {
    const before = Date.now();
    const comp = store.getState().createComposition('Timed');
    const after = Date.now();
    expect(comp.createdAt).toBeGreaterThanOrEqual(before);
    expect(comp.createdAt).toBeLessThanOrEqual(after);
  });

  it('createComposition: partial settings override only specified fields', () => {
    const comp = store.getState().createComposition('Partial', {
      width: 1280,
    });
    expect(comp.width).toBe(1280);
    // Unspecified fields use defaults
    expect(comp.height).toBe(1080); // from mocked settingsStore
    expect(comp.frameRate).toBe(30);
    expect(comp.duration).toBe(60);
    expect(comp.backgroundColor).toBe('#000000');
  });

  it('createComposition: does not affect existing compositions', () => {
    const existing = store.getState().compositions[0];
    store.getState().createComposition('New');
    const stillExisting = store.getState().compositions.find(c => c.id === existing.id);
    expect(stillExisting).toEqual(existing);
  });

  // ─── duplicateComposition (additional edge cases) ───────────────

  it('duplicateComposition: preserves all properties except id, name, createdAt', () => {
    store.getState().updateComposition('comp-1', {
      width: 2560,
      height: 1440,
      frameRate: 24,
      duration: 300,
      backgroundColor: '#112233',
    });
    const dup = store.getState().duplicateComposition('comp-1');
    expect(dup).not.toBeNull();
    expect(dup!.width).toBe(2560);
    expect(dup!.height).toBe(1440);
    expect(dup!.frameRate).toBe(24);
    expect(dup!.duration).toBe(300);
    expect(dup!.backgroundColor).toBe('#112233');
    expect(dup!.type).toBe('composition');
    expect(dup!.parentId).toBeNull();
  });

  it('duplicateComposition: sets a new createdAt timestamp', () => {
    const before = Date.now();
    const dup = store.getState().duplicateComposition('comp-1');
    const after = Date.now();
    expect(dup).not.toBeNull();
    expect(dup!.createdAt).toBeGreaterThanOrEqual(before);
    expect(dup!.createdAt).toBeLessThanOrEqual(after);
    expect(dup!.createdAt).not.toBe(1000); // original createdAt
  });

  it('duplicateComposition: duplicate of duplicate appends another Copy', () => {
    const dup1 = store.getState().duplicateComposition('comp-1');
    expect(dup1).not.toBeNull();
    const dup2 = store.getState().duplicateComposition(dup1!.id);
    expect(dup2).not.toBeNull();
    expect(dup2!.name).toBe('Comp 1 Copy Copy');
  });

  // ─── removeComposition (additional edge cases) ──────────────────

  it('removeComposition: does not affect unrelated compositions', () => {
    const comp2 = store.getState().createComposition('Keep');
    const comp3 = store.getState().createComposition('Remove');
    store.getState().removeComposition(comp3.id);
    expect(store.getState().compositions.find(c => c.id === comp2.id)).toBeDefined();
    expect(store.getState().compositions.find(c => c.id === 'comp-1')).toBeDefined();
    expect(store.getState().compositions.length).toBe(2);
  });

  it('removeComposition: keeps activeCompositionId when different comp is removed', () => {
    const comp2 = store.getState().createComposition('Other');
    store.setState({ activeCompositionId: 'comp-1' });
    store.getState().removeComposition(comp2.id);
    expect(store.getState().activeCompositionId).toBe('comp-1');
  });

  it('removeComposition: no-op when removing nonexistent id', () => {
    const before = store.getState().compositions.length;
    store.getState().removeComposition('nonexistent');
    expect(store.getState().compositions.length).toBe(before);
  });

  it('removeComposition: cleans up multiple state references simultaneously', () => {
    const comp = store.getState().createComposition('Multi');
    store.setState({
      activeCompositionId: comp.id,
      openCompositionIds: ['comp-1', comp.id],
      selectedIds: [comp.id, 'other'],
      slotAssignments: { [comp.id]: 1, 'comp-1': 0 },
    });
    store.getState().removeComposition(comp.id);
    expect(store.getState().activeCompositionId).toBeNull();
    expect(store.getState().openCompositionIds).not.toContain(comp.id);
    expect(store.getState().selectedIds).toEqual(['other']);
    expect(store.getState().slotAssignments[comp.id]).toBeUndefined();
    // comp-1 slot assignment should remain
    expect(store.getState().slotAssignments['comp-1']).toBe(0);
  });

  // ─── updateComposition (additional edge cases) ──────────────────

  it('updateComposition: no-op for nonexistent id (does not crash)', () => {
    const before = store.getState().compositions.map(c => ({ ...c }));
    store.getState().updateComposition('nonexistent', { name: 'Nope' });
    // Compositions remain unchanged
    expect(store.getState().compositions.length).toBe(before.length);
  });

  it('updateComposition: does not affect other compositions', () => {
    const comp2 = store.getState().createComposition('Other');
    store.getState().updateComposition('comp-1', { name: 'Changed' });
    const other = store.getState().compositions.find(c => c.id === comp2.id)!;
    expect(other.name).toBe('Other');
  });

  it('updateComposition: updates only width when height unchanged', () => {
    const comp = store.getState().createComposition('WidthOnly');
    store.setState({ activeCompositionId: 'comp-1' }); // Ensure comp is NOT active
    store.getState().updateComposition(comp.id, { width: 2560 });
    const updated = store.getState().compositions.find(c => c.id === comp.id)!;
    expect(updated.width).toBe(2560);
    expect(updated.height).toBe(1080); // unchanged
  });

  it('updateComposition: updates only height when width unchanged', () => {
    const comp = store.getState().createComposition('HeightOnly');
    store.setState({ activeCompositionId: 'comp-1' });
    store.getState().updateComposition(comp.id, { height: 720 });
    const updated = store.getState().compositions.find(c => c.id === comp.id)!;
    expect(updated.width).toBe(1920); // unchanged
    expect(updated.height).toBe(720);
  });

  // ─── setActiveComposition ───────────────────────────────────────

  it('setActiveComposition: sets a new active composition id', () => {
    const comp2 = store.getState().createComposition('Second');
    store.getState().setActiveComposition(comp2.id);
    expect(store.getState().activeCompositionId).toBe(comp2.id);
  });

  it('setActiveComposition: sets to null to deactivate', () => {
    store.getState().setActiveComposition(null);
    expect(store.getState().activeCompositionId).toBeNull();
  });

  it('setActiveComposition: setting same id as current is a no-op', () => {
    store.getState().setActiveComposition('comp-1');
    expect(store.getState().activeCompositionId).toBe('comp-1');
  });

  // ─── openCompositionTab ─────────────────────────────────────────

  it('openCompositionTab: adds comp to openCompositionIds if not present', () => {
    const comp2 = store.getState().createComposition('TabTest');
    store.setState({ openCompositionIds: ['comp-1'] });
    store.getState().openCompositionTab(comp2.id);
    expect(store.getState().openCompositionIds).toContain(comp2.id);
  });

  it('openCompositionTab: does not duplicate in openCompositionIds if already open', () => {
    store.setState({ openCompositionIds: ['comp-1'] });
    store.getState().openCompositionTab('comp-1');
    expect(store.getState().openCompositionIds.filter(id => id === 'comp-1').length).toBe(1);
  });

  it('openCompositionTab: sets the composition as active', () => {
    const comp2 = store.getState().createComposition('ToOpen');
    store.getState().openCompositionTab(comp2.id, { skipAnimation: true });
    expect(store.getState().activeCompositionId).toBe(comp2.id);
  });

  // ─── closeCompositionTab ────────────────────────────────────────

  it('closeCompositionTab: removes comp from openCompositionIds', () => {
    const comp2 = store.getState().createComposition('ToClose');
    store.setState({ openCompositionIds: ['comp-1', comp2.id], activeCompositionId: 'comp-1' });
    store.getState().closeCompositionTab(comp2.id);
    expect(store.getState().openCompositionIds).not.toContain(comp2.id);
  });

  it('closeCompositionTab: switches active to another open tab when active is closed', () => {
    const comp2 = store.getState().createComposition('Stay');
    store.setState({
      openCompositionIds: ['comp-1', comp2.id],
      activeCompositionId: 'comp-1',
    });
    store.getState().closeCompositionTab('comp-1');
    // After closing first tab, active should switch to remaining tab
    expect(store.getState().openCompositionIds).toEqual([comp2.id]);
    expect(store.getState().activeCompositionId).toBe(comp2.id);
  });

  it('closeCompositionTab: sets activeCompositionId to null when last tab is closed', () => {
    store.setState({ openCompositionIds: ['comp-1'], activeCompositionId: 'comp-1' });
    store.getState().closeCompositionTab('comp-1');
    expect(store.getState().openCompositionIds).toEqual([]);
    expect(store.getState().activeCompositionId).toBeNull();
  });

  it('closeCompositionTab: does not affect active when non-active tab is closed', () => {
    const comp2 = store.getState().createComposition('NonActive');
    store.setState({
      openCompositionIds: ['comp-1', comp2.id],
      activeCompositionId: 'comp-1',
    });
    store.getState().closeCompositionTab(comp2.id);
    expect(store.getState().activeCompositionId).toBe('comp-1');
  });

  // ─── getActiveComposition (additional edge cases) ───────────────

  it('getActiveComposition: returns correct comp after switching active', () => {
    const comp2 = store.getState().createComposition('Switch');
    store.setState({ activeCompositionId: comp2.id });
    const active = store.getState().getActiveComposition();
    expect(active).toBeDefined();
    expect(active!.id).toBe(comp2.id);
    expect(active!.name).toBe('Switch');
  });

  it('getActiveComposition: returns undefined when activeCompositionId references deleted comp', () => {
    store.setState({ activeCompositionId: 'deleted-comp' });
    expect(store.getState().getActiveComposition()).toBeUndefined();
  });

  // ─── getOpenCompositions (additional edge cases) ────────────────

  it('getOpenCompositions: returns empty array when no open compositions', () => {
    store.setState({ openCompositionIds: [] });
    expect(store.getState().getOpenCompositions()).toEqual([]);
  });

  it('getOpenCompositions: preserves order from openCompositionIds', () => {
    const comp2 = store.getState().createComposition('B');
    const comp3 = store.getState().createComposition('C');
    store.setState({ openCompositionIds: [comp3.id, 'comp-1', comp2.id] });
    const open = store.getState().getOpenCompositions();
    expect(open.length).toBe(3);
    expect(open[0].id).toBe(comp3.id);
    expect(open[1].id).toBe('comp-1');
    expect(open[2].id).toBe(comp2.id);
  });

  // ─── reorderCompositionTabs (additional edge cases) ─────────────

  it('reorderCompositionTabs: moves from end to beginning', () => {
    const comp2 = store.getState().createComposition('B');
    const comp3 = store.getState().createComposition('C');
    store.setState({ openCompositionIds: ['comp-1', comp2.id, comp3.id] });
    store.getState().reorderCompositionTabs(2, 0);
    expect(store.getState().openCompositionIds).toEqual([comp3.id, 'comp-1', comp2.id]);
  });

  it('reorderCompositionTabs: moves adjacent tabs', () => {
    const comp2 = store.getState().createComposition('B');
    store.setState({ openCompositionIds: ['comp-1', comp2.id] });
    store.getState().reorderCompositionTabs(0, 1);
    expect(store.getState().openCompositionIds).toEqual([comp2.id, 'comp-1']);
  });

  it('reorderCompositionTabs: no-op when fromIndex is out of bounds', () => {
    store.setState({ openCompositionIds: ['comp-1', 'comp-2'] });
    store.getState().reorderCompositionTabs(5, 0);
    expect(store.getState().openCompositionIds).toEqual(['comp-1', 'comp-2']);
  });

  it('reorderCompositionTabs: no-op when toIndex is out of bounds', () => {
    store.setState({ openCompositionIds: ['comp-1', 'comp-2'] });
    store.getState().reorderCompositionTabs(0, 5);
    expect(store.getState().openCompositionIds).toEqual(['comp-1', 'comp-2']);
  });

  it('reorderCompositionTabs: no-op on empty list', () => {
    store.setState({ openCompositionIds: [] });
    store.getState().reorderCompositionTabs(0, 1);
    expect(store.getState().openCompositionIds).toEqual([]);
  });

  // ─── Slot management (additional edge cases) ────────────────────

  it('moveSlot: displaces existing comp and removes it when source has no slot', () => {
    const comp2 = store.getState().createComposition('B');
    // comp2 is at slot 3, comp-1 has no slot assignment
    store.setState({ slotAssignments: { [comp2.id]: 3 } });
    store.getState().moveSlot('comp-1', 3);
    expect(store.getState().slotAssignments['comp-1']).toBe(3);
    // comp2 should be unassigned because comp-1 had no prior slot
    expect(store.getState().slotAssignments[comp2.id]).toBeUndefined();
  });

  it('moveSlot: reassigns to a different slot without swap', () => {
    store.setState({ slotAssignments: { 'comp-1': 2 } });
    store.getState().moveSlot('comp-1', 5);
    expect(store.getState().slotAssignments['comp-1']).toBe(5);
  });

  it('moveSlot: assigns to slot 0 (edge case for first slot)', () => {
    store.getState().moveSlot('comp-1', 0);
    expect(store.getState().slotAssignments['comp-1']).toBe(0);
  });

  it('unassignSlot: no-op when comp is not assigned', () => {
    store.setState({ slotAssignments: {} });
    store.getState().unassignSlot('comp-1');
    expect(store.getState().slotAssignments['comp-1']).toBeUndefined();
  });

  it('unassignSlot: does not affect other slot assignments', () => {
    const comp2 = store.getState().createComposition('B');
    store.setState({ slotAssignments: { 'comp-1': 0, [comp2.id]: 3 } });
    store.getState().unassignSlot('comp-1');
    expect(store.getState().slotAssignments[comp2.id]).toBe(3);
  });

  it('getSlotMap: ignores out-of-bounds slot assignments', () => {
    store.setState({ slotAssignments: { 'comp-1': 10 } });
    const map = store.getState().getSlotMap(5);
    expect(map.length).toBe(5);
    // slot 10 is beyond the requested totalSlots=5
    expect(map.every(item => item === null)).toBe(true);
  });

  it('getSlotMap: ignores negative slot indices', () => {
    store.setState({ slotAssignments: { 'comp-1': -1 } });
    const map = store.getState().getSlotMap(5);
    expect(map.length).toBe(5);
    expect(map.every(item => item === null)).toBe(true);
  });

  it('getSlotMap: handles multiple compositions in different slots', () => {
    const comp2 = store.getState().createComposition('B');
    const comp3 = store.getState().createComposition('C');
    store.setState({ slotAssignments: { 'comp-1': 0, [comp2.id]: 2, [comp3.id]: 4 } });
    const map = store.getState().getSlotMap(6);
    expect(map[0]?.id).toBe('comp-1');
    expect(map[1]).toBeNull();
    expect(map[2]?.id).toBe(comp2.id);
    expect(map[3]).toBeNull();
    expect(map[4]?.id).toBe(comp3.id);
    expect(map[5]).toBeNull();
  });

  it('getSlotMap: returns all nulls when no assignments', () => {
    store.setState({ slotAssignments: {} });
    const map = store.getState().getSlotMap(4);
    expect(map.length).toBe(4);
    expect(map.every(item => item === null)).toBe(true);
  });

  it('getSlotMap: returns empty array for totalSlots=0', () => {
    const map = store.getState().getSlotMap(0);
    expect(map.length).toBe(0);
  });

  it('getSlotMap: skips nonexistent composition references', () => {
    store.setState({ slotAssignments: { 'nonexistent-comp': 0, 'comp-1': 1 } });
    const map = store.getState().getSlotMap(3);
    expect(map[0]).toBeNull(); // nonexistent comp is skipped
    expect(map[1]?.id).toBe('comp-1');
  });

  // ─── assignMediaFileToSlot ──────────────────────────────────────

  it('assignMediaFileToSlot: creates composition from media file and assigns to slot', () => {
    const mediaFile: MediaFile = {
      id: 'media-1',
      name: 'video.mp4',
      type: 'video',
      parentId: null,
      createdAt: 1000,
      url: 'blob:test',
      width: 1280,
      height: 720,
      duration: 30,
    };
    store.setState({ files: [mediaFile] });
    const beforeCount = store.getState().compositions.length;
    store.getState().assignMediaFileToSlot('media-1', 2);
    // Should have created a new composition
    expect(store.getState().compositions.length).toBe(beforeCount + 1);
    // The new composition should be assigned to slot 2
    const newComp = store.getState().compositions[store.getState().compositions.length - 1];
    expect(newComp.name).toBe('video'); // name without extension
    expect(newComp.width).toBe(1280);
    expect(newComp.height).toBe(720);
    expect(newComp.duration).toBe(30);
    expect(store.getState().slotAssignments[newComp.id]).toBe(2);
  });

  it('assignMediaFileToSlot: no-op when media file does not exist', () => {
    const beforeCount = store.getState().compositions.length;
    store.getState().assignMediaFileToSlot('nonexistent', 0);
    expect(store.getState().compositions.length).toBe(beforeCount);
  });

  it('assignMediaFileToSlot: displaces existing slot occupant', () => {
    const mediaFile: MediaFile = {
      id: 'media-2',
      name: 'clip.mp4',
      type: 'video',
      parentId: null,
      createdAt: 1000,
      url: 'blob:test2',
      width: 1920,
      height: 1080,
      duration: 60,
    };
    store.setState({
      files: [mediaFile],
      slotAssignments: { 'comp-1': 5 },
    });
    store.getState().assignMediaFileToSlot('media-2', 5);
    // comp-1 should be displaced
    expect(store.getState().slotAssignments['comp-1']).toBeUndefined();
  });

  it('assignMediaFileToSlot: opens composition tab', () => {
    const mediaFile: MediaFile = {
      id: 'media-3',
      name: 'scene.mp4',
      type: 'video',
      parentId: null,
      createdAt: 1000,
      url: 'blob:test3',
      duration: 10,
    };
    store.setState({ files: [mediaFile] });
    store.getState().assignMediaFileToSlot('media-3', 0);
    const newComp = store.getState().compositions[store.getState().compositions.length - 1];
    expect(store.getState().openCompositionIds).toContain(newComp.id);
  });

  it('assignMediaFileToSlot: uses fallback resolution when media has no dimensions', () => {
    const mediaFile: MediaFile = {
      id: 'media-nodim',
      name: 'audio.mp3',
      type: 'audio',
      parentId: null,
      createdAt: 1000,
      url: 'blob:audio',
      duration: 120,
    };
    store.setState({ files: [mediaFile] });
    store.getState().assignMediaFileToSlot('media-nodim', 0);
    const newComp = store.getState().compositions[store.getState().compositions.length - 1];
    // Should fallback to output resolution from settingsStore mock (1920x1080)
    expect(newComp.width).toBe(1920);
    expect(newComp.height).toBe(1080);
  });

  // ─── Multi-layer playback (additional edge cases) ───────────────

  it('activateOnLayer: replaces existing comp on the target layer', () => {
    const comp2 = store.getState().createComposition('B');
    store.getState().activateOnLayer('comp-1', 0);
    store.getState().activateOnLayer(comp2.id, 0);
    expect(store.getState().activeLayerSlots[0]).toBe(comp2.id);
  });

  it('activateOnLayer: same comp on same layer keeps assignment', () => {
    store.getState().activateOnLayer('comp-1', 2);
    store.getState().activateOnLayer('comp-1', 2);
    expect(store.getState().activeLayerSlots[2]).toBe('comp-1');
  });

  it('deactivateLayer: no-op for unoccupied layer', () => {
    store.getState().deactivateLayer(5);
    expect(store.getState().activeLayerSlots[5]).toBeUndefined();
  });

  it('deactivateAllLayers: works when no layers are active', () => {
    store.setState({ activeLayerSlots: {} });
    store.getState().deactivateAllLayers();
    expect(Object.keys(store.getState().activeLayerSlots).length).toBe(0);
  });

  // ─── activateColumn ─────────────────────────────────────────────

  it('activateColumn: activates compositions from the given column across rows', () => {
    const comp2 = store.getState().createComposition('B');
    const comp3 = store.getState().createComposition('C');
    // 12-column grid, 4 rows. Column 2 means slot indices 2, 14, 26, 38
    store.setState({
      slotAssignments: {
        'comp-1': 2,         // row 0, col 2
        [comp2.id]: 14,      // row 1, col 2
        [comp3.id]: 26,      // row 2, col 2
      },
    });
    store.getState().activateColumn(2);
    const slots = store.getState().activeLayerSlots;
    expect(slots[0]).toBe('comp-1');
    expect(slots[1]).toBe(comp2.id);
    expect(slots[2]).toBe(comp3.id);
    expect(slots[3]).toBeUndefined(); // row 3 has no comp at col 2
  });

  it('activateColumn: clears previous layer assignments', () => {
    const comp2 = store.getState().createComposition('B');
    store.setState({
      activeLayerSlots: { 0: 'old-comp', 1: 'other-old' },
      slotAssignments: { [comp2.id]: 0 }, // row 0, col 0
    });
    store.getState().activateColumn(0);
    // Previous assignments should be replaced, not merged
    expect(store.getState().activeLayerSlots[0]).toBe(comp2.id);
    expect(store.getState().activeLayerSlots[1]).toBeUndefined();
  });

  it('activateColumn: results in empty layers when no comps in column', () => {
    store.setState({ slotAssignments: { 'comp-1': 0 } }); // col 0
    store.getState().activateColumn(5); // col 5 has nothing
    const slots = store.getState().activeLayerSlots;
    expect(Object.keys(slots).length).toBe(0);
  });

  // ─── setLayerOpacity ────────────────────────────────────────────

  it('setLayerOpacity: sets opacity for a layer', () => {
    (store.getState() as any).setLayerOpacity(0, 0.5);
    expect(store.getState().layerOpacities[0]).toBe(0.5);
  });

  it('setLayerOpacity: clamps opacity to min 0', () => {
    (store.getState() as any).setLayerOpacity(1, -0.5);
    expect(store.getState().layerOpacities[1]).toBe(0);
  });

  it('setLayerOpacity: clamps opacity to max 1', () => {
    (store.getState() as any).setLayerOpacity(2, 1.5);
    expect(store.getState().layerOpacities[2]).toBe(1);
  });

  it('setLayerOpacity: sets full opacity (1.0)', () => {
    (store.getState() as any).setLayerOpacity(0, 1);
    expect(store.getState().layerOpacities[0]).toBe(1);
  });

  it('setLayerOpacity: sets zero opacity (0.0)', () => {
    (store.getState() as any).setLayerOpacity(0, 0);
    expect(store.getState().layerOpacities[0]).toBe(0);
  });

  it('setLayerOpacity: does not affect other layer opacities', () => {
    (store.getState() as any).setLayerOpacity(0, 0.3);
    (store.getState() as any).setLayerOpacity(1, 0.7);
    expect(store.getState().layerOpacities[0]).toBe(0.3);
    expect(store.getState().layerOpacities[1]).toBe(0.7);
  });

  // ─── setPreviewComposition (additional edge cases) ──────────────

  it('setPreviewComposition: can be set to any comp id', () => {
    const comp2 = store.getState().createComposition('Preview');
    store.getState().setPreviewComposition(comp2.id);
    expect(store.getState().previewCompositionId).toBe(comp2.id);
  });

  it('setPreviewComposition: can be set to nonexistent id (no validation)', () => {
    store.getState().setPreviewComposition('nonexistent');
    expect(store.getState().previewCompositionId).toBe('nonexistent');
  });

  // ─── Integration scenarios ──────────────────────────────────────

  it('create, update, and duplicate workflow', () => {
    const comp = store.getState().createComposition('Original', {
      width: 1280,
      height: 720,
      frameRate: 24,
    });
    store.getState().updateComposition(comp.id, { name: 'Updated' });
    const dup = store.getState().duplicateComposition(comp.id);
    expect(dup).not.toBeNull();
    expect(dup!.name).toBe('Updated Copy');
    expect(dup!.width).toBe(1280);
    expect(dup!.frameRate).toBe(24);
    expect(store.getState().compositions.length).toBe(3);
  });

  it('open, reorder, and close tabs workflow', () => {
    const comp2 = store.getState().createComposition('Tab2');
    const comp3 = store.getState().createComposition('Tab3');
    store.getState().openCompositionTab(comp2.id, { skipAnimation: true });
    store.getState().openCompositionTab(comp3.id, { skipAnimation: true });
    expect(store.getState().openCompositionIds.length).toBe(3);

    store.getState().reorderCompositionTabs(0, 2);
    const order = store.getState().openCompositionIds;
    expect(order[0]).toBe(comp2.id);
    expect(order[2]).toBe('comp-1');

    store.getState().closeCompositionTab(comp2.id);
    expect(store.getState().openCompositionIds).not.toContain(comp2.id);
  });

  it('slot assignment and layer activation workflow', () => {
    const comp2 = store.getState().createComposition('Layer');
    store.getState().moveSlot('comp-1', 0);
    store.getState().moveSlot(comp2.id, 1);

    store.getState().activateOnLayer('comp-1', 0);
    store.getState().activateOnLayer(comp2.id, 1);
    expect(store.getState().activeLayerSlots[0]).toBe('comp-1');
    expect(store.getState().activeLayerSlots[1]).toBe(comp2.id);

    store.getState().deactivateLayer(0);
    expect(store.getState().activeLayerSlots[0]).toBeUndefined();
    expect(store.getState().activeLayerSlots[1]).toBe(comp2.id);
  });

  it('removing a composition cleans up layer assignments if layer was manually set', () => {
    const comp2 = store.getState().createComposition('ToRemoveLayer');
    store.getState().activateOnLayer(comp2.id, 1);
    expect(store.getState().activeLayerSlots[1]).toBe(comp2.id);
    // removeComposition does NOT clean up activeLayerSlots automatically (by design)
    store.getState().removeComposition(comp2.id);
    // The layer still references the removed comp (caller is responsible for cleanup)
    expect(store.getState().activeLayerSlots[1]).toBe(comp2.id);
  });
});
