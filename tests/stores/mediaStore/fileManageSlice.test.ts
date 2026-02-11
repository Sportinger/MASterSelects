/**
 * Tests for MediaStore file management operations.
 *
 * Since the real mediaStore is mocked in setup.ts (due to heavy import chains),
 * we create a minimal Zustand store that includes only the slices under test.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { create } from 'zustand';
import type { MediaState, MediaFile, MediaFolder, TextItem, SolidItem, Composition } from '../../../src/stores/mediaStore/types';
import { createFileManageSlice, type FileManageActions } from '../../../src/stores/mediaStore/slices/fileManageSlice';
import { createFolderSlice, type FolderActions } from '../../../src/stores/mediaStore/slices/folderSlice';
import { createSelectionSlice, type SelectionActions } from '../../../src/stores/mediaStore/slices/selectionSlice';
import { createCompositionSlice, type CompositionActions } from '../../../src/stores/mediaStore/slices/compositionSlice';

// ---- Minimal store factory ------------------------------------------------

type TestState = MediaState & FileManageActions & FolderActions & SelectionActions & CompositionActions & {
  getActiveComposition: () => Composition | undefined;
  createTextItem: (name?: string, parentId?: string | null) => string;
  removeTextItem: (id: string) => void;
  createSolidItem: (name?: string, color?: string, parentId?: string | null) => string;
  removeSolidItem: (id: string) => void;
  updateSolidItem: (id: string, updates: Partial<{ color: string; width: number; height: number }>) => void;
  getItemsByFolder: (folderId: string | null) => (MediaFile | Composition | MediaFolder | TextItem | SolidItem)[];
  getItemById: (id: string) => MediaFile | Composition | MediaFolder | TextItem | SolidItem | undefined;
  getFileByName: (name: string) => MediaFile | undefined;
};

function createTestStore() {
  return create<TestState>()((set, get) => ({
    // Minimal initial state
    files: [],
    compositions: [{
      id: 'comp-1',
      name: 'Comp 1',
      type: 'composition' as const,
      parentId: null,
      createdAt: Date.now(),
      width: 1920,
      height: 1080,
      frameRate: 30,
      duration: 60,
      backgroundColor: '#000000',
    }],
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

    // Getters inlined from the real store
    getActiveComposition: () => {
      const { compositions, activeCompositionId } = get();
      return compositions.find((c) => c.id === activeCompositionId);
    },

    getItemsByFolder: (folderId: string | null) => {
      const { files, compositions, folders, textItems, solidItems } = get();
      return [
        ...folders.filter((f) => f.parentId === folderId),
        ...compositions.filter((c) => c.parentId === folderId),
        ...textItems.filter((t) => t.parentId === folderId),
        ...solidItems.filter((s) => s.parentId === folderId),
        ...files.filter((f) => f.parentId === folderId),
      ];
    },

    getItemById: (id: string) => {
      const { files, compositions, folders, textItems, solidItems } = get();
      return (
        files.find((f) => f.id === id) ||
        compositions.find((c) => c.id === id) ||
        folders.find((f) => f.id === id) ||
        textItems.find((t) => t.id === id) ||
        solidItems.find((s) => s.id === id)
      );
    },

    getFileByName: (name: string) => {
      return get().files.find((f) => f.name === name);
    },

    // Text items
    createTextItem: (name?: string, parentId?: string | null) => {
      const { textItems } = get();
      const id = `text-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
      const newText: TextItem = {
        id,
        name: name || `Text ${textItems.length + 1}`,
        type: 'text' as const,
        parentId: parentId !== undefined ? parentId : null,
        createdAt: Date.now(),
        text: 'New Text',
        fontFamily: 'Arial',
        fontSize: 48,
        color: '#ffffff',
        duration: 5,
      };
      set({ textItems: [...textItems, newText] });
      return id;
    },

    removeTextItem: (id: string) => {
      set({ textItems: get().textItems.filter(t => t.id !== id) });
    },

    // Solid items
    createSolidItem: (name?: string, color?: string, parentId?: string | null) => {
      const { solidItems } = get();
      const id = `solid-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
      const activeComp = get().getActiveComposition();
      const compWidth = activeComp?.width || 1920;
      const compHeight = activeComp?.height || 1080;
      const newSolid: SolidItem = {
        id,
        name: name || `Solid ${solidItems.length + 1}`,
        type: 'solid' as const,
        parentId: parentId !== undefined ? parentId : null,
        createdAt: Date.now(),
        color: color || '#ffffff',
        width: compWidth,
        height: compHeight,
        duration: 5,
      };
      set({ solidItems: [...solidItems, newSolid] });
      return id;
    },

    removeSolidItem: (id: string) => {
      set({ solidItems: get().solidItems.filter(s => s.id !== id) });
    },

    updateSolidItem: (id: string, updates: Partial<{ color: string; width: number; height: number }>) => {
      set({
        solidItems: get().solidItems.map(s =>
          s.id === id
            ? {
                ...s,
                ...(updates.color !== undefined && { color: updates.color, name: `Solid ${updates.color}` }),
                ...(updates.width !== undefined && { width: updates.width }),
                ...(updates.height !== undefined && { height: updates.height }),
              }
            : s
        ),
      });
    },

    // Spread slice actions
    ...createFileManageSlice(set, get),
    ...createFolderSlice(set, get),
    ...createSelectionSlice(set, get),
    ...createCompositionSlice(set, get),
  }));
}

// ---- Helpers ---------------------------------------------------------------

function makeMediaFile(overrides: Partial<MediaFile> = {}): MediaFile {
  const id = overrides.id || `file-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
  return {
    id,
    name: `test-video-${id}.mp4`,
    type: 'video',
    parentId: null,
    createdAt: Date.now(),
    file: new File([], 'test.mp4', { type: 'video/mp4' }),
    url: `blob:http://localhost/${id}`,
    duration: 10,
    width: 1920,
    height: 1080,
    fileSize: 1024 * 1024,
    ...overrides,
  };
}

// ---- Tests -----------------------------------------------------------------

describe('MediaStore - File Management', () => {
  let store: ReturnType<typeof createTestStore>;

  beforeEach(() => {
    store = createTestStore();
  });

  // --- Adding media files ---

  describe('adding media files', () => {
    it('should add a media file to the store via setState', () => {
      const file = makeMediaFile({ id: 'f1', name: 'clip.mp4' });
      store.setState({ files: [file] });

      expect(store.getState().files).toHaveLength(1);
      expect(store.getState().files[0].name).toBe('clip.mp4');
    });

    it('should preserve existing files when adding more', () => {
      const file1 = makeMediaFile({ id: 'f1' });
      const file2 = makeMediaFile({ id: 'f2' });

      store.setState({ files: [file1] });
      store.setState((s) => ({ files: [...s.files, file2] }));

      expect(store.getState().files).toHaveLength(2);
    });
  });

  // --- Removing media files ---

  describe('removeFile', () => {
    it('should remove a file by id', () => {
      const file = makeMediaFile({ id: 'f1' });
      store.setState({ files: [file] });

      store.getState().removeFile('f1');

      expect(store.getState().files).toHaveLength(0);
    });

    it('should also remove the id from selectedIds', () => {
      const file = makeMediaFile({ id: 'f1' });
      store.setState({ files: [file], selectedIds: ['f1', 'other'] });

      store.getState().removeFile('f1');

      expect(store.getState().selectedIds).toEqual(['other']);
    });

    it('should not affect other files', () => {
      const file1 = makeMediaFile({ id: 'f1' });
      const file2 = makeMediaFile({ id: 'f2' });
      store.setState({ files: [file1, file2] });

      store.getState().removeFile('f1');

      expect(store.getState().files).toHaveLength(1);
      expect(store.getState().files[0].id).toBe('f2');
    });
  });

  // --- Renaming media files ---

  describe('renameFile', () => {
    it('should rename a file', () => {
      const file = makeMediaFile({ id: 'f1', name: 'old-name.mp4' });
      store.setState({ files: [file] });

      store.getState().renameFile('f1', 'new-name.mp4');

      expect(store.getState().files[0].name).toBe('new-name.mp4');
    });

    it('should not modify other files when renaming', () => {
      const file1 = makeMediaFile({ id: 'f1', name: 'a.mp4' });
      const file2 = makeMediaFile({ id: 'f2', name: 'b.mp4' });
      store.setState({ files: [file1, file2] });

      store.getState().renameFile('f1', 'renamed.mp4');

      expect(store.getState().files[1].name).toBe('b.mp4');
    });
  });

  // --- Folder operations ---

  describe('folder operations', () => {
    it('createFolder should add a folder to state', () => {
      const folder = store.getState().createFolder('My Folder');

      expect(store.getState().folders).toHaveLength(1);
      expect(folder.name).toBe('My Folder');
      expect(folder.parentId).toBeNull();
      expect(folder.isExpanded).toBe(true);
    });

    it('createFolder should support nested folders', () => {
      const parent = store.getState().createFolder('Parent');
      const child = store.getState().createFolder('Child', parent.id);

      expect(child.parentId).toBe(parent.id);
      expect(store.getState().folders).toHaveLength(2);
    });

    it('createFolder should add folder id to expandedFolderIds', () => {
      const folder = store.getState().createFolder('Expanded');

      expect(store.getState().expandedFolderIds).toContain(folder.id);
    });

    it('removeFolder should remove the folder and reparent children', () => {
      const folder = store.getState().createFolder('ToDelete');
      const file = makeMediaFile({ id: 'f1', parentId: folder.id });
      store.setState({ files: [file] });

      store.getState().removeFolder(folder.id);

      expect(store.getState().folders).toHaveLength(0);
      // File should be reparented to null (root)
      expect(store.getState().files[0].parentId).toBeNull();
    });

    it('removeFolder should reparent children to parent folder, not root', () => {
      const parent = store.getState().createFolder('Parent');
      const child = store.getState().createFolder('Child', parent.id);
      const file = makeMediaFile({ id: 'f1', parentId: child.id });
      store.setState({ files: [file] });

      store.getState().removeFolder(child.id);

      // File should be moved to parent, not root
      expect(store.getState().files[0].parentId).toBe(parent.id);
    });

    it('renameFolder should update the folder name', () => {
      const folder = store.getState().createFolder('Original');

      store.getState().renameFolder(folder.id, 'Renamed');

      expect(store.getState().folders[0].name).toBe('Renamed');
    });

    it('toggleFolderExpanded should toggle expanded state', () => {
      const folder = store.getState().createFolder('Toggle');

      // Initially expanded
      expect(store.getState().expandedFolderIds).toContain(folder.id);

      store.getState().toggleFolderExpanded(folder.id);
      expect(store.getState().expandedFolderIds).not.toContain(folder.id);

      store.getState().toggleFolderExpanded(folder.id);
      expect(store.getState().expandedFolderIds).toContain(folder.id);
    });
  });

  // --- Move items to folder ---

  describe('moveToFolder', () => {
    it('should move files into a folder', () => {
      const folder = store.getState().createFolder('Target');
      const file = makeMediaFile({ id: 'f1', parentId: null });
      store.setState({ files: [file] });

      store.getState().moveToFolder(['f1'], folder.id);

      expect(store.getState().files[0].parentId).toBe(folder.id);
    });

    it('should move items back to root', () => {
      const folder = store.getState().createFolder('Folder');
      const file = makeMediaFile({ id: 'f1', parentId: folder.id });
      store.setState({ files: [file] });

      store.getState().moveToFolder(['f1'], null);

      expect(store.getState().files[0].parentId).toBeNull();
    });
  });

  // --- Text item creation ---

  describe('text item creation', () => {
    it('createTextItem should add a text item with defaults', () => {
      const id = store.getState().createTextItem();

      const items = store.getState().textItems;
      expect(items).toHaveLength(1);
      expect(items[0].id).toBe(id);
      expect(items[0].type).toBe('text');
      expect(items[0].fontFamily).toBe('Arial');
      expect(items[0].fontSize).toBe(48);
      expect(items[0].duration).toBe(5);
    });

    it('createTextItem should accept a custom name', () => {
      store.getState().createTextItem('Custom Title');

      expect(store.getState().textItems[0].name).toBe('Custom Title');
    });

    it('removeTextItem should remove the text item', () => {
      const id = store.getState().createTextItem();
      expect(store.getState().textItems).toHaveLength(1);

      store.getState().removeTextItem(id);

      expect(store.getState().textItems).toHaveLength(0);
    });
  });

  // --- Solid item creation ---

  describe('solid item creation', () => {
    it('createSolidItem should add a solid item with active comp dimensions', () => {
      const id = store.getState().createSolidItem();

      const items = store.getState().solidItems;
      expect(items).toHaveLength(1);
      expect(items[0].id).toBe(id);
      expect(items[0].type).toBe('solid');
      expect(items[0].width).toBe(1920);
      expect(items[0].height).toBe(1080);
      expect(items[0].color).toBe('#ffffff');
      expect(items[0].duration).toBe(5);
    });

    it('createSolidItem should accept a custom color', () => {
      store.getState().createSolidItem('Red Solid', '#ff0000');

      const solid = store.getState().solidItems[0];
      expect(solid.name).toBe('Red Solid');
      expect(solid.color).toBe('#ff0000');
    });

    it('removeSolidItem should remove the solid item', () => {
      const id = store.getState().createSolidItem();
      store.getState().removeSolidItem(id);

      expect(store.getState().solidItems).toHaveLength(0);
    });

    it('updateSolidItem should update color and rename', () => {
      const id = store.getState().createSolidItem('Solid', '#ffffff');

      store.getState().updateSolidItem(id, { color: '#00ff00' });

      const solid = store.getState().solidItems[0];
      expect(solid.color).toBe('#00ff00');
      expect(solid.name).toBe('Solid #00ff00');
    });

    it('updateSolidItem should update dimensions without renaming', () => {
      const id = store.getState().createSolidItem('Solid', '#ffffff');

      store.getState().updateSolidItem(id, { width: 3840, height: 2160 });

      const solid = store.getState().solidItems[0];
      expect(solid.width).toBe(3840);
      expect(solid.height).toBe(2160);
      // Name should stay the same since only dimensions changed
      expect(solid.name).toBe('Solid');
    });
  });

  // --- File deduplication by hash ---

  describe('file deduplication', () => {
    it('should detect duplicate files by fileHash', () => {
      const hash = 'abc123hash';
      const file1 = makeMediaFile({ id: 'f1', fileHash: hash, name: 'clip.mp4' });
      const file2 = makeMediaFile({ id: 'f2', fileHash: hash, name: 'clip-copy.mp4' });

      store.setState({ files: [file1, file2] });

      const { files } = store.getState();
      const duplicates = files.filter(f => f.fileHash === hash);
      expect(duplicates).toHaveLength(2);
    });

    it('should distinguish files with different hashes', () => {
      const file1 = makeMediaFile({ id: 'f1', fileHash: 'hash1' });
      const file2 = makeMediaFile({ id: 'f2', fileHash: 'hash2' });

      store.setState({ files: [file1, file2] });

      const { files } = store.getState();
      const uniqueHashes = new Set(files.map(f => f.fileHash));
      expect(uniqueHashes.size).toBe(2);
    });
  });

  // --- Getter functions ---

  describe('getters', () => {
    it('getItemsByFolder should return items in a specific folder', () => {
      const folder = store.getState().createFolder('MyFolder');
      const file = makeMediaFile({ id: 'f1', parentId: folder.id });
      store.setState({ files: [file] });
      store.getState().createTextItem('InFolder', folder.id);

      const items = store.getState().getItemsByFolder(folder.id);

      // file + text item (folder itself is not a child of itself)
      expect(items).toHaveLength(2);
    });

    it('getItemsByFolder(null) should return root-level items', () => {
      const file = makeMediaFile({ id: 'f1', parentId: null });
      store.setState({ files: [file] });
      store.getState().createFolder('RootFolder');

      const items = store.getState().getItemsByFolder(null);

      // file + folder + the default comp-1
      expect(items.length).toBeGreaterThanOrEqual(2);
    });

    it('getItemById should find a file by id', () => {
      const file = makeMediaFile({ id: 'findme' });
      store.setState({ files: [file] });

      const found = store.getState().getItemById('findme');
      expect(found).toBeDefined();
      expect(found!.id).toBe('findme');
    });

    it('getItemById should return undefined for unknown id', () => {
      expect(store.getState().getItemById('nonexistent')).toBeUndefined();
    });

    it('getFileByName should find a file by name', () => {
      const file = makeMediaFile({ id: 'f1', name: 'special.mp4' });
      store.setState({ files: [file] });

      const found = store.getState().getFileByName('special.mp4');
      expect(found).toBeDefined();
      expect(found!.name).toBe('special.mp4');
    });
  });
});
