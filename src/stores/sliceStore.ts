// Slice Store - manages slice configurations per output target
// Separate from renderTargetStore since slices are user-editing state

import { create } from 'zustand';
import { useRenderTargetStore } from './renderTargetStore';
import { useMediaStore } from './mediaStore';
import type { RenderSource } from '../types/renderTarget';
import type {
  OutputSlice,
  TargetSliceConfig,
  SliceWarp,
  Point2D,
} from '../types/outputSlice';
import { createDefaultSlice, DEFAULT_CORNERS, migrateSlice } from '../types/outputSlice';

interface SliceState {
  configs: Map<string, TargetSliceConfig>;
  activeTab: 'input' | 'output';
  previewingTargetId: string | null; // which target the OM preview canvas mirrors
}

interface SliceActions {
  setActiveTab: (tab: 'input' | 'output') => void;
  setPreviewingTargetId: (id: string | null) => void;
  getOrCreateConfig: (targetId: string) => TargetSliceConfig;
  removeConfig: (targetId: string) => void;
  addSlice: (targetId: string, name?: string) => string;
  removeSlice: (targetId: string, sliceId: string) => void;
  selectSlice: (targetId: string, sliceId: string | null) => void;
  setSliceEnabled: (targetId: string, sliceId: string, enabled: boolean) => void;
  setInputCorner: (targetId: string, sliceId: string, cornerIndex: number, point: Point2D) => void;
  setCornerPinCorner: (targetId: string, sliceId: string, cornerIndex: number, point: Point2D) => void;
  updateWarp: (targetId: string, sliceId: string, warp: SliceWarp) => void;
  resetSliceWarp: (targetId: string, sliceId: string) => void;
  matchInputToOutput: (targetId: string, sliceId: string) => void;
  matchOutputToInput: (targetId: string, sliceId: string) => void;
  saveToLocalStorage: () => void;
  loadFromLocalStorage: () => void;
}

/** Build localStorage key from current project name */
function getStorageKey(): string {
  const name = useMediaStore.getState().currentProjectName || 'Untitled Project';
  // Sanitize: replace non-alphanumeric with underscore
  const safe = name.replace(/[^a-zA-Z0-9_\- ]/g, '_').trim();
  return `Outputmanager_${safe}`;
}

/** Serialize configs Map to a JSON-safe object */
function serializeConfigs(configs: Map<string, TargetSliceConfig>): Record<string, TargetSliceConfig> {
  const obj: Record<string, TargetSliceConfig> = {};
  for (const [key, value] of configs) {
    obj[key] = value;
  }
  return obj;
}

/** Deserialize JSON object back to configs Map */
function deserializeConfigs(obj: Record<string, TargetSliceConfig>): Map<string, TargetSliceConfig> {
  const map = new Map<string, TargetSliceConfig>();
  for (const key of Object.keys(obj)) {
    map.set(key, obj[key]);
  }
  return map;
}

function updateSliceInConfig(
  config: TargetSliceConfig,
  sliceId: string,
  updater: (slice: OutputSlice) => OutputSlice
): TargetSliceConfig {
  return {
    ...config,
    slices: config.slices.map((s) => (s.id === sliceId ? updater(s) : s)),
  };
}

export const useSliceStore = create<SliceState & SliceActions>()((set, get) => ({
  configs: new Map(),
  activeTab: 'output',
  previewingTargetId: null,

  setActiveTab: (tab) => set({ activeTab: tab }),
  setPreviewingTargetId: (id) => set({ previewingTargetId: id }),

  getOrCreateConfig: (targetId) => {
    const { configs } = get();
    const existing = configs.get(targetId);
    if (existing) return existing;

    const config: TargetSliceConfig = {
      targetId,
      slices: [],
      selectedSliceId: null,
    };
    const next = new Map(configs);
    next.set(targetId, config);
    set({ configs: next });
    return config;
  },

  removeConfig: (targetId) => {
    set((state) => {
      const next = new Map(state.configs);
      next.delete(targetId);
      return { configs: next };
    });
  },

  addSlice: (targetId, name?) => {
    const config = get().getOrCreateConfig(targetId);
    const slice = createDefaultSlice(name);
    const next = new Map(get().configs);
    next.set(targetId, {
      ...config,
      slices: [...config.slices, slice],
      selectedSliceId: slice.id,
    });
    set({ configs: next });
    return slice.id;
  },

  removeSlice: (targetId, sliceId) => {
    set((state) => {
      const config = state.configs.get(targetId);
      if (!config) return state;
      const newSlices = config.slices.filter((s) => s.id !== sliceId);
      const next = new Map(state.configs);
      next.set(targetId, {
        ...config,
        slices: newSlices,
        selectedSliceId: config.selectedSliceId === sliceId
          ? (newSlices.length > 0 ? newSlices[0].id : null)
          : config.selectedSliceId,
      });
      return { configs: next };
    });
  },

  selectSlice: (targetId, sliceId) => {
    set((state) => {
      const config = state.configs.get(targetId);
      if (!config) return state;
      const next = new Map(state.configs);
      next.set(targetId, { ...config, selectedSliceId: sliceId });
      return { configs: next };
    });
  },

  setSliceEnabled: (targetId, sliceId, enabled) => {
    set((state) => {
      const config = state.configs.get(targetId);
      if (!config) return state;
      const next = new Map(state.configs);
      next.set(targetId, updateSliceInConfig(config, sliceId, (s) => ({ ...s, enabled })));
      return { configs: next };
    });
  },

  setInputCorner: (targetId, sliceId, cornerIndex, point) => {
    set((state) => {
      const config = state.configs.get(targetId);
      if (!config) return state;
      const next = new Map(state.configs);
      next.set(targetId, updateSliceInConfig(config, sliceId, (s) => {
        const corners = [...s.inputCorners] as [Point2D, Point2D, Point2D, Point2D];
        corners[cornerIndex] = point;
        return { ...s, inputCorners: corners };
      }));
      return { configs: next };
    });
  },

  setCornerPinCorner: (targetId, sliceId, cornerIndex, point) => {
    set((state) => {
      const config = state.configs.get(targetId);
      if (!config) return state;
      const next = new Map(state.configs);
      next.set(targetId, updateSliceInConfig(config, sliceId, (s) => {
        if (s.warp.mode !== 'cornerPin') return s;
        const corners = [...s.warp.corners] as [Point2D, Point2D, Point2D, Point2D];
        corners[cornerIndex] = point;
        return { ...s, warp: { ...s.warp, corners } };
      }));
      return { configs: next };
    });
  },

  updateWarp: (targetId, sliceId, warp) => {
    set((state) => {
      const config = state.configs.get(targetId);
      if (!config) return state;
      const next = new Map(state.configs);
      next.set(targetId, updateSliceInConfig(config, sliceId, (s) => ({ ...s, warp })));
      return { configs: next };
    });
  },

  resetSliceWarp: (targetId, sliceId) => {
    set((state) => {
      const config = state.configs.get(targetId);
      if (!config) return state;
      const next = new Map(state.configs);
      next.set(targetId, updateSliceInConfig(config, sliceId, (s) => ({
        ...s,
        inputCorners: [...DEFAULT_CORNERS] as [Point2D, Point2D, Point2D, Point2D],
        warp: {
          mode: 'cornerPin' as const,
          corners: [...DEFAULT_CORNERS] as [Point2D, Point2D, Point2D, Point2D],
        },
      })));
      return { configs: next };
    });
  },

  matchInputToOutput: (targetId, sliceId) => {
    set((state) => {
      const config = state.configs.get(targetId);
      if (!config) return state;
      const slice = config.slices.find((s) => s.id === sliceId);
      if (!slice || slice.warp.mode !== 'cornerPin') return state;
      // Copy output warp corners → input corners
      const outputCorners = slice.warp.corners;
      const next = new Map(state.configs);
      next.set(targetId, updateSliceInConfig(config, sliceId, (s) => ({
        ...s,
        inputCorners: [...outputCorners] as [Point2D, Point2D, Point2D, Point2D],
      })));
      return { configs: next };
    });
  },

  matchOutputToInput: (targetId, sliceId) => {
    set((state) => {
      const config = state.configs.get(targetId);
      if (!config) return state;
      const slice = config.slices.find((s) => s.id === sliceId);
      if (!slice || slice.warp.mode !== 'cornerPin') return state;
      // Copy input corners → output warp corners
      const inputCorners = slice.inputCorners;
      const next = new Map(state.configs);
      next.set(targetId, updateSliceInConfig(config, sliceId, (s) => ({
        ...s,
        warp: {
          mode: 'cornerPin' as const,
          corners: [...inputCorners] as [Point2D, Point2D, Point2D, Point2D],
        },
      })));
      return { configs: next };
    });
  },

  saveToLocalStorage: () => {
    const { configs } = get();
    const key = getStorageKey();
    try {
      // Save slice configs
      const data = JSON.stringify(serializeConfigs(configs));
      localStorage.setItem(key, data);

      // Also save output target metadata for reconnection after refresh
      const targets = useRenderTargetStore.getState().targets;
      const targetMeta: Array<{ id: string; name: string; source: RenderSource }> = [];
      for (const t of targets.values()) {
        if (t.destinationType === 'window' || t.destinationType === 'tab') {
          targetMeta.push({ id: t.id, name: t.name, source: t.source });
        }
      }
      localStorage.setItem(key + '_targets', JSON.stringify(targetMeta));
    } catch (e) {
      console.error('Failed to save Output Manager config:', e);
    }
  },

  loadFromLocalStorage: () => {
    const key = getStorageKey();
    try {
      const raw = localStorage.getItem(key);
      if (!raw) return;
      const parsed = JSON.parse(raw) as Record<string, TargetSliceConfig>;
      // Migrate legacy slices that used inputRect
      for (const config of Object.values(parsed)) {
        config.slices = config.slices.map((s) => migrateSlice(s));
      }
      const configs = deserializeConfigs(parsed);
      set({ configs });
    } catch (e) {
      console.error('Failed to load Output Manager config:', e);
    }
  },
}));

// Auto-save configs to localStorage on every change (debounced 500ms)
let autoSaveTimer: ReturnType<typeof setTimeout> | null = null;
let prevConfigs: Map<string, TargetSliceConfig> | null = null;
useSliceStore.subscribe((state) => {
  if (state.configs !== prevConfigs) {
    prevConfigs = state.configs;
    if (autoSaveTimer) clearTimeout(autoSaveTimer);
    autoSaveTimer = setTimeout(() => {
      useSliceStore.getState().saveToLocalStorage();
    }, 500);
  }
});

// Auto-load configs from localStorage on module init
try {
  useSliceStore.getState().loadFromLocalStorage();
} catch {
  // Ignore errors during initial load
}

/** Get saved target metadata from localStorage (for reconnection) */
export function getSavedTargetMeta(): Array<{ id: string; name: string; source: RenderSource }> {
  const key = getStorageKey();
  try {
    const raw = localStorage.getItem(key + '_targets');
    if (!raw) return [];
    return JSON.parse(raw) as Array<{ id: string; name: string; source: RenderSource }>;
  } catch {
    return [];
  }
}

// Cleanup subscription: remove orphaned configs only when a target is explicitly removed
// (not when targets simply don't exist yet, e.g. after page refresh)
let knownTargetIds = new Set<string>();
useRenderTargetStore.subscribe(
  (state) => state.targets,
  (targets) => {
    // Find targets that were known before but are now gone
    for (const prevId of knownTargetIds) {
      if (!targets.has(prevId)) {
        useSliceStore.getState().removeConfig(prevId);
      }
    }
    // Update known set
    knownTargetIds = new Set(targets.keys());
  }
);
