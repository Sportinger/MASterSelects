// Zustand store for WebVJ Mixer state management

import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import type { Layer, BlendMode, Effect, OutputWindow, MIDIMapping, EngineStats } from '../types';
import { WebCodecsPlayer } from '../engine/WebCodecsPlayer';
import { audioManager } from '../services/audioManager';
import { Logger } from '../services/logger';

const log = Logger.create('MixerStore');

// Slot group - slots that are visually linked together
export type SlotGroup = number[];

interface MixerState {
  // Layers
  layers: Layer[];
  selectedLayerId: string | null;

  // Grid configuration
  gridColumns: number;
  gridRows: number;

  // Slot groups (linked slots that highlight together)
  slotGroups: SlotGroup[];

  // Output
  outputWindows: OutputWindow[];
  outputResolution: { width: number; height: number };
  fps: number;

  // Audio
  masterVolume: number; // 0-1
  eqBands: number[]; // 10-band EQ gains (-12 to +12 dB)

  // Engine
  isEngineReady: boolean;
  engineStats: EngineStats;
  gpuInfo: { vendor: string; device: string; description: string } | null;

  // MIDI
  midiEnabled: boolean;
  midiMappings: MIDIMapping[];

  // UI
  isPlaying: boolean;

  // Actions
  addLayer: () => void;
  removeLayer: (id: string) => void;
  selectLayer: (id: string | null) => void;
  updateLayer: (id: string, updates: Partial<Layer>) => void;
  reorderLayers: (fromIndex: number, toIndex: number) => void;
  swapSlots: (fromIndex: number, toIndex: number) => void;
  triggerColumn: (columnIndex: number) => void;
  triggerRow: (rowIndex: number) => void;
  triggerSlot: (slotIndex: number) => void;

  setLayerSource: (layerId: string, file: File) => void;
  createLayerAtSlot: (slotIndex: number, file: File) => void;
  clearLayerSource: (layerId: string) => void;

  setLayerOpacity: (layerId: string, opacity: number) => void;
  setLayerBlendMode: (layerId: string, blendMode: BlendMode) => void;
  setLayerVisibility: (layerId: string, visible: boolean) => void;
  setLayerTransform: (
    layerId: string,
    transform: { position?: { x: number; y: number; z: number }; scale?: { x: number; y: number }; rotation?: number }
  ) => void;
  updateLayerMaskProps: (layerId: string, feather: number, quality: number, invert: boolean) => void;

  addEffect: (layerId: string, effectType: string) => void;
  removeEffect: (layerId: string, effectId: string) => void;
  updateEffect: (layerId: string, effectId: string, params: Record<string, number | boolean | string>) => void;

  addOutputWindow: (output: OutputWindow) => void;
  removeOutputWindow: (id: string) => void;

  setEngineReady: (ready: boolean) => void;
  setGpuInfo: (info: { vendor: string; device: string; description: string } | null) => void;
  setEngineStats: (stats: EngineStats) => void;

  setPlaying: (playing: boolean) => void;
  setResolution: (width: number, height: number) => void;

  setMidiEnabled: (enabled: boolean) => void;
  addMidiMapping: (mapping: MIDIMapping) => void;
  removeMidiMapping: (index: number) => void;

  // Audio actions
  setMasterVolume: (volume: number) => void;
  setEQBand: (bandIndex: number, gain: number) => void;
  setAllEQBands: (gains: number[]) => void;
  resetEQ: () => void;

  // Grid actions
  setGridColumns: (columns: number) => void;
  setGridRows: (rows: number) => void;

  // Slot group actions
  createSlotGroup: (slotIndices: number[]) => void;
  addToSlotGroup: (groupIndex: number, slotIndex: number) => void;
  removeFromSlotGroup: (slotIndex: number) => void;
  getSlotGroup: (slotIndex: number) => SlotGroup | null;

  // Slot operations
  duplicateSlot: (fromIndex: number, toIndex: number) => void;
  findNextFreeSlot: (afterIndex?: number) => number | null;
  moveGroup: (groupSlots: number[], targetIndex: number) => void;
}

const createDefaultLayer = (index: number): Layer => ({
  id: `slot_${Date.now()}_${index}`,
  name: `Slot ${index + 1}`,
  visible: true,
  opacity: 1,
  blendMode: 'normal',
  source: null,
  effects: [],
  position: { x: 0, y: 0, z: 0 },
  scale: { x: 1, y: 1 },
  rotation: 0,
});

export const useMixerStore = create<MixerState>()(
  subscribeWithSelector((set, get) => ({
    // Initial state - empty grid
    layers: [],
    selectedLayerId: null,

    // Grid configuration (default 5x5)
    gridColumns: 5,
    gridRows: 5,

    // Slot groups
    slotGroups: [],

    outputWindows: [],
    outputResolution: { width: 1920, height: 1080 },
    fps: 60,

    masterVolume: 1, // Default to full volume
    eqBands: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], // 10 bands at 0 dB (flat)

    isEngineReady: false,
    gpuInfo: null,
    engineStats: {
      fps: 0,
      frameTime: 0,
      gpuMemory: 0,
      timing: { rafGap: 0, importTexture: 0, renderPass: 0, submit: 0, total: 0 },
      drops: { count: 0, lastSecond: 0, reason: 'none' },
      layerCount: 0,
      targetFps: 60,
      decoder: 'none',
      audio: { playing: 0, drift: 0, status: 'silent' },
      isIdle: false,
    },

    midiEnabled: false,
    midiMappings: [],

    isPlaying: false,

    // Layer actions
    addLayer: () => {
      const { layers } = get();
      set({ layers: [...layers, createDefaultLayer(layers.length)] });
    },

    removeLayer: (id: string) => {
      const { layers, selectedLayerId } = get();
      const newLayers = layers.filter((l) => l?.id !== id);
      set({
        layers: newLayers,
        selectedLayerId: selectedLayerId === id ? null : selectedLayerId,
      });
    },

    selectLayer: (id: string | null) => {
      set({ selectedLayerId: id });
    },

    updateLayer: (id: string, updates: Partial<Layer>) => {
      const { layers } = get();
      set({
        layers: layers.map((l) => (l?.id === id ? { ...l, ...updates } : l)),
      });
    },

    reorderLayers: (fromIndex: number, toIndex: number) => {
      const { layers } = get();
      const newLayers = [...layers];
      const [removed] = newLayers.splice(fromIndex, 1);
      newLayers.splice(toIndex, 0, removed);
      set({ layers: newLayers });
    },

    swapSlots: (fromIndex: number, toIndex: number) => {
      if (fromIndex === toIndex) return;
      const { layers } = get();

      // Create sparse array copy
      const newLayers = [...layers];

      // Swap the two slots (can be undefined for empty slots)
      const temp = newLayers[fromIndex];
      newLayers[fromIndex] = newLayers[toIndex];
      newLayers[toIndex] = temp;

      // Update names if layers exist at those positions
      if (newLayers[fromIndex]) {
        newLayers[fromIndex] = { ...newLayers[fromIndex], name: `Slot ${fromIndex + 1}` };
      }
      if (newLayers[toIndex]) {
        newLayers[toIndex] = { ...newLayers[toIndex], name: `Slot ${toIndex + 1}` };
      }

      set({ layers: newLayers });
    },

    triggerColumn: (columnIndex: number) => {
      const { layers, triggerSlot, gridColumns, gridRows } = get();

      // Get slot indices for this column (0-indexed)
      const columnSlots = Array.from({ length: gridRows }, (_, row) => row * gridColumns + columnIndex);

      // Bang: activate clips in this column (set visible=true) and restart from beginning
      const newLayers = layers.map((layer, index) => {
        if (!layer) return layer;
        const isInColumn = columnSlots.includes(index);
        if (isInColumn) {
          return { ...layer, visible: true };
        }
        return layer;
      });

      set({ layers: newLayers });

      // Trigger (restart) videos in this column from beginning
      columnSlots.forEach(slotIndex => triggerSlot(slotIndex));
    },

    triggerRow: (rowIndex: number) => {
      const { layers, triggerSlot, gridColumns } = get();

      // Get slot indices for this row (0-indexed)
      const rowSlots = Array.from({ length: gridColumns }, (_, col) => rowIndex * gridColumns + col);

      // Bang: activate clips in this row (set visible=true) and restart from beginning
      const newLayers = layers.map((layer, index) => {
        if (!layer) return layer;
        const isInRow = rowSlots.includes(index);
        if (isInRow) {
          return { ...layer, visible: true };
        }
        return layer;
      });

      set({ layers: newLayers });

      // Trigger (restart) videos in this row from beginning
      rowSlots.forEach(slotIndex => triggerSlot(slotIndex));
    },

    triggerSlot: (slotIndex: number) => {
      const { layers } = get();
      const layer = layers[slotIndex];

      if (!layer) return;

      // Restart video if present
      if (layer.source?.videoElement) {
        layer.source.videoElement.currentTime = 0;
        layer.source.videoElement.play();
      }
      if (layer.source?.webCodecsPlayer) {
        layer.source.webCodecsPlayer.seek(0);
        layer.source.webCodecsPlayer.play();
      }
    },

    setLayerSource: (layerId: string, file: File) => {
      const { layers } = get();
      const layer = layers.find((l) => l?.id === layerId);
      if (!layer) return;

      const isVideo = file.type.startsWith('video/');
      const isImage = file.type.startsWith('image/');

      if (isVideo) {
        // Try WebCodecs first for hardware-accelerated decoding
        const hasWebCodecs = 'VideoDecoder' in window;
        const isMp4 = file.name.toLowerCase().endsWith('.mp4');
        log.debug(`Video load: ${file.name} | WebCodecs=${hasWebCodecs} | MP4=${isMp4}`);

        if (hasWebCodecs && isMp4) {
          const player = new WebCodecsPlayer({
            loop: true,
            onFrame: (frame) => {
              // Update the layer's videoFrame reference
              const currentLayers = get().layers;
              set({
                layers: currentLayers.map((l) =>
                  l?.id === layerId && l.source
                    ? { ...l, source: { ...l.source, videoFrame: frame } }
                    : l
                ),
              });
            },
            onError: (error) => {
              log.error('WebCodecs error, falling back to video element:', error);
              // Fallback to video element
              createVideoElement(file, layerId, get, set);
            },
          });

          player.loadFile(file).then(() => {
            log.debug(`WebCodecs loaded video: ${file.name} (${player.width}x${player.height})`);
            // Get fresh state to avoid stale closure
            const currentLayers = get().layers;
            set({
              layers: currentLayers.map((l) =>
                l?.id === layerId
                  ? {
                      ...l,
                      source: {
                        type: 'video',
                        file,
                        webCodecsPlayer: player,
                      },
                    }
                  : l
              ),
            });
            player.play();
          }).catch((error) => {
            log.error('WebCodecs failed to load, falling back:', error);
            createVideoElement(file, layerId, get, set);
          });
        } else {
          // Fallback to video element for non-MP4 or no WebCodecs support
          createVideoElement(file, layerId, get, set);
        }
      } else if (isImage) {
        const img = new Image();
        img.src = URL.createObjectURL(file);

        img.onload = () => {
          // Get fresh state to avoid stale closure
          const currentLayers = get().layers;
          set({
            layers: currentLayers.map((l) =>
              l?.id === layerId
                ? {
                    ...l,
                    source: {
                      type: 'image',
                      file,
                      imageElement: img,
                    },
                  }
                : l
            ),
          });
        };
      }
    },

    createLayerAtSlot: (slotIndex: number, file: File) => {
      const { layers, setLayerSource } = get();

      // Create a new layer at this specific slot index (sparse array)
      const newLayer: Layer = {
        id: `slot_${Date.now()}_${slotIndex}`,
        name: `Slot ${slotIndex + 1}`,
        visible: true,
        opacity: 1,
        blendMode: 'normal',
        source: null,
        effects: [],
        position: { x: 0, y: 0, z: 0 },
        scale: { x: 1, y: 1 },
        rotation: 0,
      };

      // Create sparse array with the new layer at the correct index
      const newLayers = [...layers];
      newLayers[slotIndex] = newLayer;

      set({ layers: newLayers });

      // Now load the file into this layer
      setLayerSource(newLayer.id, file);
    },

    clearLayerSource: (layerId: string) => {
      const { layers } = get();
      const layer = layers.find((l) => l?.id === layerId);

      if (layer?.source?.videoElement) {
        // Disconnect from audio manager
        audioManager.disconnectMediaElement(layer.source.videoElement);
        layer.source.videoElement.pause();
        URL.revokeObjectURL(layer.source.videoElement.src);
      }
      if (layer?.source?.imageElement) {
        URL.revokeObjectURL(layer.source.imageElement.src);
      }
      // Clean up WebCodecs player
      if (layer?.source?.webCodecsPlayer) {
        layer.source.webCodecsPlayer.destroy();
      }

      set({
        layers: layers.map((l) =>
          l?.id === layerId ? { ...l, source: null } : l
        ),
      });
    },

    setLayerOpacity: (layerId: string, opacity: number) => {
      const { updateLayer } = get();
      updateLayer(layerId, { opacity: Math.max(0, Math.min(1, opacity)) });
    },

    setLayerBlendMode: (layerId: string, blendMode: BlendMode) => {
      const { updateLayer } = get();
      updateLayer(layerId, { blendMode });
    },

    setLayerVisibility: (layerId: string, visible: boolean) => {
      const { updateLayer } = get();
      updateLayer(layerId, { visible });
    },

    setLayerTransform: (layerId: string, transform) => {
      const { layers } = get();
      const layer = layers.find((l) => l?.id === layerId);
      if (!layer) return;

      const updates: Partial<Layer> = {};
      if (transform.position) updates.position = transform.position;
      if (transform.scale) updates.scale = transform.scale;
      if (transform.rotation !== undefined) updates.rotation = transform.rotation;

      set({
        layers: layers.map((l) => (l?.id === layerId ? { ...l, ...updates } : l)),
      });
    },

    // Update mask properties for GPU processing (feather blur, quality, inversion)
    // These are cheap uniform updates - no texture regeneration needed
    updateLayerMaskProps: (layerId: string, feather: number, quality: number, invert: boolean) => {
      const { layers } = get();
      const layer = layers.find((l) => l?.id === layerId);
      if (!layer) return;

      // Only update if values changed to avoid unnecessary re-renders
      if (layer.maskFeather === feather && layer.maskFeatherQuality === quality && layer.maskInvert === invert) return;

      set({
        layers: layers.map((l) =>
          l?.id === layerId ? { ...l, maskFeather: feather, maskFeatherQuality: quality, maskInvert: invert } : l
        ),
      });
    },

    // Effect actions
    addEffect: (layerId: string, effectType: string) => {
      const { layers } = get();
      const effect: Effect = {
        id: `effect_${Date.now()}`,
        name: effectType,
        type: effectType as Effect['type'],
        enabled: true,
        params: getDefaultEffectParams(effectType),
      };

      set({
        layers: layers.map((l) =>
          l?.id === layerId ? { ...l, effects: [...l.effects, effect] } : l
        ),
      });
    },

    removeEffect: (layerId: string, effectId: string) => {
      const { layers } = get();
      set({
        layers: layers.map((l) =>
          l?.id === layerId
            ? { ...l, effects: l.effects.filter((e) => e.id !== effectId) }
            : l
        ),
      });
    },

    updateEffect: (layerId: string, effectId: string, params: Record<string, number | boolean | string>) => {
      const { layers } = get();
      set({
        layers: layers.map((l) =>
          l?.id === layerId
            ? {
                ...l,
                effects: l.effects.map((e) =>
                  e.id === effectId ? { ...e, params: { ...e.params, ...params } } : e
                ),
              }
            : l
        ),
      });
    },

    // Output actions
    addOutputWindow: (output: OutputWindow) => {
      const { outputWindows } = get();
      set({ outputWindows: [...outputWindows, output] });
    },

    removeOutputWindow: (id: string) => {
      const { outputWindows } = get();
      set({ outputWindows: outputWindows.filter((o) => o.id !== id) });
    },

    // Engine actions
    setEngineReady: (ready: boolean) => {
      set({ isEngineReady: ready });
    },

    setGpuInfo: (info: { vendor: string; device: string; description: string } | null) => {
      set({ gpuInfo: info });
    },

    setEngineStats: (stats: EngineStats) => {
      set({ engineStats: stats });
    },

    setPlaying: (playing: boolean) => {
      set({ isPlaying: playing });
    },

    setResolution: (width: number, height: number) => {
      set({ outputResolution: { width, height } });
    },

    // MIDI actions
    setMidiEnabled: (enabled: boolean) => {
      set({ midiEnabled: enabled });
    },

    addMidiMapping: (mapping: MIDIMapping) => {
      const { midiMappings } = get();
      set({ midiMappings: [...midiMappings, mapping] });
    },

    removeMidiMapping: (index: number) => {
      const { midiMappings } = get();
      set({ midiMappings: midiMappings.filter((_, i) => i !== index) });
    },

    // Audio actions
    setMasterVolume: (volume: number) => {
      const clamped = Math.max(0, Math.min(1, volume));
      set({ masterVolume: clamped });
      // Sync with audio manager
      audioManager.setMasterVolume(clamped);
    },

    setEQBand: (bandIndex: number, gain: number) => {
      const { eqBands } = get();
      const clamped = Math.max(-12, Math.min(12, gain));
      const newBands = [...eqBands];
      newBands[bandIndex] = clamped;
      set({ eqBands: newBands });
      // Sync with audio manager
      audioManager.setEQBand(bandIndex, clamped);
    },

    setAllEQBands: (gains: number[]) => {
      const newBands = gains.map(g => Math.max(-12, Math.min(12, g)));
      set({ eqBands: newBands });
      // Sync with audio manager
      audioManager.setAllEQBands(newBands);
    },

    resetEQ: () => {
      const flatBands = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
      set({ eqBands: flatBands });
      audioManager.resetEQ();
    },

    // Grid actions
    setGridColumns: (columns: number) => {
      const clamped = Math.max(1, Math.min(10, columns));
      set({ gridColumns: clamped });
    },

    setGridRows: (rows: number) => {
      const clamped = Math.max(1, Math.min(10, rows));
      set({ gridRows: clamped });
    },

    // Slot group actions
    createSlotGroup: (slotIndices: number[]) => {
      const { slotGroups } = get();
      // Remove these slots from any existing groups first
      const cleanedGroups = slotGroups
        .map(group => group.filter(idx => !slotIndices.includes(idx)))
        .filter(group => group.length > 1);
      set({ slotGroups: [...cleanedGroups, slotIndices] });
    },

    addToSlotGroup: (groupIndex: number, slotIndex: number) => {
      const { slotGroups } = get();
      if (groupIndex >= 0 && groupIndex < slotGroups.length) {
        const newGroups = [...slotGroups];
        if (!newGroups[groupIndex].includes(slotIndex)) {
          newGroups[groupIndex] = [...newGroups[groupIndex], slotIndex];
        }
        set({ slotGroups: newGroups });
      }
    },

    removeFromSlotGroup: (slotIndex: number) => {
      const { slotGroups } = get();
      const newGroups = slotGroups
        .map(group => group.filter(idx => idx !== slotIndex))
        .filter(group => group.length > 1); // Remove groups with < 2 members
      set({ slotGroups: newGroups });
    },

    getSlotGroup: (slotIndex: number) => {
      const { slotGroups } = get();
      return slotGroups.find(group => group.includes(slotIndex)) || null;
    },

    // Slot operations
    duplicateSlot: (fromIndex: number, toIndex: number) => {
      const { layers } = get();
      const sourceLayer = layers[fromIndex];

      log.debug('duplicateSlot:', fromIndex, '->', toIndex, 'source:', sourceLayer?.source);

      if (!sourceLayer?.source) {
        log.debug('No source to duplicate');
        return;
      }

      // Get the file - either from the source or we need to handle video/image elements
      const file = sourceLayer.source.file;
      if (!file) {
        log.debug('No file in source, cannot duplicate');
        return;
      }

      // Create a new layer at the target slot
      const newLayer: Layer = {
        id: `slot_${Date.now()}_${toIndex}`,
        name: `Slot ${toIndex + 1}`,
        visible: sourceLayer.visible,
        opacity: sourceLayer.opacity,
        blendMode: sourceLayer.blendMode,
        source: null,
        effects: [...sourceLayer.effects],
        position: { ...sourceLayer.position },
        scale: { ...sourceLayer.scale },
        rotation: sourceLayer.rotation,
      };

      const newLayers = [...layers];
      newLayers[toIndex] = newLayer;
      set({ layers: newLayers });

      log.debug('Created new layer, loading file:', file.name);

      // Load the same file into the new slot (use setTimeout to ensure state is updated)
      setTimeout(() => {
        useMixerStore.getState().setLayerSource(newLayer.id, file);
      }, 0);
    },

    findNextFreeSlot: (afterIndex = -1) => {
      const { layers, gridColumns, gridRows } = get();
      const gridSize = gridColumns * gridRows;

      // Search from afterIndex+1 to end
      for (let i = afterIndex + 1; i < gridSize; i++) {
        if (!layers[i]?.source) return i;
      }
      // Wrap around and search from 0 to afterIndex
      for (let i = 0; i <= afterIndex; i++) {
        if (!layers[i]?.source) return i;
      }
      return null; // No free slot
    },

    moveGroup: (groupSlots: number[], targetIndex: number) => {
      const { layers, slotGroups, gridColumns, gridRows } = get();
      const gridSize = gridColumns * gridRows;

      // Sort group slots to maintain order (left to right)
      const sortedGroup = [...groupSlots].sort((a, b) => a - b);
      const groupSize = sortedGroup.length;

      // Calculate new positions based on target
      const targetCol = targetIndex % gridColumns;
      const targetRow = Math.floor(targetIndex / gridColumns);

      // Check if there's room for the group at the target position
      const newPositions: number[] = [];
      for (let i = 0; i < groupSize; i++) {
        const newCol = targetCol + i;
        if (newCol >= gridColumns) {
          log.debug('moveGroup: Not enough room at target');
          return; // Not enough room in this row
        }
        const newIndex = targetRow * gridColumns + newCol;
        if (newIndex >= gridSize) return; // Out of bounds

        // Check if position is free (or part of the moving group)
        if (layers[newIndex]?.source && !groupSlots.includes(newIndex)) {
          log.debug('moveGroup: Target position occupied');
          return; // Position is occupied by something else
        }
        newPositions.push(newIndex);
      }

      log.debug('moveGroup:', sortedGroup, '->', newPositions);

      // Move the layers
      const newLayers = [...layers];

      // First, collect the layers being moved
      const movingLayers = sortedGroup.map(idx => newLayers[idx]);

      // Clear old positions
      sortedGroup.forEach(idx => {
        newLayers[idx] = undefined as any;
      });

      // Place in new positions
      newPositions.forEach((newIdx, i) => {
        const layer = movingLayers[i];
        if (layer) {
          newLayers[newIdx] = { ...layer, name: `Slot ${newIdx + 1}` };
        }
      });

      // Update the group to use new positions
      const newSlotGroups = slotGroups.map(group => {
        if (group.some(idx => groupSlots.includes(idx))) {
          // This is the group being moved - update positions
          return newPositions;
        }
        return group;
      });

      set({ layers: newLayers, slotGroups: newSlotGroups });
    },
  }))
);

// Helper function to create video element (fallback when WebCodecs unavailable)
async function createVideoElement(
  file: File,
  layerId: string,
  get: () => MixerState,
  set: (state: Partial<MixerState> | ((state: MixerState) => Partial<MixerState>)) => void
): Promise<void> {
  const video = document.createElement('video');
  video.src = URL.createObjectURL(file);
  video.loop = true;
  video.muted = false; // We'll control audio through the audio manager
  video.playsInline = true;
  video.crossOrigin = 'anonymous';
  video.preload = 'auto';

  // Initialize audio manager if needed
  await audioManager.init();

  // Wait for canplaythrough to ensure video is ready for playback
  video.addEventListener('canplaythrough', async () => {
    log.debug(`Video canplaythrough: ${file.name}, readyState=${video.readyState}`);

    // Connect to audio manager for volume and EQ control
    audioManager.connectMediaElement(video);
    // Apply current master volume
    audioManager.setMasterVolume(get().masterVolume);

    // Get fresh state to avoid stale closure
    const currentLayers = get().layers;
    set({
      layers: currentLayers.map((l) =>
        l?.id === layerId
          ? {
              ...l,
              source: {
                type: 'video',
                file,
                videoElement: video,
              },
            }
          : l
      ),
    });

    // Play with error handling
    video.play().then(() => {
      log.debug(`Video playing: ${file.name}`);
    }).catch((err) => {
      log.error(`Video play failed: ${file.name}`, err);
    });
  }, { once: true });

  video.addEventListener('error', (e) => {
    log.error(`Video error loading: ${file.name}`, e);
  });

  // Trigger load
  video.load();
}

function getDefaultEffectParams(type: string): Record<string, number | boolean | string> {
  switch (type) {
    case 'hue-shift':
      return { shift: 0 };
    case 'saturation':
      return { amount: 1 };
    case 'brightness':
      return { amount: 0 };
    case 'contrast':
      return { amount: 1 };
    case 'blur':
      return { radius: 0 };
    case 'pixelate':
      return { size: 8 };
    case 'kaleidoscope':
      return { segments: 6, rotation: 0 };
    case 'mirror':
      return { horizontal: true, vertical: false };
    case 'invert':
      return {};
    case 'rgb-split':
      return { amount: 0.01, angle: 0 };
    case 'levels':
      return { inputBlack: 0, inputWhite: 1, gamma: 1, outputBlack: 0, outputWhite: 1 };
    // Audio effects
    case 'audio-eq':
      return {
        band31: 0, band62: 0, band125: 0, band250: 0, band500: 0,
        band1k: 0, band2k: 0, band4k: 0, band8k: 0, band16k: 0
      };
    case 'audio-volume':
      return { volume: 1 };
    default:
      return {};
  }
}
