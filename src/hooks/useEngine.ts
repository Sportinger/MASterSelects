// React hook for WebGPU engine integration - Optimized

import { useEffect, useRef, useCallback } from 'react';
import { engine } from '../engine/WebGPUEngine';
import { useMixerStore } from '../stores/mixerStore';
import { useTimelineStore } from '../stores/timeline';
import { useSettingsStore } from '../stores/settingsStore';
import type { ClipMask, MaskVertex } from '../types';
import { generateMaskTexture } from '../utils/maskRenderer';
import { layerBuilder, playheadState } from '../services/layerBuilder';

// Create a stable hash of mask properties (including feather since blur is CPU-side now)
// This is faster than JSON.stringify for comparison
function getMaskShapeHash(masks: ClipMask[]): string {
  return masks.map(m =>
    `${m.vertices.map((v: MaskVertex) => `${v.x.toFixed(2)},${v.y.toFixed(2)}`).join(';')}|` +
    `${m.position.x.toFixed(2)},${m.position.y.toFixed(2)}|` +
    `${m.opacity.toFixed(2)}|${m.mode}|${m.closed}|${(m.feather || 0).toFixed(1)}`
  ).join('||');
}

export function useEngine() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const isEngineReady = useMixerStore((state) => state.isEngineReady);
  const isPlaying = useMixerStore((state) => state.isPlaying);
  const initRef = useRef(false);

  // Initialize engine - only once
  useEffect(() => {
    if (initRef.current) return;
    initRef.current = true;

    async function init() {
      const success = await engine.initialize();
      useMixerStore.getState().setEngineReady(success);
      if (success) {
        useMixerStore.getState().setPlaying(true);
      }
    }

    init();

    return () => {
      // Don't destroy on unmount - singleton should persist
    };
  }, []);

  // Set up canvas
  useEffect(() => {
    if (isEngineReady && canvasRef.current) {
      engine.setPreviewCanvas(canvasRef.current);
    }
  }, [isEngineReady]);

  // Update engine resolution when outputResolution or previewQuality changes
  useEffect(() => {
    if (!isEngineReady) return;

    const updateResolution = () => {
      const { outputResolution } = useMixerStore.getState();
      const { previewQuality } = useSettingsStore.getState();

      // Apply preview quality scaling to base resolution
      const scaledWidth = Math.round(outputResolution.width * previewQuality);
      const scaledHeight = Math.round(outputResolution.height * previewQuality);

      engine.setResolution(scaledWidth, scaledHeight);
      console.log(`[Engine] Resolution set to ${scaledWidth}×${scaledHeight} (${previewQuality * 100}% of ${outputResolution.width}×${outputResolution.height})`);
    };

    // Initial update
    updateResolution();

    // Subscribe to outputResolution changes
    const unsubscribeMixer = useMixerStore.subscribe(
      (state) => state.outputResolution,
      () => updateResolution()
    );

    // Subscribe to previewQuality changes
    const unsubscribeSettings = useSettingsStore.subscribe(
      (state) => state.previewQuality,
      () => updateResolution()
    );

    // Subscribe to transparency grid setting
    const updateTransparencyGrid = () => {
      const { showTransparencyGrid } = useSettingsStore.getState();
      engine.setShowTransparencyGrid(showTransparencyGrid);
    };
    updateTransparencyGrid(); // Initial update
    const unsubscribeTransparency = useSettingsStore.subscribe(
      (state) => state.showTransparencyGrid,
      () => updateTransparencyGrid()
    );

    return () => {
      unsubscribeMixer();
      unsubscribeSettings();
      unsubscribeTransparency();
    };
  }, [isEngineReady]);

  // Track mask changes and update engine mask textures
  const maskVersionRef = useRef<Map<string, string>>(new Map());

  // Helper function to process a single clip's mask
  const processClipMask = useCallback((clip: { id: string; masks?: import('../types').ClipMask[] }, engineDimensions: { width: number; height: number }) => {
    if (clip.masks && clip.masks.length > 0) {
      // Create version string - includes feather since blur is applied on CPU
      const maskVersion = `${getMaskShapeHash(clip.masks)}_${engineDimensions.width}x${engineDimensions.height}`;
      const cacheKey = clip.id;
      const prevVersion = maskVersionRef.current.get(cacheKey);

      // Regenerate texture if mask properties changed (shape, feather, etc.)
      if (maskVersion !== prevVersion) {
        maskVersionRef.current.set(cacheKey, maskVersion);

        // Generate mask texture at engine render resolution (blur applied on CPU)
        const maskImageData = generateMaskTexture(
          clip.masks,
          engineDimensions.width,
          engineDimensions.height
        );

        if (maskImageData) {
          console.log(`[Mask] Generated mask texture for clip ${clip.id}: ${engineDimensions.width}x${engineDimensions.height}, masks: ${clip.masks.length}`);
          engine.updateMaskTexture(clip.id, maskImageData);
        } else {
          console.warn(`[Mask] Failed to generate mask texture for clip ${clip.id}`);
        }
      }
    } else if (clip.id) {
      // Clip exists but no masks, clear the mask texture
      const cacheKey = clip.id;
      if (maskVersionRef.current.has(cacheKey)) {
        maskVersionRef.current.delete(cacheKey);
        engine.removeMaskTexture(clip.id);
      }
    }
  }, []);

  // Helper function to update mask textures - extracted to avoid duplication
  const updateMaskTextures = useCallback(() => {
    const { clips, playheadPosition, tracks } = useTimelineStore.getState();
    const layers = useMixerStore.getState().layers;

    // Get engine output dimensions (the actual render resolution)
    const engineDimensions = engine.getOutputDimensions();

    // Find clips at current playhead position
    const videoTracks = tracks.filter(t => t.type === 'video');
    const clipsAtTime = clips.filter(c =>
      playheadPosition >= c.startTime && playheadPosition < c.startTime + c.duration
    );

    // For each layer, find the corresponding clip
    for (let layerIndex = 0; layerIndex < layers.length; layerIndex++) {
      const layer = layers[layerIndex];
      const track = videoTracks[layerIndex];

      if (!track || !layer) continue;

      // Find clip for this track at current time
      const clip = clipsAtTime.find(c => c.trackId === track.id);

      if (clip) {
        // Process main clip's mask
        processClipMask(clip, engineDimensions);

        // Process nested clips' masks if this is a nested composition
        if (clip.nestedClips && clip.nestedClips.length > 0) {
          const clipTime = playheadPosition - clip.startTime;
          for (const nestedClip of clip.nestedClips) {
            // Check if nested clip is active at current time within the nested comp
            if (clipTime >= nestedClip.startTime && clipTime < nestedClip.startTime + nestedClip.duration) {
              processClipMask(nestedClip, engineDimensions);
            }
          }
        }
      }
    }
  }, [processClipMask]);

  useEffect(() => {
    if (!isEngineReady) return;

    // Subscribe to clips changes (mask shape updates)
    // This runs when clips array changes (including mask modifications)
    const unsubscribeClips = useTimelineStore.subscribe(
      (state) => state.clips,
      () => updateMaskTextures()
    );

    // NOTE: Removed playheadPosition subscription - it was causing updateMaskTextures()
    // to run every frame during playback (~60x/sec), causing frame drops.
    // Mask textures are now updated in the render loop only when needed.

    // Subscribe to tracks changes separately
    // This runs when track structure changes (rare)
    const unsubscribeTracks = useTimelineStore.subscribe(
      (state) => state.tracks,
      () => updateMaskTextures()
    );

    return () => {
      unsubscribeClips();
      unsubscribeTracks();
    };
  }, [isEngineReady, updateMaskTextures]);

  // Render loop - optimized with direct layer building (bypasses React state)
  useEffect(() => {
    if (!isEngineReady) return;

    // Stats update throttle
    let lastStatsUpdate = 0;

    const renderFrame = () => {
      try {
        // Use high-frequency playhead position during playback
        const currentPlayhead = playheadState.isUsingInternalPosition
          ? playheadState.position
          : useTimelineStore.getState().playheadPosition;

        // Always try to use cached frame first (works even during RAM preview rendering)
        if (engine.renderCachedFrame(currentPlayhead)) {
          return;
        }

        // Skip live rendering during RAM Preview generation
        if (useTimelineStore.getState().isRamPreviewing) {
          return;
        }

        // Build layers directly from stores (single source of truth)
        const layers = layerBuilder.buildLayersFromStore();

        // Sync video and audio elements
        layerBuilder.syncVideoElements();
        layerBuilder.syncAudioElements();

        // Render layers (layerBuilder already handles mask properties)
        engine.render(layers);

        // Throttle stats updates (every 100ms)
        const now = performance.now();
        if (now - lastStatsUpdate > 100) {
          useMixerStore.getState().setEngineStats(engine.getStats());
          lastStatsUpdate = now;
        }
      } catch (e) {
        console.error('Render error:', e);
      }
    };

    if (isPlaying) {
      engine.start(renderFrame);
    } else {
      engine.stop();
    }

    return () => {
      engine.stop();
    };
  }, [isEngineReady, isPlaying]);

  const createOutputWindow = useCallback((name: string) => {
    const id = `output_${Date.now()}`;
    return engine.createOutputWindow(id, name);
  }, []);

  const closeOutputWindow = useCallback((id: string) => {
    engine.closeOutputWindow(id);
  }, []);

  const registerPreviewCanvas = useCallback((id: string, canvas: HTMLCanvasElement) => {
    engine.registerPreviewCanvas(id, canvas);
  }, []);

  const unregisterPreviewCanvas = useCallback((id: string) => {
    engine.unregisterPreviewCanvas(id);
  }, []);

  // Independent canvas registration - NOT rendered by main loop
  const registerIndependentPreviewCanvas = useCallback((id: string, canvas: HTMLCanvasElement) => {
    engine.registerIndependentPreviewCanvas(id, canvas);
  }, []);

  const unregisterIndependentPreviewCanvas = useCallback((id: string) => {
    engine.unregisterIndependentPreviewCanvas(id);
  }, []);

  const renderToPreviewCanvas = useCallback((canvasId: string, layers: import('../types').Layer[]) => {
    engine.renderToPreviewCanvas(canvasId, layers);
  }, []);

  return {
    canvasRef,
    isEngineReady,
    isPlaying,
    createOutputWindow,
    closeOutputWindow,
    registerPreviewCanvas,
    unregisterPreviewCanvas,
    registerIndependentPreviewCanvas,
    unregisterIndependentPreviewCanvas,
    renderToPreviewCanvas,
  };
}
