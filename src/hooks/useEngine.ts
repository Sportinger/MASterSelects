// React hook for WebGPU engine integration - Optimized

import { useEffect, useRef, useCallback } from 'react';
import { engine } from '../engine/WebGPUEngine';
import { useMixerStore } from '../stores/mixerStore';
import { useTimelineStore } from '../stores/timeline';
import { useMediaStore } from '../stores/mediaStore';
import { useSettingsStore } from '../stores/settingsStore';
import type { ClipMask, MaskVertex } from '../types';
import { generateMaskTexture } from '../utils/maskRenderer';
import { layerBuilder, playheadState } from '../services/layerBuilder';
import { Logger } from '../services/logger';

const log = Logger.create('Engine');

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
        // Get and store GPU info
        const gpuInfo = engine.getGPUInfo();
        useMixerStore.getState().setGpuInfo(gpuInfo);
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
      log.info(`Resolution set to ${scaledWidth}×${scaledHeight} (${previewQuality * 100}% of ${outputResolution.width}×${outputResolution.height})`);
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
          log.debug(`Generated mask texture for clip ${clip.id}: ${engineDimensions.width}x${engineDimensions.height}, masks: ${clip.masks.length}`);
          engine.updateMaskTexture(clip.id, maskImageData);
        } else {
          log.warn(`Failed to generate mask texture for clip ${clip.id}`);
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

  // Throttle mask texture updates during drag (100ms = 10fps for GPU texture)
  const lastMaskTextureUpdate = useRef(0);
  const MASK_TEXTURE_THROTTLE_MS = 100; // Update GPU texture max 10fps during drag

  // Helper function to update mask textures - extracted to avoid duplication
  const updateMaskTextures = useCallback(() => {
    const { clips, playheadPosition, tracks, maskDragging } = useTimelineStore.getState();

    // Throttle texture regeneration during drag (expensive CPU operation)
    if (maskDragging) {
      const now = performance.now();
      if (now - lastMaskTextureUpdate.current < MASK_TEXTURE_THROTTLE_MS) {
        return; // Skip this update, too soon
      }
      lastMaskTextureUpdate.current = now;
    }
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

    // Subscribe to composition changes
    // When switching compositions, we need to regenerate mask textures for the new comp
    // This handles nested comp masks showing correctly when returning to parent comp
    const unsubscribeComp = useMediaStore.subscribe(
      (state) => state.activeCompositionId,
      () => {
        // Clear mask version cache to force regeneration for the new composition
        maskVersionRef.current.clear();
        updateMaskTextures();
      }
    );

    // Subscribe to maskDragging changes
    // When drag ends (maskDragging: true -> false), regenerate mask textures
    let wasDragging = false;
    const unsubscribeDragging = useTimelineStore.subscribe(
      (state) => state.maskDragging,
      (maskDragging) => {
        if (wasDragging && !maskDragging) {
          // Drag just ended - force texture regeneration by clearing version cache
          maskVersionRef.current.clear();
          updateMaskTextures();
        }
        wasDragging = maskDragging;
      }
    );

    return () => {
      unsubscribeClips();
      unsubscribeTracks();
      unsubscribeComp();
      unsubscribeDragging();
    };
  }, [isEngineReady, updateMaskTextures]);

  // Render loop - optimized with direct layer building (bypasses React state)
  useEffect(() => {
    if (!isEngineReady) return;

    // Stats update throttle
    let lastStatsUpdate = 0;
    let lastPlayhead = -1;

    const renderFrame = () => {
      try {
        // Always update stats (even when idle) so UI shows correct status
        const now = performance.now();
        if (now - lastStatsUpdate > 100) {
          useMixerStore.getState().setEngineStats(engine.getStats());
          lastStatsUpdate = now;
        }

        // Skip actual rendering if engine is idle
        if (engine.getIsIdle()) {
          return;
        }

        // Use high-frequency playhead position during playback
        const currentPlayhead = playheadState.isUsingInternalPosition
          ? playheadState.position
          : useTimelineStore.getState().playheadPosition;

        // Track playhead changes for idle detection
        // During playback, playhead constantly changes -> keeps engine active
        // When stopped/scrubbing, only renders when playhead actually moves
        if (currentPlayhead !== lastPlayhead) {
          lastPlayhead = currentPlayhead;
          engine.requestRender();
        }

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

        // Cache rendered frame for instant scrubbing (like Premiere's playback caching)
        // Only cache if RAM preview is enabled and we're playing (not generating RAM preview)
        const { ramPreviewEnabled, addCachedFrame } = useTimelineStore.getState();
        if (ramPreviewEnabled && isPlaying) {
          engine.cacheCompositeFrame(currentPlayhead).then(() => {
            addCachedFrame(currentPlayhead);
          });
        }

        // Cache active comp output for parent preview texture sharing
        // This allows parent compositions to show the active comp without video conflicts
        const activeCompId = useMediaStore.getState().activeCompositionId;
        if (activeCompId) {
          engine.cacheActiveCompOutput(activeCompId);
        }
      } catch (e) {
        log.error('Render error', e);
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

  // Subscribe to state changes that require re-render (wake from idle)
  useEffect(() => {
    if (!isEngineReady) return;

    // Playhead position changes (scrubbing, playback)
    const unsubPlayhead = useTimelineStore.subscribe(
      (state) => state.playheadPosition,
      () => engine.requestRender()
    );

    // Clips changes (content, transforms, effects, etc.)
    const unsubClips = useTimelineStore.subscribe(
      (state) => state.clips,
      () => engine.requestRender()
    );

    // Track changes
    const unsubTracks = useTimelineStore.subscribe(
      (state) => state.tracks,
      () => engine.requestRender()
    );

    // Layer changes in mixer store
    const unsubLayers = useMixerStore.subscribe(
      (state) => state.layers,
      () => engine.requestRender()
    );

    // Output resolution changes
    const unsubResolution = useMixerStore.subscribe(
      (state) => state.outputResolution,
      () => engine.requestRender()
    );

    // Settings changes (transparency grid, preview quality)
    const unsubSettings = useSettingsStore.subscribe(
      (state) => [state.showTransparencyGrid, state.previewQuality],
      () => engine.requestRender(),
      { equalityFn: (a, b) => a[0] === b[0] && a[1] === b[1] }
    );

    // Active composition changes
    const unsubActiveComp = useMediaStore.subscribe(
      (state) => state.activeCompositionId,
      () => engine.requestRender()
    );

    return () => {
      unsubPlayhead();
      unsubClips();
      unsubTracks();
      unsubLayers();
      unsubResolution();
      unsubSettings();
      unsubActiveComp();
    };
  }, [isEngineReady]);

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
  const registerIndependentPreviewCanvas = useCallback((id: string, canvas: HTMLCanvasElement, compositionId?: string) => {
    engine.registerIndependentPreviewCanvas(id, canvas, compositionId);
  }, []);

  const unregisterIndependentPreviewCanvas = useCallback((id: string) => {
    engine.unregisterIndependentPreviewCanvas(id);
  }, []);

  // Update which composition an independent canvas is showing
  const setIndependentCanvasComposition = useCallback((canvasId: string, compositionId: string) => {
    engine.setIndependentCanvasComposition(canvasId, compositionId);
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
    setIndependentCanvasComposition,
    renderToPreviewCanvas,
  };
}
