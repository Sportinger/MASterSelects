// React hook for WebGPU engine integration - Optimized

import { useEffect, useRef, useCallback } from 'react';
import { engine } from '../engine/WebGPUEngine';
import { useMixerStore } from '../stores/mixerStore';
import { useTimelineStore } from '../stores/timeline';
import { useSettingsStore } from '../stores/settingsStore';
import type { ClipMask, MaskVertex } from '../types';
import { generateMaskTexture } from '../utils/maskRenderer';

// Create a stable hash of mask shapes only (excludes feather/invert which are GPU uniforms)
// This is faster than JSON.stringify for shape comparison
function getMaskShapeHash(masks: ClipMask[]): string {
  // Only include shape-affecting properties for hash
  return masks.map(m =>
    `${m.vertices.map((v: MaskVertex) => `${v.x.toFixed(2)},${v.y.toFixed(2)}`).join(';')}|` +
    `${m.position.x.toFixed(2)},${m.position.y.toFixed(2)}|` +
    `${m.opacity.toFixed(2)}|${m.mode}|${m.closed}`
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

    return () => {
      unsubscribeMixer();
      unsubscribeSettings();
    };
  }, [isEngineReady]);

  // Track mask changes and update engine mask textures
  const maskVersionRef = useRef<Map<string, string>>(new Map());

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

      if (clip?.masks && clip.masks.length > 0 && layer.source) {
        // Extract mask properties for GPU processing (feather/quality/invert handled in shader)
        const maxFeather = Math.max(...clip.masks.map(m => m.feather));
        const maxQuality = Math.max(...clip.masks.map(m => m.featherQuality ?? 1));
        const hasInverted = clip.masks.some(m => m.inverted);

        // Update layer with mask GPU properties (these are cheap uniform updates)
        useMixerStore.getState().updateLayerMaskProps(layer.id, maxFeather, maxQuality, hasInverted);

        // Create version string using fast hash (EXCLUDING feather/invert - they're GPU uniforms now)
        // Only include shape-affecting properties: vertices, position, opacity, mode, closed
        const maskVersion = `${getMaskShapeHash(clip.masks)}_${engineDimensions.width}x${engineDimensions.height}`;
        const cacheKey = `${clip.id}_${layer.id}`;
        const prevVersion = maskVersionRef.current.get(cacheKey);

        // Only regenerate texture if mask SHAPE changed (not feather/invert)
        if (maskVersion !== prevVersion) {
          maskVersionRef.current.set(cacheKey, maskVersion);

          // Generate mask texture at engine render resolution (no blur/invert - done in GPU)
          const maskImageData = generateMaskTexture(
            clip.masks,
            engineDimensions.width,
            engineDimensions.height
          );

          if (maskImageData) {
            console.log(`[Mask] Generated mask texture for layer ${layer.id}: ${engineDimensions.width}x${engineDimensions.height}, masks: ${clip.masks.length}`);
            engine.updateMaskTexture(layer.id, maskImageData);
          } else {
            console.warn(`[Mask] Failed to generate mask texture for layer ${layer.id}`);
          }
        }
      } else if (clip && clip.id) {
        // Clip exists but no masks, clear the mask texture for this layer
        const cacheKey = `${clip.id}_${layer.id}`;
        if (maskVersionRef.current.has(cacheKey)) {
          maskVersionRef.current.delete(cacheKey);
          engine.removeMaskTexture(layer.id);
        }
      }
    }
  }, []);

  useEffect(() => {
    if (!isEngineReady) return;

    // Subscribe to clips changes (mask shape updates)
    // This runs when clips array changes (including mask modifications)
    const unsubscribeClips = useTimelineStore.subscribe(
      (state) => state.clips,
      () => updateMaskTextures()
    );

    // Subscribe to playhead position changes separately
    // This runs when scrubbing/playing to update which masks are visible
    const unsubscribePlayhead = useTimelineStore.subscribe(
      (state) => state.playheadPosition,
      () => updateMaskTextures()
    );

    // Subscribe to tracks changes separately
    // This runs when track structure changes (rare)
    const unsubscribeTracks = useTimelineStore.subscribe(
      (state) => state.tracks,
      () => updateMaskTextures()
    );

    return () => {
      unsubscribeClips();
      unsubscribePlayhead();
      unsubscribeTracks();
    };
  }, [isEngineReady, updateMaskTextures]);

  // Render loop - optimized with layer snapshotting to prevent flickering
  useEffect(() => {
    if (!isEngineReady) return;

    // Stats update throttle
    let lastStatsUpdate = 0;

    const renderFrame = () => {
      try {
        // Always try to use cached frame first (works even during RAM preview rendering)
        const { playheadPosition, isRamPreviewing } = useTimelineStore.getState();
        if (engine.renderCachedFrame(playheadPosition)) {
          // Successfully rendered cached frame, skip live render
          return;
        }

        // CRITICAL: Skip live rendering during RAM Preview generation
        // Live rendering would seek videos and interfere with RAM Preview's seeking,
        // causing glitchy/wrong frames to be cached
        if (isRamPreviewing) {
          return;
        }

        // IMPORTANT: Snapshot layers array at start of frame to ensure consistency
        // This prevents flickering from reading partially updated state
        const layersSnapshot = useMixerStore.getState().layers;

        // Get clips and tracks to extract mask properties at render time
        // This ensures mask properties are ALWAYS in sync with the current frame
        const { clips, tracks, playheadPosition: currentPlayhead } = useTimelineStore.getState();
        const videoTracks = tracks.filter(t => t.type === 'video');
        const clipsAtTime = clips.filter(c =>
          currentPlayhead >= c.startTime && currentPlayhead < c.startTime + c.duration
        );

        // Get engine output dimensions for mask generation
        const engineDimensions = engine.getOutputDimensions();

        // Create frame layers with FRESH mask properties from clips (not stale store state)
        const frameLayers = layersSnapshot.map((layer, layerIndex) => {
          if (!layer) return layer;

          const track = videoTracks[layerIndex];
          if (!track) return layer;

          const clip = clipsAtTime.find(c => c.trackId === track.id);

          // Extract mask properties directly from clip at render time
          if (clip?.masks && clip.masks.length > 0) {
            // CRITICAL: Ensure mask texture exists BEFORE rendering this frame
            // This prevents the race condition where first frame renders without mask
            if (!engine.hasMaskTexture(layer.id)) {
              // Generate mask texture synchronously before rendering
              const maskImageData = generateMaskTexture(
                clip.masks,
                engineDimensions.width,
                engineDimensions.height
              );
              if (maskImageData) {
                engine.updateMaskTexture(layer.id, maskImageData);
              }
            }

            const maxFeather = Math.max(...clip.masks.map(m => m.feather));
            const maxQuality = Math.max(...clip.masks.map(m => m.featherQuality ?? 50));
            const hasInverted = clip.masks.some(m => m.inverted);

            // Return layer with fresh mask properties
            return {
              ...layer,
              maskFeather: maxFeather,
              maskFeatherQuality: maxQuality,
              maskInvert: hasInverted
            };
          }

          return layer;
        });

        // Render with snapshotted layers (engine handles empty layers by clearing to black)
        engine.render(frameLayers);

        // Throttle stats updates to reduce React re-renders (every 100ms for responsive display)
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
