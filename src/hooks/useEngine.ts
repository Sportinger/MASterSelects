// React hook for WebGPU engine integration - Optimized

import { useEffect, useRef, useCallback } from 'react';
import { engine } from '../engine/WebGPUEngine';
import { useMixerStore } from '../stores/mixerStore';
import { useTimelineStore } from '../stores/timelineStore';
import { generateMaskTexture } from '../utils/maskRenderer';

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

  // Track mask changes and update engine mask textures
  const maskVersionRef = useRef<Map<string, string>>(new Map());

  useEffect(() => {
    if (!isEngineReady) return;

    // Subscribe to both clips and playhead position changes
    const unsubscribe = useTimelineStore.subscribe(
      (state) => ({ clips: state.clips, playheadPosition: state.playheadPosition, tracks: state.tracks }),
      ({ clips, playheadPosition, tracks }) => {
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

          if (!track) continue;

          // Find clip for this track at current time
          const clip = clipsAtTime.find(c => c.trackId === track.id);

          if (clip?.masks && clip.masks.length > 0 && layer.source) {
            // Extract mask properties for GPU processing (feather/invert handled in shader)
            const maxFeather = Math.max(...clip.masks.map(m => m.feather));
            const hasInverted = clip.masks.some(m => m.inverted);

            // Update layer with mask GPU properties (these are cheap uniform updates)
            useMixerStore.getState().updateLayerMaskProps(layer.id, maxFeather, hasInverted);

            // Create version string EXCLUDING feather/invert (they're GPU uniforms now)
            // Only include shape-affecting properties: vertices, position, opacity, mode, closed
            const shapeVersion = clip.masks.map(m => ({
              vertices: m.vertices,
              position: m.position,
              opacity: m.opacity,
              mode: m.mode,
              closed: m.closed
            }));
            const maskVersion = `${JSON.stringify(shapeVersion)}_${engineDimensions.width}x${engineDimensions.height}`;
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
          } else if (clip) {
            // Clip exists but no masks, clear the mask texture for this layer
            const cacheKey = `${clip.id}_${layer.id}`;
            if (maskVersionRef.current.has(cacheKey)) {
              maskVersionRef.current.delete(cacheKey);
              engine.removeMaskTexture(layer.id);
            }
          }
        }
      }
    );

    return () => unsubscribe();
  }, [isEngineReady]);

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

        // Create a stable copy for this frame (shallow copy is sufficient
        // since we only read from it, never modify)
        const frameLayers = layersSnapshot.slice();

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

  return {
    canvasRef,
    isEngineReady,
    isPlaying,
    createOutputWindow,
    closeOutputWindow,
  };
}
